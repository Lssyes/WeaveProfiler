#!/usr/bin/env python3
"""capture_nsys.py (E17) — run the instrumented launch under nsys with a bounded
cudaProfilerApi capture window and 200 kHz device-wide PM sampling, then export sqlite.

FULLY IMPLEMENTED generic orchestration. It builds the nsys command, runs the
project's launch, exports the .nsys-rep to sqlite, records a GPU inventory and
writes capture_manifest.json (device / metrics_set / frequency_hz / gpu_check /
artifact paths). extract_gpu_metrics.py reads that manifest.

The ONLY project-specific input is the launch command that actually invokes YOUR
instrumented kernel between cudaProfilerStart() and cudaProfilerStop(). That is a
PROJECT HOOK provided via --launch-cmd or the contract build_and_launch section;
there is no safe default. Everything else (sampling frequency, capture range,
export, manifest) is generic.

CLI (matches the unified stage command):
  python3 -B capture_nsys.py --contract <lock> --out-dir hw
  python3 -B capture_nsys.py --launch-cmd "python -m your.entry --profile" --out-dir hw
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

FREQUENCY_HZ = 200000
METRICS_SET_DEFAULT = "gpu-pm-sampling"  # overridable via contract capture.hardware.metrics_set


def gpu_inventory() -> Dict[str, Any]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"available": False, "reason": "nvidia-smi not found", "gpus": []}
    try:
        out = subprocess.run([smi, "--query-gpu=index,name,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": str(error), "gpus": []}
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            gpus.append({"index": int(parts[0]), "name": parts[1],
                         "uuid": parts[2] if len(parts) > 2 else None})
    return {"available": out.returncode == 0, "returncode": out.returncode, "gpus": gpus}


def resolve_launch_cmd(args: argparse.Namespace, contract: Optional[Dict[str, Any]]) -> List[str]:
    if args.launch_cmd:
        return shlex.split(args.launch_cmd)
    if contract is not None:
        launch = contract.get("build_and_launch") or {}
        cmd = launch.get("launch_command") or launch.get("command")
        if isinstance(cmd, list) and cmd:
            return [str(c) for c in cmd]
        if isinstance(cmd, str) and cmd.strip():
            return shlex.split(cmd)
    # ===== PROJECT HOOK: instrumented launch command (cudaProfilerStart..Stop) =====
    raise NotImplementedError(
        "PROJECT HOOK: pass --launch-cmd \"...\" (or set build_and_launch.launch_command in the "
        "contract). It must launch YOUR instrumented run and wrap the target kernel with "
        "cudaProfilerStart()/cudaProfilerStop() (torch.cuda.cudart().cudaProfilerStart/Stop). "
        "nsys --capture-range=cudaProfilerApi bounds PM sampling to exactly that window.")


def resolve_capture_config(args: argparse.Namespace, contract: Optional[Dict[str, Any]]):
    frequency = args.frequency_hz
    metrics_set = args.metrics_set or METRICS_SET_DEFAULT
    device = args.device
    kernel_needle = args.kernel_name_contains
    if contract is not None:
        hardware = ((contract.get("capture") or {}).get("hardware") or {})
        frequency = int(hardware.get("frequency_hz") or frequency)
        metrics_set = hardware.get("metrics_set") or metrics_set
        kernel_needle = kernel_needle or hardware.get("kernel_name_contains")
        gpu = (contract.get("environment") or {}).get("gpu") or {}
        if device is None and gpu.get("index") is not None:
            device = int(gpu["index"])
    return frequency, metrics_set, device, kernel_needle


def build_nsys_cmd(nsys: str, rep_path: Path, frequency: int, metrics_set: str,
                   launch_cmd: List[str]) -> List[str]:
    """Build the nsys command line.

    NOTE on `metrics_set`: it is deliberately NOT passed to nsys. The flag name and the
    accepted set names differ across nsys builds, and --gpu-metrics-devices=all already
    selects the device-wide PM sample stream this pipeline reads. The parameter is kept
    only so the caller can record WHICH set it believes it captured; main() writes it to
    the manifest as `metrics_set` together with `metrics_set_passed_to_nsys: false`, so
    the label is never mistaken for evidence that nsys was configured with it. If your
    nsys build needs an explicit set, add the flag here AND flip that manifest field —
    do not leave the manifest claiming something the command line never said.
    """
    cmd = [nsys, "profile",
           "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
           "--gpu-metrics-devices=all",
           "--gpu-metrics-frequency={}".format(frequency),
           "--force-overwrite=true",
           "--output={}".format(rep_path.with_suffix("")),
           "--export=sqlite"]
    # nsys builds vary; keep the set that maps to a device-wide PM sample stream.
    return cmd + launch_cmd


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Capture a bounded nsys PM window and export sqlite.")
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--launch-cmd", type=str, default=None)
    parser.add_argument("--frequency-hz", type=int, default=FREQUENCY_HZ)
    parser.add_argument("--metrics-set", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--kernel-name-contains", type=str, default=None)
    parser.add_argument("--nsys", type=str, default=None, help="nsys binary (default: from PATH)")
    parser.add_argument("--dry-run", action="store_true", help="print the nsys command without running")
    args = parser.parse_args(argv)

    contract = json.loads(args.contract.read_text(encoding="utf-8")) if args.contract else None
    launch_cmd = resolve_launch_cmd(args, contract)
    frequency, metrics_set, device, kernel_needle = resolve_capture_config(args, contract)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    rep_path = out_dir / "capture.nsys-rep"
    sqlite_path = out_dir / "capture.sqlite"
    manifest_path = out_dir / "capture_manifest.json"

    nsys = args.nsys or shutil.which("nsys")
    inventory = gpu_inventory()
    inventory_gpus = inventory.get("gpus") or []
    # WHY the old `else 0` is a trap: with no inventory it picked device 0 as if that were a
    # decision, and the manifest went out with gpu_check.gpus == []. extract_gpu_metrics.py
    # derives the GPU NAME from that list, so every `scale` rule became undecidable and was
    # skipped — which is exactly how an H20 delivery lost its x4 Tensor Active correction
    # while every gate stayed green. Guessing the device is the start of that chain.
    if device is not None:
        gpu_metrics_device = device
    elif inventory_gpus:
        gpu_metrics_device = int(inventory_gpus[0]["index"])
    elif args.dry_run:
        gpu_metrics_device = None  # nothing to claim on a GPU-less host; dry-run only
    else:
        raise SystemExit(
            "[capture] cannot determine the GPU to attribute this capture to: nvidia-smi listed "
            "no GPUs ({}) and no --device was given. Defaulting to device 0 would write an "
            "unverified device index and no device NAME into capture_manifest.json, and metric "
            "`scale` rules are keyed on that name — they would all be skipped silently. Pass "
            "--device <index> explicitly, or fix nvidia-smi access in this "
            "container.".format(inventory.get("reason") or inventory.get("returncode")))
    device_name = next((g.get("name") for g in inventory_gpus
                        if g.get("index") == gpu_metrics_device), None)
    if not device_name:
        # Not fatal here (a capture without scale rules is still usable), but it must be
        # visible: extract_gpu_metrics.py hard-fails later if any scale rule needs the name.
        print("[capture] WARNING no GPU name for device {} (nvidia-smi inventory: {}). Metric "
              "`scale` rules are keyed on the device name and extract_gpu_metrics.py will "
              "refuse to apply them without it.".format(gpu_metrics_device,
                                                        inventory.get("reason") or "empty"),
              file=sys.stderr)

    if nsys is None and not args.dry_run:
        print("[capture] nsys not found on PATH; install Nsight Systems or pass --nsys", file=sys.stderr)
        return 2
    nsys_cmd = build_nsys_cmd(nsys or "nsys", rep_path, frequency, metrics_set, launch_cmd)

    manifest: Dict[str, Any] = {
        "schema": "nsys-capture-manifest/v1", "frequency_hz": frequency, "metrics_set": metrics_set,
        # metrics_set is a LABEL for what we believe was sampled; build_nsys_cmd() does not
        # pass it to nsys (flag/name spellings differ per build). Recorded explicitly so the
        # manifest is not read as evidence that nsys was configured with this set.
        "metrics_set_passed_to_nsys": False,
        "capture_range": "cudaProfilerApi", "capture_range_end": "stop",
        "gpu_metrics_device": gpu_metrics_device, "gpu_metrics_device_name": device_name,
        "kernel_name_contains": kernel_needle,
        "launch_cmd": launch_cmd, "nsys_cmd": nsys_cmd, "gpu_check": inventory,
        "artifacts": {"nsys_rep": str(rep_path), "sqlite": str(sqlite_path),
                      "stdout_log": str(logs_dir / "capture.stdout.log"),
                      "stderr_log": str(logs_dir / "capture.stderr.log")},
    }

    if args.dry_run:
        print("[capture] DRY RUN nsys cmd: {}".format(" ".join(shlex.quote(c) for c in nsys_cmd)))
        manifest["status"] = "dry-run"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return 0

    started = time.time()
    with (logs_dir / "capture.stdout.log").open("w") as out_log, \
         (logs_dir / "capture.stderr.log").open("w") as err_log:
        proc = subprocess.run(nsys_cmd, stdout=out_log, stderr=err_log)
    manifest["nsys_returncode"] = proc.returncode
    manifest["capture_wall_s"] = round(time.time() - started, 3)
    if proc.returncode != 0:
        manifest["status"] = "nsys-failed"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("[capture] nsys failed rc={}; see {}".format(proc.returncode, logs_dir), file=sys.stderr)
        return proc.returncode

    # nsys may already export sqlite via --export=sqlite; if not, do it explicitly.
    if not sqlite_path.is_file() and rep_path.is_file():
        export = subprocess.run([nsys or "nsys", "export", "--type", "sqlite",
                                 "--force-overwrite", "true", "--output", str(sqlite_path), str(rep_path)],
                                capture_output=True, text=True)
        manifest["export_returncode"] = export.returncode
        if export.returncode != 0:
            manifest["status"] = "export-failed"
            manifest["export_stderr"] = export.stderr
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print("[capture] sqlite export failed rc={}".format(export.returncode), file=sys.stderr)
            return export.returncode

    manifest["status"] = "completed"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("[capture] done rc=0 sqlite={} manifest={}".format(sqlite_path, manifest_path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
