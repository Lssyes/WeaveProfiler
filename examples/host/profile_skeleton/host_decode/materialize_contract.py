#!/usr/bin/env python3
"""materialize_contract.py (E11) — freeze a runnable contract from 5 user choices
+ a live device probe + project specifics.

Generic logic is implemented: it takes the five capture choices, probes the GPU,
resolves the selected ABI + preset, hashes every host component (so the contract
is self-describing and tamper-evident), fills sane generic defaults for
validation / alignment / artifact budgets, and writes a FROZEN contract
(lock.state=frozen, allow_in_place_edit=false).

Project specifics with no safe default (input tensor shapes/dtypes + seed, the
exact instrumented launch command, the target kernel name substring) come from
`project_specifics()`, which is a PROJECT HOOK and raises NotImplementedError
until you fill it. That keeps a half-configured contract from ever freezing.

The five user choices (contract `capture` section):
  1. --information-bundle  sw_only | sw_hw
  2. --granularity         task | phase
  3. --display-rank        which rank the perfetto trace renders
  4. --software-sm-scope   all_sms | subset:<smids>
  5. --hardware            on | off  (device-wide PM sampling at 200 kHz)

CLI:
  python3 -B materialize_contract.py --information-bundle sw_hw --granularity phase \
      --display-rank 0 --software-sm-scope all_sms --hardware on \
      --tps-path /abs/trace_processor_shell --out contract.lock.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROFILE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROFILE_ROOT / "device_record" / "abi"))
from abi_loader import load_abi, sha256_of  # noqa: E402

DEFAULT_ABI = _PROFILE_ROOT / "device_record" / "abi" / "record_abi.example.json"
PRESETS_DIR = _PROFILE_ROOT / "device_record" / "presets"
HW_FREQUENCY_HZ = 200000

# Host components hashed into the contract so a stale copy is detectable.
COMPONENT_FILES = [
    "run_profile.py",
    "host_decode/materialize_contract.py",
    "host_decode/decode_profile.py",
    "device_record/abi/abi_loader.py",
    "hw_perfetto/capture_nsys.py",
    "hw_perfetto/extract_gpu_metrics.py",
    "hw_perfetto/align_sw_hw.py",
    "hw_perfetto/emit_perfetto.py",
    "validation/validate_correctness.py",
    "validation/validate_overhead.py",
    "validation/run_offline_selftests.py",
]


def probe_gpu(device_index: int) -> Dict[str, Any]:
    """Probe the target GPU. A contract with no GPU NAME must never freeze.

    WHY every miss is fatal: the name (e.g. "NVIDIA H20") is what keys the metric `scale`
    rules in hw_perfetto/metrics/metrics_whitelist.json and what tells a reader which SKU
    the delivery describes. The old code returned name=None on every failure path, the
    contract froze with environment.gpu.name = null, and the whole chain downstream then
    "decided" that no SKU-specific correction applied — an H20 delivery understated tensor
    utilisation 4x that way with every gate green. This contract is meant to be frozen ON
    the machine that will run the capture, so a failed probe means the wrong machine.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        raise SystemExit(
            "[materialize] nvidia-smi not found, so the target GPU cannot be identified. The "
            "contract must record the GPU name: metric scale rules are keyed on it and the "
            "delivery quotes it as the SKU under test. Freeze the contract on the target GPU machine "
            "(or fix nvidia-smi visibility in this container) instead of shipping a null name.")
    try:
        out = subprocess.run([smi, "--query-gpu=index,name,uuid,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        raise SystemExit(
            "[materialize] nvidia-smi failed ({}); refusing to freeze a contract without a GPU "
            "name — see probe_gpu() for why a null name silently disables SKU-specific metric "
            "corrections.".format(error))
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0].isdigit() and int(parts[0]) == device_index:
            name = parts[1] if len(parts) > 1 else None
            if not name:
                raise SystemExit(
                    "[materialize] nvidia-smi listed device {} without a name: {!r}".format(
                        device_index, line))
            return {"index": device_index, "name": name,
                    "uuid": parts[2] if len(parts) > 2 else None,
                    "memory_total": parts[3] if len(parts) > 3 else None, "probe": "ok"}
    raise SystemExit(
        "[materialize] device {} is not listed by nvidia-smi (visible devices: {!r}). Pass "
        "--display-device with an index that exists on this host; a contract naming a device "
        "that is not there cannot describe the capture that follows.".format(
            device_index, out.stdout.strip()))


def load_preset(name: str) -> Dict[str, Any]:
    path = PRESETS_DIR / "{}.json".format(name)
    if not path.is_file():
        raise SystemExit("[materialize] preset {!r} not found under {}".format(name, PRESETS_DIR))
    return json.loads(path.read_text(encoding="utf-8"))


def component_hashes() -> List[Dict[str, Any]]:
    # WHY missing files are fatal: these hashes are the contract's tamper evidence — they
    # tie the frozen lock to the exact host scripts that produced the artifacts. Recording
    # sha256=None/present=False for an absent file froze a contract that documented its own
    # holes: a half-copied profile/ tree passed materialization, then a stage was simply
    # missing at run time with no hash to compare against.
    out = []
    missing = []
    for rel in COMPONENT_FILES:
        path = _PROFILE_ROOT / rel
        if not path.is_file():
            missing.append(rel)
            continue
        out.append({"path": rel, "sha256": sha256_of(path), "present": True})
    if missing:
        raise SystemExit(
            "[materialize] pipeline components missing under {}: {}. The contract's component "
            "hashes are its tamper evidence, so it cannot freeze around files that are not "
            "there. Copy the full profile_skeleton tree (or remove the entry from "
            "COMPONENT_FILES if your project genuinely replaced that stage — and then say so "
            "in the contract).".format(_PROFILE_ROOT, missing))
    return out


def project_specifics() -> Dict[str, Any]:
    """Return the project-specific contract fragments. PROJECT HOOK.

    Must return a dict with:
      input: {seed:int, tensors:[{name, shape:[...], dtype}]}
      build_and_launch: {grid, block, launch_command:str, kernel_name_contains:str}
    """
    # ===== PROJECT HOOK: model input tensors + instrumented launch command + kernel needle =====
    raise NotImplementedError(
        "PROJECT HOOK: fill project_specifics() with YOUR model's randn input tensors "
        "(names/shapes/dtypes + seed), the exact instrumented launch command (wrapping the "
        "target kernel in cudaProfilerStart/Stop), the launch grid/block, and the target "
        "kernel name substring. None of these have a safe default.")


def build_contract(args: argparse.Namespace) -> Dict[str, Any]:
    abi_path = Path(args.abi)
    abi = load_abi(abi_path)
    preset = load_preset(args.preset)
    gpu = probe_gpu(args.display_device)
    specifics = project_specifics()

    hardware_on = args.hardware == "on"
    scope = args.software_sm_scope
    # WHY the key names matter: gate G13 compares the DECLARED SM set here against the set
    # actually decoded from the sidecar, and it looks for capture.software_sm_scope.mode /
    # .selected_physical_smids (same shape as configs/validation/contract_lock_example.json). The
    # old {"kind": ..., "smids": ...} spelling was invisible to it, so the declaration
    # silently did not exist and the strongest scope check had nothing to compare.
    if scope == "all_sms":
        software_sm_scope = {"mode": "all_physical_sms", "selected_physical_smids": [],
                             "source_capture_was_all_sm": True}
    else:
        smids = [int(x) for x in scope.split(":", 1)[1].split(",") if x.strip()]
        if not smids:
            raise SystemExit(
                "[materialize] --software-sm-scope subset:<smids> listed no SM ids. A subset "
                "capture must name the physical SMs it instruments; an empty list is read as "
                "'whole device' downstream.")
        software_sm_scope = {"mode": "subset", "selected_physical_smids": smids,
                             "source_capture_was_all_sm": False}

    # WHY these cannot be null/absent: they size the device record buffer. A frozen contract
    # with num_sms=null made run_profile.py fall back to `or 1`, i.e. one CTA's worth of
    # slots, and every other CTA's records went nowhere while the run still "succeeded".
    if not args.num_sms:
        raise SystemExit(
            "[materialize] --num-sms is required: it is the launch grid (number of CTAs; for a "
            "persistent kernel the SM count you launch on, e.g. 78 on H20). A null num_sms in "
            "the frozen contract makes the capture size its buffer for a single CTA.")
    if not args.max_records:
        raise SystemExit(
            "[materialize] --max-records is required: 2 records (begin+end) per event times the "
            "maximum events one observer track writes per launch, plus headroom. Verify the "
            "choice by checking that decode_profile.py reports truncated_tracks == [].")

    contract: Dict[str, Any] = {
        "schema": "fast-instrument.contract-lock/v1",
        "lock": {"state": "frozen", "allow_in_place_edit": False,
                 "note": "regenerate via materialize_contract.py; do not hand-edit a frozen lock"},
        "profile_selection": {"abi": str(abi_path), "abi_schema": abi.get("schema"),
                              "abi_source_sha256": abi.get("source_sha256"), "preset": args.preset,
                              "preset_defines": preset.get("compile_defines")},
        "environment": {"gpu": gpu},
        "run": {"world_size": args.world_size},
        "capture": {
            "information_bundle": args.information_bundle,
            "granularity": args.granularity,
            "display_rank": args.display_rank,
            "software_sm_scope": software_sm_scope,
            "hardware": {"enabled": hardware_on, "frequency_hz": HW_FREQUENCY_HZ,
                         "capture_range": "cudaProfilerApi", "metrics_set": args.metrics_set,
                         "kernel_name_contains": specifics["build_and_launch"].get("kernel_name_contains")},
        },
        "input": specifics["input"],
        "build_and_launch": specifics["build_and_launch"],
        "buffer_abi": {"num_sms": args.num_sms, "num_tracks": int(abi["buffer"].get("num_tracks", len(abi["tracks"]))),
                       "max_records": args.max_records, "element_dtype": abi["buffer"]["element_dtype"]},
        # scale:1 and mode:"shift-only" are INVARIANTS, not tunables: the device clock and the
        # nsys clock advance at the same rate, so only the origin differs. Anything other than
        # scale 1 means some stage is rubber-sheeting HW counters onto software spans until they
        # "look aligned" -- that hides the structural fact that nsys_span > sw_span (nsys brackets
        # launch + teardown). emit_perfetto.merge_hw therefore hard-fails on scale != 1. Only
        # residual_max_ns below is a knob (tighten it once you have measured the real residual).
        "alignment": {"scale": 1, "mode": "shift-only", "residual_max_ns": args.residual_max_ns},
        "validation": {"correctness": {"atol": args.atol, "rtol": args.rtol, "inputs": "randn", "scope": "all_ranks_all_elements"},
                       "overhead": {"max": args.overhead_max, "presets": ["off", "entry_only", "full"]}},
        "artifacts": {"max_bytes": args.max_bytes, "events_glob": "sw/events_rank*.json",
                      "hw_metrics": "hw/hw_metrics.json", "alignment": "work/alignment.json",
                      "perfetto": "work/candidate.perfetto.json.gz"},
        "components": {"trace_processor_shell": {"absolute_path": str(Path(args.tps_path).resolve()) if args.tps_path else None,
                                                 "sha256": sha256_of(Path(args.tps_path)) if args.tps_path and Path(args.tps_path).is_file() else None},
                       "host": component_hashes()},
    }
    return contract


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Freeze a runnable contract from 5 choices + device probe.")
    parser.add_argument(
        "--information-bundle", choices=("sw_only", "sw_hw"), default="sw_hw",
        help="information bundle; default requires both software and hardware evidence")
    parser.add_argument("--granularity", choices=("task", "phase"), required=True)
    parser.add_argument("--display-rank", type=int, required=True)
    parser.add_argument("--software-sm-scope", default="all_sms",
                        help="all_sms | subset:<smid,smid,...>")
    parser.add_argument("--hardware", choices=("on", "off"), required=True)
    parser.add_argument("--abi", type=Path, default=DEFAULT_ABI)
    parser.add_argument("--preset", default="full")
    parser.add_argument("--metrics-set", default="gpu-pm-sampling")
    parser.add_argument("--display-device", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--num-sms", type=int, default=None,
                        help="REQUIRED: launch grid = number of CTAs (for a persistent kernel, the SM "
                             "count you launch on, e.g. 78 on H20). No default on purpose: a guessed "
                             "value sizes the record buffer for the wrong number of CTAs.")
    parser.add_argument("--max-records", type=int, default=None,
                        help="REQUIRED: 2 records (begin+end) per event x max events one observer track "
                             "writes per launch, plus headroom. No default on purpose: the old 4096 was "
                             "copied verbatim and silently dropped every record past it. Verify with "
                             "decode_profile.py reporting truncated_tracks == [].")
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--overhead-max", type=float, default=None,
                        help="overhead ceiling written into the contract; default follows "
                             "--granularity: 0.10 for task, 0.15 for phase (the skill-side "
                             "mandatory audit hard-caps task at 0.10 and phase at 0.15)")
    parser.add_argument("--residual-max-ns", type=int, default=5000)
    parser.add_argument("--max-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--tps-path", type=str, default=None,
                        help="absolute path to trace_processor_shell (validation real-import check)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.overhead_max is None:
        args.overhead_max = 0.15 if args.granularity == "phase" else 0.10
    contract = build_contract(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(contract, indent=2, sort_keys=False) + "\n"
    args.out.write_text(payload, encoding="utf-8")
    # freeze fingerprint over the serialized body (excluding the fingerprint itself)
    contract["lock"]["frozen_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    args.out.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print("[materialize] wrote frozen contract {} (bundle={} granularity={} rank={} hw={})".format(
        args.out, args.information_bundle, args.granularity, args.display_rank, args.hardware))
    return 0


if __name__ == "__main__":
    sys.exit(main())
