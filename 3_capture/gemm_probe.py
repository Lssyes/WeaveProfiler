#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast, correctness-first GEMM software/hardware probe.

This command is a measurement probe, not a delivery receipt.  It builds the
same GEMM source at three instrumentation levels, verifies every output element
against an independent FP32 reference, measures recorder-detached and
recorder-attached execution, captures a root-authorized 200 kHz NSYS hardware
window, and merges software spans with GPU counters on one time base.

The example command is emitted by ``run.py template --mode gemm-probe``.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "weaveprofiler-gemm-probe/v1"
RESULT_SCHEMA = "weaveprofiler-gemm-probe-result/v1"
TOOL_ID = "weaveprofiler-gemm-probe/v1"
BUILD_VERSION = "gemm-probe-2026-11-14-v4"
SKILL_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = SKILL_ROOT / "knowledge" / "templates" / "gemm_probe"
DEFAULT_SOURCE = TEMPLATE_DIR / "gemm.cu"
DEFAULT_NAMES = TEMPLATE_DIR / "names.json"
DEFAULT_METRICS_CONFIG = TEMPLATE_DIR / "gemm_h20.config"
DEFAULT_METRICS_ALLOWLIST = TEMPLATE_DIR / "metrics.json"
RECORDER_DIR = SKILL_ROOT / "2_instrument" / "recorders" / "cuda_cpp"
DECODER = SKILL_ROOT / "4_validate" / "tools" / "fi_lite_decode.py"
ALIGNER = SKILL_ROOT / "4_validate" / "tools" / "fi_align_nsys.py"
CAPTURE = SKILL_ROOT / "3_capture" / "capture_nsys.py"
EXTRACT = SKILL_ROOT / "3_capture" / "extract_gpu_metrics.py"
TRACE_PROCESSOR_FETCHER = SKILL_ROOT / "1_scan" / "get_trace_processor.py"
KERNEL_NEEDLE = "gemm_wmma"
DEFAULT_SHAPE = (2048, 2048, 4096)
DEFAULT_SHAPE_TEXT = "2048x2048x4096"
QUICK_SHAPE_TEXT = "512x512x512"
DEFAULT_CAPTURE_ITERS = 8
DEFAULT_BENCHMARK_ROUNDS = 7
DEFAULT_BENCHMARK_ITERS = 7
QUICK_BENCHMARK_ROUNDS = 3
QUICK_BENCHMARK_ITERS = 5
QUICK_CAPTURE_ITERS = 2
DEFAULT_FREQUENCY_HZ = 200_000
REGISTER_CALIBRATION_POLICY = "auto-min-cap-no-spill-v1"
REGISTER_CALIBRATION_STEP = 16
REGISTER_CALIBRATION_MAX = 255
VARIANT_DEFINES: Dict[str, List[str]] = {
    "off": ["-DFI_LEVEL=0"],
    "entry": ["-DFI_LEVEL=1", "-DFI_TIMER=0", "-DFI_GRAN=1"],
    "full": ["-DFI_LEVEL=2", "-DFI_TIMER=0", "-DFI_GRAN=1", "-DFI_GROUPS=0x7"],
}
VARIANT_ORDER = ("off", "entry", "full")
CARD_GPU_NAME_PATTERNS = {
    "h20": re.compile(r"\bH20\b", re.IGNORECASE),
}


class ProbeError(RuntimeError):
    """A probe stage failed with evidence that is safe to report."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    partial.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                       encoding="utf-8")
    os.replace(str(partial), str(path))


def parse_shape(text: str) -> Tuple[int, int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*[xX×]\s*(\d+)\s*", str(text))
    if match is None:
        raise ProbeError("shape must be MxNxK, for example 2048x2048x4096")
    m, n, k = (int(match.group(i)) for i in range(1, 4))
    if m <= 0 or n <= 0 or k <= 0:
        raise ProbeError("shape dimensions must be positive")
    if m % 64 or n % 64:
        raise ProbeError("M and N must be multiples of the 64x64 CTA tile")
    if k % 16:
        raise ProbeError("K must be a multiple of the 16-deep K tile")
    return m, n, k


def parse_nvidia_smi_inventory(gpu_text: str, app_text: str = "") -> Dict[str, Any]:
    gpus: List[Dict[str, Any]] = []
    for raw in gpu_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 6:
            raise ProbeError("unexpected nvidia-smi GPU row: {!r}".format(raw))
        try:
            index = int(parts[0])
            memory_used = float(parts[3])
            memory_total = float(parts[4])
            utilization = float(parts[5])
        except ValueError as exc:
            raise ProbeError("cannot parse nvidia-smi GPU row {!r}: {}".format(raw, exc))
        gpus.append({
            "index": index,
            "name": parts[1],
            "uuid": parts[2],
            "memory_used_mib": memory_used,
            "memory_total_mib": memory_total,
            "utilization_pct": utilization,
            "foreign_processes": [],
        })
    by_uuid = {item["uuid"]: item for item in gpus}
    for raw in app_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [item.strip() for item in line.split(",", 2)]
        if len(parts) < 2:
            continue
        gpu_uuid = parts[0]
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        process_name = parts[2] if len(parts) > 2 else ""
        if gpu_uuid in by_uuid:
            by_uuid[gpu_uuid]["foreign_processes"].append({
                "pid": pid, "process_name": process_name})
    return {"gpus": gpus, "by_uuid": by_uuid}


def query_gpu_inventory() -> Dict[str, Any]:
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProbeError("cannot query nvidia-smi GPUs: {}".format(exc))
    try:
        app = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProbeError("cannot query nvidia-smi compute processes: {}".format(exc))
    inventory = parse_nvidia_smi_inventory(gpu.stdout, app.stdout)
    inventory["gpu_query_raw"] = gpu.stdout.strip().splitlines()
    inventory["app_query_raw"] = app.stdout.strip().splitlines()
    return inventory


def choose_idle_gpu(inventory: Dict[str, Any], requested: Optional[str], *,
                    memory_limit_mib: float = 1024.0,
                    utilization_limit_pct: float = 10.0,
                    expected_card: Optional[str] = None) -> Dict[str, Any]:
    gpus = inventory.get("gpus") or []
    if not gpus:
        raise ProbeError("nvidia-smi reported no GPUs")
    model_pattern = CARD_GPU_NAME_PATTERNS.get(str(expected_card or "").lower())
    candidates: List[Dict[str, Any]] = []
    for gpu in gpus:
        item = dict(gpu)
        item["expected_card"] = expected_card
        item["model_matches"] = bool(
            model_pattern.search(str(item.get("name", ""))) if model_pattern else True)
        item["over_memory"] = float(item["memory_used_mib"]) > memory_limit_mib
        item["over_utilization"] = float(item["utilization_pct"]) > utilization_limit_pct
        item["idle"] = (not item["foreign_processes"] and not item["over_memory"]
                        and not item["over_utilization"])
        candidates.append(item)
    if requested and requested != "auto":
        try:
            index = int(requested)
        except ValueError:
            raise ProbeError("--gpu must be auto or a non-negative physical GPU index")
        selected = next((item for item in candidates if item["index"] == index), None)
        if selected is None:
            raise ProbeError("requested GPU {} does not exist".format(index))
        if not selected["model_matches"]:
            raise ProbeError(
                "requested GPU {} is {!r}, expected card {}; refusing a mismatched "
                "measurement target".format(index, selected.get("name"), expected_card))
        if not selected["idle"]:
            raise ProbeError(
                "requested GPU {} is not idle (memory={} MiB, util={}%, processes={}); "
                "wait for it to become idle".format(
                    index, selected["memory_used_mib"], selected["utilization_pct"],
                    selected["foreign_processes"]))
        selected = dict(selected)
        selected["selection"] = "explicit"
        selected["memory_limit_mib"] = memory_limit_mib
        selected["utilization_limit_pct"] = utilization_limit_pct
        return selected
    idle = [item for item in candidates if item["idle"] and item["model_matches"]]
    if not idle:
        states = ", ".join(
            "GPU{} mem={}MiB util={}% procs={}".format(
                item["index"], item["memory_used_mib"], item["utilization_pct"],
                len(item["foreign_processes"]))
            for item in candidates)
        qualifier = " {} ".format(expected_card) if expected_card else " "
        raise ProbeError("no idle{}GPU is available; refusing to disturb other processes: {}".format(
            qualifier, states))
    idle.sort(key=lambda item: (item["memory_used_mib"], item["utilization_pct"], item["index"]))
    selected = dict(idle[0])
    selected["selection"] = "auto"
    selected["memory_limit_mib"] = memory_limit_mib
    selected["utilization_limit_pct"] = utilization_limit_pct
    return selected


def run_subprocess(argv: Sequence[str], *, cwd: Path, env: Optional[Dict[str, str]],
                   timeout: float, log_path: Path) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    # nvidia-smi indices and CUDA performance counters use PCI bus order.  Keep
    # CUDA's process-visible ordering identical unless this helper is explicitly
    # given another policy by a caller.
    child_env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    try:
        proc = subprocess.Popen(list(argv), cwd=str(cwd), env=child_env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
        stdout, stderr = proc.communicate(timeout=timeout)
        code = int(proc.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = _terminate_process_tree(proc)
        if not stdout:
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        if not stderr:
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        code = 124
        timed_out = True
    elapsed = time.monotonic() - started
    log_text = "$ " + " ".join(argv) + "\n"
    if stdout:
        log_text += stdout
        if not stdout.endswith("\n"):
            log_text += "\n"
    if stderr:
        log_text += stderr
        if not stderr.endswith("\n"):
            log_text += "\n"
    log_text += "[exit_code] {}\n[elapsed_seconds] {:.6f}\n".format(code, elapsed)
    log_path.write_text(log_text, encoding="utf-8", errors="replace")
    return {"argv": list(argv), "exit_code": code, "stdout": stdout, "stderr": stderr,
            "elapsed_seconds": elapsed, "timed_out": timed_out, "log_path": str(log_path)}


def _terminate_process_tree(proc: subprocess.Popen, grace_seconds: float = 5.0
                            ) -> Tuple[str, str]:
    """Terminate the profiler's whole process group, then collect its output."""
    if hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            try:
                proc.terminate()
            except OSError:
                pass
    elif os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False, timeout=10)
    else:
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        stdout, stderr = proc.communicate(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        if hasattr(os, "killpg"):
            try:
                os.killpg(proc.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            except OSError:
                try:
                    proc.kill()
                except OSError:
                    pass
        elif os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False, timeout=10)
        else:
            try:
                proc.kill()
            except OSError:
                pass
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
    return stdout or "", stderr or ""


def require_success(result: Dict[str, Any], label: str) -> None:
    if result["exit_code"] != 0:
        tail = (result.get("stdout", "") + "\n" + result.get("stderr", "")).strip()[-4000:]
        raise ProbeError("{} failed with exit {} (log {}):\n{}".format(
            label, result["exit_code"], result["log_path"], tail))


def parse_json_result(stdout: str, label: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ProbeError("{} did not emit a JSON result".format(label))


def bounded_timeout(deadline_mono: Optional[float], cap_seconds: float) -> float:
    """Return a positive timeout bounded by the probe's absolute deadline."""
    if cap_seconds <= 0:
        raise ProbeError("subprocess timeout cap must be positive")
    if deadline_mono is None:
        return cap_seconds
    remaining = deadline_mono - time.monotonic()
    if remaining <= 0:
        raise ProbeError("probe deadline exceeded before the next subprocess")
    return min(cap_seconds, remaining)


def root_prefix() -> List[str]:
    if os.geteuid() == 0:
        return []
    sudo = shutil.which("sudo")
    if not sudo:
        raise ProbeError("GPU performance counters require root, and sudo is unavailable")
    probe = subprocess.run([sudo, "-n", "true"], capture_output=True, text=True, timeout=30)
    if probe.returncode != 0:
        raise ProbeError("GPU performance counters require root; `sudo -n true` failed")
    return [sudo, "-n"]


def _chown_tree(path: Path) -> None:
    if os.geteuid() == 0:
        return
    prefix = root_prefix()
    command = prefix + ["chown", "-R", "{}:{}".format(os.getuid(), os.getgid()), str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise ProbeError("failed to return root-created capture files to the caller: {}".format(
            (result.stderr or result.stdout).strip()))


def parse_ptxas_log(text: str, kernel_needle: str = KERNEL_NEEDLE) -> Dict[str, Any]:
    blocks = re.split(r"(?=ptxas info\s+: Compiling entry function ')", text)
    block = next((item for item in blocks
                  if "Compiling entry function" in item and kernel_needle in item), None)
    if block is None:
        return {
            "has_register_count": False,
            "has_spill_count": False,
            "matched_kernel": None,
        }
    header = re.search(r"Compiling entry function '([^']+)'", block)
    patterns = {
        "registers_per_thread": r"Used\s+(\d+)\s+registers",
        "spill_stores_bytes": r"(\d+)\s+bytes\s+spill\s+stores",
        "spill_loads_bytes": r"(\d+)\s+bytes\s+spill\s+loads",
        "stack_frame_bytes": r"(\d+)\s+bytes\s+stack frame",
        "stack_bytes": r"(\d+)\s+bytes\s+stack",
    }
    result: Dict[str, Any] = {}
    for key, pattern in patterns.items():
        values = re.findall(pattern, block)
        if values:
            result[key] = int(values[-1])
    result["matched_kernel"] = header.group(1) if header else None
    result["has_register_count"] = "registers_per_thread" in result
    result["has_spill_count"] = (
        "spill_stores_bytes" in result and "spill_loads_bytes" in result)
    return result


def register_cap_candidates(seed_registers: int) -> List[int]:
    """Escalate from the observed register footprint until ptxas can stay spill-free."""
    start = max(32, min(REGISTER_CALIBRATION_MAX, int(seed_registers)))
    values = list(range(start, REGISTER_CALIBRATION_MAX + 1,
                        REGISTER_CALIBRATION_STEP))
    if not values or values[-1] != REGISTER_CALIBRATION_MAX:
        values.append(REGISTER_CALIBRATION_MAX)
    return values


def build_cache_key(source: Path, arch: str, nvcc_version: str) -> str:
    digest = hashlib.sha256()
    digest.update(BUILD_VERSION.encode("utf-8"))
    digest.update(b"\0arch\0")
    digest.update(arch.encode("utf-8"))
    digest.update(b"\0nvcc\0")
    digest.update(nvcc_version.encode("utf-8"))
    for label, path in (("source", source), ("recorder", RECORDER_DIR / "fi_lite.cuh")):
        digest.update(("\0" + label + "\0").encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    for variant in VARIANT_ORDER:
        digest.update(("\0variant\0" + variant + "\0" + " ".join(VARIANT_DEFINES[variant])).encode("utf-8"))
    digest.update(("\0register-calibration\0{}\0step={}\0max={}\0".format(
        REGISTER_CALIBRATION_POLICY, REGISTER_CALIBRATION_STEP,
        REGISTER_CALIBRATION_MAX)).encode("ascii"))
    return digest.hexdigest()


def _cache_entry_valid(entry: Path, cache_key: str) -> bool:
    meta_path = entry / "meta.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("schema") != "weaveprofiler-gemm-build-cache/v1" or meta.get("cache_key") != cache_key:
        return False
    for variant in VARIANT_ORDER:
        info = (meta.get("variants") or {}).get(variant) or {}
        binary = Path(str(info.get("binary", "")))
        if not binary.is_file() or not os.access(str(binary), os.X_OK):
            return False
        if (info.get("spill_stores_bytes", 1) or info.get("spill_loads_bytes", 1)
                or info.get("stack_frame_bytes", 0) or info.get("stack_bytes", 0)):
            return False
    return True


def build_variants(source: Path, arch: str, cache_root: Path,
                   deadline_mono: Optional[float] = None) -> Dict[str, Any]:
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).is_file():
        raise ProbeError("nvcc not found; set PATH or install CUDA")
    version_result = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=60)
    if version_result.returncode != 0:
        raise ProbeError("nvcc --version failed: {}".format(
            (version_result.stderr or version_result.stdout).strip()))
    nvcc_version = version_result.stdout.strip()
    cache_key = build_cache_key(source, arch, nvcc_version)
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / (cache_key + ".lock")
    started = time.monotonic()
    with lock_path.open("a+", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        base_entry = cache_root / ("entry-" + cache_key[:20])
        if _cache_entry_valid(base_entry, cache_key):
            meta = json.loads((base_entry / "meta.json").read_text(encoding="utf-8"))
            meta = dict(meta)
            meta["cache_hit"] = True
            meta["build_seconds"] = 0.0
            meta["elapsed_seconds"] = time.monotonic() - started
            return meta
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        entry = base_entry if not base_entry.exists() else \
            cache_root / ("entry-" + cache_key[:20] + "-rebuild-" + stamp)
        temp = cache_root / (".building-" + cache_key[:20] + "-" + uuid.uuid4().hex)
        temp.mkdir(parents=True)
        processes: List[Tuple[str, subprocess.Popen, Any, float]] = []

        def compile_one(label: str, variant: str,
                        register_cap: Optional[int] = None) -> None:
            variant_dir = temp / label
            variant_dir.mkdir()
            log_path = variant_dir / "nvcc.log"
            command = [
                nvcc, "-O3", "-std=c++17",
                "-gencode", "arch=compute_90a,code=sm_90a",
                "-I", str(RECORDER_DIR), "-Xptxas", "-v",
            ]
            if register_cap is not None:
                command.append("--maxrregcount={}".format(register_cap))
            command.extend([
                *VARIANT_DEFINES[variant],
                str(source), "-lcurand", "-o", str(variant_dir / "gemm"),
            ])
            log_handle = log_path.open("wb")
            proc = subprocess.Popen(command, cwd=str(temp), stdout=log_handle,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((label, proc, log_handle, time.monotonic()))

        def wait_one(label: str, proc: subprocess.Popen,
                     log_handle: Any) -> int:
            try:
                code = proc.wait(timeout=bounded_timeout(deadline_mono, 1800.0))
            except subprocess.TimeoutExpired:
                _terminate_process_tree(proc)
                code = 124
            log_handle.close()
            return code

        def parsed_evidence(label: str) -> Dict[str, Any]:
            log_path = temp / label / "nvcc.log"
            parsed = parse_ptxas_log(
                log_path.read_text(encoding="utf-8", errors="replace"))
            if not parsed.get("has_register_count") or not parsed.get("has_spill_count"):
                raise ProbeError(
                    "cannot parse ptxas register/spill evidence from {}".format(log_path))
            return parsed

        def spill_free(parsed: Dict[str, Any]) -> bool:
            return not (parsed.get("spill_stores_bytes", 0)
                        or parsed.get("spill_loads_bytes", 0)
                        or parsed.get("stack_frame_bytes", 0)
                        or parsed.get("stack_bytes", 0))

        try:
            for variant in VARIANT_ORDER:
                compile_one(variant, variant)
            failures: List[str] = []
            for label, proc, log_handle, _ in list(processes):
                code = wait_one(label, proc, log_handle)
                if code != 0:
                    failures.append("{} exit={} log={}".format(
                        label, code, temp / label / "nvcc.log"))
            if failures:
                raise ProbeError("GEMM build failed: " + "; ".join(failures))
            parsed_variants = {variant: parsed_evidence(variant)
                               for variant in VARIANT_ORDER}
            for variant in ("off", "entry"):
                parsed = parsed_variants[variant]
                if not spill_free(parsed):
                    raise ProbeError("{} spills registers or stack: {}".format(
                        variant, parsed))
            variants: Dict[str, Any] = {}
            for variant in ("off", "entry"):
                variants[variant] = {
                    "binary": str(entry / variant / "gemm"),
                    "log": str(entry / variant / "nvcc.log"),
                    **parsed_variants[variant],
                }

            full_default = parsed_variants["full"]
            calibration: Dict[str, Any] = {
                "policy": REGISTER_CALIBRATION_POLICY,
                "step": REGISTER_CALIBRATION_STEP,
                "max_registers": REGISTER_CALIBRATION_MAX,
                "default": full_default,
                "attempts": [],
                "selected_cap": None,
                "selected_label": "full",
            }
            selected_label = "full"
            if not spill_free(full_default):
                observed_peak = max(
                    int(parsed_variants[name].get("registers_per_thread", 0))
                    for name in VARIANT_ORDER)
                candidates = register_cap_candidates(observed_peak)
                calibration["seed_registers"] = observed_peak
                calibration["candidate_caps"] = candidates
                selected_parsed: Optional[Dict[str, Any]] = None
                for cap in candidates:
                    label = "full_cap{:03d}".format(cap)
                    compile_one(label, "full", register_cap=cap)
                    _label, proc, log_handle, _started = processes[-1]
                    code = wait_one(label, proc, log_handle)
                    attempt: Dict[str, Any] = {
                        "cap": cap, "label": label, "exit_code": code}
                    if code == 0:
                        parsed = parsed_evidence(label)
                        attempt.update(parsed)
                        attempt["spill_free"] = spill_free(parsed)
                    else:
                        attempt["spill_free"] = False
                    calibration["attempts"].append(attempt)
                    if code == 0 and attempt["spill_free"]:
                        selected_label = label
                        selected_parsed = parsed
                        calibration["selected_cap"] = cap
                        calibration["selected_label"] = label
                        break
                if selected_parsed is None:
                    raise ProbeError(
                        "full instrumentation stays spill-bound after adaptive register "
                        "calibration: {}".format(calibration))
                calibration["selection_rule"] = (
                    "smallest maxrregcount at or above the observed peak register "
                    "footprint that compiles with zero spills and no stack")
                variants["full"] = {
                    "binary": str(entry / selected_label / "gemm"),
                    "log": str(entry / selected_label / "nvcc.log"),
                    **selected_parsed,
                }
            else:
                calibration["selection_rule"] = "default ptxas allocation already spill-free"
                variants["full"] = {
                    "binary": str(entry / "full" / "gemm"),
                    "log": str(entry / "full" / "nvcc.log"),
                    **full_default,
                }
            meta = {
                "schema": "weaveprofiler-gemm-build-cache/v1",
                "build_version": BUILD_VERSION,
                "cache_key": cache_key,
                "source": str(source),
                "source_sha256": sha256_file(source),
                "recorder_sha256": sha256_file(RECORDER_DIR / "fi_lite.cuh"),
                "arch": arch,
                "nvcc": nvcc,
                "nvcc_version": nvcc_version,
                "defines": VARIANT_DEFINES,
                "register_calibration": calibration,
                "variants": variants,
                "cache_hit": False,
            }
            (temp / "meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8")
            os.replace(str(temp), str(entry))
            meta["build_seconds"] = time.monotonic() - started
            meta["elapsed_seconds"] = meta["build_seconds"]
            return meta
        except Exception:
            raise
        finally:
            for _variant, proc, log_handle, _started in processes:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                if not log_handle.closed:
                    log_handle.close()
            if temp.exists():
                shutil.rmtree(temp, ignore_errors=True)
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def run_probe_binary(binary: Path, mode: str, shape: Tuple[int, int, int], *,
                     gpu: int, warmup: int, iters: int, log_path: Path,
                     dump: Optional[Path] = None, capture_iters: int = DEFAULT_CAPTURE_ITERS,
                     cuda_profiler_range: bool = False,
                     timeout_sec: float = 600.0) -> Dict[str, Any]:
    command = [str(binary), "--shape", "profile",
               "--m", str(shape[0]), "--n", str(shape[1]), "--k", str(shape[2]),
               "--mode", mode, "--warmup", str(warmup), "--iters", str(iters)]
    if mode == "nsys":
        command.extend(["--capture-iters", str(capture_iters)])
        if cuda_profiler_range:
            command.append("--cuda-profiler-range")
    if dump is not None:
        command.extend(["--dump", str(dump)])
    result = run_subprocess(command, cwd=log_path.parent,
                            env={"CUDA_VISIBLE_DEVICES": str(gpu)},
                            timeout=timeout_sec, log_path=log_path)
    require_success(result, "GEMM {} {}".format(binary.name, mode))
    payload = parse_json_result(result["stdout"], "GEMM {}".format(mode))
    if payload.get("status") != "PASS":
        raise ProbeError("GEMM {} returned status={!r}".format(mode, payload.get("status")))
    return payload


def check_correctness_payload(payload: Dict[str, Any], shape: Tuple[int, int, int],
                              label: str) -> Dict[str, Any]:
    expected = shape[0] * shape[1]
    checked = int(payload.get("checked_elements", -1))
    if checked != expected:
        raise ProbeError("{} checked {} elements, expected {}".format(label, checked, expected))
    if int(payload.get("mismatches", -1)) != 0 or int(payload.get("nonfinite", -1)) != 0:
        raise ProbeError("{} correctness failed: {}".format(label, payload))
    return {
        "label": label,
        "checked_elements": checked,
        "max_abs": float(payload.get("max_abs", float("nan"))),
        "relative_l2": float(payload.get("rel_l2", float("nan"))),
        "nonfinite": int(payload["nonfinite"]),
        "mismatches": int(payload["mismatches"]),
        "median_us": float(payload.get("median_us", float("nan"))) if "median_us" in payload else None,
        "measurement_scope": payload.get("measurement_scope", "recorder_detached"),
    }


def benchmark_variants(build: Dict[str, Any], shape: Tuple[int, int, int], *,
                       gpu: int, rounds: int, iters: int, out_dir: Path,
                       max_attached_overhead_pct: Optional[float],
                       deadline_mono: Optional[float] = None) -> Dict[str, Any]:
    if rounds < 1 or iters < 1:
        raise ProbeError("--benchmark-rounds and --benchmark-iters must be positive")
    tasks = ["off:detached", "entry:detached", "full:detached",
             "entry:attached", "full:attached"]
    rng = random.Random(20261114)
    raw: List[Dict[str, Any]] = []
    correctness: List[Dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for round_index in range(rounds):
        order = list(tasks)
        rng.shuffle(order)
        for task in order:
            variant, attachment = task.split(":", 1)
            mode = "bench-attached" if attachment == "attached" else "bench"
            log = out_dir / "round{:02d}_{}_{}.log".format(
                round_index + 1, variant, attachment)
            payload = run_probe_binary(
                Path(build["variants"][variant]["binary"]), mode, shape, gpu=gpu,
                warmup=3, iters=iters, log_path=log,
                timeout_sec=bounded_timeout(deadline_mono, 600.0))
            check = check_correctness_payload(
                payload, shape, "{} {} round {}".format(variant, attachment, round_index + 1))
            correctness.append(check)
            raw.append({
                "round": round_index + 1,
                "order": order.index(task) + 1,
                "variant": variant,
                "attachment": attachment,
                "measurement_scope": payload.get(
                    "measurement_scope", "recorder_detached"),
                "median_us": float(payload["median_us"]),
                "min_us": float(payload["min_us"]),
                "max_us": float(payload["max_us"]),
                "tflops": float(payload["tflops"]),
                "checked_elements": check["checked_elements"],
                "mismatches": check["mismatches"],
                "nonfinite": check["nonfinite"],
            })
    grouped: Dict[str, List[float]] = {}
    for item in raw:
        key = "{}:{}".format(item["variant"], item["attachment"])
        grouped.setdefault(key, []).append(item["median_us"])
    medians = {key: statistics.median(values) for key, values in grouped.items()}
    off = medians["off:detached"]
    overhead = {}
    for key, value in sorted(medians.items()):
        if key == "off:detached":
            continue
        overhead[key] = (value - off) / off * 100.0
    attached_full = overhead["full:attached"]
    # A tiny smoke shape magnifies fixed per-CTA probe cost and is not a
    # representative overhead measurement.  Quick mode records the number but
    # leaves the delivery threshold to the one full-shape run.
    if max_attached_overhead_pct is not None and attached_full > max_attached_overhead_pct:
        raise ProbeError(
            "full recorder-attached overhead {:.3f}% exceeds limit {:.3f}%".format(
                attached_full, max_attached_overhead_pct))
    return {
        "status": "PASS",
        "rounds": rounds,
        "iters_per_round": iters,
        "warmup": 3,
        "random_order_seed": 20261114,
        "medians_us": medians,
        "overhead_pct": overhead,
        "max_attached_overhead_pct": max_attached_overhead_pct,
        "overhead_gate_enforced": max_attached_overhead_pct is not None,
        "raw": raw,
        "correctness": {
            "runs": len(correctness),
            "checked_elements_per_run": shape[0] * shape[1],
            "all_runs_checked_all_outputs": all(
                item["checked_elements"] == shape[0] * shape[1] for item in correctness),
            "mismatches_total": sum(item["mismatches"] for item in correctness),
            "nonfinite_total": sum(item["nonfinite"] for item in correctness),
            "max_abs_max": max(item["max_abs"] for item in correctness),
            "relative_l2_max": max(item["relative_l2"] for item in correctness),
            "runs_detail": correctness,
        },
    }


def parse_ncu_csv(text: str) -> List[Dict[str, str]]:
    lines = [line for line in text.splitlines() if line.strip().startswith('"')]
    if not lines:
        return []
    return list(csv.DictReader(lines))


def run_ncu_counter(binary: Path, variant: str, gpu: int, shape: Tuple[int, int, int],
                    log_path: Path, timeout_sec: float = 900.0) -> Dict[str, Any]:
    prefix = root_prefix()
    ncu = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
    if not Path(ncu).is_file():
        raise ProbeError("ncu not found")
    metric = "sm__inst_executed_pipe_tensor_op_hmma.sum"
    command = prefix + [
        "env", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES={}".format(gpu), ncu,
        "--target-processes", "all", "--kernel-name", "regex:" + KERNEL_NEEDLE,
        "--launch-count", "1", "--metrics", metric, "--csv",
        str(binary), "--shape", "profile",
        "--m", str(shape[0]), "--n", str(shape[1]), "--k", str(shape[2]),
        "--mode", "bench", "--warmup", "1", "--iters", "1",
    ]
    result = run_subprocess(command, cwd=log_path.parent, env=None,
                            timeout=timeout_sec, log_path=log_path)
    require_success(result, "ncu {}".format(variant))
    rows = parse_ncu_csv(result["stdout"])
    matched = [row for row in rows if row.get("Metric Name") == metric]
    if not matched:
        raise ProbeError("ncu {} emitted no {} row".format(variant, metric))
    value = int(float(matched[-1]["Metric Value"]))
    return {
        "variant": variant,
        "metric": metric,
        "value": value,
        "kernel_name": matched[-1].get("Kernel Name"),
        "log": str(log_path),
    }


def check_ncu(build: Dict[str, Any], gpu: int, shape: Tuple[int, int, int],
              out_dir: Path, deadline_mono: Optional[float] = None) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    off = run_ncu_counter(Path(build["variants"]["off"]["binary"]), "off", gpu, shape,
                          out_dir / "ncu_off.log",
                          timeout_sec=bounded_timeout(deadline_mono, 900.0))
    full = run_ncu_counter(Path(build["variants"]["full"]["binary"]), "full", gpu, shape,
                           out_dir / "ncu_full.log",
                           timeout_sec=bounded_timeout(deadline_mono, 900.0))
    if off["value"] <= 0:
        raise ProbeError("ncu measured zero HMMA instructions in the off build")
    if off["value"] != full["value"]:
        raise ProbeError("HMMA instruction count moved across instrumentation: off={} full={}".format(
            off["value"], full["value"]))
    return {
        "status": "PASS",
        "metric": off["metric"],
        "off": off,
        "full": full,
        "counts_equal": True,
        "interpretation": (
            "first matching benchmark launch has the same dynamic HMMA count "
            "with and without software event insertion"),
    }


def run_hardware_capture(run_dir: Path, build: Dict[str, Any], gpu: int,
                         shape: Tuple[int, int, int], capture_iters: int,
                         frequency_hz: int, timeout_sec: float,
                         deadline_mono: Optional[float] = None) -> Dict[str, Any]:
    prefix = root_prefix()
    hardware_dir = run_dir / "hardware"
    hardware_dir.mkdir(parents=True, exist_ok=True)
    dump = hardware_dir / "gemm_nsys.fi.bin"
    capture_timeout = bounded_timeout(deadline_mono, timeout_sec)
    command = prefix + [
        sys.executable, "-B", str(CAPTURE),
        "--root", str(run_dir),
        "--out-dir", str(hardware_dir),
        "--nsys", shutil.which("nsys") or "/usr/local/bin/nsys",
        "--gpu-metrics-device", str(gpu),
        "--gpu-metrics-frequency", str(frequency_hz),
        "--metrics-set", "file:" + str(DEFAULT_METRICS_CONFIG),
        "--kernel-name-contains", KERNEL_NEEDLE,
        "--num-processes", "1",
        "--expected-launch-count", str(capture_iters),
        "--cuda-visible-devices", str(gpu),
        "--timeout-sec", str(capture_timeout),
        "--payload", str(build["variants"]["full"]["binary"]),
        "--shape", "profile",
        "--m", str(shape[0]), "--n", str(shape[1]), "--k", str(shape[2]),
        "--mode", "nsys", "--warmup", "3",
        "--capture-iters", str(capture_iters),
        "--cuda-profiler-range", "--dump", str(dump),
    ]
    log = hardware_dir / "gemm_probe_capture_wrapper.log"
    result = run_subprocess(command, cwd=run_dir, env=None,
                            timeout=bounded_timeout(deadline_mono, capture_timeout + 120),
                            log_path=log)
    if os.geteuid() != 0:
        _chown_tree(hardware_dir)
    require_success(result, "root NSYS hardware capture")
    manifest_path = hardware_dir / "capture_manifest.json"
    if not manifest_path.is_file():
        raise ProbeError("NSYS capture manifest is missing: {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("verdict") != "PASS":
        raise ProbeError("NSYS capture verdict is {!r}; see {}".format(
            manifest.get("verdict"), manifest_path))
    evidence = manifest.get("evidence") or {}
    kernel = evidence.get("kernel") or {}
    if int(kernel.get("match_rows", -1)) != capture_iters:
        raise ProbeError("NSYS target launch count mismatch: {}".format(kernel))
    if not (kernel.get("valid") is True):
        raise ProbeError("NSYS target launch validation failed: {}".format(
            kernel.get("hard_failures")))
    sampling = evidence.get("sampling_interval_ns") or {}
    max_check = sampling.get("max_check") or {}
    if not max_check.get("pass"):
        raise ProbeError("NSYS sampling continuity check failed: {}".format(sampling))
    expected_ns = 1_000_000_000 / frequency_hz
    median_ns = sampling.get("median")
    if median_ns is None or abs(float(median_ns) - expected_ns) / expected_ns > 0.25:
        raise ProbeError("NSYS sampling median is not near {:.0f} ns: {}".format(
            expected_ns, sampling))
    if not dump.is_file():
        raise ProbeError("NSYS run did not produce the software dump: {}".format(dump))
    return {
        "status": "PASS",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "capture_seconds": manifest.get("capture_seconds"),
        "gpu": gpu,
        "gpu_name": ((manifest.get("gpu_check") or {}).get("gpus") or [{}])[0].get("name"),
        "frequency_hz": frequency_hz,
        "expected_launches": capture_iters,
        "observed_launches": int(kernel.get("match_rows")),
        "sampling_interval_ns": sampling,
        "dump": str(dump),
        "dump_sha256": sha256_file(dump),
        "sqlite": str(hardware_dir / "capture.sqlite"),
        "nsys_rep": str(hardware_dir / "capture.nsys-rep"),
        "capture_log": result["log_path"],
    }


def decode_software(dump: Path, names: Path, shape: Tuple[int, int, int],
                    out_dir: Path,
                    deadline_mono: Optional[float] = None) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_base = out_dir / "gemm_sw.perfetto.json"
    summary_path = out_dir / "gemm_sw.summary.json"
    command = [sys.executable, "-B", str(DECODER), str(dump),
               "--names", str(names), "--require-semantics",
               "-o", str(trace_base), "--summary", str(summary_path)]
    result = run_subprocess(command, cwd=out_dir, env={"PYTHONDONTWRITEBYTECODE": "1"},
                            timeout=bounded_timeout(deadline_mono, 600.0),
                            log_path=out_dir / "decode.log")
    require_success(result, "software dump decode")
    trace = out_dir / "gemm_sw.perfetto.json.gz"
    if not trace.is_file() or not summary_path.is_file():
        raise ProbeError("decoder did not produce trace/summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ctas = (shape[0] // 64) * (shape[1] // 64)
    k_tiles = shape[2] // 16
    expected_spans = ctas * (4 * k_tiles + 2)
    expected_records = expected_spans * 2 + 2 * ctas
    if summary.get("errors"):
        raise ProbeError("software decode reported structural errors: {}".format(summary["errors"]))
    if int(summary.get("spans", -1)) != expected_spans:
        raise ProbeError("software span count {} != expected {}".format(
            summary.get("spans"), expected_spans))
    if int(summary.get("records", -1)) != expected_records:
        raise ProbeError("software record count {} != expected {}".format(
            summary.get("records"), expected_records))
    if summary.get("semantics_ok") is not True:
        raise ProbeError("software semantic resolution failed: {}".format(
            summary.get("semantic_errors")))
    expected_events = {
        "gemm": ctas,
        "store": ctas,
        "load": ctas * k_tiles,
        "wait": ctas * k_tiles,
        "mma": ctas * k_tiles,
        "barrier": ctas * k_tiles,
    }
    observed_events = {name: int((summary.get("events") or {}).get(name, {}).get("n", -1))
                       for name in expected_events}
    if observed_events != expected_events:
        raise ProbeError("software event counts {} != expected {}".format(
            observed_events, expected_events))
    return {
        "status": "PASS",
        "trace": str(trace),
        "trace_sha256": sha256_file(trace),
        "summary": str(summary_path),
        "summary_document": summary,
        "expected_spans": expected_spans,
        "expected_records": expected_records,
        "ctas": ctas,
        "k_tiles": k_tiles,
        "event_counts": observed_events,
    }


def extract_hardware_metrics(gpu: int, out_dir: Path, hardware: Dict[str, Any],
                             frequency_hz: int,
                             deadline_mono: Optional[float] = None) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "hw_metrics.json"
    command = [sys.executable, "-B", str(EXTRACT),
               "--sqlite", hardware["sqlite"],
               "--manifest", hardware["manifest"],
               "--kernel-name-contains", KERNEL_NEEDLE,
               "--device", str(gpu),
               "--metrics-allowlist", str(DEFAULT_METRICS_ALLOWLIST),
               "--out", str(output)]
    result = run_subprocess(command, cwd=out_dir, env={"PYTHONDONTWRITEBYTECODE": "1"},
                            timeout=bounded_timeout(deadline_mono, 600.0),
                            log_path=out_dir / "extract.log")
    require_success(result, "hardware metric extraction")
    doc = json.loads(output.read_text(encoding="utf-8"))
    schema = json.loads(DEFAULT_METRICS_ALLOWLIST.read_text(encoding="utf-8"))
    missing_required = (doc.get("metrics_filter") or {}).get("missing_required") or []
    if missing_required:
        raise ProbeError("hardware capture is missing required metrics: {}".format(missing_required))
    interval = doc.get("sampling_interval_ns") or {}
    expected_ns = 1_000_000_000 / frequency_hz
    if not interval.get("available"):
        raise ProbeError("hardware sampling interval is unavailable: {}".format(interval))
    if abs(float(interval["median"]) - expected_ns) / expected_ns > 0.25:
        raise ProbeError("hardware sampling median moved away from 200 kHz: {}".format(interval))
    if int(interval.get("max", 0)) > expected_ns * 10:
        raise ProbeError("hardware metric gap exceeds 10x expected period: {}".format(interval))
    required = list(schema.get("required") or [])
    matched = set((doc.get("metrics_filter") or {}).get("matched") or [])
    missing = [name for name in required if name not in matched]
    if missing:
        raise ProbeError("required hardware counters were not extracted: {}".format(missing))
    counters = {str(item.get("name")): item for item in doc.get("counters") or []}
    tensor = counters.get("Tensor Active") or {}
    if int(tensor.get("sample_count_before_clip", 0)) <= 0:
        raise ProbeError("Tensor Active has no sampled evidence")
    hardware_summary = doc.get("hardware_summary") or {}
    metrics = hardware_summary.get("metrics") or {}
    for semantic in ("compute_util", "memory_util", "sm_active"):
        if (metrics.get(semantic) or {}).get("status") != "available":
            raise ProbeError("hardware semantic {} is unavailable: {}".format(
                semantic, metrics.get(semantic)))
    return {
        "status": "PASS",
        "json": str(output),
        "json_sha256": sha256_file(output),
        "sampling_interval_ns": interval,
        "missing_required": missing_required,
        "tensor_active_samples_before_clip": int(tensor.get("sample_count_before_clip", 0)),
        "hardware_summary": hardware_summary,
        "counters": {name: {
            "sample_count": item.get("sample_count"),
            "sample_count_before_clip": item.get("sample_count_before_clip"),
            "unit": item.get("unit"),
        } for name, item in counters.items()},
        "log": result["log_path"],
    }


def align_software_hardware(hardware: Dict[str, Any], shape: Tuple[int, int, int],
                            build: Dict[str, Any], benchmark: Dict[str, Any],
                            out_dir: Path,
                            deadline_mono: Optional[float] = None,
                            software: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    info_path = out_dir / "alignment_info.json"
    info = {
        "probe": TOOL_ID,
        "delivery_evidence": False,
        "shape": {"m": shape[0], "n": shape[1], "k": shape[2]},
        "build": {"cache_key": build.get("cache_key"),
                  "variants": {name: {k: (build.get("variants") or {}).get(name, {}).get(k) for k in (
                      "registers_per_thread", "spill_stores_bytes", "spill_loads_bytes")}
                               for name in VARIANT_ORDER}},
        "benchmark_medians_us": benchmark.get("medians_us"),
        "instrumentation_overhead_pct": benchmark.get("overhead_pct"),
    }
    atomic_write_json(info_path, info)
    output_base = out_dir / "gemm_sw_hw.perfetto.json"
    command = [sys.executable, "-B", str(ALIGNER), hardware["dump"],
               "--names", str(DEFAULT_NAMES),
               "--sqlite", hardware["sqlite"],
               "--kernel", KERNEL_NEEDLE,
               "--instance", "-1",
               "--metrics", "SMs Active|SM Issue|Tensor Active|SM Tensor Pipe Active|DRAM Read|DRAM Write",
               "--info-json", str(info_path),
               "--min-metrics-samples", "3",
               "--min-metric-curves", "6"]
    if software:
        for key in ("trace", "summary"):
            if not software.get(key) or not Path(str(software[key])).is_file():
                raise ProbeError("cached software {} is unavailable: {}".format(key, software.get(key)))
        command.extend(["--decoded-trace", str(software["trace"]),
                        "--decoded-summary", str(software["summary"])])
    command.extend(["-o", str(output_base)])
    result = run_subprocess(command, cwd=out_dir, env={"PYTHONDONTWRITEBYTECODE": "1"},
                            timeout=bounded_timeout(deadline_mono, 900.0),
                            log_path=out_dir / "align.log")
    require_success(result, "software/hardware alignment")
    merged = out_dir / "gemm_sw_hw.perfetto.json.gz"
    summary_path = out_dir / "gemm_sw_hw.perfetto.summary.json"
    if not merged.is_file() or not summary_path.is_file():
        raise ProbeError("alignment did not produce merged trace/summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    align = summary.get("align") or {}
    if align.get("ok") is not True or align.get("tail_ok") is not True or align.get("metrics_ok") is not True:
        raise ProbeError("alignment checks failed: {}".format(align))
    kernel_us = float(align.get("kernel_us", 0.0))
    sw_span_us = float(align.get("sw_span_us", 0.0))
    if kernel_us <= 0:
        raise ProbeError("alignment kernel duration is invalid: {}".format(align))
    alignment_error_pct = abs(sw_span_us - kernel_us) / kernel_us * 100.0
    return {
        "status": "PASS",
        "trace": str(merged),
        "trace_sha256": sha256_file(merged),
        "summary": str(summary_path),
        "summary_document": summary,
        "align": align,
        "alignment_error_pct": alignment_error_pct,
        "time_zero": "selected NSYS GEMM launch start",
        "log": result["log_path"],
        "software_decode_reused": bool(software),
    }


def resolve_trace_processor(explicit: Optional[str], cache_root: Path,
                            out_dir: Path,
                            deadline_mono: Optional[float] = None) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file() or not os.access(str(candidate), os.X_OK):
            raise ProbeError("--trace-processor is not an executable file: {}".format(candidate))
        return candidate
    env_candidate = os.environ.get("TRACE_PROCESSOR_SHELL")
    if env_candidate:
        candidate = Path(env_candidate).expanduser().resolve()
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return candidate
    out_dir.mkdir(parents=True, exist_ok=True)
    tool_dir = cache_root / "trace_processor"
    tool_dir.mkdir(parents=True, exist_ok=True)
    result = run_subprocess(
        [sys.executable, "-B", str(TRACE_PROCESSOR_FETCHER), "--out-dir", str(tool_dir)],
        cwd=out_dir, env={"PYTHONDONTWRITEBYTECODE": "1"},
        timeout=bounded_timeout(deadline_mono, 900.0),
        log_path=out_dir / "get_trace_processor.log")
    require_success(result, "trace_processor_shell discovery")
    path_text = result["stdout"].strip().splitlines()
    if not path_text:
        raise ProbeError("trace_processor fetcher returned no path")
    candidate = Path(path_text[-1].strip()).expanduser().resolve()
    if not candidate.is_file() or not os.access(str(candidate), os.X_OK):
        raise ProbeError("trace_processor fetcher returned an unusable path: {}".format(candidate))
    return candidate


FORBIDDEN_IMPORT_STATS = (
    "json_parser_failure",
    "json_tokenizer_failure",
    "slice_spill_overlapping_complete_event",
    "slice_drop_overlapping_complete_event",
    "trace_sorter_negative_timestamp_dropped",
)


def trace_import_query() -> str:
    bad_names = ",".join("'{}'".format(name) for name in FORBIDDEN_IMPORT_STATS)
    return (
        "WITH bad AS (SELECT name FROM stats WHERE value > 0 AND name IN ({bad})) "
        "SELECT (SELECT COUNT(*) FROM bad) AS forbidden_stat_count, "
        "(SELECT COUNT(*) FROM slice) AS imported_slices, "
        "(SELECT COUNT(*) FROM counter) AS imported_counters, "
        "(SELECT COUNT(DISTINCT track_id) FROM counter) AS counter_tracks;"
    ).format(bad=bad_names)


def validate_trace_import(trace_processor: Optional[str], trace: Path,
                          cache_root: Path, out_dir: Path,
                          expected_slices: int,
                          deadline_mono: Optional[float] = None) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tps = resolve_trace_processor(trace_processor, cache_root, out_dir,
                                  deadline_mono=deadline_mono)
    query = trace_import_query()
    result = run_subprocess([str(tps), "-Q", query, str(trace)],
                            cwd=out_dir, env=None,
                            timeout=bounded_timeout(deadline_mono, 900.0),
                            log_path=out_dir / "trace_import.log")
    require_success(result, "real Perfetto JSON import")
    rows = list(csv.DictReader((result["stdout"] or "").splitlines()))
    if not rows:
        raise ProbeError("trace_processor returned no import statistics")
    row = rows[0]
    try:
        forbidden = int(row["forbidden_stat_count"])
        slices = int(row["imported_slices"])
        counters = int(row["imported_counters"])
        tracks = int(row["counter_tracks"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProbeError("cannot parse trace_processor import row: {} ({})".format(row, exc))
    if forbidden != 0:
        raise ProbeError("Perfetto import hit forbidden stats: {}".format(row))
    if slices < expected_slices:
        raise ProbeError("Perfetto retained {} slices, expected at least {}".format(
            slices, expected_slices))
    if counters <= 0 or tracks < 6:
        raise ProbeError("Perfetto imported no usable hardware counter tracks: {}".format(row))
    return {
        "status": "PASS",
        "trace_processor": str(tps),
        "trace": str(trace),
        "trace_sha256": sha256_file(trace),
        "forbidden_stat_count": forbidden,
        "imported_slices": slices,
        "expected_slices_minimum": expected_slices,
        "imported_counters": counters,
        "counter_tracks": tracks,
        "log": result["log_path"],
    }


def artifact(path: Path, role: str) -> Dict[str, Any]:
    if not path.is_file():
        return {"role": role, "path": str(path), "exists": False}
    return {"role": role, "path": str(path), "exists": True,
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def execute(args: argparse.Namespace) -> Dict[str, Any]:
    started_wall = time.time()
    started_mono = time.monotonic()
    deadline_mono = started_mono + args.deadline_sec
    # Keep nvidia-smi's physical index, CUPTI's device id, and nsys'
    # --gpu-metrics-devices index in one documented ordering.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    quick = args.profile == "quick"
    shape_text = args.shape or (QUICK_SHAPE_TEXT if quick else DEFAULT_SHAPE_TEXT)
    shape = parse_shape(shape_text)
    benchmark_rounds = args.benchmark_rounds if args.benchmark_rounds is not None else (
        QUICK_BENCHMARK_ROUNDS if quick else DEFAULT_BENCHMARK_ROUNDS)
    benchmark_iters = args.benchmark_iters if args.benchmark_iters is not None else (
        QUICK_BENCHMARK_ITERS if quick else DEFAULT_BENCHMARK_ITERS)
    capture_iters = args.capture_iters if args.capture_iters is not None else (
        QUICK_CAPTURE_ITERS if quick else DEFAULT_CAPTURE_ITERS)
    skip_ncu = bool(args.skip_ncu or quick)
    # Keep the real Perfetto import in quick mode: it is cheap on the
    # smoke trace and catches the class of trace bugs that otherwise surface
    # only after a full -- expensive -- run.
    skip_trace_import = bool(args.skip_trace_import)
    if args.card != "h20" or args.arch != "sm_90":
        raise ProbeError("the built-in GEMM probe currently supports only --card h20 --arch sm_90")
    if args.frequency_hz != DEFAULT_FREQUENCY_HZ:
        raise ProbeError("this probe enforces {} Hz hardware sampling; observed {}".format(
            DEFAULT_FREQUENCY_HZ, args.frequency_hz))
    if benchmark_rounds <= 0 or benchmark_iters <= 0:
        raise ProbeError("benchmark rounds and iterations must be positive")
    if capture_iters < 2:
        raise ProbeError("--capture-iters must be at least 2")
    if args.deadline_sec <= 0:
        raise ProbeError("--deadline-sec must be positive")
    source = Path(args.source).expanduser().resolve() if args.source else DEFAULT_SOURCE
    if not source.is_file():
        raise ProbeError("GEMM source does not exist: {}".format(source))
    for path in (DEFAULT_NAMES, DEFAULT_METRICS_CONFIG, DEFAULT_METRICS_ALLOWLIST,
                 RECORDER_DIR / "fi_lite.cuh", DECODER, ALIGNER, CAPTURE, EXTRACT):
        if not path.is_file():
            raise ProbeError("required skill asset is missing: {}".format(path))
    run_dir = Path(args.out).expanduser().resolve() if args.out else (
        Path.cwd() / ("gemm_probe_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ProbeError("--out must be new or empty: {}".format(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root = (Path(args.cache_dir).expanduser().resolve() if args.cache_dir else
                  run_dir.parent / ".weaveprofiler" / "cache" / "gemm_probe")
    stages: Dict[str, Any] = {}
    manifest_path = run_dir / "gemm_probe_manifest.json"
    partial: Dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL_ID,
        "delivery_evidence": False,
        "probe_scope": "single-shape GEMM hardware/software measurement probe",
        "probe_profile": args.profile,
        "started_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "target": {"card": args.card, "arch": args.arch, "language": "cuda_cpp",
                   "shape": {"m": shape[0], "n": shape[1], "k": shape[2]}},
        "deadline_seconds": args.deadline_sec,
        "stages": stages,
        "status": "RUNNING",
    }
    atomic_write_json(manifest_path, partial)

    def stage(name: str, fn: Any) -> Any:
        remaining = args.deadline_sec - (time.monotonic() - started_mono)
        if remaining <= 0:
            raise ProbeError("probe deadline of {:.0f}s exceeded before {}".format(
                args.deadline_sec, name))
        print("[gemm-probe] {} ...".format(name), flush=True)
        t0 = time.monotonic()
        try:
            value = fn()
        except Exception as exc:
            stages[name] = {"status": "FAIL", "seconds": round(time.monotonic() - t0, 3),
                            "error": "{}: {}".format(type(exc).__name__, exc)}
            raise
        if time.monotonic() > deadline_mono:
            raise ProbeError("probe deadline of {:.0f}s exceeded during {}".format(
                args.deadline_sec, name))
        stages[name] = {"status": "PASS", "seconds": round(time.monotonic() - t0, 3)}
        print("[gemm-probe] {} PASS {:.3f}s".format(name, stages[name]["seconds"]), flush=True)
        return value

    build: Optional[Dict[str, Any]] = None
    benchmark: Optional[Dict[str, Any]] = None
    ncu: Optional[Dict[str, Any]] = None
    hardware: Optional[Dict[str, Any]] = None
    software: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    alignment: Optional[Dict[str, Any]] = None
    trace_import: Optional[Dict[str, Any]] = None
    gpu: Optional[Dict[str, Any]] = None
    try:
        inventory = stage("gpu_inventory", query_gpu_inventory)
        gpu = choose_idle_gpu(inventory, args.gpu,
                              memory_limit_mib=args.memory_limit_mib,
                              utilization_limit_pct=args.utilization_limit_pct,
                              expected_card=args.card)
        partial["target"]["gpu"] = gpu
        partial["gpu_inventory"] = {
            "gpus": [{k: item.get(k) for k in (
                "index", "name", "uuid", "memory_used_mib", "memory_total_mib",
                "utilization_pct", "foreign_processes", "idle")}
                for item in inventory["gpus"]],
            "selection": gpu["selection"],
        }
        atomic_write_json(manifest_path, partial)
        build = stage("build", lambda: build_variants(
            source, args.arch, cache_root, deadline_mono=deadline_mono))
        partial["build"] = build
        atomic_write_json(manifest_path, partial)
        benchmark = stage("benchmark", lambda: benchmark_variants(
            build, shape, gpu=gpu["index"], rounds=benchmark_rounds,
            iters=benchmark_iters, out_dir=run_dir / "benchmark",
            max_attached_overhead_pct=(None if quick else args.max_attached_overhead_pct),
            deadline_mono=deadline_mono))
        partial["benchmark"] = benchmark
        partial["correctness"] = benchmark["correctness"]
        atomic_write_json(manifest_path, partial)
        if skip_ncu:
            ncu = {"status": "SKIPPED", "reason": "--profile quick" if quick else "--skip-ncu"}
            stages["ncu"] = {"status": "SKIPPED", "seconds": 0.0}
        else:
            ncu = stage("ncu", lambda: check_ncu(build, gpu["index"], shape,
                                                 run_dir / "ncu",
                                                 deadline_mono=deadline_mono))
        partial["ncu"] = ncu
        atomic_write_json(manifest_path, partial)
        # Recheck the selected GPU immediately before the long capture.  Never
        # evict or kill a process that appeared in the meantime.
        inventory2 = query_gpu_inventory()
        recheck = choose_idle_gpu(inventory2, str(gpu["index"]),
                                  memory_limit_mib=args.memory_limit_mib,
                                  utilization_limit_pct=args.utilization_limit_pct,
                                  expected_card=args.card)
        partial["gpu_recheck_before_capture"] = recheck
        hardware = stage("hardware_capture", lambda: run_hardware_capture(
            run_dir, build, gpu["index"], shape, capture_iters,
            args.frequency_hz, args.hardware_timeout_sec,
            deadline_mono=deadline_mono))
        partial["hardware_capture"] = hardware
        atomic_write_json(manifest_path, partial)
        software = stage("software_decode", lambda: decode_software(
            Path(hardware["dump"]), DEFAULT_NAMES, shape, run_dir / "software",
            deadline_mono=deadline_mono))
        partial["software"] = {k: v for k, v in software.items() if k != "summary_document"}
        atomic_write_json(manifest_path, partial)
        metrics = stage("metrics_extract", lambda: extract_hardware_metrics(
            gpu["index"], run_dir / "hardware_metrics", hardware, args.frequency_hz,
            deadline_mono=deadline_mono))
        partial["hardware_metrics"] = metrics
        atomic_write_json(manifest_path, partial)
        alignment = stage("alignment", lambda: align_software_hardware(
            hardware, shape, build, benchmark, run_dir / "alignment",
            deadline_mono=deadline_mono, software=software))
        partial["alignment"] = {k: v for k, v in alignment.items() if k != "summary_document"}
        atomic_write_json(manifest_path, partial)
        if skip_trace_import:
            trace_import = {"status": "SKIPPED",
                            "reason": "--profile quick" if quick else "--skip-trace-import"}
            stages["trace_import"] = {"status": "SKIPPED", "seconds": 0.0}
        else:
            trace_import = stage("trace_import", lambda: validate_trace_import(
                args.trace_processor, Path(alignment["trace"]), cache_root,
                run_dir / "trace_import", software["expected_spans"],
                deadline_mono=deadline_mono))
        partial["trace_import"] = trace_import
        partial["status"] = "PASS"
        return_code = 0
    except Exception as exc:
        partial["status"] = "FAIL"
        partial["error"] = "{}: {}".format(type(exc).__name__, exc)
        print("[gemm-probe] FAIL: {}".format(partial["error"]), file=sys.stderr, flush=True)
        return_code = 1
    finally:
        elapsed = time.monotonic() - started_mono
        partial["finished_at_utc"] = utc_now()
        partial["wall_seconds"] = round(elapsed, 3)
        partial["speed_budget"] = {
            "target_seconds": args.deadline_sec,
            "within_budget": elapsed <= args.deadline_sec,
        }
        artifacts: List[Dict[str, Any]] = []
        if build:
            for variant in VARIANT_ORDER:
                info = (build.get("variants") or {}).get(variant) or {}
                if info.get("binary"):
                    artifacts.append(artifact(Path(info["binary"]), "build_" + variant))
        if hardware:
            for key, role in (("manifest", "capture_manifest"), ("dump", "software_dump"),
                              ("sqlite", "capture_sqlite"), ("nsys_rep", "nsys_report")):
                if hardware.get(key):
                    artifacts.append(artifact(Path(hardware[key]), role))
        if software:
            artifacts.extend([artifact(Path(software["trace"]), "software_trace"),
                              artifact(Path(software["summary"]), "software_summary")])
        if metrics:
            artifacts.append(artifact(Path(metrics["json"]), "hardware_metrics"))
        if alignment:
            artifacts.extend([artifact(Path(alignment["trace"]), "merged_trace"),
                              artifact(Path(alignment["summary"]), "alignment_summary")])
        partial["artifacts"] = artifacts
        partial["limitations"] = [
            "measurement probe only; delivery_evidence=false",
            "one H20 sm_90 GEMM shape and one physical GPU",
            "NSYS GPU metrics are device-wide and sampled periodically",
            "communication_util is intentionally unavailable for this single-GPU GEMM",
        ]
        # Snapshot references before writing the manifest, because writing adds
        # a final timestamp only to this document.
        atomic_write_json(manifest_path, partial)
        partial["manifest"] = str(manifest_path)
        print(json.dumps({
            "schema": RESULT_SCHEMA,
            "status": partial["status"],
            "probe_profile": partial["probe_profile"],
            "delivery_evidence": False,
            "manifest": str(manifest_path),
            "wall_seconds": partial["wall_seconds"],
            "within_budget": partial["speed_budget"]["within_budget"],
            "cache_hit": bool((build or {}).get("cache_hit")),
            "benchmark_medians_us": (benchmark or {}).get("medians_us"),
            "overhead_pct": (benchmark or {}).get("overhead_pct"),
            "correctness": {
                "runs": ((benchmark or {}).get("correctness") or {}).get("runs"),
                "checked_elements_per_run": ((benchmark or {}).get("correctness") or {}).get(
                    "checked_elements_per_run"),
                "mismatches_total": ((benchmark or {}).get("correctness") or {}).get("mismatches_total"),
                "nonfinite_total": ((benchmark or {}).get("correctness") or {}).get("nonfinite_total"),
            },
            "software_spans": (software or {}).get("expected_spans"),
            "trace_import": (trace_import or {}).get("status"),
            "trace": (alignment or {}).get("trace"),
            "error": partial.get("error"),
        }, indent=2, ensure_ascii=False))
    return return_code


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build, verify, measure, and profile the built-in H20 GEMM probe")
    parser.add_argument("--card", default="h20", choices=("h20",))
    parser.add_argument("--arch", default="sm_90", choices=("sm_90",))
    parser.add_argument("--profile", choices=("full", "quick"), default="full",
                        help="full runs the complete evidence path; quick uses a small smoke shape, "
                             "3 rounds, 2 capture launches, skips NCU, and still validates Perfetto import")
    parser.add_argument("--shape", default=None,
                        help="MxNxK; default 2048x2048x4096 for full and 512x512x512 for quick")
    parser.add_argument("--gpu", default="auto",
                        help="auto or a physical GPU index; auto only selects an idle GPU")
    parser.add_argument("--out", help="new/empty absolute run directory")
    parser.add_argument("--cache-dir", help="persistent build cache directory")
    parser.add_argument("--source", help="advanced: override the built-in GEMM source")
    parser.add_argument("--benchmark-rounds", type=int, default=None)
    parser.add_argument("--benchmark-iters", type=int, default=None)
    parser.add_argument("--capture-iters", type=int, default=None)
    parser.add_argument("--frequency-hz", type=int, default=DEFAULT_FREQUENCY_HZ)
    parser.add_argument("--hardware-timeout-sec", type=float, default=600.0)
    parser.add_argument("--deadline-sec", type=float, default=1800.0,
                        help="total wall-clock budget; default 30 minutes")
    parser.add_argument("--memory-limit-mib", type=float, default=1024.0)
    parser.add_argument("--utilization-limit-pct", type=float, default=10.0)
    parser.add_argument("--max-attached-overhead-pct", type=float, default=10.0)
    parser.add_argument("--trace-processor", help="trace_processor_shell path")
    parser.add_argument("--skip-ncu", action="store_true")
    parser.add_argument("--skip-trace-import", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return execute(args)
    except ProbeError as exc:
        print("[gemm-probe] ERROR: {}".format(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print("[gemm-probe] UNEXPECTED {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
