#!/usr/bin/env python3
"""Build a same-run software + ROCm hardware timeline from rocprof-sys.

The hardware layer is collected by ``rocprof-sys-run --rocm-events`` in the
same process and Perfetto protobuf as the kernel dispatch slices.  Raw ROCm
events are paired with a dispatch by their shared start timestamp.  No amd-smi
GPU-busy or memory-used value is promoted to a compute/memory utilization
metric.

MI308X Matrix Core utilization follows ROCm's gfx942 ``MfmaUtil`` definition::

    100 * SQ_VALU_MFMA_BUSY_CYCLES /
        (GRBM_GUI_ACTIVE * (CU_NUM * SIMD_PER_CU / XCC_NUM))

For MI308X this denominator uses 80 SIMD lanes per XCC (20 CU * 4 SIMD).
Memory utilization is marked as an estimate because the raw DRAM request
counters can represent either 32-byte or 64-byte requests; the byte model and
measured HBM peak used are written into the trace metadata.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

CAPTURE_DIR = Path(__file__).resolve().parent
if str(CAPTURE_DIR) not in sys.path:
    sys.path.insert(0, str(CAPTURE_DIR))

from merge_hw_into_perfetto import (  # noqa: E402
    read_json_maybe_gz,
    rebase_timeline_nonnegative,
    timestamp_report,
    write_json_gz,
)

SCHEMA = "rocprof-sys-sw-hw/v2"
DEFAULT_DURATION_RATIO_TOLERANCE = 0.20
DEFAULT_START_PLACEMENT_TOLERANCE_US = 1.0
PER_DISPATCH_COUNTER_RESOLUTION = "per_dispatch_aggregate"
DEFAULT_KERNEL_NAME_CONTAINS = "mfma_kernel"
CANONICAL_TRACE_RE = re.compile(
    r"^trace__(?P<model>[a-z0-9]+)_(?P<stage>[a-z0-9]+)"
    r"__card-(?P<card>h20|b300|mi308x)"
    r"__bs(?P<batch>\d+)_sq(?P<seqlen_q>\d+)(?:_kv(?P<seqlen_kv>\d+))?"
    r"__(?P<dtype>[a-z0-9]+)"
    r"__grid(?P<grid>\d+)_warp(?P<warp>\d+)_ph(?P<phases>\d+)"
    r"__(?P<bundle>sw|hw|sw_hw)\.perfetto\.json\.gz$"
)

# Exact counter-track names emitted by rocprof-sys 1.3 / ROCm 7.2.
RAW_TRACKS: Dict[str, str] = {
    "mfma_busy_cycles": "GPU SQ_VALU_MFMA_BUSY_CYCLES [0]",
    "gui_active_cycles": "GPU GRBM_GUI_ACTIVE [0]",
    "simd_utilization_fraction": "GPU SIMD_UTILIZATION [0]",
    "gpu_util_percent": "GPU GPU_UTIL [0]",
    "dram_read_requests": "GPU TCC_EA0_RDREQ_DRAM [0]",
    "dram_write_requests": "GPU TCC_EA0_WRREQ_DRAM [0]",
}
TRACK_TO_RAW = {track: raw for raw, track in RAW_TRACKS.items()}

METRIC_LABELS = {
    "compute_util": "compute_util [matrix_core]",
    "tensor_util": "tensor_util [matrix_core]",
    "simd_util": "simd_util [SIMD/CU]",
    "communication_util": "communication_util [interconnect]",
    "memory_util": "memory_util [HBM request estimate]",
    "memory_read_util": "memory_read_util [HBM request estimate]",
    "memory_write_util": "memory_write_util [HBM request estimate]",
    "sm_active": "sm_active [GPU active]",
}


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def _query_rows(tps: Path, proto: Path, sql: str) -> List[Dict[str, str]]:
    proc = subprocess.run(
        [str(tps), "-Q", sql.rstrip() + "\n", str(proto)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise ValueError(
            "trace_processor failed rc={}: {}".format(
                proc.returncode,
                ((proc.stderr or "") + (proc.stdout or ""))[-3000:],
            )
        )
    text = "\n".join(
        line for line in proc.stdout.splitlines() if not line.startswith("Loading trace:")
    )
    return list(csv.DictReader(io.StringIO(text)))


def _float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nearest(grouped: Mapping[int, Mapping[str, float]], target: int,
             tolerance_ns: int) -> Tuple[Optional[int], Optional[Mapping[str, float]]]:
    if not grouped:
        return None, None
    ordered = sorted(grouped)
    pos = bisect.bisect_left(ordered, target)
    candidates = [index for index in (pos - 1, pos) if 0 <= index < len(ordered)]
    if not candidates:
        return None, None
    chosen = min(candidates, key=lambda index: abs(ordered[index] - target))
    ts = ordered[chosen]
    if abs(ts - target) > tolerance_ns:
        return None, None
    return ts, grouped[ts]


def _median(values: Iterable[float]) -> Optional[float]:
    data = [float(value) for value in values]
    return statistics.median(data) if data else None


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _range_status(value: Optional[float], low: float, high: float) -> bool:
    return value is not None and low <= value <= high


def extract_dispatches(
    tps: Path,
    proto: Path,
    kernel_name_contains: str,
    *,
    kernel_ordinal: Optional[int] = None,
    skip_first: int = 0,
    limit: Optional[int] = None,
    timestamp_tolerance_ns: int = 5_000,
    cu_count: int = 80,
    simd_per_cu: int = 4,
    xcc_count: int = 4,
    hbm_peak_gbps: Optional[float] = 3684.0,
    hbm_request_bytes: int = 64,
) -> Dict[str, Any]:
    """Extract matching dispatch slices and same-timestamp ROCm counters."""
    needle = str(kernel_name_contains or "").strip()
    if not needle:
        raise ValueError("--kernel-name-contains must not be empty")
    if cu_count <= 0 or simd_per_cu <= 0 or xcc_count <= 0:
        raise ValueError("CU/SIMD/XCC counts must be positive")
    if hbm_request_bytes <= 0:
        raise ValueError("--hbm-request-bytes must be positive")
    if skip_first < 0:
        raise ValueError("--skip-first must be non-negative")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive when supplied")

    kernel_rows = _query_rows(
        tps,
        proto,
        "SELECT ts, dur, name FROM slice "
        "WHERE name LIKE '%{}%' AND dur > 0 ORDER BY ts;".format(_sql_quote(needle)),
    )
    if not kernel_rows:
        raise ValueError("rocprof-sys proto has no kernel slice matching {!r}".format(needle))
    all_slices: List[Dict[str, Any]] = []
    for row in kernel_rows:
        start = _float(row.get("ts"))
        duration = _float(row.get("dur"))
        if start is None or duration is None or start < 0 or duration <= 0:
            continue
        all_slices.append({"ts": int(round(start)), "dur": int(round(duration)),
                           "name": row.get("name")})
    if not all_slices:
        raise ValueError("matched kernel slices have invalid timestamps/durations")

    if kernel_ordinal is not None:
        if kernel_ordinal < 0 or kernel_ordinal >= len(all_slices):
            raise ValueError("--kernel-ordinal {} out of range ({} matches)".format(
                kernel_ordinal, len(all_slices)))
        selected_slices = [all_slices[kernel_ordinal]]
        selection_method = "timestamp_ordinal"
    else:
        selected_slices = all_slices[skip_first:]
        selection_method = "all_matches_after_skip_first"
    if limit is not None:
        selected_slices = selected_slices[:limit]
    if not selected_slices:
        raise ValueError("kernel selection is empty after --skip-first/--limit")

    track_names = ", ".join("'{}'".format(_sql_quote(name)) for name in RAW_TRACKS.values())
    counter_rows = _query_rows(
        tps,
        proto,
        "SELECT ct.name AS track_name, c.ts AS ts, c.value AS value "
        "FROM counter c JOIN counter_track ct ON c.track_id = ct.id "
        "WHERE ct.name IN ({}) ORDER BY c.ts;".format(track_names),
    )
    grouped: Dict[int, Dict[str, float]] = {}
    ignored_rows = 0
    for row in counter_rows:
        raw_key = TRACK_TO_RAW.get(str(row.get("track_name") or ""))
        ts = _float(row.get("ts"))
        value = _float(row.get("value"))
        if raw_key is None or ts is None or value is None:
            ignored_rows += 1
            continue
        grouped.setdefault(int(round(ts)), {})[raw_key] = value

    effective_simd_lanes = cu_count * simd_per_cu / xcc_count
    dispatches: List[Dict[str, Any]] = []
    for index, kernel in enumerate(selected_slices, start=1):
        sample_ts, raw = _nearest(grouped, kernel["ts"], int(timestamp_tolerance_ns))
        raw = dict(raw or {})
        duration_ns = int(kernel["dur"])
        metrics: Dict[str, Optional[float]] = {key: None for key in METRIC_LABELS}
        reasons: Dict[str, str] = {}

        mfma = raw.get("mfma_busy_cycles")
        gui = raw.get("gui_active_cycles")
        if _range_status(mfma, 0.0, float("inf")) and _range_status(gui, 0.0, float("inf")) and mfma > 0 and gui > 0:
            value = 100.0 * mfma / (gui * effective_simd_lanes)
            if _range_status(value, 0.0, 110.0):
                metrics["compute_util"] = value
                metrics["tensor_util"] = value
            else:
                reasons["compute_util"] = "MFMA utilization outside [0,110]%"
                reasons["tensor_util"] = reasons["compute_util"]
        else:
            reason = "positive SQ_VALU_MFMA_BUSY_CYCLES and GRBM_GUI_ACTIVE evidence is absent"
            reasons["compute_util"] = reason
            reasons["tensor_util"] = reason

        simd_fraction = raw.get("simd_utilization_fraction")
        if _range_status(simd_fraction, 0.0, 1.0):
            metrics["simd_util"] = simd_fraction * 100.0
        else:
            reasons["simd_util"] = "SIMD_UTILIZATION is absent or outside [0,1]"

        gpu_util = raw.get("gpu_util_percent")
        if _range_status(gpu_util, 0.0, 100.0):
            metrics["sm_active"] = gpu_util
        else:
            reasons["sm_active"] = "GPU_UTIL is absent or outside [0,100]"

        read_requests = raw.get("dram_read_requests")
        write_requests = raw.get("dram_write_requests")
        if hbm_peak_gbps is None or hbm_peak_gbps <= 0:
            reason = "measured HBM peak was not supplied"
            for key in ("memory_util", "memory_read_util", "memory_write_util"):
                reasons[key] = reason
        elif not (_range_status(read_requests, 0.0, float("inf"))
                  and _range_status(write_requests, 0.0, float("inf"))):
            reason = "TCC_EA0 DRAM read/write request evidence is absent"
            for key in ("memory_util", "memory_read_util", "memory_write_util"):
                reasons[key] = reason
        else:
            peak_bytes_s = float(hbm_peak_gbps) * 1.0e9
            read_bytes = read_requests * hbm_request_bytes
            write_bytes = write_requests * hbm_request_bytes
            read_bw = read_bytes * 1.0e9 / duration_ns
            write_bw = write_bytes * 1.0e9 / duration_ns
            metrics["memory_read_util"] = 100.0 * read_bw / peak_bytes_s
            metrics["memory_write_util"] = 100.0 * write_bw / peak_bytes_s
            metrics["memory_util"] = metrics["memory_read_util"] + metrics["memory_write_util"]

        reasons["communication_util"] = (
            "no same-dispatch XGMI/RDMA/fabric counter was collected in this single-GPU run"
        )

        dispatches.append({
            "ordinal": index,
            "source_slice_ordinal": all_slices.index(kernel) if kernel in all_slices else None,
            "name": kernel["name"],
            "start_ns": int(kernel["ts"]),
            "dur_ns": duration_ns,
            "counter_sample_ts_ns": sample_ts,
            "counter_sample_offset_ns": (None if sample_ts is None else sample_ts - kernel["ts"]),
            "raw": raw,
            "metrics": metrics,
            "unavailable_reasons": reasons,
        })

    series = []
    for metric_id, label in METRIC_LABELS.items():
        points = [
            [item["start_ns"], item["dur_ns"], float(item["metrics"][metric_id])]
            for item in dispatches if item["metrics"].get(metric_id) is not None
        ]
        values = [point[2] for point in points]
        status = "measured_or_derived" if points else "unavailable"
        entry: Dict[str, Any] = {
            "semantic_id": metric_id,
            "timeline_label": label,
            "unit": "percent_of_peak",
            "evidence_status": status,
            "point_count": len(points),
            "dispatch_count": len(dispatches),
            "coverage": len(points) / len(dispatches),
            "min": min(values) if values else None,
            "median": _median(values),
            "mean": statistics.fmean(values) if values else None,
            "max": max(values) if values else None,
            "points": points,
        }
        if metric_id in ("compute_util", "tensor_util"):
            entry.update({
                "source": "rocprof-sys:SQ_VALU_MFMA_BUSY_CYCLES+GRBM_GUI_ACTIVE",
                "native_metrics": ["SQ_VALU_MFMA_BUSY_CYCLES", "GRBM_GUI_ACTIVE"],
                "compute_pipe": "matrix_core",
                "method": ("100*MFMA_BUSY/(GRBM_GUI_ACTIVE * "
                           "(CU_NUM*SIMD_PER_CU/XCC_NUM)); same value exposed as compute_util and tensor_util"),
                "derived_constants": {
                    "cu_count": cu_count,
                    "simd_per_cu": simd_per_cu,
                    "xcc_count": xcc_count,
                    "simd_lanes_per_xcc": effective_simd_lanes,
                },
            })
        elif metric_id == "simd_util":
            entry.update({
                "source": "rocprof-sys:SIMD_UTILIZATION*100",
                "native_metrics": ["SIMD_UTILIZATION"],
                "compute_pipe": "simd",
                "method": "SIMD_UTILIZATION fraction multiplied by 100",
            })
        elif metric_id == "sm_active":
            entry.update({
                "source": "rocprof-sys:GPU_UTIL",
                "native_metrics": ["GPU_UTIL"],
                "method": "GPU_UTIL is emitted directly as percent",
            })
        elif metric_id.startswith("memory_"):
            entry.update({
                "source": "rocprof-sys:TCC_EA0_RDREQ_DRAM+TCC_EA0_WRREQ_DRAM",
                "native_metrics": ["TCC_EA0_RDREQ_DRAM", "TCC_EA0_WRREQ_DRAM"],
                "evidence_status": (status if hbm_peak_gbps else "unavailable"),
                "method": ("requests * HBM_REQUEST_BYTES / dispatch_duration / measured_HBM_peak; "
                           "request size is explicit and utilization is an estimate"),
                "hbm_peak_gbps": hbm_peak_gbps,
                "hbm_request_bytes": hbm_request_bytes,
            })
        else:
            entry.update({
                "source": "unavailable",
                "native_metrics": [],
                "method": None,
            })
        if not points:
            reasons_for_metric = [
                item["unavailable_reasons"].get(metric_id)
                for item in dispatches
                if item["unavailable_reasons"].get(metric_id)
            ]
            entry["reason"] = reasons_for_metric[0] if reasons_for_metric else "no evidence"
        series.append(entry)

    instruction_names = [item["name"] for item in selected_slices]
    return {
        "schema": SCHEMA,
        "collector": "rocprof-sys",
        "source_trace": str(Path(proto).resolve()),
        "trace_processor": str(Path(tps).resolve()),
        "kernel_selection": {
            "needle": needle,
            "match": "trace_processor slice name LIKE %needle%",
            "method": selection_method,
            "kernel_ordinal": kernel_ordinal,
            "skip_first": skip_first,
            "limit": limit,
            "matching_slice_count": len(all_slices),
            "selected_slice_count": len(selected_slices),
            "all_kernel_names": sorted(set(instruction_names)),
        },
        "alignment": {
            "clock": "rocprof-sys Perfetto protobuf",
            "method": "same-run kernel slice ts paired with exact/near-exact raw counter sample timestamp",
            "timestamp_tolerance_ns": int(timestamp_tolerance_ns),
            "same_run": True,
            "same_dispatch": True,
        },
        "native_tracks": list(RAW_TRACKS.values()),
        "ignored_counter_rows": ignored_rows,
        "dispatches": dispatches,
        "series": series,
        "availability": {
            "compute_util": any(item["metrics"].get("compute_util") is not None for item in dispatches),
            "tensor_util": any(item["metrics"].get("tensor_util") is not None for item in dispatches),
            "simd_util": any(item["metrics"].get("simd_util") is not None for item in dispatches),
            "memory_util": any(item["metrics"].get("memory_util") is not None for item in dispatches),
            "communication_util": False,
            "sm_active": any(item["metrics"].get("sm_active") is not None for item in dispatches),
        },
        "policy": {
            "amd_smi_used": False,
            "gpu_busy_used_as_compute": False,
            "simd_used_as_tensor": False,
            "missing_evidence_zero_filled": False,
            "memory_estimate_disclosed": True,
            "matrix_core_formula": "100*MFMA_BUSY/(GUI_ACTIVE*SIMD_LANES_PER_XCC)",
        },
    }


def _load_optional_json(path: Optional[str]) -> Any:
    if not path:
        return None
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


SEMANTIC_EVENT_IDS = ("gemm", "load", "mma", "store")
WAVE_IDENTITY_KEYS = ("cta", "warp", "sm", "xcc", "se", "cu", "simd")


def _timed_semantic_events(events: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        args = event.get("args")
        if not isinstance(args, dict) or args.get("semantic_id") not in SEMANTIC_EVENT_IDS:
            continue
        ts = _float(event.get("ts"))
        dur = _float(event.get("dur"))
        if ts is None or dur is None or ts < 0 or dur <= 0:
            continue
        selected.append(event)
    return selected


def _semantic_counts(events: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in events:
        args = event.get("args")
        if isinstance(args, dict):
            semantic_id = str(args.get("semantic_id") or "unknown")
            counts[semantic_id] = counts.get(semantic_id, 0) + 1
    return counts


def _physical_wave_count(events: Sequence[Mapping[str, Any]]) -> int:
    identities = set()
    for event in events:
        args = event.get("args")
        if not isinstance(args, dict) or not all(key in args for key in WAVE_IDENTITY_KEYS):
            continue
        identities.add(tuple(args[key] for key in WAVE_IDENTITY_KEYS))
    return len(identities)


def _add_series_events(events: List[Dict[str, Any]], series: Mapping[str, Any],
                       start_us: float, end_us: float, *, track_id: int = 1,
                       pid: int = 1,
                       include_zero_boundary: bool = True) -> int:
    value = float(series["metric_value"])
    name = str(series["timeline_label"])
    count = 0
    for ts, sample_value in ((start_us, value), (end_us, value)):
        events.append({
            "ph": "C",
            "name": name,
            "cat": "rocprof_sys_rocm_pm",
            "ts": round(ts, 6),
            "pid": int(pid),
            "tid": int(track_id),
            "args": {"value": round(sample_value, 9)},
        })
        count += 1
    if include_zero_boundary:
        events.append({
            "ph": "C",
            "name": name,
            "cat": "rocprof_sys_rocm_pm",
            "ts": round(end_us + 0.001, 6),
            "pid": int(pid),
            "tid": int(track_id),
            "args": {"value": 0.0},
        })
        count += 1
    return count


def _add_hardware_track(events: List[Dict[str, Any]], index: int,
                        label: str, *, pid: int = 1) -> None:
    tid = 100 + index
    events.extend([
        {"ph": "M", "name": "thread_name", "pid": int(pid), "tid": tid,
         "args": {"name": label}},
        {"ph": "M", "name": "thread_sort_index", "pid": int(pid), "tid": tid,
         "args": {"sort_index": 100 + index}},
    ])


def _alignment_tolerances(
    duration_ratio_tolerance: float,
    start_placement_tolerance_us: float,
) -> Tuple[float, float]:
    ratio_tolerance = _float(duration_ratio_tolerance)
    placement_tolerance = _float(start_placement_tolerance_us)
    if ratio_tolerance is None or ratio_tolerance <= 0.0:
        raise ValueError("duration ratio tolerance must be a positive finite number")
    if placement_tolerance is None or placement_tolerance < 0.0:
        raise ValueError("start placement tolerance must be a non-negative finite number")
    return ratio_tolerance, placement_tolerance


def _alignment_row(
    *,
    semantic_id: Any,
    dispatch_ordinal: Any,
    software_start_us: float,
    software_dur_us: float,
    hardware_start_ns: int,
    hardware_dur_ns: int,
    hardware_start_placement_us: float,
    duration_ratio_tolerance: float,
    start_placement_tolerance_us: float,
    **extra: Any,
) -> Dict[str, Any]:
    software_duration = float(software_dur_us)
    hardware_duration_ns = int(hardware_dur_ns)
    hardware_duration_us = hardware_duration_ns / 1000.0
    if software_duration <= 0.0:
        raise ValueError("software alignment duration must be positive")
    if hardware_duration_ns <= 0:
        raise ValueError("hardware alignment duration must be positive")
    duration_ratio = hardware_duration_us / software_duration
    relative_deviation = abs(duration_ratio - 1.0)
    ratio_within_tolerance = relative_deviation <= duration_ratio_tolerance
    start_placement_offset_us = float(hardware_start_placement_us) - float(software_start_us)
    placement_within_tolerance = (
        abs(start_placement_offset_us) <= start_placement_tolerance_us
    )
    if not ratio_within_tolerance:
        alignment_status = "duration_ratio_outside_tolerance"
    elif not placement_within_tolerance:
        alignment_status = "start_placement_outside_tolerance"
    else:
        alignment_status = "aligned_within_tolerance"

    row: Dict[str, Any] = dict(extra)
    row.update({
        "semantic_id": semantic_id,
        "dispatch_ordinal": dispatch_ordinal,
        "software_start_us": round(float(software_start_us), 6),
        "software_dur_us": round(software_duration, 6),
        "hardware_start_ns": int(hardware_start_ns),
        "hardware_dur_ns": hardware_duration_ns,
        "hardware_dur_us": round(hardware_duration_us, 6),
        "duration_scale_hw_over_sw": round(duration_ratio, 9),
        "duration_ratio_hw_over_sw": round(duration_ratio, 9),
        "duration_ratio_relative_deviation": round(relative_deviation, 9),
        "duration_ratio_tolerance": duration_ratio_tolerance,
        "duration_ratio_within_tolerance": ratio_within_tolerance,
        "hardware_start_placement_us": round(float(hardware_start_placement_us), 6),
        "start_placement_offset_us": round(start_placement_offset_us, 6),
        "start_placement_tolerance_us": start_placement_tolerance_us,
        "start_placement_within_tolerance": placement_within_tolerance,
        "counter_time_resolution": PER_DISPATCH_COUNTER_RESOLUTION,
        "per_dispatch_aggregate": True,
        "intra_kernel_curve": False,
        "alignment_status": alignment_status,
        "alignment_claim": ratio_within_tolerance and placement_within_tolerance,
    })
    return row


def _alignment_quality(
    alignment_rows: Sequence[Mapping[str, Any]],
    *,
    duration_ratio_tolerance: float,
    start_placement_tolerance_us: float,
    absolute_clock_equality_claimed: bool = False,
) -> Dict[str, Any]:
    rows = list(alignment_rows)
    ratios = [float(row["duration_ratio_hw_over_sw"]) for row in rows]
    offsets = [float(row["start_placement_offset_us"]) for row in rows]
    evidence: List[Dict[str, Any]] = []
    ratio_failure_count = 0
    placement_failure_count = 0
    for row in rows:
        base_evidence = {
            "dispatch_ordinal": row.get("dispatch_ordinal"),
            "semantic_id": row.get("semantic_id"),
            "software_start_us": row.get("software_start_us"),
            "software_dur_us": row.get("software_dur_us"),
            "hardware_start_ns": row.get("hardware_start_ns"),
            "hardware_dur_ns": row.get("hardware_dur_ns"),
            "hardware_dur_us": row.get("hardware_dur_us"),
            "duration_ratio_hw_over_sw": row.get("duration_ratio_hw_over_sw"),
            "duration_ratio_relative_deviation": row.get(
                "duration_ratio_relative_deviation"
            ),
            "start_placement_offset_us": row.get("start_placement_offset_us"),
        }
        if not row.get("duration_ratio_within_tolerance"):
            ratio_failure_count += 1
            evidence.append({
                **base_evidence,
                "kind": "duration_ratio_outside_tolerance",
                "duration_ratio_tolerance": duration_ratio_tolerance,
                "reason": (
                    "hardware/software per-dispatch duration ratio is outside "
                    "1 +/- duration_ratio_tolerance"
                ),
            })
        if not row.get("start_placement_within_tolerance"):
            placement_failure_count += 1
            evidence.append({
                **base_evidence,
                "kind": "start_placement_outside_tolerance",
                "start_placement_tolerance_us": start_placement_tolerance_us,
                "reason": "hardware aggregate start placement differs from the software start",
            })

    if not rows:
        status = "insufficient_evidence"
    elif ratio_failure_count:
        status = "duration_ratio_outside_tolerance"
    elif placement_failure_count:
        status = "start_placement_outside_tolerance"
    else:
        status = "aligned_within_tolerance"
    return {
        "status": status,
        "alignment_claim": status == "aligned_within_tolerance",
        "duration_ratio_hw_over_sw_min": min(ratios) if ratios else None,
        "duration_ratio_hw_over_sw_median": _median(ratios),
        "duration_ratio_hw_over_sw_max": max(ratios) if ratios else None,
        "duration_ratio_acceptable_range": [
            1.0 - duration_ratio_tolerance,
            1.0 + duration_ratio_tolerance,
        ],
        "duration_ratio_tolerance": duration_ratio_tolerance,
        "duration_ratio_outside_tolerance_count": ratio_failure_count,
        "start_placement_offset_us_min": min(offsets) if offsets else None,
        "start_placement_offset_us_max": max(offsets) if offsets else None,
        "start_placement_offset_us_abs_max": (
            max(abs(value) for value in offsets) if offsets else None
        ),
        "start_placement_tolerance_us": start_placement_tolerance_us,
        "start_placement_outside_tolerance_count": placement_failure_count,
        "start_placement_basis": (
            "hardware aggregate display start minus software span start on the "
            "merged Perfetto timeline"
        ),
        "counter_time_resolution": PER_DISPATCH_COUNTER_RESOLUTION,
        "per_dispatch_aggregate": True,
        "intra_kernel_curve": False,
        "absolute_clock_equality_claimed": bool(absolute_clock_equality_claimed),
        "evidence": evidence,
    }


def _decorate_hardware_rows(
    hardware_rows: List[Dict[str, Any]],
    alignment_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_ordinal = {row.get("dispatch_ordinal"): row for row in alignment_rows}
    copied_fields = (
        "software_start_us",
        "software_dur_us",
        "duration_ratio_hw_over_sw",
        "duration_ratio_relative_deviation",
        "duration_ratio_tolerance",
        "duration_ratio_within_tolerance",
        "hardware_start_placement_us",
        "start_placement_offset_us",
        "start_placement_tolerance_us",
        "start_placement_within_tolerance",
        "counter_time_resolution",
        "per_dispatch_aggregate",
        "intra_kernel_curve",
        "alignment_status",
        "alignment_claim",
    )
    for row in hardware_rows:
        alignment = by_ordinal.get(row.get("dispatch_ordinal"))
        if alignment is None:
            raise ValueError(
                "hardware row has no matching per-dispatch alignment row: ordinal={!r}".format(
                    row.get("dispatch_ordinal")
                )
            )
        for field in copied_fields:
            row[field] = alignment.get(field)
    return hardware_rows


def _report_from_written_trace(
    out_path: Path,
    *,
    schema: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> Tuple[Path, Dict[str, Any]]:
    out = Path(out_path)
    trace = read_json_maybe_gz(out)
    if not isinstance(trace, dict) or not isinstance(trace.get("traceEvents"), list):
        raise ValueError("written trace must be {traceEvents:[...]}")
    other = trace.get("otherData")
    if not isinstance(other, dict):
        raise ValueError("written trace has no otherData")
    hardware_rows = other.get("hardware_rows")
    if not isinstance(hardware_rows, list) or not hardware_rows:
        raise ValueError("written trace has no actual hardware_rows")
    hw_merge = other.get("hw_merge")
    if not isinstance(hw_merge, dict):
        raise ValueError("written trace has no hw_merge metadata")
    alignment = hw_merge.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError("written trace has no hw_merge.alignment")
    alignment_rows = alignment.get("dispatch_rows")
    if not isinstance(alignment_rows, list) or not alignment_rows:
        raise ValueError("written trace has no per-dispatch alignment rows")
    quality = alignment.get("quality")
    if not isinstance(quality, dict):
        raise ValueError("written trace has no alignment quality")
    source_trace = hw_merge.get("source_trace")
    if not source_trace:
        raise ValueError("written trace has no hw_merge.source_trace")
    hardware_summary = hw_merge.get("hardware_summary") or {}

    digest = hashlib.sha256()
    with out.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    report: Dict[str, Any] = dict(extra or {})
    report.update({
        "schema": schema,
        "out": str(out.resolve()),
        "bytes": out.stat().st_size,
        "trace_sha256": digest.hexdigest(),
        "trace_events": len(trace["traceEvents"]),
        "x_events": sum(1 for event in trace["traceEvents"] if isinstance(event, dict) and event.get("ph") == "X"),
        "report_source": "final_written_trace",
        "mode": alignment.get("mode"),
        "source_trace": source_trace,
        "trace_processor": hw_merge.get("trace_processor"),
        "hardware_rows": hardware_rows,
        "hardware_row_count": len(hardware_rows),
        "alignment": alignment_rows,
        "alignment_quality": quality,
        "availability": hw_merge.get("availability"),
        "policy": hw_merge.get("policy"),
        "native_tracks": hw_merge.get("native_tracks"),
        "timeline_rebase": hw_merge.get("timeline_rebase"),
        "timestamps": timestamp_report(trace["traceEvents"]),
    })
    report.setdefault("series", hardware_summary.get("series"))
    return out, report


def _merge_single_dispatch_perwave(
    trace: Dict[str, Any],
    events: List[Dict[str, Any]],
    document: Mapping[str, Any],
    dispatch: Mapping[str, Any],
    out_path: Path,
    x_before: int,
    *,
    workload: Any = None,
    correctness: Any = None,
    benchmark: Any = None,
    duration_ratio_tolerance: float = DEFAULT_DURATION_RATIO_TOLERANCE,
    start_placement_tolerance_us: float = DEFAULT_START_PLACEMENT_TOLERANCE_US,
) -> Tuple[Path, Dict[str, Any]]:
    """Attach one aggregate dispatch to a trace containing per-wave events.

    The software window is inferred from every canonical semantic X event.
    All software events remain on their original tracks.  One launch marker is
    added on a dedicated software track and the same-dispatch hardware values
    are drawn on separate hardware tracks.  Starts are aligned by dispatch
    ordinal; absolute clock equality is not claimed.
    """
    ratio_tolerance, placement_tolerance = _alignment_tolerances(
        duration_ratio_tolerance, start_placement_tolerance_us
    )
    semantic_events = _timed_semantic_events(events)
    if not semantic_events:
        raise ValueError("per-wave software trace has no canonical semantic X events")
    name_errors = []
    for event in semantic_events:
        args = event.get("args") or {}
        semantic_id = str(args.get("semantic_id"))
        if event.get("name") != semantic_id or args.get("display_name") != semantic_id:
            name_errors.append({
                "event_name": event.get("name"),
                "display_name": args.get("display_name"),
                "semantic_id": semantic_id,
            })
            if len(name_errors) >= 8:
                break
    if name_errors:
        raise ValueError(
            "semantic event names must equal canonical semantic_id; examples={}".format(name_errors)
        )
    semantic_start_us = min(float(event["ts"]) for event in semantic_events)
    semantic_end_us = max(
        float(event["ts"]) + float(event["dur"]) for event in semantic_events
    )
    other = trace.get("otherData")
    other = other if isinstance(other, dict) else {}
    observation_start_us = _float(other.get("observation_start_us"))
    observation_end_us = _float(other.get("observation_end_us"))
    observation_start_ns = other.get("observation_start_ns")
    try:
        observation_start_ns = int(observation_start_ns)
    except (TypeError, ValueError):
        observation_start_ns = None
    if (
        observation_start_us is not None
        and observation_end_us is not None
        and observation_end_us > observation_start_us
        and observation_start_us <= semantic_start_us
        and observation_end_us >= semantic_end_us
    ):
        start_us, end_us = observation_start_us, observation_end_us
        software_window_source = (
            "decoded FILITE02 observation window including OPEN/CLOSE timer anchors"
        )
    else:
        start_us, end_us = semantic_start_us, semantic_end_us
        software_window_source = "per-wave canonical semantic X-event envelope"
    software_dur_us = end_us - start_us
    if software_dur_us <= 0:
        raise ValueError("per-wave software trace has a non-positive observation window")
    hardware_dur_us = float(dispatch["dur_ns"]) / 1000.0
    duration_scale = hardware_dur_us / software_dur_us
    counts = _semantic_counts(semantic_events)
    wave_count = _physical_wave_count(semantic_events)

    # Shift all software PIDs to >= 100 and pin Hardware Utilization [0-100%] at pid=1, sort_index=-1000 (first in traceEvents -> upid=1)
    SW_PID_SHIFT = 100
    for ev_sw in events:
        if isinstance(ev_sw.get("pid"), int):
            ev_sw["pid"] = int(ev_sw["pid"]) + SW_PID_SHIFT
            if ev_sw.get("ph") == "M" and ev_sw.get("name") == "process_sort_index":
                args_m = ev_sw.get("args") or {}
                if isinstance(args_m.get("sort_index"), (int, float)):
                    args_m["sort_index"] = int(args_m["sort_index"]) + SW_PID_SHIFT

    hardware_pid = 1
    launch_pid = 2
    top_events: List[Dict[str, Any]] = [
        {"ph": "M", "name": "process_name", "pid": hardware_pid, "tid": 0,
         "args": {"name": "00 · 硬件信息与利用率 [0–100%] · AMD MI308X rocprof-sys ROCm PM (same dispatch)"}},
        {"ph": "M", "name": "process_sort_index", "pid": hardware_pid, "tid": 0,
         "args": {"sort_index": -1000}},
        {"ph": "M", "name": "thread_name", "pid": hardware_pid, "tid": 1,
         "args": {"name": "GPU 硬件窗口 · same dispatch"}},
        {"ph": "M", "name": "thread_sort_index", "pid": hardware_pid, "tid": 1,
         "args": {"sort_index": 1}},
        {"ph": "X", "name": "gemm [hardware dispatch aggregate]",
         "cat": "hardware.dispatch", "ts": round(start_us, 6),
         "dur": round(hardware_dur_us, 6), "pid": hardware_pid, "tid": 1,
         "args": {
             "semantic_id": "gemm",
             "display_name": "gemm",
             "source": "rocprof-sys kernel_dispatch",
             "dispatch_ordinal": dispatch.get("ordinal"),
             "kernel_name": dispatch.get("name"),
             "software_window_us": round(software_dur_us, 6),
             "duration_scale_hw_over_sw": round(duration_scale, 9),
             "counter_time_resolution": "per_dispatch_aggregate",
             "absolute_clock_equality_claimed": False,
         }},
        {"ph": "M", "name": "process_name", "pid": launch_pid, "tid": 0,
         "args": {"name": "01 · 软件总体包络 | kernel launch"}},
        {"ph": "M", "name": "process_sort_index", "pid": launch_pid, "tid": 0,
         "args": {"sort_index": -500}},
        {"ph": "M", "name": "thread_name", "pid": launch_pid, "tid": 1,
         "args": {"name": "same dispatch"}},
        {"ph": "X", "name": "gemm", "cat": "software.kernel_launch",
         "ts": round(start_us, 6), "dur": round(software_dur_us, 6),
         "pid": launch_pid, "tid": 1,
         "args": {
             "semantic_id": "gemm",
             "display_name": "gemm",
             "source": "weaveprofiler per-wave semantic observation window",
             "dispatch_ordinal": dispatch.get("ordinal"),
             "software_clock": "s_memrealtime decoded into Perfetto microseconds",
             "hardware_clock": "rocprof-sys Perfetto protobuf nanoseconds",
             "clock_mapping": "dispatch-ordinal start alignment; absolute clock equality not claimed",
             "software_observed_count": len(semantic_events),
             "physical_wave_count": wave_count,
         }},
    ]

    counter_events = 0
    hardware_rows = []
    for index, metric in enumerate(document.get("series") or []):
        points = metric.get("points") or []
        if not points:
            continue
        chosen = None
        for point in points:
            if len(point) >= 3 and int(point[0]) == int(dispatch["start_ns"]):
                chosen = point
                break
        if chosen is None:
            chosen = points[0]
        label = str(metric["timeline_label"])
        _add_hardware_track(top_events, index, label, pid=hardware_pid)
        counter_events += _add_series_events(
            top_events,
            {"timeline_label": label, "metric_value": chosen[2]},
            start_us,
            start_us + hardware_dur_us,
            track_id=100 + index,
            pid=hardware_pid,
        )
        hardware_rows.append({
            "semantic_id": metric.get("semantic_id"),
            "timeline_label": label,
            "dispatch_ordinal": dispatch.get("ordinal"),
            "hardware_start_ns": chosen[0],
            "hardware_dur_ns": chosen[1],
            "value": chosen[2],
            "track_id": 100 + index,
        })
    if not counter_events:
        raise ValueError("no same-dispatch hardware series were available")
    events[:] = top_events + events

    rebase = rebase_timeline_nonnegative(events)
    _attach_common_metadata(
        trace, document, workload=workload,
        correctness=correctness, benchmark=benchmark,
    )
    alignment_row = _alignment_row(
        semantic_id="gemm",
        dispatch_ordinal=dispatch.get("ordinal"),
        software_start_us=start_us,
        software_dur_us=software_dur_us,
        hardware_start_ns=dispatch["start_ns"],
        hardware_dur_ns=dispatch["dur_ns"],
        hardware_start_placement_us=start_us,
        duration_ratio_tolerance=ratio_tolerance,
        start_placement_tolerance_us=placement_tolerance,
        software_duration_source=(
            software_window_source
            + "; aggregate dispatch duration proxy, not hardware elapsed-time equality"
        ),
        software_semantic_start_us=round(semantic_start_us, 6),
        software_semantic_end_us=round(semantic_end_us, 6),
        software_observation_start_ns=observation_start_ns,
        hardware_source_start_ns=int(dispatch["start_ns"]),
        source_clock_offset_ns=(
            int(dispatch["start_ns"]) - observation_start_ns
            if observation_start_ns is not None else None
        ),
        start_placement_is_integration_anchor=True,
        start_placement_measured=False,
        software_observed_count=len(semantic_events),
        software_physical_wave_count=wave_count,
        software_semantic_counts=counts,
        software_clock="weaveprofiler s_memrealtime decoded to Perfetto microseconds",
        hardware_clock="rocprof-sys Perfetto protobuf nanoseconds",
        clock_mapping=(
            "single-dispatch ordinal alignment from start; absolute equality not claimed"
        ),
    )
    quality = _alignment_quality(
        [alignment_row],
        duration_ratio_tolerance=ratio_tolerance,
        start_placement_tolerance_us=placement_tolerance,
        absolute_clock_equality_claimed=False,
    )
    quality["start_placement_basis"] = (
        "integration anchor: hardware aggregate display start is placed at the "
        "software observation start; this offset is chosen for display and is not "
        "a measured absolute-clock pairing"
    )
    quality["start_placement_measured"] = False
    hardware_rows = _decorate_hardware_rows(hardware_rows, [alignment_row])
    trace["otherData"]["hw_merge"]["alignment"] = {
        **dict(document.get("alignment") or {}),
        "mode": "existing_perwave_trace_single_dispatch_start_alignment",
        "dispatch_rows": [alignment_row],
        "quality": quality,
        "software_clock": alignment_row["software_clock"],
        "hardware_clock": alignment_row["hardware_clock"],
        "absolute_clock_equality_claimed": False,
        "counter_time_resolution": PER_DISPATCH_COUNTER_RESOLUTION,
        "per_dispatch_aggregate": True,
        "intra_kernel_curve": False,
    }
    trace["otherData"]["hw_merge"]["timeline_rebase"] = rebase
    trace["otherData"]["hardware_rows"] = hardware_rows
    trace["otherData"]["software_semantics"] = {
        "event_counts": counts,
        "physical_wave_count": wave_count,
        "observed_span_us": round(software_dur_us, 6),
        "preserved_event_count": len(semantic_events),
    }
    write_json_gz(Path(out_path), trace)
    x_after = sum(1 for event in events if isinstance(event, dict) and event.get("ph") == "X")
    return _report_from_written_trace(Path(out_path), schema="rocprof-sys-sw-hw-build/v3", extra={
        "software_x_events_preserved": x_before,
        "added_alignment_x_events": x_after - x_before,
        "software_x_events_total": x_after,
        "software_semantic_events": len(semantic_events),
        "software_semantic_counts": counts,
        "semantic_name_policy": "X.name == args.display_name == args.semantic_id",
        "physical_wave_count": wave_count,
        "counter_events": counter_events,
    })


def _attach_common_metadata(trace: Dict[str, Any], document: Mapping[str, Any], *,
                            workload: Any = None, correctness: Any = None,
                            benchmark: Any = None) -> None:
    other = trace.get("otherData")
    if not isinstance(other, dict):
        other = {}
    other["target"] = {"card": "mi308x", "platform": "amd", "arch": "gfx942"}
    other["hw_merge"] = {
        "schema": SCHEMA,
        "collector": "rocprof-sys",
        "source_trace": document.get("source_trace"),
        "trace_processor": document.get("trace_processor"),
        "alignment": document.get("alignment"),
        "kernel_selection": document.get("kernel_selection"),
        "native_tracks": document.get("native_tracks"),
        "availability": document.get("availability"),
        "policy": document.get("policy"),
        "hardware_summary": {
            "series": document.get("series"),
            "dispatches": document.get("dispatches"),
        },
        "display_note": (
            "counter tracks are per-dispatch raw-contract-derived samples; a zero boundary "
            "is a display break after a measured kernel, never missing-evidence zero fill"
        ),
    }
    other["rocprof_sys_hardware"] = {
        "series": document.get("series"),
        "native_tracks": document.get("native_tracks"),
    }
    if workload is not None:
        other["workload"] = workload
    if correctness is not None:
        other["correctness"] = correctness
    if benchmark is not None:
        other["benchmark"] = benchmark
    trace["otherData"] = other


def build_from_document(document: Mapping[str, Any], out_path: Path, *,
                        workload: Any = None, correctness: Any = None,
                        benchmark: Any = None,
                        duration_ratio_tolerance: float = DEFAULT_DURATION_RATIO_TOLERANCE,
                        start_placement_tolerance_us: float = DEFAULT_START_PLACEMENT_TOLERANCE_US,
                        ) -> Tuple[Path, Dict[str, Any]]:
    ratio_tolerance, placement_tolerance = _alignment_tolerances(
        duration_ratio_tolerance, start_placement_tolerance_us
    )
    dispatches = document.get("dispatches")
    series = document.get("series")
    if not isinstance(dispatches, list) or not dispatches:
        raise ValueError("document has no dispatches")
    if not isinstance(series, list):
        raise ValueError("document has no series")
    base_ns = min(int(item["start_ns"]) for item in dispatches)
    end_ns = max(int(item["start_ns"]) + int(item["dur_ns"]) for item in dispatches)

    def ns_to_us(value: int) -> float:
        return (int(value) - base_ns) / 1000.0

    events: List[Dict[str, Any]] = [
        {"ph": "M", "name": "process_name", "pid": 1, "tid": 0,
         "args": {"name": "00 · 硬件信息与利用率 [0–100%] · rocprof-sys ROCm PM"}},
        {"ph": "M", "name": "process_sort_index", "pid": 1, "tid": 0,
         "args": {"sort_index": -1000}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 1,
         "args": {"name": "hardware | same run + same dispatch counters"}},
        {"ph": "M", "name": "thread_sort_index", "pid": 1, "tid": 1,
         "args": {"sort_index": 5}},
        {"ph": "M", "name": "process_name", "pid": 100, "tid": 0,
         "args": {"name": "WeaveProfiler"}},
        {"ph": "M", "name": "process_sort_index", "pid": 100, "tid": 0,
         "args": {"sort_index": 100}},
        {"ph": "M", "name": "thread_name", "pid": 100, "tid": 1,
         "args": {"name": "software | instrumented Triton GEMM"}},
        {"ph": "M", "name": "thread_sort_index", "pid": 100, "tid": 1,
         "args": {"sort_index": 10}},
        {"ph": "X", "name": "gemm_instrumented_sequence", "cat": "software.phase",
         "ts": ns_to_us(base_ns), "dur": (end_ns - base_ns) / 1000.0,
         "pid": 100, "tid": 1,
         "args": {"semantic_id": "gemm", "display_name": "gemm",
                  "dispatch_count": len(dispatches)}},
    ]
    for dispatch in dispatches:
        start_us = ns_to_us(int(dispatch["start_ns"]))
        end_us = start_us + int(dispatch["dur_ns"]) / 1000.0
        events.append({
            "ph": "X",
            "name": "gemm",
            "cat": "software.kernel",
            "ts": round(start_us, 6),
            "dur": round(int(dispatch["dur_ns"]) / 1000.0, 6),
            "pid": 100,
            "tid": 1,
            "args": {
                "semantic_id": "gemm",
                "display_name": "gemm",
                "kernel_name": dispatch.get("name"),
                "dispatch_ordinal": dispatch.get("ordinal"),
                "source": "rocprof-sys kernel_dispatch",
            },
        })

    dispatch_by_window: Dict[Tuple[int, int], Mapping[str, Any]] = {}
    alignment_rows: List[Dict[str, Any]] = []
    for dispatch in dispatches:
        start_ns = int(dispatch["start_ns"])
        dispatch_dur_ns = int(dispatch["dur_ns"])
        start_us = ns_to_us(start_ns)
        dispatch_by_window.setdefault((start_ns, dispatch_dur_ns), dispatch)
        alignment_rows.append(_alignment_row(
            semantic_id="gemm",
            dispatch_ordinal=dispatch.get("ordinal"),
            software_start_us=start_us,
            software_dur_us=dispatch_dur_ns / 1000.0,
            hardware_start_ns=start_ns,
            hardware_dur_ns=dispatch_dur_ns,
            hardware_start_placement_us=start_us,
            duration_ratio_tolerance=ratio_tolerance,
            start_placement_tolerance_us=placement_tolerance,
            software_duration_source=(
                "rocprof-sys per-dispatch duration reused for the synthesized software span"
            ),
            software_clock="rocprof-sys Perfetto protobuf microseconds",
            hardware_clock="rocprof-sys Perfetto protobuf nanoseconds",
            clock_mapping=(
                "same-dispatch timestamp alignment; isochronous duration evidence"
            ),
        ))

    counter_events = 0
    hardware_rows: List[Dict[str, Any]] = []
    for metric in series:
        points = metric.get("points") or []
        if not points:
            continue
        # Emit the actual per-dispatch values, not a flat median.
        for start_ns, dur_ns, value in points:
            start_us = ns_to_us(int(start_ns))
            end_us = start_us + int(dur_ns) / 1000.0
            counter_events += _add_series_events(
                events,
                {"timeline_label": metric["timeline_label"], "metric_value": value},
                start_us,
                end_us,
            )
            aligned_dispatch = dispatch_by_window.get((int(start_ns), int(dur_ns)))
            if aligned_dispatch is None:
                raise ValueError(
                    "hardware series point has no matching dispatch window: {!r}".format(
                        [start_ns, dur_ns]
                    )
                )
            hardware_rows.append({
                "semantic_id": metric.get("semantic_id"),
                "timeline_label": metric["timeline_label"],
                "dispatch_ordinal": aligned_dispatch.get("ordinal"),
                "hardware_start_ns": int(start_ns),
                "hardware_dur_ns": int(dur_ns),
                "value": value,
                "track_id": 1,
            })
    if not counter_events:
        raise ValueError("no measurable hardware series were available")
    hardware_rows = _decorate_hardware_rows(hardware_rows, alignment_rows)
    quality = _alignment_quality(
        alignment_rows,
        duration_ratio_tolerance=ratio_tolerance,
        start_placement_tolerance_us=placement_tolerance,
        absolute_clock_equality_claimed=False,
    )

    trace: Dict[str, Any] = {
        "schemaVersion": 1,
        "traceEvents": events,
    }
    _attach_common_metadata(trace, document, workload=workload,
                            correctness=correctness, benchmark=benchmark)
    rebase = rebase_timeline_nonnegative(events)
    trace["otherData"]["hw_merge"]["timeline_rebase"] = rebase
    trace["otherData"]["hw_merge"]["alignment"] = {
        **dict(document.get("alignment") or {}),
        "mode": "direct_proto_dispatch_timestamp_alignment",
        "dispatch_rows": alignment_rows,
        "quality": quality,
        "absolute_clock_equality_claimed": False,
        "counter_time_resolution": PER_DISPATCH_COUNTER_RESOLUTION,
        "per_dispatch_aggregate": True,
        "intra_kernel_curve": False,
    }
    trace["otherData"]["hardware_rows"] = hardware_rows
    out_path = Path(out_path)
    if out_path.name.startswith("trace__"):
        match = CANONICAL_TRACE_RE.match(out_path.name)
        if match is None:
            raise ValueError("canonical output name does not match WeaveProfiler trace contract: {}".format(out_path.name))
        if match.group("card") != "mi308x" or match.group("bundle") != "sw_hw":
            raise ValueError("output must use card-mi308x and __sw_hw.perfetto.json.gz")
    write_json_gz(out_path, trace)
    return _report_from_written_trace(out_path, schema="rocprof-sys-sw-hw-build/v2", extra={
        "software_x_events": len(dispatches),
        "counter_events": counter_events,
        "series": [
            {key: metric.get(key) for key in (
                "semantic_id", "timeline_label", "evidence_status", "point_count",
                "coverage", "min", "median", "mean", "max", "method")}
            for metric in series
        ],
    })


def merge(trace_path: Path, document: Mapping[str, Any], out_path: Path, *,
          workload: Any = None, correctness: Any = None,
          benchmark: Any = None,
          duration_ratio_tolerance: float = DEFAULT_DURATION_RATIO_TOLERANCE,
          start_placement_tolerance_us: float = DEFAULT_START_PLACEMENT_TOLERANCE_US,
          ) -> Tuple[Path, Dict[str, Any]]:
    """Merge extracted counters into an existing canonical SW trace.

    A launch-level software trace maps one ``gemm``/``mma`` span per dispatch.
    A per-wave trace may contain many semantic spans for one physical dispatch;
    that case preserves every span, adds one launch marker, and draws the
    same-dispatch counters on separate hardware tracks.  Multi-dispatch inputs
    without one launch span per dispatch fail closed.
    """
    ratio_tolerance, placement_tolerance = _alignment_tolerances(
        duration_ratio_tolerance, start_placement_tolerance_us
    )
    trace = read_json_maybe_gz(Path(trace_path))
    if not isinstance(trace, dict) or not isinstance(trace.get("traceEvents"), list):
        raise ValueError("input trace must be {traceEvents:[...]}")
    events = trace["traceEvents"]
    x_before = sum(1 for event in events if isinstance(event, dict) and event.get("ph") == "X")
    semantic_events = _timed_semantic_events(events)
    launch_spans = [
        event for event in semantic_events
        if event.get("args", {}).get("semantic_id") == "gemm"
    ]
    dispatches = document.get("dispatches") or []
    if not dispatches:
        raise ValueError("document has no dispatches")

    if len(launch_spans) == len(dispatches):
        software = launch_spans
    elif len(dispatches) == 1:
        return _merge_single_dispatch_perwave(
            trace, events, document, dispatches[0], Path(out_path), x_before,
            workload=workload, correctness=correctness, benchmark=benchmark,
            duration_ratio_tolerance=ratio_tolerance,
            start_placement_tolerance_us=placement_tolerance,
        )
    else:
        raise ValueError(
            "software launch span count {} != rocprof-sys dispatch count {}".format(
                len(launch_spans), len(dispatches)))

    counter_events = 0
    alignment_rows = [
        _alignment_row(
            semantic_id=sw_event["args"].get("semantic_id"),
            dispatch_ordinal=dispatch.get("ordinal"),
            software_start_us=float(sw_event["ts"]),
            software_dur_us=float(sw_event["dur"]),
            hardware_start_ns=dispatch["start_ns"],
            hardware_dur_ns=dispatch["dur_ns"],
            hardware_start_placement_us=float(sw_event["ts"]),
            duration_ratio_tolerance=ratio_tolerance,
            start_placement_tolerance_us=placement_tolerance,
            software_duration_source="existing software trace X-event duration",
            software_clock="existing software Perfetto timeline microseconds",
            hardware_clock="rocprof-sys Perfetto protobuf nanoseconds",
            clock_mapping=(
                "dispatch-ordinal start placement; absolute clock equality not claimed"
            ),
            duration_scale=round(
                (float(dispatch["dur_ns"]) / 1000.0) / float(sw_event["dur"]), 9
            ),
        )
        for sw_event, dispatch in zip(software, dispatches)
    ]
    SW_PID_SHIFT = 100
    for ev_sw in events:
        if isinstance(ev_sw.get("pid"), int):
            ev_sw["pid"] = int(ev_sw["pid"]) + SW_PID_SHIFT
            if ev_sw.get("ph") == "M" and ev_sw.get("name") == "process_sort_index":
                args_m = ev_sw.get("args") or {}
                if isinstance(args_m.get("sort_index"), (int, float)):
                    args_m["sort_index"] = int(args_m["sort_index"]) + SW_PID_SHIFT

    hardware_rows: List[Dict[str, Any]] = []
    hardware_track_count = 0
    hardware_pid = 1
    hw_events: List[Dict[str, Any]] = [
        {"ph": "M", "name": "process_name", "pid": hardware_pid, "tid": 0,
         "args": {"name": "00 · 硬件信息与利用率 [0–100%] · rocprof-sys ROCm PM | same dispatch"}},
        {"ph": "M", "name": "process_sort_index", "pid": hardware_pid, "tid": 0,
         "args": {"sort_index": -1000}},
    ]
    for metric in document.get("series") or []:
        points = metric.get("points") or []
        if not points:
            continue
        _add_hardware_track(hw_events, hardware_track_count,
                            str(metric["timeline_label"]), pid=hardware_pid)
        track_id = 100 + hardware_track_count
        hardware_track_count += 1
        for point_position, (sw_event, dispatch) in enumerate(zip(software, dispatches)):
            start_us = float(sw_event["ts"])
            end_us = start_us + float(sw_event["dur"])
            if len(points) == len(dispatches):
                point_index = point_position
                ordinal = dispatch.get("ordinal")
                if isinstance(ordinal, int) and 1 <= ordinal <= len(points):
                    point_index = ordinal - 1
                point = points[point_index]
                value = point[2]
                counter_events += _add_series_events(
                    hw_events,
                    {"timeline_label": metric["timeline_label"], "metric_value": value},
                    start_us,
                    end_us,
                    track_id=track_id,
                    pid=hardware_pid,
                )
                hardware_rows.append({
                    "semantic_id": metric.get("semantic_id"),
                    "timeline_label": metric["timeline_label"],
                    "dispatch_ordinal": dispatch.get("ordinal"),
                    "hardware_start_ns": point[0],
                    "hardware_dur_ns": point[1],
                    "value": value,
                    "track_id": track_id,
                })
    if not counter_events:
        raise ValueError("no same-dispatch hardware series were available")
    hardware_rows = _decorate_hardware_rows(hardware_rows, alignment_rows)
    events[:] = hw_events + events
    rebase = rebase_timeline_nonnegative(events)
    x_after = sum(1 for event in events if isinstance(event, dict) and event.get("ph") == "X")
    if x_after != x_before:
        raise ValueError("launch-level merge changed software X-event count: {} -> {}".format(
            x_before, x_after))
    _attach_common_metadata(
        trace, document, workload=workload,
        correctness=correctness, benchmark=benchmark,
    )
    quality = _alignment_quality(
        alignment_rows,
        duration_ratio_tolerance=ratio_tolerance,
        start_placement_tolerance_us=placement_tolerance,
        absolute_clock_equality_claimed=False,
    )
    quality["start_placement_basis"] = (
        "integration anchor: hardware aggregate display start is placed at the "
        "software observation start; this offset is chosen for display and is not "
        "a measured absolute-clock pairing"
    )
    quality["start_placement_measured"] = False
    trace["otherData"]["hw_merge"]["alignment"] = {
        **dict(document.get("alignment") or {}),
        "mode": "existing_software_trace_ordinal_mapping",
        "dispatch_rows": alignment_rows,
        "quality": quality,
        "absolute_clock_equality_claimed": False,
        "counter_time_resolution": PER_DISPATCH_COUNTER_RESOLUTION,
        "per_dispatch_aggregate": True,
        "intra_kernel_curve": False,
    }
    trace["otherData"]["hw_merge"]["timeline_rebase"] = rebase
    trace["otherData"]["hardware_rows"] = hardware_rows
    write_json_gz(Path(out_path), trace)
    return _report_from_written_trace(Path(out_path), schema="rocprof-sys-sw-hw-build/v3", extra={
        "software_x_events": x_before,
        "counter_events": counter_events,
    })


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocprof-sys-proto", required=True)
    parser.add_argument("--trace-processor", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trace", default=None,
                        help="optional existing SW trace; omit to build directly from the same proto")
    parser.add_argument("--extracted-json", default=None)
    parser.add_argument("--kernel-name-contains", default=DEFAULT_KERNEL_NAME_CONTAINS)
    parser.add_argument("--kernel-ordinal", type=int, default=None)
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timestamp-tolerance-ns", type=int, default=5000)
    parser.add_argument("--cu-count", type=int, default=80)
    parser.add_argument("--simd-per-cu", type=int, default=4)
    parser.add_argument("--xcc-count", type=int, default=4)
    parser.add_argument("--hbm-peak-gbps", type=float, default=3684.0,
                        help="measured HBM peak; use a negative value to mark memory unavailable")
    parser.add_argument("--hbm-request-bytes", type=int, default=64)
    parser.add_argument("--workload-json", default=None)
    parser.add_argument("--correctness-json", default=None)
    parser.add_argument("--benchmark-json", default=None)
    parser.add_argument("--duration-ratio-tolerance", type=float,
                        default=DEFAULT_DURATION_RATIO_TOLERANCE,
                        help="allowed relative deviation of hardware/software dispatch duration from 1.0")
    parser.add_argument("--start-placement-tolerance-us", type=float,
                        default=DEFAULT_START_PLACEMENT_TOLERANCE_US,
                        help="allowed hardware aggregate start-placement offset in microseconds")
    args = parser.parse_args(argv)
    try:
        hbm_peak = None if args.hbm_peak_gbps < 0 else args.hbm_peak_gbps
        document = extract_dispatches(
            Path(args.trace_processor),
            Path(args.rocprof_sys_proto),
            args.kernel_name_contains,
            kernel_ordinal=args.kernel_ordinal,
            skip_first=args.skip_first,
            limit=args.limit,
            timestamp_tolerance_ns=args.timestamp_tolerance_ns,
            cu_count=args.cu_count,
            simd_per_cu=args.simd_per_cu,
            xcc_count=args.xcc_count,
            hbm_peak_gbps=hbm_peak,
            hbm_request_bytes=args.hbm_request_bytes,
        )
        if args.extracted_json:
            extracted = Path(args.extracted_json).expanduser().resolve()
            extracted.parent.mkdir(parents=True, exist_ok=True)
            extracted.write_text(
                json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        workload = _load_optional_json(args.workload_json)
        correctness = _load_optional_json(args.correctness_json)
        benchmark = _load_optional_json(args.benchmark_json)
        if args.trace:
            out, report = merge(
                Path(args.trace), document, Path(args.out),
                workload=workload, correctness=correctness, benchmark=benchmark,
                duration_ratio_tolerance=args.duration_ratio_tolerance,
                start_placement_tolerance_us=args.start_placement_tolerance_us,
            )
        else:
            out, report = build_from_document(
                document, Path(args.out), workload=workload,
                correctness=correctness, benchmark=benchmark,
                duration_ratio_tolerance=args.duration_ratio_tolerance,
                start_placement_tolerance_us=args.start_placement_tolerance_us,
            )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("[fatal] {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
