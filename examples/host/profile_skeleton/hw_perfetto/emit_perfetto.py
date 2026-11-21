#!/usr/bin/env python3
"""emit_perfetto.py (E21 + E22) — decoded SW events (+ optional HW samples + alignment)
-> a single .perfetto.json.gz with task/phase granularity.

FULLY IMPLEMENTED generic logic. Self-contained on purpose: unlike the skill's
examples/host/build_perfetto.py this file does NOT import the skill's
weaveprof/pipeline/scan/layout_contract.py, so a copied profile/ tree runs standalone.

What is generic (kept here, do not rewrite):
  * one Perfetto *process* per (rank, CTA/block): pid = PID_BASE + index,
    process_name = "rank+CTA {rank}:{block} / {smid}"                     (R3 layout)
  * one *thread* per observer track/warp: tid = index+1,
    thread_name = "warp{warp} (track{track})"                            (R3 lane)
  * ph='X' interval per event, ts/dur in microseconds, args carry
    task_id/task/layer/parent_task_id/depth/track/block/rank/physical_smid
  * per-lane overlap check: same lane may only nest or be disjoint; any
    PARTIAL overlap is a hard FAIL (rendering contract R3)
  * HW PM counters as ph='C' with pid strictly < every software pid (R6)
  * shift-only HW alignment (scale=1) + single global timeline rebase so every
    displayed ts >= 0 (never clamp/drop/rescale individual events)
  * displayTimeUnit='ns', otherData carries provenance + the oversize policy

The only project-specific input is data, not code:
  * the ABI (device_record/abi/), which fixes task/phase layering
  * OPTIONAL: a warp layout (track -> physical warp ids). The generic default
    maps each track to a single lane (warp_id == track_id); override via the
    contract topology.observers or the marked PROJECT HOOK below only if your
    kernel runs several physical warps behind one observer track.

CLI (contract-driven, matches the unified stage command):
  python3 -B emit_perfetto.py --contract <lock> --out work/candidate.perfetto.json.gz
Explicit inputs (used by selftests / manual runs):
  python3 -B emit_perfetto.py --events events_rank0.json [--events ...] \
      [--hw-metrics hw/hw_metrics.json] [--align work/alignment.json] \
      --out work/candidate.perfetto.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROFILE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROFILE_ROOT / "hw_perfetto"))
from align_sw_hw import compute_alignment, load_json_maybe_gz  # noqa: E402

PID_BASE = 1000
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # contract artifacts cap: 10485760 bytes
HW_NOTE = ("counter is a device-wide (device-wide) GPU PM sampling that continuously "
           "covers the selected kernel window; a Perfetto connector line does NOT mean "
           "the value stayed non-zero and it cannot be attributed to any single CTA, "
           "warp or phase. Lane is placed above all software tracks per render contract R6.")


class EmitError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# timeline rebase (verbatim policy from merge_hw_into_perfetto.py)
# --------------------------------------------------------------------------- #
def timestamp_report(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    values: List[float] = []
    invalid = 0
    negative = 0
    for event in events:
        if "ts" not in event:
            continue
        value = event.get("ts")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            invalid += 1
            continue
        numeric = float(value)
        values.append(numeric)
        if numeric < 0.0:
            negative += 1
    return {"timed_event_count": len(values), "min_ts_us": min(values) if values else None,
            "max_ts_us": max(values) if values else None, "invalid_count": invalid,
            "negative_count": negative}


def rebase_timeline_nonnegative(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Translate the whole display timeline once; never clamp/drop individual events."""
    before = timestamp_report(events)
    if before["invalid_count"]:
        raise EmitError("invalid (nan/inf/non-numeric) ts present before rebase")
    pre_min = before["min_ts_us"]
    offset_us = round(max(0.0, -float(pre_min)), 9) if pre_min is not None else 0.0
    shifted = 0
    if offset_us:
        for event in events:
            value = event.get("ts")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                translated = round(float(value) + offset_us, 9)
                event["ts"] = 0.0 if translated == 0.0 else translated
                shifted += 1
    after = timestamp_report(events)
    return {"schema": "timeline-rebase/v1", "applied": bool(offset_us),
            "method": "single global translation of every numeric ts; no clamp/drop/rescale",
            "offset_us": offset_us, "pre_min_ts_us": pre_min, "post_min_ts_us": after["min_ts_us"],
            "shifted_event_count": shifted, "relative_timing_preserved": True,
            "duration_fields_unchanged": True, "post_negative_count": after["negative_count"]}


# --------------------------------------------------------------------------- #
# normalize a decode_profile.py document into a builder doc
# --------------------------------------------------------------------------- #
def compute_shift_ns(decoded: Dict[str, Any]) -> int:
    """Software zero = per-rank entry rendezvous *release*, else earliest start."""
    anchors = decoded.get("anchors") or {}
    rendezvous = anchors.get("entry_rendezvous") or {}
    glob = rendezvous.get("global") or {}
    release = glob.get("entry_sync_end_ns")
    if isinstance(release, (int, float)) and not isinstance(release, bool):
        return int(release)
    starts = [int(e["start_ns"]) for e in decoded.get("events", []) if "start_ns" in e]
    return min(starts) if starts else 0


def normalize_decoded(decoded: Dict[str, Any], rank: Optional[int], device: Optional[int],
                      shift_ns: Optional[int], path: Optional[str]) -> Dict[str, Any]:
    meta = decoded.get("metadata") or {}
    rank = rank if rank is not None else meta.get("rank")
    device = device if device is not None else meta.get("device")
    raw_bm = decoded.get("block_map", [])
    if isinstance(raw_bm, list):
        block_map = {index: dict(info) for index, info in enumerate(raw_bm)}
    else:
        block_map = {int(key): dict(value) for key, value in raw_bm.items()}
    if shift_ns is None:
        shift_ns = compute_shift_ns(decoded)
    return {"rank": int(rank) if rank is not None else 0, "device": device,
            "block_map": block_map, "events": list(decoded.get("events", [])),
            "anchors": decoded.get("anchors"), "shift_ns": int(shift_ns),
            "path": path or "events"}


def smid_label(info: Optional[Dict[str, Any]]) -> str:
    if not info:
        return "smid?"
    smid = info.get("physical_smid")
    if smid is None:
        return "smid?"
    return "smid{}".format(smid) if info.get("written", True) else "smid{}(unwritten)".format(smid)


def resolve_warp_list(warp_layout: Optional[Dict[str, List[int]]], track: int,
                      lane_label: str) -> List[int]:
    if warp_layout:
        for key in (lane_label, str(track), "track{}".format(track)):
            if key in warp_layout and warp_layout[key]:
                return list(warp_layout[key])
    # ===== PROJECT HOOK: warp layout — VERIFY, do not assume =====
    # Identity fallback: warp_id == track_id, i.e. exactly one physical warp behind each
    # observer track. This is a CLAIM about your kernel, not a neutral default: if several
    # physical warps report through one track, the trace labels their merged timeline
    # "warp<track>" and a reader attributes one warp's work to all of them.
    # How to verify (once, against the kernel source): count the warps that write records
    # for each track id in the device code. If any track is written by more than one
    # physical warp, pass --warp-layout / contract topology.observers with the real mapping.
    # Whichever path was taken is recorded in otherData.warp_layout so the assumption is
    # visible in the delivered trace instead of being invisible here.
    return [track]


# --------------------------------------------------------------------------- #
# software trace (ph='X') builder — standalone re-implementation of the R3 layout
# --------------------------------------------------------------------------- #
def build_sw_trace(docs: List[Dict[str, Any]],
                   warp_layout: Optional[Dict[str, List[int]]] = None
                   ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    trace_events: List[Dict[str, Any]] = []
    improper_overlaps: List[Dict[str, Any]] = []
    process_index = 0
    num_x = 0
    global_start: Optional[float] = None
    global_end: Optional[float] = None
    physical_sm_on_software_lane = 0  # R6 guard: sw lanes must not be pure sm ids

    ordered = sorted(docs, key=lambda item: (item["rank"], str(item["path"])))
    for doc in ordered:
        shift_ns = doc["shift_ns"]
        device_label = "dev?" if doc["device"] is None else "dev{}".format(doc["device"])
        lanes: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for event in doc["events"]:
            lanes.setdefault((int(event["block"]), int(event["track"])), []).append(event)
        block_map = doc["block_map"]

        for block in sorted({b for b, _t in lanes}):
            pid = PID_BASE + process_index
            info = block_map.get(block)
            smid = None if info is None else info.get("physical_smid")
            trace_events.append({"name": "process_name", "ph": "M", "pid": pid, "tid": 0,
                                 "args": {"name": "rank+CTA {}:{} / {}".format(doc["rank"], block, smid_label(info))}})
            trace_events.append({"name": "process_sort_index", "ph": "M", "pid": pid, "tid": 0,
                                 "args": {"sort_index": process_index}})
            trace_events.append({"name": "process_labels", "ph": "M", "pid": pid, "tid": 0,
                                 "args": {"labels": "{} rank{} CTA{}".format(device_label, doc["rank"], block)}})
            process_index += 1

            threads: List[Dict[str, Any]] = []
            for track in sorted(t for b, t in lanes if b == block):
                lane_events = sorted(lanes[(block, track)],
                                     key=lambda e: (e["start_ns"], -e["dur_ns"],
                                                    e["task_id"] if e.get("task_id") is not None else -1))
                lane_label = lane_events[0].get("track_label") or lane_events[0].get("role") or "track{}".format(track)
                warp_list = resolve_warp_list(warp_layout, track, lane_label)
                threads.append({"track": track, "lane_label": lane_label,
                                "warp_id": warp_list[0], "events": lane_events})
            threads.sort(key=lambda item: (item["warp_id"] if item["warp_id"] is not None else 1 << 30, item["track"]))

            for tid_index, thread in enumerate(threads):
                tid = tid_index + 1
                track = thread["track"]
                warp_id = thread["warp_id"]
                thread_name = ("warp{} (track{})".format(warp_id, track) if warp_id is not None
                               else "track{}".format(track))
                trace_events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid,
                                     "args": {"name": thread_name}})
                trace_events.append({"name": "thread_sort_index", "ph": "M", "pid": pid, "tid": tid,
                                     "args": {"sort_index": tid_index}})

                open_stack: List[Tuple[float, str]] = []
                for event in thread["events"]:
                    # WHY no default: `layer` decides whether an event becomes a span at
                    # all. The old code read event.get("layer", "task"), so an event that
                    # lost its layer somewhere upstream was rendered as an operator. A
                    # shipped delivery rendered the whole-core entry anchor that way: one
                    # full-width bar on every warp lane, coverage envelope 100%, all 25
                    # gates PASS, true operator coverage 44.9%. decode_profile.py always
                    # writes an explicit layer, so a missing one means the data is wrong —
                    # fail instead of guessing the renderable value.
                    layer = event.get("layer")
                    if layer not in ("task", "phase", "ignored"):
                        raise EmitError(
                            "event {!r} on rank{} CTA{} track{} has layer={!r}; every decoded "
                            "event must carry exactly one of task|phase|ignored. task = an "
                            "operator that counts toward coverage, phase = a sub-stage nested "
                            "in its parent task, ignored = whole-core anchors and grid barriers "
                            "that are kept for SW/HW alignment but never drawn. Fix the ABI "
                            "tasks[].layer / decode_profile.py output; do NOT default it to "
                            "\"task\".".format(event.get("task"), doc["rank"], block, track, layer))
                    if layer == "ignored":
                        continue
                    start = (event["start_ns"] - shift_ns) / 1000.0
                    end = (event["end_ns"] - shift_ns) / 1000.0
                    while open_stack and open_stack[-1][0] <= start:
                        open_stack.pop()
                    if open_stack and end > open_stack[-1][0]:
                        improper_overlaps.append({"rank": doc["rank"], "block": block, "track": track,
                                                  "task": event.get("task"), "start_us": start, "end_us": end,
                                                  "enclosing_end_us": open_stack[-1][0]})
                    open_stack.append((end, event.get("task", "")))
                    global_start = start if global_start is None else min(global_start, start)
                    global_end = end if global_end is None else max(global_end, end)
                    num_x += 1
                    args = {
                        "task_id": event.get("task_id"), "task": event.get("task"),
                        "short_name": event.get("short_name"), "expert_idx": event.get("expert_idx"),
                        "depth": event.get("depth"), "track": track, "block": block,
                        "rank_block": "{}:{}".format(doc["rank"], block), "physical_smid": smid,
                        "smid_written": None if info is None else info.get("written"),
                        "role": event.get("role"), "layer": layer,
                        "task_role": event.get("task_role"), "parent_task_id": event.get("parent_task_id"),
                        "rank": doc["rank"], "warp_id": warp_id, "dur_ns": event.get("dur_ns"),
                        "start_ns_raw": event.get("start_ns"), "end_ns_raw": event.get("end_ns"),
                    }
                    trace_events.append({
                        "name": event.get("task", "task_{}".format(event.get("task_id"))),
                        "cat": "{}/{}".format(layer, event.get("role", "unknown")),
                        "ph": "X", "pid": pid, "tid": tid,
                        "ts": start, "dur": event.get("dur_ns", 0) / 1000.0, "args": args,
                    })

    makespan_us = 0.0 if global_start is None or global_end is None else global_end - global_start
    # The warp mapping is an assumption when no layout was supplied; publish it rather than
    # hiding it, so a reviewer of the delivered trace can tell "warp3" from "track3 that we
    # assumed is one warp".
    warp_layout_report = {
        "source": "explicit" if warp_layout else "identity_fallback",
        "mapping_claim": ("supplied track -> physical warp ids" if warp_layout
                          else "warp_id == track_id (assumes exactly one physical warp per "
                               "observer track; UNVERIFIED here — check the device code and "
                               "pass --warp-layout if a track is written by several warps)"),
        "supplied_keys": sorted(warp_layout) if warp_layout else [],
    }
    summary = {
        "num_trace_events": len(trace_events), "num_x_events": num_x,
        "num_processes": process_index, "pid_base": PID_BASE,
        "improper_overlaps": len(improper_overlaps),
        "improper_overlap_samples": improper_overlaps[:20],
        "physical_sm_on_software_lane": physical_sm_on_software_lane,
        "warp_layout": warp_layout_report,
        "window_us": [global_start, global_end], "makespan_us": makespan_us,
        "ranks": [{"rank": d["rank"], "device": d["device"], "events": len(d["events"]),
                   "shift_ns": d["shift_ns"], "path": str(d["path"])} for d in ordered],
    }
    trace = {"traceEvents": trace_events, "displayTimeUnit": "ns",
             "otherData": {"producer": "profile/hw_perfetto/emit_perfetto.py",
                           "pid_base": PID_BASE, "num_processes": process_index,
                           "warp_layout": warp_layout_report,
                           "shift_ns_by_rank": {str(d["rank"]): d["shift_ns"] for d in ordered},
                           "sources": [str(d["path"]) for d in ordered]}}
    return trace, summary


# --------------------------------------------------------------------------- #
# HW merge (ph='C') — shift-only, R6 pid ordering, from merge_hw_into_perfetto.py
# --------------------------------------------------------------------------- #
def collect_pids(events: List[Dict[str, Any]]) -> List[int]:
    return sorted({event["pid"] for event in events if "pid" in event})


def merge_hw(trace: Dict[str, Any], hw_metrics: Dict[str, Any],
             alignment: Dict[str, Any]) -> Dict[str, Any]:
    events_list: List[Dict[str, Any]] = trace["traceEvents"]
    counters = hw_metrics.get("counters") or []
    device = hw_metrics.get("device")

    # WHY every field below is checked instead of defaulted: this function maps nsys
    # timestamps onto the software timeline with ts = sw_start + (ns - nsys_start) - shift_ns.
    # A defaulted sw_window ([0,0]) or a defaulted shift_ns (0) does not fail — it silently
    # slides every HW curve by up to the whole kernel makespan, and the trace still opens,
    # still has counters, still passes the render checks. The only honest answer to a
    # missing alignment field is to stop and re-run align_sw_hw.py.
    scale = alignment.get("scale")
    if scale != 1:
        raise EmitError(
            "alignment scale is {!r}, must be exactly 1. SW/HW alignment is shift-only by "
            "contract: nsys_span > sw_span is structural (PM sampling starts before and ends "
            "after the kernel), so stretching the HW timeline to fit would fabricate "
            "correlation. Fix align_sw_hw.py output instead of rescaling here.".format(scale))
    sw_window = alignment.get("sw_window_ns")
    if not isinstance(sw_window, (list, tuple)) or len(sw_window) != 2 or sw_window[0] is None:
        raise EmitError(
            "alignment has no usable sw_window_ns ({!r}); it is the software zero the HW "
            "samples are mapped onto. Defaulting it to [0, 0] would shift every counter by "
            "the kernel start offset. Re-run align_sw_hw.py --hw-metrics.".format(sw_window))
    nsys_window = alignment.get("nsys_window_ns")
    if not nsys_window:
        raise EmitError("alignment has no nsys_window_ns; run align_sw_hw.py with --hw-metrics")
    sw_start = int(sw_window[0])
    nsys_start, nsys_end = int(nsys_window[0]), int(nsys_window[1])
    if alignment.get("shift_ns") is None:
        raise EmitError(
            "alignment has no shift_ns. shift_ns is the per-rank entry-rendezvous release "
            "used as timeline zero (align_sw_hw.py falls back to earliest_software_start and "
            "records which one it used in shift_ns_source). Silently using 0 puts the HW "
            "curves at absolute device time while the software spans are rebased.")
    shift_ns = int(alignment["shift_ns"])

    sw_pids = collect_pids(events_list)
    min_sw_pid = min(sw_pids) if sw_pids else PID_BASE
    pid_shift = 0
    hw_pid = min_sw_pid - 1
    if hw_pid < 0:
        pid_shift = 1 - min_sw_pid
        for event in events_list:
            if "pid" in event:
                event["pid"] += pid_shift
        hw_pid = 0
        sw_pids = collect_pids(events_list)
        min_sw_pid = min(sw_pids)

    def map_ts_us(ns: int) -> float:
        return (sw_start + (ns - nsys_start) - shift_ns) / 1000.0

    counter_events: List[Dict[str, Any]] = []
    per_counter = []
    for counter in counters:
        name = counter.get("name", "metric")
        unit = counter.get("unit", "")
        track_name = "{} [{}]".format(name, unit) if unit else name
        kept = 0
        for ts_ns, value in counter.get("samples", []):
            if nsys_start <= ts_ns <= nsys_end:
                counter_events.append({"ph": "C", "name": track_name, "ts": map_ts_us(ts_ns),
                                       "pid": hw_pid, "tid": 0, "args": {"value": value}})
                kept += 1
        per_counter.append({"name": name, "unit": unit, "samples_in_window": kept})

    hw_meta = [
        {"name": "process_name", "ph": "M", "pid": hw_pid, "tid": 0,
         "args": {"name": "HW dev{} (device-wide, full-kernel sampled)".format(device)}},
        {"name": "process_sort_index", "ph": "M", "pid": hw_pid, "tid": 0, "args": {"sort_index": -1}},
        {"name": "process_labels", "ph": "M", "pid": hw_pid, "tid": 0, "args": {"labels": HW_NOTE}},
    ]
    trace["traceEvents"] = events_list + hw_meta + counter_events
    rebase = rebase_timeline_nonnegative(trace["traceEvents"])

    other = trace.setdefault("otherData", {})
    other["hw_merge"] = {"schema": "hw-merge/v1", "note": HW_NOTE, "device": device,
                         "hw_counter_pid": hw_pid, "software_pids": sw_pids,
                         "software_pid_shift_applied": pid_shift,
                         "r6_hw_pid_lt_all_software": hw_pid < min_sw_pid,
                         "shift_ns": shift_ns, "alignment_mode": "shift-only (scale=1)",
                         "sw_window_ns": [sw_start, sw_window[1]], "nsys_window_ns": [nsys_start, nsys_end],
                         "timeline_rebase": rebase, "counters": per_counter}
    other["timeline_rebase"] = rebase
    if hw_pid >= min_sw_pid:
        raise EmitError("R6 violated: HW counter pid {} not < all software pids {}".format(hw_pid, sw_pids))
    return {"hw_counter_pid": hw_pid, "counter_events": len(counter_events),
            "r6_ok": hw_pid < min_sw_pid, "rebase": rebase, "per_counter": per_counter}


# --------------------------------------------------------------------------- #
# top-level build API used by both the CLI and the selftests
# --------------------------------------------------------------------------- #
def build_perfetto(events_docs: List[Tuple[Dict[str, Any], Optional[str]]],
                   hw_metrics: Optional[Dict[str, Any]] = None,
                   alignment: Optional[Dict[str, Any]] = None,
                   warp_layout: Optional[Dict[str, List[int]]] = None
                   ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    docs = [normalize_decoded(decoded, None, None, None, path) for decoded, path in events_docs]
    trace, summary = build_sw_trace(docs, warp_layout)
    if hw_metrics is not None:
        if alignment is None:
            primary = docs[0]
            alignment = compute_alignment({"metadata": {"rank": primary["rank"], "device": primary["device"]},
                                           "events": primary["events"], "anchors": primary["anchors"]},
                                          hw_metrics)
        summary["hw_merge"] = merge_hw(trace, hw_metrics, alignment)
    else:
        rebase = rebase_timeline_nonnegative(trace["traceEvents"])
        trace.setdefault("otherData", {})["timeline_rebase"] = rebase
        summary["rebase"] = rebase
    return trace, summary


def write_trace_gz(path: Path, trace: Dict[str, Any], max_bytes: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(trace, handle, ensure_ascii=False)
    size = path.stat().st_size
    if size > max_bytes:
        raise EmitError(
            "output {} is {} bytes > cap {} bytes. Oversize policy: shrink the trace "
            "(fewer CTAs/tracks in the display rank, or coarser granularity) or raise "
            "--max-bytes deliberately; do NOT silently truncate events.".format(path, size, max_bytes))
    return size


def _resolve_from_contract(contract_path: Path
                           ) -> Tuple[List[Path], Optional[Path], Optional[Path], Path, Optional[Path]]:
    """PROJECT-GENERIC contract reader: pull decoded-events / hw / alignment / out paths.

    Uses tolerant lookups so it works with the skeleton's own materialized contract
    as well as the reference contract_lock_example.json shape.
    """
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    root = contract_path.resolve().parent
    artifacts = contract.get("artifacts") or {}
    layout = contract.get("project_layout") or {}

    def _rel(*keys: str) -> Optional[Path]:
        node: Any = contract
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, str):
            candidate = Path(node)
            return candidate if candidate.is_absolute() else root / candidate
        return None

    events_glob = (artifacts.get("events_glob") or layout.get("events_glob") or "sw/events_rank*.json")
    events_paths = sorted(root.glob(events_glob)) if not Path(events_glob).is_absolute() \
        else sorted(Path(events_glob).parent.glob(Path(events_glob).name))
    hw = _rel("artifacts", "hw_metrics") or (root / "hw" / "hw_metrics.json")
    align = _rel("artifacts", "alignment") or (root / "work" / "alignment.json")
    out = _rel("artifacts", "perfetto") or (root / "work" / "candidate.perfetto.json.gz")
    max_bytes = int(artifacts.get("max_bytes") or DEFAULT_MAX_BYTES)
    contract.setdefault("_resolved_max_bytes", max_bytes)
    # WHY this is not a tolerant `if is_file() else None`: the contract states which
    # bundle is being delivered. If it says sw_hw / hardware.enabled and the HW artifact
    # is missing, silently emitting a software-only trace produces a file that opens
    # fine, renders every span, and is missing the entire hardware half — with nothing in
    # the output saying so. The declared bundle is the intent; a missing input is a
    # pipeline failure, not a downgrade.
    capture = contract.get("capture") or {}
    hardware = capture.get("hardware") or {}
    # Both spellings exist in the wild: materialize_contract.py writes "sw_hw", the
    # reference contract_lock_example.json writes "sw+hw". Accept both, otherwise the
    # check silently does nothing for half the contracts.
    hw_declared = (capture.get("information_bundle") in ("sw_hw", "sw+hw")
                   or bool(hardware.get("enabled")))
    hw_present = bool(hw and hw.is_file())
    align_present = bool(align and align.is_file())
    if hw_declared and not hw_present:
        raise EmitError(
            "contract {} declares a sw_hw bundle (capture.information_bundle / "
            "capture.hardware.enabled) but the HW metrics artifact {} does not exist. Run "
            "capture_nsys.py + extract_gpu_metrics.py first, or re-materialize the contract "
            "with --information-bundle sw_only --hardware off if this run really is software "
            "only. Emitting a SW-only trace under a sw_hw contract would hide the missing "
            "half.".format(contract_path, hw))
    if hw_declared and not align_present:
        raise EmitError(
            "contract {} declares a sw_hw bundle but the alignment artifact {} does not "
            "exist. build_perfetto() can recompute an alignment in memory, which is fine for "
            "a selftest but not for a delivery: the alignment (shift_ns_source, span_mismatch, "
            "cross-CTA residual) is the evidence that SW and HW share one timeline. Run "
            "align_sw_hw.py --hw-metrics.".format(contract_path, align))
    hw = hw if hw_present else None
    align = align if align_present else None
    return events_paths, hw, align, out, contract_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Emit a Perfetto trace from decoded SW events (+ optional HW).")
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument("--events", type=Path, action="append", default=None)
    parser.add_argument("--hw-metrics", type=Path, default=None)
    parser.add_argument("--align", type=Path, default=None)
    parser.add_argument("--warp-layout", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args(argv)

    events_paths: List[Path]
    hw_path = args.hw_metrics
    align_path = args.align
    out_path = args.out
    max_bytes = args.max_bytes
    if args.contract is not None:
        try:
            events_paths, c_hw, c_align, c_out, _ = _resolve_from_contract(args.contract)
        except EmitError as error:
            # Same exit code as every other EmitError in this stage: a contract whose
            # declared bundle cannot be honoured is a stage failure, not a traceback.
            print("[emit] {}".format(error), file=sys.stderr)
            return 4
        hw_path = hw_path or c_hw
        align_path = align_path or c_align
        out_path = out_path or c_out
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        max_bytes = int((contract.get("artifacts") or {}).get("max_bytes") or max_bytes)
    else:
        if not args.events:
            print("[emit] need --contract or at least one --events", file=sys.stderr)
            return 2
        events_paths = list(args.events)
    if out_path is None:
        print("[emit] no --out resolved", file=sys.stderr)
        return 2
    if not events_paths:
        print("[emit] no decoded events found (did decode_profile.py run?)", file=sys.stderr)
        return 2

    warp_layout = None
    if args.warp_layout is not None and args.warp_layout.is_file():
        warp_layout = json.loads(args.warp_layout.read_text(encoding="utf-8"))

    events_docs = [(load_json_maybe_gz(path), str(path)) for path in events_paths]
    hw_metrics = load_json_maybe_gz(hw_path) if hw_path and Path(hw_path).is_file() else None
    alignment = load_json_maybe_gz(align_path) if align_path and Path(align_path).is_file() else None

    try:
        trace, summary = build_perfetto(events_docs, hw_metrics, alignment, warp_layout)
        if summary["improper_overlaps"]:
            print("[emit] FAIL {} partial overlaps on a single lane (render contract R3 "
                  "allows only disjoint/nested): {}".format(summary["improper_overlaps"],
                                                             summary["improper_overlap_samples"][:3]), file=sys.stderr)
            return 3
        size = write_trace_gz(out_path, trace, max_bytes)
    except EmitError as error:
        print("[emit] {}".format(error), file=sys.stderr)
        return 4

    print("[emit] wrote {} bytes={} x_events={} processes={} hw={}".format(
        out_path, size, summary["num_x_events"], summary["num_processes"],
        "merged" if hw_metrics is not None else "none"))
    if summary.get("hw_merge"):
        print("[emit] hw pid={} r6_ok={} counter_events={}".format(
            summary["hw_merge"]["hw_counter_pid"], summary["hw_merge"]["r6_ok"],
            summary["hw_merge"]["counter_events"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
