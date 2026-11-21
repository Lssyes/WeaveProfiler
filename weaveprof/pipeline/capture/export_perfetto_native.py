#!/usr/bin/env python3
"""Export a WeaveProfiler Chrome trace JSON to a Perfetto-native protobuf.

The Chrome JSON importer is useful for debugging, but a delivery trace should
use Perfetto TrackEvent packets.  This exporter keeps one track per physical
wave, groups those tracks under their SM/CU, preserves canonical semantic names
(``load``, ``mma``, ``store``), and emits hardware aggregate counters without
pretending that per-dispatch PM counters have intra-kernel resolution.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as pb
except ImportError as exc:  # pragma: no cover - exercised on hosts without the package
    raise SystemExit(
        "export_perfetto_native.py requires the Perfetto Python protobuf package. "
        "Install it with: python3 -m pip install 'perfetto>=0.59,<1'"
    ) from exc

SEMANTIC_NAMES = ("gemm", "load", "wait", "mma", "barrier", "store")
UTILIZATION_ORDER = (
    "compute_util",
    "tensor_util",
    "simd_util",
    "memory_util",
    "memory_read_util",
    "memory_write_util",
    "sm_active",
)
UTILIZATION_LABELS = {
    "compute_util": "compute_util [matrix_core]",
    "tensor_util": "tensor_util [matrix_core]",
    "simd_util": "simd_util [SIMD/CU]",
    "memory_util": "memory_util [HBM request estimate]",
    "memory_read_util": "memory_read_util [HBM request estimate]",
    "memory_write_util": "memory_write_util [HBM request estimate]",
    "sm_active": "sm_active [GPU active]",
}
SEQUENCE_ID = 1
PROCESS_SOFTWARE_UUID = 0x100000
PROCESS_HARDWARE_UUID = 0x200000
TRACK_KERNEL_UUID = 0x300000
TRACK_SOFTWARE_COUNTER_BASE = 0x400000
TRACK_HARDWARE_BASE = 0x500000
TRACK_COMMUNICATION_UUID = 0x600000
TRACK_PROCESS_BASE = 0x700000
KNOWN_CARDS = ("h20", "b300", "mi308x")
CARD_FILENAME_RE = re.compile(
    r"(?<![A-Za-z0-9])card-(h20|b300|mi308x)(?![A-Za-z0-9])"
)


def read_json_maybe_gz(path: Path) -> Any:
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    with handle:
        return json.load(handle)


def number(value: Any, default: Optional[float] = None) -> Optional[float]:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def int_number(value: Any, default: Optional[int] = None) -> Optional[int]:
    result = number(value)
    return default if result is None else int(round(result))


def embedded_card(other_data: Mapping[str, Any]) -> Optional[str]:
    """Return the known card recorded in otherData.meta/target, if any."""
    found: Dict[str, str] = {}
    for section_name in ("meta", "target"):
        section = other_data.get(section_name)
        if not isinstance(section, dict):
            continue
        raw_card = section.get("card")
        if not isinstance(raw_card, str):
            continue
        card = raw_card.strip().lower()
        if card in KNOWN_CARDS:
            found[f"otherData.{section_name}.card"] = card
    if not found:
        return None
    cards = set(found.values())
    if len(cards) != 1:
        detail = ", ".join(f"{key}={value!r}" for key, value in sorted(found.items()))
        raise ValueError(f"conflicting embedded card metadata: {detail}")
    return next(iter(cards))


def validate_output_card(out_path: Path, other_data: Mapping[str, Any]) -> None:
    """Enforce a known embedded card token without guessing an unknown card."""
    card = embedded_card(other_data)
    if card is None:
        return
    expected = f"card-{card}"
    present = {match.group(0) for match in CARD_FILENAME_RE.finditer(out_path.name)}
    if expected not in present:
        raise ValueError(
            f"output filename must contain {expected} for otherData card {card!r}: "
            f"{out_path.name}"
        )
    mismatched = sorted(present - {expected})
    if mismatched:
        raise ValueError(
            f"output filename card token does not match otherData card {card!r}: "
            f"{', '.join(mismatched)} in {out_path.name}"
        )


def packet(trace: Any, **kwargs: Any) -> Any:
    item = trace.packet.add(**kwargs)
    item.trusted_packet_sequence_id = SEQUENCE_ID
    return item


def add_process(trace: Any, uuid: int, name: str, *, pid: int,
                sort_index: int, description: Optional[str] = None) -> None:
    descriptor = pb.TrackDescriptor(
        uuid=uuid,
        name=name,
        process=pb.ProcessDescriptor(pid=pid, process_name=name),
        description=description or "",
    )
    descriptor.sibling_order_rank = sort_index
    packet(trace, track_descriptor=descriptor)


def add_track(trace: Any, uuid: int, parent_uuid: int, name: str, *,
              sort_index: int = 0, description: Optional[str] = None,
              counter: Optional[pb.CounterDescriptor] = None) -> None:
    descriptor = pb.TrackDescriptor(
        uuid=uuid,
        parent_uuid=parent_uuid,
        name=name,
        description=description or "",
    )
    descriptor.sibling_order_rank = sort_index
    if counter is not None:
        descriptor.counter.CopyFrom(counter)
    packet(trace, track_descriptor=descriptor)


def hardware_rows(events: Sequence[Mapping[str, Any]],
                  other_data: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows = other_data.get("hardware_rows")
    if isinstance(rows, list):
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            semantic_id = str(raw.get("semantic_id") or "")
            label = str(raw.get("timeline_label") or UTILIZATION_LABELS.get(semantic_id, semantic_id))
            value = number(raw.get("value"))
            start_us = number(raw.get("hardware_start_placement_us"))
            if start_us is None:
                start_us = number(raw.get("software_start_us"), 0.0)
            duration_us = number(raw.get("hardware_dur_us"))
            if duration_us is None:
                duration_ns = number(raw.get("hardware_dur_ns"))
                duration_us = (duration_ns / 1000.0) if duration_ns is not None else None
            if not semantic_id or value is None or start_us is None or duration_us is None:
                continue
            grouped[label].append({
                "semantic_id": semantic_id,
                "label": label,
                "start_ns": int(round(start_us * 1000.0)),
                "dur_ns": max(1, int(round(duration_us * 1000.0))),
                "value": float(value),
                "dispatch_ordinal": raw.get("dispatch_ordinal"),
                "hardware_start_ns": raw.get("hardware_start_ns"),
                "hardware_dur_ns": raw.get("hardware_dur_ns"),
                "counter_time_resolution": raw.get("counter_time_resolution", "per_dispatch_aggregate"),
                "intra_kernel_curve": bool(raw.get("intra_kernel_curve", False)),
            })
    if not grouped:
        # Fallback for a Chrome trace whose hardware rows are present only as
        # counter events. Keep the label and measured values; do not synthesize
        # a series.
        for raw in events:
            if raw.get("ph") != "C":
                continue
            args = raw.get("args") or {}
            value = number(args.get("value"))
            ts = number(raw.get("ts"))
            if value is None or ts is None:
                continue
            label = str(raw.get("name") or "hardware_util")
            grouped[label].append({
                "semantic_id": label.split(" ", 1)[0],
                "label": label,
                "start_ns": int(round(ts * 1000.0)),
                "dur_ns": 1,
                "value": float(value),
                "dispatch_ordinal": None,
                "hardware_start_ns": None,
                "hardware_dur_ns": None,
                "counter_time_resolution": "counter_event",
                "intra_kernel_curve": False,
            })
    return grouped


def semantic_events(events: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for raw in events:
        if raw.get("ph") != "X":
            continue
        args = raw.get("args")
        if not isinstance(args, dict):
            continue
        semantic_id = str(args.get("semantic_id") or raw.get("name") or "")
        if semantic_id not in SEMANTIC_NAMES:
            continue
        start_us = number(raw.get("ts"))
        duration_us = number(raw.get("dur"))
        if start_us is None or duration_us is None or start_us < 0 or duration_us <= 0:
            continue
        result.append({
            "name": semantic_id,
            "start_ns": int(round(start_us * 1000.0)),
            "dur_ns": max(1, int(round(duration_us * 1000.0))),
            "args": args,
        })
    result.sort(key=lambda item: (item["start_ns"], item["name"]))
    return result


def wave_identity(args: Mapping[str, Any]) -> Tuple[Any, ...]:
    for keys in (("cta", "warp"), ("block", "lane"), ("cta", "wave_slot")):
        if all(key in args for key in keys):
            return tuple(args.get(key) for key in keys)
    if "wave_slot" in args:
        return ("wave_slot", args["wave_slot"])
    return ("unknown",)


def wave_track_name(args: Mapping[str, Any]) -> str:
    cta = args.get("cta", args.get("block", "?"))
    warp = args.get("warp", args.get("wave_slot", "?"))
    slot = args.get("wave_slot")
    name = f"wave {slot:04d} · cta {cta:03d} · warp {warp:02d}" if isinstance(slot, int) else f"cta {cta} / warp {warp}"
    role = args.get("role")
    if isinstance(role, str) and role and role not in ("cta_worker", "0"):
        name += f" ({role})"
    if args.get("cu") is not None:
        name += f" · CU {args.get('cu')}"
    return name


def parent_identity(args: Mapping[str, Any]) -> Tuple[Any, str]:
    if args.get("xcc") is not None and args.get("sm") is not None:
        return args.get("sm"), f"CU {args.get('sm')} (XCC{args.get('xcc')} SE{args.get('se')} CU{args.get('cu')})"
    if args.get("sm") is not None:
        return args.get("sm"), f"SM {args.get('sm')}"
    if args.get("cu") is not None:
        return args.get("cu"), f"CU {args.get('cu')}"
    return 0, "physical waves"


def selected_event(item: Mapping[str, Any], reserved_k: Optional[Sequence[int]]) -> bool:
    if reserved_k is None:
        return True
    name = item["name"]
    if name == "store":
        return True
    k_block = int_number(item["args"].get("k_block"))
    return k_block is None or k_block in reserved_k


def occupancy_tracks(events: Sequence[Mapping[str, Any]], trace: Any,
                     parent_uuid: int) -> Tuple[int, Dict[str, Any]]:
    if not events:
        return 0, {}
    start_us = min(item["start_ns"] for item in events) / 1000.0
    end_us = max(item["start_ns"] + item["dur_ns"] for item in events) / 1000.0
    bin_us = 1.0
    bins = max(2, int(math.ceil((end_us - start_us) / bin_us)) + 1)
    summaries: Dict[str, Any] = {}
    for offset, semantic_id in enumerate(("load", "mma", "store")):
        diff = [0] * (bins + 1)
        for item in events:
            if item["name"] != semantic_id:
                continue
            a = int(math.floor((item["start_ns"] / 1000.0 - start_us) / bin_us))
            b = int(math.ceil(((item["start_ns"] + item["dur_ns"]) / 1000.0 - start_us) / bin_us))
            a = max(0, min(bins, a))
            b = max(a + 1, min(bins, b))
            diff[a] += 1
            diff[b] -= 1
        active = 0
        points: List[Tuple[int, int]] = []
        for index in range(bins):
            active += diff[index]
            points.append((
                int(round((start_us + index * bin_us) * 1000.0)),
                int(active),
            ))
        track_uuid = TRACK_SOFTWARE_COUNTER_BASE + offset
        add_track(
            trace, track_uuid, parent_uuid,
            f"software_{semantic_id}_active_waves [derived from intervals]",
            sort_index=offset,
            description="Number of physical waves with this semantic interval active in a 1 us bin. This is a software-derived occupancy curve, not a hardware counter.",
            counter=pb.CounterDescriptor(unit_name="waves", y_axis_share_key="weaveprofiler:active-waves"),
        )
        for ts_ns, value in points:
            packet(
                trace,
                timestamp=ts_ns,
                track_event=pb.TrackEvent(
                    type=pb.TrackEvent.TYPE_COUNTER,
                    track_uuid=track_uuid,
                    counter_value=int(value),
                ),
            )
        summaries[semantic_id] = {
            "track": f"software_{semantic_id}_active_waves [derived from intervals]",
            "bin_us": bin_us,
            "point_count": len(points),
            "max_active_waves": max((value for _, value in points), default=0),
        }
    return bins, summaries


def export(trace_path: Path, out_path: Path, *, preview_k_blocks: Optional[Sequence[int]] = None,
           occupancy: bool = True) -> Dict[str, Any]:
    document = read_json_maybe_gz(trace_path)
    if not isinstance(document, dict) or not isinstance(document.get("traceEvents"), list):
        raise ValueError("input must be a Chrome trace object with traceEvents")
    events = document["traceEvents"]
    other = document.get("otherData") if isinstance(document.get("otherData"), dict) else {}
    validate_output_card(out_path, other)
    all_semantic = semantic_events(events)
    gemm_events = [item for item in all_semantic if item["name"] == "gemm"]
    semantic = [item for item in all_semantic if item["name"] != "gemm" and selected_event(item, preview_k_blocks)]
    if not semantic:
        raise ValueError("input has no physical-wave semantic events after filtering")

    trace = pb.Trace()
    platform = str(other.get("platform") or ("nvidia" if other.get("hw_sw_info") else "unknown"))
    arch = str(other.get("arch") or (other.get("hw_sw_info") or {}).get("arch") or "unknown")
    is_nv = platform.lower().startswith("nv")
    hw_proc_name = (
        f"00 · 硬件信息与利用率 [0–100%] · NVIDIA {arch} GPU Metrics"
        if is_nv else
        f"00 · 硬件信息与利用率 [0–100%] · AMD {arch} ROCm PM (same dispatch)"
    )
    add_process(trace, PROCESS_HARDWARE_UUID,
                hw_proc_name, pid=1, sort_index=-1000,
                description="Top-level GPU hardware info & compute/memory utilization [0-100%].")
    add_process(trace, PROCESS_SOFTWARE_UUID,
                f"01 · 软件总体包络 · WeaveProfiler {platform} {arch}", pid=2, sort_index=-500,
                description="Physical wave/CTA intervals from the instrumented kernel.")

    TRACK_HARDWARE_INFO_UUID = 0x250000
    add_track(trace, TRACK_HARDWARE_INFO_UUID, PROCESS_HARDWARE_UUID,
              "GPU 硬件规格与 Kernel 时间窗", sort_index=-10,
              description="Hardware device specification and measured dispatch window.")
    add_track(trace, TRACK_KERNEL_UUID, PROCESS_SOFTWARE_UUID,
              "gemm dispatch", sort_index=-10,
              description="Observation window for the selected instrumented GEMM dispatch.")

    wave_args: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for item in semantic:
        identity = wave_identity(item["args"])
        wave_args.setdefault(identity, item["args"])
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for item in semantic:
        grouped[wave_identity(item["args"])].append(item)

    process_for_parent: Dict[Any, int] = {}
    for parent, _ in sorted({parent_identity(args) for args in wave_args.values()}, key=lambda item: str(item[0])):
        process_uuid = TRACK_PROCESS_BASE + (len(process_for_parent) & 0xFFFF)
        process_for_parent[parent] = process_uuid
        add_process(
            trace, process_uuid, f"{platform.upper()} {parent_identity(next(args for args in wave_args.values() if parent_identity(args)[0] == parent))[1]}",
            pid=100 + len(process_for_parent), sort_index=100 + (int(parent) if isinstance(parent, int) else 0),
            description="Physical wave group.",
        )

    wave_track_for_identity: Dict[Tuple[Any, ...], int] = {}
    for sort_index, (identity, items) in enumerate(sorted(grouped.items(), key=lambda pair: str(pair[0]))):
        args = items[0]["args"]
        parent = parent_identity(args)[0]
        track_uuid = TRACK_PROCESS_BASE + 0x10000 + sort_index
        wave_track_for_identity[identity] = track_uuid
        add_track(
            trace, track_uuid, process_for_parent[parent], wave_track_name(args),
            sort_index=sort_index,
            description="One physical wave; events are canonical load -> mma -> store intervals.",
        )

    # Emit begin/end pairs in timestamp order.  The input decoder guarantees
    # non-overlapping intervals within one physical wave; keeping the pairs
    # separate also makes the native trace readable by older Perfetto versions.
    ordered: List[Tuple[int, int, str, int, Dict[str, Any]]] = []
    for identity, items in grouped.items():
        track_uuid = wave_track_for_identity[identity]
        for item in items:
            ordered.append((item["start_ns"], item["dur_ns"], item["name"], track_uuid, item["args"]))
    ordered.sort(key=lambda row: (row[0], row[2]))

    for start_ns, dur_ns, name, track_uuid, args in ordered:
        annotations = []
        k_block = int_number(args.get("k_block"))
        if k_block is not None:
            annotations.append(pb.DebugAnnotation(name="k_block", int_value=k_block))
        source_label = args.get("original_label")
        if source_label:
            annotations.append(pb.DebugAnnotation(name="source_label", string_value=str(source_label)))
        annotations.append(pb.DebugAnnotation(name="semantic_id", string_value=name))
        packet(
            trace, timestamp=start_ns,
            track_event=pb.TrackEvent(
                type=pb.TrackEvent.TYPE_SLICE_BEGIN,
                track_uuid=track_uuid,
                name=name,
                categories=[name],
                debug_annotations=annotations,
            ),
        )
        packet(
            trace, timestamp=start_ns + dur_ns,
            track_event=pb.TrackEvent(
                type=pb.TrackEvent.TYPE_SLICE_END,
                track_uuid=track_uuid,
            ),
        )

    # The kernel envelope is optional in the source, but useful in every output.
    starts = [item["start_ns"] for item in semantic]
    ends = [item["start_ns"] + item["dur_ns"] for item in semantic]
    kernel_start = min(starts)
    kernel_end = max(ends)
    if gemm_events:
        kernel_start = min(kernel_start, min(item["start_ns"] for item in gemm_events))
        kernel_end = max(kernel_end, max(item["start_ns"] + item["dur_ns"] for item in gemm_events))
    gpu_info_obj = (other.get("hw_sw_info") or {}).get("gpu") or other.get("target") or {}
    gpu_model_str = str(gpu_info_obj.get("name") or f"{platform.upper()} {arch}")
    hw_win_label = f"{gpu_model_str} · {arch} · kernel window ({round((kernel_end - kernel_start) / 1000.0, 2)} us)"
    packet(
        trace, timestamp=kernel_start,
        track_event=pb.TrackEvent(
            type=pb.TrackEvent.TYPE_SLICE_BEGIN,
            track_uuid=TRACK_HARDWARE_INFO_UUID,
            name=hw_win_label,
            categories=["hardware.dispatch"],
            debug_annotations=[
                pb.DebugAnnotation(name="gpu", string_value=gpu_model_str),
                pb.DebugAnnotation(name="arch", string_value=arch),
                pb.DebugAnnotation(name="platform", string_value=platform),
                pb.DebugAnnotation(name="window_us", double_value=round((kernel_end - kernel_start) / 1000.0, 3)),
            ],
        ),
    )
    packet(
        trace, timestamp=kernel_end,
        track_event=pb.TrackEvent(
            type=pb.TrackEvent.TYPE_SLICE_END,
            track_uuid=TRACK_HARDWARE_INFO_UUID,
        ),
    )
    packet(
        trace, timestamp=kernel_start,
        track_event=pb.TrackEvent(
            type=pb.TrackEvent.TYPE_SLICE_BEGIN,
            track_uuid=TRACK_KERNEL_UUID,
            name="gemm",
            categories=["gemm"],
            debug_annotations=[
                pb.DebugAnnotation(name="semantic_id", string_value="gemm"),
                pb.DebugAnnotation(name="physical_wave_count", int_value=len(grouped)),
            ],
        ),
    )
    packet(
        trace, timestamp=kernel_end,
        track_event=pb.TrackEvent(
            type=pb.TrackEvent.TYPE_SLICE_END,
            track_uuid=TRACK_KERNEL_UUID,
        ),
    )

    hardware = hardware_rows(events, other)
    hardware_track_count = 0
    hardware_point_count = 0
    hardware_summary: List[Dict[str, Any]] = []
    emitted_labels = set()
    ordered_hw_items: List[Tuple[str, str, List[Dict[str, Any]]]] = []
    for semantic_id in UTILIZATION_ORDER:
        label = UTILIZATION_LABELS[semantic_id]
        rows = hardware.get(label) or []
        if rows:
            ordered_hw_items.append((semantic_id, label, rows))
            emitted_labels.add(label)
    for label in sorted(hardware.keys()):
        if label in emitted_labels:
            continue
        rows = hardware.get(label) or []
        if not rows:
            continue
        # Only export percentage / normalized [0..100] utilization tracks on the top hardware pane
        vals = [float(r["value"]) for r in rows]
        if max(vals) > 105.0 or min(vals) < -1.0:
            continue
        ordered_hw_items.append((str(rows[0].get("semantic_id") or label), label, rows))
        emitted_labels.add(label)

    aggregate_track_uuids: List[int] = []
    for index, (semantic_id, label, rows) in enumerate(ordered_hw_items):
        track_uuid = TRACK_HARDWARE_BASE + index
        ordered_rows = sorted(rows, key=lambda r: (r["start_ns"], r["semantic_id"]))
        is_timeseries = (
            all(r.get("counter_time_resolution") == "counter_event" for r in ordered_rows)
            and len(ordered_rows) >= 2
        )
        is_intra_curve = bool(ordered_rows[0].get("intra_kernel_curve")) or (is_timeseries and len(ordered_rows) > 2)
        add_track(
            trace, track_uuid, PROCESS_HARDWARE_UUID, label,
            sort_index=index,
            description=(
                "% of peak hardware throughput (intra-kernel periodic GPU metrics)."
                if is_intra_curve else
                "% of peak hardware utilization (same-dispatch aggregate)."
            ),
            counter=pb.CounterDescriptor(
                unit_name="%",
                y_axis_share_key="weaveprofiler:utilization-percent",
            ),
        )
        hardware_track_count += 1
        if is_timeseries:
            for row in ordered_rows:
                ts_ns = int(row["start_ns"])
                packet(
                    trace, timestamp=ts_ns,
                    track_event=pb.TrackEvent(
                        type=pb.TrackEvent.TYPE_COUNTER,
                        track_uuid=track_uuid,
                        double_counter_value=float(row["value"]),
                        debug_annotations=[
                            pb.DebugAnnotation(name="semantic_id", string_value=semantic_id),
                            pb.DebugAnnotation(name="counter_time_resolution", string_value="periodic_pm" if is_intra_curve else "per_dispatch_aggregate"),
                            pb.DebugAnnotation(name="intra_kernel_curve", bool_value=is_intra_curve),
                        ],
                    ),
                )
                hardware_point_count += 1
            if not is_intra_curve:
                aggregate_track_uuids.append(track_uuid)
            hardware_summary.append({
                "semantic_id": semantic_id,
                "track": label,
                "value": round(sum(float(r["value"]) for r in ordered_rows) / len(ordered_rows), 3),
                "min": min(float(r["value"]) for r in ordered_rows),
                "max": max(float(r["value"]) for r in ordered_rows),
                "points": len(ordered_rows),
                "start_ns": int(ordered_rows[0]["start_ns"]),
                "dur_ns": max(1, int(ordered_rows[-1]["start_ns"]) - int(ordered_rows[0]["start_ns"])),
                "resolution": "periodic_pm" if is_intra_curve else "per_dispatch_aggregate",
                "intra_kernel_curve": is_intra_curve,
            })
        else:
            aggregate_track_uuids.append(track_uuid)
            for row in ordered_rows:
                start_ns = int(row["start_ns"])
                end_ns = start_ns + int(row["dur_ns"])
                for ts_ns, value in ((start_ns, row["value"]), (end_ns, row["value"])):
                    packet(
                        trace, timestamp=ts_ns,
                        track_event=pb.TrackEvent(
                            type=pb.TrackEvent.TYPE_COUNTER,
                            track_uuid=track_uuid,
                            double_counter_value=float(value),
                            debug_annotations=[
                                pb.DebugAnnotation(name="semantic_id", string_value=semantic_id),
                                pb.DebugAnnotation(name="counter_time_resolution", string_value=str(row["counter_time_resolution"])),
                                pb.DebugAnnotation(name="intra_kernel_curve", bool_value=bool(row["intra_kernel_curve"])),
                            ],
                        ),
                    )
                    hardware_point_count += 1
                hardware_summary.append({
                    "semantic_id": semantic_id,
                    "track": label,
                    "value": row["value"],
                    "start_ns": start_ns,
                    "dur_ns": int(row["dur_ns"]),
                    "resolution": row["counter_time_resolution"],
                    "intra_kernel_curve": bool(row["intra_kernel_curve"]),
                })
    # End the aggregate band with a display break at one us after the measured window.
    if aggregate_track_uuids:
        reset_ns = kernel_end + 1000
        for track_uuid in aggregate_track_uuids:
            packet(
                trace, timestamp=reset_ns,
                track_event=pb.TrackEvent(
                    type=pb.TrackEvent.TYPE_COUNTER,
                    track_uuid=track_uuid,
                    double_counter_value=0.0,
                    debug_annotations=[pb.DebugAnnotation(name="display_break", bool_value=True)],
                ),
            )

    # Missing interconnect evidence is explicit and visible.
    add_track(
        trace, TRACK_COMMUNICATION_UUID, PROCESS_HARDWARE_UUID,
        "communication_util [interconnect]", sort_index=99,
        description="No same-dispatch XGMI/RDMA/fabric counter was collected. Missing evidence is unavailable, not zero.",
        counter=pb.CounterDescriptor(unit_name="%", y_axis_share_key="weaveprofiler:utilization-percent"),
    )
    packet(
        trace, timestamp=kernel_start,
        track_event=pb.TrackEvent(
            type=pb.TrackEvent.TYPE_INSTANT,
            track_uuid=TRACK_COMMUNICATION_UUID,
            name="unavailable",
            categories=["unavailable"],
            debug_annotations=[
                pb.DebugAnnotation(name="reason", string_value="no same-dispatch XGMI/RDMA/fabric counter was collected in this single-GPU run"),
            ],
        ),
    )

    occupancy_summary: Dict[str, Any] = {}
    if occupancy:
        _, occupancy_summary = occupancy_tracks(semantic, trace, PROCESS_SOFTWARE_UUID)

    # Keep all TrackDescriptor packets first, then sort all timestamped packets monotonically
    # so trusted_packet_sequence_id=1 has zero out-of-order packets.
    desc_packets = [pkt for pkt in trace.packet if not pkt.HasField("timestamp")]
    ts_packets = [pkt for pkt in trace.packet if pkt.HasField("timestamp")]
    ts_packets.sort(key=lambda pkt: pkt.timestamp)
    sorted_trace = pb.Trace()
    sorted_trace.packet.extend(desc_packets)
    sorted_trace.packet.extend(ts_packets)
    payload = sorted_trace.SerializeToString()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".gz":
        with gzip.open(out_path, "wb", compresslevel=3) as handle:
            handle.write(payload)
    else:
        out_path.write_bytes(payload)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    report: Dict[str, Any] = {
        "schema": "weaveprofiler-perfetto-native-export/v1",
        "source": str(trace_path.resolve()),
        "out": str(out_path.resolve()),
        "bytes": out_path.stat().st_size,
        "sha256": digest,
        "format": "perfetto-native-protobuf",
        "compressed": out_path.suffix == ".gz",
        "semantic_slices": len(semantic),
        "semantic_counts": {
            **{
                name: sum(1 for item in semantic if item["name"] == name)
                for name in ("load", "mma", "store")
            },
            **{
                name: cnt
                for name in ("wait", "barrier")
                for cnt in (sum(1 for item in semantic if item["name"] == name),)
                if cnt > 0
            },
        },
        "gemm_envelopes": 1,
        "source_gemm_spans": len(gemm_events),
        "physical_waves": len(grouped),
        "wave_tracks": len(wave_track_for_identity),
        "hardware_tracks": hardware_track_count,
        "hardware_points": hardware_point_count,
        "hardware": hardware_summary,
        "occupancy": occupancy_summary,
        "preview_k_blocks": list(preview_k_blocks) if preview_k_blocks is not None else None,
        "kernel_window_ns": [kernel_start, kernel_end],
        "notes": [
            "Each physical wave has its own track and canonical semantic labels are load, mma, store.",
            "ROCm PM values are per-dispatch aggregates; the flat band is measured, not an intra-kernel curve.",
            "Software occupancy curves are derived from semantic intervals and are not hardware counters.",
        ],
    }
    return report


def parse_k_blocks(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    result = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        result.append(int(token, 10))
    if not result:
        raise argparse.ArgumentTypeError("--preview-k-blocks must contain at least one integer")
    return sorted(set(result))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="Chrome trace JSON or .json.gz")
    parser.add_argument("--out", required=True, help="output .perfetto-trace or .perfetto-trace.gz")
    parser.add_argument("--preview-k-blocks", default=None,
                        help="comma-separated K blocks for a compact preview; stores and gemm envelope remain")
    parser.add_argument("--no-occupancy", action="store_true", help="omit software-derived active-wave curves")
    args = parser.parse_args(argv)
    try:
        k_blocks = parse_k_blocks(args.preview_k_blocks)
        report = export(Path(args.trace), Path(args.out), preview_k_blocks=k_blocks,
                        occupancy=not args.no_occupancy)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
