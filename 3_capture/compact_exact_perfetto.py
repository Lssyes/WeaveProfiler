#!/usr/bin/env python3
"""Create a lossless-geometry compact Perfetto JSON.gz.

Only redundant top-level keys in phase (layer=phase) X-event args are
removed.  Task-envelope X-event args and every M/C event are preserved.
The script validates the transformation both in memory and after reopening
the generated gzip file.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


GEOMETRY_FIELDS = ("name", "cat", "ph", "pid", "tid", "ts", "dur")
REMOVABLE_FIELDS = (
    "dur_ns",
    "start_ns_raw",
    "end_ns_raw",
    "rank_block",
    "source_name",
    "derived_from",
    "smid_written",
    "physical_smid",
    "track_label",
)
REQUIRED_PHASE_FIELDS = (
    "layer",
    "task_id",
    "rank",
    "block",
    "warp_id",
    "role",
    "semantic_kind",
    "is_derived",
    "expert_idx",
)


def read_trace(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        trace = json.load(handle)
    if not isinstance(trace, dict) or not isinstance(trace.get("traceEvents"), list):
        raise ValueError("输入不是带 traceEvents 数组的 Perfetto JSON 对象")
    return trace


def event_geometry_digest(events: list[dict[str, Any]]) -> str:
    """Hash ordered event geometry, independent of non-geometry args."""
    digest = hashlib.sha256()
    for event in events:
        geometry = [event.get(field) for field in GEOMETRY_FIELDS]
        encoded = json.dumps(
            geometry,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def event_partition(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "X": [event for event in events if event.get("ph") == "X"],
        "C": [event for event in events if event.get("ph") == "C"],
        "M": [event for event in events if event.get("ph") == "M"],
    }


def compact_trace(trace: dict[str, Any], require_tx_counter: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    source_events = trace["traceEvents"]
    source_partitions = event_partition(source_events)
    source_geometry = event_geometry_digest(source_events)
    removed_counts: Counter[str] = Counter()
    compact_events: list[dict[str, Any]] = []

    for source_event in source_events:
        event = source_event
        args = source_event.get("args")
        is_phase = source_event.get("ph") == "X" and isinstance(args, dict) and args.get("layer") == "phase"
        if is_phase:
            compact_args = dict(args)
            for field in REMOVABLE_FIELDS:
                if field in compact_args:
                    del compact_args[field]
                    removed_counts[field] += 1
            event = dict(source_event)
            event["args"] = compact_args
        compact_events.append(event)

    compact = copy.copy(trace)
    compact["traceEvents"] = compact_events
    other_data = dict(trace.get("otherData") or {})
    removed_field_counts = {field: removed_counts[field] for field in REMOVABLE_FIELDS if removed_counts[field]}
    compact["otherData"] = other_data
    other_data["compact_exact"] = {
        "schema": "perfetto-compact-exact/v1",
        "scope": "ph=X and args.layer=phase only",
        "removed_fields": list(removed_field_counts),
        "removed_field_counts": removed_field_counts,
        "removed_count": sum(removed_field_counts.values()),
        "lossless_event_geometry": True,
    }

    compact_partitions = event_partition(compact_events)
    if source_geometry != event_geometry_digest(compact_events):
        raise AssertionError("内存变换破坏了 event geometry")
    if {key: len(value) for key, value in source_partitions.items()} != {
        key: len(value) for key, value in compact_partitions.items()
    }:
        raise AssertionError("X/C/M 数量变化")
    for phase_before, phase_after in zip(source_partitions["X"], compact_partitions["X"]):
        before_args = phase_before.get("args") or {}
        after_args = phase_after.get("args") or {}
        if before_args.get("layer") == "task":
            if canonical_json(before_args) != canonical_json(after_args):
                raise AssertionError("task envelope args 被修改")
        elif before_args.get("layer") == "phase":
            for field in REQUIRED_PHASE_FIELDS:
                if field in before_args and after_args.get(field) != before_args[field]:
                    raise AssertionError(f"phase 验证字段被修改或丢失: {field}")
    for kind in ("C", "M"):
        if canonical_json(source_partitions[kind]) != canonical_json(compact_partitions[kind]):
            raise AssertionError(f"{kind} 事件发生修改")

    task_events = [event for event in compact_partitions["X"] if (event.get("args") or {}).get("layer") == "task"]
    phase_events = [event for event in compact_partitions["X"] if (event.get("args") or {}).get("layer") == "phase"]
    tx_counters = [
        event
        for event in compact_partitions["C"]
        if isinstance(event.get("name"), str)
        and event["name"].startswith("NVLink TX Requests User Data")
    ]
    if require_tx_counter and not tx_counters:
        raise AssertionError("NVLink TX Requests User Data counter 缺失")

    report = {
        "source_event_count": len(source_events),
        "output_event_count": len(compact_events),
        "source_counts": {key: len(value) for key, value in source_partitions.items()},
        "output_counts": {key: len(value) for key, value in compact_partitions.items()},
        "task_count": len(task_events),
        "phase_count": len(phase_events),
        "tx_request_counter_count": len(tx_counters),
        "removed_fields": list(removed_field_counts),
        "removed_field_counts": removed_field_counts,
        "removed_count": sum(removed_field_counts.values()),
        "geometry_sha256_before": source_geometry,
        "geometry_sha256_memory_after": event_geometry_digest(compact_events),
        "lossless_event_geometry": True,
    }
    return compact, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pos", nargs="?", type=Path)
    parser.add_argument("output_pos", nargs="?", type=Path)
    parser.add_argument("--input", dest="input_opt", type=Path)
    parser.add_argument("--output", dest="output_opt", type=Path)
    parser.add_argument("--require-tx-counter", action="store_true")
    args = parser.parse_args()
    input_path = args.input_opt or args.input_pos
    output_path = args.output_opt or args.output_pos
    if input_path is None or output_path is None:
        parser.error("必须提供 --input/--output（或两个兼容 positional 参数）")
    if args.input_opt is not None and args.input_pos is not None:
        parser.error("--input 与 positional input 不能同时提供")
    if args.output_opt is not None and args.output_pos is not None:
        parser.error("--output 与 positional output 不能同时提供")
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有输出: {output_path}")
    trace = read_trace(input_path)
    compact, report = compact_trace(trace, require_tx_counter=args.require_tx_counter)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(compact, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    reopened = read_trace(output_path)
    reopened_events = reopened["traceEvents"]
    reopened_geometry = event_geometry_digest(reopened_events)
    if reopened_geometry != report["geometry_sha256_before"]:
        raise AssertionError("重新打开输出后的 geometry hash 不一致")
    if event_partition(reopened_events)["C"] != event_partition(trace["traceEvents"])["C"]:
        raise AssertionError("重新打开输出后的 C 事件不一致")
    if event_partition(reopened_events)["M"] != event_partition(trace["traceEvents"])["M"]:
        raise AssertionError("重新打开输出后的 M 事件不一致")
    report["geometry_sha256_reopened"] = reopened_geometry
    report["output_bytes"] = output_path.stat().st_size
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
