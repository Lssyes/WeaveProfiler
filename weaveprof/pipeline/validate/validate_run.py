#!/usr/bin/env python3
"""Run-level validator for fast-instrument-v2.

This file deliberately has no project imports.  Every required piece of
evidence is checked explicitly; missing evidence is a FAIL for a required
criterion, never an implicit PASS.
"""
from __future__ import print_function

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import shlex
import sys


CRITERION_IDS = (
    "clock", "write_isolation", "compile_switch", "buffer_capacity",
    "entry_sync", "event_semantics", "capture_window", "multirank_sync",
    "sampling_coverage", "buffer_format", "decoder", "perfetto_conversion",
)
REPAIR_SCOPE = {
    "clock": "device", "write_isolation": "device",
    "compile_switch": "device", "buffer_capacity": "device",
    "entry_sync": "device", "event_semantics": "device",
    "capture_window": "capture", "multirank_sync": "capture",
    "sampling_coverage": "capture", "buffer_format": "host",
    "decoder": "host", "perfetto_conversion": "host",
}
SOFTWARE = set(CRITERION_IDS[:6])
CAPTURE = set(CRITERION_IDS[6:9])
CONVERSION = set(CRITERION_IDS[9:])
DEFAULT_MAX_BUFFER_OCCUPANCY = 0.80


def _result(criterion, status, message, **extra):
    value = {"id": criterion, "status": status, "message": message,
             "repair_scope": REPAIR_SCOPE[criterion]}
    value.update(extra)
    return value


def _fail(criterion, message, **extra):
    return _result(criterion, "FAIL", message, **extra)


def _pass(criterion, message, **extra):
    return _result(criterion, "PASS", message, **extra)


def _missing(criterion, what):
    return _fail(criterion, "缺少必需证据: " + what)


def _load(path):
    if not path:
        return None
    try:
        opener = gzip.open if str(path).lower().endswith((".gz", ".gzip")) else open
        with opener(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return None


def _load_text(path):
    if not path:
        return None
    try:
        opener = gzip.open if str(path).lower().endswith((".gz", ".gzip")) else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def _load_evidence(path):
    value = _load(path)
    if value is not None:
        return value
    return _load_text(path)


def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _first(mapping, names, default=None):
    mapping = _dict(mapping)
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _nonnegative_zero(value):
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    if isinstance(value, dict):
        return all(_nonnegative_zero(item) for item in value.values())
    number = _number(value)
    return number is not None and number == 0.0


def _events(value):
    """Accept raw event lists and the common {events: [...]} wrappers."""
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        return []
    for key in ("events", "records", "decoded_events", "traceEvents"):
        if isinstance(value.get(key), list):
            return value[key]
    return []


def _metadata(value):
    value = _dict(value)
    meta = _dict(value.get("metadata"))
    for key in ("capacity", "buffer_capacity", "max_records", "records_total",
                "record_count", "records", "events_total", "event_count",
                "decoded_events", "overflow", "truncated", "dropped",
                "records_dropped", "truncated_tracks", "record_counts",
                "records_by_track", "buffer_headroom", "tier"):
        if key in value and key not in meta:
            meta[key] = value[key]
    for key in ("summary", "counters", "diagnostics"):
        if isinstance(value.get(key), dict):
            meta.update(value[key])
    return meta


def _field(event, names):
    if not isinstance(event, dict):
        return None
    value = _first(event, names)
    if value is not None:
        return value
    args = _dict(event.get("args"))
    return _first(args, names)


def _cta(event):
    return _field(event, ("cta", "block", "block_id", "cta_id", "pid"))


def _time(event):
    return _number(_field(event, ("ts", "timestamp", "time", "t", "start_ns")))


def _name(event):
    return str(_field(event, ("name", "kind", "type", "event", "semantic_kind")) or "").lower()


def _layer(event):
    layer = _field(event, ("layer",))
    if layer in ("task", "phase", "ignored"):
        return layer
    category = str(_field(event, ("cat", "category")) or "")
    for candidate in ("task", "phase", "ignored"):
        if category.startswith(candidate + "/"):
            return candidate
    return None


def _end_time(event):
    end = _number(_field(event, ("end", "end_ns", "timestamp_end")))
    if end is not None:
        return end
    start = _time(event)
    duration = _number(_field(event, ("dur", "duration", "duration_ns")))
    return start + duration if start is not None and duration is not None else None


def _layout_contract(evidence):
    contract = _dict(evidence.get("contract"))
    events = _dict(evidence.get("events"))
    other = _dict(events.get("otherData"))
    embedded = _first(events, ("layout_contract", "contract"))
    if embedded is None:
        embedded = _first(other, ("layout_contract", "contract"))
    if isinstance(embedded, dict):
        merged = dict(embedded)
        merged.update(contract)
        return merged
    return contract


def _crossing_count(rows):
    lanes = {}
    for row in rows:
        start, end = _time(row), _end_time(row)
        if start is None or end is None or end < start:
            continue
        key = (_cta(row), _field(row, ("track", "track_id", "tid", "lane")))
        lanes.setdefault(key, []).append((start, end, row))
    count = 0
    for values in lanes.values():
        values.sort(key=lambda item: (item[0], -item[1]))
        active = []
        for start, end, row in values:
            active = [item for item in active if item[1] > start]
            for other_start, other_end, other in active:
                if start < other_end < end:
                    count += 1
            active.append((start, end, row))
    return count


def _has_truthy(mapping, names):
    value = _first(mapping, names)
    return bool(value) if value is not None else False


def _find_diagnostics(value):
    """Return all explicitly reported diagnostics, preserving zero checks."""
    source = _dict(value)
    found = []
    for key in ("diagnostics", "pairing_diagnostics", "task_phase_diagnostics",
                "unmatched", "unmatched_tasks", "unmatched_phases",
                "negative_durations", "partial_overlaps", "errors"):
        if key in source:
            found.append((key, source[key]))
    meta = _metadata(value)
    for key in ("diagnostics", "pairing_diagnostics", "task_phase_diagnostics",
                "unmatched", "unmatched_tasks", "unmatched_phases",
                "negative_durations", "partial_overlaps", "errors"):
        if key in meta and (key, meta[key]) not in found:
            found.append((key, meta[key]))
    return found


def check_clock(evidence):
    """Check globaltimer capability and nondecreasing raw timestamps per track."""
    detect = _dict(evidence.get("detect"))
    events = evidence.get("events")
    if not detect:
        return _missing("clock", "--detect")
    capabilities = _dict(detect.get("capabilities"))
    text = json.dumps(detect, ensure_ascii=False).lower()
    explicit_clock = _first(capabilities, ("globaltimer", "device_timestamping"))
    if isinstance(explicit_clock, dict):
        explicit_clock = _first(explicit_clock, ("present", "exists", "available", "valid"))
    present = (explicit_clock is True or
               (explicit_clock is None and "globaltimer" in text))
    if not present:
        return _fail("clock", "detect 未确认 device globaltimer/设备时间戳")
    rows = _events(events)
    if not rows:
        return _missing("clock", "--events 中的时间记录")
    tracks = {}
    bad = []
    for row in rows:
        timestamp = _time(row)
        if timestamp is None:
            bad.append("缺少有效 timestamp")
            continue
        key = (_cta(row), _field(row, ("track", "track_id", "tid", "lane")))
        previous = tracks.get(key)
        if previous is not None and timestamp < previous:
            bad.append("track=%r 时间倒退 %s -> %s" % (key, previous, timestamp))
        tracks[key] = timestamp
    if bad:
        return _fail("clock", "时间戳无效或非单调: " + "; ".join(bad[:4]))
    return _pass("clock", "已确认 globaltimer，%d 条记录按 track 单调" % len(rows))


def check_write_isolation(evidence):
    """Check slot/record intervals are mathematically non-overlapping."""
    abi = _dict(evidence.get("abi"))
    if not abi:
        return _missing("write_isolation", "--abi")
    slots = _list(_first(abi, ("slots", "slot_ranges", "slot_addresses")))
    intervals = []
    for slot in slots:
        if isinstance(slot, dict):
            start = _number(_first(slot, ("offset", "address", "start", "base")))
            size = _number(_first(slot, ("size", "record_size", "bytes")))
            if start is not None and size is not None:
                intervals.append((start, start + size))
        elif isinstance(slot, (list, tuple)) and len(slot) >= 2:
            start, end = _number(slot[0]), _number(slot[1])
            if start is not None and end is not None:
                intervals.append((start, end if end >= start else start + end))
    record_size = _number(_first(abi, ("record_size", "event_record_size", "slot_size")))
    stride = _number(_first(abi, ("slot_stride", "track_stride", "record_stride")))
    count = _number(_first(abi, ("track_count", "tracks", "warp_count", "num_tracks")))
    if not intervals and record_size is not None and stride is not None:
        count = int(count) if count is not None else 2
        intervals = [(i * stride, i * stride + record_size) for i in range(max(1, count))]
    if not intervals:
        return _fail("write_isolation", "ABI 没有可计算的 slot 地址/步长/记录大小")
    intervals.sort()
    for left, right in zip(intervals, intervals[1:]):
        if left[1] > right[0]:
            return _fail("write_isolation", "slot 区间重叠: [%s,%s) 与 [%s,%s)" %
                         (left[0], left[1], right[0], right[1]))
    if record_size is not None and stride is not None and stride < record_size:
        return _fail("write_isolation", "slot_stride=%s 小于 record_size=%s" %
                     (stride, record_size))
    return _pass("write_isolation", "slot 区间共 %d 个，数学上无重叠" % len(intervals))


def attribute_correctness_failure(correctness):
    """把 "插桩改变了结果" 和 "kernel 自己就不可重现" 分开。

    G02 只有一条判据"插桩前后逐元素一致"，但它被两件完全不同的事破坏。只报
    all_passed=false 会把人送去查插桩，而如果未插桩档位自比 (off vs off) 拿不到逐位
    相同的结果，再干净的插桩也不可能让两次运行逐元素相等 —— 要修的是 kernel。

    判据是"比特精确"而不是"容差内通过"：偶尔才突破容差的 kernel，自比时很可能恰好落
    在容差内，那样就会把 baseline 的锅扣到插桩头上。归因一律从实测的 per-tensor 误差
    数值推导，不采信 harness 自己写的 all_passed / failure_attribution 标签。
    """
    baseline = _dict(correctness.get("baseline_self_consistency"))
    trials = [_dict(item) for item in (baseline.get("trials") or [])]
    if not trials:
        return ("correctness.all_passed 不是 true，且没有 baseline_self_consistency.trials"
                "（未插桩档位重跑、off-vs-off 自比）证据，无法区分是插桩改变了输出还是"
                " kernel 自身 run-to-run 不可重现；harness 必须在失败时补测 off-vs-off 再报")
    worst = max((_number(_dict(item).get("max_abs_err")) or 0.0
                 for trial in trials for item in (trial.get("per_tensor") or [])),
                default=0.0)
    if worst == 0.0:
        return ("correctness.all_passed 不是 true，而未插桩档位自比 {} 次全部逐位相同"
                " -> 归因：插桩改变了 kernel 输出，要修的是插桩".format(len(trials)))
    return ("correctness.all_passed 不是 true，且未插桩档位自比就不是逐位相同（{} 次重跑"
            "最大绝对误差 {:.4g}）-> 归因：kernel 自身 run-to-run 不可重现，任何插桩都不"
            "可能稳定通过 G02；先修 kernel 的不确定性，不得改插桩或放宽容差来绕过"
            .format(len(trials), worst))


def _overhead_ceiling_pct(evidence):
    """Two-tier overhead ceiling for G03: task 10%, phase 15%.

    Reads ``capture.granularity`` from the locked contract; anything missing or
    unrecognised falls back to the stricter task ceiling (fail-safe).  The
    phase allowance is additionally cross-checked by the skill-mandatory
    ``overhead_ceiling`` audit, which only grants 15% when the delivered trace
    actually carries phase-layer intervals."""
    contract = _dict(evidence.get("contract"))
    capture = _dict(contract.get("capture"))
    granularity = str(capture.get("granularity") or "").strip().lower()
    return 15.0 if granularity == "phase" else 10.0


def check_compile_switch(evidence):
    detect = _dict(evidence.get("detect"))
    correctness = _dict(evidence.get("correctness"))
    if not detect:
        return _missing("compile_switch", "--detect")
    text = json.dumps(detect, ensure_ascii=False).lower()
    switch = _first(detect, ("compile_switch", "instrumentation_switch", "build_flag"))
    if switch is False:
        return _fail("compile_switch", "detect 明确报告编译开关不存在/关闭")
    if switch is None and not any(word in text for word in
                                  ("instrument", "trace", "profile", "enable")):
        return _fail("compile_switch", "detect 没有确认编译期开关")
    if not correctness:
        return _missing("compile_switch", "--correctness")
    variant_doc = _first(correctness, ("variants", "modes", "runs"))
    variants = variant_doc
    if isinstance(variants, dict):
        variants = list(variants.keys())
    text = json.dumps(correctness, ensure_ascii=False).lower()
    enabled = bool(variants and len(variants) >= 2) or ("enabled" in text and "disabled" in text)
    if not enabled:
        return _fail("compile_switch", "correctness 没有同时给出 enabled/disabled 两个版本")
    correctness_text = json.dumps(correctness, ensure_ascii=False).lower()
    for required_word in ("randn", "full_output", "all_passed"):
        if required_word not in correctness_text:
            return _fail("compile_switch", "correctness 缺少 %s 证据" % required_word)
    all_passed = correctness.get("all_passed")
    if all_passed is not True:
        variant_values = []
        if isinstance(variant_doc, dict):
            variant_values = [_dict(item).get("all_passed") for item in variant_doc.values()]
        elif isinstance(variant_doc, list):
            variant_values = [_dict(item).get("all_passed") for item in variant_doc]
        if not variant_values or not all(item is True for item in variant_values):
            return _fail("compile_switch", attribute_correctness_failure(correctness))
    overhead = _dict(evidence.get("overhead"))
    if not overhead:
        return _missing("compile_switch", "--overhead")
    tier_plan = _first(overhead, ("tier_plan", "tiers", "three_tiers"))
    primary = _dict(_first(overhead, ("primary_metric", "pure_record_overhead")))
    if isinstance(tier_plan, (list, dict)) and len(tier_plan) >= 3:
        within = primary.get("within_threshold", overhead.get("within_threshold"))
        worst = _number(_first(primary, ("worst_pct", "value", "percent")))
        ceiling_pct = _overhead_ceiling_pct(evidence)
        if within is not True or (worst is not None and worst >= ceiling_pct):
            return _fail("compile_switch",
                         "三档 overhead 的 full/anchor 开销未低于 %g%%（按契约 granularity 分档）"
                         % ceiling_pct)
        return _pass("compile_switch", "编译开关、randn/full_output/all_passed 及三档 overhead 均通过")
    values = {}
    for label in ("anchor", "full"):
        aliases = (label, label + "_only", label + "_ms", label + "_us", label + "_ns")
        item = _first(overhead, aliases)
        if isinstance(item, dict):
            item = _first(item, ("time_ms", "latency_ms", "duration_ms", "time", "value"))
        values[label] = _number(item)
    if values["anchor"] is None or values["full"] is None:
        levels = _first(overhead, ("levels", "variants", "runs", "tier_values"))
        if isinstance(levels, list):
            for item in levels:
                item = _dict(item)
                label = str(_first(item, ("name", "level", "mode"), "")).lower()
                value = _number(_first(item, ("time_ms", "latency_ms", "duration_ms", "time", "value")))
                if label in ("anchor", "anchor_only", "full"):
                    if label == "anchor_only":
                        label = "anchor"
                    values[label] = value
    levels = _first(overhead, ("levels", "variants", "runs", "tier_values"))
    if not isinstance(levels, (list, dict)) or len(levels) < 3:
        return _fail("compile_switch", "overhead 必须提供三档测量结果: off、anchor、full")
    if values["anchor"] is None or values["full"] is None or values["anchor"] <= 0:
        return _fail("compile_switch", "overhead 三档中缺少 anchor/full 数值")
    relative = abs(values["full"] - values["anchor"]) / values["anchor"]
    ceiling_frac = _overhead_ceiling_pct(evidence) / 100.0
    if relative >= ceiling_frac:
        return _fail("compile_switch",
                     "full 相对 anchor 的开销 %.2f%% >= %g%%（按契约 granularity 分档）"
                     % (relative * 100.0, ceiling_frac * 100.0))
    return _pass("compile_switch", "编译开关、randn/full_output/all_passed 及 overhead 均通过")


def check_buffer_capacity(evidence):
    detect = _dict(evidence.get("detect"))
    events = _dict(evidence.get("events"))
    if not detect:
        return _missing("buffer_capacity", "--detect")
    capacity = _number(_first(detect, ("capacity", "buffer_capacity", "max_records")))
    meta = _metadata(events)
    if capacity is None:
        capacity = _number(_first(meta, ("capacity", "buffer_capacity", "max_records")))
    if capacity is None:
        return _fail("buffer_capacity", "没有 buffer capacity/max_records")
    count = _number(_first(meta, ("records_total", "record_count", "records", "nonempty_records")))
    tier = str(_first(meta, ("tier", "profile_tier"), "")).strip().lower()
    truncated_tracks = _list(_first(meta, ("truncated_tracks", "truncated_block_tracks")))
    if truncated_tracks:
        return _fail("buffer_capacity", "存在截断的 block/track: %d 个" % len(truncated_tracks))
    record_counts = _first(meta, ("record_counts", "records_by_track"))
    if isinstance(record_counts, list):
        saturated = []
        max_fraction = 0.0
        for item in record_counts:
            item = _dict(item)
            if bool(_first(item, ("truncated", "overflow", "dropped"), False)):
                saturated.append(item)
                continue
            records = _number(_first(item, ("records", "record_count", "count")))
            item_capacity = _number(_first(item, ("max_records", "capacity", "record_capacity")))
            if records is not None and item_capacity is not None and item_capacity > 0:
                fraction = records / item_capacity
                max_fraction = max(max_fraction, fraction)
                if records >= item_capacity and tier not in ("anchor", "anchor_only"):
                    saturated.append(item)
        if saturated:
            return _fail("buffer_capacity", "完整 tier 存在 record_count >= max_records 的泳道: %d 个" % len(saturated))
        limit = float(evidence.get("max_buffer_occupancy", DEFAULT_MAX_BUFFER_OCCUPANCY))
        if tier not in ("anchor", "anchor_only") and max_fraction >= limit:
            return _fail("buffer_capacity", "最大泳道占用率 %.3f >= 门禁 %.3f" % (max_fraction, limit))
    overflow = _first(meta, ("overflow", "truncated", "dropped", "records_dropped"))
    harmless = overflow is None or str(overflow).lower() in ("", "false", "0", "none", "null")
    if not harmless and not _nonnegative_zero(overflow):
        return _fail("buffer_capacity", "记录存在 overflow/truncated/dropped: %r" % overflow)
    if count is not None and count >= capacity and not isinstance(record_counts, list):
        return _fail("buffer_capacity", "records=%s >= capacity=%s" % (count, capacity))
    if count is None and isinstance(record_counts, list) and record_counts:
        return _pass("buffer_capacity", "按每个 block/track 核验 records < max_records 且无截断")
    if count is None:
        return _fail("buffer_capacity", "没有 records_total，无法确认容量未溢出")
    return _pass("buffer_capacity", "records=%s < capacity=%s 且无截断/溢出" % (count, capacity))


def check_entry_sync(evidence):
    rows = _events(evidence.get("events"))
    if not rows:
        return _missing("entry_sync", "--events")
    event_doc = _dict(evidence.get("events"))
    anchor_doc = _first(event_doc, ("anchors", "entry_anchors"))
    per_cta = _first(_dict(anchor_doc), ("per_cta", "ctas", "entries"))
    if isinstance(per_cta, list) and per_cta:
        failures = []
        for item in per_cta:
            item = _dict(item)
            cta = _first(item, ("cta", "block", "block_id", "cta_id"))
            release = _number(_first(item, ("release", "release_ns", "entry_sync_end_ns")))
            anchor = _number(_first(item, ("anchor", "anchor_ns", "first_event_ns")))
            business = _number(_first(item, ("business_first", "first_business_ns", "min_event_ns")))
            if cta is None or release is None or anchor is None:
                failures.append("CTA %r 缺 release/anchor" % cta)
            elif anchor < release or (business is not None and business < release):
                failures.append("CTA %r anchor/业务事件早于 release" % cta)
        if failures:
            return _fail("entry_sync", "; ".join(failures[:5]))
        return _pass("entry_sync", "每个 CTA 的 anchor/业务事件均不早于 release")
    by_cta = {}
    for row in rows:
        cta = _cta(row)
        by_cta.setdefault(cta, []).append(row)
    if not by_cta or None in by_cta:
        return _fail("entry_sync", "事件缺少 CTA 标识")
    failures = []
    for cta, cta_rows in by_cta.items():
        releases = [r for r in cta_rows if any(w in _name(r) for w in ("release", "entry_release"))]
        anchors = [r for r in cta_rows if any(w in _name(r) for w in ("anchor", "start", "entry"))]
        if not releases:
            failures.append("CTA %r 缺 release" % (cta,))
            continue
        release = min((_time(r) for r in releases if _time(r) is not None), default=None)
        if release is None:
            failures.append("CTA %r release 无时间" % (cta,))
            continue
        if not anchors:
            failures.append("CTA %r 缺 anchor" % (cta,))
        for row in anchors:
            timestamp = _time(row)
            if timestamp is None or timestamp < release:
                failures.append("CTA %r anchor 早于 release" % (cta,))
        for row in cta_rows:
            if _name(row) in ("release", "entry_release"):
                continue
            timestamp = _time(row)
            if timestamp is None or timestamp < release:
                failures.append("CTA %r 业务事件早于 release" % (cta,))
    if failures:
        return _fail("entry_sync", "; ".join(failures[:5]))
    return _pass("entry_sync", "每个 CTA 都有 release、anchor，且事件不早于 release")


def check_event_semantics(evidence):
    events = evidence.get("events")
    rows = _events(events)
    if not rows:
        return _missing("event_semantics", "--events")
    diagnostics = _find_diagnostics(events)
    if not diagnostics:
        return _fail("event_semantics", "没有 task/phase 配对诊断字段")
    bad = [(key, value) for key, value in diagnostics if not _nonnegative_zero(value)]
    if bad:
        return _fail("event_semantics", "task/phase 配对诊断非零: %r" % bad[:4])
    tasks = [r for r in rows if _layer(r) == "task"]
    phases = [r for r in rows if _layer(r) == "phase"]
    if not tasks or not phases:
        return _fail("event_semantics", "事件中缺少 task 或 phase")
    contract = _layout_contract(evidence)
    allowed_ids = _first(contract, ("allowed_task_ids", "source_task_ids"))
    allowed_roles = _first(contract, ("allowed_task_roles", "allowed_task_role", "source_task_roles"))
    if allowed_ids is not None:
        allowed_ids = set(str(item) for item in _list(allowed_ids))
    if allowed_roles is not None:
        allowed_roles = set(str(item).lower() for item in _list(allowed_roles))
    bad_tasks = []
    for task in tasks:
        args = _dict(task.get("args"))
        if args.get("is_derived") is True:
            bad_tasks.append("derived task %r" % task.get("name"))
        task_id = _first(task, ("task_id",))
        if task_id is None:
            task_id = _first(args, ("task_id",))
        if allowed_ids is not None and str(task_id) not in allowed_ids:
            bad_tasks.append("task_id=%r" % task_id)
        role = _first(task, ("task_role", "role"))
        if role is None:
            role = _first(args, ("task_role", "role"))
        if allowed_roles is not None and str(role).lower() not in allowed_roles:
            bad_tasks.append("role=%r" % role)
    if bad_tasks:
        return _fail("event_semantics", "task 来源/角色不符合 contract: %s" % ", ".join(bad_tasks[:5]))
    crossing = _crossing_count(tasks + phases)
    if crossing:
        return _fail("event_semantics", "同一 CTA/track 存在 %d 个 task/phase 交叉区间" % crossing)
    required_phases = _first(contract, ("required_phase_kinds", "required_phases"))
    if required_phases:
        names = set(_name(row) for row in phases)
        missing = [str(item).lower() for item in _list(required_phases)
                   if str(item).lower() not in names]
        if missing:
            return _fail("event_semantics", "缺少 contract 要求的 phase: " + ", ".join(missing))
    return _pass("event_semantics", "task/phase 事件存在，配对诊断为 0")


def _lifecycle_rows(value):
    if isinstance(value, dict):
        # capture_harness_reference.py 的真实产物是一个 rank 文档：
        # {rank, status, target_kernel_calls, events: [...] }。保留外层
        # rank/status，不能直接把 events 列表返回，否则 rank 信息会丢失。
        if isinstance(value.get("events"), list) and any(
                key in value for key in ("rank", "global_rank", "status",
                                         "target_kernel_calls")):
            return [value]
        for key in ("ranks", "lifecycle", "events", "rank_lifecycle"):
            if isinstance(value.get(key), list):
                return value[key]
        return []
    return value if isinstance(value, list) else []


def _lifecycle_name(row):
    raw = str(_first(row, ("event", "kind", "name", "type"), "")).lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if raw in ("start", "capture_start", "entry_start"):
        return "start"
    if "barrier_after_start" in raw or raw in ("after_start", "start_barrier"):
        return "barrier_after_start"
    if "target_kernel" in raw or raw in ("target", "kernel", "target_once"):
        return "target_kernel_once"
    if "barrier_before_stop" in raw or raw in ("before_stop", "stop_barrier"):
        return "barrier_before_stop"
    if raw in ("stop", "capture_stop", "entry_stop"):
        return "stop"
    return raw


def _lifecycle_time(row):
    return _number(_first(row, ("time_ns", "timestamp_ns", "time", "timestamp", "ts")))


def _is_true(value):
    return value is True or str(value).lower() in ("true", "1", "yes", "pass", "passed")


def check_capture_window(evidence):
    manifest = _dict(evidence.get("capture_manifest"))
    if not manifest:
        return _missing("capture_window", "--capture-manifest")
    # capture_nsys.py 的真实 manifest：采集由 cudaProfilerApi 开始，
    # 由 stop/stop-shutdown 结束；kernel match_rows 是目标 launch 计数。
    if "capture_range" in manifest or "capture_range_end" in manifest:
        capture_range = str(manifest.get("capture_range", ""))
        capture_end = str(manifest.get("capture_range_end", ""))
        if capture_range.lower() != "cudaprofilerapi":
            return _fail("capture_window", "capture_range=%r，要求 cudaProfilerApi" % capture_range)
        if capture_end.lower() not in ("stop", "stop-shutdown"):
            return _fail("capture_window", "capture_range_end=%r，要求 stop 或 stop-shutdown" % capture_end)
        capture_evidence = _dict(manifest.get("evidence"))
        verdict = _first(manifest, ("verdict",))
        if verdict is None:
            verdict = _first(capture_evidence, ("verdict",))
        if str(verdict).upper() != "PASS":
            return _fail("capture_window", "capture evidence verdict=%r，不是 PASS" % verdict)
        failures = _first(manifest, ("failures",))
        if failures is None:
            failures = _first(capture_evidence, ("failures",))
        if failures not in (None, [], "") and not _nonnegative_zero(failures):
            return _fail("capture_window", "capture evidence failures 非空: %r" % failures)
        kernel = _dict(_first(capture_evidence, ("kernel",)))
        target = _first(kernel, ("match_rows",))
        if target is None:
            by_device = _first(kernel, ("match_rows_by_device",))
            selected = _first(capture_evidence, ("gpu_metrics_device",))
            if selected is None:
                selected = _first(manifest, ("gpu_metrics_device",))
            if isinstance(by_device, dict) and selected is not None:
                target = by_device.get(str(selected), by_device.get(selected))
            elif isinstance(by_device, dict) and len(by_device) == 1:
                target = next(iter(by_device.values()))
            else:
                target = _first(kernel, ("match_rows_total",))
        if _number(target) != 1:
            return _fail("capture_window", "目标 kernel match_rows=%r，要求恰好 1" % target)
        lifecycle_rows = _lifecycle_rows(evidence.get("rank_lifecycle"))
        for rank_doc in lifecycle_rows:
            rank_doc = _dict(rank_doc)
            for key in ("warmup_outside_window", "correctness_outside_window"):
                if key in rank_doc and not _is_true(rank_doc.get(key)):
                    return _fail("capture_window", "rank lifecycle 的 %s 不是 true" % key)
        return _pass("capture_window", "cudaProfilerApi 窗口、证据 verdict 和目标 kernel launch 均通过")
    window = _dict(_first(manifest, ("capture_window", "window", "range")))
    start = _first(window, ("start", "begin", "capture_start"))
    stop = _first(window, ("stop", "end", "capture_stop"))
    if start is None or stop is None:
        start = _first(manifest, ("capture_start", "start"))
        stop = _first(manifest, ("capture_stop", "stop", "end"))
    if start is None or stop is None:
        return _fail("capture_window", "采集 manifest 没有明确 start/stop")
    if _number(stop) is not None and _number(start) is not None and _number(stop) <= _number(start):
        return _fail("capture_window", "capture stop 不晚于 start")
    target = _first(manifest, ("target_launches", "target_launch_count", "launches_target"))
    if isinstance(target, list):
        target = len(target)
    if target is None:
        launches = _first(manifest, ("launches", "kernel_launches"))
        if isinstance(launches, list):
            target = sum(1 for row in launches if _dict(row).get("target", True))
    if _number(target) != 1:
        return _fail("capture_window", "目标 launch 次数=%r，要求恰好 1 次" % target)
    if _first(manifest, ("warmup_in_capture", "residual", "leftover"), False):
        return _fail("capture_window", "采集窗口包含 warmup 或残留 launch")
    lifecycle_rows = _lifecycle_rows(evidence.get("rank_lifecycle"))
    for rank_doc in lifecycle_rows:
        rank_doc = _dict(rank_doc)
        for key in ("warmup_outside_window", "correctness_outside_window"):
            if key in rank_doc and not _is_true(rank_doc.get(key)):
                return _fail("capture_window", "rank lifecycle 的 %s 不是 true" % key)
    return _pass("capture_window", "采集窗口有效，目标 launch 恰好 1 次")


def check_multirank_sync(evidence):
    manifest = _dict(evidence.get("capture_manifest"))
    lifecycle = evidence.get("rank_lifecycle")
    if not lifecycle:
        return _missing("multirank_sync", "--rank-lifecycle")
    rows = _lifecycle_rows(lifecycle)
    if not rows:
        return _fail("multirank_sync", "rank lifecycle 为空")
    expanded = []
    for row in rows:
        row = _dict(row)
        nested = _first(row, ("events", "lifecycle"))
        if isinstance(nested, list):
            rank = _first(row, ("rank", "global_rank", "id"))
            for item in nested:
                item = dict(_dict(item))
                if _first(item, ("rank", "global_rank", "id")) is None:
                    item["rank"] = rank
                # 把 rank 文档级的硬性结论复制到每条事件，便于后面的
                # 分组检查；事件自身字段优先，避免覆盖更具体的信息。
                for key in ("status", "target_kernel_calls", "start_stop_balanced",
                            "warmup_outside_window", "correctness_outside_window",
                            "residual", "leftover", "unclosed", "unmatched"):
                    if key in row and key not in item:
                        item[key] = row[key]
                expanded.append(item)
        else:
            expanded.append(row)
    rows = expanded
    by_rank = {}
    for row in rows:
        row = _dict(row)
        rank = _first(row, ("rank", "global_rank", "id"))
        if rank is None:
            return _fail("multirank_sync", "rank lifecycle 记录缺少 rank")
        by_rank.setdefault(str(rank), []).append(row)
    bad = []
    for row in rows:
        residual = _first(row, ("residual", "leftover", "unclosed", "unmatched"))
        harmless_residual = residual is None or str(residual).lower() in ("", "false", "0", "none", "null")
        if not harmless_residual and not _nonnegative_zero(residual):
            bad.append("rank %r 有残留 lifecycle=%r" %
                       (_first(row, ("rank", "global_rank", "id")), residual))
    for rank, values in by_rank.items():
        metadata = values[0]
        status = _first(metadata, ("status",))
        if str(status).lower() != "completed":
            bad.append("rank %s status=%r，不是 completed" % (rank, status))
        calls = _number(_first(metadata, ("target_kernel_calls",)))
        if calls != 1:
            bad.append("rank %s target_kernel_calls=%r，不是 1" % (rank, calls))
        for key in ("start_stop_balanced", "warmup_outside_window", "correctness_outside_window"):
            if not _is_true(_first(metadata, (key,))):
                bad.append("rank %s %s 不是 true" % (rank, key))

        named = [_lifecycle_name(row) for row in values]
        expected = ["start", "barrier_after_start", "target_kernel_once",
                    "barrier_before_stop", "stop"]
        if named != expected:
            bad.append("rank %s lifecycle=%r，期望=%r" % (rank, named, expected))
        times = [_lifecycle_time(row) for row in values]
        if any(value is None for value in times):
            bad.append("rank %s lifecycle 缺少 time_ns" % rank)
        elif any(left >= right for left, right in zip(times, times[1:])):
            bad.append("rank %s lifecycle 时间未严格递增" % rank)
    expected = _first(manifest, ("ranks", "world_size", "rank_count"))
    if isinstance(expected, list):
        expected = len(expected)
    if expected is not None and int(_number(expected) or 0) != len(by_rank):
        bad.append("rank 数=%d，期望=%s" % (len(by_rank), expected))
    if bad:
        return _fail("multirank_sync", "; ".join(bad[:5]))
    return _pass("multirank_sync", "每个 rank 恰有一次 Start/Stop，顺序正确")


def check_sampling_coverage(evidence):
    metrics = _dict(evidence.get("hw_metrics"))
    if not metrics:
        return _missing("sampling_coverage", "--hw-metrics")
    interval = _dict(_first(metrics, ("sampling_interval_ns", "interval")))
    requested = _number(_first(metrics, ("requested_period_ns", "sampling_period_ns", "period_ns",
                                         "interval_ns", "expected_interval_ns")))
    if requested is None:
        requested = _number(_first(interval, ("expected", "expected_ns", "requested_ns")))
    if requested is None:
        requested = _number(_first(_dict(evidence.get("capture_manifest")),
                                   ("expected_interval_ns", "sampling_period_ns", "requested_period_ns")))
    observed = _number(_first(metrics, ("observed_period_ns", "median_period_ns")))
    if observed is None:
        observed = _number(_first(interval, ("median", "median_ns", "mean")))
    max_gap = _number(_first(metrics, ("max_gap_ns", "maximum_gap_ns", "max_interval_ns")))
    if max_gap is None:
        max_gap = _number(_first(interval, ("max", "max_ns", "p95")))
    if requested is None:
        return _fail("sampling_coverage", "没有 sampling period")
    if observed is None or max_gap is None:
        return _fail("sampling_coverage", "没有 observed period 或 max gap")
    if max_gap > requested * 10.0:
        return _fail("sampling_coverage", "max gap=%s 超过采样周期 10 倍=%s" %
                     (max_gap, requested * 10.0))
    required = _list(_first(metrics, ("required_metrics", "metrics_required", "required")))
    if not required:
        provenance = _dict(_first(metrics, ("metric_provenance", "provenance")))
        required = _list(_first(provenance, ("required", "required_metrics")))
    if not required:
        required = _list(_first(_dict(evidence.get("contract")),
                                ("required_metrics", "hw_required_metrics")))
    samples = _dict(_first(metrics, ("samples_per_metric", "metric_samples", "samples")))
    if not samples and isinstance(metrics.get("counters"), list):
        samples = {str(_first(row, ("name", "metric", "metric_name"))):
                   _first(row, ("samples_in_window", "sample_count", "samples"))
                   for row in metrics["counters"]}
    if not samples and isinstance(metrics.get("counter_reports"), list):
        samples = {str(_first(row, ("name", "metric", "metric_name"))):
                   _first(row, ("samples_in_window", "sample_count", "samples"))
                   for row in metrics["counter_reports"]}
    def sample_count(value):
        if isinstance(value, dict):
            nested = _first(value, ("windows", "samples", "count", "num_samples"))
            return sample_count(nested)
        if isinstance(value, (list, tuple)):
            return len(value)
        return _number(value)
    missing = [str(name) for name in required
               if sample_count(samples.get(str(name), samples.get(name))) in (None, 0)]
    if missing:
        return _fail("sampling_coverage", "required 指标没有窗口样本: " + ", ".join(missing))
    return _pass("sampling_coverage", "required 指标有样本，max gap 未超过 10 倍采样周期")


def _sha256(path):
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def check_buffer_format(evidence):
    abi = _dict(evidence.get("abi"))
    if not abi:
        return _missing("buffer_format", "--abi")
    source = _first(abi, ("source", "source_path", "abi_source", "file"))
    expected = _first(abi, ("source_sha256", "expected_sha256", "sha256", "source_hash"))
    if not source or not expected:
        return _fail("buffer_format", "ABI 必须同时提供 source 路径和 source_sha256")
    source_path = str(source)
    if not os.path.isfile(source_path) and evidence.get("abi_path") and not os.path.isabs(source_path):
        source_path = os.path.join(os.path.dirname(str(evidence["abi_path"])), source_path)
    actual = _sha256(source_path) if os.path.isfile(source_path) else None
    if actual is None:
        content = _first(abi, ("source_content", "content"))
        if content is not None:
            actual = hashlib.sha256(str(content).encode("utf-8")).hexdigest()
    if actual is None:
        return _fail("buffer_format", "ABI source 不存在且没有 source_content: %s" % source)
    if actual.lower() != str(expected).lower():
        return _fail("buffer_format", "ABI source sha256 不匹配: actual=%s expected=%s" %
                     (actual, expected))
    required = ("record_size", "tag", "slot_stride")
    missing = [name for name in required if _first(abi, (name,)) is None]
    if missing:
        return _fail("buffer_format", "ABI 缺少字段: " + ", ".join(missing))
    return _pass("buffer_format", "ABI source sha256 与 buffer 字段均已核对")


def check_decoder(evidence):
    events = evidence.get("events")
    if not events or not _events(events):
        return _missing("decoder", "--events")
    rows = _events(events)
    meta = _metadata(events)
    records = _number(_first(meta, ("records_total", "record_count", "records")))
    decoded = _number(_first(meta, ("events_total", "event_count", "decoded_events")))
    if records is None or decoded is None:
        return _fail("decoder", "events metadata 缺少 records_total/events_total 对账字段")
    if records != decoded * 2:
        return _fail("decoder", "records=%s 与 events=%s 不满足 start+end 对账" % (records, decoded))
    diagnostics = _find_diagnostics(events)
    bad = [(key, value) for key, value in diagnostics if not _nonnegative_zero(value)]
    if bad:
        return _fail("decoder", "解码诊断非零: %r" % bad[:4])
    if len(rows) < int(decoded):
        return _fail("decoder", "解码事件列表数量不足: rows=%d events=%d" % (len(rows), decoded))
    return _pass("decoder", "records/events 数量对账通过，解码诊断为 0")


def _trace_document(trace, trace_path):
    if isinstance(trace, (dict, list)):
        return trace
    if trace_path:
        return _load(trace_path)
    return None


def _check_task_phase_with_deliver_gate(visible):
    """Run the canonical task-phase ownership check used by deliver_gate.py."""
    gate_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deliver_gate.py")
    if not os.path.isfile(gate_path):
        return [("deliver_gate_missing", "缺少 weaveprof/pipeline/validate/deliver_gate.py")], []
    try:
        spec = importlib.util.spec_from_file_location("fast_instrument_deliver_gate", gate_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("无法创建 deliver_gate 模块")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.check_operator_task_shape(visible)
    except Exception as error:
        return [("deliver_gate_error", "%s: %s" % (type(error).__name__, error))], []


def check_perfetto_conversion(evidence):
    trace_path = evidence.get("trace_path")
    trace = _trace_document(evidence.get("trace"), trace_path)
    if trace is None:
        return _missing("perfetto_conversion", "project trace")
    if not isinstance(trace, (dict, list)):
        return _fail("perfetto_conversion", "trace 不是 JSON 对象/数组")
    events = trace if isinstance(trace, list) else trace.get("traceEvents")
    if not isinstance(events, list):
        return _fail("perfetto_conversion", "JSON 缺少 traceEvents")
    bundle = evidence.get("bundle", "sw")
    counters = [row for row in events if _dict(row).get("ph") == "C"]
    if bundle == "hw":
        if not counters:
            return _fail("perfetto_conversion", "纯 hw bundle 的 trace 没有 counter 事件")
        return _pass("perfetto_conversion", "Perfetto JSON/traceEvents 可读，包含硬件 counter trace")
    visible = [row for row in events if _dict(row).get("ph") == "X"]
    summary = _first(_dict(trace), ("num_x_measured", "num_x", "visible_event_count"))
    if summary is None:
        for row in events:
            if str(_dict(row).get("name", "")).lower() in ("summary", "trace_summary"):
                summary = _first(_dict(row).get("args"), ("num_x_measured", "num_x"))
                if summary is not None:
                    break
    if _number(summary) is None:
        return _fail("perfetto_conversion", "trace summary 缺少 num_x_measured")
    if int(_number(summary)) != len(visible):
        return _fail("perfetto_conversion", "可见 X 事件=%d 与 summary num_x_measured=%s 不一致" %
                     (len(visible), summary))
    if not visible:
        return _fail("perfetto_conversion", "traceEvents 没有可见 X 事件")
    missing_replica = []
    bad_replica = []
    for row in visible:
        args = _dict(row.get("args"))
        replica = _first(row, ("replica",))
        if replica is None:
            replica = _first(args, ("replica",))
        if replica is None and "is_replica" in args:
            replica = 1 if args.get("is_replica") else 0
        if replica is None:
            missing_replica.append(row.get("name"))
        elif _number(replica) != 0:
            bad_replica.append((row.get("name"), replica))
    if missing_replica:
        return _fail("perfetto_conversion", "X 事件缺 replica=0: %r" % missing_replica[:5])
    if bad_replica:
        return _fail("perfetto_conversion", "存在 replica 非 0: %r" % bad_replica[:5])
    contract = _layout_contract(evidence)
    if isinstance(trace, dict):
        trace_contract = _first(trace, ("layout_contract", "contract"))
        trace_other = _dict(trace.get("otherData"))
        if trace_contract is None:
            trace_contract = _first(trace_other, ("layout_contract", "contract"))
        if isinstance(trace_contract, dict):
            merged_contract = dict(trace_contract)
            merged_contract.update(contract)
            contract = merged_contract
    bad_layers = [row.get("name") for row in visible if _layer(row) not in ("task", "phase")]
    if bad_layers:
        return _fail("perfetto_conversion", "可见 X 事件缺少 task/phase layer: %r" % bad_layers[:5])
    forbidden = ("wait", "barrier", "sync", "acquire", "release")
    forbidden_events = [row.get("name") for row in visible
                        if any(word in _name(row) for word in forbidden)]
    if forbidden_events:
        return _fail("perfetto_conversion", "可见时间线包含不应输出的同步事件: %r" % forbidden_events[:5])
    crossing = _crossing_count(visible)
    if crossing:
        return _fail("perfetto_conversion", "Perfetto 同一 lane 有 %d 个交叉区间" % crossing)
    allowed_ids = _first(contract, ("allowed_task_ids", "source_task_ids"))
    allowed_roles = _first(contract, ("allowed_task_roles", "allowed_task_role", "source_task_roles"))
    if allowed_ids is not None or allowed_roles is not None:
        allowed_ids = set(str(item) for item in _list(allowed_ids)) if allowed_ids is not None else None
        allowed_roles = set(str(item).lower() for item in _list(allowed_roles)) if allowed_roles is not None else None
        for row in visible:
            if _layer(row) != "task":
                continue
            args = _dict(row.get("args"))
            task_id = _first(row, ("task_id",))
            task_id = task_id if task_id is not None else _first(args, ("task_id",))
            role = _first(row, ("task_role", "role"))
            role = role if role is not None else _first(args, ("task_role", "role"))
            if allowed_ids is not None and str(task_id) not in allowed_ids:
                return _fail("perfetto_conversion", "task_id=%r 不在 contract.allowed_task_ids" % task_id)
            if allowed_roles is not None and str(role).lower() not in allowed_roles:
                return _fail("perfetto_conversion", "role=%r 不在 contract.allowed_task_roles" % role)
    task_phase_fails, task_phase_notes = _check_task_phase_with_deliver_gate(visible)
    if task_phase_fails:
        return _fail("perfetto_conversion", "task-phase 检查失败: %s" %
                     "; ".join("%s: %s" % item for item in task_phase_fails[:4]),
                     task_phase_fail_ids=[item[0] for item in task_phase_fails])
    return _pass("perfetto_conversion",
                 "Perfetto 可读，事件对账、replica 和 task-phase 均通过",
                 task_phase_notes=task_phase_notes)


def _required_for(bundle):
    if bundle == "sw":
        return SOFTWARE | CONVERSION
    if bundle == "hw":
        # 纯硬件 bundle 只验证采集三项和最终 counter trace；它不要求
        # 软件 buffer 的 ABI、解码 events 或软件 task/phase 证据。
        return CAPTURE | {"perfetto_conversion"}
    return set(CRITERION_IDS)


def _not_required(criterion, bundle):
    return _result(criterion, "NOT_REQUIRED", "bundle=%s 不要求该类证据" % bundle)


def _not_applicable(criterion, bundle):
    """SPEC-4: hardware-capture judgement is inapplicable to a SW-only delivery.

    Distinct from PASS/FAIL and from NOT_REQUIRED: N/A means the evidence class
    can never exist for this bundle, so it neither blocks nor counts as passing.
    """
    return _result(criterion, "N/A",
                   "bundle=%s 为软件交付，硬件采集类判据不适用 (N/A)" % bundle)


def _write_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    if path and path != "-":
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    else:
        print(payload)


def _build_parser():
    parser = argparse.ArgumentParser(description="验证 fast-instrument 十二项运行证据")
    parser.add_argument("project", nargs="?")
    parser.add_argument("trace", nargs="?")
    parser.add_argument("--bundle", choices=("sw", "hw", "sw+hw"))
    parser.set_defaults(bundle="sw+hw")
    parser.add_argument(
        "--max-buffer-occupancy", type=float, default=DEFAULT_MAX_BUFFER_OCCUPANCY,
        help="完整 tier 最大单槽占用率上限，达到即失败")
    parser.add_argument("--detect")
    parser.add_argument("--abi")
    parser.add_argument("--events", action="append", default=[])
    parser.add_argument("--correctness")
    parser.add_argument("--overhead")
    parser.add_argument("--contract")
    parser.add_argument("--capture-manifest")
    parser.add_argument("--hw-metrics")
    parser.add_argument("--rank-lifecycle")
    parser.add_argument("--json", default="-")
    parser.add_argument("--selftest", action="store_true")
    return parser


def _context(args):
    event_values = [_load_evidence(path) for path in args.events]
    if len(event_values) == 1:
        events = event_values[0]
    else:
        merged_metadata = {"record_counts": [], "truncated_tracks": []}
        merged_rows = []
        for value in event_values:
            merged_rows.extend(_events(value))
            meta = _metadata(value)
            if isinstance(meta.get("record_counts"), list):
                merged_metadata["record_counts"].extend(meta["record_counts"])
            if isinstance(meta.get("truncated_tracks"), list):
                merged_metadata["truncated_tracks"].extend(meta["truncated_tracks"])
            for key in ("max_records", "tier", "buffer_headroom"):
                if key in meta:
                    merged_metadata[key] = meta[key]
        events = {"metadata": merged_metadata, "events": merged_rows}
    return {
        "detect": _load_evidence(args.detect), "abi": _load_evidence(args.abi),
        "abi_path": args.abi,
        "events": events, "correctness": _load_evidence(args.correctness),
        "overhead": _load_evidence(args.overhead), "contract": _load_evidence(args.contract),
        "capture_manifest": _load_evidence(args.capture_manifest),
        "hw_metrics": _load_evidence(args.hw_metrics),
        "rank_lifecycle": _load_evidence(args.rank_lifecycle),
        "max_buffer_occupancy": args.max_buffer_occupancy,
        "trace": _load(args.trace), "trace_path": args.trace,
    }


def _run(args, ctx):
    functions = {
        "clock": check_clock, "write_isolation": check_write_isolation,
        "compile_switch": check_compile_switch, "buffer_capacity": check_buffer_capacity,
        "entry_sync": check_entry_sync, "event_semantics": check_event_semantics,
        "capture_window": check_capture_window, "multirank_sync": check_multirank_sync,
        "sampling_coverage": check_sampling_coverage, "buffer_format": check_buffer_format,
        "decoder": check_decoder, "perfetto_conversion": check_perfetto_conversion,
    }
    ctx["bundle"] = args.bundle
    required = _required_for(args.bundle)
    criteria = []
    for criterion in CRITERION_IDS:
        if criterion not in required:
            if args.bundle == "sw" and criterion in CAPTURE:
                criteria.append(_not_applicable(criterion, args.bundle))
            else:
                criteria.append(_not_required(criterion, args.bundle))
        else:
            criteria.append(functions[criterion](ctx))
    fails = [{"id": row["id"], "message": row["message"],
              "repair_scope": row["repair_scope"]}
             for row in criteria if row["status"] == "FAIL"]
    not_applicable = [row["id"] for row in criteria if row["status"] == "N/A"]
    return {
        "project": args.project, "trace": args.trace, "bundle": args.bundle,
        "criteria": criteria, "fails": fails, "all_passed": not fails,
        "not_applicable": not_applicable,
        "expect": {"bundle": args.bundle, "required_criteria": sorted(required),
                   "pass_condition": "所有 required_criteria 的 status 必须为 PASS"},
        "rerun_command": " ".join(shlex.quote(str(value)) for value in [sys.executable] + sys.argv),
    }


def _selftest():
    expected_routes = {
        "sw": SOFTWARE | CONVERSION,
        "hw": CAPTURE | {"perfetto_conversion"},
        "sw+hw": set(CRITERION_IDS),
    }
    for bundle, expected in expected_routes.items():
        actual = _required_for(bundle)
        if actual != expected:
            raise AssertionError("bundle route mismatch for %s: %r != %r" %
                                 (bundle, actual, expected))
    source = "record_size=16; tag=uint64; slot_stride=32"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    good_events = {
        "events": [
            {"cta": 0, "track": 0, "name": "release", "ts": 10},
            {"cta": 0, "track": 0, "name": "anchor", "ts": 11},
            {"cta": 0, "track": 0, "name": "task", "layer": "task", "ts": 12, "dur": 1},
            {"cta": 0, "track": 0, "name": "phase", "layer": "phase", "ts": 13, "dur": 1},
        ],
        "records_total": 8, "events_total": 4, "capacity": 16,
        "task_phase_diagnostics": 0,
    }
    trace = {"num_x_measured": 1, "traceEvents": [{"ph": "X", "name": "task",
             "layer": "task", "ts": 1, "dur": 2, "pid": 0, "tid": 1,
             "args": {"replica": 0, "layer": "task", "task_id": 0,
                      "is_derived": False}}]}
    ctx = {
        "detect": {"capabilities": {"globaltimer": True}, "buffer_capacity": 16,
                    "compile_switch": "TRACE_ENABLE"},
        "abi": {"source_content": source, "source_sha256": digest,
                "source": "<memory>",
                "record_size": 16, "tag": "u64", "slot_stride": 32,
                "tracks": 2},
        "events": good_events,
        "correctness": {"input_distribution": "randn", "comparison_scope": "full_output",
                         "all_passed": True,
                         "variants": {"enabled": {"all_passed": True}, "disabled": {"all_passed": True}}},
        "overhead": {"tier_plan": [{"name": "off"}, {"name": "anchor"}, {"name": "full"}],
                      "primary_metric": {"within_threshold": True, "worst_pct": 5.0}},
        "capture_manifest": {
            "capture_range": "cudaProfilerApi", "capture_range_end": "stop",
            "gpu_metrics_device": 0,
            "evidence": {"verdict": "PASS", "failures": [],
                          "kernel": {"match_rows": 1}},
            "verdict": "PASS",
        },
        "rank_lifecycle": [{
            "rank": 0, "status": "completed", "target_kernel_calls": 1,
            "start_stop_balanced": True, "warmup_outside_window": True,
            "correctness_outside_window": True,
            "events": [
                {"name": "Start", "time_ns": 1},
                {"name": "barrier_after_start", "time_ns": 2},
                {"name": "target_kernel_once", "time_ns": 3},
                {"name": "barrier_before_stop", "time_ns": 4},
                {"name": "Stop", "time_ns": 5},
            ],
        }],
        "hw_metrics": {"requested_period_ns": 10, "observed_period_ns": 10,
                        "max_gap_ns": 20, "required_metrics": ["SMs Active"],
                        "samples_per_metric": {"SMs Active": 2}},
        "trace": trace, "trace_path": None,
        "bundle": "sw",
    }
    called = []
    funcs = [check_clock, check_write_isolation, check_compile_switch, check_buffer_capacity,
             check_entry_sync, check_event_semantics, check_capture_window, check_multirank_sync,
             check_sampling_coverage, check_buffer_format, check_decoder, check_perfetto_conversion]
    for function in funcs:
        called.append(function.__name__)
        result = function(ctx)
        if result["status"] != "PASS":
            raise AssertionError(function.__name__ + ": " + result["message"])
    # Two-tier overhead ceiling (G03): task stays 10%, phase granularity earns
    # 15%; the skill-side mandatory audit additionally requires real phase
    # intervals in the trace before granting the wider ceiling.
    tiered = dict(ctx)
    tiered["overhead"] = {"tier_plan": [{"name": "off"}, {"name": "anchor"}, {"name": "full"}],
                           "primary_metric": {"within_threshold": True, "worst_pct": 12.0}}
    if check_compile_switch(tiered)["status"] != "FAIL":
        raise AssertionError("12% overhead under task granularity must fail the 10% ceiling")
    tiered["contract"] = {"capture": {"granularity": "phase"}}
    if check_compile_switch(tiered)["status"] != "PASS":
        raise AssertionError("12% overhead under phase granularity must pass the 15% ceiling")
    tiered["overhead"] = {"tier_plan": [{"name": "off"}, {"name": "anchor"}, {"name": "full"}],
                           "primary_metric": {"within_threshold": True, "worst_pct": 16.0}}
    if check_compile_switch(tiered)["status"] != "FAIL":
        raise AssertionError("16% overhead under phase granularity must fail the 15% ceiling")
    bad = dict(ctx)
    bad["hw_metrics"] = dict(ctx["hw_metrics"], max_gap_ns=101)
    if check_sampling_coverage(bad)["status"] != "FAIL":
        raise AssertionError("bad sampling sample did not fail")
    bad_capture = dict(ctx)
    bad_capture["capture_manifest"] = {
        "capture_range": "cudaProfilerApi", "capture_range_end": "stop",
        "gpu_metrics_device": 0,
        "evidence": {"verdict": "PASS", "failures": [],
                      "kernel": {"match_rows": 2}},
        "verdict": "PASS",
    }
    if check_capture_window(bad_capture)["status"] != "FAIL":
        raise AssertionError("bad capture manifest did not fail")
    bad_lifecycle = dict(ctx)
    bad_lifecycle["rank_lifecycle"] = [dict(ctx["rank_lifecycle"][0], target_kernel_calls=2)]
    if check_multirank_sync(bad_lifecycle)["status"] != "FAIL":
        raise AssertionError("bad lifecycle did not fail")
    # A failing G02 must say WHICH side is broken.  Without this the operator is
    # always sent to debug the instrumentation, even when the kernel is simply not
    # reproducible run-to-run and no instrumentation could ever pass.  The third case
    # is the trap: a baseline that stays inside the tolerance but is not bit-identical
    # still makes the whole run-to-run comparison meaningless.
    for baseline, needle in (
            (None, "baseline_self_consistency.trials"),
            ({"trials": [{"all_passed": True, "per_tensor": [{"max_abs_err": 0.0}]},
                         {"all_passed": True, "per_tensor": [{"max_abs_err": 0.0}]}]},
             "要修的是插桩"),
            ({"trials": [{"all_passed": True, "per_tensor": [{"max_abs_err": 0.0156}]},
                         {"all_passed": True, "per_tensor": [{"max_abs_err": 0.0}]}]},
             "run-to-run 不可重现"),
            ({"trials": [{"all_passed": False, "per_tensor": [{"max_abs_err": 0.0625}]}]},
             "run-to-run 不可重现")):
        broken = dict(ctx)
        entry = {"input_distribution": "randn", "comparison_scope": "full_output",
                 "all_passed": False,
                 "variants": {"enabled": {"all_passed": False},
                              "disabled": {"all_passed": True}}}
        if baseline is not None:
            entry["baseline_self_consistency"] = baseline
        broken["correctness"] = entry
        result = check_compile_switch(broken)
        if result["status"] != "FAIL":
            raise AssertionError("failing correctness did not fail G02: %r" % (baseline,))
        if needle not in result["message"]:
            raise AssertionError("G02 归因缺少 %r: %s" % (needle, result["message"]))
    duplicate_task = dict(ctx)
    duplicate_task["trace"] = {
        "num_x_measured": 2,
        "traceEvents": [
            {"ph": "X", "name": "gemm", "layer": "task", "ts": 1, "dur": 4,
             "args": {"replica": 0, "layer": "task", "operator_id": "gemm",
                      "operator_scope": "rank0_block0"}},
            {"ph": "X", "name": "gemm", "layer": "task", "ts": 1, "dur": 4,
             "args": {"replica": 0, "layer": "task", "operator_id": "gemm",
                      "operator_scope": "rank0_block0"}},
        ],
    }
    if check_perfetto_conversion(duplicate_task)["status"] != "FAIL":
        raise AssertionError("duplicate task-phase sample did not fail")
    return {"selftest": "PASS", "called": called,
            "bad_samples": {"sampling": "FAIL as expected",
                              "capture_manifest": "FAIL as expected",
                              "rank_lifecycle": "FAIL as expected",
                              "correctness_attribution": "FAIL with attribution as expected",
                              "task_phase": "FAIL as expected"}}


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.selftest:
        result = _selftest()
        result["expect"] = {"selftest": "十二个 check_* 函数都被调用，且坏样例必须 FAIL"}
        result["rerun_command"] = " ".join(shlex.quote(str(value)) for value in [sys.executable] + sys.argv)
        _write_json(args.json, result)
        return 0
    if not args.project or not args.trace or not args.bundle:
        parser.error("正常校验必须提供 project trace --bundle sw|hw|sw+hw")
    result = _run(args, _context(args))
    _write_json(args.json, result)
    return 0 if result["all_passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
