#!/usr/bin/env python3
"""Perfetto 泳道布局契约：出图器与校验器共用的单一事实来源。

## 为什么需要它

Perfetto 的 `pid` / `tid` 是**位置性**的：它们的含义只存在于出图器和校验器共享的口头
约定里，trace 文件本身不自述。后果是实测踩到过的两类静默错误——

* 出图器把"每个 CTA 只有一条 warp 泳道"画成了合法文件（同角色 warp 只让代表 warp 打点，
  出图时忘了复制），区间数对账通过、Perfetto 能打开，但粒度是错的；
* 校验器 `validate_capability.py` 把 `pid` 当 `(rank, block)` 数，出图器改成按 SM 分组后
  它的 `--expect-blocks` 语义直接失效，却不会报错，只会给出一个无意义的比较。

两者同源：**布局是契约，而契约当时只以 docstring 散文的形式存在。**

## 做法

契约是一份显式的 dict，出图器把它作为 metadata 事件写进 trace，校验器再从 trace 里读回来
按它逐条验。于是 trace 携带指向自身所依据契约的指针——与本 skill 对 ABI 的"溯源链闭合"
要求同一个思路。

契约缺字段、字段自相矛盾，一律在出图**之前**抛错，绝不产出"貌似合理"的图。

## 自证

`--selftest` 会为每条不变量各构造一份只违反该条的 trace，断言校验器必须 FAIL；再构造一份
合规 trace 断言 PASS；再把契约逐字段改坏，断言必须拒绝。**校验器没被坏产物验过，它就只是
装饰。**
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pathlib
import sys
from collections import defaultdict

SCHEMA = "perfetto-layout-contract/v1"
METADATA_EVENT_NAME = "perfetto_layout_contract"

# 契约必须齐备的字段。少一个就拒绝出图——这是本模块存在的理由。
REQUIRED_KEYS = (
    "schema",              # 版本，用于将来兼容
    "group_by",            # 事件字段名 -> process（一个 process 一个大泳道组）
    "group_name_format",   # process_name 模板，必须含 {value}
    "group_sort_index_base",
    "lane_by",             # 事件字段名列表 -> thread（组内一条泳道）
    "lane_name_format",    # thread_name 模板，占位符与 lane_by 等长
    # 粒度检查与显示分组解耦：按 SM 分组时每个 SM 承载的 CTA 数天然不等。
    # 物理 warp 拓扑由 thread_name metadata 声明；只有 observer warp 承载实测 X，
    # 未观测 warp 保持空泳道。V2 检查 metadata 是否声明足量泳道，禁止靠复制 X 补齐。
    "cta_field",           # span.args 里代表 CTA 的字段名（如 "block"）
    "warp_field",          # span.args 里代表 warp 的字段名（如 "warp"）
    "physical_warps_per_cta",  # runtime launch 的物理 warp 数
    "observer_lanes_per_cta",  # ABI observer lane 数
    "outer_layer",         # 粗区间的 layer 值
    "inner_layer",         # 细区间的 layer 值
    "layer_from_cat_prefix",  # True: cat 形如 "<layer>/<role>"
    "counter_must_be_topmost",
    "source_events",       # 溯源：解码产物路径
    "source_abi",          # 溯源：ABI 路径
    "allowed_task_ids",
    "allowed_task_role",
    "task_source_measured_only",
    "task_required_phase_kind",
    "allow_unparented_phase",
    "slice_overlap_policy",
    "observer_spans_source_only",
    "allow_empty_warp_lanes",
)


def validate_contract(contract: dict) -> None:
    """契约自身的完整性与自洽性。任何问题都在出图前抛错。"""
    if not isinstance(contract, dict):
        raise SystemExit("[contract] 契约必须是 dict")
    missing = [k for k in REQUIRED_KEYS if k not in contract]
    if missing:
        raise SystemExit(f"[contract] 缺字段 {missing}；拒绝出图。")
    if contract["schema"] != SCHEMA:
        raise SystemExit(
            f"[contract] schema={contract['schema']!r}，本模块只认 {SCHEMA!r}。")
    if "{value}" not in str(contract["group_name_format"]):
        raise SystemExit("[contract] group_name_format 必须含 {value} 占位符。")
    lane_by = contract["lane_by"]
    if not isinstance(lane_by, list) or not lane_by:
        raise SystemExit("[contract] lane_by 必须是非空列表。")
    fmt = str(contract["lane_name_format"])
    if fmt.count("{") != len(lane_by):
        raise SystemExit(
            f"[contract] lane_name_format 的占位符数({fmt.count('{')}) "
            f"必须等于 lane_by 的长度({len(lane_by)})。")
    if contract["outer_layer"] != "task" or contract["inner_layer"] != "phase":
        raise SystemExit("[contract] outer_layer/inner_layer 必须恰为 task/phase；"
                         "禁止反向或同层契约。")
    physical = contract["physical_warps_per_cta"]
    observers = contract["observer_lanes_per_cta"]
    if not isinstance(physical, int) or physical < 1:
        raise SystemExit("[contract] physical_warps_per_cta 必须是正整数。")
    if not isinstance(observers, int) or observers < 1 or observers > physical:
        raise SystemExit("[contract] observer_lanes_per_cta 必须是 <= physical 的正整数。")
    representation = contract.get("observer_representation", [])
    if observers < physical:
        if not isinstance(representation, list):
            raise SystemExit("[contract] physical warp 缺少 observer_representation 例外声明。")
        for index, item in enumerate(representation):
            if not isinstance(item, dict) or item.get("policy") not in (
                    "hopper_mma_warpgroup", "blackwell_epilogue_warpgroup"):
                raise SystemExit(
                    f"[contract] observer_representation[{index}] 仅允许 Hopper MMA warpgroup "
                    "或 Blackwell epilogue warpgroup。")
            span = item.get("represents")
            if (not isinstance(span, list) or len(span) != 2
                    or not all(isinstance(value, int) and not isinstance(value, bool)
                               for value in span) or span[1] < span[0]):
                raise SystemExit(f"[contract] observer_representation[{index}].represents 无效。")
            role = str(item.get("role", "")).lower()
            architecture = str(item.get("architecture", "")).lower()
            warpgroup = item.get("warpgroup") is True
            policy = item.get("policy")
            valid_policy = (
                policy == "hopper_mma_warpgroup" and "hopper" in architecture
                and (("mma" in role and "warpgroup" in role)
                     or (warpgroup and "math" in role))
            ) or (
                policy == "blackwell_epilogue_warpgroup" and "blackwell" in architecture
                and (("epilogue" in role and "warpgroup" in role)
                     or (warpgroup and "epilogue" in role))
            )
            if not valid_policy:
                raise SystemExit(
                    f"[contract] observer_representation[{index}] policy/role/architecture 不匹配。")
    if (not isinstance(contract["allowed_task_ids"], list)
            or not all(isinstance(item, int) and not isinstance(item, bool)
                       for item in contract["allowed_task_ids"])):
        raise SystemExit("[contract] allowed_task_ids 必须是整数列表。")
    if (contract["allowed_task_role"] is not None
            and not isinstance(contract["allowed_task_role"], str)):
        raise SystemExit("[contract] allowed_task_role 必须是字符串或 None。")
    if not isinstance(contract["task_source_measured_only"], bool):
        raise SystemExit("[contract] task_source_measured_only 必须是 bool。")
    if (contract["task_required_phase_kind"] is not None
            and not isinstance(contract["task_required_phase_kind"], str)):
        raise SystemExit("[contract] task_required_phase_kind 必须是字符串或 None。")
    phase_by_task = contract.get("task_required_phase_by_id", {})
    if not isinstance(phase_by_task, dict):
        raise SystemExit("[contract] task_required_phase_by_id 必须是 dict。")
    for task_id, phase_kind in phase_by_task.items():
        try:
            int(task_id)
        except (TypeError, ValueError):
            raise SystemExit(
                f"[contract] task_required_phase_by_id 的 key={task_id!r} 不是 task id。")
        if phase_kind is not None and not isinstance(phase_kind, str):
            raise SystemExit(
                f"[contract] task_required_phase_by_id[{task_id!r}] 必须是字符串或 None。")
    if contract["allow_unparented_phase"] is not True:
        raise SystemExit("[contract] allow_unparented_phase 必须为 true。")
    if contract["slice_overlap_policy"] != "disjoint_or_nested":
        raise SystemExit("[contract] slice_overlap_policy 必须为 'disjoint_or_nested'。")
    if contract["observer_spans_source_only"] is not True:
        raise SystemExit("[contract] observer_spans_source_only 必须为 true。")
    if contract["allow_empty_warp_lanes"] is not True:
        raise SystemExit("[contract] allow_empty_warp_lanes 必须为 true。")


def metadata_event(contract: dict, pid: int = 0) -> dict:
    """出图器把契约写进 trace 的那条 metadata 事件。"""
    validate_contract(contract)
    return {"ph": "M", "name": METADATA_EVENT_NAME, "pid": pid, "tid": 0,
            "args": {"contract": contract}}


def load_trace(path: pathlib.Path) -> dict:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    doc = json.loads(raw)
    if "traceEvents" not in doc:
        raise SystemExit(f"{path}: 不是 chrome/perfetto trace（无 traceEvents）")
    return doc


def contract_from_trace(doc: dict) -> dict | None:
    for e in doc["traceEvents"]:
        if e.get("ph") == "M" and e.get("name") == METADATA_EVENT_NAME:
            return e.get("args", {}).get("contract")
    return None


def _layer(event: dict) -> str | None:
    args = event.get("args") or {}
    if args.get("layer") in ("task", "phase", "ignored"):
        return args["layer"]
    category = str(event.get("cat", ""))
    return category.split("/", 1)[0] if "/" in category else None


def _bounds_ns(event: dict) -> tuple[int | float, int | float, bool]:
    args = event.get("args") or {}
    start = args.get("start_ns_raw")
    end = args.get("end_ns_raw")
    if (isinstance(start, int) and not isinstance(start, bool)
            and isinstance(end, int) and not isinstance(end, bool)):
        return start, end, True
    start = float(event.get("ts", 0.0)) * 1000.0
    return start, start + float(event.get("dur", 0.0)) * 1000.0, False


def timestamp_failures(events: list[dict], sample_limit: int = 4) -> list[str]:
    """Validate displayed timestamps; positive raw ns must never hide invalid Perfetto ts."""
    invalid_count = 0
    negative_count = 0
    invalid_samples: list[dict] = []
    negative_samples: list[dict] = []
    for index, event in enumerate(events):
        if "ts" not in event:
            continue
        value = event.get("ts")
        sample = {"index": index, "ph": event.get("ph"), "name": event.get("name"),
                  "pid": event.get("pid"), "tid": event.get("tid"), "ts": value}
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            invalid_count += 1
            if len(invalid_samples) < sample_limit:
                invalid_samples.append(sample)
        elif float(value) < 0.0:
            negative_count += 1
            if len(negative_samples) < sample_limit:
                negative_samples.append(sample)
    fails: list[str] = []
    if invalid_count:
        fails.append(f"V6 [timeline_invalid_timestamp]: {invalid_count} timed events "
                     f"have non-numeric/non-finite ts; samples={invalid_samples}")
    if negative_count:
        fails.append(f"V6 [timeline_negative_timestamp]: {negative_count} timed events "
                     f"have ts<0 and Perfetto will drop them; samples={negative_samples}")
    return fails


def _contains(outer: dict, inner: dict) -> bool:
    os_, oe, oraw = _bounds_ns(outer)
    ins, ine, iraw = _bounds_ns(inner)
    if oraw and iraw:
        return os_ <= ins and ine <= oe
    epsilon_ns = 1e-3
    return os_ <= ins + epsilon_ns and ine <= oe + epsilon_ns


def build_interval_end_index(events, bounds_fn):
    """Build a range-min index keyed by interval start in O(N log N)."""
    items = []
    for event in events:
        start, end, exact = bounds_fn(event)
        if end >= start:
            items.append((start, end, exact))
    items.sort(key=lambda item: (item[0], item[1]))
    starts = [item[0] for item in items]
    size = 1
    while size < len(items):
        size <<= 1
    tree = [(float("inf"), False)] * (2 * size)
    for offset, (_, end, exact) in enumerate(items):
        tree[size + offset] = (end, exact)
    for node in range(size - 1, 0, -1):
        tree[node] = min(tree[node * 2], tree[node * 2 + 1],
                         key=lambda item: item[0])
    return starts, tree, size


def interval_index_has_contained(index, outer, bounds_fn):
    """Return whether outer contains an indexed interval in O(log N)."""
    import bisect
    starts, tree, size = index
    if not starts:
        return False
    outer_start, outer_end, outer_exact = bounds_fn(outer)
    epsilon_ns = 1e-3
    if outer_exact:
        # Do not mix 64-bit raw ns with float epsilon: values near 1e18 lose low bits.
        left = bisect.bisect_left(starts, outer_start)
        right = bisect.bisect_right(starts, outer_end)
    else:
        left = bisect.bisect_left(starts, outer_start - epsilon_ns)
        right = bisect.bisect_right(starts, outer_end + epsilon_ns)
    if left >= right:
        return False
    best = (float("inf"), False)
    left += size
    right += size
    while left < right:
        if left & 1:
            if tree[left][0] < best[0]:
                best = tree[left]
            left += 1
        if right & 1:
            right -= 1
            if tree[right][0] < best[0]:
                best = tree[right]
        left >>= 1
        right >>= 1
    if outer_exact and best[1]:
        return best[0] <= outer_end
    return best[0] <= outer_end + epsilon_ns


def _is_mma_phase(event: dict, phase_kind: str) -> bool:
    if _layer(event) != "phase":
        return False
    args = event.get("args") or {}
    fields = (args.get("semantic_kind"), args.get("source_name"),
              event.get("name"), event.get("cat"))
    text = " ".join(str(value).lower() for value in fields if value is not None)
    required = str(phase_kind).lower()
    return str(args.get("semantic_kind", "")).lower() == required or required in text


def _partial_overlaps(spans: list[dict], sample_limit: int = 4) -> tuple[int, list[dict]]:
    lanes = defaultdict(list)
    for ordinal, event in enumerate(spans):
        start, end, raw = _bounds_ns(event)
        if end >= start:
            lanes[(event.get("pid"), event.get("tid"))].append(
                (start, end, ordinal, raw, event))
    count = 0
    samples: list[dict] = []
    for lane, items in lanes.items():
        items.sort(key=lambda item: (item[0], -item[1], item[2]))
        active = []
        for start, end, ordinal, raw, event in items:
            while active:
                if raw and active[-1][3]:
                    disjoint = active[-1][1] <= start
                else:
                    disjoint = active[-1][1] <= start + 1e-3
                if disjoint:
                    active.pop()
                else:
                    break
            while active:
                top = active[-1]
                if raw and top[3]:
                    contained = end <= top[1]
                else:
                    contained = end <= top[1] + 1e-3
                if contained:
                    break
                count += 1
                if len(samples) < sample_limit:
                    samples.append({"lane": lane,
                                    "a": [top[0], top[1], top[4].get("name")],
                                    "b": [start, end, event.get("name")],
                                    "raw_ns": bool(raw and top[3])})
                active.pop()
            active.append((start, end, ordinal, raw, event))
    return count, samples


def verify(doc: dict, contract: dict) -> list[str]:
    """按契约逐条验 trace。返回失败列表，空表示 PASS。"""
    validate_contract(contract)
    fails: list[str] = []
    events = doc["traceEvents"]
    aux_pids = {entry.get("pid") for entry in contract.get("aux_processes", [])
                if isinstance(entry, dict) and isinstance(entry.get("pid"), int)}
    spans = [e for e in events if e.get("ph") == "X" and e.get("pid") not in aux_pids]
    counters = [e for e in events if e.get("ph") == "C"]
    fails.extend(timestamp_failures(events))
    if not spans:
        fails.append("trace 里没有 ph=X 区间，无可验证内容")
        return fails

    names = {(e["pid"]): e["args"]["name"]
             for e in events if e.get("name") == "process_name"}
    sort_idx = {(e["pid"]): e["args"]["sort_index"]
                for e in events if e.get("name") == "process_sort_index"}
    thread_names = {(e["pid"], e["tid"]): e["args"]["name"]
                    for e in events if e.get("name") == "thread_name"}

    span_pids = {e["pid"] for e in spans}
    prefix = str(contract["group_name_format"]).split("{value}")[0]

    # V1 分组双射：每个画了区间的 pid 必须有符合模板的 process_name，且 pid 与分组值 1:1。
    seen_values: dict[str, int] = {}
    for pid in sorted(span_pids):
        nm = names.get(pid)
        if nm is None or not nm.startswith(prefix):
            fails.append(f"V1 分组: pid={pid} 的 process_name={nm!r} "
                         f"不符合模板前缀 {prefix!r}（group_by="
                         f"{contract['group_by']}）")
            continue
        value = nm[len(prefix):].split()[0] if nm[len(prefix):] else ""
        if value in seen_values and seen_values[value] != pid:
            fails.append(f"V1 分组: 分组值 {value!r} 同时出现在 pid="
                         f"{seen_values[value]} 与 {pid}，不是双射")
        seen_values[value] = pid

    # 泳道集合：每个 pid 内的 tid，供 V2 泳道命名与 V3 复用。
    lanes = defaultdict(set)
    for e in spans:
        lanes[e["pid"]].add(e["tid"])

    # V2 topology: metadata 必须声明每个 observer lane，但不补物理空 warp。
    expected = contract["observer_lanes_per_cta"]
    if expected is not None:
        topology_by_pid = defaultdict(set)
        for pid, tid in thread_names:
            if pid in span_pids:
                topology_by_pid[pid].add(tid)
        missing_topology = sorted(span_pids - set(topology_by_pid))
        if missing_topology:
            fails.append(f"V2 warp topology: {len(missing_topology)} process groups lack "
                         f"thread_name metadata (samples {missing_topology[:5]})")
        bad = {pid: sorted(tids) for pid, tids in topology_by_pid.items()
               if len(tids) != expected}
        if bad:
            sample = dict(list(sorted(bad.items()))[:5])
            fails.append(f"V2 observer topology: {len(bad)}/{len(topology_by_pid)} groups "
                         f"have observer lane count != {expected} (samples {sample})")
    missing_tn = [(p, t) for p, ts in lanes.items() for t in ts
                  if (p, t) not in thread_names]
    if missing_tn:
        fails.append(f"V2 泳道命名: {len(missing_tn)} 条有 X slice 的泳道缺 "
                     f"thread_name（样例 {missing_tn[:3]}）")

    # V3 counter 置顶：counter 所在 process 的 sort_index 必须严格小于所有分组。
    if contract["counter_must_be_topmost"]:
        if not counters:
            fails.append("V3 counter 置顶: 契约要求 counter 置顶，但 trace 里没有 "
                         "ph=C 事件（hw 未接上）")
        else:
            cpids = {e["pid"] for e in counters}
            group_idx = [sort_idx[p] for p in span_pids if p in sort_idx]
            if not group_idx:
                fails.append("V3 counter 置顶: 分组缺 process_sort_index，无法判定顺序")
            else:
                for cp in sorted(cpids):
                    ci = sort_idx.get(cp)
                    if ci is None:
                        fails.append(f"V3 counter 置顶: counter pid={cp} 缺 "
                                     f"process_sort_index")
                    elif ci >= min(group_idx):
                        fails.append(f"V3 counter 置顶: counter pid={cp} 的 "
                                     f"sort_index={ci} 不小于分组最小值 "
                                     f"{min(group_idx)}，不会显示在最顶上")

    # V4 semantics: only source-measured GEMM tasks; phases may be independent.
    tasks = [event for event in spans if _layer(event) == contract["outer_layer"]]
    phases = [event for event in spans if _layer(event) == contract["inner_layer"]]
    aux_task_spans = [event for event in events
                      if event.get("ph") == "X" and event.get("pid") in aux_pids
                      and _layer(event) == contract["outer_layer"]]
    replicas = [event for event in spans
                if (event.get("args") or {}).get("is_replica") is True]
    if replicas:
        sample = [((event.get("pid"), event.get("tid")), event.get("name"),
                   _layer(event)) for event in replicas[:4]]
        fails.append(f"V4 [observer_replica_visible]: {len(replicas)} ph=X spans "
                     f"have args.is_replica=true; samples={sample}")
    if not tasks and not aux_task_spans:
        fails.append("V4 [task_layer_missing]: no source-measured GEMM task spans")
    phase_kind_default = contract["task_required_phase_kind"]
    phase_kind_by_task = contract.get("task_required_phase_by_id", {})
    required_phase_kinds = {
        str(kind) for kind in [phase_kind_default, *phase_kind_by_task.values()]
        if kind is not None
    }
    phase_index_by_kind_lane = {}
    for phase_kind in required_phase_kinds:
        phases_by_lane = defaultdict(list)
        for phase in phases:
            if _is_mma_phase(phase, phase_kind):
                phases_by_lane[(phase.get("pid"), phase.get("tid"))].append(phase)
        phase_index_by_kind_lane[phase_kind] = {
            lane: build_interval_end_index(events, _bounds_ns)
            for lane, events in phases_by_lane.items()
        }
    for task in tasks:
        args = task.get("args") or {}
        lane = (task.get("pid"), task.get("tid"))
        local_parent = (args.get("is_derived") is True
                        and args.get("aggregation") == "per_observer_lane_minmax")
        if args.get("is_derived") is True and not local_parent:
            fails.append(f"V4 [derived_task_invalid]: task {task.get('name')!r} "
                         f"on lane {lane} is not a local observer parent")
        if local_parent:
            count = args.get("source_event_count")
            source_start = args.get("source_start_ns_raw")
            source_end = args.get("source_end_ns_raw")
            if (not isinstance(count, int) or count < 2
                    or not isinstance(source_start, int)
                    or not isinstance(source_end, int)
                    or source_end < source_start):
                fails.append(f"V4 [local_parent_provenance]: task {task.get('name')!r} "
                             f"on lane {lane} has invalid source bounds/count")
        if args.get("is_derived") is not True and args.get("is_derived") is not False:
            fails.append(f"V4 [task_source_measurement_missing]: task "
                         f"{task.get('name')!r} on lane {lane} does not explicitly "
                         f"carry args.is_derived=false")
        try:
            task_id = int(args.get("task_id"))
        except (TypeError, ValueError):
            task_id = None
        role = str(args.get("task_role", args.get("role", ""))).lower()
        allowed_roles = contract["allowed_task_role"]
        role_mismatch = (allowed_roles is not None and role != allowed_roles)
        if task_id not in set(contract["allowed_task_ids"]) or role_mismatch:
            fails.append(f"V4 [invalid_task_provenance]: task {task.get('name')!r} "
                         f"on lane {lane} has task_id={task_id}, role={role!r}")
        required_phase_kind = phase_kind_by_task.get(
            str(task_id), phase_kind_by_task.get(task_id, phase_kind_default))
        if required_phase_kind is not None:
            phase_index = phase_index_by_kind_lane.get(
                str(required_phase_kind), {}).get(lane)
            if (phase_index is None
                    or not interval_index_has_contained(phase_index, task, _bounds_ns)):
                fails.append(
                    f"V4 [task_phase_missing]: task {task.get('name')!r} on lane {lane} "
                    f"contains no same-lane {required_phase_kind!r} phase")

    # V5 laminar slices: equal/nested/disjoint are legal; crossing intervals are not.
    partial_count, partial_samples = _partial_overlaps(spans)
    if partial_count:
        fails.append(f"V5 [slice_partial_overlap]: {partial_count} same-lane "
                     f"X-slice crossings; samples={partial_samples}")
    return fails


# --------------------------------------------------------------------------
# 自证：每条不变量都必须能被坏产物触发
# --------------------------------------------------------------------------

def _good_contract() -> dict:
    return {
        "schema": SCHEMA, "group_by": "sm_id", "group_name_format": "SM {value}",
        "group_sort_index_base": 100, "lane_by": ["block", "warp"],
        "lane_name_format": "cta{}/w{}",
        "cta_field": "block", "warp_field": "warp",
        "physical_warps_per_cta": 3, "observer_lanes_per_cta": 3,
        "outer_layer": "task", "inner_layer": "phase",
        "layer_from_cat_prefix": True, "counter_must_be_topmost": True,
        "source_events": "events.json", "source_abi": "abi.json",
        "allowed_task_ids": [7], "allowed_task_role": "math",
        "task_source_measured_only": True, "task_required_phase_kind": "mma",
        "allow_unparented_phase": True,
        "slice_overlap_policy": "disjoint_or_nested",
        "observer_spans_source_only": True, "allow_empty_warp_lanes": True,
    }


def _span(pid: int, tid: int, name: str, layer: str, start_ns: int, end_ns: int,
          block: int, warp: int, **extra) -> dict:
    args = {"block": block, "warp": warp, "sm_id": block, "layer": layer,
            "start_ns_raw": start_ns, "end_ns_raw": end_ns}
    args.update(extra)
    return {"ph": "X", "pid": pid, "tid": tid, "name": name,
            "cat": f"{layer}/{args.get('semantic_kind', 'compute')}",
            "ts": start_ns / 1000.0, "dur": (end_ns - start_ns) / 1000.0,
            "args": args}


def _good_trace() -> dict:
    ev = [metadata_event(_good_contract()),
          {"ph": "M", "name": "process_name", "pid": 1, "tid": 0,
           "args": {"name": "GPU HW counters"}},
          {"ph": "M", "name": "process_sort_index", "pid": 1, "tid": 0,
           "args": {"sort_index": 0}},
          {"ph": "C", "pid": 1, "tid": 0, "name": "SMs Active", "ts": 0.5,
           "args": {"SMs Active": 42}}]
    for sm in (0, 1):
        pid = 1000 + sm
        ev += [{"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
                "args": {"name": f"SM {sm}"}},
               {"ph": "M", "name": "process_sort_index", "pid": pid, "tid": 0,
                "args": {"sort_index": 100 + sm}}]
        for warp in (0, 1, 2):
            ev.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": warp,
                       "args": {"name": f"cta{sm}/w{warp}"}})
        # Same source measurement is intentionally rendered as task + MMA phase.
        ev.append(_span(pid, 0, "GEMM0", "task", 1000, 3000, sm, 0,
                        task_id=7, role="math", task_role="math", is_derived=False))
        ev.append(_span(pid, 0, "MMA", "phase", 1000, 3000, sm, 0,
                        task_id=7, role="math", semantic_kind="mma", is_derived=False))
        # A later real phase on another warp has no task parent and must remain legal.
        ev.append(_span(pid, 1, "Load", "phase", 4000, 5000, sm, 1,
                        task_id=5, role="loader", semantic_kind="load", is_derived=False))
    return {"traceEvents": ev, "displayTimeUnit": "ns"}

def _selftest() -> int:
    failures: list[str] = []

    def expect(label: str, fails: list[str], want_fail: bool, needle: str = "") -> None:
        got_fail = bool(fails)
        if got_fail != want_fail:
            failures.append(f"{label}: 期望 {'FAIL' if want_fail else 'PASS'}，"
                            f"实得 {fails or 'PASS'}")
        elif want_fail and needle and not any(needle in f for f in fails):
            failures.append(f"{label}: FAIL 了但原因不含 {needle!r}：{fails}")

    good = _good_trace()
    expect("合规 trace", verify(good, _good_contract()), False)

    # V6: displayed Perfetto ts is authoritative even when raw absolute ns is positive.
    t = json.loads(json.dumps(good))
    next(e for e in t["traceEvents"] if e.get("ph") == "X")["ts"] = -0.001
    expect("V6 negative X timestamp with positive raw ns",
           verify(t, _good_contract()), True, "timeline_negative_timestamp")
    t = json.loads(json.dumps(good))
    next(e for e in t["traceEvents"] if e.get("ph") == "C")["ts"] = -0.001
    expect("V6 negative C timestamp",
           verify(t, _good_contract()), True, "timeline_negative_timestamp")

    # V1：把一个分组的 process_name 改成不符合模板
    t = json.loads(json.dumps(good))
    for e in t["traceEvents"]:
        if e.get("name") == "process_name" and e["pid"] == 1001:
            e["args"]["name"] = "block 1"
    expect("V1 分组名不符模板", verify(t, _good_contract()), True, "V1")

    # V2: an empty observer lane is legal when topology metadata still declares it.
    expect("V2 empty observer lane", verify(json.loads(json.dumps(good)),
                                             _good_contract()), False)
    # Removing the empty lane's topology metadata must fail.
    t = json.loads(json.dumps(good))
    t["traceEvents"] = [e for e in t["traceEvents"]
                        if not (e.get("name") == "thread_name" and e.get("tid") == 2)]
    expect("V2 missing empty-lane topology", verify(t, _good_contract()), True, "V2")

    # Exact 12-warp regression: metadata for all lanes, X slices on only two lanes.
    c12 = _good_contract(); c12["physical_warps_per_cta"] = 12; c12["observer_lanes_per_cta"] = 12
    t12 = json.loads(json.dumps(good))
    for sm in (0, 1):
        pid = 1000 + sm
        for warp in range(3, 12):
            t12["traceEvents"].append(
                {"ph": "M", "name": "thread_name", "pid": pid, "tid": warp,
                 "args": {"name": f"cta{sm}/w{warp}"}})
    expect("V2 12-warp topology with empty lanes", verify(t12, c12), False)

    # V3：把 counter 的 sort_index 放到分组之后
    t = json.loads(json.dumps(good))
    for e in t["traceEvents"]:
        if e.get("name") == "process_sort_index" and e["pid"] == 1:
            e["args"]["sort_index"] = 999
    expect("V3 counter 未置顶", verify(t, _good_contract()), True, "V3")

    # V3b：完全没有 counter（我这轮的真实状态）
    t = json.loads(json.dumps(good))
    t["traceEvents"] = [e for e in t["traceEvents"] if e.get("ph") != "C"]
    expect("V3 无 counter", verify(t, _good_contract()), True, "V3")

    # V4: every task must contain a same-lane MMA phase.
    t = json.loads(json.dumps(good))
    t["traceEvents"] = [e for e in t["traceEvents"]
                        if not (e.get("ph") == "X"
                                and e.get("args", {}).get("semantic_kind") == "mma")]
    expect("V4 task lacks MMA phase", verify(t, _good_contract()), True,
           "task_mma_phase_missing")

    # Independent phase after task completion, on another lane, is explicitly legal.
    expect("V4 phase may outlive task on another lane",
           verify(json.loads(json.dumps(good)), _good_contract()), False)

    t = json.loads(json.dumps(good))
    task = next(e for e in t["traceEvents"] if _layer(e) == "task")
    task["args"]["is_derived"] = True
    expect("V4 derived task requires local provenance", verify(t, _good_contract()), True,
           "derived_task_invalid")

    t = json.loads(json.dumps(good))
    replica = next(e for e in t["traceEvents"] if e.get("ph") == "X")
    replica["args"]["is_replica"] = True
    expect("V4 observer replica forbidden", verify(t, _good_contract()), True,
           "observer_replica_visible")

    t = json.loads(json.dumps(good))
    task = next(e for e in t["traceEvents"] if _layer(e) == "task")
    task["args"]["task_id"] = 5
    task["args"]["role"] = "loader"
    task["args"]["task_role"] = "loader"
    expect("V4 non-GEMM task forbidden", verify(t, _good_contract()), True,
           "invalid_task_provenance")

    # Exact user regression: A=[5536,51136], B=[47552,55296] crosses A.
    t = json.loads(json.dumps(good))
    t["traceEvents"].append(_span(1000, 1, "A", "phase", 5536, 51136, 0, 1,
                                  semantic_kind="load", is_derived=False))
    t["traceEvents"].append(_span(1000, 1, "B", "phase", 47552, 55296, 0, 1,
                                  semantic_kind="store", is_derived=False))
    expect("V5 partial overlap", verify(t, _good_contract()), True,
           "slice_partial_overlap")

    # 契约自身改坏：逐字段删除，必须全部拒绝
    for key in REQUIRED_KEYS:
        c = _good_contract()
        c.pop(key)
        try:
            verify(good, c)
        except SystemExit:
            pass
        else:
            failures.append(f"契约缺字段 {key} 时未拒绝")
    # 占位符数量不匹配
    c = _good_contract(); c["lane_name_format"] = "cta{}"
    try:
        verify(good, c)
    except SystemExit:
        pass
    else:
        failures.append("lane_name_format 占位符数不匹配时未拒绝")
    # outer == inner 会让嵌套检查恒真
    c = _good_contract(); c["inner_layer"] = "task"
    try:
        verify(good, c)
    except SystemExit:
        pass
    else:
        failures.append("outer_layer == inner_layer 时未拒绝")

    if failures:
        print("[selftest] FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print(f"[selftest] PASS 6 条不变量 + 契约完整性 "
          f"({len(REQUIRED_KEYS)} 字段) 全部能被坏产物触发")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", help="要校验的 perfetto trace（可 .gz）")
    ap.add_argument("--contract", help="契约 JSON；缺省则从 trace 内嵌的 metadata 读")
    ap.add_argument("--selftest", action="store_true",
                    help="用已知坏产物证明每条不变量都会 FAIL")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.trace:
        ap.error("需要 trace 位置参数，或用 --selftest")

    doc = load_trace(pathlib.Path(args.trace))
    if args.contract:
        contract = json.loads(pathlib.Path(args.contract).read_text())
    else:
        contract = contract_from_trace(doc)
        if contract is None:
            raise SystemExit(
                f"{args.trace}: 没有内嵌 {METADATA_EVENT_NAME} metadata 事件，"
                "且未给 --contract。布局含义不可知，拒绝给出结论。")
    fails = verify(doc, contract)
    if fails:
        print("FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("PASS 布局契约六条不变量全部满足")
    return 0


if __name__ == "__main__":
    sys.exit(main())
