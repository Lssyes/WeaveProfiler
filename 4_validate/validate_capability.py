#!/usr/bin/env python3
"""功能规格判据：只量产物，不看代码。

存在性判据（溯源链/共享字段）只能证明工具与当前 kernel 匹配，
证明不了产物是否满足要求的粒度与内容。本脚本对 Perfetto trace
做量化验收，凡粒度不足/层次缺失/硬件缺失一律 FAIL。
"""
import argparse
import gzip
import json
import math
import sys
from collections import defaultdict


def load(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


VISIBLE_LAYERS = ("task", "phase")
FORBIDDEN_TIMELINE_WORDS = ("wait", "barrier", "sync", "acquire", "release")


def find_timestamp_anomalies(events, sample_limit=8):
    """Return invalid/negative displayed ts anomalies across every timed event."""
    result = {"invalid_count": 0, "negative_count": 0,
              "invalid_samples": [], "negative_samples": []}
    for index, event in enumerate(events):
        if "ts" not in event:
            continue
        value = event.get("ts")
        sample = {"index": index, "ph": event.get("ph"), "name": event.get("name"),
                  "pid": event.get("pid"), "tid": event.get("tid"), "ts": value}
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            result["invalid_count"] += 1
            if len(result["invalid_samples"]) < sample_limit:
                result["invalid_samples"].append(sample)
        elif float(value) < 0.0:
            result["negative_count"] += 1
            if len(result["negative_samples"]) < sample_limit:
                result["negative_samples"].append(sample)
    return result


def span_layer(event):
    args = event.get("args") or {}
    layer = args.get("layer")
    if layer in (*VISIBLE_LAYERS, "ignored"):
        return layer
    category = str(event.get("cat", ""))
    for candidate in (*VISIBLE_LAYERS, "ignored"):
        if category.startswith(candidate + "/"):
            return candidate
    return None


def span_text(event):
    args = event.get("args") or {}
    fields = (event.get("name"), event.get("cat"), args.get("semantic_kind"),
              args.get("source_name"))
    return " ".join(str(value).lower() for value in fields if value is not None)


def interval_contains(outer, inner):
    outer_args = outer.get("args") or {}
    inner_args = inner.get("args") or {}
    raw = (outer_args.get("start_ns_raw"), outer_args.get("end_ns_raw"),
           inner_args.get("start_ns_raw"), inner_args.get("end_ns_raw"))
    if all(value is not None for value in raw):
        outer_start, outer_end, inner_start, inner_end = map(int, raw)
        return outer_start <= inner_start and outer_end >= inner_end
    outer_start = float(outer.get("ts", 0))
    outer_end = outer_start + float(outer.get("dur", 0))
    inner_start = float(inner.get("ts", 0))
    inner_end = inner_start + float(inner.get("dur", 0))
    epsilon_us = 1e-6
    return outer_start <= inner_start + epsilon_us and outer_end + epsilon_us >= inner_end


ALLOWED_SOURCE_TASK_IDS = frozenset((7, 10))


def semantic_task_ids(document):
    if not isinstance(document, dict):
        return None
    candidates = [document.get("tasks")]
    events = document.get("events")
    if isinstance(events, dict):
        candidates.append(events.get("event_definitions"))
    for descriptors in candidates:
        if not isinstance(descriptors, list):
            continue
        ids = set()
        for descriptor in descriptors:
            if (not isinstance(descriptor, dict) or descriptor.get("layer") != "task"
                    or not isinstance(descriptor.get("task_contract"), dict)):
                continue
            value = descriptor.get("id", descriptor.get("event_id",
                                   descriptor.get("task_id")))
            if isinstance(value, int) and not isinstance(value, bool):
                ids.add(value)
        if ids:
            return ids
    return None


def interval_bounds_ns(event):
    """Return (start_ns, end_ns, exact_raw_ns), preferring integer raw bounds."""
    args = event.get("args") or {}
    raw_start = args.get("start_ns_raw")
    raw_end = args.get("end_ns_raw")
    if (isinstance(raw_start, int) and not isinstance(raw_start, bool)
            and isinstance(raw_end, int) and not isinstance(raw_end, bool)):
        return raw_start, raw_end, True
    start_ns = float(event.get("ts", 0.0)) * 1000.0
    end_ns = start_ns + float(event.get("dur", 0.0)) * 1000.0
    return start_ns, end_ns, False


def find_partial_overlaps(spans, sample_limit=8):
    """Find crossing X slices per pid/tid in O(N log N); nesting/equality is legal."""
    lanes = defaultdict(list)
    for ordinal, event in enumerate(spans):
        start_ns, end_ns, exact = interval_bounds_ns(event)
        if end_ns < start_ns:
            continue
        lanes[(event.get("pid"), event.get("tid"))].append(
            (start_ns, end_ns, ordinal, exact, event))

    count = 0
    samples = []
    for lane, items in lanes.items():
        items.sort(key=lambda item: (item[0], -item[1], item[2]))
        active = []
        for start_ns, end_ns, ordinal, exact, event in items:
            while active:
                top_end = active[-1][1]
                if exact and active[-1][3]:
                    disjoint = top_end <= start_ns
                else:
                    disjoint = top_end <= start_ns + 1e-3
                if disjoint:
                    active.pop()
                else:
                    break
            while active:
                top = active[-1]
                if exact and top[3]:
                    contained = end_ns <= top[1]
                else:
                    contained = end_ns <= top[1] + 1e-3
                if contained:
                    break
                count += 1
                if len(samples) < sample_limit:
                    samples.append({
                        "lane": lane,
                        "outer_candidate": top[4].get("name"),
                        "outer_bounds_ns": [top[0], top[1]],
                        "crossing": event.get("name"),
                        "crossing_bounds_ns": [start_ns, end_ns],
                        "raw_ns": bool(exact and top[3]),
                    })
                active.pop()
            active.append((start_ns, end_ns, ordinal, exact, event))
    return count, samples


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


def is_mma_phase(event):
    if span_layer(event) != "phase":
        return False
    args = event.get("args") or {}
    kind = str(args.get("semantic_kind", "")).lower()
    return kind == "mma" or "mma" in span_text(event)


def validate_task_phase_semantics(spans, semantic_task_ids=None):
    """Validate direct source tasks, using semantic contracts when available."""
    fails = []
    notes = []
    task_spans = [event for event in spans if span_layer(event) == "task"]
    phase_spans = [event for event in spans if span_layer(event) == "phase"]
    replicas = [event for event in spans
                if (event.get("args") or {}).get("is_replica") is True]
    if replicas:
        samples = [((event.get("pid"), event.get("tid")), event.get("name"),
                    span_layer(event)) for event in replicas[:8]]
        fails.append(("observer_replica_visible",
                      f"{len(replicas)} ph=X spans have args.is_replica=true; "
                      f"observer spans belong only to measured work; samples={samples}"))
    semantic_ids = set(semantic_task_ids or ())
    use_semantic_contract = semantic_task_ids is not None
    mma_by_lane = defaultdict(list)
    if not use_semantic_contract:
        for event in phase_spans:
            if is_mma_phase(event):
                mma_by_lane[(event.get("pid"), event.get("tid"))].append(event)
    mma_index_by_lane = {
        lane: build_interval_end_index(events, interval_bounds_ns)
        for lane, events in mma_by_lane.items()
    }

    derived = []
    non_source = []
    invalid = []
    missing_mma = []
    for task in task_spans:
        args = task.get("args") or {}
        lane = (task.get("pid"), task.get("tid"))
        if args.get("is_derived") is True:
            derived.append((lane, task.get("name"), args.get("task_id")))
        if args.get("is_derived") is not False:
            non_source.append((lane, task.get("name"), args.get("is_derived")))
        try:
            task_id = int(args.get("task_id"))
        except (TypeError, ValueError):
            task_id = None
        if use_semantic_contract:
            if task_id not in semantic_ids:
                invalid.append((lane, task.get("name"), task_id, "not_in_task_contract"))
        else:
            role = str(args.get("task_role", args.get("role", ""))).lower()
            if task_id not in ALLOWED_SOURCE_TASK_IDS or role != "math":
                invalid.append((lane, task.get("name"), task_id, role))
            mma_index = mma_index_by_lane.get(lane)
            if (mma_index is None
                    or not interval_index_has_contained(
                        mma_index, task, interval_bounds_ns)):
                missing_mma.append((lane, task.get("name"), task_id))

    if derived:
        fails.append(("derived_task_visible",
                      f"{len(derived)} visible task spans have args.is_derived=true; "
                      f"tasks must be source measured; samples={derived[:8]}"))
    if non_source:
        fails.append(("task_source_measurement_missing",
                      f"{len(non_source)} task spans do not explicitly carry "
                      f"args.is_derived=false; samples={non_source[:8]}"))
    if invalid:
        rule = "source-locked semantic task contract" if use_semantic_contract \
            else "legacy source GEMM task_id 7/10 with math role"
        fails.append(("invalid_task_provenance",
                      f"{len(invalid)} task spans violate {rule}; samples={invalid[:8]}"))
    if missing_mma:
        fails.append(("task_mma_phase_missing",
                      f"{len(missing_mma)} legacy source GEMM tasks contain no same-lane "
                      f"MMA phase; samples={missing_mma[:8]}"))
    notes.append(f"source tasks={len(task_spans)}; semantic_contract={use_semantic_contract}; "
                 f"unparented/independent real phases are allowed")
    return fails, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--contract', metavar='PATH', default=None,
                    help='项目的 profile/contract.json;从中读 grid/warps/phases/kloop 等,'
                         '避免每次命令行手敲传错。命令行显式传的参数优先级更高')
    ap.add_argument('--expect-blocks', type=int, default=None,
                    help='expected CTA count per rank, never merged across EP ranks')
    ap.add_argument('--expect-ranks', type=int, default=None,
                    help='expected EP rank count')
    ap.add_argument('--expect-warps', type=int, default=None)
    ap.add_argument('--expect-phases', type=int, default=None)
    ap.add_argument('--require-task-layer', action='store_true', default=None)
    ap.add_argument('--require-hw', action='store_true', default=None)
    ap.add_argument('--hw-required-metric', action='append', default=[])
    ap.add_argument('--expect-kloop-per-warp', type=int, default=None,
                    help='kernel 主循环声明的迭代次数(打点者视角)。给了就检查观测到的 kloop task 实例数是否接近这个值。')
    ap.add_argument('--kloop-task-pattern', type=str, default=None,
                    help='用作 kloop 密度判据的 task name 正则;正则 OR 分隔')
    ap.add_argument('--kloop-density-tolerance', type=float, default=0.5,
                    help='观测 / 期望 的容差(默认 0.5,允许 50%% 差距,给采样打点留余地)')
    ap.add_argument('--json', metavar='PATH',
                    help="输出机器可读结果（含带 id 的 fails），'-' 表示写 stdout")
    args = ap.parse_args()

    # 契约文件填空:命令行没给的字段从契约取
    contract = {}
    if args.contract:
        with open(args.contract, 'r', encoding='utf-8') as handle:
            contract = json.load(handle)
        # Newer materialized contracts nest these under build_and_launch /
        # topology / events / run; keep the flat top-level keys as the first
        # preference so older flat contracts still work unchanged.
        bl = contract.get('build_and_launch') if isinstance(contract.get('build_and_launch'), dict) else {}
        topo = contract.get('topology') if isinstance(contract.get('topology'), dict) else {}
        run = contract.get('run') if isinstance(contract.get('run'), dict) else {}
        events_block = contract.get('events') if isinstance(contract.get('events'), dict) else {}
        if args.expect_blocks is None:
            args.expect_blocks = contract.get('grid')
            if args.expect_blocks is None:
                args.expect_blocks = bl.get('num_ctas')
        if args.expect_ranks is None:
            args.expect_ranks = contract.get('ep_size', contract.get('world_size'))
            if args.expect_ranks is None:
                args.expect_ranks = run.get('world_size')
        if args.expect_warps is None:
            args.expect_warps = contract.get('warps_per_block')
            if args.expect_warps is None:
                args.expect_warps = (bl.get('warps_per_cta')
                                     or topo.get('physical_warps_per_cta')
                                     or topo.get('observer_lane_count'))
        if args.expect_phases is None:
            args.expect_phases = contract.get('phases')
            if args.expect_phases is None:
                phase_names = {d.get('name') for d in (events_block.get('event_definitions') or [])
                               if isinstance(d, dict) and d.get('layer') == 'phase'}
                if phase_names:
                    args.expect_phases = len(phase_names)
        if args.expect_kloop_per_warp is None:
            args.expect_kloop_per_warp = contract.get('kloop_per_warp')
        if args.kloop_task_pattern is None:
            args.kloop_task_pattern = contract.get('kloop_task_pattern')
        if args.require_task_layer is None:
            args.require_task_layer = contract.get('require_task_layer')
        if args.require_hw is None:
            args.require_hw = contract.get('require_hw')
        if not args.hw_required_metric:
            args.hw_required_metric = contract.get('hw_required_metrics') or []
    if args.kloop_task_pattern is None:
        args.kloop_task_pattern = 'wgmma|mma_compute|mma|attn_qk_mma|attn_softmax_pv'
    missing_spec = [name for name, value in (
        ('--expect-blocks/grid', args.expect_blocks),
        ('--expect-warps/warps_per_block', args.expect_warps),
        ('--expect-phases/phases', args.expect_phases),
    ) if value is None]
    if missing_spec:
        ap.error('缺少期望值(命令行或 --contract 至少给一处): ' + ', '.join(missing_spec))

    doc = load(args.trace)
    events = doc.get('traceEvents', doc if isinstance(doc, list) else [])
    spans = [e for e in events if e.get('ph') == 'X']
    counters = [e for e in events if e.get('ph') == 'C']
    meta = [e for e in events if e.get('ph') == 'M']

    fails, notes = [], []

    ts_anomalies = find_timestamp_anomalies(events)
    if ts_anomalies["invalid_count"]:
        fails.append(("timeline_invalid_timestamp",
                      f'{ts_anomalies["invalid_count"]} timed events have non-numeric/non-finite ts; '
                      f'samples={ts_anomalies["invalid_samples"]}'))
    if ts_anomalies["negative_count"]:
        fails.append(("timeline_negative_timestamp",
                      f'{ts_anomalies["negative_count"]} timed events have ts<0; '
                      f'Perfetto will drop them; samples={ts_anomalies["negative_samples"]}'))
    notes.append(f'ts<0 的 timed events = {ts_anomalies["negative_count"]}; '
                 f'invalid ts = {ts_anomalies["invalid_count"]}')

    # Grid is per rank. EP2 means 78 CTAs on rank 0 and 78 on rank 1.
    pids = {event.get("pid") for event in spans}
    rank_pids = defaultdict(set)
    missing_rank = 0
    for event in spans:
        rank = (event.get("args") or {}).get("rank")
        if rank is None:
            missing_rank += 1
        else:
            rank_pids[int(rank)].add(event.get("pid"))
    if missing_rank:
        fails.append(("rank_metadata_missing",
                      f"{missing_rank} spans lack args.rank; cannot validate per-rank grid"))
    blocks_per_rank = {rank: len(rank_pids[rank]) for rank in sorted(rank_pids)}
    notes.append(f"blocks per rank = {blocks_per_rank} (expected {args.expect_blocks} each)")
    bad_ranks = {rank: count for rank, count in blocks_per_rank.items()
                 if count != args.expect_blocks}
    if bad_ranks:
        fails.append(("block_coverage_per_rank",
                      f"per-rank CTA coverage mismatch: {bad_ranks}; "
                      f"expected {args.expect_blocks} on every rank"))
    if args.expect_ranks is not None and len(rank_pids) != args.expect_ranks:
        fails.append(("rank_coverage",
                      f"EP rank coverage {len(rank_pids)} != {args.expect_ranks}; "
                      f"observed ranks={sorted(rank_pids)}"))

    # Warp topology comes from thread metadata. Observer spans may leave lanes empty.
    topology_tids_per_pid = defaultdict(set)
    for event in meta:
        if event.get("name") == "thread_name" and event.get("pid") in pids:
            topology_tids_per_pid[event.get("pid")].add(event.get("tid"))
    missing_topology = sorted(pids - set(topology_tids_per_pid))
    if missing_topology:
        fails.append(("warp_topology_metadata_missing",
                      f"{len(missing_topology)} CTA processes lack thread_name topology; "
                      f"samples={missing_topology[:8]}"))
    warp_counts = sorted({len(topology_tids_per_pid[pid]) for pid in pids
                          if pid in topology_tids_per_pid})
    notes.append(f'每 block 的 topology warp 泳道数 = {warp_counts}'
                 f'（期望 [{args.expect_warps}]；空泳道允许）')
    if warp_counts != [args.expect_warps]:
        fails.append(('warp_granularity',
                      f'warp topology 粒度不符：实测 {warp_counts} != '
                      f'[{args.expect_warps}]；必须保留全部 warp 的 thread metadata，'
                      f'但未观测 warp 不得复制 X slice'))

    # Only source-measured GEMM0/GEMM1 are tasks. Real warp phases may be independent.
    depths = defaultdict(int)
    layers = defaultdict(int)
    for event in spans:
        depths[int((event.get("args") or {}).get("depth", 0) or 0)] += 1
        layers[span_layer(event)] += 1
    notes.append(f"depth distribution = {dict(sorted(depths.items()))}")
    notes.append(f"layer distribution = {dict(sorted(layers.items(), key=lambda item: str(item[0])))}")
    if layers.get(None, 0):
        fails.append(("semantic_layer_missing",
                      f"{layers[None]} spans lack args.layer or a task/phase category"))

    task_spans = [event for event in spans if span_layer(event) == "task"]
    phase_spans = [event for event in spans if span_layer(event) == "phase"]
    phase_names = {event.get("name") for event in phase_spans}
    notes.append(f"phase names = {len(phase_names)} (expected at least {args.expect_phases})")
    if len(phase_names) < args.expect_phases:
        fails.append(("phase_layer",
                      f"phase layer too small: {len(phase_names)} < {args.expect_phases}"))
    if args.require_task_layer and not task_spans:
        fails.append(("task_layer_missing", "no source-measured GEMM task spans"))
    notes.append(f"source tasks={len(task_spans)}, real warp phases={len(phase_spans)}")

    forbidden = [event for event in spans
                 if any(word in span_text(event) for word in FORBIDDEN_TIMELINE_WORDS)]
    if forbidden:
        samples = sorted({(event.get("args") or {}).get("source_name") or event.get("name")
                          for event in forbidden})[:12]
        fails.append(("wait_barrier_visible",
                      f"{len(forbidden)} wait/barrier/sync spans are visible; samples={samples}"))

    semantic_fails, semantic_notes = validate_task_phase_semantics(
        spans, semantic_task_ids(contract))
    fails.extend(semantic_fails)
    notes.extend(semantic_notes)

    phase_kinds = {
        "load": ("load", "tma", "gather", "fetch"),
        "compute": ("mma", "wgmma", "compute"),
        "store_or_epilogue": ("store", "epilogue", "scatter", "reduce"),
    }
    missing_kinds = [kind for kind, keys in phase_kinds.items()
                     if not any(any(key in span_text(event) for key in keys)
                                for event in phase_spans)]
    if missing_kinds:
        fails.append(("phase_kind_coverage",
                      f"real phase categories missing {missing_kinds}; "
                      f"observed={sorted(phase_names)}"))

    # Role-specialized warps do not form block*warp*phase Cartesian products.
    notes.append(f"ph=X = {len(spans)} (no Cartesian-product volume floor)")

    # Optional sampled K-loop density applies to inner MMA phases only.
    if args.expect_kloop_per_warp:
        import re
        pat = re.compile(args.kloop_task_pattern, re.IGNORECASE)
        kloop_spans = [e for e in spans
                       if span_layer(e) == 'phase'
                       and pat.search(e.get('name', ''))]
        observed = len(kloop_spans)
        # 每 (block, warp) 平均实例数
        block_warp_count = max(len(pids) * args.expect_warps, 1)
        per_bw = observed / block_warp_count
        expected = args.expect_kloop_per_warp
        ratio = per_bw / expected if expected else 0
        notes.append(f'kloop 密度: 匹配 "{args.kloop_task_pattern}" 的 spans={observed}, '
                     f'每 (block×warp) ≈ {per_bw:.1f}, 期望 K = {expected}, 比值 = {ratio:.3f}')
        if abs(ratio - 1.0) > args.kloop_density_tolerance:
            fails.append(('kloop_density_off',
                          f'主循环密度不符: 每 (block×warp) 实例 {per_bw:.1f} vs 期望 K={expected}, '
                          f'比值 {ratio:.2f} 偏离 1.0 超过 {args.kloop_density_tolerance}. '
                          f'ratio << 1 说明只打了 phase 首尾代表, ratio >> 1 说明打太多. '
                          f'若用了采样打点, 请把 K 传成 K/sampling_rate 后再判. '
                          f'匹配 pattern="{args.kloop_task_pattern}"'))

    # 硬件 counter
    notes.append(f'ph=C（硬件 counter 采样点）= {len(counters)}')
    if args.require_hw:
        if not counters:
            fails.append(('hw_missing',
                          '硬件 counter 缺失：ph=C 为 0，未合并 NSYS GPU Metrics'))
        else:
            got = {e.get('name', '') for e in counters}
            for want in args.hw_required_metric:
                if not any(want.lower() in name.lower() for name in got):
                    fails.append(('hw_metric_missing',
                                  f'缺 required 硬件指标 {want!r}；实际={sorted(got)}'))

    # 时间自洽
    bad_dur = sum(1 for e in spans if float(e.get('dur', 0)) < 0)
    if bad_dur:
        fails.append(('negative_duration', f'{bad_dur} 个区间 dur<0'))
    notes.append(f'dur<0 的区间 = {bad_dur}')

    partial_overlaps, overlap_samples = find_partial_overlaps(spans)
    notes.append(f'same-lane partial overlaps = {partial_overlaps}')
    if partial_overlaps:
        fails.append(('slice_partial_overlap',
                      f'{partial_overlaps} same-lane X-slice crossings; '
                      f'only disjoint or nested intervals are legal; samples={overlap_samples}'))

    if args.json:
        payload = {
            'trace': args.trace,
            'passed': not fails,
            'expect': {
                'blocks_per_rank': args.expect_blocks,
                'ranks': args.expect_ranks,
                'warps': args.expect_warps,
                'phases': args.expect_phases,
                'task_layer': args.require_task_layer,
                'hw': args.require_hw,
                'hw_metrics': args.hw_required_metric,
            },
            'observed': {
                'ranks': sorted(rank_pids),
                'blocks_per_rank': blocks_per_rank,
                'block_lane_groups_total': len(pids),
                'warp_lanes_per_block': warp_counts,
                'phase_names': len(phase_names),
                'task_spans': len(task_spans),
                'phase_spans': len(phase_spans),
                'spans': len(spans),
                'counters': len(counters),
                'meta': len(meta),
                'negative_duration': bad_dur,
                'negative_timestamp': ts_anomalies["negative_count"],
                'invalid_timestamp': ts_anomalies["invalid_count"],
                'partial_overlaps': partial_overlaps,
            },
            'notes': notes,
            'fails': [{'id': fail_id, 'message': message} for fail_id, message in fails],
        }
        out = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.json == '-':
            print(out)
        else:
            with open(args.json, 'w', encoding='utf-8') as handle:
                handle.write(out + '\n')
        return 1 if fails else 0

    print(f'--- 功能规格验收 {args.trace}')
    for note in notes:
        print(f'  {note}')
    print(f'  ph=M 元数据 = {len(meta)}')
    if fails:
        print('FAIL')
        for fail_id, message in fails:
            print(f'  - [{fail_id}] {message}')
        return 1
    print('PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
