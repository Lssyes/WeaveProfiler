#!/usr/bin/env python3
"""交付闸门:上 OSS 前必过。

把散落的判据串成一道强制门 —— 让「记得跑判据」从人的自觉变成流程强制。
读项目的 profile/contract.json,对最终 trace 跑全部自证判据,全绿才 exit 0。

用法:
    python3 -B deliver_gate.py <项目根> <trace.perfetto.json.gz>
    python3 -B deliver_gate.py <项目根> <trace> --contract <显式契约路径>

判据分两类:
  自证不变量  同一份产物内部就能查矛盾,不依赖任何外部输入
  契约比对    与 contract.json 声明的 grid/warps/phases/kloop 比对

不判 counter 数值高低 —— SMs Active 是 5% 还是 95% 是 kernel 的真实属性,
profile 的意义正是暴露它。工具只判「hw 采的和 sw 是不是同一个 build」。
"""
import argparse
import csv
import gzip
import hashlib
import json
import importlib.util
import io
import math
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from validate_capability import (find_partial_overlaps, find_timestamp_anomalies,
                                 interval_bounds_ns, span_layer)

SCRIPT_DIR = Path(__file__).resolve().parent
DETECT_DIR = SCRIPT_DIR.parent / 'scan'
if str(DETECT_DIR) not in sys.path:
    sys.path.insert(0, str(DETECT_DIR))

from semantic_contract import (audit_semantic_task_bijection,
                               audit_semantic_trace_tasks,
                               task_descriptors)

LAYOUT_CHECKER_PATH = DETECT_DIR / 'layout_contract.py'


def load_layout_checker():
    spec = importlib.util.spec_from_file_location('fast_instrument_layout_contract',
                                                  LAYOUT_CHECKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load layout checker: {LAYOUT_CHECKER_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LAYOUT_CHECKER = load_layout_checker()


def load_trace(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


# Operator/task 语义映射。新产物优先写 operator_id/operator_scope；旧产物用
# rank_block（或 rank+block）以及现有 task_id/name 兜底，避免把物理 warp 当作作用域。
OPERATOR_FIELDS = ('operator_id', 'operator', 'logical_operator', 'operator_name')
SCOPE_FIELDS = ('operator_scope', 'scope_key', 'scope', 'rank_block')
def is_global_task_envelope(event):
    """识别辅助 TASK process 上的全 trace derived envelope。"""
    args = event.get('args') or {}
    layer = args.get('layer') or str(event.get('cat', '')).split('/', 1)[0]
    aggregation = str(args.get('aggregation', '')).lower()
    parent_scope = str(args.get('parent_scope', args.get('operator_scope', ''))).lower()
    if (parent_scope in ('rank_cta', 'rank+cta', 'rank_cta_lane', 'observer_lane')
            or aggregation in ('per_rank_cta', 'rank_cta', 'per-rank-cta',
                               'per_observer_lane_minmax')):
        return False
    return (layer == 'task'
            and (args.get('task_envelope') is True
                 or args.get('aggregation') is not None
                 or str(event.get('cat', '')).startswith('task/logical_operator')))


def _envelope_source_count(args):
    for field in ('contributor_count', 'source_event_count', 'source_task_count', 'source_count'):
        value = args.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return field, value
    return None, None


def _envelope_source_scope_count(args):
    """Require source coverage evidence in addition to contributor count."""
    value = args.get('source_block_count')
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return 'source_block_count', value
    ranks = args.get('source_ranks')
    if isinstance(ranks, (list, tuple, set)) and len(ranks) > 0:
        return 'source_ranks', len(ranks)
    return None, None


def _auxiliary_task_pids(spans):
    """Return pids used only for derived logical-operator envelopes.

    These envelopes are a global aggregation lane.  They are evidence for the
    task/phase semantic gate, not CTA/warp topology lanes, so the layout gate
    must not count them as a CTA or as a warp group.
    """
    return {
        event.get('pid') for event in spans
        if is_global_task_envelope(event) and event.get('pid') is not None
    }


def _layout_trace_without_auxiliary_tasks(trace, spans):
    """Make a shallow trace view with auxiliary envelope processes removed."""
    aux_pids = _auxiliary_task_pids(spans)
    if not aux_pids:
        return trace, False, aux_pids
    events = trace.get('traceEvents', trace if isinstance(trace, list) else [])
    filtered_events = [event for event in events if event.get('pid') not in aux_pids]
    if isinstance(trace, list):
        return filtered_events, True, aux_pids
    layout_trace = dict(trace)
    layout_trace['traceEvents'] = filtered_events
    return layout_trace, True, aux_pids


def _stable_token(value):
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    text = str(value).strip()
    return text or None


def operator_scope_key(event):
    """Map one event to its logical-operator scope, never to a warp lane."""
    args = event.get('args') or {}
    if is_global_task_envelope(event):
        return 'global-envelope'
    for field in SCOPE_FIELDS:
        token = _stable_token(args.get(field))
        if token is not None:
            return f'{field}={token}'
    rank = _stable_token(args.get('rank'))
    block = _stable_token(args.get('block'))
    if rank is not None and block is not None:
        return f'rank={rank};block={block}'
    if block is not None:
        return f'block={block}'
    if rank is not None:
        return f'rank={rank}'
    # pid is only a last-resort process scope; tid/warp is deliberately excluded.
    pid = _stable_token(event.get('pid'))
    return f'pid={pid}' if pid is not None else 'global'


def _explicit_operator_key(event):
    args = event.get('args') or {}
    for field in OPERATOR_FIELDS:
        token = _stable_token(args.get(field))
        if token is not None:
            return f'explicit:{token}'
    return None


def _generic_task_key(event):
    """Return an ABI-driven identity shared by a task and its phases."""
    args = event.get('args') or {}
    for field in ('task_id', 'taskId', 'task_index', 'task_idx', 'id',
                  'task', 'task_name', 'source_name'):
        token = _stable_token(args.get(field))
        if token is not None:
            return f'{field}:{token}'
    return None


def operator_key(event):
    """Map task and its real phases to one logical operator identity."""
    explicit = _explicit_operator_key(event)
    if explicit is not None:
        return explicit
    generic = _generic_task_key(event)
    if generic is not None:
        return generic
    args = event.get('args') or {}
    if span_layer(event) in ('task', 'phase'):
        token = _stable_token(event.get('name'))
        return f'task:{token}' if token is not None else None
    # Unmapped dispatch/loader phases remain independent real phases; they are
    # not silently assigned to a task merely because they have a task-like name.
    return None


def check_operator_task_shape(spans, semantic_viewer=None):
    """Require task/phase nesting on the same observer-warp lane."""
    fails = []
    notes = []
    tasks = [event for event in spans if span_layer(event) == 'task']
    phases = [event for event in spans if span_layer(event) == 'phase']
    global_tasks = [event for event in tasks if is_global_task_envelope(event)]
    if global_tasks:
        fails.append(('global_task_lane_forbidden',
                      f'发现 {len(global_tasks)} 个脱离 observer-warp 泳道的全局 task；'
                      'task 必须与其 phase 位于同一 pid/tid'))

    task_by_lane_and_id = defaultdict(list)
    missing_task_id = []
    invalid_local_parents = []
    for event in tasks:
        args = event.get('args') or {}
        task_id = args.get('task_id', args.get('source_task_id'))
        if not isinstance(task_id, int) or isinstance(task_id, bool):
            missing_task_id.append((event.get('pid'), event.get('tid'), event.get('name')))
            continue
        task_by_lane_and_id[(event.get('pid'), event.get('tid'), task_id)].append(event)
        if args.get('is_derived') is not False:
            invalid_local_parents.append((event.get('pid'), event.get('tid'), task_id,
                                          args.get('aggregation'),
                                          args.get('source_event_count'),
                                          args.get('is_derived')))
    if missing_task_id:
        fails.append(('task_id_missing',
                      f'{len(missing_task_id)} 个 task 缺少整数 task_id；samples={missing_task_id[:8]}'))
    if invalid_local_parents:
        fails.append(('local_task_source_pair_required',
                      f'{len(invalid_local_parents)} 个 task 不是一条真实 Device start/end pair；'
                      '禁止 derived、min/max、envelope 或其他聚合 task；'
                      f'samples={invalid_local_parents[:8]}'))

    missing_parent_id = []
    ambiguous_parent = []
    out_of_bounds = []
    for phase in phases:
        args = phase.get('args') or {}
        parent_id = args.get('parent_task_id')
        if parent_id is None:
            continue
        if not isinstance(parent_id, int) or isinstance(parent_id, bool):
            missing_parent_id.append((phase.get('pid'), phase.get('tid'),
                                      phase.get('name'), parent_id))
            continue
        phase_start, phase_end = _display_interval_bounds(phase)
        epsilon_ns = 1e-3
        candidates = task_by_lane_and_id.get(
            (phase.get('pid'), phase.get('tid'), parent_id), [])
        containing = []
        for task in candidates:
            task_start, task_end = _display_interval_bounds(task)
            if task_start <= phase_start + epsilon_ns and task_end + epsilon_ns >= phase_end:
                containing.append(task)
        if len(containing) == 1:
            continue
        if not containing:
            out_of_bounds.append({
                'pid': phase.get('pid'), 'tid': phase.get('tid'),
                'parent_task_id': parent_id,
                'phase': phase.get('name'),
                'phase_bounds_ns': [phase_start, phase_end],
                'candidate_count': len(candidates),
            })
        else:
            ambiguous_parent.append((phase.get('pid'), phase.get('tid'),
                                     phase.get('name'), parent_id, len(containing)))
    if missing_parent_id:
        fails.append(('phase_parent_id_invalid',
                      f'{len(missing_parent_id)} 个 phase 的 parent_task_id 非整数；'
                      f'samples={missing_parent_id[:8]}'))
    if out_of_bounds:
        fails.append(('phase_outside_task',
                      f'{len(out_of_bounds)} 个 phase 没有被同泳道对应 task 完整包含；'
                      f'samples={out_of_bounds[:8]}'))
    if ambiguous_parent:
        fails.append(('phase_parent_ambiguous',
                      f'{len(ambiguous_parent)} 个 phase 同时落入多个同泳道 parent task；'
                      f'samples={ambiguous_parent[:8]}'))
    notes.append(f'lane-local task/phase: tasks={len(tasks)}, phases={len(phases)}, '
                 f'parented_phases={sum(1 for event in phases if (event.get("args") or {}).get("parent_task_id") is not None)}')
    return fails, notes


def _identity_key(source):
    """Return a task/phase identity shared by ABI descriptors and trace spans.

    ABI descriptors keep id/name/short_name at the top level; trace spans keep
    task_id/short_name inside args.  Resolving both against the same key lets the
    static-phase-chain closure check match an observed phase to its declaration.
    """
    if not isinstance(source, dict):
        return None
    args = source.get('args') if isinstance(source.get('args'), dict) else {}

    def pick(field):
        value = source.get(field)
        return value if value is not None else args.get(field)

    for field in ('task_id', 'taskId', 'task_index', 'task_idx', 'id'):
        value = pick(field)
        if isinstance(value, int) and not isinstance(value, bool):
            return ('task_id', int(value))
    for field in ('short_name', 'name', 'task'):
        token = _stable_token(pick(field))
        if token is not None:
            return (field, token)
    return None


def _abi_display_descriptors(abi_path):
    """Return ABI task/phase descriptor dicts, or [] when none are available."""
    if not abi_path:
        return []
    abi, error = _load_json_file(Path(abi_path).expanduser().resolve())
    if error or abi is None:
        return []
    return [item for item in _semantic_abi_tasks(abi) if isinstance(item, dict)]


def _abi_phase_chain_order(abi_path):
    """Map each ABI-declared phase identity to its position in the static chain."""
    order = {}
    ranked = []
    for index, descriptor in enumerate(_abi_display_descriptors(abi_path)):
        if descriptor.get('layer') != 'phase':
            continue
        explicit = None
        for field in ('phase_index', 'chain_order', 'chain_index', 'order', 'sequence'):
            value = descriptor.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                explicit = value
                break
        key = _identity_key(descriptor)
        if key is not None:
            ranked.append((explicit if explicit is not None else index, index, key))
    ranked.sort(key=lambda item: (item[0], item[1]))
    for position, (_, _, key) in enumerate(ranked):
        order.setdefault(key, position)
    return order


def _resolve_scheduling_model(trace, explicit=None):
    """SPEC-3: contract.target.scheduling_model, else trace metadata, else strict.

    The field is optional; when absent we default to ``work_claiming`` so the
    strict scheduler-claim audit that ships today keeps running unchanged.
    """
    value = explicit
    if value is None and isinstance(trace, dict):
        other = trace.get('otherData') or {}
        contract = other.get('contract') if isinstance(other.get('contract'), dict) else {}
        for container in (other, other.get('target'), contract, contract.get('target')):
            if isinstance(container, dict) and container.get('scheduling_model') is not None:
                value = container.get('scheduling_model')
                break
    text = str(value).strip().lower() if value is not None else ''
    return text if text in ('static_phase_chain', 'work_claiming') else 'work_claiming'


def _contract_scheduling_model(contract_data):
    """SPEC-3: pull the declared scheduling_model out of a loaded contract.json.

    Prefers contract.target.scheduling_model, then a top-level scheduling_model.
    Returns None when the contract omits it so the resolver keeps its default.
    """
    if not isinstance(contract_data, dict):
        return None
    target = contract_data.get('target') if isinstance(contract_data.get('target'), dict) else {}
    for container in (target, contract_data):
        value = container.get('scheduling_model')
        if value is not None:
            return value
    return None


def _check_phase_chain_closure(spans, abi_path=None):
    """SPEC-3 static_phase_chain replacement for the scheduler-claim audit.

    A static phase chain has no device claim ledger, so requiring
    otherData.scheduler_claim_audit would permanently BLOCK it.  Instead, for
    each observer-warp lane the top-level phase slices must run in the
    ABI-declared order, must not partially overlap, and must not skip a declared
    phase that lies between two observed phases.  Tasks nested inside a phase are
    ignored here; they are validated by check_operator_task_shape.
    """
    fails = []
    notes = []
    phases = [event for event in spans
              if span_layer(event) == 'phase' and not is_global_task_envelope(event)]
    order_by_key = _abi_phase_chain_order(abi_path)
    lanes = defaultdict(list)
    for event in phases:
        start, end = _display_interval_bounds(event)
        if math.isfinite(start) and math.isfinite(end):
            lanes[(event.get('pid'), event.get('tid'))].append((start, end, event))

    partial = []
    out_of_order = []
    missing_gap = []
    unknown = []
    for lane, intervals in lanes.items():
        ordered = sorted(intervals, key=lambda item: (item[0], -item[1]))
        stack = []
        top_level = []
        active_end = None
        for start, end, event in ordered:
            while stack and stack[-1] <= start + _OVERLAP_EPS_US:
                stack.pop()
            if stack and end > stack[-1] + _OVERLAP_EPS_US:
                partial.append((lane, event.get('name'), start, end, stack[-1]))
            stack.append(end)
            if active_end is None or start >= active_end - _OVERLAP_EPS_US:
                top_level.append(event)
                active_end = end
        if not order_by_key:
            continue
        ranks = []
        for event in top_level:
            rank = order_by_key.get(_identity_key(event))
            if rank is None:
                unknown.append((lane, event.get('name')))
            else:
                ranks.append((rank, event.get('name')))
        rank_values = [rank for rank, _ in ranks]
        # A static phase chain may loop: the declared body re-executes every
        # iteration and prologue/drain phases (e.g. grid_barrier) repeat, so the
        # observed ranks legitimately revisit earlier phases. The global
        # "sorted ranks" / "no declared gap" audit only holds for a NON-looping
        # chain where each declared phase appears at most once. When any rank
        # repeats we treat the chain as looping and rely on the partial-overlap
        # (nesting/disjoint) and unknown-phase audits, which stay valid under
        # looping; intra-iteration temporal order is already pinned by the real
        # device start/end timestamps and the no-partial-overlap invariant.
        chain_loops = len(rank_values) != len(set(rank_values))
        if not chain_loops:
            if rank_values != sorted(rank_values):
                out_of_order.append((lane, [name for _, name in ranks]))
            ascending = sorted(rank_values)
            for left, right in zip(ascending, ascending[1:]):
                if right - left > 1:
                    missing_gap.append((lane, left, right))
    if partial:
        fails.append(('phase_chain_partial_overlap',
                      f'{len(partial)} phase slices partially overlap on an observer lane; '
                      f'a static phase chain must be disjoint or nested; samples={partial[:8]}'))
    if out_of_order:
        fails.append(('phase_chain_out_of_order',
                      f'phases are not in ABI-declared order on {len(out_of_order)} lanes; '
                      f'samples={out_of_order[:8]}'))
    if missing_gap:
        fails.append(('phase_chain_missing_declared',
                      f'a declared phase is missing between two observed phases on '
                      f'{len(missing_gap)} lanes; samples={missing_gap[:8]}'))
    notes.append(f'static_phase_chain closure: lanes={len(lanes)}, phases={len(phases)}, '
                 f'abi_declared_phases={len(order_by_key)}, unknown_phase_spans={len(unknown)}')
    return fails, notes


def check_task_claim_contract(trace, spans, scheduling_model=None, abi_path=None):
    """Reconcile delivered tasks against source-locked semantics or legacy scheduling."""
    model = _resolve_scheduling_model(trace, scheduling_model)
    abi = None
    if abi_path:
        abi, _abi_error = _load_json_file(Path(abi_path).expanduser().resolve())
    semantic_tasks = [descriptor for descriptor in task_descriptors(abi)
                      if isinstance(descriptor.get('task_contract'), dict)] \
        if isinstance(abi, dict) else []
    if semantic_tasks:
        semantic_failures = audit_semantic_trace_tasks(abi, trace)
        fails = [('semantic_task_contract', message)
                 for message in semantic_failures]
        notes = [f'source-locked semantic task contracts: declared={len(semantic_tasks)}, '
                 f'trace_failures={len(semantic_failures)}']
        if model == 'static_phase_chain':
            phase_fails, phase_notes = _check_phase_chain_closure(spans, abi_path)
            fails.extend(phase_fails)
            notes.extend(phase_notes)
        return fails, notes
    if model == 'static_phase_chain':
        return _check_phase_chain_closure(spans, abi_path)
    fails = []
    notes = []
    tasks = [event for event in spans if span_layer(event) == 'task']
    per_lane_ids = defaultdict(list)
    missing = []
    invalid = []
    required_by_kind = {
        'gemm_tile': ('group', 'm_block', 'n_block'),
        'tma_tile': ('group', 'm_block', 'n_block'),
        'dispatch_row': ('grouped_row',),
        'scatter_row': ('grouped_row',),
        'gemm0_tile': ('group', 'm_block', 'n_block'),
        'gemm1_tile': ('group', 'm_block', 'n_block'),
        'dispatch_chunk': ('first_grouped_row', 'rows'),
        'scatter_chunk': ('first_grouped_row', 'rows'),
        'combine_token': ('token',),
        'sync': ('sync_instance',),
    }
    for event in tasks:
        args = event.get('args') or {}
        instance_id = args.get('task_instance_id')
        claim_kind = args.get('claim_kind')
        claim_ticket = args.get('claim_ticket')
        claim_success = args.get('claim_success')
        sample = (event.get('pid'), event.get('tid'), event.get('name'))
        if instance_id is None or claim_kind not in required_by_kind or claim_ticket is None:
            missing.append(sample + (instance_id, claim_kind, claim_ticket))
            continue
        if claim_success is not True:
            invalid.append(sample + ('claim_success', claim_success))
        for field in required_by_kind[claim_kind]:
            if args.get(field) is None:
                invalid.append(sample + (f'missing_{field}', None))
        if args.get('claim_instance_count', 1) != 1:
            invalid.append(sample + ('claim_instance_count',
                                     args.get('claim_instance_count')))
        per_lane_ids[(event.get('pid'), event.get('tid'))].append(str(instance_id))
    if missing:
        fails.append(('task_instance_missing',
                      f'{len(missing)} task spans lack a single successful-claim identity; '
                      f'samples={missing[:8]}'))
    if invalid:
        fails.append(('task_instance_invalid',
                      f'{len(invalid)} task spans violate the per-claim tuple contract; '
                      f'samples={invalid[:8]}'))
    duplicates = []
    for lane, values in per_lane_ids.items():
        repeated = [key for key, count in Counter(values).items() if count != 1]
        if repeated:
            duplicates.append((lane, repeated[:8]))
    if duplicates:
        fails.append(('task_instance_duplicate',
                      f'task_instance_id must be unique within each CTA/warp lane; '
                      f'samples={duplicates[:8]}'))

    audit = (trace.get('otherData') or {}).get('scheduler_claim_audit') \
        if isinstance(trace, dict) else None
    if not isinstance(audit, list):
        fails.append(('scheduler_claim_audit_missing',
                      'otherData.scheduler_claim_audit is required; task cardinality must be '
                      'reconciled against successful device claims, not inferred from stage spans'))
        return fails, notes
    audit_map = {}
    audit_errors = []
    for index, row in enumerate(audit):
        if not isinstance(row, dict):
            audit_errors.append((index, 'not_object'))
            continue
        pid, tid = row.get('pid'), row.get('tid')
        ids = row.get('task_instance_ids')
        count = row.get('successful_claims')
        if pid is None or tid is None or not isinstance(ids, list) \
                or not isinstance(count, int) or isinstance(count, bool) \
                or count != len(ids) or len(set(map(str, ids))) != len(ids):
            audit_errors.append((index, pid, tid, count,
                                 None if not isinstance(ids, list) else len(ids)))
            continue
        audit_map[(pid, tid)] = set(map(str, ids))
    if audit_errors:
        fails.append(('scheduler_claim_audit_invalid',
                      f'invalid scheduler claim audit rows; samples={audit_errors[:8]}'))
    mismatches = []
    all_lanes = set(per_lane_ids) | set(audit_map)
    for lane in sorted(all_lanes, key=str):
        traced = set(per_lane_ids.get(lane, ()))
        claimed = audit_map.get(lane, set())
        if traced != claimed:
            mismatches.append((lane, len(traced), len(claimed),
                               sorted(traced - claimed)[:4],
                               sorted(claimed - traced)[:4]))
    if mismatches:
        fails.append(('scheduler_claim_task_mismatch',
                      f'task instances must equal successful scheduler claims lane-by-lane; '
                      f'samples={mismatches[:8]}'))
    notes.append(f'task claim contract: tasks={len(tasks)}, audited_lanes={len(audit_map)}')
    return fails, notes


HW_CAPTURE_RANGE = 'cudaProfilerApi'
HW_CAPTURE_RANGE_END = 'stop'
HW_PREFERRED_FREQUENCY_HZ = 200000

# Contract uses decimal megabytes.  Keep this literal to avoid silently
# changing the delivery boundary to binary MiB.
TRACE_SIZE_THRESHOLD_BYTES = 10_000_000
# The size ladder is locked: hide every phase first, then halve the SM set from
# the full grid.  Spelling it out in the failure itself stops the two shortcuts
# people reach for instead -- raising the threshold, or sampling events away.
TRACE_SIZE_REMEDY = ('补救阶梯只有两级: ① 先隐藏全部 phase 只留 task; '
                     '② 仍超限再从满 grid 折半 SM (如 78->39->20->10->5->3->2->1). '
                     '严禁抬高 10MB 阈值、严禁抽样或丢弃事件、严禁改变已声明的 SM 集合')
COMPACT_EXACT_SCHEMA = 'perfetto-compact-exact/v1'
SEMANTIC_VIEWER_SCHEMA = 'perfetto-semantic-viewer/v2'

# Perfetto display coordinates (ts/dur) are microseconds derived from integer
# nanosecond device timestamps, so the smallest real interval boundary is
# 1e-3 us (= 1 ns). A stored `ts` and a neighbouring interval's computed
# `ts + dur` that are the SAME nanosecond can differ by ~1 ULP (~1e-13 us near
# 1e3 us), which a strict `>` comparison misreads as a partial overlap. This
# tolerance (1e-3 ns) mirrors the ns-domain find_partial_overlaps audit: it
# absorbs that conversion noise while staying 1000x below a real 1 ns boundary,
# so any genuine same-lane crossing is still caught.
_OVERLAP_EPS_US = 1e-6

# SPEC-1: the delivered candidate/final trace filename must be self-describing.
# Canonical schema:
#   trace__<model>_<stage>__card-<h20|b300|mi308x>__bs<B>_sq<S>[_kv<K>]__<dtype>__grid<G>_warp<W>_ph<P>__<bundle>.perfetto.json.gz
# where bundle ∈ {sw, hw, sw_hw} (the on-disk bundle token uses '_' whereas
# contract.information_bundle uses '+'; the two must not be conflated).
CARD_TARGETS = {
    'h20': {'platform': 'nv', 'arches': frozenset(('sm_90',))},
    'b300': {'platform': 'nv', 'arches': frozenset(('sm_100a', 'sm_103a'))},
    'mi308x': {'platform': 'amd', 'arches': frozenset(('gfx942',))},
}
SUPPORTED_CARDS = tuple(CARD_TARGETS)
TRACE_FILENAME_SCHEMA = (
    'trace__<model>_<stage>__card-<h20|b300|mi308x>__bs<B>_sq<S>[_kv<K>]__'
    '<dtype>__grid<G>_warp<W>_ph<P>__<bundle>.perfetto.json.gz')
CANONICAL_TRACE_FILENAME_RE = re.compile(
    r'^trace__(?P<model>[a-z0-9]+)_(?P<stage>[a-z0-9]+)'
    r'__card-(?P<card>h20|b300|mi308x)'
    r'__bs(?P<batch>\d+)_sq(?P<seqlen_q>\d+)'
    r'(?:_kv(?P<seqlen_kv>\d+))?'
    r'__(?P<dtype>[a-z0-9]+)'
    r'__grid(?P<grid>\d+)_warp(?P<warp>\d+)_ph(?P<phases>\d+)'
    r'__(?P<bundle>sw|hw|sw_hw)\.perfetto\.json\.gz$')
TRACE_RECEIPT_SUFFIX = '.delivery_receipt.json'
BUNDLE_TOKENS = {'sw': 'sw', 'hw': 'hw', 'sw+hw': 'sw_hw'}


def normalize_card(value):
    """Return the canonical card token, or None for an unknown/missing value."""
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    return token if token in CARD_TARGETS else None


def normalize_platform(value):
    """Normalize the contract platform spelling to the canonical nv/amd token."""
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    aliases = {'nvidia': 'nv', 'cuda': 'nv', 'rocm': 'amd', 'hip': 'amd'}
    token = aliases.get(token, token)
    return token if token in ('nv', 'amd') else None


def normalize_arch(value):
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    return token or None


def parse_trace_filename(trace_path):
    """Parse one canonical delivery name; return None for any non-canonical name."""
    match = CANONICAL_TRACE_FILENAME_RE.match(Path(str(trace_path)).name)
    return match.groupdict() if match else None


def validate_target_card(contract):
    """Validate contract.target.card against platform/arch, fail closed.

    Returns ``(canonical_card, failures)``.  B300 intentionally accepts both
    sm_100a and sm_103a; it must never be narrowed to one architecture.
    """
    fails = []
    target = contract.get('target') if isinstance(contract, dict) else None
    if not isinstance(target, dict):
        return None, [('target_missing',
                       '契约缺少 object 类型的 target；必须声明 target.card / platform / arch')]

    card = normalize_card(target.get('card'))
    raw_card = target.get('card')
    if raw_card is None:
        fails.append(('target_card_missing',
                      '契约缺少 target.card；合法值只有 h20、b300、mi308x'))
    elif card is None:
        fails.append(('target_card_invalid',
                      f'契约 target.card={raw_card!r} 非法；合法值只有 h20、b300、mi308x'))

    raw_platform = target.get('platform')
    platform = normalize_platform(raw_platform)
    if raw_platform is None:
        fails.append(('target_platform_missing',
                      '契约缺少 target.platform；必须为 nv 或 amd'))
    elif platform is None:
        fails.append(('target_platform_invalid',
                      f'契约 target.platform={raw_platform!r} 非法；必须为 nv 或 amd'))

    raw_arch = target.get('arch')
    arch = normalize_arch(raw_arch)
    if raw_arch is None:
        fails.append(('target_arch_missing',
                      '契约缺少 target.arch；必须声明真实目标架构'))
    elif arch is None:
        fails.append(('target_arch_invalid',
                      f'契约 target.arch={raw_arch!r} 非法；必须是非空字符串'))

    if card is not None:
        expected = CARD_TARGETS[card]
        if platform is not None and platform != expected['platform']:
            fails.append(('target_platform_mismatch',
                          f'card {card} 必须对应 platform={expected["platform"]}，'
                          f'契约声明的是 {platform}'))
        if arch is not None and arch not in expected['arches']:
            accepted = '|'.join(sorted(expected['arches']))
            fails.append(('target_arch_mismatch',
                          f'card {card} 只接受 arch={accepted}，契约声明的是 {arch}'))
    return card, fails


def check_trace_filename_schema(trace_path, contract=None):
    """SPEC-1: reject a delivery whose filename or target declaration violates schema.

    A conforming name is what lets a trace be attributed to a model/stage/card/
    shape/bundle on OSS without opening it.  When a contract is supplied, its
    canonical target.card must exist, agree with the filename, and agree with
    target.platform/arch.  A mismatch is always a hard delivery fail.
    """
    fails = []
    notes = []
    name = Path(trace_path).name
    parsed = parse_trace_filename(name)
    if parsed is None:
        fails.append(('trace_filename_schema',
                      f'trace filename {name!r} violates canonical schema '
                      f'{TRACE_FILENAME_SCHEMA} (bundle ∈ {{sw, hw, sw_hw}})'))
        return fails, notes

    notes.append(f'trace filename schema OK: {name} (card={parsed["card"]})')
    if contract is not None:
        card, target_fails = validate_target_card(contract)
        fails.extend(target_fails)
        if card is not None and card != parsed['card']:
            fails.append(('trace_filename_card_mismatch',
                          f'文件名 card={parsed["card"]} 与契约 target.card={card} 不符'))
    return fails, notes


def _display_interval_bounds(event):
    """Return the interval as displayed by Perfetto (ts/dur are authoritative).

    Some builders retain pre-rebase/aligned values in args.  Using those args
    for containment can reject a valid trace whose display coordinates were
    rebased after merge.
    """
    try:
        start = float(event.get('ts'))
        dur = float(event.get('dur', 0))
    except (TypeError, ValueError):
        return math.inf, -math.inf
    return start, start + dur


def check_nonnegative_timestamps(events):
    """Return all timed events whose numeric display ``ts`` is negative."""
    return [event for event in events
            if isinstance(event.get('ts'), (int, float))
            and not isinstance(event.get('ts'), bool)
            and math.isfinite(float(event.get('ts')))
            and float(event.get('ts')) < 0]


def check_no_partial_overlap(events):
    """Return same-lane X crossings; disjoint/nested intervals are allowed.

    Intervals sharing a lane may be disjoint or nested.  Sorting by
    ``(start asc, end desc)`` makes equal-start outer intervals precede their
    children; a stack then detects crossings without rejecting valid nesting.
    """
    partial = []
    lanes = defaultdict(list)
    for event in events:
        if event.get('ph') != 'X':
            continue
        start, end = _display_interval_bounds(event)
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        lanes[(event.get('pid'), event.get('tid'))].append((start, end, event))
    for lane, intervals in lanes.items():
        stack = []
        for start, end, event in sorted(intervals, key=lambda item: (item[0], -item[1])):
            while stack and stack[-1] <= start + _OVERLAP_EPS_US:
                stack.pop()
            if stack and end > stack[-1] + _OVERLAP_EPS_US:
                partial.append((lane, event.get('name'), start, end, stack[-1]))
            stack.append(end)
    return partial


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _event_geometry(event):
    return [event.get(field) for field in ('name', 'cat', 'ph', 'pid', 'tid', 'ts', 'dur')]


def _geometry_digest(events):
    digest = hashlib.sha256()
    for event in events:
        digest.update(_canonical_json(_event_geometry(event)).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def _event_digest(events):
    digest = hashlib.sha256()
    for event in events:
        digest.update(_canonical_json(event).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _check_perfetto_schema(trace):
    """Check the minimal Chrome/Perfetto JSON schema before delivery."""
    fails = []
    events = trace.get('traceEvents') if isinstance(trace, dict) else None
    if not isinstance(events, list):
        return [('delivery_perfetto_schema', 'traceEvents must be a JSON array')]
    allowed_ph = {'X', 'C', 'M', 'I', 'B', 'E', 'b', 'e', 'n', 's', 't'}
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            fails.append(('delivery_perfetto_event_schema', f'traceEvents[{index}] is not object'))
            continue
        phase = event.get('ph')
        if phase not in allowed_ph:
            fails.append(('delivery_perfetto_event_phase',
                          f'traceEvents[{index}].ph={phase!r} is not a Perfetto phase'))
        args = event.get('args')
        if phase in {'C', 'M'} and not isinstance(args, dict):
            fails.append(('delivery_perfetto_args_schema',
                          f'traceEvents[{index}] ph={phase} requires object args'))
        if phase == 'C' and isinstance(args, dict):
            if not args:
                fails.append(('delivery_perfetto_counter_args',
                              f'traceEvents[{index}] counter args must not be empty'))
            for key, value in args.items():
                # Perfetto counter values are numeric.  In Python bool is an int
                # subclass, so reject it explicitly: true/false caused one JSON
                # parser error per field in the 2026-08-26 regression.
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(float(value))):
                    fails.append(('delivery_perfetto_counter_value',
                                  f'traceEvents[{index}].args[{key!r}]={value!r}; '
                                  'ph=C args values must be finite numbers, never bool/string/null'))
        if phase in {'X', 'C'}:
            ts = _number(event.get('ts'))
            if ts is None or not math.isfinite(float(ts)):
                fails.append(('delivery_perfetto_timing_schema',
                              f'traceEvents[{index}] ph={phase} has invalid ts'))
            if event.get('pid') is None or event.get('tid') is None:
                fails.append(('delivery_perfetto_lane_schema',
                              f'traceEvents[{index}] ph={phase} lacks pid/tid'))
        if phase == 'X':
            ts = _number(event.get('ts'))
            dur = _number(event.get('dur'))
            if ts is None or dur is None or dur < 0:
                fails.append(('delivery_perfetto_timing_schema',
                              f'traceEvents[{index}] has invalid X ts/dur'))
    other = trace.get('otherData') if isinstance(trace, dict) else None
    producer = other.get('producer') if isinstance(other, dict) else None
    if not isinstance(producer, str) or not producer.endswith('build_perfetto.py'):
        fails.append(('delivery_builder_provenance',
                      f'otherData.producer={producer!r}; final trace must come from official '
                      'build_perfetto.py, not a custom JSON emitter'))
    return fails


# Perfetto stats that mean the delivered .perfetto.json trace SILENTLY lost or
# corrupted data during a REAL trace_processor import. Every name below is
# severity 'error' in Perfetto's own `stats` table and is reachable by the JSON
# complete-event traces build_perfetto.py emits. Subsystem-specific
# *_truncated/*_overflow stats (perf_aux_truncated, android_video_*_cap_hit,
# metatrace_overruns, ...) can never fire for our JSON trace type and are
# deliberately excluded. A non-zero value on ANY of these must FAIL G24.
FORBIDDEN_IMPORT_STATS = (
    # JSON importer, both stages: rejected event objects (parser) or rejected
    # byte-stream tokens (tokenizer) are dropped from the delivered .json.
    'json_parser_failure',
    'json_tokenizer_failure',
    # Overlapping ph=X complete events on one track: Perfetto either spills the
    # offender onto a synthetic depth or DROPS it outright; both corrupt the
    # timeline, so forbid the drop sibling too, not only the spill we caught
    # before.
    'slice_spill_overlapping_complete_event',
    'slice_drop_overlapping_complete_event',
    # The trace sorter SILENTLY drops any event with ts<0 while still exiting
    # rc=0 (verified: a negative-ts slice vanishes and import 'succeeds'). G18
    # already requires finite, non-negative time; without this stat G24 saw no
    # evidence of the violation. Real 2026 regression: 936 slices dropped.
    'trace_sorter_negative_timestamp_dropped',
)


def _check_real_perfetto_import(trace_path, executable=None):
    """Import the final artifact with trace_processor_shell and reject known errors."""
    fails = []
    notes = []
    tool = executable or shutil.which('trace_processor_shell')
    if not tool:
        return [('perfetto_import_tool_missing',
                 'trace_processor_shell is required for delivery; static JSON checks are not enough')], notes
    tool_path = Path(tool).expanduser().resolve()
    if not tool_path.is_file():
        return [('perfetto_import_tool_missing',
                 f'trace_processor_shell does not exist: {tool_path}')], notes
    forbidden_sql = ',\n    '.join(f"'{name}'" for name in FORBIDDEN_IMPORT_STATS)
    # trace_processor_shell accepts exactly one SELECT in ``-Q``.  Use scalar
    # subqueries so the gate works on the pinned v47 binary required by older
    # glibc hosts as well as on current releases.
    sql = (
        'WITH bad AS (\n'
        '  SELECT name FROM stats\n'
        '  WHERE value > 0 AND name IN (\n    '
        + forbidden_sql
        + '\n  )\n'
        ')\n'
        'SELECT\n'
        '  (SELECT COUNT(*) FROM bad) AS forbidden_stat_count,\n'
        '  (SELECT GROUP_CONCAT(name) FROM (SELECT name FROM bad ORDER BY name))'
        ' AS forbidden_stats,\n'
        '  (SELECT COUNT(*) FROM slice) AS imported_slices;'
    )
    proc = subprocess.run([str(tool_path), '-Q', sql, str(trace_path)],
                          capture_output=True, text=True, timeout=180)
    output = (proc.stdout or '') + '\n' + (proc.stderr or '')
    notes.append(f'trace_processor_shell = {tool_path}')
    notes.append(f'trace_processor returncode = {proc.returncode}')
    if proc.returncode != 0:
        fails.append(('perfetto_real_import',
                      f'trace_processor_shell failed rc={proc.returncode}: {output[-2000:]}'))
        return fails, notes
    try:
        rows = list(csv.DictReader(io.StringIO(proc.stdout or '')))
        row = rows[0] if rows else {}
        forbidden_count = int(row.get('forbidden_stat_count'))
        imported_slices = int(row.get('imported_slices'))
    except (TypeError, ValueError, IndexError):
        fails.append(('perfetto_import_result_parse',
                      'real import did not return integer forbidden_stat_count/'
                      f'imported_slices columns: {(proc.stdout or "")[-1000:]}'))
        return fails, notes
    forbidden_stats = row.get('forbidden_stats')
    found = [name for name in FORBIDDEN_IMPORT_STATS
             if name in (forbidden_stats or '')]
    if forbidden_count != 0 or found:
        fails.append(('perfetto_import_stats',
                      f'non-zero Perfetto import/trace errors: count={forbidden_count} '
                      f'stats={found or forbidden_stats}'))
    if imported_slices <= 0:
        fails.append(('perfetto_import_empty',
                      f'real import produced no slices: imported_slices={imported_slices}'))
    notes.append(f'Perfetto real import completed; imported_slices={imported_slices}; '
                 'forbidden stats = '
                 + (', '.join(found) if found else '0'))
    return fails, notes


def _semantic_event_identity(event):
    """Return an ABI identity without interpreting project-specific names."""
    args = event.get('args') or {}
    for field in ('task_id', 'taskId', 'task', 'task_name', 'source_name'):
        token = _stable_token(args.get(field, event.get(field)))
        if token is not None:
            return field, token
    token = _stable_token(event.get('name'))
    return ('name', token) if token is not None else None


def _semantic_abi_identity(record):
    if not isinstance(record, dict):
        return None
    normalized = dict(record)
    if 'id' in normalized and not any(key in normalized for key in
                                      ('task_id', 'taskId', 'task_index', 'task_idx')):
        normalized['task_id'] = normalized['id']
    return _semantic_event_identity({'name': normalized.get('name'), 'args': normalized})


def _semantic_abi_tasks(abi):
    if isinstance(abi, list):
        return abi
    if not isinstance(abi, dict):
        return []
    for key in ('tasks', 'events', 'records'):
        if isinstance(abi.get(key), list):
            return abi[key]
    nested = abi.get('visibility_contract')
    return nested.get('tasks', []) if isinstance(nested, dict) \
        and isinstance(nested.get('tasks'), list) else []


def _semantic_layer(event):
    args = event.get('args') or {}
    layer = args.get('layer', event.get('layer'))
    return layer if layer in ('task', 'phase', 'ignored') else span_layer(event)


def _semantic_descriptor_matches(event, descriptor):
    event_layer = _semantic_layer(event)
    if event_layer != descriptor['layer']:
        # The builder emits summary slices with layer=summary while their ABI
        # descriptor remains the source phase and viewer_policy=summary.
        if not (descriptor.get('policy') == 'summary'
                and descriptor['layer'] == 'phase' and event_layer == 'summary'):
            return False
    if _semantic_event_identity(event) != descriptor['identity']:
        return False
    kind = descriptor.get('semantic_kind')
    if kind is not None:
        args = event.get('args') or {}
        if str(args.get('semantic_kind', event.get('semantic_kind'))) != str(kind):
            return False
    return True


def _semantic_descriptors(abi, fails):
    descriptors = []
    for index, raw in enumerate(_semantic_abi_tasks(abi)):
        if not isinstance(raw, dict):
            fails.append(('semantic_viewer_abi_task_schema', f'ABI tasks[{index}] 必须是对象'))
            continue
        layer = raw.get('layer')
        if layer not in ('task', 'phase', 'ignored'):
            fails.append(('semantic_viewer_abi_task_layer',
                          f'ABI tasks[{index}].layer={layer!r} 无效'))
            continue
        identity = _semantic_abi_identity(raw)
        if identity is None:
            fails.append(('semantic_viewer_abi_task_identity',
                          f'ABI tasks[{index}] 缺少 task_id/task/name identity'))
            continue
        descriptors.append({'index': index, 'raw': raw, 'layer': layer,
                            'identity': identity,
                            'semantic_kind': raw.get('semantic_kind'),
                            'policy': raw.get('viewer_policy') or raw.get('visibility')})
    return descriptors


def _semantic_count(mapping, descriptor):
    if not isinstance(mapping, dict):
        return None
    field, token = descriptor['identity']
    value = None
    for key in (token, str(token), f'{field}:{token}'):
        if key in mapping:
            value = mapping[key]
            break
    if isinstance(value, dict):
        by_layer = value.get('by_layer')
        if isinstance(by_layer, dict) and descriptor['layer'] in by_layer:
            value = by_layer[descriptor['layer']]
        else:
            value = value.get(descriptor['layer'], value.get('count',
                           value.get('event_count', value.get('source_count',
                           value.get('output_count')))))
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _semantic_parent_metadata(global_parents, descriptor):
    if isinstance(global_parents, list):
        return [item for item in global_parents if isinstance(item, dict)
                and _semantic_abi_identity(item) == descriptor['identity']]
    if isinstance(global_parents, dict):
        field, token = descriptor['identity']
        for key in (token, str(token), f'{field}:{token}'):
            if key in global_parents:
                value = global_parents[key]
                return value if isinstance(value, list) else [value]
    return []


def _semantic_summary_allowed(contract, kind):
    if kind is None or not isinstance(contract, dict):
        return False
    allowed = set()
    for key in ('summary_allowed_semantic_kinds', 'allowed_summary_semantic_kinds',
                'summary_semantic_kinds'):
        values = contract.get(key)
        if isinstance(values, (list, tuple, set)):
            allowed.update(str(value) for value in values)
    rules = contract.get('semantic_kind_rules')
    if isinstance(rules, dict):
        rule = rules.get(str(kind))
        if isinstance(rule, dict) and (rule.get('allow_summary') is True
                                       or rule.get('summary') is True):
            return True
    for key in ('summary_exceptions', 'allowed_summaries', 'summary_allowed'):
        exceptions = contract.get(key)
        if isinstance(exceptions, dict):
            value = exceptions.get(str(kind))
            if value is True or (isinstance(value, dict) and
                                 (value.get('allow') is True or value.get('allowed') is True)):
                return True
            if isinstance(value, (list, tuple, set)) and any(
                    str(scope).lower() in ('cta', 'observer', 'observer_lane', 'cta_lane')
                    for scope in value):
                return True
        elif isinstance(exceptions, list):
            for item in exceptions:
                if isinstance(item, dict) and str(item.get('semantic_kind')) == str(kind):
                    if item.get('allow') is not False and item.get('allowed') is not False:
                        return True
    return str(kind) in allowed


def _semantic_local_summary(event):
    args = event.get('args') or {}
    aggregation = str(args.get('aggregation', '')).lower()
    category = str(event.get('cat', '')).lower()
    scope = str(args.get('scope', args.get('parent_scope', ''))).lower()
    if scope in ('global_trace', 'global'):
        return False
    return ('summary' in category or 'summary' in aggregation or 'cta' in aggregation
            or 'observer' in aggregation or args.get('is_summary') is True)


# Representative-SM delivery is intentionally a small, explicit ABI.  Do not
# infer it from labels or from an arbitrary modulo predicate: a delivery must
# carry the complete physical-SM source ledger and a deterministic halving
# history.  These helpers are kept here (rather than in the builder) because
# the delivery gate is the final trust boundary.
_SM_ID_FIELDS = ('physical_smid', 'physical_sm_id', 'physical_sm', 'smid', 'sm_id')
_SM_SOURCE_ID_FIELDS = (
    'source_physical_smids', 'original_physical_smids',
    'all_physical_smids', 'source_physical_sms', 'physical_smids',
)
_SM_COUNT_FIELDS = (
    'source_event_counts_by_physical_smid',
    'source_event_counts_by_physical_sm',
    'source_event_counts_by_smid',
    'source_counts_by_physical_smid',
)
_SM_HALVING_POLICIES = {
    'physical_sm_halving', 'deterministic_physical_sm_halving',
    'iterative_physical_sm_halving', 'representative_physical_sm_halving',
}


def _physical_sm_id(event):
    args = event.get('args') if isinstance(event, dict) else None
    if not isinstance(args, dict):
        args = {}
    for field in _SM_ID_FIELDS:
        value = args.get(field, event.get(field) if isinstance(event, dict) else None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _int_id_list(value):
    if not isinstance(value, list) or not value:
        return None
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        return None
    if len(set(value)) != len(value) or value != sorted(value):
        return None
    return list(value)


def _metadata_sm_ids(viewer, source):
    nested_filter = viewer.get('sm_display_filter') if isinstance(viewer, dict) else None
    for owner in (viewer, nested_filter, source):
        if not isinstance(owner, dict):
            continue
        for field in _SM_SOURCE_ID_FIELDS:
            ids = _int_id_list(owner.get(field))
            if ids is not None:
                return ids, field
    return None, None


def _metadata_sm_counts(viewer, source):
    nested_filter = viewer.get('sm_display_filter') if isinstance(viewer, dict) else None
    for owner in (viewer, nested_filter, source):
        if not isinstance(owner, dict):
            continue
        for field in _SM_COUNT_FIELDS:
            value = owner.get(field)
            if isinstance(value, dict):
                return value, field
    return None, None


def _count_by_sm(events):
    counts = Counter()
    for event in events:
        if not isinstance(event, dict) or event.get('ph') != 'X':
            continue
        smid = _physical_sm_id(event)
        if smid is not None:
            counts[smid] += 1
    return counts


def _sm_count_value(mapping, smid):
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(smid, mapping.get(str(smid)))
    if isinstance(value, dict):
        value = value.get('count', value.get('event_count', value.get('source_count')))
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _round_ids(record, names):
    for name in names:
        value = _int_id_list(record.get(name)) if isinstance(record, dict) else None
        if value is not None:
            return value
    return None


def _check_representative_sm_policy(viewer, source, source_events, output_events,
                                    descriptors):
    """Enforce all-source evidence and deterministic repeated halving.

    Return failures, notes, and whether a representative filter was present.
    The source trace remains the complete all-SM capture; only the viewer's X
    events may be restricted.  ph=C is checked separately by the source/output
    digest gate and is never part of the SM filter.
    """
    fails, notes = [], []
    display = viewer.get('sm_display_filter') if isinstance(viewer, dict) else None
    if not isinstance(display, dict):
        return fails, notes, False

    source_ids_actual = sorted({smid for event in source_events
                                if isinstance(event, dict) and event.get('ph') == 'X'
                                for smid in [_physical_sm_id(event)] if smid is not None})
    output_ids_actual = sorted({smid for event in output_events
                                if isinstance(event, dict) and event.get('ph') == 'X'
                                for smid in [_physical_sm_id(event)] if smid is not None})
    source_ids, source_ids_field = _metadata_sm_ids(viewer, source)
    all_sm_claim = viewer.get('source_capture_was_all_sm')
    if all_sm_claim is None:
        all_sm_claim = display.get('source_capture_was_all_sm')
    if all_sm_claim is None and isinstance(source, dict):
        all_sm_claim = source.get('capture_scope') in ('all physical SMs', 'all_physical_sms')
    if all_sm_claim is not True:
        fails.append(('semantic_viewer_source_all_sm_missing',
                      '代表性 SM viewer 必须声明 source_capture_was_all_sm=true'))
    if source_ids is None:
        fails.append(('semantic_viewer_source_sm_ids_missing',
                      '代表性 SM viewer 必须提供完整 source_physical_smids'))
        source_ids = []
    if source_ids != source_ids_actual:
        fails.append(('semantic_viewer_source_sm_ids_mismatch',
                      f'source SM metadata={source_ids} actual={source_ids_actual}'))
    if not source_ids_actual:
        fails.append(('semantic_viewer_source_sm_empty',
                      'source all-SM evidence contains no physical SM X event'))

    source_counts, source_counts_field = _metadata_sm_counts(viewer, source)
    actual_counts = _count_by_sm(source_events)
    if source_counts is None:
        fails.append(('semantic_viewer_source_sm_counts_missing',
                      '代表性 SM viewer 必须提供每个 physical SM 的 source event count'))
    else:
        metadata_keys = set()
        for raw_key, value in source_counts.items():
            try:
                key = int(raw_key)
            except (TypeError, ValueError):
                continue
            if str(key) == str(raw_key) or key == raw_key:
                metadata_keys.add(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                fails.append(('semantic_viewer_source_sm_count_invalid',
                              f'{source_counts_field}[{raw_key!r}] must be a positive integer'))
        if metadata_keys != set(source_ids_actual):
            fails.append(('semantic_viewer_source_sm_count_coverage',
                          f'source SM count keys={sorted(metadata_keys)} actual={source_ids_actual}'))
        for smid in source_ids_actual:
            value = _sm_count_value(source_counts, smid)
            if value != actual_counts.get(smid, 0):
                fails.append(('semantic_viewer_source_sm_count_mismatch',
                              f'physical SM {smid} metadata count={value} actual={actual_counts.get(smid, 0)}'))

    policy = display.get('policy')
    if policy not in _SM_HALVING_POLICIES:
        fails.append(('semantic_viewer_sm_policy_invalid',
                      f'sm_display_filter.policy={policy!r} must be deterministic physical-SM halving'))
    if 'stride' in display or 'predicate' in display or policy in ('physical_smid_modulo', 'modulo'):
        fails.append(('semantic_viewer_sm_stride_forbidden',
                      '代表性 SM 禁止 modulo/stride 选择，必须使用逐轮折半'))

    displayed_ids = _round_ids(display, ('displayed_physical_smids', 'final_physical_smids'))
    if displayed_ids is None:
        fails.append(('semantic_viewer_display_sm_ids_missing',
                      '代表性 SM viewer 必须提供 displayed_physical_smids'))
        displayed_ids = []
    if len(displayed_ids) < 2:
        fails.append(('semantic_viewer_display_sm_too_few',
                      f'displayed physical SM count={len(displayed_ids)}; representative viewer must keep at least 2'))
    if source_ids and (len(displayed_ids) >= len(source_ids)
                       or not set(displayed_ids).issubset(set(source_ids))):
        fails.append(('semantic_viewer_display_sm_set_invalid',
                      f'displayed SMs={displayed_ids} must be a strict subset of source SMs={source_ids}'))
    count_fields = (('source_sm_count', 'original_sm_count'),
                    ('displayed_sm_count', 'display_sm_count'))
    for fields, expected in ((count_fields[0], len(source_ids)),
                             (count_fields[1], len(displayed_ids))):
        field = next((name for name in fields if name in display), None)
        if field is None:
            fails.append(('semantic_viewer_sm_count_missing',
                          f'sm_display_filter requires one of {fields}'))
        elif display.get(field) != expected:
            fails.append(('semantic_viewer_sm_count_mismatch',
                          f'{field}={display.get(field)!r} expected={expected}'))

    rounds = display.get('rounds', display.get('halving_rounds'))
    if not isinstance(rounds, list) or not rounds:
        fails.append(('semantic_viewer_sm_rounds_missing',
                      '代表性 SM viewer 必须提供非空 rounds halving ledger'))
        rounds = []
    selector = display.get('selection_rule', 'every_other_even_index')
    selector_map = {
        'every_other_even_index': 0, 'keep_even_index': 0,
        'every_other_odd_index': 1, 'keep_odd_index': 1,
    }
    if selector not in selector_map:
        fails.append(('semantic_viewer_sm_selection_invalid',
                      f'selection_rule={selector!r} is not a deterministic every-other rule'))
    parity = selector_map.get(selector, 0)
    previous = source_ids
    expected_round = None
    for index, record in enumerate(rounds):
        if not isinstance(record, dict):
            fails.append(('semantic_viewer_sm_round_schema', f'rounds[{index}] must be an object'))
            continue
        round_value = record.get('round', record.get('round_index'))
        if not isinstance(round_value, int) or isinstance(round_value, bool):
            fails.append(('semantic_viewer_sm_round_invalid', f'rounds[{index}].round must be an integer'))
        elif expected_round is None:
            expected_round = round_value
        elif round_value != expected_round + index:
            fails.append(('semantic_viewer_sm_round_sequence',
                          f'rounds[{index}].round={round_value} is not contiguous'))
        input_ids = _round_ids(record, ('input_physical_smids', 'source_physical_smids', 'from_physical_smids'))
        output_round_ids = _round_ids(record, ('output_physical_smids', 'displayed_physical_smids', 'to_physical_smids'))
        if input_ids is None or output_round_ids is None:
            fails.append(('semantic_viewer_sm_round_ids_missing',
                          f'rounds[{index}] must provide input/output physical SM IDs'))
            continue
        if input_ids != previous:
            fails.append(('semantic_viewer_sm_round_input_mismatch',
                          f'rounds[{index}] input={input_ids} previous={previous}'))
        expected = input_ids[parity::2]
        if output_round_ids != expected:
            fails.append(('semantic_viewer_sm_round_not_halving',
                          f'rounds[{index}] output={output_round_ids} expected={expected}'))
        if len(output_round_ids) != (len(input_ids) + 1) // 2:
            fails.append(('semantic_viewer_sm_round_count_invalid',
                          f'rounds[{index}] counts {len(input_ids)} -> {len(output_round_ids)} are not half'))
        for fields, expected_count in ((('input_count', 'source_count'), len(input_ids)),
                                       (('output_count', 'displayed_count'), len(output_round_ids))):
            field = next((name for name in fields if name in record), None)
            if field is None:
                fails.append(('semantic_viewer_sm_round_count_missing',
                              f'rounds[{index}] requires one of {fields}'))
            elif record.get(field) != expected_count:
                fails.append(('semantic_viewer_sm_round_count_mismatch',
                              f'rounds[{index}].{field}={record.get(field)!r} expected={expected_count}'))
        previous = output_round_ids
    if rounds and previous != displayed_ids:
        fails.append(('semantic_viewer_sm_round_final_mismatch',
                      f'final round IDs={previous} displayed IDs={displayed_ids}'))
    final_round_field = 'round' if 'round' in display else 'final_round' if 'final_round' in display else None
    if rounds and final_round_field is None:
        fails.append(('semantic_viewer_sm_round_metadata_missing',
                      'sm_display_filter.round/final_round is required'))
    if rounds and final_round_field is not None:
        final_round = rounds[-1].get('round', rounds[-1].get('round_index')) \
            if isinstance(rounds[-1], dict) else None
        if display.get(final_round_field) != final_round:
            fails.append(('semantic_viewer_sm_round_metadata_mismatch',
                          f'display.{final_round_field}={display.get(final_round_field)} final round={final_round}'))

    pid_to_smid = {}
    for event in source_events:
        if event.get('ph') == 'X':
            smid = _physical_sm_id(event)
            if smid is not None:
                pid_to_smid[event.get('pid')] = smid
    output_ids_with_pid = set()
    for event in output_events:
        if event.get('ph') != 'X':
            continue
        smid = _physical_sm_id(event)
        if smid is None:
            smid = pid_to_smid.get(event.get('pid'))
        if smid is not None:
            output_ids_with_pid.add(smid)
    if set(output_ids_with_pid) != set(displayed_ids):
        fails.append(('semantic_viewer_display_sm_events_mismatch',
                      f'output event SMs={sorted(output_ids_with_pid)} metadata={displayed_ids}'))

    phase_descriptors = [item for item in descriptors if item.get('layer') == 'phase']
    source_phase_events = [event for event in source_events
                           if event.get('ph') == 'X' and _semantic_layer(event) == 'phase']
    output_phase_events = [event for event in output_events
                           if event.get('ph') == 'X' and _semantic_layer(event) == 'phase']
    if not phase_descriptors or not source_phase_events:
        fails.append(('semantic_viewer_phase_evidence_missing',
                      '代表性 SM viewer 必须有 source phase evidence 才能声明 phase 全隐藏'))
    if any(item.get('policy') != 'hidden' for item in phase_descriptors):
        fails.append(('semantic_viewer_phase_not_fully_hidden',
                      '代表性 SM viewer 只有在 ABI 中所有 phase 都是 hidden 后才允许'))
    phase_descriptor_keys = {item.get('identity') for item in phase_descriptors}
    unmapped_phase_events = [event for event in source_phase_events
                             if _semantic_event_identity(event) not in phase_descriptor_keys]
    if unmapped_phase_events:
        fails.append(('semantic_viewer_phase_abi_coverage',
                      f'{len(unmapped_phase_events)} source phase events are absent from hidden ABI descriptors'))
    if output_phase_events:
        fails.append(('semantic_viewer_phase_leak',
                      f'代表性 SM viewer 输出仍包含 {len(output_phase_events)} 个 phase event'))
    notes.append(f'representative SM halving: source={len(source_ids)} displayed={len(displayed_ids)} '
                 f'rounds={len(rounds)} source_ids_field={source_ids_field}')
    return fails, notes, True


def _descriptor_key(descriptor):
    return descriptor.get('identity') if isinstance(descriptor, dict) else None


def _check_phase_parent_abi(source_events, descriptors):
    """Enforce the task/phase ABI boundary before a phase can be hidden."""
    fails = []
    tasks_by_id = {}
    for descriptor in descriptors:
        if descriptor.get('layer') != 'task':
            continue
        identity = descriptor.get('identity')
        if isinstance(identity, tuple) and identity[0] in ('task_id', 'taskId', 'id'):
            tasks_by_id[str(identity[1])] = descriptor

    phase_records = [item for item in descriptors if item.get('layer') == 'phase']
    parent_ids_by_phase = {}
    shared_phase_keys = set()
    for descriptor in phase_records:
        raw = descriptor.get('raw') or {}
        parent_id = raw.get('parent_task_id', raw.get('parent_id'))
        identity = descriptor.get('identity')
        shared_parent_ids = raw.get('parent_task_ids')
        is_shared = raw.get('shared_across_parent_tasks') is True
        if isinstance(parent_id, int) and not isinstance(parent_id, bool):
            parent_ids = [parent_id]
        elif (is_shared and isinstance(shared_parent_ids, list)
              and len(shared_parent_ids) >= 2
              and all(isinstance(item, int) and not isinstance(item, bool)
                      for item in shared_parent_ids)
              and len(set(shared_parent_ids)) == len(shared_parent_ids)):
            parent_ids = list(shared_parent_ids)
            shared_phase_keys.add(identity)
        else:
            fails.append(('semantic_phase_parent_invalid',
                          f'ABI phase {identity} must declare one integer parent_task_id, or '
                          'shared_across_parent_tasks=true with at least two unique integer parent_task_ids'))
            continue
        parent_ids_by_phase[identity] = parent_ids
        for candidate_id in parent_ids:
            parent = tasks_by_id.get(str(candidate_id))
            parent_layer = (parent or {}).get('layer')
            if parent is None or parent_layer != 'task':
                fails.append(('semantic_phase_parent_abi_invalid',
                              f'ABI phase {identity} parent task id={candidate_id} '
                              f'must reference a layer=task ABI parent'))

    phase_keys = {_descriptor_key(item) for item in phase_records}
    missing_keys = set()
    invalid_parent_keys = set()
    invalid_same_lane_keys = set()
    task_intervals = defaultdict(list)
    for event in source_events:
        if isinstance(event, dict) and event.get('ph') == 'X' and _semantic_layer(event) == 'task':
            args = event.get('args') or {}
            task_id = args.get('task_id')
            if isinstance(task_id, int) and not isinstance(task_id, bool):
                task_intervals[(event.get('pid'), event.get('tid'), task_id)].append(event)
    for event in source_events:
        if not isinstance(event, dict) or event.get('ph') != 'X' or _semantic_layer(event) != 'phase':
            continue
        key = _semantic_event_identity(event)
        if key not in phase_keys:
            missing_keys.add(key)
            continue
        args = event.get('args') or {}
        parent_id = args.get('parent_task_id')
        start, end = _display_interval_bounds(event)
        allowed_parent_ids = parent_ids_by_phase.get(key, [])
        if key in shared_phase_keys:
            containing = []
            for candidate_id in allowed_parent_ids:
                containing.extend(
                    task for task in task_intervals.get(
                        (event.get('pid'), event.get('tid'), candidate_id), [])
                    if _display_interval_bounds(task)[0] <= start
                    and _display_interval_bounds(task)[1] >= end
                )
            if not containing:
                invalid_same_lane_keys.add((key, tuple(allowed_parent_ids)))
            continue
        if not isinstance(parent_id, int) or isinstance(parent_id, bool):
            invalid_parent_keys.add(key)
            continue
        if parent_id not in allowed_parent_ids:
            invalid_same_lane_keys.add((key, parent_id))
            continue
        parent = tasks_by_id.get(str(parent_id))
        if parent is None or parent.get('layer') != 'task':
            invalid_same_lane_keys.add((key, parent_id))
            continue
        containing = [task for task in task_intervals.get(
            (event.get('pid'), event.get('tid'), parent_id), [])
            if _display_interval_bounds(task)[0] <= start
            and _display_interval_bounds(task)[1] >= end]
        if len(containing) != 1:
            invalid_same_lane_keys.add((key, parent_id))
    if missing_keys:
        fails.append(('semantic_phase_abi_missing',
                      f'{len(missing_keys)} distinct source phase identities are absent from ABI; '
                      f'samples={sorted(missing_keys, key=str)[:8]}'))
    if invalid_parent_keys:
        fails.append(('semantic_phase_parent_invalid',
                      f'{len(invalid_parent_keys)} distinct source phase identities lack integer parent_task_id; '
                      f'samples={sorted(invalid_parent_keys, key=str)[:8]}'))
    if invalid_same_lane_keys:
        fails.append(('semantic_phase_parent_abi_invalid',
                      f'{len(invalid_same_lane_keys)} source phase identities lack a valid same-lane '
                      f'containing layer=task parent; samples={sorted(invalid_same_lane_keys, key=str)[:8]}'))
    return fails


def _abi_task_ids(abi_path):
    if not abi_path:
        return []
    abi, error = _load_json_file(Path(abi_path).expanduser().resolve())
    if error:
        return []
    return sorted({int(item.get('id', item.get('task_id')))
                   for item in _semantic_abi_tasks(abi)
                   if isinstance(item, dict)
                   and item.get('layer') == 'task'
                   and isinstance(item.get('id', item.get('task_id')), int)
                   and not isinstance(item.get('id', item.get('task_id')), bool)})


def _source_layout_for_hidden_phases(compact_source_path, semantic_viewer,
                                     abi_task_ids=None):
    """Validate full task/phase topology on compact-exact provenance."""
    fails, notes = [], []
    if compact_source_path is None:
        source_ref = (semantic_viewer.get('source') or {}).get('path') \
            if isinstance(semantic_viewer, dict) else None
        compact_source_path = source_ref
    if not compact_source_path:
        return [('semantic_viewer_source_layout_missing',
                 'hide-all-phases viewer 缺少 compact-exact source 路径')], notes
    source_path = Path(compact_source_path).expanduser().resolve()
    if not source_path.is_file():
        return [('semantic_viewer_source_layout_missing',
                 f'compact-exact source 不存在: {source_path}')], notes
    try:
        source_trace = load_trace(source_path)
    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError) as error:
        return [('semantic_viewer_source_layout_parse', str(error))], notes
    source_events = source_trace.get('traceEvents', []) if isinstance(source_trace, dict) else []
    source_spans = [event for event in source_events if isinstance(event, dict) and event.get('ph') == 'X']
    source_layout_trace, has_aux, aux_pids = _layout_trace_without_auxiliary_tasks(source_trace, source_spans)
    contract = LAYOUT_CHECKER.contract_from_trace(source_layout_trace)
    if contract is None:
        fails.append(('semantic_viewer_source_layout_missing',
                      'compact-exact source 缺少 perfetto_layout_contract'))
    else:
        try:
            source_contract = dict(contract)
            if abi_task_ids:
                source_contract['allowed_task_ids'] = list(abi_task_ids)
            source_layout_fails = LAYOUT_CHECKER.verify(source_layout_trace, source_contract)
        except SystemExit as error:
            fails.append(('semantic_viewer_source_layout_invalid', str(error)))
        else:
            # The source contract is used for topology/geometry, while the
            # semantic ABI decides which explicit phase children exist.  Do
            # not turn a generic "every task must have a phase" contract
            # default into a false failure for dispatch/load/scatter tasks.
            source_layout_fails = [message for message in source_layout_fails
                                   if not message.startswith(('V4 [task_phase_missing]',
                                                              'V4 [task_mma_phase_missing]'))]
            for message in source_layout_fails:
                fails.append(('semantic_viewer_source_layout', message))
            notes.append(f'compact-exact source layout failures = {len(source_layout_fails)}')
    if not source_spans:
        fails.append(('semantic_viewer_source_no_spans', 'compact-exact source has no ph=X spans'))
    return fails, notes


def _semantic_cta_scope(event):
    args = event.get('args') or {}
    scope = str(args.get('scope', args.get('parent_scope', ''))).lower()
    if scope in ('global_trace', 'global'):
        return False
    if scope in ('rank_cta', 'rank+cta', 'rank_cta_lane', 'observer_lane'):
        return True
    return (isinstance(args.get('rank'), int) and not isinstance(args.get('rank'), bool)
            and isinstance(args.get('block'), int) and not isinstance(args.get('block'), bool))


def _check_semantic_viewer(trace, trace_path, compact_source_path=None, abi_path=None):
    """Validate semantic-viewer/v2 using ABI visibility and count ledgers."""
    fails, notes = [], []
    other = trace.get('otherData') or {}
    viewer = other.get('semantic_viewer')
    trace_bytes = Path(trace_path).stat().st_size
    if not isinstance(viewer, dict):
        compact = other.get('compact_exact')
        if not isinstance(compact, dict) or compact.get('schema') != COMPACT_EXACT_SCHEMA:
            fails.append(('trace_compact_exact_missing', '直接交付必须包含 compact_exact metadata'))
        if trace_bytes > TRACE_SIZE_THRESHOLD_BYTES:
            fails.append(('compact_exact_size_exceeded',
                          f'fine compact-exact gzip={trace_bytes} bytes > '
                          f'{TRACE_SIZE_THRESHOLD_BYTES}; {TRACE_SIZE_REMEDY}'))
        notes.append(f'fine compact-exact size = {trace_bytes} bytes; threshold = {TRACE_SIZE_THRESHOLD_BYTES}')
        return fails, notes, False
    if trace_bytes > TRACE_SIZE_THRESHOLD_BYTES:
        fails.append(('semantic_viewer_size_exceeded',
                      f'semantic viewer gzip={trace_bytes} bytes > '
                      f'{TRACE_SIZE_THRESHOLD_BYTES}; {TRACE_SIZE_REMEDY}'))
    if Path(trace_path).suffix.lower() != '.gz':
        fails.append(('semantic_viewer_gzip_required', 'semantic viewer 交付文件必须是 gzip'))
    if viewer.get('schema') != SEMANTIC_VIEWER_SCHEMA:
        fails.append(('semantic_viewer_schema',
                      f'semantic_viewer.schema={viewer.get("schema")!r} != {SEMANTIC_VIEWER_SCHEMA!r}'))
    compact = other.get('compact_exact')
    if not isinstance(compact, dict) or compact.get('schema') != COMPACT_EXACT_SCHEMA:
        fails.append(('semantic_viewer_compact_exact_missing',
                      'semantic viewer 必须保留 compact_exact provenance'))
    visibility = viewer.get('visibility_contract')
    if not isinstance(visibility, dict):
        fails.append(('semantic_viewer_visibility_contract_missing', '缺少 visibility_contract'))
        visibility = {}
    sampling = viewer.get('sampling_policy')
    if sampling != 'none' and not (isinstance(sampling, dict) and sampling.get('mode') == 'none'):
        fails.append(('semantic_viewer_sampling', 'sampling_policy 必须为 none'))
    if viewer.get('no_sampling') is False or viewer.get('no_stride') is False:
        fails.append(('semantic_viewer_sampling', '禁止 no_sampling/no_stride=false'))
    if viewer.get('cross_cta_aggregate') is True or viewer.get('cross_cta_child_merge') is True:
        fails.append(('semantic_viewer_cross_cta_aggregate', '禁止跨 CTA 聚合真实事件'))

    abi = None
    abi_file = None
    source_hint = viewer.get('source') if isinstance(viewer.get('source'), dict) else {}
    candidate = (abi_path or visibility.get('abi_path') or
                 source_hint.get('abi_path'))
    if candidate:
        abi_file = Path(candidate).expanduser()
        if not abi_file.is_absolute():
            abi_file = (Path(trace_path).parent / abi_file).resolve()
        abi, error = _load_json_file(abi_file)
        if error:
            fails.append(('semantic_viewer_abi_missing', error))
        else:
            expected_sha = visibility.get('abi_sha256') or source_hint.get('abi_sha256')
            actual_sha = _file_sha256(abi_file)
            if expected_sha is not None and expected_sha != actual_sha:
                fails.append(('semantic_viewer_abi_digest', 'ABI sha256 与文件不一致'))
            notes.append(f'ABI = {abi_file} ({actual_sha})')
    elif isinstance(visibility.get('tasks'), list):
        abi = {'tasks': visibility['tasks']}
        notes.append('ABI = embedded visibility_contract.tasks')
    else:
        fails.append(('semantic_viewer_abi_missing',
                      '必须通过 --abi、visibility_contract.abi_path 或嵌入 tasks 提供 ABI'))
    descriptors = _semantic_descriptors(abi, fails) if abi is not None else []
    seen_descriptors = set()
    for descriptor in descriptors:
        key = (descriptor['layer'], descriptor['identity'], descriptor.get('semantic_kind'))
        if key in seen_descriptors:
            fails.append(('semantic_viewer_abi_task_duplicate', f'ABI duplicate descriptor={key!r}'))
        seen_descriptors.add(key)

    source = viewer.get('source') or viewer.get('source_provenance')
    if not isinstance(source, dict):
        fails.append(('semantic_viewer_source_provenance', '缺少 source provenance'))
        source = {}
    source_file = Path(compact_source_path).expanduser().resolve() if compact_source_path else None
    source_ref = (source.get('path') or source.get('compact_path') or
                  source.get('compact_exact_path') or source.get('file'))
    if source_file is None and source_ref:
        source_file = Path(source_ref).expanduser()
        if not source_file.is_absolute():
            source_file = (Path(trace_path).parent / source_file).resolve()
    source_trace = None
    source_bytes = source.get('gzip_bytes', source.get('compact_bytes', source.get(
        'compact_exact_bytes', source.get('bytes', viewer.get('source_bytes')))))
    source_sha = source.get('sha256', source.get('compact_sha256', source.get(
        'compact_exact_sha256', viewer.get('source_sha256'))))
    if source_file is None:
        fails.append(('semantic_viewer_source_missing', '必须提供 --compact-exact 或 source.path'))
    elif not source_file.is_file():
        fails.append(('semantic_viewer_source_missing', f'compact-exact source 不存在: {source_file}'))
    else:
        if source_file.suffix.lower() != '.gz':
            fails.append(('semantic_viewer_source_gzip_required', 'source compact-exact 必须是 gzip'))
        actual_bytes, actual_sha = source_file.stat().st_size, _file_sha256(source_file)
        if source_bytes != actual_bytes:
            fails.append(('semantic_viewer_source_size_mismatch',
                          f'metadata source bytes={source_bytes} != file bytes={actual_bytes}'))
        if source_sha != actual_sha:
            fails.append(('semantic_viewer_source_digest_mismatch', 'metadata source sha256 与文件不一致'))
        try:
            source_trace = load_trace(source_file)
        except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError) as error:
            fails.append(('semantic_viewer_source_parse', f'compact-exact source 无法解析: {error}'))
        else:
            if not isinstance(source_trace, dict) or not isinstance(source_trace.get('traceEvents'), list):
                fails.append(('semantic_viewer_source_schema', 'source 缺少 traceEvents'))
            # The size threshold governs automatic fallback only.  An explicit
            # user-requested task-only/hide-all-phases viewer is valid even when
            # its exact source already fits under 10 MiB.
            if (viewer.get('automatic_size_fallback') is True
                    and isinstance(source_bytes, int)
                    and source_bytes <= TRACE_SIZE_THRESHOLD_BYTES):
                fails.append(('semantic_viewer_source_size',
                              'automatic fallback requires source compact-exact gzip > 10 MiB'))

    decoded_provenance = source.get('decoded_events')
    if not isinstance(decoded_provenance, list) or not decoded_provenance:
        fails.append(('semantic_viewer_decoded_provenance_missing',
                      'source.decoded_events 必须包含 decoded event provenance'))
    else:
        for index, item in enumerate(decoded_provenance):
            if (not isinstance(item, dict) or not isinstance(item.get('sha256'), str)
                    or len(item.get('sha256', '')) != 64
                    or not isinstance(item.get('bytes'), int) or item.get('bytes') <= 0):
                fails.append(('semantic_viewer_decoded_provenance_invalid',
                              f'source.decoded_events[{index}] 缺少 bytes/sha256'))

    source_counts = viewer.get('source_event_counts_by_task')
    detailed_counts = viewer.get('detailed_event_counts_by_task')
    emitted_counts = viewer.get('emitted_event_counts_by_task')
    hidden_counts = viewer.get('hidden_event_counts_by_task')
    derived_counts = viewer.get('derived_event_counts_by_task')
    for key, value in (('source_event_counts_by_task', source_counts),
                       ('detailed_event_counts_by_task', detailed_counts),
                       ('emitted_event_counts_by_task', emitted_counts),
                       ('hidden_event_counts_by_task', hidden_counts),
                       ('derived_event_counts_by_task', derived_counts)):
        if not isinstance(value, dict):
            fails.append(('semantic_viewer_count_metadata', f'{key} 必须是对象'))

    output_events = trace.get('traceEvents', [])
    source_events = source_trace.get('traceEvents', []) if isinstance(source_trace, dict) else []
    phase_abi_fails = _check_phase_parent_abi(source_events, descriptors)
    fails.extend(phase_abi_fails)
    sm_policy_fails, sm_policy_notes, has_sm_filter = _check_representative_sm_policy(
        viewer, source, source_events, output_events, descriptors)
    fails.extend(sm_policy_fails)
    notes.extend(sm_policy_notes)
    if viewer.get('mode') in ('hide_all_phases', 'hide_all_phase_layers'):
        leaked_phases = [event for event in output_events
                         if isinstance(event, dict) and event.get('ph') == 'X'
                         and _semantic_layer(event) == 'phase']
        if leaked_phases:
            fails.append(('semantic_viewer_phase_leak',
                          f'hide-all-phases viewer contains {len(leaked_phases)} phase events'))
    if viewer.get('mode') == 'hide_gemm_child_phases':
        display_filter = viewer.get('sm_display_filter')
        semantic_source_events = source_events
        if (isinstance(display_filter, dict)
                and (display_filter.get('stride', 1) > 1
                     or display_filter.get('policy') in _SM_HALVING_POLICIES)):
            selected_smids = display_filter.get('displayed_physical_smids')
            if (not isinstance(selected_smids, list) or not selected_smids
                    or any(not isinstance(smid, int) or isinstance(smid, bool)
                           for smid in selected_smids)):
                fails.append(('semantic_viewer_sm_filter_metadata',
                              'displayed_physical_smids must be a non-empty integer list'))
                selected_smids = []
            selected_set = set(selected_smids)
            pid_to_smid = {}
            for event in source_events:
                if not isinstance(event, dict) or event.get('ph') != 'X':
                    continue
                args = event.get('args') if isinstance(event.get('args'), dict) else {}
                smid = args.get('physical_smid')
                if isinstance(smid, int) and not isinstance(smid, bool):
                    pid_to_smid[event.get('pid')] = smid

            def keep_selected_sm(event):
                if not isinstance(event, dict) or event.get('ph') != 'X':
                    return True
                args = event.get('args') if isinstance(event.get('args'), dict) else {}
                smid = args.get('physical_smid')
                if not isinstance(smid, int) or isinstance(smid, bool):
                    smid = pid_to_smid.get(event.get('pid'))
                return smid is None or smid in selected_set

            semantic_source_events = [event for event in source_events if keep_selected_sm(event)]
            bad_output_smids = []
            for event in output_events:
                if not isinstance(event, dict) or event.get('ph') != 'X':
                    continue
                args = event.get('args') if isinstance(event.get('args'), dict) else {}
                smid = args.get('physical_smid')
                if not isinstance(smid, int) or isinstance(smid, bool):
                    smid = pid_to_smid.get(event.get('pid'))
                if smid is not None and smid not in selected_set:
                    bad_output_smids.append(smid)
                    if len(bad_output_smids) >= 4:
                        break
            if bad_output_smids:
                fails.append(('semantic_viewer_sm_filter_leak',
                              f'output contains non-selected SMs: {bad_output_smids}'))
            notes.append('semantic viewer SM display filter: '
                         f'{len(selected_set)} selected from '
                         f'{display_filter.get("source_sm_count")} source SMs')

        hidden_ids_raw = viewer.get('hidden_gemm_phase_ids')
        if (not isinstance(hidden_ids_raw, list)
                or any(not isinstance(task_id, int) or isinstance(task_id, bool)
                       for task_id in hidden_ids_raw)):
            fails.append(('semantic_viewer_hidden_phase_metadata',
                          'hidden_gemm_phase_ids must be an integer list'))
            hidden_ids = set()
        else:
            hidden_ids = set(hidden_ids_raw)
        described_task_ids = {
            str(token) for descriptor in descriptors
            for field, token in [descriptor['identity']]
            if field in ('task_id', 'taskId')
        }
        hidden_id_tokens = {str(task_id) for task_id in hidden_ids}
        unknown_hidden_ids = sorted(hidden_id_tokens - described_task_ids)
        if unknown_hidden_ids:
            fails.append(('semantic_viewer_hidden_phase_unknown',
                          f'hidden task ids are absent from ABI: {unknown_hidden_ids}'))

        for descriptor in descriptors:
            source_actual = [
                event for event in semantic_source_events
                if isinstance(event, dict) and event.get('ph') == 'X'
                and _semantic_descriptor_matches(event, descriptor)
            ]
            emitted_actual = [
                event for event in output_events
                if isinstance(event, dict) and event.get('ph') == 'X'
                and _semantic_descriptor_matches(event, descriptor)
            ]
            source_count = _semantic_count(source_counts, descriptor)
            detailed_count = _semantic_count(detailed_counts, descriptor)
            emitted_count = _semantic_count(emitted_counts, descriptor)
            hidden_count = _semantic_count(hidden_counts, descriptor)
            derived_count = _semantic_count(derived_counts, descriptor)
            field, token = descriptor['identity']
            label = f'{descriptor["layer"]}:{field}={token}'
            if (source_count is None or detailed_count is None or emitted_count is None
                    or hidden_count is None or derived_count is None):
                fails.append(('semantic_viewer_count_metadata',
                              f'{label} lacks source/detailed/emitted/hidden/derived count'))
                continue
            if source_count != len(source_actual) or detailed_count != len(source_actual):
                fails.append(('semantic_viewer_source_count_mismatch',
                              f'{label} metadata source/detailed={source_count}/{detailed_count}, '
                              f'selected compact actual={len(source_actual)}'))
            if emitted_count != len(emitted_actual):
                fails.append(('semantic_viewer_emitted_count_mismatch',
                              f'{label} metadata emitted={emitted_count}, actual={len(emitted_actual)}'))
            intentionally_hidden = (
                descriptor['layer'] == 'phase'
                and field in ('task_id', 'taskId')
                and str(token) in hidden_id_tokens
            )
            if intentionally_hidden:
                if not (hidden_count == len(source_actual)
                        and emitted_count == 0 and derived_count == 0
                        and not emitted_actual):
                    fails.append(('semantic_viewer_hidden_gemm_phase_ledger',
                                  f'{label} must satisfy source=detailed=hidden and emitted=derived=0'))
            else:
                if not (hidden_count == 0 and derived_count == 0
                        and source_count == detailed_count == emitted_count):
                    fails.append(('semantic_viewer_preserved_event_ledger',
                                  f'{label} must satisfy source=detailed=emitted and hidden=derived=0'))
                if _geometry_digest(source_actual) != _geometry_digest(emitted_actual):
                    fails.append(('semantic_viewer_preserved_geometry_digest',
                                  f'{label} geometry changed'))

        counters = [event for event in output_events if event.get('ph') == 'C']
        counter_meta = viewer.get('hardware_counters')
        if isinstance(counter_meta, dict):
            if (counter_meta.get('event_count') is not None
                    and counter_meta.get('event_count') != len(counters)):
                fails.append(('semantic_viewer_counter_count',
                              'viewer HW counter event count changed'))
            expected_digest = counter_meta.get('geometry_sha256') or counter_meta.get('event_sha256')
            if expected_digest and expected_digest != _event_digest(counters):
                fails.append(('semantic_viewer_counter_digest',
                              'viewer HW counter geometry digest changed'))
        source_counters = [
            event for event in source_events
            if isinstance(event, dict) and event.get('ph') == 'C'
        ]
        if source_trace and _event_digest(source_counters) != _event_digest(counters):
            fails.append(('semantic_viewer_counter_changed',
                          'viewer and source HW counter geometry differ'))
        notes.append(f'semantic viewer size = {trace_bytes} bytes; '
                     f'ABI descriptors = {len(descriptors)}; HW counters = {len(counters)}')
        return fails, notes, True

    for descriptor in descriptors:
        source_actual = [event for event in source_events if isinstance(event, dict)
                         and event.get('ph') == 'X' and _semantic_descriptor_matches(event, descriptor)]
        emitted_actual = [event for event in output_events if isinstance(event, dict)
                          and event.get('ph') == 'X' and _semantic_descriptor_matches(event, descriptor)]
        source_count = _semantic_count(source_counts, descriptor)
        detailed_count = _semantic_count(detailed_counts, descriptor)
        emitted_count = _semantic_count(emitted_counts, descriptor)
        hidden_count = _semantic_count(hidden_counts, descriptor)
        derived_count = _semantic_count(derived_counts, descriptor)
        field, token = descriptor['identity']
        label = f'{descriptor["layer"]}:{field}={token}'
        if (source_count is None or detailed_count is None or emitted_count is None
                or hidden_count is None or derived_count is None):
            fails.append(('semantic_viewer_count_metadata',
                          f'{label} 缺少 source/detailed/emitted/hidden/derived count'))
            continue
        if detailed_count != len(source_actual):
            fails.append(('semantic_viewer_detailed_count_mismatch',
                          f'{label} metadata detailed={detailed_count}, compact actual={len(source_actual)}'))
        if emitted_count != len(emitted_actual):
            fails.append(('semantic_viewer_emitted_count_mismatch',
                          f'{label} metadata emitted={emitted_count}, actual={len(emitted_actual)}'))
        policy = descriptor.get('policy')
        if (viewer.get('mode') in ('hide_all_phases', 'hide_all_phase_layers')
                and descriptor.get('layer') == 'phase'):
            if not (source_count == detailed_count == hidden_count
                    and emitted_count == 0 and derived_count == 0):
                fails.append(('semantic_viewer_hidden_phase_ledger',
                              f'{label} hide-all ledger must satisfy '
                              f'source=detailed=hidden and emitted=derived=0'))
            continue
        if policy not in ('exact', 'summary', 'hidden', 'global_parent'):
            fails.append(('semantic_viewer_policy_missing', f'{label} ABI viewer_policy 无效: {policy!r}'))
            continue
        if policy == 'exact':
            if not (source_count == detailed_count == emitted_count
                    and hidden_count == 0 and derived_count == 0):
                fails.append(('semantic_viewer_exact_ledger',
                              f'{label} exact ledger 不满足 source=detailed=emitted, hidden=derived=0'))
            if _geometry_digest(source_actual) != _geometry_digest(emitted_actual):
                fails.append(('semantic_viewer_phase_geometry_digest', f'{label} exact geometry digest 不一致'))
            if any(not _semantic_cta_scope(event) for event in emitted_actual):
                fails.append(('semantic_viewer_cta_scope_missing', f'{label} exact event 缺少 CTA scope'))
        elif policy == 'summary':
            if not (source_count == detailed_count == hidden_count
                    and emitted_count == derived_count):
                fails.append(('semantic_viewer_summary_ledger',
                              f'{label} summary ledger 不满足 source=detailed=hidden, emitted=derived'))
            if (not _semantic_summary_allowed(visibility, descriptor.get('semantic_kind'))
                    and descriptor.get('policy') != 'summary'):
                fails.append(('semantic_viewer_phase_hidden_forbidden',
                              f'{label} summary 未被 visibility_contract 明确允许'))
            if any(not _semantic_local_summary(event) or not _semantic_cta_scope(event)
                   for event in emitted_actual):
                fails.append(('semantic_viewer_phase_summary_scope',
                              f'{label} summary 必须是 CTA/observer-lane scope'))
        elif policy == 'hidden':
            if not (source_count == hidden_count and detailed_count == 0
                    and emitted_count == 0 and derived_count == 0):
                fails.append(('semantic_viewer_hidden_ledger',
                              f'{label} hidden ledger 不满足 source=hidden,detailed=emitted=derived=0'))
        elif policy == 'global_parent':
            if not (source_count == hidden_count and source_count > 0
                    and detailed_count == emitted_count == derived_count == 1):
                fails.append(('semantic_viewer_global_parent_ledger',
                              f'{label} global_parent ledger 不满足 source=hidden>0,detailed=emitted=derived=1'))
            parents = [event for event in emitted_actual
                       if (event.get('args') or {}).get('is_derived') is True
                       and is_global_task_envelope(event)]
            if len(parents) != 1:
                fails.append(('semantic_viewer_global_parent_count',
                              f'{label} 必须恰好一个 derived global parent，实际 {len(parents)}'))
            if _geometry_digest(source_actual) != _geometry_digest(parents):
                fails.append(('semantic_viewer_global_parent_digest',
                              f'{label} compact source 与 emitted parent geometry 不一致'))
            parent_metadata = viewer.get('global_parents', visibility.get('global_parents'))
            parent_records = _semantic_parent_metadata(parent_metadata, descriptor)
            if not parent_records:
                fails.append(('semantic_viewer_global_parent_metadata', f'{label} 缺少 global_parents metadata'))
            else:
                parent_digest = parent_records[0].get('geometry_sha256') \
                    if isinstance(parent_records[0], dict) else None
                if parent_digest and parent_digest != _geometry_digest(parents):
                    fails.append(('semantic_viewer_global_parent_digest', f'{label} global parent geometry digest 不一致'))
    counters = [event for event in output_events if event.get('ph') == 'C']
    counter_meta = viewer.get('hardware_counters')
    if isinstance(counter_meta, dict):
        if counter_meta.get('event_count') is not None and counter_meta.get('event_count') != len(counters):
            fails.append(('semantic_viewer_counter_count', 'viewer HW counter event count 改变'))
        expected_digest = counter_meta.get('geometry_sha256') or counter_meta.get('event_sha256')
        if expected_digest and expected_digest != _event_digest(counters):
            fails.append(('semantic_viewer_counter_digest', 'viewer HW counter geometry digest 改变'))
    source_counters = [event for event in source_events if isinstance(event, dict) and event.get('ph') == 'C']
    if source_trace and _event_digest(source_counters) != _event_digest(counters):
        fails.append(('semantic_viewer_counter_changed', 'viewer 与 source 的 HW counter 几何不一致'))
    notes.append(f'semantic viewer size = {trace_bytes} bytes; ABI descriptors = {len(descriptors)}; HW counters = {len(counters)}')
    return fails, notes, True


def _load_json_file(path):
    """读取门禁证据；调用方把解析错误转换为具体 FAIL，而不是 traceback。"""
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle), None
    except FileNotFoundError:
        return None, f'文件不存在: {path}'
    except (OSError, json.JSONDecodeError) as error:
        return None, f'无法读取 JSON {path}: {error}'


def _metric_key(value):
    return ' '.join(str(value).strip().lower().replace('_', ' ').split())


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _path_candidates(explicit, trace_path, names):
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    parent = trace_path.parent
    for name in names:
        candidates.extend((parent / name, parent / 'hw' / name))
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def _check_hw_capture_setup(manifest):
    fails = []
    notes = []
    metrics_set = manifest.get('metrics_set')
    preflight = manifest.get('metrics_config_preflight')
    if not isinstance(metrics_set, str) or not metrics_set.strip():
        fails.append(('hw_metrics_set_missing', 'capture_manifest 缺少实际使用的 metrics_set'))
    if isinstance(preflight, dict) and preflight.get('hard_failures'):
        fails.append(('hw_metrics_config_preflight_failed',
                      f'hard_failures={preflight.get("hard_failures")!r}'))
    metrics_config = manifest.get('metrics_config')
    if isinstance(metrics_config, dict):
        notes.append('metrics config provenance: '
                     f'path={metrics_config.get("path")!r}, sha256={metrics_config.get("sha256")!r}')
    notes.append(f'metrics set = {metrics_set!r}')
    return fails, notes


def _check_capture_manifest(manifest, manifest_path):
    fails = []
    notes = []

    if not isinstance(manifest, dict):
        return [('hw_manifest_schema', 'capture_manifest 顶层必须是 JSON object')], notes

    capture_range = manifest.get('capture_range')
    capture_range_end = manifest.get('capture_range_end')
    notes.append(f'capture range = {capture_range}/{capture_range_end}')
    if capture_range != HW_CAPTURE_RANGE:
        fails.append(('hw_capture_range_mode',
                      f'capture_manifest.capture_range={capture_range!r}，必须为 {HW_CAPTURE_RANGE!r}'))
    if capture_range_end != HW_CAPTURE_RANGE_END:
        fails.append(('hw_capture_range_end',
                      f'capture_manifest.capture_range_end={capture_range_end!r}，必须为 {HW_CAPTURE_RANGE_END!r}'))

    frequency = manifest.get('frequency_hz')
    notes.append(f'GPU metrics frequency = {frequency!r} Hz')
    if not isinstance(frequency, (int, float)) or isinstance(frequency, bool) or frequency <= 0:
        fails.append(('hw_frequency',
                      f'capture_manifest.frequency_hz={frequency!r}，必须为正数'))
    elif frequency != HW_PREFERRED_FREQUENCY_HZ:
        fails.append(('hw_frequency',
                      f'capture_manifest.frequency_hz={frequency!r}; delivery requires exactly '
                      f'{HW_PREFERRED_FREQUENCY_HZ} Hz'))

    metrics_set = manifest.get('metrics_set')
    metrics_source = manifest.get('metrics_set_source')
    metric_count = manifest.get('metrics_config_metric_count')
    notes.append(f'metrics set = {metrics_set!r}, source = {metrics_source!r}, counters = {metric_count!r}')
    if not isinstance(metrics_set, str) or not metrics_set.strip():
        fails.append(('hw_metrics_set_missing', 'capture_manifest.metrics_set 缺失或为空'))
    if metric_count is not None and (not isinstance(metric_count, int) or metric_count <= 0):
        fails.append(('hw_metrics_config_count',
                      f'capture_manifest.metrics_config_metric_count={metric_count!r}，若记录则必须为正整数'))
    config_fails, config_notes = _check_hw_capture_setup(manifest)
    fails.extend(config_fails)
    notes.extend(config_notes)

    exit_codes = manifest.get('exit_codes')
    if not isinstance(exit_codes, dict):
        fails.append(('hw_exit_codes_missing', 'capture_manifest 缺少 exit_codes'))
    else:
        for name in ('nsys_profile', 'nsys_export'):
            if exit_codes.get(name) != 0:
                fails.append((f'hw_{name}_exit',
                              f'exit_codes.{name}={exit_codes.get(name)!r}，必须为 0'))

    verdict = manifest.get('verdict')
    evidence = manifest.get('evidence')
    evidence = evidence if isinstance(evidence, dict) else {}
    evidence_verdict = evidence.get('verdict')
    evidence_failures = evidence.get('failures')
    notes.append(f'capture evidence verdict = {verdict!r}/{evidence_verdict!r}')
    if verdict != 'PASS' or evidence_verdict != 'PASS':
        fails.append(('hw_capture_evidence_verdict',
                      f'capture_manifest/evidence verdict 必须均为 PASS，实际 {verdict!r}/{evidence_verdict!r}'))
    if evidence_failures:
        fails.append(('hw_capture_evidence_failures',
                      f'capture_manifest.evidence.failures 非空: {evidence_failures[:3]}'))

    kernel_evidence = evidence.get('kernel')
    if not isinstance(kernel_evidence, dict) or not isinstance(kernel_evidence.get('match_rows'), int):
        fails.append(('hw_kernel_probe_missing',
                      'capture_manifest.evidence.kernel.match_rows 缺失，无法证明目标 kernel 被采到'))
    elif kernel_evidence.get('match_rows', 0) <= 0:
        fails.append(('hw_kernel_probe_empty',
                      f'capture_manifest.evidence.kernel.match_rows={kernel_evidence.get("match_rows")}'))

    payload = manifest.get('payload_command')
    payload_line = manifest.get('payload_command_line', '')
    payload_text = ' '.join(str(item) for item in payload) if isinstance(payload, list) else str(payload_line)
    if capture_range == HW_CAPTURE_RANGE and '--cuda-profiler-range' not in payload_text:
        fails.append(('hw_payload_range_missing',
                      'capture range 为 cudaProfilerApi，但 payload 未声明 --cuda-profiler-range'))
    nsys_command = manifest.get('nsys_command')
    nsys_text = ' '.join(str(item) for item in nsys_command) if isinstance(nsys_command, list) else str(
        manifest.get('nsys_command_line', ''))
    if f'--capture-range={HW_CAPTURE_RANGE}' not in nsys_text:
        fails.append(('hw_nsys_capture_range_missing',
                      'nsys command 未显式传 --capture-range=cudaProfilerApi'))
    if f'--capture-range-end={HW_CAPTURE_RANGE_END}' not in nsys_text:
        fails.append(('hw_nsys_capture_range_end_missing',
                      'nsys command 未显式传 --capture-range-end=stop'))

    # Nsight's application log is the execution-side proof that the API window
    # actually opened and closed; command-line flags alone only prove intent.
    log_candidates = [manifest_path.parent / 'capture_nsys.log']
    for artifact in manifest.get('artifacts', []):
        if isinstance(artifact, dict) and str(artifact.get('path', '')).endswith('capture_nsys.log'):
            log_candidates.append(Path(artifact['path']))
    capture_log_path = next((path for path in log_candidates if path.is_file()), None)
    if capture_log_path is None:
        fails.append(('hw_capture_api_log_missing',
                      '找不到 capture_nsys.log，无法证明 cudaProfilerStart/Stop 实际生效'))
    else:
        try:
            capture_log = capture_log_path.read_text(encoding='utf-8', errors='replace')
        except OSError as error:
            capture_log = ''
            fails.append(('hw_capture_api_log_unreadable', f'无法读取 {capture_log_path}: {error}'))
        started = 'Capture range started in the application.' in capture_log
        ended = 'Capture range ended in the application.' in capture_log
        if not started or not ended:
            fails.append(('hw_capture_api_window_not_effective',
                          f'capture log 未同时记录 application range start/stop: started={started}, ended={ended}'))
        notes.append(f'capture API log = {capture_log_path}; application start/stop = {started}/{ended}')

    # 这是采集进程结束后的 GPU 状态证据；若采集残留 foreign process，不得交付。
    gpu_check = manifest.get('gpu_check')
    if isinstance(gpu_check, dict):
        if gpu_check.get('busy') is True:
            fails.append(('hw_gpu_still_busy', 'capture_manifest.gpu_check.busy=true'))
        foreign = gpu_check.get('foreign_processes')
        if foreign:
            fails.append(('hw_foreign_processes',
                          f'采集后仍有 foreign_processes: {foreign[:3]}'))

    # 旧版 manifest 只有 nsys profile/export 退出码；优先读取更强的 harness_exit，
    # 否则读取同一 out_dir 下 weaveprof/pipeline/validate/harness/run_manifest.json 的完整 rank 证据。
    process_evidence = manifest.get('harness_exit')
    if isinstance(process_evidence, dict):
        code = process_evidence.get('code', process_evidence.get('exit_code'))
        if code != 0:
            fails.append(('hw_harness_exit', f'harness 退出码={code!r}，必须为 0'))
        if process_evidence.get('all_ranks_ok') is not True:
            fails.append(('hw_harness_ranks', 'harness_exit.all_ranks_ok 必须为 true'))
    else:
        run_manifest_candidates = [manifest_path.parent / 'harness' / 'run_manifest.json']
        out_dir = manifest.get('out_dir')
        if isinstance(out_dir, str):
            run_manifest_candidates.append(Path(out_dir) / 'harness' / 'run_manifest.json')
        run_manifest_path = next((p for p in run_manifest_candidates if p.is_file()), None)
        if run_manifest_path is None:
            fails.append(('hw_harness_exit_evidence_missing',
                          '缺少 harness_exit，且找不到 weaveprof/pipeline/validate/harness/run_manifest.json，无法证明所有 rank 正常退出'))
        else:
            run_manifest, error = _load_json_file(run_manifest_path)
            if error:
                fails.append(('hw_harness_run_manifest_invalid', error))
            elif not isinstance(run_manifest, dict):
                fails.append(('hw_harness_run_manifest_schema', 'weaveprof/pipeline/validate/harness/run_manifest.json 顶层必须是 object'))
            else:
                expected_ranks = run_manifest.get('num_processes')
                correctness = run_manifest.get('correctness')
                per_rank = correctness.get('per_rank') if isinstance(correctness, dict) else None
                rank_records = per_rank if isinstance(per_rank, list) else []
                observed_ranks = sorted({item.get('rank') for item in rank_records
                                         if isinstance(item, dict) and isinstance(item.get('rank'), int)})
                if not isinstance(expected_ranks, int) or expected_ranks < 1:
                    fails.append(('hw_harness_rank_count_missing',
                                  'run_manifest.num_processes 缺失或非法'))
                else:
                    expected_rank_set = list(range(expected_ranks))
                    if observed_ranks != expected_rank_set:
                        fails.append(('hw_harness_rank_coverage',
                                      f'run_manifest rank={observed_ranks}，期望 {expected_rank_set}'))
                if not isinstance(correctness, dict) or correctness.get('all_passed') is not True:
                    fails.append(('hw_harness_correctness',
                                  'run_manifest.correctness.all_passed 必须为 true'))
                bad_ranks = [item.get('rank') for item in rank_records
                             if item.get('passed') is not True]
                if bad_ranks:
                    fails.append(('hw_harness_rank_failure', f'失败 rank: {bad_ranks}'))
                notes.append(f'harness run_manifest = {run_manifest_path}; ranks={observed_ranks}; all_passed={correctness.get("all_passed") if isinstance(correctness, dict) else None}')

    return fails, notes


def _check_hw_metrics(metrics, manifest, metrics_path):
    fails = []
    notes = []
    if not isinstance(metrics, dict):
        return [('hw_metrics_schema', 'hw_metrics 顶层必须是 JSON object')], notes

    frequency = metrics.get('frequency_hz')
    if not isinstance(frequency, (int, float)) or isinstance(frequency, bool) or frequency <= 0:
        fails.append(('hw_metrics_frequency',
                      f'hw_metrics.frequency_hz={frequency!r}，必须为正数'))
    elif frequency != HW_PREFERRED_FREQUENCY_HZ:
        fails.append(('hw_metrics_frequency',
                      f'hw_metrics.frequency_hz={frequency!r}; delivery requires exactly '
                      f'{HW_PREFERRED_FREQUENCY_HZ} Hz'))

    metric_filter = metrics.get('metrics_filter')
    if not isinstance(metric_filter, dict):
        fails.append(('hw_metrics_filter_missing',
                      'hw_metrics.metrics_filter 缺失，无法确定本次必需指标'))
        metric_filter = {}
    keep = metric_filter.get('keep')
    required = metric_filter.get('required')
    if not isinstance(keep, list) or not keep:
        fails.append(('hw_metrics_keep_missing',
                      f'hw_metrics.metrics_filter.keep 必须是非空指标列表，实际 {keep!r}'))
        keep = keep if isinstance(keep, list) else []
    if len({_metric_key(name) for name in keep}) != len(keep):
        fails.append(('hw_metrics_keep_duplicate', 'metrics_filter.keep 含重复 counter'))
    if not isinstance(required, list) or not required:
        fails.append(('hw_required_counter_declaration',
                      f'metrics_filter.required 必须是针对本次硬件与通路的非空指标列表；observed={required!r}'))
        required = required if isinstance(required, list) else []
    required_keys = {_metric_key(name) for name in required}
    keep_keys = {_metric_key(name) for name in keep}
    for required_name in required:
        if _metric_key(required_name) not in keep_keys:
            fails.append(('hw_required_counter_not_configured',
                          f'keep 未包含本次声明的必需 counter: {required_name}'))
    values = metric_filter.get('missing_required')
    if values:
        fails.append(('hw_metrics_missing_required',
                      f'metrics_filter.missing_required 非空: {values[:3]}'))
    notes.append(f'hw metrics configured counters = {len(keep)}; required counters = {len(required)}')

    counters = metrics.get('counters')
    if not isinstance(counters, list) or not counters:
        fails.append(('hw_metrics_counters_missing', 'hw_metrics.counters 缺失或为空'))
        counters = []
    by_key = defaultdict(list)
    for counter in counters:
        if isinstance(counter, dict):
            by_key[_metric_key(counter.get('name', ''))].append(counter)
    # `keep` 是提取白名单，允许包含不同 nsys 版本使用的可选别名；交付完整性
    # 由显式 `required` 列表定义，不能把每个可选 keep 名称都当成必需指标。
    for name in required:
        matches = by_key.get(_metric_key(name), [])
        if not matches:
            fails.append(('hw_required_counter_missing', f'缺少配置 counter: {name}'))
            continue
        for counter in matches:
            samples = counter.get('samples')
            sample_count = counter.get('sample_count')
            if not isinstance(sample_count, int) or sample_count <= 0:
                fails.append(('hw_counter_no_samples',
                              f'counter {counter.get("name")!r} sample_count={sample_count!r}，必须 > 0'))
            if not isinstance(samples, list) or not samples:
                fails.append(('hw_counter_samples_missing',
                              f'counter {counter.get("name")!r} samples 为空'))
            elif isinstance(sample_count, int) and sample_count != len(samples):
                fails.append(('hw_counter_sample_count_mismatch',
                              f'counter {counter.get("name")!r}: sample_count={sample_count} != len(samples)={len(samples)}'))

    launches = metrics.get('kernel_launches')
    if not isinstance(launches, list) or not launches:
        fails.append(('hw_kernel_launch_evidence_missing',
                      'hw_metrics.kernel_launches 缺失或为空，无法证明目标 kernel launch'))
        launches = []
    kernel_needle = metrics.get('kernel_name_contains')
    if not isinstance(kernel_needle, str) or not kernel_needle.strip():
        fails.append(('hw_kernel_name_filter_missing',
                      'hw_metrics.kernel_name_contains 缺失或为空，无法把 launch 证据绑定到目标 kernel'))
    valid_launches = []
    for index, launch in enumerate(launches):
        if not isinstance(launch, dict):
            fails.append(('hw_kernel_launch_schema', f'kernel_launches[{index}] 不是 object'))
            continue
        device = launch.get('device')
        start_ns = _number(launch.get('start_ns'))
        end_ns = _number(launch.get('end_ns'))
        name = str(launch.get('name', ''))
        if not isinstance(device, int):
            fails.append(('hw_kernel_launch_device', f'kernel_launches[{index}].device 非整数'))
        if start_ns is None or end_ns is None or end_ns <= start_ns:
            fails.append(('hw_kernel_launch_window',
                          f'kernel_launches[{index}] start/end 非法: {launch.get("start_ns")!r}/{launch.get("end_ns")!r}'))
        if not name.strip():
            fails.append(('hw_kernel_launch_name_missing',
                          f'kernel_launches[{index}].name 缺失或为空'))
        if kernel_needle and kernel_needle not in name:
            fails.append(('hw_kernel_launch_name',
                          f'kernel_launches[{index}].name 不包含目标 {kernel_needle!r}'))
        if isinstance(device, int) and start_ns is not None and end_ns is not None and end_ns > start_ns:
            valid_launches.append((device, int(start_ns), int(end_ns), name))
    counts = Counter(device for device, _, _, _ in valid_launches)
    # ours_ep8 fork: when the extractor pair-merges the L1/L2 phase launches
    # into logical MoE launches, the per-device expectation is the declared
    # per-device MoE call count from the metrics file, not 1.
    per_device_expected = None
    if metrics.get('phase_pair_merged') is True:
        per_device = metrics.get('per_device_moe_calls')
        if isinstance(per_device, dict):
            per_device_expected = {int(k): int(v) for k, v in per_device.items()}
    for device, count in sorted(counts.items()):
        expected_count = (per_device_expected or {}).get(device, 1)
        if count != expected_count:
            fails.append(('hw_kernel_launch_count',
                          f'设备 {device} 目标 kernel launch {count} 次，必须恰好 {expected_count} 次'))
    expected_device = metrics.get('device')
    if not isinstance(expected_device, int):
        expected_device = manifest.get('gpu_metrics_device')
    if isinstance(expected_device, int) and expected_device not in counts:
        fails.append(('hw_kernel_launch_device_missing',
                      f'采样设备 {expected_device} 没有目标 kernel launch 证据'))
    notes.append(f'target kernel launches = {len(valid_launches)}; per-device counts = {dict(sorted(counts.items()))}')

    clip_windows = metrics.get('clip_windows_ns')
    if not isinstance(clip_windows, list):
        clip_windows = []
    for device, start_ns, end_ns, _ in valid_launches:
        covered = False
        for window in clip_windows:
            if isinstance(window, list) and len(window) == 2:
                lo = _number(window[0])
                hi = _number(window[1])
                if lo is not None and hi is not None and lo <= start_ns and hi >= end_ns:
                    covered = True
                    break
        if not covered:
            fails.append(('hw_kernel_window_not_clipped',
                          f'目标 kernel [{start_ns}, {end_ns}] 不在 hw_metrics.clip_windows_ns 内'))

    interval = metrics.get('sampling_interval_ns')
    reference_interval_ns = None
    max_allowed_gap_ns = None
    if not isinstance(interval, dict) or interval.get('available') is not True:
        fails.append(('hw_sampling_interval_missing', 'hw_metrics.sampling_interval_ns 不可用'))
    else:
        reference_interval_ns = _number(interval.get('median'))
        global_max_gap = _number(interval.get('max'))
        if reference_interval_ns is None or reference_interval_ns <= 0:
            fails.append(('hw_sampling_interval_invalid',
                          f'采样中位间隔非法: {interval.get("median")!r}'))
        else:
            max_allowed_gap_ns = 10.0 * reference_interval_ns
        if global_max_gap is None or (max_allowed_gap_ns is not None
                                      and global_max_gap > max_allowed_gap_ns):
            fails.append(('hw_sampling_gap_global',
                          f'全局最大采样间隔={interval.get("max")!r} ns，'
                          f'实测中位间隔={reference_interval_ns!r} ns'))
        else:
            notes.append(f'sampling median/max = {reference_interval_ns:.0f}/{global_max_gap:.0f} ns')

    # 对每一个 chosen launch、每一个 configured counter 重新按原始样本计算窗口证据。
    # 这一步不信任 trace merge 后的 counter，也不把窗口外样本当作覆盖。
    for name in keep:
        for counter in by_key.get(_metric_key(name), []):
            samples = counter.get('samples') if isinstance(counter, dict) else None
            if not isinstance(samples, list):
                continue
            parsed = []
            for sample in samples:
                if not isinstance(sample, (list, tuple)) or len(sample) < 2:
                    fails.append(('hw_sample_schema', f'counter {counter.get("name")!r} 有非法样本: {sample!r}'))
                    continue
                ts = _number(sample[0])
                value = _number(sample[1])
                if ts is None or value is None:
                    fails.append(('hw_sample_nonfinite', f'counter {counter.get("name")!r} 有非有限样本: {sample!r}'))
                    continue
                parsed.append((int(ts), value))
            for device, start_ns, end_ns, _ in valid_launches:
                in_window = [item for item in parsed if start_ns <= item[0] <= end_ns]
                if not in_window:
                    fails.append(('hw_counter_window_empty',
                                  f'counter {counter.get("name")!r} 在设备 {device} 的 chosen launch 窗口内无样本'))
                    continue
                gaps = [in_window[i][0] - in_window[i - 1][0] for i in range(1, len(in_window))]
                if any(gap < 0 for gap in gaps):
                    fails.append(('hw_counter_sample_order',
                                  f'counter {counter.get("name")!r} chosen launch 样本时间戳未排序'))
                max_gap = max(gaps) if gaps else 0
                if max_allowed_gap_ns is not None and max_gap > max_allowed_gap_ns:
                    fails.append(('hw_sampling_gap_window',
                          f'counter {counter.get("name")!r} 设备 {device} 窗口最大 gap={max_gap} ns，允许 <= {max_allowed_gap_ns:.0f} ns'))
                notes.append(f'counter {counter.get("name")}: window samples={len(in_window)}, max={max(value for _, value in in_window):.3f}, maxgap={max_gap} ns')

    return fails, notes


def check_hw_bundle(trace_path, contract_path, manifest_arg=None, metrics_arg=None):
    """HW bundle 的交付级硬门禁；纯 SW trace 不调用本函数。"""
    fails = []
    notes = []
    manifest_path = _path_candidates(
        manifest_arg, trace_path,
        ('capture_manifest.json',))
    metrics_path = _path_candidates(
        metrics_arg, trace_path,
        ('hw_metrics.json',))
    if manifest_path is None:
        fails.append(('hw_capture_manifest_missing',
                      'HW 交付缺少 capture_manifest.json；仅有 nsys-rep/SQLite/counter 泳道不能替代'))
    if metrics_path is None:
        fails.append(('hw_metrics_missing',
                      'HW 交付缺少 hw_metrics.json；仅有 nsys-rep/SQLite/counter 泳道不能替代'))
    if fails:
        return fails, notes

    manifest, error = _load_json_file(manifest_path)
    if error:
        fails.append(('hw_capture_manifest_invalid', error))
        return fails, notes
    metrics, error = _load_json_file(metrics_path)
    if error:
        fails.append(('hw_metrics_invalid', error))
        return fails, notes
    manifest_fails, manifest_notes = _check_capture_manifest(manifest, manifest_path)
    metrics_fails, metrics_notes = _check_hw_metrics(metrics, manifest, metrics_path)
    fails.extend(manifest_fails)
    fails.extend(metrics_fails)
    notes.append(f'capture_manifest = {manifest_path}')
    notes.append(f'hw_metrics = {metrics_path}')
    notes.extend(manifest_notes)
    notes.extend(metrics_notes)
    return fails, notes


# SPEC-2: a phase/task slice name is a human-facing label, not an internal code.
# A bare phase code like "P0"/"P12a", a truncated "<=7 char ..._i" instance tag,
# or a <=3 char phase-layer name is never an acceptable display name.
_BARE_PHASE_CODE_RE = re.compile(r'^P\d+[a-z]?$')
# Leading builder prefix on a phase name, e.g. "phase_p3_qkv_a" or "phase_qkv_a".
_PHASE_NAME_PREFIX_RE = re.compile(r'^phase_(p\d+_)?', re.IGNORECASE)


def _is_banned_display_name(name, layer):
    """SPEC-2 banned-pattern predicate; independent of the ABI."""
    if not isinstance(name, str):
        return False
    text = name.strip()
    if not text:
        return False
    if _BARE_PHASE_CODE_RE.match(text):
        return True
    if text.endswith('_i') and len(text) <= 7:
        return True
    if layer == 'phase':
        # A phase name is cryptic only when it carries fewer than three
        # alphabetic characters (bare 'p', 't0', 'q2'); real short acronyms such
        # as 'mma'/'tma' are legitimate human-readable names and must not be
        # banned for length alone. ABI membership (the 'unexpected' audit below)
        # stays the strict semantic gate.
        alpha = re.sub(r'[^A-Za-z]', '', text)
        if len(alpha) < 3:
            return True
    return False


def _derive_display_variants(descriptor):
    """Every display name a builder may legitimately emit for one ABI descriptor.

    SPEC-2 priority order is display_name > source (strip a leading '<file> ' and
    a leading 'P<idx> ') > name (strip 'phase_pN_'/'phase_' then '_'->space) >
    short_name.  A builder is free to pick any one of these, and the gate is fed
    the same ABI the builder used, so the expected set is their union while
    membership itself stays strict.
    """
    variants = set()

    def add(value):
        token = _stable_token(value)
        if token is not None:
            variants.add(token)

    add(descriptor.get('display_name'))

    source = descriptor.get('source')
    if isinstance(source, str) and source.strip():
        stripped = re.sub(r'^\S+\.\w+\s+', '', source.strip())
        stripped = re.sub(r'^P\d+[a-z]?\s+', '', stripped)
        add(stripped)
        add(source)

    name = descriptor.get('name')
    if isinstance(name, str) and name.strip():
        add(name)
        add(_PHASE_NAME_PREFIX_RE.sub('', name.strip()).replace('_', ' '))

    add(descriptor.get('short_name'))
    add(descriptor.get('role'))
    add(descriptor.get('task_name'))
    return variants


def _expected_display_names(abi_path):
    """Union of ABI-derivable display names, or empty set when ABI lacks info."""
    expected = set()
    for descriptor in _abi_display_descriptors(abi_path):
        expected |= _derive_display_variants(descriptor)
    return expected


def check_slice_display_names(spans, abi_path=None):
    """SPEC-2: phase/task slice names must be human-readable and ABI-declared.

    Banned-pattern enforcement always runs (even with no ABI).  Membership
    against the ABI-derived expected set runs only when the ABI carries display
    info, so a SW-only or ABI-less delivery is not falsely rejected.
    """
    fails = []
    notes = []
    slices = [event for event in spans if span_layer(event) in ('task', 'phase')]
    expected = _expected_display_names(abi_path)
    banned = []
    unexpected = []
    for event in slices:
        layer = span_layer(event)
        name = event.get('name')
        if _is_banned_display_name(name, layer):
            banned.append((layer, name, event.get('pid'), event.get('tid')))
            continue
        if is_global_task_envelope(event):
            # derived aggregate envelopes are not individual ABI descriptors
            continue
        if expected and _stable_token(name) not in expected:
            unexpected.append((layer, name))
    if banned:
        fails.append(('slice_display_name_banned',
                      f'{len(banned)} phase/task 切片使用了非人类可读的内部代号 '
                      f'(裸相位号 P<n>、<=7 字符的 *_i 实例标签、或 <=3 字符 phase 名)；'
                      f'samples={banned[:8]}'))
    if unexpected:
        fails.append(('slice_display_name_unexpected',
                      f'{len(unexpected)} phase/task 切片名不在 ABI 声明的展示名集合内；'
                      f'display 名必须来自 ABI display_name/source/name/short_name；'
                      f'samples={unexpected[:8]}'))
    notes.append(f'slice display names: slices={len(slices)}, '
                 f'abi_expected_names={len(expected)}, banned={len(banned)}, '
                 f'unexpected={len(unexpected)}')
    return fails, notes


def check_self_evident(trace, compact_source_path=None, abi_path=None,
                       scheduling_model=None):
    """自证不变量:不依赖契约,同一份产物内部查矛盾。"""
    fails = []
    notes = []
    schema_fails = _check_perfetto_schema(trace)
    fails.extend(schema_fails)
    notes.append(f'Perfetto static schema failures = {len(schema_fails)}')
    events = trace.get('traceEvents', trace if isinstance(trace, list) else [])
    spans = [e for e in events if e.get('ph') == 'X']
    counters = [e for e in events if e.get('ph') == 'C']
    semantic_viewer = (trace.get('otherData') or {}).get('semantic_viewer') \
        if isinstance(trace, dict) else None
    hides_gemm_child_phases = (
        isinstance(semantic_viewer, dict)
        and semantic_viewer.get('mode') == 'hide_gemm_child_phases'
        and bool(semantic_viewer.get('hidden_gemm_phase_ids'))
    )
    hides_all_phases = (
        isinstance(semantic_viewer, dict)
        and semantic_viewer.get('mode') in ('hide_all_phases', 'hide_all_phase_layers')
    )
    # A task-granularity contract records ONLY task-layer events on the device,
    # so the delivered timeline has a task layer and no phase layer by design --
    # nothing is hidden at the viewer level. emit_perfetto records this as a
    # standalone otherData.capture_granularity ledger (semantic_viewer stays null
    # so the finalizer treats it as an ordinary direct delivery). Recognize it so
    # the two-layer requirement accepts an intentionally phase-less trace (any
    # leaked phase span is still rejected below).
    capture_granularity = (trace.get('otherData') or {}).get('capture_granularity') \
        if isinstance(trace, dict) else None
    task_only_view = (
        isinstance(capture_granularity, dict)
        and capture_granularity.get('granularity') == 'task'
    )
    abi_task_ids = _abi_task_ids(abi_path)

    ts_anomalies = find_timestamp_anomalies(events)
    if ts_anomalies["invalid_count"]:
        fails.append(("timeline_invalid_timestamp",
                      f'{ts_anomalies["invalid_count"]} timed events have non-numeric/non-finite ts; '
                      f'samples={ts_anomalies["invalid_samples"]}'))
    if ts_anomalies["negative_count"]:
        fails.append(("timeline_negative_timestamp",
                      f'{ts_anomalies["negative_count"]} timed events have ts<0; '
                      f'Perfetto will drop them; samples={ts_anomalies["negative_samples"]}'))
    notes.append(f'ts<0 timed events = {ts_anomalies["negative_count"]}; '
                 f'invalid ts = {ts_anomalies["invalid_count"]}')
    negative_display = check_nonnegative_timestamps(events)
    geometry_overlaps = check_no_partial_overlap(events)
    if negative_display:
        fails.append(('timeline_negative_display_timestamp',
                      f'{len(negative_display)} numeric ts<0; samples={negative_display[:4]}'))
    if geometry_overlaps:
        fails.append(('slice_partial_overlap_display',
                      f'{len(geometry_overlaps)} same-lane X intervals partially overlap; '
                      f'samples={geometry_overlaps[:4]}'))
    notes.append(f'display geometry: negative={len(negative_display)}, partial_overlaps={len(geometry_overlaps)}')

    # A global derived task envelope is rendered on an auxiliary TASK process.
    # Keep it in the semantic view, but remove that process from CTA/warp
    # topology validation so its pid/tid cannot be counted as a warp group.
    layout_trace, has_aux_task_process, aux_task_pids = (
        _layout_trace_without_auxiliary_tasks(trace, spans))
    if hides_all_phases:
        source_layout_fails, source_layout_notes = _source_layout_for_hidden_phases(
            compact_source_path, semantic_viewer, abi_task_ids)
        fails.extend(source_layout_fails)
        notes.extend(source_layout_notes)
    contract = LAYOUT_CHECKER.contract_from_trace(layout_trace)
    if contract is None:
        fails.append(('layout_contract_missing',
                      'trace lacks embedded perfetto_layout_contract metadata'))
    else:
        try:
            final_contract = dict(contract)
            if hides_all_phases and abi_task_ids:
                final_contract['allowed_task_ids'] = list(abi_task_ids)
            layout_fails = LAYOUT_CHECKER.verify(layout_trace, final_contract)
        except SystemExit as error:
            fails.append(('layout_contract_invalid', str(error)))
        else:
            if has_aux_task_process:
                ignored_auxiliary_v4 = (
                    'V4 [task_layer_missing]',
                    'V4 [task_mma_phase_missing]',
                    'V4 [derived_task_visible]',
                    'V4 [task_source_measurement_missing]',
                    'V4 [invalid_task_provenance]',
                )
                layout_fails = [message for message in layout_fails
                                if not any(message.startswith(prefix)
                                           for prefix in ignored_auxiliary_v4)]
                notes.append('auxiliary global task process excluded from CTA/warp layout: '
                             f'pids={sorted(aux_task_pids)}')
            if hides_gemm_child_phases:
                hidden_mma_failures = sum(
                    message.startswith('V4 [task_mma_phase_missing]')
                    for message in layout_fails)
                layout_fails = [
                    message for message in layout_fails
                    if not message.startswith('V4 [task_mma_phase_missing]')
                ]
                notes.append('hide_gemm_child_phases intentionally omits GEMM MMA children: '
                             f'ignored layout-only MMA requirements={hidden_mma_failures}')
            if hides_all_phases:
                hidden_phase_failures = sum(
                    message.startswith(('V4 [task_phase_missing]',
                                         'V4 [task_mma_phase_missing]'))
                    for message in layout_fails)
                layout_fails = [
                    message for message in layout_fails
                    if not message.startswith(('V4 [task_phase_missing]',
                                               'V4 [task_mma_phase_missing]'))
                ]
                notes.append('hide_all_phases final viewer intentionally omits all phase slices: '
                             f'ignored layout-only task-phase requirements={hidden_phase_failures}')
            for message in layout_fails:
                fails.append(('layout_contract', message))
            notes.append(f'layout contract failures = {len(layout_fails)}')

    if not spans:
        fails.append(('no_spans', 'trace 里没有 ph=X 区间'))
        return fails, notes
    if ts_anomalies["invalid_count"]:
        return fails, notes

    bad_dur = sum(1 for e in spans if float(e.get('dur', 0)) < 0)
    if bad_dur:
        fails.append(('negative_duration', f'{bad_dur} 个区间 dur<0，时钟源或解码配对有问题'))
    notes.append(f'dur<0 区间 = {bad_dur}')

    partial_overlaps, overlap_samples = find_partial_overlaps(spans)
    notes.append(f'same-lane partial overlaps = {partial_overlaps}')
    if partial_overlaps:
        fails.append(('slice_partial_overlap',
                      f'{partial_overlaps} same-lane X-slice crossings; '
                      f'only disjoint or nested intervals are legal; samples={overlap_samples}'))

    operator_fails, operator_notes = check_operator_task_shape(spans, semantic_viewer)
    fails.extend(operator_fails)
    notes.extend(operator_notes)

    display_fails, display_notes = check_slice_display_names(spans, abi_path)
    fails.extend(display_fails)
    notes.extend(display_notes)

    claim_fails, claim_notes = check_task_claim_contract(
        trace, spans, scheduling_model=scheduling_model, abi_path=abi_path)
    fails.extend(claim_fails)
    notes.extend(claim_notes)

    depths = defaultdict(int)
    for e in spans:
        depths[int(e.get('args', {}).get('depth', 0) or 0)] += 1
    notes.append(f'depth 分布 = {dict(sorted(depths.items()))}')
    layers = defaultdict(int)
    for event in spans:
        layers[span_layer(event)] += 1
    if hides_all_phases or task_only_view:
        visible_phase_count = sum(1 for event in spans if _semantic_layer(event) == 'phase')
        if visible_phase_count:
            leak_mode = 'task-only' if task_only_view else 'hide-all-phases'
            fails.append(('semantic_viewer_phase_leak',
                          f'{leak_mode} final viewer contains {visible_phase_count} phase spans'))
    notes.append(f'layer 分布 = {dict(sorted(layers.items(), key=lambda item: str(item[0])))}')
    if not layers.get('task') or (not layers.get('phase')
                                  and not (hides_all_phases or task_only_view)):
        fails.append(('no_task_layer',
                      f'缺 source task / real phase 双层，layer 分布={dict(layers)}'))

    sw_t0 = min(e['ts'] for e in spans)
    sw_t1 = max(e['ts'] + e.get('dur', 0) for e in spans)
    sw_span_us = sw_t1 - sw_t0
    notes.append(f'sw 跨度 = {sw_span_us / 1000:.3f} ms')

    if counters:
        hw_ts = [e['ts'] for e in counters]
        hw_span_us = max(hw_ts) - min(hw_ts)
        coverage = hw_span_us / sw_span_us if sw_span_us else 0
        notes.append(f'hw 跨度 = {hw_span_us / 1000:.3f} ms，覆盖 sw 的 {coverage * 100:.1f}%')
        if coverage > 2.0:
            fails.append(('hw_span_too_long',
                          f'hw 跨度是 sw 的 {coverage:.1f} 倍：counter 没裁到 kernel 窗口，'
                          f'图上 sw 会挤在开头、后面拖一长串空白'))
        if coverage < 0.5:
            fails.append(('hw_span_too_short',
                          f'hw 只覆盖 sw 的 {coverage * 100:.1f}%：hw 与 sw 多半不是同一个 build '
                          f'(常见于 hw 采未插桩 timing build、sw 是插桩 build)。两边必须同一 build。'))

        # counter 全零：整份 hw 里没有任何一个采样点 > 0。
        # 只要 kernel 在 GPU 上执行，SMs Active 之类的峰值必然 > 0；
        # 全 0 说明采样窗口落在 kernel 之外，GPU 当时空闲。
        # 区分：mean 低（3%、5%）是 kernel 真实占用低，不判；max 全 0 才是采集失败。
        per_counter_max = defaultdict(float)
        for event in counters:
            name = event.get('name', '?')
            for value in (event.get('args') or {}).values():
                try:
                    per_counter_max[name] = max(per_counter_max[name], float(value))
                except (TypeError, ValueError):
                    continue
        if per_counter_max:
            global_max = max(per_counter_max.values())
            live = sum(1 for v in per_counter_max.values() if v > 0)
            notes.append(f'counter 峰值：{live}/{len(per_counter_max)} 条非零，'
                         f'全局 max = {global_max:.2f}')
            # SMs Active 是「GPU 在不在跑目标 kernel」的直接证据，单独判。
            # 其他 counter(DRAM/NVLink)可能因 kernel 特性天然接近 0，不能代表采到了。
            sm_names = [n for n in per_counter_max if 'sms active' in n.lower()]
            if sm_names:
                sm_max = max(per_counter_max[n] for n in sm_names)
                notes.append(f'SMs Active 峰值 = {sm_max:.2f}')
                if sm_max <= 0:
                    fails.append(('hw_all_zero',
                                  f'SMs Active 峰值为 0：采样窗口落在 kernel 之外，'
                                  f'GPU 当时没在跑目标 kernel。kernel 只要在执行，'
                                  f'SMs Active 峰值不可能是 0。重采时用插桩 build + 多次 launch，'
                                  f'让采样落在 kernel 执行期间。'
                                  f'(mean 低是 kernel 真实占用低，不算错；峰值为 0 才是采集失败)'))
            elif global_max <= 0:
                fails.append(('hw_all_zero',
                              f'全部 {len(per_counter_max)} 条 counter 峰值都是 0：'
                              f'采样窗口落在 kernel 之外，GPU 当时没在跑目标 kernel。'))
    else:
        notes.append('无 ph=C(纯 sw trace)')

    return fails, notes


def _print_failures(fails):
    """Print bounded failure summaries; never flood stdout with repeated V4 rows."""
    grouped = defaultdict(list)
    for fail_id, message in fails:
        grouped[fail_id].append(message)
    for fail_id, messages in grouped.items():
        if len(messages) <= 8:
            for message in messages:
                print(f'  FAIL [{fail_id}] {message}')
            continue
        print(f'  FAIL [{fail_id}] occurrences={len(messages)}; samples:')
        for message in messages[:3]:
            print(f'    - {message}')


def main() -> int:
    ap = argparse.ArgumentParser(description='交付闸门:上 OSS 前必过')
    ap.add_argument('project', help='项目根目录')
    ap.add_argument('trace', help='最终 sw_hw trace')
    ap.add_argument('--contract', default=None,
                    help='显式 contract.json 路径')
    ap.add_argument('--information-bundle', choices=('sw', 'hw', 'sw+hw'), default=None,
                    help='显式声明交付包；hw/sw+hw 会强制读取 capture_manifest 与 hw_metrics')
    ap.add_argument('--capture-manifest', default=None,
                    help='HW 交付的 capture_manifest.json；缺省从 trace 同目录或 hw/ 推导')
    ap.add_argument('--hw-metrics', default=None,
                    help='HW 交付的 hw_metrics.json；缺省从 trace 同目录或 hw/ 推导')
    ap.add_argument('--compact-exact', default=None,
                    help='semantic viewer 的内部 compact-exact 源文件；仅用于交付前校验，不是交付物')
    ap.add_argument('--abi', default=None,
                    help='semantic viewer ABI JSON；也可由 metadata visibility_contract.abi_path 提供')
    ap.add_argument('--delivery-dir', default=None,
                    help='兼容旧命令保留；不再限制目录中的其他运行产物')
    ap.add_argument('--skip-validate', action='store_true',
                    help='只跑自证判据，跳过 validate_capability(不建议)')
    ap.add_argument('--trace-processor', default=None,
                    help='trace_processor_shell absolute path; mandatory for a PASS delivery')
    ap.add_argument('--receipt', default=None,
                    help='PASS receipt path; default is <trace>.delivery_receipt.json')
    args = ap.parse_args()

    project = Path(args.project).resolve()
    trace_path = Path(args.trace).resolve()
    contract_path = (Path(args.contract).expanduser().resolve()
                     if args.contract else project / 'profile' / 'contract.json')

    print(f'=== 交付闸门 ===')
    print(f'  项目   {project}')
    print(f'  trace  {trace_path}')
    print(f'  契约   {contract_path}{"" if contract_path.is_file() else "  (不存在)"}')

    if not trace_path.is_file():
        print(f'[fatal] trace 不存在: {trace_path}')
        return 2

    # SPEC-3: resolve the declared scheduling model from the contract up front so
    # the self-evident pass can choose the task-claim vs phase-chain closure check.
    contract_data = {}
    if contract_path.is_file():
        contract_data, contract_error = _load_json_file(contract_path)
        if contract_error:
            print(f'  WARN [contract_json] {contract_error}')
            contract_data = {}
    scheduling_model = _contract_scheduling_model(contract_data)

    print('\n--- trace 文件名规范 (SPEC-1) ---')
    name_fails, name_notes = check_trace_filename_schema(trace_path, contract_data)
    for note in name_notes:
        print(f'  {note}')
    _print_failures(name_fails)

    print('\n--- 自证不变量 ---')
    trace = load_trace(trace_path)
    fails, notes = check_self_evident(
        trace, compact_source_path=args.compact_exact, abi_path=args.abi,
        scheduling_model=scheduling_model)
    semantic_descriptors = [descriptor for descriptor in task_descriptors(contract_data)
                            if isinstance(descriptor.get('task_contract'), dict)]
    if semantic_descriptors:
        layout = contract_data.get('project_layout') \
            if isinstance(contract_data.get('project_layout'), dict) else {}
        semantic_run_root = Path(layout.get('run_root', trace_path.parent)).expanduser().resolve()
        semantic_failures = audit_semantic_task_bijection(
            contract_data, semantic_run_root, trace)
        fails.extend(('semantic_task_bijection', message)
                     for message in semantic_failures)
        notes.append(f'semantic task bijection: declared={len(semantic_descriptors)}, '
                     f'failures={len(semantic_failures)}')
    fails.extend(name_fails)
    if args.skip_validate:
        fails.append(('delivery_skip_validate_forbidden',
                      '--skip-validate is diagnostic-only and can never produce a delivery receipt'))
    for note in notes:
        print(f'  {note}')
    _print_failures(fails)

    print('\n--- size/semantic viewer policy ---')
    size_fails, size_notes, trace_is_viewer = _check_semantic_viewer(
        trace, trace_path, compact_source_path=args.compact_exact, abi_path=args.abi)
    for note in size_notes:
        print(f'  {note}')
    _print_failures(size_fails)
    fails.extend(size_fails)

    trace_events = trace.get('traceEvents', trace if isinstance(trace, list) else [])
    trace_has_hw = any(event.get('ph') == 'C' for event in trace_events
                       if isinstance(event, dict))
    trace_has_global_envelope = any(
        event.get('ph') == 'X' and is_global_task_envelope(event)
        for event in trace_events if isinstance(event, dict))
    trace_has_semantic_viewer = isinstance((trace.get('otherData') or {}).get('semantic_viewer'), dict)
    contract_requires_hw = bool(contract_data.get('require_hw')) or str(
        contract_data.get('information_bundle', '')).lower() in ('hw', 'sw+hw')
    cli_requires_hw = args.information_bundle in ('hw', 'sw+hw')
    hw_required = trace_has_hw or contract_requires_hw or cli_requires_hw
    if hw_required:
        print('\n--- HW capture evidence ---')
        hw_fails, hw_notes = check_hw_bundle(
            trace_path, contract_path,
            manifest_arg=args.capture_manifest,
            metrics_arg=args.hw_metrics)
        for note in hw_notes:
            print(f'  {note}')
        _print_failures(hw_fails)
        fails.extend(hw_fails)
    else:
        print('\n--- HW capture evidence ---')
        print('  SW-only：跳过 capture_manifest/hw_metrics 硬门禁')

    # The legacy validate_capability CLI audit assumes a single GEMM operator
    # (task_id 7/10 math role, a per-task MMA phase, hidden barriers, and block
    # coverage == the full launch grid). A static_phase_chain megakernel instead
    # declares first-class phases (load/mma/attn/store/barrier/...) and captures a
    # selected SM subset, so none of those GEMM assumptions hold. Its task/phase
    # shape is already validated authoritatively by the deliver_gate operator/
    # phase gate (G17) + phase-chain closure, and its SM scope by G13, so the
    # legacy operator audit is skipped exactly as it is for viewer/envelope traces.
    # A task-granularity delivery (otherData.capture_granularity.granularity == 'task',
    # frozen _ph0 filename) draws no phase layer BY CONSTRUCTION, and SPEC s2
    # display_rank draws exactly one rank by design, so the legacy audit's
    # phase_count / rank_coverage / phase_kind assumptions can never hold for it.
    # check_self_evident already recognizes this task-only view (task_only_view);
    # exempt it from the legacy audit exactly like the envelope/viewer traces.
    trace_capture_granularity = (trace.get('otherData') or {}).get('capture_granularity') \
        if isinstance(trace, dict) else None
    task_only_delivery = (
        isinstance(trace_capture_granularity, dict)
        and trace_capture_granularity.get('granularity') == 'task')
    legacy_validator_not_applicable = (
        str(scheduling_model or '').strip().lower() == 'static_phase_chain')
    if (trace_has_global_envelope or trace_has_semantic_viewer or task_only_delivery
            or legacy_validator_not_applicable) and not args.skip_validate:
        print('\n--- legacy validate_capability ---')
        if task_only_delivery and not (legacy_validator_not_applicable
                                       or trace_has_global_envelope
                                       or trace_has_semantic_viewer):
            print('  task-granularity delivery (capture_granularity.granularity=task): '
                  'phase layer decoded but intentionally not drawn (frozen _ph0 contract), '
                  'display_rank draws one rank by design; check_self_evident recognizes '
                  'the task-only view, deliver_gate operator/phase gate is authoritative')
        elif legacy_validator_not_applicable:
            print('  static_phase_chain megakernel: skip legacy GEMM operator audit; '
                  'deliver_gate operator/phase gate (G17) + phase-chain closure + '
                  'G13 SM scope are authoritative')
        else:
            print('  envelope/semantic-viewer trace: skip legacy same-lane task validator; '
                  'deliver_gate operator/phase gate is authoritative')
    elif not args.skip_validate:
        print('\n--- 契约比对 (validate_capability) ---')
        validator = SCRIPT_DIR / 'validate_capability.py'
        cmd = [sys.executable, '-B', str(validator), str(trace_path)]
        if contract_path.is_file():
            cmd += ['--contract', str(contract_path)]
        else:
            print('  契约文件缺失，跳过契约比对。'
                  '强烈建议按 configs/templates/capture/contract_reference.json 建一份 '
                  f'{contract_path}，否则每次都要手敲 --expect-* 且容易传错。')
            cmd = None
        if cmd:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            for line in (proc.stdout or '').splitlines():
                print(f'  {line}')
            if proc.returncode != 0:
                fails.append(('validate_capability', f'退出码 {proc.returncode}'))

    print('\n--- real Perfetto import ---')
    if fails:
        print('  skipped because an earlier delivery invariant already failed')
    else:
        import_fails, import_notes = _check_real_perfetto_import(
            trace_path, executable=args.trace_processor)
        for note in import_notes:
            print(f'  {note}')
        _print_failures(import_fails)
        fails.extend(import_fails)

    print()
    if fails:
        print(f'=== 闸门 FAIL ({len(fails)} 项) —— 禁止上传 OSS ===')
        _print_failures(fails)
        return 1
    receipt_path = (Path(args.receipt).expanduser().resolve() if args.receipt
                    else Path(str(trace_path) + '.delivery_receipt.json'))
    receipt = {
        'schema': 'fast-instrument-v2-delivery-receipt-v1',
        'status': 'PASS',
        'trace_path': str(trace_path),
        'trace_sha256': _file_sha256(trace_path),
        'trace_size_bytes': trace_path.stat().st_size,
        'producer': (trace.get('otherData') or {}).get('producer'),
        'trace_processor_shell': str(Path(args.trace_processor).expanduser().resolve())
            if args.trace_processor else shutil.which('trace_processor_shell'),
        'real_perfetto_import': 'PASS',
        'forbidden_perfetto_stats': {name: 0 for name in FORBIDDEN_IMPORT_STATS},
        'required_gpu_metrics_frequency_hz': HW_PREFERRED_FREQUENCY_HZ,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n',
                            encoding='utf-8')
    print(f'=== 闸门 PASS —— receipt {receipt_path} ===')
    return 0


if __name__ == '__main__':
    sys.exit(main())
