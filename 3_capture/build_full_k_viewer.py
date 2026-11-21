#!/usr/bin/env python3
"""Build a compact, ABI-driven Perfetto viewer without sampling source events."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SCHEMA = 'perfetto-semantic-viewer/v2'
EVENT_LIST_KEYS = ('events', 'intervals', 'records', 'spans')
TASK_ID_KEYS = ('task_id', 'taskId', 'task_index', 'task_idx')
VALID_POLICIES = {'exact', 'summary', 'hidden'}
CAPTURE_SCOPE_PATH = Path(__file__).resolve().parent / 'capture_scope.py'


class ViewerError(RuntimeError):
    pass


def load_capture_scope_checker():
    spec = importlib.util.spec_from_file_location('fast_instrument_capture_scope',
                                                  CAPTURE_SCOPE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load capture scope checker: {CAPTURE_SCOPE_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAPTURE_SCOPE = load_capture_scope_checker()


def load_json(path: Path) -> Any:
    opener = gzip.open if path.name.endswith('.gz') else open
    try:
        with opener(path, 'rt', encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError) as error:
        raise ViewerError(f'无法读取 JSON {path}: {error}') from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def event_list(doc: Any, path: Path) -> List[Dict[str, Any]]:
    if isinstance(doc, list):
        rows = doc
    elif isinstance(doc, dict):
        rows = None
        for key in EVENT_LIST_KEYS:
            if isinstance(doc.get(key), list):
                rows = doc[key]
                break
        if rows is None:
            raise ViewerError(f'{path} 缺少事件列表，候选字段={EVENT_LIST_KEYS}')
    else:
        raise ViewerError(f'{path} 顶层必须是 object 或 list')
    if not all(isinstance(item, dict) for item in rows):
        raise ViewerError(f'{path} 事件列表含非 object 元素')
    return rows


def pick_int(row: Dict[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def decoded_counts(paths: Sequence[Path]) -> Counter:
    counts: Counter = Counter()
    for path in paths:
        for row in event_list(load_json(path), path):
            task_id = pick_int(row, TASK_ID_KEYS)
            if task_id is None:
                raise ViewerError(f'{path} 中存在缺失 task_id 的 decoded event')
            counts[task_id] += 1
    return counts


def parse_abi(path: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    abi = load_json(path)
    if not isinstance(abi, dict) or not isinstance(abi.get('tasks'), list):
        raise ViewerError(f'{path} 缺少 ABI tasks 列表')
    tasks: Dict[int, Dict[str, Any]] = {}
    for item in abi['tasks']:
        if not isinstance(item, dict):
            raise ViewerError(f'{path} tasks 含非 object 元素')
        task_id = item.get('id')
        policy = item.get('viewer_policy')
        layer = item.get('layer')
        if not isinstance(task_id, int) or isinstance(task_id, bool):
            raise ViewerError(f'{path} task id 非整数: {task_id!r}')
        if task_id in tasks:
            raise ViewerError(f'{path} task id 重复: {task_id}')
        if policy not in VALID_POLICIES:
            raise ViewerError(f'{path} task_id={task_id} viewer_policy={policy!r} 无效')
        expected_policy = ('exact' if layer == 'task'
                           else 'hidden' if layer == 'ignored' else None)
        if expected_policy is not None and policy != expected_policy:
            raise ViewerError(
                f'{path} task_id={task_id} layer={layer!r} 与 viewer_policy={policy!r} 不一致')
        if layer == 'phase' and policy not in ('exact', 'summary'):
            raise ViewerError(
                f'{path} task_id={task_id} phase 必须声明 exact 或 summary')
        tasks[task_id] = copy.deepcopy(item)
    parent_ids = {item.get('parent_task_id') for item in tasks.values()
                  if item.get('parent_task_id') is not None}
    missing_parents = sorted(parent_id for parent_id in parent_ids if parent_id not in tasks)
    if missing_parents:
        raise ViewerError(f'{path} parent_task_id 不存在: {missing_parents}')
    return tasks, abi


def event_task_id(event: Dict[str, Any]) -> int | None:
    args = event.get('args')
    return args.get('task_id') if isinstance(args, dict) and isinstance(args.get('task_id'), int) else None


def finite_number(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ViewerError(f'时间值不是数值: {value!r}')
    result = float(value)
    if not math.isfinite(result):
        raise ViewerError(f'时间值非有限: {value!r}')
    return result


def geometry_digest(events: Iterable[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for event in events:
        row = [event.get(field) for field in ('name', 'cat', 'ph', 'pid', 'tid', 'ts', 'dur')]
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 separators=(',', ':'), allow_nan=False).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def counter_digest(events: Iterable[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for event in events:
        digest.update(json.dumps(event, ensure_ascii=False, sort_keys=True,
                                 separators=(',', ':'), allow_nan=False).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def filter_display_sm_stride(trace: Dict[str, Any], stride: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Keep a deterministic physical-SM subset from a complete all-SM capture."""
    raise ViewerError(
        'postprocess physical-SM filtering is forbidden; choose a physical-SM set before source capture')
    if stride < 1:
        raise ViewerError(f'--sm-display-stride must be >= 1, got {stride}')
    events = trace.get('traceEvents')
    if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
        raise ViewerError('input traceEvents must be a list of objects')
    if stride == 1:
        return copy.deepcopy(trace), {'policy': 'all_physical_sms', 'stride': 1}

    source_smids = sorted({
        args.get('physical_smid')
        for event in events if event.get('ph') == 'X'
        for args in [event.get('args') if isinstance(event.get('args'), dict) else {}]
        if isinstance(args.get('physical_smid'), int)
    })
    if not source_smids:
        raise ViewerError('cannot apply SM display filter: software intervals lack physical_smid')
    selected_smids = [smid for smid in source_smids if smid % stride == 0]
    if not selected_smids:
        raise ViewerError(f'SM display stride {stride} selected no physical SM')
    filtered, selection = filter_display_smids(trace, selected_smids)
    selection.update({
        'policy': 'physical_smid_modulo',
        'stride': stride,
        'predicate': f'physical_smid % {stride} == 0',
    })
    return filtered, selection


def filter_display_smids(trace: Dict[str, Any], selected_smids: Sequence[int]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Keep exactly the requested physical SMs while preserving every ph=C event."""
    raise ViewerError(
        'postprocess physical-SM filtering is forbidden; choose a physical-SM set before source capture')
    events = trace.get('traceEvents')
    if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
        raise ViewerError('input traceEvents must be a list of objects')
    source_smids = sorted({
        args.get('physical_smid')
        for event in events if event.get('ph') == 'X'
        for args in [event.get('args') if isinstance(event.get('args'), dict) else {}]
        if isinstance(args.get('physical_smid'), int)
    })
    selected_smids = sorted(set(selected_smids))
    if not selected_smids or not set(selected_smids).issubset(set(source_smids)):
        raise ViewerError('selected physical SMs must be a non-empty subset of the input SM set')
    selected_set = set(selected_smids)
    all_software_pids = {
        event.get('pid') for event in events
        if event.get('ph') == 'X'
        and isinstance(event.get('args'), dict)
        and isinstance(event['args'].get('physical_smid'), int)
    }
    selected_pids = {
        event.get('pid') for event in events
        if event.get('ph') == 'X'
        and isinstance(event.get('args'), dict)
        and event['args'].get('physical_smid') in selected_set
    }
    output_events: List[Dict[str, Any]] = []
    source_x = kept_x = source_c = kept_c = 0
    for event in events:
        ph = event.get('ph')
        if ph == 'C':
            source_c += 1
            kept_c += 1
            output_events.append(copy.deepcopy(event))
            continue
        if ph == 'X':
            source_x += 1
            args = event.get('args') if isinstance(event.get('args'), dict) else {}
            smid = args.get('physical_smid')
            if isinstance(smid, int) and smid not in selected_set:
                continue
            kept_x += 1
            output_events.append(copy.deepcopy(event))
            continue
        if ph == 'M' and event.get('pid') in all_software_pids and event.get('pid') not in selected_pids:
            continue
        output_events.append(copy.deepcopy(event))
    if kept_c != source_c:
        raise ViewerError('SM display filter changed hardware counter events')
    filtered = copy.deepcopy(trace)
    filtered['traceEvents'] = output_events
    return filtered, {
        'policy': 'physical_smid_explicit_subset',
        'original_physical_smids': source_smids,
        'display_physical_smids': selected_smids,
        'source_physical_smids': source_smids,
        'displayed_physical_smids': selected_smids,
        'source_sm_count': len(source_smids),
        'displayed_sm_count': len(selected_smids),
        'source_software_interval_count': source_x,
        'displayed_software_interval_count': kept_x,
        'hardware_counter_event_count': kept_c,
        'hardware_scope': 'unchanged device-wide counters',
        'source_capture_scope': 'all physical SMs',
        'event_sampling_within_displayed_sm': 'none',
    }


def trace_counts(trace: Dict[str, Any]) -> Counter:
    """Count only decoded X events present in the current display candidate."""
    events = trace.get('traceEvents')
    if not isinstance(events, list):
        raise ViewerError('input traceEvents must be a list')
    counts: Counter = Counter()
    for event in events:
        if event.get('ph') != 'X':
            continue
        task_id = event_task_id(event)
        if task_id is not None:
            counts[task_id] += 1
    return counts


def canonicalize_event_semantics(trace: Dict[str, Any],
                                 tasks: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Apply the current ABI's semantic labels without changing measured geometry."""
    events = trace.get('traceEvents')
    if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
        raise ViewerError('input traceEvents must be a list of objects')

    for task_id, task in tasks.items():
        if task.get('layer') != 'phase':
            continue
        parent_id = task.get('parent_task_id')
        # Some measured phases (for example a TMA engine interval shared by
        # GEMM0 and GEMM1) deliberately have no unique task parent.  They may
        # remain unparented in the detailed trace and can still be hidden by
        # the task-only viewer; an explicitly declared parent must be valid.
        if parent_id is None:
            parent_ids = task.get('parent_task_ids')
            if (task.get('shared_across_parent_tasks') is True
                    and isinstance(parent_ids, list) and len(parent_ids) >= 2
                    and all(isinstance(item, int) and not isinstance(item, bool)
                            and isinstance(tasks.get(item), dict)
                            and tasks[item].get('layer') == 'task'
                            for item in parent_ids)):
                continue
            raise ViewerError(
                f'ABI task_id={task_id} unparented phase must declare '
                'shared_across_parent_tasks=true and at least two valid parent_task_ids')
        parent = tasks.get(parent_id) if isinstance(parent_id, int) else None
        if not isinstance(parent, dict) or parent.get('layer') != 'task':
            raise ViewerError(
                f'ABI task_id={task_id} layer=phase must reference a layer=task parent')

    # The input document is an in-memory decode of an immutable source file;
    # relabel it in place to avoid duplicating millions of measured intervals.
    output = trace
    relabeled = 0
    for event in output['traceEvents']:
        if event.get('ph') != 'X':
            continue
        task_id = event_task_id(event)
        if task_id is None:
            continue
        task = tasks.get(task_id)
        if not isinstance(task, dict):
            raise ViewerError(f'trace contains task_id={task_id} absent from ABI')
        args = event.get('args')
        if not isinstance(args, dict):
            args = {}
            event['args'] = args
        old_geometry = (event.get('pid'), event.get('tid'), event.get('ts'), event.get('dur'))
        event['name'] = str(task.get('short_name') or task.get('name') or event.get('name'))
        event['cat'] = f"{task.get('layer')}/{task.get('semantic_kind')}"
        args['task'] = task.get('name')
        args['layer'] = task.get('layer')
        args['semantic_kind'] = task.get('semantic_kind')
        task_role = task.get('role')
        if isinstance(task_role, str) and task_role:
            args['task_role'] = task_role
        else:
            # Preserve the measured event's lane role when the task descriptor
            # intentionally has no task-global role (the same tile is observed
            # by multiple physical warp roles).
            args.pop('task_role', None)
        if old_geometry != (event.get('pid'), event.get('tid'), event.get('ts'), event.get('dur')):
            raise ViewerError('ABI semantic canonicalization changed measured event geometry')
        relabeled += 1
    other = output.setdefault('otherData', {})
    if not isinstance(other, dict):
        raise ViewerError('input otherData must be an object')
    other['abi_semantic_canonicalization'] = {
        'applied': True,
        'x_events_relabelled': relabeled,
        'geometry_fields_unchanged': ['pid', 'tid', 'ts', 'dur'],
        'source_of_labels': 'current ABI task_id mapping',
    }
    return output


def hide_all_phase_events(trace: Dict[str, Any], tasks: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Hide every ABI layer=phase interval; keep all task/CTA/SM and ph=C events."""
    phase_ids = {task_id for task_id, task in tasks.items() if task.get('layer') == 'phase'}
    events = trace.get('traceEvents')
    if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
        raise ViewerError('input traceEvents must be a list of objects')
    output_events: List[Dict[str, Any]] = []
    hidden_counts: Counter = Counter()
    source_counts: Counter = Counter()
    source_c = kept_c = 0
    for event in events:
        if event.get('ph') == 'C':
            source_c += 1
            kept_c += 1
        if event.get('ph') == 'X':
            task_id = event_task_id(event)
            if task_id is not None:
                source_counts[task_id] += 1
                if task_id in phase_ids:
                    hidden_counts[task_id] += 1
                    continue
        output_events.append(copy.deepcopy(event))
    if source_c != kept_c:
        raise ViewerError('phase hiding changed hardware counter events')
    filtered = copy.deepcopy(trace)
    filtered['traceEvents'] = output_events
    return filtered, {
        'policy': 'hide_all_abi_phase_layers',
        'hidden_phase_task_ids': sorted(phase_ids),
        'hidden_event_counts_by_task': {str(task_id): int(hidden_counts[task_id])
                                        for task_id in sorted(phase_ids)},
        'source_event_counts_by_task': {str(task_id): int(source_counts[task_id])
                                       for task_id in sorted(source_counts)},
        'hardware_counter_event_count': kept_c,
        'hardware_scope': 'unchanged; all ph=C events preserved',
    }


def candidate_bytes(trace: Dict[str, Any], output_path: Path) -> int:
    """Return the exact serialized size used by write_output without touching disk."""
    payload = json.dumps(trace, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    if not output_path.name.endswith('.gz'):
        return len(payload)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb', compresslevel=9, mtime=0) as handle:
        handle.write(payload)
    return len(buffer.getvalue())


def thread_metadata(events: Sequence[Dict[str, Any]]) -> Tuple[Dict[Tuple[Any, Any], str],
                                                               Dict[Any, int], Dict[Any, int]]:
    names: Dict[Tuple[Any, Any], str] = {}
    max_tid: Dict[Any, int] = defaultdict(int)
    max_sort: Dict[Any, int] = defaultdict(int)
    for event in events:
        if event.get('ph') != 'M':
            continue
        pid, tid = event.get('pid'), event.get('tid')
        if isinstance(tid, int):
            max_tid[pid] = max(max_tid[pid], tid)
        args = event.get('args') if isinstance(event.get('args'), dict) else {}
        if event.get('name') == 'thread_name' and isinstance(args.get('name'), str):
            names[(pid, tid)] = args['name']
        if event.get('name') == 'thread_sort_index' and isinstance(args.get('sort_index'), int):
            max_sort[pid] = max(max_sort[pid], args['sort_index'])
    return names, max_tid, max_sort


def summary_group_key(event: Dict[str, Any]) -> Tuple[Any, ...]:
    args = event.get('args') if isinstance(event.get('args'), dict) else {}
    return (event.get('pid'), event_task_id(event), args.get('rank'),
            args.get('block'), args.get('warp_id'), event.get('tid'))


def build_summaries(source: Sequence[Dict[str, Any]], task: Dict[str, Any],
                    names: Dict[Tuple[Any, Any], str], max_tid: Dict[Any, int],
                    max_sort: Dict[Any, int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for event in source:
        groups[summary_group_key(event)].append(event)
    metadata: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    provenance: List[Dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        pid, task_id, rank, block, warp_id, source_tid = key
        # Summary phases reuse the originating observer tid; no summary lane.
        tid = source_tid
        short_name = str(task.get('short_name') or task.get('name') or f'task{task_id}')
        # Kept only for compatibility with older generated traces; no lane is created.
        source_lane = ''
        lane_name = f'{source_lane}｜{short_name} 汇总'
        starts = [finite_number(row.get('ts')) for row in rows]
        ends = [finite_number(row.get('ts')) + finite_number(row.get('dur', 0)) for row in rows]
        start, end = min(starts), max(ends)
        source_digest = geometry_digest(rows)
        exemplar_args = rows[0].get('args') if isinstance(rows[0].get('args'), dict) else {}
        summary_args = {
            'task_id': task_id,
            'task': task.get('name'),
            'short_name': task.get('short_name'),
            'rank': rank,
            'block': block,
            'rank_block': exemplar_args.get('rank_block'),
            'warp_id': warp_id,
            'role': task.get('role'),
            'layer': task.get('layer'),
            'semantic_kind': task.get('semantic_kind'),
            'is_derived': True,
            'is_summary': True,
            'parent_task_id': task.get('parent_task_id'),
            'aggregation': 'per_rank_cta_observer_warp_minmax',
            'source_event_count': len(rows),
            'source_geometry_sha256': source_digest,
        }
        summaries.append({
            'name': f'{short_name} 汇总',
            'cat': f'summary/{task.get("semantic_kind", "phase")}',
            'ph': 'X', 'pid': pid, 'tid': tid, 'ts': start, 'dur': end - start,
            'args': summary_args,
        })
        provenance.append({
            'task_id': task_id, 'rank': rank, 'block': block, 'warp_id': warp_id,
            'source_tid': source_tid, 'summary_tid': tid, 'source_event_count': len(rows),
            'source_geometry_sha256': source_digest,
        })
    return metadata, summaries, provenance


def build_viewer(trace: Dict[str, Any], tasks: Dict[int, Dict[str, Any]],
                 source_counts: Counter, input_path: Path, abi_path: Path,
                 event_paths: Sequence[Path]) -> Dict[str, Any]:
    events = trace.get('traceEvents')
    if not isinstance(events, list):
        raise ViewerError('输入 trace 缺少 traceEvents')
    if not all(isinstance(event, dict) for event in events):
        raise ViewerError('输入 traceEvents 含非 object 元素')
    x_events = [event for event in events if event.get('ph') == 'X']
    counters = [event for event in events if event.get('ph') == 'C']
    metadata = [copy.deepcopy(event) for event in events if event.get('ph') == 'M']
    source_by_task: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    unclassified: List[Dict[str, Any]] = []
    for event in x_events:
        task_id = event_task_id(event)
        if task_id is None:
            unclassified.append(event)
        else:
            source_by_task[task_id].append(event)

    unknown_task_ids = sorted(set(source_by_task) - set(tasks))
    if unknown_task_ids:
        raise ViewerError(f'输入 trace 含 ABI 未声明 task_id: {unknown_task_ids}')
    unknown_decoded = sorted(set(source_counts) - set(tasks))
    if unknown_decoded:
        raise ViewerError(f'decoded events 含 ABI 未声明 task_id: {unknown_decoded}')

    names, max_tid, max_sort = thread_metadata(events)
    emitted: List[Dict[str, Any]] = [copy.deepcopy(event) for event in unclassified]
    added_metadata: List[Dict[str, Any]] = []
    source_geometry: Dict[str, str] = {}
    emitted_geometry: Dict[str, str] = {}
    emitted_counts: Counter = Counter()
    hidden_counts: Counter = Counter()
    derived_counts: Counter = Counter()
    summaries: Dict[str, Any] = {}

    for task_id, task in sorted(tasks.items()):
        policy = task['viewer_policy']
        decoded_count = int(source_counts.get(task_id, 0))
        detailed_rows = source_by_task.get(task_id, [])
        source_geometry[str(task_id)] = geometry_digest(detailed_rows)
        if policy in ('exact', 'summary') and len(detailed_rows) != decoded_count:
            raise ViewerError(
                f'task_id={task_id} detailed count={len(detailed_rows)} != decoded count={decoded_count}')
        if policy == 'hidden':
            if detailed_rows:
                raise ViewerError(
                    f'task_id={task_id} policy=hidden 但 detailed trace 仍有 {len(detailed_rows)} 条')
            hidden_counts[task_id] = decoded_count
            continue
        if False:
            if decoded_count == 0:
                if detailed_rows:
                    raise ViewerError(f'task_id={task_id} 无 source 却出现 global parent')
                continue
            parents = [row for row in detailed_rows
                       if (row.get('args') or {}).get('is_derived') is True]
            if len(parents) != 1 or len(detailed_rows) != 1:
                raise ViewerError(
                    f'task_id={task_id} source_count={decoded_count} 时必须恰好一个 derived global parent，'
                    f'实际 detailed={len(detailed_rows)}, derived={len(parents)}')
            emitted.append(copy.deepcopy(parents[0]))
            emitted_counts[task_id] = 1
            derived_counts[task_id] = 1
            hidden_counts[task_id] = decoded_count
            continue
        if policy == 'exact':
            emitted.extend(copy.deepcopy(detailed_rows))
            emitted_counts[task_id] = len(detailed_rows)
            emitted_geometry[str(task_id)] = geometry_digest(detailed_rows)
            continue
        summary_meta, summary_rows, provenance = build_summaries(
            detailed_rows, task, names, max_tid, max_sort)
        added_metadata.extend(summary_meta)
        emitted.extend(summary_rows)
        emitted_counts[task_id] = len(summary_rows)
        derived_counts[task_id] = len(summary_rows)
        hidden_counts[task_id] = decoded_count
        summaries[str(task_id)] = {
            'source_event_count': decoded_count,
            'summary_event_count': len(summary_rows),
            'source_geometry_sha256': source_geometry[str(task_id)],
            'groups': provenance,
        }

    for task_id, task in tasks.items():
        if task['viewer_policy'] == 'exact':
            expected = source_geometry[str(task_id)]
            actual = emitted_geometry.get(str(task_id), geometry_digest([]))
            if expected != actual:
                raise ViewerError(f'task_id={task_id} exact geometry digest changed')

    all_events = metadata + added_metadata + emitted + [copy.deepcopy(event) for event in counters]
    for event in all_events:
        if event.get('ph') in ('X', 'C'):
            ts = finite_number(event.get('ts'))
            if ts < 0:
                raise ViewerError(f'输出事件含负时间戳: {event.get("name")!r} ts={ts}')
            if event.get('ph') == 'X' and finite_number(event.get('dur', 0)) < 0:
                raise ViewerError(f'输出区间含负 duration: {event.get("name")!r}')

    viewer = {
        'schema': SCHEMA,
        'source': {
            'path': str(input_path),
            'bytes': input_path.stat().st_size,
            'sha256': sha256_file(input_path),
            'abi_path': str(abi_path),
            'abi_sha256': sha256_file(abi_path),
            'decoded_events': [
                {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
                for path in event_paths
            ],
        },
        'visibility_contract': {
            'source': 'abi.tasks[*].viewer_policy',
            'abi_path': str(abi_path),
            'abi_sha256': sha256_file(abi_path),
            'tasks': [copy.deepcopy(tasks[task_id]) for task_id in sorted(tasks)],
            'summary_allowed_semantic_kinds': sorted({
                str(task.get('semantic_kind')) for task in tasks.values()
                if task.get('viewer_policy') == 'summary'
            }),
        },
        'source_event_counts_by_task': {str(task_id): int(source_counts.get(task_id, 0))
                                        for task_id in sorted(tasks)},
        'detailed_event_counts_by_task': {str(task_id): len(source_by_task.get(task_id, []))
                                          for task_id in sorted(tasks)},
        'emitted_event_counts_by_task': {str(task_id): int(emitted_counts.get(task_id, 0))
                                         for task_id in sorted(tasks)},
        'hidden_event_counts_by_task': {str(task_id): int(hidden_counts.get(task_id, 0))
                                        for task_id in sorted(tasks)},
        'derived_event_counts_by_task': {str(task_id): int(derived_counts.get(task_id, 0))
                                         for task_id in sorted(tasks)},
        'source_geometry_sha256_by_task': source_geometry,
        'emitted_exact_geometry_sha256_by_task': emitted_geometry,
        'summaries_by_task': summaries,
        'unclassified_preserved_event_count': len(unclassified),
        'sampling_policy': 'none',
        'no_sampling': True,
        'no_stride': True,
        'cross_cta_child_merge': False,
        'cta_child_scope': 'rank_cta_observer_warp',
        'hardware_counters': {
            'event_count': len(counters),
            'geometry_sha256': counter_digest(counters),
            'names': sorted({str(event.get('name')) for event in counters}),
        },
    }
    output = copy.deepcopy(trace)
    output['traceEvents'] = all_events
    other = copy.deepcopy(trace.get('otherData') if isinstance(trace.get('otherData'), dict) else {})
    other['semantic_viewer'] = viewer
    output['otherData'] = other
    return output


def build_hide_gemm_phases_viewer(trace: Dict[str, Any],
                                  tasks: Dict[int, Dict[str, Any]],
                                  source_counts: Counter,
                                  input_path: Path,
                                  abi_path: Path,
                                  event_paths: Sequence[Path]) -> Dict[str, Any]:
    """Hide only phases nested under GEMM tasks; preserve every other real event."""
    events = trace.get('traceEvents')
    if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
        raise ViewerError('input traceEvents must be a list of objects')

    gemm_task_ids = {
        task_id for task_id, task in tasks.items()
        if task.get('layer') == 'task'
        and str(task.get('semantic_kind', '')).lower() == 'gemm'
    }
    if not gemm_task_ids:
        raise ViewerError('ABI does not declare a layer=task, semantic_kind=gemm task')

    def has_gemm_parent(task_id: int) -> bool:
        seen = set()
        current = tasks.get(task_id)
        while isinstance(current, dict):
            parent_id = current.get('parent_task_id')
            if parent_id in gemm_task_ids:
                return True
            if not isinstance(parent_id, int) or parent_id in seen:
                return False
            seen.add(parent_id)
            current = tasks.get(parent_id)
        return False

    hidden_phase_ids = {
        task_id for task_id, task in tasks.items()
        if task.get('layer') == 'phase' and has_gemm_parent(task_id)
    }
    if not hidden_phase_ids:
        raise ViewerError('ABI GEMM tasks have no child phases to hide')

    detailed_counts: Counter = Counter()
    emitted_counts: Counter = Counter()
    hidden_counts: Counter = Counter()
    output_events: List[Dict[str, Any]] = []
    source_non_gemm: List[Dict[str, Any]] = []
    counters: List[Dict[str, Any]] = []
    for event in events:
        if event.get('ph') == 'C':
            counters.append(event)
        task_id = event_task_id(event) if event.get('ph') == 'X' else None
        if task_id is not None:
            detailed_counts[task_id] += 1
            if task_id in hidden_phase_ids:
                hidden_counts[task_id] += 1
                continue
            emitted_counts[task_id] += 1
            if task_id not in gemm_task_ids:
                source_non_gemm.append(event)
        output_events.append(copy.deepcopy(event))

    missing_gemm_tasks = sorted(task_id for task_id in gemm_task_ids
                                if source_counts.get(task_id, 0) > 0
                                and emitted_counts.get(task_id, 0) == 0)
    if missing_gemm_tasks:
        raise ViewerError(f'GEMM task parents missing from detailed trace: {missing_gemm_tasks}')
    emitted_non_gemm = [event for event in output_events
                        if event.get('ph') == 'X'
                        and event_task_id(event) is not None
                        and event_task_id(event) not in gemm_task_ids]
    if geometry_digest(source_non_gemm) != geometry_digest(emitted_non_gemm):
        raise ViewerError('non-GEMM event geometry changed while hiding GEMM phases')
    for event in output_events:
        if event.get('ph') in ('X', 'C'):
            ts = finite_number(event.get('ts'))
            if ts < 0:
                raise ViewerError(f'output event has negative ts: {event.get("name")!r} ts={ts}')
            if event.get('ph') == 'X' and finite_number(event.get('dur', 0)) < 0:
                raise ViewerError(f'output interval has negative duration: {event.get("name")!r}')

    viewer = {
        'schema': SCHEMA,
        'mode': 'hide_gemm_child_phases',
        'rule': ('hide only ABI phase events whose ancestor is a GEMM task; preserve GEMM '
                 'task parents and every non-GEMM event exactly'),
        'source': {
            'path': str(input_path),
            'bytes': input_path.stat().st_size,
            'sha256': sha256_file(input_path),
            'abi_path': str(abi_path),
            'abi_sha256': sha256_file(abi_path),
            'decoded_events': [
                {'path': str(path), 'bytes': path.stat().st_size,
                 'sha256': sha256_file(path)} for path in event_paths
            ],
        },
        'gemm_task_ids': sorted(gemm_task_ids),
        'hidden_gemm_phase_ids': sorted(hidden_phase_ids),
        'decoded_event_counts_by_task': {
            str(task_id): int(source_counts.get(task_id, 0)) for task_id in sorted(tasks)
        },
        'detailed_event_counts_by_task': {
            str(task_id): int(detailed_counts.get(task_id, 0)) for task_id in sorted(tasks)
        },
        'emitted_event_counts_by_task': {
            str(task_id): int(emitted_counts.get(task_id, 0)) for task_id in sorted(tasks)
        },
        'hidden_event_counts_by_task': {
            str(task_id): int(hidden_counts.get(task_id, 0)) for task_id in sorted(tasks)
        },
        'derived_event_count': 0,
        'sampling_policy': 'none',
        'no_sampling': True,
        'no_stride': True,
        'non_gemm_geometry_sha256': geometry_digest(source_non_gemm),
        'hardware_counters': {
            'event_count': len(counters),
            'geometry_sha256': counter_digest(counters),
            'names': sorted({str(event.get('name')) for event in counters}),
        },
    }
    output = copy.deepcopy(trace)
    output['traceEvents'] = output_events
    other = copy.deepcopy(trace.get('otherData')
                          if isinstance(trace.get('otherData'), dict) else {})
    other['semantic_viewer'] = viewer
    output['otherData'] = other
    return output


def _attach_fallback_metadata(output: Dict[str, Any], phase_hidden: bool,
                              sm_selection: Dict[str, Any], history: List[Dict[str, Any]]) -> None:
    semantic = output.setdefault('otherData', {}).setdefault('semantic_viewer', {})
    semantic['automatic_size_fallback'] = True
    semantic['mode'] = 'hide_all_phase_layers' if phase_hidden else 'full_detail'
    semantic['phase_hidden'] = phase_hidden
    if phase_hidden:
        phase_hiding = semantic.get('phase_hiding')
        if isinstance(phase_hiding, dict):
            semantic['hidden_phase_task_ids'] = list(
                phase_hiding.get('hidden_phase_task_ids', []))
    capture_scope = output.get('otherData', {}).get('capture_scope', {})
    semantic['source_capture_was_all_sm'] = capture_scope.get('source_capture_was_all_sm')
    semantic['size_fallback_rounds'] = copy.deepcopy(history)
    halving_rounds = [record for record in history
                      if record.get('stage') == 'halve_physical_sm_set']
    if halving_rounds:
        raise ViewerError('physical-SM halving history is forbidden')
    if not halving_rounds:
        semantic.pop('sm_display_filter', None)
        semantic.pop('sm_display_filter_is_postprocess', None)
        return
    display = copy.deepcopy(sm_selection)
    display.update({
        'policy': 'representative_physical_sm_halving',
        'selection_rule': 'every_other_even_index',
        'source_physical_smids': list(display.get('original_physical_sm_ids', [])),
        'original_physical_smids': list(display.get('original_physical_sm_ids', [])),
        'displayed_physical_smids': list(display.get('display_physical_sm_ids', [])),
        'source_sm_count': len(display.get('original_physical_sm_ids', [])),
        'displayed_sm_count': len(display.get('display_physical_sm_ids', [])),
        'rounds': [{
            'round': record['round'],
            'input_physical_smids': list(record['input_physical_smids']),
            'output_physical_smids': list(record['output_physical_smids']),
            'input_count': record['input_count'],
            'output_count': record['output_count'],
            'counts': copy.deepcopy(record['counts']),
            'bytes': record['bytes'],
        } for record in halving_rounds],
    })
    display['round'] = halving_rounds[-1]['round']
    display['source_capture_was_all_sm'] = True
    semantic['sm_display_filter'] = display
    semantic['sm_display_filter_is_postprocess'] = True


def _build_fallback_candidate(trace: Dict[str, Any], tasks: Dict[int, Dict[str, Any]],
                              source_counts: Counter, input_path: Path, abi_path: Path,
                              event_paths: Sequence[Path], output_path: Path,
                              phase_hidden: bool, selected_smids: Sequence[int] | None,
                              history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
    if selected_smids is not None:
        raise ViewerError('postprocess physical-SM filtering is forbidden')
    original_smids = sorted({
        args.get('physical_smid')
        for event in trace.get('traceEvents', []) if event.get('ph') == 'X'
        for args in [event.get('args') if isinstance(event.get('args'), dict) else {}]
        if isinstance(args.get('physical_smid'), int)
    })
    display_trace = copy.deepcopy(trace)
    sm_selection: Dict[str, Any]
    if selected_smids is None:
        sm_selection = {
            'policy': 'all_physical_sms',
            'original_physical_sm_ids': original_smids,
            'display_physical_sm_ids': original_smids,
        }
    else:
        display_trace, sm_selection = filter_display_smids(trace, selected_smids)
        sm_selection['original_physical_sm_ids'] = original_smids
        sm_selection['display_physical_sm_ids'] = sorted(selected_smids)
    sm_counts = Counter()
    for event in trace.get('traceEvents', []):
        if event.get('ph') != 'X':
            continue
        args = event.get('args') if isinstance(event.get('args'), dict) else {}
        smid = args.get('physical_smid')
        if isinstance(smid, int) and not isinstance(smid, bool):
            sm_counts[smid] += 1
    sm_selection['source_event_counts_by_physical_smid'] = {
        str(smid): int(sm_counts[smid]) for smid in original_smids
    }

    candidate_tasks = copy.deepcopy(tasks)
    phase_info: Dict[str, Any] = {}
    if phase_hidden:
        display_trace, phase_info = hide_all_phase_events(display_trace, candidate_tasks)
        for task_id, task in candidate_tasks.items():
            if task.get('layer') == 'phase':
                task['viewer_policy'] = 'hidden'

    # After SM filtering, counts must be decoded from the filtered trace. The
    # original decoded counts remain authoritative only for phase intervals
    # intentionally hidden from the viewer.
    candidate_counts = trace_counts(display_trace) if selected_smids is not None else Counter(source_counts)
    if phase_hidden:
        for task_id, task in tasks.items():
            if task.get('layer') == 'phase':
                candidate_counts[task_id] = int(source_counts.get(task_id, 0))
    output = build_viewer(display_trace, candidate_tasks, candidate_counts,
                          input_path, abi_path, event_paths)
    semantic = output.setdefault('otherData', {}).setdefault('semantic_viewer', {})
    semantic['phase_hiding'] = phase_info if phase_hidden else {'policy': 'none'}
    if phase_hidden:
        # build_viewer receives the already-filtered display trace, so its
        # provisional detailed counts are zero for hidden phases.  The compact
        # exact source still contains every measured phase interval; record
        # those authoritative decoded/source counts in the provenance ledger.
        detailed_counts = semantic.setdefault('detailed_event_counts_by_task', {})
        hidden_counts = semantic.setdefault('hidden_event_counts_by_task', {})
        emitted_counts = semantic.setdefault('emitted_event_counts_by_task', {})
        derived_counts = semantic.setdefault('derived_event_counts_by_task', {})
        for task_id, task in tasks.items():
            if task.get('layer') != 'phase':
                continue
            count = int(source_counts.get(task_id, 0))
            detailed_counts[str(task_id)] = count
            hidden_counts[str(task_id)] = count
            emitted_counts[str(task_id)] = 0
            derived_counts[str(task_id)] = 0
    _attach_fallback_metadata(output, phase_hidden, sm_selection, history)
    return output, candidate_bytes(output, output_path), sm_selection


def build_size_bounded_viewer(trace: Dict[str, Any], tasks: Dict[int, Dict[str, Any]],
                              source_counts: Counter, input_path: Path, abi_path: Path,
                              event_paths: Sequence[Path], output_path: Path,
                              max_bytes: int, start_phase_hidden: bool = False) -> Tuple[Dict[str, Any], int, List[Dict[str, Any]]]:
    """Try full detail, then hide phases without ever changing SM scope."""
    original_smids = sorted({
        args.get('physical_smid')
        for event in trace.get('traceEvents', []) if event.get('ph') == 'X'
        for args in [event.get('args') if isinstance(event.get('args'), dict) else {}]
        if isinstance(args.get('physical_smid'), int)
    })
    history: List[Dict[str, Any]] = []

    def attempt(stage: str, round_no: int, phase_hidden: bool,
                selected_smids: Sequence[int] | None) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
        output, size, selection = _build_fallback_candidate(
            trace, tasks, source_counts, input_path, abi_path, event_paths,
            output_path, phase_hidden, selected_smids, history)
        displayed = (original_smids if selected_smids is None
                     else sorted(selected_smids))
        event_count = sum(1 for event in output.get('traceEvents', []) if event.get('ph') == 'X')
        counter_count = sum(1 for event in output.get('traceEvents', []) if event.get('ph') == 'C')
        record = {
            'stage': stage,
            'round': round_no,
            'original_physical_sm_ids': original_smids,
            'display_physical_sm_ids': displayed,
            'counts': {
                'trace_x_events': event_count,
                'hardware_counter_events': counter_count,
                'display_physical_sm_count': len(displayed),
            },
            'bytes': size,
        }
        if stage == 'halve_physical_sm_set':
            previous_halving = [record for record in history
                                if record.get('stage') == 'halve_physical_sm_set']
            input_ids = (previous_halving[-1]['output_physical_smids']
                         if previous_halving else original_smids)
            record['input_physical_smids'] = list(input_ids)
            record['output_physical_smids'] = list(displayed)
            record['input_count'] = len(input_ids)
            record['output_count'] = len(displayed)
        history.append(record)
        # Include this round's record in the serialized candidate and refresh
        # the byte count once so the recorded value is the actual candidate size.
        _attach_fallback_metadata(output, phase_hidden, selection, history)
        size = candidate_bytes(output, output_path)
        record['bytes'] = size
        _attach_fallback_metadata(output, phase_hidden, selection, history)
        return output, size, selection

    if start_phase_hidden:
        output, size, _ = attempt('hide_all_phase_layers', 0, True, None)
        if size <= max_bytes:
            return output, size, history
    else:
        # compact-exact input is already a complete candidate. When the
        # serialized input itself exceeds the budget, rebuilding it through
        # build_viewer and gzip would needlessly copy/compress millions of
        # events merely to discover that the phase fallback is required.
        input_size = input_path.stat().st_size
        if input_size > max_bytes:
            full_x = sum(1 for event in trace.get('traceEvents', []) if event.get('ph') == 'X')
            full_c = sum(1 for event in trace.get('traceEvents', []) if event.get('ph') == 'C')
            history.append({
                'stage': 'full_detail',
                'round': 0,
                'original_physical_sm_ids': original_smids,
                'display_physical_sm_ids': original_smids,
                'counts': {
                    'trace_x_events': full_x,
                    'hardware_counter_events': full_c,
                    'display_physical_sm_count': len(original_smids),
                },
                'bytes': input_size,
                'bytes_source': 'input_path.stat().st_size',
            })
        else:
            output, size, _ = attempt('full_detail', 0, False, None)
            if size <= max_bytes:
                return output, size, history

    if not start_phase_hidden:
        output, size, _ = attempt('hide_all_phase_layers', 0, True, None)
        if size <= max_bytes:
            return output, size, history

    raise ViewerError(
        f'viewer={size} bytes exceeds max_bytes={max_bytes} after hiding phases; '
        'postprocess SM filtering is forbidden; rerun from source with a user-selected physical-SM set')


def write_output(path: Path, trace: Dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {'ensure_ascii': False, 'separators': (',', ':')}
    if path.name.endswith('.gz'):
        with gzip.open(path, 'wt', encoding='utf-8', compresslevel=9) as handle:
            json.dump(trace, handle, **kwargs)
    else:
        with path.open('w', encoding='utf-8') as handle:
            json.dump(trace, handle, **kwargs)
    return path.stat().st_size


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description='根据目标 ABI 可见性策略生成不抽样的精简 Perfetto viewer',
        allow_abbrev=False)
    result.add_argument('--input', required=True, help='compact-exact Perfetto JSON 或 JSON.gz')
    result.add_argument('--events', nargs='+', required=True, help='每个 rank 的 decoded events JSON')
    result.add_argument('--abi', required=True, help='声明 tasks 与 viewer_policy 的目标 ABI JSON')
    result.add_argument('--output', required=True, help='输出 Perfetto JSON 或 JSON.gz')
    result.add_argument('--max-bytes', type=int, default=10_000_000)
    result.add_argument('--capture-plan', required=True,
                        help='第 0 步强制弹窗后生成的 capture plan JSON')
    result.add_argument('--hide-gemm-phases', action='store_true',
                        help='第一层精简：仅隐藏 GEMM task 的内部 phase；其他事件完整保留')
    result.add_argument('--hide-phases', action='store_true',
                        help='隐藏全部 ABI layer=phase；不改变第 0 步选择的 physical SM 范围')
    return result


def main() -> int:
    args = parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    event_paths = [Path(value).expanduser().resolve() for value in args.events]
    abi_path = Path(args.abi).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    for path in [input_path, abi_path, *event_paths]:
        if not path.is_file():
            print(f'ERROR 输入文件不存在: {path}', file=sys.stderr)
            return 3
    try:
        capture_plan = CAPTURE_SCOPE.load_plan(Path(args.capture_plan).expanduser().resolve())
        tasks, _abi = parse_abi(abi_path)
        counts = decoded_counts(event_paths)
        trace = load_json(input_path)
        if not isinstance(trace, dict):
            raise ViewerError('输入 trace 顶层必须是 object')
        CAPTURE_SCOPE.validate_trace_object(capture_plan, trace, input_path)
        trace = canonicalize_event_semantics(trace, tasks)
        if args.hide_gemm_phases:
            output = build_hide_gemm_phases_viewer(
                trace, tasks, counts, input_path, abi_path, event_paths)
            size = candidate_bytes(output, output_path)
        else:
            output, size, _history = build_size_bounded_viewer(
                trace, tasks, counts, input_path, abi_path, event_paths,
                output_path, args.max_bytes, start_phase_hidden=args.hide_phases)
        if args.hide_phases:
            semantic = output.setdefault('otherData', {}).setdefault('semantic_viewer', {})
            semantic['automatic_size_fallback'] = False
            semantic['explicit_task_only_request'] = True
        size = write_output(output_path, output)
    except (ViewerError, CAPTURE_SCOPE.CaptureScopeError) as error:
        print(f'ERROR {error}', file=sys.stderr)
        return 3
    if size > args.max_bytes:
        print(f'ERROR viewer={size} bytes 超过 max_bytes={args.max_bytes}', file=sys.stderr)
        return 4
    print(f'[semantic-viewer] OK output={output_path} bytes={size} '
          f'tasks={len(tasks)} source_events={sum(counts.values())}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
