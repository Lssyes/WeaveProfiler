#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hardware_utilization import (
    document_communication_scope, document_platform, evaluate_hardware_utilization,
    filter_utilization_evidence_to_window,
    filter_counters_to_window)

HW_NOTE = ('counter 为 GPU 设备级 (device-wide) PM 采样，连续覆盖所选 kernel 全窗口；'
           'Perfetto 连线不表示持续非零，也不可归因到某个 CTA、warp 或单一 phase。'
           'NVLink 可同时来自 Dispatch、Scatter/Combine 及同卡其它流量；'
           '泳道按渲染契约 R6 置于软件轨道之上')


# SPEC-1: the merged artifact inherits every identity field from the canonical
# software trace and changes only the bundle token to sw_hw.
CARD_TARGETS = {
    'h20': {'platform': 'nv', 'arches': frozenset(('sm_90',))},
    'b300': {'platform': 'nv', 'arches': frozenset(('sm_100a', 'sm_103a'))},
    'mi308x': {'platform': 'amd', 'arches': frozenset(('gfx942',))},
}
TRACE_FILENAME_RE = re.compile(
    r'^trace__(?P<model>[a-z0-9]+)_(?P<stage>[a-z0-9]+)'
    r'__card-(?P<card>h20|b300|mi308x)'
    r'__bs(?P<batch>\d+)_sq(?P<seqlen_q>\d+)(?:_kv(?P<seqlen_kv>\d+))?'
    r'__(?P<dtype>[a-z0-9]+)'
    r'__grid(?P<grid>\d+)_warp(?P<warp>\d+)_ph(?P<phases>\d+)'
    r'__(?P<bundle>sw|hw|sw_hw)\.perfetto\.json\.gz$')


def build_merged_trace_filename(trace_path: Path, *, input_bundle: str = 'sw') -> str:
    """Derive the sw_hw delivery name from a canonical SW or SW+HW trace."""
    match = TRACE_FILENAME_RE.match(Path(trace_path).name)
    if match is None:
        raise ValueError(
            'HW merge 输入 trace 文件名不符合 SPEC-1；必须包含 '
            'card-h20|b300|mi308x 与完整 shape/bundle 字段: {}'.format(
                Path(trace_path).name))
    actual_bundle = match.group('bundle')
    if input_bundle not in ('sw', 'sw_hw'):
        raise ValueError('input_bundle 必须是 sw 或 sw_hw')
    if actual_bundle != input_bundle:
        raise ValueError(
            'HW merge 输入必须是 __{}.perfetto.json.gz，实际是 __{}'.format(
                input_bundle, actual_bundle))
    name = Path(trace_path).name
    if input_bundle == 'sw_hw':
        merged = name
    else:
        merged = name[:-len('__sw.perfetto.json.gz')] + '__sw_hw.perfetto.json.gz'
    check = TRACE_FILENAME_RE.match(merged)
    if check is None or check.group('card') != match.group('card'):
        raise ValueError('HW merge 未能保留规范卡型文件名: {}'.format(merged))
    return merged


def validate_embedded_target(trace: Dict[str, Any], card: str) -> None:
    """If the SW trace recorded target metadata, it must agree with the filename."""
    other = trace.get('otherData') if isinstance(trace, dict) else None
    target = other.get('target') if isinstance(other, dict) else None
    if target is None:
        return
    if not isinstance(target, dict):
        raise ValueError('trace.otherData.target 必须是 object')
    raw_card = target.get('card')
    target_card = raw_card.strip().lower() if isinstance(raw_card, str) else None
    if target_card != card:
        raise ValueError(
            'trace 文件名 card={} 与 otherData.target.card={!r} 不符'.format(
                card, raw_card))
    raw_platform = target.get('platform')
    platform = raw_platform.strip().lower() if isinstance(raw_platform, str) else None
    platform = {'nvidia': 'nv', 'cuda': 'nv', 'rocm': 'amd', 'hip': 'amd'}.get(
        platform, platform)
    raw_arch = target.get('arch')
    arch = raw_arch.strip().lower() if isinstance(raw_arch, str) else None
    expected = CARD_TARGETS[card]
    if platform != expected['platform']:
        raise ValueError(
            'card {} 必须对应 platform={}，trace target.platform={!r}'.format(
                card, expected['platform'], raw_platform))
    if arch not in expected['arches']:
        raise ValueError(
            'card {} 只接受 arch={}，trace target.arch={!r}'.format(
                card, '|'.join(sorted(expected['arches'])), raw_arch))


def validate_merge_identity(trace_path: Path, trace: Dict[str, Any],
                            output_name: Optional[str] = None, *,
                            input_bundle: str = 'sw') -> str:
    """Validate the canonical identity of a merge input/output pair.

    The input must be a canonical SW trace and its embedded target (when
    present) must agree with the input card.  A canonical explicit output must
    inherit the same card and use the sw_hw bundle.  Non-canonical staging
    names remain available for exploratory work, but they cannot be used to
    turn a legacy no-card input into a delivery-looking output.
    """
    merged_name = build_merged_trace_filename(Path(trace_path), input_bundle=input_bundle)
    input_match = TRACE_FILENAME_RE.match(merged_name)
    if input_match is None:
        raise ValueError(
            'HW merge failed to preserve canonical input identity: {}'.format(
                merged_name))
    input_card = input_match.group('card')
    validate_embedded_target(trace, input_card)

    if output_name is not None and Path(output_name).name.startswith('trace__'):
        output_match = TRACE_FILENAME_RE.match(Path(output_name).name)
        if output_match is None:
            raise ValueError(
                '显式 trace 文件名不符合 SPEC-1（必须包含 card-h20|b300|mi308x）: {}'
                .format(Path(output_name).name))
        if output_match.group('bundle') != 'sw_hw':
            raise ValueError(
                'HW merge 输出必须使用 __sw_hw.perfetto.json.gz，实际是 __{}'
                .format(output_match.group('bundle')))
        if output_match.group('card') != input_card:
            raise ValueError(
                'HW merge 输出 card={} 必须继承输入 SW trace card={}'
                .format(output_match.group('card'), input_card))
    return merged_name


def read_json_maybe_gz(path: Path) -> Any:
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding='utf-8'))


def write_json_gz(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        json.dump(obj, handle, ensure_ascii=False)


def timestamp_report(events: List[Dict[str, Any]], sample_limit: int = 4) -> Dict[str, Any]:
    values: List[float] = []
    invalid_samples: List[Dict[str, Any]] = []
    negative_samples: List[Dict[str, Any]] = []
    invalid_count = 0
    negative_count = 0
    for index, event in enumerate(events):
        if 'ts' not in event:
            continue
        value = event.get('ts')
        sample = {'index': index, 'ph': event.get('ph'), 'name': event.get('name'),
                  'pid': event.get('pid'), 'tid': event.get('tid'), 'ts': value}
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            invalid_count += 1
            if len(invalid_samples) < sample_limit:
                invalid_samples.append(sample)
            continue
        numeric = float(value)
        values.append(numeric)
        if numeric < 0.0:
            negative_count += 1
            if len(negative_samples) < sample_limit:
                negative_samples.append(sample)
    return {
        'timed_event_count': len(values),
        'min_ts_us': min(values) if values else None,
        'max_ts_us': max(values) if values else None,
        'invalid_count': invalid_count,
        'negative_count': negative_count,
        'invalid_samples': invalid_samples,
        'negative_samples': negative_samples,
    }


def rebase_timeline_nonnegative(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Translate the complete display timeline once; never clamp/drop individual events."""
    before = timestamp_report(events)
    if before['invalid_count']:
        raise ValueError(f"invalid timestamps before rebase: {before['invalid_samples']}")
    pre_min = before['min_ts_us']
    offset_us = round(max(0.0, -float(pre_min)), 9) if pre_min is not None else 0.0
    shifted = 0
    if offset_us:
        for event in events:
            value = event.get('ts')
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                translated = round(float(value) + offset_us, 9)
                event['ts'] = 0.0 if translated == 0.0 else translated
                shifted += 1
    after = timestamp_report(events)
    return {
        'schema': 'timeline-rebase/v1',
        'applied': bool(offset_us),
        'method': ('single global translation of every numeric ts; no clamp/drop/rescale; '
                   'serialization normalized to 1e-9 us'),
        'reason': ('HW PM samples may legally precede entry_release software zero; '
                   'Perfetto requires every displayed timestamp >= 0'),
        'offset_us': offset_us,
        'pre_min_ts_us': pre_min,
        'post_min_ts_us': after['min_ts_us'],
        'shifted_event_count': shifted,
        'relative_timing_preserved': True,
        'duration_fields_unchanged': True,
        'post_negative_count': after['negative_count'],
        'post_invalid_count': after['invalid_count'],
    }


def software_window(events_doc: Any) -> Tuple[int, int, int]:
    if isinstance(events_doc, dict) and 'events' in events_doc:
        events = events_doc['events']
    elif isinstance(events_doc, list):
        events = events_doc
    else:
        raise SystemExit('[fatal] --events 结构无法识别（既不是 {events:[...]} 也不是 [...]）')
    starts = [int(e['start_ns']) for e in events if 'start_ns' in e]
    ends = [int(e['end_ns']) for e in events if 'end_ns' in e]
    if not starts or not ends:
        raise SystemExit('[fatal] --events 里没有 start_ns/end_ns 字段')
    return min(starts), max(ends), len(events)


def events_identity(events_doc: Any) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(events_doc, dict):
        return None, None
    meta = events_doc.get('metadata')
    if not isinstance(meta, dict):
        meta = {}
    rank = meta.get('rank')
    device = meta.get('device')
    if rank is None:
        rank = events_doc.get('rank')
    if device is None:
        device = events_doc.get('device')
    try:
        rank = None if rank is None else int(rank)
    except (TypeError, ValueError):
        rank = None
    try:
        device = None if device is None else int(device)
    except (TypeError, ValueError):
        device = None
    return rank, device


def select_events_doc(docs: List[Any], hw_device: Optional[int],
                      requested_rank: Optional[int]) -> Tuple[int, Optional[int], Optional[int]]:
    identities = [events_identity(doc) for doc in docs]
    if requested_rank is not None:
        matches = [i for i, (rank, _device) in enumerate(identities)
                   if rank == requested_rank]
        if not matches:
            raise SystemExit(f'[fatal] --events-rank {requested_rank} 未匹配任何 --events；'
                             f'实际 identities={identities}')
        if len(matches) > 1 and hw_device is not None:
            device_matches = [i for i in matches if identities[i][1] == hw_device]
            if len(device_matches) == 1:
                matches = device_matches
        if len(matches) != 1:
            raise SystemExit(f'[fatal] --events-rank {requested_rank} 匹配 {len(matches)} 个文件；'
                             '请只保留唯一对应 rank/device 的 events')
        idx = matches[0]
        return idx, identities[idx][0], identities[idx][1]
    if len(docs) == 1:
        return 0, identities[0][0], identities[0][1]
    if hw_device is not None:
        matches = [i for i, (_rank, device) in enumerate(identities)
                   if device == int(hw_device)]
        if len(matches) == 1:
            idx = matches[0]
            return idx, identities[idx][0], identities[idx][1]
    raise SystemExit('[fatal] 多个 --events 无法唯一匹配 HW device；'
                     f'identities={identities}，请显式给 --events-rank')


def shift_from_by_rank(container: Dict[str, Any],
                       rank: Optional[int]) -> Optional[int]:
    by_rank = container.get('shift_ns_by_rank')
    if by_rank is None:
        return None
    value: Any = None
    if isinstance(by_rank, dict):
        if rank is not None:
            value = by_rank.get(str(rank), by_rank.get(rank))
        elif len(by_rank) == 1:
            value = next(iter(by_rank.values()))
    elif isinstance(by_rank, list):
        if by_rank and all(isinstance(item, dict) for item in by_rank):
            matches = [item.get('shift_ns') for item in by_rank
                       if rank is not None and item.get('rank') is not None
                       and int(item['rank']) == rank]
            if len(matches) == 1:
                value = matches[0]
        elif rank is not None and 0 <= rank < len(by_rank):
            value = by_rank[rank]
        elif len(by_rank) == 1:
            value = by_rank[0]
    if value is None:
        return None
    return int(value)


def resolve_shift_ns(trace: Dict[str, Any], override: Optional[int],
                     rank: Optional[int]) -> Tuple[int, str]:
    if override is not None:
        return override, 'cli-override'
    other = trace.get('otherData') or {}
    if other.get('shift_ns') is not None:
        return int(other['shift_ns']), 'trace.otherData.shift_ns'
    by_rank = shift_from_by_rank(other, rank)
    if by_rank is not None:
        return by_rank, f'trace.otherData.shift_ns_by_rank[{rank}]'
    summary = other.get('summary') or {}
    if summary.get('shift_ns') is not None:
        return int(summary['shift_ns']), 'trace.otherData.summary.shift_ns'
    by_rank = shift_from_by_rank(summary, rank)
    if by_rank is not None:
        return by_rank, f'trace.otherData.summary.shift_ns_by_rank[{rank}]'
    raise SystemExit('[fatal] 软件 trace 找不到所选 rank 的 shift_ns，无法把 counter 放到同一时间轴；'
                     f'selected_rank={rank}。可用 --events-rank/--shift-ns 显式指定')


def collect_pids(events: List[Dict[str, Any]]) -> List[int]:
    pids = set()
    for ev in events:
        if 'pid' in ev:
            pids.add(ev['pid'])
    return sorted(pids)


def count_samples_in_window(sample_ts_sorted: List[int], start: int, end: int) -> int:
    lo = bisect.bisect_left(sample_ts_sorted, start)
    hi = bisect.bisect_right(sample_ts_sorted, end)
    return hi - lo


def choose_launch(launches: List[Dict[str, Any]], ordinal: Optional[int],
                  select: Optional[str], sw_span: int,
                  sample_ts_sorted: List[int]) -> Tuple[int, Dict[str, Any]]:
    if ordinal is not None:
        if not (0 <= ordinal < len(launches)):
            raise SystemExit(f'[fatal] --launch-ordinal {ordinal} 越界（共 {len(launches)} 次 launch，'
                             f'匹配到 0 个）')
        return ordinal, launches[ordinal]
    if select == 'most-samples':
        counts = [count_samples_in_window(sample_ts_sorted, lc['start_ns'], lc['end_ns'])
                  for lc in launches]
        best = max(counts) if counts else 0
        cands = [i for i, c in enumerate(counts) if c == best]
        # 稀疏 kernel + 高频采样时并列必然发生（例如全 0），绝不 fatal：
        # 采样点最多者优先；并列时取覆盖率最高（同样采样点数下 span 最短即密度最高），
        # 再并列取最早的一次，保证确定性。
        chosen = min(cands, key=lambda i: (launches[i]['end_ns'] - launches[i]['start_ns'], i))
        return chosen, launches[chosen]
    if select == 'longest':
        durs = [lc['end_ns'] - lc['start_ns'] for lc in launches]
        best = max(durs)
        cands = [i for i, d in enumerate(durs) if d == best]
        if len(cands) != 1:
            raise SystemExit(f'[fatal] --launch-select longest 命中并列 {len(cands)} 个，'
                             f'不唯一，请用 --launch-ordinal 明确指定')
        return cands[0], launches[cands[0]]
    if select == 'closest-span':
        diffs = [abs((lc['end_ns'] - lc['start_ns']) - sw_span) for lc in launches]
        best = min(diffs)
        cands = [i for i, d in enumerate(diffs) if d == best]
        if len(cands) != 1:
            raise SystemExit(f'[fatal] --launch-select closest-span 命中并列 {len(cands)} 个，'
                             f'不唯一，请用 --launch-ordinal 明确指定')
        return cands[0], launches[cands[0]]
    if len(launches) == 1:
        return 0, launches[0]
    raise SystemExit(f'[fatal] 目标 kernel 匹配到 {len(launches)} 次 launch（多个），'
                     f'不许随便取第一个；请用 --launch-ordinal N 或 '
                     f'--launch-select most-samples|longest|closest-span 唯一定位')


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='把 GPU 硬件 counter 后处理合并进已有软件 Perfetto trace')
    parser.add_argument('--trace', required=True, help='已有软件 trace（.json 或 .json.gz）')
    parser.add_argument('--hw-metrics', required=True, help='extract_gpu_metrics.py 产出的 hw_metrics.json')
    parser.add_argument('--events', required=True, action='append',
                        help='软件锚点 events_rank*.json（可重复；用于选择与 HW device/rank 对应的原始时间窗）')
    parser.add_argument('--events-rank', type=int, default=None,
                        help='多个 --events 时显式选择用于 HW 对齐的软件 rank；'
                             '不指定时仅在 metadata.device 唯一匹配 HW device 才自动选择')
    parser.add_argument('--out', required=True,
                        help='输出 .json.gz; 若指向目录，必须从规范 SW trace 文件名继承 '
                             'model/stage/card/shape/dtype/grid，只把 bundle 改为 sw_hw')
    parser.add_argument('--run-tag', type=str, default=None,
                        help='仅写入 otherData.run_tag；不参与交付文件名拼接')
    parser.add_argument('--launch-ordinal', type=int, default=None,
                        help='选第几次目标 kernel launch（0-based）做对齐')
    parser.add_argument('--launch-select',
                        choices=('most-samples', 'longest', 'closest-span'),
                        default='most-samples',
                        help='当有多次 launch 时的唯一选择策略；默认 most-samples：'
                             '遍历所有匹配 launch，选时间窗内 GPU metrics 采样点最多的一次。'
                             '稀疏 kernel + 高频 PM 采样存在周期性约 1s 的采样 gap，'
                             '按时长/最接近选常落在 gap 里导致 counter 全 0，most-samples 可避开')
    parser.add_argument('--shift-ns', type=int, default=None,
                        help='覆盖软件 trace 的 shift_ns（默认从 trace.otherData 读）')
    parser.add_argument('--align-warn-frac', type=float, default=0.10)
    parser.add_argument('--min-window-samples', type=int, default=100,
                        help='选中 launch 窗口内采样点的绝对下限（默认 100，只作极端兜底）。'
                             '短 kernel 的采样点上限 = kernel 时长 × 采样频率，'
                             '如 2.25 ms @ 200 kHz 上限只有 450 点，所以不能用绝对数量当主判据。')
    parser.add_argument('--min-window-coverage', type=float, default=0.8,
                        help='选中 launch 窗口内采样覆盖率（实际 / 期望）的下限，默认 0.8。'
                             '这是判断「有没有落进 nsys 采样 gap」的主判据。')
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    trace_path = Path(args.trace).expanduser().resolve()
    hw_path = Path(args.hw_metrics).expanduser().resolve()
    events_paths = [Path(value).expanduser().resolve() for value in args.events]
    out_path = Path(args.out).expanduser().resolve()
    inputs = [('--trace', trace_path), ('--hw-metrics', hw_path)]
    inputs.extend(('--events', path) for path in events_paths)
    for label, path in inputs:
        if not path.is_file():
            print(f'[fatal] {label} 不存在: {path}')
            return 1

    trace = read_json_maybe_gz(trace_path)
    if not isinstance(trace, dict) or 'traceEvents' not in trace:
        print('[fatal] --trace 不是 {traceEvents:[...]} 结构')
        return 1
    events_list: List[Dict[str, Any]] = trace['traceEvents']
    x_before = sum(1 for e in events_list if e.get('ph') == 'X')
    c_before = sum(1 for e in events_list if e.get('ph') == 'C')
    input_ts = timestamp_report(events_list)
    if input_ts['invalid_count'] or input_ts['negative_count']:
        print('[fatal] 输入 SW trace 已含非法/负显示时间戳；merge 不得替上游掩盖：'
              f'invalid={input_ts["invalid_count"]} negative={input_ts["negative_count"]} '
              f'samples={input_ts["invalid_samples"] + input_ts["negative_samples"]}')
        return 4

    hw = read_json_maybe_gz(hw_path)
    launches = hw.get('kernel_launches') or []
    counters = hw.get('counters') or []
    if not launches:
        print('[fatal] hw_metrics 里没有 kernel_launches，无法对齐')
        return 3
    device = hw.get('device')

    events_docs = [read_json_maybe_gz(path) for path in events_paths]
    selected_idx, events_rank, events_device = select_events_doc(
        events_docs, device, args.events_rank)
    events_path = events_paths[selected_idx]
    events_doc = events_docs[selected_idx]
    sw_start, sw_end, sw_event_count = software_window(events_doc)
    sw_span = sw_end - sw_start
    if sw_span <= 0:
        print(f'[fatal] 软件窗口非法 span={sw_span}')
        return 1

    shift_ns, shift_source = resolve_shift_ns(trace, args.shift_ns, events_rank)

    sample_ts_sorted = sorted({int(ts) for counter in counters
                               for ts, _value in counter.get('samples', [])})
    interval_info = hw.get('sampling_interval_ns') or {}
    interval_ns: Optional[float] = None
    if interval_info.get('available') and interval_info.get('median'):
        interval_ns = float(interval_info['median'])
    elif hw.get('frequency_hz'):
        interval_ns = 1e9 / float(hw['frequency_hz'])

    launch_sample_counts = [count_samples_in_window(sample_ts_sorted, lc['start_ns'], lc['end_ns'])
                            for lc in launches]
    launches_with_100 = sum(1 for count in launch_sample_counts if count >= 100)

    chosen_idx, chosen = choose_launch(launches, args.launch_ordinal, args.launch_select, sw_span,
                                       sample_ts_sorted)
    nsys_start = chosen['start_ns']
    nsys_end = chosen['end_ns']
    nsys_span = nsys_end - nsys_start
    if nsys_span <= 0:
        print(f'[fatal] 选定 launch 的 nsys span 非法 {nsys_span}')
        return 1

    chosen_samples = count_samples_in_window(sample_ts_sorted, nsys_start, nsys_end)
    expected_samples = (nsys_span / interval_ns) if interval_ns else None
    coverage = (chosen_samples / expected_samples) if expected_samples else None

    # nsys_span 必然略大于 sw_span：nsys 量的是 kernel 完整执行时长，
    # 而软件跨度是首个打点到末个打点，两头各短一截（序言与收尾）。
    # 这个差是结构性的，不是误差，所以只做平移、绝不按两端缩放 ——
    # 缩放会篡改 counter 采样点之间的真实间隔。
    abs_err = abs(sw_span - nsys_span)
    rel_err = abs_err / sw_span
    align_warn = rel_err > args.align_warn_frac

    def map_ts_us(ns: int) -> float:
        return (sw_start + (ns - nsys_start) - shift_ns) / 1000.0

    sw_pids = collect_pids(events_list)
    min_sw_pid = min(sw_pids) if sw_pids else 0
    pid_shift = 0
    hw_pid = min_sw_pid - 1
    if hw_pid < 0:
        pid_shift = 1 - min_sw_pid
        for ev in events_list:
            if 'pid' in ev:
                ev['pid'] += pid_shift
        # scheduler_claim_audit addresses the same displayed (pid, tid) lanes as
        # task slices.  Keep that sidecar in lock-step with the software PID
        # translation used to reserve pid=0 for device-wide HW counters.
        other_for_pid = trace.get('otherData')
        scheduler_audit = (other_for_pid.get('scheduler_claim_audit')
                           if isinstance(other_for_pid, dict) else None)
        if isinstance(scheduler_audit, list):
            for item in scheduler_audit:
                if isinstance(item, dict) and isinstance(item.get('pid'), int):
                    item['pid'] += pid_shift
        hw_pid = 0
        sw_pids = collect_pids(events_list)
        min_sw_pid = min(sw_pids)
    hw_sort_index = -1

    counter_events: List[Dict[str, Any]] = []
    per_counter_report = []
    for counter in counters:
        name = counter.get('name', 'metric')
        unit = counter.get('unit', '')
        track_name = f'{name} [{unit}]' if unit else name
        kept = 0
        for ts_ns, value in counter.get('samples', []):
            if nsys_start <= ts_ns <= nsys_end:
                counter_events.append({
                    'ph': 'C', 'name': track_name, 'ts': map_ts_us(ts_ns),
                    'pid': hw_pid, 'tid': 0, 'args': {'value': value},
                })
                kept += 1
        per_counter_report.append({'name': name, 'unit': unit,
                                   'samples_in_launch_window': kept,
                                   'samples_available': len(counter.get('samples', []))})

    window_counters = filter_counters_to_window(counters, nsys_start, nsys_end)
    hw_platform = document_platform(hw)
    hardware_summary, canonical_series = evaluate_hardware_utilization(
        window_counters, platform=hw_platform,
        communication_scope=document_communication_scope(hw),
        evidence_document=filter_utilization_evidence_to_window(
            hw, nsys_start, nsys_end),
        source_prefix=('nsys.gpu_metrics' if hw_platform == 'nv'
                       else 'hardware.counters'),
        source_document=str(hw_path), target_card=hw.get('target_card'),
        window_ns=(nsys_start, nsys_end))
    per_canonical_report = []
    canonical_event_count = 0
    for metric_id in hardware_summary['required_metric_ids']:
        metric_series = canonical_series.get(metric_id)
        kept = 0
        if metric_series:
            for ts_ns, value in metric_series['samples']:
                if nsys_start <= ts_ns <= nsys_end:
                    counter_events.append({
                        'ph': 'C', 'name': metric_id, 'ts': map_ts_us(ts_ns),
                        'pid': hw_pid, 'tid': 0,
                        'cat': 'hardware_utilization',
                        # Chrome JSON counter events accept only a numeric
                        # ``value`` here.  Extra string/bool fields make
                        # Perfetto report one json_parser_failure per field.
                        # Provenance remains in otherData.hardware_summary and
                        # otherData.hw_merge.metrics.
                        'args': {'value': value},
                    })
                    kept += 1
        canonical_event_count += kept
        per_canonical_report.append({
            'semantic_id': metric_id,
            'value': hardware_summary['metrics'][metric_id]['value'],
            'evidence_status': hardware_summary['metrics'][metric_id]['evidence_status'],
            'source': hardware_summary['metrics'][metric_id]['source'],
            'samples_in_launch_window': kept,
        })

    hw_meta = [
        {'name': 'process_name', 'ph': 'M', 'pid': hw_pid, 'tid': 0,
         'args': {'name': f'HW dev{device} (device-wide, full-kernel sampled)'}},
        {'name': 'process_sort_index', 'ph': 'M', 'pid': hw_pid, 'tid': 0,
         'args': {'sort_index': hw_sort_index}},
        {'name': 'process_labels', 'ph': 'M', 'pid': hw_pid, 'tid': 0,
         'args': {'labels': HW_NOTE}},
    ]

    trace['traceEvents'] = events_list + hw_meta + counter_events
    try:
        timeline_rebase = rebase_timeline_nonnegative(trace['traceEvents'])
    except ValueError as error:
        print(f'[fatal] timeline rebase 拒绝非法时间戳: {error}')
        return 4
    final_ts = timestamp_report(trace['traceEvents'])
    x_after = sum(1 for e in trace['traceEvents'] if e.get('ph') == 'X')
    c_after = sum(1 for e in trace['traceEvents'] if e.get('ph') == 'C')

    other = trace.get('otherData')
    if not isinstance(other, dict):
        other = {}
    other['hw_merge'] = {
        'schema': 'hw-merge/v1',
        'note': HW_NOTE,
        'metrics_set': hw.get('metrics_set'),
        'frequency_hz': hw.get('frequency_hz'),
        'device': device,
        'hardware_coverage_scope': 'single device-wide counter set; not per-rank',
        'sampling_scope': 'continuous PM samples across the selected full-kernel launch window',
        'phase_attribution': 'none; counters are not exclusive to Dispatch or any software phase',
        'sample_value_policy': 'raw extracted PM samples; zeros preserved; no interpolation or forward-fill',
        'known_nvlink_contributors': [
            'Dispatch route/pull and NVLink barriers',
            'GEMM1 rank-mapped Scatter when dst_rank is remote',
            'other concurrent activity on the same device cannot be excluded',
        ],
        'timeline_rebase': timeline_rebase,
        'events_sources': [str(path) for path in events_paths],
        'selected_events_source': str(events_path),
        'aligned_software_rank': events_rank,
        'aligned_software_device': events_device,
        'hw_counter_pid': hw_pid,
        'hw_process_sort_index': hw_sort_index,
        'software_pids': sw_pids,
        'software_pid_shift_applied': pid_shift,
        'r6_hw_pid_lt_all_software': hw_pid < min_sw_pid,
        'shift_ns': shift_ns,
        'shift_ns_source': shift_source,
        'alignment': {
            'formula': ('pre_rebase_us = (sw_start + (nsys_ts - nsys_start) - shift_ns) / 1000; '
                        'final_ts_us = pre_rebase_us + timeline_rebase.offset_us'),
            'sw_window_ns': [sw_start, sw_end],
            'sw_span_ns': sw_span,
            'sw_event_count': sw_event_count,
            'nsys_window_ns': [nsys_start, nsys_end],
            'nsys_span_ns': nsys_span,
            'alignment_mode': 'shift-only (no scaling; nsys_span > sw_span is structural)',
            'abs_error_ns': abs_err,
            'relative_error': rel_err,
            'align_warn_frac': args.align_warn_frac,
            'align_warning': align_warn,
        },
        'chosen_launch': {
            'ordinal': chosen_idx,
            'total_matched_launches': len(launches),
            'launch_select': args.launch_select if args.launch_ordinal is None else 'ordinal',
            'correlation_id': chosen.get('correlation_id'),
            'device': chosen.get('device'),
            'name': chosen.get('name'),
            'start_ns': nsys_start, 'end_ns': nsys_end,
            'samples_in_window': chosen_samples,
            'expected_samples': expected_samples,
            'coverage': coverage,
            'sample_interval_ns': interval_ns,
            'launch_sample_counts': launch_sample_counts,
            'launches_with_ge_100_samples': launches_with_100,
        },
        'counters': per_counter_report,
        'canonical_metrics': per_canonical_report,
        'canonical_metric_event_count': canonical_event_count,
        'hardware_summary_schema': hardware_summary['schema'],
    }
    other['hardware_summary'] = hardware_summary
    other['timeline_rebase'] = timeline_rebase
    other['timeline_coordinate_systems'] = {
        'raw_args_ns': 'absolute device %globaltimer values; unchanged',
        'logical_software_time': 'per-rank entry-release normalization before final HW merge',
        'displayed_ts_us': 'logical software time plus timeline_rebase.offset_us',
        'summary_and_barrier_ns': 'logical software coordinates; add rebase offset for display',
    }
    trace['otherData'] = other

    effective_tag = args.run_tag or (other.get('run_tag') if isinstance(other, dict) else None)
    if effective_tag:
        other['run_tag'] = effective_tag
    if out_path.is_dir():
        try:
            merged_name = validate_merge_identity(trace_path, trace)
        except ValueError as error:
            print('[fatal] {}'.format(error))
            return 2
        out_path = out_path / merged_name
    elif out_path.name.startswith('trace__'):
        try:
            validate_merge_identity(trace_path, trace, out_path.name)
        except ValueError as error:
            print('[fatal] {}'.format(error))
            return 2

    print(f'[align] shift_ns={shift_ns} ({shift_source})')
    print(f'[align] sw_span={sw_span} ns  nsys_span={nsys_span} ns  '
          f'abs_err={abs_err} ns  rel_err={rel_err * 100:.2f}%')
    print(f'[align] chosen launch ordinal={chosen_idx}/{len(launches)} '
          f'select={args.launch_select if args.launch_ordinal is None else "ordinal"} '
          f'corr={chosen.get("correlation_id")} dev={chosen.get("device")} '
          f'dur={nsys_span} ns')
    coverage_text = 'n/a' if coverage is None else f'{coverage * 100:.1f}%'
    expected_text = 'n/a' if expected_samples is None else f'{expected_samples:.0f}'
    print(f'[align] 窗口内采样点={chosen_samples}（期望 {expected_text}，覆盖率 {coverage_text}）；'
          f'采样点>=100 的 launch {launches_with_100}/{len(launches)} 次')
    if align_warn:
        print(f'[align] WARNING 相对误差 {rel_err * 100:.2f}% 超过阈值 '
              f'{args.align_warn_frac * 100:.0f}%，可能匹配到了错误的 kernel')
    print(f'[pid] software pids={sw_pids} (min={min_sw_pid})  hw_counter_pid={hw_pid}  '
          f'pid_shift_applied={pid_shift}  R6(hw<all sw)={hw_pid < min_sw_pid}')
    print(f'[merge] ph==X {x_before} -> {x_after}（必须相等）  ph==C {c_before} -> {c_after}')
    for rep in per_counter_report:
        print(f'[merge]   {rep["name"]} [{rep["unit"]}] 窗口内样本 {rep["samples_in_launch_window"]}')
    for rep in per_canonical_report:
        print(f'[util] {rep["semantic_id"]}: value={rep["value"]} '
              f'status={rep["evidence_status"]} samples={rep["samples_in_launch_window"]} '
              f'source={rep["source"]}')
    print(f'[timeline] global_rebase_us={timeline_rebase["offset_us"]:.6f} '
          f'pre_min={timeline_rebase["pre_min_ts_us"]} '
          f'post_min={timeline_rebase["post_min_ts_us"]}')

    if x_after != x_before:
        print('[fatal] 软件 ph==X 事件数发生变化，合并破坏了软件事件！')
        return 4
    if hw_pid >= min_sw_pid:
        print('[fatal] R6 违反：counter pid 未严格小于软件 pid')
        return 4
    if coverage is not None and coverage < args.min_window_coverage:
        print(f'[fatal] 选中 launch 窗口内采样覆盖率 {coverage * 100:.1f}% < '
              f'{args.min_window_coverage * 100:.0f}%（实际 {chosen_samples} / 期望 {expected_samples:.0f}）：'
              f'多半是这次 launch 落进了 nsys 的采样 gap。'
              f'建议改用 --capture-range=cudaProfilerApi 圈定窗口重采（见 knowledge/guides/nsys_pm_sampling.md）。')
        return 4
    if chosen_samples < args.min_window_samples:
        print(f'[fatal] 选中 launch 窗口内采样点 {chosen_samples} < {args.min_window_samples}：'
              f'绝对数量过少，即使覆盖率达标也不足以做时间线分析。'
              f'注意短 kernel 的采样点上限 = kernel 时长 × 采样频率'
              f'（如 2.25 ms @ 200 kHz 上限只有 450 点），'
              f'这种情况应下调 --min-window-samples 而不是重采。')
        return 4
    if sw_span > 0 and nsys_span > 2 * sw_span:
        print(f'[fatal] hw 时间跨度 {nsys_span} ns 超过 sw 跨度 {sw_span} ns 的 2 倍：'
              f'counter 没有被裁到 kernel 窗口内(可能把 nsys 全程 counter 都塞进来了)，'
              f'图上会出现 sw 挤在开头、后面拖一长串 hw 空白。检查 launch 窗口选择。')
        return 4
    if sw_span > 0 and nsys_span < 0.5 * sw_span:
        print(f'[fatal] hw 选中 kernel 时长 {nsys_span} ns 不到 sw 跨度 {sw_span} ns 的一半：'
              f'多半是 hw 采的和 sw 不是同一个 build —— 常见于 hw 采了未插桩的 timing build '
              f'(kernel 短)而 sw 是插桩后的 build(打点把 kernel 撑长)。'
              f'两边必须用同一个插桩 build 采集。counter 数值本身高低不判(那是 kernel 真实属性)，'
              f'但时长对不上说明采错了对象。')
        return 4
    if final_ts['invalid_count'] or final_ts['negative_count']:
        print('[fatal] [timeline_negative_timestamp] final trace contains invalid/negative ts: '
              f'invalid={final_ts["invalid_count"]} negative={final_ts["negative_count"]} '
              f'samples={final_ts["invalid_samples"] + final_ts["negative_samples"]}')
        return 4

    write_json_gz(out_path, trace)
    out_bytes = out_path.stat().st_size
    print(f'[out] {out_path} bytes={out_bytes}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
