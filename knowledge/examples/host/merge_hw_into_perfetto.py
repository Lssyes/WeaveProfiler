#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HW_NOTE = ('counter 为 GPU 设备级 (device-wide) PM 采样，连续覆盖所选 kernel 全窗口；'
           'Perfetto 连线不表示持续非零，也不可归因到某个 CTA、warp 或单一 phase。'
           'NVLink 可同时来自 Dispatch、Scatter/Combine 及同卡其它流量；'
           '泳道按渲染契约 R6 置于软件轨道之上')


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
                        help='输出 .json.gz;若指向目录且有 run_tag(参数给或从上游 trace 继承),'
                             '自动拼 trace__<tag>__sw_hw.perfetto.json.gz')
    parser.add_argument('--run-tag', type=str, default=None,
                        help='run 标签(如 mega_moe_bs32_topk8_ep1),写入 otherData.run_tag,'
                             '并用作默认文件名后缀;未给则沿用上游 trace 的 run_tag')
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
    }
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
        tag = effective_tag or 'run'
        out_path = out_path / f'trace__{tag}__sw_hw.perfetto.json.gz'

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
