#!/usr/bin/env python3
"""把 extract_gpu_metrics.py 的硬件 counter 转成独立的 Perfetto trace。

这个文件只处理一个目标 kernel launch。hw_metrics 中有多个 launch 时，必须用
--launch-ordinal 明确指定；不会猜测第一个、最长的或样本最多的 launch。
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


TRACE_SCHEMA = 'hw-perfetto/v1'
INPUT_SCHEMA = 'gpu-hw-metrics/v1'
HW_PID = 1


# --- SPEC-1: canonical trace filename construction ------------------------
# Duplicated from build_perfetto.py on purpose: the reference host scripts are
# standalone and copied per target. This script writes the HW-only counter
# trace, so the bundle token defaults to 'hw'. 'sw+hw' maps to 'sw_hw' for the
# merged artifact naming when a caller explicitly asks for it.
BUNDLE_TOKENS = {'sw': 'sw', 'hw': 'hw', 'sw+hw': 'sw_hw', 'sw_hw': 'sw_hw',
                 'hw+sw': 'sw_hw', 'hw_sw': 'sw_hw'}
TRACE_FILENAME_RE = re.compile(
    r'^trace__[a-z0-9]+_[a-z0-9]+__bs[a-z0-9]+_sq[a-z0-9]+(?:_kv[a-z0-9]+)?'
    r'__[a-z0-9]+__grid\d+_warp\d+_ph\d+__(?:sw|hw|sw_hw)\.perfetto\.json\.gz$')


def _slug_token(value: Any) -> str:
    if value is None:
        return 'na'
    slug = re.sub(r'[^a-z0-9]', '', str(value).lower())
    return slug or 'na'


def _int_token(value: Any) -> str:
    if isinstance(value, bool):
        return 'na'
    if isinstance(value, int):
        return str(value) if value >= 0 else 'na'
    if isinstance(value, float) and float(value).is_integer() and value >= 0:
        return str(int(value))
    if isinstance(value, (list, tuple)) and value:
        product = 1
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)) \
                    or not float(item).is_integer() or item < 0:
                return 'na'
            product *= int(item)
        return str(product)
    if isinstance(value, str) and value.strip().isdigit():
        return value.strip()
    return 'na'


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _pick(mapping: Any, keys: Sequence[str]) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def contract_field(contract: Any, section: Optional[str], *keys: str) -> Any:
    if not isinstance(contract, dict):
        return None
    if section is not None:
        sub = contract.get(section)
        if isinstance(sub, dict):
            value = _pick(sub, keys)
            if value is not None:
                return value
    return _pick(contract, keys)


def resolve_bundle_token(bundle: Any) -> str:
    if bundle is None:
        return 'hw'
    text = str(bundle).strip().lower()
    return BUNDLE_TOKENS.get(text, _slug_token(text))


def build_trace_filename(contract: Any, bundle: str = 'hw', grid: Any = None,
                         warps_per_cta: Any = None, num_phases: Any = None) -> str:
    """SPEC-1 canonical trace filename. See build_perfetto.py for field docs.

    The HW builder has no decoded topology, so grid/warp/phase come from the
    contract (runtime_launch or launch or top-level); the explicit args are only
    a fallback."""
    model = _slug_token(contract_field(contract, 'target', 'model', 'model_name', 'name'))
    stage = _slug_token(contract_field(contract, 'target', 'stage', 'phase', 'mode'))
    batch = _slug_token(contract_field(contract, 'input', 'batch', 'batch_size', 'bs'))
    seqlen_q = _slug_token(contract_field(contract, 'input',
                                          'seqlen_q', 'seq_len_q', 'seqlen', 'sq'))
    dtype = _slug_token(contract_field(contract, 'input',
                                       'dtype', 'data_type', 'precision'))
    seqlen_kv_raw = contract_field(contract, 'input',
                                   'seqlen_kv', 'seq_len_kv', 'kv_len', 'kv')
    grid_token = _int_token(_first_present(
        contract_field(contract, 'runtime_launch', 'grid', 'grid_size',
                       'num_blocks', 'num_ctas', 'grid_dim'),
        contract_field(contract, 'launch', 'grid', 'grid_size', 'num_blocks'),
        contract.get('grid') if isinstance(contract, dict) else None,
        grid))
    warp_token = _int_token(_first_present(
        contract_field(contract, 'runtime_launch', 'warps_per_block',
                       'warps_per_cta', 'num_warps'),
        contract.get('warps_per_block') if isinstance(contract, dict) else None,
        warps_per_cta))
    phase_token = _int_token(_first_present(
        contract_field(contract, 'runtime_launch', 'phases', 'num_phases'),
        contract.get('phases') if isinstance(contract, dict) else None,
        num_phases))
    kv_part = ''
    if seqlen_kv_raw is not None:
        kv_token = _slug_token(seqlen_kv_raw)
        if kv_token != 'na':
            kv_part = '_kv{}'.format(kv_token)
    bundle_token = resolve_bundle_token(bundle)
    return ('trace__{}_{}__bs{}_sq{}{}__{}__grid{}_warp{}_ph{}__{}'
            '.perfetto.json.gz').format(model, stage, batch, seqlen_q, kv_part,
                                        dtype, grid_token, warp_token, phase_token,
                                        bundle_token)


def _load_contract(path: Path) -> Dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError('无法读取 contract JSON {}: {}'.format(path, error))
    if not isinstance(document, dict):
        raise ValueError('contract 顶层必须是 JSON object')
    return document


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError('{} 不能是布尔值'.format(label))
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError('{} 不是数字: {!r}'.format(label, value))
    if not math.isfinite(number):
        raise ValueError('{} 不是有限数: {!r}'.format(label, value))
    return number


def _nonnegative_timestamp(value: Any, label: str) -> float:
    timestamp = _finite_number(value, label)
    if timestamp < 0:
        raise ValueError('{} 不能为负数: {}'.format(label, timestamp))
    return timestamp


def load_hw_metrics(path: Path) -> Dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError('无法读取 hw_metrics JSON {}: {}'.format(path, error))
    if not isinstance(document, dict):
        raise ValueError('hw_metrics 顶层必须是 JSON object')
    return document


def choose_launch(document: Dict[str, Any], ordinal: Optional[int]
                  ) -> Tuple[int, Dict[str, Any]]:
    launches = document.get('kernel_launches')
    if not isinstance(launches, list) or not launches:
        raise ValueError('hw_metrics.kernel_launches 为空，无法确定目标 launch')
    if ordinal is None:
        if len(launches) != 1:
            raise ValueError(
                '匹配到 {} 个目标 launch；必须用 --launch-ordinal N 明确指定，不能自动猜测'
                .format(len(launches)))
        index = 0
    else:
        if ordinal < 0 or ordinal >= len(launches):
            raise ValueError('--launch-ordinal {} 越界（共有 {} 个 launch）'
                             .format(ordinal, len(launches)))
        index = ordinal
    launch = launches[index]
    if not isinstance(launch, dict):
        raise ValueError('kernel_launches[{}] 不是 JSON object'.format(index))
    start = _nonnegative_timestamp(launch.get('start_ns'),
                                    'kernel_launches[{}].start_ns'.format(index))
    end = _nonnegative_timestamp(launch.get('end_ns'),
                                  'kernel_launches[{}].end_ns'.format(index))
    if end < start:
        raise ValueError('选定 launch 的 end_ns 小于 start_ns')
    # 保留归一化后的整数/浮点值，后续不再重新解释输入。
    selected = dict(launch)
    selected['start_ns'] = start
    selected['end_ns'] = end
    return index, selected


def _counter_name(counter: Dict[str, Any], index: int) -> str:
    name = counter.get('name')
    if not isinstance(name, str) or not name.strip():
        raise ValueError('counters[{}].name 为空'.format(index))
    unit = counter.get('unit', '')
    if unit is None:
        unit = ''
    if not isinstance(unit, str):
        raise ValueError('counters[{}].unit 必须是字符串'.format(index))
    return '{} [{}]'.format(name.strip(), unit.strip()) if unit.strip() else name.strip()


def build_trace(document: Dict[str, Any], ordinal: Optional[int] = None
                ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    selected_index, launch = choose_launch(document, ordinal)
    start_ns = launch['start_ns']
    end_ns = launch['end_ns']
    counters = document.get('counters')
    if not isinstance(counters, list) or not counters:
        raise ValueError('hw_metrics.counters 为空')

    trace_events: List[Dict[str, Any]] = [
        {'name': 'process_name', 'ph': 'M', 'pid': HW_PID, 'tid': 0,
         'args': {'name': 'HW counters (selected kernel launch)'}},
        {'name': 'process_sort_index', 'ph': 'M', 'pid': HW_PID, 'tid': 0,
         'args': {'sort_index': -1}},
    ]
    counter_summary: List[Dict[str, Any]] = []
    counter_event_count = 0

    for counter_index, counter in enumerate(counters):
        if not isinstance(counter, dict):
            raise ValueError('counters[{}] 不是 JSON object'.format(counter_index))
        track_name = _counter_name(counter, counter_index)
        samples = counter.get('samples')
        if not isinstance(samples, list):
            raise ValueError('{}.samples 必须是列表'.format(track_name))

        kept: List[Tuple[float, float]] = []
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, (list, tuple)) or len(sample) < 2:
                raise ValueError('{} sample[{}] 必须是 [timestamp, value]'
                                 .format(track_name, sample_index))
            timestamp = _nonnegative_timestamp(
                sample[0], '{} sample[{}].timestamp'.format(track_name, sample_index))
            value = _finite_number(sample[1],
                                   '{} sample[{}].value'.format(track_name, sample_index))
            if start_ns <= timestamp <= end_ns:
                kept.append((timestamp, value))
        if not kept:
            raise ValueError('{} 在选定 launch 窗口 [{}, {}] 内没有样本'
                             .format(track_name, start_ns, end_ns))

        kept.sort(key=lambda item: item[0])
        for timestamp, value in kept:
            # 项目现有 Perfetto trace 使用微秒作为 ts 存储单位；时间原点仍是
            # 选定 launch 的 start_ns，otherData 明确记录该单位。
            trace_events.append({
                'name': track_name,
                'ph': 'C',
                'ts': (timestamp - start_ns) / 1000.0,
                'pid': HW_PID,
                'tid': 0,
                'args': {'value': value},
            })
        counter_event_count += len(kept)
        counter_summary.append({
            'name': counter.get('name'),
            'unit': counter.get('unit', ''),
            'samples_in_launch_window': len(kept),
            'first_ts_ns_from_launch': kept[0][0] - start_ns,
            'last_ts_ns_from_launch': kept[-1][0] - start_ns,
        })

    summary = {
        'schema': TRACE_SCHEMA,
        'input_schema': document.get('schema'),
        'device': document.get('device'),
        'metrics_set': document.get('metrics_set'),
        'frequency_hz': document.get('frequency_hz'),
        'selected_launch_ordinal': selected_index,
        'total_kernel_launches': len(document.get('kernel_launches') or []),
        'selected_launch': launch,
        'window_ns': [start_ns, end_ns],
        'timestamp_origin': 'selected launch start_ns',
        'timestamp_unit': 'us (Perfetto JSON ts; source and offsets are ns)',
        'counter_count': len(counter_summary),
        'counter_event_count': counter_event_count,
        'counters': counter_summary,
    }
    trace = {
        'traceEvents': trace_events,
        'displayTimeUnit': 'ns',
        'otherData': {
            'schema': TRACE_SCHEMA,
            'summary': summary,
        },
    }
    return trace, summary


def write_trace_gzip(trace: Dict[str, Any], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(out_path), 'wt', encoding='utf-8') as handle:
        json.dump(trace, handle, ensure_ascii=False, separators=(',', ':'))
    return out_path.stat().st_size


def _selftest() -> None:
    document = {
        'schema': INPUT_SCHEMA,
        'device': 0,
        'kernel_launches': [
            {'start_ns': 100, 'end_ns': 200},
            {'start_ns': 1000, 'end_ns': 1300, 'correlation_id': 7},
        ],
        'counters': [
            {'name': 'SM Active', 'unit': '%',
             'samples': [[900, 0], [1100, 12.5], [1300, 18.0], [1400, 99]]},
            {'name': 'DRAM Throughput', 'unit': 'GB/s',
             'samples': [[1001, 2], [1200, 3]]},
        ],
    }
    try:
        build_trace(document)
    except ValueError as error:
        assert '必须用 --launch-ordinal' in str(error)
    else:
        raise AssertionError('多个 launch 未被拒绝')
    trace, summary = build_trace(document, ordinal=1)
    counter_events = [event for event in trace['traceEvents'] if event['ph'] == 'C']
    assert summary['selected_launch_ordinal'] == 1
    assert len(counter_events) == 4
    assert min(event['ts'] for event in counter_events) == 0.001
    assert all(math.isfinite(float(event['ts'])) and event['ts'] >= 0
               for event in counter_events)
    assert all(item['samples_in_launch_window'] >= 1 for item in summary['counters'])
    payload = io.BytesIO()
    with gzip.GzipFile(fileobj=payload, mode='wb') as handle:
        handle.write(json.dumps(trace, ensure_ascii=False).encode('utf-8'))
    with gzip.GzipFile(fileobj=io.BytesIO(payload.getvalue()), mode='rb') as handle:
        decoded = json.loads(handle.read().decode('utf-8'))
    assert decoded['otherData']['summary']['timestamp_origin'] == 'selected launch start_ns'
    bad = dict(document)
    bad['kernel_launches'] = [{'start_ns': 0, 'end_ns': 1}]
    bad['counters'] = [{'name': 'bad', 'samples': [[-1, 1]]}]
    try:
        build_trace(bad)
    except ValueError as error:
        assert '不能为负数' in str(error)
    else:
        raise AssertionError('负 timestamp 未被拒绝')
    print('[selftest] build_hw_perfetto PASS')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='把 hw_metrics.json 的单个 kernel launch 窗口转成 Perfetto .json.gz counter trace',
        allow_abbrev=False)
    parser.add_argument('--hw-metrics', help='extract_gpu_metrics.py 产出的 hw_metrics.json')
    parser.add_argument('--out', help='输出 .json.gz 路径；指向目录且给了 --contract 时，'
                                      '按 SPEC-1 规范拼 trace__..._...__..__hw.perfetto.json.gz')
    parser.add_argument('--contract', default=None,
                        help='项目 contract.json；--out 为目录时用于拼 SPEC-1 规范文件名')
    parser.add_argument('--information-bundle', dest='information_bundle', default=None,
                        choices=('sw', 'hw', 'sw+hw'),
                        help='覆盖文件名 bundle 段；build_hw_perfetto 产出的是 HW trace，默认 hw')
    parser.add_argument('--bundle', dest='bundle', default=None,
                        help='等价于 --information-bundle 的自由文本写法（sw/hw/sw+hw/sw_hw）')
    parser.add_argument('--launch-ordinal', type=int, default=None,
                        help='多个目标 launch 时必须指定的 0-based ordinal')
    parser.add_argument('--selftest', action='store_true', help='运行内存自测，不读写外部文件')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    if not args.hw_metrics or not args.out:
        parser.error('正常转换必须同时提供 --hw-metrics 和 --out')
    try:
        trace, summary = build_trace(load_hw_metrics(Path(args.hw_metrics).expanduser().resolve()),
                                     args.launch_ordinal)
        out_path = Path(args.out).expanduser().resolve()
        if out_path.is_dir():
            bundle_request = _first_present(args.information_bundle, args.bundle, 'hw')
            if args.contract:
                contract = _load_contract(Path(args.contract).expanduser().resolve())
                filename = build_trace_filename(contract, bundle=bundle_request)
            else:
                filename = 'trace__run__{}.perfetto.json.gz'.format(
                    resolve_bundle_token(bundle_request))
            out_path = out_path / filename
        if not out_path.name.endswith('.json.gz'):
            raise ValueError('--out 必须以 .json.gz 结尾')
        size = write_trace_gzip(trace, out_path)
    except (OSError, ValueError) as error:
        print('[fatal] {}'.format(error), file=sys.stderr)
        return 2
    print('[out] {} bytes={} launch={} counters={} events={}'.format(
        out_path, size, summary['selected_launch_ordinal'],
        summary['counter_count'], summary['counter_event_count']))
    return 0


if __name__ == '__main__':
    sys.exit(main())

