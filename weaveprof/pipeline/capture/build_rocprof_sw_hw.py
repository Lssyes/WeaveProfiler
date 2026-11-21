#!/usr/bin/env python3
"""Build MI308X software+hardware Perfetto bundles from rocprofv3 PMC output.

Two public modes:

``extract``
    rocprofv3 JSON -> ``gpu-hw-metrics/v1``.  The file keeps raw per-dispatch
    counters and the four stable curve ids.  Missing evidence is represented as
    ``value=None`` / ``evidence_status=unavailable``.

``merge``
    Merge a canonical ``__sw.perfetto.json.gz`` with the extracted document and
    write the matching ``__sw_hw.perfetto.json.gz``.  AMD PMC is per dispatch,
    so the selected dispatch aggregate is displayed as a flat segment across
    the software event span.  No intra-dispatch timing is invented.

``all`` runs both steps.  The implementation intentionally uses rocprofv3's
official derived counters when present.  For MFMA/GEMM kernels, canonical
``compute_util`` and auxiliary ``tensor_util`` come only from ``MfmaUtil``.
``SIMD_UTILIZATION`` is kept separately as ``simd_util`` and is never used as a
Tensor/Matrix Core fallback.  ``GPU_UTIL`` remains the device-level
``sm_active`` evidence.  Arbitrary raw counters are not reinterpreted as
percentages.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

CAPTURE_DIR = Path(__file__).resolve().parent
if str(CAPTURE_DIR) not in sys.path:
    sys.path.insert(0, str(CAPTURE_DIR))

from hardware_utilization import (  # noqa: E402
    CANONICAL_METRIC_IDS, document_platform, evaluate_hardware_utilization,
    filter_utilization_evidence_to_window,
)
from merge_hw_into_perfetto import (  # noqa: E402
    build_merged_trace_filename, collect_pids, read_json_maybe_gz,
    rebase_timeline_nonnegative, timestamp_report, validate_merge_identity,
    write_json_gz,
)

INPUT_SCHEMA = 'rocprofv3-sdk-results/v1'
OUTPUT_SCHEMA = 'gpu-hw-metrics/v1'
POLICY = 'weaveprofiler-rocprof-dispatch-v1'
DEFAULT_HBM_PEAK_GBPS = 3684.0
DEFAULT_REQUEST_BYTES = 64.0
DEFAULT_CARD = 'mi308x'
DEFAULT_ARCH = 'gfx942'
# ``Cijk_`` is the stable rocBLAS/Tensile GEMM kernel prefix emitted for this
# gfx942 MFMA route.  Keep it separate from generic ``gemm`` so vendor library
# kernels are recognized without guessing from the enclosing Python operation.
MATRIX_CORE_KERNEL_TOKENS: Tuple[str, ...] = ('mfma', 'gemm', 'cijk_')
AUXILIARY_METRIC_IDS: Tuple[str, ...] = ('tensor_util', 'simd_util')
TIMELINE_LABELS = {
    'compute_util': 'compute_util [matrix_core]',
    'tensor_util': 'tensor_util [matrix_core]',
    'simd_util': 'simd_util [SIMD/CU]',
}


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _tool_root(document: Mapping[str, Any]) -> Dict[str, Any]:
    root = document.get('rocprofiler-sdk-tool')
    if not isinstance(root, list) or not root or not isinstance(root[0], dict):
        raise ValueError('rocprof results 缺 rocprofiler-sdk-tool[0]')
    return root[0]


def _kernel_names(tool: Mapping[str, Any]) -> Dict[Any, str]:
    names: Dict[Any, str] = {}
    for entry in tool.get('kernel_symbols') or []:
        if not isinstance(entry, dict) or 'kernel_id' not in entry:
            continue
        kernel_id = entry.get('kernel_id')
        name = None
        for key in ('demangled_kernel_name', 'formatted_kernel_name',
                    'kernel_name', 'truncated_kernel_name'):
            candidate = entry.get(key)
            if isinstance(candidate, str) and candidate.strip() and candidate.strip() != '0':
                name = candidate.strip()
                break
        if name:
            names[kernel_id] = name
    return names


def _gpu_agents(tool: Mapping[str, Any]) -> List[Dict[str, Any]]:
    agents = []
    for entry in tool.get('agents') or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get('name') or '')
        product = str(entry.get('product_name') or '')
        if name.startswith('gfx') or 'Instinct' in product:
            agents.append(entry)
    return sorted(agents, key=lambda item: int((item.get('id') or {}).get('handle') or 0))


def _agent_handle(entry: Mapping[str, Any]) -> Optional[int]:
    handle = (entry.get('id') or {}).get('handle') if isinstance(entry.get('id'), dict) else None
    return int(handle) if handle is not None else None


def _counter_definitions(tool: Mapping[str, Any], agent_handle: int
                         ) -> Dict[int, Dict[str, Any]]:
    definitions: Dict[int, Dict[str, Any]] = {}
    for entry in tool.get('counters') or []:
        if not isinstance(entry, dict):
            continue
        owner = (entry.get('agent_id') or {}).get('handle') if isinstance(entry.get('agent_id'), dict) else None
        counter_id = (entry.get('id') or {}).get('handle') if isinstance(entry.get('id'), dict) else None
        if owner is None or counter_id is None or int(owner) != int(agent_handle):
            continue
        definitions[int(counter_id)] = entry
    return definitions


def _counter_unit(name: str) -> str:
    if name in ('GPU_UTIL', 'MfmaUtil'):
        return 'percent'
    if name == 'SIMD_UTILIZATION':
        return 'fraction'
    if name.endswith('_sum') and ('REQ' in name or 'DRAM' in name):
        return 'requests'
    if name.startswith('SQ_') or name.startswith('GRBM_'):
        return 'count'
    return 'count'


def _matrix_core_route_tokens(kernel_name: str) -> List[str]:
    lowered = str(kernel_name or '').casefold()
    return [token for token in MATRIX_CORE_KERNEL_TOKENS if token in lowered]


def _available_evidence(metric_id: str, samples: Sequence[Sequence[Any]], *,
                        source: str, native_metric: str, compute_pipe: str,
                        method: str, extra: Optional[Mapping[str, Any]] = None
                        ) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        'semantic_id': metric_id,
        'display_name': metric_id,
        'timeline_label': TIMELINE_LABELS.get(metric_id, metric_id),
        'unit': 'percent_of_peak',
        'source': source,
        'native_metric': native_metric,
        'compute_pipe': compute_pipe,
        'evidence_status': 'derived',
        'method': method,
        'scope': 'device',
        'samples': [[int(timestamp), value] for timestamp, value in samples],
    }
    if extra:
        entry.update(extra)
    return entry


def _unavailable_evidence(metric_id: str, *, source: str, reason: str,
                          native_metric: Optional[str] = None,
                          compute_pipe: Optional[str] = None,
                          extra: Optional[Mapping[str, Any]] = None
                          ) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        'semantic_id': metric_id,
        'display_name': metric_id,
        'value': None,
        'unit': 'percent_of_peak',
        'source': source,
        'evidence_status': 'unavailable',
        'reason': reason,
    }
    if native_metric is not None:
        entry['native_metric'] = native_metric
    if compute_pipe is not None:
        entry['compute_pipe'] = compute_pipe
        entry['timeline_label'] = TIMELINE_LABELS.get(metric_id, metric_id)
    if extra:
        entry.update(extra)
    return entry


def _dispatch_info(record: Mapping[str, Any]) -> Mapping[str, Any]:
    data = record.get('dispatch_data') if isinstance(record.get('dispatch_data'), dict) else record
    info = data.get('dispatch_info') if isinstance(data, dict) else None
    return info if isinstance(info, dict) else {}


def _start_end(record: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    data = record.get('dispatch_data') if isinstance(record.get('dispatch_data'), dict) else record
    start = data.get('start_timestamp') if isinstance(data, dict) else None
    end = data.get('end_timestamp') if isinstance(data, dict) else None
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and int(end) > int(start):
        return int(start), int(end)
    return None


def extract_results(results_path: Path, kernel_name_contains: str, *,
                    card: str = DEFAULT_CARD, arch: str = DEFAULT_ARCH,
                    hbm_peak_gbps: float = DEFAULT_HBM_PEAK_GBPS,
                    request_bytes: float = DEFAULT_REQUEST_BYTES,
                    counter_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Parse one rocprofv3 results JSON into the canonical hardware document."""
    try:
        document = json.loads(Path(results_path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('无法读取 rocprof results JSON: {}'.format(exc)) from exc
    tool = _tool_root(document)
    names = _kernel_names(tool)
    gpu_agents = _gpu_agents(tool)
    ordinal_by_handle = {_agent_handle(agent): index for index, agent in enumerate(gpu_agents)
                         if _agent_handle(agent) is not None}

    records = ((tool.get('callback_records') or {}).get('counter_collection') or [])
    launches: List[Dict[str, Any]] = []
    raw_values: Dict[str, Dict[str, Any]] = {}
    launch_values: List[Dict[str, Any]] = []
    wanted = set(counter_names or ())
    needle = kernel_name_contains or ''

    for record in records:
        if not isinstance(record, dict):
            continue
        info = _dispatch_info(record)
        kernel_id = info.get('kernel_id')
        name = names.get(kernel_id)
        if not name or (needle and needle not in name):
            continue
        window = _start_end(record)
        if window is None:
            continue
        start_ns, end_ns = window
        agent_handle = (info.get('agent_id') or {}).get('handle')
        agent_handle = int(agent_handle) if agent_handle is not None else None
        if agent_handle is None:
            continue
        definitions = _counter_definitions(tool, agent_handle)
        values: Dict[str, float] = {}
        value_expressions: Dict[str, str] = {}
        for item in record.get('records') or []:
            if not isinstance(item, dict):
                continue
            handle = (item.get('counter_id') or {}).get('handle')
            value = _finite(item.get('value'))
            if handle is None or value is None:
                continue
            definition = definitions.get(int(handle))
            if not definition:
                continue
            counter_name = str(definition.get('name') or '').strip()
            if not counter_name or (wanted and counter_name not in wanted):
                continue
            values[counter_name] = value
            expression = definition.get('expression')
            if isinstance(expression, str) and expression:
                value_expressions[counter_name] = expression

        dispatch_data = record.get('dispatch_data') if isinstance(record.get('dispatch_data'), dict) else record
        correlation = (dispatch_data.get('correlation_id') or {}).get('internal') \
            if isinstance(dispatch_data, dict) else None
        route_tokens = _matrix_core_route_tokens(name)
        launch = {
            'start_ns': start_ns,
            'end_ns': end_ns,
            'duration_ns': end_ns - start_ns,
            'name': name,
            'device': ordinal_by_handle.get(agent_handle, 0),
            'agent_handle': agent_handle,
            'kernel_id': kernel_id,
            'correlation_id': correlation,
            'grid_size': info.get('grid_size'),
            'workgroup_size': info.get('workgroup_size'),
            'matrix_core_route': bool(route_tokens),
            'matrix_core_route_tokens': route_tokens,
        }
        launches.append(launch)
        launch_values.append({'launch': launch, 'values': values})
        for counter_name, value in values.items():
            item = raw_values.setdefault(counter_name, {
                'name': counter_name, 'unit': _counter_unit(counter_name),
                'samples': [], 'sample_count': 0,
                'counter_expressions': [],
            })
            item['samples'].append([start_ns, value])
            item['sample_count'] += 1
            expression = value_expressions.get(counter_name)
            if expression and expression not in item['counter_expressions']:
                item['counter_expressions'].append(expression)

    if not launches:
        raise ValueError('rocprof results 中没有匹配 kernel {!r} 的 counter_collection dispatch'
                         .format(needle))
    launches.sort(key=lambda item: (item['start_ns'], item['correlation_id'] or 0))
    launch_values.sort(key=lambda item: (item['launch']['start_ns'],
                                         item['launch']['correlation_id'] or 0))
    for counters in raw_values.values():
        counters['samples'].sort(key=lambda sample: sample[0])

    evidence: Dict[str, Dict[str, Any]] = {}

    matrix_launches = [item for item in launch_values
                       if item['launch'].get('matrix_core_route')]
    matrix_route_tokens = sorted({
        token for item in matrix_launches
        for token in item['launch'].get('matrix_core_route_tokens') or []
    })
    matrix_samples: List[List[Any]] = []
    for item in matrix_launches:
        value = item['values'].get('MfmaUtil')
        if value is not None and 0.0 <= value <= 100.0:
            matrix_samples.append([item['launch']['start_ns'], round(value, 9)])
    matrix_extra = {
        'semantic_route': 'matrix_core',
        'route_basis': ('kernel_name_contains:' + '|'.join(matrix_route_tokens)
                        if matrix_route_tokens else 'kernel_name_contains'),
    }
    if matrix_samples:
        evidence['compute_util'] = _available_evidence(
            'compute_util', matrix_samples,
            source='rocprofv3:MfmaUtil', native_metric='MfmaUtil',
            compute_pipe='matrix_core',
            method=('rocprofv3 derived MfmaUtil is a percent-of-peak Matrix Core '
                    'utilization and is used directly; no SIMD fallback'),
            extra=matrix_extra)
    elif not matrix_launches:
        evidence['compute_util'] = _unavailable_evidence(
            'compute_util', source='rocprofv3:MfmaUtil',
            native_metric='MfmaUtil',
            reason=('matched kernel names do not identify an MFMA/GEMM route; '
                    'compute_util is not assigned from SIMD_UTILIZATION or GPU busy'))
    else:
        evidence['compute_util'] = _unavailable_evidence(
            'compute_util', source='rocprofv3:MfmaUtil',
            native_metric='MfmaUtil', compute_pipe='matrix_core',
            reason=('MfmaUtil is absent or outside [0,100] for the MFMA/GEMM '
                    'dispatch; SIMD_UTILIZATION is not used as a fallback'),
            extra=matrix_extra)

    if matrix_samples:
        evidence['tensor_util'] = _available_evidence(
            'tensor_util', matrix_samples,
            source='rocprofv3:MfmaUtil', native_metric='MfmaUtil',
            compute_pipe='matrix_core',
            method=('auxiliary Matrix/Tensor Core evidence; the value comes only '
                    'from rocprofv3 MfmaUtil and never from SIMD_UTILIZATION'),
            extra=dict(matrix_extra, parent='compute_util', auxiliary=True))
    elif not matrix_launches:
        evidence['tensor_util'] = _unavailable_evidence(
            'tensor_util', source='rocprofv3:MfmaUtil',
            native_metric='MfmaUtil',
            reason=('matched kernel names do not identify an MFMA/GEMM route; '
                    'tensor_util is unavailable rather than zero'))
    else:
        evidence['tensor_util'] = _unavailable_evidence(
            'tensor_util', source='rocprofv3:MfmaUtil',
            native_metric='MfmaUtil', compute_pipe='matrix_core',
            reason=('MfmaUtil is absent or outside [0,100] for the MFMA/GEMM '
                    'dispatch; tensor_util is unavailable rather than zero'),
            extra=dict(matrix_extra, parent='compute_util', auxiliary=True))

    simd_samples: List[List[Any]] = []
    for item in launch_values:
        value = item['values'].get('SIMD_UTILIZATION')
        if value is not None and 0.0 <= value <= 1.01:
            simd_samples.append([item['launch']['start_ns'], round(value * 100.0, 9)])
    if simd_samples:
        evidence['simd_util'] = _available_evidence(
            'simd_util', simd_samples,
            source='rocprofv3:SIMD_UTILIZATION*100',
            native_metric='SIMD_UTILIZATION', compute_pipe='simd',
            method=('rocprofv3 derived SIMD_UTILIZATION = '
                    'sum(SQ_BUSY_CU_CYCLES)/max(GRBM_COUNT)/CU_NUM; multiplied by 100; '
                    'SIMD/CU evidence only'),
            extra={'parent': 'compute_util', 'auxiliary': True})
    else:
        evidence['simd_util'] = _unavailable_evidence(
            'simd_util', source='rocprofv3:SIMD_UTILIZATION*100',
            native_metric='SIMD_UTILIZATION', compute_pipe='simd',
            reason='rocprofv3 derived SIMD_UTILIZATION is absent or outside [0,1]',
            extra={'parent': 'compute_util', 'auxiliary': True})

    sm_samples: List[List[Any]] = []
    for item in launch_values:
        value = item['values'].get('GPU_UTIL')
        if value is not None and 0.0 <= value <= 100.0:
            sm_samples.append([item['launch']['start_ns'], round(value, 9)])
    if sm_samples:
        evidence['sm_active'] = {
            'semantic_id': 'sm_active', 'display_name': 'sm_active',
            'unit': 'percent_of_peak', 'source': 'rocprofv3:GPU_UTIL',
            'evidence_status': 'derived',
            'method': 'rocprofv3 GPU_UTIL = 100*max(GRBM_GUI_ACTIVE)/max(GRBM_COUNT)',
            'scope': 'device', 'samples': sm_samples,
        }
    else:
        evidence['sm_active'] = {
            'semantic_id': 'sm_active', 'display_name': 'sm_active',
            'value': None, 'unit': 'percent_of_peak', 'source': 'rocprofv3:GPU_UTIL',
            'evidence_status': 'unavailable',
            'reason': 'rocprofv3 derived GPU_UTIL is absent or outside [0,100]',
        }

    memory_samples: List[List[Any]] = []
    memory_details: List[Dict[str, Any]] = []
    peak_bytes_s = float(hbm_peak_gbps) * 1e9
    for item in launch_values:
        read_req = item['values'].get('TCC_EA0_RDREQ_DRAM_sum')
        write_req = item['values'].get('TCC_EA0_WRREQ_DRAM_sum')
        launch = item['launch']
        if read_req is None or write_req is None:
            continue
        duration_s = launch['duration_ns'] / 1e9
        if duration_s <= 0 or peak_bytes_s <= 0:
            continue
        requests = read_req + write_req
        bytes_est = requests * float(request_bytes)
        value = (bytes_est / duration_s / peak_bytes_s) * 100.0
        memory_samples.append([launch['start_ns'], round(value, 9)])
        memory_details.append({
            'correlation_id': launch['correlation_id'],
            'duration_ns': launch['duration_ns'],
            'read_requests': read_req, 'write_requests': write_req,
            'bytes_estimate': bytes_est,
            'request_bytes_assumption': float(request_bytes),
            'hbm_peak_gbps': float(hbm_peak_gbps),
            'memory_util': round(value, 9),
        })
    if memory_samples:
        evidence['memory_util'] = {
            'semantic_id': 'memory_util', 'display_name': 'memory_util',
            'unit': 'percent_of_peak',
            'source': ('rocprofv3:TCC_EA0_{{RD,WR}}REQ_DRAM_sum; '
                       '{}B/request; {}GB/s measured HBM peak').format(
                           float(request_bytes), float(hbm_peak_gbps)),
            'evidence_status': 'derived',
            'method': ('dispatch bytes estimate / dispatch duration / measured HBM peak; '
                       'request size is an explicit assumption when the counter does not expose it'),
            'scope': 'device', 'samples': memory_samples,
            'details': memory_details,
        }
    else:
        evidence['memory_util'] = {
            'semantic_id': 'memory_util', 'display_name': 'memory_util',
            'value': None, 'unit': 'percent_of_peak',
            'source': 'rocprofv3:TCC_EA0_RDREQ_DRAM_sum+TCC_EA0_WRREQ_DRAM_sum',
            'evidence_status': 'unavailable',
            'reason': ('both DRAM request directions and a positive dispatch duration are '
                       'required; missing values are not filled with zero'),
        }

    evidence['communication_util'] = {
        'semantic_id': 'communication_util', 'display_name': 'communication_util',
        'value': None, 'unit': 'percent_of_peak',
        'source': 'rocprofv3:XGMI-bandwidth-or-NIC',
        'evidence_status': 'unavailable',
        'reason': ('this gfx942 rocprofv3 build exposes no normalized XGMI bandwidth counter; '
                   'GMI credit-stall cycles are congestion, not bandwidth. Cross-host '
                   'communication additionally requires NIC/RDMA/fabric evidence.'),
    }

    active_agent = launches[0]['agent_handle']
    agent_info = next((agent for agent in gpu_agents if _agent_handle(agent) == active_agent), {})
    if counter_names:
        missing = sorted(set(counter_names) - set(raw_values))
        if missing:
            raise ValueError('请求的 PMC 没有采到: {}'.format(missing))
    return {
        'schema': OUTPUT_SCHEMA,
        'policy': POLICY,
        'platform': 'amd',
        'target_card': card,
        'target_arch': arch,
        'arch': arch,
        'device': launches[0]['device'],
        'agent_handle': active_agent,
        'source_results': str(Path(results_path).resolve()),
        'metrics_set': 'rocprofv3-pmc-dispatch-aggregate/v1',
        'sampling_model': 'per_dispatch_pmc_plus_optional_amdsmi_coarse',
        'compute_routing': {
            'matrix_core_kernel_tokens': list(MATRIX_CORE_KERNEL_TOKENS),
            'matrix_core_route_tokens': matrix_route_tokens,
            'compute_util_native_metric': 'MfmaUtil',
            'tensor_util_native_metric': 'MfmaUtil',
            'simd_util_native_metric': 'SIMD_UTILIZATION',
            'simd_fallback_for_tensor_forbidden': True,
        },
        'sampling_interval_ns': {
            'available': False,
            'reason': ('rocprofv3 PMC is aggregated per kernel dispatch; it is not a '
                       'periodic PM time series'),
        },
        'kernel_launches': launches,
        'counters': sorted(raw_values.values(), key=lambda item: item['name']),
        'utilization_evidence': evidence,
        'hardware_info': {
            'product_name': agent_info.get('product_name'),
            'isa_name': agent_info.get('name'),
            'cu_count': agent_info.get('cu_count'),
            'wave_front_size': agent_info.get('wave_front_size'),
            'simd_count': agent_info.get('simd_count'),
            'max_engine_clk_fcompute_mhz': agent_info.get('max_engine_clk_fcompute'),
            'hbm_peak_gbps': float(hbm_peak_gbps),
            'hbm_peak_source': ('docs/amd_mi308x.md measured stream-read peak'
                                if float(hbm_peak_gbps) == DEFAULT_HBM_PEAK_GBPS else 'cli'),
            'dram_request_bytes_assumption': float(request_bytes),
            'aggregation_scope': 'one counter value pair per kernel dispatch',
        },
    }


def _sw_window_ns(trace: Mapping[str, Any]) -> Tuple[int, int, int]:
    events = trace.get('traceEvents')
    if not isinstance(events, list):
        raise ValueError('software trace 缺 traceEvents')
    intervals = []
    for event in events:
        if not isinstance(event, dict) or event.get('ph') != 'X':
            continue
        ts = _finite(event.get('ts'))
        dur = _finite(event.get('dur'))
        if ts is None or dur is None or dur < 0:
            continue
        intervals.append((ts * 1000.0, (ts + dur) * 1000.0))
    if not intervals:
        raise ValueError('software trace 没有可用于对齐的 ph=X 事件')
    start_ns = int(round(min(item[0] for item in intervals)))
    end_ns = int(round(max(item[1] for item in intervals)))
    if end_ns <= start_ns:
        raise ValueError('software trace X-event 窗口非法')
    return start_ns, end_ns, len(intervals)


def _select_launch(document: Mapping[str, Any], ordinal: Optional[int]) -> Tuple[int, Dict[str, Any]]:
    launches = document.get('kernel_launches')
    if not isinstance(launches, list) or not launches:
        raise ValueError('hardware document 缺 kernel_launches')
    if ordinal is None:
        if len(launches) != 1:
            raise ValueError('匹配到 {} 个 dispatch；请用 --launch-ordinal 明确选择'
                             .format(len(launches)))
        ordinal = 0
    if ordinal < 0 or ordinal >= len(launches):
        raise ValueError('--launch-ordinal {} 越界（共 {} 个）'.format(ordinal, len(launches)))
    chosen = launches[ordinal]
    if not isinstance(chosen, dict):
        raise ValueError('kernel_launches[{}] 不是 object'.format(ordinal))
    return ordinal, chosen


def _sanitize_amd_utilization_evidence(
        document: Mapping[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Reject any AMD tensor/compute provenance that could confuse pipelines."""
    result = dict(document)
    raw_wrapper = document.get('utilization_evidence')
    if not isinstance(raw_wrapper, Mapping):
        return result, []
    evidence: Dict[str, Any] = {}
    rejections: List[Dict[str, Any]] = []
    for metric_id, raw in raw_wrapper.items():
        if not isinstance(raw, Mapping):
            evidence[str(metric_id)] = raw
            continue
        item = dict(raw)
        status = str(item.get('evidence_status') or 'measured').strip().lower()
        required_pipe: Optional[str] = None
        required_native: Optional[str] = None
        required_source: Optional[str] = None
        if metric_id in ('compute_util', 'tensor_util'):
            required_pipe = 'matrix_core'
            required_native = 'MfmaUtil'
            required_source = 'rocprofv3:MfmaUtil'
        elif metric_id == 'simd_util':
            required_pipe = 'simd'
            required_native = 'SIMD_UTILIZATION'
            required_source = 'rocprofv3:SIMD_UTILIZATION*100'
        if required_pipe is not None and status != 'unavailable':
            observed = {
                'source': item.get('source'),
                'native_metric': item.get('native_metric'),
                'compute_pipe': item.get('compute_pipe'),
            }
            if (observed['source'] != required_source
                    or observed['native_metric'] != required_native
                    or observed['compute_pipe'] != required_pipe):
                reason = (
                    '{} requires source={!r}, native_metric={!r}, compute_pipe={!r}; '
                    'observed source={!r}, native_metric={!r}, compute_pipe={!r}. '
                    'The evidence is rejected instead of being relabeled.'
                ).format(metric_id, required_source, required_native, required_pipe,
                         observed['source'], observed['native_metric'],
                         observed['compute_pipe'])
                item = _unavailable_evidence(
                    str(metric_id), source=str(observed['source'] or required_source),
                    native_metric=str(observed['native_metric'] or required_native),
                    compute_pipe=str(observed['compute_pipe'] or required_pipe),
                    reason=reason)
                rejections.append({
                    'metric_id': str(metric_id),
                    'observed': observed,
                    'required': {
                        'source': required_source,
                        'native_metric': required_native,
                        'compute_pipe': required_pipe,
                    },
                    'reason': reason,
                })
        evidence[str(metric_id)] = item
    result['utilization_evidence'] = evidence
    return result, rejections


def _summarize_auxiliary_evidence(
        document: Mapping[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    raw_wrapper = document.get('utilization_evidence')
    evidence = raw_wrapper if isinstance(raw_wrapper, Mapping) else document
    summaries: Dict[str, Dict[str, Any]] = {}
    series: Dict[str, Dict[str, Any]] = {}
    defaults = {
        'tensor_util': ('rocprofv3:MfmaUtil', 'MfmaUtil', 'matrix_core', 'compute_util'),
        'simd_util': ('rocprofv3:SIMD_UTILIZATION*100', 'SIMD_UTILIZATION',
                      'simd', 'compute_util'),
    }
    for metric_id in AUXILIARY_METRIC_IDS:
        default_source, default_native, default_pipe, parent = defaults[metric_id]
        raw = evidence.get(metric_id) if isinstance(evidence, Mapping) else None
        if not isinstance(raw, Mapping):
            summaries[metric_id] = _unavailable_evidence(
                metric_id, source=default_source, native_metric=default_native,
                compute_pipe=default_pipe,
                reason='explicit auxiliary evidence is absent',
                extra={'parent': parent, 'auxiliary': True})
            continue
        source = str(raw.get('source') or default_source)
        native_metric = str(raw.get('native_metric') or default_native)
        compute_pipe = str(raw.get('compute_pipe') or default_pipe)
        status = str(raw.get('evidence_status') or 'measured').strip().lower()
        extra = {'parent': str(raw.get('parent') or parent), 'auxiliary': True}
        if status == 'unavailable':
            summaries[metric_id] = _unavailable_evidence(
                metric_id, source=source, native_metric=native_metric,
                compute_pipe=compute_pipe,
                reason=str(raw.get('reason') or 'explicit auxiliary evidence is unavailable'),
                extra=extra)
            continue
        unit = str(raw.get('unit') or 'percent_of_peak').strip().lower()
        if '%' not in unit and 'percent' not in unit:
            summaries[metric_id] = _unavailable_evidence(
                metric_id, source=source, native_metric=native_metric,
                compute_pipe=compute_pipe,
                reason='explicit auxiliary evidence unit {!r} is not a percentage'
                       .format(raw.get('unit')),
                extra=extra)
            continue
        samples: List[List[Any]] = []
        for sample in raw.get('samples') or []:
            if not isinstance(sample, (list, tuple)) or len(sample) < 2:
                continue
            number = _finite(sample[1])
            if number is None:
                continue
            try:
                timestamp = int(sample[0])
            except (TypeError, ValueError):
                continue
            samples.append([timestamp, round(float(number), 9)])
        if not samples:
            reason = ('explicit auxiliary evidence has no finite samples in the selected '
                      'kernel window' if raw.get('_window_filtered') else
                      'explicit auxiliary evidence has no finite samples')
            summaries[metric_id] = _unavailable_evidence(
                metric_id, source=source, native_metric=native_metric,
                compute_pipe=compute_pipe, reason=reason, extra=extra)
            continue
        value = round(sum(float(sample[1]) for sample in samples) / len(samples), 6)
        entry = _available_evidence(
            metric_id, samples, source=source, native_metric=native_metric,
            compute_pipe=compute_pipe,
            method='arithmetic_mean_of_clipped_auxiliary_samples', extra=extra)
        entry.update({
            'value': value,
            'status': 'available',
            'reason': None,
            'sample_count': len(samples),
            'observed_names': [source],
        })
        summaries[metric_id] = entry
        series[metric_id] = {
            'name': metric_id,
            'samples': samples,
            'source': source,
            'native_metric': native_metric,
            'compute_pipe': compute_pipe,
            'timeline_label': TIMELINE_LABELS[metric_id],
            'evidence_status': 'derived',
            'scope': 'device',
            'parent': extra['parent'],
            'auxiliary': True,
        }
    return summaries, series


def _canonical_timeline_label(metric_id: str, entry: Mapping[str, Any]) -> str:
    if metric_id == 'compute_util' and entry.get('compute_pipe') == 'matrix_core':
        return TIMELINE_LABELS['compute_util']
    return metric_id


def merge_sw_hw(trace_path: Path, hw_metrics_path: Path, out_path: Path, *,
                launch_ordinal: Optional[int] = None) -> Tuple[Path, Dict[str, Any]]:
    trace = read_json_maybe_gz(Path(trace_path))
    hw = read_json_maybe_gz(Path(hw_metrics_path))
    if not isinstance(trace, dict) or not isinstance(trace.get('traceEvents'), list):
        raise ValueError('software trace 必须是 {traceEvents:[...]}')
    if not isinstance(hw, dict):
        raise ValueError('hardware document 顶层必须是 object')
    if document_platform(hw) != 'amd':
        raise ValueError('本合并器只接受 AMD/gfx942 dispatch-aggregate 证据')
    sanitized_hw, evidence_rejections = _sanitize_amd_utilization_evidence(hw)

    events = trace['traceEvents']
    x_before = sum(1 for event in events if event.get('ph') == 'X')
    sw_start_ns, sw_end_ns, sw_event_count = _sw_window_ns(trace)
    ordinal, launch = _select_launch(hw, launch_ordinal)
    hw_start_ns = int(launch['start_ns'])
    hw_end_ns = int(launch['end_ns'])
    if hw_end_ns <= hw_start_ns:
        raise ValueError('选定 dispatch 的 start/end 非法')
    scale = (sw_end_ns - sw_start_ns) / (hw_end_ns - hw_start_ns)

    def map_ns(timestamp_ns: float) -> float:
        return (sw_start_ns + (float(timestamp_ns) - hw_start_ns) * scale) / 1000.0

    window_counters = []
    for counter in hw.get('counters') or []:
        samples = [[timestamp, value] for timestamp, value in counter.get('samples') or []
                   if hw_start_ns <= int(timestamp) <= hw_end_ns]
        item = dict(counter)
        item['samples'] = samples
        window_counters.append(item)
    evidence = filter_utilization_evidence_to_window(
        sanitized_hw, hw_start_ns, hw_end_ns)
    hardware_summary, canonical_series = evaluate_hardware_utilization(
        window_counters, platform='amd', communication_scope='same_host',
        evidence_document=evidence, source_prefix='rocprofv3.pmc',
        source_document=str(hw_metrics_path), target_card=hw.get('target_card'),
        window_ns=(hw_start_ns, hw_end_ns))
    auxiliary_metrics, auxiliary_series = _summarize_auxiliary_evidence(evidence)
    hardware_summary['auxiliary_metrics'] = auxiliary_metrics

    pids = collect_pids(events)
    min_pid = min(pids) if pids else 0
    pid_shift = 0
    hw_pid = min_pid - 1
    if hw_pid < 0:
        pid_shift = 1 - min_pid
        for event in events:
            if isinstance(event.get('pid'), int):
                event['pid'] += pid_shift
        hw_pid = 0

    metadata = [
        {'ph': 'M', 'name': 'process_name', 'pid': hw_pid, 'tid': 0,
         'args': {'name': 'HW dispatch aggregate (AMD MI308X)'}},
        {'ph': 'M', 'name': 'process_sort_index', 'pid': hw_pid, 'tid': 0,
         'args': {'sort_index': -1}},
        {'ph': 'M', 'name': 'process_labels', 'pid': hw_pid, 'tid': 0,
         'args': {'labels': ('Flat segment = one per-dispatch PMC aggregate. '
                             'No intra-dispatch timing is implied.')}},
        {'ph': 'M', 'name': 'thread_name', 'pid': hw_pid, 'tid': 0,
         'args': {'name': 'Canonical utilization (AMD rocprofv3)'}},
        {'ph': 'M', 'name': 'thread_name', 'pid': hw_pid, 'tid': 1,
         'args': {'name': 'Matrix core evidence (MfmaUtil)'}},
        {'ph': 'M', 'name': 'thread_name', 'pid': hw_pid, 'tid': 2,
         'args': {'name': 'SIMD/CU evidence (SIMD_UTILIZATION)'}},
        {'ph': 'M', 'name': 'thread_sort_index', 'pid': hw_pid, 'tid': 0,
         'args': {'sort_index': 0}},
        {'ph': 'M', 'name': 'thread_sort_index', 'pid': hw_pid, 'tid': 1,
         'args': {'sort_index': 1}},
        {'ph': 'M', 'name': 'thread_sort_index', 'pid': hw_pid, 'tid': 2,
         'args': {'sort_index': 2}},
    ]
    curve_events = []
    metric_report = []
    auxiliary_report = []
    endpoint_duplicated = 0
    for metric_id in CANONICAL_METRIC_IDS:
        series = canonical_series.get(metric_id)
        entry = hardware_summary['metrics'][metric_id]
        timeline_label = _canonical_timeline_label(metric_id, entry)
        kept = 0
        duplicated = False
        if series and series.get('samples'):
            points = [(map_ns(timestamp), float(value))
                      for timestamp, value in series['samples']]
            if len(points) == 1:
                points.append((sw_end_ns / 1000.0, points[0][1]))
                endpoint_duplicated += 1
                duplicated = True
            for display_ts, value in points:
                curve_events.append({
                    'ph': 'C', 'name': timeline_label, 'cat': 'hardware_utilization',
                    'ts': display_ts, 'pid': hw_pid, 'tid': 0,
                    # Perfetto's Chrome-JSON counter parser accepts only a
                    # numeric ``value``.  Non-numeric args silently increase
                    # json_parser_failure even though trace_processor exits 0.
                    # Source/status/display policy live in otherData below.
                    'args': {'value': value},
                })
                kept += 1
        metric_report.append({
            'semantic_id': metric_id,
            'timeline_label': timeline_label,
            'value': entry.get('value'),
            'unit': entry.get('unit'),
            'evidence_status': entry.get('evidence_status'),
            'source': entry.get('source'),
            'native_metric': entry.get('native_metric'),
            'compute_pipe': entry.get('compute_pipe'),
            'semantic_route': entry.get('semantic_route'),
            'auxiliary': False,
            'reason': entry.get('reason'),
            'display_samples': kept,
            'endpoint_duplicated_for_flat_aggregate': duplicated,
        })

    auxiliary_tids = {'tensor_util': 1, 'simd_util': 2}
    for metric_id in AUXILIARY_METRIC_IDS:
        entry = auxiliary_metrics[metric_id]
        series = auxiliary_series.get(metric_id)
        timeline_label = TIMELINE_LABELS[metric_id]
        kept = 0
        duplicated = False
        if series and series.get('samples'):
            points = [(map_ns(timestamp), float(value))
                      for timestamp, value in series['samples']]
            if len(points) == 1:
                points.append((sw_end_ns / 1000.0, points[0][1]))
                endpoint_duplicated += 1
                duplicated = True
            for display_ts, value in points:
                curve_events.append({
                    'ph': 'C', 'name': timeline_label, 'cat': 'hardware_auxiliary',
                    'ts': display_ts, 'pid': hw_pid, 'tid': auxiliary_tids[metric_id],
                    'args': {'value': value},
                })
                kept += 1
        auxiliary_report.append({
            'semantic_id': metric_id,
            'timeline_label': timeline_label,
            'value': entry.get('value'),
            'unit': entry.get('unit'),
            'evidence_status': entry.get('evidence_status'),
            'source': entry.get('source'),
            'native_metric': entry.get('native_metric'),
            'compute_pipe': entry.get('compute_pipe'),
            'parent': entry.get('parent'),
            'auxiliary': True,
            'reason': entry.get('reason'),
            'display_samples': kept,
            'endpoint_duplicated_for_flat_aggregate': duplicated,
        })

    trace['traceEvents'] = events + metadata + curve_events
    timeline_rebase = rebase_timeline_nonnegative(trace['traceEvents'])
    x_after = sum(1 for event in trace['traceEvents'] if event.get('ph') == 'X')
    if x_after != x_before:
        raise ValueError('合并改变了软件 ph=X 事件数: {} -> {}'.format(x_before, x_after))

    other = trace.get('otherData')
    if not isinstance(other, dict):
        other = {}
    other['hw_merge'] = {
        'schema': 'rocprof-dispatch-merge/v1',
        'source_software_trace': str(Path(trace_path).resolve()),
        'source_hw_metrics': str(Path(hw_metrics_path).resolve()),
        'source_rocprof_results': hw.get('source_results'),
        'selected_launch_ordinal': ordinal,
        'selected_launch': launch,
        'alignment': {
            'method': 'affine map from selected dispatch span to software X-event span',
            'software_window_ns': [sw_start_ns, sw_end_ns],
            'hardware_window_ns': [hw_start_ns, hw_end_ns],
            'scale': scale,
            'software_x_event_count': sw_event_count,
            'intra_dispatch_time_resolution': None,
            'claim': ('the PMC value is valid for the selected dispatch; its horizontal '
                      'placement is for display alignment only'),
        },
        'display_policy': {
            'aggregate_constant_segment': True,
            'single_evidence_sample_endpoint_duplicated': endpoint_duplicated,
            'raw_label_policy': ('stable semantic ids in otherData; timeline labels add the '
                                 'explicit compute pipe where needed'),
            'timeline_labels': dict(TIMELINE_LABELS),
            'tensor_util_source_policy': 'rocprofv3:MfmaUtil only',
            'simd_util_source_policy': 'rocprofv3:SIMD_UTILIZATION*100 only',
            'gpu_busy_tensor_mapping': None,
        },
        'metrics': metric_report,
        'auxiliary_metrics': auxiliary_report,
        'evidence_rejections': evidence_rejections,
        'timeline_rebase': timeline_rebase,
        'pid_shift_applied': pid_shift,
        'hw_pid': hw_pid,
        'r6_hw_pid_lt_all_software': hw_pid < min(event.get('pid', min_pid)
                                                  for event in events
                                                  if isinstance(event.get('pid'), int)),
    }
    other['hardware_summary'] = hardware_summary
    trace['otherData'] = other

    out_path = Path(out_path)
    if out_path.is_dir() or out_path.suffix == '':
        merged_name = validate_merge_identity(Path(trace_path), trace)
        out_path = out_path / merged_name
    elif out_path.name.startswith('trace__'):
        validate_merge_identity(Path(trace_path), trace, out_path.name)
    write_json_gz(out_path, trace)
    return out_path, {
        'schema': 'rocprof-sw-hw-build/v1',
        'out': str(out_path.resolve()),
        'bytes': out_path.stat().st_size,
        'selected_launch_ordinal': ordinal,
        'software_window_ns': [sw_start_ns, sw_end_ns],
        'hardware_window_ns': [hw_start_ns, hw_end_ns],
        'software_x_events': x_before,
        'curve_events': len(curve_events),
        'timeline_rebase': timeline_rebase,
        'metrics': metric_report,
        'auxiliary_metrics': auxiliary_report,
        'evidence_rejections': evidence_rejections,
        'final_timestamp_report': timestamp_report(trace['traceEvents']),
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    extract = sub.add_parser('extract', help='rocprofv3 JSON -> gpu-hw-metrics/v1')
    extract.add_argument('--results', required=True)
    extract.add_argument('--kernel-name-contains', required=True)
    extract.add_argument('--out', required=True)
    extract.add_argument('--card', default=DEFAULT_CARD)
    extract.add_argument('--arch', default=DEFAULT_ARCH)
    extract.add_argument('--hbm-peak-gbps', type=float, default=DEFAULT_HBM_PEAK_GBPS)
    extract.add_argument('--dram-request-bytes', type=float, default=DEFAULT_REQUEST_BYTES)
    extract.add_argument('--counter', action='append', default=None)
    merge = sub.add_parser('merge', help='SW trace + gpu-hw-metrics -> canonical sw_hw gz')
    merge.add_argument('--sw-trace', required=True)
    merge.add_argument('--hw-metrics', required=True)
    merge.add_argument('--out', required=True)
    merge.add_argument('--launch-ordinal', type=int, default=None)
    all_parser = sub.add_parser('all', help='extract then merge')
    all_parser.add_argument('--results', required=True)
    all_parser.add_argument('--sw-trace', required=True)
    all_parser.add_argument('--kernel-name-contains', required=True)
    all_parser.add_argument('--out-dir', required=True)
    all_parser.add_argument('--launch-ordinal', type=int, default=None)
    all_parser.add_argument('--card', default=DEFAULT_CARD)
    all_parser.add_argument('--arch', default=DEFAULT_ARCH)
    all_parser.add_argument('--hbm-peak-gbps', type=float, default=DEFAULT_HBM_PEAK_GBPS)
    all_parser.add_argument('--dram-request-bytes', type=float, default=DEFAULT_REQUEST_BYTES)
    all_parser.add_argument('--counter', action='append', default=None)
    return parser.parse_args(list(argv))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.mode == 'extract':
            doc = extract_results(Path(args.results), args.kernel_name_contains,
                                  card=args.card, arch=args.arch,
                                  hbm_peak_gbps=args.hbm_peak_gbps,
                                  request_bytes=args.dram_request_bytes,
                                  counter_names=args.counter)
            out = Path(args.out).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + '\n',
                           encoding='utf-8')
            print(json.dumps({'out': str(out), 'bytes': out.stat().st_size,
                              'launches': len(doc['kernel_launches']),
                              'counters': [item['name'] for item in doc['counters']],
                              'metrics': {key: {'value': value.get('value'),
                                                'status': value.get('evidence_status')}
                                          for key, value in doc['utilization_evidence'].items()}},
                             indent=2, ensure_ascii=False))
            return 0
        if args.mode == 'merge':
            out, report = merge_sw_hw(Path(args.sw_trace), Path(args.hw_metrics),
                                      Path(args.out), launch_ordinal=args.launch_ordinal)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        hw_path = out_dir / 'hw_metrics.json'
        doc = extract_results(Path(args.results), args.kernel_name_contains,
                              card=args.card, arch=args.arch,
                              hbm_peak_gbps=args.hbm_peak_gbps,
                              request_bytes=args.dram_request_bytes,
                              counter_names=args.counter)
        hw_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + '\n',
                           encoding='utf-8')
        out, report = merge_sw_hw(Path(args.sw_trace), hw_path, out_dir,
                                  launch_ordinal=args.launch_ordinal)
        report['hw_metrics'] = str(hw_path)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except ValueError as exc:
        print('[fatal] {}'.format(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
