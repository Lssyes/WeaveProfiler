#!/usr/bin/env python3
"""Canonical hardware-utilization summaries for extracted hardware evidence.

The four public ids are stable across NVIDIA and AMD outputs:

    compute_util, memory_util, communication_util, sm_active

Every entry carries ``value``, ``source`` and ``evidence_status``.  A metric
without suitable evidence is represented with ``value=None`` and
``evidence_status="unavailable"``; zero is reserved for a measured zero.

NVIDIA nsys GPU-metrics samples are already normalized percentages.  AMD
rocprofv3 PMC records are per-dispatch raw counters and must not be presented as
utilization percentages without a separately documented derivation and peak.
Communication over NVLink/XGMI is same-host GPU interconnect evidence.  A
cross-host communication value additionally requires NIC/RDMA/fabric evidence.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA = 'hardware-utilization-summary/v1'
POLICY = 'weaveprofiler-hardware-metrics/v1'
CANONICAL_METRIC_IDS: Tuple[str, ...] = (
    'compute_util',
    'memory_util',
    'communication_util',
    'sm_active',
)
AUXILIARY_METRIC_IDS: Tuple[str, ...] = (
    'tensor_util',
    'simd_util',
)
EVIDENCE_METRIC_IDS: Tuple[str, ...] = CANONICAL_METRIC_IDS + AUXILIARY_METRIC_IDS
UNIT = 'percent_of_peak'
_EVIDENCE_STATUSES = frozenset(('measured', 'derived', 'unavailable'))
_COMMUNICATION_SCOPES = frozenset(('unknown', 'same_host', 'cross_host'))
_NIC_EVIDENCE_KINDS = frozenset(('nic', 'rdma', 'fabric'))
_LINK_EVIDENCE_KINDS = frozenset(('nvlink', 'xgmi'))
_UNIT_SUFFIX_RE = re.compile(r'\s*\[[^\]]+\]\s*$')
_NIC_RE = re.compile(
    r'(?:\bnic\b|rdma|infiniband|\bib(?:_|port|_port)|\bib[0-9]|'
    r'port_(?:rcv|xmit|receive|transmit)|fabric)', re.IGNORECASE)
_NVLINK_RE = re.compile(r'nvl(?:rx|tx)|nvlink', re.IGNORECASE)
_XGMI_RE = re.compile(r'xgmi|gmi_(?:read|write|credit)|tcc_ea0_.*gmi', re.IGNORECASE)


def normalize_platform(value: Any) -> str:
    text = str(value or '').strip().lower()
    if text in ('nv', 'nvidia', 'cuda'):
        return 'nv'
    if text in ('amd', 'rocm', 'hip', 'gfx942'):
        return 'amd'
    if 'nvidia' in text or text.startswith('sm_') or text.startswith('gb') or text.startswith('gh'):
        return 'nv'
    if 'amd' in text or 'gfx' in text or 'rocm' in text:
        return 'amd'
    return 'unknown'


def normalize_communication_scope(value: Any) -> str:
    text = str(value or 'unknown').strip().lower().replace('-', '_')
    aliases = {
        'single_host': 'same_host',
        'intra_node': 'same_host',
        'intranode': 'same_host',
        'local': 'same_host',
        'multi_host': 'cross_host',
        'inter_node': 'cross_host',
        'internode': 'cross_host',
        'remote': 'cross_host',
    }
    text = aliases.get(text, text)
    return text if text in _COMMUNICATION_SCOPES else 'unknown'


def document_platform(document: Mapping[str, Any]) -> str:
    direct = normalize_platform(document.get('platform'))
    if direct != 'unknown':
        return direct
    card = str(document.get('target_card') or '').strip().lower()
    arch = str(document.get('target_arch') or document.get('arch') or '').strip().lower()
    if card == 'mi308x' or arch == 'gfx942':
        return 'amd'
    if card in ('h20', 'b300') or arch.startswith('sm_'):
        return 'nv'
    if document.get('source_sqlite') is not None:
        return 'nv'
    if str(document.get('schema') or '') == 'gpu-hw-metrics/v1':
        return 'nv'
    return 'unknown'


def document_communication_scope(document: Mapping[str, Any]) -> str:
    direct = normalize_communication_scope(document.get('communication_scope'))
    if direct != 'unknown':
        return direct
    contract = document.get('capture_contract')
    if isinstance(contract, Mapping):
        return normalize_communication_scope(contract.get('communication_scope'))
    return 'unknown'


def filter_utilization_evidence_to_window(
        evidence_document: Optional[Mapping[str, Any]], start_ns: int,
        end_ns: int) -> Optional[Mapping[str, Any]]:
    """Clip explicit normalized evidence samples to the selected launch.

    Entries whose samples existed but all fell outside the window are marked so
    the evaluator cannot fall back to a whole-capture scalar value.
    """
    if not isinstance(evidence_document, Mapping):
        return evidence_document
    wrapper = isinstance(evidence_document.get('utilization_evidence'), Mapping)
    evidence = (evidence_document.get('utilization_evidence')
                if wrapper else evidence_document)
    if not isinstance(evidence, Mapping):
        return evidence_document
    filtered: Dict[str, Any] = {}
    for metric_id in EVIDENCE_METRIC_IDS:
        raw = evidence.get(metric_id)
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        original_samples = raw.get('samples')
        if isinstance(original_samples, list):
            clipped = []
            for sample in original_samples:
                if not isinstance(sample, (list, tuple)) or len(sample) < 2:
                    continue
                try:
                    timestamp = int(sample[0])
                except (TypeError, ValueError):
                    continue
                if int(start_ns) <= timestamp <= int(end_ns):
                    clipped.append([timestamp, sample[1]])
            item['samples'] = clipped
            item['_window_filtered'] = True
        filtered[metric_id] = item
    if wrapper:
        result = dict(evidence_document)
        result['utilization_evidence'] = filtered
        return result
    return filtered


def filter_counters_to_window(counters: Iterable[Any], start_ns: int,
                              end_ns: int) -> List[Dict[str, Any]]:
    """Return shallow counter copies clipped to one launch's strict window."""
    clipped: List[Dict[str, Any]] = []
    for raw in counters:
        if not isinstance(raw, Mapping):
            continue
        samples = []
        for sample in raw.get('samples') or []:
            if not isinstance(sample, (list, tuple)) or len(sample) < 2:
                continue
            try:
                timestamp = int(sample[0])
            except (TypeError, ValueError):
                continue
            if int(start_ns) <= timestamp <= int(end_ns):
                samples.append([timestamp, sample[1]])
        item = dict(raw)
        item['samples'] = samples
        clipped.append(item)
    return clipped


def _base_name(counter: Mapping[str, Any]) -> str:
    name = counter.get('name') or counter.get('raw_name') or ''
    return _UNIT_SUFFIX_RE.sub('', str(name)).strip()


def _normalized_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', str(name).lower()).strip()


def _is_percent_counter(counter: Mapping[str, Any]) -> bool:
    unit = str(counter.get('unit') or '').strip().lower()
    return '%' in unit or 'percent' in unit


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _samples(counter: Mapping[str, Any]) -> List[List[float]]:
    by_timestamp: Dict[int, float] = {}
    for sample in counter.get('samples') or []:
        if not isinstance(sample, (list, tuple)) or len(sample) < 2:
            continue
        try:
            timestamp = int(sample[0])
        except (TypeError, ValueError):
            continue
        value = _finite_number(sample[1])
        if value is None:
            continue
        by_timestamp[timestamp] = value
    return [[timestamp, by_timestamp[timestamp]] for timestamp in sorted(by_timestamp)]


def _mean(samples: Sequence[Sequence[float]]) -> Optional[float]:
    if not samples:
        return None
    values = [float(sample[1]) for sample in samples]
    return round(sum(values) / len(values), 6)


def _available(metric_id: str, samples: Sequence[Sequence[float]], *,
               source: str, evidence_status: str, method: str, scope: str,
               observed_names: Sequence[str] = (),
               extra: Optional[Mapping[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    value = _mean(samples)
    if value is None:
        return _unavailable(metric_id, source=source,
                            reason='no finite samples remain after validation')
    entry: Dict[str, Any] = {
        'semantic_id': metric_id,
        'display_name': metric_id,
        'value': value,
        'unit': UNIT,
        'source': source,
        'evidence_status': evidence_status,
        'status': 'available',
        'reason': None,
        'method': method,
        'scope': scope,
        'sample_count': len(samples),
        'observed_names': list(observed_names),
    }
    if extra:
        entry.update(extra)
    series = {
        'name': metric_id,
        'samples': [[int(ts), float(val)] for ts, val in samples],
        'source': source,
        'evidence_status': evidence_status,
        'scope': scope,
    }
    if extra:
        series.update({key: value for key, value in extra.items()
                       if key in ('cross_host_claim_allowed', 'evidence_kind',
                                  'timeline_label', 'native_metric', 'compute_pipe',
                                  'parent', 'auxiliary', 'semantic_route', 'route_basis')})
    return entry, series


def _unavailable(metric_id: str, *, source: str, reason: str,
                 scope: str = 'unavailable') -> Tuple[Dict[str, Any], Dict[str, Any]]:
    entry = {
        'semantic_id': metric_id,
        'display_name': metric_id,
        'value': None,
        'unit': UNIT,
        'source': source,
        'evidence_status': 'unavailable',
        'status': 'unavailable',
        'reason': str(reason),
        'method': None,
        'scope': scope,
        'sample_count': 0,
        'observed_names': [],
    }
    return entry, {}


def _counter_by_names(counters: Sequence[Mapping[str, Any]],
                      names: Sequence[str]) -> Optional[Mapping[str, Any]]:
    wanted = {_normalized_name(name) for name in names}
    for counter in counters:
        if _normalized_name(_base_name(counter)) in wanted:
            return counter
    return None


def _single_counter_metric(metric_id: str, counters: Sequence[Mapping[str, Any]], *,
                           names: Sequence[str], source_prefix: str
                           ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    counter = _counter_by_names(counters, names)
    source = '{}[{}]'.format(source_prefix, '|'.join(names))
    if counter is None:
        return _unavailable(metric_id, source=source,
                            reason='no matching observed counter: {}'.format(list(names)))
    observed = _base_name(counter)
    if not _is_percent_counter(counter):
        return _unavailable(
            metric_id, source='{}:{}'.format(source_prefix, observed),
            reason='counter unit {!r} is not percent_of_peak'.format(counter.get('unit')))
    samples = _samples(counter)
    if not samples:
        return _unavailable(
            metric_id, source='{}:{}'.format(source_prefix, observed),
            reason='matching counter has no finite samples in the selected kernel window')
    return _available(
        metric_id, samples,
        source='{}:{}'.format(source_prefix, observed),
        evidence_status='measured',
        method='arithmetic_mean_of_clipped_samples',
        scope='device',
        observed_names=[observed])


def _memory_metric(counters: Sequence[Mapping[str, Any]], source_prefix: str
                   ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    metric_id = 'memory_util'
    combined = _counter_by_names(counters, (
        'DRAM Bandwidth Total', 'DRAM Throughput', 'Memory Throughput'))
    if combined is not None:
        if not _is_percent_counter(combined):
            return _unavailable(
                metric_id,
                source='{}:{}'.format(source_prefix, _base_name(combined)),
                reason='fallback memory counter unit {!r} is not percent_of_peak'
                       .format(combined.get('unit')))
        samples = _samples(combined)
        if samples:
            return _available(
                metric_id, samples,
                source='{}:{}'.format(source_prefix, _base_name(combined)),
                evidence_status='measured',
                method='arithmetic_mean_of_existing_combined_memory_counter',
                scope='device',
                observed_names=[_base_name(combined)])
        return _unavailable(
            metric_id,
            source='{}:{}'.format(source_prefix, _base_name(combined)),
            reason='fallback memory counter has no finite samples in the selected kernel window')
    read = _counter_by_names(counters, (
        'DRAM Read Bandwidth', 'DRAM Read Throughput', 'Memory Read Throughput'))
    write = _counter_by_names(counters, (
        'DRAM Write Bandwidth', 'DRAM Write Throughput', 'Memory Write Throughput'))
    useful = [(name, counter) for name, counter in (('read', read), ('write', write))
              if counter is not None and _is_percent_counter(counter)]
    if useful:
        series_by_direction: Dict[str, Dict[int, float]] = {}
        observed = []
        for direction, counter in useful:
            direction_samples = _samples(counter)
            if not direction_samples:
                continue
            observed.append(_base_name(counter))
            series_by_direction[direction] = {
                int(ts): float(value) for ts, value in direction_samples}
        if not series_by_direction:
            return _unavailable(
                metric_id,
                source='{}[{}]'.format(source_prefix, '|'.join(observed) or 'DRAM read/write'),
                reason='matching memory counters have no finite samples in the selected kernel window')
        timestamps = sorted({ts for values in series_by_direction.values() for ts in values})
        merged = []
        for timestamp in timestamps:
            values = [values[timestamp] for values in series_by_direction.values()
                      if timestamp in values]
            if values:
                merged.append([timestamp, sum(values)])
        evidence_status = 'derived' if len(series_by_direction) > 1 else 'measured'
        method = ('per-timestamp sum(read,write); directional counters remain in the trace'
                  if len(series_by_direction) > 1 else
                  'arithmetic_mean_of_one_directional_counter')
        return _available(
            metric_id, merged,
            source='{}[{}]; per-sample sum'.format(source_prefix, '+'.join(observed)),
            evidence_status=evidence_status,
            method=method,
            scope='device',
            observed_names=observed,
            extra={'directional_sample_counts': {
                direction: len(values) for direction, values in series_by_direction.items()}})

    return _unavailable(
        metric_id,
        source='{}[DRAM Read Bandwidth|DRAM Write Bandwidth]'.format(source_prefix),
        reason='no normalized DRAM/memory utilization counter is present')


def _evidence_kind(entry: Mapping[str, Any]) -> str:
    explicit = str(entry.get('evidence_kind') or '').strip().lower()
    if explicit:
        return explicit
    source = str(entry.get('source') or '').lower()
    if _NIC_RE.search(source):
        return 'nic'
    if _NVLINK_RE.search(source):
        return 'nvlink'
    if _XGMI_RE.search(source):
        return 'xgmi'
    return 'unknown'


def _explicit_metric(metric_id: str, evidence_document: Optional[Mapping[str, Any]],
                     requested_scope: str
                     ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    if not isinstance(evidence_document, Mapping):
        return None
    evidence = (evidence_document.get('utilization_evidence')
                if isinstance(evidence_document.get('utilization_evidence'), Mapping)
                else evidence_document)
    if not isinstance(evidence, Mapping):
        return None
    raw = evidence.get(metric_id)
    if not isinstance(raw, Mapping):
        return None
    source = str(raw.get('source') or f'explicit_evidence:{metric_id}')
    status = str(raw.get('evidence_status') or 'measured').strip().lower()
    if status == 'unavailable':
        return _unavailable(
            metric_id, source=source,
            reason=str(raw.get('reason') or 'explicit evidence marks this metric unavailable'))
    if status not in _EVIDENCE_STATUSES:
        return _unavailable(metric_id, source=source,
                            reason='invalid explicit evidence_status {!r}'.format(status))

    unit = str(raw.get('unit') or UNIT).strip().lower()
    if '%' not in unit and 'percent' not in unit:
        return _unavailable(
            metric_id, source=source,
            reason='explicit evidence unit {!r} is not a normalized percentage'
                   .format(raw.get('unit')))
    scope = normalize_communication_scope(raw.get('scope') or requested_scope)
    kind = _evidence_kind(raw)
    extra: Dict[str, Any] = {
        'evidence_kind': kind,
        'cross_host_claim_allowed': kind in _NIC_EVIDENCE_KINDS,
    }
    for key in ('timeline_label', 'native_metric', 'compute_pipe', 'parent',
                'auxiliary', 'semantic_route', 'route_basis'):
        if key in raw:
            extra[key] = raw[key]
    if metric_id == 'communication_util':
        if scope == 'cross_host' and kind not in _NIC_EVIDENCE_KINDS:
            return _unavailable(
                metric_id, source=source, scope='cross_host',
                reason='cross-host communication requires NIC/RDMA/fabric evidence; '
                       'observed evidence kind is {!r}'.format(kind))
        if scope == 'unknown':
            scope = 'fabric' if kind in _NIC_EVIDENCE_KINDS else (
                'same_host' if kind in _LINK_EVIDENCE_KINDS else 'unknown')
        extra['cross_host_claim_allowed'] = kind in _NIC_EVIDENCE_KINDS
    else:
        scope = 'device'

    samples: List[List[float]] = []
    for sample in raw.get('samples') or []:
        if not isinstance(sample, (list, tuple)) or len(sample) < 2:
            continue
        number = _finite_number(sample[1])
        if number is None:
            continue
        try:
            timestamp = int(sample[0])
        except (TypeError, ValueError):
            continue
        samples.append([timestamp, number])
    if samples:
        return _available(
            metric_id, samples, source=source, evidence_status=status,
            method=str(raw.get('method') or 'explicit_normalized_utilization_samples'),
            scope=scope, observed_names=[source], extra=extra)
    if raw.get('_window_filtered'):
        return _unavailable(
            metric_id, source=source, scope=scope,
            reason='explicit evidence has no finite samples in the selected kernel window')
    explicit_value = _finite_number(raw.get('value'))
    if explicit_value is None:
        return _unavailable(
            metric_id, source=source,
            reason='explicit evidence has neither a finite value nor finite samples')
    entry = {
        'semantic_id': metric_id,
        'display_name': metric_id,
        'value': round(explicit_value, 6),
        'unit': str(raw.get('unit') or UNIT),
        'source': source,
        'evidence_status': status,
        'status': 'available',
        'reason': None,
        'method': str(raw.get('method') or 'explicit_normalized_utilization_value'),
        'scope': scope,
        'sample_count': 0,
        'observed_names': [source],
    }
    entry.update(extra)
    return entry, {}


def _communication_metric(counters: Sequence[Mapping[str, Any]], *,
                          requested_scope: str, source_prefix: str
                          ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    metric_id = 'communication_util'
    nic = [counter for counter in counters
           if _NIC_RE.search(_base_name(counter)) and _is_percent_counter(counter)]
    link = [counter for counter in counters
            if (_NVLINK_RE.search(_base_name(counter))
                or _XGMI_RE.search(_base_name(counter)))
            and _is_percent_counter(counter)]
    chosen = nic or link
    observed = [_base_name(counter) for counter in chosen]
    if requested_scope == 'cross_host' and not nic:
        return _unavailable(
            metric_id,
            source='{}[{}]'.format(source_prefix, '+'.join(observed) or 'NIC/RDMA/fabric'),
            scope='cross_host',
            reason='cross-host communication requires NIC/RDMA/fabric utilization evidence; '
                   'GPU NVLink/XGMI counters cannot prove RDMA/fabric traffic')
    if not chosen:
        return _unavailable(
            metric_id,
            source='{}[NIC/RDMA/fabric or NVLink/XGMI]'.format(source_prefix),
            reason='no normalized NIC/RDMA/fabric or NVLink/XGMI utilization counter is present',
            scope=requested_scope)
    series: Dict[int, float] = {}
    useful_names = []
    for counter in chosen:
        samples = _samples(counter)
        if not samples:
            continue
        useful_names.append(_base_name(counter))
        for timestamp, value in samples:
            series[int(timestamp)] = max(series.get(int(timestamp), float('-inf')), float(value))
    if not series:
        return _unavailable(
            metric_id, source='{}[{}]'.format(source_prefix, '+'.join(observed)),
            scope=requested_scope,
            reason='matching interconnect counters have no finite samples in the selected kernel window')
    merged = [[timestamp, series[timestamp]] for timestamp in sorted(series)]
    if nic:
        evidence_kind = 'nic'
        scope = 'fabric'
        cross_host_allowed = True
    else:
        evidence_kind = 'nvlink' if any(_NVLINK_RE.search(name) for name in useful_names) else 'xgmi'
        scope = 'same_host'
        cross_host_allowed = False
    return _available(
        metric_id, merged,
        source='{}[{}]; per-sample max'.format(source_prefix, '+'.join(useful_names)),
        evidence_status='derived' if len(useful_names) > 1 else 'measured',
        method=('per-timestamp max across reported directions/counters; directional raw '
                'counters remain in the trace'),
        scope=scope,
        observed_names=useful_names,
        extra={'evidence_kind': evidence_kind,
               'cross_host_claim_allowed': cross_host_allowed,
               'coverage_scope': ('NIC/RDMA/fabric' if cross_host_allowed else
                                  'same-host GPU interconnect only')})


def evaluate_hardware_utilization(
        counters: Iterable[Any], *,
        platform: Any,
        communication_scope: Any = 'unknown',
        evidence_document: Optional[Mapping[str, Any]] = None,
        source_prefix: str = 'nsys.gpu_metrics',
        source_document: Optional[str] = None,
        target_card: Optional[str] = None,
        window_ns: Optional[Sequence[int]] = None
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Build the four-metric summary and optional timestamped canonical series."""
    normalized_platform = normalize_platform(platform)
    scope = normalize_communication_scope(communication_scope)
    counter_list = [counter for counter in counters if isinstance(counter, Mapping)]
    metrics: Dict[str, Dict[str, Any]] = {}
    series: Dict[str, Dict[str, Any]] = {}

    for metric_id in CANONICAL_METRIC_IDS:
        explicit = _explicit_metric(metric_id, evidence_document, scope)
        if explicit is not None:
            entry, metric_series = explicit
        elif normalized_platform == 'amd':
            if metric_id == 'communication_util':
                reason = ('AMD rocprofv3 PMC/XGMI data is not NIC/RDMA/fabric utilization; '
                          'cross-host communication requires an external NIC/RDMA/fabric counter')
                source = 'amd.rocprofv3.pmc[not_normalized]'
            elif metric_id in ('compute_util', 'memory_util', 'sm_active'):
                reason = ('AMD rocprofv3 PMC values are per-dispatch raw counts and are not '
                          'normalized percent_of_peak; no verified derivation and peak are available')
                source = 'amd.rocprofv3.pmc[not_normalized]'
            else:
                reason = 'AMD PMC cannot supply this canonical utilization metric'
                source = 'amd.rocprofv3.pmc[not_normalized]'
            entry, metric_series = _unavailable(
                metric_id, source=source,
                reason=reason, scope='unavailable')
        elif normalized_platform != 'nv':
            entry, metric_series = _unavailable(
                metric_id,
                source='{}[platform_unknown]'.format(source_prefix),
                reason='hardware platform is unknown; refusing to assign a utilization source')
        elif metric_id == 'compute_util':
            entry, metric_series = _single_counter_metric(
                metric_id, counter_list, source_prefix=source_prefix,
                names=('GPU Throughput', 'Compute (SM) Throughput', 'Compute Throughput',
                       'SM Issue'))
        elif metric_id == 'memory_util':
            entry, metric_series = _memory_metric(counter_list, source_prefix)
        elif metric_id == 'communication_util':
            entry, metric_series = _communication_metric(
                counter_list, requested_scope=scope, source_prefix=source_prefix)
        else:
            entry, metric_series = _single_counter_metric(
                metric_id, counter_list, source_prefix=source_prefix,
                names=('SMs Active', 'SM Active', 'SM Cycles Active'))
        metrics[metric_id] = entry
        if metric_series:
            series[metric_id] = metric_series

    available = [metric_id for metric_id in CANONICAL_METRIC_IDS
                 if metrics[metric_id].get('status') == 'available']
    summary: Dict[str, Any] = {
        'schema': SCHEMA,
        'policy': POLICY,
        'platform': normalized_platform,
        'communication_scope': scope,
        'source_prefix': source_prefix,
        'source_document': source_document,
        'target_card': target_card,
        'required_metric_ids': list(CANONICAL_METRIC_IDS),
        'all_available': len(available) == len(CANONICAL_METRIC_IDS),
        'available_metric_ids': available,
        'unavailable_metric_ids': [metric_id for metric_id in CANONICAL_METRIC_IDS
                                   if metric_id not in available],
        'metrics': metrics,
    }
    if window_ns is not None and len(window_ns) >= 2:
        summary['window_ns'] = [int(window_ns[0]), int(window_ns[1])]
    return summary, series
