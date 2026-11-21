import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
INFRA_DIR = ROOT / 'infra'
VERIFIER_DIR = INFRA_DIR / 'verifier'
ABI_PATH = ROOT / 'profile' / 'abi' / 'megamoe_profiler_abi.json'
for _extra_path in (str(INFRA_DIR), str(VERIFIER_DIR)):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

NUM_TRACKS = 7
MAX_RECORDS_LIMIT = 1 << 29
TIER_ANCHOR_OFF = 'anchor_off'
TIER_ANCHOR_ONLY = 'anchor_only'
TIER_FULL = 'full'
TIER_ORDER = (TIER_ANCHOR_OFF, TIER_ANCHOR_ONLY, TIER_FULL)
ANCHOR_MIN_MAX_RECORDS = 2
PRIMARY_METRIC = 'pure_record_overhead_pct'
LEGACY_METRIC = 'legacy_overhead_pct'
OVERHEAD_SCHEMA = 'megamoe.profile-overhead/three-tier-v1'
KNOWN_SM_COUNTS = (
    ('H20', 78),
    ('H200', 132),
    ('H100', 132),
    ('H800', 132),
    ('A100', 108),
)
FALLBACK_NUM_SMS = 78


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def profiler_numel(num_sms: int, max_records: int) -> int:
    return 1 + num_sms + num_sms * NUM_TRACKS * max_records * 2


def build_tiers(max_records: int, anchor_max_records: int, num_sms: int) -> List[Dict[str, Any]]:
    return [
        {
            'tier': 1,
            'name': TIER_ANCHOR_OFF,
            'time_symbol': 'T_none',
            'profiled': False,
            'profiler_buf': 'None',
            'max_records': 0,
            'buffer_numel': 0,
            'k_enable_profiler': False,
            'entry_rank_sync_executed': False,
            'record_capacity': 0,
            'description': ('profiler_buf=None -> kEnableProfiler=false -> fixed/kernel.cuh:192-220 的 '
                            'entry rank sync 整段被 if constexpr 编译掉，既无 rank sync 也无记录'),
        },
        {
            'tier': 2,
            'name': TIER_ANCHOR_ONLY,
            'time_symbol': 'T_anchor',
            'profiled': True,
            'profiler_buf': 'tensor',
            'max_records': anchor_max_records,
            'buffer_numel': profiler_numel(num_sms, anchor_max_records),
            'k_enable_profiler': True,
            'entry_rank_sync_executed': True,
            'record_capacity': num_sms * NUM_TRACKS * anchor_max_records,
            'description': (f'profiler_buf!=None -> kEnableProfiler=true -> entry rank sync 照常执行；'
                            f'max_records={anchor_max_records} 让 track 0 恰好记满 kEntryRankSync 的 '
                            f'start+end，其余 track 只记各自前 {anchor_max_records} 条，'
                            f'record() 在 offset>=max_records 时直接 return（满即丢弃、不回绕、无 atomic）'),
        },
        {
            'tier': 3,
            'name': TIER_FULL,
            'time_symbol': 'T_full',
            'profiled': True,
            'profiler_buf': 'tensor',
            'max_records': max_records,
            'buffer_numel': profiler_numel(num_sms, max_records),
            'k_enable_profiler': True,
            'entry_rank_sync_executed': True,
            'record_capacity': num_sms * NUM_TRACKS * max_records,
            'description': (f'profiler_buf!=None -> kEnableProfiler=true -> entry rank sync 照常执行；'
                            f'max_records={max_records} 为完整记录档，与档 2 共用同一份 JIT 产物'
                            f'（profiler_max_records 是运行时参数，不进 JIT key）'),
        },
    ]


def load_anchor_spec() -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        'abi_path': str(ABI_PATH),
        'abi_source': 'builtin-fallback',
        'task_name': 'kEntryRankSync',
        'task_id': 19,
        'track': 0,
        'start_semantics': 'rank_arrive',
        'end_semantics': 'rank_release',
        'task_lsb': 4,
        'task_mask': 31,
        'is_start_lsb': 3,
        'is_start_mask': 1,
        'block_lsb': 20,
        'block_mask': 4095,
        'track_lsb': 17,
        'track_mask': 7,
    }
    try:
        doc = json.loads(ABI_PATH.read_text())
    except (OSError, ValueError, TypeError):
        return spec
    anchor = ((doc.get('sync_anchors') or {}).get('entry_rendezvous') or {})
    fields = doc.get('tag_bitfields') or {}
    if not isinstance(anchor, dict) or not isinstance(fields, dict):
        return spec
    ok = True
    for key, source in (('task_name', 'task'), ('task_id', 'task_id'), ('track', 'track'),
                        ('start_semantics', 'start_semantics'), ('end_semantics', 'end_semantics')):
        value = anchor.get(source)
        if value is None:
            ok = False
        else:
            spec[key] = value
    for key, source in (('task', 'task'), ('is_start', 'is_start'),
                        ('block', 'block'), ('track', 'track')):
        field = fields.get(source)
        if not isinstance(field, dict) or 'lsb' not in field or 'mask' not in field:
            ok = False
            continue
        spec[f'{key}_lsb'] = int(field['lsb'])
        spec[f'{key}_mask'] = int(field['mask'])
    spec['abi_source'] = 'abi-json' if ok else 'abi-json-partial+builtin-fallback'
    spec['task_id'] = int(spec['task_id'])
    spec['track'] = int(anchor.get('track', spec['track']))
    return spec


def extract_entry_rank_sync(raw_cpu: torch.Tensor, num_sms: int, num_tracks: int,
                            max_records: int, spec: Dict[str, Any]) -> Dict[str, Any]:
    header = 1 + num_sms
    total = num_sms * num_tracks * max_records * 2
    records = raw_cpu[header:header + total].view(num_sms, num_tracks, max_records, 2)
    live_all = records[..., 0] != 0
    result: Dict[str, Any] = {
        'task': spec['task_name'],
        'task_id': spec['task_id'],
        'track': spec['track'],
        'start_semantics': spec['start_semantics'],
        'end_semantics': spec['end_semantics'],
        'abi_source': spec['abi_source'],
        'records_written_total': int(live_all.sum().item()),
        'record_capacity': num_sms * num_tracks * max_records,
        'records_written_per_track': [int(live_all[:, track].sum().item()) for track in range(num_tracks)],
    }
    anchor_track = int(spec['track'])
    timestamps = records[:, anchor_track, :, 0]
    tags = records[:, anchor_track, :, 1]
    live = timestamps != 0
    task_ids = torch.bitwise_and(torch.bitwise_right_shift(tags, spec['task_lsb']), spec['task_mask'])
    is_start = torch.bitwise_and(torch.bitwise_right_shift(tags, spec['is_start_lsb']), spec['is_start_mask'])
    tag_blocks = torch.bitwise_and(torch.bitwise_right_shift(tags, spec['block_lsb']), spec['block_mask'])
    tag_tracks = torch.bitwise_and(torch.bitwise_right_shift(tags, spec['track_lsb']), spec['track_mask'])
    hit = live & (task_ids == spec['task_id'])

    per_cta: List[Dict[str, Any]] = []
    missing: List[int] = []
    tag_mismatch: List[int] = []
    not_first_record: List[int] = []
    for block in range(num_sms):
        indices = hit[block].nonzero(as_tuple=False).flatten().tolist()
        starts = [index for index in indices if int(is_start[block, index].item()) == 1]
        ends = [index for index in indices if int(is_start[block, index].item()) == 0]
        if not starts or not ends:
            missing.append(block)
            continue
        start_index, end_index = starts[0], ends[0]
        if start_index != 0 or end_index != 1:
            not_first_record.append(block)
        if (int(tag_blocks[block, start_index].item()) != block
                or int(tag_tracks[block, start_index].item()) != anchor_track):
            tag_mismatch.append(block)
        start_ns = int(timestamps[block, start_index].item())
        end_ns = int(timestamps[block, end_index].item())
        per_cta.append({
            'block': block,
            'entry_sync_start_ns': start_ns,
            'entry_sync_end_ns': end_ns,
            'dur_ns': end_ns - start_ns,
            'start_record_offset': start_index,
            'end_record_offset': end_index,
            'occurrences': len(indices),
        })

    result.update({
        'ctas_with_anchor': len(per_cta),
        'ctas_missing_anchor_total': len(missing),
        'ctas_missing_anchor': missing[:200],
        'ctas_tag_mismatch': tag_mismatch[:200],
        'ctas_anchor_not_first_record': not_first_record[:200],
        'anchor_is_first_event_on_track': not not_first_record and not missing,
        'per_cta': per_cta,
    })
    if not per_cta:
        result['global'] = None
        return result

    durations = [entry['dur_ns'] for entry in per_cta]
    starts_ns = [entry['entry_sync_start_ns'] for entry in per_cta]
    ends_ns = [entry['entry_sync_end_ns'] for entry in per_cta]
    result['global'] = {
        'entry_sync_start_ns': min(starts_ns),
        'entry_sync_end_ns': max(ends_ns),
        'min_start_ns': min(starts_ns),
        'max_start_ns': max(starts_ns),
        'min_end_ns': min(ends_ns),
        'max_end_ns': max(ends_ns),
        'start_skew_ns': max(starts_ns) - min(starts_ns),
        'end_skew_ns': max(ends_ns) - min(ends_ns),
        'rendezvous_span_ns': max(ends_ns) - min(starts_ns),
        'dur_max_ns': max(durations),
        'dur_median_ns': int(statistics.median(durations)),
        'dur_min_ns': min(durations),
        'dur_mean_ns': sum(durations) / len(durations),
    }
    return result


def quantity(value: Any, unit: str, formula: str, window: str, definition: str,
             **extra: Any) -> Dict[str, Any]:
    payload = {
        'value': value,
        'unit': unit,
        'formula': formula,
        'window': window,
        'definition': definition,
    }
    payload.update(extra)
    return payload


def build_overhead_metrics(kernel_us: Dict[str, float], rank_sync_us: Dict[str, Optional[float]],
                           threshold_pct: float, rank_sync_statistic: str,
                           rank_sync_variants: Optional[Dict[str, Dict[str, Any]]] = None
                           ) -> Dict[str, Any]:
    t_none = kernel_us.get(TIER_ANCHOR_OFF)
    t_anchor = kernel_us.get(TIER_ANCHOR_ONLY)
    t_full = kernel_us.get(TIER_FULL)
    s_anchor = rank_sync_us.get(TIER_ANCHOR_ONLY)
    s_full = rank_sync_us.get(TIER_FULL)

    def sub(left: Optional[float], right: Optional[float]) -> float:
        if left is None or right is None:
            return float('nan')
        return left - right

    def pct(new: float, old: float) -> float:
        if not (isinstance(new, float) and isinstance(old, float)):
            return float('nan')
        if not math.isfinite(new) or not math.isfinite(old) or old <= 0.0:
            return float('nan')
        return (new - old) / old * 100.0

    rank_sync_cost = sub(t_anchor, t_none)
    pure_anchor = sub(t_anchor, s_anchor)
    pure_full = sub(t_full, s_full)
    primary = pct(pure_full, pure_anchor)
    legacy = pct(t_full, t_none) if isinstance(t_full, float) and isinstance(t_none, float) else float('nan')

    full_window = ('kineto 报告的 sm90_fp8_mega_moe kernel 全长（含 entry rank sync，'
                   '从 kernel 启动到最后一个 CTA 退出）')
    pure_window = ('kernel 全长减去 entry rank sync 时长，即 rank sync 的 release 锚点之后的'
                   '「纯 kernel 执行」窗口；release 锚点同时是精确时间原点')
    metrics = {
        'rank_sync_cost_us': quantity(
            rank_sync_cost, 'us', 'T_anchor - T_none',
            '两侧都是 kernel 全长；差值来自档 2 多执行的 entry rank sync',
            ('entry rank sync（kEntryRankSync + comm::nvlink_barrier grid-sync idx 2 + '
             'comm::grid_sync idx 3）整套机制在 kernel 全长口径下的成本。'
             '这是旧口径 3% 开销里被误算成「记录开销」的那一部分。'),
            tier_minuend=TIER_ANCHOR_ONLY, tier_subtrahend=TIER_ANCHOR_OFF),
        'pure_anchor_us': quantity(
            pure_anchor, 'us', 'T_anchor - S_anchor', pure_window,
            '档 2（有 rank sync、记录量可忽略）的纯 kernel 执行时间，作为纯记录开销的分母基准',
            tier=TIER_ANCHOR_ONLY),
        'pure_full_us': quantity(
            pure_full, 'us', 'T_full - S_full', pure_window,
            '档 3（有 rank sync、完整记录）的纯 kernel 执行时间',
            tier=TIER_FULL),
        PRIMARY_METRIC: quantity(
            primary, 'percent', '(pure_full_us - pure_anchor_us) / pure_anchor_us * 100',
            pure_window,
            ('主指标：纯记录开销。两侧都执行 entry rank sync 且共用同一份 JIT 产物，'
             '测量窗口都已剔除 rank sync，因此差值只剩 profiler 写记录本身的成本。'),
            role='primary', threshold_pct=threshold_pct,
            gating=True,
            within_threshold=bool(math.isfinite(primary) and primary < threshold_pct)
            if isinstance(primary, float) else False),
        LEGACY_METRIC: quantity(
            legacy, 'percent', '(T_full - T_none) / T_none * 100', full_window,
            ('旧口径，仅用于对照：baseline 是 profiler_buf=None，完全不执行 entry rank sync，'
             '所以该百分比 = 纯记录开销 + 整套 rank sync 成本，且两侧 CTA 启动抖动未对齐。'),
            role='legacy-reference-only', threshold_pct=threshold_pct,
            gating=False,
            within_threshold=bool(math.isfinite(legacy) and legacy < threshold_pct)
            if isinstance(legacy, float) else False),
    }
    metrics['rank_sync_statistic'] = {
        'value': rank_sync_statistic,
        'definition': ('S 的统计量选择：max = 各 CTA (entry_sync_end_ns - entry_sync_start_ns) 的最大值，'
                       '即最慢 CTA 的对齐耗时；median = 中位数；span = max_end_ns - min_start_ns，'
                       '把 CTA 启动抖动也算进 rank sync 窗口'),
        'applies_to': ['pure_anchor_us', 'pure_full_us', PRIMARY_METRIC],
    }
    if rank_sync_variants:
        sensitivity: Dict[str, Any] = {}
        anchor_variants = rank_sync_variants.get(TIER_ANCHOR_ONLY) or {}
        full_variants = rank_sync_variants.get(TIER_FULL) or {}
        for stat in ('max', 'median', 'span'):
            for aggregation in ('mean', 'median'):
                key = f'{aggregation}_of_{stat}'
                sa = (anchor_variants.get(stat) or {}).get(f'{aggregation}_us')
                sf = (full_variants.get(stat) or {}).get(f'{aggregation}_us')
                pa = sub(t_anchor, sa)
                pf = sub(t_full, sf)
                sensitivity[key] = {
                    'rank_sync_anchor_us': sa,
                    'rank_sync_full_us': sf,
                    'pure_anchor_us': pa,
                    'pure_full_us': pf,
                    'pure_record_overhead_pct': pct(pf, pa),
                    'selected': bool(stat == rank_sync_statistic and aggregation == 'mean'),
                }
        metrics['primary_metric_sensitivity'] = {
            'definition': ('主指标对 S 统计量选择的敏感性：层1 = 单次 launch 内跨 CTA 的统计量，'
                           '层2 = 跨 launch 的聚合方式。selected=true 的那一项即 '
                           f'{PRIMARY_METRIC} 采用的口径。'
                           'S 的分布重尾（典型数 us，偶发数百 us 的跨 rank 启动错位），'
                           'mean 口径无偏但方差大，median 口径抗离群但会低估 S 的期望；'
                           '若各口径下主指标结论一致，则该结论对 S 的统计量选择稳健。'),
            'combinations': sensitivity,
        }
    return metrics


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + '\n')


def run_nvidia_smi(query: str) -> Optional[List[List[str]]]:
    try:
        completed = subprocess.run(
            ['nvidia-smi', f'--query-{query}', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    rows = []
    for line in completed.stdout.strip().splitlines():
        line = line.strip()
        if not line or 'No running processes' in line or 'no running processes' in line:
            continue
        rows.append([field.strip() for field in line.split(',')])
    return rows


def query_gpus() -> Optional[List[Dict[str, Any]]]:
    rows = run_nvidia_smi('gpu=index,uuid,name,memory.used,memory.total,utilization.gpu')
    if rows is None:
        return None
    gpus = []
    for row in rows:
        if len(row) < 6:
            continue

        def _number(text: str) -> float:
            try:
                return float(text)
            except ValueError:
                return float('nan')

        gpus.append({
            'index': int(row[0]),
            'uuid': row[1],
            'name': row[2],
            'memory_used_mib': _number(row[3]),
            'memory_total_mib': _number(row[4]),
            'utilization_pct': _number(row[5]),
        })
    return gpus


def query_compute_apps() -> Optional[List[Dict[str, Any]]]:
    rows = run_nvidia_smi('compute-apps=gpu_uuid,pid,process_name,used_memory')
    if rows is None:
        return None
    apps = []
    for row in rows:
        if len(row) < 4:
            continue
        try:
            pid = int(row[1])
        except ValueError:
            continue
        try:
            used = float(row[3])
        except ValueError:
            used = float('nan')
        apps.append({'gpu_uuid': row[0], 'pid': pid, 'process_name': row[2], 'used_memory_mib': used})
    return apps


def resolve_target_devices(num_processes: int) -> Dict[str, Any]:
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None or visible.strip() == '':
        return {'selector': 'index', 'values': list(range(num_processes)), 'cuda_visible_devices': None}
    tokens = [token.strip() for token in visible.split(',') if token.strip() != '']
    if len(tokens) < num_processes:
        raise ValueError(
            f'CUDA_VISIBLE_DEVICES={visible!r} 只暴露 {len(tokens)} 个设备，少于 --num-processes {num_processes}')
    selected = tokens[:num_processes]
    if all(token.lstrip('-').isdigit() for token in selected):
        return {'selector': 'index', 'values': [int(token) for token in selected], 'cuda_visible_devices': visible}
    return {'selector': 'uuid', 'values': selected, 'cuda_visible_devices': visible}


def assumed_num_sms(gpu_name: Optional[str]) -> Dict[str, Any]:
    if gpu_name:
        upper = gpu_name.upper()
        for token, count in KNOWN_SM_COUNTS:
            if token in upper:
                return {'num_sms': count, 'source': f'name-table:{token}'}
    return {'num_sms': FALLBACK_NUM_SMS, 'source': 'fallback-default'}


def inspect_target_gpus(num_processes: int, mem_threshold_mib: float,
                        util_threshold_pct: float) -> Dict[str, Any]:
    target = resolve_target_devices(num_processes)
    gpus = query_gpus()
    apps = query_compute_apps()
    report: Dict[str, Any] = {
        'selector': target['selector'],
        'cuda_visible_devices': target['cuda_visible_devices'],
        'nvidia_smi_available': gpus is not None,
        'memory_threshold_mib': mem_threshold_mib,
        'utilization_threshold_pct': util_threshold_pct,
        'devices': [],
        'busy': False,
    }
    if gpus is None:
        report['error'] = 'nvidia-smi 不可用或返回失败，无法判断 GPU 占用'
        report['busy'] = True
        return report

    by_index = {gpu['index']: gpu for gpu in gpus}
    by_uuid = {gpu['uuid']: gpu for gpu in gpus}
    self_pid = os.getpid()
    for local_rank, value in enumerate(target['values']):
        gpu = by_index.get(value) if target['selector'] == 'index' else by_uuid.get(value)
        if gpu is None:
            report['devices'].append({
                'local_rank': local_rank, 'requested': value, 'found': False,
                'reasons': ['nvidia-smi 中找不到该设备'],
            })
            report['busy'] = True
            continue
        others = [app for app in (apps or []) if app['gpu_uuid'] == gpu['uuid'] and app['pid'] != self_pid]
        reasons = []
        if others:
            listed = ', '.join(f"pid={app['pid']}({app['used_memory_mib']:.0f}MiB)" for app in others)
            reasons.append(f'存在其他 compute 进程: {listed}')
        if gpu['memory_used_mib'] == gpu['memory_used_mib'] and gpu['memory_used_mib'] > mem_threshold_mib:
            reasons.append(f"已用显存 {gpu['memory_used_mib']:.0f}MiB > 阈值 {mem_threshold_mib:.0f}MiB")
        if gpu['utilization_pct'] == gpu['utilization_pct'] and gpu['utilization_pct'] > util_threshold_pct:
            reasons.append(f"利用率 {gpu['utilization_pct']:.0f}% > 阈值 {util_threshold_pct:.0f}%")
        report['devices'].append({
            'local_rank': local_rank, 'requested': value, 'found': True,
            'index': gpu['index'], 'uuid': gpu['uuid'], 'name': gpu['name'],
            'memory_used_mib': gpu['memory_used_mib'], 'memory_total_mib': gpu['memory_total_mib'],
            'utilization_pct': gpu['utilization_pct'],
            'other_processes': others,
            'reasons': reasons,
        })
        if reasons:
            report['busy'] = True
    return report


def gather_stack(tensor: torch.Tensor, num_ranks: int, group) -> torch.Tensor:
    parts = [torch.empty_like(tensor) for _ in range(num_ranks)]
    dist.all_gather(parts, tensor, group=group)
    stacked = torch.stack(parts, dim=0)
    del parts
    return stacked


def reference_output(refmod, x_fp8, x_sf, topk_idx, topk_weights,
                     l1_w_fp8, l1_w_sf, l2_w_fp8, l2_w_sf,
                     rank_idx: int, num_ranks: int, group,
                     num_experts: int, num_topk: int,
                     hidden: int, intermediate_hidden: int,
                     activation_clamp: float, chunk: int) -> torch.Tensor:
    from deep_gemm.utils.dist import uneven_all_gather

    num_experts_per_rank = num_experts // num_ranks
    x_fp8_g = uneven_all_gather(x_fp8, group=group)
    x_sf_g = uneven_all_gather(x_sf, group=group)
    topk_idx_g = uneven_all_gather(topk_idx, group=group)
    topk_w_g = uneven_all_gather(topk_weights, group=group)
    global_tokens = x_fp8_g.size(0)

    local_size = torch.tensor([x_fp8.size(0)], device='cuda', dtype=torch.long)
    sizes = torch.empty(num_ranks, dtype=torch.long, device='cuda')
    dist.all_gather_into_tensor(sizes, local_size, group=group)
    sizes_list = sizes.tolist()
    assert sum(sizes_list) == global_tokens

    l1_w_all = gather_stack(l1_w_fp8, num_ranks, group)
    l1_sf_all = gather_stack(l1_w_sf, num_ranks, group)
    l2_w_all = gather_stack(l2_w_fp8, num_ranks, group)
    l2_sf_all = gather_stack(l2_w_sf, num_ranks, group)

    combine_buf = torch.zeros(global_tokens, num_topk, hidden, dtype=torch.float32, device='cuda')
    x_fp32 = refmod._dequant_per_token_per_128_k(x_fp8_g, x_sf_g)

    for k in range(num_topk):
        mask = topk_idx_g[:, k] >= 0
        if not bool(mask.any()):
            continue
        selected_all = mask.nonzero(as_tuple=False).squeeze(-1)
        for begin in range(0, selected_all.numel(), chunk):
            selected = selected_all[begin:begin + chunk]
            expert_ids = topk_idx_g[selected, k]
            weights = topk_w_g[selected, k]
            x_selected = x_fp32[selected]
            dst_rank = (expert_ids // num_experts_per_rank).long()
            dst_local = (expert_ids % num_experts_per_rank).long()

            l1_w_selected = refmod._dequant_block_128_128(
                l1_w_all[dst_rank, dst_local], l1_sf_all[dst_rank, dst_local])
            l1_y = torch.einsum('sk,snk->sn', x_selected, l1_w_selected)
            del l1_w_selected
            l1_y = refmod._swiglu_fp32(l1_y, activation_clamp) * weights.unsqueeze(-1)

            rows, inner = l1_y.shape
            assert inner == intermediate_hidden and inner % 64 == 0
            l1_view = l1_y.view(rows, inner // 64, 64)
            amax = l1_view.abs().amax(dim=-1).clamp(1e-4)
            sf2 = amax / 448.0
            l1_q = (l1_view / sf2.unsqueeze(-1)).to(torch.float8_e4m3fn).float()
            l2_in = (l1_q * sf2.unsqueeze(-1)).view(rows, inner)

            l2_w_selected = refmod._dequant_block_128_128(
                l2_w_all[dst_rank, dst_local], l2_sf_all[dst_rank, dst_local])
            l2_y = torch.einsum('sn,smn->sm', l2_in, l2_w_selected)
            del l2_w_selected
            combine_buf[selected, k] = l2_y.to(torch.bfloat16).float()

    del l1_w_all, l1_sf_all, l2_w_all, l2_sf_all
    torch.cuda.empty_cache()
    y_global = combine_buf.to(torch.bfloat16).sum(dim=1).to(torch.bfloat16)
    start = sum(sizes_list[:rank_idx])
    return y_global[start:start + sizes_list[rank_idx]].contiguous()


def _worker(local_rank: int, num_local_ranks: int, cfg: Dict[str, Any]) -> None:
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp8
    from deep_gemm.utils.dist import init_dist, uneven_all_gather
    from deep_gemm.testing import bench_kineto, calc_diff, full_tensor_error_metrics, get_arch_major
    import test_mega_moe_sm90 as refmod

    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    out_dir = Path(cfg['out_dir'])

    if get_arch_major() != 9:
        print(f'[rank{rank_idx}] 需要 SM90，实测 SM{get_arch_major()}0，终止', flush=True)
        dist.destroy_process_group()
        sys.exit(3)

    if cfg['num_sms'] > 0:
        deep_gemm.set_num_sms(cfg['num_sms'])
    num_sms = deep_gemm.get_num_sms()
    max_records = cfg['max_records']
    anchor_max_records = cfg['anchor_only_max_records']
    tiers = build_tiers(max_records, anchor_max_records, num_sms)
    tier_by_name = {tier['name']: tier for tier in tiers}
    anchor_spec = load_anchor_spec()
    rank_sync_statistic = cfg['rank_sync_statistic']

    hidden = cfg['hidden']
    intermediate_hidden = cfg['intermediate_hidden']
    num_experts = cfg['num_experts']
    num_topk = cfg['num_topk']
    activation_clamp = cfg['activation_clamp']
    fast_math = bool(cfg['fast_math'])
    masked_ratio = cfg['masked_ratio']
    num_experts_per_rank = num_experts // num_ranks
    batches = cfg['batches']
    num_max_tokens_per_rank = max(batches)
    multi_batch = len(batches) > 1

    seed = cfg['seed'] + rank_idx * 1000003
    torch.manual_seed(seed)
    random.seed(seed)

    device_name = torch.cuda.get_device_name(local_rank)
    if rank_idx == 0:
        print(f'[profile] num_sms={num_sms} num_tracks={NUM_TRACKS} device={device_name} '
              f'anchor_task={anchor_spec["task_name"]}(id={anchor_spec["task_id"]},'
              f'track={anchor_spec["track"]},abi={anchor_spec["abi_source"]}) '
              f'rank_sync_statistic={rank_sync_statistic}', flush=True)
        for tier in tiers:
            print(f'[profile] 档{tier["tier"]} {tier["name"]:<11s} {tier["time_symbol"]:<8s} '
                  f'profiler_buf={tier["profiler_buf"]:<7s} max_records={tier["max_records"]:<6d} '
                  f'kEnableProfiler={str(tier["k_enable_profiler"]).lower():<5s} '
                  f'rank_sync={"yes" if tier["entry_rank_sync_executed"] else "no":<3s} '
                  f'record_capacity={tier["record_capacity"]:<8d} '
                  f'numel={tier["buffer_numel"]} '
                  f'({tier["buffer_numel"] * 8 / 2 ** 20:.1f}MiB)', flush=True)

    reference_bytes_per_token = 2 * intermediate_hidden * hidden * 4
    reference_chunk = cfg['reference_chunk']
    if reference_chunk <= 0:
        budget = int(cfg['reference_mem_budget_gb'] * (1 << 30))
        reference_chunk = max(1, budget // max(1, reference_bytes_per_token))
    reference_chunk = int(min(reference_chunk, max(1, num_max_tokens_per_rank * num_ranks)))

    raw_records: List[Dict[str, Any]] = []
    correctness_records: List[Dict[str, Any]] = []
    overhead_records: List[Dict[str, Any]] = []

    for num_tokens in batches:
        suffix = f'_t{num_tokens}' if multi_batch else ''
        raw_path = out_dir / f'raw_rank{rank_idx}{suffix}.pt'
        meta_path = out_dir / f'raw_rank{rank_idx}{suffix}.meta.json'

        x_bf = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        l1_bf = torch.randn((num_experts_per_rank, intermediate_hidden * 2, hidden),
                            dtype=torch.bfloat16, device='cuda') * 0.05
        l2_bf = torch.randn((num_experts_per_rank, hidden, intermediate_hidden),
                            dtype=torch.bfloat16, device='cuda') * 0.05
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        if masked_ratio > 0:
            rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
            topk_idx.masked_fill_(rand_mask < masked_ratio, -1)
            topk_weights.masked_fill_(topk_idx < 0, 0)

        x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
        l1_w_fp8, l1_w_sf = refmod._quantize_grouped_fp8_block_128_128(l1_bf)
        l2_w_fp8, l2_w_sf = refmod._quantize_grouped_fp8_block_128_128(l2_bf)
        del l1_bf, l2_bf, scores, x_bf
        torch.cuda.empty_cache()

        transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
            (l1_w_fp8, l1_w_sf), (l2_w_fp8, l2_w_sf))

        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden)
        tier_buffers = {
            TIER_ANCHOR_OFF: None,
            TIER_ANCHOR_ONLY: torch.zeros(tier_by_name[TIER_ANCHOR_ONLY]['buffer_numel'],
                                          dtype=torch.int64, device='cuda'),
            TIER_FULL: torch.zeros(tier_by_name[TIER_FULL]['buffer_numel'],
                                   dtype=torch.int64, device='cuda'),
        }
        cum_stats = torch.zeros(num_experts_per_rank, dtype=torch.int, device='cuda')

        def launch(tier_name: str, stats: torch.Tensor,
                   profiler_buf: Optional[torch.Tensor] = None) -> torch.Tensor:
            tier = tier_by_name[tier_name]
            buffer.x[:num_tokens].copy_(x_fp8)
            buffer.x_sf[:num_tokens].copy_(x_sf)
            buffer.topk_idx[:num_tokens].copy_(topk_idx)
            buffer.topk_weights[:num_tokens].copy_(topk_weights)
            y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            deep_gemm.fp8_mega_moe(
                y, transformed_l1, transformed_l2, buffer,
                cumulative_local_expert_recv_stats=stats,
                recipe=(128, 128, 128),
                activation='swiglu',
                activation_clamp=activation_clamp if math.isfinite(activation_clamp) else None,
                fast_math=fast_math,
                profiler_buf=(tier_buffers[tier_name] if profiler_buf is None else profiler_buf),
                profiler_max_records=tier['max_records'],
            )
            return y

        for tier in tiers:
            launch(tier['name'], cum_stats)
        torch.cuda.synchronize()
        dist.barrier()

        time_rounds = cfg['time_rounds']
        kernel_samples: Dict[str, List[float]] = {tier['name']: [] for tier in tiers}
        for round_index in range(time_rounds):
            shift = round_index % len(tiers)
            order = tiers[shift:] + tiers[:shift]
            for tier in order:
                seconds = bench_kineto(
                    lambda name=tier['name']: launch(name, cum_stats), 'sm90_fp8_mega_moe',
                    barrier=lambda: dist.barrier(), num_tests=cfg['num_tests'],
                    suppress_kineto_output=True)
                kernel_samples[tier['name']].append(seconds * 1e6)
                dist.barrier()
        kernel_us: Dict[str, float] = {
            name: statistics.median(values) for name, values in kernel_samples.items()
        }
        kernel_sample_stats = {
            name: {
                'num_rounds': len(values),
                'num_tests_per_round': cfg['num_tests'],
                'median_us': statistics.median(values),
                'mean_us': statistics.mean(values),
                'min_us': min(values),
                'max_us': max(values),
                'stdev_us': statistics.stdev(values) if len(values) > 1 else 0.0,
                'samples_us': values,
                'aggregation': 'median',
                'round_robin_order': ('每轮把档位顺序循环右移一位，使三档在时间上交替，'
                                      '外部负载漂移被均摊而不会被误认为档间差异'),
            }
            for name, values in kernel_samples.items()
        }

        anchors: Dict[str, Dict[str, Any]] = {}
        rank_sync_us: Dict[str, Optional[float]] = {TIER_ANCHOR_OFF: None}
        rank_sync_sample_stats: Dict[str, Dict[str, Any]] = {}
        rank_sync_variant_stats: Dict[str, Dict[str, Any]] = {}
        flush_l2_numel = int(8e9 // 4)
        rank_sync_samples = cfg['rank_sync_samples']

        def pick_rank_sync_us(anchor: Dict[str, Any]) -> Optional[float]:
            global_anchor = anchor.get('global')
            if global_anchor is None:
                return None
            return {
                'max': global_anchor['dur_max_ns'],
                'median': global_anchor['dur_median_ns'],
                'span': global_anchor['rendezvous_span_ns'],
            }[rank_sync_statistic] / 1e3

        sample_stats = torch.zeros(num_experts_per_rank, dtype=torch.int, device='cuda')
        sampled_tiers = (TIER_ANCHOR_ONLY, TIER_FULL)
        warmup_buffers = {
            tier_name: torch.zeros(tier_by_name[tier_name]['buffer_numel'],
                                   dtype=torch.int64, device='cuda')
            for tier_name in sampled_tiers
        }
        for _ in range(rank_sync_samples):
            for tier_name in sampled_tiers:
                torch.empty(flush_l2_numel, dtype=torch.int, device='cuda').zero_()
                torch.cuda._sleep(int(2e7))
                dist.barrier()
                launch(tier_name, sample_stats, profiler_buf=warmup_buffers[tier_name])
        torch.cuda.synchronize()
        dist.barrier()
        del warmup_buffers
        torch.cuda.empty_cache()

        sample_buffers = {
            tier_name: [torch.zeros(tier_by_name[tier_name]['buffer_numel'],
                                    dtype=torch.int64, device='cuda')
                        for _ in range(rank_sync_samples)]
            for tier_name in sampled_tiers
        }
        for index in range(rank_sync_samples):
            for tier_name in sampled_tiers:
                torch.empty(flush_l2_numel, dtype=torch.int, device='cuda').zero_()
                torch.cuda._sleep(int(2e7))
                dist.barrier()
                launch(tier_name, sample_stats, profiler_buf=sample_buffers[tier_name][index])
        torch.cuda.synchronize()
        dist.barrier()

        for tier_name in sampled_tiers:
            tier = tier_by_name[tier_name]
            samples: List[float] = []
            variant_samples: Dict[str, List[float]] = {'max': [], 'median': [], 'span': []}
            sample_integrity_ok = True
            for index in range(rank_sync_samples):
                sample_cpu = sample_buffers[tier_name][index].cpu()
                sample_anchor = extract_entry_rank_sync(
                    sample_cpu, num_sms, NUM_TRACKS, tier['max_records'], anchor_spec)
                global_anchor = sample_anchor.get('global')
                if global_anchor is not None:
                    variant_samples['max'].append(global_anchor['dur_max_ns'] / 1e3)
                    variant_samples['median'].append(global_anchor['dur_median_ns'] / 1e3)
                    variant_samples['span'].append(global_anchor['rendezvous_span_ns'] / 1e3)
                value = pick_rank_sync_us(sample_anchor)
                if value is not None:
                    samples.append(value)
                sample_integrity_ok = (sample_integrity_ok
                                       and sample_anchor['anchor_is_first_event_on_track'])
                del sample_cpu, sample_anchor
            rank_sync_variant_stats[tier_name] = {
                stat: {
                    'mean_us': statistics.mean(values) if values else None,
                    'median_us': statistics.median(values) if values else None,
                    'min_us': min(values) if values else None,
                    'max_us': max(values) if values else None,
                    'stdev_us': statistics.stdev(values) if len(values) > 1 else 0.0,
                }
                for stat, values in variant_samples.items()
            }
            rank_sync_us[tier_name] = statistics.mean(samples) if samples else None
            rank_sync_sample_stats[tier_name] = {
                'num_samples_requested': rank_sync_samples,
                'num_warmup_launches': rank_sync_samples,
                'num_samples_valid': len(samples),
                'mean_us': statistics.mean(samples) if samples else None,
                'median_us': statistics.median(samples) if samples else None,
                'min_us': min(samples) if samples else None,
                'max_us': max(samples) if samples else None,
                'stdev_us': statistics.stdev(samples) if len(samples) > 1 else 0.0,
                'samples_us': samples,
                'all_samples_anchor_complete': sample_integrity_ok,
                'per_launch_statistic': rank_sync_statistic,
                'across_launch_aggregation': 'mean',
                'iteration_cadence': ('采样循环只做 8GB L2 flush + torch.cuda._sleep(2e7) + '
                                      'dist.barrier() + launch；每次 launch 写入独立预分配 buffer，'
                                      '回拷与解码全部推迟到循环结束后，避免 CPU 侧回拷打乱两 rank 的对齐。'
                                      '档 2 与档 3 在采样循环内逐次交替，使两档在时间上完全交错，'
                                      '外部负载漂移不会被误认为档间差异。'
                                      '正式采样前先跑等量的 warmup launch 并丢弃，'
                                      '对齐 bench_kineto 跑两轮、只统计第二轮的语义，剔除首次冷启动'),
            }
        del sample_buffers
        torch.cuda.empty_cache()
        dist.barrier()

        capture_stats = torch.zeros(num_experts_per_rank, dtype=torch.int, device='cuda')
        y_captured = None
        raw_cpu = None
        anchor_raw_cpu = None
        for tier_name in (TIER_ANCHOR_ONLY, TIER_FULL):
            tier = tier_by_name[tier_name]
            tier_buffers[tier_name].zero_()
            stats = capture_stats if tier_name == TIER_FULL else torch.zeros(
                num_experts_per_rank, dtype=torch.int, device='cuda')
            torch.cuda.synchronize()
            dist.barrier()
            y_tier = launch(tier_name, stats)
            torch.cuda.synchronize()
            dist.barrier()
            tier_cpu = tier_buffers[tier_name].cpu()
            anchors[tier_name] = extract_entry_rank_sync(
                tier_cpu, num_sms, NUM_TRACKS, tier['max_records'], anchor_spec)
            if tier_name == TIER_FULL:
                y_captured = y_tier
                raw_cpu = tier_cpu
            else:
                anchor_raw_cpu = tier_cpu
                del y_tier

        overhead_metrics = build_overhead_metrics(
            kernel_us, rank_sync_us, cfg['overhead_threshold_pct'], rank_sync_statistic,
            rank_sync_variants=rank_sync_variant_stats)
        primary_pct = overhead_metrics[PRIMARY_METRIC]['value']
        legacy_pct = overhead_metrics[LEGACY_METRIC]['value']
        overhead_entry = {
            'rank': rank_idx,
            'num_tokens': num_tokens,
            'num_tests': cfg['num_tests'],
            'time_rounds': time_rounds,
            'kernel_time_sampling': kernel_sample_stats,
            'rank_sync_samples': rank_sync_samples,
            'rank_sync_sampling': rank_sync_sample_stats,
            'rank_sync_variant_stats': rank_sync_variant_stats,
            'num_sms': num_sms,
            'schema': OVERHEAD_SCHEMA,
            'primary_metric': PRIMARY_METRIC,
            'legacy_metric': LEGACY_METRIC,
            'tiers': [
                {
                    'tier': tier['tier'],
                    'name': tier['name'],
                    'time_symbol': tier['time_symbol'],
                    'profiler_buf': tier['profiler_buf'],
                    'max_records': tier['max_records'],
                    'k_enable_profiler': tier['k_enable_profiler'],
                    'entry_rank_sync_executed': tier['entry_rank_sync_executed'],
                    'record_capacity': tier['record_capacity'],
                    'records_written': (anchors[tier['name']]['records_written_total']
                                        if tier['name'] in anchors else 0),
                    'records_written_fraction_of_full': (
                        anchors[tier['name']]['records_written_total']
                        / max(1, anchors[TIER_FULL]['records_written_total'])
                        if tier['name'] in anchors and TIER_FULL in anchors else 0.0),
                    'kernel_us': quantity(
                        kernel_us[tier['name']], 'us', f'bench_kineto({tier["name"]})',
                        'kineto 报告的 sm90_fp8_mega_moe kernel 全长平均值（含 entry rank sync）',
                        tier['description']),
                    'rank_sync_us': quantity(
                        rank_sync_us[tier['name']], 'us',
                        f'mean over {rank_sync_samples} launches of '
                        f'{rank_sync_statistic}(entry_sync_end_ns - entry_sync_start_ns) / 1e3',
                        'kEntryRankSync 的 rank_arrive -> rank_release 区间',
                        ('该档 entry rank sync 的时长 S，用于从 kernel 全长里剔除 rank sync。'
                         f'层1（单次 launch 跨 CTA）取 {rank_sync_statistic}，层2（跨 launch）取 mean，'
                         '采样节奏与 bench_kineto 一致，使 S 与 T 同口径。'
                         'None 表示该档不执行 rank sync'),
                        statistic=rank_sync_statistic,
                        num_samples=rank_sync_samples,
                        sample_distribution=rank_sync_sample_stats.get(tier['name'])),
                    'rank_sync_single_launch_variants_us': (
                        {
                            'max': anchors[tier['name']]['global']['dur_max_ns'] / 1e3,
                            'median': anchors[tier['name']]['global']['dur_median_ns'] / 1e3,
                            'min': anchors[tier['name']]['global']['dur_min_ns'] / 1e3,
                            'span': anchors[tier['name']]['global']['rendezvous_span_ns'] / 1e3,
                            'start_skew': anchors[tier['name']]['global']['start_skew_ns'] / 1e3,
                            'end_skew': anchors[tier['name']]['global']['end_skew_ns'] / 1e3,
                            'source': '存盘那一次 launch 的单次采样，非 S 的均值；仅用于与 raw buffer 对照',
                        }
                        if tier['name'] in anchors and anchors[tier['name']].get('global') else None),
                    'description': tier['description'],
                }
                for tier in tiers
            ],
            'metrics': overhead_metrics,
            'threshold_pct': cfg['overhead_threshold_pct'],
            'within_threshold': bool(overhead_metrics[PRIMARY_METRIC]['within_threshold']),
            'legacy_within_threshold': bool(overhead_metrics[LEGACY_METRIC]['within_threshold']),
            'anchor_integrity': {
                tier_name: {
                    'ctas_with_anchor': anchor['ctas_with_anchor'],
                    'ctas_missing_anchor_total': anchor['ctas_missing_anchor_total'],
                    'ctas_tag_mismatch': anchor['ctas_tag_mismatch'],
                    'ctas_anchor_not_first_record': anchor['ctas_anchor_not_first_record'],
                    'anchor_is_first_event_on_track': anchor['anchor_is_first_event_on_track'],
                    'records_written_total': anchor['records_written_total'],
                    'records_written_per_track': anchor['records_written_per_track'],
                    'record_capacity': anchor['record_capacity'],
                    'global': anchor['global'],
                }
                for tier_name, anchor in anchors.items()
            },
        }
        overhead_records.append(overhead_entry)

        anchor_ok = all(anchor['anchor_is_first_event_on_track'] for anchor in anchors.values())
        label = f'[rank{rank_idx}] tokens={num_tokens}'
        t_none = kernel_us[TIER_ANCHOR_OFF]
        t_anchor = kernel_us[TIER_ANCHOR_ONLY]
        print(f'{label} 三档测量 (T: {time_rounds} 轮交替 x bench_kineto num_tests={cfg["num_tests"]}, '
              f'取轮间 median; S: {rank_sync_samples} 次交替采样取 mean，单次跨 CTA 取 {rank_sync_statistic})',
              flush=True)
        print(f'{label}   {"档":<3s}{"name":<12s}{"symbol":<9s}{"max_records":>12s}'
              f'{"rank_sync":>10s}{"records":>9s}{"T (us)":>11s}{"T_std":>8s}{"S (us)":>10s}'
              f'{"S_std":>9s}{"T-S (us)":>11s}', flush=True)
        for tier in tiers:
            name = tier['name']
            sync_value = rank_sync_us[name]
            written = anchors[name]['records_written_total'] if name in anchors else 0
            sync_text = '-' if sync_value is None else f'{sync_value:.1f}'
            pure_text = '-' if sync_value is None else f'{kernel_us[name] - sync_value:.1f}'
            stdev = (rank_sync_sample_stats.get(name) or {}).get('stdev_us')
            stdev_text = '-' if not isinstance(stdev, (int, float)) else f'{stdev:.1f}'
            t_stdev = (kernel_sample_stats.get(name) or {}).get('stdev_us')
            t_stdev_text = '-' if not isinstance(t_stdev, (int, float)) else f'{t_stdev:.1f}'
            print(f'{label}   {tier["tier"]:<3d}{name:<12s}{tier["time_symbol"]:<9s}'
                  f'{tier["max_records"]:>12d}'
                  f'{("yes" if tier["entry_rank_sync_executed"] else "no"):>10s}'
                  f'{written:>9d}{kernel_us[name]:>11.1f}{t_stdev_text:>8s}{sync_text:>10s}'
                  f'{stdev_text:>9s}{pure_text:>11s}', flush=True)
        for tier in tiers:
            rounds = (kernel_sample_stats.get(tier['name']) or {}).get('samples_us') or []
            print(f'{label}     {tier["name"]:<12s} 轮次 T 样本: '
                  + ' '.join(f'{value:.1f}' for value in rounds), flush=True)
        print(f'{label}   rank_sync_cost_us = T_anchor - T_none = '
              f'{t_anchor:.1f} - {t_none:.1f} = '
              f'{overhead_metrics["rank_sync_cost_us"]["value"]:.1f} us', flush=True)
        print(f'{label}   pure_anchor_us = T_anchor - S_anchor = '
              f'{overhead_metrics["pure_anchor_us"]["value"]:.1f} us   '
              f'pure_full_us = T_full - S_full = '
              f'{overhead_metrics["pure_full_us"]["value"]:.1f} us', flush=True)
        print(f'{label}   >>> 主指标 {PRIMARY_METRIC} = {primary_pct:.3f}% '
              f'(窗口已剔除 rank sync，两侧都有 rank sync；阈值 {cfg["overhead_threshold_pct"]:.1f}%, '
              f'{"PASS" if overhead_entry["within_threshold"] else "FAIL"})', flush=True)
        print(f'{label}       对照 {LEGACY_METRIC} = {legacy_pct:.3f}% '
              f'(旧口径 = 纯记录开销 + 整套 rank sync 成本，仅参考不作为失败判据)', flush=True)
        sensitivity = (overhead_metrics.get('primary_metric_sensitivity') or {}).get('combinations') or {}
        if sensitivity:
            print(f'{label}   主指标口径敏感性 (S 的层1统计量 x 层2聚合):', flush=True)
            print(f'{label}     {"combination":<18s}{"S_anchor":>10s}{"S_full":>10s}'
                  f'{"pure_anchor":>13s}{"pure_full":>11s}{"pure_record%":>14s}{"":>4s}', flush=True)
            for key, item in sensitivity.items():
                marker = ' <==' if item.get('selected') else ''

                def cell(value: Any, digits: int = 1) -> str:
                    return '-' if not isinstance(value, (int, float)) else f'{value:.{digits}f}'

                print(f'{label}     {key:<18s}'
                      f'{cell(item["rank_sync_anchor_us"]):>10s}{cell(item["rank_sync_full_us"]):>10s}'
                      f'{cell(item["pure_anchor_us"]):>13s}{cell(item["pure_full_us"]):>11s}'
                      f'{cell(item["pure_record_overhead_pct"], 3):>14s}{marker:>4s}', flush=True)
        if not overhead_entry['within_threshold']:
            print(f'{label} WARNING 主指标 {PRIMARY_METRIC} {primary_pct:.3f}% >= '
                  f'{cfg["overhead_threshold_pct"]:.1f}% 阈值', flush=True)
        if not overhead_entry['legacy_within_threshold']:
            print(f'{label} NOTE 旧口径 {LEGACY_METRIC} {legacy_pct:.3f}% >= '
                  f'{cfg["overhead_threshold_pct"]:.1f}%，仅提示（含 rank sync 成本，不作为失败）', flush=True)
        if not anchor_ok:
            for tier_name, anchor in anchors.items():
                if anchor['anchor_is_first_event_on_track']:
                    continue
                print(f'{label} WARNING {tier_name} 的 {anchor_spec["task_name"]} 锚点不完整: '
                      f'ctas_with_anchor={anchor["ctas_with_anchor"]}/{num_sms} '
                      f'missing={anchor["ctas_missing_anchor_total"]} '
                      f'not_first={len(anchor["ctas_anchor_not_first_record"])}', flush=True)

        anchor_raw_path = out_dir / f'raw_rank{rank_idx}{suffix}_anchor_only.pt'
        anchor_meta_path = out_dir / f'raw_rank{rank_idx}{suffix}_anchor_only.meta.json'
        torch.save(anchor_raw_cpu, anchor_raw_path)

        torch.save(raw_cpu, raw_path)
        nonzero_slots = int((raw_cpu[1 + num_sms:].view(-1, 2)[:, 0] != 0).sum().item())
        anchor_nonzero_slots = int(
            (anchor_raw_cpu[1 + num_sms:].view(-1, 2)[:, 0] != 0).sum().item())

        def build_meta(tier_name: str, numel_value: int, nonzero: int) -> Dict[str, Any]:
            tier = tier_by_name[tier_name]
            return {
                'rank': rank_idx,
                'device': local_rank,
                'device_name': device_name,
                'num_sms': num_sms,
                'num_tracks': NUM_TRACKS,
                'max_records': tier['max_records'],
                'numel': numel_value,
                'batches': batches,
                'num_tokens': num_tokens,
                'tier': tier['name'],
                'tier_index': tier['tier'],
                'tier_description': tier['description'],
                'shapes': {
                    'hidden': hidden,
                    'intermediate_hidden': intermediate_hidden,
                    'num_experts': num_experts,
                    'num_experts_per_rank': num_experts_per_rank,
                    'num_topk': num_topk,
                    'num_max_tokens_per_rank': num_max_tokens_per_rank,
                    'num_ranks': num_ranks,
                    'activation_clamp': activation_clamp,
                    'masked_ratio': masked_ratio,
                    'fast_math': fast_math,
                },
                'dg_mega_opt_version': os.environ.get('DG_MEGA_OPT_VERSION', 'v0'),
                'timestamp_utc': utc_now(),
                'header_elements': 1 + num_sms,
                'nonzero_tag_slots': nonzero,
                'seed': seed,
            }

        meta = build_meta(TIER_FULL, int(raw_cpu.numel()), nonzero_slots)
        dump_json(meta_path, meta)
        anchor_meta = build_meta(TIER_ANCHOR_ONLY, int(anchor_raw_cpu.numel()), anchor_nonzero_slots)
        dump_json(anchor_meta_path, anchor_meta)
        raw_records.append({'num_tokens': num_tokens, 'raw': str(raw_path), 'meta': str(meta_path),
                            'tier': TIER_FULL, 'nonzero_tag_slots': nonzero_slots})
        raw_records.append({'num_tokens': num_tokens, 'raw': str(anchor_raw_path),
                            'meta': str(anchor_meta_path), 'tier': TIER_ANCHOR_ONLY,
                            'nonzero_tag_slots': anchor_nonzero_slots})
        print(f'{label} 写出 {anchor_raw_path.name} (非空 tag 槽位 {anchor_nonzero_slots})', flush=True)
        print(f'[rank{rank_idx}] tokens={num_tokens} 写出 {raw_path.name} '
              f'(非空 tag 槽位 {nonzero_slots})', flush=True)

        y_ref = reference_output(
            refmod, x_fp8, x_sf, topk_idx, topk_weights,
            l1_w_fp8, l1_w_sf, l2_w_fp8, l2_w_sf,
            rank_idx, num_ranks, group, num_experts, num_topk,
            hidden, intermediate_hidden, activation_clamp, reference_chunk)
        reference_rms = float(y_ref.double().pow(2).mean().sqrt())
        atol_floor = cfg['elementwise_atol']
        atol_rms_scale = cfg['elementwise_atol_rms_scale']
        atol_effective = max(atol_floor, atol_rms_scale * reference_rms)
        metrics = full_tensor_error_metrics(
            y_captured, y_ref, atol=atol_effective, rtol=cfg['elementwise_rtol'])
        diff = float(calc_diff(y_captured, y_ref))

        global_topk_idx = uneven_all_gather(topk_idx, group=group)
        begin = rank_idx * num_experts_per_rank
        local_routes = global_topk_idx[(global_topk_idx >= begin) &
                                       (global_topk_idx < begin + num_experts_per_rank)]
        expected_stats = torch.bincount(local_routes - begin,
                                        minlength=num_experts_per_rank).to(dtype=capture_stats.dtype)
        mismatched = (capture_stats != expected_stats).nonzero(as_tuple=False).flatten().tolist()
        elementwise_passed = bool(metrics['all_elements_within_tolerance'])
        calc_diff_passed = bool(diff < cfg['diff_tol'])
        cum_stats_passed = not mismatched
        correctness_entry = {
            'rank': rank_idx,
            'num_tokens': num_tokens,
            'seed': seed,
            'reference_chunk': reference_chunk,
            'elementwise_atol': atol_effective,
            'elementwise_atol_floor': atol_floor,
            'elementwise_atol_rms_scale': atol_rms_scale,
            'reference_rms': reference_rms,
            'elementwise_rtol': cfg['elementwise_rtol'],
            'diff_tol': cfg['diff_tol'],
            'calc_diff': diff,
            'metrics': metrics,
            'metrics_nonfinite_fields': sorted(
                name for name, value in metrics.items()
                if isinstance(value, float) and not math.isfinite(value)),
            'cum_stats_sum': int(capture_stats.sum().item()),
            'cum_stats_expected_sum': int(expected_stats.sum().item()),
            'cum_stats_mismatched_experts': mismatched[:64],
            'cum_stats_num_mismatched_experts': len(mismatched),
            'elementwise_passed': elementwise_passed,
            'calc_diff_passed': calc_diff_passed,
            'cum_stats_passed': cum_stats_passed,
            'passed': bool(elementwise_passed and calc_diff_passed and cum_stats_passed),
        }
        correctness_records.append(correctness_entry)
        print(f'[rank{rank_idx}] tokens={num_tokens} numel={metrics["numel"]} '
              f'max_abs={metrics["max_abs"]:.6g} mean_abs={metrics["mean_abs"]:.6g} '
              f'fro_rel={metrics["frobenius_relative"]:.6g} '
              f'ref_rms={reference_rms:.6g} atol_eff={atol_effective:.6g} '
              f'violations={metrics["elementwise_violations"]} calc_diff={diff:.6g} '
              f'elementwise={elementwise_passed} calc_diff_ok={calc_diff_passed} '
              f'cum_stats_ok={cum_stats_passed} passed={correctness_entry["passed"]}', flush=True)

        del y_captured, y_ref, x_fp8, x_sf, topk_idx, topk_weights
        del l1_w_fp8, l1_w_sf, l2_w_fp8, l2_w_sf, transformed_l1, transformed_l2
        del tier_buffers, raw_cpu, anchor_raw_cpu
        buffer.destroy()
        torch.cuda.empty_cache()
        dist.barrier()

    dump_json(out_dir / f'correctness_rank{rank_idx}.json', correctness_records)
    dump_json(out_dir / f'overhead_rank{rank_idx}.json', overhead_records)
    dump_json(out_dir / f'files_rank{rank_idx}.json', raw_records)
    dist.barrier()

    if rank_idx == 0:
        aggregate_outputs(out_dir, num_ranks, cfg)

    dist.barrier()
    dist.destroy_process_group()


def aggregate_outputs(out_dir: Path, num_ranks: int, cfg: Dict[str, Any]) -> None:
    correctness: List[Dict[str, Any]] = []
    overhead: List[Dict[str, Any]] = []
    files: List[Dict[str, Any]] = []
    for rank in range(num_ranks):
        correctness += json.loads((out_dir / f'correctness_rank{rank}.json').read_text())
        overhead += json.loads((out_dir / f'overhead_rank{rank}.json').read_text())
        for record in json.loads((out_dir / f'files_rank{rank}.json').read_text()):
            for kind in ('raw', 'meta'):
                path = Path(record[kind])
                files.append({
                    'rank': rank, 'kind': kind, 'num_tokens': record['num_tokens'],
                    'tier': record.get('tier'),
                    'path': str(path), 'relative_path': path.name,
                    'bytes': path.stat().st_size, 'sha256': sha256_of(path),
                })

    def metric_values(name: str) -> List[float]:
        values = []
        for entry in overhead:
            item = ((entry.get('metrics') or {}).get(name) or {}).get('value')
            if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
                values.append(float(item))
        return values

    primary_values = metric_values(PRIMARY_METRIC)
    legacy_values = metric_values(LEGACY_METRIC)
    rank_sync_values = metric_values('rank_sync_cost_us')
    worst_primary = max(primary_values) if primary_values else None
    worst_legacy = max(legacy_values) if legacy_values else None
    threshold = cfg['overhead_threshold_pct']
    plan_num_sms = next((entry['num_sms'] for entry in overhead
                         if isinstance(entry.get('num_sms'), int) and entry['num_sms'] > 0),
                        cfg['num_sms'] if cfg['num_sms'] > 0 else FALLBACK_NUM_SMS)
    anchor_integrity_ok = all(
        all(item.get('anchor_is_first_event_on_track') for item in
            (entry.get('anchor_integrity') or {}).values())
        for entry in overhead) and bool(overhead)

    def sub_verdict(entry: Dict[str, Any], key: str) -> bool:
        if key in entry:
            return bool(entry[key])
        metrics = entry.get('metrics') or {}
        if key == 'elementwise_passed' and 'all_elements_within_tolerance' in metrics:
            return bool(metrics['all_elements_within_tolerance'])
        if key == 'calc_diff_passed':
            diff, tolerance = entry.get('calc_diff'), entry.get('diff_tol')
            if isinstance(diff, (int, float)) and isinstance(tolerance, (int, float)):
                return diff < tolerance
        if key == 'cum_stats_passed':
            count = entry.get('cum_stats_num_mismatched_experts')
            if isinstance(count, int):
                return count == 0
        return bool(entry.get('passed', False))

    def failed(entry: Dict[str, Any], key: str) -> bool:
        return not sub_verdict(entry, key)

    def sensitivity_summary() -> Dict[str, Any]:
        buckets: Dict[str, List[float]] = {}
        for entry in overhead:
            combinations = (((entry.get('metrics') or {}).get('primary_metric_sensitivity') or {})
                            .get('combinations') or {})
            for key, item in combinations.items():
                value = item.get('pure_record_overhead_pct')
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    buckets.setdefault(key, []).append(float(value))
        return {
            'definition': ('对每种 S 口径（层1跨CTA统计量 x 层2跨launch聚合），'
                           f'取所有 rank/batch 测量中 {PRIMARY_METRIC} 的最差值。'
                           '若所有口径都远低于阈值，则主指标结论对 S 的统计量选择稳健。'),
            'worst_pct_by_combination': {key: max(values) for key, values in buckets.items()},
            'selected_combination': f'mean_of_{cfg["rank_sync_statistic"]}',
        }

    overhead_doc = {
        'schema': OVERHEAD_SCHEMA,
        'threshold_pct': threshold,
        'num_measurements': len(overhead),
        'rank_sync_statistic': cfg['rank_sync_statistic'],
        'rank_sync_samples': cfg['rank_sync_samples'],
        'time_rounds': cfg['time_rounds'],
        'kernel_time_sampling_definition': (
            f'三档 kernel 时间交替测量 {cfg["time_rounds"]} 轮，每轮把档序循环右移一位，'
            f'每轮内每档跑 bench_kineto(num_tests={cfg["num_tests"]})，每档取各轮中位数。'
            '顺序测量时外部负载漂移会被误认为档间差异，交替+中位数用于抵消这一点。'),
        'worst_kernel_time_stdev_us': max(
            (item['stdev_us'] for entry in overhead
             for item in (entry.get('kernel_time_sampling') or {}).values()
             if isinstance(item.get('stdev_us'), (int, float))), default=None),
        'rank_sync_sampling_definition': (
            f'S 分两层统计：层1 在单次 launch 内跨 CTA 取 {cfg["rank_sync_statistic"]}，'
            f'层2 跨 {cfg["rank_sync_samples"]} 次 launch 取 mean。'
            '采样循环复现 bench_kineto 的迭代节奏（8GB L2 flush + cuda._sleep(2e7) + barrier），'
            '使 S 与 T 的跨 rank 启动偏差分布同口径。'),
        'worst_rank_sync_stdev_us': max(
            (item['stdev_us'] for entry in overhead
             for item in (entry.get('rank_sync_sampling') or {}).values()
             if isinstance(item.get('stdev_us'), (int, float))), default=None),
        'num_tests': cfg['num_tests'],
        'tier_plan': build_tiers(cfg['max_records'], cfg['anchor_only_max_records'], plan_num_sms),
        'primary_metric': {
            'name': PRIMARY_METRIC,
            'worst_pct': worst_primary,
            'num_measurements': len(primary_values),
            'threshold_pct': threshold,
            'within_threshold': bool(worst_primary is not None and worst_primary < threshold),
            'gating': True,
            'formula': '(pure_full_us - pure_anchor_us) / pure_anchor_us * 100',
            'window': ('两侧都执行 entry rank sync，且 kernel 全长各自减去本档实测的 rank sync 时长 S，'
                       '比较的是 rank sync release 锚点之后的纯 kernel 执行时间'),
            'definition': ('纯记录开销：档 3（完整记录）相对档 2（记录量可忽略）的纯 kernel 执行时间增幅。'
                           '档 2 与档 3 共用同一份 JIT 产物（profiler_max_records 不进 JIT key），'
                           '因此差值只剩 profiler 写记录本身的成本。'),
        },
        'legacy_metric': {
            'name': LEGACY_METRIC,
            'worst_pct': worst_legacy,
            'num_measurements': len(legacy_values),
            'threshold_pct': threshold,
            'within_threshold': bool(worst_legacy is not None and worst_legacy < threshold),
            'gating': False,
            'formula': '(T_full - T_none) / T_none * 100',
            'window': 'kernel 全长，不做任何剔除',
            'definition': ('旧口径，保留用于对照：baseline 档 1 是 profiler_buf=None，'
                           'kEnableProfiler=false 使 entry rank sync 整段被 if constexpr 编译掉，'
                           '所以该百分比 = 纯记录开销 + 整套 rank sync 成本，且两侧 CTA 启动抖动未对齐。'
                           '超阈值仅提示，不作为失败判据。'),
        },
        'rank_sync_cost_us': {
            'worst_us': max(rank_sync_values) if rank_sync_values else None,
            'formula': 'T_anchor - T_none',
            'window': 'kernel 全长之差',
            'definition': 'entry rank sync 整套机制的成本，即旧口径里被误算成记录开销的那一部分',
        },
        'anchor_integrity_ok': anchor_integrity_ok,
        'primary_metric_sensitivity': sensitivity_summary(),
        'anchor_integrity_definition': ('所有采样档的 track 0 都拿到了 kEntryRankSync 的完整 start+end 配对，'
                                        '且该配对占据 offset 0/1（每个 persistent CTA 的第一个语义事件）'),
        'within_threshold': bool(worst_primary is not None and worst_primary < threshold),
        'within_threshold_definition': f'由主指标 {PRIMARY_METRIC} 单独决定，旧口径不参与判定',
        'per_rank': overhead,
    }
    correctness_doc = {
        'all_passed': all(entry['passed'] for entry in correctness) and bool(correctness),
        'num_checks': len(correctness),
        'num_failed_elementwise': sum(1 for entry in correctness if failed(entry, 'elementwise_passed')),
        'num_failed_calc_diff': sum(1 for entry in correctness if failed(entry, 'calc_diff_passed')),
        'num_failed_cum_stats': sum(1 for entry in correctness if failed(entry, 'cum_stats_passed')),
        'per_rank': correctness,
    }
    dump_json(out_dir / 'overhead.json', overhead_doc)
    dump_json(out_dir / 'correctness.json', correctness_doc)

    manifest = {
        'run_id': cfg['run_id'],
        'created_utc': utc_now(),
        'out_dir': str(out_dir),
        'command_line': cfg['command_line'],
        'argv': cfg['argv'],
        'cwd': cfg['cwd'],
        'num_processes': num_ranks,
        'batches': cfg['batches'],
        'max_records': cfg['max_records'],
        'anchor_only_max_records': cfg['anchor_only_max_records'],
        'rank_sync_statistic': cfg['rank_sync_statistic'],
        'rank_sync_samples': cfg['rank_sync_samples'],
        'time_rounds': cfg['time_rounds'],
        'num_tests': cfg['num_tests'],
        'measurement_scheme': OVERHEAD_SCHEMA,
        'num_tracks': NUM_TRACKS,
        'shapes': {
            'hidden': cfg['hidden'],
            'intermediate_hidden': cfg['intermediate_hidden'],
            'num_experts': cfg['num_experts'],
            'num_topk': cfg['num_topk'],
            'num_max_tokens_per_rank': max(cfg['batches']),
            'activation_clamp': cfg['activation_clamp'],
            'masked_ratio': cfg['masked_ratio'],
            'fast_math': bool(cfg['fast_math']),
        },
        'environment': cfg['environment'],
        'torch_version': torch.__version__,
        'gpu_check': cfg['gpu_check'],
        'files': files,
        'overhead': overhead_doc,
        'correctness': correctness_doc,
    }
    dump_json(out_dir / 'run_manifest.json', manifest)
    print(f'[profile] manifest -> {out_dir / "run_manifest.json"}', flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='SM90 MegaMoE profiler 采集 harness：跑一次带 profiler 的 kernel 并把 raw buffer 存盘',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-processes', type=int, default=2, help='EP world size（SM90 EP2 用 2）')
    parser.add_argument('--batches', type=int, nargs='+', default=[32], help='num_tokens 列表')
    parser.add_argument('--max-records', type=int, default=4096, help='每 (block, track) 的最大记录条数')
    parser.add_argument('--anchor-only-max-records', type=int, default=ANCHOR_MIN_MAX_RECORDS,
                        help='档 2 anchor_only 的 max_records：只需装下 track 0 上 kEntryRankSync 的 '
                             'start+end 两条，即可做到「有 rank sync、记录量可忽略」')
    parser.add_argument('--rank-sync-statistic', type=str, default='max',
                        choices=('max', 'median', 'span'),
                        help='从 kernel 全长里剔除的 rank sync 时长 S 用哪个统计量：'
                             'max/median 为各 CTA (end-start) 的最大值/中位数，'
                             'span 为 max_end_ns - min_start_ns（把 CTA 启动抖动也算进去）')
    parser.add_argument('--rank-sync-samples', type=int, default=20,
                        help='档 2/档 3 各采样多少次 launch 求 S 的均值；'
                             'nvlink_barrier 的等待时长取决于两 rank 谁先到达，单次采样噪声很大，'
                             '需与 --num-tests 同量级才能让 S 与 T 统计口径一致')
    parser.add_argument('--time-rounds', type=int, default=3,
                        help='三档 kernel 时间的交替测量轮数：每轮把档序循环右移一位，'
                             '每档取各轮结果的中位数，用于抵消外部负载引起的时间漂移'
                             '（顺序测量时漂移会被误认为档间差异）')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='输出目录，默认 <ROOT>/.work/profile_runs/<run_id>')
    parser.add_argument('--run-id', type=str, default=None, help='默认使用 UTC 时间戳')
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--intermediate-hidden', type=int, default=2048)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--activation-clamp', type=float, default=10.0)
    parser.add_argument('--masked-ratio', type=float, default=0.0)
    parser.add_argument('--fast-math', type=int, default=1)
    parser.add_argument('--num-tests', type=int, default=20, help='baseline / instrumented 计时的重复次数')
    parser.add_argument('--num-sms', type=int, default=0,
                        help='0 表示使用设备实际 SM 数；>0 时调用 deep_gemm.set_num_sms')
    parser.add_argument('--seed', type=int, default=20260725)
    parser.add_argument('--diff-tol', type=float, default=0.07)
    parser.add_argument('--elementwise-atol', type=float, default=2.0,
                        help='逐元素绝对容差的下限；实际容差取 max(该值, --elementwise-atol-rms-scale * 参考输出 RMS)')
    parser.add_argument('--elementwise-atol-rms-scale', type=float, default=0.5,
                        help='逐元素绝对容差相对参考输出 RMS 的比例系数，用于让容差随输出量级一起缩放')
    parser.add_argument('--elementwise-rtol', type=float, default=0.20)
    parser.add_argument('--reference-chunk', type=int, default=0,
                        help='参考实现的 token 分块，0 表示按 --reference-mem-budget-gb 自动推算')
    parser.add_argument('--reference-mem-budget-gb', type=float, default=4.0)
    parser.add_argument('--overhead-threshold-pct', type=float, default=10.0,
                        help=f'主指标 {PRIMARY_METRIC} 的告警阈值；'
                             f'旧口径 {LEGACY_METRIC} 超标只提示，不作为失败')
    parser.add_argument('--gpu-mem-threshold-mib', type=float, default=1024.0)
    parser.add_argument('--gpu-util-threshold-pct', type=float, default=10.0)
    parser.add_argument('--force', action='store_true', help='目标 GPU 有占用时仍然继续')
    parser.add_argument('--dry-run', action='store_true',
                        help='只做参数校验 / 显存估算 / GPU 占用检查 / 目录规划，不启动 CUDA')
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.num_processes < 1:
        parser.error('--num-processes 必须 >= 1')
    if not args.batches or any(tokens <= 0 for tokens in args.batches):
        parser.error('--batches 必须都是正整数')
    if len(set(args.batches)) != len(args.batches):
        parser.error('--batches 不允许重复值')
    if not (0 < args.max_records < MAX_RECORDS_LIMIT):
        parser.error(f'--max-records 必须落在 (0, {MAX_RECORDS_LIMIT}) 区间')
    if not (ANCHOR_MIN_MAX_RECORDS <= args.anchor_only_max_records < MAX_RECORDS_LIMIT):
        parser.error(f'--anchor-only-max-records 必须落在 '
                     f'[{ANCHOR_MIN_MAX_RECORDS}, {MAX_RECORDS_LIMIT}) 区间：'
                     f'至少 {ANCHOR_MIN_MAX_RECORDS} 条才能装下 kEntryRankSync 的 start+end')
    if args.anchor_only_max_records > args.max_records:
        parser.error(f'--anchor-only-max-records {args.anchor_only_max_records} 不能大于 '
                     f'--max-records {args.max_records}，否则档 2 不再是记录量可忽略的档位')
    if args.num_experts % args.num_processes != 0:
        parser.error(f'--num-experts {args.num_experts} 必须能被 --num-processes {args.num_processes} 整除')
    if args.hidden % 128 != 0 or args.intermediate_hidden % 128 != 0:
        parser.error('--hidden 与 --intermediate-hidden 必须是 128 的倍数')
    if args.intermediate_hidden % 64 != 0:
        parser.error('--intermediate-hidden 必须是 64 的倍数')
    if not (1 <= args.num_topk <= args.num_experts):
        parser.error('--num-topk 必须落在 [1, num_experts]')
    if not (0.0 <= args.masked_ratio <= 1.0):
        parser.error('--masked-ratio 必须落在 [0, 1]')
    if args.num_tests < 1:
        parser.error('--num-tests 必须 >= 1')
    if args.rank_sync_samples < 1:
        parser.error('--rank-sync-samples 必须 >= 1')
    if args.time_rounds < 1:
        parser.error('--time-rounds 必须 >= 1')
    if args.num_sms < 0:
        parser.error('--num-sms 必须 >= 0')
    if (args.diff_tol < 0 or args.elementwise_atol < 0 or args.elementwise_rtol < 0
            or args.elementwise_atol_rms_scale < 0):
        parser.error('容差必须非负')
    if args.reference_mem_budget_gb <= 0:
        parser.error('--reference-mem-budget-gb 必须为正')
    if args.overhead_threshold_pct <= 0:
        parser.error('--overhead-threshold-pct 必须为正')


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    run_id = args.run_id or f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}_ep{args.num_processes}'
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (ROOT / '.work' / 'profile_runs' / run_id)

    try:
        gpu_check = inspect_target_gpus(args.num_processes, args.gpu_mem_threshold_mib,
                                        args.gpu_util_threshold_pct)
    except ValueError as error:
        parser.error(str(error))
        return 2

    device_name = None
    for device in gpu_check['devices']:
        if device.get('found'):
            device_name = device.get('name')
            break
    sm_guess = assumed_num_sms(device_name)
    num_sms_for_estimate = args.num_sms if args.num_sms > 0 else sm_guess['num_sms']
    sm_source = 'cli:--num-sms' if args.num_sms > 0 else sm_guess['source']
    tiers = build_tiers(args.max_records, args.anchor_only_max_records, num_sms_for_estimate)
    anchor_spec = load_anchor_spec()
    total_mib = sum(tier['buffer_numel'] for tier in tiers) * 8 / 2 ** 20

    print(f'[plan] run_id={run_id}')
    print(f'[plan] out_dir={out_dir}')
    print(f'[plan] num_processes={args.num_processes} batches={args.batches} '
          f'num_max_tokens_per_rank={max(args.batches)}')
    print(f'[plan] shapes hidden={args.hidden} intermediate_hidden={args.intermediate_hidden} '
          f'num_experts={args.num_experts} num_topk={args.num_topk} '
          f'activation_clamp={args.activation_clamp} masked_ratio={args.masked_ratio} '
          f'fast_math={bool(args.fast_math)}')
    print(f'[plan] profiler buffer: num_sms={num_sms_for_estimate} ({sm_source}) '
          f'num_tracks={NUM_TRACKS} num_tests={args.num_tests}')
    print(f'[plan] 锚点定义: task={anchor_spec["task_name"]} id={anchor_spec["task_id"]} '
          f'track={anchor_spec["track"]} '
          f'start={anchor_spec["start_semantics"]} end={anchor_spec["end_semantics"]} '
          f'(来源 {anchor_spec["abi_source"]}: {anchor_spec["abi_path"]})')
    print(f'[plan] rank_sync_statistic={args.rank_sync_statistic} '
          f'rank_sync_samples={args.rank_sync_samples} time_rounds={args.time_rounds}')
    print(f'[plan] T 交替测量: {args.time_rounds} 轮，每轮档序循环右移一位，'
          f'每轮每档 bench_kineto(num_tests={args.num_tests})，每档取轮间 median')
    print(f'[plan] S 的两层统计: 层1 单次 launch 内跨 CTA 取 {args.rank_sync_statistic}，'
          f'层2 跨 {args.rank_sync_samples} 次 launch 取 mean')
    print('[plan] S 采样节奏复现 bench_kineto: 8GB L2 flush + cuda._sleep(2e7) + barrier，'
          '使 S 与 T 的跨 rank 启动偏差分布同口径')
    print('[plan] 三档测量计划:')
    print(f'[plan]   {"档":<3s}{"name":<13s}{"symbol":<9s}{"profiler_buf":<14s}'
          f'{"max_records":>12s}{"kEnableProf":>13s}{"rank_sync":>10s}'
          f'{"record_cap":>12s}{"numel":>12s}{"MiB":>9s}')
    for tier in tiers:
        print(f'[plan]   {tier["tier"]:<3d}{tier["name"]:<13s}{tier["time_symbol"]:<9s}'
              f'{tier["profiler_buf"]:<14s}{tier["max_records"]:>12d}'
              f'{str(tier["k_enable_profiler"]).lower():>13s}'
              f'{("yes" if tier["entry_rank_sync_executed"] else "no"):>10s}'
              f'{tier["record_capacity"]:>12d}{tier["buffer_numel"]:>12d}'
              f'{tier["buffer_numel"] * 8 / 2 ** 20:>9.1f}')
    for tier in tiers:
        print(f'[plan]   档{tier["tier"]} {tier["name"]}: {tier["description"]}')
    anchor_tier = next(tier for tier in tiers if tier['name'] == TIER_ANCHOR_ONLY)
    full_tier = next(tier for tier in tiers if tier['name'] == TIER_FULL)
    ratio = (anchor_tier['record_capacity'] / full_tier['record_capacity'] * 100.0
             if full_tier['record_capacity'] else float('nan'))
    print(f'[plan]   档2/档3 记录容量比 = {anchor_tier["record_capacity"]}/'
          f'{full_tier["record_capacity"]} = {ratio:.2f}%')
    print(f'[plan]   profiler buffer 合计 {total_mib:.1f} MiB/rank, '
          f'{total_mib * args.num_processes:.1f} MiB total')
    print('[plan] 将要计算的指标:')
    print('[plan]   rank_sync_cost_us        = T_anchor - T_none          '
          '(entry rank sync 整套机制成本，kernel 全长口径)')
    print('[plan]   pure_anchor_us           = T_anchor - S_anchor        (档 2 纯 kernel 执行时间)')
    print('[plan]   pure_full_us             = T_full   - S_full          (档 3 纯 kernel 执行时间)')
    print(f'[plan]   {PRIMARY_METRIC:<24s} = (pure_full - pure_anchor) / pure_anchor * 100  '
          f'<- 主指标，阈值 {args.overhead_threshold_pct:.1f}%，超标告警')
    print(f'[plan]   {LEGACY_METRIC:<24s} = (T_full - T_none) / T_none * 100               '
          '<- 旧口径，仅对照，超标只提示不失败')
    print(f'[plan] DG_MEGA_OPT_VERSION={os.environ.get("DG_MEGA_OPT_VERSION", "v0 (未设置，C++ 默认)")}')
    print(f'[plan] CUDA_VISIBLE_DEVICES={gpu_check["cuda_visible_devices"]}')
    print(f'[plan] PYTHONPATH 追加 {INFRA_DIR}')
    print('[plan] 将要写出的文件:')
    multi_batch = len(args.batches) > 1
    for rank in range(args.num_processes):
        for tokens in args.batches:
            suffix = f'_t{tokens}' if multi_batch else ''
            print(f'[plan]   {out_dir / f"raw_rank{rank}{suffix}.pt"} (档3 {TIER_FULL})')
            print(f'[plan]   {out_dir / f"raw_rank{rank}{suffix}.meta.json"} (档3 {TIER_FULL})')
            print(f'[plan]   {out_dir / f"raw_rank{rank}{suffix}_anchor_only.pt"} '
                  f'(档2 {TIER_ANCHOR_ONLY})')
            print(f'[plan]   {out_dir / f"raw_rank{rank}{suffix}_anchor_only.meta.json"} '
                  f'(档2 {TIER_ANCHOR_ONLY})')
    for name in ('correctness.json', 'overhead.json', 'run_manifest.json'):
        print(f'[plan]   {out_dir / name}')

    print('[gpu] 目标设备占用检查:')
    if not gpu_check['nvidia_smi_available']:
        print(f'[gpu]   {gpu_check.get("error")}')
    for device in gpu_check['devices']:
        label = (f'local_rank={device["local_rank"]} requested={device["requested"]}')
        if not device.get('found'):
            print(f'[gpu]   {label} -> 未找到: {device["reasons"]}')
            continue
        state = 'BUSY' if device['reasons'] else 'idle'
        print(f'[gpu]   {label} index={device["index"]} name={device["name"]} '
              f'mem={device["memory_used_mib"]:.0f}/{device["memory_total_mib"]:.0f}MiB '
              f'util={device["utilization_pct"]:.0f}% -> {state}')
        for reason in device['reasons']:
            print(f'[gpu]     - {reason}')

    if gpu_check['busy']:
        if args.force:
            print('[gpu] WARNING 目标 GPU 有占用，但 --force 已指定，继续执行')
        else:
            print('[gpu] ERROR 目标 GPU 被占用（不会 kill 任何进程）。等待空闲后重跑，或加 --force 强制继续。')
            return 2

    if args.dry_run:
        print('[dry-run] 参数校验 / 显存估算 / GPU 检查 / 目录规划全部完成，未启动 CUDA，未写任何文件')
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = os.environ.get('PYTHONPATH', '')
    parts = [str(INFRA_DIR)] + ([existing] if existing else [])
    os.environ['PYTHONPATH'] = os.pathsep.join(parts)

    cfg = {
        'run_id': run_id,
        'out_dir': str(out_dir),
        'command_line': ' '.join([sys.executable, '-B'] + sys.argv),
        'argv': list(sys.argv),
        'cwd': os.getcwd(),
        'batches': list(args.batches),
        'max_records': args.max_records,
        'anchor_only_max_records': args.anchor_only_max_records,
        'rank_sync_statistic': args.rank_sync_statistic,
        'rank_sync_samples': args.rank_sync_samples,
        'time_rounds': args.time_rounds,
        'hidden': args.hidden,
        'intermediate_hidden': args.intermediate_hidden,
        'num_experts': args.num_experts,
        'num_topk': args.num_topk,
        'activation_clamp': args.activation_clamp,
        'masked_ratio': args.masked_ratio,
        'fast_math': args.fast_math,
        'num_tests': args.num_tests,
        'num_sms': args.num_sms,
        'seed': args.seed,
        'diff_tol': args.diff_tol,
        'elementwise_atol': args.elementwise_atol,
        'elementwise_atol_rms_scale': args.elementwise_atol_rms_scale,
        'elementwise_rtol': args.elementwise_rtol,
        'reference_chunk': args.reference_chunk,
        'reference_mem_budget_gb': args.reference_mem_budget_gb,
        'overhead_threshold_pct': args.overhead_threshold_pct,
        'gpu_check': gpu_check,
        'environment': {
            'DG_MEGA_OPT_VERSION': os.environ.get('DG_MEGA_OPT_VERSION'),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'PYTHONPATH': os.environ.get('PYTHONPATH'),
            'MASTER_ADDR': os.environ.get('MASTER_ADDR'),
            'MASTER_PORT': os.environ.get('MASTER_PORT'),
        },
    }

    started = time.time()
    torch.multiprocessing.spawn(_worker, args=(args.num_processes, cfg), nprocs=args.num_processes)
    print(f'[profile] 采集完成，用时 {time.time() - started:.1f}s，输出目录 {out_dir}')

    overhead_path = out_dir / 'overhead.json'
    correctness_path = out_dir / 'correctness.json'
    exit_code = 0
    if overhead_path.exists():
        doc = json.loads(overhead_path.read_text())
        print(f'[overhead] 口径 schema={doc.get("schema")} '
              f'rank_sync_statistic={doc.get("rank_sync_statistic")} '
              f'rank_sync_samples={doc.get("rank_sync_samples")} '
              f'num_tests={doc.get("num_tests")} time_rounds={doc.get("time_rounds")}')
        worst_t_stdev = doc.get('worst_kernel_time_stdev_us')
        print(f'[overhead] T 轮间噪声 worst_stdev='
              f'{"n/a" if not isinstance(worst_t_stdev, (int, float)) else f"{worst_t_stdev:.1f} us"} '
              f'({doc.get("kernel_time_sampling_definition")})')
        worst_stdev = doc.get('worst_rank_sync_stdev_us')
        print(f'[overhead] S 采样噪声 worst_stdev='
              f'{"n/a" if not isinstance(worst_stdev, (int, float)) else f"{worst_stdev:.1f} us"} '
              f'({doc.get("rank_sync_sampling_definition")})')
        header = (f'{"rank":>5s}{"tokens":>8s}{"T_none":>11s}{"T_anchor":>11s}{"T_full":>11s}'
                  f'{"S_anchor":>10s}{"S_full":>10s}{"rank_sync":>11s}'
                  f'{"pure_anchor":>13s}{"pure_full":>11s}'
                  f'{"PRIMARY pure_record%":>22s}{"legacy%":>10s}')
        print(f'[overhead] {header}')
        for entry in doc.get('per_rank', []):
            metrics = entry.get('metrics') or {}
            by_name = {tier['name']: tier for tier in entry.get('tiers', [])}

            def tier_number(name: str, field: str) -> str:
                item = (by_name.get(name) or {}).get(field) or {}
                value = item.get('value')
                return '-' if not isinstance(value, (int, float)) else f'{value:.1f}'

            def metric_number(name: str, digits: int = 1) -> str:
                value = (metrics.get(name) or {}).get('value')
                return '-' if not isinstance(value, (int, float)) else f'{value:.{digits}f}'

            print(f'[overhead] {entry.get("rank"):>5d}{entry.get("num_tokens"):>8d}'
                  f'{tier_number(TIER_ANCHOR_OFF, "kernel_us"):>11s}'
                  f'{tier_number(TIER_ANCHOR_ONLY, "kernel_us"):>11s}'
                  f'{tier_number(TIER_FULL, "kernel_us"):>11s}'
                  f'{tier_number(TIER_ANCHOR_ONLY, "rank_sync_us"):>10s}'
                  f'{tier_number(TIER_FULL, "rank_sync_us"):>10s}'
                  f'{metric_number("rank_sync_cost_us"):>11s}'
                  f'{metric_number("pure_anchor_us"):>13s}'
                  f'{metric_number("pure_full_us"):>11s}'
                  f'{metric_number(PRIMARY_METRIC, 3):>22s}'
                  f'{metric_number(LEGACY_METRIC, 3):>10s}')
        primary = doc.get('primary_metric') or {}
        legacy = doc.get('legacy_metric') or {}
        rank_sync = doc.get('rank_sync_cost_us') or {}

        def pct_text(value: Any) -> str:
            return 'n/a' if not isinstance(value, (int, float)) else f'{value:.3f}%'

        worst_sync = rank_sync.get('worst_us')
        print(f'[overhead] 最差 rank_sync_cost_us = '
              f'{"n/a" if not isinstance(worst_sync, (int, float)) else f"{worst_sync:.1f} us"} '
              f'({rank_sync.get("formula")}) <- 旧口径里被误算成记录开销的部分')
        print(f'[overhead] >>> 主指标 {primary.get("name")} 最差 {pct_text(primary.get("worst_pct"))} '
              f'(阈值 {doc["threshold_pct"]:.1f}%, within_threshold={primary.get("within_threshold")}) '
              f'窗口: 两侧都有 rank sync 且各自剔除 S')
        print(f'[overhead]     对照 {legacy.get("name")} 最差 {pct_text(legacy.get("worst_pct"))} '
              f'(within_threshold={legacy.get("within_threshold")}) '
              f'窗口: kernel 全长，baseline 无 rank sync —— 仅参考，不作为失败判据')
        sens = doc.get('primary_metric_sensitivity') or {}
        worst_by_combo = sens.get('worst_pct_by_combination') or {}
        if worst_by_combo:
            selected = sens.get('selected_combination')
            print(f'[overhead] 主指标口径敏感性（每种 S 口径下 {PRIMARY_METRIC} 的最差值）:')
            for key, value in worst_by_combo.items():
                marker = ' <== 采用' if key == selected else ''
                print(f'[overhead]     {key:<18s} {pct_text(value):>10s}{marker}')
        print(f'[overhead] 锚点完整性 anchor_integrity_ok={doc.get("anchor_integrity_ok")} '
              f'({doc.get("anchor_integrity_definition")})')
        if not doc['within_threshold']:
            print(f'[overhead] WARNING 主指标 {primary.get("name")} 超过 '
                  f'{doc["threshold_pct"]:.1f}% 阈值')
        if not legacy.get('within_threshold'):
            print(f'[overhead] NOTE 旧口径 {legacy.get("name")} 超过 {doc["threshold_pct"]:.1f}%，'
                  f'仅提示（含整套 rank sync 成本，不作为失败）')
        if not doc.get('anchor_integrity_ok'):
            print('[overhead] WARNING 锚点不完整，pure_* 与主指标的剔除量可能不可信')
    if correctness_path.exists():
        doc = json.loads(correctness_path.read_text())
        print(f'[profile] 正确性 all_passed={doc["all_passed"]} checks={doc["num_checks"]} '
              f'failed_elementwise={doc.get("num_failed_elementwise")} '
              f'failed_calc_diff={doc.get("num_failed_calc_diff")} '
              f'failed_cum_stats={doc.get("num_failed_cum_stats")}')
        if not doc['all_passed']:
            print('[profile] ERROR 正确性比对失败')
            exit_code = 1
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
