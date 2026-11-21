#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = 'gpu-hw-metrics/v1'
DEFAULT_KERNEL_NEEDLE = 'sm90_fp8_mega_moe_impl'
UNIT_RE = re.compile(r'\[([^\]]+)\]\s*$')


def list_tables(conn: sqlite3.Connection) -> List[str]:
    return [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]


def pick_table(tables: List[str], want: str) -> Optional[str]:
    for name in tables:
        if name.upper() == want.upper():
            return name
    return None


def split_name_unit(metric_name: str) -> Tuple[str, str]:
    match = UNIT_RE.search(metric_name)
    if match:
        unit = match.group(1).strip()
        name = metric_name[:match.start()].strip()
        return name, unit
    return metric_name.strip(), ''


DEFAULT_ALLOWLIST_NAME = 'ep_moe_metrics.json'


def normalize_metric(name: str) -> str:
    return re.sub(r'\s+', ' ', str(name).strip().lower())


def load_allowlist(path: Path) -> Tuple[List[str], List[str], Dict[str, Any]]:
    config = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise SystemExit(f'[fatal] metrics allowlist 必须是 JSON object: {path}')
    keep = config.get('keep') or []
    required = config.get('required') or []
    if not isinstance(keep, list) or not all(isinstance(item, str) for item in keep):
        raise SystemExit(f'[fatal] metrics allowlist 的 "keep" 必须是字符串列表: {path}')
    return list(keep), list(required), config


def apply_combine_rules(counters: List[Dict[str, Any]],
                        rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 rules 合并 counter。每条规则 {name, sum_of:[名字…], unit?, drop_sources?}。
    位置对齐求和(同频采样时 ts 位置一致),源缺失或 ts 不一致则跳过并警告。"""
    if not rules:
        return counters
    for rule in rules:
        target = rule.get('name')
        sources = rule.get('sum_of') or []
        if not target or not sources:
            continue
        found: List[Dict[str, Any]] = []
        skip = False
        for source_name in sources:
            hit = next((c for c in counters if c.get('name') == source_name), None)
            if hit is None:
                print(f'[combine] "{target}": 源 "{source_name}" 未在 counters 中，跳过')
                skip = True
                break
            found.append(hit)
        if skip:
            continue
        sample_lens = {len(c['samples']) for c in found}
        if len(sample_lens) != 1:
            print(f'[combine] "{target}": 源采样数不一致 {sample_lens}，跳过')
            continue
        merged_samples: List[List[Any]] = []
        ok = True
        for i in range(next(iter(sample_lens))):
            ts_vals = [c['samples'][i] for c in found]
            ts_set = {tv[0] for tv in ts_vals}
            if len(ts_set) != 1:
                print(f'[combine] "{target}": 位置 {i} 时间戳不一致 {ts_set}，跳过')
                ok = False
                break
            merged_samples.append([ts_vals[0][0], sum(tv[1] for tv in ts_vals)])
        if not ok:
            continue
        counters.append({
            'metric_id': None,
            'name': target,
            'raw_name': target,
            'unit': rule.get('unit') or found[0].get('unit', ''),
            'sample_count': len(merged_samples),
            'sample_count_before_clip': sum(c.get('sample_count_before_clip', 0) for c in found),
            'samples': merged_samples,
            'combined_from': [c['name'] for c in found],
        })
        print(f'[combine] {target} = sum({" + ".join(sources)}), {len(merged_samples)} 样本')
        if rule.get('drop_sources'):
            drop_set = set(sources)
            counters[:] = [c for c in counters if c.get('name') not in drop_set]
    return counters


def apply_scale_rules(counters: List[Dict[str, Any]],
                      rules: List[Dict[str, Any]],
                      device_name: Optional[str]) -> List[Dict[str, Any]]:
    """按 rules 对指定 counter 的 samples 值缩放。每条规则 {name, factor, rationale?}。
    用于 H20 之类算力阉割卡的校正:nsys 以 GH100 peak 归一,H20 上 Tensor Active 显示只有
    真值的 1/4,乘 4 才反映 H20 自身占用。name 会加 " (H20-corrected ×N)" 后缀便于识别。"""
    reports: List[Dict[str, Any]] = []
    derived_counters: List[Dict[str, Any]] = []
    if not rules:
        return reports
    for rule in rules:
        target = rule.get('name')
        factor = rule.get('factor')
        device_pattern = rule.get('device_name_regex')
        report = {
            'source_name': target,
            'factor': factor,
            'device_name_regex': device_pattern,
            'device_name': device_name,
            'applied': False,
        }
        reports.append(report)
        if not target or not factor or factor == 1:
            report['reason'] = 'invalid_or_identity_rule'
            continue
        if not isinstance(device_pattern, str) or not device_pattern:
            report['reason'] = 'missing_device_name_regex'
            print(f'[scale] "{target}": missing device_name_regex; skipped')
            continue
        if not isinstance(device_name, str) or re.search(device_pattern, device_name) is None:
            report['reason'] = 'device_name_mismatch_or_unknown'
            print(f'[scale] "{target}": device={device_name!r} does not match '
                  f'{device_pattern!r}; derived counter skipped')
            continue
        hit = next((c for c in counters if c.get('name') == target), None)
        if hit is None:
            report['reason'] = 'source_counter_missing'
            print(f'[scale] "{target}": counter 未找到,跳过')
            continue
        original = hit.get('name', target)
        derived = dict(hit)
        derived['samples'] = [[ts, v * factor] for ts, v in hit['samples']]
        derived['name'] = rule.get('derived_name') or (
            f'{original} (H20-corrected ×{int(factor)})'
            if factor == int(factor) else f'{original} (H20-corrected ×{factor})'
        )
        derived['derived_from'] = original
        derived['is_derived'] = True
        derived['scale_factor'] = factor
        derived['scale_rationale'] = rule.get('rationale', '')
        derived['scale_device_name'] = device_name
        derived_counters.append(derived)
        report.update({'applied': True, 'reason': 'device_name_matched',
                       'derived_name': derived['name']})
        print(f'[scale] {original} × {factor} -> {derived["name"]} '
              f'(device={device_name})')
    counters.extend(derived_counters)
    return reports


def match_keep(name: str, keep_norm: List[str]) -> Optional[str]:
    normalized = normalize_metric(name)
    for original, keep in keep_norm:
        if normalized == keep or keep in normalized or normalized in keep:
            return original
    return None


def validate_required_metrics(counters: List[Dict[str, Any]],
                              required: List[str]) -> Dict[str, Any]:
    """确认 required 指标存在，并且每个 required 指标在时间窗内至少有一个样本。"""
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for counter in counters:
        key = normalize_metric(counter.get('name', ''))
        by_name.setdefault(key, []).append(counter)
    missing: List[str] = []
    zero_samples: List[str] = []
    for name in required:
        hits = by_name.get(normalize_metric(name), [])
        if not hits:
            missing.append(name)
        elif all(int(item.get('sample_count', 0)) == 0 for item in hits):
            zero_samples.append(name)
    failures = [
        '缺少 required 指标: {}'.format(missing),
        'required 指标样本数为 0: {}'.format(zero_samples),
    ]
    failures = [item for item, values in zip(failures, (missing, zero_samples)) if values]
    return {
        'required': list(required),
        'missing': missing,
        'zero_samples': zero_samples,
        'failures': failures,
        'valid': not failures,
    }


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.is_file():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def manifest_device_name(manifest: Dict[str, Any],
                         device: Optional[int]) -> Optional[str]:
    """Resolve the measured physical GPU model recorded by capture_nsys.py."""
    gpu_check = manifest.get('gpu_check')
    if not isinstance(gpu_check, dict):
        return None
    gpus = gpu_check.get('gpus')
    if not isinstance(gpus, list):
        return None
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        try:
            matches = device is not None and int(gpu.get('index')) == int(device)
        except (TypeError, ValueError):
            matches = False
        name = gpu.get('name')
        if matches and isinstance(name, str) and name.strip():
            return name.strip()
    return None


def pick_device_type_id(conn: sqlite3.Connection) -> Optional[int]:
    row = conn.execute(
        'SELECT typeId, COUNT(*) AS c FROM GPU_METRICS GROUP BY typeId ORDER BY c DESC LIMIT 1'
    ).fetchone()
    return int(row[0]) if row else None


def metric_id_names(conn: sqlite3.Connection, tables: List[str],
                    device_type_id: int) -> Dict[int, str]:
    table = pick_table(tables, 'TARGET_INFO_GPU_METRICS')
    if table is None:
        raise SystemExit('[fatal] sqlite 缺少 TARGET_INFO_GPU_METRICS，无法映射 metricId->名字')
    mapping: Dict[int, str] = {}
    for metric_id, metric_name in conn.execute(
            f'SELECT metricId, metricName FROM "{table}" WHERE typeId = ?', (device_type_id,)):
        mapping[int(metric_id)] = metric_name
    return mapping


def find_kernels(conn: sqlite3.Connection, tables: List[str], needle: str,
                 device: Optional[int]) -> Tuple[str, List[Dict[str, Any]]]:
    ktable = pick_table(tables, 'CUPTI_ACTIVITY_KIND_KERNEL')
    if ktable is None:
        raise SystemExit('[fatal] sqlite 缺少 CUPTI_ACTIVITY_KIND_KERNEL')
    if pick_table(tables, 'StringIds') is None:
        raise SystemExit('[fatal] sqlite 缺少 StringIds，无法解析 kernel 名字')
    columns = table_columns(conn, ktable)
    name_col = 'demangledName' if 'demangledName' in columns else (
        'shortName' if 'shortName' in columns else None)
    if name_col is None:
        raise SystemExit(f'[fatal] kernel 表无名字列，列={columns}')
    sql = (f'SELECT k.correlationId, k.deviceId, k.start, k.end, s.value '
           f'FROM "{ktable}" k JOIN StringIds s ON s.id = k."{name_col}" '
           f'WHERE s.value LIKE ?')
    params: List[Any] = [f'%{needle}%']
    if device is not None:
        sql += ' AND k.deviceId = ?'
        params.append(device)
    sql += ' ORDER BY k.start'
    launches = []
    for corr, dev, start, end, name in conn.execute(sql, params):
        launches.append({
            'correlation_id': int(corr) if corr is not None else None,
            'device': int(dev),
            'start_ns': int(start),
            'end_ns': int(end),
            'name': name,
        })
    return sql, launches


def merge_windows(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for lo, hi in ordered[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def _interval_from_timestamps(ts: List[int], sample_metric_id: int) -> Dict[str, Any]:
    if len(ts) < 2:
        return {'available': False, 'reason': f'metricId={sample_metric_id} 样本不足', 'count': len(ts)}
    deltas = sorted(ts[i] - ts[i - 1] for i in range(1, len(ts)))
    n = len(deltas)

    def quantile(frac: float) -> int:
        idx = min(n - 1, max(0, int(round(frac * (n - 1)))))
        return deltas[idx]

    return {
        'available': True,
        'sample_metric_id': sample_metric_id,
        'count': len(ts),
        'median': quantile(0.5),
        'p95': quantile(0.95),
        'min': deltas[0],
        'max': deltas[-1],
        'mean': sum(deltas) / n,
    }


def interval_stats_full(conn: sqlite3.Connection, sample_metric_id: int,
                        device_type_id: int) -> Dict[str, Any]:
    rows = conn.execute(
        'SELECT timestamp FROM GPU_METRICS WHERE metricId = ? AND typeId = ? ORDER BY timestamp',
        (sample_metric_id, device_type_id)).fetchall()
    ts = [int(r[0]) for r in rows]
    return _interval_from_timestamps(ts, sample_metric_id)


def read_gpu_metrics_duckdb(sqlite_path: Path, device_type_id: int,
                            metric_ids: List[int],
                            windows: List[Tuple[int, int]]
                            ) -> Tuple[Dict[int, List[List[int]]], Dict[int, int], Dict[str, Any]]:
    """duckdb 路径：ATTACH sqlite 后一次 range-join 拉出窗口内样本，再拉一次 GROUP BY 拿 before_counts，
    再拉一次窗口外的采样间隔 timestamp。三查询共用同一 duckdb 连接，实测总耗时 <5s（对比 sqlite3 全表 4 min）。"""
    import duckdb  # 由 --backend auto 的调用方处理 ImportError
    con = duckdb.connect(':memory:')
    try:
        con.execute('INSTALL sqlite; LOAD sqlite;')
        con.execute(f"ATTACH '{str(sqlite_path)}' AS s (TYPE sqlite, READ_ONLY);")
        try:
            con.execute('SET threads TO 8;')
        except Exception:
            pass
        wanted_ids = sorted(int(mid) for mid in metric_ids)
        samples_by_metric: Dict[int, List[List[int]]] = {mid: [] for mid in wanted_ids}
        # 1) before_counts：单次 GROUP BY 拿到所有 metric 的行数
        before_counts: Dict[int, int] = {mid: 0 for mid in wanted_ids}
        wanted_set = set(wanted_ids)
        for mid, cnt in con.execute(
                'SELECT metricId, COUNT(*) FROM s.GPU_METRICS WHERE typeId = ? GROUP BY metricId',
                [device_type_id]).fetchall():
            mid_int = int(mid)
            if mid_int in wanted_set:
                before_counts[mid_int] = int(cnt)
        # 2) 窗口内样本：把窗口物化成小表后做 range join，一次拉出所有过滤后的样本
        con.execute('CREATE OR REPLACE TABLE _wins(lo BIGINT, hi BIGINT)')
        if windows:
            con.executemany('INSERT INTO _wins VALUES (?, ?)', windows)
        placeholders = ','.join(str(mid) for mid in wanted_ids) or 'NULL'
        sql = (f'SELECT g.metricId, g.timestamp, g.value '
               f'FROM s.GPU_METRICS g, _wins w '
               f'WHERE g.typeId = {int(device_type_id)} '
               f'AND g.metricId IN ({placeholders}) '
               f'AND g.timestamp BETWEEN w.lo AND w.hi '
               f'ORDER BY g.metricId, g.timestamp')
        for mid, ts_ns, val in con.execute(sql).fetchall():
            mid_int = int(mid)
            if mid_int in samples_by_metric:
                samples_by_metric[mid_int].append([int(ts_ns), int(val)])
        # 3) 采样间隔：只拉第一个 metric 的全 timestamp 序列
        sample_metric_id = wanted_ids[0] if wanted_ids else None
        if sample_metric_id is None:
            interval = {'available': False, 'reason': '无 metricId'}
        else:
            ts_rows = con.execute(
                'SELECT timestamp FROM s.GPU_METRICS '
                'WHERE typeId = ? AND metricId = ? ORDER BY timestamp',
                [device_type_id, sample_metric_id]).fetchall()
            interval = _interval_from_timestamps([int(row[0]) for row in ts_rows], sample_metric_id)
        return samples_by_metric, before_counts, interval
    finally:
        con.close()


def read_gpu_metrics_sqlite(conn: sqlite3.Connection, device_type_id: int,
                            metric_ids: List[int],
                            windows: List[Tuple[int, int]]
                            ) -> Tuple[Dict[int, List[List[int]]], Dict[int, int], Dict[str, Any]]:
    """sqlite3 回退路径：加载 PRAGMA cache_size + temp_store=MEMORY，然后按 typeId ORDER BY timestamp 
    单次全表扫描，配合已排好序的 windows 做 sweep-line 分派，避免 N×full-scan。相比原实现（250s）
    实测降到 60~90s 量级，仍慢于 duckdb 但作为兜底可用。"""
    try:
        conn.execute('PRAGMA cache_size = -524288')  # 512 MiB page cache
        conn.execute('PRAGMA temp_store = MEMORY')
        conn.execute('PRAGMA mmap_size = 1073741824')  # 1 GiB
    except sqlite3.Error:
        pass
    wanted_ids = sorted(int(mid) for mid in metric_ids)
    samples_by_metric: Dict[int, List[List[int]]] = {mid: [] for mid in wanted_ids}
    wanted_set = set(wanted_ids)
    # 1) before_counts：一次 GROUP BY
    before_counts: Dict[int, int] = {mid: 0 for mid in wanted_ids}
    for mid, cnt in conn.execute(
            'SELECT metricId, COUNT(*) FROM GPU_METRICS WHERE typeId = ? GROUP BY metricId',
            (device_type_id,)):
        mid_int = int(mid)
        if mid_int in wanted_set:
            before_counts[mid_int] = int(cnt)
    # 2) 窗口内样本：单次全表扫描 + sweep-line。窗口已 merged 且有序（见 merge_windows）
    ordered = sorted(windows)
    if ordered and wanted_ids:
        idx = 0
        cursor = conn.execute(
            'SELECT metricId, timestamp, value FROM GPU_METRICS '
            'WHERE typeId = ? ORDER BY timestamp',
            (device_type_id,))
        n_win = len(ordered)
        cur_lo, cur_hi = ordered[0]
        for mid, ts_ns, val in cursor:
            ts_ns = int(ts_ns)
            while ts_ns > cur_hi:
                idx += 1
                if idx >= n_win:
                    cur_lo, cur_hi = None, None
                    break
                cur_lo, cur_hi = ordered[idx]
            if cur_hi is None:
                break
            if ts_ns >= cur_lo:
                mid_int = int(mid)
                if mid_int in samples_by_metric:
                    samples_by_metric[mid_int].append([ts_ns, int(val)])
        # 稳定输出：按 (metricId, timestamp) 排序以匹配旧路径
        for mid in samples_by_metric:
            samples_by_metric[mid].sort(key=lambda item: item[0])
    # 3) 采样间隔
    sample_metric_id = wanted_ids[0] if wanted_ids else None
    if sample_metric_id is None:
        interval = {'available': False, 'reason': '无 metricId'}
    else:
        interval = interval_stats_full(conn, sample_metric_id, device_type_id)
    return samples_by_metric, before_counts, interval


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='从 nsys sqlite 提取目标 kernel 时间窗内的 GPU 硬件 counter')
    parser.add_argument('--sqlite', required=True)
    parser.add_argument('--kernel-name-contains', default=DEFAULT_KERNEL_NEEDLE)
    parser.add_argument('--out', required=True)
    parser.add_argument('--run-tag', type=str, default=None,
                        help='run 标签(如 mega_moe_bs32_topk8_ep1),写入 hw_metrics.json 顶层 run_tag,'
                             '并在 --out 为目录时用作默认文件名 hw_metrics__<tag>.json')
    parser.add_argument('--manifest', default=None,
                        help='默认取 sqlite 同目录的 capture_manifest.json，用于读 device/metrics_set/frequency')
    parser.add_argument('--device', type=int, default=None,
                        help='覆盖 GPU metrics 归属的物理设备号；默认取 manifest 的 gpu_metrics_device')
    parser.add_argument('--margin-frac', type=float, default=0.10,
                        help='每次 kernel launch 时间窗前后各留的余量比例（相对该次 launch 时长）')
    parser.add_argument('--metrics-allowlist', default=None,
                        help='指标白名单 JSON（含 keep 列表，如 ep_moe_metrics.json）；'
                             '缺省时自动取本脚本同目录的 ep_moe_metrics.json（存在才启用），'
                             '只保留 EP-MoE 关心的 SM/DRAM/NVLink 指标')
    parser.add_argument('--keep-metric', action='append', default=None,
                        help='额外保留的指标名（可重复），与白名单合并')
    parser.add_argument('--keep-all', action='store_true',
                        help='关闭过滤，保留 sqlite 里全部 GPU metrics')
    parser.add_argument('--backend', choices=('auto', 'duckdb', 'sqlite'), default='auto',
                        help='读取 GPU_METRICS 的后端：auto 优先 duckdb（对 sqlite 做列存 vectorized 扫描，'
                             '实测比 sqlite3 快 50~100x），失败自动退回 sqlite3；'
                             'duckdb 强制走 duckdb，sqlite 强制走 sqlite3 sweep-line。'
                             '两条路径输出的 hw_metrics.json 逐字段等价')
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    sqlite_path = Path(args.sqlite).expanduser().resolve()
    if not sqlite_path.is_file():
        print(f'[fatal] sqlite 不存在: {sqlite_path}')
        return 1
    out_path = Path(args.out).expanduser().resolve()

    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest \
        else sqlite_path.parent / 'capture_manifest.json'
    manifest = load_manifest(manifest_path)
    device = args.device if args.device is not None else manifest.get('gpu_metrics_device')
    device_name = manifest_device_name(manifest, device)
    metrics_set = manifest.get('metrics_set', 'unknown')
    frequency_hz = manifest.get('frequency_hz')

    keep_list: List[str] = []
    required_list: List[str] = []
    allowlist_source: Optional[str] = None
    combine_rules: List[Dict[str, Any]] = []
    scale_rules: List[Dict[str, Any]] = []
    if not args.keep_all:
        allowlist_path: Optional[Path] = None
        if args.metrics_allowlist:
            allowlist_path = Path(args.metrics_allowlist).expanduser().resolve()
            if not allowlist_path.is_file():
                print(f'[fatal] --metrics-allowlist 不存在: {allowlist_path}')
                return 1
        else:
            candidate = Path(__file__).resolve().parent / DEFAULT_ALLOWLIST_NAME
            if candidate.is_file():
                allowlist_path = candidate
        if allowlist_path is not None:
            keep_list, required_list, _config = load_allowlist(allowlist_path)
            allowlist_source = str(allowlist_path)
            combine_rules = _config.get('combine') or []
            scale_rules = _config.get('scale') or []
    if args.keep_metric:
        keep_list = list(keep_list) + list(args.keep_metric)
    filtering_enabled = (not args.keep_all) and bool(keep_list)
    keep_norm = [(item, normalize_metric(item)) for item in keep_list]

    conn = sqlite3.connect(f'file:{sqlite_path}?mode=ro', uri=True)
    try:
        tables = list_tables(conn)
        print(f'[schema] sqlite 表数 {len(tables)}')
        for name in ('GPU_METRICS', 'TARGET_INFO_GPU_METRICS',
                     'CUPTI_ACTIVITY_KIND_KERNEL', 'StringIds'):
            resolved = pick_table(tables, name)
            if resolved:
                print(f'[schema]   {resolved} 列: {table_columns(conn, resolved)}')
            else:
                print(f'[schema]   缺表 {name}')
        if pick_table(tables, 'GPU_METRICS') is None:
            print('[fatal] sqlite 无 GPU_METRICS 表')
            return 1

        device_type_id = pick_device_type_id(conn)
        if device_type_id is None:
            print('[fatal] GPU_METRICS 为空，未采到任何 GPU metrics 样本')
            return 1
        print(f'[schema] 主 device typeId={device_type_id}（排除 EndTimestamp 等其它 typeId 的 meta）')
        id_to_name = metric_id_names(conn, tables, device_type_id)
        kernel_sql, launches = find_kernels(conn, tables, args.kernel_name_contains, device)
        print(f'[kernel] SQL: {kernel_sql}')
        print(f'[kernel] 匹配 "{args.kernel_name_contains}" (device={device}) 的 launch 数 {len(launches)}')
        if not launches:
            print('[fatal] 未匹配到目标 kernel，无法确定时间窗')
            return 3

        durations = [lc['end_ns'] - lc['start_ns'] for lc in launches]
        windows_raw: List[Tuple[int, int]] = []
        for lc, dur in zip(launches, durations):
            margin = int(max(dur, 0) * args.margin_frac)
            windows_raw.append((lc['start_ns'] - margin, lc['end_ns'] + margin))
        windows = merge_windows(windows_raw)
        span_all = (min(lc['start_ns'] for lc in launches),
                    max(lc['end_ns'] for lc in launches))
        print(f'[window] launch 时长 min/median/max ns = '
              f'{min(durations)}/{sorted(durations)[len(durations)//2]}/{max(durations)}')
        print(f'[window] 合并后裁剪区间数 {len(windows)}，'
              f'并集覆盖 {sum(hi - lo for lo, hi in windows)} ns，'
              f'全 launch 跨度 {span_all[1] - span_all[0]} ns')

        all_metric_ids = sorted(id_to_name.keys())
        matched_keep: set = set()
        dropped_names: List[str] = []
        missing_required: List[str] = []
        if filtering_enabled:
            kept_ids: List[int] = []
            for mid in all_metric_ids:
                base, _unit = split_name_unit(id_to_name[mid])
                hit = match_keep(base, keep_norm)
                if hit is not None:
                    kept_ids.append(mid)
                    matched_keep.add(hit)
                else:
                    dropped_names.append(base)
            if kept_ids:
                metric_ids = sorted(kept_ids)
            else:
                print('[metrics] WARNING 白名单没匹配到任何指标，回退为保留全部')
                metric_ids = all_metric_ids
                filtering_enabled = False
            missing_required = [item for item in required_list
                                if normalize_metric(item) not in
                                {normalize_metric(name) for name in matched_keep}]
        else:
            metric_ids = all_metric_ids
        if filtering_enabled:
            print(f'[metrics] 白名单过滤（{allowlist_source}）: 保留 {len(metric_ids)}/'
                  f'{len(all_metric_ids)} 个指标，丢弃 {len(dropped_names)} 个')
            print(f'[metrics] 保留: {[split_name_unit(id_to_name[mid])[0] for mid in metric_ids]}')
            if dropped_names:
                print(f'[metrics] 丢弃: {dropped_names}')
            not_found = [item for item in keep_list if item not in matched_keep]
            if not_found:
                print(f'[metrics] note 白名单里这些名字在本 sqlite 未出现: {not_found}')
            if missing_required:
                print(f'[metrics] WARNING 缺少 required 指标: {missing_required}')
        else:
            print(f'[metrics] 不过滤，保留全部 {len(metric_ids)} 个指标'
                  f'（--keep-all 或未找到白名单）')
        before_counts: Dict[int, int]
        samples_by_metric: Dict[int, List[List[int]]]
        interval: Dict[str, Any]
        backend_used = 'sqlite'
        backend_error: Optional[str] = None
        read_started = __import__('time').time()

        def _run_sqlite() -> Tuple[Dict[int, List[List[int]]], Dict[int, int], Dict[str, Any]]:
            return read_gpu_metrics_sqlite(conn, device_type_id, metric_ids, windows)

        if args.backend in ('auto', 'duckdb'):
            try:
                samples_by_metric, before_counts, interval = read_gpu_metrics_duckdb(
                    sqlite_path, device_type_id, metric_ids, windows)
                backend_used = 'duckdb'
            except ImportError as exc:
                backend_error = f'duckdb 未安装（{exc}）'
                if args.backend == 'duckdb':
                    print(f'[fatal] 强制 --backend duckdb 但导入失败: {exc}')
                    return 1
                print(f'[backend] duckdb 不可用，退回 sqlite3 sweep-line: {exc}')
                samples_by_metric, before_counts, interval = _run_sqlite()
            except Exception as exc:  # duckdb API 层面的意外
                backend_error = f'duckdb 运行时错误 ({type(exc).__name__}): {exc}'
                if args.backend == 'duckdb':
                    print(f'[fatal] 强制 --backend duckdb 但运行失败: {exc}')
                    return 1
                print(f'[backend] duckdb 路径失败，退回 sqlite3: {exc}')
                samples_by_metric, before_counts, interval = _run_sqlite()
        else:
            samples_by_metric, before_counts, interval = _run_sqlite()
        read_elapsed = __import__('time').time() - read_started
        total_before = sum(before_counts.get(mid, 0) for mid in metric_ids)
        total_after = sum(len(v) for v in samples_by_metric.values())
        print(f'[backend] 使用 {backend_used}，读 GPU_METRICS 耗时 {read_elapsed:.2f} s，'
              f'样本 {total_before} -> {total_after}'
              + (f'（duckdb 首选失败原因: {backend_error}）' if backend_error and backend_used == 'sqlite' else ''))

        counters = []
        for mid in metric_ids:
            raw_name = id_to_name[mid]
            name, unit = split_name_unit(raw_name)
            counters.append({
                'metric_id': mid,
                'name': name,
                'raw_name': raw_name,
                'unit': unit,
                'sample_count': len(samples_by_metric[mid]),
                'sample_count_before_clip': before_counts.get(mid, 0),
                'samples': samples_by_metric[mid],
            })
    finally:
        conn.close()

    apply_combine_rules(counters, combine_rules)
    scale_rule_report = apply_scale_rules(counters, scale_rules, device_name)

    required_validation = validate_required_metrics(counters, required_list)
    missing_required = required_validation['missing']
    if required_validation['zero_samples']:
        print('[metrics] FAIL required 指标没有样本: {}'.format(
            required_validation['zero_samples']))
    if missing_required:
        print('[metrics] FAIL 缺少 required 指标: {}'.format(missing_required))

    result = {
        'schema': SCHEMA,
        'run_tag': args.run_tag,
        'source_sqlite': str(sqlite_path),
        'source_manifest': str(manifest_path) if manifest else None,
        'device': device,
        'device_name': device_name,
        'device_type_id': device_type_id,
        'metrics_set': metrics_set,
        'frequency_hz': frequency_hz,
        'scale_rules': scale_rule_report,
        'kernel_name_contains': args.kernel_name_contains,
        'margin_frac': args.margin_frac,
        'metrics_filter': {
            'enabled': filtering_enabled,
            'allowlist_source': allowlist_source,
            'keep': keep_list,
            'required': required_list,
            'matched': sorted(matched_keep),
            'dropped': dropped_names,
            'missing_required': missing_required,
            'zero_sample_required': required_validation['zero_samples'],
            'kept_metric_count': len(metric_ids),
            'total_metric_count': len(all_metric_ids),
        },
        'clip_windows_ns': [[lo, hi] for lo, hi in windows],
        'clip_report': {
            'samples_before_clip': total_before,
            'samples_after_clip': total_after,
            'dropped': total_before - total_after,
            'launch_span_ns': [span_all[0], span_all[1]],
            'clip_union_ns': sum(hi - lo for lo, hi in windows),
        },
        'sampling_interval_ns': interval,
        'kernel_launches': launches,
        'counters': counters,
        'required_validation': required_validation,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_dir():
        tag = args.run_tag or 'run'
        out_path = out_path / f'hw_metrics__{tag}.json'
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')

    print(f'[out] {out_path} bytes={out_path.stat().st_size}')
    print(f'[out] counters={len(counters)} '
          f'样本裁剪 {total_before} -> {total_after}（丢弃 {total_before - total_after}）')
    if interval.get('available'):
        print(f'[out] 采样间隔 median={interval["median"]} ns p95={interval["p95"]} ns '
              f'count={interval["count"]}'
              + (f'（期望 {1e9 / frequency_hz:.0f} ns）' if frequency_hz else ''))
    for counter in counters:
        print(f'[out]   {counter["name"]} [{counter["unit"]}] 样本 '
              f'{counter["sample_count_before_clip"]} -> {counter["sample_count"]}')
    return 0 if required_validation['valid'] else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
