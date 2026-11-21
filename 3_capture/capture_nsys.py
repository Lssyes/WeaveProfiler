#!/usr/bin/env python3
"""nsys GPU Metrics 采集包装器（information_bundle 含 hw 时使用）。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import resource
except ImportError:
    resource = None

SCHEMA = 'nsys-hw-capture/v1'
DEFAULT_FREQUENCY_HZ = 200000
DEFAULT_CAPTURE_RANGE = 'cudaProfilerApi'
DEFAULT_CAPTURE_RANGE_END = 'stop'
REQUIRED_NSYS_CONTROLS = (
    ('--flush-on-cudaprofilerstop', 'flush_on_cudaprofiler_stop'),
    ('--trace-fork-before-exec', 'trace_fork_before_exec'),
)
REQUIRED_HW_METRIC_NAMES: Tuple[str, ...] = ()
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_METRICS_CONFIG_NAME = 'nsys_metrics_ep_moe_hopper.config'
DEFAULT_METRICS_CONFIG_ABSPATH = SCRIPT_DIR / DEFAULT_METRICS_CONFIG_NAME
DEFAULT_FALLBACK_METRICS_SET = 'gh100'
# 默认使用与本脚本同目录的 Hopper 示例配置。实际任务应按基础指标、目标硬件
# 以及真实通信通路选择指标；200 kHz 是优先请求值，稳定性由采集结果判断。
# 兜底：--fallback-metrics-set gh100（内置全集），resolve_metrics_set 在自定义 config 探测失败时才走。
DEFAULT_METRICS_SET = f'file:{DEFAULT_METRICS_CONFIG_ABSPATH}'
DEFAULT_EXPORT_TABLES = (
    'GPU_METRICS',
    'TARGET_INFO',
    'TARGET_INFO_GPU_METRICS',
    'CUPTI_ACTIVITY_KIND_KERNEL',
    'CUPTI_ACTIVITY_KIND_RUNTIME',
    'StringIds',
    'ANALYSIS_DETAILS',
    'META_DATA',
    'OVERFLOW',
    'PROFILER_OVERHEAD',
    'GENERIC_EVENT',
    'UNKNOWN',
)
LOG_PATTERNS = (
    ('overflow', re.compile(r'overflow', re.IGNORECASE)),
    ('dropped', re.compile(r'\bdrop(ped|ping|s)?\b', re.IGNORECASE)),
    ('missing', re.compile(r'\bmissing\b', re.IGNORECASE)),
    ('incomplete', re.compile(r'\bincomplete\b', re.IGNORECASE)),
    ('cupti', re.compile(r'\bcupti\b', re.IGNORECASE)),
    ('error', re.compile(r'\berror\b', re.IGNORECASE)),
    ('warning', re.compile(r'\bwarn(ing)?\b', re.IGNORECASE)),
    ('permission', re.compile(r'permission|not permitted|admin', re.IGNORECASE)),
)
BLOCKING_KINDS = ('overflow', 'dropped')


def utc_iso(epoch: float) -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(epoch))


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def dir_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def _preexec_drop_core() -> None:
    if resource is not None:
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            pass


def _kill_process_group(proc: subprocess.Popen) -> None:
    # start_new_session=True saves the direct child's pid as the process-group
    # id on POSIX; on Windows fall back to taskkill /T /F or proc.kill().
    if hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            return
        except OSError:
            pass
    elif os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False, timeout=10)
            return
        except Exception:
            pass
    try:
        proc.kill()
    except OSError:
        pass


def run_capture(cmd: List[str], cwd: Path, log_path: Path,
                timeout: Optional[float],
                env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    chunks: List[str] = []
    with log_path.open('w', encoding='utf-8') as log:
        header = '$ ' + ' '.join(shlex.quote(part) for part in cmd) + '\n'
        log.write(header)
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env=env,
                                start_new_session=True,
                                preexec_fn=_preexec_drop_core if resource is not None else None)
        timed_out = threading.Event()
        timer: Optional[threading.Timer] = None
        if timeout is not None:
            timer = threading.Timer(timeout, lambda: (timed_out.set(), _kill_process_group(proc)))
            timer.daemon = True
            timer.start()
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                chunks.append(line)
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
        finally:
            if timer is not None:
                timer.cancel()
            code = proc.wait()
            if proc.stdout is not None:
                proc.stdout.close()
        if timed_out.is_set():
            message = '[capture] TIMEOUT，已终止整个 nsys 进程组\n'
            chunks.append(message)
            log.write(message)
            code = 124
    return code, ''.join(chunks)


def query_gpus() -> Optional[List[Dict[str, Any]]]:
    fields = 'index,uuid,name,memory.used,memory.total,utilization.gpu'
    try:
        out = subprocess.run(['nvidia-smi', f'--query-gpu={fields}',
                              '--format=csv,noheader,nounits'],
                             check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    rows = []
    for line in out.strip().splitlines():
        parts = [item.strip() for item in line.split(',')]
        if len(parts) < 6:
            continue
        rows.append({
            'index': int(parts[0]),
            'uuid': parts[1],
            'name': parts[2],
            'memory_used_mib': float(parts[3]),
            'memory_total_mib': float(parts[4]),
            'utilization_pct': float(parts[5]),
        })
    return rows


def query_compute_apps() -> Optional[List[Dict[str, Any]]]:
    fields = 'gpu_uuid,pid,process_name,used_gpu_memory'
    try:
        out = subprocess.run(['nvidia-smi', f'--query-compute-apps={fields}',
                              '--format=csv,noheader,nounits'],
                             check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    apps = []
    for line in out.strip().splitlines():
        parts = [item.strip() for item in line.split(',')]
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        apps.append({'gpu_uuid': parts[0], 'pid': pid, 'process_name': parts[2],
                     'used_memory_mib': parts[3]})
    return apps


def inspect_gpus(target_indices: List[int], mem_threshold_mib: float,
                 util_threshold_pct: float) -> Dict[str, Any]:
    gpus = query_gpus()
    apps = query_compute_apps() or []
    report: Dict[str, Any] = {
        'nvidia_smi_available': gpus is not None,
        'target_indices': target_indices,
        'memory_threshold_mib': mem_threshold_mib,
        'utilization_threshold_pct': util_threshold_pct,
        'gpus': [],
        'foreign_processes': [],
        'busy': False,
        'reasons': [],
    }
    if gpus is None:
        report['busy'] = True
        report['reasons'].append('nvidia-smi 不可用，无法确认 GPU 空闲')
        return report
    by_index = {gpu['index']: gpu for gpu in gpus}
    uuid_to_index = {gpu['uuid']: gpu['index'] for gpu in gpus}
    for index in target_indices:
        gpu = by_index.get(index)
        if gpu is None:
            report['busy'] = True
            report['reasons'].append(f'GPU {index} 不存在')
            continue
        entry = dict(gpu)
        entry['over_memory'] = gpu['memory_used_mib'] > mem_threshold_mib
        entry['over_utilization'] = gpu['utilization_pct'] > util_threshold_pct
        report['gpus'].append(entry)
        if entry['over_memory']:
            report['busy'] = True
            report['reasons'].append(
                f'GPU {index} 已用显存 {gpu["memory_used_mib"]:.0f} MiB '
                f'> 阈值 {mem_threshold_mib:.0f} MiB')
        if entry['over_utilization']:
            report['busy'] = True
            report['reasons'].append(
                f'GPU {index} 利用率 {gpu["utilization_pct"]:.0f}% '
                f'> 阈值 {util_threshold_pct:.0f}%')
    my_pids = {os.getpid(), os.getppid()}
    for app in apps:
        index = uuid_to_index.get(app['gpu_uuid'])
        if index is None or index not in target_indices:
            continue
        if app['pid'] in my_pids:
            continue
        report['foreign_processes'].append(dict(app, index=index))
        report['busy'] = True
        report['reasons'].append(
            f'GPU {index} 上存在其他进程 pid={app["pid"]} '
            f'name={app["process_name"]}（不会 kill，请等待空闲）')
    return report


def resolve_visible_devices(cuda_visible: Optional[str],
                            num_processes: int) -> Dict[str, Any]:
    if cuda_visible is None or cuda_visible.strip() == '':
        return {'cuda_visible_devices': None, 'selector': 'physical-index',
                'harness_physical_indices': list(range(max(num_processes, 0))),
                'tokens': None, 'mappable': True}
    tokens = [item.strip() for item in cuda_visible.split(',') if item.strip()]
    # CUDA_VISIBLE_DEVICES describes devices visible to the payload, not its
    # OS process count.  A single Python process may legitimately launch one
    # coordinated kernel on every visible GPU.
    used = tokens
    if used and all(item.lstrip('-').isdigit() for item in used):
        return {'cuda_visible_devices': cuda_visible, 'selector': 'cuda-visible-index',
                'harness_physical_indices': [int(item) for item in used],
                'tokens': tokens, 'mappable': True,
                'enough_devices': len(tokens) >= num_processes}
    return {'cuda_visible_devices': cuda_visible, 'selector': 'uuid',
            'harness_physical_indices': [], 'tokens': tokens, 'mappable': False,
            'enough_devices': len(tokens) >= num_processes}


PRUNE_DIRS = {'.git', '.work', '.cache', 'node_modules', '__pycache__', 'site-packages',
              'venvs', 'venv', '.venv', 'build', 'dist', 'data', 'datasets', 'core',
              '.mypy_cache', '.pytest_cache', 'third_party', 'sweep_results'}


def find_config(root: Path, name: str, max_depth: int = 6,
                budget_sec: Optional[float] = None) -> Optional[Path]:
    root_parts = len(root.parts)
    deadline = None if budget_sec is None else time.time() + budget_sec
    for current, dirs, files in os.walk(root):
        if deadline is not None and time.time() > deadline:
            return None
        here = Path(current)
        depth = len(here.parts) - root_parts
        if depth >= max_depth:
            dirs[:] = []
        else:
            dirs[:] = sorted(item for item in dirs
                             if item not in PRUNE_DIRS and not item.startswith('core-'))
        if name in files:
            return here / name
    return None


def resolve_metrics_set(args: argparse.Namespace, script_dir: Path,
                        root: Path) -> Dict[str, Any]:
    if args.metrics_set != 'auto':
        if args.metrics_set.startswith('file:'):
            path = Path(args.metrics_set[len('file:'):]).expanduser().resolve()
            if not path.is_file():
                raise SystemExit(f'[metrics] 自定义 metrics 配置不存在: {path}')
            return {'value': f'file:{path}', 'source': 'explicit-file', 'config_path': str(path)}
        return {'value': args.metrics_set, 'source': 'explicit-builtin', 'config_path': None}
    if args.metrics_config:
        path = Path(args.metrics_config).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f'[metrics] --metrics-config 指向的文件不存在: {path}')
        return {'value': f'file:{path}', 'source': 'metrics-config-arg', 'config_path': str(path)}
    known_subdir = (Path('skills') / 'profile-gpu-kernel-timeline' / 'backends'
                    / 'hardware' / 'generic_sm_90' / 'nsys-metrics')
    for base in [script_dir, *script_dir.parents]:
        cand = base / known_subdir / args.metrics_config_name
        if cand.is_file():
            resolved = cand.resolve()
            return {'value': f'file:{resolved}',
                    'source': f'anchor-probe under {base}',
                    'config_path': str(resolved)}
    search_roots: List[Path] = []
    for raw in args.metrics_search_root or []:
        search_roots.append(Path(raw).expanduser().resolve())
    for level in (1, 2, 3):
        if len(script_dir.parents) > level:
            search_roots.append(script_dir.parents[level])
    seen: List[Path] = []
    for candidate in search_roots:
        if candidate in seen or not candidate.is_dir():
            continue
        seen.append(candidate)
        hit = find_config(candidate, args.metrics_config_name, max_depth=8,
                          budget_sec=args.metrics_search_budget_sec)
        if hit is not None:
            resolved = hit.resolve()
            return {'value': f'file:{resolved}',
                    'source': f'bounded-find under {candidate}',
                    'config_path': str(resolved)}
    return {'value': args.fallback_metrics_set,
            'source': f'fallback（锚点探测与有界搜索都未找到 {args.metrics_config_name}，'
                      f'搜索根 {[str(item) for item in seen]}）',
            'config_path': None}


def build_payload(args: argparse.Namespace, root: Path, out_dir: Path,
                  harness_tmpdir: str) -> List[str]:
    if args.payload:
        return list(args.payload)
    harness = Path(args.harness).expanduser().resolve() if args.harness else \
        (root / 'profile' / 'harness' / 'run_profile.py').resolve()
    if not harness.is_file():
        raise SystemExit(f'[payload] harness 不存在: {harness}')
    cmd = ['env', f'TMPDIR={harness_tmpdir}',
           sys.executable, '-B', str(harness),
           '--out-dir', str(out_dir / 'harness'),
           '--num-processes', str(args.num_processes),
           '--batches', *[str(item) for item in args.batches],
           '--max-records', str(args.max_records)]
    if args.force:
        cmd.append('--force')
    if args.capture_range == 'cudaProfilerApi':
        cmd.append('--cuda-profiler-range')
    if args.harness_skip_overhead:
        cmd.append('--skip-overhead')
    cmd.extend(args.harness_extra or [])
    return cmd


def _parse_bool_token(value: str) -> Optional[bool]:
    normalized = value.strip().lower()
    if normalized in ('1', 'true', 'yes', 'on', 'enable', 'enabled'):
        return True
    if normalized in ('0', 'false', 'no', 'off', 'disable', 'disabled'):
        return False
    return None


def normalize_nsys_controls(extras: Optional[List[str]]) -> Tuple[List[str], Dict[str, bool]]:
    """Return nsys extras with the two HW-capture safety controls forced on.

    Nsight's fork tracing is needed because the harness launches one process per
    rank, while the flush option is needed to retain CUDA records at
    cudaProfilerStop.  A user may still pass either option for CLI compatibility,
    but an explicit false/invalid value is a hard configuration error.
    """
    normalized = list(extras or [])
    controls: Dict[str, bool] = {}
    for option, field in REQUIRED_NSYS_CONTROLS:
        seen: List[bool] = []
        for index, item in enumerate(normalized):
            if item == option:
                # A bare Nsight boolean option means enabled.
                seen.append(True)
                continue
            if not item.startswith(option + '='):
                continue
            parsed = _parse_bool_token(item.split('=', 1)[1])
            if parsed is None:
                raise ValueError(f'{option} 的值无效: {item!r}；必须是 true/false')
            seen.append(parsed)
        if any(value is False for value in seen):
            raise ValueError(f'HW capture 强制要求 {option}=true')
        controls[field] = True
        if not seen:
            normalized.append(f'{option}=true')
    return normalized, controls


def _find_column(columns: List[str], *candidates: str) -> Optional[str]:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _quantile_from_sorted(values: List[int], fraction: float) -> int:
    if not values:
        raise ValueError('quantile requires at least one value')
    target = max(1, int((fraction * len(values)) + 0.999999999))
    return values[min(target, len(values)) - 1]


def build_nsys_cmd(args: argparse.Namespace, nsys: str, metrics_set: str,
                   out_prefix: Path, payload: List[str]) -> List[str]:
    cmd = [nsys, 'profile',
           '--output', str(out_prefix),
           '--force-overwrite', 'true',
           '--trace', args.trace,
           '--sample', 'none',
           '--cpuctxsw', 'none',
           '--cuda-memory-usage', 'false',
           '--wait', 'all',
           f'--gpu-metrics-devices={args.gpu_metrics_device}',
           f'--gpu-metrics-set={metrics_set}',
           f'--gpu-metrics-frequency={args.gpu_metrics_frequency}']
    if args.capture_range != 'none':
        cmd.append(f'--capture-range={args.capture_range}')
        cmd.append(f'--capture-range-end={args.capture_range_end}')
    normalized_extras, _controls = normalize_nsys_controls(args.nsys_extra)
    cmd.extend(normalized_extras)
    cmd.extend(payload)
    return cmd


def build_export_cmd(args: argparse.Namespace, nsys: str, rep: Path,
                     sqlite_path: Path) -> List[str]:
    cmd = [nsys, 'export', '--type', 'sqlite', '--output', str(sqlite_path),
           '--force-overwrite', 'true']
    if not args.export_all_tables:
        tables = args.export_tables or ','.join(DEFAULT_EXPORT_TABLES)
        cmd.extend(['--tables', tables])
    cmd.append(str(rep))
    return cmd


def scan_log(text: str) -> Dict[str, Any]:
    findings: Dict[str, Any] = {kind: {'count': 0, 'lines': []} for kind, _ in LOG_PATTERNS}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for kind, pattern in LOG_PATTERNS:
            # Harnesses may print structured success diagnostics such as
            # `"overflow": 0`.  Those are explicit negative evidence and
            # must not be promoted to a runtime overflow/drop failure.
            if kind in ('overflow', 'dropped') and re.search(
                    rf'["\']?{kind}["\']?\s*[:=]\s*(?:0|false)\b',
                    stripped, re.IGNORECASE):
                continue
            if pattern.search(stripped):
                findings[kind]['count'] += 1
                if len(findings[kind]['lines']) < 12:
                    findings[kind]['lines'].append(stripped)
    return findings


def sqlite_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    return [row[0] for row in rows]


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def safe_count(conn: sqlite3.Connection, table: str) -> Optional[int]:
    try:
        return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    except sqlite3.Error:
        return None


def pick_metrics_table(tables: List[str]) -> Optional[str]:
    exact = [name for name in tables if name.upper() == 'GPU_METRICS']
    if exact:
        return exact[0]
    candidates = [name for name in tables
                  if 'GPU_METRIC' in name.upper() and not name.upper().startswith('TARGET_INFO')]
    return candidates[0] if candidates else None


def interval_stats(conn: sqlite3.Connection, table: str,
                   columns: List[str]) -> Dict[str, Any]:
    lower = {name.lower(): name for name in columns}
    ts_col = lower.get('timestamp') or lower.get('start') or lower.get('rawtimestamp')
    if ts_col is None:
        return {'available': False, 'reason': f'{table} 无时间戳列，列={columns}'}
    group_col = lower.get('metricid') or lower.get('typeid') or lower.get('metricname')
    where = ''
    if group_col is not None:
        where = f'WHERE "{group_col}" = (SELECT MIN("{group_col}") FROM "{table}")'
    sql = (f'WITH t AS (SELECT "{ts_col}" AS ts, LAG("{ts_col}") OVER (ORDER BY "{ts_col}") '
           f'AS prev FROM "{table}" {where}) '
           f'SELECT ts - prev AS delta, COUNT(*) AS n FROM t WHERE prev IS NOT NULL '
           f'GROUP BY delta ORDER BY delta')
    try:
        rows = conn.execute(sql).fetchall()
    except sqlite3.Error as exc:
        return {'available': False, 'reason': f'SQL 失败: {exc}', 'sql': sql}
    if not rows:
        return {'available': False, 'reason': '无相邻样本对', 'sql': sql}
    total = sum(int(row[1]) for row in rows)
    hist = [(int(row[0]), int(row[1])) for row in rows]

    def quantile(fraction: float) -> int:
        target = fraction * total
        seen = 0
        for delta, count in hist:
            seen += count
            if seen >= target:
                return delta
        return hist[-1][0]

    span_row = conn.execute(
        f'SELECT MIN("{ts_col}"), MAX("{ts_col}") FROM "{table}" {where}').fetchone()
    return {
        'available': True,
        'sql': sql,
        'timestamp_column': ts_col,
        'group_column': group_col,
        'count': total + 1,
        'median': quantile(0.5),
        'p95': quantile(0.95),
        'min': hist[0][0],
        'max': hist[-1][0],
        'mean': sum(delta * count for delta, count in hist) / total,
        'distinct_deltas': len(hist),
        'top_deltas': sorted(hist, key=lambda item: -item[1])[:5],
        'window_ns': [int(span_row[0]), int(span_row[1])] if span_row and span_row[0] is not None else None,
    }


def interval_stats_strict(conn: sqlite3.Connection, table: str,
                          columns: List[str]) -> Dict[str, Any]:
    """Compute sampling intervals for every metric group, not just metric 0."""
    ts_col = _find_column(columns, 'timestamp', 'start', 'rawTimestamp')
    if ts_col is None:
        return {'available': False, 'reason': f'{table} has no timestamp column',
                'timestamp_column': None, 'group_column': None, 'groups': {}}
    group_col = _find_column(columns, 'metricId', 'typeId', 'metricName')
    try:
        if group_col is None:
            group_values: List[Any] = [None]
        else:
            group_values = [row[0] for row in conn.execute(
                f'SELECT DISTINCT "{group_col}" FROM "{table}" ORDER BY "{group_col}"')]
    except sqlite3.Error as exc:
        return {'available': False, 'reason': f'SQL failed while listing metric groups: {exc}',
                'timestamp_column': ts_col, 'group_column': group_col, 'groups': {}}
    if not group_values:
        return {'available': False, 'reason': 'metrics table has no groups/samples',
                'timestamp_column': ts_col, 'group_column': group_col, 'groups': {}}

    groups: Dict[str, Dict[str, Any]] = {}
    all_deltas: List[int] = []
    windows: List[Tuple[int, int]] = []
    for group_value in group_values:
        if group_col is None:
            sql = f'SELECT "{ts_col}" FROM "{table}" ORDER BY "{ts_col}"'
            params: Tuple[Any, ...] = ()
        elif group_value is None:
            sql = (f'SELECT "{ts_col}" FROM "{table}" '
                   f'WHERE "{group_col}" IS NULL ORDER BY "{ts_col}"')
            params = ()
        else:
            sql = (f'SELECT "{ts_col}" FROM "{table}" '
                   f'WHERE "{group_col}" = ? ORDER BY "{ts_col}"')
            params = (group_value,)
        key = str(group_value)
        try:
            values = [row[0] for row in conn.execute(sql, params)
                      if row[0] is not None]
            timestamps = [int(value) for value in values]
        except (sqlite3.Error, TypeError, ValueError) as exc:
            groups[key] = {
                'group_value': group_value, 'available': False,
                'sample_count': 0, 'interval_count': 0,
                'reason': f'timestamp is not numeric: {exc}',
            }
            continue
        deltas = [right - left for left, right in zip(timestamps, timestamps[1:])]
        item: Dict[str, Any] = {
            'group_value': group_value,
            'available': bool(deltas),
            'sample_count': len(timestamps),
            'interval_count': len(deltas),
            'window_ns': [timestamps[0], timestamps[-1]] if timestamps else None,
            'nonpositive_intervals': sum(delta <= 0 for delta in deltas),
        }
        if deltas:
            ordered = sorted(deltas)
            item.update({
                'median': _quantile_from_sorted(ordered, 0.5),
                'p95': _quantile_from_sorted(ordered, 0.95),
                'min': ordered[0],
                'max': ordered[-1],
                'mean': sum(deltas) / len(deltas),
                'distinct_deltas': len(set(deltas)),
                'top_deltas': sorted(Counter(deltas).items(),
                                     key=lambda pair: -pair[1])[:5],
            })
            all_deltas.extend(deltas)
            windows.append((timestamps[0], timestamps[-1]))
        else:
            item['reason'] = 'no adjacent sample pair'
        groups[key] = item

    available = [item for item in groups.values() if item.get('available')]
    if not available:
        return {
            'available': False,
            'reason': 'no metric group has an adjacent sample pair',
            'timestamp_column': ts_col, 'group_column': group_col,
            'groups': groups,
            'sample_count': sum(item.get('sample_count', 0) for item in groups.values()),
        }
    ordered_all = sorted(all_deltas)
    first = available[0]
    all_groups_available = all(item.get('available') for item in groups.values())
    return {
        'available': all_groups_available,
        'reason': None if all_groups_available else
                  'at least one metric group has no adjacent sample pair',
        'timestamp_column': ts_col,
        'group_column': group_col,
        'count': sum(item.get('sample_count', 0) for item in groups.values()),
        'sample_count': sum(item.get('sample_count', 0) for item in groups.values()),
        'interval_count': len(all_deltas),
        'median': first['median'],
        'p95': first['p95'],
        'min': min(item['min'] for item in available),
        'max': max(item['max'] for item in available),
        'mean': sum(all_deltas) / len(all_deltas),
        'distinct_deltas': len(set(all_deltas)),
        'top_deltas': sorted(Counter(all_deltas).items(),
                             key=lambda pair: -pair[1])[:5],
        'window_ns': [min(window[0] for window in windows),
                      max(window[1] for window in windows)] if windows else None,
        'groups': groups,
        'groups_checked': len(groups),
        'max_by_group': {key: item.get('max') for key, item in groups.items()},
    }


def kernel_probe(conn: sqlite3.Connection, tables: List[str],
                 needle: str) -> Dict[str, Any]:
    table = next((name for name in tables if name.upper() == 'CUPTI_ACTIVITY_KIND_KERNEL'), None)
    if table is None:
        return {'available': False, 'reason': '缺少 CUPTI_ACTIVITY_KIND_KERNEL 表'}
    total = safe_count(conn, table)
    result: Dict[str, Any] = {'available': True, 'table': table, 'total_rows': total}
    if 'StringIds' not in tables:
        result['match_rows'] = None
        result['reason'] = '缺少 StringIds 表，无法按名字匹配'
        return result
    columns = table_columns(conn, table)
    name_col = 'demangledName' if 'demangledName' in columns else (
        'shortName' if 'shortName' in columns else None)
    if name_col is None:
        result['match_rows'] = None
        result['reason'] = f'kernel 表无名字列，列={columns}'
        return result
    sql = (f'SELECT COUNT(*) FROM "{table}" k JOIN StringIds s ON s.id = k."{name_col}" '
           f'WHERE s.value LIKE ?')
    try:
        result['match_rows'] = int(conn.execute(sql, (f'%{needle}%',)).fetchone()[0])
        result['match_sql'] = sql
        result['name_column'] = name_col
    except sqlite3.Error as exc:
        result['match_rows'] = None
        result['reason'] = f'SQL 失败: {exc}'
    return result


def kernel_probe_strict(conn: sqlite3.Connection, tables: List[str],
                        needle: str, expected_devices: Optional[List[int]],
                        expected_process_count: Optional[int],
                        expected_launch_count: Optional[int] = None) -> Dict[str, Any]:
    """Validate an expected number of complete target launches.

    With the default contract this is one launch per expected device/process
    pair.  A repeated-launch capture can explicitly request N total launches;
    the expected counts are then distributed evenly across the declared
    device/process pairs.  CUPTI's kernel table is the only exported table that
    carries both deviceId and globalPid.  If a reduced SQLite export omits
    either field, row-count evidence is not sufficient and this check
    deliberately hard-fails.
    """
    table = next((name for name in tables
                  if name.upper() == 'CUPTI_ACTIVITY_KIND_KERNEL'), None)
    result: Dict[str, Any] = {
        'available': table is not None,
        'table': table,
        'needle': needle,
        'expected_devices': expected_devices,
        'expected_process_count': expected_process_count,
        'expected_launch_count': expected_launch_count,
        'hard_failures': [],
    }
    if table is None:
        result['hard_failures'].append(
            'SQLite schema capability failure: missing CUPTI_ACTIVITY_KIND_KERNEL table')
        return result
    columns = table_columns(conn, table)
    string_table = next((name for name in tables if name.lower() == 'stringids'), None)
    name_col = _find_column(columns, 'demangledName', 'shortName')
    canonical = {
        'start': _find_column(columns, 'start'),
        'end': _find_column(columns, 'end'),
        'deviceId': _find_column(columns, 'deviceId'),
        'globalPid': _find_column(columns, 'globalPid'),
        'name': name_col,
    }
    required = ['start', 'end', 'deviceId', 'globalPid', 'name']
    missing = [field for field in required if canonical[field] is None]
    result['schema_capability'] = {
        'required_fields': required,
        'available_columns': columns,
        'resolved_columns': canonical,
        'missing_fields': missing,
        'identifies_device_and_process': not missing,
    }
    if string_table is None:
        result['hard_failures'].append(
            'SQLite schema capability failure: missing StringIds table for kernel-name matching')
    if missing:
        result['hard_failures'].append(
            'SQLite schema capability failure: target launch validation requires fields '
            f'{missing}; available kernel columns={columns}')
    if result['hard_failures']:
        result['match_rows'] = None
        return result

    sql = (f'SELECT k."{canonical["start"]}", k."{canonical["end"]}", '
           f'k."{canonical["deviceId"]}", k."{canonical["globalPid"]}", '
           f's.value FROM "{table}" k JOIN "{string_table}" s '
           f'ON s.id = k."{canonical["name"]}" WHERE s.value LIKE ? '
           f'ORDER BY k."{canonical["start"]}"')
    try:
        rows = conn.execute(sql, (f'%{needle}%',)).fetchall()
    except sqlite3.Error as exc:
        result['hard_failures'].append(f'target kernel query failed: {exc}')
        result['match_rows'] = None
        result['match_sql'] = sql
        return result
    launches: List[Dict[str, Any]] = []
    for start, end, device, process, name in rows:
        complete = False
        duration = None
        try:
            if start is not None and end is not None:
                duration = int(end) - int(start)
                complete = duration > 0
        except (TypeError, ValueError):
            complete = False
        launches.append({
            'start': start,
            'end': end,
            'device_id': device,
            'global_pid': process,
            'name': name,
            'duration_ns': duration,
            'complete': complete,
        })
    result.update({
        'match_rows': len(launches),
        'match_sql': sql,
        'name_column': canonical['name'],
        'launches': launches,
        'complete_launches': sum(item['complete'] for item in launches),
        'incomplete_launches': [item for item in launches if not item['complete']],
    })
    device_counts: Dict[str, int] = {}
    process_counts: Dict[str, int] = {}
    pair_counts: Dict[str, int] = {}
    for item in launches:
        device_key = str(item['device_id'])
        process_key = str(item['global_pid'])
        pair_key = f'{device_key}/{process_key}'
        device_counts[device_key] = device_counts.get(device_key, 0) + 1
        process_counts[process_key] = process_counts.get(process_key, 0) + 1
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
    result['launch_count_by_device'] = device_counts
    result['launch_count_by_process'] = process_counts
    result['launch_count_by_device_process'] = pair_counts
    result['observed_devices'] = sorted(
        {item['device_id'] for item in launches if item['device_id'] is not None})
    result['observed_processes'] = sorted(
        {item['global_pid'] for item in launches if item['global_pid'] is not None})
    result['observed_device_process_pairs'] = sorted(pair_counts)

    if expected_devices is None or not expected_devices:
        result['hard_failures'].append(
            'target launch validation lacks a mappable expected device set; '
            'CUDA_VISIBLE_DEVICES must resolve to physical indices')
    else:
        expected_physical = list(dict.fromkeys(int(device) for device in expected_devices))
        observed_devices = result['observed_devices']
        logical_devices = list(range(len(expected_physical)))
        if observed_devices == sorted(expected_physical):
            expected_kernel_devices = sorted(expected_physical)
            device_id_space = 'physical-or-identity-visible'
            kernel_to_physical = {str(device): device for device in expected_kernel_devices}
        elif observed_devices == logical_devices:
            # CUPTI kernel rows use the CUDA-visible logical ordinal, while nsys
            # --gpu-metrics-devices uses the physical ordinal.  Preserve and
            # validate the explicit CUDA_VISIBLE_DEVICES order instead of
            # incorrectly requiring the physical IDs in the kernel table.
            expected_kernel_devices = logical_devices
            device_id_space = 'cuda-visible-logical'
            kernel_to_physical = {
                str(logical): physical
                for logical, physical in enumerate(expected_physical)
            }
        else:
            expected_kernel_devices = sorted(expected_physical)
            device_id_space = 'unresolved'
            kernel_to_physical = {}
        result['expected_devices'] = expected_kernel_devices
        result['expected_physical_devices'] = expected_physical
        result['kernel_device_id_space'] = device_id_space
        result['kernel_device_to_physical'] = kernel_to_physical
        result['expected_device_count'] = len(expected_kernel_devices)
        expected_total = (expected_launch_count if expected_launch_count is not None else
                          len(expected_kernel_devices) * max(expected_process_count or 0, 0))
        expected_per_device, remainder = divmod(
            expected_total, len(expected_kernel_devices))
        result['expected_launch_count_per_device'] = expected_per_device
        if remainder:
            result['hard_failures'].append(
                'expected launch count cannot be distributed evenly across expected devices: '
                f'total={expected_total}, devices={len(expected_kernel_devices)}')
        for device in expected_kernel_devices:
            count = device_counts.get(str(device), 0)
            if count != expected_per_device:
                result['hard_failures'].append(
                    f'target kernel must launch {expected_per_device} time(s) on device '
                    f'{device}; observed {count}')
    if expected_process_count is None or expected_process_count <= 0:
        result['hard_failures'].append(
            'target launch validation lacks a positive expected process count')
    else:
        if expected_launch_count is not None and expected_launch_count <= 0:
            result['hard_failures'].append(
                'expected launch count must be positive; '
                f'observed {expected_launch_count}')
        expected_device_count = len(result.get('expected_devices') or [])
        launch_count = (expected_launch_count if expected_launch_count is not None else
                        expected_device_count * expected_process_count)
        result['expected_total_launch_count'] = launch_count
        if len(launches) != launch_count:
            result['hard_failures'].append(
                'target kernel launch count mismatch: '
                f'expected {launch_count}, observed {len(launches)}')
        if len(result['observed_processes']) != expected_process_count:
            result['hard_failures'].append(
                'target kernel process count mismatch: '
                f'expected {expected_process_count}, observed {len(result["observed_processes"])}')
    if result['incomplete_launches']:
        result['hard_failures'].append(
            f'target kernel has {len(result["incomplete_launches"])} incomplete launch row(s); '
            'start/end must both be present and end > start')
    expected_per_pair = None
    if expected_devices and expected_process_count and expected_process_count > 0:
        pair_count = len(dict.fromkeys(int(device) for device in expected_devices))
        expected_total = (expected_launch_count if expected_launch_count is not None else
                          pair_count * expected_process_count)
        expected_per_pair, remainder = divmod(
            expected_total, pair_count * expected_process_count)
        result['expected_launch_count_per_pair'] = expected_per_pair
        if remainder:
            result['hard_failures'].append(
                'expected launch count cannot be distributed evenly across expected '
                f'device/process pairs: total={expected_total}, '
                f'pairs={pair_count * expected_process_count}')
    unexpected_pairs = [pair for pair, count in pair_counts.items()
                        if expected_per_pair is not None and count != expected_per_pair]
    if unexpected_pairs:
        result['hard_failures'].append(
            'target kernel launch count differs from the expected per-device/process count '
            f'for {unexpected_pairs}; expected_per_pair={expected_per_pair}')
    if not launches:
        result['hard_failures'].append(
            f'no target kernel launch matched kernel-name needle {needle!r}')
    result['valid'] = not result['hard_failures']
    return result


def required_metric_window_probe(conn: sqlite3.Connection, tables: List[str],
                                 metrics_table: Optional[str],
                                 kernel: Dict[str, Any]) -> Dict[str, Any]:
    """Require all four directional NVLink request/response counters in the target window."""
    result: Dict[str, Any] = {
        'required_metric_names': list(REQUIRED_HW_METRIC_NAMES),
        'available': False,
        'hard_failures': [],
    }
    info_table = next((name for name in tables
                       if name.upper() == 'TARGET_INFO_GPU_METRICS'), None)
    if metrics_table is None or info_table is None:
        result['hard_failures'].append(
            'NVLink request/response checks lack GPU_METRICS or '
            'TARGET_INFO_GPU_METRICS table')
        return result
    info_columns = table_columns(conn, info_table)
    metric_columns = table_columns(conn, metrics_table)
    info_name_col = _find_column(info_columns, 'metricName')
    info_id_col = _find_column(info_columns, 'metricId')
    metric_id_col = _find_column(metric_columns, 'metricId')
    ts_col = _find_column(metric_columns, 'timestamp', 'rawTimestamp')
    result['schema'] = {
        'info_table': info_table,
        'info_columns': info_columns,
        'metrics_table': metrics_table,
        'metrics_columns': metric_columns,
        'metric_name_column': info_name_col,
        'info_metric_id_column': info_id_col,
        'metrics_metric_id_column': metric_id_col,
        'timestamp_column': ts_col,
    }
    missing = [name for name, value in (
        ('TARGET_INFO_GPU_METRICS.metricName', info_name_col),
        ('TARGET_INFO_GPU_METRICS.metricId', info_id_col),
        ('GPU_METRICS.metricId', metric_id_col),
        ('GPU_METRICS.timestamp/rawTimestamp', ts_col),
    ) if value is None]
    if missing:
        result['hard_failures'].append(
            'NVLink request/response checks have missing schema fields: '
            f'{missing}')
        return result
    try:
        metric_rows = conn.execute(
            f'SELECT "{info_id_col}", "{info_name_col}" FROM "{info_table}"').fetchall()
    except sqlite3.Error as exc:
        result['hard_failures'].append(f'cannot read GPU metric names: {exc}')
        return result
    required_rows = [
        (metric_id, name) for metric_id, name in metric_rows
        if isinstance(name, str)
        and any(required in name for required in REQUIRED_HW_METRIC_NAMES)
    ]
    result['matching_metric_rows'] = [
        {'metric_id': metric_id, 'metric_name': name}
        for metric_id, name in required_rows]
    missing_required = [
        required for required in REQUIRED_HW_METRIC_NAMES
        if not any(required in name for _metric_id, name in required_rows)
    ]
    if missing_required:
        result['hard_failures'].append(
            f'required NVLink counters are absent: {missing_required}')
        return result
    windows = []
    for launch in kernel.get('launches', []):
        if launch.get('complete') and launch.get('start') is not None and launch.get('end') is not None:
            windows.append((int(launch['start']), int(launch['end'])))
    if not windows:
        result['hard_failures'].append(
            'cannot check required NVLink counters in target kernel window: '
            'no complete target launch window')
        return result
    window = [min(item[0] for item in windows), max(item[1] for item in windows)]
    result['target_kernel_window_ns'] = window
    counts: Dict[str, int] = {}
    for metric_id, metric_name in required_rows:
        try:
            count = int(conn.execute(
                f'SELECT COUNT(*) FROM "{metrics_table}" '
                f'WHERE "{metric_id_col}" = ? AND "{ts_col}" >= ? AND "{ts_col}" <= ?',
                (metric_id, window[0], window[1])).fetchone()[0])
        except sqlite3.Error as exc:
            result['hard_failures'].append(
                f'cannot count NVLink samples for {metric_name!r} '
                f'(metric_id={metric_id}): {exc}')
            continue
        counts[str(metric_id)] = count
        if count <= 0:
            result['hard_failures'].append(
                f'{metric_name} has zero samples in target kernel window '
                f'for metric_id={metric_id}')
    result['sample_counts_in_kernel_window'] = counts
    result['available'] = True
    result['valid'] = not result['hard_failures']
    return result


def verify_evidence(sqlite_path: Path, log_text: str, frequency_hz: int,
                    kernel_needle: str,
                    expected_devices: Optional[List[int]] = None,
                    expected_process_count: Optional[int] = None,
                    expected_launch_count: Optional[int] = None,
                    capture_contract: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {
        'expected_interval_ns': 1e9 / frequency_hz if frequency_hz else None,
        'log_findings': scan_log(log_text),
        'sqlite': {'available': sqlite_path.is_file(), 'path': str(sqlite_path)},
        'failures': [],
        'warnings': [],
    }
    if frequency_hz <= 0:
        evidence['failures'].append(f'HW sampling frequency must be positive; observed {frequency_hz} Hz')
    elif frequency_hz != DEFAULT_FREQUENCY_HZ:
        evidence['failures'].append(
            f'HW sampling frequency {frequency_hz} Hz differs from required '
            f'{DEFAULT_FREQUENCY_HZ} Hz; shorten/fix the cudaProfilerApi window and recapture')
    if capture_contract is not None:
        evidence['capture_contract'] = dict(capture_contract)
        if capture_contract.get('capture_range') != DEFAULT_CAPTURE_RANGE:
            evidence['failures'].append(
                'HW capture range must be cudaProfilerApi (kernel-only)')
        if capture_contract.get('capture_range_end') != DEFAULT_CAPTURE_RANGE_END:
            evidence['failures'].append(
                'HW capture range end must be stop for a kernel-only range')
        for field in ('flush_on_cudaprofiler_stop', 'trace_fork_before_exec'):
            if capture_contract.get(field) is not True:
                evidence['failures'].append(
                    f'HW capture control {field}=true is required')
        metrics_set = capture_contract.get('metrics_set')
        if not metrics_set:
            evidence['failures'].append(
                'HW capture metrics configuration is missing from the manifest/command')
    for kind in BLOCKING_KINDS:
        hit = evidence['log_findings'].get(kind, {})
        if hit.get('count'):
            evidence['failures'].append(
                f'nsys 日志出现 {kind} 相关行 {hit["count"]} 条: {hit["lines"][:3]}')
    for kind in ('missing', 'incomplete', 'permission', 'error'):
        hit = evidence['log_findings'].get(kind, {})
        if hit.get('count'):
            evidence['warnings'].append(
                f'nsys 日志出现 {kind} 相关行 {hit["count"]} 条: {hit["lines"][:3]}')
    if not sqlite_path.is_file():
        evidence['failures'].append('sqlite 未生成，无法做证据校验')
        return evidence
    conn = sqlite3.connect(f'file:{sqlite_path}?mode=ro', uri=True)
    try:
        tables = sqlite_tables(conn)
        evidence['sqlite']['tables'] = tables
        evidence['sqlite']['row_counts'] = {name: safe_count(conn, name) for name in tables}
        suspicious = [name for name in tables
                      if re.search(r'OVERFLOW|DROP|DISCARD|LOST', name, re.IGNORECASE)]
        evidence['sqlite']['suspicious_tables'] = {
            name: safe_count(conn, name) for name in suspicious}
        for name, count in evidence['sqlite']['suspicious_tables'].items():
            if count:
                evidence['failures'].append(f'表 {name} 有 {count} 行（overflow/dropped 迹象）')
        metrics_table = pick_metrics_table(tables)
        evidence['sqlite']['metrics_table'] = metrics_table
        stats: Dict[str, Any] = {}
        columns: List[str] = []
        if metrics_table is None:
            evidence['failures'].append('sqlite 中没有 GPU metrics 表，GPU Metrics 未采到')
        else:
            columns = table_columns(conn, metrics_table)
            evidence['sqlite']['metrics_columns'] = columns
            rows = safe_count(conn, metrics_table)
            evidence['sqlite']['metrics_rows'] = rows
            if not rows:
                evidence['failures'].append(f'{metrics_table} 为空，GPU Metrics 未采到')
            else:
                stats = interval_stats_strict(conn, metrics_table, columns)
                evidence['sampling_interval_ns'] = stats
                if not stats.get('available'):
                    evidence['failures'].append(
                        'GPU Metrics sampling interval is not verifiable for every metric group: '
                        f'{stats.get("reason")}')
                expected = evidence['expected_interval_ns']
                if stats.get('available') and expected:
                    median = stats['median']
                    rel = abs(median - expected) / expected
                    evidence['sampling_interval_ns']['relative_error_vs_expected'] = rel
                    if rel > 0.25:
                        evidence['warnings'].append(
                            f'采样中位间隔 {median} ns 与期望 {expected:.0f} ns 相差 {rel * 100:.1f}%')
                    gap_threshold = expected * 10
                    if stats['max'] > gap_threshold:
                        evidence['warnings'].append(
                            f'存在 {stats["max"]} ns 的采样间隙（> 10x 期望间隔），可能有丢样本')
        if stats.get('available') and evidence['expected_interval_ns']:
            threshold = evidence['expected_interval_ns'] * 10
            bad_groups = {
                key: value for key, value in stats.get('groups', {}).items()
                if value.get('max') is None or value.get('max') > threshold
            }
            evidence['sampling_interval_ns']['hard_limit_ns'] = threshold
            evidence['sampling_interval_ns']['max_check'] = {
                'threshold_ns': threshold,
                'observed_max_ns': stats.get('max'),
                'bad_groups': bad_groups,
                'pass': not bad_groups and stats.get('sample_count', 0) > 0,
            }
            if stats.get('sample_count', 0) <= 0:
                evidence['failures'].append('GPU Metrics sample count is zero')
            if bad_groups:
                evidence['failures'].append(
                    'GPU Metrics sampling interval exceeds 10x expected period: '
                    f'{bad_groups}')
        target_info = [name for name in tables if name.upper().startswith('TARGET_INFO')]
        evidence['sqlite']['target_info_tables'] = target_info
        metric_info_table = next(
            (name for name in target_info if name.upper() == 'TARGET_INFO_GPU_METRICS'),
            target_info[0] if target_info else None)
        sample_check: Dict[str, Any] = {
            'available': False,
            'table': metric_info_table,
            'sample_counts_by_metric': {},
            'zero_sample_metrics': [],
            'hard_failures': [],
        }
        if metric_info_table is None:
            sample_check['hard_failures'].append(
                'SQLite schema capability failure: missing TARGET_INFO_GPU_METRICS '
                'to identify required counter samples')
        else:
            info_columns = table_columns(conn, metric_info_table)
            info_metric_col = _find_column(info_columns, 'metricId')
            metrics_metric_col = _find_column(columns, 'metricId')
            if info_metric_col is None or metrics_metric_col is None:
                sample_check['hard_failures'].append(
                    'SQLite schema capability failure: metricId is required to '
                    f'account for every counter; info columns={info_columns}, '
                    f'metrics columns={columns}')
            else:
                expected_ids = [row[0] for row in conn.execute(
                    f'SELECT DISTINCT "{info_metric_col}" FROM "{metric_info_table}" '
                    f'WHERE "{info_metric_col}" IS NOT NULL')]
                observed_counts = {
                    row[0]: int(row[1]) for row in conn.execute(
                        f'SELECT "{metrics_metric_col}", COUNT(*) FROM "{metrics_table}" '
                        f'GROUP BY "{metrics_metric_col}"')}
                sample_check['expected_metric_ids'] = expected_ids
                sample_check['sample_counts_by_metric'] = observed_counts
                sample_check['zero_sample_metrics'] = [
                    metric_id for metric_id in expected_ids
                    if observed_counts.get(metric_id, 0) <= 0]
                sample_check['available'] = bool(expected_ids)
                if not expected_ids:
                    sample_check['hard_failures'].append(
                        'TARGET_INFO_GPU_METRICS contains no metric IDs')
                if sample_check['zero_sample_metrics']:
                    sample_check['hard_failures'].append(
                        'one or more required GPU Metrics counters have zero samples: '
                        f'{sample_check["zero_sample_metrics"]}')
        evidence['gpu_metrics_samples'] = sample_check
        evidence['failures'].extend(sample_check['hard_failures'])
        evidence['kernel'] = kernel_probe_strict(
            conn, tables, kernel_needle, expected_devices, expected_process_count,
            expected_launch_count)
        evidence['failures'].extend(evidence['kernel'].get('hard_failures', []))
        evidence['required_hw_metrics'] = required_metric_window_probe(
            conn, tables, metrics_table, evidence['kernel'])
        evidence['failures'].extend(
            evidence['required_hw_metrics'].get('hard_failures', []))
        if evidence['kernel'].get('available') and evidence['kernel'].get('match_rows') == 0:
            evidence['failures'].append(
                f'CUPTI kernel 表里没有匹配 {kernel_needle} 的 launch：'
                f'needle 写错了(常见于沿用默认的 sm90_fp8_mega_moe_impl 去采别的项目)，'
                f'这次采集没有采到目标 kernel，counter 会结构性全零。'
                f'从 sqlite 里 grep 真实 kernel 名后重采：'
                f'SELECT DISTINCT s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k '
                f'JOIN StringIds s ON s.id=k.demangledName')
    finally:
        conn.close()
    evidence['verdict'] = 'FAIL' if evidence['failures'] else 'PASS'
    return evidence


def estimate_sizes(frequency_hz: int, metrics_count: int,
                   seconds: float) -> Dict[str, Any]:
    samples = frequency_hz * seconds
    rows = samples * metrics_count
    return {
        'assumed_capture_seconds': seconds,
        'assumed_metric_count': metrics_count,
        'estimated_metric_samples': samples,
        'estimated_metric_rows': rows,
        'estimated_nsys_rep_bytes': rows * 1,
        'estimated_sqlite_bytes': rows * 32,
        'note': '系数来自实测 probe（gh100 集 10kHz ~3.5s: 1.04e6 行 -> sqlite 33.1 MB '
                '(~32 B/行), nsys-rep 0.97 MB (~1 B/行)）；仅 GPU metrics 行的粗估，用于容量预警',
    }


DEFAULT_METRIC_COUNT_ESTIMATE = 32


def count_config_metrics(config_path: Optional[str]) -> int:
    if not config_path:
        # Built-in metric sets do not expose a YAML path.  This value is used
        # only for disk-capacity estimation, so use a conservative finite
        # fallback instead of referencing a configuration-specific constant.
        return DEFAULT_METRIC_COUNT_ESTIMATE
    try:
        text = Path(config_path).read_text(encoding='utf-8')
    except OSError:
        return DEFAULT_METRIC_COUNT_ESTIMATE
    inside = False
    count = 0
    for line in text.splitlines():
        if re.match(r'^\s{2}metrics:\s*$', line):
            inside = True
            continue
        if inside:
            if re.match(r'^\s{2}-\s+name:', line):
                count += 1
            elif re.match(r'^\S', line):
                break
    return count


def inspect_metrics_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Validate the custom metrics file before any GPU process is started."""
    result: Dict[str, Any] = {
        'path': config_path,
        'file_exists': False,
        'file_stem': None,
        'aliases': [],
        'raw_counter_names': [],
        'raw_counter_count': 0,
        'required_counters': list(REQUIRED_HW_METRIC_NAMES),
        'hard_failures': [],
    }
    if not config_path:
        result['skipped_builtin_metrics_set'] = True
        return result
    path = Path(config_path).expanduser().resolve()
    result['path'] = str(path)
    result['file_stem'] = path.stem
    if not path.is_file():
        result['hard_failures'].append(f'metrics config does not exist: {path}')
        return result
    result['file_exists'] = True
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeError) as exc:
        result['hard_failures'].append(f'cannot read metrics config {path}: {exc}')
        return result
    aliases = []
    alias_indices: List[int] = []
    for index, line in enumerate(lines):
        match = re.match(r'^\s*-\s+alias:\s*([^\s#]+)', line)
        if match:
            aliases.append(match.group(1))
            alias_indices.append(index)
    result['aliases'] = aliases
    selected_alias = None
    if aliases:
        if path.stem in aliases:
            selected_alias = path.stem
            result['alias_matches_file_stem'] = True
        else:
            result['alias_matches_file_stem'] = False
            result['hard_failures'].append(
                'custom metrics config filename stem must match a YAML alias: '
                f'stem={path.stem!r}, aliases={aliases!r}; nsys rejects a file whose '
                'stem and selected metric-set alias differ')
    else:
        result['alias_matches_file_stem'] = False
        result['hard_failures'].append(
            'custom metrics config contains no metricSets[*].alias')
    result['selected_alias'] = selected_alias
    raw_names: List[str] = []
    raw_section = False
    for line in lines:
        if re.match(r'^\s{2}metrics:\s*$', line):
            raw_section = True
            continue
        if raw_section:
            if re.match(r'^\s*-\s+alias:', line):
                break
            match = re.match(r'^\s{2}-\s+name:\s*(.*?)\s*$', line)
            if match:
                raw_names.append(match.group(1))
            elif line and not line.startswith(' '):
                break
    result['raw_counter_names'] = raw_names
    result['raw_counter_count'] = len(raw_names)
    if not raw_names:
        result['hard_failures'].append('custom metrics config contains no counters')
    result['valid'] = not result['hard_failures']
    return result


def resolve_cuda_home(requested: Optional[str]) -> Dict[str, Any]:
    """Record an optional CUDA installation; nsys capture does not require nvcc."""
    candidates: List[Tuple[str, Optional[str]]] = []
    if requested:
        candidates.append(('--cuda-home', requested))
    else:
        candidates.extend((
            ('CUDA_HOME', os.environ.get('CUDA_HOME')),
            ('CUDA_PATH', os.environ.get('CUDA_PATH')),
        ))
    result: Dict[str, Any] = {
        'requested': requested,
        'candidates': [{'source': source, 'value': value} for source, value in candidates],
        'resolved': None,
        'nvcc': None,
        'hard_failures': [],
    }
    for source, value in candidates:
        if not value:
            continue
        home = Path(value).expanduser().resolve()
        nvcc = home / 'bin' / 'nvcc'
        if home.is_dir() and nvcc.is_file():
            result['source'] = source
            result['resolved'] = str(home)
            result['nvcc'] = str(nvcc)
            result['valid'] = True
            return result
        if requested:
            break
    result['valid'] = False
    return result


def inject_cuda_env(payload: List[str], cuda_home: str) -> List[str]:
    """Make CUDA_HOME and CUDA_PATH explicit in the nsys child command."""
    assignments = [f'CUDA_HOME={cuda_home}', f'CUDA_PATH={cuda_home}']
    if payload and payload[0] == 'env':
        return [payload[0], *assignments, *payload[1:]]
    return ['env', *assignments, *payload]


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='nsys GPU Metrics 采集包装器')
    parser.add_argument('--root', required=False, default=None)
    parser.add_argument('--harness', default=None)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--gpu-metrics-device', type=int, default=0,
                        help='只允许一个物理 GPU；多卡高频采样会导致 sampling buffer overflow')
    parser.add_argument('--gpu-metrics-frequency', type=int, default=DEFAULT_FREQUENCY_HZ)
    parser.add_argument('--metrics-set', default=DEFAULT_METRICS_SET,
                        help='nsys GPU metrics 集；默认 file:<脚本同目录 nsys_metrics_ep_moe_hopper.config> '
                             '作为 Hopper 示例。auto=按 --metrics-config-name 搜索 file: '
                             '自定义 config；也可传 file:<path> 或内置 alias 如 gh100。'
                             '下游 extract 按本次硬件与通路对应的指标清单过滤')
    parser.add_argument('--metrics-config', default=None)
    parser.add_argument('--metrics-config-name', default=DEFAULT_METRICS_CONFIG_NAME)
    parser.add_argument('--metrics-search-root', action='append', default=None)
    parser.add_argument('--metrics-search-budget-sec', type=float, default=20.0,
                        help='有界兜底搜索 metrics 配置的墙钟预算，超时即放弃并回退')
    parser.add_argument('--fallback-metrics-set', default=DEFAULT_FALLBACK_METRICS_SET)
    parser.add_argument('--num-processes', type=int, default=2)
    parser.add_argument('--expected-launch-count', type=int, default=None,
                        help='目标 kernel 的总 launch 数；默认等于 device/process pair 数 '
                             '（device 数 * --num-processes），并要求平均分配')
    parser.add_argument('--cuda-visible-devices', default=None,
                        help='传给被采集进程的 CUDA_VISIBLE_DEVICES；默认沿用当前环境')
    parser.add_argument('--batches', type=int, nargs='+', default=[32])
    parser.add_argument('--max-records', type=int, default=4096,
                        help='per-track event capacity; leave headroom for the 0.80 occupancy gate')
    parser.add_argument('--harness-extra', action='append', default=None)
    parser.add_argument('--harness-skip-overhead', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='给 harness 加 --skip-overhead。默认 False：采 hw 时必须多次 launch，'
                             '因为 nsys GPU Metrics 有周期性 ~1 秒采样 gap，单次 2-3ms 短 kernel 大概率'
                             '落进 gap，hw counter 采样严重不足(实测仅覆盖 24%%)。'
                             'skip-overhead 只适合 sw-only 快路径(不采 hw)。'
                             '要用 skip-overhead 采 hw，请同时 --batches 32 之类跑多次 launch')
    parser.add_argument('--nsys-extra', action='append', default=None)
    parser.add_argument('--nsys', default=None)
    parser.add_argument('--cuda-home', default=None,
                        help='CUDA 安装根目录；必须包含可执行的 bin/nvcc，并显式传给 nsys payload')
    parser.add_argument('--harness-tmpdir', default=None,
                        help='harness 专用 TMPDIR（必须是本地 tmpfs 短路径，默认自动用 '
                             '/dev/shm/...）；避开 symm_mem UDS 的 108B sun_path 限制与 NAS 无法 bind')
    parser.add_argument('--trace', default='cuda,nvtx')
    parser.add_argument('--capture-range', default=DEFAULT_CAPTURE_RANGE,
                        choices=('none', 'cudaProfilerApi', 'nvtx'),
                        help='nsys capture range；cudaProfilerApi 会联动默认 harness 的 --cuda-profiler-range')
    parser.add_argument('--capture-range-end', default='stop',
                        choices=('none', 'stop', 'stop-shutdown'),
                        help='nsys capture range 结束策略；cudaProfilerApi 推荐 stop')
    parser.add_argument('--export-tables', default=None)
    parser.add_argument('--export-all-tables', action='store_true')
    parser.add_argument('--kernel-name-contains', default='sm90_fp8_mega_moe_impl')
    parser.add_argument('--mem-threshold-mib', type=float, default=1024.0)
    parser.add_argument('--util-threshold-pct', type=float, default=10.0)
    parser.add_argument('--estimate-seconds', type=float, default=60.0)
    parser.add_argument('--timeout-sec', type=float, default=3600.0)
    parser.add_argument('--payload', nargs=argparse.REMAINDER, default=None,
                        help='替代 harness 的原样命令，必须放在最后')
    parser.add_argument('--skip-export', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--validate-existing', default=None, metavar='PATH',
                        help='只读校验已有 capture_manifest.json/目录，不启动 nsys 或 GPU')
    parser.add_argument('--validation-output', default=None, metavar='PATH',
                        help='可选：把补强后的校验 manifest 写到指定路径')
    return parser.parse_args(argv)


def _command_option_value(command: List[str], option: str) -> Optional[str]:
    for index, item in enumerate(command):
        if item.startswith(option + '='):
            return item.split('=', 1)[1]
        if item == option and index + 1 < len(command):
            candidate = command[index + 1]
            if not candidate.startswith('--'):
                return candidate
    return None


def _command_option_bool(command: List[str], option: str) -> Optional[bool]:
    values: List[bool] = []
    for index, item in enumerate(command):
        if item == option:
            values.append(True)
        elif item.startswith(option + '='):
            parsed = _parse_bool_token(item.split('=', 1)[1])
            if parsed is not None:
                values.append(parsed)
    if not values:
        return None
    return False if any(value is False for value in values) else True


def _command_option_int(command: List[str], option: str) -> Optional[int]:
    value = _command_option_value(command, option)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _load_existing_manifest(path_arg: str) -> Tuple[Path, Dict[str, Any]]:
    source = Path(path_arg).expanduser()
    if source.is_dir():
        manifest_path = source / 'capture_manifest.json'
    elif source.is_file():
        manifest_path = source
    else:
        manifest_path = source / 'capture_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'找不到已有 manifest: {manifest_path}')
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'读取已有 manifest 失败: {manifest_path}: {exc}') from exc
    if not isinstance(manifest, dict):
        raise ValueError(f'已有 manifest 顶层必须是 JSON object: {manifest_path}')
    return manifest_path, manifest


def validate_existing_capture(args: argparse.Namespace) -> int:
    """Read-only validation path for an already completed HW capture."""
    try:
        manifest_path, original = _load_existing_manifest(args.validate_existing)
    except (FileNotFoundError, ValueError) as exc:
        print(f'[validate-existing] ERROR {exc}', file=sys.stderr)
        return 4
    out_dir = manifest_path.parent
    sqlite_path = out_dir / 'capture.sqlite'
    manifest_sqlite = ((original.get('evidence') or {}).get('sqlite') or {}).get('path')
    if manifest_sqlite and Path(manifest_sqlite).is_file():
        sqlite_path = Path(manifest_sqlite)
    if not sqlite_path.is_file():
        print(f'[validate-existing] ERROR 找不到 SQLite: {sqlite_path}', file=sys.stderr)
        return 4

    nsys_command = original.get('nsys_command')
    if not isinstance(nsys_command, list):
        line = original.get('nsys_command_line', '')
        nsys_command = shlex.split(line) if line else []
    nsys_command = [str(item) for item in nsys_command]
    payload_command = original.get('payload_command')
    if not isinstance(payload_command, list):
        payload_line = original.get('payload_command_line', '')
        payload_command = shlex.split(payload_line) if payload_line else []
    payload_command = [str(item) for item in payload_command]
    argv = original.get('argv') if isinstance(original.get('argv'), list) else []
    argv = [str(item) for item in argv]

    visible = original.get('cuda_visible_devices')
    expected_devices: Optional[List[int]] = None
    if isinstance(visible, dict) and visible.get('harness_physical_indices'):
        try:
            expected_devices = [int(item) for item in visible['harness_physical_indices']]
        except (TypeError, ValueError):
            expected_devices = None
    if args.cuda_visible_devices:
        resolved = resolve_visible_devices(args.cuda_visible_devices, args.num_processes)
        if resolved['mappable']:
            expected_devices = resolved['harness_physical_indices']
    expected_process_count = original.get('num_processes')
    if expected_process_count is None:
        for command in (payload_command, argv, nsys_command):
            expected_process_count = _command_option_int(command, '--num-processes')
            if expected_process_count is not None:
                break
    if expected_process_count is None and expected_devices:
        expected_process_count = len(expected_devices)
    if expected_process_count is None:
        expected_process_count = args.num_processes
    expected_launch_count = original.get('expected_launch_count')
    if expected_launch_count is None:
        for command in (argv, payload_command, nsys_command):
            expected_launch_count = _command_option_int(
                command, '--expected-launch-count')
            if expected_launch_count is not None:
                break

    frequency = original.get('frequency_hz')
    if frequency is None:
        frequency = _command_option_int(nsys_command, '--gpu-metrics-frequency')
    if frequency is None:
        frequency = args.gpu_metrics_frequency
    try:
        frequency = int(frequency)
    except (TypeError, ValueError):
        frequency = 0
    capture_range = original.get('capture_range') or \
        _command_option_value(nsys_command, '--capture-range')
    capture_range_end = original.get('capture_range_end') or \
        _command_option_value(nsys_command, '--capture-range-end')
    metrics_set = original.get('metrics_set') or \
        _command_option_value(nsys_command, '--gpu-metrics-set')
    metrics_config_path = original.get('metrics_config_path')
    if not metrics_config_path and isinstance(metrics_set, str) and metrics_set.startswith('file:'):
        metrics_config_path = metrics_set[len('file:'):]
    metrics_preflight = inspect_metrics_config(metrics_config_path)
    if not metrics_config_path:
        metrics_preflight = dict(metrics_preflight)
        metrics_preflight['legacy_unrecorded'] = True
        metrics_preflight['warning'] = (
            'existing manifest has no custom metrics config path; '
            'built-in metrics alias was not re-parsed')
    cuda_home_recorded = original.get('cuda_home') or original.get('cuda_path')
    if not cuda_home_recorded and isinstance(original.get('cuda_preflight'), dict):
        cuda_home_recorded = original['cuda_preflight'].get('resolved')
    if not cuda_home_recorded and isinstance(original.get('child_env_overrides'), dict):
        cuda_home_recorded = (original['child_env_overrides'].get('CUDA_HOME') or
                              original['child_env_overrides'].get('CUDA_PATH'))
    if not cuda_home_recorded:
        for item in payload_command:
            if item.startswith('CUDA_HOME='):
                cuda_home_recorded = item.split('=', 1)[1]
                break
    if cuda_home_recorded:
        cuda_preflight = resolve_cuda_home(str(cuda_home_recorded))
    else:
        cuda_preflight = {
            'resolved': None,
            'nvcc': None,
            'hard_failures': [],
            'legacy_unrecorded': True,
            'warning': 'existing manifest predates explicit CUDA_HOME/CUDA_PATH recording',
        }
    contract = {
        'capture_range': capture_range,
        'capture_range_end': capture_range_end,
        'flush_on_cudaprofiler_stop': (
            original.get('flush_on_cudaprofiler_stop')
            if 'flush_on_cudaprofiler_stop' in original else
            _command_option_bool(nsys_command, '--flush-on-cudaprofilerstop')),
        'trace_fork_before_exec': (
            original.get('trace_fork_before_exec')
            if 'trace_fork_before_exec' in original else
            _command_option_bool(nsys_command, '--trace-fork-before-exec')),
        'frequency_hz': frequency,
        'metrics_set': metrics_set,
        'metrics_config_path': metrics_config_path,
        'metrics_config_preflight': metrics_preflight,
        'cuda_home': cuda_preflight.get('resolved'),
        'cuda_path': cuda_preflight.get('resolved'),
    }
    log_parts: List[str] = []
    for log_name in ('capture_nsys.log', 'capture_export.log'):
        log_path = out_dir / log_name
        if log_path.is_file():
            try:
                log_text = log_path.read_text(encoding='utf-8', errors='replace')
                # run_capture returns child output without its shell header;
                # remove that header here too so table names such as OVERFLOW
                # are not misclassified as runtime overflow evidence.
                if log_text.startswith('$ '):
                    log_text = log_text.split('\n', 1)[1] if '\n' in log_text else ''
                log_parts.append(log_text)
            except OSError:
                pass
    kernel_needle = original.get('kernel_name_contains') or args.kernel_name_contains
    evidence = verify_evidence(
        sqlite_path, '\n'.join(log_parts), frequency, kernel_needle,
        expected_devices=expected_devices,
        expected_process_count=expected_process_count,
        expected_launch_count=expected_launch_count,
        capture_contract=contract)
    evidence['metrics_config_preflight'] = metrics_preflight
    evidence['cuda_preflight'] = cuda_preflight
    evidence['failures'].extend(metrics_preflight.get('hard_failures', []))
    evidence['failures'].extend(cuda_preflight.get('hard_failures', []))
    evidence['verdict'] = 'FAIL' if evidence['failures'] else 'PASS'
    strengthened = dict(original)
    strengthened['capture_contract'] = contract
    strengthened['metrics_config_preflight'] = metrics_preflight
    strengthened['cuda_preflight'] = cuda_preflight
    strengthened['cuda_home'] = cuda_preflight.get('resolved')
    strengthened['cuda_path'] = cuda_preflight.get('resolved')
    strengthened['num_processes'] = expected_process_count
    strengthened['expected_launch_count'] = expected_launch_count
    strengthened['kernel_name_contains'] = kernel_needle
    strengthened['expected_target_devices'] = expected_devices
    strengthened['validation_source_manifest'] = str(manifest_path)
    strengthened['evidence'] = evidence
    strengthened['verdict'] = evidence['verdict']
    if args.validation_output:
        output = Path(args.validation_output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(strengthened, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'[validate-existing] manifest {output}')
    else:
        print(json.dumps(strengthened, indent=2, ensure_ascii=False))
    return 0 if evidence['verdict'] == 'PASS' else 4


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    if args.validate_existing is not None:
        return validate_existing_capture(args)
    if not args.root:
        print('[fatal] 普通 HW capture 必须指定 --root')
        return 1
    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f'[fatal] --root 不是目录: {root}')
        return 1
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else \
        (root / '.work' / 'profile_runs' / 'hw_capture').resolve()
    if out_dir == root or root not in out_dir.parents:
        print(f'[fatal] out-dir 必须在 --root 之下且不能是仓库根: {out_dir}')
        return 1
    if args.capture_range != DEFAULT_CAPTURE_RANGE:
        print('[fatal] HW capture requires --capture-range cudaProfilerApi (kernel-only)')
        return 1
    if args.capture_range_end != DEFAULT_CAPTURE_RANGE_END:
        print('[fatal] HW capture requires --capture-range-end stop')
        return 1
    try:
        args.nsys_extra, nsys_controls = normalize_nsys_controls(args.nsys_extra)
    except ValueError as exc:
        print(f'[fatal] {exc}')
        return 1
    if args.num_processes <= 0:
        print('[fatal] --num-processes must be positive')
        return 1
    nsys = args.nsys or shutil.which('nsys')
    if not nsys:
        print('[fatal] 找不到 nsys 可执行文件，用 --nsys 指定')
        return 1
    if args.gpu_metrics_frequency < 10:
        print('[fatal] --gpu-metrics-frequency 最小为 10')
        return 1

    metrics = resolve_metrics_set(args, script_dir, root)
    metrics_preflight = inspect_metrics_config(metrics['config_path'])
    if metrics_preflight['hard_failures']:
        for failure in metrics_preflight['hard_failures']:
            print(f'[fatal] {failure}')
        return 1
    metric_count = metrics_preflight['raw_counter_count'] or \
        count_config_metrics(metrics['config_path'])
    cuda_preflight = resolve_cuda_home(args.cuda_home)
    harness_tmpdir_auto = args.harness_tmpdir is None
    harness_run_id = hashlib.sha256(str(out_dir.resolve()).encode('utf-8')).hexdigest()[:16]
    harness_tmpdir = Path(args.harness_tmpdir).expanduser() if args.harness_tmpdir else \
        Path('/dev/shm') / f'megamoe_hwcap_{harness_run_id}'
    payload_dir_placeholder = out_dir
    payload = build_payload(args, root, payload_dir_placeholder, str(harness_tmpdir))
    if cuda_preflight['resolved']:
        payload = inject_cuda_env(payload, cuda_preflight['resolved'])
    if '--cuda-profiler-range' not in payload:
        print('[fatal] HW capture payload must enable --cuda-profiler-range so only '
              'the target kernel window is collected')
        return 1
    out_prefix = out_dir / 'capture'
    rep_path = out_dir / 'capture.nsys-rep'
    sqlite_path = out_dir / 'capture.sqlite'
    nsys_cmd = build_nsys_cmd(args, nsys, metrics['value'], out_prefix, payload)
    export_cmd = build_export_cmd(args, nsys, rep_path, sqlite_path)

    try:
        version = subprocess.run([nsys, '--version'], check=True, capture_output=True,
                                 text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f'[fatal] nsys --version 失败: {exc}')
        return 1

    cuda_visible = args.cuda_visible_devices if args.cuda_visible_devices is not None \
        else os.environ.get('CUDA_VISIBLE_DEVICES')
    visible = resolve_visible_devices(cuda_visible, args.num_processes)
    child_env = dict(os.environ)
    if cuda_preflight['resolved']:
        child_env['CUDA_HOME'] = cuda_preflight['resolved']
        child_env['CUDA_PATH'] = cuda_preflight['resolved']
    if cuda_visible is not None:
        child_env['CUDA_VISIBLE_DEVICES'] = cuda_visible
    child_env.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    nsys_tmpdir = out_dir / 'nsys_tmp'
    child_env['TMPDIR'] = str(nsys_tmpdir)
    dg_tools_was_set = 'DG_USE_NVIDIA_TOOLS' in child_env
    child_env.setdefault('DG_USE_NVIDIA_TOOLS', '1')
    env_overrides = {
        'TMPDIR': child_env['TMPDIR'],
        'DG_USE_NVIDIA_TOOLS': child_env['DG_USE_NVIDIA_TOOLS'],
        'DG_USE_NVIDIA_TOOLS_source': 'inherited' if dg_tools_was_set else 'injected',
        'CUDA_HOME': child_env.get('CUDA_HOME'),
        'CUDA_PATH': child_env.get('CUDA_PATH'),
        'CUDA_VISIBLE_DEVICES': child_env.get('CUDA_VISIBLE_DEVICES'),
        'CUDA_DEVICE_ORDER': child_env.get('CUDA_DEVICE_ORDER'),
    }
    target_indices = sorted({args.gpu_metrics_device} |
                            set(visible['harness_physical_indices']))
    gpu_check = inspect_gpus(target_indices, args.mem_threshold_mib, args.util_threshold_pct)
    gpu_check['cuda_visible_devices'] = visible
    estimate = estimate_sizes(args.gpu_metrics_frequency, metric_count, args.estimate_seconds)

    print(f'[plan] nsys           {nsys}')
    print(f'[plan] nsys version   {version}')
    print(f'[plan] root           {root}')
    print(f'[plan] out-dir (cwd)  {out_dir}')
    print(f'[plan] TMPDIR         {nsys_tmpdir}（nsys 临时 qdstrm 重定向到此，杜绝写 /tmp）')
    print(f'[plan] DG_USE_NVIDIA_TOOLS={child_env["DG_USE_NVIDIA_TOOLS"]} '
          f'({env_overrides["DG_USE_NVIDIA_TOOLS_source"]})：让 deep_gemm bench_kineto 短路，'
          f'不与 nsys 抢 CUPTI subscriber（否则 harness 崩、NCCL watchdog 写 GB 级 core）')
    print(f'[plan] harness TMPDIR  {harness_tmpdir}（本地 tmpfs 短路径，供 symm_mem UDS；'
          f'{"自动生成" if harness_tmpdir_auto else "用户指定"}，与 nsys 的 .work TMPDIR 分离）')
    print(f'[plan] metrics set    {metrics["value"]}')
    print(f'[plan] metrics source {metrics["source"]}')
    print(f'[plan] metric count   {metric_count}（用于体积预估）')
    print(f'[plan] trace          {args.trace}')
    print(f'[plan] capture range  {args.capture_range} end={args.capture_range_end}')
    print(f'[plan] frequency      {args.gpu_metrics_frequency} Hz '
          f'(期望采样间隔 {1e9 / args.gpu_metrics_frequency:.0f} ns)')
    print(f'[plan] gpu metrics on 物理 GPU {args.gpu_metrics_device}（硬约束：只采一张卡）')
    print(f'[plan] gpu check on   {target_indices}')
    print(f'[plan] CUDA_VISIBLE_DEVICES={visible["cuda_visible_devices"]!r} '
          f'selector={visible["selector"]} '
          f'harness 物理卡={visible["harness_physical_indices"]}')
    if visible['mappable'] and visible['harness_physical_indices'] and \
            args.gpu_metrics_device not in visible['harness_physical_indices']:
        print(f'[plan] WARNING --gpu-metrics-device {args.gpu_metrics_device} '
              f'不在 harness 使用的物理卡 {visible["harness_physical_indices"]} 内，'
              f'会采到没有 workload 的卡')
    print(f'[plan] payload        {"自定义 --payload" if args.payload else "harness"}')
    print('[plan] nsys 命令:')
    print('       ' + ' '.join(shlex.quote(part) for part in nsys_cmd))
    print('[plan] export 命令:')
    print('       ' + ' '.join(shlex.quote(part) for part in export_cmd))
    print(f'[plan] 预估产物（假设采集 {estimate["assumed_capture_seconds"]:.0f} s）: '
          f'metric 样本 {estimate["estimated_metric_samples"]:.3g}，'
          f'行数 {estimate["estimated_metric_rows"]:.3g}，'
          f'nsys-rep ~{estimate["estimated_nsys_rep_bytes"] / 1e9:.2f} GB，'
          f'sqlite ~{estimate["estimated_sqlite_bytes"] / 1e9:.2f} GB')
    print(f'[gpu] nvidia_smi_available={gpu_check["nvidia_smi_available"]} '
          f'busy={gpu_check["busy"]}')
    for entry in gpu_check['gpus']:
        print(f'[gpu]   GPU {entry["index"]} {entry["name"]} '
              f'mem_used={entry["memory_used_mib"]:.0f} MiB '
              f'util={entry["utilization_pct"]:.0f}% '
              f'over_mem={entry["over_memory"]} over_util={entry["over_utilization"]}')
    for reason in gpu_check['reasons']:
        print(f'[gpu]   reason: {reason}')

    if gpu_check['busy'] and not args.force:
        print('[gpu] ERROR 目标 GPU 被占用或状态未知（绝不 kill 任何进程）。'
              '等空闲后重跑，或显式加 --force。')
        return 2
    if gpu_check['busy']:
        print('[gpu] WARNING 目标 GPU 有占用，--force 已指定，继续执行')

    if args.dry_run:
        print('[dry-run] 仅打印计划，未创建目录、未启动 nsys、未写任何文件')
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'harness').mkdir(parents=True, exist_ok=True)
    nsys_tmpdir.mkdir(parents=True, exist_ok=True)
    if harness_tmpdir_auto and harness_tmpdir.exists():
        shutil.rmtree(harness_tmpdir, ignore_errors=True)
    harness_tmpdir.mkdir(parents=True, exist_ok=True)
    before_bytes = dir_bytes(out_dir)
    started = time.time()
    print(f'[run] cd {out_dir} 后启动 nsys（core dump / nsys 临时文件都落在这里，不污染仓库根）')
    capture_log = out_dir / 'capture_nsys.log'
    code, log_text = run_capture(nsys_cmd, out_dir, capture_log, args.timeout_sec, child_env)
    capture_seconds = time.time() - started
    print(f'[run] nsys profile 退出码 {code}，耗时 {capture_seconds:.1f} s')

    export_log_text = ''
    export_code: Optional[int] = None
    if not args.skip_export and rep_path.is_file():
        export_log = out_dir / 'capture_export.log'
        export_started = time.time()
        export_code, export_log_text = run_capture(export_cmd, out_dir, export_log,
                                                   args.timeout_sec, child_env)
        print(f'[run] nsys export 退出码 {export_code}，'
              f'耗时 {time.time() - export_started:.1f} s')
    elif not rep_path.is_file():
        print(f'[run] 缺少 {rep_path}，跳过 sqlite 导出')

    artifacts = []
    for path in (rep_path, sqlite_path, capture_log, out_dir / 'capture_export.log'):
        if path.is_file():
            artifacts.append({'path': str(path), 'bytes': path.stat().st_size,
                              'sha256': sha256_of(path)})
    capture_contract = {
        'capture_range': args.capture_range,
        'capture_range_end': args.capture_range_end,
        'flush_on_cudaprofiler_stop': nsys_controls['flush_on_cudaprofiler_stop'],
        'trace_fork_before_exec': nsys_controls['trace_fork_before_exec'],
        'frequency_hz': args.gpu_metrics_frequency,
        'metrics_set': metrics['value'],
        'metrics_config_path': metrics['config_path'],
        'metrics_config_preflight': metrics_preflight,
        'cuda_home': cuda_preflight['resolved'],
        'cuda_path': cuda_preflight['resolved'],
    }
    evidence = verify_evidence(sqlite_path, log_text + '\n' + export_log_text,
                               args.gpu_metrics_frequency, args.kernel_name_contains,
                               expected_devices=visible['harness_physical_indices'],
                               expected_process_count=args.num_processes,
                               expected_launch_count=args.expected_launch_count,
                               capture_contract=capture_contract)
    if code != 0:
        evidence['failures'].append(f'nsys profile 退出码 {code}')
    if export_code not in (None, 0):
        evidence['failures'].append(f'nsys export 退出码 {export_code}')
    evidence['verdict'] = 'FAIL' if evidence['failures'] else 'PASS'

    after_bytes = dir_bytes(out_dir)
    # For the v9 EP2 harness the Python payload is a single process which
    # launches and synchronizes both ranks.  nsys propagates that process'
    # exit status as its profile status, so this is real harness evidence,
    # not an inferred success from the nsys export.  Keep this deliberately
    # narrow: arbitrary payloads still need their own run_manifest.
    payload_is_known_two_gpu_harness = any(
        Path(str(part)).name == 'bench_full_ep2.py' for part in payload)
    harness_exit = None
    if payload_is_known_two_gpu_harness:
        harness_exit = {
            'code': code,
            'all_ranks_ok': code == 0,
            'evidence_source': 'nsys_profile_propagated_payload_exit',
            'payload_launch_model': 'single_process_two_gpu_synchronized',
        }

    manifest = {
        'schema': SCHEMA,
        'generated_at_utc': utc_iso(time.time()),
        'nsys': {'path': nsys, 'version': version},
        'root': str(root),
        'out_dir': str(out_dir),
        'cwd_for_capture': str(out_dir),
        'tmpdir': str(nsys_tmpdir),
        'child_env_overrides': env_overrides,
        'harness_tmpdir': str(harness_tmpdir),
        'harness_tmpdir_auto': harness_tmpdir_auto,
        'metrics_set': metrics['value'],
        'metrics_set_source': metrics['source'],
        'metrics_config_path': metrics['config_path'],
        'metrics_config_metric_count': metric_count,
        'metrics_config_preflight': metrics_preflight,
        'cuda_preflight': cuda_preflight,
        'cuda_home': cuda_preflight['resolved'],
        'cuda_path': cuda_preflight['resolved'],
        'frequency_hz': args.gpu_metrics_frequency,
        'expected_interval_ns': 1e9 / args.gpu_metrics_frequency,
        'frequency_contract': {
            'requested_hz': args.gpu_metrics_frequency,
            'preferred_hz': DEFAULT_FREQUENCY_HZ,
            'expected_interval_ns': 1e9 / args.gpu_metrics_frequency,
        },
        'gpu_metrics_device': args.gpu_metrics_device,
        'gpu_metrics_device_constraint': '本次只采一个代表 GPU；200 kHz 是交付硬约束，稳定性由实测采样间隔判断',
        'cuda_visible_devices': visible,
        'trace': args.trace,
        'capture_range': args.capture_range,
        'capture_range_end': args.capture_range_end,
        'flush_on_cudaprofiler_stop': nsys_controls['flush_on_cudaprofiler_stop'],
        'trace_fork_before_exec': nsys_controls['trace_fork_before_exec'],
        'capture_contract': capture_contract,
        'metrics_config': {
            'requested': args.metrics_set,
            'resolved': metrics['value'],
            'source': metrics['source'],
            'path': metrics['config_path'],
            'sha256': (sha256_of(Path(metrics['config_path']))
                       if metrics['config_path'] and Path(metrics['config_path']).is_file()
                       else None),
            'metric_count': metric_count,
        },
        'num_processes': args.num_processes,
        'expected_launch_count': args.expected_launch_count,
        'kernel_name_contains': args.kernel_name_contains,
        'argv': list(argv),
        'nsys_command': nsys_cmd,
        'nsys_command_line': ' '.join(shlex.quote(part) for part in nsys_cmd),
        'export_command': export_cmd,
        'export_command_line': ' '.join(shlex.quote(part) for part in export_cmd),
        'payload_command': payload,
        'payload_command_line': ' '.join(shlex.quote(part) for part in payload),
        'exit_codes': {'nsys_profile': code, 'nsys_export': export_code},
        **({'harness_exit': harness_exit} if harness_exit is not None else {}),
        'capture_seconds': capture_seconds,
        'gpu_check': gpu_check,
        'size_estimate': estimate,
        'artifacts': artifacts,
        'out_dir_bytes': {'before': before_bytes, 'after': after_bytes},
        'evidence': evidence,
        'verdict': evidence['verdict'],
    }
    manifest_path = out_dir / 'capture_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'[out] manifest {manifest_path}')
    for item in artifacts:
        print(f'[out]   {item["path"]} {item["bytes"]} B sha256={item["sha256"][:16]}...')
    interval = evidence.get('sampling_interval_ns', {})
    if interval.get('available'):
        print(f'[evidence] 采样间隔 median={interval["median"]} ns '
              f'p95={interval["p95"]} ns min={interval["min"]} max={interval["max"]} '
              f'样本数={interval["count"]}（期望 {evidence["expected_interval_ns"]:.0f} ns）')
    else:
        print(f'[evidence] 采样间隔不可用: {interval.get("reason")}')
    for warning in evidence['warnings']:
        print(f'[evidence] WARN {warning}')
    for failure in evidence['failures']:
        print(f'[evidence] FAIL {failure}')
    print(f'[evidence] verdict={evidence["verdict"]}')
    if harness_tmpdir_auto:
        shutil.rmtree(harness_tmpdir, ignore_errors=True)
        print(f'[cleanup] 已删除 harness 临时 tmpfs 目录 {harness_tmpdir}')
    return 0 if evidence['verdict'] == 'PASS' else 4


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
