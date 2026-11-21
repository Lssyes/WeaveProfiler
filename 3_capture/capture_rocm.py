#!/usr/bin/env python3
"""ROCm 采集包装器（AMD MI308X / gfx942）——镜像 capture_nsys.py 的对外接口。

对外接口与 capture_nsys.py 一一对应：run_capture / build_rocprof_cmd / scan_log /
inspect_gpus / resolve_visible_devices / verify_evidence / validate_existing_capture /
main，产物为 capture_manifest.json（schema='rocprof-hw-capture/v1'）。底层把
nsys profile/export 换成 rocprof-sys 或 rocprofv3：

    rocprof-sys-run --trace --use-rocm [--rocm-domains ... --rocm-events ...] \
        --output <outdir> capture -- <app>

timeline-only 默认优先 rocprof-sys（兼容探测 rocprof-sys / rocprof-sys-run /
rocprof-sys-record），实际产物位于其 timestamp/PID 子目录，通常是
*-perfetto-trace-*.proto 和 *-metadata-*.json。rocprof-sys 不可用时明确降级到：

    rocprofv3 --kernel-trace --hip-trace --memory-copy-trace \
        --output-format pftrace json -d <outdir> -o capture [--pmc ...] -- <app>

timeline 产物 = capture_results.pftrace（Perfetto 直接可开）；结构化产物 =
capture_results.json（buffer_records.kernel_dispatch / hip_api +
callback_records.counter_collection），替代 nsys 的 sqlite 导出。

三条关键纪律（硬编码在流程里，不是建议）：

1. persistent kernel（grid >= CU 数且 kernel 内自旋等待，如 mn_mega/mn_w8x 一族）
   禁止挂 PMC/counter：counter 采集要抢占/串行化 dispatch，与自旋等待互踩，
   本机有实测 hang + GPU reset 风暴前科。检测方法 = 应用侧契约声明
   `platform.persistent`（经 --contract 传入 CONTRACT.lock.json）：
   - persistent=true 且请求 hw bundle（--pmc）→ fail-closed 拒绝，提示改走
     相位戳法（s_memrealtime/rdclock 打点 + prog[] 槽位），参考
     /workspace/weave_new1/weave_v1/rebuttal/amd_mi308x/megamoe/mn_v2/notes/instrumentation_map.md
   - 请求 hw 但契约缺失或未显式声明 persistent=false → 同样 fail-closed
     （无法证明非 persistent 就不许挂 counter）。
   timeline-only（不带 --pmc）对 persistent kernel 是安全的：kernel-trace/
   hip-trace 只包 dispatch 边界，不进 CU。

2. GPU 空闲检查用 `amd-smi process`（不用 nvidia-smi 语义）：本机 8 卡共有
   一个 host 侧系统进程（同一 pid 出现在全部 8 卡的 process_list 里，本容器
   namespace 里 ps 查不到）——把它识别为"全卡交集 pid"并豁免，其余目标卡上
   的 pid 一律视为 foreign（绝不 kill，报 busy 等人走）。可见性 env 用
   HIP_VISIBLE_DEVICES（不是 CUDA_VISIBLE_DEVICES）。

3. nsys 的 --gpu-metrics 200kHz 时间序列采样在 AMD 无对应物。hw bundle 降级为
   两件事，且采样率在 capture_contract 里如实标注，不许虚标：
   (a) per-dispatch PMC（--pmc 计数器按 kernel dispatch 出数，粒度=一次
       dispatch 一组计数，不是周期采样；counter 集必须单 pass 装下，否则
       rocprofv3 直接拒跑——这是它的硬性行为）；
   (b) amd-smi 粗采样（采集期间后台按 ~0.25s 间隔轮询 `amd-smi metric --json`，
       落 amdsmi_samples.jsonl；实测间隔中位数写进契约，典型 2-5 Hz 量级，
       与 200 kHz 差 5 个数量级，做对齐时禁止当等值物用）。

rocprof-sys 的 --trace 也只记录事件边界，不是周期 PM 采样。请求 --pmc 时，
本包装器明确路由到 rocprofv3 的结构化 counter_collection，并在 manifest 的
collector.fallback 与 collector.capabilities 中记录原因和能力边界。
显式传入 --rocm-events 时，rocprof-sys 在同一次 Perfetto trace 里附带原始
ROCm per-dispatch counters；它们按 dispatch 输出，不是周期 PM 时间序列。
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rocprof_sys_adapter import (
    COLLECTOR_CHOICES,
    DEFAULT_ROCM_DOMAINS,
    DEFAULT_ROCM_EVENTS,
    ROCPROF_SYS_FAMILY,
    ROCPROFV3_FAMILY,
    build_rocprof_sys_cmd,
    command_line,
    discover_rocprof_sys_artifacts,
    inspect_rocprof_sys_artifacts,
    select_collector,
)

try:
    import resource
except ImportError:
    resource = None

SCHEMA = 'rocprof-hw-capture/v1'
# 默认 PMC 集（已对照本机 rocprofv3 counters 目录逐名核实存在，gfx942 单 XCC 视角
# ×4 副本属正常）。语义：每次 kernel dispatch 一组计数（per-dispatch，非周期采样）。
# 装不进单 pass 时 rocprofv3 会拒跑，届时裁掉尾部 TCC/FETCH/WRITE 四项再试。
DEFAULT_PMC: Tuple[str, ...] = (
    'GRBM_GUI_ACTIVE',      # GPU 有活在干的周期数
    'SQ_WAVES',             # 发射的 wave 数
    'SQ_BUSY_CYCLES',       # SQ 忙周期
    'SQ_INSTS_VALU',        # VALU 指令（含 FMA 类）
    'SQ_INSTS_MFMA',        # MFMA 矩阵指令（GEMM 主干）
    'SQ_INSTS_SMEM',        # 共享内存指令（LDS 压力）
    'SQ_INSTS_SALU',        # 标量指令
    'SQ_WAIT_ANY',          # wave 等待周期（stall 总量代理）
    'TCC_HIT',              # L2 命中
    'TCC_MISS',             # L2 未命中
    'FETCH_SIZE',           # 显存读量
    'WRITE_SIZE',           # 显存写量
)
DEFAULT_AMDSMI_INTERVAL_SEC = 0.25
INSTRUMENTATION_MAP_REF = (
    '/workspace/weave_new1/weave_v1/rebuttal/amd_mi308x/megamoe/'
    'mn_v2/notes/instrumentation_map.md')
LOG_PATTERNS = (
    ('overflow', re.compile(r'overflow', re.IGNORECASE)),
    ('dropped', re.compile(r'\bdrop(ped|ping|s)?\b', re.IGNORECASE)),
    ('missing', re.compile(r'\bmissing\b', re.IGNORECASE)),
    ('incomplete', re.compile(r'\bincomplete\b', re.IGNORECASE)),
    ('hsa_error', re.compile(r'HSA_STATUS_ERROR', re.IGNORECASE)),
    ('rocm_error', re.compile(r'\berror\b', re.IGNORECASE)),
    ('warning', re.compile(r'\bwarn(ing)?\b', re.IGNORECASE)),
    ('permission', re.compile(r'permission|not permitted|admin', re.IGNORECASE)),
)
# rocprofv3 的 glog 用 E 前缀打良性行（"Opened result file"），不能当错误证据。
BENIGN_LOG_LINES = re.compile(r'Opened result file|Configured tracing session|'
                              r'Tracing session .* ended', re.IGNORECASE)
BLOCKING_KINDS = ('overflow', 'dropped', 'hsa_error')


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


def run_capture(cmd: List[str], cwd: Path, log_path: Path,
                timeout: Optional[float],
                env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """与 capture_nsys.run_capture 同语义：起子进程、逐行 tee 到日志与 stdout。"""
    chunks: List[str] = []
    with log_path.open('w', encoding='utf-8') as log:
        header = '$ ' + ' '.join(shlex.quote(part) for part in cmd) + '\n'
        log.write(header)
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env=env,
                                preexec_fn=_preexec_drop_core if resource is not None else None)
        deadline = None if timeout is None else time.time() + timeout
        assert proc.stdout is not None
        for line in proc.stdout:
            chunks.append(line)
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            if deadline is not None and time.time() > deadline:
                proc.kill()
                chunks.append('[capture] TIMEOUT，已终止 collector 子进程\n')
                break
        code = proc.wait()
    return code, ''.join(chunks)


def scan_log(text: str) -> Dict[str, Any]:
    """与 capture_nsys.scan_log 同接口；ROCm collector 日志统一扫描。"""
    findings: Dict[str, Any] = {kind: {'count': 0, 'lines': []} for kind, _ in LOG_PATTERNS}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('$ '):
            continue
        if BENIGN_LOG_LINES.search(stripped):
            continue
        for kind, pattern in LOG_PATTERNS:
            if kind in ('overflow', 'dropped') and re.search(
                    rf'["\']?{kind}["\']?\s*[:=]\s*(?:0|false)\b',
                    stripped, re.IGNORECASE):
                continue
            if pattern.search(stripped):
                findings[kind]['count'] += 1
                if len(findings[kind]['lines']) < 12:
                    findings[kind]['lines'].append(stripped)
    return findings


# ---------------------------------------------------------------------------
# amd-smi 封装（纪律 2：空闲检查走 amd-smi process，全卡交集 pid = 系统进程豁免）
# ---------------------------------------------------------------------------

def _run_json(cmd: List[str]) -> Optional[Any]:
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def query_gpu_list(amdsmi: str) -> Optional[List[Dict[str, Any]]]:
    data = _run_json([amdsmi, 'list', '--json'])
    if not isinstance(data, list):
        return None
    rows = []
    for entry in data:
        if not isinstance(entry, dict) or 'gpu' not in entry:
            continue
        rows.append({'index': int(entry['gpu']),
                     'uuid': entry.get('uuid'),
                     'bdf': entry.get('bdf')})
    return rows


def _metric_value(node: Any) -> Optional[float]:
    """amd-smi metric 的叶子是 {'value': X, 'unit': ...} 或裸 'N/A'。"""
    if isinstance(node, dict) and 'value' in node:
        node = node['value']
    if isinstance(node, (int, float)):
        return float(node)
    return None


def query_gpu_metric(amdsmi: str, index: int) -> Optional[Dict[str, Any]]:
    data = _run_json([amdsmi, 'metric', '-g', str(index), '--json'])
    if not isinstance(data, list) or not data:
        return None
    entry = data[0]
    mem = entry.get('mem_usage') or {}
    usage = entry.get('usage')
    gfx_util = None
    if isinstance(usage, dict):
        gfx_util = _metric_value(usage.get('gfx_activity')) or \
            _metric_value(usage.get('gfx'))
    return {
        'index': int(entry.get('gpu', index)),
        'memory_used_mib': _metric_value(mem.get('used_vram')),
        'memory_total_mib': _metric_value(mem.get('total_vram')),
        'utilization_pct': gfx_util,   # 空闲卡上 amd-smi 常报 N/A → None（如实记录）
    }


def query_processes(amdsmi: str) -> Optional[List[Dict[str, Any]]]:
    data = _run_json([amdsmi, 'process', '--json'])
    if not isinstance(data, list):
        return None
    rows = []
    for entry in data:
        if not isinstance(entry, dict) or 'gpu' not in entry:
            continue
        procs = []
        for item in entry.get('process_list') or []:
            info = (item or {}).get('process_info') or {}
            pid = info.get('pid')
            if pid is None:
                continue
            vram = info.get('memory_usage') or {}
            procs.append({'pid': int(pid),
                          'name': info.get('name'),
                          'vram_bytes': _metric_value(vram.get('vram_mem'))})
        rows.append({'index': int(entry['gpu']), 'processes': procs})
    return rows


def shared_system_pids(process_rows: List[Dict[str, Any]]) -> List[int]:
    """全部 GPU process_list 的 pid 交集 = host 侧系统进程（8 卡共有），豁免。"""
    sets = []
    for row in process_rows:
        sets.append({proc['pid'] for proc in row['processes']})
    if not sets:
        return []
    common = set.intersection(*sets) if len(sets) > 1 else set()
    return sorted(common)


def inspect_gpus(amdsmi: str, target_indices: List[int], mem_threshold_mib: float,
                 util_threshold_pct: float) -> Dict[str, Any]:
    """与 capture_nsys.inspect_gpus 同报告形状；底层 amd-smi list/metric/process。"""
    gpu_list = query_gpu_list(amdsmi)
    proc_rows = query_processes(amdsmi)
    report: Dict[str, Any] = {
        'amd_smi_available': gpu_list is not None and proc_rows is not None,
        'target_indices': target_indices,
        'memory_threshold_mib': mem_threshold_mib,
        'utilization_threshold_pct': util_threshold_pct,
        'gpus': [],
        'foreign_processes': [],
        'shared_system_pids': [],
        'busy': False,
        'reasons': [],
    }
    if gpu_list is None or proc_rows is None:
        report['busy'] = True
        report['reasons'].append('amd-smi 不可用，无法确认 GPU 空闲')
        return report
    system_pids = set(shared_system_pids(proc_rows))
    report['shared_system_pids'] = sorted(system_pids)
    by_index = {gpu['index']: gpu for gpu in gpu_list}
    my_pids = {os.getpid(), os.getppid()}
    for index in target_indices:
        if index not in by_index:
            report['busy'] = True
            report['reasons'].append(f'GPU {index} 不存在')
            continue
        metric = query_gpu_metric(amdsmi, index) or {'index': index}
        entry = dict(by_index[index])
        entry.update(metric)
        used = metric.get('memory_used_mib')
        util = metric.get('utilization_pct')
        entry['over_memory'] = used is not None and used > mem_threshold_mib
        entry['over_utilization'] = util is not None and util > util_threshold_pct
        report['gpus'].append(entry)
        if entry['over_memory']:
            report['busy'] = True
            report['reasons'].append(
                f'GPU {index} 已用显存 {used:.0f} MiB > 阈值 {mem_threshold_mib:.0f} MiB')
        if entry['over_utilization']:
            report['busy'] = True
            report['reasons'].append(
                f'GPU {index} gfx 利用率 {util:.0f}% > 阈值 {util_threshold_pct:.0f}%')
    procs_by_gpu = {row['index']: row['processes'] for row in proc_rows}
    for index in target_indices:
        for proc in procs_by_gpu.get(index, []):
            if proc['pid'] in system_pids or proc['pid'] in my_pids:
                continue
            report['foreign_processes'].append(dict(proc, index=index))
            report['busy'] = True
            report['reasons'].append(
                f'GPU {index} 上存在其他进程 pid={proc["pid"]} '
                f'name={proc["name"]}（不会 kill，请等待空闲）')
    return report


def resolve_visible_devices(hip_visible: Optional[str],
                            num_processes: int) -> Dict[str, Any]:
    """HIP_VISIBLE_DEVICES 解析，语义同 capture_nsys 的 CUDA 版。"""
    if hip_visible is None or hip_visible.strip() == '':
        return {'hip_visible_devices': None, 'selector': 'physical-index',
                'harness_physical_indices': list(range(max(num_processes, 0))),
                'tokens': None, 'mappable': True}
    tokens = [item.strip() for item in hip_visible.split(',') if item.strip()]
    used = tokens
    if used and all(item.lstrip('-').isdigit() for item in used):
        return {'hip_visible_devices': hip_visible, 'selector': 'hip-visible-index',
                'harness_physical_indices': [int(item) for item in used],
                'tokens': tokens, 'mappable': True,
                'enough_devices': len(tokens) >= num_processes}
    # ROCm 侧 HIP_VISIBLE_DEVICES 只接受索引；非数字 token 视为不可映射。
    return {'hip_visible_devices': hip_visible, 'selector': 'unparsed',
            'harness_physical_indices': [], 'tokens': tokens, 'mappable': False,
            'enough_devices': len(tokens) >= num_processes}


# ---------------------------------------------------------------------------
# 纪律 1：persistent kernel fail-closed
# ---------------------------------------------------------------------------

def load_app_contract(path: Optional[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {'path': path, 'loaded': False, 'persistent': None,
                              'hard_failures': []}
    if not path:
        return result
    p = Path(path).expanduser().resolve()
    result['path'] = str(p)
    if not p.is_file():
        result['hard_failures'].append(f'契约文件不存在: {p}')
        return result
    try:
        contract = json.loads(p.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        result['hard_failures'].append(f'契约解析失败: {p}: {exc}')
        return result
    if not isinstance(contract, dict):
        result['hard_failures'].append(f'契约顶层必须是 JSON object: {p}')
        return result
    platform = contract.get('platform')
    if isinstance(platform, dict) and 'persistent' in platform:
        result['persistent'] = bool(platform['persistent'])
    result['loaded'] = True
    result['contract_sha256'] = sha256_of(p)
    return result


def persistent_guard(contract_info: Dict[str, Any],
                     hw_requested: bool) -> List[str]:
    """persistent + hardware counters = 禁。返回 failure 列表（空 = 放行）。"""
    if not hw_requested:
        return []
    failures = []
    if contract_info['persistent'] is True:
        failures.append(
            '契约 platform.persistent=true：persistent kernel（grid>=CU 且 kernel 内'
            '自旋等待）禁止挂 PMC/counter（本机有 hang + reset 风暴前科）。'
            f'改走相位戳法（s_memrealtime 打点），参考 {INSTRUMENTATION_MAP_REF}。'
            'timeline-only（去掉 --pmc 和 --rocm-events）可以照常采。')
    elif contract_info['persistent'] is None:
        if not contract_info['loaded']:
            failures.append(
                '请求计数器采集（--pmc 或 rocprof-sys --rocm-events）但未提供有效 '
                '--contract（CONTRACT.lock.json）：'
                '无法证明被测 kernel 非 persistent，fail-closed 拒绝。'
                '请在契约 platform.persistent 显式声明 false 后重试，或去掉计数器选项。')
        else:
            failures.append(
                '契约缺少 platform.persistent 字段：无法证明被测 kernel 非 '
                'persistent，fail-closed 拒绝挂 PMC。请显式声明 '
                'platform.persistent=false，或去掉计数器选项。')
    return failures


# ---------------------------------------------------------------------------
# rocprofv3 命令构建 + flag 自检（不跑 GPU）
# ---------------------------------------------------------------------------

# 本模块实际用到的 rocprofv3 flag 全集；--selftest 用 --help 文本逐一核对存在性。
ROCPROF_FLAGS_USED: Tuple[str, ...] = (
    '--kernel-trace', '--hip-trace', '--memory-copy-trace',
    '--output-format', '-d', '--output-directory', '-o', '--output-file',
    '--pmc', '--log-level', '--stats', '--kernel-include-regex',
    '--perfetto-buffer-size', '--help',
)


def selftest_flags(rocprof: str) -> Dict[str, Any]:
    """解析 rocprofv3 --help，确认 ROCPROF_FLAGS_USED 每个 flag 都存在（不跑 GPU）。"""
    try:
        out = subprocess.run([rocprof, '--help'], check=True, capture_output=True,
                             text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return {'available': False, 'hard_failures': [f'rocprofv3 --help 失败: {exc}']}
    missing = [flag for flag in ROCPROF_FLAGS_USED if flag not in out]
    result: Dict[str, Any] = {
        'available': True,
        'rocprof': rocprof,
        'flags_checked': list(ROCPROF_FLAGS_USED),
        'flags_missing': missing,
        'hard_failures': ([f'rocprofv3 --help 缺少本模块用到的 flag: {missing}']
                          if missing else []),
    }
    # --version 单独探测（部分版本只认 --help）。
    try:
        ver = subprocess.run([rocprof, '--version'], check=True, capture_output=True,
                             text=True).stdout.strip()
        result['version'] = ver
    except (OSError, subprocess.CalledProcessError):
        result['version'] = None
        result['version_note'] = '--version 不可用（不阻断；版本字段记 None）'
    return result


def build_rocprof_cmd(args: argparse.Namespace, rocprof: str, out_dir: Path,
                      payload: List[str]) -> List[str]:
    """rocprofv3 --kernel-trace --hip-trace --output-format pftrace json -d <> -- <app>。"""
    cmd = [rocprof,
           '--kernel-trace',
           '--hip-trace',
           '--memory-copy-trace',
           '--output-format', 'pftrace', 'json',
           '-d', str(out_dir),
           '-o', 'capture',
           '--log-level', args.log_level,
           '--perfetto-buffer-size', str(args.perfetto_buffer_kb),
           '--stats', 'true']
    if args.pmc:
        cmd.append('--pmc')
        cmd.extend(args.pmc)
        if args.pmc_filter_target and args.kernel_name_contains:
            # --kernel-include-regex 只作用于 counter-collection，不动 trace。
            cmd.extend(['--kernel-include-regex', args.kernel_name_contains])
    for extra in args.rocprof_extra or []:
        cmd.extend(shlex.split(extra))
    cmd.append('--')
    cmd.extend(payload)
    return cmd


def resolve_payload_paths(payload: List[str], root: Path) -> Tuple[List[str], Dict[str, Any]]:
    """Resolve relative payload paths against the project root before ``cwd`` changes.

    The collector is launched with ``cwd=<out_dir>`` so that its timestamped
    output directories stay inside the run directory.  A command such as
    ``./build/app data/input.bin`` is nevertheless normally written relative to
    the project root.  Resolve those paths first and record the exact mapping in
    the manifest.

    Bare executable names are left to PATH unless a matching executable exists
    directly under ``root``.  Path-like arguments are resolved only when they
    exist under ``root``; this avoids rewriting flags, numbers and opaque string
    arguments into accidental paths.
    """
    if not payload:
        return [], {'original': [], 'resolved': [], 'rewrites': []}

    root = root.expanduser().resolve()
    resolved = list(payload)
    rewrites: List[Dict[str, str]] = []

    executable = payload[0]
    if executable and not os.path.isabs(executable):
        candidate = root / executable
        use_root_candidate = False
        if os.sep in executable or (os.altsep and os.altsep in executable):
            use_root_candidate = candidate.exists()
        else:
            use_root_candidate = candidate.is_file() and os.access(str(candidate), os.X_OK)
        if use_root_candidate:
            resolved[0] = str(candidate.resolve())
            rewrites.append({'index': '0', 'from': executable, 'to': resolved[0],
                             'role': 'executable'})

    for index, token in enumerate(payload[1:], start=1):
        if not token or token.startswith('-') or os.path.isabs(token):
            continue
        # ``KEY=VALUE`` is normally an environment assignment or an option
        # value, never a path token for the payload itself.
        if '=' in token or '://' in token:
            continue
        candidate = root / token
        if candidate.exists():
            replacement = str(candidate.resolve())
            if replacement != token:
                resolved[index] = replacement
                rewrites.append({'index': str(index), 'from': token,
                                 'to': replacement, 'role': 'argument'})

    return resolved, {'original': list(payload), 'resolved': list(resolved),
                      'rewrites': rewrites}


# ---------------------------------------------------------------------------
# amd-smi 粗采样器（纪律 3b：hw bundle 的第二半，如实记录实测频率）
# ---------------------------------------------------------------------------

class AmdSmiSampler:
    """采集期间后台轮询 `amd-smi metric -g <dev> --json`，写 JSONL。

    每次调用是一次 amd-smi 子进程（本机实测单次 ~50-250 ms），所以有效频率
    2-5 Hz 量级；实测间隔中位数由 verify_evidence 统计并写进契约。
    """

    def __init__(self, amdsmi: str, gpu_index: int, interval: float,
                 out_path: Path) -> None:
        self.amdsmi = amdsmi
        self.gpu_index = gpu_index
        self.interval = interval
        self.out_path = out_path
        self.samples = 0
        self.errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        with self.out_path.open('w', encoding='utf-8') as out:
            while not self._stop.is_set():
                t0 = time.time()
                metric = query_gpu_metric(self.amdsmi, self.gpu_index)
                row = {'t_mono': time.monotonic(), 't_utc': utc_iso(time.time()),
                       'gpu': self.gpu_index, 'metric': metric}
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                out.flush()
                if metric is None:
                    self.errors += 1
                else:
                    self.samples += 1
                elapsed = time.time() - t0
                self._stop.wait(max(self.interval - elapsed, 0.05))

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)


def sampler_stats(jsonl_path: Path, requested_interval: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        'path': str(jsonl_path), 'available': jsonl_path.is_file(),
        'requested_interval_s': requested_interval,
        'sample_count': 0, 'error_count': 0,
    }
    if not jsonl_path.is_file():
        result['reason'] = '采样文件未生成'
        return result
    timestamps: List[float] = []
    try:
        for line in jsonl_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('metric') is None:
                result['error_count'] += 1
                continue
            timestamps.append(float(row['t_mono']))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        result['reason'] = f'采样文件解析失败: {exc}'
        return result
    result['sample_count'] = len(timestamps)
    deltas = [right - left for left, right in zip(timestamps, timestamps[1:])]
    if deltas:
        ordered = sorted(deltas)
        result['interval_s'] = {
            'median': statistics.median(ordered),
            'min': ordered[0],
            'max': ordered[-1],
            'mean': sum(deltas) / len(deltas),
        }
        result['measured_hz'] = 1.0 / statistics.median(ordered)
    if timestamps:
        result['window_s'] = [timestamps[0], timestamps[-1]]
    return result


# ---------------------------------------------------------------------------
# 证据校验（rocprof json 替代 nsys sqlite）
# ---------------------------------------------------------------------------

def _rocprof_root(data: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    tool = data.get('rocprofiler-sdk-tool')
    if isinstance(tool, list) and tool and isinstance(tool[0], dict):
        return tool[0]
    return None


def _kernel_name_map(root: Dict[str, Any]) -> Dict[Any, str]:
    """kernel_symbols → kernel_id 到名字。字段名做防御性解析（版本间有漂移）。"""
    mapping: Dict[Any, str] = {}
    for entry in root.get('kernel_symbols') or []:
        if not isinstance(entry, dict):
            continue
        kid = None
        for key in ('kernel_id', 'id', 'handle'):
            if key in entry:
                kid = entry[key]
                break
        name = None
        for key in ('formatted_kernel_name', 'kernel_name', 'name', 'demangled_name'):
            if isinstance(entry.get(key), str):
                name = entry[key]
                break
        if kid is not None and name:
            mapping[kid] = name
    return mapping


def _record_kernel_id(record: Dict[str, Any]) -> Any:
    info = record.get('dispatch_info')
    if isinstance(info, dict):
        for key in ('kernel_id', 'id'):
            if key in info:
                return info[key]
    for key in ('kernel_id', 'id'):
        if key in record:
            return record[key]
    return None


def _record_agent(record: Dict[str, Any]) -> Any:
    for node in (record.get('dispatch_info'), record):
        if isinstance(node, dict) and isinstance(node.get('agent_id'), dict):
            return node['agent_id'].get('handle')
    return None


def _record_window(record: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    start = record.get('start_timestamp')
    end = record.get('end_timestamp')
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return int(start), int(end)
    return None


def kernel_probe_json(root: Dict[str, Any], needle: str) -> Dict[str, Any]:
    buffer_records = root.get('buffer_records') or {}
    dispatches = buffer_records.get('kernel_dispatch')
    result: Dict[str, Any] = {
        'available': isinstance(dispatches, list),
        'needle': needle,
        'hard_failures': [],
    }
    if not isinstance(dispatches, list):
        result['hard_failures'].append(
            'rocprof json 缺 buffer_records.kernel_dispatch（kernel-trace 未采到）')
        return result
    name_map = _kernel_name_map(root)
    result['total_dispatches'] = len(dispatches)
    result['kernel_symbols'] = len(name_map)
    launches: List[Dict[str, Any]] = []
    by_agent: Counter = Counter()
    unresolved = 0
    for record in dispatches:
        if not isinstance(record, dict):
            continue
        agent = _record_agent(record)
        by_agent[str(agent)] += 1
        window = _record_window(record)
        kid = _record_kernel_id(record)
        name = name_map.get(kid)
        if name is None:
            unresolved += 1
            continue
        if needle and needle not in name:
            continue
        launches.append({
            'name': name,
            'agent_handle': agent,
            'window_ns': list(window) if window else None,
            'complete': bool(window and window[1] > window[0]),
        })
    result['dispatch_count_by_agent_handle'] = dict(by_agent)
    result['unresolved_kernel_ids'] = unresolved
    result['match_rows'] = len(launches)
    result['launches'] = launches[:64]
    result['complete_launches'] = sum(1 for item in launches if item['complete'])
    if needle and not launches:
        result['hard_failures'].append(
            f'kernel_dispatch 里没有匹配 {needle!r} 的 launch（needle 写错或目标未跑）')
    if needle and any(not item['complete'] for item in launches):
        result['hard_failures'].append('存在 start/end 不完整的目标 kernel launch')
    result['valid'] = not result['hard_failures']
    return result


def counter_probe_json(root: Dict[str, Any], pmc: List[str]) -> Dict[str, Any]:
    callback_records = root.get('callback_records') or {}
    collection = callback_records.get('counter_collection')
    result: Dict[str, Any] = {
        'available': isinstance(collection, list),
        'requested_pmc': list(pmc),
        'semantics': 'per-dispatch（每次 kernel dispatch 一组计数；'
                     'AMD 无 nsys 200kHz 周期采样对应物）',
        'hard_failures': [],
    }
    if not isinstance(collection, list):
        result['hard_failures'].append(
            'rocprof json 缺 callback_records.counter_collection（--pmc 未采到）')
        return result
    result['record_count'] = len(collection)
    counter_names: Counter = Counter()
    for record in collection:
        if not isinstance(record, dict):
            continue
        for item in record.get('counters') or record.get('records') or []:
            if isinstance(item, dict):
                name = item.get('name') or item.get('counter_name')
                if name:
                    counter_names[str(name)] += 1
    result['samples_by_counter'] = dict(counter_names)
    if not collection:
        result['hard_failures'].append('counter_collection 为空：PMC 一组都没采到')
    else:
        zero = [name for name in pmc
                if counter_names and not any(name in key for key in counter_names)]
        if counter_names and zero:
            result['hard_failures'].append(f'请求的 PMC 无样本: {zero}')
    result['valid'] = not result['hard_failures']
    return result


def verify_evidence(json_path: Path, pftrace_path: Path, log_text: str,
                    kernel_needle: str, hw_requested: bool,
                    pmc: List[str],
                    sampler: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {
        'log_findings': scan_log(log_text),
        'pftrace': {'available': pftrace_path.is_file(), 'path': str(pftrace_path)},
        'json': {'available': json_path.is_file(), 'path': str(json_path)},
        'hw_requested': hw_requested,
        'failures': [],
        'warnings': [],
    }
    for kind in BLOCKING_KINDS:
        hit = evidence['log_findings'].get(kind, {})
        if hit.get('count'):
            evidence['failures'].append(
                f'collector 日志出现 {kind} 相关行 {hit["count"]} 条: {hit["lines"][:3]}')
    for kind in ('missing', 'incomplete', 'permission', 'rocm_error'):
        hit = evidence['log_findings'].get(kind, {})
        if hit.get('count'):
            evidence['warnings'].append(
                f'collector 日志出现 {kind} 相关行 {hit["count"]} 条: {hit["lines"][:3]}')
    if pftrace_path.is_file():
        size = pftrace_path.stat().st_size
        evidence['pftrace']['bytes'] = size
        if size <= 0:
            evidence['failures'].append('pftrace 为空')
    else:
        evidence['failures'].append('pftrace 未生成（Perfetto timeline 产物缺失）')
    if not json_path.is_file():
        evidence['failures'].append('rocprof json 未生成，无法做证据校验')
        evidence['verdict'] = 'FAIL'
        return evidence
    try:
        data = json.loads(json_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        evidence['failures'].append(f'rocprof json 解析失败: {exc}')
        evidence['verdict'] = 'FAIL'
        return evidence
    root = _rocprof_root(data)
    if root is None:
        evidence['failures'].append('rocprof json 缺 rocprofiler-sdk-tool 根')
        evidence['verdict'] = 'FAIL'
        return evidence
    evidence['kernel'] = kernel_probe_json(root, kernel_needle)
    evidence['failures'].extend(evidence['kernel'].get('hard_failures', []))
    if hw_requested:
        evidence['counters'] = counter_probe_json(root, pmc)
        evidence['failures'].extend(evidence['counters'].get('hard_failures', []))
    if sampler is not None:
        evidence['amdsmi_sampler'] = sampler
        if hw_requested and sampler.get('sample_count', 0) <= 0:
            evidence['warnings'].append('amd-smi 粗采样 0 个样本（hw bundle 少一条腿）')
    evidence['verdict'] = 'FAIL' if evidence['failures'] else 'PASS'
    return evidence


def verify_rocprof_sys_evidence(discovery: Dict[str, Any], log_text: str,
                                kernel_needle: str) -> Dict[str, Any]:
    """校验 rocprof-sys 的 Perfetto/metadata 产物，并复用统一日志扫描。"""
    artifact_check = inspect_rocprof_sys_artifacts(discovery, kernel_needle)
    evidence: Dict[str, Any] = {
        'collector': ROCPROF_SYS_FAMILY,
        'log_findings': scan_log(log_text),
        'artifact_validation': artifact_check,
        'failures': list(artifact_check.get('hard_failures') or []),
        'warnings': list(artifact_check.get('warnings') or []),
        'hw_requested': False,
    }
    trace = artifact_check.get('trace') or {}
    evidence['pftrace'] = {
        'available': bool(trace),
        'path': trace.get('path'),
        'bytes': trace.get('bytes'),
        'format': trace.get('format'),
    }
    evidence['kernel'] = artifact_check.get('kernel', {})
    for kind in BLOCKING_KINDS:
        hit = evidence['log_findings'].get(kind, {})
        if hit.get('count'):
            evidence['failures'].append(
                f'rocprof-sys 日志出现 {kind} 相关行 {hit["count"]} 条: '
                f'{hit["lines"][:3]}')
    for kind in ('missing', 'incomplete', 'permission', 'rocm_error'):
        hit = evidence['log_findings'].get(kind, {})
        if hit.get('count'):
            evidence['warnings'].append(
                f'rocprof-sys 日志出现 {kind} 相关行 {hit["count"]} 条: '
                f'{hit["lines"][:3]}')
    evidence['verdict'] = 'FAIL' if evidence['failures'] else 'PASS'
    return evidence


def estimate_sizes(dispatch_count: int, seconds: float,
                   hw_requested: bool) -> Dict[str, Any]:
    return {
        'assumed_capture_seconds': seconds,
        'assumed_dispatches': dispatch_count,
        'estimated_pftrace_bytes': dispatch_count * 200,
        'estimated_json_bytes': dispatch_count * 600,
        'estimated_amdsmi_jsonl_bytes': int(seconds / DEFAULT_AMDSMI_INTERVAL_SEC) * 4096,
        'note': '粗估：每 dispatch ~200B pftrace / ~600B json；PMC 每 dispatch '
                '每 counter 再加 ~32B。仅用于容量预警。'
                if hw_requested else '粗估，仅 trace 路径。仅用于容量预警。',
    }


def resolve_rocm_options(domains: Optional[List[str]],
                         events: Optional[List[str]]) -> Tuple[List[str], List[str]]:
    """Resolve opt-in rocprof-sys ROCm options without changing timeline defaults.

    ``None`` means the option was omitted, preserving timeline-only capture.
    Supplying ``--rocm-events`` with no values selects the MI308X default set;
    custom values retain their command-line order.  Domains default to
    ``kernel_dispatch`` whenever raw counters are requested.
    """
    def clean(values: Optional[List[str]]) -> List[str]:
        if values is None:
            return []
        return [str(value).strip() for value in values if str(value).strip()]

    domain_values = clean(domains)
    event_values = clean(events)
    if events is None:
        return domain_values, []
    if not event_values:
        event_values = list(DEFAULT_ROCM_EVENTS)
    if not domain_values:
        domain_values = list(DEFAULT_ROCM_DOMAINS)
    return domain_values, event_values


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='ROCm 采集包装器（AMD MI308X/gfx942）：timeline 优先 '
                    'rocprof-sys，PMC 使用 rocprofv3；镜像 capture_nsys.py 接口')
    parser.add_argument('--root', required=False, default=None)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--payload', nargs=argparse.REMAINDER, default=None,
                        help='被采集程序的原样命令，必须放在最后')
    parser.add_argument('--num-processes', type=int, default=1)
    parser.add_argument('--hip-visible-devices', default=None,
                        help='传给被采集进程的 HIP_VISIBLE_DEVICES；默认沿用当前环境')
    parser.add_argument('--contract', default=None,
                        help='应用侧 CONTRACT.lock.json；hw bundle 必须提供且 '
                             'platform.persistent 显式为 false（纪律 1 fail-closed）')
    parser.add_argument('--pmc', nargs='+', default=None,
                        help='gfx942 PMC 计数器列表；不给则 timeline-only（默认对 '
                             'persistent kernel 唯一安全的路径）。'
                             f'传空字符串触发默认集 {len(DEFAULT_PMC)} 项')
    parser.add_argument('--pmc-defaults', action='store_true',
                        help='用内置默认 PMC 集（GRBM/SQ/TCC/FETCH/WRITE 12 项）')
    parser.add_argument('--pmc-filter-target', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='PMC 加 --kernel-include-regex 限定目标 kernel（默认开）')
    parser.add_argument('--kernel-name-contains', default='',
                        help='目标 kernel 名字 needle；空 = 跳过目标 kernel 校验'
                             '（只验 trace 完整性）')
    parser.add_argument('--amdsmi-sample-interval', type=float,
                        default=DEFAULT_AMDSMI_INTERVAL_SEC,
                        help='amd-smi 粗采样目标间隔秒（实测频率如实记入契约）')
    parser.add_argument('--no-amdsmi-sampler', action='store_true')
    parser.add_argument('--mem-threshold-mib', type=float, default=1024.0)
    parser.add_argument('--util-threshold-pct', type=float, default=10.0)
    parser.add_argument('--estimate-seconds', type=float, default=60.0)
    parser.add_argument('--estimate-dispatches', type=int, default=10000)
    parser.add_argument('--timeout-sec', type=float, default=3600.0)
    parser.add_argument('--perfetto-buffer-kb', type=int, default=1048576)
    parser.add_argument('--log-level', default='warning',
                        choices=('fatal', 'error', 'warning', 'info', 'trace', 'env'))
    parser.add_argument('--rocprof-extra', action='append', default=None,
                        help='追加给 rocprofv3 的原样参数串（可多次）')
    parser.add_argument('--rocm-domains', nargs='*', default=None,
                        metavar='DOMAIN',
                        help='rocprof-sys ROCm domains；省略此选项则不启用 ROCm '
                             'counter。传 --rocm-domains 不带值使用默认 '
                             f'{list(DEFAULT_ROCM_DOMAINS)}')
    parser.add_argument('--rocm-events', nargs='*', default=None,
                        metavar='COUNTER',
                        help='rocprof-sys 原始 ROCm counters；省略此选项保持 '
                             'timeline-only。传 --rocm-events 不带值使用 MI308X '
                             f'gfx942 默认六项 {list(DEFAULT_ROCM_EVENTS)}')
    parser.add_argument('--collector', choices=COLLECTOR_CHOICES, default='auto',
                        help='auto=timeline 优先 rocprof-sys；--pmc 明确路由 '
                             'rocprofv3。也可显式指定 rocprof-sys/rocprofv3')
    parser.add_argument('--rocprof-sys', default=None,
                        help='rocprof-sys 可执行文件路径；不指定时依次探测 '
                             'rocprof-sys / rocprof-sys-run / rocprof-sys-record')
    parser.add_argument('--rocprof', default=None,
                        help='rocprofv3 可执行文件路径（兼容旧参数）')
    parser.add_argument('--amd-smi', dest='amdsmi', default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--selftest', action='store_true',
                        help='只做 collector 探测与 --help flag 校验，不跑 GPU')
    parser.add_argument('--validate-existing', default=None, metavar='PATH',
                        help='只读校验已有 capture_manifest.json/目录')
    parser.add_argument('--validation-output', default=None, metavar='PATH')
    return parser.parse_args(argv)


def _load_existing_manifest(path_arg: str) -> Tuple[Path, Dict[str, Any]]:
    source = Path(path_arg).expanduser()
    manifest_path = source / 'capture_manifest.json' if source.is_dir() else source
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
    """只读校验路径：重跑证据校验，不启动 collector、不碰 GPU。"""
    try:
        manifest_path, original = _load_existing_manifest(args.validate_existing)
    except (FileNotFoundError, ValueError) as exc:
        print(f'[validate-existing] ERROR {exc}', file=sys.stderr)
        return 4
    out_dir = manifest_path.parent
    artifacts = original.get('artifacts') or []
    by_name = {Path(item.get('path', '')).name: Path(item['path'])
               for item in artifacts if isinstance(item, dict) and item.get('path')}
    family = ((original.get('collector') or {}).get('family')
              or (original.get('rocprof') or {}).get('collector')
              or ROCPROFV3_FAMILY)
    sampler_path = out_dir / 'amdsmi_samples.jsonl'
    log_text = ''
    log_path = out_dir / 'capture_rocprof.log'
    if log_path.is_file():
        log_text = log_path.read_text(encoding='utf-8', errors='replace')
        if log_text.startswith('$ '):
            log_text = log_text.split('\n', 1)[1] if '\n' in log_text else ''
    contract = original.get('capture_contract') or {}
    hw_requested = bool(contract.get('hw_requested'))
    pmc = contract.get('pmc') or []
    sampler = None
    if sampler_path.is_file():
        sampler = sampler_stats(sampler_path,
                                (contract.get('amdsmi_sample') or {})
                                .get('requested_interval_s',
                                     DEFAULT_AMDSMI_INTERVAL_SEC))
    needle = original.get('kernel_name_contains') or ''
    if family == ROCPROF_SYS_FAMILY:
        discovery = (original.get('collector_artifacts')
                     or original.get('rocprof_sys_artifacts')
                     or discover_rocprof_sys_artifacts(
                         out_dir / 'rocprof_out'))
        evidence = verify_rocprof_sys_evidence(discovery, log_text, needle)
    else:
        json_path = by_name.get('capture_results.json',
                                out_dir / 'capture_results.json')
        pftrace_path = by_name.get('capture_results.pftrace',
                                   out_dir / 'capture_results.pftrace')
        evidence = verify_evidence(json_path, pftrace_path, log_text, needle,
                                   hw_requested, pmc, sampler)
    strengthened = dict(original)
    strengthened['validation_source_manifest'] = str(manifest_path)
    strengthened['evidence'] = evidence
    strengthened['verdict'] = evidence['verdict']
    if args.validation_output:
        output = Path(args.validation_output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(strengthened, indent=2, ensure_ascii=False),
                          encoding='utf-8')
        print(f'[validate-existing] manifest {output}')
    else:
        print(json.dumps(strengthened, indent=2, ensure_ascii=False))
    return 0 if evidence['verdict'] == 'PASS' else 4


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    if args.validate_existing is not None:
        return validate_existing_capture(args)

    rocm_domains, rocm_events = resolve_rocm_options(
        args.rocm_domains, args.rocm_events)

    # 默认 PMC 集解析（--pmc-defaults 或 --pmc ''）。在 collector 选择和
    # persistent 预检之前完成，确保 --pmc 路径不会被误判成 timeline-only。
    if args.pmc_defaults:
        args.pmc = list(DEFAULT_PMC)
    elif args.pmc is not None:
        args.pmc = [item for item in args.pmc if item.strip()] or list(DEFAULT_PMC)
    hw_requested = bool(args.pmc)

    selection = select_collector(
        requested=args.collector,
        rocprof_sys=args.rocprof_sys,
        rocprofv3=args.rocprof,
        hw_requested=hw_requested,
        raw_rocm_events=rocm_events,
        amdsmi_sampler=hw_requested and not args.no_amdsmi_sampler,
        which=shutil.which,
        verify_flags=True,
    )

    if args.selftest:
        if selection.get('hard_failures'):
            report = {'collector_selection': selection,
                      'hard_failures': selection['hard_failures']}
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 4
        flag_report = ({'collector': ROCPROF_SYS_FAMILY,
                        'probe': selection.get('capability_probe')}
                       if selection['family'] == ROCPROF_SYS_FAMILY
                       else selftest_flags(selection['path']))
        report = {
            'collector_selection': selection,
            'flag_selftest': flag_report,
            'hard_failures': list(flag_report.get('hard_failures') or []),
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if not report['hard_failures'] else 4

    if selection.get('hard_failures'):
        for failure in selection['hard_failures']:
            print(f'[fatal] {failure}')
        return 1

    raw_rocm_applied = (selection.get('family') == ROCPROF_SYS_FAMILY
                        and bool(rocm_events))
    counter_requested = hw_requested or raw_rocm_applied

    if not args.root:
        print('[fatal] 采集必须指定 --root')
        return 1
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f'[fatal] --root 不是目录: {root}')
        return 1
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else \
        (root / '.work' / 'profile_runs' / 'hw_capture_rocm').resolve()
    if out_dir == root or root not in out_dir.parents:
        print(f'[fatal] out-dir 必须在 --root 之下且不能是仓库根: {out_dir}')
        return 1
    if not args.payload:
        print('[fatal] 需要 --payload 指定被采集程序（放在所有参数最后）')
        return 1
    payload_original = list(args.payload)
    payload, payload_resolution = resolve_payload_paths(payload_original, root)

    # 纪律 1：persistent fail-closed（在任何 GPU 动作之前）
    contract_info = load_app_contract(args.contract)
    guard_failures = list(contract_info.get('hard_failures', []))
    guard_failures.extend(persistent_guard(contract_info, counter_requested))
    if guard_failures:
        for failure in guard_failures:
            print(f'[fatal] {failure}')
        return 1

    amdsmi = args.amdsmi or shutil.which('amd-smi')
    if not amdsmi:
        print('[fatal] 找不到 amd-smi 可执行文件，用 --amd-smi 指定')
        return 1

    collector_family = selection['family']
    collector_path = selection['path']
    version = selection.get('version')
    if collector_family == ROCPROFV3_FAMILY:
        flag_report = selftest_flags(collector_path)
        selection['capability_probe'] = flag_report
    else:
        flag_report = selection.get('capability_probe') or {}
    if flag_report.get('hard_failures'):
        for failure in flag_report['hard_failures']:
            print(f'[fatal] {failure}')
        return 1

    hip_visible = args.hip_visible_devices if args.hip_visible_devices is not None \
        else os.environ.get('HIP_VISIBLE_DEVICES')
    visible = resolve_visible_devices(hip_visible, args.num_processes)
    child_env = dict(os.environ)
    if hip_visible is not None:
        child_env['HIP_VISIBLE_DEVICES'] = hip_visible

    target_indices = sorted(set(visible['harness_physical_indices']) or {0})
    sample_gpu = target_indices[0]
    gpu_check = inspect_gpus(amdsmi, target_indices, args.mem_threshold_mib,
                             args.util_threshold_pct)
    gpu_check['hip_visible_devices'] = visible

    rocprof_out_dir = out_dir / 'rocprof_out'
    json_path = rocprof_out_dir / 'capture_results.json'
    pftrace_path = rocprof_out_dir / 'capture_results.pftrace'
    sampler_path = out_dir / 'amdsmi_samples.jsonl'
    if collector_family == ROCPROF_SYS_FAMILY:
        rocprof_cmd = build_rocprof_sys_cmd(
            collector_path, rocprof_out_dir, payload,
            buffer_kb=args.perfetto_buffer_kb,
            rocm_domains=rocm_domains,
            rocm_events=rocm_events)
    else:
        rocprof_cmd = build_rocprof_cmd(args, collector_path,
                                        rocprof_out_dir, payload)
    estimate = estimate_sizes(args.estimate_dispatches, args.estimate_seconds,
                              hw_requested)

    print(f'[plan] collector      {collector_family}')
    print(f'[plan] executable     {collector_path}')
    print(f'[plan] version        {version}')
    fallback = selection.get('fallback') or {}
    if fallback.get('used'):
        print(f'[plan] fallback       {fallback.get("from")} -> '
              f'{fallback.get("to")}；{fallback.get("reason")}')
    print(f'[plan] selection      {selection.get("selection_reason")}')
    print(f'[plan] root           {root}')
    print(f'[plan] out-dir (cwd)  {out_dir}')
    if payload_resolution['rewrites']:
        print(f'[plan] payload paths  {len(payload_resolution["rewrites"])} 项已按 root 解析为绝对路径')
    bundle_label = ('sw+hw' if hw_requested else
                    'sw+rocprof-sys raw counters' if raw_rocm_applied
                    else 'sw（timeline-only）')
    print(f'[plan] bundle         {bundle_label}')
    if raw_rocm_applied:
        print('[plan] ROCm counters  rocprof-sys same-run per-dispatch raw events；'
              '非周期 PM 采样')
        print(f'[plan] ROCm domains   {",".join(rocm_domains)}')
        print(f'[plan] ROCm events    {",".join(rocm_events)}')
    if hw_requested:
        print(f'[plan] PMC            {len(args.pmc)} 项 per-dispatch: '
              f'{",".join(args.pmc)}')
        print('[plan] hw 语义        per-dispatch PMC + amd-smi 粗采样；'
              'AMD 无 nsys 200kHz 对应物（纪律 3，契约如实标注）')
        print(f'[plan] amd-smi 采样   GPU {sample_gpu}，目标间隔 '
              f'{args.amdsmi_sample_interval} s（实测频率写入契约）')
    print(f'[plan] 契约           {contract_info["path"]} '
          f'(persistent={contract_info["persistent"]})')
    print(f'[plan] HIP_VISIBLE_DEVICES={visible["hip_visible_devices"]!r} '
          f'selector={visible["selector"]} '
          f'harness 物理卡={visible["harness_physical_indices"]}')
    print(f'[plan] {collector_family} 命令:')
    print('       ' + ' '.join(shlex.quote(part) for part in rocprof_cmd))
    print(f'[plan] 预估产物（假设 {estimate["assumed_dispatches"]} dispatch / '
          f'{estimate["assumed_capture_seconds"]:.0f} s）: '
          f'pftrace ~{estimate["estimated_pftrace_bytes"] / 1e6:.1f} MB, '
          f'json ~{estimate["estimated_json_bytes"] / 1e6:.1f} MB')
    print(f'[gpu] amd_smi_available={gpu_check["amd_smi_available"]} '
          f'busy={gpu_check["busy"]} '
          f'shared_system_pids={gpu_check["shared_system_pids"]}')
    for entry in gpu_check['gpus']:
        print(f'[gpu]   GPU {entry["index"]} mem_used={entry.get("memory_used_mib")} MiB '
              f'util={entry.get("utilization_pct")} '
              f'over_mem={entry.get("over_memory")} '
              f'over_util={entry.get("over_utilization")}')
    for reason in gpu_check['reasons']:
        print(f'[gpu]   reason: {reason}')

    if gpu_check['busy'] and not args.force:
        print('[gpu] ERROR 目标 GPU 被占用或状态未知（绝不 kill 任何进程）。'
              '等空闲后重跑，或显式加 --force。')
        return 2
    if gpu_check['busy']:
        print('[gpu] WARNING 目标 GPU 有占用，--force 已指定，继续执行')

    if args.dry_run:
        print('[dry-run] 仅打印计划，未创建目录、未启动 collector、未写任何文件')
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    rocprof_out_dir.mkdir(parents=True, exist_ok=True)
    before_bytes = dir_bytes(out_dir)

    sampler: Optional[AmdSmiSampler] = None
    if hw_requested and not args.no_amdsmi_sampler:
        sampler = AmdSmiSampler(amdsmi, sample_gpu, args.amdsmi_sample_interval,
                                sampler_path)
        sampler.start()
        print(f'[run] amd-smi 粗采样已启动（GPU {sample_gpu} → {sampler_path}）')

    started = time.time()
    print(f'[run] cd {out_dir} 后启动 {collector_family}')
    capture_log = out_dir / 'capture_rocprof.log'
    code, log_text = run_capture(rocprof_cmd, out_dir, capture_log,
                                 args.timeout_sec, child_env)
    capture_seconds = time.time() - started
    print(f'[run] {collector_family} 退出码 {code}，耗时 '
          f'{capture_seconds:.1f} s')

    if sampler is not None:
        sampler.stop()
        print(f'[run] amd-smi 粗采样停止：{sampler.samples} 样本 / '
              f'{sampler.errors} 错误')

    sampler_report = None
    if sampler_path.is_file():
        sampler_report = sampler_stats(sampler_path, args.amdsmi_sample_interval)

    collector_artifacts = None
    artifact_rows = []
    if collector_family == ROCPROF_SYS_FAMILY:
        collector_artifacts = discover_rocprof_sys_artifacts(
            rocprof_out_dir, since_epoch=started)
        for role, key in (('perfetto-trace', 'trace_files'),
                          ('metadata', 'metadata_files'),
                          ('functions', 'function_files')):
            for path_text in collector_artifacts.get(key) or []:
                artifact_rows.append((Path(path_text), role))
        evidence = verify_rocprof_sys_evidence(
            collector_artifacts, log_text, args.kernel_name_contains)
    else:
        for role, path in (('perfetto-trace', pftrace_path),
                           ('rocprofv3-json', json_path)):
            if path.is_file():
                artifact_rows.append((path, role))
        evidence = verify_evidence(json_path, pftrace_path, log_text,
                                   args.kernel_name_contains, hw_requested,
                                   args.pmc or [], sampler_report)
    for role, path in (('log', capture_log), ('amdsmi-samples', sampler_path)):
        if path.is_file():
            artifact_rows.append((path, role))

    artifacts = []
    seen_artifacts = set()
    for path, role in artifact_rows:
        key = str(path.resolve())
        if key in seen_artifacts or not path.is_file():
            continue
        seen_artifacts.add(key)
        artifacts.append({
            'path': str(path),
            'role': role,
            'bytes': path.stat().st_size,
            'sha256': sha256_of(path),
        })

    if code != 0:
        evidence['failures'].append(f'{collector_family} 退出码 {code}')
    evidence['verdict'] = 'FAIL' if evidence['failures'] else 'PASS'

    rocm_counter_contract = {
        'requested': args.rocm_events is not None,
        'applied': raw_rocm_applied,
        'domains': list(rocm_domains),
        'events': list(rocm_events),
        'collector': (
            ROCPROF_SYS_FAMILY if raw_rocm_applied else None),
        'counter_granularity': 'per-dispatch' if raw_rocm_applied else None,
        'periodic_pm_sampling': False,
        'equivalent_to_nsys_200khz': False,
        'semantics': (
            'rocprof-sys --rocm-events emits same-run raw ROCm counters in '
            'Perfetto at dispatch boundaries; values are per-dispatch, not a '
            'periodic PM time series' if raw_rocm_applied else
            'rocprof-sys raw ROCm counters were not applied on this capture'),
    }

    capture_contract = {
        'platform': 'rocm/gfx942',
        'target_card': 'mi308x',
        'target_arch': 'gfx942',
        'bundle': ('sw+hw' if hw_requested else
                   'sw+raw-rocm-counters' if raw_rocm_applied else 'sw'),
        'collector': collector_family,
        'collector_path': collector_path,
        'collector_version': version,
        'collector_mode': selection['capabilities']['sampling']['mode'],
        'sampling': selection['capabilities']['sampling'],
        'persistent': contract_info['persistent'],
        'contract_path': contract_info['path'],
        'hw_requested': hw_requested,
        'raw_rocm_counters_requested': bool(rocm_events),
        'raw_rocm_counters_applied': raw_rocm_applied,
        'hw_semantics': ('per-dispatch PMC（--pmc）+ amd-smi 粗采样；'
                         'AMD 无 nsys --gpu-metrics 200kHz 周期采样对应物'
                         if hw_requested else None),
        'timeline_semantics': (
            ('rocprof-sys 事件边界 trace + same-run per-dispatch 原始 ROCm '
             'counters；非周期 GPU PM 采样；不等价于 nsys 200 kHz GPU metrics')
            if raw_rocm_applied else
            ('rocprof-sys 事件边界 trace；非周期 GPU PM 采样；不等价于 '
             'nsys 200 kHz GPU metrics') if collector_family == ROCPROF_SYS_FAMILY
            else 'rocprofv3 事件边界 trace；非周期 GPU PM 采样；不等价于 '
                 'nsys 200 kHz GPU metrics'),
        'rocm_domains': list(rocm_domains),
        'rocm_events': list(rocm_events),
        'rocm_counter_semantics': rocm_counter_contract,
        'pmc': args.pmc or [],
        'amdsmi_sample': ({
            'gpu': sample_gpu,
            'requested_interval_s': args.amdsmi_sample_interval,
            'measured_hz': (sampler_report or {}).get('measured_hz'),
            'interval_s': (sampler_report or {}).get('interval_s'),
            'sample_count': (sampler_report or {}).get('sample_count'),
            'note': '粗采样量级 2-5 Hz，比 nsys 200 kHz 低 5 个数量级；'
                    '对齐时禁止当等值物用',
        } if sampler_report is not None else None),
    }

    collector_record = dict(selection)
    collector_record['argv'] = list(rocprof_cmd)
    collector_record['argv_shell'] = command_line(rocprof_cmd)
    collector_record['exit_code'] = code
    collector_record['capture_seconds'] = capture_seconds
    collector_record['capability_probe'] = flag_report
    collector_record['rocm_domains'] = list(rocm_domains)
    collector_record['rocm_events'] = list(rocm_events)
    collector_record['rocm_counter_semantics'] = rocm_counter_contract

    manifest = {
        'schema': SCHEMA,
        'generated_at_utc': utc_iso(time.time()),
        'actual_collector': collector_family,
        'collector_family': collector_family,
        'target_card': 'mi308x',
        'target_arch': 'gfx942',
        'collector': collector_record,
        'collector_argv': list(rocprof_cmd),
        'collector_command_line': command_line(rocprof_cmd),
        'rocm_domains': list(rocm_domains),
        'rocm_events': list(rocm_events),
        'rocm_counter_semantics': rocm_counter_contract,
        'fallback_reason': (collector_record.get('fallback') or {}).get('reason'),
        'sampling_capabilities': selection['capabilities'],
        'collector_artifacts': collector_artifacts,
        # 兼容既有读取方；新增的真相以 collector 字段为准。
        'rocprof': {'collector': collector_family, 'path': collector_path,
                    'version': version, 'flag_selftest': flag_report},
        'amd_smi': {'path': amdsmi},
        'root': str(root),
        'out_dir': str(out_dir),
        'cwd_for_capture': str(out_dir),
        'child_env_overrides': {'HIP_VISIBLE_DEVICES': child_env.get('HIP_VISIBLE_DEVICES')},
        'hip_visible_devices': visible,
        'capture_contract': capture_contract,
        'contract_preflight': contract_info,
        'kernel_name_contains': args.kernel_name_contains,
        'num_processes': args.num_processes,
        'argv': list(argv),
        'rocprof_command': rocprof_cmd,
        'rocprof_command_line': command_line(rocprof_cmd),
        'payload_command': payload,
        'payload_command_line': ' '.join(shlex.quote(part) for part in payload),
        'payload_command_original': payload_original,
        'payload_path_resolution': payload_resolution,
        'exit_codes': {collector_family: code},
        'capture_seconds': capture_seconds,
        'gpu_check': gpu_check,
        'size_estimate': estimate,
        'artifacts': artifacts,
        'out_dir_bytes': {'before': before_bytes, 'after': dir_bytes(out_dir)},
        'evidence': evidence,
        'verdict': evidence['verdict'],
    }
    manifest_path = out_dir / 'capture_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding='utf-8')

    print(f'[out] manifest {manifest_path}')
    for item in artifacts:
        print(f'[out]   {item["path"]} {item["bytes"]} B sha256={item["sha256"][:16]}...')
    kernel = evidence.get('kernel', {})
    if kernel.get('available') and kernel.get('structural_verification', True):
        print(f'[evidence] kernel_dispatch 总数={kernel.get("total_dispatches")} '
              f'匹配 needle={kernel.get("match_rows")} '
              f'(agents={kernel.get("dispatch_count_by_agent_handle")})')
    elif kernel.get('checked'):
        print(f'[evidence] kernel needle 明文观测={kernel.get("observed")} '
              '（rocprof-sys timeline 非结构化校验）')
    if hw_requested and evidence.get('counters'):
        print(f'[evidence] PMC counter_collection 记录='
              f'{evidence["counters"].get("record_count")}（per-dispatch）')
    if sampler_report:
        print(f'[evidence] amd-smi 粗采样 {sampler_report.get("sample_count")} 样本，'
              f'实测 {sampler_report.get("measured_hz", 0):.1f} Hz')
    for warning in evidence['warnings']:
        print(f'[evidence] WARN {warning}')
    for failure in evidence['failures']:
        print(f'[evidence] FAIL {failure}')
    print(f'[evidence] verdict={evidence["verdict"]}')
    return 0 if evidence['verdict'] == 'PASS' else 4


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
