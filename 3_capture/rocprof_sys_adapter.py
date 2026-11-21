#!/usr/bin/env python3
"""ROCm collector selection and rocprof-sys command/evidence adapter.

This module is intentionally independent of ``capture_rocm.py`` so the
collector policy can be tested without a GPU.  It prefers rocprof-sys for
event/timeline capture on MI308X and uses rocprofv3 when that collector is
unavailable or when the requested workflow needs rocprofv3's structured
per-dispatch ``counter_collection`` output.

The declarations in this module are deliberately conservative:

* trace capture records event boundaries; it is not periodic GPU PM sampling;
* rocprof-sys ``--rocm-events`` adds same-run raw counters at dispatch
  boundaries, still not a periodic PM time series;
* rocprofv3 ``--pmc`` is a per-dispatch aggregate, not a time series;
* amd-smi polling is coarse utilization telemetry, not ROCm PM counters; and
* none of these paths is equivalent to nsys 200 kHz GPU metrics.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROCPROF_SYS_FAMILY = 'rocprof-sys'
ROCPROFV3_FAMILY = 'rocprofv3'
COLLECTOR_CHOICES = ('auto', ROCPROF_SYS_FAMILY, ROCPROFV3_FAMILY)

# Prefer the canonical name, then the current ROCm launcher, then the older
# record-style wrapper.  Detection checks each name independently; no shell
# alias or shell function is involved.
ROCPROF_SYS_ALIASES: Tuple[str, ...] = (
    'rocprof-sys',
    'rocprof-sys-run',
    'rocprof-sys-record',
)

# Flags used by build_rocprof_sys_cmd.  A candidate that lacks these flags is
# rejected before a GPU is touched and the policy can fall back to rocprofv3.
ROCPROF_SYS_FLAGS_USED: Tuple[str, ...] = (
    '--trace',
    '--use-rocm',
    '--rocm-domains',
    '--rocm-events',
    '--output',
    '--perfetto-backend',
    '--trace-buffer-size',
    '--perfetto-shmem-size-hint-kb',
    '--help',
    '--version',
)

# Minimal same-run route for the MI308X GEMM hardware layer.  The six event
# names match the raw Perfetto counter tracks consumed by
# build_rocprof_sys_sw_hw.py.  Values remain per-dispatch counters emitted at
# dispatch boundaries; they are not periodic PM samples.
DEFAULT_ROCM_DOMAINS: Tuple[str, ...] = (
    'kernel_dispatch',
)
DEFAULT_ROCM_EVENTS: Tuple[str, ...] = (
    'SQ_VALU_MFMA_BUSY_CYCLES',
    'GRBM_GUI_ACTIVE',
    'SIMD_UTILIZATION',
    'GPU_UTIL',
    'TCC_EA0_RDREQ_DRAM',
    'TCC_EA0_WRREQ_DRAM',
)

ROCPROF_SYS_PMC_ROUTE_REASON = (
    '请求了 --pmc：当前增强沿用 rocprofv3 的 '
    'callback_records.counter_collection 以提供可校验的 per-dispatch PMC；'
    'rocprof-sys 的 --rocm-events 是另一条同 run 原始 counter 路线，'
    '不声明周期 PM 采样，也不提供 --pmc 所需的 callback_records。'
)

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


def _clean_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    return _ANSI_RE.sub('', str(value))


def _excerpt(text: str, limit: int = 12000) -> Dict[str, Any]:
    clean = _clean_text(text)
    result: Dict[str, Any] = {
        'text': clean[:limit],
        'bytes': len(clean.encode('utf-8')),
    }
    if len(clean) > limit:
        result['truncated'] = True
    return result


def _resolve_binary(explicit: Optional[str], aliases: Sequence[str],
                    which: Callable[[str], Optional[str]]) -> Optional[str]:
    """Resolve an explicit path or the first executable alias found on PATH."""
    if explicit:
        expanded = os.path.expanduser(explicit)
        candidate = Path(expanded)
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return str(candidate.resolve())
        found = which(expanded)
        if found:
            return str(Path(found).resolve())
        return None
    for name in aliases:
        found = which(name)
        if found:
            return str(Path(found).resolve())
    return None


def probe_version(binary: str, *,
                  runner: Callable[..., Any] = subprocess.run,
                  timeout: float = 15.0) -> Dict[str, Any]:
    """Probe ``--version`` and retain the exact argv plus both output streams."""
    argv = [binary, '--version']
    probe: Dict[str, Any] = {
        'method': '--version',
        'argv': argv,
        'success': False,
        'version': None,
    }
    try:
        proc = runner(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        probe.update({'exit_code': None, 'stdout': '', 'stderr': '',
                      'error': f'{type(exc).__name__}: {exc}'})
        return probe

    stdout = _clean_text(getattr(proc, 'stdout', ''))
    stderr = _clean_text(getattr(proc, 'stderr', ''))
    code = getattr(proc, 'returncode', None)
    probe.update({
        'exit_code': code,
        'stdout': stdout,
        'stderr': stderr,
        'success': code == 0,
    })
    if code != 0:
        probe['error'] = f'--version exited {code}'
        return probe

    candidates = [line.strip() for line in (stdout + '\n' + stderr).splitlines()
                  if line.strip()]
    if candidates:
        # Keep the human-readable first line, while also exposing a compact
        # semantic version when one is present.
        probe['version'] = candidates[0]
        match = re.search(r'(?<!\d)v?(\d+\.\d+(?:\.\d+)?(?:[-+][0-9A-Za-z.-]+)?)',
                          candidates[0])
        if match:
            probe['version_number'] = match.group(1)
    return probe


def probe_cli(binary: str, family: str, *,
              runner: Callable[..., Any] = subprocess.run,
              timeout: float = 20.0,
              required_flags: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Check the flags used by this adapter without launching a payload."""
    flags = tuple(required_flags if required_flags is not None
                  else (ROCPROF_SYS_FLAGS_USED if family == ROCPROF_SYS_FAMILY
                        else ()))
    argv = [binary, '--help']
    result: Dict[str, Any] = {
        'argv': argv,
        'success': False,
        'flags_checked': list(flags),
        'flags_missing': [],
        'hard_failures': [],
    }
    try:
        proc = runner(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['hard_failures'].append(f'{family} --help 失败: {exc}')
        return result

    stdout = _clean_text(getattr(proc, 'stdout', ''))
    stderr = _clean_text(getattr(proc, 'stderr', ''))
    help_text = stdout + '\n' + stderr
    code = getattr(proc, 'returncode', None)
    missing = [flag for flag in flags if flag not in help_text]
    result.update({
        'exit_code': code,
        'success': code == 0,
        'flags_missing': missing,
        'stdout_sha256': hashlib.sha256(stdout.encode('utf-8')).hexdigest(),
        'stdout_bytes': len(stdout.encode('utf-8')),
        'stdout_excerpt': _excerpt(stdout, 1200),
        'stderr': stderr,
    })
    if code != 0:
        result['hard_failures'].append(f'{family} --help 退出码 {code}')
    if missing:
        result['hard_failures'].append(
            f'{family} --help 缺少本适配器使用的 flag: {missing}')
    return result


def command_line(argv: Sequence[str]) -> str:
    """Return a shell-replayable line while preserving argv as the truth."""
    return ' '.join(shlex.quote(str(part)) for part in argv)


def install_manifest(binary: str, version_probe: Dict[str, Any],
                     cli_probe: Optional[Dict[str, Any]], family: str) -> Dict[str, Any]:
    """Build the serializable install/probe section stored in the manifest."""
    return {
        'family': family,
        'binary': Path(binary).name,
        'path': str(binary),
        'version': version_probe.get('version'),
        'version_number': version_probe.get('version_number'),
        'version_probe': version_probe,
        'cli_probe': cli_probe,
    }


def capability_declaration(family: str, *, pmc_requested: bool,
                           amdsmi_sampler: bool = False,
                           raw_rocm_events: Sequence[str] = ()) -> Dict[str, Any]:
    """Describe what this collector route really collects.

    ``periodic_pm_sampling`` is false on every route.  rocprof-sys traces
    individual event boundaries and can add same-run raw counters at those
    boundaries; rocprofv3 PMC is emitted per kernel dispatch; amd-smi is a
    coarse side channel.  Keeping these fields explicit prevents downstream
    consumers from relabeling any of them as a 200 kHz time series.
    """
    requested_raw_events = list(raw_rocm_events or ())
    raw_applied = family == ROCPROF_SYS_FAMILY and bool(requested_raw_events)
    raw_events = requested_raw_events if raw_applied else []
    if family == ROCPROF_SYS_FAMILY:
        granularity = 'per-dispatch' if raw_events else 'event-boundary'
        mode = 'event-trace+per-dispatch-raw-counters' if raw_events else 'event-trace'
        pmc_supported = False
        amdsmi_used = False
        if raw_events:
            detail = (
                'rocprof-sys --trace/--use-rocm --rocm-events 在同一次运行中记录 '
                'HIP/HSA/kernel dispatch 事件边界和每个 dispatch 的原始 ROCm counter；'
                'counter 按 dispatch 粒度输出，不是周期 GPU PM 采样。'
            )
        else:
            detail = (
                'rocprof-sys --trace/--use-rocm 记录 HIP/HSA/kernel dispatch 等事件边界；'
                '本次未请求 --rocm-events，不是周期 GPU PM 采样。'
            )
    elif family == ROCPROFV3_FAMILY and pmc_requested:
        granularity = 'per-dispatch'
        mode = 'per-dispatch-counter+amdsmi-coarse'
        pmc_supported = True
        amdsmi_used = bool(amdsmi_sampler)
        detail = (
            'rocprofv3 --pmc 为每次 kernel dispatch 输出一组汇总计数；'
            '不是周期时间序列。amd-smi 侧车是约 2-5 Hz 的粗采样。'
        )
    elif family == ROCPROFV3_FAMILY:
        granularity = 'event-boundary'
        mode = 'event-trace'
        pmc_supported = False
        amdsmi_used = False
        detail = 'rocprofv3 此次仅做 kernel/HIP timeline trace，未请求 PMC。'
    else:
        raise ValueError(f'unknown collector family: {family!r}')

    sampling = {
        'mode': mode,
        'counter_granularity': granularity,
        'periodic_pm_sampling': False,
        'equivalent_to_nsys_200khz': False,
        'pmc_counters': pmc_supported,
        'raw_rocm_counters': raw_applied,
        'raw_rocm_events': raw_events,
        'amdsmi_coarse_sampling': amdsmi_used,
        'notes': [
            detail,
            '任何 trace 或 per-dispatch 计数都不能改写为 nsys 200 kHz 周期采样。',
        ],
    }
    return {
        'timeline_trace': True,
        'kernel_dispatch_trace': True,
        'hip_hsa_trace': True,
        'periodic_pm_sampling': False,
        'equivalent_to_nsys_200khz': False,
        'pmc_counters': pmc_supported,
        'raw_rocm_counters': raw_applied,
        'raw_rocm_events': raw_events,
        'amdsmi_coarse_sampling': amdsmi_used,
        'sampling': sampling,
    }


def select_collector(*, requested: str = 'auto',
                     rocprof_sys: Optional[str] = None,
                     rocprofv3: Optional[str] = None,
                     hw_requested: bool = False,
                     raw_rocm_events: Sequence[str] = (),
                     amdsmi_sampler: bool = False,
                     which: Callable[[str], Optional[str]] = shutil.which,
                     runner: Callable[..., Any] = subprocess.run,
                     verify_flags: bool = True) -> Dict[str, Any]:
    """Choose the actual collector and record every fallback/probe decision.

    Policy:
    * timeline-only: rocprof-sys first, rocprofv3 only after a preflight reason;
    * ``--rocm-events``: keep rocprof-sys and attach same-run per-dispatch raw
      counters to its Perfetto trace;
    * ``--pmc``: route to rocprofv3 for structured per-dispatch counters;
    * an explicit collector request is still represented, but an unavailable
      rocprof-sys falls back to rocprofv3 with a machine-readable reason.
    """
    if requested not in COLLECTOR_CHOICES:
        raise ValueError(f'collector must be one of {COLLECTOR_CHOICES}, got {requested!r}')

    def choose(family: str, reason: str,
               fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if family == ROCPROF_SYS_FAMILY:
            binary = sys_path
            probe = sys_version_probe
            cli = sys_cli_probe
        else:
            binary = v3_path
            probe = v3_version_probe
            cli = v3_cli_probe
        assert binary is not None
        assert probe is not None
        fallback_value = fallback or {
            'used': False,
            'from': None,
            'to': None,
            'reason': None,
        }
        sys_candidate = ({
            'path': sys_path,
            'version_probe': sys_version_probe,
            'cli_probe': sys_cli_probe,
        } if sys_path else None)
        if fallback_value.get('used') and sys_candidate is not None:
            fallback_value = dict(fallback_value)
            fallback_value['source_candidate'] = sys_candidate
        selection = {
            'requested': requested,
            'family': family,
            'binary': Path(binary).name,
            'path': str(binary),
            'version': probe.get('version'),
            'version_probe': probe,
            'capability_probe': cli,
            'selection_reason': reason,
            'fallback': fallback_value,
            'rocprof_sys_candidate': sys_candidate,
            'capabilities': capability_declaration(
                family, pmc_requested=hw_requested,
                raw_rocm_events=raw_rocm_events,
                amdsmi_sampler=amdsmi_sampler),
            'hard_failures': [],
        }
        selection['install'] = install_manifest(binary, probe, cli, family)
        return selection

    sys_path: Optional[str] = None
    sys_version_probe: Optional[Dict[str, Any]] = None
    sys_cli_probe: Optional[Dict[str, Any]] = None
    v3_path: Optional[str] = None
    v3_version_probe: Optional[Dict[str, Any]] = None
    v3_cli_probe: Optional[Dict[str, Any]] = None

    if requested != ROCPROFV3_FAMILY:
        sys_path = _resolve_binary(rocprof_sys, ROCPROF_SYS_ALIASES, which)
        if sys_path:
            sys_version_probe = probe_version(sys_path, runner=runner)
            # A non-zero --version is recorded but is not itself a reason to
            # discard an otherwise executable rocprof-sys.
            if verify_flags:
                sys_cli_probe = probe_cli(
                    sys_path, ROCPROF_SYS_FAMILY, runner=runner)
    else:
        v3_path = _resolve_binary(rocprofv3, ('rocprofv3',), which)

    def resolve_v3() -> Optional[str]:
        nonlocal v3_path, v3_version_probe, v3_cli_probe
        if not v3_path:
            v3_path = _resolve_binary(rocprofv3, ('rocprofv3',), which)
        if v3_path and v3_version_probe is None:
            v3_version_probe = probe_version(v3_path, runner=runner)
            if verify_flags and requested == ROCPROFV3_FAMILY:
                v3_cli_probe = probe_cli(v3_path, ROCPROFV3_FAMILY, runner=runner)
        return v3_path

    if requested == ROCPROFV3_FAMILY:
        if not resolve_v3():
            return {
                'requested': requested,
                'family': None,
                'binary': None,
                'path': None,
                'hard_failures': [
                    '显式请求 rocprofv3，但找不到该可执行文件。'],
                'fallback': {'used': False, 'from': None, 'to': None,
                             'reason': 'collector=rocprofv3 且未找到 rocprofv3'},
            }
        return choose(ROCPROFV3_FAMILY,
                      '用户显式指定 collector=rocprofv3。')

    sys_preflight_reason: Optional[str] = None
    if not sys_path:
        sys_preflight_reason = (
            'rocprof-sys 不可用：未找到以下任一可执行文件 '
            f'{list(ROCPROF_SYS_ALIASES)}。')
    elif sys_cli_probe and sys_cli_probe.get('hard_failures'):
        sys_preflight_reason = (
            'rocprof-sys 命令预检失败：'
            + '; '.join(sys_cli_probe.get('hard_failures') or []))

    if hw_requested:
        capability_reason = ROCPROF_SYS_PMC_ROUTE_REASON
        if sys_preflight_reason:
            reason = sys_preflight_reason + ' ' + capability_reason
        else:
            reason = (f'已检测到 {sys_path}（版本 '
                      f'{sys_version_probe.get("version") if sys_version_probe else None}），'
                      f'但{capability_reason}')
        if not resolve_v3():
            return {
                'requested': requested,
                'family': None,
                'binary': None,
                'path': None,
                'hard_failures': [
                    '请求 --pmc 需要 rocprofv3 的结构化 counter_collection，'
                    '但 rocprofv3 不可用；拒绝用无证据能力的 rocprof-sys '
                    'timeline 冒充 PMC。'],
                'fallback': {'used': True, 'from': ROCPROF_SYS_FAMILY,
                             'to': ROCPROFV3_FAMILY, 'trigger': 'capability',
                             'reason': reason},
                'rocprof_sys_candidate': ({
                    'path': sys_path,
                    'version_probe': sys_version_probe,
                    'cli_probe': sys_cli_probe,
                } if sys_path else None),
            }
        return choose(
            ROCPROFV3_FAMILY, reason,
            {'used': True, 'from': ROCPROF_SYS_FAMILY,
             'to': ROCPROFV3_FAMILY,
             'trigger': 'capability', 'reason': reason})

    if sys_path and not sys_preflight_reason:
        return choose(
            ROCPROF_SYS_FAMILY,
            f'auto/显式请求优先使用 rocprof-sys timeline；命中 {sys_path}。')

    if not resolve_v3():
        return {
            'requested': requested,
            'family': None,
            'binary': None,
            'path': None,
            'hard_failures': [
                (sys_preflight_reason or 'rocprof-sys 不可用') +
                ' 同时找不到 rocprofv3 可执行文件（可用 --rocprof 指定）。'],
            'fallback': {'used': True, 'from': ROCPROF_SYS_FAMILY,
                         'to': ROCPROFV3_FAMILY, 'trigger': 'not-found',
                         'reason': sys_preflight_reason or 'rocprof-sys 不可用'},
            'rocprof_sys_candidate': ({
                'path': sys_path,
                'version_probe': sys_version_probe,
                'cli_probe': sys_cli_probe,
            } if sys_path else None),
        }
    reason = (sys_preflight_reason or 'rocprof-sys 不可用') + \
        ' 已明确降级到 rocprofv3 timeline。'
    trigger = 'preflight' if sys_path else 'not-found'
    return choose(
        ROCPROFV3_FAMILY, reason,
        {'used': True, 'from': ROCPROF_SYS_FAMILY,
         'to': ROCPROFV3_FAMILY, 'trigger': trigger,
         'reason': sys_preflight_reason or 'rocprof-sys 不可用'})


def _normalize_rocm_values(option: str,
                           values: Optional[Sequence[str]]) -> List[str]:
    """Normalize a variadic ROCm option while preserving caller order."""
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        raise TypeError(f'{option} must be a sequence of values, not a scalar string')
    normalized: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            raise ValueError(f'{option} values must be non-empty strings')
        normalized.append(text)
    return normalized


def build_rocprof_sys_cmd(binary: str, out_dir: Path, payload: Sequence[str], *,
                          buffer_kb: int = 1048576,
                          backend: str = 'inprocess',
                          rocm_domains: Optional[Sequence[str]] = None,
                          rocm_events: Optional[Sequence[str]] = None) -> List[str]:
    """Build a deterministic rocprof-sys trace invocation.

    ``--output <path> <prefix>`` is the current rocprof-sys CLI.  The tool adds
    a run timestamp directory and PID suffix to its artifacts; discovery scans
    for those actual files instead of inventing a filename.
    """
    if not payload:
        raise ValueError('payload must not be empty')
    if int(buffer_kb) <= 0:
        raise ValueError('buffer_kb must be positive')
    domains = _normalize_rocm_values('--rocm-domains', rocm_domains)
    events = _normalize_rocm_values('--rocm-events', rocm_events)
    cmd = [
        str(binary),
        '--trace',
        '--use-rocm',
    ]
    if domains:
        cmd.extend(['--rocm-domains', *domains])
    if events:
        cmd.extend(['--rocm-events', *events])
    cmd.extend([
        '--output', str(out_dir), 'capture',
        '--perfetto-backend', backend,
        '--trace-buffer-size', str(int(buffer_kb)),
        '--perfetto-shmem-size-hint-kb', str(int(buffer_kb)),
        '--',
        *[str(part) for part in payload],
    ])
    return cmd


def discover_rocprof_sys_artifacts(out_dir: Path,
                                   since_epoch: Optional[float] = None) -> Dict[str, Any]:
    """Find rocprof-sys timestamp/PID-named artifacts below ``out_dir``."""
    root = Path(out_dir)
    trace_files: List[Path] = []
    metadata_files: List[Path] = []
    function_files: List[Path] = []
    if root.is_dir():
        for path in root.rglob('*'):
            if not path.is_file():
                continue
            if since_epoch is not None:
                try:
                    if path.stat().st_mtime < since_epoch - 2.0:
                        continue
                except OSError:
                    continue
            name = path.name.lower()
            if name.endswith(('.proto', '.pftrace')) or 'perfetto-trace' in name:
                trace_files.append(path)
            elif 'metadata' in name and name.endswith('.json'):
                metadata_files.append(path)
            elif 'functions' in name and name.endswith('.json'):
                function_files.append(path)

    def newest(paths: List[Path]) -> Optional[Path]:
        return max(paths, key=lambda item: item.stat().st_mtime, default=None)

    return {
        'root': str(root),
        'trace_files': [str(path) for path in sorted(trace_files)],
        'metadata_files': [str(path) for path in sorted(metadata_files)],
        'function_files': [str(path) for path in sorted(function_files)],
        'selected_trace': str(newest(trace_files)) if trace_files else None,
        'selected_metadata': str(newest(metadata_files)) if metadata_files else None,
    }


def _contains_needle(paths: Sequence[str], needle: str,
                     max_bytes: int = 64 * 1024 * 1024) -> Dict[str, Any]:
    if not needle:
        return {'checked': False, 'reason': '未提供 kernel needle'}
    raw = needle.encode('utf-8')
    checked = 0
    for item in paths:
        path = Path(item)
        try:
            with path.open('rb') as handle:
                while checked < max_bytes:
                    block = handle.read(min(1 << 20, max_bytes - checked))
                    if not block:
                        break
                    checked += len(block)
                    if raw in block:
                        return {'checked': True, 'observed': True,
                                'path': str(path), 'bytes_scanned': checked,
                                'method': 'best-effort-byte-search'}
        except OSError:
            continue
    return {'checked': True, 'observed': False, 'bytes_scanned': checked,
            'method': 'best-effort-byte-search',
            'limit_bytes': max_bytes}


def inspect_rocprof_sys_artifacts(discovery: Dict[str, Any],
                                  kernel_needle: str = '') -> Dict[str, Any]:
    """Validate trace/metadata existence without claiming a GPU run occurred."""
    result: Dict[str, Any] = {
        'collector': ROCPROF_SYS_FAMILY,
        'artifacts': discovery,
        'hard_failures': [],
        'warnings': [],
    }
    trace_path = discovery.get('selected_trace')
    metadata_path = discovery.get('selected_metadata')
    if not trace_path or not Path(trace_path).is_file():
        result['hard_failures'].append(
            'rocprof-sys Perfetto trace 未生成（找不到 .proto/.pftrace 产物）')
    else:
        size = Path(trace_path).stat().st_size
        result['trace'] = {'path': trace_path, 'bytes': size,
                           'format': 'perfetto-protobuf'}
        if size <= 0:
            result['hard_failures'].append('rocprof-sys Perfetto trace 为空')
    if not metadata_path or not Path(metadata_path).is_file():
        result['hard_failures'].append(
            'rocprof-sys metadata JSON 未生成，无法校验工具/版本上下文')
    else:
        try:
            metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
            result['metadata'] = {
                'path': metadata_path,
                'bytes': Path(metadata_path).stat().st_size,
                'root_present': isinstance(metadata, dict)
                and 'rocprofiler-systems' in metadata,
            }
            if not result['metadata']['root_present']:
                result['hard_failures'].append(
                    'rocprof-sys metadata JSON 缺 rocprofiler-systems 根')
        except (OSError, json.JSONDecodeError) as exc:
            result['hard_failures'].append(
                f'rocprof-sys metadata JSON 解析失败: {exc}')

    needle_paths = [path for path in (trace_path, metadata_path) if path]
    if kernel_needle:
        probe = _contains_needle(needle_paths, kernel_needle)
        result['kernel'] = {
            'needle': kernel_needle,
            'available': False,
            'structural_verification': False,
            **probe,
        }
        if not probe.get('observed'):
            result['warnings'].append(
                'kernel needle 未在 Perfetto proto/metadata 的明文扫描范围内出现；'
                '该工具链不做结构化 kernel_dispatch 校验，此结果不能单独判 PASS/FAIL。')
    else:
        result['kernel'] = {
            'needle': '',
            'available': False,
            'structural_verification': False,
            'reason': '未提供 kernel-name-contains；只校验 trace/metadata 产物',
        }
    result['valid'] = not result['hard_failures']
    return result
