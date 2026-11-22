"""Deterministic outer agent harness for WeaveProfiler (Atrex + KREX inspired).

Enforces hard programmatic guarantees outside probabilistic LLM/Skill guidance:
  1. PolicyGuard           : Anti-cheat & immutable run policy (overhead limit lock,
                             source immutability, forbidden sampling/MLIR checks).
  2. EpisodeSealer         : Immutable per-round SHA-256 sealing & fail-closed
                             receipt verification (`verify_seal`).
  3. IncumbentTracker      : Deterministic candidate ranking & supervised promotion
                             (never overwrites a better round with a regression).
  4. RoundStateMachine     : Explicit episode FSM with oscillation/stagnation
                             detection and deterministic loop-hoist escalation.
  5. FailedTimelineSummary : Compact structured hotspot attribution distilled from
                             failed-run `trace.perfetto` binaries for LLM feedback.
  6. RegionExclusivityLock : KREX-style region-granular per-GPU exclusivity lock
                             (off-region concurrency + in-region strict isolation)
                             and strict ABBA timing schedule / drift validation.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import hmac
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import statistics
import subprocess
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .contract import enclosing_loop, loop_spans, normalize_probe_spec
from .workspace import digest as file_sha256, read_json, write_json

if os.name == "nt":
    import msvcrt
else:
    import fcntl


SEAL_SCHEMA = "weaveprof.episode-seal/v1"
LEDGER_SCHEMA = "weaveprof.episode-ledger/v1"


class PolicyViolation(ValueError):
    """Raised when an agent proposal or runtime violates a hard harness policy."""

    def __init__(self, code: str, message: str, detail: Optional[Mapping[str, Any]] = None):
        self.code = code
        self.detail = dict(detail or {})
        super().__init__(f"[{code}] {message}")


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


@dataclass(frozen=True)
class HarnessPolicy:
    """Immutable deterministic policy enforced by the outer Supervisor."""

    max_overhead_limit: float = 0.10
    min_paired_samples: int = 3
    min_warmup: int = 1
    require_abba_order: bool = False
    allow_loop_sampling: bool = False
    allow_mlir: bool = False
    require_source_immutable: bool = True
    enforce_trace_processor: bool = False
    enable_region_lock: bool = True
    region_lock_timeout_s: float = 120.0
    command_timeout_s: float = 600.0
    escalation_on_stagnation: bool = True
    export_failed_timeline: bool = True
    max_timeline_hotspots: int = 6
    allow_single_stage_collapse: bool = False
    allow_partial_warps: bool = False
    max_delta_spill_bytes: int = 0
    max_delta_stack_bytes: int = 0
    max_delta_stl: int = 4
    max_delta_ldl: int = 4
    require_output_files: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


_FORBIDDEN_SAMPLING_RE = re.compile(
    r"(?:\b(?:FI_SAMPLE|WP_SAMPLE|--fi-sample|--sample-stride)\s*(?:=|\s)\s*[+]?(?:0x[0-9a-f]+|[2-9]|\d{2,})"
    r"|['\"](?:FI_SAMPLE|WP_SAMPLE)['\"]\s*\]\s*=\s*['\"]?[+]?(?:0x[0-9a-f]+|[2-9]|\d{2,}))",
    re.I,
)
_FORBIDDEN_MLIR_RE = re.compile(r"\b(?:mlir-opt|mlir-translate|--use-mlir)\b", re.I)
_EVIDENCE_TAMPER_RE = re.compile(
    r"(?:>|tee\s+|\b(?:cp|mv|rm|truncate|sed\s+-i|dd\s+of=)\b).*"
    r"(?:[\w.-]*seal\.json|verify\.json|feedback\.json|episode_ledger\.json|"
    r"ALL_GATES\.json|PREFLIGHT\.json|\.delivery_receipt\.json)"
)


def _parse_int_env(val: Any) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(s, 0)
    except ValueError:
        return None


class PolicyGuard:
    """Fail-closed policy checks separating the LLM proposer from Supervisor rules."""

    def __init__(self, policy: Optional[HarnessPolicy] = None):
        self.policy = policy or HarnessPolicy()

    def validate_session_start(
        self,
        source_path: Path,
        source_sha: str,
        limit: float,
        bench_cmd: Optional[Sequence[str]] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not math.isfinite(limit) or limit < 0:
            raise PolicyViolation("INVALID_LIMIT", f"Overhead limit must be finite and nonnegative, got {limit!r}")
        if limit > self.policy.max_overhead_limit + 1e-12:
            raise PolicyViolation(
                "LIMIT_RELAXATION_BLOCKED",
                f"Requested overhead limit {limit:.4f} exceeds hard policy ceiling {self.policy.max_overhead_limit:.4f}.",
                {"requested_limit": limit, "policy_ceiling": self.policy.max_overhead_limit},
            )
        if self.policy.require_source_immutable:
            live = file_sha256(source_path)
            if live != source_sha:
                raise PolicyViolation(
                    "SOURCE_MUTATED",
                    f"Target kernel source {source_path} changed unexpectedly before execution.",
                    {"expected_sha256": source_sha, "live_sha256": live},
                )
        active_env = env if env is not None else os.environ
        if not self.policy.allow_loop_sampling:
            for env_k in ("FI_SAMPLE", "WP_SAMPLE", "WEAVEPROF_SAMPLE"):
                parsed = _parse_int_env(active_env.get(env_k))
                if parsed is not None and parsed > 1:
                    raise PolicyViolation(
                        "LOOP_SAMPLING_FORBIDDEN",
                        f"Environment variable {env_k}={active_env.get(env_k)!r} violates full-audit policy.",
                        {"env_var": env_k, "value": active_env.get(env_k)},
                    )
        if bench_cmd:
            joined = " ".join(str(x) for x in bench_cmd)
            if not self.policy.allow_loop_sampling and _FORBIDDEN_SAMPLING_RE.search(joined):
                raise PolicyViolation(
                    "LOOP_SAMPLING_FORBIDDEN",
                    "Loop sampling flags (FI_SAMPLE/WP_SAMPLE > 1) are forbidden by full-audit policy.",
                    {"bench_cmd": joined[:300]},
                )
            if not self.policy.allow_mlir and _FORBIDDEN_MLIR_RE.search(joined):
                raise PolicyViolation(
                    "MLIR_FORBIDDEN",
                    "MLIR tooling is forbidden in the profiling harness.",
                    {"bench_cmd": joined[:300]},
                )
            if _EVIDENCE_TAMPER_RE.search(joined):
                raise PolicyViolation(
                    "EVIDENCE_TAMPER_BLOCKED",
                    "Benchmark command must not write directly to harness evidence/seal files.",
                    {"bench_cmd": joined[:300]},
                )

    def validate_source_immutable(self, source_path: Path, expected_sha: str) -> None:
        if not self.policy.require_source_immutable:
            return
        live = file_sha256(source_path)
        if live != expected_sha:
            raise PolicyViolation(
                "SOURCE_MUTATED",
                f"Target kernel source {source_path} was modified during profiling (expected {expected_sha[:12]}, got {live[:12]}).",
                {"expected_sha256": expected_sha, "live_sha256": live},
            )

    def validate_probe_spec(self, spec: Mapping[str, Any], source_text: str) -> List[Dict[str, Any]]:
        """Validate probe spec structurally and detect static policy violations."""
        issues: List[Dict[str, Any]] = []
        try:
            norm = normalize_probe_spec(spec, source_text)
        except Exception as exc:
            return [{"code": "INVALID_PROBE_SCHEMA", "severity": "error", "message": str(exc)}]

        # Check spin-loop placement statically (hard rule, not just SKILL.md advice)
        lines = source_text.splitlines()
        spin_spans = _detect_spin_loops(lines)
        for p in norm.get("probes") or []:
            b_ln = int(p["begin_line"])
            e_ln = int(p["end_line"])
            for s_beg, s_end, snip in spin_spans:
                if s_end > s_beg and ((s_beg < b_ln <= s_end) or (s_beg < e_ln < s_end)):
                    issues.append({
                        "code": "SPIN_LOOP_PROBE_POLICY",
                        "severity": "error",
                        "probe_id": p["id"],
                        "lines": [b_ln, e_ln],
                        "spin_loop_lines": [s_beg, s_end],
                        "message": (
                            f"Probe {p['id']!r} (L{b_ln}-L{e_ln}) sits inside polling spin-loop "
                            f"L{s_beg}-L{s_end} ({snip[:60]!r}); bracket the entire wait loop from outside."
                        ),
                    })
        return issues

    def validate_runtime_protocol(self, runtime: Mapping[str, Any], limit: float) -> Tuple[bool, Dict[str, Any], List[str]]:
        errors: List[str] = []
        warmup = int(runtime.get("warmup") or 0)
        base_ns = list(runtime.get("baseline_ns") or [])
        inst_ns = list(runtime.get("instrumented_ns") or [])
        if warmup < self.policy.min_warmup:
            errors.append(f"warmup={warmup} < policy min_warmup={self.policy.min_warmup}")
        if len(base_ns) < self.policy.min_paired_samples or len(base_ns) != len(inst_ns):
            errors.append(
                f"paired samples mismatch: baseline={len(base_ns)}, instrumented={len(inst_ns)} "
                f"(min={self.policy.min_paired_samples})"
            )
        proto = str(runtime.get("timing_protocol") or "").upper()
        if self.policy.require_abba_order and proto != "ABBA":
            errors.append(f"timing_protocol={proto!r}, policy requires 'ABBA'")
        abba = summarize_abba(base_ns, inst_ns, protocol=proto or "interleaved") if (base_ns and len(base_ns) == len(inst_ns)) else {}
        if limit > self.policy.max_overhead_limit + 1e-12:
            errors.append(f"limit={limit} exceeds policy ceiling {self.policy.max_overhead_limit}")
        return not errors, abba, errors


_SPIN_CUE_RE = re.compile(
    r"\b(try_wait|test_wait|mbarrier|atomic|ld\.acquire|__hip_atomic|volatile|ready|flag|phase|arrive|spin|poll)\b",
    re.I,
)


def _detect_spin_loops(lines: Sequence[str]) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    n = len(lines)
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "#")):
            continue
        if re.match(r"^(?:while|do)\b", stripped):
            window = "\n".join(lines[idx - 1:min(n, idx + 6)])
            if not _SPIN_CUE_RE.search(window):
                continue
            if stripped.startswith("while") and (stripped.endswith(";") or ("{" in stripped and "}" in stripped)):
                spans.append((idx, idx, stripped))
                continue
            if "{" in stripped or (idx < n and "{" in lines[idx]):
                depth = 0
                end_ln = idx
                for j in range(idx, min(n, idx + 40) + 1):
                    depth += lines[j - 1].count("{") - lines[j - 1].count("}")
                    end_ln = j
                    if depth <= 0 and j > idx:
                        break
                spans.append((idx, end_ln, stripped))
            elif stripped.endswith(":"):
                base_indent = len(line) - len(line.lstrip())
                end_ln = idx
                for j in range(idx + 1, min(n, idx + 40) + 1):
                    lj = lines[j - 1]
                    if lj.strip() and not lj.strip().startswith("#"):
                        if (len(lj) - len(lj.lstrip())) <= base_indent:
                            break
                        end_ln = j
                spans.append((idx, end_ln, stripped))
    return spans


# ---------------------------------------------------------------------------
# ABBA Timing Protocol (Atrex-style drift-cancelling interleaved comparison)
# ---------------------------------------------------------------------------

def abba_schedule(n_blocks: int) -> List[str]:
    """Generate strict ABBA launch blocks (`baseline, instrumented, instrumented, baseline`)."""
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1")
    seq: List[str] = []
    for _ in range(n_blocks):
        seq.extend(("baseline", "instrumented", "instrumented", "baseline"))
    return seq


def summarize_abba(
    baseline_ns: Sequence[float],
    instrumented_ns: Sequence[float],
    *,
    protocol: str = "ABBA",
) -> Dict[str, Any]:
    """Compute paired median slowdown, paired IQR/MAD stability, and half-split drift."""
    if not baseline_ns or len(baseline_ns) != len(instrumented_ns):
        return {}
    base = [float(x) for x in baseline_ns]
    inst = [float(x) for x in instrumented_ns]
    if any(not math.isfinite(v) or v <= 0 for v in (*base, *inst)):
        return {}
    med_b = statistics.median(base)
    med_i = statistics.median(inst)
    ratios = [i / b - 1.0 for b, i in zip(base, inst)]
    med_r = statistics.median(ratios)
    mad_r = statistics.median([abs(r - med_r) for r in ratios])
    half = max(1, len(base) // 2)
    first_half_b = statistics.median(base[:half])
    second_half_b = statistics.median(base[half:]) if len(base) > half else first_half_b
    drift_pct = 100.0 * abs(second_half_b - first_half_b) / max(1e-9, med_b)
    return {
        "protocol": protocol,
        "paired_samples": len(base),
        "baseline_median_ns": round(med_b, 3),
        "instrumented_median_ns": round(med_i, 3),
        "median_overhead_ratio": round(med_i / med_b - 1.0, 6),
        "paired_ratio_median": round(med_r, 6),
        "paired_ratio_mad": round(mad_r, 6),
        "baseline_half_drift_pct": round(drift_pct, 4),
    }


# ---------------------------------------------------------------------------
# KREX-Style Region-Granular GPU Exclusivity Lock
# ---------------------------------------------------------------------------

@dataclass
class RegionLease:
    """Telemetry recorded for one critical-region exclusive GPU lease."""

    gpu_id: str
    lock_path: str
    pid: int
    host: str
    label: str
    wait_ms: float
    hold_ms: float = 0.0
    acquired_at: float = 0.0
    released_at: float = 0.0
    cpu_cores_pinned: Optional[List[int]] = None
    provider: str = "local_file"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Pluggable KREX Lock Provider Registry (default: OS-level local flock/msvcrt).
# External orchestrators (e.g. Alibaba TRE KREX API) register a custom context-manager
# factory once at startup via `register_lock_provider("tre_krex", factory)` and select
# it via `lock_provider="tre_krex"` or env `WEAVEPROF_LOCK_PROVIDER=tre_krex`.
_LOCK_PROVIDERS: Dict[str, Any] = {}


def register_lock_provider(name: str, factory: Any) -> None:
    """Register a custom KREX GPU critical-region lock provider context-manager factory.

    The factory must accept `(gpu_id, *, timeout_s, label, cpu_cores)` and act as a
    context manager yielding either a `RegionLease`, a `dict`, or `None`.
    """
    key = str(name or "").strip().lower()
    if not key or key == "local_file":
        raise ValueError("Lock provider name must be non-empty and cannot override 'local_file'")
    if not callable(factory):
        raise TypeError("Lock provider factory must be callable")
    _LOCK_PROVIDERS[key] = factory


def unregister_lock_provider(name: str) -> None:
    """Remove a previously registered custom lock provider (useful in tests)."""
    _LOCK_PROVIDERS.pop(str(name or "").strip().lower(), None)


def get_lock_provider(name: Optional[str] = None) -> Tuple[str, Any]:
    """Resolve the active lock provider `(provider_name, factory_or_None)`."""
    selected = (name or os.environ.get("WEAVEPROF_LOCK_PROVIDER") or "local_file").strip().lower()
    if selected in ("", "local", "local_file"):
        return "local_file", None
    if selected not in _LOCK_PROVIDERS:
        raise ValueError(
            f"Unknown KREX lock provider {selected!r}; registered={sorted(['local_file', *_LOCK_PROVIDERS])}"
        )
    return selected, _LOCK_PROVIDERS[selected]


def list_lock_providers() -> List[str]:
    return sorted(["local_file", *_LOCK_PROVIDERS.keys()])


def _resolve_gpu_lock_id(gpu_id: str) -> str:
    if gpu_id and str(gpu_id).lower() != "auto":
        return re.sub(r"[^A-Za-z0-9_.-]", "_", str(gpu_id))
    for env_key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        val = os.environ.get(env_key)
        if val and val.strip():
            first = val.split(",")[0].strip()
            if first:
                return re.sub(r"[^A-Za-z0-9_.-]", "_", f"gpu_{first}")
    return "gpu_0"


def _default_lock_dir() -> Path:
    configured = os.environ.get("WEAVEPROF_LOCK_DIR")
    if configured:
        return Path(configured).resolve()
    return (Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "weaveprof" / "krex_locks").resolve()


def _try_lock_file(fd: int) -> bool:
    if os.name == "nt":
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    else:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False


def _unlock_file(fd: int) -> None:
    if os.name == "nt":
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass


@contextmanager
def gpu_critical_region(
    gpu_id: str = "auto",
    *,
    timeout_s: float = 120.0,
    lock_dir: Optional[Path] = None,
    label: str = "measure",
    cpu_cores: Optional[Sequence[int]] = None,
    poll_interval_s: float = 0.02,
    lock_provider: Optional[str] = None,
):
    """KREX-inspired region-granular per-GPU exclusivity context manager.

    Off-region work (LLM calls, workspace cloning, CPU compilation, SASS LCS,
    Perfetto export, PNG rendering) runs concurrently across agent processes.
    Only the marked timing/profiling critical region acquires this per-GPU lock
    (and optionally pins CPU affinity on Linux) to preserve measurement fidelity.
    Supports pluggable external lock providers via `register_lock_provider()`.
    """
    resolved_gpu = _resolve_gpu_lock_id(gpu_id)
    prov_name, prov_factory = get_lock_provider(lock_provider)
    if prov_factory is not None:
        t0 = time.perf_counter()
        with prov_factory(
            resolved_gpu,
            timeout_s=float(timeout_s),
            label=label,
            cpu_cores=list(cpu_cores) if cpu_cores else None,
        ) as ext_lease:
            t_acq = time.perf_counter()
            if isinstance(ext_lease, RegionLease):
                lease = ext_lease
                lease.provider = prov_name
            elif isinstance(ext_lease, Mapping):
                lease = RegionLease(
                    gpu_id=str(ext_lease.get("gpu_id") or resolved_gpu),
                    lock_path=str(ext_lease.get("lock_path") or f"provider://{prov_name}/{resolved_gpu}"),
                    pid=int(ext_lease.get("pid") or os.getpid()),
                    host=str(ext_lease.get("host") or socket.gethostname()),
                    label=str(ext_lease.get("label") or label),
                    wait_ms=float(ext_lease.get("wait_ms") if ext_lease.get("wait_ms") is not None else round((t_acq - t0) * 1000.0, 3)),
                    acquired_at=float(ext_lease.get("acquired_at") or round(time.time(), 6)),
                    cpu_cores_pinned=list(ext_lease["cpu_cores_pinned"]) if ext_lease.get("cpu_cores_pinned") else None,
                    provider=prov_name,
                )
            else:
                lease = RegionLease(
                    gpu_id=resolved_gpu,
                    lock_path=f"provider://{prov_name}/{resolved_gpu}",
                    pid=os.getpid(),
                    host=socket.gethostname(),
                    label=label,
                    wait_ms=round((t_acq - t0) * 1000.0, 3),
                    acquired_at=round(time.time(), 6),
                    provider=prov_name,
                )
            try:
                yield lease
            finally:
                lease.hold_ms = round((time.perf_counter() - t_acq) * 1000.0, 3)
                lease.released_at = round(time.time(), 6)
        return

    base_dir = Path(lock_dir).resolve() if lock_dir else _default_lock_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    lock_path = base_dir / f"{resolved_gpu}.lock"
    meta_path = base_dir / f"{resolved_gpu}.owner.json"

    t_wait0 = time.perf_counter()
    deadline = t_wait0 + max(0.05, float(timeout_s))
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o666)
    # Ensure at least 1 byte exists for Windows msvcrt byte-range locking.
    if os.fstat(fd).st_size == 0:
        os.write(fd, b"0")

    acquired = False
    try:
        while True:
            if _try_lock_file(fd):
                acquired = True
                break
            if time.perf_counter() >= deadline:
                owner = ""
                try:
                    owner = meta_path.read_text(encoding="utf-8").strip()
                except OSError:
                    pass
                raise TimeoutError(
                    f"Timed out after {timeout_s:.1f}s waiting for KREX GPU critical region lock "
                    f"on {resolved_gpu} ({lock_path}); current owner={owner[:200]}"
                )
            time.sleep(poll_interval_s)

        t_acq = time.perf_counter()
        wait_ms = round((t_acq - t_wait0) * 1000.0, 3)
        prev_affinity = None
        pinned: Optional[List[int]] = None
        if cpu_cores and hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
            try:
                prev_affinity = os.sched_getaffinity(0)
                target = set(int(c) for c in cpu_cores)
                os.sched_setaffinity(0, target)
                pinned = sorted(target)
            except OSError:
                prev_affinity = None

        lease = RegionLease(
            gpu_id=resolved_gpu,
            lock_path=str(lock_path),
            pid=os.getpid(),
            host=socket.gethostname(),
            label=label,
            wait_ms=wait_ms,
            acquired_at=round(time.time(), 6),
            cpu_cores_pinned=pinned,
        )
        try:
            write_json(meta_path, lease.to_dict())
        except OSError:
            pass

        try:
            yield lease
        finally:
            t_rel = time.perf_counter()
            lease.hold_ms = round((t_rel - t_acq) * 1000.0, 3)
            lease.released_at = round(time.time(), 6)
            if prev_affinity is not None and hasattr(os, "sched_setaffinity"):
                try:
                    os.sched_setaffinity(0, prev_affinity)
                except OSError:
                    pass
            try:
                if meta_path.is_file():
                    meta_path.unlink()
            except OSError:
                pass
    finally:
        if acquired:
            _unlock_file(fd)
        os.close(fd)


# ---------------------------------------------------------------------------
# Failed-Run Timeline Summary (distilled from failed `trace.perfetto`)
# ---------------------------------------------------------------------------

def summarize_failed_timeline(
    trace: Optional[Mapping[str, Any]],
    summary: Optional[Mapping[str, Any]],
    spec: Mapping[str, Any],
    source_text: str,
    runtime: Optional[Mapping[str, Any]] = None,
    *,
    failed_trace_path: Optional[Path] = None,
    max_hotspots: int = 6,
) -> Dict[str, Any]:
    """Distill a compact, actionable bottleneck summary from a failed-run timeline.

    Allows the Stage Identifier to inspect exact per-stage fire counts, average/total
    slice durations, and enclosing loop spans without stuffing raw multi-MB traces
    into the LLM context window.
    """
    runtime = runtime or {}
    norm = normalize_probe_spec(spec, source_text)
    spans = loop_spans(source_text, norm["language"])
    probe_by_id = {p["id"]: p for p in norm.get("probes") or []}
    fires_map = dict((summary or {}).get("fires") or {})

    # Aggregate slice durations per probe/stage name from decoded trace events
    durs_by_stage: Dict[str, List[float]] = defaultdict(list)
    roles_by_stage: Dict[str, set] = defaultdict(set)
    for ev in (trace or {}).get("traceEvents") or []:
        if ev.get("ph") != "X":
            continue
        args = ev.get("args") or {}
        if args.get("kind") == "residency":
            continue
        nm = str(args.get("probe_id") or ev.get("name") or "")
        if not nm:
            continue
        dur_us = float(ev.get("dur") or 0.0)
        if math.isfinite(dur_us) and dur_us >= 0:
            durs_by_stage[nm].append(dur_us)
        role_val = args.get("role")
        if role_val:
            roles_by_stage[nm].add(str(role_val))

    stages: List[Dict[str, Any]] = []
    for pid, p in probe_by_id.items():
        durs = durs_by_stage.get(pid) or durs_by_stage.get(p.get("semantic_id", ""), [])
        raw_fire = fires_map.get(pid, len(durs))
        try:
            f_cnt = int(raw_fire) if raw_fire is not None else len(durs)
        except (TypeError, ValueError):
            f_cnt = len(durs)
        avg_us = round(sum(durs) / len(durs), 4) if durs else None
        total_us = round(sum(durs), 4) if durs else None
        max_us = round(max(durs), 4) if durs else None
        loop = enclosing_loop(p, spans)
        stages.append({
            "id": pid,
            "role": p["begin"]["role"],
            "end_role": p["end"]["role"],
            "lines": [int(p["begin"]["line"]), int(p["end"]["line"])],
            "fires": f_cnt,
            "observed_slices": len(durs),
            "avg_us": avg_us,
            "total_us": total_us,
            "max_us": max_us,
            "in_loop": loop is not None,
            "enclosing_loop": loop,
            "cross_warp": p["begin"]["role"] != p["end"]["role"],
        })

    stages.sort(key=lambda s: (s["fires"], s["total_us"] or 0.0), reverse=True)

    def _safe_median_us(vals):
        nums = []
        for v in vals or ():
            try:
                f = float(v)
                if math.isfinite(f) and f > 0:
                    nums.append(f)
            except (TypeError, ValueError):
                continue
        return round(statistics.median(nums) / 1000.0, 3) if nums else None

    base_us = _safe_median_us(runtime.get("baseline_ns"))
    inst_us = _safe_median_us(runtime.get("instrumented_ns"))
    ovh_pct = round(100.0 * (inst_us / base_us - 1.0), 3) if (base_us and inst_us) else None

    hotspots: List[str] = []
    for st in stages[:max_hotspots]:
        if st["fires"] <= 0:
            continue
        avg_str = f"{st['avg_us']:.4f}us" if st["avg_us"] is not None else "n/a"
        loop_str = f" inside loop L{st['enclosing_loop'][0]}-L{st['enclosing_loop'][1]}" if st["enclosing_loop"] else ""
        hotspots.append(
            f"{st['role']} \"{st['id']}\" (L{st['lines'][0]}-L{st['lines'][1]}): "
            f"{avg_str} x {st['fires']}{loop_str}"
        )

    out: Dict[str, Any] = {
        "kernel_baseline_us": base_us,
        "kernel_instrumented_us": inst_us,
        "overhead_pct": ovh_pct,
        "stages": stages,
        "hotspots": hotspots,
    }
    if failed_trace_path is not None and Path(failed_trace_path).is_file():
        fp = Path(failed_trace_path).resolve()
        out["failed_trace_path"] = str(fp)
        out["failed_trace_sha256"] = file_sha256(fp)
        out["failed_trace_bytes"] = fp.stat().st_size
    return out


# ---------------------------------------------------------------------------
# Automatic Perfetto trace_processor_shell Verification (Hardens SKILL.md SQL)
# ---------------------------------------------------------------------------

def verify_trace_processor_sql(trace_path: Path, *, timeout_s: float = 60.0) -> Dict[str, Any]:
    """Run canonical Perfetto SQL import checks automatically when trace_processor_shell exists."""
    tp = os.environ.get("TRACE_PROCESSOR_SHELL") or shutil.which("trace_processor_shell")
    if not tp:
        return {"available": False, "ok": None, "reason": "trace_processor_shell not found in PATH"}
    tp_path = Path(tp)
    if not trace_path.is_file():
        return {"available": True, "ok": False, "errors": [f"Trace file missing: {trace_path}"]}

    sql_summary = (
        "SELECT "
        "(SELECT COUNT(*) FROM slice) AS slices, "
        "(SELECT COUNT(*) FROM stats WHERE value > 0 AND (severity = 'error' OR name IN ("
        "'json_parser_failure','json_tokenizer_failure','slice_spill_overlapping_complete_event',"
        "'slice_drop_overlapping_complete_event','trace_sorter_negative_timestamp_dropped'))) AS import_errors, "
        "(SELECT COUNT(*) FROM track WHERE name GLOB 'wave *' OR name GLOB 'warp *' OR name GLOB 'cta * / warp *') AS wave_tracks;"
    )
    try:
        proc = subprocess.run(
            [str(tp_path), "-Q", sql_summary, str(trace_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "ok": False, "errors": [f"trace_processor_shell failed: {exc}"]}

    if proc.returncode != 0:
        return {
            "available": True,
            "ok": False,
            "returncode": proc.returncode,
            "errors": [(proc.stderr or proc.stdout or "non-zero exit")[-400:]],
        }
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return {"available": True, "ok": False, "errors": [f"Unexpected SQL output: {proc.stdout[:200]!r}"]}
    last_row = lines[-1]
    if "," in last_row:
        cols = next(csv.reader(io.StringIO(last_row)))
    else:
        cols = re.split(r"\s+", last_row)
    try:
        slices, import_errors, wave_tracks = int(cols[0]), int(cols[1]), int(cols[2])
    except (IndexError, ValueError):
        return {"available": True, "ok": False, "errors": [f"Failed to parse SQL row: {last_row!r}"]}
    errors = []
    if slices <= 0:
        errors.append(f"slices={slices} (expected > 0)")
    if import_errors > 0:
        errors.append(f"import_errors={import_errors} (expected 0)")
    if wave_tracks <= 0:
        errors.append(f"wave_tracks={wave_tracks} (expected > 0)")
    return {
        "available": True,
        "ok": not errors,
        "binary": str(tp_path),
        "slices": slices,
        "import_errors": import_errors,
        "wave_tracks": wave_tracks,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Atrex-Style Immutable Episode Sealing (HMAC-signed + hash-chained) & Verifier
# ---------------------------------------------------------------------------

def _supervisor_key_path() -> Path:
    configured = os.environ.get("WEAVEPROF_SUPERVISOR_KEY_FILE")
    if configured:
        return Path(configured).resolve()
    return (Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "weaveprof" / ".supervisor.key").resolve()


def _get_supervisor_hmac_key(*, create: bool = True) -> Optional[bytes]:
    env_key = os.environ.get("WEAVEPROF_SUPERVISOR_HMAC_KEY")
    if env_key:
        return env_key.encode("utf-8")
    kpath = _supervisor_key_path()
    if kpath.is_file():
        try:
            raw = kpath.read_bytes().strip()
            if raw:
                return raw
        except OSError:
            return None
    if not create:
        return None
    try:
        kpath.parent.mkdir(parents=True, exist_ok=True)
        token = secrets.token_hex(32).encode("ascii")
        kpath.write_bytes(token)
        try:
            os.chmod(kpath, 0o600)
        except OSError:
            pass
        return token
    except OSError:
        return None


def _compute_seal_hmac(seal_sha256: str, key: bytes) -> str:
    return hmac.new(key, seal_sha256.encode("ascii"), hashlib.sha256).hexdigest()


def _bind_file(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    p = Path(path).resolve()
    if not p.is_file():
        return None
    return {"path": str(p), "sha256": file_sha256(p), "bytes": p.stat().st_size}


class EpisodeSealer:
    """Seals every round's exact source, probe, measurement inputs, diagnostics, and trace artifacts."""

    @staticmethod
    def seal_round(
        *,
        round_dir: Path,
        round_number: int,
        source_path: Path,
        source_sha256: str,
        probe_path: Optional[Path],
        instrumented_path: Optional[Path],
        names_path: Optional[Path],
        feedback_path: Path,
        verify_path: Optional[Path],
        trace_path: Optional[Path],
        failed_trace_path: Optional[Path],
        result: Mapping[str, Any],
        policy: HarnessPolicy,
        limit: float,
        bench_cmd: Optional[Sequence[str]] = None,
        fsm_state: str = "VERIFY_AND_SEAL",
        escalation: Optional[str] = None,
        region_lock: Optional[Mapping[str, Any]] = None,
        prev_seal_sha256: Optional[str] = None,
        extra_artifacts: Optional[Mapping[str, Optional[Path]]] = None,
        out_seal_path: Optional[Path] = None,
        mode: Optional[str] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        checks = dict(result.get("checks") or {})
        metrics = dict(result.get("metrics") or {})
        artifacts: Dict[str, Any] = {
            "source": {"path": str(Path(source_path).resolve()), "sha256": source_sha256},
            "probe": _bind_file(probe_path),
            "instrumented_source": _bind_file(instrumented_path),
            "names": _bind_file(names_path),
            "feedback": _bind_file(feedback_path),
            "verify": _bind_file(verify_path),
            "trace": _bind_file(trace_path),
            "failed_trace": _bind_file(failed_trace_path),
        }
        for k, v in (extra_artifacts or {}).items():
            if k not in artifacts:
                artifacts[k] = _bind_file(v)
        payload: Dict[str, Any] = {
            "schema": SEAL_SCHEMA,
            "sealed_at": round(time.time(), 6),
            "round": int(round_number),
            "prev_seal_sha256": prev_seal_sha256,
            "status": str(result.get("status") or "FAIL"),
            "fsm_state": fsm_state,
            "escalation": escalation,
            "limit": float(limit),
            "policy": policy.to_dict(),
            "policy_digest": policy.digest(),
            "bench_cmd_sha256": canonical_sha256(list(bench_cmd)) if bench_cmd else None,
            "checks": {
                "assembly": bool(checks.get("assembly")),
                "output": bool(checks.get("output")),
                "events": bool(checks.get("events", checks.get("event"))),
                "overhead": bool(checks.get("overhead")),
            },
            "failed": list(result.get("failed") or []),
            "overhead_ratio": metrics.get("overhead_ratio"),
            "region_lock": dict(region_lock) if region_lock else metrics.get("region_lock"),
            "trace_processor": metrics.get("trace_processor"),
            "artifacts": artifacts,
        }
        if mode is not None:
            payload["mode"] = str(mode)
        payload["seal_sha256"] = canonical_sha256(payload)
        key = _get_supervisor_hmac_key(create=True)
        if key is None:
            raise RuntimeError("Supervisor HMAC key unavailable; refusing to write unsigned seal")
        payload["seal_hmac"] = _compute_seal_hmac(payload["seal_sha256"], key)
        target = Path(out_seal_path) if out_seal_path else (Path(round_dir) / "seal.json")
        write_json(target, payload)
        return target, payload


def verify_seal(
    seal_path: Path | str,
    *,
    require_pass: bool = True,
    expected_source_sha: Optional[str] = None,
    expected_policy_digest: Optional[str] = None,
    max_overhead_limit: Optional[float] = None,
    require_hmac: bool = True,
) -> Dict[str, Any]:
    """Fail-closed verifier for a sealed episode receipt (`*.seal.json`). Never raises."""
    errors: List[str] = []
    doc: Optional[Dict[str, Any]] = None
    try:
        sp = Path(seal_path).resolve()
        if not sp.is_file():
            return {"ok": False, "errors": [f"Seal file missing: {sp}"], "seal": None}
        doc = read_json(sp)
        if not isinstance(doc, dict):
            return {"ok": False, "errors": ["Seal JSON root must be an object"], "seal": None}

        if doc.get("schema") != SEAL_SCHEMA:
            errors.append(f"Unexpected seal schema {doc.get('schema')!r}; expected {SEAL_SCHEMA!r}")

        claimed_seal_sha = doc.get("seal_sha256")
        body = {k: v for k, v in doc.items() if k not in ("seal_sha256", "seal_hmac")}
        computed_seal_sha = canonical_sha256(body)
        if not isinstance(claimed_seal_sha, str) or claimed_seal_sha != computed_seal_sha:
            errors.append(f"Seal self-hash tampered (claimed={claimed_seal_sha}, computed={computed_seal_sha})")

        claimed_hmac = doc.get("seal_hmac")
        key = _get_supervisor_hmac_key(create=False)
        if require_hmac:
            if key is None:
                errors.append("Supervisor HMAC key missing; cannot verify seal signature (fail-closed)")
            elif not isinstance(claimed_hmac, str):
                errors.append("Seal missing required supervisor HMAC signature (possible forgery)")
            else:
                expected_hmac = _compute_seal_hmac(computed_seal_sha, key)
                if not hmac.compare_digest(claimed_hmac, expected_hmac):
                    errors.append("Seal supervisor HMAC mismatch (possible forgery)")
        elif isinstance(claimed_hmac, str) and key is not None:
            expected_hmac = _compute_seal_hmac(computed_seal_sha, key)
            if not hmac.compare_digest(claimed_hmac, expected_hmac):
                errors.append("Seal supervisor HMAC mismatch")

        policy_doc = doc.get("policy")
        if not isinstance(policy_doc, dict):
            errors.append("Seal policy block missing or invalid")
            policy_doc = {}
        computed_policy_digest = canonical_sha256(policy_doc)
        if doc.get("policy_digest") != computed_policy_digest:
            errors.append("Policy digest mismatch inside seal")
        if expected_policy_digest and doc.get("policy_digest") != expected_policy_digest:
            errors.append(
                f"Policy digest mismatch vs expected supervisor policy "
                f"(seal={doc.get('policy_digest')}, expected={expected_policy_digest})"
            )

        ceiling = max_overhead_limit if max_overhead_limit is not None else policy_doc.get("max_overhead_limit")
        try:
            lim_val = float(doc["limit"])
            if not math.isfinite(lim_val) or lim_val < 0:
                errors.append(f"Invalid seal limit: {doc.get('limit')!r}")
            elif ceiling is not None and lim_val > float(ceiling) + 1e-12:
                errors.append(f"Seal limit {lim_val} exceeds policy ceiling {ceiling}")
        except (KeyError, TypeError, ValueError):
            errors.append(f"Seal limit missing or non-numeric: {doc.get('limit')!r}")

        artifacts = doc.get("artifacts")
        if not isinstance(artifacts, dict):
            errors.append("Seal artifacts block missing or invalid")
            artifacts = {}
        src_art = artifacts.get("source")
        if not isinstance(src_art, dict):
            errors.append("Seal missing bound source artifact")
            src_art = {}
        if expected_source_sha and src_art.get("sha256") != expected_source_sha:
            errors.append(f"Source SHA mismatch: seal={src_art.get('sha256')} vs expected={expected_source_sha}")

        for name, entry in artifacts.items():
            if entry is None:
                continue
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
                errors.append(f"Malformed artifact binding for {name!r}")
                continue
            p = Path(entry["path"])
            if not p.is_file():
                errors.append(f"Bound artifact {name!r} missing on disk: {p}")
                continue
            live_sha = file_sha256(p)
            if live_sha != entry["sha256"]:
                errors.append(
                    f"Bound artifact {name!r} hash mismatch: {p} "
                    f"(seal={entry['sha256'][:12]} != live={live_sha[:12]})"
                )

        if require_pass:
            if doc.get("status") != "PASS":
                errors.append(f"Seal status is {doc.get('status')!r}, required 'PASS'")
            checks = doc.get("checks")
            if not isinstance(checks, dict):
                errors.append("Seal checks block missing or invalid")
                checks = {}
            for gate in ("assembly", "output", "events", "overhead"):
                if checks.get(gate) is not True:
                    errors.append(f"Required check {gate!r} is not True in seal")
            for req_art in ("source", "feedback", "trace"):
                if not artifacts.get(req_art):
                    errors.append(f"PASS seal must bind required artifact {req_art!r}")
            ovh = doc.get("overhead_ratio")
            try:
                if ovh is None:
                    raise ValueError("missing overhead_ratio")
                ovh_f = float(ovh)
                if not math.isfinite(ovh_f):
                    raise ValueError(f"non-finite overhead_ratio={ovh!r}")
                if ceiling is not None and ovh_f > float(ceiling) + 1e-12:
                    errors.append(f"Measured overhead_ratio={ovh_f} exceeds ceiling={ceiling}")
            except (TypeError, ValueError) as exc:
                errors.append(f"PASS seal requires finite overhead_ratio: {exc}")
            if policy_doc.get("enable_region_lock") and not isinstance(doc.get("region_lock"), dict):
                errors.append("PASS seal requires KREX region_lock telemetry when enable_region_lock=True")
            tp = doc.get("trace_processor") or {}
            if policy_doc.get("enforce_trace_processor") and (not isinstance(tp, dict) or tp.get("ok") is not True):
                errors.append(f"trace_processor SQL verification did not pass: {tp}")
    except Exception as exc:
        errors.append(f"Fail-closed seal verification exception: {type(exc).__name__}: {exc}")

    return {"ok": not errors, "errors": errors, "seal": doc}


# ---------------------------------------------------------------------------
# Atrex-Style Incumbent Tracker & Deterministic Round State Machine (FSM)
# ---------------------------------------------------------------------------

def probe_fingerprint(spec: Mapping[str, Any], source_text: str) -> str:
    norm = normalize_probe_spec(spec, source_text)
    core = {
        "roles": {k: str(v) for k, v in sorted((norm.get("roles") or {}).items())},
        "probes": sorted(
            (
                (
                    p["id"],
                    p["begin"]["role"],
                    int(p["begin"]["line"]),
                    p["end"]["role"],
                    int(p["end"]["line"]),
                    str(p.get("slot") or ""),
                )
                for p in norm.get("probes") or []
            )
        ),
    }
    return canonical_sha256(core)


@dataclass
class SealedRoundRecord:
    round_number: int
    probe_fp: str
    status: str
    checks: Dict[str, bool]
    failed: List[str]
    overhead_ratio: Optional[float]
    seal_path: str
    seal_sha256: str
    round_dir: str
    escalation: Optional[str] = None

    def rank_key(self) -> Tuple[int, int, int, int, int, float]:
        passed_all = 1 if self.status == "PASS" else 0
        out_ok = 1 if self.checks.get("output") else 0
        ev_ok = 1 if self.checks.get("events") else 0
        asm_ok = 1 if self.checks.get("assembly") else 0
        n_pass = sum(1 for v in self.checks.values() if v)
        ovh = self.overhead_ratio if (self.overhead_ratio is not None and math.isfinite(self.overhead_ratio)) else 1e9
        return (passed_all, out_ok, ev_ok, asm_ok, n_pass, -ovh)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IncumbentTracker:
    """Tracks all sealed rounds and promotes only the verified best candidate."""

    def __init__(self) -> None:
        self.records: List[SealedRoundRecord] = []
        self.incumbent: Optional[SealedRoundRecord] = None

    def record(self, rec: SealedRoundRecord) -> bool:
        self.records.append(rec)
        if self.incumbent is None or rec.rank_key() > self.incumbent.rank_key():
            self.incumbent = rec
            return True
        return False

    @property
    def promoted(self) -> Optional[SealedRoundRecord]:
        if self.incumbent and self.incumbent.status == "PASS":
            return self.incumbent
        return None


_ALLOWED_FSM_TRANSITIONS: Dict[str, frozenset] = {
    "INIT": frozenset(("POLICY_CHECK", "HALT_BLOCKED")),
    "POLICY_CHECK": frozenset(("CRITICAL_REGION_MEASURE", "REFINE_LLM", "ESCALATE_DETERMINISTIC", "HALT_BLOCKED")),
    "CRITICAL_REGION_MEASURE": frozenset(("PROMOTED", "REFINE_LLM", "ESCALATE_DETERMINISTIC", "HALT_BLOCKED")),
    "REFINE_LLM": frozenset(("POLICY_CHECK", "ESCALATE_DETERMINISTIC", "HALT_BLOCKED")),
    "ESCALATE_DETERMINISTIC": frozenset(("POLICY_CHECK", "HALT_BLOCKED")),
    "PROMOTED": frozenset(),
    "HALT_BLOCKED": frozenset(),
}


@dataclass
class RoundStateMachine:
    """Deterministic Supervisor state machine preventing blind LLM loops/oscillation."""

    max_rounds: int
    escalation_on_stagnation: bool = True
    state: str = "INIT"
    seen_fingerprints: Dict[str, int] = field(default_factory=dict)
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    _last_failed_key: Optional[Tuple[str, ...]] = None
    _last_overhead: Optional[float] = None
    _stagnation_count: int = 0

    def transition(self, next_state: str, **meta: Any) -> None:
        allowed = _ALLOWED_FSM_TRANSITIONS.get(self.state, frozenset())
        if next_state not in allowed:
            raise PolicyViolation(
                "INVALID_FSM_TRANSITION",
                f"Illegal Supervisor FSM transition: {self.state} -> {next_state}",
                {"from": self.state, "to": next_state},
            )
        self.transitions.append({
            "ts": round(time.time(), 6),
            "from": self.state,
            "to": next_state,
            **meta,
        })
        self.state = next_state

    def decide_after_round(
        self,
        round_number: int,
        probe_fp: str,
        result: Mapping[str, Any],
    ) -> Tuple[str, str]:
        """Return `(action, reason)` where action is one of:
        `PROMOTE`, `REFINE_LLM`, `ESCALATE_DETERMINISTIC`, `HALT_BLOCKED`.
        """
        if result.get("status") == "PASS":
            self.transition("PROMOTED", round=round_number, probe_fp=probe_fp[:12])
            return "PROMOTE", "all_gates_passed"

        if round_number >= self.max_rounds:
            self.transition("HALT_BLOCKED", round=round_number, reason="max_rounds_exhausted")
            return "HALT_BLOCKED", "max_rounds_exhausted"

        if result.get("export_error"):
            self.transition("HALT_BLOCKED", round=round_number, reason="export_error")
            return "HALT_BLOCKED", "export_error"

        errs = (result.get("metrics") or {}).get("errors") or {}
        if "measurement" in errs and not result.get("fires"):
            self.transition("HALT_BLOCKED", round=round_number, reason="measurement_crash")
            return "HALT_BLOCKED", "measurement_crash"

        oscillating = probe_fp in self.seen_fingerprints
        self.seen_fingerprints[probe_fp] = round_number

        failed_key = tuple(sorted(result.get("failed") or ()))
        ovh = (result.get("metrics") or {}).get("overhead_ratio")
        stagnating = False
        if self._last_failed_key == failed_key:
            if ovh is None or self._last_overhead is None or ovh >= self._last_overhead - 0.005:
                self._stagnation_count += 1
                stagnating = self._stagnation_count >= 1
            else:
                self._stagnation_count = 0
        else:
            self._stagnation_count = 0
        self._last_failed_key = failed_key
        self._last_overhead = ovh

        has_hoist_target = bool(result.get("enclosing_loop")) and "pairing" not in failed_key and "events" not in failed_key
        final_retry = (round_number >= self.max_rounds - 1) and (self.max_rounds >= 3)
        if self.escalation_on_stagnation and (oscillating or stagnating or final_retry) and has_hoist_target:
            reason = (
                "oscillation_detected" if oscillating else
                "stagnation_detected" if stagnating else
                "final_round_guaranteed_hoist"
            )
            self.transition("ESCALATE_DETERMINISTIC", round=round_number, reason=reason)
            return "ESCALATE_DETERMINISTIC", reason

        self.transition("REFINE_LLM", round=round_number, failed=list(failed_key))
        return "REFINE_LLM", "failed_gates_refinement"


# ---------------------------------------------------------------------------
# Production-Silicon Fidelity Gates (Pillar 1 — Anti-Distortion & Completeness)
# ---------------------------------------------------------------------------

def verify_compiler_spill_gate(
    l0_stats: Mapping[str, Any],
    inst_stats: Mapping[str, Any],
    *,
    max_delta_spill_bytes: int = 0,
    max_delta_stack_bytes: int = 0,
    max_delta_stl: int = 4,
    max_delta_ldl: int = 4,
    has_setmaxnreg: bool = False,
    smem_free_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Catch register-spill and SASS STL/LDL explosions masked by unchanged `Used N regs`.

    Real-world motivation (B300 sm_103a):
      - `v82` K2 (512 thr, 128r cap): `Used 128 regs` unchanged while `spill_stores: 0 -> 220 B`,
        `STL: 27 -> 97`, slowdown `+51.32%` (remediated by `WP_CTX_SMEM` -> 0 B spill, `+7.36%`).
      - `bs128` decode (`setmaxnreg.dec 48`): `Used 72 regs` unchanged across Baseline/Pre/Post while
        `spill_stores: 16 -> 760 B`, `STL: 2 -> 134`, slowdown `+28.94%` (remediated by
        compiler-in-the-loop `{pre, post}` placement search -> `+3.61%`).
    """
    def _ival(d: Mapping[str, Any], *keys: str) -> Optional[int]:
        for k in keys:
            if d.get(k) is not None:
                try:
                    return int(d[k])
                except (TypeError, ValueError):
                    return None
        return None

    l0_regs = _ival(l0_stats, "regs", "registers")
    in_regs = _ival(inst_stats, "regs", "registers")
    l0_spill_st = _ival(l0_stats, "spill_stores", "spill_store_bytes", "spill_bytes")
    in_spill_st = _ival(inst_stats, "spill_stores", "spill_store_bytes", "spill_bytes")
    l0_spill_ld = _ival(l0_stats, "spill_loads", "spill_load_bytes")
    in_spill_ld = _ival(inst_stats, "spill_loads", "spill_load_bytes")
    l0_stack = _ival(l0_stats, "stack_bytes", "stack_frame")
    in_stack = _ival(inst_stats, "stack_bytes", "stack_frame")
    l0_stl = _ival(l0_stats, "stl", "stl_count", "sass_stl")
    in_stl = _ival(inst_stats, "stl", "stl_count", "sass_stl")
    l0_ldl = _ival(l0_stats, "ldl", "ldl_count", "sass_ldl")
    in_ldl = _ival(inst_stats, "ldl", "ldl_count", "sass_ldl")

    delta_spill_st = (in_spill_st - l0_spill_st) if (l0_spill_st is not None and in_spill_st is not None) else None
    delta_spill_ld = (in_spill_ld - l0_spill_ld) if (l0_spill_ld is not None and in_spill_ld is not None) else None
    delta_stack = (in_stack - l0_stack) if (l0_stack is not None and in_stack is not None) else None
    delta_stl = (in_stl - l0_stl) if (l0_stl is not None and in_stl is not None) else None
    delta_ldl = (in_ldl - l0_ldl) if (l0_ldl is not None and in_ldl is not None) else None

    violations: List[Dict[str, Any]] = []
    if delta_spill_st is not None and delta_spill_st > max_delta_spill_bytes:
        violations.append({
            "code": "SPILL_STORE_INCREASE",
            "delta": delta_spill_st,
            "limit": max_delta_spill_bytes,
            "message": f"Spill stores increased by +{delta_spill_st} B ({l0_spill_st} -> {in_spill_st} B).",
        })
    if delta_spill_ld is not None and delta_spill_ld > max_delta_spill_bytes:
        violations.append({
            "code": "SPILL_LOAD_INCREASE",
            "delta": delta_spill_ld,
            "limit": max_delta_spill_bytes,
            "message": f"Spill loads increased by +{delta_spill_ld} B ({l0_spill_ld} -> {in_spill_ld} B).",
        })
    if delta_stack is not None and delta_stack > max_delta_stack_bytes:
        violations.append({
            "code": "STACK_FRAME_INCREASE",
            "delta": delta_stack,
            "limit": max_delta_stack_bytes,
            "message": f"Stack frame increased by +{delta_stack} B ({l0_stack} -> {in_stack} B).",
        })
    if delta_stl is not None and delta_stl > max_delta_stl:
        violations.append({
            "code": "SASS_STL_EXPLOSION",
            "delta": delta_stl,
            "limit": max_delta_stl,
            "message": f"SASS STL (local store) instructions jumped by +{delta_stl} ({l0_stl} -> {in_stl}).",
        })
    if delta_ldl is not None and delta_ldl > max_delta_ldl:
        violations.append({
            "code": "SASS_LDL_EXPLOSION",
            "delta": delta_ldl,
            "limit": max_delta_ldl,
            "message": f"SASS LDL (local load) instructions jumped by +{delta_ldl} ({l0_ldl} -> {in_ldl}).",
        })

    reg_blind_spot = bool(
        violations and l0_regs is not None and in_regs is not None and in_regs <= l0_regs
    )
    remediation: List[str] = []
    if violations:
        if has_setmaxnreg:
            remediation.append("search_setmaxnreg_placement:{pre_setmaxnreg,post_setmaxnreg}")
        if smem_free_bytes is None or smem_free_bytes >= 16:
            remediation.append("enable_WP_CTX_SMEM (-DWP_CTX_SMEM, 16B SMEM + __noinline__ cold path)")
        else:
            remediation.append("coarsen_probes_or_hoist_outside_hot_loop")

    return {
        "ok": not violations,
        "register_count_blind_spot": reg_blind_spot,
        "violations": violations,
        "remediation": remediation,
        "metrics": {
            "l0_regs": l0_regs,
            "inst_regs": in_regs,
            "delta_spill_stores_bytes": delta_spill_st,
            "delta_spill_loads_bytes": delta_spill_ld,
            "delta_stack_bytes": delta_stack,
            "delta_sass_stl": delta_stl,
            "delta_sass_ldl": delta_ldl,
        },
    }


COMPILER_FEEDBACK_SCHEMA = "weaveprof.compiler-feedback/v1"

_FAST_ARM: Dict[str, Any] = {"mode": "WP_CTX_REG", "defines": ["-DWP_CTX_REG"], "smem_bytes_per_slot": 0}
_SMEM_ARM: Dict[str, Any] = {"mode": "WP_CTX_SMEM", "defines": ["-DWP_CTX_SMEM"], "smem_bytes_per_slot": 16}
_NOINLINE_ARM: Dict[str, Any] = {"mode": "WP_CTX_REG_NOINLINE", "defines": ["-DWP_CTX_REG_NOINLINE"], "smem_bytes_per_slot": 0}


def _first_int(d: Mapping[str, Any], *keys: str) -> Optional[int]:
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
    return None


def run_compiler_feedback_selection(
    compile_fn: Any,
    source_text: str,
    *,
    baseline_metrics: Mapping[str, Any],
    arch: Optional[str] = None,
    gpu_name: Optional[str] = None,
    block_threads: Optional[int] = None,
    l0_smem_bytes: int = 0,
    smem_limit_bytes: Optional[int] = None,
    l0_regs: Optional[int] = None,
    try_fast_first: bool = True,
    max_delta_stl: int = 4,
    max_delta_ldl: int = 4,
) -> Dict[str, Any]:
    """D2 compiler-in-the-loop recorder-context selection driven by real compiler telemetry.

    `compile_fn(arm)` is invoked once per (arm, placement) attempt with
    `{"mode", "defines", "placement", "smem_bytes_per_slot", "stage"}` and must return the
    compiled telemetry `{"regs", "spill_bytes"|"spill_stores", "stack_bytes",
    "stl_count"|"stl", "ldl_count"|"ldl"}` (optionally `overhead_pct`, `build_ok`).
    Exceptions are recorded as `build_ok=False` attempts (fail closed, never raised).

    Tiering:
      1. Fast arm first (`-DWP_CTX_REG`, 1 live reg, 0 B SMEM) at every placement in the
         search space (`{pre_setmaxnreg, post_setmaxnreg}` when `setmaxnreg` is present,
         else `default_entry`). With `try_fast_first=False` the planner's resource-arithmetic
         `context_mode` is tried first instead.
      2. ONLY if no first-stage attempt passes `verify_compiler_spill_gate`, compile the
         fallback arms: `-DWP_CTX_SMEM` (when >= 16 B SMEM free) then
         `-DWP_CTX_REG_NOINLINE`.
      3. Rank every compiled attempt with `select_compiler_feedback_arm`; the winner is the
         best-ranked attempt that passes the spill/STL gate (or the best-ranked overall,
         flagged `gate_ok=False`, if none does).
    """
    from .causal import plan_context_and_search_space, select_compiler_feedback_arm

    if not callable(compile_fn):
        raise TypeError("compile_fn must be callable: compile_fn(arm) -> compiler telemetry dict")
    base = dict(baseline_metrics or {})
    if l0_regs is None:
        l0_regs = _first_int(base, "regs", "registers")

    plan = plan_context_and_search_space(
        source_text or "",
        arch=arch,
        gpu_name=gpu_name,
        block_threads=block_threads,
        l0_regs=l0_regs,
        l0_smem_bytes=l0_smem_bytes,
        smem_limit_bytes=smem_limit_bytes,
    )
    smem_free = int(plan["smem_free_bytes"])
    has_setmaxnreg = bool(plan["has_setmaxnreg"])
    placements = list(plan["placement_search_space"])

    if try_fast_first:
        first_arm = dict(_FAST_ARM)
        fallback_arms = ([dict(_SMEM_ARM)] if smem_free >= 16 else []) + [dict(_NOINLINE_ARM)]
    else:
        first_arm = {
            "mode": plan["context_mode"],
            "defines": list(plan["compile_defines"]),
            "smem_bytes_per_slot": 16 if plan["context_mode"] == "WP_CTX_SMEM" else 0,
        }
        fallback_arms = [dict(a) for a in plan["fallback_arms"]]
        if smem_free < 16:
            fallback_arms = [a for a in fallback_arms if int(a.get("smem_bytes_per_slot") or 0) == 0]

    attempts: List[Dict[str, Any]] = []
    gate_reports: List[Dict[str, Any]] = []

    def _run_stage(arms: Sequence[Mapping[str, Any]], stage: str) -> None:
        for arm in arms:
            for placement in placements:
                req = {
                    "mode": str(arm["mode"]),
                    "defines": list(arm.get("defines") or []),
                    "placement": placement,
                    "smem_bytes_per_slot": int(arm.get("smem_bytes_per_slot") or 0),
                    "stage": stage,
                }
                error: Optional[str] = None
                try:
                    raw = compile_fn(dict(req))
                    metrics = dict(raw or {})
                except Exception as exc:  # fail closed: a crashing build is a failed arm
                    metrics = {}
                    error = f"{type(exc).__name__}: {exc}"
                build_ok = bool(metrics.get("build_ok", error is None)) and error is None
                if build_ok:
                    report = verify_compiler_spill_gate(
                        base,
                        metrics,
                        max_delta_stl=max_delta_stl,
                        max_delta_ldl=max_delta_ldl,
                        has_setmaxnreg=has_setmaxnreg,
                        smem_free_bytes=smem_free,
                    )
                else:
                    report = {
                        "ok": False,
                        "register_count_blind_spot": False,
                        "violations": [{
                            "code": "BUILD_FAILED",
                            "message": error or str(metrics.get("error") or "compile_fn reported build_ok=False"),
                        }],
                        "remediation": [],
                        "metrics": {},
                    }
                attempt = {
                    **req,
                    "build_ok": build_ok,
                    "error": error,
                    "gate_ok": bool(report["ok"]),
                    "regs": _first_int(metrics, "regs", "registers"),
                    "spill_bytes": _first_int(metrics, "spill_bytes", "spill_stores", "spill_store_bytes"),
                    "stack_bytes": _first_int(metrics, "stack_bytes", "stack_frame"),
                    "stl_count": _first_int(metrics, "stl_count", "stl", "sass_stl"),
                    "ldl_count": _first_int(metrics, "ldl_count", "ldl", "sass_ldl"),
                    "overhead_pct": metrics.get("overhead_pct"),
                }
                attempts.append(attempt)
                gate_reports.append({"mode": req["mode"], "placement": placement, "stage": stage, "report": report})

    _run_stage([first_arm], "fast" if try_fast_first else "planned")
    fallback_triggered = not any(a["gate_ok"] for a in attempts)
    if fallback_triggered and fallback_arms:
        _run_stage(fallback_arms, "fallback")

    baseline_spill = _first_int(base, "spill_bytes", "spill_stores", "spill_store_bytes") or 0
    baseline_stl = _first_int(base, "stl_count", "stl", "sass_stl") or 0
    selection = select_compiler_feedback_arm(
        attempts,
        baseline_regs=_first_int(base, "regs", "registers"),
        baseline_spill_bytes=baseline_spill,
        baseline_stl=baseline_stl,
    )
    gate_by_key = {(a["mode"], a["placement"]): a for a in attempts}
    ranked: List[Dict[str, Any]] = []
    for entry in selection["ranked"]:
        src = gate_by_key.get((entry["mode"], entry["placement"]), {})
        ranked.append({**entry, "gate_ok": bool(src.get("gate_ok")), "stage": src.get("stage")})

    winner = next((e for e in ranked if e["gate_ok"]), None)
    if winner is None and ranked:
        winner = ranked[0]
    if winner is not None:
        winner = dict(winner)
        winner["compile_defines"] = list(winner["defines"])

    status = "CLEAN" if (winner and winner["gate_ok"]) else "NO_CLEAN_ARM"
    return {
        "schema": COMPILER_FEEDBACK_SCHEMA,
        "status": status,
        "winner": winner,
        "winner_is_fast_arm": bool(winner and winner["mode"] == first_arm["mode"]),
        "fallback_triggered": fallback_triggered,
        "attempts": attempts,
        "compile_calls": len(attempts),
        "spill_gate_reports": gate_reports,
        "ranked": ranked,
        "placement_search_space": placements,
        "has_setmaxnreg": has_setmaxnreg,
        "smem_free_bytes": smem_free,
        "reg_headroom": plan["reg_headroom"],
        "context_plan": plan,
    }


def verify_semantic_body_and_stubs(
    stage_durations_ns: Mapping[str, Sequence[float]],
    *,
    min_median_ns: float = 128.0,
    sass_body_insns: Optional[Mapping[str, int]] = None,
    min_fires_for_stub: int = 4,
) -> Dict[str, Any]:
    """Reject empty/0-instruction stub probes masquerading as valid spans.

    Real-world motivation (H20 GLM5.3 CuTeDSL):
      `TASK_EPILOG_R2S` wrapped 0 executable instructions (commented body) yet logged
      160 events with non-zero median duration `64 ns` (pure recorder self-latency),
      falsely passing naive `dur > 0` checks.
    """
    stubs: List[Dict[str, Any]] = []
    valid_stages: List[str] = []
    sass_map = dict(sass_body_insns or {})

    for stage, raw_durs in sorted((stage_durations_ns or {}).items()):
        durs = [float(x) for x in raw_durs if x is not None and math.isfinite(float(x)) and float(x) >= 0.0]
        if not durs:
            continue
        med_ns = statistics.median(durs)
        p90_ns = sorted(durs)[min(len(durs) - 1, int(round(0.90 * (len(durs) - 1))))]
        body_ins = sass_map.get(stage)
        reasons: List[str] = []
        if body_ins is not None and int(body_ins) <= 0:
            reasons.append("zero_non_probe_sass_instructions")
        if med_ns <= float(min_median_ns) and len(durs) >= int(min_fires_for_stub) and p90_ns <= float(min_median_ns) * 2.0:
            reasons.append(f"recorder_self_latency_stub (median={med_ns:.1f}ns <= {min_median_ns:.1f}ns, fires={len(durs)})")
        if reasons:
            stubs.append({
                "stage": stage,
                "fires": len(durs),
                "median_ns": round(med_ns, 2),
                "p90_ns": round(p90_ns, 2),
                "sass_body_insns": body_ins,
                "reasons": reasons,
            })
        else:
            valid_stages.append(stage)

    return {
        "ok": not stubs,
        "stubs": stubs,
        "valid_stages": valid_stages,
        "stub_count": len(stubs),
    }


_COLLAPSE_WAIT_RE = re.compile(r"(?:wait|barrier|grid_sync|sync_wait|stall|idle|spin|poll)", re.I)
_MEM_SEM_RE = re.compile(r"(?:load|dma|tma|cp_async|copy|prefetch|gather|store|epilogue|scatter|pull|dispatch_pull)", re.I)
_COMP_SEM_RE = re.compile(r"(?:mma|wgmma|umma|gemm|mfma|compute|softmax|attention|conv|norm|reduce|reduction|moe)", re.I)
_IDLE_ROLE_RE = re.compile(r"(?:idle|unused|noop|empty|disabled)", re.I)


def verify_stage_granularity(
    trace: Mapping[str, Any],
    *,
    max_single_stage_share: float = 0.95,
    min_active_us: float = 20.0,
    max_collapsed_spans_per_lane: float = 1.05,
) -> Dict[str, Any]:
    """Fail closed when intra-loop structure is hidden behind 1 or 2-4 coarse outer macro-blocks.

    Catches both:
      1) `SINGLE_STAGE_LOOP_COLLAPSE`: a single non-wait stage absorbs >= 95% of active time in
         <= 1.05 spans/lane (per_lane_us >= 20 us).
      2) `COARSE_PHASE_ONLY_COLLAPSE`: a multi-phase kernel is sliced into only a handful of
         coarse outer blocks (e.g. `CSA Attn 25% + Router 5% + MoE 70%`, <= 1.5 spans/lane each)
         without decomposing memory (`load`/`store`) vs compute (`mma`/`compute`) stages or
         leaving >= 250 us macro-blocks un-decomposed at warp granularity (`gran == 0`).
    """
    stage_us: Dict[str, float] = {}
    stage_n: Dict[str, int] = {}
    stage_lanes: Dict[str, set] = {}
    lane_intervals: Dict[Any, List[Tuple[float, float, str]]] = defaultdict(list)
    all_active_lanes: set = set()
    has_mem_sem = False
    has_comp_sem = False
    other = trace.get("otherData") if isinstance(trace.get("otherData"), Mapping) else {}
    gran = int(other.get("gran") if other.get("gran") is not None else 1)
    for ev in (trace.get("traceEvents") or []):
        if ev.get("ph") != "X" or ev.get("cat") in ("nsys_kernel", "info"):
            continue
        args = ev.get("args") or {}
        if args.get("kind") == "residency":
            continue
        if "warp" in args and other.get("gran") is None:
            gran = 0
        dur = float(ev.get("dur") or 0.0)
        if dur <= 0.0 or not math.isfinite(dur):
            continue
        ts = float(ev.get("ts") or 0.0)
        name = str(ev.get("name") or "stage")
        sem = str(args.get("semantic_id") or ev.get("cat") or name)
        if _COLLAPSE_WAIT_RE.search(name) or _COLLAPSE_WAIT_RE.search(sem):
            continue
        if _MEM_SEM_RE.search(sem) or _MEM_SEM_RE.search(name):
            has_mem_sem = True
        if _COMP_SEM_RE.search(sem) or _COMP_SEM_RE.search(name):
            has_comp_sem = True
        key = str(args.get("original_label") or name)
        lane = args.get("wave_slot") if args.get("wave_slot") is not None else (ev.get("pid"), ev.get("tid"))
        all_active_lanes.add(lane)
        stage_us[key] = stage_us.get(key, 0.0) + dur
        stage_n[key] = stage_n.get(key, 0) + 1
        stage_lanes.setdefault(key, set()).add(lane)
        lane_intervals[lane].append((ts, ts + dur, key))

    total_us = sum(stage_us.values())
    if total_us <= 0.0 or not stage_us:
        return {
            "ok": True, "code": None, "distinct_active_stages": 0,
            "dominant_stage": None, "dominant_share_pct": 0.0, "spans_per_lane": 0.0,
        }
    # Compute true union active time per lane and detect enclosing macro-wrappers
    # (e.g. an outer `attention`/`mma` wrapper covering >=85% of lane time while enclosing `load`/`store`).
    union_total_us = 0.0
    enclosed_counts: Dict[str, int] = defaultdict(int)
    enclosed_children: Dict[str, set] = defaultdict(set)
    for ivs in lane_intervals.values():
        ivs_sorted = sorted(ivs, key=lambda x: (x[0], -x[1]))
        cur_s, cur_e = ivs_sorted[0][0], ivs_sorted[0][1]
        for s0, e0, _k in ivs_sorted[1:]:
            if s0 <= cur_e:
                if e0 > cur_e:
                    cur_e = e0
            else:
                union_total_us += max(0.0, cur_e - cur_s)
                cur_s, cur_e = s0, e0
        union_total_us += max(0.0, cur_e - cur_s)
        # Check if any long outer span strictly encloses spans from other distinct stages.
        if len(ivs_sorted) >= 2 and len(ivs_sorted) <= 64:
            for idx_a, (sa, ea, ka) in enumerate(ivs_sorted):
                for idx_b, (sb, eb, kb) in enumerate(ivs_sorted):
                    if idx_a == idx_b or ka == kb:
                        continue
                    if sa <= sb + 1e-6 and eb <= ea + 1e-6 and (ea - sa) > (eb - sb) * 1.05:
                        enclosed_counts[ka] += 1
                        enclosed_children[ka].add(kb)

    dom_stage = max(stage_us, key=lambda k: stage_us[k])
    dom_us = stage_us[dom_stage]
    dom_share = dom_us / total_us
    dom_union_share = dom_us / max(union_total_us, 1e-9)
    dom_lanes = max(1, len(stage_lanes.get(dom_stage) or ()))
    spans_per_lane = stage_n[dom_stage] / float(dom_lanes)
    per_lane_us = dom_us / float(dom_lanes)
    total_lanes = max(1, len(all_active_lanes))
    overall_spans_per_lane = sum(stage_n.values()) / float(total_lanes)
    overall_per_lane_us = union_total_us / float(total_lanes)

    nested_wrappers = [
        k for k, kids in enclosed_children.items()
        if kids
        and (stage_n[k] / float(max(1, len(stage_lanes.get(k) or ())))) <= 1.5
        and (stage_us[k] / float(max(1, len(stage_lanes.get(k) or ())))) >= float(min_active_us)
        and (stage_us[k] / max(union_total_us, 1e-9)) >= 0.85
    ]
    nested_wrapper_collapsed = bool(nested_wrappers)
    single_collapsed = bool(
        not nested_wrapper_collapsed
        and per_lane_us >= float(min_active_us)
        and (dom_share >= float(max_single_stage_share) or dom_union_share >= float(max_single_stage_share))
        and spans_per_lane <= float(max_collapsed_spans_per_lane)
    )
    multi_coarse_no_mem = bool(
        overall_per_lane_us >= 60.0
        and overall_spans_per_lane <= 8.0
        and spans_per_lane <= 1.5
        and per_lane_us >= 50.0
        and not (has_mem_sem and has_comp_sem)
    )
    multi_coarse_long_warp = bool(
        gran == 0
        and overall_per_lane_us >= 450.0
        and dom_share >= 0.40
        and per_lane_us >= 250.0
        and spans_per_lane <= 1.5
        and overall_spans_per_lane <= 6.0
    )
    coarse_collapsed = (
        (not nested_wrapper_collapsed)
        and (not single_collapsed)
        and (multi_coarse_no_mem or multi_coarse_long_warp)
    )
    ok = not (nested_wrapper_collapsed or single_collapsed or coarse_collapsed)
    code = (
        "NESTED_WRAPPER_COLLAPSE" if nested_wrapper_collapsed
        else ("SINGLE_STAGE_LOOP_COLLAPSE" if single_collapsed
              else ("COARSE_PHASE_ONLY_COLLAPSE" if coarse_collapsed else None))
    )
    if nested_wrapper_collapsed:
        w0 = max(nested_wrappers, key=lambda k: stage_us[k])
        kids = sorted(enclosed_children[w0])[:4]
        w_pct = 100.0 * stage_us[w0] / max(union_total_us, 1e-9)
        w_spl = stage_n[w0] / float(max(1, len(stage_lanes.get(w0) or ())))
        msg = (
            f"Nested outer macro-wrapper detected: stage '{w0}' covers {w_pct:.1f}% of union active time in "
            f"{w_spl:.2f} spans/lane while enclosing inner stage(s) {kids}. Do not wrap the entire kernel/device "
            "call in an outer macro-probe; place disjoint sibling per-tile stages (load, mma_qk, softmax, mma_pv, "
            "epilogue/store) instead."
        )
    elif single_collapsed:
        msg = (
            f"Stage '{dom_stage}' absorbs {100.0 * dom_share:.1f}% of active time in {spans_per_lane:.2f} "
            f"spans/lane ({per_lane_us:.1f} us/lane): main loop collapsed into a single coarse bar; "
            "preserve per-tile stages (load/mma/softmax/epilogue) inside the loop."
        )
    elif coarse_collapsed:
        msg = (
            f"Coarse outer-phase collapse detected: {len(stage_us)} stage(s) average only "
            f"{overall_spans_per_lane:.2f} total spans/lane ({overall_per_lane_us:.1f} us/lane; "
            f"dominant '{dom_stage}' = {100.0 * dom_share:.1f}%, {per_lane_us:.1f} us/lane, "
            f"{spans_per_lane:.2f} spans/lane, has_mem={has_mem_sem}, has_comp={has_comp_sem}). "
            "Instrument per-tile/per-task load, mma/compute, and epilogue/store stages inside the loop "
            "instead of 2-4 outer macro-phase blocks."
        )
    else:
        msg = ""
    return {
        "ok": ok,
        "code": code,
        "distinct_active_stages": len(stage_us),
        "dominant_stage": dom_stage,
        "dominant_share_pct": round(100.0 * dom_share, 2),
        "spans_per_lane": round(spans_per_lane, 3),
        "overall_spans_per_lane": round(overall_spans_per_lane, 3),
        "per_lane_us": round(per_lane_us, 3),
        "has_memory_stage": has_mem_sem,
        "has_compute_stage": has_comp_sem,
        "message": msg,
    }


def verify_warp_completeness(
    summary: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    expected_warps: Optional[int] = None,
    min_hollow_lifetime_us: float = 50.0,
    min_warp_coverage_ratio: float = 0.05,
) -> Dict[str, Any]:
    """Fail closed when warp-granularity profiling drops warps OR leaves active warps hollow.

    1) `WARP_COVERAGE_INCOMPLETE`: every warp `0..wpc-1` must be recorded at `gran == 0`.
    2) `HOLLOW_WARP_LANE`: if a warp stays resident for `>= min_hollow_lifetime_us` (i.e. NOT a
       genuine early-exit warp that returns in `< 50 us`, nor an explicitly named `idle` role)
       yet logs `<= 1.05` spans/CTA covering `< 5%` of its resident lifetime (e.g. `WP_OPEN` +
       1 tiny dummy dot + `WP_CLOSE` leaving >95% blank space), fail closed.
    """
    gran = int(summary.get("gran") or 0)
    wpc = int(expected_warps or summary.get("wpc") or 1)
    warps_seen: set = set()
    warp_lifetime_us: Dict[int, List[float]] = {}
    warp_span_us: Dict[int, float] = {}
    warp_span_count: Dict[int, int] = {}
    warp_lane_count: Dict[int, int] = {}
    warp_roles: Dict[int, set] = {}

    lb = summary.get("lane_bounds_ns") or {}
    if isinstance(lb, Mapping) and wpc > 0:
        for slot_k, bounds in lb.items():
            try:
                w = int(slot_k) % wpc
                warps_seen.add(w)
                warp_lane_count[w] = warp_lane_count.get(w, 0) + 1
                if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                    life_us = max(0.0, (float(bounds[1]) - float(bounds[0])) / 1000.0)
                    warp_lifetime_us.setdefault(w, []).append(life_us)
            except (ValueError, TypeError):
                pass

    for ev in (trace.get("traceEvents") or []):
        ph = ev.get("ph")
        if ph == "X" and ev.get("cat") not in ("nsys_kernel", "info"):
            args = ev.get("args") or {}
            if args.get("kind") == "residency":
                if "warp" in args:
                    w = int(args["warp"])
                    warps_seen.add(w)
                    dur_u = max(0.0, float(ev.get("dur") or 0.0))
                    if dur_u > 0.0 and w not in warp_lifetime_us:
                        warp_lifetime_us.setdefault(w, []).append(dur_u)
                continue
            if "warp" in args:
                w = int(args["warp"])
                warps_seen.add(w)
                dur_u = max(0.0, float(ev.get("dur") or 0.0))
                warp_span_us[w] = warp_span_us.get(w, 0.0) + dur_u
                warp_span_count[w] = warp_span_count.get(w, 0) + 1
                rname = str(args.get("role") or "")
                if rname:
                    warp_roles.setdefault(w, set()).add(rname)
        elif ph == "M" and ev.get("name") == "thread_name":
            tname = str((ev.get("args") or {}).get("name") or "")
            m = re.search(r"\b(?:w|warp\s*)(\d+)\b", tname, re.I)
            if m:
                w = int(m.group(1))
                warp_roles.setdefault(w, set()).add(tname)

    if gran == 0 and wpc > 1:
        expected = set(range(wpc))
        missing = sorted(expected - warps_seen)
        if missing:
            return {
                "ok": False,
                "code": "WARP_COVERAGE_INCOMPLETE",
                "expected_warps": wpc,
                "observed_warps": sorted(warps_seen),
                "missing_warps": missing,
                "hollow_warps": [],
                "message": (
                    f"Only {len(warps_seen)}/{wpc} warps/CTA recorded (observed={sorted(warps_seen)}, "
                    f"missing={missing}); record ALL warps 0..{wpc - 1} (do not filter to 1 warp per warpgroup)."
                ),
            }
        hollow_warps = []
        for w in sorted(expected):
            lives = warp_lifetime_us.get(w) or []
            if not lives:
                continue
            med_life = sorted(lives)[len(lives) // 2]
            if med_life < float(min_hollow_lifetime_us):
                continue  # Legitimate early-exit warp (e.g. FA3 W01..W03 exiting in 0.54 us)
            roles_w = " ".join(sorted(warp_roles.get(w) or ()))
            if roles_w and _IDLE_ROLE_RE.search(roles_w):
                continue
            n_lanes_w = max(1, warp_lane_count.get(w) or len(lives))
            spans_per_cta = warp_span_count.get(w, 0) / float(n_lanes_w)
            cov_ratio = (warp_span_us.get(w, 0.0) / float(n_lanes_w)) / max(med_life, 1e-6)
            if spans_per_cta <= 1.05 and cov_ratio < float(min_warp_coverage_ratio):
                hollow_warps.append({
                    "warp": w,
                    "median_lifetime_us": round(med_life, 2),
                    "spans_per_cta": round(spans_per_cta, 3),
                    "coverage_pct": round(100.0 * cov_ratio, 2),
                })
        if hollow_warps:
            hw_ids = [h["warp"] for h in hollow_warps]
            return {
                "ok": False,
                "code": "HOLLOW_WARP_LANE",
                "expected_warps": wpc,
                "observed_warps": sorted(warps_seen),
                "missing_warps": [],
                "hollow_warps": hollow_warps,
                "message": (
                    f"Hollow resident warp lanes detected on warps {hw_ids}: {hollow_warps}. "
                    "Long-lived warps (>= 50 us) cannot just emit WP_OPEN/WP_CLOSE or a single tiny stub dot "
                    "leaving >95% blank space; instrument their actual producer/DMA/compute loop or cross-warp slot handoffs."
                ),
            }
    return {
        "ok": True,
        "code": None,
        "expected_warps": wpc,
        "observed_warps": sorted(warps_seen),
        "missing_warps": [],
        "hollow_warps": [],
        "message": "",
    }


def verify_sw_hw_coverage(
    sw_span_us: float,
    hw_kernel_us: float,
    *,
    in_window_samples: Optional[int] = None,
    expected_freq_hz: float = 200000.0,
    max_span_delta_ratio: float = 0.05,
    min_pmu_density: float = 0.90,
) -> Dict[str, Any]:
    """Enforce lossless SW-to-HW time coverage and in-window PMU sampling density.

    Real-world motivations:
      1. CuTeDSL `prof_record` unconditional offset increment created `tag==0` holes that
         silently truncated break-on-zero decoders (`168 us / 741 us`, `77.3%` missing,
         while `overflow == 0`!).
      2. `nsys --gpu-metrics-frequency=200000` without `--capture-range=cudaProfilerApi`
         overflows ring buffers during Python startup, leaving `0` in-window PMU samples.
    """
    violations: List[Dict[str, Any]] = []
    sw = float(sw_span_us or 0.0)
    hw = float(hw_kernel_us or 0.0)

    span_delta_ratio: Optional[float] = None
    if hw <= 0.0 or not math.isfinite(hw):
        violations.append({"code": "INVALID_HW_KERNEL_DURATION", "message": f"Invalid hw_kernel_us={hw_kernel_us!r}"})
    elif sw <= 0.0 or not math.isfinite(sw):
        violations.append({"code": "EMPTY_SW_SPAN", "message": f"Empty sw_span_us={sw_span_us!r}"})
    else:
        span_delta_ratio = abs(sw - hw) / hw
        if span_delta_ratio > float(max_span_delta_ratio):
            violations.append({
                "code": "SW_HW_SPAN_TRUNCATION",
                "sw_span_us": round(sw, 3),
                "hw_kernel_us": round(hw, 3),
                "delta_ratio": round(span_delta_ratio, 4),
                "limit": float(max_span_delta_ratio),
                "message": (
                    f"SW timeline span ({sw:.2f} us) diverges from HW kernel duration ({hw:.2f} us) "
                    f"by {span_delta_ratio * 100.0:.2f}% (> {max_span_delta_ratio * 100.0:.1f}%); "
                    "check for tag==0 holes, missing WP_CLOSE, or early CTA truncation."
                ),
            })

    pmu_density: Optional[float] = None
    expected_samples: Optional[float] = None
    if in_window_samples is not None and hw > 0.0 and expected_freq_hz > 0.0:
        expected_samples = max(1.0, hw * 1e-6 * float(expected_freq_hz))
        pmu_density = float(in_window_samples) / expected_samples
        if int(in_window_samples) <= 0 or (expected_samples >= 5.0 and pmu_density < float(min_pmu_density)):
            violations.append({
                "code": "PMU_RING_BUFFER_DROPOUT",
                "in_window_samples": int(in_window_samples),
                "expected_samples": round(expected_samples, 1),
                "pmu_density": round(pmu_density, 4),
                "limit": float(min_pmu_density),
                "message": (
                    f"In-window HW PMU density {pmu_density * 100.0:.1f}% ({in_window_samples}/{expected_samples:.0f} "
                    f"samples @ {expected_freq_hz/1e3:.0f} kHz) < {min_pmu_density * 100.0:.0f}%; "
                    "wrap steady-state launch with cudaProfilerStart/Stop and --capture-range=cudaProfilerApi."
                ),
            })

    return {
        "ok": not violations,
        "sw_span_us": round(sw, 3),
        "hw_kernel_us": round(hw, 3),
        "span_delta_ratio": round(span_delta_ratio, 5) if span_delta_ratio is not None else None,
        "in_window_samples": in_window_samples,
        "expected_samples": round(expected_samples, 1) if expected_samples is not None else None,
        "pmu_density": round(pmu_density, 4) if pmu_density is not None else None,
        "violations": violations,
    }
