#!/usr/bin/env python3
"""Golden replay harness for ``weaveprof profile`` closed-loop episodes.

Each real fixture under ``tests/golden/cases/wpprofile_<name>/`` follows the
layout:

    case.json                          CLI flags + capture provenance
    kernel/<entry_source>              uninstrumented kernel source
    kernel/probe.json                  input probe specification
    recorded/baseline_source.sha256    expected SHA-256 of baseline source
    recorded/instrumented_source.sha256 expected SHA-256 of injected source
    recorded/runtime.json              recorded ABBA timing & resource metrics
    recorded/l0.sass                   recorded baseline disassembly
    recorded/inst.sass                 recorded instrumented disassembly
    recorded/baseline.bin              recorded baseline output tensor bytes
    recorded/instrumented.bin          recorded instrumented output tensor bytes
    recorded/capture.fi02.bin.xz       compressed WPLITE02/FILITE02 software dump
    recorded/capture.sqlite.xz         compressed trimmed Nsight Systems SQLite
    expected.json                      normalized expected replay observation

Planned H20 capture branches (wired into ``EXPECTED_REAL_CASES`` once recorded):
    - ``wpprofile_h20_handoff_round1_fail``   (initial inner-loop probe: ``assembly`` + ``overhead`` FAIL)
    - ``wpprofile_h20_handoff_collapse_fail`` (hoisted probe at K=8192: ``SINGLE_STAGE_LOOP_COLLAPSE`` ``event``/``events`` FAIL)
    - ``wpprofile_h20_handoff_round2_pass``   (hoisted probe at K=4096 with H20 ``GPU_METRICS``: PASS)

All ``FAIL`` and ``PASS`` branches are replayed through the identical
``weaveprof profile`` CLI entrypoint (default ``HarnessPolicy``), unified
normalization, and ``_first_diff`` comparison.

Regenerate ``expected.json`` with ``WEAVEPROF_WPPROFILE_REPLAY_UPDATE=1``.

Run:  python -B -m unittest tests/golden/test_wpprofile_replay.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from collections import Counter
import contextlib
import gzip
import hashlib
import io
import ipaddress
import json
import lzma
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASES = HERE / "cases"
REPLAY_BENCH = HERE / "wpprofile_replay_bench.py"

# Active real H20 capture cases checked into tests/golden/cases/.
EXPECTED_REAL_CASES: tuple[str, ...] = (
    "wpprofile_h20_handoff_round1_fail",
    "wpprofile_h20_handoff_collapse_fail",
    "wpprofile_h20_handoff_no_window_samples",
    "wpprofile_h20_handoff_round2_pass",
)

UPDATE_ENV = "WEAVEPROF_WPPROFILE_REPLAY_UPDATE"
UPDATE = os.environ.get(UPDATE_ENV) == "1"
_HMAC_KEY = "weaveprof-wpprofile-replay-test-key"

# Keys whose values legitimately vary across runs, machines, or temp directories.
_VOLATILE = {
    "seal_path",
    "seal_sha256",
    "seal_hmac",
    "region_lock",
    "lock_path",
    "pid",
    "host",
    "elapsed_ms",
    "wall_clock_ms",
    "elapsed_s",
    "wait_ms",
    "hold_ms",
    "created_at",
    "acquired_at",
    "released_at",
    "generated_at",
    "sealed_at",
}

_CLEAR_CHILD_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "DASHSCOPE_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "WEAVEPROF_API_KEY",
    "WEAVEPROF_LLM_API_KEY",
    "WEAVEPROF_GOLDEN_UPDATE",
    "WEAVEPROF_METRICS_UPDATE",
    UPDATE_ENV,
    "WEAVEPROF_LOCK_PROVIDER",
    "WEAVEPROF_KEEP_SHADOW",
    "WEAVEPROF_SUPERVISOR_KEY_FILE",
    "TRACE_PROCESSOR_SHELL",
    "FI_SAMPLE",
    "WP_SAMPLE",
    "WEAVEPROF_SAMPLE",
)

# Synthetic 2-round self-test kernel & pre-verified source SHA-256 digests.
_SELFTEST_KERNEL_SRC = """#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void selftest_handoff(const uint32_t *x, float *out, int steps) {
    __shared__ uint32_t smem_tile[2][64];
    const unsigned warp = threadIdx.x / 32;
    const unsigned lane = threadIdx.x % 32;
    for (int tile = 0; tile < steps; ++tile) {
        const unsigned s = tile % 2;
        if (warp == 1) {
            smem_tile[s][lane] = x[blockIdx.x * 64 + lane];
            __syncthreads();
        } else if (warp == 0) {
            __syncthreads();
            float acc = 0.0f;
            for (int k = 0; k < 64; ++k) {
                acc = acc + float(smem_tile[s][(lane + k) % 32]) * 1.25f;
                out[blockIdx.x * 32 + lane] = acc;
            }
        }
    }
}
"""
_SELFTEST_BASE_SHA = "df73d9cc1ca591e269e000c90658c06c47c2cc2616c65e7ae7bf40f3a84a1984"
_SELFTEST_R1_INST_SHA = "5b0603ae51eb39727f92462c5a3bc8f02b953221e532eb7bf61c454a39e346c4"
_SELFTEST_R2_INST_SHA = "e33fbe7e0481f715188c2119ded49e6232b6c6051bbc526dfefaf619f62c9f4c"

_OPEN_FLAG = 0x40000000
_END_FLAG = 0x80000000
_CLOSE_FLAG = 0x08000000


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def discover_cases(cases_dir: Path = CASES) -> list[str]:
    """Discover ``wpprofile_*`` replay cases under ``cases_dir`` without matching megakernel cases."""
    if not cases_dir.is_dir():
        return []
    return sorted(
        p.name
        for p in cases_dir.iterdir()
        if p.is_dir() and p.name.startswith("wpprofile_") and (p / "case.json").is_file()
    )


def validate_case_layout(case_dir: Path, *, require_expected: bool = True) -> list[str]:
    """Validate that a replay fixture directory contains all required files."""
    errors: list[str] = []
    if not case_dir.is_dir():
        return [f"missing case directory: {case_dir}"]
    case_json = case_dir / "case.json"
    if not case_json.is_file():
        errors.append("missing case.json")
        return errors
    meta = json.loads(case_json.read_text(encoding="utf-8"))
    entry = meta.get("entry_source")
    probe_rel = meta.get("probe", "probe.json")
    if not entry or not (case_dir / "kernel" / entry).is_file():
        errors.append(f"missing kernel source: kernel/{entry}")
    if not (case_dir / "kernel" / probe_rel).is_file():
        errors.append(f"missing probe spec: kernel/{probe_rel}")
    rec = case_dir / "recorded"
    for req in ("baseline_source.sha256", "instrumented_source.sha256", "runtime.json"):
        if not (rec / req).is_file():
            errors.append(f"missing recorded/{req}")
    if require_expected and not (case_dir / "expected.json").is_file():
        errors.append("missing expected.json")
    return errors


def _path_tokens(tmp: str, work_run_dirs: list[Path] | None = None) -> list[tuple[str, str]]:
    """Build ``(real_path, placeholder)`` pairs ordered longest-first."""
    pairs: set[tuple[str, str]] = {
        (str(ROOT), "<REPO>"),
        (os.path.realpath(str(ROOT)), "<REPO>"),
    }
    if tmp:
        pairs |= {
            (tmp, "<TMP>"),
            (os.path.realpath(tmp), "<TMP>"),
        }
    for idx, wdir in enumerate(work_run_dirs or ()):
        tok = "<WORK_RUN>" if idx == 0 else f"<WORK_RUN_{idx}>"
        s = str(wdir)
        pairs |= {(s, tok), (os.path.realpath(s), tok)}
    return sorted((p for p in pairs if p[0]), key=lambda item: (-len(item[0]), item[1]))


def _norm(obj, tokens: list[tuple[str, str]]):
    """Drop volatile keys, replace machine/run paths with placeholders, and round floats to 6 digits."""
    if isinstance(obj, str):
        for real, placeholder in tokens:
            if real in obj:
                obj = obj.replace(real, placeholder)
        return obj
    if isinstance(obj, dict):
        return {k: _norm(v, tokens) for k, v in sorted(obj.items()) if k not in _VOLATILE}
    if isinstance(obj, list):
        return [_norm(v, tokens) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def _norm_ledger(raw_ledger: dict, tokens: list[tuple[str, str]]) -> dict:
    """Normalize ``episode_ledger.json``, stripping wall-clock ``ts`` from ``fsm_transitions``."""
    cleaned = dict(raw_ledger)
    transitions = []
    for item in raw_ledger.get("fsm_transitions") or []:
        if isinstance(item, dict):
            transitions.append({k: v for k, v in item.items() if k != "ts"})
        else:
            transitions.append(item)
    cleaned["fsm_transitions"] = transitions
    return _norm(cleaned, tokens)


def _first_diff(a, b, path: str = "$") -> str | None:
    """Return the JSON path and values of the first difference between ``a`` and ``b``."""
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__} ({a!r:.80} vs {b!r:.80})"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: {'missing in expected' if k not in a else 'missing in actual'}"
            diff = _first_diff(a[k], b[k], f"{path}.{k}")
            if diff:
                return diff
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for idx, (x, y) in enumerate(zip(a, b)):
            diff = _first_diff(x, y, f"{path}[{idx}]")
            if diff:
                return diff
        return None
    return None if a == b else f"{path}: expected {a!r:.120} got {b!r:.120}"


def _inspect_png(png_path: Path, has_mpl: bool, has_aux_warnings: bool) -> dict:
    if not has_mpl:
        return {
            "checked": False,
            "exists": None,
            "valid": None,
            "renderer": None,
            "has_auxiliary_warnings": has_aux_warnings,
        }
    if not png_path.is_file():
        return {
            "checked": True,
            "exists": False,
            "valid": False,
            "renderer": None,
            "has_auxiliary_warnings": has_aux_warnings,
        }
    raw = png_path.read_bytes()
    is_png = len(raw) > 1024 and raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[12:16] == b"IHDR"
    renderer = None
    if is_png:
        width_px, _height_px = struct.unpack(">II", raw[16:24])
        if width_px in (3520, 3696):
            renderer = "viz"
        elif width_px == 3300:
            renderer = "preview"
        else:
            renderer = f"width_{width_px}"
    return {
        "checked": True,
        "exists": True,
        "valid": is_png,
        "renderer": renderer,
        "has_auxiliary_warnings": has_aux_warnings,
    }


def _norm_trace_events(obj, tokens: list[tuple[str, str]]):
    """Normalize Perfetto ``traceEvents`` paths and floats without dropping Perfetto track ``pid``."""
    if isinstance(obj, str):
        for real, placeholder in tokens:
            if real in obj:
                obj = obj.replace(real, placeholder)
        return obj
    if isinstance(obj, dict):
        return {k: _norm_trace_events(v, tokens) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_norm_trace_events(v, tokens) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def _collect_aligned_trace(
    round_dir: Path,
    raw_verify: dict,
    tokens: list[tuple[str, str]],
) -> dict | None:
    """Summarize the aligned Perfetto trace for ``PASS`` rounds; return ``None`` on ``FAIL`` rounds."""
    metrics = raw_verify.get("metrics") or {}
    if raw_verify.get("status") != "PASS" or "alignment" not in metrics:
        return None

    aligned_doc = None
    gz_path_str = metrics.get("perfetto_json_gz")
    if gz_path_str and Path(gz_path_str).is_file():
        aligned_doc = json.loads(gzip.decompress(Path(gz_path_str).read_bytes()).decode("utf-8"))
    elif round_dir.is_dir():
        from weaveprof.workspace import implementation

        decoder = implementation("weaveprof.pipeline.validate.tools.fi_lite_decode")
        merge = implementation("weaveprof.pipeline.validate.tools.fi_merge_perfetto")
        rt = json.loads((round_dir / "runtime.json").read_text(encoding="utf-8"))
        names = json.loads((round_dir / "names.json").read_text(encoding="utf-8"))
        dump = decoder.read_dump(str(round_dir / "capture.fi02.bin"))
        trace, summary = decoder.decode(dump, names)
        window = merge.hardware_window(round_dir, rt)
        aligned_doc = merge.align(trace, summary, window, rt)

    if aligned_doc is None:
        return None

    events = _norm_trace_events(aligned_doc.get("traceEvents") or [], tokens)
    sorted_events = sorted(
        events,
        key=lambda ev: (
            int(ev.get("pid", 0)),
            int(ev.get("tid", 0)),
            int(ev.get("ts_ns", -1)),
            str(ev.get("name", "")),
            _canon(ev),
        ),
    )
    by_ph = Counter(str(ev.get("ph")) for ev in sorted_events)
    by_name = Counter(str(ev.get("name")) for ev in sorted_events)
    return {
        "events": len(sorted_events),
        "sha256": _sha(_canon(sorted_events)),
        "by_ph": dict(sorted(by_ph.items())),
        "by_name": dict(sorted(by_name.items())),
    }


def _record_child_pid(pid: int, role: str) -> None:
    pid_report = os.environ.get("P1D_PID_REPORT_PATH")
    if not pid_report:
        return
    try:
        with Path(pid_report).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"pid": pid, "role": role}) + "\n")
    except OSError:
        pass


def run_case(case_dir: Path, tmp: Path, *, mpl_cache_dir: Path | None = None) -> dict:
    """Execute ``python -B -m weaveprof profile`` for ``case_dir`` inside isolated ``tmp``."""
    from weaveprof.harness import verify_seal

    meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    entry_source = meta["entry_source"]
    probe_rel = meta.get("probe", "probe.json")
    card = meta.get("card", "h20")
    limit = float(meta.get("limit", 0.10))
    max_rounds = int(meta.get("max_rounds", 1))
    out_name = meta.get("out_name", "overview.png")
    runtime_overrides = meta.get("runtime_overrides")

    kernel_dir = tmp / "kernel"
    shutil.copytree(case_dir / "kernel", kernel_dir)
    out_dir = tmp / "out"
    work_dir = tmp / "work"
    cache_dir = tmp / "cache"
    lock_dir = tmp / "locks"
    home_dir = tmp / "home"
    sys_tmp = tmp / "systmp"
    mpl_dir = mpl_cache_dir if mpl_cache_dir is not None else (tmp / "mpl")
    for directory in (out_dir, work_dir, cache_dir, lock_dir, home_dir, sys_tmp, mpl_dir):
        directory.mkdir(parents=True, exist_ok=True)

    bench_argv = [
        sys.executable,
        "-B",
        str(REPLAY_BENCH),
        "--recorded-dir",
        str((case_dir / "recorded").resolve()),
    ]
    if runtime_overrides:
        bench_argv.extend(["--runtime-override-json", json.dumps(runtime_overrides, sort_keys=True)])

    out_target = out_dir / out_name
    cli_argv = [
        sys.executable,
        "-B",
        "-m",
        "weaveprof",
        "profile",
        "--probe",
        str(kernel_dir / probe_rel),
        "--max-rounds",
        str(max_rounds),
        "--limit",
        str(limit),
        "--card",
        str(card),
        "-o",
        str(out_target),
        str(kernel_dir / entry_source),
        "--",
        "--",
        *bench_argv,
    ]

    env = {k: v for k, v in os.environ.items() if k not in _CLEAR_CHILD_ENV_KEYS}
    existing_pp = env.get("PYTHONPATH", "")
    env.update(
        {
            "PYTHONPATH": str(ROOT) if not existing_pp else f"{ROOT}{os.pathsep}{existing_pp}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "HOME": str(home_dir),
            "XDG_CACHE_HOME": str(home_dir / ".cache"),
            "XDG_CONFIG_HOME": str(home_dir / ".config"),
            "TMPDIR": str(sys_tmp),
            "MPLCONFIGDIR": str(mpl_dir),
            "MPLBACKEND": "Agg",
            "WEAVEPROF_WORKDIR": str(work_dir),
            "WEAVEPROF_CACHE": str(cache_dir),
            "WEAVEPROF_LOCK_DIR": str(lock_dir),
            "WEAVEPROF_SUPERVISOR_HMAC_KEY": _HMAC_KEY,
        }
    )

    proc = subprocess.Popen(
        cli_argv,
        cwd=str(kernel_dir),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _record_child_pid(proc.pid, "weaveprof_cli_profile")
    stdout_text, _stderr_text = proc.communicate(timeout=180)

    ledger_p = out_dir / "episode_ledger.json"
    raw_ledger = json.loads(ledger_p.read_text(encoding="utf-8")) if ledger_p.is_file() else {}
    work_run_dirs: list[Path] = []
    for r_item in raw_ledger.get("rounds") or []:
        rdir_str = r_item.get("round_dir")
        if rdir_str:
            parent = Path(rdir_str).parent
            if parent not in work_run_dirs:
                work_run_dirs.append(parent)
    if not work_run_dirs and work_dir.is_dir():
        work_run_dirs.extend(sorted(p for p in work_dir.iterdir() if p.is_dir()))

    tokens = _path_tokens(str(tmp), work_run_dirs)

    verify_p = out_dir / "verify.json"
    feedback_p = out_dir / "feedback.json"
    raw_verify = json.loads(verify_p.read_text(encoding="utf-8")) if verify_p.is_file() else None
    raw_feedback = json.loads(feedback_p.read_text(encoding="utf-8")) if feedback_p.is_file() else None

    old_hmac = os.environ.get("WEAVEPROF_SUPERVISOR_HMAC_KEY")
    os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = _HMAC_KEY
    try:
        src_sha = raw_ledger.get("source_sha256")
        rounds_obs: list[dict] = []
        last_round_dir: Path | None = None
        for r_item in raw_ledger.get("rounds") or []:
            rdir = Path(r_item["round_dir"])
            last_round_dir = rdir
            r_status = str(r_item.get("status") or "FAIL")
            names_p = rdir / "names.json"
            probe_p = rdir / "probe.json"
            base_src_p = rdir / "L0" / entry_source
            inst_src_p = rdir / "L_inst" / entry_source
            rseal_p = rdir / "seal.json"
            if rseal_p.is_file():
                rchk = verify_seal(rseal_p, require_pass=(r_status == "PASS"), expected_source_sha=src_sha)
                rseal_obs = {
                    "exists": True,
                    "ok": bool(rchk["ok"]),
                    "errors": _norm(rchk["errors"], tokens),
                    "status": (rchk.get("seal") or {}).get("status"),
                    "fsm_state": (rchk.get("seal") or {}).get("fsm_state"),
                }
            else:
                rseal_obs = {"exists": False, "ok": False, "errors": ["missing round seal.json"], "status": None, "fsm_state": None}
            rounds_obs.append(
                {
                    "round_number": int(r_item["round_number"]),
                    "names": _norm(json.loads(names_p.read_text(encoding="utf-8")), tokens) if names_p.is_file() else None,
                    "bound_probe": _norm(json.loads(probe_p.read_text(encoding="utf-8")), tokens) if probe_p.is_file() else None,
                    "baseline_source_sha256": _sha(base_src_p.read_bytes()) if base_src_p.is_file() else None,
                    "instrumented_source_sha256": _sha(inst_src_p.read_bytes()) if inst_src_p.is_file() else None,
                    "round_seal": rseal_obs,
                }
            )

        prom_seal_p = Path(str(out_target) + ".seal.json")
        if prom_seal_p.is_file():
            pchk = verify_seal(prom_seal_p, require_pass=True, expected_source_sha=src_sha)
            promoted_seal_obs = {
                "exists": True,
                "ok": bool(pchk["ok"]),
                "errors": _norm(pchk["errors"], tokens),
            }
        else:
            promoted_seal_obs = {"exists": False, "ok": None, "errors": []}
    finally:
        if old_hmac is None:
            os.environ.pop("WEAVEPROF_SUPERVISOR_HMAC_KEY", None)
        else:
            os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = old_hmac

    try:
        import matplotlib  # noqa: F401
        has_mpl = True
    except ImportError:
        has_mpl = False

    png_path = out_target if out_target.suffix.lower() == ".png" else out_target.with_suffix(".png")
    has_aux_warn = bool(((raw_verify or {}).get("metrics") or {}).get("auxiliary_warnings"))
    png_obs = _inspect_png(png_path, has_mpl, has_aux_warn)

    aligned_trace_obs = (
        _collect_aligned_trace(last_round_dir, raw_verify, tokens)
        if (last_round_dir is not None and raw_verify is not None)
        else None
    )

    failed_timeline_obs = None
    if last_round_dir is not None:
        ft_path = last_round_dir / "failed_trace.perfetto-trace"
        if ft_path.is_file():
            ft_raw = ft_path.read_bytes()
            failed_timeline_obs = {
                "exists": True,
                "bytes": len(ft_raw),
                "sha256": _sha(ft_raw),
            }

    stdout_lines = [ln.strip() for ln in (stdout_text or "").splitlines() if ln.strip()]
    last_stdout = _norm(stdout_lines[-1], tokens) if stdout_lines else ""

    return {
        "cli_returncode": proc.returncode,
        "cli_stdout_last_line": last_stdout,
        "verify": _norm(raw_verify, tokens) if raw_verify is not None else None,
        "feedback": _norm(raw_feedback, tokens) if raw_feedback is not None else None,
        "ledger": _norm_ledger(raw_ledger, tokens) if raw_ledger else None,
        "rounds": rounds_obs,
        "promoted_seal": promoted_seal_obs,
        "aligned_trace": aligned_trace_obs,
        "failed_timeline": failed_timeline_obs,
        "png": png_obs,
    }


def _write_xz(path: Path, payload: bytes) -> None:
    with lzma.open(path, "wb", preset=6) as fh:
        fh.write(payload)


def _format_sass(ops: list[str]) -> str:
    lines = [
        "\t\tFunction : selftest_handoff",
        '\t.headerflags\t@"EF_CUDA_SM90 EF_CUDA_PTX_SM(EF_CUDA_SM90)"',
    ]
    for idx, body in enumerate(ops):
        lines.append(f"        /*{idx * 16:04x}*/                   {body} ;")
    return "\n".join(lines) + "\n"


def _build_selftest_dump(inner_iters: int) -> bytes:
    """Synthesize a deterministic WPLITE02 dump for 2 CTAs x 2 warps (DMA->Math cross-warp handoff)."""
    wpc = 2
    n_ctas = 2
    n_slots = n_ctas * wpc
    cap = 1024
    meta = json.dumps({"arch": "sm_90a", "backend": "nvidia_cuda", "gran": 0}, sort_keys=True).encode("utf-8")
    rec_struct = struct.Struct("<QII")
    slot_records: list[list[tuple[int, int, int]]] = []
    base_t0 = 1_000_000
    for cta in range(n_ctas):
        sm_id = cta * 2
        cta_off = cta * 1_000
        math_recs: list[tuple[int, int, int]] = [(base_t0 + cta_off, _OPEN_FLAG | 1, sm_id)]
        dma_recs: list[tuple[int, int, int]] = [(base_t0 + cta_off, _OPEN_FLAG | 0, sm_id)]
        for tile in range(4):
            s = tile % 2
            t_tile = base_t0 + cta_off + 1_000 + tile * 9_000
            dma_recs.append((t_tile, 1, s))
            math_recs.append((t_tile + 2_500, _END_FLAG | 1, s))
            mma_start = t_tile + 3_000
            step_ns = 5_000 // inner_iters
            dur_ns = max(40, step_ns - 20)
            for k in range(inner_iters):
                b_ns = mma_start + k * step_ns
                math_recs.append((b_ns, 3, 0))
                math_recs.append((b_ns + dur_ns, _END_FLAG | 3, 0))
        math_recs.append((base_t0 + cta_off + 39_000, _CLOSE_FLAG | 1, sm_id))
        dma_recs.append((base_t0 + cta_off + 39_000, _CLOSE_FLAG | 0, sm_id))
        slot_records.append(math_recs)
        slot_records.append(dma_recs)

    counts = [len(r) for r in slot_records]
    header = b"WPLITE02" + struct.pack("<6I", n_slots, cap, wpc, 0, 0, len(meta))
    cnt_bytes = struct.pack(f"<{n_slots}I", *counts)
    body = b"".join(b"".join(rec_struct.pack(*rec) for rec in recs) for recs in slot_records)
    return header + meta + cnt_bytes + body


def _build_selftest_collapse_dump() -> bytes:
    """Synthesize a WPLITE02 dump where hoisted ``mma`` spans 120 us in 1 span/lane (>= 100 us collapse limit)."""
    wpc = 2
    n_ctas = 2
    n_slots = n_ctas * wpc
    cap = 1024
    meta = json.dumps({"arch": "sm_90a", "backend": "nvidia_cuda", "gran": 0}, sort_keys=True).encode("utf-8")
    rec_struct = struct.Struct("<QII")
    slot_records: list[list[tuple[int, int, int]]] = []
    base_t0 = 1_000_000
    for cta in range(n_ctas):
        sm_id = cta * 2
        t0 = base_t0 + cta * 1_000
        dma_recs: list[tuple[int, int, int]] = [
            (t0, _OPEN_FLAG | 0, sm_id),
            (t0 + 1_000, 1, 0),
            (t0 + 4_000, _CLOSE_FLAG | 0, sm_id),
        ]
        math_recs: list[tuple[int, int, int]] = [
            (t0, _OPEN_FLAG | 1, sm_id),
            (t0 + 3_000, _END_FLAG | 1, 0),
            (t0 + 4_000, 3, 0),
            (t0 + 124_000, _END_FLAG | 3, 0),
            (t0 + 130_000, _CLOSE_FLAG | 1, sm_id),
        ]
        slot_records.append(math_recs)
        slot_records.append(dma_recs)

    counts = [len(r) for r in slot_records]
    header = b"WPLITE02" + struct.pack("<6I", n_slots, cap, wpc, 0, 0, len(meta))
    cnt_bytes = struct.pack(f"<{n_slots}I", *counts)
    body = b"".join(b"".join(rec_struct.pack(*rec) for rec in recs) for recs in slot_records)
    return header + meta + cnt_bytes + body


def _build_selftest_sqlite(
    tmp_dir: Path,
    *,
    with_gpu_metrics: bool,
    kernel_end_ns: int = 10_050_000,
) -> bytes:
    """Synthesize a minimal Nsight Systems SQLite capture (with 200 kHz H20 GPU_METRICS when requested)."""
    db_path = tmp_dir / f"_build_{'hw' if with_gpu_metrics else 'nohw'}_{kernel_end_ns}.sqlite"
    db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE TARGET_INFO_GPU (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER,
                end INTEGER,
                demangledName INTEGER,
                deviceId INTEGER
            );
            """
        )
        conn.execute("INSERT INTO StringIds (id, value) VALUES (1, 'selftest_handoff')")
        conn.execute("INSERT INTO TARGET_INFO_GPU (id, name) VALUES (0, 'NVIDIA H20')")
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, demangledName, deviceId) VALUES (?, ?, 1, 0)",
            (10_000_000, int(kernel_end_ns)),
        )
        if with_gpu_metrics:
            conn.executescript(
                """
                CREATE TABLE TARGET_INFO_GPU_METRICS (
                    typeId INTEGER,
                    metricId INTEGER,
                    metricName TEXT
                );
                CREATE TABLE GPU_METRICS (
                    typeId INTEGER,
                    metricId INTEGER,
                    timestamp INTEGER,
                    value REAL
                );
                """
            )
            conn.executemany(
                "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?, ?)",
                [
                    (1, 10, "Tensor Active [Throughput %]"),
                    (1, 11, "DRAM Read Bandwidth [Throughput %]"),
                    (1, 12, "DRAM Write Bandwidth [Throughput %]"),
                    (1, 13, "SMs Active [Throughput %]"),
                ],
            )
            rows = []
            for idx in range(11):
                ts = 10_000_000 + idx * 5_000
                rows.extend(
                    [
                        (1, 10, ts, 18.0 + (idx % 3) * 2.0),
                        (1, 11, ts, 28.0 + (idx % 2) * 4.0),
                        (1, 12, ts, 10.0 + (idx % 2) * 2.0),
                        (1, 13, ts, 50.0),
                    ]
                )
            conn.executemany("INSERT INTO GPU_METRICS VALUES (?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()
    raw = db_path.read_bytes()
    db_path.unlink(missing_ok=True)
    return raw


def generate_selftest_cases(dest_cases_dir: Path) -> tuple[Path, Path, Path]:
    """Generate three synthetic replay fixtures (round1 FAIL, collapse FAIL, round2 PASS) in ``dest_cases_dir``."""
    dest_cases_dir.mkdir(parents=True, exist_ok=True)
    l0_ops = [
        "MOV R1, c[0x0][0x28]",
        "S2R R0, SR_TID.X",
        "ISETP.GE.AND P0, PT, R0, 0x20, PT",
        "LDGSTS.E.BYPASS.128 [R4], [R2]",
        "BAR.SYNC 0x0",
        "LDS.U.128 R8, [R4]",
        "HMMA.16816.F32 R12, R8, R10, R12",
        "HMMA.16816.F32 R16, R8, R10, R16",
        "STG.E [R6], R12",
        "EXIT",
    ]
    r1_inst_ops = [
        "MOV R1, c[0x0][0x28]",
        "CS2R R20, SR_GLOBALTIMERLO",
        "STG.E.EF.128 [R22], R20",
        "LDGSTS.E.BYPASS.128 [R4], [R2]",
        "BAR.SYNC 0x0",
        "CS2R R24, SR_GLOBALTIMERLO",
        "STG.E.EF.128 [R22], R24",
        "HMMA.16816.F32 R12, R8, R10, R12",
        "STG.E [R6], R12",
        "EXIT",
    ]
    r2_inst_ops = [
        "S2R R0, SR_TID.X",
        "MOV R1, c[0x0][0x28]",
        "CS2R R20, SR_GLOBALTIMERLO",
        "ISETP.GE.AND P0, PT, R0, 0x20, PT",
        "LDGSTS.E.BYPASS.128 [R4], [R2]",
        "BAR.SYNC 0x0",
        "LDS.U.128 R8, [R4]",
        "HMMA.16816.F32 R12, R8, R10, R12",
        "HMMA.16816.F32 R16, R8, R10, R16",
        "STG.E.EF.128 [R22], R20",
        "STG.E [R6], R12",
        "EXIT",
    ]
    output_bytes = b"".join(struct.pack("<f", float(i) * 0.5) for i in range(64))

    hoisted_probe = {
        "roles": {"dma": "warp 1", "math": "warp 0"},
        "probes": [
            {"id": "load", "begin": {"role": "dma", "line": 11}, "end": {"role": "math", "line": 14}, "slot": "s"},
            {"id": "mma", "begin": {"role": "math", "line": 16}, "end": {"role": "math", "line": 19}},
        ],
    }

    specs = (
        (
            "wpprofile_selftest_round1_fail",
            "round1_fail",
            {
                "roles": {"dma": "warp 1", "math": "warp 0"},
                "probes": [
                    {"id": "load", "begin": {"role": "dma", "line": 11}, "end": {"role": "math", "line": 14}, "slot": "s"},
                    {"id": "mma", "begin": {"role": "math", "line": 17}, "end": {"role": "math", "line": 18}},
                ],
            },
            _SELFTEST_R1_INST_SHA,
            r1_inst_ops,
            _build_selftest_dump(64),
            10_050_000,
            False,
            [40000, 40200, 39800, 40100, 40000, 39900],
            [55000, 55200, 54800, 55100, 55000, 54900],
        ),
        (
            "wpprofile_selftest_collapse_fail",
            "collapse_fail",
            hoisted_probe,
            _SELFTEST_R2_INST_SHA,
            r2_inst_ops,
            _build_selftest_collapse_dump(),
            10_150_000,
            False,
            [140000, 140200, 139800, 140100, 140000, 139900],
            [144200, 144400, 144000, 144300, 144200, 144100],
        ),
        (
            "wpprofile_selftest_round2_pass",
            "round2_pass",
            hoisted_probe,
            _SELFTEST_R2_INST_SHA,
            r2_inst_ops,
            _build_selftest_dump(1),
            10_050_000,
            True,
            [40000, 40200, 39800, 40100, 40000, 39900],
            [41600, 41800, 41400, 41700, 41600, 41500],
        ),
    )

    built: list[Path] = []
    for cname, role_tag, probe_doc, inst_sha, inst_ops, dump_bytes, k_end_ns, with_hw, base_ns, inst_ns in specs:
        cdir = dest_cases_dir / cname
        if cdir.exists():
            shutil.rmtree(cdir)
        (cdir / "kernel").mkdir(parents=True)
        (cdir / "recorded").mkdir(parents=True)
        (cdir / "case.json").write_text(
            json.dumps(
                {
                    "entry_source": "selftest_handoff.cu",
                    "probe": "probe.json",
                    "card": "h20",
                    "limit": 0.10,
                    "max_rounds": 1,
                    "out_name": "overview.png",
                    "provenance": {
                        "kind": "runtime_generated_selftest",
                        "round_role": role_tag,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (cdir / "kernel" / "selftest_handoff.cu").write_text(_SELFTEST_KERNEL_SRC, encoding="utf-8")
        (cdir / "kernel" / "probe.json").write_text(
            json.dumps(probe_doc, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rec = cdir / "recorded"
        (rec / "baseline_source.sha256").write_text(_SELFTEST_BASE_SHA + "\n", encoding="utf-8")
        (rec / "instrumented_source.sha256").write_text(inst_sha + "\n", encoding="utf-8")
        (rec / "l0.sass").write_text(_format_sass(l0_ops), encoding="utf-8")
        (rec / "inst.sass").write_text(_format_sass(inst_ops), encoding="utf-8")
        (rec / "baseline.bin").write_bytes(output_bytes)
        (rec / "instrumented.bin").write_bytes(output_bytes)
        _write_xz(rec / "capture.fi02.bin.xz", dump_bytes)
        _write_xz(
            rec / "capture.sqlite.xz",
            _build_selftest_sqlite(dest_cases_dir, with_gpu_metrics=with_hw, kernel_end_ns=k_end_ns),
        )
        runtime_doc = {
            "kernel_name": "selftest_handoff",
            "card": "h20",
            "baseline_sass": "l0.sass",
            "instrumented_sass": "inst.sass",
            "baseline_output": "baseline.bin",
            "instrumented_output": "instrumented.bin",
            "baseline_regs": 40,
            "instrumented_regs": 42,
            "baseline_spill_bytes": 0,
            "instrumented_spill_bytes": 0,
            "baseline_stack_bytes": 0,
            "instrumented_stack_bytes": 0,
            "extra_smem_bytes": 0,
            "warmup": 5,
            "timing_protocol": "ABBA",
            "baseline_ns": base_ns,
            "instrumented_ns": inst_ns,
            "expected_warps_per_cta": 2,
        }
        (rec / "runtime.json").write_text(
            json.dumps(runtime_doc, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        built.append(cdir)
    return built[0], built[1], built[2]


def _is_loopback_host(host) -> bool:
    if not host:
        return True
    s = str(host).strip()
    if s.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(s).is_loopback
    except ValueError:
        return False


class WpProfileReplayTest(unittest.TestCase):
    _env_keys = (
        "WEAVEPROF_SUPERVISOR_HMAC_KEY",
        "WEAVEPROF_SUPERVISOR_KEY_FILE",
        "WEAVEPROF_CACHE",
        "WEAVEPROF_LOCK_DIR",
        "WEAVEPROF_WORKDIR",
        "WEAVEPROF_LOCK_PROVIDER",
        "WEAVEPROF_KEEP_SHADOW",
        "MPLCONFIGDIR",
        "MPLBACKEND",
    )

    @classmethod
    def setUpClass(cls):
        cls._saved_env = {k: os.environ.get(k) for k in cls._env_keys}
        cls._root = Path(tempfile.mkdtemp(prefix="wp-profile-replay-"))
        cls._mpl_dir = cls._root / "shared_mpl"
        cls._mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = _HMAC_KEY
        os.environ.pop("WEAVEPROF_SUPERVISOR_KEY_FILE", None)
        os.environ["WEAVEPROF_CACHE"] = str(cls._root / "cache")
        os.environ["WEAVEPROF_LOCK_DIR"] = str(cls._root / "locks")
        os.environ["WEAVEPROF_WORKDIR"] = str(cls._root / "workdir")
        os.environ.pop("WEAVEPROF_LOCK_PROVIDER", None)
        os.environ.pop("WEAVEPROF_KEEP_SHADOW", None)
        os.environ["MPLCONFIGDIR"] = str(cls._mpl_dir)
        os.environ["MPLBACKEND"] = "Agg"

        orig_connect = socket.socket.connect
        orig_connect_ex = socket.socket.connect_ex
        orig_create_conn = socket.create_connection

        def _guarded_connect(sock_self, address):
            if sock_self.family in (socket.AF_INET, socket.AF_INET6) and isinstance(address, tuple):
                if not _is_loopback_host(address[0]):
                    raise OSError(f"Outbound network access blocked in offline test: {address!r}")
            return orig_connect(sock_self, address)

        def _guarded_connect_ex(sock_self, address):
            if sock_self.family in (socket.AF_INET, socket.AF_INET6) and isinstance(address, tuple):
                if not _is_loopback_host(address[0]):
                    raise OSError(f"Outbound network access blocked in offline test: {address!r}")
            return orig_connect_ex(sock_self, address)

        def _guarded_create_conn(address, *args, **kwargs):
            if isinstance(address, tuple) and address and not _is_loopback_host(address[0]):
                raise OSError(f"Outbound network access blocked in offline test: {address!r}")
            return orig_create_conn(address, *args, **kwargs)

        cls._net_patchers = [
            mock.patch.object(socket.socket, "connect", _guarded_connect),
            mock.patch.object(socket.socket, "connect_ex", _guarded_connect_ex),
            mock.patch.object(socket, "create_connection", _guarded_create_conn),
        ]
        for p in cls._net_patchers:
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in reversed(getattr(cls, "_net_patchers", ())):
            p.stop()
        for k, v in cls._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(cls._root, ignore_errors=True)

    def _check_case_dir(self, case_dir: Path, run_tmp: Path, *, update: bool = UPDATE) -> dict:
        """Unified replay runner + ``expected.json`` writer/comparator shared by PASS and FAIL cases."""
        layout_errs = validate_case_layout(case_dir, require_expected=not update)
        self.assertEqual(layout_errs, [], f"invalid case layout in {case_dir}: {layout_errs}")
        run_tmp.mkdir(parents=True, exist_ok=True)
        obs = run_case(case_dir, run_tmp, mpl_cache_dir=self._mpl_dir)
        exp_path = case_dir / "expected.json"
        if update or not exp_path.is_file():
            if not update:
                self.fail(f"{exp_path} missing; generate it with {UPDATE_ENV}=1")
            exp_path.write_text(
                json.dumps(obs, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            return obs
        expected = json.loads(exp_path.read_text(encoding="utf-8"))
        if obs["png"]["checked"] is False:
            expected["png"] = obs["png"]
        diff = _first_diff(expected, json.loads(_canon(obs)))
        self.assertIsNone(diff, f"wp.profile golden mismatch for {case_dir.name}: {diff}")
        return obs

    def test_registered_real_cases_inventory(self):
        self.assertIsInstance(EXPECTED_REAL_CASES, tuple)
        discovered = discover_cases(CASES)
        self.assertEqual(discovered, sorted(EXPECTED_REAL_CASES))
        for name in EXPECTED_REAL_CASES:
            self.assertEqual(validate_case_layout(CASES / name, require_expected=True), [])

    def test_cli_profile_double_dash_arg_parsing(self):
        from weaveprof import cli

        captured: list[dict] = []

        def _fake_profile(**kwargs):
            captured.append(dict(kwargs))
            return {"status": "PASS"}

        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with mock.patch("weaveprof.cli.api.profile", side_effect=_fake_profile), contextlib.redirect_stdout(
            out_buf
        ), contextlib.redirect_stderr(err_buf):
            rc = cli.main(
                [
                    "profile",
                    "--probe",
                    "probe.json",
                    "--max-rounds",
                    "1",
                    "--limit",
                    "0.10",
                    "--card",
                    "h20",
                    "-o",
                    "out/overview.png",
                    "handoff.cu",
                    "--",
                    "--",
                    "python3",
                    "-B",
                    "wpprofile_replay_bench.py",
                    "--recorded-dir",
                    "recorded",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(out_buf.getvalue().strip(), "out/overview.png")
            self.assertEqual(len(captured), 1)
            call_kw = captured[0]
            self.assertEqual(call_kw["source"], "handoff.cu")
            self.assertEqual(call_kw["probe"], "probe.json")
            self.assertEqual(
                call_kw["bench"],
                ["python3", "-B", "wpprofile_replay_bench.py", "--recorded-dir", "recorded"],
            )
            self.assertEqual(call_kw["max_rounds"], 1)
            self.assertEqual(call_kw["limit"], 0.10)
            self.assertEqual(call_kw["card"], "h20")
            self.assertEqual(call_kw["out"], "out/overview.png")

            with self.assertRaises(SystemExit) as bad_ctx:
                cli.main(
                    [
                        "profile",
                        "handoff.cu",
                        "--probe",
                        "probe.json",
                        "--max-rounds",
                        "1",
                        "-o",
                        "out/overview.png",
                        "--",
                        "python3",
                        "-B",
                        "wpprofile_replay_bench.py",
                    ]
                )
            self.assertEqual(bad_ctx.exception.code, 2)
            self.assertIn("expected -- <benchmark command>", err_buf.getvalue())
            self.assertEqual(len(captured), 1)

    def test_volatile_and_placeholder_normalization_contract(self):
        self.assertEqual(
            _VOLATILE,
            {
                "seal_path",
                "seal_sha256",
                "seal_hmac",
                "region_lock",
                "lock_path",
                "pid",
                "host",
                "elapsed_ms",
                "wall_clock_ms",
                "elapsed_s",
                "wait_ms",
                "hold_ms",
                "created_at",
                "acquired_at",
                "released_at",
                "generated_at",
                "sealed_at",
            },
        )
        tmp_a = str(self._root / "norm_a")
        tmp_b = str(self._root / "回放 路径_b")
        wrun_a = Path(tmp_a) / "work" / "11111111111111111111111111111111"
        wrun_b = Path(tmp_b) / "work" / "22222222222222222222222222222222"
        tok_a = _path_tokens(tmp_a, [wrun_a])
        tok_b = _path_tokens(tmp_b, [wrun_b])

        raw_a = {
            "status": "FAIL",
            "failed_trace": f"{wrun_a}/round-1/failed_trace.perfetto-trace",
            "out_png": f"{tmp_a}/out/overview.png",
            "repo_ref": f"{ROOT}/configs/semantic_ids.json",
            "ratio": 0.123456789,
            "seal_path": f"{wrun_a}/round-1/seal.json",
            "seal_sha256": "aaa111",
            "seal_hmac": "bbb222",
            "region_lock": {"pid": 101, "host": "host-a", "wait_ms": 1.2},
            "lock_path": f"{tmp_a}/locks/gpu.lock",
            "pid": 101,
            "host": "host-a",
            "elapsed_ms": 12.3,
            "wall_clock_ms": 45.6,
            "elapsed_s": 0.045,
            "wait_ms": 0.2,
            "hold_ms": 9.8,
            "created_at": 1700000001.0,
            "acquired_at": 1700000002.0,
            "released_at": 1700000003.0,
            "generated_at": 1700000004.0,
            "sealed_at": 1700000005.0,
            "fsm_transitions": [{"ts": 1700000001.5, "from": "INIT", "to": "POLICY_CHECK", "round": 1}],
        }
        raw_b = {
            "status": "FAIL",
            "failed_trace": f"{wrun_b}/round-1/failed_trace.perfetto-trace",
            "out_png": f"{tmp_b}/out/overview.png",
            "repo_ref": f"{ROOT}/configs/semantic_ids.json",
            "ratio": 0.123456611,
            "seal_path": f"{wrun_b}/round-1/seal.json",
            "seal_sha256": "ccc333",
            "seal_hmac": "ddd444",
            "region_lock": {"pid": 999, "host": "host-b", "wait_ms": 8.9},
            "lock_path": f"{tmp_b}/locks/gpu.lock",
            "pid": 999,
            "host": "host-b",
            "elapsed_ms": 99.1,
            "wall_clock_ms": 180.4,
            "elapsed_s": 0.180,
            "wait_ms": 4.4,
            "hold_ms": 77.2,
            "created_at": 1800000001.0,
            "acquired_at": 1800000002.0,
            "released_at": 1800000003.0,
            "generated_at": 1800000004.0,
            "sealed_at": 1800000005.0,
            "fsm_transitions": [{"ts": 1800000099.2, "from": "INIT", "to": "POLICY_CHECK", "round": 1}],
        }
        norm_a = _norm_ledger(raw_a, tok_a)
        norm_b = _norm_ledger(raw_b, tok_b)
        self.assertIsNone(_first_diff(norm_a, norm_b))
        self.assertEqual(norm_a["failed_trace"], "<WORK_RUN>/round-1/failed_trace.perfetto-trace")
        self.assertEqual(norm_a["out_png"], "<TMP>/out/overview.png")
        self.assertEqual(norm_a["repo_ref"], "<REPO>/configs/semantic_ids.json")
        self.assertEqual(norm_a["ratio"], 0.123457)
        self.assertEqual(norm_a["fsm_transitions"], [{"from": "INIT", "round": 1, "to": "POLICY_CHECK"}])

    def test_generated_three_branches_end_to_end_and_update_switch(self):
        gen_cases_dir = self._root / "generated_cases"
        r1_dir, col_dir, r2_dir = generate_selftest_cases(gen_cases_dir)
        self.assertEqual(
            discover_cases(gen_cases_dir),
            [
                "wpprofile_selftest_collapse_fail",
                "wpprofile_selftest_round1_fail",
                "wpprofile_selftest_round2_pass",
            ],
        )

        # Missing expected.json without WEAVEPROF_WPPROFILE_REPLAY_UPDATE=1 must fail closed.
        with self.assertRaises(AssertionError) as missing_ctx:
            self._check_case_dir(r1_dir, self._root / "missing_exp_check", update=False)
        self.assertIn("missing expected.json", str(missing_ctx.exception))

        # Step 1: Generate expected.json via update=True in directory A for all 3 branches.
        self._check_case_dir(r1_dir, self._root / "update_run_r1", update=True)
        self._check_case_dir(col_dir, self._root / "update_run_col", update=True)
        self._check_case_dir(r2_dir, self._root / "update_run_r2", update=True)
        self.assertTrue((r1_dir / "expected.json").is_file())
        self.assertTrue((col_dir / "expected.json").is_file())
        self.assertTrue((r2_dir / "expected.json").is_file())

        # Step 2: Replay all 3 branches with update=False in a distinct Unicode + space directory.
        obs_r1 = self._check_case_dir(r1_dir, self._root / "回放 校验" / "r1_fail", update=False)
        obs_col = self._check_case_dir(col_dir, self._root / "回放 校验" / "collapse_fail", update=False)
        obs_r2 = self._check_case_dir(r2_dir, self._root / "回放 校验" / "r2_pass", update=False)

        # Concrete literal assertions on Branch A1 (round1_fail: assembly + overhead FAIL, failed_trace exported).
        self.assertEqual(obs_r1["cli_returncode"], 1)
        self.assertEqual(obs_r1["cli_stdout_last_line"], "")
        v1 = obs_r1["verify"]
        self.assertEqual(v1["status"], "FAIL")
        self.assertEqual(v1["failed"], ["assembly", "overhead"])
        self.assertEqual(
            v1["checks"],
            {"assembly": False, "event": True, "events": True, "output": True, "overhead": False},
        )
        self.assertEqual(v1["fires"], {"load": 4, "mma": 256})
        self.assertEqual(v1["enclosing_loop"], [16, 19])
        self.assertEqual(v1["failed_trace"], "<WORK_RUN>/round-1/failed_trace.perfetto-trace")
        m1 = v1["metrics"]
        self.assertNotIn("alignment", m1)
        self.assertNotIn("preview_png", m1)
        self.assertEqual(m1["lcs_ratio"], 0.6)
        self.assertEqual(m1["core_retention_ratio"], 0.666667)
        self.assertEqual(m1["core_multiset_ratio"], 0.666667)
        self.assertEqual(m1["assembly_change_rate"], 0.4)
        self.assertEqual(m1["order_change_ratio"], 0.0)
        self.assertEqual(m1["overhead_ratio"], 0.375)
        self.assertEqual(m1["delta_spill_bytes"], 0)
        self.assertEqual(m1["extra_smem_bytes"], 0)
        self.assertEqual(m1["compiler_spill_gate"]["ok"], True)
        self.assertEqual(m1["granularity_gate"]["ok"], True)
        self.assertIsNone(m1["granularity_gate"]["code"])
        self.assertEqual(m1["output_bytes"], 256)
        self.assertEqual(m1["overflow"], 0)
        self.assertEqual(m1["abba"]["protocol"], "ABBA")
        self.assertEqual(m1["abba"]["paired_samples"], 6)
        self.assertEqual(m1["abba"]["median_overhead_ratio"], 0.375)
        ts1 = v1["timeline_summary"]
        self.assertEqual(ts1["failed_trace_path"], "<WORK_RUN>/round-1/failed_trace.perfetto-trace")
        self.assertEqual(
            ts1["failed_trace_sha256"],
            "aa22daf39a2d12d5ac0124a595cf973e410661a1edfbe9ec022cd3e10220cce5",
        )
        self.assertEqual(ts1["failed_trace_bytes"], 151368)
        self.assertEqual(ts1["kernel_baseline_us"], 40.0)
        self.assertEqual(ts1["kernel_instrumented_us"], 55.0)
        self.assertEqual(ts1["overhead_pct"], 37.5)
        self.assertEqual(
            ts1["hotspots"],
            [
                'math "mma" (L17-L18): 0.0580us x 256 inside loop L16-L19',
                'dma "load" (L11-L14): 2.5000us x 4 inside loop L8-L21',
            ],
        )
        self.assertEqual(
            obs_r1["failed_timeline"],
            {
                "bytes": 151368,
                "exists": True,
                "sha256": "aa22daf39a2d12d5ac0124a595cf973e410661a1edfbe9ec022cd3e10220cce5",
            },
        )
        self.assertIsNone(obs_r1["aligned_trace"])
        self.assertEqual(obs_r1["promoted_seal"], {"errors": [], "exists": False, "ok": None})
        self.assertEqual(
            obs_r1["png"],
            {"checked": True, "exists": False, "has_auxiliary_warnings": False, "renderer": None, "valid": False},
        )
        l1 = obs_r1["ledger"]
        self.assertEqual(l1["status"], "FAIL")
        self.assertEqual(l1["reason"], "max_rounds_exhausted")
        self.assertEqual(l1["rounds_executed"], 1)
        self.assertIsNone(l1["promoted_round"])
        self.assertEqual(l1["source_sha256"], _SELFTEST_BASE_SHA)
        self.assertEqual(
            l1["policy_digest"],
            "f0977ae463c701d0dcdf606b35319ce1f427a794d03c62259d54a655821d777d",
        )
        self.assertEqual(
            [(t["from"], t["to"]) for t in l1["fsm_transitions"]],
            [
                ("INIT", "POLICY_CHECK"),
                ("POLICY_CHECK", "CRITICAL_REGION_MEASURE"),
                ("CRITICAL_REGION_MEASURE", "HALT_BLOCKED"),
            ],
        )
        self.assertEqual(len(obs_r1["rounds"]), 1)
        self.assertEqual(obs_r1["rounds"][0]["baseline_source_sha256"], _SELFTEST_BASE_SHA)
        self.assertEqual(obs_r1["rounds"][0]["instrumented_source_sha256"], _SELFTEST_R1_INST_SHA)
        self.assertEqual(obs_r1["rounds"][0]["names"]["roles"], {"0": "dma", "1": "math"})
        self.assertEqual(obs_r1["rounds"][0]["names"]["probe_ids"], {"1": "load", "3": "mma"})
        self.assertEqual(
            obs_r1["rounds"][0]["round_seal"],
            {"errors": [], "exists": True, "fsm_state": "VERIFY_AND_SEAL", "ok": True, "status": "FAIL"},
        )

        # Concrete literal assertions on Branch A1' (collapse_fail: hoisted probe, SINGLE_STAGE_LOOP_COLLAPSE FAIL).
        self.assertEqual(obs_col["cli_returncode"], 1)
        self.assertEqual(obs_col["cli_stdout_last_line"], "")
        vc = obs_col["verify"]
        self.assertEqual(vc["status"], "FAIL")
        self.assertEqual(vc["failed"], ["pairing"])
        self.assertEqual(
            vc["checks"],
            {"assembly": True, "event": False, "events": False, "output": True, "overhead": True},
        )
        self.assertEqual(vc["fires"], {"load": 1, "mma": 1})
        self.assertEqual(vc["enclosing_loop"], [8, 21])
        self.assertEqual(vc["failed_trace"], "<WORK_RUN>/round-1/failed_trace.perfetto-trace")
        mc = vc["metrics"]
        self.assertNotIn("alignment", mc)
        self.assertNotIn("preview_png", mc)
        self.assertEqual(mc["lcs_ratio"], 0.9)
        self.assertEqual(mc["order_change_ratio"], 0.1)
        self.assertEqual(mc["overhead_ratio"], 0.03)
        self.assertEqual(
            mc["granularity_gate"],
            {
                "code": "SINGLE_STAGE_LOOP_COLLAPSE",
                "distinct_active_stages": 2,
                "dominant_share_pct": 98.36,
                "dominant_stage": "mma",
                "has_compute_stage": True,
                "has_memory_stage": True,
                "message": (
                    "Stage 'mma' absorbs 98.4% of active time in 1.00 spans/lane (120.0 us/lane): "
                    "main loop collapsed into a single coarse bar; preserve per-tile stages "
                    "(load/mma/softmax/epilogue) inside the loop."
                ),
                "ok": False,
                "overall_spans_per_lane": 2.0,
                "per_lane_us": 120.0,
                "spans_per_lane": 1.0,
            },
        )
        self.assertEqual(mc["warp_gate"]["ok"], True)
        self.assertIn("SINGLE_STAGE_LOOP_COLLAPSE", mc["granularity_gate"]["code"])
        self.assertIn("120.0 us/lane", mc["errors"]["granularity"])
        tsc = vc["timeline_summary"]
        self.assertEqual(tsc["failed_trace_path"], "<WORK_RUN>/round-1/failed_trace.perfetto-trace")
        self.assertEqual(
            tsc["failed_trace_sha256"],
            "cf1dff48bf13e1d62258563af9f3256d0946c4678aaefbbe3d09fa1b9a7f381d",
        )
        self.assertEqual(tsc["failed_trace_bytes"], 4184)
        self.assertEqual(tsc["kernel_baseline_us"], 140.0)
        self.assertEqual(tsc["kernel_instrumented_us"], 144.2)
        self.assertEqual(tsc["overhead_pct"], 3.0)
        self.assertEqual(
            obs_col["failed_timeline"],
            {
                "bytes": 4184,
                "exists": True,
                "sha256": "cf1dff48bf13e1d62258563af9f3256d0946c4678aaefbbe3d09fa1b9a7f381d",
            },
        )
        self.assertIsNone(obs_col["aligned_trace"])
        self.assertEqual(obs_col["promoted_seal"], {"errors": [], "exists": False, "ok": None})
        self.assertEqual(
            obs_col["rounds"][0]["round_seal"],
            {"errors": [], "exists": True, "fsm_state": "VERIFY_AND_SEAL", "ok": True, "status": "FAIL"},
        )

        # Concrete literal assertions on Round 2 (PASS: hoisted mma [16, 19], H20 x4 counters, viz PNG, promoted seal).
        self.assertEqual(obs_r2["cli_returncode"], 0)
        self.assertEqual(obs_r2["cli_stdout_last_line"], "<TMP>/out/overview.png")
        v2 = obs_r2["verify"]
        self.assertEqual(v2["status"], "PASS")
        self.assertEqual(v2["failed"], [])
        self.assertEqual(
            v2["checks"],
            {"assembly": True, "event": True, "events": True, "output": True, "overhead": True},
        )
        self.assertEqual(v2["fires"], {"load": 4, "mma": 4})
        self.assertIsNone(v2["enclosing_loop"])
        self.assertNotIn("failed_trace", v2)
        self.assertNotIn("timeline_summary", v2)
        m2 = v2["metrics"]
        self.assertEqual(m2["lcs_ratio"], 0.9)
        self.assertEqual(m2["core_retention_ratio"], 1.0)
        self.assertEqual(m2["core_multiset_ratio"], 1.0)
        self.assertEqual(m2["assembly_change_rate"], 0.1)
        self.assertEqual(m2["order_change_ratio"], 0.1)
        self.assertEqual(m2["overhead_ratio"], 0.04)
        self.assertEqual(m2["preview_png"], "<TMP>/out/overview.png")
        al2 = m2["alignment"]
        self.assertEqual(al2["kernel"], "selftest_handoff")
        self.assertEqual(al2["kernel_start_ns"], 10000000)
        self.assertEqual(al2["kernel_end_ns"], 10050000)
        self.assertEqual(al2["kernel_duration_us"], 50.0)
        self.assertEqual(al2["first_open_ns"], 1000000)
        self.assertEqual(al2["offset_ns"], 9005000)
        self.assertEqual(al2["hardware_samples"], 44)
        self.assertEqual(al2["physical_units_observed"], 2)
        self.assertEqual(al2["ctas_observed"], 2)
        self.assertEqual(al2["warps_per_cta"], 2)
        self.assertEqual(al2["cross_warp_spans"], 8)
        self.assertEqual(al2["has_compute_curve"], True)
        self.assertEqual(al2["has_memory_curve"], True)
        self.assertEqual(al2["has_sm_curve"], True)
        self.assertEqual(
            al2["tensor_active_scale"],
            {
                "applied": True,
                "card_factor": 4.0,
                "raw_ceiling_pct": 29.0,
                "raw_max_pct": 22.0,
                "raw_mean_pct": 19.818,
                "reason": (
                    "nsys normalizes Tensor Active to the full GH100 die; H20 Tensor throughput per SM is "
                    "1/4 of it, so values are multiplied by 4"
                ),
                "scale": 4.0,
                "source": "nsys TARGET_INFO_GPU name 'NVIDIA H20'",
            },
        )
        self.assertIsNone(obs_r2["failed_timeline"])
        self.assertEqual(
            obs_r2["aligned_trace"],
            {
                "by_name": {
                    "Active SM Residency (%) [78 SMs]": 26,
                    "Active SMs (%) [NSYS PMU]": 11,
                    "DRAM Read Bandwidth [Throughput %]": 11,
                    "DRAM Write Bandwidth [Throughput %]": 11,
                    "Memory Util — HBM Read+Write (%) [NSYS PMU]": 11,
                    "SMs Active [Throughput %]": 11,
                    "Warp residency": 4,
                    "load": 8,
                    "mma": 8,
                    "process_name": 3,
                    "process_sort_index": 3,
                    "selftest_handoff": 1,
                    "tensor_active [Tensor Active x4 (H20 die-normalization corrected)]": 11,
                    "thread_name": 4,
                    "thread_sort_index": 4,
                },
                "by_ph": {"C": 92, "M": 14, "X": 21},
                "events": 127,
                "sha256": "6c3d2a195055f4bd01b087a683c9dbb52f5be02bb3a699ec809fc0d554659744",
            },
        )
        self.assertEqual(obs_r2["promoted_seal"], {"errors": [], "exists": True, "ok": True})
        self.assertEqual(
            obs_r2["png"],
            {"checked": True, "exists": True, "has_auxiliary_warnings": False, "renderer": "viz", "valid": True},
        )
        l2 = obs_r2["ledger"]
        self.assertEqual(l2["status"], "PASS")
        self.assertEqual(l2["reason"], "all_gates_passed")
        self.assertEqual(l2["rounds_executed"], 1)
        self.assertEqual(l2["promoted_round"], 1)
        self.assertEqual(
            [(t["from"], t["to"]) for t in l2["fsm_transitions"]],
            [
                ("INIT", "POLICY_CHECK"),
                ("POLICY_CHECK", "CRITICAL_REGION_MEASURE"),
                ("CRITICAL_REGION_MEASURE", "PROMOTED"),
            ],
        )
        self.assertEqual(obs_r2["rounds"][0]["instrumented_source_sha256"], _SELFTEST_R2_INST_SHA)
        self.assertEqual(obs_r2["rounds"][0]["bound_probe"]["source_path"], "<TMP>/kernel/selftest_handoff.cu")
        self.assertEqual(
            obs_r2["rounds"][0]["round_seal"],
            {"errors": [], "exists": True, "fsm_state": "VERIFY_AND_SEAL", "ok": True, "status": "PASS"},
        )

    def test_negative_first_diff_reports_mutated_pinned_field(self):
        neg_cases_dir = self._root / "neg_diff_cases"
        r1_dir, _col_dir, r2_dir = generate_selftest_cases(neg_cases_dir)
        self._check_case_dir(r1_dir, self._root / "neg_diff_seed_r1", update=True)
        self._check_case_dir(r2_dir, self._root / "neg_diff_seed_r2", update=True)

        # Mutate a pinned field inside round1_fail expected.json and verify _check_case_dir fails with exact path.
        exp1 = json.loads((r1_dir / "expected.json").read_text(encoding="utf-8"))
        exp1["verify"]["metrics"]["overhead_ratio"] = 0.05
        (r1_dir / "expected.json").write_text(json.dumps(exp1, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaises(AssertionError) as err1:
            self._check_case_dir(r1_dir, self._root / "neg_diff_check_r1", update=False)
        msg1 = str(err1.exception)
        self.assertIn("$.verify.metrics.overhead_ratio: expected 0.05 got 0.375", msg1)

        # Mutate a pinned alignment field inside round2_pass expected.json and verify exact path + both values.
        exp2 = json.loads((r2_dir / "expected.json").read_text(encoding="utf-8"))
        exp2["verify"]["metrics"]["alignment"]["offset_ns"] = 0
        (r2_dir / "expected.json").write_text(json.dumps(exp2, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaises(AssertionError) as err2:
            self._check_case_dir(r2_dir, self._root / "neg_diff_check_r2", update=False)
        msg2 = str(err2.exception)
        self.assertIn("$.verify.metrics.alignment.offset_ns: expected 0 got 9005000", msg2)

        # Structural diff coverage on _first_diff (missing key, list length, type mismatch).
        self.assertEqual(
            _first_diff({"a": 1}, {"a": 1, "b": 2}),
            "$.b: missing in expected",
        )
        self.assertEqual(
            _first_diff([1, 2], [1]),
            "$: length 2 != 1",
        )
        self.assertEqual(
            _first_diff({"x": "1"}, {"x": 1}),
            "$.x: type str != int ('1' vs 1)",
        )

    def test_negative_tampered_source_sha256_causes_replay_bench_nonzero_exit(self):
        tamper_cases_dir = self._root / "neg_sha_cases"
        r1_dir, _col_dir, _r2_dir = generate_selftest_cases(tamper_cases_dir)

        # 1. Tamper instrumented_source.sha256 and run wpprofile_replay_bench.py directly + via full CLI replay.
        wrong_inst_sha = "0" * 64
        (r1_dir / "recorded" / "instrumented_source.sha256").write_text(wrong_inst_sha + "\n", encoding="utf-8")

        obs_tampered = run_case(r1_dir, self._root / "neg_sha_full_run", mpl_cache_dir=self._mpl_dir)
        self.assertEqual(obs_tampered["cli_returncode"], 1)
        self.assertEqual(obs_tampered["verify"]["status"], "FAIL")
        self.assertEqual(obs_tampered["ledger"]["reason"], "max_rounds_exhausted")
        meas_err = obs_tampered["verify"]["metrics"]["errors"]["measurement"]
        self.assertIn("instrumented_source sha256 mismatch:", meas_err)
        self.assertIn(f"expected={wrong_inst_sha}", meas_err)
        self.assertIn(f"actual={_SELFTEST_R1_INST_SHA}", meas_err)

        # Direct subprocess check on wpprofile_replay_bench.py exit code (rc == 2) for both digests.
        direct_round = self._root / "direct_bench_round"
        direct_round.mkdir(parents=True, exist_ok=True)
        base_f = direct_round / "base.cu"
        inst_f = direct_round / "inst.cu"
        base_f.write_text(_SELFTEST_KERNEL_SRC, encoding="utf-8")
        inst_f.write_text("// modified instrumented body\n", encoding="utf-8")
        actual_mod_sha = _sha(inst_f.read_bytes())

        bench_env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "WEAVEPROF_BASELINE_SOURCE": str(base_f),
            "WEAVEPROF_INSTRUMENTED_SOURCE": str(inst_f),
            "WEAVEPROF_ROUND_DIR": str(direct_round),
        }
        if os.environ.get("P1D_PID_REPORT_PATH"):
            bench_env["P1D_PID_REPORT_PATH"] = os.environ["P1D_PID_REPORT_PATH"]

        p_inst = subprocess.run(
            [sys.executable, "-B", str(REPLAY_BENCH), "--recorded-dir", str(r1_dir / "recorded")],
            env=bench_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
        self.assertEqual(p_inst.returncode, 2)
        self.assertIn(
            f"instrumented_source sha256 mismatch: expected={wrong_inst_sha} actual={actual_mod_sha}",
            p_inst.stderr,
        )
        self.assertFalse((direct_round / "runtime.json").exists())

        # Restore instrumented_source.sha256, tamper baseline_source.sha256, verify rc == 2 and both digests printed.
        (r1_dir / "recorded" / "instrumented_source.sha256").write_text(actual_mod_sha + "\n", encoding="utf-8")
        wrong_base_sha = "f" * 64
        (r1_dir / "recorded" / "baseline_source.sha256").write_text(wrong_base_sha + "\n", encoding="utf-8")
        p_base = subprocess.run(
            [sys.executable, "-B", str(REPLAY_BENCH), "--recorded-dir", str(r1_dir / "recorded")],
            env=bench_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
        self.assertEqual(p_base.returncode, 2)
        self.assertIn(
            f"baseline_source sha256 mismatch: expected={wrong_base_sha} actual={_SELFTEST_BASE_SHA}",
            p_base.stderr,
        )


def _make_real_case_test(case_name: str):
    suffix = (
        "_suspected_bug_A4_A6"
        if case_name == "wpprofile_h20_handoff_no_window_samples"
        else ""
    )
    test_name = f"test_real_{case_name}{suffix}"

    def _test(self: WpProfileReplayTest):
        self._check_case_dir(CASES / case_name, self._root / f"real_{case_name}", update=UPDATE)

    _test.__name__ = test_name
    _test.__doc__ = f"golden wp.profile replay of tests/golden/cases/{case_name}"
    return _test


for _real_case in discover_cases(CASES):
    _real_test_fn = _make_real_case_test(_real_case)
    setattr(WpProfileReplayTest, _real_test_fn.__name__, _real_test_fn)


if __name__ == "__main__":
    unittest.main()
