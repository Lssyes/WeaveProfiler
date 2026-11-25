#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Characterization tests for `weaveprof/api.py` and `weaveprof/cli.py` public entrypoints (B5-G1)."""
from __future__ import annotations

import contextlib
import http.server
import io
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import struct
import subprocess
import sys
import tempfile
import threading
import unittest

sys.dont_write_bytecode = True

# ---------------------------------------------------------------------------
# In-process zero-external-network socket guard (installed BEFORE importing weaveprof)
# ---------------------------------------------------------------------------
_ORIG_SOCKET_CONNECT = socket.socket.connect
_ORIG_SOCKET_CONNECT_EX = socket.socket.connect_ex
_ORIG_CREATE_CONNECTION = socket.create_connection
_ORIG_GETADDRINFO = socket.getaddrinfo
_NON_LOOPBACK_ATTEMPTS: list[str] = []


def _is_loopback_host(host: object) -> bool:
    if not isinstance(host, str):
        return False
    h = host.strip()
    return h in ("127.0.0.1", "::1", "localhost")


def _is_ip_literal(host: object) -> bool:
    if not isinstance(host, str):
        return False
    try:
        ipaddress.ip_address(host.strip())
        return True
    except ValueError:
        return False


def _guarded_connect(self: socket.socket, address: object) -> object:
    if isinstance(address, tuple) and len(address) >= 1:
        host = address[0]
        if not _is_loopback_host(host):
            _NON_LOOPBACK_ATTEMPTS.append(f"connect:{address!r}")
            raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _ORIG_SOCKET_CONNECT(self, address)


def _guarded_connect_ex(self: socket.socket, address: object) -> int:
    if isinstance(address, tuple) and len(address) >= 1:
        host = address[0]
        if not _is_loopback_host(host):
            _NON_LOOPBACK_ATTEMPTS.append(f"connect_ex:{address!r}")
            raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _ORIG_SOCKET_CONNECT_EX(self, address)


def _guarded_create_connection(address: tuple[str, int], *args: object, **kwargs: object) -> socket.socket:
    host = address[0] if isinstance(address, tuple) and address else ""
    if not _is_loopback_host(host):
        _NON_LOOPBACK_ATTEMPTS.append(f"create_connection:{address!r}")
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _ORIG_CREATE_CONNECTION(address, *args, **kwargs)


def _guarded_getaddrinfo(host: object, port: object, *args: object, **kwargs: object) -> list[tuple]:
    if not (_is_loopback_host(host) or _is_ip_literal(host)):
        _NON_LOOPBACK_ATTEMPTS.append(f"getaddrinfo:{host!r}:{port!r}")
        raise OSError(f"blocked by test guard: DNS lookup for {host!r}")
    return _ORIG_GETADDRINFO(host, port, *args, **kwargs)


socket.socket.connect = _guarded_connect  # type: ignore[assignment]
socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[assignment]
socket.create_connection = _guarded_create_connection  # type: ignore[assignment]
socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.api import (  # noqa: E402
    _Session,
    identify,
    locate,
    profile,
    profile_for_agent,
    validate,
    verify,
    wrap,
)
from weaveprof.backend import VerificationError  # noqa: E402
from weaveprof.cli import main as cli_main  # noqa: E402
from weaveprof.contract import normalize_probe_spec  # noqa: E402
from weaveprof.harness import verify_seal  # noqa: E402


FIXED_HMAC_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

SITECUSTOMIZE_GUARD_SOURCE = """\
import ipaddress
import socket

_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_create_connection = socket.create_connection
_orig_getaddrinfo = socket.getaddrinfo


def _is_loopback(h):
    return isinstance(h, str) and h.strip() in ("127.0.0.1", "::1", "localhost")


def _is_ip(h):
    if not isinstance(h, str):
        return False
    try:
        ipaddress.ip_address(h.strip())
        return True
    except ValueError:
        return False


def _g_connect(self, address):
    if isinstance(address, tuple) and address and not _is_loopback(address[0]):
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _orig_connect(self, address)


def _g_connect_ex(self, address):
    if isinstance(address, tuple) and address and not _is_loopback(address[0]):
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _orig_connect_ex(self, address)


def _g_create_conn(address, *args, **kwargs):
    if isinstance(address, tuple) and address and not _is_loopback(address[0]):
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _orig_create_connection(address, *args, **kwargs)


def _g_getaddrinfo(host, port, *args, **kwargs):
    if not (_is_loopback(host) or _is_ip(host)):
        raise OSError(f"blocked by test guard: DNS lookup for {host!r}")
    return _orig_getaddrinfo(host, port, *args, **kwargs)


socket.socket.connect = _g_connect
socket.socket.connect_ex = _g_connect_ex
socket.create_connection = _g_create_conn
socket.getaddrinfo = _g_getaddrinfo
"""

# 112-byte inline Python `-c` command copying pre-seeded round inputs into `WEAVEPROF_ROUND_DIR`
BENCH_COPY_SEED_CODE = (
    "import os,shutil;"
    "shutil.copytree(os.environ['WP_SEED_DIR'],os.environ['WEAVEPROF_ROUND_DIR'],dirs_exist_ok=True)"
)

EXPECTED_IDENTIFY_KEYS = {
    "arch",
    "card",
    "kernel_name",
    "language",
    "platform",
    "probes",
    "roles",
    "source_path",
    "source_sha256",
}

EXPECTED_AGENT_REPORT_KEYS = {
    "bottleneck_hint",
    "caliber",
    "causal_opportunities",
    "counterfactual_upper_bound_us",
    "failed",
    "fidelity_gates",
    "global_hw_regime",
    "hw_summary",
    "kernel_us",
    "ledger_path",
    "overhead_pct",
    "overlap_recommendations",
    "perfetto_json_gz",
    "phases",
    "preview_png",
    "refinement_advice",
    "round_diff",
    "schema",
    "seal_errors",
    "seal_path",
    "seal_sha256",
    "seal_verified",
    "stage_breakdown",
    "status",
    "structural_blockers",
    "trace_path",
    "verdict",
}


class _NoopLoopbackHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        body = b'{"error": "unexpected_llm_call"}'
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def _sample_cuda_kernel() -> str:
    return (
        "#include <cuda_runtime.h>\n"
        "__global__ void pipeline(float* out, int n) {\n"
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (idx >= n) return;\n"
        "    tma_load_async(idx);\n"
        "    wgmma_mma_op(out, idx);\n"
        "    store_global(out, idx);\n"
        "}\n"
    )


def _sample_triton_kernel() -> str:
    return (
        "import triton\n"
        "import triton.language as tl\n\n"
        "@triton.jit\n"
        "def triton_vec_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
        "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
        "    x = tl.load(x_ptr + offs)\n"
        "    tl.store(y_ptr + offs, x)\n"
    )


def _populate_seed_dir_for_spec(seed_dir: Path, spec: dict, *, kernel_name: str = "pipeline") -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    norm = normalize_probe_spec(spec, _sample_cuda_kernel())
    ops = ["MOV", "LDG", "HMMA", "IADD", "IMAD", "MUL", "STG", "FADD", "EXIT"]
    sass = f"Function : {kernel_name}\n" + "".join(f"/*{16 * i:04x}*/ {op} ;\n" for i, op in enumerate(ops))
    (seed_dir / "l0.sass").write_text(sass, encoding="utf-8")
    (seed_dir / "inst.sass").write_text(sass, encoding="utf-8")
    (seed_dir / "l0.bin").write_bytes(b"\x01\x02\x03\x04" * 16)
    (seed_dir / "inst.bin").write_bytes(b"\x01\x02\x03\x04" * 16)

    tag_open, tag_end, tag_close = 0x40000000, 0x80000000, 0x08000000
    t0 = 1_000_000
    recs = [(t0, tag_open, 0)]
    tick = t0 + 200
    for _ in range(4):
        for p in norm["probes"]:
            eid = int(p["event_id"])
            recs.append((tick, eid, 0))
            recs.append((tick + 12_000, tag_end | eid, 0))
            tick += 12_200
    recs.append((tick + 300, tag_close, 0))

    meta = json.dumps({"gran": 0, "arch": "sm_90a", "backend": "nvidia_ptx"}).encode("utf-8")
    blob = bytearray(b"FILITE02")
    blob.extend(struct.pack("<6I", 1, len(recs), 1, 0, 0, len(meta)))
    blob.extend(meta)
    blob.extend(struct.pack("<1I", len(recs)))
    for rec in recs:
        blob.extend(struct.pack("<QII", *rec))
    (seed_dir / "capture.fi02.bin").write_bytes(blob)

    db = sqlite3.connect(str(seed_dir / "capture.sqlite"))
    try:
        db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, demangledName INTEGER)")
        db.execute("INSERT INTO StringIds VALUES (1, ?)", (kernel_name,))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 1)", (t0 - 1_000, tick + 5_000))
        db.commit()
    finally:
        db.close()

    rt = {
        "kernel_name": kernel_name,
        "baseline_sass": "l0.sass",
        "instrumented_sass": "inst.sass",
        "baseline_spill_bytes": 0,
        "instrumented_spill_bytes": 0,
        "extra_smem_bytes": 0,
        "baseline_output": "l0.bin",
        "instrumented_output": "inst.bin",
        "warmup": 4,
        "timing_protocol": "ABBA",
        "baseline_ns": [42000, 42100, 41950, 42050, 42000, 42020],
        "instrumented_ns": [42750, 42820, 42700, 42790, 42760, 42770],
    }
    (seed_dir / "runtime.json").write_text(json.dumps(rt, indent=2) + "\n", encoding="utf-8")


class ApiAndCliCharacterizationTests(unittest.TestCase):
    _ENV_SAVE_KEYS = (
        "HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "MPLCONFIGDIR",
        "WEAVEPROF_SESSIONS_DIR",
        "WEAVEPROF_CACHE",
        "WEAVEPROF_LOCK_DIR",
        "WEAVEPROF_WORKDIR",
        "TMPDIR",
        "PYTHONPATH",
        "WEAVEPROF_SUPERVISOR_HMAC_KEY",
        "WEAVEPROF_SUPERVISOR_KEY_FILE",
        "QWEN_PROJECT_DIR",
        "QWEN_SESSION_ID",
        "WEAVEPROF_LOCK_PROVIDER",
        "WEAVEPROF_KEEP_SHADOW",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "WEAVEPROF_BASE_URL",
        "OPENAI_BASE_URL",
        "WEAVEPROF_API_KEY",
        "OPENAI_API_KEY",
        "WEAVEPROF_MODEL",
        "OPENAI_MODEL",
        "WEAVEPROF_SEND_REASONING_EFFORT",
        "DASHSCOPE_API_KEY",
        "BAILIAN_API_KEY",
        "FI_SAMPLE",
        "WP_SAMPLE",
        "WEAVEPROF_SAMPLE",
        "WP_SEED_DIR",
        "PYTHONDONTWRITEBYTECODE",
    )

    @classmethod
    def setUpClass(cls) -> None:
        _NON_LOOPBACK_ATTEMPTS.clear()
        cls._saved_cwd = os.getcwd()
        cls._saved_tempdir = tempfile.tempdir
        cls._saved_env = {k: os.environ.get(k) for k in cls._ENV_SAVE_KEYS}

        cls._iso_root = Path(tempfile.mkdtemp(prefix="wp_b5_g1_iso_")).resolve()
        for sub in (
            "home",
            "xdg_cache",
            "xdg_config",
            "mpl",
            "sessions",
            "cache",
            "locks",
            "workdir",
            "tmp",
            "cwd",
            "netguard",
        ):
            (cls._iso_root / sub).mkdir(parents=True, exist_ok=True)

        cls._guard_dir = cls._iso_root / "netguard"
        (cls._guard_dir / "sitecustomize.py").write_text(SITECUSTOMIZE_GUARD_SOURCE, encoding="utf-8")

        cls._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _NoopLoopbackHandler)
        cls._port = cls._server.server_address[1]
        cls._server_thread = threading.Thread(target=cls._server.serve_forever, daemon=True)
        cls._server_thread.start()

        os.environ["HOME"] = str(cls._iso_root / "home")
        os.environ["XDG_CACHE_HOME"] = str(cls._iso_root / "xdg_cache")
        os.environ["XDG_CONFIG_HOME"] = str(cls._iso_root / "xdg_config")
        os.environ["MPLCONFIGDIR"] = str(cls._iso_root / "mpl")
        os.environ["WEAVEPROF_SESSIONS_DIR"] = str(cls._iso_root / "sessions")
        os.environ["WEAVEPROF_CACHE"] = str(cls._iso_root / "cache")
        os.environ["WEAVEPROF_LOCK_DIR"] = str(cls._iso_root / "locks")
        os.environ["WEAVEPROF_WORKDIR"] = str(cls._iso_root / "workdir")
        os.environ["TMPDIR"] = str(cls._iso_root / "tmp")
        os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = FIXED_HMAC_KEY
        os.environ["WEAVEPROF_BASE_URL"] = f"http://127.0.0.1:{cls._port}"
        os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{cls._port}"
        os.environ["WEAVEPROF_API_KEY"] = "test-dummy-key"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

        existing_pp = [p for p in (cls._saved_env.get("PYTHONPATH") or "").split(os.pathsep) if p]
        os.environ["PYTHONPATH"] = os.pathsep.join([str(cls._guard_dir), str(ROOT), *existing_pp])

        for unset_key in (
            "WEAVEPROF_SUPERVISOR_KEY_FILE",
            "QWEN_PROJECT_DIR",
            "QWEN_SESSION_ID",
            "WEAVEPROF_LOCK_PROVIDER",
            "WEAVEPROF_KEEP_SHADOW",
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "OPENAI_API_KEY",
            "WEAVEPROF_MODEL",
            "OPENAI_MODEL",
            "WEAVEPROF_SEND_REASONING_EFFORT",
            "DASHSCOPE_API_KEY",
            "BAILIAN_API_KEY",
            "FI_SAMPLE",
            "WP_SAMPLE",
            "WEAVEPROF_SAMPLE",
            "WP_SEED_DIR",
        ):
            os.environ.pop(unset_key, None)

        tempfile.tempdir = str(cls._iso_root / "tmp")
        os.chdir(cls._iso_root / "cwd")

        # Subprocess guard proof (`004.md` line 20): external IP connection must be refused by `sitecustomize.py`
        probe_proc = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "import socket; socket.create_connection(('93.184.216.34', 443), timeout=1.0)",
            ],
            cwd=str(cls._iso_root / "cwd"),
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if probe_proc.returncode == 0 or "blocked by test guard: ('93.184.216.34', 443)" not in probe_proc.stderr:
            raise AssertionError(f"Subprocess sitecustomize guard failed: rc={probe_proc.returncode}, stderr={probe_proc.stderr!r}")
        cls.subproc_guard_proof = probe_proc.stderr.strip().splitlines()[-1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server.shutdown()
        cls._server.server_close()
        cls._server_thread.join(timeout=5.0)
        os.chdir(cls._saved_cwd)
        tempfile.tempdir = cls._saved_tempdir
        for k, val in cls._saved_env.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
        shutil.rmtree(cls._iso_root, ignore_errors=True)
        if _NON_LOOPBACK_ATTEMPTS:
            raise AssertionError(f"Unexpected non-loopback network attempts: {_NON_LOOPBACK_ATTEMPTS}")

    def _new_case_dir(self, tag: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=f"{tag}_", dir=str(self._iso_root / "tmp"))).resolve()

    def test_identify_and_locate_offline_cache_and_error_paths(self) -> None:
        case_dir = self._new_case_dir("api_identify")
        src = case_dir / "pipeline.cu"
        src.write_text(_sample_cuda_kernel(), encoding="utf-8")
        out_probe = case_dir / "probe.json"

        self.assertIs(locate, identify)
        self.assertIs(validate, verify)

        doc = identify(src, out=out_probe, card="h20", offline_heuristic=True)
        self.assertEqual(set(doc.keys()), EXPECTED_IDENTIFY_KEYS)
        self.assertEqual(doc["card"], "h20")
        self.assertEqual(doc["arch"], "sm_90a")
        self.assertEqual(doc["platform"], "nv")
        self.assertEqual(doc["language"], "cuda_cpp")
        self.assertEqual(doc["kernel_name"], "pipeline")
        self.assertEqual(doc["roles"], {"worker": "all"})
        self.assertEqual(
            doc["probes"],
            [
                {"id": "load", "begin": {"role": "worker", "line": 5}, "end": {"role": "worker", "line": 5}},
                {"id": "mma", "begin": {"role": "worker", "line": 6}, "end": {"role": "worker", "line": 6}},
                {"id": "store", "begin": {"role": "worker", "line": 7}, "end": {"role": "worker", "line": 7}},
            ],
        )
        self.assertTrue(out_probe.is_file())
        cached_file = self._iso_root / "cache" / doc["source_sha256"] / "probe.json"
        self.assertTrue(cached_file.is_file())

        # Second call via `locate` without `offline_heuristic=True` hits cache without touching LLM
        rebound = locate(src, card="mi308x")
        self.assertEqual(rebound["card"], "mi308x")
        self.assertEqual(rebound["arch"], "gfx942")
        self.assertEqual(rebound["platform"], "amd")
        self.assertEqual(rebound["language"], "hip_cpp")
        self.assertEqual(rebound["source_sha256"], doc["source_sha256"])

        # Feedback without prior cache/out raises ValueError
        uncached_src = case_dir / "uncached.cu"
        uncached_src.write_text("__global__ void k(float* p) {\n    p[0] = 1.0f;\n}\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ref_ctx:
            identify(uncached_src, feedback={"status": "FAIL", "failed": ["overhead"]})
        self.assertEqual(
            str(ref_ctx.exception),
            "Refinement needs the original placement in the source cache or output probe",
        )

        # `safe_products` collision when `out == source`
        with self.assertRaises(ValueError) as dup_ctx:
            identify(src, out=src, offline_heuristic=True)
        self.assertEqual(
            str(dup_ctx.exception),
            "Source, probe, feedback and output must have distinct paths",
        )

    def test_session_with_context_and_wrap_decorator_offline_paths(self) -> None:
        case_dir = self._new_case_dir("session_ctx")

        # 1) `profile(..., bench=None)` returns `_Session`; spin-loop probe fails `_prepare(1)` in `__enter__`
        spin_src = case_dir / "spin.cu"
        spin_src.write_text(
            "#include <cuda_runtime.h>\n"
            "__global__ void spin_k(volatile int* flag, float* out) {\n"
            "    while (*flag == 0) {\n"
            "        flag += 0;\n"
            "    }\n"
            "    out[0] = 1.0f;\n"
            "}\n",
            encoding="utf-8",
        )
        bad_probe = {
            "card": "h20",
            "roles": {"worker": "all"},
            "probes": [{"id": "wait", "begin": {"role": "worker", "line": 4}, "end": {"role": "worker", "line": 4}}],
        }
        sess_bad = profile(spin_src, out=case_dir / "spin_out.png", probe=bad_probe, card="h20", bench=None)
        self.assertIsInstance(sess_bad, _Session)
        with self.assertRaises(VerificationError) as spin_ctx:
            with sess_bad:
                pass
        self.assertEqual(spin_ctx.exception.feedback["status"], "FAIL")
        self.assertEqual(spin_ctx.exception.feedback["failed"], ["assembly"])
        self.assertEqual(
            spin_ctx.exception.feedback["metrics"]["policy_violations"][0]["code"],
            "SPIN_LOOP_PROBE_POLICY",
        )
        self.assertTrue((spin_ctx.exception.directory / "seal.json").is_file())

        # 2) Valid Triton kernel: `_Session.__enter__` and `@wrap` prepare `round-1` then hit `replay.capture` (`import torch`)
        triton_src = case_dir / "vec_triton.py"
        triton_src.write_text(_sample_triton_kernel(), encoding="utf-8")
        triton_spec = identify(triton_src, card="h20", offline_heuristic=True)
        sess_ok = profile(
            triton_src,
            out=case_dir / "triton_out.png",
            probe=triton_spec,
            card="h20",
            bench=None,
        )
        with self.assertRaises(ModuleNotFoundError) as enter_ctx:
            with sess_ok:
                pass
        self.assertEqual(enter_ctx.exception.name, "torch")
        self.assertIsNotNone(sess_ok.prepared)
        self.assertTrue((sess_ok.prepared[0] / "names.json").is_file())

        calls: list[int] = []

        @wrap(source=triton_src, probe=triton_spec, out=case_dir / "wrap_out.png", card="h20")
        def launch_vec(x: int) -> int:
            calls.append(x)
            return x + 1

        with self.assertRaises(ModuleNotFoundError) as wrap_ctx:
            launch_vec(7)
        self.assertEqual(wrap_ctx.exception.name, "torch")
        self.assertEqual(calls, [])

    def test_verify_profile_and_profile_for_agent_end_to_end(self) -> None:
        case_dir = self._new_case_dir("api_e2e")
        proj = case_dir / "proj"
        proj.mkdir()
        src = proj / "pipeline.cu"
        src.write_text(_sample_cuda_kernel(), encoding="utf-8")
        spec = identify(src, out=case_dir / "id_probe.json", card="h20", offline_heuristic=True)

        seed_dir = case_dir / "seed"
        _populate_seed_dir_for_spec(seed_dir, spec)
        prev_seed = os.environ.get("WP_SEED_DIR")
        os.environ["WP_SEED_DIR"] = str(seed_dir)
        try:
            bench_ok = [sys.executable, "-c", BENCH_COPY_SEED_CODE]

            # 1) `verify` missing `source_path` error path
            spec_no_src = {"roles": spec["roles"], "probes": spec["probes"]}
            with self.assertRaises(ValueError) as no_src_ctx:
                verify(spec_no_src, out=case_dir / "nosrc.png", bench=bench_ok)
            self.assertEqual(str(no_src_ctx.exception), "This placement has no source_path. Supply --source")

            # 2) `verify` PASS path
            ver_out = case_dir / "ver_out" / "verified.perfetto-trace"
            ver_out.parent.mkdir(parents=True, exist_ok=True)
            ver_res = verify(case_dir / "id_probe.json", out=ver_out, bench=bench_ok)
            self.assertEqual(ver_res["status"], "PASS")
            self.assertEqual(ver_res["failed"], [])
            self.assertEqual(ver_res["fires"], {"load": 4, "mma": 4, "store": 4})
            self.assertTrue(ver_out.is_file())
            self.assertIs(verify_seal(ver_res["seal_path"], require_pass=True)["ok"], True)

            # 3) `profile` PASS path
            prof_out = case_dir / "prof_out" / "profiled.perfetto-trace"
            prof_out.parent.mkdir(parents=True, exist_ok=True)
            prof_res = profile(src, out=prof_out, probe=spec, card="h20", bench=bench_ok)
            self.assertEqual(prof_res["status"], "PASS")
            self.assertEqual(prof_res["failed"], [])
            self.assertTrue(Path(prof_res["ledger_path"]).is_file())

            # 4) `profile_for_agent` PASS path with `run_dir`
            agent_run_dir = case_dir / "agent_pass"
            agent_run_dir.mkdir()
            rep_pass = profile_for_agent(
                src,
                bench=bench_ok,
                run_dir=agent_run_dir,
                probe=spec,
                card="h20",
            )
            self.assertEqual(set(rep_pass.keys()), EXPECTED_AGENT_REPORT_KEYS)
            self.assertEqual(rep_pass["schema"], "weaveprof.agent-report/v1")
            self.assertEqual(rep_pass["status"], "PASS")
            self.assertIs(rep_pass["seal_verified"], True)
            self.assertEqual(rep_pass["seal_errors"], [])
            self.assertEqual(rep_pass["failed"], [])
            self.assertEqual(rep_pass["overhead_pct"], 1.797)
            self.assertEqual(rep_pass["trace_path"], str((agent_run_dir / "overview.png").resolve()))

            # 5) `profile_for_agent` FAIL path (captures `VerificationError` and includes `round_dir`)
            agent_fail_dir = case_dir / "agent_fail"
            agent_fail_dir.mkdir()
            rep_fail = profile_for_agent(
                src,
                bench=[sys.executable, "-c", "import sys; sys.exit(4)"],
                run_dir=agent_fail_dir,
                probe=spec,
                card="h20",
            )
            self.assertEqual(rep_fail["schema"], "weaveprof.agent-report/v1")
            self.assertEqual(rep_fail["status"], "FAIL")
            self.assertEqual(sorted(rep_fail["failed"]), ["assembly", "output", "overhead", "pairing"])
            self.assertEqual(Path(rep_fail["round_dir"]).name, "round-1")
        finally:
            if prev_seed is None:
                os.environ.pop("WP_SEED_DIR", None)
            else:
                os.environ["WP_SEED_DIR"] = prev_seed

    def test_cli_subcommands_inproc_and_subprocess(self) -> None:
        case_dir = self._new_case_dir("cli_cmds")
        proj = case_dir / "proj"
        proj.mkdir()
        src = proj / "pipeline.cu"
        src.write_text(_sample_cuda_kernel(), encoding="utf-8")

        # 1) `identify` in-process
        id_probe = case_dir / "cli_id_probe.json"
        buf_id = io.StringIO()
        with contextlib.redirect_stdout(buf_id):
            rc_id = cli_main(["identify", str(src), "-o", str(id_probe), "--card", "h20", "--offline-heuristic"])
        self.assertEqual(rc_id, 0)
        self.assertEqual(buf_id.getvalue().strip(), str(id_probe))
        id_doc = json.loads(id_probe.read_text(encoding="utf-8"))
        self.assertEqual(id_doc["card"], "h20")
        self.assertEqual(id_doc["kernel_name"], "pipeline")

        # 2) `locate` via subprocess (`python -B -m weaveprof locate ...` with `sitecustomize.py` active)
        loc_probe = case_dir / "cli_loc_probe.json"
        proc_loc = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "weaveprof",
                "locate",
                str(src),
                "-o",
                str(loc_probe),
                "--card",
                "mi308x",
                "--offline-heuristic",
            ],
            cwd=str(case_dir),
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc_loc.returncode, 0, proc_loc.stderr)
        self.assertEqual(proc_loc.stdout.strip(), str(loc_probe))
        loc_doc = json.loads(loc_probe.read_text(encoding="utf-8"))
        self.assertEqual(loc_doc["card"], "mi308x")
        self.assertEqual(loc_doc["arch"], "gfx942")
        self.assertEqual(loc_doc["platform"], "amd")

        # 3) `verify`, `validate`, `profile`, `seal-check`, `doctor --no-gpu`, and CLI failure exit code
        seed_dir = case_dir / "seed"
        _populate_seed_dir_for_spec(seed_dir, id_doc)
        prev_seed = os.environ.get("WP_SEED_DIR")
        os.environ["WP_SEED_DIR"] = str(seed_dir)
        try:
            ver_dir = case_dir / "ver"
            ver_dir.mkdir()
            ver_trace = ver_dir / "cli_ver.perfetto-trace"

            # Pin BASE CLI bug (`weaveprof/cli.py:20,40,104-106`):
            # - Positional before options makes `nargs=argparse.REMAINDER` swallow `--source` into `bench` (`bench[0] != "--"`).
            # - Options before positional with a single `--` makes `argparse` strip `--` as end-of-options (`bench[0] != "--"`).
            for bad_argv in (
                [
                    "verify",
                    str(id_probe),
                    "--source",
                    str(src),
                    "-o",
                    str(ver_trace),
                    "--",
                    sys.executable,
                    "-c",
                    BENCH_COPY_SEED_CODE,
                ],
                [
                    "verify",
                    "--source",
                    str(src),
                    "-o",
                    str(ver_trace),
                    str(id_probe),
                    "--",
                    sys.executable,
                    "-c",
                    BENCH_COPY_SEED_CODE,
                ],
            ):
                err_parse = io.StringIO()
                with contextlib.redirect_stderr(err_parse), self.assertRaises(SystemExit) as parse_exc:
                    cli_main(bad_argv)
                self.assertEqual(parse_exc.exception.code, 2)
                self.assertIn("expected -- <benchmark command>", err_parse.getvalue())

            out_ver = io.StringIO()
            with contextlib.redirect_stdout(out_ver):
                rc_ver = cli_main(
                    [
                        "verify",
                        "--source",
                        str(src),
                        "-o",
                        str(ver_trace),
                        str(id_probe),
                        "--",
                        "--",
                        sys.executable,
                        "-c",
                        BENCH_COPY_SEED_CODE,
                    ]
                )
            self.assertEqual(rc_ver, 0)
            self.assertEqual(out_ver.getvalue().strip(), str(ver_trace))
            self.assertTrue(ver_trace.is_file())

            out_seal = io.StringIO()
            with contextlib.redirect_stdout(out_seal):
                rc_seal = cli_main(["seal-check", f"{ver_trace}.seal.json"])
            self.assertEqual(rc_seal, 0)
            self.assertEqual(json.loads(out_seal.getvalue()), {"ok": True, "errors": []})

            val_dir = case_dir / "val"
            val_dir.mkdir()
            val_png = val_dir / "cli_val.png"
            out_val = io.StringIO()
            with contextlib.redirect_stdout(out_val):
                rc_val = cli_main(
                    [
                        "validate",
                        "--probe",
                        str(id_probe),
                        "--source",
                        str(src),
                        "-o",
                        str(val_png),
                        "--",
                        "--",
                        "--",
                        sys.executable,
                        "-c",
                        BENCH_COPY_SEED_CODE,
                    ]
                )
            self.assertEqual(rc_val, 0)
            self.assertEqual(val_png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

            prof_dir = case_dir / "prof"
            prof_dir.mkdir()
            prof_trace = prof_dir / "cli_prof.perfetto-trace"
            out_prof = io.StringIO()
            with contextlib.redirect_stdout(out_prof):
                rc_prof = cli_main(
                    [
                        "profile",
                        "--probe",
                        str(id_probe),
                        "--card",
                        "h20",
                        "-o",
                        str(prof_trace),
                        str(src),
                        "--",
                        "--",
                        sys.executable,
                        "-c",
                        BENCH_COPY_SEED_CODE,
                    ]
                )
            self.assertEqual(rc_prof, 0)
            self.assertTrue(prof_trace.is_file())

            # Failing bench returns `rc == 1` and writes VerificationError message to stderr
            fail_dir = case_dir / "prof_fail"
            fail_dir.mkdir()
            err_buf = io.StringIO()
            with contextlib.redirect_stderr(err_buf):
                rc_fail = cli_main(
                    [
                        "profile",
                        "--probe",
                        str(id_probe),
                        "--card",
                        "h20",
                        "-o",
                        str(fail_dir / "fail.perfetto-trace"),
                        str(src),
                        "--",
                        "--",
                        sys.executable,
                        "-c",
                        "import sys; sys.exit(3)",
                    ]
                )
            self.assertEqual(rc_fail, 1)
            self.assertTrue(err_buf.getvalue().startswith("Verification failed in "))

            # `causal` missing input returns `rc == 1`
            causal_err = io.StringIO()
            with contextlib.redirect_stderr(causal_err):
                rc_causal_missing = cli_main(["causal", str(case_dir / "missing_trace.json")])
            self.assertEqual(rc_causal_missing, 1)
            self.assertTrue(causal_err.getvalue().startswith("Input not found: "))

            # `doctor --no-gpu` reflects isolated loopback LLM config and local_file lock dir
            doc_buf = io.StringIO()
            with contextlib.redirect_stdout(doc_buf):
                rc_doc = cli_main(["doctor", "--no-gpu"])
            self.assertEqual(rc_doc, 0)
            doc_rep = json.loads(doc_buf.getvalue())
            self.assertEqual(doc_rep["schema"], "weaveprof.doctor/v1")
            self.assertEqual(doc_rep["gpus"], {"checked": False, "nvidia": None, "amd": None})
            self.assertEqual(
                doc_rep["llm"],
                {
                    "endpoint_mode": "custom_env",
                    "endpoint_url": f"http://127.0.0.1:{self._port}/v1/chat/completions",
                    "default_model": "qwen4-max",
                    "openai_base_url_set": True,
                    "api_key_detected": True,
                    "api_key_source": "WEAVEPROF_API_KEY",
                },
            )
            self.assertIs(doc_rep["ready_for_offline_tests"], True)
            self.assertIs(doc_rep["ready_for_gpu_capture"], False)
        finally:
            if prev_seed is None:
                os.environ.pop("WP_SEED_DIR", None)
            else:
                os.environ["WP_SEED_DIR"] = prev_seed


if __name__ == "__main__":
    unittest.main()
