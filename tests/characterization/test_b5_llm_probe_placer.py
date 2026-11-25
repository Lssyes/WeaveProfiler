#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Characterization tests for `weaveprof/pipeline/scan/llm_probe_placer.py` offline & loopback layers (B5-G2..G4, G7)."""
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
    return host.strip() in ("127.0.0.1", "::1", "localhost")


def _is_ip_literal(host: object) -> bool:
    if not isinstance(host, str):
        return False
    try:
        ipaddress.ip_address(host.strip())
        return True
    except ValueError:
        return False


def _guarded_connect(self: socket.socket, address: object) -> object:
    if isinstance(address, tuple) and len(address) >= 1 and not _is_loopback_host(address[0]):
        _NON_LOOPBACK_ATTEMPTS.append(f"connect:{address!r}")
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _ORIG_SOCKET_CONNECT(self, address)


def _guarded_connect_ex(self: socket.socket, address: object) -> int:
    if isinstance(address, tuple) and len(address) >= 1 and not _is_loopback_host(address[0]):
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

from weaveprof.pipeline.scan import llm_probe_placer as placer  # noqa: E402


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


class _LoopbackLLMState:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.scripted_responses: list[dict] = []


class _LoopbackLLMHandler(http.server.BaseHTTPRequestHandler):
    state: _LoopbackLLMState

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length).decode("utf-8") if length else "{}"
        payload = json.loads(raw_body)
        self.state.requests.append(
            {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "payload": payload,
            }
        )
        spec = (
            self.state.scripted_responses.pop(0)
            if self.state.scripted_responses
            else {"status": 500, "ctype": "text/plain", "body": b"no_scripted_response"}
        )
        status = int(spec["status"])
        ctype = str(spec["ctype"])
        body = spec["body"]
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


SAMPLE_CUDA_KERNEL = (
    "#include <cuda_runtime.h>\n"
    "__global__ void gemm_pipe(float* out, int n) {\n"
    "    int tid = threadIdx.x;\n"
    "    if (tid >= n) return;\n"
    "    tma_load_tile(tid);\n"
    "    mbar.wait(0);\n"
    "    for (int k = 0; k < 16; ++k) {\n"
    "        wgmma_mma_tile(out, k);\n"
    "    }\n"
    "    store_global_tile(out, tid);\n"
    "}\n"
)

SAMPLE_TRITON_KERNEL = (
    "import triton\n"
    "import triton.language as tl\n\n"
    "@triton.jit\n"
    "def triton_gemm(a_ptr, b_ptr, c_ptr, BLOCK: tl.constexpr):\n"
    "    offs = tl.arange(0, BLOCK)\n"
    "    a = tl.load(\n"
    "        a_ptr + offs\n"
    "    )\n"
    "    acc = tl.dot(a, a)\n"
    "    tl.store(c_ptr + offs, acc)\n"
)

SAMPLE_CUTEDSL_MULTI_DEC = (
    "import cutlass.cute as cute\n\n"
    "@cute.jit\n"
    "def helper_sub(x):\n"
    "    cute.copy(x, x)\n\n"
    "@cute.kernel\n"
    "def entry_cute_mm(gA, gB, gC):\n"
    "    cute.load(gA, gB)\n"
    "    cute.gemm(gA, gB, gC)\n"
    "    cute.store(gB, gC)\n"
)


class LlmProbePlacerCharacterizationTests(unittest.TestCase):
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
        "PYTHONDONTWRITEBYTECODE",
    )

    @classmethod
    def setUpClass(cls) -> None:
        _NON_LOOPBACK_ATTEMPTS.clear()
        cls._saved_cwd = os.getcwd()
        cls._saved_tempdir = tempfile.tempdir
        cls._saved_env = {k: os.environ.get(k) for k in cls._ENV_SAVE_KEYS}

        cls._iso_root = Path(tempfile.mkdtemp(prefix="wp_b5_placer_iso_")).resolve()
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

        cls._llm_state = _LoopbackLLMState()
        handler_cls = type("BoundLoopbackHandler", (_LoopbackLLMHandler,), {"state": cls._llm_state})
        cls._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
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
        ):
            os.environ.pop(unset_key, None)

        tempfile.tempdir = str(cls._iso_root / "tmp")
        os.chdir(cls._iso_root / "cwd")

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

    def test_offline_probe_spec_branches_and_error_paths(self) -> None:
        # 1) CUDA C++ with load / wait / mma / store primitives
        cpp_spec = placer._offline_probe_spec(SAMPLE_CUDA_KERNEL, "cuda_cpp")
        self.assertEqual(
            cpp_spec,
            {
                "roles": {"worker": "all"},
                "kernel_name": "gemm_pipe",
                "probes": [
                    {"id": "load", "begin": {"role": "worker", "line": 5}, "end": {"role": "worker", "line": 5}},
                    {"id": "wait", "begin": {"role": "worker", "line": 6}, "end": {"role": "worker", "line": 6}},
                    {"id": "mma", "begin": {"role": "worker", "line": 8}, "end": {"role": "worker", "line": 8}},
                    {"id": "store", "begin": {"role": "worker", "line": 10}, "end": {"role": "worker", "line": 10}},
                ],
            },
        )

        # 2) C++ fallback to `"compute"` on last instrumentable statement when no primitive cues match
        plain_cpp = (
            "__global__ void vec_scale(float* x, int n) {\n"
            "    if (threadIdx.x >= n) return;\n"
            "    float v = x[threadIdx.x];\n"
            "    x[threadIdx.x] = v * 2.0f;\n"
            "    return;\n"
            "}\n"
        )
        plain_spec = placer._offline_probe_spec(plain_cpp, "cuda_cpp")
        self.assertEqual(
            plain_spec,
            {
                "roles": {"worker": "all"},
                "kernel_name": "vec_scale",
                "probes": [
                    {"id": "compute", "begin": {"role": "worker", "line": 4}, "end": {"role": "worker", "line": 4}},
                ],
            },
        )

        # 3) Triton multi-line AST statement span & CuTeDSL `@cute.kernel` preference over `@cute.jit`
        tri_spec = placer._offline_probe_spec(SAMPLE_TRITON_KERNEL, "triton")
        self.assertEqual(
            tri_spec,
            {
                "roles": {"worker": "all"},
                "kernel_name": "triton_gemm",
                "probes": [
                    {"id": "load", "begin": {"role": "worker", "line": 7}, "end": {"role": "worker", "line": 9}},
                    {"id": "mma", "begin": {"role": "worker", "line": 10}, "end": {"role": "worker", "line": 10}},
                    {"id": "store", "begin": {"role": "worker", "line": 11}, "end": {"role": "worker", "line": 11}},
                ],
            },
        )
        cute_spec = placer._offline_probe_spec(SAMPLE_CUTEDSL_MULTI_DEC, "cutedsl")
        self.assertEqual(cute_spec["kernel_name"], "entry_cute_mm")
        self.assertEqual([p["id"] for p in cute_spec["probes"]], ["load", "mma", "store"])

        # 4) Three ValueError branches
        with self.assertRaises(ValueError) as py_err:
            placer._offline_probe_spec("def host_fn(x):\n    return x + 1\n", "triton")
        self.assertEqual(str(py_err.exception), "No decorated GPU kernel in this Python source")

        with self.assertRaises(ValueError) as cpp_err:
            placer._offline_probe_spec("void host_only(float* x) { x[0] = 1.0f; }\n", "cuda_cpp")
        self.assertEqual(str(cpp_err.exception), "No __global__ GPU kernel in this C++ source")

        with self.assertRaises(ValueError) as empty_err:
            placer._offline_probe_spec("__global__ void empty_k() {\n    return;\n}\n", "cuda_cpp")
        self.assertEqual(str(empty_err.exception), "Offline identification found no instrumentable kernel statement")

    def test_identify_source_offline_refinement_and_python_owner_inference(self) -> None:
        case_dir = self._new_case_dir("identify_source")
        src_cu = case_dir / "gemm_pipe.cu"
        src_cu.write_text(SAMPLE_CUDA_KERNEL, encoding="utf-8")

        spec0 = placer.identify_source(src_cu, card="h20", offline_heuristic=True)
        self.assertEqual(spec0["card"], "h20")
        self.assertEqual(spec0["arch"], "sm_90a")
        self.assertEqual(spec0["platform"], "nv")
        self.assertEqual(spec0["kernel_name"], "gemm_pipe")
        self.assertEqual(spec0["source_sha256"], placer.sha256_file(src_cu))

        # SHA mismatch on `previous`
        bad_prev = dict(spec0, source_sha256="0" * 64)
        with self.assertRaises(ValueError) as sha_ctx:
            placer.identify_source(src_cu, card="h20", previous=bad_prev, offline_heuristic=True)
        self.assertEqual(str(sha_ctx.exception), "Previous probe.json is bound to different source bytes")

        # Deterministic loop hoist on overhead failure (`enclosing_loop=[7, 9]`, `mma` at line 8 -> `[7, 9]`)
        hoisted = placer.identify_source(
            src_cu,
            card="h20",
            offline_heuristic=True,
            previous=spec0,
            feedback={
                "status": "FAIL",
                "failed": ["overhead"],
                "fires": {"load": 1, "wait": 1, "mma": 16, "store": 1},
                "enclosing_loop": [7, 9],
            },
        )
        mma_probe = next(p for p in hoisted["probes"] if p["id"] == "mma")
        self.assertEqual(mma_probe["begin"], {"role": "worker", "line": 7})
        self.assertEqual(mma_probe["end"], {"role": "worker", "line": 9})

        # `status == "PASS"` returns compact `previous` without calling LLM
        kept = placer.identify_source(
            src_cu,
            card="h20",
            offline_heuristic=True,
            previous=spec0,
            feedback={"status": "PASS", "failed": []},
        )
        self.assertEqual(kept["kernel_name"], "gemm_pipe")

        # `pairing` failure cannot use deterministic loop hoist -> ValueError in `offline_heuristic=True`
        with self.assertRaises(ValueError) as pair_ctx:
            placer.identify_source(
                src_cu,
                card="h20",
                offline_heuristic=True,
                previous=spec0,
                feedback={
                    "status": "FAIL",
                    "failed": ["pairing"],
                    "fires": {"mma": 16},
                    "enclosing_loop": [7, 9],
                },
            )
        self.assertEqual(str(pair_ctx.exception), "No deterministic loop refinement for this feedback")

        # `offline_heuristic=False` with `llm_caller` omitting `kernel_name` infers Python AST owner function
        src_tri = case_dir / "tri.py"
        src_tri.write_text(SAMPLE_TRITON_KERNEL, encoding="utf-8")

        def anon_llm(model: str, messages: list[dict], **kwargs: object) -> dict:
            return {
                "ok": True,
                "model": "qwen4-max",
                "reasoning_effort": "max",
                "elapsed_s": 0.12,
                "usage": {"prompt_tokens": 50, "completion_tokens": 20, "reasoning_tokens": 5, "total_tokens": 70},
                "content": json.dumps(
                    {
                        "roles": {"worker": "all"},
                        "probes": [{"id": "mma", "begin": {"role": "worker", "line": 10}, "end": {"role": "worker", "line": 10}}],
                    }
                ),
            }

        inferred = placer.identify_source(src_tri, card="h20", offline_heuristic=False, llm_caller=anon_llm)
        self.assertEqual(inferred["kernel_name"], "triton_gemm")
        self.assertEqual(inferred["language"], "triton")

    def test_kb_sanitize_spin_loops_repair_prompt_and_direct_ablation_modes(self) -> None:
        case_dir = self._new_case_dir("kb_and_modes")
        src_cu = case_dir / "gemm_pipe.cu"
        src_cu.write_text(SAMPLE_CUDA_KERNEL, encoding="utf-8")

        # 1) `retrieve_route_kb` across HIP gfx942 and Triton AMD/NV
        kb_hip = placer.retrieve_route_kb("hip_cpp", "mi308x", "gfx942", SAMPLE_CUDA_KERNEL)
        self.assertEqual(kb_hip["route"]["platform"], "amd")
        self.assertEqual(kb_hip["provenance"]["card_file"], "configs/cards/mi308x.json")
        self.assertEqual(len(kb_hip["provenance"]["extra_notes"]), 1)
        self.assertTrue(kb_hip["provenance"]["extra_notes"][0].startswith("gfx942 recorder KB: recorder_gfx942.json"))

        kb_tri_amd = placer.retrieve_route_kb("triton", "mi308x", "gfx942", SAMPLE_TRITON_KERNEL)
        kb_tri_nv = placer.retrieve_route_kb("triton", "h20", "sm_90a", SAMPLE_TRITON_KERNEL)
        self.assertEqual(kb_tri_amd["provenance"]["patterns_file"], "configs/languages/triton/amd_patterns.json")
        self.assertEqual(kb_tri_nv["provenance"]["patterns_file"], "configs/languages/triton/nvidia_patterns.json")

        # 2) `sanitize_and_bind_probe_spec`: snippet line resolution, async completion_wait_snippet, semantic warning
        bound = placer.sanitize_and_bind_probe_spec(
            {
                "kernel_name": "gemm_pipe",
                "operator_family": "gemm",
                "probes": [
                    {
                        "probe_id": "async_ld",
                        "semantic_id": "load",
                        "mode": "async",
                        "begin_line": 2,
                        "begin_snippet": "tma_load_tile(tid);",
                        "end_line": 3,
                        "completion_wait_snippet": "mbar.wait(0);",
                    },
                    {
                        "probe_id": "mislabeled_mma",
                        "semantic_id": "mma",
                        "mode": "sync",
                        "begin_line": 10,
                        "end_line": 10,
                    },
                ],
            },
            source_text=SAMPLE_CUDA_KERNEL,
            source_path=src_cu,
            route=placer.resolve_route("h20", "sm_90a", "cuda_cpp"),
        )
        self.assertEqual(bound["schema"], "weaveprofiler.probe-spec/v1")
        self.assertEqual(bound["probes"][0]["begin_line"], 5)
        self.assertEqual(bound["probes"][0]["end_line"], 6)
        self.assertEqual(bound["probes"][0]["completion_wait_line"], 6)
        # Pin BASE behavior: `store_global_tile` matches `_OFFLINE_PRIMITIVES["store"]` (`:1345`)
        # but NOT `_SEMANTIC_CUES["store"]` (`:658`), so `semantic_warnings` is empty here.
        self.assertEqual(bound["semantic_warnings"], [])

        bound_warn = placer.sanitize_and_bind_probe_spec(
            {
                "kernel_name": "gemm_pipe",
                "operator_family": "gemm",
                "probes": [
                    {
                        "probe_id": "mislabeled_mma",
                        "semantic_id": "mma",
                        "mode": "sync",
                        "begin_line": 10,
                        "end_line": 10,
                    },
                ],
            },
            source_text=SAMPLE_CUDA_KERNEL.replace("store_global_tile", "tma_store_tile"),
            source_path=src_cu,
            route=placer.resolve_route("h20", "sm_90a", "cuda_cpp"),
        )
        self.assertEqual(len(bound_warn["semantic_warnings"]), 1)
        self.assertEqual(bound_warn["semantic_warnings"][0]["code"], "SEMANTIC_MISMATCH")
        self.assertEqual(bound_warn["semantic_warnings"][0]["suggested_semantic"], "store")

        # 3) `detect_spin_loop_lines` on C++ single-line, C++ braced, Python indented, and non-spin loop
        spin_mix = (
            "__global__ void k(volatile int* flag, int n) {\n"
            "    while (*flag == 0);\n"
            "    while (atomicAdd((int*)flag, 0) == 0) {\n"
            "        int spin_v = 1;\n"
            "    }\n"
            "    while (n > 0) {\n"
            "        n -= 1;\n"
            "    }\n"
            "}\n"
        )
        self.assertEqual(
            placer.detect_spin_loop_lines(spin_mix),
            [
                (2, 2, "while (*flag == 0);"),
                (3, 5, "while (atomicAdd((int*)flag, 0) == 0) {"),
            ],
        )
        py_spin = "def f(ready):\n    while ready == 0:\n        x = 1\n    return 0\n"
        self.assertEqual(placer.detect_spin_loop_lines(py_spin), [(2, 3, "while ready == 0:")])

        # 4) `_build_readonly_probe_prompt` repair prompt structure
        kb_cu = placer.retrieve_route_kb("cuda_cpp", "h20", "sm_90a", SAMPLE_CUDA_KERNEL)
        repair_msgs = placer._build_readonly_probe_prompt(
            SAMPLE_CUDA_KERNEL,
            kb_cu,
            prev_spec=bound_warn,
            diagnostics={
                "status": "FAIL",
                "failed": ["overhead"],
                "timeline_summary": {
                    "kernel_baseline_us": 40.0,
                    "kernel_instrumented_us": 52.0,
                    "overhead_pct": 30.0,
                    "hotspots": ['worker "mma" (L7-L9): 16 fires'],
                },
            },
        )
        user_txt = repair_msgs[1]["content"]
        self.assertEqual([m["role"] for m in repair_msgs], ["system", "user"])
        self.assertTrue("Semantic disambiguation warnings from previous probe.json" in user_txt)
        self.assertTrue(
            "Failed-run timeline binary summary (from failed trace.perfetto): baseline=40.0us, instrumented=52.0us (+30.0%)."
            in user_txt
        )

        # 5) `place_probes` `bare_direct` vs `kb_direct` rewrite extraction & `reasoning_content` fallback
        seen_prompts: list[str] = []

        # Pin BASE bug at `llm_probe_placer.py:1152`: `\b(\d+)\b` rejects C++ `1u` unsigned literals (`None`)
        # and `val <= 9` at `:1154` drops `10` (`gemm`), while bare decimal `1, 0` resolves to `"load"`.
        self.assertIsNone(placer._infer_sem_from_probe_arg("1u, 0u"))
        self.assertIsNone(placer._infer_sem_from_probe_arg("10, 0"))
        self.assertEqual(placer._infer_sem_from_probe_arg("1, 0"), "load")

        def rewrite_llm(model: str, messages: list[dict], **kwargs: object) -> dict:
            seen_prompts.append(messages[1]["content"])
            rewritten = (
                "Short note:\n```cpp\n// short\n```\n"
                "Full rewrite:\n```cuda\n"
                '#include "wp_lite.cuh"\n'
                "#include <cuda_runtime.h>\n"
                "__global__ void gemm_pipe(float* out, int n) {\n"
                "    int tid = threadIdx.x;\n"
                "    if (tid >= n) return;\n"
                "    FI_OPEN(1);\n"
                "    FI_B(1, 0);\n"
                "    tma_load_tile(tid);\n"
                "    mbar.wait(0);\n"
                "    FI_E(1, 0);\n"
                "    FI_B(3, 0);\n"
                "    for (int k = 0; k < 16; ++k) {\n"
                "        wgmma_mma_tile(out, k);\n"
                "    }\n"
                "    FI_E(3, 0);\n"
                "    store_global_tile(out, tid);\n"
                "}\n"
                "```\n"
            )
            return {
                "ok": True,
                "model": "qwen4-max",
                "reasoning_effort": "max",
                "elapsed_s": 0.25,
                "usage": {"prompt_tokens": 120, "completion_tokens": 80, "reasoning_tokens": 10, "total_tokens": 200},
                "content": rewritten,
            }

        res_bare = placer.place_probes(src_cu, card="h20", ablation_mode="bare_direct", llm_caller=rewrite_llm)
        res_kb = placer.place_probes(src_cu, card="h20", ablation_mode="kb_direct", llm_caller=rewrite_llm)
        self.assertIs(res_bare["ok"], True)
        self.assertFalse("Operator & Language Profiling Knowledge Base:" in seen_prompts[0])
        self.assertTrue("Operator & Language Profiling Knowledge Base:" in seen_prompts[1])
        self.assertEqual(
            [(p["probe_id"], p["semantic_id"], p["role"], p["begin_line"], p["end_line"]) for p in res_kb["probe_spec"]["probes"]],
            [
                ("load_0", "load", "role_1", 5, 6),
                ("mma_1", "mma", "role_1", 7, 9),
            ],
        )

        # Fallback to `reasoning_content` when `content` has no JSON object (`llm_probe_placer.py:1316-1321`)
        def reasoning_only_llm(model: str, messages: list[dict], **kwargs: object) -> dict:
            return {
                "ok": True,
                "model": "deepseek-v4.1-flash-1017",
                "reasoning_effort": "max",
                "elapsed_s": 0.15,
                "usage": {"prompt_tokens": 40, "completion_tokens": 30, "reasoning_tokens": 25, "total_tokens": 70},
                "content": "Final answer omitted.",
                "reasoning_content": json.dumps(
                    {
                        "kernel_name": "gemm_pipe",
                        "probes": [{"probe_id": "st_0", "semantic_id": "store", "begin_line": 10, "end_line": 10}],
                    }
                ),
            }

        res_rc = placer.place_probes(src_cu, card="h20", ablation_mode="open_loop", llm_caller=reasoning_only_llm)
        self.assertEqual(res_rc["probe_spec"]["probes"][0]["probe_id"], "st_0")
        self.assertEqual(res_rc["probe_spec"]["probes"][0]["event_id"], 6)

        with self.assertRaises(ValueError) as bad_mode_ctx:
            placer.place_probes(src_cu, card="h20", ablation_mode="invalid_mode", llm_caller=reasoning_only_llm)
        self.assertTrue(str(bad_mode_ctx.exception).startswith("Unsupported ablation_mode 'invalid_mode'"))

    def test_evaluate_probe_predictions_scoring_spin_penalty_and_async_bounds(self) -> None:
        eval_src = (
            "__global__ void ws_pipe(volatile int* flag, float* out) {\n"
            "    int warp = threadIdx.x / 32;\n"
            "    if (warp == 4) {\n"
            "        tma_load_tile(0);\n"
            "    }\n"
            "    while (*flag == 0) {\n"
            "        int spin_iter = 0;\n"
            "    }\n"
            "    if (warp < 4) {\n"
            "        wgmma_issue();\n"
            "        wgmma_commit();\n"
            "        wgmma_wait();\n"
            "        epilogue_store(out);\n"
            "    }\n"
            "}\n"
        )
        gt_spec = {
            "warp_roles": [{"name": "dma_prod"}, {"name": "math_cons"}],
            "probes": [
                {"probe_id": "gt_load", "semantic_id": "load", "role": "dma_prod", "mode": "sync", "begin_line": 4, "end_line": 4},
                {"probe_id": "gt_wait", "semantic_id": "wait", "role": "math_cons", "mode": "sync", "begin_line": 6, "end_line": 8},
                {"probe_id": "gt_mma_async", "semantic_id": "mma", "role": "math_cons", "mode": "async", "begin_line": 10, "end_line": 12},
                {"probe_id": "gt_store", "semantic_id": "store", "role": "math_cons", "mode": "sync", "begin_line": 13, "end_line": 13},
            ],
        }
        pred_spec = {
            "probes": [
                # 1) Exact match on producer role -> hit=True, boundary_ok=True
                {"probe_id": "p_load", "semantic_id": "load", "role": "producer_dma", "begin_line": 4, "end_line": 4},
                # 2) Placed inside spin loop L6-L8 (line 7 is strictly inside L6-L8) -> hit=True, in_spin_loop=True, boundary_ok=False
                {"probe_id": "p_wait_spin", "semantic_id": "wait", "role": "consumer_math", "begin_line": 7, "end_line": 7},
                # 3) Async wider bracket [9, 13] around [10, 12] within tolerance=2 -> hit=True, boundary_ok=True
                {"probe_id": "p_mma", "semantic_id": "compute", "role": "mma_math", "begin_line": 9, "end_line": 13},
                # 4) Unmatched spurious barrier probe (`gt_store` missed)
                {"probe_id": "p_extra_bar", "semantic_id": "barrier", "role": "math_cons", "begin_line": 14, "end_line": 14},
            ],
        }
        metrics = placer.evaluate_probe_predictions(pred_spec, gt_spec, eval_src, line_tolerance=2)
        self.assertEqual(
            metrics,
            {
                "n_gt": 4,
                "n_pred": 4,
                "n_hit": 3,
                "n_boundary_ok": 2,
                "recall": 0.75,
                "precision": 0.75,
                "probe_boundary_accuracy": 0.5,
                "details": [
                    {
                        "gt_probe": "gt_load",
                        "hit": True,
                        "boundary_ok": True,
                        "in_spin_loop": False,
                        "pred_lines": [4, 4],
                        "gt_lines": [4, 4],
                    },
                    {
                        "gt_probe": "gt_wait",
                        "hit": True,
                        "boundary_ok": False,
                        "in_spin_loop": True,
                        "pred_lines": [7, 7],
                        "gt_lines": [6, 8],
                    },
                    {
                        "gt_probe": "gt_mma_async",
                        "hit": True,
                        "boundary_ok": True,
                        "in_spin_loop": False,
                        "pred_lines": [9, 13],
                        "gt_lines": [10, 12],
                    },
                    {
                        "gt_probe": "gt_store",
                        "hit": False,
                        "boundary_ok": False,
                    },
                ],
            },
        )

    def test_loopback_http_url_sse_reassembly_usage_retry_cli_and_missing_key(self) -> None:
        case_dir = self._new_case_dir("loopback_http")
        src_cu = case_dir / "gemm_pipe.cu"
        src_cu.write_text(SAMPLE_CUDA_KERNEL, encoding="utf-8")
        self._llm_state.requests.clear()
        self._llm_state.scripted_responses.clear()

        # 1) Endpoint URL normalization rules (`_normalize_chat_endpoint` & `~/.qwen/settings.json` discovery)
        self.assertEqual(
            placer.get_llm_endpoint_url(f"http://127.0.0.1:{self._port}"),
            f"http://127.0.0.1:{self._port}/v1/chat/completions",
        )
        self.assertEqual(
            placer.get_llm_endpoint_url(f"http://127.0.0.1:{self._port}/compatible-mode/v1/"),
            f"http://127.0.0.1:{self._port}/compatible-mode/v1/chat/completions",
        )
        self.assertEqual(
            placer.get_llm_endpoint_url(f"http://127.0.0.1:{self._port}/custom/chat/completions"),
            f"http://127.0.0.1:{self._port}/custom/chat/completions",
        )

        qwen_dir = self._iso_root / "home" / ".qwen"
        qwen_dir.mkdir(parents=True, exist_ok=True)
        (qwen_dir / "settings.json").write_text(
            json.dumps(
                {"modelProviders": {"openai": [{"baseUrl": f"http://127.0.0.1:{self._port}/qwen-local/v1"}]}}
            ),
            encoding="utf-8",
        )
        saved_wb = os.environ.pop("WEAVEPROF_BASE_URL", None)
        saved_ob = os.environ.pop("OPENAI_BASE_URL", None)
        try:
            qcfg = placer.resolve_llm_config()
            self.assertEqual(qcfg["endpoint_mode"], "agent_settings")
            self.assertEqual(qcfg["endpoint_url"], f"http://127.0.0.1:{self._port}/qwen-local/v1/chat/completions")
        finally:
            if saved_wb is not None:
                os.environ["WEAVEPROF_BASE_URL"] = saved_wb
            if saved_ob is not None:
                os.environ["OPENAI_BASE_URL"] = saved_ob
            shutil.rmtree(qwen_dir, ignore_errors=True)

        # 2) SSE `data:` multi-chunk reassembly + `[DONE]` cutoff + `reasoning_content` + `usage` + snippet binding
        sse_lines = [
            b": keep-alive comment\n\n",
            b"data: {broken-json-ignored}\n\n",
            b'data: {"choices":[{"delta":{"reasoning_content":"analyzing "}}]}\n\n',
            b'data: {"choices":[{"delta":{"reasoning_content":"lines 5..10"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"{\\"kernel_name\\": \\"gemm_pipe\\", \\"probes\\": ["}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"{\\"probe_id\\": \\"ld_sse\\", \\"semantic_id\\": \\"load\\", "}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"\\"begin_line\\": 1, \\"end_line\\": 1, \\"begin_snippet\\": \\"tma_load_tile(tid);\\", \\"end_snippet\\": \\"tma_load_tile(tid);\\"}]}"}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":310,"completion_tokens":95,"total_tokens":405,"completion_tokens_details":{"reasoning_tokens":42}}}\n\n',
            b"data: [DONE]\n\n",
            b'data: {"choices":[{"delta":{"content":"MUST_NOT_BE_APPENDED_AFTER_DONE"}}]}\n\n',
        ]
        self._llm_state.scripted_responses.append(
            {"status": 200, "ctype": "text/event-stream; charset=utf-8", "body": b"".join(sse_lines)}
        )
        placed = placer.place_probes(
            src_cu,
            card="h20",
            model="auto",
            base_url=f"http://127.0.0.1:{self._port}/compatible-mode/v1",
        )
        self.assertEqual(len(self._llm_state.requests), 1)
        req0 = self._llm_state.requests[0]
        self.assertEqual(req0["path"], "/compatible-mode/v1/chat/completions")
        self.assertEqual(req0["auth"], "Bearer test-dummy-key")
        self.assertIs(req0["payload"]["stream"], True)
        self.assertEqual(req0["payload"]["reasoning_effort"], "max")
        # Downstream processing assertions: snippet bound line 1 -> line 5, event_id=1, usage & telemetry extracted
        self.assertEqual(placed["probe_spec"]["probes"][0]["probe_id"], "ld_sse")
        self.assertEqual(placed["probe_spec"]["probes"][0]["begin_line"], 5)
        self.assertEqual(placed["probe_spec"]["probes"][0]["end_line"], 5)
        self.assertEqual(placed["probe_spec"]["probes"][0]["event_id"], 1)
        self.assertEqual(
            {k: placed["telemetry"][k] for k in ("model", "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")},
            {
                "model": "qwen4-max",
                "prompt_tokens": 310,
                "completion_tokens": 95,
                "reasoning_tokens": 42,
                "total_tokens": 405,
            },
        )
        self.assertNotIn("MUST_NOT_BE_APPENDED_AFTER_DONE", placed["raw_response"])

        # 3) HTTP 400 `reasoning_effort` rejection: `retries=1` exhausts `range(1)` after `continue`,
        # whereas `retries=2` immediately re-sends on attempt 1 (`continue` bypasses backoff) without `reasoning_effort`.
        self._llm_state.scripted_responses.append(
            {
                "status": 400,
                "ctype": "application/json",
                "body": b'{"error": "Unsupported field reasoning_effort"}',
            }
        )
        with self.assertRaises(RuntimeError) as r1_400_ctx:
            placer.call_llm(
                "kimi-k3.1-pro",
                [{"role": "user", "content": "ping"}],
                retries=1,
                stream=False,
            )
        self.assertTrue("HTTP 400: " in str(r1_400_ctx.exception))

        self._llm_state.scripted_responses.extend(
            [
                {
                    "status": 400,
                    "ctype": "application/json",
                    "body": b'{"error": "Unsupported field reasoning_effort"}',
                },
                {
                    "status": 200,
                    "ctype": "application/json",
                    "body": json.dumps(
                        {
                            "choices": [
                                {
                                    "message": {"content": '{"ok_json": 1}', "reasoning_content": "r_part"},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "reasoning_tokens": 3, "total_tokens": 20},
                        }
                    ).encode("utf-8"),
                },
            ]
        )
        downgraded = placer.call_llm(
            "kimi-k3.1-pro",
            [{"role": "user", "content": "ping"}],
            retries=2,
            stream=False,
        )
        self.assertEqual(len(self._llm_state.requests), 4)
        self.assertIn("reasoning_effort", self._llm_state.requests[2]["payload"])
        self.assertNotIn("reasoning_effort", self._llm_state.requests[3]["payload"])
        self.assertIsNone(downgraded["reasoning_effort"])
        self.assertEqual(downgraded["finish_reason"], "stop")
        self.assertEqual(
            downgraded["usage"],
            {"prompt_tokens": 12, "completion_tokens": 8, "reasoning_tokens": 3, "total_tokens": 20},
        )

        # 4) HTTP 500 error path with `retries=1` (zero backoff delay)
        self._llm_state.scripted_responses.append(
            {"status": 500, "ctype": "text/plain", "body": b"local_gateway_500"}
        )
        with self.assertRaises(RuntimeError) as http500_ctx:
            placer.call_llm("qwen4-max", [{"role": "user", "content": "x"}], retries=1)
        self.assertEqual(
            str(http500_ctx.exception),
            f"LLM call failed for qwen4-max (http://127.0.0.1:{self._port}/v1/chat/completions): HTTP 500: local_gateway_500",
        )

        # 5) Missing API key error path (both env keys cleared)
        saved_key = os.environ.pop("WEAVEPROF_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)
        try:
            req_count_before = len(self._llm_state.requests)
            with self.assertRaises(RuntimeError) as nokey_ctx:
                placer.call_llm("qwen4-max", [{"role": "user", "content": "x"}], retries=1)
            self.assertEqual(
                str(nokey_ctx.exception),
                "No API key found (pass api_key=... / --api-key, or set WEAVEPROF_API_KEY / OPENAI_API_KEY)",
            )
            self.assertEqual(len(self._llm_state.requests), req_count_before)
        finally:
            if saved_key is not None:
                os.environ["WEAVEPROF_API_KEY"] = saved_key

        # 6) `llm_probe_placer.main` in-process and `run.py probe-place` subprocess via loopback server
        cli_body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "```cuda\n"
                                '#include "wp_lite.cuh"\n'
                                "#include <cuda_runtime.h>\n"
                                "__global__ void gemm_pipe(float* out, int n) {\n"
                                "    int tid = threadIdx.x;\n"
                                "    if (tid >= n) return;\n"
                                "    FI_OPEN(0);\n"
                                "    FI_B(3, 0);\n"
                                "    for (int k = 0; k < 16; ++k) {\n"
                                "        wgmma_mma_tile(out, k);\n"
                                "    }\n"
                                "    FI_E(3, 0);\n"
                                "    store_global_tile(out, tid);\n"
                                "}\n"
                                "```\n"
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120},
            }
        ).encode("utf-8")
        self._llm_state.scripted_responses.append(
            {"status": 200, "ctype": "application/json", "body": cli_body}
        )
        out_spec_p = case_dir / "cli_out_spec.json"
        out_rw_p = case_dir / "cli_rewritten.cu"
        err_log_p = case_dir / "verify_err.log"
        err_log_p.write_text("Overhead 18.0% > 10.0%", encoding="utf-8")

        buf_main = io.StringIO()
        with contextlib.redirect_stdout(buf_main):
            rc_main = placer.main(
                [
                    "--source",
                    str(src_cu),
                    "--card",
                    "h20",
                    "--mode",
                    "kb_direct",
                    "--error-log",
                    str(err_log_p),
                    "--out",
                    str(out_spec_p),
                    "--out-rewritten",
                    str(out_rw_p),
                ]
            )
        self.assertEqual(rc_main, 0)
        main_summary = json.loads(buf_main.getvalue())
        self.assertEqual(
            {k: main_summary[k] for k in ("ok", "probes", "warp_roles")},
            {"ok": True, "probes": 1, "warp_roles": 1},
        )
        self.assertTrue(out_spec_p.is_file())
        self.assertTrue(out_rw_p.is_file())

        # `run.py probe-place` subprocess (guarded by `sitecustomize.py` in `PYTHONPATH`)
        self._llm_state.scripted_responses.append(
            {
                "status": 200,
                "ctype": "application/json",
                "body": json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "kernel_name": "gemm_pipe",
                                            "probes": [
                                                {"probe_id": "mma_sub", "semantic_id": "mma", "begin_line": 7, "end_line": 9}
                                            ],
                                        }
                                    )
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 60, "completion_tokens": 25, "total_tokens": 85},
                    }
                ).encode("utf-8"),
            }
        )
        sub_out_p = case_dir / "sub_probe_spec.json"
        proc_run = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "run.py"),
                "probe-place",
                "--source",
                str(src_cu),
                "--card",
                "h20",
                "--mode",
                "open_loop",
                "--out",
                str(sub_out_p),
            ],
            cwd=str(case_dir),
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc_run.returncode, 0, proc_run.stderr)
        sub_summary = json.loads(proc_run.stdout)
        self.assertIs(sub_summary["ok"], True)
        self.assertEqual(sub_summary["probes"], 1)
        sub_spec = json.loads(sub_out_p.read_text(encoding="utf-8"))
        self.assertEqual(sub_spec["probes"][0]["probe_id"], "mma_sub")
        self.assertEqual(sub_spec["probes"][0]["event_id"], 3)


if __name__ == "__main__":
    unittest.main()
