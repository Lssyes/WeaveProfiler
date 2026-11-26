#!/usr/bin/env python3
"""Characterization tests pinning the MCP server entrypoint end to end (P1c R22).

Covers:
  (a) Provisioning an isolated session with ``agent_installer.main(["install", ...])`` and
      launching the MCP server subprocess from the exact ``.mcp.json`` command written by
      the installer.
  (b) JSON-RPC ``initialize``, ``tools/list``, and ``weaveprof_megakernel_audit`` rejection
      paths (5 forbidden keys, ``write_perfetto_gz=True`` under PNG-only policy, missing
      locked ``bench_cmd``, and out-of-bounds ``out_dir``).
  (c) Argument forwarding from ``_tool_megakernel`` into ``run_megakernel_audit`` via a
      registered subprocess launcher stub (asserting only normalized received arguments,
      never the stub's return payload).

Run:  python -B -m unittest tests/characterization/test_p1c_entry_mcp.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import contextlib
import io
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import tempfile
import unittest
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if sys.path[:1] != [str(REPO_ROOT)]:
    sys.path.insert(0, str(REPO_ROOT))

from weaveprof import agent_installer  # noqa: E402

_ENV_KEYS = (
    "HOME",
    "USERPROFILE",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "WEAVEPROF_SESSIONS_DIR",
    "WEAVEPROF_CACHE",
    "WEAVEPROF_LOCK_DIR",
    "TMPDIR",
    "TEMP",
    "TMP",
    "MPLCONFIGDIR",
    "WEAVEPROF_SUPERVISOR_HMAC_KEY",
    "WEAVEPROF_SUPERVISOR_KEY_FILE",
    "QWEN_PROJECT_DIR",
    "QWEN_SESSION_ID",
    "WEAVEPROF_LOCK_PROVIDER",
    "WEAVEPROF_KEEP_SHADOW",
    "WEAVEPROF_GOLDEN_UPDATE",
    "WEAVEPROF_METRICS_UPDATE",
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
)

_STUB_LAUNCHER_SOURCE = """#!/usr/bin/env python3
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import inspect
import json
import os
from pathlib import Path
from weaveprof.harness import HarnessPolicy
import weaveprof.megakernel as mk
import weaveprof.mcp_server as mcp_server

_orig = mk.run_megakernel_audit
_sig = inspect.signature(_orig)

def _serialize(val):
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, HarnessPolicy):
        return {"__type__": "HarnessPolicy", **val.to_dict()}
    if isinstance(val, dict):
        return {str(k): _serialize(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize(x) for x in val]
    return val

def _stub_audit(*args, **kwargs):
    bound = _sig.bind_partial(*args, **kwargs)
    payload = {k: _serialize(v) for k, v in bound.arguments.items()}
    Path(os.environ["P1C_MCP_CAPTURED_ARGS"]).write_text(
        json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8"
    )
    return {"schema": "weaveprof.megakernel-audit/v1", "status": "STUB_FORWARDED"}

_stub_audit.__signature__ = _sig
mk.run_megakernel_audit = _stub_audit
raise SystemExit(mcp_server.main(sys.argv[1:]))
"""


def _norm_tmp(obj: Any, tmp_root: Path) -> Any:
    tokens = sorted({str(tmp_root), os.path.realpath(str(tmp_root))}, key=lambda s: -len(s))
    if isinstance(obj, str):
        for real in tokens:
            if real in obj:
                obj = obj.replace(real, "<TMP>")
        return obj
    if isinstance(obj, dict):
        return {k: _norm_tmp(v, tmp_root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_norm_tmp(v, tmp_root) for v in obj]
    return obj


class _McpSessionHandle:
    """Manages one stdio MCP server child process with file-backed stderr and bounded reads."""

    def __init__(self, argv: List[str], *, cwd: Path, env: Dict[str, str], stderr_path: Path, label: str) -> None:
        self.label = label
        self.stderr_path = stderr_path
        self.stderr_fh = open(stderr_path, "wb")
        self.proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_fh,
        )
        self.pid = self.proc.pid
        self._buf = b""

    def request(self, payload: Dict[str, Any], *, timeout_s: float = 30.0) -> Dict[str, Any]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        raw_out = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        self.proc.stdin.write(raw_out)
        self.proc.stdin.flush()

        fd = self.proc.stdout.fileno()
        sel = selectors.DefaultSelector()
        sel.register(fd, selectors.EVENT_READ)
        try:
            while b"\n" not in self._buf:
                ready = sel.select(timeout=timeout_s)
                if not ready:
                    raise TimeoutError(f"Timed out waiting for MCP reply on {self.label}")
                chunk = os.read(fd, 65536)
                if not chunk:
                    raise EOFError(f"MCP server {self.label} closed stdout before replying")
                self._buf += chunk
        finally:
            sel.close()
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))

    def shutdown(self) -> int:
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2.0)
        if self.proc.stdout is not None and not self.proc.stdout.closed:
            self.proc.stdout.close()
        if not self.stderr_fh.closed:
            self.stderr_fh.close()
        rc = self.proc.poll()
        assert rc is not None
        return rc


class McpEntrypointTest(unittest.TestCase):
    PID_RECORDS: List[Tuple[int, str, int | None]] = []

    @classmethod
    def setUpClass(cls) -> None:
        cls._saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}
        cls._saved_tempdir = tempfile.tempdir
        cls._root = Path(tempfile.mkdtemp(prefix="wp-p1c-r22-mcp-")).resolve()
        for env_k, sub in (
            ("HOME", "home"),
            ("USERPROFILE", "home"),
            ("XDG_CACHE_HOME", "xdg_cache"),
            ("XDG_CONFIG_HOME", "xdg_config"),
            ("WEAVEPROF_SESSIONS_DIR", "sessions"),
            ("WEAVEPROF_CACHE", "wp_cache"),
            ("WEAVEPROF_LOCK_DIR", "locks"),
            ("TMPDIR", "tmp"),
            ("TEMP", "tmp"),
            ("TMP", "tmp"),
            ("MPLCONFIGDIR", "mpl"),
        ):
            d = cls._root / sub
            d.mkdir(parents=True, exist_ok=True)
            os.environ[env_k] = str(d)
        tempfile.tempdir = None
        for clear_k in (
            "WEAVEPROF_SUPERVISOR_KEY_FILE",
            "QWEN_PROJECT_DIR",
            "QWEN_SESSION_ID",
            "WEAVEPROF_LOCK_PROVIDER",
            "WEAVEPROF_KEEP_SHADOW",
            "WEAVEPROF_GOLDEN_UPDATE",
            "WEAVEPROF_METRICS_UPDATE",
        ):
            os.environ.pop(clear_k, None)
        os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = "weaveprof-golden-test-key"
        os.environ["PYTHONPATH"] = str(REPO_ROOT)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        cls._env = dict(os.environ)
        cls._cwd = cls._root / "cwd"
        cls._cwd.mkdir(parents=True, exist_ok=True)

        probe = subprocess.run(
            [sys.executable, "-B", "-c", "import weaveprof; print(weaveprof.__file__)"],
            cwd=str(cls._cwd),
            env=cls._env,
            capture_output=True,
            text=True,
            check=True,
        )
        rel_init = Path(probe.stdout.strip()).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        if rel_init != "weaveprof/__init__.py":
            raise AssertionError(f"weaveprof imported from outside repo under test: {probe.stdout!r}")

        # Provision workspace 1 (with locked placeholder bench_cmd) via agent_installer.main(["install", ...])
        case_src = REPO_ROOT / "tests" / "golden" / "cases" / "h20_pingpong_v1" / "workspace"
        cls._ws_locked = cls._root / "ws_locked"
        shutil.copytree(case_src, cls._ws_locked)
        cls._locked_bench_str = "python3 placeholder_bench.py --check"
        cls._locked_bench_list = ["python3", "placeholder_bench.py", "--check"]
        buf_install = io.StringIO()
        with contextlib.redirect_stdout(buf_install):
            rc_inst = agent_installer.main(
                [
                    "install",
                    "--workspace",
                    str(cls._ws_locked),
                    "--protect",
                    "matmul-pingpong-v1.cu",
                    "--agent",
                    "claude",
                    "--bench",
                    cls._locked_bench_str,
                    "--card",
                    "h20",
                    "--arch",
                    "sm_90a",
                    "--python",
                    sys.executable,
                ]
            )
        if rc_inst != 0:
            raise AssertionError(f"agent_installer install failed: rc={rc_inst}")
        cls._install_doc = json.loads(buf_install.getvalue())

        # Provision workspace 2 (without locked bench_cmd) for the missing-bench rejection case
        cls._ws_nobench = cls._root / "ws_nobench"
        shutil.copytree(case_src, cls._ws_nobench)
        buf_nb = io.StringIO()
        with contextlib.redirect_stdout(buf_nb):
            rc_nb = agent_installer.main(
                [
                    "install",
                    "--workspace",
                    str(cls._ws_nobench),
                    "--protect",
                    "matmul-pingpong-v1.cu",
                    "--agent",
                    "claude",
                    "--card",
                    "h20",
                    "--arch",
                    "sm_90a",
                    "--python",
                    sys.executable,
                ]
            )
        if rc_nb != 0:
            raise AssertionError(f"agent_installer nobench install failed: rc={rc_nb}")

    @classmethod
    def tearDownClass(cls) -> None:
        tempfile.tempdir = cls._saved_tempdir
        for k, v in cls._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(cls._root, ignore_errors=True)

    def setUp(self) -> None:
        self._handles: List[_McpSessionHandle] = []

    def tearDown(self) -> None:
        for h in self._handles:
            rc = h.shutdown()
            poll_after = h.proc.poll()
            self.assertIsNotNone(poll_after)
            self.assertEqual(rc, 0)
            self.__class__.PID_RECORDS.append((h.pid, h.label, poll_after))

    def _spawn(self, argv: List[str], *, env: Dict[str, str] | None = None, label: str) -> _McpSessionHandle:
        stderr_p = self._root / f"mcp_stderr_{label}_{len(self._handles)}.log"
        h = _McpSessionHandle(
            argv,
            cwd=self._cwd,
            env=env if env is not None else self._env,
            stderr_path=stderr_p,
            label=label,
        )
        self._handles.append(h)
        return h

    def test_a_and_b_installer_launch_protocol_and_megakernel_rejections(self) -> None:
        mcp_json_path = self._ws_locked / ".mcp.json"
        self.assertTrue(mcp_json_path.is_file())
        mcp_cfg = json.loads(mcp_json_path.read_text(encoding="utf-8"))["mcpServers"]["weaveprofiler"]
        self.assertEqual(mcp_cfg["type"], "stdio")
        self.assertEqual(mcp_cfg["command"], sys.executable)
        self.assertEqual(
            mcp_cfg["args"],
            [
                "-B",
                str(REPO_ROOT / "weaveprof" / "mcp_server.py"),
                "--workspace",
                str(self._ws_locked),
                "--session-dir",
                self._install_doc["session_dir"],
            ],
        )

        srv = self._spawn([mcp_cfg["command"], *mcp_cfg["args"]], label="locked_protocol_and_rejections")

        # (b.1) initialize
        r_init = srv.request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "p1c-r22", "version": "1.0"},
                },
            }
        )
        self.assertEqual(r_init["id"], 1)
        self.assertEqual(r_init["result"]["protocolVersion"], "2025-06-18")
        self.assertEqual(
            r_init["result"]["serverInfo"],
            {"name": "weaveprofiler", "title": "WeaveProfiler (wp)", "version": "1.0.0"},
        )
        self.assertEqual(
            r_init["result"]["capabilities"],
            {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
        )

        # (b.2) tools/list
        r_list = srv.request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tool_names = [t["name"] for t in r_list["result"]["tools"]]
        self.assertEqual(
            tool_names,
            [
                "weaveprof_status",
                "weaveprof_megakernel_audit",
                "weaveprof_profile",
                "weaveprof_plan_context",
                "weaveprof_analyze",
                "weaveprof_compare_rounds",
                "weaveprof_verify_seal",
            ],
        )

        # Unknown JSON-RPC method & unknown tool name error codes
        r_bad_method = srv.request({"jsonrpc": "2.0", "id": 3, "method": "bogus/method", "params": {}})
        self.assertEqual(
            r_bad_method["error"],
            {"code": -32601, "message": "Method not found: bogus/method"},
        )
        r_bad_tool = srv.request(
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "no_such_tool", "arguments": {}}}
        )
        self.assertEqual(
            r_bad_tool["error"],
            {"code": -32602, "message": "Unknown tool: 'no_such_tool'"},
        )

        # (b.3) 5 forbidden keys on weaveprof_megakernel_audit
        forbidden_cases = [
            (
                {"dump_path": "cap.wp02.bin"},
                "ValueError: `dump_path` is not allowed on MCP weaveprof_megakernel_audit (MCP audits always execute the isolated benchmark workflow).",
            ),
            (
                {"runtime_metrics": {}},
                "ValueError: `runtime_metrics` is not allowed on MCP weaveprof_megakernel_audit (MCP audits always execute the isolated benchmark workflow).",
            ),
            (
                {"names": {}},
                "ValueError: `names` is not allowed on MCP weaveprof_megakernel_audit (MCP audits always execute the isolated benchmark workflow).",
            ),
            (
                {"nsys_cmd": "echo nsys"},
                "ValueError: `nsys_cmd` is not allowed on MCP weaveprof_megakernel_audit (MCP audits always execute the isolated benchmark workflow).",
            ),
            (
                {"policy": {}},
                "ValueError: `policy` is not allowed on MCP weaveprof_megakernel_audit (MCP audits always execute the isolated benchmark workflow).",
            ),
        ]
        req_id = 10
        for bad_args, expected_err in forbidden_cases:
            req_id += 1
            key_name = next(iter(bad_args))
            with self.subTest(forbidden_key=key_name):
                resp = srv.request(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "method": "tools/call",
                        "params": {"name": "weaveprof_megakernel_audit", "arguments": bad_args},
                    }
                )
                self.assertIs(resp["result"]["isError"], True)
                sc = resp["result"]["structuredContent"]
                self.assertIs(sc["ok"], False)
                self.assertEqual(sc["tool"], "weaveprof_megakernel_audit")
                self.assertEqual(sc["error"], expected_err)

        # (b.4) write_perfetto_gz=True blocked by PNG-only session policy
        r_gz = srv.request(
            {
                "jsonrpc": "2.0",
                "id": 20,
                "method": "tools/call",
                "params": {"name": "weaveprof_megakernel_audit", "arguments": {"write_perfetto_gz": True}},
            }
        )
        self.assertIs(r_gz["result"]["isError"], True)
        self.assertIs(r_gz["result"]["structuredContent"]["ok"], False)
        self.assertEqual(
            r_gz["result"]["structuredContent"]["error"],
            "ValueError: `write_perfetto_gz=True` is blocked by the session PNG-only policy; deliver `overview.png` instead (or re-install with `--allow-gz`).",
        )

        # (b.5) out_dir out-of-bounds rejections (inside workspace, protected agent config, protected seal filename)
        r_out_ws = srv.request(
            {
                "jsonrpc": "2.0",
                "id": 21,
                "method": "tools/call",
                "params": {"name": "weaveprof_megakernel_audit", "arguments": {"out_dir": "inside_ws_out"}},
            }
        )
        self.assertIs(r_out_ws["result"]["isError"], True)
        self.assertEqual(
            _norm_tmp(r_out_ws["result"]["structuredContent"]["error"], self._root),
            "ValueError: out_dir='<TMP>/ws_locked/inside_ws_out' must be outside the workspace to avoid mutating real project files",
        )

        r_out_cfg = srv.request(
            {
                "jsonrpc": "2.0",
                "id": 22,
                "method": "tools/call",
                "params": {
                    "name": "weaveprof_megakernel_audit",
                    "arguments": {"out_dir": str(self._ws_locked / ".claude" / "sub")},
                },
            }
        )
        self.assertIs(r_out_cfg["result"]["isError"], True)
        self.assertEqual(
            _norm_tmp(r_out_cfg["result"]["structuredContent"]["error"], self._root),
            "ValueError: out_dir='<TMP>/ws_locked/.claude/sub' targets protected supervisor/agent configuration",
        )

        r_out_seal = srv.request(
            {
                "jsonrpc": "2.0",
                "id": 23,
                "method": "tools/call",
                "params": {
                    "name": "weaveprof_megakernel_audit",
                    "arguments": {"out_dir": str(self._root / "ext_out" / "seal.json")},
                },
            }
        )
        self.assertIs(r_out_seal["result"]["isError"], True)
        self.assertEqual(
            _norm_tmp(r_out_seal["result"]["structuredContent"]["error"], self._root),
            "ValueError: out_dir='<TMP>/ext_out/seal.json' targets protected evidence/session file",
        )

        # (b.6) Overriding locked session bench_cmd (via bench_cmd or bench alias) is rejected
        r_bench_override = srv.request(
            {
                "jsonrpc": "2.0",
                "id": 24,
                "method": "tools/call",
                "params": {
                    "name": "weaveprof_megakernel_audit",
                    "arguments": {
                        "out_dir": str(self._root / "ext_out_ok"),
                        "bench_cmd": ["python3", "other_bench.py"],
                    },
                },
            }
        )
        self.assertIs(r_bench_override["result"]["isError"], True)
        self.assertEqual(
            r_bench_override["result"]["structuredContent"]["error"],
            "ValueError: bench_cmd is locked by the session installed at provisioning time (['python3', 'placeholder_bench.py', '--check'])",
        )

        r_bench_alias_override = srv.request(
            {
                "jsonrpc": "2.0",
                "id": 25,
                "method": "tools/call",
                "params": {
                    "name": "weaveprof_megakernel_audit",
                    "arguments": {
                        "out_dir": str(self._root / "ext_out_ok"),
                        "bench": "python3 other_bench.py",
                    },
                },
            }
        )
        self.assertIs(r_bench_alias_override["result"]["isError"], True)
        self.assertEqual(
            r_bench_alias_override["result"]["structuredContent"]["error"],
            "ValueError: bench_cmd is locked by the session installed at provisioning time (['python3', 'placeholder_bench.py', '--check'])",
        )

    def test_b_missing_locked_bench_cmd_rejected(self) -> None:
        nb_cfg = json.loads((self._ws_nobench / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["weaveprofiler"]
        srv = self._spawn([nb_cfg["command"], *nb_cfg["args"]], label="nobench_rejection")
        srv.request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
        resp = srv.request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "weaveprof_megakernel_audit", "arguments": {"out_dir": str(self._root / "nb_out")}},
            }
        )
        self.assertIs(resp["result"]["isError"], True)
        sc = resp["result"]["structuredContent"]
        self.assertIs(sc["ok"], False)
        self.assertEqual(
            sc["error"],
            "ValueError: `bench_cmd` is required (no default bench command recorded at install time)",
        )

    def test_c_megakernel_argument_forwarding_to_audit_engine(self) -> None:
        mcp_cfg = json.loads((self._ws_locked / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["weaveprofiler"]
        launcher = self._root / "mcp_stub_launcher.py"
        launcher.write_text(_STUB_LAUNCHER_SOURCE, encoding="utf-8")
        captured_file = self._root / "captured_mcp_args.json"

        env = dict(self._env)
        env["P1C_MCP_CAPTURED_ARGS"] = str(captured_file)
        # Replace mcp_server.py script path with our stub launcher while preserving the exact CLI flags from .mcp.json
        server_flags = mcp_cfg["args"][2:]
        srv = self._spawn([mcp_cfg["command"], "-B", str(launcher), *server_flags], env=env, label="stub_forwarding")
        srv.request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})

        # Call 1: omit workspace/entry_source/bench_cmd/card/arch/limit so session defaults & path resolution apply
        captured_file.unlink(missing_ok=True)
        resp1 = srv.request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "weaveprof_megakernel_audit",
                    "arguments": {"out_dir": str(self._root / "forwarded_out_1"), "overview_sm": 5},
                },
            }
        )
        self.assertTrue(captured_file.is_file(), f"Audit stub was not reached on Call 1: {resp1}")
        cap1 = _norm_tmp(json.loads(captured_file.read_text(encoding="utf-8")), self._root)
        self.assertEqual(
            sorted(cap1.keys()),
            ["arch", "bench_cmd", "card", "entry_source", "limit", "out_dir", "overview_sm", "policy", "workspace"],
        )
        self.assertEqual(cap1["workspace"], "<TMP>/ws_locked")
        self.assertEqual(cap1["entry_source"], "<TMP>/ws_locked/matmul-pingpong-v1.cu")
        self.assertEqual(cap1["out_dir"], "<TMP>/forwarded_out_1")
        self.assertEqual(cap1["bench_cmd"], ["python3", "placeholder_bench.py", "--check"])
        self.assertEqual(cap1["card"], "h20")
        self.assertEqual(cap1["arch"], "sm_90a")
        self.assertEqual(cap1["limit"], 0.10)
        self.assertEqual(cap1["overview_sm"], 5)
        self.assertEqual(cap1["policy"]["__type__"], "HarnessPolicy")
        self.assertEqual(cap1["policy"]["max_overhead_limit"], 0.10)

        # Call 2: relative sources alias (distinct from session default) + explicit tighter limit=0.075
        captured_file.unlink(missing_ok=True)
        resp2 = srv.request(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "weaveprof_megakernel_audit",
                    "arguments": {
                        "sources": ["alt_entry.cu"],
                        "out_dir": str(self._root / "forwarded_out_2"),
                        "limit": 0.075,
                    },
                },
            }
        )
        self.assertTrue(captured_file.is_file(), f"Audit stub was not reached on Call 2: {resp2}")
        cap2 = _norm_tmp(json.loads(captured_file.read_text(encoding="utf-8")), self._root)
        self.assertEqual(cap2["workspace"], "<TMP>/ws_locked")
        self.assertEqual(cap2["entry_source"], "<TMP>/ws_locked/alt_entry.cu")
        self.assertEqual(cap2["out_dir"], "<TMP>/forwarded_out_2")
        self.assertEqual(cap2["bench_cmd"], ["python3", "placeholder_bench.py", "--check"])
        self.assertEqual(cap2["limit"], 0.075)
        self.assertEqual(cap2["policy"]["max_overhead_limit"], 0.10)


if __name__ == "__main__":
    unittest.main()
