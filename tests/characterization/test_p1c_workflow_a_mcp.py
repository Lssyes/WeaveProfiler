#!/usr/bin/env python3
"""Characterization tests pinning full Workflow A execution through the MCP server entrypoint (P1c R23).

Provisions an isolated supervised session with ``agent_installer.main(["install", ..., "--bench", ..., "--allow-gz"])``
locking a runtime-generated fake ``bench_cmd``, launches the real MCP stdio server from ``.mcp.json`` with a fake
``nsys`` on ``PATH``, invokes ``weaveprof_megakernel_audit`` end to end through Workflow A, and verifies:
  1. Concrete literal report values, ``weaveprof_session`` seal registration (``registered=True``, ``status='PASS'``,
     ``covers_protected=True``, ``edit_unlocked=True``, ``stop_unlocked=True``), and ``weaveprof_status`` /
     ``weaveprof_verify_seal`` round-trips.
  2. Full ``h20_pingpong_v1`` golden artifact equivalence modulo the 3 pre-registered Workflow-A diff paths
     (``$.report.mode``, ``$.feedback.mode``, ``$.feedback.l0_cache``).
  3. Exact fake build/bench/nsys call sequence and session-locked ``bench_cmd`` execution, plus verification that
     all installer-written paths reside strictly inside the temporary root.

Run:  python -B -m unittest tests/characterization/test_p1c_workflow_a_mcp.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from collections import Counter
import contextlib
import gzip
import importlib
import io
import json
import lzma
import os
from pathlib import Path
import re
import selectors
import shlex
import shutil
import subprocess
import tempfile
import unittest
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if sys.path[:1] != [str(REPO_ROOT)]:
    sys.path.insert(0, str(REPO_ROOT))

# Clear golden update flags before importing the golden module (N4), and import only the module
# object so unittest does not re-collect GoldenAuditTest (N1).
os.environ.pop("WEAVEPROF_GOLDEN_UPDATE", None)
os.environ.pop("WEAVEPROF_METRICS_UPDATE", None)
golden = importlib.import_module("tests.golden.test_golden_audit")

from weaveprof import agent_installer  # noqa: E402
from weaveprof import megakernel as mk  # noqa: E402

WORKFLOW_A_EXPECTED_DIFF_PATHS: Tuple[Tuple[Tuple[str, ...], str, str], ...] = (
    (
        ("report", "mode"),
        "$.report.mode",
        "Workflow A sets report['mode'] = 'shadow_build_run' vs 'precaptured' (weaveprof/megakernel.py:1673,2049).",
    ),
    (
        ("feedback", "mode"),
        "$.feedback.mode",
        "Workflow A writes feedback['mode'] = 'shadow_build_run' vs 'precaptured' (weaveprof/megakernel.py:1673,1943).",
    ),
    (
        ("feedback", "l0_cache"),
        "$.feedback.l0_cache",
        "Workflow A populates shadow_info['l0_cache'] instead of None (weaveprof/megakernel.py:1129-1134,1194,1230,1948).",
    ),
)

_ENV_KEYS = (
    "PATH",
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
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "P1C_INVOCATION_LOG",
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
)

_FAKE_BUILD_BODY = """from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import fcntl
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--l0", required=True)
ap.add_argument("--inst", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tree", required=True)
args = ap.parse_args()

l0_p = Path(args.l0).resolve()
inst_p = Path(args.inst).resolve()
tree_p = Path(args.tree).resolve()
role = "L0" if tree_p == l0_p else ("L_inst" if tree_p == inst_p else "other")

log_file = os.environ.get("P1C_INVOCATION_LOG")
if log_file:
    rec = {
        "kind": "build",
        "role": role,
        "pid": os.getpid(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "env": {
            "WEAVEPROF_L0_DIR": os.environ.get("WEAVEPROF_L0_DIR"),
            "WEAVEPROF_INST_DIR": os.environ.get("WEAVEPROF_INST_DIR"),
            "WEAVEPROF_OUT_DIR": os.environ.get("WEAVEPROF_OUT_DIR"),
            "WP_LEVEL": os.environ.get("WP_LEVEL"),
            "FI_LEVEL": os.environ.get("FI_LEVEL"),
            "has_hmac_key": "WEAVEPROF_SUPERVISOR_HMAC_KEY" in os.environ,
            "has_key_file": "WEAVEPROF_SUPERVISOR_KEY_FILE" in os.environ,
        },
    }
    with open(log_file, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(json.dumps(rec) + "\\n")
        fh.flush()
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

(tree_p / "fake_compiled_kernel.cubin").write_bytes(f"FAKE_CUBIN_V1:{role}\\n".encode("ascii"))
"""

_FAKE_BENCH_BODY = """from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil

STAGE_DIR = Path(__STAGE_DIR__)

ap = argparse.ArgumentParser()
ap.add_argument("--locked-session-tag", required=True)
ap.add_argument("--require-build", action="store_true")
ap.add_argument("--l0", required=True)
ap.add_argument("--inst", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tree", required=True)
args = ap.parse_args()

log_file = os.environ.get("P1C_INVOCATION_LOG")
if log_file:
    rec = {
        "kind": "bench",
        "tag": args.locked_session_tag,
        "pid": os.getpid(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "env": {
            "WEAVEPROF_L0_DIR": os.environ.get("WEAVEPROF_L0_DIR"),
            "WEAVEPROF_INST_DIR": os.environ.get("WEAVEPROF_INST_DIR"),
            "WEAVEPROF_OUT_DIR": os.environ.get("WEAVEPROF_OUT_DIR"),
            "WEAVEPROF_NSYS_CAPTURE": os.environ.get("WEAVEPROF_NSYS_CAPTURE"),
            "has_hmac_key": "WEAVEPROF_SUPERVISOR_HMAC_KEY" in os.environ,
            "has_key_file": "WEAVEPROF_SUPERVISOR_KEY_FILE" in os.environ,
        },
    }
    with open(log_file, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(json.dumps(rec) + "\\n")
        fh.flush()
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

if args.require_build:
    l0_prod = Path(args.l0) / "fake_compiled_kernel.cubin"
    inst_prod = Path(args.inst) / "fake_compiled_kernel.cubin"
    if not l0_prod.is_file() or l0_prod.read_bytes() != b"FAKE_CUBIN_V1:L0\\n":
        raise SystemExit(126)
    if not inst_prod.is_file() or inst_prod.read_bytes() != b"FAKE_CUBIN_V1:L_inst\\n":
        raise SystemExit(127)

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
for fname in ("capture.fi02.bin", "runtime.json", "names.json"):
    shutil.copy2(STAGE_DIR / fname, out_dir / fname)
"""

_FAKE_NSYS_BODY = """from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

STAGE_DIR = Path(__STAGE_DIR__)

def _log(rec):
    log_file = os.environ.get("P1C_INVOCATION_LOG")
    if not log_file:
        return
    with open(log_file, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(json.dumps(rec) + "\\n")
        fh.flush()
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

subcmd = sys.argv[1] if len(sys.argv) > 1 else ""
if subcmd == "profile":
    _log({
        "kind": "nsys_profile",
        "pid": os.getpid(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "env": {
            "WEAVEPROF_OUT_DIR": os.environ.get("WEAVEPROF_OUT_DIR"),
            "WEAVEPROF_NSYS_CAPTURE": os.environ.get("WEAVEPROF_NSYS_CAPTURE"),
            "has_hmac_key": "WEAVEPROF_SUPERVISOR_HMAC_KEY" in os.environ,
            "has_key_file": "WEAVEPROF_SUPERVISOR_KEY_FILE" in os.environ,
        },
    })
    idx = sys.argv.index("-o")
    rep_base = sys.argv[idx + 1]
    child_argv = sys.argv[idx + 2:]
    child = subprocess.run(child_argv, env=os.environ.copy(), check=False)
    if child.returncode != 0:
        raise SystemExit(child.returncode)
    Path(rep_base + ".nsys-rep").write_bytes(b"FAKE_NSYS_REP_V1\\n")
    raise SystemExit(0)

if subcmd == "export":
    _log({
        "kind": "nsys_export",
        "pid": os.getpid(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
    })
    out_idx = sys.argv.index("--output")
    sqlite_dst = Path(sys.argv[out_idx + 1])
    rep_src = Path(sys.argv[out_idx + 2])
    if not rep_src.is_file():
        raise SystemExit(2)
    shutil.copy2(STAGE_DIR / "nsys.sqlite", sqlite_dst)
    raise SystemExit(0)

raise SystemExit(2)
"""


def _unxz(src: Path, dst: Path) -> Path:
    with lzma.open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    return dst


def _norm_paths(obj: Any, pairs: List[Tuple[str, str]]) -> Any:
    if isinstance(obj, str):
        for real, token in pairs:
            if real in obj:
                obj = obj.replace(real, token)
        return obj
    if isinstance(obj, dict):
        return {k: _norm_paths(v, pairs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_norm_paths(v, pairs) for v in obj]
    return obj


def _make_pairs(*items: Tuple[Path, str]) -> List[Tuple[str, str]]:
    raw = set()
    for p, tok in items:
        raw.add((str(p), tok))
        raw.add((os.path.realpath(str(p)), tok))
    return sorted(raw, key=lambda kv: -len(kv[0]))


def _pop_nested(doc: Dict[str, Any], key_path: Tuple[str, ...]) -> Any:
    cur: Any = doc
    for part in key_path[:-1]:
        cur = cur[part]
    return cur.pop(key_path[-1])


def _assemble_golden_observation(report: Dict[str, Any], out: Path, tmp_base: Path) -> Dict[str, Any]:
    """Build the exact normalized golden observation bundle for an MCP-executed audit directory."""
    t = str(tmp_base)
    obs: Dict[str, Any] = {"report": golden._norm(report, t)}
    for name in ("verify", "feedback"):
        p = out / f"{name}.json"
        obs[name] = golden._norm(json.loads(p.read_text(encoding="utf-8")), t) if p.is_file() else None
    nsys_p = out / "nsys_hw.json"
    if nsys_p.is_file():
        hw = json.loads(nsys_p.read_text(encoding="utf-8"))
        series = hw.pop("hw_series", {}) or {}
        obs["nsys"] = golden._norm(hw, t)
        obs["nsys_series"] = {
            k: {
                "n": len(c.get("val") or []),
                "metric_name": c.get("metric_name"),
                "sha256": golden._sha(golden._canon([c.get("t_us"), c.get("val")])),
            }
            for k, c in sorted(series.items())
        }
    trace_p = out / "trace.perfetto.json.gz"
    if trace_p.is_file():
        tr = json.loads(gzip.decompress(trace_p.read_bytes()).decode("utf-8"))
        events = golden._norm(tr.get("traceEvents") or [], t)
        by_ph = Counter(str(e.get("ph")) for e in events)
        by_name = Counter(str(e.get("name")) for e in events if e.get("ph") in ("X", "C"))
        obs["trace"] = {
            "events": len(events),
            "sha256": golden._sha(golden._canon(events)),
            "other_data_sha256": golden._sha(golden._canon(golden._norm(tr.get("otherData") or {}, t))),
            "by_ph": dict(sorted(by_ph.items())),
            "by_name": dict(sorted(by_name.items())),
        }
    png = out / "overview.png"
    obs["png"] = {
        "checked": True,
        "valid": png.is_file() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and png.stat().st_size > 1024,
    }
    return obs


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

    def request(self, payload: Dict[str, Any], *, timeout_s: float = 180.0) -> Dict[str, Any]:
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


class WorkflowAMcpTest(unittest.TestCase):
    PID_RECORDS: List[Tuple[int, str, int | None]] = []

    @classmethod
    def setUpClass(cls) -> None:
        cls._saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}
        cls._saved_tempdir = tempfile.tempdir
        cls._root = Path(tempfile.mkdtemp(prefix="wp-p1c-r23-mcp-")).resolve()
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
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        ):
            os.environ.pop(clear_k, None)
        os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = "weaveprof-golden-test-key"
        os.environ["PYTHONPATH"] = str(REPO_ROOT)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

        cls._case_dir = REPO_ROOT / "tests" / "golden" / "cases" / "h20_pingpong_v1"
        cls._case_meta = json.loads((cls._case_dir / "case.json").read_text(encoding="utf-8"))
        cls._stage_dir = cls._root / "golden_stage"
        cls._stage_dir.mkdir(parents=True, exist_ok=True)
        _unxz(cls._case_dir / "capture.fi02.bin.xz", cls._stage_dir / "capture.fi02.bin")
        _unxz(cls._case_dir / "nsys.sqlite.xz", cls._stage_dir / "nsys.sqlite")
        shutil.copy2(cls._case_dir / "runtime.json", cls._stage_dir / "runtime.json")
        shutil.copy2(cls._case_dir / "names.json", cls._stage_dir / "names.json")

        shebang = f"#!{sys.executable} -B\n"
        cls._tools_dir = cls._root / "fake_tools"
        cls._tools_dir.mkdir(parents=True, exist_ok=True)
        cls._fake_build_py = cls._tools_dir / "fake_build.py"
        cls._fake_build_py.write_text(shebang + _FAKE_BUILD_BODY, encoding="utf-8")
        cls._fake_build_py.chmod(0o755)

        cls._fake_bench_py = cls._tools_dir / "fake_bench.py"
        cls._fake_bench_py.write_text(
            shebang + _FAKE_BENCH_BODY.replace("__STAGE_DIR__", repr(str(cls._stage_dir))),
            encoding="utf-8",
        )
        cls._fake_bench_py.chmod(0o755)

        cls._bin_dir = cls._root / "bin"
        cls._bin_dir.mkdir(parents=True, exist_ok=True)
        cls._fake_nsys = cls._bin_dir / "nsys"
        cls._fake_nsys.write_text(
            shebang + _FAKE_NSYS_BODY.replace("__STAGE_DIR__", repr(str(cls._stage_dir))),
            encoding="utf-8",
        )
        cls._fake_nsys.chmod(0o755)

        orig_path = cls._saved_env.get("PATH") or ""
        os.environ["PATH"] = f"{cls._bin_dir}{os.pathsep}{orig_path}"

        cls._build_cmd = [
            sys.executable,
            "-B",
            str(cls._fake_build_py),
            "--l0",
            "{l0}",
            "--inst",
            "{inst}",
            "--out",
            "{out}",
            "--tree",
            "{tree}",
        ]
        cls._bench_cmd = [
            sys.executable,
            "-B",
            str(cls._fake_bench_py),
            "--locked-session-tag",
            "mcp-session-locked-v1",
            "--require-build",
            "--l0",
            "{l0}",
            "--inst",
            "{inst}",
            "--out",
            "{out}",
            "--tree",
            "{tree}",
        ]
        cls._bench_cmd_quoted = " ".join(shlex.quote(tok) for tok in cls._bench_cmd)

        cls._cwd = cls._root / "cwd"
        cls._cwd.mkdir(parents=True, exist_ok=True)
        cls._env = dict(os.environ)

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

        # Provision workspace with locked fake bench_cmd and --allow-gz via agent_installer.main(["install", ...])
        cls._ws = cls._root / "ws_mcp"
        shutil.copytree(cls._case_dir / "workspace", cls._ws)
        buf_inst = io.StringIO()
        with contextlib.redirect_stdout(buf_inst):
            rc_inst = agent_installer.main(
                [
                    "install",
                    "--workspace",
                    str(cls._ws),
                    "--protect",
                    "matmul-pingpong-v1.cu",
                    "--agent",
                    "claude",
                    "--bench",
                    cls._bench_cmd_quoted,
                    "--allow-gz",
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
        cls._install_doc = json.loads(buf_inst.getvalue())

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

    def test_1_mcp_full_workflow_a_audit_and_seal_registration(self) -> None:
        # Assert find_nsys_binary() resolves to our fake nsys script
        resolved_nsys = mk.find_nsys_binary()
        self.assertIsNotNone(resolved_nsys)
        self.assertEqual(Path(str(resolved_nsys)).resolve(), self._fake_nsys.resolve())

        # Assert every path touched by agent_installer resides strictly inside our temporary root (Hard Req #13 T4)
        self.assertTrue(Path(self._install_doc["session_dir"]).resolve().is_relative_to(self._root))
        self.assertTrue(Path(self._install_doc["workspace"]).resolve().is_relative_to(self._root))
        self.assertGreaterEqual(len(self._install_doc["files_written"]), 4)
        for written_p in self._install_doc["files_written"]:
            self.assertTrue(
                Path(written_p).resolve().is_relative_to(self._root),
                f"Installer touched path outside temp root: {written_p}",
            )

        mcp_cfg = json.loads((self._ws / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["weaveprofiler"]
        self.assertEqual(mcp_cfg["type"], "stdio")
        self.assertEqual(mcp_cfg["command"], sys.executable)
        self.assertEqual(
            mcp_cfg["args"],
            [
                "-B",
                str(REPO_ROOT / "weaveprof" / "mcp_server.py"),
                "--workspace",
                str(self._ws),
                "--session-dir",
                self._install_doc["session_dir"],
            ],
        )

        inv_log = self._root / "inv_mcp_workflow_a.jsonl"
        cache_dir = self._root / "cache_mcp_workflow_a"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_dir = self._root / "mcp_audit_out"

        srv_env = dict(self._env)
        srv_env["WEAVEPROF_CACHE"] = str(cache_dir)
        srv_env["P1C_INVOCATION_LOG"] = str(inv_log)

        srv = _McpSessionHandle(
            [mcp_cfg["command"], *mcp_cfg["args"]],
            cwd=self._cwd,
            env=srv_env,
            stderr_path=self._root / "mcp_workflow_a.stderr.log",
            label="mcp_workflow_a",
        )
        self._handles.append(srv)

        r_init = srv.request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "p1c-r23", "version": "1.0"},
                },
            }
        )
        self.assertEqual(r_init["result"]["protocolVersion"], "2025-06-18")

        # Trigger full Workflow A audit via MCP (omitting workspace/entry_source/bench_cmd/card/arch/limit
        # so session defaults & locked bench_cmd are exercised end to end)
        r_audit = srv.request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "weaveprof_megakernel_audit",
                    "arguments": {
                        "build_cmd": list(self._build_cmd),
                        "capture_nsys": True,
                        "total_sms": 78,
                        "overview_title": self._case_meta["overview_title"],
                        "render_overview_png": True,
                        "write_perfetto_gz": True,
                        "out_dir": str(out_dir),
                    },
                },
            }
        )
        self.assertIs(r_audit["result"]["isError"], False, f"MCP Workflow A failed: {r_audit}")
        sc = dict(r_audit["result"]["structuredContent"])

        # 1. Assert weaveprof_session registration literals (MCP seals use session key, not golden key)
        wp_sess = sc.pop("weaveprof_session")
        self.assertIs(wp_sess["registered"], True)
        self.assertEqual(wp_sess["status"], "PASS")
        self.assertIs(wp_sess["covers_protected"], True)
        self.assertIsNone(wp_sess["note"])
        self.assertEqual(wp_sess["errors"], [])
        self.assertIs(wp_sess["edit_unlocked"], True)
        self.assertIs(wp_sess["stop_unlocked"], True)
        self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{64}", str(sc["seal_sha256"])))
        self.assertEqual(
            wp_sess["incumbent"],
            {
                "kernel_us": 1067.939,
                "seal_sha256": sc["seal_sha256"],
                "overhead_pct": 3.9501,
                "protected_shas": {
                    "matmul-pingpong-v1.cu": "6cea508de081153f26edfd9aa4d1269ebd7d3556e48168efad354fba9991e976"
                },
            },
        )

        # 2. Assert MCP audit report concrete literal fields independently (never comparing against direct test)
        self.assertEqual(sc["schema"], "weaveprof.megakernel-audit/v1")
        self.assertEqual(sc["status"], "PASS")
        self.assertEqual(sc["mode"], "shadow_build_run")
        self.assertIs(sc["seal_verified"], True)
        self.assertEqual(sc["overhead_pct"], 3.95)
        self.assertEqual(sc["kernel_us"], 1067.939)
        self.assertEqual(sc["tensor_active_scale"], 4.0)
        self.assertEqual(sc["causal_opportunities"], [])
        self.assertEqual(sc["structural_blockers"], ["LOW_MATH_WAIT_NO_LATENCY_HIDING"])
        self.assertIsNone(sc["round_diff"])
        self.assertEqual(
            sc["artifacts"]["files"],
            ["feedback.json", "verify.json", "overview.png", "trace.perfetto.json.gz"],
        )
        self.assertEqual(
            sc["fidelity_gates"],
            {
                "compiler_spill": {"codes": [], "ok": True, "reg_blind_spot": False},
                "granularity": {"ok": True},
                "output": {"ok": True},
                "overflow": 0,
                "overhead": {"ok": True, "ratio": 0.0395},
                "pairing_errors": 0,
                "stub": {"ok": True, "stubs": []},
                "warps": {"ok": True},
            },
        )
        self.assertEqual(
            sc["hw_quartiles"],
            {
                "dram_read": [7.1, 5.1, 5.2, 3.1],
                "dram_write": [3.4, 1.6, 1.5, 1.3],
                "sm_active": [100.0, 100.0, 100.0, 74.0],
                "tensor_active": [96.5, 96.1, 96.1, 69.2],
                "warp_issue": [12.7, 12.9, 12.9, 9.4],
            },
        )
        self.assertEqual(
            sc["megakernel_skew"],
            {
                "barrier_starved": 0,
                "early": [0, None],
                "hints": ["WARP_GROUP_REBALANCE", "OPTIMIZE_DOMINANT_STAGE"],
                "idle_sm_fraction": 0.0,
                "imbalance_pct": 0.4,
                "role_skew_ratio": 0.9992,
                "sm_busy_us": {
                    "max": 520.958,
                    "median": 519.199,
                    "min": 516.047,
                    "p90": 519.969,
                    "spread": 4.911,
                },
                "sm_spread_ratio": 0.0094,
                "stragglers": [78, 519.199],
            },
        )
        self.assertEqual(
            sc["stage_summary"],
            {
                "barrier": [4096, 64.69, 11.1],
                "compute": [262144, 1.93, 21.1],
                "epilogue": [4096, 5.01, 0.9],
                "load": [262144, 5.73, 62.7],
                "mma": [262144, 0.12, 1.3],
                "wait": [65536, 1.07, 2.9],
            },
        )

        # 3. Compare full observation bundle against golden expected.json using the same 3 registered diff paths
        obs = _assemble_golden_observation(sc, out_dir, self._root)
        self.assertEqual(obs["png"], {"checked": True, "valid": True})
        expected = json.loads((self._case_dir / "expected.json").read_text(encoding="utf-8"))
        self.assertEqual(obs["report"]["mode"], "shadow_build_run")
        self.assertEqual(obs["feedback"]["mode"], "shadow_build_run")
        l0_cache = obs["feedback"]["l0_cache"]
        self.assertEqual(set(l0_cache.keys()), {"enabled", "hit", "key", "stored_products"})
        self.assertIs(l0_cache["enabled"], True)
        self.assertIs(l0_cache["hit"], False)
        self.assertEqual(l0_cache["stored_products"], 1)
        self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{16}", str(l0_cache["key"])))

        obs_cmp = json.loads(golden._canon(obs))
        for key_path, _json_path, _reason in WORKFLOW_A_EXPECTED_DIFF_PATHS:
            _pop_nested(obs_cmp, key_path)
            _pop_nested(expected, key_path)
        diff = golden._first_diff(expected, obs_cmp)
        self.assertIsNone(diff, f"Unexpected MCP Workflow A diff outside registered paths: {diff}")

        # 4. Verify weaveprof_status and weaveprof_verify_seal over the live MCP session
        r_status = srv.request(
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "weaveprof_status", "arguments": {}}}
        )
        st_sc = r_status["result"]["structuredContent"]
        self.assertIs(st_sc["installed"], True)
        self.assertEqual(st_sc["seals_recorded"], 1)
        self.assertEqual(st_sc["valid_pass_seals"], 1)
        self.assertIs(st_sc["edit_unlocked"], True)
        self.assertIs(st_sc["stop_unlocked"], True)
        self.assertEqual(st_sc["stop_blockers"], [])

        r_ver = srv.request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "weaveprof_verify_seal", "arguments": {"seal_path": sc["seal_path"]}},
            }
        )
        ver_sc = r_ver["result"]["structuredContent"]
        self.assertIs(ver_sc["ok"], True)
        self.assertEqual(ver_sc["errors"], [])

        # 5. Verify all 6 child invocations ran with the exact session-locked bench_cmd and stripped secrets
        pairs = _make_pairs((self._root, "<TMP>"), (REPO_ROOT, "<REPO>"))
        raw_rows = [json.loads(line) for line in inv_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        for r in raw_rows:
            self.__class__.PID_RECORDS.append((int(r["pid"]), f"mcp_child:{r['kind']}", 0))
        rows = _norm_paths(raw_rows, pairs)
        self.assertEqual(len(rows), 6)
        self.assertEqual({rows[0]["kind"], rows[1]["kind"]}, {"build"})
        self.assertEqual({rows[0]["role"], rows[1]["role"]}, {"L0", "L_inst"})
        self.assertEqual([r["kind"] for r in rows[2:]], ["bench", "nsys_profile", "bench", "nsys_export"])
        self.assertEqual(rows[2]["tag"], "mcp-session-locked-v1")
        self.assertEqual(rows[4]["tag"], "mcp-session-locked-v1")
        for r in rows[:5]:
            self.assertIs(r["env"]["has_hmac_key"], False)
            self.assertIs(r["env"]["has_key_file"], False)


if __name__ == "__main__":
    unittest.main()
