#!/usr/bin/env python3
"""Hermetic tests for the WeaveProfiler (`wp`) MCP server, enforcement hook and agent installer.

Run:  python -B -m unittest tests/test_mcp_hook_installer.py
All state lives in tempfile.TemporaryDirectory (WEAVEPROF_SESSIONS_DIR points there).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import contextlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.agent_installer import _merge_codex_toml, install_agent_workspace  # noqa: E402
from weaveprof.guard_hook import (  # noqa: E402
    read_session,
    register_seal,
    session_dir_for,
    supervisor_env,
    verify_session_completion,
)
from weaveprof.harness import EpisodeSealer, HarnessPolicy  # noqa: E402

try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # Python 3.10
    tomllib = None

GUARD = ROOT / "weaveprof" / "guard_hook.py"
MCP = ROOT / "weaveprof" / "mcp_server.py"
KERNEL = (
    "#include <cuda_runtime.h>\n"
    "__global__ void vadd(const float* a, const float* b, float* c, int n) {\n"
    "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "  if (i < n) c[i] = a[i] + b[i];\n"
    "}\n"
)
OPTIMIZED = KERNEL.replace("if (i < n) c[i] = a[i] + b[i];", "if (i < n) c[i] = __fadd_rn(a[i], b[i]);")


def _child_env(sessions: Path) -> dict:
    env = dict(os.environ)
    env["WEAVEPROF_SESSIONS_DIR"] = str(sessions)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("WEAVEPROF_SUPERVISOR_HMAC_KEY", None)
    env.pop("WEAVEPROF_SUPERVISOR_KEY_FILE", None)
    return env


class _WorkspaceCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="wp_agent_")
        self.tmp = Path(self._tmp.name).resolve()
        self.ws = self.tmp / "ws"
        self.ws.mkdir()
        self.sessions = self.tmp / "sessions"
        self.kernel = self.ws / "kernel.cu"
        self.kernel.write_bytes(KERNEL.encode("utf-8"))
        (self.ws / "bench.py").write_text("print('bench')\n", encoding="utf-8", newline="\n")
        self._saved_env = {k: os.environ.get(k) for k in (
            "WEAVEPROF_SESSIONS_DIR", "WEAVEPROF_SUPERVISOR_HMAC_KEY", "WEAVEPROF_SUPERVISOR_KEY_FILE")}
        os.environ["WEAVEPROF_SESSIONS_DIR"] = str(self.sessions)
        os.environ.pop("WEAVEPROF_SUPERVISOR_HMAC_KEY", None)
        os.environ.pop("WEAVEPROF_SUPERVISOR_KEY_FILE", None)
        self.summary = install_agent_workspace(self.ws, ["kernel.cu"], bench_cmd="python bench.py",
                                               sessions_root=self.sessions)
        self.sd = Path(self.summary["session_dir"])
        self.env = _child_env(self.sessions)
        self._seal_n = 0

    def tearDown(self) -> None:
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        default_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "weaveprof" / "sessions"
        leaked = default_root / session_dir_for(self.ws, self.sessions).name
        self._tmp.cleanup()
        self.assertFalse(leaked.exists(), f"session leaked into real cache: {leaked}")

    @staticmethod
    def _make_png(width: int = 400, height: int = 200) -> bytes:
        def _chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        raw = b"".join(b"\x00" + (b"\x20\x40\x80" * width) for _ in range(height))
        return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 9)) + _chunk(b"IEND", b"")

    _TINY_PNG = _make_png.__func__(400, 200)
    _FAKE_1X1_PNG = _make_png.__func__(1, 1)

    # -- helpers -----------------------------------------------------------
    def make_seal(self, *, status: str = "PASS", evil_key: bool = False, include_png: bool = True) -> Path:
        self._seal_n += 1
        run = self.tmp / f"run{self._seal_n}"
        run.mkdir()
        feedback = run / "feedback.json"
        feedback.write_text(json.dumps({"status": status}), encoding="utf-8")
        trace = run / "out.perfetto-trace"
        trace.write_bytes(b"\x0a\x00trace")
        extra_arts = {}
        if include_png:
            png = run / "overview.png"
            png.write_bytes(self._TINY_PNG)
            extra_arts["overview_png"] = png
        result = {"status": status, "checks": {"assembly": True, "output": True, "events": True, "overhead": True},
                  "metrics": {"overhead_ratio": 0.031}}
        ctx = contextlib.nullcontext()
        if evil_key:
            evil = self.tmp / f"evil{self._seal_n}.key"
            evil.write_text("f" * 64, encoding="ascii")
            ctx = _env_key(evil)
        else:
            ctx = supervisor_env(self.sd)
        with ctx:
            seal, _doc = EpisodeSealer.seal_round(
                round_dir=run, round_number=self._seal_n, source_path=self.kernel,
                source_sha256=_sha(self.kernel), probe_path=None, instrumented_path=None, names_path=None,
                feedback_path=feedback, verify_path=None, trace_path=trace, failed_trace_path=None,
                result=result, policy=HarnessPolicy(), limit=0.10, region_lock={"ok": True, "gpu": 0},
                extra_artifacts=extra_arts or None,
                out_seal_path=run / "out.perfetto-trace.seal.json")
        return seal

    def hook(self, event: str, payload: dict) -> tuple[int, str]:
        payload = dict(payload)
        payload.setdefault("hook_event_name", event)
        payload.setdefault("cwd", str(self.ws))
        proc = subprocess.run(
            [sys.executable, "-B", str(GUARD), "--event", event, "--workspace", str(self.ws),
             "--session-dir", str(self.sd)],
            input=json.dumps(payload).encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self.env, timeout=120)
        return proc.returncode, proc.stderr.decode("utf-8", "replace")

    def edit(self, new_string: str) -> tuple[int, str]:
        return self.hook("PreToolUse", {"tool_name": "Edit", "tool_input": {
            "file_path": str(self.kernel), "old_string": "c[i] = a[i] + b[i];", "new_string": new_string}})


@contextlib.contextmanager
def _env_key(path: Path):
    saved = os.environ.get("WEAVEPROF_SUPERVISOR_KEY_FILE")
    os.environ["WEAVEPROF_SUPERVISOR_KEY_FILE"] = str(path)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("WEAVEPROF_SUPERVISOR_KEY_FILE", None)
        else:
            os.environ["WEAVEPROF_SUPERVISOR_KEY_FILE"] = saved


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Installer
# ---------------------------------------------------------------------------

class InstallerTests(_WorkspaceCase):
    def test_install_creates_all_artifacts(self) -> None:
        ws, sd = self.ws, self.sd
        self.assertTrue(str(sd).startswith(str(self.sessions)))
        self.assertTrue((sd / ".supervisor.key").is_file())
        self.assertEqual(len((sd / ".supervisor.key").read_bytes().strip()), 64)
        self.assertEqual((sd / "backups" / "kernel.cu").read_bytes(), KERNEL.encode("utf-8"))
        session = read_session(sd)
        self.assertEqual(session["protected"][0]["baseline_sha256"], _sha(self.kernel))
        self.assertEqual(session["bench_cmd"], ["python", "bench.py"])
        self.assertTrue(session["require_final_seal"])

        mcp = json.loads((ws / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["weaveprofiler"]
        self.assertTrue(any(a.endswith("mcp_server.py") for a in mcp["args"]))
        self.assertIn("--session-dir", mcp["args"])
        settings = json.loads((ws / ".claude/settings.json").read_text(encoding="utf-8"))
        local = json.loads((ws / ".claude/settings.local.json").read_text(encoding="utf-8"))
        self.assertNotIn("enableAllProjectMcpServers", local)
        self.assertIn("weaveprofiler", local.get("enabledMcpjsonServers", []))
        codex_hooks = json.loads((ws / ".codex/hooks.json").read_text(encoding="utf-8"))
        for doc in (settings, codex_hooks):
            for ev in ("PreToolUse", "PostToolUse", "Stop"):
                cmds = [h["command"] for g in doc["hooks"][ev] for h in g["hooks"]]
                self.assertEqual(sum("guard_hook.py" in c for c in cmds), 1, ev)
                self.assertTrue(any(f"--event {ev}" in c for c in cmds))
        for rel in ("CLAUDE.md", "AGENTS.md", "QWEN.md", "GEMINI.md"):
            text = (ws / rel).read_text(encoding="utf-8")
            self.assertEqual(text.count("<!-- WEAVEPROFILER:BEGIN -->"), 1)
            self.assertIn("weaveprof_megakernel_audit", text)
            self.assertIn("`kernel.cu`", text)
        for rel in (
            ".claude/skills/weaveprofiler/SKILL.md",
            ".agents/skills/weaveprofiler/SKILL.md",
            ".qwen/skills/weaveprofiler/SKILL.md",
            ".agents/plugins/weaveprofiler/skills/weaveprofiler/SKILL.md",
        ):
            text = (ws / rel).read_text(encoding="utf-8")
            self.assertTrue(text.startswith("---\n"))
            self.assertIn("Installed WeaveProfiler (`wp`) skill copy", text)
        # Qwen Code settings
        qwen_settings = json.loads((ws / ".qwen/settings.json").read_text(encoding="utf-8"))
        self.assertIs(qwen_settings["disableAllHooks"], False)
        self.assertTrue(any(a.endswith("mcp_server.py") for a in qwen_settings["mcpServers"]["weaveprofiler"]["args"]))
        for ev in ("PreToolUse", "PostToolUse", "PostToolUseFailure", "Stop", "SubagentStop"):
            cmds = [h["command"] for g in qwen_settings["hooks"][ev] for h in g["hooks"]]
            self.assertEqual(sum("guard_hook.py" in c for c in cmds), 1, ev)
            self.assertTrue(any("--event" in c and ev in c and "--protocol" in c and "qwen" in c for c in cmds), (ev, cmds))
        # Antigravity workspace plugin bundle
        agy_plugin = json.loads((ws / ".agents/plugins/weaveprofiler/plugin.json").read_text(encoding="utf-8"))
        self.assertEqual(agy_plugin["name"], "weaveprofiler")
        agy_mcp = json.loads((ws / ".agents/plugins/weaveprofiler/mcp_config.json").read_text(encoding="utf-8"))
        self.assertTrue(any(a.endswith("mcp_server.py") for a in agy_mcp["mcpServers"]["weaveprofiler"]["args"]))
        agy_hooks = json.loads((ws / ".agents/plugins/weaveprofiler/hooks.json").read_text(encoding="utf-8"))
        self.assertIn("weaveprofiler", agy_hooks)
        for ev in ("PreToolUse", "PostToolUse"):
            groups = agy_hooks["weaveprofiler"][ev]
            self.assertEqual(groups[0]["matcher"], "*")
            self.assertIn(f"--event {ev}", groups[0]["hooks"][0]["command"])
            self.assertIn("--protocol agy", groups[0]["hooks"][0]["command"])
        for ev in ("PostInvocation", "Stop"):
            flat = agy_hooks["weaveprofiler"][ev]
            self.assertIsInstance(flat, list)
            self.assertIn("command", flat[0])
            self.assertNotIn("hooks", flat[0])
            self.assertIn(f"--event {ev}", flat[0]["command"])
            self.assertIn("--protocol agy", flat[0]["command"])
        toml_text = (ws / ".codex/config.toml").read_text(encoding="utf-8")
        self.assertIn("[mcp_servers.weaveprofiler]", toml_text)
        if tomllib is not None:
            cfg = tomllib.loads(toml_text)
            self.assertIs(cfg["features"]["hooks"], True)
            self.assertTrue(cfg["mcp_servers"]["weaveprofiler"]["args"][1].endswith("mcp_server.py"))
        for path in ws.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".md", ".toml"):
                self.assertNotIn(b"\r\n", path.read_bytes(), path)
        launch = self.summary["launch"]
        self.assertIn("claude", launch["claude"])
        self.assertIn("codex", launch["codex"])
        self.assertIn("qwen", launch["qwen"])
        self.assertIn("agy", launch["agy"])
        self.assertIn("guard-exec", launch["guard_exec"])

    def test_install_is_idempotent_and_preserves_user_config(self) -> None:
        ws = self.ws
        settings_path = ws / ".claude/settings.json"
        doc = json.loads(settings_path.read_text(encoding="utf-8"))
        doc["hooks"]["PreToolUse"].insert(0, {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo user"}]})
        doc["model"] = "opus"
        settings_path.write_text(json.dumps(doc), encoding="utf-8")
        cfg = ws / ".codex/config.toml"
        old = cfg.read_text(encoding="utf-8")
        mcp_part = old[old.index("[mcp_servers.weaveprofiler]"):]
        cfg.write_text('model = "o3"\n\n[features]\nweb_search = true\nhooks = false\n\n' + mcp_part
                       + '\n[mcp_servers.other]\ncommand = "x"\n', encoding="utf-8")
        (ws / "CLAUDE.md").write_text("# My project\n\n" + (ws / "CLAUDE.md").read_text(encoding="utf-8"),
                                      encoding="utf-8")
        key_before = (self.sd / ".supervisor.key").read_bytes()
        install_agent_workspace(ws, ["kernel.cu"], sessions_root=self.sessions)
        install_agent_workspace(ws, ["kernel.cu"], sessions_root=self.sessions)
        self.assertEqual((self.sd / ".supervisor.key").read_bytes(), key_before)
        doc = json.loads(settings_path.read_text(encoding="utf-8"))
        self.assertEqual(doc["model"], "opus")
        cmds = [h["command"] for g in doc["hooks"]["PreToolUse"] for h in g["hooks"]]
        self.assertIn("echo user", cmds)
        self.assertEqual(sum("guard_hook.py" in c for c in cmds), 1)
        claude_md = (ws / "CLAUDE.md").read_text(encoding="utf-8")
        self.assertTrue(claude_md.startswith("# My project"))
        self.assertEqual(claude_md.count("<!-- WEAVEPROFILER:END -->"), 1)
        text = cfg.read_text(encoding="utf-8")
        self.assertEqual(text.count("[mcp_servers.weaveprofiler]"), 1)
        self.assertEqual(text.count("[features]"), 1)
        if tomllib is not None:
            parsed = tomllib.loads(text)
            self.assertEqual(parsed["model"], "o3")
            self.assertIs(parsed["features"]["hooks"], True)
            self.assertIs(parsed["features"]["web_search"], True)
            self.assertEqual(parsed["mcp_servers"]["other"]["command"], "x")

    def test_toml_escaping_windows_and_posix_paths(self) -> None:
        text = _merge_codex_toml("", "C:\\Program Files\\Python\\python.exe",
                                 ["-B", "C:\\Users\\a b\\\"q\"\\mcp_server.py", "/home/u/ws", "tab\there"])
        if tomllib is None:
            self.skipTest("tomllib unavailable")
        cfg = tomllib.loads(text)
        srv = cfg["mcp_servers"]["weaveprofiler"]
        self.assertEqual(srv["command"], "C:\\Program Files\\Python\\python.exe")
        self.assertEqual(srv["args"][1], "C:\\Users\\a b\\\"q\"\\mcp_server.py")
        self.assertEqual(srv["args"][3], "tab\there")

    def test_guard_exec_reverts_and_reports_completion(self) -> None:
        from weaveprof.agent_installer import guard_exec

        code = ("import os,sys,time\n"
                "assert 'WEAVEPROF_SUPERVISOR_KEY_FILE' not in os.environ\n"
                "open('kernel.cu','w').write('// mutated\\n'); time.sleep(0.6)\n")
        os.environ["WEAVEPROF_SUPERVISOR_KEY_FILE"] = str(self.sd / ".supervisor.key")
        res = guard_exec(self.ws, [sys.executable, "-B", "-c", code], use_bwrap="never",
                         sessions_root=self.sessions)
        os.environ.pop("WEAVEPROF_SUPERVISOR_KEY_FILE", None)
        self.assertEqual(res["agent_returncode"], 0)
        self.assertFalse(res["ok"])
        self.assertTrue(res["reverts"])
        self.assertEqual(self.kernel.read_bytes(), KERNEL.encode("utf-8"))
        self.assertTrue(any("NO_VERIFIED_PASS_SEAL" in b for b in res["completion"]["blockers"]))

    def test_autodiscovery_and_validation(self) -> None:
        other = self.tmp / "ws2"
        (other / "src").mkdir(parents=True)
        (other / "src" / "attn.cuh").write_text("// k\n", encoding="utf-8")
        (other / "tri.py").write_text("import triton\n@triton.jit\ndef k(x):\n    pass\n", encoding="utf-8")
        (other / "util.py").write_text("print(1)\n", encoding="utf-8")
        res = install_agent_workspace(other, agent="codex", sessions_root=self.sessions)
        self.assertEqual(sorted(p["rel"] for p in res["protected"]), ["src/attn.cuh", "tri.py"])
        self.assertFalse((other / ".mcp.json").exists())
        self.assertTrue((other / ".codex/hooks.json").is_file())
        with self.assertRaises(FileNotFoundError):
            install_agent_workspace(other, ["missing.cu"], sessions_root=self.sessions)
        empty = self.tmp / "ws3"
        empty.mkdir()
        with self.assertRaises(ValueError):
            install_agent_workspace(empty, sessions_root=self.sessions)

    def test_single_agent_qwen_and_agy_isolation(self) -> None:
        ws_q = self.tmp / "ws_qwen"
        ws_q.mkdir()
        (ws_q / "k.cu").write_bytes(KERNEL.encode("utf-8"))
        install_agent_workspace(ws_q, ["k.cu"], agent="qwen", sessions_root=self.sessions)
        self.assertTrue((ws_q / ".qwen/settings.json").is_file())
        self.assertTrue((ws_q / "QWEN.md").is_file())
        self.assertFalse((ws_q / ".claude").exists())
        self.assertFalse((ws_q / ".codex").exists())
        self.assertFalse((ws_q / ".agents").exists())

        ws_a = self.tmp / "ws_agy"
        ws_a.mkdir()
        (ws_a / "k.cu").write_bytes(KERNEL.encode("utf-8"))
        install_agent_workspace(ws_a, ["k.cu"], agent="antigravity", sessions_root=self.sessions)
        self.assertTrue((ws_a / ".agents/plugins/weaveprofiler/hooks.json").is_file())
        self.assertTrue((ws_a / ".agents/plugins/weaveprofiler/mcp_config.json").is_file())
        self.assertTrue((ws_a / "GEMINI.md").is_file())
        self.assertTrue((ws_a / "AGENTS.md").is_file())
        self.assertFalse((ws_a / ".claude").exists())
        self.assertFalse((ws_a / ".codex").exists())
        self.assertFalse((ws_a / ".qwen").exists())


# ---------------------------------------------------------------------------
# MCP server over real stdio pipes
# ---------------------------------------------------------------------------

class McpServerTests(_WorkspaceCase):
    def _requests(self, protocol: str, seal: Path) -> list:
        return [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": protocol, "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "weaveprof_status", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
             "params": {"name": "weaveprof_plan_context", "arguments": {"source_path": "kernel.cu"}}},
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
             "params": {"name": "weaveprof_verify_seal", "arguments": {"seal_path": str(seal)}}},
            {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "weaveprof_status"}},
            {"jsonrpc": "2.0", "id": 7, "method": "ping"},
            {"jsonrpc": "2.0", "id": 8, "method": "no/such"},
            {"jsonrpc": "2.0", "id": 9, "method": "resources/list"},
            {"jsonrpc": "2.0", "id": 10, "method": "prompts/list"},
        ]

    def _spawn(self, data: bytes) -> bytes:
        with tempfile.TemporaryFile() as err:
            proc = subprocess.run(
                [sys.executable, "-B", str(MCP), "--workspace", str(self.ws), "--session-dir", str(self.sd)],
                input=data, stdout=subprocess.PIPE, stderr=err, env=self.env, timeout=180)
            err.seek(0)
            self.assertEqual(proc.returncode, 0, err.read().decode("utf-8", "replace")[-2000:])
        return proc.stdout

    def _check(self, replies: dict, expected_protocol: str, seal: Path) -> None:
        self.assertEqual(sorted(replies), list(range(1, 11)))
        self.assertEqual(replies[1]["result"]["protocolVersion"], expected_protocol)
        self.assertEqual(replies[1]["result"]["serverInfo"]["name"], "weaveprofiler")
        names = {t["name"] for t in replies[2]["result"]["tools"]}
        self.assertEqual(names, {"weaveprof_status", "weaveprof_megakernel_audit", "weaveprof_profile",
                                 "weaveprof_plan_context", "weaveprof_analyze", "weaveprof_compare_rounds",
                                 "weaveprof_verify_seal"})
        for tool in replies[2]["result"]["tools"]:
            self.assertEqual(tool["inputSchema"]["type"], "object")
        st = json.loads(replies[3]["result"]["content"][0]["text"])
        self.assertTrue(st["installed"])
        self.assertFalse(st["edit_unlocked"])
        self.assertFalse(st["stop_unlocked"])
        self.assertEqual(st["protected"][0]["live_sha256"], st["protected"][0]["baseline_sha256"])
        self.assertFalse(replies[4]["result"]["isError"], replies[4]["result"]["content"][0]["text"][:800])
        json.loads(replies[4]["result"]["content"][0]["text"])
        vs = json.loads(replies[5]["result"]["content"][0]["text"])
        self.assertTrue(vs["ok"], vs.get("errors"))
        self.assertTrue(vs["weaveprof_session"]["registered"])
        self.assertTrue(vs["weaveprof_session"]["edit_unlocked"])
        st2 = json.loads(replies[6]["result"]["content"][0]["text"])
        self.assertTrue(st2["edit_unlocked"])
        self.assertTrue(st2["stop_unlocked"])
        self.assertEqual(replies[7]["result"], {})
        self.assertEqual(replies[8]["error"]["code"], -32601)
        self.assertEqual(replies[9]["result"], {"resources": []})
        self.assertEqual(replies[10]["result"], {"prompts": []})
        entry = read_session(self.sd)["seals"][-1]
        self.assertEqual(entry["status"], "PASS")
        self.assertEqual(entry["source_sha256"], _sha(self.kernel))
        self.assertEqual(entry["protected_shas"], {"kernel.cu": _sha(self.kernel)})
        for k in ("seal_path", "seal_sha256", "source_path", "overhead_pct", "kernel_us", "timestamp"):
            self.assertIn(k, entry)

    def test_newline_delimited_jsonrpc(self) -> None:
        seal = self.make_seal()
        data = b"".join(json.dumps(m).encode("utf-8") + b"\n" for m in self._requests("2025-03-26", seal))
        out = self._spawn(data)
        replies = {}
        for line in out.splitlines():
            if line.strip():
                msg = json.loads(line)
                replies[msg["id"]] = msg
        self._check(replies, "2025-03-26", seal)

    def test_content_length_framed_jsonrpc(self) -> None:
        seal = self.make_seal()
        data = b""
        for m in self._requests("1999-01-01", seal):
            body = json.dumps(m).encode("utf-8")
            data += b"Content-Length: %d\r\n\r\n" % len(body) + body
        out = self._spawn(data)
        replies = {}
        while out:
            head, _, rest = out.partition(b"\r\n\r\n")
            self.assertTrue(head.lower().startswith(b"content-length:"), head[:80])
            n = int(head.split(b":", 1)[1])
            msg = json.loads(rest[:n])
            replies[msg["id"]] = msg
            out = rest[n:]
        self._check(replies, "2025-06-18", seal)


# ---------------------------------------------------------------------------
# Adversarial hook scenarios (real subprocesses)
# ---------------------------------------------------------------------------

class GuardHookTests(_WorkspaceCase):
    def test_1_read_supervisor_key_blocked(self) -> None:
        rc, err = self.hook("PreToolUse", {"tool_name": "Read", "tool_input": {"file_path": str(self.sd / ".supervisor.key")}})
        self.assertEqual(rc, 2)
        self.assertIn("BLOCKED", err)
        rc, err = self.hook("PreToolUse", {"tool_name": "Bash", "tool_input": {
            "command": "cat ~/.cache/weaveprof/sessions/*/.supervisor.key"}})
        self.assertEqual(rc, 2)
        rc, _ = self.hook("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "WP_SAMPLE=4 python bench.py"}})
        self.assertEqual(rc, 2)
        rc, _ = self.hook("PreToolUse", {"tool_name": "Write", "tool_input": {
            "file_path": str(self.tmp / "run1" / "out.perfetto-trace.seal.json"), "content": "{}"}})
        self.assertEqual(rc, 2)
        rc, _ = self.hook("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "rm -f run1/episode_ledger.json"}})
        self.assertEqual(rc, 2)
        rc, _ = self.hook("PreToolUse", {"tool_name": "Edit", "tool_input": {
            "file_path": str(self.ws / ".claude" / "settings.json"), "old_string": "a", "new_string": "b"}})
        self.assertEqual(rc, 2)
        rc, err = self.hook("PreToolUse", {"tool_name": "Read", "tool_input": {"file_path": str(self.kernel)}})
        self.assertEqual(rc, 0, err)
        rc, err = self.hook("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "python bench.py 2>&1 | tail"}})
        self.assertEqual(rc, 0, err)

    def test_2_edit_protected_before_seal_blocked(self) -> None:
        rc, err = self.edit("c[i] = __fadd_rn(a[i], b[i]);")
        self.assertEqual(rc, 2)
        self.assertIn("SEAL_BEFORE_OPTIMIZE", err)
        self.assertIn("weaveprof_megakernel_audit", err)
        rc, _ = self.hook("PreToolUse", {"tool_name": "apply_patch", "tool_input": {"command": (
            "*** Begin Patch\n*** Update File: kernel.cu\n@@\n-  if (i < n) c[i] = a[i] + b[i];\n"
            "+  if (i < n) c[i] = a[i] * b[i];\n*** End Patch\n")}})
        self.assertEqual(rc, 2)
        rc, _ = self.hook("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "sed -i 's/+/*/' kernel.cu"}})
        self.assertEqual(rc, 2)
        rc, err = self.hook("PreToolUse", {"tool_name": "Write", "tool_input": {
            "file_path": str(self.ws / "notes.txt"), "content": "ok"}})
        self.assertEqual(rc, 0, err)

    def test_3_bash_mutation_reverted_by_post_hook(self) -> None:
        self.kernel.write_bytes(b"// hijacked by python -c\n")
        rc, err = self.hook("PostToolUse", {"tool_name": "Bash", "tool_input": {"command": "python evil.py"},
                                            "tool_response": {"stdout": ""}})
        self.assertEqual(rc, 2)
        self.assertIn("restored", err)
        self.assertEqual(self.kernel.read_bytes(), KERNEL.encode("utf-8"))
        rc, err = self.hook("PostToolUse", {"tool_name": "Bash", "tool_input": {"command": "ls"}})
        self.assertEqual(rc, 0, err)

    def test_4_forged_seal_does_not_unlock(self) -> None:
        forged = self.make_seal(evil_key=True)
        self.assertFalse(register_seal(self.sd, forged)["registered"])
        doc = json.loads(forged.read_text(encoding="utf-8"))
        session = read_session(self.sd)
        session["seals"] = [{"seal_path": str(forged), "seal_sha256": doc["seal_sha256"], "status": "PASS",
                             "covers_protected": True, "source_sha256": _sha(self.kernel),
                             "protected_shas": {"kernel.cu": _sha(self.kernel)}}]
        (self.sd / "session.json").write_text(json.dumps(session), encoding="utf-8")
        rc, err = self.hook("Stop", {"stop_hook_active": False})
        self.assertEqual(rc, 2)
        self.assertIn("NO_VERIFIED_PASS_SEAL", err)
        rc, _ = self.edit("c[i] = __fadd_rn(a[i], b[i]);")
        self.assertEqual(rc, 2)

    def test_5_valid_seal_unlocks_clean_edit_but_not_probes(self) -> None:
        reg = register_seal(self.sd, self.make_seal(), kernel_us=12.5)
        self.assertTrue(reg["registered"], reg)
        self.assertEqual(reg["status"], "PASS")
        rc, err = self.edit("c[i] = __fadd_rn(a[i], b[i]);")
        self.assertEqual(rc, 0, err)
        rc, err = self.edit("WP_OPEN(0); c[i] = a[i] + b[i];")
        self.assertEqual(rc, 2)
        self.assertIn("PROBE_POLLUTION", err)
        rc, _ = self.edit("FI_OPEN(0); c[i] = a[i] + b[i];")
        self.assertEqual(rc, 2)
        rc, _ = self.hook("PreToolUse", {"tool_name": "Write", "tool_input": {
            "file_path": str(self.kernel), "content": '#include "wp_lite.cuh"\n' + KERNEL}})
        self.assertEqual(rc, 2)
        self.kernel.write_bytes(('#include "wp_lite.cuh"\n' + KERNEL).encode("utf-8"))
        rc, err = self.hook("PostToolUse", {"tool_name": "Bash", "tool_input": {"command": "x"}})
        self.assertEqual(rc, 2)
        self.assertIn("probe", err)

    def test_6_stop_requires_final_seal(self) -> None:
        self.assertTrue(register_seal(self.sd, self.make_seal(), kernel_us=12.5)["registered"])
        rc, err = self.hook("Stop", {})
        self.assertEqual(rc, 0, err)
        self.kernel.write_bytes(OPTIMIZED.encode("utf-8"))
        rc, err = self.hook("PostToolUse", {"tool_name": "Edit", "tool_input": {"file_path": str(self.kernel)}})
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.kernel.read_bytes(), OPTIMIZED.encode("utf-8"))
        rc, err = self.hook("Stop", {"stop_hook_active": False})
        self.assertEqual(rc, 2)
        self.assertIn("UNSEALED_FINAL_KERNEL", err)
        reg = register_seal(self.sd, self.make_seal(), kernel_us=10.0)
        self.assertTrue(reg["registered"] and reg["covers_protected"], reg)
        self.assertEqual(read_session(self.sd)["stop_blocks"], 0)
        rc, err = self.hook("Stop", {"stop_hook_active": True})
        self.assertEqual(rc, 0, err)
        comp = verify_session_completion(session_dir=str(self.sd))
        self.assertTrue(comp["ok"])
        self.assertEqual(comp["incumbent"]["kernel_us"], 10.0)
        # Reverting to the first sealed incumbent is also accepted.
        self.kernel.write_bytes(KERNEL.encode("utf-8"))
        rc, err = self.hook("Stop", {})
        self.assertEqual(rc, 0, err)

    def test_stop_loop_safety(self) -> None:
        codes = [self.hook("Stop", {"stop_hook_active": i > 0})[0] for i in range(5)]
        self.assertEqual(codes[:3], [2, 2, 2])
        self.assertEqual(codes[3], 0)
        # Escape is per stop-sequence: a subsequent turn re-arms the Stop gate (never permanently disabled).
        rc_next_turn, _ = self.hook("Stop", {"stop_hook_active": False})
        self.assertEqual(rc_next_turn, 2)

    def test_mcp_tool_command_and_out_path_hardening(self) -> None:
        rc, _ = self.hook("PreToolUse", {
            "tool_name": "mcp__weaveprofiler__weaveprof_megakernel_audit",
            "tool_input": {"bench_cmd": "python run.py install-agent --reset"},
        })
        self.assertEqual(rc, 2)
        rc, _ = self.hook("PreToolUse", {
            "tool_name": "mcp__weaveprofiler__weaveprof_megakernel_audit",
            "tool_input": {"out_dir": str(self.ws / ".claude")},
        })
        self.assertEqual(rc, 2)
        rc, err = self.hook("PreToolUse", {
            "tool_name": "mcp__weaveprofiler__weaveprof_megakernel_audit",
            "tool_input": {"bench_cmd": "python bench.py", "out_dir": str(self.tmp / "ok_out")},
        })
        self.assertEqual(rc, 0, err)
        # Server itself also rejects malicious out_dir / bench_cmd even if invoked directly without hooks.
        from weaveprof.mcp_server import WeaveProfServer
        srv = WeaveProfServer(self.ws, self.sd, None, None)
        with self.assertRaises(ValueError):
            srv._validate_out_path(str(self.ws / ".claude" / "settings.json"), "out_dir")
        with self.assertRaises(ValueError):
            srv._validate_cmd("python run.py install-agent --reset", "bench_cmd")
        srv._validate_out_path(str(self.tmp / "clean_run"), "out_dir")
        srv._validate_cmd(["python", "bench.py"], "bench_cmd")
        with self.assertRaises(ValueError):
            srv._validate_out_path(str(self.ws / ".qwen" / "settings.json"), "out_dir")
        with self.assertRaises(ValueError):
            srv._validate_out_path(str(self.ws / ".agents" / "plugins" / "weaveprofiler" / "hooks.json"), "out_dir")


class QwenAndAgyProtocolTests(_WorkspaceCase):
    def _run_proto(self, event: str, payload: dict, *, protocol: str = "auto", extra_env: dict | None = None) -> tuple[int, dict, str]:
        env = dict(self.env)
        if extra_env:
            env.update(extra_env)
        cmd = [sys.executable, "-B", str(GUARD), "--event", event, "--workspace", str(self.ws),
               "--session-dir", str(self.sd)]
        if protocol != "auto":
            cmd.extend(["--protocol", protocol])
        proc = subprocess.run(
            cmd, input=json.dumps(payload).encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=120,
        )
        out_text = proc.stdout.decode("utf-8", "replace").strip()
        err_text = proc.stderr.decode("utf-8", "replace")
        return proc.returncode, (json.loads(out_text) if out_text else {}), err_text

    def test_qwen_protocol_pre_post_stop_and_subagent_stop(self) -> None:
        # PreToolUse blocks edit on protected file before baseline seal with native JSON decision.
        rc, out, _ = self._run_proto("PreToolUse", {
            "hook_event_name": "PreToolUse", "cwd": str(self.ws),
            "tool_name": "edit",
            "tool_input": {"file_path": str(self.kernel), "old_string": "a[i] + b[i]", "new_string": "a[i] * b[i]"},
        }, protocol="qwen")
        self.assertEqual(rc, 0)
        self.assertEqual(out["decision"], "block")
        self.assertEqual(out["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertIn("SEAL_BEFORE_OPTIMIZE", out["reason"])

        # Auto-detection via QWEN_PROJECT_DIR also speaks Qwen JSON and blocks tampering with .qwen/settings.json.
        rc, out, _ = self._run_proto("PreToolUse", {
            "hook_event_name": "PreToolUse", "cwd": str(self.ws),
            "tool_name": "run_shell_command",
            "tool_input": {"command": "rm -f .qwen/settings.json"},
        }, protocol="auto", extra_env={"QWEN_PROJECT_DIR": str(self.ws)})
        self.assertEqual(rc, 0)
        self.assertEqual(out["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertIn("SUPERVISOR_CONFIG_TAMPER", out["reason"])

        # Read-only Qwen tool allowed cleanly.
        rc, out, _ = self._run_proto("PreToolUse", {
            "hook_event_name": "PreToolUse", "cwd": str(self.ws),
            "tool_name": "read_file", "tool_input": {"file_path": str(self.kernel)},
        }, protocol="qwen")
        self.assertEqual(rc, 0)
        self.assertEqual(out["decision"], "allow")

        # PostToolUse reverts unauthorized mutation and returns block + additionalContext.
        self.kernel.write_bytes(b"// sneaky edit\n")
        rc, out, _ = self._run_proto("PostToolUse", {
            "hook_event_name": "PostToolUse", "cwd": str(self.ws),
            "tool_name": "exec", "tool_input": {"command": "python evil.py"},
        }, protocol="qwen")
        self.assertEqual(rc, 0)
        self.assertEqual(out["decision"], "block")
        self.assertIn("restored", out["hookSpecificOutput"]["additionalContext"])
        self.assertEqual(self.kernel.read_bytes(), KERNEL.encode("utf-8"))

        # SubagentStop checks probe pollution / unauthorized edits / config integrity WITHOUT requiring final seal.
        rc, out, _ = self._run_proto("SubagentStop", {"stop_hook_active": False, "cwd": str(self.ws)}, protocol="qwen")
        self.assertEqual(rc, 0)
        self.assertEqual(out["decision"], "allow")
        self.assertEqual(read_session(self.sd).get("stop_blocks", 0), 0)
        # Stop still enforces NO_VERIFIED_PASS_SEAL before seal, respecting 3-block cap.
        decisions = [
            self._run_proto("Stop", {"stop_hook_active": i > 0, "cwd": str(self.ws)}, protocol="qwen")[1]["decision"]
            for i in range(4)
        ]
        self.assertEqual(decisions, ["block", "block", "block", "allow"])

        # After registering a valid PASS seal, edit & Stop unlock, while probe pollution remains blocked.
        self.assertTrue(register_seal(self.sd, self.make_seal(), kernel_us=11.2)["registered"])
        rc, out, _ = self._run_proto("PreToolUse", {
            "tool_name": "edit",
            "tool_input": {"file_path": str(self.kernel), "old_string": "+", "new_string": "*"},
            "cwd": str(self.ws),
        }, protocol="qwen")
        self.assertEqual(rc, 0)
        self.assertEqual(out["decision"], "allow")
        rc, out, _ = self._run_proto("PreToolUse", {
            "tool_name": "edit",
            "tool_input": {"file_path": str(self.kernel), "old_string": "+", "new_string": "WP_OPEN(0); +"},
            "cwd": str(self.ws),
        }, protocol="qwen")
        self.assertEqual(rc, 0)
        self.assertEqual(out["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertIn("PROBE_POLLUTION", out["reason"])
        rc, out, _ = self._run_proto("Stop", {"stop_hook_active": False, "cwd": str(self.ws)}, protocol="qwen")
        self.assertEqual(rc, 0)
        self.assertEqual(out["decision"], "allow")

    def test_agy_protocol_pre_post_post_invocation_and_stop(self) -> None:
        # Auto-detect Antigravity camelCase toolCall payload on edit_file before seal.
        rc, out, _ = self._run_proto("PreToolUse", {
            "hookEventName": "PreToolUse",
            "cwd": str(self.ws),
            "workspacePaths": [str(self.ws)],
            "toolCall": {
                "id": "call_1",
                "name": "edit_file",
                "args": {"Path": str(self.kernel), "Target": "a[i] + b[i]", "Replacement": "a[i] * b[i]"},
            },
        }, protocol="auto")
        self.assertEqual(rc, 0)
        self.assertEqual(out["decision"], "deny")
        self.assertIn("SEAL_BEFORE_OPTIMIZE", out["reason"])

        # Antigravity run_command with PascalCase CommandLine editing kernel or plugin config is denied.
        rc, out, _ = self._run_proto("PreToolUse", {
            "hookEventName": "PreToolUse", "cwd": str(self.ws),
            "toolCall": {"id": "call_2", "name": "run_command", "args": {"CommandLine": "sed -i 's/+/*/' kernel.cu"}},
        }, protocol="agy")
        self.assertEqual(rc, 0)
        self.assertEqual(out["decision"], "deny")
        rc, out, _ = self._run_proto("PreToolUse", {
            "hookEventName": "PreToolUse", "cwd": str(self.ws),
            "toolCall": {"id": "call_3", "name": "write_file", "args": {
                "Path": str(self.ws / ".agents/plugins/weaveprofiler/hooks.json"), "Content": "{}"}},
        }, protocol="agy")
        self.assertEqual(rc, 0)
        self.assertEqual(out["decision"], "deny")
        self.assertIn("SUPERVISOR_CONFIG_TAMPER", out["reason"])

        # Antigravity read_file allowed cleanly.
        rc, out, _ = self._run_proto("PreToolUse", {
            "hookEventName": "PreToolUse", "cwd": str(self.ws),
            "toolCall": {"id": "call_4", "name": "read_file", "args": {"Path": str(self.kernel)}},
        }, protocol="agy")
        self.assertEqual(rc, 0)
        self.assertEqual(out, {"decision": "allow"})

        # Antigravity PostToolUse reverts unauthorized mutation, returns `{}`, and PostInvocation delivers the alert.
        self.kernel.write_bytes(b"// sneaky agy run_command\n")
        rc, out, _ = self._run_proto("PostToolUse", {
            "hookEventName": "PostToolUse", "cwd": str(self.ws),
            "toolCall": {"id": "call_5", "name": "run_command", "args": {"CommandLine": "python evil.py"}},
        }, protocol="agy")
        self.assertEqual(rc, 0)
        self.assertEqual(out, {})
        self.assertEqual(self.kernel.read_bytes(), KERNEL.encode("utf-8"))
        rc, out, _ = self._run_proto("PostInvocation", {
            "hookEventName": "PostInvocation", "cwd": str(self.ws), "invocationNum": 1,
        }, protocol="agy")
        self.assertEqual(rc, 0)
        self.assertEqual(out["terminationBehavior"], "force_continue")
        self.assertIn("restored", out["injectSteps"][0]["systemMessage"]["systemMessage"])
        # Delivered once and cleared.
        rc, out, _ = self._run_proto("PostInvocation", {
            "hookEventName": "PostInvocation", "cwd": str(self.ws), "invocationNum": 2,
        }, protocol="agy")
        self.assertEqual(rc, 0)
        self.assertEqual(out, {})

        # Antigravity Stop returns decision=continue before seal, respects USER_CANCELED, and caps at 3 blocks.
        rc, out, _ = self._run_proto("Stop", {
            "hookEventName": "Stop", "cwd": str(self.ws),
            "terminationReason": "USER_CANCELED", "continuationActive": False,
        }, protocol="agy")
        self.assertEqual(rc, 0)
        self.assertEqual(out, {"decision": "allow"})
        decisions = [
            self._run_proto("Stop", {
                "hookEventName": "Stop", "cwd": str(self.ws),
                "terminationReason": "NO_TOOL_CALL", "continuationActive": i > 0,
            }, protocol="agy")[1]["decision"]
            for i in range(4)
        ]
        self.assertEqual(decisions, ["continue", "continue", "continue", "allow"])

        # After PASS seal, edit_file & Stop allow cleanly, while probe macros remain denied.
        self.assertTrue(register_seal(self.sd, self.make_seal(), kernel_us=9.8)["registered"])
        rc, out, _ = self._run_proto("PreToolUse", {
            "hookEventName": "PreToolUse", "cwd": str(self.ws),
            "toolCall": {"id": "call_6", "name": "edit_file", "args": {
                "Path": str(self.kernel), "Target": "a[i] + b[i]", "Replacement": "__fadd_rn(a[i], b[i])"}},
        }, protocol="agy")
        self.assertEqual(rc, 0)
        self.assertEqual(out, {"decision": "allow"})
        rc, out, _ = self._run_proto("PreToolUse", {
            "hookEventName": "PreToolUse", "cwd": str(self.ws),
            "toolCall": {"id": "call_7", "name": "edit_file", "args": {
                "Path": str(self.kernel), "Target": "a[i] + b[i]", "Replacement": "WP_B(0, 0); a[i] + b[i]"}},
        }, protocol="agy")
        self.assertEqual(rc, 0)
        self.assertEqual(out["decision"], "deny")
        self.assertIn("PROBE_POLLUTION", out["reason"])
        rc, out, _ = self._run_proto("Stop", {
            "hookEventName": "Stop", "cwd": str(self.ws),
            "terminationReason": "NO_TOOL_CALL", "continuationActive": False,
        }, protocol="agy")
        self.assertEqual(rc, 0)
        self.assertEqual(out, {"decision": "allow"})

    def test_security_audit_p0_p1_regressions(self) -> None:
        # P0-1: MCP megakernel_audit rejects precaptured/caller-forged fields (`dump_path`, `runtime_metrics`, `names`).
        from weaveprof.mcp_server import WeaveProfServer
        srv = WeaveProfServer(self.ws, self.sd, None, None)
        for bad_key in ("dump_path", "runtime_metrics", "names", "nsys_cmd"):
            with self.assertRaises(ValueError):
                srv._tool_megakernel({"entry_source": "kernel.cu", bad_key: "forged"})
        # P0-1: register_seal rejects seals produced in `precaptured` mode!
        prec_seal = self.make_seal()
        with supervisor_env(self.sd):
            prec_path, _ = EpisodeSealer.seal_round(
                round_dir=prec_seal.parent, round_number=99, source_path=self.kernel,
                source_sha256=_sha(self.kernel), probe_path=None, instrumented_path=None, names_path=None,
                feedback_path=prec_seal.parent / "feedback.json", verify_path=None,
                trace_path=prec_seal.parent / "out.perfetto-trace", failed_trace_path=None,
                result={"status": "PASS", "checks": {"assembly": True, "output": True, "events": True, "overhead": True}},
                policy=HarnessPolicy(), limit=0.10, region_lock={"ok": True}, mode="precaptured",
                out_seal_path=prec_seal.parent / "precaptured.seal.json",
            )
        reg_prec = register_seal(self.sd, prec_path)
        self.assertFalse(reg_prec["registered"])

        # P0-4: verify_seal fails closed when require_hmac=True and key is absent!
        from weaveprof.harness import verify_seal
        good_seal = self.make_seal()
        res_no_key = verify_seal(good_seal, require_hmac=True)
        self.assertFalse(res_no_key["ok"])
        self.assertTrue(any("key" in e.lower() or "hmac" in e.lower() for e in res_no_key["errors"]), res_no_key)

        # P0-5: Antigravity Stop honors executionNum > 0 AND allowlist (MAX_INVOCATIONS allowed).
        rc, out, _ = self._run_proto("Stop", {
            "hookEventName": "Stop", "cwd": str(self.ws),
            "terminationReason": "MAX_INVOCATIONS", "executionNum": 0,
        }, protocol="agy")
        self.assertEqual(rc, 0)
        self.assertEqual(out, {"decision": "allow"})
        decs = [
            self._run_proto("Stop", {
                "hookEventName": "Stop", "cwd": str(self.ws),
                "terminationReason": "NO_TOOL_CALL", "executionNum": i,
            }, protocol="agy")[1]["decision"]
            for i in range(4)
        ]
        self.assertEqual(decs, ["continue", "continue", "continue", "allow"])

        # P0-6: Directory-level deletion of `.claude` blocked in PreToolUse AND tampered config restored in PostToolUse!
        rc, err = self.hook("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "rm -rf .claude"}})
        self.assertEqual(rc, 2)
        self.assertIn("SUPERVISOR_CONFIG_TAMPER", err)
        orig_cfg = (self.ws / ".claude" / "settings.json").read_bytes()
        (self.ws / ".claude" / "settings.json").write_bytes(b"{}")
        rc, err = self.hook("PostToolUse", {"tool_name": "Bash", "tool_input": {"command": "python oops.py"}})
        self.assertEqual(rc, 2)
        self.assertEqual((self.ws / ".claude" / "settings.json").read_bytes(), orig_cfg)

        # P1-8 & P1-12: PostToolUseFailure reverts edits; real Antigravity replace_file_content & manage_task blocked!
        rc, out, _ = self._run_proto("PreToolUse", {
            "hookEventName": "PreToolUse", "cwd": str(self.ws),
            "toolCall": {"id": "c_w", "name": "replace_file_content",
                         "args": {"TargetFile": str(self.kernel), "TargetContent": "+", "ReplacementContent": "*"}},
        }, protocol="agy")
        self.assertEqual(out["decision"], "deny")
        rc, out, _ = self._run_proto("PreToolUse", {
            "hookEventName": "PreToolUse", "cwd": str(self.ws),
            "toolCall": {"id": "c_m", "name": "manage_task",
                         "args": {"Action": "send_input", "Input": "cat ~/.cache/weaveprof/sessions/*/.supervisor.key"}},
        }, protocol="agy")
        self.assertEqual(out["decision"], "deny")
        self.kernel.write_bytes(b"// mutated before failure\n")
        rc, err = self.hook("PostToolUseFailure", {"tool_name": "Bash", "tool_input": {"command": "python fail.py"}})
        self.assertEqual(rc, 2)
        self.assertEqual(self.kernel.read_bytes(), KERNEL.encode("utf-8"))

        # P0-7: Tampering session.json (`require_final_seal: False`) fails closed via policy_hmac!
        sess_path = self.sd / "session.json"
        tampered = json.loads(sess_path.read_text(encoding="utf-8"))
        tampered["require_final_seal"] = False
        sess_path.write_text(json.dumps(tampered), encoding="utf-8")
        rc, err = self.hook("PreToolUse", {"tool_name": "Edit", "tool_input": {
            "file_path": str(self.kernel), "old_string": "+", "new_string": "*"}})
        self.assertEqual(rc, 2)
        self.assertIn("SESSION_POLICY_TAMPER", err)

    def test_png_only_enforcement_and_global_qwen_guard(self) -> None:
        # 1. PreToolUse blocks generating .perfetto.json.gz / .trace.json.gz / --write-gz / write_perfetto_gz
        rc, out, _ = self._run_proto("PreToolUse", {
            "cwd": str(self.ws), "tool_name": "run_shell_command",
            "tool_input": {"command": "python wp_decode.py dump.bin --names names.json -o trace.perfetto.json.gz"},
        }, protocol="qwen")
        self.assertEqual(rc, 0)
        self.assertEqual(out["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertIn("FORBIDDEN_GZ_ARTIFACT", out["reason"])

        rc, out, _ = self._run_proto("PreToolUse", {
            "cwd": str(self.ws), "tool_name": "mcp__weaveprofiler__weaveprof_megakernel_audit",
            "tool_input": {"bench_cmd": "python bench.py", "write_perfetto_gz": True},
        }, protocol="qwen")
        self.assertEqual(out["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertIn("FORBIDDEN_GZ_ARTIFACT", out["reason"])

        # 2. PostToolUse deletes stray .perfetto.json.gz created out-of-band and blocks!
        stray_gz = self.ws / "stray.perfetto.json.gz"
        stray_gz.write_bytes(b"\x1f\x8b\x08\x00fakegz")
        rc, out, _ = self._run_proto("PostToolUse", {
            "cwd": str(self.ws), "tool_name": "run_shell_command",
            "tool_input": {"command": "python run_script.py"},
        }, protocol="qwen")
        self.assertEqual(out["decision"], "block")
        self.assertIn("FORBIDDEN_GZ_ARTIFACT", out["reason"])
        self.assertFalse(stray_gz.exists(), "PostToolUse must delete stray .gz file")

        # 3. Workspace Stop gate rejects PASS seal missing a valid overview PNG!
        seal_no_png = self.make_seal(include_png=False)
        self.assertTrue(register_seal(self.sd, seal_no_png, kernel_us=10.0)["registered"])
        rc, out, _ = self._run_proto("Stop", {"cwd": str(self.ws), "stop_hook_active": False}, protocol="qwen")
        self.assertEqual(out["decision"], "block")
        self.assertIn("MISSING_OVERVIEW_PNG", out["reason"])

        # Once a valid PNG seal is registered, Stop succeeds!
        seal_with_png = self.make_seal(include_png=True)
        self.assertTrue(register_seal(self.sd, seal_with_png, kernel_us=9.5)["registered"])
        rc, out, _ = self._run_proto("Stop", {"cwd": str(self.ws), "stop_hook_active": False}, protocol="qwen")
        self.assertEqual(out["decision"], "allow")

        # 4. Global guard mode (--global-guard) on an UNPROVISIONED workspace:
        unprov = self.tmp / "unprovisioned_exp"
        unprov.mkdir()
        def run_global(event: str, payload: dict) -> dict:
            proc = subprocess.run(
                [sys.executable, "-B", str(GUARD), "--global-guard", "--protocol", "qwen",
                 "--event", event, "--sessions-root", str(self.sessions)],
                input=json.dumps(payload).encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=self.env, timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
            return json.loads(proc.stdout.decode("utf-8", "replace"))

        # Unrelated session passes Stop immediately without interference
        out_idle = run_global("Stop", {"session_id": "unrelated-chat", "cwd": str(unprov)})
        self.assertEqual(out_idle["decision"], "allow")

        # WeaveProfiler profiling session: PreToolUse activates & blocks .gz; Stop blocks without PNG!
        sid = "qwen-wp-session-1"
        out_gz = run_global("PreToolUse", {
            "session_id": sid, "cwd": str(unprov), "tool_name": "run_shell_command",
            "tool_input": {"command": f"python wp_decode.py {unprov}/capture.fi02.bin --names n.json -o {unprov}/out.trace.json.gz"},
        })
        self.assertEqual(out_gz["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertIn("FORBIDDEN_GZ_ARTIFACT", out_gz["reason"])

        # Legit profiling command activates session tracking
        out_ok = run_global("PreToolUse", {
            "session_id": sid, "cwd": str(unprov), "tool_name": "run_shell_command",
            "tool_input": {"command": f"python wp_decode.py {unprov}/capture.fi02.bin --names n.json -o {unprov}/overview.png"},
        })
        self.assertEqual(out_ok["decision"], "allow")

        # Trying to Stop before generating overview.png MUST be blocked!
        out_stop1 = run_global("Stop", {"session_id": sid, "cwd": str(unprov), "stop_hook_active": False})
        self.assertEqual(out_stop1["decision"], "block")
        self.assertIn("MISSING_OVERVIEW_PNG", out_stop1["reason"])

        # Stray .gz created by a script is deleted & blocked on PostToolUse
        bad_g = unprov / "accidental.trace.json.gz"
        bad_g.write_bytes(b"\x1f\x8b\x08\x00bad")
        out_post = run_global("PostToolUse", {
            "session_id": sid, "cwd": str(unprov), "tool_name": "run_shell_command",
            "tool_input": {"command": f"cd {unprov} && python run.py"},
        })
        self.assertEqual(out_post["decision"], "block")
        self.assertIn("FORBIDDEN_GZ_ARTIFACT", out_post["reason"])
        self.assertFalse(bad_g.exists())

        # Read-only `--help` invocation must NOT falsely arm the profiling Stop gate
        out_help = run_global("PreToolUse", {
            "session_id": "help-only-chat", "cwd": str(unprov), "tool_name": "run_shell_command",
            "tool_input": {"command": "python wp_decode.py --help"},
        })
        self.assertEqual(out_help["decision"], "allow")
        out_help_stop = run_global("Stop", {"session_id": "help-only-chat", "cwd": str(unprov)})
        self.assertEqual(out_help_stop["decision"], "allow")

        # Tiny 1x1 dummy PNG must NOT fool the Stop gate!
        (unprov / "overview.png").write_bytes(self._FAKE_1X1_PNG)
        out_stop_fake = run_global("Stop", {"session_id": sid, "cwd": str(unprov), "stop_hook_active": True})
        self.assertEqual(out_stop_fake["decision"], "block")
        self.assertIn("MISSING_OVERVIEW_PNG", out_stop_fake["reason"])

        # Once a real chart-sized overview.png exists and no .gz remains, Stop unlocks cleanly!
        (unprov / "overview.png").write_bytes(self._TINY_PNG)
        out_stop2 = run_global("Stop", {"session_id": sid, "cwd": str(unprov), "stop_hook_active": True})
        self.assertEqual(out_stop2["decision"], "allow")

        # 5. install_global_hooks writes clean ~/.qwen/settings.json without touching existing settings
        from weaveprof.agent_installer import install_global_hooks
        fake_home = self.tmp / "fake_home"
        (fake_home / ".qwen").mkdir(parents=True)
        (fake_home / ".qwen" / "settings.json").write_text(json.dumps({"model": {"name": "qwen4-max"}}), encoding="utf-8")
        gres = install_global_hooks(agent="qwen", home_dir=fake_home, sessions_root=self.sessions)
        self.assertTrue(gres["ok"])
        qcfg = json.loads((fake_home / ".qwen" / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(qcfg["model"]["name"], "qwen4-max")
        self.assertIs(qcfg["disableAllHooks"], False)
        for ev in ("PreToolUse", "PostToolUse", "PostToolUseFailure", "Stop", "SubagentStop"):
            cmds = [h["command"] for g in qcfg["hooks"][ev] for h in g["hooks"]]
            self.assertTrue(any("--global-guard" in c and ev in c for c in cmds), (ev, cmds))


if __name__ == "__main__":
    unittest.main()
