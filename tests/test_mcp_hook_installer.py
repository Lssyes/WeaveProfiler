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
import subprocess
import tempfile
import unittest

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

    # -- helpers -----------------------------------------------------------
    def make_seal(self, *, status: str = "PASS", evil_key: bool = False) -> Path:
        self._seal_n += 1
        run = self.tmp / f"run{self._seal_n}"
        run.mkdir()
        feedback = run / "feedback.json"
        feedback.write_text(json.dumps({"status": status}), encoding="utf-8")
        trace = run / "out.perfetto-trace"
        trace.write_bytes(b"\x0a\x00trace")
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
        self.assertTrue(local["enableAllProjectMcpServers"])
        codex_hooks = json.loads((ws / ".codex/hooks.json").read_text(encoding="utf-8"))
        for doc in (settings, codex_hooks):
            for ev in ("PreToolUse", "PostToolUse", "Stop"):
                cmds = [h["command"] for g in doc["hooks"][ev] for h in g["hooks"]]
                self.assertEqual(sum("guard_hook.py" in c for c in cmds), 1, ev)
                self.assertTrue(any(f"--event {ev}" in c for c in cmds))
        for rel in ("CLAUDE.md", "AGENTS.md"):
            text = (ws / rel).read_text(encoding="utf-8")
            self.assertEqual(text.count("<!-- WEAVEPROFILER:BEGIN -->"), 1)
            self.assertIn("weaveprof_megakernel_audit", text)
            self.assertIn("`kernel.cu`", text)
        for rel in (".claude/skills/weaveprofiler/SKILL.md", ".agents/skills/weaveprofiler/SKILL.md"):
            text = (ws / rel).read_text(encoding="utf-8")
            self.assertTrue(text.startswith("---\n"))
            self.assertIn("Installed WeaveProfiler (`wp`) skill copy", text)
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
        self._check(replies, "2024-11-05", seal)


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


if __name__ == "__main__":
    unittest.main()
