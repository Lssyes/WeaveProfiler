#!/usr/bin/env python3
"""WeaveProfiler (`wp`) turnkey agent provisioning + outer supervisor.

    python -B <skill_root>/weaveprof/agent_installer.py install --workspace <ws> [--protect kernel.cu ...]
    python -B <skill_root>/weaveprof/agent_installer.py guard-exec --workspace <ws> -- codex exec "..."
    python -B <skill_root>/weaveprof/agent_installer.py status --workspace <ws>

`install` creates the out-of-workspace session (supervisor key, byte-exact backups,
session.json) and wires Claude Code (.mcp.json, .claude/settings*.json, skill, CLAUDE.md)
and OpenAI Codex CLI (.codex/config.toml, .codex/hooks.json, skill, AGENTS.md).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shlex
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Union

_HERE = Path(__file__).resolve().parent
SKILL_ROOT = _HERE.parent
try:
    if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == _HERE:
        sys.path.pop(0)
except OSError:
    pass
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

from weaveprof.guard_hook import (  # noqa: E402
    BACKUP_DIRNAME,
    KEY_FILE,
    SESSION_FILE,
    SESSION_SCHEMA,
    mutated_since_baseline,
    read_session,
    restore_protected,
    session_dir_for,
    sha256_file,
    valid_pass_seals,
    verify_session_completion,
    write_json_atomic,
)

GUARD_HOOK = _HERE / "guard_hook.py"
MCP_SERVER = _HERE / "mcp_server.py"
BEGIN_MARK = "<!-- WEAVEPROFILER:BEGIN -->"
END_MARK = "<!-- WEAVEPROFILER:END -->"
KERNEL_SUFFIXES = (".cu", ".cuh", ".hip")
PY_KERNEL_MARKERS = ("@triton.jit", "@triton.autotune", "@cute.kernel", "@cute.jit", "cutlass.cute")
SKIP_DIRS = {".git", ".hg", ".svn", ".claude", ".codex", ".agents", "node_modules", "build", "dist", "__pycache__",
             ".venv", "venv", "env", ".weaveprof", ".weaveprof_cache", ".mypy_cache", ".pytest_cache", "third_party"}
MAX_AUTODISCOVER = 64
HOOK_EVENTS = ("PreToolUse", "PostToolUse", "Stop")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
    except ValueError as exc:
        raise ValueError(f"Refusing to overwrite unparsable JSON {path}: {exc}") from exc
    return doc if isinstance(doc, dict) else {}


def _dump_json(path: Path, doc: Any) -> None:
    _write_text(path, json.dumps(doc, indent=2, ensure_ascii=False) + "\n")


def _posix(p: Union[str, Path]) -> str:
    return str(p).replace("\\", "/") if os.name == "nt" else str(p)


def _shell_quote(s: str) -> str:
    if os.name != "nt":
        return shlex.quote(s)
    return f'"{s}"' if re.search(r"[\s&()^|<>\"']", s) else s


def _toml_str(s: str) -> str:
    out = ['"']
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif o < 0x20 or o == 0x7F:
            out.append(f"\\u{o:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _is_kernel_py(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return any(m in text for m in PY_KERNEL_MARKERS)


def discover_kernel_files(workspace: Path) -> List[Path]:
    found: List[Path] = []
    for root, dirs, files in os.walk(workspace):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            p = Path(root) / name
            suf = p.suffix.lower()
            if suf in KERNEL_SUFFIXES or (suf == ".py" and _is_kernel_py(p)):
                found.append(p.resolve())
                if len(found) > MAX_AUTODISCOVER:
                    raise ValueError(f"Auto-discovery found > {MAX_AUTODISCOVER} kernel files under {workspace}; "
                                     "pass --protect explicitly.")
    return found


def _resolve_protect(workspace: Path, protect_files: Optional[Sequence[Union[str, Path]]]) -> List[Path]:
    if not protect_files:
        files = discover_kernel_files(workspace)
        if not files:
            raise ValueError(f"No kernel files (.cu/.cuh/.hip or Triton/CuTeDSL .py) found under {workspace}; "
                             "pass --protect <file> explicitly.")
        return files
    out: List[Path] = []
    for f in protect_files:
        p = Path(os.path.expanduser(str(f)))
        if not p.is_absolute():
            p = workspace / p
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Protected file does not exist: {p}")
        try:
            p.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(f"Protected file {p} is outside workspace {workspace}") from exc
        if p not in out:
            out.append(p)
    return out


def _bench_list(bench_cmd: Union[None, str, Sequence[str]]) -> Optional[List[str]]:
    if bench_cmd is None or bench_cmd == "" or bench_cmd == []:
        return None
    if isinstance(bench_cmd, str):
        return shlex.split(bench_cmd, posix=os.name != "nt")
    return [str(x) for x in bench_cmd]


# ---------------------------------------------------------------------------
# Session provisioning
# ---------------------------------------------------------------------------

def _provision_session(workspace: Path, files: List[Path], *, sd: Path, bench: Optional[List[str]], card: str,
                       arch: Optional[str], max_overhead: float, require_final_seal: bool, reset: bool) -> Dict[str, Any]:
    sd.mkdir(parents=True, exist_ok=True)
    _chmod(sd.parent, 0o700)
    _chmod(sd, 0o700)
    key = sd / KEY_FILE
    if reset or not key.is_file() or not key.read_bytes().strip():
        tmp = sd / f".{KEY_FILE}.tmp"
        fd = os.open(str(tmp), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(secrets.token_hex(32).encode("ascii"))
        os.replace(tmp, key)
    _chmod(key, 0o600)
    existing = None if reset else read_session(sd)
    prev_by_rel = {e.get("rel"): e for e in (existing or {}).get("protected") or []}
    backups = sd / BACKUP_DIRNAME
    backups.mkdir(parents=True, exist_ok=True)
    _chmod(backups, 0o700)
    protected = []
    for p in files:
        rel = p.relative_to(workspace).as_posix()
        prev = prev_by_rel.get(rel)
        bpath = backups / rel
        if prev and Path(prev.get("backup", "")).is_file() and sha256_file(prev["backup"]) == prev.get("baseline_sha256"):
            protected.append(prev)
            continue
        bpath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, bpath)
        _chmod(bpath, 0o600)
        protected.append({"path": str(p), "rel": rel, "baseline_sha256": sha256_file(p), "backup": str(bpath),
                          "bytes": p.stat().st_size})
    now = round(time.time(), 3)
    session = dict(existing or {})
    session.update({
        "schema": SESSION_SCHEMA,
        "workspace": str(workspace),
        "session_id": sd.name,
        "created_at": session.get("created_at", now),
        "updated_at": now,
        "protected": protected,
        "require_final_seal": bool(require_final_seal),
        "max_overhead": float(max_overhead),
        "bench_cmd": bench if bench is not None else session.get("bench_cmd"),
        "card": card,
        "arch": arch,
        "seals": session.get("seals") or [],
        "stop_blocks": int(session.get("stop_blocks") or 0),
        "skill_root": str(SKILL_ROOT),
    })
    write_json_atomic(sd / SESSION_FILE, session)
    return session


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------

def _hook_command(py: str, event: str, workspace: Path, sd: Path) -> str:
    parts = [_posix(py), "-B", _posix(GUARD_HOOK), "--event", event, "--workspace", _posix(workspace),
             "--session-dir", _posix(sd)]
    return " ".join(_shell_quote(x) for x in parts)


def _merge_hooks(doc: Dict[str, Any], py: str, workspace: Path, sd: Path, matcher: str) -> Dict[str, Any]:
    hooks = doc.get("hooks") if isinstance(doc.get("hooks"), dict) else {}
    for event in HOOK_EVENTS:
        groups = hooks.get(event) if isinstance(hooks.get(event), list) else []
        cleaned = []
        for g in groups:
            if not isinstance(g, dict):
                continue
            inner = [h for h in (g.get("hooks") or [])
                     if not (isinstance(h, dict) and "guard_hook.py" in str(h.get("command", "")))]
            if inner:
                g = dict(g)
                g["hooks"] = inner
                cleaned.append(g)
        entry: Dict[str, Any] = {"hooks": [{"type": "command", "command": _hook_command(py, event, workspace, sd),
                                            "timeout": 120}]}
        if event != "Stop":
            entry = {"matcher": matcher, **entry}
        cleaned.append(entry)
        hooks[event] = cleaned
    doc["hooks"] = hooks
    return doc


def _mcp_args(workspace: Path, sd: Path) -> List[str]:
    return ["-B", _posix(MCP_SERVER), "--workspace", _posix(workspace), "--session-dir", _posix(sd)]


def _skill_text(workspace: Path) -> str:
    src = SKILL_ROOT / "SKILL.md"
    text = src.read_text(encoding="utf-8") if src.is_file() else (
        "---\nname: weaveprofiler\ndescription: WeaveProfiler (wp) supervised GPU kernel profiling.\n---\n")
    text = text.replace("\r\n", "\n")
    root = _posix(SKILL_ROOT)

    def _link(m: "re.Match[str]") -> str:
        target = m.group(2)
        if re.match(r"^(?:[a-z][a-z0-9+.-]*:|#|/|[A-Za-z]:)", target):
            return m.group(0)
        return f"{m.group(1)}({root}/{target})"

    text = re.sub(r"(\]\s*)\(([^)\s]+)\)", _link, text)
    note = (f"\n> **Installed WeaveProfiler (`wp`) skill copy.** Package root: `{root}`. Relative paths in this "
            "document resolve against that root. In this workspace use the `weaveprofiler` MCP tools "
            "(`weaveprof_status`, `weaveprof_megakernel_audit`, `weaveprof_profile`, `weaveprof_plan_context`, "
            "`weaveprof_analyze`, `weaveprof_compare_rounds`, `weaveprof_verify_seal`); hooks enforce "
            "seal-before-optimize and a sealed final kernel.\n")
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            cut = text.find("\n", end + 4)
            cut = len(text) if cut == -1 else cut + 1
            return text[:cut] + note + "\n" + text[cut:]
    return note + "\n" + text


def _workflow_block(session: Dict[str, Any]) -> str:
    rels = ", ".join(f"`{e['rel']}`" for e in session.get("protected") or [])
    bench = session.get("bench_cmd")
    bench_line = f"- Default bench: `{' '.join(bench)}`\n" if bench else ""
    return (
        f"{BEGIN_MARK}\n"
        "## WeaveProfiler (`wp`) supervised kernel optimization\n\n"
        f"- Protected kernel files (hook-enforced): {rels}\n"
        f"- Overhead ceiling: {float(session.get('max_overhead', 0.10)):.3f}; card: `{session.get('card')}`\n"
        f"{bench_line}\n"
        "1. Call MCP `weaveprof_status` to see protected files, seal state and gate status.\n"
        "2. Before ANY edit of a protected file, seal the unmodified baseline with `weaveprof_megakernel_audit` "
        "(fast shadow-copy megakernel audit) or `weaveprof_profile`. Edits stay blocked until an HMAC-verified "
        "PASS seal exists; out-of-band mutations (shell, scripts) are reverted to the exact baseline bytes.\n"
        "3. Choose optimizations from evidence: `weaveprof_plan_context`, `weaveprof_analyze`, "
        "`weaveprof_compare_rounds` (follow `next_round_directive`, revert regressions).\n"
        "4. Keep production kernel source probe-free: never add `wp_lite.cuh`, `WP_OPEN`, `WP_B(`/`WP_E(` (or "
        "legacy `FI_*`) probes; profiling instruments a shadow copy.\n"
        "5. After each optimization re-run `weaveprof_megakernel_audit` / `weaveprof_profile`. The Stop gate "
        "refuses to finish until the final kernel bytes match a verified PASS seal (or you revert to a sealed "
        "incumbent).\n"
        "6. Off-limits: the supervisor key, the session store, seal/verify/ledger JSON, hook & MCP scripts, "
        "`.claude/`, `.codex/`, `.mcp.json`; never set `WP_SAMPLE` above 1.\n"
        f"{END_MARK}\n"
    )


def _upsert_block(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n") if path.is_file() else ""
    pat = re.compile(re.escape(BEGIN_MARK) + r".*?" + re.escape(END_MARK) + r"\n?", re.S)
    if pat.search(text):
        text = pat.sub(lambda _m: block, text, count=1)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text = text + ("\n" if text else "") + block
    _write_text(path, text)


_TOML_HDR_RE = re.compile(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*(?:#.*)?$")


def _merge_codex_toml(text: str, py: str, args: List[str]) -> str:
    lines = text.replace("\r\n", "\n").split("\n")
    out: List[str] = []
    skipping = False
    in_features = False
    features_seen = False
    hooks_set = False
    for line in lines:
        m = _TOML_HDR_RE.match(line)
        if m:
            name = m.group(1).replace('"', "").replace(" ", "")
            if in_features and not hooks_set:
                out.append("hooks = true")
                hooks_set = True
            in_features = name == "features"
            if in_features:
                features_seen = True
            skipping = name == "mcp_servers.weaveprofiler" or name.startswith("mcp_servers.weaveprofiler.")
            if skipping:
                continue
        elif skipping:
            continue
        elif in_features and re.match(r"^\s*hooks\s*=", line):
            out.append("hooks = true")
            hooks_set = True
            continue
        out.append(line)
    if in_features and not hooks_set:
        out.append("hooks = true")
        hooks_set = True
    while out and not out[-1].strip():
        out.pop()
    if not features_seen:
        body = list(out)
        first_hdr = next((i for i, ln in enumerate(body) if _TOML_HDR_RE.match(ln)), len(body))
        head, tail = body[:first_hdr], body[first_hdr:]
        while head and not head[-1].strip():
            head.pop()
        out = head + ([""] if head else []) + ["[features]", "hooks = true", ""] + tail
        while out and not out[-1].strip():
            out.pop()
    out += ["", "[mcp_servers.weaveprofiler]",
            f"command = {_toml_str(py)}",
            "args = [" + ", ".join(_toml_str(a) for a in args) + "]",
            "startup_timeout_sec = 30",
            "tool_timeout_sec = 1800",
            ""]
    return "\n".join(out).lstrip("\n")


def install_agent_workspace(
    workspace: Union[str, Path],
    protect_files: Optional[Sequence[Union[str, Path]]] = None,
    *,
    agent: str = "all",
    bench_cmd: Union[None, str, Sequence[str]] = None,
    card: str = "auto",
    arch: Optional[str] = None,
    max_overhead: float = 0.10,
    require_final_seal: bool = True,
    python_exe: Optional[str] = None,
    sessions_root: Optional[Union[str, Path]] = None,
    reset: bool = False,
) -> Dict[str, Any]:
    """Provision the session + Claude Code / Codex CLI wiring for a workspace (idempotent)."""
    ws = Path(workspace).expanduser().resolve()
    if not ws.is_dir():
        raise NotADirectoryError(f"Workspace not found: {ws}")
    agent = (agent or "all").lower()
    if agent not in ("all", "both", "claude", "codex"):
        raise ValueError("agent must be one of: all, claude, codex")
    if not (0.0 <= float(max_overhead) <= 0.5):
        raise ValueError("max_overhead must be within [0, 0.5]")
    files = _resolve_protect(ws, protect_files)
    sd = session_dir_for(ws, sessions_root)
    py = python_exe or sys.executable
    session = _provision_session(ws, files, sd=sd, bench=_bench_list(bench_cmd), card=card, arch=arch,
                                 max_overhead=max_overhead, require_final_seal=require_final_seal, reset=reset)
    written: List[str] = []
    want_claude = agent in ("all", "both", "claude")
    want_codex = agent in ("all", "both", "codex")
    skill = _skill_text(ws)
    block = _workflow_block(session)
    mcp_args = _mcp_args(ws, sd)

    if want_claude:
        mcp_path = ws / ".mcp.json"
        doc = _load_json(mcp_path)
        servers = doc.get("mcpServers") if isinstance(doc.get("mcpServers"), dict) else {}
        servers["weaveprofiler"] = {"type": "stdio", "command": _posix(py), "args": mcp_args, "env": {}}
        doc["mcpServers"] = servers
        _dump_json(mcp_path, doc)
        settings_path = ws / ".claude" / "settings.json"
        _dump_json(settings_path, _merge_hooks(_load_json(settings_path), py, ws, sd, "*"))
        local_path = ws / ".claude" / "settings.local.json"
        local = _load_json(local_path)
        local["enableAllProjectMcpServers"] = True
        enabled = local.get("enabledMcpjsonServers") if isinstance(local.get("enabledMcpjsonServers"), list) else []
        if "weaveprofiler" not in enabled:
            enabled.append("weaveprofiler")
        local["enabledMcpjsonServers"] = enabled
        _dump_json(local_path, local)
        skill_path = ws / ".claude" / "skills" / "weaveprofiler" / "SKILL.md"
        _write_text(skill_path, skill)
        _upsert_block(ws / "CLAUDE.md", block)
        written += [str(mcp_path), str(settings_path), str(local_path), str(skill_path), str(ws / "CLAUDE.md")]

    if want_codex:
        cfg_path = ws / ".codex" / "config.toml"
        old = cfg_path.read_text(encoding="utf-8") if cfg_path.is_file() else ""
        _write_text(cfg_path, _merge_codex_toml(old, _posix(py), mcp_args))
        hooks_path = ws / ".codex" / "hooks.json"
        _dump_json(hooks_path, _merge_hooks(_load_json(hooks_path), py, ws, sd, ".*"))
        skill_path = ws / ".agents" / "skills" / "weaveprofiler" / "SKILL.md"
        _write_text(skill_path, skill)
        _upsert_block(ws / "AGENTS.md", block)
        written += [str(cfg_path), str(hooks_path), str(skill_path), str(ws / "AGENTS.md")]

    q = _shell_quote
    run_py = _posix(SKILL_ROOT / "run.py")
    return {
        "ok": True,
        "workspace": str(ws),
        "session_dir": str(sd),
        "session_id": sd.name,
        "protected": [{"rel": e["rel"], "baseline_sha256": e["baseline_sha256"]} for e in session["protected"]],
        "agent": agent,
        "files_written": written,
        "launch": {
            "claude": f"cd {q(_posix(ws))} && claude",
            "codex": f"cd {q(_posix(ws))} && codex",
            "guard_exec": (f"{q(_posix(py))} -B {q(run_py)} guard-exec --workspace {q(_posix(ws))} -- "
                           "codex exec \"optimize the protected kernel with WeaveProfiler\""),
        },
        "notes": [
            "Claude Code: project MCP servers are pre-approved via .claude/settings.local.json.",
            "Codex CLI: trust the project so .codex/config.toml and .codex/hooks.json are loaded.",
        ],
    }


# ---------------------------------------------------------------------------
# Outer supervisor
# ---------------------------------------------------------------------------

def _bwrap_ok() -> bool:
    exe = shutil.which("bwrap")
    if not exe:
        return False
    try:
        return subprocess.run([exe, "--ro-bind", "/", "/", "true"], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class _Guardian(threading.Thread):
    """Reverts protected-file mutations every `interval` seconds while no verified PASS seal exists."""

    def __init__(self, sd: Path, interval: float = 0.1):
        super().__init__(name="weaveprof-guardian", daemon=True)
        self.sd = sd
        self.interval = interval
        self.stop_event = threading.Event()
        self.reverts: List[Dict[str, Any]] = []
        self._sealed = False
        self._seal_key: Any = None

    def _has_seal(self, session: Dict[str, Any]) -> bool:
        key = (len(session.get("seals") or []), session.get("updated_at"))
        if key != self._seal_key:
            self._seal_key = key
            self._sealed = bool(valid_pass_seals(self.sd, session))
        return self._sealed

    def check_once(self) -> None:
        session = read_session(self.sd)
        if not session:
            return
        mutated = mutated_since_baseline(session)
        if mutated and not self._has_seal(session):
            restored = restore_protected(session, mutated)
            if restored:
                self.reverts.append({"t": round(time.time(), 3), "restored": restored})
                sys.stderr.write(f"[WeaveProfiler wp guard-exec] reverted unsealed mutation of {restored}\n")

    def run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                self.check_once()
            except Exception as exc:  # keep guarding
                sys.stderr.write(f"[WeaveProfiler wp guard-exec] guardian error: {exc}\n")


def guard_exec(
    workspace: Union[str, Path],
    agent_cmd: Union[str, Sequence[str]],
    *,
    protect_files: Optional[Sequence[Union[str, Path]]] = None,
    use_bwrap: str = "auto",
    sessions_root: Optional[Union[str, Path]] = None,
    interval: float = 0.1,
    **install_kwargs: Any,
) -> Dict[str, Any]:
    """Run any CLI agent under the wp supervisor and return the final Stop-gate verdict."""
    ws = Path(workspace).expanduser().resolve()
    sd = session_dir_for(ws, sessions_root)
    installed = None
    if read_session(sd) is None:
        installed = install_agent_workspace(ws, protect_files, sessions_root=sessions_root, **install_kwargs)
    if isinstance(agent_cmd, str):
        cmd: Union[str, List[str]] = shlex.split(agent_cmd) if os.name != "nt" else agent_cmd
    else:
        cmd = [str(c) for c in agent_cmd]
    if not cmd:
        raise ValueError("agent_cmd is empty")
    env = dict(os.environ)
    for k in ("WEAVEPROF_SUPERVISOR_HMAC_KEY", "WEAVEPROF_SUPERVISOR_KEY_FILE"):
        env.pop(k, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    use_bwrap = (use_bwrap or "auto").lower()
    wrapped = False
    if use_bwrap in ("auto", "always") and platform.system() == "Linux" and _bwrap_ok():
        root = str(SKILL_ROOT)
        argv = cmd if isinstance(cmd, list) else shlex.split(cmd)
        ro_mounts = ["--ro-bind", root, root]
        for target in (ws / ".claude", ws / ".codex", ws / ".agents" / "skills" / "weaveprofiler",
                       ws / ".mcp.json", sd / KEY_FILE):
            if target.exists():
                t = str(target)
                ro_mounts.extend(["--ro-bind", t, t])
        cmd = ["bwrap", "--dev-bind", "/", "/", *ro_mounts, "--die-with-parent", "--"] + argv
        wrapped = True
    elif use_bwrap == "always":
        raise RuntimeError("use_bwrap='always' but bubblewrap (bwrap --ro-bind / / true) is unavailable")
    guardian = _Guardian(sd, interval)
    guardian.check_once()
    guardian.start()
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(ws), env=env, shell=isinstance(cmd, str))
        rc = proc.returncode
    finally:
        guardian.stop_event.set()
        guardian.join(timeout=5)
    guardian.check_once()
    completion = verify_session_completion(session_dir=str(sd))
    return {
        "ok": rc == 0 and bool(completion.get("ok")),
        "agent_returncode": rc,
        "bwrap": wrapped,
        "elapsed_s": round(time.time() - t0, 3),
        "reverts": guardian.reverts,
        "installed": installed is not None,
        "completion": completion,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_install_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workspace", default=".")
    p.add_argument("--protect", action="append", default=None, help="Protected kernel file (repeatable)")
    p.add_argument("--agent", default="all", choices=("all", "both", "claude", "codex"))
    p.add_argument("--bench", default=None, help="Default benchmark command (string)")
    p.add_argument("--card", default="auto")
    p.add_argument("--arch", default=None)
    p.add_argument("--max-overhead", type=float, default=0.10)
    p.add_argument("--no-require-final-seal", action="store_true")
    p.add_argument("--python", default=None)
    p.add_argument("--sessions-root", default=None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args_in = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="weaveprof-agent", description="WeaveProfiler (wp) agent installer/supervisor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_inst = sub.add_parser("install", help="Provision Claude Code / Codex CLI wiring + session")
    _add_install_args(p_inst)
    p_inst.add_argument("--reset", action="store_true", help="Regenerate key, baseline and seals")
    p_ge = sub.add_parser("guard-exec", help="Run an agent command under the wp supervisor")
    _add_install_args(p_ge)
    p_ge.add_argument("--bwrap", default="auto", choices=("auto", "always", "never"))
    p_ge.add_argument("agent_cmd", nargs=argparse.REMAINDER)
    p_st = sub.add_parser("status", help="Print the Stop-gate verdict for a workspace")
    p_st.add_argument("--workspace", default=".")
    p_st.add_argument("--sessions-root", default=None)
    args = ap.parse_args(args_in)
    try:
        if args.cmd == "install":
            res = install_agent_workspace(
                args.workspace, args.protect, agent=args.agent, bench_cmd=args.bench, card=args.card,
                arch=args.arch, max_overhead=args.max_overhead, require_final_seal=not args.no_require_final_seal,
                python_exe=args.python, sessions_root=args.sessions_root, reset=args.reset)
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return 0
        if args.cmd == "guard-exec":
            cmd = list(args.agent_cmd)
            if cmd and cmd[0] == "--":
                cmd = cmd[1:]
            if not cmd:
                ap.error("guard-exec requires an agent command after `--`")
            res = guard_exec(
                args.workspace, cmd, protect_files=args.protect, use_bwrap=args.bwrap,
                sessions_root=args.sessions_root, agent=args.agent, bench_cmd=args.bench, card=args.card,
                arch=args.arch, max_overhead=args.max_overhead, require_final_seal=not args.no_require_final_seal,
                python_exe=args.python)
            print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
            return 0 if res["ok"] else 1
        res = verify_session_completion(str(Path(args.workspace).resolve()), sessions_root_dir=args.sessions_root)
        print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
        return 0 if res.get("ok") else 1
    except (ValueError, OSError, RuntimeError) as exc:
        sys.stderr.write(f"weaveprof-agent {args.cmd} error: {type(exc).__name__}: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
