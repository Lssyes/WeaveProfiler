#!/usr/bin/env python3
"""Shared protocol helpers for the weaveprofiler enforcement hooks.

Wire protocol (verified against codex-cli 0.157.0):
  * stdin  : one JSON object; hook_event_name, session_id, cwd,
             tool_name ("Bash" for shell) + tool_input.command on PreToolUse,
             last_assistant_message + stop_hook_active on Stop.
  * deny   : stdout {"hookSpecificOutput": {"hookEventName": "PreToolUse",
             "permissionDecision": "deny",
             "permissionDecisionReason": <fed back to the model>}}, exit 0.
  * block stop: stdout {"decision": "block", "reason": <fed back>}, exit 0.
             stop_hook_active in the payload marks a forced continuation --
             never block twice in a row or the session deadlocks.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = SKILL_ROOT / "4_validate" / "harness"
STATE_DIR = Path(os.environ.get("FI_STATE_DIR", str(HARNESS_DIR / "state")))
CONFIG_PATH = HARNESS_DIR / "config.json"
VERIFY = HARNESS_DIR / "verify_receipt.py"

CARD_TOKENS = ("h20", "b300", "mi308x")
SPEC1 = re.compile(
    r"^trace__(?P<model>[a-z0-9]+)_(?P<stage>[a-z0-9]+)"
    r"__card-(?P<card>h20|b300|mi308x)"
    r"__bs(?P<batch>\d+)_sq(?P<seqlen_q>\d+)(?:_kv(?P<seqlen_kv>\d+))?"
    r"__(?P<dtype>[a-z0-9]+)"
    r"__grid(?P<grid>\d+)_warp(?P<warp>\d+)_ph(?P<phases>\d+)"
    r"__(?P<bundle>sw|hw|sw_hw)\.perfetto\.json\.gz$")


# Shell-command detection is deliberately broader than the canonical SPEC-1:
# an old no-card trace name must still enter the guarded delivery namespace and
# then be rejected by verify_receipt.py instead of slipping past the hook.
SPEC1_IN_TEXT = re.compile(r"trace__[^\s;'\"]+\.perfetto\.json\.gz")
EXPLORATORY_PREFIX = "exploratory__"

DEFAULT_CONFIG = {
    "scan_roots": [],
    "scan_prune_dir_names": [".git", "node_modules", "__pycache__", "3rdparty",
                             "ThunderKittens", "target"],
    "scan_max_seconds": 30,
    "upload_tool_patterns": ["ossutil", "ossutil64", "aws s3", "rclone", "mc cp"],
    "max_consecutive_stop_blocks": 2,
    "max_overhead_limit": 0.10,
}


def load_config():
    try:
        with open(str(CONFIG_PATH), encoding="utf-8") as handle:
            config = json.load(handle)
        merged = dict(DEFAULT_CONFIG)
        merged.update(config)
        return merged
    except (OSError, ValueError):
        return dict(DEFAULT_CONFIG)


def read_payload():
    try:
        return json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return {}


def audit(event, **fields):
    """Append one audit line; never raises (hooks must not crash the host)."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.time(), "event": event}
        record.update(fields)
        with open(str(STATE_DIR / "hook_events.log"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        pass


def allow():
    sys.exit(0)


def deny(reason):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason}}, ensure_ascii=False))
    sys.exit(0)


def block_stop(reason, payload, session_state_key):
    """Block stop at most max_consecutive_stop_blocks times per session."""
    config = load_config()
    limit = int(config.get("max_consecutive_stop_blocks", 2))
    session = payload.get("session_id") or "unknown"
    counter_path = STATE_DIR / ("stop_blocks_%s" % re.sub(r"[^A-Za-z0-9_-]", "_", session))
    count = 0
    try:
        count = int(counter_path.read_text().strip())
    except (OSError, ValueError):
        pass
    if payload.get("stop_hook_active") and count >= limit:
        audit("stop_block_limit_reached", session=session, reason=reason)
        allow()
    try:
        counter_path.write_text(str(count + 1))
    except OSError:
        pass
    print(json.dumps({"decision": "block", "reason": reason}, ensure_ascii=False))
    sys.exit(0)


def shell_command(payload):
    """Extract the shell command string from a PreToolUse payload, if any."""
    tool_input = payload.get("tool_input") or {}
    command = tool_input.get("command") or tool_input.get("cmd")
    if isinstance(command, list):
        command = " ".join(str(part) for part in command)
    if isinstance(command, str):
        return command
    return None


def session_marker_path(session_id):
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", session_id or "unknown")
    return STATE_DIR / ("session_%s.json" % safe)


def journal_cwd(session_id, cwd):
    """Remember every cwd a session runs commands in; the stop gate scans
    these subtrees instead of walking the whole workspace (too slow on
    network filesystems)."""
    if not session_id or not cwd:
        return
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", session_id)
        path = STATE_DIR / ("cwds_%s.log" % safe)
        existing = set()
        try:
            existing = set(path.read_text(encoding="utf-8").splitlines())
        except OSError:
            pass
        if cwd not in existing:
            with open(str(path), "a", encoding="utf-8") as handle:
                handle.write(cwd + "\n")
    except OSError:
        pass


# --- armed-session model -----------------------------------------------------
# A session arms itself the moment it touches the measurement stack.  While
# armed, artifact creation is monopolized by the orchestrator and uploads
# require permits; the only exits are a PASS receipt or an honest BLOCKED
# evidence trail.
ARM_TRIGGERS = (
    "run_measurement.py", "validate_preflight.py", "capture_nsys.py",
    "decode_profile.py", "emit_perfetto.py", "materialize_contract.py",
    "run_profile.py", "nsys profile", "nsys start", "nsys launch",
    "nsys export", "nsys stats", "ncu ",
    "weaveprof", "autoprof", "closed_loop_verifier.py", "llm_probe_placer.py",
    "rocprof-sys-run", "rocprofv3",
)


def armed_marker_path(session_id):
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", session_id or "unknown")
    return STATE_DIR / ("armed_%s.json" % safe)


def arm_session(session_id, reason):
    path = armed_marker_path(session_id)
    if path.is_file():
        return
    try:
        path.write_text(json.dumps(
            {"session_id": session_id, "armed_at": time.time(),
             "reason": reason}, ensure_ascii=False))
        audit("armed", session=session_id, reason=reason[:200])
    except OSError:
        pass


def armed_state(session_id):
    try:
        return json.loads(armed_marker_path(session_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


PERMITS_PATH = STATE_DIR / "upload_permits.json"


def issue_permits(receipt_path, receipt):
    """On PASS, permit every file bound in the receipt for upload."""
    permits = {}
    try:
        permits = json.loads(PERMITS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    import hashlib

    def _sha(p):
        digest = hashlib.sha256()
        with open(str(p), "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    bound = [receipt.get("trace_path"), receipt.get("contract_path"),
             receipt.get("all_gates_path"), receipt.get("preflight_path")]
    bound += [r.get("path") for r in (receipt.get("delivery_inputs") or {}).values()]
    issued = 0
    for path_str in bound:
        if not path_str:
            continue
        path = Path(path_str)
        if not path.is_file():
            continue
        permits[_sha(path)] = {"path": str(path), "receipt": str(receipt_path),
                               "issued_at": time.time()}
        issued += 1
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        PERMITS_PATH.write_text(json.dumps(permits, indent=1, sort_keys=True))
    except OSError:
        pass
    return issued


def permit_for(path):
    """Return the permit dict if this exact file content is upload-permitted."""
    import hashlib
    try:
        if Path(path).stat().st_size > 2 * 1024**3:
            return None
        digest = hashlib.sha256()
        with open(str(path), "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        permits = json.loads(PERMITS_PATH.read_text(encoding="utf-8"))
        return permits.get(digest.hexdigest())
    except (OSError, ValueError):
        return None
