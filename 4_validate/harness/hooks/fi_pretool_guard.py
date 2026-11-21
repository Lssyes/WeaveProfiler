#!/usr/bin/env python3
"""PreToolUse guard for weaveprofiler.

Global rules (every session):
  R1 upload zero-trust: upload-tool commands carrying measurement files
     (trace__*, *.perfetto.json.gz, *.nsys-rep, exploratory__*,
     *.delivery_receipt.json, PREFLIGHT/ALL_GATES.json) are denied unless each
     file holds an upload permit (issued by verify_receipt.py on PASS, bound
     to SHA256) or the target is a run directory containing a verifiable
     receipt.
  R2 SPEC-1 namespace lock: trace__*.perfetto.json.gz may only be produced by
     run_measurement.py internals.
  R3 evidence lock: gate evidence files are read-only from the shell.

Armed sessions (the session touched the measurement stack -- skill scripts,
nsys profile, ncu): artifact outputs (*.perfetto.json.gz, *.nsys-rep) may be
created only by the orchestrator or under the exploratory__ namespace,
regardless of how the write is performed.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hooklib import (ARM_TRIGGERS, SPEC1_IN_TEXT, allow, arm_session,
                     armed_state, audit, deny, journal_cwd, load_config,
                     permit_for, read_payload, shell_command)

EVIDENCE = re.compile(
    r"(?:^|[\s/\\'\"=])(?:PREFLIGHT\.json|ALL_GATES\.json|episode_ledger\.json|"
    r"verify\.json|feedback\.json|[\w.-]*seal\.json|[\w.-]*\.delivery_receipt\.json)"
)
READONLY_BASE = re.compile(
    r"^\s*(?:sudo\s+)?(?:cat|ls|head|tail|grep|zgrep|zcat|stat|file|wc|du|find|"
    r"sha256sum|md5sum|realpath|basename|dirname|test)\b"
)
READONLY_VERIFY = re.compile(
    r"^\s*python3?\s+(?:-[BuI]\s+)*(?:(?P<vscript>\S*verify_receipt\.py)|-m\s+weaveprof\s+seal-check)\b"
)
FIND_DANGEROUS = re.compile(r"\s-(?:exec|execdir|delete|ok|okdir)\b")
SHELL_SUB_RE = re.compile(r"\$\(|`")
RENOUNCE = re.compile(
    r"^\s*(?:mv|rename)\s+(?:-\w+\s+)*\S*(?:trace__|\.perfetto\.json\.gz|\.perfetto-trace|\.nsys-rep)\S*"
    r"\s+\S*exploratory__\S*\s*$"
    r"|^\s*rm\s+(?:-\w+\s+)*\S*(?:trace__|\.perfetto\.json\.gz|\.perfetto-trace|\.nsys-rep)"
)
ORCHESTRATOR_HEAD = re.compile(
    r"^\s*(?:(?:[A-Za-z_]\w*=[^\s;]*\s+)*)"
    r"(?:(?:env\s+(?:[A-Za-z_]\w*=[^\s;]*\s+)*)?"
    r"(?:python3?\s+(?:-[BuI]\s+)*(?:\S*(?:run_measurement\.py|closed_loop_verifier\.py)|-m\s+(?:weaveprof|autoprof)\b)"
    r"|(?:weaveprof|autoprof)\b))"
)
ARTIFACT = re.compile(r"[^\s;'\"]+\.(?:perfetto\.json\.gz|perfetto-trace|nsys-rep)")
EXPLORATORY_OK = re.compile(r"exploratory__|/exploratory/|failed_trace\.perfetto-trace")
UPLOAD_MEASUREMENT = re.compile(
    r"(?:^|/|\\)trace__[^\s;'\"]*|[^\s;'\"]+\.perfetto\.json\.gz|[^\s;'\"]+\.perfetto-trace|[^\s;'\"]+\.nsys-rep|"
    r"[^\s;'\"]*exploratory__[^\s;'\"]*|[^\s;'\"]*\.delivery_receipt\.json|[^\s;'\"]*seal\.json|"
    r"(?:^|/|\\)(?:PREFLIGHT|ALL_GATES)\.json"
)
_FLOAT_TOKEN = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
LIMIT_FLAG_RE = re.compile(r"(?:--limit)(?:=|\s+)" + _FLOAT_TOKEN)
LIMIT_PCT_FLAG_RE = re.compile(r"(?:--overhead-limit-pct)(?:=|\s+)" + _FLOAT_TOKEN)
FORBIDDEN_SAMPLE_RE = re.compile(
    r"(?:\b(?:FI_SAMPLE|WP_SAMPLE|--fi-sample|--sample-stride)\s*(?:=|\s)\s*[+]?(?:0x[0-9a-f]+|[2-9]|\d{2,})"
    r"|['\"](?:FI_SAMPLE|WP_SAMPLE)['\"]\s*\]\s*=\s*['\"]?[+]?(?:0x[0-9a-f]+|[2-9]|\d{2,}))",
    re.I,
)
WRITE_TOOL_NAMES = frozenset((
    "write", "edit", "multiedit", "notebookedit",
    "write_file", "edit_file", "replace_file_content", "multi_replace_file_content",
))


def strip_unquoted_comment(seg):
    """Remove trailing `# ...` shell comments outside single/double quotes."""
    in_s = False
    in_d = False
    esc = False
    for idx, ch in enumerate(seg):
        if esc:
            esc = False
            continue
        if ch == "\\" and not in_s:
            esc = True
            continue
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d and (idx == 0 or seg[idx - 1].isspace()):
            return seg[:idx].rstrip()
    return seg


def is_readonly_cmd(seg, cwd=None):
    clean = strip_unquoted_comment(seg)
    if SHELL_SUB_RE.search(clean) or ">" in clean:
        return False
    if READONLY_BASE.match(clean):
        if re.match(r"^\s*(?:sudo\s+)?find\b", clean) and FIND_DANGEROUS.search(clean):
            return False
        return True
    m = READONLY_VERIFY.match(clean)
    if m:
        vscript = m.group("vscript")
        if vscript:
            # Audit #2: verify_receipt.py must resolve inside this skill tree, not an agent-authored fake in cwd.
            skill_root = Path(__file__).resolve().parents[3]
            cand = Path(vscript)
            if not cand.is_absolute():
                cand = Path(cwd or ".") / cand
            try:
                if not cand.resolve().is_relative_to(skill_root.resolve()):
                    return False
            except OSError:
                return False
        return True
    return False


def is_orchestrator_cmd(seg):
    clean = strip_unquoted_comment(seg)
    if SHELL_SUB_RE.search(clean):
        return False
    return bool(ORCHESTRATOR_HEAD.match(clean))


def segments(command):
    # Audit #3: Split on newlines, `;`, `&&`, `||`, `|`, and single background `&` (preserving `>&` / `2>&1`).
    for part in re.split(r"\r?\n|;|&&|\|\||\||(?<![<>0-9])&(?!&)", command):
        part = part.strip()
        if part:
            yield part


def check_write_tool(payload, armed):
    """Audit #1: Enforce R2/R3/armed on non-Bash file-writing tools (Write/Edit/write_file/etc.)."""
    tool_name = str(payload.get("tool_name") or "").strip().lower()
    if tool_name not in WRITE_TOOL_NAMES:
        return None
    tool_input = payload.get("tool_input") or {}
    targets = []
    for k in ("file_path", "path", "Path", "TargetFile", "target_file", "notebook_path"):
        val = tool_input.get(k)
        if isinstance(val, str) and val.strip():
            targets.append(val.strip())
    for edit_list_key in ("edits", "files"):
        for item in tool_input.get(edit_list_key) or []:
            if isinstance(item, dict):
                for k in ("file_path", "path", "Path"):
                    val = item.get(k)
                    if isinstance(val, str) and val.strip():
                        targets.append(val.strip())
    for target in targets:
        if EVIDENCE.search(" " + target):
            return (
                "[weaveprofiler R3] 禁止通过文件写入工具（%s）直接修改门禁证据/封存文件: %s"
                % (tool_name, target[:140])
            )
        if SPEC1_IN_TEXT.search(target):
            return (
                "[weaveprofiler R2] 禁止通过文件写入工具（%s）直接生成 SPEC-1 交付产物: %s"
                % (tool_name, target[:140])
            )
        if armed and ARTIFACT.search(target) and not EXPLORATORY_OK.search(target):
            return (
                "[weaveprofiler armed] 本会话已进入测量门禁态，禁止通过 %s 直接写测量产物: %s"
                % (tool_name, target[:140])
            )
    return None


def check_segment(seg, cwd=None):
    """R2/R3 for all sessions: return deny reason or None."""
    clean = strip_unquoted_comment(seg)
    has_spec1 = bool(SPEC1_IN_TEXT.search(clean))
    has_evidence = bool(EVIDENCE.search(clean))
    if not has_spec1 and not has_evidence:
        return None
    if SHELL_SUB_RE.search(clean):
        return ("[weaveprofiler R3] 禁止在涉及门禁证据/交付文件的命令中使用命令替换 ($(...) / 反引号): %s"
                % clean[:120])
    if is_orchestrator_cmd(clean) or is_readonly_cmd(clean, cwd=cwd):
        return None
    if has_spec1 and RENOUNCE.match(clean):
        return None
    if has_evidence:
        return ("[weaveprofiler R3] 门禁证据文件（PREFLIGHT/ALL_GATES/receipt/seal/verify/feedback）"
                "只允许由编排器流程写入。被拦段: %s" % clean[:120])
    return ("[weaveprofiler R2] trace__*.perfetto.json.gz 是交付命名空间，只能由 "
            "run_measurement.py + finalize_delivery.py 产出。读取/验票/改名到 "
            "exploratory__/删除伪产物都是允许的；被拦段: %s" % clean[:120])


def check_armed_segment(seg, cwd=None):
    """Armed mode: artifact outputs monopolized by orchestrator/exploratory."""
    clean = strip_unquoted_comment(seg)
    if is_readonly_cmd(clean, cwd=cwd) or is_orchestrator_cmd(clean) or RENOUNCE.match(clean):
        return None
    for match in ARTIFACT.finditer(clean):
        token = match.group(0)
        if EXPLORATORY_OK.search(token):
            continue
        return ("[weaveprofiler armed] 本会话已进入测量门禁态：测量产物 %s 只能由 "
                "编排器产出，或落 exploratory__ 命名空间（不得交付、不得上传）。被拦段: %s"
                % (token, clean[:120]))
    return None


def local_path_tokens(command, cwd):
    for raw in re.split(r"\s+", command):
        raw = raw.strip("'\"")
        if not raw or "://" in raw:
            continue
        # Audit #13: Also extract `--flag=path` values so upload flags cannot smuggle paths.
        if raw.startswith("-"):
            if "=" not in raw:
                continue
            raw = raw.split("=", 1)[1].strip("'\"")
            if not raw or "://" in raw:
                continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = Path(cwd or ".") / candidate
        yield raw, candidate


def dir_has_valid_receipt(path):
    receipts = list(path.glob("*.delivery_receipt.json"))
    receipts += list(path.glob("*/*.delivery_receipt.json"))
    verifier = Path(__file__).resolve().parent.parent / "verify_receipt.py"
    for receipt in receipts[:4]:
        try:
            proc = subprocess.run([sys.executable, "-B", str(verifier),
                                   "--receipt", str(receipt)],
                                  capture_output=True, text=True, timeout=60)
            if proc.returncode == 0:
                return True
        except (OSError, subprocess.TimeoutExpired):
            continue
    return False


def check_upload(command, cwd):
    """R1: every measurement file in an upload command needs a permit."""
    for token, candidate in local_path_tokens(command, cwd):
        if candidate.is_dir():
            if dir_has_valid_receipt(candidate):
                continue
            try:
                next(candidate.iterdir())
            except StopIteration:
                continue
            except OSError:
                continue
            return ("[weaveprofiler R1] 上传目录 %s 内没有可通过验票的 receipt，"
                    "整目录上传被拒绝。" % token)
        if not candidate.is_file() or not UPLOAD_MEASUREMENT.search(token):
            continue
        permit = permit_for(candidate)
        if permit:
            audit("allow", rule="R1", file=str(candidate))
            continue
        return ("[weaveprofiler R1] %s 没有上传许可证。许可证唯一来源："
                "verify_receipt.py 验票 PASS（即走完 run_measurement.py 门禁链）。"
                "exploratory__ 产物定义为本地证据，同样不得上传。" % token)
    return None


def check_policy_segment(seg, config):
    """R4 anti-cheat policy: block limit relaxation and forbidden loop sampling."""
    clean = strip_unquoted_comment(seg)
    ceiling = float(config.get("max_overhead_limit", 0.10))
    if FORBIDDEN_SAMPLE_RE.search(clean):
        return (
            "[weaveprofiler R4] 禁止使用循环抽样参数（FI_SAMPLE/WP_SAMPLE > 1）；"
            "开销超限时必须走闭环外提（Scheme A coarsening）。被拦段: %s" % clean[:120]
        )
    for match in LIMIT_FLAG_RE.finditer(clean):
        val = float(match.group(1))
        if val > ceiling + 1e-12:
            return (
                "[weaveprofiler R4] 禁止放宽开销阈值（--limit %.4f > 硬上限 %.4f）。被拦段: %s"
                % (val, ceiling, clean[:120])
            )
    for match in LIMIT_PCT_FLAG_RE.finditer(clean):
        val = float(match.group(1))
        if val > ceiling * 100.0 + 1e-9:
            return (
                "[weaveprofiler R4] 禁止放宽开销阈值（--overhead-limit-pct %.2f%% > 硬上限 %.2f%%）。被拦段: %s"
                % (val, ceiling * 100.0, clean[:120])
            )
    return None


def main():
    try:
        payload = read_payload()
        session = payload.get("session_id") or "unknown"
        cwd = payload.get("cwd")
        armed = armed_state(session) is not None

        write_deny = check_write_tool(payload, armed)
        if write_deny:
            audit("deny", rule="R2/R3/write_tool", tool=payload.get("tool_name"))
            deny(write_deny)

        command = shell_command(payload)
        if not command:
            allow()
        journal_cwd(session, cwd)

        for trigger in ARM_TRIGGERS:
            if trigger in command:
                arm_session(session, "trigger=%s" % trigger)
                armed = True
                break

        config = load_config()
        upload_tools = tuple(config.get("upload_tool_patterns", []))
        for seg in segments(command):
            if any(tool in strip_unquoted_comment(seg) for tool in upload_tools):
                reason = check_upload(seg, cwd)
                if reason:
                    audit("deny", rule="R1", command=seg[:400])
                    deny(reason)
                continue
            reason = check_policy_segment(seg, config)
            if not reason:
                reason = check_segment(seg, cwd=cwd)
            if not reason and armed:
                reason = check_armed_segment(seg, cwd=cwd)
            if reason:
                audit("deny", rule="R2/R3/R4/armed", command=seg[:400])
                deny(reason)
        allow()
    except SystemExit:
        raise
    except Exception as exc:
        # Audit #18: Fail-closed on unexpected hook error.
        audit("error_deny", error=f"{type(exc).__name__}: {exc}")
        deny(f"[weaveprofiler guard fail-closed] 门禁钩子解析异常: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
