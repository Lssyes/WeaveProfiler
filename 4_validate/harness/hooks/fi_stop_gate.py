#!/usr/bin/env python3
"""Stop/SubagentStop gate: armed sessions may only exit via receipt or BLOCKED.

Normal sessions: scan session directories for SPEC-1 delivery files newer
than the session marker and verify each one.

Armed sessions (measurement stack was touched): every measurement artifact
(*.perfetto.json.gz, *.nsys-rep) found in session directories must either
verify or live under the exploratory__ namespace; and if the session produced
no verified delivery at all, it must leave honest BLOCKED evidence (a FAIL
PREFLIGHT/ALL_GATES from a real run) before it may stop.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hooklib import (SKILL_ROOT, SPEC1, STATE_DIR, allow, armed_state, audit, block_stop,
                     load_config, read_payload, session_marker_path)

ARTIFACT_NAME = re.compile(r"\.(perfetto\.json\.gz|perfetto-trace|nsys-rep)$")
EXPLORATORY_OK = re.compile(r"exploratory__|/exploratory/|failed_trace\.perfetto-trace")


def scan(root, since, prune, deadline, pred):
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if time.time() > deadline:
            raise TimeoutError(root)
        dirnames[:] = [d for d in dirnames if d not in prune]
        for name in filenames:
            if not pred(name):
                continue
            path = os.path.join(dirpath, name)
            try:
                if os.path.getmtime(path) >= since:
                    found.append(path)
            except OSError:
                continue
    return found


def scan_roots_for(config, marker, session):
    roots = list(config.get("scan_roots", []))
    if marker.get("cwd"):
        roots.append(marker["cwd"])
    if config.get("scan_journaled_cwds", True):
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", session)
        try:
            roots.extend((STATE_DIR / ("cwds_%s.log" % safe))
                         .read_text(encoding="utf-8").splitlines())
        except OSError:
            pass
    seen, result = set(), []
    for root in roots:
        if root and root not in seen and os.path.isdir(root):
            seen.add(root)
            result.append(root)
    return result


def blocked_evidence(roots, since, prune, deadline, max_limit=0.10):
    """Honest-exit evidence: FAIL PREFLIGHT/ALL_GATES or HMAC-verified sealed FAIL evidence."""
    try:
        for root in roots:
            for path in scan(root, since, prune, deadline,
                             lambda n: n in ("PREFLIGHT.json", "ALL_GATES.json", "seal.json") or n.endswith(".seal.json")):
                if path.endswith("seal.json"):
                    if str(SKILL_ROOT) not in sys.path:
                        sys.path.insert(0, str(SKILL_ROOT))
                    try:
                        from weaveprof.harness import verify_seal
                        chk = verify_seal(path, require_pass=False, max_overhead_limit=max_limit, require_hmac=True)
                        if chk.get("ok") and (chk.get("seal") or {}).get("status") == "FAIL":
                            return path
                    except Exception:
                        continue
                    continue
                try:
                    head = Path(path).read_text(encoding="utf-8",
                                                errors="replace")[:200000]
                except OSError:
                    continue
                if ('"all_required_pass": false' in head
                        or '"status": "FAIL"' in head
                        or '"overall_status": "FAIL"' in head):
                    return path
    except TimeoutError:
        return None
    return None


def verify_sealed_trace(artifact, max_limit):
    """Check whether `<artifact>.seal.json` passes fail-closed EpisodeSealer verification."""
    seal_file = Path(str(artifact) + ".seal.json")
    if not seal_file.is_file():
        return None
    if str(SKILL_ROOT) not in sys.path:
        sys.path.insert(0, str(SKILL_ROOT))
    try:
        from weaveprof.harness import verify_seal
        return verify_seal(seal_file, require_pass=True, max_overhead_limit=max_limit)
    except Exception as exc:
        return {"ok": False, "errors": [f"seal verification error: {exc}"]}


def main():
    payload = read_payload()
    session = payload.get("session_id") or "unknown"
    config = load_config()
    try:
        marker = json.loads(session_marker_path(session).read_text(encoding="utf-8"))
        since = float(marker.get("started_at")) - 5.0
    except (OSError, ValueError, TypeError, KeyError):
        audit("stop_gate_skip", session=session, reason="no session marker")
        allow()

    deadline = time.time() + float(config.get("scan_max_seconds", 30))
    prune = set(config.get("scan_prune_dir_names", []))
    roots = scan_roots_for(config, marker, session)
    armed = armed_state(session) is not None
    verifier = Path(__file__).resolve().parent.parent / "verify_receipt.py"

    pred = (lambda n: bool(ARTIFACT_NAME.search(n))) if armed \
        else (lambda n: bool(SPEC1.match(n)))
    try:
        artifacts = []
        for root in roots:
            artifacts.extend(scan(root, since, prune, deadline, pred))
    except TimeoutError:
        audit("stop_gate_skip", session=session, reason="scan timeout")
        allow()

    failures = []
    verified_any = False
    max_limit = float(config.get("max_overhead_limit", 0.10))
    for artifact in sorted(set(artifacts)):
        if armed and EXPLORATORY_OK.search(artifact):
            audit("stop_gate_exploratory", session=session, path=artifact)
            continue
        seal_res = verify_sealed_trace(artifact, max_limit)
        if seal_res is not None:
            if seal_res.get("ok"):
                verified_any = True
                continue
            errs = (seal_res.get("errors") or ["seal verification failed"])[:3]
            failures.append((artifact, errs))
            audit("stop_gate_fail", session=session, trace=artifact, errors=errs)
            continue
        try:
            proc = subprocess.run([sys.executable, "-B", str(verifier),
                                   "--trace", artifact, "--json"],
                                  capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append((artifact, [f"verify_receipt error: {exc}"]))
            audit("stop_gate_fail", session=session, trace=artifact, errors=[str(exc)])
            continue
        if proc.returncode == 0:
            verified_any = True
            continue
        try:
            errors = json.loads(proc.stdout).get("errors") or []
        except ValueError:
            errors = []
        if not errors:
            errors = [proc.stdout.strip().splitlines()[0]
                      if proc.stdout.strip() else "verify failed"]
        failures.append((artifact, errors[:3]))
        audit("stop_gate_fail", session=session, trace=artifact, errors=errors[:3])

    if not failures and armed and not verified_any:
        evidence = blocked_evidence(roots, since, prune, deadline, max_limit=max_limit)
        if evidence:
            audit("stop_gate_blocked_exit", session=session, evidence=evidence)
            allow()
        failures.append(("(本会话未产出任何交付物)", [
            "armed 会话的合法出口只有两个：有效 receipt，或真实的 BLOCKED "
            "证据（PREFLIGHT/ALL_GATES 显示 FAIL 的文件）。不允许空手收尾。"]))

    if not failures:
        allow()
    lines = ["[weaveprofiler 收工门禁] 本会话存在无法验票的测量产物或缺少合法出口。"
             "每个文件只有两条合法出路："]
    for artifact, errors in failures:
        lines.append("- %s" % artifact)
        for error in errors:
            lines.append("    原因: %s" % error)
    lines.append("合法出路 A：走 run_measurement.py 完整测量链，让 finalize_delivery.py "
                 "产出 PASS receipt；")
    lines.append("合法出路 B：确认它只是探索性中间产物，重命名为 exploratory__ 前缀"
                 "（保留在本地，不得交付、不得上传）。")
    block_stop("\n".join(lines), payload, "fi_stop_gate")


if __name__ == "__main__":
    main()
