#!/usr/bin/env python3
"""SessionStart/SubagentStart/SessionEnd hook: timestamped session markers.

The stop gate scans for delivery-shaped files newer than the session marker,
so markers are what scopes "this session's outputs" on a shared machine.
"""
import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from hooklib import audit, read_payload, session_marker_path


def main():
    payload = read_payload()
    event = payload.get("hook_event_name", "")
    session = payload.get("session_id") or "unknown"
    marker = session_marker_path(session)
    record = {}
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if event in ("SessionStart", "SubagentStart"):
        record.update({"session_id": session, "event": event,
                       "started_at": time.time(),
                       "cwd": payload.get("cwd"),
                       "transcript_path": payload.get("transcript_path")})
    elif event == "SessionEnd":
        record.update({"session_id": session, "ended_at": time.time()})
    else:
        sys.exit(0)
    try:
        marker.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    audit("marker", hook_event=event, session=session)
    sys.exit(0)


if __name__ == "__main__":
    main()
