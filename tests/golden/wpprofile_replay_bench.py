#!/usr/bin/env python3
"""Offline replay benchmark for ``weaveprof profile`` golden tests.

Instead of compiling with ``nvcc`` or capturing on a GPU with ``nsys``, this
script verifies the SHA-256 digests of ``WEAVEPROF_BASELINE_SOURCE`` and
``WEAVEPROF_INSTRUMENTED_SOURCE`` against ``recorded/*.sha256`` (pinning
``rule_engine.inject_probes`` byte-for-byte), then stages the recorded build and
capture artifacts into ``WEAVEPROF_ROUND_DIR`` (decompressing ``*.xz`` files).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil

_SHA_FILES = ("baseline_source.sha256", "instrumented_source.sha256")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _record_pid() -> None:
    pid_report = os.environ.get("P1D_PID_REPORT_PATH")
    if not pid_report:
        return
    try:
        with Path(pid_report).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"pid": os.getpid(), "role": "wpprofile_replay_bench"}) + "\n")
    except OSError:
        pass


def _verify_source_digest(label: str, env_key: str, expected_file: Path) -> tuple[bool, str]:
    src_raw = os.environ.get(env_key)
    if not src_raw:
        return False, f"{label}: environment variable {env_key} is not set"
    src_path = Path(src_raw)
    if not src_path.is_file():
        return False, f"{label}: source file missing at {src_path}"
    if not expected_file.is_file():
        return False, f"{label}: expected digest file missing at {expected_file}"
    expected = expected_file.read_text(encoding="utf-8").strip().lower()
    actual = _sha256_file(src_path).lower()
    if actual != expected:
        return (
            False,
            f"{label} sha256 mismatch: expected={expected} actual={actual} ({src_path})",
        )
    return True, actual


def main(argv: list[str] | None = None) -> int:
    _record_pid()
    ap = argparse.ArgumentParser(description="Offline replay benchmark for weaveprof profile.")
    ap.add_argument(
        "--recorded-dir",
        default=os.environ.get("WEAVEPROF_REPLAY_RECORDED_DIR"),
        help="Directory containing recorded runtime/SASS/capture artifacts and *.sha256 digests",
    )
    ap.add_argument(
        "--runtime-override-json",
        default=os.environ.get("WEAVEPROF_REPLAY_RUNTIME_OVERRIDES"),
        help="Optional JSON object merged into staged runtime.json (e.g. require_product_gate)",
    )
    args = ap.parse_args(argv)

    if not args.recorded_dir:
        print("wpprofile_replay_bench: --recorded-dir is required", file=sys.stderr)
        return 2
    recorded_dir = Path(args.recorded_dir)
    if not recorded_dir.is_dir():
        print(f"wpprofile_replay_bench: recorded directory not found: {recorded_dir}", file=sys.stderr)
        return 2

    round_raw = os.environ.get("WEAVEPROF_ROUND_DIR")
    if not round_raw:
        print("wpprofile_replay_bench: WEAVEPROF_ROUND_DIR is not set", file=sys.stderr)
        return 2
    round_dir = Path(round_raw)
    if not round_dir.is_dir():
        print(f"wpprofile_replay_bench: round directory not found: {round_dir}", file=sys.stderr)
        return 2

    checks = (
        ("baseline_source", "WEAVEPROF_BASELINE_SOURCE", recorded_dir / "baseline_source.sha256"),
        ("instrumented_source", "WEAVEPROF_INSTRUMENTED_SOURCE", recorded_dir / "instrumented_source.sha256"),
    )
    errors: list[str] = []
    for label, env_key, sha_path in checks:
        ok, msg = _verify_source_digest(label, env_key, sha_path)
        if not ok:
            errors.append(msg)
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 2

    for item in sorted(recorded_dir.iterdir()):
        if item.name in _SHA_FILES or item.name.startswith(".") or not item.is_file():
            continue
        if item.suffix == ".xz":
            dst = round_dir / item.name[:-3]
            with lzma.open(item, "rb") as fi, dst.open("wb") as fo:
                shutil.copyfileobj(fi, fo, 1 << 20)
        else:
            shutil.copy2(item, round_dir / item.name)

    if args.runtime_override_json:
        overrides = json.loads(args.runtime_override_json)
        if not isinstance(overrides, dict):
            print("wpprofile_replay_bench: --runtime-override-json must decode to an object", file=sys.stderr)
            return 2
        rt_path = round_dir / "runtime.json"
        rt_doc = json.loads(rt_path.read_text(encoding="utf-8")) if rt_path.is_file() else {}
        rt_doc.update(overrides)
        rt_path.write_text(json.dumps(rt_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
