#!/usr/bin/env python3
"""Golden end-to-end tests: replay real H20 captures through run_megakernel_audit (Workflow B).

Each case under tests/golden/cases/ is a compact fixture built by make_fixture.py from a real audit
(dump + trimmed nsys SQLite + runtime.json + names.json + clean source). The test re-runs the audit and
compares a normalized view of every output against cases/<case>/expected.json:

    report      the compact report returned to agents
    verify      verify.json (gate decisions)
    feedback    feedback.json (decode summary, skew, causal analysis, gates)
    nsys        nsys_hw.json without the raw series (series are covered by the digest below)
    nsys_series per-curve sample count + sha256 of the values
    trace       sha256 of the canonical merged Perfetto trace + per-phase / per-name event counts
    png         overview.png exists and is a valid PNG (pixels are not compared)

A refactor PR must keep every case identical. A PR that intentionally changes behavior regenerates the
expected files with WEAVEPROF_GOLDEN_UPDATE=1 and explains the diff in the PR description.

Run:  python -B -m unittest tests/golden/test_golden_audit.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import gzip
import hashlib
import json
import lzma
import os
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASES = HERE / "cases"
UPDATE = os.environ.get("WEAVEPROF_GOLDEN_UPDATE") == "1"

# Keys whose values legitimately differ between runs or machines (timing, paths, seal digests over paths).
_VOLATILE = {
    "elapsed_ms", "wall_clock_ms", "elapsed_s", "created_at", "acquired_at", "wait_ms", "generated_at",
    "sqlite", "out_dir", "seal_path", "seal_sha256", "region_lock", "lock_path", "pid", "host",
    "source_dump_sha256",
    "report_bytes",  # size of the compact report, which includes wall_clock_ms
}


def _norm(obj, tmp: str):
    """Drop volatile keys, replace the temp dir with <TMP>, round floats (platform noise)."""
    if isinstance(obj, dict):
        return {k: _norm(v, tmp) for k, v in sorted(obj.items()) if k not in _VOLATILE}
    if isinstance(obj, list):
        return [_norm(v, tmp) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, str) and tmp and tmp in obj:
        return obj.replace(tmp, "<TMP>")
    return obj


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _unxz(src: Path, dst: Path) -> Path:
    with lzma.open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    return dst


def run_case(case_dir: Path, tmp: Path) -> dict:
    """Run the audit for one fixture inside ``tmp`` and return the normalized observation."""
    from weaveprof.megakernel import run_megakernel_audit

    meta = json.loads((case_dir / "case.json").read_text())
    ws = tmp / "workspace"
    shutil.copytree(case_dir / "workspace", ws)
    dump = _unxz(case_dir / "capture.fi02.bin.xz", tmp / "capture.fi02.bin")
    sqlite = _unxz(case_dir / "nsys.sqlite.xz", tmp / "nsys.sqlite")
    rm = tmp / "runtime.json"
    shutil.copy2(case_dir / "runtime.json", rm)
    out = tmp / "out"
    report = run_megakernel_audit(
        ws, entry_source=meta["entry_source"], dump_path=dump, names=case_dir / "names.json",
        runtime_metrics=rm, nsys_sqlite=sqlite, card=meta["card"], arch=meta["arch"],
        total_sms=meta["total_sms"], limit=meta["limit"], out_dir=out,
        overview_title=meta.get("overview_title") or None, render_overview_png=True, write_perfetto_gz=True)

    t = str(tmp)
    obs = {"report": _norm(report, t)}
    for name in ("verify", "feedback"):
        p = out / f"{name}.json"
        obs[name] = _norm(json.loads(p.read_text()), t) if p.is_file() else None
    nsys_p = out / "nsys_hw.json"
    if nsys_p.is_file():
        hw = json.loads(nsys_p.read_text())
        series = hw.pop("hw_series", {}) or {}
        obs["nsys"] = _norm(hw, t)
        obs["nsys_series"] = {
            k: {"n": len(c.get("val") or []), "metric_name": c.get("metric_name"),
                "sha256": _sha(_canon([c.get("t_us"), c.get("val")]))}
            for k, c in sorted(series.items())
        }
    trace_p = out / "trace.perfetto.json.gz"
    if trace_p.is_file():
        tr = json.loads(gzip.decompress(trace_p.read_bytes()).decode("utf-8"))
        events = _norm(tr.get("traceEvents") or [], t)
        by_ph = Counter(str(e.get("ph")) for e in events)
        by_name = Counter(str(e.get("name")) for e in events if e.get("ph") in ("X", "C"))
        obs["trace"] = {
            "events": len(events),
            "sha256": _sha(_canon(events)),
            "other_data_sha256": _sha(_canon(_norm(tr.get("otherData") or {}, t))),
            "by_ph": dict(sorted(by_ph.items())),
            "by_name": dict(sorted(by_name.items())),
        }
    png = out / "overview.png"
    try:
        import matplotlib  # noqa: F401
        has_mpl = True
    except ImportError:
        has_mpl = False
    obs["png"] = {
        "checked": has_mpl,
        "valid": (png.is_file() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and png.stat().st_size > 1024)
        if has_mpl else None,
    }
    return obs


def _first_diff(a, b, path="$"):
    """Human-readable location of the first difference between two JSON-like values."""
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__} ({a!r:.80} vs {b!r:.80})"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: {'missing in expected' if k not in a else 'missing in actual'}"
            d = _first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: expected {a!r:.120} got {b!r:.120}"


class GoldenAuditTest(unittest.TestCase):
    _env_keys = ("WEAVEPROF_SUPERVISOR_HMAC_KEY", "WEAVEPROF_CACHE", "WEAVEPROF_LOCK_DIR",
                 "WEAVEPROF_LOCK_PROVIDER", "WEAVEPROF_KEEP_SHADOW")

    @classmethod
    def setUpClass(cls):
        cls._saved = {k: os.environ.get(k) for k in cls._env_keys}
        cls._root = Path(tempfile.mkdtemp(prefix="wp-golden-"))
        os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = "weaveprof-golden-test-key"  # never touch ~/.cache
        os.environ["WEAVEPROF_CACHE"] = str(cls._root / "cache")
        os.environ["WEAVEPROF_LOCK_DIR"] = str(cls._root / "locks")
        os.environ.pop("WEAVEPROF_LOCK_PROVIDER", None)
        os.environ.pop("WEAVEPROF_KEEP_SHADOW", None)

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(cls._root, ignore_errors=True)

    def _check(self, case: str):
        case_dir = CASES / case
        tmp = self._root / case
        tmp.mkdir()
        obs = run_case(case_dir, tmp)
        exp_p = case_dir / "expected.json"
        if UPDATE or not exp_p.is_file():
            if not UPDATE:
                self.fail(f"{exp_p} missing; generate it with WEAVEPROF_GOLDEN_UPDATE=1")
            exp_p.write_text(json.dumps(obs, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
            return
        expected = json.loads(exp_p.read_text())
        if obs["png"]["checked"] is False:  # environment without matplotlib: do not compare the PNG check
            expected["png"] = obs["png"]
        diff = _first_diff(expected, json.loads(_canon(obs)))
        self.assertIsNone(diff, f"golden mismatch for {case}: {diff}")


def _make_test(case: str):
    def test(self):
        self._check(case)
    test.__name__ = f"test_{case}"
    test.__doc__ = f"golden replay of tests/golden/cases/{case}"
    return test


for _case in sorted(p.name for p in CASES.iterdir() if (p / "case.json").is_file()) if CASES.is_dir() else []:
    setattr(GoldenAuditTest, f"test_{_case}", _make_test(_case))


if __name__ == "__main__":
    unittest.main()
