#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.pipeline.core import fast_path


FAKE_BUILD = r'''
import json
import os
import pathlib
import sys
import time
out = pathlib.Path(sys.argv[1])
trace = pathlib.Path(os.environ["FAKE_TRACE"])
if os.environ.get("FAKE_FAIL") == "1":
    print("injected build failure", flush=True)
    raise SystemExit(7)
out.parent.mkdir(parents=True, exist_ok=True)
start = time.time()
time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
source = pathlib.Path("src/kernel.cu").read_text(encoding="utf-8")
out.write_text(os.environ.get("FI_LEVEL", "?") + "\n" + source, encoding="utf-8")
trace.write_text(json.dumps({"start": start, "end": time.time()}), encoding="utf-8")
print("built", out, flush=True)
'''

FAKE_SMOKE = r'''
import os
import pathlib
import sys
artifact = pathlib.Path(os.environ["FAKE_ARTIFACT"])
if not artifact.is_file():
    print("missing artifact", artifact, flush=True)
    raise SystemExit(3)
out = pathlib.Path(os.environ["FAKE_SMOKE_OUT"])
out.write_text("smoke-ok:" + artifact.read_text(encoding="utf-8")[:16], encoding="utf-8")
print("smoke ok", flush=True)
'''


class FastPathTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory(prefix="weaveprofiler_fast_")
        self.root = Path(self.td.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "kernel.cu").write_text("kernel-v1\n", encoding="utf-8")
        (self.root / "fake_build.py").write_text(FAKE_BUILD, encoding="utf-8")
        (self.root / "fake_smoke.py").write_text(FAKE_SMOKE, encoding="utf-8")

    def tearDown(self):
        self.td.cleanup()

    def job(self, *, fail_build=False, sleep=0.0):
        py = sys.executable
        job = {
            "schema": fast_path.SCHEMA,
            "project_root": ".",
            "target": {"card": "h20", "arch": "sm_90", "language": "cuda_cpp"},
            "source_files": ["src/kernel.cu"],
            "cache_dir": ".weaveprofiler/cache/fast",
            "run_root": ".weaveprofiler/runs",
            "max_parallel": 2,
            "parallel_smoke": False,
            "build": [
                {"name": "off", "argv": [py, "fake_build.py", "build/off/libkernel.so"],
                 "env": {"FI_LEVEL": "0", "FAKE_TRACE": "trace_off.json",
                         "FAKE_SLEEP": str(sleep), "FAKE_FAIL": "1" if fail_build else "0"},
                 "outputs": ["build/off/libkernel.so"]},
                {"name": "full", "argv": [py, "fake_build.py", "build/full/libkernel.so"],
                 "env": {"FI_LEVEL": "2", "FAKE_TRACE": "trace_full.json",
                         "FAKE_SLEEP": str(sleep), "FAKE_FAIL": "0"},
                 "outputs": ["build/full/libkernel.so"]},
            ],
            "smoke": [
                {"name": "off_smoke", "argv": [py, "fake_smoke.py"],
                 "env": {"FAKE_ARTIFACT": "build/off/libkernel.so", "FAKE_SMOKE_OUT": "smoke_off.txt"}},
                {"name": "full_smoke", "argv": [py, "fake_smoke.py"],
                 "env": {"FAKE_ARTIFACT": "build/full/libkernel.so", "FAKE_SMOKE_OUT": "smoke_full.txt"}},
            ],
        }
        return self.root / "fast_job.json", job

    def load(self, job):
        path, payload = job
        path.write_text(json.dumps(payload), encoding="utf-8")
        return fast_path.load_job(path)

    def test_template_has_two_builds_and_no_delivery_claim(self):
        doc = fast_path.template()
        self.assertEqual([b["name"] for b in doc["build"]], ["off", "full"])
        result = {"delivery_evidence": False}
        self.assertFalse(result["delivery_evidence"])

    def test_public_run_py_fast_entry_accepts_job(self):
        path, payload = self.job()
        path.write_text(json.dumps(payload), encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "run.py"), "fast",
             "--job", str(path), "--no-smoke"],
            cwd=str(self.root), env=env, text=True, capture_output=True,
            timeout=120, check=False)
        self.assertEqual(0, completed.returncode,
                         "exit=%s\nstdout:\n%s\nstderr:\n%s" % (
                             completed.returncode, completed.stdout, completed.stderr))
        result = json.loads(completed.stdout)
        self.assertEqual("PASS", result["status"])
        self.assertFalse(result["delivery_evidence"])

    def test_off_full_build_and_cache_hit(self):
        job = self.load(self.job())
        first = fast_path.run_fast(job)
        self.assertEqual(first["status"], "PASS", first)
        self.assertEqual([b["cached"] for b in first["builds"]], [False, False])
        self.assertFalse(first["delivery_evidence"])
        self.assertTrue((self.root / "smoke_off.txt").is_file())
        self.assertTrue((self.root / "smoke_full.txt").is_file())
        off = (self.root / "build/off/libkernel.so").read_text(encoding="utf-8")
        full = (self.root / "build/full/libkernel.so").read_text(encoding="utf-8")
        self.assertTrue(off.startswith("0\n"))
        self.assertTrue(full.startswith("2\n"))

        second = fast_path.run_fast(job)
        self.assertEqual(second["status"], "PASS", second)
        self.assertEqual([b["cached"] for b in second["builds"]], [True, True])
        self.assertEqual(second["cache_key"], first["cache_key"])

    def test_source_change_invalidates_cache(self):
        job = self.load(self.job())
        first = fast_path.run_fast(job)
        self.assertEqual(first["status"], "PASS", first)
        (self.root / "src" / "kernel.cu").write_text("kernel-v2-changed\n", encoding="utf-8")
        second = fast_path.run_fast(job)
        self.assertEqual(second["status"], "PASS", second)
        self.assertNotEqual(second["cache_key"], first["cache_key"])
        self.assertEqual([b["cached"] for b in second["builds"]], [False, False])
        self.assertIn("kernel-v2-changed", (self.root / "build/full/libkernel.so").read_text(encoding="utf-8"))

    def test_builds_overlap_in_time(self):
        job = self.load(self.job(sleep=0.35))
        started = time.monotonic()
        result = fast_path.run_fast(job)
        elapsed = time.monotonic() - started
        self.assertEqual(result["status"], "PASS", result)
        off = json.loads((self.root / "trace_off.json").read_text(encoding="utf-8"))
        full = json.loads((self.root / "trace_full.json").read_text(encoding="utf-8"))
        self.assertLess(off["start"], full["end"])
        self.assertLess(full["start"], off["end"])
        self.assertLess(elapsed, 1.0)

    def test_build_failure_is_fail_closed(self):
        job = self.load(self.job(fail_build=True))
        result = fast_path.run_fast(job)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("build failed", result["error"])
        self.assertEqual([b["cached"] for b in result["builds"]], [False, False])

    def test_shell_and_skip_tokens_are_rejected(self):
        path, payload = self.job()
        payload["build"][0]["argv"] += ["--skip-correctness"]
        with self.assertRaises(fast_path.FastPathError):
            self.load((path, payload))

    def test_missing_source_is_rejected(self):
        path, payload = self.job()
        payload["source_files"] = ["src/missing.cu"]
        with self.assertRaises(fast_path.FastPathError):
            self.load((path, payload))

    def test_smoke_failure_is_reported(self):
        path, payload = self.job()
        payload["smoke"][0]["argv"] = [sys.executable, "-c", "raise SystemExit(9)"]
        result = fast_path.run_fast(self.load((path, payload)))
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("smoke failed", result["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
