#!/usr/bin/env python3
"""GPU known-answer smoke test for the CUDA recorder (recorders/cuda_cpp/wp_lite.cuh) + decoder.

Needs nvcc and one NVIDIA GPU; skipped unless WEAVEPROF_GPU_SMOKE=1. On a shared machine, run it under the
host's GPU lock, e.g. on mail-19:

    WEAVEPROF_GPU_SMOKE=1 bash /home/hengfeng.hzy/agent_mail/tools/gpu_lock/gpulock_run.sh -n 1 -t 600 -- \
        python3 -B -m unittest tests/gpu/test_recorder_smoke.py

Checks (one smoke kernel, see smoke_recorder.cu; every warp of every CTA runs ITERS x [2 us A, 10 us B]):
  1. Timing: decoded spans match the busy-wait lengths within the tolerance.
  2. Completeness: exact span count, every lane recorded with OPEN and CLOSE, no decoder errors, no overflow.
  3. Context and timer invariance: every build decodes to the same per-lane event sequence (ids, begin/end, aux).
  4. Overflow: a cap that is too small sets the dump overflow flag and the decoder reports it.

Known recorder bugs at baseline (ae39cc4), written as expected failures so the suite stays green and starts
failing ("unexpected success") the day they are fixed:
  - WP_CTX_SMEM: WP_KERNEL_SCOPE never writes the CLOSE record, so WP_TIMER=1 dumps cannot be decoded.
  - WP_TIMER=1 + lane overflow: the CLOSE anchor is dropped with the overflowing records and read_dump exits
    instead of reporting the overflow.
Environment: WEAVEPROF_NVCC_ARCH (default "-gencode arch=compute_90a,code=sm_90a").
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
import shutil
import statistics
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENABLED = os.environ.get("WEAVEPROF_GPU_SMOKE") == "1"
ARCH = os.environ.get("WEAVEPROF_NVCC_ARCH", "-gencode arch=compute_90a,code=sm_90a").split()
ITERS, A_NS, B_NS = 20, 2_000, 10_000
CAP_OK = 4 * ITERS + 16
NAMES = {"events": {"1": "load", "2": "mma"}, "roles": {"0": "role_even", "1": "role_odd"}}
VARIANTS = {
    "default_t0": ["-DWP_TIMER=0"],
    "default_t1": ["-DWP_TIMER=1"],
    "ctx_reg_t0": ["-DWP_TIMER=0", "-DWP_CTX_REG"],
    "ctx_reg_t1": ["-DWP_TIMER=1", "-DWP_CTX_REG"],
    "ctx_smem_t0": ["-DWP_TIMER=0", "-DWP_CTX_SMEM"],
    "ctx_smem_t1": ["-DWP_TIMER=1", "-DWP_CTX_SMEM"],
}
# Builds that work correctly at baseline; the WP_CTX_SMEM ones are covered by the known-bug tests below.
WORKING = ("default_t0", "default_t1", "ctx_reg_t0", "ctx_reg_t1")


@unittest.skipUnless(ENABLED, "set WEAVEPROF_GPU_SMOKE=1 on a machine with nvcc and an NVIDIA GPU")
class RecorderSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("nvcc"):
            raise unittest.SkipTest("nvcc not on PATH")
        from weaveprof.megakernel import load_wp_decode
        cls.wp = load_wp_decode()
        cls.tmp = Path(tempfile.mkdtemp(prefix="wp-gpu-smoke-"))
        cls.exe = {}
        cls.results = {}
        for name, flags in VARIANTS.items():
            exe = cls.tmp / f"smoke_{name}"
            cmd = ["nvcc", "-O3", "-std=c++17", *ARCH, "-DWP_LEVEL=2", *flags,
                   "-I", str(ROOT / "recorders" / "cuda_cpp"), str(HERE / "smoke_recorder.cu"), "-o", str(exe)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            cls.exe[name] = exe
            cls.results[name] = cls._run(exe, cap=CAP_OK)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def _run(cls, exe: Path, cap: int):
        """Run once; decoding errors are captured (``error``) instead of aborting the whole class."""
        dump = exe.with_suffix(f".cap{cap}.bin")
        out = subprocess.run([str(exe), str(dump), str(cap), str(ITERS), str(A_NS), str(B_NS)],
                             check=True, capture_output=True, text=True, timeout=120)
        res = {"info": json.loads(out.stdout.strip().splitlines()[-1]), "error": None}
        try:
            res["dump"] = cls.wp.read_dump(str(dump))
            res["trace"], res["summary"] = cls.wp.decode(res["dump"], NAMES)
        except SystemExit as exc:  # the decoder reports fatal problems with SystemExit
            res["error"] = str(exc)
        return res

    def _ok(self, name):
        res = self.results[name]
        self.assertIsNone(res["error"], f"{name}: decoder failed: {res['error']}")
        return res

    def _spans(self, res, ev_name):
        return [float(e["dur"]) * 1000.0 for e in res["trace"]["traceEvents"]
                if e.get("ph") == "X" and str((e.get("args") or {}).get("original_label") or e.get("name")) == ev_name]

    def _anchors(self, res):
        return [(sum(1 for r in recs if r[1] & self.wp.OPEN), sum(1 for r in recs if r[1] & self.wp.CLOSE))
                for _, recs in sorted(res["dump"]["slots"].items())]

    def test_timing_matches_busy_wait(self):
        for name in WORKING:
            res = self._ok(name)
            for ev, target in (("load", A_NS), ("mma", B_NS)):
                with self.subTest(variant=name, event=ev):
                    durs = self._spans(res, ev)
                    self.assertTrue(durs, f"no {ev} spans")
                    med = statistics.median(durs)
                    # never shorter than the spin (minus one 32 ns globaltimer step); median within 3% / 250 ns
                    self.assertGreaterEqual(min(durs), target - 64.0, (name, ev, min(durs)))
                    self.assertLessEqual(abs(med - target), max(0.03 * target, 250.0), (name, ev, med))

    def test_complete_and_error_free(self):
        for name in WORKING:
            with self.subTest(variant=name):
                res = self._ok(name)
                info, s = res["info"], res["summary"]
                lanes = info["ctas"] * info["warps_per_cta"]
                self.assertEqual(info["overflow"], 0)
                self.assertEqual(s.get("errors") or {}, {})
                self.assertEqual(s["spans"], lanes * ITERS * 2)
                self.assertEqual(s["lanes"], lanes)
                self.assertEqual(set(self._anchors(res)), {(1, 1)}, "every lane needs exactly one OPEN and CLOSE")

    def test_sequence_independent_of_context_and_timer(self):
        skip = self.wp.OPEN | self.wp.CLOSE | self.wp.CLK

        def seq(res):
            return {lane: [(tag, aux) for _, tag, aux in recs if not tag & skip]
                    for lane, recs in sorted(res["dump"]["slots"].items())}

        ref = seq(self._ok("default_t0"))
        for name in (*WORKING, "ctx_smem_t0"):
            with self.subTest(variant=name):
                self.assertEqual(seq(self._ok(name)), ref, f"{name} decodes a different event sequence")

    def test_overflow_is_reported_timer0(self):
        res = self._run(self.exe["default_t0"], cap=8)
        self.assertIsNone(res["error"])
        self.assertNotEqual(res["info"]["overflow"], 0)
        self.assertTrue(res["summary"].get("errors"), "decoder must report the overflow")

    @unittest.expectedFailure
    def test_known_bug_ctx_smem_kernel_scope_writes_close(self):
        for name in ("ctx_smem_t0", "ctx_smem_t1"):
            res = self.results[name]
            self.assertIsNone(res["error"], f"{name}: {res['error']}")
            self.assertEqual(set(self._anchors(res)), {(1, 1)}, f"{name}: WP_KERNEL_SCOPE must write CLOSE")

    @unittest.expectedFailure
    def test_known_bug_overflow_with_clock64_is_reported_not_fatal(self):
        res = self._run(self.exe["default_t1"], cap=8)
        self.assertIsNone(res["error"], f"decoder aborted instead of reporting the overflow: {res['error']}")
        self.assertTrue(res["summary"].get("errors"))


if __name__ == "__main__":
    unittest.main()
