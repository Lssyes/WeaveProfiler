#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最终交付门禁回归：C2 全量输出、C4 对齐闭合、最终 trace、日志和 AMD 探针。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
CHECK = PKG / "weaveprof/pipeline/validate" / "tools" / "fi_check.py"
LOGCHECK = PKG / "weaveprof/pipeline/validate" / "tools" / "fi_logcheck.py"
AMD_PROFILES = HERE / "cases" / "amd_mi308x_resource_profiles.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


fi_check = load_module("fi_check_delivery_gate_test", CHECK)
fi_logcheck = load_module("fi_logcheck_delivery_gate_test", LOGCHECK)
amd_fixtures = load_module("amd_resource_profiles_test", HERE / "amd_c1_regression.py")
AMD_PROFILE_DOC = amd_fixtures.load_profiles(AMD_PROFILES)


class C2GateTest(unittest.TestCase):
    def setUp(self):
        self.tmp_obj = tempfile.TemporaryDirectory(prefix="fi_c2_gate_")
        self.tmp = Path(self.tmp_obj.name)

    def tearDown(self):
        self.tmp_obj.cleanup()

    def args(self, name="compare.json", **extra):
        values = dict(compare_json=str(self.tmp / name), compare_cmd=None, compare_timeout=10,
                      compare_out=None, max_abs=0.0, max_rel=0.0, expected_elements=None,
                      compare_rc=0, require_script_pass=True)
        values.update(extra)
        return argparse.Namespace(**values)

    def write(self, name, doc):
        path = self.tmp / name
        path.write_text(json.dumps(doc), encoding="utf-8")
        return path

    def strict_doc(self, elements=100, checked=100, **extra):
        doc = {
            "pass": True,
            "checked_elements": checked,
            "total_elements": elements,
            "max_abs": 0.0,
            "max_rel": 0.0,
            "nonfinite": 0,
            "keys": ["out"],
            "key_sets_exact": True,
            "all_outputs_compared": True,
            "output_manifest": {
                "selection": "all_outputs",
                "key_sets_exact": True,
                "keys": ["out"],
                "outputs": [{"name": "out", "shape": [10, 10], "dtype": "float32",
                             "elements": elements, "checked_elements": checked,
                             "sha256": "0" * 64}],
            },
        }
        doc.update(extra)
        return doc

    def test_missing_pass_or_outputs_cannot_pass(self):
        self.write("compare.json", {"checked_elements": 1, "max_abs": 0.0,
                                    "max_rel": 0.0, "nonfinite": 0})
        result = fi_check.check_c2(self.args())
        self.assertFalse(result["pass"])
        text = " ".join(result["suggestions"])
        self.assertIn("outputs", text)
        self.assertIn("pass=true", text)

        self.write("compare.json", self.strict_doc(**{"pass": False}))
        result = fi_check.check_c2(self.args())
        self.assertFalse(result["pass"])

    def test_script_rc_must_be_zero_even_with_valid_json(self):
        self.write("compare.json", self.strict_doc())
        result = fi_check.check_c2(self.args(compare_rc=3))
        self.assertFalse(result["pass"])
        self.assertEqual(result["metrics"]["script_returncode"], 3)

    def test_output_total_checked_and_expected_must_agree(self):
        self.write("compare.json", self.strict_doc(elements=100, checked=99))
        self.assertFalse(fi_check.check_c2(self.args(expected_elements=100))["pass"])

        self.write("compare.json", self.strict_doc(elements=100, checked=100))
        self.assertFalse(fi_check.check_c2(self.args(expected_elements=101))["pass"])
        self.assertTrue(fi_check.check_c2(self.args(expected_elements=100))["pass"])

    def test_single_element_requires_explicit_expected(self):
        scalar = {
            "pass": True, "checked_elements": 1, "total_elements": 1,
            "max_abs": 0.0, "max_rel": 0.0, "nonfinite": 0,
            "keys": ["scalar"], "key_sets_exact": True, "all_outputs_compared": True,
            "outputs": [{"name": "scalar", "shape": [], "dtype": "float32",
                         "elements": 1, "checked_elements": 1}],
        }
        self.write("compare.json", scalar)
        self.assertFalse(fi_check.check_c2(self.args())["pass"])
        self.assertTrue(fi_check.check_c2(self.args(expected_elements=1))["pass"])


class C4GateTest(unittest.TestCase):
    TIMING_TH = {"max_timer_min_step_ns": 128, "min_clock_mhz": 300,
                 "max_clock_mhz": 5000, "max_clock_spread_pct": 5,
                 "min_end_margin_us": 0, "max_end_margin_us": 50}

    def summary(self):
        return {"timer_min_step_ns": 32.0, "clock_mhz_per_lane": None, "timer": "globaltimer"}

    def strict_align(self, **extra):
        align = {"ok": True, "metrics_ok": True, "axis_check": {"ok": True},
                 "end_margin_us": 2.0, "returncode": 0}
        align.update(extra)
        return {"align": align, "errors": {}}

    def test_strict_align_contract_passes(self):
        result = fi_check.evaluate_timing(self.summary(), self.strict_align(), self.TIMING_TH)
        self.assertTrue(result["pass"], result)
        self.assertTrue(result["align_ok"])
        self.assertTrue(result["metrics_ok"])
        self.assertTrue(result["axis_check_ok"])
        self.assertTrue(result["align_returncode_ok"])

    def test_each_align_failure_is_closed(self):
        cases = [
            self.strict_align(ok=False),
            self.strict_align(metrics_ok=False),
            self.strict_align(axis_check={"ok": False}),
            self.strict_align(returncode=4),
            {"align": {"ok": True, "axis_check": {"ok": True}, "end_margin_us": 2.0,
                       "returncode": 0}, "errors": {}},
        ]
        bad_errors = self.strict_align()
        bad_errors["errors"] = {"decode": 1}
        cases.append(bad_errors)
        for index, align in enumerate(cases):
            with self.subTest(index=index):
                self.assertFalse(fi_check.evaluate_timing(self.summary(), align, self.TIMING_TH)["pass"])

    def test_missing_align_rc_fails_closed(self):
        align = self.strict_align()
        align["align"].pop("returncode")
        self.assertFalse(fi_check.evaluate_timing(self.summary(), align, self.TIMING_TH)["pass"])

    def test_missing_align_errors_field_fails_closed(self):
        align = self.strict_align()
        align.pop("errors")
        self.assertFalse(fi_check.evaluate_timing(self.summary(), align, self.TIMING_TH)["pass"])

    def test_null_align_errors_is_not_an_empty_collection(self):
        align = self.strict_align()
        align["errors"] = None
        self.assertFalse(fi_check.evaluate_timing(self.summary(), align, self.TIMING_TH)["pass"])


class ArtifactLogGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp_obj = tempfile.TemporaryDirectory(prefix="fi_artifact_gate_")
        self.tmp = Path(self.tmp_obj.name)

    def tearDown(self):
        self.tmp_obj.cleanup()

    def write_trace(self, events, compressed=False):
        path = self.tmp / ("trace.perfetto.json.gz" if compressed else "trace.perfetto.json")
        raw = json.dumps({"traceEvents": events, "otherData": {"schema_version": "fi-perfetto"}}).encode()
        if compressed:
            with gzip.open(path, "wb") as f:
                f.write(raw)
        else:
            path.write_bytes(raw)
        return path

    def test_trace_sha_and_business_x_are_required(self):
        info_only = [{"ph": "X", "name": "软硬件信息", "cat": "info", "pid": 1, "tid": 0, "ts": 0, "dur": 1}]
        path = self.write_trace(info_only, compressed=True)
        result = fi_check.check_trace_artifact(argparse.Namespace(trace=str(path), expected_trace_sha256=None))
        self.assertFalse(result["pass"])

        events = info_only + [
            {"ph": "X", "name": "mma", "cat": "计算（mma）", "pid": 2, "tid": 3, "ts": 0.0, "dur": 4.0},
            {"ph": "X", "name": "kernel", "cat": "nsys_kernel", "pid": 3, "tid": 0, "ts": 0, "dur": 4},
        ]
        path = self.write_trace(events, compressed=True)
        good = fi_check.check_trace_artifact(argparse.Namespace(trace=str(path), expected_trace_sha256=None))
        self.assertTrue(good["pass"], good)
        self.assertEqual(good["metrics"]["business_x_events"], 1)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(good["metrics"]["sha256"], actual)

        bad = fi_check.check_trace_artifact(argparse.Namespace(trace=str(path), expected_trace_sha256="f" * 64))
        self.assertFalse(bad["pass"])

    def log_args(self, log_dir, exec_rc=None):
        return argparse.Namespace(
            logs=[str(log_dir)], logcheck_script=str(LOGCHECK), logcheck_max=0,
            log_allow=[], strict_glog=False, log_exec_rc=exec_rc, logcheck_timeout=60,
            work_dir=str(self.tmp / "evidence"), output=str(self.tmp / "check.json"))

    def test_logs_and_excp_gz_are_bound_into_gate(self):
        logs = self.tmp / "logs"
        logs.mkdir()
        (logs / "clean.log").write_text("kernel duration 12 us\n", encoding="utf-8")
        with gzip.open(logs / "crash.excp.gz", "wt", encoding="utf-8") as f:
            f.write("triton.runtime.errors.OutOfResources: shared memory\n")
        result = fi_check.check_logs(self.log_args(logs))
        self.assertFalse(result["pass"])
        self.assertEqual(result["evidence"]["summary"]["rc"], 1)
        self.assertTrue(any(str(x["path"]).endswith(".excp.gz") for x in result["evidence"]["files"]))

        (logs / "crash.excp.gz").unlink()
        result = fi_check.check_logs(self.log_args(logs))
        self.assertTrue(result["pass"], result)
        self.assertTrue(result["metrics"]["clean"])

        bad_rc = fi_check.check_logs(self.log_args(logs, exec_rc=9))
        self.assertFalse(bad_rc["pass"])

    def test_overall_summary_includes_extra_checks(self):
        checks = {name: {"pass": True} for name in ("C1", "C2", "C3", "C4", "Artifacts", "Logs")}
        counts, overall = fi_check.summarize_checks(checks)
        self.assertTrue(overall)
        self.assertEqual(counts, {"pass": 6, "fail": 0, "total": 6})
        checks["Logs"] = {"pass": False}
        counts, overall = fi_check.summarize_checks(checks)
        self.assertFalse(overall)
        self.assertEqual(counts["total"], 6)


class AmdProbeGateTest(unittest.TestCase):
    def test_missing_probe_fails_unless_explicitly_waived(self):
        with tempfile.TemporaryDirectory(prefix="fi_amd_probe_gate_") as td:
            td_path = Path(td)
            l0_res, fi_res = amd_fixtures.materialize_resource_pair(
                AMD_PROFILE_DOC, "demo", "correct_instrumentation", td_path / "resources")
            args = argparse.Namespace(
                platform="amd", arch="gfx942", amd_gfx="gfx942", rocm_llvm=None,
                l0=None, fi=None,
                l0_amd_res=str(l0_res),
                fi_amd_res=str(fi_res),
                kernel="_Z4demoPKfPfPjiii", work_dir=td,
                max_scratch_delta_bytes=256, min_occupancy_ratio=0.5,
                max_reg_delta_pct=10.0, max_reg_delta_abs=8,
                strict_resources=False, min_sim_strict=0.60, min_sim_opcode=0.75,
                min_sim_norm=0.60, min_key_lcs=0.80, name="probe_gate",
                allow_unverified_probe=False)
            strict = fi_check.check_c1_amd(args, None)
            self.assertFalse(strict["pass"])
            self.assertTrue(any("探针" in x for x in strict["metrics"]["fails"]))
            self.assertTrue(strict["thresholds"]["require_fi_probe"])

            args.allow_unverified_probe = True
            waived = fi_check.check_c1_amd(args, None)
            self.assertTrue(waived["pass"], waived["metrics"]["fails"])
            self.assertFalse(waived["thresholds"]["require_fi_probe"])


class EndToEndGateTest(unittest.TestCase):
    def test_six_checks_and_artifact_binding_close_together(self):
        with tempfile.TemporaryDirectory(prefix="fi_delivery_e2e_") as td_text:
            td = Path(td_text)
            decoder = fi_check._load_decoder()
            names = td / "names.json"
            names.write_text(json.dumps({"events": {"1": "mma"}, "roles": {"0": "lane"}}), encoding="utf-8")
            dump = td / "run.bin"
            fi_check._mk_dump(dump, {0: [(1000, decoder.OPEN, 3), (1100, 1, 0),
                                         (1200, decoder.END | 1, 0), (1300, decoder.CLOSE, 0)]})
            compare = td / "compare.json"
            compare.write_text(json.dumps({
                "pass": True, "checked_elements": 100, "total_elements": 100,
                "max_abs": 0.0, "max_rel": 0.0, "nonfinite": 0,
                "keys": ["out"], "key_sets_exact": True, "all_outputs_compared": True,
                "outputs": [{"name": "out", "shape": [10, 10], "dtype": "float32",
                             "elements": 100, "checked_elements": 100}],
            }), encoding="utf-8")
            align = td / "align.summary.json"
            align.write_text(json.dumps({
                "errors": {},
                "align": {"ok": True, "metrics_ok": True, "axis_check": {"ok": True},
                          "end_margin_us": 1.0, "returncode": 0},
            }), encoding="utf-8")
            trace = td / "final.perfetto.json.gz"
            with gzip.open(trace, "wb") as f:
                f.write(json.dumps({
                    "otherData": {"schema_version": "fi-perfetto"},
                    "traceEvents": [{"ph": "X", "name": "mma", "cat": "compute",
                                     "pid": 2, "tid": 3, "ts": 0, "dur": 4}],
                }).encode())
            logs = td / "logs"
            logs.mkdir()
            (logs / "clean.log").write_text("kernel duration 10 us\n", encoding="utf-8")
            l0_res, fi_res = amd_fixtures.materialize_resource_pair(
                AMD_PROFILE_DOC, "demo", "correct_instrumentation", td / "resources")
            out = td / "check.json"
            argv = [
                sys.executable, "-I", str(CHECK),
                "--platform", "amd", "--arch", "gfx942", "--name", "integration",
                "--kernel", "_Z4demoPKfPfPjiii",
                "--l0-amd-res", str(l0_res),
                "--fi-amd-res", str(fi_res),
                "--allow-unverified-probe",
                "--dump", str(dump), "--names", str(names),
                "--compare-json", str(compare), "--compare-rc", "0", "--expected-elements", "100",
                "--align-summary", str(align), "--align-rc", "0",
                "--trace", str(trace), "--logs", str(logs),
                "--work-dir", str(td / "evidence"), "-o", str(out),
            ]
            proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(result["overall_pass"], result["counts"])
            self.assertEqual(result["counts"], {"pass": 6, "fail": 0, "total": 6})
            self.assertEqual(result["artifacts"]["trace"]["sha256"],
                             hashlib.sha256(trace.read_bytes()).hexdigest())
            self.assertEqual(result["artifacts"]["align_summary"]["sha256"],
                             hashlib.sha256(align.read_bytes()).hexdigest())
            self.assertEqual(len(result["artifact_binding_sha256"]), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
