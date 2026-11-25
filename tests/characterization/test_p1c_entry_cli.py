#!/usr/bin/env python3
"""Characterization tests pinning the CLI entrypoints (``-m weaveprof`` & ``run.py``) (P1c R22).

Covers ``causal``, ``seal-check``, and ``render`` subcommands across both ``python -B -m weaveprof``
and ``python -B run.py`` against identical concrete literal expectations:
  * ``causal`` on ``h20_pingpong_v1``'s ``trace.perfetto.json.gz`` with ``--arch sm_90a --sms 78``.
  * ``seal-check`` on ``h20_pingpong_v1`` (PASS seal, tampered result field, tampered bound source
    artifact) and ``h20_v6_persistent`` (FAIL seal without and with ``--allow-fail``).
  * ``render`` producing a valid PNG and rejecting a hardware-stripped trace without ``--allow-no-hw``.

Run:  python -B -m unittest tests/characterization/test_p1c_entry_cli.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import gzip
import json
import lzma
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if sys.path[:1] != [str(REPO_ROOT)]:
    sys.path.insert(0, str(REPO_ROOT))

from weaveprof.megakernel import run_megakernel_audit  # noqa: E402

_ENV_KEYS = (
    "HOME",
    "USERPROFILE",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "WEAVEPROF_SESSIONS_DIR",
    "WEAVEPROF_CACHE",
    "WEAVEPROF_LOCK_DIR",
    "TMPDIR",
    "TEMP",
    "TMP",
    "MPLCONFIGDIR",
    "WEAVEPROF_SUPERVISOR_HMAC_KEY",
    "WEAVEPROF_SUPERVISOR_KEY_FILE",
    "QWEN_PROJECT_DIR",
    "QWEN_SESSION_ID",
    "WEAVEPROF_LOCK_PROVIDER",
    "WEAVEPROF_KEEP_SHADOW",
    "WEAVEPROF_GOLDEN_UPDATE",
    "WEAVEPROF_METRICS_UPDATE",
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
)

_EXPECTED_RENDER_NO_HW_LAST_LINE = (
    "ValueError: MISSING_HARDWARE_UTILIZATION_CURVES: Panel (A) requires same-run hardware profiler "
    "utilization curves (200 kHz nsys on NVIDIA or rocprof-sys on AMD). Refusing to emit "
    "an incomplete SW-only PNG without Panel (A). Capture real GPU metrics (e.g., sudo nsys "
    "on Linux, or enable NVIDIA Control Panel Developer GPU Performance Counters on Windows) "
    "or pass require_hw_curves=False / --allow-no-hw explicitly for offline debugging only."
)


def _unxz(src: Path, dst: Path) -> Path:
    with lzma.open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    return dst


def _norm_tmp(text: str, tmp_root: Path) -> str:
    for real in sorted({str(tmp_root), os.path.realpath(str(tmp_root))}, key=lambda s: -len(s)):
        text = text.replace(real, "<TMP>")
    return text


def _norm_sha64(text: str) -> str:
    return re.sub(r"\b[0-9a-f]{64}\b", "<SHA256>", text)


class CliEntrypointTest(unittest.TestCase):
    PID_RECORDS: List[Tuple[int, str, int | None]] = []

    @classmethod
    def setUpClass(cls) -> None:
        cls._saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}
        cls._saved_tempdir = tempfile.tempdir
        cls._root = Path(tempfile.mkdtemp(prefix="wp-p1c-r22-cli-")).resolve()
        for env_k, sub in (
            ("HOME", "home"),
            ("USERPROFILE", "home"),
            ("XDG_CACHE_HOME", "xdg_cache"),
            ("XDG_CONFIG_HOME", "xdg_config"),
            ("WEAVEPROF_SESSIONS_DIR", "sessions"),
            ("WEAVEPROF_CACHE", "wp_cache"),
            ("WEAVEPROF_LOCK_DIR", "locks"),
            ("TMPDIR", "tmp"),
            ("TEMP", "tmp"),
            ("TMP", "tmp"),
            ("MPLCONFIGDIR", "mpl"),
        ):
            d = cls._root / sub
            d.mkdir(parents=True, exist_ok=True)
            os.environ[env_k] = str(d)
        tempfile.tempdir = None
        for clear_k in (
            "WEAVEPROF_SUPERVISOR_KEY_FILE",
            "QWEN_PROJECT_DIR",
            "QWEN_SESSION_ID",
            "WEAVEPROF_LOCK_PROVIDER",
            "WEAVEPROF_KEEP_SHADOW",
            "WEAVEPROF_GOLDEN_UPDATE",
            "WEAVEPROF_METRICS_UPDATE",
        ):
            os.environ.pop(clear_k, None)
        os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = "weaveprof-golden-test-key"
        os.environ["PYTHONPATH"] = str(REPO_ROOT)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        cls._env = dict(os.environ)
        cls._cwd = cls._root / "cwd"
        cls._cwd.mkdir(parents=True, exist_ok=True)

        probe = subprocess.run(
            [sys.executable, "-B", "-c", "import weaveprof; print(weaveprof.__file__)"],
            cwd=str(cls._cwd),
            env=cls._env,
            capture_output=True,
            text=True,
            check=True,
        )
        rel_init = Path(probe.stdout.strip()).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        if rel_init != "weaveprof/__init__.py":
            raise AssertionError(f"weaveprof imported from outside repo under test: {probe.stdout!r}")

        cls._pp_work, cls._pp_out, cls._pp_meta = cls._prepare_case_audit("h20_pingpong_v1")
        cls._v6_work, cls._v6_out, cls._v6_meta = cls._prepare_case_audit("h20_v6_persistent")

    @classmethod
    def _prepare_case_audit(cls, case_name: str) -> Tuple[Path, Path, Dict[str, Any]]:
        case_dir = REPO_ROOT / "tests" / "golden" / "cases" / case_name
        meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        work = cls._root / case_name
        work.mkdir(parents=True, exist_ok=True)
        ws = work / "workspace"
        shutil.copytree(case_dir / "workspace", ws)
        dump = _unxz(case_dir / "capture.fi02.bin.xz", work / "capture.fi02.bin")
        sqlite = _unxz(case_dir / "nsys.sqlite.xz", work / "nsys.sqlite")
        rm = work / "runtime.json"
        shutil.copy2(case_dir / "runtime.json", rm)
        out = work / "out"
        run_megakernel_audit(
            ws,
            entry_source=meta["entry_source"],
            dump_path=dump,
            names=case_dir / "names.json",
            runtime_metrics=rm,
            nsys_sqlite=sqlite,
            card=meta["card"],
            arch=meta["arch"],
            total_sms=meta["total_sms"],
            limit=meta["limit"],
            out_dir=out,
            overview_title=meta.get("overview_title") or None,
            render_overview_png=True,
            write_perfetto_gz=True,
        )
        return work, out, meta

    @classmethod
    def tearDownClass(cls) -> None:
        tempfile.tempdir = cls._saved_tempdir
        for k, v in cls._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(cls._root, ignore_errors=True)

    def _invoke_cli(self, entry: str, sub_argv: Sequence[str], label: str) -> Tuple[int, str, str]:
        if entry == "module":
            argv = [sys.executable, "-B", "-m", "weaveprof", *sub_argv]
        elif entry == "run_py":
            argv = [sys.executable, "-B", str(REPO_ROOT / "run.py"), *sub_argv]
        else:
            raise ValueError(f"Unknown entry {entry!r}")
        proc = subprocess.Popen(
            argv,
            cwd=str(self._cwd),
            env=self._env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate(timeout=120.0)
        poll_rc = proc.poll()
        self.__class__.PID_RECORDS.append((proc.pid, f"{entry}:{label}", poll_rc))
        self.assertIsNotNone(poll_rc)
        return proc.returncode, stdout, stderr

    def _assert_pingpong_causal_literals(self, rc: int, stdout: str, stderr: str) -> None:
        self.assertEqual(rc, 0, f"causal exited {rc}: {stderr}")
        doc = json.loads(stdout)
        self.assertEqual(doc["causal_opportunities"], [])
        self.assertEqual(doc["schema"], "weaveprof.causal-report/v1")
        self.assertEqual(doc["target"], "perfetto_trace")
        self.assertEqual(doc["arch"], "sm_90a")
        self.assertEqual(doc["gpu"], "")
        self.assertEqual(doc["total_sms"], 78)
        self.assertEqual(doc["global_hw_regime"], "TENSOR_SATURATED")
        self.assertEqual(doc["verdict"], "SATURATED_STOP_REFINING")
        self.assertEqual(doc["tensor_calibration_factor"], 1.0)
        self.assertEqual(
            doc["caliber"],
            {
                "roofline_axis": "TENSOR",
                "nom_rf_pct": 89.5,
                "emp_rf_pct": 93.23,
                "effective_mfu_pct": None,
                "padding_waste_pct": None,
            },
        )
        self.assertEqual(
            doc["hw_summary"],
            {
                "sm_active_mean_pct": 93.52,
                "tensor_active_cal_mean_pct": 89.5,
                "tensor_active_cal_peak_pct": 100.0,
                "dram_read_mean_pct": 5.11,
                "dram_read_peak_pct": 13.0,
                "dram_write_mean_pct": 1.93,
                "warp_issue_mean_pct": 11.98,
            },
        )
        self.assertEqual(doc["sw_span_us"], 0.0)
        self.assertEqual(doc["kernel_us"], 0.0)
        self.assertEqual(doc["observed_sms"], 0)
        self.assertEqual(doc["observed_ctas"], 0)
        self.assertEqual(doc["structural_blockers"], [])
        self.assertEqual(
            doc["bottleneck_hint"],
            "Kernel is TENSOR_SATURATED (Tensor=89.5%, DRAM_Rd=5.1%, SM=93.5%).",
        )

    def test_causal_via_module_and_run_py(self) -> None:
        trace_gz = str(self._pp_out / "trace.perfetto.json.gz")
        cli_args = ["causal", trace_gz, "--arch", str(self._pp_meta["arch"]), "--sms", str(self._pp_meta["total_sms"])]

        rc_mod, out_mod, err_mod = self._invoke_cli("module", cli_args, "causal_pingpong")
        self._assert_pingpong_causal_literals(rc_mod, out_mod, err_mod)

        rc_rpy, out_rpy, err_rpy = self._invoke_cli("run_py", cli_args, "causal_pingpong")
        self._assert_pingpong_causal_literals(rc_rpy, out_rpy, err_rpy)

        # Also pass non-default --arch / --sms overrides so forwarding does not coincide with trace otherData defaults
        override_args = ["causal", trace_gz, "--arch", "sm_103a", "--sms", "120"]
        for entry in ("module", "run_py"):
            with self.subTest(case="override_arch_sms", entry=entry):
                rc_ov, out_ov, err_ov = self._invoke_cli(entry, override_args, "causal_override")
                self.assertEqual(rc_ov, 0, err_ov)
                doc_ov = json.loads(out_ov)
                self.assertEqual(doc_ov["schema"], "weaveprof.causal-report/v1")
                self.assertEqual(doc_ov["arch"], "sm_103a")
                self.assertEqual(doc_ov["total_sms"], 120)
                self.assertEqual(doc_ov["global_hw_regime"], "TENSOR_SATURATED")
                self.assertEqual(doc_ov["verdict"], "SATURATED_STOP_REFINING")
                self.assertEqual(doc_ov["caliber"]["emp_rf_pct"], 95.21)
                self.assertEqual(doc_ov["causal_opportunities"], [])

    def test_seal_check_pass_and_tamper_cases_via_module_and_run_py(self) -> None:
        seal_pp = self._pp_out / "seal.json"

        # 1. Valid PASS seal without --allow-fail (both -m weaveprof and run.py)
        for entry in ("module", "run_py"):
            with self.subTest(case="pass_valid", entry=entry):
                rc, out, err = self._invoke_cli(entry, ["seal-check", str(seal_pp)], "seal_pass")
                self.assertEqual(rc, 0, err)
                self.assertEqual(json.loads(out), {"ok": True, "errors": []})

        # 2. Tamper 1: modify result field status="FAIL" inside seal JSON
        tampered_seal = self._root / "tampered_status.seal.json"
        sdoc = json.loads(seal_pp.read_text(encoding="utf-8"))
        sdoc["status"] = "FAIL"
        tampered_seal.write_text(json.dumps(sdoc), encoding="utf-8")
        for entry in ("module", "run_py"):
            with self.subTest(case="tamper_result_field", entry=entry):
                rc, out, _err = self._invoke_cli(entry, ["seal-check", str(tampered_seal)], "seal_tamper_field")
                self.assertEqual(rc, 1)
                parsed = json.loads(out)
                raw_hashes = re.findall(r"\b[0-9a-f]{64}\b", parsed["errors"][0])
                self.assertEqual(len(raw_hashes), 2)
                self.assertNotEqual(raw_hashes[0], raw_hashes[1])
                norm_doc = {"ok": parsed["ok"], "errors": [_norm_sha64(e) for e in parsed["errors"]]}
                self.assertEqual(
                    norm_doc,
                    {
                        "ok": False,
                        "errors": [
                            "Seal self-hash tampered (claimed=<SHA256>, computed=<SHA256>)",
                            "Seal supervisor HMAC mismatch (possible forgery)",
                            "Seal status is 'FAIL', required 'PASS'",
                        ],
                    },
                )

        # 3. Tamper 2: modify bound source artifact bytes on disk
        bound_src = self._pp_work / "workspace" / "matmul-pingpong-v1.cu"
        orig_bytes = bound_src.read_bytes()
        try:
            bound_src.write_bytes(b"tampered\n")
            for entry in ("module", "run_py"):
                with self.subTest(case="tamper_bound_artifact", entry=entry):
                    rc, out, _err = self._invoke_cli(entry, ["seal-check", str(seal_pp)], "seal_tamper_artifact")
                    self.assertEqual(rc, 1)
                    parsed = json.loads(out)
                    norm_doc = {"ok": parsed["ok"], "errors": [_norm_tmp(e, self._root) for e in parsed["errors"]]}
                    self.assertEqual(
                        norm_doc,
                        {
                            "ok": False,
                            "errors": [
                                "Bound artifact 'source' hash mismatch: "
                                "<TMP>/h20_pingpong_v1/workspace/matmul-pingpong-v1.cu "
                                "(seal=6cea508de081 != live=92e78d0b0329)"
                            ],
                        },
                    )
        finally:
            bound_src.write_bytes(orig_bytes)

    def test_seal_check_fail_case_without_and_with_allow_fail(self) -> None:
        seal_v6 = self._v6_out / "seal.json"

        for entry in ("module", "run_py"):
            with self.subTest(entry=entry, allow_fail=False):
                rc_no, out_no, _err_no = self._invoke_cli(entry, ["seal-check", str(seal_v6)], "seal_v6_strict")
                self.assertEqual(rc_no, 1)
                self.assertEqual(
                    json.loads(out_no),
                    {
                        "ok": False,
                        "errors": [
                            "Seal status is 'FAIL', required 'PASS'",
                            "Required check 'assembly' is not True in seal",
                            "PASS seal must bind required artifact 'trace'",
                        ],
                    },
                )
            with self.subTest(entry=entry, allow_fail=True):
                rc_ok, out_ok, err_ok = self._invoke_cli(
                    entry, ["seal-check", "--allow-fail", str(seal_v6)], "seal_v6_allow_fail"
                )
                self.assertEqual(rc_ok, 0, err_ok)
                self.assertEqual(json.loads(out_ok), {"ok": True, "errors": []})

    def test_render_valid_and_missing_hw_curves_via_module_and_run_py(self) -> None:
        trace_gz = self._pp_out / "trace.perfetto.json.gz"

        out_mod_png = self._root / "rendered_module.png"
        rc_m, stdout_m, stderr_m = self._invoke_cli(
            "module", ["render", str(trace_gz), "-o", str(out_mod_png)], "render_valid"
        )
        self.assertEqual(rc_m, 0, stderr_m)
        self.assertEqual(_norm_tmp(stdout_m.strip(), self._root), "<TMP>/rendered_module.png")
        self.assertTrue(out_mod_png.is_file())
        self.assertEqual(out_mod_png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(out_mod_png.stat().st_size, 1024)

        out_rpy_png = self._root / "rendered_runpy.png"
        rc_r, stdout_r, stderr_r = self._invoke_cli(
            "run_py", ["render", str(trace_gz), "-o", str(out_rpy_png)], "render_valid"
        )
        self.assertEqual(rc_r, 0, stderr_r)
        self.assertEqual(_norm_tmp(stdout_r.strip(), self._root), "<TMP>/rendered_runpy.png")
        self.assertTrue(out_rpy_png.is_file())
        self.assertEqual(out_rpy_png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(out_rpy_png.stat().st_size, 1024)

        # Strip hardware counter ("C") and nsys_kernel events; without --allow-no-hw render must fail
        full_trace = json.loads(gzip.decompress(trace_gz.read_bytes()).decode("utf-8"))
        sw_only_trace = dict(full_trace)
        sw_only_trace["traceEvents"] = [
            ev
            for ev in (full_trace.get("traceEvents") or [])
            if ev.get("ph") != "C" and ev.get("cat") != "nsys_kernel"
        ]
        no_hw_json = self._root / "sw_only_trace.perfetto.json"
        no_hw_json.write_text(json.dumps(sw_only_trace), encoding="utf-8")

        for entry in ("module", "run_py"):
            with self.subTest(entry=entry):
                bad_png = self._root / f"should_not_exist_{entry}.png"
                rc_bad, _out_bad, err_bad = self._invoke_cli(
                    entry, ["render", str(no_hw_json), "-o", str(bad_png)], "render_no_hw"
                )
                self.assertEqual(rc_bad, 1)
                err_lines = err_bad.strip().splitlines()
                self.assertEqual(err_lines[0], "Traceback (most recent call last):")
                self.assertEqual(err_lines[-1], _EXPECTED_RENDER_NO_HW_LAST_LINE)
                self.assertFalse(bad_png.exists())


if __name__ == "__main__":
    unittest.main()
