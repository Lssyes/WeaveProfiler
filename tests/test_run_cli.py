#!/usr/bin/env python3
"""Real subprocess end-to-end tests for the public ``run.py`` CLI."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "run.py"
SEMANTICS = ROOT / "weaveprof/pipeline/core/semantics.py"

TEMPLATE_MODES = ("scan", "instrument", "capture", "validate", "check", "deliver", "gemm-probe")

KERNEL_SOURCE = '''import triton
import triton.language as tl

@triton.jit
def kernel(a_ptr, b_ptr, out_ptr):
    a = tl.load(a_ptr)  # canonical load A
    b = tl.load(b_ptr)  # canonical load B
    acc = tl.dot(a, b)  # canonical mma A
    acc = tl.dot(acc, b)  # canonical mma B
    tl.store(out_ptr, acc)
'''


def _json_documents(text: str) -> List[Any]:
    """Decode every JSON document printed by a command stack."""
    decoder = json.JSONDecoder()
    documents: List[Any] = []
    offset = 0
    while True:
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            return documents
        value, offset = decoder.raw_decode(text, offset)
        documents.append(value)


class RunCliTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="weaveprofiler-run-cli-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.env = os.environ.copy()
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.cache_before = set(ROOT.rglob("__pycache__"))
        self.addCleanup(self._assert_no_bytecode_was_written)

    def _assert_no_bytecode_was_written(self) -> None:
        self.assertEqual(self.cache_before, set(ROOT.rglob("__pycache__")))
        self.assertEqual([], list(self.root.rglob("__pycache__")))

    def run_cli(self, argv: Sequence[str], *, cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", str(RUN), *argv],
            cwd=str(cwd or self.project),
            env=self.env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )

    def run_script(self, script: Path, argv: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", str(script), *argv],
            cwd=str(cwd),
            env=self.env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )

    @staticmethod
    def command_diagnostic(completed: subprocess.CompletedProcess[str]) -> str:
        return "exit={}\nstdout:\n{}\nstderr:\n{}".format(
            completed.returncode, completed.stdout, completed.stderr
        )

    def assert_success(self, completed: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(0, completed.returncode, self.command_diagnostic(completed))

    def test_template_modes_emit_parseable_json(self) -> None:
        for mode in TEMPLATE_MODES:
            with self.subTest(mode=mode):
                completed = self.run_cli(("template", "--mode", mode))
                self.assert_success(completed)
                document = json.loads(completed.stdout)
                self.assertEqual("weaveprofiler-template/v1", document["schema"])
                self.assertEqual(mode, document["mode"])
                self.assertIsInstance(document["argv"], list)
                self.assertGreaterEqual(len(document["argv"]), 4)
                self.assertEqual(mode, document["argv"][3])
                if mode == "gemm-probe":
                    self.assertEqual(
                        "/opt/venv/python3.14.4/bin/python3", document["argv"][0]
                    )

    def test_gemm_probe_dispatches_to_the_packaged_probe(self) -> None:
        completed = self.run_cli(("gemm-probe", "--help"))
        self.assert_success(completed)
        self.assertTrue(completed.stdout.lstrip().startswith("usage: gemm_probe.py"))

    def test_capture_routes_nvidia_and_amd_help_without_starting_gpu_work(self) -> None:
        h20 = self.run_cli(
            (
                "capture",
                "--card",
                "h20",
                "--arch",
                "sm_90",
                "--project",
                str(self.project),
                "--",
                "--help",
            )
        )
        self.assert_success(h20)
        self.assertTrue(h20.stdout.lstrip().startswith("usage: capture_nsys.py"))

        mi308x = self.run_cli(
            (
                "capture",
                "--card",
                "mi308x",
                "--arch",
                "gfx942",
                "--project",
                str(self.project),
                "--",
                "--help",
            )
        )
        self.assert_success(mi308x)
        self.assertTrue(mi308x.stdout.lstrip().startswith("usage: capture_rocm.py"))

    def test_b300_capture_requires_and_accepts_both_exact_architectures(self) -> None:
        missing = self.run_cli(
            ("capture", "--card", "b300", "--project", str(self.project), "--", "--help")
        )
        self.assertEqual(2, missing.returncode, self.command_diagnostic(missing))
        self.assertIn(
            "B300 requires --arch sm_100a or sm_103a", missing.stderr
        )

        for arch in ("sm_100a", "sm_103a"):
            with self.subTest(arch=arch):
                completed = self.run_cli(
                    (
                        "capture",
                        "--card",
                        "b300",
                        "--arch",
                        arch,
                        "--project",
                        str(self.project),
                        "--",
                        "--help",
                    )
                )
                self.assert_success(completed)
                self.assertTrue(
                    completed.stdout.lstrip().startswith("usage: capture_nsys.py")
                )

    def write_kernel(self) -> Path:
        path = self.project / "kernel.py"
        path.write_text(KERNEL_SOURCE, encoding="utf-8")
        return path

    def scan_kernel(self) -> dict[str, Any]:
        completed = self.run_cli(
            (
                "scan",
                "--project",
                ".",
                "--card",
                "h20",
                "--arch",
                "sm_90",
                "--language",
                "triton",
                "--source",
                "kernel.py",
                "--out",
                "scan.json",
            )
        )
        self.assert_success(completed)
        summary = json.loads(completed.stdout)
        self.assertEqual("PASS", summary["status"])
        self.assertEqual("triton", summary["language"])
        self.assertEqual(0, summary["ambiguities"])
        scan = json.loads((self.project / "scan.json").read_text(encoding="utf-8"))
        self.assertEqual("weaveprofiler-scan/v1", scan["schema"])
        self.assertEqual("h20", scan["target"]["card"])
        self.assertEqual("sm_90", scan["target"]["arch"])
        self.assertEqual("triton", scan["language"])
        self.assertEqual([], scan["ambiguities"])
        return scan

    def write_selection(self) -> Path:
        selection = {
            "schema": "weaveprofiler-instrument-selection/v1",
            "version": 1,
            "selections": [
                {
                    "path": "kernel.py",
                    "semantic_id": "load",
                    "begin_anchor": "a = tl.load(a_ptr)  # canonical load A",
                    "end_anchor": "b = tl.load(b_ptr)  # canonical load B",
                    "occurrence": 1,
                    "expect_count": 1,
                    "aux": "tile_11",
                    "event_id": 1,
                },
                {
                    "path": "kernel.py",
                    "semantic_id": "mma",
                    "begin_anchor": "acc = tl.dot(a, b)  # canonical mma A",
                    "end_anchor": "acc = tl.dot(acc, b)  # canonical mma B",
                    "occurrence": 1,
                    "expect_count": 1,
                    "aux": "tile_12",
                    "event_id": 3,
                },
            ],
        }
        path = self.project / "selection.json"
        path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
        return path

    def prepare_generated_plan(self) -> tuple[Path, Path, str]:
        self.write_kernel()
        scan = self.scan_kernel()
        observed = {(item["path"], item["line"], item["semantic_id"]) for item in scan["candidates"]}
        self.assertIn(("kernel.py", 6, "load"), observed)
        self.assertIn(("kernel.py", 7, "load"), observed)
        self.assertIn(("kernel.py", 8, "mma"), observed)
        self.assertIn(("kernel.py", 9, "mma"), observed)

        selection_path = self.write_selection()
        original = (self.project / "kernel.py").read_text(encoding="utf-8")
        completed = self.run_cli(
            (
                "instrument",
                "--scan",
                "scan.json",
                "--selection",
                str(selection_path),
                "--plan",
                "edits.json",
            )
        )
        self.assert_success(completed)
        documents = _json_documents(completed.stdout)
        apply_results = [
            document
            for document in documents
            if isinstance(document, dict)
            and document.get("schema") == "weaveprofiler.apply-plan-result"
        ]
        self.assertEqual(1, len(apply_results), completed.stdout)
        self.assertTrue(apply_results[0]["ok"])
        self.assertEqual("validated", apply_results[0]["status"])
        self.assertTrue(apply_results[0]["check"])
        self.assertFalse(apply_results[0]["writes_performed"])
        self.assertEqual(4, apply_results[0]["files"][0]["edit_count"])

        self.assertEqual(original, (self.project / "kernel.py").read_text(encoding="utf-8"))
        plan_path = self.project / "edits.json"
        names_path = self.project / "edits.json.names.json"
        self.assertTrue(plan_path.is_file())
        self.assertTrue(names_path.is_file())
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual("weaveprofiler.patch-plan", plan["schema"])
        self.assertEqual(1, len(plan["files"]))
        self.assertEqual(4, len(plan["files"][0]["edits"]))

        wrapper = documents[-1]
        self.assertTrue(wrapper["ok"])
        self.assertFalse(wrapper["applied"])
        self.assertTrue(wrapper["dry_run"])
        self.assertEqual(str(names_path.resolve()), wrapper["names"])
        return plan_path, names_path, original

    def test_scan_generate_and_dry_run_check_preserve_canonical_names(self) -> None:
        plan_path, names_path, _original = self.prepare_generated_plan()
        self.assertTrue(plan_path.is_file())

        names = json.loads(names_path.read_text(encoding="utf-8"))
        self.assertEqual({"1": "load", "3": "mma"}, names["events"])
        self.assertEqual(
            {"1": "load", "3": "mma"},
            {event_id: row["semantic_id"] for event_id, row in names["semantics"].items()},
        )

        resolved = self.run_script(
            SEMANTICS, ("--names", str(names_path), "--require"), cwd=self.project
        )
        self.assert_success(resolved)
        contract = json.loads(resolved.stdout)
        self.assertTrue(contract["ok"])
        self.assertEqual({"1", "3"}, set(contract["events"]))
        for event_id, row in contract["events"].items():
            with self.subTest(event_id=event_id):
                self.assertEqual(row["semantic_id"], row["display_name"])
                self.assertEqual(names["events"][event_id], row["display_name"])
                self.assertIn(row["semantic_id"], {"load", "mma"})

    def test_apply_runs_one_real_transaction_and_modifies_each_event_once(self) -> None:
        plan_path, _names_path, _original = self.prepare_generated_plan()
        completed = self.run_cli(
            ("instrument", "--plan", str(plan_path), "--apply")
        )
        self.assert_success(completed)
        documents = _json_documents(completed.stdout)
        apply_results = [
            document
            for document in documents
            if isinstance(document, dict)
            and document.get("schema") == "weaveprofiler.apply-plan-result"
        ]
        self.assertEqual(1, len(apply_results), completed.stdout)
        applied = apply_results[0]
        self.assertTrue(applied["ok"])
        self.assertEqual("applied", applied["status"])
        self.assertFalse(applied["check"])
        self.assertTrue(applied["writes_performed"])
        self.assertEqual(1, len(applied["files"]))

        source_path = self.project / "kernel.py"
        instrumented = source_path.read_text(encoding="utf-8")
        self.assertEqual(
            1,
            instrumented.count(
                "    n = fi_b(rec, ovf, pid, n, FI, ev=1, aux=tile_11)"
            ),
        )
        self.assertEqual(
            1,
            instrumented.count(
                "    n = fi_e(rec, ovf, pid, n, FI, ev=1, aux=tile_11)"
            ),
        )
        self.assertEqual(
            1,
            instrumented.count(
                "    n = fi_b(rec, ovf, pid, n, FI, ev=3, aux=tile_12)"
            ),
        )
        self.assertEqual(
            1,
            instrumented.count(
                "    n = fi_e(rec, ovf, pid, n, FI, ev=3, aux=tile_12)"
            ),
        )
        self.assertEqual(2, instrumented.count("n = fi_b("))
        self.assertEqual(2, instrumented.count("n = fi_e("))
        self.assertEqual(
            applied["files"][0]["result_sha256"],
            hashlib.sha256(source_path.read_bytes()).hexdigest(),
        )

        wrapper = documents[-1]
        self.assertTrue(wrapper["ok"])
        self.assertTrue(wrapper["applied"])
        self.assertFalse(wrapper["dry_run"])

    def test_check_with_missing_run_directory_fails_without_a_receipt(self) -> None:
        missing = self.root / "missing-run"
        check_json = missing / "validation" / "check.json"
        completed = self.run_cli(
            (
                "check",
                "--run-dir",
                str(missing),
                "--json",
                str(check_json),
            )
        )
        self.assertNotEqual(0, completed.returncode, self.command_diagnostic(completed))
        self.assertIn("--run-dir is not a directory", completed.stderr)
        self.assertFalse(missing.exists())
        self.assertFalse(check_json.exists())
        self.assertEqual(
            [],
            [path for path in self.root.rglob("*") if "receipt" in path.name.lower()],
        )

        existing_empty = self.root / "empty-run"
        existing_empty.mkdir()
        empty_check = self.run_cli(("check", "--run-dir", str(existing_empty)))
        self.assertEqual(2, empty_check.returncode, self.command_diagnostic(empty_check))
        self.assertEqual([], list(existing_empty.iterdir()))
        self.assertEqual(
            [],
            [path for path in self.root.rglob("*") if "receipt" in path.name.lower()],
        )


if __name__ == "__main__":
    unittest.main()
