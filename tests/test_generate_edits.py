#!/usr/bin/env python3
"""Contract tests for deterministic instrumentation edit generation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "weaveprof/pipeline/instrument" / "generate_edits.py"
APPLY_PLAN_PATH = ROOT / "weaveprof/pipeline/instrument" / "apply_plan.py"
SPEC = importlib.util.spec_from_file_location(
    "weaveprofiler_generate_edits_tests", GENERATOR_PATH
)
assert SPEC is not None and SPEC.loader is not None
generate_edits_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generate_edits_module
SPEC.loader.exec_module(generate_edits_module)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class GenerateEditsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="weaveprofiler-generate-edits-test-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def write_source(self, relative: str, text: str) -> str:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return sha256(path.read_bytes())

    def write_json(self, relative: str, value: Any) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return path

    def scan(
        self,
        language: str,
        paths: Sequence[str],
        candidates: Sequence[Mapping[str, Any]],
        *,
        ambiguities: Sequence[Mapping[str, Any]] = (),
        include_project_root: bool = True,
        digest_overrides: Mapping[str, str] | None = None,
    ) -> dict:
        files = []
        for relative in paths:
            payload = (self.root / relative).read_bytes()
            digest = (digest_overrides or {}).get(relative, sha256(payload))
            files.append(
                {"path": relative, "bytes": len(payload), "sha256": digest}
            )
        scan = {
            "schema": "weaveprofiler-scan/v1",
            "status": "PASS",
            "language": language,
            "route": {"language": language},
            "files": files,
            "candidates": [dict(item) for item in candidates],
            "ambiguities": [dict(item) for item in ambiguities],
        }
        if include_project_root:
            scan["project_root"] = str(self.root)
        return scan

    @staticmethod
    def selection(*items: Mapping[str, Any]) -> dict:
        return {
            "schema": "weaveprofiler-instrument-selection/v1",
            "version": 1,
            "selections": [dict(item) for item in items],
        }

    def cpp_fixture(self) -> tuple[dict, dict]:
        digest = self.write_source(
            "src/kernel.cu",
            "extern \"C\" __global__ void kernel() {\n"
            "  wmma::mma_sync(c, a, b, c);\n"
            "  wmma::store_matrix_sync(out, c);\n"
            "}\n",
        )
        scan = self.scan(
            "cuda_cpp",
            ["src/kernel.cu"],
            [
                {"path": "src/kernel.cu", "line": 2, "semantic_id": "mma"},
                {"path": "src/kernel.cu", "line": 3, "semantic_id": "mma"},
            ],
        )
        self.assertEqual(digest, scan["files"][0]["sha256"])
        selection = self.selection(
            {
                "path": "src/kernel.cu",
                "semantic_id": "mma",
                "begin_anchor": "wmma::mma_sync(c, a, b, c);",
                "end_anchor": "wmma::store_matrix_sync(out, c);",
                "occurrence": 1,
                "expect_count": 1,
                "aux": "tile_id",
                "event_id": 3,
            }
        )
        return scan, selection

    def triton_fixture(self) -> tuple[dict, dict]:
        digest = self.write_source(
            "src/kernel.py",
            "import triton\n"
            "import triton.language as tl\n"
            "\n"
            "@triton.jit\n"
            "def kernel(a, b, c):\n"
            "    acc = tl.dot(a, b)\n"
            "    tl.store(c, acc)\n",
        )
        scan = self.scan(
            "triton",
            ["src/kernel.py"],
            [
                {"path": "src/kernel.py", "line": 6, "semantic_id": "mma"},
                {"path": "src/kernel.py", "line": 7, "semantic_id": "mma"},
            ],
        )
        self.assertEqual(digest, scan["files"][0]["sha256"])
        selection = self.selection(
            {
                "path": "src/kernel.py",
                "semantic_id": "mma",
                "begin_anchor": "acc = tl.dot(a, b)",
                "end_anchor": "tl.store(c, acc)",
                "occurrence": 1,
                "expect_count": 1,
                "aux": "tile_id",
                "event_id": 3,
            }
        )
        return scan, selection

    def run_cli(
        self,
        scan_path: Path,
        selection_path: Path,
        out_path: Path,
        *extra: str,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-B",
                str(GENERATOR_PATH),
                "--scan",
                str(scan_path),
                "--selection",
                str(selection_path),
                "--out",
                str(out_path),
                *extra,
            ],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

    def test_cpp_generation_is_sha_bound_and_deterministic(self) -> None:
        scan, selection = self.cpp_fixture()
        first = generate_edits_module.generate_edits(scan, selection)
        second = generate_edits_module.generate_edits(scan, selection)

        self.assertEqual(first.plan, second.plan)
        self.assertEqual(
            {"schema", "version", "project_root", "files"},
            set(first.plan),
        )
        file_entry = first.plan["files"][0]
        self.assertEqual("src/kernel.cu", file_entry["path"])
        self.assertEqual(scan["files"][0]["sha256"], file_entry["base_sha256"])
        self.assertEqual(2, len(file_entry["edits"]))
        begin, end = file_entry["edits"]
        self.assertEqual("\nFI_B(3, tile_id);", begin["text"])
        self.assertEqual("after", begin["position"])
        self.assertEqual("FI_E(3, tile_id);\n", end["text"])
        self.assertEqual("before", end["position"])
        self.assertEqual("anchor", begin["indent"])
        self.assertIn("FI_B(3, tile_id);", first.check_diff)
        self.assertEqual(
            "extern \"C\" __global__ void kernel() {\n"
            "  wmma::mma_sync(c, a, b, c);\n"
            "  wmma::store_matrix_sync(out, c);\n"
            "}\n",
            (self.root / "src/kernel.cu").read_text(encoding="utf-8"),
        )

    def test_triton_generation_preserves_python_syntax(self) -> None:
        scan, selection = self.triton_fixture()
        generated = generate_edits_module.generate_edits(scan, selection)
        begin, end = generated.plan["files"][0]["edits"]

        self.assertEqual(
            "\nn = fi_b(rec, ovf, pid, n, FI, ev=3, aux=tile_id)",
            begin["text"],
        )
        self.assertEqual(
            "n = fi_e(rec, ovf, pid, n, FI, ev=3, aux=tile_id)\n",
            end["text"],
        )
        self.assertNotIn(";", begin["text"])
        self.assertNotIn("FI_B(", begin["text"])

    def test_amd_triton_generation_uses_wave64_recorder_signature(self) -> None:
        scan, selection = self.triton_fixture()
        scan["route"]["target"] = {"platform": "amd", "card": "mi308x"}
        scan["route"]["recorder"] = (
            "recorders/triton/fi_triton_amd.py"
        )

        generated = generate_edits_module.generate_edits(scan, selection)
        begin, end = generated.plan["files"][0]["edits"]

        self.assertIn("time_ptr", begin["text"])
        self.assertIn("slot, selected, n", begin["text"])
        self.assertIn("EVENT=3", begin["text"])
        self.assertIn("CAP=FI_CAP", begin["text"])
        self.assertIn("TIMER=FI_TIMER", begin["text"])
        self.assertNotIn("rec, ovf", begin["text"])
        self.assertIn("time_ptr", end["text"])
        self.assertIn("EVENT=3", end["text"])

    def test_names_sidecar_uses_only_canonical_semantic_ids(self) -> None:
        scan, selection = self.cpp_fixture()
        scan_path = self.write_json("scan.json", scan)
        selection_path = self.write_json("selection.json", selection)
        out_path = self.root / "edits.json"

        completed = self.run_cli(scan_path, selection_path, out_path)
        self.assertEqual(0, completed.returncode, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertTrue(result["ok"])
        self.assertEqual([], result["allocations"])

        names_path = Path(str(out_path) + ".names.json")
        self.assertTrue(names_path.is_file())
        names = json.loads(names_path.read_text(encoding="utf-8"))
        self.assertEqual({"3": "mma"}, names["events"])
        self.assertEqual({"3": {"semantic_id": "mma"}}, names["semantics"])
        self.assertNotIn("matrix multiply", names_path.read_text(encoding="utf-8"))

    def test_source_change_after_scan_fails_sha_binding(self) -> None:
        scan, selection = self.cpp_fixture()
        (self.root / "src/kernel.cu").write_text(
            "changed after scan\n", encoding="utf-8"
        )

        with self.assertRaises(generate_edits_module.GenerationError) as caught:
            generate_edits_module.generate_edits(scan, selection)
        self.assertEqual("base_sha256_mismatch", caught.exception.code)

    def test_missing_end_boundary_is_rejected(self) -> None:
        scan, selection = self.cpp_fixture()
        del selection["selections"][0]["end_anchor"]

        with self.assertRaises(generate_edits_module.GenerationError) as caught:
            generate_edits_module.generate_edits(scan, selection)
        self.assertEqual("missing_end_boundary", caught.exception.code)

    def test_noncanonical_and_missing_semantics_are_rejected(self) -> None:
        scan, selection = self.cpp_fixture()
        selection["selections"][0]["semantic_id"] = "matmul"
        with self.assertRaises(generate_edits_module.GenerationError) as caught:
            generate_edits_module.generate_edits(scan, selection)
        self.assertEqual("unknown_semantic", caught.exception.code)

        _, missing = self.cpp_fixture()
        del missing["selections"][0]["semantic_id"]
        with self.assertRaises(generate_edits_module.GenerationError) as caught:
            generate_edits_module.generate_edits(scan, missing)
        self.assertEqual("unknown_semantic", caught.exception.code)

    def test_explicit_selection_resolves_same_line_ambiguity(self) -> None:
        scan, selection = self.cpp_fixture()
        scan["candidates"].append(
            {"path": "src/kernel.cu", "line": 2, "semantic_id": "gemm"}
        )
        scan["ambiguities"] = [
            {
                "path": "src/kernel.cu",
                "line": 2,
                "candidates": ["gemm", "mma"],
            }
        ]

        generated = generate_edits_module.generate_edits(scan, selection)
        self.assertEqual({"3": {"semantic_id": "mma"}}, generated.names["semantics"])

    def test_one_ambiguous_line_cannot_have_two_semantics(self) -> None:
        self.write_source(
            "kernel.cu",
            "void kernel() {\n"
            "  both_begin();\n"
            "  both_end();\n"
            "}\n",
        )
        scan = self.scan(
            "cuda_cpp",
            ["kernel.cu"],
            [
                {"path": "kernel.cu", "line": 2, "semantic_id": "mma"},
                {"path": "kernel.cu", "line": 2, "semantic_id": "gemm"},
                {"path": "kernel.cu", "line": 3, "semantic_id": "mma"},
                {"path": "kernel.cu", "line": 3, "semantic_id": "gemm"},
            ],
            ambiguities=[
                {"path": "kernel.cu", "line": 2, "candidates": ["gemm", "mma"]},
                {"path": "kernel.cu", "line": 3, "candidates": ["gemm", "mma"]},
            ],
        )
        selection = self.selection(
            {
                "path": "kernel.cu",
                "semantic_id": "mma",
                "begin_anchor": "both_begin();",
                "end_anchor": "both_end();",
                "aux": "tile_id",
                "event_id": 3,
            },
            {
                "path": "kernel.cu",
                "semantic_id": "gemm",
                "begin_anchor": "both_begin();",
                "end_anchor": "both_end();",
                "aux": "tile_id",
                "event_id": 10,
            },
        )

        with self.assertRaises(generate_edits_module.GenerationError) as caught:
            generate_edits_module.generate_edits(scan, selection)
        self.assertEqual("conflicting_ambiguous_selection", caught.exception.code)

    def test_explicit_id_conflicting_with_canonical_semantic_is_rejected(self) -> None:
        self.write_source(
            "kernel.cu",
            "void kernel() {\n"
            "  mma_sync(a);\n"
            "  mma_store(a);\n"
            "}\n",
        )
        scan = self.scan(
            "cuda_cpp",
            ["kernel.cu"],
            [
                {"path": "kernel.cu", "line": 2, "semantic_id": "mma"},
                {"path": "kernel.cu", "line": 3, "semantic_id": "mma"},
            ],
        )
        selection = self.selection(
            {
                "path": "kernel.cu",
                "semantic_id": "mma",
                "begin_anchor": "mma_sync(a);",
                "end_anchor": "mma_store(a);",
                "aux": "tile_id",
                "event_id": 10,
            },
        )

        with self.assertRaises(generate_edits_module.GenerationError) as caught:
            generate_edits_module.generate_edits(scan, selection)
        self.assertEqual("event_id_semantic_conflict", caught.exception.code)
        self.assertEqual(3, caught.exception.details["canonical_event_id"])

    def test_overlapping_selected_ranges_are_rejected(self) -> None:
        self.write_source(
            "kernel.cu",
            "void kernel() {\n"
            "  mma_begin();\n"
            "  gemm_begin();\n"
            "  mma_end();\n"
            "  gemm_end();\n"
            "}\n",
        )
        scan = self.scan(
            "cuda_cpp",
            ["kernel.cu"],
            [
                {"path": "kernel.cu", "line": 2, "semantic_id": "mma"},
                {"path": "kernel.cu", "line": 3, "semantic_id": "gemm"},
                {"path": "kernel.cu", "line": 4, "semantic_id": "mma"},
                {"path": "kernel.cu", "line": 5, "semantic_id": "gemm"},
            ],
        )
        selection = self.selection(
            {
                "path": "kernel.cu",
                "semantic_id": "mma",
                "begin_anchor": "mma_begin();",
                "end_anchor": "mma_end();",
                "aux": "mma_tile",
                "event_id": 3,
            },
            {
                "path": "kernel.cu",
                "semantic_id": "gemm",
                "begin_anchor": "gemm_begin();",
                "end_anchor": "gemm_end();",
                "aux": "gemm_tile",
                "event_id": 10,
            },
        )

        with self.assertRaises(generate_edits_module.GenerationError) as caught:
            generate_edits_module.generate_edits(scan, selection)
        self.assertEqual("overlapping_ranges", caught.exception.code)

    def test_line_ranges_use_canonical_event_id(self) -> None:
        self.write_source(
            "kernel.py",
            "import triton\n"
            "import triton.language as tl\n"
            "@triton.jit\n"
            "def kernel(a, b, c):\n"
            "    acc = tl.dot(a, b)\n"
            "    tl.store(c, acc)\n",
        )
        scan = self.scan(
            "triton",
            ["kernel.py"],
            [
                {"path": "kernel.py", "line": 5, "semantic_id": "mma"},
                {"path": "kernel.py", "line": 6, "semantic_id": "mma"},
            ],
            include_project_root=False,
        )
        selection = self.selection(
            {
                "path": "kernel.py",
                "semantic_id": "mma",
                "begin_line": 5,
                "end_line": 6,
                "aux": "tile_id",
            }
        )
        scan_path = self.write_json("scan.json", scan)
        selection_path = self.write_json("selection.json", selection)
        out_path = self.root / "edits.json"

        completed = self.run_cli(
            scan_path,
            selection_path,
            out_path,
            "--project-root",
            str(self.root),
        )
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        summary = json.loads(completed.stdout)
        plan = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(str(self.root.resolve()), plan["project_root"])
        self.assertEqual(
            [
                {
                    "selection_index": 0,
                    "semantic_id": "mma",
                    "event_id": 3,
                    "source": "canonical_event_id",
                }
            ],
            summary["allocations"],
        )
        begin, end = plan["files"][0]["edits"]
        self.assertEqual("acc = tl.dot(a, b)", begin["anchor"])
        self.assertEqual("after", begin["position"])
        self.assertEqual("tl.store(c, acc)", end["anchor"])
        self.assertTrue(begin["text"].startswith("\nn = fi_b("))
        self.assertIn("ev=3", begin["text"])
        self.assertTrue(end["text"].startswith("\nn = fi_e("))
        self.assertIn("ev=3", end["text"])

    def test_semantic_kb_event_id_takes_precedence_over_compatibility_map(self) -> None:
        scan, selection = self.cpp_fixture()
        del selection["selections"][0]["event_id"]
        original_load_kb = generate_edits_module.semantics.load_kb

        def load_with_custom_mma_id(path=None, *, validate=True):
            kb = original_load_kb(path, validate=False)
            for row in kb["semantics"]:
                if row["semantic_id"] == "mma":
                    row["event_id"] = 321
            return kb

        with mock.patch.object(
            generate_edits_module.semantics,
            "load_kb",
            side_effect=load_with_custom_mma_id,
        ):
            generated = generate_edits_module.generate_edits(scan, selection)

        begin, end = generated.plan["files"][0]["edits"]
        self.assertIn("FI_B(321, tile_id);", begin["text"])
        self.assertIn("FI_E(321, tile_id);", end["text"])
        self.assertEqual({"321": "mma"}, generated.names["events"])
        self.assertEqual("anchor", begin["indent"])

    def test_project_root_override_is_rejected_when_scan_root_is_available(self) -> None:
        scan, selection = self.cpp_fixture()
        scan_path = self.write_json("scan.json", scan)
        selection_path = self.write_json("selection.json", selection)
        out_path = self.root / "edits.json"

        completed = self.run_cli(
            scan_path,
            selection_path,
            out_path,
            "--project-root",
            str(self.root),
        )

        self.assertEqual(2, completed.returncode, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertFalse(result["ok"])
        self.assertEqual(
            "project_root_override_not_allowed", result["error"]["code"]
        )
        self.assertFalse(out_path.exists())

    def test_real_apply_plan_check_accepts_generated_plan_without_writes(self) -> None:
        scan, selection = self.cpp_fixture()
        scan_path = self.write_json("scan.json", scan)
        selection_path = self.write_json("selection.json", selection)
        out_path = self.root / "edits.json"
        completed = self.run_cli(scan_path, selection_path, out_path)
        self.assertEqual(0, completed.returncode, completed.stderr)
        source_path = self.root / "src/kernel.cu"
        before = source_path.read_bytes()

        checked = subprocess.run(
            [sys.executable, "-B", str(APPLY_PLAN_PATH), str(out_path), "--check"],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        self.assertEqual(0, checked.returncode, checked.stderr + checked.stdout)
        result = json.loads(checked.stdout)
        self.assertTrue(result["ok"])
        self.assertEqual("validated", result["status"])
        self.assertFalse(result["writes_performed"])
        self.assertIn("+++ b/src/kernel.cu", result["diff"])
        self.assertIn("FI_B(3, tile_id);", result["diff"])
        self.assertEqual(before, source_path.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
