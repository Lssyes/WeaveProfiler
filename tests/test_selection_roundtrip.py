#!/usr/bin/env python3
"""Executable selection-schema to patch-plan round-trip contract test."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "2_instrument" / "generate_edits.py"
SCHEMA_PATH = (
    ROOT / "knowledge" / "validation" / "schemas" / "scan-selection.schema.json"
)
SPEC = importlib.util.spec_from_file_location(
    "weaveprofiler_selection_roundtrip_generate_edits", GENERATOR_PATH
)
assert SPEC is not None and SPEC.loader is not None
generate_edits_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generate_edits_module
SPEC.loader.exec_module(generate_edits_module)

try:
    import jsonschema
except ImportError:  # pragma: no cover - exercised only without the optional dependency
    jsonschema = None


class SelectionRoundtripTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="weaveprofiler-selection-roundtrip-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    @staticmethod
    def _sha256(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def _sample_document(self) -> tuple[dict[str, Any], dict[str, Any]]:
        source = (
            "void kernel() {\n"
            "  mma_begin();\n"
            "  mma_end();\n"
            "  mma_begin_2();\n"
            "  mma_end_2();\n"
            "}\n"
        )
        source_path = self.root / "kernel.cu"
        source_path.write_text(source, encoding="utf-8")
        payload = source_path.read_bytes()
        scan = {
            "schema": "weaveprofiler-scan/v1",
            "project_root": str(self.root),
            "route": {"language": "cuda_cpp"},
            "files": [
                {
                    "path": "kernel.cu",
                    "bytes": len(payload),
                    "sha256": self._sha256(payload),
                }
            ],
            "candidates": [
                {"path": "kernel.cu", "line": 2, "semantic_id": "mma"},
                {"path": "kernel.cu", "line": 3, "semantic_id": "mma"},
                {"path": "kernel.cu", "line": 4, "semantic_id": "mma"},
                {"path": "kernel.cu", "line": 5, "semantic_id": "mma"},
            ],
            "ambiguities": [],
        }
        selection = {
            "schema": "weaveprofiler-instrument-selection/v1",
            "version": 1,
            "selections": [
                {
                    "path": "kernel.cu",
                    "semantic_id": "mma",
                    "begin_anchor": "mma_begin();",
                    "end_anchor": "mma_end();",
                    "aux": "tile_id",
                },
                {
                    "path": "kernel.cu",
                    "semantic_id": "mma",
                    "begin_line": 4,
                    "end_line": 5,
                    "occurrence": 1,
                    "expect_count": 1,
                    "aux": "tile_id",
                    "event_id": 3,
                },
            ],
        }
        return scan, selection

    def _validate_without_jsonschema(self, document: Mapping[str, Any]) -> None:
        """Validate the sample with the same structural rules as the schema."""

        self.assertEqual(
            {"schema", "version", "selections"},
            set(document),
        )
        self.assertEqual("weaveprofiler-instrument-selection/v1", document["schema"])
        self.assertIs(type(document["version"]), int)
        self.assertEqual(1, document["version"])
        selections = document["selections"]
        self.assertIsInstance(selections, list)
        self.assertTrue(selections)
        semantic_ids = set(self.schema["$defs"]["semanticId"]["enum"])
        allowed = {
            "path",
            "semantic_id",
            "begin_anchor",
            "end_anchor",
            "begin_line",
            "end_line",
            "occurrence",
            "expect_count",
            "aux",
            "event_id",
        }
        for item in selections:
            self.assertIsInstance(item, dict)
            self.assertLessEqual(set(item), allowed)
            self.assertLessEqual({"path", "semantic_id", "aux"}, set(item))
            self.assertIsInstance(item["path"], str)
            self.assertTrue(item["path"])
            self.assertFalse(item["path"].startswith(("/", "\\")))
            self.assertNotRegex(item["path"], r"\\")
            self.assertNotRegex(item["path"], r"(?:^|/)\.{1,2}(?:/|$)")
            self.assertNotIn("//", item["path"])
            self.assertFalse(item["path"].endswith("/"))
            self.assertNotRegex(item["path"], r"^[A-Za-z]:")
            self.assertIn(item["semantic_id"], semantic_ids)
            self.assertTrue(
                isinstance(item["aux"], int)
                and not isinstance(item["aux"], bool)
                or isinstance(item["aux"], str)
                and bool(re.search(r"\S", item["aux"]))
                and not re.search(r"[\x00\r\n;{}]", item["aux"])
            )
            anchor_mode = "begin_anchor" in item or "end_anchor" in item
            line_mode = "begin_line" in item or "end_line" in item
            self.assertNotEqual(anchor_mode, line_mode)
            if anchor_mode:
                self.assertIn("begin_anchor", item)
                self.assertIn("end_anchor", item)
                self.assertNotIn("begin_line", item)
                self.assertNotIn("end_line", item)
                self.assertIsInstance(item["begin_anchor"], str)
                self.assertIsInstance(item["end_anchor"], str)
                self.assertTrue(item["begin_anchor"])
                self.assertTrue(item["end_anchor"])
            else:
                self.assertIn("begin_line", item)
                self.assertIn("end_line", item)
                self.assertNotIn("begin_anchor", item)
                self.assertNotIn("end_anchor", item)
                for key in ("begin_line", "end_line"):
                    self.assertIs(type(item[key]), int)
                    self.assertGreaterEqual(item[key], 1)
            for key in ("occurrence", "expect_count"):
                if key in item:
                    self.assertIs(type(item[key]), int)
                    self.assertGreaterEqual(item[key], 1)
            if "event_id" in item:
                self.assertIs(type(item["event_id"]), int)
                self.assertGreaterEqual(item["event_id"], 1)
                self.assertLessEqual(item["event_id"], 4095)

    def _validate_sample(self, document: Mapping[str, Any]) -> None:
        if jsonschema is not None:
            jsonschema.Draft202012Validator.check_schema(self.schema)
            errors = sorted(
                jsonschema.Draft202012Validator(self.schema).iter_errors(document),
                key=lambda error: list(error.absolute_path),
            )
            self.assertEqual([], [error.message for error in errors])
            return
        self._validate_without_jsonschema(document)

    def test_schema_valid_selection_round_trips_through_generator(self) -> None:
        selection_schema = self.schema["$defs"]["selection"]
        self.assertEqual(
            generate_edits_module.SELECTION_TOP_LEVEL_KEYS,
            set(self.schema["properties"]),
        )
        self.assertEqual(
            generate_edits_module.SELECTION_KEYS,
            set(selection_schema["properties"]),
        )
        self.assertEqual(
            set(generate_edits_module.CANONICAL_EVENT_IDS),
            set(self.schema["$defs"]["semanticId"]["enum"]),
        )

        scan, selection = self._sample_document()
        self._validate_sample(selection)

        generated = generate_edits_module.generate_edits(scan, selection)
        begin, end, begin_2, end_2 = generated.plan["files"][0]["edits"]
        self.assertEqual("\nFI_B(3, tile_id);", begin["text"])
        self.assertEqual("FI_E(3, tile_id);\n", end["text"])
        self.assertEqual("\nFI_B(3, tile_id);", begin_2["text"])
        self.assertEqual("\nFI_E(3, tile_id);", end_2["text"])
        self.assertEqual({"3": "mma"}, generated.names["events"])
        self.assertEqual(
            [
                {
                    "selection_index": 0,
                    "semantic_id": "mma",
                    "event_id": 3,
                    "source": "canonical_event_id",
                }
            ],
            list(generated.allocations),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
