#!/usr/bin/env python3
"""Contract tests for 2_instrument/apply_plan.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "2_instrument" / "apply_plan.py"
SPEC = importlib.util.spec_from_file_location("weaveprofiler_apply_plan", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
apply_plan = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = apply_plan
SPEC.loader.exec_module(apply_plan)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ApplyPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, relative: str, content: str | bytes) -> tuple[Path, str]:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = content.encode("utf-8") if isinstance(content, str) else content
        path.write_bytes(payload)
        return path, sha256(payload)

    def edit(
        self,
        edit_id: str,
        anchor: str,
        text: str,
        *,
        position: str = "after",
        occurrence: int | None = None,
        expect_count: int | None = None,
        indent: str | None = None,
    ) -> dict:
        value = {
            "id": edit_id,
            "anchor": anchor,
            "position": position,
            "text": text,
        }
        if occurrence is not None:
            value["occurrence"] = occurrence
        if expect_count is not None:
            value["expect_count"] = expect_count
        if indent is not None:
            value["indent"] = indent
        return value

    def plan(self, *files: dict) -> dict:
        return {
            "schema": apply_plan.PLAN_SCHEMA,
            "version": apply_plan.PLAN_VERSION,
            "project_root": str(self.root),
            "files": list(files),
        }

    def file_entry(self, relative: str, base_sha: str, edits: list[dict]) -> dict:
        return {"path": relative, "base_sha256": base_sha, "edits": edits}

    def read(self, relative: str) -> str:
        return (self.root / relative).read_text(encoding="utf-8")

    def test_successful_before_after_replace_and_diff(self) -> None:
        source, digest = self.write(
            "kernel.py",
            "def kernel():\n    alpha();\n    beta();\n",
        )
        document = self.plan(
            self.file_entry(
                "kernel.py",
                digest,
                [
                    self.edit(
                        "replace-alpha",
                        "alpha();",
                        "alpha_instrumented();",
                        position="replace",
                        expect_count=1,
                    ),
                    self.edit(
                        "before-beta",
                        "beta();",
                        "FI_B(1);\n",
                        position="before",
                        expect_count=1,
                        indent="anchor",
                    ),
                    self.edit(
                        "after-beta",
                        "beta();",
                        "\nFI_E(1);",
                        position="after",
                        expect_count=1,
                        indent="anchor",
                    ),
                ],
            )
        )

        result = apply_plan.apply_patch_plan(document)

        self.assertTrue(result["ok"], result)
        self.assertEqual("applied", result["status"])
        self.assertTrue(result["writes_performed"])
        self.assertIn("--- a/kernel.py", result["diff"])
        self.assertIn("+++ b/kernel.py", result["diff"])
        self.assertEqual(
            "def kernel():\n"
            "    alpha_instrumented();\n"
            "    FI_B(1);\n"
            "    beta();\n"
            "    FI_E(1);\n",
            source.read_text(encoding="utf-8"),
        )
        self.assertEqual(sha256(source.read_bytes()), result["files"][0]["result_sha256"])

    def test_multi_file_stale_hash_rolls_back_before_any_write(self) -> None:
        first, first_sha = self.write("first.py", "first();\n")
        second, _ = self.write("second.py", "second();\n")
        stale_sha = "0" * 64
        document = self.plan(
            self.file_entry(
                "first.py",
                first_sha,
                [self.edit("first-edit", "first();", "first_fi();", position="replace", expect_count=1)],
            ),
            self.file_entry(
                "second.py",
                stale_sha,
                [self.edit("second-edit", "second();", "second_fi();", position="replace", expect_count=1)],
            ),
        )
        first_before = first.read_bytes()
        second_before = second.read_bytes()

        result = apply_plan.apply_patch_plan(document)

        self.assertFalse(result["ok"])
        self.assertEqual("base_sha256_mismatch", result["errors"][0]["code"])
        self.assertFalse(result["writes_performed"])
        self.assertEqual(first_before, first.read_bytes())
        self.assertEqual(second_before, second.read_bytes())
        self.assertEqual([], list(self.root.glob(".*.weaveprofiler-*.tmp")))

    def test_replace_failure_restores_already_replaced_files(self) -> None:
        first, first_sha = self.write("first.py", "first();\n")
        second, second_sha = self.write("second.py", "second();\n")
        document = self.plan(
            self.file_entry(
                "first.py",
                first_sha,
                [self.edit("first-edit", "first();", "first_fi();", position="replace", expect_count=1)],
            ),
            self.file_entry(
                "second.py",
                second_sha,
                [self.edit("second-edit", "second();", "second_fi();", position="replace", expect_count=1)],
            ),
        )
        first_before = first.read_bytes()
        second_before = second.read_bytes()
        real_replace = os.replace
        failed = {"once": False}

        def fail_second_once(source: str, destination: str) -> None:
            if (
                not failed["once"]
                and Path(destination) == second
                and ".weaveprofiler-new." in Path(source).name
            ):
                failed["once"] = True
                raise OSError("injected replace failure")
            real_replace(source, destination)

        with mock.patch.object(apply_plan.os, "replace", side_effect=fail_second_once):
            result = apply_plan.apply_patch_plan(document)

        self.assertTrue(failed["once"])
        self.assertFalse(result["ok"])
        self.assertEqual("write_failed", result["errors"][0]["code"])
        self.assertEqual(first_before, first.read_bytes())
        self.assertEqual(second_before, second.read_bytes())

    def test_base_hash_expiry_is_fail_closed(self) -> None:
        path, digest = self.write("kernel.py", "alpha();\n")
        document = self.plan(
            self.file_entry(
                "kernel.py",
                digest,
                [self.edit("one", "alpha();", "beta();", position="replace", expect_count=1)],
            )
        )
        path.write_text("externally changed();\n", encoding="utf-8")

        result = apply_plan.apply_patch_plan(document)

        self.assertFalse(result["ok"])
        self.assertEqual("base_sha256_mismatch", result["errors"][0]["code"])
        self.assertEqual("externally changed();\n", self.read("kernel.py"))

    def test_anchor_zero_matches_fails(self) -> None:
        _, digest = self.write("kernel.py", "alpha();\n")
        document = self.plan(
            self.file_entry(
                "kernel.py",
                digest,
                [self.edit("missing", "not_here", "FI();", position="before", expect_count=1)],
            )
        )

        result = apply_plan.apply_patch_plan(document)

        self.assertFalse(result["ok"])
        self.assertEqual("anchor_not_found", result["errors"][0]["code"])

    def test_anchor_multiple_matches_fails_default_count(self) -> None:
        _, digest = self.write("kernel.py", "call();\ncall();\n")
        document = self.plan(
            self.file_entry(
                "kernel.py",
                digest,
                [self.edit("ambiguous", "call();", "FI();", position="before")],
            )
        )

        result = apply_plan.apply_patch_plan(document)

        self.assertFalse(result["ok"])
        self.assertEqual("anchor_count_mismatch", result["errors"][0]["code"])
        self.assertEqual(2, result["errors"][0]["details"]["actual_count"])

    def test_occurrence_selects_exact_anchor_from_declared_count(self) -> None:
        _, digest = self.write("kernel.py", "call();\ncall();\n")
        document = self.plan(
            self.file_entry(
                "kernel.py",
                digest,
                [
                    self.edit(
                        "second-only",
                        "call();",
                        "second();",
                        position="replace",
                        occurrence=2,
                        expect_count=2,
                    )
                ],
            )
        )

        result = apply_plan.apply_patch_plan(document)

        self.assertTrue(result["ok"], result)
        self.assertEqual("call();\nsecond();\n", self.read("kernel.py"))

    def test_overlapping_replacements_fail_without_writes(self) -> None:
        _, digest = self.write("kernel.py", "alpha_beta();\n")
        document = self.plan(
            self.file_entry(
                "kernel.py",
                digest,
                [
                    self.edit("left", "alpha_beta", "FI_A();", position="replace", expect_count=1),
                    self.edit("right", "beta();", "FI_B();", position="replace", expect_count=1),
                ],
            )
        )

        result = apply_plan.apply_patch_plan(document)

        self.assertFalse(result["ok"])
        self.assertEqual("overlapping_edits", result["errors"][0]["code"])
        self.assertEqual("alpha_beta();\n", self.read("kernel.py"))

    def test_path_escape_absolute_and_symlink_are_rejected(self) -> None:
        self.write("inside.py", "inside();\n")
        outside_dir = Path(tempfile.mkdtemp())
        try:
            outside = outside_dir / "outside.py"
            outside.write_text("outside();\n", encoding="utf-8")
            link = self.root / "linked.py"
            link.symlink_to(outside)

            cases = [
                "../outside.py",
                str(outside),
                "linked.py",
            ]
            for relative in cases:
                with self.subTest(relative=relative):
                    sha = sha256(outside.read_bytes())
                    document = self.plan(
                        self.file_entry(
                            relative,
                            sha,
                            [
                                self.edit(
                                    "escape",
                                    "outside();",
                                    "changed();",
                                    position="replace",
                                    expect_count=1,
                                )
                            ],
                        )
                    )
                    result = apply_plan.apply_patch_plan(document)
                    self.assertFalse(result["ok"])
                    self.assertIn(
                        result["errors"][0]["code"],
                        {"path_escape", "path_symlink"},
                    )
            self.assertEqual("outside();\n", outside.read_text(encoding="utf-8"))
        finally:
            for child in outside_dir.iterdir():
                child.unlink()
            outside_dir.rmdir()

    def test_indent_anchor_preserves_existing_crlf_and_line_indentation(self) -> None:
        _, digest = self.write("kernel.cpp", b"void f() {\r\n    alpha();\r\n}\r\n")
        document = self.plan(
            self.file_entry(
                "kernel.cpp",
                digest,
                [
                    self.edit(
                        "multiline-before",
                        "alpha();",
                        "FI_B(1);\r\nFI_B(2);\r\n",
                        position="before",
                        expect_count=1,
                        indent="anchor",
                    ),
                    self.edit(
                        "multiline-after",
                        "alpha();",
                        "\r\nFI_E(1);",
                        position="after",
                        expect_count=1,
                        indent="anchor",
                    ),
                ],
            )
        )

        result = apply_plan.apply_patch_plan(document)

        self.assertTrue(result["ok"], result)
        self.assertEqual(
            b"void f() {\r\n"
            b"    FI_B(1);\r\n"
            b"    FI_B(2);\r\n"
            b"    alpha();\r\n"
            b"    FI_E(1);\r\n"
            b"}\r\n",
            (self.root / "kernel.cpp").read_bytes(),
        )

    def test_check_mode_validates_and_writes_nothing(self) -> None:
        path, digest = self.write("kernel.py", "alpha();\n")
        document = self.plan(
            self.file_entry(
                "kernel.py",
                digest,
                [self.edit("one", "alpha();", "beta();", position="replace", expect_count=1)],
            )
        )
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(document), encoding="utf-8")
        before = path.read_bytes()

        completed = subprocess.run(
            [sys.executable, "-B", str(MODULE_PATH), str(plan_path), "--check"],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertTrue(result["ok"])
        self.assertTrue(result["check"])
        self.assertEqual("validated", result["status"])
        self.assertFalse(result["writes_performed"])
        self.assertIn("+++ b/kernel.py", result["diff"])
        self.assertEqual(before, path.read_bytes())

    def test_repeating_the_same_plan_fails_without_duplicate_insertion(self) -> None:
        path, digest = self.write("kernel.py", "alpha();\n")
        document = self.plan(
            self.file_entry(
                "kernel.py",
                digest,
                [
                    self.edit(
                        "one-open",
                        "alpha();",
                        "FI_B(1);\n",
                        position="before",
                        expect_count=1,
                        indent="anchor",
                    )
                ],
            )
        )

        first = apply_plan.apply_patch_plan(document)
        after_first = path.read_bytes()
        second = apply_plan.apply_patch_plan(document)

        self.assertTrue(first["ok"], first)
        self.assertFalse(second["ok"])
        self.assertEqual("base_sha256_mismatch", second["errors"][0]["code"])
        self.assertEqual(after_first, path.read_bytes())
        self.assertEqual("FI_B(1);\nalpha();\n", after_first.decode("utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
