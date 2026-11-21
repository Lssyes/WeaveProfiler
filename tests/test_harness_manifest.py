#!/usr/bin/env python3
"""Regression tests for the harness manifest's canonical tree location."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PY = ROOT / "4_validate" / "harness" / "manifest.py"
MANIFEST_REL = Path("4_validate") / "harness" / "MANIFEST.json"


def load_manifest_module():
    spec = importlib.util.spec_from_file_location("manifest_under_test", MANIFEST_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(MANIFEST_PY))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MANIFEST = load_manifest_module()


class HarnessManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp_obj = tempfile.TemporaryDirectory(prefix="weaveprofiler-manifest-")
        self.addCleanup(self.tmp_obj.cleanup)
        self.root = Path(self.tmp_obj.name)
        (self.root / "kernel.txt").write_text("stable input\n", encoding="utf-8")
        (self.root / "4_validate" / "harness").mkdir(parents=True)

    def test_build_writes_canonical_path_and_tree_verify_uses_it(self):
        payload = MANIFEST.build(self.root)
        canonical = self.root / MANIFEST_REL
        legacy = self.root / "harness" / "MANIFEST.json"
        self.assertTrue(canonical.is_file())
        self.assertFalse(legacy.exists())
        self.assertEqual(payload["file_count"], 1)
        self.assertEqual(payload["skill_root"], ".")

        report, code = MANIFEST.verify(self.root, "tree")
        self.assertEqual(code, 0, report)
        self.assertEqual(report["status"], "CLEAN")

    @unittest.skipUnless(shutil.which("git"), "git is required for the committed anchor test")
    def test_committed_manifest_reads_canonical_path_and_git_verify_works(self):
        MANIFEST.build(self.root)
        subprocess.run(["git", "-C", str(self.root), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(self.root), "-c", "user.name=weaveprofiler-test",
             "-c", "user.email=weaveprofiler-test@example.invalid",
             "commit", "-q", "-m", "manifest fixture"],
            check=True,
        )

        document, error = MANIFEST.committed_manifest(self.root)
        self.assertIsNone(error)
        self.assertEqual(document, json.loads((self.root / MANIFEST_REL).read_text(encoding="utf-8")))

        report, code = MANIFEST.verify(self.root, "git")
        self.assertEqual(code, 0, report)
        self.assertEqual(report["status"], "CLEAN")


if __name__ == "__main__":
    unittest.main()
