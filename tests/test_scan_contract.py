#!/usr/bin/env python3
"""Contract tests for lexical masking, scan blocking and route provenance."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_PATH = ROOT / "weaveprof/pipeline/scan" / "scan.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.pipeline.core import registry  # noqa: E402


def load_scan_module():
    spec = importlib.util.spec_from_file_location("weaveprofiler_scan_contract", SCAN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


scan_mod = load_scan_module()


class NonCodeMaskingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="weaveprofiler-scan-mask-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _scan(self, name: str, text: str, language: str, card: str = "h20",
              arch: str = "sm_90", platform: str = "nv") -> dict:
        (self.root / name).write_text(text, encoding="utf-8")
        return scan_mod.scan(
            self.root,
            card=card,
            arch=arch,
            platform=platform,
            language=language,
            sources=[name],
            globs=[],
            cache_dir=self.root / ".weaveprofiler" / "cache" / "scan",
        )

    def test_c_comments_strings_and_chars_are_not_candidates(self) -> None:
        source = (
            "__global__ void kernel() {\n"
            "  // GEMM tl.dot wgmma\n"
            "  const char* text = \"gemm tl.dot\";\n"
            "  char grade = 'G';\n"
            "  /* gemm\n"
            "     tl.dot wgmma */\n"
            "  float x = cp.async.load(in);\n"
            "  __syncthreads();\n"
            "  st.global.f32(out, x);\n"
            "}\n"
        )
        result = self._scan("kernel.cu", source, "cuda_cpp")
        ids = {item["semantic_id"] for item in result["candidates"]}
        self.assertEqual({"load", "barrier", "store"}, ids)
        self.assertEqual("PASS", result["status"])

        load = next(item for item in result["candidates"] if item["semantic_id"] == "load")
        self.assertEqual(7, load["line"])
        self.assertEqual(13, load["column"])
        self.assertEqual("float x = cp.async.load(in);", load["source_excerpt"])

        masked = scan_mod.neutralize_non_code(source, "cuda_cpp")
        self.assertEqual(len(source), len(masked))
        self.assertEqual(source.count("\n"), masked.count("\n"))
        self.assertIn("cp.async", masked)
        self.assertNotIn("GEMM", masked)
        self.assertNotIn("gemm", masked)
        self.assertNotIn("wgmma", masked)

    def test_python_comments_quotes_and_triple_quotes_are_not_candidates(self) -> None:
        source = (
            "import triton\n"
            "import triton.language as tl\n"
            "\n"
            "# gemm tl.dot tl.store\n"
            "def kernel(x):\n"
            "    note = \"gemm tl.dot\"\n"
            "    other = 'tl.store'\n"
            "    block = '''gemm\n"
            "tl.dot\n"
            "tl.store'''\n"
            "    # wgmma tl.dot\n"
            "    value = tl.load(x)  # gemm tl.dot\n"
            "    return value\n"
        )
        result = self._scan("kernel.py", source, "triton")
        ids = [item["semantic_id"] for item in result["candidates"]]
        self.assertEqual(["load"], ids)
        self.assertEqual("PASS", result["status"])

        load = result["candidates"][0]
        self.assertEqual(12, load["line"])
        self.assertEqual(13, load["column"])
        self.assertEqual("value = tl.load(x)  # gemm tl.dot", load["source_excerpt"])

        masked = scan_mod.neutralize_non_code(source, "triton")
        self.assertEqual(len(source), len(masked))
        self.assertEqual(source.count("\n"), masked.count("\n"))
        self.assertIn("tl.load", masked)
        self.assertNotIn("gemm", masked)
        self.assertNotIn("tl.dot", masked)
        self.assertNotIn("tl.store", masked)


class ScanStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="weaveprofiler-scan-status-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _scan(self, refresh: bool = False) -> dict:
        return scan_mod.scan(
            self.root,
            card="h20",
            arch="sm_90",
            platform="nv",
            language="triton",
            sources=["kernel.py"],
            globs=[],
            cache_dir=self.root / ".weaveprofiler" / "cache" / "scan",
            refresh=refresh,
        )

    def test_ambiguity_blocks_and_writes_unknown_at_each_location(self) -> None:
        (self.root / "kernel.py").write_text("y = exp2(x)\n", encoding="utf-8")
        first = self._scan()
        self.assertEqual("BLOCKED", first["status"])
        self.assertEqual(1, len(first["ambiguities"]))
        ambiguity = first["ambiguities"][0]
        self.assertEqual({"compute", "softmax"}, set(ambiguity["candidates"]))
        self.assertEqual("ambiguous_semantic_candidates", ambiguity["reason"])

        self.assertEqual(1, len(first["unknown"]))
        unknown = first["unknown"][0]
        self.assertEqual("unknown", unknown["semantic_id"])
        self.assertEqual((ambiguity["path"], ambiguity["line"], ambiguity["column"]),
                         (unknown["path"], unknown["line"], unknown["column"]))
        self.assertEqual(ambiguity["candidates"], unknown["candidates"])
        self.assertEqual("ambiguous_semantic_candidates", unknown["reason"])

        cached = self._scan()
        self.assertTrue(cached["cache"]["hit"])
        self.assertEqual("BLOCKED", cached["status"])
        self.assertEqual(first["ambiguities"], cached["ambiguities"])
        self.assertEqual(first["unknown"], cached["unknown"])

    def test_cli_returns_nonzero_and_preserves_blocked_result(self) -> None:
        (self.root / "kernel.py").write_text("y = exp2(x)\n", encoding="utf-8")
        output = self.root / "scan.json"
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [
                sys.executable, "-B", str(SCAN_PATH),
                "--project", str(self.root),
                "--card", "h20",
                "--arch", "sm_90",
                "--language", "triton",
                "--source", "kernel.py",
                "--out", str(output),
            ],
            cwd=str(self.root),
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        self.assertNotEqual(0, completed.returncode, completed.stderr)
        summary = json.loads(completed.stdout)
        self.assertEqual("BLOCKED", summary["status"])
        self.assertEqual(1, summary["unknown"])
        self.assertEqual(["ambiguous_semantic_candidates"], summary["blocking_reasons"])
        document = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual("BLOCKED", document["status"])
        self.assertTrue(document["ambiguities"])
        self.assertTrue(document["unknown"])


class RouteEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="weaveprofiler-route-evidence-")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.semantic_path = self.repo / "configs" / "semantic_ids.json"
        self.recipe_index_path = self.repo / "configs" / "recipes" / "index.json"
        self.recipe_path = self.repo / "configs" / "recipes" / "nv" / "b300" / "triton.json"
        self.profile_100_path = self.repo / "configs" / "cards" / "b300_sm_100a.json"
        self.profile_103_path = self.repo / "configs" / "cards" / "b300_sm_103a.json"
        self.pattern_path = self.repo / "configs" / "languages" / "triton" / "nvidia_patterns.json"
        self.recorder_path = self.repo / "recorders" / "triton" / "wp_triton.py"
        self._write_repo()

    def _write_json(self, relative: str, value: object) -> Path:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return path

    def _write_repo(self) -> None:
        self._write_json("configs/semantic_ids.json", {"semantics": []})
        self._write_json("configs/recipes/index.json", {
            "schema": "weaveprofiler.recipe-index/v1",
            "semantic_authority": "configs/semantic_ids.json",
            "recipes": [{
                "recipe_id": "nv/b300/triton",
                "card": "b300",
                "architectures": ["sm_100a", "sm_103a"],
                "language": "triton",
                "path": "configs/recipes/nv/b300/triton.json",
            }],
        })
        self._write_json("configs/recipes/nv/b300/triton.json", {
            "schema": "weaveprofiler.recipe/v1",
            "recipe_id": "nv/b300/triton",
            "language": "triton",
            "target": {"card": "b300", "platform": "nv"},
            "machine_profiles": [
                "configs/cards/b300_sm_100a.json",
                "configs/cards/b300_sm_103a.json",
            ],
            "recorder": {"path": "recorders/triton/wp_triton.py"},
        })
        self._write_json("configs/cards/b300_sm_100a.json", {
            "schema": "weaveprofiler.card-machine/v1", "card": "b300", "arch": "sm_100a"})
        self._write_json("configs/cards/b300_sm_103a.json", {
            "schema": "weaveprofiler.card-machine/v1", "card": "b300", "arch": "sm_103a"})
        self._write_json("configs/languages/triton/nvidia_patterns.json", {
            "schema": "weaveprofiler-code-patterns/v1", "language": "triton", "backend": "nv"})
        self.recorder_path.parent.mkdir(parents=True, exist_ok=True)
        self.recorder_path.write_text("// test recorder\n", encoding="utf-8")

    def _route(self, arch: str = "sm_100a") -> dict:
        return registry.resolve(card="b300", arch=arch, platform="nv", language="triton")

    def _evidence(self, route: dict) -> dict:
        return scan_mod.build_route_evidence(
            route, repo_root=self.repo, semantic_kb_path=self.semantic_path)

    def _key(self, route: dict, evidence: dict) -> str:
        return scan_mod._cache_key(
            route, [],
            evidence["semantic_kb"]["sha256"],
            {"max_file_bytes": 1024},
            evidence,
        )

    def test_b300_exact_arch_selects_one_card_profile(self) -> None:
        evidence_100 = self._evidence(self._route("sm_100a"))
        evidence_103 = self._evidence(self._route("sm_103a"))
        self.assertEqual(
            ["configs/cards/b300_sm_100a.json"],
            [item["path"] for item in evidence_100["card_profiles"]],
        )
        self.assertEqual(
            ["configs/cards/b300_sm_103a.json"],
            [item["path"] for item in evidence_103["card_profiles"]],
        )
        self.assertEqual("sm_100a", evidence_100["card_profiles"][0]["arch"])
        self.assertEqual("sm_103a", evidence_103["card_profiles"][0]["arch"])

        route_without_arch = registry.resolve(card="b300", language="triton", platform="nv")
        with self.assertRaises(scan_mod.ScanError):
            self._evidence(route_without_arch)

    def test_cache_key_tracks_each_route_knowledge_hash(self) -> None:
        route = self._route("sm_100a")
        baseline = self._evidence(route)
        baseline_key = self._key(route, baseline)
        tracked = (
            ("semantic_kb", self.semantic_path),
            ("recipe_index", self.recipe_index_path),
            ("route_recipe", self.recipe_path),
            ("card_profile", self.profile_100_path),
            ("language_patterns", self.pattern_path),
            ("recorder", self.recorder_path),
        )
        for role, path in tracked:
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"\n")
                changed = self._evidence(route)
                if role == "card_profile":
                    changed_entry = changed["card_profiles"][0]
                else:
                    changed_entry = changed[role]
                self.assertNotEqual(
                    baseline[role]["sha256"] if role != "card_profile"
                    else baseline["card_profiles"][0]["sha256"],
                    changed_entry["sha256"],
                    role,
                )
                self.assertNotEqual(baseline_key, self._key(route, changed), role)
            finally:
                path.write_bytes(original)

    def test_route_evidence_paths_are_repo_relative_and_complete(self) -> None:
        evidence = self._evidence(self._route("sm_100a"))
        entries = [
            evidence["semantic_kb"], evidence["recipe_index"],
            evidence["route_recipe"], evidence["card_profiles"][0],
            evidence["language_patterns"], evidence["recorder"],
        ]
        for entry in entries:
            self.assertFalse(Path(entry["path"]).is_absolute())
            self.assertFalse(str(entry["path"]).startswith(".."))
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")


class LanguageCueTests(unittest.TestCase):
    def test_cpp_and_triton_markers_do_not_cross_platforms(self) -> None:
        cuda_markers = " ".join(registry.LANGUAGE_SPECS["cuda_cpp"]["markers"]).lower()
        hip_markers = " ".join(registry.LANGUAGE_SPECS["hip_cpp"]["markers"]).lower()
        triton_markers = " ".join(registry.LANGUAGE_SPECS["triton"]["markers"]).lower()

        for forbidden in ("rocwmma", "mfma", "tl.dot", "tl\\.dot", "hip"):
            self.assertNotIn(forbidden, cuda_markers)
        for forbidden in ("cuda", "nvidia", "rocm"):
            self.assertNotIn(forbidden, hip_markers)
        self.assertIsNone(re.search(r"(?<![a-z])nv(?![a-z])", hip_markers))
        for forbidden in ("cuda", "nvidia", "amd", "rocm", "hip"):
            self.assertNotIn(forbidden, triton_markers)
        self.assertIn("tl\\.dot", triton_markers)

    def test_registry_detection_prefers_hip_over_cuda_and_keeps_triton_neutral(self) -> None:
        mixed_cpp = (
            "#include <cuda_runtime.h>\n"
            "#include <hip/hip_runtime.h>\n"
            "__global__ void kernel() { hipLaunchKernelGGL(kernel, 1, 1, 0, 0); }\n"
        )
        self.assertEqual("hip_cpp", registry.detect_language(source_texts=[mixed_cpp]))
        amd_route = registry.resolve(
            card="mi308x", arch="gfx942", source_texts=[mixed_cpp], file_paths=["kernel.cpp"])
        self.assertEqual("hip_cpp", amd_route["language"])

        nv_triton = registry.resolve(
            card="h20", arch="sm_90",
            source_texts=["import triton\nimport triton.language as tl\nx = tl.dot(a, b)\n"],
            file_paths=["kernel.py"],
        )
        amd_triton = registry.resolve(
            card="mi308x", arch="gfx942",
            source_texts=["import triton\nimport triton.language as tl\nx = tl.dot(a, b)\n"],
            file_paths=["kernel.py"],
        )
        self.assertEqual("triton", nv_triton["language"])
        self.assertEqual("triton", amd_triton["language"])
        self.assertEqual(
            "configs/languages/triton/nvidia_patterns.json",
            nv_triton["language_pattern_library"],
        )
        self.assertEqual(
            "configs/languages/triton/amd_patterns.json",
            amd_triton["language_pattern_library"],
        )

    def test_language_pattern_library_mapping(self) -> None:
        cases = (
            ("cuda_cpp", "nv", "configs/languages/cuda_cpp/patterns.json"),
            ("hip_cpp", "amd", "configs/languages/hip_cpp/patterns.json"),
            ("triton", "nv", "configs/languages/triton/nvidia_patterns.json"),
            ("triton", "amd", "configs/languages/triton/amd_patterns.json"),
            ("cutedsl", "nv", "configs/languages/cutedsl/patterns.json"),
        )
        for language, platform, expected in cases:
            self.assertEqual(expected, registry.language_pattern_path(language, platform))


if __name__ == "__main__":
    unittest.main(verbosity=2)
