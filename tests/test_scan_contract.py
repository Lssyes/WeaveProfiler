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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.pipeline.core import registry  # noqa: E402


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
