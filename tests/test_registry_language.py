#!/usr/bin/env python3
"""Card/language routing invariants for CUDA-style source on MI308X."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.pipeline.core import registry


class RegistryLanguageTests(unittest.TestCase):
    def test_mi308x_has_one_cpp_route(self) -> None:
        exact = registry.resolve(card="mi308x", arch="gfx942", language="hip_cpp")
        self.assertEqual(exact["language"], "hip_cpp")
        self.assertEqual(exact["recorder"], "recorders/hip_cpp/wp_lite_hip.h")
        self.assertNotIn("cuda_cpp", registry.CARD_SPECS["mi308x"]["languages"])

    def test_explicit_cuda_style_amd_language_routes_to_hip(self) -> None:
        route = registry.resolve(card="mi308x", arch="gfx942", language="cuda_cpp")
        self.assertEqual(route["language"], "hip_cpp")
        self.assertEqual(route["requested_language"], "cuda_cpp")
        self.assertEqual(route["recorder"], "recorders/hip_cpp/wp_lite_hip.h")
        self.assertEqual(route["device_clock"], "s_memrealtime")

    def test_auto_detected_cuda_source_on_amd_routes_to_hip(self) -> None:
        route = registry.resolve(
            card="mi308x", arch="gfx942",
            source_texts=["#include <cuda_runtime.h>\n__global__ void k() {}\n"],
            file_paths=["kernel.cu"],
        )
        self.assertEqual(route["language"], "hip_cpp")
        self.assertEqual(route["recorder"], "recorders/hip_cpp/wp_lite_hip.h")
        self.assertIn(".cu", route["source_extensions"])

    def test_nvidia_cuda_remains_cuda(self) -> None:
        route = registry.resolve(card="h20", arch="sm_90", language="cuda_cpp")
        self.assertEqual(route["language"], "cuda_cpp")
        self.assertEqual(route["recorder"], "recorders/cuda_cpp/wp_lite.cuh")
        self.assertNotIn("requested_language", route)


if __name__ == "__main__":
    unittest.main(verbosity=2)
