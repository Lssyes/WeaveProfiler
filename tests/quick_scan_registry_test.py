#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for registry-driven quick_scan and preflight routing."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.pipeline.core import registry  # noqa: E402


def load_module(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


quick_scan = load_module("weaveprofiler_quick_scan_registry_test", "weaveprof/pipeline/scan/quick_scan.py")
preflight = load_module("weaveprofiler_fi_preflight_registry_test", "weaveprof/pipeline/validate/tools/fi_preflight.py")


HIP_SOURCE = """\
#include <hip/hip_runtime.h>
__global__ void gemm_kernel(float* out) {
  unsigned long long t = s_memrealtime;
  (void)t;
}
int main() { hipLaunchKernelGGL(gemm_kernel, 1, 1, 0, 0, nullptr); }
"""

TRITON_SOURCE = """\
import triton
import triton.language as tl

@triton.jit
def gemm_kernel(x, y):
    return tl.dot(tl.load(x), tl.load(y))
"""


def args(**overrides):
    values = dict(card=None, arch=None, platform="auto", language=None,
                  inspect_host=False, gpu=None, busy_util=5, busy_mem=1024,
                  lock_dir=None, expect_lock_dir=None, need_nsys=False,
                  rocm_llvm=None)
    values.update(overrides)
    return SimpleNamespace(**values)


class FrameworkDetectionTests(unittest.TestCase):
    def test_hip_cpp_with_global_is_not_cuda(self):
        framework = quick_scan.detect_framework([("gemm.cpp", HIP_SOURCE)])
        self.assertEqual(framework["dominant"], "hip_cpp", framework)
        self.assertGreaterEqual(framework["file_counts"]["hip_cpp"], 1)
        route = quick_scan.resolve_triage_route(
            card="mi308x", arch="gfx942", language=framework["dominant"],
            file_texts=[("gemm.cpp", HIP_SOURCE)])
        self.assertEqual(route["target"]["card"], "mi308x")
        self.assertEqual(route["target"]["platform"], "amd")
        self.assertEqual(route["target"]["arch"], "gfx942")
        self.assertEqual(route["language"], "hip_cpp")
        self.assertEqual(route["recorder"], "recorders/hip_cpp/wp_lite_hip.h")
        self.assertEqual(route["profiler"]["primary"], "rocprof-sys")

    def test_hip_extension_wins_even_with_only_global_marker(self):
        framework = quick_scan.detect_framework(
            [("kernel.hip", "__global__ void k() {}\n")])
        self.assertEqual(framework["dominant"], "hip_cpp", framework)

    def test_amd_target_disambiguates_bare_global_cpp_as_hip(self):
        with tempfile.TemporaryDirectory(prefix="wq_hip_bare_") as tmp:
            (Path(tmp) / "kernel.cpp").write_text(
                "__global__ void k() {}\n", encoding="utf-8")
            report = quick_scan.build_report(
                tmp, None, card="mi308x", arch="gfx942")
        self.assertEqual(report["framework"]["detected"], "cuda_cpp")
        self.assertEqual(report["framework"]["dominant"], "hip_cpp")
        self.assertEqual(report["framework"]["selected_by"],
                         "target_platform_disambiguation")
        self.assertEqual(report["language"], "hip_cpp")

    def test_amd_triton_uses_amd_adapter(self):
        framework = quick_scan.detect_framework([("gemm.py", TRITON_SOURCE)])
        self.assertEqual(framework["dominant"], "triton", framework)
        route = quick_scan.resolve_triage_route(
            card="mi308x", arch="gfx942", language=framework["dominant"],
            file_texts=[("gemm.py", TRITON_SOURCE)])
        self.assertEqual(route["language"], "triton")
        self.assertEqual(route["recorder"], "recorders/triton/wp_triton_amd.py")
        self.assertEqual(route["profiler"]["primary"], "rocprof-sys")


class QuickScanRouteTests(unittest.TestCase):
    def _report(self, source_name, source, **route_args):
        with tempfile.TemporaryDirectory(prefix="wq_quick_scan_") as tmp:
            (Path(tmp) / source_name).write_text(source, encoding="utf-8")
            return quick_scan.build_report(tmp, None, **route_args)

    def test_h20_cuda_route(self):
        report = self._report(
            "gemm.cu",
            "#include <cuda_runtime.h>\n__global__ void k() {}\n",
            card="h20", arch="sm_90")
        self.assertEqual(report["target"]["card"], "h20")
        self.assertEqual(report["target"]["platform"], "nv")
        self.assertEqual(report["target"]["arch"], "sm_90")
        self.assertEqual(report["language"], "cuda_cpp")
        self.assertEqual(report["recorder"], "recorders/cuda_cpp/wp_lite.cuh")
        self.assertEqual(report["profiler"]["primary"], "nsys")
        self.assertTrue(report["route_resolved"])
        self.assertIsNone(report["route_error"])

    def test_b300_triton_route_keeps_arch_set_support(self):
        report = self._report(
            "gemm.py", TRITON_SOURCE,
            card="b300", arch="sm_103a")
        self.assertEqual(report["target"]["card"], "b300")
        self.assertEqual(report["target"]["platform"], "nv")
        self.assertEqual(report["target"]["arch"], "sm_103a")
        self.assertEqual(report["language"], "triton")
        self.assertEqual(report["recorder"], "recorders/triton/wp_triton.py")
        self.assertEqual(report["profiler"]["primary"], "nsys")
        self.assertEqual(report["evidence_status"], "cross_compile_only")

    def test_mi308x_hip_route_and_unresolved_is_slow_path(self):
        report = self._report(
            "gemm.cpp", HIP_SOURCE, card="mi308x", arch="gfx942")
        self.assertEqual(report["target"]["card"], "mi308x")
        self.assertEqual(report["target"]["platform"], "amd")
        self.assertEqual(report["target"]["arch"], "gfx942")
        self.assertEqual(report["language"], "hip_cpp")
        self.assertEqual(report["recorder"], "recorders/hip_cpp/wp_lite_hip.h")
        self.assertEqual(report["profiler"]["primary"], "rocprof-sys")

        unresolved = self._report(
            "gemm.cu", "#include <cuda_runtime.h>\n__global__ void k() {}\n",
            allow_unresolved=True)
        self.assertFalse(unresolved["route_resolved"])
        self.assertEqual(unresolved["evidence_status"], "unverified")
        self.assertEqual(unresolved["decision"]["path"], "slow_path")
        self.assertEqual(unresolved["exit_code"], quick_scan.EXIT_SLOW)
        self.assertIsNotNone(unresolved["route_error"])


class PreflightRouteTests(unittest.TestCase):
    def test_legacy_host_probe_has_no_nvidia_default(self):
        with mock.patch.object(preflight.shutil, "which", return_value=None):
            self.assertIsNone(preflight.platform_of(None))

    def test_three_supported_card_routes(self):
        cases = (
            (args(card="h20", arch="sm_90", language="cuda_cpp"), "nv", "nsys"),
            (args(card="b300", arch="sm_100a", language="triton"), "nv", "nsys"),
            (args(card="b300", arch="sm_103a", language="cuda_cpp"), "nv", "nsys"),
            (args(card="mi308x", arch="gfx942", language="hip_cpp"), "amd", "rocprof-sys"),
            (args(card="mi308x", arch="gfx942", language="triton"), "amd", "rocprof-sys"),
        )
        for namespace, platform, profiler in cases:
            route = preflight.resolve_preflight_target(namespace)
            self.assertEqual(route["target"]["platform"], platform)
            self.assertEqual(route["target"]["card"], namespace.card)
            self.assertEqual(route["target"]["arch"], namespace.arch)
            self.assertEqual(route["profiler"]["primary"], profiler)
            self.assertTrue(route["recorder"])
            self.assertTrue(route["evidence_status"])

    def test_missing_target_fails_closed_without_host_probe(self):
        with mock.patch.object(preflight, "platform_of", side_effect=AssertionError("host probe")), \
                mock.patch.object(preflight, "inspect_host_target", side_effect=AssertionError("host probe")):
            with self.assertRaises(registry.UnsupportedTarget) as caught:
                preflight.resolve_preflight_target(args())
        self.assertEqual(caught.exception.kind, "target_required")

        with self.assertRaises(registry.UnsupportedTarget) as caught:
            preflight.resolve_preflight_target(args(platform="amd"))
        self.assertEqual(caught.exception.kind, "target_required")

    def test_inspect_host_is_an_explicit_opt_in(self):
        host = {"platform": "nv", "card": "h20", "arch": "sm_90",
                "evidence": ["fake nvidia-smi"]}
        namespace = args(inspect_host=True)
        with mock.patch.object(preflight, "inspect_host_target", return_value=host) as probe:
            route = preflight.resolve_preflight_target(namespace)
        probe.assert_called_once_with(None)
        self.assertEqual(route["target"]["card"], "h20")
        self.assertEqual(route["target"]["platform"], "nv")
        self.assertEqual(route["target"]["arch"], "sm_90")
        self.assertEqual(route["host_inspection"], host)

    def test_cli_missing_target_writes_structured_failure(self):
        with tempfile.TemporaryDirectory(prefix="wq_preflight_json_") as tmp:
            output = Path(tmp) / "preflight.json"
            argv = ["fi_preflight.py", "--json", str(output)]
            stdout, stderr = io.StringIO(), io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout), \
                    contextlib.redirect_stderr(stderr):
                rc = preflight.main()
            self.assertEqual(rc, 1, (stdout.getvalue(), stderr.getvalue()))
            doc = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(doc["verdict"], "FAIL")
        self.assertEqual(doc["target_route_error"]["kind"], "target_required")
        self.assertEqual(doc["target"], {"card": None, "platform": None, "arch": None})
        self.assertEqual(doc["evidence_status"], "unverified")


class QuickScanCliTests(unittest.TestCase):
    def test_explicit_hip_route_in_json(self):
        with tempfile.TemporaryDirectory(prefix="wq_quick_cli_") as tmp:
            source = Path(tmp) / "gemm.cpp"
            source.write_text(HIP_SOURCE, encoding="utf-8")
            output = Path(tmp) / "scan.json"
            command = [sys.executable, "-B", str(ROOT / "weaveprof/pipeline/scan" / "quick_scan.py"),
                       "--version-root", tmp, "--card", "mi308x", "--arch", "gfx942",
                       "--language", "hip_cpp", "--json", str(output)]
            completed = subprocess.run(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, cwd=str(ROOT))
            self.assertIn(completed.returncode, (0, quick_scan.EXIT_SLOW), completed.stderr.decode("utf-8", "replace"))
            doc = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(doc["target"]["card"], "mi308x")
        self.assertEqual(doc["target"]["platform"], "amd")
        self.assertEqual(doc["target"]["arch"], "gfx942")
        self.assertEqual(doc["language"], "hip_cpp")
        self.assertEqual(doc["recorder"], "recorders/hip_cpp/wp_lite_hip.h")
        self.assertEqual(doc["profiler"]["primary"], "rocprof-sys")
        self.assertEqual(doc["evidence_status"], "measured_on_mi308x_gemm_route")


if __name__ == "__main__":
    unittest.main(verbosity=2)
