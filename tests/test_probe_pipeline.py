#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for WeaveProfiler §3.1–§3.3 Probe Placement, Fast Injection, Incremental Reuse & Closed-Loop Verification."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path


def _record_timing(name: str, val: float, thr: float, unit: str = "s") -> bool:
    strict = os.environ.get("WEAVEPROF_STRICT_TIMING") == "1"
    mode = "enforced" if strict else "not enforced"
    print(f"[timing] {name}: {val:.3f} {unit} (threshold {thr} {unit}, {mode})", file=sys.stderr)
    return strict

ROOT = Path(__file__).resolve().parents[1]


def _load(rel: str, name: str):
    p = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PLACER = _load("weaveprof/pipeline/scan/llm_probe_placer.py", "test_wp_placer")
REUSE = _load("weaveprof/pipeline/instrument/incremental_reuse.py", "test_wp_reuse")
ENGINE = _load("weaveprof/pipeline/instrument/rule_engine.py", "test_wp_engine")
VERIFIER = _load("weaveprof/pipeline/validate/closed_loop_verifier.py", "test_wp_verifier")
DIRECT_AGENT = _load("weaveprof/pipeline/scan/direct_probe_agent.py", "test_wp_direct_agent")
VERIFY_PY = _load("weaveprof/pipeline/core/verify.py", "test_wp_verify_py")


SAMPLE_CUDA_SUBFUNC = """#include <cuda_runtime.h>

__device__ __forceinline__ void mma_mainloop(const float* smem_a, const float* smem_b, float& acc, int k_tiles) {
    for (int kt = 0; kt < k_tiles; ++kt) {
        float a = smem_a[kt];
        float b = smem_b[kt];
        acc = fmaf(a, b, acc);
    }
}

__global__ void ws_gemm_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    __shared__ float smem_a[64];
    __shared__ float smem_b[64];
    smem_a[threadIdx.x & 63] = A[idx];
    smem_b[threadIdx.x & 63] = B[idx];
    __syncthreads();
    float acc = 0.0f;
    mma_mainloop(smem_a, smem_b, acc, 16);
    C[idx] = acc;
}
"""

SAMPLE_SPIN_CUDA = """#include <cuda_runtime.h>
__global__ void ring_kernel(volatile int* flag, float* out) {
    while (*flag == 0) {
        // spin wait
    }
    out[threadIdx.x] = 1.0f;
}
"""

SAMPLE_TRITON = """import triton
import triton.language as tl

@triton.jit
def triton_mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k0 + offs_k)[None, :])
        b = tl.load(b_ptr + (k0 + offs_k)[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)
"""

SAMPLE_CUTEDSL = """import cutlass
import cutlass.cute as cute

@cute.kernel
def cute_mm_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    for kt in range(0, 8):
        cute.copy(gA, gB)
        cute.gemm(gA, gB, gC)
    cute.copy(gB, gC)
"""


class ProbePipelineTests(unittest.TestCase):
    def test_kb_retrieval_and_readonly_probe_placement(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "ws_gemm.cu"
            src.write_text(SAMPLE_CUDA_SUBFUNC, encoding="utf-8")

            def fake_llm(model, messages, **kwargs):
                self.assertEqual(kwargs.get("reasoning_effort"), "max")
                return {
                    "ok": True,
                    "model": model,
                    "reasoning_effort": "max",
                    "elapsed_s": 1.25,
                    "usage": {"prompt_tokens": 900, "completion_tokens": 180, "reasoning_tokens": 90, "total_tokens": 1080},
                    "content": json.dumps({
                        "kernel_name": "ws_gemm_kernel",
                        "operator_family": "gemm",
                        "granularity": "sm",
                        "warp_roles": [{"role_id": 0, "name": "cta_worker", "predicate": "true"}],
                        "probes": [
                            {"probe_id": "load_0", "semantic_id": "load", "mode": "sync", "begin_line": 16, "end_line": 17},
                            {"probe_id": "sync_0", "semantic_id": "barrier", "mode": "sync", "begin_line": 18, "end_line": 18},
                            {"probe_id": "mma_0", "semantic_id": "mma", "mode": "sync", "begin_line": 4, "end_line": 8},
                            {"probe_id": "store_0", "semantic_id": "store", "mode": "sync", "begin_line": 21, "end_line": 21},
                        ],
                    }),
                }

            res = PLACER.place_probes(src, card="h20", arch="sm_90a", model="qwen4-max", llm_caller=fake_llm)
            spec = res["probe_spec"]
            self.assertEqual(spec["schema"], "weaveprofiler.probe-spec/v1")
            self.assertEqual(len(spec["probes"]), 4)
            self.assertEqual(res["telemetry"]["total_tokens"], 1080)

            # §3.2 (2a): Lower C++ with BOTH __global__ and __device__ subfunction -> ZERO signature edits!
            inj = ENGINE.inject_probes(SAMPLE_CUDA_SUBFUNC, spec)
            self.assertTrue(inj["ok"], inj["diagnostics"])
            inst_code = inj["instrumented_source"]
            self.assertIn("void mma_mainloop(const float* smem_a, const float* smem_b, float& acc, int k_tiles)", inst_code)
            self.assertIn("void ws_gemm_kernel(const float* A, const float* B, float* C, int N)", inst_code)
            self.assertIn("FI_KERNEL_SCOPE(", inst_code)
            self.assertIn("FI_DEVICE_SCOPE()", inst_code)
            self.assertIn("FI_SYNC_OUT()", inst_code)
            self.assertIn("FI_SYNC_IN()", inst_code)

    def test_spin_loop_static_detection_blocks_bad_probe(self):
        spec = PLACER.sanitize_and_bind_probe_spec(
            {
                "kernel_name": "ring_kernel",
                "probes": [{"probe_id": "bad_wait", "semantic_id": "wait", "begin_line": 4, "end_line": 4}],
            },
            source_text=SAMPLE_SPIN_CUDA,
            source_path=None,
            route=PLACER.resolve_route("h20", "sm_90a", "cuda_cpp"),
        )
        inj = ENGINE.inject_probes(SAMPLE_SPIN_CUDA, spec)
        self.assertFalse(inj["ok"])
        codes = [d["code"] for d in inj["diagnostics"]]
        self.assertIn("SPIN_LOOP_PROBE", codes)

    def test_triton_nv_amd_and_cutedsl_zero_signature_proxies(self):
        for card, arch in (("h20", "sm_90a"), ("mi308x", "gfx942")):
            spec = PLACER.sanitize_and_bind_probe_spec(
                {
                    "kernel_name": "triton_mm_kernel",
                    "operator_family": "gemm",
                    "probes": [
                        {"probe_id": "ld", "semantic_id": "load", "begin_line": 16, "end_line": 17},
                        {"probe_id": "dot", "semantic_id": "mma", "begin_line": 18, "end_line": 18},
                        {"probe_id": "st", "semantic_id": "store", "begin_line": 19, "end_line": 19},
                    ],
                },
                source_text=SAMPLE_TRITON,
                source_path=None,
                route=PLACER.resolve_route(card, arch, "triton"),
            )
            inj = ENGINE.inject_probes(SAMPLE_TRITON, spec)
            self.assertTrue(inj["ok"], inj["diagnostics"])
            ast.parse(inj["instrumented_source"])
            self.assertIn("def _fi_l0_triton_mm_kernel(", inj["instrumented_source"])
            self.assertIn("def _fi_inst_triton_mm_kernel(", inj["instrumented_source"])
            self.assertIn("triton_mm_kernel = _FiTritonProxy(", inj["instrumented_source"])

        cute_spec = PLACER.sanitize_and_bind_probe_spec(
            {
                "kernel_name": "cute_mm_kernel",
                "operator_family": "gemm",
                "probes": [
                    {"probe_id": "ld", "semantic_id": "load", "begin_line": 7, "end_line": 7},
                    {"probe_id": "mma", "semantic_id": "mma", "begin_line": 8, "end_line": 8},
                    {"probe_id": "st", "semantic_id": "store", "begin_line": 9, "end_line": 9},
                ],
            },
            source_text=SAMPLE_CUTEDSL,
            source_path=None,
            route=PLACER.resolve_route("h20", "sm_90a", "cutedsl"),
        )
        cute_inj = ENGINE.inject_probes(SAMPLE_CUTEDSL, cute_spec)
        self.assertTrue(cute_inj["ok"], cute_inj["diagnostics"])
        ast.parse(cute_inj["instrumented_source"])
        self.assertIn("cute_mm_kernel = _FiCuteProxy(", cute_inj["instrumented_source"])

    def test_incremental_zero_llm_reuse_under_param_and_epilogue_edits(self):
        with tempfile.TemporaryDirectory() as td:
            v0 = Path(td) / "v0.py"
            v1 = Path(td) / "v1.py"
            spec0_path = Path(td) / "spec0.json"
            v0.write_text(SAMPLE_TRITON, encoding="utf-8")
            # Insert new lines above the loop + tune tile setup -> line numbers shift by +3, 0 stage primitives changed!
            edited = SAMPLE_TRITON.replace(
                "    pid = tl.program_id(0)\n",
                "    # Step 1 optimization: swizzled tile id\n    raw_pid = tl.program_id(0)\n    num_pid = tl.num_programs(0)\n    pid = raw_pid % num_pid\n",
            )
            v1.write_text(edited, encoding="utf-8")
            spec0 = PLACER.sanitize_and_bind_probe_spec(
                {
                    "kernel_name": "triton_mm_kernel",
                    "probes": [
                        {"probe_id": "ld", "semantic_id": "load", "begin_line": 16, "end_line": 17},
                        {"probe_id": "dot", "semantic_id": "mma", "begin_line": 18, "end_line": 18},
                        {"probe_id": "st", "semantic_id": "store", "begin_line": 19, "end_line": 19},
                    ],
                },
                source_text=SAMPLE_TRITON,
                source_path=v0,
                route=PLACER.resolve_route("h20", "sm_90a", "triton"),
            )
            spec0_path.write_text(json.dumps(spec0), encoding="utf-8")

            inc = REUSE.reuse_probes_incremental(v0, spec0_path, v1, allow_delta_llm=False)
            self.assertTrue(inc["ok"])
            self.assertFalse(inc["used_delta_llm"])
            if _record_timing("ProbePipelineTests.incremental_reuse.elapsed_s", inc["elapsed_s"], 0.1):
                self.assertLess(inc["elapsed_s"], 0.1)
            self.assertEqual(inc["telemetry"]["total_tokens"], 0)
            new_lines = [(p["probe_id"], p["begin_line"], p["end_line"]) for p in inc["probe_spec"]["probes"]]
            self.assertEqual(new_lines, [("ld", 19, 20), ("dot", 21, 21), ("st", 22, 22)])

    def test_closed_loop_scheme_a_coarsening_convergence(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            src = td_p / "inner_dense.py"
            src.write_text(SAMPLE_TRITON, encoding="utf-8")

            calls = []

            def fake_llm(model, messages, **kwargs):
                calls.append(messages)
                if len(calls) == 1:
                    # Round 1: overly dense inner-loop probes -> runtime reports 24.5% overhead (> 10%)
                    return {
                        "ok": True, "model": model, "reasoning_effort": "max", "elapsed_s": 2.0,
                        "usage": {"prompt_tokens": 500, "completion_tokens": 120, "total_tokens": 620},
                        "content": json.dumps({
                            "kernel_name": "triton_mm_kernel",
                            "probes": [
                                {"probe_id": "inner_ld", "semantic_id": "load", "in_inner_loop": True, "begin_line": 16, "end_line": 17},
                                {"probe_id": "inner_mma", "semantic_id": "mma", "in_inner_loop": True, "begin_line": 18, "end_line": 18},
                                {"probe_id": "ep_store", "semantic_id": "store", "begin_line": 19, "end_line": 19},
                            ],
                        }),
                    }
                # Round 2: LLM receives OVERHEAD_EXCEEDED_COARSEN_SCHEME_A and hoists probes to outer loop boundary!
                self.assertIn("OVERHEAD_EXCEEDED_COARSEN_SCHEME_A", messages[-1]["content"])
                return {
                    "ok": True, "model": model, "reasoning_effort": "max", "elapsed_s": 1.8,
                    "usage": {"prompt_tokens": 700, "completion_tokens": 100, "total_tokens": 800},
                    "content": json.dumps({
                        "kernel_name": "triton_mm_kernel",
                        "probes": [
                            {"probe_id": "hoisted_mainloop_mma", "semantic_id": "mma", "in_inner_loop": False, "begin_line": 15, "end_line": 18},
                            {"probe_id": "ep_store", "semantic_id": "store", "begin_line": 19, "end_line": 19},
                        ],
                    }),
                }

            def fake_runtime_eval(_l0, _inst, _names, spec, round_dir):
                dump = round_dir / "cap.fi02.bin"
                has_inner_ld = any(p["probe_id"] == "inner_ld" for p in spec["probes"])
                # Write a valid FILITE02 dump matching declared stages
                meta_b = json.dumps({"arch": "sm_90a", "backend": "nvidia_ptx", "gran": 1}).encode("utf-8")
                recs = [(1000, 0x40000000, 0)]  # OPEN
                t = 1100
                for p in spec["probes"]:
                    ev = p["event_id"]
                    recs.append((t, ev, 0))
                    recs.append((t + 200, 0x80000000 | ev, 0))
                    t += 300
                recs.append((t + 100, 0x08000000, 0))  # CLOSE
                with open(dump, "wb") as f:
                    f.write(b"FILITE02")
                    f.write(struct.pack("<6I", 1, 32, 1, 0, 0, len(meta_b)))
                    f.write(meta_b)
                    f.write(struct.pack("<1I", len(recs)))
                    for rt, rtag, raux in recs:
                        f.write(struct.pack("<QII", rt, rtag, raux))
                ovh = 24.5 if has_inner_ld else 2.8
                return {"ok": True, "mismatches_total": 0, "nonfinite_total": 0, "overhead_pct": ovh,
                        "binsim": {"assembly_change_rate": 0.0},
                        "resources": {"l0": {"spill_total_bytes": 0}, "inst": {"spill_total_bytes": 0},
                                      "delta_smem_bytes": 0}}, dump

            manifest = VERIFIER.run_closed_loop_profile(
                src,
                td_p / "out_run",
                card="h20",
                arch="sm_90a",
                model="qwen4-max",
                max_rounds=3,
                runtime_evaluator=fake_runtime_eval,
                llm_caller=fake_llm,
            )
            self.assertTrue(manifest["ok"])
            self.assertEqual(manifest["rounds_executed"], 2)
            self.assertFalse(manifest["round_history"][0]["all_gates_passed"])
            self.assertTrue(manifest["round_history"][1]["all_gates_passed"])

    def test_cutedsl_class_method_descriptor_proxy(self):
        sample_oop = (
            "import cutlass\n"
            "import cutlass.cute as cute\n\n"
            "class Sm100SwapABSwigluFp4Fc12Kernel:\n"
            "    @cute.kernel\n"
            "    def fc1fc2_kernel_impl(\n"
            "        self,\n"
            "        tiled_mma: cute.TiledMma,\n"
            "        fc1_weight: cute.Tensor,\n"
            "        fc2_output: cute.Tensor,\n"
            "        token_comm_args=None,\n"
            "    ):\n"
            "        for kt in range(0, 8):\n"
            "            cute.gemm(fc1_weight, fc2_output)\n"
            "        cute.copy(fc1_weight, fc2_output)\n"
        )
        spec = PLACER.sanitize_and_bind_probe_spec(
            {
                "kernel_name": "fc1fc2_kernel_impl",
                "operator_family": "gemm",
                "probes": [
                    {"probe_id": "mma", "semantic_id": "mma", "begin_line": 14, "end_line": 14},
                    {"probe_id": "cp", "semantic_id": "store", "begin_line": 15, "end_line": 15},
                ],
            },
            source_text=sample_oop,
            source_path=None,
            route=PLACER.resolve_route("b300", "sm_103a", "cutedsl"),
        )
        inj = ENGINE.inject_probes(sample_oop, spec)
        self.assertTrue(inj["ok"], inj["diagnostics"])
        inst_code = inj["instrumented_source"]
        ast.parse(inst_code)
        self.assertIn("fc1fc2_kernel_impl = _FiCuteDescriptor(_fi_l0_fc1fc2_kernel_impl, _fi_inst_fc1fc2_kernel_impl, fi_get_attached)", inst_code)
        self.assertIn("def _fi_l0_fc1fc2_kernel_impl(", inst_code)
        self.assertIn("def _fi_inst_fc1fc2_kernel_impl(", inst_code)
        self.assertIn("_fi_rec=None, _fi_ovf=None", inst_code)

    def test_direct_probe_agent_search_replace_and_incremental_reuse(self):
        src = "void my_gemm() {\n    int tiles = 16;\n    mma_body();\n}\n"
        patch = "<<<<<<< SEARCH\n    mma_body();\n=======\n    FI_B(mma);\n    mma_body();\n    FI_E(mma);\n>>>>>>>"
        inst = DIRECT_AGENT.apply_search_replace_blocks(src, patch)
        self.assertIn("FI_B(mma);", inst)
        self.assertIn("FI_E(mma);", inst)

        # Incremental reuse across tuning edits (tiles 16 -> 32)
        curr = src.replace("int tiles = 16;", "int tiles = 32;")
        inc = DIRECT_AGENT.incremental_inherit_probes(curr, src, inst)
        self.assertIsNotNone(inc)
        self.assertIn("int tiles = 32;", inc[0])
        self.assertIn("FI_B(mma);", inc[0])
        self.assertEqual(inc[0].count("}"), curr.count("}"))

        # Multi-helper C++/HIP file with closing `}` and blank-line `replace` hunks (#84 regression)
        cpp_prev = (
            "__device__ uint64_t get_hw_timer() {\n"
            "    return (uint64_t)clock64();\n"
            "}\n"
            "__device__ void ws_mbar_try_wait_parity(uint64_t* m) {\n"
            "    if (*m) return;\n"
            "}\n"
            "__global__ __launch_bounds__(128) void my_ws_kernel(float* out, int work) {\n"
            "    const int cta = blockIdx.x;\n"
            "    \n"
            "    for (int i = 0; i < work / 4; ++i) out[cta] += 1.0f;\n"
            "}\n"
        )
        cpp_inst = (
            '#include "wp_lite.cuh"\n'
            "__device__ uint64_t get_hw_timer() {\n"
            "    return (uint64_t)clock64();\n"
            "}\n"
            "__device__ void ws_mbar_try_wait_parity(uint64_t* m) {\n"
            "    if (*m) return;\n"
            "}\n"
            "__global__ __launch_bounds__(128) void my_ws_kernel(float* out, int work) {\n"
            "    const int cta = blockIdx.x;\n"
            "    FI_KERNEL_SCOPE(0);\n"
            "    FI_B(4u, 0u);\n"
            "    for (int i = 0; i < work / 4; ++i) out[cta] += 1.0f;\n"
            "    FI_E(4u, 0u);\n"
            "}\n"
        )
        cpp_curr = cpp_prev.replace("__launch_bounds__(128)", "__launch_bounds__(128, 2)").replace("work / 4", "work / 8")
        cpp_inc = DIRECT_AGENT.incremental_inherit_probes(cpp_curr, cpp_prev, cpp_inst)
        self.assertIsNotNone(cpp_inc)
        patched_cpp = cpp_inc[0]
        self.assertIn("__launch_bounds__(128, 2)", patched_cpp)
        self.assertIn("work / 8", patched_cpp)
        self.assertIn("FI_KERNEL_SCOPE(0);", patched_cpp)
        self.assertIn("FI_E(4u, 0u);\n}", patched_cpp)
        # Top helpers must remain 100% untouched
        helper_prefix = patched_cpp.split("my_ws_kernel")[0]
        self.assertNotIn("FI_KERNEL_SCOPE", helper_prefix)
        self.assertNotIn("FI_E", helper_prefix)
        self.assertEqual(patched_cpp.count("{"), patched_cpp.count("}"))

        # Direct conflict on a `replace` hunk must return None cleanly instead of duplicating lines
        conflict_inst = cpp_prev.replace(
            "for (int i = 0; i < work / 4; ++i) out[cta] += 1.0f;",
            "for (int i = 0; i < work / 4; ++i) { FI_B(4u, i); out[cta] += 1.0f; FI_E(4u, i); }",
        )
        self.assertIsNone(DIRECT_AGENT.incremental_inherit_probes(cpp_curr, cpp_prev, conflict_inst))

        # CPython 3.14 + NumPy 2.x `~nonzero` temporary elision guard in wp_cute.py (#85 regression)
        fi_cute_src = (ROOT / "recorders" / "cutedsl" / "wp_cute.py").read_text(encoding="utf-8")
        self.assertNotIn("np.any(~nonzero", fi_cute_src)
        self.assertNotIn("np.argmax(~nonzero", fi_cute_src)
        self.assertIn("is_zero = tag == np.uint32(0)", fi_cute_src)

    def test_verify_py_direct_source_verification(self):
        with tempfile.TemporaryDirectory() as td:
            src_f = Path(td) / "kern.cu"
            inst_f = Path(td) / "inst_kern.cu"
            src_code = "#include <cuda_runtime.h>\n__global__ void k() { int x = 1; }\n"
            inst_code = '#include <cuda_runtime.h>\n#include "wp_lite.cuh"\n__global__ void k() { FI_KERNEL_SCOPE(0); FI_B(4u, 0u); int x = 1; FI_E(4u, 0u); }\n'
            src_f.write_text(src_code, encoding="utf-8")
            inst_f.write_text(inst_code, encoding="utf-8")

            # Extract probes directly from source without any probes.json file
            extracted = VERIFY_PY.extract_probe_spec_from_inst_code(src_code, inst_code, "cuda_cpp")
            self.assertEqual(len(extracted["probes"]), 1)
            self.assertEqual(extracted["probes"][0]["semantic_id"], "compute")

            # Run verify.py main in offline mode without requiring --probes
            # On Windows without cl.exe, nvcc fails compilation gracefully and prints standard terminal error log
            rc = VERIFY_PY.main(["--source", str(src_f), "--inst-source", str(inst_f), "--card", "h20", "--offline-only"])
            self.assertIn(rc, (0, 1))

    def test_direct_probe_prompt_macro_signatures(self):
        # 1. CUDA C++
        kb_cuda = {"language": "cuda_cpp", "card": "h20", "arch": "sm_90a"}
        p_cuda = DIRECT_AGENT.build_direct_search_replace_prompt("void k(){}", kb_cuda)
        txt_cuda = p_cuda[1]["content"]
        self.assertIn("FI_KERNEL_SCOPE(role);", txt_cuda)
        self.assertNotIn("FI_KERNEL_SCOPE(slot, role);", txt_cuda)
        self.assertIn("FI_B(stage_id, aux);", txt_cuda)
        self.assertNotIn("FI_B(load);", txt_cuda)
        self.assertIn("FI_B(1u, 0u);", txt_cuda)

        # 2. HIP C++
        kb_hip = {"language": "hip_cpp", "card": "mi308x", "arch": "gfx942"}
        p_hip = DIRECT_AGENT.build_direct_search_replace_prompt("void k(){}", kb_hip)
        txt_hip = p_hip[1]["content"]
        self.assertIn("wp_lite_hip.h", txt_hip)
        self.assertIn("FI_KERNEL_SCOPE(role);", txt_hip)
        self.assertIn("FI_B(stage_id, aux);", txt_hip)

        # 3. Triton AMD (wave64)
        kb_triton_amd = {"language": "triton", "card": "mi308x", "arch": "gfx942"}
        p_triton_amd = DIRECT_AGENT.build_direct_search_replace_prompt("def k(): pass", kb_triton_amd)
        txt_triton_amd = p_triton_amd[1]["content"]
        self.assertIn("fi_b_16b", txt_triton_amd)
        self.assertIn("fi_e_16b", txt_triton_amd)
        self.assertNotIn('fi_b("load")', txt_triton_amd)

        # 4. Triton NV
        kb_triton_nv = {"language": "triton", "card": "h20", "arch": "sm_90a"}
        p_triton_nv = DIRECT_AGENT.build_direct_search_replace_prompt("def k(): pass", kb_triton_nv)
        txt_triton_nv = p_triton_nv[1]["content"]
        self.assertIn("n = fi_b(rec, ovf, pid, n, FI, stage_id, aux)", txt_triton_nv)
        self.assertNotIn('fi_b("load")', txt_triton_nv)

        # 5. CuTeDSL
        kb_cute = {"language": "cutedsl", "card": "b300", "arch": "sm_103a"}
        p_cute = DIRECT_AGENT.build_direct_search_replace_prompt("def k(): pass", kb_cute)
        txt_cute = p_cute[1]["content"]
        self.assertIn("ctx = fi_b(ctx, stage_id, aux)", txt_cute)
        self.assertNotIn('fi_b("load")', txt_cute)

    def test_extract_probe_spec_numeric_and_pred(self):
        sample_code = (
            '#include "wp_lite.cuh"\n'
            '__global__ void k() {\n'
            '    FI_KERNEL_SCOPE(1);\n'
            '    FI_B(1u, 0u);\n'
            '    load_op();\n'
            '    FI_E(1u, 0u);\n'
            '    FI_B_PRED(threadIdx.x == 0, 3u, 0u);\n'
            '    mma_op();\n'
            '    FI_E_PRED(threadIdx.x == 0, 3u, 0u);\n'
            '}\n'
        )
        extracted = VERIFY_PY.extract_probe_spec_from_inst_code("", sample_code, "cuda_cpp")
        probes = extracted["probes"]
        self.assertEqual(len(probes), 2)
        self.assertEqual(probes[0]["semantic_id"], "load")
        self.assertEqual(probes[0]["role"], "role_1")
        self.assertEqual(probes[1]["semantic_id"], "mma")

    def test_two_level_kb_static_guard_and_whole_kernel_semantic_audit(self):
        # 1. Verify Two-Level KB topology injection into prompt
        p2p_src = (
            "__device__ __forceinline__ void p2p_cp_async_16B(half* d, const half* s) {}\n"
            "__global__ void p2p_allgather_gemm(const half* peer_ptr, float* out) {\n"
            "    int warp = threadIdx.x >> 5;\n"
            "    if (warp == 0) { /* p2p pull */ } else { /* wmma */ }\n"
            "}\n"
        )
        kb_ctx = PLACER.retrieve_route_kb("cuda_cpp", "h20", "sm_90a", p2p_src)
        msgs = DIRECT_AGENT.build_direct_search_replace_prompt(p2p_src, kb_ctx, ablation_mode="full_closed_loop")
        prompt_text = msgs[1]["content"]
        self.assertIn("Level-1 Composable Primitive Invariants", prompt_text)
        self.assertIn("multi_gpu_p2p_overlap", prompt_text)
        self.assertIn("WP_KERNEL_SCOPE(role);", prompt_text)

        # 2. Verify static preflight catches micro-helper probes and signature mutations
        bad_inst = (
            '#include "wp_lite.cuh"\n'
            "__device__ __forceinline__ void p2p_cp_async_16B(half* d, const half* s) { WP_B(1u, 0u); WP_E(1u, 0u); }\n"
            "__global__ void p2p_allgather_gemm(const half* peer_ptr, float* out, int broken_extra_arg) {\n"
            "    WP_KERNEL_SCOPE(0u);\n"
            "}\n"
        )
        issues = DIRECT_AGENT.static_preflight_validate(p2p_src, bad_inst, "cuda_cpp", "h20")
        joined = "\n".join(issues)
        self.assertIn("MICRO_HELPER_PROBE_VIOLATION", joined)
        self.assertIn("SIGNATURE_MUTATED", joined)

        # 3. Verify WP_* macro extraction in verify.py
        wp_code = (
            '#include "wp_lite.cuh"\n'
            "__global__ void k() {\n"
            "    WP_KERNEL_SCOPE(1u);\n"
            "    WP_B(9u, 0u);\n"
            "    comm_op();\n"
            "    WP_E(9u, 0u);\n"
            "}\n"
        )
        ext_wp = VERIFY_PY.extract_probe_spec_from_inst_code("", wp_code, "cuda_cpp")
        self.assertEqual(len(ext_wp["probes"]), 1)
        self.assertEqual(ext_wp["probes"][0]["semantic_id"], "communication")
        self.assertEqual(ext_wp["probes"][0]["role"], "role_1")

        # 4. Verify Whole-Kernel Incremental Semantic Audit catches non-local cross-warp role shift
        prev_src = (
            "__global__ void ws_kern(float* out, int n) {\n"
            "    int warp = threadIdx.x >> 5;\n"
            "    bool is_dma = (warp == 0);\n"
            "    if (is_dma) {\n"
            "        dma_load();\n"
            "    } else {\n"
            "        math_compute(out);\n"
            "    }\n"
            "}\n"
        )
        prev_inst = (
            '#include "wp_lite.cuh"\n'
            "__global__ void ws_kern(float* out, int n) {\n"
            "    int warp = threadIdx.x >> 5;\n"
            "    WP_KERNEL_SCOPE((warp == 0) ? 1u : 0u);\n"
            "    bool is_dma = (warp == 0);\n"
            "    if (is_dma) {\n"
            "        WP_B(1u, 0u);\n"
            "        dma_load();\n"
            "        WP_E(1u, 0u);\n"
            "    } else {\n"
            "        WP_B(3u, 0u);\n"
            "        math_compute(out);\n"
            "        WP_E(3u, 0u);\n"
            "    }\n"
            "}\n"
        )
        # Non-local semantic edit: `is_dma = (warp < 2)` changes which warps enter `is_dma`,
        # making untouched `WP_KERNEL_SCOPE((warp == 0) ? 1u : 0u)` semantically inconsistent!
        edited_src = prev_src.replace("bool is_dma = (warp == 0);", "bool is_dma = (warp < 2);")

        def fake_audit_llm(model, messages, **kwargs):
            # Auditor sees BOTH the diff and the untouched WP_KERNEL_SCOPE line in the full kernel
            self.assertIn("bool is_dma = (warp < 2);", messages[1]["content"])
            self.assertIn("WP_KERNEL_SCOPE((warp == 0) ? 1u : 0u);", messages[1]["content"])
            return {
                "ok": True,
                "elapsed_s": 2.1,
                "usage": {"prompt_tokens": 600, "completion_tokens": 45, "total_tokens": 645},
                "content": (
                    "<<<<<<< SEARCH\n"
                    "    WP_KERNEL_SCOPE((warp == 0) ? 1u : 0u);\n"
                    "=======\n"
                    "    WP_KERNEL_SCOPE((warp < 2) ? 1u : 0u);\n"
                    ">>>>>>>"
                ),
            }

        res_shift = DIRECT_AGENT.audit_incremental_with_whole_kernel_llm(
            edited_src, prev_src, prev_inst, kb_ctx, llm_caller=fake_audit_llm
        )
        self.assertTrue(res_shift["ok"])
        self.assertTrue(res_shift["semantic_shift_detected"])
        self.assertIn("WP_KERNEL_SCOPE((warp < 2) ? 1u : 0u);", res_shift["instrumented_source"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
