#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zero-manual-glue automatic GEMM instrumentation, validation, and SW+HW profiler.

Workflow (strictly obeys 0-GPU prep / 1-GPU resident batching):
  1. ``prep`` (0 GPUs, pure CPU):
     - Runs ``weaveprof/pipeline/scan/auto_gemm_plan.py`` on an uninstrumented CUDA C++, HIP C++,
       Triton, or CuTeDSL GEMM file -> emits ``instrumented_<src>``, ``names.json``,
       and ``auto_plan.json``.
     - For C++ (``cuda_cpp`` / ``hip_cpp``), compiles 3 shared libraries
       (``L0`` off, ``sample1`` first-tile fast mode, ``full`` full-K audit mode)
       with C-ABI host wrappers and runs C1 static binary similarity & resource
       perturbation checks (``fi_binsim.py`` / ``fi_binsim_hip.py``).
  2. ``gpu`` (1 GPU, single resident process for C2 + overhead, plus optional
     same-run co-axial SW+HW timeline capture):
     - Allocates ``torch.randn`` inputs, computes exact FP32 reference ``A.float() @ B.float()``,
       verifies every output element of ``sample1`` and ``full`` against ``L0`` and FP32 ref (C2),
       measures steady-state ``L0`` vs ``sample1`` vs ``full`` overhead & single-GPU TFLOPS,
       dumps ``FILITE02`` binaries, decodes Perfetto SW traces (C3), and optionally runs
       same-run co-axial ``nsys`` (200 kHz GSP) or ``rocprof-sys-run --rocm-events`` SW+HW merge (C4).
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import importlib.util
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SKILL_ROOT = Path(__file__).resolve().parents[3]
PLANNER_PATH = SKILL_ROOT / "weaveprof/pipeline/scan" / "auto_gemm_plan.py"
DECODER_PATH = SKILL_ROOT / "weaveprof/pipeline/validate" / "tools" / "fi_lite_decode.py"
ALIGN_NSYS_PATH = SKILL_ROOT / "weaveprof/pipeline/validate" / "tools" / "fi_align_nsys.py"
ATTRIB_PATH = SKILL_ROOT / "weaveprof/pipeline/validate" / "tools" / "fi_attrib.py"
BINSIM_NV_PATH = SKILL_ROOT / "weaveprof/pipeline/validate" / "tools" / "fi_binsim.py"
BINSIM_HIP_PATH = SKILL_ROOT / "weaveprof/pipeline/validate" / "tools" / "fi_binsim_hip.py"
TRACE_GATE_PATH = SKILL_ROOT / "weaveprof/pipeline/validate" / "tools" / "fi_trace_gate.py"
ROCPROF_SYS_MERGE_PATH = SKILL_ROOT / "weaveprof/pipeline/capture" / "build_rocprof_sys_sw_hw.py"


def _write_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    _write_lf(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _run(cmd: Sequence[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None,
         timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    merged_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        merged_env.update(env)
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Python module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _infer_cpp_launch_geometry(src: str) -> Tuple[int, int, int]:
    """Infer (BM, BN, THREADS) from constexpr/define declarations in C++ GEMM kernel."""
    bm_m = re.search(r"\b(?:BM|BLOCK_M|TILE_M)\s*=\s*(\d+)", src)
    bn_m = re.search(r"\b(?:BN|BLOCK_N|TILE_N)\s*=\s*(\d+)", src)
    th_m = re.search(r"\b(?:THREADS|BLOCK_THREADS|NUM_THREADS)\s*=\s*(\d+)", src)
    if not th_m:
        th_m = re.search(r"__launch_bounds__\s*\(\s*(\d+)", src)
    bm = int(bm_m.group(1)) if bm_m else 64
    bn = int(bn_m.group(1)) if bn_m else 64
    threads = int(th_m.group(1)) if th_m else 128
    return bm, bn, threads


def _host_lacks_crt_startfiles() -> bool:
    """Return True on minimal Linux containers where crti.o / crtbeginS.o are absent."""
    if os.name == "nt":
        return False
    cc = shutil.which("gcc") or shutil.which("cc")
    if not cc:
        return False
    r = _run([cc, "-print-file-name=crti.o"], timeout=10.0)
    out = (r.stdout or "").strip()
    return not (out and "/" in out and Path(out).is_file())


def _ensure_cc_shim_if_needed(work_dir: Path) -> Optional[str]:
    """Configure CC wrapper with -nostartfiles on minimal containers lacking crti.o."""
    if not _host_lacks_crt_startfiles():
        return None
    cc = shutil.which("gcc") or shutil.which("cc")
    if not cc:
        return None
    shim = work_dir / "_cc_shim" / "gcc_nostartfiles.sh"
    _write_lf(shim, f"#!/bin/sh\nexec {cc} -nostartfiles \"$@\"\n")
    try:
        os.chmod(shim, 0o755)
    except OSError:
        pass
    os.environ["CC"] = str(shim)
    return str(shim)


def _build_cpp_harness_source(
    instrumented_src: str,
    *,
    language: str,
    kernel_name: str,
    bm: int,
    bn: int,
    threads: int,
    arch: str,
    define_dso_handle: bool = False,
) -> str:
    """Append an extern-C shared-library runner to the instrumented CUDA/HIP GEMM source."""
    dso_shim = (
        'extern "C" { void* __dso_handle __attribute__((visibility("hidden"), weak)) = &__dso_handle; }\n'
        if define_dso_handle
        else ""
    )
    if language == "cuda_cpp":
        a_type = "const __half*"
        extra_inc = "#include <cuda_profiler_api.h>\n"
        launch_body = f"""
extern "C" int wp_run_gemm(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    int warmup, int iters,
    int cap_per_lane,
    double* out_median_us,
    const char* dump_path,
    int use_profiler_range) {{
    dim3 block({threads}, 1, 1);
    dim3 grid((N + {bn} - 1) / {bn}, (M + {bm} - 1) / {bm}, 1);
    int warps_per_cta = ({threads} + 31) / 32;
    uint32_t total_ctas = grid.x * grid.y * grid.z;
    FiBuf fi_buf;
    memset(&fi_buf, 0, sizeof(fi_buf));
    if (FI_LEVEL >= 1 && cap_per_lane > 0) {{
        cudaError_t e_a = fi_alloc(&fi_buf, total_ctas, (uint32_t)warps_per_cta, (uint32_t)cap_per_lane);
        if (e_a != cudaSuccess) return (int)e_a;
        e_a = fi_attach(&fi_buf);
        if (e_a != cudaSuccess) {{ fi_free(&fi_buf); return (int)e_a; }}
    }}
    const bool want_dump = (dump_path != nullptr && dump_path[0] != '\\0');
    for (int w = 0; w < warmup; ++w) {{
        {kernel_name}<<<grid, block>>>(({a_type})A, ({a_type})B, (float*)C, M, N, K);
    }}
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {{
        if (FI_LEVEL >= 1 && cap_per_lane > 0) fi_free(&fi_buf);
        return (int)err;
    }}

    if (use_profiler_range) {{ cudaError_t ep_ = cudaProfilerStart(); (void)ep_; }}
    std::vector<double> samples;
    samples.reserve(iters > 0 ? iters : 1);
    cudaEvent_t ev0, ev1;
    (void)cudaEventCreate(&ev0);
    (void)cudaEventCreate(&ev1);
    for (int it = 0; it < iters; ++it) {{
        if (FI_LEVEL >= 1 && cap_per_lane > 0 && want_dump && it == iters - 1) {{
            cudaError_t er_ = fi_reset(&fi_buf);
            (void)er_;
            (void)cudaDeviceSynchronize();
        }}
        (void)cudaEventRecord(ev0);
        {kernel_name}<<<grid, block>>>(({a_type})A, ({a_type})B, (float*)C, M, N, K);
        (void)cudaEventRecord(ev1);
        err = cudaEventSynchronize(ev1);
        if (err != cudaSuccess) {{
            if (use_profiler_range) {{ cudaError_t ep_ = cudaProfilerStop(); (void)ep_; }}
            (void)cudaEventDestroy(ev0);
            (void)cudaEventDestroy(ev1);
            if (FI_LEVEL >= 1 && cap_per_lane > 0) fi_free(&fi_buf);
            return (int)err;
        }}
        float ms = 0.0f;
        (void)cudaEventElapsedTime(&ms, ev0, ev1);
        samples.push_back((double)ms * 1000.0);
    }}
    if (use_profiler_range) {{ cudaError_t ep_ = cudaProfilerStop(); (void)ep_; }}
    (void)cudaEventDestroy(ev0);
    (void)cudaEventDestroy(ev1);
    if (!samples.empty() && out_median_us) {{
        std::sort(samples.begin(), samples.end());
        *out_median_us = samples[samples.size() / 2];
    }}
    if (FI_LEVEL >= 1 && cap_per_lane > 0 && dump_path && dump_path[0]) {{
        std::string meta = std::string("{{\\"arch\\":\\"{arch}\\",\\"backend\\":\\"cuda_cpp\\",\\"kernel\\":\\"{kernel_name}\\"}}");
        int drc_ = fi_dump(&fi_buf, dump_path, meta.c_str());
        (void)drc_;
    }}
    if (FI_LEVEL >= 1 && cap_per_lane > 0) fi_free(&fi_buf);
    return 0;
}}
"""
    else:
        a_type = "const _Float16*"
        extra_inc = ""
        launch_body = f"""
extern "C" int wp_run_gemm(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    int warmup, int iters,
    int cap_per_lane,
    double* out_median_us,
    const char* dump_path,
    int use_profiler_range) {{
    (void)use_profiler_range;
    dim3 block({threads}, 1, 1);
    dim3 grid((N + {bn} - 1) / {bn}, (M + {bm} - 1) / {bm}, 1);
    int waves_per_cta = ({threads} + 63) / 64;
    uint32_t total_ctas = grid.x * grid.y * grid.z;
    FiBuf fi_buf;
    memset(&fi_buf, 0, sizeof(fi_buf));
    if (FI_LEVEL >= 1 && cap_per_lane > 0) {{
        hipError_t e_a = fi_alloc(&fi_buf, total_ctas, (uint32_t)waves_per_cta, (uint32_t)cap_per_lane);
        if (e_a != hipSuccess) return (int)e_a;
        e_a = fi_attach(&fi_buf);
        if (e_a != hipSuccess) {{ fi_free(&fi_buf); return (int)e_a; }}
    }}
    const bool want_dump = (dump_path != nullptr && dump_path[0] != '\\0');
    for (int w = 0; w < warmup; ++w) {{
        hipLaunchKernelGGL({kernel_name}, grid, block, 0, 0, ({a_type})A, ({a_type})B, (float*)C, M, N, K);
    }}
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {{
        if (FI_LEVEL >= 1 && cap_per_lane > 0) fi_free(&fi_buf);
        return (int)err;
    }}

    std::vector<double> samples;
    samples.reserve(iters > 0 ? iters : 1);
    hipEvent_t ev0, ev1;
    hipError_t he0_ = hipEventCreate(&ev0); (void)he0_;
    hipError_t he1_ = hipEventCreate(&ev1); (void)he1_;
    for (int it = 0; it < iters; ++it) {{
        if (FI_LEVEL >= 1 && cap_per_lane > 0 && want_dump && it == iters - 1) {{
            hipError_t er_ = fi_reset(&fi_buf);
            (void)er_;
            (void)hipDeviceSynchronize();
        }}
        hipError_t hr0_ = hipEventRecord(ev0); (void)hr0_;
        hipLaunchKernelGGL({kernel_name}, grid, block, 0, 0, ({a_type})A, ({a_type})B, (float*)C, M, N, K);
        hipError_t hr1_ = hipEventRecord(ev1); (void)hr1_;
        err = hipEventSynchronize(ev1);
        if (err != hipSuccess) {{
            hipError_t hd0_ = hipEventDestroy(ev0); (void)hd0_;
            hipError_t hd1_ = hipEventDestroy(ev1); (void)hd1_;
            if (FI_LEVEL >= 1 && cap_per_lane > 0) fi_free(&fi_buf);
            return (int)err;
        }}
        float ms = 0.0f;
        hipError_t het_ = hipEventElapsedTime(&ms, ev0, ev1); (void)het_;
        samples.push_back((double)ms * 1000.0);
    }}
    hipError_t hd0_ = hipEventDestroy(ev0); (void)hd0_;
    hipError_t hd1_ = hipEventDestroy(ev1); (void)hd1_;
    if (!samples.empty() && out_median_us) {{
        std::sort(samples.begin(), samples.end());
        *out_median_us = samples[samples.size() / 2];
    }}
    if (FI_LEVEL >= 1 && cap_per_lane > 0 && dump_path && dump_path[0]) {{
        std::string meta = std::string("{{\\"arch\\":\\"{arch}\\",\\"backend\\":\\"hip_cpp\\",\\"kernel\\":\\"{kernel_name}\\"}}");
        int drc_ = fi_dump(&fi_buf, dump_path, meta.c_str());
        (void)drc_;
    }}
    if (FI_LEVEL >= 1 && cap_per_lane > 0) fi_free(&fi_buf);
    return 0;
}}
"""
    return (
        extra_inc
        + dso_shim
        + "#include <algorithm>\n#include <string>\n#include <vector>\n"
        + instrumented_src
        + "\n"
        + launch_body
    )


def run_prep_phase(
    source_path: Path,
    out_dir: Path,
    *,
    card: str,
    arch: Optional[str],
    kernel_name: Optional[str],
) -> Dict[str, Any]:
    """0-GPU CPU preparation: plan, instrument, compile C++ variants, and run static C1."""
    planner = _load_module(PLANNER_PATH, "wp_auto_gemm_plan")
    plan = planner.plan_and_instrument(
        source_path, out_dir, card=card, arch=arch, kernel_name=kernel_name
    )
    resolved_arch = plan["arch"]
    language = plan["language"]
    inst_path = Path(plan["instrumented_source"])
    inst_src = inst_path.read_text(encoding="utf-8")

    prep_info: Dict[str, Any] = {
        "schema": "weaveprofiler.auto-gemm-prep/v1",
        "plan": plan,
        "c1_static": None,
        "cpp_builds": None,
    }

    if language in ("cuda_cpp", "hip_cpp"):
        bm, bn, threads = _infer_cpp_launch_geometry(inst_src)
        build_dir = out_dir / "build_cpp"
        build_dir.mkdir(parents=True, exist_ok=True)
        no_crt = (language == "cuda_cpp") and _host_lacks_crt_startfiles()
        harness_ext = ".cu" if language == "cuda_cpp" else ".cpp"
        harness_src_path = build_dir / f"wrapped_{source_path.stem}{harness_ext}"
        _write_lf(
            harness_src_path,
            _build_cpp_harness_source(
                inst_src,
                language=language,
                kernel_name=plan["kernel"],
                bm=bm,
                bn=bn,
                threads=threads,
                arch=resolved_arch,
                define_dso_handle=no_crt,
            ),
        )

        fi_gran = int(plan.get("gran", 1))
        variants = {
            "L0": ["-DFI_LEVEL=0", f"-DFI_GRAN={fi_gran}", "-DFI_TIMER=1", "-DFI_SAMPLE=0"],
            "sample1": ["-DFI_LEVEL=2", f"-DFI_GRAN={fi_gran}", "-DFI_TIMER=1", "-DFI_SAMPLE=1"],
            "full": ["-DFI_LEVEL=2", f"-DFI_GRAN={fi_gran}", "-DFI_TIMER=1", "-DFI_SAMPLE=0"],
            "stride4": ["-DFI_LEVEL=2", f"-DFI_GRAN={fi_gran}", "-DFI_TIMER=1", "-DFI_SAMPLE=4"],
            "stride8": ["-DFI_LEVEL=2", f"-DFI_GRAN={fi_gran}", "-DFI_TIMER=1", "-DFI_SAMPLE=8"],
        }
        built: Dict[str, Any] = {
            "bm": bm,
            "bn": bn,
            "threads": threads,
            "fi_gran": fi_gran,
            "warp_specialized": bool(plan.get("warp_specialized", False)),
            "no_crt_startfiles": no_crt,
            "variants": {},
        }
        t_build0 = time.monotonic()
        for label, defs in variants.items():
            so_path = build_dir / f"libgemm_{label}.so"
            log_path = build_dir / f"build_{label}.log"
            variant_cwd = build_dir
            if language == "cuda_cpp":
                nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
                sm_num = resolved_arch.replace("sm_", "")
                inc_dir = SKILL_ROOT / "recorders" / "cuda_cpp"
                extra_nvcc = ["-Xcompiler", "-nostartfiles"] if no_crt else []
                cmd = [
                    nvcc, "-O3", "-std=c++17", "--shared",
                    "-Xcompiler", "-fPIC",
                    *extra_nvcc,
                    "-gencode", f"arch=compute_{sm_num},code={resolved_arch}",
                    "-Xptxas", "-v",
                    "-I", str(inc_dir),
                    *defs,
                    str(harness_src_path),
                    "-o", str(so_path),
                ]
            else:
                hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
                inc_dir = SKILL_ROOT / "recorders" / "hip_cpp"
                variant_cwd = build_dir / f"temps_{label}"
                variant_cwd.mkdir(parents=True, exist_ok=True)
                cmd = [
                    hipcc, "-O3", "-std=c++17", "-shared", "-fPIC",
                    "--save-temps",
                    f"--offload-arch={resolved_arch}",
                    "-Rpass-analysis=kernel-resource-usage",
                    "-I", str(inc_dir),
                    *defs,
                    str(harness_src_path),
                    "-o", str(so_path),
                ]
            res = _run(cmd, cwd=variant_cwd, timeout=300.0)
            _write_lf(log_path, (res.stdout or "") + "\n" + (res.stderr or ""))
            if res.returncode != 0:
                raise RuntimeError(f"C++ build failed for {label} (rc={res.returncode}):\n{res.stderr}\n{res.stdout}")
            asm_s = None
            if language == "hip_cpp":
                s_files = sorted(variant_cwd.glob(f"*{resolved_arch}*.s"))
                if s_files:
                    asm_s = str(s_files[0])
            built["variants"][label] = {"so": str(so_path), "log": str(log_path), "asm_s": asm_s}
        built["build_elapsed_s"] = round(time.monotonic() - t_build0, 3)
        prep_info["cpp_builds"] = built

        # Run C1 static ISA similarity check (0 GPUs!)
        binsim_out = out_dir / "c1_binsim_full.json"
        binsim_s1_out = out_dir / "c1_binsim_sample1.json"
        for b_label, b_out in (("full", binsim_out), ("sample1", binsim_s1_out)):
            if language == "cuda_cpp":
                cmd_bs = [
                    sys.executable, "-B", str(BINSIM_NV_PATH),
                    built["variants"]["L0"]["so"],
                    built["variants"][b_label]["so"],
                    "--kernel", plan["kernel"],
                    "--arch", resolved_arch,
                    "-o", str(b_out),
                ]
            else:
                l0_in = built["variants"]["L0"].get("asm_s") or built["variants"]["L0"]["so"]
                fi_in = built["variants"][b_label].get("asm_s") or built["variants"][b_label]["so"]
                co_wd = build_dir / f"binsim_work_{b_label}"
                co_wd.mkdir(parents=True, exist_ok=True)
                cmd_bs = [
                    sys.executable, "-B", str(BINSIM_HIP_PATH),
                    "--l0", l0_in,
                    "--l2", fi_in,
                    "--kernel", plan["kernel"],
                    "--gfx", resolved_arch,
                    "--work-dir", str(co_wd),
                    "-o", str(b_out),
                ]
            bs_res = _run(cmd_bs, cwd=out_dir, timeout=180.0)
            if bs_res.returncode == 0 and b_out.is_file():
                built["variants"][b_label]["binsim"] = json.loads(b_out.read_text(encoding="utf-8"))
            else:
                built["variants"][b_label]["binsim_error"] = (bs_res.stderr or bs_res.stdout)[-500:]
        prep_info["c1_static"] = {
            "sample1": built["variants"]["sample1"].get("binsim"),
            "full": built["variants"]["full"].get("binsim"),
        }

    _write_json(out_dir / "prep_summary.json", prep_info)
    return prep_info


def _decode_dump(dump_path: Path, names_path: Path, out_prefix: Path) -> Dict[str, Any]:
    trace_base = Path(str(out_prefix) + ".perfetto.json")
    summary_path = Path(str(out_prefix) + ".summary.json")
    res = _run(
        [
            sys.executable, "-B", str(DECODER_PATH),
            str(dump_path),
            "--names", str(names_path),
            "--require-semantics",
            "-o", str(trace_base),
            "--summary", str(summary_path),
        ],
        timeout=300.0,
    )
    if res.returncode != 0 or not summary_path.is_file():
        return {"ok": False, "rc": res.returncode, "stderr": (res.stderr or res.stdout)[-800:]}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gz_path = Path(str(trace_base) + ".gz")
    final_trace = gz_path if gz_path.is_file() else trace_base
    gate_mod = _load_module(TRACE_GATE_PATH, "wp_fi_trace_gate")
    gate_res = gate_mod.verify_trace(final_trace, require_hw=False)
    return {
        "ok": (
            not bool(summary.get("errors"))
            and summary.get("semantics_ok") is True
            and int(summary.get("overflow", 0)) == 0
            and bool(gate_res.get("ok"))
        ),
        "trace": str(final_trace),
        "summary_path": str(summary_path),
        "records": summary.get("records"),
        "spans": summary.get("spans"),
        "overflow": summary.get("overflow"),
        "errors": summary.get("errors"),
        "events": {k: v.get("n") for k, v in (summary.get("events") or {}).items() if isinstance(v, dict)},
        "trace_gate": {"ok": gate_res.get("ok"), "errors": gate_res.get("errors"), "spill_count": gate_res.get("spill_count")},
    }


def _check_c2_tensors(c_out, c_l0, c_ref, *, atol: float = 1e-1, rtol: float = 1e-2) -> Dict[str, Any]:
    import torch
    c_f = c_out.float()
    l0_f = c_l0.float()
    total = int(c_f.numel())
    nonfinite = int((~torch.isfinite(c_f)).sum().item())
    max_vs_l0 = float((c_f - l0_f).abs().max().item())
    diff_ref = (c_f - c_ref).abs()
    max_vs_ref = float(diff_ref.max().item())
    tol = atol + rtol * c_ref.abs()
    mismatches = int((diff_ref > tol).sum().item())
    return {
        "pass": (nonfinite == 0 and mismatches == 0 and max_vs_l0 == 0.0),
        "checked_elements": total,
        "nonfinite": nonfinite,
        "max_abs_vs_l0": max_vs_l0,
        "max_abs_vs_fp32_ref": max_vs_ref,
        "mismatches_vs_fp32_ref": mismatches,
    }


def _run_gpu_cpp(
    prep: Dict[str, Any],
    out_dir: Path,
    *,
    M: int,
    N: int,
    K: int,
    warmup: int,
    iters: int,
    rounds: int,
    with_hw: bool,
) -> Dict[str, Any]:
    import torch
    plan = prep["plan"]
    builds = prep["cpp_builds"]
    names_path = Path(plan["names_path"])
    bk_guess = 16
    k_tiles = (K + bk_guess - 1) // bk_guess
    cap_full = max(64, 16 * k_tiles + 16)
    cap_s4 = max(64, 16 * ((k_tiles + 3) // 4) + 16)
    cap_s8 = max(64, 16 * ((k_tiles + 7) // 8) + 16)
    cap_s1 = 64
    all_vars = [v for v in ("L0", "sample1", "full", "stride4", "stride8") if v in builds["variants"]]
    caps_by_var = {"L0": 0, "sample1": cap_s1, "full": cap_full, "stride4": cap_s4, "stride8": cap_s8}

    libs = {}
    for label in all_vars:
        lib = ctypes.CDLL(builds["variants"][label]["so"])
        lib.wp_run_gemm.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_int,
        ]
        lib.wp_run_gemm.restype = ctypes.c_int
        libs[label] = lib

    torch.manual_seed(20261115)
    A = torch.randn((M, K), dtype=torch.float16, device="cuda")
    B = torch.randn((K, N), dtype=torch.float16, device="cuda")
    C_ref = A.float() @ B.float()
    C_l0 = torch.empty((M, N), dtype=torch.float32, device="cuda")
    C_s1 = torch.empty((M, N), dtype=torch.float32, device="cuda")
    C_full = torch.empty((M, N), dtype=torch.float32, device="cuda")

    dump_s1 = out_dir / "fi_sample1.bin"
    dump_full = out_dir / "fi_full.bin"

    # Initial correctness + dump pass
    med = ctypes.c_double(0.0)
    rc0 = libs["L0"].wp_run_gemm(A.data_ptr(), B.data_ptr(), C_l0.data_ptr(), M, N, K, 2, 1, 0, ctypes.byref(med), None, 0)
    rc1 = libs["sample1"].wp_run_gemm(A.data_ptr(), B.data_ptr(), C_s1.data_ptr(), M, N, K, 1, 1, cap_s1, ctypes.byref(med), str(dump_s1).encode("utf-8"), 0)
    rc2 = libs["full"].wp_run_gemm(A.data_ptr(), B.data_ptr(), C_full.data_ptr(), M, N, K, 1, 1, cap_full, ctypes.byref(med), str(dump_full).encode("utf-8"), 0)
    if rc0 != 0 or rc1 != 0 or rc2 != 0:
        raise RuntimeError(f"wp_run_gemm failed: L0={rc0}, sample1={rc1}, full={rc2}")

    c2_l0 = _check_c2_tensors(C_l0, C_l0, C_ref)
    c2_s1 = _check_c2_tensors(C_s1, C_l0, C_ref)
    c2_full = _check_c2_tensors(C_full, C_l0, C_ref)

    # Interleaved randomized steady-state timing in ONE resident GPU process (zero L2-evicting resets)
    rng = random.Random(20261115)
    timings: Dict[str, List[float]] = {v: [] for v in all_vars}
    C_scratch = torch.empty((M, N), dtype=torch.float32, device="cuda")
    for _ in range(rounds):
        order = list(all_vars)
        rng.shuffle(order)
        for v in order:
            cap_v = caps_by_var[v]
            d_us = ctypes.c_double(0.0)
            rc = libs[v].wp_run_gemm(
                A.data_ptr(), B.data_ptr(), C_scratch.data_ptr(),
                M, N, K, warmup, iters, cap_v, ctypes.byref(d_us), None, 0
            )
            if rc != 0:
                raise RuntimeError(f"Benchmark {v} failed rc={rc}")
            timings[v].append(float(d_us.value))

    med_by_var = {v: statistics.median(timings[v]) for v in all_vars}
    med_l0 = med_by_var["L0"]
    med_s1 = med_by_var["sample1"]
    med_full = med_by_var["full"]
    ovh_by_var = {v: round((med_by_var[v] - med_l0) / med_l0 * 100.0, 3) for v in all_vars if v != "L0"}
    flops = 2.0 * M * N * K
    tflops_l0 = (flops / (med_l0 * 1e-6)) / 1e12

    # Closed-Loop <= 10.0% Perturbation-Bounded Stride Controller:
    # Choose the densest continuous K-loop variant ("full" -> "stride4" -> "stride8") within 10.0% budget.
    max_overhead_budget_pct = 10.0
    dense_candidates = [v for v in ("full", "stride4", "stride8") if v in ovh_by_var]
    selected_var = next((v for v in dense_candidates if ovh_by_var[v] <= max_overhead_budget_pct), None)
    if selected_var is None:
        selected_var = min(dense_candidates, key=lambda v: ovh_by_var[v])

    c3_s1 = _decode_dump(dump_s1, names_path, out_dir / "sw_sample1")
    c3_full = _decode_dump(dump_full, names_path, out_dir / "sw_full")
    c3_selected = c3_full
    if selected_var != "full":
        dump_sel = out_dir / f"fi_{selected_var}.bin"
        C_sel = torch.empty((M, N), dtype=torch.float32, device="cuda")
        rc_sel = libs[selected_var].wp_run_gemm(
            A.data_ptr(), B.data_ptr(), C_sel.data_ptr(),
            M, N, K, 1, 1, caps_by_var[selected_var], ctypes.byref(med), str(dump_sel).encode("utf-8"), 0
        )
        if rc_sel != 0:
            raise RuntimeError(f"wp_run_gemm dump {selected_var} failed rc={rc_sel}")
        c3_selected = _decode_dump(dump_sel, names_path, out_dir / "sw_selected")

    hw_result = None
    if with_hw:
        n_spans_sel = int(c3_selected.get("spans") or c3_selected.get("n_spans") or 0)
        hw_var = "sample1" if n_spans_sel > 2_000_000 else selected_var
        hw_cap = caps_by_var[hw_var]
        hw_result = _capture_cpp_sw_hw(
            plan=plan,
            so_path=Path(builds["variants"][hw_var]["so"]),
            names_path=names_path,
            out_dir=out_dir / "sw_hw_capture",
            M=M, N=N, K=K, cap=hw_cap,
        )
        if isinstance(hw_result, dict):
            hw_result["sw_variant_used"] = hw_var
            hw_result["overhead_selected_pct"] = ovh_by_var.get(hw_var)

    return {
        "shape": {"M": M, "N": N, "K": K},
        "c2_correctness": {"L0": c2_l0, "sample1": c2_s1, "full": c2_full},
        "benchmark": {
            "rounds": rounds,
            "iters_per_round": iters,
            "perturbation_budget_pct": max_overhead_budget_pct,
            "selected_variant": selected_var,
            "overhead_selected_pct": ovh_by_var[selected_var],
            "overhead_within_10pct_budget": ovh_by_var[selected_var] <= max_overhead_budget_pct,
            "median_us": {v: round(med_by_var[v], 3) for v in all_vars},
            "overheads_pct": ovh_by_var,
            "single_gpu_tflops_L0": round(tflops_l0, 3),
            "overhead_sample1_pct": ovh_by_var["sample1"],
            "overhead_full_pct": ovh_by_var["full"],
            "raw_us": timings,
        },
        "c3_decode": {"sample1": c3_s1, "full": c3_full, "selected": c3_selected},
        "c4_sw_hw": hw_result,
    }


def _resolve_nv_phys_gpu_index() -> str:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    if vis.isdigit():
        return vis
    smi = shutil.which("nvidia-smi")
    if smi:
        r = _run([smi, "--query-gpu=index,uuid", "--format=csv,noheader"], timeout=15.0)
        if r.returncode == 0:
            for ln in (r.stdout or "").splitlines():
                parts = [x.strip() for x in ln.split(",")]
                if len(parts) >= 2 and parts[1] == vis:
                    return parts[0]
    return "0"


def _capture_cpp_sw_hw(
    *,
    plan: Dict[str, Any],
    so_path: Path,
    names_path: Path,
    out_dir: Path,
    M: int,
    N: int,
    K: int,
    cap: int,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(out_dir, 0o777)
    except OSError:
        pass
    dump_hw = out_dir / "hw_run.fi.bin"
    tmp_dump = out_dir / f"raw_hw_{os.getpid()}_{plan['kernel']}_cpp.fi.bin"
    if plan.get("platform") == "amd":
        # Standalone HIP C++ runner links directly against /opt/rocm/lib/libamdhip64.so,
        # avoiding Conda PyTorch's private torch/lib/libamdhip64.so ABI collision under rocprof-sys LD_PRELOAD.
        burst_cpp = out_dir / "run_one_burst.cpp"
        burst_exe = out_dir / "run_one_burst"
        _write_lf(
            burst_cpp,
            f"""#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <vector>
int main() {{
    void* h = dlopen({str(so_path)!r}, RTLD_NOW);
    if (!h) {{ std::fprintf(stderr, "dlopen: %s\\n", dlerror()); return 2; }}
    using fn_t = int(*)(const void*, const void*, void*, int, int, int, int, int, int, double*, const char*, int);
    auto fn = reinterpret_cast<fn_t>(dlsym(h, "wp_run_gemm"));
    if (!fn) {{ std::fprintf(stderr, "dlsym failed\\n"); return 3; }}
    const size_t nA = (size_t){M} * {K}, nB = (size_t){K} * {N}, nC = (size_t){M} * {N};
    std::vector<uint16_t> hA(nA, 0x3C00u), hB(nB, 0x3800u);
    for (size_t i = 0; i < nA; ++i) hA[i] = (i % 7 == 0) ? 0x3800u : 0x3400u;
    for (size_t i = 0; i < nB; ++i) hB[i] = (i % 5 == 0) ? 0x3800u : 0x3400u;
    const int RING = 12;
    std::vector<void*> dAs(RING, nullptr), dBs(RING, nullptr);
    void *dC = nullptr;
    if (hipMalloc(&dC, nC * 4) != hipSuccess) return 4;
    for (int r = 0; r < RING; ++r) {{
        if (hipMalloc(&dAs[r], nA * 2) != hipSuccess || hipMalloc(&dBs[r], nB * 2) != hipSuccess) return 4;
        hipMemcpy(dAs[r], hA.data(), nA * 2, hipMemcpyHostToDevice);
        hipMemcpy(dBs[r], hB.data(), nB * 2, hipMemcpyHostToDevice);
    }}
    double med = 0.0;
    int rc = 0;
    for (int r = 0; r < RING; ++r) {{
        const char* dp = (r == RING - 1) ? {str(tmp_dump)!r} : nullptr;
        rc = fn(dAs[r], dBs[r], dC, {M}, {N}, {K}, 0, 1, {cap}, &med, dp, 0);
        if (rc != 0) break;
    }}
    for (int r = 0; r < RING; ++r) {{ hipFree(dAs[r]); hipFree(dBs[r]); }}
    hipFree(dC);
    return rc;
}}
""",
        )
        hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
        b_res = _run([hipcc, "-O2", "-std=c++17", str(burst_cpp), "-ldl", "-o", str(burst_exe)], cwd=out_dir, timeout=120.0)
        if b_res.returncode == 0 and burst_exe.is_file():
            return _execute_profiler_wrap(
                plan=plan,
                payload_cmd=[str(burst_exe)],
                tmp_dump=tmp_dump,
                dump_hw=dump_hw,
                names_path=names_path,
                out_dir=out_dir,
                M=M, N=N, K=K,
            )

    payload_py = out_dir / "run_one_burst.py"
    _write_lf(
        payload_py,
        f"""import ctypes, torch
lib = ctypes.CDLL({str(so_path)!r})
lib.wp_run_gemm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_int]
lib.wp_run_gemm.restype = ctypes.c_int
RING = 12
As = [torch.randn(({M}, {K}), dtype=torch.float16, device='cuda') for _ in range(RING)]
Bs = [torch.randn(({K}, {N}), dtype=torch.float16, device='cuda') for _ in range(RING)]
C = torch.empty(({M}, {N}), dtype=torch.float32, device='cuda')
med = ctypes.c_double(0.0)
for r in range(3):
    rc = lib.wp_run_gemm(As[r].data_ptr(), Bs[r].data_ptr(), C.data_ptr(), {M}, {N}, {K}, 0, 1, {cap}, ctypes.byref(med), None, 0)
    assert rc == 0, f"rc={{rc}}"
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for r in range(3, RING):
    dp = {str(tmp_dump).encode('utf-8')!r} if r == RING - 1 else None
    rc = lib.wp_run_gemm(As[r].data_ptr(), Bs[r].data_ptr(), C.data_ptr(), {M}, {N}, {K}, 0, 1, {cap}, ctypes.byref(med), dp, 0)
    assert rc == 0, f"rc={{rc}}"
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
""",
    )
    return _execute_profiler_wrap(
        plan=plan,
        payload_cmd=[sys.executable, "-B", str(payload_py)],
        tmp_dump=tmp_dump,
        dump_hw=dump_hw,
        names_path=names_path,
        out_dir=out_dir,
        M=M, N=N, K=K,
    )


def _execute_profiler_wrap(
    *,
    plan: Dict[str, Any],
    payload_cmd: List[str],
    tmp_dump: Path,
    dump_hw: Path,
    names_path: Path,
    out_dir: Path,
    M: int,
    N: int,
    K: int,
) -> Dict[str, Any]:
    card = plan["card"]
    kname = plan["kernel"]
    if tmp_dump.exists():
        try:
            tmp_dump.unlink()
        except OSError:
            pass

    if card in ("h20", "b300"):
        nsys = shutil.which("nsys") or "/usr/local/bin/nsys"
        phys_dev = _resolve_nv_phys_gpu_index()
        freq_hz = 50000 if card == "h20" else 200000
        nsys_tmp = out_dir / "nsys_tmp"
        nsys_tmp.mkdir(parents=True, exist_ok=True)
        rep_base = out_dir / "cap"
        cmd = [
            nsys, "profile",
            "--trace=cuda",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=none",
            "--flush-on-cudaprofilerstop=true",
            f"--gpu-metrics-devices={phys_dev}",
            f"--gpu-metrics-frequency={freq_hz}",
            "--force-overwrite=true",
            "-o", str(rep_base),
            *payload_cmd,
        ]
        res = _run(cmd, cwd=out_dir, env={"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "TMPDIR": str(nsys_tmp)}, timeout=300.0)
        hw_metrics_note = "measured_gsp_metrics"
        min_samples = "1"
        min_curves = "2"
        combined_err = (res.stderr or "") + "\n" + (res.stdout or "")
        if res.returncode != 0 and ("ERR_NVGPUCTRPERM" in combined_err or "Insufficient privilege" in combined_err):
            hw_metrics_note = "unavailable: ERR_NVGPUCTRPERM (non-root container lacks GSP counter privilege; aligned via CUDA kernel timestamps)"
            min_samples = "0"
            min_curves = "0"
            cmd_no_pm = [
                nsys, "profile",
                "--trace=cuda",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=none",
                "--flush-on-cudaprofilerstop=true",
                "--force-overwrite=true",
                "-o", str(rep_base),
                *payload_cmd,
            ]
            res = _run(cmd_no_pm, cwd=out_dir, env={"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "TMPDIR": str(nsys_tmp)}, timeout=300.0)
        if tmp_dump.is_file():
            shutil.copy2(tmp_dump, dump_hw)
            try:
                tmp_dump.unlink()
            except OSError:
                pass
        rep_file = Path(str(rep_base) + ".nsys-rep")
        if res.returncode != 0 or not rep_file.is_file() or not dump_hw.is_file():
            return {
                "ok": False,
                "stage": "nsys_profile",
                "rc": res.returncode,
                "dump_exists": dump_hw.is_file(),
                "stderr": (res.stderr or res.stdout)[-800:],
            }
        sqlite_file = out_dir / "cap.sqlite"
        exp = _run([nsys, "export", "--type=sqlite", "--force-overwrite=true", "-o", str(sqlite_file), str(rep_file)], cwd=out_dir, timeout=180.0)
        if exp.returncode != 0 or not sqlite_file.is_file():
            return {"ok": False, "stage": "nsys_export", "rc": exp.returncode, "stderr": (exp.stderr or exp.stdout)[-800:]}

        merged_base = out_dir / "sw_hw.perfetto.json"
        al_cmd = [
            sys.executable, "-B", str(ALIGN_NSYS_PATH),
            str(dump_hw),
            "--names", str(names_path),
            "--sqlite", str(sqlite_file),
            "--kernel", kname,
            "--instance", "-1",
            "--min-metrics-samples", min_samples,
            "--min-metric-curves", min_curves,
            "-o", str(merged_base),
        ]
        if min_curves != "0":
            al_cmd.extend(["--metrics", "Tensor Active|SMs Active|SM Active|SM Issue|DRAM Read|DRAM Write"])
        else:
            al_cmd.append("--allow-missing-metrics")
        al = _run(al_cmd, cwd=out_dir, timeout=300.0)
        summary_file = out_dir / "sw_hw.perfetto.summary.json"
        if not summary_file.is_file():
            alt_sum = Path(str(merged_base) + ".summary.json")
            if alt_sum.is_file():
                summary_file = alt_sum
        align_summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.is_file() else None
        gz_file = Path(str(merged_base) + ".gz")
        trace_for_attrib = gz_file if gz_file.is_file() else merged_base

        attrib_doc = None
        att_rc = -1
        if trace_for_attrib.is_file():
            problem_json = out_dir / "problem.json"
            _write_json(problem_json, {
                "useful_flops": int(2 * M * N * K),
                "required_bytes": int(2 * (M * K + K * N) + 4 * M * N),
            })
            attrib_json = out_dir / "attrib.json"
            att_cmd = [
                sys.executable, "-B", str(ATTRIB_PATH),
                str(trace_for_attrib),
                "--problem", str(problem_json),
                "-o", str(attrib_json),
                "--quiet",
            ]
            if summary_file.is_file():
                att_cmd.extend(["--summary", str(summary_file)])
            att = _run(att_cmd, cwd=out_dir, timeout=180.0)
            att_rc = att.returncode
            if attrib_json.is_file():
                attrib_doc = json.loads(attrib_json.read_text(encoding="utf-8"))

            # Pin GPU hardware info & utilization [0-100%] process at Row #1 (pid=1, upid=1, sort_index=-1000)
            # and synthesize DRAM Total Bandwidth [Throughput %] = Read% + Write%.
            import gzip as _gzip_nv
            if trace_for_attrib.suffix == ".gz":
                with _gzip_nv.open(trace_for_attrib, "rt", encoding="utf-8") as _fh:
                    _nv_doc = json.load(_fh)
            else:
                _nv_doc = json.loads(trace_for_attrib.read_text(encoding="utf-8"))
            _evs_nv = _nv_doc.get("traceEvents") or []
            _nv_other = _nv_doc.setdefault("otherData", {})
            _nv_other.setdefault("target", {"card": card, "platform": "nv", "arch": plan.get("arch")})
            _c_pids = {int(e["pid"]) for e in _evs_nv if e.get("ph") == "C" and isinstance(e.get("pid"), int)}
            if _c_pids:
                _hp = min(_c_pids)
                _rd_by_ts: Dict[float, float] = {}
                _wr_by_ts: Dict[float, float] = {}
                for _e in _evs_nv:
                    if _e.get("ph") == "M" and _e.get("pid") == _hp:
                        if _e.get("name") == "process_name":
                            _e["args"] = {"name": f"00 · 硬件信息与利用率 [0–100%]（NVIDIA {card.upper()} nsys {freq_hz // 1000}kHz 实测曲线）"}
                        elif _e.get("name") == "process_sort_index":
                            _e["args"] = {"sort_index": -1000}
                    elif _e.get("ph") == "C" and _e.get("pid") == _hp:
                        _nm = str(_e.get("name") or "")
                        _v = float((_e.get("args") or {}).get("value") or 0.0)
                        if "DRAM Read" in _nm:
                            _rd_by_ts[float(_e["ts"])] = _v
                        elif "DRAM Write" in _nm:
                            _wr_by_ts[float(_e["ts"])] = _v
                _tot_evs = [
                    {
                        "ph": "C",
                        "pid": _hp,
                        "ts": _ts_k,
                        "name": "DRAM Total Bandwidth [Throughput %]",
                        "args": {"value": round(min(100.0, _rd_by_ts[_ts_k] + _wr_by_ts[_ts_k]), 3)},
                    }
                    for _ts_k in sorted(set(_rd_by_ts) & set(_wr_by_ts))
                ]
                # Keep all pid=_hp events at the front of traceEvents so TraceProcessor assigns upid=1
                _hp_evs = [e for e in _evs_nv if e.get("pid") == _hp] + _tot_evs
                _rest_evs = [e for e in _evs_nv if e.get("pid") != _hp]
                _nv_doc["traceEvents"] = _hp_evs + _rest_evs
                if trace_for_attrib.suffix == ".gz":
                    with _gzip_nv.open(trace_for_attrib, "wt", encoding="utf-8") as _fh_out:
                        json.dump(_nv_doc, _fh_out, ensure_ascii=False)
                else:
                    trace_for_attrib.write_text(json.dumps(_nv_doc, ensure_ascii=False), encoding="utf-8")

        gate_mod = _load_module(TRACE_GATE_PATH, "wp_fi_trace_gate")
        nv_gate = (
            gate_mod.verify_trace(trace_for_attrib, require_hw=(min_curves != "0"), card=card)
            if trace_for_attrib.is_file()
            else {"ok": False, "errors": ["missing sw_hw trace"]}
        )
        align_obj = (align_summary or {}).get("align") or {}
        if min_curves == "0":
            c4_ok = al.returncode in (0, 2) and trace_for_attrib.is_file() and bool((align_obj.get("axis_check") or {}).get("ok")) and bool(nv_gate.get("ok"))
        else:
            c4_ok = al.returncode == 0 and bool(align_obj.get("ok")) and bool(nv_gate.get("ok"))

        return {
            "ok": c4_ok,
            "collector": f"nsys_{freq_hz // 1000}khz" if min_curves != "0" else "nsys_cuda_timeline",
            "hw_metrics_status": hw_metrics_note,
            "perfetto_trace": str(trace_for_attrib),
            "trace_gate": {"ok": nv_gate.get("ok"), "errors": nv_gate.get("errors"), "hw_pids": nv_gate.get("hw_pids")},
            "align": align_obj or None,
            "attribution_summary": {
                "primary_attribution": (attrib_doc or {}).get("primary_attribution"),
                "bottleneck_resource": (attrib_doc or {}).get("bottleneck_resource"),
                "trust": (attrib_doc or {}).get("trust"),
                "summary_zh": (attrib_doc or {}).get("summary_zh"),
            } if attrib_doc else None,
            "align_rc": al.returncode,
            "attrib_rc": att_rc,
            "stderr_tail": (al.stderr or al.stdout)[-500:] if al.returncode != 0 else "",
        }

    # AMD MI308X: same-run rocprof-sys-run --rocm-events + build_rocprof_sys_sw_hw.py
    from rocprof_sys_adapter import DEFAULT_ROCM_DOMAINS, DEFAULT_ROCM_EVENTS, build_rocprof_sys_cmd, discover_rocprof_sys_artifacts
    rps = shutil.which("rocprof-sys-run") or shutil.which("rocprof-sys") or "/opt/rocm/bin/rocprof-sys-run"
    rps_out = out_dir / "rocprof_sys_out"
    if rps_out.exists():
        shutil.rmtree(rps_out)
    rps_out.mkdir(parents=True, exist_ok=True)
    events = list(DEFAULT_ROCM_EVENTS)
    cmd = build_rocprof_sys_cmd(
        rps,
        rps_out,
        payload_cmd,
        rocm_domains=DEFAULT_ROCM_DOMAINS,
        rocm_events=DEFAULT_ROCM_EVENTS,
    )
    res = _run(cmd, cwd=out_dir, timeout=300.0)
    if tmp_dump.is_file():
        shutil.copy2(tmp_dump, dump_hw)
        try:
            tmp_dump.unlink()
        except OSError:
            pass
    disc = discover_rocprof_sys_artifacts(rps_out)
    protos = [Path(p) for p in (disc.get("trace_files") or [])] or sorted(rps_out.rglob("*perfetto-trace-*.proto"))
    kstage = re.sub(r"[^a-z0-9]", "", kname.lower()) or "gemm"
    sw_canon_prefix = out_dir / f"trace__autogemm_{kstage}__card-mi308x__bs1_sq2048__fp16__grid1_warp4_ph4__sw"
    merged_gz = out_dir / f"trace__autogemm_{kstage}__card-mi308x__bs1_sq2048__fp16__grid1_warp4_ph4__sw_hw.perfetto.json.gz"

    if res.returncode != 0 or not protos or not dump_hw.is_file():
        # Conda PyTorch bundles private site-packages/torch/lib/libamdhip64.so which conflicts with
        # rocprof-sys LD_PRELOAD; fall back to rocprofv3 --pmc (rocprofiler-sdk runtime registration)
        # per rocprof_sys_adapter.py policy!
        rv3 = shutil.which("rocprofv3") or "/opt/rocm/bin/rocprofv3"
        rv3_out = out_dir / "rocprofv3_fallback"
        if rv3_out.exists():
            shutil.rmtree(rv3_out)
        rv3_out.mkdir(parents=True, exist_ok=True)
        pmc_events = [
            "SQ_VALU_MFMA_BUSY_CYCLES",
            "SQ_BUSY_CU_CYCLES",
            "GRBM_GUI_ACTIVE",
            "TCC_EA0_RDREQ_DRAM",
            "TCC_EA0_WRREQ_DRAM",
            "TCC_HIT",
            "TCC_MISS",
        ]
        rv3_cmd = [
            rv3, "--kernel-trace", "--pmc", *pmc_events,
            "--output-format", "csv", "pftrace",
            "-d", str(rv3_out), "--", *payload_cmd,
        ]
        rv3_res = _run(rv3_cmd, cwd=out_dir, timeout=300.0)
        if tmp_dump.is_file():
            shutil.copy2(tmp_dump, dump_hw)
            try:
                tmp_dump.unlink()
            except OSError:
                pass
        csvs = sorted(rv3_out.rglob("*counter_collection.csv"))
        if rv3_res.returncode != 0 or not csvs or not dump_hw.is_file():
            return {
                "ok": False,
                "stage": "rocprofv3_pmc_fallback",
                "rocprof_sys_rc": res.returncode,
                "rocprofv3_rc": rv3_res.returncode,
                "dump_exists": dump_hw.is_file(),
                "stderr": (rv3_res.stderr or rv3_res.stdout)[-800:],
            }
        sw_dec = _decode_dump(dump_hw, names_path, sw_canon_prefix)
        if not sw_dec.get("ok"):
            return {"ok": False, "stage": "decode_for_merge", "decode": sw_dec}

        import csv as _csv
        import gzip as _gzip
        by_disp: Dict[str, Dict[str, Any]] = {}
        with csvs[0].open("r", encoding="utf-8", newline="") as fh:
            for row in _csv.DictReader(fh):
                kn = row.get("Kernel_Name") or ""
                if kname not in kn:
                    continue
                did = str(row.get("Dispatch_Id") or row.get("Correlation_Id") or len(by_disp))
                rec_d = by_disp.setdefault(did, {
                    "dispatch_id": did,
                    "kernel_name": kn,
                    "start_ns": int(float(row.get("Start_Timestamp") or 0)),
                    "end_ns": int(float(row.get("End_Timestamp") or 0)),
                    "counters": {},
                })
                cname = (row.get("Counter_Name") or "").strip()
                cval = float(row.get("Counter_Value") or 0.0)
                rec_d["counters"][cname] = rec_d["counters"].get(cname, 0.0) + cval
        if not by_disp:
            return {"ok": False, "stage": "rocprofv3_no_matching_kernel", "kernel": kname}
        last_disp = sorted(by_disp.values(), key=lambda x: x["start_ns"])[-1]
        ctrs = last_disp["counters"]
        gui = max(ctrs.get("GRBM_GUI_ACTIVE", 0.0), 1.0)
        mfma_busy = ctrs.get("SQ_VALU_MFMA_BUSY_CYCLES", 0.0)
        cu_busy = ctrs.get("SQ_BUSY_CU_CYCLES", 0.0)
        # Across 4 XCCs on MI308X, rocprofv3 sums counter values across XCC instances:
        # 20 CU/XCC * 4 SIMD/CU = 80 SIMD/XCC (320 total SIMDs, 80 total CUs).
        mfma_util_pct = min(100.0, round(100.0 * mfma_busy / (gui * 80.0), 3))
        cu_active_pct = min(100.0, round(100.0 * cu_busy / (gui * 20.0), 3))
        dur_ns = max(1, int(last_disp["end_ns"] - last_disp["start_ns"]))
        hw_dur_us = round(dur_ns / 1000.0, 3)
        # HBM bandwidth utilization [%] per build_rocprof_sys_sw_hw.py (MI308X measured HBM peak = 3684.0 GB/s, 64B DRAM req)
        hbm_peak_bytes_s = 3684.0 * 1.0e9
        rd_bytes = ctrs.get("TCC_EA0_RDREQ_DRAM", 0.0) * 64.0
        wr_bytes = ctrs.get("TCC_EA0_WRREQ_DRAM", 0.0) * 64.0
        mem_rd_pct = min(100.0, round(100.0 * (rd_bytes * 1.0e9 / dur_ns) / hbm_peak_bytes_s, 3))
        mem_wr_pct = min(100.0, round(100.0 * (wr_bytes * 1.0e9 / dur_ns) / hbm_peak_bytes_s, 3))
        mem_tot_pct = min(100.0, round(mem_rd_pct + mem_wr_pct, 3))
        tcc_tot = ctrs.get("TCC_HIT", 0.0) + ctrs.get("TCC_MISS", 0.0)
        l2_hit_pct = round(100.0 * ctrs.get("TCC_HIT", 0.0) / tcc_tot, 3) if tcc_tot > 0 else 0.0

        with _gzip.open(sw_dec["trace"], "rt", encoding="utf-8") as gz_in:
            tr_doc = json.load(gz_in)
        evs = tr_doc.get("traceEvents") or []
        xs = [float(e.get("ts", 0.0)) for e in evs if e.get("ph") == "X"]
        xe = [float(e.get("ts", 0.0)) + float(e.get("dur", 0.0)) for e in evs if e.get("ph") == "X"]
        t0_us = min(xs) if xs else 0.0
        t1_us = max(xe) if xe else hw_dur_us
        sw_dur_us = max(0.001, round(t1_us - t0_us, 3))
        # Shift all software CU/wave PIDs to >= 100 and pin Hardware Info & Utilization [0-100%] at pid=1, upid=1, sort_index=-1000!
        SW_PID_SHIFT = 100
        for _ev_s in evs:
            if isinstance(_ev_s.get("pid"), int):
                _ev_s["pid"] = int(_ev_s["pid"]) + SW_PID_SHIFT
                if _ev_s.get("ph") == "M" and _ev_s.get("name") == "process_sort_index":
                    _am = _ev_s.get("args") or {}
                    if isinstance(_am.get("sort_index"), (int, float)):
                        _am["sort_index"] = int(_am["sort_index"]) + SW_PID_SHIFT

        hw_pid = 1
        hw_spec_rows = [
            ("compute_util", "compute_util [matrix_core]", mfma_util_pct),
            ("memory_util", "memory_util [HBM request estimate]", mem_tot_pct),
            ("memory_read_util", "memory_read_util [HBM request estimate]", mem_rd_pct),
            ("memory_write_util", "memory_write_util [HBM request estimate]", mem_wr_pct),
            ("sm_active", "sm_active [GPU active]", cu_active_pct),
            ("l2_hit_rate", "l2_hit_rate [TCC %]", l2_hit_pct),
        ]
        metrics_out: Dict[str, float] = {m_label: m_val for _, m_label, m_val in hw_spec_rows}
        rocm_merge_info = {
            "gpu": "AMD Instinct MI308X (gfx942, 80 CU / 4 XCC / 192 GB HBM3)",
            "collector": "rocprofv3_pmc_fallback",
            "fallback_reason": "rocprof-sys LD_PRELOAD conflicts with Conda PyTorch private libamdhip64.so; routed to rocprofv3 --pmc per rocprof_sys_adapter policy",
            "resolution": "per_dispatch_aggregate",
            "start_placement_measured": False,
            "hw_dispatch_duration_us": hw_dur_us,
            "sw_span_us": sw_dur_us,
            "duration_ratio_hw_over_sw": round(hw_dur_us / sw_dur_us, 4),
            "metrics_pct": metrics_out,
            "raw_counters": {k: round(v, 1) for k, v in ctrs.items()},
        }
        hw_evs: List[Dict[str, Any]] = [
            {"ph": "M", "pid": hw_pid, "name": "process_name", "args": {"name": "00 · 硬件信息与利用率 [0–100%]（AMD MI308X 同趟 ROCm PMC 实测）"}},
            {"ph": "M", "pid": hw_pid, "name": "process_sort_index", "args": {"sort_index": -1000}},
            {"ph": "M", "pid": hw_pid, "tid": 1, "name": "thread_name", "args": {"name": "GPU 硬件规格与 Kernel 时间窗（点开查看 args）"}},
            {"ph": "M", "pid": hw_pid, "tid": 1, "name": "thread_sort_index", "args": {"sort_index": 1}},
            {
                "ph": "X",
                "pid": hw_pid,
                "tid": 1,
                "ts": round(t0_us, 3),
                "dur": sw_dur_us,
                "name": f"AMD Instinct MI308X · 80 CU (4 XCC) · {kname} ({hw_dur_us} us)",
                "cat": "hardware.dispatch",
                "args": rocm_merge_info,
            },
        ]
        hardware_rows_meta: List[Dict[str, Any]] = []
        for idx_h, (sem_id, m_label, m_val) in enumerate(hw_spec_rows, start=2):
            hw_evs.append({"ph": "M", "pid": hw_pid, "tid": idx_h, "name": "thread_name", "args": {"name": m_label}})
            hw_evs.append({"ph": "M", "pid": hw_pid, "tid": idx_h, "name": "thread_sort_index", "args": {"sort_index": idx_h}})
            hw_evs.append({"ph": "C", "pid": hw_pid, "tid": idx_h, "ts": round(t0_us, 3), "name": m_label, "args": {"value": m_val}})
            hw_evs.append({"ph": "C", "pid": hw_pid, "tid": idx_h, "ts": round(t1_us, 3), "name": m_label, "args": {"value": m_val}})
            hw_evs.append({"ph": "C", "pid": hw_pid, "tid": idx_h, "ts": round(t1_us + 0.001, 3), "name": m_label, "args": {"value": 0.0}})
            hardware_rows_meta.append({
                "semantic_id": sem_id,
                "timeline_label": m_label,
                "value": m_val,
                "hardware_start_placement_us": round(t0_us, 3),
                "hardware_dur_us": sw_dur_us,
                "hardware_start_ns": int(last_disp["start_ns"]),
                "hardware_dur_ns": dur_ns,
                "dispatch_ordinal": 0,
                "counter_time_resolution": "per_dispatch_aggregate",
                "intra_kernel_curve": False,
            })
        tr_doc["traceEvents"] = hw_evs + evs
        other = tr_doc.setdefault("otherData", {})
        other["target"] = {"card": "mi308x", "platform": "amd", "arch": "gfx942"}
        other["hardware_rows"] = hardware_rows_meta
        other["rocm_hw_merge"] = rocm_merge_info
        with _gzip.open(merged_gz, "wt", encoding="utf-8") as gz_out:
            json.dump(tr_doc, gz_out, ensure_ascii=False)
        gate_mod = _load_module(TRACE_GATE_PATH, "wp_fi_trace_gate")
        amd_gate = gate_mod.verify_trace(merged_gz, require_hw=True, card="mi308x")
        pftraces = sorted(rv3_out.rglob("*.pftrace"))
        return {
            "ok": merged_gz.is_file() and bool(amd_gate.get("ok")),
            "collector": "rocprofv3_pmc_fallback",
            "fallback_reason": rocm_merge_info["fallback_reason"],
            "pmc_events": pmc_events,
            "pftrace": str(pftraces[0]) if pftraces else None,
            "perfetto_trace": str(merged_gz),
            "trace_gate": {"ok": amd_gate.get("ok"), "errors": amd_gate.get("errors"), "hw_pids": amd_gate.get("hw_pids")},
            "hw_dispatch_duration_us": hw_dur_us,
            "sw_span_us": sw_dur_us,
            "duration_ratio_hw_over_sw": rocm_merge_info["duration_ratio_hw_over_sw"],
            "selected_dispatch_metrics": metrics_out,
        }

    # 1) Decode software dump first into canonical __sw.perfetto.json.gz
    sw_dec = _decode_dump(dump_hw, names_path, sw_canon_prefix)
    if not sw_dec.get("ok"):
        return {"ok": False, "stage": "decode_for_merge", "decode": sw_dec}

    # 2) Obtain trace_processor_shell via get_trace_processor.py
    tps_res = _run(
        [sys.executable, "-B", str(SKILL_ROOT / "weaveprof/pipeline/scan" / "get_trace_processor.py"), "--out-dir", str(out_dir / "_tps")],
        cwd=out_dir,
        timeout=180.0,
    )
    tps_bin = (tps_res.stdout or "").strip().splitlines()[-1].strip() if tps_res.returncode == 0 and (tps_res.stdout or "").strip() else ""
    if not tps_bin or not Path(tps_bin).is_file():
        return {"ok": False, "stage": "get_trace_processor", "rc": tps_res.returncode, "stderr": (tps_res.stderr or tps_res.stdout)[-500:]}

    # 3) Query total matching kernel slices so --kernel-ordinal selects the exact last launch dumped into dump_hw
    cnt_sql = f"SELECT COUNT(*) AS n FROM slice WHERE name LIKE '%{kname}%' AND dur > 0;"
    q_proc = subprocess.run([tps_bin, "-Q", cnt_sql, str(protos[0])], text=True, capture_output=True, timeout=60.0, check=False)
    last_ordinal = 0
    for ln in (q_proc.stdout or "").splitlines():
        ln_s = ln.strip().strip('"')
        if ln_s.isdigit() and int(ln_s) > 0:
            last_ordinal = int(ln_s) - 1

    extracted_hw = out_dir / "rocprof_sys_hw_extracted.json"
    mg = _run(
        [
            sys.executable, "-B", str(ROCPROF_SYS_MERGE_PATH),
            "--rocprof-sys-proto", str(protos[0]),
            "--trace-processor", tps_bin,
            "--trace", sw_dec["trace"],
            "--kernel-name-contains", kname,
            "--kernel-ordinal", str(last_ordinal),
            "--extracted-json", str(extracted_hw),
            "--out", str(merged_gz),
        ],
        cwd=out_dir,
        timeout=300.0,
    )
    merge_report = None
    if mg.returncode == 0 and (mg.stdout or "").strip():
        try:
            merge_report = json.loads(mg.stdout)
        except json.JSONDecodeError:
            merge_report = {"raw_stdout": mg.stdout[-500:]}
    hw_doc = json.loads(extracted_hw.read_text(encoding="utf-8")) if extracted_hw.is_file() else None
    gate_mod = _load_module(TRACE_GATE_PATH, "wp_fi_trace_gate")
    rps_gate = gate_mod.verify_trace(merged_gz, require_hw=True, card="mi308x") if merged_gz.is_file() else {"ok": False, "errors": ["missing merged_gz"]}
    return {
        "ok": mg.returncode == 0 and merged_gz.is_file() and bool(rps_gate.get("ok")),
        "collector": "rocprof-sys",
        "rocm_events": events,
        "proto": str(protos[0]),
        "selected_kernel_ordinal": last_ordinal,
        "perfetto_trace": str(merged_gz),
        "trace_gate": {"ok": rps_gate.get("ok"), "errors": rps_gate.get("errors"), "hw_pids": rps_gate.get("hw_pids")},
        "hw_availability": (hw_doc or {}).get("availability"),
        "selected_dispatch_metrics": ((hw_doc or {}).get("dispatches") or [{}])[0].get("metrics") if hw_doc else None,
        "merge_report": merge_report,
        "merge_rc": mg.returncode,
        "stderr_tail": (mg.stderr or mg.stdout)[-500:] if mg.returncode != 0 else "",
    }


def _run_gpu_triton(
    prep: Dict[str, Any],
    out_dir: Path,
    *,
    M: int,
    N: int,
    K: int,
    warmup: int,
    iters: int,
    rounds: int,
    with_hw: bool,
) -> Dict[str, Any]:
    cc_shim = _ensure_cc_shim_if_needed(out_dir)
    import torch
    import triton

    plan = prep["plan"]
    card = plan["card"]
    arch = plan["arch"]
    platform = plan["platform"]
    names_path = Path(plan["names_path"])
    inst_path = Path(plan["instrumented_source"])

    triton_rec_dir = str(SKILL_ROOT / "recorders" / "triton")
    os.environ["WEAVEPROFILER_TRITON_DIR"] = triton_rec_dir
    if triton_rec_dir not in sys.path:
        sys.path.insert(0, triton_rec_dir)

    mod = _load_module(inst_path, "wp_inst_triton_gemm")
    kernel_fn = getattr(mod, plan["kernel"])

    BM, BN, BK, GM = 128, 128, 32, 8
    num_warps = 4
    threads = num_warps * (64 if platform == "amd" else 32)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    n_ctas = grid[0]
    k_tiles = triton.cdiv(K, BK)
    cap_full = max(64, 6 * k_tiles + 16)
    cap_s4 = max(64, 6 * ((k_tiles + 3) // 4) + 16)
    cap_s8 = max(64, 6 * ((k_tiles + 7) // 8) + 16)
    cap_s1 = 32
    all_t_vars = ["L0", "sample1", "full", "stride4", "stride8"]
    sample_arg_by_var = {"L0": 0, "sample1": 1, "full": 0, "stride4": 4, "stride8": 8}
    cap_by_var = {"L0": 8, "sample1": cap_s1, "full": cap_full, "stride4": cap_s4, "stride8": cap_s8}

    torch.manual_seed(20261115)
    A = torch.randn((M, K), dtype=torch.float16, device="cuda")
    B = torch.randn((K, N), dtype=torch.float16, device="cuda")
    C_ref = A.float() @ B.float()
    C_l0 = torch.empty((M, N), dtype=torch.float32, device="cuda")
    C_s1 = torch.empty((M, N), dtype=torch.float32, device="cuda")
    C_full = torch.empty((M, N), dtype=torch.float32, device="cuda")

    dump_s1 = out_dir / "fi_sample1.bin"
    dump_full = out_dir / "fi_full.bin"

    if platform == "nv":
        import fi_triton as FT
        buf_by_var = {
            "L0": FT.FiBuffer(1, num_warps, 8, gran=1, timer=1, level=0),
            "sample1": FT.FiBuffer(n_ctas, num_warps, cap_s1, gran=1, timer=1, level=2),
            "full": FT.FiBuffer(n_ctas, num_warps, cap_full, gran=1, timer=1, level=2),
            "stride4": FT.FiBuffer(n_ctas, num_warps, cap_s4, gran=1, timer=1, level=2),
            "stride8": FT.FiBuffer(n_ctas, num_warps, cap_s8, gran=1, timer=1, level=2),
        }
        cfg_by_var = {
            "L0": FT.FiCfg(cap=8, warps=1, n_slots=1, gran=1, timer=1, level=0, threads=threads, arch=arch),
            "sample1": FT.FiCfg(cap=cap_s1, warps=1, n_slots=n_ctas, gran=1, timer=1, level=2, threads=threads, arch=arch),
            "full": FT.FiCfg(cap=cap_full, warps=1, n_slots=n_ctas, gran=1, timer=1, level=2, threads=threads, arch=arch),
            "stride4": FT.FiCfg(cap=cap_s4, warps=1, n_slots=n_ctas, gran=1, timer=1, level=2, threads=threads, arch=arch),
            "stride8": FT.FiCfg(cap=cap_s8, warps=1, n_slots=n_ctas, gran=1, timer=1, level=2, threads=threads, arch=arch),
        }
        stages_for_variant = {v: 3 for v in all_t_vars}

        def launch_variant(v: str, c_out, zero_buf: bool = False):
            st = stages_for_variant[v]
            b_v = buf_by_var[v]
            c_v = cfg_by_var[v]
            s_arg = sample_arg_by_var[v]
            if zero_buf and v != "L0":
                b_v.reset()
                torch.cuda.synchronize()
            try:
                kernel_fn[grid](
                    A, B, c_out, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), c_out.stride(0), c_out.stride(1),
                    BM, BN, BK, GM, b_v.rec, b_v.overflow, c_v, s_arg, 2, num_warps=num_warps, num_stages=st
                )
            except RuntimeError as exc:
                if st > 1 and "PassManager::run failed" in str(exc):
                    stages_for_variant[v] = 1
                    if zero_buf and v != "L0":
                        b_v.reset()
                        torch.cuda.synchronize()
                    kernel_fn[grid](
                        A, B, c_out, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), c_out.stride(0), c_out.stride(1),
                        BM, BN, BK, GM, b_v.rec, b_v.overflow, c_v, s_arg, 2, num_warps=num_warps, num_stages=1
                    )
                else:
                    raise

        tcache_dirs = {v: out_dir / f"tcache_{v}" for v in all_t_vars}
        tcache_l0 = tcache_dirs["L0"]
        tcache_s1 = tcache_dirs["sample1"]
        tcache_full = tcache_dirs["full"]
        for d in tcache_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        for v_init, c_tgt in (("L0", C_l0), ("sample1", C_s1), ("full", C_full), ("stride4", C_full), ("stride8", C_full)):
            os.environ["TRITON_CACHE_DIR"] = str(tcache_dirs[v_init])
            out_t = c_tgt if v_init in ("L0", "sample1", "full") else torch.empty((M, N), dtype=torch.float32, device="cuda")
            launch_variant(v_init, out_t, zero_buf=(v_init != "L0"))
            torch.cuda.synchronize()
        buf_by_var["sample1"].dump(str(dump_s1), meta={"kernel": plan["kernel"], "arch": arch})
        buf_by_var["full"].dump(str(dump_full), meta={"kernel": plan["kernel"], "arch": arch})
    else:
        import fi_triton_amd as FTA
        buf_by_var = {
            "L0": FTA.FiPackedBuffer(n_ctas=1, cap=8, gran=1, num_warps=num_warps, timer_mode=1),
            "sample1": FTA.FiPackedBuffer(n_ctas=n_ctas, cap=cap_s1, gran=1, num_warps=num_warps, timer_mode=1),
            "full": FTA.FiPackedBuffer(n_ctas=n_ctas, cap=cap_full, gran=1, num_warps=num_warps, timer_mode=1),
            "stride4": FTA.FiPackedBuffer(n_ctas=n_ctas, cap=cap_s4, gran=1, num_warps=num_warps, timer_mode=1),
            "stride8": FTA.FiPackedBuffer(n_ctas=n_ctas, cap=cap_s8, gran=1, num_warps=num_warps, timer_mode=1),
        }
        stages_for_variant = {v: 2 for v in all_t_vars}

        def launch_variant(v: str, c_out, zero_buf: bool = False):
            st = stages_for_variant[v]
            b_v = buf_by_var[v]
            lvl = 0 if v == "L0" else 2
            s_arg = sample_arg_by_var[v]
            c_arg = cap_by_var[v]
            if zero_buf and v != "L0":
                b_v.reset()
                torch.cuda.synchronize()
            try:
                kernel_fn[grid](
                    A, B, c_out, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), c_out.stride(0), c_out.stride(1),
                    BM, BN, BK, GM, b_v.rec, b_v.overflow,
                    lvl, 1, c_arg, 1, threads, 1, s_arg, num_warps=num_warps, num_stages=st
                )
            except RuntimeError as exc:
                if st > 1 and "PassManager::run failed" in str(exc):
                    stages_for_variant[v] = 1
                    if zero_buf and v != "L0":
                        b_v.reset()
                        torch.cuda.synchronize()
                    kernel_fn[grid](
                        A, B, c_out, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), c_out.stride(0), c_out.stride(1),
                        BM, BN, BK, GM, b_v.rec, b_v.overflow,
                        lvl, 1, c_arg, 1, threads, 1, s_arg, num_warps=num_warps, num_stages=1
                    )
                else:
                    raise

        tcache_dirs = {v: out_dir / f"tcache_{v}" for v in all_t_vars}
        tcache_l0 = tcache_dirs["L0"]
        tcache_s1 = tcache_dirs["sample1"]
        tcache_full = tcache_dirs["full"]
        for d in tcache_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        for v_init, c_tgt in (("L0", C_l0), ("sample1", C_s1), ("full", C_full), ("stride4", C_full), ("stride8", C_full)):
            os.environ["TRITON_CACHE_DIR"] = str(tcache_dirs[v_init])
            out_t = c_tgt if v_init in ("L0", "sample1", "full") else torch.empty((M, N), dtype=torch.float32, device="cuda")
            launch_variant(v_init, out_t, zero_buf=(v_init != "L0"))
            torch.cuda.synchronize()
        buf_by_var["sample1"].dump(str(dump_s1), extra_meta={"kernel": plan["kernel"]})
        buf_by_var["full"].dump(str(dump_full), extra_meta={"kernel": plan["kernel"]})

    # Run C1 static ISA check on Triton compiled cache artifacts (.cubin / .amdgcn / .hsaco)
    c1_triton: Dict[str, Any] = {}
    for label, tc_dir in (("sample1", tcache_s1), ("full", tcache_full)):
        bs_out = out_dir / f"c1_binsim_triton_{label}.json"
        if platform == "nv":
            cmd_bs = [
                sys.executable, "-B", str(BINSIM_NV_PATH),
                str(tcache_l0), str(tc_dir),
                "--kernel", plan["kernel"],
                "--arch", arch,
                "-o", str(bs_out),
            ]
        else:
            l0_art = next(tcache_l0.rglob("*.amdgcn"), next(tcache_l0.rglob("*.hsaco"), tcache_l0))
            fi_art = next(tc_dir.rglob("*.amdgcn"), next(tc_dir.rglob("*.hsaco"), tc_dir))
            co_wd = out_dir / f"binsim_triton_work_{label}"
            co_wd.mkdir(parents=True, exist_ok=True)
            cmd_bs = [
                sys.executable, "-B", str(BINSIM_HIP_PATH),
                "--l0", str(l0_art),
                "--l2", str(fi_art),
                "--kernel", plan["kernel"],
                "--gfx", arch,
                "--work-dir", str(co_wd),
                "-o", str(bs_out),
            ]
        bs_res = _run(cmd_bs, cwd=out_dir, timeout=180.0)
        if bs_res.returncode == 0 and bs_out.is_file():
            c1_triton[label] = json.loads(bs_out.read_text(encoding="utf-8"))
        else:
            c1_triton[label] = {"error": (bs_res.stderr or bs_res.stdout)[-500:]}

    c2_l0 = _check_c2_tensors(C_l0, C_l0, C_ref)
    c2_s1 = _check_c2_tensors(C_s1, C_l0, C_ref)
    c2_full = _check_c2_tensors(C_full, C_l0, C_ref)

    # Interleaved randomized steady-state timing in ONE resident GPU process (zero L2-evicting resets)
    rng = random.Random(20261115)
    timings: Dict[str, List[float]] = {v: [] for v in all_t_vars}
    C_scratch = torch.empty((M, N), dtype=torch.float32, device="cuda")
    for _ in range(rounds):
        order = list(all_t_vars)
        rng.shuffle(order)
        for v in order:
            for _ in range(warmup):
                launch_variant(v, C_scratch, zero_buf=False)
            torch.cuda.synchronize()
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(iters):
                ev0.record()
                launch_variant(v, C_scratch, zero_buf=False)
                ev1.record()
                torch.cuda.synchronize()
                samples.append(ev0.elapsed_time(ev1) * 1000.0)
            timings[v].append(statistics.median(samples))

    med_by_var = {v: statistics.median(timings[v]) for v in all_t_vars}
    med_l0 = med_by_var["L0"]
    med_s1 = med_by_var["sample1"]
    med_full = med_by_var["full"]
    ovh_by_var = {v: round((med_by_var[v] - med_l0) / med_l0 * 100.0, 3) for v in all_t_vars if v != "L0"}
    tflops_l0 = (2.0 * M * N * K / (med_l0 * 1e-6)) / 1e12

    max_overhead_budget_pct = 10.0
    dense_candidates = ["full", "stride4", "stride8"]
    selected_var = next((v for v in dense_candidates if ovh_by_var[v] <= max_overhead_budget_pct), None)
    if selected_var is None:
        selected_var = min(dense_candidates, key=lambda v: ovh_by_var[v])

    c3_s1 = _decode_dump(dump_s1, names_path, out_dir / "sw_sample1")
    c3_full = _decode_dump(dump_full, names_path, out_dir / "sw_full")
    c3_selected = c3_full
    if selected_var != "full":
        dump_sel = out_dir / f"fi_{selected_var}.bin"
        if platform == "nv":
            buf_by_var[selected_var].dump(str(dump_sel), meta={"kernel": plan["kernel"], "arch": arch})
        else:
            buf_by_var[selected_var].dump(str(dump_sel), extra_meta={"kernel": plan["kernel"]})
        c3_selected = _decode_dump(dump_sel, names_path, out_dir / "sw_selected")

    hw_result = None
    if with_hw:
        sel_s_arg = sample_arg_by_var[selected_var]
        sel_cap = cap_by_var[selected_var]
        hw_dir = out_dir / "sw_hw_capture"
        hw_dir.mkdir(parents=True, exist_ok=True)
        tcache_hw = hw_dir / "tcache_hw"
        tcache_hw.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(hw_dir, 0o777)
            os.chmod(tcache_hw, 0o777)
        except OSError:
            pass
        dump_hw = hw_dir / "hw_run.fi.bin"
        tmp_dump = hw_dir / f"raw_hw_{os.getpid()}_{plan['kernel']}_triton.fi.bin"
        burst_py = hw_dir / "run_triton_burst.py"
        _write_lf(
            burst_py,
            f"""import os, sys, importlib.util
if {cc_shim!r}:
    os.environ['CC'] = {cc_shim!r}
os.environ['TRITON_CACHE_DIR'] = {str(tcache_hw)!r}
os.environ['WEAVEPROFILER_TRITON_DIR'] = {triton_rec_dir!r}
sys.path.insert(0, {triton_rec_dir!r})
import torch, triton
spec = importlib.util.spec_from_file_location('m', {str(inst_path)!r})
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
kfn = getattr(m, {plan['kernel']!r})
M, N, K = {M}, {N}, {K}
BM, BN, BK, GM = 128, 128, 32, 8
grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
n_ctas = grid[0]
RING = 12
As = [torch.randn((M, K), dtype=torch.float16, device='cuda') for _ in range(RING)]
Bs = [torch.randn((K, N), dtype=torch.float16, device='cuda') for _ in range(RING)]
C = torch.empty((M, N), dtype=torch.float32, device='cuda')
if {platform!r} == 'nv':
    import fi_triton as FT
    st_hw = {stages_for_variant.get(selected_var, 3) if platform == 'nv' else 3}
    buf = FT.FiBuffer(n_ctas, 4, {sel_cap}, gran=1, timer=1, level=2)
    cfg = FT.FiCfg(cap={sel_cap}, warps=1, n_slots=n_ctas, gran=1, timer=1, level=2, threads={threads}, arch={arch!r})
    for r in range(3):
        A, B = As[r], Bs[r]
        if r == 0: buf.reset()
        kfn[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), BM, BN, BK, GM, buf.rec, buf.overflow, cfg, {sel_s_arg}, 2, num_warps=4, num_stages=st_hw)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for r in range(3, RING):
        A, B = As[r], Bs[r]
        if r == RING - 1: buf.reset(); torch.cuda.synchronize()
        kfn[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), BM, BN, BK, GM, buf.rec, buf.overflow, cfg, {sel_s_arg}, 2, num_warps=4, num_stages=st_hw)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    buf.dump({str(tmp_dump)!r}, meta={{'kernel': {plan['kernel']!r}, 'arch': {arch!r}}})
else:
    import fi_triton_amd as FTA
    st_hw_amd = {stages_for_variant.get(selected_var, 2) if platform != 'nv' else 2}
    buf = FTA.FiPackedBuffer(n_ctas=n_ctas, cap={sel_cap}, gran=1, num_warps=4, timer_mode=1)
    for r in range(3):
        A, B = As[r], Bs[r]
        if r == 0: buf.reset()
        kfn[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), BM, BN, BK, GM, buf.rec, buf.overflow, 2, 1, {sel_cap}, 1, {threads}, 1, {sel_s_arg}, num_warps=4, num_stages=st_hw_amd)
    torch.cuda.synchronize()
    for r in range(3, RING):
        A, B = As[r], Bs[r]
        if r == RING - 1: buf.reset(); torch.cuda.synchronize()
        kfn[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), BM, BN, BK, GM, buf.rec, buf.overflow, 2, 1, {sel_cap}, 1, {threads}, 1, {sel_s_arg}, num_warps=4, num_stages=st_hw_amd)
    torch.cuda.synchronize()
    buf.dump({str(tmp_dump)!r}, extra_meta={{'kernel': {plan['kernel']!r}}})
""",
        )
        hw_result = _execute_profiler_wrap(
            plan=plan,
            payload_cmd=[sys.executable, "-B", str(burst_py)],
            tmp_dump=tmp_dump,
            dump_hw=dump_hw,
            names_path=names_path,
            out_dir=hw_dir,
            M=M, N=N, K=K,
        )
        if isinstance(hw_result, dict):
            hw_result["sw_variant_used"] = selected_var
            hw_result["overhead_selected_pct"] = ovh_by_var[selected_var]

    return {
        "shape": {"M": M, "N": N, "K": K},
        "c1_static_triton": c1_triton,
        "c2_correctness": {"L0": c2_l0, "sample1": c2_s1, "full": c2_full},
        "benchmark": {
            "rounds": rounds,
            "iters_per_round": iters,
            "perturbation_budget_pct": max_overhead_budget_pct,
            "selected_variant": selected_var,
            "overhead_selected_pct": ovh_by_var[selected_var],
            "overhead_within_10pct_budget": ovh_by_var[selected_var] <= max_overhead_budget_pct,
            "median_us": {v: round(med_by_var[v], 3) for v in all_t_vars},
            "overheads_pct": ovh_by_var,
            "single_gpu_tflops_L0": round(tflops_l0, 3),
            "overhead_sample1_pct": ovh_by_var["sample1"],
            "overhead_full_pct": ovh_by_var["full"],
            "raw_us": timings,
        },
        "c3_decode": {"sample1": c3_s1, "full": c3_full, "selected": c3_selected},
        "c4_sw_hw": hw_result,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Automatic cross-language / cross-hardware GEMM profiler")
    parser.add_argument("--source", required=True, help="Path to uninstrumented GEMM kernel (.cu, .hip/.cpp, .py)")
    parser.add_argument("--card", required=True, choices=("h20", "b300", "mi308x"))
    parser.add_argument("--arch", help="Explicit target arch (sm_90a, sm_103a, gfx942)")
    parser.add_argument("--kernel", help="Optional target kernel function name")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--phase", choices=("prep", "gpu", "all"), default="all",
                        help="prep=0-GPU CPU plan+compile+C1; gpu=1-GPU C2+bench+C3+C4; all=both")
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--no-hw", action="store_true", help="Skip nsys/rocprof-sys co-axial hardware capture")
    args = parser.parse_args(argv)

    source_path = Path(args.source).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in ("prep", "all"):
        prep = run_prep_phase(
            source_path, out_dir, card=args.card, arch=args.arch, kernel_name=args.kernel
        )
    else:
        prep_file = out_dir / "prep_summary.json"
        if not prep_file.is_file():
            raise RuntimeError(f"--phase gpu requires existing {prep_file}; run --phase prep first")
        prep = json.loads(prep_file.read_text(encoding="utf-8"))

    if args.phase == "prep":
        print(json.dumps(prep, indent=2, ensure_ascii=False))
        return 0

    lang = prep["plan"]["language"]
    if lang in ("cuda_cpp", "hip_cpp"):
        gpu_res = _run_gpu_cpp(
            prep, out_dir, M=args.m, N=args.n, K=args.k,
            warmup=args.warmup, iters=args.iters, rounds=args.rounds, with_hw=not args.no_hw
        )
    elif lang == "triton":
        gpu_res = _run_gpu_triton(
            prep, out_dir, M=args.m, N=args.n, K=args.k,
            warmup=args.warmup, iters=args.iters, rounds=args.rounds, with_hw=not args.no_hw
        )
    else:
        raise RuntimeError(f"Language {lang} runner should be invoked via dedicated module harness")

    receipt = {
        "schema": "weaveprofiler.auto-gemm-receipt/v1",
        "plan": prep["plan"],
        "c1_static": prep.get("c1_static"),
        **gpu_res,
    }
    _write_json(out_dir / "auto_gemm_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
