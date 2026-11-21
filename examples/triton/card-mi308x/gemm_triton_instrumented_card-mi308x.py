#!/usr/bin/env python3
"""Runnable MI308X FP16 GEMM with per-physical-wave WeaveProfiler intervals.

Verified package reference for the MI308X Triton route. Run it from any project;
``WEAVEPROFILER_ROOT`` overrides the skill root when needed.

The measured kernel is a 2048^3 Triton GEMM. On gfx942 each ``tl.dot`` lowers
to MFMA instructions. One FI record lane is one physical wave64, selected by
lane 0 after reading HW_ID. ``FI_LEVEL`` selects recorder verbosity only:
0=off, 1=entry-only, 2=full-events. ``FI_SAMPLE`` selects event density
independently: 0=every K block (full audit), 1=one representative K block
(fast sampled feedback). The final output writeback has its own pair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

SKILL_ROOT = Path(os.environ.get(
    "WEAVEPROFILER_ROOT", str(Path(__file__).resolve().parents[4])))
RECORDER = SKILL_ROOT / "recorders" / "triton"
if str(RECORDER) not in sys.path:
    sys.path.insert(0, str(RECORDER))
from fi_triton_amd import FiBuffer, fi_b, fi_close, fi_e, fi_open  # noqa: E402

CARD_FILE_TOKEN = "card-mi308x"
DEFAULT_M = 2048
DEFAULT_N = 2048
DEFAULT_K = 2048
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
FI_CAP = 512
EV_LOAD_A = tl.constexpr(1)
EV_LOAD_B = tl.constexpr(2)
EV_MMA = tl.constexpr(3)
EV_STORE = tl.constexpr(4)
EV_GEMM = tl.constexpr(10)


@triton.jit
def weaveprofiler_gemm_fp16_fi(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    fi_time, fi_tag, fi_aux, fi_overflow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    FI_LEVEL: tl.constexpr, FI_SAMPLE: tl.constexpr, FI_CAP: tl.constexpr,
    FI_WPC: tl.constexpr,
    FI_THREADS: tl.constexpr,
    FI_TIMER: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n = tl.zeros([FI_THREADS], dtype=tl.int32)
    slot = tl.zeros([FI_THREADS], dtype=tl.int32)
    selected = tl.full([FI_THREADS], 0, tl.int1)
    if FI_LEVEL != 0:
        slot, selected, n = fi_open(
            fi_time, fi_tag, fi_aux, fi_overflow,
            ROLE=0, GRAN=0, CAP=FI_CAP, WPC=FI_WPC, THREADS=FI_THREADS, TIMER=FI_TIMER)

    # FI_LEVEL controls recorder verbosity only: 0=off, 1=entry-only,
    # 2=full-events. FI_SAMPLE is orthogonal: 0=all K blocks (full audit),
    # 1=one representative K block (fast sampled feedback). A full audit always
    # requires FI_LEVEL=2 + FI_SAMPLE=0; entry-only FI_LEVEL=1 records no
    # semantic K-block events.
    if FI_LEVEL == 2:
        n = fi_b(fi_time, fi_tag, fi_aux, fi_overflow, slot, selected, n,
                 EVENT=EV_GEMM, aux=pid, CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
        if FI_SAMPLE == 1:
            n = fi_b(fi_time, fi_tag, fi_aux, fi_overflow, slot, selected, n,
                     EVENT=EV_LOAD_A, aux=0, CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            n = fi_e(fi_time, fi_tag, fi_aux, fi_overflow, slot, selected, n,
                     EVENT=EV_LOAD_A, aux=0, CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
            n = fi_b(fi_time, fi_tag, fi_aux, fi_overflow, slot, selected, n,
                     EVENT=EV_LOAD_B, aux=0, CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            n = fi_e(fi_time, fi_tag, fi_aux, fi_overflow, slot, selected, n,
                     EVENT=EV_LOAD_B, aux=0, CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
            n = fi_b(fi_time, fi_tag, fi_aux, fi_overflow, slot, selected, n,
                     EVENT=EV_MMA, aux=0, CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
            acc = tl.dot(a, b, acc)
            n = fi_e(fi_time, fi_tag, fi_aux, fi_overflow, slot, selected, n,
                     EVENT=EV_MMA, aux=0, CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    k_start: tl.constexpr = BLOCK_K if FI_LEVEL == 2 and FI_SAMPLE == 1 else 0

    for k0 in tl.range(k_start, K, BLOCK_K, loop_unroll_factor=1):
        if FI_LEVEL == 2:
            if FI_SAMPLE == 0:
                n = fi_b(fi_time, fi_tag, fi_aux, fi_overflow, slot, selected, n, EVENT=EV_LOAD_A, aux=k0,
                         CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K), other=0.0)
        if FI_LEVEL == 2:
            if FI_SAMPLE == 0:
                n = fi_e(fi_time, fi_tag, fi_aux, fi_overflow, slot, selected, n, EVENT=EV_LOAD_A, aux=k0,
                         CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
                n = fi_b(fi_time, fi_tag, fi_aux, fi_overflow, slot, selected, n, EVENT=EV_LOAD_B, aux=k0,
                         CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
        b = tl.load(b_ptrs, mask=((k0 + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        if FI_LEVEL == 2:
            if FI_SAMPLE == 0:
                n = fi_e(fi_time, fi_tag, fi_aux, fi_overflow, slot, selected, n, EVENT=EV_LOAD_B, aux=k0,
                         CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
                n = fi_b(fi_time, fi_tag, fi_aux, fi_overflow, slot, selected, n, EVENT=EV_MMA, aux=k0,
                         CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
        acc = tl.dot(a, b, acc)
        if FI_LEVEL == 2:
            if FI_SAMPLE == 0:
                n = fi_e(fi_time, fi_tag, fi_aux, fi_overflow, slot, selected, n, EVENT=EV_MMA, aux=k0,
                         CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if FI_LEVEL == 2:
        n = fi_b(fi_time, fi_tag, fi_aux, fi_overflow, slot, selected, n, EVENT=EV_STORE, aux=pid,
                 CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    if FI_LEVEL == 2:
        n = fi_e(fi_time, fi_tag, fi_aux, fi_overflow, slot, selected, n, EVENT=EV_STORE, aux=pid,
                 CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
        n = fi_e(fi_time, fi_tag, fi_aux, fi_overflow, slot, selected, n, EVENT=EV_GEMM, aux=pid,
                 CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
    if FI_LEVEL != 0:
        n = fi_close(fi_time, fi_tag, fi_aux, fi_overflow, slot, selected, n,
                     CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
                    encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_visible_gpu0() -> None:
    expected = {"ROCR_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0"}
    got = {k: os.environ.get(k) for k in expected}
    if got != expected:
        raise RuntimeError(f"GPU0 lock missing: expected {expected}, got {got}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible ROCm GPU")
    torch.cuda.set_device(0)


def _load_inputs(m: int, n: int, k: int, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a_cpu = torch.randn((m, k), generator=gen, dtype=torch.float32).half()
    b_cpu = torch.randn((k, n), generator=gen, dtype=torch.float32).half()
    a = a_cpu.to(device="cuda")
    b = b_cpu.to(device="cuda")
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    return a_cpu, b_cpu, a, b, out


def _launch(a, b, out, args, fb: FiBuffer | None = None):
    m, k = a.shape
    _, n = b.shape
    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
    if fb is None:
        fi_args = (a, a, a, a)
        level, sample, cap, wpc, threads, timer = 0, 0, 1, 64, args.num_warps * 64, 0
    else:
        fi_args = fb.ptr_args()
        level, sample, cap, wpc, threads, timer = (
            args.fi_level, args.fi_sample, fb.cap, fb.wpc, fb.threads, fb.timer_mode)
    return weaveprofiler_gemm_fp16_fi[grid](
        a, b, out, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1), *fi_args,
        BLOCK_M, BLOCK_N, BLOCK_K,
        FI_LEVEL=level, FI_SAMPLE=sample, FI_CAP=cap, FI_WPC=wpc,
        FI_THREADS=threads, FI_TIMER=timer,
        num_warps=args.num_warps, num_stages=args.num_stages,
    )


def _compare(got: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    got64 = got.detach().to(device="cpu", dtype=torch.float64)
    ref64 = ref.detach().to(device="cpu", dtype=torch.float64)
    diff = (got64 - ref64).abs()
    return {
        "checked_elements": int(ref.numel()),
        "full_output_checked": True,
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
        "allclose_rtol_1e-2_atol_0.25": bool(torch.allclose(got64, ref64, rtol=1e-2, atol=0.25)),
        "bitwise_equal_to_uninstrumented": bool(torch.equal(got.detach().cpu(), ref.detach().to(device="cpu", dtype=got.dtype))),
        "finite": bool(torch.isfinite(got).all().item()),
    }


def _capture_variant(level: int, sample: int) -> str:
    if level == 0:
        return "off"
    if level == 1:
        return "entry_only"
    if level == 2 and sample == 0:
        return "full_audit_all_k_blocks"
    if level == 2 and sample == 1:
        return "fast_sampled_reference_k_block"
    raise ValueError(f"invalid FI_LEVEL/FI_SAMPLE pair: {level}/{sample}")


def _expected_records(level: int, sample: int, k_blocks: int, timer: int) -> int:
    if level == 0:
        base = 0
    elif level == 1:
        base = 2  # OPEN + CLOSE
    elif level == 2 and sample == 0:
        # OPEN + GEMM begin/end + C-store pair + CLOSE, plus 6 records per K block.
        # The historical 390-record reference predates the GEMM envelope pair;
        # the corrected full-audit layout has 392 with two timer anchors at 64 blocks.
        base = 6 + 6 * k_blocks
    elif level == 2 and sample == 1:
        base = 12  # envelope + one representative K block + store
    else:
        raise ValueError(f"invalid FI_LEVEL/FI_SAMPLE pair: {level}/{sample}")
    return base + (2 if timer == 1 and level >= 1 else 0)


def _record_layout(level: int, sample: int) -> str:
    if level == 0:
        return "no records"
    if level == 1:
        return "OPEN + CLOSE (entry-only; no semantic K-block events)"
    if level == 2 and sample == 0:
        return ("OPEN + gemm begin + (A load, B load, MMA)*all K_blocks + "
                "C store + gemm end + CLOSE")
    if level == 2 and sample == 1:
        return ("OPEN + gemm begin + representative A load + B load + MMA + "
                "C store + gemm end + CLOSE")
    raise ValueError(f"invalid FI_LEVEL/FI_SAMPLE pair: {level}/{sample}")


def _sampling_description(level: int, sample: int) -> str:
    if level == 0:
        return "off"
    if level == 1:
        return "FI_SAMPLE is not applicable: FI_LEVEL=1 records entry only"
    if level == 2 and sample == 0:
        return "FI_SAMPLE=0: every K block per physical wave (full audit)"
    if level == 2 and sample == 1:
        return "FI_SAMPLE=1: one representative K block per physical wave (fast sampled feedback)"
    raise ValueError(f"invalid FI_LEVEL/FI_SAMPLE pair: {level}/{sample}")


def _evidence_contract(level: int, sample: int) -> dict[str, Any]:
    return {
        "correctness": (
            "independent FP32/FP64 reference; every element of every output is checked; "
            "FI_SAMPLE changes event density only and never correctness coverage"),
        "performance_gate": "steady-state kernel time only; maximum instrumentation overhead is 10%",
        "fast_sampled": {"fi_level": 2, "fi_sample": 1, "delivery_evidence": False},
        "full_audit": {"fi_level": 2, "fi_sample": 0},
        "entry_only": {
            "fi_level": 1,
            "fi_sample": "not applicable",
            "warning": "entry-only must never be labeled or used as full-audit evidence",
        },
        "this_run": {"fi_level": level, "fi_sample": sample},
    }


def _base(args: argparse.Namespace) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    grid = triton.cdiv(args.m, BLOCK_M) * triton.cdiv(args.n, BLOCK_N)
    return {
        "schema": "weaveprofiler-fi-gemm/v1",
        "card": "mi308x",
        "card_file_token": CARD_FILE_TOKEN,
        "arch": props.gcnArchName,
        "operation": "weaveprofiler_gemm_fp16_fi (self-written Triton tl.dot -> MFMA)",
        "language": "triton",
        "dtype": "float16",
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "grid": grid,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
        "k_blocks": triton.cdiv(args.k, BLOCK_K),
        "fi_cap": FI_CAP,
        "fi_level": args.fi_level,
        "fi_sample": args.fi_sample,
        "capture_variant": _capture_variant(args.fi_level, args.fi_sample),
        "fi_timer": args.fi_timer,
        "evidence_contract": _evidence_contract(args.fi_level, args.fi_sample),
        "fi_events": {
            "1": {"source_label": "A load", "semantic_id": "load", "aux": "k_block"},
            "2": {"source_label": "B load", "semantic_id": "load", "aux": "k_block"},
            "3": {"source_label": "MMA", "semantic_id": "mma", "aux": "k_block"},
            "4": {"source_label": "C store", "semantic_id": "store", "aux": "cta"},
            "10": {"source_label": "GEMM wave", "semantic_id": "gemm", "aux": "cta"},
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "gcnArchName": props.gcnArchName,
            "multi_processor_count": props.multi_processor_count,
            "total_memory_bytes": props.total_memory,
            "uuid": str(props.uuid),
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "triton": triton.__version__,
            "platform": platform.platform(),
        },
    }


def _names() -> dict[str, Any]:
    return {
        "events": {"1": "load", "2": "load", "3": "mma", "4": "store", "10": "gemm"},
        "semantics": {"1": "load", "2": "load", "3": "mma", "4": "store", "10": "gemm"},
        "roles": {"0": "GEMM physical wave"},
        "aux": {"1": "k_block", "2": "k_block", "3": "k_block", "4": "cta", "10": "cta"},
        "label_aux": True,
        "group": "cu",
    }


def run(args: argparse.Namespace) -> int:
    _require_visible_gpu0()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    a_cpu, b_cpu, a, b, out = _load_inputs(args.m, args.n, args.k, args.seed)

    # Compile and establish the uninstrumented output on exactly the same inputs.
    _launch(a, b, out, args)
    torch.cuda.synchronize()
    baseline = out.clone()
    ref = torch.matmul(a_cpu.to(torch.float64), b_cpu.to(torch.float64))
    baseline_check = _compare(baseline, ref)

    fb = FiBuffer(n_ctas=args.m // BLOCK_M * (args.n // BLOCK_N), cap=FI_CAP,
                  gran=0, num_warps=args.num_warps, timer_mode=args.fi_timer)
    fb.reset()
    capture_token = f"fi-l{args.fi_level}-s{args.fi_sample}"
    torch.cuda.nvtx.range_push(
        f"weaveprofiler_gemm_fi_{CARD_FILE_TOKEN}_{args.m}x{args.n}x{args.k}_fp16_{capture_token}")
    handle = _launch(a, b, out, args, fb)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    counts, overflow = fb.read_counts()
    active = [int(c) for c in counts if c]
    k_blocks = args.k // BLOCK_K
    expected_records = _expected_records(
        args.fi_level, args.fi_sample, k_blocks, args.fi_timer)
    capture_token = f"fi-l{args.fi_level}-s{args.fi_sample}"
    dump_path = outdir / (
        f"wave_events__gemm_custom__{CARD_FILE_TOKEN}__bs{args.m}_sq{args.n}"
        f"__fp16__{capture_token}.fi02.bin")
    dump = fb.dump(str(dump_path), meta=_base(args), extra_meta={
        "fi_cap": FI_CAP,
        "expected_records_per_wave": expected_records,
        "record_layout": _record_layout(args.fi_level, args.fi_sample),
        "sampling": _sampling_description(args.fi_level, args.fi_sample),
        "timer_mode": ("s_memtime with OPEN/CLOSE s_memrealtime anchors"
                       if args.fi_timer == 1 else "s_memrealtime on every record"),
    })
    counts = dump.pop("counts")
    dump["active_lanes"] = len(active)
    dump["record_min"] = min(active) if active else 0
    dump["record_max"] = max(active) if active else 0
    dump["expected_records_per_active_lane"] = expected_records
    dump["count_matches_contract"] = (
        (args.fi_level == 0 and not active)
        or (bool(active) and all(int(c) == expected_records for c in active)))

    names_path = outdir / f"names__gemm_custom__{CARD_FILE_TOKEN}__{capture_token}.json"
    _write_json(names_path, _names())
    instrumented_check = _compare(out, ref)
    instrumented_check["bitwise_equal_to_uninstrumented"] = bool(torch.equal(out, baseline))
    report = {
        **_base(args),
        "status": "PASS" if baseline_check["allclose_rtol_1e-2_atol_0.25"] and
                           instrumented_check["allclose_rtol_1e-2_atol_0.25"] and
                           not overflow and dump["count_matches_contract"] else "FAIL",
        "same_run_uninstrumented_vs_cpu_float64": baseline_check,
        "same_run_instrumented_vs_cpu_float64": instrumented_check,
        "correctness_evidence": {
            "reference": "independent CPU float64 torch.matmul",
            "full_output_checked": True,
            "event_sampling_does_not_change_correctness_scope": True,
        },
        "performance_evidence_contract": {
            "gate_metric": "steady_state_kernel_time_only",
            "maximum_overhead_fraction": 0.10,
            "fast_sampled_requires": {"fi_level": 2, "fi_sample": 1},
            "full_audit_requires": {"fi_level": 2, "fi_sample": 0},
            "fi_level_1_is_entry_only": True,
        },
        "wave_events_dump": dump,
        "weaveprofiler_names": str(names_path),
        "wave_events_output": str(dump_path),
        "wave_events_output_sha256": _sha256(dump_path),
        "instruction_evidence": {
            "amdgcn_contains_v_mfma": "v_mfma" in (getattr(handle, "asm", {}) or {}).get("amdgcn", ""),
            "amdgcn_v_mfma_count": (getattr(handle, "asm", {}) or {}).get("amdgcn", "").count("v_mfma"),
        },
        "timestamp_ns": time.time_ns(),
        "delivery_evidence": False,
    }
    report_path = outdir / (
        f"weaveprofiler_report__gemm_custom__{CARD_FILE_TOKEN}__bs{args.m}_sq{args.n}"
        f"__fp16__{capture_token}.json")
    _write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return 0 if report["status"] == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--num-stages", type=int, default=1)
    parser.add_argument(
        "--fi-level", type=int, choices=(0, 1, 2), default=2,
        help="recorder verbosity only: 0=off, 1=entry-only, 2=full-events")
    parser.add_argument(
        "--fi-sample", type=int, choices=(0, 1), default=0,
        help="event density for FI_LEVEL=2: 0=all K blocks (full audit), "
             "1=one representative K block (fast sampled feedback)")
    parser.add_argument("--fi-timer", type=int, choices=(0, 1), default=0,
                        help="0=s_memrealtime every record; 1=s_memtime events with wall-clock anchors")
    parser.add_argument(
        "--output-dir", type=Path,
        default=here.parent.parent / "fi"
    )
    args = parser.parse_args(argv)
    if args.m % BLOCK_M or args.n % BLOCK_N or args.k % BLOCK_K:
        parser.error("M/N/K must be divisible by the configured block sizes")
    if args.fi_sample == 1 and args.fi_level != 2:
        parser.error("--fi-sample 1 requires --fi-level 2; FI_LEVEL=1 is entry-only")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
