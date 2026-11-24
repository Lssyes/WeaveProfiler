"""SW+HW Causal Bottleneck & Optimization-Opportunity Attribution Engine (Pillar 2).

Transforms validated software timelines + hardware PMU curves into:
  1. Hardware-calibrated regime segmentation (TENSOR_SATURATED, DRAM_READ_BOUND,
     SM_UNDERFILLED, UNDERUTILIZED_DIP).
  2. Quantitative SW->HW causal proofs (chip-wide active-SM fraction vs HW Tensor /
     SM-Active time-series Pearson correlation, straggler barrier walls, bimodal CTA
     workload clusters, and exact wasted SM-time %).
  3. Objective semantic-span validity & refinement verdicts:
       - SATURATED_STOP_REFINING : Phase already saturates its bound HW unit.
       - CAUSE_EXPLAINED         : Underutilized phase already explained by SW spans
                                   (wait/barrier/straggler/grid underfill).
       - NEED_SUBSTAGE_DRILLDOWN : Material underutilized phase with unexplained dip.
  4. Ranked optimization opportunities with honest Amdahl / work-redistribution
     counterfactual speedup upper bounds.
  5. Deterministic context-placement & compiler search-space planning (`WP_CTX_SMEM`
     routing at 0-register headroom + `{pre, post}` search around `setmaxnreg`).
"""
from __future__ import annotations

from collections import defaultdict
import math
import re
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CAUSAL_SCHEMA = "weaveprof.causal-report/v1"

# Compact, verifiable physical architecture constants (no prose KB bloat).
ARCH_PROFILES: Dict[str, Dict[str, Any]] = {
    "sm_90a": {
        "card": "h20",
        "platform": "nv",
        "default_sms": 78,
        "max_smem_bytes": 232448,
        "reg_file_per_sm": 65536,
        "max_regs_per_thread": 255,
        # H20 Tensor Active PMU is normalized to full GH100 pipe width (peak raw ~25%).
        "tensor_active_scale": 4.0,
        "tensor_raw_uncalibrated_ceiling": 29.0,
        "tensor_saturated_pct": 85.0,
        "dram_saturated_pct": 70.0,
        # Dual-denominator roofline constants (H20 nominal vs 1830 MHz empirical ceiling).
        "nominal_tflops_bf16": 148.0,
        "empirical_tflops_bf16": 142.9,
        "nominal_tflops_fp8": 296.0,
        "empirical_tflops_fp8": 282.5,
        "nominal_hbm_tbps": 4.0,
        "empirical_hbm_tbps": 3.40,
        "empirical_tensor_ratio": 0.96,
        "empirical_hbm_ratio": 0.85,
    },
    "sm_90": {
        "card": "h20",
        "platform": "nv",
        "default_sms": 78,
        "max_smem_bytes": 232448,
        "reg_file_per_sm": 65536,
        "max_regs_per_thread": 255,
        "tensor_active_scale": 4.0,
        "tensor_raw_uncalibrated_ceiling": 29.0,
        "tensor_saturated_pct": 85.0,
        "dram_saturated_pct": 70.0,
        "nominal_tflops_bf16": 148.0,
        "empirical_tflops_bf16": 142.9,
        "nominal_tflops_fp8": 296.0,
        "empirical_tflops_fp8": 282.5,
        "nominal_hbm_tbps": 4.0,
        "empirical_hbm_tbps": 3.40,
        "empirical_tensor_ratio": 0.96,
        "empirical_hbm_ratio": 0.85,
    },
    "sm_103a": {
        "card": "b300",
        "platform": "nv",
        "default_sms": 148,
        "max_smem_bytes": 233472,
        "reg_file_per_sm": 65536,
        "max_regs_per_thread": 255,
        "tensor_active_scale": 1.0,
        "tensor_raw_uncalibrated_ceiling": 101.0,
        "tensor_saturated_pct": 80.0,
        "dram_saturated_pct": 70.0,
        "nominal_hbm_tbps": 8.0,
        "empirical_hbm_tbps": 6.80,
        "empirical_tensor_ratio": 0.94,
        "empirical_hbm_ratio": 0.85,
    },
    "sm_100a": {
        "card": "b300",
        "platform": "nv",
        "default_sms": 148,
        "max_smem_bytes": 233472,
        "reg_file_per_sm": 65536,
        "max_regs_per_thread": 255,
        "tensor_active_scale": 1.0,
        "tensor_raw_uncalibrated_ceiling": 101.0,
        "tensor_saturated_pct": 80.0,
        "dram_saturated_pct": 70.0,
        "nominal_hbm_tbps": 8.0,
        "empirical_hbm_tbps": 6.80,
        "empirical_tensor_ratio": 0.94,
        "empirical_hbm_ratio": 0.85,
    },
    "sm_86": {
        "card": "sm86",
        "platform": "nv",
        "default_sms": 16,
        "max_smem_bytes": 101376,
        "reg_file_per_sm": 65536,
        "max_regs_per_thread": 255,
        "tensor_active_scale": 1.0,
        "tensor_raw_uncalibrated_ceiling": 101.0,
        "tensor_saturated_pct": 80.0,
        "dram_saturated_pct": 70.0,
        "empirical_tensor_ratio": 0.92,
        "empirical_hbm_ratio": 0.85,
        "default_timer": 1,
    },
    "gfx942": {
        "card": "mi308x",
        "platform": "amd",
        "default_sms": 80,
        "max_smem_bytes": 65536,
        "reg_file_per_sm": 65536,
        "max_regs_per_thread": 256,
        # Unlike H20 (where nsys normalizes Tensor Active to full GH100 pipe width -> x4),
        # export-regulated MI308X caps FP8 at 398.7 TFLOPS (vs 606.2T @ 1.85 GHz unthrottled CDNA3)
        # via BOTH dynamic shader clock throttling (~1.378 GHz -> 451.5T raw pipe clock ceiling)
        # AND power/issue duty-cycle capping (~88.3% max duty at 1.378 GHz -> 398.7T SKU cap).
        # Consequently:
        #   - Raw pipe duty 100 * SQ_VALU_MFMA_BUSY_CYCLES / (80 * GRBM_GUI_ACTIVE) divides by
        #     ~451.5T (reaching 82.9% max / 76.3% p95 at 376.7T = 94.5% of 398.7T; multiplying by
        #     606.2/398.7 = 1.5205x would overflow to 126.1%, whereas 451.5/398.7 = 1.1325x).
        #   - Timestamped periodic FP8 PMC sampling computes exact SKU utilization directly via
        #     100 * (d_mfma * 1024 FLOPs) / (dt * 398.7e12) -> 94.5% on dense GEMM, 70.4% whole-GPU
        #     (83.4% on the 540/640 Compute wave pool) on EP16 MegaMoE.
        "tensor_active_scale": 1.0,
        "tensor_raw_uncalibrated_ceiling": 101.0,
        "tensor_saturated_pct": 70.0,
        "dram_saturated_pct": 70.0,
        "unthrottled_cdna3_tflops_fp8": 606.2,
        "raw_pipe_clock_tflops_fp8": 451.5,
        "pipe_to_sku_fp8_rebase": 1.1325,
        "nominal_tflops_fp8": 398.7,
        "empirical_tflops_fp8": 376.7,
        "empirical_shader_ghz_mfma": 1.378,
        "nominal_tflops_bf16": 199.3,
        "nominal_hbm_tbps": 5.30,
        "empirical_hbm_tbps": 3.684,
        "empirical_tensor_ratio": 0.85,
        "empirical_hbm_ratio": 0.70,
    },
}

_WAIT_NAME_RE = re.compile(r"(?:wait|barrier|grid_sync|sync_wait|stall|idle|spin|poll)", re.I)
_COMM_NAME_RE = re.compile(r"(?:allreduce|allgather|alltoall|dispatch|combine|nvlink|comm|rdma)", re.I)
_SETMAXNREG_RE = re.compile(
    r"\b(?:setmaxnreg\.(?:dec|inc)|warpgroup_reg_(?:dealloc|alloc)|reg_dealloc|reg_alloc)\b",
    re.I,
)
_GRID_SYNC_RE = re.compile(r"\b(?:grid_barrier|grid_sync|cooperative_groups::this_grid|atomicAdd.*barrier)\b", re.I)
_ROLE_SWITCH_RE = re.compile(r"\b(?:FI_ROLE_SWITCH|WP_ROLE_SWITCH|role_switch)\b", re.I)


def get_arch_profile(arch: Optional[str] = None, gpu_name: Optional[str] = None) -> Dict[str, Any]:
    """Resolve compact physical constants for H20 / B300 / MI308X."""
    key = (arch or "").strip().lower()
    gpu = (gpu_name or "").strip().lower()
    if key in ARCH_PROFILES:
        prof = dict(ARCH_PROFILES[key])
        prof["arch"] = key
        return prof
    if "h20" in gpu or "h20" in key:
        prof = dict(ARCH_PROFILES["sm_90a"])
        prof["arch"] = "sm_90a"
        return prof
    if "b300" in gpu or "l20d" in gpu or "103" in key or "100" in key:
        prof = dict(ARCH_PROFILES["sm_103a"])
        prof["arch"] = "sm_103a"
        return prof
    if "mi308" in gpu or "gfx942" in key or "amd" in gpu:
        prof = dict(ARCH_PROFILES["gfx942"])
        prof["arch"] = "gfx942"
        return prof
    if "sm_86" in key or "sm86" in key or "sm86" in gpu or "rtx 30" in gpu or "rtx30" in gpu or "ampere" in gpu:
        prof = dict(ARCH_PROFILES["sm_86"])
        prof["arch"] = "sm_86"
        return prof
    return {
        "arch": key or "unknown",
        "card": "unknown",
        "platform": "nv",
        "default_sms": 0,
        "max_smem_bytes": 232448,
        "reg_file_per_sm": 65536,
        "max_regs_per_thread": 255,
        "tensor_active_scale": 1.0,
        "tensor_raw_uncalibrated_ceiling": 101.0,
        "tensor_saturated_pct": 80.0,
        "dram_saturated_pct": 70.0,
    }


def plan_context_and_search_space(
    source_text: str,
    *,
    arch: Optional[str] = None,
    gpu_name: Optional[str] = None,
    block_threads: Optional[int] = None,
    l0_regs: Optional[int] = None,
    l0_smem_bytes: int = 0,
    smem_limit_bytes: Optional[int] = None,
    max_warps_per_cta: int = 32,
) -> Dict[str, Any]:
    """Deterministically plan recorder context placement and compiler search arms.

    Replaces heuristic rules with measurable hardware resource arithmetic:
      - If register headroom <= 2 regs/thread and SMEM has >= 16B free -> WP_CTX_SMEM.
      - If `setmaxnreg` / warpgroup reg realloc is present -> emit {pre, post} search
        space for the Supervisor to empirically compile & rank by spill/STL/overhead.
    """
    prof = get_arch_profile(arch, gpu_name)
    smem_limit = int(smem_limit_bytes) if smem_limit_bytes is not None else int(prof["max_smem_bytes"])
    smem_free = max(0, smem_limit - max(0, int(l0_smem_bytes)))

    reg_budget: Optional[int] = None
    reg_headroom: Optional[int] = None
    if block_threads and int(block_threads) > 0:
        reg_budget = min(int(prof["max_regs_per_thread"]), int(prof["reg_file_per_sm"]) // int(block_threads))
        if l0_regs is not None:
            reg_headroom = reg_budget - int(l0_regs)

    compile_defines: List[str] = []
    reasons: List[str] = []
    # Compiler-in-the-loop tiered plan (D2):
    #   Arm 1 (default fast): 1-register bit-packed context (`-DWP_CTX_REG`, 0 B SMEM).
    #   Arm 2 (if spill/STL regression & SMEM free >= 16B): `-DWP_CTX_SMEM` (0 live regs, 16B SMEM + __noinline__).
    #   Arm 3 (if spill/STL regression & SMEM exhausted): `-DWP_CTX_REG_NOINLINE` (1 live reg, 0 B SMEM, __noinline__).
    fallback_arms: List[Dict[str, Any]] = []
    if reg_headroom is not None and reg_headroom <= 2:
        if smem_free >= 16:
            context_mode = "WP_CTX_SMEM"
            compile_defines.append("-DWP_CTX_SMEM")
            reasons.append(
                f"Zero/tight register headroom ({l0_regs}/{reg_budget} regs @ {block_threads} thr) "
                f"with {smem_free} B free SMEM -> route context to 16B static SMEM + __noinline__ cold path."
            )
            fallback_arms = [
                {"mode": "WP_CTX_REG_NOINLINE", "defines": ["-DWP_CTX_REG_NOINLINE"], "smem_bytes_per_slot": 0},
                {"mode": "WP_CTX_REG", "defines": ["-DWP_CTX_REG"], "smem_bytes_per_slot": 0},
            ]
        else:
            context_mode = "WP_CTX_REG_NOINLINE"
            compile_defines.append("-DWP_CTX_REG_NOINLINE")
            reasons.append(
                f"Both register headroom ({reg_headroom}) and SMEM ({smem_free} B free) are exhausted -> "
                f"use 1-register bit-packed context + __noinline__ cold outlining (0 B SMEM) and verify spill/STL gate."
            )
            fallback_arms = [
                {"mode": "WP_CTX_REG_CTA", "defines": ["-DWP_CTX_REG_NOINLINE", "-DWP_GRAN=1"], "smem_bytes_per_slot": 0},
            ]
    else:
        context_mode = "WP_CTX_REG"
        compile_defines.append("-DWP_CTX_REG")
        if smem_free < 16:
            reasons.append(
                f"SMEM saturated ({l0_smem_bytes}/{smem_limit} B) -> default to 1-register bit-packed WpCtxReg (0 B SMEM); "
                f"on compiler spill/STL regression, fall back to WP_CTX_REG_NOINLINE."
            )
            fallback_arms = [
                {"mode": "WP_CTX_REG_NOINLINE", "defines": ["-DWP_CTX_REG_NOINLINE"], "smem_bytes_per_slot": 0},
            ]
        else:
            reasons.append(
                "Try 1-register bit-packed WpCtxReg (0 B SMEM) first; "
                "if ptxas/cuobjdump reports spill or STL/LDL regression, fall back to WP_CTX_SMEM (16B SMEM + __noinline__)."
            )
            fallback_arms = [
                {"mode": "WP_CTX_SMEM", "defines": ["-DWP_CTX_SMEM"], "smem_bytes_per_slot": 16},
                {"mode": "WP_CTX_REG_NOINLINE", "defines": ["-DWP_CTX_REG_NOINLINE"], "smem_bytes_per_slot": 0},
            ]

    has_setmaxnreg = bool(_SETMAXNREG_RE.search(source_text or ""))
    placement_search_space = ["pre_setmaxnreg", "post_setmaxnreg"] if has_setmaxnreg else ["default_entry"]
    if has_setmaxnreg:
        reasons.append(
            "Detected setmaxnreg / warpgroup register reallocation -> compiler-in-the-loop search over "
            "{pre_setmaxnreg, post_setmaxnreg} ranked by spill/STL delta and steady-state overhead."
        )

    has_grid_sync = bool(_GRID_SYNC_RE.search(source_text or ""))
    recommended_granularity = "warp" if (has_setmaxnreg or max_warps_per_cta > 1) else "sm"
    if context_mode == "WP_CTX_SMEM" and smem_free < 16 * max_warps_per_cta:
        recommended_granularity = "sm"
        compile_defines.append("-DWP_GRAN=1")

    return {
        "arch": prof["arch"],
        "context_mode": context_mode,
        "compile_defines": compile_defines,
        "fallback_arms": fallback_arms,
        "placement_search_space": placement_search_space,
        "recommended_granularity": recommended_granularity,
        "has_setmaxnreg": has_setmaxnreg,
        "has_grid_sync": has_grid_sync,
        "has_role_switch_macro": bool(_ROLE_SWITCH_RE.search(source_text or "")),
        "reg_budget_per_thread": reg_budget,
        "reg_headroom": reg_headroom,
        "smem_free_bytes": smem_free,
        "reasons": reasons,
    }


def select_compiler_feedback_arm(
    candidates: Sequence[Mapping[str, Any]],
    *,
    baseline_regs: Optional[int] = None,
    baseline_spill_bytes: int = 0,
    baseline_stl: int = 0,
) -> Dict[str, Any]:
    """Rank empirically compiled recorder arms using compiler spill/STL telemetry and overhead.

    Each candidate dict contains:
      mode, defines, regs, spill_bytes, stl_count, ldl_count, overhead_pct (optional),
      build_ok (bool, default True).
    Prefers: (1) build_ok, (2) zero added spill bytes, (3) zero added STL instructions,
    (4) lowest measured overhead_pct (if available), (5) lowest register count.
    """
    scored: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
    for cand in candidates:
        ok = bool(cand.get("build_ok", True))
        spill_delta = max(0, int(cand.get("spill_bytes") or 0) - int(baseline_spill_bytes))
        stl_delta = max(0, int(cand.get("stl_count") or 0) - int(baseline_stl))
        regs = int(cand.get("regs") or 999)
        reg_delta = (regs - int(baseline_regs)) if baseline_regs is not None else regs
        ovh = cand.get("overhead_pct")
        ovh_val = float(ovh) if ovh is not None else 1e6
        entry = {
            "mode": str(cand.get("mode") or "unknown"),
            "defines": list(cand.get("defines") or []),
            "placement": str(cand.get("placement") or "default"),
            "build_ok": ok,
            "regs": regs,
            "reg_delta": reg_delta,
            "spill_delta_bytes": spill_delta,
            "stl_delta": stl_delta,
            "overhead_pct": ovh,
            "spill_clean": (ok and spill_delta == 0 and stl_delta == 0),
        }
        key = (
            0 if ok else 1,
            0 if spill_delta == 0 else 1,
            spill_delta,
            0 if stl_delta == 0 else 1,
            stl_delta,
            ovh_val,
            regs,
        )
        scored.append((key, entry))
    scored.sort(key=lambda x: x[0])
    winner = scored[0][1] if scored else None
    return {
        "winner": winner,
        "ranked": [e for _, e in scored],
        "fallback_triggered": bool(
            winner and candidates and winner["mode"] != str(candidates[0].get("mode") or "")
        ),
    }


def _safe_mean(vals: Sequence[float]) -> float:
    clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return (sum(clean) / len(clean)) if clean else 0.0


def _safe_percentile(vals: Sequence[float], q: float) -> float:
    clean = sorted(float(v) for v in vals if v is not None and math.isfinite(float(v)))
    if not clean:
        return 0.0
    idx = min(len(clean) - 1, max(0, int(round(q * (len(clean) - 1)))))
    return clean[idx]


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = min(len(xs), len(ys))
    if n < 5:
        return None
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    vx = sum((x - mx) ** 2 for x in xs[:n])
    vy = sum((y - my) ** 2 for y in ys[:n])
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return max(-1.0, min(1.0, cov / math.sqrt(vx * vy)))


def _extract_curve_xy(entry: Any) -> Tuple[List[float], List[float]]:
    if isinstance(entry, Mapping):
        ts = entry.get("t_us") or entry.get("t") or []
        vs = entry.get("val") or entry.get("v") or entry.get("values") or []
        if ts and vs and len(ts) == len(vs):
            return [float(t) for t in ts], [float(v) for v in vs]
        if isinstance(entry.get("series"), list):
            pts = entry["series"]
            if pts and isinstance(pts[0], (list, tuple)) and len(pts[0]) >= 2:
                return [float(p[0]) for p in pts], [float(p[1]) for p in pts]
    elif isinstance(entry, list) and entry and isinstance(entry[0], (list, tuple)) and len(entry[0]) >= 2:
        return [float(p[0]) for p in entry], [float(p[1]) for p in entry]
    return [], []


def calibrate_hw_series(
    hw_series: Mapping[str, Any],
    *,
    arch: Optional[str] = None,
    gpu_name: Optional[str] = None,
    kernel_us: Optional[float] = None,
) -> Dict[str, Any]:
    """Normalize and architecture-calibrate hardware PMU time series."""
    prof = get_arch_profile(arch, gpu_name)
    out: Dict[str, Dict[str, Any]] = {}
    tensor_scale = 1.0

    for key, raw in (hw_series or {}).items():
        ts, vs = _extract_curve_xy(raw)
        if kernel_us is not None and kernel_us > 0 and ts:
            in_pts = [(t, v) for t, v in zip(ts, vs) if -1e-6 <= t <= kernel_us + 1e-6]
            if in_pts:
                in_ts = [p[0] for p in in_pts]
                in_vs = [p[1] for p in in_pts]
            else:
                in_ts, in_vs = ts, vs
        else:
            in_ts, in_vs = ts, vs

        mean_val = (
            float(raw["mean_in_win"])
            if isinstance(raw, Mapping) and raw.get("mean_in_win") is not None
            else _safe_mean(in_vs)
        )
        max_val = (
            float(raw["max_in_win"])
            if isinstance(raw, Mapping) and raw.get("max_in_win") is not None
            else (max(in_vs) if in_vs else 0.0)
        )
        p90_val = (
            float(raw["p90_in_win"])
            if isinstance(raw, Mapping) and raw.get("p90_in_win") is not None
            else _safe_percentile(in_vs, 0.90)
        )
        in_win_count = (
            int(raw["in_win_count"])
            if isinstance(raw, Mapping) and raw.get("in_win_count") is not None
            else len(in_vs)
        )

        scale = 1.0
        if key in ("tensor_active", "compute_util") and prof["tensor_active_scale"] > 1.0:
            if max(mean_val, max_val) <= float(prof["tensor_raw_uncalibrated_ceiling"]):
                scale = float(prof["tensor_active_scale"])
                tensor_scale = scale

        cal_vs = [min(100.0, v * scale) for v in in_vs]
        cal_mean = min(100.0, mean_val * scale)
        cal_max = min(100.0, max_val * scale)
        cal_p90 = min(100.0, p90_val * scale)
        out[key] = {
            "t_us": in_ts,
            "val": cal_vs,
            "raw_mean": round(mean_val, 3),
            "raw_max": round(max_val, 3),
            "mean_pct": round(cal_mean, 3),
            "p90_pct": round(cal_p90, 3),
            "max_pct": round(cal_max, 3),
            "in_win_count": in_win_count,
            "scale_applied": scale,
        }

    return {"curves": out, "tensor_calibration_factor": tensor_scale, "arch_profile": prof}


def _window_curve_stats(curve: Mapping[str, Any], t0: float, t1: float) -> Optional[Dict[str, float]]:
    ts = curve.get("t_us") or []
    vs = curve.get("val") or []
    if not ts or len(ts) != len(vs) or t1 <= t0:
        return None
    sub = [v for t, v in zip(ts, vs) if t0 <= t <= t1]
    if not sub:
        return None
    return {
        "n": len(sub),
        "mean_pct": round(_safe_mean(sub), 2),
        "p90_pct": round(_safe_percentile(sub, 0.90), 2),
        "max_pct": round(max(sub), 2),
        "min_pct": round(min(sub), 2),
    }


def _classify_regime(
    *,
    tensor_mean: float,
    tensor_peak: float,
    dram_read_mean: float,
    dram_read_peak: float,
    sm_active_mean: float,
    prof: Mapping[str, Any],
) -> str:
    t_sat = float(prof.get("tensor_saturated_pct", 80.0))
    d_sat = float(prof.get("dram_saturated_pct", 70.0))
    if tensor_mean >= t_sat or (tensor_peak >= t_sat + 5.0 and tensor_mean >= t_sat - 10.0):
        return "TENSOR_SATURATED"
    if dram_read_mean >= d_sat - 10.0 or (dram_read_peak >= d_sat + 10.0 and tensor_mean < 25.0):
        return "DRAM_READ_BOUND"
    if sm_active_mean > 0 and sm_active_mean < 90.0:
        return "SM_UNDERFILLED"
    return "UNDERUTILIZED_DIP"


def _normalize_from_perfetto(doc: Mapping[str, Any], arch: Optional[str], total_sms: Optional[int]) -> Dict[str, Any]:
    """Adapt a standard aligned WeaveProfiler Perfetto JSON into the causal IR."""
    other = doc.get("otherData") or {}
    al = other.get("alignment") or {}
    detected_arch = arch or other.get("arch") or al.get("arch") or ("gfx942" if other.get("platform") == "amd" else None)
    gpu_name = other.get("gpu") or other.get("card") or ""
    prof = get_arch_profile(detected_arch, gpu_name)

    events = doc.get("traceEvents") or []
    hw_raw: Dict[str, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    sw_spans: List[Dict[str, Any]] = []
    cta_bounds: Dict[int, Dict[str, Any]] = {}
    stage_durs: Dict[str, List[float]] = defaultdict(list)
    stage_windows: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    t_min_sw: Optional[float] = None
    t_max_sw: Optional[float] = None

    for ev in events:
        ph = ev.get("ph")
        if ph == "C":
            cname = str(ev.get("name") or "").lower()
            args = ev.get("args") or {}
            val = next((float(v) for v in args.values() if isinstance(v, (int, float))), None)
            ts = ev.get("ts")
            if val is None or ts is None:
                continue
            metric_key = None
            if any(k in cname for k in ("tensor", "compute_util", "mfma", "pipe_tensor")):
                metric_key = "tensor_active"
            elif any(k in cname for k in ("sm_active", "sms active", "cu_active")):
                metric_key = "sm_active"
            elif any(k in cname for k in ("dram_read", "dram read", "memory_util", "hbm_read")):
                metric_key = "dram_read"
            elif any(k in cname for k in ("dram_write", "dram write", "hbm_write")):
                metric_key = "dram_write"
            elif "nvlink" in cname and "protocol" not in cname and "tx" in cname:
                metric_key = "nvlink_tx"
            elif "nvlink" in cname and "protocol" not in cname and "rx" in cname:
                metric_key = "nvlink_rx"
            elif "issue" in cname:
                metric_key = "warp_issue"
            if metric_key:
                hw_raw[metric_key][0].append(float(ts))
                hw_raw[metric_key][1].append(val)
        elif ph == "X":
            pid = ev.get("pid")
            try:
                pid_i = int(pid)
            except (TypeError, ValueError):
                pid_i = 0
            if pid_i < 100000:
                continue
            ts = float(ev.get("ts") or 0.0)
            dur = float(ev.get("dur") or 0.0)
            if dur < 0:
                continue
            te = ts + dur
            t_min_sw = ts if t_min_sw is None else min(t_min_sw, ts)
            t_max_sw = te if t_max_sw is None else max(t_max_sw, te)
            args = ev.get("args") or {}
            cta = int(args.get("cta", -1))
            smid = int(args.get("sm", args.get("smid", -1)))
            if cta >= 0:
                rec = cta_bounds.setdefault(cta, {"cta": cta, "smid": smid, "open_us": ts, "close_us": te, "spans": []})
                rec["open_us"] = min(rec["open_us"], ts)
                rec["close_us"] = max(rec["close_us"], te)
            if args.get("kind") == "residency":
                continue
            sname = str(args.get("detail") or ev.get("name") or args.get("semantic_id") or "stage")
            sem = str(args.get("semantic_id") or sname).lower()
            sw_spans.append({"name": sname, "semantic": sem, "cta": cta, "smid": smid, "t0": ts, "t1": te, "dur": dur})
            stage_durs[sname].append(dur)
            stage_windows[sname].append((ts, te))
            if cta >= 0:
                cta_bounds[cta]["spans"].append((sname, ts, te))

    t0_ref = t_min_sw or 0.0
    hw_series = {
        k: {"t_us": [t - t0_ref for t in ts], "val": vs, "in_win_count": len(vs)}
        for k, (ts, vs) in hw_raw.items()
    }
    for rec in cta_bounds.values():
        rec["open_us"] -= t0_ref
        rec["close_us"] -= t0_ref
        rec["spans"] = [(s, a - t0_ref, b - t0_ref) for s, a, b in rec["spans"]]

    sw_span_us = (t_max_sw - t_min_sw) if (t_min_sw is not None and t_max_sw is not None) else 0.0
    kernel_us = float(al.get("kernel_duration_us") or al.get("hw_kernel_us") or sw_span_us or 0.0)
    stages = []
    for sname, durs in stage_durs.items():
        wins = [(a - t0_ref, b - t0_ref) for a, b in stage_windows[sname]]
        stages.append({
            "name": sname,
            "is_wait": bool(_WAIT_NAME_RE.search(sname)),
            "median_us": round(statistics.median(durs), 3),
            "min_us": round(min(durs), 3),
            "max_us": round(max(durs), 3),
            "total_us": round(sum(durs), 3),
            "count": len(durs),
            "window_t0": round(min(a for a, _ in wins), 3),
            "window_t1": round(max(b for _, b in wins), 3),
        })

    return {
        "target": str(other.get("kernel") or other.get("target") or "perfetto_trace"),
        "arch": prof["arch"],
        "gpu": gpu_name,
        "total_sms": int(total_sms or prof["default_sms"] or len({r["smid"] for r in cta_bounds.values() if r["smid"] >= 0})),
        "kernel_us": kernel_us,
        "sw_span_us": sw_span_us,
        "hw_series": hw_series,
        "per_sm_records": list(cta_bounds.values()),
        "stages": stages,
        "cta_warps": [],
        "extra": {},
    }


def _normalize_from_capture_json(doc: Mapping[str, Any], arch: Optional[str], total_sms: Optional[int]) -> Dict[str, Any]:
    """Adapt structured SW+HW capture JSON (e.g., H20/B300 experiment bundles) into causal IR."""
    detected_arch = arch or doc.get("arch")
    gpu_name = str(doc.get("gpu") or "").strip()
    prof = get_arch_profile(detected_arch, gpu_name)
    sms = int(total_sms or doc.get("sms") or prof["default_sms"] or 0)

    # Select primary HW series (for 2-kernel pipelines like K1+K2, keep K2 and overall both accessible).
    hw_series = doc.get("hw_series") or doc.get("hw_k2") or doc.get("hw_both") or {}
    if doc.get("nsys_kernel_us") is not None:
        kernel_us = float(doc["nsys_kernel_us"])
    elif doc.get("nsys_k2_us") is not None:
        kernel_us = float(doc["nsys_k2_us"])
    else:
        kernel_us = float(doc.get("sw_span_us") or 0.0)

    sw_span_us = float(doc.get("sw_span_us") or (doc.get("trace_wp") or {}).get("span_us") or 0.0)
    phase_names_map = {str(k): str(v) for k, v in (doc.get("phase_names") or {}).items()}

    # Normalize per-SM records with named spans where possible.
    raw_per_sm = (
        doc.get("per_sm_w0")
        or (doc.get("trace_wp") or {}).get("per_sm_w0")
        or doc.get("k2_sm")
        or doc.get("all_sm_span")
        or []
    )
    per_sm_records: List[Dict[str, Any]] = []
    span_by_id: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)

    for rec in raw_per_sm:
        cta = int(rec.get("cta", len(per_sm_records)))
        smid = int(rec.get("smid", -1))
        spans_norm: List[Tuple[str, float, float]] = []
        raw_spans = rec.get("spans") or []
        t_open = rec.get("open_us")
        t_close = rec.get("close_us")
        for sp in raw_spans:
            if len(sp) == 4:
                ev_id, _aux, t0, t1 = sp
            elif len(sp) == 3:
                ev_id, t0, t1 = sp
            else:
                continue
            ev_key = str(ev_id)
            ev_name = phase_names_map.get(ev_key, f"ev_{ev_key}")
            t0f, t1f = float(t0), float(t1)
            spans_norm.append((ev_name, t0f, t1f))
            span_by_id[ev_name].append((cta, t0f, t1f))
            t_open = t0f if t_open is None else min(float(t_open), t0f)
            t_close = t1f if t_close is None else max(float(t_close), t1f)
        per_sm_records.append({
            "cta": cta,
            "smid": smid,
            "open_us": float(t_open or 0.0),
            "close_us": float(t_close or 0.0),
            "spans": spans_norm,
        })

    if sw_span_us <= 0 and per_sm_records:
        sw_span_us = max(r["close_us"] for r in per_sm_records) - min(r["open_us"] for r in per_sm_records)

    # Build stage table from explicit summaries or per-SM spans.
    stages: List[Dict[str, Any]] = []
    if isinstance(doc.get("phase_stats_all78sm"), Mapping):
        for name, st in doc["phase_stats_all78sm"].items():
            pts = span_by_id.get(name) or []
            t0_win = min((p[1] for p in pts), default=None)
            t1_win = max((p[2] for p in pts), default=None)
            stages.append({
                "name": name,
                "is_wait": bool(_WAIT_NAME_RE.search(name)),
                "median_us": float(st.get("median_us", 0.0)),
                "min_us": float(st.get("min_us", 0.0)),
                "max_us": float(st.get("max_us", 0.0)),
                "skew_us": float(st.get("skew_us", 0.0)),
                "count": len(pts) or len(per_sm_records),
                "window_t0": round(t0_win, 3) if t0_win is not None else None,
                "window_t1": round(t1_win, 3) if t1_win is not None else None,
            })
    elif isinstance(doc.get("k2_phase_summary"), Mapping):
        for name, st in doc["k2_phase_summary"].items():
            cnt = int(st.get("count", 0))
            med = float(st.get("median_us", 0.0))
            stages.append({
                "name": name,
                "is_wait": bool(_WAIT_NAME_RE.search(name)),
                "median_us": med,
                "p90_us": float(st.get("p90_us", med)),
                "total_us": round(med * cnt, 3),
                "count": cnt,
                "window_t0": 0.0,
                "window_t1": round(kernel_us, 3),
            })
    elif isinstance(doc.get("event_summary"), Mapping):
        for name, st in doc["event_summary"].items():
            cnt = int(st.get("count", 0))
            tot = float(st.get("total_warp_us", st.get("total_us", 0.0)))
            stages.append({
                "name": name,
                "is_wait": bool(_WAIT_NAME_RE.search(name)),
                "median_us": round(tot / max(1, cnt), 3),
                "total_us": round(tot, 3),
                "count": cnt,
                "window_t0": None,
                "window_t1": None,
            })
    elif span_by_id:
        for name, pts in sorted(span_by_id.items()):
            durs = [b - a for _, a, b in pts]
            stages.append({
                "name": name,
                "is_wait": bool(_WAIT_NAME_RE.search(name)),
                "median_us": round(statistics.median(durs), 3),
                "min_us": round(min(durs), 3),
                "max_us": round(max(durs), 3),
                "skew_us": round(max(durs) - min(durs), 3),
                "count": len(durs),
                "window_t0": round(min(a for _, a, _ in pts), 3),
                "window_t1": round(max(b for _, _, b in pts), 3),
            })

    cta_warps = doc.get("cta0_warps") or (doc.get("trace_wp") or {}).get("cta0_all_warps") or []
    # Extract optional compile/resource & workload caliber metadata if present.
    compile_l0 = (doc.get("compile") or {}).get("l0") if isinstance(doc.get("compile"), Mapping) else {}
    workload = dict(doc.get("workload") or doc.get("caliber_input") or {})
    if isinstance(compile_l0, Mapping):
        workload.setdefault("reg_per_thread", compile_l0.get("regs") or compile_l0.get("k2_regs"))
        workload.setdefault("smem_bytes", compile_l0.get("smem_bytes") or compile_l0.get("smem"))
    extra = {
        "p4_load_imbalance_all78sm": doc.get("p4_load_imbalance_all78sm"),
        "removed_empty_stubs": doc.get("removed_empty_stubs"),
        "dynamic_role_switch_warp6_median_us": doc.get("dynamic_role_switch_warp6_median_us"),
        "dynamic_role_switch_count": doc.get("dynamic_role_switch_count"),
        "nsys_k1_us": doc.get("nsys_k1_us"),
        "nsys_k2_us": doc.get("nsys_k2_us"),
        "hw_k1": doc.get("hw_k1"),
        "hw_both": doc.get("hw_both"),
        "workload": workload,
    }
    return {
        "target": str(doc.get("target") or "capture"),
        "arch": prof["arch"],
        "gpu": gpu_name,
        "total_sms": sms,
        "kernel_us": kernel_us,
        "sw_span_us": sw_span_us,
        "hw_series": hw_series,
        "per_sm_records": per_sm_records,
        "stages": stages,
        "cta_warps": cta_warps,
        "extra": extra,
    }


def _detect_barrier_stragglers(
    per_sm_records: Sequence[Mapping[str, Any]],
    kernel_us: float,
    total_sms: int,
    hw_curves: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Generically detect synchronized grid-barrier straggler walls across all SMs.

    Works both when barrier spans are explicitly named (e.g. `grid_barrier_wait`) AND
    when only numeric event IDs are present (synchronized release timestamp + bimodal
    wait durations across CTAs).
    """
    if len(per_sm_records) < 8 or kernel_us <= 0:
        return []

    # Group occurrences by (event_name, barrier_release_cluster).
    # First cluster spans with similar end timestamps across CTAs.
    by_ev: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    for rec in per_sm_records:
        cta = int(rec.get("cta", 0))
        for sname, t0, t1 in rec.get("spans") or []:
            if t1 > t0:
                by_ev[sname].append((cta, t0, t1))

    # Tolerance for synchronized barrier release across SMs: max(15 us, 0.5% of kernel).
    sync_tol = max(15.0, 0.005 * kernel_us)
    walls: List[Dict[str, Any]] = []

    for sname, pts in by_ev.items():
        # Sort by release timestamp t1 and cluster into synchronized barrier waves.
        # A single barrier wave contains at most 1 span per CTA; encountering an
        # already-seen CTA or a timestamp jump > sync_tol starts a new wave.
        pts_sorted = sorted(pts, key=lambda x: x[2])
        clusters: List[List[Tuple[int, float, float]]] = []
        cluster_ctas: List[set] = []
        for item in pts_sorted:
            cta_i = item[0]
            if (
                not clusters
                or (item[2] - clusters[-1][-1][2]) > sync_tol
                or cta_i in cluster_ctas[-1]
            ):
                clusters.append([item])
                cluster_ctas.append({cta_i})
            else:
                clusters[-1].append(item)
                cluster_ctas[-1].add(cta_i)

        for cl in clusters:
            ctas = {c for c, _, _ in cl}
            if len(ctas) < max(8, int(0.75 * len(per_sm_records))):
                continue
            durs = [t1 - t0 for _, t0, t1 in cl]
            min_d = min(durs)
            max_d = max(durs)
            skew = max_d - min_d
            is_named_wait = bool(_WAIT_NAME_RE.search(sname))
            t_rel_end = max(t1 for _, _, t1 in cl)
            t_rel_min = min(t1 for _, _, t1 in cl)
            # Check barrier release tightness: a grid barrier releases all CTAs nearly together.
            release_spread = t_rel_end - t_rel_min
            if not is_named_wait:
                # For unnamed numeric spans, require tight synchronized release AND large wait skew
                # AND some CTAs waiting very briefly while others wait much longer.
                if release_spread > sync_tol * 2.0 or skew < max(30.0, 0.015 * kernel_us):
                    continue
                if min_d > 0.25 * max_d:
                    continue
            else:
                if skew < max(15.0, 0.01 * kernel_us):
                    continue

            # Split into fast-arriving (idle/waiting) CTAs vs slow-straggler CTAs via midpoint threshold.
            thresh = min_d + 0.40 * skew
            waiting_ctas = [d for d in durs if d >= thresh]
            straggler_ctas = [d for d in durs if d < thresh]
            if not waiting_ctas or not straggler_ctas:
                continue

            t_wait_start = min(t0 for _, t0, _ in cl)
            # Deduplicate per CTA (take total wait per CTA inside this barrier cluster) so multi-slice
            # waits on the same CTA never inflate avg_wait_us above the physical wall window.
            per_cta_wait: Dict[int, float] = defaultdict(float)
            for cta_id, a_w, b_w in cl:
                per_cta_wait[cta_id] += max(0.0, b_w - a_w)
            total_wait_sm_us = sum(per_cta_wait.values())
            denom_sms = max(total_sms, len(per_sm_records), len(per_cta_wait), 1)
            eff_kernel_us = max(kernel_us, t_rel_end)
            wasted_chip_pct = min(100.0, 100.0 * total_wait_sm_us / max(1e-6, denom_sms * eff_kernel_us))
            # Counterfactual upper bound if work before this barrier were perfectly balanced across SMs:
            # Saved wall time = average wait time across all chip SMs = total_wait_sm_us / denom_sms.
            avg_wait_us = min(eff_kernel_us * 0.95, total_wait_sm_us / denom_sms)
            counterfactual_speedup_pct = 100.0 * avg_wait_us / max(1e-3, eff_kernel_us - avg_wait_us)

            # Compute quantitative SW->HW causal correlation over [t_wait_start - pre_margin, t_rel_end].
            # Ending at t_rel_end avoids mixing in an unrelated post-barrier phase (e.g. RMSNorm after GEMM/GDN).
            corr_tensor = None
            hw_drop = None
            t_curve = hw_curves.get("tensor_active") or hw_curves.get("compute_util")
            if t_curve and t_curve.get("t_us"):
                pre_margin = max(20.0, 0.75 * (t_rel_end - t_wait_start))
                w0 = max(0.0, t_wait_start - pre_margin)
                w1 = min(kernel_us, t_rel_end)
                ts_hw = t_curve["t_us"]
                vs_hw = t_curve["val"]
                sw_active_frac: List[float] = []
                hw_sub: List[float] = []
                before_hw: List[float] = []
                during_hw: List[float] = []
                for t_pt, v_pt in zip(ts_hw, vs_hw):
                    if w0 <= t_pt <= w1:
                        # Fraction of CTAs NOT sitting in this barrier wait at time t_pt
                        n_waiting_now = sum(1 for _, a, b in cl if a <= t_pt <= b)
                        sw_active_frac.append(1.0 - n_waiting_now / len(cl))
                        hw_sub.append(v_pt)
                        if t_pt < t_wait_start:
                            before_hw.append(v_pt)
                        elif t_wait_start <= t_pt <= t_rel_end:
                            during_hw.append(v_pt)
                corr_tensor = _pearson(sw_active_frac, hw_sub)
                if before_hw and during_hw:
                    hw_drop = {
                        "before_mean_pct": round(_safe_mean(before_hw), 2),
                        "during_mean_pct": round(_safe_mean(during_hw), 2),
                        "delta_pct": round(_safe_mean(during_hw) - _safe_mean(before_hw), 2),
                    }

            walls.append({
                "event": sname,
                "window_us": [round(t_wait_start, 2), round(t_rel_end, 2)],
                "ctas_total": len(cl),
                "early_waiting_ctas": len(waiting_ctas),
                "straggler_ctas": len(straggler_ctas),
                "waiting_median_us": round(statistics.median(waiting_ctas), 2),
                "straggler_wait_median_us": round(statistics.median(straggler_ctas), 2),
                "skew_us": round(skew, 2),
                "wasted_chip_sm_time_pct": round(wasted_chip_pct, 2),
                "saved_wall_us_ideal": round(avg_wait_us, 2),
                "counterfactual_speedup_pct": round(counterfactual_speedup_pct, 2),
                "sw_hw_tensor_pearson_r": round(corr_tensor, 3) if corr_tensor is not None else None,
                "hw_tensor_drop": hw_drop,
            })

    walls.sort(key=lambda w: w["wasted_chip_sm_time_pct"], reverse=True)
    return walls


def _detect_warp_tail_imbalance(cta_warps: Sequence[Mapping[str, Any]], kernel_us: float) -> Optional[Dict[str, Any]]:
    """Detect intra-CTA warp specialization tail imbalance (early-exiting warp groups)."""
    if len(cta_warps) < 4 or kernel_us <= 0:
        return None
    end_times: List[Tuple[int, int, float]] = []
    for w in cta_warps:
        wid = int(w.get("warp", -1))
        role = int(w.get("role", -1))
        spans = w.get("spans") or []
        if not spans:
            continue
        t_last = max(float(sp[-1]) for sp in spans if len(sp) >= 3)
        end_times.append((wid, role, t_last))
    if len(end_times) < 4:
        return None
    max_t = max(t for _, _, t in end_times)
    early = [(wid, role, t) for wid, role, t in end_times if max_t - t > max(50.0, 0.15 * max_t)]
    if not early or len(early) == len(end_times):
        return None
    return {
        "total_warps_observed": len(end_times),
        "early_exit_warps": [wid for wid, _, _ in early],
        "early_exit_count": len(early),
        "early_exit_median_us": round(statistics.median(t for _, _, t in early), 2),
        "latest_warp_end_us": round(max_t, 2),
        "idle_tail_us": round(max_t - statistics.median(t for _, _, t in early), 2),
    }


def analyze_sw_hw_causality(
    data: Mapping[str, Any],
    *,
    arch: Optional[str] = None,
    total_sms: Optional[int] = None,
) -> Dict[str, Any]:
    """Run deterministic SW+HW causal bottleneck attribution and semantic-validity analysis.

    Accepts either an aligned Perfetto JSON trace (`traceEvents` present) or a
    structured SW+HW capture JSON bundle (`hw_series` + `per_sm_w0`/`stages`).
    """
    if "traceEvents" in data:
        ir = _normalize_from_perfetto(data, arch, total_sms)
    else:
        ir = _normalize_from_capture_json(data, arch, total_sms)

    cal = calibrate_hw_series(
        ir["hw_series"],
        arch=ir["arch"],
        gpu_name=ir["gpu"],
        kernel_us=ir["kernel_us"],
    )
    curves = cal["curves"]
    prof = cal["arch_profile"]

    sm_curve = curves.get("sm_active") or {}
    tensor_curve = curves.get("tensor_active") or curves.get("compute_util") or {}
    dram_r_curve = curves.get("dram_read") or {}
    dram_w_curve = curves.get("dram_write") or {}
    issue_curve = curves.get("warp_issue") or {}

    sm_mean = float(sm_curve.get("mean_pct") or 0.0)
    tensor_mean = float(tensor_curve.get("mean_pct") or 0.0)
    tensor_peak = float(tensor_curve.get("max_pct") or 0.0)
    dram_r_mean = float(dram_r_curve.get("mean_pct") or 0.0)
    dram_r_peak = float(dram_r_curve.get("max_pct") or 0.0)
    dram_w_mean = float(dram_w_curve.get("mean_pct") or 0.0)
    issue_mean = float(issue_curve.get("mean_pct") or 0.0)

    global_regime = _classify_regime(
        tensor_mean=tensor_mean,
        tensor_peak=tensor_peak,
        dram_read_mean=dram_r_mean,
        dram_read_peak=dram_r_peak,
        sm_active_mean=sm_mean,
        prof=prof,
    )

    kernel_us = float(ir["kernel_us"] or ir["sw_span_us"] or 0.0)
    sw_span_us = float(ir["sw_span_us"] or 0.0)
    total_sms_val = int(ir["total_sms"] or prof["default_sms"] or 0)
    observed_ctas = len(ir["per_sm_records"])
    observed_sms = len({r["smid"] for r in ir["per_sm_records"] if r.get("smid", -1) >= 0})

    # 1. Detect chip-wide grid-barrier straggler walls + SW->HW Pearson proof.
    straggler_walls = _detect_barrier_stragglers(
        ir["per_sm_records"], kernel_us, total_sms_val, curves
    )

    # 2. Detect intra-CTA warp tail imbalance.
    warp_tail = _detect_warp_tail_imbalance(ir["cta_warps"], kernel_us)

    # 3. Evaluate semantic-span validity & refinement verdicts per stage.
    stage_verdicts: List[Dict[str, Any]] = []
    explained_windows = [w["window_us"] for w in straggler_walls]
    total_stage_dur = sum(
        float(s.get("median_us", 0.0)) for s in ir["stages"] if not s.get("is_wait")
    ) or max(1.0, kernel_us)

    sum_stage_totals = sum(float(x.get("total_us") or 0.0) for x in ir["stages"])
    has_repeated_loop_stages = bool(
        sum_stage_totals > 0
        and any(int(x.get("count") or 0) > max(1, observed_ctas) * 2 for x in ir["stages"])
    )

    for st in ir["stages"]:
        name = st["name"]
        med_us = float(st.get("median_us", 0.0))
        is_wait = bool(st.get("is_wait"))
        t0 = st.get("window_t0")
        t1 = st.get("window_t1")
        if has_repeated_loop_stages and st.get("total_us") is not None:
            share_pct = round(100.0 * float(st["total_us"]) / max(1e-9, sum_stage_totals), 2)
        elif kernel_us > 0 and med_us <= kernel_us * 1.05:
            share_pct = round(100.0 * (med_us / max(1e-9, kernel_us)), 2)
        else:
            share_pct = round(
                100.0 * float(st.get("total_us") or med_us) / max(
                    1e-9, sum(float(x.get("total_us") or x.get("median_us") or 0.0) for x in ir["stages"])
                ),
                2,
            )

        # Evaluate hardware regime specifically during this stage's window if known.
        st_tensor = _window_curve_stats(tensor_curve, t0, t1) if (t0 is not None and t1 is not None and t1 > t0) else None
        st_dram_r = _window_curve_stats(dram_r_curve, t0, t1) if (t0 is not None and t1 is not None and t1 > t0) else None
        st_sm = _window_curve_stats(sm_curve, t0, t1) if (t0 is not None and t1 is not None and t1 > t0) else None

        eff_t_mean = st_tensor["mean_pct"] if st_tensor else tensor_mean
        eff_t_peak = st_tensor["max_pct"] if st_tensor else tensor_peak
        eff_d_mean = st_dram_r["mean_pct"] if st_dram_r else dram_r_mean
        eff_d_peak = st_dram_r["max_pct"] if st_dram_r else dram_r_peak
        eff_sm_mean = st_sm["mean_pct"] if st_sm else sm_mean

        st_regime = _classify_regime(
            tensor_mean=eff_t_mean,
            tensor_peak=eff_t_peak,
            dram_read_mean=eff_d_mean,
            dram_read_peak=eff_d_peak,
            sm_active_mean=eff_sm_mean,
            prof=prof,
        )

        # Objective semantic-span validity verdict:
        #   - Empty stub check: median <= 0.128 us (128 ns)
        #   - Saturated HW unit: STOP REFINING (no optimization ROI from subdividing)
        #   - Wait/barrier or overlaps explained straggler/underfill: CAUSE_EXPLAINED
        #   - Material underutilized compute stage (>8% share) without explanation: NEED_SUBSTAGE_DRILLDOWN
        overlaps_wall = bool(
            t0 is not None and t1 is not None
            and any(not (t1 < w0 or t0 > w1) for w0, w1 in explained_windows)
        )
        if med_us <= 0.128 and not is_wait:
            verdict = "SUSPECT_EMPTY_STUB"
            reason = f"Median duration {med_us * 1000.0:.1f} ns <= 128 ns stub floor; verify non-empty SASS body."
        elif is_wait:
            verdict = "CAUSE_EXPLAINED"
            reason = "Explicit synchronization/wait boundary separating physical stall from active compute."
        elif st_regime in ("TENSOR_SATURATED", "DRAM_READ_BOUND"):
            verdict = "SATURATED_STOP_REFINING"
            reason = (
                f"Hardware unit saturated ({st_regime}: Tensor={eff_t_mean:.1f}%, "
                f"DRAM_Rd={eff_d_mean:.1f}%); subdividing this span has minimal optimization ROI."
            )
        elif overlaps_wall or (st.get("skew_us", 0.0) > max(30.0, 0.15 * max(med_us, 1.0))):
            verdict = "CAUSE_EXPLAINED"
            reason = "Underutilization attributed to cross-SM load imbalance / barrier straggler skew."
        elif st_regime == "SM_UNDERFILLED":
            verdict = "CAUSE_EXPLAINED"
            reason = f"Chip SM underfill ({eff_sm_mean:.1f}% SMs active) already explains reduced chip throughput."
        elif share_pct >= 8.0 and eff_t_mean < 60.0 and eff_d_mean < 60.0:
            verdict = "NEED_SUBSTAGE_DRILLDOWN"
            reason = (
                f"Material stage ({share_pct:.1f}% share) in UNDERUTILIZED_DIP "
                f"(Tensor={eff_t_mean:.1f}%, DRAM_Rd={eff_d_mean:.1f}%) with unexplained stall; drill down."
            )
        else:
            verdict = "VALID_SUFFICIENT"
            reason = "Non-empty stage with low wall-time share or bounded overhead."

        stage_verdicts.append({
            "stage": name,
            "is_wait": is_wait,
            "median_us": round(med_us, 3),
            "skew_us": round(float(st.get("skew_us", 0.0)), 3),
            "share_pct": share_pct,
            "hw_regime": st_regime,
            "window_tensor_mean_pct": st_tensor["mean_pct"] if st_tensor else None,
            "window_dram_read_mean_pct": st_dram_r["mean_pct"] if st_dram_r else None,
            "refinement_verdict": verdict,
            "reason": reason,
        })

    # 4. Synthesize ranked causal optimization opportunities with counterfactual bounds.
    opportunities: List[Dict[str, Any]] = []

    # 4a. Grid-barrier straggler / bimodal load imbalance opportunities.
    for w in straggler_walls:
        corr_str = (
            f", SW-active vs HW-Tensor Pearson r={w['sw_hw_tensor_pearson_r']:+.2f}"
            if w["sw_hw_tensor_pearson_r"] is not None
            else ""
        )
        drop_str = ""
        if w["hw_tensor_drop"]:
            d = w["hw_tensor_drop"]
            drop_str = f" (HW Tensor drops {d['before_mean_pct']:.1f}% -> {d['during_mean_pct']:.1f}%)"
        opportunities.append({
            "category": "LOAD_IMBALANCE_GRID_BARRIER",
            "title": f"Rebalance work before barrier '{w['event']}' ({w['early_waiting_ctas']}/{w['ctas_total']} SMs idling)",
            "evidence": (
                f"{w['early_waiting_ctas']} CTAs wait median {w['waiting_median_us']:.1f} us vs "
                f"{w['straggler_ctas']} straggler CTAs ({w['straggler_wait_median_us']:.1f} us) in "
                f"[{w['window_us'][0]:.1f}, {w['window_us'][1]:.1f}] us; wastes "
                f"{w['wasted_chip_sm_time_pct']:.2f}% of total chip SM-time{drop_str}{corr_str}."
            ),
            "counterfactual_speedup_upper_bound_pct": w["counterfactual_speedup_pct"],
            "wasted_chip_sm_time_pct": w["wasted_chip_sm_time_pct"],
            "sw_hw_pearson_r": w["sw_hw_tensor_pearson_r"],
            "recommended_optimization": (
                "Split heavy head/token tasks (Split-KV / Split-Head / persistent dynamic work queue) "
                "across all physical SMs before the grid barrier."
            ),
        })

    # 4b. Chip-wide SM underfill opportunity (e.g. 128 CTAs on 148-SM B300 -> 20 idle SMs).
    if total_sms_val > 0 and observed_ctas > 0 and observed_ctas < total_sms_val and sm_mean < 92.0:
        idle_sms = total_sms_val - observed_ctas
        ideal_speedup = 100.0 * (total_sms_val / float(observed_ctas) - 1.0)
        # If this is K2 inside a K1+K2 pair, also compute end-to-end bound.
        k1_us = ir["extra"].get("nsys_k1_us")
        k2_us = ir["extra"].get("nsys_k2_us")
        e2e_bound = ideal_speedup
        scope_note = ""
        if k1_us and k2_us:
            k2_ideal = float(k2_us) * (observed_ctas / float(total_sms_val))
            e2e_bound = 100.0 * ((float(k1_us) + float(k2_us)) / (float(k1_us) + k2_ideal) - 1.0)
            scope_note = f" (K2-only ideal +{ideal_speedup:.1f}%, K1+K2 end-to-end +{e2e_bound:.1f}%)"
        opportunities.append({
            "category": "CHIP_SM_UNDERFILL",
            "title": f"Eliminate {idle_sms}/{total_sms_val} idle SMs ({observed_ctas} CTAs launched on {total_sms_val}-SM GPU)",
            "evidence": (
                f"SW observes {observed_ctas} CTAs across {observed_sms or observed_ctas} SMs while "
                f"HW SMs Active averages {sm_mean:.1f}% (Tensor={tensor_mean:.1f}%, DRAM_Rd={dram_r_mean:.1f}%){scope_note}."
            ),
            "counterfactual_speedup_upper_bound_pct": round(e2e_bound, 2),
            "idle_sms": idle_sms,
            "recommended_optimization": (
                f"Increase parallelism / chunk-split or fuse persistent schedule to occupy all {total_sms_val} SMs."
            ),
        })

    # 4c. DRAM Read Bandwidth Roofline Bound (e.g. B300 Decode at 76.4% mean / 94% peak DRAM vs 3.5% Tensor).
    if global_regime == "DRAM_READ_BOUND":
        opportunities.append({
            "category": "DRAM_BANDWIDTH_BOUND",
            "title": f"Memory-bandwidth bound regime (DRAM Read {dram_r_mean:.1f}% mean / {dram_r_peak:.1f}% peak vs Tensor {tensor_mean:.1f}%)",
            "evidence": (
                f"HW PMU proves kernel is HBM read-bandwidth limited (DRAM Read mean={dram_r_mean:.1f}%, "
                f"peak={dram_r_peak:.1f}%, Tensor={tensor_mean:.1f}%); compute-side micro-tuning cannot break roofline."
            ),
            "counterfactual_speedup_upper_bound_pct": round(max(0.0, 100.0 - dram_r_mean), 2),
            "recommended_optimization": (
                "Reduce HBM bytes moved per step: compress/quantize recurrent state (e.g. BF16->FP8 state), "
                "increase effective batch size, or fuse adjacent HBM-touching passes."
            ),
        })

    # 4d. Exposed explicit wait stages (e.g. cp.async wait or Conv/Rec producer waits).
    wait_stages = [s for s in stage_verdicts if s["is_wait"] and s["share_pct"] >= 3.0]
    for ws in wait_stages:
        if any(ws["stage"] == w["event"] for w in straggler_walls):
            continue
        bound = round(100.0 * ws["share_pct"] / max(1.0, 100.0 - ws["share_pct"]), 2)
        opportunities.append({
            "category": "MEMORY_OR_SYNC_WAIT_EXPOSURE",
            "title": f"Hide exposed wait stage '{ws['stage']}' ({ws['share_pct']:.1f}% of stage time)",
            "evidence": (
                f"SW stage '{ws['stage']}' accounts for {ws['share_pct']:.1f}% (median {ws['median_us']:.2f} us) "
                f"in regime {ws['hw_regime']} (Tensor={tensor_mean:.1f}%, DRAM_Rd={dram_r_mean:.1f}%)."
            ),
            "counterfactual_speedup_upper_bound_pct": min(bound, 50.0),
            "recommended_optimization": (
                "Deepen software pipeline prefetch distance / overlap producer-consumer barriers with independent math."
            ),
        })

    # 4e. Intra-CTA warp tail / dynamic role repurposing (e.g. B300 Decode w8-11 early exit).
    if warp_tail and warp_tail["idle_tail_us"] > 0.15 * max(1.0, kernel_us):
        tail_pct = round(100.0 * warp_tail["idle_tail_us"] / max(1.0, kernel_us), 2)
        role_sw_us = ir["extra"].get("dynamic_role_switch_warp6_median_us")
        role_note = f"; dynamic role switch captured at t={role_sw_us:.1f} us" if role_sw_us else ""
        opportunities.append({
            "category": "INTRABLOCK_WARP_TAIL",
            "title": f"Reclaim {warp_tail['early_exit_count']}/{warp_tail['total_warps_observed']} early-exiting warps ({warp_tail['idle_tail_us']:.1f} us tail)",
            "evidence": (
                f"Warps {warp_tail['early_exit_warps'][:8]} finish at median {warp_tail['early_exit_median_us']:.1f} us "
                f"while CTA runs until {warp_tail['latest_warp_end_us']:.1f} us ({tail_pct:.1f}% tail){role_note}."
            ),
            "counterfactual_speedup_upper_bound_pct": round(min(tail_pct * 0.25, 15.0), 2),
            "recommended_optimization": (
                "Dynamically repurpose early-exiting warps (like Conv->OutProj role switch) to assist tail stages "
                "or release their register quota earlier via setmaxnreg."
            ),
        })

    # 4f. Unexplained underutilized compute dip (triggers drill-down).
    drill_stages = [s for s in stage_verdicts if s["refinement_verdict"] == "NEED_SUBSTAGE_DRILLDOWN"]
    for ds in drill_stages:
        opportunities.append({
            "category": "UNDERUTILIZED_COMPUTE_DIP",
            "title": f"Drill down underutilized stage '{ds['stage']}' ({ds['share_pct']:.1f}% share)",
            "evidence": ds["reason"],
            "counterfactual_speedup_upper_bound_pct": round(min(ds["share_pct"], 40.0), 2),
            "recommended_optimization": (
                f"Insert sub-stage load/wait/compute probes inside '{ds['stage']}' to isolate ALU/SFU, "
                "shared-memory bank conflicts, or unmeasured barrier waits."
            ),
        })

    # Rank opportunities by counterfactual impact descending.
    opportunities.sort(key=lambda o: float(o.get("counterfactual_speedup_upper_bound_pct") or 0.0), reverse=True)
    for idx, opp in enumerate(opportunities, start=1):
        opp["rank"] = idx

    # Build top-level bottleneck hint & refinement summary.
    # 5. Dual-Denominator Roofline Caliber + Effective Algorithmic MFU / Padding Waste
    workload = dict((ir.get("extra") or {}).get("workload") or {})
    if isinstance(data.get("workload"), Mapping):
        workload.update(data["workload"])
    if isinstance(data.get("caliber_input"), Mapping):
        workload.update(data["caliber_input"])

    emp_t_ratio = float(prof.get("empirical_tensor_ratio", 0.95))
    emp_h_ratio = float(prof.get("empirical_hbm_ratio", 0.85))
    dram_rw_mean = dram_r_mean + dram_w_mean
    if global_regime == "DRAM_READ_BOUND" or dram_rw_mean > tensor_mean:
        nom_rf_pct = round(min(100.0, dram_rw_mean), 2)
        emp_rf_pct = round(min(100.0, dram_rw_mean / max(0.1, emp_h_ratio)), 2)
        rf_axis = "HBM"
    else:
        nom_rf_pct = round(min(100.0, tensor_mean), 2)
        emp_rf_pct = round(min(100.0, tensor_mean / max(0.1, emp_t_ratio)), 2)
        rf_axis = "TENSOR"

    # Override nominal/empirical roofline when explicit FLOPs or HBM bytes are provided.
    eff_flops = workload.get("effective_flops")
    hw_flops = workload.get("hw_executed_flops")
    hbm_bytes = workload.get("hbm_bytes")
    dtype = str(workload.get("dtype") or "bf16").lower()
    if hbm_bytes is not None and kernel_us > 0 and prof.get("nominal_hbm_tbps") and rf_axis == "HBM":
        achieved_tbps = (float(hbm_bytes) / (kernel_us * 1e-6)) / 1e12
        nom_rf_pct = round(min(100.0, 100.0 * achieved_tbps / float(prof["nominal_hbm_tbps"])), 2)
        emp_rf_pct = round(min(100.0, 100.0 * achieved_tbps / float(prof["empirical_hbm_tbps"])), 2)

    effective_mfu_pct = workload.get("effective_mfu_pct")
    if effective_mfu_pct is None and eff_flops is not None and kernel_us > 0:
        nom_peak = prof.get("nominal_tflops_fp8") if "fp8" in dtype else prof.get("nominal_tflops_bf16")
        if nom_peak:
            achieved_tflops = (float(eff_flops) / (kernel_us * 1e-6)) / 1e12
            effective_mfu_pct = round(100.0 * achieved_tflops / float(nom_peak), 2)

    padding_waste_pct: Optional[float] = None
    if eff_flops is not None and hw_flops is not None and float(hw_flops) > 0:
        padding_waste_pct = round(100.0 * max(0.0, 1.0 - float(eff_flops) / float(hw_flops)), 2)
    elif effective_mfu_pct is not None and tensor_mean > 1.0:
        padding_waste_pct = round(100.0 * max(0.0, 1.0 - float(effective_mfu_pct) / tensor_mean), 2)

    # Guard against fake Tensor saturation caused by WGMMA M=64 / MoE padding waste!
    if padding_waste_pct is not None and padding_waste_pct >= 20.0:
        for sv in stage_verdicts:
            if sv["refinement_verdict"] == "SATURATED_STOP_REFINING":
                sv["refinement_verdict"] = "PADDING_WASTE_OPPORTUNITY"
                sv["reason"] = (
                    f"High HW Tensor Active ({tensor_mean:.1f}%) masks {padding_waste_pct:.1f}% padding waste "
                    f"(effective MFU={effective_mfu_pct}%); do NOT stop refining."
                )
        opportunities.append({
            "category": "TENSOR_PADDING_WASTE",
            "title": f"Eliminate {padding_waste_pct:.1f}% WGMMA/MoE padding waste (HW Tensor {tensor_mean:.1f}% vs effective MFU {effective_mfu_pct}%)",
            "evidence": (
                f"Hardware Tensor Active is {tensor_mean:.1f}% (nom_rf={nom_rf_pct}%, emp_rf={emp_rf_pct}%), "
                f"but effective algorithmic MFU is only {effective_mfu_pct}% ({padding_waste_pct:.1f}% wasted FLOPs)."
            ),
            "counterfactual_speedup_upper_bound_pct": round(
                min(100.0, 100.0 * padding_waste_pct / max(1.0, 100.0 - padding_waste_pct)), 2
            ),
            "recommended_optimization": (
                "Reduce WGMMA M-tile padding / dynamically group small-token experts or switch tile shape for low-n buckets."
            ),
        })

    # 6. Structural Negative-ROI Guardrails (from real H20 MegaMoE + B300 silicon audit)
    reg_per_thr = workload.get("reg_per_thread")
    smem_bytes = workload.get("smem_bytes")
    math_wait_pct = workload.get("math_warp_wait_pct")
    if math_wait_pct is None and stage_verdicts:
        math_wait_pct = round(sum(s["share_pct"] for s in stage_verdicts if s["is_wait"]), 2)
    comm_pct = workload.get("comm_pct")
    combine_pct = workload.get("combine_pct")
    if comm_pct is None and stage_verdicts:
        comm_hits = [s["share_pct"] for s in stage_verdicts if _COMM_NAME_RE.search(s["stage"])]
        if comm_hits:
            comm_pct = round(sum(comm_hits), 2)

    structural_blockers: List[Dict[str, Any]] = []
    if math_wait_pct is not None and float(math_wait_pct) < 5.5:
        structural_blockers.append({
            "code": "LOW_MATH_WAIT_NO_LATENCY_HIDING",
            "metric": {"math_warp_wait_pct": float(math_wait_pct), "threshold_pct": 5.5},
            "blocked_actions": ["fine_grained_ready_signaling", "extra_load_latency_hiding"],
            "reason": f"Math warp wait is only {float(math_wait_pct):.2f}% (<5.5%); load-latency hiding yields negligible/negative ROI.",
        })
    if (reg_per_thr is not None and int(reg_per_thr) >= 112) or (smem_bytes is not None and int(smem_bytes) >= 232448):
        structural_blockers.append({
            "code": "RESOURCE_WALL_NO_TILE_GROWTH_OR_CTA_FUSION",
            "metric": {"reg_per_thread": reg_per_thr, "smem_bytes": smem_bytes},
            "blocked_actions": ["enlarge_tile", "fuse_extra_op_into_same_cta"],
            "reason": (
                f"CTA already at resource ceiling (regs={reg_per_thr}, smem={smem_bytes} B); "
                "enlarging tile or fusing into the same CTA triggers spills/occupancy drop — prefer TBO or cross-CTA overlap."
            ),
        })
    if (comm_pct is not None and float(comm_pct) < 7.0) or (combine_pct is not None and float(combine_pct) < 2.0):
        structural_blockers.append({
            "code": "LOW_COMM_NO_DEDUP_REFACTOR",
            "metric": {"comm_pct": comm_pct, "combine_pct": combine_pct},
            "blocked_actions": ["complex_token_dedup", "communication_protocol_refactor"],
            "reason": f"Communication/combine share is tiny (comm={comm_pct}%, combine={combine_pct}%); complex comm refactor is negative ROI.",
        })
    if workload.get("uses_per_thread_atomic_instrumentation"):
        structural_blockers.append({
            "code": "FORBIDDEN_PER_THREAD_ATOMIC_INSTRUMENTATION",
            "metric": {"observed_h20_pollution_pct": 51.6},
            "blocked_actions": ["per_thread_atomicAdd_counters"],
            "reason": "Per-thread atomicAdd instrumentation pollutes H20 wall time by +51.6%; use wp_lite / WP_CTX_SMEM private slots.",
        })
    if rf_axis == "HBM" and emp_rf_pct >= 90.0:
        structural_blockers.append({
            "code": "EMPIRICAL_HBM_CEILING_REACHED",
            "metric": {"nom_rf_pct": nom_rf_pct, "emp_rf_pct": emp_rf_pct, "emp_hbm_ratio": emp_h_ratio},
            "blocked_actions": ["micro_tune_memory_coalescing_for_phantom_bw"],
            "reason": (
                f"Nominal HBM {nom_rf_pct:.1f}% already equals {emp_rf_pct:.1f}% of silicon empirical ceiling "
                "(e.g. H20 3.40 TB/s vs 4.0 TB/s nominal); only reducing bytes moved (quantization/fusion) can speed up."
            ),
        })

    # 7. Cross-Phase Complementary Overlap Diagnostics (`Compute <-> Memory Interleaving`)
    phase_profiles: List[Dict[str, Any]] = []
    comp_only_us = 0.0
    mem_comm_only_us = 0.0
    for sv in stage_verdicts:
        name = sv["stage"]
        dur_us = float(sv["median_us"])
        t_pct = sv["window_tensor_mean_pct"] if sv["window_tensor_mean_pct"] is not None else tensor_mean
        d_pct = sv["window_dram_read_mean_pct"] if sv["window_dram_read_mean_pct"] is not None else dram_r_mean
        if sv["is_wait"]:
            btype = "SYNC"
            idle_res = "SM"
        elif _COMM_NAME_RE.search(name):
            btype = "COMM"
            idle_res = "TENSOR"
            mem_comm_only_us += dur_us
        elif t_pct >= 55.0 and d_pct < 55.0:
            btype = "COMP"
            idle_res = "HBM"
            comp_only_us += dur_us
        elif d_pct >= 40.0 or (t_pct < 35.0 and not sv["is_wait"]):
            btype = "MEM"
            idle_res = "TENSOR"
            mem_comm_only_us += dur_us
        else:
            btype = "COMP" if t_pct >= d_pct else "MEM"
            idle_res = "HBM" if btype == "COMP" else "TENSOR"
        phase_profiles.append({
            "phase": name,
            "duration_us": round(dur_us, 3),
            "bound_type": btype,
            "tensor_pct": round(float(t_pct), 2),
            "dram_rw_pct": round(float(d_pct + dram_w_mean), 2),
            "sm_active_pct": round(sm_mean, 2),
            "reg_per_thr": reg_per_thr,
            "smem_bytes": smem_bytes,
            "idle_resource": idle_res,
        })

    interleaving_waste_us = round(min(comp_only_us, mem_comm_only_us), 2)
    overlap_recommendations: List[Dict[str, Any]] = []
    if interleaving_waste_us > max(20.0, 0.03 * max(1.0, kernel_us)):
        cta_resource_tight = bool(
            (reg_per_thr is not None and int(reg_per_thr) >= 112)
            or (smem_bytes is not None and int(smem_bytes) >= 225280)
        )
        strategy = "TBO_TWO_MICROBATCH_OVERLAP" if cta_resource_tight else "INTRA_CTA_WARP_OR_SPATIAL_OVERLAP"
        overlap_bound_pct = round(
            100.0 * interleaving_waste_us / max(1e-9, kernel_us - interleaving_waste_us), 2
        )
        comp_names = [p["phase"] for p in phase_profiles if p["bound_type"] == "COMP"][:4]
        mem_names = [p["phase"] for p in phase_profiles if p["bound_type"] in ("MEM", "COMM")][:4]
        overlap_recommendations.append({
            "strategy": strategy,
            "comp_phases": comp_names,
            "mem_or_comm_phases": mem_names,
            "interleaving_waste_us": interleaving_waste_us,
            "counterfactual_speedup_upper_bound_pct": min(overlap_bound_pct, 50.0),
            "reason": (
                f"Alternating COMP phases ({comp_names}, idle HBM) and MEM/COMM phases ({mem_names}, idle Tensor) "
                f"leave {interleaving_waste_us:.1f} us of complementary hardware headroom; "
                + (
                    "CTA registers/SMEM are tight -> use TBO two-microbatch or cross-CTA stream overlap."
                    if cta_resource_tight
                    else "CTA has register/SMEM headroom -> overlap via warp specialization or spatial CTA split."
                )
            ),
        })
        opportunities.append({
            "category": "COMPLEMENTARY_OVERLAP",
            "title": f"Overlap COMP {comp_names[:2]} with MEM/COMM {mem_names[:2]} via {strategy} ({interleaving_waste_us:.1f} us)",
            "evidence": overlap_recommendations[0]["reason"],
            "counterfactual_speedup_upper_bound_pct": min(overlap_bound_pct, 50.0),
            "interleaving_waste_us": interleaving_waste_us,
            "recommended_optimization": overlap_recommendations[0]["reason"],
        })

    # Re-rank all opportunities by counterfactual impact descending.
    opportunities.sort(key=lambda o: float(o.get("counterfactual_speedup_upper_bound_pct") or 0.0), reverse=True)
    for idx, opp in enumerate(opportunities, start=1):
        opp["rank"] = idx

    top_bound_pct = float(opportunities[0]["counterfactual_speedup_upper_bound_pct"]) if opportunities else 0.0
    counterfactual_upper_bound_us = round(
        kernel_us * (top_bound_pct / (100.0 + top_bound_pct)), 3
    ) if (kernel_us > 0 and top_bound_pct > 0) else 0.0

    # Unified top-level verdict aligned with H20 MegaKernel & Atrex schema.
    if any(s["refinement_verdict"] == "NEED_SUBSTAGE_DRILLDOWN" for s in stage_verdicts):
        top_verdict = "NEED_SUBSTAGE_DRILLDOWN"
    elif overlap_recommendations and (
        not opportunities or opportunities[0]["category"] == "COMPLEMENTARY_OVERLAP" or interleaving_waste_us >= 0.05 * max(1.0, kernel_us)
    ):
        top_verdict = "OVERLAP_CANDIDATE"
    elif (padding_waste_pct is None or padding_waste_pct < 20.0) and (
        global_regime == "TENSOR_SATURATED" or (rf_axis == "HBM" and emp_rf_pct >= 90.0)
    ) and top_bound_pct < 5.0:
        top_verdict = "SATURATED_STOP_REFINING"
    else:
        top_verdict = "CAUSE_EXPLAINED"

    top_hint = (
        f"{opportunities[0]['title']} (upper bound +{opportunities[0]['counterfactual_speedup_upper_bound_pct']:.1f}%)"
        if opportunities
        else f"Kernel is {global_regime} (Tensor={tensor_mean:.1f}%, DRAM_Rd={dram_r_mean:.1f}%, SM={sm_mean:.1f}%)."
    )

    return {
        "schema": CAUSAL_SCHEMA,
        "target": ir["target"],
        "arch": ir["arch"],
        "gpu": ir["gpu"],
        "total_sms": total_sms_val,
        "observed_ctas": observed_ctas,
        "observed_sms": observed_sms,
        "kernel_us": round(kernel_us, 3),
        "sw_span_us": round(sw_span_us, 3),
        "global_hw_regime": global_regime,
        "verdict": top_verdict,
        "tensor_calibration_factor": cal["tensor_calibration_factor"],
        "caliber": {
            "roofline_axis": rf_axis,
            "nom_rf_pct": nom_rf_pct,
            "emp_rf_pct": emp_rf_pct,
            "effective_mfu_pct": effective_mfu_pct,
            "padding_waste_pct": padding_waste_pct,
        },
        "hw_summary": {
            "sm_active_mean_pct": round(sm_mean, 2),
            "tensor_active_cal_mean_pct": round(tensor_mean, 2),
            "tensor_active_cal_peak_pct": round(tensor_peak, 2),
            "dram_read_mean_pct": round(dram_r_mean, 2),
            "dram_read_peak_pct": round(dram_r_peak, 2),
            "dram_write_mean_pct": round(dram_w_mean, 2),
            "warp_issue_mean_pct": round(issue_mean, 2),
        },
        "phases": phase_profiles,
        "interleaving_waste_us": interleaving_waste_us,
        "overlap_recommendations": overlap_recommendations,
        "structural_blockers": structural_blockers,
        "counterfactual_upper_bound_us": counterfactual_upper_bound_us,
        "straggler_walls": straggler_walls,
        "warp_tail_imbalance": warp_tail,
        "stage_validity": stage_verdicts,
        "causal_opportunities": opportunities,
        "bottleneck_hint": top_hint,
        "refinement_summary": {
            "saturated_stop_refining": [s["stage"] for s in stage_verdicts if s["refinement_verdict"] == "SATURATED_STOP_REFINING"],
            "cause_explained": [s["stage"] for s in stage_verdicts if s["refinement_verdict"] == "CAUSE_EXPLAINED"],
            "need_substage_drilldown": [s["stage"] for s in stage_verdicts if s["refinement_verdict"] == "NEED_SUBSTAGE_DRILLDOWN"],
            "padding_waste_opportunities": [s["stage"] for s in stage_verdicts if s["refinement_verdict"] == "PADDING_WASTE_OPPORTUNITY"],
            "suspect_empty_stubs": [s["stage"] for s in stage_verdicts if s["refinement_verdict"] == "SUSPECT_EMPTY_STUB"],
        },
    }


# ---------------------------------------------------------------------------
# D3: Autonomous-optimizer round-to-round diff feedback
# ---------------------------------------------------------------------------

ROUND_DIFF_SCHEMA = "weaveprof.round-diff/v1"
_ROUND_HW_KEYS = ("tensor_active_cal_mean_pct", "dram_read_mean_pct", "sm_active_mean_pct")
_QUOTED_RE = re.compile(r"'([^']+)'")


def _opt_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _opportunity_key(opp: Mapping[str, Any]) -> str:
    """Stable identity for an opportunity across rounds (titles embed changing numbers)."""
    cat = str(opp.get("category") or "UNKNOWN")
    m = _QUOTED_RE.search(str(opp.get("title") or ""))
    return f"{cat}::{m.group(1)}" if m else cat


def _report_stage_medians(rep: Mapping[str, Any]) -> List[Tuple[str, float]]:
    rows = rep.get("stage_validity") or rep.get("stage_breakdown") or []
    out: List[Tuple[str, float]] = []
    seen = set()
    for r in rows:
        name = r.get("stage")
        med = _opt_float(r.get("median_us"))
        if name is None or med is None or name in seen:
            continue
        seen.add(name)
        out.append((str(name), med))
    return out


def compare_optimization_rounds(
    prev_report: Mapping[str, Any],
    curr_report: Mapping[str, Any],
    *,
    noise_floor_pct: float = 1.0,
) -> Dict[str, Any]:
    """Deterministically diff two `analyze_sw_hw_causality()` reports (round N-1 -> round N).

    Gives an autonomous optimizer closed-loop feedback: did the edit help, which hardware
    counters moved, which stages got faster/slower, which opportunities were eliminated /
    persist / were newly introduced, which actions are structurally negative-ROI, and what
    the next round should target (or `STOP_SATURATED`).

    `noise_floor_pct`: latency changes within +/- this percentage are not counted as
    improvement or regression.
    """
    prev = dict(prev_report or {})
    curr = dict(curr_report or {})

    # 1. End-to-end latency.
    prev_us = _opt_float(prev.get("kernel_us"))
    curr_us = _opt_float(curr.get("kernel_us"))
    latency_delta_us: Optional[float] = None
    speedup_ratio: Optional[float] = None
    latency_delta_pct: Optional[float] = None
    if prev_us is not None and curr_us is not None:
        latency_delta_us = round(curr_us - prev_us, 3)
        if prev_us > 0:
            latency_delta_pct = round(100.0 * (curr_us - prev_us) / prev_us, 3)
        if curr_us > 0:
            speedup_ratio = round(prev_us / curr_us, 4)
    tol = max(0.0, float(noise_floor_pct))
    improved = bool(latency_delta_pct is not None and latency_delta_pct < -tol)
    regressed = bool(latency_delta_pct is not None and latency_delta_pct > tol)

    # 2. Hardware counter deltas.
    prev_hw = dict(prev.get("hw_summary") or {})
    curr_hw = dict(curr.get("hw_summary") or {})
    hw_deltas: Dict[str, Dict[str, Optional[float]]] = {}
    for k in _ROUND_HW_KEYS:
        pv = _opt_float(prev_hw.get(k))
        cv = _opt_float(curr_hw.get(k))
        hw_deltas[k] = {
            "prev": pv,
            "curr": cv,
            "delta": round(cv - pv, 2) if (pv is not None and cv is not None) else None,
        }

    # 3. Per-stage median deltas (curr order first, then stages removed this round).
    prev_st = _report_stage_medians(prev)
    curr_st = _report_stage_medians(curr)
    prev_map = dict(prev_st)
    curr_names = {n for n, _ in curr_st}
    stage_deltas: List[Dict[str, Any]] = []
    for name, cmed in curr_st:
        pmed = prev_map.get(name)
        if pmed is None:
            stage_deltas.append({
                "stage": name, "prev_median_us": None, "curr_median_us": round(cmed, 3),
                "delta_us": None, "delta_pct": None, "status": "NEW_STAGE",
            })
            continue
        d = cmed - pmed
        stage_deltas.append({
            "stage": name,
            "prev_median_us": round(pmed, 3),
            "curr_median_us": round(cmed, 3),
            "delta_us": round(d, 3),
            "delta_pct": round(100.0 * d / pmed, 2) if pmed > 0 else None,
            "status": "FASTER" if d < 0 else ("SLOWER" if d > 0 else "UNCHANGED"),
        })
    for name, pmed in prev_st:
        if name not in curr_names:
            stage_deltas.append({
                "stage": name, "prev_median_us": round(pmed, 3), "curr_median_us": None,
                "delta_us": None, "delta_pct": None, "status": "REMOVED_STAGE",
            })

    # 4. Opportunity set diff keyed by (category, quoted target).
    def _index(opps: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
        idx: Dict[str, Mapping[str, Any]] = {}
        for o in opps:
            idx.setdefault(_opportunity_key(o), o)
        return idx

    prev_opps = _index(prev.get("causal_opportunities") or [])
    curr_opps = _index(curr.get("causal_opportunities") or [])

    def _brief(key: str, o: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "key": key,
            "category": o.get("category"),
            "title": o.get("title"),
            "rank": o.get("rank"),
            "counterfactual_speedup_upper_bound_pct": _opt_float(o.get("counterfactual_speedup_upper_bound_pct")),
        }

    eliminated = [_brief(k, o) for k, o in prev_opps.items() if k not in curr_opps]
    newly_introduced = [_brief(k, o) for k, o in curr_opps.items() if k not in prev_opps]
    persisting: List[Dict[str, Any]] = []
    for k, o in curr_opps.items():
        if k not in prev_opps:
            continue
        pb = _opt_float(prev_opps[k].get("counterfactual_speedup_upper_bound_pct"))
        cb = _opt_float(o.get("counterfactual_speedup_upper_bound_pct"))
        entry = _brief(k, o)
        entry["prev_bound_pct"] = pb
        entry["bound_delta_pct"] = round(cb - pb, 2) if (pb is not None and cb is not None) else None
        persisting.append(entry)

    # 5. Negative-ROI guardrails from the current round's structural blockers.
    guardrails: List[Dict[str, Any]] = []
    blocked_actions: List[str] = []
    for b in curr.get("structural_blockers") or []:
        acts = [str(a) for a in (b.get("blocked_actions") or [])]
        guardrails.append({"code": b.get("code"), "blocked_actions": acts, "reason": b.get("reason")})
        for a in acts:
            if a not in blocked_actions:
                blocked_actions.append(a)

    # 6. Next-round directive.
    curr_verdict = str(curr.get("verdict") or "")
    ranked_curr = sorted(
        curr_opps.items(),
        key=lambda kv: (
            int(kv[1].get("rank")) if isinstance(kv[1].get("rank"), int) else 10 ** 6,
            -(_opt_float(kv[1].get("counterfactual_speedup_upper_bound_pct")) or 0.0),
            kv[0],
        ),
    )
    revert_note = (
        f"Round regressed {latency_delta_pct:+.2f}% ({prev_us:.3f} -> {curr_us:.3f} us); revert last edit first. "
        if regressed and latency_delta_pct is not None and prev_us is not None and curr_us is not None
        else ""
    )
    guard_note = f" Avoid negative-ROI actions: {', '.join(blocked_actions)}." if blocked_actions else ""
    if curr_verdict == "SATURATED_STOP_REFINING":
        directive: Dict[str, Any] = {
            "action": "STOP_SATURATED",
            "target_key": None,
            "text": (
                f"{revert_note}Kernel is saturated ({curr.get('global_hw_regime')}); "
                "stop refining - remaining headroom is below the actionable threshold."
            ),
        }
    elif ranked_curr:
        key, top = ranked_curr[0]
        bound = _opt_float(top.get("counterfactual_speedup_upper_bound_pct")) or 0.0
        status = "PERSISTING" if key in prev_opps else "NEW"
        directive = {
            "action": "PURSUE_TOP_OPPORTUNITY",
            "target_key": key,
            "category": top.get("category"),
            "title": top.get("title"),
            "opportunity_status": status,
            "counterfactual_speedup_upper_bound_pct": bound,
            "recommended_optimization": top.get("recommended_optimization"),
            "text": (
                f"{revert_note}Next: {top.get('category')} [{status}] - {top.get('title')} "
                f"(upper bound +{bound:.1f}%). {top.get('recommended_optimization') or ''}".rstrip()
                + guard_note
            ),
        }
    else:
        directive = {
            "action": "NO_OPPORTUNITY_REMAINING",
            "target_key": None,
            "text": f"{revert_note}No ranked causal opportunity remains; capture finer sub-stage probes or stop.{guard_note}",
        }
    directive["revert_previous_change"] = regressed
    directive["blocked_actions"] = blocked_actions

    return {
        "schema": ROUND_DIFF_SCHEMA,
        "prev_kernel_us": prev_us,
        "curr_kernel_us": curr_us,
        "latency_delta_us": latency_delta_us,
        "latency_delta_pct": latency_delta_pct,
        "speedup_ratio": speedup_ratio,
        "improved": improved,
        "regressed": regressed,
        "noise_floor_pct": tol,
        "prev_verdict": prev.get("verdict"),
        "curr_verdict": curr.get("verdict"),
        "hw_deltas": hw_deltas,
        "stage_deltas": stage_deltas,
        "eliminated_opportunities": eliminated,
        "persisting_opportunities": persisting,
        "newly_introduced_opportunities": newly_introduced,
        "negative_roi_guardrails": guardrails,
        "next_round_directive": directive,
    }
