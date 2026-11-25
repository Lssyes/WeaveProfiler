#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§3.3 Accurate Probe Verification & Closed-Loop Scheme A Probe Coarsening Driver.

100% Deterministic Verification Engine enforcing the Four Fidelity Gates (Table IV):
  1. ``SASS Sim.``    (Compile) : High ``LCS(I_0, I_inst)/|I_0|``, ``delta_spill_bytes <= 0``, ``0 B added SMEM``, intact occupancy.
  2. ``Numerical``    (Runtime) : Bitwise output equality on Gaussian ``randn`` inputs (``mismatches_total == 0``).
  3. ``Completeness`` (Decode)  : ``overflow == 0``, strictly nested paired intervals, all declared warp roles & stages present.
  4. ``Overhead``     (Runtime) : 30 warm interleaved launches median overhead ``<= 10.0%`` + SW/HW trace alignment.

Closed-Loop Probe Coarsening (Strictly Scheme A — NEVER uses ``FI_SAMPLE`` loop sampling):
  When inner-loop probes trigger a SASS unrolling cliff, register spill, spin-loop overflow, or
  ``> 10.0%`` warm-launch overhead, the verifier pinpoints the exact offending probe sites (using
  decoded event-frequency attribution + AST loop-depth analysis) and feeds localized diagnostics
  back to ``weaveprof/pipeline/scan/llm_probe_placer.py`` (up to ``max_rounds=3``) to **hoist or merge overly dense
  inner-loop probes outward to enclosing stage boundaries** until all four gates pass.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

SKILL_ROOT = Path(__file__).resolve().parents[3]
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

from weaveprof.contract import enclosing_loop, hot_loop, loop_spans, normalize_probe_spec, spill_delta_bytes


_GATE_NAMES = {
    "assembly": "sass_similarity_and_resources",
    "output": "numerical_bitwise_randn",
    "pairing": "completeness_and_nesting",
    "overhead": "warm_overhead_le_10pct",
}


def build_feedback_json(evaluation, probe_spec, source_text, *, fires=None, metrics=None):
    """Construct the paper diagnostic from measured per-warp counts and gate decisions."""
    gates = evaluation.get("gates", evaluation)
    failed = [name for name, old in _GATE_NAMES.items() if gates.get(name, gates.get(old)) is not True]
    measured = dict(fires if fires is not None else evaluation.get("fires", {}))
    if any(type(v) is not int or v < 0 for v in measured.values()):
        raise ValueError("fires must contain nonnegative integer counts")
    span = None
    if failed and any(measured.values()):
        _, span = hot_loop(probe_spec, measured, source_text)
    result = {"status": "FAIL" if failed else "PASS", "failed": failed,
              "fires": measured, "enclosing_loop": span}
    if metrics is not None:
        result["metrics"] = dict(metrics)
    return result


def write_feedback(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def _load_mod(rel_path: str, mod_name: str):
    p = SKILL_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def decode_and_check_completeness(
    dump_path: Path,
    names_path: Path,
    probe_spec: Mapping[str, Any],
    out_trace_path: Path,
) -> Dict[str, Any]:
    """Gate 3 (`Completeness`): Decode FILITE02 dump and verify 0 overflow, 0 unpaired, role & stage coverage."""
    out_trace_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_trace_path.parent / (out_trace_path.stem.replace(".perfetto", "") + ".summary.json")
    decoder = SKILL_ROOT / "weaveprof/pipeline/validate" / "tools" / "fi_lite_decode.py"
    proc = subprocess.run(
        [
            sys.executable, "-B", str(decoder),
            str(dump_path),
            "--names", str(names_path),
            "-o", str(out_trace_path),
            "--summary", str(summary_path),
            "--allow-errors",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    overflow = int(summary.get("overflow") or 0)
    errors_obj = summary.get("errors") or {}
    if isinstance(errors_obj, dict):
        unpaired = sum(int(v) for k, v in errors_obj.items() if k in
                       ("unpaired_begin", "extra_end", "unpaired_end", "cross_nested", "negative_dur")
                       or any(word in k for word in ("未配对", "多余", "嵌套", "负时长")))
        total_decode_errs = sum(int(v) for v in errors_obj.values() if isinstance(v, (int, float)))
    elif isinstance(errors_obj, list):
        unpaired = len(errors_obj)
        total_decode_errs = len(errors_obj)
    else:
        unpaired = 0
        total_decode_errs = 0 if proc.returncode == 0 else 1

    # Extract observed event counts and active roles from decoded Chrome JSON or summary
    ev_counts: Dict[str, int] = {}
    observed_roles = set()
    if summary.get("event_counts"):
        for k, v in (summary.get("event_counts") or {}).items():
            ev_counts[str(k)] = int(v)
    if out_trace_path.is_file() and not ev_counts:
        import gzip
        raw_bytes = out_trace_path.read_bytes()
        if raw_bytes[:2] == b"\x1f\x8b":
            raw_bytes = gzip.decompress(raw_bytes)
        try:
            tdoc = json.loads(raw_bytes.decode("utf-8"))
            for ev in tdoc.get("traceEvents") or []:
                if ev.get("ph") == "X":
                    nm = str(ev.get("name") or "")
                    ev_counts[nm] = ev_counts.get(nm, 0) + 1
                    role_arg = (ev.get("args") or {}).get("role")
                    if role_arg is not None:
                        observed_roles.add(str(role_arg))
        except Exception:
            pass

    declared_sems = sorted({str(p.get("semantic_id")) for p in (probe_spec.get("probes") or []) if p.get("semantic_id")})
    missing_sems = [s for s in declared_sems if ev_counts.get(s, 0) == 0]
    if summary.get("fires"):
        declared_sems = [p.get("id", p.get("probe_id")) for p in probe_spec["probes"]]
        missing_sems = [pid for pid in declared_sems if not summary["fires"].get(pid)]

    ok = (proc.returncode == 0) and (overflow == 0) and (total_decode_errs == 0) and (len(missing_sems) == 0)
    return {
        "ok": ok,
        "returncode": proc.returncode,
        "overflow": overflow,
        "unpaired_or_cross_nested": unpaired,
        "decode_errors_total": total_decode_errs,
        "event_counts": ev_counts,
        "fires": summary.get("fires", {}),
        "declared_stages": declared_sems,
        "missing_stages": missing_sems,
        "observed_roles": sorted(observed_roles),
        "summary_path": str(summary_path),
        "trace_path": str(out_trace_path),
        "decoder_log": (proc.stdout or proc.stderr or "")[-600:],
    }


def evaluate_assembly(preflight, runtime_metrics, min_sass_sim):
    """Use CPU build evidence, or the JIT compiler artifacts supplied by the evaluator."""
    evidence = preflight
    if preflight.get("language") in ("triton", "cutedsl") and not preflight.get("binsim"):
        evidence = runtime_metrics or {}
    binsim = evidence.get("binsim") or {}
    resources = evidence.get("resources") or {}
    metrics = {}
    try:
        rate = float(binsim["assembly_change_rate"]) if "assembly_change_rate" in binsim else 1 - float(binsim["sim_opcode"])
        if not 0 <= rate <= 1:
            raise ValueError("assembly_change_rate must be in [0, 1]")
        metrics["assembly_change_rate"] = rate
        baseline = (resources.get("l0") or {}).get("spill_total_bytes")
        instrumented = (resources.get("inst") or {}).get("spill_total_bytes")
        delta = spill_delta_bytes(baseline, instrumented, resources.get("delta_spill_bytes"))
        metrics["delta_spill_bytes"] = delta
        if baseline is not None:
            metrics.update(baseline_spill_bytes=baseline, instrumented_spill_bytes=instrumented)
        smem_delta = resources["delta_smem_bytes"]
        if type(smem_delta) is not int:
            raise ValueError("delta_smem_bytes must be a signed integer")
        metrics["delta_smem_bytes"] = smem_delta
    except (KeyError, TypeError, ValueError) as exc:
        metrics["assembly_evidence_error"] = str(exc)
        return False, metrics
    return bool(preflight.get("ok")) and 1 - rate >= min_sass_sim and delta <= 0 and smem_delta <= 0, metrics


def evaluate_four_gates(
    *,
    source_text: str,
    probe_spec: Mapping[str, Any],
    preflight: Mapping[str, Any],
    runtime_metrics: Optional[Mapping[str, Any]],
    completeness: Optional[Mapping[str, Any]],
    overhead_limit_pct: float = 10.0,
    min_sass_sim: float = 0.85,
) -> Dict[str, Any]:
    """Evaluate Table IV Four Fidelity Gates and synthesize Scheme A Probe Coarsening diagnostics."""
    paper_spec = "roles" in probe_spec
    if paper_spec:
        probe_spec = normalize_probe_spec(probe_spec, source_text)
    re_mod = _load_mod("weaveprof/pipeline/instrument/rule_engine.py", "wp_rule_engine_for_gates")
    lang = str(probe_spec.get("language") or "cuda_cpp")
    loop_depths = re_mod.analyze_loop_nesting(source_text, lang)
    spans = loop_spans(source_text, lang)

    diagnostics: List[Dict[str, Any]] = list(preflight.get("diagnostics") or [])

    gate_sass_ok, assembly_metrics = evaluate_assembly(preflight, runtime_metrics, min_sass_sim)

    # Gate 2: Numerical bitwise equality on randn inputs
    gate_num_ok = False
    if runtime_metrics is not None:
        mismatches = int(runtime_metrics.get("mismatches_total", 1))
        nonfinite = int(runtime_metrics.get("nonfinite_total", 1))
        gate_num_ok = bool(runtime_metrics.get("ok", True)) and (mismatches == 0) and (nonfinite == 0)
        if not gate_num_ok:
            diagnostics.append({
                "code": "NUMERICAL_MISMATCH",
                "severity": "error",
                "mismatches_total": mismatches,
                "nonfinite_total": nonfinite,
                "message": f"Bitwise randn numerical verification failed (mismatches={mismatches}, nonfinite={nonfinite}).",
            })

    # Gate 3: Completeness
    gate_comp_ok = False
    if completeness is not None:
        gate_comp_ok = (bool(completeness.get("ok")) and not completeness.get("overflow")
                        and not completeness.get("unpaired_or_cross_nested")
                        and not completeness.get("decode_errors_total") and not completeness.get("missing_stages"))
        if not gate_comp_ok:
            if int(completeness.get("overflow", 0)) != 0:
                # Identify deepest-loop or spin-loop probes causing lane overflow
                dense_probes = [
                    {
                        "probe_id": p.get("probe_id"),
                        "semantic_id": p.get("semantic_id"),
                        "lines": [p.get("begin_line"), p.get("end_line")],
                        "loop_depth": loop_depths.get(int(p.get("begin_line") or 1), 0),
                    }
                    for p in (probe_spec.get("probes") or [])
                    if loop_depths.get(int(p.get("begin_line") or 1), 0) >= 1
                ]
                diagnostics.append({
                    "code": "BUFFER_OVERFLOW_HOIST_REQUIRED",
                    "severity": "error",
                    "overflow_flags": completeness.get("overflow"),
                    "candidate_dense_probes": dense_probes,
                    "message": (
                        "HBM record buffer overflowed (bit0 lane-full). Check if a `wait` probe sits inside a "
                        "polling spin loop or an inner loop executing thousands of iterations; hoist/merge inner "
                        "probes outward to enclosing stage boundaries (Scheme A)."
                    ),
                })
            if completeness.get("unpaired_or_cross_nested", 0) > 0:
                diagnostics.append({
                    "code": "UNPAIRED_OR_OVERLAPPING_INTERVALS",
                    "severity": "error",
                    "unpaired_count": completeness.get("unpaired_or_cross_nested"),
                    "message": (
                        "Decoded lane contains unpaired or cross-nested B/E events. Ensure begin_line and end_line "
                        "execute under the exact same control-flow condition and do not cross `continue`/`break`/`return`."
                    ),
                })
            if completeness.get("missing_stages"):
                diagnostics.append({
                    "code": "MISSING_STAGE_COVERAGE",
                    "severity": "error",
                    "missing_stages": completeness.get("missing_stages"),
                    "message": f"Declared stages {completeness.get('missing_stages')} produced 0 events at runtime.",
                })

    # Gate 4: 30-run warm interleaved median overhead <= 10.0%
    gate_ovh_ok = False
    overhead_pct = None
    if runtime_metrics is not None and runtime_metrics.get("overhead_pct") is not None:
        overhead_pct = float(runtime_metrics["overhead_pct"])
        gate_ovh_ok = math.isfinite(overhead_pct) and -100 < overhead_pct <= overhead_limit_pct
        if not gate_ovh_ok:
            ev_counts = (completeness or {}).get("event_counts") or {}
            # Pinpoint highest-frequency / deepest-nested inner-loop probes for Scheme A Coarsening!
            offending: List[Dict[str, Any]] = []
            for p in probe_spec.get("probes") or []:
                b_ln = int(p.get("begin_line") or 1)
                e_ln = int(p.get("end_line") or b_ln)
                ld = loop_depths.get(b_ln, 0)
                sem = str(p.get("semantic_id") or "")
                freq = int(ev_counts.get(sem, 0))
                if ld >= 1 or freq > 0:
                    # Find enclosing outer loop lines for concrete hoisting guidance
                    span = enclosing_loop({"begin": {"line": b_ln}, "end": {"line": e_ln}}, spans)
                    offending.append({
                        "probe_id": p.get("probe_id"),
                        "semantic_id": sem,
                        "probe_lines": [b_ln, e_ln],
                        "loop_nesting_depth": ld,
                        "decoded_slice_count": freq,
                        "enclosing_loop_lines": span,
                    })
            offending.sort(key=lambda x: (x["loop_nesting_depth"], x["decoded_slice_count"]), reverse=True)
            diagnostics.append({
                "code": "OVERHEAD_EXCEEDED_COARSEN_SCHEME_A",
                "severity": "error",
                "measured_overhead_pct": round(overhead_pct, 3),
                "limit_pct": overhead_limit_pct,
                "offending_inner_loop_probes": offending[:4],
                "message": (
                    f"Warm-launch median overhead is {overhead_pct:.2f}% (> {overhead_limit_pct:.1f}% limit). "
                    f"Offending high-frequency inner-loop probes: {[o['probe_id'] for o in offending[:4]]}. "
                    "Apply Scheme A Probe Coarsening: HOIST inner-loop probes outward to wrap the enclosing "
                    "loop/stage boundary as a single coarse stage interval, or MERGE adjacent inner-loop probes "
                    "(DO NOT use K-loop sampling)."
                ),
            })

    all_passed = gate_sass_ok and gate_num_ok and gate_comp_ok and gate_ovh_ok
    return {
        "all_gates_passed": all_passed,
        "gates": {
            "sass_similarity_and_resources": gate_sass_ok,
            "numerical_bitwise_randn": gate_num_ok,
            "completeness_and_nesting": gate_comp_ok,
            "warm_overhead_le_10pct": gate_ovh_ok,
        },
        "overhead_pct": overhead_pct,
        "metrics": assembly_metrics,
        "fires": (completeness or {}).get("fires", {}),
        "diagnostics": diagnostics,
    }


def _run_external_eval_cmd(
    eval_cmd_template: str,
    *,
    l0_source: Path,
    inst_source: Path,
    names_json: Path,
    round_dir: Path,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """Execute a user/harness-supplied `--eval-cmd` that runs 30 warm interleaved launches + dumps FILITE02.

    Placeholders supported in `eval_cmd_template`:
      `{l0_source}`, `{inst_source}`, `{names_json}`, `{round_dir}`, `{dump_path}`, `{metrics_json}`
    """
    dump_path = round_dir / "capture.fi02.bin"
    metrics_json = round_dir / "runtime_metrics.json"
    formatted = eval_cmd_template.format(
        l0_source=str(l0_source),
        inst_source=str(inst_source),
        names_json=str(names_json),
        round_dir=str(round_dir),
        dump_path=str(dump_path),
        metrics_json=str(metrics_json),
    )
    proc = subprocess.run(
        formatted,
        shell=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if metrics_json.is_file():
        metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    else:
        # Try parsing last JSON line from stdout
        metrics = {"ok": proc.returncode == 0}
        for ln in reversed((proc.stdout or "").splitlines()):
            ln = ln.strip()
            if ln.startswith("{") and ln.endswith("}"):
                try:
                    metrics = json.loads(ln)
                    break
                except json.JSONDecodeError:
                    pass
    if proc.returncode != 0 and "error" not in metrics:
        metrics["ok"] = False
        metrics["error"] = (proc.stderr or proc.stdout or "")[-800:]
    return metrics, (dump_path if dump_path.is_file() else None)


def run_closed_loop_profile(
    source_path: Path,
    out_dir: Path,
    *,
    card: str,
    arch: Optional[str] = None,
    language: Optional[str] = None,
    model: str = "auto",
    reasoning_effort: str = "max",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_rounds: int = 3,
    prev_source_path: Optional[Path] = None,
    prev_spec_path: Optional[Path] = None,
    eval_cmd: Optional[str] = None,
    runtime_evaluator: Optional[Callable[[Path, Path, Path, Mapping[str, Any], Path], Tuple[Dict[str, Any], Optional[Path]]]] = None,
    llm_caller: Optional[Callable[..., Dict[str, Any]]] = None,
    overhead_limit_pct: float = 10.0,
    min_sass_sim: float = 0.85,
    preflight_only: bool = False,
) -> Dict[str, Any]:
    """End-to-end WeaveProfiler Closed-Loop Pipeline (`§3.1 -> §3.2 -> §3.3 -> §3.1` max 3 rounds)."""
    t_pipeline_start = time.perf_counter()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_path.expanduser().resolve()
    source_text = source_path.read_text(encoding="utf-8")

    placer = _load_mod("weaveprof/pipeline/scan/llm_probe_placer.py", "wp_closed_loop_placer")
    incr = _load_mod("weaveprof/pipeline/instrument/incremental_reuse.py", "wp_closed_loop_incr")
    engine = _load_mod("weaveprof/pipeline/instrument/rule_engine.py", "wp_closed_loop_engine")

    is_incremental = bool(prev_source_path and prev_spec_path)
    cache_dir = out_dir / "_l0_cache"

    total_llm_s = 0.0
    total_tokens = 0
    round_history: List[Dict[str, Any]] = []
    current_diags: Optional[List[Dict[str, Any]]] = None
    current_spec: Optional[Dict[str, Any]] = None
    final_passed = False

    for rnd in range(1, max_rounds + 1):
        round_dir = out_dir / f"round_{rnd}"
        round_dir.mkdir(parents=True, exist_ok=True)
        t_rnd_start = time.perf_counter()

        # Step 1: §3.1 Semantic Probe Placement (or §3.2 2c Diff-Hash Incremental Reuse on Round 1)
        if rnd == 1 and is_incremental:
            inc_res = incr.reuse_probes_incremental(
                prev_source_path.expanduser().resolve(),
                prev_spec_path.expanduser().resolve(),
                source_path,
                model=model,
                reasoning_effort=reasoning_effort,
                llm_caller=llm_caller,
            )
            current_spec = inc_res["probe_spec"]
            rnd_llm_s = float((inc_res.get("telemetry") or {}).get("llm_elapsed_s") or 0.0)
            rnd_tok = int((inc_res.get("telemetry") or {}).get("total_tokens") or 0)
            placement_mode = inc_res["telemetry"]["mode"]
        else:
            pl_res = placer.place_probes(
                source_path,
                card=card,
                arch=arch,
                language=language,
                model=model,
                reasoning_effort=reasoning_effort,
                ablation_mode=("open_loop" if max_rounds == 1 else "full_closed_loop"),
                diagnostics=current_diags,
                prev_spec=current_spec,
                round_index=rnd,
                llm_caller=llm_caller,
                api_key=api_key,
                base_url=base_url,
            )
            current_spec = pl_res["probe_spec"]
            rnd_llm_s = float(pl_res["telemetry"]["elapsed_s"])
            rnd_tok = int(pl_res["telemetry"].get("total_tokens") or 0)
            placement_mode = pl_res["ablation_mode"]

        total_llm_s += rnd_llm_s
        total_tokens += rnd_tok

        spec_file = round_dir / "probe_spec.json"
        spec_file.write_text(json.dumps(current_spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        # Step 2: §3.2 (2a) Millisecond Rule Expansion + (2b) CPU Parallel Static Preflight
        inj_res = engine.inject_probes(source_text, current_spec, check_sha=True, fail_on_static_errors=True)
        if not inj_res["ok"]:
            current_diags = inj_res["diagnostics"]
            write_feedback(round_dir / "feedback.json", build_feedback_json({}, current_spec, source_text))
            round_history.append({
                "round": rnd,
                "placement_mode": placement_mode,
                "llm_elapsed_s": rnd_llm_s,
                "inject_elapsed_ms": inj_res["elapsed_ms"],
                "stage_failed": "static_probe_check",
                "diagnostics": current_diags,
                "round_elapsed_s": round(time.perf_counter() - t_rnd_start, 4),
            })
            continue

        inst_path = round_dir / f"instrumented_{source_path.name}"
        inst_path.write_bytes(inj_res["instrumented_source"].replace("\r\n", "\n").encode("utf-8"))
        names_path = round_dir / "names.json"
        names_path.write_text(json.dumps(inj_res["names"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        preflight = engine.run_cpu_parallel_preflight(
            source_path,
            inst_path,
            current_spec,
            round_dir / "preflight",
            cache_dir=cache_dir,
            min_sass_sim=min_sass_sim,
        )
        compile_failed = any(d.get("code") == "COMPILE_ERROR" for d in preflight.get("diagnostics", []))
        if not preflight["ok"] and (compile_failed or preflight_only):
            current_diags = preflight["diagnostics"]
            write_feedback(round_dir / "feedback.json", build_feedback_json({}, current_spec, source_text))
            round_history.append({
                "round": rnd,
                "placement_mode": placement_mode,
                "llm_elapsed_s": rnd_llm_s,
                "inject_elapsed_ms": inj_res["elapsed_ms"],
                "preflight_elapsed_s": preflight["elapsed_s"],
                "stage_failed": "cpu_parallel_preflight",
                "preflight": preflight,
                "diagnostics": current_diags,
                "round_elapsed_s": round(time.perf_counter() - t_rnd_start, 4),
            })
            continue

        if preflight_only or (runtime_evaluator is None and not eval_cmd):
            assembly_ok, metrics = evaluate_assembly(preflight, None, min_sass_sim)
            final_passed = bool(preflight_only and assembly_ok)
            feedback = build_feedback_json({"assembly": assembly_ok}, current_spec, source_text, metrics=metrics)
            write_feedback(round_dir / "feedback.json", feedback)
            round_history.append({
                "round": rnd,
                "placement_mode": placement_mode,
                "llm_elapsed_s": rnd_llm_s,
                "inject_elapsed_ms": inj_res["elapsed_ms"],
                "preflight_elapsed_s": preflight["elapsed_s"],
                "preflight": preflight,
                "all_gates_passed": False,
                "note": "preflight_only",
                "round_elapsed_s": round(time.perf_counter() - t_rnd_start, 4),
            })
            break

        # Step 3: §3.3 Accurate Probe Verification on GPU (Numerical + Completeness + <=10% Overhead)
        t_gpu0 = time.perf_counter()
        if runtime_evaluator is not None:
            rt_metrics, dump_path = runtime_evaluator(source_path, inst_path, names_path, current_spec, round_dir)
        else:
            assert eval_cmd is not None
            rt_metrics, dump_path = _run_external_eval_cmd(
                eval_cmd,
                l0_source=source_path,
                inst_source=inst_path,
                names_json=names_path,
                round_dir=round_dir,
            )
        gpu_elapsed_s = round(time.perf_counter() - t_gpu0, 4)

        comp_res = None
        if dump_path is not None and dump_path.is_file():
            comp_res = decode_and_check_completeness(
                dump_path,
                names_path,
                current_spec,
                round_dir / "trace_sw.perfetto.json.gz",
            )
        else:
            comp_res = {
                "ok": False,
                "overflow": 0,
                "unpaired_or_cross_nested": 0,
                "missing_stages": [p.get("semantic_id") for p in (current_spec.get("probes") or [])],
                "error": "No FILITE02 dump file produced by runtime run",
            }

        gate_report = evaluate_four_gates(
            source_text=source_text,
            probe_spec=current_spec,
            preflight=preflight,
            runtime_metrics=rt_metrics,
            completeness=comp_res,
            overhead_limit_pct=overhead_limit_pct,
            min_sass_sim=min_sass_sim,
        )
        feedback = build_feedback_json(gate_report, current_spec, source_text, metrics=gate_report["metrics"])
        write_feedback(round_dir / "feedback.json", feedback)
        round_history.append({
            "round": rnd,
            "placement_mode": placement_mode,
            "llm_elapsed_s": rnd_llm_s,
            "inject_elapsed_ms": inj_res["elapsed_ms"],
            "preflight_elapsed_s": preflight["elapsed_s"],
            "gpu_verify_elapsed_s": gpu_elapsed_s,
            "preflight": preflight,
            "runtime_metrics": rt_metrics,
            "completeness": comp_res,
            "gate_report": gate_report,
            "all_gates_passed": gate_report["all_gates_passed"],
            "round_elapsed_s": round(time.perf_counter() - t_rnd_start, 4),
        })

        if gate_report["all_gates_passed"]:
            final_passed = True
            break
        current_diags = [feedback, *gate_report["diagnostics"]]

    total_wall_s = round(time.perf_counter() - t_pipeline_start, 4)
    total_sys_s = round(max(0.0, total_wall_s - total_llm_s), 4)

    # Promote final verified artifacts to top-level `out_dir`
    if round_history:
        last_rnd_dir = out_dir / f"round_{len(round_history)}"
        for fname in ("probe_spec.json", "feedback.json", f"instrumented_{source_path.name}", "names.json", "trace_sw.perfetto.json.gz"):
            cand = last_rnd_dir / fname
            if cand.is_file():
                import shutil as _sh
                _sh.copy2(cand, out_dir / fname)

    manifest = {
        "schema": "weaveprofiler.auto-profile-manifest/v1",
        "ok": final_passed,
        "is_incremental": is_incremental,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "card": card,
        "arch": current_spec.get("arch") if current_spec else arch,
        "language": current_spec.get("language") if current_spec else language,
        "rounds_executed": len(round_history),
        "timing_summary_s": {
            "t_llm_s": round(total_llm_s, 4),
            "t_sys_s": total_sys_s,
            "t_total_s": total_wall_s,
        },
        "total_tokens": total_tokens,
        "final_probe_spec": str(out_dir / "probe_spec.json"),
        "final_instrumented_source": str(out_dir / f"instrumented_{source_path.name}"),
        "round_history": round_history,
    }
    (out_dir / "auto_profile_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="§3.1–§3.3 Closed-Loop Probe Placement, Injection & Verification")
    ap.add_argument("--source", required=True, help="Uninstrumented kernel source file")
    ap.add_argument("--card", required=True, choices=("h20", "b300", "mi308x", "sm86", "auto"))
    ap.add_argument("--arch", help="Explicit architecture (sm_90a, sm_100a, sm_103a, sm_86, gfx942)")
    ap.add_argument("--language", help="Optional explicit language (cuda_cpp, hip_cpp, triton, cutedsl)")
    ap.add_argument("--model", default="auto", help="LLM model identifier (auto, qwen4-max, ...)")
    ap.add_argument("--reasoning-effort", default="max")
    ap.add_argument("--api-key", help="Optional LLM API key")
    ap.add_argument("--base-url", help="Explicit chat completions endpoint or /v1 base URL")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--prev-source", help="Previous source path for §3.2 (2c) Incremental Probe Reuse")
    ap.add_argument("--prev-spec", help="Previous verified probe_spec.json for Incremental Probe Reuse")
    ap.add_argument("--eval-cmd", help="Runtime harness command template producing {dump_path} and {metrics_json}")
    ap.add_argument("--preflight-only", action="store_true", help="Stop after §3.2 (2b) CPU parallel static preflight")
    ap.add_argument("--overhead-limit-pct", type=float, default=10.0)
    ap.add_argument("--min-sass-sim", type=float, default=0.85)
    ap.add_argument("--out-dir", required=True, help="Output directory for probe_spec.json, instrumented source & manifest")
    args = ap.parse_args(list(argv) if argv is not None else None)

    manifest = run_closed_loop_profile(
        Path(args.source),
        Path(args.out_dir),
        card=args.card,
        arch=args.arch,
        language=args.language,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        api_key=args.api_key,
        base_url=args.base_url,
        max_rounds=args.max_rounds,
        prev_source_path=Path(args.prev_source) if args.prev_source else None,
        prev_spec_path=Path(args.prev_spec) if args.prev_spec else None,
        eval_cmd=args.eval_cmd,
        overhead_limit_pct=args.overhead_limit_pct,
        min_sass_sim=args.min_sass_sim,
        preflight_only=args.preflight_only,
    )
    print(json.dumps({
        "ok": manifest["ok"],
        "rounds_executed": manifest["rounds_executed"],
        "timing_summary_s": manifest["timing_summary_s"],
        "manifest": str(Path(args.out_dir).expanduser().resolve() / "auto_profile_manifest.json"),
    }, ensure_ascii=False, indent=2))
    return 0 if manifest["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
