"""Deterministic Trace Validator (§3.2): 4-gate + product-level trace & hardware curve validation."""
from __future__ import annotations

import filecmp
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess

from .contract import spill_delta_bytes
from .harness import (
    EpisodeSealer,
    HarnessPolicy,
    PolicyGuard,
    gpu_critical_region,
    summarize_failed_timeline,
    verify_compiler_spill_gate,
    verify_semantic_body_and_stubs,
    verify_sw_hw_coverage,
    verify_trace_processor_sql,
)
from .workspace import implementation, read_json, write_json


class VerificationError(RuntimeError):
    def __init__(self, feedback, directory):
        self.feedback, self.directory = feedback, Path(directory)
        super().__init__(f"Verification failed in {directory}: {feedback}")


_CORE_OP_RE = re.compile(
    r"^(?:HMMA|GMMA|WGMMA|QMMA|OMMA|UTCMA|UTCMMA|LDGSTS|UTMALDG|UTMASTG|FFMA|HFMA2|DFMA|"
    r"v_mfma_|v_wmma_|v_fmac_|v_pk_fma_|global_load_lds)",
    re.IGNORECASE,
)


def _sequence(path, kernel, amd):
    text = path.read_text(encoding="utf-8")
    if amd:
        hip = implementation("4_validate.tools.fi_binsim_hip")
        instructions, found = hip.extract_kernel(hip.parse_amdgcn(text.splitlines()), re.escape(kernel))
        if not found:
            raise ValueError(f"{path}: function {kernel!r} not found")
        return [ins["op"] for ins in instructions if ins["op"] != "LABEL"]
    binsim = implementation("4_validate.tools.fi_binsim")
    candidates = [f for f in binsim.parse_sass(text, str(path)) if kernel in f.name]
    if len(candidates) != 1:
        raise ValueError(f"{path}: expected one compiled function matching {kernel!r}, found {len(candidates)}")
    return [ins.opcode_token for ins in candidates[0].instructions]


def assess(directory, spec, limit, runtime=None):
    runtime = read_json(directory / "runtime.json") if runtime is None else runtime
    metrics, errors = {}, {}
    checks = dict.fromkeys(("assembly", "output", "pairing", "overhead"), False)
    fires, trace, summary = {}, None, {}
    try:
        amd = spec.get("platform") == "amd" or spec.get("card") == "mi308x"
        sass_paths = [directory / runtime[f"{variant}_sass"] for variant in ("baseline", "instrumented")]
        if sass_paths[0].samefile(sass_paths[1]):
            raise ValueError("Expected two independently compiled disassembly files (baseline_sass == instrumented_sass)")
        sequences = [_sequence(path, runtime.get(f"{variant}_kernel") or runtime["kernel_name"], amd)
                     for variant, path in zip(("baseline", "instrumented"), sass_paths)]
        if not all(sequences):
            raise ValueError("Empty disassembly")
        from collections import Counter
        binsim = implementation("4_validate.tools.fi_binsim")
        ratio = binsim.bitset_lcs(*sequences) / len(sequences[0])
        c0, c1 = Counter(sequences[0]), Counter(sequences[1])
        mbag = sum(min(c0[k], c1[k]) for k in c0)
        order_change = max(0.0, 1.0 - binsim.bitset_lcs(*sequences) / max(1, mbag))
        core_seqs = [[op.split(".")[0] for op in seq if _CORE_OP_RE.match(op)] for seq in sequences]
        core_ratio = (binsim.bitset_lcs(*core_seqs) / len(core_seqs[0])) if core_seqs[0] else ratio
        metrics.update(lcs_ratio=ratio, core_retention_ratio=core_ratio,
                       assembly_change_rate=1 - ratio, order_change_ratio=order_change)
        baseline = runtime.get("baseline_spill_bytes")
        instrumented = runtime.get("instrumented_spill_bytes", runtime.get("spill_bytes"))
        delta = spill_delta_bytes(baseline, instrumented, runtime.get("delta_spill_bytes"))
        smem = int(runtime["extra_smem_bytes"])
        metrics.update(delta_spill_bytes=delta, extra_smem_bytes=smem)
        if baseline is not None:
            metrics.update(baseline_spill_bytes=baseline, instrumented_spill_bytes=instrumented)
        # Count SASS local spill instructions (STL/LDL on NVIDIA, scratch_* on AMD) directly
        # to catch spill explosions hidden behind unchanged `Used N regs`.
        def _count_spill_ops(seq):
            if amd:
                st = sum(1 for op in seq if op.lower().startswith(("scratch_store", "buffer_store_dword")))
                ld = sum(1 for op in seq if op.lower().startswith(("scratch_load", "buffer_load_dword")))
                return st, ld
            st = sum(1 for op in seq if op.split(".")[0].upper() == "STL")
            ld = sum(1 for op in seq if op.split(".")[0].upper() == "LDL")
            return st, ld

        b_stl, b_ldl = _count_spill_ops(sequences[0])
        i_stl, i_ldl = _count_spill_ops(sequences[1])
        l0_spill_stats = {
            "regs": runtime.get("baseline_regs"),
            "spill_stores": baseline,
            "stack_bytes": runtime.get("baseline_stack_bytes"),
            "stl": runtime.get("baseline_stl_count", b_stl),
            "ldl": runtime.get("baseline_ldl_count", b_ldl),
        }
        inst_spill_stats = {
            "regs": runtime.get("instrumented_regs"),
            "spill_stores": instrumented,
            "stack_bytes": runtime.get("instrumented_stack_bytes"),
            "stl": runtime.get("instrumented_stl_count", i_stl),
            "ldl": runtime.get("instrumented_ldl_count", i_ldl),
        }
        spill_gate = verify_compiler_spill_gate(
            l0_spill_stats,
            inst_spill_stats,
            max_delta_spill_bytes=int(runtime.get("max_delta_spill_bytes", 0)),
            max_delta_stack_bytes=int(runtime.get("max_delta_stack_bytes", 0)),
            max_delta_stl=int(runtime.get("max_delta_stl", 4)),
            max_delta_ldl=int(runtime.get("max_delta_ldl", 4)),
            has_setmaxnreg=bool(runtime.get("has_setmaxnreg", False)),
            smem_free_bytes=runtime.get("smem_free_bytes"),
        )
        metrics["compiler_spill_gate"] = spill_gate
        code_preserved = (core_ratio >= 0.95 and ratio >= 0.60) if core_seqs[0] else (ratio >= 0.85)
        checks["assembly"] = code_preserved and delta <= 0 and smem <= 0 and spill_gate["ok"]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors["assembly"] = str(exc)
    try:
        a, b = [directory / runtime[f"{key}_output"] for key in ("baseline", "instrumented")]
        if a.samefile(b) or a.stat().st_size == 0:
            raise ValueError("Expected two independently captured, nonempty full outputs")
        metrics["output_bytes"] = a.stat().st_size
        checks["output"] = filecmp.cmp(a, b, shallow=False)
        if runtime.get("expected_output"):
            checks["output"] &= filecmp.cmp(a, directory / runtime["expected_output"], shallow=False)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors["output"] = str(exc)
    try:
        decoder = implementation("4_validate.tools.fi_lite_decode")
        dump = decoder.read_dump(str(directory / "capture.fi02.bin"))
        trace, summary = decoder.decode(dump, read_json(directory / "names.json"))
        fires = summary["fires"]
        metrics["overflow"] = dump["overflow"]
        metrics["pairing_errors"] = summary["errors"]
        checks["pairing"] = not any(summary["errors"].values()) and dump["overflow"] == 0 and all(
            fires.get(p.get("id") or p.get("probe_id"), 0) > 0 for p in spec["probes"])
    except (OSError, ValueError, TypeError, KeyError, SystemExit) as exc:
        errors["pairing"] = str(exc)
    try:
        samples = [runtime[key + "_ns"] for key in ("baseline", "instrumented")]
        if (runtime["warmup"] < 1 or len(samples[0]) < 3 or len(samples[0]) != len(samples[1]) or
                not all(math.isfinite(t) and t > 0 for series in samples for t in series)):
            raise ValueError("Expected positive, paired steady-state GPU timings after warmup")
        base, inst = map(statistics.median, samples)
        ratio = inst / base - 1
        metrics.update(overhead_ratio=ratio, baseline_median_ns=base, instrumented_median_ns=inst,
                       paired_samples=len(samples[0]))
        checks["overhead"] = ratio <= limit
    except (ValueError, TypeError, KeyError) as exc:
        errors["overhead"] = str(exc)
    return checks, metrics, errors, fires, trace, summary


def validate_trace_product(aligned, summary, spec, source_text, runtime):
    """Product-level gates inspired by fast-instrument v1 `validate_capability.py`:
    rejects single-CTA truncation, out-of-range warp indices (e.g. Warp 449),
    collapsed single-blob stages missing load/mma, missing cross-warp async handoffs,
    or missing hardware Compute / Memory utilization curves.
    """
    al = (aligned.get("otherData") or {}).get("alignment") or {}
    amd = (aligned.get("otherData") or {}).get("platform") == "amd"
    wpc = max(1, int(al.get("warps_per_cta") or summary.get("wpc") or 1))
    max_arch_warps = 16 if amd else 32
    expected_ctas = int(runtime.get("grid_ctas") or summary.get("n_ctas") or 0)
    min_units = int(runtime.get("min_physical_units") or min(expected_ctas, 8))
    min_tile_fires = int(runtime.get("min_tile_fires") or 2)

    def _as_int(val, default=-1):
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    events = aligned.get("traceEvents") or []
    sw_spans = [e for e in events if e.get("ph") == "X" and _as_int(e.get("pid"), 0) >= 100000]
    stage_spans = [e for e in sw_spans if (e.get("args") or {}).get("kind") != "residency"]

    topo_spans = stage_spans or sw_spans
    warps_seen = {_as_int((e.get("args") or {}).get("warp"), -1) for e in topo_spans}
    ctas_seen = {_as_int((e.get("args") or {}).get("cta"), -1) for e in topo_spans}
    units_seen = _as_int(al.get("physical_units_observed"), 0)

    failures = []
    # 1. Topology & Warp-ID sanity gate
    if not sw_spans or wpc > max_arch_warps or any(w < 0 or w >= wpc for w in warps_seen):
        failures.append(f"Invalid intra-CTA warp indices {sorted(warps_seen)[:8]} for warps_per_cta={wpc}")
    if expected_ctas > 0 and len(ctas_seen) != expected_ctas:
        failures.append(f"Incomplete CTA coverage: observed {len(ctas_seen)}/{expected_ctas} CTAs")
    if expected_ctas > 1 and units_seen < min_units:
        failures.append(f"Insufficient physical SM/CU coverage: observed {units_seen} < {min_units}")

    # 2. Resource-class aware stage coverage & cross-warp async handoff gate
    sem_seen = {str((e.get("args") or {}).get("semantic_id") or e.get("name") or "").lower() for e in stage_spans}
    has_load = bool(sem_seen & {"load", "copy", "communication"})
    has_mma = bool(sem_seen & {"mma", "gemm", "compute"})
    has_any_active = bool(sem_seen & {"load", "copy", "communication", "mma", "gemm", "compute", "store", "epilogue", "reduction", "softmax", "convolution"})
    kernel_class = str(runtime.get("kernel_class") or "").strip().lower()
    non_gemm_classes = {"non_gemm", "conv1d", "convolution", "rmsnorm", "layernorm", "softmax", "reduction", "elementwise", "recurrence"}
    sass_ops = runtime.get("sass_opcodes") or []
    if sass_ops:
        sass_blob = " ".join(str(x) for x in sass_ops)
        has_tensor_ops_in_src = bool(re.search(
            r"\b(?:HMMA|GMMA|QGMMA|UTCHMMA|UTCOMMA|UTCQMMA|IMMA|DMMA|BMMA|UTMALDG|V_MFMA|V_WMMA)\b",
            sass_blob,
            re.I,
        ))
    else:
        has_tensor_ops_in_src = bool(re.search(
            r"\b(?:wgmma|tcgen05|wmma|mma\.sync|tl\.dot|cute\.gemm|mfma|HMMA|GMMA|tma_load|cp\.async)\b",
            source_text or "",
            re.I,
        ))
    has_non_gemm_cues = bool(re.search(
        r"\b(?:rmsnorm|layernorm|conv1d|causal_conv|softmax|__shfl_xor_sync|__shfl_down_sync)\b",
        source_text or "",
        re.I,
    ))
    is_non_gemm = (kernel_class in non_gemm_classes) or (
        not kernel_class and not has_tensor_ops_in_src and has_non_gemm_cues
    )
    if is_non_gemm:
        if not has_any_active:
            failures.append(
                f"Degenerate non-GEMM stage coverage {sorted(sem_seen)}: trace must expose at least one active compute/memory stage."
            )
    elif not (has_load and has_mma):
        failures.append(
            f"Degenerate stage coverage {sorted(sem_seen)}: trace must expose both 'load' and 'mma' stages "
            "(never wrap the entire K-loop into a single coarse 'compute' block)."
        )
    max_fires = max((int(v) for v in (summary.get("fires") or {}).values()), default=0)
    if max_fires < min_tile_fires:
        failures.append(
            f"Insufficient tile granularity (max probe fires={max_fires} < {min_tile_fires}): "
            "place 'load' and 'mma' probes per K-tile inside the main pipeline loop."
        )
    roles = spec.get("roles") or {}
    has_async_mbar = bool(re.search(r"mbarrier|cp\.async|tma_load", source_text))
    if len(roles) >= 2 and has_async_mbar and int(al.get("cross_warp_spans") or 0) <= 0:
        failures.append(
            "Missing cross-warp async pairing: warp-specialized pipeline has producer/consumer roles and "
            "mbarrier, so 'load' must pair begin on DMA role with end on Math role via 'slot'."
        )

    # 3. Semantic body / empty-stub & SW-vs-HW coverage fidelity gates
    from collections import defaultdict
    stage_durs_ns = defaultdict(list)
    t_min_us, t_max_us = None, None
    for e in stage_spans:
        ts = e.get("ts")
        dur = e.get("dur")
        if ts is not None and dur is not None:
            ts_f, dur_f = float(ts), float(dur)
            if dur_f >= 0.0:
                stage_durs_ns[str((e.get("args") or {}).get("detail") or e.get("name") or "stage")].append(dur_f * 1000.0)
                t_min_us = ts_f if t_min_us is None else min(t_min_us, ts_f)
                t_max_us = (ts_f + dur_f) if t_max_us is None else max(t_max_us, ts_f + dur_f)
    stub_report = verify_semantic_body_and_stubs(
        stage_durs_ns,
        min_median_ns=float(runtime.get("min_stage_median_ns", 128.0)),
        sass_body_insns=runtime.get("sass_body_insns"),
    )
    if runtime.get("enforce_stub_gate") and not stub_report["ok"]:
        failures.append(f"Empty/self-latency stub probes detected: {[s['stage'] for s in stub_report['stubs']]}")

    coverage_report = None
    hw_kernel_us = al.get("kernel_duration_us") or runtime.get("hw_kernel_us")
    sw_span_us = (t_max_us - t_min_us) if (t_min_us is not None and t_max_us is not None) else None
    if hw_kernel_us is not None and sw_span_us is not None:
        coverage_report = verify_sw_hw_coverage(
            sw_span_us,
            float(hw_kernel_us),
            in_window_samples=runtime.get("pmu_in_window_samples"),
            expected_freq_hz=float(runtime.get("pmu_freq_hz", 200000.0)),
            max_span_delta_ratio=float(runtime.get("max_span_delta_ratio", 0.05)),
            min_pmu_density=float(runtime.get("min_pmu_density", 0.90)),
        )
        if runtime.get("enforce_coverage_gate") and not coverage_report["ok"]:
            failures.extend(v["message"] for v in coverage_report["violations"])

    # 4. Hardware utilization curves gate
    if not (al.get("has_compute_curve") and al.get("has_memory_curve") and al.get("has_sm_curve")):
        failures.append(
            f"Missing hardware utilization curves: compute={al.get('has_compute_curve')}, "
            f"memory={al.get('has_memory_curve')}, sm_active={al.get('has_sm_curve')}"
        )

    return {
        "ok": not failures,
        "failures": failures,
        "kernel_class": "non_gemm" if is_non_gemm else "gemm_or_tensor",
        "ctas_observed": len(ctas_seen),
        "expected_ctas": expected_ctas,
        "warps_per_cta": wpc,
        "max_warp_index": max(warps_seen) if warps_seen else -1,
        "physical_units_observed": units_seen,
        "stages_observed": sorted(sem_seen),
        "max_tile_fires": max_fires,
        "cross_warp_spans": int(al.get("cross_warp_spans") or 0),
        "stub_gate": stub_report,
        "coverage_gate": coverage_report,
        "has_compute_curve": bool(al.get("has_compute_curve")),
        "has_memory_curve": bool(al.get("has_memory_curve")),
        "has_sm_curve": bool(al.get("has_sm_curve")),
    }


def _terminate_process_tree(proc):
    """Kill the entire child process group/tree so GPU workers cannot outlive timeout."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=15,
            )
        else:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def run_cmd(argv, cwd, env, log, *, timeout_s=None):
    popen_kwargs = dict(cwd=cwd, env=env, stdin=subprocess.DEVNULL)
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True
    with Path(log).open("wb") as stream:
        proc = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, **popen_kwargs)
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_tree(proc)
            detail = Path(log).read_bytes()[-1000:].decode("utf-8", "replace") if Path(log).is_file() else ""
            raise RuntimeError(
                f"{Path(argv[0]).name} timed out after {timeout_s}s ({log}): {detail}"
            ) from exc
    if rc:
        detail = Path(log).read_bytes()[-1400:].decode("utf-8", "replace")
        raise RuntimeError(f"{Path(argv[0]).name} exited {rc} ({log}): {detail}")


def run_benchmark(workspace, prepared, bench, spec, *, timeout_s=None):
    directory, base, inst = prepared
    env = workspace.environment(directory, base, inst)
    if spec.get("language") == "triton" and Path(bench[0]).stem.lower().startswith("python"):
        argv, cwd = workspace.command(bench, directory, baseline=True)
        run_cmd([argv[0], "-B", "-m", "weaveprof.replay", "collect", str(base),
                 str(directory), *argv[1:]], cwd, env, directory / "bench.log", timeout_s=timeout_s)
        from .replay import measure
        return measure(workspace, prepared)
    argv, cwd = workspace.command(bench, directory)
    run_cmd(argv, cwd, env, directory / "bench.log", timeout_s=timeout_s)


def _export_failed_timeline(directory, trace, summary, rt, spec, source_text, max_hotspots=6):
    """Export failed-run timeline binary (`failed_trace.perfetto-trace`) and compact hotspot summary."""
    failed_trace = directory / "failed_trace.perfetto-trace"
    export_doc = trace
    try:
        merge = implementation("4_validate.tools.fi_merge_perfetto")
        window = merge.hardware_window(directory, rt)
        export_doc = merge.align(trace, summary, window, rt)
    except Exception:
        export_doc = trace
    try:
        exporter = implementation("4_validate.tools.fi_perfetto_proto")
        exporter.export(export_doc, failed_trace)
    except Exception:
        failed_trace = None
    return failed_trace, summarize_failed_timeline(
        export_doc,
        summary,
        spec,
        source_text,
        rt,
        failed_trace_path=failed_trace,
        max_hotspots=max_hotspots,
    )


def verify_round(directory, spec, source_text, limit, out, feedback, measure,
                 verify_json=None, preview_png=None, *,
                 policy=None, gpu_id="auto", round_number=1,
                 source_path=None, source_sha256=None, probe_path=None,
                 instrumented_path=None, bench_cmd=None,
                 fsm_state="VERIFY_AND_SEAL", escalation=None,
                 prev_seal_sha256=None):
    active_policy = policy if isinstance(policy, HarnessPolicy) else HarnessPolicy()
    guard = PolicyGuard(active_policy)
    failures, metrics, fires = {}, {}, {}
    gates = dict.fromkeys(("assembly", "output", "pairing", "overhead"), False)
    alignment_error = None
    product_report = None
    region_lock_info = None
    failed_trace_path = None
    timeline_summary = None
    trace, summary, rt = None, {}, {}
    out_path = Path(out)
    # Audit #8: Never allow a stale previous-round/session trace at `out` to survive into seal binding.
    if out_path.is_file():
        try:
            out_path.unlink()
        except OSError:
            pass
    try:
        if active_policy.enable_region_lock:
            with gpu_critical_region(
                gpu_id=gpu_id,
                timeout_s=active_policy.region_lock_timeout_s,
                label=f"verify_round_{round_number}",
            ) as lease:
                measure()
            region_lock_info = lease.to_dict()
        else:
            measure()
        rt = read_json(directory / "runtime.json")
        gates, metrics, failures, fires, trace, summary = assess(directory, spec, limit, rt)
        if region_lock_info is not None:
            metrics["region_lock"] = region_lock_info
        proto_ok, abba_info, proto_errs = guard.validate_runtime_protocol(rt, limit)
        if abba_info:
            metrics["abba"] = abba_info
        if not proto_ok:
            gates["overhead"] = False
            failures["runtime_protocol"] = "; ".join(proto_errs)
        if all(gates.values()):
            try:
                merge = implementation("4_validate.tools.fi_merge_perfetto")
                window = merge.hardware_window(directory, rt)
                aligned = merge.align(trace, summary, window, rt)
                metrics["alignment"] = aligned["otherData"]["alignment"]
                if rt.get("require_product_gate") or rt.get("grid_ctas"):
                    product_report = validate_trace_product(aligned, summary, spec, source_text, rt)
                    metrics["product_gate"] = product_report
                    if not product_report["ok"]:
                        gates["pairing"] = False
                        failures["product_gate"] = "; ".join(product_report["failures"])
                if all(gates.values()):
                    exporter = implementation("4_validate.tools.fi_perfetto_proto")
                    exporter.export(aligned, out_path)
                    tp_report = verify_trace_processor_sql(out_path)
                    metrics["trace_processor"] = tp_report
                    if active_policy.enforce_trace_processor and tp_report.get("ok") is not True:
                        gates["pairing"] = False
                        failures["trace_processor"] = "; ".join(
                            tp_report.get("errors") or [tp_report.get("reason", "trace_processor check failed")]
                        )
                    if all(gates.values()):
                        # Audit #20: Auxiliary json.gz / preview PNG errors must not invalidate a verified trace.
                        aux_warnings = []
                        try:
                            json_gz = out_path.with_suffix(".sw_hw.perfetto.json.gz")
                            exporter.export(aligned, json_gz)
                            metrics["perfetto_json_gz"] = str(json_gz)
                        except Exception as aux_exc:
                            aux_warnings.append(f"json_gz export warning: {aux_exc}")
                        try:
                            png_target = Path(preview_png) if preview_png else out_path.with_suffix(".png")
                            from .preview import render_timeline_png
                            render_timeline_png(
                                aligned,
                                {"metrics": metrics},
                                png_target,
                                title_override=rt.get("plot_title"),
                            )
                            metrics["preview_png"] = str(png_target)
                        except Exception as aux_exc:
                            aux_warnings.append(f"preview_png warning: {aux_exc}")
                        if aux_warnings:
                            metrics["auxiliary_warnings"] = aux_warnings
            except Exception as exc:
                alignment_error = str(exc)
        if not all(gates.values()) and active_policy.export_failed_timeline and trace is not None:
            # Audit #21: Isolate failed-timeline summary formatting from measurement/gate errors.
            try:
                failed_trace_path, timeline_summary = _export_failed_timeline(
                    directory, trace, summary, rt, spec, source_text,
                    max_hotspots=active_policy.max_timeline_hotspots,
                )
                if failed_trace_path is not None:
                    metrics["failed_trace"] = str(failed_trace_path)
                if timeline_summary is not None:
                    metrics["timeline_summary"] = timeline_summary
            except Exception as sum_exc:
                metrics.setdefault("auxiliary_warnings", []).append(f"failed_timeline_summary warning: {sum_exc}")
    except (Exception, SystemExit) as exc:
        # Audit #7: Fail-closed reset of ALL gates on any unhandled exception during measurement/assessment.
        gates = dict.fromkeys(("assembly", "output", "pairing", "overhead"), False)
        failures["measurement"] = str(exc)
    if failures:
        metrics["errors"] = failures
    verifier = implementation("4_validate.closed_loop_verifier")
    document = verifier.build_feedback_json({"gates": gates}, spec, source_text, fires=fires, metrics=metrics)
    document["checks"] = {
        "assembly": bool(gates["assembly"]),
        "output": bool(gates["output"]),
        "event": bool(gates["pairing"]),
        "events": bool(gates["pairing"]),
        "overhead": bool(gates["overhead"]),
    }
    if failed_trace_path is not None:
        document["failed_trace"] = str(failed_trace_path)
    if timeline_summary is not None:
        document["timeline_summary"] = timeline_summary
    if alignment_error is not None:
        document.update(status="FAIL", export_error=alignment_error)
    fb_round = directory / "feedback.json"
    ver_round = directory / "verify.json"
    write_json(fb_round, document)
    write_json(ver_round, document)
    write_json(feedback, document)
    v_path = Path(verify_json) if verify_json else out_path.parent / "verify.json"
    write_json(v_path, document)
    if source_path is not None and source_sha256 is not None:
        # Audit #6: Bind raw measurement inputs (runtime, capture dump, SASS, outputs) into the round seal.
        extra_arts = {
            "runtime": directory / "runtime.json",
            "capture_dump": directory / "capture.fi02.bin",
        }
        for key in ("baseline_sass", "instrumented_sass", "baseline_output", "instrumented_output"):
            if isinstance(rt, dict) and rt.get(key):
                extra_arts[key] = directory / rt[key]
        seal_file, seal_doc = EpisodeSealer.seal_round(
            round_dir=directory,
            round_number=round_number,
            source_path=Path(source_path),
            source_sha256=source_sha256,
            probe_path=Path(probe_path) if probe_path else (directory / "probe.json"),
            instrumented_path=Path(instrumented_path) if instrumented_path else None,
            names_path=(directory / "names.json") if (directory / "names.json").is_file() else None,
            feedback_path=fb_round,
            verify_path=ver_round,
            trace_path=out_path if (document.get("status") == "PASS" and out_path.is_file()) else None,
            failed_trace_path=failed_trace_path,
            result=document,
            policy=active_policy,
            limit=limit,
            bench_cmd=bench_cmd,
            fsm_state=fsm_state,
            escalation=escalation,
            region_lock=region_lock_info,
            prev_seal_sha256=prev_seal_sha256,
            extra_artifacts=extra_arts,
        )
        document["seal_path"] = str(seal_file)
        document["seal_sha256"] = seal_doc["seal_sha256"]
    return document
