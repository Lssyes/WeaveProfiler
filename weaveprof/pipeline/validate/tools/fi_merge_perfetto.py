"""Align decoded software warp events with hardware profiler dispatches and utilization curves."""
from __future__ import annotations

from collections import defaultdict
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3


def _select(rows, instance):
    if not rows or (instance is None and len(rows) != 1):
        raise ValueError(f"Expected one matching hardware dispatch, found {len(rows)}; specify capture_instance")
    return rows[0 if instance is None else int(instance)]


def _nsys(path, kernel, instance):
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        rows = list(db.execute(
            "SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k.demangledName WHERE instr(s.value,?) > 0 ORDER BY k.start", (kernel,)))
        start, end, name = _select(rows, instance)
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master")}
        samples = []
        if {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"} <= tables:
            cols = {r[1] for r in db.execute("PRAGMA table_info(TARGET_INFO_GPU_METRICS)")}
            join_clause = "USING(typeId,metricId)" if "typeId" in cols else "ON m.metricId=i.metricId"
            samples = list(db.execute(
                f"SELECT m.timestamp,i.metricName,m.value FROM GPU_METRICS m "
                f"JOIN TARGET_INFO_GPU_METRICS i {join_clause} "
                f"WHERE m.timestamp BETWEEN ? AND ? ORDER BY m.timestamp", (start, end)))
        return int(start), int(end), str(name), [(int(t), str(n), float(v)) for t, n, v in samples]


def _pm_json(path, kernel):
    doc = json.loads(path.read_text(encoding="utf-8"))
    disp = doc["dispatch"]
    start = int(disp["ts"])
    dur = int(disp["dur"])
    end = start + dur
    name = str(disp.get("name") or kernel)
    raw = doc.get("raw") or {}
    samples = []
    gui = float(raw.get("GRBM_GUI_ACTIVE", 0.0))
    mfma = float(raw.get("SQ_VALU_MFMA_BUSY_CYCLES", 0.0))
    if gui > 0:
        mfma_pct = min(100.0, 100.0 * mfma / (gui * 80.0))
        samples.extend([(start, "Compute Util — Matrix Core MFMA (%) [ROCm PMC]", mfma_pct),
                        (end, "Compute Util — Matrix Core MFMA (%) [ROCm PMC]", mfma_pct)])
    rd = float(raw.get("TCC_EA0_RDREQ_DRAM", 0.0))
    wr = float(raw.get("TCC_EA0_WRREQ_DRAM", 0.0))
    if dur > 0 and (rd > 0 or wr > 0):
        # MI308X measured HBM3 peak = 3684 GB/s (3684 bytes/ns), 64 B per TCC DRAM request
        hbm_pct = min(100.0, 100.0 * (rd + wr) * 64.0 / (dur * 3684.0))
        samples.extend([(start, "Memory Util — HBM3 Read+Write (%) [ROCm PMC]", hbm_pct),
                        (end, "Memory Util — HBM3 Read+Write (%) [ROCm PMC]", hbm_pct)])
    if "SIMD_UTILIZATION" in raw:
        simd_val = float(raw["SIMD_UTILIZATION"])
        simd_pct = simd_val * 100.0 if simd_val <= 1.5 else simd_val
        samples.extend([(start, "SIMD VALU Util (%) [ROCm PMC]", min(100.0, simd_pct)),
                        (end, "SIMD VALU Util (%) [ROCm PMC]", min(100.0, simd_pct))])
    if "GPU_UTIL" in raw:
        gpu_pct = float(raw["GPU_UTIL"])
        samples.extend([(start, "Active CUs / GPU Busy (%) [ROCm PMC]", min(100.0, gpu_pct)),
                        (end, "Active CUs / GPU Busy (%) [ROCm PMC]", min(100.0, gpu_pct))])
    return start, end, name, samples


def hardware_window(directory, runtime):
    kernel = runtime["kernel_name"]
    if not isinstance(kernel, str) or not kernel:
        raise ValueError("A hardware kernel name is required")
    instance = runtime.get("capture_instance")
    sqlite = directory / "capture.sqlite"
    if sqlite.is_file():
        return _nsys(sqlite, kernel, instance)
    pm_file = directory / "pm.json"
    if pm_file.is_file():
        return _pm_json(pm_file, kernel)
    results = directory / "rocprof.json"
    if results.is_file():
        adapter = importlib.import_module("weaveprof.pipeline.capture.build_rocprof_sw_hw")
        tool = adapter._tool_root(json.loads(results.read_text(encoding="utf-8")))
        symbols = adapter._kernel_names(tool)
        records = (tool.get("callback_records") or {}).get("counter_collection") or (
            tool.get("buffer_records") or {}).get("kernel_dispatch", [])
        rows = []
        for record in records:
            info = adapter._dispatch_info(record)
            name = symbols.get(info.get("kernel_id"), "")
            window = adapter._start_end(record)
            if window and kernel in name:
                defs = adapter._counter_definitions(tool, (info.get("agent_id") or {}).get("handle"))
                counters = [(window[0], defs[item["counter_id"]["handle"]]["name"], float(item["value"]))
                            for item in record.get("records", []) if item["counter_id"]["handle"] in defs]
                rows.append((*window, name, counters))
        return _select(sorted(rows, key=lambda r: r[0]), instance)
    proto = directory / "rocprof-sys.pftrace"
    if proto.is_file():
        adapter = importlib.import_module("weaveprof.pipeline.capture.build_rocprof_sys_sw_hw")
        tool = os.environ.get("TRACE_PROCESSOR_SHELL") or shutil.which("trace_processor_shell")
        if not tool:
            raise ValueError("Reading rocprof-sys.pftrace requires trace_processor_shell")
        rows = adapter._query_rows(Path(tool), proto, "SELECT ts,dur,name FROM slice WHERE dur>0 "
                                  f"AND instr(name,'{adapter._sql_quote(kernel)}')>0 ORDER BY ts")
        row = _select(rows, instance)
        start, end = int(row["ts"]), int(row["ts"]) + int(row["dur"])
        counters = adapter._query_rows(Path(tool), proto, "SELECT c.ts,t.name,c.value FROM counter c "
                                     "JOIN counter_track t ON t.id=c.track_id "
                                     f"WHERE c.ts BETWEEN {start} AND {end} ORDER BY c.ts")
        return start, end, row["name"], [(int(c["ts"]), c["name"], float(c["value"])) for c in counters]
    raise ValueError("Missing hardware capture (capture.sqlite, pm.json, rocprof.json or rocprof-sys.pftrace)")


def _interval_union(intervals):
    out = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def _bin_exposure(intervals, dur_ns, bin_ns=2000):
    n = max(1, math.ceil(dur_ns / bin_ns))
    area = [0.0] * n
    for s, e in intervals:
        s = max(0, min(dur_ns, int(s)))
        e = max(0, min(dur_ns, int(e)))
        if e <= s:
            continue
        lo, hi = s // bin_ns, (e - 1) // bin_ns
        if lo == hi:
            area[lo] += e - s
        else:
            area[lo] += (lo + 1) * bin_ns - s
            area[hi] += e - hi * bin_ns
            for k in range(lo + 1, hi):
                area[k] += bin_ns
    widths = [min(bin_ns, dur_ns - i * bin_ns) for i in range(n)]
    return [area[i] / max(1, widths[i]) for i in range(n)], widths


def align(trace, summary, window, runtime=None):
    start, end, kernel, samples = window
    dur_ns = int(end) - int(start)
    if dur_ns <= 0:
        raise ValueError("Invalid hardware dispatch duration")
    runtime = runtime or {}

    origin = int(summary["observation_start_ns"])
    obs_end = int(summary.get("observation_end_ns", origin))
    # Midpoint of feasible clock-offset interval when span fits inside hardware window;
    # otherwise anchor at first_open_ns -> start (preserving exact device durations).
    lower = int(start) - origin
    upper = int(end) - obs_end
    offset = (lower + upper) // 2 if lower <= upper else (int(start) - int(summary["first_open_ns"]))

    wpc = max(1, int(summary.get("wpc") or 1))
    amd = summary.get("platform") == "amd"
    unit_tag = "CU" if amd else "SM"
    cta_sm = {int(k): int(v) for k, v in (summary.get("cta_sm") or {}).items()}
    cta_hw = {int(k): v for k, v in (summary.get("cta_hw") or {}).items()}
    lane_bounds = summary.get("lane_bounds_ns") or {}
    role_names = {int(k): str(v) for k, v in (trace.get("otherData", {}).get("roles") or {}).items()}

    # Assign physical SM/CU process IDs (100000 + ordinal) and greedy CTA residency slots per SM/CU
    cta_life = defaultdict(lambda: [1 << 62, 0])
    for s_str, (b_raw, e_raw, _r) in lane_bounds.items():
        cta = int(s_str) // wpc
        b_rel = max(0, int(b_raw) + offset - int(start))
        e_rel = min(dur_ns, max(b_rel + 1, int(e_raw) + offset - int(start)))
        cta_life[cta][0] = min(cta_life[cta][0], b_rel)
        cta_life[cta][1] = max(cta_life[cta][1], e_rel)

    sm_ids = sorted(set(cta_sm.values())) if cta_sm else []
    sm_ord = {sm: idx for idx, sm in enumerate(sm_ids)}
    cta_res_slot = {}
    sm_ctas = defaultdict(list)
    for cta, (b_rel, e_rel) in cta_life.items():
        sm_ctas[cta_sm.get(cta, 0)].append((b_rel, e_rel, cta))
    sm_max_slots = {}
    for sm, items in sm_ctas.items():
        avail = []
        for b_rel, e_rel, cta in sorted(items):
            sl = next((j for j, fin in enumerate(avail) if fin <= b_rel), len(avail))
            if sl == len(avail):
                avail.append(e_rel)
            else:
                avail[sl] = e_rel
            cta_res_slot[cta] = sl
        sm_max_slots[sm] = max(1, len(avail))

    # Shift all existing events to aligned ns & rebuild clean physical SM/CU + bounded Warp (0..wpc-1) tracks
    clean_events = []
    per_sem_rel = defaultdict(list)
    if sm_ids and lane_bounds:
        for sm in sm_ids:
            pid = 100000 + sm_ord[sm]
            loc = ""
            if amd:
                any_cta = sm_ctas[sm][0][2]
                hw = cta_hw.get(any_cta)
                if hw:
                    loc = f" [XCC{hw[0]} SE{hw[1]} CU{hw[2]}]"
            clean_events.append(dict(ph="M", pid=pid, tid=0, name="process_name",
                                     args=dict(name=f"{unit_tag} {sm:03d}{loc}")))
            clean_events.append(dict(ph="M", pid=pid, tid=0, name="process_sort_index",
                                     args=dict(sort_index=1000 + sm_ord[sm])))
            for sl in range(sm_max_slots[sm]):
                for w in range(wpc):
                    tid = 1 + sl * 32 + w
                    w_lbl = f"wave64 {w:02d}" if amd else f"warp {w:02d}"
                    clean_events.append(dict(ph="M", pid=pid, tid=tid, name="thread_name",
                                             args=dict(name=f"{w_lbl} · slot {sl}")))
                    clean_events.append(dict(ph="M", pid=pid, tid=tid, name="thread_sort_index",
                                             args=dict(sort_index=tid)))

        # Emit Warp/Wave residency envelopes (OPEN..CLOSE)
        for s_str, (b_raw, e_raw, r_id) in sorted(lane_bounds.items(), key=lambda kv: int(kv[0])):
            s = int(s_str)
            cta, w = divmod(s, wpc)
            sm = cta_sm.get(cta, 0)
            pid = 100000 + sm_ord[sm]
            sl = cta_res_slot.get(cta, 0)
            tid = 1 + sl * 32 + w
            ts_ns = int(b_raw) + offset
            end_ns = max(ts_ns + 1, int(e_raw) + offset)
            r_nm = role_names.get(int(r_id), str(r_id))
            res_args = dict(cta=cta, warp=w, wave=w, sm=sm, role=r_nm, resident_slot=sl, kind="residency")
            if amd and cta in cta_hw:
                hw = cta_hw[cta]
                res_args.update(xcc=hw[0], se=hw[1], cu=hw[2], simd=hw[3])
            clean_events.append(dict(
                ph="X", pid=pid, tid=tid, ts_ns=ts_ns, ts=ts_ns / 1000.0,
                dur=(end_ns - ts_ns) / 1000.0,
                name="Wave residency" if amd else "Warp residency",
                cat="residency", args=res_args,
            ))

        for e in trace["traceEvents"]:
            if e.get("ph") != "X":
                continue
            args = dict(e.get("args") or {})
            cta = int(args.get("cta", 0))
            w = int(args.get("warp", 0))
            sm = cta_sm.get(cta, int(args.get("sm") or 0))
            pid = 100000 + sm_ord.get(sm, 0)
            sl = cta_res_slot.get(cta, 0)
            sub_inst = int(args.get("instance", 0))
            tid = 1 + sl * 32 + w
            ts_ns = origin + round(float(e["ts"]) * 1000) + offset
            dur_us = float(e["dur"])
            end_ns = ts_ns + round(dur_us * 1000)
            args.update(cta=cta, warp=w, wave=w, sm=sm, resident_slot=sl, sublane=sub_inst, kind="stage")
            clean_events.append(dict(
                ph="X", pid=pid, tid=tid, ts_ns=ts_ns, ts=ts_ns / 1000.0,
                dur=dur_us, name=e["name"], cat=e.get("cat", "stage"), args=args,
            ))
            rel_a = max(0, ts_ns - int(start))
            rel_b = min(dur_ns, end_ns - int(start))
            if rel_b > rel_a:
                sem = str(args.get("semantic_id") or e["name"]).lower()
                per_sem_rel[sem].append((rel_a, rel_b))
    else:
        for e in trace["traceEvents"]:
            ec = dict(e)
            if "ts" in ec:
                ec["ts_ns"] = origin + round(float(ec["ts"]) * 1000) + offset
                ec["ts"] = ec["ts_ns"] / 1000.0
            clean_events.append(ec)

    # Hardware & Activity Counter Process (pid = 90000)
    hw_pid = 90000
    clean_events.extend([
        dict(ph="M", pid=hw_pid, tid=0, name="process_name", args=dict(name="Hardware & Activity Curves")),
        dict(ph="M", pid=hw_pid, tid=0, name="process_sort_index", args=dict(sort_index=-100)),
        dict(ph="X", pid=hw_pid, tid=0, ts_ns=int(start), ts=int(start) / 1000.0,
             dur=dur_ns / 1000.0, name=kernel, cat="hardware", args=dict(kind="dispatch")),
    ])

    # Normalize & aggregate raw NSYS GPU_METRICS (e.g. sum DRAM Read + Write bandwidth) + emit raw curves
    pmu_by_ts = defaultdict(dict)
    has_compute_hw = False
    has_mem_hw = False
    has_sm_hw = False
    for ts, mname, val in samples:
        low = mname.lower()
        clean_events.append(dict(ph="C", pid=hw_pid, tid=1, ts_ns=int(ts), ts=int(ts) / 1000.0,
                                 name=mname, args=dict(value=float(val))))
        pmu_by_ts[int(ts)][mname] = float(val)
        if any(k in low for k in ("tensor active", "compute util", "mfma", "hmma", "math pipe")):
            has_compute_hw = True
        if any(k in low for k in ("memory util", "hbm", "dram throughput")):
            has_mem_hw = True
        if any(k in low for k in ("sms active", "active cus", "gpu busy")):
            has_sm_hw = True

    # Synthesize combined NSYS PMU DRAM Read+Write curve & Tensor Core curve if raw NSYS metrics exist
    if pmu_by_ts and not amd:
        for ts in sorted(pmu_by_ts):
            row = pmu_by_ts[ts]
            rd_vals = [v for k, v in row.items() if "dram read" in k.lower()]
            wr_vals = [v for k, v in row.items() if "dram write" in k.lower()]
            if rd_vals or wr_vals:
                bw_pct = min(100.0, (rd_vals[0] if rd_vals else 0.0) + (wr_vals[0] if wr_vals else 0.0))
                clean_events.append(dict(ph="C", pid=hw_pid, tid=1, ts_ns=ts, ts=ts / 1000.0,
                                         name="Memory Util — HBM Read+Write (%) [NSYS PMU]",
                                         args=dict(value=bw_pct)))
                has_mem_hw = True
            tc_vals = [v for k, v in row.items() if "tensor active" in k.lower() or "tensor pipe" in k.lower()]
            if tc_vals:
                clean_events.append(dict(ph="C", pid=hw_pid, tid=1, ts_ns=ts, ts=ts / 1000.0,
                                         name="Compute Util — Tensor Core (%) [NSYS PMU]",
                                         args=dict(value=min(100.0, max(tc_vals)))))
                has_compute_hw = True
            sm_vals = [v for k, v in row.items() if "sms active" in k.lower()]
            if sm_vals:
                clean_events.append(dict(ph="C", pid=hw_pid, tid=1, ts_ns=ts, ts=ts / 1000.0,
                                         name="Active SMs (%) [NSYS PMU]",
                                         args=dict(value=min(100.0, sm_vals[0]))))
                has_sm_hw = True

    # Always emit 2 us binned physical SM/CU residency curve; and when container blocks PMU counters,
    # emit stage-exposure-calibrated Compute & HBM curves anchored to exact hardware dispatch duration.
    bin_ns = 2000 if dur_ns <= 600_000 else 5000
    if sm_ctas:
        total_hw_units = int(runtime.get("total_hw_units") or max(len(sm_ids), 80 if amd else 78))
        active_union = [iv for items in sm_ctas.values() for iv in _interval_union((b, e) for b, e, _ in items)]
        sm_curve, widths = _bin_exposure(active_union, dur_ns, bin_ns)
        occ_name = f"Active {unit_tag} Residency (%) [{total_hw_units} {unit_tag}s]"
        for idx, val in enumerate(sm_curve):
            ts = int(start) + idx * bin_ns
            clean_events.append(dict(ph="C", pid=hw_pid, tid=2, ts_ns=ts, ts=ts / 1000.0,
                                     name=occ_name,
                                     args=dict(value=min(100.0, 100.0 * val / max(1, total_hw_units)))))
        clean_events.append(dict(ph="C", pid=hw_pid, tid=2, ts_ns=int(end), ts=int(end) / 1000.0,
                                 name=occ_name, args=dict(value=0.0)))
        has_sm_hw = True

        # Stage-modulated intra-kernel Compute & Memory curves (calibrated to measured dispatch TFLOP/s & GB/s)
        flops = float(runtime.get("workload_flops") or 0.0)
        peak_tflops = float(runtime.get("peak_tflops") or 0.0)
        mem_bytes = float(runtime.get("workload_bytes") or 0.0)
        peak_bw_gbps = float(runtime.get("peak_bw_gbps") or 0.0)

        mma_spans = per_sem_rel.get("mma", []) + per_sem_rel.get("compute", []) + per_sem_rel.get("gemm", [])
        mem_spans = per_sem_rel.get("load", []) + per_sem_rel.get("store", []) + per_sem_rel.get("epilogue", [])

        if mma_spans and flops > 0 and peak_tflops > 0 and (amd or not has_compute_hw):
            mma_exp, widths = _bin_exposure(mma_spans, dur_ns, bin_ns)
            tot_exp = sum(x * w for x, w in zip(mma_exp, widths))
            mean_pct = 100.0 * (flops / dur_ns / 1e3) / peak_tflops
            if tot_exp > 0:
                cname = ("Wave MFMA Stage Activity (%) [calibrated]" if amd
                         else "Compute Util — Tensor Core (%) [stage-calibrated]")
                for idx, x in enumerate(mma_exp):
                    ts = int(start) + idx * bin_ns
                    pct = min(100.0, x * mean_pct * dur_ns / tot_exp)
                    clean_events.append(dict(ph="C", pid=hw_pid, tid=2, ts_ns=ts, ts=ts / 1000.0,
                                             name=cname, args=dict(value=pct)))
                clean_events.append(dict(ph="C", pid=hw_pid, tid=2, ts_ns=int(end), ts=int(end) / 1000.0,
                                         name=cname, args=dict(value=0.0)))
                has_compute_hw = True

        if mem_spans and mem_bytes > 0 and peak_bw_gbps > 0 and (amd or not has_mem_hw):
            mem_exp, widths = _bin_exposure(mem_spans, dur_ns, bin_ns)
            tot_exp = sum(x * w for x, w in zip(mem_exp, widths))
            mean_bw_pct = 100.0 * (mem_bytes / dur_ns) / peak_bw_gbps
            if tot_exp > 0:
                mname = ("Wave HBM Load/Store Activity (%) [calibrated]" if amd
                         else "Memory Util — HBM Read+Write (%) [stage-calibrated]")
                for idx, x in enumerate(mem_exp):
                    ts = int(start) + idx * bin_ns
                    pct = min(100.0, x * mean_bw_pct * dur_ns / tot_exp)
                    clean_events.append(dict(ph="C", pid=hw_pid, tid=2, ts_ns=ts, ts=ts / 1000.0,
                                             name=mname, args=dict(value=pct)))
                clean_events.append(dict(ph="C", pid=hw_pid, tid=2, ts_ns=int(end), ts=int(end) / 1000.0,
                                         name=mname, args=dict(value=0.0)))
                has_mem_hw = True

    out_trace = dict(trace)
    out_trace["traceEvents"] = clean_events
    other = dict(trace.get("otherData") or {})
    other["alignment"] = dict(
        kernel=kernel, kernel_start_ns=int(start), kernel_end_ns=int(end),
        kernel_duration_us=round(dur_ns / 1000.0, 3),
        first_open_ns=summary["first_open_ns"], offset_ns=int(offset),
        hardware_samples=len(samples),
        physical_units_observed=len(sm_ids),
        ctas_observed=len(cta_life),
        warps_per_cta=wpc,
        cross_warp_spans=int(summary.get("cross_warp_spans") or 0),
        has_compute_curve=has_compute_hw,
        has_memory_curve=has_mem_hw,
        has_sm_curve=has_sm_hw,
    )
    out_trace["otherData"] = other
    return out_trace
