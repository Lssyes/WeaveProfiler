#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed Perfetto trace correctness gate for WeaveProfiler.

Enforces four non-negotiable correctness invariants on both Chrome Trace JSON
(.perfetto.json / .perfetto.json.gz) and Perfetto Native Protobuf
(.perfetto-trace / .perfetto-trace.gz):

  G1. Zero import errors: no tid=0 track collision, no partial X-slice spill
      overlaps, monotonic timestamp ordering per packet sequence, and zero
      non-info rows in Perfetto TraceProcessor `stats` when installed.
  G2. Hardware utilization [0-100%] contract (when --require-hw): hardware
      tracks must represent normalized [0, 100]% percentages (never raw cycle
      or instruction counts) and cover both compute% and memory/DRAM%.
  G3. Hardware topmost pinning (Rendering Contract R6): when hardware
      utilization tracks exist, their process must have strictly smaller pid
      AND strictly smaller sort_index / sibling_order_rank than all SM/CU
      software processes, and occupy the minimum upid in TraceProcessor.
  G4. Physical SM/CU occupancy ceiling: simultaneously active resident CTA/warp
      tracks per SM/CU must never exceed hardware architectural capacity
      (AMD gfx942: <= 32 waves/CU, <= 16 WGs/CU;
       NVIDIA Hopper/Blackwell: <= 64 warps/SM, <= 32 CTAs/SM).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


COMPUTE_KEYWORDS = ("tensor", "compute_util", "matrix_core", "mfma", "fma pipe")
MEMORY_KEYWORDS = ("dram", "memory_util", "memory_read", "memory_write", "hbm")


def _is_native_proto(path: Path) -> bool:
    name = path.name.lower()
    return ".perfetto-trace" in name or name.endswith(".pb") or name.endswith(".proto")


def _read_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if path.suffix.lower() == ".gz" or raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    return raw


def _max_concurrent_tracks(track_spans: Dict[int, List[Tuple[int, int]]]) -> int:
    """Return sweep-line peak count of tracks simultaneously active on one SM/CU."""
    points: List[Tuple[int, int]] = []
    for _tid, spans in track_spans.items():
        if not spans:
            continue
        # Envelope [min_start, max_end] of each distinct resident track
        # Wait: if a track represents sequential CTAs (like CTA驻留槽1), merge overlapping/nested spans on that track first!
        ordered = sorted(spans)
        merged: List[List[int]] = []
        for s_ns, e_ns in ordered:
            if e_ns <= s_ns:
                continue
            if not merged or s_ns >= merged[-1][1]:
                merged.append([s_ns, e_ns])
            else:
                merged[-1][1] = max(merged[-1][1], e_ns)
        for s_ns, e_ns in merged:
            points.append((s_ns, +1))
            points.append((e_ns, -1))
    # Sort (-1 before +1 at identical timestamp so back-to-back sequential CTAs do not add up)
    points.sort(key=lambda item: (item[0], item[1]))
    cur = 0
    peak = 0
    for _ts, delta in points:
        cur += delta
        if cur > peak:
            peak = cur
    return peak


def audit_json_trace(
    path: Path,
    *,
    require_hw: bool = False,
    card: Optional[str] = None,
    waves_or_warps_per_cta: int = 4,
) -> Dict[str, Any]:
    doc = json.loads(_read_bytes(path).decode("utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("traceEvents"), list):
        return {"ok": False, "errors": ["invalid Chrome JSON: missing traceEvents list"]}

    events = doc["traceEvents"]
    other = doc.get("otherData") if isinstance(doc.get("otherData"), dict) else {}
    inferred_card = (
        card
        or (other.get("target") or {}).get("card")
        or (other.get("meta") or {}).get("card")
        or ("mi308x" if str(other.get("platform") or "").lower() == "amd" else None)
    )
    is_amd = inferred_card == "mi308x" or str(other.get("platform") or "").lower() == "amd"

    errors: List[str] = []
    proc_names: Dict[int, str] = {}
    proc_sort: Dict[int, int] = {}
    by_track_x: Dict[Tuple[int, int], List[Tuple[int, int, str]]] = defaultdict(list)
    sw_track_spans_by_pid: Dict[int, Dict[int, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    hw_counters: Dict[str, List[float]] = defaultdict(list)
    hw_pids = set()
    sw_sm_pids = set()
    n_gemm_slices = 0
    n_inner_k_slices = 0

    for ev in events:
        if not isinstance(ev, dict):
            continue
        ph = ev.get("ph")
        pid = ev.get("pid")
        tid = ev.get("tid")
        if ph == "M" and isinstance(pid, int):
            nm = ev.get("name")
            args = ev.get("args") or {}
            if nm == "process_name":
                proc_names[pid] = str(args.get("name") or "")
            elif nm == "process_sort_index" and isinstance(args.get("sort_index"), (int, float)):
                proc_sort[pid] = int(args["sort_index"])
        elif ph == "X" and isinstance(pid, int):
            cat = str(ev.get("cat") or "")
            nm_x = str(ev.get("name") or "")
            eff_tid = pid if (tid is None or tid == 0) else int(tid)
            ts_ns = int(round(float(ev.get("ts") or 0.0) * 1000.0))
            dur_ns = int(round(float(ev.get("dur") or 0.0) * 1000.0))
            if dur_ns < 0:
                errors.append(f"G1 negative duration slice on pid={pid} tid={tid}: dur_ns={dur_ns}")
            by_track_x[(pid, eff_tid)].append((ts_ns, ts_ns + dur_ns, nm_x))
            if cat not in ("info", "nsys_kernel", "hardware.dispatch", "software.kernel_launch", "software.phase", "software.kernel"):
                if tid == 0:
                    errors.append(f"G1 software X event uses forbidden tid=0 on pid={pid}")
                sw_sm_pids.add(pid)
                sw_track_spans_by_pid[pid][eff_tid].append((ts_ns, ts_ns + dur_ns))
                if nm_x == "gemm" or cat == "gemm":
                    n_gemm_slices += 1
                elif nm_x in ("load", "wait", "mma", "barrier") or cat in ("load", "wait", "mma", "barrier"):
                    n_inner_k_slices += 1
        elif ph == "C" and isinstance(pid, int):
            nm = str(ev.get("name") or "")
            args = ev.get("args") or {}
            for k, v in args.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    full_nm = nm if k == "value" else f"{nm}.{k}"
                    hw_counters[full_nm].append(float(v))
                    hw_pids.add(pid)

    # G1: check partial overlaps on every (pid, eff_tid) track
    spill_count = 0
    for _tk, intervals in by_track_x.items():
        intervals.sort(key=lambda r: (r[0], -r[1]))
        stack: List[Tuple[int, int]] = []
        for s_ns, e_ns, _nm in intervals:
            while stack and stack[-1][1] <= s_ns:
                stack.pop()
            if stack and e_ns > stack[-1][1]:
                spill_count += 1
            stack.append((s_ns, e_ns))
    if spill_count > 0:
        errors.append(f"G1 partial slice overlap (slice_spill) count={spill_count}")
    if (require_hw or "fi-l2-s0" in path.name) and n_gemm_slices >= 16:
        if n_inner_k_slices < n_gemm_slices * 8:
            errors.append(
                f"G1 sparse K-tile coverage in full/sw_hw trace: n_inner_k_slices={n_inner_k_slices} "
                f"for n_gemm_slices={n_gemm_slices} (appears to be sample1 kt=0 instead of full K-loop)"
            )

    # G2 & G3: hardware utilization [0-100%] & topmost pinning (sort_index AND R6 pid < sw_pid)
    if require_hw and not hw_counters:
        errors.append("G2 --require-hw set, but trace has 0 hardware counter tracks")
    if hw_counters:
        for cname, vals in hw_counters.items():
            lo, hi = min(vals), max(vals)
            if lo < -0.5 or hi > 105.0:
                errors.append(f"G2 hardware track {cname!r} outside [0, 100]% range: min={lo}, max={hi}")
        lower_names = [n.lower() for n in hw_counters]
        if require_hw:
            if not any(any(k in ln for k in COMPUTE_KEYWORDS) for ln in lower_names):
                errors.append(f"G2 missing compute utilization [%] track in {sorted(hw_counters)}")
            if not any(any(k in ln for k in MEMORY_KEYWORDS) for ln in lower_names):
                errors.append(f"G2 missing memory/DRAM bandwidth utilization [%] track in {sorted(hw_counters)}")
        if sw_sm_pids:
            min_sw_sort = min(proc_sort.get(p, p) for p in sw_sm_pids)
            max_hw_sort = max(proc_sort.get(p, p) for p in hw_pids)
            if not (max_hw_sort < min_sw_sort):
                errors.append(
                    f"G3 hardware process sort_index ({max_hw_sort}) is not strictly above "
                    f"SM/CU software tracks ({min_sw_sort})"
                )
            if not (max(hw_pids) < min(sw_sm_pids)):
                errors.append(
                    f"G3 R6 violation: hardware pid ({max(hw_pids)}) must be strictly < "
                    f"software SM/CU pid ({min(sw_sm_pids)}) so Perfetto UI pins hardware at Row #1"
                )

    # G4: physical SM/CU occupancy ceiling (sweep-line simultaneously active tracks)
    if sw_track_spans_by_pid and other.get("group") in ("cu", "sm_resident", "sm"):
        gran = int(other.get("gran", 1))
        max_active = max(_max_concurrent_tracks(ts_map) for ts_map in sw_track_spans_by_pid.values())
        if gran == 1:
            hw_slot_cap = (32 // max(1, waves_or_warps_per_cta)) if is_amd else (64 // max(1, waves_or_warps_per_cta))
            arch_wg_cap = 16 if is_amd else 32
            ceil_slots = min(arch_wg_cap, hw_slot_cap)
            if max_active > ceil_slots:
                errors.append(
                    f"G4 physical occupancy violation: max concurrent CTA tracks per {'CU' if is_amd else 'SM'} "
                    f"is {max_active} > hardware ceiling {ceil_slots} (waves_or_warps_per_cta={waves_or_warps_per_cta})"
                )
        else:
            ceil_warps = 32 if is_amd else 64
            if max_active > ceil_warps:
                errors.append(
                    f"G4 physical warp/wave occupancy violation: max concurrent warp tracks per {'CU' if is_amd else 'SM'} "
                    f"is {max_active} > hardware ceiling {ceil_warps}"
                )

    return {
        "format": "chrome_json",
        "errors": errors,
        "hw_pids": sorted(hw_pids),
        "min_sw_pid": min(sw_sm_pids) if sw_sm_pids else None,
        "hw_tracks": {k: {"n": len(v), "min": min(v), "max": max(v)} for k, v in sorted(hw_counters.items())},
        "spill_count": spill_count,
    }


def audit_native_proto(
    path: Path,
    *,
    require_hw: bool = False,
) -> Dict[str, Any]:
    from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as pb

    tr = pb.Trace()
    tr.ParseFromString(_read_bytes(path))
    errors: List[str] = []

    seq_last_ts: Dict[int, int] = {}
    ooo_count = 0
    procs: Dict[int, Tuple[str, int, int]] = {}
    counter_meta: Dict[int, Tuple[str, int, str]] = {}
    counter_vals: Dict[int, List[float]] = defaultdict(list)
    slice_begins = 0

    for pkt in tr.packet:
        if pkt.HasField("track_descriptor"):
            td = pkt.track_descriptor
            if td.HasField("process"):
                procs[td.uuid] = (
                    td.process.process_name or td.name,
                    int(td.sibling_order_rank),
                    int(td.process.pid),
                )
            if td.HasField("counter"):
                counter_meta[td.uuid] = (td.name, int(td.parent_uuid), td.counter.unit_name or "")
        if pkt.HasField("timestamp"):
            seq = int(pkt.trusted_packet_sequence_id)
            ts = int(pkt.timestamp)
            if seq in seq_last_ts and ts < seq_last_ts[seq]:
                ooo_count += 1
            seq_last_ts[seq] = ts
        if pkt.HasField("track_event"):
            te = pkt.track_event
            if te.type == pb.TrackEvent.TYPE_SLICE_BEGIN:
                slice_begins += 1
            elif te.type == pb.TrackEvent.TYPE_COUNTER:
                val = float(te.double_counter_value) if te.HasField("double_counter_value") else float(te.counter_value)
                counter_vals[int(te.track_uuid)].append(val)

    if ooo_count > 0:
        errors.append(f"G1 out-of-order timestamped TracePackets on sequence: count={ooo_count}")
    if slice_begins == 0:
        errors.append("G1 native protobuf has 0 TYPE_SLICE_BEGIN events")

    hw_proc_uuids = {u for u, (nm, _, _) in procs.items() if "硬件" in nm or "hardware" in nm.lower() or "gpu metrics" in nm.lower()}
    sm_proc_uuids = {u for u, (nm, _, _) in procs.items() if u not in hw_proc_uuids and ("SM " in nm or "CU " in nm)}

    hw_tracks_summary: Dict[str, Dict[str, Any]] = {}
    for tuuid, (tname, puuid, unit) in counter_meta.items():
        if puuid not in hw_proc_uuids or tname.startswith("communication_util"):
            continue
        vals = counter_vals.get(tuuid, [])
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        hw_tracks_summary[tname] = {"n": len(vals), "min": lo, "max": hi, "unit": unit}
        if lo < -0.5 or hi > 105.0:
            errors.append(f"G2 native hardware track {tname!r} outside [0, 100]%: min={lo}, max={hi}")

    if require_hw:
        if not hw_tracks_summary:
            errors.append("G2 --require-hw set, but native trace has 0 populated hardware utilization tracks")
        else:
            lnames = [n.lower() for n in hw_tracks_summary]
            if not any(any(k in ln for k in COMPUTE_KEYWORDS) for ln in lnames):
                errors.append(f"G2 native trace missing compute utilization [%] track in {sorted(hw_tracks_summary)}")
            if not any(any(k in ln for k in MEMORY_KEYWORDS) for ln in lnames):
                errors.append(f"G2 native trace missing memory/DRAM utilization [%] track in {sorted(hw_tracks_summary)}")

    if hw_proc_uuids and sm_proc_uuids:
        max_hw_rank = max(procs[u][1] for u in hw_proc_uuids)
        min_sm_rank = min(procs[u][1] for u in sm_proc_uuids)
        if not (max_hw_rank < min_sm_rank):
            errors.append(f"G3 native hardware sibling_order_rank ({max_hw_rank}) not above SM/CU ({min_sm_rank})")
        max_hw_pid = max(procs[u][2] for u in hw_proc_uuids)
        min_sm_pid = min(procs[u][2] for u in sm_proc_uuids)
        if not (max_hw_pid < min_sm_pid):
            errors.append(f"G3 native hardware pid ({max_hw_pid}) not strictly < SM/CU pid ({min_sm_pid})")

    return {
        "format": "perfetto_native",
        "errors": errors,
        "ooo_packets": ooo_count,
        "slice_begins": slice_begins,
        "hw_tracks": hw_tracks_summary,
    }


def _find_local_trace_processor_bin() -> Optional[str]:
    """Locate an already-cached local trace_processor_shell binary without network calls."""
    env_bin = os.environ.get("WEAVEPROFILER_TPS_BIN", "").strip()
    if env_bin and Path(env_bin).is_file():
        return env_bin
    for cmd_name in ("trace_processor_shell", "trace_processor"):
        found = shutil.which(cmd_name)
        if found:
            return found
    candidates: List[Path] = []
    for base in (
        Path.home() / ".local" / "share" / "perfetto" / "prebuilts",
        Path(os.environ.get("LOCALAPPDATA", "") or ".") / "perfetto" / "prebuilts",
        Path(os.environ.get("APPDATA", "") or ".") / "perfetto" / "prebuilts",
        Path("/tmp/weaveprofiler_tps_probe"),
    ):
        if base.is_dir():
            for p in sorted(base.rglob("*"), reverse=True):
                if p.is_file() and (
                    p.name.startswith("trace_processor_shell")
                    or p.name in ("v47", "trace_processor", "trace_processor.exe")
                ):
                    candidates.append(p)
    return str(candidates[0]) if candidates else None


def audit_with_trace_processor(path: Path, *, require_hw: bool = False) -> List[str]:
    try:
        from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig
    except ImportError:
        return []
    local_bin = _find_local_trace_processor_bin()
    if not local_bin:
        # Never block on air-gapped hosts trying to reach commondatastorage.googleapis.com
        return []
    tp = TraceProcessor(trace=str(path), config=TraceProcessorConfig(bin_path=local_bin))
    try:
        bad_stats = [
            f"{r.name}({r.severity})={r.value}"
            for r in tp.query("SELECT name, severity, value FROM stats WHERE value > 0 AND severity != 'info'")
        ]
        n_slices = next(iter(tp.query("SELECT COUNT(*) AS c FROM slice"))).c
        upid_err = None
        if require_hw:
            proc_rows = list(tp.query("SELECT upid, pid, name FROM process WHERE pid > 0 ORDER BY upid"))
            hw_upids = [
                r.upid for r in proc_rows
                if r.name and ("硬件" in r.name or "hardware" in r.name.lower() or "rocprof-sys" in r.name.lower() or "gpu metrics" in r.name.lower())
            ]
            if proc_rows and hw_upids:
                min_upid = proc_rows[0].upid
                if min(hw_upids) != min_upid:
                    upid_err = (
                        f"G3 TraceProcessor upid ordering violation: hardware upid={min(hw_upids)} "
                        f"is not first process (min upid={min_upid}, pid={proc_rows[0].pid}, name={proc_rows[0].name!r})"
                    )
    finally:
        tp.close()
    errs = []
    if bad_stats:
        errs.append("G1 TraceProcessor import errors: " + ", ".join(bad_stats))
    if n_slices <= 0:
        errs.append("G1 TraceProcessor imported 0 slices")
    if upid_err:
        errs.append(upid_err)
    return errs


def verify_trace(
    path: Path,
    *,
    require_hw: bool = False,
    card: Optional[str] = None,
    waves_or_warps_per_cta: int = 4,
    use_tp: bool = True,
) -> Dict[str, Any]:
    if _is_native_proto(path):
        res = audit_native_proto(path, require_hw=require_hw)
    else:
        res = audit_json_trace(path, require_hw=require_hw, card=card, waves_or_warps_per_cta=waves_or_warps_per_cta)
    if use_tp:
        res["errors"].extend(audit_with_trace_processor(path, require_hw=require_hw))
    res["ok"] = len(res["errors"]) == 0
    res["path"] = str(path)
    return res


def selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="fi_trace_gate_st_") as td:
        root = Path(td)
        # 1. Valid SW+HW JSON trace (pid=1, upid=1, sort_index=-1000, software pid >= 100) -> MUST PASS
        good = root / "good.perfetto.json"
        good.write_text(
            json.dumps({
                "traceEvents": [
                    {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "00 · 硬件信息与利用率 [0–100%]"}},
                    {"ph": "M", "pid": 1, "name": "process_sort_index", "args": {"sort_index": -1000}},
                    {"ph": "X", "pid": 1, "tid": 1, "ts": 0.0, "dur": 10.0, "name": "GPU HW window", "cat": "hardware.dispatch"},
                    {"ph": "C", "pid": 1, "tid": 2, "ts": 0.0, "name": "compute_util [matrix_core]", "args": {"value": 35.2}},
                    {"ph": "C", "pid": 1, "tid": 3, "ts": 0.0, "name": "memory_util [HBM request estimate]", "args": {"value": 48.5}},
                    {"ph": "M", "pid": 104, "name": "process_name", "args": {"name": "CU4"}},
                    {"ph": "M", "pid": 104, "name": "process_sort_index", "args": {"sort_index": 104}},
                    {"ph": "X", "pid": 104, "tid": 1, "ts": 0.0, "dur": 10.0, "name": "gemm", "cat": "gemm"},
                    {"ph": "X", "pid": 104, "tid": 1, "ts": 1.0, "dur": 4.0, "name": "load", "cat": "load"},
                ],
                "otherData": {"platform": "amd", "gran": 1, "group": "cu"},
            }),
            encoding="utf-8",
        )
        r_good = verify_trace(good, require_hw=True, card="mi308x", waves_or_warps_per_cta=4)
        assert r_good["ok"], f"expected good trace to pass: {r_good}"

        # 2. Broken trace with raw cycle count (8825899), buried pid=900000/sort_index=5, tid=0 collision, and 16 bands -> MUST FAIL ALL 4 GATES!
        bad_evs = [
            {"ph": "M", "pid": 900000, "name": "process_name", "args": {"name": "HW"}},
            {"ph": "M", "pid": 900000, "name": "process_sort_index", "args": {"sort_index": 5}},
            {"ph": "C", "pid": 900000, "tid": 1, "ts": 0.0, "name": "GRBM_GUI_ACTIVE [cycles]", "args": {"value": 8825899.0}},
            {"ph": "M", "pid": 4, "name": "process_name", "args": {"name": "CU4"}},
            {"ph": "M", "pid": 4, "name": "process_sort_index", "args": {"sort_index": 0}},
            {"ph": "X", "pid": 4, "tid": 0, "ts": 0.0, "dur": 10.0, "name": "gemm", "cat": "gemm"},
            {"ph": "X", "pid": 4, "tid": 4, "ts": 5.0, "dur": 10.0, "name": "load", "cat": "load"},
        ]
        for b in range(1, 17):
            bad_evs.append({"ph": "X", "pid": 4, "tid": b, "ts": 0.0, "dur": 2.0, "name": "mma", "cat": "mma"})
        bad = root / "bad.perfetto.json"
        bad.write_text(json.dumps({"traceEvents": bad_evs, "otherData": {"platform": "amd", "gran": 1, "group": "cu"}}), encoding="utf-8")
        r_bad = verify_trace(bad, require_hw=True, card="mi308x", waves_or_warps_per_cta=4)
        assert not r_bad["ok"], "expected bad trace to fail"
        joined = "\n".join(r_bad["errors"])
        for gate_tag in ("G1", "G2", "G3", "G4"):
            assert gate_tag in joined, f"missing {gate_tag} failure in: {joined}"
    print("FI_TRACE_GATE SELFTEST PASS (G1/G2/G3/G4 all verified)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fail-closed Perfetto trace correctness gate")
    ap.add_argument("traces", nargs="*", help="Perfetto .json[.gz] or .perfetto-trace[.gz] files")
    ap.add_argument("--require-hw", action="store_true", help="Require [0-100]%% compute and memory hardware tracks")
    ap.add_argument("--card", choices=("h20", "b300", "mi308x"), default=None)
    ap.add_argument("--waves-per-cta", type=int, default=4)
    ap.add_argument("--no-tp", action="store_true", help="Skip TraceProcessor SQL stats check")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.traces:
        ap.error("provide at least one trace path or --selftest")
    all_ok = True
    results = []
    for raw_p in args.traces:
        res = verify_trace(
            Path(raw_p),
            require_hw=args.require_hw,
            card=args.card,
            waves_or_warps_per_cta=args.waves_per_cta,
            use_tp=not args.no_tp,
        )
        results.append(res)
        if not res["ok"]:
            all_ok = False
    print(json.dumps(results if len(results) > 1 else results[0], indent=2, ensure_ascii=False))
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
