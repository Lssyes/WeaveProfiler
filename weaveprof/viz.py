"""Publication-style single-run 3-panel SW+HW overview renderer for WeaveProfiler traces.

Renders any WeaveProfiler ``trace.perfetto.json[.gz]`` (or in-memory trace dict) across
NVIDIA (H20 / B300, SM + warp, nsys) and AMD (MI308X, CU + wave, rocprofiler-sdk / rocprof)
into a single high-DPI PNG with one locked microsecond time axis across all three panels:
  (A) Same-run real hardware utilization curves (Compute / HBM / Intra-Node NVLink or xGMI /
      optional 5th Inter-Node RDMA NIC curve or explicit unavailable HCA footnote / Active SM or CU)
  (B) All physical SMs or CUs dense multi-stage / dual-fabric waterfall
  (C) Representative SM 0 / CU 0 per-warp / per-wave role timeline (preserving rich domain stage
      labels and crisp >= 2 px visibility for short streaming Combine / Reduce / RDMA slices)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

STAGE_COLORS: Dict[str, Tuple[str, str]] = {
    "load": ("#2E7D32", "TMA / Tile Load"),
    "mma": ("#1565C0", "Tensor Core MMA / WGMMA"),
    "softmax": ("#00ACC1", "Online Softmax"),
    "compute": ("#42A5F5", "Compute / ALU"),
    "epilogue": ("#FB8C00", "Epilogue"),
    "store": ("#D84315", "Store"),
    "wait": ("#90A4AE", "Wait / Sync"),
    "barrier": ("#607D8B", "Route / Barrier Sync"),
    "communication": ("#8E24AA", "Communication"),
    "reduction": ("#C2185B", "Combine / Reduce"),
    "dispatch": ("#00838F", "Dispatch / Schedule"),
    "gemm": ("#1E88E5", "GEMM / MMA"),
    "attention": ("#1976D2", "Attention"),
    "moe": ("#6A1B9A", "MoE"),
}
# Globally disjoint sub-palette shades used when multiple distinct named stages share one semantic
# family (e.g., `mma_qk` vs `mma_pv`, `gmm1` vs `gmm2`, `dispatch` vs `combine`), so distinct code
# segments stay visually distinguishable with zero cross-family color collisions.
_SEMANTIC_SHADES: Dict[str, Tuple[str, ...]] = {
    "mma": ("#1565C0", "#0D47A1", "#283593", "#0277BD", "#3F51B5", "#01579B"),
    "gemm": ("#1E88E5", "#1976D2", "#2196F3", "#039BE5"),
    "load": ("#2E7D32", "#43A047", "#558B2F", "#1B5E20", "#689F38", "#33691E"),
    "compute": ("#42A5F5", "#29B6F6", "#5C6BC0", "#64B5F6", "#7986CB"),
    "softmax": ("#00ACC1", "#26C6DA", "#0097A7", "#4DD0E1"),
    "epilogue": ("#FB8C00", "#FFA726", "#F57C00", "#FFB74D"),
    "store": ("#D84315", "#E64A19", "#BF360C", "#FF7043"),
    "communication": ("#8E24AA", "#AB47BC", "#7B1FA2", "#9C27B0"),
    "reduction": ("#C2185B", "#D81B60", "#AD1457", "#E91E63"),
    "dispatch": ("#00838F", "#00897B", "#00695C", "#26A69A"),
    "wait": ("#90A4AE", "#B0BEC5", "#A1B2BA"),
    "barrier": ("#607D8B", "#78909C", "#455A64"),
}
_OVERFLOW_SHADES: Tuple[str, ...] = (
    "#8D6E63", "#5D4037", "#AFB42B", "#827717", "#FBC02D", "#F9A825", "#EC407A", "#7E57C2",
)
_DEFAULT_COLOR = ("#37474F", "Other Stage")
_STAGE_DRAW_PRIORITY = {
    "wait": 0, "barrier": 1, "load": 2, "dispatch": 3, "compute": 4,
    "softmax": 5, "epilogue": 6, "store": 6, "communication": 7, "reduction": 7,
    "mma": 8, "gemm": 8,
}

# Dual-color Inter-Node vs Intra-Node Fabric Palette (NVIDIA IB/RDMA vs NVLink, AMD RDMA vs xGMI).
# Warm crimson / amber is strictly reserved for Inter-Node RDMA / IB; cool teal / royal blue for
# Intra-Node xGMI / NVLink; emerald / forest green for GEMM1 / GEMM2; slate for Route / Sync;
# light violet for on-GPU Local Stage prep. Min RGB channel separation >= 60 across all pairs.
FABRIC_COLORS_BY_ROLE: Dict[str, str] = {
    "inter_dispatch": "#D62728",  # crimson red
    "inter_combine": "#F28E2B",   # warm amber / orange
    "intra_dispatch": "#009FB7",  # vivid teal / cyan
    "intra_combine": "#2563EB",   # royal blue
    "gemm1": "#16A34A",           # emerald green
    "gemm2": "#166534",           # deep forest green
    "route_sync": "#94A3B8",      # cool slate gray
    "local_prep": "#8B5CF6",      # light violet
}

_FABRIC_WAIT_DROP = frozenset({
    "gemm1_wait_payload", "gemm2_wait_stage1", "post_plan_wait",
    "scatter_ready_wait", "dispatch_wait_plan", "home_ready_wait",
    "gemm1_wait_route", "route_wait_meta",
    "internode_rdma_rxwatch", "rdma_wqe_cqe_service", "legacy_gda_service",
    "disp_idle_wait",
})
_GENERIC_LABELS = frozenset(STAGE_COLORS.keys()) | {"other", "stage", "unknown", ""}
_INTER_SLICE_RE = re.compile(
    r"(?:^|[_\-\s])(?:inter[_\-\s]?node|internode|rdma|roce|infiniband|ibgda|remote[_\-\s]?node)",
    re.I,
)
_INTRA_SLICE_RE = re.compile(
    r"(?:^|[_\-\s])(?:intra[_\-\s]?node|intranode|xgmi|nvlink|home[_\-]?combine|scatter[_\-]?xgmi|dispatch[_\-]?xgmi|p2p)",
    re.I,
)
_COMBINE_RE = re.compile(r"(?:combine|reduce|return|scatter|push|store|epilogue)", re.I)
_DISPATCH_RE = re.compile(r"(?:dispatch|pull|alltoall|a2a|send|recv|load)", re.I)
_GEMM1_RE = re.compile(r"(?:gemm1|gmm1|fc1|swiglu|gate_up|l1_gemm|mma_qk|qk_mma)", re.I)
_GEMM2_RE = re.compile(r"(?:gemm2|gmm2|fc2|down_proj|l2_gemm|mma_pv|pv_mma)", re.I)
_LOCAL_PREP_RE = re.compile(r"(?:local[_\-\s]?stage|local[_\-\s]?prep|quant[_\-\s]?prep)", re.I)
_ROUTE_SYNC_RE = re.compile(r"(?:route|ballot|plan|sync|barrier|meta|cleanup)", re.I)


def _short_kernel_name(raw: str, max_len: int = 72) -> str:
    s = str(raw or "").strip()
    if not s:
        return "WeaveProfiler Kernel"
    s = s.split("(")[0].strip()
    if s.startswith("void "):
        s = s[5:].strip()
    if "<" in s and len(s) > max_len:
        idx = s.find("<")
        prefix = s[:idx]
        suffix = s[s.rfind(">") + 1:] if ">" in s else ""
        s = f"{prefix}<...>{suffix}"
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


def _pack_concurrent_ctas(spans: Sequence[Mapping[str, Any]]) -> Tuple[Dict[int, int], int]:
    """Pack overlapping CTAs on one SM/CU into the minimal number of non-overlapping resident slots."""
    cta_win: Dict[int, List[float]] = {}
    for sp in spans:
        c = int(sp["cta"])
        w = cta_win.get(c)
        if w is None:
            cta_win[c] = [float(sp["ts"]), float(sp["end"])]
        else:
            if sp["ts"] < w[0]:
                w[0] = float(sp["ts"])
            if sp["end"] > w[1]:
                w[1] = float(sp["end"])
    ordered = sorted(cta_win.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[0]))
    slot_ends: List[float] = []
    cta_to_slot: Dict[int, int] = {}
    for cta, (t0, t1) in ordered:
        chosen = None
        for idx, end_t in enumerate(slot_ends):
            if t0 >= end_t - 1e-6:
                chosen = idx
                slot_ends[idx] = t1
                break
        if chosen is None:
            chosen = len(slot_ends)
            slot_ends.append(t1)
        cta_to_slot[cta] = chosen
    return cta_to_slot, max(1, len(slot_ends))


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _load_trace(trace_or_path: Mapping[str, Any] | str | Path) -> Dict[str, Any]:
    if isinstance(trace_or_path, Mapping):
        return dict(trace_or_path)
    p = Path(trace_or_path)
    raw = p.read_bytes()
    if p.suffix == ".gz" or raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def _classify_hw_metric(name: str) -> Optional[str]:
    low = name.lower()
    if "rdma_util" in low or "inter-node rdma" in low or "infiniband" in low or "roce" in low:
        return "inter_rdma"
    if "tensor" in low or "compute_util" in low or "matrix core" in low or "mfma" in low:
        return "tensor_active"
    if "sms active" in low or "sm active" in low or "sm_active" in low or "active mi308x cu" in low or ("active" in low and "cu" in low):
        return "sm_active"
    if "sm issue" in low or "warp issue" in low or "issue" in low:
        return "sm_issue"
    if "memory_util" in low or ("hbm" in low and "read" not in low and "write" not in low):
        return "dram_bw_direct"
    if "dram read" in low or ("hbm" in low and "read" in low):
        return "dram_read"
    if "dram write" in low or ("hbm" in low and "write" in low):
        return "dram_write"
    if "communication_util" in low or "xgmi" in low:
        return "intra_comm_direct"
    if "nvlink" in low:
        if "protocol" in low:
            return None
        if "rx" in low and "request" in low:
            return "nvlink_rx_req"
        if "rx" in low and "response" in low:
            return "nvlink_rx_rsp"
        if "tx" in low and "request" in low:
            return "nvlink_tx_req"
        if "tx" in low and "response" in low:
            return "nvlink_tx_rsp"
        if "rx" in low:
            return "nvlink_rx_total"
        if "tx" in low:
            return "nvlink_tx_total"
        return "nvlink_user"
    return None


def _pretty_hw_curve_label(raw_name: str, role: str, *, is_amd: bool, multi_node: bool) -> Tuple[str, str]:
    """Turn a raw Perfetto counter track name into a publication-grade legend label + short chip title."""
    m = re.match(r"^[a-z0-9_]+\s*\[(.+)\]\s*$", str(raw_name or "").strip(), re.I)
    bracket = m.group(1).strip() if m else ""
    if role == "tensor_active":
        if bracket:
            return f"Compute — {bracket}", "Compute"
        return ("Compute — Matrix Core / MFMA %" if is_amd else "Tensor Active / MMA Pipe %"), "Compute"
    if role == "dram_bw":
        if bracket:
            return f"HBM — {bracket}", "HBM BW"
        return "HBM / DRAM Bandwidth (R+W) %", "HBM BW"
    if role == "nvlink_tx":
        prefix = "Intra-Node NVLink TX" if multi_node else "NVLink TX"
        return f"{prefix} (Egress User Data) %", prefix
    if role == "nvlink_rx":
        prefix = "Intra-Node NVLink RX" if multi_node else "NVLink RX"
        return f"{prefix} (Ingress User Data) %", prefix
    if role == "nvlink_bw":
        fabric_nm = "xGMI" if is_amd else "NVLink"
        prefix = f"Intra-Node {fabric_nm}" if multi_node else fabric_nm
        if bracket:
            b_clean = re.sub(r"^(?:Intra-Node\s+)?(?:xGMI|NVLink\d*)\s*", "", bracket, flags=re.I).strip()
            return (f"{prefix} — {b_clean}" if b_clean else prefix), prefix
        return f"{prefix} — P2P / User Data %", prefix
    if role == "inter_rdma":
        if bracket:
            b_clean = re.sub(r"^Inter-Node\s+RDMA\s*(?:\(NIC\)|NIC)?\s*", "", bracket, flags=re.I).strip()
            return (f"Inter-Node RDMA (NIC) — {b_clean}" if b_clean else "Inter-Node RDMA (NIC) %"), "Inter-Node RDMA"
        return "Inter-Node RDMA (NIC) %", "Inter-Node RDMA"
    if role == "sm_active":
        unit_pl = "CUs" if is_amd else "SMs"
        if bracket:
            m_of = re.search(r"%\s*of\s*\d+", bracket, re.I)
            if m_of:
                return f"Active — {unit_pl} {m_of.group(0)}", f"Active {unit_pl}"
            return f"Active — {bracket}", f"Active {unit_pl}"
        return f"Active {unit_pl} %", f"Active {unit_pl}"
    if role == "sm_issue":
        return "SM Issue %", "SM Issue"
    return str(raw_name), role


def _classify_fabric_role(sp: Mapping[str, Any], *, is_amd: bool) -> Optional[str]:
    """Classify a span into a dual-fabric display role when rendering multi-node traces."""
    raw = str(sp.get("label") or sp.get("raw_name") or sp.get("sem") or "").strip()
    low = raw.lower()
    fab = str(sp.get("fabric") or "").strip().lower()
    sem = str(sp.get("sem") or "").lower()

    if fab.startswith("inter_node") or _INTER_SLICE_RE.search(raw):
        if _COMBINE_RE.search(low) or sem in ("reduction", "store", "epilogue"):
            return "inter_combine"
        return "inter_dispatch"
    if _LOCAL_PREP_RE.search(low):
        return "local_prep"
    if fab.startswith("intra_node") or _INTRA_SLICE_RE.search(raw):
        if _COMBINE_RE.search(low) or sem in ("reduction", "store", "epilogue"):
            return "intra_combine"
        return "intra_dispatch"
    if _GEMM1_RE.search(low):
        return "gemm1"
    if _GEMM2_RE.search(low):
        return "gemm2"
    if _ROUTE_SYNC_RE.search(low) or sem in ("barrier", "wait"):
        return "route_sync"
    if sem in ("mma", "gemm"):
        return "gemm1"
    if sem == "reduction":
        return "intra_combine"
    if sem in ("communication", "dispatch"):
        return "intra_dispatch"
    if sem == "load":
        return "local_prep"
    return None


def _fabric_role_label(role: str, *, is_amd: bool, use_ib: bool = False) -> str:
    intra_nm = "xGMI" if is_amd else "NVLink"
    inter_nm = "IB/RDMA" if use_ib else "RDMA"
    return {
        "inter_dispatch": f"Inter-Node {inter_nm} Dispatch",
        "inter_combine": f"Inter-Node {inter_nm} Combine",
        "intra_dispatch": f"Intra-Node {intra_nm} Dispatch",
        "intra_combine": f"Intra-Node {intra_nm} Combine",
        "gemm1": "GEMM1 (FC1+SwiGLU)",
        "gemm2": "GEMM2 (FC2)",
        "route_sync": "Route / Sync",
        "local_prep": "Local Stage (prep)",
    }[role]


def _humanize_stage_label(raw_label: str, sem: str) -> str:
    """Preserve meaningful domain/stage names instead of collapsing them to bare `load` / `mma`."""
    lbl = str(raw_label or "").strip()
    low = lbl.lower()
    if not lbl or low in _GENERIC_LABELS:
        return STAGE_COLORS.get(sem, _DEFAULT_COLOR)[1]
    # Already a richly formatted human display label (contains spaces, parens, arrows, or dashes)
    if any(ch in lbl for ch in (" ", "(", "/", "—", "->")):
        return lbl
    # Known rich compound stage tokens
    pretty_known = {
        "compute_gemm1_swiglu": "GEMM1 (FC1+SwiGLU)",
        "gemm1_tile_swiglu": "GEMM1 (FC1+SwiGLU)",
        "compute_gemm2_mfma": "GEMM2 (FC2)",
        "gemm2_tile_mfma": "GEMM2 (FC2)",
        "intranode_localstage": "Local Stage (prep)",
        "intranode_xgmi_pull": "Intra-Node xGMI Dispatch",
        "intranode_xgmi_scatterpush": "Intra-Node xGMI Combine",
        "intranode_homecombine": "Intra-Node xGMI Combine",
        "internode_rdma_dispatch": "Inter-Node RDMA Dispatch",
        "internode_rdma_return": "Inter-Node RDMA Combine",
        "internode_partial_reduce": "Inter-Node RDMA Combine",
        "route_plan_local": "Route / Sync",
        "route_plan_remote": "Route / Sync",
    }
    if low in pretty_known:
        return pretty_known[low]
    return f"{lbl} ({sem})" if low != sem else STAGE_COLORS.get(sem, _DEFAULT_COLOR)[1]


def _detect_platform(other: Mapping[str, Any], events: Sequence[Mapping[str, Any]], hw_labels: Mapping[str, str]) -> bool:
    """Return True when the trace belongs to an AMD GPU (CU / wave / rocprof), else False (NVIDIA SM / warp)."""
    meta = other.get("meta") if isinstance(other.get("meta"), Mapping) else {}
    plat = str(other.get("platform") or meta.get("platform") or "").lower()
    if plat == "amd":
        return True
    if plat in ("nv", "nvidia"):
        return False
    card = str(other.get("card") or meta.get("card") or "").lower()
    arch = str(other.get("arch") or meta.get("arch") or "").lower()
    if "mi308" in card or "mi300" in card or arch.startswith("gfx"):
        return True
    if any("xgmi" in v.lower() or "mi308" in v.lower() or "matrix core" in v.lower() for v in hw_labels.values()):
        return True
    for ev in events[:120]:
        if ev.get("ph") == "M" and ev.get("name") == "process_name":
            nm = str((ev.get("args") or {}).get("name") or "")
            if re.match(r"^\s*CU\s*\d+", nm):
                return True
    return False


def _parse_trace(trace: Mapping[str, Any]) -> Dict[str, Any]:
    events = trace.get("traceEvents") or []
    other = dict(trace.get("otherData") or {})
    meta = other.get("meta") if isinstance(other.get("meta"), Mapping) else {}
    hw_curves: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    hw_labels: Dict[str, str] = {}
    nsys_kernel: Optional[Dict[str, Any]] = None
    info_dur_us: Optional[float] = None
    sw_spans: List[Dict[str, Any]] = []
    sm_set: set[int] = set()
    stages_seen: Dict[str, float] = defaultdict(float)
    meta_warps_by_sm: Dict[int, Dict[int, str]] = defaultdict(dict)
    proc_names: Dict[int, str] = {}
    has_high_unit_pids = any(
        ev.get("ph") == "X" and isinstance(ev.get("pid"), int) and int(ev["pid"]) >= 100
        and str(ev.get("cat") or "") not in ("nsys_kernel", "info")
        for ev in events
    )

    for ev in events:
        ph = ev.get("ph")
        if ph == "M" and ev.get("name") == "process_name":
            pid = ev.get("pid")
            if isinstance(pid, int):
                proc_names[pid] = str((ev.get("args") or {}).get("name") or "")
        elif ph == "M" and ev.get("name") == "thread_name":
            pid = ev.get("pid")
            tid = ev.get("tid")
            tname = str((ev.get("args") or {}).get("name") or "")
            if isinstance(pid, int) and pid != 1 and (not has_high_unit_pids or pid >= 100):
                sm = pid - 100 if pid >= 100 else pid
                m = re.search(r"\b[wW](\d+)\b", tname)
                if not m:
                    m = re.search(r"\bWave\s+(\d+)\b", tname, re.I)
                if m:
                    w_id = int(m.group(1))
                    m_role_bracket = re.search(r"\[(Compute|Comm|Dispatch|Scatter|Math|Epilogue|DMA|Producer|Consumer)[^\]]*\]", tname, re.I)
                    if m_role_bracket:
                        role_part = m_role_bracket.group(1).strip()
                    elif "·" in tname:
                        role_part = tname.split("·")[-1].strip()
                    elif "/" in tname:
                        role_part = tname.split("/")[-1].strip()
                    else:
                        role_part = f"w{w_id:02d}"
                    if re.fullmatch(r"[wW]\d+", role_part):
                        role_part = f"w{w_id:02d}"
                    meta_warps_by_sm[sm].setdefault(w_id, role_part)
                    sm_set.add(sm)
        elif ph == "C":
            cname = str(ev.get("name") or "")
            key = _classify_hw_metric(cname)
            if key is None:
                continue
            ts = float(ev.get("ts") or 0.0)
            args = ev.get("args") or {}
            val = args.get("pct", args.get("value", 0.0))
            hw_curves[key].append((ts, float(val)))
            hw_labels.setdefault(key, cname)
        elif ph == "X":
            cat = str(ev.get("cat") or "")
            pid = ev.get("pid")
            if cat == "nsys_kernel":
                nsys_kernel = dict(ev)
                continue
            if cat == "info":
                d_info = float(ev.get("dur") or 0.0)
                if d_info > 0.0 and info_dur_us is None:
                    info_dur_us = d_info
                continue
            # Skip non-unit macro summary tracks (e.g. pid=1 hardware header, pid=2 macro stage summary)
            if has_high_unit_pids and isinstance(pid, int) and pid < 100:
                continue
            ts = float(ev.get("ts") or 0.0)
            dur = float(ev.get("dur") or 0.0)
            if dur <= 0.0 or not math.isfinite(dur):
                continue
            args = ev.get("args") or {}
            raw_name = str(ev.get("name") or "").strip()
            disp_label = str(
                args.get("display_label")
                or args.get("original_label")
                or raw_name
                or args.get("semantic_id")
                or "compute"
            ).strip()
            sem_raw = str(args.get("semantic_id") or cat or raw_name or "compute").lower()
            if sem_raw not in STAGE_COLORS:
                from .contract import resolve_semantic_id
                sem = resolve_semantic_id(disp_label or raw_name or sem_raw)
            else:
                sem = sem_raw
            sm = args.get("sm")
            if sm is None and args.get("cu") is not None and isinstance(pid, int) and pid >= 100:
                sm = int(pid) - 100
            elif sm is None:
                sm = int(pid) - 100 if isinstance(pid, int) and pid >= 100 else (int(pid) if isinstance(pid, int) else 0)
            else:
                sm = int(sm)
            if args.get("warp") is not None:
                warp = int(args["warp"])
            elif args.get("wave") is not None:
                warp = int(args["wave"])
            elif isinstance(ev.get("tid"), int) and has_high_unit_pids and int(ev["tid"]) >= 1:
                warp = int(ev["tid"]) - 1 if (sm in meta_warps_by_sm and 0 in meta_warps_by_sm[sm]) else int(ev["tid"])
            else:
                warp = 0
            cta = int(args.get("cta", 0))
            role = str(args.get("role") or meta_warps_by_sm.get(sm, {}).get(warp) or f"w{warp:02d}")
            fabric = str(args.get("fabric") or "").strip().lower()
            sm_set.add(sm)
            stages_seen[sem] += dur
            sw_spans.append({
                "ts": ts, "dur": dur, "end": ts + dur, "sem": sem,
                "label": disp_label, "raw_name": raw_name, "fabric": fabric,
                "sm": sm, "warp": warp, "cta": cta, "role": role,
            })

    for k in hw_curves:
        hw_curves[k].sort(key=lambda x: x[0])

    # Combine DRAM read + write (shared half-duplex HBM bus) or use direct memory_util curve
    if "dram_bw_direct" in hw_curves:
        hw_curves["dram_bw"] = list(hw_curves["dram_bw_direct"])
        hw_labels["dram_bw"] = hw_labels.get("dram_bw_direct", "memory_util")
    elif "dram_read" in hw_curves or "dram_write" in hw_curves:
        by_t: Dict[float, float] = defaultdict(float)
        for k in ("dram_read", "dram_write"):
            for t, v in hw_curves.get(k, ()):
                by_t[round(t, 3)] += v
        hw_curves["dram_bw"] = [(t, min(100.0, v)) for t, v in sorted(by_t.items())]

    # NVLink TX (egress) and NVLink RX (ingress) are TWO independent full-duplex physical links:
    # sum Request + Response WITHIN each direction only; NEVER sum or merge TX with RX!
    if "nvlink_tx_total" in hw_curves:
        hw_curves["nvlink_tx"] = list(hw_curves["nvlink_tx_total"])
        hw_labels["nvlink_tx"] = hw_labels.get("nvlink_tx_total", "NVLink TX User Data Total")
    elif "nvlink_tx_req" in hw_curves or "nvlink_tx_rsp" in hw_curves:
        tx_t: Dict[float, float] = defaultdict(float)
        for k in ("nvlink_tx_req", "nvlink_tx_rsp"):
            for t, v in hw_curves.get(k, ()):
                tx_t[round(t, 3)] += v
        if tx_t:
            hw_curves["nvlink_tx"] = [(t, min(100.0, v)) for t, v in sorted(tx_t.items())]
            hw_labels["nvlink_tx"] = "NVLink TX User Data (Req+Rsp)"

    if "nvlink_rx_total" in hw_curves:
        hw_curves["nvlink_rx"] = list(hw_curves["nvlink_rx_total"])
        hw_labels["nvlink_rx"] = hw_labels.get("nvlink_rx_total", "NVLink RX User Data Total")
    elif "nvlink_rx_req" in hw_curves or "nvlink_rx_rsp" in hw_curves:
        rx_t: Dict[float, float] = defaultdict(float)
        for k in ("nvlink_rx_req", "nvlink_rx_rsp"):
            for t, v in hw_curves.get(k, ()):
                rx_t[round(t, 3)] += v
        if rx_t:
            hw_curves["nvlink_rx"] = [(t, min(100.0, v)) for t, v in sorted(rx_t.items())]
            hw_labels["nvlink_rx"] = "NVLink RX User Data (Req+Rsp)"

    # Single-series intra-node fabric curve (e.g. AMD xGMI P2P or unsplit generic counter)
    if "intra_comm_direct" in hw_curves:
        hw_curves["nvlink_bw"] = list(hw_curves["intra_comm_direct"])
        hw_labels["nvlink_bw"] = hw_labels.get("intra_comm_direct", "communication_util")
    elif "nvlink_user" in hw_curves and "nvlink_tx" not in hw_curves and "nvlink_rx" not in hw_curves:
        hw_curves["nvlink_bw"] = list(hw_curves["nvlink_user"])

    is_amd = _detect_platform(other, events, hw_labels)
    has_inter_slices = any(
        sp["fabric"].startswith("inter_node") or bool(_INTER_SLICE_RE.search(sp["label"]))
        for sp in sw_spans
    )
    has_intra_slices = any(
        sp["fabric"].startswith("intra_node") or bool(_INTRA_SLICE_RE.search(sp["label"]))
        for sp in sw_spans
    )
    ep_val = int(meta.get("ep") or other.get("ep") or 0)
    multi_node = bool(ep_val > 8 or meta.get("multi_node") or "inter_rdma" in hw_curves or has_inter_slices)
    dual_fabric = bool(has_inter_slices and (has_intra_slices or multi_node))
    use_ib = any(
        sp["fabric"] == "inter_node_ib" or bool(re.search(r"\b(?:infiniband|ibgda|ib_)\b", sp["label"], re.I))
        for sp in sw_spans
    )

    if dual_fabric:
        # Filter pure polling/background wait slices in multi-node fabric timelines so active fabric bars stay crisp.
        sw_spans = [sp for sp in sw_spans if sp["label"].lower() not in _FABRIC_WAIT_DROP]

    sw_end = max((s["end"] for s in sw_spans), default=0.0)
    k_dur = (
        float(nsys_kernel["dur"]) if (nsys_kernel and nsys_kernel.get("dur"))
        else (info_dur_us if info_dur_us else sw_end)
    )
    t_max = max(sw_end, k_dur, 1.0)

    return {
        "hw_curves": dict(hw_curves),
        "hw_labels": hw_labels,
        "nsys_kernel": nsys_kernel,
        "sw_spans": sw_spans,
        "sms": sorted(sm_set),
        "meta_warps_by_sm": {k: dict(v) for k, v in meta_warps_by_sm.items()},
        "stages_seen": dict(stages_seen),
        "t_max": t_max,
        "k_dur": k_dur,
        "sw_end": sw_end,
        "other": other,
        "is_amd": is_amd,
        "multi_node": multi_node,
        "dual_fabric": dual_fabric,
        "use_ib": use_ib,
        "ep": ep_val,
    }


def _compute_panel_c_role_labels(
    rep_warps: Sequence[Tuple[int, str]],
    rep_spans: Sequence[Mapping[str, Any]],
    all_spans: Sequence[Mapping[str, Any]],
    *,
    dual_fabric: bool,
    is_amd: bool,
) -> List[str]:
    """Build clean per-warp / per-wave Y-axis labels, detecting Streaming Reduce on comm lanes."""
    if not rep_warps:
        return ["W00"]
    if not dual_fabric:
        return [f"W{w:02d} · {r}" for w, r in rep_warps]

    # Detect streaming Reduce on comm lanes using per-(unit, lane) comm identification so
    # pure-comm CTAs (where W00..W03 are also comm lanes) never pollute `tail_comb_start`.
    comm_tracks: set[Tuple[int, int]] = set()
    for sp in all_spans:
        frole = sp.get("_frole")
        rl = str(sp.get("role") or "").lower()
        if frole in ("inter_dispatch", "intra_dispatch") or any(k in rl for k in ("comm", "dispatch", "scatter")):
            comm_tracks.add((int(sp["sm"]), int(sp["warp"])))
    rep_sm_id = int(rep_spans[0]["sm"]) if rep_spans else 0
    for w, r in rep_warps:
        if any(k in r.lower() for k in ("comm", "dispatch", "scatter")):
            comm_tracks.add((rep_sm_id, int(w)))
    non_comm_comb = [
        float(sp["ts"]) for sp in all_spans
        if (int(sp["sm"]), int(sp["warp"])) not in comm_tracks
        and ("combine" in sp["label"].lower() or "reduce" in sp["label"].lower())
    ]
    tail_comb_start = min(non_comm_comb) if non_comm_comb else float("inf")
    streaming_warps = set()
    for sp in rep_spans:
        if (int(sp["sm"]), int(sp["warp"])) in comm_tracks and float(sp["ts"]) < tail_comb_start:
            low = sp["label"].lower()
            if "combine" in low or "reduce" in low or "return" in low:
                streaming_warps.add(int(sp["warp"]))
    out = []
    for w, r in rep_warps:
        rl = r.lower()
        if "compute" in rl or "math" in rl:
            out.append(f"W{w:02d} (Compute / Combine)")
        elif "comm" in rl or "dispatch" in rl or "scatter" in rl:
            suffix = "Comm / Scatter -> Streaming Reduce" if w in streaming_warps else "Comm / Scatter"
            out.append(f"W{w:02d} ({suffix})")
        else:
            out.append(f"W{w:02d} ({r})")
    return out


def render_swhw_overview_png(
    trace_or_path: Mapping[str, Any] | str | Path,
    out_png: str | Path,
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    focus_sm: Optional[int] = None,
    dpi: int = 220,
) -> Path:
    """Render a publication-grade 3-panel single-run SW+HW overview PNG on one locked time axis."""
    import matplotlib
    matplotlib.use("Agg")
    if sys.version_info >= (3, 14):
        from matplotlib.path import Path as _MplPath
        def _safe_path_deepcopy(self, memo):
            p = _MplPath(
                self.vertices.copy(),
                self.codes.copy() if self.codes is not None else None,
            )
            p._interpolation_steps = getattr(self, "_interpolation_steps", 1)
            p._simplify_threshold = getattr(self, "_simplify_threshold", 0.0)
            p._should_simplify = getattr(self, "_should_simplify", False)
            return p
        _MplPath.__deepcopy__ = _safe_path_deepcopy
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
    import numpy as np

    trace = _load_trace(trace_or_path)
    parsed = _parse_trace(trace)
    hw_curves = parsed["hw_curves"]
    hw_labels = parsed["hw_labels"]
    sw_spans = parsed["sw_spans"]
    sms = parsed["sms"]
    t_max = parsed["t_max"]
    k_dur = parsed["k_dur"]
    nsys_k = parsed["nsys_kernel"]
    other = parsed["other"]
    meta = other.get("meta") if isinstance(other.get("meta"), Mapping) else {}
    nsys_meta = (other.get("nsys_hw") or {})
    is_amd = bool(parsed["is_amd"])
    multi_node = bool(parsed["multi_node"])
    dual_fabric = bool(parsed["dual_fabric"])
    use_ib = bool(parsed["use_ib"])
    unit_nm = "CU" if is_amd else "SM"
    lane_nm = "wave" if is_amd else "warp"

    # Pick representative SM/CU for Panel (C)
    sm_warp_role: Dict[int, Dict[int, str]] = defaultdict(dict)
    for sm_id, wmap in (parsed.get("meta_warps_by_sm") or {}).items():
        for w_id, r_lbl in wmap.items():
            sm_warp_role[int(sm_id)][int(w_id)] = str(r_lbl)
    other_sm_warps = other.get("sm_warps") or {}
    if isinstance(other_sm_warps, Mapping):
        for sm_k, wmap in other_sm_warps.items():
            if isinstance(wmap, Mapping):
                for w_k, r_lbl in wmap.items():
                    try:
                        sm_warp_role[int(sm_k)][int(w_k)] = str(r_lbl)
                    except (ValueError, TypeError):
                        pass
    for sp in sw_spans:
        sm_warp_role[sp["sm"]].setdefault(sp["warp"], sp["role"])
    wpc_meta = int(
        other.get("wpc")
        or meta.get("wpc")
        or (other.get("summary") or {}).get("wpc")
        or 0
    )
    global_warp_roles = other.get("warp_roles") or {}
    if wpc_meta > 1:
        for sm_id in list(sm_warp_role.keys()):
            for w_id in range(wpc_meta):
                sm_warp_role[sm_id].setdefault(w_id, str(global_warp_roles.get(str(w_id), "idle")))
    sm_roles: Dict[int, List[Tuple[int, str]]] = {
        sm_id: sorted(wmap.items()) for sm_id, wmap in sm_warp_role.items()
    }
    if focus_sm is not None and focus_sm in sm_roles:
        rep_sm = int(focus_sm)
    elif 0 in sm_roles:
        rep_sm = 0
    elif sms:
        rep_sm = max(sms, key=lambda s: (len(sm_roles.get(s, ())), -s))
    else:
        rep_sm = 0

    rep_warps = list(sm_roles.get(rep_sm, ()))
    rep_spans_raw = [sp for sp in sw_spans if sp["sm"] == rep_sm]
    use_cta_slots = False
    rep_cta_slot: Dict[int, int] = {}
    rep_slot_count = 1
    if len(rep_warps) == 1 and rep_spans_raw:
        rep_cta_slot, rep_slot_count = _pack_concurrent_ctas(rep_spans_raw)
        if rep_slot_count > 1:
            use_cta_slots = True

    n_sms = max(1, len(sms))
    n_rep_rows = rep_slot_count if use_cta_slots else max(1, len(rep_warps))

    has_hw = any(hw_curves.get(k) for k in ("tensor_active", "sm_active", "sm_issue", "dram_bw", "nvlink_tx", "nvlink_rx", "nvlink_bw", "inter_rdma"))
    panel_c_h = max(2.2, min(4.8, (0.26 if n_rep_rows >= 8 else 0.40) * n_rep_rows + 1.05))
    panel_b_h = max(3.1, min(4.4, 0.036 * n_sms + 1.55))
    panel_a_h = 2.25 if has_hw else 0.0
    fig_h = 2.25 + panel_a_h + panel_b_h + panel_c_h
    fig_w = 16.0

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#B0BEC5",
        "axes.linewidth": 0.8,
        "xtick.color": "#37474F",
        "ytick.color": "#37474F",
        "text.color": "#111111",
    })

    if has_hw:
        fig, axes = plt.subplots(
            3, 1, figsize=(fig_w, fig_h), sharex=True,
            gridspec_kw={"height_ratios": [panel_a_h, panel_b_h, panel_c_h], "hspace": 0.42 if dual_fabric else 0.28},
        )
        ax_hw, ax_sm, ax_wp = axes
    else:
        fig, axes = plt.subplots(
            2, 1, figsize=(fig_w, fig_h), sharex=True,
            gridspec_kw={"height_ratios": [panel_b_h, panel_c_h], "hspace": 0.28},
        )
        ax_hw = None
        ax_sm, ax_wp = axes

    fig.patch.set_facecolor("#FFFFFF")

    # ---- Title & subtitle ----
    kname = ""
    if nsys_k and nsys_k.get("name"):
        kname = _short_kernel_name(str(nsys_k["name"]))
    if not kname:
        kname = _short_kernel_name(str(meta.get("kernel") or meta.get("model") or "WeaveProfiler Kernel"))
    if not title:
        ep_tag = f" · EP{parsed['ep']}" if parsed["ep"] else ""
        plat_tag = "AMD MI308X" if is_amd else (str(meta.get("card") or "").upper() or "GPU")
        title = f"{kname} ({plat_tag}{ep_tag}) — {k_dur:,.1f} µs · {len(sms)} Physical {unit_nm}s"
    if not subtitle:
        freq_khz = None
        if nsys_meta.get("sample_freq_hz"):
            freq_khz = float(nsys_meta["sample_freq_hz"]) / 1000.0
        elif nsys_k and (nsys_k.get("args") or {}).get("sample_freq_hz"):
            freq_khz = float(nsys_k["args"]["sample_freq_hz"]) / 1000.0
        hw_tool_lbl = "rocprofiler-sdk PMC" if is_amd else (f"{freq_khz:.1f} kHz nsys GPU metrics" if freq_khz else "nsys GPU metrics")
        arch = str(meta.get("arch") or other.get("arch") or "")
        arch_str = f"{arch} · " if arch else ""
        magic = str(meta.get("magic") or "WPLITE02")
        lane_desc = (
            f"up to {rep_slot_count} concurrent CTAs/{unit_nm}"
            if use_cta_slots
            else f"{len(rep_warps) or 1} recorded {lane_nm} lane{'s' if len(rep_warps) != 1 else ''}/{unit_nm}"
        )
        subtitle = (
            f"{arch_str}{len(sms)} physical {unit_nm}s × {lane_desc} · "
            f"{len(sw_spans):,} software stage spans · {hw_tool_lbl} + {magic} timeline"
        )

    fig.text(0.5, 0.985, title, ha="center", va="top", fontsize=14.2, fontweight="bold", color="#111111")
    fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=9.1, color="#455A64")

    # ---- Stage / Dual-Fabric Color Map & Legends ----
    panel_b_fabric_handles: List[Patch] = []
    panel_c_handles: List[Patch] = []
    top_legend_handles: List[Patch] = []

    if dual_fabric:
        # Multi-node dual-color fabric mode: Inter-Node RDMA/IB (warm red/amber) vs Intra-Node xGMI/NVLink (cool teal/blue)
        role_order = (
            "inter_dispatch", "inter_combine", "intra_dispatch", "intra_combine",
            "gemm1", "gemm2", "route_sync", "local_prep",
        )
        seen_roles: set[str] = set()
        for sp in sw_spans:
            frole = _classify_fabric_role(sp, is_amd=is_amd)
            sp["_frole"] = frole
            if frole:
                seen_roles.add(frole)
        # Panel B dedicated 4-swatch Inter-Node vs Intra-Node legend strip
        for frole in ("inter_dispatch", "inter_combine", "intra_dispatch", "intra_combine"):
            col = FABRIC_COLORS_BY_ROLE[frole]
            lbl = _fabric_role_label(frole, is_amd=is_amd, use_ib=use_ib)
            panel_b_fabric_handles.append(Patch(facecolor=col, edgecolor="none", label=lbl))
        # Panel C complete legend (4 fabrics + compute/route/prep stages present)
        for frole in role_order:
            if frole in ("inter_dispatch", "inter_combine", "intra_dispatch", "intra_combine") or frole in seen_roles:
                col = FABRIC_COLORS_BY_ROLE[frole]
                lbl = _fabric_role_label(frole, is_amd=is_amd, use_ib=use_ib)
                panel_c_handles.append(Patch(facecolor=col, edgecolor="none", label=lbl))

        def _span_color(sp: Mapping[str, Any]) -> str:
            fr = sp.get("_frole")
            if fr and fr in FABRIC_COLORS_BY_ROLE:
                return FABRIC_COLORS_BY_ROLE[fr]
            return STAGE_COLORS.get(sp["sem"], _DEFAULT_COLOR)[0]
    else:
        # Single-node / general kernel mode: preserve rich human-readable stage labels for EVERY stage!
        labels_by_sem: Dict[str, List[str]] = defaultdict(list)
        for sp in sw_spans:
            lbl = str(sp.get("label") or sp["sem"]).strip()
            if lbl and lbl not in labels_by_sem[sp["sem"]]:
                labels_by_sem[sp["sem"]].append(lbl)
        stage_color_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
        stage_order = ("load", "mma", "gemm", "softmax", "compute", "epilogue", "store", "wait", "barrier", "communication", "reduction", "dispatch")
        ordered_sems = [s for s in stage_order if s in parsed["stages_seen"]] + sorted(
            s for s in parsed["stages_seen"] if s not in stage_order
        )
        seen_leg = set()
        used_colors = set()
        overflow_idx = 0
        for sem in ordered_sems:
            base_col, _base_desc = STAGE_COLORS.get(sem, _DEFAULT_COLOR)
            lbls = sorted(labels_by_sem.get(sem) or [sem], key=lambda x: (x.lower() == sem, x.lower()))
            shades = list(_SEMANTIC_SHADES.get(sem, (base_col,)))
            if len(lbls) >= 2:
                for idx_l, lbl in enumerate(lbls):
                    col = next((c for c in shades[idx_l:] if c not in used_colors), None)
                    if col is None:
                        col = _OVERFLOW_SHADES[overflow_idx % len(_OVERFLOW_SHADES)]
                        overflow_idx += 1
                    used_colors.add(col)
                    desc = _humanize_stage_label(lbl, sem)
                    stage_color_map[(sem, lbl)] = (col, desc)
                    if desc not in seen_leg:
                        seen_leg.add(desc)
                        top_legend_handles.append(Patch(facecolor=col, edgecolor="none", label=desc))
            else:
                only_lbl = lbls[0]
                used_colors.add(base_col)
                desc = _humanize_stage_label(only_lbl, sem)
                stage_color_map[(sem, only_lbl)] = (base_col, desc)
                if desc not in seen_leg:
                    seen_leg.add(desc)
                    top_legend_handles.append(Patch(facecolor=base_col, edgecolor="none", label=desc))

        def _span_color(sp: Mapping[str, Any]) -> str:
            lbl = str(sp.get("label") or sp["sem"]).strip()
            return stage_color_map.get((sp["sem"], lbl), STAGE_COLORS.get(sp["sem"], _DEFAULT_COLOR))[0]

        if top_legend_handles:
            fig.legend(
                handles=top_legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.930),
                ncol=min(len(top_legend_handles), 6), frameon=False, fontsize=8.8, handlelength=1.4, columnspacing=1.5,
            )

    x_pad = max(1.0, t_max * 0.015)
    x_lim = (0.0, t_max + x_pad)

    # ---- Panel (A): real same-run hardware utilization curves ----
    if ax_hw is not None:
        ax_hw.set_facecolor("#FFFFFF")
        ax_hw.grid(True, axis="y", linestyle="-", linewidth=0.5, color="#ECEFF1", zorder=0)
        ax_hw.grid(True, axis="x", linestyle="-", linewidth=0.4, color="#F5F7F8", zorder=0)
        ax_hw.axhline(100.0, color="#CFD8DC", linestyle="--", linewidth=0.8, zorder=1)
        ax_hw.axvline(k_dur, color="#263238", linestyle="--", linewidth=1.0, alpha=0.75, zorder=2)

        comm_pts = hw_curves.get("nvlink_bw") or []
        nvl_tx_pts = hw_curves.get("nvlink_tx") or []
        nvl_rx_pts = hw_curves.get("nvlink_rx") or []
        has_active_nvl_tx = bool(nvl_tx_pts and max((p[1] for p in nvl_tx_pts if 0.0 <= p[0] <= k_dur), default=0.0) >= 0.5)
        has_active_nvl_rx = bool(nvl_rx_pts and max((p[1] for p in nvl_rx_pts if 0.0 <= p[0] <= k_dur), default=0.0) >= 0.5)
        has_active_comm = bool(comm_pts and max((p[1] for p in comm_pts if 0.0 <= p[0] <= k_dur), default=0.0) >= 0.5)
        rdma_pts = hw_curves.get("inter_rdma") or []
        has_active_rdma = bool(rdma_pts and max((p[1] for p in rdma_pts if 0.0 <= p[0] <= k_dur), default=0.0) > 0.0)

        curve_order_keys = ["tensor_active", "dram_bw"]
        # NVLink TX (egress) and NVLink RX (ingress) are independent full-duplex paths: plot separately, NEVER merged.
        if has_active_nvl_tx:
            curve_order_keys.append("nvlink_tx")
        if has_active_nvl_rx:
            curve_order_keys.append("nvlink_rx")
        if not (has_active_nvl_tx or has_active_nvl_rx):
            if has_active_comm:
                curve_order_keys.append("nvlink_bw")
            elif hw_curves.get("sm_issue"):
                curve_order_keys.append("sm_issue")
        if has_active_rdma:
            curve_order_keys.append("inter_rdma")
        if hw_curves.get("sm_active"):
            curve_order_keys.append("sm_active")

        style_map = {
            "tensor_active": ("#1565C0", "-", 1.9),
            "dram_bw": ("#E65100", "-", 1.6),
            "nvlink_tx": ("#00897B", "-", 1.6),
            "nvlink_rx": ("#2E7D32", "-.", 1.55),
            "nvlink_bw": ("#00897B", "-", 1.6),
            "inter_rdma": ("#7B1FA2", "-", 1.7),
            "sm_issue": ("#00897B", "-", 1.5),
            "sm_active": ("#37474F", "--", 1.35),
        }
        chips: List[Tuple[str, str]] = []
        hw_max_val = 100.0
        for key in curve_order_keys:
            pts = hw_curves.get(key)
            if not pts:
                continue
            xs = [p[0] for p in pts if 0.0 <= p[0] <= t_max + x_pad]
            ys = [p[1] for p in pts if 0.0 <= p[0] <= t_max + x_pad]
            if not xs:
                continue
            color, ls, lw = style_map[key]
            label, short = _pretty_hw_curve_label(hw_labels.get(key, ""), key, is_amd=is_amd, multi_node=multi_node)
            ax_hw.plot(xs, ys, label=label, color=color, linestyle=ls, linewidth=lw, zorder=4)
            in_w = [p[1] for p in pts if 0.0 <= p[0] <= k_dur] or ys
            mean_v = sum(in_w) / len(in_w)
            peak_v = max(in_w)
            hw_max_val = max(hw_max_val, peak_v)
            chips.append((f"{short}  peak {peak_v:.1f}% · mean {mean_v:.1f}%", color))

        y_top = 108.0 if hw_max_val <= 100.0 else max(108.0, hw_max_val * 1.08)
        ax_hw.set_ylim(-4, y_top)
        ax_hw.set_yticks([0, 25, 50, 75, 100])
        ax_hw.set_ylabel("Utilization (%)", fontsize=9.5, color="#263238")
        hw_title_src = "same-run SW+HW capture" if is_amd else "real nsys GPU metric curves"
        ax_hw.set_title(f"(A) Aligned Hardware Utilization Curves ({hw_title_src})",
                        fontsize=10.5, fontweight="bold", color="#111111", pad=44)
        ax_hw.legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.33),
            ncol=min(max(2, len(curve_order_keys)), 3 if len(curve_order_keys) >= 5 else 4),
            frameon=False, fontsize=8.4, handlelength=2.0,
        )

        if multi_node and not has_active_rdma:
            ax_hw.text(
                0.012, 0.78,
                "Inter-Node RDMA NIC counters unavailable on this HCA (omitted, not zero-filled)",
                transform=ax_hw.transAxes, ha="left", va="center",
                fontsize=8.2, fontstyle="italic", color="#8B1E1E", zorder=6,
            )

        if chips and not dual_fabric:
            n_ch = len(chips)
            for idx_c, (txt, col) in enumerate(chips):
                chip_x = 0.004 + idx_c * (0.992 / max(1, n_ch))
                ax_hw.text(
                    chip_x, 1.045, txt, transform=ax_hw.transAxes, ha="left", va="bottom",
                    fontsize=7.9, fontweight="bold", color="#111111",
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#F8FAFC", edgecolor=col, linewidth=1.0),
                )

    # ---- Panel (B): All physical SMs / CUs dense rasterized waterfall ----
    ax_sm.set_facecolor("#FFFFFF")
    img_w = 2400
    sm_to_row = {sm: idx for idx, sm in enumerate(sms)}
    all_warps = sorted({w for wmap in sm_warp_role.values() for w in wmap} | {sp["warp"] for sp in sw_spans})
    warp_to_sub = {w: idx for idx, w in enumerate(all_warps)}
    sm_cta_slots: Dict[Tuple[int, int], int] = {}
    if len(all_warps) == 1 and use_cta_slots:
        spans_by_sm: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
        for sp in sw_spans:
            spans_by_sm[sp["sm"]].append(sp)
        max_slots = 1
        for sm_id, sps in spans_by_sm.items():
            c2s, sc = _pack_concurrent_ctas(sps)
            max_slots = max(max_slots, sc)
            for c, sl in c2s.items():
                sm_cta_slots[(sm_id, c)] = sl
        n_sub = max_slots
    else:
        n_sub = max(1, len(all_warps))
    img = np.full((n_sms * n_sub, img_w, 3), 255, dtype=np.uint8)
    sorted_spans = sorted(
        sw_spans,
        key=lambda s: (0 if s["sem"] in ("wait", "barrier") else 1, -float(s["dur"]), _STAGE_DRAW_PRIORITY.get(s["sem"], 4)),
    )
    scale_x = (img_w - 1) / max(t_max + x_pad, 1e-6)
    for sp in sorted_spans:
        row = sm_to_row.get(sp["sm"])
        if row is None:
            continue
        if sm_cta_slots:
            sub = sm_cta_slots.get((sp["sm"], sp["cta"]), 0)
        else:
            sub = warp_to_sub.get(sp["warp"], 0)
        x0 = max(0, min(img_w - 1, int(sp["ts"] * scale_x)))
        x1 = max(x0 + 1, min(img_w, int(math.ceil(sp["end"] * scale_x))))
        if dual_fabric and sp.get("_frole") in ("inter_dispatch", "inter_combine", "intra_combine"):
            x1 = max(x1, min(img_w, x0 + 2))
        img[row * n_sub + sub, x0:x1] = _hex_to_rgb(_span_color(sp))

    ax_sm.imshow(img, aspect="auto", interpolation="nearest", extent=[0.0, t_max + x_pad, n_sms - 0.5, -0.5])
    ax_sm.axvline(k_dur, color="#263238", linestyle="--", linewidth=1.0, alpha=0.75)
    step = max(1, (n_sms // 6 + 4) // 5 * 5) if n_sms >= 15 else max(1, n_sms // 4)
    yt = list(range(0, n_sms, step))
    if n_sms > 1 and ((n_sms - 1) - yt[-1]) >= max(2, step // 2):
        yt.append(n_sms - 1)
    ax_sm.set_yticks(yt)
    ax_sm.set_yticklabels([f"{unit_nm}{sms[i]}" for i in yt] if sms else [f"{unit_nm}0"], fontsize=8.2)
    rank_tag = f" (rank {meta['rank']})" if meta.get("rank") is not None else f" ({len(sms)} {unit_nm}s)"
    ax_sm.set_ylabel(f"Physical {unit_nm}{rank_tag}", fontsize=9.5, color="#263238")
    if dual_fabric and panel_b_fabric_handles:
        ax_sm.set_title(
            f"(B) Multi-CTA Execution Across All {len(sms)} Physical {unit_nm}s",
            fontsize=10.5, fontweight="bold", color="#111111", loc="left", pad=8,
        )
        ax_sm.legend(
            handles=panel_b_fabric_handles, loc="lower right", bbox_to_anchor=(1.0, 1.005),
            ncol=4, frameon=False, fontsize=8.1, handlelength=1.25, columnspacing=1.1,
        )
    else:
        ax_sm.set_title(
            f"(B) All {len(sms)} physical {unit_nm}s ({unit_nm}{sms[0]}–{unit_nm}{sms[-1]}) — full per-tile stage waterfall"
            if sms else f"(B) Physical {unit_nm} timeline",
            fontsize=10.5, fontweight="bold", color="#111111", pad=6,
        )

    # ---- Panel (C): Representative SM/CU warp/wave role breakdown ----
    ax_wp.set_facecolor("#FFFFFF")
    ax_wp.grid(True, axis="x", linestyle="-", linewidth=0.4, color="#ECEFF1", zorder=0)
    warp_to_y = {w: idx for idx, (w, _r) in enumerate(rep_warps)}
    verts_by_color: Dict[str, List[List[Tuple[float, float]]]] = defaultdict(list)
    durs_by_color: Dict[str, List[float]] = defaultdict(list)
    sem_by_color: Dict[str, str] = {}
    is_priority_color: Dict[str, bool] = {}
    bar_h = 0.72
    rep_spans = [sp for sp in sorted_spans if sp["sm"] == rep_sm]
    # Crisp minimum display width (~2 px on 16-inch 220 DPI figure) for short combine/RDMA slices on Panel C
    min_vis_us = (t_max + x_pad) * (2.0 / 2800.0)
    for sp in rep_spans:
        if use_cta_slots:
            y = rep_cta_slot.get(sp["cta"])
        else:
            y = warp_to_y.get(sp["warp"])
        if y is None:
            continue
        x0, x1 = float(sp["ts"]), float(sp["end"])
        is_crit = dual_fabric and sp.get("_frole") in ("inter_dispatch", "inter_combine", "intra_combine")
        if is_crit and (x1 - x0) < min_vis_us:
            x1 = x0 + min_vis_us
        y0, y1 = y - bar_h / 2.0, y + bar_h / 2.0
        col_hex = _span_color(sp)
        verts_by_color[col_hex].append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        durs_by_color[col_hex].append(float(sp["dur"]))
        sem_by_color.setdefault(col_hex, sp["sem"])
        if is_crit:
            is_priority_color[col_hex] = True

    ordered_color_layers = sorted(
        verts_by_color.items(),
        key=lambda kv: (
            2 if is_priority_color.get(kv[0]) else (0 if sem_by_color.get(kv[0]) in ("wait", "barrier") else 1),
            -sorted(durs_by_color[kv[0]])[len(durs_by_color[kv[0]]) // 2],
        ),
    )
    for layer_idx, (col_hex, verts) in enumerate(ordered_color_layers):
        pc = PolyCollection(
            verts, facecolors=col_hex, edgecolors="none" if is_priority_color.get(col_hex) else "#FFFFFF",
            linewidths=0.0 if is_priority_color.get(col_hex) else 0.16,
            antialiased=not is_priority_color.get(col_hex),
            zorder=3 + layer_idx * 0.05,
        )
        ax_wp.add_collection(pc)

    ax_wp.axvline(k_dur, color="#263238", linestyle="--", linewidth=1.0, alpha=0.75, zorder=6)
    ax_wp.set_ylim(n_rep_rows - 0.45, -0.55)
    ax_wp.set_yticks(range(n_rep_rows))
    yt_fs = 7.8 if n_rep_rows >= 12 else 8.2
    if use_cta_slots:
        role_lbl = rep_warps[0][1] if rep_warps else "cta"
        ax_wp.set_yticklabels([f"Slot {i} · {role_lbl}" for i in range(n_rep_rows)], fontsize=yt_fs)
        ax_wp.set_ylabel(f"{unit_nm} {rep_sm} CTA slots", fontsize=9.5, color="#263238")
        ax_wp.set_title(
            f"(C) {unit_nm} {rep_sm} — {n_rep_rows} concurrent CTA slots ({len(rep_cta_slot)} CTAs total on {unit_nm} {rep_sm})",
            fontsize=10.5, fontweight="bold", color="#111111", pad=6,
        )
    else:
        c_labels = _compute_panel_c_role_labels(rep_warps, rep_spans, sw_spans, dual_fabric=dual_fabric, is_amd=is_amd)
        ax_wp.set_yticklabels(c_labels, fontsize=yt_fs)
        ax_wp.set_ylabel(f"{unit_nm} {rep_sm} {lane_nm}", fontsize=9.5, color="#263238")
        if dual_fabric and panel_c_handles:
            ax_wp.set_title(
                f"(C) Representative {unit_nm} {rep_sm} — Complete Pipeline by Real {lane_nm.title()} Role",
                fontsize=10.5, fontweight="bold", color="#111111", pad=30,
            )
            ax_wp.legend(
                handles=panel_c_handles, loc="upper center", bbox_to_anchor=(0.5, 1.135),
                ncol=min(len(panel_c_handles), 8), frameon=False, fontsize=7.9, handlelength=1.2, columnspacing=1.0,
            )
        else:
            role_names = ", ".join(dict.fromkeys(r for _, r in rep_warps))
            ax_wp.set_title(
                f"(C) {unit_nm} {rep_sm} — all {len(rep_warps)} {lane_nm}s ({role_names})",
                fontsize=10.5, fontweight="bold", color="#111111", pad=6,
            )
    ax_wp.set_xlim(*x_lim)
    ax_wp.set_xlabel("Time (µs) — identical locked scale across all panels", fontsize=9.5, color="#263238")

    yt_labels = [t.get_text() for t in ax_wp.get_yticklabels()]
    max_lbl = max((len(s) for s in yt_labels), default=10)
    left_m = max(0.098, min(0.185, 0.052 + 0.0044 * max_lbl))
    top_m = 0.855 if (has_hw and dual_fabric) else (0.805 if has_hw else 0.875)
    fig.subplots_adjust(left=left_m, right=0.985, top=top_m, bottom=0.065)

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=int(dpi), facecolor="#FFFFFF")
    plt.close(fig)
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Render a WeaveProfiler Perfetto trace into a 3-panel SW+HW overview PNG.")
    ap.add_argument("trace", help="Path to trace.perfetto.json[.gz]")
    ap.add_argument("-o", "--out", required=True, help="Output PNG path")
    ap.add_argument("--title", default=None)
    ap.add_argument("--subtitle", default=None)
    ap.add_argument("--sm", type=int, default=None, help="Physical SM/CU id to feature in Panel (C)")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args(argv)
    out = render_swhw_overview_png(args.trace, args.out, title=args.title, subtitle=args.subtitle, focus_sm=args.sm, dpi=args.dpi)
    print(f"RENDERED {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
