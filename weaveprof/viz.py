"""Publication-style single-run 3-panel SW+HW overview renderer for WeaveProfiler traces.

Renders a WeaveProfiler ``trace.perfetto.json.gz`` (or in-memory trace dict) into a single
high-DPI PNG with one locked microsecond time axis across all three panels:
  (A) Same-run hardware utilization curves from nsys (Tensor Active, DRAM BW, SM Issue, SMs Active)
  (B) All physical SMs (SM0..SM{N-1}) dense per-tile stage waterfall
  (C) Representative SM (SM 0 or first observed SM) per-warp-role stage timeline
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

STAGE_COLORS: Dict[str, Tuple[str, str]] = {
    "load": ("#2E7D32", "TMA / Tile Load"),
    "mma": ("#1565C0", "Tensor Core MMA / WGMMA"),
    "softmax": ("#00ACC1", "Online Softmax"),
    "compute": ("#42A5F5", "Compute / ALU"),
    "epilogue": ("#EF6C00", "Epilogue / Store"),
    "store": ("#EF6C00", "Epilogue / Store"),
    "wait": ("#90A4AE", "Wait / Sync"),
    "barrier": ("#78909C", "Barrier / Sync"),
    "communication": ("#8E24AA", "Communication / Reduce"),
    "reduction": ("#8E24AA", "Communication / Reduce"),
    "dispatch": ("#00838F", "Dispatch / Schedule"),
    "gemm": ("#1565C0", "GEMM / MMA"),
    "attention": ("#1976D2", "Attention"),
    "moe": ("#6A1B9A", "MoE"),
}
_DEFAULT_COLOR = ("#546E7A", "Other Stage")
_STAGE_DRAW_PRIORITY = {
    "wait": 0, "barrier": 1, "load": 2, "dispatch": 3, "compute": 4,
    "softmax": 5, "epilogue": 6, "store": 6, "communication": 7, "reduction": 7,
    "mma": 8, "gemm": 8,
}


def _short_kernel_name(raw: str, max_len: int = 72) -> str:
    s = str(raw or "").strip()
    if not s:
        return "WeaveProfiler Kernel"
    s = s.split("(")[0].strip()
    if s.startswith("void "):
        s = s[5:].strip()
    # Collapse verbose C++ template argument lists so title fits cleanly on one line
    if "<" in s and len(s) > max_len:
        idx = s.find("<")
        prefix = s[:idx]
        suffix = s[s.rfind(">") + 1:] if ">" in s else ""
        s = f"{prefix}<...>{suffix}"
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


def _pack_concurrent_ctas(spans: Sequence[Mapping[str, Any]]) -> Tuple[Dict[int, int], int]:
    """Pack overlapping CTAs on one SM into the minimal number of non-overlapping resident slots."""
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
    if "tensor" in low:
        return "tensor_active"
    if "sms active" in low or "sm active" in low:
        return "sm_active"
    if "sm issue" in low or "issue" in low:
        return "sm_issue"
    if "dram read" in low or "hbm" in low:
        return "dram_read"
    if "dram write" in low:
        return "dram_write"
    return None


def _parse_trace(trace: Mapping[str, Any]) -> Dict[str, Any]:
    events = trace.get("traceEvents") or []
    hw_curves: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    hw_labels: Dict[str, str] = {}
    nsys_kernel: Optional[Dict[str, Any]] = None
    sw_spans: List[Dict[str, Any]] = []
    sm_set: set[int] = set()
    stages_seen: Dict[str, float] = defaultdict(float)
    meta_warps_by_sm: Dict[int, Dict[int, str]] = defaultdict(dict)

    import re as _re
    for ev in events:
        ph = ev.get("ph")
        if ph == "M" and ev.get("name") == "thread_name":
            pid = ev.get("pid")
            tname = str((ev.get("args") or {}).get("name") or "")
            if isinstance(pid, int) and pid != 1:
                sm = pid - 100 if pid >= 100 else pid
                m = _re.search(r"\bw(\d+)\b", tname)
                if m:
                    w_id = int(m.group(1))
                    role_part = tname.split("·")[-1].strip() if "·" in tname else (
                        tname.split("/")[-1].strip() if "/" in tname else f"w{w_id:02d}"
                    )
                    # Strip leading `w00` prefix if role_part is just `w00` or `cta0 w00`
                    if _re.fullmatch(r"w\d+", role_part):
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
            if cat == "nsys_kernel":
                nsys_kernel = dict(ev)
                continue
            if cat == "info":
                continue
            ts = float(ev.get("ts") or 0.0)
            dur = float(ev.get("dur") or 0.0)
            if dur <= 0.0 or not math.isfinite(dur):
                continue
            args = ev.get("args") or {}
            sem = str(args.get("semantic_id") or ev.get("name") or "compute").lower()
            if sem not in STAGE_COLORS:
                sem = str(ev.get("name") or "compute").lower()
            sm = args.get("sm")
            if sm is None:
                pid = ev.get("pid")
                sm = int(pid) - 100 if isinstance(pid, int) and pid >= 100 else (int(pid) if isinstance(pid, int) else 0)
            else:
                sm = int(sm)
            warp = int(args.get("warp", 0))
            cta = int(args.get("cta", 0))
            role = str(args.get("role") or f"w{warp:02d}")
            sm_set.add(sm)
            stages_seen[sem] += dur
            sw_spans.append({
                "ts": ts, "dur": dur, "end": ts + dur, "sem": sem,
                "label": str(args.get("original_label") or sem),
                "sm": sm, "warp": warp, "cta": cta, "role": role,
            })

    for k in hw_curves:
        hw_curves[k].sort(key=lambda x: x[0])

    # Combine DRAM read + write into aggregate DRAM bandwidth curve if both present
    if "dram_read" in hw_curves or "dram_write" in hw_curves:
        by_t: Dict[float, float] = defaultdict(float)
        for k in ("dram_read", "dram_write"):
            for t, v in hw_curves.get(k, ()):
                by_t[round(t, 3)] += v
        hw_curves["dram_bw"] = sorted(by_t.items())

    sw_end = max((s["end"] for s in sw_spans), default=0.0)
    k_dur = float(nsys_kernel["dur"]) if nsys_kernel and nsys_kernel.get("dur") else sw_end
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
        "other": dict(trace.get("otherData") or {}),
    }


def render_swhw_overview_png(
    trace_or_path: Mapping[str, Any] | str | Path,
    out_png: str | Path,
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    focus_sm: Optional[int] = None,
    dpi: int = 220,
) -> Path:
    """Render a compact 3-panel single-run SW+HW overview PNG on one locked time axis."""
    import matplotlib
    matplotlib.use("Agg")
    if sys.version_info >= (3, 14):
        # Work around Python 3.14 copy.deepcopy(super(), memo) recursion in older Matplotlib Path.__deepcopy__
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
    sw_spans = parsed["sw_spans"]
    sms = parsed["sms"]
    t_max = parsed["t_max"]
    k_dur = parsed["k_dur"]
    nsys_k = parsed["nsys_kernel"]
    nsys_meta = (parsed["other"].get("nsys_hw") or {})

    # Pick representative SM for Panel (C): focus_sm if present, else SM 0 if present, else first SM with most roles
    sm_warp_role: Dict[int, Dict[int, str]] = defaultdict(dict)
    for sm_id, wmap in (parsed.get("meta_warps_by_sm") or {}).items():
        for w_id, r_lbl in wmap.items():
            sm_warp_role[int(sm_id)][int(w_id)] = str(r_lbl)
    # Include all opened warp lanes recorded in otherData.sm_warps / otherData.warp_roles (e.g. OPEN/CLOSE-only idle warps)
    other_sm_warps = parsed["other"].get("sm_warps") or {}
    if isinstance(other_sm_warps, Mapping):
        for sm_k, wmap in other_sm_warps.items():
            if isinstance(wmap, Mapping):
                for w_k, r_lbl in wmap.items():
                    try:
                        sm_warp_role[int(sm_k)][int(w_k)] = str(r_lbl)
                    except (ValueError, TypeError):
                        pass
    for sp in sw_spans:
        sm_warp_role[sp["sm"]][sp["warp"]] = sp["role"]
    # Ensure every warp 0..wpc-1 exists on observed SMs if wpc > 1 (using otherData.warp_roles when available)
    wpc_meta = int(
        parsed["other"].get("wpc")
        or (parsed["other"].get("meta") or {}).get("wpc")
        or (parsed["other"].get("summary") or {}).get("wpc")
        or 0
    )
    global_warp_roles = parsed["other"].get("warp_roles") or {}
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

    has_hw = any(hw_curves.get(k) for k in ("tensor_active", "sm_active", "sm_issue", "dram_bw"))
    panel_c_h = max(2.1, min(4.8, (0.25 if n_rep_rows >= 8 else 0.40) * n_rep_rows + 0.95))
    panel_b_h = max(3.0, min(4.3, 0.036 * n_sms + 1.5))
    panel_a_h = 2.15 if has_hw else 0.0
    fig_h = 2.15 + panel_a_h + panel_b_h + panel_c_h
    fig_w = 15.5

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
            gridspec_kw={"height_ratios": [panel_a_h, panel_b_h, panel_c_h], "hspace": 0.28},
        )
        ax_hw, ax_sm, ax_wp = axes
    else:
        fig, axes = plt.subplots(
            2, 1, figsize=(fig_w, fig_h), sharex=True,
            gridspec_kw={"height_ratios": [panel_b_h, panel_c_h], "hspace": 0.24},
        )
        ax_hw = None
        ax_sm, ax_wp = axes

    fig.patch.set_facecolor("#FFFFFF")

    # ---- Title & legend ----
    kname = ""
    if nsys_k and nsys_k.get("name"):
        kname = _short_kernel_name(str(nsys_k["name"]))
    if not kname:
        kname = _short_kernel_name(str((parsed["other"].get("meta") or {}).get("kernel") or "WeaveProfiler Kernel"))
    if not title:
        title = f"{kname} — {k_dur:,.1f} µs · {len(sms)} Physical SMs"
    if not subtitle:
        freq_khz = None
        if nsys_meta.get("sample_freq_hz"):
            freq_khz = float(nsys_meta["sample_freq_hz"]) / 1000.0
        elif nsys_k and (nsys_k.get("args") or {}).get("sample_freq_hz"):
            freq_khz = float(nsys_k["args"]["sample_freq_hz"]) / 1000.0
        freq_str = f"{freq_khz:.1f} kHz nsys GPU metrics + " if freq_khz else ""
        arch = (parsed["other"].get("meta") or {}).get("arch") or ""
        arch_str = f"{arch} · " if arch else ""
        magic = str((parsed["other"].get("meta") or {}).get("magic") or "WPLITE02")
        lane_desc = (
            f"up to {rep_slot_count} concurrent CTAs/SM"
            if use_cta_slots
            else f"{len(rep_warps) or 1} recorded warp lane{'s' if len(rep_warps) != 1 else ''}/SM"
        )
        subtitle = (
            f"{arch_str}{len(sms)} physical SMs × {lane_desc} · "
            f"{len(sw_spans):,} software stage spans · {freq_str}{magic} timeline"
        )

    fig.text(0.5, 0.985, title, ha="center", va="top", fontsize=14.2, fontweight="bold", color="#111111")
    fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=9.1, color="#455A64")

    # Legend of actually present semantic stages (deduplicated by label/color)
    seen_leg = set()
    leg_handles: List[Patch] = []
    stage_order = ("load", "mma", "gemm", "softmax", "compute", "epilogue", "store", "wait", "barrier", "communication", "reduction", "dispatch")
    ordered_sems = [s for s in stage_order if s in parsed["stages_seen"]] + [
        s for s in parsed["stages_seen"] if s not in stage_order
    ]
    for sem in ordered_sems:
        col, desc = STAGE_COLORS.get(sem, _DEFAULT_COLOR)
        if desc in seen_leg:
            continue
        seen_leg.add(desc)
        leg_handles.append(Patch(facecolor=col, edgecolor="none", label=desc))
    if leg_handles:
        fig.legend(
            handles=leg_handles, loc="upper center", bbox_to_anchor=(0.5, 0.930),
            ncol=min(len(leg_handles), 6), frameon=False, fontsize=8.8, handlelength=1.4, columnspacing=1.5,
        )

    x_pad = max(1.0, t_max * 0.015)
    x_lim = (0.0, t_max + x_pad)

    # ---- Panel (A): nsys hardware utilization curves ----
    if ax_hw is not None:
        ax_hw.set_facecolor("#FFFFFF")
        ax_hw.grid(True, axis="y", linestyle="-", linewidth=0.5, color="#ECEFF1", zorder=0)
        ax_hw.grid(True, axis="x", linestyle="-", linewidth=0.4, color="#F5F7F8", zorder=0)
        ax_hw.axhline(100.0, color="#CFD8DC", linestyle="--", linewidth=0.8, zorder=1)
        ax_hw.axvline(k_dur, color="#263238", linestyle="--", linewidth=1.0, alpha=0.75, zorder=2)

        curve_specs = [
            ("tensor_active", "Tensor Active / MMA Pipe %", "#1565C0", "-", 1.9, "Compute"),
            ("dram_bw", "HBM / DRAM Bandwidth %", "#E65100", "-", 1.5, "DRAM BW"),
            ("sm_issue", "SM Issue %", "#00897B", "-", 1.5, "SM Issue"),
            ("sm_active", "Active SMs %", "#37474F", "--", 1.35, "Active SMs"),
        ]
        chips: List[Tuple[str, str]] = []
        hw_max_val = 100.0
        for key, label, color, ls, lw, short in curve_specs:
            pts = hw_curves.get(key)
            if not pts:
                continue
            xs = [p[0] for p in pts if 0.0 <= p[0] <= t_max + x_pad]
            ys = [p[1] for p in pts if 0.0 <= p[0] <= t_max + x_pad]
            if not xs:
                continue
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
        ax_hw.set_title("(A) Same-run hardware utilization — real nsys GPU metric curves",
                        fontsize=10.5, fontweight="bold", color="#111111", pad=44)
        ax_hw.legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=4, frameon=False, fontsize=8.4, handlelength=2.0)

        if chips:
            n_ch = len(chips)
            for idx_c, (txt, col) in enumerate(chips):
                chip_x = 0.004 + idx_c * (0.992 / max(1, n_ch))
                ax_hw.text(
                    chip_x, 1.045, txt, transform=ax_hw.transAxes, ha="left", va="bottom",
                    fontsize=7.9, fontweight="bold", color="#111111",
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#F8FAFC", edgecolor=col, linewidth=1.0),
                )

    # ---- Panel (B): All physical SMs dense rasterized waterfall (sub-row per warp role or concurrent CTA slot) ----
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
    sorted_spans = sorted(sw_spans, key=lambda s: _STAGE_DRAW_PRIORITY.get(s["sem"], 4))
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
        col_hex = STAGE_COLORS.get(sp["sem"], _DEFAULT_COLOR)[0]
        img[row * n_sub + sub, x0:x1] = _hex_to_rgb(col_hex)

    ax_sm.imshow(img, aspect="auto", interpolation="nearest", extent=[0.0, t_max + x_pad, n_sms - 0.5, -0.5])
    ax_sm.axvline(k_dur, color="#263238", linestyle="--", linewidth=1.0, alpha=0.75)
    step = max(1, (n_sms // 6 + 4) // 5 * 5) if n_sms >= 15 else max(1, n_sms // 4)
    yt = list(range(0, n_sms, step))
    if n_sms > 1 and ((n_sms - 1) - yt[-1]) >= max(2, step // 2):
        yt.append(n_sms - 1)
    ax_sm.set_yticks(yt)
    ax_sm.set_yticklabels([f"SM{sms[i]}" for i in yt] if sms else ["SM0"], fontsize=8.2)
    ax_sm.set_ylabel(f"Physical SM ({len(sms)} SMs)", fontsize=9.5, color="#263238")
    ax_sm.set_title(f"(B) All {len(sms)} physical SMs (SM{sms[0]}–SM{sms[-1]}) — full per-tile stage waterfall"
                    if sms else "(B) Physical SM timeline",
                    fontsize=10.5, fontweight="bold", color="#111111", pad=6)

    # ---- Panel (C): Representative SM warp-role (or concurrent resident CTA) breakdown ----
    ax_wp.set_facecolor("#FFFFFF")
    ax_wp.grid(True, axis="x", linestyle="-", linewidth=0.4, color="#ECEFF1", zorder=0)
    warp_to_y = {w: idx for idx, (w, _r) in enumerate(rep_warps)}
    verts_by_color: Dict[str, List[List[Tuple[float, float]]]] = defaultdict(list)
    bar_h = 0.72
    rep_spans = [sp for sp in sorted_spans if sp["sm"] == rep_sm]
    for sp in rep_spans:
        if use_cta_slots:
            y = rep_cta_slot.get(sp["cta"])
        else:
            y = warp_to_y.get(sp["warp"])
        if y is None:
            continue
        x0, x1 = sp["ts"], sp["end"]
        y0, y1 = y - bar_h / 2.0, y + bar_h / 2.0
        col_hex = STAGE_COLORS.get(sp["sem"], _DEFAULT_COLOR)[0]
        verts_by_color[col_hex].append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    for col_hex, verts in verts_by_color.items():
        pc = PolyCollection(verts, facecolors=col_hex, edgecolors="#FFFFFF", linewidths=0.18, zorder=3)
        ax_wp.add_collection(pc)

    ax_wp.axvline(k_dur, color="#263238", linestyle="--", linewidth=1.0, alpha=0.75, zorder=5)
    ax_wp.set_ylim(n_rep_rows - 0.45, -0.55)
    ax_wp.set_yticks(range(n_rep_rows))
    yt_fs = 7.8 if n_rep_rows >= 12 else 8.4
    if use_cta_slots:
        role_lbl = rep_warps[0][1] if rep_warps else "cta"
        ax_wp.set_yticklabels([f"Slot {i} · {role_lbl}" for i in range(n_rep_rows)], fontsize=yt_fs)
        ax_wp.set_ylabel(f"SM {rep_sm} CTA slots", fontsize=9.5, color="#263238")
        ax_wp.set_title(
            f"(C) SM {rep_sm} — {n_rep_rows} concurrent CTA slots ({len(rep_cta_slot)} CTAs total on SM {rep_sm})",
            fontsize=10.5, fontweight="bold", color="#111111", pad=6,
        )
    else:
        ax_wp.set_yticklabels([f"W{w:02d} · {r}" for w, r in rep_warps] if rep_warps else ["W00"], fontsize=yt_fs)
        ax_wp.set_ylabel(f"SM {rep_sm} warp", fontsize=9.5, color="#263238")
        role_names = ", ".join(dict.fromkeys(r for _, r in rep_warps))
        ax_wp.set_title(f"(C) SM {rep_sm} — all {len(rep_warps)} warps ({role_names})",
                        fontsize=10.5, fontweight="bold", color="#111111", pad=6)
    ax_wp.set_xlim(*x_lim)
    ax_wp.set_xlabel("Time (µs) — identical locked scale across all panels", fontsize=9.5, color="#263238")

    fig.subplots_adjust(left=0.095, right=0.985, top=0.805 if has_hw else 0.875, bottom=0.065)

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
    ap.add_argument("--sm", type=int, default=None, help="Physical SM id to feature in Panel (C)")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args(argv)
    out = render_swhw_overview_png(args.trace, args.out, title=args.title, subtitle=args.subtitle, focus_sm=args.sm, dpi=args.dpi)
    print(f"RENDERED {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
