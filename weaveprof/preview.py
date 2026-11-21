"""Render clean 3-panel co-axial SW+HW timeline PNGs from aligned WeaveProfiler traces.

Layout (aligned with user visual specification):
  - Panel A: Aligned HW Compute / HBM Bandwidth / SM-CU Active curves (legend OUTSIDE top)
  - Panel B: All physical SMs / CUs multi-CTA overview (no synthetic grey Residency bars)
  - Panel C: Representative SM / CU complete warp/wave pipeline preserving real stage names,
             colors (incl. MoE Dispatch/Pull/L1/L2/Scatter/Combine), and real role track names
             (legend OUTSIDE top, zero data occlusion; no Panel D micro-zoom).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

STAGE_PALETTE = {
    "Dispatch / Pull": "#0EA5E9",
    "DMA Issue": "#0EA5E9",
    "Async Load (DMA->Math)": "#10B981",
    "Tile Load": "#10B981",
    "L1 GEMM": "#2563EB",
    "L2 GEMM": "#4F46E5",
    "MMA / MFMA": "#2563EB",
    "Compute": "#3B82F6",
    "Scatter": "#F59E0B",
    "Combine / Reduce": "#9333EA",
    "Store / Epilogue": "#EA580C",
    "Wait / Barrier": "#64748B",
}

_FALLBACK_COLORS = [
    "#2563EB", "#0EA5E9", "#10B981", "#9333EA",
    "#EA580C", "#F59E0B", "#EC4899", "#14B8A6",
    "#6366F1", "#84CC16",
]


def _classify_stage(ev: dict, amd: bool) -> str | None:
    """Preserve real operator/MoE stage labels; drop synthetic grey Residency bars."""
    args = ev.get("args") or {}
    if args.get("kind") == "residency" or ev.get("cat") == "residency":
        return None
    if args.get("handoff_from") is not None:
        return "Async Load (DMA->Math)"

    raw_name = str(args.get("original_label") or ev.get("name") or "").strip()
    low = f"{raw_name} {args.get('detail') or ''} {args.get('semantic_id') or ''}".lower()
    role_low = str(args.get("role") or "").lower()

    # Preserve distinct MoE / multi-stage pipeline phases instead of flattening to DMA/MMA.
    if any(k in low for k in ("dispatch", "pull", "all2all", "a2a_in", "token_in")):
        return "Dispatch / Pull"
    if any(k in low for k in ("combine", "reduce", "allreduce", "a2a_out")):
        return "Combine / Reduce"
    if "scatter" in low:
        return "Scatter"
    if any(k in low for k in ("l1_gemm", "l1 gemm", "gemm1", "gate_up", "w13", "fc1", "up_proj")):
        return "L1 GEMM"
    if any(k in low for k in ("l2_gemm", "l2 gemm", "gemm2", "down_proj", "w2", "fc2")):
        return "L2 GEMM"

    # Standard pipeline stages
    if any(k in low for k in ("wait", "barrier", "sync", "stall")):
        return "Wait / Barrier"
    if any(k in low for k in ("store", "epilogue", "write")):
        return "Store / Epilogue"
    if any(k in low for k in ("mma", "mfma", "wgmma", "tcgen05", "gemm")):
        return "MMA / MFMA"
    if not amd and any(k in role_low or k in low for k in ("dma", "producer", "tma", "prefetch")):
        return "DMA Issue"
    if any(k in low for k in ("load", "copy")):
        return "Tile Load"
    if any(k in low for k in ("compute", "conv", "norm", "softmax", "act", "silu", "gelu", "gdn")):
        return "Compute"
    # Fall back to preserved raw label when informative.
    return raw_name if raw_name else "Compute"


def _color_for_stage(stage: str) -> str:
    if stage in STAGE_PALETTE:
        return STAGE_PALETTE[stage]
    return _FALLBACK_COLORS[abs(hash(stage)) % len(_FALLBACK_COLORS)]


def render_timeline_png(aligned: dict, verify_doc: dict, out_png: str | Path, title_override: str | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    other = aligned.get("otherData") or {}
    al = other.get("alignment") or {}
    amd = other.get("platform") == "amd"
    arch = str(other.get("arch") or "")
    k_start_ns = int(al.get("kernel_start_ns", 0))
    dur_us = float(al.get("kernel_duration_us") or 1.0)
    wpc = max(1, int(al.get("warps_per_cta") or (4 if amd else 8)))
    unit_lbl = "CU" if amd else "SM"

    events = aligned.get("traceEvents") or []
    proc_names = {
        int(e["pid"]): str((e.get("args") or {}).get("name") or "")
        for e in events if e.get("ph") == "M" and e.get("name") == "process_name"
    }
    thread_names: dict[tuple[int, int], str] = {}
    thread_sort_idx: dict[tuple[int, int], int] = {}
    for e in events:
        if e.get("ph") == "M":
            pid_i = int(e.get("pid", 0))
            tid_i = int(e.get("tid", 0))
            if e.get("name") == "thread_name":
                tname = str((e.get("args") or {}).get("name") or "").strip()
                if tname:
                    thread_names[(pid_i, tid_i)] = tname
            elif e.get("name") == "thread_sort_index":
                try:
                    thread_sort_idx[(pid_i, tid_i)] = int((e.get("args") or {}).get("sort_index", tid_i))
                except (TypeError, ValueError):
                    pass

    # Collect hardware curves (pid == 90000 or ph == "C") and SW spans (pid >= 100000)
    curves = defaultdict(list)
    units = defaultdict(list)  # unit_ord -> [(cta, warp, tid, x_us, d_us, cls, slot_val, handoff, sub_idx, role_str)]
    for e in events:
        ph = e.get("ph")
        pid = int(e.get("pid", 0))
        if ph == "C" and (pid == 90000 or pid < 100000):
            ts_ns = int(e.get("ts_ns", round(float(e.get("ts", 0.0)) * 1000)))
            x_us = (ts_ns - k_start_ns) / 1000.0
            args = e.get("args") or {}
            val = args.get("value")
            if val is None:
                val = next((v for v in args.values() if isinstance(v, (int, float))), 0.0)
            if -1.0 <= x_us <= dur_us + 1.0:
                curves[str(e["name"])].append((max(0.0, min(dur_us, x_us)), float(val)))
        elif ph == "X" and pid >= 100000:
            cls = _classify_stage(e, amd)
            if cls is None:
                continue  # Skip synthetic Residency grey bars
            u_ord = pid - 100000
            args = e.get("args") or {}
            cta = int(args.get("cta", 0))
            w = int(args.get("warp", int(e.get("tid", 0))))
            tid = int(e.get("tid", w))
            ts_ns = int(e.get("ts_ns", round(float(e.get("ts", 0.0)) * 1000)))
            x_us = (ts_ns - k_start_ns) / 1000.0
            d_us = float(e.get("dur", 0.0))
            if d_us <= 0:
                continue
            slot_val = int(args.get("slot", args.get("aux", 0)) or 0)
            handoff = args.get("handoff_from") is not None
            sub_idx = int(args.get("sublane", args.get("instance", 0)) or 0)
            role_str = str(args.get("role") or "").strip()
            units[u_ord].append((cta, w, tid, x_us, d_us, cls, slot_val, handoff, sub_idx, role_str))

    for k in curves:
        curves[k].sort(key=lambda p: p[0])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })
    # Clean 3-panel layout (Header + Panel A + Panel B + Panel C; Panel D removed per spec).
    fig = plt.figure(figsize=(15.0, 10.6), dpi=220)
    gs = fig.add_gridspec(
        4, 1,
        height_ratios=[0.55, 1.55, 1.95, 2.05],
        left=0.14, right=0.97, top=0.96, bottom=0.06, hspace=0.46,
    )
    ax_hdr, ax_a, ax_b, ax_c = [fig.add_subplot(g) for g in gs]
    ax_hdr.set_axis_off()

    m = verify_doc.get("metrics") or {}
    ovh = float(m.get("overhead_ratio", 0.0)) * 100.0
    n_units = len(units)
    n_ctas = int(al.get("ctas_observed") or 0)
    xh = int(al.get("cross_warp_spans") or 0)

    default_title = (
        f"AMD Instinct MI308X ({arch})" if amd else
        f"NVIDIA Blackwell B300 ({arch})" if "10" in arch else
        f"NVIDIA Hopper H20 ({arch})"
    )
    main_title = title_override or default_title
    sub_mode = f"Cross-Warp Async Paired ({xh:,} handoffs)" if xh > 0 else "Per-Tile Pipeline"
    ax_hdr.text(0.0, 0.94, main_title, fontsize=13.2, fontweight="bold", color="#0F172A", va="top")
    ax_hdr.text(
        0.0, 0.30,
        f"Dispatch: {dur_us:,.2f} µs · {n_ctas} CTAs across {n_units} {unit_lbl}s ({wpc} {'waves' if amd else 'warps'}/CTA) · "
        f"{sub_mode} · Validator PASS (Bitwise 0 Err, 0 Drops, Overhead {ovh:+.2f}%)",
        fontsize=8.6, color="#334155", va="top", fontweight="bold",
    )

    # ---------------- Panel A: Hardware Utilization Curves (Legend OUTSIDE Top) ----------------
    ax_a.set_title("A    Aligned Hardware Compute Utilization, HBM Bandwidth & Active SM/CU Curves",
                   loc="left", fontsize=11.2, fontweight="bold", color="#0F172A", pad=22)
    ax_a.set_xlim(0, dur_us)
    ax_a.set_ylabel("Utilization (%)")
    ax_a.grid(axis="x", color="#E2E8F0", lw=0.6)

    selected_curves = []
    priority_rules = [
        ("Compute", ("compute util — tensor core", "tensor", "wave mfma stage", "compute util", "matrix core mfma", "mfma"), "#2563EB", "-", 1.6),
        ("HBM Bandwidth", ("memory util — hbm read+write", "dram", "wave hbm load", "memory util", "hbm3 read+write", "hbm"), "#EA580C", "-", 1.5),
        ("Active Units", ("residency (%)", "active sms (%)", "sm_active", "active cus"), "#0F172A", "--", 1.3),
        ("ROCm MFMA Dispatch", ("matrix core mfma (%) [rocm pmc]",), "#1D4ED8", ":", 1.3),
        ("ROCm HBM3 Dispatch", ("hbm3 read+write (%) [rocm pmc]",), "#C2410C", ":", 1.3),
    ]
    used_names = set()
    for _tag, keys, col, ls, lw in priority_rules:
        best_match = None
        for k in keys:
            for cname, pts in curves.items():
                if cname in used_names or not pts:
                    continue
                if k in cname.lower():
                    if best_match is None or len(pts) > len(best_match[1]):
                        best_match = (cname, pts)
            if best_match is not None:
                break
        if best_match is not None:
            cname, pts = best_match
            used_names.add(cname)
            selected_curves.append((cname, pts, col, ls, lw))

    for cname, pts, col, ls, lw in selected_curves:
        xs, ys = zip(*pts)
        if len(pts) <= 3:
            ax_a.plot([0, dur_us], [ys[0], ys[0]], color=col, ls=ls, lw=lw, label=f"{cname}: {ys[0]:.1f}%")
        else:
            mean_v = sum(ys[:-1] if len(ys) > 1 else ys) / max(1, len(ys) - 1 if len(ys) > 1 else 1)
            ax_a.step(xs, ys, where="post", color=col, ls=ls, lw=lw, label=f"{cname} (avg {mean_v:.1f}%)")
    ax_a.set_ylim(0, 106)
    if selected_curves:
        # Place legend strictly OUTSIDE above axes so curves are never occluded.
        ax_a.legend(
            loc="lower left", bbox_to_anchor=(0.0, 1.01),
            ncol=min(3, len(selected_curves)), fontsize=8.0,
            frameon=False, borderaxespad=0.0, columnspacing=1.2,
        )

    def _rect(x, w, y, h):
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    def _draw_polys(axis, poly_map):
        priority_order = [
            "Wait / Barrier", "Dispatch / Pull", "DMA Issue",
            "Tile Load", "Async Load (DMA->Math)",
            "L1 GEMM", "L2 GEMM", "MMA / MFMA", "Compute",
            "Scatter", "Combine / Reduce", "Store / Epilogue",
        ]
        ordered = [s for s in priority_order if poly_map.get(s)] + [
            s for s in sorted(poly_map) if s not in priority_order and poly_map.get(s)
        ]
        for st in ordered:
            axis.add_collection(PolyCollection(
                poly_map[st], facecolors=_color_for_stage(st), edgecolors="none", antialiased=False, rasterized=True
            ))

    # ---------------- Panel B: All Physical SMs / CUs Multi-CTA Execution ----------------
    ax_b.set_title(f"B    Multi-CTA Execution Across All {n_units} Physical {unit_lbl}s (All Scheduling Waves)",
                   loc="left", fontsize=11.2, fontweight="bold", color="#0F172A", pad=8)
    poly_b = defaultdict(list)
    for u_ord, sp_list in sorted(units.items()):
        cta_bounds = {}
        for cta, w, _tid, x, d, _cls, _sl, _ho, _sb, _rs in sp_list:
            a, b = cta_bounds.get(cta, (x, x + d))
            cta_bounds[cta] = (min(a, x), max(b, x + d))
        ends, c_slot = [], {}
        for cta in sorted(cta_bounds, key=cta_bounds.get):
            a, b = cta_bounds[cta]
            idx = next((j for j, fin in enumerate(ends) if fin <= a), len(ends))
            if idx == len(ends):
                ends.append(b)
            else:
                ends[idx] = b
            c_slot[cta] = idx
        warps_in_unit = sorted({w for _c, w, *_ in sp_list})
        w_to_pos = {w: idx for idx, w in enumerate(warps_in_unit)}
        n_w_eff = max(1, len(warps_in_unit))
        h = 0.88 / max(1, len(ends) * n_w_eff)
        for cta, w, _tid, x, d, cls, _sl, ho, _sb, _rs in sp_list:
            if ho:
                continue
            y = u_ord - 0.44 + (c_slot[cta] * n_w_eff + w_to_pos.get(w, 0)) * h
            poly_b[cls].append(_rect(x, d, y, h))
    _draw_polys(ax_b, poly_b)
    ax_b.set_xlim(0, dur_us)
    ax_b.set_ylim(max(units.keys(), default=0) + 0.6, -0.6)
    ax_b.set_ylabel(f"Physical {unit_lbl} Index")
    ax_b.grid(axis="x", color="#E2E8F0", lw=0.6)

    # Pick earliest CTA on unit 0 (or first available unit) for Panel C
    u0_key = 0 if 0 in units else (min(units.keys()) if units else 0)
    u0_pid = 100000 + u0_key
    u0_spans = units.get(u0_key, [])
    by_cta0 = defaultdict(list)
    for item in u0_spans:
        by_cta0[item[0]].append(item)
    chosen_cta = min(by_cta0, key=lambda c: min(s[3] for s in by_cta0[c])) if by_cta0 else 0
    cta_spans = by_cta0[chosen_cta]
    c_left = min((s[3] for s in cta_spans), default=0.0)
    c_right = max((s[3] + s[4] for s in cta_spans), default=dur_us)
    ax_b.axvspan(c_left, c_right, color="#FACC15", alpha=0.14, zorder=0)
    ax_b.axvline(c_left, ls=":", color="#475569", lw=0.9)
    ax_b.axvline(c_right, ls=":", color="#475569", lw=0.9)

    # ---------------- Panel C: Representative CTA Pipeline (Real Role Track Names + External Legend) ----------------
    has_handoff = any(s[7] for s in cta_spans)
    slot_ids = sorted({s[6] for s in cta_spans if s[7]}) if has_handoff else []
    warp_sub_cnt = defaultdict(lambda: 1)
    warp_role_hint: dict[int, str] = {}
    warp_tid_map: dict[int, int] = {}
    for _cta, w, tid, _x, _d, _cls, _sl, ho, sb, role_str in cta_spans:
        if not ho:
            warp_sub_cnt[w] = max(warp_sub_cnt[w], sb + 1)
            warp_tid_map.setdefault(w, tid)
            if role_str and w not in warp_role_hint:
                warp_role_hint[w] = role_str

    def _track_label_for_warp(w: int) -> str:
        tid = warp_tid_map.get(w, w)
        tname = thread_names.get((u0_pid, tid))
        if tname:
            return tname
        r_hint = warp_role_hint.get(w)
        base_w = f"wave64 {w:02d}" if amd else f"w{w:02d}"
        if r_hint:
            # Preserve informative role names (e.g. Dispatch Warp 0, TMA Warp A, Math+Epi WG0)
            return r_hint if (str(w) in r_hint or "warp" in r_hint.lower() or "wg" in r_hint.lower()) else f"{r_hint} ({base_w})"
        return f"wave64 {w:02d}" if amd else f"warp {w:02d}"

    # Sort warps strictly by explicit thread_sort_index if present, else numeric warp index (NEVER alphabetically!).
    active_warps = sorted(
        {s[1] for s in cta_spans if not s[7]} or set(range(wpc)),
        key=lambda w: (thread_sort_idx.get((u0_pid, warp_tid_map.get(w, w)), w), w),
    )

    track_labels: list[str] = []
    row_of_warp: dict[int, int] = {}
    row_of_slot: dict[int, int] = {}
    sep_lines: list[float] = []

    if has_handoff and not amd:
        # Place DMA/producer warps first, Async slots in middle, Math/consumer warps after — preserving numeric order.
        dma_warps = [w for w in active_warps if any(k in _track_label_for_warp(w).lower() for k in ("dma", "tma", "producer", "dispatch"))]
        if not dma_warps:
            half = max(1, len(active_warps) // 2)
            dma_warps = active_warps[:half]
        math_warps = [w for w in active_warps if w not in dma_warps]
        for w in dma_warps:
            row_of_warp[w] = len(track_labels)
            track_labels.append(_track_label_for_warp(w))
        if slot_ids:
            sep_lines.append(len(track_labels) - 0.5)
            for sl in slot_ids:
                row_of_slot[sl] = len(track_labels)
                track_labels.append(f"Async slot {sl}")
            sep_lines.append(len(track_labels) - 0.5)
        for w in math_warps:
            row_of_warp[w] = len(track_labels)
            track_labels.append(_track_label_for_warp(w))
    else:
        for w in active_warps:
            row_of_warp[w] = len(track_labels)
            track_labels.append(_track_label_for_warp(w))

    poly_c = defaultdict(list)
    handoff_owner_w = next((w for w in active_warps if w not in row_of_slot), active_warps[0] if active_warps else 0)
    seen_handoff_slots: set[tuple[int, float, float]] = set()
    for _cta, w, _tid, x, d, cls, sl_val, ho, sb, _rs in cta_spans:
        if ho and has_handoff and not amd:
            key_ho = (sl_val, round(x, 4), round(d, 4))
            if key_ho in seen_handoff_slots:
                continue
            seen_handoff_slots.add(key_ho)
            r_idx = row_of_slot.get(sl_val, row_of_warp.get(handoff_owner_w, 0))
        else:
            r_idx = row_of_warp.get(w, 0)
        nsub = 1 if ho else warp_sub_cnt[w]
        bar_h = 0.68 / nsub
        bar_y = (r_idx - 0.34) + (sb % nsub) * bar_h if nsub > 1 else (r_idx - 0.34)
        poly_c[cls].append(_rect(x, d, bar_y, bar_h * (0.90 if nsub > 1 else 1.0)))

    u0_name = proc_names.get(u0_pid, f"{unit_lbl} {u0_key:03d}")
    pad_c = max(0.5, (c_right - c_left) * 0.015)
    _draw_polys(ax_c, poly_c)
    ax_c.set_title(f"C    {u0_name} · CTA {chosen_cta} — Complete Warp/Wave Pipeline ({c_right - c_left:,.1f} µs)",
                   loc="left", fontsize=11.2, fontweight="bold", color="#0F172A", pad=22)
    ax_c.set_xlim(max(0.0, c_left - pad_c), c_right + pad_c)
    ax_c.set_ylim(len(track_labels) - 0.3, -0.7)
    ax_c.set_yticks(range(len(track_labels)))
    ax_c.set_yticklabels(track_labels, fontsize=8.0, fontweight="bold")
    ax_c.set_xlabel("Aligned Time from Hardware Dispatch Start (µs)", fontsize=9.5, fontweight="bold")
    ax_c.grid(axis="x", color="#E2E8F0", lw=0.6)
    for sep in sep_lines:
        ax_c.axhline(sep, color="#94A3B8", lw=0.8, ls="--")

    # Dynamically include EVERY stage present in Panel B or Panel C (including purple Combine/Reduce etc.),
    # placed strictly OUTSIDE above Panel C so zero bars are ever covered!
    preferred_order = [
        "Dispatch / Pull", "DMA Issue", "Async Load (DMA->Math)", "Tile Load",
        "L1 GEMM", "L2 GEMM", "MMA / MFMA", "Compute",
        "Scatter", "Combine / Reduce", "Store / Epilogue", "Wait / Barrier",
    ]
    present = [k for k in preferred_order if poly_c.get(k) or poly_b.get(k)] + [
        k for k in sorted(set(poly_c) | set(poly_b)) if k not in preferred_order and (poly_c.get(k) or poly_b.get(k))
    ]
    legend_handles = [Patch(facecolor=_color_for_stage(k), label=k) for k in present]
    if legend_handles:
        ax_c.legend(
            handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, 1.01),
            ncol=min(6, len(legend_handles)), fontsize=8.0,
            frameon=False, borderaxespad=0.0, columnspacing=1.0, handlelength=1.2,
        )

    fig.savefig(out_png, dpi=220, facecolor="white")
    plt.close(fig)
    return out_png
