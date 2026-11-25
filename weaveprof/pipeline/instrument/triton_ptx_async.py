# -*- coding: utf-8 -*-
"""Deterministic PTX-level async pipeline instrumentation for NVIDIA Triton kernels.

Triton's software pipeliner lowers ``tl.load`` into asynchronous ``cp.async`` / TMA
operations and automatically inserts ``cp.async.wait_group N; bar.sync 0;`` right
before ``tl.dot`` — invisible in Python source.  Wrapping ``tl.load`` in Python only
measures ~40 ns of issue latency and lumps the real exposed wait into ``tl.dot``.

This module instruments the compiled PTX directly (zero disruption to Triton's
software pipeliner and zero Python K-loop edits) using hardware FIFO semantics:
  * ``cp.async.commit_group`` -> ``WP_B(EV_LOAD, seq_in++)`` (in-flight launch)
  * ``cp.async.wait_group N [+ bar.sync 0]`` ->
      ``WP_B(EV_WAIT, seq_out)`` before wait,
      ``WP_E(EV_WAIT, seq_out)`` after wait,
      ``WP_E(EV_LOAD, seq_out++)`` for every group completed (``seq_out < seq_in - N``)
  * ``wgmma.commit_group`` / ``wgmma.wait_group N`` (Hopper ``sm_90``) ->
      same exact FIFO tracking on ``EV_MMA`` (``async_engine="mma"``)
  * Contiguous ``mma.sync`` clusters (Ampere/Ada) -> ``WP_B(EV_MMA)`` / ``WP_E(EV_MMA)``

All injected records append through a single live PTX cursor register ``%wp_cur``
synchronized into subsequent ``wp_b`` / ``wp_e`` / ``wp_close`` blocks, emitting
standard ``FILITE02`` / ``WPLITE02`` binary dumps decoded directly by ``wp_decode.py``.
"""
from __future__ import annotations

import contextlib
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

EV_LOAD = 1
EV_WAIT = 2
EV_WAIT_LOAD = 2
EV_MMA = 3
EV_WAIT_MMA = 4

_TAG_OPEN = 0x40000000
_TAG_CLOSE = 0x08000000
_TAG_END = 0x80000000
_TAG_CLK = 0x10000000


def _emit_rec_ptx(
    tag_imm: int,
    aux_reg: str,
    *,
    cap: int,
    ovf_reg: str,
    timer_clk: str,
    reuse_ts: bool = False,
) -> List[str]:
    """Emit a minimal, lane-0 predicated 16-byte FILITE02 record append in PTX (zero hot-loop atomics)."""
    out: List[str] = []
    out.append(f"\tsetp.lt.u32\t%wp_ok, %wp_cur, {cap};")
    out.append("\tand.pred\t%wp_ok, %wp_ok, %wp_rec;")
    if not reuse_ts:
        out.append(f"\t@%wp_ok mov.u64\t%wp_ts, {timer_clk};")
        out.append("\t@%wp_ok mov.b64\t{%wp_tlo, %wp_thi}, %wp_ts;")
    out.append("\t@%wp_ok mul.wide.u32\t%wp_off, %wp_cur, 16;")
    out.append("\t@%wp_ok add.u64\t%wp_addr, %wp_base, %wp_off;")
    out.append(f"\t@%wp_ok st.global.cs.v4.u32\t[%wp_addr], {{%wp_tlo, %wp_thi, {tag_imm}, {aux_reg}}};")
    out.append("\tadd.u32\t%wp_cur, %wp_cur, 1;")
    return out


def instrument_triton_ptx_async(
    ptx: str,
    *,
    instrument_mma_sync: bool = True,
    instrument_wgmma_async: bool = True,
) -> Tuple[str, Dict[str, int]]:
    """Transform Triton-compiled PTX containing ``wp_open``/``wp_close`` with exact async probes.

    Returns ``(patched_ptx, stats)``.
    """
    lines = ptx.splitlines()

    # 1) Parse recorder metadata from the emitted `wp_open` inline-asm block
    open_store_idx = next(
        (i for i, ln in enumerate(lines) if f", {_TAG_OPEN}," in ln and "st.global.cs.v4.u32" in ln),
        None,
    )
    if open_store_idx is None:
        return ptx, {"patched": 0, "reason": "no_wp_open_found"}

    # Locate the enclosing inline-asm block of wp_open
    open_start = open_store_idx
    while open_start > 0 and "// begin inline asm" not in lines[open_start]:
        open_start -= 1
    open_end = open_store_idx
    while open_end < len(lines) - 1 and "// end inline asm" not in lines[open_end]:
        open_end += 1

    open_block = "\n".join(lines[open_start : open_end + 1])
    m_rec = re.search(r"add\.u64\s+addr,\s*(%rd\d+),\s*off;", open_block)
    m_ovf = re.search(r"atom\.global\.or\.b32\s+ovfold,\s*\[(%rd\d+)\],\s*1;", open_block)
    m_cap = re.search(r"setp\.lt\.u32\s+p_cap,\s*n,\s*(\d+);", open_block)
    m_warps = re.search(r"setp\.lt\.u32\s+p_bounds,\s*warp,\s*(\d+);", open_block)
    m_nslots = re.search(r"setp\.lt\.u32\s+p_cap,\s*slot,\s*(\d+);", open_block)
    m_pid = re.search(r"mov\.u32\s+pid,\s*(%r\d+);", open_block)
    has_clk = f", {_TAG_CLK}," in open_block

    if not (m_rec and m_ovf and m_cap and m_warps and m_nslots and m_pid):
        return ptx, {"patched": 0, "reason": "unrecognized_wp_open_layout"}

    rec_reg = m_rec.group(1)
    ovf_reg = m_ovf.group(1)
    cap = int(m_cap.group(1))
    warps = int(m_warps.group(1))
    n_slots = int(m_nslots.group(1))
    pid_reg = m_pid.group(1)
    timer_clk = "%clock64" if has_clk else "%globaltimer"
    open_nr = 2 if has_clk else 1

    # 2) Find insertion point for kernel-level PTX virtual register declarations
    last_reg_decl = next(
        (i for i in range(open_start, -1, -1) if lines[i].strip().startswith(".reg ")),
        None,
    )
    if last_reg_decl is None:
        return ptx, {"patched": 0, "reason": "no_reg_decls"}

    # Identify all subsequent wp_* inline-asm blocks after open_end so we rebind their `n` to `%wp_cur`
    wp_blocks: List[Tuple[int, int, int]] = []  # (start_idx, end_idx, nr_records)
    idx = open_end + 1
    while idx < len(lines):
        if "// begin inline asm" in lines[idx]:
            j = idx + 1
            while j < len(lines) and "// end inline asm" not in lines[j]:
                j += 1
            blk = "\n".join(lines[idx : j + 1])
            if "st.global.cs.v4.u32" in blk and "mov.u32 n," in blk:
                nr = 2 if (f", {_TAG_CLOSE}," in blk and f", {_TAG_CLK}," in blk) else blk.count("st.global.cs.v4.u32")
                wp_blocks.append((idx, j, max(1, nr)))
            idx = j + 1
        else:
            idx += 1

    wp_block_map = {s: (e, nr) for s, e, nr in wp_blocks}

    stats = {
        "patched": 1,
        "cp_async_commits": 0,
        "cp_async_waits": 0,
        "wgmma_commits": 0,
        "wgmma_waits": 0,
        "mma_clusters": 0,
        "rebound_wp_blocks": len(wp_blocks),
    }

    out: List[str] = []
    drain_uid = 0
    in_mma_cluster = False

    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        # Insert virtual register declarations right after the last `.reg` declaration
        if i == last_reg_decl:
            out.append(ln)
            out.extend([
                "\t// [WeaveProfiler PTX async pass] virtual registers",
                "\t.reg .pred\t%wp_rec, %wp_ok, %wp_ovf_p, %wp_loop, %wp_b1, %wp_b2;",
                "\t.reg .u32\t%wp_cur, %wp_in, %wp_out, %wp_win, %wp_wout, %wp_tgt, %wp_tlo, %wp_thi, %wp_tmp, %wp_lane, %wp_tid, %wp_warp, %wp_slot;",
                "\t.reg .u64\t%wp_base, %wp_off, %wp_addr, %wp_ts;",
            ])
            i += 1
            continue

        # Immediately after `wp_open`, initialize %wp_base, %wp_rec, %wp_cur, FIFO counters
        if i == open_end:
            out.append(ln)
            stride_bytes = cap * 16
            out.extend([
                "\t// [WeaveProfiler PTX async pass] init per-warp base & FIFO counters",
                "\tmov.u32\t%wp_lane, %laneid;",
                "\tmov.u32\t%wp_tid, %tid.x;",
                "\tshr.u32\t%wp_warp, %wp_tid, 5;",
                f"\tmul.lo.u32\t%wp_slot, {pid_reg}, {warps};",
                "\tadd.u32\t%wp_slot, %wp_slot, %wp_warp;",
                "\tsetp.eq.u32\t%wp_rec, %wp_lane, 0;",
                f"\tsetp.lt.u32\t%wp_b1, %wp_warp, {warps};",
                f"\tsetp.lt.u32\t%wp_b2, %wp_slot, {n_slots};",
                "\tand.pred\t%wp_b1, %wp_b1, %wp_b2;",
                "\tand.pred\t%wp_rec, %wp_rec, %wp_b1;",
                f"\tmul.wide.u32\t%wp_off, %wp_slot, {stride_bytes};",
                f"\tadd.u64\t%wp_base, {rec_reg}, %wp_off;",
                f"\tmov.u32\t%wp_cur, {open_nr};",
                "\tmov.u32\t%wp_in, 0;",
                "\tmov.u32\t%wp_out, 0;",
                "\tmov.u32\t%wp_win, 0;",
                "\tmov.u32\t%wp_wout, 0;",
            ])
            i += 1
            continue

        # Before open_end, pass untouched
        if i < open_end:
            out.append(ln)
            i += 1
            continue

        # Close an active mma.sync cluster before any barrier / async / branch / label / wp block
        if in_mma_cluster and (
            s.startswith("$L__")
            or s.startswith("@")
            or s.startswith("bra")
            or s.startswith("bar.sync")
            or s.startswith("cp.async")
            or s.startswith("wgmma.")
            or i in wp_block_map
            or s.startswith("ret;")
        ):
            out.extend(_emit_rec_ptx(_TAG_END | EV_MMA, "0", cap=cap, ovf_reg=ovf_reg, timer_clk=timer_clk))
            in_mma_cluster = False

        # Rebind existing wp_b / wp_e / wp_close inline-asm blocks to live `%wp_cur`
        if i in wp_block_map:
            end_j, nr = wp_block_map[i]
            for k in range(i, end_j + 1):
                blk_ln = lines[k]
                if re.search(r"\bmov\.u32\s+n,\s*%r\d+;", blk_ln):
                    blk_ln = re.sub(r"\bmov\.u32\s+n,\s*%r\d+;", "mov.u32 n, %wp_cur;", blk_ln)
                out.append(blk_ln)
            out.append(f"\tadd.u32\t%wp_cur, %wp_cur, {nr};")
            i = end_j + 1
            continue

        # 1) cp.async.commit_group -> launch one async memory group (`EV_LOAD`, seq=%wp_in++)
        if s == "cp.async.commit_group;":
            out.append(ln)
            out.extend(_emit_rec_ptx(EV_LOAD, "%wp_in", cap=cap, ovf_reg=ovf_reg, timer_clk=timer_clk))
            out.append("\tadd.u32\t%wp_in, %wp_in, 1;")
            stats["cp_async_commits"] += 1
            i += 1
            continue

        # 2) cp.async.wait_group N [+ adjacent duplicate wait_group + bar.sync 0]
        m_wg = re.match(r"^cp\.async\.wait_group\s+(\d+)\s*;$", s)
        if m_wg:
            wait_n = int(m_wg.group(1))
            j = i + 1
            # Consume consecutive cp.async.wait_group / .loc / blank lines + immediate bar.sync 0
            while j < len(lines):
                sj = lines[j].strip()
                if not sj or sj.startswith(".loc") or sj.startswith("//"):
                    j += 1
                    continue
                m_wg2 = re.match(r"^cp\.async\.wait_group\s+(\d+)\s*;$", sj)
                if m_wg2:
                    wait_n = min(wait_n, int(m_wg2.group(1)))
                    j += 1
                    continue
                if re.match(r"^bar\.sync\s+0\s*;$", sj):
                    j += 1
                break
            # Trim trailing .loc/blank lines that weren't followed by wait/bar
            while j > i + 1 and (not lines[j - 1].strip() or lines[j - 1].strip().startswith(".loc")):
                j -= 1

            # Emit WP_B(EV_WAIT, %wp_out) BEFORE wait_group + bar.sync
            out.extend(_emit_rec_ptx(EV_WAIT, "%wp_out", cap=cap, ovf_reg=ovf_reg, timer_clk=timer_clk))
            out.extend(lines[i:j])
            # Emit WP_E(EV_WAIT, %wp_out) AFTER wait_group + bar.sync (captures %wp_ts once!)
            out.extend(_emit_rec_ptx(_TAG_END | EV_WAIT, "%wp_out", cap=cap, ovf_reg=ovf_reg, timer_clk=timer_clk))
            # Drain all completed cp.async groups: while (%wp_out < %wp_in - wait_n) emit WP_E(EV_LOAD, %wp_out++)
            drain_uid += 1
            lbl_loop = f"$L__WP_DRAIN_{drain_uid}"
            lbl_done = f"$L__WP_DONE_{drain_uid}"
            out.append(f"\tsub.s32\t%wp_tgt, %wp_in, {wait_n};")
            out.append(f"{lbl_loop}:")
            out.append("\tsetp.lt.s32\t%wp_loop, %wp_out, %wp_tgt;")
            out.append(f"\t@!%wp_loop bra\t{lbl_done};")
            out.extend(_emit_rec_ptx(_TAG_END | EV_LOAD, "%wp_out", cap=cap, ovf_reg=ovf_reg, timer_clk=timer_clk, reuse_ts=True))
            out.append("\tadd.u32\t%wp_out, %wp_out, 1;")
            out.append(f"\tbra.uni\t{lbl_loop};")
            out.append(f"{lbl_done}:")
            stats["cp_async_waits"] += 1
            i = j
            continue

        # 3) Hopper wgmma.commit_group / wgmma.wait_group N -> exact MMA async FIFO tracking!
        if instrument_wgmma_async and s.startswith("wgmma.commit_group"):
            out.append(ln)
            out.extend(_emit_rec_ptx(EV_MMA, "%wp_win", cap=cap, ovf_reg=ovf_reg, timer_clk=timer_clk))
            out.append("\tadd.u32\t%wp_win, %wp_win, 1;")
            stats["wgmma_commits"] += 1
            i += 1
            continue

        m_wwg = re.match(r"^wgmma\.wait_group\.sync\.aligned\s+(\d+)\s*;$", s) if instrument_wgmma_async else None
        if m_wwg:
            wait_n = int(m_wwg.group(1))
            out.extend(_emit_rec_ptx(EV_WAIT_MMA, "%wp_wout", cap=cap, ovf_reg=ovf_reg, timer_clk=timer_clk))
            out.append(ln)
            out.extend(_emit_rec_ptx(_TAG_END | EV_WAIT_MMA, "%wp_wout", cap=cap, ovf_reg=ovf_reg, timer_clk=timer_clk))
            drain_uid += 1
            lbl_loop = f"$L__WP_WDRAIN_{drain_uid}"
            lbl_done = f"$L__WP_WDONE_{drain_uid}"
            out.append(f"\tsub.s32\t%wp_tgt, %wp_win, {wait_n};")
            out.append(f"{lbl_loop}:")
            out.append("\tsetp.lt.s32\t%wp_loop, %wp_wout, %wp_tgt;")
            out.append(f"\t@!%wp_loop bra\t{lbl_done};")
            out.extend(_emit_rec_ptx(_TAG_END | EV_MMA, "%wp_wout", cap=cap, ovf_reg=ovf_reg, timer_clk=timer_clk, reuse_ts=True))
            out.append("\tadd.u32\t%wp_wout, %wp_wout, 1;")
            out.append(f"\tbra.uni\t{lbl_loop};")
            out.append(f"{lbl_done}:")
            stats["wgmma_waits"] += 1
            i += 1
            continue

        # 4) Contiguous Ampere/Ada mma.sync cluster -> bracket with EV_MMA
        if instrument_mma_sync and s.startswith("mma.sync."):
            if not in_mma_cluster:
                out.extend(_emit_rec_ptx(EV_MMA, "0", cap=cap, ovf_reg=ovf_reg, timer_clk=timer_clk))
                in_mma_cluster = True
                stats["mma_clusters"] += 1
            out.append(ln)
            i += 1
            continue

        out.append(ln)
        i += 1

    if in_mma_cluster:
        out.extend(_emit_rec_ptx(_TAG_END | EV_MMA, "0", cap=cap, ovf_reg=ovf_reg, timer_clk=timer_clk))

    return "\n".join(out) + "\n", stats


@contextlib.contextmanager
def triton_ptx_async_hook(
    stats_out: Optional[Dict[str, int]] = None,
    *,
    instrument_mma_sync: bool = True,
    instrument_wgmma_async: bool = True,
) -> Iterator[None]:
    """Context manager hooking Triton's NVIDIA PTX->cubin stage to inject exact async probes."""
    from triton.backends.nvidia.compiler import CUDABackend

    orig_make_cubin = CUDABackend.make_cubin

    def _patched_make_cubin(self, src, metadata, options, capability):
        if isinstance(src, str) and f", {_TAG_OPEN}," in src:
            patched_ptx, st = instrument_triton_ptx_async(
                src,
                instrument_mma_sync=instrument_mma_sync,
                instrument_wgmma_async=instrument_wgmma_async,
            )
            if stats_out is not None:
                stats_out.clear()
                stats_out.update(st)
            src = patched_ptx
        return orig_make_cubin(self, src, metadata, options, capability)

    CUDABackend.make_cubin = _patched_make_cubin
    try:
        yield
    finally:
        CUDABackend.make_cubin = orig_make_cubin


def build_triton_ptx_async_names(
    stats: Optional[Mapping[str, Any]] = None,
    *,
    has_wgmma_async: Optional[bool] = None,
    store_label: str = "C Tile Store (HBM)",
) -> Dict[str, Any]:
    """Build a canonical `names.json` dict for decoding Triton PTX-async dumps.

    Automatically registers Hopper `wgmma` (`ev=3`) as a FIFO slotted async MMA engine
    when ``stats.get('wgmma_commits', 0) > 0`` (or ``has_wgmma_async=True``).
    """
    use_wgmma = bool(has_wgmma_async) if has_wgmma_async is not None else bool((stats or {}).get("wgmma_commits", 0) > 0)
    slotted: Dict[str, Any] = {
        "1": {"id": "cp_async_tile_load", "slot": "seq", "cross_warp": False},
    }
    async_engines: Dict[str, str] = {"1": "mem"}
    if use_wgmma:
        slotted["3"] = {"id": "wgmma_async", "slot": "seq", "cross_warp": False}
        async_engines["3"] = "mma"
    events_map: Dict[str, str] = {
        "1": "cp_async_tile_load",
        "2": "async_wait_group",
        "3": "wgmma_async" if use_wgmma else "mma_sync_hmma",
        "6": "tl_store_c",
    }
    labels_map: Dict[str, str] = {
        "1": "Triton Async Load (cp.async/TMA FIFO)",
        "2": "Exposed Load Wait (cp.async.wait_group + bar.sync)",
        "3": "Tensor Core WGMMA (async FIFO)" if use_wgmma else "Tensor Core MMA",
        "6": store_label,
    }
    sem_map: Dict[str, str] = {"1": "load", "2": "wait", "3": "mma", "6": "store"}
    if use_wgmma:
        events_map["4"] = "wgmma_wait_group"
        labels_map["4"] = "Exposed MMA Wait (wgmma.wait_group)"
        sem_map["4"] = "wait"
    return {
        "events": events_map,
        "labels": labels_map,
        "semantics": sem_map,
        "slotted_events": slotted,
        "async_engines": async_engines,
        "roles": {"0": "cta_worker"},
        "group": "sm",
        "pair": "stack",
    }
