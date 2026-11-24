# -*- coding: utf-8 -*-
"""Deterministic AMDGCN-level async pipeline instrumentation for AMD ROCm Triton kernels.

ROCm Triton's software pipeliner lowers ``tl.load`` into asynchronous vector memory
loads (``buffer_load_dwordx4`` / ``global_load_dwordx4`` / ``buffer_load_* lds``) and
automatically inserts ``s_waitcnt vmcnt(N)`` right before LDS commit or ``v_mfma_*``
consumption — invisible in Python source.  Wrapping ``tl.load`` in Python only measures
non-blocking VMEM issue latency and lumps ``s_waitcnt vmcnt(N)`` stalls into downstream ops.

This module instruments compiled ``.amdgcn`` assembly emitted by ``HIPBackend.make_amdgcn``
when ``wp_open_16b`` / ``wp_close_16b`` (``recorders/triton/wp_triton_amd.py``) is present:
  * Contiguous ``buffer_load_*`` / ``global_load_*`` tile-prefetch bursts ->
      ``WP_B(EV_LOAD, seq_in++)`` (async VMEM group launch, ``async_engine="mem"``)
  * ``s_waitcnt vmcnt(N)`` synchronization clusters ->
      ``WP_B(EV_WAIT, seq_out)`` before wait,
      ``WP_E(EV_WAIT, seq_out)`` after wait,
      ``WP_E(EV_LOAD, seq_out++)`` retiring completed VMEM groups in FIFO order
  * Contiguous ``v_mfma_*`` Matrix Core clusters ->
      ``WP_B(EV_MMA, mma_seq)`` / ``WP_E(EV_MMA, mma_seq++)``

Crucially, on AMD CDNA3 (``gfx942`` / MI308X):
  * ``s_memtime`` / ``s_memrealtime`` use ``lgkmcnt`` (scalar RTC), NOT ``vmcnt``.
  * ``buffer_store_dwordx4`` / ``global_store_dwordx4`` on CDNA3 increment ``vscnt``
    (the separate vector store counter), NOT ``vmcnt``!
  * Live ``SCC`` across scalar comparisons/branches is explicitly preserved around every injected block.
Therefore, streaming 16-byte WeaveProfiler records NEVER perturbs outstanding ``vmcnt`` counts or ``SCC``!
"""
from __future__ import annotations

import contextlib
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

EV_LOAD = 1
EV_WAIT = 2
EV_MMA = 3

_TAG_END = 0x80000000


def _emit_rec_amdgcn(
    *,
    tag_imm: int,
    aux_src: str,
    cap_bytes: int,
    s_pred_lo: int,
    s_our_srd: int,
    s_base_byte: int,
    s_cur_byte: int,
    s_ts_lo: int,
    s_tmp: int,
    s_step: int,
    v_base: int,
    reuse_ts: bool = False,
) -> List[str]:
    """Emit a branch-free, lane-0 predicated 16-byte FILITE02 record append in AMDGCN (caller saves SCC)."""
    v0, v1, v2, v3 = v_base, v_base + 1, v_base + 2, v_base + 3
    v_off, v_oob = v_base + 4, v_base + 5
    s_pred_pair = f"s[{s_pred_lo}:{s_pred_lo + 1}]"
    s_ts_pair = f"s[{s_ts_lo}:{s_ts_lo + 1}]"
    s_srd_quad = f"s[{s_our_srd}:{s_our_srd + 3}]"
    out: List[str] = []
    if not reuse_ts:
        out.append(f"\ts_memtime {s_ts_pair}")
        out.append("\ts_waitcnt lgkmcnt(0)")
    out.append(f"\tv_mov_b32_e32 v{v0}, s{s_ts_lo}")
    out.append(f"\tv_mov_b32_e32 v{v1}, s{s_ts_lo + 1}")
    out.append(f"\tv_mov_b32_e32 v{v2}, {hex(tag_imm & 0xFFFFFFFF)}")
    out.append(f"\tv_mov_b32_e32 v{v3}, {aux_src}")
    out.append(f"\ts_add_u32 s{s_tmp}, s{s_base_byte}, s{s_cur_byte}")
    out.append(f"\tv_mov_b32_e32 v{v_off}, s{s_tmp}")
    out.append(f"\tv_mov_b32_e32 v{v_oob}, 0x80000000")
    out.append(f"\ts_cmp_lt_u32 s{s_cur_byte}, {cap_bytes}")
    out.append(f"\ts_cselect_b32 s{s_step}, 16, 0")
    out.append(f"\ts_cselect_b32 s{s_tmp}, 0, 0x80000000")
    out.append(f"\tv_cndmask_b32_e64 v{v_off}, v{v_oob}, v{v_off}, {s_pred_pair}")
    out.append(f"\tv_or_b32_e32 v{v_off}, s{s_tmp}, v{v_off}")
    out.append(f"\tbuffer_store_dwordx4 v[{v0}:{v3}], v{v_off}, {s_srd_quad}, 0 offen")
    out.append(f"\ts_add_u32 s{s_cur_byte}, s{s_cur_byte}, s{s_step}")
    return out


def instrument_triton_amdgcn_async(
    amdgcn: str,
    *,
    cap: int = 256,
    max_hot_events: Optional[int] = None,
    instrument_mfma: bool = True,
) -> Tuple[str, Dict[str, int]]:
    """Transform ROCm Triton `.amdgcn` containing ``wp_open_16b``/``wp_close_16b`` with CDNA3 async probes."""
    lines = amdgcn.splitlines()

    memrt_indices = [i for i, ln in enumerate(lines) if "s_memrealtime " in ln]
    if len(memrt_indices) < 2:
        return amdgcn, {"patched": 0, "reason": "no_wp_open_close_found"}

    first_memrt = memrt_indices[0]
    last_memrt = memrt_indices[-1]

    open_end = None
    s_base_byte = None
    s_pred_lo = None
    s_orig_srd = None
    for idx in range(first_memrt, min(len(lines), first_memrt + 160)):
        m_or = re.match(r"^\s*s_or_b32\s+s\d+,\s*s(\d+),\s*16\s*$", lines[idx])
        if m_or:
            s_base_byte = int(m_or.group(1))
        m_cnd = re.match(r"^\s*v_cndmask_b32_e64\s+v\d+,\s*v\d+,\s*v\d+,\s*s\[(\d+):(\d+)\]\s*$", lines[idx])
        if m_cnd:
            s_pred_lo = int(m_cnd.group(1))
        m_bst = re.match(
            r"^\s*buffer_store_dwordx4\s+v\[\d+:\d+\],\s*v\d+,\s*s\[(\d+):(\d+)\],\s*0\s+offen\s*$",
            lines[idx],
        )
        if m_bst and s_base_byte is not None:
            s_orig_srd = int(m_bst.group(1))
            open_end = idx
            break

    if open_end is None or s_base_byte is None or s_pred_lo is None or s_orig_srd is None:
        return amdgcn, {"patched": 0, "reason": "unrecognized_wp_open_16b_amdgcn"}

    epilogue_start = last_memrt
    for idx in range(open_end + 1, last_memrt):
        if re.match(rf"^\s*s_add_i32\s+s\d+,\s*s{s_base_byte},\s*32\s*$", lines[idx]):
            epilogue_start = idx
            for back in range(idx - 1, max(open_end + 1, idx - 24), -1):
                if "s_memtime " in lines[back] or ";;#ASMSTART" in lines[back]:
                    epilogue_start = back
            break

    m_vgpr = re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", amdgcn)
    m_num_sgpr = re.search(r"\.set\s+\S+\.numbered_sgpr,\s*(\d+)", amdgcn)
    if not (m_vgpr and m_num_sgpr):
        return amdgcn, {"patched": 0, "reason": "missing_amdhsa_gpr_directives"}

    orig_vgpr = int(m_vgpr.group(1))
    orig_sgpr = int(m_num_sgpr.group(1))

    v_base = (orig_vgpr + 3) & ~3  # 4-aligned for `v[v0:v3]`
    new_vgpr = v_base + 6
    s_our_srd = (orig_sgpr + 3) & ~3  # 4-aligned private SRD `s[s_our_srd : s_our_srd + 3]`!
    s_ts_lo = s_our_srd + 4           # 2-aligned for `s_memtime s[s_ts_lo : s_ts_lo + 1]`
    s_cur_byte = s_our_srd + 6
    s_seq_in = s_our_srd + 7
    s_seq_out = s_our_srd + 8
    s_mma_seq = s_our_srd + 9
    s_tmp = s_our_srd + 10
    s_step = s_our_srd + 11
    s_scc = s_our_srd + 12
    new_numbered_sgpr = s_our_srd + 14
    if new_vgpr > 252 or new_numbered_sgpr > 94:
        return amdgcn, {"patched": 0, "reason": "insufficient_free_gprs"}

    hot_cap_bytes = min(cap - 6, max_hot_events) * 16 if max_hot_events is not None else (cap - 6) * 16

    def with_scc_guard(body: List[str]) -> List[str]:
        return [f"\ts_cselect_b32 s{s_scc}, 1, 0", *body, f"\ts_cmp_lg_u32 s{s_scc}, 0"]

    def emit_rec(tag_imm: int, aux_src: str, *, reuse_ts: bool = False) -> List[str]:
        return _emit_rec_amdgcn(
            tag_imm=tag_imm,
            aux_src=aux_src,
            cap_bytes=hot_cap_bytes,
            s_pred_lo=s_pred_lo,
            s_our_srd=s_our_srd,
            s_base_byte=s_base_byte,
            s_cur_byte=s_cur_byte,
            s_ts_lo=s_ts_lo,
            s_tmp=s_tmp,
            s_step=s_step,
            v_base=v_base,
            reuse_ts=reuse_ts,
        )

    stats = {
        "patched": 1,
        "vmem_load_groups": 0,
        "vmem_wait_clusters": 0,
        "mfma_clusters": 0,
        "rebound_epilogue_offsets": 0,
        "orig_vgpr": orig_vgpr,
        "new_vgpr": new_vgpr,
        "orig_sgpr": orig_sgpr,
        "new_sgpr": new_numbered_sgpr,
    }

    out: List[str] = []
    in_mfma = False
    in_vmem_burst = False
    vmem_insts_in_cur_group = 0
    insts_per_group_est = 0
    drain_uid = 0
    epilogue_shifted = False

    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if s.startswith(".amdhsa_next_free_vgpr "):
            out.append(re.sub(r"\d+", str(new_vgpr), ln))
            i += 1
            continue
        if s.startswith(".amdhsa_accum_offset "):
            cur_acc = int(re.search(r"\d+", s).group(0))
            new_acc = max(cur_acc, (new_vgpr + 3) & ~3)
            out.append(re.sub(r"\d+", str(new_acc), ln))
            i += 1
            continue
        if re.match(r"^\.set\s+\S+\.num_vgpr,\s*\d+", s):
            out.append(re.sub(r"\d+$", str(new_vgpr), ln))
            i += 1
            continue
        if re.match(r"^\.set\s+\S+\.numbered_sgpr,\s*\d+", s):
            out.append(re.sub(r"\d+$", str(new_numbered_sgpr), ln))
            i += 1
            continue
        if re.match(r"^\s*\.vgpr_count:\s*\d+", ln):
            out.append(re.sub(r"\d+", str(new_vgpr), ln))
            i += 1
            continue
        if re.match(r"^\s*\.sgpr_count:\s*\d+", ln):
            cur_sc = int(re.search(r"\d+", s).group(0))
            out.append(re.sub(r"\d+", str(max(cur_sc, new_numbered_sgpr + 6)), ln))
            i += 1
            continue

        if i == open_end:
            out.append(ln)
            out.extend([
                "\t; [WeaveProfiler AMDGCN async pass] init private SRD, cursor & FIFO counters",
                f"\ts_mov_b64 s[{s_our_srd}:{s_our_srd + 1}], s[{s_orig_srd}:{s_orig_srd + 1}]",
                f"\ts_mov_b32 s{s_our_srd + 2}, 0x7ffffffe",
                f"\ts_mov_b32 s{s_our_srd + 3}, 0x27000",
                f"\ts_mov_b32 s{s_cur_byte}, 32",
                f"\ts_mov_b32 s{s_seq_in}, 0",
                f"\ts_mov_b32 s{s_seq_out}, 0",
                f"\ts_mov_b32 s{s_mma_seq}, 0",
            ])
            i += 1
            continue

        if i <= open_end or i >= epilogue_start:
            if i >= epilogue_start and not epilogue_shifted:
                blk: List[str] = []
                if in_mfma:
                    blk.extend(emit_rec(_TAG_END | EV_MMA, f"s{s_mma_seq}"))
                    blk.append(f"\ts_add_u32 s{s_mma_seq}, s{s_mma_seq}, 1")
                    in_mfma = False
                if in_vmem_burst:
                    blk.extend(emit_rec(EV_LOAD, f"s{s_seq_in}"))
                    blk.append(f"\ts_add_u32 s{s_seq_in}, s{s_seq_in}, 1")
                    in_vmem_burst = False
                # Retire any still-open VMEM groups if epilogue lacks an explicit vmcnt(0) before store
                drain_uid += 1
                lbl_ep_loop = f".LWP_AMD_EPDRAIN_{drain_uid}"
                lbl_ep_done = f".LWP_AMD_EPDONE_{drain_uid}"
                blk.append(f"{lbl_ep_loop}:")
                blk.append(f"\ts_cmp_lt_i32 s{s_seq_out}, s{s_seq_in}")
                blk.append(f"\ts_cbranch_scc0 {lbl_ep_done}")
                blk.extend(emit_rec(_TAG_END | EV_LOAD, f"s{s_seq_out}"))
                blk.append(f"\ts_add_u32 s{s_seq_out}, s{s_seq_out}, 1")
                blk.append(f"\ts_branch {lbl_ep_loop}")
                blk.append(f"{lbl_ep_done}:")
                # Shift `s_base_byte` once by `(s_cur_byte - 32)` so all trailing `wp_*_16b` + `wp_close_16b`
                # stores land contiguously right after our hot-loop records!
                blk.append(f"\ts_sub_i32 s{s_tmp}, s{s_cur_byte}, 32")
                blk.append(f"\ts_add_u32 s{s_base_byte}, s{s_base_byte}, s{s_tmp}")
                out.extend(with_scc_guard(blk))
                stats["rebound_epilogue_offsets"] = 1
                epilogue_shifted = True
            out.append(ln)
            i += 1
            continue

        # Inside `[open_end + 1, epilogue_start)`:
        is_vmem_load = bool(re.match(r"^(?:buffer_load_|global_load_)\w+", s))
        is_mfma = s.startswith("v_mfma_")
        m_vmwait = re.match(r"^s_waitcnt\s+.*?\bvmcnt\((\d+)\)", s)
        is_ctrl = (
            (s.startswith(".LBB") and s.endswith(":"))
            or s.startswith("s_cbranch")
            or s.startswith("s_branch")
            or s == "s_barrier"
        )

        if in_vmem_burst and (is_mfma or m_vmwait or is_ctrl):
            blk = emit_rec(EV_LOAD, f"s{s_seq_in}") + [f"\ts_add_u32 s{s_seq_in}, s{s_seq_in}, 1"]
            out.extend(with_scc_guard(blk))
            if insts_per_group_est == 0 and vmem_insts_in_cur_group > 0:
                insts_per_group_est = vmem_insts_in_cur_group
            in_vmem_burst = False
            vmem_insts_in_cur_group = 0
            stats["vmem_load_groups"] += 1

        if in_mfma and (is_vmem_load or m_vmwait or is_ctrl):
            blk = emit_rec(_TAG_END | EV_MMA, f"s{s_mma_seq}") + [f"\ts_add_u32 s{s_mma_seq}, s{s_mma_seq}, 1"]
            out.extend(with_scc_guard(blk))
            in_mfma = False

        if is_vmem_load:
            in_vmem_burst = True
            vmem_insts_in_cur_group += 1
            out.append(ln)
            i += 1
            continue

        if m_vmwait:
            min_vmcnt = int(m_vmwait.group(1))
            j = i + 1
            while j < epilogue_start:
                sj = lines[j].strip()
                if (
                    sj.startswith("v_mfma_")
                    or re.match(r"^(?:buffer_load_|global_load_)\w+", sj)
                    or sj.startswith("s_cbranch")
                    or sj.startswith("s_branch")
                    or (sj.startswith(".LBB") and sj.endswith(":"))
                ):
                    break
                mw2 = re.match(r"^s_waitcnt\s+.*?\bvmcnt\((\d+)\)", sj)
                if mw2:
                    min_vmcnt = min(min_vmcnt, int(mw2.group(1)))
                    j += 1
                    continue
                if sj == "s_barrier":
                    break
                if not any("vmcnt(" in lines[k] for k in range(j, min(epilogue_start, j + 32))):
                    break
                j += 1

            ipg = max(1, insts_per_group_est or 8)
            keep_groups = min_vmcnt // ipg

            out.extend(with_scc_guard(emit_rec(EV_WAIT, f"s{s_seq_out}")))
            out.extend(lines[i:j])

            drain_uid += 1
            lbl_loop = f".LWP_AMD_DRAIN_{drain_uid}"
            lbl_done = f".LWP_AMD_DONE_{drain_uid}"
            post_blk = list(emit_rec(_TAG_END | EV_WAIT, f"s{s_seq_out}"))
            post_blk.append(f"\ts_sub_i32 s{s_tmp}, s{s_seq_in}, {keep_groups}")
            post_blk.append(f"{lbl_loop}:")
            post_blk.append(f"\ts_cmp_lt_i32 s{s_seq_out}, s{s_tmp}")
            post_blk.append(f"\ts_cbranch_scc0 {lbl_done}")
            post_blk.extend(emit_rec(_TAG_END | EV_LOAD, f"s{s_seq_out}", reuse_ts=True))
            post_blk.append(f"\ts_add_u32 s{s_seq_out}, s{s_seq_out}, 1")
            post_blk.append(f"\ts_branch {lbl_loop}")
            post_blk.append(f"{lbl_done}:")
            out.extend(with_scc_guard(post_blk))
            stats["vmem_wait_clusters"] += 1
            i = j
            continue

        if instrument_mfma and is_mfma:
            if not in_mfma:
                out.extend(with_scc_guard(emit_rec(EV_MMA, f"s{s_mma_seq}")))
                in_mfma = True
                stats["mfma_clusters"] += 1
            out.append(ln)
            i += 1
            continue

        out.append(ln)
        i += 1

    return "\n".join(out) + "\n", stats


@contextlib.contextmanager
def triton_amdgcn_async_hook(
    stats_out: Optional[Dict[str, int]] = None,
    *,
    cap: int = 256,
    max_hot_events: Optional[int] = None,
    instrument_mfma: bool = True,
) -> Iterator[None]:
    """Context manager hooking ROCm Triton's ``HIPBackend.make_amdgcn`` stage to inject CDNA3 async probes."""
    from triton.backends.amd.compiler import HIPBackend

    raw_orig = HIPBackend.__dict__["make_amdgcn"]
    is_static = isinstance(raw_orig, staticmethod)
    orig_fn = raw_orig.__func__ if is_static else raw_orig

    def _patched(*args, **kwargs):
        amdgcn = orig_fn(*args, **kwargs)
        if isinstance(amdgcn, str) and "s_memrealtime " in amdgcn:
            patched, st = instrument_triton_amdgcn_async(
                amdgcn,
                cap=cap,
                max_hot_events=max_hot_events,
                instrument_mfma=instrument_mfma,
            )
            if stats_out is not None:
                stats_out.clear()
                stats_out.update(st)
            if st.get("patched"):
                return patched
        return amdgcn

    HIPBackend.make_amdgcn = staticmethod(_patched) if is_static else _patched
    try:
        yield
    finally:
        HIPBackend.make_amdgcn = raw_orig


def build_triton_amd_async_names(
    *,
    store_label: str = "C Tile Store (HBM3)",
) -> Dict[str, Any]:
    """Build a canonical ``names.json`` dict for decoding AMD Triton AMDGCN-async dumps."""
    return {
        "events": {
            "1": "vmem_async_tile_load",
            "2": "s_waitcnt_vmcnt",
            "3": "cdna3_mfma_cluster",
            "6": "tl_store_c",
        },
        "labels": {
            "1": "AMD Async VMEM Load (buffer_load FIFO)",
            "2": "Exposed Wait (s_waitcnt vmcnt)",
            "3": "Matrix Core MFMA",
            "6": store_label,
        },
        "semantics": {"1": "load", "2": "wait", "3": "mma", "6": "store"},
        "slotted_events": {
            "1": {"id": "vmem_async_tile_load", "slot": "seq", "cross_warp": False},
        },
        "async_engines": {"1": "mem"},
        "roles": {"0": "wave64_worker"},
        "group": "cu",
        "pair": "stack",
    }
