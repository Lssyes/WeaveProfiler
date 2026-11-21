"""Minimal CuTe DSL reference for the per-CTA capture context and work item.

SEMANTIC-ID: dispatch
EXECUTION-SCOPE: cta.
ABI: sets up the recording context; emits the layer=ignored whole-kernel anchor.

A persistent megakernel does not dispatch work items from a queue -- the CTA
index *is* the work item and it keeps it for the whole launch.  What actually has
to happen at kernel entry is the recording setup, so that is what this file
shows:

1. the source-side ``%smid`` filter: every warp looks its own physical SM up in
   the host-materialised ``prof_sel`` map and gets a dense ``capture_slot``, or
   ``-1`` if this SM is outside the capture scope and must record nothing.  A
   host-side post-filter is not an acceptable substitute;
2. the ``ctx`` tuple every record helper takes (explicit, not a closure, so it
   can be used inside ``cutlass.range``);
3. the whole-kernel anchor, at layer=ignored: recorded already at
   ``PROF_ENTRY_ONLY`` because SW/HW alignment reads its entry/release
   timestamps and the overhead denominator is its span -- and dropped by the
   emitter, because a full-width bar replicated onto every lane is what once
   made a 44.9%-covered timeline report 100%.
"""

import cutlass
from cutlass import Int32, cute

import profiler as prof


@cute.jit
def open_capture_context(prof_buf, prof_cnt, prof_ovf, prof_sel,
                         prof_level: cutlass.Constexpr,
                         record_phases: cutlass.Constexpr = True):
    """Build the record context and open the whole-kernel anchor."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    ctx = prof.make_ctx(
        prof_buf, prof_cnt, prof_ovf, prof_level,
        prof.capture_slot_of(prof_sel),      # -1 => this SM records nothing
        bidx,                                # logical_cta == the work item
        Int32(cute.arch.smid()),             # physical_smid
        tidx, record_phases,
    )
    # TRACE-BEGIN
    prof.anchor_begin(ctx, prof.EV_KERNEL)
    # TRACE-END
    grid, _, _ = cute.arch.grid_dim()
    return ctx, bidx, grid


@cute.jit
def close_capture_context(ctx):
    """Close the whole-kernel anchor as the last act of the kernel."""
    # TRACE-BEGIN
    prof.anchor_end(ctx, prof.EV_KERNEL)
    # TRACE-END
