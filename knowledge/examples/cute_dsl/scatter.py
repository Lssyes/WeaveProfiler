"""Minimal CuTe DSL reference for a data-dependent scatter, at both layers.

SEMANTIC-ID: dispatch
EXECUTION-SCOPE: one physical thread per source/index element.

This file is the clearest example of the rule that decides the layer: **what the
``grid_sync`` boundaries carve out, not what the code looks like.**

``compress_scatter``  layer=task.  In the measured decode megakernel the
                      Compress-KV Scatter *is* a whole ``grid_sync`` segment, so
                      it is a first-class operator with its own task span and its
                      own bar on the timeline.
``scatter_stage``     layer=phase.  The same scatter code called as the write-out
                      stage of a larger operator is an internal of that operator.

Same instruction stream, different layer, because the enclosing structure is
different.  Deciding by keyword ("a scatter is always a phase") is how whole
segments end up with no task span at all.

The caller supplies CuTe tensors whose first dimension is indexed by the physical
thread.  Indices must be in range and non-conflicting unless the surrounding
operation defines the conflict semantics.
"""

from cutlass import Int32, cute

import profiler as prof


@cute.jit
def compress_scatter(src: cute.Tensor, indices: cute.Tensor, dst: cute.Tensor,
                     ctx, work_id: Int32 = Int32(0)):
    """layer=task: the scatter occupies a whole grid_sync segment."""
    tidx, _, _ = cute.arch.thread_idx()
    # TRACE-BEGIN
    prof.task_begin(ctx, prof.EV_COMPRESS_SCATTER, work_id)
    dst[indices[tidx]] = src[tidx]
    prof.task_end(ctx, prof.EV_COMPRESS_SCATTER, work_id)
    # TRACE-END


@cute.jit
def scatter_stage(src: cute.Tensor, indices: cute.Tensor, dst: cute.Tensor,
                  ctx, work_id: Int32 = Int32(0)):
    """layer=phase: the same scatter as the write-out stage of an operator."""
    tidx, _, _ = cute.arch.thread_idx()
    # TRACE-BEGIN
    prof.phase_begin(ctx, prof.EV_STORE, work_id)
    dst[indices[tidx]] = src[tidx]
    prof.phase_end(ctx, prof.EV_STORE, work_id)
    # TRACE-END
