"""Minimal CuTe DSL reference for the two barrier kinds, and their two layers.

SEMANTIC-ID: barrier
EXECUTION-SCOPE: ``tile_ready_barrier`` = all threads of one CTA;
                 ``segment_boundary`` = every thread of every CTA.

The two barriers are NOT the same layer:

``tile_ready_barrier``  layer=phase.  A CTA-local barrier *inside* an operator
                        (cp.async wait + ``cute.arch.barrier``).  It is part of
                        that operator's internal structure, so it is drawn
                        nested inside the operator task when phases are on.
``segment_boundary``    layer=ignored.  The ``grid_sync`` rendezvous between two
                        operators.  Recorded for diagnostics (how much of the
                        boundary is wait) and kept in the decoded data, but
                        **never rendered** and never counted as coverage.

The historical bug: instrumenting only the segment boundaries and calling that
"the megakernel is instrumented".  Barrier events are ``ignored``, a task-only
delivery drops them, and the timeline showed one operator with 769us of empty
head and 337us of empty tail -- 44.9% real coverage.  A boundary record is
evidence about a boundary; it is never a substitute for the operator task around
the compute.  See ``megakernel.py`` for the required pairing.
"""

import cutlass
from cutlass import Int32, cute

import profiler as prof


@cute.jit
def tile_ready_barrier(ctx, work_id: Int32 = Int32(0)):
    """layer=phase: CTA-local tile handoff inside one operator."""
    # TRACE-BEGIN
    prof.phase_begin(ctx, prof.EV_TILE_BARRIER, work_id)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()
    prof.phase_end(ctx, prof.EV_TILE_BARRIER, work_id)
    # TRACE-END


@cute.jit
def segment_boundary(ctx, grid_sync, lock_ptr, tidx: Int32,
                     boundary: cutlass.Constexpr):
    """layer=ignored: the ``grid_sync`` rendezvous between two operators.

    The caller must already have closed the preceding operator's task span and
    must open the next one after this returns -- see ``megakernel.py``.
    """
    # TRACE-BEGIN
    prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(boundary))
    grid_sync(lock_ptr, tidx, boundary)
    prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(boundary))
    # TRACE-END
