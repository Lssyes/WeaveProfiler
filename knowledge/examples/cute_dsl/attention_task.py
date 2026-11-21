"""CuTe DSL reference: one operator task, internals demoted to phase.

SEMANTIC-ID: unknown  # requires model review: sparse attention is not in the supplied canonical semantic_id set
EXECUTION-SCOPE: one CTA per SM; warpgroup 0 stages KV, warpgroups 1+ run WGMMA.
ABI: the enclosing Sparse Attention span is layer=task and is opened by the
     caller (``megakernel.py``).  **Everything recorded in this file is
     layer=phase**, so a task-only delivery folds the whole operator into that
     single bar.

Why the convergence matters
---------------------------
Attention is the segment where per-token / per-KV-chunk detail is genuinely
useful, so the temptation is to expand it into many task spans.  Do not: the
task layer is the operator layer, and it is what the coverage gate measures.
Give attention exactly one task and put the detail one level down:

    record_phases=False   one "Sparse Attention" bar, peer of every other
                          operator; the timeline stays readable and the
                          coverage union is unchanged
    record_phases=True    the same bar opens into query rows, KV chunks, KV
                          loads, WGMMA issues, softmax rescales and stores

Measured shape of this decomposition: 22 operator tasks total, 157k spans,
99.1% coverage, largest hole 1.18us, 5.05% overhead.

Nesting a task inside a task (allowed, narrowly)
------------------------------------------------
An inner span may be promoted to ``task`` layer only if it is strictly contained
in the enclosing operator task on the same lane -- then the coverage union does
not change, and the ABI must say so with ``nested_within_task_id``.  It may never
be the *only* attention span, and it may never extend past the operator.  When
in doubt keep it ``phase``: phases are free to disappear, tasks are not.

The ``ctx`` tuple is not a style choice
---------------------------------------
CuTe DSL forbids calling a closure that *captures* dynamic SSA values from inside
dynamic control flow (``cutlass.range``).  Every record helper here therefore
receives the explicit ``ctx`` tuple built by ``profiler.make_ctx``, which is what
lets the very same helper be called inside the per-KV-chunk loop.

Fidelity rules obeyed here
--------------------------
* the arithmetic, the SMEM layouts, the loop structure, the ``cute.arch.barrier``
  placement and the WGMMA issue order are byte-for-byte the production kernel's;
* ``profiler.rec`` contains no barrier and no warp-wide vote, so it is safe
  inside the divergent ``if wg >= 1`` consumer region and cannot change control
  flow;
* only the record append is gated -- with ``record_phases=False`` or
  ``PROF_OFF`` the compute and the barriers still run exactly as before.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32

import profiler as prof

# Reserved slice ids so that (event_id, work_id) stays unique on every lane.
SL_SCORES = 0        # QK^T + softmax chunks do not use the slice field
SL_SINK_GATE = 14    # attention-sink gate
SL_STAGE_Q = 15      # Q staging

FA_NCHUNK = 18       # KV chunks per query row
O_SLICES = 4         # output dim-slices of the P*V loop


def _work(row: Int32, slice_: int, chunk) -> Int32:
    """Pack (query row, slice, KV chunk) into the 20-bit work_id field."""
    return ((row & Int32(0xFF)) << Int32(prof.WORK_ROW_LSB)) \
        | Int32((slice_ & 0xF) << prof.WORK_SLICE_LSB) \
        | (chunk & Int32(0xFF))


class AttentionInstrumented:
    """Mixin providing the instrumented twin of the production attention phase."""

    @cute.jit
    def attention_instr(self, q: cute.Tensor, indices_final: cute.Tensor,
                        kv_cache: cute.Tensor, attn_sink: cute.Tensor,
                        attn_out: cute.Tensor, storage,
                        tidx: Int32, bidx: Int32, ctx) -> None:
        # ``ctx`` comes from the caller together with the already-open
        # "Sparse Attention" task span.  Nothing in this function opens a task.
        smem = self.attention_smem(storage)
        wg = tidx // Int32(128)
        n_rows = self.rows_owned_by(bidx)

        for ri in cutlass.range(n_rows, unroll=1):
            row = self.row_of(bidx, ri)

            # ---- phase: one decode query row ------------------------
            w_row = _work(row, SL_SCORES, Int32(0))
            prof.phase_begin(ctx, prof.EV_ATTN_QUERY_ROW, w_row)

            # ---- phase: count masked keys ---------------------------
            prof.phase_begin(ctx, prof.EV_ATTN_MASK_COUNT, w_row)
            n_invalid = self.count_masked_keys(indices_final, row, tidx, smem)
            prof.phase_end(ctx, prof.EV_ATTN_MASK_COUNT, w_row)

            # ---- phase: stage Q to SMEM, with its own load/barrier --
            w_q = _work(row, SL_STAGE_Q, Int32(0))
            prof.phase_begin(ctx, prof.EV_ATTN_STAGE_Q, w_q)
            prof.phase_begin(ctx, prof.EV_LOAD, w_q)
            self.stage_q_to_smem(q, row, smem, tidx)
            prof.phase_end(ctx, prof.EV_LOAD, w_q)
            prof.phase_begin(ctx, prof.EV_TILE_BARRIER, w_q)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            prof.phase_end(ctx, prof.EV_TILE_BARRIER, w_q)
            prof.phase_end(ctx, prof.EV_ATTN_STAGE_Q, w_q)

            # ---- phase: scores QK^T + online softmax, per KV chunk --
            # One span per KV chunk.  This is the innermost level worth
            # recording; it stays at phase layer so the task-only delivery is
            # not flooded with FA_NCHUNK spans per row.
            for chunk in cutlass.range(FA_NCHUNK, unroll=1):
                w_c = _work(row, SL_SCORES, chunk)
                prof.phase_begin(ctx, prof.EV_ATTN_QK_SOFTMAX, w_c)

                prof.phase_begin(ctx, prof.EV_LOAD, w_c)
                self.stage_kv_chunk(kv_cache, indices_final, row, chunk,
                                    smem, tidx)
                prof.phase_end(ctx, prof.EV_LOAD, w_c)

                prof.phase_begin(ctx, prof.EV_TILE_BARRIER, w_c)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                prof.phase_end(ctx, prof.EV_TILE_BARRIER, w_c)

                if wg >= Int32(1):
                    # A record inside a divergent region is safe: rec() has no
                    # barrier and no warp-wide vote.
                    prof.phase_begin(ctx, prof.EV_MMA, w_c)
                    self.qk_wgmma(smem)
                    prof.phase_end(ctx, prof.EV_MMA, w_c)

                    prof.phase_begin(ctx, prof.EV_EPILOGUE, w_c)
                    self.online_softmax_update(smem, row, chunk)
                    prof.phase_end(ctx, prof.EV_EPILOGUE, w_c)

                    prof.phase_begin(ctx, prof.EV_STORE, w_c)
                    self.store_probabilities(smem, row, chunk)
                    prof.phase_end(ctx, prof.EV_STORE, w_c)

                prof.phase_begin(ctx, prof.EV_TAIL_BARRIER, w_c)
                cute.arch.barrier()
                prof.phase_end(ctx, prof.EV_TAIL_BARRIER, w_c)

                prof.phase_end(ctx, prof.EV_ATTN_QK_SOFTMAX, w_c)

            # ---- phase: attention-sink gate -------------------------
            w_gate = _work(row, SL_SINK_GATE, Int32(0))
            prof.phase_begin(ctx, prof.EV_ATTN_SINK_GATE, w_gate)
            gate = self.attn_sink_gate(attn_sink, row, n_invalid, wg)
            prof.phase_end(ctx, prof.EV_ATTN_SINK_GATE, w_gate)

            # ---- phase: output P*V, per (dim-slice, KV chunk) -------
            for sl in cutlass.range(O_SLICES, unroll=1):
                for chunk in cutlass.range(FA_NCHUNK, unroll=1):
                    # slice id + 1 keeps this work_id distinct from the
                    # QK^T spans, which reserve slice 0.
                    w_pv = _work(row, 0, chunk) | Int32(1)
                    prof.phase_begin(ctx, prof.EV_ATTN_PV, w_pv)

                    prof.phase_begin(ctx, prof.EV_LOAD, w_pv)
                    self.stage_v_chunk(kv_cache, indices_final, row, sl,
                                       chunk, smem, tidx)
                    prof.phase_end(ctx, prof.EV_LOAD, w_pv)

                    prof.phase_begin(ctx, prof.EV_TILE_BARRIER, w_pv)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()
                    prof.phase_end(ctx, prof.EV_TILE_BARRIER, w_pv)

                    if wg >= Int32(1):
                        prof.phase_begin(ctx, prof.EV_MMA, w_pv)
                        self.pv_wgmma(smem, sl)
                        prof.phase_end(ctx, prof.EV_MMA, w_pv)

                        prof.phase_begin(ctx, prof.EV_EPILOGUE, w_pv)
                        self.accumulate_output(smem, row, sl, chunk, gate)
                        prof.phase_end(ctx, prof.EV_EPILOGUE, w_pv)

                    prof.phase_begin(ctx, prof.EV_TAIL_BARRIER, w_pv)
                    cute.arch.barrier()
                    prof.phase_end(ctx, prof.EV_TAIL_BARRIER, w_pv)

                    prof.phase_end(ctx, prof.EV_ATTN_PV, w_pv)

            # ---- phase: inverse-RoPE + store ------------------------
            for sl in cutlass.range(O_SLICES, unroll=1):
                w_ep = _work(row, 0, Int32(0)) | Int32(2)
                prof.phase_begin(ctx, prof.EV_ATTN_INV_ROPE_STORE, w_ep)
                prof.phase_begin(ctx, prof.EV_STORE, w_ep)
                self.inverse_rope_store(attn_out, smem, row, sl, tidx)
                prof.phase_end(ctx, prof.EV_STORE, w_ep)
                prof.phase_end(ctx, prof.EV_ATTN_INV_ROPE_STORE, w_ep)

            prof.phase_end(ctx, prof.EV_ATTN_QUERY_ROW, w_row)
