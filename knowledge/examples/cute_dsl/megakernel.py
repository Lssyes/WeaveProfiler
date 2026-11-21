"""CuTe DSL reference: a persistent megakernel instrumented segment by segment.

SEMANTIC-ID: megakernel
EXECUTION-SCOPE: one CTA per SM, one launch, ~20 ``grid_sync`` compute segments.
ABI: layer=task for every compute segment; layer=ignored for the whole-kernel
     anchor and every ``grid_sync`` rendezvous.

This is the structural reference the other files in this directory plug into.
It is the shape that was measured: 20 operator tasks + the single Sparse
Attention task's internals, 157k spans, 99.1% operator coverage of the
whole-kernel window, largest hole 1.18us, 5.05% overhead.

The one rule
------------
**Every ``grid_sync``-separated compute segment gets its own task span**, named
after the operator it runs.  Not "the interesting segment gets a task and the
rest get barrier records" -- barrier records are ``ignored``, they are dropped in
a task-only delivery, and the timeline then shows one operator floating in an
empty kernel window.  That shipped once: 769us of head and 337us of tail with no
operator span, real coverage 44.9%.  See ``knowledge/guides/design.md`` section 2.7.

Two-layer convergence
---------------------
A segment whose internals are worth seeing (here: attention) still gets exactly
**one** task.  Its per-row / per-chunk / load / mma detail is recorded at
``phase`` layer inside ``attention_task.py``, so:

    record_phases=False  ->  attention is one bar, like every other operator
    record_phases=True   ->  the same bar opens into its internals

Fidelity rules
--------------
* the instrumented kernel is a **subclass** of the pristine production kernel and
  reuses its operator mixins verbatim; the production file is never edited;
* only the record append is ever gated -- the compute and every ``grid_sync``
  run unchanged in all configurations;
* ``prof_level == PROF_OFF`` strips every record at trace time, so the
  production build keeps its original instruction stream.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

import profiler as prof
from attention_task import AttentionInstrumented

# Production kernel and its ``grid_sync`` primitive; never modified here.
from megakernel_production import MegaKernel, grid_sync


class MegaKernelInstrumented(MegaKernel, AttentionInstrumented):
    """Production kernel + software event records.

    ``prof_level``    compile-time PROF_OFF / PROF_ENTRY_ONLY / PROF_FULL.
    ``record_phases`` compile-time: False delivers task-only (operators fold to
                      one bar each); True adds the phase layer.
    """

    def __init__(self, prof_level: int = prof.PROF_FULL,
                 record_phases: bool = True):
        self.prof_level = prof_level
        self.record_phases = record_phases

    @cute.kernel
    def kernel_instr(self, *tensors_and_atoms,
                     locks: cute.Tensor,
                     prof_buf: cute.Tensor, prof_cnt: cute.Tensor,
                     prof_ovf: cute.Tensor, prof_sel: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lock_ptr = locks.iterator
        storage = self.allocate_shared_storage()

        # ---- capture context -----------------------------------------
        # Source-side %smid filter: an SM outside the selected capture scope
        # gets capture_slot < 0 and records nothing.  No host post-filter.
        ctx = prof.make_ctx(
            prof_buf, prof_cnt, prof_ovf, self.prof_level,
            prof.capture_slot_of(prof_sel),          # capture_slot
            bidx,                                    # logical_cta
            Int32(cute.arch.smid()),                 # physical_smid
            tidx, self.record_phases,
        )

        # ---- layer=ignored: whole-kernel anchor ----------------------
        # Recorded at PROF_ENTRY_ONLY: SW/HW alignment reads these two
        # timestamps and the overhead denominator is this span.  The emitter
        # MUST NOT render it -- a full-width bar replicated onto every lane is
        # what once turned 44.9% real coverage into a green 100%.
        prof.anchor_begin(ctx, prof.EV_KERNEL)

        # ==============================================================
        # One task per grid_sync compute segment, in execution order.
        # Pattern, repeated verbatim for every segment:
        #     task_begin(op)   compute   task_end(op)
        #     barrier_begin(boundary)   grid_sync   barrier_end(boundary)
        # ==============================================================

        # -- Quantize X to FP8 --
        prof.task_begin(ctx, prof.EV_QUANT_X)
        self.quant_activation(*tensors_and_atoms, tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_QUANT_X)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(0))
        grid_sync(lock_ptr, tidx, 0)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(0))

        # -- QKV-A Proj (FP8 GEMM) --
        prof.task_begin(ctx, prof.EV_QKV_A_GEMM)
        self.fp8_gemm(*tensors_and_atoms, storage=storage, tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_QKV_A_GEMM)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(1))
        grid_sync(lock_ptr, tidx, 1)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(1))

        # -- Q-LoRA + KV Compress --
        prof.task_begin(ctx, prof.EV_Q_LORA_KV)
        self.q_lora_kv(*tensors_and_atoms, tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_Q_LORA_KV)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(2))
        grid_sync(lock_ptr, tidx, 2)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(2))

        # -- Quantize Q-LoRA to FP8 --
        prof.task_begin(ctx, prof.EV_QUANT_Q_LORA)
        self.quant_activation(*tensors_and_atoms, tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_QUANT_Q_LORA)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(3))
        grid_sync(lock_ptr, tidx, 3)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(3))

        # -- Q-B Proj (FP8 GEMM) --
        prof.task_begin(ctx, prof.EV_WQ_B_GEMM)
        self.fp8_gemm(*tensors_and_atoms, storage=storage, tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_WQ_B_GEMM)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(4))
        grid_sync(lock_ptr, tidx, 4)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(4))

        # -- Q RMSNorm + RoPE --
        prof.task_begin(ctx, prof.EV_Q_NORM_ROPE)
        self.q_norm_rope(*tensors_and_atoms, tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_Q_NORM_ROPE)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(5))
        grid_sync(lock_ptr, tidx, 5)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(5))

        # -- Compress-KV Proj (BF16 GEMM) --
        prof.task_begin(ctx, prof.EV_COMPRESS_GEMM)
        self.bf16_gemm(*tensors_and_atoms, storage=storage, tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_COMPRESS_GEMM)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(6))
        grid_sync(lock_ptr, tidx, 6)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(6))

        # -- Compress-KV Scatter --
        # Two operators inside one masked phase still get two tasks: the task
        # boundary follows the grid_sync boundary, not the source `if`.
        prof.task_begin(ctx, prof.EV_COMPRESS_SCATTER)
        self.compress_scatter(*tensors_and_atoms, tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_COMPRESS_SCATTER)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(7))
        grid_sync(lock_ptr, tidx, 7)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(7))

        # -- CSA Pooling --
        prof.task_begin(ctx, prof.EV_CSA_POOL)
        self.csa_pool(*tensors_and_atoms, tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_CSA_POOL)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(8))
        grid_sync(lock_ptr, tidx, 8)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(8))

        # -- CSA RoPE Finalize --
        prof.task_begin(ctx, prof.EV_CSA_ROPE)
        self.csa_rope_finalize(*tensors_and_atoms, tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_CSA_ROPE)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(9))
        grid_sync(lock_ptr, tidx, 9)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(9))

        # -- CSA Cache Append --
        prof.task_begin(ctx, prof.EV_CSA_APPEND)
        self.csa_cache_append(*tensors_and_atoms, tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_CSA_APPEND)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(10))
        grid_sync(lock_ptr, tidx, 10)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(10))

        # -- Indexer-Q Proj (FP8 GEMM) --
        prof.task_begin(ctx, prof.EV_INDEXER_GEMM)
        self.fp8_gemm(*tensors_and_atoms, storage=storage, tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_INDEXER_GEMM)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(11))
        grid_sync(lock_ptr, tidx, 11)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(11))

        # -- Indexer Q RoPE --
        prof.task_begin(ctx, prof.EV_INDEXER_Q_ROPE)
        self.indexer_q_rope(*tensors_and_atoms, tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_INDEXER_Q_ROPE)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(12))
        grid_sync(lock_ptr, tidx, 12)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(12))

        # -- Indexer Logits (MQA) --
        prof.task_begin(ctx, prof.EV_LOGITS)
        self.indexer_logits(*tensors_and_atoms, storage=storage,
                            tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_LOGITS)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(13))
        grid_sync(lock_ptr, tidx, 13)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(13))

        # -- Top-k Select --
        prof.task_begin(ctx, prof.EV_TOPK)
        self.topk_select(*tensors_and_atoms, storage=storage,
                         tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_TOPK)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(14))
        grid_sync(lock_ptr, tidx, 14)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(14))

        # -- Build Sparse Indices --
        prof.task_begin(ctx, prof.EV_BUILD_SPARSE_INDICES)
        self.build_sparse_indices(*tensors_and_atoms, tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_BUILD_SPARSE_INDICES)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(15))
        grid_sync(lock_ptr, tidx, 15)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(15))

        # -- Sparse Attention: ONE task, internals at phase layer ------
        # ``attention_instr`` receives ``ctx`` and records only phase-layer
        # events inside, so a task-only build renders this segment as a single
        # "Sparse Attention" bar exactly like every other operator.
        prof.task_begin(ctx, prof.EV_ATTENTION)
        self.attention_instr(*tensors_and_atoms, storage=storage,
                             tidx=tidx, bidx=bidx, ctx=ctx)
        prof.task_end(ctx, prof.EV_ATTENTION)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(16))
        grid_sync(lock_ptr, tidx, 16)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(16))

        # -- Output-A Proj (BF16 GEMM) --
        prof.task_begin(ctx, prof.EV_WO_A_GEMM)
        self.bf16_gemm(*tensors_and_atoms, storage=storage, tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_WO_A_GEMM)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(17))
        grid_sync(lock_ptr, tidx, 17)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(17))

        # -- Quantize O-mid to FP8 --
        prof.task_begin(ctx, prof.EV_QUANT_O_MID)
        self.quant_activation(*tensors_and_atoms, tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_QUANT_O_MID)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(18))
        grid_sync(lock_ptr, tidx, 18)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(18))

        # -- Output-B Proj (FP8 GEMM) --
        prof.task_begin(ctx, prof.EV_WO_B_GEMM)
        self.fp8_gemm(*tensors_and_atoms, storage=storage, tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_WO_B_GEMM)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(19))
        grid_sync(lock_ptr, tidx, 19)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(19))

        # ---- layer=ignored: whole-kernel anchor close ----------------
        prof.anchor_end(ctx, prof.EV_KERNEL)


# ----------------------------------------------------------------------
# Segments hidden behind a compile-time phase mask
# ----------------------------------------------------------------------
# When the production kernel guards a segment with a compile-time mask, keep the
# guard and put the task inside it.  The task must never wrap the guard itself:
# a task span that covers a compiled-away segment is a fabricated interval.
@cute.jit
def masked_segment(self, ctx, enabled: cutlass.Constexpr, lock_ptr,
                   tidx: Int32, bidx: Int32, boundary: int):
    if const_expr(enabled):
        prof.task_begin(ctx, prof.EV_CSA_POOL)
        self.csa_pool(tidx=tidx, bidx=bidx)
        prof.task_end(ctx, prof.EV_CSA_POOL)
        prof.barrier_begin(ctx, prof.EV_GRID_BARRIER, Int32(boundary))
        grid_sync(lock_ptr, tidx, boundary)
        prof.barrier_end(ctx, prof.EV_GRID_BARRIER, Int32(boundary))


# ----------------------------------------------------------------------
# Self-check before running anything
# ----------------------------------------------------------------------
# 1. count of ``grid_sync`` calls in the production kernel
#      == count of ``EV_GRID_BARRIER`` pairs here;
# 2. count of compute segments between those boundaries
#      == count of distinct task-layer ids used here
#      == count of ``layer == "task"`` entries in the ABI JSON;
# 3. every task-layer display name in the ABI is an operator name a reader can
#    map back to code (the delivery gate FAILs on ``passA`` / bare ``token`` /
#    ``EV_*`` / ``P11``-style codenames);
# 4. the whole-kernel anchor and the grid barriers are ``layer: "ignored"`` in
#    the ABI, and the emitter drops every ignored-layer event;
# 5. after the run: union of task-layer spans vs the whole-kernel window leaves
#    no hole larger than max(5% of the kernel span, 50us).
