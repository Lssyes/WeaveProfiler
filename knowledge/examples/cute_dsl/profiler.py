"""CuTe DSL device event recorder: three-state layers + task-only folding.

This is the framework half of the CuTe DSL reference set.  It is a trimmed copy
of an instrumentation that was actually measured end to end on a persistent
decode megakernel (single launch, ~20 ``grid_sync`` compute segments): 22
operator tasks, 157k spans, 99.1% operator coverage of the whole-kernel window,
largest un-instrumented hole 1.18us, 5.05% recording overhead.

Granularity contract (the part that is easy to get wrong)
--------------------------------------------------------
``task``    one span per **real operator**.  In a persistent megakernel that
            means **one task per ``grid_sync``-separated compute segment** --
            not one task for the interesting segment and barriers for the rest.
            The display name is the human-readable operator name.
``phase``   operator *internals*: per-row / per-KV-chunk iterations, TMA load,
            WGMMA issue, online-softmax rescale, tile-ready barrier, store.
            Recorded only when ``record_phases`` is true, so the same source
            compiles to a task-only build with no phase records at all.
``ignored`` the whole-kernel anchor and every ``grid_sync`` rendezvous.  Kept in
            the decoded data (SW/HW alignment reads the entry/release
            timestamps) but **never rendered as a span**.

Why ``ignored`` may not be drawn: a full-width whole-kernel bar replicated onto
every lane fills a min/max coverage envelope and reports 100% while the timeline
is empty outside one operator.  That exact failure shipped once -- 769us head /
337us tail with no operator span, true coverage 44.9%, all 25 gates green.  See
``knowledge/guides/design.md`` section 2.7.

Event id space (mirror this table into the ABI JSON verbatim)
------------------------------------------------------------
    1 .. 9      ignored : whole-kernel anchor, grid_sync rendezvous
    20 .. 99    task    : one id per grid_sync compute segment
    100 .. 255  phase   : operator internals

Record path
-----------
* clock: 64-bit ``%globaltimer`` (ns, one domain for the whole device).
* capture scope: filtered **on the device**.  Every warp reads its own ``%smid``
  and looks up a dense ``capture_slot`` in the host-materialised ``prof_sel``
  map; an unselected SM gets ``capture_slot < 0`` and writes nothing.  No host
  post-filter is allowed.
* one append lane per physical warp: ``lane = capture_slot * WARPS_PER_CTA +
  warp``, single writer (that warp's leader), no atomics.
* overflow is explicit: a full lane stops writing and raises ``prof_ovf``; the
  host must reject the run.  Records are never silently dropped.
* ``prof_level`` is a compile-time ``cutlass.Constexpr``; at ``PROF_OFF`` every
  record body is eliminated at trace time, so the production kernel keeps its
  original instruction stream.
"""

from __future__ import annotations

import os

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr
from cutlass.cutlass_dsl import T
from cutlass.cute.arch.nvvm_wrappers import nvvm


SUPPORTED_ARCHES = frozenset({"sm_90", "sm_100a", "sm_103a"})


def _require_arch() -> str:
    """Read the explicit host build target; never guess an architecture."""
    raw = os.environ.get("ARCH")
    if raw is None:
        raise ValueError(
            "ARCH must be set explicitly to one of: sm_90, sm_100a, sm_103a")
    arch = raw.strip()
    if arch not in SUPPORTED_ARCHES:
        raise ValueError(
            f"ARCH={arch!r} is unsupported; expected one of: "
            "sm_90, sm_100a, sm_103a")
    return arch


ARCH = _require_arch()

# ----------------------------------------------------------------------
# buffer geometry -- retarget these three numbers, nothing else
# ----------------------------------------------------------------------
WARPS_PER_CTA = 12                  # THREADS / 32
MAX_RECORDS_PER_LANE = 8192
RECORD_WORDS = 2                    # (timestamp_ns, tag)

# ----------------------------------------------------------------------
# compile-time switch
# ----------------------------------------------------------------------
PROF_OFF = 0                        # no records at all
PROF_ENTRY_ONLY = 1                 # whole-kernel anchor only (overhead base)
PROF_FULL = 2                       # anchors + operator tasks + phases

# ----------------------------------------------------------------------
# the three ABI layers
# ----------------------------------------------------------------------
LAYER_TASK = "task"                 # rendered; counted by the coverage gate
LAYER_PHASE = "phase"               # rendered; nested inside its task
LAYER_IGNORED = "ignored"           # decoded, never rendered, never counted

REC_BEGIN = 1
REC_END = 0

# ----------------------------------------------------------------------
# tag bitfields (LSB first) -- authoritative copy lives in the ABI JSON
# ----------------------------------------------------------------------
TAG_LSB = {
    "is_start": 0,
    "event_id": 1,
    "warp": 9,
    "capture_slot": 13,
    "logical_cta": 20,
    "physical_smid": 27,
    "work_id": 34,
}
TAG_MASK = {
    "is_start": 0x1,
    "event_id": 0xFF,
    "warp": 0xF,
    "capture_slot": 0x7F,
    "logical_cta": 0x7F,
    "physical_smid": 0x7F,
    "work_id": 0xFFFFF,
}
TAG_USED_BITS = 54

# ----------------------------------------------------------------------
# ignored layer: anchors and rendezvous
# ----------------------------------------------------------------------
EV_KERNEL = 1                   # whole-kernel span; SW/HW alignment anchor
EV_GRID_BARRIER = 2             # one per grid_sync boundary; work_id = boundary

# ----------------------------------------------------------------------
# task layer: exactly one id per grid_sync compute segment, in execution order.
# These names are what a human reads on the timeline, so they are the real
# operator names.  Never ``passA`` / ``stage1`` / ``P0`` / ``EV_*``.
# ----------------------------------------------------------------------
EV_QUANT_X = 20                 # Quantize X to FP8
EV_QKV_A_GEMM = 21              # QKV-A Proj (FP8 GEMM)
EV_Q_LORA_KV = 22               # Q-LoRA + KV Compress
EV_QUANT_Q_LORA = 23            # Quantize Q-LoRA to FP8
EV_WQ_B_GEMM = 24               # Q-B Proj (FP8 GEMM)
EV_Q_NORM_ROPE = 25             # Q RMSNorm + RoPE
EV_COMPRESS_GEMM = 26           # Compress-KV Proj (BF16 GEMM)
EV_COMPRESS_SCATTER = 27        # Compress-KV Scatter
EV_CSA_POOL = 28                # CSA Pooling
EV_CSA_ROPE = 29                # CSA RoPE Finalize
EV_CSA_APPEND = 30              # CSA Cache Append
EV_INDEXER_GEMM = 31            # Indexer-Q Proj (FP8 GEMM)
EV_INDEXER_Q_ROPE = 32          # Indexer Q RoPE
EV_LOGITS = 33                  # Indexer Logits (MQA)
EV_TOPK = 34                    # Top-k Select
EV_BUILD_SPARSE_INDICES = 35    # Build Sparse Indices
EV_ATTENTION = 36               # Sparse Attention -- encloses the phases below
EV_WO_A_GEMM = 37               # Output-A Proj (BF16 GEMM)
EV_QUANT_O_MID = 38             # Quantize O-mid to FP8
EV_WO_B_GEMM = 39               # Output-B Proj (FP8 GEMM)

# ----------------------------------------------------------------------
# phase layer: operator internals.  Everything here folds away in a task-only
# build; the attention operator therefore collapses to the single EV_ATTENTION
# bar instead of exploding into per-token / per-chunk detail.
# ----------------------------------------------------------------------
EV_ATTN_QUERY_ROW = 110         # one decode query row this CTA owns
EV_ATTN_MASK_COUNT = 111        # count masked keys
EV_ATTN_STAGE_Q = 112           # stage Q into SMEM
EV_ATTN_QK_SOFTMAX = 113        # scores QK^T + online softmax, per KV chunk
EV_ATTN_SINK_GATE = 114         # attention-sink gate
EV_ATTN_PV = 115                # output P*V, per (dim-slice, KV chunk)
EV_ATTN_INV_ROPE_STORE = 116    # inverse-RoPE + store

EV_LOAD = 120                   # TMA / cp.async issue
EV_MMA = 121                    # WGMMA issue + wait
EV_EPILOGUE = 122               # rescale / online-softmax update
EV_TILE_BARRIER = 123           # cp.async wait + CTA barrier (tile ready)
EV_STORE = 124                  # global write
EV_TAIL_BARRIER = 125           # trailing CTA barrier

# ----------------------------------------------------------------------
# The ABI table, in code.  Copy it into ``*_abi.json`` unchanged: the delivery
# gate reads the *display name* of every task-layer span and FAILs on internal
# codenames, so this table is the single place where naming is decided.
#   event id -> (symbol, display name, short name, layer)
# ----------------------------------------------------------------------
EVENT_TABLE = {
    EV_KERNEL: ("EV_KERNEL", "kernel", "kernel", LAYER_IGNORED),
    EV_GRID_BARRIER: ("EV_GRID_BARRIER", "grid_barrier", "grid_barrier",
                      LAYER_IGNORED),

    EV_QUANT_X: ("EV_QUANT_X", "Quantize X to FP8", "quant_x", LAYER_TASK),
    EV_QKV_A_GEMM: ("EV_QKV_A_GEMM", "QKV-A Proj (FP8 GEMM)", "qkv_a_gemm",
                    LAYER_TASK),
    EV_Q_LORA_KV: ("EV_Q_LORA_KV", "Q-LoRA + KV Compress", "q_lora_kv",
                   LAYER_TASK),
    EV_QUANT_Q_LORA: ("EV_QUANT_Q_LORA", "Quantize Q-LoRA to FP8",
                      "quant_q_lora", LAYER_TASK),
    EV_WQ_B_GEMM: ("EV_WQ_B_GEMM", "Q-B Proj (FP8 GEMM)", "wq_b_gemm",
                   LAYER_TASK),
    EV_Q_NORM_ROPE: ("EV_Q_NORM_ROPE", "Q RMSNorm + RoPE", "q_norm_rope",
                     LAYER_TASK),
    EV_COMPRESS_GEMM: ("EV_COMPRESS_GEMM", "Compress-KV Proj (BF16 GEMM)",
                       "compress_gemm", LAYER_TASK),
    EV_COMPRESS_SCATTER: ("EV_COMPRESS_SCATTER", "Compress-KV Scatter",
                          "compress_scatter", LAYER_TASK),
    EV_CSA_POOL: ("EV_CSA_POOL", "CSA Pooling", "csa_pool", LAYER_TASK),
    EV_CSA_ROPE: ("EV_CSA_ROPE", "CSA RoPE Finalize", "csa_rope", LAYER_TASK),
    EV_CSA_APPEND: ("EV_CSA_APPEND", "CSA Cache Append", "csa_append",
                    LAYER_TASK),
    EV_INDEXER_GEMM: ("EV_INDEXER_GEMM", "Indexer-Q Proj (FP8 GEMM)",
                      "indexer_gemm", LAYER_TASK),
    EV_INDEXER_Q_ROPE: ("EV_INDEXER_Q_ROPE", "Indexer Q RoPE",
                        "indexer_q_rope", LAYER_TASK),
    EV_LOGITS: ("EV_LOGITS", "Indexer Logits (MQA)", "logits", LAYER_TASK),
    EV_TOPK: ("EV_TOPK", "Top-k Select", "topk", LAYER_TASK),
    EV_BUILD_SPARSE_INDICES: ("EV_BUILD_SPARSE_INDICES",
                              "Build Sparse Indices", "build_sparse_indices",
                              LAYER_TASK),
    EV_ATTENTION: ("EV_ATTENTION", "Sparse Attention", "attention", LAYER_TASK),
    EV_WO_A_GEMM: ("EV_WO_A_GEMM", "Output-A Proj (BF16 GEMM)", "wo_a_gemm",
                   LAYER_TASK),
    EV_QUANT_O_MID: ("EV_QUANT_O_MID", "Quantize O-mid to FP8", "quant_o_mid",
                     LAYER_TASK),
    EV_WO_B_GEMM: ("EV_WO_B_GEMM", "Output-B Proj (FP8 GEMM)", "wo_b_gemm",
                   LAYER_TASK),

    EV_ATTN_QUERY_ROW: ("EV_ATTN_QUERY_ROW", "Attend Query Row",
                        "attn_query_row", LAYER_PHASE),
    EV_ATTN_MASK_COUNT: ("EV_ATTN_MASK_COUNT", "Count Masked Keys",
                         "attn_mask_count", LAYER_PHASE),
    EV_ATTN_STAGE_Q: ("EV_ATTN_STAGE_Q", "Stage Q to SMEM", "attn_stage_q",
                      LAYER_PHASE),
    EV_ATTN_QK_SOFTMAX: ("EV_ATTN_QK_SOFTMAX", "Scores QK^T + Softmax",
                         "attn_qk_softmax", LAYER_PHASE),
    EV_ATTN_SINK_GATE: ("EV_ATTN_SINK_GATE", "Attn-Sink Gate",
                        "attn_sink_gate", LAYER_PHASE),
    EV_ATTN_PV: ("EV_ATTN_PV", "Output P*V", "attn_output_pv", LAYER_PHASE),
    EV_ATTN_INV_ROPE_STORE: ("EV_ATTN_INV_ROPE_STORE", "Inverse-RoPE + Store",
                             "attn_inv_rope_store", LAYER_PHASE),

    EV_LOAD: ("EV_LOAD", "Load KV Tile", "load_kv_tile", LAYER_PHASE),
    EV_MMA: ("EV_MMA", "WGMMA Issue", "wgmma_issue", LAYER_PHASE),
    EV_EPILOGUE: ("EV_EPILOGUE", "Online-Softmax Rescale", "softmax_rescale",
                  LAYER_PHASE),
    EV_TILE_BARRIER: ("EV_TILE_BARRIER", "Tile-Ready Barrier",
                      "tile_ready_barrier", LAYER_PHASE),
    EV_STORE: ("EV_STORE", "Store Tile to GMEM", "store_tile", LAYER_PHASE),
    EV_TAIL_BARRIER: ("EV_TAIL_BARRIER", "Trailing CTA Barrier",
                      "trailing_barrier", LAYER_PHASE),
}

OPERATOR_TASK_IDS = tuple(
    event_id for event_id, entry in sorted(EVENT_TABLE.items())
    if entry[3] == LAYER_TASK
)

# ----------------------------------------------------------------------
# work_id packing: (row << 12) | (slice << 8) | chunk.  It only has to make
# (event_id, work_id) unique on a lane so the decoder can pair begin/end.
# ----------------------------------------------------------------------
WORK_ROW_LSB = 12
WORK_SLICE_LSB = 8
WORK_CHUNK_LSB = 0


def pack_work(row: int = 0, slice_: int = 0, chunk: int = 0) -> int:
    """Host-side mirror of the device work_id packing (pure Python ints)."""
    return (((row & 0xFF) << WORK_ROW_LSB)
            | ((slice_ & 0xF) << WORK_SLICE_LSB)
            | (chunk & 0xFF))


# ----------------------------------------------------------------------
# device helpers
# ----------------------------------------------------------------------
@cute.jit
def capture_slot_of(prof_sel: cute.Tensor) -> Int32:
    """Source-side ``%smid`` filter.

    ``prof_sel[physical_smid]`` is the dense ``capture_slot`` of a selected SM
    and ``-1`` for every SM outside the capture scope.
    """
    physical_smid = Int32(cute.arch.smid())
    return Int32(prof_sel[physical_smid])


@cute.jit
def make_tag(is_start: Int32, event_id: Int32, warp: Int32,
             capture_slot: Int32, logical_cta: Int32, physical_smid: Int32,
             work_id: Int32) -> Int64:
    tag = Int64(is_start) & Int64(TAG_MASK["is_start"])
    tag = tag | ((Int64(event_id) & Int64(TAG_MASK["event_id"]))
                 << Int64(TAG_LSB["event_id"]))
    tag = tag | ((Int64(warp) & Int64(TAG_MASK["warp"]))
                 << Int64(TAG_LSB["warp"]))
    tag = tag | ((Int64(capture_slot) & Int64(TAG_MASK["capture_slot"]))
                 << Int64(TAG_LSB["capture_slot"]))
    tag = tag | ((Int64(logical_cta) & Int64(TAG_MASK["logical_cta"]))
                 << Int64(TAG_LSB["logical_cta"]))
    tag = tag | ((Int64(physical_smid) & Int64(TAG_MASK["physical_smid"]))
                 << Int64(TAG_LSB["physical_smid"]))
    tag = tag | ((Int64(work_id) & Int64(TAG_MASK["work_id"]))
                 << Int64(TAG_LSB["work_id"]))
    return tag


@cute.jit
def rec(
    prof_buf: cute.Tensor, prof_cnt: cute.Tensor, prof_ovf: cute.Tensor,
    prof_level: cutlass.Constexpr, min_level: cutlass.Constexpr,
    capture_slot: Int32, logical_cta: Int32, physical_smid: Int32,
    tidx: Int32, event_id: cutlass.Constexpr, work_id: Int32,
    is_start: cutlass.Constexpr,
) -> None:
    """Append one ``(timestamp_ns, tag)`` record on this warp's exclusive lane.

    Compiled away entirely when ``prof_level < min_level``.  Emits nothing when
    this CTA's physical SM is outside the selected capture scope.  Contains no
    barrier and no warp-wide vote, so it is safe inside a divergent consumer
    region and cannot change the original control flow.
    """
    if const_expr(prof_level >= min_level):
        warp = tidx >> Int32(5)
        if (tidx & Int32(31)) == Int32(0):
            if capture_slot >= Int32(0):
                lane = capture_slot * Int32(WARPS_PER_CTA) + warp
                cnt = Int32(prof_cnt[lane])
                if cnt < Int32(MAX_RECORDS_PER_LANE):
                    ts = Int64(nvvm.read_ptx_sreg_globaltimer(T.i64()))
                    prof_buf[lane, cnt, Int32(0)] = ts
                    prof_buf[lane, cnt, Int32(1)] = make_tag(
                        Int32(is_start), Int32(event_id), warp,
                        capture_slot, logical_cta, physical_smid, work_id)
                    prof_cnt[lane] = cnt + Int32(1)
                else:
                    # explicit overflow: stop recording, flag the lane, let the
                    # host fail the run.  Records are never silently lost.
                    prof_ovf[lane] = Int32(1)


@cute.jit
def rec_begin(prof_buf, prof_cnt, prof_ovf,
              prof_level: cutlass.Constexpr, min_level: cutlass.Constexpr,
              capture_slot: Int32, logical_cta: Int32, physical_smid: Int32,
              tidx: Int32, event_id: cutlass.Constexpr,
              work_id: Int32) -> None:
    rec(prof_buf, prof_cnt, prof_ovf, prof_level, min_level,
        capture_slot, logical_cta, physical_smid, tidx, event_id, work_id,
        REC_BEGIN)


@cute.jit
def rec_end(prof_buf, prof_cnt, prof_ovf,
            prof_level: cutlass.Constexpr, min_level: cutlass.Constexpr,
            capture_slot: Int32, logical_cta: Int32, physical_smid: Int32,
            tidx: Int32, event_id: cutlass.Constexpr,
            work_id: Int32) -> None:
    rec(prof_buf, prof_cnt, prof_ovf, prof_level, min_level,
        capture_slot, logical_cta, physical_smid, tidx, event_id, work_id,
        REC_END)


# ----------------------------------------------------------------------
# record context
#
# ``ctx`` is a plain tuple, and the layer helpers below are plain Python
# functions that unpack it.  This is not a style choice: CuTe DSL forbids
# calling a closure that *captures* dynamic SSA values from inside dynamic
# control flow (``cutlass.range``), and its own remedy is "pass the captured
# values as arguments".  Threading them through ``ctx`` is exactly that, so the
# same helpers work at kernel top level and inside a per-KV-chunk loop.
# ----------------------------------------------------------------------
def make_ctx(prof_buf, prof_cnt, prof_ovf, prof_level, capture_slot,
             logical_cta, physical_smid, tidx, record_phases=True):
    """Pack every value the layer helpers need into one positional tuple."""
    return (prof_buf, prof_cnt, prof_ovf, prof_level, capture_slot,
            logical_cta, physical_smid, tidx, record_phases)


def _emit(ctx, event_id, work_id, min_level, is_start):
    pb, pc, po, level, slot, cta, smid, tidx, _record_phases = ctx
    if is_start:
        rec_begin(pb, pc, po, level, min_level, slot, cta, smid, tidx,
                  event_id, work_id)
    else:
        rec_end(pb, pc, po, level, min_level, slot, cta, smid, tidx,
                event_id, work_id)


# ---- layer=task: one per grid_sync compute segment -------------------
# Never gated by ``record_phases``.  A task-only delivery keeps exactly these
# spans, and they are the spans the operator-coverage gate measures.
def task_begin(ctx, event_id, work_id=0):
    _emit(ctx, event_id, work_id, PROF_FULL, True)


def task_end(ctx, event_id, work_id=0):
    _emit(ctx, event_id, work_id, PROF_FULL, False)


# ---- layer=phase: operator internals --------------------------------
# Gated by the compile-time ``record_phases`` flag (``ctx[8]``).  Only the
# record append is gated; the surrounding compute and every barrier always run
# unchanged, so a task-only build is still the same kernel.
def phase_begin(ctx, event_id, work_id=0):
    if const_expr(ctx[8]):
        _emit(ctx, event_id, work_id, PROF_FULL, True)


def phase_end(ctx, event_id, work_id=0):
    if const_expr(ctx[8]):
        _emit(ctx, event_id, work_id, PROF_FULL, False)


# ---- layer=ignored: whole-kernel anchor ------------------------------
# Recorded already at PROF_ENTRY_ONLY because SW/HW alignment needs the entry /
# release timestamps and the overhead denominator needs the kernel span.  The
# emitter MUST drop it: it is never a rendered span, and the coverage gate never
# counts it.  Drawing it is how a 44.9%-covered timeline once reported 100%.
def anchor_begin(ctx, event_id=EV_KERNEL, work_id=0):
    _emit(ctx, event_id, work_id, PROF_ENTRY_ONLY, True)


def anchor_end(ctx, event_id=EV_KERNEL, work_id=0):
    _emit(ctx, event_id, work_id, PROF_ENTRY_ONLY, False)


# ---- layer=ignored: grid_sync rendezvous -----------------------------
# Diagnostics only (it shows how much of a segment boundary is wait).  Declared
# ``ignored`` in the ABI, and additionally gated by ``record_phases`` so a
# task-only build drops it.  A barrier record is NOT a substitute for the
# operator task around the compute: that substitution is the original bug.
def barrier_begin(ctx, event_id, work_id=0):
    if const_expr(ctx[8]):
        _emit(ctx, event_id, work_id, PROF_FULL, True)


def barrier_end(ctx, event_id, work_id=0):
    if const_expr(ctx[8]):
        _emit(ctx, event_id, work_id, PROF_FULL, False)
