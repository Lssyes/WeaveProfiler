#pragma once
// WeaveProfiler (`wp`) canonical CUDA recorder header.
//
// Build switches (first-class WP_* spelling; mapped onto the legacy FI_* switches inside fi_lite.cuh
// before any default is applied):
//   -DWP_LEVEL=0|1|2        0 = all macros empty (production timing), 1 = OPEN/CLOSE only, 2 = all events
//   -DWP_TIMER=0|1          0 = %globaltimer per event, 1 = %clock64 per event + OPEN/CLOSE anchors
//   -DWP_GRAN=0|1           0 = one lane per warp (lane 0 records), 1 = one lane per CTA (thread 0 records)
//   -DWP_GROUPS=<mask>      event-group bitmask for WP_BG/WP_EG/WP_MG (default: all groups)
//   -DWP_CTX_1R             1-register bit-packed recorder context (0 B SMEM)
//   -DWP_CTX_1R_NOINLINE    WP_CTX_1R + __noinline__ cold-path helpers
//   -DWP_SMEM_ZREG          16 B static __shared__ context + __noinline__ helpers (0 live registers)
//   -DWP_DEFINE_HOST_API    export extern "C" wp_init / wp_reset / wp_dump / wp_free for ctypes .so builds
//
// Device API:  WP_OPEN / WP_OPEN_BY / WP_B / WP_E / WP_M / WP_BG / WP_EG / WP_MG / *_PRED / WP_TRANS_PRED /
//              WP_ROLE_SWITCH / WP_CLOSE / WP_CTX_PARAM / WP_CTX_ARG / WP_KERNEL_SCOPE / WP_DEVICE_SCOPE /
//              WP_SYNC_CURSOR_OUT / WP_SYNC_CURSOR_IN.
// Host API:    wp_alloc / wp_attach / wp_reset / wp_dump / wp_free / wp_host_init / wp_host_dump.
// The WP_* / wp_* names share the legacy recorder's storage, cursor and FILITE02 record representation;
// 4_validate/tools/wp_decode.py decodes both FILITE02 and WPLITE02 dumps.
#include "fi_lite.cuh"

static_assert(sizeof(WpRec) == 16, "WeaveProfiler records must be 16 bytes");

// WP_SYNC_CURSOR_OUT() publishes the live per-lane record cursor to wp_g.cur[slot];
// WP_SYNC_CURSOR_IN() reloads it (e.g. after a callee __device__ function appended records through
// WP_DEVICE_SCOPE).  The cursor lives in a different place for every recorder-context mode, so each
// mode gets its own implementation instead of poking `fi_ctx_.p` (which does not exist under
// -DWP_CTX_1R or -DWP_SMEM_ZREG):
//   FI_SMEM_ZREG / WP_SMEM_ZREG : FiSmemCtx in static __shared__ (slot recovered from the lane base pointer)
//   WP_CTX_1R(_NOINLINE)        : bit-packed WpCtx1R::state (slot = bits 30..15, cursor = bits 14..0)
//   default                     : 5-register FiCtx {p, n, cap, slot}
#if FI_LEVEL >= 1
#if defined(FI_SMEM_ZREG)
__device__ __forceinline__ void wp_sync_cursor_out_zreg(const FiSmemCtx* sc) {
    const FiBuf b = fi_g;
    if (sc->p != nullptr && b.cur != nullptr && b.rec != nullptr && b.cap != 0u) {
        const uint32_t slot = (uint32_t)((size_t)(sc->p - b.rec) / (size_t)b.cap);
        if (slot < b.n_slots) b.cur[slot] = sc->n;
    }
}
__device__ __forceinline__ void wp_sync_cursor_in_zreg(FiSmemCtx* sc) {
    const FiBuf b = fi_g;
    if (sc->p != nullptr && b.cur != nullptr && b.rec != nullptr && b.cap != 0u) {
        const uint32_t slot = (uint32_t)((size_t)(sc->p - b.rec) / (size_t)b.cap);
        if (slot < b.n_slots) sc->n = b.cur[slot];
    }
}
#define WP_SYNC_CURSOR_OUT() do { if (fi_recorder()) wp_sync_cursor_out_zreg(FI_ZREG_SLOT()); } while (0)
#define WP_SYNC_CURSOR_IN() do { if (fi_recorder()) wp_sync_cursor_in_zreg(FI_ZREG_SLOT()); } while (0)
#elif defined(WP_CTX_1R)
__device__ __forceinline__ void wp_sync_cursor_out_1r(const WpCtx1R& c) {
    if ((int32_t)c.state >= 0 && fi_g.cur != nullptr) {
        const uint32_t slot = (c.state >> WP_1R_SLOT_SHIFT) & WP_1R_SLOT_MASK;
        if (slot < fi_g.n_slots) fi_g.cur[slot] = c.state & WP_1R_CUR_MASK;
    }
}
__device__ __forceinline__ void wp_sync_cursor_in_1r(WpCtx1R& c) {
    if ((int32_t)c.state >= 0 && fi_g.cur != nullptr) {
        const uint32_t slot = (c.state >> WP_1R_SLOT_SHIFT) & WP_1R_SLOT_MASK;
        if (slot >= fi_g.n_slots) return;
        const uint32_t eff_cap = (fi_g.cap > WP_1R_CUR_MASK) ? WP_1R_CUR_MASK : fi_g.cap;
        const uint32_t n = fi_g.cur[slot];
        c.state = (n >= eff_cap) ? ((slot << WP_1R_SLOT_SHIFT) | WP_1R_DISABLED)
                                 : ((slot << WP_1R_SLOT_SHIFT) | (n & WP_1R_CUR_MASK));
    }
}
#define WP_SYNC_CURSOR_OUT() do { wp_sync_cursor_out_1r(fi_ctx_); } while (0)
#define WP_SYNC_CURSOR_IN() do { wp_sync_cursor_in_1r(fi_ctx_); } while (0)
#else
__device__ __forceinline__ void wp_sync_cursor_out_ctx(const FiCtx& c) {
    if (c.p != nullptr && fi_g.cur != nullptr && c.slot < fi_g.n_slots) fi_g.cur[c.slot] = c.n;
}
__device__ __forceinline__ void wp_sync_cursor_in_ctx(FiCtx& c) {
    if (c.p != nullptr && fi_g.cur != nullptr && c.slot < fi_g.n_slots) c.n = fi_g.cur[c.slot];
}
#define WP_SYNC_CURSOR_OUT() do { wp_sync_cursor_out_ctx(fi_ctx_); } while (0)
#define WP_SYNC_CURSOR_IN() do { wp_sync_cursor_in_ctx(fi_ctx_); } while (0)
#endif
#else
#define WP_SYNC_CURSOR_OUT() do {} while (0)
#define WP_SYNC_CURSOR_IN() do {} while (0)
#endif

// Runtime `.cubin` JIT binder (for DeepGEMM / Weave `cudaLibrary_t` runtime-compiled kernels such as
// `sm90_fp8_mqa_logits.cuh`, `sm100_sparse_mqa_logits.cuh`, `sm90_fp8_mega_moe.cuh`): binds `wp_fi_c`
// inside a dynamically loaded `cudaLibrary_t` with zero `__global__` argument changes and zero extra
// `STO_ENTRY` symbols.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
static inline cudaError_t wp_attach_library(cudaLibrary_t lib, const WpBuf* b) {
    WpBuf z = {};
    const WpBuf* src = b ? b : &z;
    void* sym_c = nullptr;
    size_t sym_bytes = 0;
    cudaError_t e = cudaLibraryGetGlobal(&sym_c, &sym_bytes, lib, "wp_fi_c");
    if (e != cudaSuccess) return e;
    if (sym_c == nullptr || sym_bytes < sizeof(WpBuf)) return cudaErrorInvalidValue;
    return cudaMemcpy(sym_c, src, sizeof(WpBuf), cudaMemcpyHostToDevice);
}
#endif
