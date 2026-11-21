#pragma once
// WeaveProfiler (`wp`) canonical HIP/ROCm recorder header (CDNA wave64, gfx942 / MI308X).
//
// Build switches (first-class WP_* spelling, mapped onto FI_* BEFORE fi_lite_hip.h applies defaults):
//   -DWP_LEVEL=0|1|2   -DWP_TIMER=0|1 (default 1 here: readcyclecounter + s_memrealtime anchors)
//   -DWP_GRAN=0|1      -DWP_GROUPS=<mask>
// Device API: WP_OPEN / WP_OPEN_BY / WP_B / WP_E / WP_M / WP_BG / WP_EG / WP_MG / *_PRED / WP_TRANS_PRED /
//             WP_CLOSE / WP_CTX_PARAM / WP_CTX_ARG / WP_SYNC_CURSOR_OUT / WP_SYNC_CURSOR_IN.
// Host API:   wp_alloc / wp_attach / wp_reset / wp_dump / wp_free / wp_host_init / wp_host_dump.
#if defined(WP_LEVEL) && !defined(FI_LEVEL)
#define FI_LEVEL WP_LEVEL
#endif
#if defined(WP_TIMER) && !defined(FI_TIMER)
#define FI_TIMER WP_TIMER
#endif
#if defined(WP_GRAN) && !defined(FI_GRAN)
#define FI_GRAN WP_GRAN
#endif
#if defined(WP_GROUPS) && !defined(FI_GROUPS)
#define FI_GROUPS WP_GROUPS
#endif
#ifndef FI_TIMER
#define FI_TIMER 1
#endif
#include "fi_lite_hip.h"
#if defined(WP_LEVEL) && ((WP_LEVEL) != (FI_LEVEL))
#error "WP_LEVEL and FI_LEVEL are both defined with different values"
#endif
#if defined(WP_TIMER) && ((WP_TIMER) != (FI_TIMER))
#error "WP_TIMER and FI_TIMER are both defined with different values"
#endif
#if defined(WP_GRAN) && ((WP_GRAN) != (FI_GRAN))
#error "WP_GRAN and FI_GRAN are both defined with different values"
#endif
#ifndef WP_LEVEL
#define WP_LEVEL FI_LEVEL
#endif
#ifndef WP_TIMER
#define WP_TIMER FI_TIMER
#endif
#ifndef WP_GRAN
#define WP_GRAN FI_GRAN
#endif
#ifndef WP_GROUPS
#define WP_GROUPS FI_GROUPS
#endif

// The public WP_OPEN/WP_CLOSE, WP_B/WP_E, WpScope and wp_g aliases share
// the legacy recorder's storage, cursor and FILITE02 representation.
static_assert(sizeof(WpRec) == 16, "WeaveProfiler records must be 16 bytes");

#define WP_TAG_END FI_TAG_END
#define WP_TAG_OPEN FI_TAG_OPEN
#define WP_TAG_MARK FI_TAG_MARK
#define WP_TAG_CLK FI_TAG_CLK
#define WP_TAG_CLOSE FI_TAG_CLOSE
#define WP_TAG_IDMASK FI_TAG_IDMASK
#define WP_CTX_PARAM FI_CTX_PARAM
#define WP_CTX_ARG FI_CTX_ARG
#define WP_GROUP_ON(g) FI_GROUP_ON(g)
#define WP_BG(g, id, aux) FI_BG(g, id, aux)
#define WP_EG(g, id, aux) FI_EG(g, id, aux)
#define WP_MG(g, id, aux) FI_MG(g, id, aux)
#define WP_BG_PRED(pred, g, id, aux) FI_BG_PRED(pred, g, id, aux)
#define WP_EG_PRED(pred, g, id, aux) FI_EG_PRED(pred, g, id, aux)
#define WP_MG_PRED(pred, g, id, aux) FI_MG_PRED(pred, g, id, aux)
#define WP_TRANS_PRED(pred, g_end, end_id, g_beg, beg_id, aux) FI_TRANS_PRED(pred, g_end, end_id, g_beg, beg_id, aux)

#if FI_LEVEL >= 1
__device__ __forceinline__ void wp_sync_cursor_out_ctx(const FiCtx& c) {
    if (c.p != nullptr && fi_g.cur != nullptr && c.slot < fi_g.n_slots) fi_g.cur[c.slot] = c.n;
}
__device__ __forceinline__ void wp_sync_cursor_in_ctx(FiCtx& c) {
    if (c.p != nullptr && fi_g.cur != nullptr && c.slot < fi_g.n_slots) c.n = fi_g.cur[c.slot];
}
#define WP_SYNC_CURSOR_OUT() do { wp_sync_cursor_out_ctx(fi_ctx_); } while (0)
#define WP_SYNC_CURSOR_IN() do { wp_sync_cursor_in_ctx(fi_ctx_); } while (0)
#else
#define WP_SYNC_CURSOR_OUT() do {} while (0)
#define WP_SYNC_CURSOR_IN() do {} while (0)
#endif

static inline hipError_t wp_alloc(WpBuf* b, uint32_t n_ctas, uint32_t warps_per_cta, uint32_t cap) {
    return fi_alloc(b, n_ctas, warps_per_cta, cap);
}
static inline hipError_t wp_attach(const WpBuf* b) { return fi_attach(b); }
static inline hipError_t wp_reset(const WpBuf* b, hipStream_t st = 0) { return fi_reset(b, st); }
static inline int wp_dump(const WpBuf* b, const char* path, const char* meta_json = "{}") {
    return fi_dump(b, path, meta_json);
}
static inline void wp_free(WpBuf* b) { fi_free(b); }

static inline hipError_t wp_host_init(WpBuf* buf, uint32_t ctas, uint32_t warps_per_cta, uint32_t capacity) {
    hipError_t status = fi_alloc(buf, ctas, warps_per_cta, capacity);
    if (status == hipSuccess) status = fi_attach(buf);
    if (status != hipSuccess) fi_free(buf);
    return status;
}

static inline int wp_host_dump(const WpBuf* buf, const char* path, const char* meta = "{}") {
    return fi_dump(buf, path, meta);
}

// Optional zero-boilerplate C ABI host exports (`-DWP_DEFINE_HOST_API` or `-DFI_DEFINE_HOST_API`)
// for multi-file PyTorch / ctypes `.so` HIP megakernels (zero __global__ / host call-stack changes).
#if defined(WP_DEFINE_HOST_API) || defined(FI_DEFINE_HOST_API)
static WpBuf g_wp_hip_host_buf_{};
extern "C" int wp_init(uint32_t n_ctas, uint32_t warps_per_cta, uint32_t cap) {
    if (g_wp_hip_host_buf_.rec != nullptr) fi_free(&g_wp_hip_host_buf_);
    hipError_t e = fi_alloc(&g_wp_hip_host_buf_, n_ctas, warps_per_cta, cap);
    if (e != hipSuccess) return (int)e;
    return (int)fi_attach(&g_wp_hip_host_buf_);
}
extern "C" int wp_reset(uint64_t stream_ptr) {
    if (g_wp_hip_host_buf_.rec == nullptr) return -1;
    return (int)fi_reset(&g_wp_hip_host_buf_, reinterpret_cast<hipStream_t>(stream_ptr));
}
extern "C" int wp_dump(const char* path, const char* meta_json) {
    if (g_wp_hip_host_buf_.rec == nullptr) return -1;
    return fi_dump(&g_wp_hip_host_buf_, path, meta_json);
}
extern "C" void wp_free() {
    if (g_wp_hip_host_buf_.rec != nullptr) {
        (void)fi_attach(nullptr);
        fi_free(&g_wp_hip_host_buf_);
    }
}
#endif
