#pragma once
// WeaveProfiler (`wp`) canonical HIP/ROCm recorder header (CDNA wave64, gfx942 / MI308X).
//
// Build switches (first-class WP_* spelling, mapped onto FI_* before defaults are applied):
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
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <hip/hip_runtime.h>

#ifndef FI_LEVEL
#define FI_LEVEL 0
#endif
#ifndef FI_TIMER
#define FI_TIMER 0
#endif
#ifndef FI_GRAN
#define FI_GRAN 0
#endif
#ifndef FI_GROUPS
#define FI_GROUPS 0xffffffffu
#endif

#if defined(__gfx942__)
#define FI_ARCH_STR "gfx942"
#elif defined(__gfx941__)
#define FI_ARCH_STR "gfx941"
#elif defined(__gfx940__)
#define FI_ARCH_STR "gfx940"
#elif defined(__gfx90a__)
#define FI_ARCH_STR "gfx90a"
#elif defined(__gfx90__)
#define FI_ARCH_STR "gfx90"
#elif defined(__gfx950__)
#define FI_ARCH_STR "gfx950"
#elif defined(__gfx1030__)
#define FI_ARCH_STR "gfx1030"
#elif defined(__gfx1100__)
#define FI_ARCH_STR "gfx1100"
#elif defined(__gfx1101__)
#define FI_ARCH_STR "gfx1101"
#elif defined(__gfx1103__)
#define FI_ARCH_STR "gfx1103"
#elif defined(__gfx1200__)
#define FI_ARCH_STR "gfx1200"
#elif defined(__gfx1201__)
#define FI_ARCH_STR "gfx1201"
#else
#define FI_ARCH_STR "gfx-unknown"
#endif

#define FI_TAG_END 0x80000000u     // 区间结束
#define FI_TAG_OPEN 0x40000000u    // FI_OPEN：低 16 位=角色号；aux=HW_ID 编码（见头注）；t=计时器
#define FI_TAG_MARK 0x20000000u    // 瞬时事件
#define FI_TAG_CLK 0x10000000u     // 锚点的 cycle 半边（紧跟 OPEN/CLOSE 之后）；aux=0 开、1 关
#define FI_TAG_CLOSE 0x08000000u   // FI_CLOSE
#define FI_TAG_IDMASK 0x0fffu

struct FiRec {
    uint64_t t;    // FI_TIMER=0：s_memrealtime tick（meta.timer_hz 换算 ns）；=1：cycle（OPEN/CLOSE 仍是 s_memrealtime）
    uint32_t tag;  // 事件号 | 标志位
    uint32_t aux;
};
struct FiBuf {
    FiRec* rec;          // [n_slots][cap]，launch 前清零
    uint32_t* overflow;  // bit0=某泳道写满；bit1=slot 越界
    uint32_t* cur;       // [n_slots] 每泳道游标（支持 __device__ 子函数免改签名跨作用域续写）
    uint64_t timer_hz;   // s_memrealtime 的固定频率（host 填；device 只读，用于超时换算）
    uint32_t cap;
    uint32_t warps_per_cta;
    uint32_t n_slots;
    char arch[32];       // 运行设备的 arch 名（fi_alloc 从 hipGetDeviceProperties.gcnArchName 取， ':' 前段）
};
struct FiCtx {
    FiRec* p;      // 本泳道基址；非记录线程为 nullptr
    uint32_t n;    // 已写条数（寄存器）
    uint32_t cap;
    uint32_t slot; // 本泳道槽号（供 FiScope 同步 fi_g.cur[slot]）
};

static __device__ FiBuf fi_g;

__device__ __forceinline__ uint64_t fi_gtimer() {
    return __builtin_amdgcn_s_memrealtime();   // s_memrealtime：固定频率墙钟，全卡同一时基
}
__device__ __forceinline__ uint64_t fi_clk() {
    return __builtin_readcyclecounter();       // CU 本地周期计数器
}
__device__ __forceinline__ uint64_t fi_now() {
#if FI_TIMER == 1
    return fi_clk();
#else
    return fi_gtimer();
#endif
}
__device__ __forceinline__ bool fi_u64_mul_add(uint64_t a, uint64_t b, uint64_t c, uint64_t* out) {
    if (a != 0ull && b > UINT64_MAX / a) return false;
    const uint64_t p = a * b;
    if (c > UINT64_MAX - p) return false;
    *out = p + c;
    return true;
}
// HIP 的 threadIdx/blockDim 可组成二维或三维 block。wave 边界扫过的是这个线性编号，
// 不是 threadIdx.x；只用 x 会把同一 wave 拆成多个 writer，也会漏掉真正的 lane 0。
__device__ __forceinline__ uint64_t fi_linear_tid() {
    uint64_t row = 0, linear = 0;
    if (!fi_u64_mul_add((uint64_t)blockDim.y, (uint64_t)threadIdx.z,
                        (uint64_t)threadIdx.y, &row)) return UINT64_MAX;
    if (!fi_u64_mul_add((uint64_t)blockDim.x, row,
                        (uint64_t)threadIdx.x, &linear)) return UINT64_MAX;
    return linear;
}
__device__ __forceinline__ uint32_t fi_lane() {
    return (uint32_t)(fi_linear_tid() & 63ull);  // wave64 内线性 lane 号
}
__device__ __forceinline__ uint32_t fi_hw_aux() {
    // gfx942：HW_ID 寄存器号 4，CU[11:8](4b)、SIMD[5:4](2b)、SE[14:13](2b)；XCC_ID=20，XCC[3:0](4b)。
    // __builtin_amdgcn_s_getreg((size<<11)|(offset<<6)|hwreg)
    const uint32_t cu = __builtin_amdgcn_s_getreg((3u << 11) | (8u << 6) | 4u);
    const uint32_t simd = __builtin_amdgcn_s_getreg((1u << 11) | (4u << 6) | 4u);
    const uint32_t se = __builtin_amdgcn_s_getreg((1u << 11) | (13u << 6) | 4u);
    const uint32_t xcc = __builtin_amdgcn_s_getreg((3u << 11) | (0u << 6) | 20u);
    return (cu & 15u) | ((simd & 3u) << 4) | ((se & 3u) << 8) | ((xcc & 15u) << 12);
}
__device__ __forceinline__ void fi_store(FiCtx& c, uint64_t t, uint32_t tag, uint32_t aux) {
    if (c.p == nullptr) return;
    if (c.n >= c.cap) {
        if (c.n == c.cap && fi_g.overflow != nullptr) atomicOr(fi_g.overflow, 1u);
        c.n = c.cap + 1u;
        return;
    }
    FiRec r;
    r.t = t;
    r.tag = tag;
    r.aux = aux;
    c.p[c.n] = r;          // 16B 一次写齐（编译为 global_store_dwordx4）
    c.n++;
}
// 单基本块零分支 EXEC 掩码流式记录（0 C++ if 分裂、0 s_cbranch、无 memory clobber），
// 溢出 atomicOr 统一提升到 fi_close()。
#if defined(__clang__)
typedef uint32_t fi_hip_u32x4_ __attribute__((ext_vector_type(4)));
#endif
__device__ __forceinline__ void fi_record_pred(FiCtx& c, bool pred, uint32_t tag, uint32_t aux) {
#if defined(__AMDGCN__) && defined(__clang__)
    const uint32_t valid = (pred && (c.p != nullptr)) ? 1u : 0u;
    const uint32_t do_st = ((valid != 0u) && (c.n < c.cap)) ? 1u : 0u;
    const uint32_t do_inc = ((valid != 0u) && (c.n <= c.cap)) ? 1u : 0u;
    FiRec* const dst = c.p + c.n;
    const uint64_t t = fi_now();
    const fi_hip_u32x4_ vec = {(uint32_t)t, (uint32_t)(t >> 32), tag, aux};
    uint64_t saved_exec = 0ull, st_mask = 0ull;
    asm volatile(
        "v_cmp_ne_u32_e64 %[m], %[flag], 0\n\t"
        "s_and_saveexec_b64 %[sv], %[m]\n\t"
        "s_nop 0\n\t"
        "global_store_dwordx4 %[addr], %[v], off nt\n\t"
        "s_mov_b64 exec, %[sv]"
        : [sv] "=&s"(saved_exec), [m] "=&s"(st_mask)
        : [flag] "v"(do_st), [addr] "v"(dst), [v] "v"(vec)
        : "scc");
    c.n += do_inc;
#else
    if (!pred || c.p == nullptr) return;
    if (c.n < c.cap) {
        FiRec r;
        r.t = fi_now();
        r.tag = tag;
        r.aux = aux;
        c.p[c.n] = r;
    }
    if (c.n <= c.cap) c.n++;
#endif
}
__device__ __forceinline__ void fi_trans_pred(FiCtx& c, bool pred, uint32_t end_tag, uint32_t beg_tag, uint32_t aux) {
#if defined(__AMDGCN__) && defined(__clang__)
    const uint32_t valid = (pred && (c.p != nullptr)) ? 1u : 0u;
    const uint32_t do_st = ((valid != 0u) && (c.n + 1u < c.cap)) ? 1u : 0u;
    const uint32_t do_inc = ((valid != 0u) && (c.n <= c.cap)) ? 2u : 0u;
    FiRec* const dst0 = c.p + c.n;
    FiRec* const dst1 = dst0 + 1;
    const uint64_t t = fi_now();
    const fi_hip_u32x4_ vec0 = {(uint32_t)t, (uint32_t)(t >> 32), end_tag, aux};
    const fi_hip_u32x4_ vec1 = {(uint32_t)t, (uint32_t)(t >> 32), beg_tag, aux};
    uint64_t saved_exec = 0ull, st_mask = 0ull;
    asm volatile(
        "v_cmp_ne_u32_e64 %[m], %[flag], 0\n\t"
        "s_and_saveexec_b64 %[sv], %[m]\n\t"
        "s_nop 0\n\t"
        "global_store_dwordx4 %[a0], %[v0], off nt\n\t"
        "global_store_dwordx4 %[a1], %[v1], off nt\n\t"
        "s_mov_b64 exec, %[sv]"
        : [sv] "=&s"(saved_exec), [m] "=&s"(st_mask)
        : [flag] "v"(do_st), [a0] "v"(dst0), [a1] "v"(dst1), [v0] "v"(vec0), [v1] "v"(vec1)
        : "scc");
    c.n += do_inc;
#else
    if (!pred || c.p == nullptr) return;
    if (c.n + 1u < c.cap) {
        const uint64_t t = fi_now();
        FiRec r0{t, end_tag, aux};
        FiRec r1{t, beg_tag, aux};
        c.p[c.n] = r0;
        c.p[c.n + 1u] = r1;
    }
    if (c.n <= c.cap) c.n += 2u;
#endif
}
__device__ __forceinline__ void fi_record_now(FiCtx& c, uint32_t tag, uint32_t aux) {
    fi_record_pred(c, true, tag, aux);
}
__device__ __forceinline__ FiCtx fi_open(bool recorder, uint32_t role) {
    FiCtx c;
    c.p = nullptr;
    c.n = 0;
    c.cap = 0;
    c.slot = 0;
    const FiBuf b = fi_g;
    if (!recorder || b.rec == nullptr) return c;
    const uint64_t linear = fi_linear_tid();
    uint64_t zterm = 0, cta = 0, slot = 0;
    bool index_ok = fi_u64_mul_add((uint64_t)gridDim.y, (uint64_t)blockIdx.z,
                                  (uint64_t)blockIdx.y, &zterm);
    index_ok = index_ok && fi_u64_mul_add((uint64_t)gridDim.x, zterm,
                                         (uint64_t)blockIdx.x, &cta);
#if FI_GRAN == 1
    const uint64_t w = 0;  // CU 粒度：每 CTA 一条泳道，host 侧 fi_alloc 强制 warps_per_cta=1
#else
    const uint64_t w = linear >> 6;
#endif
    index_ok = index_ok && fi_u64_mul_add(cta, (uint64_t)b.warps_per_cta, w, &slot);
    if (!index_ok || w >= b.warps_per_cta || slot >= b.n_slots) {
        if (b.overflow != nullptr) atomicOr(b.overflow, 2u);
        return c;
    }
    c.p = b.rec + (size_t)slot * (size_t)b.cap;
    c.cap = b.cap;
    c.slot = (uint32_t)slot;
    const uint64_t g = fi_gtimer();
#if FI_TIMER == 1
    const uint64_t k = fi_clk();
#endif
    fi_store(c, g, FI_TAG_OPEN | (role & 0xffffu), fi_hw_aux());
#if FI_TIMER == 1
    fi_store(c, k, FI_TAG_CLK, 0u);
#endif
    if (b.cur != nullptr) b.cur[c.slot] = c.n;
    return c;
}
__device__ __forceinline__ FiCtx fi_open_subscope(bool recorder) {
    FiCtx c;
    c.p = nullptr;
    c.n = 0;
    c.cap = 0;
    c.slot = 0;
    const FiBuf b = fi_g;
    if (!recorder || b.rec == nullptr || b.cur == nullptr) return c;
    const uint64_t linear = fi_linear_tid();
    uint64_t zterm = 0, cta = 0, slot = 0;
    bool index_ok = fi_u64_mul_add((uint64_t)gridDim.y, (uint64_t)blockIdx.z,
                                  (uint64_t)blockIdx.y, &zterm);
    index_ok = index_ok && fi_u64_mul_add((uint64_t)gridDim.x, zterm,
                                         (uint64_t)blockIdx.x, &cta);
#if FI_GRAN == 1
    const uint64_t w = 0;
#else
    const uint64_t w = linear >> 6;
#endif
    index_ok = index_ok && fi_u64_mul_add(cta, (uint64_t)b.warps_per_cta, w, &slot);
    if (!index_ok || w >= b.warps_per_cta || slot >= b.n_slots) return c;
    c.p = b.rec + (size_t)slot * (size_t)b.cap;
    c.cap = b.cap;
    c.slot = (uint32_t)slot;
    c.n = b.cur[c.slot];
    return c;
}
__device__ __forceinline__ void fi_close(FiCtx& c) {
    if (c.p == nullptr) return;
    if (c.n > c.cap && fi_g.overflow != nullptr) atomicOr(fi_g.overflow, 1u);
    const uint64_t g = fi_gtimer();
#if FI_TIMER == 1
    const uint64_t k = fi_clk();
#endif
    fi_store(c, g, FI_TAG_CLOSE, 0u);
#if FI_TIMER == 1
    fi_store(c, k, FI_TAG_CLK, 1u);
#endif
    if (fi_g.cur != nullptr) fi_g.cur[c.slot] = c.n;
}

struct FiScope {
    FiCtx ctx;
    bool is_root;
    __device__ __forceinline__ FiScope(bool recorder, uint32_t role, bool root)
        : ctx(root ? fi_open(recorder, role) : fi_open_subscope(recorder)), is_root(root) {}
    __device__ __forceinline__ void sync_out() {
        if (ctx.p != nullptr && fi_g.cur != nullptr) fi_g.cur[ctx.slot] = ctx.n;
    }
    __device__ __forceinline__ void sync_in() {
        if (ctx.p != nullptr && fi_g.cur != nullptr) ctx.n = fi_g.cur[ctx.slot];
    }
    __device__ __forceinline__ ~FiScope() {
        if (is_root) {
            fi_close(ctx);
        } else if (ctx.p != nullptr && fi_g.cur != nullptr) {
            fi_g.cur[ctx.slot] = ctx.n;
        }
    }
};

#define FI_GROUP_ON(g) (((FI_GROUPS) >> ((g) & 31)) & 1u)

// FI_GRAN=0：每个 wave64 的 lane 0 记录；FI_GRAN=1：每 CTA 只由线性线程号 0（一维块即 threadIdx.x==0）记录。
#if FI_LEVEL >= 1
#define FI_CTX_PARAM , FiCtx& fi_ctx_
#define FI_CTX_ARG , fi_ctx_
#if FI_GRAN == 1
#define FI_OPEN(role) FiCtx fi_ctx_ = fi_open(fi_linear_tid() == 0ull, (uint32_t)(role))
#define FI_KERNEL_SCOPE(role) FiScope fi_scope_(fi_linear_tid() == 0ull, (uint32_t)(role), true); FiCtx& fi_ctx_ = fi_scope_.ctx
#define FI_DEVICE_SCOPE() FiScope fi_scope_(fi_linear_tid() == 0ull, 0u, false); FiCtx& fi_ctx_ = fi_scope_.ctx
#else
#define FI_OPEN(role) FiCtx fi_ctx_ = fi_open(fi_lane() == 0u, (uint32_t)(role))
#define FI_KERNEL_SCOPE(role) FiScope fi_scope_(fi_lane() == 0u, (uint32_t)(role), true); FiCtx& fi_ctx_ = fi_scope_.ctx
#define FI_DEVICE_SCOPE() FiScope fi_scope_(fi_lane() == 0u, 0u, false); FiCtx& fi_ctx_ = fi_scope_.ctx
#endif
#define FI_OPEN_BY(pred, role) FiCtx fi_ctx_ = fi_open((pred), (uint32_t)(role))
#define FI_CLOSE() fi_close(fi_ctx_)
#define FI_KERNEL_SCOPE_BY(pred, role) FiScope fi_scope_((pred), (uint32_t)(role), true); FiCtx& fi_ctx_ = fi_scope_.ctx
#define FI_DEVICE_SCOPE_BY(pred) FiScope fi_scope_((pred), 0u, false); FiCtx& fi_ctx_ = fi_scope_.ctx
#define FI_SYNC_OUT() fi_scope_.sync_out()
#define FI_SYNC_IN() fi_scope_.sync_in()
#else
#define FI_CTX_PARAM
#define FI_CTX_ARG
#define FI_OPEN(role) do { } while (0)
#define FI_OPEN_BY(pred, role) do { } while (0)
#define FI_CLOSE() do { } while (0)
#define FI_KERNEL_SCOPE(role) do { } while (0)
#define FI_KERNEL_SCOPE_BY(pred, role) do { (void)(pred); } while (0)
#define FI_DEVICE_SCOPE() do { } while (0)
#define FI_DEVICE_SCOPE_BY(pred) do { (void)(pred); } while (0)
#define FI_SYNC_OUT() do { } while (0)
#define FI_SYNC_IN() do { } while (0)
#endif

#if FI_LEVEL >= 2
#define FI_BG(g, id, aux) do { if (FI_GROUP_ON(g)) fi_record_now(fi_ctx_, (uint32_t)(id) & FI_TAG_IDMASK, (uint32_t)(aux)); } while (0)
#define FI_EG(g, id, aux) do { if (FI_GROUP_ON(g)) fi_record_now(fi_ctx_, ((uint32_t)(id) & FI_TAG_IDMASK) | FI_TAG_END, (uint32_t)(aux)); } while (0)
#define FI_MG(g, id, aux) do { if (FI_GROUP_ON(g)) fi_record_now(fi_ctx_, ((uint32_t)(id) & FI_TAG_IDMASK) | FI_TAG_MARK, (uint32_t)(aux)); } while (0)
#define FI_BG_PRED(pred, g, id, aux) do { if (FI_GROUP_ON(g)) fi_record_pred(fi_ctx_, (bool)(pred), (uint32_t)(id) & FI_TAG_IDMASK, (uint32_t)(aux)); } while (0)
#define FI_EG_PRED(pred, g, id, aux) do { if (FI_GROUP_ON(g)) fi_record_pred(fi_ctx_, (bool)(pred), ((uint32_t)(id) & FI_TAG_IDMASK) | FI_TAG_END, (uint32_t)(aux)); } while (0)
#define FI_MG_PRED(pred, g, id, aux) do { if (FI_GROUP_ON(g)) fi_record_pred(fi_ctx_, (bool)(pred), ((uint32_t)(id) & FI_TAG_IDMASK) | FI_TAG_MARK, (uint32_t)(aux)); } while (0)
#define FI_TRANS_PRED(pred, g_end, end_id, g_beg, beg_id, aux) do { if (FI_GROUP_ON(g_end) && FI_GROUP_ON(g_beg)) fi_trans_pred(fi_ctx_, (bool)(pred), ((uint32_t)(end_id) & FI_TAG_IDMASK) | FI_TAG_END, (uint32_t)(beg_id) & FI_TAG_IDMASK, (uint32_t)(aux)); else { FI_EG_PRED(pred, g_end, end_id, aux); FI_BG_PRED(pred, g_beg, beg_id, aux); } } while (0)
#else
#define FI_BG(g, id, aux) do { } while (0)
#define FI_EG(g, id, aux) do { } while (0)
#define FI_MG(g, id, aux) do { } while (0)
#define FI_BG_PRED(pred, g, id, aux) do { (void)(pred); } while (0)
#define FI_EG_PRED(pred, g, id, aux) do { (void)(pred); } while (0)
#define FI_MG_PRED(pred, g, id, aux) do { (void)(pred); } while (0)
#define FI_TRANS_PRED(pred, g_end, end_id, g_beg, beg_id, aux) do { (void)(pred); } while (0)
#endif
#define FI_B(id, aux) FI_BG(0, id, aux)
#define FI_E(id, aux) FI_EG(0, id, aux)
#define FI_M(id, aux) FI_MG(0, id, aux)
#define FI_B_PRED(pred, id, aux) FI_BG_PRED(pred, 0, id, aux)
#define FI_E_PRED(pred, id, aux) FI_EG_PRED(pred, 0, id, aux)
#define FI_M_PRED(pred, id, aux) FI_MG_PRED(pred, 0, id, aux)

// Canonical WeaveProfiler (`WP_*` / `wp_*`) aliases (§3.2)
using WpRec = FiRec;
using WpBuf = FiBuf;
using WpCtx = FiCtx;
using WpScope = FiScope;
#define wp_g fi_g
#define WP_OPEN(role) FI_OPEN(role)
#define WP_OPEN_BY(pred, role) FI_OPEN_BY(pred, role)
#define WP_CLOSE() FI_CLOSE()
#define WP_KERNEL_SCOPE(role) FI_KERNEL_SCOPE(role)
#define WP_KERNEL_SCOPE_BY(pred, role) FI_KERNEL_SCOPE_BY(pred, role)
#define WP_DEVICE_SCOPE() FI_DEVICE_SCOPE()
#define WP_DEVICE_SCOPE_BY(pred) FI_DEVICE_SCOPE_BY(pred)
#define WP_SYNC_OUT() FI_SYNC_OUT()
#define WP_SYNC_IN() FI_SYNC_IN()
#define WP_B(id, aux) FI_B(id, aux)
#define WP_E(id, aux) FI_E(id, aux)
#define WP_M(id, aux) FI_M(id, aux)
#define WP_B_PRED(pred, id, aux) FI_B_PRED(pred, id, aux)
#define WP_E_PRED(pred, id, aux) FI_E_PRED(pred, id, aux)
#define WP_M_PRED(pred, id, aux) FI_M_PRED(pred, id, aux)

// ---------------- host 侧 ----------------
// n_ctas：grid 的 CTA 总数；warps_per_cta：FI_GRAN=0 时按 wave64 上取整（threads/64），FI_GRAN=1 时忽略输入并强制为 1；
// cap：每泳道最大记录数（OPEN/CLOSE 及锚点也占条数）。
static inline bool fi_record_count(uint32_t n_slots, uint32_t cap, size_t* count) {
    if (count == nullptr || n_slots == 0u || cap == 0u) return false;
    if ((size_t)cap > SIZE_MAX / (size_t)n_slots) return false;
    const size_t n = (size_t)n_slots * (size_t)cap;
    if (n > SIZE_MAX / sizeof(FiRec)) return false;
    *count = n;
    return true;
}
static inline bool fi_record_bytes(uint32_t n_slots, uint32_t cap, size_t* bytes) {
    size_t n = 0;
    if (!fi_record_count(n_slots, cap, &n)) return false;
    *bytes = n * sizeof(FiRec);
    return true;
}
static inline hipError_t fi_alloc(FiBuf* b, uint32_t n_ctas, uint32_t warps_per_cta, uint32_t cap) {
    if (b == nullptr) return hipErrorInvalidValue;
    memset(b, 0, sizeof(*b));
    uint32_t slots = 0;
#if FI_GRAN == 1
    // CU 粒度只有线性 0 号线程写；调用方误传 warps_per_cta 也必须折算成每 CTA 一条泳道。
    (void)warps_per_cta;
    b->warps_per_cta = 1u;
    slots = n_ctas;
#else
    if (warps_per_cta == 0u || (n_ctas != 0u && warps_per_cta > UINT32_MAX / n_ctas))
        return hipErrorInvalidValue;
    b->warps_per_cta = warps_per_cta;
    slots = n_ctas * warps_per_cta;
#endif
    size_t all_bytes = 0;
    if (!fi_record_bytes(slots, cap, &all_bytes)) return hipErrorInvalidValue;
    b->cap = cap;
    b->n_slots = slots;
    int device = 0;
    hipError_t e = hipGetDevice(&device);
    if (e != hipSuccess) return e;
    int wall_khz = 0;
    e = hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, device);
    if (e != hipSuccess) return e;
    b->timer_hz = (uint64_t)wall_khz * 1000ull;   // MI308X 实测报 100 MHz
    {   // arch 名在主机侧取不到编译期 __gfx* 宏（主机编译趟不定义），改为运行时查询
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, device) == hipSuccess) {
            snprintf(b->arch, sizeof b->arch, "%s", prop.gcnArchName);
            char* c = strchr(b->arch, ':');
            if (c) *c = '\0';
        }
        if (!b->arch[0]) snprintf(b->arch, sizeof b->arch, "%s", FI_ARCH_STR);
    }
    if (const char* ov = getenv("FI_TIMER_HZ"); ov != nullptr && *ov != '\0')
        b->timer_hz = strtoull(ov, nullptr, 10);
    if (b->timer_hz == 0) return hipErrorInvalidValue;
    e = hipMalloc(&b->rec, all_bytes);
    if (e == hipSuccess) e = hipMalloc(&b->overflow, sizeof(uint32_t));
    if (e == hipSuccess) e = hipMalloc(&b->cur, (size_t)slots * sizeof(uint32_t));
    if (e == hipSuccess) e = hipMemset(b->rec, 0, all_bytes);
    if (e == hipSuccess) e = hipMemset(b->overflow, 0, sizeof(uint32_t));
    if (e == hipSuccess) e = hipMemset(b->cur, 0, (size_t)slots * sizeof(uint32_t));
    if (e != hipSuccess) {
        if (b->rec != nullptr) { hipError_t ignored = hipFree(b->rec); (void)ignored; }
        if (b->overflow != nullptr) { hipError_t ignored = hipFree(b->overflow); (void)ignored; }
        if (b->cur != nullptr) { hipError_t ignored = hipFree(b->cur); (void)ignored; }
        b->rec = nullptr;
        b->overflow = nullptr;
        b->cur = nullptr;
    }
    return e;
}
static inline hipError_t fi_attach(const FiBuf* b) {
    FiBuf z;
    memset(&z, 0, sizeof(z));
    return hipMemcpyToSymbol(HIP_SYMBOL(fi_g), b ? b : &z, sizeof(FiBuf));
}
// 要采的那次 launch 前调用（异步，排在同一 stream 上）。
static inline hipError_t fi_reset(const FiBuf* b, hipStream_t st = 0) {
    if (b == nullptr || b->rec == nullptr || b->overflow == nullptr) return hipErrorInvalidValue;
    size_t all_bytes = 0;
    if (!fi_record_bytes(b->n_slots, b->cap, &all_bytes)) return hipErrorInvalidValue;
    hipError_t e = hipMemsetAsync(b->rec, 0, all_bytes, st);
    if (e == hipSuccess) e = hipMemsetAsync(b->overflow, 0, sizeof(uint32_t), st);
    if (e == hipSuccess && b->cur != nullptr) e = hipMemsetAsync(b->cur, 0, (size_t)b->n_slots * sizeof(uint32_t), st);
    return e;
}
static inline std::string fi_json_escape(const char* s) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    for (const unsigned char* p = (const unsigned char*)(s ? s : ""); *p; ++p) {
        const unsigned char ch = *p;
        if (ch == '"' || ch == '\\') { out.push_back('\\'); out.push_back((char)ch); }
        else if (ch < 0x20u) {
            out += "\\u00";
            out.push_back(hex[ch >> 4]);
            out.push_back(hex[ch & 15u]);
        } else out.push_back((char)ch);
    }
    return out;
}
// 在调用方 JSON 对象末尾注入固定字段。对象为空（含 {}、{ }、带空白）时不能生成尾逗号；
// 非对象输入按空对象处理，保证 FILITE02 的 meta 段始终可被 JSON 解码。
static inline std::string fi_meta_augment(const char* meta_json, const FiBuf* b) {
    std::string m = (meta_json && *meta_json) ? std::string(meta_json) : std::string("{}");
    const size_t first = m.find_first_not_of(" \t\r\n");
    const size_t last = m.find_last_not_of(" \t\r\n");
    if (first == std::string::npos || last == std::string::npos || first >= last ||
        m[first] != '{' || m[last] != '}') {
        m = "{}";
    }
    const size_t open = m.find_first_not_of(" \t\r\n");
    const size_t close = m.find_last_of('}');
    if (open == std::string::npos || close == std::string::npos || close <= open) return m;
    const std::string body = m.substr(open + 1u, close - open - 1u);
    const bool empty_obj = body.find_first_not_of(" \t\r\n") == std::string::npos;
    const std::string arch = fi_json_escape(b && b->arch[0] ? b->arch : FI_ARCH_STR);
    char inject[320];
    const uint64_t timer_hz = b ? b->timer_hz : 0ull;
    snprintf(inject, sizeof inject, "\"timer_hz\":%llu,\"gran\":%d,\"arch\":\"%s\"",
             (unsigned long long)timer_hz, (int)FI_GRAN, arch.c_str());
    m.insert(close, std::string(empty_obj ? "" : ",") + inject);
    return m;
}
// 转储：magic "FILITE02" | u32 n_slots, cap, warps_per_cta, overflow, timer, meta_len | meta(JSON) | u32 cnt[n_slots] | 紧凑记录
// meta 在调用方 JSON 上注入三个字段：timer_hz（s_memrealtime 频率）、gran（FI_GRAN）、arch（编译目标架构）。
static inline int fi_dump(const FiBuf* b, const char* path, const char* meta_json) {
    if (b == nullptr || b->rec == nullptr || b->overflow == nullptr || path == nullptr) return -4;
    size_t record_count = 0;
    if (!fi_record_count(b->n_slots, b->cap, &record_count)) return -4;
    if (hipDeviceSynchronize() != hipSuccess) return -1;
    std::vector<FiRec> all(record_count);
    uint32_t ovf = 0;
    if (hipMemcpy(all.data(), b->rec, all.size() * sizeof(FiRec), hipMemcpyDeviceToHost) != hipSuccess) return -2;
    if (hipMemcpy(&ovf, b->overflow, 4, hipMemcpyDeviceToHost) != hipSuccess) return -2;
    std::vector<uint32_t> cnt(b->n_slots, 0);
    for (uint32_t s = 0; s < b->n_slots; ++s)
        while (cnt[s] < b->cap && all[(size_t)s * b->cap + cnt[s]].tag != 0u) ++cnt[s];
    FILE* f = fopen(path, "wb");
    if (!f) {
        // 同 CUDA 侧：把 errno 原文打出来，别只返回 -3（fi_bugs_h20-sk-qwen B10）。
        fprintf(stderr,
                "fi_dump: fopen(\"%s\", \"wb\") 失败：%s (errno=%d)。"
                "rocprofv3/nsys 下被测进程可能写不了工作目录：把转储路径换成 /tmp 下的文件，"
                "采集结束后再由 shell 拷回；或先把目标目录 chmod 0777。\n",
                path, strerror(errno), errno);
        return -3;
    }
    const std::string meta = fi_meta_augment(meta_json, b);
    if (meta.size() > UINT32_MAX) { fclose(f); return -4; }
    const uint32_t ml = (uint32_t)meta.size();
    const uint32_t hdr[6] = {b->n_slots, b->cap, b->warps_per_cta, ovf, (uint32_t)FI_TIMER, ml};
    fwrite("FILITE02", 1, 8, f);
    fwrite(hdr, 4, 6, f);
    fwrite(meta.data(), 1, ml, f);
    fwrite(cnt.data(), 4, cnt.size(), f);
    for (uint32_t s = 0; s < b->n_slots; ++s)
        if (cnt[s]) fwrite(&all[(size_t)s * b->cap], sizeof(FiRec), cnt[s], f);
    fclose(f);
    return (int)ovf;
}
static inline void fi_free(FiBuf* b) {
    // hipFree 带 nodiscard：直接丢掉返回值会让 hipcc 报 -Wunused-value（-Wno-unused-result 压不住），
    // 而 fi_logcheck 把编译告警当作日志不干净。这里显式接住并丢弃。
    hipError_t e_rec_ = hipFree(b->rec);
    hipError_t e_ovf_ = hipFree(b->overflow);
    hipError_t e_cur_ = (b->cur != nullptr) ? hipFree(b->cur) : hipSuccess;
    (void)e_rec_;
    (void)e_ovf_;
    (void)e_cur_;
    memset(b, 0, sizeof(*b));
}

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
