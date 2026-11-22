#pragma once
// WeaveProfiler (`wp`) canonical CUDA recorder header.
//
// Build switches (first-class WP_* spelling; first-class WP_* switches
// before any default is applied):
//   -DWP_LEVEL=0|1|2        0 = all macros empty (production timing), 1 = OPEN/CLOSE only, 2 = all events
//   -DWP_TIMER=0|1          0 = %globaltimer per event, 1 = %clock64 per event + OPEN/CLOSE anchors
//   -DWP_GRAN=0|1           0 = one lane per warp (lane 0 records), 1 = one lane per CTA (thread 0 records)
//   -DWP_GROUPS=<mask>      event-group bitmask for WP_BG/WP_EG/WP_MG (default: all groups)
//   -DWP_CTX_REG            1-register bit-packed recorder context (0 B SMEM)
//   -DWP_CTX_REG_NOINLINE   WP_CTX_REG + __noinline__ cold-path helpers
//   -DWP_CTX_SMEM           16 B static __shared__ context + __noinline__ helpers (0 live registers)
//   -DWP_DEFINE_HOST_API    export extern "C" wp_init / wp_reset / wp_dump / wp_free for ctypes .so builds
//
// Device API:  WP_OPEN / WP_OPEN_BY / WP_B / WP_E / WP_M / WP_BG / WP_EG / WP_MG / *_PRED / WP_TRANS_PRED /
//              WP_ROLE_SWITCH / WP_CLOSE / WP_CTX_PARAM / WP_CTX_ARG / WP_KERNEL_SCOPE / WP_DEVICE_SCOPE /
//              WP_SYNC_CURSOR_OUT / WP_SYNC_CURSOR_IN.
// Host API:    wp_alloc / wp_attach / wp_reset / wp_dump / wp_free / wp_host_init / wp_host_dump.
// The WP_* / wp_* names share the legacy recorder's storage, cursor and FILITE02 record representation;
// weaveprof/pipeline/validate/tools/wp_decode.py decodes both FILITE02 and WPLITE02 dumps.
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

// Canonical WeaveProfiler build switches (first-class `WP_*` spelling).  They are mapped onto the
// legacy FI_* switches BEFORE defaults are applied, so `-DWP_LEVEL=2 -DWP_TIMER=1 -DWP_GRAN=1
// -DWP_GROUPS=0x5` builds exactly the same recorder as the FI_* spelling.  Giving both spellings
// with different values is a hard error (silent precedence would make A/B builds lie).
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
#if defined(WP_LEVEL) && ((WP_LEVEL) != (FI_LEVEL))
#error "WP_LEVEL and FI_LEVEL are both defined with different values"
#endif
#if defined(WP_TIMER) && ((WP_TIMER) != (FI_TIMER))
#error "WP_TIMER and FI_TIMER are both defined with different values"
#endif
#if defined(WP_GRAN) && ((WP_GRAN) != (FI_GRAN))
#error "WP_GRAN and FI_GRAN are both defined with different values"
#endif

#ifndef FI_LEVEL
#define FI_LEVEL 0
#endif
#ifndef FI_TIMER
#define FI_TIMER 0
#endif
#ifndef FI_GROUPS
#define FI_GROUPS 0xffffffffu
#endif
#ifndef FI_GRAN
#define FI_GRAN 0  // 0=warp 粒度（每 warp 的 lane 0 记录）；1=SM 粒度（每 CTA 只由 0 号线程记录，解码按物理 SM 归组）
#endif
#if (FI_GRAN != 0) && (FI_GRAN != 1)
#error "FI_GRAN 只能是 0（warp 粒度）或 1（SM 粒度）"
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

#define FI_TAG_END 0x80000000u         // 区间结束
#define FI_TAG_OPEN 0x40000000u        // FI_OPEN：低 16 位=角色号，aux=smid；t=globaltimer
#define FI_TAG_MARK 0x20000000u        // 瞬时事件
#define FI_TAG_CLK 0x10000000u         // 锚点的 clock64 半边（紧跟在 OPEN/CLOSE 之后）；aux=0 开、1 关
#define FI_TAG_CLOSE 0x08000000u       // FI_CLOSE：t=globaltimer
#define FI_TAG_ROLE_SWITCH 0x04000000u // 动态角色切换（如 Warp 由 Conv1D 转 OutProj），aux=new_role
#define FI_TAG_IDMASK 0x0fffu
#define WP_TAG_END FI_TAG_END
#define WP_TAG_OPEN FI_TAG_OPEN
#define WP_TAG_MARK FI_TAG_MARK
#define WP_TAG_CLK FI_TAG_CLK
#define WP_TAG_CLOSE FI_TAG_CLOSE
#define WP_TAG_ROLE_SWITCH FI_TAG_ROLE_SWITCH
#define WP_TAG_IDMASK FI_TAG_IDMASK

struct FiRec {
    uint64_t t;    // FI_TIMER=0：globaltimer ns；=1：clock64（OPEN/CLOSE 本身仍是 globaltimer）
    uint32_t tag;  // 事件号 | 标志位
    uint32_t aux;
};
struct FiBuf {
    FiRec* rec;          // [n_slots][cap]，launch 前清零
    uint32_t* overflow;  // bit0=某泳道写满；bit1=slot 越界
    uint32_t* cur;       // [n_slots] 每泳道游标（支持 __device__ 子函数免改签名跨作用域续写）
    uint32_t cap;
    uint32_t warps_per_cta;
    uint32_t n_slots;
};
struct FiCtx {
    FiRec* p;      // 本泳道基址；非记录线程为 nullptr
    uint32_t n;    // 已写条数（寄存器）
    uint32_t cap;
    uint32_t slot; // 本泳道槽号（供 FiScope 同步 fi_g.cur[slot]）
};

static __device__ FiBuf fi_g;
#if defined(__CUDACC_RDC__)
static __constant__ FiBuf wp_fi_c;
#else
extern "C" {
__constant__ FiBuf wp_fi_c;
}
#endif

__device__ __forceinline__ FiBuf wp_active_buf() {
    return (wp_fi_c.rec != nullptr) ? wp_fi_c : fi_g;
}

__device__ __forceinline__ uint64_t fi_gtimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ uint64_t fi_clk() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
    return t;
}
__device__ __forceinline__ uint64_t fi_now() {
#if FI_TIMER == 1
    return fi_clk();
#else
    return fi_gtimer();
#endif
}
__device__ __forceinline__ uint32_t fi_smid() {
    uint32_t s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    return s;
}
__device__ __forceinline__ uint32_t fi_lane() {
    uint32_t l;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(l));
    return l;
}
// CTA 线性号（1D/2D/3D grid 都按 x 最快展开，与解码器的 slot→CTA 反算一致）
__device__ __forceinline__ uint32_t fi_cta_id() {
    return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
}
// 谁真正写记录：FI_GRAN=0 每 warp 的 lane 0；FI_GRAN=1 每 CTA 的 0 号线程（threadIdx.x/y/z 全 0）。
// 一维 block 时后者就是 threadIdx.x==0；要求三个分量都为 0 是为了多维 block 下仍然"每 CTA 只有一个记录者"，
// 否则同一 CTA 会有多个线程往同一条泳道交叉写，记录顺序就乱了。
__device__ __forceinline__ bool fi_recorder() {
#if FI_GRAN == 1
    return threadIdx.x == 0u && threadIdx.y == 0u && threadIdx.z == 0u;
#else
    return fi_lane() == 0u;
#endif
}
__device__ __forceinline__ void fi_store(FiCtx& c, uint64_t t, uint32_t tag, uint32_t aux) {
    if (c.p == nullptr) return;
    if (c.n >= c.cap) {
        const FiBuf b = wp_active_buf();
        if (c.n == c.cap && b.overflow != nullptr) atomicOr(b.overflow, 1u);
        c.n = c.cap + 1u;  // 只报一次
        return;
    }
    // 16B 向量写，流式缓存提示（不污染 L2 里业务数据的驻留）
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(c.p + c.n), "r"((uint32_t)t), "r"((uint32_t)(t >> 32)),
                 "r"(tag), "r"(aux));
    c.n++;
}
// 零分支谓词化记录（Branchless PTX Predication，无 memory clobber）：
//   1) 在 K 循环内完全不产生 C++ if/else 基本块分裂（0 额外 BB），也不在循环内调用 atomicOr；
//   2) 不加 "memory" clobber（与 fi_store 一致），保留编译器对业务张量/共享内存的寄存器驻留与
//      LDSM/HMMA/LDGSTS 软流水调度，同时 asm volatile 保证不跨 mbarrier/cp.async/mma 重排；
//   3) 搭配 FI_TIMER=1（%clock64 本地 2 周期计数器 + OPEN/CLOSE 双锚点校准）实现 <10% 低扰动采集。
// 零 C++ 分支、PTX 内部跨阈值跳转记录（LLVM-Opaque / ptxas If-Conversion-Proof Predication）：
//   1) 对 LLVM/cicc 完全零控制流分裂（0 额外 LLVM BB），寄存器分配保持与 L0 一致（178->180 regs）；
//   2) 将地址计算 mad.wide.u32 与条数递增 add.u32 一并纳入 @!%p bra 块内部（8 条指令 > ptxas 4 指令
//      if-conversion 阈值），阻止 ptxas 将跳转逆转为谓词化 CS2R，彻底消除非记录 warp 与非采样 K 步上的
//      CS2R 记分牌（SB0）等待！
__device__ __forceinline__ void fi_record_pred(FiCtx& c, bool pred, uint32_t tag, uint32_t aux) {
    const uint32_t active = (pred && (c.cap != 0u)) ? 1u : 0u;
#if FI_TIMER == 1
    asm volatile(
        "{\n\t"
        ".reg .pred %%p, %%pst, %%pinc;\n\t"
        ".reg .u64 %%t, %%dst;\n\t"
        ".reg .u32 %%tlo, %%thi;\n\t"
        "setp.ne.u32 %%p, %1, 0;\n\t"
        "@!%%p bra L_FI_SKIP_%=;\n\t"
        "setp.lt.u32 %%pst, %0, %3;\n\t"
        "setp.le.u32 %%pinc, %0, %3;\n\t"
        "mad.wide.u32 %%dst, %0, 16, %2;\n\t"
        "@%%pst mov.u64 %%t, %%clock64;\n\t"
        "@%%pst mov.b64 {%%tlo, %%thi}, %%t;\n\t"
        "@%%pst st.global.cs.v4.u32 [%%dst], {%%tlo, %%thi, %4, %5};\n\t"
        "@%%pinc add.u32 %0, %0, 1;\n\t"
        "L_FI_SKIP_%=:\n\t"
        "}"
        : "+r"(c.n)
        : "r"(active), "l"(c.p), "r"(c.cap), "r"(tag), "r"(aux));
#else
    asm volatile(
        "{\n\t"
        ".reg .pred %%p, %%pst, %%pinc;\n\t"
        ".reg .u64 %%t, %%dst;\n\t"
        ".reg .u32 %%tlo, %%thi;\n\t"
        "setp.ne.u32 %%p, %1, 0;\n\t"
        "@!%%p bra L_FI_SKIP_%=;\n\t"
        "setp.lt.u32 %%pst, %0, %3;\n\t"
        "setp.le.u32 %%pinc, %0, %3;\n\t"
        "mad.wide.u32 %%dst, %0, 16, %2;\n\t"
        "@%%pst mov.u64 %%t, %%globaltimer;\n\t"
        "@%%pst mov.b64 {%%tlo, %%thi}, %%t;\n\t"
        "@%%pst st.global.cs.v4.u32 [%%dst], {%%tlo, %%thi, %4, %5};\n\t"
        "@%%pinc add.u32 %0, %0, 1;\n\t"
        "L_FI_SKIP_%=:\n\t"
        "}"
        : "+r"(c.n)
        : "r"(active), "l"(c.p), "r"(c.cap), "r"(tag), "r"(aux));
#endif
}
// 相邻阶段零间隙原子切换（Phase-Transition Timestamp Coalescing）：
//   当前一阶段结束与下一阶段开始紧邻同一指令边界（如 WAIT 结束紧接 LOAD/MMA 开始）时，
//   共享单次计时器读取写出两条 16B 记录，既令 t_end == t_begin 严格零缝隙，又削减 50% 边界计时器读取。
__device__ __forceinline__ void fi_trans_pred(FiCtx& c, bool pred, uint32_t end_tag, uint32_t beg_tag, uint32_t aux) {
    const uint32_t active = (pred && (c.cap != 0u)) ? 1u : 0u;
#if FI_TIMER == 1
    asm volatile(
        "{\n\t"
        ".reg .pred %%p, %%pst, %%pinc;\n\t"
        ".reg .u64 %%t, %%dst0, %%dst1;\n\t"
        ".reg .u32 %%tlo, %%thi, %%np1;\n\t"
        "setp.ne.u32 %%p, %1, 0;\n\t"
        "@!%%p bra L_FI_TSKIP_%=;\n\t"
        "add.u32 %%np1, %0, 1;\n\t"
        "setp.lt.u32 %%pst, %%np1, %3;\n\t"
        "setp.le.u32 %%pinc, %0, %3;\n\t"
        "mad.wide.u32 %%dst0, %0, 16, %2;\n\t"
        "add.u64 %%dst1, %%dst0, 16;\n\t"
        "@%%pst mov.u64 %%t, %%clock64;\n\t"
        "@%%pst mov.b64 {%%tlo, %%thi}, %%t;\n\t"
        "@%%pst st.global.cs.v4.u32 [%%dst0], {%%tlo, %%thi, %4, %6};\n\t"
        "@%%pst st.global.cs.v4.u32 [%%dst1], {%%tlo, %%thi, %5, %6};\n\t"
        "@%%pinc add.u32 %0, %0, 2;\n\t"
        "L_FI_TSKIP_%=:\n\t"
        "}"
        : "+r"(c.n)
        : "r"(active), "l"(c.p), "r"(c.cap), "r"(end_tag), "r"(beg_tag), "r"(aux));
#else
    asm volatile(
        "{\n\t"
        ".reg .pred %%p, %%pst, %%pinc;\n\t"
        ".reg .u64 %%t, %%dst0, %%dst1;\n\t"
        ".reg .u32 %%tlo, %%thi, %%np1;\n\t"
        "setp.ne.u32 %%p, %1, 0;\n\t"
        "@!%%p bra L_FI_TSKIP_%=;\n\t"
        "add.u32 %%np1, %0, 1;\n\t"
        "setp.lt.u32 %%pst, %%np1, %3;\n\t"
        "setp.le.u32 %%pinc, %0, %3;\n\t"
        "mad.wide.u32 %%dst0, %0, 16, %2;\n\t"
        "add.u64 %%dst1, %%dst0, 16;\n\t"
        "@%%pst mov.u64 %%t, %%globaltimer;\n\t"
        "@%%pst mov.b64 {%%tlo, %%thi}, %%t;\n\t"
        "@%%pst st.global.cs.v4.u32 [%%dst0], {%%tlo, %%thi, %4, %6};\n\t"
        "@%%pst st.global.cs.v4.u32 [%%dst1], {%%tlo, %%thi, %5, %6};\n\t"
        "@%%pinc add.u32 %0, %0, 2;\n\t"
        "L_FI_TSKIP_%=:\n\t"
        "}"
        : "+r"(c.n)
        : "r"(active), "l"(c.p), "r"(c.cap), "r"(end_tag), "r"(beg_tag), "r"(aux));
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
    const FiBuf b = wp_active_buf();
    if (!recorder || b.rec == nullptr) return c;
    const uint32_t cta = fi_cta_id();
#if FI_GRAN == 1
    const uint32_t s = cta;  // SM 粒度：每 CTA 一条泳道；归到哪个物理 SM 由 OPEN 记录的 %smid 决定（解码器做）
    if (s >= b.n_slots) {
#else
    const uint32_t w = (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) >> 5;
    const uint32_t s = cta * b.warps_per_cta + w;  // warp 粒度：每 warp 一条泳道
    if (w >= b.warps_per_cta || s >= b.n_slots) {
#endif
        if (b.overflow != nullptr) atomicOr(b.overflow, 2u);
        return c;
    }
    c.p = b.rec + (size_t)s * b.cap;
    c.cap = b.cap;
    c.slot = s;
    const uint64_t g = fi_gtimer();
#if FI_TIMER == 1
    const uint64_t k = fi_clk();
#endif
    fi_store(c, g, FI_TAG_OPEN | (role & 0xffffu), fi_smid());
#if FI_TIMER == 1
    fi_store(c, k, FI_TAG_CLK, 0u);
#endif
    if (b.cur != nullptr) b.cur[s] = c.n;
    return c;
}
__device__ __forceinline__ FiCtx fi_open_subscope(bool recorder) {
    FiCtx c;
    c.p = nullptr;
    c.n = 0;
    c.cap = 0;
    c.slot = 0;
    const FiBuf b = wp_active_buf();
    if (!recorder || b.rec == nullptr || b.cur == nullptr) return c;
    const uint32_t cta = fi_cta_id();
#if FI_GRAN == 1
    const uint32_t s = cta;
    if (s >= b.n_slots) return c;
#else
    const uint32_t w = (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) >> 5;
    const uint32_t s = cta * b.warps_per_cta + w;
    if (w >= b.warps_per_cta || s >= b.n_slots) return c;
#endif
    c.p = b.rec + (size_t)s * b.cap;
    c.cap = b.cap;
    c.slot = s;
    c.n = b.cur[s];
    return c;
}
__device__ __forceinline__ void fi_close(FiCtx& c) {
    if (c.p == nullptr) return;
    const FiBuf b = wp_active_buf();
    if (c.n > c.cap && b.overflow != nullptr) atomicOr(b.overflow, 1u);
    const uint64_t g = fi_gtimer();
#if FI_TIMER == 1
    const uint64_t k = fi_clk();
#endif
    fi_store(c, g, FI_TAG_CLOSE, 0u);
#if FI_TIMER == 1
    fi_store(c, k, FI_TAG_CLK, 1u);
#endif
    if (b.cur != nullptr) b.cur[c.slot] = c.n;
}

// §3.2 (2a) Zero Call-Chain Signature Edits RAII Guard:
// Caches per-warp cursor `b.cur[slot]` in a Lane-0 local register on scope entry and writes
// back on scope exit (`~FiScope`), so neither `__global__` kernels NOR inner `__device__`
// subfunctions (`mma_mainloop()`, etc.) ever modify their parameter signatures!
struct FiScope {
    FiCtx ctx;
    bool is_root;
    __device__ __forceinline__ FiScope(bool recorder, uint32_t role, bool root)
        : ctx(root ? fi_open(recorder, role) : fi_open_subscope(recorder)), is_root(root) {}
    __device__ __forceinline__ void sync_out() {
        const FiBuf b = wp_active_buf();
        if (ctx.p != nullptr && b.cur != nullptr) b.cur[ctx.slot] = ctx.n;
    }
    __device__ __forceinline__ void sync_in() {
        const FiBuf b = wp_active_buf();
        if (ctx.p != nullptr && b.cur != nullptr) ctx.n = b.cur[ctx.slot];
    }
    __device__ __forceinline__ ~FiScope() {
        if (is_root) {
            fi_close(ctx);
        } else {
            const FiBuf b = wp_active_buf();
            if (ctx.p != nullptr && b.cur != nullptr) b.cur[ctx.slot] = ctx.n;
        }
    }
};

#define FI_GROUP_ON(g) (((FI_GROUPS) >> ((g) & 31)) & 1u)

// -----------------------------------------------------------------------------
// Shared-Memory Context (`-DWP_CTX_SMEM`) and
// Single-Register Bit-Packed Context (`-DWP_CTX_REG` / `-DWP_CTX_REG_NOINLINE`):
//   1) `WP_CTX_REG` collapses the 5-register `FiCtx` (`p`: 2 regs, `n`/`cap`/`slot`: 1 reg each)
//      into a SINGLE 32-bit register `uint32_t state` (0 Bytes SMEM!):
//        - bit  31    : disabled / overflow flag (`(int32_t)state < 0` skips in 1 op)
//        - bits 30..15: 16-bit lane slot (up to 65,535 concurrent warp/CTA lanes)
//        - bits 14..0 : 15-bit record cursor `n` (up to 32,767 records/lane)
//      Global base pointer `wp_fi_c.rec` and `wp_fi_c.cap` stay grid-uniform (loaded only
//      on the cold recorder branch, never occupying 3 persistent per-thread registers).
//   2) `WP_CTX_REG_NOINLINE` additionally outlines cold-path `st.global.cs.v4` helpers
//      with `__noinline__` (1 live reg, 0 B SMEM, zero hot-loop v4 interference).
//   3) `WP_CTX_SMEM` parks the 16B context in static `__shared__`
//      + `__noinline__` helpers (0 live regs across the hot loop, 16B SMEM/slot)
//      as the compiler-in-the-loop fallback when even 1 reg crosses a spill cliff.
// -----------------------------------------------------------------------------
#if defined(WP_CTX_REG_NOINLINE) && !defined(WP_CTX_REG)
#define WP_CTX_REG 1
#endif

#ifdef WP_CTX_SMEM
#ifndef WP_MAX_WARPS_PER_CTA
#define WP_MAX_WARPS_PER_CTA 32u
#endif

struct WpSmemCtx {
    FiRec* p;
    uint32_t n;
    uint32_t cap;
};

__device__ __noinline__ void wp_smem_open(WpSmemCtx* sc, uint32_t role) {
    sc->p = nullptr;
    sc->n = 0u;
    sc->cap = 0u;
    const FiBuf b = wp_active_buf();
    if (b.rec == nullptr) return;
    const uint32_t cta = fi_cta_id();
#if FI_GRAN == 1
    const uint32_t s = cta;
    if (s >= b.n_slots) {
#else
    const uint32_t w = (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) >> 5;
    const uint32_t s = cta * b.warps_per_cta + w;
    if (w >= b.warps_per_cta || s >= b.n_slots) {
#endif
        if (b.overflow != nullptr) atomicOr(b.overflow, 2u);
        return;
    }
    FiRec* base = b.rec + (size_t)s * b.cap;
    uint32_t n = 0u;
    const uint64_t g = fi_gtimer();
#if FI_TIMER == 1
    const uint64_t k = fi_clk();
#endif
    if (n < b.cap) {
        asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(base + n),
                     "r"((uint32_t)g), "r"((uint32_t)(g >> 32)),
                     "r"(FI_TAG_OPEN | (role & 0xffffu)), "r"(fi_smid()));
        ++n;
    }
#if FI_TIMER == 1
    if (n < b.cap) {
        asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(base + n),
                     "r"((uint32_t)k), "r"((uint32_t)(k >> 32)),
                     "r"(FI_TAG_CLK), "r"(0u));
        ++n;
    }
#endif
    sc->p = base;
    sc->n = n;
    sc->cap = b.cap;
    if (b.cur != nullptr) b.cur[s] = n;
}

__device__ __noinline__ void wp_smem_rec(WpSmemCtx* sc, uint32_t tag, uint32_t aux) {
    FiRec* base = sc->p;
    if (base == nullptr) return;
    uint32_t n = sc->n;
    const uint32_t cap = sc->cap;
    if (n >= cap) {
        const FiBuf b = wp_active_buf();
        if (n == cap && b.overflow != nullptr) atomicOr(b.overflow, 1u);
        sc->n = cap + 1u;
        return;
    }
    const uint64_t t = fi_now();
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(base + n),
                 "r"((uint32_t)t), "r"((uint32_t)(t >> 32)), "r"(tag), "r"(aux));
    sc->n = n + 1u;
}

__device__ __noinline__ void wp_smem_trans(WpSmemCtx* sc, uint32_t end_tag, uint32_t beg_tag, uint32_t aux) {
    FiRec* base = sc->p;
    if (base == nullptr) return;
    uint32_t n = sc->n;
    const uint32_t cap = sc->cap;
    if (n + 1u >= cap) {
        const FiBuf b = wp_active_buf();
        if (n <= cap && b.overflow != nullptr) atomicOr(b.overflow, 1u);
        sc->n = cap + 1u;
        return;
    }
    const uint64_t t = fi_now();
    const uint32_t tlo = (uint32_t)t;
    const uint32_t thi = (uint32_t)(t >> 32);
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(base + n),
                 "r"(tlo), "r"(thi), "r"(end_tag), "r"(aux));
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(base + n + 1u),
                 "r"(tlo), "r"(thi), "r"(beg_tag), "r"(aux));
    sc->n = n + 2u;
}

__device__ __noinline__ void wp_smem_close(WpSmemCtx* sc) {
    FiRec* base = sc->p;
    if (base == nullptr) return;
    uint32_t n = sc->n;
    const uint32_t cap = sc->cap;
    const FiBuf b = wp_active_buf();
    if (n > cap && b.overflow != nullptr) atomicOr(b.overflow, 1u);
    const uint64_t g = fi_gtimer();
#if FI_TIMER == 1
    const uint64_t k = fi_clk();
#endif
    if (n < cap) {
        asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(base + n),
                     "r"((uint32_t)g), "r"((uint32_t)(g >> 32)),
                     "r"(FI_TAG_CLOSE), "r"(0u));
        ++n;
    }
#if FI_TIMER == 1
    if (n < cap) {
        asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(base + n),
                     "r"((uint32_t)k), "r"((uint32_t)(k >> 32)),
                     "r"(FI_TAG_CLK), "r"(1u));
        ++n;
    }
#endif
    sc->n = n;
}

#if FI_GRAN == 1
#define WP_SMEM_DECL() __shared__ WpSmemCtx fi_sctx_
#define WP_SMEM_SLOT() (&fi_sctx_)
// Device functions receive the CTA's shared context by reference under the SAME name, so every
// macro (WP_SMEM_SLOT) resolves identically inside the kernel and inside WP_CTX_PARAM callees.
#define WP_SMEM_PARAM_DECL WpSmemCtx& fi_sctx_
#else
#define WP_SMEM_DECL() __shared__ WpSmemCtx fi_sctx_[WP_MAX_WARPS_PER_CTA]
#define WP_SMEM_SLOT() (&fi_sctx_[((threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) >> 5)])
// Callees receive the base of the per-warp shared context array under the SAME name.
#define WP_SMEM_PARAM_DECL WpSmemCtx* fi_sctx_
#endif

#if FI_LEVEL >= 1
#define FI_CTX_PARAM , WP_SMEM_PARAM_DECL
#define FI_CTX_ARG , fi_sctx_
#define FI_OPEN(role) WP_SMEM_DECL(); if (fi_recorder()) wp_smem_open(WP_SMEM_SLOT(), (uint32_t)(role))
#define FI_OPEN_BY(pred, role) WP_SMEM_DECL(); if (pred) wp_smem_open(WP_SMEM_SLOT(), (uint32_t)(role))
#define FI_CLOSE() do { if (fi_recorder()) wp_smem_close(WP_SMEM_SLOT()); } while (0)
#define FI_ROLE_SWITCH(new_role) do { if (fi_recorder()) wp_smem_rec(WP_SMEM_SLOT(), FI_TAG_ROLE_SWITCH | FI_TAG_MARK | ((uint32_t)(new_role) & FI_TAG_IDMASK), (uint32_t)(new_role)); } while (0)
#define FI_KERNEL_SCOPE(role) FI_OPEN(role)
#define FI_KERNEL_SCOPE_BY(pred, role) FI_OPEN_BY(pred, role)
#define FI_DEVICE_SCOPE() do { } while (0)
#define FI_DEVICE_SCOPE_BY(pred) do { (void)(pred); } while (0)
#define FI_SYNC_OUT() do { } while (0)
#define FI_SYNC_IN() do { } while (0)
#else
#define FI_CTX_PARAM
#define FI_CTX_ARG
#define FI_OPEN(role) do { } while (0)
#define FI_OPEN_BY(pred, role) do { } while (0)
#define FI_CLOSE() do { } while (0)
#define FI_ROLE_SWITCH(new_role) do { (void)(new_role); } while (0)
#define FI_KERNEL_SCOPE(role) do { } while (0)
#define FI_KERNEL_SCOPE_BY(pred, role) do { (void)(pred); } while (0)
#define FI_DEVICE_SCOPE() do { } while (0)
#define FI_DEVICE_SCOPE_BY(pred) do { (void)(pred); } while (0)
#define FI_SYNC_OUT() do { } while (0)
#define FI_SYNC_IN() do { } while (0)
#endif

#if FI_LEVEL >= 2
#define FI_BG(g, id, aux) do { if (FI_GROUP_ON(g) && fi_recorder()) wp_smem_rec(WP_SMEM_SLOT(), (uint32_t)(id) & FI_TAG_IDMASK, (uint32_t)(aux)); } while (0)
#define FI_EG(g, id, aux) do { if (FI_GROUP_ON(g) && fi_recorder()) wp_smem_rec(WP_SMEM_SLOT(), ((uint32_t)(id) & FI_TAG_IDMASK) | FI_TAG_END, (uint32_t)(aux)); } while (0)
#define FI_MG(g, id, aux) do { if (FI_GROUP_ON(g) && fi_recorder()) wp_smem_rec(WP_SMEM_SLOT(), ((uint32_t)(id) & FI_TAG_IDMASK) | FI_TAG_MARK, (uint32_t)(aux)); } while (0)
#define FI_BG_PRED(pred, g, id, aux) do { if (FI_GROUP_ON(g) && fi_recorder() && (bool)(pred)) wp_smem_rec(WP_SMEM_SLOT(), (uint32_t)(id) & FI_TAG_IDMASK, (uint32_t)(aux)); } while (0)
#define FI_EG_PRED(pred, g, id, aux) do { if (FI_GROUP_ON(g) && fi_recorder() && (bool)(pred)) wp_smem_rec(WP_SMEM_SLOT(), ((uint32_t)(id) & FI_TAG_IDMASK) | FI_TAG_END, (uint32_t)(aux)); } while (0)
#define FI_MG_PRED(pred, g, id, aux) do { if (FI_GROUP_ON(g) && fi_recorder() && (bool)(pred)) wp_smem_rec(WP_SMEM_SLOT(), ((uint32_t)(id) & FI_TAG_IDMASK) | FI_TAG_MARK, (uint32_t)(aux)); } while (0)
#define FI_TRANS_PRED(pred, g_end, end_id, g_beg, beg_id, aux) do { if (FI_GROUP_ON(g_end) && FI_GROUP_ON(g_beg) && fi_recorder() && (bool)(pred)) wp_smem_trans(WP_SMEM_SLOT(), ((uint32_t)(end_id) & FI_TAG_IDMASK) | FI_TAG_END, (uint32_t)(beg_id) & FI_TAG_IDMASK, (uint32_t)(aux)); else { FI_EG_PRED(pred, g_end, end_id, aux); FI_BG_PRED(pred, g_beg, beg_id, aux); } } while (0)
#else
#define FI_BG(g, id, aux) do { } while (0)
#define FI_EG(g, id, aux) do { } while (0)
#define FI_MG(g, id, aux) do { } while (0)
#define FI_BG_PRED(pred, g, id, aux) do { (void)(pred); } while (0)
#define FI_EG_PRED(pred, g, id, aux) do { (void)(pred); } while (0)
#define FI_MG_PRED(pred, g, id, aux) do { (void)(pred); } while (0)
#define FI_TRANS_PRED(pred, g_end, end_id, g_beg, beg_id, aux) do { (void)(pred); } while (0)
#endif

#elif defined(WP_CTX_REG)  // single-register bit-packed context (0 B SMEM)

#define WP_REG_DISABLED 0x80000000u
#define WP_REG_SLOT_SHIFT 15u
#define WP_REG_SLOT_MASK 0xffffu
#define WP_REG_CUR_MASK 0x7fffu

#if defined(WP_CTX_REG_NOINLINE)
#define WP_REG_COLD_ATTR __device__ __noinline__
#else
#define WP_REG_COLD_ATTR __device__ __forceinline__
#endif

struct WpCtxReg {
    uint32_t state;  // bit31=disabled/ovf, bits30..15=slot(0..65535), bits14..0=cursor(0..32767)
};

WP_REG_COLD_ATTR uint32_t wp_reg_open_cold(uint32_t role) {
    const FiBuf b = wp_active_buf();
    if (b.rec == nullptr || b.cap == 0u) return WP_REG_DISABLED;
    const uint32_t cta = fi_cta_id();
#if FI_GRAN == 1
    const uint32_t s = cta;
    if (s >= b.n_slots || s > WP_REG_SLOT_MASK) {
#else
    const uint32_t w = (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) >> 5;
    const uint32_t s = cta * b.warps_per_cta + w;
    if (w >= b.warps_per_cta || s >= b.n_slots || s > WP_REG_SLOT_MASK) {
#endif
        if (b.overflow != nullptr) atomicOr(b.overflow, 2u);
        return WP_REG_DISABLED;
    }
    const uint32_t eff_cap = (b.cap > WP_REG_CUR_MASK) ? WP_REG_CUR_MASK : b.cap;
    FiRec* base = b.rec + (size_t)s * b.cap;
    uint32_t n = 0u;
    const uint64_t g = fi_gtimer();
#if FI_TIMER == 1
    const uint64_t k = fi_clk();
#endif
    if (n < eff_cap) {
        asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(base + n),
                     "r"((uint32_t)g), "r"((uint32_t)(g >> 32)),
                     "r"(FI_TAG_OPEN | (role & 0xffffu)), "r"(fi_smid()));
        ++n;
    }
#if FI_TIMER == 1
    if (n < eff_cap) {
        asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(base + n),
                     "r"((uint32_t)k), "r"((uint32_t)(k >> 32)),
                     "r"(FI_TAG_CLK), "r"(0u));
        ++n;
    }
#endif
    if (b.cur != nullptr) b.cur[s] = n;
    const uint32_t packed = (s << WP_REG_SLOT_SHIFT) | (n & WP_REG_CUR_MASK);
    return (n >= eff_cap) ? (packed | WP_REG_DISABLED) : packed;
}

WP_REG_COLD_ATTR uint32_t wp_reg_open_subscope_cold() {
    const FiBuf b = wp_active_buf();
    if (b.rec == nullptr || b.cur == nullptr || b.cap == 0u) return WP_REG_DISABLED;
    const uint32_t cta = fi_cta_id();
#if FI_GRAN == 1
    const uint32_t s = cta;
    if (s >= b.n_slots || s > WP_REG_SLOT_MASK) return WP_REG_DISABLED;
#else
    const uint32_t w = (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) >> 5;
    const uint32_t s = cta * b.warps_per_cta + w;
    if (w >= b.warps_per_cta || s >= b.n_slots || s > WP_REG_SLOT_MASK) return WP_REG_DISABLED;
#endif
    const uint32_t eff_cap = (b.cap > WP_REG_CUR_MASK) ? WP_REG_CUR_MASK : b.cap;
    const uint32_t n = b.cur[s];
    if (n >= eff_cap) return (s << WP_REG_SLOT_SHIFT) | WP_REG_DISABLED;
    return (s << WP_REG_SLOT_SHIFT) | (n & WP_REG_CUR_MASK);
}

WP_REG_COLD_ATTR uint32_t wp_reg_emit_cold(uint32_t state, uint32_t tag, uint32_t aux) {
    const uint32_t slot = (state >> WP_REG_SLOT_SHIFT) & WP_REG_SLOT_MASK;
    const uint32_t n = state & WP_REG_CUR_MASK;
    const FiBuf b = wp_active_buf();
    const uint32_t eff_cap = (b.cap > WP_REG_CUR_MASK) ? WP_REG_CUR_MASK : b.cap;
    if (n >= eff_cap) {
        if (b.overflow != nullptr) atomicOr(b.overflow, 1u);
        return state | WP_REG_DISABLED;
    }
    const uint64_t t = fi_now();
    FiRec* dst = b.rec + (size_t)slot * b.cap + n;
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(dst),
                 "r"((uint32_t)t), "r"((uint32_t)(t >> 32)), "r"(tag), "r"(aux));
    return state + 1u;  // n < eff_cap <= 0x7FFF guarantees zero carry into bit 15
}

WP_REG_COLD_ATTR uint32_t wp_reg_trans_cold(uint32_t state, uint32_t end_tag, uint32_t beg_tag, uint32_t aux) {
    const uint32_t slot = (state >> WP_REG_SLOT_SHIFT) & WP_REG_SLOT_MASK;
    const uint32_t n = state & WP_REG_CUR_MASK;
    const FiBuf b = wp_active_buf();
    const uint32_t eff_cap = (b.cap > WP_REG_CUR_MASK) ? WP_REG_CUR_MASK : b.cap;
    if (n + 1u >= eff_cap) {
        if (b.overflow != nullptr) atomicOr(b.overflow, 1u);
        return state | WP_REG_DISABLED;
    }
    const uint64_t t = fi_now();
    const uint32_t tlo = (uint32_t)t;
    const uint32_t thi = (uint32_t)(t >> 32);
    FiRec* dst = b.rec + (size_t)slot * b.cap + n;
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(dst),
                 "r"(tlo), "r"(thi), "r"(end_tag), "r"(aux));
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(dst + 1u),
                 "r"(tlo), "r"(thi), "r"(beg_tag), "r"(aux));
    return state + 2u;  // n + 1 < eff_cap <= 0x7FFF guarantees zero carry into bit 15
}

WP_REG_COLD_ATTR uint32_t wp_reg_close_cold(uint32_t state) {
    const uint32_t slot = (state >> WP_REG_SLOT_SHIFT) & WP_REG_SLOT_MASK;
    uint32_t n = state & WP_REG_CUR_MASK;
    const FiBuf b = wp_active_buf();
    const uint32_t eff_cap = (b.cap > WP_REG_CUR_MASK) ? WP_REG_CUR_MASK : b.cap;
    if ((state & WP_REG_DISABLED) != 0u) {
        if (n != 0u && b.overflow != nullptr) atomicOr(b.overflow, 1u);
        return state;
    }
    const uint64_t g = fi_gtimer();
#if FI_TIMER == 1
    const uint64_t k = fi_clk();
#endif
    FiRec* base = b.rec + (size_t)slot * b.cap;
    if (n < eff_cap) {
        asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(base + n),
                     "r"((uint32_t)g), "r"((uint32_t)(g >> 32)),
                     "r"(FI_TAG_CLOSE), "r"(0u));
        ++n;
    } else if (b.overflow != nullptr) {
        atomicOr(b.overflow, 1u);
    }
#if FI_TIMER == 1
    if (n < eff_cap) {
        asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(base + n),
                     "r"((uint32_t)k), "r"((uint32_t)(k >> 32)),
                     "r"(FI_TAG_CLK), "r"(1u));
        ++n;
    } else if (b.overflow != nullptr) {
        atomicOr(b.overflow, 1u);
    }
#endif
    if (b.cur != nullptr) b.cur[slot] = n;
    return (slot << WP_REG_SLOT_SHIFT) | WP_REG_DISABLED;
}

__device__ __forceinline__ WpCtxReg wp_reg_open(bool recorder, uint32_t role) {
    WpCtxReg c;
    c.state = recorder ? wp_reg_open_cold(role) : WP_REG_DISABLED;
    return c;
}
#if defined(WP_CTX_REG_NOINLINE)
__device__ __forceinline__ void wp_reg_rec(WpCtxReg& c, uint32_t tag, uint32_t aux) {
    if ((int32_t)c.state >= 0) c.state = wp_reg_emit_cold(c.state, tag, aux);
}
__device__ __forceinline__ void wp_reg_rec_pred(WpCtxReg& c, bool pred, uint32_t tag, uint32_t aux) {
    if ((int32_t)c.state >= 0 && pred) c.state = wp_reg_emit_cold(c.state, tag, aux);
}
__device__ __forceinline__ void wp_reg_trans_pred(WpCtxReg& c, bool pred, uint32_t end_tag, uint32_t beg_tag, uint32_t aux) {
    if ((int32_t)c.state >= 0 && pred) c.state = wp_reg_trans_cold(c.state, end_tag, beg_tag, aux);
}
#else
// Zero-BB LLVM-opaque single-register inline PTX path backed by direct __constant__ symbol (`wp_fi_c`):
//   - Literally ONE live 32-bit register (`c.state`) across the entire hot loop (0 B SMEM, 0 pointer reg!).
//   - Direct `ld.const` symbol loads fold into 1-cycle uniform `LDC` instructions inside the cold branch.
//   - Minimal scratch-register footprint (4x u32 + 2x u64 reused) prevents spill cliffs at 128 regs.
static_assert(offsetof(FiBuf, rec) == 0 && offsetof(FiBuf, overflow) == 8 && offsetof(FiBuf, cap) == 24,
              "FiBuf layout mismatch for WpCtxReg inline PTX");

#if defined(__CUDACC_RDC__)
__device__ __forceinline__ void wp_reg_rec_pred(WpCtxReg& c, bool pred, uint32_t tag, uint32_t aux) {
    if ((int32_t)c.state >= 0 && pred) c.state = wp_reg_emit_cold(c.state, tag, aux);
}
__device__ __forceinline__ void wp_reg_rec(WpCtxReg& c, uint32_t tag, uint32_t aux) {
    if ((int32_t)c.state >= 0) c.state = wp_reg_emit_cold(c.state, tag, aux);
}
#else
__device__ __forceinline__ void wp_reg_rec(WpCtxReg& c, uint32_t tag, uint32_t aux) {
#if FI_TIMER == 1
    asm volatile(
        "{\n\t"
        ".reg .pred %%p, %%pst;\n\t"
        ".reg .u32 %%u0, %%u1, %%u2;\n\t"
        ".reg .u64 %%p0, %%p1;\n\t"
        "setp.lt.s32 %%p, %0, 0;\n\t"
        "@%%p bra L_WPREG_SKIP_%=;\n\t"
        "ld.const.u32 %%u1, [wp_fi_c + 24];\n\t"
        "and.b32 %%u0, %0, 32767;\n\t"
        "min.u32 %%u2, %%u1, 32767;\n\t"
        "setp.lt.u32 %%pst, %%u0, %%u2;\n\t"
        "@!%%pst bra L_WPREG_OVF_%=;\n\t"
        "shr.u32 %%u2, %0, 15;\n\t"
        "mad.lo.u32 %%u0, %%u2, %%u1, %%u0;\n\t"
        "ld.const.u64 %%p0, [wp_fi_c + 0];\n\t"
        "mad.wide.u32 %%p0, %%u0, 16, %%p0;\n\t"
        "mov.u64 %%p1, %%clock64;\n\t"
        "mov.b64 {%%u0, %%u1}, %%p1;\n\t"
        "st.global.cs.v4.u32 [%%p0], {%%u0, %%u1, %1, %2};\n\t"
        "add.u32 %0, %0, 1;\n\t"
        "bra L_WPREG_SKIP_%=;\n\t"
        "L_WPREG_OVF_%=:\n\t"
        "or.b32 %0, %0, 2147483648;\n\t"
        "L_WPREG_SKIP_%=:\n\t"
        "}"
        : "+r"(c.state)
        : "r"(tag), "r"(aux));
#else
    asm volatile(
        "{\n\t"
        ".reg .pred %%p, %%pst;\n\t"
        ".reg .u32 %%u0, %%u1, %%u2;\n\t"
        ".reg .u64 %%p0, %%p1;\n\t"
        "setp.lt.s32 %%p, %0, 0;\n\t"
        "@%%p bra L_WPREG_SKIP_%=;\n\t"
        "ld.const.u32 %%u1, [wp_fi_c + 24];\n\t"
        "and.b32 %%u0, %0, 32767;\n\t"
        "min.u32 %%u2, %%u1, 32767;\n\t"
        "setp.lt.u32 %%pst, %%u0, %%u2;\n\t"
        "@!%%pst bra L_WPREG_OVF_%=;\n\t"
        "shr.u32 %%u2, %0, 15;\n\t"
        "mad.lo.u32 %%u0, %%u2, %%u1, %%u0;\n\t"
        "ld.const.u64 %%p0, [wp_fi_c + 0];\n\t"
        "mad.wide.u32 %%p0, %%u0, 16, %%p0;\n\t"
        "mov.u64 %%p1, %%globaltimer;\n\t"
        "mov.b64 {%%u0, %%u1}, %%p1;\n\t"
        "st.global.cs.v4.u32 [%%p0], {%%u0, %%u1, %1, %2};\n\t"
        "add.u32 %0, %0, 1;\n\t"
        "bra L_WPREG_SKIP_%=;\n\t"
        "L_WPREG_OVF_%=:\n\t"
        "or.b32 %0, %0, 2147483648;\n\t"
        "L_WPREG_SKIP_%=:\n\t"
        "}"
        : "+r"(c.state)
        : "r"(tag), "r"(aux));
#endif
}

__device__ __forceinline__ void wp_reg_rec_pred(WpCtxReg& c, bool pred, uint32_t tag, uint32_t aux) {
    const uint32_t active_pred = pred ? 1u : 0u;
#if FI_TIMER == 1
    asm volatile(
        "{\n\t"
        ".reg .pred %%p, %%pst;\n\t"
        ".reg .u32 %%u0, %%u1, %%u2;\n\t"
        ".reg .u64 %%p0, %%p1;\n\t"
        "setp.eq.u32 %%p, %1, 0;\n\t"
        "@%%p bra L_WPREGP_SKIP_%=;\n\t"
        "setp.lt.s32 %%p, %0, 0;\n\t"
        "@%%p bra L_WPREGP_SKIP_%=;\n\t"
        "ld.const.u32 %%u1, [wp_fi_c + 24];\n\t"
        "and.b32 %%u0, %0, 32767;\n\t"
        "min.u32 %%u2, %%u1, 32767;\n\t"
        "setp.lt.u32 %%pst, %%u0, %%u2;\n\t"
        "@!%%pst bra L_WPREGP_OVF_%=;\n\t"
        "shr.u32 %%u2, %0, 15;\n\t"
        "mad.lo.u32 %%u0, %%u2, %%u1, %%u0;\n\t"
        "ld.const.u64 %%p0, [wp_fi_c + 0];\n\t"
        "mad.wide.u32 %%p0, %%u0, 16, %%p0;\n\t"
        "mov.u64 %%p1, %%clock64;\n\t"
        "mov.b64 {%%u0, %%u1}, %%p1;\n\t"
        "st.global.cs.v4.u32 [%%p0], {%%u0, %%u1, %2, %3};\n\t"
        "add.u32 %0, %0, 1;\n\t"
        "bra L_WPREGP_SKIP_%=;\n\t"
        "L_WPREGP_OVF_%=:\n\t"
        "or.b32 %0, %0, 2147483648;\n\t"
        "L_WPREGP_SKIP_%=:\n\t"
        "}"
        : "+r"(c.state)
        : "r"(active_pred), "r"(tag), "r"(aux));
#else
    asm volatile(
        "{\n\t"
        ".reg .pred %%p, %%pst;\n\t"
        ".reg .u32 %%u0, %%u1, %%u2;\n\t"
        ".reg .u64 %%p0, %%p1;\n\t"
        "setp.eq.u32 %%p, %1, 0;\n\t"
        "@%%p bra L_WPREGP_SKIP_%=;\n\t"
        "setp.lt.s32 %%p, %0, 0;\n\t"
        "@%%p bra L_WPREGP_SKIP_%=;\n\t"
        "ld.const.u32 %%u1, [wp_fi_c + 24];\n\t"
        "and.b32 %%u0, %0, 32767;\n\t"
        "min.u32 %%u2, %%u1, 32767;\n\t"
        "setp.lt.u32 %%pst, %%u0, %%u2;\n\t"
        "@!%%pst bra L_WPREGP_OVF_%=;\n\t"
        "shr.u32 %%u2, %0, 15;\n\t"
        "mad.lo.u32 %%u0, %%u2, %%u1, %%u0;\n\t"
        "ld.const.u64 %%p0, [wp_fi_c + 0];\n\t"
        "mad.wide.u32 %%p0, %%u0, 16, %%p0;\n\t"
        "mov.u64 %%p1, %%globaltimer;\n\t"
        "mov.b64 {%%u0, %%u1}, %%p1;\n\t"
        "st.global.cs.v4.u32 [%%p0], {%%u0, %%u1, %2, %3};\n\t"
        "add.u32 %0, %0, 1;\n\t"
        "bra L_WPREGP_SKIP_%=;\n\t"
        "L_WPREGP_OVF_%=:\n\t"
        "or.b32 %0, %0, 2147483648;\n\t"
        "L_WPREGP_SKIP_%=:\n\t"
        "}"
        : "+r"(c.state)
        : "r"(active_pred), "r"(tag), "r"(aux));
#endif
}
#endif
__device__ __forceinline__ void wp_reg_trans_pred(WpCtxReg& c, bool pred, uint32_t end_tag, uint32_t beg_tag, uint32_t aux) {
    if ((int32_t)c.state >= 0 && pred) c.state = wp_reg_trans_cold(c.state, end_tag, beg_tag, aux);
}
#endif
__device__ __forceinline__ void wp_reg_close(WpCtxReg& c) {
    if ((int32_t)c.state >= 0 || (c.state & WP_REG_CUR_MASK) != 0u) c.state = wp_reg_close_cold(c.state);
}

struct WpScopeReg {
    WpCtxReg ctx;
    bool is_root;
    __device__ __forceinline__ WpScopeReg(bool recorder, uint32_t role, bool root)
        : ctx{recorder ? (root ? wp_reg_open_cold(role) : wp_reg_open_subscope_cold()) : WP_REG_DISABLED},
          is_root(root) {}
    __device__ __forceinline__ void sync_out() {
        const FiBuf b = wp_active_buf();
        if ((int32_t)ctx.state >= 0 && b.cur != nullptr) {
            const uint32_t slot = (ctx.state >> WP_REG_SLOT_SHIFT) & WP_REG_SLOT_MASK;
            if (slot < b.n_slots) b.cur[slot] = ctx.state & WP_REG_CUR_MASK;
        }
    }
    __device__ __forceinline__ void sync_in() {
        const FiBuf b = wp_active_buf();
        if ((int32_t)ctx.state >= 0 && b.cur != nullptr) {
            const uint32_t slot = (ctx.state >> WP_REG_SLOT_SHIFT) & WP_REG_SLOT_MASK;
            if (slot >= b.n_slots) return;
            const uint32_t eff_cap = (b.cap > WP_REG_CUR_MASK) ? WP_REG_CUR_MASK : b.cap;
            const uint32_t n = b.cur[slot];
            ctx.state = (n >= eff_cap) ? ((slot << WP_REG_SLOT_SHIFT) | WP_REG_DISABLED)
                                       : ((slot << WP_REG_SLOT_SHIFT) | (n & WP_REG_CUR_MASK));
        }
    }
    __device__ __forceinline__ ~WpScopeReg() {
        if (is_root) {
            wp_reg_close(ctx);
        } else {
            sync_out();
        }
    }
};

#if FI_LEVEL >= 1
#define FI_CTX_PARAM , WpCtxReg& fi_ctx_
#define FI_CTX_ARG , fi_ctx_
#define FI_OPEN(role) WpCtxReg fi_ctx_ = wp_reg_open(fi_recorder(), (uint32_t)(role))
#define FI_OPEN_BY(pred, role) WpCtxReg fi_ctx_ = wp_reg_open((pred), (uint32_t)(role))
#define FI_CLOSE() wp_reg_close(fi_ctx_)
#define FI_ROLE_SWITCH(new_role) do { wp_reg_rec(fi_ctx_, FI_TAG_ROLE_SWITCH | FI_TAG_MARK | ((uint32_t)(new_role) & FI_TAG_IDMASK), (uint32_t)(new_role)); } while (0)
#define FI_KERNEL_SCOPE(role) WpScopeReg fi_scope_(fi_recorder(), (uint32_t)(role), true); WpCtxReg& fi_ctx_ = fi_scope_.ctx
#define FI_KERNEL_SCOPE_BY(pred, role) WpScopeReg fi_scope_((pred), (uint32_t)(role), true); WpCtxReg& fi_ctx_ = fi_scope_.ctx
#define FI_DEVICE_SCOPE() WpScopeReg fi_scope_(fi_recorder(), 0u, false); WpCtxReg& fi_ctx_ = fi_scope_.ctx
#define FI_DEVICE_SCOPE_BY(pred) WpScopeReg fi_scope_((pred), 0u, false); WpCtxReg& fi_ctx_ = fi_scope_.ctx
#define FI_SYNC_OUT() fi_scope_.sync_out()
#define FI_SYNC_IN() fi_scope_.sync_in()
#else
#define FI_CTX_PARAM
#define FI_CTX_ARG
#define FI_OPEN(role) do { } while (0)
#define FI_OPEN_BY(pred, role) do { } while (0)
#define FI_CLOSE() do { } while (0)
#define FI_ROLE_SWITCH(new_role) do { (void)(new_role); } while (0)
#define FI_KERNEL_SCOPE(role) do { } while (0)
#define FI_KERNEL_SCOPE_BY(pred, role) do { (void)(pred); } while (0)
#define FI_DEVICE_SCOPE() do { } while (0)
#define FI_DEVICE_SCOPE_BY(pred) do { (void)(pred); } while (0)
#define FI_SYNC_OUT() do { } while (0)
#define FI_SYNC_IN() do { } while (0)
#endif

#if FI_LEVEL >= 2
#define FI_BG(g, id, aux) do { if (FI_GROUP_ON(g)) wp_reg_rec(fi_ctx_, (uint32_t)(id) & FI_TAG_IDMASK, (uint32_t)(aux)); } while (0)
#define FI_EG(g, id, aux) do { if (FI_GROUP_ON(g)) wp_reg_rec(fi_ctx_, ((uint32_t)(id) & FI_TAG_IDMASK) | FI_TAG_END, (uint32_t)(aux)); } while (0)
#define FI_MG(g, id, aux) do { if (FI_GROUP_ON(g)) wp_reg_rec(fi_ctx_, ((uint32_t)(id) & FI_TAG_IDMASK) | FI_TAG_MARK, (uint32_t)(aux)); } while (0)
#define FI_BG_PRED(pred, g, id, aux) do { if (FI_GROUP_ON(g)) wp_reg_rec_pred(fi_ctx_, (bool)(pred), (uint32_t)(id) & FI_TAG_IDMASK, (uint32_t)(aux)); } while (0)
#define FI_EG_PRED(pred, g, id, aux) do { if (FI_GROUP_ON(g)) wp_reg_rec_pred(fi_ctx_, (bool)(pred), ((uint32_t)(id) & FI_TAG_IDMASK) | FI_TAG_END, (uint32_t)(aux)); } while (0)
#define FI_MG_PRED(pred, g, id, aux) do { if (FI_GROUP_ON(g)) wp_reg_rec_pred(fi_ctx_, (bool)(pred), ((uint32_t)(id) & FI_TAG_IDMASK) | FI_TAG_MARK, (uint32_t)(aux)); } while (0)
#define FI_TRANS_PRED(pred, g_end, end_id, g_beg, beg_id, aux) do { if (FI_GROUP_ON(g_end) && FI_GROUP_ON(g_beg)) wp_reg_trans_pred(fi_ctx_, (bool)(pred), ((uint32_t)(end_id) & FI_TAG_IDMASK) | FI_TAG_END, (uint32_t)(beg_id) & FI_TAG_IDMASK, (uint32_t)(aux)); else { FI_EG_PRED(pred, g_end, end_id, aux); FI_BG_PRED(pred, g_beg, beg_id, aux); } } while (0)
#else
#define FI_BG(g, id, aux) do { } while (0)
#define FI_EG(g, id, aux) do { } while (0)
#define FI_MG(g, id, aux) do { } while (0)
#define FI_BG_PRED(pred, g, id, aux) do { (void)(pred); } while (0)
#define FI_EG_PRED(pred, g, id, aux) do { (void)(pred); } while (0)
#define FI_MG_PRED(pred, g, id, aux) do { (void)(pred); } while (0)
#define FI_TRANS_PRED(pred, g_end, end_id, g_beg, beg_id, aux) do { (void)(pred); } while (0)
#endif

#else  // !WP_CTX_SMEM && !WP_CTX_REG (legacy 5-register FiCtx)

#if FI_LEVEL >= 1
#define FI_CTX_PARAM , FiCtx& fi_ctx_
#define FI_CTX_ARG , fi_ctx_
#define FI_OPEN(role) FiCtx fi_ctx_ = fi_open(fi_recorder(), (uint32_t)(role))
#define FI_OPEN_BY(pred, role) FiCtx fi_ctx_ = fi_open((pred), (uint32_t)(role))
#define FI_CLOSE() fi_close(fi_ctx_)
#define FI_ROLE_SWITCH(new_role) do { fi_record_now(fi_ctx_, FI_TAG_ROLE_SWITCH | FI_TAG_MARK | ((uint32_t)(new_role) & FI_TAG_IDMASK), (uint32_t)(new_role)); } while (0)
#define FI_KERNEL_SCOPE(role) FiScope fi_scope_(fi_recorder(), (uint32_t)(role), true); FiCtx& fi_ctx_ = fi_scope_.ctx
#define FI_KERNEL_SCOPE_BY(pred, role) FiScope fi_scope_((pred), (uint32_t)(role), true); FiCtx& fi_ctx_ = fi_scope_.ctx
#define FI_DEVICE_SCOPE() FiScope fi_scope_(fi_recorder(), 0u, false); FiCtx& fi_ctx_ = fi_scope_.ctx
#define FI_DEVICE_SCOPE_BY(pred) FiScope fi_scope_((pred), 0u, false); FiCtx& fi_ctx_ = fi_scope_.ctx
#define FI_SYNC_OUT() fi_scope_.sync_out()
#define FI_SYNC_IN() fi_scope_.sync_in()
#else
#define FI_CTX_PARAM
#define FI_CTX_ARG
#define FI_OPEN(role) do { } while (0)
#define FI_OPEN_BY(pred, role) do { } while (0)
#define FI_CLOSE() do { } while (0)
#define FI_ROLE_SWITCH(new_role) do { (void)(new_role); } while (0)
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

#endif  // WP_CTX_SMEM
#define FI_B(id, aux) FI_BG(0, id, aux)
#define FI_E(id, aux) FI_EG(0, id, aux)
#define FI_M(id, aux) FI_MG(0, id, aux)
#define FI_B_PRED(pred, id, aux) FI_BG_PRED(pred, 0, id, aux)
#define FI_E_PRED(pred, id, aux) FI_EG_PRED(pred, 0, id, aux)
#define FI_M_PRED(pred, id, aux) FI_MG_PRED(pred, 0, id, aux)

// Canonical WeaveProfiler (`WP_*` / `wp_*`) aliases (§3.2)
using WpRec = FiRec;
using WpBuf = FiBuf;
#if defined(WP_CTX_REG) && !defined(WP_CTX_SMEM)
using WpCtx = WpCtxReg;
using WpScope = WpScopeReg;
#else
using WpCtx = FiCtx;
using WpScope = FiScope;
#endif
#define wp_g fi_g
#define WP_OPEN(role) FI_OPEN(role)
#define WP_OPEN_BY(pred, role) FI_OPEN_BY(pred, role)
#define WP_CLOSE() FI_CLOSE()
#define WP_ROLE_SWITCH(new_role) FI_ROLE_SWITCH(new_role)
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

// ---------------- host 侧 ----------------
// n_ctas：grid 的 CTA 总数；warps_per_cta：每 CTA 线程数/32 上取整；cap：每泳道最多条数（OPEN/CLOSE 及锚点也占条数）。
// FI_GRAN=0 时一条泳道 = 一个 warp，cap 就是"每 warp 最多条数"；FI_GRAN=1 时一条泳道 = 一个 CTA（warps_per_cta 参数被忽略，
// 缓冲按 n_ctas 条泳道分配），cap 要给成"每 CTA 最多条数"（只有 0 号线程写，通常比 warp 粒度小一个数量级）。
static inline cudaError_t fi_alloc(FiBuf* b, uint32_t n_ctas, uint32_t warps_per_cta, uint32_t cap) {
    memset(b, 0, sizeof(*b));
    b->cap = cap;
#if FI_GRAN == 1
    // SM 粒度：每 CTA 一条泳道，cap 的含义变成"每 CTA 最多条数"；warps_per_cta 一律写 1，
    // 于是 host 调用处、转储头字段、解码器的 slot→(CTA,warp) 反算在两种粒度下形式完全一致。
    (void)warps_per_cta;
    b->warps_per_cta = 1u;
    b->n_slots = n_ctas;
#else
    b->warps_per_cta = warps_per_cta;
    b->n_slots = n_ctas * warps_per_cta;
#endif
    cudaError_t e = cudaMalloc(&b->rec, (size_t)b->n_slots * cap * sizeof(FiRec));
    if (e == cudaSuccess) e = cudaMalloc(&b->overflow, sizeof(uint32_t));
    if (e == cudaSuccess) e = cudaMalloc(&b->cur, (size_t)b->n_slots * sizeof(uint32_t));
    if (e == cudaSuccess) e = cudaMemset(b->rec, 0, (size_t)b->n_slots * cap * sizeof(FiRec));
    if (e == cudaSuccess) e = cudaMemset(b->overflow, 0, sizeof(uint32_t));
    if (e == cudaSuccess) e = cudaMemset(b->cur, 0, (size_t)b->n_slots * sizeof(uint32_t));
    return e;
}
static inline cudaError_t fi_attach(const FiBuf* b) {
    FiBuf z;
    memset(&z, 0, sizeof(z));
    const FiBuf* src = b ? b : &z;
    cudaError_t e = cudaMemcpyToSymbol(fi_g, src, sizeof(FiBuf));
    if (e == cudaSuccess) e = cudaMemcpyToSymbol(wp_fi_c, src, sizeof(FiBuf));
    return e;
}
// 要采的那次 launch 前调用（异步，排在同一 stream 上）。
static inline cudaError_t fi_reset(const FiBuf* b, cudaStream_t st = 0) {
    cudaError_t e = cudaMemsetAsync(b->rec, 0, (size_t)b->n_slots * b->cap * sizeof(FiRec), st);
    if (e == cudaSuccess) e = cudaMemsetAsync(b->overflow, 0, sizeof(uint32_t), st);
    if (e == cudaSuccess && b->cur != nullptr) e = cudaMemsetAsync(b->cur, 0, (size_t)b->n_slots * sizeof(uint32_t), st);
    return e;
}
// 转储格式固定 FILITE02（16 字节记录），粒度信息只能加在 meta JSON 里（统一规格允许的加字段位置）：
// fi_dump 自动把 "gran"/"slot" 并进调用方给的 meta 对象，解码器据此决定按 warp 还是按物理 SM 归组。
// 老转储没有这些字段，解码器按 gran=0（warp 粒度）处理，行为不变。
static inline std::string fi_meta_augment(const char* meta_json) {
    std::string m = (meta_json && *meta_json) ? std::string(meta_json) : std::string("{}");
    char extra[96];
    snprintf(extra, sizeof extra, "\"gran\":%d,\"slot\":\"%s\"", FI_GRAN, FI_GRAN == 1 ? "cta" : "warp");
    const size_t p = m.find_last_of('}');
    if (p == std::string::npos) return m;  // 调用方给的不是 JSON 对象：原样写，解码器按 gran=0 处理
    std::string head = m.substr(0, p);
    while (!head.empty() && (head.back() == ' ' || head.back() == '\n' || head.back() == '\t' || head.back() == '\r')) head.pop_back();
    const bool empty_obj = !head.empty() && head.back() == '{';
    return head + (empty_obj ? "" : ",") + extra + m.substr(p);
}

// 转储：magic "FILITE02" | u32 n_slots, cap, warps_per_cta, overflow, timer, meta_len | meta(JSON) | u32 cnt[n_slots] | 紧凑记录
static inline int fi_dump(const FiBuf* b, const char* path, const char* meta_json) {
    if (cudaDeviceSynchronize() != cudaSuccess) return -1;
    std::vector<FiRec> all((size_t)b->n_slots * b->cap);
    uint32_t ovf = 0;
    if (cudaMemcpy(all.data(), b->rec, all.size() * sizeof(FiRec), cudaMemcpyDeviceToHost) != cudaSuccess) return -2;
    if (cudaMemcpy(&ovf, b->overflow, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return -2;
    std::vector<uint32_t> cnt(b->n_slots, 0);
    for (uint32_t s = 0; s < b->n_slots; ++s)
        while (cnt[s] < b->cap && all[(size_t)s * b->cap + cnt[s]].tag != 0u) ++cnt[s];
    FILE* f = fopen(path, "wb");
    if (!f) {
        // 只返回 -3 会让人去查磁盘、权限位、插桩缓冲区，而真因常常是：nsys 下被测进程对非 /tmp 的
        // 工作目录写打开被拒（errno=13），一整轮持锁采集就此白跑（fi_bugs_h20-sk-qwen B10）。
        // 所以把 errno 原文打出来，并直接给可行的下一步。
        fprintf(stderr,
                "fi_dump: fopen(\"%s\", \"wb\") failed: %s (errno=%d). "
                "Check write access to the dump directory for the profiled process.\n",
                path, strerror(errno), errno);
        return -3;
    }
    const std::string meta = fi_meta_augment(meta_json);
    const uint32_t ml = (uint32_t)meta.size();
    const uint32_t hdr[6] = {b->n_slots, b->cap, b->warps_per_cta, ovf, (uint32_t)FI_TIMER, ml};
    fwrite("FILITE02", 1, 8, f);
    fwrite(hdr, 4, 6, f);
    if (ml) fwrite(meta.data(), 1, ml, f);
    fwrite(cnt.data(), 4, cnt.size(), f);
    for (uint32_t s = 0; s < b->n_slots; ++s)
        if (cnt[s]) fwrite(&all[(size_t)s * b->cap], sizeof(FiRec), cnt[s], f);
    fclose(f);
    return (int)ovf;
}
static inline void fi_free(FiBuf* b) {
    // 与 HIP 侧同样的写法：显式接住再丢弃，避免 nodiscard 告警污染编译日志。
    cudaError_t e_rec_ = cudaFree(b->rec);
    cudaError_t e_ovf_ = cudaFree(b->overflow);
    cudaError_t e_cur_ = (b->cur != nullptr) ? cudaFree(b->cur) : cudaSuccess;
    (void)e_rec_;
    (void)e_ovf_;
    (void)e_cur_;
    memset(b, 0, sizeof(*b));
}

// Canonical WeaveProfiler host API (`wp_*`).  Same buffers, same FILITE02 wire format; the decoder
// (weaveprof/pipeline/validate/tools/wp_decode.py) accepts both FILITE02 and WPLITE02 magics.  These C++-linkage
// overloads coexist with the optional `extern "C"` ctypes exports below (different signatures).
static inline cudaError_t wp_alloc(WpBuf* b, uint32_t n_ctas, uint32_t warps_per_cta, uint32_t cap) {
    return fi_alloc(b, n_ctas, warps_per_cta, cap);
}
static inline cudaError_t wp_attach(const WpBuf* b) { return fi_attach(b); }
static inline cudaError_t wp_reset(const WpBuf* b, cudaStream_t st = 0) { return fi_reset(b, st); }
static inline int wp_dump(const WpBuf* b, const char* path, const char* meta_json = "{}") {
    return fi_dump(b, path, meta_json);
}
static inline void wp_free(WpBuf* b) { fi_free(b); }
static inline cudaError_t wp_host_init(WpBuf* buf, uint32_t ctas, uint32_t warps_per_cta, uint32_t capacity) {
    cudaError_t status = fi_alloc(buf, ctas, warps_per_cta, capacity);
    if (status == cudaSuccess) status = fi_attach(buf);
    if (status != cudaSuccess) fi_free(buf);
    return status;
}
static inline int wp_host_dump(const WpBuf* buf, const char* path, const char* meta = "{}") {
    return fi_dump(buf, path, meta);
}

// Optional zero-boilerplate C ABI host exports (`-DFI_DEFINE_HOST_API` or `-DWP_DEFINE_HOST_API`)
// for multi-file PyTorch / ctypes `.so` megakernels (Sparse Attention, GDN, MoE).
#if defined(FI_DEFINE_HOST_API) || defined(WP_DEFINE_HOST_API)
static FiBuf g_wp_host_buf_{};
extern "C" int wp_init(uint32_t n_ctas, uint32_t warps_per_cta, uint32_t cap) {
    if (g_wp_host_buf_.rec != nullptr) fi_free(&g_wp_host_buf_);
    cudaError_t e = fi_alloc(&g_wp_host_buf_, n_ctas, warps_per_cta, cap);
    if (e != cudaSuccess) return (int)e;
    return (int)fi_attach(&g_wp_host_buf_);
}
extern "C" int wp_reset(uint64_t stream_ptr) {
    if (g_wp_host_buf_.rec == nullptr) return -1;
    return (int)fi_reset(&g_wp_host_buf_, reinterpret_cast<cudaStream_t>(stream_ptr));
}
extern "C" int wp_dump(const char* path, const char* meta_json) {
    if (g_wp_host_buf_.rec == nullptr) return -1;
    return fi_dump(&g_wp_host_buf_, path, meta_json);
}
extern "C" void wp_free() {
    if (g_wp_host_buf_.rec != nullptr) {
        (void)fi_attach(nullptr);
        fi_free(&g_wp_host_buf_);
    }
}
#endif

static_assert(sizeof(WpRec) == 16, "WeaveProfiler records must be 16 bytes");

// WP_SYNC_CURSOR_OUT() publishes the live per-lane record cursor to wp_fi_c.cur[slot];
// WP_SYNC_CURSOR_IN() reloads it (e.g. after a callee __device__ function appended records through
// WP_DEVICE_SCOPE).  The cursor lives in a different place for every recorder-context mode, so each
// mode gets its own implementation instead of poking `fi_ctx_.p` (which does not exist under
// -DWP_CTX_REG or -DWP_CTX_SMEM):
//   WP_CTX_SMEM                 : WpSmemCtx in static __shared__ (slot recovered from the lane base pointer)
//   WP_CTX_REG(_NOINLINE)       : bit-packed WpCtxReg::state (slot = bits 30..15, cursor = bits 14..0)
//   default                     : 5-register FiCtx {p, n, cap, slot}
#if FI_LEVEL >= 1
#if defined(WP_CTX_SMEM)
__device__ __forceinline__ void wp_sync_cursor_out_smem(const WpSmemCtx* sc) {
    const FiBuf b = wp_active_buf();
    if (sc->p != nullptr && b.cur != nullptr && b.rec != nullptr && b.cap != 0u) {
        const uint32_t slot = (uint32_t)((size_t)(sc->p - b.rec) / (size_t)b.cap);
        if (slot < b.n_slots) b.cur[slot] = sc->n;
    }
}
__device__ __forceinline__ void wp_sync_cursor_in_smem(WpSmemCtx* sc) {
    const FiBuf b = wp_active_buf();
    if (sc->p != nullptr && b.cur != nullptr && b.rec != nullptr && b.cap != 0u) {
        const uint32_t slot = (uint32_t)((size_t)(sc->p - b.rec) / (size_t)b.cap);
        if (slot < b.n_slots) sc->n = b.cur[slot];
    }
}
#define WP_SYNC_CURSOR_OUT() do { if (fi_recorder()) wp_sync_cursor_out_smem(WP_SMEM_SLOT()); } while (0)
#define WP_SYNC_CURSOR_IN() do { if (fi_recorder()) wp_sync_cursor_in_smem(WP_SMEM_SLOT()); } while (0)
#elif defined(WP_CTX_REG)
__device__ __forceinline__ void wp_sync_cursor_out_reg(const WpCtxReg& c) {
    const FiBuf b = wp_active_buf();
    if ((int32_t)c.state >= 0 && b.cur != nullptr) {
        const uint32_t slot = (c.state >> WP_REG_SLOT_SHIFT) & WP_REG_SLOT_MASK;
        if (slot < b.n_slots) b.cur[slot] = c.state & WP_REG_CUR_MASK;
    }
}
__device__ __forceinline__ void wp_sync_cursor_in_reg(WpCtxReg& c) {
    const FiBuf b = wp_active_buf();
    if ((int32_t)c.state >= 0 && b.cur != nullptr) {
        const uint32_t slot = (c.state >> WP_REG_SLOT_SHIFT) & WP_REG_SLOT_MASK;
        if (slot >= b.n_slots) return;
        const uint32_t eff_cap = (b.cap > WP_REG_CUR_MASK) ? WP_REG_CUR_MASK : b.cap;
        const uint32_t n = b.cur[slot];
        c.state = (n >= eff_cap) ? ((slot << WP_REG_SLOT_SHIFT) | WP_REG_DISABLED)
                                 : ((slot << WP_REG_SLOT_SHIFT) | (n & WP_REG_CUR_MASK));
    }
}
#define WP_SYNC_CURSOR_OUT() do { wp_sync_cursor_out_reg(fi_ctx_); } while (0)
#define WP_SYNC_CURSOR_IN() do { wp_sync_cursor_in_reg(fi_ctx_); } while (0)
#else
__device__ __forceinline__ void wp_sync_cursor_out_ctx(const FiCtx& c) {
    const FiBuf b = wp_active_buf();
    if (c.p != nullptr && b.cur != nullptr && c.slot < b.n_slots) b.cur[c.slot] = c.n;
}
__device__ __forceinline__ void wp_sync_cursor_in_ctx(FiCtx& c) {
    const FiBuf b = wp_active_buf();
    if (c.p != nullptr && b.cur != nullptr && c.slot < b.n_slots) c.n = b.cur[c.slot];
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
