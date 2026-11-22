// fast-instrument 两角色 HIP 示例：链 CTA 等就绪→计算；辅助 CTA 生产。用于验收与开销测量。
// FI_GRAN=0：每 wave64 的 lane 0 记录；FI_GRAN=1：每 CTA 只由 threadIdx.x==0 记录（本 kernel 的角色按 CTA 分，两种粒度都能看全）。
#include "wp_lite_hip.h"
#include <algorithm>
#include <cstdlib>

enum : uint32_t { EV_WAIT = 1, EV_COMPUTE = 2, EV_PRODUCE = 3 };

__global__ void demo(float* out, unsigned* ready, int n_chain, int K, int work) {
    const int cta = blockIdx.x;
    const bool helper = cta >= n_chain;
    FI_OPEN(helper ? 1 : 0);
    float x = threadIdx.x * 1e-3f;
    if (helper) {
        const int h = cta - n_chain, nh = gridDim.x - n_chain;
        for (int k = h; k < K; k += nh) {
            FI_B(EV_PRODUCE, k);
            for (int i = 0; i < work; ++i) x = x * 1.0001f + 0.5f;
            __syncthreads();
            FI_E(EV_PRODUCE, k);
            if (threadIdx.x == 0) {
                __threadfence();
                atomicExch(ready + k, 1u);
            }
        }
    } else {
        for (int k = 0; k < K; ++k) {
            FI_B(EV_WAIT, k);
            if (threadIdx.x == 0) {
                const uint64_t t_start = fi_gtimer();
                const uint64_t timeout = fi_g.timer_hz;  // 1 秒对应的固定频率 tick 数
                while (atomicAdd(ready + k, 0u) == 0u) {
                    __builtin_amdgcn_s_sleep(64);
                    if (fi_gtimer() - t_start > timeout) { atomicExch(ready + k, 2u); break; }
                }
            }
            __syncthreads();
            FI_E(EV_WAIT, k);
            FI_B(EV_COMPUTE, k);
            for (int i = 0; i < work / 4; ++i) x = x * 1.0001f + 0.5f;
            FI_E(EV_COMPUTE, k);
        }
    }
    FI_CLOSE();
    out[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); return 1; } } while (0)

// 跑一次并转储；warmup=false 用于正确性/转储，warmup=true 用于计时循环。
static int run_once(FiBuf* fb, float* out, unsigned* ready, int grid, int threads, int n_chain, int K, int work, const char* path) {
    CK(hipMemset(ready, 0, K * 4));
    CK(fi_reset(fb));
    demo<<<grid, threads>>>(out, ready, n_chain, K, work);
    CK(hipDeviceSynchronize());
    CK(hipGetLastError());
    if (path) {
        char meta[256];
        snprintf(meta, sizeof meta, "{\"kernel\":\"demo\",\"grid\":%d,\"n_chain\":%d,\"n_helper\":%d,\"K\":%d,\"fi_level\":%d,\"fi_timer\":%d}",
                 grid, n_chain, grid - n_chain, K, FI_LEVEL, FI_TIMER);
        const int r = fi_dump(fb, path, meta);
        printf("DUMP %s rc=%d\n", path, r);
        if (r) return 2;
    }
    return 0;
}

int main(int argc, char** argv) {
    const char* out_path = argc > 1 ? argv[1] : "demo_hip.bin";       // FI 转储路径
    const char* data_out = argc > 2 ? argv[2] : nullptr;              // 正确性：out 数组写到这个文件
    int sms = 0;
    CK(hipDeviceGetAttribute(&sms, hipDeviceAttributeMultiprocessorCount, 0));
    const int n_helper = std::max(1, sms / 8), n_chain = std::min(128, sms - n_helper);
    const int grid = n_chain + n_helper, threads = 128, K = 64, work = 20000;
    float* out; unsigned* ready;
    CK(hipMalloc(&out, (size_t)grid * threads * 4));
    CK(hipMalloc(&ready, K * 4));
    FiBuf fb;
    const uint32_t wpc = FI_GRAN ? 1u : (uint32_t)((threads + 63) / 64);
    const uint32_t cap = (uint32_t)(4 * K + 8);  // 两种粒度下每记录线程的条数相同（gran 只减记录线程数）
    CK(fi_alloc(&fb, grid, wpc, cap));
    CK(fi_attach(&fb));
    // 开关按“值”判，不按“变量是否存在”判：FI_ROCPROF_ONCE=0 就地执行时也算关闭
    // （fi_bugs_amd-sk-ds B17 的同款坑：presence 当开关会让插桩侧只剩 1 个计时样本）。
    const char* once_env = getenv("FI_ROCPROF_ONCE");
    if (once_env && atoi(once_env) != 0) {  // rocprof 采集模式：只跑一次并转储
        const int rc = run_once(&fb, out, ready, grid, threads, n_chain, K, work, out_path);
        goto done;
    }
    {
        hipEvent_t a, b;
        CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
        std::vector<float> ts;
        const int iters = data_out ? 1 : 25;
        for (int it = 0; it < iters; ++it) {
            CK(hipMemset(ready, 0, K * 4));
            CK(fi_reset(&fb));
            CK(hipEventRecord(a));
            demo<<<grid, threads>>>(out, ready, n_chain, K, work);
            CK(hipEventRecord(b));
            CK(hipEventSynchronize(b));
            CK(hipGetLastError());
            float ms; CK(hipEventElapsedTime(&ms, a, b));
            if (it >= 5) ts.push_back(ms * 1000.f);
        }
        std::sort(ts.begin(), ts.end());
        printf("demo_hip FI_LEVEL=%d FI_TIMER=%d FI_GRAN=%d grid=%d (chain %d, helper %d) ITER_US median=%.2f min=%.2f max=%.2f timer_hz=%llu\n",
               FI_LEVEL, FI_TIMER, FI_GRAN, grid, n_chain, n_helper,
               ts.empty() ? 0.f : ts[ts.size() / 2], ts.empty() ? 0.f : ts.front(), ts.empty() ? 0.f : ts.back(),
               (unsigned long long)fb.timer_hz);
        if (FI_LEVEL >= 1) {
            char meta[256];
            snprintf(meta, sizeof meta, "{\"kernel\":\"demo\",\"grid\":%d,\"n_chain\":%d,\"n_helper\":%d,\"K\":%d,\"fi_level\":%d,\"fi_timer\":%d}",
                     grid, n_chain, grid - n_chain, K, FI_LEVEL, FI_TIMER);
            const int r = fi_dump(&fb, out_path, meta);
            printf("DUMP %s rc=%d\n", out_path, r);
            if (r) return 2;
        }
        CK(hipEventDestroy(a)); CK(hipEventDestroy(b));
    }
    if (data_out) {
        std::vector<float> h((size_t)grid * threads);
        CK(hipMemcpy(h.data(), out, h.size() * 4, hipMemcpyDeviceToHost));
        FILE* f = fopen(data_out, "wb");
        if (!f) { fprintf(stderr, "cannot open %s\n", data_out); return 1; }
        fwrite(h.data(), 4, h.size(), f);
        fclose(f);
        printf("WROTE %s (%zu floats)\n", data_out, h.size());
    }
done:
    fi_free(&fb); hipFree(ready); hipFree(out);
    return 0;
}
