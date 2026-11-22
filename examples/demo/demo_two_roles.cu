// fast-instrument 示例：一个 kernel 两种角色（链 CTA 逐块等就绪再计算；辅助 CTA 生产块并置就绪计数），
// 结构上模仿 GDN 复合算子。用来验证 wp_lite.cuh 在目标机上能编、能记、能解码，并量出插桩开销。
// 编译：nvcc -O3 -std=c++17 -arch=<sm> -DFI_LEVEL=2 [-DFI_TIMER=1] -I../include demo_two_roles.cu -o demo_l2
//      （再编一份 -DFI_LEVEL=0 对照开销）
// 运行：./demo_l2 <out.bin>     打印 ITER_US 中位，并在 FI_LEVEL>=1 时写转储
#include "wp_lite.cuh"
#include <algorithm>
#include <cstdlib>
#include <cuda_profiler_api.h>

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
                // 自旋带退避与超时（约 1 s）：生产者没跑起来时不死等，避免拖住 GPU
                const uint64_t t_start = fi_gtimer();
                while (atomicAdd(ready + k, 0u) == 0u) {
                    __nanosleep(64);
                    if (fi_gtimer() - t_start > 1000000000ull) { atomicExch(ready + k, 2u); break; }
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

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)

int main(int argc, char** argv) {
    const char* out_path = argc > 1 ? argv[1] : "demo.bin";
    int sms = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int n_helper = std::max(1, sms / 8), n_chain = std::min(128, sms - n_helper);  // 全部常驻，避免自旋死锁
    const int grid = n_chain + n_helper, threads = 128, K = 64, work = 20000;
    float* out;
    unsigned* ready;
    CK(cudaMalloc(&out, (size_t)grid * threads * 4));
    CK(cudaMalloc(&ready, K * 4));
    FiBuf fb;
    CK(fi_alloc(&fb, grid, threads / 32, 4 * K + 8));
    CK(fi_attach(&fb));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<float> ts;
    for (int it = 0; it < 25; ++it) {
        CK(cudaMemset(ready, 0, K * 4));
        CK(fi_reset(&fb));
        CK(cudaEventRecord(a));
        demo<<<grid, threads>>>(out, ready, n_chain, K, work);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        CK(cudaGetLastError());
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        if (it >= 5) ts.push_back(ms * 1000.f);
    }
    // nsys 模式：开录后先跑 FI_DEMO_NSYS_MS 毫秒不记录的 launch（垫掉 nsys 开录延迟、并让 GPU 在锁内一直有活），
    // 再跑最后一次带记录的 launch；对齐时 --instance -1 取的就是它。
    // 开关按“值”判，不按“变量是否存在”判：FI_DEMO_NSYS_MS=0 就地执行时也必须当成关闭，
    // 否则一次普通计时跑会被塞进 nsys 模式，插桩侧只剩 1 个计时样本（fi_bugs_amd-sk-ds B17 的同款坑）。
    const char* nm = getenv("FI_DEMO_NSYS_MS");
    const int nsys_ms = nm ? atoi(nm) : 0;
    if (nm && nsys_ms > 0) {
        CK(fi_attach(nullptr));
        CK(cudaProfilerStart());
        cudaEvent_t t0, t1;
        CK(cudaEventCreate(&t0));
        CK(cudaEventCreate(&t1));
        CK(cudaEventRecord(t0));
        float el = 0.f;
        while (el < (float)nsys_ms) {
            for (int j = 0; j < 20; ++j) {
                CK(cudaMemsetAsync(ready, 0, K * 4));
                demo<<<grid, threads>>>(out, ready, n_chain, K, work);
            }
            CK(cudaEventRecord(t1));
            CK(cudaEventSynchronize(t1));
            CK(cudaEventElapsedTime(&el, t0, t1));
        }
        CK(fi_attach(&fb));
        CK(cudaMemset(ready, 0, K * 4));
        CK(fi_reset(&fb));
        demo<<<grid, threads>>>(out, ready, n_chain, K, work);
        CK(cudaDeviceSynchronize());
        CK(cudaProfilerStop());
        printf("NSYS_MODE padded_ms=%.0f then 1 instrumented launch\n", el);
    }
    std::sort(ts.begin(), ts.end());
    printf("FI_LEVEL=%d FI_TIMER=%d grid=%d (chain %d, helper %d) ITER_US median=%.2f min=%.2f max=%.2f\n", FI_LEVEL, FI_TIMER, grid, n_chain, n_helper,
           ts[ts.size() / 2], ts.front(), ts.back());
    if (FI_LEVEL >= 1) {
        char meta[256];
        snprintf(meta, sizeof meta, "{\"kernel\":\"demo\",\"grid\":%d,\"n_chain\":%d,\"n_helper\":%d,\"K\":%d,\"fi_level\":%d,\"fi_timer\":%d}", grid,
                 n_chain, n_helper, K, FI_LEVEL, FI_TIMER);
        const int r = fi_dump(&fb, out_path, meta);
        printf("DUMP %s rc=%d\n", out_path, r);
    }
    fi_free(&fb);
    return 0;
}
