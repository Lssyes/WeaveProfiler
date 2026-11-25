// Known-answer smoke kernel for recorders/cuda_cpp/wp_lite.cuh (used by tests/gpu/test_recorder_smoke.py).
//
// Every warp runs `iters` rounds of two stages whose length is fixed by busy-waiting on %globaltimer:
//   EV_A ("load") for a_ns, then EV_B ("mma") for b_ns.
// Warps alternate between role 0 and role 1. The decoded span durations must match a_ns / b_ns, and the
// decoded event sequence must not depend on the recorder context mode (default / WP_CTX_REG / WP_CTX_SMEM)
// or the timer (WP_TIMER=0 globaltimer, WP_TIMER=1 clock64 + anchors).
//
// Usage: smoke_recorder <dump path> <cap per lane> <iters> <a_ns> <b_ns>
#include "wp_lite.cuh"
#include <cstdio>
#include <cstdlib>

enum : uint32_t { EV_A = 1, EV_B = 2 };

__device__ __forceinline__ unsigned long long smoke_gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ void spin_ns(unsigned long long ns) {
    const unsigned long long t0 = smoke_gtimer();
    while (smoke_gtimer() - t0 < ns) {
    }
}

__global__ void smoke(int iters, unsigned long long a_ns, unsigned long long b_ns, int* sink) {
    WP_KERNEL_SCOPE((threadIdx.x / 32) % 2);
    for (int i = 0; i < iters; ++i) {
        WP_B(EV_A, i);
        spin_ns(a_ns);
        WP_E(EV_A, i);
        WP_B(EV_B, i);
        spin_ns(b_ns);
        WP_E(EV_B, i);
    }
    if (threadIdx.x == 0) sink[blockIdx.x] = iters;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <dump> <cap> <iters> <a_ns> <b_ns>\n", argv[0]);
        return 2;
    }
    const char* out = argv[1];
    const unsigned cap = (unsigned)atoi(argv[2]);
    const int iters = atoi(argv[3]);
    const unsigned long long a_ns = strtoull(argv[4], nullptr, 10), b_ns = strtoull(argv[5], nullptr, 10);
    int sms = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int ctas = sms, threads = 128, wpc = threads / 32;
    int* sink;
    CK(cudaMalloc(&sink, ctas * sizeof(int)));
    WpBuf buf;
    CK(wp_host_init(&buf, (unsigned)ctas, (unsigned)wpc, cap));
    smoke<<<ctas, threads>>>(1, 100, 100, sink);  // warm-up (module load) outside the recorded launch
    CK(cudaDeviceSynchronize());
    CK(wp_reset(&buf));
    smoke<<<ctas, threads>>>(iters, a_ns, b_ns, sink);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    char meta[256];
    snprintf(meta, sizeof meta, "{\"kernel\":\"smoke\",\"ctas\":%d,\"warps_per_cta\":%d,\"iters\":%d,\"a_ns\":%llu,\"b_ns\":%llu}",
             ctas, wpc, iters, a_ns, b_ns);
    const int ovf = wp_host_dump(&buf, out, meta);
    printf("{\"ctas\":%d,\"warps_per_cta\":%d,\"overflow\":%d}\n", ctas, wpc, ovf);
    wp_free(&buf);
    return ovf < 0 ? 1 : 0;
}
