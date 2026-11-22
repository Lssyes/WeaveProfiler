// Self-contained H20 GEMM instrumentation probe.
// C[M,N] = A[M,K] * B[K,N], FP16 inputs, FP32 accumulation.
// FI_LEVEL controls recorder verbosity only: 0=off, 1=entry-only,
// 2=full-events. FI_SAMPLE controls K-tile event density independently:
// 0=all K tiles (full audit), 1=one representative K tile (fast sampled
// feedback). Sampled capture requires FI_LEVEL=2 FI_SAMPLE=1; full audit
// requires FI_LEVEL=2 FI_SAMPLE=0; FI_LEVEL=1 is never full-audit evidence.
#ifndef FI_SAMPLE
#define FI_SAMPLE 0
#endif
#if (FI_SAMPLE != 0) && (FI_SAMPLE != 1)
#error "FI_SAMPLE must be 0 (all K tiles/full audit) or 1 (one representative K tile)"
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "wp_lite.cuh"

using namespace nvcuda;

namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int THREADS = 128;
constexpr int WARPS = THREADS / 32;

enum : uint32_t {
    EV_LOAD = 1,
    EV_WAIT = 2,
    EV_MMA = 3,
    EV_BARRIER = 4,
    EV_STORE = 5,
    EV_GEMM = 6,
};

__global__ void init_half_from_float(const float* src, __half* dst, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = __float2half_rn(src[i]);
    }
}

__global__ __launch_bounds__(THREADS) void gemm_wmma(
        const __half* __restrict__ A,
        const __half* __restrict__ B,
        float* __restrict__ C,
        int M, int N, int K) {
    __shared__ __align__(16) __half As[BM * BK];
    __shared__ __align__(16) __half Bs[BK * BN];

    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    const int tile_id = tile_m * gridDim.x + tile_n;
    const int warp = threadIdx.x >> 5;
    const int warp_m = warp >> 1;
    const int warp_n = warp & 1;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            wmma::fill_fragment(acc[mi][ni], 0.0f);
        }
    }

    FI_OPEN(0);
    FI_BG(2, EV_GEMM, tile_id);

    for (int kb = 0; kb < K; kb += BK) {
        const bool record_semantic = (FI_SAMPLE == 0) || (kb == 0);
        if (record_semantic) FI_BG(0, EV_LOAD, kb / BK);
        // 128 threads move 2 KiB A tile and 2 KiB B tile with one 16-byte copy each.
        {
            const int row = threadIdx.x >> 1;
            const int off = (threadIdx.x & 1) * 8;
            *reinterpret_cast<uint4*>(&As[row * BK + off]) =
                *reinterpret_cast<const uint4*>(A + (size_t)(tile_m * BM + row) * K + kb + off);
        }
        {
            const int row = threadIdx.x >> 3;
            const int off = (threadIdx.x & 7) * 8;
            *reinterpret_cast<uint4*>(&Bs[row * BN + off]) =
                *reinterpret_cast<const uint4*>(B + (size_t)(kb + row) * N + tile_n * BN + off);
        }
        if (record_semantic) FI_EG(0, EV_LOAD, kb / BK);

        if (record_semantic) FI_BG(1, EV_WAIT, kb / BK);
        __syncthreads();
        if (record_semantic) FI_EG(1, EV_WAIT, kb / BK);

        if (record_semantic) FI_BG(1, EV_MMA, kb / BK);
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, As + (warp_m * 32 + mi * 16) * BK, BK);
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, Bs + (warp_n * 32 + ni * 16), BN);
                wmma::mma_sync(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
            }
        }
        if (record_semantic) FI_EG(1, EV_MMA, kb / BK);

        if (record_semantic) FI_BG(1, EV_BARRIER, kb / BK);
        __syncthreads();
        if (record_semantic) FI_EG(1, EV_BARRIER, kb / BK);
    }

    FI_BG(0, EV_STORE, tile_id);
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            float* dst = C + (size_t)(tile_m * BM + warp_m * 32 + mi * 16) * N +
                         tile_n * BN + warp_n * 32 + ni * 16;
            wmma::store_matrix_sync(dst, acc[mi][ni], N, wmma::mem_row_major);
        }
    }
    FI_EG(0, EV_STORE, tile_id);

    FI_EG(2, EV_GEMM, tile_id);
    FI_CLOSE();
}

struct Shape {
    int m;
    int n;
    int k;
    const char* name;
};

constexpr Shape kSmall{256, 256, 256, "small"};
constexpr Shape kProfile{2048, 2048, 4096, "profile"};

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)
#define CR(x) do { curandStatus_t e_ = (x); if (e_ != CURAND_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuRAND %s @%d: %d\n", #x, __LINE__, (int)e_); return 1; } } while (0)

double now_epoch() {
    struct timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

struct DeviceBuffers {
    __half* a = nullptr;
    __half* b = nullptr;
    float* c = nullptr;
    float* ref = nullptr;
    float* af = nullptr;
    float* bf = nullptr;
    curandGenerator_t gen = nullptr;

    ~DeviceBuffers() {
        if (gen) curandDestroyGenerator(gen);
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        cudaFree(ref);
        cudaFree(af);
        cudaFree(bf);
    }
};

int init_buffers(DeviceBuffers& d, const Shape& s) {
    const size_t ae = (size_t)s.m * s.k;
    const size_t be = (size_t)s.k * s.n;
    const size_t ce = (size_t)s.m * s.n;
    CK(cudaMalloc(&d.a, ae * sizeof(__half)));
    CK(cudaMalloc(&d.b, be * sizeof(__half)));
    CK(cudaMalloc(&d.c, ce * sizeof(float)));
    CK(cudaMalloc(&d.ref, ce * sizeof(float)));
    CK(cudaMalloc(&d.af, ae * sizeof(float)));
    CK(cudaMalloc(&d.bf, be * sizeof(float)));
    CR(curandCreateGenerator(&d.gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CR(curandSetPseudoRandomGeneratorSeed(d.gen, 20261114ULL));
    CR(curandGenerateNormal(d.gen, d.af, ae, 0.0f, 1.0f));
    CR(curandGenerateNormal(d.gen, d.bf, be, 0.0f, 1.0f));
    init_half_from_float<<<256, 256>>>(d.af, d.a, ae);
    init_half_from_float<<<256, 256>>>(d.bf, d.b, be);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    return 0;
}

int launch_gemm(const DeviceBuffers& d, const Shape& s, cudaStream_t stream = 0) {
    dim3 grid((s.n + BN - 1) / BN, (s.m + BM - 1) / BM);
    gemm_wmma<<<grid, THREADS, 0, stream>>>(d.a, d.b, d.c, s.m, s.n, s.k);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

__global__ __launch_bounds__(256) void gemm_simt_reference(
        const __half* __restrict__ A,
        const __half* __restrict__ B,
        float* __restrict__ C,
        int M, int N, int K) {
    __shared__ float As[16 * 16];
    __shared__ float Bs[16 * 16];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int col = blockIdx.x * 16 + tx;
    const int row = blockIdx.y * 16 + ty;
    float acc = 0.0f;
    for (int kb = 0; kb < K; kb += 16) {
        As[ty * 16 + tx] = __half2float(A[(size_t)row * K + kb + tx]);
        Bs[ty * 16 + tx] = __half2float(B[(size_t)(kb + ty) * N + col]);
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            acc = fmaf(As[ty * 16 + kk], Bs[kk * 16 + tx], acc);
        }
        __syncthreads();
    }
    C[(size_t)row * N + col] = acc;
}

int compute_reference(DeviceBuffers& d, const Shape& s) {
    dim3 block(16, 16);
    dim3 grid((s.n + 15) / 16, (s.m + 15) / 16);
    gemm_simt_reference<<<grid, block>>>(d.a, d.b, d.ref, s.m, s.n, s.k);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

int verify(const DeviceBuffers& d, const Shape& s, double* max_abs, double* rel_l2,
           long long* nonfinite, long long* mismatches) {
    const size_t n = (size_t)s.m * s.n;
    std::vector<float> got(n), ref(n);
    CK(cudaMemcpy(got.data(), d.c, n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ref.data(), d.ref, n * sizeof(float), cudaMemcpyDeviceToHost));
    double sum_err2 = 0.0, sum_ref2 = 0.0;
    *max_abs = 0.0;
    *nonfinite = 0;
    *mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        const double g = got[i], r = ref[i];
        if (!std::isfinite(g) || !std::isfinite(r)) {
            ++*nonfinite;
            continue;
        }
        const double err = std::fabs(g - r);
        *max_abs = std::max(*max_abs, err);
        sum_err2 += err * err;
        sum_ref2 += r * r;
        const double tol = 5e-3 + 2e-2 * std::fabs(r);
        if (err > tol) {
            if (*mismatches < 5) {
                std::fprintf(stderr, "mismatch[%zu] got=%.9g ref=%.9g err=%.9g\n", i, g, r, err);
            }
            ++*mismatches;
        }
    }
    *rel_l2 = sum_ref2 > 0.0 ? std::sqrt(sum_err2 / sum_ref2) : 0.0;
    return (*nonfinite == 0 && *mismatches == 0) ? 0 : 1;
}

struct Timings {
    double median_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
};

int time_kernel(const DeviceBuffers& d, const Shape& s, int warmup, int iters, Timings* out) {
    for (int i = 0; i < warmup; ++i) {
        if (launch_gemm(d, s)) return 1;
    }
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<double> us;
    for (int i = 0; i < iters; ++i) {
        CK(cudaEventRecord(a));
        if (launch_gemm(d, s)) return 1;
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, a, b));
        us.push_back((double)ms * 1000.0);
    }
    std::sort(us.begin(), us.end());
    out->median_us = us[us.size() / 2];
    out->min_us = us.front();
    out->max_us = us.back();
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return 0;
}

int time_kernel_attached(const DeviceBuffers& d, const Shape& s, FiBuf& fb,
                        int warmup, int iters, Timings* out) {
    CK(fi_attach(&fb));
    for (int i = 0; i < warmup; ++i) {
        CK(fi_reset(&fb));
        if (launch_gemm(d, s)) return 1;
    }
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<double> us;
    for (int i = 0; i < iters; ++i) {
        CK(fi_reset(&fb));
        CK(cudaEventRecord(a));
        if (launch_gemm(d, s)) return 1;
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, a, b));
        us.push_back((double)ms * 1000.0);
    }
    std::sort(us.begin(), us.end());
    out->median_us = us[us.size() / 2];
    out->min_us = us.front();
    out->max_us = us.back();
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return 0;
}

int parse_shape(const char* name, Shape* s) {
    if (std::strcmp(name, "small") == 0) { *s = kSmall; return 0; }
    if (std::strcmp(name, "profile") == 0) { *s = kProfile; return 0; }
    return 1;
}

const char* arg_value(int argc, char** argv, const char* key, const char* fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return fallback;
}

int run_one(const Shape& s, const char* mode, const char* dump_path, int warmup, int iters,
            int capture_iters, bool do_verify) {
    DeviceBuffers d;
    if (init_buffers(d, s)) return 1;
    if (compute_reference(d, s)) return 1;

    double max_abs = 0.0, rel_l2 = 0.0;
    long long nonfinite = 0, mismatches = 0;
    if (do_verify) {
        if (launch_gemm(d, s)) return 1;
        CK(cudaDeviceSynchronize());
        if (verify(d, s, &max_abs, &rel_l2, &nonfinite, &mismatches)) {
            std::fprintf(stderr, "correctness FAIL shape=%s max_abs=%.9g rel_l2=%.9g nonfinite=%lld mismatches=%lld\n",
                         s.name, max_abs, rel_l2, nonfinite, mismatches);
            return 1;
        }
    }

    Timings t;
    if (std::strcmp(mode, "bench") == 0) {
        if (time_kernel(d, s, warmup, iters, &t)) return 1;
        const double flops = 2.0 * s.m * s.n * s.k;
        std::printf("{\"status\":\"PASS\",\"shape\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,"
                    "\"fi_level\":%d,\"fi_sample\":%d,\"fi_timer\":%d,\"gran\":%d,"
                    "\"checked_elements\":%zu,\"max_abs\":%.9g,\"rel_l2\":%.9g,"
                    "\"nonfinite\":%lld,\"mismatches\":%lld,"
                    "\"median_us\":%.6f,\"min_us\":%.6f,\"max_us\":%.6f,\"tflops\":%.6f}\n",
                    s.name, s.m, s.n, s.k, FI_LEVEL, FI_SAMPLE, FI_TIMER, FI_GRAN,
                    (size_t)s.m * s.n, max_abs, rel_l2, nonfinite, mismatches,
                    t.median_us, t.min_us, t.max_us,
                    flops / (t.median_us * 1e6));
        return 0;
    }

#if FI_LEVEL >= 1
    const uint32_t n_ctas = (uint32_t)(((s.m + BM - 1) / BM) * ((s.n + BN - 1) / BN));
    // FI_LEVEL=2 emits 8 semantic records per selected K tile, plus 6
    // CTA-level records (open, GEMM begin/end, store begin/end, close).
    // FI_SAMPLE=0 selects every K tile; FI_SAMPLE=1 selects the first tile only.
    const uint32_t k_tiles = (uint32_t)s.k / BK;
    const uint32_t recorded_tiles = FI_SAMPLE == 1 ? 1u : k_tiles;
    const uint32_t cap = 8u * recorded_tiles + 16u;
    FiBuf fb;
    CK(fi_alloc(&fb, n_ctas, WARPS, cap));
    CK(fi_attach(nullptr));

    if (std::strcmp(mode, "bench-attached") == 0) {
        Timings t;
        if (time_kernel_attached(d, s, fb, warmup, iters, &t)) return 1;
        if (do_verify) {
            if (verify(d, s, &max_abs, &rel_l2, &nonfinite, &mismatches)) {
                std::fprintf(stderr, "attached bench correctness FAIL shape=%s max_abs=%.9g rel_l2=%.9g nonfinite=%lld mismatches=%lld\n",
                             s.name, max_abs, rel_l2, nonfinite, mismatches);
                return 1;
            }
        }
        const double flops = 2.0 * s.m * s.n * s.k;
        std::printf("{\"status\":\"PASS\",\"mode\":\"bench_attached\",\"measurement_scope\":\"recorder_attached\","
                    "\"shape\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,"
                    "\"fi_level\":%d,\"fi_sample\":%d,\"fi_timer\":%d,\"gran\":%d,"
                    "\"checked_elements\":%zu,\"max_abs\":%.9g,\"rel_l2\":%.9g,"
                    "\"nonfinite\":%lld,\"mismatches\":%lld,"
                    "\"median_us\":%.6f,\"min_us\":%.6f,\"max_us\":%.6f,\"tflops\":%.6f}\n",
                    s.name, s.m, s.n, s.k, FI_LEVEL, FI_SAMPLE, FI_TIMER, FI_GRAN,
                    (size_t)s.m * s.n, max_abs, rel_l2, nonfinite, mismatches,
                    t.median_us, t.min_us, t.max_us,
                    flops / (t.median_us * 1e6));
        fi_free(&fb);
        return 0;
    }

    if (std::strcmp(mode, "dump") == 0) {
        for (int i = 0; i < warmup; ++i) if (launch_gemm(d, s)) return 1;
        CK(cudaDeviceSynchronize());
        CK(fi_attach(&fb));
        CK(fi_reset(&fb));
        if (launch_gemm(d, s)) return 1;
        CK(cudaDeviceSynchronize());
        // Prove that an attached recorder cannot corrupt the GEMM output.
        if (do_verify) {
            if (verify(d, s, &max_abs, &rel_l2, &nonfinite, &mismatches)) {
                std::fprintf(stderr, "attached correctness FAIL shape=%s max_abs=%.9g rel_l2=%.9g nonfinite=%lld mismatches=%lld\n",
                             s.name, max_abs, rel_l2, nonfinite, mismatches);
                return 1;
            }
        }
        char meta[512];
        std::snprintf(meta, sizeof(meta),
                      "{\"kernel\":\"gemm_wmma\",\"arch\":\"sm_90\",\"shape\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,"
                      "\"grid\":%u,\"threads\":%d,\"fi_level\":%d,\"fi_sample\":%d,"
                      "\"fi_timer\":%d,\"gran\":%d}",
                      s.name, s.m, s.n, s.k, n_ctas, THREADS, FI_LEVEL, FI_SAMPLE,
                      FI_TIMER, FI_GRAN);
        const int rc = fi_dump(&fb, dump_path, meta);
        std::printf("{\"status\":\"%s\",\"mode\":\"dump\",\"attached_verify\":%s,\"checked_elements\":%zu,\"max_abs\":%.9g,\"rel_l2\":%.9g,\"dump\":\"%s\",\"overflow\":%d,\"ctas\":%u,\"cap\":%u}\n",
                    rc == 0 ? "PASS" : "FAIL", do_verify ? "true" : "false",
                    (size_t)s.m * s.n, max_abs, rel_l2, dump_path, rc, n_ctas, cap);
        fi_free(&fb);
        return rc == 0 ? 0 : 2;
    }

    if (std::strcmp(mode, "nsys") == 0) {
        for (int i = 0; i < warmup; ++i) if (launch_gemm(d, s)) return 1;
        CK(cudaDeviceSynchronize());
        CK(cudaProfilerStart());
        for (int i = 0; i < capture_iters - 1; ++i) if (launch_gemm(d, s)) return 1;
        CK(cudaDeviceSynchronize());
        CK(fi_attach(&fb));
        CK(fi_reset(&fb));
        if (launch_gemm(d, s)) return 1;
        CK(cudaDeviceSynchronize());
        CK(cudaProfilerStop());
        // The final captured launch must still match every reference element.
        if (do_verify) {
            if (verify(d, s, &max_abs, &rel_l2, &nonfinite, &mismatches)) {
                std::fprintf(stderr, "attached correctness FAIL shape=%s max_abs=%.9g rel_l2=%.9g nonfinite=%lld mismatches=%lld\n",
                             s.name, max_abs, rel_l2, nonfinite, mismatches);
                return 1;
            }
        }
        char meta[512];
        std::snprintf(meta, sizeof(meta),
                      "{\"kernel\":\"gemm_wmma\",\"arch\":\"sm_90\",\"shape\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,"
                      "\"grid\":%u,\"threads\":%d,\"fi_level\":%d,\"fi_sample\":%d,"
                      "\"fi_timer\":%d,\"gran\":%d}",
                      s.name, s.m, s.n, s.k, n_ctas, THREADS, FI_LEVEL, FI_SAMPLE,
                      FI_TIMER, FI_GRAN);
        const int rc = fi_dump(&fb, dump_path, meta);
        std::printf("{\"status\":\"%s\",\"mode\":\"nsys\",\"attached_verify\":%s,\"checked_elements\":%zu,\"max_abs\":%.9g,\"rel_l2\":%.9g,\"dump\":\"%s\",\"overflow\":%d,"
                    "\"capture_iters\":%d,\"ctas\":%u,\"cap\":%u}\n",
                    rc == 0 ? "PASS" : "FAIL", do_verify ? "true" : "false",
                    (size_t)s.m * s.n, max_abs, rel_l2, dump_path, rc, capture_iters, n_ctas, cap);
        fi_free(&fb);
        return rc == 0 ? 0 : 2;
    }
#else
    (void)dump_path;
    (void)capture_iters;
    if (std::strcmp(mode, "dump") == 0 || std::strcmp(mode, "nsys") == 0) {
        std::fprintf(stderr, "mode %s requires FI_LEVEL>=1\n", mode);
        return 2;
    }
#endif

    std::fprintf(stderr, "unknown mode: %s\n", mode);
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    const char* shape_name = arg_value(argc, argv, "--shape", "small");
    const char* mode = arg_value(argc, argv, "--mode", "bench");
    const char* dump_path = arg_value(argc, argv, "--dump", "run/gemm.fi.bin");
    const int warmup = std::atoi(arg_value(argc, argv, "--warmup", "5"));
    const int iters = std::atoi(arg_value(argc, argv, "--iters", "25"));
    const int capture_iters = std::atoi(arg_value(argc, argv, "--capture-iters", "8"));
    const bool no_verify = std::strcmp(arg_value(argc, argv, "--no-verify", "0"), "1") == 0;
    Shape s{};
    if (parse_shape(shape_name, &s)) {
        std::fprintf(stderr, "shape must be small or profile\n");
        return 2;
    }
    const int om = std::atoi(arg_value(argc, argv, "--m", "0"));
    const int on = std::atoi(arg_value(argc, argv, "--n", "0"));
    const int ok = std::atoi(arg_value(argc, argv, "--k", "0"));
    if (om > 0) s.m = om;
    if (on > 0) s.n = on;
    if (ok > 0) s.k = ok;
    if ((s.m & 63) || (s.n & 63) || (s.k & 15)) {
        std::fprintf(stderr, "m,n must be multiples of 64 and k a multiple of 16\n");
        return 2;
    }
    return run_one(s, mode, dump_path, warmup, iters, capture_iters, !no_verify);
}
