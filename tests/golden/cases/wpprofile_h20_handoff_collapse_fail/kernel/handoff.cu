#include <cuda_runtime.h>
#include <stdint.h>

// One producing warp copies a complete slot asynchronously. Four consuming
// warps wait on a split named CTA barrier before issuing dependent tensor ops.
extern "C" __global__ void warp_handoff(const uint32_t *x, float *out, int k_steps) {
    __shared__ uint32_t tiles[2][128];
    const unsigned warp = threadIdx.x / 32;
    const unsigned lane = threadIdx.x % 32;
    const unsigned s = blockIdx.x % 2;
    if (warp == 4) {
        const unsigned address = __cvta_generic_to_shared(&tiles[s][4 * lane]);
        const uint32_t *src = x + 128 * blockIdx.x + 4 * lane;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(address), "l"(src) : "memory");
        asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
        asm volatile("bar.arrive 1, 160;" ::: "memory");
    } else {
        asm volatile("bar.sync 1, 160;" ::: "memory");
        const uint32_t *a = tiles[s] + 4 * lane;
        const uint32_t *b = tiles[s] + 4 * ((lane + warp) % 32);
        float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        for (int k = 0; k < k_steps; ++k) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        }
        const unsigned offset = 4 * (blockIdx.x * 128 + threadIdx.x);
        out[offset] = c0;
        out[offset + 1] = c1;
        out[offset + 2] = c2;
        out[offset + 3] = c3;
    }
}
