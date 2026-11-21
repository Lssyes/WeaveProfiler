// Minimal CUDA C++ reference for one warp-level WMMA operation.
// SEMANTIC-ID: mma
// EXECUTION-SCOPE: physical warp 0, proven by the threadIdx.x predicate.

#include <cuda_fp16.h>
#include <mma.h>

#include "profiler.cuh"

using namespace nvcuda;

extern "C" __global__ void mma_kernel(const half* a, const half* b, float* c,
                                       std::int64_t* profile,
                                       std::uint32_t max_records,
                                       std::uint32_t header_elems,
                                       std::uint32_t num_tracks) {
  if (threadIdx.x >= 32) return;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::load_matrix_sync(a_frag, a, 16);
  wmma::load_matrix_sync(b_frag, b, 16);
  wmma::fill_fragment(c_frag, 0.0f);
  const std::uint32_t track = threadIdx.x >> 5;
  const std::uint32_t offset = 0;
  // TRACE-BEGIN
  if (threadIdx.x == 0) {
    fast_instrument_ref::interval_start(
        profile, header_elems, blockIdx.x, track, num_tracks, max_records,
        offset, 2u);
  }
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  if (threadIdx.x == 0) {
    fast_instrument_ref::interval_end(
        profile, header_elems, blockIdx.x, track, num_tracks, max_records,
        offset + 1, 2u);
  }
  // TRACE-END
  wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}

