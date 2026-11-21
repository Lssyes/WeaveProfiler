// Minimal CUDA C++ reference for one-warp stable softmax over 32 values.
// SEMANTIC-ID: softmax
// EXECUTION-SCOPE: one physical warp.

#include <math.h>

#include "profiler.cuh"

extern "C" __global__ void softmax_kernel(
    const float* src, float* dst, std::int64_t* profile,
    std::uint32_t max_records, std::uint32_t header_elems,
    std::uint32_t num_tracks) {
  if (threadIdx.x >= 32) return;

  const unsigned lane = threadIdx.x & 31u;
  const unsigned row_offset = blockIdx.x * 32u;
  float value = src[row_offset + lane];
  const std::uint32_t track = threadIdx.x >> 5;
  const std::uint32_t offset = 0;
  // TRACE-BEGIN
  if (lane == 0) {
    fast_instrument_ref::interval_start(
        profile, header_elems, blockIdx.x, track, num_tracks, max_records,
        offset, 5u);
  }
  float row_max = value;
  for (int delta = 16; delta > 0; delta >>= 1) {
    row_max = fmaxf(row_max, __shfl_down_sync(0xffffffffu, row_max, delta));
  }
  row_max = __shfl_sync(0xffffffffu, row_max, 0);
  const float numerator = expf(value - row_max);
  float denominator = numerator;
  for (int delta = 16; delta > 0; delta >>= 1) {
    denominator +=
        __shfl_down_sync(0xffffffffu, denominator, delta);
  }
  denominator = __shfl_sync(0xffffffffu, denominator, 0);
  const float probability = numerator / denominator;
  if (lane == 0) {
    fast_instrument_ref::interval_end(
        profile, header_elems, blockIdx.x, track, num_tracks, max_records,
        offset + 1, 5u);
  }
  // TRACE-END
  dst[row_offset + lane] = probability;
}

