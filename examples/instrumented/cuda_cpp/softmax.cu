// Minimal CUDA C++ reference for one-warp stable softmax over 32 values.
// SEMANTIC-ID: softmax
// EXECUTION-SCOPE: one physical warp.

#include <math.h>

extern "C" __global__ void softmax_kernel(const float* src, float* dst) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned row_offset = blockIdx.x * 32u;
  float value = src[row_offset + lane];
  // TRACE-BEGIN
  float row_max = value;
  for (int delta = 16; delta > 0; delta >>= 1)
    row_max = fmaxf(row_max, __shfl_down_sync(0xffffffffu, row_max, delta));
  row_max = __shfl_sync(0xffffffffu, row_max, 0);
  float numerator = expf(value - row_max);
  float denominator = numerator;
  for (int delta = 16; delta > 0; delta >>= 1)
    denominator += __shfl_down_sync(0xffffffffu, denominator, delta);
  denominator = __shfl_sync(0xffffffffu, denominator, 0);
  const float probability = numerator / denominator;
  // TRACE-END
  dst[row_offset + lane] = probability;
}
