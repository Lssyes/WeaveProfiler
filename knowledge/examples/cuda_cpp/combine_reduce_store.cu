// Minimal CUDA C++ reference for a fused combine reduction and final store.
// SEMANTIC-ID: unknown  # requires model review: fused reduction+store has no single canonical semantic_id
// EXECUTION-SCOPE: one physical warp; this observer never represents another warp.
//
// Layout: src[(block, source, lane)] -> dst[(block, lane)].  Each lane reduces
// all routed contributions for one output element and writes the final value.
// The measured interval begins before the first accumulation and ends only
// after every lane has issued its final store.

#include "profiler.cuh"

namespace fast_instrument_combine_ref {

// Reference-local semantic slot.  A target ABI maps it to its own task/phase
// identifier; this numeric value is not a project task ID.
constexpr std::uint32_t kCombineReduceStoreEvent = 0;

}  // namespace fast_instrument_combine_ref

extern "C" __global__ void combine_reduce_store_kernel(
    const float* src, float* dst, std::uint32_t num_sources,
    std::int64_t* profile, std::uint32_t max_records,
    std::uint32_t header_elems, std::uint32_t num_tracks) {
  if (threadIdx.x >= 32) return;

  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t track = threadIdx.x >> 5;
  const std::uint32_t block = blockIdx.x;
  constexpr std::uint32_t kStartOffset = 0;

  if (lane == 0) {
    fast_instrument_ref::interval_start(
        profile, header_elems, block, track, num_tracks, max_records,
        kStartOffset,
        fast_instrument_combine_ref::kCombineReduceStoreEvent);
  }
  __syncwarp();

  float value = 0.0f;
  for (std::uint32_t source = 0; source < num_sources; ++source) {
    value += src[(block * num_sources + source) * 32u + lane];
  }
  dst[block * 32u + lane] = value;

  // The end timestamp belongs after the final store instruction.  The warp
  // rendezvous ensures lane 0 cannot close the interval while another lane is
  // still accumulating or has not yet issued its store.
  __syncwarp();
  if (lane == 0) {
    fast_instrument_ref::interval_end(
        profile, header_elems, block, track, num_tracks, max_records,
        kStartOffset + 1,
        fast_instrument_combine_ref::kCombineReduceStoreEvent);
  }
}
