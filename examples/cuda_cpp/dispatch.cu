// Minimal CUDA C++ reference for work-item acquisition.
// SEMANTIC-ID: dispatch
// EXECUTION-SCOPE: one source-declared representative lane.

#include "profiler.cuh"

// The business operation is intentionally atomic: dispatch assigns each CTA a
// distinct work item.  The profiler writes below do not use atomics.
__device__ __forceinline__ unsigned int acquire_work_item(
    unsigned int* next_work_item) {
  return atomicAdd(next_work_item, 1u);
}

extern "C" __global__ void dispatch_kernel(
    unsigned int* next_work_item, unsigned int* assigned_work_items,
    std::int64_t* profile, std::uint32_t max_records,
    std::uint32_t header_elems, std::uint32_t num_tracks) {
  // One representative lane owns both the dispatch result and its trace
  // record, so no profiler slot is concurrently written by peer lanes.
  if (threadIdx.x == 0) {
    const std::uint32_t track = 0;
    const std::uint32_t offset = 0;
    // TRACE-BEGIN
    fast_instrument_ref::interval_start(
        profile, header_elems, blockIdx.x, track, num_tracks, max_records,
        offset, 0u);
    const unsigned int work_item = acquire_work_item(next_work_item);
    fast_instrument_ref::interval_end(
        profile, header_elems, blockIdx.x, track, num_tracks, max_records,
        offset + 1, 0u);
    // TRACE-END
    assigned_work_items[blockIdx.x] = work_item;
  }
}

