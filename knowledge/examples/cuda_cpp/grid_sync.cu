// Generic CUDA C++ reference for a cooperative grid rendezvous.
// SEMANTIC-ID: barrier
// EXECUTION-SCOPE: every thread in every CTA; launch cooperatively.
//
// The host must use cudaLaunchCooperativeKernel (or an equivalent cooperative
// launch path) and ensure that the complete grid can be resident.  Every CTA
// and every thread executes the grid synchronization; no subset is allowed.

#include <cooperative_groups.h>

#include "profiler.cuh"

namespace fast_instrument_grid_sync_ref {

namespace cg = cooperative_groups;

// These are reference-local semantic slots.  A host ABI maps them to its own
// task/phase identifiers; the numeric values are not MegaMoE task IDs.
constexpr std::uint32_t kPublishStartEvent = 0;
constexpr std::uint32_t kGridRendezvousEvent = 1;
constexpr std::uint32_t kPublishEndEvent = 2;
constexpr std::uint32_t kGridSyncTrack = 0;

}  // namespace fast_instrument_grid_sync_ref

extern "C" __global__ void grid_sync_kernel(
    const float* src, float* dst, volatile std::uint64_t* cta_start_ns,
    volatile std::uint64_t* cta_end_ns, std::int64_t* profile,
    std::uint32_t max_records, std::uint32_t header_elems,
    std::uint32_t num_tracks) {
  const std::uint32_t index =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);

  // The source read is real work before the rendezvous.  The CTA barrier keeps
  // all threads' work ordered before the representative publishes its start.
  const float value = src[index];
  __syncthreads();

  if (threadIdx.x == 0) {
    constexpr std::uint32_t kStartOffset = 0;
    fast_instrument_ref::interval_start(
        profile, header_elems, blockIdx.x,
        fast_instrument_grid_sync_ref::kGridSyncTrack, num_tracks,
        max_records, kStartOffset,
        fast_instrument_grid_sync_ref::kPublishStartEvent);
    cta_start_ns[blockIdx.x] = fast_instrument_ref::globaltimer_ns();
    __threadfence();
    fast_instrument_ref::interval_end(
        profile, header_elems, blockIdx.x,
        fast_instrument_grid_sync_ref::kGridSyncTrack, num_tracks,
        max_records, kStartOffset + 1,
        fast_instrument_grid_sync_ref::kPublishStartEvent);
  }
  __syncthreads();

  // This interval is an ignored synchronization phase in a visualization,
  // but its timestamps are real and must remain available for validation.
  if (threadIdx.x == 0) {
    constexpr std::uint32_t kGridOffset = 2;
    fast_instrument_ref::interval_start(
        profile, header_elems, blockIdx.x,
        fast_instrument_grid_sync_ref::kGridSyncTrack, num_tracks,
        max_records, kGridOffset,
        fast_instrument_grid_sync_ref::kGridRendezvousEvent);
  }
  // All threads in all CTAs must execute this rendezvous.
  cooperative_groups::this_grid().sync();
  if (threadIdx.x == 0) {
    constexpr std::uint32_t kGridOffset = 2;
    fast_instrument_ref::interval_end(
        profile, header_elems, blockIdx.x,
        fast_instrument_grid_sync_ref::kGridSyncTrack, num_tracks,
        max_records, kGridOffset + 1,
        fast_instrument_grid_sync_ref::kGridRendezvousEvent);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    constexpr std::uint32_t kEndOffset = 4;
    fast_instrument_ref::interval_start(
        profile, header_elems, blockIdx.x,
        fast_instrument_grid_sync_ref::kGridSyncTrack, num_tracks,
        max_records, kEndOffset,
        fast_instrument_grid_sync_ref::kPublishEndEvent);
    cta_end_ns[blockIdx.x] = fast_instrument_ref::globaltimer_ns();
    __threadfence();
    fast_instrument_ref::interval_end(
        profile, header_elems, blockIdx.x,
        fast_instrument_grid_sync_ref::kGridSyncTrack, num_tracks,
        max_records, kEndOffset + 1,
        fast_instrument_grid_sync_ref::kPublishEndEvent);
  }
  __syncthreads();

  // Keep the post-rendezvous output observable and ordered after publication.
  dst[index] = value;
}
