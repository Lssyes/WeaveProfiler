// Generic CUDA C++ reference for a system-scope rank barrier.
// SEMANTIC-ID: barrier
// EXECUTION-SCOPE: one representative lane per rank, with CTA participation.
//
// The arrive/release arrays must be visible to every participating rank.  They
// may be backed by NVSHMEM symmetric memory, CUDA IPC/P2P mapped memory, or
// another system-scope allocation supplied by the host.  This example does not
// assume a particular communication library.

#include "profiler.cuh"

namespace fast_instrument_rank_sync_ref {

// These are reference-local semantic slots.  A host ABI maps them to its own
// task/phase identifiers; the numeric values are not MegaMoE task IDs.
constexpr std::uint32_t kRankArriveEvent = 0;
constexpr std::uint32_t kRankBarrierEvent = 1;
constexpr std::uint32_t kRankReleaseEvent = 2;
constexpr std::uint32_t kRankSyncTrack = 0;

__device__ __forceinline__ void wait_for_all_ranks(
    volatile const std::uint32_t* rank_arrive, std::uint32_t epoch,
    std::uint32_t num_ranks) {
  for (std::uint32_t rank = 0; rank < num_ranks; ++rank) {
    while (rank_arrive[rank] != epoch) {
      // The load is volatile and therefore remains a system-visible poll.
      __nanosleep(64);
    }
  }
}

}  // namespace fast_instrument_rank_sync_ref

// Each rank launches one CTA.  Lane 0 performs the system-scope publication
// and owns the corresponding profile slots; the other CTA threads participate
// in the local fences so the rank operation has explicit CTA visibility.
extern "C" __global__ void rank_sync_kernel(
    volatile std::uint32_t* rank_arrive,
    volatile std::uint32_t* rank_release,
    std::uint32_t rank_id, std::uint32_t num_ranks, std::uint32_t epoch,
    std::int64_t* profile, std::uint32_t max_records,
    std::uint32_t header_elems, std::uint32_t num_tracks,
    std::uint32_t* visible_epoch) {
  if (threadIdx.x == 0) {
    // Every interval consumes two records.  The caller must reserve at least
    // six records in track 0 for this reference kernel.
    constexpr std::uint32_t kArriveOffset = 0;
    constexpr std::uint32_t kBarrierOffset = 2;
    constexpr std::uint32_t kReleaseOffset = 4;

    // rank_arrive: this rank publishes its arrival with system visibility.
    fast_instrument_ref::interval_start(
        profile, header_elems, blockIdx.x,
        fast_instrument_rank_sync_ref::kRankSyncTrack, num_tracks, max_records,
        kArriveOffset, fast_instrument_rank_sync_ref::kRankArriveEvent);
    rank_arrive[rank_id] = epoch;
    __threadfence_system();
    fast_instrument_ref::interval_end(
        profile, header_elems, blockIdx.x,
        fast_instrument_rank_sync_ref::kRankSyncTrack, num_tracks, max_records,
        kArriveOffset + 1, fast_instrument_rank_sync_ref::kRankArriveEvent);

    // Cross-rank barrier: every rank must observe every rank's arrive token.
    fast_instrument_ref::interval_start(
        profile, header_elems, blockIdx.x,
        fast_instrument_rank_sync_ref::kRankSyncTrack, num_tracks, max_records,
        kBarrierOffset, fast_instrument_rank_sync_ref::kRankBarrierEvent);
    fast_instrument_rank_sync_ref::wait_for_all_ranks(
        rank_arrive, epoch, num_ranks);
    fast_instrument_ref::interval_end(
        profile, header_elems, blockIdx.x,
        fast_instrument_rank_sync_ref::kRankSyncTrack, num_tracks, max_records,
        kBarrierOffset + 1, fast_instrument_rank_sync_ref::kRankBarrierEvent);

    // rank_release: this rank publishes that it passed the barrier.  The
    // release array is deliberately separate so consumers can distinguish
    // arrival from post-barrier visibility.
    fast_instrument_ref::interval_start(
        profile, header_elems, blockIdx.x,
        fast_instrument_rank_sync_ref::kRankSyncTrack, num_tracks, max_records,
        kReleaseOffset, fast_instrument_rank_sync_ref::kRankReleaseEvent);
    rank_release[rank_id] = epoch;
    __threadfence_system();
    fast_instrument_ref::interval_end(
        profile, header_elems, blockIdx.x,
        fast_instrument_rank_sync_ref::kRankSyncTrack, num_tracks, max_records,
        kReleaseOffset + 1, fast_instrument_rank_sync_ref::kRankReleaseEvent);
    visible_epoch[rank_id] = epoch;
  }

  // Make the rank operation a CTA-visible phase as well.  The barrier is
  // outside the profile guards and therefore executes in every configuration.
  __syncthreads();
}
