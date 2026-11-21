#pragma once

#include <cstdint>

namespace fast_instrument_ref {

constexpr std::uint32_t kCaptureSlotDisabled = 0xffffffffu;

__device__ __forceinline__ std::uint32_t physical_smid() {
  std::uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

// CTA leader calls this once and publishes the slot to the complete CTA.
// Only CTAs running on physical SM 0 or SM 1 reserve compact storage.
__device__ __forceinline__ std::uint32_t reserve_sm01_capture_slot(
    std::uint32_t* next_slot, std::uint32_t* overflow,
    std::uint32_t capture_capacity) {
  if (physical_smid() > 1u) return kCaptureSlotDisabled;
  const std::uint32_t slot = atomicAdd(next_slot, 1u);
  if (slot >= capture_capacity) {
    atomicExch(overflow, 1u);
    return kCaptureSlotDisabled;
  }
  return slot;
}

// The caller reserves offsets per (CTA, track).  No atomic allocation is
// performed here: each record occupies two consecutive int64 elements
// (timestamp, tag), at an ABI-stable address.
__device__ __forceinline__ std::uint64_t globaltimer_ns() {
  std::uint64_t timestamp;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp) :: "memory");
  return timestamp;
}

__device__ __forceinline__ std::uint64_t record_element_index(
    std::uint32_t header_elems, std::uint32_t cta, std::uint32_t track,
    std::uint32_t num_tracks, std::uint32_t max_records,
    std::uint32_t offset) {
  const std::uint64_t slot =
      (static_cast<std::uint64_t>(cta) * num_tracks * max_records +
       static_cast<std::uint64_t>(track) * max_records + offset) * 2ull;
  return static_cast<std::uint64_t>(header_elems) + slot;
}

__device__ __forceinline__ void record(
    std::int64_t* profile, std::uint32_t header_elems, std::uint32_t cta,
    std::uint32_t track, std::uint32_t num_tracks, std::uint32_t max_records,
    std::uint32_t offset, std::uint32_t task_id, bool is_start) {
  if (profile == nullptr || offset >= max_records) return;

  const std::uint64_t index = record_element_index(
      header_elems, cta, track, num_tracks, max_records, offset);
  // ABI tag fields: block[31:20], track[19:17], task[8:4], is_start[3].
  const std::uint32_t tag = (cta << 20) | (track << 17) |
                            (task_id << 4) | static_cast<std::uint32_t>(is_start << 3);
  profile[index] = static_cast<std::int64_t>(globaltimer_ns());
  profile[index + 1] = static_cast<std::int64_t>(tag);
}

// capture_slot selects compact storage; logical_cta remains in the event tag.
__device__ __forceinline__ void record_compact(
    std::int64_t* profile, std::uint32_t header_elems,
    std::uint32_t capture_slot, std::uint32_t logical_cta,
    std::uint32_t track, std::uint32_t num_tracks,
    std::uint32_t max_records, std::uint32_t offset,
    std::uint32_t task_id, bool is_start) {
  if (profile == nullptr || capture_slot == kCaptureSlotDisabled ||
      offset >= max_records) return;
  const std::uint64_t index = record_element_index(
      header_elems, capture_slot, track, num_tracks, max_records, offset);
  const std::uint32_t tag =
      (logical_cta << 20) | (track << 17) | (task_id << 4) |
      static_cast<std::uint32_t>(is_start << 3);
  profile[index] = static_cast<std::int64_t>(globaltimer_ns());
  profile[index + 1] = static_cast<std::int64_t>(tag);
}

// One interval consumes offset and offset + 1 in the selected track.
__device__ __forceinline__ void interval_start(
    std::int64_t* profile, std::uint32_t header_elems, std::uint32_t cta,
    std::uint32_t track, std::uint32_t num_tracks, std::uint32_t max_records,
    std::uint32_t offset, std::uint32_t task_id) {
  record(profile, header_elems, cta, track, num_tracks, max_records, offset,
         task_id, true);
}

__device__ __forceinline__ void interval_end(
    std::int64_t* profile, std::uint32_t header_elems, std::uint32_t cta,
    std::uint32_t track, std::uint32_t num_tracks, std::uint32_t max_records,
    std::uint32_t offset, std::uint32_t task_id) {
  record(profile, header_elems, cta, track, num_tracks, max_records, offset,
         task_id, false);
}

}  // namespace fast_instrument_ref
