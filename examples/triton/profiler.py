"""Fixed-slot device profiler helpers for the Triton references.

``profile`` is an int64 buffer.  A record is ``(timestamp, tag)`` and its
element index is ``header_elems + (cta * num_tracks * max_records +
track * max_records + offset) * 2``.  The caller owns offsets; this helper
does not use atomic allocation or wrap around.
"""

import triton
import triton.language as tl
from triton.language.extra import cuda


@triton.jit
def record(profile, header_elems, cta, track, num_tracks, max_records, offset,
           task_id: tl.constexpr, is_start: tl.constexpr):
    if offset < max_records:
        timestamp = cuda.globaltimer()
        element = header_elems + (
            cta * num_tracks * max_records + track * max_records + offset
        ) * 2
        # ABI tag fields: block[31:20], track[19:17], task[8:4], is_start[3].
        tag = (cta << 20) | (track << 17) | (task_id << 4) | (is_start << 3)
        tl.store(profile + element, timestamp)
        tl.store(profile + element + 1, tl.cast(tag, tl.int64))


@triton.jit
def interval_start(profile, header_elems, cta, track, num_tracks, max_records,
                   offset, task_id: tl.constexpr):
    record(profile, header_elems, cta, track, num_tracks, max_records, offset,
           task_id, True)


@triton.jit
def interval_end(profile, header_elems, cta, track, num_tracks, max_records,
                 offset, task_id: tl.constexpr):
    record(profile, header_elems, cta, track, num_tracks, max_records, offset,
           task_id, False)

