"""Serialize generic stage, handoff and hardware tracks as Perfetto TrackEvents.

Only the public TrackEvent protobuf fields used below are needed on the GPU host.
This also avoids loading PyTorch's sometimes incompatible protobuf installation.
"""
import gzip
import json
import os
from pathlib import Path
import struct
import uuid


def _varint(value):
    value = int(value) & ((1 << 64) - 1)
    buf = bytearray()
    while value >= 128:
        buf.append((value & 127) | 128)
        value >>= 7
    buf.append(value)
    return bytes(buf)


def _int(field, value):
    return _varint(field << 3) + _varint(value)


def _bytes(field, value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return _varint(field << 3 | 2) + _varint(len(value)) + value


def _double(field, value):
    return _varint(field << 3 | 1) + struct.pack("<d", value)


def _annotation(name, value):
    if isinstance(value, bool):
        payload = _int(2, value)
    elif isinstance(value, int):
        payload = _int(4, value)
    elif isinstance(value, float):
        payload = _double(5, value)
    else:
        payload = _bytes(6, str(value))
    return _bytes(4, _bytes(10, name) + payload)


def write_proto(trace, stream):
    events = trace["traceEvents"]
    labels = {(e["pid"], e.get("tid") if e["name"] == "thread_name" else None): e["args"]["name"]
              for e in events if e["ph"] == "M" and e["name"] in ("process_name", "thread_name")}
    tracks, busy, packets = {}, {}, []

    def track(key, label, parent=None, counter=False):
        if key not in tracks:
            uid = len(tracks) + 1
            desc = _int(1, uid) + _bytes(2, label)
            if parent:
                desc += _int(5, parent)
            if counter:
                desc += _bytes(8, b"")
            stream.write(_bytes(1, _int(10, 1) + _bytes(60, desc)))
            tracks[key] = uid
        return tracks[key]

    for e in sorted((e for e in events if "ts" in e), key=lambda e: e["ts"]):
        pid = e.get("pid", 0)
        tid = e.get("tid", 0)
        name = str(e["name"])
        timestamp = e.get("ts_ns", round(e["ts"] * 1000))
        parent = track(("pid", pid), labels.get((pid, None), str(pid)))
        args = e.get("args", {})
        ph = e["ph"]
        if ph == "C":
            for key, val in args.items():
                uid = track(("C", pid, tid, name, key), name if key == "value" else f"{name} {key}", parent, True)
                packets.append((timestamp, _int(9, 4) + _int(11, uid) + _double(44, val)))
            continue
        if ph not in ("X", "i", "I"):
            continue
        # Concurrent handoffs may overlap on a single consumer. Independent
        # subtracks preserve both intervals without inventing a nesting order.
        ends = busy.setdefault((pid, tid), [])
        sub = next((i for i, end in enumerate(ends) if end <= timestamp), len(ends))
        label = labels.get((pid, tid), f"Track {tid}")
        uid = track(("sw", pid, tid, sub), label if not sub else f"{label} [{sub}]", parent)
        payload = _int(11, uid) + _bytes(23, name) + _bytes(22, str(e.get("cat", "")))
        payload += b"".join(_annotation(str(k), v) for k, v in args.items() if v is not None)
        packets.append((timestamp, _int(9, 1 if ph == "X" else 3) + payload))
        if ph == "X":
            end = timestamp + round(e["dur"] * 1000)
            if end < timestamp:
                raise ValueError(f"Negative duration on {name}")
            if sub == len(ends):
                ends.append(end)
            else:
                ends[sub] = end
            packets.append((end, _int(9, 2) + _int(11, uid)))
    for timestamp, payload in sorted(packets, key=lambda p: p[0]):
        stream.write(_bytes(1, _int(8, timestamp) + _int(10, 1) + _bytes(11, payload)))


def export(trace, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    if path.suffix == ".gz":
        with gzip.open(pending, "wt", encoding="utf-8") as stream:
            json.dump(trace, stream, ensure_ascii=False, allow_nan=False)
    else:
        with pending.open("wb") as stream:
            write_proto(trace, stream)
    os.replace(pending, path)
