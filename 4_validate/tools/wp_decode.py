# -*- coding: utf-8 -*-
"""WeaveProfiler (`wp`) canonical decoder: WPLITE02 / FILITE02 dump + names -> Perfetto trace + summary.

This is the first-class CLI and Python module for decoding WeaveProfiler recorder dumps.
The binary layout is identical for both magics (``WPLITE02`` canonical, ``FILITE02`` legacy):

    magic[8] | u32 n_slots, cap, warps_per_cta, overflow, timer, meta_len | meta JSON | u32 cnt[n_slots] | 16 B records

All decoding semantics (pairing, sub-lanes, clock64 anchors, AMD tick conversion, SM/CU grouping,
canonical semantic IDs) are delegated to ``fi_lite_decode`` so both spellings produce byte-identical
traces.

CLI::

    python -B wp_decode.py <dump.bin> --names names.json -o out.perfetto.json.gz [--summary s.json] ...
    python -B wp_decode.py --selftest

Python::

    import wp_decode
    trace, summary = wp_decode.decode_file("capture.wp02.bin", names)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
import struct
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import fi_lite_decode as _fi  # noqa: E402

WP_MAGIC = b"WPLITE02"
FI_MAGIC = b"FILITE02"
MAGICS = (WP_MAGIC, FI_MAGIC)
HEADER = struct.Struct("<6I")
RECORD = struct.Struct("<QII")

END, OPEN, MARK, CLK, CLOSE, ROLE_SWITCH, IDM = (
    _fi.END, _fi.OPEN, _fi.MARK, _fi.CLK, _fi.CLOSE, _fi.ROLE_SWITCH, _fi.IDM,
)
read_dump = _fi.read_dump
write_json = _fi.write_json
print_summary = _fi.print_summary


def decode(d: Mapping[str, Any], names: Mapping[str, Any], group_override=None, sm_bar=True, semantic_kb=None,
           require_semantics=False, cta_cu_map=None, cta_hw_map=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Decode a ``read_dump()`` result; returns ``(trace, summary)`` (summary gains ``magic``)."""
    trace, summary = _fi.decode(dict(d), dict(names), group_override=group_override, sm_bar=sm_bar,
                                semantic_kb=semantic_kb, require_semantics=require_semantics,
                                cta_cu_map=cta_cu_map, cta_hw_map=cta_hw_map)
    summary["magic"] = d.get("magic", FI_MAGIC.decode("ascii"))
    return trace, summary


def load_names(names: Mapping[str, Any] | str | os.PathLike) -> Dict[str, Any]:
    if isinstance(names, Mapping):
        return dict(names)
    with open(os.fspath(names), encoding="utf-8") as f:
        return json.load(f)


def decode_file(dump_path: str | os.PathLike, names: Mapping[str, Any] | str | os.PathLike, *,
                timer_hz: Optional[int] = None, platform: Optional[str] = None,
                group_override: Optional[str] = None, sm_bar: bool = True,
                require_semantics: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """One-call ``read_dump`` + ``decode`` with input lineage stamped into the summary."""
    path = os.fspath(dump_path)
    d = read_dump(path, timer_hz=timer_hz, platform=platform)
    trace, summary = decode(d, load_names(names), group_override=group_override, sm_bar=sm_bar,
                            require_semantics=require_semantics)
    summary["source_dump_path"] = os.path.abspath(path)
    summary["source_dump_sha256"] = _fi._file_sha256(path)
    return trace, summary


def build_dump_bytes(slots: Mapping[int, Any], *, n_slots: int, cap: int, wpc: int, timer: int = 0,
                     overflow: int = 0, meta: Mapping[str, Any] | bytes | None = None,
                     magic: bytes = WP_MAGIC) -> bytes:
    """Serialize ``{slot: [(t, tag, aux), ...]}`` into a WPLITE02/FILITE02 blob (tests, simulators)."""
    if magic not in MAGICS:
        raise ValueError(f"magic must be one of {MAGICS}, got {magic!r}")
    if isinstance(meta, (bytes, bytearray)):
        meta_b = bytes(meta)
    else:
        meta_b = json.dumps(dict(meta or {}), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    cnt = [min(len(slots.get(s, ())), cap) for s in range(n_slots)]
    parts = [magic, HEADER.pack(n_slots, cap, wpc, overflow, timer, len(meta_b)), meta_b,
             struct.pack(f"<{n_slots}I", *cnt)]
    for s in range(n_slots):
        for rec in list(slots.get(s, ()))[:cap]:
            parts.append(RECORD.pack(int(rec[0]), int(rec[1]) & 0xFFFFFFFF, int(rec[2]) & 0xFFFFFFFF))
    return b"".join(parts)


def _decode_blob(blob: bytes, names: Mapping[str, Any]):
    fd, path = tempfile.mkstemp(suffix=".wp02.bin")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        return decode(read_dump(path), names)
    finally:
        os.remove(path)


def selftest() -> bool:
    """CPU self-test: legacy fi_lite_decode selftest + WPLITE02/FILITE02 byte-identical decode."""
    _fi.selftest()
    kb_ids = {s["semantic_id"]: int(s["event_id"]) for s in _fi.load_kb()["semantics"] if s.get("event_id") is not None}
    e_load, e_mma = kb_ids["load"], kb_ids["mma"]
    names = {"events": {str(e_load): "load", str(e_mma): "mma"}, "roles": {"0": "producer", "1": "consumer"}}
    slots = {
        0: [(1000, OPEN | 0, 3), (2000, e_load, 0), (6000, END | e_load, 0), (7000, CLOSE, 0)],
        1: [(1100, OPEN | 1, 3), (6100, e_mma, 0), (9100, END | e_mma, 0), (9500, CLOSE, 0)],
    }
    meta = {"selftest": "wp_decode", "arch": "sm_90"}
    results = {}
    for magic in MAGICS:
        blob = build_dump_bytes(slots, n_slots=2, cap=8, wpc=2, meta=meta, magic=magic)
        trace, summary = _decode_blob(blob, names)
        assert summary["magic"] == magic.decode("ascii"), summary["magic"]
        assert not summary["errors"], summary["errors"]
        assert summary["events"]["load"]["n"] == 1 and abs(summary["events"]["load"]["total_us"] - 4.0) < 1e-9
        assert summary["events"]["mma"]["n"] == 1 and abs(summary["events"]["mma"]["total_us"] - 3.0) < 1e-9
        assert summary["overflow"] == 0 and summary["cta_sm"] == {"0": 3}
        results[magic] = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
    assert results[WP_MAGIC] == results[FI_MAGIC], "WPLITE02 and FILITE02 must decode identically"
    try:
        _decode_blob(b"BADMAGIC" + bytes(24), names)
    except SystemExit:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("bad magic must be rejected")
    print("WP_DECODE SELFTEST PASS")
    return True


def main() -> None:
    """Canonical CLI; argument surface identical to fi_lite_decode (``--selftest`` adds the WPLITE02 check)."""
    if "--selftest" in sys.argv[1:]:
        selftest()
        return
    _fi.main()


__all__ = ["WP_MAGIC", "FI_MAGIC", "MAGICS", "read_dump", "decode", "decode_file", "load_names", "write_json",
           "print_summary", "build_dump_bytes", "selftest", "main"]

if __name__ == "__main__":
    main()
