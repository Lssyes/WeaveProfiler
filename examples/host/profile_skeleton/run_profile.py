#!/usr/bin/env python3
"""run_profile.py (E14) — runtime launch + software capture of the device profiler buffer.

Generic orchestration is implemented: it resolves grid / threads-per-CTA / ranks /
target device from the contract, hashes the device source for provenance (so the
decoded events can be tied back to the exact build), sizes and zero-inits the
int64 profiler buffer from the ABI, and writes raw_rank*.bin + a decoder sidecar +
a stream_manifest.json.

The actual kernel launch + buffer copy-back is a PROJECT HOOK: only your project
knows how to launch the instrumented kernel and read its profiler buffer back to
the host. `capture_raw_buffer(...)` raises NotImplementedError until you fill it,
so a half-copied skeleton fails loudly instead of silently producing empty traces.

CLI (matches the unified stage command):
  python3 -B run_profile.py --contract <lock> --mode sw-capture --out sw/stream_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROFILE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROFILE_ROOT / "device_record" / "abi"))
from abi_loader import eval_expr, load_abi, sha256_of  # noqa: E402

DEFAULT_ABI = _PROFILE_ROOT / "device_record" / "abi" / "record_abi.example.json"


def _get(contract: Dict[str, Any], *path: str, default: Any = None) -> Any:
    node: Any = contract
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def resolve_launch_geometry(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Generic: pull grid / block / ranks / device out of the contract."""
    build = contract.get("build_and_launch") or {}
    grid = build.get("grid")
    block = build.get("block") or build.get("threads_per_cta")
    ranks = _get(contract, "capture", "display_rank")
    gpu = _get(contract, "environment", "gpu", default={}) or {}
    run = contract.get("run") or {}
    world = run.get("world_size") or run.get("ranks")
    # WHY not `or 1`: run_sw_capture() loops over range(world_size), so a defaulted 1 captures
    # rank 0 only. On a multi-rank run that produces a complete-looking delivery that is
    # missing every other rank — and nothing downstream can tell, because the manifest simply
    # lists one rank. The contract knows the world size; make it say so.
    if not world:
        raise ValueError(
            "contract run.world_size / run.ranks is missing. Capturing would default to 1 rank "
            "and silently drop every other rank of a distributed run. Re-materialize the "
            "contract with --world-size <ranks> (use 1 explicitly for a single-GPU run).")
    return {"grid": grid, "block": block, "world_size": int(world),
            "display_rank": ranks, "device": gpu.get("index")}


def resolve_buffer_sizing(contract: Dict[str, Any], abi: Dict[str, Any],
                          num_sms: int, num_tracks: int, max_records: int) -> Dict[str, Any]:
    env = {"num_sms": int(num_sms), "num_tracks": int(num_tracks), "max_records": int(max_records)}
    numel = eval_expr(abi["buffer"]["numel_formula"], env)
    return {"num_sms": num_sms, "num_tracks": num_tracks, "max_records": max_records,
            "numel": numel, "element_dtype": abi["buffer"]["element_dtype"],
            "numel_formula": abi["buffer"]["numel_formula"],
            "zero_init_required": bool(abi["buffer"].get("zero_init_required", True))}


def device_source_sha256(abi: Dict[str, Any], abi_path: Path) -> Dict[str, Any]:
    """Provenance: hash the device source the ABI points at.

    WHY the misses are fatal: this hash is the only link between the decoded events and the
    exact instrumented source that produced them. Returning status="no_source_file_in_abi"
    or "source_missing" with sha256=None still let the capture run and still wrote a
    manifest, so a delivery could claim to describe a build nobody can identify — and a
    stale ABI pointing at a file that had since moved looked identical to a good run.
    """
    recorded = abi.get("source_file")
    if not recorded:
        raise ValueError(
            "ABI {} has no source_file: the device source that produced these records cannot "
            "be hashed, so the events cannot be tied back to a build. Set source_file (path to "
            "the instrumented kernel source) in the ABI — extract_abi.py writes it.".format(abi_path))
    if "REPLACE" in str(recorded).upper() or "PROJECT_HOOK" in str(recorded).upper():
        raise ValueError(
            "ABI {} source_file is still the template placeholder {!r}. Point it at YOUR "
            "instrumented device source before capturing.".format(abi_path, recorded))
    source_path = Path(recorded)
    if not source_path.is_absolute():
        source_path = abi_path.resolve().parent / source_path
    if not source_path.is_file():
        raise ValueError(
            "ABI {} points at device source {} which does not exist. Either the ABI is stale "
            "(re-run extract_abi.py after moving/renaming the source) or the path is relative "
            "to the wrong directory — it is resolved against the ABI's own directory. Capturing "
            "without this hash would produce events with no traceable build."
            .format(abi_path, source_path))
    return {"source_file": str(source_path), "sha256": sha256_of(source_path), "status": "ok"}


def resolve_capture_scope(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Derive the sidecar capture_scope from the contract. Never assume whole-device.

    WHY: capture_scope is a provenance CLAIM that the decoded events cover every physical
    SM. The old code hardcoded {"capture_scope": "all_physical_sms",
    "selected_physical_smids": [], "source_capture_was_all_sm": True} regardless of the
    contract, so a deliberate subset capture (a handful of instrumented SMs) was published
    as if it were the whole device, and gate G13 compared that fabricated "measured" set
    against the declaration it was copied from — self-confirming and always green.

    The contract is the single declaration. Both spellings are accepted: the current
    materialize_contract.py and configs/validation/contract_lock_example.json write
    mode/selected_physical_smids (the only spelling gate G13 reads), while contracts frozen
    by older generators wrote kind/smids -- those must still resolve rather than silently
    fall through to a whole-device claim.
    """
    scope = _get(contract, "capture", "software_sm_scope")
    if not isinstance(scope, dict) or not scope:
        raise ValueError(
            "contract capture.software_sm_scope is missing; it declares which physical SMs "
            "the software capture covers and the sidecar must mirror it. Re-materialize the "
            "contract with --software-sm-scope all_sms (whole device) or "
            "--software-sm-scope subset:<smid,smid,...>. Defaulting to all_physical_sms would "
            "write an unverified whole-device claim into every sidecar.")
    mode = str(scope.get("mode") or scope.get("kind") or "").strip().lower()
    smids = scope.get("selected_physical_smids")
    if smids is None:
        smids = scope.get("smids")
    if mode in ("all", "all_sms", "all_physical_sms"):
        return {"capture_scope": "all_physical_sms", "selected_physical_smids": [],
                "source_capture_was_all_sm": True}
    if mode in ("subset", "selected"):
        if not isinstance(smids, list) or not smids:
            raise ValueError(
                "contract capture.software_sm_scope declares mode={!r} but lists no SM ids. A "
                "subset capture must name the physical SMs it instrumented (gate G13 compares "
                "the decoded set against this declaration); an empty list would be read as "
                "'whole device'.".format(mode))
        return {"capture_scope": "subset", "selected_physical_smids": [int(s) for s in smids],
                "source_capture_was_all_sm": False}
    raise ValueError(
        "contract capture.software_sm_scope has an unrecognised mode/kind {!r} (expected one "
        "of all / all_sms / all_physical_sms / subset / selected). Say explicitly what the "
        "capture covered — this string ends up in the delivery as the coverage claim."
        .format(mode))


def write_sidecar(path: Path, rank: int, device: Optional[int], sizing: Dict[str, Any],
                  capture_scope: Dict[str, Any]) -> None:
    sidecar = {"rank": rank, "device": device, "num_sms": sizing["num_sms"],
               "num_tracks": sizing["num_tracks"], "max_records": sizing["max_records"],
               "capture_scope": capture_scope}
    path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")


def capture_raw_buffer(rank: int, device: Optional[int], geometry: Dict[str, Any],
                       sizing: Dict[str, Any], contract: Dict[str, Any]) -> "Any":
    """Launch the instrumented kernel for one rank and return its profiler buffer.

    Return a flat host int64 array/ndarray of length sizing["numel"] (the buffer the
    device wrote). The generic pipeline handles everything else.
    """
    # ===== PROJECT HOOK: launch the instrumented kernel + copy back the buffer =====
    raise NotImplementedError(
        "PROJECT HOOK: implement capture_raw_buffer() for rank {}. Steps your project must do:\n"
        "  1. allocate a zero-inited int64 device tensor of numel={} (buffer.zero_init_required={});\n"
        "  2. build the '{}' compile preset (device_record/presets/) and launch the target kernel\n"
        "     with grid={} block={} on device={}, wrapped in cudaProfilerStart()/Stop() for HW capture;\n"
        "  3. copy the profiler buffer back to host and return it as a flat int64 sequence.\n"
        "Only your project knows the exact kernel entry + argument marshalling.".format(
            rank, sizing["numel"], sizing["zero_init_required"],
            _get(contract, "profile_selection", "preset", default="full"),
            geometry.get("grid"), geometry.get("block"), device))


def save_raw(path: Path, buffer: "Any") -> int:
    """Generic: persist a flat int64 buffer as little-endian .bin. Accepts ndarray/list/array."""
    import array
    try:
        import numpy as np  # optional fast path
        if isinstance(buffer, np.ndarray):
            data = buffer.astype("<i8", copy=False).tobytes()
            path.write_bytes(data)
            return len(data)
    except ImportError:
        pass
    arr = array.array("q", [int(x) for x in buffer])
    if sys.byteorder == "big":
        arr.byteswap()
    data = arr.tobytes()
    path.write_bytes(data)
    return len(data)


def run_sw_capture(contract: Dict[str, Any], contract_path: Path, abi_path: Path,
                   out_path: Path, num_sms: Optional[int], num_tracks: Optional[int],
                   max_records: Optional[int]) -> int:
    abi = load_abi(abi_path)
    geometry = resolve_launch_geometry(contract)
    provenance = device_source_sha256(abi, abi_path)

    buf_abi = contract.get("buffer_abi") or {}
    num_sms = num_sms or buf_abi.get("num_sms") or geometry.get("grid")
    if isinstance(num_sms, (list, tuple)):
        num_sms = int(num_sms[0]) if num_sms else None
    # WHY these two are fatal instead of `or 1` / `or 4096`: they size the record buffer.
    # num_sms = 1 allocates one CTA's worth of slots, so every other CTA either records
    # nothing or writes past its region — the decode then shows a single lane and still
    # reports success. max_records = 4096 is an invented depth: too small and tracks come
    # back truncated (events silently missing from the middle of the timeline), too large
    # and the allocation dwarfs the kernel it measures. Neither value can be guessed here.
    if not num_sms:
        raise ValueError(
            "cannot size the profiler buffer: num_sms is unset (tried --num-sms, contract "
            "buffer_abi.num_sms, contract build_and_launch.grid). Use the launch grid, i.e. "
            "the number of CTAs your kernel launches (for a persistent kernel that is the SM "
            "count you launch on, e.g. 78 on H20 — NOT 1). Re-materialize the contract with "
            "--num-sms or pass it on the command line.")
    num_tracks = num_tracks or int(abi["buffer"].get("num_tracks", len(abi["tracks"])))
    max_records = max_records or buf_abi.get("max_records")
    if not max_records:
        raise ValueError(
            "cannot size the profiler buffer: max_records is unset (tried --max-records and "
            "contract buffer_abi.max_records). Choose it from the recording plan: 2 records "
            "(begin+end) per event, times the maximum events one observer track writes in a "
            "single launch, plus headroom — then confirm decode_profile.py reports "
            "truncated_tracks == []. A guessed 4096 either truncates the middle of the "
            "timeline or allocates far more than the kernel needs.")
    sizing = resolve_buffer_sizing(contract, abi, int(num_sms), int(num_tracks), int(max_records))

    world = geometry["world_size"]
    ranks = list(range(world))
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_scope = resolve_capture_scope(contract)

    raw_entries: List[Dict[str, Any]] = []
    for rank in ranks:
        buffer = capture_raw_buffer(rank, geometry.get("device"), geometry, sizing, contract)
        raw_path = out_dir / "raw_rank{}.bin".format(rank)
        sidecar_path = out_dir / "raw_rank{}.bin.json".format(rank)
        nbytes = save_raw(raw_path, buffer)
        write_sidecar(sidecar_path, rank, geometry.get("device"), sizing, capture_scope)
        raw_entries.append({"rank": rank, "raw": str(raw_path), "sidecar": str(sidecar_path),
                            "bytes": nbytes})

    manifest = {"schema": "sw-capture-manifest/v1", "mode": "sw-capture",
                "contract": str(contract_path), "abi": str(abi_path),
                "device_source_provenance": provenance, "geometry": geometry,
                "buffer_sizing": sizing, "capture_scope": capture_scope,
                "ranks": raw_entries}
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("[run_profile] wrote {} ranks={}".format(out_path, len(raw_entries)))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Launch the instrumented kernel and capture the profiler buffer.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--mode", choices=("sw-capture",), default="sw-capture")
    parser.add_argument("--abi", type=Path, default=DEFAULT_ABI)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-sms", type=int, default=None)
    parser.add_argument("--num-tracks", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args(argv)

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    abi_path = args.abi
    contract_abi = _get(contract, "profile_selection", "abi")
    if contract_abi:
        candidate = Path(contract_abi)
        if not candidate.is_absolute():
            candidate = args.contract.resolve().parent / candidate
        if candidate.is_file():
            abi_path = candidate
    return run_sw_capture(contract, args.contract, abi_path, args.out,
                          args.num_sms, args.num_tracks, args.max_records)


if __name__ == "__main__":
    sys.exit(main())
