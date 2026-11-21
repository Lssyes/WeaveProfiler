#!/usr/bin/env python3
"""decode_profile.py (E16) — ABI + raw buffer + sidecar -> paired interval events + diagnostics.

FULLY IMPLEMENTED generic logic, adapted from knowledge/examples/host/decode.py. Do not
rewrite for a normal project: the record layout, start/end pairing, per-track
scan/stop, tag consistency checks, anchor extraction and the strict diagnostic
counters are all kernel-agnostic and are exactly what the unified gates inspect.

The ONLY project-specific input is the ABI JSON in ../device_record/abi/, which
you regenerate with your project's extract_abi.py. No code hook is needed here.

CLI (matches the contract decoder invocation):
  python3 -B decode_profile.py --raw raw_rank0.bin --abi <abi.json> --out events_rank0.json
"""
from __future__ import annotations

import argparse
import array
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROFILE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROFILE_ROOT / "device_record" / "abi"))
from abi_loader import ABIError, check_abi_provenance, eval_expr, load_abi  # noqa: E402

DEFAULT_ABI = _PROFILE_ROOT / "device_record" / "abi" / "record_abi.example.json"
MASK64 = (1 << 64) - 1
SAMPLE_LIMIT = 20

SIDECAR_KEYS = {
    "max_records": ("max_records", "profiler_max_records", "records_per_track"),
    "num_sms": ("num_sms", "num_ctas", "grid_size"),
    "num_tracks": ("num_tracks", "tracks"),
    "rank": ("rank", "ep_rank", "local_rank"),
    "device": ("device", "device_index", "gpu", "cuda_device"),
}

COUNTER_KINDS = (
    "unmatched_starts", "unmatched_ends", "negative_durations",
    "zero_timestamp_with_nonzero_tag", "tag_slot_mismatch", "tag_high_bits_set",
    "unknown_task_ids", "expert_idx_mismatch", "stray_slots_after_gap",
    "timestamp_not_monotonic", "abi_source_sha256_mismatch",
)


class DecodeError(RuntimeError):
    pass


def load_npy(path: Path) -> Tuple[array.array, Dict[str, Any]]:
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise DecodeError("{} is not a .npy file".format(path))
        major = handle.read(1)[0]
        handle.read(1)
        if major == 1:
            header_len = int.from_bytes(handle.read(2), "little")
        elif major in (2, 3):
            header_len = int.from_bytes(handle.read(4), "little")
        else:
            raise DecodeError("unsupported .npy major version {}".format(major))
        header = ast.literal_eval(handle.read(header_len).decode("latin1"))
        descr = str(header["descr"])
        shape = tuple(header["shape"])
        if descr not in ("<i8", "|i8", "=i8", ">i8", "<u8", ">u8", "int64", "i8", "u8"):
            raise DecodeError("{} dtype {!r}; profiler buffer must be int64".format(path, descr))
        count = 1
        for dim in shape:
            count *= int(dim)
        payload = handle.read(count * 8)
        if len(payload) != count * 8:
            raise DecodeError("{} is truncated".format(path))
    values = array.array("q")
    values.frombytes(payload)
    if descr.startswith(">") != (sys.byteorder == "big"):
        values.byteswap()
    return values, {}


def load_bin(path: Path) -> Tuple[array.array, Dict[str, Any]]:
    payload = path.read_bytes()
    if len(payload) % 8:
        raise DecodeError("{} size {} is not a multiple of 8".format(path, len(payload)))
    values = array.array("q")
    values.frombytes(payload)
    if sys.byteorder == "big":
        values.byteswap()
    return values, {}


def load_pt(path: Path) -> Tuple[array.array, Dict[str, Any]]:
    try:
        import torch
    except ImportError:
        raise DecodeError("reading {} needs torch; save as .npy or raw int64 .bin".format(path))
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    sidecar: Dict[str, Any] = {}
    tensor = None
    if torch.is_tensor(obj):
        tensor = obj
    elif isinstance(obj, dict):
        sidecar = {k: v for k, v in obj.items() if isinstance(v, (int, float, str, bool))}
        for key in ("profiler_buf", "buffer", "buf", "raw", "records", "data"):
            if key in obj and torch.is_tensor(obj[key]):
                tensor = obj[key]
                break
        if tensor is None:
            for key in sorted(obj):
                if torch.is_tensor(obj[key]):
                    tensor = obj[key]
                    break
    if tensor is None:
        raise DecodeError("no tensor found inside {}".format(path))
    flat = tensor.detach().reshape(-1).to(torch.int64).cpu()
    return array.array("q", flat.tolist()), sidecar


def load_raw(path: Path) -> Tuple[array.array, Dict[str, Any]]:
    if not path.is_file():
        raise DecodeError("raw buffer not found: {}".format(path))
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return load_npy(path)
    if suffix in (".pt", ".pth"):
        return load_pt(path)
    if suffix in (".bin", ".raw", ".i64", ".int64", ""):
        return load_bin(path)
    raise DecodeError("unsupported raw extension {!r}; use .npy/.pt/.bin".format(suffix))


def normalize_sidecar(raw_meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for canonical, aliases in SIDECAR_KEYS.items():
        for alias in aliases:
            if alias in raw_meta and raw_meta[alias] is not None and not isinstance(raw_meta[alias], bool):
                value = raw_meta[alias]
                if isinstance(value, (int, float)):
                    out[canonical] = int(value)
                elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
                    out[canonical] = int(value.strip())
                else:
                    out[canonical] = value
                break
    return out


def find_sidecar(raw_path: Path, explicit: Optional[Path]) -> Tuple[Optional[Path], Dict[str, Any]]:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    else:
        candidates.extend([
            Path(str(raw_path) + ".json"),
            raw_path.with_suffix(".json"),
            raw_path.with_name(raw_path.stem + ".meta.json"),
            raw_path.with_name("profiler_meta.json"),
        ])
    for candidate in candidates:
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise DecodeError("sidecar {} must be a JSON object".format(candidate))
            return candidate, payload
    if explicit is not None:
        raise DecodeError("sidecar not found: {}".format(explicit))
    return None, {}


def _pick(*candidates: Any) -> Tuple[int, str]:
    labels = ("cli", "sidecar", "buffer_metadata", "abi_default")
    for index, candidate in enumerate(candidates):
        if candidate:
            return int(candidate), labels[index] if index < len(labels) else "fallback"
    return 0, "unavailable"


def _track_labels(tracks: List[Dict[str, Any]]) -> Dict[int, str]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for track in tracks:
        grouped[track.get("role", "unknown")].append(int(track["track_id"]))
    labels: Dict[int, str] = {}
    for role, ids in grouped.items():
        ordered = sorted(ids)
        for index, track_id in enumerate(ordered):
            labels[track_id] = role if len(ordered) == 1 else "{}_{}".format(role, index)
    return labels


class Diagnostics:
    def __init__(self) -> None:
        self.counters: Dict[str, int] = defaultdict(int)
        self.samples: Dict[str, List[Any]] = defaultdict(list)

    def note(self, kind: str, detail: Any) -> None:
        self.counters[kind] += 1
        if len(self.samples[kind]) < SAMPLE_LIMIT:
            self.samples[kind].append(detail)


def pair_records(parsed, block, track, tasks_by_id, tracks_by_id, labels, diag):
    track_meta = tracks_by_id.get(track, {})
    stacks: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    events: List[Dict[str, Any]] = []
    open_count = 0
    for record in parsed:
        if record["is_start"]:
            record["depth"] = open_count
            stacks[record["task_id"]].append(record)
            open_count += 1
            continue
        stack = stacks[record["task_id"]]
        if not stack:
            diag.note("unmatched_ends", {"block": block, "track": track, "offset": record["offset"],
                                         "task_id": record["task_id"], "timestamp_ns": record["timestamp"]})
            continue
        start = stack.pop()
        open_count -= 1
        expert_idx = start["expert_idx"]
        if expert_idx != record["expert_idx"]:
            diag.note("expert_idx_mismatch", {"block": block, "track": track,
                      "start_offset": start["offset"], "end_offset": record["offset"],
                      "task_id": record["task_id"], "start_expert_idx": start["expert_idx"],
                      "end_expert_idx": record["expert_idx"]})
            if expert_idx == 0:
                expert_idx = record["expert_idx"]
        duration = record["timestamp"] - start["timestamp"]
        if duration < 0:
            diag.note("negative_durations", {"block": block, "track": track,
                      "start_offset": start["offset"], "end_offset": record["offset"],
                      "task_id": record["task_id"], "start_ns": start["timestamp"],
                      "end_ns": record["timestamp"]})
        # An id the ABI does not describe is already recorded in unknown_task_ids (a strict
        # failure), but it must ALSO be non-renderable: giving it layer "task" would let
        # time whose meaning nobody knows count as operator coverage, and under
        # --allow-invalid that inflated number would be the one quoted. The name is made
        # visibly wrong so it cannot be mistaken for a real operator in summary_by_task.
        task = tasks_by_id.get(record["task_id"],
                               {"name": "UNKNOWN_task_id_{}".format(record["task_id"]),
                                "short_name": "unknown_{}".format(record["task_id"]),
                                "role": "unknown", "layer": "ignored", "parent_task_id": None})
        events.append({
            "block": block, "track": track, "role": track_meta.get("role", "unknown"),
            "track_label": labels.get(track, "track{}".format(track)),
            "task_id": record["task_id"], "task": task["name"], "short_name": task.get("short_name"),
            # No .get(..., "task") default: decode_buffer() has already refused any ABI whose
            # tasks do not declare an explicit layer, so this read must not re-introduce one.
            "task_role": task.get("role", "unknown"), "layer": task["layer"],
            "parent_task_id": task.get("parent_task_id"), "expert_idx": expert_idx,
            "start_ns": start["timestamp"], "end_ns": record["timestamp"], "dur_ns": duration,
            "start_offset": start["offset"], "end_offset": record["offset"], "depth": start["depth"],
        })
    for task_id, stack in stacks.items():
        for record in stack:
            diag.note("unmatched_starts", {"block": block, "track": track, "offset": record["offset"],
                                           "task_id": task_id, "timestamp_ns": record["timestamp"]})
    return events


def build_anchors(abi, events, num_sms):
    anchor_abi = abi.get("sync_anchors", {}).get("entry_rendezvous")
    if anchor_abi is None:
        return {"entry_rendezvous": None}
    task_name = anchor_abi["task"]
    per_cta: Dict[int, Dict[str, Any]] = {}
    for event in events:
        if event["task"] != task_name:
            continue
        current = per_cta.setdefault(event["block"], {"block": event["block"],
                  "entry_sync_start_ns": event["start_ns"], "entry_sync_end_ns": event["end_ns"],
                  "dur_ns": event["dur_ns"], "occurrences": 0})
        current["occurrences"] += 1
        current["entry_sync_start_ns"] = min(current["entry_sync_start_ns"], event["start_ns"])
        current["entry_sync_end_ns"] = max(current["entry_sync_end_ns"], event["end_ns"])
        current["dur_ns"] = current["entry_sync_end_ns"] - current["entry_sync_start_ns"]
    ordered = [per_cta[b] for b in sorted(per_cta)]
    result: Dict[str, Any] = {"entry_rendezvous": {
        "task": task_name, "task_id": anchor_abi.get("task_id"),
        "start_semantics": anchor_abi.get("start_semantics"),
        "end_semantics": anchor_abi.get("end_semantics"), "per_cta": ordered,
        "ctas_with_anchor": len(ordered),
        "ctas_missing_anchor_total": sum(1 for b in range(num_sms) if b not in per_cta)}}
    if ordered:
        starts = [e["entry_sync_start_ns"] for e in ordered]
        ends = [e["entry_sync_end_ns"] for e in ordered]
        result["entry_rendezvous"]["global"] = {
            "entry_sync_start_ns": min(starts), "entry_sync_end_ns": max(ends),
            "min_start_ns": min(starts), "max_start_ns": max(starts),
            "min_end_ns": min(ends), "max_end_ns": max(ends),
            "start_skew_ns": max(starts) - min(starts), "end_skew_ns": max(ends) - min(ends),
            "rendezvous_span_ns": max(ends) - min(starts)}
    else:
        result["entry_rendezvous"]["global"] = None
    return result


def summarize(events):
    by_role: Dict[str, Dict[str, int]] = defaultdict(lambda: {"events": 0, "total_dur_ns": 0})
    by_task: Dict[Any, Dict[str, int]] = defaultdict(lambda: {"events": 0, "total_dur_ns": 0})
    for event in events:
        by_role[event["role"]]["events"] += 1
        by_role[event["role"]]["total_dur_ns"] += max(event["dur_ns"], 0)
        key = (event["task"], event["short_name"], event["task_role"])
        by_task[key]["events"] += 1
        by_task[key]["total_dur_ns"] += max(event["dur_ns"], 0)
    if events:
        min_start = min(e["start_ns"] for e in events)
        max_end = max(e["end_ns"] for e in events)
        makespan = max_end - min_start
    else:
        min_start = max_end = None
        makespan = 0
    return {
        "by_role": [{"role": r, **v} for r, v in sorted(by_role.items())],
        "by_task": [{"task": k[0], "short_name": k[1], "task_role": k[2], **v}
                    for k, v in sorted(by_task.items(), key=lambda i: -i[1]["total_dur_ns"])],
        "time_window": {"min_start_ns": min_start, "max_end_ns": max_end, "makespan_ns": makespan,
                        "sum_event_dur_ns": sum(v["total_dur_ns"] for v in by_role.values())},
    }


def decode_buffer(values, abi, options):
    """Core decode. `values` is a flat int64 sequence; `abi` is a loaded ABI dict."""
    buffer_abi = abi["buffer"]
    tag_bits = abi["tag_bitfields"]
    tag_value_mask = (1 << int(abi["tag_storage"]["value_bits"])) - 1
    elems_per_record = int(buffer_abi["elems_per_record"])
    metadata_fields = {f["name"]: f for f in abi["regions"]["metadata"]["fields"]}
    block_map_fields = {f["name"]: f for f in abi["regions"]["block_map"]["fields"]}
    tasks_by_id = {int(t["id"]): t for t in abi["tasks"]}
    tracks_by_id = {int(t["track_id"]): t for t in abi["tracks"]}
    labels = _track_labels(abi["tracks"])

    # WHY validated here, once, instead of defaulted per event: `layer` decides whether an
    # event is an operator that counts toward coverage (task), a sub-stage nested inside one
    # (phase), or an alignment-only record that must never be drawn (ignored: whole-core
    # entry rendezvous, grid barriers). The old code read task.get("layer", "task"), so an
    # ABI entry that forgot the field became a renderable operator. That is how a whole-core
    # anchor was once drawn as a full-width bar on every lane: coverage envelope 100%, 25
    # gates PASS, real operator coverage 44.9%. There is no safe default — only the project
    # knows which of its ids are operators.
    bad_layers = [(t.get("id"), t.get("name"), t.get("layer")) for t in abi["tasks"]
                  if t.get("layer") not in ("task", "phase", "ignored")]
    if bad_layers:
        raise DecodeError(
            "ABI tasks are missing a valid `layer` (id, name, layer): {}. Every entry must "
            "declare exactly one of: task = an operator whose duration counts as operator "
            "coverage; phase = a sub-stage nested inside its parent_task_id; ignored = kept "
            "in the decoded data for SW/HW and cross-rank alignment but never rendered and "
            "never counted (whole-kernel anchors, grid barriers). Fix "
            "device_record/abi/*.json — do not let it default to \"task\".".format(bad_layers))

    numel = len(values)
    if numel < 1:
        raise DecodeError("raw buffer is empty")
    header_word = values[0] & MASK64
    meta_num_sms = (header_word >> metadata_fields["num_sms"]["lsb"]) & metadata_fields["num_sms"]["mask"]
    meta_num_tracks = (header_word >> metadata_fields["num_tracks"]["lsb"]) & metadata_fields["num_tracks"]["mask"]

    num_sms, num_sms_source = _pick(options.get("num_sms"), options.get("sidecar_num_sms"), meta_num_sms or None)
    if not num_sms:
        raise DecodeError("num_sms is 0: buffer metadata never written (profiler disabled or wrong tensor)")
    num_tracks, num_tracks_source = _pick(options.get("num_tracks"), options.get("sidecar_num_tracks"),
                                          meta_num_tracks or None, int(buffer_abi["num_tracks"]))
    # WHY: _pick() returns 0/"unavailable" when nothing supplied a value. With num_tracks == 0
    # the record loop below iterates over no tracks at all, so the decode produces zero events
    # and diagnostic_summary.valid stays True — an empty trace that reports success. num_sms
    # is already guarded; num_tracks was not.
    if not num_tracks:
        raise DecodeError(
            "num_tracks resolved to 0 (sources tried: --num-tracks, sidecar, buffer metadata "
            "header, ABI buffer.num_tracks). Decoding would walk zero observer tracks and "
            "report a valid, empty trace. Set buffer.num_tracks in the ABI to the number of "
            "observer lanes your kernel records, or pass --num-tracks.")

    header_elems = eval_expr(buffer_abi["header_elems_expr"], {"num_sms": num_sms, "num_tracks": num_tracks})
    max_records = options.get("max_records") or options.get("sidecar_max_records")
    max_records_source = "cli" if options.get("max_records") else "sidecar"
    if not max_records:
        span = numel - header_elems
        per_track = num_tracks * elems_per_record
        if span <= 0 or num_sms * per_track <= 0:
            raise DecodeError("cannot infer max_records; pass --max-records")
        max_records = span // (num_sms * per_track)
        max_records_source = "inferred_from_numel"
    max_records = int(max_records)
    if max_records <= 0:
        raise DecodeError("max_records must be positive, got {}".format(max_records))

    env = {"num_sms": num_sms, "num_tracks": num_tracks, "max_records": max_records}
    required_numel = eval_expr(buffer_abi["numel_formula"], env)
    if numel < required_numel:
        raise DecodeError("raw buffer has {} elems but layout needs {} (num_sms={} num_tracks={} max_records={})".format(
            numel, required_numel, num_sms, num_tracks, max_records))

    layout = abi["regions"]["records"]["layout"]
    block_stride = eval_expr(layout["block_stride_expr"], env)
    track_stride = eval_expr(layout["track_stride_expr"], env)
    record_stride = eval_expr(layout["record_stride_expr"], env)

    block_map = []
    for block in range(num_sms):
        word = values[1 + block] & MASK64
        block_map.append({
            "logical_cta": int((word >> block_map_fields["logical_cta"]["lsb"]) & block_map_fields["logical_cta"]["mask"]),
            "physical_smid": int((word >> block_map_fields["physical_smid"]["lsb"]) & block_map_fields["physical_smid"]["mask"]),
            "written": word != 0})

    diag = Diagnostics()
    provenance = check_abi_provenance(abi, options["abi_path"])
    for role, entry in provenance.items():
        if entry.get("status") == "mismatch":
            diag.note("abi_source_sha256_mismatch", {"role": role, **entry})

    events: List[Dict[str, Any]] = []
    record_counts = []
    truncated = []
    records_per_track: Dict[int, int] = defaultdict(int)
    total_records = 0

    for block in range(num_sms):
        for track in range(num_tracks):
            base = header_elems + block * block_stride + track * track_stride
            parsed = []
            written = 0
            stop_offset = max_records
            previous_ts = None
            for offset in range(max_records):
                index = base + offset * record_stride
                timestamp = values[index] & MASK64
                tag_word = values[index + 1] & MASK64
                if timestamp == 0 and tag_word == 0:
                    stop_offset = offset
                    break
                written += 1
                if previous_ts is not None and timestamp < previous_ts:
                    diag.note("timestamp_not_monotonic", {"block": block, "track": track, "offset": offset,
                              "previous_timestamp": int(previous_ts), "timestamp": int(timestamp)})
                previous_ts = timestamp
                if tag_word >> int(abi["tag_storage"]["value_bits"]):
                    diag.note("tag_high_bits_set", {"block": block, "track": track, "offset": offset,
                              "tag_word": int(tag_word)})
                tag = tag_word & tag_value_mask
                fields = {name: (tag >> spec["lsb"]) & spec["mask"] for name, spec in tag_bits.items()}
                if timestamp == 0:
                    diag.note("zero_timestamp_with_nonzero_tag", {"block": block, "track": track,
                              "offset": offset, "tag": int(tag), "task_id": int(fields["task"])})
                    continue
                if fields["block"] != block or fields["track"] != track:
                    diag.note("tag_slot_mismatch", {"slot_block": block, "slot_track": track, "offset": offset,
                              "tag_block": int(fields["block"]), "tag_track": int(fields["track"])})
                task_id = int(fields["task"])
                if task_id not in tasks_by_id:
                    diag.note("unknown_task_ids", {"block": block, "track": track, "offset": offset, "task_id": task_id})
                parsed.append({"offset": offset, "timestamp": int(timestamp), "task_id": task_id,
                               "is_start": bool(fields["is_start"]), "expert_idx": int(fields["expert_idx"])})
            if stop_offset < max_records:
                tail_begin = base + stop_offset * record_stride
                tail_end = base + max_records * record_stride
                stray = sum(1 for v in values[tail_begin:tail_end] if v != 0)
                if stray:
                    diag.counters["stray_slots_after_gap"] += stray
                    if len(diag.samples["stray_slots_after_gap"]) < SAMPLE_LIMIT:
                        diag.samples["stray_slots_after_gap"].append(
                            {"block": block, "track": track, "first_gap_offset": stop_offset, "stray_elements": stray})
            total_records += written
            records_per_track[track] += written
            is_truncated = written >= max_records
            record_counts.append({"block": block, "track": track,
                                  "role": tracks_by_id.get(track, {}).get("role", "unknown"),
                                  "records": written, "truncated": is_truncated})
            if is_truncated:
                truncated.append({"block": block, "track": track, "records": written, "max_records": max_records})
            events.extend(pair_records(parsed, block, track, tasks_by_id, tracks_by_id, labels, diag))

    events.sort(key=lambda e: (e["start_ns"], e["block"], e["track"], e["task_id"]))
    empty_tracks = [t for t in range(num_tracks) if records_per_track[t] == 0]
    # WHY this is checked and why absence means "must not be empty": a track that recorded
    # nothing is an observer warp that never fired, i.e. a whole lane of the delivery is
    # missing — yet records_total == 2*events_total still holds and every counter stays 0,
    # so the decode looks perfect. Emptiness is legal ONLY where the ABI says so explicitly
    # (tracks[].may_be_empty = true, decided from the reference config). A track with no
    # may_be_empty key has not been thought about, so it fails rather than passing quietly.
    illegal_empty = [{"track": t, "role": tracks_by_id.get(t, {}).get("role", "unknown"),
                      "may_be_empty": tracks_by_id.get(t, {}).get("may_be_empty")}
                     for t in empty_tracks
                     if tracks_by_id.get(t, {}).get("may_be_empty") is not True]
    anchors = build_anchors(abi, events, num_sms)
    summary = summarize(events)

    strict_failures = []
    for kind in COUNTER_KINDS:
        if diag.counters.get(kind, 0):
            strict_failures.append("{}={}".format(kind, diag.counters[kind]))
    expected_records = 2 * len(events)
    if total_records != expected_records:
        strict_failures.append("records_total={} != 2*events_total={}".format(total_records, expected_records))
    if truncated:
        strict_failures.append("truncated_tracks={}".format(len(truncated)))
    if illegal_empty:
        strict_failures.append(
            "empty_tracks_not_declared_may_be_empty={} (recorded zero records; set "
            "tracks[].may_be_empty=true in the ABI only if that lane is legitimately silent "
            "in this config, otherwise the observer warp never fired)".format(
                [e["track"] for e in illegal_empty]))

    counters = {kind: int(diag.counters.get(kind, 0)) for kind in COUNTER_KINDS}
    samples = {kind: diag.samples[kind] for kind in COUNTER_KINDS if diag.samples.get(kind)}
    metadata = {
        "abi": {"schema": abi["schema"], "source_file": abi.get("source_file"),
                "source_sha256": abi.get("source_sha256")},
        "raw_path": options.get("raw_path"), "raw_numel": numel, "required_numel": required_numel,
        "header_elems": header_elems, "elems_per_record": elems_per_record,
        "num_sms": num_sms, "num_sms_source": num_sms_source,
        "num_tracks": num_tracks, "num_tracks_source": num_tracks_source,
        "max_records": max_records, "max_records_source": max_records_source,
        "rank": options.get("rank"), "device": options.get("device"),
        "sidecar_path": options.get("sidecar_path"),
        "events_total": len(events), "records_total": total_records, "records_expected": expected_records,
        "record_counts": record_counts, "truncated_tracks": truncated, "empty_tracks": empty_tracks,
        "empty_tracks_not_declared_may_be_empty": illegal_empty,
        "counters": counters, "samples": samples, "abi_provenance": provenance,
        "diagnostic_summary": {"valid": not strict_failures, "failures": strict_failures,
                               "allow_invalid": bool(options.get("allow_invalid"))},
        "time_window": summary["time_window"], "summary_by_role": summary["by_role"],
        "summary_by_task": summary["by_task"],
    }
    return {"metadata": metadata, "block_map": block_map, "anchors": anchors, "events": events}


def decode_file(raw_path, abi_path, out_path=None, sidecar=None, num_sms=None, num_tracks=None,
                max_records=None, rank=None, device=None, allow_invalid=False):
    abi = load_abi(Path(abi_path))
    values, container_meta = load_raw(Path(raw_path))
    sidecar_path, sidecar_payload = find_sidecar(Path(raw_path), sidecar)
    merged = dict(container_meta)
    merged.update(sidecar_payload)
    sc = normalize_sidecar(merged)
    options = {
        "raw_path": str(raw_path), "sidecar_path": str(sidecar_path) if sidecar_path else None,
        "max_records": max_records, "num_sms": num_sms, "num_tracks": num_tracks,
        "sidecar_max_records": sc.get("max_records"), "sidecar_num_sms": sc.get("num_sms"),
        "sidecar_num_tracks": sc.get("num_tracks"),
        "rank": rank if rank is not None else sc.get("rank"),
        "device": device if device is not None else sc.get("device"),
        "abi_path": Path(abi_path), "allow_invalid": allow_invalid,
    }
    decoded = decode_buffer(values, abi, options)
    # WHY not a default: capture_scope is a CLAIM about the run — "these events cover every
    # physical SM". Fabricating all_physical_sms/True here makes an unverified assertion that
    # downstream gates and the delivery text then quote: a subset capture (a few SMs
    # instrumented) would be published as if it were whole-device. The producer of the raw
    # buffer knows the answer; the decoder does not.
    scope = merged.get("capture_scope")
    if not isinstance(scope, dict):
        raise DecodeError(
            "sidecar {} carries no capture_scope object (got {!r}). It must state what the raw "
            "buffer actually covers, e.g. "
            "{{\"capture_scope\": \"all_physical_sms\", \"selected_physical_smids\": [], "
            "\"source_capture_was_all_sm\": true}} for a whole-device run, or "
            "{{\"capture_scope\": \"subset\", \"selected_physical_smids\": [0, 1, 2], "
            "\"source_capture_was_all_sm\": false}} when only some SMs were instrumented "
            "(mirror contract capture.software_sm_scope). run_profile.py writes this beside the "
            "raw buffer; assuming all_physical_sms would publish a subset capture as "
            "whole-device coverage.".format(options["sidecar_path"], scope))
    decoded["capture_scope"] = scope
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(decoded, indent=2) + "\n", encoding="utf-8")
    return decoded


def main(argv=None):
    parser = argparse.ArgumentParser(description="Decode a raw device profiler buffer into paired events + diagnostics.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--abi", type=Path, default=DEFAULT_ABI)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sidecar", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--num-sms", type=int, default=None)
    parser.add_argument("--num-tracks", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--allow-invalid", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        out = args.out if args.out is not None else args.raw.with_name(args.raw.stem + ".events.json")
        decoded = decode_file(args.raw, args.abi, out_path=out, sidecar=args.sidecar,
                              num_sms=args.num_sms, num_tracks=args.num_tracks, max_records=args.max_records,
                              rank=args.rank, device=args.device, allow_invalid=args.allow_invalid)
    except (DecodeError, ABIError) as error:
        print("[decode] {}".format(error), file=sys.stderr)
        return 2
    meta = decoded["metadata"]
    if not args.quiet:
        print("[decode] raw={} numel={} num_sms={} num_tracks={} max_records={} events={} records={}".format(
            meta["raw_path"], meta["raw_numel"], meta["num_sms"], meta["num_tracks"],
            meta["max_records"], meta["events_total"], meta["records_total"]))
        print("[decode] anomalies: " + " ".join("{}={}".format(k, meta["counters"][k]) for k in COUNTER_KINDS))
        print("[decode] wrote {}".format(out))
    diagnostic = meta.get("diagnostic_summary", {})
    if not diagnostic.get("valid", True):
        print("[decode] FAIL strict diagnostics: {}".format("; ".join(diagnostic.get("failures", []))), file=sys.stderr)
        return 0 if args.allow_invalid else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
