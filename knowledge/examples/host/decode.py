#!/usr/bin/env python3
import argparse
import array
import ast
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_ABI = WORKSPACE / "profile/abi/megamoe_profiler_abi.json"
ABI_SCHEMA = "megamoe.profiler-abi/v1"
MASK64 = (1 << 64) - 1
SAMPLE_LIMIT = 20
EXPR_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Div: lambda a, b: a // b,
}
SIDECAR_KEYS = {
    "max_records": ("max_records", "profiler_max_records", "records_per_track"),
    "num_sms": ("num_sms", "num_ctas", "grid_size"),
    "num_tracks": ("num_tracks", "tracks"),
    "rank": ("rank", "ep_rank", "local_rank"),
    "device": ("device", "device_index", "gpu", "cuda_device"),
}


class DecodeError(RuntimeError):
    pass


def eval_expr(expr, env):
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in EXPR_BINOPS:
            return EXPR_BINOPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise DecodeError(
                    "ABI expression {!r} uses unknown symbol {!r}".format(expr, node.id)
                )
            return int(env[node.id])
        raise DecodeError(
            "ABI expression {!r} contains an unsupported construct".format(expr)
        )

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as error:
        raise DecodeError("cannot parse ABI expression {!r}: {}".format(expr, error))
    return int(walk(tree))


def load_npy(path):
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
            raise DecodeError(
                "{} has dtype {!r}; the profiler buffer must be int64".format(path, descr)
            )
        if header.get("fortran_order") and len(shape) > 1:
            raise DecodeError("fortran-ordered multi-dimensional .npy is not supported")
        count = 1
        for dim in shape:
            count *= int(dim)
        payload = handle.read(count * 8)
        if len(payload) != count * 8:
            raise DecodeError("{} is truncated".format(path))
    values = array.array("q")
    values.frombytes(payload)
    big_endian_file = descr.startswith(">")
    if big_endian_file != (sys.byteorder == "big"):
        values.byteswap()
    return values, {}


def load_bin(path):
    payload = path.read_bytes()
    if len(payload) % 8:
        raise DecodeError("{} size {} is not a multiple of 8".format(path, len(payload)))
    values = array.array("q")
    values.frombytes(payload)
    if sys.byteorder == "big":
        values.byteswap()
    return values, {}


def load_pt(path):
    try:
        import torch
    except ImportError:
        raise DecodeError(
            "reading {} needs torch; save the buffer as .npy "
            "(numpy) or a raw int64 .bin instead".format(path)
        )
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        print(
            "[decode] {} could not be loaded in weights-only mode; falling back to "
            "full unpickling, so only use raw buffers you produced yourself".format(path),
            file=sys.stderr,
        )
        obj = torch.load(path, map_location="cpu", weights_only=False)
    sidecar = {}
    tensor = None
    if torch.is_tensor(obj):
        tensor = obj
    elif isinstance(obj, dict):
        sidecar = {
            key: value
            for key, value in obj.items()
            if isinstance(value, (int, float, str, bool))
        }
        for key in ("profiler_buf", "buffer", "buf", "raw", "records", "data"):
            if key in obj and torch.is_tensor(obj[key]):
                tensor = obj[key]
                break
        if tensor is None:
            for key in sorted(obj):
                if torch.is_tensor(obj[key]):
                    tensor = obj[key]
                    break
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            if torch.is_tensor(item):
                tensor = item
                break
    if tensor is None:
        raise DecodeError("no tensor found inside {}".format(path))
    flat = tensor.detach().reshape(-1).to(torch.int64).cpu()
    values = array.array("q", flat.tolist())
    return values, sidecar


def load_raw(path):
    if not path.is_file():
        raise DecodeError("raw buffer not found: {}".format(path))
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return load_npy(path)
    if suffix in (".pt", ".pth"):
        return load_pt(path)
    if suffix in (".bin", ".raw", ".i64", ".int64", ""):
        return load_bin(path)
    raise DecodeError(
        "unsupported raw extension {!r}; use .npy, .pt or a raw int64 .bin".format(suffix)
    )


def normalize_sidecar(raw_meta):
    out = {}
    for canonical, aliases in SIDECAR_KEYS.items():
        for alias in aliases:
            if alias in raw_meta and raw_meta[alias] is not None:
                value = raw_meta[alias]
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    out[canonical] = int(value)
                    break
                if isinstance(value, str) and value.strip().lstrip("-").isdigit():
                    out[canonical] = int(value.strip())
                    break
                out[canonical] = value
                break
    return out


def find_sidecar(raw_path, explicit):
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    else:
        candidates.extend(
            [
                Path(str(raw_path) + ".json"),
                raw_path.with_suffix(".json"),
                raw_path.with_name(raw_path.stem + ".meta.json"),
                raw_path.with_name("profiler_meta.json"),
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise DecodeError("sidecar {} is invalid JSON: {}".format(candidate, error))
            if not isinstance(payload, dict):
                raise DecodeError("sidecar {} must contain a JSON object".format(candidate))
            return candidate, payload
    if explicit is not None:
        raise DecodeError("sidecar not found: {}".format(explicit))
    return None, {}


def load_abi(path):
    if not path.is_file():
        raise DecodeError(
            "ABI file not found: {}; run python3 -B profile/scripts/extract_abi.py "
            "--write".format(path)
        )
    abi = json.loads(path.read_text(encoding="utf-8"))
    if abi.get("schema") != ABI_SCHEMA:
        raise DecodeError(
            "ABI schema {!r} is not {!r}".format(abi.get("schema"), ABI_SCHEMA)
        )
    for key in ("tag_bitfields", "tasks", "tracks", "buffer", "regions"):
        if key not in abi:
            raise DecodeError("ABI file is missing section {!r}".format(key))
    return abi


def track_labels(tracks):
    grouped = defaultdict(list)
    for track in tracks:
        grouped[track["role"]].append(track["track_id"])
    labels = {}
    for role, ids in grouped.items():
        ordered = sorted(ids)
        for index, track_id in enumerate(ordered):
            if len(ordered) == 1:
                labels[track_id] = role
            elif role == "math":
                labels[track_id] = "{}_wg{}".format(role, index)
            else:
                labels[track_id] = "{}_{}".format(role, index)
    return labels


class Diagnostics:
    def __init__(self):
        self.counters = defaultdict(int)
        self.samples = defaultdict(list)

    def note(self, kind, detail):
        self.counters[kind] += 1
        if len(self.samples[kind]) < SAMPLE_LIMIT:
            self.samples[kind].append(detail)

    def as_dict(self, kinds):
        counters = {kind: int(self.counters.get(kind, 0)) for kind in kinds}
        samples = {
            kind: self.samples[kind] for kind in kinds if self.samples.get(kind)
        }
        return counters, samples


COUNTER_KINDS = (
    "unmatched_starts",
    "unmatched_ends",
    "negative_durations",
    "zero_timestamp_with_nonzero_tag",
    "tag_slot_mismatch",
    "tag_high_bits_set",
    "unknown_task_ids",
    "expert_idx_mismatch",
    "stray_slots_after_gap",
    "timestamp_not_monotonic",
    "abi_provenance_missing",
    "abi_source_sha256_mismatch",
)


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def check_abi_provenance(abi, abi_path, diag):
    """检查 ABI 记录的 device source、binding source、host source 是否仍一致。"""
    root = abi_path.resolve().parents[2]
    checks = (
        ("source_file", "source_sha256", "source"),
        ("binding_source_file", "binding_source_sha256", "binding"),
        ("host_source_file", "host_source_sha256", "host"),
    )
    result = {}
    for file_key, hash_key, role in checks:
        recorded_path = abi.get(file_key)
        recorded_hash = abi.get(hash_key)
        if not recorded_path or not recorded_hash:
            diag.note("abi_provenance_missing", {
                "role": role, "file_key": file_key, "hash_key": hash_key,
            })
            result[role] = {"status": "missing", "path": recorded_path}
            continue
        source_path = Path(recorded_path)
        if not source_path.is_absolute():
            source_path = root / source_path
        if not source_path.is_file():
            diag.note("abi_provenance_missing", {
                "role": role, "path": str(source_path),
                "reason": "source file does not exist",
            })
            result[role] = {"status": "missing", "path": str(source_path),
                            "recorded_sha256": recorded_hash}
            continue
        actual_hash = sha256_of(source_path)
        entry = {"path": str(source_path), "recorded_sha256": recorded_hash,
                 "actual_sha256": actual_hash}
        if actual_hash != recorded_hash:
            diag.note("abi_source_sha256_mismatch", {
                "role": role, "path": str(source_path),
                "recorded_sha256": recorded_hash, "actual_sha256": actual_hash,
            })
            entry["status"] = "mismatch"
        else:
            entry["status"] = "ok"
        result[role] = entry
    return result


def decode(values, abi, options):
    buffer_abi = abi["buffer"]
    tag_bits = abi["tag_bitfields"]
    elems_per_record = int(buffer_abi["elems_per_record"])
    tag_value_mask = (1 << int(abi["tag_storage"]["value_bits"])) - 1

    metadata_fields = {
        field["name"]: field for field in abi["regions"]["metadata"]["fields"]
    }
    block_map_fields = {
        field["name"]: field for field in abi["regions"]["block_map"]["fields"]
    }
    tasks_by_id = {int(task["id"]): task for task in abi["tasks"]}
    tracks_by_id = {int(track["track_id"]): track for track in abi["tracks"]}
    labels = track_labels(abi["tracks"])

    numel = len(values)
    if numel < 1:
        raise DecodeError("raw buffer is empty")
    header_word = values[0] & MASK64
    meta_num_sms = (header_word >> metadata_fields["num_sms"]["lsb"]) & metadata_fields[
        "num_sms"
    ]["mask"]
    meta_num_tracks = (
        header_word >> metadata_fields["num_tracks"]["lsb"]
    ) & metadata_fields["num_tracks"]["mask"]

    num_sms, num_sms_source = pick(
        options.get("num_sms"), options.get("sidecar_num_sms"), meta_num_sms or None
    )
    if not num_sms:
        raise DecodeError(
            "num_sms is 0: buffer metadata was never written (profiler disabled, "
            "buffer not copied back, or wrong tensor); pass --num-sms to force"
        )
    num_tracks, num_tracks_source = pick(
        options.get("num_tracks"),
        options.get("sidecar_num_tracks"),
        meta_num_tracks or None,
        int(buffer_abi["num_tracks"]),
    )

    header_elems = eval_expr(
        buffer_abi["header_elems_expr"], {"num_sms": num_sms, "num_tracks": num_tracks}
    )
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
        raise DecodeError(
            "raw buffer has {} elements but layout {} needs {} "
            "(num_sms={}, num_tracks={}, max_records={})".format(
                numel, buffer_abi["numel_formula"], required_numel, num_sms,
                num_tracks, max_records,
            )
        )

    layout = abi["regions"]["records"]["layout"]
    block_stride = eval_expr(layout["block_stride_expr"], env)
    track_stride = eval_expr(layout["track_stride_expr"], env)
    record_stride = eval_expr(layout["record_stride_expr"], env)

    block_map = []
    for block in range(num_sms):
        word = values[1 + block] & MASK64
        block_map.append(
            {
                "logical_cta": int(
                    (word >> block_map_fields["logical_cta"]["lsb"])
                    & block_map_fields["logical_cta"]["mask"]
                ),
                "physical_smid": int(
                    (word >> block_map_fields["physical_smid"]["lsb"])
                    & block_map_fields["physical_smid"]["mask"]
                ),
                "written": word != 0,
            }
        )

    diag = Diagnostics()
    abi_provenance = check_abi_provenance(abi, options["abi_path"], diag)
    events = []
    record_counts = []
    truncated = []
    empty_block_tracks = []
    records_per_track = defaultdict(int)
    blocks_with_records = defaultdict(int)
    total_records = 0

    for block in range(num_sms):
        for track in range(num_tracks):
            base = header_elems + block * block_stride + track * track_stride
            parsed = []
            written = 0
            stop_offset = max_records
            previous_timestamp = None
            for offset in range(max_records):
                index = base + offset * record_stride
                timestamp = values[index] & MASK64
                tag_word = values[index + 1] & MASK64
                if timestamp == 0 and tag_word == 0:
                    stop_offset = offset
                    break
                written += 1
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    diag.note(
                        "timestamp_not_monotonic",
                        {"block": block, "track": track, "offset": offset,
                         "previous_timestamp": int(previous_timestamp),
                         "timestamp": int(timestamp)},
                    )
                previous_timestamp = timestamp
                if tag_word >> abi["tag_storage"]["value_bits"]:
                    diag.note(
                        "tag_high_bits_set",
                        {"block": block, "track": track, "offset": offset,
                         "tag_word": int(tag_word)},
                    )
                tag = tag_word & tag_value_mask
                fields = {
                    name: (tag >> spec["lsb"]) & spec["mask"]
                    for name, spec in tag_bits.items()
                }
                if timestamp == 0:
                    diag.note(
                        "zero_timestamp_with_nonzero_tag",
                        {"block": block, "track": track, "offset": offset,
                         "tag": int(tag), "task_id": int(fields["task"]),
                         "is_start": bool(fields["is_start"])},
                    )
                    continue
                if fields["block"] != block or fields["track"] != track:
                    diag.note(
                        "tag_slot_mismatch",
                        {"slot_block": block, "slot_track": track, "offset": offset,
                         "tag_block": int(fields["block"]),
                         "tag_track": int(fields["track"]), "tag": int(tag)},
                    )
                task_id = int(fields["task"])
                if task_id not in tasks_by_id:
                    diag.note(
                        "unknown_task_ids",
                        {"block": block, "track": track, "offset": offset,
                         "task_id": task_id},
                    )
                parsed.append(
                    {
                        "offset": offset,
                        "timestamp": int(timestamp),
                        "task_id": task_id,
                        "is_start": bool(fields["is_start"]),
                        "expert_idx": int(fields["expert_idx"]),
                    }
                )
            if stop_offset < max_records:
                tail_begin = base + stop_offset * record_stride
                tail_end = base + max_records * record_stride
                stray = sum(1 for value in values[tail_begin:tail_end] if value != 0)
                if stray:
                    diag.counters["stray_slots_after_gap"] += stray
                    if len(diag.samples["stray_slots_after_gap"]) < SAMPLE_LIMIT:
                        diag.samples["stray_slots_after_gap"].append(
                            {"block": block, "track": track,
                             "first_gap_offset": stop_offset, "stray_elements": stray}
                        )

            total_records += written
            records_per_track[track] += written
            if written:
                blocks_with_records[track] += 1
            else:
                empty_block_tracks.append({"block": block, "track": track})
            is_truncated = written >= max_records
            record_counts.append(
                {
                    "block": block,
                    "track": track,
                    "role": tracks_by_id.get(track, {}).get("role", "unknown"),
                    "records": written,
                    "truncated": is_truncated,
                }
            )
            if is_truncated:
                truncated.append(
                    {
                        "block": block,
                        "track": track,
                        "role": tracks_by_id.get(track, {}).get("role", "unknown"),
                        "records": written,
                        "max_records": max_records,
                        "note": "offset saturated: later events on this track were dropped on device",
                    }
                )
            events.extend(
                pair_records(parsed, block, track, tasks_by_id, tracks_by_id, labels, diag)
            )

    events.sort(key=lambda e: (e["start_ns"], e["block"], e["track"], e["task_id"]))

    empty_tracks = [
        track for track in range(num_tracks) if records_per_track[track] == 0
    ]
    anchors = build_anchors(abi, events, num_sms)
    counters, samples = diag.as_dict(COUNTER_KINDS)
    summary = summarize(events)
    strict_failures = []
    for kind in COUNTER_KINDS:
        if counters.get(kind, 0):
            strict_failures.append(
                "{}={}".format(kind, counters[kind]))
    expected_records = 2 * len(events)
    if total_records != expected_records:
        strict_failures.append(
            "records_total={} != 2*events_total={}".format(
                total_records, expected_records))
    if truncated:
        strict_failures.append("truncated_tracks={}".format(len(truncated)))
    strict_valid = not strict_failures

    metadata = {
        "abi": {
            "schema": abi["schema"],
            "source_file": abi.get("source_file"),
            "source_sha256": abi.get("source_sha256"),
        },
        "raw_path": options.get("raw_path"),
        "raw_numel": numel,
        "required_numel": required_numel,
        "header_elems": header_elems,
        "elems_per_record": elems_per_record,
        "num_sms": num_sms,
        "num_sms_source": num_sms_source,
        "num_tracks": num_tracks,
        "num_tracks_source": num_tracks_source,
        "max_records": max_records,
        "max_records_source": max_records_source,
        "rank": options.get("rank"),
        "device": options.get("device"),
        "sidecar_path": options.get("sidecar_path"),
        "metadata_words": {
            "num_sms": int(meta_num_sms),
            "num_tracks": int(meta_num_tracks),
        },
        "events_total": len(events),
        "records_total": total_records,
        "records_expected": expected_records,
        "record_counts": record_counts,
        "records_by_track": [
            {
                "track": track,
                "role": tracks_by_id.get(track, {}).get("role", "unknown"),
                "label": labels.get(track, "track{}".format(track)),
                "records": records_per_track[track],
                "blocks_with_records": blocks_with_records[track],
                "may_be_empty": bool(tracks_by_id.get(track, {}).get("may_be_empty")),
            }
            for track in range(num_tracks)
        ],
        "truncated_tracks": truncated,
        "empty_tracks": empty_tracks,
        "empty_block_tracks_total": len(empty_block_tracks),
        "empty_block_tracks": empty_block_tracks[:200],
        "counters": counters,
        "samples": samples,
        "abi_provenance": abi_provenance,
        "diagnostic_summary": {
            "valid": strict_valid,
            "failures": strict_failures,
            "allow_invalid": bool(options.get("allow_invalid")),
        },
        "time_window": summary["time_window"],
        "summary_by_role": summary["by_role"],
        "summary_by_track": summary["by_track"],
        "summary_by_task": summary["by_task"],
    }
    return {
        "metadata": metadata,
        "block_map": block_map,
        "anchors": anchors,
        "events": events,
    }


def pick(*candidates):
    labels = ("cli", "sidecar", "buffer_metadata", "abi_default")
    for index, candidate in enumerate(candidates):
        if candidate:
            return int(candidate), labels[index] if index < len(labels) else "fallback"
    return 0, "unavailable"


def pair_records(parsed, block, track, tasks_by_id, tracks_by_id, labels, diag):
    track_meta = tracks_by_id.get(track, {})
    stacks = defaultdict(list)
    events = []
    open_count = 0
    for record in parsed:
        if record["is_start"]:
            record["depth"] = open_count
            stacks[record["task_id"]].append(record)
            open_count += 1
            continue
        stack = stacks[record["task_id"]]
        if not stack:
            diag.note(
                "unmatched_ends",
                {"block": block, "track": track, "offset": record["offset"],
                 "task_id": record["task_id"], "timestamp_ns": record["timestamp"]},
            )
            continue
        start = stack.pop()
        open_count -= 1
        expert_idx = start["expert_idx"]
        if expert_idx != record["expert_idx"]:
            diag.note(
                "expert_idx_mismatch",
                {"block": block, "track": track, "start_offset": start["offset"],
                 "end_offset": record["offset"], "task_id": record["task_id"],
                 "start_expert_idx": start["expert_idx"],
                 "end_expert_idx": record["expert_idx"]},
            )
            if expert_idx == 0:
                expert_idx = record["expert_idx"]
        duration = record["timestamp"] - start["timestamp"]
        if duration < 0:
            diag.note(
                "negative_durations",
                {"block": block, "track": track, "start_offset": start["offset"],
                 "end_offset": record["offset"], "task_id": record["task_id"],
                 "start_ns": start["timestamp"], "end_ns": record["timestamp"]},
            )
        task = tasks_by_id.get(
            record["task_id"],
            {
                "name": "task_{}".format(record["task_id"]),
                "short_name": "task_{}".format(record["task_id"]),
                "role": "unknown",
            },
        )
        events.append(
            {
                "block": block,
                "track": track,
                "role": track_meta.get("role", "unknown"),
                "track_label": labels.get(track, "track{}".format(track)),
                "task_id": record["task_id"],
                "task": task["name"],
                "short_name": task["short_name"],
                "task_role": task["role"],
                "expert_idx": expert_idx,
                "start_ns": start["timestamp"],
                "end_ns": record["timestamp"],
                "dur_ns": duration,
                "start_offset": start["offset"],
                "end_offset": record["offset"],
                "depth": start["depth"],
            }
        )
    for task_id, stack in stacks.items():
        for record in stack:
            diag.note(
                "unmatched_starts",
                {"block": block, "track": track, "offset": record["offset"],
                 "task_id": task_id, "timestamp_ns": record["timestamp"]},
            )
    return events


def build_anchors(abi, events, num_sms):
    anchor_abi = abi.get("sync_anchors", {}).get("entry_rendezvous")
    if anchor_abi is None:
        return {"entry_rendezvous": None}
    task_name = anchor_abi["task"]
    per_cta = {}
    for event in events:
        if event["task"] != task_name:
            continue
        current = per_cta.setdefault(
            event["block"],
            {
                "block": event["block"],
                "entry_sync_start_ns": event["start_ns"],
                "entry_sync_end_ns": event["end_ns"],
                "dur_ns": event["dur_ns"],
                "occurrences": 0,
            },
        )
        current["occurrences"] += 1
        if event["start_ns"] < current["entry_sync_start_ns"]:
            current["entry_sync_start_ns"] = event["start_ns"]
        if event["end_ns"] > current["entry_sync_end_ns"]:
            current["entry_sync_end_ns"] = event["end_ns"]
        current["dur_ns"] = current["entry_sync_end_ns"] - current["entry_sync_start_ns"]
    ordered = [per_cta[block] for block in sorted(per_cta)]
    result = {
        "entry_rendezvous": {
            "task": task_name,
            "task_id": anchor_abi.get("task_id"),
            "start_semantics": anchor_abi.get("start_semantics"),
            "end_semantics": anchor_abi.get("end_semantics"),
            "mechanism": anchor_abi.get("mechanism"),
            "per_cta": ordered,
            "ctas_with_anchor": len(ordered),
            "ctas_missing_anchor": [
                block for block in range(num_sms) if block not in per_cta
            ][:200],
            "ctas_missing_anchor_total": sum(
                1 for block in range(num_sms) if block not in per_cta
            ),
        }
    }
    if ordered:
        starts = [entry["entry_sync_start_ns"] for entry in ordered]
        ends = [entry["entry_sync_end_ns"] for entry in ordered]
        result["entry_rendezvous"]["global"] = {
            "entry_sync_start_ns": min(starts),
            "entry_sync_end_ns": max(ends),
            "min_start_ns": min(starts),
            "max_start_ns": max(starts),
            "min_end_ns": min(ends),
            "max_end_ns": max(ends),
            "start_skew_ns": max(starts) - min(starts),
            "end_skew_ns": max(ends) - min(ends),
            "rendezvous_span_ns": max(ends) - min(starts),
        }
    else:
        result["entry_rendezvous"]["global"] = None
    return result


def summarize(events):
    by_role = defaultdict(lambda: {"events": 0, "total_dur_ns": 0})
    by_track = defaultdict(lambda: {"events": 0, "total_dur_ns": 0})
    by_task = defaultdict(lambda: {"events": 0, "total_dur_ns": 0})
    for event in events:
        for bucket, key in (
            (by_role, event["role"]),
            (by_track, (event["track"], event["track_label"])),
            (by_task, (event["task"], event["short_name"], event["task_role"])),
        ):
            bucket[key]["events"] += 1
            bucket[key]["total_dur_ns"] += max(event["dur_ns"], 0)
    total_dur = sum(entry["total_dur_ns"] for entry in by_role.values())
    if events:
        min_start = min(event["start_ns"] for event in events)
        max_end = max(event["end_ns"] for event in events)
        makespan = max_end - min_start
    else:
        min_start = None
        max_end = None
        makespan = 0

    def share(value):
        return round(100.0 * value / total_dur, 3) if total_dur else 0.0

    def over_makespan(value):
        return round(value / makespan, 4) if makespan else 0.0

    role_rows = [
        {
            "role": role,
            "events": entry["events"],
            "total_dur_ns": entry["total_dur_ns"],
            "share_of_total_dur_pct": share(entry["total_dur_ns"]),
            "dur_over_makespan": over_makespan(entry["total_dur_ns"]),
        }
        for role, entry in sorted(
            by_role.items(), key=lambda item: -item[1]["total_dur_ns"]
        )
    ]
    track_rows = [
        {
            "track": key[0],
            "label": key[1],
            "events": entry["events"],
            "total_dur_ns": entry["total_dur_ns"],
            "share_of_total_dur_pct": share(entry["total_dur_ns"]),
        }
        for key, entry in sorted(by_track.items(), key=lambda item: item[0][0])
    ]
    task_rows = [
        {
            "task": key[0],
            "short_name": key[1],
            "task_role": key[2],
            "events": entry["events"],
            "total_dur_ns": entry["total_dur_ns"],
            "share_of_total_dur_pct": share(entry["total_dur_ns"]),
            "mean_dur_ns": round(entry["total_dur_ns"] / entry["events"], 1)
            if entry["events"]
            else 0,
        }
        for key, entry in sorted(
            by_task.items(), key=lambda item: -item[1]["total_dur_ns"]
        )
    ]
    return {
        "by_role": role_rows,
        "by_track": track_rows,
        "by_task": task_rows,
        "time_window": {
            "min_start_ns": min_start,
            "max_end_ns": max_end,
            "makespan_ns": makespan,
            "sum_event_dur_ns": total_dur,
        },
    }


def print_summary(decoded, top_tasks):
    metadata = decoded["metadata"]
    window = metadata["time_window"]
    print(
        "[decode] raw={} numel={} num_sms={}({}) num_tracks={}({}) "
        "max_records={}({}) rank={} device={}".format(
            metadata["raw_path"],
            metadata["raw_numel"],
            metadata["num_sms"],
            metadata["num_sms_source"],
            metadata["num_tracks"],
            metadata["num_tracks_source"],
            metadata["max_records"],
            metadata["max_records_source"],
            metadata["rank"],
            metadata["device"],
        )
    )
    makespan = window["makespan_ns"] or 0
    print(
        "[decode] events={} records={} makespan={} ns ({:.3f} us) "
        "sum_event_dur={} ns".format(
            metadata["events_total"],
            metadata["records_total"],
            makespan,
            makespan / 1000.0,
            window["sum_event_dur_ns"],
        )
    )
    print(
        "[decode] role breakdown (share% of summed event duration; dur/makespan "
        "aggregates every CTA, so it scales with num_sms):"
    )
    print(
        "  {:<10} {:>8} {:>15} {:>9} {:>14}".format(
            "role", "events", "total_dur_ns", "share%", "dur/makespan"
        )
    )
    for row in metadata["summary_by_role"]:
        print(
            "  {:<10} {:>8} {:>15} {:>8.2f}% {:>14.4f}".format(
                row["role"],
                row["events"],
                row["total_dur_ns"],
                row["share_of_total_dur_pct"],
                row["dur_over_makespan"],
            )
        )
    if top_tasks > 0 and metadata["summary_by_task"]:
        print("[decode] top {} tasks by total duration:".format(top_tasks))
        for row in metadata["summary_by_task"][:top_tasks]:
            print(
                "  {:<18} {:<8} {:>8} {:>15} {:>8.2f}%".format(
                    row["short_name"],
                    row["task_role"],
                    row["events"],
                    row["total_dur_ns"],
                    row["share_of_total_dur_pct"],
                )
            )
    anchor = decoded["anchors"].get("entry_rendezvous")
    if anchor and anchor.get("global"):
        info = anchor["global"]
        print(
            "[decode] entry_rank_sync: ctas={} start_min={} end_max={} "
            "start_skew={} ns end_skew={} ns span={} ns".format(
                anchor["ctas_with_anchor"],
                info["entry_sync_start_ns"],
                info["entry_sync_end_ns"],
                info["start_skew_ns"],
                info["end_skew_ns"],
                info["rendezvous_span_ns"],
            )
        )
    else:
        print("[decode] entry_rank_sync: no anchor events decoded")
    counters = metadata["counters"]
    print(
        "[decode] anomalies: "
        + " ".join("{}={}".format(key, counters[key]) for key in COUNTER_KINDS)
    )
    print(
        "[decode] truncated_block_tracks={} empty_tracks={} "
        "empty_block_tracks={}".format(
            len(metadata["truncated_tracks"]),
            metadata["empty_tracks"],
            metadata["empty_block_tracks_total"],
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Decode a raw MegaMoE device profiler buffer into normalized "
            "interval events using the machine readable ABI JSON."
        )
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--capture-plan", type=Path, required=True)
    parser.add_argument("--abi", type=Path, default=DEFAULT_ABI)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sidecar", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--num-sms", type=int, default=None)
    parser.add_argument("--num-tracks", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument("--top-tasks", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--allow-invalid", action="store_true",
        help="仅调试用：仍写出诊断 JSON，但允许严格校验失败时返回 0",
    )
    args = parser.parse_args(argv)

    try:
        capture_plan = json.loads(args.capture_plan.read_text(encoding="utf-8"))
        if capture_plan.get("schema") != "fast-instrument-capture-plan/v1":
            raise DecodeError("invalid --capture-plan schema")
        abi = load_abi(args.abi)
        values, container_meta = load_raw(args.raw)
        sidecar_path, sidecar_payload = find_sidecar(args.raw, args.sidecar)
        merged = dict(container_meta)
        merged.update(sidecar_payload)
        sidecar = normalize_sidecar(merged)
        options = {
            "raw_path": str(args.raw),
            "sidecar_path": str(sidecar_path) if sidecar_path else None,
            "max_records": args.max_records,
            "num_sms": args.num_sms,
            "num_tracks": args.num_tracks,
            "sidecar_max_records": sidecar.get("max_records"),
            "sidecar_num_sms": sidecar.get("num_sms"),
            "sidecar_num_tracks": sidecar.get("num_tracks"),
            "rank": args.rank if args.rank is not None else sidecar.get("rank"),
            "device": args.device if args.device is not None else sidecar.get("device"),
            "abi_path": args.abi,
            "allow_invalid": args.allow_invalid,
        }
        decoded = decode(values, abi, options)
        if capture_plan.get("mode") != "all":
            scope = merged.get("capture_scope")
            if not isinstance(scope, dict):
                raise DecodeError("selected physical-SM raw sidecar is missing capture_scope metadata")
            decoded["capture_scope"] = dict(scope)
        else:
            decoded["capture_scope"] = {
                "capture_scope": "all_physical_sms",
                "selected_physical_smids": [],
                "source_capture_was_all_sm": True,
            }
    except DecodeError as error:
        print("[decode] {}".format(error), file=sys.stderr)
        return 2

    out_path = args.out
    if out_path is None:
        out_path = args.raw.with_name(args.raw.stem + ".events.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(decoded, indent=args.indent) + "\n", encoding="utf-8"
    )
    if not args.quiet:
        print_summary(decoded, args.top_tasks)
        print("[decode] wrote {}".format(out_path))
    diagnostic = decoded["metadata"].get("diagnostic_summary", {})
    if not diagnostic.get("valid", True):
        print(
            "[decode] FAIL strict diagnostics: {}".format(
                "; ".join(diagnostic.get("failures", []))),
            file=sys.stderr,
        )
        return 0 if args.allow_invalid else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
