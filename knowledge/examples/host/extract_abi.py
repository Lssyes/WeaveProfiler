#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
SCHEMA_ID = "megamoe.profiler-abi/v1"
SOURCE_REL = "infra/deep_gemm/include/deep_gemm/profiler/mega_moe_profiler.cuh"
KERNEL_REL = "fixed/kernel.cuh"
HOST_REL = "infra/csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp"
ABI_REL = "profile/abi/megamoe_profiler_abi.json"

TAG_OPERAND_NAMES = (
    ("blockIdx.x", "block"),
    ("track_id", "track"),
    ("expert_idx", "expert_idx"),
    ("is_start", "is_start"),
    ("task", "task"),
)

TAG_FIELD_MEANING = {
    "block": "blockIdx.x of the recording persistent CTA",
    "track": "observer track id (role slot) inside the CTA",
    "expert_idx": "optional expert index argument, 0 when unused",
    "task": "MegaMoEProfileTask enum value",
    "is_start": "1 = interval open, 0 = interval close",
}

ROLE_BY_NAME_PREFIX = (
    ("kDispatch", "dispatch"),
    ("kLoaderA", "loader_a"),
    ("kLoaderB", "loader_b"),
    ("kMath", "math"),
    ("kEntry", "sync"),
)

SHORT_NAMES = {
    "kDispatchRoutePublish": "route_publish",
    "kDispatchNvlinkPull": "nvlink_pull",
    "kDispatchCleanup": "cleanup",
    "kDispatchSync": "dispatch_sync",
    "kLoaderAWaitPool": "a_wait_pool",
    "kLoaderAPipeline": "a_pipe",
    "kLoaderBPipeline": "b_pipe",
    "kMathL1GemmPipeline": "l1_gemm",
    "kMathL1SwiGLUQuantize": "l1_swiglu_q",
    "kMathL1TmaStore": "l1_tma_store",
    "kMathL2GemmPipeline": "l2_gemm",
    "kMathL2Bf16Scatter": "l2_scatter",
    "kMathCombineReduceStore": "combine_store",
    "kMathBeforeCombineBarrier": "pre_combine_bar",
    "kMathWaitDispatch": "wait_dispatch",
    "kMathWaitFullBarrier": "wait_full_bar",
    "kLoaderAWaitEmpty": "a_wait_empty",
    "kLoaderBWaitEmpty": "b_wait_empty",
    "kDispatchWaitBeforeCleanup": "wait_pre_cleanup",
    "kEntryRankSync": "entry_rank_sync",
}

TRACK_ROLES = {0: "dispatch", 1: "loader_a", 2: "loader_b"}

REFERENCE_CONFIG = {
    "name": "qwen3.6-ep2-sm90",
    "num_dispatch_warps": 2,
    "num_mma_non_epilogue_warps": 2,
    "num_epilogue_warps": 8,
    "num_math_warpgroups": 2,
    "num_warps": 12,
    "num_threads": 384,
    "math_warp_start": 4,
    "observer_warp_by_track": {"0": 0, "1": 2, "2": 3, "3": 4, "4": 8},
    "expected_empty_tracks": [5, 6],
    "provenance": (
        "warp counts come from the JIT specialization launched through "
        + HOST_REL
        + "; observer warp ids follow the binding block in "
        + KERNEL_REL
    ),
}


class ExtractError(RuntimeError):
    pass


def fail(message):
    raise ExtractError(message)


def normalize_ws(text):
    return re.sub(r"\s+", " ", text).strip()


def line_of(text, offset):
    return text.count("\n", 0, offset) + 1


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_text(path):
    if not path.is_file():
        fail("missing source file: {}".format(path))
    return path.read_text(encoding="utf-8")


def split_top_level(expr, separator):
    parts = []
    depth = 0
    current = []
    for ch in expr:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == separator and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    parts.append("".join(current))
    return [p for p in (normalize_ws(x) for x in parts) if p]


def balanced_slice(text, open_index):
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : i], i
    fail("unbalanced parenthesis starting at offset {}".format(open_index))


def extract_tasks(text):
    match = re.search(
        r"enum\s+class\s+MegaMoEProfileTask\s*:\s*(\w+)\s*\{(.*?)\}\s*;", text, re.S
    )
    if match is None:
        fail("enum class MegaMoEProfileTask not found")
    underlying = match.group(1)
    entries = re.findall(r"(k\w+)\s*=\s*(\d+)\s*,", match.group(2))
    if not entries:
        fail("no enumerators parsed from MegaMoEProfileTask")
    tasks = []
    for name, value in entries:
        task_id = int(value)
        role = None
        for prefix, candidate in ROLE_BY_NAME_PREFIX:
            if name.startswith(prefix):
                role = candidate
                break
        if role is None:
            fail("cannot derive role for task {}; add a prefix rule".format(name))
        short_name = SHORT_NAMES.get(name)
        if short_name is None:
            fail(
                "task {} has no short_name; add it to SHORT_NAMES in {}".format(
                    name, Path(__file__).name
                )
            )
        tasks.append(
            {"id": task_id, "name": name, "short_name": short_name, "role": role}
        )
    ids = [t["id"] for t in tasks]
    if ids != list(range(len(ids))):
        fail("task ids are not contiguous from 0: {}".format(ids))
    return underlying, tasks


def extract_int(text, pattern, what, base=10):
    match = re.search(pattern, text)
    if match is None:
        fail("{} not found (pattern {!r})".format(what, pattern))
    return int(match.group(1), base)


def extract_tag_fields(text):
    match = re.search(r"const\s+uint32_t\s+tag\s*=(.*?);", text, re.S)
    if match is None:
        fail("tag composition expression not found")
    fields = []
    for term in split_top_level(match.group(1), "|"):
        shift = re.search(r"<<\s*(\d+)", term)
        if shift is None:
            fail("tag term without shift: {}".format(term))
        lsb = int(shift.group(1))
        mask_hex = re.search(r"&\s*0x([0-9a-fA-F]+)u", term)
        if mask_hex is not None:
            width = int(mask_hex.group(1), 16).bit_length()
        elif "?" in term:
            width = 1
        else:
            width = 32 - lsb
        name = None
        for token, candidate in TAG_OPERAND_NAMES:
            if token in term:
                name = candidate
                break
        if name is None:
            fail("cannot name tag term: {}".format(term))
        fields.append(
            {
                "name": name,
                "lsb": lsb,
                "width": width,
                "mask": (1 << width) - 1,
                "msb": lsb + width - 1,
                "meaning": TAG_FIELD_MEANING[name],
                "source_term": term,
            }
        )
    fields.sort(key=lambda f: f["lsb"], reverse=True)
    lowest = min(f["lsb"] for f in fields)
    if lowest > 0:
        fields.append(
            {
                "name": "unused",
                "lsb": 0,
                "width": lowest,
                "mask": (1 << lowest) - 1,
                "msb": lowest - 1,
                "meaning": "not written by the kernel, always 0",
                "source_term": None,
            }
        )
    covered = 0
    for field in fields:
        span = field["mask"] << field["lsb"]
        if covered & span:
            fail("tag bitfields overlap at field {}".format(field["name"]))
        covered |= span
    if covered != 0xFFFFFFFF:
        fail("tag bitfields do not cover 32 bits, covered=0x{:08x}".format(covered))
    return fields


def extract_slot_layout(text, num_tracks):
    match = re.search(r"const\s+uint64_t\s+slot\s*=(.*?);", text, re.S)
    if match is None:
        fail("slot expression not found")
    expr = normalize_ws(match.group(1))
    if re.search(r"blockIdx\.x\)\s*\*\s*kMegaMoEProfileNumTracks\s*\*\s*max_records", expr) is None:
        fail("slot block stride changed: {}".format(expr))
    if re.search(r"track_id\)\s*\*\s*max_records", expr) is None:
        fail("slot track stride changed: {}".format(expr))
    if re.search(r"\+\s*offset\)", expr) is None:
        fail("slot offset term changed: {}".format(expr))
    scale = re.search(r"\*\s*(\d+)ull", expr)
    if scale is None:
        fail("slot element scale not found: {}".format(expr))
    elems_per_record = int(scale.group(1))
    ts_match = re.search(r"buf\[kHeader \+ slot\]\s*=\s*static_cast<int64_t>\((\w+)\)", text)
    tag_match = re.search(r"buf\[kHeader \+ slot \+ 1\]\s*=\s*static_cast<int64_t>\((\w+)\)", text)
    if ts_match is None or tag_match is None:
        fail("record store pattern changed")
    return {
        "elems_per_record": elems_per_record,
        "block_stride_expr": "num_tracks * max_records * {}".format(elems_per_record),
        "track_stride_expr": "max_records * {}".format(elems_per_record),
        "record_stride_expr": str(elems_per_record),
        "slot_formula": (
            "slot = (block * num_tracks * max_records + track * max_records + offset) * "
            + str(elems_per_record)
        ),
        "element_index_formula": "header_elems + slot + field_index",
        "fields": [
            {"index": 0, "name": ts_match.group(1), "dtype": "uint64", "unit": "ns"},
            {"index": 1, "name": tag_match.group(1), "dtype": "uint32", "unit": "bitfield"},
        ],
        "num_tracks_symbol_value": num_tracks,
        "source_expr": expr,
    }


def extract_region_layout(text):
    metadata_writes = re.findall(r"metadata\[(\d+)\]\s*=\s*([A-Za-z_][\w:]*)\s*;", text)
    if len(metadata_writes) < 2:
        fail("metadata writes not found")
    metadata_fields = []
    names = {"kNumSMs": "num_sms", "kMegaMoEProfileNumTracks": "num_tracks"}
    for word_index, symbol in metadata_writes:
        word = int(word_index)
        key = names.get(symbol)
        if key is None:
            fail("unknown metadata symbol {}".format(symbol))
        metadata_fields.append(
            {
                "name": key,
                "word_index": word,
                "lsb": word * 32,
                "width": 32,
                "mask": 0xFFFFFFFF,
                "source_symbol": symbol,
            }
        )
    base_match = re.search(
        r"reinterpret_cast<uint32_t\*>\(buf \+ ([^)]+)\)", text
    )
    if base_match is None:
        fail("block map base expression not found")
    block_writes = re.findall(r"block_map\[(\d+)\]\s*=\s*([^;]+);", text)
    if len(block_writes) < 2:
        fail("block map writes not found")
    block_names = {"blockIdx.x": "logical_cta", "physical_smid": "physical_smid"}
    block_fields = []
    for word_index, rhs in block_writes:
        word = int(word_index)
        key = None
        for token, candidate in block_names.items():
            if token in rhs:
                key = candidate
                break
        if key is None:
            fail("unknown block map rhs {}".format(rhs))
        block_fields.append(
            {
                "name": key,
                "word_index": word,
                "lsb": word * 32,
                "width": 32,
                "mask": 0xFFFFFFFF,
                "source_expr": normalize_ws(rhs),
            }
        )
    writer_guard = "track_id == 0" if "if (track_id == 0)" in text else None
    return metadata_fields, normalize_ws(base_match.group(1)), block_fields, writer_guard


def nearest_condition(prefix):
    matches = list(re.finditer(r"\bif\s*\(", prefix))
    if not matches:
        return None
    last = matches[-1]
    open_index = prefix.index("(", last.start())
    condition, _ = balanced_slice(prefix, open_index)
    head = "if"
    before = normalize_ws(prefix[max(0, last.start() - 12) : last.start()])
    if before.endswith("else"):
        head = "else if"
    return "{} ({})".format(head, normalize_ws(condition))


def extract_track_bindings(kernel_text, num_tracks):
    inits = list(re.finditer(r"role_profiler\.init\((.*?)\)\s*;", kernel_text, re.S))
    if not inits:
        fail("no role_profiler.init call found in kernel source")
    literal = {}
    family = None
    for match in inits:
        args = split_top_level(match.group(1), ",")
        track_expr = args[-1]
        condition = nearest_condition(kernel_text[: match.start()])
        source_line = line_of(kernel_text, match.start())
        if re.fullmatch(r"\d+", track_expr):
            literal[int(track_expr)] = (track_expr, condition, source_line)
        else:
            base = re.match(r"(\d+)\s*\+", track_expr)
            if base is None:
                fail("cannot interpret track expression {!r}".format(track_expr))
            family = (int(base.group(1)), track_expr, condition, source_line)
    tracks = []
    populated = set()
    if family is not None:
        base_id = family[0]
        populated = {
            base_id + i for i in range(REFERENCE_CONFIG["num_math_warpgroups"])
        }
    for track_id in range(num_tracks):
        family_track = False
        if track_id in literal:
            track_expr, condition, source_line = literal[track_id]
            populated.add(track_id)
        elif family is not None and track_id >= family[0]:
            track_expr, condition, source_line = family[1], family[2], family[3]
            family_track = True
        else:
            fail("track {} has no binding rule".format(track_id))
        role = TRACK_ROLES.get(track_id, "math")
        warp_expr = None
        if condition is not None:
            warp_match = re.search(
                r"warp_idx\s*==\s*([A-Za-z_0-9]+(?:\s*\+\s*\d+)?)", condition
            )
            if warp_match is not None:
                warp_expr = normalize_ws(warp_match.group(1))
            elif family_track:
                lower = re.search(r"warp_idx\s*>=\s*([A-Za-z_][\w]*)", condition)
                stride = re.search(r"%\s*(\d+)\s*==\s*0", condition)
                if lower is not None and stride is not None:
                    warp_expr = "{} + (track_id - {}) * {}".format(
                        lower.group(1), family[0], stride.group(1)
                    )
        reference_warp = REFERENCE_CONFIG["observer_warp_by_track"].get(str(track_id))
        tracks.append(
            {
                "track_id": track_id,
                "role": role,
                "track_id_expr": track_expr,
                "binding_rule": condition,
                "binding_source_line": source_line,
                "observer_warp_expr": warp_expr,
                "observer_lane": 0,
                "reference_warp_idx": reference_warp,
                "may_be_empty": track_id not in populated,
                "populated_in_reference_config": track_id in populated,
            }
        )
    return tracks


def extract_entry_sync(kernel_text, tasks_by_name):
    if "kEntryRankSync" not in kernel_text:
        fail("kEntryRankSync is not referenced by the kernel source")
    indices = {}
    for symbol, key in (
        ("kProfileEntryGridSyncIndex", "entry_barrier_grid_sync_index"),
        ("kProfileEntryPublishGridSyncIndex", "entry_publish_grid_sync_index"),
        ("kProfileEntryCtaBarrierIdx", "cta_barrier_idx"),
    ):
        indices[key] = extract_int(
            kernel_text, r"{}\s*=\s*(\d+)\s*;".format(symbol), symbol
        )
    guard_match = re.search(r"if\s+constexpr\s*\(kEnableProfiler\)", kernel_text)
    has_nvlink = re.search(r"comm::nvlink_barrier<", kernel_text) is not None
    has_grid_sync = re.search(r"comm::grid_sync<", kernel_text) is not None
    if not (guard_match is not None and has_nvlink and has_grid_sync):
        fail("entry rendezvous structure changed in kernel source")
    source = KERNEL_REL
    brace = kernel_text.find("{", guard_match.end())
    if brace != -1:
        depth = 0
        for position in range(brace, len(kernel_text)):
            if kernel_text[position] == "{":
                depth += 1
            elif kernel_text[position] == "}":
                depth -= 1
                if depth == 0:
                    source = "{}:{}-{}".format(
                        KERNEL_REL,
                        line_of(kernel_text, guard_match.start()),
                        line_of(kernel_text, position),
                    )
                    break
    return {
        "entry_rendezvous": {
            "task": "kEntryRankSync",
            "task_id": tasks_by_name["kEntryRankSync"]["id"],
            "track": 0,
            "recorded_by": "dispatch observer warp (track 0, lane 0)",
            "guaranteed_first_event_per_cta": True,
            "start_semantics": "rank_arrive",
            "end_semantics": "rank_release",
            "end_aliases": ["rank_release", "core_begin"],
            "mechanism": {
                "between_start_and_end": "comm::nvlink_barrier across ranks",
                "after_end": (
                    "comm::grid_sync publishes every CTA end record before any "
                    "dispatch/loader/math warp starts business work"
                ),
                "profile_only": True,
                "guard": "if constexpr (kEnableProfiler)",
            },
            "grid_sync_indices": indices,
            "alignment_usage": (
                "cross-rank timeline alignment baseline: shift ranks so that "
                "max(entry_sync_end_ns) or min(entry_sync_start_ns) coincide"
            ),
            "source": source,
            "source_file": KERNEL_REL,
        }
    }


def extract_host_contract(host_text, num_tracks, offset_bits):
    contract = {
        "gate": "args.profiler_buf != nullptr",
        "source": HOST_REL,
        "dtype": "torch.long",
        "device": "cuda",
        "contiguous": True,
        "zero_init_required": True,
    }
    tracks_host = re.search(r"kProfileNumTracks\s*=\s*(\d+)", host_text)
    if tracks_host is None:
        fail("host side kProfileNumTracks not found in {}".format(HOST_REL))
    if int(tracks_host.group(1)) != num_tracks:
        fail(
            "host num_tracks {} disagrees with device num_tracks {}".format(
                tracks_host.group(1), num_tracks
            )
        )
    limit = re.search(r"profiler_max_records\)\s*<\s*\(1ull\s*<<\s*(\d+)\)", host_text)
    if limit is None:
        fail("host max_records limit assertion not found")
    if int(limit.group(1)) != offset_bits:
        fail(
            "host max_records limit 1<<{} disagrees with kOffsetBits {}".format(
                limit.group(1), offset_bits
            )
        )
    if re.search(r"required_elements\s*=\s*1ull\s*\+", host_text) is None:
        fail("host required_elements formula changed")
    contract["max_records_exclusive_limit"] = 1 << offset_bits
    contract["required_numel_expr"] = (
        "1 + num_sms + num_sms * {} * max_records * 2".format(num_tracks)
    )
    contract["asserts_numel_at_least_required"] = True
    return contract


def build_abi(source_path, kernel_path, host_path):
    source_text = read_text(source_path)
    kernel_text = read_text(kernel_path)
    host_text = read_text(host_path)

    enum_underlying, tasks = extract_tasks(source_text)
    tasks_by_name = {t["name"]: t for t in tasks}
    num_tracks = extract_int(
        source_text,
        r"constexpr\s+uint32_t\s+kMegaMoEProfileNumTracks\s*=\s*(\d+)\s*;",
        "kMegaMoEProfileNumTracks",
    )
    offset_bits = extract_int(
        source_text, r"kOffsetBits\s*=\s*(\d+)\s*;", "kOffsetBits"
    )
    disabled_state = extract_int(
        source_text, r"kDisabled\s*=\s*0x([0-9a-fA-F]+)u\s*;", "kDisabled", base=16
    )
    header_match = re.search(r"kHeader\s*=\s*([^;]+);", source_text)
    if header_match is None:
        fail("kHeader expression not found")
    header_expr = normalize_ws(header_match.group(1))
    if re.search(r"kOffsetMask\s*=\s*\(1u\s*<<\s*kOffsetBits\)\s*-\s*1u", source_text) is None:
        fail("kOffsetMask definition changed")
    if re.search(r"if\s*\(offset\s*>=\s*max_records\)\s*return\s*;", source_text) is None:
        fail("overflow guard changed; expected `if (offset >= max_records) return;`")
    if re.search(r"mov\.u64\s+%0,\s*%%globaltimer", source_text) is None:
        fail("timestamp source changed; %globaltimer read not found")
    if re.search(r"mov\.u32\s+%0,\s*%%smid", source_text) is None:
        fail("physical smid read not found")
    if "atom" in source_text:
        fail("unexpected atomic usage in profiler source")

    tag_fields = extract_tag_fields(source_text)
    slot_layout = extract_slot_layout(source_text, num_tracks)
    metadata_fields, block_map_base, block_fields, writer_guard = extract_region_layout(
        source_text
    )
    tracks = extract_track_bindings(kernel_text, num_tracks)
    sync_anchors = extract_entry_sync(kernel_text, tasks_by_name)
    host_contract = extract_host_contract(host_text, num_tracks, offset_bits)
    elems_per_record = slot_layout["elems_per_record"]

    return {
        "schema": SCHEMA_ID,
        "source_file": SOURCE_REL,
        "source_sha256": sha256_of(source_path),
        "binding_source_file": KERNEL_REL,
        "binding_source_sha256": sha256_of(kernel_path),
        "host_source_file": HOST_REL,
        "host_source_sha256": sha256_of(host_path),
        "generator": "profile/scripts/extract_abi.py",
        "regeneration": (
            "python3 -B profile/scripts/extract_abi.py --write, then "
            "python3 -B profile/scripts/extract_abi.py --check"
        ),
        "target": {
            "arch": "sm_90",
            "device_family": "Hopper/H20",
            "kernel": "SM90 FP8 MegaMoE megakernel (persistent CTAs, warp specialized)",
        },
        "clock": {
            "source": "%globaltimer",
            "unit": "ns",
            "width_bits": 64,
            "scope": "gpu-global",
            "monotonic_within_device": True,
            "comparable_across_ctas": True,
            "comparable_across_ranks": False,
            "cross_rank_alignment": "use sync_anchors.entry_rendezvous",
        },
        "buffer": {
            "element_dtype": "int64",
            "torch_dtype": "torch.long",
            "element_bytes": 8,
            "zero_init_required": True,
            "zero_init_rationale": (
                "unwritten record slots are only recognizable because both "
                "elements stay 0"
            ),
            "header_elems_expr": header_expr.replace("kNumSMs", "num_sms"),
            "header_elems_source_expr": header_expr,
            "elems_per_record": elems_per_record,
            "slot_formula": slot_layout["slot_formula"],
            "element_index_formula": slot_layout["element_index_formula"],
            "numel_formula": "1 + num_sms + num_sms * num_tracks * max_records * {}".format(
                elems_per_record
            ),
            "num_tracks": num_tracks,
            "empty_slot_sentinel": {
                "field": "timestamp_ns",
                "value": 0,
                "note": "timestamp_ns == 0 marks a never-written slot",
            },
            "overflow_policy": "drop_on_full_no_wrap",
            "overflow_detail": (
                "record() returns early once offset >= max_records; dropped "
                "events are not counted anywhere on device"
            ),
            "append_semantics": "per (block, track) monotonically increasing offset",
            "atomic_free": True,
            "offset_bits": offset_bits,
            "offset_mask": (1 << offset_bits) - 1,
            "disabled_state": disabled_state,
            "state_packing": "state = (track_id << kOffsetBits) | offset",
        },
        "regions": {
            "metadata": {
                "elem_index": 0,
                "elem_count": 1,
                "view_dtype": "uint32",
                "writer": "track 0 observer of blockIdx.x == 0",
                "fields": metadata_fields,
            },
            "block_map": {
                "elem_index_expr": block_map_base.replace("blockIdx.x", "block"),
                "elem_index_source_expr": block_map_base,
                "elem_count_expr": "num_sms",
                "view_dtype": "uint32",
                "writer": "track 0 observer of every CTA",
                "writer_guard": writer_guard,
                "purpose": "logical CTA -> physical SM id mapping",
                "fields": block_fields,
            },
            "records": {
                "elem_index_expr": header_expr.replace("kNumSMs", "num_sms"),
                "elem_count_expr": "num_sms * num_tracks * max_records * {}".format(
                    elems_per_record
                ),
                "layout": slot_layout,
            },
        },
        "tag_storage": {
            "container_dtype": "int64",
            "value_bits": 32,
            "high_bits_must_be_zero": True,
            "note": "tag occupies the low 32 bits of the second record element",
        },
        "tag_bitfields": {
            field["name"]: {
                "lsb": field["lsb"],
                "width": field["width"],
                "mask": field["mask"],
                "msb": field["msb"],
                "meaning": field["meaning"],
            }
            for field in tag_fields
        },
        "tag_field_order_msb_first": [field["name"] for field in tag_fields],
        "enum": {
            "name": "MegaMoEProfileTask",
            "underlying_type": enum_underlying,
            "count": len(tasks),
        },
        "tasks": tasks,
        "tracks": tracks,
        "sync_anchors": sync_anchors,
        "enable_switch": {
            "device_template": "MegaMoEProfiler<kEnabled, kNumSMs>",
            "disabled_specialization": (
                "MegaMoEProfiler<false, kNumSMs> has empty init/start/end, so "
                "the non-profiled JIT specialization compiles the probes away"
            ),
            "host_contract": host_contract,
        },
        "decoder_hints": {
            "scan_order": "per (block, track), ascending offset",
            "stop_condition": (
                "first slot with timestamp_ns == 0 and tag == 0 terminates the "
                "track; anything non-zero after it is corruption"
            ),
            "pairing": (
                "stack per task_id inside one (block, track); nesting of "
                "different tasks is expected (waits inside pipelines)"
            ),
            "truncation_check": "records == max_records means events may be missing",
            "consistency_check": (
                "tag.block / tag.track must equal the block / track implied by "
                "the slot index"
            ),
            "empty_tracks_are_legal": True,
        },
        "reference_config": REFERENCE_CONFIG,
    }


def diff_json(expected, actual, path="$"):
    diffs = []
    if type(expected) is not type(actual) and not (
        isinstance(expected, (int, float))
        and isinstance(actual, (int, float))
        and not isinstance(expected, bool)
        and not isinstance(actual, bool)
    ):
        diffs.append(
            {
                "path": path,
                "issue": "type mismatch",
                "expected": type(expected).__name__,
                "actual": type(actual).__name__,
            }
        )
        return diffs
    if isinstance(expected, dict):
        for key in expected:
            child = "{}.{}".format(path, key)
            if key not in actual:
                diffs.append(
                    {
                        "path": child,
                        "issue": "missing in ABI file",
                        "expected": expected[key],
                        "actual": None,
                    }
                )
            else:
                diffs.extend(diff_json(expected[key], actual[key], child))
        for key in actual:
            if key not in expected:
                diffs.append(
                    {
                        "path": "{}.{}".format(path, key),
                        "issue": "unexpected key in ABI file",
                        "expected": None,
                        "actual": actual[key],
                    }
                )
        return diffs
    if isinstance(expected, list):
        if len(expected) != len(actual):
            diffs.append(
                {
                    "path": path,
                    "issue": "length mismatch",
                    "expected": len(expected),
                    "actual": len(actual),
                }
            )
        for index in range(min(len(expected), len(actual))):
            diffs.extend(
                diff_json(expected[index], actual[index], "{}[{}]".format(path, index))
            )
        return diffs
    if expected != actual:
        diffs.append(
            {
                "path": path,
                "issue": "value mismatch",
                "expected": expected,
                "actual": actual,
            }
        )
    return diffs


def abbreviate(value, limit=200):
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if len(text) > limit:
            text = text[:limit] + "...<truncated>"
        return text
    return value


def format_diff(diff):
    return "{path}: {issue}; expected={expected!r}; actual={actual!r}".format(
        path=diff["path"],
        issue=diff["issue"],
        expected=abbreviate(diff["expected"]),
        actual=abbreviate(diff["actual"]),
    )


def relative_abi(abi_path):
    if abi_path.is_relative_to(WORKSPACE):
        return abi_path.relative_to(WORKSPACE)
    return abi_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Extract the MegaMoE device profiler ABI from CUDA sources and "
            "check or regenerate the machine readable ABI JSON."
        )
    )
    parser.add_argument("--source", type=Path, default=WORKSPACE / SOURCE_REL)
    parser.add_argument("--kernel", type=Path, default=WORKSPACE / KERNEL_REL)
    parser.add_argument("--host", type=Path, default=WORKSPACE / HOST_REL)
    parser.add_argument("--abi", type=Path, default=WORKSPACE / ABI_REL)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", default=True)
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        expected = build_abi(args.source, args.kernel, args.host)
    except ExtractError as error:
        print("[extract_abi] extraction failed: {}".format(error), file=sys.stderr)
        return 2

    if args.write:
        args.abi.parent.mkdir(parents=True, exist_ok=True)
        args.abi.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        print(
            "[extract_abi] wrote {} ({} tasks, {} tracks)".format(
                args.abi, len(expected["tasks"]), len(expected["tracks"])
            )
        )
        return 0

    if not args.abi.is_file():
        print(
            "[extract_abi] FAIL {}: ABI file missing; run with --write".format(args.abi)
        )
        return 1
    try:
        actual = json.loads(args.abi.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        print(
            "[extract_abi] FAIL {}: ABI file is not valid JSON: {}".format(
                args.abi, error
            )
        )
        return 1

    hash_checks = (
        ("source_sha256", args.source),
        ("binding_source_sha256", args.kernel),
        ("host_source_sha256", args.host),
    )
    already_reported = {"$." + key for key, _ in hash_checks}

    failures = []
    for key, source_path in hash_checks:
        recorded = actual.get(key)
        current = expected[key]
        if recorded != current:
            failures.append(
                "$.{key}: source hash drift; expected={current!r}; actual={recorded!r}; "
                "{path} now hashes to {current}; rerun --write and manually "
                "re-confirm the tag bitfields and slot formula".format(
                    key=key, current=current, recorded=recorded, path=source_path
                )
            )
    for diff in diff_json(expected, actual):
        if diff["path"] in already_reported:
            continue
        failures.append(format_diff(diff))

    if failures:
        print(
            "[extract_abi] FAIL {abi}: {count} inconsistency(ies) between the CUDA "
            "sources and the ABI JSON".format(
                abi=relative_abi(args.abi), count=len(failures)
            )
        )
        for item in failures:
            print("[extract_abi]   " + item)
        return 1

    if not args.quiet:
        print(
            "[extract_abi] OK {}: {} tasks, {} tracks, "
            "tag={}, sha256={}".format(
                relative_abi(args.abi),
                len(expected["tasks"]),
                len(expected["tracks"]),
                ",".join(
                    "{}[{}:{}]".format(
                        name,
                        expected["tag_bitfields"][name]["msb"],
                        expected["tag_bitfields"][name]["lsb"],
                    )
                    for name in expected["tag_field_order_msb_first"]
                ),
                expected["source_sha256"][:12],
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

