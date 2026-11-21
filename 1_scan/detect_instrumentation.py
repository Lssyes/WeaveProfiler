#!/usr/bin/env python3
"""Fingerprint-scan a repository and decide fast-path vs slow-path instrumentation.

Emits a machine-readable verdict (instrumentation-detect/v1) on stdout and a
human summary on stderr. Structural fingerprints are scanned mechanically; task
semantics are never inferred here and pass only through a model-reviewed,
source-locked ABI contract.

Exit status: 0 when next_action.path == "fast", 3 when it is "slow", and 4
when --require-hw-capture-range is set but the kernel-only cudaProfilerApi
range is missing.  --exit-zero does not override that explicit HW hard gate.
"""

from __future__ import annotations

import argparse
import bisect
import fnmatch
import io
import json
import os
import re
import sys
import tokenize

from semantic_contract import validate_semantic_contract

SCHEMA = "instrumentation-detect/v1"

DEFAULT_PRUNE_DIRS = (".git", "__pycache__", "node_modules", ".work", "build")
DEFAULT_EXCLUDES = ("refs", "references", "docs")

# .py is a device ext too: CuTe-DSL kernels are authored in Python. Device-scope
# signals (globaltimer / fixed-slot) therefore also scan .py, but every device
# capability stays gated on a co-located timestamp read (see the fixed_slot loop),
# so a pure host/decoder .py cannot gain a device capability from this.
DEVICE_EXTS = frozenset({".cu", ".cuh", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx", ".inc", ".ptx", ".py"})
HOST_EXTS = frozenset({".py", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx"})
DECODER_EXTS = frozenset({".py", ".cc", ".cpp", ".cxx"})
BUILD_EXTS = frozenset({".py", ".sh", ".cmake", ".txt", ".toml", ".yaml", ".yml", ".json", ".mk"})
SCAN_EXTS = DEVICE_EXTS | HOST_EXTS | BUILD_EXTS

CAPABILITY_ORDER = (
    "device_timestamping",
    "fixed_slot_no_contention",
    "compile_time_switch",
    "buffer_allocation",
    "rank_sync_anchor",
    "task_phase_definition",
    "source_locked_semantic_contract",
    "format_spec",
    "host_decoder",
    "perfetto_emitter",
)

HW_CAPTURE_RANGE_CAPABILITY = "hw_cuda_profiler_capture_range"

RECORD_CAPACITY = r"(?:max_records|kMaxRecords|max_recs|profiler_max_records|num_records|slots_per_cta)"

FIXED_SLOT_STRICT = (
    r"blockIdx\s*\.\s*x[^;{}]{0,240}?\*[^;{}]{0,240}?\b" + RECORD_CAPACITY + r"\b"
    r"[^;{}]{0,240}?\+[^;{}]{0,240}?\btrack\w*\b[^;{}]{0,200}?\*[^;{}]{0,200}?\b" + RECORD_CAPACITY + r"\b"
)
FIXED_SLOT_GENERIC = r"\*[^;{}]{0,100}?\b" + RECORD_CAPACITY + r"\b[^;{}]{0,140}?\+"

SEVEN_ANCHORS = ("KernelBegin", "EntryBefore", "EntryAfter", "MetadataEnd", "ExitBefore", "ExitAfter", "KernelEnd")

PATTERN_SPECS = (
    ("clock_globaltimer", r"%%?globaltimer|read_ptx_sreg_globaltimer", "device", 1),
    ("clock_clock64", r"\bclock64\s*\(", "device", 1),
    ("clock_smid", r"%%?smid", "device", 1),
    # AMD gfx9xx: s_memrealtime is the 64-bit global realtime clock, coherent
    # across CUs (measured on MI308X; clock64()/s_clock stay per-CU cycle
    # counters with >100ms cross-CU phase skew, unusable for timelines).
    # clock_amd_hwid is the %smid analogue: HW_REG_HW_ID via s_getreg.
    ("clock_smemrealtime", r"s_memrealtime|__builtin_amdgcn_s_memrealtime", "device", 1),
    ("clock_amd_hwid", r"HW_REG_HW_ID|__builtin_amdgcn_s_getreg", "device", 1),
    ("profiler_enum_task", r"enum\s+class\s+(\w*Profile\w*Task)\b", "device", 1),
    ("task_enum_def", r"enum\s+class\s+\w*Task\w*\b|\bk\w*Task\w*\s*=\s*\d"
                      r"|\bTASK_\w+\s*[=:]\s*\d|\btask_names?\s*[=:]|\bTASK_NAMES\b", "any", 1),
    ("phase_enum_def", r"enum\s+class\s+\w*Phase\w*\b|\bk\w*Phase\w*\s*=\s*\d"
                       r"|\bPHASE_\w+\s*[=:]\s*\d|\bphase_names?\s*[=:]|\bPHASE_NAMES\b", "any", 1),
    ("profiler_struct", r"\bstruct\s+(\w*Profiler)\b", "device", 1),
    ("profiler_enable_flag", r"\bk\w*EnableProfiler\w*\b|\bkEnableProfiler\b", "any", 1),
    ("profiler_buf_symbol", r"\bprofiler_buf\b", "any", 1),
    ("profiler_max_records", r"\bprofiler_max_records\b", "any", 1),
    ("profiler_num_tracks", r"\bnum_tracks\b|\b\w*NumTracks\b", "any", 1),
    ("abi_magic", r"\b(GPTRACE4|GPPACKD1|TKTIMEL1|mega_moe_arena_v2)\b", "any", 1),
    ("fixed_slot_formula", FIXED_SLOT_STRICT, "device", 3),
    ("fixed_slot_formula_generic", FIXED_SLOT_GENERIC, "device", 3),
    # CuTe-DSL (Python) device signals. Scope "pydevice" restricts them to .py so
    # C++ scanning is byte-for-byte unchanged. A direct write of a value into a
    # record buffer subscript -- profiler-named (prof_buf[...] = / mProf[...] =) or
    # the m-prefixed memref record buffer idiom (mBuf[slot, warp, n, 0] = ts) -- is
    # the CuTe analogue of the fixed-slot record store; it only contributes to the
    # fixed_slot capability when a %globaltimer/read_ptx_sreg_globaltimer read is
    # co-located in the same file (see the fixed_slot loop), so pure host/decoder
    # Python cannot gain a device capability from it. The name alternation stays
    # tight (needs 'prof' or an m...buf name) so compute tensors like mA/mC and a
    # plain output 'buf' never match; (?!=) avoids == comparisons.
    ("fixed_slot_cute_record", r"(?i)\b(?:\w*prof\w*|m\w*buf)\s*\[[^\]]*\]\s*=(?!=)", "pydevice", 1),
    ("profiler_buf_cute", r"(?i)\b\w*prof\w*\s*:\s*cute\.Tensor\b", "pydevice", 1),
    ("atomic_add", r"\batomicAdd\s*\(", "device", 1),
    ("switch_template_bool", r"template\s*<\s*bool\s+k?\w*[Ee]nabl\w*", "device", 2),
    ("switch_false_specialization", r"struct\s+\w*Profiler\w*\s*<\s*false\b", "device", 2),
    ("switch_macro_ifdef", r"#\s*if(?:def)?\s*(?:defined\s*\(\s*)?\w*(?:PROFIL|INSTRUMENT|GPTRACE)\w*", "any", 1),
    ("switch_build_define", r"-D\s*\w*(?:PROFILER|PROFILE|INSTRUMENT)\w*", "any", 1),
    ("rank_sync_named", r"\bk?\w*EntryRankSync\b|\brank_sync\b|\bcentral_rank_sync\b", "any", 1),
    ("rank_sync_nvlink", r"\bnvlink_barrier\b", "any", 1),
    ("rank_sync_grid", r"\bgrid_sync\b", "any", 1),
    ("rank_sync_anchor_name", r"\bk?(" + "|".join(SEVEN_ANCHORS) + r")\b", "any", 1),
    ("host_buffer_validate", r"\bnumel\s*\(|\bis_contiguous\s*\(|torch::kLong|torch\.long\b|torch\.int64\b|\bscalar_type\s*\(", "host", 1),
    ("host_python_param", r"\bprofiler_buf\s*[:=]|\bprofiler_max_records\s*[:=]", "host", 1),
    ("decode_intent", r"\bdecode\w*|\bDecode\w*", "decoder", 1),
    ("perfetto_intent", r"\btraceEvents\b|\bperfetto\b|\bPerfetto\b", "decoder", 1),
    ("bitmask_hex", r"&\s*0x[0-9a-fA-F]+", "decoder", 1),
    (
        "hw_capture_cli",
        r"\badd_argument\s*\([^)]{0,320}['\"]--(?:cuda-profiler(?:-api)?(?:-range)?|hw-capture-range|capture-range)['\"]",
        "host",
        4,
    ),
    ("hw_cuda_profiler_start", r"\bcudaProfilerStart\s*\(", "host", 1),
    ("hw_cuda_profiler_stop", r"\bcudaProfilerStop\s*\(", "host", 1),
    (
        "hw_torch_profiler_start",
        r"\btorch\s*\.\s*cuda\s*\.\s*profiler\s*\.\s*start\s*\(",
        "host",
        1,
    ),
    (
        "hw_torch_profiler_stop",
        r"\btorch\s*\.\s*cuda\s*\.\s*profiler\s*\.\s*stop\s*\(",
        "host",
        1,
    ),
)

NUM_TRACKS_VALUE_RE = re.compile(r"\b(?:num_tracks|\w*NumTracks)\b\s*=\s*(\d+)")
TAG_ASSIGN_RE = re.compile(r"\btag\w*\s*=[^=]")
SHIFT_LEFT_RE = re.compile(r"<<\s*(\d{1,2})\b")
SHIFT_RIGHT_RE = re.compile(r">>\s*(\d{1,2})\b")
MASK_RE = re.compile(r"&\s*(0x[0-9a-fA-F]+)")

ABI_JSON_GLOBS = ("*profiler*abi*.json", "*abi*profiler*.json", "*-layout.json", "*_layout.json")
ABI_COUPLING_TOKENS = ("tag", "track", "max_records", "num_tracks", "globaltimer", "s_memrealtime", "smid", "record_stride")

JSON_PATH_LITERAL_RE = re.compile(r"""['"]([^'"\s]+\.json)['"]""")
ABI_SOURCE_FIELDS = (
    "source_file",
    "source",
    "source_path",
    "device_source",
    "source_files",
    "sources",
)
EVENT_SCHEMA_TOKEN_RE = re.compile(
    r"\btraceEvents\b|\bdisplayTimeUnit\b|\bstart_ns\b|\bdur_ns\b|\bend_ns\b"
    r"|['\"](?:events|ph|tid|pid|cat|args)['\"]"
)
MAX_JSON_REFS = 12
MIN_EVENT_SCHEMA_OVERLAP = 2
EVENT_CONTAINER_FIELDS = frozenset({"events", "traceEvents"})

RECORD_PATH_WINDOW = 25
TAG_WINDOW = 15

CORE_PIDS = (
    "clock_globaltimer",
    "fixed_slot_formula",
    "fixed_slot_cute_record",
    "profiler_struct",
    "profiler_buf_cute",
    "profiler_enum_task",
    "profiler_buf_symbol",
    "profiler_max_records",
    "switch_false_specialization",
)

RANK_SYNC_PIDS = ("rank_sync_named", "rank_sync_nvlink", "rank_sync_grid")

KERNEL_ENTRY_RE = re.compile(r"\b__global__\b|\bCUTLASS_GLOBAL\b|\b__launch_bounds__\s*\(")
SYNC_DEFINITION_RE = re.compile(
    r"\b(?:void|auto)\s+(?:grid_sync|nvlink_barrier|\w*rank_sync\w*)\s*\("
)
BUSINESS_WORK_RE = re.compile(
    r"\bfor\s*\(|\bwhile\s*\(|\bdo\s*\{"
    r"|\b(?:tma_load|tma_store|tma_copy|cp_async|wgmma|warpgroup_arrive|warpgroup_wait)\w*\s*[(<]"
    r"|\bcute::(?:copy|gemm)\b"
    r"|^\s*#\s*include\b.*\boperations\b"
    r"|\b(?:dispatch|combine|epilogue|routing|reduce|scatter|gather|loader)_\w*\s*\("
    r"|\batomicAdd\w*\s*\(|\bptx::atomic_add\w*\s*\("
)
ENTRY_PROLOGUE_MIN_LINES = 96
ENTRY_PROLOGUE_FRACTION = 0.25
FALSE_SPEC_MAX_BODY_LINES = 12
MIN_SYNC_SITE_EVIDENCE = 16

ENABLE_FLAG_NAME_RE = re.compile(r"\b(k\w*[Ee]nable[Pp]rofiler\w*)\b")
PP_SWITCH_NAME_RE = re.compile(
    r"#\s*if(?:n?def)?\s+(?:defined\s*\(\s*)?(\w*(?:PROFIL|INSTRUMENT|GPTRACE)\w*)"
)
BUILD_DEFINE_NAME_RE = re.compile(r"-D\s*(\w*(?:PROFILER|PROFILE|INSTRUMENT)\w*)")
TEMPLATE_HEAD_RE = re.compile(r"\btemplate\s*<")
TEMPLATE_BOOL_PARAM_RE = re.compile(r"\bbool\s+(k?\w*[Ee]nabl\w*)")
TEMPLATE_CONTEXT_WINDOW = 4
STRONG_SWITCH_RE = re.compile(r"profil|instrument|gptrace", re.IGNORECASE)
PROFILER_CONTEXT_RE = re.compile(r"profil|instrument|trace", re.IGNORECASE)

HW_KERNEL_LAUNCH_RE = re.compile(
    r"<<<"
    r"|\b(?:launch(?:_kernel)?|run_kernel|kernel_launch|fp8_mega_moe|mega_moe)\s*\("
    r"|\btorch\s*\.\s*ops\s*\."
)
HW_CAPTURE_SYNC_RE = re.compile(
    r"\btorch\s*\.\s*cuda\s*\.\s*synchronize\s*\("
    r"|\bcudaDeviceSynchronize\s*\("
    r"|\b(?:dist\s*\.\s*)?barrier\s*\("
)
HW_SIMPLE_IF_RE = re.compile(r"^\s*if\s+(?:not\s+)?([A-Za-z_]\w*)\s*:")
HW_PROFILER_STATE_GUARD_RE = re.compile(
    r"^\s*if\s+(?:not\s+)?(?:profiler|capture|range)_?(?:started|active)\s*:"
)

IF_CONSTEXPR_RE = re.compile(r"\bif\s+constexpr\s*\(")
PP_DIRECTIVE_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")
PP_NEGATION_RE = re.compile(r"!|\bnot\b")
SPECIALIZATION_RE = re.compile(r"\b(?:struct|class)\s+(\w+)\s*<([^<>;{}]*)>")


def compile_specs():
    compiled = []
    for pid, pattern, scope, window in PATTERN_SPECS:
        compiled.append((pid, re.compile(pattern), scope, window))
    return compiled


def scope_matches(scope, ext):
    if scope == "any":
        return ext in SCAN_EXTS
    if scope == "device":
        return ext in DEVICE_EXTS
    if scope == "host":
        return ext in HOST_EXTS
    if scope == "decoder":
        return ext in DECODER_EXTS
    if scope == "pydevice":
        return ext == ".py"
    return False


def path_is_excluded(rel_dir, name, excludes):
    candidate = os.path.join(rel_dir, name) if rel_dir not in ("", ".") else name
    normalized = candidate.replace(os.sep, "/")
    for pattern in excludes:
        cleaned = pattern.replace(os.sep, "/").strip("/")
        if not cleaned:
            continue
        if "/" in cleaned or "*" in cleaned or "?" in cleaned:
            if fnmatch.fnmatch(normalized, cleaned) or normalized.startswith(cleaned + "/"):
                return True
        elif name == cleaned:
            return True
    return False


def iter_files(root, prune_dirs, excludes, max_bytes, include_hidden, stats):
    prune = set(prune_dirs)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""
        kept = []
        for name in dirnames:
            if name in prune:
                stats["pruned_dirs"] += 1
                continue
            if not include_hidden and name.startswith("."):
                stats["pruned_dirs"] += 1
                continue
            if path_is_excluded(rel_dir, name, excludes):
                stats["excluded_dirs"] += 1
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext not in SCAN_EXTS:
                continue
            if not include_hidden and name.startswith("."):
                continue
            if path_is_excluded(rel_dir, name, excludes):
                continue
            abs_path = os.path.join(dirpath, name)
            try:
                if os.path.islink(abs_path) or not os.path.isfile(abs_path):
                    continue
                size = os.path.getsize(abs_path)
            except OSError:
                continue
            if size > max_bytes:
                stats["skipped_large"] += 1
                continue
            rel_path = os.path.join(rel_dir, name) if rel_dir else name
            yield abs_path, rel_path.replace(os.sep, "/"), ext


def read_lines(abs_path, stats):
    try:
        with open(abs_path, "rb") as handle:
            data = handle.read()
    except OSError:
        return None
    if b"\x00" in data[:4096]:
        stats["skipped_binary"] += 1
        return None
    return data.decode("utf-8", "replace").splitlines()


def joined_window(lines, index, window):
    parts = lines[index:index + window]
    offsets = []
    position = 0
    for part in parts:
        offsets.append(position)
        position += len(part) + 1
    return " ".join(parts), offsets


def resolve_offset(index, offsets, match_start):
    resolved = index
    for order, offset in enumerate(offsets):
        if offset <= match_start:
            resolved = index + order
        else:
            break
    return resolved


_MASK_TOKEN_TYPES = {tokenize.COMMENT, tokenize.STRING}
for _fstr_name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
    _fstr_type = getattr(tokenize, _fstr_name, None)
    if _fstr_type is not None:
        _MASK_TOKEN_TYPES.add(_fstr_type)


def python_code_mask(lines):
    """Blank Python comments and string/docstring literals to spaces (line
    structure preserved) so device-signal regexes match real code rather than
    ABI prose documented in docstrings/comments.  Returns a copy of the original
    lines unchanged when the source cannot be tokenized."""
    try:
        masked = list(lines)
        readline = io.StringIO("\n".join(lines)).readline
        for tok in tokenize.generate_tokens(readline):
            if tok.type not in _MASK_TOKEN_TYPES:
                continue
            (srow, scol), (erow, ecol) = tok.start, tok.end
            for row in range(srow, erow + 1):
                idx = row - 1
                if idx < 0 or idx >= len(masked):
                    continue
                line = masked[idx]
                start = scol if row == srow else 0
                end = ecol if row == erow else len(line)
                start = max(0, min(start, len(line)))
                end = max(start, min(end, len(line)))
                masked[idx] = line[:start] + (" " * (end - start)) + line[end:]
        return masked
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return list(lines)


def scan_file(lines, ext, compiled):
    hits = {}
    raw_cache = {}
    code_cache = {}
    code_lines = None
    is_python = ext == ".py"

    def find(regex, window, source_lines, cache):
        # Match against source_lines but always report evidence text from the
        # original (unmasked) lines so the display stays readable.
        if window <= 1:
            found = []
            for lineno, line in enumerate(source_lines, start=1):
                if regex.search(line):
                    found.append((lineno, lines[lineno - 1].strip()[:180]))
            return found
        resolved_lines = set()
        for index in range(len(source_lines)):
            entry = cache.get(index)
            if entry is None:
                entry = joined_window(source_lines, index, window)
                cache[index] = entry
            text, offsets = entry
            match = regex.search(text)
            if match:
                resolved_lines.add(resolve_offset(index, offsets, match.start()))
        return [(line_index + 1, lines[line_index].strip()[:180]) for line_index in sorted(resolved_lines)]

    for pid, regex, scope, window in compiled:
        if not scope_matches(scope, ext):
            continue
        # CuTe-DSL device signals live in Python *code*, not in ABI documentation.
        # For .py device/pydevice scopes match against a comment/string-masked copy
        # so a host decoder that documents "mBuf[...] = %globaltimer" in a docstring
        # cannot spuriously acquire device capabilities.  Every other scope and all
        # non-Python files keep the exact original raw-line behavior.
        if is_python and scope in ("device", "pydevice"):
            if code_lines is None:
                code_lines = python_code_mask(lines)
            source_lines = code_lines
            cache = code_cache.setdefault(window, {})
        else:
            source_lines = lines
            cache = raw_cache.setdefault(window, {})
        found = find(regex, window, source_lines, cache)
        if found:
            hits[pid] = found
    return hits


def collect_shifts(lines, anchors, regex, window):
    amounts = set()
    for lineno in anchors:
        start = lineno - 1
        for line in lines[start:start + window]:
            for value in regex.findall(line):
                amounts.add(int(value))
    return amounts


def line_numbers(hits, pid):
    return [lineno for lineno, _ in hits.get(pid, ())]


def near(line_numbers_a, line_numbers_b, window):
    for a in line_numbers_a:
        for b in line_numbers_b:
            if abs(a - b) <= window:
                return True
    return False


def json_path_literals(lines, limit):
    refs = []
    seen = set()
    for lineno, line in enumerate(lines, start=1):
        for match in JSON_PATH_LITERAL_RE.finditer(line):
            literal = match.group(1)
            if literal in seen:
                continue
            seen.add(literal)
            refs.append((lineno, literal))
            if len(refs) >= limit:
                return refs
    return refs


def event_schema_tokens(lines):
    tokens = set()
    for line in lines:
        for match in EVENT_SCHEMA_TOKEN_RE.finditer(line):
            tokens.add(match.group(0).strip("'\""))
    return tokens


def as_string_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str)]
    return []


def ancestor_joins(literal, base_rel):
    candidates = [literal]
    base_dir = os.path.dirname(base_rel)
    while base_dir:
        candidates.append(base_dir + "/" + literal)
        base_dir = os.path.dirname(base_dir)
    out = []
    for candidate in candidates:
        normalized = os.path.normpath(candidate).replace(os.sep, "/")
        if normalized.startswith("..") or normalized.startswith("/"):
            continue
        if normalized not in out:
            out.append(normalized)
    return out


def coupling_rank(item):
    return (-len(item["coupling_routes"]), item["file"])


def sanitize_source(text):
    out = list(text)
    index = 0
    total = len(text)
    state = None
    while index < total:
        char = text[index]
        if state is None:
            if char == "/" and index + 1 < total and text[index + 1] == "/":
                out[index] = " "
                out[index + 1] = " "
                state = "line"
                index += 2
                continue
            if char == "/" and index + 1 < total and text[index + 1] == "*":
                out[index] = " "
                out[index + 1] = " "
                state = "block"
                index += 2
                continue
            if char == '"':
                out[index] = " "
                state = "string"
                index += 1
                continue
            if char == "'":
                out[index] = " "
                state = "char"
                index += 1
                continue
            index += 1
            continue
        if state == "line":
            if char == "\n":
                state = None
            else:
                out[index] = " "
            index += 1
            continue
        if state == "block":
            if char == "*" and index + 1 < total and text[index + 1] == "/":
                out[index] = " "
                out[index + 1] = " "
                state = None
                index += 2
                continue
            if char != "\n":
                out[index] = " "
            index += 1
            continue
        if char == "\\" and index + 1 < total:
            out[index] = " "
            if text[index + 1] != "\n":
                out[index + 1] = " "
            index += 2
            continue
        if char == "\n":
            state = None
            index += 1
            continue
        if (state == "string" and char == '"') or (state == "char" and char == "'"):
            out[index] = " "
            state = None
            index += 1
            continue
        out[index] = " "
        index += 1
    return "".join(out)


def line_offsets(text):
    offsets = [0]
    for index, char in enumerate(text):
        if char == "\n":
            offsets.append(index + 1)
    return offsets


def pos_to_line(offsets, position):
    return bisect.bisect_right(offsets, position)


def matching_brace(text, open_pos):
    depth = 0
    for position in range(open_pos, len(text)):
        char = text[position]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return position
    return len(text) - 1


def matching_paren(text, open_pos):
    depth = 0
    for position in range(open_pos, len(text)):
        char = text[position]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return position
    return -1


def kernel_entry_spans(sanitized, offsets):
    spans = []
    seen = set()
    for match in KERNEL_ENTRY_RE.finditer(sanitized):
        depth = 0
        body_open = -1
        for position in range(match.end(), len(sanitized)):
            char = sanitized[position]
            if char == "(":
                depth += 1
            elif char == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and char == ";":
                break
            elif depth == 0 and char == "{":
                body_open = position
                break
        if body_open < 0 or body_open in seen:
            continue
        seen.add(body_open)
        close = matching_brace(sanitized, body_open)
        spans.append(
            {
                "entry_line": pos_to_line(offsets, match.start()),
                "body_line": pos_to_line(offsets, body_open),
                "end_line": pos_to_line(offsets, close),
            }
        )
    return spans


def switch_reference(condition, switch_names):
    for name in sorted(switch_names, key=len, reverse=True):
        if re.search(r"\b" + re.escape(name) + r"\b", condition):
            return name
    return None


def if_constexpr_guard_spans(sanitized, offsets, switch_names):
    spans = []
    for match in IF_CONSTEXPR_RE.finditer(sanitized):
        paren_open = sanitized.find("(", match.start())
        if paren_open < 0:
            continue
        paren_close = matching_paren(sanitized, paren_open)
        if paren_close < 0:
            continue
        condition = sanitized[paren_open + 1:paren_close]
        name = switch_reference(condition, switch_names)
        if name is None:
            continue
        if re.search(r"(?:!|\bnot\b)\s*(?:\(\s*)?" + re.escape(name), condition):
            continue
        cursor = paren_close + 1
        while cursor < len(sanitized) and sanitized[cursor] in " \t\r\n":
            cursor += 1
        if cursor < len(sanitized) and sanitized[cursor] == "{":
            end = matching_brace(sanitized, cursor)
        else:
            end = sanitized.find(";", cursor)
            if end < 0:
                end = len(sanitized) - 1
        spans.append(
            {
                "start": pos_to_line(offsets, paren_open),
                "end": pos_to_line(offsets, end),
                "switch": name,
                "kind": "if_constexpr",
            }
        )
    return spans


def preprocessor_guard_spans(sanitized_lines, switch_names):
    spans = []
    stack = []

    def close_branch(frame, lineno):
        if frame["positive"] and frame["branch_start"] <= lineno - 1:
            spans.append(
                {
                    "start": frame["branch_start"],
                    "end": lineno - 1,
                    "switch": frame["name"],
                    "kind": "preprocessor_" + frame["kind"],
                }
            )

    for lineno, line in enumerate(sanitized_lines, start=1):
        match = PP_DIRECTIVE_RE.match(line)
        if not match:
            continue
        kind, rest = match.group(1), match.group(2)
        if kind in ("if", "ifdef", "ifndef"):
            name = switch_reference(rest, switch_names)
            positive = bool(name) and kind != "ifndef" and not PP_NEGATION_RE.search(rest)
            stack.append(
                {
                    "name": name,
                    "kind": kind,
                    "positive": positive,
                    "branch_start": lineno + 1,
                    "opened_as": kind,
                }
            )
            continue
        if not stack:
            continue
        if kind in ("elif", "else"):
            frame = stack[-1]
            close_branch(frame, lineno)
            if kind == "elif":
                name = switch_reference(rest, switch_names)
                frame["positive"] = bool(name) and not PP_NEGATION_RE.search(rest)
                frame["name"] = name or frame["name"]
                frame["kind"] = "elif"
            elif frame["name"] and frame["opened_as"] == "ifndef":
                frame["positive"] = True
                frame["kind"] = "ifndef_else"
            else:
                frame["positive"] = False
            frame["branch_start"] = lineno + 1
            continue
        frame = stack.pop()
        close_branch(frame, lineno)
    return spans


def statement_line_count(sanitized, open_pos, close_pos):
    body = sanitized[open_pos:close_pos]
    count = 0
    for raw in body.splitlines():
        stripped = raw.strip().strip("{}").strip()
        if stripped:
            count += 1
    return count


def specialization_switch_name(sanitized, struct_name, switch_names):
    pattern = re.compile(
        r"template\s*<([^<>]*)>\s*(?:struct|class)\s+" + re.escape(struct_name) + r"\b"
    )
    for match in pattern.finditer(sanitized):
        for bool_match in TEMPLATE_BOOL_PARAM_RE.finditer(match.group(1)):
            return bool_match.group(1)
    for name in sorted(switch_names):
        if STRONG_SWITCH_RE.search(name):
            return name
    return struct_name + "<true>"


def true_specialization_guard_spans(sanitized, offsets, switch_names):
    bodies = {}
    for match in SPECIALIZATION_RE.finditer(sanitized):
        struct_name = match.group(1)
        args = [item.strip() for item in match.group(2).split(",")]
        if "true" in args:
            polarity = True
        elif "false" in args:
            polarity = False
        else:
            continue
        brace = sanitized.find("{", match.end())
        semi = sanitized.find(";", match.end())
        if brace < 0 or (0 <= semi < brace):
            continue
        bodies.setdefault((struct_name, polarity), (brace, matching_brace(sanitized, brace)))
    spans = []
    for (struct_name, polarity), (brace, close) in bodies.items():
        if not polarity:
            continue
        stub = bodies.get((struct_name, False))
        if stub is None:
            continue
        if statement_line_count(sanitized, stub[0], stub[1]) > FALSE_SPEC_MAX_BODY_LINES:
            continue
        name = specialization_switch_name(sanitized, struct_name, switch_names)
        if name not in switch_names and not PROFILER_CONTEXT_RE.search(struct_name):
            continue
        spans.append(
            {
                "start": pos_to_line(offsets, brace),
                "end": pos_to_line(offsets, close),
                "switch": name,
                "kind": "template_bool_true_specialization",
            }
        )
    return spans


def guard_spans(sanitized, sanitized_lines, offsets, switch_names):
    if not switch_names:
        return []
    spans = if_constexpr_guard_spans(sanitized, offsets, switch_names)
    spans += preprocessor_guard_spans(sanitized_lines, switch_names)
    spans += true_specialization_guard_spans(sanitized, offsets, switch_names)
    return spans


def innermost_guard(spans, lineno):
    best = None
    for span in spans:
        if span["start"] <= lineno <= span["end"]:
            if best is None or span["start"] > best["start"] or (
                span["start"] == best["start"] and span["end"] < best["end"]
            ):
                best = span
    return best


def entry_alignment_verdict(sanitized_lines, entries, lineno):
    if not entries:
        return False, "no_kernel_entry_marker_in_file"
    holder = None
    for entry in entries:
        if entry["body_line"] <= lineno <= entry["end_line"]:
            if holder is None or entry["body_line"] > holder["body_line"]:
                holder = entry
    if holder is None:
        return False, "outside_any_kernel_entry_body"
    for probe in range(holder["body_line"], lineno):
        if BUSINESS_WORK_RE.search(sanitized_lines[probe - 1]):
            return False, "business_work_at_line_{0}_precedes_this_barrier".format(probe)
    budget = max(
        ENTRY_PROLOGUE_MIN_LINES,
        int((holder["end_line"] - holder["body_line"]) * ENTRY_PROLOGUE_FRACTION),
    )
    offset = lineno - holder["body_line"]
    if offset > budget:
        return False, "offset_{0}_lines_into_kernel_body_exceeds_entry_prologue_budget_{1}".format(
            offset, budget
        )
    return True, "entry_prologue_of_kernel_at_line_{0}_no_business_work_before".format(
        holder["entry_line"]
    )


def switch_templated_dispatch(sanitized_lines, lineno, switch_names):
    window = " ".join(sanitized_lines[lineno - 1:lineno + 2])
    for match in re.finditer(r"<([^<>]*)>", window):
        name = switch_reference(match.group(1), switch_names)
        if name and not PP_NEGATION_RE.search(match.group(1)):
            return {"start": lineno, "end": lineno, "switch": name, "kind": "switch_templated_dispatch"}
    return None


class Detector:
    def __init__(self, args):
        self.root = os.path.abspath(os.path.expanduser(args.root))
        self.args = args
        self.compiled = compile_specs()
        self.excludes = list(args.exclude_list)
        self.stats = {
            "scanned_files": 0,
            "matched_files": 0,
            "skipped_large": 0,
            "skipped_binary": 0,
            "pruned_dirs": 0,
            "excluded_dirs": 0,
        }
        self.files = []
        self.aggregate = {}
        self.device_tag_shifts = set()
        self.device_tag_masks = set()
        self.num_tracks = None
        self.profiler_type_names = set()
        self.switch_names = set()
        self.rank_sync_summary = {}
        self.abi_json_files = []
        self.json_by_rel = {}
        self.json_index = {}
        self.json_docs = {}
        self.self_path = os.path.realpath(__file__)

    def run(self):
        self.scan_tree()
        self.scan_abi_json()
        capabilities, evidence, signals = self.derive()
        required_capabilities = list(CAPABILITY_ORDER)
        if self.args.require_hw_capture_range:
            required_capabilities.append(HW_CAPTURE_RANGE_CAPABILITY)
        verdict, path, reason = self.classify(capabilities, signals, required_capabilities)
        missing = [name for name in required_capabilities if not capabilities[name]]
        report = {
            "schema": SCHEMA,
            "root": self.root,
            "verdict": verdict,
            "confidence": self.confidence(verdict, capabilities, signals),
            "requirements": {
                "require_hw_capture_range": self.args.require_hw_capture_range,
                "required_capabilities": required_capabilities,
                "hw_capture_range_hard_gate_exit_code": 4,
            },
            "capabilities": {
                name: capabilities[name]
                for name in CAPABILITY_ORDER + (HW_CAPTURE_RANGE_CAPABILITY,)
            },
            "evidence": evidence,
            "missing": missing,
            "overhead_measurement_caveat": self.overhead_caveat(capabilities),
            "warnings": self.warnings(capabilities),
            "clock": self.clock_block(),
            "next_action": {
                "path": path,
                "reason": reason,
                "commands": self.commands(verdict, path, capabilities, signals),
            },
            "transport": self.transport_block(),
            "signals": signals,
            "scan": self.stats,
        }
        return report

    def scan_tree(self):
        for abs_path, rel_path, ext in iter_files(
            self.root,
            self.args.prune_list,
            self.excludes,
            self.args.max_bytes,
            self.args.include_hidden,
            self.stats,
        ):
            if os.path.realpath(abs_path) == self.self_path:
                continue
            lines = read_lines(abs_path, self.stats)
            if lines is None:
                continue
            self.stats["scanned_files"] += 1
            hits = scan_file(lines, ext, self.compiled)
            self.harvest_switch_names(lines)
            right_shifts = set()
            json_refs = []
            schema_tokens = set()
            if ext in DECODER_EXTS:
                for line in lines:
                    for value in SHIFT_RIGHT_RE.findall(line):
                        right_shifts.add(int(value))
                json_refs = json_path_literals(lines, MAX_JSON_REFS)
                schema_tokens = event_schema_tokens(lines)
            if not hits and not right_shifts and not json_refs:
                continue
            self.stats["matched_files"] += 1
            record = {
                "rel": rel_path,
                "abs": abs_path,
                "ext": ext,
                "hits": hits,
                "right_shifts": right_shifts,
                "json_refs": json_refs,
                "event_schema_tokens": schema_tokens,
                "lines": len(lines),
            }
            self.files.append(record)
            for pid in hits:
                self.aggregate.setdefault(pid, []).append(record)
            self.harvest_device_details(lines, ext, hits)

    def harvest_switch_names(self, lines):
        for line in lines:
            for regex in (ENABLE_FLAG_NAME_RE, PP_SWITCH_NAME_RE, BUILD_DEFINE_NAME_RE):
                for match in regex.finditer(line):
                    self.switch_names.add(match.group(1))
        for index, line in enumerate(lines):
            if not TEMPLATE_HEAD_RE.search(line):
                continue
            window = " ".join(lines[index:index + TEMPLATE_CONTEXT_WINDOW])
            profiler_context = bool(PROFILER_CONTEXT_RE.search(window))
            for match in TEMPLATE_BOOL_PARAM_RE.finditer(window):
                name = match.group(1)
                if STRONG_SWITCH_RE.search(name) or profiler_context:
                    self.switch_names.add(name)

    def harvest_device_details(self, lines, ext, hits):
        for pid in ("profiler_struct", "profiler_enum_task"):
            for _, text in hits.get(pid, ()):
                for _pid, regex, _scope, _window in self.compiled:
                    if _pid != pid:
                        continue
                    found = regex.search(text)
                    if found and found.groups():
                        self.profiler_type_names.add(found.group(1))
        if self.num_tracks is None:
            for line in lines:
                found = NUM_TRACKS_VALUE_RE.search(line)
                if found:
                    self.num_tracks = int(found.group(1))
                    break
        if ext not in DEVICE_EXTS:
            return
        anchors = [i + 1 for i, line in enumerate(lines) if TAG_ASSIGN_RE.search(line)]
        if not anchors:
            anchors = (line_numbers(hits, "clock_globaltimer") + line_numbers(hits, "clock_smemrealtime")
                        + line_numbers(hits, "clock_clock64"))
        if not anchors:
            return
        self.device_tag_shifts |= collect_shifts(lines, anchors, SHIFT_LEFT_RE, TAG_WINDOW)
        for lineno in anchors:
            for line in lines[lineno - 1:lineno - 1 + TAG_WINDOW]:
                for value in MASK_RE.findall(line):
                    self.device_tag_masks.add(value.lower())

    def scan_abi_json(self):
        for abs_path, rel_path, ext in iter_files(
            self.root,
            self.args.prune_list,
            self.excludes,
            self.args.max_bytes,
            self.args.include_hidden,
            {"skipped_large": 0, "skipped_binary": 0, "pruned_dirs": 0, "excluded_dirs": 0},
        ):
            if ext != ".json":
                continue
            self.json_by_rel[rel_path] = abs_path
            self.json_index.setdefault(os.path.basename(rel_path).lower(), []).append(rel_path)
            name = os.path.basename(rel_path).lower()
            if not any(fnmatch.fnmatch(name, pattern) for pattern in ABI_JSON_GLOBS):
                continue
            try:
                with open(abs_path, "rb") as handle:
                    blob = handle.read(self.args.max_bytes).decode("utf-8", "replace")
            except OSError:
                continue
            lowered = blob.lower()
            tokens = [token for token in ABI_COUPLING_TOKENS if token in lowered]
            tokens += [name for name in self.profiler_type_names if name.lower() in lowered]
            try:
                document = json.loads(blob)
            except ValueError as error:
                semantic_errors = ["invalid JSON: {}".format(error)]
            else:
                semantic_errors = validate_semantic_contract(
                    document, base_dir=self.root, verify_sources=True)
            self.abi_json_files.append(
                {"file": rel_path, "coupling_tokens": sorted(set(tokens)),
                 "coupled": len(set(tokens)) >= 2,
                 "semantic_contract_valid": not semantic_errors,
                 "semantic_contract_errors": semantic_errors[:8]}
            )

    def profiler_source_anchors(self, fixed_files):
        anchors = {}
        fixed = set(fixed_files)
        for record in self.files:
            if record["ext"] not in DEVICE_EXTS:
                continue
            hits = record["hits"]
            stamp_lines = (line_numbers(hits, "clock_globaltimer") + line_numbers(hits, "clock_smemrealtime")
                        + line_numbers(hits, "clock_clock64"))
            if not stamp_lines:
                continue
            identity = []
            if "profiler_struct" in hits:
                identity.append("profiler_struct")
            if "profiler_enum_task" in hits:
                identity.append("profiler_enum_task")
            if record["rel"] in fixed:
                identity.append("fixed_slot_record_path")
            if not identity:
                continue
            anchors[record["rel"]] = {
                "device_timestamp_line": min(stamp_lines),
                "profiler_identity": identity,
            }
        return anchors

    def resolve_json_ref(self, literal, base_rel):
        cleaned = literal.replace("\\", "/").strip()
        if not cleaned or cleaned.startswith("/"):
            return None
        if cleaned.startswith("./"):
            cleaned = cleaned[2:]
        for candidate in ancestor_joins(cleaned, base_rel):
            if candidate in self.json_by_rel:
                return candidate
        if "/" not in cleaned:
            matches = self.json_index.get(cleaned.lower(), ())
            if len(matches) == 1:
                return matches[0]
        return None

    def load_json_doc(self, rel):
        if rel in self.json_docs:
            return self.json_docs[rel]
        doc = None
        abs_path = self.json_by_rel.get(rel)
        if abs_path:
            try:
                with open(abs_path, "rb") as handle:
                    blob = handle.read(self.args.max_bytes)
                doc = json.loads(blob.decode("utf-8", "replace"))
            except (OSError, ValueError):
                doc = None
        self.json_docs[rel] = doc
        return doc

    def declared_source_values(self, doc):
        if not isinstance(doc, dict):
            return []
        found = []
        for field in ABI_SOURCE_FIELDS:
            for value in as_string_list(doc.get(field)):
                found.append((field, value))
        if found:
            return found
        for key, value in doc.items():
            if not isinstance(value, dict):
                continue
            for field in ABI_SOURCE_FIELDS:
                for item in as_string_list(value.get(field)):
                    found.append(("{0}.{1}".format(key, field), item))
        return found

    def match_profiler_anchor(self, value, anchors, base_rel):
        cleaned = value.replace("\\", "/").strip().split(":")[0].strip().lstrip("/")
        if cleaned.startswith("./"):
            cleaned = cleaned[2:]
        if not cleaned:
            return None
        for candidate in ancestor_joins(cleaned, base_rel):
            if candidate in anchors:
                return candidate
        basename = os.path.basename(cleaned).lower()
        matches = [rel for rel in anchors if os.path.basename(rel).lower() == basename]
        if len(matches) == 1:
            return matches[0]
        return None

    def provenance_chain(self, record, anchors):
        if not anchors:
            return None
        for lineno, literal in record["json_refs"]:
            abi_rel = self.resolve_json_ref(literal, record["rel"])
            if abi_rel is None:
                continue
            doc = self.load_json_doc(abi_rel)
            if doc is None:
                continue
            for field, value in self.declared_source_values(doc):
                anchor_rel = self.match_profiler_anchor(value, anchors, abi_rel)
                if anchor_rel is None:
                    continue
                anchor = anchors[anchor_rel]
                return {
                    "decoder": record["rel"],
                    "decoder_line": lineno,
                    "decoder_json_literal": literal,
                    "abi_json": abi_rel,
                    "abi_source_field": field,
                    "abi_source_value": value,
                    "profiler_source": anchor_rel,
                    "profiler_identity": anchor["profiler_identity"],
                    "profiler_device_timestamp_line": anchor["device_timestamp_line"],
                    "chain": "{decoder}:{line} -> {abi} -> {field}={value} -> {profiler}".format(
                        decoder=record["rel"],
                        line=lineno,
                        abi=abi_rel,
                        field=field,
                        value=value,
                        profiler=anchor_rel,
                    ),
                }
        return None

    def any_hit(self, pid):
        return bool(self.aggregate.get(pid))

    def rank_records(self, records):
        def score(record):
            return -sum(1 for pid in CORE_PIDS if pid in record["hits"])

        return sorted(records, key=score)

    def first_evidence(self, pid, limit, only_files=None):
        out = []
        for record in self.rank_records(self.aggregate.get(pid, ())):
            if only_files is not None and record["rel"] not in only_files:
                continue
            for lineno, text in record["hits"][pid]:
                out.append({"file": record["rel"], "line": lineno, "text": text})
                if len(out) >= limit:
                    return out
        return out

    def analyze_hw_capture_range(self):
        """Conservatively prove that one harness exposes a kernel-only HW range.

        This is deliberately an existence/control-flow fingerprint, not a Python
        parser.  A qualifying file must contain the CLI declaration, one complete
        profiler API family, an ordered Start -> launch -> synchronize -> Stop
        range, and symmetric Start/Stop guards.  Extra Stop calls are accepted
        only when they are visibly guarded cleanup in a finally block.
        """

        api_families = (
            ("cuda_runtime", "hw_cuda_profiler_start", "hw_cuda_profiler_stop"),
            ("torch_cuda", "hw_torch_profiler_start", "hw_torch_profiler_stop"),
        )

        def nearest_simple_guard(lines, lineno, window=10):
            start_index = max(0, lineno - 1 - window)
            for index in range(lineno - 2, start_index - 1, -1):
                match = HW_SIMPLE_IF_RE.match(lines[index])
                if match:
                    return match.group(1)
                stripped = lines[index].lstrip()
                if stripped.startswith(("def ", "async def ", "class ")):
                    break
            return None

        def cleanup_stop(lines, lineno, window=10):
            start_index = max(0, lineno - 1 - window)
            context = lines[start_index:lineno - 1]
            finally_lines = [
                start_index + index + 1
                for index, text in enumerate(context)
                if re.match(r"^\s*finally\s*:", text)
            ]
            state_guard_lines = [
                start_index + index + 1
                for index, text in enumerate(context)
                if HW_PROFILER_STATE_GUARD_RE.match(text)
            ]
            return {
                "line": lineno,
                "guarded_cleanup": bool(finally_lines and state_guard_lines),
                "finally_lines": finally_lines,
                "state_guard_lines": state_guard_lines,
            }

        candidates = []
        qualifying_files = []
        for record in self.files:
            if record["ext"] not in HOST_EXTS:
                continue
            hits = record["hits"]
            cli_lines = sorted(set(line_numbers(hits, "hw_capture_cli")))
            has_api_signal = any(
                line_numbers(hits, start_pid) or line_numbers(hits, stop_pid)
                for _, start_pid, stop_pid in api_families
            )
            if not cli_lines and not has_api_signal:
                continue

            scratch = {"skipped_binary": 0}
            lines = read_lines(record["abs"], scratch)
            if lines is None:
                continue

            family_results = []
            for family_name, start_pid, stop_pid in api_families:
                start_lines = sorted(set(line_numbers(hits, start_pid)))
                stop_lines = sorted(set(line_numbers(hits, stop_pid)))
                if not start_lines and not stop_lines:
                    continue

                unused_stop_lines = list(stop_lines)
                pairs = []
                unmatched_start_lines = []
                for start_line in start_lines:
                    stop_line = next((line for line in unused_stop_lines if line > start_line), None)
                    if stop_line is None:
                        unmatched_start_lines.append(start_line)
                        continue
                    unused_stop_lines.remove(stop_line)

                    launch_lines = [
                        line
                        for line in range(start_line + 1, stop_line)
                        if HW_KERNEL_LAUNCH_RE.search(lines[line - 1])
                    ]
                    sync_lines = [
                        line
                        for line in range(start_line + 1, stop_line)
                        if HW_CAPTURE_SYNC_RE.search(lines[line - 1])
                    ]
                    first_launch = launch_lines[0] if launch_lines else None
                    pre_launch_sync_lines = (
                        [line for line in sync_lines if line < first_launch]
                        if first_launch is not None
                        else []
                    )
                    post_launch_sync_lines = (
                        [line for line in sync_lines if line > first_launch]
                        if first_launch is not None
                        else []
                    )
                    start_guard = nearest_simple_guard(lines, start_line)
                    stop_guard = nearest_simple_guard(lines, stop_line)
                    guard_symmetric = start_guard == stop_guard
                    pair_qualified = bool(
                        launch_lines and post_launch_sync_lines and guard_symmetric
                    )
                    pairs.append(
                        {
                            "start_line": start_line,
                            "stop_line": stop_line,
                            "start_guard": start_guard,
                            "stop_guard": stop_guard,
                            "guard_symmetric": guard_symmetric,
                            "kernel_launch_lines": launch_lines,
                            "pre_launch_sync_lines": pre_launch_sync_lines,
                            "post_launch_sync_lines": post_launch_sync_lines,
                            "kernel_only_ordered_range": pair_qualified,
                        }
                    )

                cleanup_stops = [cleanup_stop(lines, line) for line in unused_stop_lines]
                rejected_stop_lines = [
                    item["line"] for item in cleanup_stops if not item["guarded_cleanup"]
                ]
                balanced_control_flow = bool(
                    start_lines
                    and pairs
                    and not unmatched_start_lines
                    and not rejected_stop_lines
                    and len(pairs) == len(start_lines)
                )
                family_qualified = bool(
                    balanced_control_flow
                    and all(pair["kernel_only_ordered_range"] for pair in pairs)
                )
                family_results.append(
                    {
                        "api_family": family_name,
                        "start_count": len(start_lines),
                        "stop_count": len(stop_lines),
                        "start_lines": start_lines,
                        "stop_lines": stop_lines,
                        "ordered_pairs": pairs,
                        "unmatched_start_lines": unmatched_start_lines,
                        "extra_stop_cleanup": cleanup_stops,
                        "rejected_unmatched_stop_lines": rejected_stop_lines,
                        "balanced_control_flow": balanced_control_flow,
                        "qualified": family_qualified,
                    }
                )

            file_qualified = bool(
                cli_lines and any(item["qualified"] for item in family_results)
            )
            if file_qualified:
                qualifying_files.append(record["rel"])
            candidates.append(
                {
                    "file": record["rel"],
                    "harness_cli_lines": cli_lines,
                    "api_families": family_results,
                    "qualified": file_qualified,
                }
            )

        evidence = []
        limit = self.args.max_evidence
        for candidate in sorted(candidates, key=lambda item: (not item["qualified"], item["file"])):
            if len(evidence) >= limit:
                break
            if candidate["harness_cli_lines"]:
                evidence.append(
                    {
                        "file": candidate["file"],
                        "line": candidate["harness_cli_lines"][0],
                        "text": "harness CLI enables cudaProfilerApi capture range",
                    }
                )
            for family in candidate["api_families"]:
                if len(evidence) >= limit or not family["ordered_pairs"]:
                    continue
                pair = family["ordered_pairs"][0]
                evidence.append(
                    {
                        "file": candidate["file"],
                        "line": pair["start_line"],
                        "text": "{0} Start -> kernel launch -> synchronize -> Stop at line {1}; "
                        "guard_symmetric={2}; guarded_cleanup_stops={3}".format(
                            family["api_family"],
                            pair["stop_line"],
                            pair["guard_symmetric"],
                            sum(
                                1
                                for item in family["extra_stop_cleanup"]
                                if item["guarded_cleanup"]
                            ),
                        ),
                    }
                )

        return (
            {
                "present": bool(qualifying_files),
                "required_signals": [
                    "same-file harness CLI declaration",
                    "cudaProfilerStart/Stop or torch.cuda.profiler.start/stop",
                    "ordered Start -> kernel launch -> post-launch synchronization -> Stop",
                    "symmetric Start/Stop guards",
                    "no unmatched Start and only finally+state-guarded cleanup Stop",
                ],
                "qualifying_files": sorted(qualifying_files),
                "candidate_files": candidates,
            },
            evidence,
        )

    def analyze_rank_sync_sites(self):
        scratch = {"skipped_binary": 0}
        sites = []
        for record in self.files:
            if record["ext"] not in DEVICE_EXTS:
                continue
            by_line = {}
            for pid in RANK_SYNC_PIDS:
                for lineno, text in record["hits"].get(pid, ()):
                    slot = by_line.setdefault(lineno, {"text": text, "patterns": set()})
                    slot["patterns"].add(pid)
            if not by_line:
                continue
            lines = read_lines(record["abs"], scratch)
            if lines is None:
                continue
            sanitized = sanitize_source("\n".join(lines))
            sanitized_lines = sanitized.split("\n")
            offsets = line_offsets(sanitized)
            entries = kernel_entry_spans(sanitized, offsets)
            guards = guard_spans(sanitized, sanitized_lines, offsets, self.switch_names)
            for lineno in sorted(by_line):
                slot = by_line[lineno]
                if SYNC_DEFINITION_RE.search(slot["text"]):
                    continue
                guard = innermost_guard(guards, lineno)
                if guard is None:
                    guard = switch_templated_dispatch(sanitized_lines, lineno, self.switch_names)
                aligned, reason = entry_alignment_verdict(sanitized_lines, entries, lineno)
                sites.append(
                    {
                        "file": record["rel"],
                        "line": lineno,
                        "text": slot["text"],
                        "guarded_by": guard["switch"] if guard else None,
                        "is_entry_alignment": aligned,
                        "guard_kind": guard["kind"] if guard else None,
                        "guard_span": [guard["start"], guard["end"]] if guard else None,
                        "entry_alignment_reason": reason,
                        "patterns": sorted(slot["patterns"]),
                    }
                )
        sites.sort(key=lambda item: (not item["is_entry_alignment"], item["file"], item["line"]))
        return sites

    def derive(self):
        limit = self.args.max_evidence
        evidence = {}
        signals = {}

        def put(key, pids, only_files=None):
            items = []
            for pid in pids:
                for item in self.first_evidence(pid, limit, only_files):
                    if item not in items:
                        items.append(item)
                    if len(items) >= limit:
                        break
                if len(items) >= limit:
                    break
            if items:
                evidence[key] = items

        has_globaltimer = self.any_hit("clock_globaltimer")
        has_clock64 = self.any_hit("clock_clock64")
        has_smemrealtime = self.any_hit("clock_smemrealtime")
        # NV 只认 %globaltimer；AMD gfx9xx 认 s_memrealtime（64 位全局实时钟，
        # MI308X 实测跨 CU 同基线）。clock64() 在两侧都是 per-SM/per-CU 周期计数
        # 且随频率变化（AMD 侧实测跨 CU 相位差 >100ms），两个单元的读数之间没有
        # 可比性，拿它排跨单元时间线是错的。所以命中 clock64 不算通过，
        # 只作为「改动量小于从零插桩」的诊断线索。
        device_timestamping = has_globaltimer or has_smemrealtime
        put("device_timestamping", ("clock_globaltimer", "clock_smemrealtime", "clock_clock64", "clock_smid", "clock_amd_hwid"))

        fixed_ok = False
        contention = False
        fixed_files = []
        for record in self.files:
            if record["ext"] not in DEVICE_EXTS:
                continue
            hits = record["hits"]
            ts_lines = (line_numbers(hits, "clock_globaltimer") + line_numbers(hits, "clock_smemrealtime")
                        + line_numbers(hits, "clock_clock64"))
            if not ts_lines:
                continue
            slot_lines = (
                line_numbers(hits, "fixed_slot_formula")
                or line_numbers(hits, "fixed_slot_formula_generic")
                or line_numbers(hits, "fixed_slot_cute_record")
            )
            atomic_lines = line_numbers(hits, "atomic_add")
            record_path_atomic = near(atomic_lines, ts_lines, RECORD_PATH_WINDOW)
            if record_path_atomic:
                contention = True
            if slot_lines and not record_path_atomic:
                fixed_ok = True
                fixed_files.append(record["rel"])
        fixed_slot_no_contention = fixed_ok
        put(
            "fixed_slot_no_contention",
            ("fixed_slot_formula", "fixed_slot_formula_generic", "fixed_slot_cute_record"),
            set(fixed_files) or None,
        )

        strong_switch = self.any_hit("switch_template_bool") and self.any_hit("switch_false_specialization")
        compile_time_switch = bool(
            strong_switch
            or self.any_hit("profiler_enable_flag")
            or self.any_hit("switch_macro_ifdef")
            or self.any_hit("switch_build_define")
        )
        put(
            "compile_time_switch",
            ("switch_template_bool", "switch_false_specialization", "profiler_enable_flag", "switch_macro_ifdef", "switch_build_define"),
        )

        anchor_names = set()
        for record in self.aggregate.get("rank_sync_anchor_name", ()):
            for _, text in record["hits"]["rank_sync_anchor_name"]:
                for anchor in SEVEN_ANCHORS:
                    if anchor in text:
                        anchor_names.add(anchor)
        profiler_coupled_barrier = False
        for pid in ("rank_sync_nvlink", "rank_sync_grid"):
            for record in self.aggregate.get(pid, ()):
                hits = record["hits"]
                if "profiler_enable_flag" in hits or "profiler_buf_symbol" in hits or "clock_globaltimer" in hits:
                    profiler_coupled_barrier = True
        rank_sync_sites = self.analyze_rank_sync_sites()
        rank_sync_anchor = bool(
            rank_sync_sites
            and (
                self.any_hit("rank_sync_named")
                or profiler_coupled_barrier
                or len(anchor_names) >= 3
            )
        )
        put("rank_sync_anchor", ("rank_sync_named", "rank_sync_anchor_name", "rank_sync_nvlink", "rank_sync_grid"))

        entry_sites = [site for site in rank_sync_sites if site["is_entry_alignment"]]
        baseline_entry_sites = [site for site in entry_sites if site["guarded_by"] is None]
        guarded_entry_sites = [site for site in entry_sites if site["guarded_by"] is not None]
        # 判据：sync 可以受插桩总开关控制（不插桩时不执行，无妨 —— 开销口径用
        # (C-B)/B，B 与 C 都开着插桩，都做 sync）；但不能受「记录量」开关控制，
        # 否则 B 配置不做 sync，B 与 C 就没有共同起跑线。
        # 总开关名含 PROFILER/kEnable*，记录量开关名含 max_records/records/verbose/fine。
        record_volume_guard = re.compile(
            r"max_?records|record_?count|n_?records|verbose|fine_?grain|detail", re.IGNORECASE
        )
        volume_guarded = [
            site for site in guarded_entry_sites
            if record_volume_guard.search(str(site.get("guarded_by") or ""))
        ]
        baseline_rank_sync = bool(
            rank_sync_anchor and entry_sites and not volume_guarded
        )
        if not rank_sync_sites:
            baseline_reason = "no_rank_sync_site_detected"
        elif not entry_sites:
            baseline_reason = "rank_sync_exists_but_none_is_a_kernel_entry_launch_alignment"
        elif baseline_entry_sites:
            baseline_reason = "entry_alignment_at_{0}:{1}_is_outside_any_instrumentation_switch".format(
                baseline_entry_sites[0]["file"], baseline_entry_sites[0]["line"]
            )
        else:
            baseline_reason = "every_entry_alignment_is_guarded_by_{0}".format(
                ",".join(sorted({site["guarded_by"] for site in guarded_entry_sites}))
            )
        if rank_sync_sites:
            evidence["baseline_rank_sync"] = rank_sync_sites[: max(limit, MIN_SYNC_SITE_EVIDENCE)]
        self.rank_sync_summary = {
            "sites": rank_sync_sites,
            "entry_alignment_sites": entry_sites,
            "baseline_entry_sites": baseline_entry_sites,
            "guarded_entry_sites": guarded_entry_sites,
            "reason": baseline_reason,
        }

        buffer_allocation = False
        wiring_files = []
        for record in self.files:
            if record["ext"] not in HOST_EXTS:
                continue
            hits = record["hits"]
            if not ("profiler_buf_symbol" in hits or "profiler_max_records" in hits):
                continue
            if "host_buffer_validate" in hits or "host_python_param" in hits:
                buffer_allocation = True
                wiring_files.append(record["rel"])
        put(
            "buffer_allocation",
            ("host_python_param", "host_buffer_validate", "profiler_buf_symbol"),
            set(wiring_files) or None,
        )

        task_definition = self.any_hit("task_enum_def") or self.any_hit("profiler_enum_task")
        phase_definition = self.any_hit("phase_enum_def")
        task_phase_definition = task_definition and phase_definition
        put("task_phase_definition", ("task_enum_def", "phase_enum_def", "profiler_enum_task"))

        semantic_abi = [entry for entry in self.abi_json_files
                        if entry.get("semantic_contract_valid")]
        source_locked_semantic_contract = bool(semantic_abi)
        if self.abi_json_files:
            evidence["source_locked_semantic_contract"] = [
                {"file": entry["file"], "line": 1,
                 "text": ("source-locked semantic contract valid" if entry.get("semantic_contract_valid")
                          else "semantic contract invalid: " + "; ".join(
                              entry.get("semantic_contract_errors", ())))}
                for entry in self.abi_json_files[:limit]
            ]

        coupled_abi = [entry for entry in self.abi_json_files if entry["coupled"]]
        format_spec = bool(coupled_abi)
        if self.abi_json_files:
            evidence["format_spec"] = [
                {"file": entry["file"], "line": 1, "text": "coupling_tokens=" + ",".join(entry["coupling_tokens"])}
                for entry in self.abi_json_files[:limit]
            ]

        anchors = self.profiler_source_anchors(fixed_files)
        identity_tokens = set(self.profiler_type_names) | {"profiler_buf", "profiler_max_records"}
        decoder_hits = []
        perfetto_hits = []
        coupled_entries = []
        uncoupled = []
        for record in self.files:
            if record["ext"] not in DECODER_EXTS:
                continue
            hits = record["hits"]
            basename = os.path.basename(record["rel"]).lower()
            named_decoder = any(token in basename for token in ("decode", "perfetto", "trace"))
            has_bit_ops = "bitmask_hex" in hits or bool(record["right_shifts"])
            has_decode_intent = "decode_intent" in hits
            has_perfetto = "perfetto_intent" in hits
            if not (named_decoder or ((has_decode_intent or has_perfetto) and has_bit_ops)):
                continue
            matched_shifts = sorted(self.device_tag_shifts & record["right_shifts"])
            all_text = [text for pid_items in hits.values() for _, text in pid_items]
            references_identity = bool(
                "profiler_buf_symbol" in hits
                or "profiler_max_records" in hits
                or any(token in text for token in identity_tokens for text in all_text)
            )
            chain = self.provenance_chain(record, anchors)
            route_a = len(matched_shifts) >= 2
            route_b = references_identity and (has_decode_intent or has_perfetto or named_decoder) and has_bit_ops
            route_c = chain is not None
            coupled = route_a or route_b or route_c
            entry = {
                "file": record["rel"],
                "matched_tag_shifts": matched_shifts,
                "references_abi_identity": references_identity,
                "decode_intent": has_decode_intent or ("decode" in basename),
                "perfetto_intent": has_perfetto,
                "coupling_routes": [
                    name
                    for name, fired in (
                        ("a_tag_shift_match", route_a),
                        ("b_abi_identity_plus_bit_ops", route_b),
                        ("c_abi_json_provenance", route_c),
                    )
                    if fired
                ],
                "event_schema_fields": sorted(record["event_schema_tokens"]),
            }
            if chain is not None:
                entry["provenance"] = chain
            if coupled:
                coupled_entries.append(entry)
                if entry["decode_intent"]:
                    decoder_hits.append(entry)
                if has_perfetto:
                    perfetto_hits.append(entry)
            else:
                entry["named_decoder"] = named_decoder
                entry["has_bit_ops"] = has_bit_ops
                uncoupled.append(entry)
        uncoupled.sort(key=lambda item: (not item["named_decoder"], not item["has_bit_ops"], item["file"]))
        coupled_entries.sort(key=coupling_rank)
        decoder_hits.sort(key=coupling_rank)
        perfetto_hits.sort(key=coupling_rank)
        host_decoder = bool(decoder_hits)

        decoder_schema = set()
        for entry in decoder_hits:
            decoder_schema |= set(entry["event_schema_fields"])
        perfetto_chain = []
        if host_decoder and decoder_schema:
            for entry in uncoupled:
                if not entry["perfetto_intent"]:
                    continue
                shared = sorted(set(entry["event_schema_fields"]) & decoder_schema)
                if len(shared) < MIN_EVENT_SCHEMA_OVERLAP:
                    continue
                if not (set(shared) & EVENT_CONTAINER_FIELDS):
                    continue
                perfetto_chain.append(
                    {
                        "file": entry["file"],
                        "consumes_decoded_output_of": decoder_hits[0]["file"],
                        "shared_event_schema_fields": shared,
                    }
                )
        perfetto_chain.sort(key=lambda item: (-len(item["shared_event_schema_fields"]), item["file"]))
        perfetto_emitter = bool(perfetto_hits) or bool(perfetto_chain)

        if decoder_hits:
            evidence["host_decoder"] = [
                {
                    "file": item["file"],
                    "line": item["provenance"]["decoder_line"] if "provenance" in item else 1,
                    "text": "routes={0}; {1}".format(
                        "+".join(item["coupling_routes"]),
                        "provenance " + item["provenance"]["chain"]
                        if "provenance" in item
                        else "matched_tag_shifts=" + str(item["matched_tag_shifts"]),
                    ),
                }
                for item in decoder_hits[:limit]
            ]
        if perfetto_hits or perfetto_chain:
            emitter_evidence = [
                {
                    "file": item["file"],
                    "line": 1,
                    "text": "abi-coupled perfetto emitter; routes=" + "+".join(item["coupling_routes"]),
                }
                for item in perfetto_hits[:limit]
            ]
            for item in perfetto_chain:
                if len(emitter_evidence) >= limit:
                    break
                emitter_evidence.append(
                    {
                        "file": item["file"],
                        "line": 1,
                        "text": "consumes decoded events of {0}; shared event schema fields={1}".format(
                            item["consumes_decoded_output_of"],
                            ",".join(item["shared_event_schema_fields"]),
                        ),
                    }
                )
            evidence["perfetto_emitter"] = emitter_evidence

        put("profiler_struct", ("profiler_enum_task", "profiler_struct", "profiler_num_tracks", "profiler_buf_cute"))
        put("abi_magic", ("abi_magic",))
        if contention:
            put("slot_contention", ("atomic_add",))

        hw_capture_signal, hw_capture_evidence = self.analyze_hw_capture_range()
        hw_capture_signal["required"] = self.args.require_hw_capture_range
        if hw_capture_evidence:
            evidence[HW_CAPTURE_RANGE_CAPABILITY] = hw_capture_evidence

        signals.update(
            {
                "profiler_struct_present": bool(
                    self.any_hit("profiler_enum_task")
                    or self.any_hit("profiler_struct")
                    or self.any_hit("profiler_buf_cute")
                ),
                "abi_magic_present": self.any_hit("abi_magic"),
                "record_path_contention": contention,
                # 命中 clock64 但没有 globaltimer：判据不通过，但改动量小于从零插桩
                "clock_source_clock64_only": bool(has_clock64 and not has_globaltimer),
                "instrumentation_call_sites": self.count_call_sites(),
                "fixed_slot_files": fixed_files[:limit],
                "host_wiring_files": wiring_files[:limit],
                "abi_json_candidates": self.abi_json_files[:limit],
                "coupled_decoder_candidates": coupled_entries[:limit],
                "uncoupled_decoder_candidates": uncoupled[:limit],
                "perfetto_chain_candidates": perfetto_chain[:limit],
                "profiler_source_anchors": sorted(anchors),
                "profiler_type_names": sorted(self.profiler_type_names),
                "instrumentation_switch_names": sorted(self.switch_names),
                "rank_sync_site_count": len(rank_sync_sites),
                "entry_alignment_site_count": len(entry_sites),
                "baseline_entry_alignment_site_count": len(baseline_entry_sites),
                "baseline_rank_sync_reason": baseline_reason,
                "hw_capture_range": hw_capture_signal,
            }
        )
        capabilities = {
            "device_timestamping": device_timestamping,
            "fixed_slot_no_contention": fixed_slot_no_contention,
            "compile_time_switch": compile_time_switch,
            "rank_sync_anchor": rank_sync_anchor,
            "baseline_rank_sync": baseline_rank_sync,
            "buffer_allocation": buffer_allocation,
            "task_phase_definition": task_phase_definition,
            "source_locked_semantic_contract": source_locked_semantic_contract,
            "format_spec": format_spec,
            "host_decoder": host_decoder,
            "perfetto_emitter": perfetto_emitter,
            HW_CAPTURE_RANGE_CAPABILITY: hw_capture_signal["present"],
        }
        return capabilities, evidence, signals

    def overhead_caveat(self, capabilities):
        if not (capabilities["rank_sync_anchor"] and not capabilities["baseline_rank_sync"]):
            return None
        summary = self.rank_sync_summary
        guarded = summary.get("guarded_entry_sites", ())
        tail = (
            "Any measured delta therefore bundles (record cost + a whole extra entry rank sync) and "
            "is a conservative UPPER BOUND on the record cost. To obtain pure record overhead, either "
            "make the baseline execute the same rank sync (move it outside the instrumentation "
            "switch), or subtract the measured rank sync interval from the delta. The same pollution "
            "applies to version-to-version comparisons (v0 vs v1) built with different switch settings."
        )
        if not guarded:
            return (
                "Instrumentation overhead measured against this baseline is NOT pure record overhead. "
                "Rank sync anchoring was detected, but no rank sync qualifies as a kernel-entry launch "
                "alignment, so neither configuration has a verified common starting line and "
                "CTA-launch jitter is not removed from either side. " + tail
            )
        switches = sorted({site["guarded_by"] for site in guarded if site["guarded_by"]})
        where = ", ".join("{0}:{1}".format(site["file"], site["line"]) for site in guarded)
        return (
            "Instrumentation overhead measured against this baseline is NOT pure record overhead. "
            "The entry rank sync / launch alignment exists only in the instrumented build ({where}"
            "{switch}), so the baseline build never pays it and never gets its CTA-launch jitter "
            "removed -- the two configurations do not share a starting line. ".format(
                where=where,
                switch=" guarded by " + "/".join(switches) if switches else "",
            )
            + tail
        )

    def warnings(self, capabilities):
        out = []
        if capabilities["rank_sync_anchor"] and not capabilities["baseline_rank_sync"]:
            summary = self.rank_sync_summary
            guarded = summary.get("guarded_entry_sites", ())
            detail = (
                "rank sync anchoring is present but only in the instrumented build; baseline and "
                "instrumented runs do not share a launch-alignment starting line"
                if guarded
                else "rank sync anchoring is present but no rank sync qualifies as a kernel-entry "
                "launch alignment, so no common starting line is established for either build"
            )
            out.append(
                {
                    "id": "baseline_rank_sync_missing",
                    "severity": "warning",
                    "capability": "baseline_rank_sync",
                    "reason": summary.get("reason"),
                    "detail": detail,
                    "remediation": "add the same entry rank sync to the baseline build (outside the "
                    "instrumentation compile-time switch), or subtract the rank sync interval from "
                    "the measured overhead before quoting it",
                    "sites": [
                        {
                            "file": site["file"],
                            "line": site["line"],
                            "guarded_by": site["guarded_by"],
                        }
                        for site in guarded
                    ],
                }
            )
        return out

    def count_call_sites(self):
        definition_files = set()
        for pid in ("profiler_struct", "profiler_enum_task"):
            for record in self.aggregate.get(pid, ()):
                definition_files.add(record["rel"])
        count = 0
        for record in self.files:
            if record["rel"] in definition_files or record["ext"] not in DEVICE_EXTS:
                continue
            hits = record["hits"]
            if (
                "profiler_enable_flag" in hits
                or "profiler_buf_symbol" in hits
                or "profiler_buf_cute" in hits
            ):
                count += 1
        return count

    def clock_block(self):
        if self.any_hit("clock_globaltimer"):
            block = {"source": "%globaltimer", "unit": "ns", "cross_sm_alignable": True}
        elif self.any_hit("clock_smemrealtime"):
            block = {"source": "s_memrealtime", "unit": "ns", "cross_sm_alignable": True,
                     "platform": "amd",
                     "note": "gfx9xx global realtime clock; MI308X 实测跨 CU 同基线, 可用于跨 CU 时间线"}
        elif self.any_hit("clock_clock64"):
            block = {
                "source": "clock64()",
                "unit": "cycles",
                "cross_sm_alignable": False,
                "note": "clock64() is per-SM; records cannot be aligned across SMs without a common reference",
            }
        else:
            block = {"source": None, "unit": None, "cross_sm_alignable": False}
        block["sm_identity"] = self.any_hit("clock_smid") or self.any_hit("clock_amd_hwid")
        return block

    def transport_block(self):
        return {
            "num_tracks": self.num_tracks,
            "tag_left_shifts": sorted(self.device_tag_shifts, reverse=True),
            "tag_masks": sorted(self.device_tag_masks),
            "profiler_type_names": sorted(self.profiler_type_names),
        }

    def classify(self, capabilities, signals, required_capabilities=None):
        # 九项全在才判 fast（走脚本路径）；任一项缺就是 slow（大模型补齐后重判）。
        # 各项缺失的修补代价不同，所以 reason 里按代价分组报出，便于按缺项逐条修。
        required_capabilities = tuple(required_capabilities or CAPABILITY_ORDER)
        missing = [name for name in required_capabilities if not capabilities[name]]
        if not missing:
            hw_suffix = (
                " The harness also exposes a statically balanced cudaProfilerApi "
                "kernel-only range for HW capture."
                if HW_CAPTURE_RANGE_CAPABILITY in required_capabilities
                else ""
            )
            return (
                "complete",
                "fast",
                "All {0} required capabilities are present: device-side recording is complete and "
                "an ABI-coupled host decoder plus emitter exist. Capture, decode and render "
                "run as plain scripts with no model source reading.{1}".format(
                    len(required_capabilities), hw_suffix
                ),
            )

        device_core = capabilities["device_timestamping"] and capabilities["fixed_slot_no_contention"]
        gaps = []
        if not capabilities["device_timestamping"]:
            gaps.append(
                "no %globaltimer timestamp source"
                + (" (only clock64 found: per-SM and frequency-dependent, "
                   "not comparable across SMs; must be replaced by %globaltimer)"
                   if signals.get("clock_source_clock64_only") else "")
            )
        if not capabilities["fixed_slot_no_contention"]:
            gaps.append(
                "no contention-free fixed-slot record path"
                + (" (atomicAdd found on the record path)" if signals["record_path_contention"] else "")
            )
        for name, note in (
            ("compile_time_switch", "no compile-time switch to remove the profiler from production builds"),
            ("buffer_allocation", "no host-side capacity-checked zeroed int64 buffer"),
            ("rank_sync_anchor", "no entry alignment barrier (expected a named `rank sync` at kernel entry)"),
            ("task_phase_definition", "no task/phase event enumeration with matching Device probes"),
            ("source_locked_semantic_contract",
             "no valid model-reviewed semantic_analysis/tasks[].task_contract whose source_closure and conclusion SHA256 are current"),
            ("format_spec", "no buffer-layout description file (bitfields / slot formula / task enum)"),
            ("host_decoder", "no decoder whose provenance chain closes onto the instrumented kernel file"),
            ("perfetto_emitter", "no emitter coupled to the decoder's event schema"),
        ):
            if not capabilities[name]:
                gaps.append(note)
        if (
            HW_CAPTURE_RANGE_CAPABILITY in required_capabilities
            and not capabilities[HW_CAPTURE_RANGE_CAPABILITY]
        ):
            gaps.append(
                "no single harness file proves the HW capture contract: CLI plus balanced "
                "cudaProfilerStart/Stop (or torch.cuda.profiler.start/stop), an ordered "
                "kernel launch, and post-launch synchronization"
            )

        verdict = "absent" if not device_core else "partial"
        return (
            verdict,
            "slow",
            "Missing " + str(len(missing)) + " of " + str(len(required_capabilities)) + " capabilities: "
            + "; ".join(gaps)
            + ". Repair cost by group: clock source / fixed slot / entry alignment / task+phase probes "
              "need kernel edits; compile-time switch and buffer allocation need launch-code edits; "
              "format spec, decoder and emitter are host-side Python.",
        )

    def confidence(self, verdict, capabilities, signals):
        if verdict == "absent":
            if self.stats["scanned_files"] == 0:
                return 0.2
            ambiguous = (
                signals["profiler_struct_present"]
                or capabilities["device_timestamping"]
                or capabilities["fixed_slot_no_contention"]
                or signals["abi_magic_present"]
            )
            return 0.62 if ambiguous else 0.93
        weights = {
            "device_timestamping": 0.24 if self.any_hit("clock_globaltimer") else 0.14,
            "fixed_slot_no_contention": 0.19,
            "compile_time_switch": 0.14,
            "rank_sync_anchor": 0.08,
            "buffer_allocation": 0.12,
            "task_phase_definition": 0.03,
            "source_locked_semantic_contract": 0.06,
            "format_spec": 0.03,
            "host_decoder": 0.08,
            "perfetto_emitter": 0.03,
        }
        score = sum(weight for name, weight in weights.items() if capabilities[name])
        if signals["instrumentation_call_sites"] == 0:
            score -= 0.10
        if self.args.require_hw_capture_range:
            score = 0.92 * score + (
                0.08 if capabilities[HW_CAPTURE_RANGE_CAPABILITY] else 0.0
            )
        return round(max(0.0, min(1.0, score)), 2)

    def commands(self, verdict, path, capabilities, signals):
        cmds = self.base_commands(verdict, path, capabilities, signals)
        if capabilities["rank_sync_anchor"] and not capabilities["baseline_rank_sync"]:
            guarded = self.rank_sync_summary.get("guarded_entry_sites", ())
            switches = sorted({site["guarded_by"] for site in guarded if site["guarded_by"]})
            cmds.append(
                "TODO(baseline-rank-sync): the entry rank sync exists only in the instrumented build"
                "{switch}; add the same rendezvous to the baseline build or subtract the rank sync "
                "interval before quoting instrumentation overhead".format(
                    switch=" (guarded by " + "/".join(switches) + ")" if switches else ""
                )
            )
        return cmds

    def base_commands(self, verdict, path, capabilities, signals):
        tracks = self.num_tracks if self.num_tracks else "<num_tracks>"
        shifts = sorted(self.device_tag_shifts, reverse=True)
        self_ref = os.path.relpath(os.path.abspath(__file__), self.root).replace(os.sep, "/")
        if verdict == "absent":
            return [
                "TODO(slow-path): read the target kernel sources with the model and author instrumentation",
                "TODO(slow-path): add a %globaltimer read plus a fixed-slot (blockIdx/track/max_records) record path",
                "TODO(slow-path): add a compile-time kEnabled switch with an empty specialization so production builds are unaffected",
                "TODO(slow-path): allocate and validate the profiler buffer on the host, then expose it at the Python entry point",
                "python3 -B {script} --root {root}".format(script=self_ref, root=self.root),
            ]
        capture = (
            "TODO(capture): allocate profiler_buf = torch.zeros(1 + num_sms + num_sms * {tracks} * max_records * 2, "
            "dtype=torch.int64, device='cuda') and call the profiled entry point with profiler_max_records=max_records"
        ).format(tracks=tracks)
        if verdict == "complete":
            coupled = signals.get("coupled_decoder_candidates", [])
            decoder = None
            for item in coupled:
                if item.get("decode_intent"):
                    decoder = item["file"]
                    break
            if decoder is None and coupled:
                decoder = coupled[0]["file"]
            cmds = [
                capture,
                "python3 -B {decoder} --help    # ABI-coupled decoder detected; capture then decode".format(
                    decoder=decoder or "<host_decoder>"
                ),
            ]
            emitters = signals.get("perfetto_chain_candidates", [])
            if emitters:
                cmds.append(
                    "python3 -B {emitter} --help    # consumes the decoded events.json and emits Perfetto traceEvents".format(
                        emitter=emitters[0]["file"]
                    )
                )
            cmds.append("python3 -B {script} --root {root}".format(script=self_ref, root=self.root))
            return cmds
        if path == "fast":
            cmds = [
                "TODO(decoder): author a host decoder for the detected transport ONLY -- do not re-read kernel sources",
                "  layout: buf[0]=metadata, buf[1 + block]=block/smid map, "
                "buf[header + (block * {tracks} * max_records + track * max_records + record) * 2] = (timestamp_ns, tag)".format(
                    tracks=tracks
                ),
                "  tag bitfields: right-shift by {shifts} with masks {masks}".format(
                    shifts=shifts or "<detect failed>", masks=sorted(self.device_tag_masks) or "<detect failed>"
                ),
                capture,
                "TODO(perfetto): emit Chrome/Perfetto traceEvents from the decoded records",
                "python3 -B {script} --root {root}".format(script=self_ref, root=self.root),
            ]
            uncoupled = signals.get("uncoupled_decoder_candidates", [])
            if uncoupled:
                cmds.insert(
                    1,
                    "  note: {n} decoder-looking script(s) exist but decode a different ABI (e.g. {sample}); do not reuse blindly".format(
                        n=len(uncoupled), sample=uncoupled[0]["file"]
                    ),
                )
            return cmds
        return [
            "TODO(slow-path): complete the device-side stack ({gaps})".format(
                gaps=", ".join(name for name in ("compile_time_switch", "buffer_allocation") if not capabilities[name])
            ),
            capture,
            "python3 -B {script} --root {root}".format(script=self_ref, root=self.root),
        ]


def print_summary(report, stream):
    caps = report["capabilities"]
    required_capabilities = report.get("requirements", {}).get(
        "required_capabilities", list(CAPABILITY_ORDER)
    )
    stream.write("[detect_instrumentation] root={root}\n".format(root=report["root"]))
    stream.write(
        "  verdict={verdict}  confidence={conf}  path={path}\n".format(
            verdict=report["verdict"], conf=report["confidence"], path=report["next_action"]["path"]
        )
    )
    stream.write(
        "  capabilities: "
        + " ".join(
            "{0}={1}".format(name, "yes" if caps[name] else "NO")
            for name in required_capabilities
        )
        + "\n"
    )
    stream.write("  missing: " + (", ".join(report["missing"]) or "(none)") + "\n")
    for warning in report.get("warnings", ()):
        stream.write(
            "  WARNING[{id}]: {detail} ({reason})\n".format(
                id=warning["id"], detail=warning["detail"], reason=warning["reason"]
            )
        )
        for site in warning.get("sites", ()):
            stream.write(
                "    guarded site: {file}:{line} guarded_by={guard}\n".format(
                    file=site["file"], line=site["line"], guard=site["guarded_by"]
                )
            )
    if report.get("overhead_measurement_caveat"):
        stream.write("  overhead caveat: {0}\n".format(report["overhead_measurement_caveat"]))
    clock = report["clock"]
    stream.write(
        "  clock: {source} unit={unit} cross_sm_alignable={align}\n".format(
            source=clock["source"], unit=clock["unit"], align=clock["cross_sm_alignable"]
        )
    )
    stream.write("  reason: {0}\n".format(report["next_action"]["reason"]))
    stream.write("  next commands:\n")
    for index, command in enumerate(report["next_action"]["commands"], start=1):
        stream.write("    {0}. {1}\n".format(index, command))
    scan = report["scan"]
    stream.write(
        "  scan: {scanned} files scanned, {matched} with hits, {large} oversized skipped, {binary} binary skipped\n".format(
            scanned=scan["scanned_files"],
            matched=scan["matched_files"],
            large=scan["skipped_large"],
            binary=scan["skipped_binary"],
        )
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="detect_instrumentation.py",
        description="Fingerprint-scan a repository for GPU kernel instrumentation completeness "
        "and choose the fast (script-only) or slow (model-driven) path.",
    )
    parser.add_argument("--root", required=True, help="repository root to scan")
    parser.add_argument("--out", default=None, help="write the verdict JSON to this path")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="NAME_OR_GLOB",
        help="additional directory name, relative path or glob to skip (repeatable)",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="do not skip the default {0} directories".format(", ".join(DEFAULT_EXCLUDES)),
    )
    parser.add_argument(
        "--prune",
        action="append",
        default=[],
        metavar="NAME",
        help="additional directory name to prune (repeatable)",
    )
    parser.add_argument("--max-bytes", type=int, default=2 * 1024 * 1024, help="skip files larger than this")
    parser.add_argument("--max-evidence", type=int, default=5, help="evidence entries kept per capability")
    parser.add_argument("--include-hidden", action="store_true", help="also scan dot-directories and dot-files")
    parser.add_argument(
        "--require-hw-capture-range",
        action="store_true",
        help="hard-gate HW mode on a harness CLI plus a balanced kernel-only "
        "cudaProfilerStart/Stop range (missing range exits 4)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the stderr summary")
    parser.add_argument(
        "--exit-zero",
        action="store_true",
        help="exit 0 for ordinary capability gaps; does not override the explicit HW range hard gate",
    )
    args = parser.parse_args(argv)
    args.exclude_list = list(args.exclude) if args.no_default_excludes else list(DEFAULT_EXCLUDES) + list(args.exclude)
    args.prune_list = list(DEFAULT_PRUNE_DIRS) + list(args.prune)
    return args


def main(argv):
    args = parse_args(argv)
    root = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.isdir(root):
        sys.stderr.write("error: --root is not a directory: {0}\n".format(root))
        return 2
    report = Detector(args).run()
    payload = json.dumps(report, indent=2, sort_keys=False, ensure_ascii=False)
    sys.stdout.write(payload + "\n")
    if args.out:
        out_path = os.path.abspath(os.path.expanduser(args.out))
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    if not args.quiet:
        print_summary(report, sys.stderr)
    if (
        args.require_hw_capture_range
        and not report["capabilities"][HW_CAPTURE_RANGE_CAPABILITY]
    ):
        return 4
    if args.exit_zero:
        return 0
    return 0 if report["next_action"]["path"] == "fast" else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
