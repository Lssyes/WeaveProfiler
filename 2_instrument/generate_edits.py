#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate deterministic SHA-bound instrumentation edits from a scan selection.

The input selection is explicit: every item names one source file, one canonical
semantic ID, and both boundaries.  An optional event ID must agree with the
canonical event ID declared for that semantic.  The generator verifies the scan's
path/hash evidence and the scan candidates at the selected locations.  It never
edits project sources.

Calls emitted by route language
-------------------------------
``cuda_cpp`` and ``hip_cpp``::

    FI_B(event_id, aux);
    FI_E(event_id, aux);

``triton`` on NVIDIA (state-threaded recorder): ::

    n = fi_b(rec, ovf, pid, n, FI, ev=event_id, aux=aux)
    n = fi_e(rec, ovf, pid, n, FI, ev=event_id, aux=aux)

``triton`` on AMD/MI308X (explicit wave64 recorder): ::

    n = fi_b(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
             EVENT=event_id, aux=aux, CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
    n = fi_e(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
             EVENT=event_id, aux=aux, CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)

``cutedsl`` (the recorder's value-style form)::

    ctx = fi_b(ctx, event_id, aux)
    ctx = fi_e(ctx, event_id, aux)

The emitted patch plan contains only the fields accepted by
``2_instrument/apply_plan.py``.  Canonical event names are written separately to
``<out>.names.json``; that sidecar is the decoder mapping authority.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import semantics  # noqa: E402


SELECTION_SCHEMA = "weaveprofiler-instrument-selection/v1"
SELECTION_VERSION = 1
SCAN_SCHEMA = "weaveprofiler-scan/v1"
RESULT_SCHEMA = "weaveprofiler-generate-edits-result/v1"
MAX_EVENT_ID = 4095
CANONICAL_EVENT_IDS = {
    "load": 1,
    "wait": 2,
    "mma": 3,
    "compute": 4,
    "epilogue": 5,
    "store": 6,
    "barrier": 7,
    "dispatch": 8,
    "communication": 9,
    "gemm": 10,
    "attention": 11,
    "reduction": 12,
    "moe": 13,
    "convolution": 14,
    "megakernel": 15,
    "softmax": 16,
}
CPP_LANGUAGES = frozenset({"cuda_cpp", "hip_cpp"})
PYTHON_LANGUAGES = frozenset({"triton", "cutedsl"})
SUPPORTED_LANGUAGES = CPP_LANGUAGES | PYTHON_LANGUAGES
SELECTION_KEYS = frozenset(
    {
        "path",
        "semantic_id",
        "begin_anchor",
        "end_anchor",
        "begin_line",
        "end_line",
        "occurrence",
        "expect_count",
        "aux",
        "event_id",
    }
)
SELECTION_TOP_LEVEL_KEYS = frozenset({"schema", "version", "selections"})
HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
AUX_FORBIDDEN = frozenset({"\x00", "\r", "\n", ";", "{", "}"})
_APPLY_PLAN_MODULE = None


class GenerationError(ValueError):
    """A fail-closed generation error with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        selection_index: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.selection_index = selection_index
        self.details = dict(details or {})

    def as_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.selection_index is not None:
            value["selection_index"] = self.selection_index
        if self.details:
            value["details"] = self.details
        return value


@dataclass(frozen=True)
class SourceState:
    path: str
    absolute_path: Path
    base_sha256: str
    text: str
    line_starts: Tuple[int, ...]
    line_content_ends: Tuple[int, ...]
    newline: str


@dataclass(frozen=True)
class Boundary:
    kind: str
    anchor: str
    position: str
    occurrence: int
    expect_count: int
    match_start: int
    match_end: int
    first_line: int
    last_line: int
    manual_indent: str


@dataclass
class PendingSelection:
    index: int
    source: SourceState
    semantic_id: str
    aux: str
    event_id: Optional[int]
    begin: Boundary
    end: Boundary
    range_start: int
    range_end: int
    allocated: bool = False
    id_fingerprint: str = field(default="")


@dataclass(frozen=True)
class GeneratedEdits:
    plan: Dict[str, Any]
    names: Dict[str, Any]
    allocations: Tuple[Dict[str, Any], ...]
    project_root: str
    language: str
    check_diff: str

    def summary(self, out: Path, names_out: Optional[Path]) -> Dict[str, Any]:
        return {
            "schema": RESULT_SCHEMA,
            "ok": True,
            "out": str(out),
            "names_out": str(names_out) if names_out is not None else None,
            "project_root": self.project_root,
            "language": self.language,
            "files": len(self.plan["files"]),
            "edits": sum(len(item["edits"]) for item in self.plan["files"]),
            "allocations": list(self.allocations),
        }


def _apply_plan_module():
    global _APPLY_PLAN_MODULE
    if _APPLY_PLAN_MODULE is not None:
        return _APPLY_PLAN_MODULE
    path = Path(__file__).resolve().parent / "apply_plan.py"
    spec = importlib.util.spec_from_file_location(
        "weaveprofiler_generate_edits_apply_plan", path
    )
    if spec is None or spec.loader is None:
        raise GenerationError(
            "apply_plan_unavailable", "cannot load apply_plan.py from {}".format(path)
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _APPLY_PLAN_MODULE = module
    return module


def _duplicate_key_error(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GenerationError(
                "duplicate_json_key",
                "JSON object repeats key {!r}".format(key),
                details={"key": key},
            )
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> Any:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise GenerationError(
            "{}_read_failed".format(label), "cannot read {}: {}".format(label, exc)
        ) from exc
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GenerationError(
            "{}_invalid_utf8".format(label), "{} is not UTF-8: {}".format(label, exc)
        ) from exc
    try:
        return json.loads(text, object_pairs_hook=_duplicate_key_error)
    except GenerationError:
        raise
    except json.JSONDecodeError as exc:
        raise GenerationError(
            "{}_invalid_json".format(label),
            "{} is not valid JSON: {}".format(label, exc),
            details={"line": exc.lineno, "column": exc.colno},
        ) from exc


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_replace_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(
        prefix=".{}.generate-edits.".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return temp
    except BaseException:
        try:
            temp.unlink()
        except OSError:
            pass
        raise


def _safe_relative(raw: Any, *, label: str, selection_index: Optional[int] = None) -> str:
    if not isinstance(raw, str) or not raw:
        raise GenerationError(
            "invalid_path",
            "{} must be a non-empty relative POSIX path".format(label),
            selection_index=selection_index,
        )
    if "\x00" in raw or raw.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", raw):
        raise GenerationError(
            "path_escape", "{} must be relative to project_root".format(label),
            selection_index=selection_index, details={"path": raw},
        )
    if "\\" in raw:
        raise GenerationError(
            "path_escape", "{} must use POSIX separators".format(label),
            selection_index=selection_index, details={"path": raw},
        )
    pieces = raw.split("/")
    if any(piece in ("", ".", "..") for piece in pieces):
        raise GenerationError(
            "path_escape", "{} contains an empty, '.' or '..' component".format(label),
            selection_index=selection_index, details={"path": raw},
        )
    return raw


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _project_root(
    scan: Mapping[str, Any], project_root_override: Optional[str | os.PathLike[str]]
) -> Path:
    raw_scan_root = scan.get("project_root")
    scan_root: Optional[Path] = None
    scan_root_available = False
    if isinstance(raw_scan_root, str) and raw_scan_root.strip():
        candidate = Path(raw_scan_root).expanduser()
        if candidate.is_absolute():
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                resolved = candidate.resolve()
            if resolved.is_dir():
                scan_root = resolved
                scan_root_available = True
            else:
                scan_root = resolved
        else:
            scan_root = candidate.resolve()

    if project_root_override is not None:
        override = Path(project_root_override).expanduser()
        if not override.is_absolute():
            override = (Path.cwd() / override).resolve()
        else:
            try:
                override = override.resolve(strict=True)
            except OSError as exc:
                raise GenerationError(
                    "invalid_project_root",
                    "project root override does not exist: {}".format(override),
                ) from exc
        if not override.is_dir():
            raise GenerationError(
                "invalid_project_root", "project root override is not a directory"
            )
        if scan_root_available:
            raise GenerationError(
                "project_root_override_not_allowed",
                "--project-root may only be used when scan project_root is unavailable",
                details={"scan_project_root": str(scan_root), "override": str(override)},
            )
        return override

    if scan_root is not None and scan_root.is_dir():
        return scan_root
    raise GenerationError(
        "invalid_project_root",
        "scan project_root is unavailable and no --project-root override was supplied",
        details={"scan_project_root": raw_scan_root},
    )


def _scan_language(scan: Mapping[str, Any]) -> str:
    route = scan.get("route")
    route_language = route.get("language") if isinstance(route, Mapping) else None
    top_language = scan.get("language")
    if route_language is not None and top_language is not None and route_language != top_language:
        raise GenerationError(
            "invalid_scan",
            "scan.route.language conflicts with scan.language",
            details={"route_language": route_language, "language": top_language},
        )
    language = route_language if route_language is not None else top_language
    if language not in SUPPORTED_LANGUAGES:
        raise GenerationError(
            "unsupported_language",
            "scan route.language must be one of {}".format(sorted(SUPPORTED_LANGUAGES)),
            details={"language": language},
        )
    return str(language)


def _scan_evidence(
    scan: Mapping[str, Any], root: Path
) -> Tuple[str, Dict[str, Mapping[str, Any]], List[Mapping[str, Any]]]:
    if (scan.get("schema") if isinstance(scan, Mapping) else None) != SCAN_SCHEMA:
        raise GenerationError(
            "invalid_scan_schema", "scan schema must be {!r}".format(SCAN_SCHEMA)
        )
    language = _scan_language(scan)

    raw_files = scan.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise GenerationError("invalid_scan", "scan.files must be a non-empty array")
    files: Dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(raw_files):
        if not isinstance(item, Mapping):
            raise GenerationError(
                "invalid_scan", "scan.files[{}] must be an object".format(index)
            )
        path = _safe_relative(item.get("path"), label="scan.files[{}].path".format(index))
        digest = item.get("sha256")
        if not isinstance(digest, str) or not HEX_SHA256.fullmatch(digest):
            raise GenerationError(
                "invalid_scan",
                "scan.files[{}].sha256 must be 64 hexadecimal characters".format(index),
                details={"path": path},
            )
        if path in files:
            raise GenerationError(
                "invalid_scan", "scan.files repeats path {!r}".format(path)
            )
        files[path] = item

    raw_candidates = scan.get("candidates")
    if not isinstance(raw_candidates, list):
        raise GenerationError("invalid_scan", "scan.candidates must be an array")
    candidates: List[Mapping[str, Any]] = []
    for index, item in enumerate(raw_candidates):
        if not isinstance(item, Mapping):
            raise GenerationError(
                "invalid_scan", "scan.candidates[{}] must be an object".format(index)
            )
        _safe_relative(item.get("path"), label="scan.candidates[{}].path".format(index))
        line = item.get("line")
        semantic_id = item.get("semantic_id")
        if isinstance(line, bool) or not isinstance(line, int) or line < 1:
            raise GenerationError(
                "invalid_scan",
                "scan.candidates[{}].line must be a positive integer".format(index),
            )
        if not isinstance(semantic_id, str) or not semantic_id:
            raise GenerationError(
                "invalid_scan",
                "scan.candidates[{}].semantic_id must be non-empty".format(index),
            )
        candidates.append(item)
    return language, files, candidates


def _selection_document(document: Any) -> List[Mapping[str, Any]]:
    if not isinstance(document, Mapping):
        raise GenerationError("invalid_selection", "selection must be a JSON object")
    unknown = sorted(set(document) - SELECTION_TOP_LEVEL_KEYS)
    if unknown:
        raise GenerationError(
            "invalid_selection",
            "selection has unknown field(s): {}".format(", ".join(unknown)),
            details={"unknown_fields": unknown},
        )
    if document.get("schema") != SELECTION_SCHEMA:
        raise GenerationError(
            "invalid_selection_schema",
            "selection schema must be {!r}".format(SELECTION_SCHEMA),
        )
    version = document.get("version")
    if isinstance(version, bool) or version != SELECTION_VERSION:
        raise GenerationError(
            "invalid_selection_version",
            "selection version must be {}".format(SELECTION_VERSION),
        )
    selections = document.get("selections")
    if not isinstance(selections, list) or not selections:
        raise GenerationError(
            "invalid_selection", "selection.selections must be a non-empty array"
        )
    for index, item in enumerate(selections):
        if not isinstance(item, Mapping):
            raise GenerationError(
                "invalid_selection",
                "selection.selections[{}] must be an object".format(index),
                selection_index=index,
            )
        unknown = sorted(set(item) - SELECTION_KEYS)
        if unknown:
            raise GenerationError(
                "invalid_selection",
                "selection has unknown field(s): {}".format(", ".join(unknown)),
                selection_index=index,
                details={"unknown_fields": unknown},
            )
    return selections


def _canonical_semantic(
    raw: Mapping[str, Any], canonical_ids: set[str], index: int
) -> str:
    semantic_id = raw.get("semantic_id")
    if not isinstance(semantic_id, str) or not semantic_id:
        raise GenerationError(
            "unknown_semantic",
            "semantic_id is required and must be a canonical ID from semantic_ids.json",
            selection_index=index,
            details={"semantic_id": semantic_id},
        )
    if semantic_id not in canonical_ids or semantic_id == "unknown":
        raise GenerationError(
            "unknown_semantic",
            "semantic_id {!r} is not a declared resolvable canonical ID".format(
                semantic_id
            ),
            selection_index=index,
            details={"semantic_id": semantic_id},
        )
    return semantic_id


def _load_semantic_kb() -> Dict[str, Any]:
    """Load semantic IDs while preserving ``event_id`` values from the KB.

    ``knowledge/semantic_ids.json`` is the event-ID authority whenever a row
    carries ``event_id``.  The compatibility table is only used for rows that
    predate that field, so an updated KB can change the reservation without a
    generator change.  ``event_id_const`` is likewise an existing KB invariant;
    fill it in memory only when it is absent so older KB revisions remain valid.
    """

    try:
        kb = semantics.load_kb(validate=False)
    except semantics.SemanticKBError as exc:
        raise GenerationError("invalid_semantic_ids", str(exc)) from exc
    raw_rows = kb.get("semantics")
    if not isinstance(raw_rows, list):
        raise GenerationError(
            "invalid_semantic_ids", "knowledge/semantic_ids.json semantics must be a list"
        )
    rows: List[Any] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            rows.append(raw)
            continue
        row = dict(raw)
        semantic_id = row.get("semantic_id")
        if isinstance(semantic_id, str) and semantic_id != "unknown":
            if row.get("event_id") is None:
                fallback = CANONICAL_EVENT_IDS.get(semantic_id)
                if fallback is not None:
                    row["event_id"] = fallback
            if row.get("event_id_const") is None:
                row["event_id_const"] = "EV_{}".format(semantic_id.upper())
        rows.append(row)
    kb["semantics"] = rows
    errors = semantics.validate_kb(kb)
    if errors:
        raise GenerationError(
            "invalid_semantic_ids",
            "knowledge/semantic_ids.json is invalid: {}".format("; ".join(errors[:12])),
            details={"errors": errors},
        )
    return kb


def _canonical_event_id_map(kb: Mapping[str, Any]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for row in kb["semantics"]:
        semantic_id = str(row["semantic_id"])
        if semantic_id == "unknown":
            continue
        result[semantic_id] = int(row["event_id"])
    return result


def _positive_int(
    value: Any,
    *,
    field_name: str,
    index: int,
    default: int = 1,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GenerationError(
            "invalid_{}".format(field_name),
            "{} must be a positive integer".format(field_name),
            selection_index=index,
            details={field_name: value},
        )
    return value


def _selection_event_id(
    raw: Mapping[str, Any],
    index: int,
    *,
    semantic_id: str,
    canonical_event_ids: Mapping[str, int],
) -> Optional[int]:
    canonical = canonical_event_ids.get(semantic_id)
    if canonical is None:
        raise GenerationError(
            "canonical_event_id_missing",
            "semantic_id {!r} has no canonical event_id".format(semantic_id),
            selection_index=index,
            details={"semantic_id": semantic_id},
        )
    if "event_id" not in raw:
        return None
    value = raw.get("event_id")
    if isinstance(value, bool) or not isinstance(value, int):
        raise GenerationError(
            "invalid_event_id",
            "event_id must be an integer in 1..{}".format(MAX_EVENT_ID),
            selection_index=index,
        )
    if not 1 <= value <= MAX_EVENT_ID:
        raise GenerationError(
            "invalid_event_id",
            "event_id must be in 1..{}".format(MAX_EVENT_ID),
            selection_index=index,
            details={"event_id": value},
        )
    if value != canonical:
        raise GenerationError(
            "event_id_semantic_conflict",
            "event_id {} conflicts with canonical event_id {} for {!r}".format(
                value, canonical, semantic_id
            ),
            selection_index=index,
            details={
                "semantic_id": semantic_id,
                "event_id": value,
                "canonical_event_id": canonical,
            },
        )
    return value


def _selection_aux(raw: Mapping[str, Any], index: int, language: str) -> str:
    if "aux" not in raw:
        raise GenerationError(
            "invalid_aux", "aux expression is required", selection_index=index
        )
    value = raw.get("aux")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise GenerationError(
            "invalid_aux",
            "aux must be a non-empty source expression string or integer",
            selection_index=index,
        )
    aux = str(value).strip()
    if not aux:
        raise GenerationError(
            "invalid_aux", "aux expression cannot be empty", selection_index=index
        )
    forbidden = sorted(character for character in AUX_FORBIDDEN if character in aux)
    if forbidden:
        raise GenerationError(
            "invalid_aux",
            "aux expression contains a forbidden character",
            selection_index=index,
            details={"characters": forbidden},
        )
    if language in PYTHON_LANGUAGES:
        try:
            ast.parse(aux, mode="eval")
        except SyntaxError as exc:
            raise GenerationError(
                "invalid_aux",
                "aux is not a valid Python expression: {}".format(exc.msg),
                selection_index=index,
            ) from exc
    return aux


def _line_records(text: str) -> Tuple[Tuple[int, ...], Tuple[int, ...], str]:
    starts: List[int] = []
    content_ends: List[int] = []
    cursor = 0
    newline = "\n"
    while cursor < len(text):
        found = re.search(r"\r\n|\r|\n", text[cursor:])
        starts.append(cursor)
        if found is None:
            content_ends.append(len(text))
            break
        content_end = cursor + found.start()
        content_ends.append(content_end)
        if len(starts) == 1:
            newline = found.group(0)
        cursor += found.end()
    return tuple(starts), tuple(content_ends), newline


def _load_source(root: Path, path: str, evidence: Mapping[str, Any]) -> SourceState:
    candidate = root.joinpath(*path.split("/"))
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise GenerationError(
            "source_not_found", "source parent does not exist: {}".format(exc),
            details={"path": path},
        ) from exc
    if parent != root and root not in parent.parents:
        raise GenerationError(
            "path_escape", "resolved source parent is outside project_root",
            details={"path": path},
        )
    if candidate.is_symlink():
        raise GenerationError(
            "path_symlink", "source file cannot be a symbolic link",
            details={"path": path},
        )
    try:
        absolute = candidate.resolve(strict=True)
    except OSError as exc:
        raise GenerationError(
            "source_not_found", "source file does not exist: {}".format(exc),
            details={"path": path},
        ) from exc
    if absolute != root and root not in absolute.parents:
        raise GenerationError(
            "path_escape", "resolved source is outside project_root",
            details={"path": path},
        )
    if not absolute.is_file():
        raise GenerationError(
            "source_not_found", "source path is not a regular file",
            details={"path": path},
        )
    try:
        payload = absolute.read_bytes()
    except OSError as exc:
        raise GenerationError(
            "source_read_failed", "cannot read source: {}".format(exc),
            details={"path": path},
        ) from exc
    actual_sha = hashlib.sha256(payload).hexdigest()
    scan_sha = str(evidence["sha256"]).lower()
    if actual_sha != scan_sha:
        raise GenerationError(
            "base_sha256_mismatch",
            "source bytes no longer match the scan's SHA-256",
            details={"path": path, "expected": scan_sha, "actual": actual_sha},
        )
    expected_bytes = evidence.get("bytes")
    if expected_bytes is not None and expected_bytes != len(payload):
        raise GenerationError(
            "source_size_mismatch",
            "source byte count no longer matches the scan",
            details={"path": path, "expected": expected_bytes, "actual": len(payload)},
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GenerationError(
            "source_invalid_utf8", "source is not valid UTF-8: {}".format(exc),
            details={"path": path},
        ) from exc
    starts, content_ends, newline = _line_records(text)
    return SourceState(
        path=path,
        absolute_path=absolute,
        base_sha256=str(evidence["sha256"]),
        text=text,
        line_starts=starts,
        line_content_ends=content_ends,
        newline=newline,
    )


def _anchor_matches(text: str, anchor: str) -> List[int]:
    matches: List[int] = []
    cursor = text.find(anchor)
    while cursor != -1:
        matches.append(cursor)
        cursor = text.find(anchor, cursor + 1)
    return matches


def _line_for_offset(source: SourceState, offset: int) -> int:
    if not source.line_starts:
        return 1
    offset = max(0, min(offset, max(source.line_starts)))
    return bisect_right(source.line_starts, offset)


def _line_indent(source: SourceState, line: int) -> str:
    if not 1 <= line <= len(source.line_starts):
        return ""
    line_start = source.line_starts[line - 1]
    content_end = source.line_content_ends[line - 1]
    return re.match(r"[ \t]*", source.text[line_start:content_end]).group(0)


def _apply_anchor_indent(source: SourceState, match_start: int) -> str:
    line = _line_for_offset(source, match_start)
    line_start = source.line_starts[line - 1]
    prefix = source.text[line_start:match_start]
    return re.match(r"[ \t]*", prefix).group(0)


def _after_manual_indent(
    source: SourceState, match_start: int, match_end: int
) -> str:
    desired = _line_indent(source, _line_for_offset(source, max(match_start, match_end - 1)))
    automatic = _apply_anchor_indent(source, match_start)
    if desired.startswith(automatic):
        return desired[len(automatic) :]
    return desired


def _before_manual_indent(source: SourceState, match_start: int) -> str:
    line = _line_for_offset(source, match_start)
    desired = _line_indent(source, line)
    line_start = source.line_starts[line - 1]
    prefix = source.text[line_start:match_start]
    if not prefix:
        return desired
    if not prefix.strip(" \t"):
        return ""
    return ""


def _boundary_from_anchor(
    source: SourceState,
    raw: Mapping[str, Any],
    index: int,
    *,
    kind: str,
    field_name: str,
    position: str,
    occurrence: int,
    expect_count: int,
) -> Boundary:
    anchor_value = raw.get(field_name)
    if not isinstance(anchor_value, str) or not anchor_value:
        raise GenerationError(
            "invalid_anchor",
            "{} must be a non-empty source string".format(field_name),
            selection_index=index,
        )
    if "\x00" in anchor_value:
        raise GenerationError(
            "invalid_anchor", "{} must not contain NUL".format(field_name),
            selection_index=index,
        )
    matches = _anchor_matches(source.text, anchor_value)
    if not matches:
        raise GenerationError(
            "anchor_not_found",
            "{} matched zero times in {}".format(field_name, source.path),
            selection_index=index,
            details={"anchor": anchor_value, "path": source.path},
        )
    if len(matches) != expect_count:
        raise GenerationError(
            "anchor_count_mismatch",
            "{} matched {} time(s), expected exactly {}".format(
                field_name, len(matches), expect_count
            ),
            selection_index=index,
            details={"actual_count": len(matches), "expect_count": expect_count},
        )
    if occurrence > expect_count:
        raise GenerationError(
            "occurrence_out_of_range",
            "occurrence cannot exceed expect_count",
            selection_index=index,
            details={"occurrence": occurrence, "expect_count": expect_count},
        )
    match_start = matches[occurrence - 1]
    match_end = match_start + len(anchor_value)
    first_line = _line_for_offset(source, match_start)
    last_line = _line_for_offset(source, max(match_start, match_end - 1))
    manual = (
        _after_manual_indent(source, match_start, match_end)
        if position == "after"
        else _before_manual_indent(source, match_start)
    )
    return Boundary(
        kind=kind,
        anchor=anchor_value,
        position=position,
        occurrence=occurrence,
        expect_count=expect_count,
        match_start=match_start,
        match_end=match_end,
        first_line=first_line,
        last_line=last_line,
        manual_indent=manual,
    )


def _boundary_from_line(
    source: SourceState,
    raw: Mapping[str, Any],
    index: int,
    *,
    field_name: str,
) -> Boundary:
    line_value = raw.get(field_name)
    if isinstance(line_value, bool) or not isinstance(line_value, int) or line_value < 1:
        raise GenerationError(
            "invalid_line",
            "{} must be an explicit positive integer".format(field_name),
            selection_index=index,
        )
    line = line_value
    if line > len(source.line_starts):
        raise GenerationError(
            "line_out_of_range",
            "{} is beyond the end of {} ({} line(s))".format(
                field_name, source.path, len(source.line_starts)
            ),
            selection_index=index,
            details={field_name: line},
        )
    line_start = source.line_starts[line - 1]
    content_end = source.line_content_ends[line - 1]
    line_text = source.text[line_start:content_end]
    indent_len = len(re.match(r"[ \t]*", line_text).group(0))
    anchor = line_text[indent_len:]
    if not anchor:
        raise GenerationError(
            "invalid_line_boundary",
            "{} selects an empty or whitespace-only line".format(field_name),
            selection_index=index,
            details={field_name: line},
        )
    matches = _anchor_matches(source.text, anchor)
    match_start = line_start + indent_len
    try:
        selected_occurrence = matches.index(match_start) + 1
    except ValueError as exc:
        raise GenerationError(
            "line_occurrence_mismatch",
            "{} selected line text is not an exact source anchor".format(field_name),
            selection_index=index,
            details={field_name: line},
        ) from exc

    if "expect_count" in raw:
        expect_count = _positive_int(
            raw.get("expect_count"), field_name="expect_count", index=index
        )
        if len(matches) != expect_count:
            raise GenerationError(
                "anchor_count_mismatch",
                "{} line text matched {} time(s), expected exactly {}".format(
                    field_name, len(matches), expect_count
                ),
                selection_index=index,
                details={"actual_count": len(matches), "expect_count": expect_count},
            )
    else:
        expect_count = len(matches)

    if "occurrence" in raw:
        occurrence = _positive_int(
            raw.get("occurrence"), field_name="occurrence", index=index
        )
        if occurrence != selected_occurrence:
            raise GenerationError(
                "line_occurrence_mismatch",
                "{} line {} is occurrence {}, not occurrence {}".format(
                    field_name, line, selected_occurrence, occurrence
                ),
                selection_index=index,
                details={
                    field_name: line,
                    "occurrence": occurrence,
                    "selected_occurrence": selected_occurrence,
                    "expect_count": expect_count,
                },
            )
    else:
        occurrence = selected_occurrence

    return Boundary(
        kind="line",
        anchor=anchor,
        position="after",
        occurrence=occurrence,
        expect_count=expect_count,
        match_start=match_start,
        match_end=content_end,
        first_line=line,
        last_line=line,
        manual_indent=_after_manual_indent(source, match_start, content_end),
    )


def _resolve_boundaries(
    source: SourceState, raw: Mapping[str, Any], index: int
) -> Tuple[Boundary, Boundary]:
    anchor_fields = [name for name in ("begin_anchor", "end_anchor") if name in raw]
    line_fields = [name for name in ("begin_line", "end_line") if name in raw]
    if anchor_fields and line_fields:
        raise GenerationError(
            "mixed_boundaries",
            "anchor boundaries and line boundaries are mutually exclusive",
            selection_index=index,
            details={"anchor_fields": anchor_fields, "line_fields": line_fields},
        )
    if not anchor_fields and not line_fields:
        raise GenerationError(
            "missing_end_boundary",
            "both begin and end boundaries must be explicit",
            selection_index=index,
        )
    if anchor_fields:
        if len(anchor_fields) != 2:
            raise GenerationError(
                "missing_end_boundary",
                "both begin_anchor and end_anchor are required",
                selection_index=index,
                details={"present": anchor_fields},
            )
        mode = "anchor"
    else:
        if len(line_fields) != 2:
            raise GenerationError(
                "missing_end_boundary",
                "both begin_line and end_line are required",
                selection_index=index,
                details={"present": line_fields},
            )
        mode = "line"

    if mode == "anchor":
        occurrence = _positive_int(
            raw.get("occurrence"), field_name="occurrence", index=index
        )
        expect_count = _positive_int(
            raw.get("expect_count"), field_name="expect_count", index=index
        )
        if occurrence > expect_count:
            raise GenerationError(
                "occurrence_out_of_range",
                "occurrence cannot exceed expect_count",
                selection_index=index,
                details={"occurrence": occurrence, "expect_count": expect_count},
            )
        begin = _boundary_from_anchor(
            source,
            raw,
            index,
            kind="anchor",
            field_name="begin_anchor",
            position="after",
            occurrence=occurrence,
            expect_count=expect_count,
        )
        end = _boundary_from_anchor(
            source,
            raw,
            index,
            kind="anchor",
            field_name="end_anchor",
            position="before",
            occurrence=occurrence,
            expect_count=expect_count,
        )
    else:
        begin = _boundary_from_line(
            source,
            raw,
            index,
            field_name="begin_line",
        )
        end = _boundary_from_line(
            source,
            raw,
            index,
            field_name="end_line",
        )

    if end.match_start < begin.match_start or end.match_end <= begin.match_start:
        raise GenerationError(
            "end_not_after_begin",
            "the explicit end boundary must come after the begin boundary",
            selection_index=index,
            details={
                "begin_line": begin.first_line,
                "end_line": end.first_line,
            },
        )
    if mode == "anchor" and end.match_start < begin.match_end:
        raise GenerationError(
            "overlapping_boundaries",
            "begin and end anchors overlap within one selection",
            selection_index=index,
        )
    if mode == "line" and end.first_line <= begin.first_line:
        raise GenerationError(
            "end_not_after_begin",
            "end_line must be greater than begin_line",
            selection_index=index,
            details={"begin_line": begin.first_line, "end_line": end.first_line},
        )
    return begin, end


def _candidate_indexes(
    candidates: Sequence[Mapping[str, Any]],
    scan: Mapping[str, Any],
) -> Tuple[Dict[Tuple[str, int], set[str]], Dict[Tuple[str, int], set[str]]]:
    at_location: Dict[Tuple[str, int], set[str]] = {}
    for item in candidates:
        key = (str(item["path"]), int(item["line"]))
        at_location.setdefault(key, set()).add(str(item["semantic_id"]))
    ambiguous = {key: set(values) for key, values in at_location.items()}

    raw_ambiguities = scan.get("ambiguities", [])
    if not isinstance(raw_ambiguities, list):
        raise GenerationError("invalid_scan", "scan.ambiguities must be an array")
    for index, item in enumerate(raw_ambiguities):
        if not isinstance(item, Mapping):
            raise GenerationError(
                "invalid_scan", "scan.ambiguities[{}] must be an object".format(index)
            )
        path = _safe_relative(
            item.get("path"), label="scan.ambiguities[{}].path".format(index)
        )
        line = item.get("line")
        ids = item.get("candidates")
        if isinstance(line, bool) or not isinstance(line, int) or line < 1:
            raise GenerationError(
                "invalid_scan",
                "scan.ambiguities[{}].line must be positive".format(index),
            )
        if not isinstance(ids, list) or not ids or any(
            not isinstance(value, str) or not value for value in ids
        ):
            raise GenerationError(
                "invalid_scan",
                "scan.ambiguities[{}].candidates must be a non-empty string array".format(
                    index
                ),
            )
        ambiguous.setdefault((path, line), set()).update(ids)
    return at_location, ambiguous


def _ensure_candidate(
    *,
    path: str,
    lines: Iterable[int],
    semantic_id: str,
    index: int,
    boundary_name: str,
    at_location: Mapping[Tuple[str, int], set[str]],
    ambiguous: Mapping[Tuple[str, int], set[str]],
) -> None:
    selected_lines = sorted(set(lines))
    observed: set[str] = set()
    for line in selected_lines:
        key = (path, line)
        observed.update(at_location.get(key, set()))
        observed.update(ambiguous.get(key, set()))
    if semantic_id not in observed:
        raise GenerationError(
            "semantic_not_candidate",
            "{} location is not a scan candidate for {!r}".format(
                boundary_name, semantic_id
            ),
            selection_index=index,
            details={
                "path": path,
                "lines": selected_lines,
                "semantic_id": semantic_id,
                "candidates": sorted(observed),
            },
        )


def _validate_range_ambiguities(
    pending: Sequence[PendingSelection],
    ambiguous: Mapping[Tuple[str, int], set[str]],
) -> None:
    # A scan can legitimately have several hint-level candidates on one line
    # (for example ``tl.load(..., mask=...)`` is both ``load`` and ``compute``).
    # The explicit selection resolves that ambiguity: it may choose one candidate,
    # but it must never silently select a semantic the scan did not find there,
    # and two selections may not assign different meanings to the same line.
    selected_by_line: Dict[Tuple[str, int], set[str]] = {}
    for item in pending:
        for (path, line), candidates in sorted(ambiguous.items()):
            if len(candidates) <= 1 or path != item.source.path:
                continue
            if item.begin.first_line <= line <= item.end.last_line:
                if item.semantic_id not in candidates:
                    raise GenerationError(
                        "ambiguous_scan_location",
                        "selected range contains an ambiguous scan location "
                        "without the selected semantic",
                        selection_index=item.index,
                        details={
                            "path": path,
                            "line": line,
                            "semantic_id": item.semantic_id,
                            "candidates": sorted(candidates),
                        },
                    )
                selected_by_line.setdefault((path, line), set()).add(
                    item.semantic_id
                )
    conflicts = {
        key: sorted(value)
        for key, value in selected_by_line.items()
        if len(value) > 1
    }
    if conflicts:
        (path, line), semantic_ids = sorted(conflicts.items())[0]
        raise GenerationError(
            "conflicting_ambiguous_selection",
            "one ambiguous source line was selected with multiple semantics",
            details={
                "path": path,
                "line": line,
                "semantic_ids": semantic_ids,
            },
        )


def _validate_overlaps(pending: Sequence[PendingSelection]) -> None:
    by_path: Dict[str, List[PendingSelection]] = {}
    for item in pending:
        by_path.setdefault(item.source.path, []).append(item)
    for path, items in sorted(by_path.items()):
        ordered = sorted(items, key=lambda item: (item.range_start, item.range_end, item.index))
        furthest = ordered[0]
        for current in ordered[1:]:
            if current.range_start < furthest.range_end:
                raise GenerationError(
                    "overlapping_ranges",
                    "selected ranges overlap in {}".format(path),
                    selection_index=current.index,
                    details={
                        "path": path,
                        "other_selection_index": furthest.index,
                        "other_range": [furthest.range_start, furthest.range_end],
                        "range": [current.range_start, current.range_end],
                    },
                )
            if current.range_end > furthest.range_end:
                furthest = current


def _resolve_event_ids(
    pending: Sequence[PendingSelection],
    canonical_event_ids: Mapping[str, int],
) -> Tuple[Dict[int, int], Tuple[Dict[str, Any], ...]]:
    resolved: Dict[int, int] = {}
    report: List[Dict[str, Any]] = []
    for item in pending:
        canonical = canonical_event_ids.get(item.semantic_id)
        if canonical is None:
            raise GenerationError(
                "canonical_event_id_missing",
                "semantic_id {!r} has no canonical event_id".format(item.semantic_id),
                selection_index=item.index,
                details={"semantic_id": item.semantic_id},
            )
        if item.event_id is not None and item.event_id != canonical:
            raise GenerationError(
                "event_id_semantic_conflict",
                "event_id {} conflicts with canonical event_id {} for {!r}".format(
                    item.event_id, canonical, item.semantic_id
                ),
                selection_index=item.index,
                details={
                    "semantic_id": item.semantic_id,
                    "event_id": item.event_id,
                    "canonical_event_id": canonical,
                },
            )
        if item.event_id is None:
            item.event_id = canonical
            item.allocated = True
            report.append(
                {
                    "selection_index": item.index,
                    "semantic_id": item.semantic_id,
                    "event_id": canonical,
                    "source": "canonical_event_id",
                }
            )
        resolved[item.index] = canonical
    report.sort(key=lambda item: int(item["selection_index"]))
    return resolved, tuple(report)


def _fingerprint(pending: PendingSelection) -> str:
    payload = "\x00".join(
        [
            pending.source.path,
            pending.semantic_id,
            pending.begin.kind,
            str(pending.begin.match_start),
            pending.begin.anchor,
            str(pending.end.match_start),
            pending.end.anchor,
            str(pending.begin.occurrence),
            str(pending.begin.expect_count),
            pending.aux,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _triton_frontend(scan: Mapping[str, Any]) -> str:
    """Return the selected Triton recorder family from scan route evidence."""
    route = scan.get("route")
    route = route if isinstance(route, Mapping) else {}
    target = route.get("target")
    target = target if isinstance(target, Mapping) else {}
    platform = str(target.get("platform") or "").lower()
    recorder = str(route.get("recorder") or "").lower()
    if platform == "amd" or "fi_triton_amd" in recorder:
        return "amd"
    return "nv"


def _call_text(language: str, kind: str, event_id: int, aux: str,
               triton_frontend: str = "nv") -> str:
    if language in CPP_LANGUAGES:
        return "FI_{}({}, {});".format(kind.upper(), event_id, aux)
    if language == "triton":
        if triton_frontend == "amd":
            return (
                "n = fi_{}(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, "
                "selected, n, EVENT={}, aux={}, CAP=FI_CAP, THREADS=FI_THREADS, "
                "TIMER=FI_TIMER)"
            ).format(kind, event_id, aux)
        return "n = fi_{}(rec, ovf, pid, n, FI, ev={}, aux={})".format(
            kind, event_id, aux
        )
    if language == "cutedsl":
        return "ctx = fi_{}(ctx, {}, {})".format(kind, event_id, aux)
    raise GenerationError(
        "unsupported_language", "no call syntax for {!r}".format(language)
    )


def _edit_text(
    source: SourceState, boundary: Boundary, call: str
) -> str:
    if boundary.position == "after":
        return source.newline + boundary.manual_indent + call
    return boundary.manual_indent + call + source.newline


def _build_plan(
    pending: Sequence[PendingSelection], root: Path, language: str,
    triton_frontend: str = "nv",
) -> Dict[str, Any]:
    by_path: Dict[str, Dict[str, Any]] = {}
    ordered_edits: Dict[str, List[Tuple[int, int, Dict[str, Any]]]] = {}
    for item in pending:
        assert item.event_id is not None
        item.id_fingerprint = _fingerprint(item)
        group = by_path.setdefault(
            item.source.path,
            {
                "path": item.source.path,
                "base_sha256": item.source.base_sha256,
                "edits": [],
            },
        )
        if group["base_sha256"] != item.source.base_sha256:
            raise GenerationError(
                "inconsistent_scan_hash",
                "scan contains inconsistent hashes for {}".format(item.source.path),
                selection_index=item.index,
            )
        edits = ordered_edits.setdefault(item.source.path, [])
        begin_call = _call_text(
            language, "b", item.event_id, item.aux, triton_frontend
        )
        end_call = _call_text(
            language, "e", item.event_id, item.aux, triton_frontend
        )
        begin_edit = {
            "id": "fi-{:04d}-{}-begin".format(item.event_id, item.id_fingerprint),
            "anchor": item.begin.anchor,
            "position": item.begin.position,
            "text": _edit_text(item.source, item.begin, begin_call),
            "occurrence": item.begin.occurrence,
            "expect_count": item.begin.expect_count,
            "indent": "anchor",
        }
        end_edit = {
            "id": "fi-{:04d}-{}-end".format(item.event_id, item.id_fingerprint),
            "anchor": item.end.anchor,
            "position": item.end.position,
            "text": _edit_text(item.source, item.end, end_call),
            "occurrence": item.end.occurrence,
            "expect_count": item.end.expect_count,
            "indent": "anchor",
        }
        begin_offset = (
            item.begin.match_end
            if item.begin.position == "after"
            else item.begin.match_start
        )
        end_offset = (
            item.end.match_end
            if item.end.position == "after"
            else item.end.match_start
        )
        edits.append((begin_offset, 0, begin_edit))
        edits.append((end_offset, 1, end_edit))

    files: List[Dict[str, Any]] = []
    for path in sorted(by_path):
        group = by_path[path]
        group["edits"] = [
            edit for _, _, edit in sorted(
                ordered_edits[path],
                key=lambda value: (
                    value[0],
                    value[1],
                    value[2]["id"],
                ),
            )
        ]
        files.append(group)
    plan = {
        "schema": "weaveprofiler.patch-plan",
        "version": 1,
        "project_root": str(root),
        "files": files,
    }
    return plan


def _build_names(pending: Sequence[PendingSelection]) -> Dict[str, Any]:
    event_semantics: Dict[int, str] = {}
    for item in pending:
        assert item.event_id is not None
        previous = event_semantics.get(item.event_id)
        if previous is not None and previous != item.semantic_id:
            raise GenerationError(
                "duplicate_event_id_semantics",
                "event_id {} is assigned to both {!r} and {!r}".format(
                    item.event_id, previous, item.semantic_id
                ),
                selection_index=item.index,
                details={
                    "event_id": item.event_id,
                    "semantic_ids": sorted({previous, item.semantic_id}),
                },
            )
        event_semantics[item.event_id] = item.semantic_id
    event_ids = sorted(event_semantics)
    names = {
        "events": {str(event_id): event_semantics[event_id] for event_id in event_ids},
        "semantics": {
            str(event_id): {"semantic_id": event_semantics[event_id]}
            for event_id in event_ids
        },
    }
    return names


def _check_plan(plan: Mapping[str, Any]) -> str:
    apply_plan = _apply_plan_module()
    result = apply_plan.apply_patch_plan(dict(plan), check=True)
    if not result.get("ok"):
        error = (result.get("errors") or [{}])[0]
        raise GenerationError(
            "apply_plan_rejected",
            "generated plan failed apply_plan check: {}".format(
                error.get("message", "unknown error")
            ),
            details={"apply_plan_error": error},
        )
    return str(result.get("diff") or "")


def generate_edits(
    scan_document: Mapping[str, Any],
    selection_document: Mapping[str, Any],
    *,
    project_root_override: Optional[str | os.PathLike[str]] = None,
    validate: bool = True,
) -> GeneratedEdits:
    """Build one validated patch plan and canonical names sidecar document."""

    if not isinstance(scan_document, Mapping):
        raise GenerationError("invalid_scan", "scan must be a JSON object")
    root = _project_root(scan_document, project_root_override)
    language, evidence, candidates = _scan_evidence(scan_document, root)
    selections = _selection_document(selection_document)
    kb = _load_semantic_kb()
    canonical_ids = {str(row["semantic_id"]) for row in kb["semantics"]}
    canonical_event_ids = _canonical_event_id_map(kb)
    at_location, ambiguous = _candidate_indexes(candidates, scan_document)

    sources: Dict[str, SourceState] = {}
    pending: List[PendingSelection] = []
    explicit_semantics: Dict[int, str] = {}
    for index, raw in enumerate(selections):
        path = _safe_relative(
            raw.get("path"), label="selection.path", selection_index=index
        )
        if path not in evidence:
            raise GenerationError(
                "path_not_in_scan",
                "selection path is not present in scan.files",
                selection_index=index,
                details={"path": path},
            )
        semantic_id = _canonical_semantic(raw, canonical_ids, index)
        aux = _selection_aux(raw, index, language)
        event_id = _selection_event_id(
            raw,
            index,
            semantic_id=semantic_id,
            canonical_event_ids=canonical_event_ids,
        )
        if event_id is not None:
            previous = explicit_semantics.get(event_id)
            if previous is not None and previous != semantic_id:
                raise GenerationError(
                    "duplicate_event_id_semantics",
                    "explicit event_id {} is assigned to both {!r} and {!r}".format(
                        event_id, previous, semantic_id
                    ),
                    selection_index=index,
                    details={
                        "event_id": event_id,
                        "semantic_ids": sorted({previous, semantic_id}),
                    },
                )
            explicit_semantics[event_id] = semantic_id

        source = sources.get(path)
        if source is None:
            source = _load_source(root, path, evidence[path])
            sources[path] = source
        begin, end = _resolve_boundaries(source, raw, index)
        _ensure_candidate(
            path=path,
            lines=range(begin.first_line, begin.last_line + 1),
            semantic_id=semantic_id,
            index=index,
            boundary_name="begin",
            at_location=at_location,
            ambiguous=ambiguous,
        )
        _ensure_candidate(
            path=path,
            lines=range(end.first_line, end.last_line + 1),
            semantic_id=semantic_id,
            index=index,
            boundary_name="end",
            at_location=at_location,
            ambiguous=ambiguous,
        )
        pending.append(
            PendingSelection(
                index=index,
                source=source,
                semantic_id=semantic_id,
                aux=aux,
                event_id=event_id,
                begin=begin,
                end=end,
                range_start=begin.match_start,
                range_end=end.match_end,
            )
        )

    _validate_range_ambiguities(pending, ambiguous)
    _validate_overlaps(pending)
    _, allocations = _resolve_event_ids(pending, canonical_event_ids)
    names = _build_names(pending)
    resolved_names = semantics.resolve_names(names, kb)
    if not resolved_names.get("ok"):
        raise GenerationError(
            "invalid_names_sidecar",
            "canonical names document failed semantic resolution",
            details={"errors": resolved_names.get("errors", [])},
        )
    plan = _build_plan(pending, root, language, _triton_frontend(scan_document))
    check_diff = _check_plan(plan) if validate else ""
    return GeneratedEdits(
        plan=plan,
        names=names,
        allocations=allocations,
        project_root=str(root),
        language=language,
        check_diff=check_diff,
    )


def write_generated_edits(
    generated: GeneratedEdits,
    out: str | os.PathLike[str],
    *,
    names_out: Optional[str | os.PathLike[str]] = None,
    write_names: bool = True,
) -> Optional[Path]:
    """Serialize and validate the plan, then atomically publish plan and names."""

    out_path = Path(out).expanduser().resolve()
    if names_out is None:
        names_path: Optional[Path] = Path(str(out_path) + ".names.json")
    else:
        names_path = Path(names_out).expanduser().resolve()
    if not write_names:
        names_path = None

    plan_temp: Optional[Path] = None
    names_temp: Optional[Path] = None
    try:
        plan_temp = _atomic_replace_bytes(out_path, _json_bytes(generated.plan))
        apply_plan = _apply_plan_module()
        try:
            loaded = apply_plan.load_plan(plan_temp)
        except Exception as exc:
            raise GenerationError(
                "plan_load_failed",
                "generated plan cannot be loaded: {}".format(exc),
            ) from exc
        result = apply_plan.apply_patch_plan(loaded, check=True)
        if not result.get("ok"):
            error = (result.get("errors") or [{}])[0]
            raise GenerationError(
                "apply_plan_rejected",
                "generated output failed apply_plan --check semantics: {}".format(
                    error.get("message", "unknown error")
                ),
                details={"apply_plan_error": error},
            )
        if names_path is not None:
            names_temp = _atomic_replace_bytes(names_path, _json_bytes(generated.names))
        os.replace(str(plan_temp), str(out_path))
        plan_temp = None
        if names_path is not None and names_temp is not None:
            os.replace(str(names_temp), str(names_path))
            names_temp = None
        return names_path
    finally:
        for temp in (plan_temp, names_temp):
            if temp is not None:
                try:
                    temp.unlink()
                except OSError:
                    pass


def _selftest() -> int:
    work = Path(tempfile.mkdtemp(prefix="weaveprofiler-generate-edits-selftest-"))
    try:
        source_path = work / "kernel.cu"
        source_path.write_text(
            "void kernel() {\n"
            "  mma_sync(acc, a, b);\n"
            "  store(acc, out);\n"
            "}\n",
            encoding="utf-8",
        )
        payload = source_path.read_bytes()
        scan = {
            "schema": SCAN_SCHEMA,
            "project_root": str(work),
            "route": {"language": "cuda_cpp"},
            "files": [
                {
                    "path": "kernel.cu",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ],
            "candidates": [
                {"path": "kernel.cu", "line": 2, "semantic_id": "mma"},
                {"path": "kernel.cu", "line": 3, "semantic_id": "mma"},
            ],
            "ambiguities": [],
        }
        selection = {
            "schema": SELECTION_SCHEMA,
            "version": SELECTION_VERSION,
            "selections": [
                {
                    "path": "kernel.cu",
                    "semantic_id": "mma",
                    "begin_anchor": "mma_sync(acc, a, b);",
                    "end_anchor": "store(acc, out);",
                    "occurrence": 1,
                    "expect_count": 1,
                    "aux": "tile_id",
                }
            ],
        }
        generated = generate_edits(scan, selection)
        out = work / "edits.json"
        write_generated_edits(generated, out)
        print(
            json.dumps(
                generated.summary(out, Path(str(out) + ".names.json")),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic SHA-bound weaveprofiler instrumentation edits"
    )
    parser.add_argument("--scan", help="weaveprofiler scan JSON")
    parser.add_argument("--selection", help="explicit instrumentation selection JSON")
    parser.add_argument("--out", help="output patch-plan JSON")
    parser.add_argument(
        "--project-root",
        help="project root override; allowed only when scan project_root is unavailable",
    )
    parser.add_argument(
        "--no-names",
        action="store_true",
        help="do not write the <out>.names.json canonical decoder mapping",
    )
    parser.add_argument("--selftest", action="store_true", help="run an isolated smoke test")
    args = parser.parse_args(argv)

    if args.selftest:
        try:
            return _selftest()
        except GenerationError as exc:
            print(json.dumps({"ok": False, "error": exc.as_dict()}, ensure_ascii=False))
            return 1
    if not args.scan or not args.selection or not args.out:
        parser.error("--scan, --selection, and --out are required unless --selftest is used")

    try:
        scan = _read_json(Path(args.scan).expanduser().resolve(), "scan")
        selection = _read_json(
            Path(args.selection).expanduser().resolve(), "selection"
        )
        generated = generate_edits(
            scan,
            selection,
            project_root_override=args.project_root,
        )
        out = Path(args.out).expanduser().resolve()
        names_path = write_generated_edits(
            generated, out, write_names=not args.no_names
        )
        print(
            json.dumps(
                generated.summary(out, names_path),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except GenerationError as exc:
        print(
            json.dumps(
                {"schema": RESULT_SCHEMA, "ok": False, "error": exc.as_dict()},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    except OSError as exc:
        error = GenerationError("io_error", str(exc))
        print(
            json.dumps(
                {"schema": RESULT_SCHEMA, "ok": False, "error": error.as_dict()},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
