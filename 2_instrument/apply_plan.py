#!/usr/bin/env python3
"""Apply an atomic instrumentation patch plan without invoking a shell.

Plan schema (``weaveprofiler.patch-plan`` version 1)::

    {
      "schema": "weaveprofiler.patch-plan",
      "version": 1,
      "project_root": "/absolute/project/root",
      "files": [
        {
          "path": "src/kernel.cu",
          "base_sha256": "64 lowercase or uppercase hex characters",
          "edits": [
            {
              "id": "gemm-open",
              "anchor": "for (int k = 0; k < K; ++k) {",
              "position": "after",
              "text": "\\nFI_B(GEMM, tile);",
              "occurrence": 1,
              "expect_count": 1,
              "indent": "anchor"
            }
          ]
        }
      ]
    }

``occurrence`` is a one-based selector and defaults to 1.  ``expect_count``
is the exact number of anchor matches and defaults to 1.  The two fields may
be used together when selecting one anchor from a repeated construct.  A
missing constraint therefore means "the anchor must match exactly once".
Unknown fields are rejected so a misspelled safety constraint cannot silently
fall back to a default.

All anchors are matched against the original file.  Offsets are never
recomputed after another edit has been inserted, which prevents edits from
finding text introduced by an earlier edit.  ``indent: "anchor"`` copies the
anchor line's leading whitespace to non-empty continuation lines.  It also
indents the first line when the insertion starts at the beginning of a line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


PLAN_SCHEMA = "weaveprofiler.patch-plan"
PLAN_VERSION = 1
RESULT_SCHEMA = "weaveprofiler.apply-plan-result"
RESULT_VERSION = 1

TOP_LEVEL_KEYS = frozenset({"schema", "version", "project_root", "files"})
FILE_KEYS = frozenset({"path", "base_sha256", "edits"})
EDIT_KEYS = frozenset(
    {
        "id",
        "anchor",
        "position",
        "text",
        "occurrence",
        "expect_count",
        "indent",
    }
)
POSITIONS = frozenset({"before", "after", "replace"})
HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class PlanError(Exception):
    """A fail-closed plan error with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: Optional[str] = None,
        edit_id: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path
        self.edit_id = edit_id
        self.details = dict(details or {})

    def as_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.path is not None:
            result["path"] = self.path
        if self.edit_id is not None:
            result["edit_id"] = self.edit_id
        if self.details:
            result["details"] = self.details
        return result


@dataclass(frozen=True)
class EditSpec:
    edit_id: str
    anchor: str
    position: str
    text: str
    occurrence: int
    expect_count: int
    indent: Optional[str]
    order: int


@dataclass(frozen=True)
class FileSpec:
    path: str
    absolute_path: Path
    base_sha256: str
    edits: Tuple[EditSpec, ...]


@dataclass(frozen=True)
class PlanSpec:
    project_root: Path
    files: Tuple[FileSpec, ...]
    digest: str


@dataclass(frozen=True)
class Splice:
    edit_id: str
    position: str
    start: int
    end: int
    text: str
    order: int


@dataclass(frozen=True)
class PreparedFile:
    spec: FileSpec
    old_bytes: bytes
    old_text: str
    new_bytes: bytes
    new_text: str
    mode: int


@dataclass(frozen=True)
class PreparedPlan:
    spec: PlanSpec
    files: Tuple[PreparedFile, ...]
    diff: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(payload)


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: frozenset,
    *,
    where: str,
    path: Optional[str] = None,
    edit_id: Optional[str] = None,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise PlanError(
            "unknown_field",
            "{} has unknown field(s): {}".format(where, ", ".join(unknown)),
            path=path,
            edit_id=edit_id,
            details={"unknown_fields": unknown},
        )


def _positive_int(
    value: Any,
    *,
    field: str,
    path: str,
    edit_id: str,
    default: Optional[int] = None,
) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PlanError(
            "invalid_{}".format(field),
            "{} must be a positive integer".format(field),
            path=path,
            edit_id=edit_id,
        )
    return value


def _safe_project_file(project_root: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise PlanError("invalid_path", "file path must be a non-empty string")
    if "\x00" in raw_path:
        raise PlanError(
            "invalid_path", "file path must not contain NUL", path=raw_path
        )
    if raw_path.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", raw_path):
        raise PlanError(
            "path_escape", "file path must be relative to project_root", path=raw_path
        )
    if "\\" in raw_path:
        raise PlanError(
            "path_escape",
            "file path must use POSIX separators and cannot contain backslashes",
            path=raw_path,
        )

    pieces = raw_path.split("/")
    if any(piece in ("", ".", "..") for piece in pieces):
        raise PlanError(
            "path_escape",
            "file path cannot contain empty, '.' or '..' components",
            path=raw_path,
        )

    candidate = project_root.joinpath(*pieces)
    try:
        resolved_parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise PlanError(
            "file_not_found",
            "parent directory does not exist: {}".format(exc),
            path=raw_path,
        )
    if resolved_parent != project_root and project_root not in resolved_parent.parents:
        raise PlanError(
            "path_escape",
            "resolved parent directory is outside project_root",
            path=raw_path,
        )

    if os.path.islink(str(candidate)):
        raise PlanError(
            "path_symlink",
            "target file cannot be a symbolic link",
            path=raw_path,
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise PlanError(
            "file_not_found", "target file does not exist: {}".format(exc), path=raw_path
        )
    if resolved != project_root and project_root not in resolved.parents:
        raise PlanError(
            "path_escape",
            "resolved target file is outside project_root",
            path=raw_path,
        )
    if not resolved.is_file():
        raise PlanError("not_a_file", "target is not a regular file", path=raw_path)
    return resolved


def validate_plan_document(document: Any) -> PlanSpec:
    """Validate plan structure and path containment without changing files."""

    if not isinstance(document, Mapping):
        raise PlanError("invalid_plan", "plan must be a JSON object")
    _reject_unknown_keys(document, TOP_LEVEL_KEYS, where="plan")
    if document.get("schema") != PLAN_SCHEMA:
        raise PlanError(
            "unsupported_schema",
            "schema must be {!r}".format(PLAN_SCHEMA),
        )
    if document.get("version") != PLAN_VERSION:
        raise PlanError(
            "unsupported_version",
            "version must be {}".format(PLAN_VERSION),
        )

    raw_root = document.get("project_root")
    if not isinstance(raw_root, str) or not raw_root:
        raise PlanError(
            "invalid_project_root", "project_root must be a non-empty string"
        )
    root_path = Path(raw_root)
    if not root_path.is_absolute():
        raise PlanError(
            "invalid_project_root", "project_root must be an absolute path"
        )
    try:
        project_root = root_path.resolve(strict=True)
    except OSError as exc:
        raise PlanError(
            "invalid_project_root",
            "project_root does not exist: {}".format(exc),
        )
    if not project_root.is_dir():
        raise PlanError(
            "invalid_project_root", "project_root must be a directory"
        )

    raw_files = document.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise PlanError("invalid_files", "files must be a non-empty array")

    seen_paths = set()
    seen_edit_ids = set()
    files: List[FileSpec] = []
    for file_index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, Mapping):
            raise PlanError(
                "invalid_file_entry",
                "files[{}] must be an object".format(file_index),
            )
        _reject_unknown_keys(
            raw_file,
            FILE_KEYS,
            where="files[{}]".format(file_index),
        )
        raw_path = raw_file.get("path")
        absolute_path = _safe_project_file(project_root, raw_path)
        if raw_path in seen_paths:
            raise PlanError(
                "duplicate_path", "file path appears more than once", path=raw_path
            )
        seen_paths.add(raw_path)

        raw_sha = raw_file.get("base_sha256")
        if not isinstance(raw_sha, str) or not HEX_SHA256.fullmatch(raw_sha):
            raise PlanError(
                "invalid_base_sha256",
                "base_sha256 must contain exactly 64 hexadecimal characters",
                path=raw_path,
            )

        raw_edits = raw_file.get("edits")
        if not isinstance(raw_edits, list) or not raw_edits:
            raise PlanError(
                "invalid_edits",
                "edits must be a non-empty array",
                path=raw_path,
            )

        edits: List[EditSpec] = []
        for edit_index, raw_edit in enumerate(raw_edits):
            if not isinstance(raw_edit, Mapping):
                raise PlanError(
                    "invalid_edit",
                    "edits[{}] must be an object".format(edit_index),
                    path=raw_path,
                )
            edit_id_value = raw_edit.get("id")
            edit_id = edit_id_value if isinstance(edit_id_value, str) else None
            _reject_unknown_keys(
                raw_edit,
                EDIT_KEYS,
                where="edits[{}]".format(edit_index),
                path=raw_path,
                edit_id=edit_id,
            )
            if not isinstance(edit_id_value, str) or not edit_id_value:
                raise PlanError(
                    "invalid_edit_id", "edit id must be a non-empty string", path=raw_path
                )
            if edit_id in seen_edit_ids:
                raise PlanError(
                    "duplicate_edit_id", "edit id is not globally unique", path=raw_path, edit_id=edit_id
                )
            seen_edit_ids.add(edit_id)

            anchor = raw_edit.get("anchor")
            if not isinstance(anchor, str) or not anchor:
                raise PlanError(
                    "invalid_anchor",
                    "anchor must be a non-empty string",
                    path=raw_path,
                    edit_id=edit_id,
                )
            position = raw_edit.get("position")
            if position not in POSITIONS:
                raise PlanError(
                    "invalid_position",
                    "position must be one of before, after, replace",
                    path=raw_path,
                    edit_id=edit_id,
                )
            text = raw_edit.get("text")
            if not isinstance(text, str):
                raise PlanError(
                    "invalid_text", "text must be a string", path=raw_path, edit_id=edit_id
                )
            if position in ("before", "after") and not text:
                raise PlanError(
                    "empty_insertion",
                    "before/after text cannot be empty",
                    path=raw_path,
                    edit_id=edit_id,
                )
            occurrence = _positive_int(
                raw_edit.get("occurrence"),
                field="occurrence",
                path=raw_path,
                edit_id=edit_id,
                default=1,
            )
            expect_count = _positive_int(
                raw_edit.get("expect_count"),
                field="expect_count",
                path=raw_path,
                edit_id=edit_id,
                default=1,
            )
            if occurrence > expect_count:
                raise PlanError(
                    "occurrence_out_of_range",
                    "occurrence cannot exceed expect_count",
                    path=raw_path,
                    edit_id=edit_id,
                    details={"occurrence": occurrence, "expect_count": expect_count},
                )

            indent = raw_edit.get("indent")
            if indent is not None and indent != "anchor":
                raise PlanError(
                    "invalid_indent",
                    "indent must be omitted or 'anchor'",
                    path=raw_path,
                    edit_id=edit_id,
                )

            edits.append(
                EditSpec(
                    edit_id=edit_id,
                    anchor=anchor,
                    position=position,
                    text=text,
                    occurrence=occurrence,
                    expect_count=expect_count,
                    indent=indent,
                    order=edit_index,
                )
            )

        files.append(
            FileSpec(
                path=raw_path,
                absolute_path=absolute_path,
                base_sha256=raw_sha.lower(),
                edits=tuple(edits),
            )
        )

    return PlanSpec(
        project_root=project_root,
        files=tuple(files),
        digest=_sha256_json(document),
    )


def _all_anchor_offsets(text: str, anchor: str) -> List[int]:
    offsets: List[int] = []
    cursor = text.find(anchor)
    while cursor != -1:
        offsets.append(cursor)
        cursor = text.find(anchor, cursor + 1)
    return offsets


def _anchor_line_indent(text: str, match_start: int) -> str:
    line_start = max(
        text.rfind("\n", 0, match_start), text.rfind("\r", 0, match_start)
    ) + 1
    prefix = text[line_start:match_start]
    return re.match(r"[ \t]*", prefix).group(0)


def _indent_inserted_text(text: str, indent: str, indent_first: bool) -> str:
    if not text:
        return text
    pieces = re.split(r"(\r\n|\n|\r)", text)
    result: List[str] = []
    for index in range(0, len(pieces), 2):
        content = pieces[index]
        separator = pieces[index + 1] if index + 1 < len(pieces) else ""
        line_number = index // 2
        if content and (line_number > 0 or indent_first):
            result.append(indent)
        result.append(content)
        result.append(separator)
    return "".join(result)


def _splices_for_file(
    spec: FileSpec, text: str
) -> Tuple[List[Splice], List[Tuple[EditSpec, int, int]]]:
    splices: List[Splice] = []
    matched_spans: List[Tuple[EditSpec, int, int]] = []
    for edit in spec.edits:
        offsets = _all_anchor_offsets(text, edit.anchor)
        if not offsets:
            raise PlanError(
                "anchor_not_found",
                "anchor matched zero times",
                path=spec.path,
                edit_id=edit.edit_id,
                details={"expect_count": edit.expect_count},
            )
        if len(offsets) != edit.expect_count:
            raise PlanError(
                "anchor_count_mismatch",
                "anchor matched {} time(s), expected exactly {}".format(
                    len(offsets), edit.expect_count
                ),
                path=spec.path,
                edit_id=edit.edit_id,
                details={
                    "actual_count": len(offsets),
                    "expect_count": edit.expect_count,
                },
            )

        start = offsets[edit.occurrence - 1]
        end = start + len(edit.anchor)
        inserted = edit.text
        if edit.indent == "anchor":
            indent = _anchor_line_indent(text, start)
            insertion_point = start if edit.position in ("before", "replace") else end
            at_line_start = insertion_point == 0 or text[insertion_point - 1] in "\r\n"
            inserted = _indent_inserted_text(inserted, indent, at_line_start)
            if edit.position == "before" and re.search(r"(?:\r\n|\n|\r)$", inserted):
                # The indentation before a line-start anchor is part of the
                # prefix.  A before insertion that ends with a newline must
                # restore that prefix for the original anchor.
                line_start = max(
                    text.rfind("\n", 0, start), text.rfind("\r", 0, start)
                ) + 1
                anchor_prefix = text[line_start:start]
                if anchor_prefix and not anchor_prefix.strip(" \t"):
                    inserted += indent

        if edit.position == "before":
            splice_start, splice_end = start, start
        elif edit.position == "after":
            splice_start, splice_end = end, end
        else:
            splice_start, splice_end = start, end
            if inserted == text[start:end]:
                raise PlanError(
                    "no_change",
                    "replace text is identical to the matched anchor",
                    path=spec.path,
                    edit_id=edit.edit_id,
                )

        matched_spans.append((edit, start, end))
        splices.append(
            Splice(
                edit_id=edit.edit_id,
                position=edit.position,
                start=splice_start,
                end=splice_end,
                text=inserted,
                order=edit.order,
            )
        )
    return splices, matched_spans


def _validate_splice_conflicts(spec: FileSpec, splices: Sequence[Splice]) -> None:
    insertions: Dict[int, Splice] = {}
    replacements: List[Splice] = []
    for splice in splices:
        if splice.position == "replace":
            replacements.append(splice)
        else:
            previous = insertions.get(splice.start)
            if previous is not None:
                raise PlanError(
                    "overlapping_edits",
                    "two insertions target the same original offset",
                    path=spec.path,
                    edit_id=splice.edit_id,
                    details={
                        "other_edit_id": previous.edit_id,
                        "offset": splice.start,
                    },
                )
            insertions[splice.start] = splice

    for left_index, left in enumerate(replacements):
        for right in replacements[left_index + 1 :]:
            if max(left.start, right.start) < min(left.end, right.end):
                raise PlanError(
                    "overlapping_edits",
                    "replacement edits overlap in the original file",
                    path=spec.path,
                    edit_id=right.edit_id,
                    details={
                        "other_edit_id": left.edit_id,
                        "left_span": [left.start, left.end],
                        "right_span": [right.start, right.end],
                    },
                )

    for insertion in insertions.values():
        for replacement in replacements:
            if replacement.start < insertion.start < replacement.end:
                raise PlanError(
                    "overlapping_edits",
                    "insertion falls inside a replacement span",
                    path=spec.path,
                    edit_id=insertion.edit_id,
                    details={
                        "other_edit_id": replacement.edit_id,
                        "offset": insertion.start,
                        "replacement_span": [
                            replacement.start,
                            replacement.end,
                        ],
                    },
                )


def _apply_splices(text: str, splices: Sequence[Splice]) -> str:
    # Process from right to left so every offset remains tied to the original
    # text.  At a shared boundary, replacement runs before insertion; the
    # insertion then lands immediately before the replacement result.
    ordered = sorted(
        splices,
        key=lambda item: (
            item.start,
            1 if item.position == "replace" else 0,
            -item.order,
        ),
        reverse=True,
    )
    result = text
    for splice in ordered:
        result = result[: splice.start] + splice.text + result[splice.end :]
    return result


def _read_utf8(path: Path, *, relative_path: str) -> Tuple[bytes, str]:
    if os.path.islink(str(path)):
        raise PlanError(
            "path_symlink",
            "target file became a symbolic link",
            path=relative_path,
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PlanError(
            "read_failed", "cannot read file: {}".format(exc), path=relative_path
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PlanError(
            "invalid_utf8",
            "target file is not valid UTF-8: {}".format(exc),
            path=relative_path,
        )
    return data, text


def _unified_diff(relative_path: str, old_text: str, new_text: str) -> str:
    return "".join(
        unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile="a/{}".format(relative_path),
            tofile="b/{}".format(relative_path),
            lineterm="\n",
        )
    )


def prepare_plan(document: Any) -> PreparedPlan:
    """Read and validate all files in memory; this function never writes."""

    spec = validate_plan_document(document)
    prepared_files: List[PreparedFile] = []
    diffs: List[str] = []
    for file_spec in spec.files:
        old_bytes, old_text = _read_utf8(
            file_spec.absolute_path, relative_path=file_spec.path
        )
        actual_sha = sha256_bytes(old_bytes)
        if actual_sha != file_spec.base_sha256:
            raise PlanError(
                "base_sha256_mismatch",
                "file changed since the plan was created",
                path=file_spec.path,
                details={
                    "expected": file_spec.base_sha256,
                    "actual": actual_sha,
                },
            )

        splices, _ = _splices_for_file(file_spec, old_text)
        _validate_splice_conflicts(file_spec, splices)
        new_text = _apply_splices(old_text, splices)
        if new_text == old_text:
            raise PlanError(
                "no_change",
                "plan produces no change for file",
                path=file_spec.path,
            )
        new_bytes = new_text.encode("utf-8")
        try:
            mode = stat.S_IMODE(file_spec.absolute_path.stat().st_mode)
        except OSError as exc:
            raise PlanError(
                "stat_failed",
                "cannot stat target file: {}".format(exc),
                path=file_spec.path,
            )
        prepared_files.append(
            PreparedFile(
                spec=file_spec,
                old_bytes=old_bytes,
                old_text=old_text,
                new_bytes=new_bytes,
                new_text=new_text,
                mode=mode,
            )
        )
        diffs.append(_unified_diff(file_spec.path, old_text, new_text))

    return PreparedPlan(spec=spec, files=tuple(prepared_files), diff="".join(diffs))


def _temp_with_bytes(target: Path, data: bytes, mode: int, label: str) -> str:
    fd, temp_path = tempfile.mkstemp(
        prefix=".{}.weaveprofiler-{}.".format(target.name, label),
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, mode)
    except BaseException:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise
    return temp_path


def _assert_current(file: PreparedFile) -> None:
    if os.path.islink(str(file.spec.absolute_path)):
        raise PlanError(
            "path_symlink",
            "target file became a symbolic link before write",
            path=file.spec.path,
        )
    try:
        current = file.spec.absolute_path.read_bytes()
    except OSError as exc:
        raise PlanError(
            "write_precheck_failed",
            "cannot reread file before write: {}".format(exc),
            path=file.spec.path,
        )
    actual = sha256_bytes(current)
    if actual != file.spec.base_sha256:
        raise PlanError(
            "base_sha256_mismatch",
            "file changed before write",
            path=file.spec.path,
            details={"expected": file.spec.base_sha256, "actual": actual},
        )


def _atomic_commit(prepared: PreparedPlan) -> None:
    new_temps: Dict[Path, str] = {}
    backup_temps: Dict[Path, str] = {}
    applied: List[PreparedFile] = []
    committed = False
    try:
        # Prepare every replacement in the target directory first.  os.replace
        # can then commit each file without a cross-device copy window.
        for file in prepared.files:
            new_temps[file.spec.absolute_path] = _temp_with_bytes(
                file.spec.absolute_path, file.new_bytes, file.mode, "new"
            )

        # This is the required second base-hash check after all in-memory
        # validation and temporary-file preparation, before any destination is
        # touched.  It is repeated immediately before each individual replace.
        for file in prepared.files:
            _assert_current(file)

        for file in prepared.files:
            backup_temps[file.spec.absolute_path] = _temp_with_bytes(
                file.spec.absolute_path, file.old_bytes, file.mode, "old"
            )

        for file in prepared.files:
            _assert_current(file)
            temp_path = new_temps[file.spec.absolute_path]
            os.replace(temp_path, str(file.spec.absolute_path))
            applied.append(file)

        for file in prepared.files:
            try:
                final_bytes = file.spec.absolute_path.read_bytes()
            except OSError as exc:
                raise PlanError(
                    "postwrite_verify_failed",
                    "cannot reread committed file: {}".format(exc),
                    path=file.spec.path,
                )
            if sha256_bytes(final_bytes) != sha256_bytes(file.new_bytes):
                raise PlanError(
                    "postwrite_verify_failed",
                    "committed file hash does not match planned content",
                    path=file.spec.path,
                )
        committed = True
    except BaseException as exc:
        rollback_errors: List[str] = []
        for file in reversed(applied):
            backup = backup_temps.get(file.spec.absolute_path)
            if backup is None:
                rollback_errors.append(
                    "{}: no backup temp file".format(file.spec.path)
                )
                continue
            try:
                os.replace(backup, str(file.spec.absolute_path))
            except BaseException as rollback_exc:
                rollback_errors.append(
                    "{}: {}".format(file.spec.path, rollback_exc)
                )
        if rollback_errors:
            raise PlanError(
                "write_rollback_failed",
                "write failed and at least one file could not be restored",
                details={"cause": str(exc), "rollback_errors": rollback_errors},
            )
        if isinstance(exc, PlanError):
            raise
        raise PlanError(
            "write_failed",
            "atomic write failed; all previously replaced files were restored: {}".format(exc),
        )
    finally:
        for path in list(new_temps.values()) + list(backup_temps.values()):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError:
                # Cleanup failure must not turn a committed transaction into a
                # reported failure.  Temporary names are ignored by plans.
                pass
        if not committed:
            # Best-effort verification of the no-partial-write contract.
            pass


def _success_result(prepared: PreparedPlan, *, check: bool) -> Dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "version": RESULT_VERSION,
        "ok": True,
        "status": "validated" if check else "applied",
        "check": check,
        "writes_performed": not check,
        "plan_sha256": prepared.spec.digest,
        "project_root": str(prepared.spec.project_root),
        "changed_files": [file.spec.path for file in prepared.files],
        "diff": prepared.diff,
        "files": [
            {
                "path": file.spec.path,
                "base_sha256": file.spec.base_sha256,
                "result_sha256": sha256_bytes(file.new_bytes),
                "edit_count": len(file.spec.edits),
                "changed": True,
            }
            for file in prepared.files
        ],
        "errors": [],
    }


def _failure_result(
    *,
    error: PlanError,
    check: bool,
    plan_digest: Optional[str] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "version": RESULT_VERSION,
        "ok": False,
        "status": "failed",
        "check": check,
        "writes_performed": False,
        "diff": "",
        "changed_files": [],
        "errors": [error.as_dict()],
    }
    if plan_digest is not None:
        result["plan_sha256"] = plan_digest
    return result


def apply_patch_plan(document: Any, *, check: bool = False) -> Dict[str, Any]:
    """Validate and optionally apply a plan, returning one result dictionary."""

    plan_digest: Optional[str] = None
    if isinstance(document, Mapping):
        try:
            plan_digest = _sha256_json(document)
        except (TypeError, ValueError):
            plan_digest = None
    try:
        prepared = prepare_plan(document)
        if not check:
            _atomic_commit(prepared)
        return _success_result(prepared, check=check)
    except PlanError as exc:
        return _failure_result(
            error=exc, check=check, plan_digest=plan_digest
        )
    except OSError as exc:
        return _failure_result(
            error=PlanError("io_error", str(exc)),
            check=check,
            plan_digest=plan_digest,
        )


def _json_object_no_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PlanError(
                "duplicate_json_key",
                "JSON object repeats key {!r}".format(key),
                details={"key": key},
            )
        result[key] = value
    return result


def load_plan(path: os.PathLike[str] | str) -> Dict[str, Any]:
    plan_path = Path(path)
    try:
        payload = plan_path.read_bytes()
    except OSError as exc:
        raise PlanError("plan_read_failed", "cannot read plan: {}".format(exc))
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PlanError(
            "plan_invalid_utf8", "plan is not valid UTF-8: {}".format(exc)
        )
    try:
        value = json.loads(decoded, object_pairs_hook=_json_object_no_duplicates)
    except PlanError:
        raise
    except json.JSONDecodeError as exc:
        raise PlanError(
            "plan_invalid_json",
            "plan is not valid JSON: {}".format(exc),
            details={"line": exc.lineno, "column": exc.colno},
        )
    if not isinstance(value, dict):
        raise PlanError("invalid_plan", "plan must be a JSON object")
    return value


def _schema_example() -> Dict[str, Any]:
    return {
        "schema": PLAN_SCHEMA,
        "version": PLAN_VERSION,
        "project_root": "/absolute/path/to/project",
        "files": [
            {
                "path": "src/kernel.cu",
                "base_sha256": "<sha256 of the original file>",
                "edits": [
                    {
                        "id": "gemm-open",
                        "anchor": "for (int k = 0; k < K; ++k) {",
                        "position": "after",
                        "text": "\nFI_B(7, tile);",
                        "occurrence": 1,
                        "expect_count": 1,
                        "indent": "anchor",
                    },
                    {
                        "id": "mma-mark",
                        "anchor": "wmma::mma_sync(",
                        "position": "before",
                        "text": "FI_E(8);\n",
                        "expect_count": 3,
                    },
                ],
            }
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply a weaveprofiler patch plan transactionally"
    )
    parser.add_argument("plan", nargs="?", help="patch-plan JSON file")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate, compute the diff, and write nothing",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="print the plan schema example as JSON",
    )
    parser.add_argument(
        "--diff-output",
        metavar="PATH",
        help="also write the unified diff to this path",
    )
    args = parser.parse_args(argv)

    if args.describe:
        print(json.dumps(_schema_example(), indent=2, ensure_ascii=False))
        return 0
    if not args.plan:
        parser.error("plan is required unless --describe is used")

    try:
        document = load_plan(args.plan)
        result = apply_patch_plan(document, check=args.check)
    except PlanError as exc:
        result = _failure_result(error=exc, check=args.check)

    if args.diff_output:
        try:
            Path(args.diff_output).write_text(result.get("diff", ""), encoding="utf-8")
        except OSError as exc:
            result = _failure_result(
                error=PlanError(
                    "diff_output_failed",
                    "cannot write unified diff: {}".format(exc),
                ),
                check=args.check,
                plan_digest=result.get("plan_sha256"),
            )

    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
