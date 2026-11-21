#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast, cached source scan for weaveprofiler.

The scan reads only the selected project files, emits line-level canonical
semantic candidates from knowledge/semantic_ids.json, and never edits the project.
Comments and string/character literals are masked before marker matching while
original source text continues to supply line numbers, columns and excerpts.
Cached results are invalidated by source bytes, target route, scan options and
the route's recipes, card profile, language patterns, recorder and semantic KB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import registry, semantics  # noqa: E402

SCHEMA = "weaveprofiler-scan/v1"
TOOL_ID = "weaveprofiler-fast-scan/v1"
PRUNE_PARTS = {".git", ".hg", ".svn", "__pycache__", ".weaveprofiler", ".mypy_cache", ".pytest_cache"}
AMBIGUOUS_REASON = "ambiguous_semantic_candidates"
_C_LANGUAGES = {"cuda_cpp", "hip_cpp"}
_PYTHON_LANGUAGES = {"triton", "cutedsl"}
_C_RAW_PREFIX_RE = re.compile(r'(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\"\t\r\n]{0,16})\(')


class ScanError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(partial), str(path))


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve(raw: str, base: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _select_files(project: Path, sources: Sequence[str], globs: Sequence[str]) -> List[Path]:
    selected: Dict[Path, None] = {}
    for raw in sources:
        path = _resolve(raw, project)
        if not path.is_file():
            raise ScanError("source file does not exist: {}".format(path))
        selected[path] = None
    for pattern in globs:
        matches: Iterable[Path]
        if Path(pattern).is_absolute():
            matches = [Path(pattern)]
        else:
            matches = project.glob(pattern)
        for path in matches:
            if not path.is_file():
                continue
            path = path.resolve()
            if any(part in PRUNE_PARTS for part in path.parts):
                continue
            selected[path] = None
    files = sorted(selected)
    if not files:
        raise ScanError("select at least one source with --source or --glob")
    outside = [str(path) for path in files if not _inside(path, project)]
    if outside:
        raise ScanError("source files must stay under --project: {}".format(", ".join(outside[:3])))
    return files


def _marker_regex(marker: str) -> re.Pattern:
    text = str(marker).strip()
    if not text:
        raise ScanError("empty source marker")
    escaped = re.escape(text)
    if re.match(r"^[A-Za-z0-9_]", text):
        escaped = r"(?<![A-Za-z0-9_])" + escaped
    if re.search(r"[A-Za-z0-9_]$", text):
        escaped = escaped + r"(?![A-Za-z0-9_])"
    return re.compile(escaped, re.IGNORECASE)


def _compile_markers(kb: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for semantic in kb["semantics"]:
        sid = str(semantic["semantic_id"])
        if sid == "unknown":
            continue
        compiled = [{"marker": marker, "regex": _marker_regex(marker)}
                    for marker in semantic.get("source_markers") or []]
        rows.append({"semantic_id": sid, "kind": semantic.get("kind"),
                     "attribution": semantic.get("attribution"), "markers": compiled})
    return rows


def _file_records(project: Path, files: Sequence[Path], max_bytes: int) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    records: List[Dict[str, Any]] = []
    lines_by_rel: Dict[str, List[str]] = {}
    for path in files:
        size = path.stat().st_size
        if size > max_bytes:
            raise ScanError("source file exceeds --max-file-bytes: {} ({} > {})".format(path, size, max_bytes))
        data = path.read_bytes()
        rel = str(path.relative_to(project)).replace(os.sep, "/")
        records.append({"path": rel, "bytes": len(data), "sha256": _sha256_bytes(data)})
        # Split only on LF so masking can join the lines back into the exact
        # same text and preserve terminal blank lines and offsets.
        lines_by_rel[rel] = data.decode("utf-8", errors="replace").split("\n")
    return records, lines_by_rel


def _mask_span(chars: List[str], start: int, end: int) -> None:
    """Blank a source span without changing newlines or character offsets."""
    for index in range(start, min(end, len(chars))):
        if chars[index] not in "\r\n":
            chars[index] = " "


def _mask_c_non_code(source: str) -> str:
    chars = list(source)
    length = len(source)
    index = 0
    state = "code"
    quote = ""

    while index < length:
        char = source[index]
        if state == "line_comment":
            if char in "\r\n":
                escaped = index > 0 and source[index - 1] == "\\"
                state = "line_comment" if escaped else "code"
            else:
                _mask_span(chars, index, index + 1)
            index += 1
            continue
        if state == "block_comment":
            if source.startswith("*/", index):
                _mask_span(chars, index, index + 2)
                state = "code"
                index += 2
            else:
                _mask_span(chars, index, index + 1)
                index += 1
            continue
        if state == "string":
            if char == "\\":
                _mask_span(chars, index, index + 2)
                index += 2
                continue
            if char == quote:
                _mask_span(chars, index, index + 1)
                state = "code"
                index += 1
                continue
            if char in "\r\n":
                state = "code"
                index += 1
                continue
            _mask_span(chars, index, index + 1)
            index += 1
            continue

        if source.startswith("//", index):
            _mask_span(chars, index, index + 2)
            state = "line_comment"
            index += 2
            continue
        if source.startswith("/*", index):
            _mask_span(chars, index, index + 2)
            state = "block_comment"
            index += 2
            continue

        raw_match = _C_RAW_PREFIX_RE.match(source, index)
        if raw_match and (index == 0 or not (source[index - 1].isalnum() or source[index - 1] == "_")):
            delimiter = raw_match.group("delimiter")
            closing = ")" + delimiter + '"'
            closing_at = source.find(closing, raw_match.end())
            raw_end = length if closing_at < 0 else closing_at + len(closing)
            _mask_span(chars, index, raw_end)
            index = raw_end
            continue

        # C++ permits apostrophes between hexadecimal digits as numeric
        # separators; those are code, not character literals.
        is_digit_separator = (
            char == "'" and index > 0 and index + 1 < length
            and source[index - 1] in "0123456789abcdefABCDEF"
            and source[index + 1] in "0123456789abcdefABCDEF")
        if char in "'\"" and not is_digit_separator:
            quote = char
            _mask_span(chars, index, index + 1)
            state = "string"
        index += 1

    return "".join(chars)


def _mask_python_non_code(source: str) -> str:
    chars = list(source)
    length = len(source)
    index = 0
    state = "code"
    quote = ""
    triple = False

    while index < length:
        char = source[index]
        if state == "comment":
            if char in "\r\n":
                state = "code"
            else:
                _mask_span(chars, index, index + 1)
            index += 1
            continue
        if state == "string":
            if char == "\\":
                _mask_span(chars, index, index + 2)
                index += 2
                continue
            if triple and source.startswith(quote * 3, index):
                _mask_span(chars, index, index + 3)
                state = "code"
                index += 3
                continue
            if not triple and char == quote:
                _mask_span(chars, index, index + 1)
                state = "code"
                index += 1
                continue
            if not triple and char in "\r\n":
                state = "code"
                index += 1
                continue
            _mask_span(chars, index, index + 1)
            index += 1
            continue

        if char == "#":
            _mask_span(chars, index, index + 1)
            state = "comment"
            index += 1
            continue
        if char in "'\"":
            quote = char
            triple = source.startswith(char * 3, index)
            width = 3 if triple else 1
            _mask_span(chars, index, index + width)
            state = "string"
            index += width
            continue
        index += 1

    return "".join(chars)


def neutralize_non_code(source: str, language: str) -> str:
    """Return a same-length source copy with comments and literals blanked.

    Only matching uses the returned copy. Callers retain the original source for
    line numbers, columns and excerpts.
    """
    normalized = registry.normalize_language(language)
    raw = str(language or "").strip().lower()
    if normalized in _C_LANGUAGES or raw in {"c", "c++", "cpp"}:
        return _mask_c_non_code(source)
    if normalized in _PYTHON_LANGUAGES or raw in {"python", "python3", "py"}:
        return _mask_python_non_code(source)
    raise ScanError("unsupported language for lexical masking: {}".format(language))


# A stable, descriptive alias for callers that describe this as masking.
mask_non_code = neutralize_non_code


def _unknown_records(ambiguities: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unknown = []
    for ambiguity in ambiguities:
        unknown.append({
            "semantic_id": "unknown",
            "display_name": "unknown",
            "kind": "unknown",
            "attribution": "other",
            "path": ambiguity["path"],
            "line": ambiguity["line"],
            "column": ambiguity["column"],
            "candidates": list(ambiguity["candidates"]),
            "reason": AMBIGUOUS_REASON,
            "source_excerpt": ambiguity.get("source_excerpt", ""),
        })
    return unknown


def _scan_lines(lines_by_rel: Dict[str, List[str]], marker_rows: Sequence[Dict[str, Any]],
                language: str = "cuda_cpp") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    by_location: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for rel, lines in lines_by_rel.items():
        original_text = "\n".join(lines)
        masked_text = neutralize_non_code(original_text, language)
        masked_lines = masked_text.split("\n")
        if len(masked_lines) != len(lines):
            raise ScanError("lexical masking changed line count for {}".format(rel))
        for lineno, original_line in enumerate(lines, 1):
            code_line = masked_lines[lineno - 1]
            for row in marker_rows:
                hits = []
                for marker in row["markers"]:
                    match = marker["regex"].search(code_line)
                    if match:
                        hits.append((match.start(), marker["marker"]))
                if not hits:
                    continue
                hits.sort(key=lambda item: (item[0], item[1]))
                first_column = hits[0][0] + 1
                item = {
                    "semantic_id": row["semantic_id"],
                    "display_name": row["semantic_id"],
                    "kind": row["kind"],
                    "attribution": row["attribution"],
                    "path": rel,
                    "line": lineno,
                    "column": first_column,
                    "matched_markers": [marker for _start, marker in hits],
                    "source_excerpt": original_line.strip()[:320],
                }
                candidates.append(item)
                by_location.setdefault((rel, lineno), []).append(item)
    ambiguities = []
    for (rel, lineno), items in sorted(by_location.items()):
        ids = sorted({item["semantic_id"] for item in items})
        if len(ids) > 1:
            ambiguities.append({
                "path": rel,
                "line": lineno,
                "column": min(item["column"] for item in items),
                "candidates": ids,
                "reason": AMBIGUOUS_REASON,
                "source_excerpt": items[0]["source_excerpt"],
            })
    candidates.sort(key=lambda item: (item["path"], item["line"], item["semantic_id"], item["column"]))
    return candidates, ambiguities


def _repo_relative(path: Path, repo_root: Path, label: str) -> str:
    root = repo_root.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.resolve().relative_to(root)
    except ValueError as exc:
        raise ScanError("{} must stay under the repository root: {}".format(label, path)) from exc
    return relative.as_posix()


def _evidence_entry(role: str, path: Path, repo_root: Path) -> Dict[str, Any]:
    root = repo_root.resolve()
    candidate = path if path.is_absolute() else root / path
    relative = _repo_relative(candidate, root, role)
    resolved = (root / relative).resolve()
    if not resolved.is_file():
        raise ScanError("{} evidence file does not exist: {}".format(role, relative))
    return {"role": role, "path": relative, "sha256": _sha256_file(resolved)}


def _evidence_json(entry: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    path = repo_root.resolve() / entry["path"]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScanError("cannot read {} evidence {}: {}".format(
            entry["role"], entry["path"], exc)) from exc
    if not isinstance(value, dict):
        raise ScanError("{} evidence must be a JSON object: {}".format(entry["role"], entry["path"]))
    return value


def build_route_evidence(route: Dict[str, Any], *, repo_root: Path = ROOT,
                         semantic_kb_path: Optional[Path] = None) -> Dict[str, Any]:
    """Hash the exact knowledge files that selected and describe this route."""
    root = repo_root.resolve()
    target = dict(route.get("target") or {})
    card = str(target.get("card") or "")
    arch = str(target.get("arch") or "")
    language = str(route.get("language") or "")
    platform = str(target.get("platform") or "")
    if card == "b300" and not arch:
        raise ScanError("B300 requires an exact --arch: sm_100a or sm_103a")
    if not card or not arch or not language or not platform:
        raise ScanError("route evidence requires exact card, architecture, platform and language")

    semantic_entry = _evidence_entry(
        "semantic_kb", semantic_kb_path or semantics.DEFAULT_KB, root)
    recipe_index_entry = _evidence_entry(
        "recipe_index", Path("knowledge/recipes/index.json"), root)
    recipe_index = _evidence_json(recipe_index_entry, root)
    recipes = recipe_index.get("recipes")
    if not isinstance(recipes, list):
        raise ScanError("recipe index has no recipes list")
    matches = []
    for recipe in recipes:
        if not isinstance(recipe, dict):
            continue
        architectures = recipe.get("architectures") or []
        if (str(recipe.get("card") or "") == card
                and str(recipe.get("language") or "") == language
                and arch in architectures):
            matches.append(recipe)
    if len(matches) != 1:
        raise ScanError(
            "recipe index must contain exactly one recipe for card={} arch={} language={}; found {}"
            .format(card, arch, language, len(matches)))
    route_recipe_entry = _evidence_entry(
        "route_recipe", Path(str(matches[0].get("path") or "")), root)
    route_recipe = _evidence_json(route_recipe_entry, root)
    if str(route_recipe.get("language") or "") != language:
        raise ScanError("route recipe language does not match route: {}".format(route_recipe_entry["path"]))
    recipe_target = dict(route_recipe.get("target") or {})
    if recipe_target.get("card") and str(recipe_target["card"]) != card:
        raise ScanError("route recipe card does not match route: {}".format(route_recipe_entry["path"]))

    card_profiles = []
    for raw_profile in route_recipe.get("machine_profiles") or []:
        entry = _evidence_entry("card_profile", Path(str(raw_profile)), root)
        profile = _evidence_json(entry, root)
        if str(profile.get("card") or "") == card and str(profile.get("arch") or "") == arch:
            entry["arch"] = arch
            card_profiles.append(entry)
    if len(card_profiles) != 1:
        raise ScanError(
            "route recipe must select exactly one exact card profile for card={} arch={}; found {}"
            .format(card, arch, len(card_profiles)))

    pattern_entry = _evidence_entry(
        "language_patterns",
        Path(registry.language_pattern_path(language, platform)), root)
    recorder_value = route.get("recorder")
    if not recorder_value:
        raise ScanError("route has no recorder")
    recorder_entry = _evidence_entry("recorder", Path(str(recorder_value)), root)

    semantic_authority = recipe_index.get("semantic_authority")
    if semantic_authority:
        authority_path = _repo_relative(Path(str(semantic_authority)), root, "semantic_authority")
        if authority_path != semantic_entry["path"]:
            raise ScanError(
                "recipe index semantic authority {} does not match semantic KB {}"
                .format(authority_path, semantic_entry["path"]))

    return {
        "semantic_kb": semantic_entry,
        "recipe_index": recipe_index_entry,
        "route_recipe": route_recipe_entry,
        "card_profiles": card_profiles,
        "language_patterns": pattern_entry,
        "recorder": recorder_entry,
    }


def _cache_key(route: Dict[str, Any], file_records: Sequence[Dict[str, Any]],
               kb_sha: str, options: Dict[str, Any],
               knowledge_evidence: Optional[Dict[str, Any]] = None) -> str:
    payload = {"tool": TOOL_ID, "tool_sha256": _sha256_file(Path(__file__).resolve()),
               "route_resolver_sha256": _sha256_file(ROOT / "registry.py"),
               "semantic_loader_sha256": _sha256_file(ROOT / "semantics.py"),
               "target": route.get("target"), "language": route.get("language"),
               "files": list(file_records), "semantic_kb_sha256": kb_sha,
               "knowledge_evidence": knowledge_evidence, "options": options}
    return _sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def scan(project: Path, *, card: str, arch: Optional[str], platform: Optional[str], language: str,
         sources: Sequence[str], globs: Sequence[str], cache_dir: Path,
         max_file_bytes: int = 8 * 1024 * 1024, refresh: bool = False) -> Dict[str, Any]:
    started = time.monotonic()
    project = project.resolve()
    if not project.is_dir():
        raise ScanError("project is not a directory: {}".format(project))
    try:
        route = registry.resolve(card=card, arch=arch, platform=platform, language=language)
    except Exception as exc:
        raise ScanError("target routing failed: {}".format(exc))
    target = route.get("target") or {}
    if target.get("card") == "b300" and not target.get("arch"):
        raise ScanError("B300 requires an exact --arch: sm_100a or sm_103a")
    if not target.get("arch"):
        raise ScanError("route requires an exact architecture")
    files = _select_files(project, sources, globs)
    file_records, lines_by_rel = _file_records(project, files, max_file_bytes)
    try:
        # Scan only consumes source markers. Event-ID completeness is enforced
        # by the instrumentation/delivery paths, not by this source scan.
        kb = semantics.load_kb(validate=False)
    except Exception as exc:
        raise ScanError("cannot load semantic KB: {}".format(exc)) from exc
    if not isinstance(kb.get("semantics"), list) or not kb["semantics"]:
        raise ScanError("semantic KB has no semantics list")
    knowledge_evidence = build_route_evidence(
        route, semantic_kb_path=Path(str(kb["_path"])))
    route = dict(route)
    route["knowledge_evidence"] = knowledge_evidence
    options = {"max_file_bytes": max_file_bytes}
    key = _cache_key(route, file_records, kb["_sha256"], options, knowledge_evidence)
    cache_path = cache_dir.resolve() / (key + ".json")
    if cache_path.is_file() and not refresh:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("schema") == SCHEMA and cached.get("cache_key") == key:
                cached["cache"] = {"hit": True, "key": key, "path": str(cache_path)}
                cached["scan_wall_seconds"] = round(time.monotonic() - started, 3)
                return cached
        except (OSError, ValueError):
            pass
    markers = _compile_markers(kb)
    candidates, ambiguities = _scan_lines(lines_by_rel, markers, str(route.get("language") or language))
    unknown = _unknown_records(ambiguities)
    status = "BLOCKED" if ambiguities else "PASS"
    counts: Dict[str, int] = {}
    for item in candidates:
        counts[item["semantic_id"]] = counts.get(item["semantic_id"], 0) + 1
    operators = sorted({item["semantic_id"] for item in candidates if item["kind"] == "operator"})
    phases = sorted({item["semantic_id"] for item in candidates
                     if item["kind"] in ("phase", "shared")})
    result = {
        "schema": SCHEMA,
        "status": status,
        "fast_path": True,
        "delivery_evidence": False,
        "notice": "source candidate scan only; event boundaries still require review",
        "scanned_at": _utc_now(),
        "project_root": str(project),
        "target": route.get("target"),
        "language": route.get("language"),
        "route": route,
        "knowledge_evidence": knowledge_evidence,
        "semantic_kb": {
            "path": knowledge_evidence["semantic_kb"]["path"],
            "sha256": knowledge_evidence["semantic_kb"]["sha256"],
        },
        "files": file_records,
        "operators": operators,
        "phases": phases,
        "counts_by_semantic": counts,
        "candidates": candidates,
        "ambiguities": ambiguities,
        "unknown": unknown,
        "cache_key": key,
        "cache": {"hit": False, "key": key, "path": str(cache_path)},
        "scan_wall_seconds": round(time.monotonic() - started, 3),
    }
    _atomic_json(cache_path, result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="fast cached weaveprofiler source scan")
    parser.add_argument("--project", required=True)
    parser.add_argument("--card", required=True)
    parser.add_argument("--arch")
    parser.add_argument("--platform")
    parser.add_argument("--language", required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--glob", action="append", default=[])
    parser.add_argument("--cache-dir", default=".weaveprofiler/cache/scan")
    parser.add_argument("--max-file-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    project = _resolve(args.project, Path.cwd())
    cache_dir = _resolve(args.cache_dir, project)
    try:
        result = scan(project, card=args.card, arch=args.arch, platform=args.platform,
                      language=args.language, sources=args.source, globs=args.glob,
                      cache_dir=cache_dir, max_file_bytes=args.max_file_bytes,
                      refresh=args.refresh)
    except ScanError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    if args.out:
        _atomic_json(_resolve(args.out, project), result)
    print(json.dumps({"status": result["status"], "language": result["language"],
                      "files": len(result["files"]), "candidates": len(result["candidates"]),
                      "operators": result["operators"], "phases": result["phases"],
                      "ambiguities": len(result["ambiguities"]), "cache_hit": result["cache"]["hit"],
                      "unknown": len(result["unknown"]),
                      "blocking_reasons": sorted({item.get("reason") for item in result["unknown"]}),
                      "wall_seconds": result["scan_wall_seconds"]}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
