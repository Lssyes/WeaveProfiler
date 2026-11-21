#!/usr/bin/env python3
"""Offline cross-file contract checks for weaveprofiler knowledge artifacts.

These checks deliberately join values from several files.  A file may exist and
parse while still disagreeing with the canonical event map, a recipe, an
example, or the instrumentation generator.
"""
from __future__ import annotations

import ast
import json
import re
import sys
import unittest
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
PKG = HERE.parent

EVENT_IDS = {
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
EVENT_CONSTS = {semantic_id: "EV_" + semantic_id.upper() for semantic_id in EVENT_IDS}
CANONICAL_IDS = frozenset(EVENT_IDS)
ALL_SEMANTIC_IDS = CANONICAL_IDS | {"unknown"}

RECIPES = {
    ("h20", "cuda_cpp"): ("nv/h20/cuda_cpp", "knowledge/recipes/nv/h20/cuda_cpp.json"),
    ("h20", "triton"): ("nv/h20/triton", "knowledge/recipes/nv/h20/triton.json"),
    ("h20", "cutedsl"): ("nv/h20/cutedsl", "knowledge/recipes/nv/h20/cutedsl.json"),
    ("b300", "cuda_cpp"): ("nv/b300/cuda_cpp", "knowledge/recipes/nv/b300/cuda_cpp.json"),
    ("b300", "triton"): ("nv/b300/triton", "knowledge/recipes/nv/b300/triton.json"),
    ("b300", "cutedsl"): ("nv/b300/cutedsl", "knowledge/recipes/nv/b300/cutedsl.json"),
    ("mi308x", "hip_cpp"): ("amd/mi308x/hip_cpp", "knowledge/recipes/amd/mi308x/hip_cpp.json"),
    ("mi308x", "triton"): ("amd/mi308x/triton", "knowledge/recipes/amd/mi308x/triton.json"),
}
B300_LANGUAGES = ("cuda_cpp", "triton", "cutedsl")
B300_PROFILES = {
    "sm_100a": "b300_sm_100a.json",
    "sm_103a": "b300_sm_103a.json",
}
LANGUAGE_PATTERN_FILES = (
    "knowledge/languages/cuda_cpp/patterns.json",
    "knowledge/languages/hip_cpp/patterns.json",
    "knowledge/languages/triton/nvidia_patterns.json",
    "knowledge/languages/triton/amd_patterns.json",
)
TEXT_SUFFIXES = {
    ".py", ".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp", ".md", ".json",
    ".jsonl", ".sh", ".txt", ".yaml", ".yml",
}
EV_ASSIGNMENT = re.compile(
    r"\b(EV_[A-Z0-9_]+)\s*(?::\s*[A-Za-z_][A-Za-z0-9_.]*\s*)?="
    r"\s*(?:[A-Za-z_][A-Za-z0-9_.]*\s*\()?\s*(\d+)\b"
)
SEMANTIC_MARKER = re.compile(r"SEMANTIC-ID\s*:\s*([A-Za-z0-9_.-]+)")
HOST_PATH = re.compile(r"(?:/home/|/localssd/|/opt/conda/)")
FULL_AUDIT = re.compile(r"full[ _-]?audit", re.IGNORECASE)
FI_LEVEL_ONE = re.compile(r"(?:FI_LEVEL\s*=\s*1\b|--fi-level\s+1\b)", re.IGNORECASE)
FI_LEVEL_TWO = re.compile(r"(?:FI_LEVEL\s*=\s*2\b|--fi-level\s+2\b)", re.IGNORECASE)
NEGATION = re.compile(
    r"\b(?:not|never|cannot|can't|isn't|is not|must not)\b|不得|不是|不等于|不要|禁止|不能",
    re.IGNORECASE,
)


def _load_json(relative: str, failures: list[str]) -> Any:
    path = PKG / relative
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        failures.append("{}: cannot read JSON: {}".format(relative, exc))
        return None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _semantic_row(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    semantic_id = value.get("semantic_id")
    return semantic_id if isinstance(semantic_id, str) else None


def check_semantic_id_authority(failures: list[str]) -> None:
    """Check the canonical IDs and the fixed event-id/const mapping."""
    relative = "knowledge/semantic_ids.json"
    document = _load_json(relative, failures)
    if not isinstance(document, dict):
        return

    required = _list(document.get("required_ids"))
    if len(required) != len(set(required)) or set(required) != ALL_SEMANTIC_IDS:
        failures.append(
            "{} required_ids must be exactly the 16 event semantics plus unknown; got {!r}".format(
                relative, required
            )
        )

    rows = _list(document.get("semantics"))
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for row in rows:
        semantic_id = _semantic_row(row)
        if semantic_id is None:
            failures.append("{} has a semantic row without a string semantic_id".format(relative))
            continue
        if semantic_id in by_id:
            duplicate_ids.add(semantic_id)
        else:
            by_id[semantic_id] = row
    for semantic_id in sorted(duplicate_ids):
        failures.append("{} duplicates semantic_id {!r}".format(relative, semantic_id))

    if set(by_id) != ALL_SEMANTIC_IDS:
        failures.append(
            "{} semantic set mismatch: missing={} extra={}".format(
                relative,
                sorted(ALL_SEMANTIC_IDS - set(by_id)),
                sorted(set(by_id) - ALL_SEMANTIC_IDS),
            )
        )

    id_owners: dict[int, str] = {}
    const_owners: dict[str, str] = {}
    for semantic_id, expected_event_id in EVENT_IDS.items():
        row = by_id.get(semantic_id)
        if row is None:
            continue
        expected_const = EVENT_CONSTS[semantic_id]
        event_id = row.get("event_id")
        event_id_const = row.get("event_id_const")
        if isinstance(event_id, bool) or not isinstance(event_id, int):
            failures.append("{} {} has non-integer event_id {!r}".format(
                relative, semantic_id, event_id))
        elif event_id != expected_event_id:
            failures.append("{} {} event_id={} expected {}".format(
                relative, semantic_id, event_id, expected_event_id))
        else:
            previous = id_owners.get(event_id)
            if previous is not None:
                failures.append("{} event_id {} is shared by {} and {}".format(
                    relative, event_id, previous, semantic_id))
            else:
                id_owners[event_id] = semantic_id

        if event_id_const != expected_const:
            failures.append("{} {} event_id_const={!r} expected {!r}".format(
                relative, semantic_id, event_id_const, expected_const))
        if isinstance(event_id_const, str):
            previous = const_owners.get(event_id_const)
            if previous is not None and previous != semantic_id:
                failures.append("{} event_id_const {!r} is shared by {} and {}".format(
                    relative, event_id_const, previous, semantic_id))
            else:
                const_owners[event_id_const] = semantic_id

        if row.get("display_name") != semantic_id:
            failures.append("{} {} display_name must equal semantic_id".format(
                relative, semantic_id))

    unknown = by_id.get("unknown")
    if unknown is None:
        failures.append("{} does not define the explicit unknown semantic".format(relative))


def _index_rows(document: Any) -> list[dict[str, Any]]:
    return [row for row in _list(_dict(document).get("recipes")) if isinstance(row, dict)]


def _matches_recipe_path(value: Any, expected: str) -> bool:
    if not isinstance(value, str):
        return False
    return value.replace("\\", "/").lstrip("./") == expected


def check_recipe_contracts(failures: list[str]) -> None:
    """Check all eight recipes, their index rows, and per-semantic parameters."""
    index = _load_json("knowledge/recipes/index.json", failures)
    rows = _index_rows(index)
    route_keys = {
        (row.get("card"), row.get("language"))
        for row in rows
        if isinstance(row.get("card"), str) and isinstance(row.get("language"), str)
    }
    expected_keys = set(RECIPES)
    if route_keys != expected_keys:
        failures.append(
            "recipe index route set mismatch: missing={} extra={}".format(
                sorted(expected_keys - route_keys), sorted(route_keys - expected_keys)
            )
        )

    for (card, language), (recipe_id, relative) in RECIPES.items():
        matching_rows = [
            row for row in rows
            if row.get("card") == card and row.get("language") == language
        ]
        if not matching_rows:
            failures.append("recipe index has no {}/{} route".format(card, language))
        else:
            linked = [
                row for row in matching_rows
                if _matches_recipe_path(row.get("path"), relative)
            ]
            if not linked:
                failures.append("{} route does not link {}".format(
                    "recipe index {}/{}".format(card, language), relative))
            for row in matching_rows:
                if row.get("semantic_count") != len(EVENT_IDS):
                    failures.append("{} index semantic_count={} expected {}".format(
                        relative, row.get("semantic_count"), len(EVENT_IDS)))
                if row.get("required_semantic_ids_present") is not True:
                    failures.append("{} index must set required_semantic_ids_present=true".format(
                        relative))
            for row in matching_rows:
                row_recipe_id = row.get("recipe_id")
                if row_recipe_id is not None and row_recipe_id != recipe_id:
                    failures.append("{} index recipe_id={!r} expected {!r}".format(
                        relative, row_recipe_id, recipe_id))

        document = _load_json(relative, failures)
        if not isinstance(document, dict):
            continue
        if document.get("recipe_id") != recipe_id:
            failures.append("{} recipe_id={!r} expected {!r}".format(
                relative, document.get("recipe_id"), recipe_id))
        if document.get("language") != language:
            failures.append("{} language={!r} expected {!r}".format(
                relative, document.get("language"), language))

        event_contract = _dict(document.get("event_contract"))
        if event_contract.get("event_ids") != EVENT_IDS:
            failures.append("{} event_contract.event_ids differs from the authority".format(
                relative))

        contract_rows = _list(event_contract.get("semantic_ids"))
        contract_by_id: dict[str, dict[str, Any]] = {}
        for row in contract_rows:
            semantic_id = _semantic_row(row)
            if semantic_id in contract_by_id:
                failures.append("{} event_contract duplicates {!r}".format(
                    relative, semantic_id))
            elif semantic_id is not None:
                contract_by_id[semantic_id] = row
        if len(contract_rows) != len(EVENT_IDS) or set(contract_by_id) != CANONICAL_IDS:
            failures.append("{} event_contract must contain exactly 16 canonical semantics".format(
                relative))
        for semantic_id, expected_event_id in EVENT_IDS.items():
            row = contract_by_id.get(semantic_id)
            if row is None:
                continue
            if row.get("display_name") != semantic_id:
                failures.append("{} event_contract {} display_name must equal semantic_id".format(
                    relative, semantic_id))
            if row.get("event_id") != expected_event_id:
                failures.append("{} event_contract {} event_id={} expected {}".format(
                    relative, semantic_id, row.get("event_id"), expected_event_id))

        semantics = _list(document.get("semantics"))
        semantic_by_id: dict[str, dict[str, Any]] = {}
        for row in semantics:
            semantic_id = _semantic_row(row)
            if semantic_id in semantic_by_id:
                failures.append("{} semantics duplicates {!r}".format(relative, semantic_id))
            elif semantic_id is not None:
                semantic_by_id[semantic_id] = row
        if len(semantics) != len(EVENT_IDS) or set(semantic_by_id) != CANONICAL_IDS:
            failures.append(
                "{} semantics must contain exactly 16 canonical entries; missing={} extra={}".format(
                    relative,
                    sorted(CANONICAL_IDS - set(semantic_by_id)),
                    sorted(set(semantic_by_id) - CANONICAL_IDS),
                )
            )
        for semantic_id, expected_event_id in EVENT_IDS.items():
            row = semantic_by_id.get(semantic_id)
            if row is None:
                continue
            if row.get("display_name") != semantic_id:
                failures.append("{} semantics {} display_name must equal semantic_id".format(
                    relative, semantic_id))
            parameterization = row.get("parameterization")
            if not isinstance(parameterization, dict):
                failures.append("{} semantics {} has no parameterization object".format(
                    relative, semantic_id))
                continue
            if parameterization.get("event_id") != expected_event_id:
                failures.append("{} semantics {} parameterization.event_id={} expected {}".format(
                    relative, semantic_id, parameterization.get("event_id"), expected_event_id))
            expected_const = EVENT_CONSTS[semantic_id]
            if parameterization.get("event_id_const") != expected_const:
                failures.append(
                    "{} semantics {} parameterization.event_id_const={!r} expected {!r}".format(
                        relative, semantic_id,
                        parameterization.get("event_id_const"), expected_const,
                    )
                )


def _ev_assignments(source: str) -> list[tuple[str, int]]:
    return [(name, int(value)) for name, value in EV_ASSIGNMENT.findall(source)]


def check_language_pattern_examples(failures: list[str]) -> None:
    """Check EV_* constants in all four executable language examples."""
    const_to_id = {value: key for key, value in EVENT_CONSTS.items()}
    for relative in LANGUAGE_PATTERN_FILES:
        document = _load_json(relative, failures)
        if not isinstance(document, dict):
            continue
        patterns = _list(document.get("patterns"))
        by_id: dict[str, dict[str, Any]] = {}
        for pattern in patterns:
            semantic_id = _semantic_row(pattern)
            if semantic_id in by_id:
                failures.append("{} duplicates pattern {!r}".format(relative, semantic_id))
            elif semantic_id is not None:
                by_id[semantic_id] = pattern
        if len(patterns) != len(EVENT_IDS) or set(by_id) != CANONICAL_IDS:
            failures.append("{} must contain 16 canonical language patterns; got {}".format(
                relative, sorted(by_id)))

        for semantic_id, pattern in by_id.items():
            source = pattern.get("source_example")
            if not isinstance(source, str) or not source.strip():
                failures.append("{} {} has no source_example".format(relative, semantic_id))
                continue
            assignments = _ev_assignments(source)
            for const_name, value in assignments:
                expected_id = const_to_id.get(const_name)
                if expected_id is None:
                    failures.append("{} {} uses unknown event constant {!r}".format(
                        relative, semantic_id, const_name))
                    continue
                expected_value = EVENT_IDS[expected_id]
                if value != expected_value:
                    failures.append(
                        "{} {} example sets {}={} expected {} ({})".format(
                            relative, semantic_id, const_name, value,
                            expected_value, expected_id,
                        )
                    )
            expected_const = EVENT_CONSTS[semantic_id]
            declarations = {name for name, _ in assignments}
            if expected_const not in declarations:
                failures.append("{} {} example does not declare {}".format(
                    relative, semantic_id, expected_const))


def _profile_pointer(record: dict[str, Any]) -> Any:
    for key in (
        "machine_profile", "profile", "profile_path", "card_profile",
        "card_profile_path",
    ):
        if key in record:
            return record[key]
    path = record.get("path")
    if isinstance(path, str) and Path(path).name in B300_PROFILES.values():
        return path
    return None


def _routes_from_value(value: Any) -> list[tuple[str, Any]]:
    routes: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        arch = value.get("arch", value.get("architecture"))
        profile = _profile_pointer(value)
        if isinstance(arch, str) and profile is not None:
            routes.append((arch, profile))
        for arch, profile in value.items():
            if arch in B300_PROFILES:
                if isinstance(profile, str):
                    routes.append((arch, profile))
                elif isinstance(profile, dict):
                    pointer = _profile_pointer(profile)
                    if pointer is not None:
                        routes.append((arch, pointer))
    elif isinstance(value, list):
        for item in value:
            routes.extend(_routes_from_value(item))
    return routes


def _b300_routes(row: dict[str, Any]) -> list[tuple[str, Any]]:
    routes: list[tuple[str, Any]] = []
    for key in (
        "exact_arch_routes", "arch_routes", "machine_profiles", "routes",
        "architectures",
    ):
        if key in row:
            routes.extend(_routes_from_value(row[key]))
    routes.extend(_routes_from_value(row))
    deduped: list[tuple[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for arch, profile in routes:
        marker = (arch, str(profile))
        if marker not in seen:
            seen.add(marker)
            deduped.append((arch, profile))
    return deduped


def check_b300_exact_arch_routes(failures: list[str]) -> None:
    """Require two exact, separately named B300 profile routes per language."""
    index = _load_json("knowledge/recipes/index.json", failures)
    rows = _index_rows(index)
    for language in B300_LANGUAGES:
        language_rows = [
            row for row in rows
            if row.get("card") == "b300" and row.get("language") == language
        ]
        if not language_rows:
            failures.append("B300 index has no {} recipe routes".format(language))
            continue
        found: dict[str, str] = {}
        for row in language_rows:
            for arch, profile in _b300_routes(row):
                if arch not in B300_PROFILES:
                    failures.append("B300 {} has unexpected route arch {!r}".format(
                        language, arch))
                    continue
                profile_text = str(profile).replace("\\", "/")
                expected_name = B300_PROFILES[arch]
                allowed = {expected_name, "knowledge/cards/" + expected_name}
                if profile_text not in allowed:
                    failures.append(
                        "B300 {} {} route points to {!r}; expected {!r} or {!r}".format(
                            language, arch, profile, expected_name,
                            "knowledge/cards/" + expected_name,
                        )
                    )
                    continue
                previous = found.get(arch)
                if previous is not None and previous != profile_text:
                    failures.append("B300 {} {} has conflicting exact routes".format(
                        language, arch))
                else:
                    found[arch] = profile_text
        if set(found) != set(B300_PROFILES):
            failures.append(
                "B300 {} needs separate sm_100a and sm_103a exact-arch routes; got {}".format(
                    language, sorted(found)
                )
            )
        elif len(found) != 2:
            failures.append("B300 {} routes were merged".format(language))

        relative = RECIPES[("b300", language)][1]
        recipe = _load_json(relative, failures)
        if not isinstance(recipe, dict):
            continue
        expected_profiles = [
            "knowledge/cards/" + B300_PROFILES["sm_100a"],
            "knowledge/cards/" + B300_PROFILES["sm_103a"],
        ]
        if recipe.get("machine_profiles") != expected_profiles:
            failures.append("{} must retain both B300 machine profiles in order".format(
                relative))
        target = _dict(recipe.get("target"))
        if target.get("architectures") != ["sm_100a", "sm_103a"]:
            failures.append("{} must retain both exact B300 architectures".format(relative))
        if target.get("arch_selection_required") is not True:
            failures.append("{} must require explicit B300 arch selection".format(relative))


def _example_marker_counts(failures: list[str]) -> dict[str, int]:
    root = PKG / "knowledge" / "examples"
    counts: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for match in SEMANTIC_MARKER.finditer(text):
            marker = match.group(1).rstrip(".,;:)]}`")
            relative = path.relative_to(PKG).as_posix()
            group = relative.split("/")[2] if relative.startswith("knowledge/examples/") else ""
            counts[group] = counts.get(group, 0) + 1
            if marker not in ALL_SEMANTIC_IDS:
                failures.append("{} has noncanonical SEMANTIC-ID {!r}".format(
                    relative, marker))
            if marker.startswith("gpu."):
                failures.append("{} still uses obsolete gpu.* semantic name {!r}".format(
                    relative, marker))
    for group in ("cuda_cpp", "cute_dsl", "triton", "instrumented"):
        if counts.get(group, 0) == 0:
            failures.append(
                "knowledge/examples/{}/ has no SEMANTIC-ID markers; marker audit would be vacuous".format(
                    group)
            )
    return counts


def check_example_semantic_ids(failures: list[str]) -> None:
    """Reject stale gpu.* semantic markers throughout the example set."""
    _example_marker_counts(failures)


def check_b300_cutedsl_evidence(failures: list[str]) -> None:
    """Keep B300 CuTeDSL experimental and its examples architecture neutral."""
    allowed_status = {"unverified", "experimental_unverified"}
    index = _load_json("knowledge/recipes/index.json", failures)
    rows = [
        row for row in _index_rows(index)
        if row.get("card") == "b300" and row.get("language") == "cutedsl"
    ]
    if not rows:
        failures.append("B300 CuTeDSL has no recipe index route")
    for row in rows:
        status = row.get("evidence_status")
        if status not in allowed_status:
            failures.append("B300 CuTeDSL index evidence_status={!r}; expected one of {}".format(
                status, sorted(allowed_status)))

    recipe_relative = RECIPES[("b300", "cutedsl")][1]
    recipe = _load_json(recipe_relative, failures)
    if isinstance(recipe, dict):
        status = recipe.get("evidence_status")
        if status not in allowed_status:
            failures.append("{} evidence_status={!r}; expected one of {}".format(
                recipe_relative, status, sorted(allowed_status)))
        for row in _list(recipe.get("semantics")):
            semantic_id = _semantic_row(row)
            row_status = _dict(row).get("evidence_status")
            if isinstance(row, dict) and row_status not in allowed_status:
                failures.append(
                    "{} semantic {} evidence_status={!r}; B300 CuTeDSL must remain unverified".format(
                        recipe_relative, semantic_id, row_status)
                )

    arch_assignment = re.compile(r"\bARCH\s*=\s*(['\"])sm_90\1")
    pattern_relative = "knowledge/languages/cutedsl/patterns.json"
    pattern_doc = _load_json(pattern_relative, failures)
    if isinstance(pattern_doc, dict):
        for pattern in _list(pattern_doc.get("patterns")):
            semantic_id = _semantic_row(pattern)
            source = _dict(pattern).get("source_example")
            if isinstance(source, str) and arch_assignment.search(source):
                failures.append(
                    "{} {} source_example hardcodes ARCH='sm_90' instead of selecting the exact B300 target".format(
                        pattern_relative, semantic_id)
                )

    for root in (PKG / "knowledge/examples/cute_dsl",
                 PKG / "knowledge/examples/instrumented/cute_dsl"):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if arch_assignment.search(text):
                failures.append("{} hardcodes ARCH='sm_90'".format(
                    path.relative_to(PKG).as_posix()))


def _audit_documents() -> list[tuple[Path, str]]:
    paths = [PKG / "SKILL.md"]
    knowledge = PKG / "knowledge"
    if knowledge.is_dir():
        paths.extend(sorted(knowledge.rglob("*.md")))
    result: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            result.append((path, path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue
    return result


def check_fi_level_docs(failures: list[str]) -> None:
    """Check the FI_LEVEL/FI_SAMPLE audit contract in prose documentation."""
    docs = _audit_documents()

    for path, text in docs:
        relative = path.relative_to(PKG).as_posix()
        for paragraph in re.split(r"\n\s*\n", text):
            if not (FULL_AUDIT.search(paragraph) and FI_LEVEL_ONE.search(paragraph)):
                continue
            for sentence in re.split(r"(?<=[.!?。；;])\s+|\n+", paragraph):
                if FULL_AUDIT.search(sentence) and FI_LEVEL_ONE.search(sentence):
                    if not NEGATION.search(sentence):
                        failures.append(
                            "{} treats FI_LEVEL=1 as a full audit: {!r}".format(
                                relative, sentence.strip()[:240])
                        )

        # A command block can carry the level while the nearby prose names the
        # audit mode.  Entry-only diagnostics are valid; associating them with
        # full audit is not.
        for block in re.finditer(r"```[^\n]*\n(.*?)```", text, re.DOTALL):
            code = block.group(1)
            if not FI_LEVEL_ONE.search(code):
                continue
            context = text[max(0, block.start() - 500):min(len(text), block.end() + 300)]
            if not FULL_AUDIT.search(context):
                continue
            lowered = context.lower()
            if NEGATION.search(context) or "entry-only" in lowered or "entry only" in lowered:
                continue
            failures.append(
                "{} has a FI_LEVEL=1 command block near full-audit guidance".format(relative)
            )

    sample_explained = False
    sample_contexts: list[tuple[Path, str]] = []
    for path, text in docs:
        for match in re.finditer(r"FI_SAMPLE", text):
            context = text[max(0, match.start() - 320):min(len(text), match.end() + 700)]
            sample_contexts.append((path, context))
            lowered = context.lower()
            explains_values = (
                re.search(r"(?<![A-Za-z0-9])0(?![A-Za-z0-9])", context) is not None
                and re.search(r"(?<![A-Za-z0-9])1(?![A-Za-z0-9])", context) is not None
            )
            explains_behavior = (
                any(word in lowered for word in ("every", "all ", "all k", "全部", "每个"))
                and any(word in lowered for word in (
                    "representative", "one ", "one representative", "sample", "代表", "一个", "采样",
                ))
            )
            if explains_values and explains_behavior:
                sample_explained = True
    if not sample_contexts:
        failures.append("documentation never mentions FI_SAMPLE")
    elif not sample_explained:
        failures.append("FI_SAMPLE documentation must explain both 0 and 1 and their sampling behavior")

    mi_text = "\n\n".join(text for _, text in docs if "mi308x" in text.lower())
    if not mi_text:
        failures.append("documentation contains no MI308X capture guidance")
    fast_paragraphs = [
        paragraph for paragraph in re.split(r"\n\s*\n", mi_text)
        if re.search(r"\bfast\b", paragraph, re.IGNORECASE)
        and re.search(r"low[- ]overhead|fast path|fast low", paragraph, re.IGNORECASE)
    ]
    full_paragraphs = [
        paragraph for paragraph in re.split(r"\n\s*\n", mi_text)
        if FULL_AUDIT.search(paragraph)
    ]
    if not any(FI_LEVEL_TWO.search(paragraph) for paragraph in fast_paragraphs):
        failures.append("MI308X fast/low-overhead documentation must require FI_LEVEL=2")
    if not any(FI_LEVEL_TWO.search(paragraph) for paragraph in full_paragraphs):
        failures.append("MI308X full-audit documentation must require FI_LEVEL=2")


def _contract_example_is_explicitly_example_only(path: Path) -> bool:
    if path.name != "contract_lock_example.json":
        return False
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(document, dict) and document.get("example_only") is True


def check_host_specific_paths(failures: list[str]) -> None:
    """Reject internal host paths while allowing the marked contract example."""
    for path in sorted(PKG.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".md"}:
            continue
        relative = path.relative_to(PKG).as_posix()
        if _contract_example_is_explicitly_example_only(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            match = HOST_PATH.search(line)
            if match:
                failures.append(
                    "{}:{} contains host-specific absolute path fragment {!r}".format(
                        relative, line_number, match.group(0)
                    )
                )


def _selection_schema_constant() -> str:
    path = PKG / "2_instrument/generate_edits.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value_node = node.value
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == "SELECTION_SCHEMA"
               for target in targets):
            value = ast.literal_eval(value_node)
            if not isinstance(value, str) or not value:
                raise ValueError("SELECTION_SCHEMA must be a non-empty string")
            return value
    raise ValueError("generate_edits.py does not define SELECTION_SCHEMA")


def check_selection_schema_id(failures: list[str]) -> None:
    """Join the JSON Schema payload id with generate_edits.SELECTION_SCHEMA."""
    try:
        constant = _selection_schema_constant()
    except (OSError, SyntaxError, ValueError) as exc:
        failures.append("generate_edits.py SELECTION_SCHEMA cannot be read: {}".format(exc))
        return

    relative = "knowledge/validation/schemas/scan-selection.schema.json"
    schema = _load_json(relative, failures)
    if not isinstance(schema, dict):
        return
    try:
        schema_id = schema["properties"]["schema"]["const"]
    except (KeyError, TypeError):
        failures.append("{} has no properties.schema.const".format(relative))
        return
    if schema_id != constant:
        failures.append(
            "{} schema id {!r} differs from generate_edits.SELECTION_SCHEMA {!r}".format(
                relative, schema_id, constant
            )
        )


CHECKS = (
    check_semantic_id_authority,
    check_recipe_contracts,
    check_language_pattern_examples,
    check_b300_exact_arch_routes,
    check_example_semantic_ids,
    check_b300_cutedsl_evidence,
    check_fi_level_docs,
    check_host_specific_paths,
    check_selection_schema_id,
)


def collect_contract_failures() -> list[str]:
    """Run every cross-file check and return deterministic failure messages."""
    failures: list[str] = []
    for check in CHECKS:
        try:
            check(failures)
        except Exception as exc:  # A malformed artifact must fail the contract.
            failures.append("{} crashed: {}: {}".format(
                check.__name__, type(exc).__name__, exc))
    return failures


class ContractConsistencyTest(unittest.TestCase):
    def _assert_clean(self, check) -> None:
        failures: list[str] = []
        check(failures)
        self.assertFalse(failures, "\n".join(failures))

    def test_01_semantic_id_authority(self) -> None:
        self._assert_clean(check_semantic_id_authority)

    def test_02_recipe_contracts(self) -> None:
        self._assert_clean(check_recipe_contracts)

    def test_03_language_pattern_examples(self) -> None:
        self._assert_clean(check_language_pattern_examples)

    def test_04_b300_exact_arch_routes(self) -> None:
        self._assert_clean(check_b300_exact_arch_routes)

    def test_05_example_semantic_ids(self) -> None:
        self._assert_clean(check_example_semantic_ids)

    def test_06_b300_cutedsl_evidence(self) -> None:
        self._assert_clean(check_b300_cutedsl_evidence)

    def test_07_fi_level_docs(self) -> None:
        self._assert_clean(check_fi_level_docs)

    def test_08_host_specific_paths(self) -> None:
        self._assert_clean(check_host_specific_paths)

    def test_09_selection_schema_id(self) -> None:
        self._assert_clean(check_selection_schema_id)


def main() -> int:
    failures = collect_contract_failures()
    if failures:
        print("CONTRACT CONSISTENCY FAIL")
        for failure in failures:
            print("  -", failure)
        return 1
    print("CONTRACT CONSISTENCY PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
