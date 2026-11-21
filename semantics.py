#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical semantic IDs for weaveprofiler.

`knowledge/semantic_ids.json` is the authority for event labels.  Source names and
natural-language descriptions are matching hints only; the rendered label is
always the canonical ``semantic_id`` (for example ``mma``).  When a mapping is
missing or ambiguous, resolution returns ``unknown`` and the strict delivery
path fails closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

SCHEMA = "weaveprofiler-semantic-ids/v1"
CONTRACT_SCHEMA = "weaveprofiler-semantic-contract/v1"
PKG = Path(__file__).resolve().parent
DEFAULT_KB = PKG / "knowledge" / "semantic_ids.json"
VALID_ATTRIBUTIONS = {"load", "wait", "compute", "store", "memory", "control", "other"}
VALID_WAIT_SUBTYPES = {"wait_tail", "wait_dependency", "wait_load", "wait_sync", "wait_other"}
VALID_KINDS = {"operator", "phase", "shared", "unknown"}
SEMANTIC_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
EVENT_ID_CONST_RE = re.compile(r"^EV_[A-Z][A-Z0-9_]*$")
MIN_EVENT_ID = 1
MAX_EVENT_ID = 4095
EXPECTED_IDS = {
    "gemm", "attention", "softmax", "reduction", "moe", "convolution", "megakernel",
    "load", "wait", "mma", "compute", "epilogue", "store", "barrier", "dispatch",
    "communication", "unknown",
}


class SemanticKBError(ValueError):
    """Raised when the semantic knowledge base itself is malformed."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_kb(path: Optional[str | Path] = None, *, validate: bool = True) -> Dict[str, Any]:
    p = Path(path or DEFAULT_KB).resolve()
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticKBError(f"cannot load semantic KB {p}: {exc}") from exc
    if validate:
        errors = validate_kb(doc)
        if errors:
            raise SemanticKBError("invalid semantic KB %s: %s" % (p, "; ".join(errors[:12])))
    doc["_path"] = str(p)
    doc["_sha256"] = _sha256(p)
    return doc


def _normalize_label(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower().strip()
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"[._:/\\\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_alias(text: str, alias: str) -> bool:
    if not alias:
        return False
    if not text or not alias:
        return False
    if re.search(r"[\u3400-\u9fff]", alias):
        return alias in text
    pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def validate_kb(doc: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if doc.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    rows = doc.get("semantics")
    if not isinstance(rows, list) or not rows:
        return errors + ["semantics must be a non-empty list"]
    ids: Dict[str, int] = {}
    aliases: Dict[str, str] = {}
    event_ids: Dict[int, str] = {}
    event_constants: Dict[str, str] = {}
    for index, row in enumerate(rows):
        where = f"semantics[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{where} must be an object")
            continue
        sid = row.get("semantic_id")
        if not isinstance(sid, str) or not SEMANTIC_ID_RE.match(sid):
            errors.append(f"{where}.semantic_id must match {SEMANTIC_ID_RE.pattern}")
            continue
        if sid in ids:
            errors.append(f"duplicate semantic_id {sid!r}")
        ids[sid] = index
        event_id = row.get("event_id")
        event_id_const = row.get("event_id_const")
        if sid == "unknown":
            if event_id is not None or event_id_const is not None:
                errors.append(f"{where}.unknown must not claim a device event id or constant")
        else:
            if (isinstance(event_id, bool) or not isinstance(event_id, int)
                    or not MIN_EVENT_ID <= event_id <= MAX_EVENT_ID):
                errors.append(
                    f"{where}.event_id must be an integer in "
                    f"{MIN_EVENT_ID}..{MAX_EVENT_ID}"
                )
            else:
                owner = event_ids.get(event_id)
                if owner is not None and owner != sid:
                    errors.append(
                        f"event_id {event_id} is claimed by both {owner!r} and {sid!r}"
                    )
                event_ids[event_id] = sid
            if (not isinstance(event_id_const, str)
                    or not EVENT_ID_CONST_RE.match(event_id_const)):
                errors.append(
                    f"{where}.event_id_const must match {EVENT_ID_CONST_RE.pattern}"
                )
            else:
                owner = event_constants.get(event_id_const)
                if owner is not None and owner != sid:
                    errors.append(
                        f"event_id_const {event_id_const!r} is claimed by both "
                        f"{owner!r} and {sid!r}"
                    )
                event_constants[event_id_const] = sid
        if row.get("display_name") != sid:
            errors.append(f"{where}.display_name must equal semantic_id {sid!r}")
        kind = row.get("kind")
        if kind not in VALID_KINDS:
            errors.append(f"{where}.kind must be one of {sorted(VALID_KINDS)}")
        attribution = row.get("attribution")
        if attribution not in VALID_ATTRIBUTIONS:
            errors.append(f"{where}.attribution must be one of {sorted(VALID_ATTRIBUTIONS)}")
        if attribution == "wait":
            subtype = row.get("wait_subtype", "wait_other")
            if subtype not in VALID_WAIT_SUBTYPES:
                errors.append(f"{where}.wait_subtype must be one of {sorted(VALID_WAIT_SUBTYPES)}")
        if not isinstance(row.get("description"), str) or not row["description"].strip():
            errors.append(f"{where}.description must be non-empty")
        row_aliases = row.get("aliases")
        if not isinstance(row_aliases, list):
            errors.append(f"{where}.aliases must be a list")
            row_aliases = []
        for alias in row_aliases:
            norm = _normalize_label(alias)
            if not norm:
                errors.append(f"{where} has an empty alias")
                continue
            owner = aliases.get(norm)
            if owner is not None and owner != sid:
                errors.append(f"alias {alias!r} is claimed by both {owner!r} and {sid!r}")
            aliases[norm] = sid
        markers = row.get("source_markers")
        if not isinstance(markers, list) or any(not str(x).strip() for x in markers):
            errors.append(f"{where}.source_markers must be a list of non-empty strings")
    missing = EXPECTED_IDS - set(ids)
    if missing:
        errors.append(f"missing required semantic ids: {sorted(missing)}")
    unknown = next((row for row in rows if row.get("semantic_id") == "unknown"), None)
    if unknown is not None and unknown.get("aliases"):
        errors.append("unknown semantic_id must not have aliases")
    required = doc.get("required_ids")
    if not isinstance(required, list) or set(required) != EXPECTED_IDS:
        errors.append(f"required_ids must be exactly {sorted(EXPECTED_IDS)}")
    return errors


def semantic_index(kb: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(row["semantic_id"]): dict(row) for row in kb["semantics"]}


def event_id_index(kb: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Return the canonical device-event-id to semantic mapping."""
    result: Dict[int, Dict[str, Any]] = {}
    for row in kb["semantics"]:
        event_id = row.get("event_id")
        if isinstance(event_id, int) and not isinstance(event_id, bool):
            result[int(event_id)] = dict(row)
    return result


def semantic_for_event_id(event_id: Any, kb: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the canonical semantic row registered for one device event id."""
    if isinstance(event_id, bool):
        return None
    try:
        key = int(event_id)
    except (TypeError, ValueError):
        return None
    return event_id_index(kb).get(key)


def get_semantic(semantic_id: str, kb: Mapping[str, Any]) -> Dict[str, Any]:
    row = semantic_index(kb).get(str(semantic_id))
    if row is None:
        raise SemanticKBError(f"unknown semantic_id {semantic_id!r}")
    return row


def _unknown_record(label: str, event_id: Any, reason: str,
                    candidates: Sequence[str] = ()) -> Dict[str, Any]:
    return {
        "event_id": str(event_id),
        "semantic_id": "unknown",
        "display_name": "unknown",
        "kind": "unknown",
        "attribution": "other",
        "semantic_source": "unknown",
        "input_label": str(label or ""),
        "reason": reason,
        "candidates": list(candidates),
    }


def match_label(label: Any, kb: Mapping[str, Any]) -> Tuple[Optional[str], List[str]]:
    """Return one unique alias match or the full candidate set.

    Ambiguity is intentionally not resolved by longest-prefix heuristics.  Event
    IDs must then carry an explicit semantic_id instead of letting display names
    drift between releases.
    """
    text = _normalize_label(label)
    if not text:
        return None, []
    candidates: List[str] = []
    for row in kb["semantics"]:
        sid = str(row["semantic_id"])
        if sid == "unknown":
            continue
        if _normalize_label(sid) == text:
            candidates.append(sid)
            continue
        if any(_contains_alias(text, _normalize_label(alias)) for alias in row.get("aliases") or []):
            candidates.append(sid)
    unique = sorted(set(candidates))
    return (unique[0], unique) if len(unique) == 1 else (None, unique)


def resolve_event(event_id: Any, label: Any, names: Mapping[str, Any],
                  kb: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve one event ID using explicit mapping first, unique alias second."""
    canonical = semantic_for_event_id(event_id, kb)
    explicit_map = names.get("semantics") or names.get("event_semantics") or {}
    if not isinstance(explicit_map, Mapping):
        return _unknown_record(label, event_id, "names.json semantics must be an object")
    explicit = explicit_map.get(str(event_id))
    if explicit is not None:
        detail = None
        if isinstance(explicit, str):
            sid = explicit
        elif isinstance(explicit, Mapping):
            sid = explicit.get("semantic_id")
            detail = explicit.get("detail")
            declared = explicit.get("display_name")
            if declared is not None and declared != sid:
                return _unknown_record(
                    label, event_id,
                    "display_name overrides are forbidden; display_name must equal semantic_id",
                    [str(declared)])
        else:
            return _unknown_record(label, event_id, "semantic mapping must be a string or object")
        try:
            row = get_semantic(str(sid), kb)
        except SemanticKBError:
            return _unknown_record(label, event_id, "semantic_id is not declared in knowledge/semantic_ids.json",
                                   [str(sid)])
        if canonical is not None and row["semantic_id"] != canonical["semantic_id"]:
            return _unknown_record(
                label,
                event_id,
                "event_id is a reserved canonical id for a different semantic",
                [canonical["semantic_id"], str(sid)],
            )
        record = dict(row)
        record.update({
            "event_id": str(event_id),
            "semantic_source": "explicit",
            "input_label": str(label or ""),
            "reason": None,
            "candidates": [],
        })
        if detail is not None:
            record["detail"] = str(detail)
        return record
    if canonical is not None:
        record = dict(canonical)
        record.update({
            "event_id": str(event_id),
            "semantic_source": "canonical_event_id",
            "input_label": str(label or ""),
            "reason": None,
            "candidates": [],
        })
        return record
    sid, candidates = match_label(label, kb)
    if sid:
        row = get_semantic(sid, kb)
        record = dict(row)
        record.update({
            "event_id": str(event_id),
            "semantic_source": "alias",
            "input_label": str(label or ""),
            "reason": None,
            "candidates": candidates,
        })
        return record
    reason = "label matches multiple semantic ids" if candidates else "no declared semantic_id or unique alias match"
    return _unknown_record(label, event_id, reason, candidates)


def resolve_names(names: Mapping[str, Any], kb: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    kb = dict(kb or load_kb())
    events = names.get("events") if isinstance(names, Mapping) else None
    if not isinstance(events, Mapping):
        raise SemanticKBError("names.json events must be an object")
    resolved: Dict[str, Dict[str, Any]] = {}
    for event_id, label in sorted(events.items(), key=lambda item: str(item[0])):
        resolved[str(event_id)] = resolve_event(str(event_id), label, names, kb)
    explicit_map = names.get("semantics") or names.get("event_semantics") or {}
    extra = sorted(set(map(str, explicit_map)) - set(resolved)) if isinstance(explicit_map, Mapping) else []
    unknown = [eid for eid, row in resolved.items() if row["semantic_id"] == "unknown"]
    ambiguous = [eid for eid, row in resolved.items()
                 if row["semantic_id"] == "unknown" and row.get("candidates")]
    errors: List[str] = []
    for eid in unknown:
        row = resolved[eid]
        suffix = (" candidates=" + ",".join(row.get("candidates") or [])) if row.get("candidates") else ""
        errors.append(f"event {eid} ({row.get('input_label')!r}): {row.get('reason')}{suffix}")
    if extra:
        errors.append("semantics contains event ids absent from events: " + ", ".join(extra))
    return {
        "schema": CONTRACT_SCHEMA,
        "kb_schema": kb.get("schema"),
        "kb_path": kb.get("_path"),
        "kb_sha256": kb.get("_sha256"),
        "policy": dict(kb.get("policy") or {}),
        "events": resolved,
        "unknown_event_ids": unknown,
        "ambiguous_event_ids": ambiguous,
        "errors": errors,
        "ok": not errors,
    }


def source_hints(source_text: str, kb: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Return non-authoritative source markers.  A model must review real code."""
    kb = dict(kb or load_kb())
    text = str(source_text or "")
    lowered = text.lower()
    hits: Dict[str, List[str]] = {}
    for row in kb["semantics"]:
        sid = str(row["semantic_id"])
        matched = [m for m in row.get("source_markers") or [] if str(m).lower() in lowered]
        if matched:
            hits[sid] = matched
    return {"authoritative": False, "semantic_ids": sorted(hits), "markers": hits,
            "instruction": "Treat markers only as review hints; assign semantic_id after reading the source."}


def validate_names_file(path: str | Path, kb_path: Optional[str | Path] = None,
                        *, require: bool = False) -> Dict[str, Any]:
    names = json.loads(Path(path).read_text(encoding="utf-8"))
    kb = load_kb(kb_path)
    contract = resolve_names(names, kb)
    contract["strict_required"] = bool(require)
    if require and not contract["ok"]:
        raise SemanticKBError("canonical semantic mapping failed: " + "; ".join(contract["errors"][:12]))
    return contract


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate and resolve canonical weaveprofiler event semantics")
    ap.add_argument("--kb", default=str(DEFAULT_KB))
    ap.add_argument("--names")
    ap.add_argument("--text")
    ap.add_argument("--semantic-id")
    ap.add_argument("--require", action="store_true", help="exit 2 if any event resolves to unknown")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    try:
        kb = load_kb(args.kb)
        if args.list:
            print(json.dumps({"schema": kb["schema"], "path": kb["_path"], "sha256": kb["_sha256"],
                              "semantics": kb["semantics"]}, ensure_ascii=False, indent=2))
            return 0
        if args.semantic_id:
            print(json.dumps(get_semantic(args.semantic_id, kb), ensure_ascii=False, indent=2))
            return 0
        if args.text is not None:
            sid, candidates = match_label(args.text, kb)
            print(json.dumps({"text": args.text, "semantic_id": sid or "unknown",
                              "candidates": candidates,
                              "source_hints": source_hints(args.text, kb)},
                             ensure_ascii=False, indent=2))
            return 0 if sid else 2
        if args.selftest:
            names = {"events": {"1": "装载 A tile", "3": "wgmma 主循环", "99": "unknown work"},
                     "semantics": {"99": "unknown"}}
            contract = resolve_names(names, kb)
            assert contract["events"]["1"]["semantic_id"] == "load"
            assert contract["events"]["3"]["semantic_id"] == "mma"
            assert contract["events"]["99"]["semantic_id"] == "unknown"
            assert contract["unknown_event_ids"] == ["99"]
            print("WEAVEPROFILER SEMANTICS SELFTEST PASS")
            return 0
        if args.names:
            contract = validate_names_file(args.names, args.kb, require=args.require)
            print(json.dumps(contract, ensure_ascii=False, indent=2))
            return 0 if contract["ok"] or not args.require else 2
    except (OSError, json.JSONDecodeError, SemanticKBError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    ap.error("provide --names, --text, --semantic-id, --list, or --selftest")


if __name__ == "__main__":
    raise SystemExit(main())
