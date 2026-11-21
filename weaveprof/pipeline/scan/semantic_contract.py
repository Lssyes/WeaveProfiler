#!/usr/bin/env python3
"""Source-locked semantic task contracts and raw-to-trace bijection audits."""

import glob
import gzip
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


SCHEMA = "fast-instrument-semantic-analysis/v1"
MAX_DOMAIN_INSTANCES = 2_000_000
_HASH_FIELDS = frozenset(("conclusion_sha256", "semantic_conclusion_sha256"))
_HASH_EXCLUDED_FIELDS = _HASH_FIELDS | frozenset((
    "abi", "abi_path", "abi_sha256", "bijection_evidence", "evidence",
))
_REQUIRED_TASK_FIELDS = (
    "semantic_unit", "instance_key", "instance_domain", "expected_cardinality",
    "collaborating_warps", "recording_representative", "tile_shape",
    "source_boundary",
)
_FORBIDDEN_AGGREGATIONS = frozenset((
    "minmax", "min_max", "envelope", "per_observer_lane_minmax",
    "per_rank_cta", "global", "aggregate",
))


class SemanticContractError(ValueError):
    pass


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_analysis(document):
    if not isinstance(document, dict):
        return None
    for container in (document, document.get("events"), document.get("target")):
        if isinstance(container, dict) and isinstance(container.get("semantic_analysis"), dict):
            return container["semantic_analysis"]
    return None


def event_descriptors(document):
    if not isinstance(document, dict):
        return []
    candidates = (
        document.get("tasks"),
        (document.get("events") or {}).get("event_definitions")
            if isinstance(document.get("events"), dict) else None,
        (document.get("buffer_abi") or {}).get("tasks")
            if isinstance(document.get("buffer_abi"), dict) else None,
    )
    for value in candidates:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def descriptor_id(descriptor):
    for field in ("id", "event_id", "task_id"):
        value = descriptor.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    return None


def task_descriptors(document):
    return [item for item in event_descriptors(document) if item.get("layer") == "task"]


def semantic_conclusion_payload(document):
    analysis = semantic_analysis(document)
    if not isinstance(analysis, dict):
        raise SemanticContractError("semantic_analysis is missing")
    locked_analysis = {
        key: value for key, value in analysis.items()
        if key not in _HASH_EXCLUDED_FIELDS
    }
    tasks = []
    for descriptor in task_descriptors(document):
        tasks.append({
            "id": descriptor_id(descriptor),
            "name": descriptor.get("name"),
            "display_name": descriptor.get("display_name"),
            "layer": descriptor.get("layer"),
            "task_contract": descriptor.get("task_contract"),
        })
    tasks.sort(key=lambda item: (
        item["id"] is None, item["id"] if item["id"] is not None else 0,
        str(item.get("name", "")),
    ))
    return {"semantic_analysis": locked_analysis, "tasks": tasks}


def semantic_conclusion_sha256(document):
    return _sha256_bytes(_canonical_json(semantic_conclusion_payload(document)))


def _declared_conclusion_sha256(analysis):
    for field in ("conclusion_sha256", "semantic_conclusion_sha256"):
        value = analysis.get(field) if isinstance(analysis, dict) else None
        if isinstance(value, str) and len(value) == 64:
            return value.lower()
    return None


def _version_root(document, base_dir):
    layout = document.get("project_layout") if isinstance(document, dict) else None
    value = layout.get("version_root") if isinstance(layout, dict) else None
    if isinstance(value, str) and value:
        return Path(value).expanduser().resolve()
    return Path(base_dir).expanduser().resolve()


def _resolve_path(value, base):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    return path.resolve()


def _source_closure_map(analysis, version_root):
    entries = analysis.get("source_closure") if isinstance(analysis, dict) else None
    failures = []
    result = {}
    if not isinstance(entries, list) or not entries:
        return result, ["semantic_analysis.source_closure must be a non-empty list"]
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            failures.append("source_closure[{}] must be an object".format(index))
            continue
        raw_path = entry.get("path", entry.get("absolute_path"))
        expected = entry.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            failures.append("source_closure[{}].path is missing".format(index))
            continue
        if not isinstance(expected, str) or len(expected) != 64:
            failures.append("source_closure[{}].sha256 must be 64 hex characters".format(index))
            continue
        path = _resolve_path(raw_path, version_root)
        key = str(path)
        if key in result:
            failures.append("source_closure repeats {}".format(path))
            continue
        result[key] = {"path": path, "declared_path": raw_path,
                       "sha256": expected.lower()}
        if not path.is_file():
            failures.append("source closure file does not exist: {}".format(path))
            continue
        try:
            observed = sha256_file(path)
        except OSError as error:
            failures.append("cannot hash source closure file {}: {}".format(path, error))
            continue
        if observed != expected.lower():
            failures.append("source closure SHA256 drift: {} expected={} observed={}".format(
                path, expected.lower(), observed))
    return result, failures


def _warp_expressions(value):
    if isinstance(value, list):
        values = value
    elif isinstance(value, dict):
        values = value.get("warps", value.get("values"))
    else:
        return None
    return values if isinstance(values, list) and values else None


def _participant_warps(value, environment=None, context=None):
    values = _warp_expressions(value)
    if values is None:
        return None
    result = []
    for item in values:
        try:
            resolved = item if isinstance(item, int) and not isinstance(item, bool) else _scalar(
                item, environment or {}, context or {})
        except SemanticContractError:
            return None
        if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved < 0:
            return None
        result.append(int(resolved))
    return result if len(set(result)) == len(result) else None


def _representative_warp(value, environment=None, context=None):
    if not isinstance(value, dict):
        return None
    for field in ("warp", "warp_id", "observer_warp"):
        warp = value.get(field)
        if warp is None:
            continue
        try:
            resolved = warp if isinstance(warp, int) and not isinstance(warp, bool) else _scalar(
                warp, environment or {}, context or {})
        except SemanticContractError:
            return None
        if isinstance(resolved, int) and not isinstance(resolved, bool) and resolved >= 0:
            return int(resolved)
    return None


def _resolve_template(value, environment, context):
    if isinstance(value, dict):
        if set(value) in ({"ref"}, {"const"}) or "op" in value:
            return _scalar(value, environment, context)
        return {key: _resolve_template(item, environment, context)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_template(item, environment, context) for item in value]
    return value


def _resolved_recorder(task_contract, key, context):
    environment = dict(key)
    return _resolve_template(task_contract.get("recording_representative"),
                             environment, context)


def _anchor_path(anchor, version_root):
    value = anchor.get("path") if isinstance(anchor, dict) else None
    if not isinstance(value, str) or not value:
        return None
    return _resolve_path(value, version_root)


def _validate_source_anchor(label, anchor, closure, version_root):
    failures = []
    if not isinstance(anchor, dict):
        return ["{} must be an object".format(label)]
    path = _anchor_path(anchor, version_root)
    line = anchor.get("line")
    marker = anchor.get("anchor", anchor.get("symbol"))
    if path is None:
        failures.append("{}.path is missing".format(label))
    elif str(path) not in closure:
        failures.append("{}.path is not in semantic source_closure: {}".format(label, path))
    if not isinstance(line, int) or isinstance(line, bool) or line < 1:
        failures.append("{}.line must be a positive integer".format(label))
    if not isinstance(marker, str) or not marker.strip():
        failures.append("{}.anchor or symbol is missing".format(label))
    if failures or path is None:
        return failures
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        return ["cannot read {}: {}".format(path, error)]
    if line > len(lines):
        return ["{}.line={} exceeds {} lines in {}".format(label, line, len(lines), path)]
    lo = max(0, line - 3)
    hi = min(len(lines), line + 2)
    if marker not in "\n".join(lines[lo:hi]):
        failures.append("{} marker {!r} is absent near {}:{}".format(
            label, marker, path, line))
    return failures


def validate_semantic_contract(document, base_dir=".", verify_sources=True):
    failures = []
    analysis = semantic_analysis(document)
    if not isinstance(analysis, dict):
        return ["semantic_analysis is required; task boundaries must be model-reviewed and source-locked"]
    if analysis.get("schema") != SCHEMA:
        failures.append("semantic_analysis.schema must be {}".format(SCHEMA))
    if analysis.get("model_reviewed") is not True:
        failures.append("semantic_analysis.model_reviewed must be true")
    version_root = _version_root(document, base_dir)
    closure = {}
    if verify_sources:
        closure, closure_failures = _source_closure_map(analysis, version_root)
        failures.extend(closure_failures)
    else:
        entries = analysis.get("source_closure")
        if not isinstance(entries, list) or not entries:
            failures.append("semantic_analysis.source_closure must be a non-empty list")
    declared_hash = _declared_conclusion_sha256(analysis)
    if declared_hash is None:
        failures.append("semantic_analysis.conclusion_sha256 is missing")
    else:
        try:
            observed_hash = semantic_conclusion_sha256(document)
        except SemanticContractError as error:
            failures.append(str(error))
        else:
            if declared_hash != observed_hash:
                failures.append("semantic conclusion SHA256 mismatch: expected={} observed={}".format(
                    declared_hash, observed_hash))
    tasks = task_descriptors(document)
    if not tasks:
        failures.append("no layer=task descriptor carries a semantic task contract")
        return failures
    seen_ids = set()
    for index, descriptor in enumerate(tasks):
        task_id = descriptor_id(descriptor)
        label = "task[{}]".format(task_id if task_id is not None else index)
        if task_id is None:
            failures.append("{}.id/event_id must be an integer".format(label))
        elif task_id in seen_ids:
            failures.append("duplicate task descriptor id {}".format(task_id))
        else:
            seen_ids.add(task_id)
        contract = descriptor.get("task_contract")
        if not isinstance(contract, dict):
            failures.append("{}.task_contract is required".format(label))
            continue
        missing = [field for field in _REQUIRED_TASK_FIELDS if contract.get(field) is None]
        if missing:
            failures.append("{}.task_contract misses {}".format(label, missing))
            continue
        if contract.get("classification", "task") != "task":
            failures.append("{}.task_contract.classification must be task".format(label))
        semantic_unit = contract.get("semantic_unit")
        if not isinstance(semantic_unit, str) or not semantic_unit.strip():
            failures.append("{}.semantic_unit must be a non-empty string".format(label))
        key_fields = contract.get("instance_key")
        if (not isinstance(key_fields, list) or not key_fields
                or not all(isinstance(field, str) and field for field in key_fields)
                or len(set(key_fields)) != len(key_fields)):
            failures.append("{}.instance_key must be a non-empty unique string list".format(label))
        cardinality = contract.get("expected_cardinality")
        if not isinstance(cardinality, int) or isinstance(cardinality, bool) or cardinality < 1:
            failures.append("{}.expected_cardinality must be a positive integer".format(label))
        warp_expressions = _warp_expressions(contract.get("collaborating_warps"))
        if warp_expressions is None:
            failures.append("{}.collaborating_warps must be a non-empty warp list".format(label))
        representative = contract.get("recording_representative")
        if not isinstance(representative, dict) or not any(
                field in representative for field in ("warp", "warp_id", "observer_warp")):
            failures.append("{}.recording_representative must declare one physical warp".format(label))
        if not isinstance(representative, dict) or not any(
                representative.get(field) is not None
                for field in ("thread", "thread_idx", "lane", "predicate")):
            failures.append("{}.recording_representative must identify one canonical thread".format(label))
        tile_shape = contract.get("tile_shape")
        if not isinstance(tile_shape, dict) or not tile_shape:
            failures.append("{}.tile_shape must be a non-empty object".format(label))
        boundary = contract.get("source_boundary")
        if not isinstance(boundary, dict):
            failures.append("{}.source_boundary must be an object".format(label))
        else:
            for field in ("begin", "end", "completion_sync"):
                if verify_sources:
                    failures.extend(_validate_source_anchor(
                        "{}.source_boundary.{}".format(label, field),
                        boundary.get(field), closure, version_root))
                elif not isinstance(boundary.get(field), dict):
                    failures.append("{}.source_boundary.{} must be an object".format(label, field))
        try:
            keys = enumerate_task_instances(contract, document)
        except SemanticContractError as error:
            failures.append("{} instance_domain invalid: {}".format(label, error))
        else:
            if isinstance(cardinality, int) and len(keys) != cardinality:
                failures.append("{} instance_domain enumerates {} keys, expected_cardinality={}".format(
                    label, len(keys), cardinality))
            invalid_membership = []
            for key in keys:
                environment = dict(key)
                warps = _participant_warps(contract.get("collaborating_warps"),
                                           environment, document)
                recorder_warp = _representative_warp(representative, environment, document)
                if warps is None or recorder_warp is None or recorder_warp not in warps:
                    invalid_membership.append((dict(key), warps, recorder_warp))
                    if len(invalid_membership) >= 8:
                        break
            if invalid_membership:
                failures.append("{} recorder warp must belong to collaborating_warps for every instance; samples={}".format(
                    label, invalid_membership))
    return failures


def _lookup_ref(name, environment, context):
    if name in environment:
        return environment[name]
    node = context
    for part in str(name).split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise SemanticContractError("unknown ref {!r}".format(name))
    return node


def _scalar(expression, environment, context):
    if isinstance(expression, (str, int, float, bool)) or expression is None:
        return expression
    if not isinstance(expression, dict):
        raise SemanticContractError("expression must be scalar or object, got {!r}".format(expression))
    if set(expression) == {"ref"}:
        return _lookup_ref(expression["ref"], environment, context)
    if set(expression) == {"const"}:
        return expression["const"]
    op = expression.get("op")
    if not isinstance(op, str):
        raise SemanticContractError("expression object misses op/ref/const: {!r}".format(expression))
    if op in ("and", "or"):
        args = expression.get("args")
        if not isinstance(args, list) or not args:
            raise SemanticContractError("{} requires non-empty args".format(op))
        values = [bool(_scalar(arg, environment, context)) for arg in args]
        return all(values) if op == "and" else any(values)
    if op == "not":
        return not bool(_scalar(expression.get("value"), environment, context))
    left = _scalar(expression.get("left"), environment, context)
    right = _scalar(expression.get("right"), environment, context)
    if op == "eq":
        return left == right
    if op == "ne":
        return left != right
    if op == "lt":
        return left < right
    if op == "le":
        return left <= right
    if op == "gt":
        return left > right
    if op == "ge":
        return left >= right
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    if op == "mod":
        return left % right
    if op == "floor_div":
        return left // right
    raise SemanticContractError("unsupported instance_domain op {!r}".format(op))


def _axis_values(expression, environment, context):
    if isinstance(expression, list):
        values = expression
    elif isinstance(expression, dict) and expression.get("op") == "range":
        start = _scalar(expression.get("start", 0), environment, context)
        stop = _scalar(expression.get("stop"), environment, context)
        step = _scalar(expression.get("step", 1), environment, context)
        if any(not isinstance(value, int) or isinstance(value, bool)
               for value in (start, stop, step)) or step == 0:
            raise SemanticContractError("range start/stop/step must be integers and step != 0")
        values = list(range(start, stop, step))
    elif isinstance(expression, dict) and "values" in expression:
        values = expression["values"]
    else:
        value = _scalar(expression, environment, context)
        values = value if isinstance(value, list) else [value]
    if not isinstance(values, list) or not values:
        raise SemanticContractError("axis values must be a non-empty list")
    canonical = [_canonical_json(value) for value in values]
    if len(set(canonical)) != len(canonical):
        raise SemanticContractError("axis values contain duplicates")
    return values


def _domain_axes(domain):
    axes = domain.get("axes") if isinstance(domain, dict) else None
    if isinstance(axes, dict):
        return [(name, expression) for name, expression in axes.items()]
    if isinstance(axes, list):
        result = []
        for item in axes:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise SemanticContractError("each axis must have a string name")
            result.append((item["name"], item.get("values", item.get("domain"))))
        return result
    raise SemanticContractError("instance_domain.axes must be an object or list")


def enumerate_task_instances(task_contract, context):
    domain = task_contract.get("instance_domain")
    key_fields = task_contract.get("instance_key")
    if not isinstance(domain, dict):
        raise SemanticContractError("instance_domain must be an object")
    if not isinstance(key_fields, list) or not key_fields:
        raise SemanticContractError("instance_key must be a non-empty list")
    axes = _domain_axes(domain)
    axis_names = [name for name, _ in axes]
    if len(axis_names) != len(set(axis_names)):
        raise SemanticContractError("instance_domain axis names repeat")
    missing = [field for field in key_fields if field not in axis_names]
    if missing:
        raise SemanticContractError("instance_key fields are not domain axes: {}".format(missing))
    constraints = domain.get("constraints", [])
    if isinstance(constraints, dict):
        constraints = [constraints]
    if not isinstance(constraints, list):
        raise SemanticContractError("instance_domain.constraints must be a list")
    environments = [{}]
    for name, expression in axes:
        expanded = []
        for environment in environments:
            for value in _axis_values(expression, environment, context):
                item = dict(environment)
                item[name] = value
                expanded.append(item)
                if len(expanded) > MAX_DOMAIN_INSTANCES:
                    raise SemanticContractError("instance domain exceeds {} candidates".format(
                        MAX_DOMAIN_INSTANCES))
        environments = expanded
    keys = []
    for environment in environments:
        if all(bool(_scalar(expr, environment, context)) for expr in constraints):
            key = tuple((field, environment[field]) for field in key_fields)
            keys.append(key)
    canonical = [_canonical_json(key) for key in keys]
    if len(canonical) != len(set(canonical)):
        raise SemanticContractError("instance_domain emits duplicate instance keys")
    return keys


def _semantic_abi_declaration(contract):
    analysis = semantic_analysis(contract)
    candidates = []
    if isinstance(analysis, dict):
        if isinstance(analysis.get("abi"), dict):
            candidates.append(analysis["abi"])
        if analysis.get("abi_path") is not None:
            candidates.append({"path": analysis.get("abi_path"),
                               "sha256": analysis.get("abi_sha256")})
    code = contract.get("code") if isinstance(contract, dict) else None
    if isinstance(code, dict):
        for key in ("abi", "buffer_abi"):
            if isinstance(code.get(key), dict):
                candidates.append(code[key])
    for candidate in candidates:
        path = candidate.get("path", candidate.get("absolute_path"))
        if isinstance(path, str) and path:
            return candidate
    return None


def validate_abi_lock(contract, base_dir="."):
    failures = []
    declaration = _semantic_abi_declaration(contract)
    if declaration is None:
        return ["semantic_analysis must bind the source ABI with path and sha256"], None
    version_root = _version_root(contract, base_dir)
    abi_path = _resolve_path(declaration.get("path", declaration.get("absolute_path")), version_root)
    expected = declaration.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        failures.append("semantic ABI sha256 must be 64 hex characters")
    if not abi_path.is_file():
        failures.append("semantic ABI does not exist: {}".format(abi_path))
        return failures, None
    observed = sha256_file(abi_path)
    if isinstance(expected, str) and observed != expected.lower():
        failures.append("semantic ABI SHA256 mismatch: {} expected={} observed={}".format(
            abi_path, expected, observed))
    try:
        abi = json.loads(abi_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        failures.append("cannot load semantic ABI {}: {}".format(abi_path, error))
        return failures, None
    failures.extend(validate_semantic_contract(abi, base_dir=version_root, verify_sources=True))
    contract_hash = _declared_conclusion_sha256(semantic_analysis(contract))
    abi_hash = _declared_conclusion_sha256(semantic_analysis(abi))
    if contract_hash != abi_hash:
        failures.append("contract/ABI semantic conclusion hash mismatch: {}/{}".format(
            contract_hash, abi_hash))
    try:
        contract_payload_hash = semantic_conclusion_sha256(contract)
        abi_payload_hash = semantic_conclusion_sha256(abi)
    except SemanticContractError as error:
        failures.append(str(error))
    else:
        if contract_payload_hash != abi_payload_hash:
            failures.append("contract did not copy ABI semantic_analysis/task_contract exactly")
    return failures, abi


def _load_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _evidence_patterns(contract, kind):
    analysis = semantic_analysis(contract) or {}
    evidence = analysis.get("bijection_evidence")
    if not isinstance(evidence, dict):
        evidence = contract.get("semantic_evidence") if isinstance(contract, dict) else None
    value = evidence.get(kind) if isinstance(evidence, dict) else None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return []


def _expand_evidence_paths(contract, run_dir, kind):
    patterns = _evidence_patterns(contract, kind)
    if not patterns:
        defaults = {
            "raw_pairs": ("sw/raw_pairs_rank*.json", "sw/raw_pair_manifest*.json"),
            "decoded_events": ("sw/events_rank*.json",),
        }
        patterns = list(defaults.get(kind, ()))
    paths = []
    for pattern in patterns:
        path = Path(pattern).expanduser()
        if path.is_absolute():
            matches = glob.glob(str(path))
        else:
            matches = glob.glob(str(Path(run_dir) / path))
        paths.extend(Path(item).resolve() for item in matches if Path(item).is_file())
    return sorted(set(paths))


def _document_items(document, kind):
    if isinstance(document, list):
        return document
    if not isinstance(document, dict):
        return []
    names = ("raw_pairs", "pairs", "events") if kind == "raw_pairs" else (
        "decoded_events", "events", "pairs")
    for name in names:
        value = document.get(name)
        if isinstance(value, list):
            return value
    return []


def _field(item, name):
    if not isinstance(item, dict):
        return None
    if name in item:
        return item.get(name)
    args = item.get("args")
    return args.get(name) if isinstance(args, dict) else None


def _event_id(item):
    for name in ("event_id", "task_id", "id"):
        value = _field(item, name)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    return None


def _instance_key(item, fields):
    value = _field(item, "instance_key")
    if not isinstance(value, dict):
        return None
    if set(value) != set(fields):
        return None
    return tuple((field, value[field]) for field in fields)


_UID_FIELDS = ("rank", "lane", "start_offset", "end_offset", "event_id", "work_id")


def _uid(item):
    value = _field(item, "source_event_uid")
    if isinstance(value, (list, tuple)) and len(value) == len(_UID_FIELDS):
        fields = dict(zip(_UID_FIELDS, value))
    elif isinstance(value, dict) and set(value) == set(_UID_FIELDS):
        fields = {field: value[field] for field in _UID_FIELDS}
    else:
        return None, None, "must be a six-field list or exact object {}".format(
            list(_UID_FIELDS))
    if (not isinstance(fields["start_offset"], int)
            or isinstance(fields["start_offset"], bool)
            or not isinstance(fields["end_offset"], int)
            or isinstance(fields["end_offset"], bool)
            or fields["end_offset"] <= fields["start_offset"]):
        return None, None, "contains invalid raw record offsets"
    if (not isinstance(fields["event_id"], int)
            or isinstance(fields["event_id"], bool)):
        return None, None, "contains a non-integer event_id"
    return _canonical_json([fields[field] for field in _UID_FIELDS]).decode("utf-8"), fields, None


def _item_provenance_failures(item, label, contract, key, context, uid_fields):
    failures = []
    if _field(item, "is_derived") is not False:
        failures.append("{} must carry is_derived=false".format(label))
    aggregation = str(_field(item, "aggregation") or "").strip().lower()
    if aggregation in _FORBIDDEN_AGGREGATIONS or _field(item, "task_envelope") is True:
        failures.append("{} is an aggregation/envelope rather than one device pair".format(label))
    contributors = _field(item, "source_contributor_count")
    if contributors is None:
        contributors = _field(item, "source_event_count")
    if contributors is not None and contributors != 1:
        failures.append("{} has {} source contributors; exactly one pair is required".format(
            label, contributors))
    # Dynamic work claiming (v13 comm pool scatter): the recording warp is
    # decided by an HBM atomic race at run time, so no static reverse
    # inference can name it.  The contract then carries
    # recorder_attribution.static_warp_reverse_inference=false plus the legal
    # ACTIVE recorder set, and the audit asserts membership in that set --
    # every record still proves it came from a warp that legally performs the
    # task -- instead of an impossible exact equality.  Statically attributed
    # tasks keep the exact check unchanged.
    attribution = contract.get("recorder_attribution")
    dynamic_claim = (
        isinstance(attribution, dict)
        and attribution.get("static_warp_reverse_inference") is False
        and isinstance(attribution.get("active_physical_warps"), list)
    )
    if dynamic_claim:
        actual_recorder = _field(item, "canonical_recorder")
        actual_warp = _representative_warp(actual_recorder) if isinstance(
            actual_recorder, dict) else None
        if actual_warp not in attribution["active_physical_warps"]:
            failures.append(
                "{} canonical_recorder warp {!r} is outside the declared "
                "active recorder set {}".format(
                    label, actual_warp, attribution["active_physical_warps"]))
    else:
        try:
            expected_recorder = _resolved_recorder(contract, key, context)
        except SemanticContractError as error:
            failures.append("{} cannot resolve canonical recorder: {}".format(label, error))
        else:
            actual_recorder = _field(item, "canonical_recorder")
            if actual_recorder != expected_recorder:
                failures.append("{} canonical_recorder does not equal resolved task_contract".format(label))
    event_id = _event_id(item)
    if uid_fields["event_id"] != event_id:
        failures.append("{} source_event_uid event_id does not match the event".format(label))
    key_values = dict(key)
    if "rank" in key_values and uid_fields["rank"] != key_values["rank"]:
        failures.append("{} source_event_uid rank does not match instance_key".format(label))
    lane = _field(item, "lane")
    if lane is None:
        lane = _field(item, "observer_lane")
    if lane is not None and uid_fields["lane"] != lane:
        failures.append("{} source_event_uid lane does not match event provenance".format(label))
    work_id = _field(item, "work_id")
    if work_id is not None and uid_fields["work_id"] != work_id:
        failures.append("{} source_event_uid work_id does not match event provenance".format(label))
    return failures


def _collect_layer(paths, kind, contracts, context):
    failures = []
    records = defaultdict(list)
    for path in paths:
        try:
            document = _load_json(path)
        except (OSError, ValueError) as error:
            failures.append("cannot read {} evidence {}: {}".format(kind, path, error))
            continue
        for index, item in enumerate(_document_items(document, kind)):
            event_id = _event_id(item)
            if event_id not in contracts:
                continue
            task_contract = contracts[event_id]
            key = _instance_key(item, task_contract["instance_key"])
            label = "{}:{}[{}]".format(kind, path.name, index)
            if key is None:
                failures.append("{} misses the exact instance_key fields {}".format(
                    label, task_contract["instance_key"]))
                continue
            uid, uid_fields, uid_error = _uid(item)
            if uid_error:
                failures.append("{} source_event_uid {}".format(label, uid_error))
                continue
            if kind == "raw_pairs":
                start = _field(item, "start_offset")
                end = _field(item, "end_offset")
                if (not isinstance(start, int) or isinstance(start, bool)
                        or not isinstance(end, int) or isinstance(end, bool) or end <= start):
                    failures.append("{} is not one complete raw start/end pair".format(label))
                elif (uid_fields["start_offset"] != start
                      or uid_fields["end_offset"] != end):
                    failures.append("{} source_event_uid offsets do not match raw pair".format(label))
            failures.extend(_item_provenance_failures(
                item, label, task_contract, key, context, uid_fields))
            records[(event_id, key)].append(uid)
    return records, failures


def _collect_trace(trace, contracts, context):
    failures = []
    records = defaultdict(list)
    events = trace.get("traceEvents", trace if isinstance(trace, list) else [])
    for index, item in enumerate(events):
        if not isinstance(item, dict) or item.get("ph") != "X":
            continue
        args = item.get("args") or {}
        if args.get("layer") != "task":
            continue
        event_id = _event_id(item)
        label = "trace:X[{}]".format(index)
        if event_id not in contracts:
            failures.append("{} uses task id {} absent from semantic task contracts".format(
                label, event_id))
            continue
        task_contract = contracts[event_id]
        key = _instance_key(item, task_contract["instance_key"])
        if key is None:
            failures.append("{} misses the exact instance_key fields {}".format(
                label, task_contract["instance_key"]))
            continue
        uid, uid_fields, uid_error = _uid(item)
        if uid_error:
            failures.append("{} source_event_uid {}".format(label, uid_error))
            continue
        failures.extend(_item_provenance_failures(
            item, label, task_contract, key, context, uid_fields))
        records[(event_id, key)].append(uid)
    return records, failures


def _format_key(key):
    event_id, values = key
    return "event_id={} {}".format(event_id, dict(values))


def _compare_layer(name, records, expected):
    failures = []
    actual = set(records)
    missing = sorted(expected - actual, key=str)
    extra = sorted(actual - expected, key=str)
    duplicate = [(key, len(values)) for key, values in records.items() if len(values) != 1]
    if missing:
        failures.append("{} misses {} semantic instances; samples={}".format(
            name, len(missing), [_format_key(key) for key in missing[:8]]))
    if extra:
        failures.append("{} contains {} undeclared/cross-scope instances; samples={}".format(
            name, len(extra), [_format_key(key) for key in extra[:8]]))
    if duplicate:
        failures.append("{} has {} duplicate semantic instances (warp replicas or repeated spans); samples={}".format(
            name, len(duplicate), [(_format_key(key), count) for key, count in duplicate[:8]]))
    return failures


def _unit_key_fields(task_contract):
    fields = list(task_contract.get("instance_key") or [])
    return [f for f in fields if f not in ("warp", "recorder_class")]


def _task_is_dynamic_claim(task_contract):
    attribution = task_contract.get("recorder_attribution")
    return (isinstance(attribution, dict)
            and attribution.get("static_warp_reverse_inference") is False
            and "warp" in (task_contract.get("instance_key") or []))


def _coverage_compare_layer(name, records, declared_full, task_contract):
    """Coverage audit for dynamic work-claimed tasks (v13 comm pool math).

    The declared domain is the class-level UNION of legal recorder warpgroups;
    each unit is realized by exactly ONE claiming CTA, so the realized key set
    is a strict subset of the declared union by design.  What remains provable
    and is asserted here, per unit (key without the warp axis):
      * nothing emitted outside the declared union (no extra/cross-scope);
      * no key emitted twice (no duplicates);
      * every declared unit is covered by at least one emitted key;
      * the warps realizing one unit all belong to ONE legal 4-warp
        warpgroup of the declared union (the claiming CTA's class).
    """
    failures = []
    unit_fields = _unit_key_fields(task_contract)
    declared_units = {}
    for key in declared_full:
        values = dict(key)
        unit = tuple((f, values[f]) for f in unit_fields)
        declared_units.setdefault(unit, set()).add(values["warp"])
    actual_by_unit = {}
    for key in records:
        values = dict(key)
        unit = tuple((f, values[f]) for f in unit_fields)
        actual_by_unit.setdefault(unit, set()).add(values["warp"])
    actual_full = set(records)
    extra = sorted(actual_full - declared_full, key=str)
    duplicate = [(key, len(v)) for key, v in records.items() if len(v) != 1]
    if extra:
        failures.append("{} contains {} undeclared/cross-scope instances; samples={}".format(
            name, len(extra), [dict(key) for key in extra[:8]]))
    if duplicate:
        failures.append("{} has {} duplicate semantic instances; samples={}".format(
            name, len(duplicate), [(dict(k), c) for k, c in duplicate[:8]]))
    uncovered = sorted(set(declared_units) - set(actual_by_unit), key=str)
    if uncovered:
        failures.append("{} leaves {} claimed-work units unrecorded; samples={}".format(
            name, len(uncovered), [dict(u) for u in uncovered[:8]]))
    for unit, warps in actual_by_unit.items():
        legal = declared_units.get(unit)
        if not legal:
            continue
        groups = [set(sorted(legal)[i:i + 4]) for i in range(0, len(legal), 4)]
        if not any(warps <= g for g in groups):
            failures.append(
                "{} unit {} realized by warps {} spanning multiple warpgroups "
                "(legal: {})".format(name, dict(unit), sorted(warps),
                                     [sorted(g) for g in groups]))
            break
    return failures


def audit_semantic_trace_tasks(document, trace):
    contracts = {}
    for descriptor in task_descriptors(document):
        event_id = descriptor_id(descriptor)
        task_contract = descriptor.get("task_contract")
        if event_id is not None and isinstance(task_contract, dict):
            contracts[event_id] = task_contract
    if not contracts:
        return ["no semantic task contracts are available"]
    records, failures = _collect_trace(trace, contracts, document)
    duplicates = [(key, len(values)) for key, values in records.items()
                  if len(values) != 1]
    if duplicates:
        failures.append("Perfetto task spans have {} duplicate semantic instances; samples={}".format(
            len(duplicates), [(_format_key(key), count)
                              for key, count in duplicates[:8]]))
    return failures


def audit_semantic_task_bijection(contract, run_dir, trace):
    failures = []
    failures.extend(validate_semantic_contract(contract, base_dir=run_dir, verify_sources=True))
    abi_failures, _abi = validate_abi_lock(contract, base_dir=run_dir)
    failures.extend(abi_failures)
    descriptors = task_descriptors(contract)
    contracts = {}
    expected = set()
    dynamic_declared = {}
    dynamic_contracts = {}
    for descriptor in descriptors:
        event_id = descriptor_id(descriptor)
        task_contract = descriptor.get("task_contract")
        if event_id is None or not isinstance(task_contract, dict):
            continue
        contracts[event_id] = task_contract
        try:
            keys = enumerate_task_instances(task_contract, contract)
        except SemanticContractError:
            continue
        if _task_is_dynamic_claim(task_contract):
            dynamic_declared[event_id] = set(keys)
            dynamic_contracts[event_id] = task_contract
            continue
        expected.update((event_id, key) for key in keys)
    if not contracts:
        return failures or ["no semantic task contracts are available"]
    raw_paths = _expand_evidence_paths(contract, run_dir, "raw_pairs")
    decoded_paths = _expand_evidence_paths(contract, run_dir, "decoded_events")
    if not raw_paths:
        failures.append("semantic bijection raw pair evidence is missing")
    if not decoded_paths:
        failures.append("semantic bijection decoded event evidence is missing")
    raw, raw_failures = _collect_layer(raw_paths, "raw_pairs", contracts, contract)
    decoded, decoded_failures = _collect_layer(
        decoded_paths, "decoded_events", contracts, contract)
    traced, trace_failures = _collect_trace(trace, contracts, contract)
    failures.extend(raw_failures)
    failures.extend(decoded_failures)
    failures.extend(trace_failures)
    # Static tasks keep exact bijection; dynamic-claimed tasks are excluded
    # from the static expected set and audited in coverage mode below.
    static_raw = {key: uids for key, uids in raw.items()
                  if key[0] not in dynamic_declared}
    static_decoded = {key: uids for key, uids in decoded.items()
                      if key[0] not in dynamic_declared}
    failures.extend(_compare_layer("raw pairs", static_raw, expected))
    failures.extend(_compare_layer("decoded events", static_decoded, expected))
    for event_id, declared_keys in dynamic_declared.items():
        task_contract = dynamic_contracts[event_id]
        for layer_name, layer in (("raw pairs", raw), ("decoded events", decoded)):
            sub = {key: uids for (eid, key), uids in layer.items()
                   if eid == event_id}
            failures.extend(_coverage_compare_layer(
                "{} (dynamic-claim)".format(layer_name), sub, declared_keys,
                task_contract))
    display_rank = None
    capture = contract.get("capture") if isinstance(contract, dict) else None
    if isinstance(capture, dict):
        display_rank = capture.get("display_rank")
    trace_expected = expected
    if isinstance(display_rank, int) and not isinstance(display_rank, bool):
        filtered = set()
        for key in expected:
            values = dict(key[1])
            if "rank" not in values or values["rank"] == display_rank:
                filtered.add(key)
        trace_expected = filtered
    static_traced = {key: uids for key, uids in traced.items()
                     if key[0] not in dynamic_declared}
    failures.extend(_compare_layer("Perfetto task spans", static_traced, trace_expected))
    for event_id, declared_keys in dynamic_declared.items():
        task_contract = dynamic_contracts[event_id]
        sub = {key: uids for (eid, key), uids in traced.items()
               if eid == event_id}
        trace_declared = declared_keys
        if isinstance(display_rank, int) and not isinstance(display_rank, bool):
            sub = {key: uids for key, uids in sub.items()
                   if "rank" not in dict(key) or dict(key)["rank"] == display_rank}
            trace_declared = {key for key in declared_keys
                              if "rank" not in dict(key)
                              or dict(key)["rank"] == display_rank}
        failures.extend(_coverage_compare_layer(
            "Perfetto task spans (dynamic-claim)", sub, trace_declared,
            task_contract))
    for key in sorted(trace_expected & set(raw) & set(decoded) & set(traced), key=str):
        raw_uid = raw[key][0] if len(raw[key]) == 1 else None
        decoded_uid = decoded[key][0] if len(decoded[key]) == 1 else None
        trace_uid = traced[key][0] if len(traced[key]) == 1 else None
        if raw_uid is not None and (raw_uid != decoded_uid or raw_uid != trace_uid):
            failures.append("source_event_uid mismatch for {}: raw={} decoded={} trace={}".format(
                _format_key(key), raw_uid, decoded_uid, trace_uid))
    return failures


def _selftest_fixture(root, axes, key_fields, collaborating_warps,
                      recording_representative, delivered_keys):
    source = root / "kernel.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("semantic_begin()\nsemantic_sync()\nsemantic_end()\n",
                      encoding="utf-8")
    task_contract = {
        "classification": "task",
        "semantic_unit": "one source-defined work instance",
        "instance_key": list(key_fields),
        "instance_domain": {"axes": axes},
        "expected_cardinality": len(enumerate_task_instances(
            {"instance_key": list(key_fields), "instance_domain": {"axes": axes}}, {})),
        "collaborating_warps": collaborating_warps,
        "recording_representative": recording_representative,
        "tile_shape": {"rows": 128, "columns": 64},
        "source_boundary": {
            "begin": {"path": "kernel.py", "line": 1, "anchor": "semantic_begin"},
            "completion_sync": {"path": "kernel.py", "line": 2,
                                "anchor": "semantic_sync"},
            "end": {"path": "kernel.py", "line": 3, "anchor": "semantic_end"},
        },
    }
    descriptor = {"id": 20, "name": "Cooperative Tile",
                  "display_name": "Cooperative Tile", "layer": "task",
                  "task_contract": task_contract}
    analysis = {
        "schema": SCHEMA,
        "model_reviewed": True,
        "source_closure": [{"path": "kernel.py", "sha256": sha256_file(source)}],
    }
    abi = {"semantic_analysis": analysis, "tasks": [descriptor]}
    analysis["conclusion_sha256"] = semantic_conclusion_sha256(abi)
    abi_path = root / "profiler_abi.json"
    abi_path.write_text(json.dumps(abi, sort_keys=True), encoding="utf-8")
    contract = {
        "project_layout": {"version_root": str(root)},
        "capture": {"display_rank": 0},
        "semantic_analysis": json.loads(json.dumps(analysis)),
        "tasks": json.loads(json.dumps([descriptor])),
    }
    contract["semantic_analysis"].update({
        "abi_path": str(abi_path),
        "abi_sha256": sha256_file(abi_path),
        "bijection_evidence": {
            "raw_pairs": ["sw/raw_pairs.json"],
            "decoded_events": ["sw/events.json"],
        },
    })
    raw_items = []
    decoded_items = []
    trace_events = []
    for index, key_values in enumerate(delivered_keys):
        key = {field: key_values[field] for field in key_fields}
        key_tuple = tuple((field, key[field]) for field in key_fields)
        recorder = _resolved_recorder(task_contract, key_tuple, contract)
        lane = key.get("warp", key.get("cta", 0))
        start_offset = index * 2
        end_offset = start_offset + 1
        work_id = index
        uid = [key.get("rank", 0), lane, start_offset, end_offset, 20, work_id]
        common = {
            "event_id": 20,
            "instance_key": key,
            "source_event_uid": uid,
            "is_derived": False,
            "canonical_recorder": recorder,
            "lane": lane,
            "work_id": work_id,
            "source_contributor_count": 1,
        }
        raw = dict(common)
        raw.update({"start_offset": start_offset, "end_offset": end_offset})
        raw_items.append(raw)
        decoded_items.append(dict(common))
        trace_events.append({
            "name": "Cooperative Tile", "ph": "X", "pid": lane + 1,
            "tid": lane, "ts": float(index), "dur": 0.5,
            "args": dict(common, layer="task", task_id=20),
        })
    sw = root / "sw"
    sw.mkdir(parents=True, exist_ok=True)
    (sw / "raw_pairs.json").write_text(
        json.dumps({"raw_pairs": raw_items}), encoding="utf-8")
    (sw / "events.json").write_text(
        json.dumps({"events": decoded_items}), encoding="utf-8")
    return contract, {"traceEvents": trace_events}, source, raw_items, decoded_items


def selftest_semantic_task_bijection():
    failures = []
    temp = Path(tempfile.mkdtemp(prefix="semantic_bijection_selftest_"))

    def run_case(name, axes, key_fields, warps, recorder, delivered,
                 mutate=None, expect_failure=False, mention=None):
        root = temp / name
        contract, trace, source, raw, decoded = _selftest_fixture(
            root, axes, key_fields, warps, recorder, delivered)
        if mutate is not None:
            mutate(contract, trace, source, raw, decoded, root)
        found = audit_semantic_task_bijection(contract, root, trace)
        if expect_failure and not found:
            failures.append("{}: semantic audit accepted invalid evidence".format(name))
        elif not expect_failure and found:
            failures.append("{}: semantic audit rejected valid evidence: {}".format(
                name, found[:2]))
        elif expect_failure and mention and not any(mention in item for item in found):
            failures.append("{}: failure does not mention {!r}: {}".format(
                name, mention, found[:3]))

    try:
        cooperative_axes = {"rank": [0], "cta": [0], "tile": [0]}
        cooperative_key = ["rank", "cta", "tile"]
        cooperative_warps = list(range(12))
        cooperative_recorder = {"warp": 0, "lane": 0, "thread_idx": 0}
        one_tile = [{"rank": 0, "cta": 0, "tile": 0}]
        run_case("cooperative_canonical", cooperative_axes, cooperative_key,
                 cooperative_warps, cooperative_recorder, one_tile)

        run_case("warp_replicas", cooperative_axes, cooperative_key,
                 cooperative_warps, cooperative_recorder, one_tile * 12,
                 expect_failure=True, mention="duplicate semantic instances")

        run_case("missing_loop_tile",
                 {"rank": [0], "cta": [0], "tile": [0, 1]}, cooperative_key,
                 cooperative_warps, cooperative_recorder, one_tile,
                 expect_failure=True, mention="misses 1 semantic instances")

        def mark_cross_cta_envelope(_contract, trace, _source, raw, decoded, root):
            for item in raw + decoded:
                item["source_contributor_count"] = 2
                item["task_envelope"] = True
            for event in trace["traceEvents"]:
                event["args"]["source_contributor_count"] = 2
                event["args"]["task_envelope"] = True
            sw = root / "sw"
            (sw / "raw_pairs.json").write_text(
                json.dumps({"raw_pairs": raw}), encoding="utf-8")
            (sw / "events.json").write_text(
                json.dumps({"events": decoded}), encoding="utf-8")

        run_case("cross_cta_envelope",
                 {"rank": [0], "cta": [0, 1], "tile": [0]}, cooperative_key,
                 cooperative_warps, cooperative_recorder, one_tile,
                 mutate=mark_cross_cta_envelope, expect_failure=True,
                 mention="aggregation/envelope")

        warp_axes = {"rank": [0], "warp": [3, 4], "tile": [0]}
        warp_key = ["rank", "warp", "tile"]
        warp_members = {"warps": [{"ref": "warp"}]}
        warp_recorder = {"warp": {"ref": "warp"}, "lane": 0,
                         "thread_idx": {"op": "mul", "left": {"ref": "warp"},
                                        "right": 32}}
        run_case("explicit_warp_scope", warp_axes, warp_key, warp_members,
                 warp_recorder,
                 [{"rank": 0, "warp": 3, "tile": 0},
                  {"rank": 0, "warp": 4, "tile": 0}])

        def drift_source(_contract, _trace, source, _raw, _decoded, _root):
            source.write_text(source.read_text(encoding="utf-8") + "changed()\n",
                              encoding="utf-8")

        run_case("source_drift", cooperative_axes, cooperative_key,
                 cooperative_warps, cooperative_recorder, one_tile,
                 mutate=drift_source, expect_failure=True, mention="SHA256 drift")

        def drop_uid(_contract, trace, _source, _raw, _decoded, _root):
            trace["traceEvents"][0]["args"].pop("source_event_uid")

        run_case("uid_missing", cooperative_axes, cooperative_key,
                 cooperative_warps, cooperative_recorder, one_tile,
                 mutate=drop_uid, expect_failure=True, mention="source_event_uid")

        def mismatch_uid(_contract, trace, _source, _raw, _decoded, _root):
            trace["traceEvents"][0]["args"]["source_event_uid"][5] = 999

        run_case("uid_mismatch", cooperative_axes, cooperative_key,
                 cooperative_warps, cooperative_recorder, one_tile,
                 mutate=mismatch_uid, expect_failure=True,
                 mention="source_event_uid mismatch")
    finally:
        shutil.rmtree(str(temp), ignore_errors=True)
    return failures

