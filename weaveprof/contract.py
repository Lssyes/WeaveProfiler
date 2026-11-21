"""Public, source-line-based probe contract and deterministic refinement."""
from __future__ import annotations

import ast
import bisect
from collections.abc import Mapping
from copy import deepcopy
import re


STAGES = {name: i for i, name in enumerate((
    "load", "wait", "mma", "compute", "epilogue", "store", "barrier", "dispatch",
    "communication", "gemm", "attention", "reduction", "moe", "convolution",
    "megakernel", "softmax",
), 1)}


def spill_delta_bytes(baseline, instrumented, reported=None):
    """Read measured spill bytes from both builds, or a previously computed delta."""
    if reported is not None and type(reported) is not int:
        raise ValueError("delta_spill_bytes must be a signed integer")
    if baseline is None and instrumented is None and reported is not None:
        return reported
    if any(type(value) is not int or value < 0 for value in (baseline, instrumented)):
        raise ValueError("Expected nonnegative spill byte counts for both builds")
    delta = instrumented - baseline
    if reported is not None and reported != delta:
        raise ValueError("delta_spill_bytes disagrees with instrumented minus baseline spill bytes")
    return delta


def compile_role(expression, language="cuda_cpp"):
    """Translate warp ranges without depending on a kernel-local variable's scope."""
    expression = _expression(expression, "role", language=language)
    python = language in ("triton", "cutedsl")
    if expression.lower() in ("all", "cta", "true"):
        return "True" if python else "true"
    match = re.fullmatch(r"warps?\s+(\d+)(?:\s*-\s*(\d+))?", expression, re.I)
    if match is None:
        if re.match(r"^warps?\s", expression, re.I):
            raise ValueError(f"Invalid warp range {expression!r}")
        return expression
    low, high = match.groups()
    warp = {"triton": "_wp_warp_id", "cutedsl": "cute.arch.warp_idx()"}.get(language, "(threadIdx.x / warpSize)")
    if high is None:
        return f"({warp} == {int(low)})"
    if int(low) > int(high):
        raise ValueError(f"Invalid warp range {expression!r}")
    conjunction = "&" if python else "&&"
    return f"(({warp} >= {int(low)}) {conjunction} ({warp} <= {int(high)}))"


def _expression(value, field, *, language="cuda_cpp"):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a single expression")
    expr = value.strip()
    if language in ("triton", "cutedsl", "python"):
        # Allow Python floor division (`//`) while blocking statements/comments/newlines.
        if re.search(r"[;#\r\n]", expr):
            raise ValueError(f"{field} must be a single expression")
        if field != "role":
            try:
                ast.parse(expr, mode="eval")
            except SyntaxError as exc:
                raise ValueError(f"{field} must be a single expression") from exc
        return expr
    # Tolerate harmless inline comments / trailing semicolon emitted by weaker LLMs on a single line.
    if "\n" not in expr and "\r" not in expr:
        expr = re.sub(r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/", "", expr)
        expr = re.sub(r"//[^\n]*$", "", expr).strip()
        if expr.endswith(";") and expr.count(";") == 1:
            expr = expr[:-1].strip()
    if not expr or re.search(r"[;{}#\r\n]|//|/\*|\*/", expr):
        raise ValueError(f"{field} must be a single expression")
    return expr


def _integer(value, field, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {field}: {value!r}")
    return value


def normalize_probe_spec(doc, source_text=None, *, language=None):
    """Return independent paper endpoints and v1 aliases, retaining extension fields.

    Paper fields are authoritative when both views exist. Distinct paper probes
    receive distinct 12-bit event IDs even when they share a semantic class.
    The optional v1 extension preserves role IDs, predicates and extra fields
    across JSON serialization; v1 consumers can read the aliases directly.
    """
    if not isinstance(doc, Mapping):
        raise ValueError("probe.json must contain an object")
    result = deepcopy(dict(doc))
    text = source_text or ""
    if language is None:
        language = result.get("language") or (
            "cutedsl" if "cute.kernel" in text or "cutlass.cute" in text else
            "triton" if "triton" in text else
            "hip_cpp" if "hip/" in text or "__builtin_amdgcn" in text else "cuda_cpp"
        )
    result.setdefault("language", language)
    legacy = "roles" not in result
    old_roles = result.get("warp_roles", result.get("v1_roles", []))
    if legacy:
        old_roles = old_roles or [{"name": "cta_worker", "role_id": 0, "predicate": "true"}]
        result["roles"] = {r["name"]: r.get("predicate") or "true" for r in old_roles}
        result["v1_roles"] = deepcopy(old_roles)
    roles = result["roles"]
    if not isinstance(roles, dict) or not roles:
        raise ValueError("roles must map role names to predicates")
    lookup = {r["name"]: r for r in old_roles}
    role_ids = {}
    result["warp_roles"] = []
    for index, (name, expression) in enumerate(roles.items()):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each role must have a nonempty name")
        entry = deepcopy(lookup.get(name, {}))
        rid = _integer(entry.get("role_id", index), "role ID", maximum=0xfff)
        if rid in role_ids.values():
            raise ValueError(f"Duplicate role ID {rid}")
        role_ids[name] = rid
        entry.update(name=name, role_id=rid, predicate=compile_role(expression, language))
        result["warp_roles"].append(entry)

    raw_probes = result.get("probes")
    if not isinstance(raw_probes, list) or not raw_probes:
        raise ValueError("probes must be a nonempty array")
    n_lines = len(text.splitlines()) if source_text is not None else None
    ids = set()
    event_ids = set()
    # Reserve supplied IDs before allocating IDs for arbitrary, named stages.
    reserved = {p["event_id"] for p in raw_probes if isinstance(p, dict) and "event_id" in p}
    next_id = max(STAGES.values()) + 1
    for p in raw_probes:
        if not isinstance(p, dict):
            raise ValueError("Each probe must be an object")
        pid = p.get("id", p.get("probe_id"))
        if not isinstance(pid, str) or not pid.strip() or pid in ids:
            raise ValueError(f"Invalid or duplicate probe ID {pid!r}")
        ids.add(pid)
        if legacy:
            role = p.get("role") or next((k for k,v in role_ids.items() if v == p.get("role_id")), next(iter(roles)))
            p.setdefault("begin", {"role": role, "line": p.get("begin_line")})
            p.setdefault("end", {"role": role, "line": p.get("end_line", p.get("begin_line"))})
            if p.get("aux_expr"):
                p.setdefault("slot", p["aux_expr"])
        for side in ("begin", "end"):
            endpoint = p.get(side)
            if not isinstance(endpoint, dict) or endpoint.get("role") not in roles:
                raise ValueError(f"{pid}.{side} must identify a declared role")
            p[side + "_line"] = _integer(endpoint.get("line"), f"{pid}.{side}.line", 1, n_lines)
            p[side + "_role_id"] = role_ids[endpoint["role"]]
        if p["begin"]["role"] == p["end"]["role"] and p["end_line"] < p["begin_line"]:
            raise ValueError(f"Reversed endpoints for {pid}")
        p.update(id=pid, probe_id=pid, role=p["begin"]["role"], role_id=p["begin_role_id"])
        if "slot" in p:
            raw_slot = p["slot"]
            if raw_slot is None or (isinstance(raw_slot, str) and not raw_slot.strip()):
                p.pop("slot", None)
                p.setdefault("aux_expr", "")
            else:
                p["aux_expr"] = _expression(str(raw_slot) if type(raw_slot) is int else raw_slot, "slot", language=language)
        else:
            p.setdefault("aux_expr", "")
        sem = p.get("semantic_id") or next(
            (s for s in STAGES if pid == s or pid.startswith(s + "_") or re.fullmatch(rf"{s}\d+", pid)),
            None,
        )
        if sem is None:
            low_pid = pid.lower()
            if low_pid.startswith(("dma", "tma", "prefetch", "copy", "cp_async")):
                sem = "load"
            elif low_pid.startswith(("wgmma", "hmma", "tcgen")):
                sem = "mma"
            elif low_pid.startswith(("sync", "mbar")):
                sem = "barrier"
            else:
                sem = "compute"
        eid = p.get("event_id")
        if eid is None:
            preferred = STAGES.get(sem)
            if preferred is not None and preferred not in event_ids | reserved:
                eid = preferred
            else:
                while next_id in event_ids | reserved:
                    next_id += 1
                eid = next_id
        _integer(eid, "event ID", 1, 0xfff)
        if eid in event_ids and not (legacy or "v1_roles" in doc):
            raise ValueError(f"Duplicate paper event ID {eid}")
        event_ids.add(eid)
        p.update(semantic_id=sem, event_id=eid)
    result.setdefault("schema", "weaveprofiler.probe-spec/v1")
    return result


def compact_probe_spec(doc):
    """Materialize the paper representation; keep v1-only metadata if supplied."""
    result = deepcopy(dict(doc)) if "roles" in doc else normalize_probe_spec(doc)
    result.pop("warp_roles", None)
    result.pop("schema", None) if result.get("schema") == "weaveprofiler.probe-spec/v1" else None
    for p in result["probes"]:
        for key in ("probe_id", "role", "role_id", "begin_line", "end_line", "begin_role_id", "end_role_id", "aux_expr"):
            p.pop(key, None)
        if "v1_roles" not in result:
            p.pop("event_id", None) if p.get("event_id") == STAGES.get(p.get("semantic_id")) else None
            if p.get("semantic_id") == p.get("id"):
                p.pop("semantic_id", None)
    return result


def loop_spans(source_text, language):
    """Return inclusive 1-based loop extents, including the last body/brace line."""
    if language in ("triton", "cutedsl", "python"):
        return [(n.lineno, n.end_lineno) for n in ast.walk(ast.parse(source_text))
                if isinstance(n, (ast.For, ast.AsyncFor, ast.While))]
    # Tokenize once. Comments, literals and preprocessor directives cannot
    # introduce braces or control-flow tokens into the host grammar.
    token_rx = re.compile(r'''//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|^[ \t]*\#[^\n]*(?:\\\n[^\n]*)*|\w+|[^\s]''', re.M)
    tokens, lines = [], []
    newlines = [m.start() for m in re.finditer("\n", source_text)]
    for match in token_rx.finditer(source_text):
        token = match.group()
        if token.startswith(("//", "/*")) or token.lstrip().startswith("#"):
            continue
        tokens.append(token)
        lines.append(1 + bisect.bisect_left(newlines, match.start()))
    pairs, stack = {}, []
    for i, token in enumerate(tokens):
        if token in ("(", "[", "{"):
            stack.append(i)
        elif token in (")", "]", "}") and stack:
            pairs[stack.pop()] = i
    def statement(i):
        if i >= len(tokens):
            return i - 1
        if tokens[i] == "{":
            return pairs.get(i, i)
        if tokens[i] in ("for", "while", "if") and i + 1 in pairs:
            end = statement(pairs[i + 1] + 1)
            if tokens[i] == "if" and tokens[end + 1:end + 2] == ["else"]:
                return statement(end + 2)
            return end
        while i < len(tokens) and tokens[i] not in (";", "}"):
            i = pairs.get(i, i) + 1
        return min(i, len(tokens) - 1)
    spans = []
    for i, token in enumerate(tokens):
        if token in ("for", "while") and tokens[i + 1:i + 2] == ["("] and i + 1 in pairs:
            end = statement(pairs[i + 1] + 1)
            spans.append((lines[i], lines[end]))
        elif token == "do":
            end = statement(i + 1)
            if tokens[end + 1:end + 2] == ["while"] and end + 2 in pairs:
                spans.append((lines[i], lines[min(pairs[end + 2] + 1, len(tokens) - 1)]))
    return spans


def enclosing_loop(probe, spans):
    a, b = sorted((probe["begin"]["line"], probe["end"]["line"]))
    candidates = [(start, end) for start, end in spans if start <= a <= b <= end and (a, b) != (start, end)]
    return list(min(candidates, key=lambda s: s[1] - s[0])) if candidates else None


def hot_loop(doc, fires, source_text):
    spec = normalize_probe_spec(doc, source_text)
    spans = loop_spans(source_text, spec["language"])
    for p in sorted(spec["probes"], key=lambda x: -fires.get(x["id"], 0)):
        if fires.get(p["id"], 0) <= 0 or p["begin"]["role"] != p["end"]["role"]:
            continue
        loop = enclosing_loop(p, spans)
        if loop:
            return p["id"], loop
    return None, None


_STAGE_MERGE_RANK = {
    "mma": 100, "gemm": 95, "attention": 90, "moe": 90, "compute": 85,
    "softmax": 80, "reduction": 75, "epilogue": 70, "communication": 65,
    "store": 50, "load": 40, "dispatch": 30, "wait": 20, "barrier": 15, "dma_issue": 10,
}


def revise_probe_spec(doc, feedback, source_text):
    """Move a measured hot synchronous probe to its enclosing loop boundaries."""
    result = compact_probe_spec(doc)
    if feedback["status"] != "FAIL" or not feedback.get("enclosing_loop"):
        return result
    pid, loop = hot_loop(result, feedback["fires"], source_text)
    if loop != feedback["enclosing_loop"]:
        raise ValueError("Feedback loop span does not match this probe/source revision")
    p = next(p for p in result["probes"] if p["id"] == pid)
    p["begin"]["line"], p["end"]["line"] = loop
    p.pop("slot", None)
    if "in_inner_loop" in p:
        p["in_inner_loop"] = False
    colliding = [
        q for q in result["probes"]
        if q["id"] != pid and (q["begin"], q["end"]) == (p["begin"], p["end"])
    ]
    if colliding:
        winner = max(
            [p, *colliding],
            key=lambda x: _STAGE_MERGE_RANK.get(
                x.get("semantic_id") or next((s for s in STAGES if x["id"] == s or x["id"].startswith(s + "_")), ""),
                0,
            ),
        )
        drop_ids = {q["id"] for q in [p, *colliding] if q["id"] != winner["id"]}
        result["probes"] = [q for q in result["probes"] if q["id"] not in drop_ids]
    else:
        result["probes"] = [
            q for q in result["probes"]
            if q["id"] == pid or (q["begin"], q["end"]) != (p["begin"], p["end"])
        ]
    return result
