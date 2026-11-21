#!/usr/bin/env python3
"""Generic ABI loader + safe integer-expression evaluator for the record ABI.

FULLY IMPLEMENTED — do not edit for a normal project. Adapted from
knowledge/examples/host/decode.py (`eval_expr`, `load_abi`, `sha256_of`,
`check_abi_provenance`). decode_profile.py, run_profile.py and the offline
selftests all import from here so the layout math lives in exactly one place.

The ABI *data* (device_record/abi/record_abi.example.json) is the project hook;
this loader is generic.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

# Accept both the generic skeleton schema and the historical reference schema so
# an ABI copied verbatim from knowledge/examples/host/abi_reference.json still loads.
ABI_SCHEMAS = ("fast-instrument.record-abi/v1", "megamoe.profiler-abi/v1")
REQUIRED_SECTIONS = ("tag_bitfields", "tasks", "tracks", "buffer", "regions")


class ABIError(RuntimeError):
    pass


def _exact_div(a: int, b: int) -> int:
    """"/" in an ABI expression must divide exactly.

    WHY not just floor it: these expressions size the record buffer and compute slot
    addresses. Silently flooring `num_sms * warps / 4` when it does not divide evenly
    under-allocates the buffer, and the symptom surfaces far away as dropped records on the
    last lanes -- with every counter still zero. Write `//` if you really mean flooring.
    """
    if b == 0:
        raise ABIError("ABI expression divides by zero")
    if a % b != 0:
        raise ABIError(
            "ABI expression uses '/' but {} / {} is not exact. These formulas size the record "
            "buffer and address slots, so a truncated result silently drops records. Either fix "
            "the formula so it divides evenly, or write '//' to state that flooring is "
            "intended.".format(a, b))
    return a // b


_EXPR_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Div: _exact_div,
}


def eval_expr(expr: str, env: Dict[str, int]) -> int:
    """Evaluate an ABI layout expression (e.g. "1 + num_sms") over integers only."""
    def walk(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in _EXPR_BINOPS:
            return _EXPR_BINOPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ABIError("ABI expression {!r} uses unknown symbol {!r}".format(expr, node.id))
            return int(env[node.id])
        raise ABIError("ABI expression {!r} contains an unsupported construct".format(expr))

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as error:
        raise ABIError("cannot parse ABI expression {!r}: {}".format(expr, error))
    return int(walk(tree))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_abi(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ABIError(
            "ABI file not found: {}; regenerate it with the project's extract_abi.py".format(path))
    abi = json.loads(path.read_text(encoding="utf-8"))
    if abi.get("schema") not in ABI_SCHEMAS:
        raise ABIError("ABI schema {!r} is not one of {}".format(abi.get("schema"), ABI_SCHEMAS))
    for key in REQUIRED_SECTIONS:
        if key not in abi:
            raise ABIError("ABI file is missing section {!r}".format(key))
    return abi


# The sentinels this skeleton ships in unfilled hooks. Same two leading tokens as
# run_profile.device_source_sha256 and validation/selftest's PROJECT_HOOK_REPLACE sentinel;
# TODO/FIXME cover ABIs generated from other templates. Deliberately NOT generic words like
# "xxx" — those appear in real directory names and would misreport a filled hook.
PLACEHOLDER_TOKENS = ("PROJECT_HOOK", "REPLACE", "TODO", "FIXME")


def _is_placeholder(value: Any) -> bool:
    text = str(value or "")
    upper = text.upper()
    return any(token.upper() in upper for token in PLACEHOLDER_TOKENS)


def check_abi_provenance(abi: Dict[str, Any], abi_path: Path) -> Dict[str, Any]:
    """Verify the ABI's recorded source_sha256 still matches the on-disk source.

    Returns a dict {role: {status, ...}} where status is one of:
      ok           — recorded hash equals the on-disk source hash.
      mismatch     — the source changed (or the hash is a stale/placeholder string).
                     decode_profile.py escalates ONLY this one to a hard decode failure.
      placeholder  — the hook was never filled (path/hash still say PROJECT_HOOK/REPLACE).
      missing      — declared, filled, but the file is not there.
      not_declared — the ABI does not claim this role at all.

    WHY `placeholder` is a status of its own and why decode does not hard-fail on it: the
    skeleton ships an example ABI pointing at PROJECT_HOOK_REPLACE/device_profiler.cuh and the
    offline selftests decode against it (the skeleton contains no device source), so decode has
    to stay runnable. The enforcement point for a real run is run_profile.device_source_sha256,
    which refuses to capture with a placeholder source path. Do not "fix" this by silencing the
    status — an unfilled hook must stay visible in the decoded metadata and in the delivery.
    """
    abi_path = Path(abi_path)
    root = abi_path.resolve().parent
    result: Dict[str, Any] = {}
    for file_key, hash_key, role in (
        ("source_file", "source_sha256", "source"),
        ("binding_source_file", "binding_source_sha256", "binding"),
        ("host_source_file", "host_source_sha256", "host"),
    ):
        recorded_path = abi.get(file_key)
        recorded_hash = abi.get(hash_key)
        if _is_placeholder(recorded_path) or _is_placeholder(recorded_hash):
            result[role] = {
                "status": "placeholder", "path": recorded_path, "recorded_sha256": recorded_hash,
                "action": "fill {}/{} from the project's extract_abi.py; until then this ABI has "
                          "no tamper evidence for the {} layer".format(file_key, hash_key, role)}
            continue
        if not recorded_path or not recorded_hash:
            # WHY not `continue`: an absent role used to leave no trace, so a delivery could not
            # distinguish "this project genuinely has no binding layer" from "nobody hashed the
            # binding that re-encodes the tag bitfields". Say which one it is.
            result[role] = {
                "status": "not_declared", "path": recorded_path, "recorded_sha256": recorded_hash,
                "action": "if a {} layer exists, record {}/{} in the ABI; if it genuinely does "
                          "not, this entry documents that".format(role, file_key, hash_key)}
            continue
        source_path = Path(recorded_path)
        if not source_path.is_absolute():
            source_path = root / source_path
        if not source_path.is_file():
            result[role] = {"status": "missing", "path": str(source_path),
                            "recorded_sha256": recorded_hash}
            continue
        actual = sha256_of(source_path)
        result[role] = {"status": "ok" if actual == recorded_hash else "mismatch",
                        "path": str(source_path), "recorded_sha256": recorded_hash,
                        "actual_sha256": actual}
    return result


def layout_env(abi: Dict[str, Any], num_sms: int, num_tracks: int, max_records: int) -> Dict[str, int]:
    return {"num_sms": int(num_sms), "num_tracks": int(num_tracks), "max_records": int(max_records)}


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        abi = load_abi(Path(arg))
        prov = check_abi_provenance(abi, Path(arg))
        print("[abi_loader] {} schema={} tasks={} tracks={} provenance={}".format(
            arg, abi.get("schema"), len(abi.get("tasks", [])), len(abi.get("tracks", [])),
            {k: v.get("status") for k, v in prov.items()}))
