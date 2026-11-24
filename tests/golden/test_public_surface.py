#!/usr/bin/env python3
"""Contract test: snapshot of everything users and agents call from outside the package.

Covers the `run.py` subcommands, the `weaveprof` CLI subcommands, `weaveprof.__all__`, the MCP tool names and
input schemas, the AKA plugin tools and schemas, the public signature of run_megakernel_audit, and the device /
host API of the recorders (CUDA, HIP, Triton, CuTeDSL). Internal functions are deliberately not covered so a
refactor can rewrite them freely; any change here is an API change and must be intentional:

    WEAVEPROF_GOLDEN_UPDATE=1 python -B -m unittest tests/golden/test_public_surface.py

Run:  python -B -m unittest tests/golden/test_public_surface.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import ast
import hashlib
import inspect
import json
import os
import re
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SNAPSHOT = HERE / "public_surface.json"
UPDATE = os.environ.get("WEAVEPROF_GOLDEN_UPDATE") == "1"


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _run_py_commands():
    tree = ast.parse((ROOT / "run.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "handlers" for t in node.targets):
            return sorted(k.value for k in node.value.keys)
    raise AssertionError("run.py handlers dict not found")


def _cli_commands():
    out = subprocess.run([sys.executable, "-B", "-m", "weaveprof", "--help"], cwd=str(ROOT), capture_output=True,
                         text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=120)
    m = re.search(r"\{([a-z0-9_,-]+)\}", out.stdout + out.stderr)
    if not m:
        raise AssertionError(f"could not parse `python -m weaveprof --help`:\n{out.stdout}\n{out.stderr}")
    return sorted(m.group(1).split(","))


def _python_api():
    import weaveprof
    return sorted(weaveprof.__all__)


def _mcp_tools():
    from weaveprof import mcp_server
    return {t["name"]: _sha(t.get("inputSchema")) for t in mcp_server.TOOL_DEFS}


def _aka_plugin():
    doc = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
    tools = {}
    for name, spec in sorted((doc.get("tools") or {}).items()):
        schema = ROOT / spec["input_schema"]
        tools[name] = {"command": spec.get("command"),
                       "input_schema_sha": _sha(json.loads(schema.read_text(encoding="utf-8")))}
    return tools


def _audit_signature():
    from weaveprof.megakernel import run_megakernel_audit
    sig = inspect.signature(run_megakernel_audit)
    return [f"{p.name}={p.default!r}" if p.default is not inspect.Parameter.empty else p.name
            for p in sig.parameters.values()]


def _header_api(path: Path, prefixes=("WP_",)):
    text = path.read_text(encoding="utf-8", errors="replace")
    macros = set()
    for m in re.finditer(r"^\s*#\s*define\s+([A-Z][A-Z0-9_]*)\s*\(", text, re.M):
        if m.group(1).startswith(prefixes):
            macros.add(m.group(1) + "()")
    for m in re.finditer(r"^\s*#\s*if(?:n?def)?\s+[^\n]*?\b(WP_[A-Z0-9_]+)\b", text, re.M):
        macros.add(m.group(1))  # build switches tested by the header
    funcs = set(re.findall(r"\b(?:static\s+inline|extern\s+\"C\")[^;{(]*?\b(wp_[a-z0-9_]+)\s*\(", text))
    return {"macros": sorted(macros), "host_functions": sorted(funcs)}


def _py_module_api(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                    names |= {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    return sorted(names)


def collect():
    return {
        "run_py_commands": _run_py_commands(),
        "weaveprof_cli_commands": _cli_commands(),
        "python_api": _python_api(),
        "mcp_tools": _mcp_tools(),
        "aka_plugin_tools": _aka_plugin(),
        "run_megakernel_audit_signature": _audit_signature(),
        "recorder_cuda_cpp": _header_api(ROOT / "recorders" / "cuda_cpp" / "wp_lite.cuh"),
        "recorder_hip_cpp": _header_api(ROOT / "recorders" / "hip_cpp" / "wp_lite_hip.h"),
        "recorder_triton": {p.name: _py_module_api(p) for p in sorted((ROOT / "recorders" / "triton").glob("*.py"))},
        "recorder_cutedsl": {p.name: _py_module_api(p) for p in sorted((ROOT / "recorders" / "cutedsl").glob("*.py"))},
    }


def _explain(expected, actual, path="$"):
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for k in sorted(set(expected) | set(actual)):
            if k not in actual:
                out.append(f"{path}.{k}: removed")
            elif k not in expected:
                out.append(f"{path}.{k}: added")
            else:
                out += _explain(expected[k], actual[k], f"{path}.{k}")
        return out
    if isinstance(expected, list) and isinstance(actual, list) and expected != actual:
        e, a = set(map(str, expected)), set(map(str, actual))
        if e != a:
            return [f"{path}: removed {sorted(e - a)} added {sorted(a - e)}"]
        return [f"{path}: order changed"]
    return [] if expected == actual else [f"{path}: {expected!r} -> {actual!r}"]


class PublicSurfaceTest(unittest.TestCase):
    def test_public_surface_unchanged(self):
        actual = collect()
        if UPDATE or not SNAPSHOT.is_file():
            if not UPDATE:
                self.fail(f"{SNAPSHOT} missing; generate it with WEAVEPROF_GOLDEN_UPDATE=1")
            SNAPSHOT.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            return
        expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        changes = _explain(expected, json.loads(json.dumps(actual)))
        self.assertEqual(changes, [], "public surface changed:\n  " + "\n  ".join(changes))


if __name__ == "__main__":
    unittest.main()
