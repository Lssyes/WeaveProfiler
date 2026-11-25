#!/usr/bin/env python3
"""Size and hygiene metrics of the core code (the ratchet for refactoring rounds).

Core = weaveprof/**/*.py, run.py, recorders/** (tests, examples, configs, docs are excluded).

    python3 -B tests/quality/code_metrics.py            # print metrics + the 20 largest core files
    python3 -B tests/quality/code_metrics.py --json     # machine-readable

tests/quality/test_code_metrics_ratchet.py fails when any metric grows past tests/quality/baseline_metrics.json.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CORE_GLOBS = ("weaveprof/**/*.py", "run.py", "recorders/**/*.py", "recorders/**/*.cuh", "recorders/**/*.h")

_BROAD = re.compile(r"except\s+(?:Exception|BaseException)\b|except\s*:")
_SWALLOW = re.compile(r"except[^\n]*:\s*\n\s*(?:pass|continue)\b")
_PRINT = re.compile(r"(?<![\w.])print\(")


def core_files():
    seen = set()
    for g in CORE_GLOBS:
        for p in sorted(ROOT.glob(g)):
            if p.is_file() and "__pycache__" not in p.parts and p not in seen:
                seen.add(p)
                yield p


def _code_lines(text: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.strip())


def collect():
    per_file = {}
    m = {"core_files": 0, "core_code_lines": 0, "py_functions": 0, "py_functions_over_100_lines": 0,
         "py_functions_over_300_lines": 0, "broad_except": 0, "swallowed_except": 0, "print_calls": 0}
    for p in core_files():
        text = p.read_text(encoding="utf-8", errors="replace")
        rel = p.relative_to(ROOT).as_posix()
        n = _code_lines(text)
        per_file[rel] = n
        m["core_files"] += 1
        m["core_code_lines"] += n
        if p.suffix == ".py":
            m["broad_except"] += len(_BROAD.findall(text))
            m["swallowed_except"] += len(_SWALLOW.findall(text))
            m["print_calls"] += len(_PRINT.findall(text))
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    length = (node.end_lineno or node.lineno) - node.lineno + 1
                    m["py_functions"] += 1
                    m["py_functions_over_100_lines"] += length > 100
                    m["py_functions_over_300_lines"] += length > 300
    return m, per_file


def main() -> int:
    metrics, per_file = collect()
    if "--json" in sys.argv:
        print(json.dumps({"metrics": metrics, "per_file_code_lines": per_file}, indent=1, sort_keys=True))
        return 0
    for k, v in metrics.items():
        print(f"{k:32s} {v}")
    print("\nlargest core files (non-blank lines):")
    for rel, n in sorted(per_file.items(), key=lambda kv: -kv[1])[:20]:
        print(f"{n:7d}  {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
