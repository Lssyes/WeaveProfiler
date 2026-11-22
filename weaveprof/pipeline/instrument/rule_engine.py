#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§3.2 Fast Probe Injection — Deterministic Rule Engine & CPU Parallel Static Preflight.

100% Pure Deterministic Code (Zero LLM):
  - **(2a) Millisecond Rule Expansion + Zero Call-Chain Signature Edits**:
    * ``CUDA C++`` / ``HIP C++``: Lowers declarative ``JSON Probe Spec`` using ``fi_g.cur`` +
      RAII ``FiScope`` (``FI_KERNEL_SCOPE`` / ``FI_DEVICE_SCOPE``), leaving ``__global__``
      kernels, ``__device__`` subfunctions (``mma_mainloop()``, etc.), templates, and host
      callers with **0 signature changes**.
    * ``Triton`` (NV & AMD) / ``CuTeDSL``: Emits module-level ``fi_attach(buf)`` and same-name
      transparent launcher proxies (``_FiTritonProxy`` / ``_FiCuteProxy``) so external Python
      callers require **0 lines changed**.
  - **(2b) CPU-Side Parallel Static Preflight & Batched Diagnostics**:
    * Checks syntax, spin-loop violations, and interval nesting; compiles ``L0`` (cached) and
      ``L_inst`` in parallel on CPU; runs static SASS/AMDGCN disassembly checks (``Sim_SASS``,
      ``delta_spill_bytes <= 0``, ``0 B added SMEM``, intact occupancy) and aggregates all violations into
      one diagnostic batch before touching any GPU.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SKILL_ROOT = Path(__file__).resolve().parents[3]
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))


def _load_mod(rel_path: str, mod_name: str):
    p = SKILL_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


STAGE_EVENT_IDS: Dict[str, int] = {
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_names_from_probe_spec(probe_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Emit canonical `names.json` compatible with `semantics.py` and `fi_lite_decode.py`."""
    events: Dict[str, str] = {}
    semantics_map: Dict[str, Dict[str, str]] = {}
    if "roles" in probe_spec and not probe_spec.get("warp_roles"):
        from weaveprof.contract import normalize_probe_spec
        probe_spec = normalize_probe_spec(probe_spec)
    cross_warp = {}
    slotted_events = {}
    probe_ids = {}

    op_fam = str(probe_spec.get("operator_family") or "gemm").strip().lower()
    if op_fam in STAGE_EVENT_IDS:
        op_ev = str(STAGE_EVENT_IDS[op_fam])
        events[op_ev] = op_fam
        semantics_map[op_ev] = {"semantic_id": op_fam, "display_name": op_fam}

    for pr in probe_spec.get("probes") or []:
        sem = str(pr.get("semantic_id") or "compute").strip().lower()
        ev_id = int(pr.get("event_id") or STAGE_EVENT_IDS.get(sem, 4))
        events[str(ev_id)] = sem
        semantics_map[str(ev_id)] = {"semantic_id": sem, "display_name": sem}
        probe_id = str(pr.get("id") or pr.get("probe_id") or sem)
        probe_ids[str(ev_id)] = probe_id
        begin_role = int(pr.get("begin_role_id", pr.get("role_id", 0)))
        end_role = int(pr.get("end_role_id", begin_role))
        slot_expr = pr.get("slot", pr.get("aux_expr"))
        if slot_expr is not None and str(slot_expr).strip() not in ("", "0", "0u"):
            slotted_events[str(ev_id)] = dict(id=probe_id, slot=slot_expr, cross_warp=(begin_role != end_role))
        if begin_role != end_role:
            cross_warp[str(ev_id)] = dict(id=probe_id, begin_role=begin_role,
                                          end_role=end_role, slot=pr.get("slot", 0))

    roles_map: Dict[str, str] = {"0": "cta_worker"}
    for r in probe_spec.get("warp_roles") or []:
        rid = str(int(r.get("role_id") or 0))
        roles_map[rid] = str(r.get("name") or f"role_{rid}")

    multi_role = len(probe_spec.get("warp_roles") or []) > 1
    gran = str(probe_spec.get("granularity") or ("warp" if multi_role else "sm"))
    return {
        "events": events,
        "semantics": semantics_map,
        "roles": roles_map,
        "probe_ids": probe_ids,
        "cross_warp": cross_warp,
        "slotted_events": slotted_events,
        "aux": {},
        "label_aux": False,
        "pair": "stack",
        "group": "role" if multi_role else ("sm" if gran == "sm" else "role"),
        "meta": {
            "generator": "weaveprofiler/rule_engine/v1",
            "card": probe_spec.get("card"),
            "arch": probe_spec.get("arch"),
            "language": probe_spec.get("language"),
            "kernel": probe_spec.get("kernel_name"),
            "zero_signature_edits": True,
        },
    }


# ---------------------------------------------------------------------------
# Static Spin-Loop, Loop-Depth & Interval Nesting Validator
# ---------------------------------------------------------------------------

def analyze_loop_nesting(source_text: str, language: str) -> Dict[int, int]:
    """Return 1-based line -> loop nesting depth (0 = outside loops, 1 = outer loop, >=2 = nested inner loop)."""
    lines = source_text.splitlines()
    depth_by_line: Dict[int, int] = {i: 0 for i in range(1, len(lines) + 1)}
    if language in ("triton", "cutedsl"):
        try:
            tree = ast.parse(source_text)
        except SyntaxError:
            return depth_by_line

        def _walk(node: ast.AST, cur_depth: int) -> None:
            is_loop = isinstance(node, (ast.For, ast.While))
            nd = cur_depth + (1 if is_loop else 0)
            if is_loop:
                end_ln = getattr(node, "end_lineno", node.lineno) or node.lineno
                for ln in range(node.lineno, min(len(lines), end_ln) + 1):
                    depth_by_line[ln] = max(depth_by_line[ln], nd)
            for child in ast.iter_child_nodes(node):
                _walk(child, nd)

        _walk(tree, 0)
        return depth_by_line

    # C++ brace-stack loop depth tracker
    stack: List[bool] = []
    pending_loop = False
    for ln, raw in enumerate(lines, start=1):
        clean = re.sub(r"//.*$", "", raw)
        if re.search(r"\b(?:for|while|do)\b", clean):
            pending_loop = True
        for ch in clean:
            if ch == "{":
                stack.append(pending_loop)
                pending_loop = False
            elif ch == "}" and stack:
                stack.pop()
        depth_by_line[ln] = sum(1 for x in stack if x)
    return depth_by_line


def validate_probe_spec_static(
    source_text: str,
    probe_spec: Mapping[str, Any],
    *,
    check_sha: bool = True,
) -> List[Dict[str, Any]]:
    """Run pure-CPU static invariant checks on `probe_spec` before code expansion."""
    placer = _load_mod("weaveprof/pipeline/scan/llm_probe_placer.py", "wp_llm_probe_placer")
    diagnostics: List[Dict[str, Any]] = []
    if "roles" in probe_spec and not probe_spec.get("warp_roles"):
        from weaveprof.contract import normalize_probe_spec
        probe_spec = normalize_probe_spec(probe_spec, source_text=source_text)

    if check_sha and probe_spec.get("source_sha256"):
        actual_sha = sha256_text(source_text)
        if probe_spec["source_sha256"] != actual_sha:
            diagnostics.append({
                "code": "STALE_SOURCE_SHA256",
                "severity": "error",
                "message": f"Probe spec SHA {probe_spec['source_sha256'][:12]} != source SHA {actual_sha[:12]}",
            })

    spin_spans = placer.detect_spin_loop_lines(source_text)
    lang = str(probe_spec.get("language") or "cuda_cpp")
    loop_depths = analyze_loop_nesting(source_text, lang)

    by_role: Dict[int, List[Dict[str, Any]]] = {}
    for pr in probe_spec.get("probes") or []:
        pid = str(pr.get("probe_id") or pr.get("semantic_id"))
        b_ln = int(pr.get("begin_line") or 1)
        e_ln = int(pr.get("end_line") or b_ln)
        rid = int(pr.get("role_id") or 0)
        if pr.get("begin_role_id", rid) == pr.get("end_role_id", rid):
            by_role.setdefault(rid, []).append(pr)

        for s_beg, s_end, s_snip in spin_spans:
            # Violation if probe is strictly inside spin loop body (or begins/ends inside multi-line spin loop)
            if s_end > s_beg and ((s_beg < b_ln <= s_end) or (s_beg <= b_ln and e_ln < s_end and b_ln > s_beg)):
                diagnostics.append({
                    "code": "SPIN_LOOP_PROBE",
                    "severity": "error",
                    "probe_id": pid,
                    "probe_lines": [b_ln, e_ln],
                    "spin_loop_lines": [s_beg, s_end],
                    "message": (
                        f"Probe '{pid}' (lines {b_ln}..{e_ln}) is placed INSIDE polling/spin loop "
                        f"at lines {s_beg}..{s_end} (`{s_snip[:60]}`). Hoist `wait` probe boundaries "
                        f"strictly OUTSIDE lines {s_beg}..{s_end} (e.g. begin_line={s_beg}, end_line={s_end})."
                    ),
                })

    # Check interval partial overlap per warp role (must be disjoint or strictly nested)
    for rid, rprobes in by_role.items():
        ordered = sorted(rprobes, key=lambda x: (int(x.get("begin_line") or 1), -int(x.get("end_line") or 1)))
        for i in range(len(ordered)):
            a = ordered[i]
            ab, ae = int(a["begin_line"]), int(a["end_line"])
            for j in range(i + 1, len(ordered)):
                b = ordered[j]
                bb, be = int(b["begin_line"]), int(b["end_line"])
                if bb > ae:
                    break
                # Overlap exists: valid only if b is strictly nested inside a (`be <= ae`)
                if be > ae:
                    diagnostics.append({
                        "code": "CROSS_OVERLAPPING_PROBES",
                        "severity": "error",
                        "role_id": rid,
                        "probe_a": a.get("probe_id"),
                        "probe_b": b.get("probe_id"),
                        "message": (
                            f"Probes '{a.get('probe_id')}' ({ab}..{ae}) and '{b.get('probe_id')}' ({bb}..{be}) "
                            f"partially cross-overlap on role_id={rid}; intervals on one lane must be disjoint or strictly nested."
                        ),
                    })

    if lang in ("triton", "cutedsl"):
        try:
            py_tree = ast.parse(source_text)
        except SyntaxError:
            py_tree = None
        if py_tree is not None:
            def _dec_kind(fn_node: ast.FunctionDef) -> Optional[str]:
                for d in fn_node.decorator_list:
                    t = d.func if isinstance(d, ast.Call) else d
                    nm = t.attr if isinstance(t, ast.Attribute) else (t.id if isinstance(t, ast.Name) else "")
                    if nm in ("jit", "kernel"):
                        return nm
                return None

            py_fns: List[ast.FunctionDef] = []
            for top in py_tree.body:
                if isinstance(top, ast.FunctionDef) and _dec_kind(top):
                    py_fns.append(top)
                elif isinstance(top, ast.ClassDef):
                    for sub in top.body:
                        if isinstance(sub, ast.FunctionDef) and _dec_kind(sub):
                            py_fns.append(sub)
            fn_by_name = {f.name: f for f in py_fns}
            called_by: Dict[str, List[Tuple[ast.FunctionDef, int, int]]] = {}
            for caller in py_fns:
                caller_kind = _dec_kind(caller)
                for stmt in ast.walk(caller):
                    if isinstance(stmt, ast.Call):
                        callee_nm = (
                            stmt.func.id if isinstance(stmt.func, ast.Name)
                            else (stmt.func.attr if isinstance(stmt.func, ast.Attribute) else None)
                        )
                        if callee_nm and callee_nm in fn_by_name and callee_nm != caller.name:
                            callee_fn = fn_by_name[callee_nm]
                            # In CuTeDSL, @cute.jit host launcher calling @cute.kernel is an entry launcher, NOT a device helper
                            if lang == "cutedsl" and caller_kind == "jit" and _dec_kind(callee_fn) == "kernel":
                                continue
                            called_by.setdefault(callee_nm, []).append((
                                caller,
                                stmt.lineno,
                                getattr(stmt, "end_lineno", stmt.lineno) or stmt.lineno,
                            ))
            for pr in probe_spec.get("probes") or []:
                pid = str(pr.get("probe_id") or pr.get("id") or pr.get("semantic_id"))
                b_ln = int(pr.get("begin_line") or 1)
                e_ln = int(pr.get("end_line") or b_ln)
                for f in py_fns:
                    f_end = getattr(f, "end_lineno", f.lineno) or f.lineno
                    if f.lineno <= b_ln <= f_end and f.name in called_by:
                        caller_fn, c_beg, c_end = called_by[f.name][0]
                        diagnostics.append({
                            "code": "INNER_JIT_HELPER_PROBE",
                            "severity": "error",
                            "probe_id": pid,
                            "helper_function": f.name,
                            "entry_kernel": caller_fn.name,
                            "enclosing_call_lines": [c_beg, c_end],
                            "message": (
                                f"Probe '{pid}' (lines {b_ln}..{e_ln}) is inside inner @{_dec_kind(f)} helper "
                                f"'{f.name}', which has no direct recorder buffer scope. Bracket its call site at "
                                f"lines {c_beg}..{c_end} inside entry kernel '{caller_fn.name}' instead."
                            ),
                        })
                        break
    return diagnostics


# ---------------------------------------------------------------------------
# C++ (CUDA C++ & HIP C++) Zero-Signature-Edit Rule Lowering (`FiScope`)
# ---------------------------------------------------------------------------

_CPP_FUNC_RE = re.compile(
    r"((?:__global__|CUTLASS_GLOBAL|CUTE_GLOBAL|__device__|CUTLASS_DEVICE|CUTE_DEVICE|CUTLASS_HOST_DEVICE|CUTE_HOST_DEVICE|DG_DEVICE)"
    r"(?:\s+__forceinline__|\s+inline|\s+static|\s+__launch_bounds__\s*\([^)]*\)|\s+__cluster_dims__\s*\([^)]*\))*)\s+"
    r"(?:[\w:<>,\s*&]+?)\s+([A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)


def _strip_cpp_comments_keep_len(src: str) -> str:
    out = list(src)
    n = len(src)
    i = 0
    while i < n:
        ch = src[i]
        if ch == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                out[i] = " "
                i += 1
        elif ch == "/" and i + 1 < n and src[i + 1] == "*":
            out[i] = " "
            out[i + 1] = " "
            i += 2
            while i < n:
                if src[i] == "*" and i + 1 < n and src[i + 1] == "/":
                    out[i] = " "
                    out[i + 1] = " "
                    i += 2
                    break
                if src[i] != "\n":
                    out[i] = " "
                i += 1
        elif ch in ('"', "'"):
            q = ch
            out[i] = " "
            i += 1
            while i < n:
                if src[i] == "\\" and i + 1 < n:
                    if src[i] != "\n":
                        out[i] = " "
                    if src[i + 1] != "\n":
                        out[i + 1] = " "
                    i += 2
                    continue
                if src[i] == q:
                    out[i] = " "
                    i += 1
                    break
                if src[i] != "\n":
                    out[i] = " "
                i += 1
        else:
            i += 1
    return "".join(out)


def _find_matching(clean: str, open_pos: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    for i in range(open_pos, len(clean)):
        if clean[i] == open_ch:
            depth += 1
        elif clean[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"Unbalanced {open_ch}{close_ch} at offset {open_pos}")


def _line_offsets(src: str) -> List[int]:
    """Return 0-based byte offset of the start of each 1-based line (index 0 unused)."""
    offs = [0, 0]
    for idx, ch in enumerate(src):
        if ch == "\n":
            offs.append(idx + 1)
    return offs


def _snap_cpp_end_offset(src: str, clean: str, line_end_offset: int, max_offset: int) -> int:
    """Ensure an `end_line` injection lands after a complete `;` or `}` statement boundary."""
    # Check if parens are unbalanced before line_end_offset; if so, advance to closing `);`
    i = line_end_offset
    # Walk backward to start of line
    line_start = src.rfind("\n", 0, max(0, line_end_offset - 1)) + 1
    seg = clean[line_start:line_end_offset]
    stripped = seg.strip()
    if not stripped or stripped.endswith((";", "}")):
        return line_end_offset
    while i < max_offset and clean[i] not in (";", "}"):
        if clean[i] == "(":
            i = _find_matching(clean, i, "(", ")") + 1
        elif clean[i] == "{":
            i = _find_matching(clean, i, "{", "}") + 1
            return i
        else:
            i += 1
    if i < max_offset and clean[i] == ";":
        return i + 1
    return line_end_offset


def _lower_cpp_probe_spec(source_text: str, probe_spec: Mapping[str, Any]) -> str:
    language = str(probe_spec.get("language") or "cuda_cpp")
    header_name = "wp_lite.cuh" if language == "cuda_cpp" else "wp_lite_hip.h"
    paper = bool(probe_spec.get("_paper_endpoints"))
    clean = _strip_cpp_comments_keep_len(source_text)
    loffs = _line_offsets(source_text)
    lines = source_text.splitlines(keepends=True)
    n_lines = len(lines)

    # Discover all __global__ and __device__ functions and their [body_open, body_close] line spans
    funcs: List[Dict[str, Any]] = []
    for m in _CPP_FUNC_RE.finditer(clean):
        qual = m.group(1)
        fname = m.group(2)
        p_open = m.end() - 1
        try:
            p_close = _find_matching(clean, p_open, "(", ")")
        except ValueError:
            continue
        j = p_close + 1
        while j < len(clean) and clean[j].isspace():
            j += 1
        if j >= len(clean) or clean[j] != "{":
            continue  # prototype only
        b_open = j
        b_close = _find_matching(clean, b_open, "{", "}")
        start_ln = source_text.count("\n", 0, m.start()) + 1
        open_ln = source_text.count("\n", 0, b_open) + 1
        close_ln = source_text.count("\n", 0, b_close) + 1
        funcs.append({
            "name": fname,
            "is_global": any(tok in qual for tok in ("__global__", "CUTLASS_GLOBAL", "CUTE_GLOBAL")),
            "start_ln": start_ln,
            "open_ln": open_ln,
            "close_ln": close_ln,
            "b_open": b_open,
            "b_close": b_close,
        })

    if not funcs:
        raise ValueError("No __global__ or __device__ function definition found in C++ source")

    probes = list(probe_spec.get("probes") or [])
    warp_roles = list(probe_spec.get("warp_roles") or [])
    multi_role = len(warp_roles) > 1
    target_kname = str(probe_spec.get("kernel_name") or "").strip()

    # Map each probe to its enclosing function
    probed_funcs: Dict[str, Dict[str, Any]] = {}
    for b_ln in {int(pr[key]) for pr in probes for key in ("begin_line", "end_line")}:
        owner = None
        for fn in funcs:
            if fn["open_ln"] <= b_ln <= fn["close_ln"]:
                owner = fn
                break
        if owner is None:
            # Fallback to target kernel or first __global__
            owner = next((f for f in funcs if f["name"] == target_kname), None) or next((f for f in funcs if f["is_global"]), funcs[0])
        probed_funcs[owner["name"]] = owner

    # Also ensure the root __global__ kernel has FI_KERNEL_SCOPE so OPEN/CLOSE records are always emitted!
    root_kernel = (
        next((f for f in funcs if f["is_global"] and f["name"] == target_kname), None)
        or next((f for f in funcs if f["is_global"] and f["name"] in probed_funcs), None)
        or next((f for f in funcs if f["is_global"] and not any(s in f["name"].lower() for s in ("ref", "init", "verify"))), None)
        or funcs[0]
    )
    probed_funcs[root_kernel["name"]] = root_kernel
    probed_device_subfuncs = [f for f in probed_funcs.values() if not f["is_global"] and f["name"] != root_kernel["name"]]

    edits: List[Tuple[int, int, str]] = []

    # 1. Header include after last #include before first probed function
    first_fn_pos = min(loffs[f["start_ln"]] for f in probed_funcs.values())
    includes = list(re.finditer(r"^\s*#\s*include\s+[<\"][^>\"]+[>\"][^\n]*\n", source_text[:first_fn_pos], re.MULTILINE))
    inc_pos = includes[-1].end() if includes else 0
    op_fam = str(probe_spec.get("operator_family") or "gemm").strip().lower()
    op_ev = STAGE_EVENT_IDS.get(op_fam, 10)
    preamble = f'\n#include "{header_name}"\n'
    edits.append((inc_pos, inc_pos, preamble))

    # 2. Inject FI_KERNEL_SCOPE in root __global__ kernel (after early-return bounds guards if any)
    rk_open = root_kernel["b_open"]
    rk_close = root_kernel["b_close"]
    first_probe_in_rk = min(
        (loffs[int(p["begin_line"])] for p in probes if root_kernel["open_ln"] <= int(p["begin_line"]) <= root_kernel["close_ln"]),
        default=rk_close,
    )
    # Place scope init right after `{` unless there is an early `if (...) return;` before first probe,
    # or role predicates reference kernel-entry local variables (e.g. `warp_idx`, `is_leader_cta`)
    # or the kernel body opens with an arch guard (`#if defined(__CUDA_ARCH__) ... #else ... #endif`).
    scope_insert_pos = rk_open + 1
    root_close_pos = rk_close
    early_ret_region = clean[rk_open + 1:first_probe_in_rk]
    last_early_ret = None
    for rm in re.finditer(r"\bif\s*\([^)]*\)\s*return\s*;", early_ret_region):
        last_early_ret = rk_open + 1 + rm.end()
    if last_early_ret is not None and not paper:
        scope_insert_pos = last_early_ret

    lane_mask = "63u" if language == "hip_cpp" else "31u"
    warp_shift = "6" if language == "hip_cpp" else "5"
    role_guards = {}
    if paper:
        # Kernel-local identifiers may be declared after `{`; advance OPEN past
        # any leading `#if` arch guard and past top-level declarations of
        # identifiers referenced by role predicates (never past first probe).
        warp = "((threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) / warpSize)"
        role_sel = "65535u"
        for role in reversed(warp_roles):
            rid = int(role["role_id"])
            pred = re.sub(r"\bwarp_id\b", warp, str(role.get("predicate") or "true"))
            role_guards[rid] = pred
            role_sel = f"(({pred}) ? {rid}u : {role_sel})"
        _cpp_kw = {
            "true", "false", "and", "or", "not", "threadIdx", "blockIdx",
            "blockDim", "gridDim", "warpSize", "x", "y", "z", "u",
        }
        pred_idents = {
            t for t in re.findall(r"\b[A-Za-z_]\w*\b", role_sel)
            if t not in _cpp_kw and not t.endswith("u")
        }
        # Scan top-level lines in [rk_open+1, first_probe_in_rk) for leading `#if`
        # and local declarations of any predicate identifier.
        rk_open_ln = root_kernel["open_ln"]
        first_probe_ln = min(
            (int(p["begin_line"]) for p in probes if root_kernel["open_ln"] <= int(p["begin_line"]) <= root_kernel["close_ln"]),
            default=root_kernel["close_ln"],
        )
        depth = 0
        leading_pp_depth = 0
        saw_code_stmt = False
        decl_end_pos = scope_insert_pos
        for ln_no in range(rk_open_ln + 1, first_probe_ln):
            raw_ln = lines[ln_no - 1]
            st = raw_ln.strip()
            if not st or st.startswith("//"):
                continue
            if st.startswith("#"):
                if re.match(r"^#\s*if\b", st):
                    if not saw_code_stmt and depth == 0:
                        leading_pp_depth += 1
                        decl_end_pos = max(decl_end_pos, loffs[ln_no] + len(raw_ln) + 1)
                elif re.match(r"^#\s*endif\b", st) and leading_pp_depth > 0 and not saw_code_stmt:
                    leading_pp_depth -= 1
                continue
            saw_code_stmt = True
            c_ln = clean[loffs[ln_no]:loffs[ln_no] + len(raw_ln)]
            if depth == 0 and pred_idents:
                for ident in pred_idents:
                    if re.search(
                        r"(?:^|[;{}])\s*(?:const\s+|constexpr\s+|static\s+|volatile\s+)*"
                        r"(?:auto|bool|int|unsigned|uint\d+_t|int\d+_t|size_t)\s+(?:unsigned\s+|int\s+)?"
                        + re.escape(ident) + r"\s*(?:=|;|\()",
                        c_ln,
                    ):
                        # Advance past statement semicolon
                        semi = clean.find(";", loffs[ln_no], first_probe_in_rk)
                        if semi != -1:
                            nl = source_text.find("\n", semi + 1)
                            decl_end_pos = max(decl_end_pos, (nl + 1) if nl != -1 else (semi + 1))
            depth += c_ln.count("{") - c_ln.count("}")
            if depth < 0:
                break
        if decl_end_pos <= first_probe_in_rk:
            scope_insert_pos = decl_end_pos
        # If OPEN landed inside a kernel-wide `#if ... [#else ...] #endif` arch guard,
        # place WP_CLOSE() right before the matching top-level `#else` or `#endif`
        # so host/fallback compilation never sees an undeclared recorder context.
        if leading_pp_depth > 0:
            pp_lvl = 0
            for ln_no in range(rk_open_ln + 1, root_kernel["close_ln"] + 1):
                st = lines[ln_no - 1].strip()
                if re.match(r"^#\s*if\b", st):
                    pp_lvl += 1
                elif re.match(r"^#\s*(?:else|elif|endif)\b", st):
                    if pp_lvl == leading_pp_depth and loffs[ln_no] > scope_insert_pos:
                        root_close_pos = loffs[ln_no]
                        break
                    if re.match(r"^#\s*endif\b", st) and pp_lvl > 0:
                        pp_lvl -= 1
        scope_code = f"\n    const uint32_t wp_role_ = {role_sel};\n"
        recorder = "fi_recorder()" if language == "cuda_cpp" else "(fi_lane() == 0u)"
        scope_code += f"    WP_OPEN_BY({recorder} && wp_role_ != 65535u, wp_role_);\n"
    elif multi_role:
        role_expr_parts = []
        for r in warp_roles:
            pred = str(r.get("predicate") or "").strip()
            rid = int(r.get("role_id") or 0)
            if pred and pred.lower() != "true":
                role_expr_parts.append((pred, rid))
        default_rid = int(warp_roles[-1].get("role_id") or 0)
        role_sel = str(default_rid) + "u"
        for pred, rid in reversed(role_expr_parts):
            role_sel = f"(({pred}) ? {rid}u : {role_sel})"
        scope_code = (
            f"\n    const uint32_t fi_lin_tid_ = (uint32_t)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));\n"
            f"    const uint32_t warp_id = fi_lin_tid_ >> {warp_shift};\n"
            f"    (void)warp_id;\n"
            f"    const uint32_t fi_tile_id_ = (uint32_t)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));\n"
            f"    const uint32_t fi_role_id_ = {role_sel};\n"
            f"    FI_KERNEL_SCOPE_BY((fi_lin_tid_ & {lane_mask}) == 0u, fi_role_id_);\n"
            f"    FI_B({op_ev}u, fi_tile_id_);\n"
        )
    else:
        r0 = int(warp_roles[0].get("role_id") or 0) if warp_roles else 0
        scope_code = (
            f"\n    const uint32_t fi_tile_id_ = (uint32_t)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));\n"
            f"    FI_KERNEL_SCOPE({r0}u);\n"
            f"    FI_B({op_ev}u, fi_tile_id_);\n"
        )
    edits.append((scope_insert_pos, scope_insert_pos, scope_code))

    # Close operator task envelope before any later `return;` and before `rk_close` (`~FiScope()` handles `FI_CLOSE()` automatically!)
    for rm in re.finditer(r"\breturn\s*;", clean[scope_insert_pos:rk_close]):
        rpos = scope_insert_pos + rm.start()
        end_call = "WP_CLOSE();" if paper else f"FI_E({op_ev}u, fi_tile_id_);"
        edits.append((rpos, scope_insert_pos + rm.end(), f"{{ {end_call} return; }}"))
    root_end = "\n    WP_CLOSE();\n" if paper else f"\n    FI_E({op_ev}u, fi_tile_id_);\n"

    # 3. Inject FI_DEVICE_SCOPE() in any probed __device__ subfunctions (0 signature edits!)
    #    and wrap calls to them with FI_SYNC_OUT() / FI_SYNC_IN()!
    for sub in probed_device_subfuncs:
        sub_open = sub["b_open"] + 1
        sub_init = (
            "\n    const uint32_t fi_tile_id_ = (uint32_t)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));\n"
            "    (void)fi_tile_id_;\n"
            "    FI_DEVICE_SCOPE();\n"
        )
        edits.append((sub_open, sub_open, sub_init))
        call_re = re.compile(r"\b" + re.escape(sub["name"]) + r"\s*\(")
        for caller in probed_funcs.values():
            c_beg, c_end = caller["b_open"] + 1, caller["b_close"]
            for cm in call_re.finditer(clean[c_beg:c_end]):
                cpos = c_beg + cm.start()
                # Find statement semicolon
                p_op = c_beg + cm.end() - 1
                p_cl = _find_matching(clean, p_op, "(", ")")
                semi = clean.find(";", p_cl, c_end)
                if semi != -1:
                    # Walk back to start of statement on this line
                    ls = source_text.rfind("\n", c_beg, cpos) + 1
                    if paper and caller["is_global"]:
                        edits.append((ls, ls, "WP_SYNC_CURSOR_OUT(); "))
                        edits.append((semi + 1, semi + 1, " WP_SYNC_CURSOR_IN();"))
                    else:
                        edits.append((ls, ls, "FI_SYNC_OUT(); "))
                        edits.append((semi + 1, semi + 1, " FI_SYNC_IN();"))

    # 4. Inject stage probes `[begin_line, end_line]`
    stage_edits = []
    for pr in probes:
        b_ln = max(1, min(n_lines, int(pr["begin_line"])))
        e_ln = max(1, min(n_lines, int(pr["end_line"])))
        sem = str(pr.get("semantic_id") or "compute").lower()
        ev_id = int(pr.get("event_id") or STAGE_EVENT_IDS.get(sem, 4))
        aux = str(pr.get("aux_expr") or "fi_tile_id_").strip() or "fi_tile_id_"

        # Owner function close boundary
        owner_close = next(
            (f["b_close"] for f in funcs if f["open_ln"] <= e_ln <= f["close_ln"]),
            len(source_text),
        )
        owner_open = next((f["b_open"] + 1 for f in funcs if f["open_ln"] <= b_ln <= f["close_ln"]), 0)
        b_off = max(loffs[b_ln], owner_open)
        # A `#pragma` (e.g. unroll / ivdep) must stay attached to the statement
        # that follows it. When begin_line is that statement, inserting WP_B at
        # the line start splits the pragma from its loop and clang rejects the
        # source ("expected a for, while, or do-while loop to follow '#pragma
        # unroll'"). Hoist the insertion point above the contiguous pragma chain.
        pl = b_ln
        while pl > 1 and lines[pl - 2].strip().startswith("#pragma"):
            pl -= 1
        if pl < b_ln:
            b_off = max(loffs[pl], owner_open)
        raw_e_off = loffs[e_ln] + len(lines[e_ln - 1])
        # If a same-role synchronous probe begins inside a block/loop and end_line
        # points at that enclosing block's closing '}', insert WP_E BEFORE the '}'
        # so begin and end remain in the same lexical scope (variables & trip count).
        if (
            int(pr.get("begin_role_id", 0)) == int(pr.get("end_role_id", 0))
            and e_ln > b_ln
            and lines[e_ln - 1].strip().startswith("}")
            and clean[b_off:raw_e_off].count("{") < clean[b_off:raw_e_off].count("}")
        ):
            e_off = min(owner_close, loffs[e_ln])
        else:
            e_off = min(owner_close, _snap_cpp_end_offset(source_text, clean, raw_e_off, owner_close))

        indent_m = re.match(r"[ \t]*", lines[b_ln - 1])
        indent = indent_m.group(0) if indent_m else "    "

        if paper:
            slot_expr = str(pr["slot"]).strip() if "slot" in pr and pr["slot"] is not None else ""
            if slot_expr:
                # If begin_line itself declares a variable referenced by `slot` (e.g.
                # `const uint32_t tma_stage_idx = s % 2;`), move WP_B right after that
                # declaration so `slot` is in scope at WP_B (or fall back to 0u if b_ln == e_ln).
                slot_idents = re.findall(r"\b[A-Za-z_]\w*\b", slot_expr)
                b_line_clean = clean[loffs[b_ln]:loffs[b_ln] + len(lines[b_ln - 1])]
                declared_on_begin = any(
                    re.search(
                        r"(?:const\s+|constexpr\s+|auto\s+|int\s+|uint\d+_t\s+|int\d+_t\s+|size_t\s+)"
                        r"(?:(?:const|unsigned|int)\s+)*" + re.escape(si) + r"\s*=",
                        b_line_clean,
                    )
                    for si in slot_idents
                )
                if declared_on_begin:
                    semi = clean.find(";", loffs[b_ln], e_off)
                    if b_ln < e_ln and semi != -1 and (semi + 1) < e_off:
                        nl = source_text.find("\n", semi + 1)
                        b_off = (nl + 1) if (nl != -1 and nl + 1 < e_off) else (semi + 1)
                    else:
                        slot_expr = ""
            aux = f"(uint32_t)({slot_expr})" if slot_expr else "0u"
            bg = role_guards[int(pr["begin_role_id"])]
            eg = role_guards[int(pr["end_role_id"])]
            b_stmt = f"\n{indent}if ({bg}) {{ WP_B({ev_id}u, {aux}); }}\n"
            e_stmt = f"\n{indent}if ({eg}) {{ WP_E({ev_id}u, {aux}); }}\n"
        elif multi_role and pr.get("role_id") is not None:
            rid = int(pr["role_id"])
            # Check if probe is already inside the role's dedicated if/else branch
            role_obj = next((r for r in warp_roles if int(r.get("role_id") or -1) == rid), None)
            in_dedicated_branch = bool(
                role_obj
                and role_obj.get("begin_line")
                and role_obj.get("end_line")
                and int(role_obj["begin_line"]) <= b_ln
                and e_ln <= int(role_obj["end_line"])
            )
            if in_dedicated_branch:
                b_stmt = f"{indent}FI_B({ev_id}u, {aux});\n"
                e_stmt = f"\n{indent}FI_E({ev_id}u, {aux});\n"
            else:
                b_stmt = f"{indent}FI_B_PRED(fi_role_id_ == {rid}u, {ev_id}u, {aux});\n"
                e_stmt = f"\n{indent}FI_E_PRED(fi_role_id_ == {rid}u, {ev_id}u, {aux});\n"
        else:
            b_stmt = f"{indent}FI_B({ev_id}u, {aux});\n"
            e_stmt = f"\n{indent}FI_E({ev_id}u, {aux});\n"

        # SM100 field lesson: if begin_line sits directly under a braceless
        # `if (...)` / `else` / `for (...)` / `while (...)` header, wrap the
        # instrumented body in `{ ... }` so stmt and WP_E stay guarded by the header.
        prev_idx = b_ln - 2
        while prev_idx >= 0 and not lines[prev_idx].strip():
            prev_idx -= 1
        prev_clean = re.sub(r"//[^\n]*$", "", lines[prev_idx]).strip() if prev_idx >= 0 else ""
        braceless_parent = bool(
            int(pr.get("begin_role_id", 0)) == int(pr.get("end_role_id", 0))
            and prev_clean
            and not prev_clean.endswith(("{", ";", "}"))
            and re.match(r"^(?:(?:else\s+)?if|for|while)\s*\(.*\)$|^else$", prev_clean)
        )
        if braceless_parent:
            b_stmt = f"\n{indent}{{\n" + b_stmt.lstrip("\n")
            e_stmt = e_stmt + f"{indent}}}\n"

        stage_edits.append((b_off, 1, -e_off, b_stmt))
        stage_edits.append((e_off, 0, -b_off, e_stmt))

    for pos, _, _, statement in sorted(stage_edits):
        edits.append((pos, pos, statement))
    edits.append((root_close_pos, root_close_pos, root_end))

    indexed = list(enumerate(edits))
    indexed.sort(key=lambda item: (item[1][0], item[1][1], item[0]))
    out: List[str] = []
    cur = 0
    for _, (eb, ee, repl) in indexed:
        if eb < cur:
            raise RuntimeError(f"Overlapping C++ rule_engine edit at offset {eb} < {cur}")
        out.append(source_text[cur:eb])
        out.append(repl)
        cur = ee
    out.append(source_text[cur:])
    return "".join(out)


# ---------------------------------------------------------------------------
# Python AST (Triton NV/AMD & CuTeDSL) Zero-Signature-Edit Proxy Lowering
# ---------------------------------------------------------------------------

def _collect_ast_stmts(node: ast.AST) -> List[ast.stmt]:
    stmts: List[ast.stmt] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.stmt) and not isinstance(sub, ast.FunctionDef):
            stmts.append(sub)
    return stmts


def _snap_python_probe_lines(fn_node: ast.FunctionDef, b_ln: int, e_ln: int) -> Tuple[int, int, str]:
    """Snap `[b_ln, e_ln]` to clean AST statement boundaries inside `fn_node`."""
    stmts = _collect_ast_stmts(fn_node)
    # Find deepest statement list containing b_ln..e_ln so begin and end share the exact same indentation block!
    best_body: Optional[List[ast.stmt]] = None

    def _visit_block(body: List[ast.stmt]) -> None:
        nonlocal best_body
        if not body:
            return
        blk_start = body[0].lineno
        blk_end = getattr(body[-1], "end_lineno", body[-1].lineno) or body[-1].lineno
        if blk_start <= b_ln and e_ln <= blk_end:
            best_body = body
        for st in body:
            for attr in ("body", "orelse", "finalbody"):
                sub_body = getattr(st, attr, None)
                if isinstance(sub_body, list) and sub_body and isinstance(sub_body[0], ast.stmt):
                    _visit_block(sub_body)

    _visit_block(fn_node.body)
    target_body = best_body or fn_node.body

    beg_stmt = target_body[0]
    for st in target_body:
        st_end = getattr(st, "end_lineno", st.lineno) or st.lineno
        if st_end >= b_ln:
            beg_stmt = st
            break

    end_stmt = beg_stmt
    for st in target_body:
        if st.lineno <= e_ln:
            end_stmt = st

    snapped_b = beg_stmt.lineno
    snapped_e = getattr(end_stmt, "end_lineno", end_stmt.lineno) or end_stmt.lineno
    if snapped_e < snapped_b:
        snapped_e = snapped_b
    indent = " " * beg_stmt.col_offset
    return snapped_b, snapped_e, indent


def _patch_python_signature(fn_lines: List[str], extra_params: str) -> List[str]:
    """Append `extra_params` before closing `):` of a Python `def ...(...):` header."""
    out = list(fn_lines)
    # Locate closing `):` before function body
    sig_end_idx = 0
    for i, ln in enumerate(out):
        if "):" in re.sub(r"#.*$", "", ln) or ") ->" in re.sub(r"#.*$", "", ln):
            sig_end_idx = i
            break
    line = out[sig_end_idx]
    close_pos = line.rfind("):")
    if close_pos == -1:
        close_pos = line.rfind(")")
    prefix_all = "".join(out[:sig_end_idx]) + line[:close_pos]
    clean_prefix = re.sub(r"#[^\n]*", "", prefix_all).rstrip()
    has_comma_or_open = clean_prefix.endswith(",") or clean_prefix.endswith("(")
    if line[:close_pos].strip() == "" and not has_comma_or_open and sig_end_idx > 0:
        prev = sig_end_idx - 1
        while prev >= 0 and not re.sub(r"#.*$", "", out[prev]).strip():
            prev -= 1
        code_part = re.sub(r"(\s*(?:#.*)?\r?\n)$", "", out[prev])
        out[prev] = code_part + "," + out[prev][len(code_part):]
        sep = "    "
    else:
        sep = "" if clean_prefix.endswith("(") else (" " if clean_prefix.endswith(",") else ", ")
    out[sig_end_idx] = line[:close_pos] + sep + extra_params + line[close_pos:]
    return out


def _endpoint_statements(fn, pr):
    """Each endpoint has its own AST boundary and indentation (warp branches can differ)."""
    if pr.get("begin_role_id") == pr.get("end_role_id"):
        b, e, ind = _snap_python_probe_lines(fn, int(pr["begin_line"]), int(pr["end_line"]))
        return b, e, ind, ind
    b, _, bi = _snap_python_probe_lines(fn, pr["begin_line"], pr["begin_line"])
    _, e, ei = _snap_python_probe_lines(fn, pr["end_line"], pr["end_line"])
    # A hoisted endpoint on the final body line must follow the whole loop.
    for stmt in ast.walk(fn):
        if isinstance(stmt, (ast.While, ast.For)) and (stmt.lineno, stmt.end_lineno) == (b, e):
            ei = " " * stmt.col_offset
            break
    return b, e, bi, ei


def _specialized_roles(spec):
    roles = spec.get("warp_roles") or []
    return bool(spec.get("_paper_endpoints") and (
        len(roles) > 1 or any(r["predicate"].lower() != "true" for r in roles)
    ))


def _lower_triton_probe_spec(source_text: str, probe_spec: Mapping[str, Any]) -> str:
    platform = str(probe_spec.get("platform") or ("amd" if probe_spec.get("card") == "mi308x" else "nv"))
    tree = ast.parse(source_text)
    all_lines = source_text.splitlines(keepends=True)
    target_kname = str(probe_spec.get("kernel_name") or "").strip()

    jit_fns: List[ast.FunctionDef] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            is_jit = any(
                (isinstance(d, ast.Attribute) and d.attr == "jit")
                or (isinstance(d, ast.Name) and d.id == "jit")
                or (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "jit")
                for d in node.decorator_list
            )
            if is_jit:
                jit_fns.append(node)
    if not jit_fns:
        raise ValueError("No @triton.jit function found in Python source")

    probe_lns = [int(p.get("begin_line") or 0) for p in (probe_spec.get("probes") or [])]
    fn = next((f for f in jit_fns if f.name == target_kname), None)
    if fn is None:
        fn = next(
            (f for f in jit_fns if any(f.lineno <= pl <= (getattr(f, "end_lineno", f.lineno) or f.lineno) for pl in probe_lns)),
            jit_fns[0],
        )
    orig_name = fn.name
    dec_start = fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno
    fn_end = getattr(fn, "end_lineno", fn.body[-1].lineno) or fn.body[-1].lineno

    # Untouched L0 copy renamed to `_fi_l0_<orig_name>` (retains @triton.autotune / @triton.heuristics)
    l0_lines = list(all_lines[dec_start - 1:fn_end])
    def_rel_idx = fn.lineno - dec_start
    l0_lines[def_rel_idx] = re.sub(
        r"\bdef\s+" + re.escape(orig_name) + r"\b",
        f"def _fi_l0_{orig_name}",
        l0_lines[def_rel_idx],
        count=1,
    )

    # Instrumented copy `_fi_inst_<orig_name>`: strip `@triton.autotune` (proxy reuses L0's winning config
    # without running 20+ do_bench launches on the probe buffer) and add `do_not_specialize` on buffer pointers.
    inst_lines = list(all_lines[dec_start - 1:fn_end])
    inst_lines[def_rel_idx] = re.sub(
        r"\bdef\s+" + re.escape(orig_name) + r"\b",
        f"def _fi_inst_{orig_name}",
        inst_lines[def_rel_idx],
        count=1,
    )
    dec_indent = " " * fn.col_offset
    for dec in fn.decorator_list:
        d_target = dec.func if isinstance(dec, ast.Call) else dec
        d_name = d_target.attr if isinstance(d_target, ast.Attribute) else (d_target.id if isinstance(d_target, ast.Name) else "")
        d_beg = dec.lineno - dec_start
        d_end = (getattr(dec, "end_lineno", dec.lineno) or dec.lineno) - dec_start
        if d_name == "autotune":
            for r_i in range(d_beg, d_end + 1):
                inst_lines[r_i] = ""
        elif d_name == "jit":
            dec_text = ast.unparse(dec)
            if isinstance(dec, ast.Call):
                dns_kw = next((kw for kw in dec.keywords if kw.arg == "do_not_specialize"), None)
                if dns_kw is not None and isinstance(dns_kw.value, (ast.List, ast.Tuple)):
                    existing = [ast.unparse(el) for el in dns_kw.value.elts]
                    for extra_nm in ("'_fi_rec_ptr'", "'_fi_ovf_ptr'"):
                        if extra_nm not in existing and extra_nm.replace("'", '"') not in existing:
                            existing.append(extra_nm)
                    other_args = [ast.unparse(a) for a in dec.args] + [
                        f"{kw.arg}={ast.unparse(kw.value)}" for kw in dec.keywords if kw.arg != "do_not_specialize"
                    ]
                    other_args.append(f"do_not_specialize=[{', '.join(existing)}]")
                    new_dec = f"{dec_indent}@{ast.unparse(dec.func)}({', '.join(other_args)})\n"
                elif dns_kw is None:
                    all_args = [ast.unparse(a) for a in dec.args] + [
                        f"{kw.arg}={ast.unparse(kw.value)}" for kw in dec.keywords
                    ] + ["do_not_specialize=['_fi_rec_ptr', '_fi_ovf_ptr']"]
                    new_dec = f"{dec_indent}@{ast.unparse(dec.func)}({', '.join(all_args)})\n"
                else:
                    new_dec = f"{dec_indent}@{dec_text}\n"
            else:
                new_dec = f"{dec_indent}@{dec_text}(do_not_specialize=['_fi_rec_ptr', '_fi_ovf_ptr'])\n"
            inst_lines[d_beg] = new_dec
            for r_i in range(d_beg + 1, d_end + 1):
                inst_lines[r_i] = ""

    body_indent = " " * fn.body[0].col_offset
    op_fam = str(probe_spec.get("operator_family") or "gemm").strip().lower()
    op_ev = STAGE_EVENT_IDS.get(op_fam, 10)
    triton_dir = str(SKILL_ROOT / "recorders" / "triton")

    insert_before: Dict[int, List[str]] = {}
    insert_after: Dict[int, List[str]] = {}

    if platform != "amd":
        import_block = (
            "\n# --- WeaveProfiler §3.2 Fast Probe Injection (NVIDIA Triton Proxy) ---\n"
            "import os as _fi_os, sys as _fi_sys\n"
            f"for _fi_d in (_fi_os.environ.get('WEAVEPROFILER_TRITON_DIR', ''), {triton_dir!r}):\n"
            "    if _fi_d and _fi_os.path.isdir(_fi_d) and _fi_d not in _fi_sys.path:\n"
            "        _fi_sys.path.insert(0, _fi_d)\n"
            "from wp_triton import (\n"
            "    FiCfg as FiAutoCfg, FiBuffer as FiAutoBuffer,\n"
            "    fi_attach, fi_get_attached, _FiTritonProxy,\n"
            "    fi_open as _fi_open, fi_b as _fi_b, fi_e as _fi_e, fi_close as _fi_close,\n"
            ")\n"
        )
        extra_params = "_fi_rec_ptr=None, _fi_ovf_ptr=None, _FI_CFG: tl.constexpr=None"
        open_block = (
            f"{body_indent}_fi_cta = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0) + tl.program_id(2) * tl.num_programs(0) * tl.num_programs(1)\n"
            f"{body_indent}_fi_n = _fi_open(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _FI_CFG, 0)\n"
            f"{body_indent}_fi_n = _fi_b(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG, {op_ev}, _fi_cta)\n"
        )
        close_block = (
            f"{body_indent}_fi_n = _fi_e(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG, {op_ev}, _fi_cta)\n"
            f"{body_indent}_fi_n = _fi_close(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG)\n"
        )
    else:
        import_block = (
            "\n# --- WeaveProfiler §3.2 Fast Probe Injection (AMD MI308X Triton Proxy) ---\n"
            "import os as _fi_os, sys as _fi_sys\n"
            f"for _fi_d in (_fi_os.environ.get('WEAVEPROFILER_TRITON_DIR', ''), {triton_dir!r}):\n"
            "    if _fi_d and _fi_os.path.isdir(_fi_d) and _fi_d not in _fi_sys.path:\n"
            "        _fi_sys.path.insert(0, _fi_d)\n"
            "from wp_triton_amd import (\n"
            "    FiPackedBuffer as FiAutoBuffer, fi_attach, fi_get_attached, _FiTritonProxy,\n"
            "    fi_open_16b as _fi_open_16b, fi_b_16b as _fi_b_16b, fi_e_16b as _fi_e_16b, fi_close_16b as _fi_close_16b,\n"
            ")\n"
        )
        extra_params = (
            "_fi_rec_ptr=None, _fi_ovf_ptr=None, _FI_LEVEL: tl.constexpr = 2, _FI_GRAN: tl.constexpr = 1, "
            "_FI_CAP: tl.constexpr = 256, _FI_WPC: tl.constexpr = 1, _FI_THREADS: tl.constexpr = 256, _FI_TIMER: tl.constexpr = 1"
        )
        open_block = (
            f"{body_indent}_fi_cta = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0) + tl.program_id(2) * tl.num_programs(0) * tl.num_programs(1)\n"
            f"{body_indent}_fi_base, _fi_sel, _fi_n = _fi_open_16b(_fi_rec_ptr, _fi_ovf_ptr, ROLE=0, GRAN=_FI_GRAN, CAP=_FI_CAP, WPC=_FI_WPC, THREADS=_FI_THREADS, TIMER=_FI_TIMER)\n"
            f"{body_indent}_fi_n = _fi_b_16b(_fi_rec_ptr, _fi_ovf_ptr, _fi_base, _fi_sel, _fi_n, EVENT={op_ev}, aux=_fi_cta, CAP=_FI_CAP, THREADS=_FI_THREADS, TIMER=_FI_TIMER)\n"
        )
        close_block = (
            f"{body_indent}_fi_n = _fi_e_16b(_fi_rec_ptr, _fi_ovf_ptr, _fi_base, _fi_sel, _fi_n, EVENT={op_ev}, aux=_fi_cta, CAP=_FI_CAP, THREADS=_FI_THREADS, TIMER=_FI_TIMER)\n"
            f"{body_indent}_fi_n = _fi_close_16b(_fi_rec_ptr, _fi_ovf_ptr, _fi_base, _fi_sel, _fi_n, CAP=_FI_CAP, THREADS=_FI_THREADS, TIMER=_FI_TIMER)\n"
        )

    paper = probe_spec.get("_paper_endpoints")
    specialized = _specialized_roles(probe_spec)
    if paper:
        nv = platform != "amd"
        module = "wp_triton" if nv else "wp_triton_amd"
        if nv:
            import_block += f"from {module} import WP_B, WP_E, WP_TRANS, _WpTritonProxy, wp_attach\n"
        else:
            import_block += f"from {module} import WP_B, WP_E, _WpTritonProxy, wp_attach\n"
        # The paper records the requested stages; L0 and L_inst share the caller's launch signature.
        open_block = open_block.splitlines(keepends=True)[0]
        if specialized:
            import_block += f"from {module} import wp_thread_ids\n"
            threads = "_FI_CFG.threads" if nv else "_FI_THREADS"
            open_block += f"{body_indent}_wp_tid = wp_thread_ids({threads})\n"
            open_block += f"{body_indent}_wp_warp_id = _wp_tid // {32 if nv else 64}\n"
            open_block += f"{body_indent}_wp_role = tl.full(({threads},), -1, tl.int32)\n"
            for role in reversed(probe_spec["warp_roles"]):
                pred = re.sub(r"\bwarp_id\b", "_wp_warp_id", role["predicate"])
                open_block += f"{body_indent}_wp_role = tl.where({pred}, {role['role_id']}, _wp_role)\n"
        if specialized and nv:
            import_block += "from wp_triton import wp_open_by, wp_close_by, wp_b_by, wp_e_by\n"
            open_block += f"{body_indent}_wp_lane0 = (_wp_tid % 32) == 0\n"
            for role in probe_spec["warp_roles"]:
                rid = role["role_id"]
                open_block += (f"{body_indent}_fi_open_n = wp_open_by(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, "
                               f"_FI_CFG, ((_wp_role == {rid}) & _wp_lane0).to(tl.int32), {rid})\n")
            open_block += f"{body_indent}_fi_n = tl.full((_FI_CFG.threads,), 0, tl.int32) + _fi_open_n\n"
            close_block = (f"{body_indent}_fi_n = wp_close_by(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, "
                           "_FI_CFG, ((_wp_role >= 0) & _wp_lane0).to(tl.int32))\n")
        elif nv:
            rid = probe_spec["warp_roles"][0]["role_id"]
            open_block += f"{body_indent}_fi_n = _fi_open(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _FI_CFG, {rid})\n"
            close_block = close_block.splitlines(keepends=True)[-1]
        else:
            rid = probe_spec["warp_roles"][0]["role_id"]
            role_arg = ", role_value=_wp_role" if specialized else ""
            open_block += (f"{body_indent}_fi_base, _fi_sel, _fi_n = _fi_open_16b(_fi_rec_ptr, _fi_ovf_ptr, "
                           f"ROLE={rid}, GRAN=_FI_GRAN, CAP=_FI_CAP, WPC=_FI_WPC, THREADS=_FI_THREADS, "
                           f"TIMER=_FI_TIMER{role_arg})\n")
            close_block = close_block.splitlines(keepends=True)[-1]
    insert_before.setdefault(fn.body[0].lineno, []).append(open_block)
    for pr in probe_spec.get("probes") or []:
        sem = str(pr.get("semantic_id") or "compute").lower()
        ev_id = int(pr.get("event_id") or STAGE_EVENT_IDS.get(sem, 4))
        sb, se, ind = _snap_python_probe_lines(fn, int(pr["begin_line"]), int(pr["end_line"]))
        if paper:
            sb, se, bi, ei = _endpoint_statements(fn, pr)
            aux = str(pr.get("slot", 0))
            endpoints = []
            for suffix, ind, rid in (("B", bi, pr["begin_role_id"]), ("E", ei, pr["end_role_id"])):
                pred = f"(_wp_role == {rid})"
                if nv:
                    args = f"_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG, {ev_id}, {aux}"
                    if specialized:
                        updated = f"wp_{suffix.lower()}_by({args}, ({pred} & _wp_lane0).to(tl.int32))"
                    else:
                        updated = f"WP_{suffix}({args})"
                else:
                    selected = f"(_fi_sel & {pred})" if specialized else "_fi_sel"
                    updated = (f"WP_{suffix}(_fi_rec_ptr, _fi_ovf_ptr, _fi_base, {selected}, _fi_n, EVENT={ev_id}, "
                               f"aux={aux}, CAP=_FI_CAP, THREADS=_FI_THREADS, TIMER=_FI_TIMER)")
                if specialized:
                    updated = f"tl.where({pred}, {updated}, _fi_n)"
                endpoints.append(f"{ind}_fi_n = {updated}\n")
            b_call, e_call = endpoints
        elif platform != "amd":
            b_call = f"{ind}_fi_n = _fi_b(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG, {ev_id}, _fi_cta)\n"
            e_call = f"{ind}_fi_n = _fi_e(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG, {ev_id}, _fi_cta)\n"
        else:
            b_call = f"{ind}_fi_n = _fi_b_16b(_fi_rec_ptr, _fi_ovf_ptr, _fi_base, _fi_sel, _fi_n, EVENT={ev_id}, aux=_fi_cta, CAP=_FI_CAP, THREADS=_FI_THREADS, TIMER=_FI_TIMER)\n"
            e_call = f"{ind}_fi_n = _fi_e_16b(_fi_rec_ptr, _fi_ovf_ptr, _fi_base, _fi_sel, _fi_n, EVENT={ev_id}, aux=_fi_cta, CAP=_FI_CAP, THREADS=_FI_THREADS, TIMER=_FI_TIMER)\n"
        insert_before.setdefault(sb, []).append(b_call)
        insert_after.setdefault(se, []).append(e_call)

    # Fuse contiguous same-indent NVIDIA Triton WP_E(prev) + WP_B(next) into 1 single WP_TRANS barrier!
    if paper and platform != "amd" and not specialized:
        e_pat = re.compile(r"^([ \t]*)_fi_n = WP_E\(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG, (\d+), ([^)]+)\)\n$")
        b_pat = re.compile(r"^([ \t]*)_fi_n = WP_B\(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG, (\d+), ([^)]+)\)\n$")
        for ln_after in sorted(list(insert_after.keys())):
            aft_list = insert_after.get(ln_after) or []
            if not aft_list:
                continue
            m_e = e_pat.match(aft_list[-1])
            if not m_e:
                continue
            # Walk forward over blank/comment lines to find next executable source line
            nxt = ln_after + 1
            while nxt <= fn_end and (not all_lines[nxt - 1].strip() or all_lines[nxt - 1].lstrip().startswith("#")):
                nxt += 1
            bef_list = insert_before.get(nxt) or []
            if not bef_list:
                continue
            m_b = b_pat.match(bef_list[0])
            if not m_b or m_e.group(1) != m_b.group(1) or m_e.group(3) != m_b.group(3):
                continue
            ind_t, prev_ev, aux_t = m_e.group(1), m_e.group(2), m_e.group(3)
            next_ev = m_b.group(2)
            aft_list[-1] = f"{ind_t}_fi_n = WP_TRANS(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG, {prev_ev}, {next_ev}, {aux_t})\n"
            bef_list.pop(0)

    # Close before early `return` statements if any, or at end of body
    last_stmt = fn.body[-1]
    if isinstance(last_stmt, ast.Return):
        insert_before.setdefault(last_stmt.lineno, []).append(close_block)
    else:
        insert_after.setdefault(fn_end, []).append(close_block)

    # Reconstruct `inst_lines` with body insertions first, then patch signature
    rebuilt_inst: List[str] = []
    for offset_idx, raw_ln in enumerate(inst_lines):
        abs_ln = dec_start + offset_idx
        for chunk in insert_before.get(abs_ln, ()):
            rebuilt_inst.append(chunk)
        rebuilt_inst.append(raw_ln)
        for chunk in insert_after.get(abs_ln, ()):
            rebuilt_inst.append(chunk)

    # Patch signature only on the `def _fi_inst_...:` header slice
    header_len = fn.body[0].lineno - dec_start
    patched_hdr = _patch_python_signature(rebuilt_inst[:header_len], extra_params)
    rebuilt_inst = patched_hdr + rebuilt_inst[header_len:]

    proxy = "_WpTritonProxy" if paper else "_FiTritonProxy"
    proxy_bind = f"\n{orig_name} = {proxy}(_fi_l0_{orig_name}, _fi_inst_{orig_name}, fi_get_attached)\n"
    final_src = (
        "".join(all_lines[:dec_start - 1])
        + import_block
        + "\n"
        + "".join(l0_lines)
        + "\n"
        + "".join(rebuilt_inst)
        + proxy_bind
        + "".join(all_lines[fn_end:])
    )
    ast.parse(final_src)
    return final_src


def _lower_cutedsl_probe_spec(source_text: str, probe_spec: Mapping[str, Any]) -> str:
    tree = ast.parse(source_text)
    all_lines = source_text.splitlines(keepends=True)
    target_kname = str(probe_spec.get("kernel_name") or "").strip()

    def _is_cute(fn_node: ast.FunctionDef) -> bool:
        return any(
            (isinstance(d, ast.Attribute) and d.attr in ("kernel", "jit"))
            or (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr in ("kernel", "jit"))
            or (isinstance(d, ast.Name) and d.id in ("kernel", "jit"))
            for d in fn_node.decorator_list
        )

    cute_targets: List[Tuple[Optional[ast.ClassDef], ast.FunctionDef]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if _is_cute(node):
                cute_targets.append((None, node))
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and _is_cute(sub):
                    cute_targets.append((node, sub))

    if not cute_targets:
        raise ValueError("No @cute.kernel or @cute.jit function found in CuTeDSL source")

    # Select probed kernel function/method
    target_pair = next((p for p in cute_targets if p[1].name == target_kname), None)
    if target_pair is None:
        p_lines = [int(p["begin_line"]) for p in (probe_spec.get("probes") or [])]
        for p in cute_targets:
            fe = getattr(p[1], "end_lineno", p[1].lineno) or p[1].lineno
            if any(p[1].lineno <= ln <= fe for ln in p_lines):
                target_pair = p
                break
        if target_pair is None:
            target_pair = cute_targets[0]

    parent_cls, kfn = target_pair
    is_method = parent_cls is not None

    orig_name = kfn.name
    dec_start = kfn.decorator_list[0].lineno if kfn.decorator_list else kfn.lineno
    fn_end = getattr(kfn, "end_lineno", kfn.body[-1].lineno) or kfn.body[-1].lineno

    l0_lines = list(all_lines[dec_start - 1:fn_end])
    def_rel_idx = kfn.lineno - dec_start
    l0_lines[def_rel_idx] = re.sub(
        r"\bdef\s+" + re.escape(orig_name) + r"\b",
        f"def _fi_l0_{orig_name}",
        l0_lines[def_rel_idx],
        count=1,
    )

    inst_lines = list(all_lines[dec_start - 1:fn_end])
    inst_lines[def_rel_idx] = re.sub(
        r"\bdef\s+" + re.escape(orig_name) + r"\b",
        f"def _fi_inst_{orig_name}",
        inst_lines[def_rel_idx],
        count=1,
    )

    body_indent = " " * kfn.body[0].col_offset
    method_indent = " " * (parent_cls.body[0].col_offset if is_method else 0)
    op_fam = str(probe_spec.get("operator_family") or "gemm").strip().lower()
    op_ev = STAGE_EVENT_IDS.get(op_fam, 10)
    paper = probe_spec.get("_paper_endpoints")
    gran_int = 0 if paper or str(probe_spec.get("granularity")) == "warp" else 1
    cute_dir = str(SKILL_ROOT / "recorders" / "cutedsl")

    import_block = (
        "\n# --- WeaveProfiler §3.2 Fast Probe Injection (CuTeDSL Proxy) ---\n"
        "import os as _fi_os, sys as _fi_sys\n"
        f"for _fi_d in (_fi_os.environ.get('WEAVEPROFILER_CUTEDSL_DIR', ''), {cute_dir!r}):\n"
        "    if _fi_d and _fi_os.path.isdir(_fi_d) and _fi_d not in _fi_sys.path:\n"
        "        _fi_sys.path.insert(0, _fi_d)\n"
        "from wp_cute import (\n"
        "    FiBuffer as FiAutoBuffer, fi_attach, fi_get_attached, _FiCuteProxy, _FiCuteDescriptor,\n"
        "    fi_open as _fi_open, fi_b as _fi_b, fi_e as _fi_e, fi_close as _fi_close,\n"
        ")\n"
    )

    insert_before: Dict[int, List[str]] = {}
    insert_after: Dict[int, List[str]] = {}

    open_block = (
        f"{body_indent}_fi_bidx, _fi_bidy, _fi_bidz = cute.arch.block_idx()\n"
        f"{body_indent}_fi_gidx, _fi_gidy, _ = cute.arch.grid_dim()\n"
        f"{body_indent}_fi_tile_id = cutlass.Int32(_fi_bidx + _fi_bidy * _fi_gidx + _fi_bidz * _fi_gidx * _fi_gidy)\n"
        f"{body_indent}_fi_bx, _fi_by, _fi_bz = cute.arch.block_dim()\n"
        f"{body_indent}_fi_wpc = cutlass.Int32((_fi_bx * _fi_by * _fi_bz + 31) // 32)\n"
        f"{body_indent}_fi_ctx = _fi_open(_fi_rec, _fi_ovf, cutlass.Int32(0), _fi_wpc, fi_level=2, fi_gran={gran_int})\n"
        f"{body_indent}_fi_ctx = _fi_b(_fi_ctx, cutlass.Int32({op_ev}), _fi_tile_id, fi_level=2)\n"
    )
    close_block = (
        f"{body_indent}_fi_ctx = _fi_e(_fi_ctx, cutlass.Int32({op_ev}), _fi_tile_id, fi_level=2)\n"
        f"{body_indent}_fi_ctx = _fi_close(_fi_ctx, fi_level=2)\n"
    )

    specialized = _specialized_roles(probe_spec)
    if paper:
        import_block += "from wp_cute import WP_B, WP_E, _WpCuteProxy, _WpCuteDescriptor, wp_attach\n"
        open_block = "".join(open_block.splitlines(keepends=True)[:-2])
        if specialized:
            open_block += f"{body_indent}_wp_role = cutlass.Int32(-1)\n"
            for role in reversed(probe_spec["warp_roles"]):
                pred = re.sub(r"\bwarp_id\b", "cute.arch.warp_idx()", role["predicate"])
                open_block += f"{body_indent}if {pred}:\n{body_indent}    _wp_role = cutlass.Int32({role['role_id']})\n"
        else:
            open_block += f"{body_indent}_wp_role = cutlass.Int32({probe_spec['warp_roles'][0]['role_id']})\n"
        open_block += (f"{body_indent}_fi_ctx = _fi_open(_fi_rec, _fi_ovf, _wp_role, _fi_wpc, "
                       f"fi_level=2, fi_gran={gran_int})\n")
        close_block = close_block.splitlines(keepends=True)[-1]
    insert_before.setdefault(kfn.body[0].lineno, []).append(open_block)
    for pr in probe_spec.get("probes") or []:
        sem = str(pr.get("semantic_id") or "compute").lower()
        ev_id = int(pr.get("event_id") or STAGE_EVENT_IDS.get(sem, 4))
        if paper:
            sb, se, bi, ei = _endpoint_statements(kfn, pr)
            for op, side, where, line, ind in (("B", "begin", insert_before, sb, bi), ("E", "end", insert_after, se, ei)):
                chunk = ""
                if specialized:
                    chunk += f"{ind}if _wp_role == {pr[side + '_role_id']}:\n"
                    ind += "    "
                chunk += f"{ind}_fi_ctx = WP_{op}(_fi_ctx, cutlass.Int32({ev_id}), cutlass.Int32({pr.get('slot', 0)}), fi_level=2)\n"
                where.setdefault(line, []).append(chunk)
            continue
        sb, se, ind = _snap_python_probe_lines(kfn, int(pr["begin_line"]), int(pr["end_line"]))
        insert_before.setdefault(sb, []).append(
            f"{ind}_fi_ctx = _fi_b(_fi_ctx, cutlass.Int32({ev_id}), _fi_tile_id, fi_level=2)\n"
        )
        insert_after.setdefault(se, []).append(
            f"{ind}_fi_ctx = _fi_e(_fi_ctx, cutlass.Int32({ev_id}), _fi_tile_id, fi_level=2)\n"
        )

    last_stmt = kfn.body[-1]
    if isinstance(last_stmt, ast.Return):
        insert_before.setdefault(last_stmt.lineno, []).append(close_block)
    else:
        insert_after.setdefault(fn_end, []).append(close_block)

    rebuilt_inst: List[str] = []
    for offset_idx, raw_ln in enumerate(inst_lines):
        abs_ln = dec_start + offset_idx
        for chunk in insert_before.get(abs_ln, ()):
            rebuilt_inst.append(chunk)
        rebuilt_inst.append(raw_ln)
        for chunk in insert_after.get(abs_ln, ()):
            rebuilt_inst.append(chunk)

    header_len = kfn.body[0].lineno - dec_start
    patched_hdr = _patch_python_signature(rebuilt_inst[:header_len], "_fi_rec=None, _fi_ovf=None")
    rebuilt_inst = patched_hdr + rebuilt_inst[header_len:]

    def _cute_dec_kind(fn_node: ast.FunctionDef) -> str:
        for d in fn_node.decorator_list:
            t = d.func if isinstance(d, ast.Call) else d
            nm = t.attr if isinstance(t, ast.Attribute) else (t.id if isinstance(t, ast.Name) else "")
            if nm in ("kernel", "jit"):
                return nm
        return ""

    def _rewrite_host_jit_launcher(hfn: ast.FunctionDef) -> Optional[Tuple[int, int, str]]:
        # Thread `_fi_rec=None, _fi_ovf=None` through enclosing `@cute.jit` host launchers calling `@cute.kernel`
        # so MLIR tracing never bakes a stale Python global buffer pointer into its JIT cache.
        calls: List[ast.Call] = []
        for node in ast.walk(hfn):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == orig_name:
                    calls.append(node)
                elif (isinstance(node.func, ast.Attribute) and node.func.attr == orig_name
                      and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"):
                    calls.append(node)
        if not calls:
            return None
        h_dec_start = hfn.decorator_list[0].lineno if hfn.decorator_list else hfn.lineno
        h_end = getattr(hfn, "end_lineno", hfn.body[-1].lineno) or hfn.body[-1].lineno
        h_name = hfn.name
        h_raw = "".join(all_lines[h_dec_start - 1:h_end])
        h_loffs = _line_offsets(h_raw)

        def _byte_pos(abs_line: int, col: int) -> int:
            return h_loffs[abs_line - h_dec_start + 1] + col

        # Build L0 host launcher (`_fi_l0_<h_name>` calling `_fi_l0_<orig_name>`)
        l0_edits: List[Tuple[int, int, str]] = []
        def_pos = _byte_pos(hfn.lineno, 0)
        def_line = all_lines[hfn.lineno - 1]
        m_def = re.search(r"\bdef\s+(" + re.escape(h_name) + r")\b", def_line)
        if m_def:
            l0_edits.append((def_pos + m_def.start(1), def_pos + m_def.end(1), f"_fi_l0_{h_name}"))
        inst_edits: List[Tuple[int, int, str]] = []
        if m_def:
            inst_edits.append((def_pos + m_def.start(1), def_pos + m_def.end(1), f"_fi_inst_{h_name}"))
        for cnode in calls:
            if isinstance(cnode.func, ast.Name):
                f_beg = _byte_pos(cnode.func.lineno, cnode.func.col_offset)
                f_end = f_beg + len(orig_name)
            else:
                # `self.<orig_name>` -> end of attribute
                f_end = _byte_pos(cnode.func.end_lineno, cnode.func.end_col_offset)
                f_beg = f_end - len(orig_name)
            l0_edits.append((f_beg, f_end, f"_fi_l0_{orig_name}"))
            inst_edits.append((f_beg, f_end, f"_fi_inst_{orig_name}"))
            close_paren = _byte_pos(cnode.end_lineno, cnode.end_col_offset) - 1
            sep = ", " if (cnode.args or cnode.keywords) else ""
            inst_edits.append((close_paren, close_paren, f"{sep}_fi_rec=_fi_rec, _fi_ovf=_fi_ovf"))

        def _apply_str_edits(base_s: str, ed_list: List[Tuple[int, int, str]]) -> str:
            cur = base_s
            for s_i, e_i, rep in sorted(ed_list, key=lambda x: (x[0], x[1]), reverse=True):
                cur = cur[:s_i] + rep + cur[e_i:]
            return cur

        h_l0_txt = _apply_str_edits(h_raw, l0_edits)
        h_inst_raw = _apply_str_edits(h_raw, inst_edits).splitlines(keepends=True)
        h_hdr_len = hfn.body[0].lineno - h_dec_start
        h_inst_txt = "".join(_patch_python_signature(h_inst_raw[:h_hdr_len], "_fi_rec=None, _fi_ovf=None") + h_inst_raw[h_hdr_len:])
        wrap_cls = ("_WpCuteDescriptor" if paper else "_FiCuteDescriptor") if is_method else ("_WpCuteProxy" if paper else "_FiCuteProxy")
        h_bind = f"\n{method_indent}{h_name} = {wrap_cls}(_fi_l0_{h_name}, _fi_inst_{h_name}, fi_get_attached)\n"
        return (h_dec_start, h_end, h_l0_txt + "\n" + h_inst_txt + h_bind)

    tail_lines = list(all_lines[fn_end:])
    if _cute_dec_kind(kfn) == "kernel":
        siblings = [
            f for p_c, f in cute_targets
            if p_c is parent_cls and f is not kfn and _cute_dec_kind(f) == "jit" and f.lineno > fn_end
        ]
        for hfn in sorted(siblings, key=lambda f: f.lineno, reverse=True):
            rewritten_h = _rewrite_host_jit_launcher(hfn)
            if rewritten_h is not None:
                h_beg, h_end, h_replacement = rewritten_h
                rel_b = h_beg - 1 - fn_end
                rel_e = h_end - fn_end
                tail_lines[rel_b:rel_e] = [h_replacement]

    if is_method:
        descriptor = "_WpCuteDescriptor" if paper else "_FiCuteDescriptor"
        proxy_bind = f"\n{method_indent}{orig_name} = {descriptor}(_fi_l0_{orig_name}, _fi_inst_{orig_name}, fi_get_attached)\n"
        final_src = (
            "".join(all_lines[:parent_cls.lineno - 1])
            + import_block
            + "".join(all_lines[parent_cls.lineno - 1:dec_start - 1])
            + "".join(l0_lines)
            + "\n"
            + "".join(rebuilt_inst)
            + proxy_bind
            + "".join(tail_lines)
        )
    else:
        proxy = "_WpCuteProxy" if paper else "_FiCuteProxy"
        proxy_bind = f"\n{orig_name} = {proxy}(_fi_l0_{orig_name}, _fi_inst_{orig_name}, fi_get_attached)\n"
        final_src = (
            "".join(all_lines[:dec_start - 1])
            + import_block
            + "\n"
            + "".join(l0_lines)
            + "\n"
            + "".join(rebuilt_inst)
            + proxy_bind
            + "".join(tail_lines)
        )
    ast.parse(final_src)
    return final_src


def inject_probes(
    source_text: str,
    probe_spec: Mapping[str, Any],
    *,
    check_sha: bool = True,
    fail_on_static_errors: bool = True,
) -> Dict[str, Any]:
    """Deterministic §3.2 (2a) rule lowering across all 4 languages in milliseconds."""
    t0 = time.perf_counter()
    if "roles" in probe_spec:
        from weaveprof.contract import normalize_probe_spec
        paper = "v1_roles" not in probe_spec or any(
            p.get("begin_role_id") != p.get("end_role_id") for p in probe_spec.get("probes", ())
        )
        probe_spec = dict(normalize_probe_spec(probe_spec, source_text=source_text), _paper_endpoints=paper)
    static_diags = validate_probe_spec_static(source_text, probe_spec, check_sha=check_sha)
    if fail_on_static_errors and any(d.get("severity") == "error" for d in static_diags):
        return {
            "ok": False,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "diagnostics": static_diags,
            "instrumented_source": None,
            "names": None,
        }

    lang = str(probe_spec.get("language") or "cuda_cpp").lower()
    if lang in ("cuda_cpp", "hip_cpp"):
        inst_src = _lower_cpp_probe_spec(source_text, probe_spec)
    elif lang == "triton":
        inst_src = _lower_triton_probe_spec(source_text, probe_spec)
    elif lang == "cutedsl":
        inst_src = _lower_cutedsl_probe_spec(source_text, probe_spec)
    else:
        raise ValueError(f"Unsupported language in probe_spec: {lang!r}")

    names_doc = build_names_from_probe_spec(probe_spec)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "ok": True,
        "elapsed_ms": elapsed_ms,
        "diagnostics": static_diags,
        "instrumented_source": inst_src,
        "instrumented_sha256": sha256_text(inst_src),
        "names": names_doc,
        "zero_signature_edits": True,
    }


# ---------------------------------------------------------------------------
# §3.2 (2b) CPU-Side Parallel Static Preflight & Batched Diagnostics
# ---------------------------------------------------------------------------

def _parse_nvcc_ptxas_resources(log: str, kernel: str) -> Dict[str, int]:
    """Read one kernel's ptxas entry, excluding helper and unrelated kernel properties."""
    entries = re.split(r"Compiling entry function '([^']+)'", log)
    matched = [(name, block) for name, block in zip(entries[1::2], entries[2::2]) if kernel in name]
    if len(matched) != 1:
        raise ValueError(f"Expected one ptxas entry matching {kernel!r}, found {len(matched)}")
    name, block = matched[0]
    properties = re.split(r"Function properties for\s+['\"]?([^\s'\"]+)", block)
    section = next((body for symbol, body in zip(properties[1::2], properties[2::2]) if symbol == name), "")
    spills = re.search(r"(\d+)\s+bytes spill stores,\s*(\d+)\s+bytes spill loads", section)
    regs = re.search(r"Used\s+(\d+)\s+registers", section)
    if spills is None or regs is None:
        raise ValueError(f"Missing ptxas register or spill byte counts for {name}")
    spill_stores, spill_loads = map(int, spills.groups())
    smem = re.search(r"(\d+)\s+bytes smem", section)
    return {
        "registers": int(regs[1]),
        "spill_stores_bytes": spill_stores,
        "spill_loads_bytes": spill_loads,
        "spill_total_bytes": spill_stores + spill_loads,
        "smem_bytes": int(smem[1]) if smem else 0,
    }


def _parse_amd_asm_resources(asm_text: str, kernel: str) -> Dict[str, int]:
    """Sum compiler-annotated spill/reload instruction widths in the target kernel."""
    descriptors = re.findall(r"(?ms)^\s*\.amdhsa_kernel\s+(\S+)\n(.*?)^\s*\.end_amdhsa_kernel", asm_text)
    matched = [(name, meta) for name, meta in descriptors if kernel in name]
    if len(matched) != 1:
        raise ValueError(f"Expected one GCN kernel descriptor matching {kernel!r}, found {len(matched)}")
    name, meta = matched[0]
    fields = dict(re.findall(r"\.amdhsa_(\w+)\s+(\d+)", meta))
    required = ("next_free_vgpr", "next_free_sgpr", "private_segment_fixed_size", "group_segment_fixed_size")
    if any(field not in fields for field in required):
        raise ValueError(f"Incomplete GCN resource descriptor for {name}")
    symbol = re.escape(name)
    body = re.search(rf"(?ms)^{symbol}:.*?^\s*\.size\s+{symbol},", asm_text)
    if not body:
        raise ValueError(f"Missing GCN instructions for {name}")
    annotations = re.findall(r";\s*(\d+)-byte (?:Folded )?(Spill|Reload)\b", body[0], re.I)
    scratch = int(fields["private_segment_fixed_size"])
    if scratch and not annotations:
        raise ValueError(f"Nonzero private memory for {name} without quantified spill/reload evidence")
    stores = sum(int(width) for width, kind in annotations if kind.lower() == "spill")
    loads = sum(int(width) for width, kind in annotations if kind.lower() == "reload")
    return {
        "registers": int(fields["next_free_vgpr"]),
        "sgprs": int(fields["next_free_sgpr"]),
        "private_segment_bytes": scratch,
        "spill_stores_bytes": stores,
        "spill_loads_bytes": loads,
        "spill_total_bytes": stores + loads,
        "smem_bytes": int(fields["group_segment_fixed_size"]),
    }


def _parallel_build(baseline, instrumented):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = (pool.submit(baseline), pool.submit(instrumented))
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except ValueError as exc:
                results.append({"ok": False, "err": str(exc)})
    return results


def run_cpu_parallel_preflight(
    original_source_path: Path,
    instrumented_source_path: Path,
    probe_spec: Mapping[str, Any],
    work_dir: Path,
    *,
    cache_dir: Optional[Path] = None,
    min_sass_sim: float = 0.85,
) -> Dict[str, Any]:
    """Compile both builds and check SASS/GCN changes, added spills and added SMEM."""
    t0 = time.perf_counter()
    work_dir.mkdir(parents=True, exist_ok=True)
    orig_text = original_source_path.read_text(encoding="utf-8")
    inst_text = instrumented_source_path.read_text(encoding="utf-8")
    lang = str(probe_spec.get("language") or "cuda_cpp")
    card = str(probe_spec.get("card") or "h20")
    arch = str(probe_spec.get("arch") or ("gfx942" if card == "mi308x" else "sm_90a"))
    kname = str(probe_spec.get("kernel_name") or "").strip()

    diagnostics = validate_probe_spec_static(orig_text, probe_spec, check_sha=False)

    # Python syntax check for Triton / CuTeDSL
    if lang in ("triton", "cutedsl"):
        try:
            ast.parse(inst_text)
        except SyntaxError as exc:
            diagnostics.append({
                "code": "PYTHON_SYNTAX_ERROR",
                "severity": "error",
                "line": exc.lineno,
                "message": f"Generated Python syntax error at line {exc.lineno}: {exc.msg}",
            })
        return {
            "ok": not any(d.get("severity") == "error" for d in diagnostics),
            "elapsed_s": round(time.perf_counter() - t0, 4),
            "language": lang,
            "diagnostics": diagnostics,
            "l0_cached": False,
            "resources": None,
            "binsim": None,
        }

    incr_mod = _load_mod("weaveprof/pipeline/instrument/incremental_reuse.py", "wp_incr_reuse_preflight")
    cache = incr_mod.BuildArtifactCache(cache_dir or (work_dir / "_l0_build_cache"))
    gran_int = 0 if (len(probe_spec.get("warp_roles") or []) > 1 or probe_spec.get("granularity") == "warp") else 1

    if lang == "cuda_cpp":
        nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        if not os.path.exists(nvcc):
            return {
                "ok": not any(d.get("severity") == "error" for d in diagnostics),
                "elapsed_s": round(time.perf_counter() - t0, 4),
                "compiler_available": False,
                "diagnostics": diagnostics,
            }
        inc_dir = str(SKILL_ROOT / "recorders" / "cuda_cpp")
        l0_key = cache.compute_key(source_text=orig_text, card=card, arch=arch, language=lang,
                                  variant="L0_cubin_v2", extra_flags=kname)

        def _compile_l0() -> Dict[str, Any]:
            hit = cache.lookup(l0_key)
            if hit and hit.get("binary_path"):
                return {"ok": True, "cached": True, "binary": Path(hit["binary_path"]), "res": hit["resources"]}
            out_cubin = work_dir / "l0.cubin"
            cmd = [nvcc, "-O3", "-std=c++17", f"-arch={arch}", "-Xptxas=-v", "-cubin",
                   f"-I{inc_dir}", "-x", "cu", str(original_source_path), "-o", str(out_cubin)]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if p.returncode != 0:
                return {"ok": False, "cached": False, "err": (p.stderr or p.stdout)[-1200:]}
            res = _parse_nvcc_ptxas_resources((p.stderr or "") + "\n" + (p.stdout or ""), kname)
            cache.store(l0_key, binary_path=out_cubin, metadata={"resources": res})
            return {"ok": True, "cached": False, "binary": out_cubin, "res": res}

        def _compile_inst() -> Dict[str, Any]:
            out_cubin = work_dir / "fi_inst.cubin"
            cmd = [nvcc, "-O3", "-std=c++17", f"-arch={arch}", "-Xptxas=-v", "-cubin",
                   "-DFI_LEVEL=2", "-DFI_TIMER=1", f"-DFI_GRAN={gran_int}",
                   f"-I{inc_dir}", "-x", "cu", str(instrumented_source_path), "-o", str(out_cubin)]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if p.returncode != 0:
                return {"ok": False, "err": (p.stderr or p.stdout)[-1200:]}
            res = _parse_nvcc_ptxas_resources((p.stderr or "") + "\n" + (p.stdout or ""), kname)
            return {"ok": True, "binary": out_cubin, "res": res}

        r0, r1 = _parallel_build(_compile_l0, _compile_inst)

        if not r0["ok"] or not r1["ok"]:
            diagnostics.append({
                "code": "COMPILE_ERROR",
                "severity": "error",
                "message": (r1.get("err") if not r1["ok"] else r0.get("err")) or "nvcc failed",
            })
            return {
                "ok": False,
                "elapsed_s": round(time.perf_counter() - t0, 4),
                "compiler_available": True,
                "l0_cached": bool(r0.get("cached")),
                "diagnostics": diagnostics,
            }

        binsim_mod = _load_mod("weaveprof/pipeline/validate/tools/fi_binsim.py", "wp_fi_binsim_nv")
        binsim_res = binsim_mod.analyze(
            r0["binary"], r1["binary"], kname or ".*", arch=arch, work_dir=work_dir / "disasm"
        )
        sim_strict = float(binsim_res["sim_strict"])
        sim_opcode = float(binsim_res["sim_opcode"])

    else:
        # HIP C++ parallel compile (`hipcc -S` emits exact AMDGCN `.s` with `.amdhsa_*` resource metadata!)
        hipcc = shutil.which("hipcc") or "/usr/bin/hipcc"
        if not os.path.exists(hipcc):
            return {
                "ok": not any(d.get("severity") == "error" for d in diagnostics),
                "elapsed_s": round(time.perf_counter() - t0, 4),
                "compiler_available": False,
                "diagnostics": diagnostics,
            }
        inc_dir = str(SKILL_ROOT / "recorders" / "hip_cpp")
        l0_key = cache.compute_key(source_text=orig_text, card=card, arch=arch, language=lang,
                                  variant="L0_amdgcn_s_v2", extra_flags=kname)

        def _compile_hip_l0() -> Dict[str, Any]:
            hit = cache.lookup(l0_key)
            if hit and hit.get("binary_path"):
                return {"ok": True, "cached": True, "asm": Path(hit["binary_path"]), "res": hit["resources"]}
            out_s = work_dir / "l0.s"
            cmd = [hipcc, "-O3", "-std=c++17", "--cuda-device-only", f"--offload-arch={arch}", "-S", "-fverbose-asm",
                   f"-I{inc_dir}", "-x", "hip", str(original_source_path), "-o", str(out_s)]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if p.returncode != 0:
                return {"ok": False, "cached": False, "err": (p.stderr or p.stdout)[-1200:]}
            res = _parse_amd_asm_resources(out_s.read_text(encoding="utf-8", errors="replace"), kname)
            cache.store(l0_key, binary_path=out_s, metadata={"resources": res})
            return {"ok": True, "cached": False, "asm": out_s, "res": res}

        def _compile_hip_inst() -> Dict[str, Any]:
            out_s = work_dir / "fi_inst.s"
            cmd = [hipcc, "-O3", "-std=c++17", "--cuda-device-only", f"--offload-arch={arch}", "-S", "-fverbose-asm",
                   "-DFI_LEVEL=2", "-DFI_TIMER=1", f"-DFI_GRAN={gran_int}",
                   f"-I{inc_dir}", "-x", "hip", str(instrumented_source_path), "-o", str(out_s)]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if p.returncode != 0:
                return {"ok": False, "err": (p.stderr or p.stdout)[-1200:]}
            res = _parse_amd_asm_resources(out_s.read_text(encoding="utf-8", errors="replace"), kname)
            return {"ok": True, "asm": out_s, "res": res}

        r0, r1 = _parallel_build(_compile_hip_l0, _compile_hip_inst)

        if not r0["ok"] or not r1["ok"]:
            diagnostics.append({
                "code": "COMPILE_ERROR",
                "severity": "error",
                "message": (r1.get("err") if not r1["ok"] else r0.get("err")) or "hipcc failed",
            })
            return {
                "ok": False,
                "elapsed_s": round(time.perf_counter() - t0, 4),
                "compiler_available": True,
                "l0_cached": bool(r0.get("cached")),
                "diagnostics": diagnostics,
            }

        binsim_out = work_dir / "binsim_hip.json"
        p_bs = subprocess.run(
            [sys.executable, "-B", str(SKILL_ROOT / "weaveprof/pipeline/validate" / "tools" / "fi_binsim_hip.py"),
             "--l0", str(r0["asm"]), "--l2", str(r1["asm"]), "--kernel", kname or ".*",
             "--isa", "amdgcn", "-o", str(binsim_out)],
            capture_output=True, text=True, timeout=120,
        )
        binsim_res = json.loads(binsim_out.read_text(encoding="utf-8")) if binsim_out.is_file() else {}
        if p_bs.returncode or "assembly_change_rate" not in binsim_res:
            diagnostics.append({"code": "ASSEMBLY_COMPARISON_FAILED", "severity": "error",
                                "message": (p_bs.stderr or p_bs.stdout)[-1200:]})
            return {"ok": False, "compiler_available": True, "diagnostics": diagnostics,
                    "elapsed_s": round(time.perf_counter() - t0, 4)}
        sim_strict = float(binsim_res["sim_strict"])
        sim_opcode = float(binsim_res["sim_opcode"])

    assembly_change_rate = binsim_res["assembly_change_rate"]
    res0, res1 = r0["res"], r1["res"]
    delta_reg = res1["registers"] - res0["registers"]
    delta_spill = res1["spill_total_bytes"] - res0["spill_total_bytes"]
    delta_smem = res1["smem_bytes"] - res0["smem_bytes"]

    loop_depths = analyze_loop_nesting(orig_text, lang)
    inner_probes = [
        p.get("probe_id")
        for p in (probe_spec.get("probes") or [])
        if loop_depths.get(int(p.get("begin_line") or 1), 0) >= 2
        or bool(p.get("in_inner_loop"))
    ]

    if delta_spill > 0:
        diagnostics.append({
            "code": "REGISTER_SPILL_INTRODUCED",
            "severity": "error",
            "delta_spill_bytes": delta_spill,
            "offending_probes": inner_probes,
            "message": (
                f"Instrumentation introduced {delta_spill} B of register spill (L0={res0['spill_total_bytes']} B -> "
                f"L_inst={res1['spill_total_bytes']} B). Hoist/merge inner-loop probes {inner_probes} outward to enclosing stage boundaries."
            ),
        })
    if delta_smem > 0:
        diagnostics.append({
            "code": "ADDED_SHARED_MEMORY_NONZERO",
            "severity": "error",
            "delta_smem_bytes": delta_smem,
            "message": f"Instrumentation added {delta_smem} B of shared memory (invariant requires 0 B added SMEM).",
        })
    if 1 - assembly_change_rate < min_sass_sim:
        diagnostics.append({
            "code": "SASS_SIMILARITY_LOW",
            "severity": "error",
            "assembly_change_rate": assembly_change_rate,
            "sim_opcode": sim_opcode,
            "sim_strict": sim_strict,
            "offending_probes": inner_probes,
            "message": (
                f"Binary instruction sequence similarity dropped to {1 - assembly_change_rate:.3f} (< {min_sass_sim:.2f}), "
                f"indicating a compiler loop-unrolling cliff around inner-loop probes {inner_probes}. "
                "Apply Scheme A Probe Coarsening: hoist/merge inner-loop probes outward to enclosing stage boundaries."
            ),
        })

    return {
        "ok": not any(d.get("severity") == "error" for d in diagnostics),
        "elapsed_s": round(time.perf_counter() - t0, 4),
        "compiler_available": True,
        "l0_cached": bool(r0.get("cached")),
        "resources": {
            "l0": res0,
            "inst": res1,
            "delta_reg": delta_reg,
            "delta_spill_bytes": delta_spill,
            "delta_smem_bytes": delta_smem,
        },
        "binsim": {
            "sim_strict": sim_strict,
            "sim_opcode": sim_opcode,
            "assembly_change_rate": assembly_change_rate,
        },
        "diagnostics": diagnostics,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="§3.2 Fast Probe Injection — Deterministic Rule Engine & CPU Preflight")
    ap.add_argument("--source", required=True, help="Uninstrumented kernel source path")
    ap.add_argument("--spec", required=True, help="JSON Probe Spec path (`probe_spec.json`)")
    ap.add_argument("--out-source", required=True, help="Output instrumented source path")
    ap.add_argument("--out-names", help="Output canonical `names.json` path")
    ap.add_argument("--preflight-dir", help="Optional directory to run parallel CPU compile + SASS/GCN preflight")
    ap.add_argument("--cache-dir", help="Optional L0 build cache directory")
    ap.add_argument("--ignore-sha", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    src_path = Path(args.source).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()
    src_text = src_path.read_text(encoding="utf-8")
    probe_spec = json.loads(spec_path.read_text(encoding="utf-8"))

    res = inject_probes(src_text, probe_spec, check_sha=(not args.ignore_sha))
    if not res["ok"]:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 2

    out_src = Path(args.out_source).expanduser().resolve()
    out_src.parent.mkdir(parents=True, exist_ok=True)
    out_src.write_bytes(res["instrumented_source"].replace("\r\n", "\n").encode("utf-8"))

    out_names = Path(args.out_names).expanduser().resolve() if args.out_names else out_src.with_suffix(".names.json")
    out_names.write_text(json.dumps(res["names"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    preflight_report = None
    if args.preflight_dir:
        preflight_report = run_cpu_parallel_preflight(
            src_path,
            out_src,
            probe_spec,
            Path(args.preflight_dir).expanduser().resolve(),
            cache_dir=Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None,
        )

    summary = {
        "ok": True if preflight_report is None else preflight_report["ok"],
        "inject_elapsed_ms": res["elapsed_ms"],
        "instrumented_source": str(out_src),
        "names_json": str(out_names),
        "zero_signature_edits": True,
        "preflight": preflight_report,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
