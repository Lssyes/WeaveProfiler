#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automatic cross-language / cross-hardware GEMM semantic planner & instrumenter.

Given an uninstrumented GEMM kernel source file in CUDA C++ (.cu), HIP C++ (.hip/.cpp),
Triton (.py with @triton.jit), or CuTeDSL (.py with @cute.kernel), this module:
  1. Detects language and resolves target hardware profile (h20 / b300 / mi308x);
  2. Performs deterministic AST / brace-balanced structural analysis (zero manual
     selection.json authoring) to locate:
       - kernel entry & bounds guards,
       - K-reduction loop,
       - load issue stage,
       - sync / async-wait stage (explicitly separated from MMA so software MMA spans
         correlate with hardware Tensor Core / Matrix Core activity),
       - MMA / Tensor Core / Matrix Core compute stage,
       - post-MMA barrier stage,
       - epilogue & global store stage;
  3. Emits:
       - instrumented kernel source (supporting compile-time FI_LEVEL=0/1/2 and
         FI_SAMPLE=0 full-K audit, FI_SAMPLE=1 first-tile fast mode, FI_SAMPLE=S strided),
       - canonical names.json (aligned with knowledge/semantic_ids.json),
       - auto_plan.json (auditable line ranges, phases, and recipe binding).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SKILL_ROOT = Path(__file__).resolve().parents[1]

# Canonical semantic event IDs matching knowledge/semantic_ids.json
SEMANTIC_EVENTS: Dict[str, int] = {
    "load": 1,
    "wait": 2,
    "mma": 3,
    "compute": 4,
    "epilogue": 5,
    "store": 6,
    "barrier": 7,
    "gemm": 10,
}

CARD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "h20": {"platform": "nv", "arch": "sm_90a", "warp_size": 32},
    "b300": {"platform": "nv", "arch": "sm_103a", "warp_size": 32},
    "mi308x": {"platform": "amd", "arch": "gfx942", "warp_size": 64},
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_language(path: Path, text: str) -> str:
    suffix = path.suffix.lower()
    if suffix == ".cu" or suffix == ".cuh":
        return "cuda_cpp"
    if suffix in (".hip", ".hpp", ".cpp", ".cc", ".cxx"):
        if "hip_runtime" in text or "__builtin_amdgcn" in text or "hipLaunchKernel" in text or suffix == ".hip":
            return "hip_cpp"
        if "cuda_runtime" in text or "mma.h" in text or "wgmma" in text or "tcgen05" in text:
            return "cuda_cpp"
        return "hip_cpp"
    if suffix == ".py":
        if "cutlass.cute" in text or "@cute.kernel" in text or "@cute.jit" in text:
            return "cutedsl"
        if "triton" in text or "@triton.jit" in text:
            return "triton"
    raise ValueError(f"Cannot determine kernel language for {path}")


def build_names_doc(
    phases_used: Sequence[str],
    *,
    card: str,
    arch: str,
    language: str,
    kernel_name: str,
    warp_specialized: bool = False,
) -> Dict[str, Any]:
    events = {}
    for name in phases_used:
        ev_id = SEMANTIC_EVENTS[name]
        events[str(ev_id)] = {
            "name": name,
            "semantic_id": name,
            "layer": "task" if name == "gemm" else "phase",
        }
    doc: Dict[str, Any] = {
        "events": events,
        "roles": (
            {"0": "cta_worker", "1": "Producer_DMA", "2": "Consumer_MMA"}
            if warp_specialized
            else {"0": "cta_worker", "1": "producer", "2": "consumer"}
        ),
        "meta": {
            "generator": "weaveprofiler/auto_gemm_plan/v2",
            "card": card,
            "arch": arch,
            "language": language,
            "kernel": kernel_name,
            "warp_specialized": warp_specialized,
        },
    }
    if warp_specialized:
        doc["group"] = "sm"
    return doc


# ---------------------------------------------------------------------------
# C++ (CUDA C++ & HIP C++) Structural Analyzer & Instrumenter
# ---------------------------------------------------------------------------

CPP_KERNEL_RE = re.compile(
    r"__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s+)?void\s+([A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)

MMA_PATTERNS = re.compile(
    r"\b(mma_sync|wgmma|tcgen05|__builtin_amdgcn_mfma\w*|mma\.sync|cute::gemm|fmaf)\b"
)
WAIT_PATTERNS = re.compile(
    r"\b(__pipeline_wait_prior|cp_async_wait\w*|wait_group|mbarrier_wait\w*|(?:\w*_)?try_wait\w*|(?:\w*_)?mbar_wait\w*|s_waitcnt)\b"
)
SYNC_PATTERNS = re.compile(r"\b(__syncthreads|__builtin_amdgcn_s_barrier|cluster_sync)\s*\(")
LOAD_PATTERNS = re.compile(
    r"\b(__pipeline_memcpy_async|cp_async\w*|memcpy_async|tma_load\w*|load_matrix_sync)\b|"
    r"\b(As|Bs|smem_[AB]|lds_[AB]|s_[ab]|sh_[ab])\s*\[",
    re.IGNORECASE,
)
STORE_PATTERNS = re.compile(
    r"\b(store_matrix_sync|st_global|tma_store\w*)\b|"
    r"\b(C|D|out|dst|c_ptr|out_ptr)\s*\[[^\]]+\]\s*=",
)


def _strip_cpp_comments_keep_len(src: str) -> str:
    """Replace comments and string literals with spaces (preserving newlines and offsets)."""
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
            quote = ch
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
                if src[i] == quote:
                    out[i] = " "
                    i += 1
                    break
                if src[i] != "\n":
                    out[i] = " "
                i += 1
        else:
            i += 1
    return "".join(out)


def _find_matching_delim(clean: str, open_pos: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    for i in range(open_pos, len(clean)):
        if clean[i] == open_ch:
            depth += 1
        elif clean[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"Unbalanced {open_ch}{close_ch} at offset {open_pos}")


def _split_top_level_cpp_statements(src: str, clean: str, body_start: int, body_end: int) -> List[Tuple[int, int, str]]:
    """Split C++ block interior [body_start, body_end) into top-level statements (start, end, text)."""
    stmts: List[Tuple[int, int, str]] = []
    i = body_start
    while i < body_end:
        while i < body_end and clean[i].isspace():
            i += 1
        if i >= body_end:
            break
        # Preprocessor directive line (#pragma unroll, #if, etc.)
        if src[i] == "#":
            nl = src.find("\n", i, body_end)
            end = body_end if nl == -1 else nl + 1
            # Attach #pragma unroll to the immediately following loop/statement
            pragma_txt = src[i:end].strip()
            if pragma_txt.startswith("#pragma"):
                j = end
                while j < body_end and clean[j].isspace():
                    j += 1
                if j < body_end:
                    # Let statement start at `i` and continue scanning from `j`
                    stmt_start = i
                    i = j
                else:
                    stmts.append((i, end, src[i:end]))
                    break
            else:
                stmts.append((i, end, src[i:end]))
                i = end
                continue
        else:
            stmt_start = i

        # Control statements (`for (...) stmt`, `if (...) stmt [else stmt]`, `while (...) stmt`)
        kw_m = re.match(r"\b(for|while|if)\b", clean[i:body_end])
        if kw_m:
            kw = kw_m.group(1)
            p_open = clean.find("(", i + len(kw), body_end)
            if p_open != -1:
                p_close = _find_matching_delim(clean, p_open, "(", ")")
                j = p_close + 1
                while j < body_end and clean[j].isspace():
                    j += 1
                if j < body_end and clean[j] == "{":
                    b_close = _find_matching_delim(clean, j, "{", "}")
                    end = b_close + 1
                else:
                    semi = clean.find(";", j, body_end)
                    end = (semi + 1) if semi != -1 else body_end
                # Check optional `else` for `if`
                if kw == "if":
                    k = end
                    while k < body_end and clean[k].isspace():
                        k += 1
                    if re.match(r"\belse\b", clean[k:body_end]):
                        k += 4
                        while k < body_end and clean[k].isspace():
                            k += 1
                        if k < body_end and clean[k] == "{":
                            end = _find_matching_delim(clean, k, "{", "}") + 1
                        else:
                            semi = clean.find(";", k, body_end)
                            end = (semi + 1) if semi != -1 else body_end
                stmts.append((stmt_start, end, src[stmt_start:end]))
                i = end
                continue

        if clean[i] == "{":
            b_close = _find_matching_delim(clean, i, "{", "}")
            end = b_close + 1
            stmts.append((stmt_start, end, src[stmt_start:end]))
            i = end
            continue

        # Ordinary semicolon-terminated statement (skipping nested parens/braces)
        depth_p = 0
        depth_b = 0
        j = i
        while j < body_end:
            c = clean[j]
            if c == "(":
                depth_p += 1
            elif c == ")":
                depth_p = max(0, depth_p - 1)
            elif c == "{":
                depth_b += 1
            elif c == "}":
                depth_b = max(0, depth_b - 1)
            elif c == ";" and depth_p == 0 and depth_b == 0:
                j += 1
                break
            j += 1
        stmts.append((stmt_start, j, src[stmt_start:j]))
        i = j
    return stmts


def instrument_cpp_gemm(
    src: str,
    *,
    language: str,
    card: str,
    arch: str,
    kernel_name: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Automatically analyze and instrument an uninstrumented CUDA/HIP C++ GEMM file."""
    header_name = "fi_lite.cuh" if language == "cuda_cpp" else "fi_lite_hip.h"
    if header_name in src and "FI_OPEN" in src:
        # Already instrumented: extract metadata without double-injecting
        kname = kernel_name or "gemm_kernel"
        m = CPP_KERNEL_RE.search(src)
        if m:
            kname = m.group(1)
        phases = ["gemm", "load", "wait", "mma", "barrier", "store"]
        names = build_names_doc(phases, card=card, arch=arch, language=language, kernel_name=kname)
        plan = {
            "already_instrumented": True,
            "kernel": kname,
            "language": language,
            "card": card,
            "arch": arch,
            "phases": phases,
        }
        return src, names, plan

    clean = _strip_cpp_comments_keep_len(src)
    candidates = []
    for m in CPP_KERNEL_RE.finditer(clean):
        name = m.group(1)
        if any(skip in name.lower() for skip in ("init_", "reference", "ref_", "fill_", "verify")):
            continue
        p_open = m.end() - 1
        p_close = _find_matching_delim(clean, p_open, "(", ")")
        b_open = clean.find("{", p_close + 1)
        if b_open == -1:
            continue
        b_close = _find_matching_delim(clean, b_open, "{", "}")
        body = src[b_open + 1:b_close]
        score = (10 if MMA_PATTERNS.search(body) else 0) + (5 if "for" in body else 0)
        if kernel_name and name == kernel_name:
            score += 100
        candidates.append((score, name, p_open, p_close, b_open, b_close))

    if not candidates:
        raise ValueError("No candidate __global__ GEMM kernel found in C++ source")
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, chosen_name, _, _, b_open, b_close = candidates[0]

    top_stmts = _split_top_level_cpp_statements(src, clean, b_open + 1, b_close)
    # Locate a top-level K-reduction `for` loop (`for (...) { ... }` at top level, not nested inside `if/else`)
    kloop_idx: Optional[int] = None
    for idx, (s_beg, s_end, s_txt) in enumerate(top_stmts):
        s_clean = clean[s_beg:s_end]
        if re.match(r"\s*for\s*\(", s_clean) and (MMA_PATTERNS.search(s_txt) or SYNC_PATTERNS.search(s_txt)):
            kloop_idx = idx
            break
    if kloop_idx is None:
        # Check for Producer/Consumer Warp-Specialized `if (...) { ... } else { ... }` structure!
        for idx, (s_beg, s_end, _s_txt) in enumerate(top_stmts):
            s_clean = clean[s_beg:s_end]
            if_m = re.match(r"\s*if\s*\(", s_clean)
            if not if_m:
                continue
            p_open_if = clean.find("(", s_beg, s_end)
            if p_open_if == -1:
                continue
            p_close_if = _find_matching_delim(clean, p_open_if, "(", ")")
            j_then = p_close_if + 1
            while j_then < s_end and clean[j_then].isspace():
                j_then += 1
            if j_then >= s_end or clean[j_then] != "{":
                continue
            then_close = _find_matching_delim(clean, j_then, "{", "}")
            k_else = then_close + 1
            while k_else < s_end and clean[k_else].isspace():
                k_else += 1
            if not re.match(r"\belse\b", clean[k_else:s_end]):
                continue
            j_else = k_else + 4
            while j_else < s_end and clean[j_else].isspace():
                j_else += 1
            if j_else >= s_end or clean[j_else] != "{":
                continue
            else_close = _find_matching_delim(clean, j_else, "{", "}")
            then_txt = src[j_then + 1:then_close]
            else_txt = src[j_else + 1:else_close]
            then_has_mma = bool(MMA_PATTERNS.search(then_txt))
            else_has_mma = bool(MMA_PATTERNS.search(else_txt))
            if then_has_mma != else_has_mma:
                prod_bounds = (j_then, then_close) if else_has_mma else (j_else, else_close)
                cons_bounds = (j_else, else_close) if else_has_mma else (j_then, then_close)
                return _instrument_cpp_warp_specialized(
                    src, clean,
                    header_name=header_name,
                    chosen_name=chosen_name,
                    b_open=b_open,
                    prod_bounds=prod_bounds,
                    cons_bounds=cons_bounds,
                    language=language,
                    card=card,
                    arch=arch,
                )
        raise ValueError(f"Kernel {chosen_name}: could not locate main K-reduction for-loop")

    k_beg, k_end, _ = top_stmts[kloop_idx]
    for_m = re.search(r"\bfor\s*\(", clean[k_beg:k_end])
    assert for_m is not None
    lp_open = k_beg + for_m.end() - 1
    lp_close = _find_matching_delim(clean, lp_open, "(", ")")
    lb_open = clean.find("{", lp_close + 1, k_end)
    if lb_open == -1:
        raise ValueError(f"Kernel {chosen_name}: K-loop must have a brace-delimited body")
    lb_close = _find_matching_delim(clean, lb_open, "{", "}")

    loop_stmts = _split_top_level_cpp_statements(src, clean, lb_open + 1, lb_close)
    # Classify each statement inside K-loop into load / wait / mma / barrier / other
    mma_indices = [
        i for i, (_, _, txt) in enumerate(loop_stmts)
        if MMA_PATTERNS.search(txt) and not (LOAD_PATTERNS.search(txt) and not re.search(r"mma_sync|wgmma|tcgen05|mfma|fmaf", txt))
    ]
    if not mma_indices:
        raise ValueError(f"Kernel {chosen_name}: no MMA/FMA compute statement detected inside K-loop")
    first_mma = min(mma_indices)
    last_mma = max(mma_indices)

    # Pre-MMA wait/sync statements immediately preceding `first_mma`
    wait_indices: List[int] = []
    for i in range(first_mma - 1, -1, -1):
        txt = loop_stmts[i][2]
        if WAIT_PATTERNS.search(txt) or SYNC_PATTERNS.search(txt) or "pipeline_commit" in txt or "commit_group" in txt:
            wait_indices.insert(0, i)
        else:
            break

    load_end_limit = wait_indices[0] if wait_indices else first_mma
    load_indices = [
        i for i in range(0, load_end_limit)
        if LOAD_PATTERNS.search(loop_stmts[i][2]) or "*reinterpret_cast" in loop_stmts[i][2]
    ]

    barrier_indices = [
        i for i in range(last_mma + 1, len(loop_stmts))
        if SYNC_PATTERNS.search(loop_stmts[i][2]) or WAIT_PATTERNS.search(loop_stmts[i][2])
    ]

    # Post-K-loop store statements
    store_indices = [
        idx for idx in range(kloop_idx + 1, len(top_stmts))
        if STORE_PATTERNS.search(top_stmts[idx][2]) or "store_matrix_sync" in top_stmts[idx][2]
    ]

    edits: List[Tuple[int, int, str]] = []
    phases_used = ["gemm"]
    phase_spans_meta: List[Dict[str, Any]] = []

    def line_of(offset: int) -> int:
        return src.count("\n", 0, offset) + 1

    # Wrap K-loop sub-stages using branchless predicated FI_BG_PRED / FI_EG_PRED
    if load_indices:
        phases_used.append("load")
        beg_pos = loop_stmts[load_indices[0]][0]
        end_pos = loop_stmts[load_indices[-1]][1]
        edits.append((beg_pos, beg_pos, "FI_BG_PRED(fi_rec_k_, 0, FI_EV_LOAD_, fi_k_idx_);\n        "))
        edits.append((end_pos, end_pos, "\n        FI_EG_PRED(fi_rec_k_, 0, FI_EV_LOAD_, fi_k_idx_);"))
        phase_spans_meta.append({"semantic_id": "load", "start_line": line_of(beg_pos), "end_line": line_of(end_pos)})

    if wait_indices:
        phases_used.append("wait")
        beg_pos = loop_stmts[wait_indices[0]][0]
        end_pos = loop_stmts[wait_indices[-1]][1]
        edits.append((beg_pos, beg_pos, "FI_BG_PRED(fi_rec_k_, 1, FI_EV_WAIT_, fi_k_idx_);\n        "))
        edits.append((end_pos, end_pos, "\n        FI_EG_PRED(fi_rec_k_, 1, FI_EV_WAIT_, fi_k_idx_);"))
        phase_spans_meta.append({"semantic_id": "wait", "start_line": line_of(beg_pos), "end_line": line_of(end_pos)})

    phases_used.append("mma")
    mma_beg = loop_stmts[first_mma][0]
    mma_end = loop_stmts[last_mma][1]
    edits.append((mma_beg, mma_beg, "FI_BG_PRED(fi_rec_k_, 1, FI_EV_MMA_, fi_k_idx_);\n        "))
    edits.append((mma_end, mma_end, "\n        FI_EG_PRED(fi_rec_k_, 1, FI_EV_MMA_, fi_k_idx_);"))
    phase_spans_meta.append({"semantic_id": "mma", "start_line": line_of(mma_beg), "end_line": line_of(mma_end)})

    if barrier_indices:
        phases_used.append("barrier")
        beg_pos = loop_stmts[barrier_indices[0]][0]
        end_pos = loop_stmts[barrier_indices[-1]][1]
        edits.append((beg_pos, beg_pos, "FI_BG_PRED(fi_rec_k_, 1, FI_EV_BARRIER_, fi_k_idx_);\n        "))
        edits.append((end_pos, end_pos, "\n        FI_EG_PRED(fi_rec_k_, 1, FI_EV_BARRIER_, fi_k_idx_);"))
        phase_spans_meta.append({"semantic_id": "barrier", "start_line": line_of(beg_pos), "end_line": line_of(end_pos)})

    # Loop counter header & increment inside K-loop
    edits.insert(0, (
        lb_open + 1,
        lb_open + 1,
        "\n        const bool fi_rec_k_ = (FI_SAMPLE == 0) || "
        "(FI_SAMPLE == 1 ? (fi_k_idx_ == 0) : ((fi_k_idx_ % (FI_SAMPLE > 1 ? FI_SAMPLE : 1)) == 0));\n"
        "        (void)fi_rec_k_;",
    ))
    edits.append((lb_close, lb_close, "\n        ++fi_k_idx_;\n    "))

    # Place FI_OPEN + FI_BG(gemm) right before the K-loop (safely after any prologue early-return guards!)
    open_inject = (
        "const uint32_t fi_tile_id_ = (uint32_t)(blockIdx.y * gridDim.x + blockIdx.x);\n"
        "    (void)fi_tile_id_;\n"
        "    uint32_t fi_k_idx_ = 0u;\n"
        "    FI_OPEN(0);\n"
        "    FI_BG(2, FI_EV_GEMM_, fi_tile_id_);\n    "
    )
    edits.insert(0, (k_beg, k_beg, open_inject))

    # Wrap post-loop store
    if store_indices:
        phases_used.append("store")
        sb = top_stmts[store_indices[0]][0]
        se = top_stmts[store_indices[-1]][1]
        edits.append((sb, sb, "FI_BG(0, FI_EV_STORE_, fi_tile_id_);\n    "))
        edits.append((se, se, "\n    FI_EG(0, FI_EV_STORE_, fi_tile_id_);"))
        phase_spans_meta.append({"semantic_id": "store", "start_line": line_of(sb), "end_line": line_of(se)})

    # Inject FI_CLOSE before any `return;` after k_end, and at function exit `b_close`
    for ret_m in re.finditer(r"\breturn\s*;", clean[k_end:b_close]):
        rpos = k_end + ret_m.start()
        edits.append((rpos, rpos, "{ FI_EG(2, FI_EV_GEMM_, fi_tile_id_); FI_CLOSE(); } "))

    edits.append((b_close, b_close, "\n    FI_EG(2, FI_EV_GEMM_, fi_tile_id_);\n    FI_CLOSE();\n"))

    # Preamble header & FI_SAMPLE / event constants injected after last #include before kernel
    includes = list(re.finditer(r"^\s*#\s*include\s+[<\"][^>\"]+[>\"][^\n]*\n", src[:b_open], re.MULTILINE))
    preamble_pos = includes[-1].end() if includes else 0
    preamble = (
        f"\n#ifndef FI_SAMPLE\n#define FI_SAMPLE 0\n#endif\n"
        f"#include \"{header_name}\"\n"
        f"enum : uint32_t {{\n"
        f"    FI_EV_LOAD_ = {SEMANTIC_EVENTS['load']}u,\n"
        f"    FI_EV_WAIT_ = {SEMANTIC_EVENTS['wait']}u,\n"
        f"    FI_EV_MMA_ = {SEMANTIC_EVENTS['mma']}u,\n"
        f"    FI_EV_EPILOGUE_ = {SEMANTIC_EVENTS['epilogue']}u,\n"
        f"    FI_EV_STORE_ = {SEMANTIC_EVENTS['store']}u,\n"
        f"    FI_EV_BARRIER_ = {SEMANTIC_EVENTS['barrier']}u,\n"
        f"    FI_EV_GEMM_ = {SEMANTIC_EVENTS['gemm']}u,\n"
        f"}};\n"
    )
    edits.insert(0, (preamble_pos, preamble_pos, preamble))

    # Apply edits from back to front (stable indices; preserve relative order for same-offset edits)
    indexed_edits = list(enumerate(edits))
    indexed_edits.sort(key=lambda item: (item[1][0], item[1][1], item[0]))
    pieces: List[str] = []
    cursor = 0
    for _, (eb, ee, repl) in indexed_edits:
        if eb < cursor:
            raise RuntimeError(f"Overlapping C++ instrumentation edit at offset {eb} < {cursor}")
        pieces.append(src[cursor:eb])
        pieces.append(repl)
        cursor = ee
    pieces.append(src[cursor:])
    instrumented = "".join(pieces)

    names = build_names_doc(phases_used, card=card, arch=arch, language=language, kernel_name=chosen_name)
    plan = {
        "schema": "weaveprofiler.auto-gemm-plan/v1",
        "already_instrumented": False,
        "kernel": chosen_name,
        "language": language,
        "platform": CARD_DEFAULTS[card]["platform"],
        "card": card,
        "arch": arch,
        "gran": 1,
        "warp_specialized": False,
        "source_sha256": sha256_text(src),
        "instrumented_sha256": sha256_text(instrumented),
        "phases": phases_used,
        "phase_spans": phase_spans_meta,
    }
    return instrumented, names, plan


def _instrument_cpp_warp_specialized(
    src: str,
    clean: str,
    *,
    header_name: str,
    chosen_name: str,
    b_open: int,
    prod_bounds: Tuple[int, int],
    cons_bounds: Tuple[int, int],
    language: str,
    card: str,
    arch: str,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Instrument a Producer/Consumer Warp-Specialized C++ GEMM kernel at warp granularity (FI_GRAN=0)."""
    edits: List[Tuple[int, int, str]] = []
    phases_set = {"gemm"}
    phase_spans_meta: List[Dict[str, Any]] = []

    def line_of(offset: int) -> int:
        return src.count("\n", 0, offset) + 1

    rec_k_decl = (
        "\n            const bool fi_rec_k_ = (FI_SAMPLE == 0) || "
        "(FI_SAMPLE == 1 ? (fi_k_idx_ == 0) : ((fi_k_idx_ % (FI_SAMPLE > 1 ? FI_SAMPLE : 1)) == 0));\n"
        "            (void)fi_rec_k_;"
    )

    # 1. Producer_DMA role (role=1)
    pb_open, pb_close = prod_bounds
    p_stmts = _split_top_level_cpp_statements(src, clean, pb_open + 1, pb_close)
    pk_idx = next(
        (i for i, (sb, se, txt) in enumerate(p_stmts) if re.search(r"\bfor\s*\(", clean[sb:se]) and LOAD_PATTERNS.search(txt)),
        None,
    )
    if pk_idx is None:
        raise ValueError(f"Kernel {chosen_name}: Producer branch lacks K-loop with async/shared load")
    pk_beg, pk_end, _ = p_stmts[pk_idx]
    p_for_m = re.search(r"\bfor\s*\(", clean[pk_beg:pk_end])
    assert p_for_m is not None
    plp_close = _find_matching_delim(clean, pk_beg + p_for_m.end() - 1, "(", ")")
    plb_open = clean.find("{", plp_close + 1, pk_end)
    plb_close = _find_matching_delim(clean, plb_open, "{", "}")
    p_loop_stmts = _split_top_level_cpp_statements(src, clean, plb_open + 1, plb_close)

    p_load_idxs = [
        i for i, (_, _, t) in enumerate(p_loop_stmts)
        if LOAD_PATTERNS.search(t) or "mbar_arrive" in t
    ]
    p_wait_idxs = [
        i for i in range(0, p_load_idxs[0] if p_load_idxs else len(p_loop_stmts))
        if WAIT_PATTERNS.search(p_loop_stmts[i][2]) or SYNC_PATTERNS.search(p_loop_stmts[i][2])
    ]

    edits.append((
        pk_beg, pk_beg,
        "const uint32_t fi_tile_id_ = (uint32_t)(blockIdx.y * gridDim.x + blockIdx.x);\n"
        "        (void)fi_tile_id_;\n"
        "        uint32_t fi_k_idx_ = 0u;\n"
        "        const uint32_t fi_lin_tid_ = (uint32_t)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));\n"
        "        const bool fi_wg_leader_ = ((blockDim.x * blockDim.y * blockDim.z) >= 256u) ? ((fi_lin_tid_ & 127u) == 0u) : ((fi_lin_tid_ & 31u) == 0u);\n"
        "        FI_OPEN_BY(fi_wg_leader_, 1);\n"
        "        FI_BG(2, FI_EV_GEMM_, fi_tile_id_);\n        ",
    ))
    edits.append((plb_open + 1, plb_open + 1, rec_k_decl))
    if p_wait_idxs and p_load_idxs and (p_wait_idxs[-1] + 1 == p_load_idxs[0]):
        phases_set.add("wait")
        phases_set.add("load")
        wb, we = p_loop_stmts[p_wait_idxs[0]][0], p_loop_stmts[p_wait_idxs[-1]][1]
        lb, le = p_loop_stmts[p_load_idxs[0]][0], p_loop_stmts[p_load_idxs[-1]][1]
        edits.append((wb, wb, "FI_BG_PRED(fi_rec_k_, 1, FI_EV_WAIT_, fi_k_idx_);\n            "))
        edits.append((we, lb, "\n            FI_TRANS_PRED(fi_rec_k_, 1, FI_EV_WAIT_, 0, FI_EV_LOAD_, fi_k_idx_);\n            "))
        edits.append((le, le, "\n            FI_EG_PRED(fi_rec_k_, 0, FI_EV_LOAD_, fi_k_idx_);"))
        phase_spans_meta.append({"role": "Producer_DMA", "semantic_id": "wait", "start_line": line_of(wb), "end_line": line_of(we)})
        phase_spans_meta.append({"role": "Producer_DMA", "semantic_id": "load", "start_line": line_of(lb), "end_line": line_of(le)})
    else:
        if p_wait_idxs:
            phases_set.add("wait")
            wb, we = p_loop_stmts[p_wait_idxs[0]][0], p_loop_stmts[p_wait_idxs[-1]][1]
            edits.append((wb, wb, "FI_BG_PRED(fi_rec_k_, 1, FI_EV_WAIT_, fi_k_idx_);\n            "))
            edits.append((we, we, "\n            FI_EG_PRED(fi_rec_k_, 1, FI_EV_WAIT_, fi_k_idx_);"))
            phase_spans_meta.append({"role": "Producer_DMA", "semantic_id": "wait", "start_line": line_of(wb), "end_line": line_of(we)})
        if p_load_idxs:
            phases_set.add("load")
            lb, le = p_loop_stmts[p_load_idxs[0]][0], p_loop_stmts[p_load_idxs[-1]][1]
            edits.append((lb, lb, "FI_BG_PRED(fi_rec_k_, 0, FI_EV_LOAD_, fi_k_idx_);\n            "))
            edits.append((le, le, "\n            FI_EG_PRED(fi_rec_k_, 0, FI_EV_LOAD_, fi_k_idx_);"))
            phase_spans_meta.append({"role": "Producer_DMA", "semantic_id": "load", "start_line": line_of(lb), "end_line": line_of(le)})
    edits.append((plb_close, plb_close, "\n            ++fi_k_idx_;\n        "))
    edits.append((pb_close, pb_close, "\n        FI_EG(2, FI_EV_GEMM_, fi_tile_id_);\n        FI_CLOSE();\n    "))

    # 2. Consumer_MMA role (role=2)
    cb_open, cb_close = cons_bounds
    c_stmts = _split_top_level_cpp_statements(src, clean, cb_open + 1, cb_close)
    ck_idx = next(
        (i for i, (sb, se, txt) in enumerate(c_stmts) if re.search(r"\bfor\s*\(", clean[sb:se]) and MMA_PATTERNS.search(txt)),
        None,
    )
    if ck_idx is None:
        raise ValueError(f"Kernel {chosen_name}: Consumer branch lacks K-loop with MMA")
    ck_beg, ck_end, _ = c_stmts[ck_idx]
    c_for_m = re.search(r"\bfor\s*\(", clean[ck_beg:ck_end])
    assert c_for_m is not None
    clp_close = _find_matching_delim(clean, ck_beg + c_for_m.end() - 1, "(", ")")
    clb_open = clean.find("{", clp_close + 1, ck_end)
    clb_close = _find_matching_delim(clean, clb_open, "{", "}")
    c_loop_stmts = _split_top_level_cpp_statements(src, clean, clb_open + 1, clb_close)

    c_mma_idxs = [i for i, (_, _, t) in enumerate(c_loop_stmts) if MMA_PATTERNS.search(t)]
    first_c_mma, last_c_mma = min(c_mma_idxs), max(c_mma_idxs)
    # Keep non-blocking empty_mbar release (`mbarrier.arrive` / `mbar_arrive`) fused inside the Consumer MMA span
    # so `ptxas` can hoist `SYNCS.ARRIVE` above the tail of register-resident `HMMA` instructions (avoiding a
    # +22% Producer-Consumer pipeline stall and +67 register bloat!). Only split out true blocking barriers.
    mma_span_end_idx = last_c_mma
    while mma_span_end_idx + 1 < len(c_loop_stmts):
        nxt_txt = c_loop_stmts[mma_span_end_idx + 1][2]
        if ("arrive" in nxt_txt) and not WAIT_PATTERNS.search(nxt_txt) and "__syncthreads" not in nxt_txt:
            mma_span_end_idx += 1
        else:
            break
    c_wait_idxs = [
        i for i in range(0, first_c_mma)
        if WAIT_PATTERNS.search(c_loop_stmts[i][2]) or SYNC_PATTERNS.search(c_loop_stmts[i][2])
    ]
    c_bar_idxs = [
        i for i in range(mma_span_end_idx + 1, len(c_loop_stmts))
        if WAIT_PATTERNS.search(c_loop_stmts[i][2]) or "__syncthreads" in c_loop_stmts[i][2]
    ]
    c_store_idxs = [
        i for i in range(ck_idx + 1, len(c_stmts))
        if STORE_PATTERNS.search(c_stmts[i][2]) or "store_matrix_sync" in c_stmts[i][2]
    ]

    edits.append((
        ck_beg, ck_beg,
        "const uint32_t fi_tile_id_ = (uint32_t)(blockIdx.y * gridDim.x + blockIdx.x);\n"
        "        (void)fi_tile_id_;\n"
        "        uint32_t fi_k_idx_ = 0u;\n"
        "        const uint32_t fi_lin_tid_ = (uint32_t)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));\n"
        "        const bool fi_wg_leader_ = ((blockDim.x * blockDim.y * blockDim.z) >= 256u) ? ((fi_lin_tid_ & 127u) == 0u) : ((fi_lin_tid_ & 31u) == 0u);\n"
        "        FI_OPEN_BY(fi_wg_leader_, 2);\n"
        "        FI_BG(2, FI_EV_GEMM_, fi_tile_id_);\n        ",
    ))
    edits.append((clb_open + 1, clb_open + 1, rec_k_decl))
    mb, me = c_loop_stmts[first_c_mma][0], c_loop_stmts[mma_span_end_idx][1]
    if c_wait_idxs and (c_wait_idxs[-1] + 1 == first_c_mma):
        phases_set.add("wait")
        phases_set.add("mma")
        wb, we = c_loop_stmts[c_wait_idxs[0]][0], c_loop_stmts[c_wait_idxs[-1]][1]
        edits.append((wb, wb, "FI_BG_PRED(fi_rec_k_, 1, FI_EV_WAIT_, fi_k_idx_);\n            "))
        edits.append((we, mb, "\n            FI_TRANS_PRED(fi_rec_k_, 1, FI_EV_WAIT_, 1, FI_EV_MMA_, fi_k_idx_);\n            "))
        edits.append((me, me, "\n            FI_EG_PRED(fi_rec_k_, 1, FI_EV_MMA_, fi_k_idx_);"))
        phase_spans_meta.append({"role": "Consumer_MMA", "semantic_id": "wait", "start_line": line_of(wb), "end_line": line_of(we)})
        phase_spans_meta.append({"role": "Consumer_MMA", "semantic_id": "mma", "start_line": line_of(mb), "end_line": line_of(me)})
    else:
        if c_wait_idxs:
            phases_set.add("wait")
            wb, we = c_loop_stmts[c_wait_idxs[0]][0], c_loop_stmts[c_wait_idxs[-1]][1]
            edits.append((wb, wb, "FI_BG_PRED(fi_rec_k_, 1, FI_EV_WAIT_, fi_k_idx_);\n            "))
            edits.append((we, we, "\n            FI_EG_PRED(fi_rec_k_, 1, FI_EV_WAIT_, fi_k_idx_);"))
            phase_spans_meta.append({"role": "Consumer_MMA", "semantic_id": "wait", "start_line": line_of(wb), "end_line": line_of(we)})
        phases_set.add("mma")
        edits.append((mb, mb, "FI_BG_PRED(fi_rec_k_, 1, FI_EV_MMA_, fi_k_idx_);\n            "))
        edits.append((me, me, "\n            FI_EG_PRED(fi_rec_k_, 1, FI_EV_MMA_, fi_k_idx_);"))
        phase_spans_meta.append({"role": "Consumer_MMA", "semantic_id": "mma", "start_line": line_of(mb), "end_line": line_of(me)})
    if c_bar_idxs:
        phases_set.add("barrier")
        bb, be = c_loop_stmts[c_bar_idxs[0]][0], c_loop_stmts[c_bar_idxs[-1]][1]
        edits.append((bb, bb, "FI_BG_PRED(fi_rec_k_, 1, FI_EV_BARRIER_, fi_k_idx_);\n            "))
        edits.append((be, be, "\n            FI_EG_PRED(fi_rec_k_, 1, FI_EV_BARRIER_, fi_k_idx_);"))
        phase_spans_meta.append({"role": "Consumer_MMA", "semantic_id": "barrier", "start_line": line_of(bb), "end_line": line_of(be)})
    edits.append((clb_close, clb_close, "\n            ++fi_k_idx_;\n        "))
    if c_store_idxs:
        phases_set.add("store")
        sb, se = c_stmts[c_store_idxs[0]][0], c_stmts[c_store_idxs[-1]][1]
        edits.append((sb, sb, "FI_BG(0, FI_EV_STORE_, fi_tile_id_);\n        "))
        edits.append((se, se, "\n        FI_EG(0, FI_EV_STORE_, fi_tile_id_);"))
        phase_spans_meta.append({"role": "Consumer_MMA", "semantic_id": "store", "start_line": line_of(sb), "end_line": line_of(se)})
    edits.append((cb_close, cb_close, "\n        FI_EG(2, FI_EV_GEMM_, fi_tile_id_);\n        FI_CLOSE();\n    "))

    includes = list(re.finditer(r"^\s*#\s*include\s+[<\"][^>\"]+[>\"][^\n]*\n", src[:b_open], re.MULTILINE))
    preamble_pos = includes[-1].end() if includes else 0
    preamble = (
        f"\n#ifndef FI_SAMPLE\n#define FI_SAMPLE 0\n#endif\n"
        f"#include \"{header_name}\"\n"
        f"enum : uint32_t {{\n"
        f"    FI_EV_LOAD_ = {SEMANTIC_EVENTS['load']}u,\n"
        f"    FI_EV_WAIT_ = {SEMANTIC_EVENTS['wait']}u,\n"
        f"    FI_EV_MMA_ = {SEMANTIC_EVENTS['mma']}u,\n"
        f"    FI_EV_EPILOGUE_ = {SEMANTIC_EVENTS['epilogue']}u,\n"
        f"    FI_EV_STORE_ = {SEMANTIC_EVENTS['store']}u,\n"
        f"    FI_EV_BARRIER_ = {SEMANTIC_EVENTS['barrier']}u,\n"
        f"    FI_EV_GEMM_ = {SEMANTIC_EVENTS['gemm']}u,\n"
        f"}};\n"
    )
    edits.insert(0, (preamble_pos, preamble_pos, preamble))

    indexed_edits = list(enumerate(edits))
    indexed_edits.sort(key=lambda item: (item[1][0], item[1][1], item[0]))
    pieces: List[str] = []
    cursor = 0
    for _, (eb, ee, repl) in indexed_edits:
        if eb < cursor:
            raise RuntimeError(f"Overlapping Warp-Specialized C++ edit at offset {eb} < {cursor}")
        pieces.append(src[cursor:eb])
        pieces.append(repl)
        cursor = ee
    pieces.append(src[cursor:])
    instrumented = "".join(pieces)

    phases_ordered = [p for p in ("gemm", "load", "wait", "mma", "barrier", "store") if p in phases_set]
    names = build_names_doc(
        phases_ordered,
        card=card,
        arch=arch,
        language=language,
        kernel_name=chosen_name,
        warp_specialized=True,
    )
    plan = {
        "schema": "weaveprofiler.auto-gemm-plan/v1",
        "already_instrumented": False,
        "kernel": chosen_name,
        "language": language,
        "platform": CARD_DEFAULTS[card]["platform"],
        "card": card,
        "arch": arch,
        "gran": 0,
        "warp_specialized": True,
        "roles": {"1": "Producer_DMA", "2": "Consumer_MMA"},
        "source_sha256": sha256_text(src),
        "instrumented_sha256": sha256_text(instrumented),
        "phases": phases_ordered,
        "phase_spans": phase_spans_meta,
    }
    return instrumented, names, plan


# ---------------------------------------------------------------------------
# Python AST (@triton.jit & @cute.kernel) Analyzer & Instrumenter
# ---------------------------------------------------------------------------

def _ast_contains_call(node: ast.AST, names: Tuple[str, ...]) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            if isinstance(fn, ast.Attribute) and fn.attr in names:
                return True
            if isinstance(fn, ast.Name) and fn.id in names:
                return True
    return False


def instrument_triton_gemm(
    src: str,
    *,
    card: str,
    arch: str,
    kernel_name: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Automatically analyze & instrument an uninstrumented @triton.jit GEMM kernel."""
    platform = CARD_DEFAULTS[card]["platform"]
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)

    jit_funcs: List[ast.FunctionDef] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            is_jit = any(
                (isinstance(d, ast.Attribute) and d.attr == "jit")
                or (isinstance(d, ast.Name) and d.id == "jit")
                or (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "jit")
                for d in node.decorator_list
            )
            if is_jit and (kernel_name is None or node.name == kernel_name):
                if _ast_contains_call(node, ("dot", "dot_scaled")):
                    jit_funcs.append(node)
    if not jit_funcs:
        raise ValueError("No @triton.jit GEMM kernel containing tl.dot found")
    fn = jit_funcs[0]

    # Locate K-reduction For loop containing tl.dot
    k_for: Optional[ast.For] = None
    for stmt in fn.body:
        if isinstance(stmt, ast.For) and _ast_contains_call(stmt, ("dot", "dot_scaled")):
            k_for = stmt
            break
    if k_for is None:
        raise ValueError(f"Triton kernel {fn.name}: no top-level K-reduction `for` loop with tl.dot found")

    # Locate load statements and dot statements inside k_for.body
    dot_idxs = [i for i, st in enumerate(k_for.body) if _ast_contains_call(st, ("dot", "dot_scaled"))]
    load_idxs = [i for i, st in enumerate(k_for.body[:dot_idxs[0]]) if _ast_contains_call(st, ("load",))]
    store_stmts = [st for st in fn.body if st.lineno > k_for.end_lineno and _ast_contains_call(st, ("store",))]

    body_indent = " " * fn.body[0].col_offset
    loop_indent = " " * k_for.body[0].col_offset

    phases_used = ["gemm"]
    if load_idxs:
        phases_used.append("load")
    phases_used.append("mma")
    if store_stmts:
        phases_used.append("store")

    # Line-based AST-guided rewrite (preserves 100% of user math, tile swizzling, masks, and dtypes)
    if platform == "nv":
        import_block = (
            "\n# --- Auto-injected by weaveprofiler auto_gemm_plan (NVIDIA Triton PTX) ---\n"
            "import os as _fi_os, sys as _fi_sys\n"
            "from pathlib import Path as _FiPath\n"
            "for _fi_cand in (\n"
            "    _fi_os.environ.get('WEAVEPROFILER_TRITON_DIR', ''),\n"
            f"    {str(SKILL_ROOT / '2_instrument' / 'recorders' / 'triton')!r},\n"
            "    str(_FiPath.home() / '.claude' / 'skills' / 'weaveprofiler' / '2_instrument' / 'recorders' / 'triton'),\n"
            "    str(_FiPath.home() / '.codex' / 'skills' / 'weaveprofiler' / '2_instrument' / 'recorders' / 'triton'),\n"
            "    str(_FiPath.home() / '.gemini' / 'config' / 'skills' / 'weaveprofiler' / '2_instrument' / 'recorders' / 'triton'),\n"
            "):\n"
            "    if _fi_cand and _fi_os.path.isdir(_fi_cand) and _fi_cand not in _fi_sys.path:\n"
            "        _fi_sys.path.insert(0, _fi_cand)\n"
            "        break\n"
            "from fi_triton import FiCfg as FiAutoCfg, FiBuffer as FiAutoBuffer, "
            "fi_open as _fi_open, fi_b as _fi_b, fi_e as _fi_e, fi_close as _fi_close\n"
            "_FI_EV_LOAD = tl.constexpr(1)\n"
            "_FI_EV_MMA = tl.constexpr(3)\n"
            "_FI_EV_STORE = tl.constexpr(6)\n"
            "_FI_EV_GEMM = tl.constexpr(10)\n"
        )
        extra_params_body = (
            "_fi_rec_ptr, _fi_ovf_ptr, _FI_CFG: tl.constexpr, _FI_SAMPLE: tl.constexpr = 0, _fi_rt_gate = 2"
        )
        open_lines = (
            f"{body_indent}_fi_cta = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0) + tl.program_id(2) * tl.num_programs(0) * tl.num_programs(1)\n"
            f"{body_indent}_fi_k_idx = 0\n"
            f"{body_indent}_fi_n = _fi_open(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _FI_CFG, 0)\n"
            f"{body_indent}_fi_n = _fi_b(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG, _FI_EV_GEMM, _fi_cta)\n"
        )
        loop_head = (
            f"{loop_indent}_fi_do_k = False if _FI_CFG.level < 2 else ((_fi_rt_gate > 0) if _FI_SAMPLE == 0 else (_fi_k_idx == 0 if _FI_SAMPLE == 1 else (_fi_k_idx % _FI_SAMPLE) == 0))\n"
        )
        load_b = f"{loop_indent}if _fi_do_k:\n{loop_indent}    _fi_n = _fi_b(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG, _FI_EV_LOAD, _fi_k_idx)\n"
        load_e = f"{loop_indent}if _fi_do_k:\n{loop_indent}    _fi_n = _fi_e(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG, _FI_EV_LOAD, _fi_k_idx)\n"
        mma_b = f"{loop_indent}if _fi_do_k:\n{loop_indent}    _fi_n = _fi_b(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG, _FI_EV_MMA, _fi_k_idx)\n"
        mma_e = f"{loop_indent}if _fi_do_k:\n{loop_indent}    _fi_n = _fi_e(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG, _FI_EV_MMA, _fi_k_idx)\n"
        loop_tail = f"{loop_indent}_fi_k_idx += 1\n"
        store_b = f"{body_indent}_fi_n = _fi_b(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG, _FI_EV_STORE, _fi_cta)\n"
        store_e = f"{body_indent}_fi_n = _fi_e(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG, _FI_EV_STORE, _fi_cta)\n"
        close_lines = (
            f"{body_indent}_fi_n = _fi_e(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG, _FI_EV_GEMM, _fi_cta)\n"
            f"{body_indent}_fi_n = _fi_close(_fi_rec_ptr, _fi_ovf_ptr, _fi_cta, _fi_n, _FI_CFG)\n"
        )
    else:
        # AMD MI308X gfx942: uses single-store 16B AoS fi_*_16b + compile-time _FI_LEVEL gate!
        import_block = (
            "\n# --- Auto-injected by weaveprofiler auto_gemm_plan (AMD MI308X gfx942 wave64 16B AoS) ---\n"
            "import os as _fi_os, sys as _fi_sys\n"
            "from pathlib import Path as _FiPath\n"
            "for _fi_cand in (\n"
            "    _fi_os.environ.get('WEAVEPROFILER_TRITON_DIR', ''),\n"
            f"    {str(SKILL_ROOT / '2_instrument' / 'recorders' / 'triton')!r},\n"
            "    str(_FiPath.home() / '.claude' / 'skills' / 'weaveprofiler' / '2_instrument' / 'recorders' / 'triton'),\n"
            "    str(_FiPath.home() / '.codex' / 'skills' / 'weaveprofiler' / '2_instrument' / 'recorders' / 'triton'),\n"
            "    str(_FiPath.home() / '.gemini' / 'config' / 'skills' / 'weaveprofiler' / '2_instrument' / 'recorders' / 'triton'),\n"
            "):\n"
            "    if _fi_cand and _fi_os.path.isdir(_fi_cand) and _fi_cand not in _fi_sys.path:\n"
            "        _fi_sys.path.insert(0, _fi_cand)\n"
            "        break\n"
            "from fi_triton_amd import FiPackedBuffer as FiAutoBuffer, "
            "fi_open_16b as _fi_open_16b, fi_b_16b as _fi_b_16b, fi_e_16b as _fi_e_16b, fi_close_16b as _fi_close_16b\n"
        )
        extra_params_body = (
            "_fi_rec_ptr, _fi_ovf_ptr, "
            "_FI_LEVEL: tl.constexpr = 0, _FI_GRAN: tl.constexpr = 1, "
            "_FI_CAP: tl.constexpr = 64, _FI_WPC: tl.constexpr = 1, "
            "_FI_THREADS: tl.constexpr = 256, _FI_TIMER: tl.constexpr = 1, "
            "_FI_SAMPLE: tl.constexpr = 0"
        )
        open_lines = (
            f"{body_indent}_fi_pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0) + tl.program_id(2) * tl.num_programs(0) * tl.num_programs(1)\n"
            f"{body_indent}_fi_k_idx = 0\n"
            f"{body_indent}_fi_base = tl.zeros([_FI_THREADS], dtype=tl.int32)\n"
            f"{body_indent}_fi_sel = _fi_base < 0\n"
            f"{body_indent}_fi_n = tl.zeros([_FI_THREADS], dtype=tl.int32)\n"
            f"{body_indent}if _FI_LEVEL >= 1:\n"
            f"{body_indent}    _fi_base, _fi_sel, _fi_n = _fi_open_16b(_fi_rec_ptr, _fi_ovf_ptr, ROLE=0, GRAN=_FI_GRAN, CAP=_FI_CAP, WPC=_FI_WPC, THREADS=_FI_THREADS, TIMER=_FI_TIMER)\n"
            f"{body_indent}if _FI_LEVEL >= 2:\n"
            f"{body_indent}    _fi_n = _fi_b_16b(_fi_rec_ptr, _fi_ovf_ptr, _fi_base, _fi_sel, _fi_n, EVENT=10, aux=_fi_pid, CAP=_FI_CAP, THREADS=_FI_THREADS, TIMER=_FI_TIMER)\n"
        )
        loop_head = (
            f"{loop_indent}_fi_do_k = (_FI_SAMPLE == 0) or (_fi_k_idx == 0 if _FI_SAMPLE == 1 else (_fi_k_idx % _FI_SAMPLE) == 0)\n"
        )
        load_b = f"{loop_indent}if _FI_LEVEL >= 2 and _fi_do_k:\n{loop_indent}    _fi_n = _fi_b_16b(_fi_rec_ptr, _fi_ovf_ptr, _fi_base, _fi_sel, _fi_n, EVENT=1, aux=_fi_k_idx, CAP=_FI_CAP, THREADS=_FI_THREADS, TIMER=_FI_TIMER)\n"
        load_e = f"{loop_indent}if _FI_LEVEL >= 2 and _fi_do_k:\n{loop_indent}    _fi_n = _fi_e_16b(_fi_rec_ptr, _fi_ovf_ptr, _fi_base, _fi_sel, _fi_n, EVENT=1, aux=_fi_k_idx, CAP=_FI_CAP, THREADS=_FI_THREADS, TIMER=_FI_TIMER)\n"
        mma_b = f"{loop_indent}if _FI_LEVEL >= 2 and _fi_do_k:\n{loop_indent}    _fi_n = _fi_b_16b(_fi_rec_ptr, _fi_ovf_ptr, _fi_base, _fi_sel, _fi_n, EVENT=3, aux=_fi_k_idx, CAP=_FI_CAP, THREADS=_FI_THREADS, TIMER=_FI_TIMER)\n"
        mma_e = f"{loop_indent}if _FI_LEVEL >= 2 and _fi_do_k:\n{loop_indent}    _fi_n = _fi_e_16b(_fi_rec_ptr, _fi_ovf_ptr, _fi_base, _fi_sel, _fi_n, EVENT=3, aux=_fi_k_idx, CAP=_FI_CAP, THREADS=_FI_THREADS, TIMER=_FI_TIMER)\n"
        loop_tail = f"{loop_indent}_fi_k_idx += 1\n"
        store_b = f"{body_indent}if _FI_LEVEL >= 2:\n{body_indent}    _fi_n = _fi_b_16b(_fi_rec_ptr, _fi_ovf_ptr, _fi_base, _fi_sel, _fi_n, EVENT=6, aux=_fi_pid, CAP=_FI_CAP, THREADS=_FI_THREADS, TIMER=_FI_TIMER)\n"
        store_e = f"{body_indent}if _FI_LEVEL >= 2:\n{body_indent}    _fi_n = _fi_e_16b(_fi_rec_ptr, _fi_ovf_ptr, _fi_base, _fi_sel, _fi_n, EVENT=6, aux=_fi_pid, CAP=_FI_CAP, THREADS=_FI_THREADS, TIMER=_FI_TIMER)\n"
        close_lines = (
            f"{body_indent}if _FI_LEVEL >= 2:\n"
            f"{body_indent}    _fi_n = _fi_e_16b(_fi_rec_ptr, _fi_ovf_ptr, _fi_base, _fi_sel, _fi_n, EVENT=10, aux=_fi_pid, CAP=_FI_CAP, THREADS=_FI_THREADS, TIMER=_FI_TIMER)\n"
            f"{body_indent}if _FI_LEVEL >= 1:\n"
            f"{body_indent}    _fi_n = _fi_close_16b(_fi_rec_ptr, _fi_ovf_ptr, _fi_base, _fi_sel, _fi_n, CAP=_FI_CAP, THREADS=_FI_THREADS, TIMER=_FI_TIMER)\n"
        )

    # Compute exact line insertions (1-based line numbers)
    # Find signature closing `):` line
    sig_end_line = fn.body[0].lineno - 1
    while sig_end_line >= fn.lineno and "):" not in lines[sig_end_line - 1]:
        sig_end_line -= 1
    if sig_end_line < fn.lineno:
        raise ValueError(f"Cannot locate closing '):' of {fn.name} signature")

    # Patch signature line (handling trailing comma on previous line cleanly)
    sig_prefix = "".join(lines[fn.lineno - 1:sig_end_line - 1]) + lines[sig_end_line - 1][:lines[sig_end_line - 1].rfind("):")]
    stripped_sig = re.sub(r"#[^\n]*", "", sig_prefix).rstrip()
    has_trailing_comma = stripped_sig.endswith(",") or stripped_sig.endswith("(")
    sep = "    " if lines[sig_end_line - 1][:lines[sig_end_line - 1].rfind("):")].strip() == "" else (", " if not has_trailing_comma else " ")
    if not has_trailing_comma and lines[sig_end_line - 1][:lines[sig_end_line - 1].rfind("):")].strip() == "":
        # Previous non-empty line did not end with comma; append comma to previous argument
        prev_idx = sig_end_line - 2
        while prev_idx >= fn.lineno - 1 and not re.sub(r"#.*$", "", lines[prev_idx]).strip():
            prev_idx -= 1
        code_part = re.sub(r"(\s*(?:#.*)?\r?\n)$", "", lines[prev_idx])
        tail_part = lines[prev_idx][len(code_part):]
        lines[prev_idx] = code_part + "," + tail_part
    sig_line_txt = lines[sig_end_line - 1]
    rparen = sig_line_txt.rfind("):")
    lines[sig_end_line - 1] = sig_line_txt[:rparen] + sep + extra_params_body + sig_line_txt[rparen:]

    insert_before: Dict[int, List[str]] = {}
    insert_after: Dict[int, List[str]] = {}

    def add_before(ln: int, txt: str) -> None:
        insert_before.setdefault(ln, []).append(txt)

    def add_after(ln: int, txt: str) -> None:
        insert_after.setdefault(ln, []).append(txt)

    # Inject import block right before the kernel's first decorator
    first_dec_line = fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno
    add_before(first_dec_line, import_block + "\n")

    # Inject OPEN right before K-loop
    add_before(k_for.lineno, open_lines)

    # Inside K-loop
    add_before(k_for.body[0].lineno, loop_head)
    if load_idxs:
        add_before(k_for.body[load_idxs[0]].lineno, load_b)
        add_after(k_for.body[load_idxs[-1]].end_lineno, load_e)
    add_before(k_for.body[dot_idxs[0]].lineno, mma_b)
    add_after(k_for.body[dot_idxs[-1]].end_lineno, mma_e)
    add_after(k_for.body[-1].end_lineno, loop_tail)

    # Store & CLOSE
    if store_stmts:
        add_before(store_stmts[0].lineno, store_b)
        add_after(store_stmts[-1].end_lineno, store_e)
    add_after(fn.body[-1].end_lineno, close_lines)

    out_lines: List[str] = []
    for idx, raw_line in enumerate(lines, start=1):
        for chunk in insert_before.get(idx, ()):
            out_lines.append(chunk)
        out_lines.append(raw_line)
        for chunk in insert_after.get(idx, ()):
            out_lines.append(chunk)
    instrumented = "".join(out_lines)
    # Verify generated Python syntax immediately!
    ast.parse(instrumented)

    names = build_names_doc(phases_used, card=card, arch=arch, language="triton", kernel_name=fn.name)
    plan = {
        "schema": "weaveprofiler.auto-gemm-plan/v1",
        "already_instrumented": False,
        "kernel": fn.name,
        "language": "triton",
        "platform": platform,
        "card": card,
        "arch": arch,
        "source_sha256": sha256_text(src),
        "instrumented_sha256": sha256_text(instrumented),
        "phases": phases_used,
        "k_loop_lines": [k_for.lineno, k_for.end_lineno],
    }
    return instrumented, names, plan


def instrument_cutedsl_gemm(
    src: str,
    *,
    card: str,
    arch: str,
    kernel_name: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Automatically analyze & instrument an uninstrumented @cute.kernel GEMM file."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)

    kernel_fns: List[ast.FunctionDef] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            is_cute_kernel = any(
                (isinstance(d, ast.Attribute) and d.attr in ("kernel", "jit"))
                or (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr in ("kernel", "jit"))
                for d in node.decorator_list
            )
            if is_cute_kernel and (kernel_name is None or node.name == kernel_name):
                has_loop = any(isinstance(st, ast.For) for st in node.body)
                if has_loop:
                    kernel_fns.append(node)

    if "fi_cute" in src or not kernel_fns:
        kname = kernel_fns[0].name if kernel_fns else (kernel_name or "cute_gemm_kernel")
        phases = ["gemm", "load", "wait", "mma", "store"]
        names = build_names_doc(phases, card=card, arch=arch, language="cutedsl", kernel_name=kname)
        plan = {
            "schema": "weaveprofiler.auto-gemm-plan/v1",
            "already_instrumented": ("fi_cute" in src),
            "kernel": kname,
            "language": "cutedsl",
            "card": card,
            "arch": arch,
            "source_sha256": sha256_text(src),
            "instrumented_sha256": sha256_text(src),
            "phases": phases,
        }
        return src, names, plan

    fn = kernel_fns[0]
    k_for: Optional[ast.For] = None
    for stmt in fn.body:
        if isinstance(stmt, ast.For):
            k_for = stmt
            break
    assert k_for is not None

    # Classify statements inside k_for.body
    mma_idxs = [
        i for i, st in enumerate(k_for.body)
        if _ast_contains_call(st, ("gemm", "fma"))
        or any(isinstance(sub, (ast.AugAssign, ast.For)) for sub in ast.walk(st) if i > 0)
    ]
    if not mma_idxs:
        mma_idxs = [len(k_for.body) - 1]
    first_mma = mma_idxs[0]
    wait_idxs = [
        i for i in range(0, first_mma)
        if _ast_contains_call(k_for.body[i], ("cp_async_wait_group", "barrier", "sync_threads"))
    ]
    load_limit = wait_idxs[0] if wait_idxs else first_mma
    load_idxs = [
        i for i in range(0, load_limit)
        if _ast_contains_call(k_for.body[i], ("copy",)) or any(isinstance(sub, ast.Subscript) for sub in ast.walk(k_for.body[i]))
    ]
    store_stmts = [
        st for st in fn.body
        if st.lineno > k_for.end_lineno
        and (_ast_contains_call(st, ("copy",)) or any(isinstance(sub, (ast.Subscript, ast.For)) for sub in ast.walk(st)))
    ]

    body_indent = " " * fn.body[0].col_offset
    loop_indent = " " * k_for.body[0].col_offset

    phases_used = ["gemm"]
    if load_idxs:
        phases_used.append("load")
    if wait_idxs:
        phases_used.append("wait")
    phases_used.append("mma")
    if store_stmts:
        phases_used.append("store")

    import_block = (
        "\n# --- Auto-injected by weaveprofiler auto_gemm_plan (CuTeDSL PTX %globaltimer) ---\n"
        "import os as _fi_os, sys as _fi_sys\n"
        "_fi_cute_dir = _fi_os.environ.get('WEAVEPROFILER_CUTEDSL_DIR', '')\n"
        "if _fi_cute_dir and _fi_cute_dir not in _fi_sys.path:\n"
        "    _fi_sys.path.insert(0, _fi_cute_dir)\n"
        "from fi_cute import FiBuffer as FiAutoBuffer, fi_open as _fi_open, "
        "fi_b as _fi_b, fi_e as _fi_e, fi_close as _fi_close\n"
    )
    extra_params_body = (
        "_fi_records: cute.Tensor, _fi_overflow: cute.Tensor, _fi_warps: cutlass.Int32, "
        "_fi_level: cutlass.Constexpr[int] = 0, _fi_gran: cutlass.Constexpr[int] = 1, "
        "_fi_sample: cutlass.Constexpr[int] = 0"
    )
    open_lines = (
        f"{body_indent}_fi_bx, _fi_by, _fi_bz = cute.arch.block_idx()\n"
        f"{body_indent}_fi_gx, _fi_gy, _ = cute.arch.grid_dim()\n"
        f"{body_indent}_fi_cta = _fi_bx + _fi_by * _fi_gx + _fi_bz * _fi_gx * _fi_gy\n"
        f"{body_indent}_fi_k_idx = cutlass.Int32(0)\n"
        f"{body_indent}_fi_ctx = _fi_open(_fi_records, _fi_overflow, cutlass.Int32(0), _fi_warps, fi_level=_fi_level, fi_gran=_fi_gran)\n"
        f"{body_indent}_fi_ctx = _fi_b(_fi_ctx, cutlass.Int32(10), _fi_cta, fi_level=_fi_level)\n"
    )
    loop_head = (
        f"{loop_indent}_fi_do_k = cutlass.Boolean(True)\n"
        f"{loop_indent}if cutlass.const_expr(_fi_sample == 1):\n"
        f"{loop_indent}    _fi_do_k = _fi_k_idx == cutlass.Int32(0)\n"
        f"{loop_indent}elif cutlass.const_expr(_fi_sample > 1):\n"
        f"{loop_indent}    _fi_do_k = (_fi_k_idx % cutlass.Int32(_fi_sample)) == cutlass.Int32(0)\n"
    )
    load_b = f"{loop_indent}if _fi_do_k:\n{loop_indent}    _fi_ctx = _fi_b(_fi_ctx, cutlass.Int32(1), _fi_k_idx, fi_level=_fi_level)\n"
    load_e = f"{loop_indent}if _fi_do_k:\n{loop_indent}    _fi_ctx = _fi_e(_fi_ctx, cutlass.Int32(1), _fi_k_idx, fi_level=_fi_level)\n"
    wait_b = f"{loop_indent}if _fi_do_k:\n{loop_indent}    _fi_ctx = _fi_b(_fi_ctx, cutlass.Int32(2), _fi_k_idx, fi_level=_fi_level)\n"
    wait_e = f"{loop_indent}if _fi_do_k:\n{loop_indent}    _fi_ctx = _fi_e(_fi_ctx, cutlass.Int32(2), _fi_k_idx, fi_level=_fi_level)\n"
    mma_b = f"{loop_indent}if _fi_do_k:\n{loop_indent}    _fi_ctx = _fi_b(_fi_ctx, cutlass.Int32(3), _fi_k_idx, fi_level=_fi_level)\n"
    mma_e = f"{loop_indent}if _fi_do_k:\n{loop_indent}    _fi_ctx = _fi_e(_fi_ctx, cutlass.Int32(3), _fi_k_idx, fi_level=_fi_level)\n"
    loop_tail = f"{loop_indent}_fi_k_idx = _fi_k_idx + cutlass.Int32(1)\n"
    store_b = f"{body_indent}_fi_ctx = _fi_b(_fi_ctx, cutlass.Int32(6), _fi_cta, fi_level=_fi_level)\n"
    store_e = f"{body_indent}_fi_ctx = _fi_e(_fi_ctx, cutlass.Int32(6), _fi_cta, fi_level=_fi_level)\n"
    close_lines = (
        f"{body_indent}_fi_ctx = _fi_e(_fi_ctx, cutlass.Int32(10), _fi_cta, fi_level=_fi_level)\n"
        f"{body_indent}_fi_ctx = _fi_close(_fi_ctx, fi_level=_fi_level)\n"
    )

    sig_end_line = fn.body[0].lineno - 1
    while sig_end_line >= fn.lineno and "):" not in lines[sig_end_line - 1]:
        sig_end_line -= 1
    if sig_end_line < fn.lineno:
        raise ValueError(f"Cannot locate closing '):' of {fn.name} signature")

    sig_prefix = "".join(lines[fn.lineno - 1:sig_end_line - 1]) + lines[sig_end_line - 1][:lines[sig_end_line - 1].rfind("):")]
    stripped_sig = re.sub(r"#[^\n]*", "", sig_prefix).rstrip()
    has_trailing_comma = stripped_sig.endswith(",") or stripped_sig.endswith("(")
    sep = "    " if lines[sig_end_line - 1][:lines[sig_end_line - 1].rfind("):")].strip() == "" else (", " if not has_trailing_comma else " ")
    if not has_trailing_comma and lines[sig_end_line - 1][:lines[sig_end_line - 1].rfind("):")].strip() == "":
        prev_idx = sig_end_line - 2
        while prev_idx >= fn.lineno - 1 and not re.sub(r"#.*$", "", lines[prev_idx]).strip():
            prev_idx -= 1
        code_part = re.sub(r"(\s*(?:#.*)?\r?\n)$", "", lines[prev_idx])
        tail_part = lines[prev_idx][len(code_part):]
        lines[prev_idx] = code_part + "," + tail_part
    sig_line_txt = lines[sig_end_line - 1]
    rparen = sig_line_txt.rfind("):")
    lines[sig_end_line - 1] = sig_line_txt[:rparen] + sep + extra_params_body + sig_line_txt[rparen:]

    insert_before: Dict[int, List[str]] = {}
    insert_after: Dict[int, List[str]] = {}

    def add_before(ln: int, txt: str) -> None:
        insert_before.setdefault(ln, []).append(txt)

    def add_after(ln: int, txt: str) -> None:
        insert_after.setdefault(ln, []).append(txt)

    first_dec_line = fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno
    add_before(first_dec_line, import_block + "\n")
    add_before(k_for.lineno, open_lines)
    add_before(k_for.body[0].lineno, loop_head)
    if load_idxs:
        add_before(k_for.body[load_idxs[0]].lineno, load_b)
        add_after(k_for.body[load_idxs[-1]].end_lineno, load_e)
    if wait_idxs:
        add_before(k_for.body[wait_idxs[0]].lineno, wait_b)
        add_after(k_for.body[wait_idxs[-1]].end_lineno, wait_e)
    add_before(k_for.body[mma_idxs[0]].lineno, mma_b)
    add_after(k_for.body[mma_idxs[-1]].end_lineno, mma_e)
    add_after(k_for.body[-1].end_lineno, loop_tail)
    if store_stmts:
        add_before(store_stmts[0].lineno, store_b)
        add_after(store_stmts[-1].end_lineno, store_e)
    add_after(fn.body[-1].end_lineno, close_lines)

    out_lines: List[str] = []
    for idx, raw_line in enumerate(lines, start=1):
        for chunk in insert_before.get(idx, ()):
            out_lines.append(chunk)
        out_lines.append(raw_line)
        for chunk in insert_after.get(idx, ()):
            out_lines.append(chunk)
    instrumented = "".join(out_lines)
    ast.parse(instrumented)

    names = build_names_doc(phases_used, card=card, arch=arch, language="cutedsl", kernel_name=fn.name)
    plan = {
        "schema": "weaveprofiler.auto-gemm-plan/v1",
        "already_instrumented": False,
        "kernel": fn.name,
        "language": "cutedsl",
        "platform": "nv",
        "card": card,
        "arch": arch,
        "source_sha256": sha256_text(src),
        "instrumented_sha256": sha256_text(instrumented),
        "phases": phases_used,
        "k_loop_lines": [k_for.lineno, k_for.end_lineno],
    }
    return instrumented, names, plan


def _write_lf(path: Path, text: str) -> None:
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def plan_and_instrument(
    source_path: Path,
    out_dir: Path,
    *,
    card: str,
    arch: Optional[str] = None,
    kernel_name: Optional[str] = None,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    card_key = card.lower().strip()
    if card_key not in CARD_DEFAULTS:
        raise ValueError(f"Unsupported card {card!r}; expected one of {sorted(CARD_DEFAULTS)}")
    resolved_arch = arch or CARD_DEFAULTS[card_key]["arch"]
    src = source_path.read_text(encoding="utf-8")
    language = detect_language(source_path, src)

    out_dir.mkdir(parents=True, exist_ok=True)
    if language in ("cuda_cpp", "hip_cpp"):
        instrumented, names, plan = instrument_cpp_gemm(
            src, language=language, card=card_key, arch=resolved_arch, kernel_name=kernel_name
        )
    elif language == "triton":
        instrumented, names, plan = instrument_triton_gemm(
            src, card=card_key, arch=resolved_arch, kernel_name=kernel_name
        )
    elif language == "cutedsl":
        instrumented, names, plan = instrument_cutedsl_gemm(
            src, card=card_key, arch=resolved_arch, kernel_name=kernel_name
        )
    else:
        raise ValueError(f"Unsupported language {language}")

    inst_path = out_dir / f"instrumented_{source_path.name}"
    names_path = out_dir / "names.json"
    plan_path = out_dir / "auto_plan.json"
    _write_lf(inst_path, instrumented)
    _write_lf(names_path, json.dumps(names, indent=2, ensure_ascii=False) + "\n")
    plan["planning_elapsed_ms"] = round((time.monotonic() - t0) * 1000.0, 3)
    plan["instrumented_source"] = str(inst_path)
    plan["names_path"] = str(names_path)
    _write_lf(plan_path, json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    return plan


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Automatic cross-language GEMM planner & instrumenter")
    parser.add_argument("--source", required=True, help="Uninstrumented GEMM kernel source (.cu, .hip, .py)")
    parser.add_argument("--card", required=True, choices=sorted(CARD_DEFAULTS))
    parser.add_argument("--arch", help="Explicit target arch (e.g. sm_90a, sm_103a, gfx942)")
    parser.add_argument("--kernel", help="Optional target kernel function name")
    parser.add_argument("--out-dir", required=True, help="Directory for instrumented source, names.json, and auto_plan.json")
    args = parser.parse_args(argv)
    plan = plan_and_instrument(
        Path(args.source).expanduser().resolve(),
        Path(args.out_dir).expanduser().resolve(),
        card=args.card,
        arch=args.arch,
        kernel_name=args.kernel,
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
