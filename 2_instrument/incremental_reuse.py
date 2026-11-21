#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§3.2 (2c) Diff-Aware Context-Hash Incremental Probe Reuse & Build Cache Manager.

Given ``--prev-source``, ``--prev-spec`` (previous verified ``probe_spec.json``), and
``--new-source`` (edited kernel source during iterative optimization):
  1. Computes normalized statement context hashes + AST/structural block hierarchy in <0.05s.
  2. Hyperparameter edits (``BM, BN, BK, NUM_STAGES``) and unaffected regions directly inherit
     verified probe placements in **0 seconds** with **0 LLM calls**.
  3. Localized ``git diff`` code edits invoke ``1_scan/llm_probe_placer.py`` **strictly on modified
     diff hunks** when a hunk alters/introduces stage primitives, merging updated hunk probes with
     the deterministically inherited probes.
  4. Provides ``BuildArtifactCache`` to reuse cached ``L0`` baseline binaries and disassemblies.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

SKILL_ROOT = Path(__file__).resolve().parents[1]
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))


def _load_probe_placer():
    p = SKILL_ROOT / "1_scan" / "llm_probe_placer.py"
    spec = importlib.util.spec_from_file_location("wp_llm_probe_placer", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_STAGE_PRIMITIVE_CUES = re.compile(
    r"\b(cp\.async|cp_async|memcpy_async|tma_load|tma_store|wgmma|tcgen05|mma_sync|"
    r"__builtin_amdgcn_mfma|tl\.dot|tl\.load|tl\.store|cute\.gemm|cute\.copy|"
    r"mbarrier|wait_group|try_wait|__syncthreads|s_waitcnt|s_barrier|atomicAdd)\b"
)


def _normalize_code_line(line: str) -> str:
    s = re.sub(r"//.*$", "", line)
    if s.lstrip().startswith("#") and not s.lstrip().startswith("#pragma") and not s.lstrip().startswith("#include"):
        s = ""
    elif " #" in s:
        s = re.sub(r"\s+#.*$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _normalize_param_insensitive(line: str) -> str:
    s = _normalize_code_line(line)
    return re.sub(r"\b\d+[uUlL]*\b", "<N>", s)


def _context_hash(norm_lines: Sequence[str], idx_0: int, radius: int = 2) -> str:
    lo = max(0, idx_0 - radius)
    hi = min(len(norm_lines), idx_0 + radius + 1)
    window = "\n".join(norm_lines[lo:hi])
    return hashlib.sha1(window.encode("utf-8")).hexdigest()[:16]


def _extract_python_ast_blocks(source_text: str) -> List[Tuple[int, int, str]]:
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return []
    blocks: List[Tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.For, ast.While, ast.If)):
            end_ln = getattr(node, "end_lineno", node.lineno) or node.lineno
            label = f"{type(node).__name__}:{getattr(node, 'name', '')}"
            blocks.append((node.lineno, end_ln, label))
    blocks.sort()
    return blocks


def _extract_cpp_structural_blocks(source_text: str) -> List[Tuple[int, int, str]]:
    lines = source_text.splitlines()
    blocks: List[Tuple[int, int, str]] = []
    stack: List[Tuple[int, str]] = []
    pending_kw = "block"
    for ln, raw in enumerate(lines, start=1):
        clean = _normalize_code_line(raw)
        m = re.search(r"\b(__global__|__device__|for|while|if|else)\b", clean)
        if m:
            pending_kw = m.group(1)
        for ch in clean:
            if ch == "{":
                stack.append((ln, pending_kw))
                pending_kw = "block"
            elif ch == "}" and stack:
                start_ln, kw = stack.pop()
                blocks.append((start_ln, ln, kw))
    blocks.sort()
    return blocks


def _enclosing_block_sig(blocks: Sequence[Tuple[int, int, str]], line_no: int) -> str:
    enclosing = [b[2] for b in blocks if b[0] <= line_no <= b[1]]
    return "/".join(enclosing[-3:]) if enclosing else "root"


def _map_line_via_diff_and_context(
    old_line_1: int,
    snippet_text: str,
    old_norm: Sequence[str],
    new_norm: Sequence[str],
    old_param_norm: Sequence[str],
    new_param_norm: Sequence[str],
    old_blocks: Sequence[Tuple[int, int, str]],
    new_blocks: Sequence[Tuple[int, int, str]],
    opcodes: Sequence[Tuple[str, int, int, int, int]],
) -> Tuple[Optional[int], str]:
    if not new_norm:
        return None, "empty_new_source"
    idx0 = max(0, min(len(old_norm) - 1, old_line_1 - 1))
    old_exact = old_norm[idx0]
    old_pnorm = old_param_norm[idx0]
    snip_clean = _normalize_code_line(snippet_text or "")
    old_ctx = _context_hash(old_param_norm, idx0, radius=2)
    old_blk = _enclosing_block_sig(old_blocks, old_line_1)

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal" and i1 <= idx0 < i2:
            return (j1 + (idx0 - i1)) + 1, "diff_equal"

    for tag, i1, i2, j1, j2 in opcodes:
        if tag in ("replace", "delete") and i1 <= idx0 < i2:
            for cand_0 in range(j1, j2):
                if new_norm[cand_0] and (
                    new_norm[cand_0] == old_exact
                    or (snip_clean and snip_clean in new_norm[cand_0])
                    or (old_pnorm and new_param_norm[cand_0] == old_pnorm and len(old_pnorm) >= 6)
                ):
                    if _enclosing_block_sig(new_blocks, cand_0 + 1) == old_blk or _context_hash(new_param_norm, cand_0, 1) == _context_hash(old_param_norm, idx0, 1):
                        return cand_0 + 1, "hunk_param_normalized"
                    return cand_0 + 1, "hunk_snippet_match"

    candidates: List[Tuple[int, int]] = []
    for cand_0, nline in enumerate(new_norm):
        if not nline:
            continue
        exact_ok = (nline == old_exact) or (snip_clean and snip_clean in nline)
        param_ok = bool(old_pnorm and len(old_pnorm) >= 8 and new_param_norm[cand_0] == old_pnorm)
        if not (exact_ok or param_ok):
            continue
        score = 0
        if nline == old_exact:
            score += 20
        if _context_hash(new_param_norm, cand_0, radius=2) == old_ctx:
            score += 30
        if _enclosing_block_sig(new_blocks, cand_0 + 1) == old_blk:
            score += 15
        score -= min(14, abs(cand_0 - idx0) // 4)
        candidates.append((score, cand_0))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_0 = candidates[0]
        if best_score >= 15:
            return best_0 + 1, "ast_context_hash"

    return None, "unmapped"


def _collect_modified_hunks(
    prev_lines: Sequence[str],
    new_lines: Sequence[str],
    opcodes: Sequence[Tuple[str, int, int, int, int]],
    *,
    context_pad: int = 8,
) -> List[Dict[str, Any]]:
    hunks: List[Dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        added_text = "\n".join(new_lines[j1:j2])
        removed_text = "\n".join(prev_lines[i1:i2])
        lo = max(0, j1 - context_pad)
        hi = min(len(new_lines), j2 + context_pad)
        numbered_window = "\n".join(
            f"{ln:4d}: {new_lines[ln - 1]}" for ln in range(lo + 1, hi + 1)
        )
        hunks.append({
            "tag": tag,
            "prev_range": [i1 + 1, max(i1 + 1, i2)],
            "new_range": [j1 + 1, max(j1 + 1, j2)],
            "window_range": [lo + 1, hi],
            "added_stage_primitives": bool(_STAGE_PRIMITIVE_CUES.search(added_text)),
            "removed_stage_primitives": bool(_STAGE_PRIMITIVE_CUES.search(removed_text)),
            "numbered_new_window": numbered_window,
            "removed_snippet": removed_text[:600],
            "added_snippet": added_text[:600],
        })
    return hunks


def _run_delta_llm_update(
    placer_mod: Any,
    *,
    prev_spec: Mapping[str, Any],
    inherited_probes: List[Dict[str, Any]],
    unmapped_probes: List[Dict[str, Any]],
    relevant_hunks: List[Dict[str, Any]],
    model: str,
    reasoning_effort: str,
    llm_caller: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    caller = llm_caller or placer_mod.call_bailian_llm
    sys_msg = (
        "You are WeaveProfiler's §3.1 Semantic Probe Placement Agent operating in Incremental Hunk Mode.\n"
        "Unchanged AST blocks have ALREADY inherited their verified probe placements in 0 seconds. "
        "Inspect ONLY the modified diff hunks and output a compact JSON object:\n"
        "{\"updated_probes\": [...], \"warp_roles\": [...]}\n"
        "containing updated probe placements ONLY for the unmapped/modified probes or newly added stage primitives inside the diff hunk. "
        "Use canonical semantic_ids: [\"load\", \"wait\", \"mma\", \"compute\", \"barrier\", \"store\", \"epilogue\", \"dispatch\", \"communication\"], "
        "keep `wait` outside polling spin loops, and pair async launch->wait properly."
    )
    user_msg = json.dumps({
        "language": prev_spec.get("language"),
        "card": prev_spec.get("card"),
        "arch": prev_spec.get("arch"),
        "kernel_name": prev_spec.get("kernel_name"),
        "already_inherited_probes": [
            {
                "probe_id": s["probe_id"],
                "semantic_id": s["semantic_id"],
                "role": s["role"],
                "begin_line": s["begin_line"],
                "end_line": s["end_line"],
            }
            for s in inherited_probes
        ],
        "probes_needing_hunk_remap": unmapped_probes,
        "modified_diff_hunks": relevant_hunks,
        "existing_warp_roles": prev_spec.get("warp_roles"),
    }, ensure_ascii=False, indent=2)

    res = caller(
        model,
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        reasoning_effort=reasoning_effort,
    )
    parsed = placer_mod.extract_json_object(res["content"])
    delta_probes = list(parsed.get("updated_probes") or parsed.get("probes") or [])
    delta_roles = list(parsed.get("warp_roles") or prev_spec.get("warp_roles") or [])
    telemetry = {
        "mode": "incremental_delta_llm",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "elapsed_s": res["elapsed_s"],
        **res.get("usage", {}),
    }
    return delta_probes, delta_roles, telemetry


def reuse_probes_incremental(
    prev_source_path: Path,
    prev_spec_path: Path,
    new_source_path: Path,
    *,
    allow_delta_llm: bool = True,
    model: str = "qwen4-max",
    reasoning_effort: str = "max",
    llm_caller: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Incrementally inherit verified probes from `prev_spec` onto `new_source` via diff + context hashes."""
    t0 = time.perf_counter()
    placer = _load_probe_placer()

    prev_text = prev_source_path.read_text(encoding="utf-8")
    new_text = new_source_path.read_text(encoding="utf-8")
    prev_spec = json.loads(prev_spec_path.read_text(encoding="utf-8"))

    lang = str(prev_spec.get("language") or placer.detect_language(new_source_path, new_text))
    card = str(prev_spec.get("card") or "h20")
    arch = str(prev_spec.get("arch") or "")
    route = placer.resolve_route(card, arch or None, lang)

    if prev_text == new_text:
        elapsed = round(time.perf_counter() - t0, 5)
        bound = dict(prev_spec)
        bound["source_path"] = str(new_source_path)
        bound["source_sha256"] = placer.sha256_text(new_text)
        bound["telemetry"] = {
            "mode": "incremental_identical",
            "elapsed_s": elapsed,
            "llm_elapsed_s": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "inherited_probes": len(bound.get("probes") or []),
            "delta_llm_probes": 0,
        }
        return {"ok": True, "probe_spec": bound, "used_delta_llm": False, "elapsed_s": elapsed}

    prev_lines = prev_text.splitlines()
    new_lines = new_text.splitlines()
    prev_norm = [_normalize_code_line(l) for l in prev_lines]
    new_norm = [_normalize_code_line(l) for l in new_lines]
    prev_pnorm = [_normalize_param_insensitive(l) for l in prev_lines]
    new_pnorm = [_normalize_param_insensitive(l) for l in new_lines]

    if lang in ("triton", "cutedsl"):
        old_blocks = _extract_python_ast_blocks(prev_text)
        new_blocks = _extract_python_ast_blocks(new_text)
    else:
        old_blocks = _extract_cpp_structural_blocks(prev_text)
        new_blocks = _extract_cpp_structural_blocks(new_text)

    sm = difflib.SequenceMatcher(a=prev_pnorm, b=new_pnorm, autojunk=False)
    opcodes = sm.get_opcodes()
    hunks = _collect_modified_hunks(prev_lines, new_lines, opcodes)

    remapped_roles: List[Dict[str, Any]] = []
    for role in prev_spec.get("warp_roles") or []:
        r_copy = dict(role)
        if r_copy.get("begin_line"):
            nb, _ = _map_line_via_diff_and_context(
                int(r_copy["begin_line"]), "", prev_norm, new_norm, prev_pnorm, new_pnorm, old_blocks, new_blocks, opcodes
            )
            if nb is not None:
                r_copy["begin_line"] = nb
        if r_copy.get("end_line"):
            ne, _ = _map_line_via_diff_and_context(
                int(r_copy["end_line"]), "", prev_norm, new_norm, prev_pnorm, new_pnorm, old_blocks, new_blocks, opcodes
            )
            if ne is not None:
                r_copy["end_line"] = ne
        remapped_roles.append(r_copy)

    inherited_probes: List[Dict[str, Any]] = []
    unmapped_probes: List[Dict[str, Any]] = []
    mapping_audit: List[Dict[str, Any]] = []

    for st in prev_spec.get("probes") or prev_spec.get("stages") or []:
        old_b = int(st.get("begin_line") or 1)
        old_e = int(st.get("end_line") or old_b)
        b_snip = str(st.get("begin_snippet") or "")
        e_snip = str(st.get("end_snippet") or "")

        new_b, b_method = _map_line_via_diff_and_context(
            old_b, b_snip, prev_norm, new_norm, prev_pnorm, new_pnorm, old_blocks, new_blocks, opcodes
        )
        new_e, e_method = _map_line_via_diff_and_context(
            old_e, e_snip, prev_norm, new_norm, prev_pnorm, new_pnorm, old_blocks, new_blocks, opcodes
        )

        new_cw = None
        if st.get("completion_wait_line"):
            new_cw, _ = _map_line_via_diff_and_context(
                int(st["completion_wait_line"]),
                str(st.get("completion_wait_snippet") or ""),
                prev_norm, new_norm, prev_pnorm, new_pnorm, old_blocks, new_blocks, opcodes,
            )

        pid = st.get("probe_id") or st.get("stage_id")
        if new_b is not None and new_e is not None and new_e >= new_b:
            st_new = dict(st)
            st_new["probe_id"] = pid
            st_new["begin_line"] = new_b
            st_new["end_line"] = new_e
            st_new["begin_snippet"] = new_lines[new_b - 1].strip()
            st_new["end_snippet"] = new_lines[new_e - 1].strip()
            if new_cw is not None:
                st_new["completion_wait_line"] = new_cw
                st_new["completion_wait_snippet"] = new_lines[new_cw - 1].strip()
            inherited_probes.append(st_new)
            mapping_audit.append({
                "probe_id": pid,
                "status": "inherited",
                "old_lines": [old_b, old_e],
                "new_lines": [new_b, new_e],
                "methods": [b_method, e_method],
            })
        else:
            unmapped_probes.append(dict(st))
            mapping_audit.append({
                "probe_id": pid,
                "status": "needs_delta_llm",
                "old_lines": [old_b, old_e],
                "methods": [b_method, e_method],
            })

    uncovered_primitive_hunks: List[Dict[str, Any]] = []
    for h in hunks:
        if not h["added_stage_primitives"] or h["removed_stage_primitives"]:
            continue
        hr_b, hr_e = h["new_range"]
        covered = any((s["begin_line"] - 2) <= hr_b and hr_e <= (s["end_line"] + 2) for s in inherited_probes)
        if not covered:
            uncovered_primitive_hunks.append(h)

    need_delta = bool(unmapped_probes or uncovered_primitive_hunks) and allow_delta_llm
    llm_telemetry: Dict[str, Any] = {
        "llm_elapsed_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    final_probes = list(inherited_probes)

    if need_delta:
        delta_probes, delta_roles, delta_tel = _run_delta_llm_update(
            placer,
            prev_spec=prev_spec,
            inherited_probes=inherited_probes,
            unmapped_probes=unmapped_probes,
            relevant_hunks=hunks[:6],
            model=model,
            reasoning_effort=reasoning_effort,
            llm_caller=llm_caller,
        )
        final_probes.extend(delta_probes)
        if delta_roles:
            remapped_roles = delta_roles
        llm_telemetry = {
            "llm_elapsed_s": delta_tel.get("elapsed_s", 0.0),
            "prompt_tokens": delta_tel.get("prompt_tokens", 0),
            "completion_tokens": delta_tel.get("completion_tokens", 0),
            "total_tokens": delta_tel.get("total_tokens", 0),
            "model": model,
            "reasoning_effort": reasoning_effort,
        }

    total_elapsed = round(time.perf_counter() - t0, 5)
    telemetry = {
        "mode": "incremental_delta_llm" if need_delta else "incremental_zero_llm",
        "elapsed_s": total_elapsed,
        "ast_diff_elapsed_s": round(max(0.0, total_elapsed - float(llm_telemetry["llm_elapsed_s"])), 5),
        "inherited_probes": len(inherited_probes),
        "unmapped_probes": len(unmapped_probes),
        "delta_llm_probes": len(final_probes) - len(inherited_probes),
        "diff_hunks_total": len(hunks),
        **llm_telemetry,
    }

    raw_merged = {
        "kernel_name": prev_spec.get("kernel_name"),
        "operator_family": prev_spec.get("operator_family", "gemm"),
        "granularity": prev_spec.get("granularity", "sm"),
        "warp_roles": remapped_roles,
        "probes": final_probes,
    }
    bound_spec = placer.sanitize_and_bind_probe_spec(
        raw_merged,
        source_text=new_text,
        source_path=new_source_path,
        route=route,
        telemetry=telemetry,
        kb_provenance=prev_spec.get("kb_provenance"),
    )
    bound_spec["incremental_provenance"] = {
        "prev_source_sha256": placer.sha256_text(prev_text),
        "new_source_sha256": bound_spec["source_sha256"],
        "mapping_audit": mapping_audit,
    }
    return {
        "ok": True,
        "probe_spec": bound_spec,
        "used_delta_llm": need_delta,
        "elapsed_s": total_elapsed,
        "telemetry": telemetry,
        "mapping_audit": mapping_audit,
    }


class BuildArtifactCache:
    """Content-addressed cache for `L0` baseline binaries, SASS/GCN disassemblies, and timings."""

    def __init__(self, cache_root: Path):
        self.root = cache_root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_key(
        *,
        source_text: str,
        card: str,
        arch: str,
        language: str,
        variant: str = "L0",
        extra_flags: str = "",
    ) -> str:
        payload = json.dumps({
            "src_sha": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "card": card,
            "arch": arch,
            "language": language,
            "variant": variant,
            "flags": extra_flags,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def lookup(self, key: str) -> Optional[Dict[str, Any]]:
        entry_dir = self.root / key
        meta_path = entry_dir / "cache_meta.json"
        if not meta_path.is_file():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        artifact = meta.get("binary_name")
        if artifact and not (entry_dir / artifact).is_file():
            return None
        meta["cache_dir"] = str(entry_dir)
        if artifact:
            meta["binary_path"] = str(entry_dir / artifact)
        return meta

    def store(self, key: str, *, binary_path: Optional[Path], metadata: Mapping[str, Any]) -> Dict[str, Any]:
        entry_dir = self.root / key
        entry_dir.mkdir(parents=True, exist_ok=True)
        saved_meta = dict(metadata)
        if binary_path and binary_path.is_file():
            dst = entry_dir / binary_path.name
            if dst.resolve() != binary_path.resolve():
                shutil.copy2(binary_path, dst)
            saved_meta["binary_name"] = binary_path.name
            saved_meta["binary_sha256"] = hashlib.sha256(dst.read_bytes()).hexdigest()
        (entry_dir / "cache_meta.json").write_text(
            json.dumps(saved_meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        saved_meta["cache_dir"] = str(entry_dir)
        return saved_meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="§3.2 (2c) Diff-Hash Incremental Probe Reuse across local GPU kernel edits")
    ap.add_argument("--prev-source", required=True, help="Previous uninstrumented kernel source path")
    ap.add_argument("--prev-spec", required=True, help="Previous verified JSON Probe Spec path")
    ap.add_argument("--new-source", "--source", dest="new_source", required=True, help="Edited kernel source path")
    ap.add_argument("--out-spec", "--out", dest="out_spec", required=True, help="Output inherited JSON Probe Spec")
    ap.add_argument("--no-delta-llm", action="store_true", help="Disable fallback delta-LLM even on unmapped hunks")
    ap.add_argument("--model", default="qwen4-max")
    ap.add_argument("--reasoning-effort", default="max")
    args = ap.parse_args(list(argv) if argv is not None else None)

    res = reuse_probes_incremental(
        Path(args.prev_source).expanduser().resolve(),
        Path(args.prev_spec).expanduser().resolve(),
        Path(args.new_source).expanduser().resolve(),
        allow_delta_llm=(not args.no_delta_llm),
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    out_path = Path(args.out_spec).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res["probe_spec"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "out_spec": str(out_path),
        "used_delta_llm": res["used_delta_llm"],
        "elapsed_s": res["elapsed_s"],
        "telemetry": res["telemetry"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
