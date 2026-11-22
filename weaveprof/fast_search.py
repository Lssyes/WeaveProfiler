"""RSI (Recursive Self-Improving) Fast Placement & Recorder Search Engine.

Cuts end-to-end instrumentation wall time from ~30 min manual trial-and-error to <5-10 min by:
  1. Minimal include-overlay (`build_minimal_include_overlay`): copy ONLY the 1-2 modified
     headers into a tiny `-I` overlay directory instead of duplicating hundreds of headers.
  2. Static stage-completeness & disjoint-sibling pre-filter (`detect_kernel_stage_requirements`):
     prevents missing `load`/`mma`/`softmax`/`epilogue` or nested macro-wrappers BEFORE compiling.
  3. Parallel CPU compile + SASS/spill pre-screen (`run_rsi_placement_search`): compiles candidate
     `(placement_strategy, recorder_mode)` arms concurrently on CPU cores and prunes spilling or
     broken arms before touching the GPU.
  4. Persistent local RSI prior learning: records which `(arch, operator_family, strategy, mode)`
     arms pass cleanly with lowest overhead so subsequent rounds & kernels converge on Round 1.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .contract import resolve_semantic_id
from .harness import verify_compiler_spill_gate
from .workspace import read_json, write_json

RSI_SEARCH_SCHEMA = "weaveprof.rsi-placement-search/v1"
RSI_PRIOR_SCHEMA = "weaveprof.rsi-priors/v1"

_ANCHOR_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("load", re.compile(r"\b(?:produce_kv|cp_async|cp\.async|tma_load|load_tile|tl\.load|cute\.copy)\b", re.I)),
    ("mma", re.compile(r"\b(?:compute_qk|compute_sv|wgmma|tcgen05|mma\.sync|wmma::mma|mfma|tl\.dot|cute\.gemm)\b", re.I)),
    ("softmax", re.compile(r"\b(?:update_mdo_states|finalize_m|online_softmax|softmax_rescale|softmax)\b", re.I)),
    ("epilogue", re.compile(r"\b(?:write_o_reg_gmem|vosplit_write_o|threadblock_sync_mdo_states|epilogue|tma_store|tl\.store)\b", re.I)),
    ("wait", re.compile(r"\b(?:cp_async::wait_group|mbarrier[\w.]*wait|pipeline\w*\.consumer_wait|warpgroup_wait|wgmma\.wait)\b", re.I)),
    ("communication", re.compile(r"\b(?:nvshmem\w*|nccl\w*|rccl\w*|allreduce\w*|alltoall\w*|multimem\w*)\b", re.I)),
)
_ATTENTION_RE = re.compile(
    r"\b(?:SinglePrefill|BatchPrefill|SingleDecode|BatchDecode|flash_?attn|fmha|update_mdo_states|finalize_m|online_softmax|softmax_rescale|mla_decode)\b",
    re.I,
)
_MOE_RE = re.compile(r"\b(?:mega_?moe|moe_?fused|topk_reduce|expert_id|dispatch_rank)\b", re.I)


def default_rsi_prior_path() -> Path:
    configured = os.environ.get("WEAVEPROF_RSI_PRIORS")
    if configured:
        return Path(configured).expanduser().resolve()
    base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "weaveprof"
    return (base / "rsi_priors.json").resolve()


def build_minimal_include_overlay(
    orig_include_root: str | Path,
    modified_files: Mapping[str, str],
    overlay_dir: str | Path,
) -> Dict[str, Any]:
    """Write ONLY the modified header/source files into a minimal `-I` overlay tree.

    Compile with `-I <overlay_dir> -I <orig_include_root>` so nvcc/hipcc resolves modified
    headers from `<overlay_dir>` and all untouched headers directly from `<orig_include_root>`
    without copying hundreds of megabytes of upstream headers.
    """
    orig = Path(orig_include_root).expanduser().resolve()
    overlay = Path(overlay_dir).expanduser().resolve()
    overlay.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for rel, content in sorted(modified_files.items()):
        rel_norm = Path(rel)
        if rel_norm.is_absolute() or ".." in rel_norm.parts:
            raise ValueError(f"Overlay relative path escapes overlay root: {rel!r}")
        dst = (overlay / rel_norm).resolve()
        if not dst.is_relative_to(overlay):
            raise ValueError(f"Overlay relative path escapes overlay root: {rel!r}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(str(content).replace("\r\n", "\n").encode("utf-8"))
        written.append(rel_norm.as_posix())
    return {
        "overlay_dir": str(overlay),
        "orig_include_root": str(orig),
        "include_flags": ["-I", str(overlay), "-I", str(orig)],
        "files_overlaid": written,
    }


def detect_kernel_stage_requirements(
    source_text: str,
    *,
    kernel_name: str = "",
    kernel_class: str = "",
) -> Dict[str, Any]:
    """Deterministically infer operator family, required semantic stages, and code anchor lines."""
    text = source_text or ""
    lines = text.splitlines()
    anchors: Dict[str, List[int]] = {k: [] for k, _ in _ANCHOR_PATTERNS}
    for lineno, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped.startswith(("//", "#", "/*", "*")):
            continue
        for stage, rx in _ANCHOR_PATTERNS:
            if rx.search(line) and len(anchors[stage]) < 16:
                anchors[stage].append(lineno)

    kc = (kernel_class or "").strip().lower()
    if kc:
        op_family = kc
    elif _ATTENTION_RE.search(text) or _ATTENTION_RE.search(kernel_name or ""):
        op_family = "attention"
    elif _MOE_RE.search(text) or _MOE_RE.search(kernel_name or ""):
        op_family = "moe"
    elif anchors["mma"]:
        op_family = "gemm"
    else:
        op_family = "non_gemm"

    required: List[str] = []
    if op_family == "attention":
        required = ["load", "mma", "softmax"]
        if anchors["epilogue"]:
            required.append("store_or_epilogue")
    elif op_family in ("gemm", "moe"):
        required = ["load", "mma"]
        if anchors["epilogue"]:
            required.append("store_or_epilogue")
        if anchors["communication"]:
            required.append("communication")
    else:
        required = ["compute"]

    return {
        "operator_family": op_family,
        "required_stages": required,
        "anchors": {k: v for k, v in anchors.items() if v},
    }


def check_candidate_static_coverage(
    candidate: Mapping[str, Any],
    requirements: Mapping[str, Any],
) -> Tuple[bool, List[str]]:
    """Fast (<1 ms) static check before compiling: reject missing required stages or outer wrappers."""
    issues: List[str] = []
    raw_stages = list(candidate.get("stages") or [])
    if not raw_stages and isinstance(candidate.get("probes"), Sequence):
        raw_stages = [
            str(p.get("semantic_id") or p.get("id") or p.get("probe_id") or "")
            for p in candidate["probes"]
            if isinstance(p, Mapping)
        ]
    sems = {resolve_semantic_id(s) for s in raw_stages if str(s).strip()}
    for req in requirements.get("required_stages") or []:
        if req == "store_or_epilogue":
            if not (sems & {"store", "epilogue"}):
                issues.append("MISSING_REQUIRED_STAGE: epilogue/store")
        elif req == "compute":
            if not (sems & {"compute", "mma", "softmax", "reduction", "load", "store", "epilogue"}):
                issues.append("MISSING_REQUIRED_STAGE: active compute/memory stage")
        elif req not in sems:
            issues.append(f"MISSING_REQUIRED_STAGE: {req}")
    if candidate.get("wraps_entire_kernel") is True:
        issues.append("NESTED_WRAPPER_FORBIDDEN: candidate wraps entire kernel in an outer macro-probe")
    return not issues, issues


def _load_priors(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"schema": RSI_PRIOR_SCHEMA, "stats": {}}
    try:
        doc = read_json(path)
        if isinstance(doc, dict) and isinstance(doc.get("stats"), dict):
            return doc
    except (OSError, ValueError):
        pass
    return {"schema": RSI_PRIOR_SCHEMA, "stats": {}}


def _prior_key(arch: str, op_family: str, strategy: str, mode: str) -> str:
    return f"{(arch or 'generic').lower()}::{(op_family or 'generic').lower()}::{strategy}::{mode}"


def _prior_score(priors: Mapping[str, Any], key: str) -> float:
    st = (priors.get("stats") or {}).get(key)
    if not isinstance(st, Mapping):
        return 0.5  # optimistic neutral prior for unseen arms
    trials = max(1, int(st.get("trials") or 0))
    passes = int(st.get("passes") or 0)
    # Laplace-smoothed pass rate minus small overhead penalty
    pass_rate = (passes + 1.0) / (trials + 2.0)
    ovh = st.get("best_overhead_pct")
    ovh_pen = min(0.25, max(0.0, float(ovh)) / 100.0) if isinstance(ovh, (int, float)) else 0.05
    return pass_rate - ovh_pen


def _update_priors(path: Path, priors: Dict[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
    stats = dict(priors.get("stats") or {})
    for rec in records:
        key = str(rec["prior_key"])
        cur = dict(stats.get(key) or {"trials": 0, "passes": 0, "spill_clean": 0, "best_overhead_pct": None})
        cur["trials"] = int(cur.get("trials") or 0) + 1
        if rec.get("passed"):
            cur["passes"] = int(cur.get("passes") or 0) + 1
        if rec.get("compile_ok"):
            cur["spill_clean"] = int(cur.get("spill_clean") or 0) + 1
        ovh = rec.get("overhead_pct")
        if rec.get("passed") and isinstance(ovh, (int, float)):
            prev = cur.get("best_overhead_pct")
            cur["best_overhead_pct"] = round(float(ovh), 3) if prev is None else round(min(float(prev), float(ovh)), 3)
        cur["last_updated"] = round(time.time(), 3)
        stats[key] = cur
    priors["schema"] = RSI_PRIOR_SCHEMA
    priors["stats"] = stats
    try:
        write_json(path, priors)
    except OSError:
        pass


def run_rsi_placement_search(
    source_text: str,
    candidates: Sequence[Mapping[str, Any]],
    compile_fn: Callable[[Dict[str, Any]], Mapping[str, Any]],
    screen_gpu_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Mapping[str, Any]]] = None,
    *,
    baseline_metrics: Mapping[str, Any],
    arch: str = "",
    kernel_name: str = "",
    kernel_class: str = "",
    max_overhead_pct: float = 10.0,
    max_workers: int = 4,
    prior_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Two-stage parallel RSI search over `(placement_strategy, recorder_mode)` candidates.

    1. Ranks candidates using persistent local RSI priors (`rsi_priors.json`).
    2. Statically rejects candidates missing required operator stages or using outer wrappers.
    3. Compiles surviving candidates IN PARALLEL on CPU cores and checks `verify_compiler_spill_gate`
       plus scheduling-aware core-opcode retention before touching the GPU.
    4. Screens only compile-clean candidates on GPU (`screen_gpu_fn`), stopping on the first arm
       that satisfies all gates and `abs(overhead_pct) <= max_overhead_pct`.
    5. Persists outcomes back into `rsi_priors.json` so future runs try winning strategies first.
    """
    t0 = time.perf_counter()
    reqs = detect_kernel_stage_requirements(source_text, kernel_name=kernel_name, kernel_class=kernel_class)
    op_family = reqs["operator_family"]
    ppath = Path(prior_path).expanduser().resolve() if prior_path else default_rsi_prior_path()
    priors = _load_priors(ppath)

    normalized: List[Dict[str, Any]] = []
    rejected_static: List[Dict[str, Any]] = []
    for idx, raw in enumerate(candidates):
        cand = dict(raw)
        strategy = str(cand.get("strategy") or f"cand_{idx}")
        mode = str(cand.get("mode") or "WP_CTX_REG")
        cand["strategy"] = strategy
        cand["mode"] = mode
        cand.setdefault("defines", [f"-D{mode}"])
        pkey = _prior_key(arch, op_family, strategy, mode)
        cand["_prior_key"] = pkey
        cand["_prior_score"] = round(_prior_score(priors, pkey), 4)
        ok_static, issues = check_candidate_static_coverage(cand, reqs)
        if not ok_static:
            rejected_static.append({"strategy": strategy, "mode": mode, "issues": issues})
            continue
        normalized.append(cand)

    normalized.sort(key=lambda c: -float(c["_prior_score"]))

    # Stage 2: parallel CPU compilation + compiler spill/SASS screen.
    def _compile_one(cand: Dict[str, Any]) -> Dict[str, Any]:
        tc0 = time.perf_counter()
        try:
            res = dict(compile_fn(dict(cand)) or {})
            err = None
        except Exception as exc:
            res = {}
            err = f"{type(exc).__name__}: {exc}"
        build_ok = bool(res.get("build_ok", err is None)) and err is None
        spill_gate = (
            verify_compiler_spill_gate(baseline_metrics, res)
            if build_ok
            else {"ok": False, "violations": [{"code": "BUILD_FAILED", "message": err or "build failed"}]}
        )
        core_multiset = res.get("core_multiset_ratio")
        core_lcs = res.get("core_retention_ratio")
        asm_ok = True
        if core_multiset is not None and core_lcs is not None:
            asm_ok = (float(core_lcs) >= 0.95) or (float(core_multiset) >= 0.98 and float(core_lcs) >= 0.85)
        elif core_lcs is not None:
            asm_ok = float(core_lcs) >= 0.95
        compile_ok = bool(build_ok and spill_gate["ok"] and asm_ok)
        return {
            "candidate": cand,
            "strategy": cand["strategy"],
            "mode": cand["mode"],
            "prior_key": cand["_prior_key"],
            "prior_score": cand["_prior_score"],
            "build_ok": build_ok,
            "asm_ok": asm_ok,
            "compile_ok": compile_ok,
            "spill_gate": spill_gate,
            "compile_metrics": res,
            "compile_s": round(time.perf_counter() - tc0, 3),
            "error": err,
            "gpu_screen": None,
            "overhead_pct": None,
            "passed": False,
        }

    compiled: List[Dict[str, Any]] = []
    if normalized:
        workers = max(1, min(int(max_workers), len(normalized)))
        if workers == 1:
            compiled = [_compile_one(c) for c in normalized]
        else:
            order_map = {id(c): i for i, c in enumerate(normalized)}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(_compile_one, c): order_map[id(c)] for c in normalized}
                ordered: Dict[int, Dict[str, Any]] = {}
                for fut in as_completed(futs):
                    ordered[futs[fut]] = fut.result()
                compiled = [ordered[i] for i in range(len(normalized))]

    # Stage 3: fast GPU screen on compile-clean arms in prior-ranked order.
    winner: Optional[Dict[str, Any]] = None
    prior_updates: List[Dict[str, Any]] = []
    gpu_screens = 0
    for item in compiled:
        gpu_res: Optional[Dict[str, Any]] = None
        passed = False
        ovh_pct: Optional[float] = None
        if item["compile_ok"]:
            if screen_gpu_fn is None:
                passed = True
                ovh_pct = item["compile_metrics"].get("overhead_pct")
            else:
                gpu_screens += 1
                tg0 = time.perf_counter()
                try:
                    gpu_res = dict(screen_gpu_fn(dict(item["candidate"]), dict(item["compile_metrics"])) or {})
                    gpu_err = None
                except Exception as exc:
                    gpu_res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    gpu_err = gpu_res["error"]
                gpu_res["gpu_screen_s"] = round(time.perf_counter() - tg0, 3)
                ovh_raw = gpu_res.get("overhead_pct")
                ovh_pct = float(ovh_raw) if isinstance(ovh_raw, (int, float)) else None
                ovh_ok = True if ovh_pct is None else (abs(ovh_pct) <= float(max_overhead_pct) + 1e-9)
                passed = bool(gpu_err is None and gpu_res.get("ok") is True and ovh_ok)
        item["gpu_screen"] = gpu_res
        item["overhead_pct"] = ovh_pct
        item["passed"] = passed
        prior_updates.append({
            "prior_key": item["prior_key"],
            "compile_ok": item["compile_ok"],
            "passed": passed,
            "overhead_pct": ovh_pct,
        })
        if passed and winner is None:
            winner = {
                "strategy": item["strategy"],
                "mode": item["mode"],
                "defines": list(item["candidate"].get("defines") or []),
                "overhead_pct": ovh_pct,
                "compile_s": item["compile_s"],
                "gpu_screen": gpu_res,
            }
            break

    if prior_updates:
        _update_priors(ppath, priors, prior_updates)

    return {
        "schema": RSI_SEARCH_SCHEMA,
        "status": "PASS" if winner is not None else "NO_CLEAN_CANDIDATE",
        "operator_family": op_family,
        "stage_requirements": reqs,
        "winner": winner,
        "candidates_total": len(candidates),
        "rejected_static": rejected_static,
        "compiled_count": len(compiled),
        "compile_clean_count": sum(1 for x in compiled if x["compile_ok"]),
        "gpu_screens": gpu_screens,
        "attempts": [
            {
                "strategy": x["strategy"],
                "mode": x["mode"],
                "prior_score": x["prior_score"],
                "compile_ok": x["compile_ok"],
                "asm_ok": x["asm_ok"],
                "passed": x["passed"],
                "overhead_pct": x["overhead_pct"],
                "compile_s": x["compile_s"],
                "error": x["error"],
            }
            for x in compiled
        ],
        "prior_path": str(ppath),
        "wall_s": round(time.perf_counter() - t0, 3),
    }


__all__ = [
    "RSI_SEARCH_SCHEMA",
    "RSI_PRIOR_SCHEMA",
    "build_minimal_include_overlay",
    "check_candidate_static_coverage",
    "default_rsi_prior_path",
    "detect_kernel_stage_requirements",
    "run_rsi_placement_search",
]
