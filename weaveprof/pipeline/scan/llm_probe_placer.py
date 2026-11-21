#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§3.1 Semantic Probe Placement — Read-Only LLM JSON Probe Spec Generator & §4.2 Ablation Driver.

Supports the three evaluated DAC'27 models at ``reasoning_effort="max"``:
  - ``qwen4-max``
  - ``deepseek-v4.1-flash-1017``
  - ``glm-5.3``

Retrieves route-specific exemplars from ``configs/``:
  - ``configs/semantic_ids.json``
  - ``configs/operators/operators.json``
  - ``configs/languages/<lang>/patterns.json`` (+ ``recorder_gfx942.json``)
  - ``configs/cards/<card>.json``

Emits a SHA-bound declarative **JSON Probe Spec** (``weaveprofiler.probe-spec/v1``) and
supports all four §4.2 progressive ablation modes:
  1. ``bare_direct``      : Bare LLM directly rewrites full kernel source.
  2. ``kb_direct``        : LLM + route KB exemplars directly rewrites full kernel source.
  3. ``open_loop``        : Read-only JSON Probe Spec + KB + deterministic rule injection (single round).
  4. ``full_closed_loop`` : Full WeaveProfiler (+ deterministic compile/SASS/numerical/completeness/overhead
                            closed-loop feedback & Scheme A Probe Coarsening).
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import sys
import time

sys.dont_write_bytecode = True
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

SKILL_ROOT = Path(__file__).resolve().parents[3]
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

from weaveprof.pipeline.core.semantics import load_kb as load_semantic_kb, source_hints  # noqa: E402
from weaveprof.contract import compact_probe_spec, normalize_probe_spec, revise_probe_spec

PROBE_SPEC_SCHEMA = "weaveprofiler.probe-spec/v1"
BAILIAN_BASE_URL = "https://ws-3o1qb69p061drz4g.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"

SUPPORTED_MODELS = (
    "qwen4-max",
    "deepseek-v4.1-flash-1017",
    "kimi-k3.1-pro",
)

ABLATION_MODES = (
    "bare_direct",
    "kb_direct",
    "open_loop",
    "full_closed_loop",
)

CANONICAL_STAGE_IDS = (
    "load",
    "wait",
    "mma",
    "compute",
    "barrier",
    "store",
    "epilogue",
    "dispatch",
    "communication",
)

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

FIVE_CLASS_MAP: Dict[str, str] = {
    "load": "load",
    "wait": "wait",
    "mma": "compute",
    "compute": "compute",
    "softmax": "compute",
    "reduction": "compute",
    "barrier": "sync",
    "store": "store",
    "epilogue": "store",
    "dispatch": "compute",
    "communication": "load",
}

STAGE_ALIAS_NORMALIZE: Dict[str, str] = {
    "sync": "barrier",
    "synchronization": "barrier",
    "fence": "barrier",
    "mbarrier": "barrier",
    "wgmma": "mma",
    "mfma": "mma",
    "tcgen05": "mma",
    "dot": "mma",
    "gemm_mma": "mma",
    "softmax": "compute",
    "reduction": "compute",
    "reduce": "compute",
    "writeback": "store",
    "tma_load": "load",
    "cp_async": "load",
    "tma_store": "store",
    "comm": "communication",
    "allgather": "communication",
    "allreduce": "communication",
}

CARD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "h20": {"platform": "nv", "arch": "sm_90a", "warp_size": 32, "card_file": "h20.json"},
    "b300": {"platform": "nv", "arch": "sm_103a", "warp_size": 32, "card_file": "b300_sm_103a.json"},
    "mi308x": {"platform": "amd", "arch": "gfx942", "warp_size": 64, "card_file": "mi308x.json"},
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detect_language(path: Optional[Path], text: str, explicit: Optional[str] = None) -> str:
    if explicit:
        norm = explicit.strip().lower()
        if norm in ("cuda", "cuda_cpp", "cu"):
            return "cuda_cpp"
        if norm in ("hip", "hip_cpp", "rocm"):
            return "hip_cpp"
        if norm in ("triton", "triton_nv", "triton_amd"):
            return "triton"
        if norm in ("cute", "cutedsl", "cute_dsl"):
            return "cutedsl"
        raise ValueError(f"Unsupported language: {explicit!r}")
    suffix = path.suffix.lower() if path else ""
    if suffix in (".cu", ".cuh"):
        if "__builtin_amdgcn" in text or "hip/hip_runtime.h" in text:
            return "hip_cpp"
        return "cuda_cpp"
    if suffix in (".hip", ".cpp", ".cc", ".cxx", ".hpp", ".h"):
        if "cuda_runtime.h" in text or "mma.h" in text or "wgmma" in text or "tcgen05" in text:
            return "cuda_cpp"
        return "hip_cpp"
    if suffix == ".py":
        if "cutlass.cute" in text or "@cute.kernel" in text or "@cute.jit" in text:
            return "cutedsl"
        if "triton" in text or "@triton.jit" in text:
            return "triton"
    if "@cute.kernel" in text or "cutlass.cute" in text:
        return "cutedsl"
    if "@triton.jit" in text or "import triton" in text:
        return "triton"
    if "__builtin_amdgcn" in text or "hip/hip_runtime.h" in text:
        return "hip_cpp"
    return "cuda_cpp"


def resolve_route(card: str, arch: Optional[str], language: str) -> Dict[str, Any]:
    c = card.strip().lower()
    if c == "auto":
        c = "mi308x" if language == "hip_cpp" else "h20"
        try:
            import torch
            if torch.cuda.is_available():
                label = torch.cuda.get_device_name().lower()
                if getattr(torch.version, "hip", None):
                    c = "mi308x"
                elif "b300" in label:
                    c = "b300"
        except ImportError:
            pass
    if c not in CARD_DEFAULTS:
        raise ValueError(f"Unsupported card {card!r}; expected one of {sorted(CARD_DEFAULTS)}")
    info = dict(CARD_DEFAULTS[c])
    if c == "mi308x" and language == "cuda_cpp":
        language = "hip_cpp"
    if arch:
        info["arch"] = arch.strip().lower()
        if c == "b300" and info["arch"] == "sm_100a":
            info["card_file"] = "b300_sm_100a.json"
        elif c == "b300":
            info["card_file"] = "b300_sm_103a.json"
    info["card"] = c
    info["language"] = language
    return info


def get_llm_endpoint_url() -> str:
    """Resolve OpenAI-compatible chat/completions endpoint (`OPENAI_BASE_URL` or default Bailian)."""
    configured = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    if configured:
        url = configured.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        if url.endswith("/v1") or url.endswith("/compatible-mode/v1"):
            return url + "/chat/completions"
        return url + "/v1/chat/completions"
    return BAILIAN_BASE_URL


def get_bailian_api_key() -> str:
    env_order = (
        ("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "BAILIAN_API_KEY")
        if (os.environ.get("OPENAI_BASE_URL") or "").strip()
        else ("DASHSCOPE_API_KEY", "BAILIAN_API_KEY", "OPENAI_API_KEY")
    )
    for name in env_order:
        val = os.environ.get(name)
        if val and val.strip():
            return val.strip()
    if os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                for name in env_order:
                    try:
                        val, _ = winreg.QueryValueEx(key, name)
                        if val and str(val).strip():
                            return str(val).strip()
                    except FileNotFoundError:
                        pass
        except OSError:
            pass
    raise RuntimeError("No API key found in environment (OPENAI_API_KEY / DASHSCOPE_API_KEY / BAILIAN_API_KEY)")


def call_bailian_llm(
    model: str,
    messages: List[Dict[str, str]],
    *,
    reasoning_effort: str = "max",
    max_tokens: int = 32768,
    temperature: Optional[float] = None,
    response_format: Optional[Dict[str, Any]] = None,
    timeout: float = 300.0,
    retries: int = 5,
    stream: bool = True,
) -> Dict[str, Any]:
    """Call Aliyun Bailian MaaS directly (bypassing local proxy) with SSE streaming & token telemetry."""
    api_key = get_bailian_api_key()
    endpoint_url = get_llm_endpoint_url()
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if reasoning_effort and endpoint_url == BAILIAN_BASE_URL:
        payload["reasoning_effort"] = reasoning_effort
    elif reasoning_effort and os.environ.get("WEAVEPROF_SEND_REASONING_EFFORT", "1") != "0":
        payload["reasoning_effort"] = reasoning_effort
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    if temperature is not None:
        payload["temperature"] = temperature
    if response_format is not None:
        payload["response_format"] = response_format

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_err: Optional[Exception] = None

    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                endpoint_url,
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            with opener.open(req, timeout=timeout) as resp:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if stream and ("event-stream" in ctype or "text/plain" in ctype):
                    content_chunks: List[str] = []
                    reasoning_chunks: List[str] = []
                    usage: Dict[str, Any] = {}
                    finish_reason = None
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        choices = chunk.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            if delta.get("content"):
                                content_chunks.append(str(delta["content"]))
                            if delta.get("reasoning_content"):
                                reasoning_chunks.append(str(delta["reasoning_content"]))
                            if choices[0].get("finish_reason"):
                                finish_reason = choices[0]["finish_reason"]
                    elapsed_s = time.perf_counter() - t0
                    content = "".join(content_chunks)
                    reasoning_content = "".join(reasoning_chunks)
                else:
                    raw = resp.read().decode("utf-8")
                    elapsed_s = time.perf_counter() - t0
                    doc = json.loads(raw)
                    choice = (doc.get("choices") or [{}])[0]
                    msg = choice.get("message") or {}
                    usage = doc.get("usage") or {}
                    content = msg.get("content") or ""
                    reasoning_content = msg.get("reasoning_content") or ""
                    finish_reason = choice.get("finish_reason")
            comp_details = usage.get("completion_tokens_details") or {}
            return {
                "ok": True,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "elapsed_s": round(elapsed_s, 4),
                "content": content,
                "reasoning_content": reasoning_content,
                "usage": {
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "reasoning_tokens": int(
                        comp_details.get("reasoning_tokens")
                        or usage.get("reasoning_tokens")
                        or 0
                    ),
                    "total_tokens": int(usage.get("total_tokens") or 0),
                },
                "finish_reason": finish_reason,
            }
        except Exception as exc:
            last_err = exc
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    err_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    err_body = ""
                last_err = RuntimeError(f"HTTP {exc.code}: {err_body[:600]}")
            if attempt + 1 < retries:
                is_rate_limit = ("429" in str(last_err)) or ("Throttling" in str(last_err))
                base_wait = (4.0 * (2 ** attempt)) if is_rate_limit else (2.0 * (attempt + 1))
                jitter = (hash((model, attempt, time.time_ns())) % 1000) / 1000.0
                time.sleep(min(30.0, base_wait + jitter))
    raise RuntimeError(f"Bailian call failed for {model}: {last_err}")


def retrieve_route_kb(
    language: str,
    card: str,
    arch: Optional[str],
    source_text: str,
) -> Dict[str, Any]:
    """Retrieve compact, route-specific knowledge from operators.json, patterns.json, and card.json."""
    route = resolve_route(card, arch, language)
    lang = route["language"]
    kb_sem = load_semantic_kb()
    hints = source_hints(source_text, kb_sem)

    card_path = SKILL_ROOT / "configs" / "cards" / route["card_file"]
    if not card_path.is_file():
        card_path = SKILL_ROOT / "configs" / "cards" / f"{route['card']}.json"
    card_doc = json.loads(card_path.read_text(encoding="utf-8"))

    if lang == "triton":
        pat_rel = "amd_patterns.json" if route["platform"] == "amd" else "nvidia_patterns.json"
        pat_path = SKILL_ROOT / "configs" / "languages" / "triton" / pat_rel
    else:
        pat_path = SKILL_ROOT / "configs" / "languages" / lang / "patterns.json"
    pat_doc = json.loads(pat_path.read_text(encoding="utf-8"))

    extra_recorder_notes: List[str] = []
    if lang == "hip_cpp":
        gfx_rec = SKILL_ROOT / "configs" / "languages" / "hip_cpp" / "recorder_gfx942.json"
        if gfx_rec.is_file():
            extra_recorder_notes.append(f"gfx942 recorder KB: {gfx_rec.name} (sha={sha256_file(gfx_rec)[:12]})")

    op_path = SKILL_ROOT / "configs" / "operators" / "operators.json"
    op_doc = json.loads(op_path.read_text(encoding="utf-8"))
    lowered_src = source_text.lower()
    two_level_path = SKILL_ROOT / "configs" / "composable_two_level_kb.json"
    two_level = json.loads(two_level_path.read_text(encoding="utf-8"))
    topologies = sorted(two_level["level_2_cta_topologies"],
                        key=lambda t: -sum(cue.lower() in lowered_src for cue in t["cues"]))
    hint_set = set(hints.get("semantic_ids") or [])

    scored_ops: List[Tuple[int, Dict[str, Any]]] = []
    for op_entry in op_doc.get("operators") or []:
        sid = str(op_entry.get("semantic_id") or "")
        score = 0
        if sid in hint_set:
            score += 10
        for alias in op_entry.get("aliases") or []:
            if str(alias).lower() in lowered_src:
                score += 5
        if sid == "gemm" and any(k in lowered_src for k in ("mma", "wgmma", "mfma", "dot", "gemm", "tcgen05")):
            score += 8
        if sid == "moe" and any(k in lowered_src for k in ("expert", "moe", "topk", "dispatch", "combine")):
            score += 12
        if sid == "megakernel" and any(k in lowered_src for k in ("megamoe", "megakernel", "role", "fc1", "fc2", "swiglu")):
            score += 12
        if sid == "attention" and any(k in lowered_src for k in ("qk", "softmax", "exp2", "attn", "flash", "kv")):
            score += 12
        if sid == "reduction" and any(k in lowered_src for k in ("shfl", "reduce", "atomicadd", "row_sum", "row_max")):
            score += 8
        scored_ops.append((score, op_entry))

    scored_ops.sort(key=lambda item: item[0], reverse=True)
    selected_ops = [entry for sc, entry in scored_ops[:3] if sc > 0]
    if not selected_ops and scored_ops:
        selected_ops = [scored_ops[0][1]]

    op_summaries: List[Dict[str, Any]] = []
    for op_entry in selected_ops:
        stages_brief = []
        for st in op_entry.get("stages") or []:
            stages_brief.append({
                "stage": st.get("stage"),
                "phase": st.get("phase"),
                "typical_role": st.get("typical_role"),
                "semantic_ids": st.get("semantic_ids"),
                "code_signature": st.get("code_signature"),
                "placement_rule": st.get("instrument"),
            })
        op_summaries.append({
            "operator_semantic_id": op_entry.get("semantic_id"),
            "name": op_entry.get("name"),
            "roles": op_entry.get("roles"),
            "stages": stages_brief,
        })

    selected_op_ids = {o["operator_semantic_id"] for o in op_summaries}
    relevant_sem_ids = selected_op_ids | set(CANONICAL_STAGE_IDS)
    lang_exemplars: List[Dict[str, Any]] = []
    for pat in pat_doc.get("patterns") or []:
        psid = str(pat.get("semantic_id") or "")
        if psid in relevant_sem_ids:
            lang_exemplars.append({
                "semantic_id": psid,
                "event_boundaries": pat.get("event_boundaries"),
                "writer_rule": pat.get("writer_rule"),
                "language_notes": (pat.get("language_notes") or [])[:3],
            })

    return {
        "route": route,
        "hints": hints,
        "card_summary": {
            "card": card_doc.get("card"),
            "arch": route["arch"],
            "warp_size": card_doc.get("warp_size"),
            "profiler": (card_doc.get("profiler") or {}).get("primary"),
        },
        "conventions": pat_doc.get("conventions") or {},
        "operators": op_summaries,
        "language_patterns": lang_exemplars[:10],
        "level_1": {
            "primitives": {k: v["meaning"] for k, v in two_level["canonical_events"].items()},
            "language": two_level["level_1_primitives"]["language_contracts"][lang],
        },
        "level_2": [{k: t[k] for k in ("topology_id", "cues", "warp_roles")} for t in topologies[:3]],
        "provenance": {
            "two_level_sha256": sha256_file(two_level_path),
            "semantic_ids_sha256": kb_sem.get("_sha256"),
            "card_file": str(card_path.relative_to(SKILL_ROOT)),
            "card_sha256": sha256_file(card_path),
            "patterns_file": str(pat_path.relative_to(SKILL_ROOT)),
            "patterns_sha256": sha256_file(pat_path),
            "operators_file": "configs/operators/operators.json",
            "operators_sha256": sha256_file(op_path),
            "extra_notes": extra_recorder_notes,
        },
    }


def format_numbered_source(source_text: str) -> str:
    lines = source_text.splitlines()
    width = max(3, len(str(len(lines))))
    return "\n".join(f"{idx:>{width}d}: {line}" for idx, line in enumerate(lines, start=1))


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    candidates = []
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(text[first:last + 1])
    
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        # Clean // comments and /* */ comments
        cleaned = re.sub(r"//.*$", "", cand, flags=re.MULTILINE)
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
        # Remove trailing commas before } or ]
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Try fixing unescaped newlines in strings or common formatting errors
        try:
            import ast
            # If valid python dict literal (e.g. single quotes, True/False)
            val = ast.literal_eval(cleaned.replace("true", "True").replace("false", "False").replace("null", "None"))
            if isinstance(val, dict):
                return val
        except Exception:
            pass

    raise ValueError(f"LLM output did not contain a valid JSON Probe Spec: {text[:400]!r}")


def extract_code_block(raw_text: str) -> str:
    text = raw_text.strip()
    blocks = re.findall(r"```(?:cuda|cpp|c\+\+|hip|python|py|c)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        blocks.sort(key=len, reverse=True)
        return blocks[0].strip() + "\n"
    return text + "\n"


def normalize_semantic_id(raw_id: Any) -> str:
    s = str(raw_id or "").strip().lower()
    if s in CANONICAL_STAGE_IDS:
        return s
    if s in STAGE_ALIAS_NORMALIZE:
        return STAGE_ALIAS_NORMALIZE[s]
    for cid in CANONICAL_STAGE_IDS:
        if cid in s:
            return cid
    return "compute"


def sanitize_and_bind_probe_spec(
    raw_spec: Mapping[str, Any],
    *,
    source_text: str,
    source_path: Optional[Path],
    route: Mapping[str, Any],
    telemetry: Optional[Mapping[str, Any]] = None,
    kb_provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize, line-clamp, snippet-bind, and SHA-stamp a declarative JSON Probe Spec."""
    if "roles" in raw_spec:
        result = normalize_probe_spec(raw_spec, source_text, language=route["language"])
        result.update({key: route[key] for key in ("language", "card", "platform", "arch")})
        result.update(source_path=str(source_path or ""), source_sha256=sha256_text(source_text),
                      kb_provenance=dict(kb_provenance or {}), telemetry=dict(telemetry or {}))
        return result
    lines = source_text.splitlines()
    n_lines = max(1, len(lines))

    def _clamp_line(val: Any, default: int = 1) -> int:
        try:
            ln = int(val)
        except (TypeError, ValueError):
            ln = default
        return max(1, min(n_lines, ln))

    def _resolve_snippet_line(hint_line: int, snippet: Optional[str]) -> int:
        if not snippet or not str(snippet).strip():
            return hint_line
        target = str(snippet).strip()
        lo = max(1, hint_line - 8)
        hi = min(n_lines, hint_line + 8)
        for ln in range(lo, hi + 1):
            if lines[ln - 1].strip() and (target in lines[ln - 1] or lines[ln - 1].strip() in target):
                return ln
        for ln in range(1, n_lines + 1):
            if target in lines[ln - 1]:
                return ln
        return hint_line

    raw_roles = raw_spec.get("warp_roles") or []
    warp_roles: List[Dict[str, Any]] = []
    role_name_to_id: Dict[str, int] = {}
    if isinstance(raw_roles, list) and raw_roles:
        for idx, r in enumerate(raw_roles):
            if not isinstance(r, Mapping):
                continue
            rname = str(r.get("name") or r.get("role") or f"role_{idx}").strip()
            rid = int(r.get("role_id") if r.get("role_id") is not None else idx)
            warp_roles.append({
                "role_id": rid,
                "name": rname,
                "predicate": str(r.get("predicate") or "true").strip(),
                "leader_predicate": str(r.get("leader_predicate") or "").strip(),
                "begin_line": _clamp_line(r["begin_line"]) if r.get("begin_line") else None,
                "end_line": _clamp_line(r["end_line"]) if r.get("end_line") else None,
                "description": str(r.get("description") or "").strip(),
            })
            role_name_to_id[rname.lower()] = rid

    if not warp_roles:
        warp_roles = [{
            "role_id": 0,
            "name": "cta_worker",
            "predicate": "true",
            "leader_predicate": "",
            "begin_line": None,
            "end_line": None,
            "description": "Homogeneous CTA/program execution stream",
        }]
        role_name_to_id["cta_worker"] = 0

    raw_probes = raw_spec.get("probes") or raw_spec.get("stages") or []
    probes: List[Dict[str, Any]] = []
    for idx, st in enumerate(raw_probes):
        if not isinstance(st, Mapping):
            continue
        sem_id = normalize_semantic_id(st.get("semantic_id") or st.get("phase") or st.get("class"))
        mode = str(st.get("mode") or "sync").strip().lower()
        if mode not in ("sync", "async"):
            mode = "async" if st.get("completion_wait_line") or st.get("completion_wait_snippet") else "sync"

        b_hint = _clamp_line(st.get("begin_line") or st.get("start_line") or 1)
        e_hint = _clamp_line(st.get("end_line") or b_hint, default=b_hint)
        b_snip_in = st.get("begin_snippet")
        e_snip_in = st.get("end_snippet")
        cw_snip_in = st.get("completion_wait_snippet")
        cw_line_in = st.get("completion_wait_line")

        b_line = _resolve_snippet_line(b_hint, b_snip_in)
        e_line = _resolve_snippet_line(e_hint, e_snip_in)
        if cw_line_in is not None or cw_snip_in:
            cw_line = _resolve_snippet_line(_clamp_line(cw_line_in or e_line, default=e_line), cw_snip_in)
            if mode == "async" and cw_line >= b_line:
                e_line = cw_line
        else:
            cw_line = e_line if mode == "async" else None

        if e_line < b_line:
            e_line = b_line

        role_str = str(st.get("role") or warp_roles[0]["name"]).strip()
        rid = st.get("role_id")
        if rid is None:
            rid = role_name_to_id.get(role_str.lower(), warp_roles[0]["role_id"])
        else:
            rid = int(rid)

        probes.append({
            "probe_id": str(st.get("probe_id") or st.get("stage_id") or f"{sem_id}_{idx}"),
            "semantic_id": sem_id,
            "canonical_class": FIVE_CLASS_MAP.get(sem_id, "compute"),
            "event_id": STAGE_EVENT_IDS[sem_id],
            "role": role_str,
            "role_id": rid,
            "mode": mode,
            "target_func": str(st.get("target_func") or "").strip(),
            "in_inner_loop": bool(st.get("in_inner_loop", st.get("in_hot_loop", False))),
            "begin_line": b_line,
            "end_line": e_line,
            "begin_snippet": lines[b_line - 1].strip() if lines else "",
            "end_snippet": lines[e_line - 1].strip() if lines else "",
            "completion_wait_line": cw_line,
            "completion_wait_snippet": (lines[cw_line - 1].strip() if (cw_line and lines) else None),
            "aux_expr": str(st.get("aux_expr") or "").strip(),
            "notes": str(st.get("notes") or "").strip(),
        })

    gran = str(raw_spec.get("granularity") or ("warp" if len(warp_roles) > 1 else "sm")).strip().lower()
    if gran not in ("warp", "sm"):
        gran = "warp" if len(warp_roles) > 1 else "sm"

    op_fam = str(raw_spec.get("operator_family") or "gemm").strip().lower()
    if op_fam not in STAGE_EVENT_IDS:
        op_fam = "gemm"

    return {
        "schema": PROBE_SPEC_SCHEMA,
        "source_path": str(source_path) if source_path else "",
        "source_sha256": sha256_text(source_text),
        "language": route["language"],
        "platform": route["platform"],
        "card": route["card"],
        "arch": route["arch"],
        "operator_family": op_fam,
        "kernel_name": str(raw_spec.get("kernel_name") or "").strip(),
        "granularity": gran,
        "warp_roles": warp_roles,
        "probes": probes,
        "kb_provenance": dict(kb_provenance or {}),
        "telemetry": dict(telemetry or {}),
    }


def _build_readonly_probe_prompt(
    source_text: str,
    kb_ctx: Mapping[str, Any],
    *,
    diagnostics: Optional[Sequence[Mapping[str, Any]]] = None,
    prev_spec: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, str]]:
    route = kb_ctx["route"]
    system = (
        "You are WeaveProfiler's Stage Locator (§3.1). Read the numbered kernel source; never "
        "change it. Emit a single JSON object (probe.json) with roles and probes. Each probe "
        "needs a unique string id ('load', 'mma', 'store', 'dma_issue', 'wait', 'communication'), "
        "and begin and end objects containing a declared role and a 1-based source line. "
        "Optionally include slot, a pipeline buffer-index expression valid at both endpoints. "
        "Roles may use 'warp 4', 'warps 0-3', 'all' or a source predicate. Level 1 supplies "
        "asynchronous load, tile math, store and cross-GPU communication primitive rules for "
        "CUDA C++, HIP C++, Triton and CuTeDSL. Level 2 composes these into CTA warp-role "
        "pipelines. CRITICAL STAGE GRANULARITY RULES:\n"
        "1) In Round 1, ALWAYS locate fine-grained per-tile 'load', 'mma', and 'store' stages "
        "inside the main K-tile loop (and epilogue) so the timeline exposes actual tile-by-tile "
        "load and mma execution — NEVER collapse the entire K-loop into a single coarse 'compute' "
        "probe in Round 1.\n"
        "2) For warp-specialized producer-consumer pipelines (e.g. DMA warps + Math warps with "
        "mbarrier), emit: (a) a cross-warp paired 'load' probe whose begin is on the producer "
        "DMA role right before issuing the tile's async copies (OUTSIDE any inner unrolled "
        "sub-loop like for c=0..2!) and whose end is on the consumer Math role right after "
        "full_mbar[stage] wait returns, with slot set to the circular stage index (e.g. 'stage'); "
        "(b) a synchronous 'dma_issue' probe on the DMA role bracketing one tile's producer body "
        "(wait empty + cp.async + arrive) so DMA warps show active per-tile intervals; "
        "(c) a synchronous 'mma' probe on the Math role bracketing the per-tile Tensor Core MMA "
        "block; and (d) a 'store' probe on the epilogue.\n"
        "3) Respect lexical scope and trip counts: begin and end of any probe MUST execute the "
        "exact same number of times per tile. Hoist/coarsen ONLY when feedback.json reports "
        "overhead > limit or assembly failure."
    )
    example = {"roles": {"dma": "warp 4", "math": "warps 0-3"},
               "probes": [{"id": "load", "begin": {"role": "dma", "line": 42},
                           "end": {"role": "math", "line": 55}, "slot": "s"}]}
    parts = [
        "Example of the output schema (choose actual roles and lines from the input): "
        + json.dumps(example),
        f"Target: {route['language']}, {route['card']}, {route['arch']}",
        "Level 1 ? primitive rules: " + json.dumps(kb_ctx["level_1"], ensure_ascii=False),
        "Level 2 ? CTA warp-role pipeline templates: " + json.dumps(kb_ctx["level_2"], ensure_ascii=False),
        "Primitive boundaries: " + json.dumps(kb_ctx["language_patterns"], ensure_ascii=False),
        "Read-only kernel:\n" + format_numbered_source(source_text),
    ]
    if prev_spec and diagnostics:
        parts.append("Previous probe.json: " + json.dumps(compact_probe_spec(prev_spec)))
        parts.append("Verification feedback (feedback.json): " + json.dumps(diagnostics, ensure_ascii=False))
        tsum = None
        if isinstance(diagnostics, Mapping):
            tsum = diagnostics.get("timeline_summary") or (diagnostics.get("metrics") or {}).get("timeline_summary")
        if isinstance(tsum, Mapping) and tsum.get("hotspots"):
            parts.append(
                "Failed-run timeline binary summary (from failed trace.perfetto): "
                f"baseline={tsum.get('kernel_baseline_us')}us, "
                f"instrumented={tsum.get('kernel_instrumented_us')}us "
                f"(+{tsum.get('overhead_pct')}%). Top stage hotspots:\n  - "
                + "\n  - ".join(str(h) for h in tsum["hotspots"])
            )
        parts.append(
            "Use fires, enclosing_loop, timeline_summary hotspots, and metrics (including pairing_errors "
            "and overhead_ratio) to repair all failed checks: ensure begin and end execute the exact same "
            "trip count (never place one endpoint inside an inner unrolled sub-loop like c=0..2 while the "
            "other sits outside), never place probes inside polling spin loops, and hoist/merge "
            "high-frequency inner-loop probes to coarse enclosing loop boundaries when overhead or "
            "assembly fails (omit slot on probes outside the slot loop)."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(parts)}]


def _recorder_signature_hint(language: str, platform: str) -> str:
    if language == "cuda_cpp":
        return (
            '#include "fi_lite.cuh"\n'
            "// Macros: FI_OPEN(role_id); FI_OPEN_BY(pred, role_id); "
            "FI_B(event_id, aux); FI_E(event_id, aux); FI_CLOSE();\n"
            "// Canonical event_ids: 1=load, 2=wait, 3=mma, 4=compute, 5=epilogue, 6=store, 7=barrier, 8=dispatch, 9=communication, 10=gemm"
        )
    if language == "hip_cpp":
        return (
            '#include "fi_lite_hip.h"\n'
            "// Macros: FI_OPEN(role_id); FI_OPEN_BY(pred, role_id); "
            "FI_B(event_id, aux); FI_E(event_id, aux); FI_CLOSE();\n"
            "// Canonical event_ids: 1=load, 2=wait, 3=mma, 4=compute, 5=epilogue, 6=store, 7=barrier, 8=dispatch, 9=communication, 10=gemm"
        )
    if language == "triton" and platform == "amd":
        return (
            "from fi_triton_amd import FiPackedBuffer as FiAutoBuffer, fi_open_16b, fi_b_16b, fi_e_16b, fi_close_16b\n"
            "# Event IDs: 1=load, 2=wait, 3=mma, 4=compute, 5=epilogue, 6=store, 7=barrier, 8=dispatch, 9=communication, 10=gemm"
        )
    if language == "triton":
        return (
            "from fi_triton import FiCfg as FiAutoCfg, FiBuffer as FiAutoBuffer, fi_open, fi_b, fi_e, fi_close\n"
            "# Event IDs: 1=load, 2=wait, 3=mma, 4=compute, 5=epilogue, 6=store, 7=barrier, 8=dispatch, 9=communication, 10=gemm"
        )
    return (
        "from fi_cute import FiBuffer as FiAutoBuffer, fi_open, fi_b, fi_e, fi_close\n"
        "# Event IDs: 1=load, 2=wait, 3=mma, 4=compute, 5=epilogue, 6=store, 7=barrier, 8=dispatch, 9=communication, 10=gemm"
    )


def _build_direct_rewrite_prompt(
    source_text: str,
    route: Mapping[str, Any],
    *,
    kb_ctx: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Build prompt for §4.2 `bare_direct` (no KB) and `kb_direct` (+KB, still direct full-source rewrite)."""
    rec_hint = _recorder_signature_hint(route["language"], route["platform"])
    sys_msg = (
        "You are a GPU kernel profiling assistant. Directly rewrite the COMPLETE kernel source file "
        "to insert intra-kernel stage timing probes using the provided recorder primitives.\n"
        "Return the complete rewritten source file inside a single fenced code block (```...```). "
        "Do not omit any existing functions, includes, or logic."
    )
    parts = [
        f"Target: language={route['language']}, card={route['card']}, arch={route['arch']}\n",
        f"Available Recorder Primitives:\n```cpp\n{rec_hint}\n```\n",
    ]
    if kb_ctx is not None:
        parts.append(
            "Operator & Language Profiling Knowledge Base:\n"
            + json.dumps({
                "conventions": kb_ctx["conventions"],
                "operator_topologies": kb_ctx["operators"],
                "language_patterns": kb_ctx["language_patterns"],
            }, ensure_ascii=False, indent=2)
            + "\n"
        )
    parts.append(f"Original Kernel Source:\n```\n{source_text}\n```\n")
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": "\n".join(parts)},
    ]


_DIRECT_BEGIN_RE = re.compile(
    r"\b(?:FI_BG_PRED|FI_B_PRED|FI_BG|FI_B|_fi_b_16b|fi_b_16b|_fi_b|fi_b)\s*\(([^;)]*)\)"
)
_DIRECT_END_RE = re.compile(
    r"\b(?:FI_EG_PRED|FI_E_PRED|FI_EG|FI_E|_fi_e_16b|fi_e_16b|_fi_e|fi_e)\s*\(([^;)]*)\)"
)
_EV_ID_TO_SEM: Dict[int, str] = {v: k for k, v in STAGE_EVENT_IDS.items()}


def _infer_sem_from_probe_arg(arg_str: str) -> Optional[str]:
    low = arg_str.lower()
    for name in CANONICAL_STAGE_IDS:
        if name in low:
            return name
    for tok in re.findall(r"\b(\d+)\b", arg_str):
        val = int(tok)
        if val in _EV_ID_TO_SEM and val <= 9:
            return _EV_ID_TO_SEM[val]
    return None


def extract_probe_spec_from_rewritten_source(
    original_source: str,
    rewritten_source: str,
    *,
    source_path: Optional[Path],
    route: Mapping[str, Any],
    telemetry: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Map probe calls inserted by `bare_direct`/`kb_direct` back onto original source line numbers."""
    orig_lines = original_source.splitlines()
    new_lines = rewritten_source.splitlines()
    sm = difflib.SequenceMatcher(a=[l.strip() for l in orig_lines], b=[l.strip() for l in new_lines], autojunk=False)

    new_to_orig_before: List[int] = [1] * (len(new_lines) + 1)
    new_to_orig_after: List[int] = [max(1, len(orig_lines))] * (len(new_lines) + 1)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                new_to_orig_before[j1 + k] = i1 + k + 1
                new_to_orig_after[j1 + k] = i1 + k + 1
        else:
            pos_after = min(max(1, len(orig_lines)), i1 + 1)
            pos_before = max(1, min(len(orig_lines), i2))
            for j in range(j1, j2):
                new_to_orig_after[j] = pos_after
                new_to_orig_before[j] = pos_before

    open_stack: List[Tuple[str, int, str]] = []
    probes: List[Dict[str, Any]] = []
    current_role = "cta_worker"

    for j, line in enumerate(new_lines):
        role_m = re.search(r"\b(?:FI_OPEN|FI_OPEN_BY|FI_KERNEL_SCOPE|fi_open|fi_open_16b)\s*\([^)]*?(\d+)\s*\)", line)
        if role_m:
            current_role = f"role_{role_m.group(1)}"
        bm = _DIRECT_BEGIN_RE.search(line)
        if bm:
            sem = _infer_sem_from_probe_arg(bm.group(1))
            if sem:
                open_stack.append((sem, new_to_orig_after[j], current_role))
        em = _DIRECT_END_RE.search(line)
        if em:
            sem = _infer_sem_from_probe_arg(em.group(1))
            if sem:
                matched_idx = None
                for k in range(len(open_stack) - 1, -1, -1):
                    if open_stack[k][0] == sem:
                        matched_idx = k
                        break
                if matched_idx is not None:
                    s_sem, b_ln, r_nm = open_stack.pop(matched_idx)
                    e_ln = max(b_ln, new_to_orig_before[j])
                    probes.append({
                        "probe_id": f"{s_sem}_{len(probes)}",
                        "semantic_id": s_sem,
                        "role": r_nm,
                        "mode": "sync",
                        "begin_line": b_ln,
                        "end_line": e_ln,
                    })

    return sanitize_and_bind_probe_spec(
        {"probes": probes},
        source_text=original_source,
        source_path=source_path,
        route=route,
        telemetry=telemetry,
    )


def place_probes(
    source_path: Path,
    *,
    card: str,
    arch: Optional[str] = None,
    language: Optional[str] = None,
    model: str = "qwen4-max",
    reasoning_effort: str = "max",
    ablation_mode: str = "full_closed_loop",
    diagnostics: Optional[Sequence[Mapping[str, Any]]] = None,
    prev_spec: Optional[Mapping[str, Any]] = None,
    round_index: int = 1,
    llm_caller: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run one round of §3.1 Semantic Probe Placement (or §4.2 direct-rewrite ablation).

    Returns:
      - ``ok``: bool
      - ``ablation_mode``: str
      - ``probe_spec``: validated SHA-bound JSON Probe Spec
      - ``rewritten_source``: str | None (populated in ``bare_direct`` / ``kb_direct``)
      - ``telemetry``: exact ``elapsed_s`` (``T_llm``), token usage, model, reasoning_effort, round
      - ``raw_response``: raw LLM text for full provenance
    """
    if ablation_mode not in ABLATION_MODES:
        raise ValueError(f"Unsupported ablation_mode {ablation_mode!r}; expected {ABLATION_MODES}")
    source_text = source_path.read_text(encoding="utf-8")
    det_lang = detect_language(source_path, source_text, explicit=language)
    route = resolve_route(card, arch, det_lang)
    caller = llm_caller or call_bailian_llm

    kb_ctx = retrieve_route_kb(route["language"], route["card"], route["arch"], source_text)

    if ablation_mode in ("bare_direct", "kb_direct"):
        messages = _build_direct_rewrite_prompt(
            source_text,
            route,
            kb_ctx=(kb_ctx if ablation_mode == "kb_direct" else None),
        )
        llm_res = caller(model, messages, reasoning_effort=reasoning_effort)
        telemetry = {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "ablation_mode": ablation_mode,
            "round": round_index,
            "elapsed_s": llm_res["elapsed_s"],
            **llm_res.get("usage", {}),
        }
        rewritten = extract_code_block(llm_res["content"])
        probe_spec = extract_probe_spec_from_rewritten_source(
            source_text,
            rewritten,
            source_path=source_path,
            route=route,
            telemetry=telemetry,
        )
        return {
            "ok": True,
            "ablation_mode": ablation_mode,
            "probe_spec": probe_spec,
            "rewritten_source": rewritten,
            "telemetry": telemetry,
            "raw_response": llm_res["content"],
        }

    messages = _build_readonly_probe_prompt(
        source_text,
        kb_ctx,
        diagnostics=diagnostics,
        prev_spec=prev_spec,
    )
    llm_res = caller(model, messages, reasoning_effort=reasoning_effort)
    telemetry = {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "ablation_mode": ablation_mode,
        "round": round_index,
        "elapsed_s": llm_res["elapsed_s"],
        **llm_res.get("usage", {}),
    }
    try:
        parsed = extract_json_object(llm_res["content"])
    except Exception:
        reasoning = llm_res.get("reasoning_content") or llm_res.get("reasoning") or ""
        if reasoning:
            parsed = extract_json_object(reasoning)
        else:
            raise
    probe_spec = sanitize_and_bind_probe_spec(
        parsed,
        source_text=source_text,
        source_path=source_path,
        route=route,
        telemetry=telemetry,
        kb_provenance=kb_ctx["provenance"],
    )
    return {
        "ok": True,
        "ablation_mode": ablation_mode,
        "probe_spec": probe_spec,
        "rewritten_source": None,
        "telemetry": telemetry,
        "raw_response": llm_res["content"],
    }


_OFFLINE_PRIMITIVES = {
    "communication": r"\b(?:nvshmem\w*|rocshmem\w*|allgather|allreduce|p2p_\w+)\s*\(",
    "wait": r"(?:\.wait|\b\w*(?:wait_parity|barrier_wait|mbarrier_wait)\w*)\s*\(",
    "mma": r"(?:\btl\.dot|\bcute\.gemm|\b(?:fmaf?|\w*mma\w*|\w*mfma\w*))\s*\(",
    "store": r"(?:\btl\.store|\bcute\.store|\b\w*(?:stmatrix|store_global)\w*)\s*\(",
    "load": r"(?:\btl\.load|\bcute\.(?:load|copy)|\b\w*(?:tma_load|cp_async|memcpy_async|load_async)\w*)\s*\(",
}


def _offline_probe_spec(text, lang):
    lines = text.splitlines()
    if lang in ("triton", "cutedsl"):
        targets = [fn for fn in ast.walk(ast.parse(text)) if isinstance(fn, ast.FunctionDef)
                   and any(re.search(r"(?:^|\.)(?:jit|kernel)(?:\(|$)", ast.unparse(d))
                           for d in fn.decorator_list)]
        if not targets:
            raise ValueError("No decorated GPU kernel in this Python source")
        fn = next((f for f in targets if any("kernel" in ast.unparse(d) for d in f.decorator_list)), targets[0])
        name = fn.name
        statements = [n for n in ast.walk(fn) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.Expr, ast.AugAssign))]
        bounds = [(n.lineno, n.end_lineno) for n in sorted(statements, key=lambda n: n.lineno)]
    else:
        match = re.search(r"\b(?:__global__|CUTLASS_GLOBAL|CUTE_GLOBAL)\b[^;{]+?\b(\w+)\s*\([^;{}]*\)\s*\{", text, re.S)
        if not match:
            raise ValueError("No __global__ GPU kernel in this C++ source")
        name = match.group(1)
        first_line = text.count("\n", 0, match.end()) + 1
        depth, bounds = 1, []
        for ln in range(first_line + 1, len(lines) + 1):
            code = lines[ln - 1].split("//")[0]
            depth += code.count("{") - code.count("}")
            if depth <= 0:
                break
            if code.strip() and code.rstrip().endswith(";") and not code.lstrip().startswith(("return", "#", "break", "continue")):
                bounds.append((ln, ln))
    if not bounds:
        raise ValueError("Offline identification found no instrumentable kernel statement")
    candidates = {}
    for start, end in bounds:
        statement = "\n".join(lines[start-1:end])
        for stage, pattern in _OFFLINE_PRIMITIVES.items():
            if stage not in candidates and re.search(pattern, statement, re.I):
                candidates[stage] = (start, end)
                break
    if not candidates:
        # Keep the interval within one statement, so no control-flow edge can
        # enter it in the middle or leave it without executing its endpoint.
        candidates["compute"] = bounds[-1]
    probes = [{"id": stage, "begin": {"role": "worker", "line": start},
               "end": {"role": "worker", "line": end}} for stage, (start, end) in candidates.items()]
    return {"roles": {"worker": "all"}, "probes": probes, "kernel_name": name}


def identify_source(source_path, *, model="qwen4-max", card="auto", offline_heuristic=False,
                    previous=None, feedback=None, llm_caller=None, prefer_llm_on_feedback=False,
                    reasoning_effort=None):
    """Identify or revise a placement. Cache policy is owned by the public API."""
    source_path = Path(source_path).expanduser().resolve()
    raw = source_path.read_bytes()
    text = raw.decode("utf-8")
    language = detect_language(source_path, text)
    route = resolve_route(card, None, language)
    if previous is not None:
        if previous.get("source_sha256") not in (None, hashlib.sha256(raw).hexdigest()):
            raise ValueError("Previous probe.json is bound to different source bytes")
        normalize_probe_spec(previous, text, language=language)
        failed_gates = set(feedback.get("failed") or ()) if isinstance(feedback, Mapping) else set()
        if (not prefer_llm_on_feedback and feedback and feedback.get("enclosing_loop")
                and "pairing" not in failed_gates):
            return revise_probe_spec(previous, feedback, text)
        if not feedback or feedback.get("status") == "PASS":
            return compact_probe_spec(previous)
    if offline_heuristic:
        if previous is not None:
            raise ValueError("No deterministic loop refinement for this feedback")
        spec = _offline_probe_spec(text, language)
    else:
        eff = reasoning_effort or ("max" if feedback is not None else "low")
        data = place_probes(source_path, card=route["card"], language=language, model=model,
                            reasoning_effort=eff, prev_spec=previous, diagnostics=feedback,
                            llm_caller=llm_caller)
        spec = compact_probe_spec(data["probe_spec"])
        if not spec.get("kernel_name"):
            # Infer the containing decorated function from actual endpoints;
            # C++ injection resolves the containing function independently.
            if language in ("cutedsl", "triton"):
                line = spec["probes"][0]["begin"]["line"]
                owners = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef)
                          and n.lineno <= line <= n.end_lineno]
                if owners:
                    spec["kernel_name"] = min(owners, key=lambda n: n.end_lineno - n.lineno).name
    normalize_probe_spec(spec, text, language=language)
    spec.update({key: route[key] for key in ("language", "card", "arch", "platform")})
    spec.update(source_path=str(source_path), source_sha256=hashlib.sha256(raw).hexdigest())
    return spec


def detect_spin_loop_lines(source_text: str) -> List[Tuple[int, int, str]]:
    """Detect 1-based `(start_line, end_line, snippet)` of polling/spin loops in C++ or Python."""
    lines = source_text.splitlines()
    spans: List[Tuple[int, int, str]] = []
    spin_cue = re.compile(
        r"\b(try_wait|test_wait|mbarrier|atomic|ld\.acquire|__hip_atomic|volatile|ready|flag|phase|arrive|spin|poll)\b",
        re.IGNORECASE,
    )
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "#")):
            continue
        if re.match(r"^(?:while|do)\b", stripped):
            window = "\n".join(lines[idx - 1:min(len(lines), idx + 6)])
            if not spin_cue.search(window):
                continue
            if stripped.startswith("while") and (stripped.endswith(";") or ("{" in stripped and "}" in stripped)):
                spans.append((idx, idx, stripped))
                continue
            if "{" in stripped or (idx < len(lines) and "{" in lines[idx]):
                depth = 0
                end_ln = idx
                for j in range(idx, min(len(lines), idx + 40) + 1):
                    depth += lines[j - 1].count("{") - lines[j - 1].count("}")
                    end_ln = j
                    if depth <= 0 and j > idx:
                        break
                spans.append((idx, end_ln, stripped))
            elif stripped.endswith(":"):
                base_indent = len(line) - len(line.lstrip())
                end_ln = idx
                for j in range(idx + 1, min(len(lines), idx + 40) + 1):
                    lj = lines[j - 1]
                    if lj.strip() and not lj.strip().startswith("#"):
                        if (len(lj) - len(lj.lstrip())) <= base_indent:
                            break
                        end_ln = j
                spans.append((idx, end_ln, stripped))
    return spans


def evaluate_probe_predictions(
    pred_spec: Mapping[str, Any],
    gt_spec: Mapping[str, Any],
    source_text: str,
    *,
    line_tolerance: int = 3,
) -> Dict[str, Any]:
    """Compute §4.2 Table VI metrics: Stage Recall, Stage Precision, and Probe Boundary Accuracy."""
    gt_probes = list(gt_spec.get("probes") or gt_spec.get("stages") or [])
    pred_probes = list(pred_spec.get("probes") or pred_spec.get("stages") or [])
    spin_spans = detect_spin_loop_lines(source_text)

    def _role_norm(st: Mapping[str, Any], total_roles: int) -> str:
        if total_roles <= 1:
            return "all"
        r = str(st.get("role") or "").lower()
        if any(k in r for k in ("prod", "dma", "load")):
            return "producer"
        if any(k in r for k in ("cons", "mma", "math", "comp")):
            return "consumer"
        if "epilogue" in r or "store" in r or "combine" in r:
            return "epilogue"
        if "disp" in r or "comm" in r or "sched" in r:
            return "dispatch"
        return r or "all"

    gt_multi = len(gt_spec.get("warp_roles") or []) > 1
    used_pred = set()
    n_hit = 0
    n_boundary_ok = 0
    per_gt_details: List[Dict[str, Any]] = []

    for gt in gt_probes:
        gt_sem = normalize_semantic_id(gt.get("semantic_id") or gt.get("canonical_class"))
        gt_cls = FIVE_CLASS_MAP.get(gt_sem, gt_sem)
        gt_role = _role_norm(gt, 2 if gt_multi else 1)
        gt_b = int(gt.get("begin_line") or 1)
        gt_e = int(gt.get("end_line") or gt_b)
        gt_mode = str(gt.get("mode") or "sync").lower()

        best_j: Optional[int] = None
        best_dist = 10**9
        for j, pr in enumerate(pred_probes):
            if j in used_pred:
                continue
            pr_sem = normalize_semantic_id(pr.get("semantic_id") or pr.get("canonical_class"))
            pr_cls = FIVE_CLASS_MAP.get(pr_sem, pr_sem)
            if pr_sem != gt_sem and pr_cls != gt_cls:
                continue
            pr_role = _role_norm(pr, 2 if gt_multi else 1)
            if gt_multi and gt_role != "all" and pr_role != "all" and pr_role != gt_role:
                continue
            pr_b = int(pr.get("begin_line") or 1)
            pr_e = int(pr.get("end_line") or pr_b)
            overlaps = not (pr_e < gt_b - line_tolerance or pr_b > gt_e + line_tolerance)
            if not overlaps:
                continue
            dist = abs(pr_b - gt_b) + abs(pr_e - gt_e)
            if dist < best_dist:
                best_dist = dist
                best_j = j

        gid = gt.get("probe_id") or gt.get("stage_id") or gt_sem
        if best_j is None:
            per_gt_details.append({"gt_probe": gid, "hit": False, "boundary_ok": False})
            continue

        used_pred.add(best_j)
        n_hit += 1
        pr = pred_probes[best_j]
        pr_b = int(pr.get("begin_line") or 1)
        pr_e = int(pr.get("end_line") or pr_b)

        in_spin = False
        for s_beg, s_end, _snip in spin_spans:
            if s_end > s_beg and ((s_beg < pr_b <= s_end) or (s_beg < pr_e < s_end)):
                in_spin = True
                break

        tight_ok = (abs(pr_b - gt_b) <= line_tolerance) and (abs(pr_e - gt_e) <= line_tolerance)
        if gt_mode == "async" and gt_e > gt_b:
            async_ok = (pr_b <= gt_b + line_tolerance) and (pr_e >= gt_e - line_tolerance)
            boundary_ok = async_ok and (not in_spin)
        else:
            boundary_ok = tight_ok and (not in_spin)

        if boundary_ok:
            n_boundary_ok += 1
        per_gt_details.append({
            "gt_probe": gid,
            "hit": True,
            "boundary_ok": boundary_ok,
            "in_spin_loop": in_spin,
            "pred_lines": [pr_b, pr_e],
            "gt_lines": [gt_b, gt_e],
        })

    n_gt = len(gt_probes)
    n_pred = len(pred_probes)
    recall = (n_hit / n_gt) if n_gt > 0 else 1.0
    precision = (n_hit / n_pred) if n_pred > 0 else 0.0
    boundary_acc = (n_boundary_ok / n_gt) if n_gt > 0 else 1.0
    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_hit": n_hit,
        "n_boundary_ok": n_boundary_ok,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "probe_boundary_accuracy": round(boundary_acc, 4),
        "details": per_gt_details,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="§3.1 Read-Only Semantic Probe Placement -> JSON Probe Spec")
    ap.add_argument("--source", required=True, help="Path to kernel source file")
    ap.add_argument("--card", required=True, choices=("h20", "b300", "mi308x"))
    ap.add_argument("--arch", help="Explicit architecture (e.g. sm_90a, sm_103a, gfx942)")
    ap.add_argument("--language", help="Optional explicit language (cuda_cpp, hip_cpp, triton, cutedsl)")
    ap.add_argument("--model", default="qwen4-max", help="Evaluated LLM (" + ", ".join(SUPPORTED_MODELS) + ")")
    ap.add_argument("--reasoning-effort", default="max")
    ap.add_argument("--mode", default="open_loop", choices=ABLATION_MODES)
    ap.add_argument("--diagnostics-json", help="Optional diagnostics JSON file from previous verification round")
    ap.add_argument("--error-log", "--diagnostics-log", help="Optional plain-text terminal error log from `verify.py`")
    ap.add_argument("--prev-spec", help="Optional previous JSON Probe Spec for closed-loop repair")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--out", required=True, help="Output JSON Probe Spec path")
    ap.add_argument("--out-rewritten", help="Output path for rewritten source in bare_direct/kb_direct modes")
    args = ap.parse_args(list(argv) if argv is not None else None)

    src_path = Path(args.source).expanduser().resolve()
    diag = None
    if args.error_log:
        diag = Path(args.error_log).read_text(encoding="utf-8")
    elif args.diagnostics_json:
        diag = json.loads(Path(args.diagnostics_json).read_text(encoding="utf-8"))
    pspec = json.loads(Path(args.prev_spec).read_text(encoding="utf-8")) if args.prev_spec else None

    res = place_probes(
        src_path,
        card=args.card,
        arch=args.arch,
        language=args.language,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        ablation_mode=args.mode,
        diagnostics=diag,
        prev_spec=pspec,
        round_index=args.round,
    )
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res["probe_spec"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.out_rewritten and res.get("rewritten_source"):
        rw_path = Path(args.out_rewritten).expanduser().resolve()
        rw_path.parent.mkdir(parents=True, exist_ok=True)
        rw_path.write_text(res["rewritten_source"], encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "out": str(out_path),
        "probes": len(res["probe_spec"]["probes"]),
        "warp_roles": len(res["probe_spec"]["warp_roles"]),
        "telemetry": res["telemetry"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
