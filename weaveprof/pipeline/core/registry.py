#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single capability registry for weaveprofiler.

The registry is the only place that maps a supported card to its architecture,
device clock, physical-unit identity, profiler stack and language recorder.
Callers must fail closed when a target is unknown; they must not infer NVIDIA
or AMD from the host unless the user explicitly asks preflight to inspect the
local machine.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "weaveprofiler-backend-registry/v1"
PKG = Path(__file__).resolve().parents[3]

CARD_SPECS: Dict[str, Dict[str, Any]] = {
    "h20": {
        "card": "h20",
        "display_name": "NVIDIA H20",
        "platform": "nv",
        "architectures": ["sm_90"],
        "device_clock": "%globaltimer",
        "cycle_clock": "clock64",
        "physical_id": "%smid",
        "physical_unit": "SM",
        "profiler": {
            "primary": "nsys",
            "fallback": [],
            "sampling_model": "periodic_pm",
            "preferred_sample_hz": 200000,
            "evidence": "measured_on_h20_200khz",
            "note": "默认请求 200 kHz（--gpu-metrics-frequency=200000，配合 --capture-range=cudaProfilerApi 框定热态目标区间避免分段）；最终频率与断档以 capture manifest 为准。",
        },
        "languages": ["cuda_cpp", "triton", "cutedsl"],
    },
    "b300": {
        "card": "b300",
        "display_name": "NVIDIA B300",
        "platform": "nv",
        "architectures": ["sm_100a", "sm_103a"],
        "device_clock": "%globaltimer",
        "cycle_clock": "clock64",
        "physical_id": "%smid",
        "physical_unit": "SM",
        "profiler": {
            "primary": "nsys",
            "fallback": [],
            "sampling_model": "periodic_pm",
            "preferred_sample_hz": 200000,
            "evidence": "measured_on_b300_200khz",
            "note": "B300 的 sm_100a/sm_103a 都必须保留；默认请求 200 kHz 高频采样（配合 --capture-range=cudaProfilerApi），实际采样率与连续性以卡上 manifest 为准。",
        },
        "languages": ["cuda_cpp", "triton", "cutedsl"],
    },
    "mi308x": {
        "card": "mi308x",
        "display_name": "AMD Instinct MI308X",
        "platform": "amd",
        "architectures": ["gfx942"],
        "device_clock": "s_memrealtime",
        "cycle_clock": "s_memtime/readcyclecounter",
        "physical_id": "HW_REG_HW_ID",
        "physical_unit": "CU",
        "profiler": {
            "primary": "rocprof-sys",
            "fallback": ["rocprofv3"],
            "sampling_model": "per_dispatch_rocm_events",
            "preferred_sample_hz": None,
            "amd_smi_coarse_hz_range": [2, 5],
            "evidence": "measured_on_mi308x_gemm_route",
            "note": "Primary MI308X route: rocprof-sys-run --rocm-events collects raw ROCm PM counters and kernel dispatch slices in the same Perfetto proto, paired by same-run/same-dispatch start timestamp with a declared tolerance. Counter evidence is per dispatch; there is no periodic PM sampling equivalent to nsys 200 kHz. amd-smi polling is optional coarse debug context only, never compute or memory utilization. rocprofv3 MfmaUtil is optional independent calibration/fallback, not the primary source.",
        },
        "languages": ["hip_cpp", "triton"],
        "constraints": [
            "persistent kernel must not attach rocprof PMC counters",
            "cross-host communication counters belong to NIC/fabric evidence, not GPU PMC",
        ],
    },
    "sm86": {
        "card": "sm86",
        "display_name": "NVIDIA Ampere sm_86",
        "platform": "nv",
        "architectures": ["sm_86"],
        "device_clock": "%clock64+%globaltimer_anchors",
        "cycle_clock": "clock64",
        "physical_id": "%smid",
        "physical_unit": "SM",
        "profiler": {
            "primary": "nsys",
            "fallback": [],
            "sampling_model": "periodic_pm",
            "preferred_sample_hz": 200000,
            "evidence": "measured_on_local_sm86",
            "note": "Ampere sm_86 (RTX 30xx / A10 / A40 / A6000): defaults to WP_TIMER=1 (%clock64 + OPEN/CLOSE %globaltimer anchors, ~27 ns step) because consumer sm_86 %globaltimer steps at 1024 ns.",
        },
        "languages": ["cuda_cpp", "triton", "cutedsl"],
    },
}

CARD_ALIASES = {
    "h20": "h20",
    "nvidia h20": "h20",
    "mi308x": "mi308x",
    "amd mi308x": "mi308x",
    "amd instinct mi308x": "mi308x",
    "b300": "b300",
    "nvidia b300": "b300",
    "sm86": "sm86",
    "sm-86": "sm86",
    "nvidia sm86": "sm86",
    "rtx3050": "sm86",
    "ampere": "sm86",
}

LANGUAGE_SPECS: Dict[str, Dict[str, Any]] = {
    "cuda_cpp": {
        "language": "cuda_cpp",
        "display_name": "CUDA C++",
        "source_extensions": [".cu", ".cuh"],
        "header": "recorders/cuda_cpp/wp_lite.cuh",
        "recorder": "recorders/cuda_cpp/wp_lite.cuh",
        "sample_gate": "FI_LEVEL",
        "timer_macro": "FI_TIMER",
        "markers": [
            r"__global__", r"cudaLaunchKernel", r"cuda_runtime", r"#include\s*<cuda",
            r"cudaProfilerStart", r"cudaStream",
        ],
    },
    "hip_cpp": {
        "language": "hip_cpp",
        "display_name": "HIP C++",
        "source_extensions": [".hip", ".cpp", ".cc", ".cxx", ".cu"],
        "header": "recorders/hip_cpp/wp_lite_hip.h",
        "recorder": "recorders/hip_cpp/wp_lite_hip.h",
        "sample_gate": "FI_LEVEL",
        "timer_macro": "FI_TIMER",
        "markers": [
            r"#include\s*[<\"]hip/", r"\bhipLaunchKernel\b", r"\bhipMalloc\b",
            r"\bhipMemcpy\b", r"\bHIP_PLATFORM_AMD__\b", r"\b__HIP_PLATFORM_AMD__\b",
            r"\b__HIP_PLATFORM_HCC__\b", r"\bhip_runtime", r"\bhipcc\b",
        ],
    },
    "triton": {
        "language": "triton",
        "display_name": "Triton",
        "source_extensions": [".py"],
        "header": None,
        "recorder_by_platform": {
            "nv": "recorders/triton/wp_triton.py",
            "amd": "recorders/triton/wp_triton_amd.py",
        },
        "markers": [r"import\s+triton", r"@triton\.jit", r"\btl\.load\b", r"\btl\.dot\b"],
    },
    "cutedsl": {
        "language": "cutedsl",
        "display_name": "CUTLASS CuTeDSL",
        "source_extensions": [".py"],
        "header": None,
        "recorder": "recorders/cutedsl/wp_cute.py",
        "markers": [r"import\s+cutlass\.cute", r"from\s+cutlass\s+import\s+cute", r"@cute\.jit"],
    },
}

LANGUAGE_PATTERN_PATHS = {
    ("cuda_cpp", "nv"): "configs/languages/cuda_cpp/patterns.json",
    ("hip_cpp", "amd"): "configs/languages/hip_cpp/patterns.json",
    ("triton", "nv"): "configs/languages/triton/nvidia_patterns.json",
    ("triton", "amd"): "configs/languages/triton/amd_patterns.json",
    ("cutedsl", "nv"): "configs/languages/cutedsl/patterns.json",
}

CARD_FROM_ARCH = {
    "sm_86": "sm86",
    "sm_90": "h20",
    "sm_100a": "b300",
    "sm_103a": "b300",
    "gfx942": "mi308x",
}

PLATFORM_ALIASES = {"nv": "nv", "nvidia": "nv", "cuda": "nv", "amd": "amd", "rocm": "amd", "hip": "amd"}

LANGUAGE_ALIASES = {
    "cuda": "cuda_cpp", "cuda-cpp": "cuda_cpp", "cuda_cpp": "cuda_cpp",
    "hip": "hip_cpp", "hip-cpp": "hip_cpp", "hip_cpp": "hip_cpp",
    "triton": "triton", "triton-nv": "triton", "triton-amd": "triton",
    "cute": "cutedsl", "cutedsl": "cutedsl", "cute-dsl": "cutedsl",
}


class UnsupportedTarget(ValueError):
    """Raised when a target or language cannot be routed without guessing."""

    def __init__(self, message: str, *, kind: str = "unsupported_target", details: Optional[dict] = None):
        super().__init__(message)
        self.kind = kind
        self.details = details or {}

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "message": str(self), "details": self.details,
                "supported_cards": sorted(CARD_SPECS), "supported_languages": sorted(LANGUAGE_SPECS)}


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def normalize_card(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = _norm(value).replace("_", "-")
    if key in ("", "auto", "none", "unknown"):
        return None
    if key.startswith("card-"):
        key = key[5:]
    return CARD_ALIASES.get(key, key if key in CARD_SPECS else None)


def normalize_arch(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = _norm(value)
    if text in ("", "auto", "none", "unknown"):
        return None
    text = text.split(",", 1)[0].strip()
    if text.startswith("compute_"):
        text = "sm_" + text.split("_", 1)[1]
    elif re.match(r"^sm\d", text):
        text = "sm_" + text[2:]
    elif text.startswith("gfx"):
        text = re.sub(r"[:/].*$", "", text)
    return text


def normalize_platform(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = _norm(value)
    if key in ("", "auto", "none", "unknown"):
        return None
    return PLATFORM_ALIASES.get(key)


def normalize_language(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = _norm(value).replace("_", "-")
    if key in ("", "auto", "none", "unknown"):
        return None
    return LANGUAGE_ALIASES.get(key)


def language_pattern_path(language: Optional[str], platform: Optional[str]) -> str:
    """Return the one platform-specific source-pattern library for a route."""
    lang = normalize_language(language)
    platform_name = normalize_platform(platform)
    path = LANGUAGE_PATTERN_PATHS.get((lang, platform_name))
    if not path:
        raise UnsupportedTarget(
            "no language pattern library for language %s on platform %s" %
            (lang or language, platform_name or platform),
            kind="language_platform_conflict",
            details={"language": lang or language, "platform": platform_name or platform})
    return path


pattern_library_path = language_pattern_path


def _sniff_binaries(paths: Iterable[Optional[str]]) -> Optional[str]:
    saw_hip = False
    for raw in paths:
        if not raw:
            continue
        path = Path(str(raw))
        if path.suffix.lower() in (".cubin", ".ptx", ".sass"):
            return "nv"
        if path.suffix.lower() in (".hsaco", ".amdgpu"):
            saw_hip = True
        if not path.is_file():
            continue
        try:
            head = path.read_bytes()[:65536]
        except OSError:
            continue
        if b".hip_fatbin" in head or b"elf64-amdgpu" in head.lower() or b"AMDGPU" in head:
            saw_hip = True
    return "amd" if saw_hip else None


def _target_record(card: str, arch: str) -> Dict[str, Any]:
    spec = CARD_SPECS[card]
    return {
        "card": card,
        "card_display_name": spec["display_name"],
        "platform": spec["platform"],
        "arch": arch,
        "device_clock": spec["device_clock"],
        "cycle_clock": spec["cycle_clock"],
        "physical_id": spec["physical_id"],
        "physical_unit": spec["physical_unit"],
        "profiler": dict(spec["profiler"]),
        "constraints": list(spec.get("constraints") or []),
    }


def resolve_target(*, card: Optional[str] = None, arch: Optional[str] = None,
                   platform: Optional[str] = None,
                   binary_paths: Iterable[Optional[str]] = (),
                   source_texts: Iterable[str] = ()) -> Dict[str, Any]:
    """Resolve one supported target, or raise instead of guessing."""
    card_value = normalize_card(card)
    arch_value = normalize_arch(arch)
    platform_value = normalize_platform(platform)

    if platform_value is None and arch_value is not None:
        if arch_value.startswith("gfx"):
            platform_value = "amd"
        elif arch_value.startswith("sm_"):
            platform_value = "nv"
    if platform_value is None:
        platform_value = _sniff_binaries(binary_paths)
    if platform_value is None and card_value is None and arch_value is None:
        if any("HIP_PLATFORM_AMD" in str(text) or "hipLaunchKernel" in str(text)
               for text in source_texts):
            platform_value = "amd"

    if arch_value is not None and arch_value not in CARD_FROM_ARCH:
        raise UnsupportedTarget(
            "explicit architecture %r is not registered; supported architectures are %s" %
            (arch_value, ", ".join(sorted(CARD_FROM_ARCH))),
            details={"arch": arch_value, "platform": platform_value, "card": card_value})

    inferred = CARD_FROM_ARCH.get(arch_value) if arch_value else None
    if card_value is None:
        card_value = inferred
    elif arch_value is not None and inferred != card_value:
        raise UnsupportedTarget(
            "card %s does not match architecture %s; allowed architectures for %s are %s" %
            (card_value, arch_value, card_value,
             ", ".join(CARD_SPECS[card_value]["architectures"])),
            kind="card_arch_conflict",
            details={"card": card_value, "arch": arch_value})

    if card_value is None and platform_value is not None:
        candidates = [name for name, spec in CARD_SPECS.items() if spec["platform"] == platform_value]
        if len(candidates) == 1:
            card_value = candidates[0]
        elif candidates:
            raise UnsupportedTarget(
                "platform %s has multiple supported cards (%s); provide --arch or --card explicitly" %
                (platform_value, ", ".join(candidates)),
                kind="ambiguous_target",
                details={"platform": platform_value, "candidates": candidates})
    if card_value is None:
        if arch_value:
            raise UnsupportedTarget("architecture %s is not mapped to a supported card" % arch_value,
                                    details={"arch": arch_value})
        raise UnsupportedTarget(
            "target is unspecified; provide --card or --arch (H20: sm_90, B300: sm_100a/sm_103a, MI308X: gfx942)",
            details={"platform": platform_value, "arch": arch_value})
    if card_value not in CARD_SPECS:
        raise UnsupportedTarget("unsupported card %r; supported cards are %s" %
                                (card, ", ".join(sorted(CARD_SPECS))),
                                details={"card": card, "requested_card": card})

    spec = CARD_SPECS[card_value]
    if platform_value is not None and platform_value != spec["platform"]:
        raise UnsupportedTarget(
            "card %s is on platform %s, not %s" % (card_value, spec["platform"], platform_value),
            kind="card_platform_conflict",
            details={"card": card_value, "platform": platform_value})
    if arch_value is None:
        if len(spec["architectures"]) == 1:
            arch_value = spec["architectures"][0]
        else:
            # B300 keeps both real targets. Callers must choose the compiler target;
            # the registry records the full set rather than collapsing it.
            return {**_target_record(card_value, "+".join(spec["architectures"])),
                    "architectures": list(spec["architectures"]),
                    "arch": None,
                    "resolution": "card_architecture_set"}
    if arch_value not in spec["architectures"]:
        raise UnsupportedTarget(
            "card %s supports %s, not %s" % (card_value, ", ".join(spec["architectures"]), arch_value),
            kind="card_arch_conflict",
            details={"card": card_value, "arch": arch_value,
                     "allowed_architectures": list(spec["architectures"])})
    return {**_target_record(card_value, arch_value), "architectures": [arch_value],
            "resolution": "explicit" if card else "inferred_from_arch"}


def _language_for_target(language: str, target: Optional[Mapping[str, Any]]) -> str:
    """Use the card's one C++ recorder for CUDA-style source on AMD.

    AMD source often keeps ``.cu``, ``__global__`` and CUDA-style launch names.
    That source is compiled and instrumented as HIP on MI308X; exposing a second
    ``cuda_cpp`` route would select the NVIDIA timer/header and is invalid.
    """
    if target and target.get("platform") == "amd" and language == "cuda_cpp":
        return "hip_cpp"
    return language


def detect_language(*, language: Optional[str] = None, source_texts: Iterable[str] = (),
                    file_paths: Iterable[Optional[str]] = (),
                    target: Optional[Mapping[str, Any]] = None) -> str:
    explicit = normalize_language(language)
    if explicit:
        return _language_for_target(explicit, target)
    texts: List[str] = list(source_texts)
    names = [str(p) for p in file_paths if p]
    if any(Path(p).suffix.lower() == ".hip" for p in names):
        return "hip_cpp"
    joined = "\n".join(texts)
    marker_checks: Sequence[Tuple[str, Sequence[str]]] = (
        ("cutedsl", LANGUAGE_SPECS["cutedsl"]["markers"]),
        ("triton", LANGUAGE_SPECS["triton"]["markers"]),
        ("hip_cpp", LANGUAGE_SPECS["hip_cpp"]["markers"]),
        ("cuda_cpp", LANGUAGE_SPECS["cuda_cpp"]["markers"]),
    )
    matches = {
        lang for lang, patterns in marker_checks
        if any(re.search(pattern, joined, re.MULTILINE) for pattern in patterns)
    }
    # HIP is deliberately checked before CUDA. A TU carrying a HIP runtime cue
    # and a CUDA compatibility include is still HIP source.
    for lang, _patterns in marker_checks:
        if lang in matches:
            return _language_for_target(lang, target)
    if target and target.get("platform") == "amd":
        raise UnsupportedTarget("AMD target detected but language is not HIP C++ or Triton",
                                kind="unsupported_language", details={"target": dict(target)})
    raise UnsupportedTarget("language cannot be identified; provide --language cuda_cpp|hip_cpp|triton|cutedsl",
                            kind="unsupported_language")


def route(*, language: str, target: Mapping[str, Any]) -> Dict[str, Any]:
    requested = normalize_language(language)
    lang = _language_for_target(requested, target)
    if lang not in LANGUAGE_SPECS:
        raise UnsupportedTarget("unsupported language %r" % language, kind="unsupported_language")
    spec = LANGUAGE_SPECS[lang]
    card = str(target.get("card") or "")
    card_spec = CARD_SPECS.get(card)
    if not card_spec or lang not in card_spec["languages"]:
        raise UnsupportedTarget(
            "language %s is not registered for card %s" % (lang, card or "?"),
            kind="language_card_conflict", details={"language": lang, "card": card})
    recorder = spec.get("recorder")
    if not recorder:
        recorder = (spec.get("recorder_by_platform") or {}).get(str(target.get("platform")))
    if not recorder:
        raise UnsupportedTarget("no recorder for language %s on platform %s" %
                                (lang, target.get("platform")), kind="unsupported_language")
    result = {
        "schema": SCHEMA,
        "language": lang,
        "language_display_name": spec["display_name"],
        "recorder": recorder,
        "language_pattern_library": language_pattern_path(lang, target.get("platform")),
        "source_extensions": list(spec.get("source_extensions") or []),
        "target": dict(target),
        "device_clock": target.get("device_clock"),
        "physical_id": target.get("physical_id"),
        "profiler": dict(target.get("profiler") or {}),
        "constraints": list(target.get("constraints") or []),
        "evidence_status": (target.get("profiler") or {}).get("evidence", "unverified"),
    }
    if requested != lang:
        result["requested_language"] = requested
        result["language_resolution"] = "cuda_style_source_compiled_as_hip_on_amd"
    return result


def resolve(*, card: Optional[str] = None, arch: Optional[str] = None,
            platform: Optional[str] = None, language: Optional[str] = None,
            source_texts: Iterable[str] = (), file_paths: Iterable[Optional[str]] = (),
            binary_paths: Iterable[Optional[str]] = ()) -> Dict[str, Any]:
    target = resolve_target(card=card, arch=arch, platform=platform,
                            binary_paths=binary_paths, source_texts=source_texts)
    lang = detect_language(language=language, source_texts=source_texts,
                           file_paths=file_paths, target=target)
    # Keep an explicitly requested language visible to the audit trail even
    # when the AMD route compiles CUDA-style source through the HIP recorder.
    requested = normalize_language(language)
    return route(language=requested or lang, target=target)


def capabilities() -> Dict[str, Any]:
    rows = []
    for card, spec in sorted(CARD_SPECS.items()):
        for lang in spec["languages"]:
            recorder = LANGUAGE_SPECS[lang].get("recorder") or \
                (LANGUAGE_SPECS[lang].get("recorder_by_platform") or {}).get(spec["platform"])
            rows.append({
                "card": card,
                "platform": spec["platform"],
                "architectures": list(spec["architectures"]),
                "language": lang,
                "recorder": recorder,
                "profiler": spec["profiler"]["primary"],
                "profiler_fallback": list(spec["profiler"].get("fallback") or []),
                "evidence_status": spec["profiler"].get("evidence", "unverified"),
            })
    return {"schema": SCHEMA, "cards": CARD_SPECS, "languages": LANGUAGE_SPECS, "routes": rows}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Resolve one weaveprofiler hardware/language/profiler route")
    ap.add_argument("--card", default="auto")
    ap.add_argument("--arch")
    ap.add_argument("--platform", default="auto", choices=["auto", "nvidia", "nv", "amd", "rocm"])
    ap.add_argument("--language", default="auto")
    ap.add_argument("--source", action="append", default=[], help="source file to inspect")
    ap.add_argument("--binary", action="append", default=[], help="binary artifact to inspect")
    ap.add_argument("--list", action="store_true", help="emit the complete capability registry")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        doc = capabilities()
    else:
        texts = []
        for raw in args.source:
            try:
                texts.append(Path(raw).read_text(encoding="utf-8", errors="replace"))
            except OSError as exc:
                print(json.dumps({"kind": "input_error", "message": str(exc)}, ensure_ascii=False))
                return 2
        try:
            doc = resolve(card=args.card, arch=args.arch, platform=args.platform,
                          language=args.language, source_texts=texts,
                          file_paths=args.source, binary_paths=args.binary)
        except UnsupportedTarget as exc:
            print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2))
            return 2
    print(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
