#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""quick_scan.py -- zero-dependency, read-only, contract-free triage entry.

WHAT THIS IS
------------
A fast heuristic pre-triage for the weaveprofiler skill.  It answers ONE
question, *before* any CONTRACT.lock.json exists:

    "Is the instrumentation infrastructure for this target already complete
     enough to jump straight to measurement (fast path), or must code be
     repaired first (slow path)?"

It requires ONLY a target project version root.  It never reads, writes, or
requires a contract, and it touches nothing under the target (read-only).

WHAT THIS IS **NOT**  (read this before trusting the output)
------------------------------------------------------------
quick_scan is a HEURISTIC pre-check.  It is NOT a substitute for preflight,
it produces NO delivery-grade evidence, and it can authorize NOTHING.

  * The authoritative existence gate is 1_scan/validate_preflight.py, which
    runs E01-E27 against a frozen contract and writes PREFLIGHT.json.
  * Result validity is decided only by the G00-G24 gates via run_measurement.py.
  * Exit code 0 ("fast_path") means "no blocking gap was found heuristically,
    so it is worth locking a contract and running preflight/measurement
    directly" -- it does NOT mean preflight passed.

Do not wire this script into any gate, receipt, or delivery flow.

USAGE
-----
    python3 -B 1_scan/quick_scan.py --version-root <project version root> \
        --card <h20|b300|mi308x> [--arch <arch>] [--language <language>] \
        [--target-kernel <name>] [--json <output path>]

Target routing is resolved through registry.py.  If no target is
given, the report stays diagnostic-only and is forced to slow_path; it never
guesses a card from host tools.

EXIT CODES (use these, do not parse text)
    0   fast_path   -- heuristics found no blocking gap
    20  slow_path   -- heuristics found gaps; repair before measuring
    2   bad input   -- --version-root missing / not a directory
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

# ``python -I 1_scan/quick_scan.py`` intentionally does not add the skill
# root to sys.path.  The backend registry is part of the skill, so add only
# that root here; quick_scan still has no third-party dependency.
SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SKILL_ROOT not in sys.path:
    sys.path.insert(0, SKILL_ROOT)

import registry as backend_registry

# --------------------------------------------------------------------------
# Positioning disclaimer -- surfaced in --help and in every JSON/stdout report
# so nobody can pass quick_scan off as the real gate.
# --------------------------------------------------------------------------
DISCLAIMER = (
    "quick_scan is a HEURISTIC pre-triage only. It is NOT a substitute for "
    "preflight, produces NO delivery-grade evidence, and authorizes nothing. "
    "The authoritative existence gate is 1_scan/validate_preflight.py "
    "(E01-E27); result validity is decided only by the G00-G24 gates. "
    "Exit 0 (fast_path) means 'no blocking gap found heuristically -> worth "
    "locking a contract and running preflight directly', NOT 'preflight passed'."
)

SCHEMA = "fast-instrument-v2/quick-scan/v1"

EXIT_FAST = 0
EXIT_SLOW = 20
EXIT_BAD_INPUT = 2

# --------------------------------------------------------------------------
# Filesystem scanning knobs
# --------------------------------------------------------------------------
SOURCE_EXTS = frozenset({
    ".py", ".cu", ".cuh", ".hip", ".cc", ".cpp", ".cxx", ".c",
    ".h", ".hpp", ".hxx", ".inc", ".ptx",
})

# Directories pruned everywhere.  E01 (SKILL.md 1.3) explicitly demands that
# reference implementations, test data and stale run dirs never be mistaken
# for the project's own implementation.
PRUNE_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", "build", "dist",
    ".work", ".archive", ".instr_v2", ".instr_v2_backup",
    "results", "result",
    "refs", "references", "docs",
    "tests", "test", "testdata", "test_data", "testsuite",
    ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache",
    ".venv", "venv", ".tox", ".idea", ".vscode",
})

# The host tool-chain tree is host-side, not device source.  Excluding it from
# the *device* scan stops e.g. a decoder that mentions "%globaltimer" from
# masquerading as a real device-side clock read.
HOST_TOOLCHAIN_TOP = "profile"

MAX_FILE_BYTES = 4_000_000

# --------------------------------------------------------------------------
# Canonical profile/ tool-chain tree (SKILL.md 1.3, the fixed directory tree).
# Each entry: (relative path under profile/, kind, human label).
# Reference paths (relative to skill root) come from SKILL.md part 3.
# --------------------------------------------------------------------------
REF_DESIGN = "knowledge/guides/design.md"
REF_SLOWPATH = "knowledge/guides/slow-path.md"
REF_CONTRACT = "knowledge/validation/contract_lock_example.json"

TOOLCHAIN_ITEMS = (
    # (rel_path, kind, label, [reference paths])
    ("run_profile.py", "file", "runtime launch harness",
     ["knowledge/examples/host/harness.py"]),
    ("device_record/abi", "dir", "buffer ABI descriptor(s)",
     ["knowledge/examples/host/extract_abi.py", "knowledge/examples/host/abi_reference.json"]),
    ("device_record/presets", "dir", "measurement presets",
     [REF_CONTRACT, REF_DESIGN]),
    ("host_decode/materialize_contract.py", "file", "contract generator",
     [REF_CONTRACT, REF_DESIGN]),
    ("host_decode/decode_profile.py", "file", "raw-buffer decoder",
     ["knowledge/examples/host/decode.py"]),
    ("hw_perfetto/metrics", "dir", "200000 Hz metrics config",
     ["knowledge/guides/nsys_pm_sampling.md"]),
    ("hw_perfetto/capture_nsys.py", "file", "Nsys capture entry",
     ["3_capture/capture_nsys.py", "knowledge/guides/nsys_pm_sampling.md"]),
    ("hw_perfetto/extract_gpu_metrics.py", "file", "SQLite HW reader",
     ["knowledge/examples/host/extract_gpu_metrics.py"]),
    ("hw_perfetto/align_sw_hw.py", "file", "SW/HW alignment",
     ["knowledge/examples/host/merge_hw_into_perfetto.py"]),
    ("hw_perfetto/emit_perfetto.py", "file", "Perfetto emitter",
     ["knowledge/examples/host/build_perfetto.py", "knowledge/examples/host/build_hw_perfetto.py"]),
    ("validation/selftest", "dir", "offline self-test suite",
     ["4_validate/selftest_hw_merge.py", "4_validate/selftest_perfetto.py"]),
    ("validation/run_offline_selftests.py", "file", "offline self-test runner",
     ["4_validate/selftest_hw_merge.py", "4_validate/selftest_perfetto.py"]),
    ("validation/wait_for_gpu_idle.py", "file", "GPU idle wait",
     ["3_capture/capture_nsys.py"]),
    ("validation/check_preflight_item.py", "file", "argv-only preflight adapter",
     ["1_scan/validate_preflight.py"]),
    ("validation/check_gate_item.py", "file", "argv-only gate adapter",
     ["4_validate/validate_all.py"]),
    ("validation/validate_correctness.py", "file", "full correctness entry",
     ["knowledge/examples/host/harness.py"]),
    ("validation/validate_overhead.py", "file", "instrumentation overhead entry",
     ["knowledge/examples/host/harness.py", "knowledge/examples/host/capture_harness.py"]),
)

# --------------------------------------------------------------------------
# Device-side instrumentation traces (E01-E10).
# kind:
#   "kernel_locate" - E01, handled specially.
#   "anywhere"      - present if the pattern matches any non-excluded impl file.
#   "combo"         - present if BOTH patterns match inside the SAME file.
#   "integration"   - E03, tied to whether a primary compiled file calls the
#                     profiler (present) vs only *_instr sidecars (uncertain).
# --------------------------------------------------------------------------
DEVICE_CHECKS = (
    {"id": "E01", "kind": "kernel_locate",
     "name": "target kernel located in compiled source"},
    {"id": "E02", "kind": "anywhere",
     "name": "64-bit device clock read (globaltimer or AMD counter)",
     "pattern": (r"read_ptx_sreg_globaltimer|%%?globaltimer|read_globaltimer|"
                 r"\bs_memrealtime\b|\bs_memtime\b|\breadcyclecounter\b|\bclock64\s*\(")},
    {"id": "E03", "kind": "integration",
     "name": "task/phase start/end boundaries in kernel"},
    {"id": "E04", "kind": "anywhere",
     "name": "compile-time off/entry/full switch",
     "pattern": (r"\bMODE_OFF\b|\bMODE_FULL\b|\bMODE_ENTRY\b|PROFILER_ENABLED|"
                 r"MK_PROFILER|PROFILER_ENTRY_ONLY|kEnableProfiler")},
    {"id": "E05", "kind": "anywhere",
     "name": "source-side SM/CU physical id filter",
     "pattern": (r"read_ptx_sreg_smid|%%?smid|read_smid|physical_smid|"
                 r"capture_slot_of|prof_capture_slot|HW_REG_HW_ID|\bhw_id\b|\bcu_id\b")},
    {"id": "E06", "kind": "anywhere",
     "name": "compact capture_slot mapping",
     "pattern": r"\bcapture_slot\b|\bCAPTURE_SLOTS\b|\bcapture_slots\b"},
    {"id": "E07", "kind": "combo",
     "name": "per-lane write isolation / address formula",
     "patterns": [r"slot_base|slot_elem|HEADER_ELEMS|PROF_HEADER_ELEMS",
                  r"max_records|MAX_RECORDS|kMaxRecords|record_stride"]},
    {"id": "E08", "kind": "combo",
     "name": "buffer capacity + overflow flag",
     "patterns": [r"\boverflow\b",
                  r"max_records|MAX_RECORDS|kMaxRecords|\bcapacity\b"]},
    {"id": "E09", "kind": "combo", "same_file": False,
     "name": "CTA arrive/release entry rendezvous",
     "patterns": [r"arrive|arrive_add|arrive_and_wait|pipeline_init_arrive",
                  r"release|release_add|EntryRankSync|rank_sync|rendezvous|"
                  r"grid_sync|entry_sync"]},
    {"id": "E10", "kind": "anywhere",
     "name": "event id/name table (task/phase)",
     "pattern": (r"EVENT_TABLE|EVENT_NAMES|TASK_NAMES|PHASE_NAMES|"
                 r"\btask_names\b|\bphase_names\b|\bTASK_ROLE\b")},
)

# Files that carry instrumentation but whose "is this the compiled entry?"
# status is unconfirmed (parallel *_instr copies).
SIDECAR_RE = re.compile(r"_instr\.(?:py|cu|cuh|cc|cpp|cxx)$", re.IGNORECASE)
# Path component that marks the profiler *definition* module (not a call site).
PROFILER_PATH_RE = re.compile(r"(^|/)profiler(/|_|\.|$)", re.IGNORECASE)
# A file "uses" the profiler (import alias or call site).
PROFILER_USE_RE = re.compile(
    r"\bprof\.|\bprofiler\.|prof_interval_start|prof_interval_end|prof_record|"
    r"record_start|record_end|\bprobe\s*\(|import\s+profiler|"
    r"from\s+[\w.]*profiler|\bas\s+prof\b",
)
KERNEL_FILE_RE = re.compile(
    r"(^|/)(megakernel|kernel)\w*\.(?:py|cu|cuh|cc|cpp|cxx)$", re.IGNORECASE)
KERNEL_DIR_RE = re.compile(r"(^|/)(kernels|operators)(/|$)", re.IGNORECASE)

# Framework markers come from the backend registry.  Keep the ordering here:
# when a translation unit has both ``__global__`` and HIP runtime markers, HIP
# must win instead of being misreported as CUDA C++.
FRAMEWORK_ORDER = ("cutedsl", "triton", "hip_cpp", "cuda_cpp")


def _compile_framework_markers():
    markers = []
    for language in FRAMEWORK_ORDER:
        patterns = backend_registry.LANGUAGE_SPECS[language].get("markers") or []
        markers.append((language, re.compile("|".join(patterns), re.MULTILINE)))
    return tuple(markers)


FRAMEWORK_MARKERS = _compile_framework_markers()
FRAMEWORK_REF_DIR = {
    "cutedsl": "knowledge/examples/cute_dsl/",
    "triton": "knowledge/examples/triton/",
    "hip_cpp": "knowledge/examples/demo/",
    "cuda_cpp": "knowledge/examples/cuda_cpp/",
}
FRAMEWORK_DEVICE_REFS = {
    "cutedsl": ["knowledge/examples/cute_dsl/profiler.py", "knowledge/examples/cute_dsl/load.py",
                "knowledge/examples/cute_dsl/mma.py", "knowledge/examples/cute_dsl/barrier.py",
                "knowledge/examples/cute_dsl/dispatch.py", "knowledge/examples/cute_dsl/scatter.py"],
    "triton": ["knowledge/examples/triton/profiler.py", "knowledge/examples/triton/load.py",
               "knowledge/examples/triton/mma.py", "knowledge/examples/triton/barrier.py",
               "knowledge/examples/triton/dispatch.py", "knowledge/examples/triton/scatter.py"],
    "hip_cpp": ["2_instrument/recorders/hip_cpp/fi_lite_hip.h", "knowledge/examples/demo/demo_two_roles_hip.cpp",
                "knowledge/guides/amd_mi308x.md"],
    "cuda_cpp": ["knowledge/examples/cuda_cpp/profiler.cuh", "knowledge/examples/cuda_cpp/load.cu",
                 "knowledge/examples/cuda_cpp/mma.cu", "knowledge/examples/cuda_cpp/barrier.cu",
                 "knowledge/examples/cuda_cpp/dispatch.cu", "knowledge/examples/cuda_cpp/scatter.cu"],
}

# Effort tiers (heuristic weighted-gap bands).
TIER_BANDS = (
    (0, 0, "ready"),        # exactly 0 weighted gaps
    (1, 6, "light"),
    (7, 13, "moderate"),
    (14, 10 ** 9, "heavy"),
)


# ==========================================================================
# Filesystem helpers
# ==========================================================================
def read_text(path):
    """Read a source file defensively; never raise."""
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            return ""
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def iter_source_files(root):
    """Yield (abs_path, rel_path) for impl source files, pruning noise dirs.

    Symlinks are not followed; reference/test/stale-run trees are pruned so a
    reference implementation is never mistaken for the project's own code.
    """
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        kept = []
        for name in dirnames:
            if name in PRUNE_DIRS:
                continue
            if name.startswith(".") and name not in (".",):
                continue
            if name.endswith("_backup") or name.endswith(".bak"):
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext not in SOURCE_EXTS:
                continue
            abs_path = os.path.join(dirpath, name)
            if os.path.islink(abs_path) or not os.path.isfile(abs_path):
                continue
            rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")
            yield abs_path, rel_path


def classify_file(rel_path):
    """primary | sidecar | profiler_module -- for integration reasoning."""
    if SIDECAR_RE.search(rel_path):
        return "sidecar"
    if PROFILER_PATH_RE.search(rel_path):
        return "profiler_module"
    return "primary"


def dir_entry_count(path):
    """Count meaningful entries in a directory (ignore hidden / __pycache__)."""
    try:
        names = os.listdir(path)
    except OSError:
        return 0
    count = 0
    for name in names:
        if name.startswith(".") or name == "__pycache__":
            continue
        count += 1
    return count


# ==========================================================================
# Section 1: profile/ tool-chain existence
# ==========================================================================
def check_toolchain(version_root):
    profile_root = os.path.join(version_root, "profile")
    profile_exists = os.path.isdir(profile_root)
    items = []
    hits = 0
    for rel, kind, label, refs in TOOLCHAIN_ITEMS:
        target = os.path.join(profile_root, rel)
        detail = ""
        if kind == "file":
            present = os.path.isfile(target)
            status = "present" if present else "missing"
        else:  # dir
            if not os.path.isdir(target):
                status = "missing"
            else:
                n = dir_entry_count(target)
                detail = "{0} entr{1}".format(n, "y" if n == 1 else "ies")
                status = "present" if n > 0 else "empty"
        if status == "present":
            hits += 1
        items.append({
            "path": "profile/" + rel,
            "kind": kind,
            "label": label,
            "status": status,
            "detail": detail,
            "references": refs,
        })
    return {
        "profile_root": profile_root,
        "profile_root_exists": profile_exists,
        "total": len(TOOLCHAIN_ITEMS),
        "hits": hits,
        "missing_or_empty": [it for it in items if it["status"] != "present"],
        "items": items,
    }


# ==========================================================================
# Section 2: framework identification
# ==========================================================================
def detect_framework(file_texts, explicit_language=None, platform=None):
    """Identify source language using the registry's canonical language keys.

    HIP files often contain ``__global__`` too, so extension and HIP runtime
    markers are evaluated before CUDA.  The returned ``dominant`` value is the
    canonical registry language (``hip_cpp``/``triton``/...), not a display
    alias such as ``cute_dsl``.
    """
    counts = {name: 0 for name, _ in FRAMEWORK_MARKERS}
    for rel, text in file_texts:
        hits = set()
        if str(rel).lower().endswith(".hip"):
            hits.add("hip_cpp")
        for name, rx in FRAMEWORK_MARKERS:
            if rx.search(text):
                hits.add(name)
        for name in hits:
            counts[name] += 1
    priority = {name: index for index, name in enumerate(FRAMEWORK_ORDER)}
    ordered = sorted(counts.items(),
                     key=lambda kv: (-kv[1], priority.get(kv[0], len(priority))))
    detected = ordered[0][0] if ordered and ordered[0][1] > 0 else "unknown"
    raw_explicit = str(explicit_language).strip() if explicit_language is not None else ""
    explicit_given = raw_explicit.lower() not in ("", "auto", "none", "unknown")
    explicit = backend_registry.normalize_language(raw_explicit) if explicit_given else None
    # Preserve an invalid explicit value so registry.route() rejects it instead
    # of silently falling back to a detected language.
    dominant = explicit or (raw_explicit.lower() if explicit_given else detected)
    selected_by = "explicit" if explicit_given else (
        "source_scan" if detected != "unknown" else "unresolved")
    # On an AMD target, a C/C++ translation unit containing ``__global__`` is a
    # HIP kernel even when it does not include a HIP header in the slice that
    # quick_scan can see.  An explicit --language still wins.
    if not explicit_given and dominant == "cuda_cpp" and \
            backend_registry.normalize_platform(platform) == "amd":
        dominant = "hip_cpp"
        selected_by = "target_platform_disambiguation"
    return {
        "dominant": dominant,
        "detected": detected,
        "selected_by": selected_by,
        "file_counts": counts,
        "reference_dir": FRAMEWORK_REF_DIR.get(dominant, "knowledge/examples/demo/"),
    }


def resolve_triage_route(*, card=None, arch=None, platform=None, language=None,
                         file_texts=(), binary_paths=()):
    """Resolve the one hardware/language/profiler route using the registry.

    The helper is intentionally small so callers and tests can exercise route
    resolution without creating a fake profile/ tree.  ``UnsupportedTarget``
    propagates unchanged and callers must fail closed unless they explicitly
    choose to render an unresolved diagnostic.
    """
    return backend_registry.resolve(
        card=card,
        arch=arch,
        platform=platform,
        language=language,
        source_texts=[text for _rel, text in file_texts],
        file_paths=[rel for rel, _text in file_texts],
        binary_paths=binary_paths,
    )


# ==========================================================================
# Section 3: device-side instrumentation traces (E01-E10)
# ==========================================================================
def scan_device(version_root, file_texts, target_kernel):
    # Restrict device scan to non-host-toolchain source.
    device_texts = [(rel, text) for rel, text in file_texts
                    if not (rel == HOST_TOOLCHAIN_TOP
                            or rel.startswith(HOST_TOOLCHAIN_TOP + "/"))]

    # --- profiler integration (drives E01/E03 and the headline metric) ---
    primary_call_files, sidecar_call_files = [], []
    profiler_module_files = []
    for rel, text in device_texts:
        klass = classify_file(rel)
        if klass == "profiler_module":
            if PROFILER_USE_RE.search(text) or "globaltimer" in text or "%smid" in text:
                profiler_module_files.append(rel)
            continue
        if PROFILER_USE_RE.search(text):
            if klass == "sidecar":
                sidecar_call_files.append(rel)
            else:
                primary_call_files.append(rel)

    if primary_call_files:
        integration_verdict = "integrated_in_compiled_path"
    elif sidecar_call_files:
        integration_verdict = "sidecar_only"
    elif profiler_module_files:
        integration_verdict = "defined_not_wired"
    else:
        integration_verdict = "absent"

    # --- kernel location (E01) ---
    kernel_files = [rel for rel, _ in device_texts
                    if KERNEL_FILE_RE.search(rel) or KERNEL_DIR_RE.search(rel)]
    kernel_name_hits = []
    if target_kernel:
        name_rx = re.compile(r"\b" + re.escape(target_kernel) + r"\b")
        kernel_name_hits = [rel for rel, text in device_texts if name_rx.search(text)]

    # --- per-check evaluation ---
    results = []
    for spec in DEVICE_CHECKS:
        cid, kind, name = spec["id"], spec["kind"], spec["name"]
        status, evidence = "absent", []

        if kind == "kernel_locate":
            if target_kernel:
                if kernel_name_hits:
                    status, evidence = "present", kernel_name_hits[:6]
                elif kernel_files:
                    status, evidence = "uncertain", kernel_files[:6]
                else:
                    status = "absent"
            else:
                if kernel_files:
                    status, evidence = "present", kernel_files[:6]

        elif kind == "anywhere":
            rx = re.compile(spec["pattern"])
            evidence = [rel for rel, text in device_texts if rx.search(text)]
            status = "present" if evidence else "absent"

        elif kind == "combo":
            rx0 = re.compile(spec["patterns"][0])
            rx1 = re.compile(spec["patterns"][1])
            if spec.get("same_file", True):
                evidence = [rel for rel, text in device_texts
                            if rx0.search(text) and rx1.search(text)]
                status = "present" if evidence else "absent"
            else:
                files0 = [rel for rel, text in device_texts if rx0.search(text)]
                files1 = [rel for rel, text in device_texts if rx1.search(text)]
                if files0 and files1:
                    seen, evidence = set(), []
                    for rel in files0 + files1:
                        if rel not in seen:
                            seen.add(rel)
                            evidence.append(rel)
                    status = "present"
                else:
                    status = "absent"

        elif kind == "integration":  # E03
            if integration_verdict == "integrated_in_compiled_path":
                status, evidence = "present", primary_call_files[:6]
            elif integration_verdict in ("sidecar_only", "defined_not_wired"):
                status, evidence = "uncertain", (sidecar_call_files
                                                 or profiler_module_files)[:6]
            else:
                status = "absent"

        results.append({
            "id": cid, "name": name, "status": status,
            "evidence_files": evidence[:6],
            "evidence_count": len(evidence),
        })

    present = sum(1 for r in results if r["status"] == "present")
    uncertain = sum(1 for r in results if r["status"] == "uncertain")
    absent = sum(1 for r in results if r["status"] == "absent")

    return {
        "checks": results,
        "present": present,
        "uncertain": uncertain,
        "absent": absent,
        "integration": {
            "verdict": integration_verdict,
            "primary_call_site_files": sorted(set(primary_call_files)),
            "sidecar_call_site_files": sorted(set(sidecar_call_files)),
            "profiler_module_files": sorted(set(profiler_module_files)),
        },
        "kernel_files_found": kernel_files[:12],
        "target_kernel_name_hits": kernel_name_hits[:12],
    }


# ==========================================================================
# Section 4: gaps -> reference paths
# ==========================================================================
def build_gaps(toolchain, device, framework):
    gaps = []

    # Device-side gaps (absent or uncertain E-items) share the framework
    # device references + the design doc.
    dev_refs = FRAMEWORK_DEVICE_REFS.get(framework["dominant"], [REF_DESIGN])
    for chk in device["checks"]:
        if chk["status"] == "present":
            continue
        gaps.append({
            "category": "device",
            "id": chk["id"],
            "what": chk["name"],
            "status": chk["status"],
            "references": [dev_refs[0], REF_DESIGN],
        })
    if device["integration"]["verdict"] in ("sidecar_only", "defined_not_wired", "absent"):
        gaps.append({
            "category": "device",
            "id": "E01/E03-integration",
            "what": ("profiler is not wired into the primary compiled kernel "
                     "(verdict=%s)" % device["integration"]["verdict"]),
            "status": "uncertain" if device["integration"]["verdict"] != "absent" else "absent",
            "references": dev_refs + [REF_DESIGN, REF_SLOWPATH],
        })

    # Tool-chain gaps.
    for item in toolchain["missing_or_empty"]:
        gaps.append({
            "category": "toolchain",
            "id": item["path"],
            "what": "%s (%s)" % (item["label"], item["status"]),
            "status": item["status"],
            "references": item["references"],
        })

    return gaps


# ==========================================================================
# Section 5: effort estimate + fast/slow decision
# ==========================================================================
def estimate_effort(toolchain, device):
    toolchain_missing = toolchain["total"] - toolchain["hits"]
    verdict = device["integration"]["verdict"]
    integration_penalty = {
        "integrated_in_compiled_path": 0,
        "defined_not_wired": 3,
        "sidecar_only": 3,
        "absent": 6,
    }.get(verdict, 3)
    weighted = (toolchain_missing
                + 2 * device["absent"]
                + 1 * device["uncertain"]
                + integration_penalty)
    tier = "heavy"
    for lo, hi, name in TIER_BANDS:
        if lo <= weighted <= hi:
            tier = name
            break
    return {
        "tier": tier,
        "weighted_gap_score": weighted,
        "toolchain_missing": toolchain_missing,
        "device_absent": device["absent"],
        "device_uncertain": device["uncertain"],
        "integration_penalty": integration_penalty,
        "note": ("Heuristic estimate only, NOT a commitment. Real scope is "
                 "decided by validate_preflight.py (E01-E27) and the G-gates."),
    }


def decide_path(toolchain, device):
    reasons = []
    blocking = False

    tm = toolchain["total"] - toolchain["hits"]
    if tm > 0:
        blocking = True
        reasons.append(
            "profile/ tool-chain incomplete: %d/%d present (%d missing/empty)"
            % (toolchain["hits"], toolchain["total"], tm))
    else:
        reasons.append("profile/ tool-chain complete: %d/%d present"
                       % (toolchain["hits"], toolchain["total"]))

    if device["absent"] > 0:
        blocking = True
        reasons.append("device traces missing: %d of E01-E10 absent"
                       % device["absent"])
    if device["uncertain"] > 0:
        blocking = True
        reasons.append("device traces uncertain: %d of E01-E10 unconfirmed"
                       % device["uncertain"])
    if device["absent"] == 0 and device["uncertain"] == 0:
        reasons.append("device traces E01-E10 all heuristically present")

    verdict = device["integration"]["verdict"]
    if verdict != "integrated_in_compiled_path":
        blocking = True
        reasons.append("profiler integration verdict=%s (not wired into the "
                       "primary compiled kernel)" % verdict)
    else:
        reasons.append("profiler integrated into primary compiled kernel "
                       "(%d call-site file(s))"
                       % len(device["integration"]["primary_call_site_files"]))

    return {"path": "slow_path" if blocking else "fast_path", "reasons": reasons}


# ==========================================================================
# Rendering
# ==========================================================================
STATUS_MARK = {"present": "OK ", "missing": "-- ", "empty": "~  ",
               "absent": "-- ", "uncertain": "?  "}


def _wrap(text, width, indent):
    out, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = word if not line else line + " " + word
    if line:
        out.append(line)
    pad = " " * indent
    return ("\n" + pad).join(out) if out else ""


def render_human(report):
    L = []
    a = L.append
    a("=" * 78)
    a("weaveprofiler :: quick_scan (heuristic pre-triage)")
    a("=" * 78)
    a("scanned_at    : %s" % report["scanned_at"])
    a("version_root  : %s" % report["version_root"])
    a("target_kernel : %s" % (report["target_kernel"] or "(not specified)"))
    a("framework     : %s  (device refs: %s)"
      % (report["framework"]["dominant"], report["framework"]["reference_dir"]))
    target = report.get("target") or {}
    arch = target.get("arch") or ",".join(target.get("architectures") or []) or "(unresolved)"
    a("target        : card=%s platform=%s arch=%s"
      % (target.get("card") or "(unresolved)",
         target.get("platform") or "(unresolved)", arch))
    profiler = report.get("profiler") or {}
    a("route         : language=%s recorder=%s profiler=%s evidence_status=%s"
      % (report.get("language") or "(unresolved)",
         report.get("recorder") or "(unresolved)",
         profiler.get("primary") or "(unresolved)",
         report.get("evidence_status") or "unverified"))
    if report.get("route_error"):
        a("route_error   : %s" % report["route_error"]["message"])
    a("")
    a("NOTE: " + _wrap(report["disclaimer"], 70, 6))
    a("")

    dec = report["decision"]
    eff = report["effort"]
    a("-" * 78)
    a(" DECISION : %-9s   (exit code %d)   effort=%s  [heuristic]"
      % (dec["path"].upper(), report["exit_code"], eff["tier"]))
    a("-" * 78)
    for r in dec["reasons"]:
        a("   * " + _wrap(r, 68, 5))
    a("")

    tc = report["toolchain"]
    a("-" * 78)
    a(" profile/ TOOL-CHAIN   hits=%d/%d   (profile/ exists: %s)"
      % (tc["hits"], tc["total"], tc["profile_root_exists"]))
    a("-" * 78)
    for it in tc["items"]:
        det = (" [%s]" % it["detail"]) if it["detail"] else ""
        a("  %s %-42s %s%s"
          % (STATUS_MARK.get(it["status"], "?  "), it["path"], it["status"], det))
    if tc["missing_or_empty"]:
        a("  missing/empty: %d" % len(tc["missing_or_empty"]))
    a("")

    dv = report["device"]
    integ = dv["integration"]
    a("-" * 78)
    a(" DEVICE TRACES  E01-E10   present=%d uncertain=%d absent=%d"
      % (dv["present"], dv["uncertain"], dv["absent"]))
    a("-" * 78)
    for chk in dv["checks"]:
        ev = ("  <- %s%s" % (", ".join(chk["evidence_files"][:2]),
                             (" +%d" % (chk["evidence_count"] - 2))
                             if chk["evidence_count"] > 2 else "")) \
            if chk["evidence_files"] else ""
        a("  %s %-4s %-42s %s"
          % (STATUS_MARK.get(chk["status"], "?  "), chk["id"], chk["name"],
             chk["status"]))
        if ev:
            a("            %s" % _wrap(ev.strip(), 60, 12))
    a("")
    a("  integration verdict : %s" % integ["verdict"])
    a("    primary call-site files : %d %s"
      % (len(integ["primary_call_site_files"]),
         integ["primary_call_site_files"][:4]))
    a("    sidecar call-site files : %d %s"
      % (len(integ["sidecar_call_site_files"]),
         integ["sidecar_call_site_files"][:4]))
    a("")

    gaps = report["gaps"]
    a("-" * 78)
    a(" GAPS -> REFERENCE (SKILL.md part 3 relative paths)   total=%d" % len(gaps))
    a("-" * 78)
    if not gaps:
        a("  (none)")
    for g in gaps:
        a("  [%s] %-18s %s"
          % (g["category"], g["id"], _wrap(g["what"], 52, 30)))
        a("        ref: %s" % ", ".join(g["references"]))
    a("")

    a("-" * 78)
    a(" EFFORT (heuristic, NOT a commitment)")
    a("-" * 78)
    a("  tier=%s  weighted_gap_score=%d" % (eff["tier"], eff["weighted_gap_score"]))
    a("  toolchain_missing=%d  device_absent=%d  device_uncertain=%d  integration_penalty=%d"
      % (eff["toolchain_missing"], eff["device_absent"],
         eff["device_uncertain"], eff["integration_penalty"]))
    a("  " + _wrap(eff["note"], 72, 2))
    a("=" * 78)
    return "\n".join(L)


# ==========================================================================
# Main
# ==========================================================================
def build_report(version_root, target_kernel, *, card=None, arch=None,
                 platform=None, language=None, allow_unresolved=False):
    file_texts = [(rel, read_text(ap)) for ap, rel in iter_source_files(version_root)]
    toolchain = check_toolchain(version_root)
    target_platform = backend_registry.normalize_platform(platform)
    if any(_argument_given(value) for value in (card, arch, platform)):
        try:
            preview = backend_registry.resolve_target(
                card=card, arch=arch, platform=platform)
            target_platform = preview["platform"]
        except backend_registry.UnsupportedTarget:
            target_platform = backend_registry.normalize_platform(platform)
    framework = detect_framework(file_texts, explicit_language=language,
                                 platform=target_platform)
    route = None
    route_error = None
    route_language = framework["dominant"] if framework["dominant"] != "unknown" else None
    try:
        route = resolve_triage_route(card=card, arch=arch, platform=platform,
                                     language=route_language, file_texts=file_texts)
    except backend_registry.UnsupportedTarget as exc:
        if not allow_unresolved:
            raise
        route_error = exc.as_dict()

    if route is not None:
        target = route["target"]
        route_language = route["language"]
        recorder = route["recorder"]
        profiler = route["profiler"]
        evidence_status = route["evidence_status"]
    else:
        target = {
            "card": backend_registry.normalize_card(card) or card,
            "platform": backend_registry.normalize_platform(platform) or platform,
            "arch": backend_registry.normalize_arch(arch) or arch,
            "architectures": [],
        }
        recorder = None
        profiler = None
        evidence_status = "unverified"
    device = scan_device(version_root, file_texts, target_kernel)
    gaps = build_gaps(toolchain, device, framework)
    effort = estimate_effort(toolchain, device)
    decision = decide_path(toolchain, device)
    if route_error:
        decision["path"] = "slow_path"
        decision["reasons"].insert(
            0, "target route unresolved (no fast-path authorization): %s"
               % route_error["message"])
    exit_code = EXIT_FAST if decision["path"] == "fast_path" else EXIT_SLOW
    return {
        "schema": SCHEMA,
        "disclaimer": DISCLAIMER,
        "authoritative_gate": "1_scan/validate_preflight.py (E01-E27); "
                              "result validity via G00-G24",
        "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "version_root": os.path.abspath(version_root),
        "target_kernel": target_kernel,
        "files_scanned": len(file_texts),
        "target": target,
        "language": route_language,
        "recorder": recorder,
        "profiler": profiler,
        "evidence_status": evidence_status,
        "route_resolved": route is not None,
        "route_error": route_error,
        "backend_route": route,
        "framework": framework,
        "toolchain": toolchain,
        "device": device,
        "gaps": gaps,
        "effort": effort,
        "decision": decision,
        "exit_code": exit_code,
    }


def _write_json(path, document):
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _argument_given(value):
    return value is not None and str(value).strip().lower() not in (
        "", "auto", "none", "unknown")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="quick_scan.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Zero-dependency, read-only, contract-free triage for "
                    "weaveprofiler: decide fast_path vs slow_path and resolve "
                    "the card/language/profiler route before writing any "
                    "CONTRACT.lock.json.",
        epilog="POSITIONING\n-----------\n" + _wrap(DISCLAIMER, 76, 0)
        + "\n\nEXIT CODES\n  0  fast_path   (no blocking gap found heuristically)"
          "\n  20 slow_path   (gaps found; repair before measuring)"
          "\n  2  bad input   (--version-root missing / not a directory)",
        allow_abbrev=False,
    )
    parser.add_argument("--version-root", required=True,
                        help="target project version root directory (required)")
    parser.add_argument("--target-kernel", default=None,
                        help="optional kernel name to locate (refines E01)")
    parser.add_argument("--card", default=None,
                        help="target card: h20, b300, or mi308x")
    parser.add_argument("--arch", default=None,
                        help="target architecture, e.g. sm_90, sm_100a, sm_103a, gfx942")
    parser.add_argument("--platform", default=None,
                        help="target platform: nv/nvidia or amd/rocm; auto is not guessed")
    parser.add_argument("--language", default=None,
                        help="source language: cuda_cpp, hip_cpp, triton, or cutedsl")
    parser.add_argument("--json", default=None,
                        help="optional path to write the machine-readable report")
    args = parser.parse_args(argv)

    version_root = args.version_root
    if not os.path.isdir(version_root):
        sys.stderr.write("ERROR: --version-root is not a directory: %s\n"
                         % version_root)
        return EXIT_BAD_INPUT

    allow_unresolved = not any(_argument_given(value) for value in
                               (args.card, args.arch, args.platform))
    try:
        report = build_report(version_root, args.target_kernel,
                              card=args.card, arch=args.arch,
                              platform=args.platform, language=args.language,
                              allow_unresolved=allow_unresolved)
    except backend_registry.UnsupportedTarget as exc:
        sys.stderr.write("ERROR: target route failed closed: %s\n" % exc)
        if args.json:
            error_doc = {
                "schema": SCHEMA,
                "error": exc.as_dict(),
                "target": {
                    "card": args.card,
                    "platform": args.platform,
                    "arch": args.arch,
                },
                "language": args.language,
                "recorder": None,
                "profiler": None,
                "evidence_status": "unverified",
                "route_resolved": False,
            }
            try:
                _write_json(args.json, error_doc)
            except OSError as error:
                sys.stderr.write("ERROR: cannot write --json %s: %s\n"
                                 % (args.json, error))
        return EXIT_BAD_INPUT

    if args.json:
        try:
            _write_json(args.json, report)
        except OSError as error:
            sys.stderr.write("ERROR: cannot write --json %s: %s\n"
                             % (args.json, error))
            return EXIT_BAD_INPUT

    print(render_human(report))
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
