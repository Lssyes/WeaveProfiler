#!/usr/bin/env python3
"""Offline contract tests for knowledge/languages/**/patterns.json.

The library is intentionally data-only.  These tests enforce the shared
schema, complete canonical coverage, stable trace names, real language APIs,
and the safety rules that keep a recorder from writing outside its slot.
"""
from __future__ import annotations

import copy
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN_DIR = ROOT / "knowledge" / "languages"
SEMANTIC_PATH = ROOT / "knowledge" / "semantic_ids.json"

EXPECTED_FILES = {
    "cuda_cpp/patterns.json": ("cuda_cpp", "nv", {"h20", "b300"}),
    "hip_cpp/patterns.json": ("hip_cpp", "amd", {"mi308x"}),
    "triton/nvidia_patterns.json": ("triton_nv", "nv", {"h20", "b300"}),
    "triton/amd_patterns.json": ("triton_amd", "amd", {"mi308x"}),
    "cutedsl/patterns.json": ("cutedsl", "nv", {"h20", "b300"}),
}
TOP_LEVEL_KEYS = {
    "schema",
    "version",
    "language",
    "backend",
    "targets",
    "evidence_status",
    "semantic_policy",
    "record_contract",
    "required_semantic_ids",
    "conventions",
    "patterns",
}
PATTERN_KEYS = {
    "semantic_id",
    "display_name",
    "evidence_status",
    "source_cues",
    "source_example",
    "event_boundaries",
    "record_scope",
    "writer_rule",
    "buffer_rule",
    "language_notes",
    "expected_trace_name",
    "validation",
}
POLICY_KEYS = {
    "canonical_source",
    "semantic_id_is_display_name",
    "expected_trace_name_is_semantic_id",
    "unknown_fails_closed",
    "source_marker_is_hint_only",
}
CONTRACT_KEYS = {"record_layout", "event_id_range", "levels", "clock"}
CONVENTION_KEYS = {
    "compiler",
    "collector",
    "event_ids",
    "levels",
    "scope_selection",
    "name_policy",
    "evidence_policy",
    "safety",
}
ALLOWED_EVIDENCE = {"measured", "compiled", "simulated", "unverified", "reference"}
PLACEHOLDER_RE = re.compile(r"(?i)\b(?:TODO|TBD|FIXME|pseudocode|placeholder)\b|\.\.\.")

LANGUAGE_API_KEYWORDS = {
    "cuda_cpp": ("#include <cuda_runtime.h>", '#include "fi_lite.cuh"', "FI_OPEN(", "FI_B(", "FI_E(", "FI_CLOSE("),
    "hip_cpp": ("#include <hip/hip_runtime.h>", '#include "fi_lite_hip.h"', "FI_OPEN(", "FI_B(", "FI_E(", "FI_CLOSE("),
    "triton_nv": ("@triton.jit", "from fi_triton import", "fi_open(", "fi_b(", "fi_e(", "fi_close(", "tl."),
    "triton_amd": ("@triton.jit", "from fi_triton_amd import", "fi_open(", "fi_b(", "fi_e(", "fi_close(", "FI_CAP", "FI_THREADS"),
    "cutedsl": ("@cute.jit", "from fi_cute import", "fi_open(", "fi_b(", "fi_e(", "fi_close(", "<< 32) |", "ARCH", "BACKEND", "GRAN"),
}

SEMANTIC_TOKENS = {
    "cuda_cpp": {
        "gemm": ("fmaf", "FI_B("),
        "attention": ("expf", "FI_B("),
        "softmax": ("expf", "FI_B("),
        "reduction": ("__shfl_down_sync", "FI_B("),
        "moe": ("expert", "FI_B("),
        "convolution": ("conv3x3", "FI_B("),
        "megakernel": ("atomicAdd", "for (;;)"),
        "load": ("src[index]", "FI_B("),
        "wait": ("while", "FI_B("),
        "mma": ("wmma::mma_sync", "FI_B("),
        "compute": ("fmaf", "FI_B("),
        "epilogue": ("__float2half_rn", "FI_B("),
        "store": ("dst[index] =", "FI_B("),
        "barrier": ("__syncthreads", "FI_B("),
        "dispatch": ("atomicAdd", "FI_B("),
        "communication": ("__threadfence_system", "FI_B("),
    },
    "hip_cpp": {
        "gemm": ("fmaf", "FI_B("),
        "attention": ("expf", "FI_B("),
        "softmax": ("expf", "FI_B("),
        "reduction": ("__shfl_down", "FI_B("),
        "moe": ("expert", "FI_B("),
        "convolution": ("conv3x3", "FI_B("),
        "megakernel": ("atomicAdd", "for (;;)"),
        "load": ("src[index]", "FI_B("),
        "wait": ("while", "FI_B("),
        "mma": ("__builtin_amdgcn_mfma", "FI_B("),
        "compute": ("fmaf", "FI_B("),
        "epilogue": ("__float2half_rn", "FI_B("),
        "store": ("dst[index] =", "FI_B("),
        "barrier": ("__syncthreads", "FI_B("),
        "dispatch": ("atomicAdd", "FI_B("),
        "communication": ("__threadfence_system", "FI_B("),
    },
    "triton_nv": {
        "gemm": ("tl.dot(", "fi_b("),
        "attention": ("tl.exp(", "fi_b("),
        "softmax": ("tl.exp(", "fi_b("),
        "reduction": ("tl.sum(", "fi_b("),
        "moe": ("expert", "fi_b("),
        "convolution": ("patch", "fi_b("),
        "megakernel": ("tl.range(", "fi_b("),
        "load": ("tl.load(", "fi_b("),
        "wait": ("while ready", "fi_b("),
        "mma": ("tl.dot(", "fi_b("),
        "compute": ("tl.exp2(", "fi_b("),
        "epilogue": ("to(tl.bfloat16)", "fi_b("),
        "store": ("tl.store(", "fi_b("),
        "barrier": ("tl.debug_barrier(", "fi_b("),
        "dispatch": ("tl.atomic_add(", "fi_b("),
        "communication": ("peer_ready", "fi_b("),
    },
    "triton_amd": {
        "gemm": ("tl.dot(", "fi_b("),
        "attention": ("tl.exp(", "fi_b("),
        "softmax": ("tl.exp(", "fi_b("),
        "reduction": ("tl.sum(", "fi_b("),
        "moe": ("expert", "fi_b("),
        "convolution": ("patch", "fi_b("),
        "megakernel": ("tl.range(", "fi_b("),
        "load": ("tl.load(", "fi_b("),
        "wait": ("while ready", "fi_b("),
        "mma": ("tl.dot(", "fi_b("),
        "compute": ("tl.exp2(", "fi_b("),
        "epilogue": ("to(tl.bfloat16)", "fi_b("),
        "store": ("tl.store(", "fi_b("),
        "barrier": ("tl.debug_barrier(", "fi_b("),
        "dispatch": ("tl.atomic_add(", "fi_b("),
        "communication": ("peer_ready", "fi_b("),
    },
    "cutedsl": {
        "gemm": ("cute.gemm(", "fi_b("),
        "attention": ("cute.gemm(", "fi_b("),
        "softmax": ("exp2", "fi_b("),
        "reduction": ("values.reduce(", "fi_b("),
        "moe": ("expert", "fi_b("),
        "convolution": ("im2col", "fi_b("),
        "megakernel": ("grid_sync", "fi_b("),
        "load": ("cute.copy(", "fi_b("),
        "wait": ("cp_async_wait_group(", "fi_b("),
        "mma": ("cute.gemm(", "fi_b("),
        "compute": ("exp2", "fi_b("),
        "epilogue": ("Float16", "fi_b("),
        "store": ("cute.copy(", "fi_b("),
        "barrier": ("cute.arch.barrier(", "fi_b("),
        "dispatch": ("atomic_add(", "fi_b("),
        "communication": ("peer_ready", "fi_b("),
    },
}

SAFETY_PHRASES = {
    "triton_nv": ("n < cap", "warp < warps", "slot < n_slots", "ev & 0x0FFF"),
    "triton_amd": ("n < CAP", "slot < n_slots", "EVENT & 0x0FFF", "wave64"),
    "hip_cpp": ("linear_tid", "threadIdx.x alone is invalid", "linear_tid & 63", "n < cap"),
    "cuda_cpp": ("linear_tid", "max_records", "n_slots", "overflow"),
    "cutedsl": ("(aux << 32) | tag", "slot < n_slots", "count < cap", "arch", "backend", "gran=warp|cta"),
}


def load_semantics() -> tuple[list[str], set[str]]:
    doc = json.loads(SEMANTIC_PATH.read_text(encoding="utf-8"))
    ordered = list(doc["required_ids"])
    canonical = {entry["semantic_id"] for entry in doc["semantics"]}
    return ordered, canonical


def validate_document(doc: dict, filename: str) -> None:
    expected_language, expected_backend, expected_targets = EXPECTED_FILES[filename]
    if set(doc) != TOP_LEVEL_KEYS:
        missing = TOP_LEVEL_KEYS - set(doc)
        extra = set(doc) - TOP_LEVEL_KEYS
        raise AssertionError(f"{filename}: top-level keys mismatch; missing={sorted(missing)} extra={sorted(extra)}")
    if doc["schema"] != "weaveprofiler-code-patterns/v1":
        raise AssertionError(f"{filename}: wrong schema {doc['schema']!r}")
    if doc["version"] != 1:
        raise AssertionError(f"{filename}: version must be 1")
    if doc["language"] != expected_language:
        raise AssertionError(f"{filename}: language {doc['language']!r} != {expected_language!r}")
    if doc["backend"] != expected_backend:
        raise AssertionError(f"{filename}: backend {doc['backend']!r} != {expected_backend!r}")
    if set(doc["targets"]) != expected_targets:
        raise AssertionError(f"{filename}: targets {doc['targets']!r} != {sorted(expected_targets)!r}")
    if doc["evidence_status"] not in ALLOWED_EVIDENCE:
        raise AssertionError(f"{filename}: invalid evidence_status")
    if doc["evidence_status"] == "measured":
        raise AssertionError(f"{filename}: checked-in reference library cannot claim measured without card evidence")

    policy = doc["semantic_policy"]
    if set(policy) != POLICY_KEYS:
        raise AssertionError(f"{filename}: semantic_policy keys mismatch")
    if not all(policy[key] is True for key in POLICY_KEYS - {"canonical_source"}):
        raise AssertionError(f"{filename}: semantic policy flags must all be true")
    if policy["canonical_source"] != "knowledge/semantic_ids.json":
        raise AssertionError(f"{filename}: canonical_source must point to knowledge/semantic_ids.json")

    contract = doc["record_contract"]
    if set(contract) != CONTRACT_KEYS:
        raise AssertionError(f"{filename}: record_contract keys mismatch")
    if contract["levels"] != {"0": "off", "1": "entry_only", "2": "full"}:
        raise AssertionError(f"{filename}: record levels changed")
    if not contract["record_layout"] or not contract["event_id_range"] or not contract["clock"]:
        raise AssertionError(f"{filename}: incomplete record contract")

    if set(doc["conventions"]) != CONVENTION_KEYS:
        raise AssertionError(f"{filename}: conventions keys mismatch")

    canonical_order, canonical_ids = load_semantics()
    expected_required = [sid for sid in canonical_order if sid != "unknown"]
    if doc["required_semantic_ids"] != expected_required:
        raise AssertionError(f"{filename}: required_semantic_ids must equal canonical order minus unknown")

    patterns = doc["patterns"]
    if not isinstance(patterns, list) or not patterns:
        raise AssertionError(f"{filename}: patterns must be a non-empty list")
    seen: set[str] = set()
    for index, pattern in enumerate(patterns):
        where = f"{filename}:patterns[{index}]"
        if set(pattern) != PATTERN_KEYS:
            missing = PATTERN_KEYS - set(pattern)
            extra = set(pattern) - PATTERN_KEYS
            raise AssertionError(f"{where}: pattern keys mismatch; missing={sorted(missing)} extra={sorted(extra)}")
        sid = pattern["semantic_id"]
        if sid == "unknown" or sid not in canonical_ids:
            raise AssertionError(f"{where}: unknown or non-canonical semantic_id {sid!r}")
        if sid in seen:
            raise AssertionError(f"{where}: duplicate semantic_id {sid!r}")
        seen.add(sid)
        if pattern["display_name"] != sid:
            raise AssertionError(f"{where}: display_name must equal semantic_id")
        if pattern["expected_trace_name"] != sid:
            raise AssertionError(f"{where}: expected_trace_name must equal semantic_id")
        if pattern["evidence_status"] not in ALLOWED_EVIDENCE:
            raise AssertionError(f"{where}: invalid evidence_status")
        if pattern["evidence_status"] == "measured":
            raise AssertionError(f"{where}: checked-in reference pattern cannot claim measured")
        if sid not in pattern["source_cues"]:
            raise AssertionError(f"{where}: canonical id must appear in source_cues")
        if not isinstance(pattern["source_cues"], list) or len(pattern["source_cues"]) < 3:
            raise AssertionError(f"{where}: source_cues must contain at least three cues")
        if not isinstance(pattern["source_example"], str) or len(pattern["source_example"].strip()) < 80:
            raise AssertionError(f"{where}: source_example is empty or too small to be usable")
        if PLACEHOLDER_RE.search(pattern["source_example"]):
            raise AssertionError(f"{where}: source_example contains a placeholder or ellipsis")
        boundaries = pattern["event_boundaries"]
        if not isinstance(boundaries, dict) or set(boundaries) != {"begin", "end"}:
            raise AssertionError(f"{where}: event_boundaries must have begin and end")
        if not all(isinstance(value, str) and value.strip() for value in boundaries.values()):
            raise AssertionError(f"{where}: empty event boundary")
        for field in ("record_scope", "writer_rule", "buffer_rule", "expected_trace_name"):
            if not isinstance(pattern[field], str) or not pattern[field].strip():
                raise AssertionError(f"{where}: {field} must be a non-empty string")
        if not isinstance(pattern["language_notes"], list) or not pattern["language_notes"]:
            raise AssertionError(f"{where}: language_notes must be a non-empty list")
        if not isinstance(pattern["validation"], list) or len(pattern["validation"]) < 3:
            raise AssertionError(f"{where}: validation must contain at least three checks")

    if seen != set(expected_required):
        missing = set(expected_required) - seen
        extra = seen - set(expected_required)
        raise AssertionError(f"{filename}: canonical coverage mismatch; missing={sorted(missing)} extra={sorted(extra)}")

    language = doc["language"]
    for pattern in patterns:
        source = pattern["source_example"]
        for keyword in LANGUAGE_API_KEYWORDS[language]:
            if keyword not in source:
                raise AssertionError(f"{filename}:{pattern['semantic_id']}: missing API keyword {keyword!r}")
        for keyword in SEMANTIC_TOKENS[language][pattern["semantic_id"]]:
            if keyword not in source:
                raise AssertionError(f"{filename}:{pattern['semantic_id']}: source example misses {keyword!r}")
        for phrase in SAFETY_PHRASES[language]:
            if phrase not in pattern["buffer_rule"] and phrase not in pattern["writer_rule"]:
                raise AssertionError(f"{filename}:{pattern['semantic_id']}: safety rule misses {phrase!r}")


class CodePatternLibraryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = {}
        for filename in EXPECTED_FILES:
            path = PATTERN_DIR / filename
            if not path.is_file():
                raise AssertionError(f"missing {path}")
            cls.docs[filename] = json.loads(path.read_text(encoding="utf-8"))

    def test_all_five_documents_share_schema_and_full_canonical_coverage(self):
        semantic_order, canonical_ids = load_semantics()
        required = [sid for sid in semantic_order if sid != "unknown"]
        total = 0
        for filename, doc in self.docs.items():
            validate_document(doc, filename)
            total += len(doc["patterns"])
            self.assertEqual(len(doc["patterns"]), len(required))
            self.assertEqual({p["semantic_id"] for p in doc["patterns"]}, set(required) - {"unknown"})
        self.assertEqual(total, len(EXPECTED_FILES) * len(required))
        print(f"CODE PATTERNS PASS: {len(EXPECTED_FILES)} files, {total} patterns, {len(required)} canonical IDs each")

    def test_trace_names_are_canonical_and_never_custom_display_names(self):
        for filename, doc in self.docs.items():
            for pattern in doc["patterns"]:
                self.assertEqual(pattern["display_name"], pattern["semantic_id"], filename)
                self.assertEqual(pattern["expected_trace_name"], pattern["semantic_id"], filename)
                self.assertNotIn("unknown", pattern["expected_trace_name"].lower())

    def test_source_examples_are_real_language_patterns(self):
        for filename, doc in self.docs.items():
            language = doc["language"]
            for pattern in doc["patterns"]:
                source = pattern["source_example"]
                self.assertGreater(len(source.strip()), 80, (filename, pattern["semantic_id"]))
                self.assertNotRegex(source, PLACEHOLDER_RE, (filename, pattern["semantic_id"]))
                for keyword in LANGUAGE_API_KEYWORDS[language]:
                    self.assertIn(keyword, source, (filename, pattern["semantic_id"], keyword))

    def test_safety_constraints_are_present_for_every_language(self):
        for filename, doc in self.docs.items():
            language = doc["language"]
            for pattern in doc["patterns"]:
                safety_text = pattern["writer_rule"] + "\n" + pattern["buffer_rule"] + "\n" + "\n".join(pattern["language_notes"])
                for phrase in SAFETY_PHRASES[language]:
                    self.assertIn(phrase, safety_text, (filename, pattern["semantic_id"], phrase))

    def test_duplicate_id_fails(self):
        doc = copy.deepcopy(self.docs["triton/nvidia_patterns.json"])
        doc["patterns"].append(copy.deepcopy(doc["patterns"][0]))
        with self.assertRaisesRegex(AssertionError, "duplicate semantic_id"):
            validate_document(doc, "triton/nvidia_patterns.json")

    def test_unknown_semantic_id_fails(self):
        doc = copy.deepcopy(self.docs["cuda_cpp/patterns.json"])
        doc["patterns"][0]["semantic_id"] = "unknown"
        doc["patterns"][0]["display_name"] = "unknown"
        doc["patterns"][0]["expected_trace_name"] = "unknown"
        with self.assertRaisesRegex(AssertionError, "unknown or non-canonical"):
            validate_document(doc, "cuda_cpp/patterns.json")

    def test_missing_required_field_fails(self):
        doc = copy.deepcopy(self.docs["cutedsl/patterns.json"])
        del doc["patterns"][0]["buffer_rule"]
        with self.assertRaisesRegex(AssertionError, "pattern keys mismatch"):
            validate_document(doc, "cutedsl/patterns.json")

    def test_missing_canonical_pattern_fails(self):
        doc = copy.deepcopy(self.docs["hip_cpp/patterns.json"])
        doc["patterns"].pop()
        with self.assertRaisesRegex(AssertionError, "canonical coverage mismatch"):
            validate_document(doc, "hip_cpp/patterns.json")

    def test_noncanonical_display_name_fails(self):
        doc = copy.deepcopy(self.docs["triton/amd_patterns.json"])
        doc["patterns"][0]["display_name"] = "MatrixMultiply"
        with self.assertRaisesRegex(AssertionError, "display_name"):
            validate_document(doc, "triton/amd_patterns.json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
