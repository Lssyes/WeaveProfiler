#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Characterization tests for `weaveprof/backend.py` AMD disassembly & spill gates (B6-G1)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.backend import _sequence, assess  # noqa: E402
from weaveprof.contract import spill_delta_bytes  # noqa: E402


FIXED_HMAC_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

# FILITE02 binary tag bits (literal constants per `fi_lite_decode.py:80-88`; never imported to compute expectations)
TAG_END = 0x80000000
TAG_OPEN = 0x40000000
TAG_CLOSE = 0x08000000

# Synthetic gfx942 AMDGCN inputs adapted from `tests/test_triton_amdgcn_async.py:20`
# (`SYNTHETIC_GFX942_AMDGCN`) and structured after the two tool formats supported by
# `weaveprof/pipeline/validate/tools/fi_binsim_hip.py:288-316` (`hipcc --save-temps` `.s`
# and `llvm-objdump -d` without / with raw hex instruction bytes). Copied inline so tests
# never read `tests/test_triton_amdgcn_async.py` at runtime.
GFX942_HIPCC_SAVE_TEMPS_ASM = """\t.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
\t.text
\t.globl\tgemm_amd_wp
__device_stub__gemm_amd_wp:
\ts_nop 0x0                                     ; host wrapper stub ignored by extract_kernel
gemm_amd_wp:                                    ; @gemm_amd_wp
.LBB0_0:
\ts_load_dwordx4 s[0:3], s[4:5], 0x0            ; load kernarg descriptors
\ts_waitcnt lgkmcnt(0)
\tglobal_load_lds_dword v0, s[0:1] offset:0     ; async global->LDS
\tds_write_b128 v128, v[8:11]
.LBB0_1:
\ts_barrier
\tds_read_b128 v[16:19], v128
\tv_mfma_f32_16x16x16_f16 v[4:7], v[16:17], v[18:19], v[4:7] ; tile mma 0
\tv_mfma_f32_16x16x16_f16 v[8:11], v[20:21], v[22:23], v[8:11] ; tile mma 1
\tv_fmac_f32_e32 v4, v2, v3                     ; epilogue scale
.LBB0_2:
\ts_waitcnt vmcnt(0)
\tbuffer_store_dwordx4 v[4:7], v0, s[20:23], 0 offen ; output store
\ts_endpgm
helper_other_kernel:
\tv_mov_b32_e32 v0, 0
\ts_endpgm
"""

GFX942_HIPCC_INSTRUMENTED_CLEAN_ASM = """\t.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
\t.text
\t.globl\tgemm_amd_wp
gemm_amd_wp:
.LBB0_0:
\ts_memrealtime s[6:7]                          ; probe begin timestamp
\ts_load_dwordx4 s[0:3], s[4:5], 0x0
\ts_waitcnt lgkmcnt(0)
\tglobal_load_lds_dword v0, s[0:1] offset:0
\tds_write_b128 v128, v[8:11]
.LBB0_1:
\ts_barrier
\tds_read_b128 v[16:19], v128
\tv_mfma_f32_16x16x16_f16 v[4:7], v[16:17], v[18:19], v[4:7]
\tv_mfma_f32_16x16x16_f16 v[8:11], v[20:21], v[22:23], v[8:11]
\tv_fmac_f32_e32 v4, v2, v3
.LBB0_2:
\ts_memrealtime s[8:9]                          ; probe end timestamp
\ts_add_i32 s10, s8, 16
\ts_waitcnt vmcnt(0)
\tbuffer_store_dwordx4 v[4:7], v0, s[20:23], 0 offen
\ts_endpgm
"""

GFX942_OBJDUMP_NO_RAW_ASM = """kernel.hsaco:\tfile format elf64-amdgpu

Disassembly of section .text:

0000000000001000 <__device_stub__gemm_amd_wp>:
    1000: s_nop 0x0 // host stub must not match

0000000000002000 <gemm_amd_wp>:
    2000: s_load_dwordx4 s[0:3], s[4:5], 0x0 // 000000002000: C00A0002 00000000
    2008: s_waitcnt lgkmcnt(0)
.LBB0_0:
    200c: global_load_lds_dword v0, s[0:1] offset:0
    2014: ds_write_b128 v128, v[8:11]
.LBB0_1:
    201c: s_barrier
    2020: ds_read_b128 v[16:19], v128
    2028: v_mfma_f32_16x16x16_f16 v[4:7], v[16:17], v[18:19], v[4:7] // core mma 0
    2030: v_mfma_f32_16x16x16_f16 v[8:11], v[20:21], v[22:23], v[8:11] // core mma 1
    2038: v_fmac_f32_e32 v4, v2, v3
.LBB0_2:
    203c: s_waitcnt vmcnt(0)
    2040: buffer_store_dwordx4 v[4:7], v0, s[20:23], 0 offen
    2048: s_endpgm

0000000000003000 <trailing_kernel>:
    3000: v_mov_b32_e32 v0, 0
    3004: s_endpgm
"""

GFX942_OBJDUMP_WITH_RAW_ASM = """kernel.hsaco:\tfile format elf64-amdgpu

Disassembly of section .text:

0000000000002000 <gemm_amd_wp>:
    2000: 02 00 0a c0 00 00 00 00      s_load_dwordx4 s[0:3], s[4:5], 0x0 // with raw encoding bytes
    2008: 7f 00 8c bf                  s_waitcnt lgkmcnt(0)
    200c: 00 00 81 dc 00 00 00 00      global_load_lds_dword v0, s[0:1] offset:0
    2014: 00 80 3f d8 08 00 00 00      ds_write_b128 v128, v[8:11]
    201c: 00 00 8a bf                  s_barrier
    2020: 00 80 ff d8 00 10 00 00      ds_read_b128 v[16:19], v128
    2028: 04 00 94 d3 10 21 12 04      v_mfma_f32_16x16x16_f16 v[4:7], v[16:17], v[18:19], v[4:7]
    2030: 08 00 94 d3 14 29 16 04      v_mfma_f32_16x16x16_f16 v[8:11], v[20:21], v[22:23], v[8:11]
    2038: 02 07 08 2b                  v_fmac_f32_e32 v4, v2, v3
    203c: 70 0f 8c bf                  s_waitcnt vmcnt(0)
    2040: 00 10 7c e0 00 04 05 80      buffer_store_dwordx4 v[4:7], v0, s[20:23], 0 offen
    2048: 00 00 81 bf                  s_endpgm
"""

EXPECTED_GEMM_AMD_OPCODES = [
    "s_load_dwordx4",
    "s_waitcnt",
    "global_load_lds_dword",
    "ds_write_b128",
    "s_barrier",
    "ds_read_b128",
    "v_mfma_f32_16x16x16_f16",
    "v_mfma_f32_16x16x16_f16",
    "v_fmac_f32_e32",
    "s_waitcnt",
    "buffer_store_dwordx4",
    "s_endpgm",
]


def _write_amd_filite02_and_names(directory: Path) -> None:
    names = {
        "events": {"1": "load", "3": "mma"},
        "semantics": {"1": "load", "3": "mma"},
        "probe_ids": {"1": "load", "3": "mma"},
        "roles": {"0": "wave"},
        "group": "role",
    }
    (directory / "names.json").write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")

    meta = json.dumps(
        {"gran": 0, "arch": "gfx942", "backend": "amd_hip", "timer_hz": 100_000_000}
    ).encode("utf-8")
    # 100 MHz ticks (1 tick = 10 ns): 4 tiles of (load=15 us, mma=30 us) = 8 spans/lane, 180 us active
    records = [(100_000, TAG_OPEN | 0, 0)]
    tick = 100_100
    for _ in range(4):
        records.append((tick, 1, 0))
        records.append((tick + 1_500, TAG_END | 1, 0))
        tick += 1_550
        records.append((tick, 3, 0))
        records.append((tick + 3_000, TAG_END | 3, 0))
        tick += 3_050
    records.append((tick + 200, TAG_CLOSE, 0))
    blob = bytearray(b"FILITE02")
    blob.extend(struct.pack("<6I", 1, len(records), 1, 0, 0, len(meta)))
    blob.extend(meta)
    blob.extend(struct.pack("<1I", len(records)))
    for rec in records:
        blob.extend(struct.pack("<QII", *rec))
    (directory / "capture.fi02.bin").write_bytes(blob)


def _populate_amd_round_dir(
    directory: Path,
    *,
    baseline_asm: str = GFX942_HIPCC_SAVE_TEMPS_ASM,
    instrumented_asm: str = GFX942_HIPCC_INSTRUMENTED_CLEAN_ASM,
    baseline_spill_bytes=0,
    instrumented_spill_bytes=0,
    extra_runtime: dict | None = None,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "l0.s").write_text(baseline_asm, encoding="utf-8")
    (directory / "inst.s").write_text(instrumented_asm, encoding="utf-8")
    (directory / "l0.bin").write_bytes(b"\x11\x22\x33\x44" * 16)
    (directory / "inst.bin").write_bytes(b"\x11\x22\x33\x44" * 16)
    _write_amd_filite02_and_names(directory)

    runtime = {
        "platform": "amd",
        "card": "mi308x",
        "kernel_name": "gemm_amd_wp",
        "baseline_sass": "l0.s",
        "instrumented_sass": "inst.s",
        "baseline_spill_bytes": baseline_spill_bytes,
        "instrumented_spill_bytes": instrumented_spill_bytes,
        "extra_smem_bytes": 0,
        "baseline_output": "l0.bin",
        "instrumented_output": "inst.bin",
        "warmup": 4,
        "timing_protocol": "ABBA",
        "baseline_ns": [50_000, 50_100, 49_950, 50_050, 50_000, 50_020],
        "instrumented_ns": [51_000, 51_100, 50_950, 51_050, 51_000, 51_020],
    }
    if extra_runtime:
        runtime.update(extra_runtime)
    (directory / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    return runtime


def _amd_spec(*, include_platform: bool = True, include_card: bool = True) -> dict:
    spec = {
        "probes": [
            {"id": "load", "event_id": 1},
            {"id": "mma", "event_id": 3},
        ],
    }
    if include_platform:
        spec["platform"] = "amd"
    if include_card:
        spec["card"] = "mi308x"
    return spec


def _with_extra_inst_ops(extra_lines: list[str]) -> str:
    extra_block = "".join(f"\t{ln}\n" for ln in extra_lines)
    return GFX942_HIPCC_INSTRUMENTED_CLEAN_ASM.replace(
        "\ts_endpgm\n",
        f"{extra_block}\ts_endpgm\n",
    )


class BackendAssessAmdCharacterizationTests(unittest.TestCase):
    _ENV_SAVE_KEYS = (
        "HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "MPLCONFIGDIR",
        "WEAVEPROF_SESSIONS_DIR",
        "WEAVEPROF_CACHE",
        "WEAVEPROF_LOCK_DIR",
        "WEAVEPROF_WORKDIR",
        "TMPDIR",
        "WEAVEPROF_SUPERVISOR_HMAC_KEY",
        "WEAVEPROF_SUPERVISOR_KEY_FILE",
        "QWEN_PROJECT_DIR",
        "QWEN_SESSION_ID",
        "WEAVEPROF_LOCK_PROVIDER",
        "WEAVEPROF_KEEP_SHADOW",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "FI_SAMPLE",
        "WP_SAMPLE",
        "WEAVEPROF_SAMPLE",
        "PYTHONDONTWRITEBYTECODE",
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls._saved_cwd = os.getcwd()
        cls._saved_tempdir = tempfile.tempdir
        cls._saved_env = {k: os.environ.get(k) for k in cls._ENV_SAVE_KEYS}

        cls._iso_root = Path(tempfile.mkdtemp(prefix="wp_b6_g1_iso_")).resolve()
        for sub in ("home", "xdg_cache", "xdg_config", "mpl", "sessions", "cache", "locks", "workdir", "tmp", "cwd"):
            (cls._iso_root / sub).mkdir(parents=True, exist_ok=True)

        os.environ["HOME"] = str(cls._iso_root / "home")
        os.environ["XDG_CACHE_HOME"] = str(cls._iso_root / "xdg_cache")
        os.environ["XDG_CONFIG_HOME"] = str(cls._iso_root / "xdg_config")
        os.environ["MPLCONFIGDIR"] = str(cls._iso_root / "mpl")
        os.environ["WEAVEPROF_SESSIONS_DIR"] = str(cls._iso_root / "sessions")
        os.environ["WEAVEPROF_CACHE"] = str(cls._iso_root / "cache")
        os.environ["WEAVEPROF_LOCK_DIR"] = str(cls._iso_root / "locks")
        os.environ["WEAVEPROF_WORKDIR"] = str(cls._iso_root / "workdir")
        os.environ["TMPDIR"] = str(cls._iso_root / "tmp")
        os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = FIXED_HMAC_KEY
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

        for unset_key in (
            "WEAVEPROF_SUPERVISOR_KEY_FILE",
            "QWEN_PROJECT_DIR",
            "QWEN_SESSION_ID",
            "WEAVEPROF_LOCK_PROVIDER",
            "WEAVEPROF_KEEP_SHADOW",
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "FI_SAMPLE",
            "WP_SAMPLE",
            "WEAVEPROF_SAMPLE",
        ):
            os.environ.pop(unset_key, None)

        tempfile.tempdir = str(cls._iso_root / "tmp")
        os.chdir(cls._iso_root / "cwd")

    @classmethod
    def tearDownClass(cls) -> None:
        os.chdir(cls._saved_cwd)
        tempfile.tempdir = cls._saved_tempdir
        for k, val in cls._saved_env.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
        shutil.rmtree(cls._iso_root, ignore_errors=True)

    def _new_case_dir(self, tag: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=f"{tag}_", dir=str(self._iso_root / "tmp"))).resolve()

    def test_sequence_amd_hipcc_save_temps_and_label_stripping(self) -> None:
        case_dir = self._new_case_dir("seq_hipcc")
        asm_path = case_dir / "gemm_save_temps.s"
        asm_path.write_text(GFX942_HIPCC_SAVE_TEMPS_ASM, encoding="utf-8")

        seq = _sequence(asm_path, "gemm_amd_wp", amd=True)
        self.assertEqual(seq, EXPECTED_GEMM_AMD_OPCODES)
        self.assertEqual(len(seq), 12)
        self.assertNotIn("LABEL", seq)
        self.assertNotIn("s_nop", seq)
        self.assertNotIn("v_mov_b32_e32", seq)

    def test_sequence_amd_llvm_objdump_without_and_with_raw_bytes(self) -> None:
        case_dir = self._new_case_dir("seq_objdump")
        no_raw_path = case_dir / "objdump_no_raw.txt"
        with_raw_path = case_dir / "objdump_with_raw.txt"
        no_raw_path.write_text(GFX942_OBJDUMP_NO_RAW_ASM, encoding="utf-8")
        with_raw_path.write_text(GFX942_OBJDUMP_WITH_RAW_ASM, encoding="utf-8")

        seq_no_raw = _sequence(no_raw_path, "gemm_amd_wp", amd=True)
        seq_with_raw = _sequence(with_raw_path, "gemm_amd_wp", amd=True)
        self.assertEqual(seq_no_raw, EXPECTED_GEMM_AMD_OPCODES)
        self.assertEqual(seq_with_raw, EXPECTED_GEMM_AMD_OPCODES)
        self.assertNotIn("LABEL", seq_no_raw)
        self.assertNotIn("LABEL", seq_with_raw)

    def test_sequence_amd_missing_kernel_and_host_stub_only_errors(self) -> None:
        case_dir = self._new_case_dir("seq_errors")
        asm_path = case_dir / "l0.s"
        asm_path.write_text(GFX942_HIPCC_SAVE_TEMPS_ASM, encoding="utf-8")

        with self.assertRaises(ValueError) as missing_ctx:
            _sequence(asm_path, "missing_kernel_xyz", amd=True)
        self.assertEqual(
            str(missing_ctx.exception).replace(str(case_dir), "<TMP>"),
            "<TMP>/l0.s: function 'missing_kernel_xyz' not found",
        )

        stub_s_path = case_dir / "stub_only.s"
        stub_s_path.write_text(
            "\t.text\n"
            "__device_stub__gemm_amd_wp:\n"
            "\ts_nop 0x0\n"
            "gemm_amd_wp.cold:\n"
            "\ts_endpgm\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError) as stub_s_ctx:
            _sequence(stub_s_path, "gemm_amd_wp", amd=True)
        self.assertEqual(
            str(stub_s_ctx.exception).replace(str(case_dir), "<TMP>"),
            "<TMP>/stub_only.s: function 'gemm_amd_wp' not found",
        )

        stub_obj_path = case_dir / "stub_objdump.txt"
        stub_obj_path.write_text(
            "kernel.out:\tfile format elf64-x86-64\n\n"
            "Disassembly of section .text:\n\n"
            "0000000000001000 <__device_stub__gemm_amd_wp>:\n"
            "    1000: s_nop 0x0\n\n"
            "0000000000001020 <gemm_amd_wp.cold>:\n"
            "    1020: s_endpgm\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError) as stub_obj_ctx:
            _sequence(stub_obj_path, "gemm_amd_wp", amd=True)
        self.assertEqual(
            str(stub_obj_ctx.exception).replace(str(case_dir), "<TMP>"),
            "<TMP>/stub_objdump.txt: function 'gemm_amd_wp' not found",
        )

    def test_assess_amd_compile_gate_pass_and_spec_platform_selector(self) -> None:
        # 1) Full AMD 4-gate PASS via `spec["platform"] == "amd"` + `runtime={"platform": "amd", ...}`
        dir_platform = self._new_case_dir("assess_amd_platform")
        rt_platform = _populate_amd_round_dir(dir_platform)
        checks, metrics, errors, fires, _trace, _summary = assess(
            dir_platform,
            _amd_spec(include_platform=True, include_card=False),
            0.10,
            rt_platform,
        )
        self.assertEqual(
            checks,
            {"assembly": True, "output": True, "pairing": True, "overhead": True},
        )
        self.assertEqual(errors, {})
        self.assertEqual(fires, {"load": 4, "mma": 4})
        self.assertEqual(metrics["lcs_ratio"], 1.0)
        self.assertEqual(metrics["core_retention_ratio"], 1.0)
        self.assertEqual(metrics["core_multiset_ratio"], 1.0)
        self.assertEqual(metrics["assembly_change_rate"], 0.0)
        self.assertEqual(metrics["order_change_ratio"], 0.0)
        self.assertEqual(metrics["baseline_spill_bytes"], 0)
        self.assertEqual(metrics["instrumented_spill_bytes"], 0)
        self.assertEqual(metrics["delta_spill_bytes"], 0)
        self.assertEqual(metrics["extra_smem_bytes"], 0)
        self.assertEqual(metrics["output_bytes"], 64)
        self.assertEqual(metrics["overflow"], 0)
        self.assertEqual(metrics["paired_samples"], 6)
        self.assertEqual(metrics["baseline_median_ns"], 50010.0)
        self.assertEqual(metrics["instrumented_median_ns"], 51010.0)
        self.assertAlmostEqual(metrics["overhead_ratio"], 0.019996000799840097, places=12)
        self.assertEqual(
            metrics["compiler_spill_gate"],
            {
                "ok": True,
                "register_count_blind_spot": False,
                "violations": [],
                "remediation": [],
                "metrics": {
                    "l0_regs": None,
                    "inst_regs": None,
                    "delta_spill_stores_bytes": 0,
                    "delta_spill_loads_bytes": None,
                    "delta_stack_bytes": None,
                    "delta_sass_stl": 0,
                    "delta_sass_ldl": 0,
                },
            },
        )

        # 2) `spec["card"] == "mi308x"` (without `spec["platform"]`) using llvm-objdump inputs
        dir_card = self._new_case_dir("assess_amd_card")
        rt_card = _populate_amd_round_dir(
            dir_card,
            baseline_asm=GFX942_OBJDUMP_NO_RAW_ASM,
            instrumented_asm=GFX942_OBJDUMP_WITH_RAW_ASM,
        )
        checks_card, metrics_card, errors_card, _, _, _ = assess(
            dir_card,
            _amd_spec(include_platform=False, include_card=True),
            0.10,
            rt_card,
        )
        self.assertEqual(
            checks_card,
            {"assembly": True, "output": True, "pairing": True, "overhead": True},
        )
        self.assertEqual(errors_card, {})
        self.assertEqual(metrics_card["lcs_ratio"], 1.0)

        # 3) Pin BASE `backend.py:65` behavior: `runtime={"platform": "amd"}` alone without
        # `spec["platform"]`/`spec["card"]` falls into NVIDIA `parse_sass` and fails assembly.
        dir_no_spec_flag = self._new_case_dir("assess_runtime_only_amd")
        rt_only = _populate_amd_round_dir(dir_no_spec_flag)
        checks_only, _, errors_only, _, _, _ = assess(
            dir_no_spec_flag,
            _amd_spec(include_platform=False, include_card=False),
            0.10,
            rt_only,
        )
        self.assertEqual(
            checks_only,
            {"assembly": False, "output": True, "pairing": True, "overhead": True},
        )
        self.assertEqual(
            errors_only["assembly"].replace(str(dir_no_spec_flag), "<TMP>"),
            "<TMP>/l0.s: expected one compiled function matching 'gemm_amd_wp', found 0",
        )

    def test_assess_amd_spill_instruction_counting_prefixes_and_thresholds(self) -> None:
        # At default `HarnessPolicy()` limit (`max_delta_stl == 4`, `max_delta_ldl == 4`):
        # +4 stores (`scratch_store*` + `buffer_store_dword*`) and +4 loads (`scratch_load*` + `buffer_load_dword*`)
        dir_at_limit = self._new_case_dir("amd_spill_ops_limit")
        inst_at_limit = _with_extra_inst_ops([
            "scratch_store_dword off, v30, s32",
            "SCRATCH_STORE_DWORDX2 off, v[30:31], s32",
            "buffer_store_dword v30, off, s[20:23], s32",
            "buffer_store_dwordx4 v[30:33], off, s[20:23], s32",
            "scratch_load_dword v30, off, s32",
            "SCRATCH_LOAD_DWORDX2 v[30:31], off, s32",
            "buffer_load_dword v30, off, s[20:23], s32",
            "buffer_load_dwordx4 v[30:33], off, s[20:23], s32",
        ])
        rt_at_limit = _populate_amd_round_dir(dir_at_limit, instrumented_asm=inst_at_limit)
        checks_ok, metrics_ok, errors_ok, _, _, _ = assess(
            dir_at_limit, _amd_spec(), 0.10, rt_at_limit
        )
        self.assertEqual(
            checks_ok,
            {"assembly": True, "output": True, "pairing": True, "overhead": True},
        )
        self.assertEqual(errors_ok, {})
        self.assertIs(metrics_ok["compiler_spill_gate"]["ok"], True)
        self.assertEqual(metrics_ok["compiler_spill_gate"]["violations"], [])
        self.assertEqual(metrics_ok["compiler_spill_gate"]["metrics"]["delta_sass_stl"], 4)
        self.assertEqual(metrics_ok["compiler_spill_gate"]["metrics"]["delta_sass_ldl"], 4)

        # +5 stores -> `SASS_STL_EXPLOSION` (`delta_sass_stl == 5 > 4`)
        dir_over_stl = self._new_case_dir("amd_spill_ops_stl")
        inst_over_stl = _with_extra_inst_ops([
            "scratch_store_dword off, v30, s32",
            "scratch_store_dwordx2 off, v[30:31], s32",
            "buffer_store_dword v30, off, s[20:23], s32",
            "buffer_store_dwordx2 v[30:31], off, s[20:23], s32",
            "buffer_store_dwordx4 v[30:33], off, s[20:23], s32",
        ])
        rt_over_stl = _populate_amd_round_dir(dir_over_stl, instrumented_asm=inst_over_stl)
        checks_stl, metrics_stl, errors_stl, _, _, _ = assess(
            dir_over_stl, _amd_spec(), 0.10, rt_over_stl
        )
        self.assertEqual(
            checks_stl,
            {"assembly": False, "output": True, "pairing": True, "overhead": True},
        )
        self.assertEqual(errors_stl, {})
        self.assertIs(metrics_stl["compiler_spill_gate"]["ok"], False)
        self.assertEqual(
            metrics_stl["compiler_spill_gate"]["violations"],
            [
                {
                    "code": "SASS_STL_EXPLOSION",
                    "delta": 5,
                    "limit": 4,
                    "message": "SASS STL (local store) instructions jumped by +5 (1 -> 6).",
                }
            ],
        )

        # +5 loads -> `SASS_LDL_EXPLOSION` (`delta_sass_ldl == 5 > 4`)
        dir_over_ldl = self._new_case_dir("amd_spill_ops_ldl")
        inst_over_ldl = _with_extra_inst_ops([
            "scratch_load_dword v30, off, s32",
            "scratch_load_dwordx2 v[30:31], off, s32",
            "buffer_load_dword v30, off, s[20:23], s32",
            "buffer_load_dwordx2 v[30:31], off, s[20:23], s32",
            "buffer_load_dwordx4 v[30:33], off, s[20:23], s32",
        ])
        rt_over_ldl = _populate_amd_round_dir(dir_over_ldl, instrumented_asm=inst_over_ldl)
        checks_ldl, metrics_ldl, errors_ldl, _, _, _ = assess(
            dir_over_ldl, _amd_spec(), 0.10, rt_over_ldl
        )
        self.assertEqual(
            checks_ldl,
            {"assembly": False, "output": True, "pairing": True, "overhead": True},
        )
        self.assertEqual(errors_ldl, {})
        self.assertIs(metrics_ldl["compiler_spill_gate"]["ok"], False)
        self.assertEqual(
            metrics_ldl["compiler_spill_gate"]["violations"],
            [
                {
                    "code": "SASS_LDL_EXPLOSION",
                    "delta": 5,
                    "limit": 4,
                    "message": "SASS LDL (local load) instructions jumped by +5 (0 -> 5).",
                }
            ],
        )

    def test_assess_amd_nonzero_baseline_spill_default_policy_boundary(self) -> None:
        # Default `HarnessPolicy()` (no explicit `max_delta_spill_bytes` argument; default limit is 0)
        # 1) Exact boundary: baseline == instrumented == 64 -> delta == 0 -> PASS
        dir_eq = self._new_case_dir("nonzero_spill_eq")
        rt_eq = _populate_amd_round_dir(
            dir_eq,
            baseline_spill_bytes=64,
            instrumented_spill_bytes=64,
        )
        checks_eq, metrics_eq, errors_eq, _, _, _ = assess(
            dir_eq, _amd_spec(), 0.10, rt_eq
        )
        self.assertEqual(
            checks_eq,
            {"assembly": True, "output": True, "pairing": True, "overhead": True},
        )
        self.assertEqual(errors_eq, {})
        self.assertEqual(metrics_eq["baseline_spill_bytes"], 64)
        self.assertEqual(metrics_eq["instrumented_spill_bytes"], 64)
        self.assertEqual(metrics_eq["delta_spill_bytes"], 0)
        self.assertIs(metrics_eq["compiler_spill_gate"]["ok"], True)
        self.assertEqual(metrics_eq["compiler_spill_gate"]["metrics"]["delta_spill_stores_bytes"], 0)

        # 2) Just over boundary: baseline == 64, instrumented == 65 -> delta == +1 -> FAIL
        dir_over = self._new_case_dir("nonzero_spill_over")
        rt_over = _populate_amd_round_dir(
            dir_over,
            baseline_spill_bytes=64,
            instrumented_spill_bytes=65,
        )
        checks_over, metrics_over, errors_over, _, _, _ = assess(
            dir_over, _amd_spec(), 0.10, rt_over
        )
        self.assertEqual(
            checks_over,
            {"assembly": False, "output": True, "pairing": True, "overhead": True},
        )
        self.assertEqual(errors_over, {})
        self.assertEqual(metrics_over["baseline_spill_bytes"], 64)
        self.assertEqual(metrics_over["instrumented_spill_bytes"], 65)
        self.assertEqual(metrics_over["delta_spill_bytes"], 1)
        self.assertIs(metrics_over["compiler_spill_gate"]["ok"], False)
        self.assertEqual(
            metrics_over["compiler_spill_gate"]["violations"],
            [
                {
                    "code": "SPILL_STORE_INCREASE",
                    "delta": 1,
                    "limit": 0,
                    "message": "Spill stores increased by +1 B (64 -> 65 B).",
                }
            ],
        )

        # 3) Spill decreased relative to non-zero baseline: 64 -> 48 -> delta == -16 -> PASS
        dir_dec = self._new_case_dir("nonzero_spill_dec")
        rt_dec = _populate_amd_round_dir(
            dir_dec,
            baseline_spill_bytes=64,
            instrumented_spill_bytes=48,
        )
        checks_dec, metrics_dec, errors_dec, _, _, _ = assess(
            dir_dec, _amd_spec(), 0.10, rt_dec
        )
        self.assertEqual(
            checks_dec,
            {"assembly": True, "output": True, "pairing": True, "overhead": True},
        )
        self.assertEqual(errors_dec, {})
        self.assertEqual(metrics_dec["baseline_spill_bytes"], 64)
        self.assertEqual(metrics_dec["instrumented_spill_bytes"], 48)
        self.assertEqual(metrics_dec["delta_spill_bytes"], -16)
        self.assertIs(metrics_dec["compiler_spill_gate"]["ok"], True)
        self.assertEqual(metrics_dec["compiler_spill_gate"]["metrics"]["delta_spill_stores_bytes"], -16)

    def test_spill_delta_bytes_error_paths_direct_and_via_assess(self) -> None:
        # Error path 1: reported `delta_spill_bytes` disagrees with `instrumented - baseline`
        with self.assertRaises(ValueError) as disagree_ctx:
            spill_delta_bytes(64, 64, reported=4)
        self.assertEqual(
            str(disagree_ctx.exception),
            "delta_spill_bytes disagrees with instrumented minus baseline spill bytes",
        )

        dir_disagree = self._new_case_dir("spill_err_disagree")
        rt_disagree = _populate_amd_round_dir(
            dir_disagree,
            baseline_spill_bytes=64,
            instrumented_spill_bytes=64,
            extra_runtime={"delta_spill_bytes": 4},
        )
        checks_dis, _, errors_dis, _, _, _ = assess(
            dir_disagree, _amd_spec(), 0.10, rt_disagree
        )
        self.assertEqual(
            checks_dis,
            {"assembly": False, "output": True, "pairing": True, "overhead": True},
        )
        self.assertEqual(
            errors_dis,
            {"assembly": "delta_spill_bytes disagrees with instrumented minus baseline spill bytes"},
        )

        # Error path 2a: negative spill byte counts
        for bad_base, bad_inst in ((-4, 0), (0, -1)):
            with self.assertRaises(ValueError) as neg_ctx:
                spill_delta_bytes(bad_base, bad_inst)
            self.assertEqual(
                str(neg_ctx.exception),
                "Expected nonnegative spill byte counts for both builds",
            )

        dir_neg = self._new_case_dir("spill_err_neg")
        rt_neg = _populate_amd_round_dir(
            dir_neg,
            baseline_spill_bytes=-8,
            instrumented_spill_bytes=0,
        )
        checks_neg, _, errors_neg, _, _, _ = assess(
            dir_neg, _amd_spec(), 0.10, rt_neg
        )
        self.assertEqual(
            checks_neg,
            {"assembly": False, "output": True, "pairing": True, "overhead": True},
        )
        self.assertEqual(
            errors_neg,
            {"assembly": "Expected nonnegative spill byte counts for both builds"},
        )

        # Error path 2b: non-integer spill counts (`float`, `str`, `bool`) and non-integer `reported`
        for bad_base, bad_inst in ((64.0, 64), (64, "64"), (True, 0), (None, 0)):
            with self.assertRaises(ValueError) as type_ctx:
                spill_delta_bytes(bad_base, bad_inst)
            self.assertEqual(
                str(type_ctx.exception),
                "Expected nonnegative spill byte counts for both builds",
            )

        for bad_reported in (0.0, "0", False):
            with self.assertRaises(ValueError) as rep_ctx:
                spill_delta_bytes(64, 64, reported=bad_reported)
            self.assertEqual(
                str(rep_ctx.exception),
                "delta_spill_bytes must be a signed integer",
            )

        dir_bad_rep = self._new_case_dir("spill_err_rep_float")
        rt_bad_rep = _populate_amd_round_dir(
            dir_bad_rep,
            baseline_spill_bytes=64,
            instrumented_spill_bytes=64,
            extra_runtime={"delta_spill_bytes": 0.0},
        )
        checks_rep, _, errors_rep, _, _, _ = assess(
            dir_bad_rep, _amd_spec(), 0.10, rt_bad_rep
        )
        self.assertEqual(
            checks_rep,
            {"assembly": False, "output": True, "pairing": True, "overhead": True},
        )
        self.assertEqual(
            errors_rep,
            {"assembly": "delta_spill_bytes must be a signed integer"},
        )


if __name__ == "__main__":
    unittest.main()
