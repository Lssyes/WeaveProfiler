# -*- coding: utf-8 -*-
"""Unit tests for deterministic AMDGCN-level async pipeline instrumentation (`triton_amdgcn_async`)."""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.pipeline.instrument.triton_amdgcn_async import (
    build_triton_amd_async_names,
    instrument_triton_amdgcn_async,
)


SYNTHETIC_GFX942_AMDGCN = """\t.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
\t.text
\t.globl\tgemm_amd_wp
gemm_amd_wp:
.LBB0_0:
\ts_memrealtime s[6:7]
\ts_waitcnt lgkmcnt(0)
\tv_cndmask_b32_e64 v1, v32, v1, s[0:1]
\tbuffer_store_dwordx4 v[6:9], v1, s[20:23], 0 offen
\ts_memtime s[6:7]
\ts_waitcnt lgkmcnt(0)
\ts_or_b32 s7, s2, 16
\tv_mov_b32_e32 v2, s7
\tv_cndmask_b32_e64 v2, v32, v2, s[0:1]
\tbuffer_store_dwordx4 v[8:11], v2, s[20:23], 0 offen
\tbuffer_load_dwordx4 v[8:11], v3, s[16:19], 0 offen
\tbuffer_load_dwordx4 v[12:15], v4, s[16:19], 0 offen
\ts_cmpk_lt_i32 s11, 0x80
\ts_waitcnt vmcnt(1)
\tds_write_b128 v128, v[8:11]
\ts_waitcnt vmcnt(0)
\tds_write_b128 v128, v[12:15] offset:4096
\ts_cbranch_scc1 .LBB0_2
.LBB0_1:
\ts_barrier
\tbuffer_load_dwordx4 v[86:89], v147, s[28:31], 0 offen
\tbuffer_load_dwordx4 v[90:93], v146, s[28:31], 0 offen
\tv_mfma_f32_16x16x16_f16 v[6:9], v[182:183], v[150:151], v[6:9]
\tv_mfma_f32_16x16x16_f16 v[10:13], v[184:185], v[152:153], v[10:13]
\ts_waitcnt vmcnt(0)
\ts_barrier
\tds_write_b128 v128, v[86:89]
\ts_cmp_lg_u32 s12, 0
\ts_cbranch_scc1 .LBB0_1
.LBB0_2:
\ts_memtime s[4:5]
\ts_waitcnt lgkmcnt(0)
\ts_add_i32 s4, s2, 32
\tv_cndmask_b32_e64 v67, v6, v67, s[0:1]
\tbuffer_store_dwordx4 v[68:71], v67, s[20:23], 0 offen
\ts_add_i32 s3, s2, 48
\ts_memtime s[4:5]
\ts_waitcnt lgkmcnt(0)
\tbuffer_store_dwordx4 v[68:71], v0, s[20:23], 0 offen
\ts_add_i32 s3, s2, 64
\ts_addk_i32 s2, 0x50
\ts_memrealtime s[4:5]
\ts_waitcnt lgkmcnt(0)
\tbuffer_store_dwordx4 v[68:71], v0, s[20:23], 0 offen
\ts_memtime s[4:5]
\ts_waitcnt lgkmcnt(0)
\tbuffer_store_dwordx4 v[0:3], v4, s[20:23], 0 offen
\ts_endpgm
\t.amdhsa_kernel gemm_amd_wp
\t\t.amdhsa_next_free_vgpr 190
\t\t.amdhsa_next_free_sgpr 96
\t\t.amdhsa_accum_offset 192
\t.end_amdhsa_kernel
\t.set gemm_amd_wp.num_vgpr, 190
\t.set gemm_amd_wp.numbered_sgpr, 36
\t.amdgpu_metadata
---
amdhsa.kernels:
  - .sgpr_count:     42
    .vgpr_count:     190
...
\t.end_amdgpu_metadata
"""


class TestTritonAmdgcnAsync(unittest.TestCase):
    def test_synthetic_gfx942_async_injection(self) -> None:
        patched, stats = instrument_triton_amdgcn_async(SYNTHETIC_GFX942_AMDGCN, cap=256)
        self.assertEqual(stats["patched"], 1)
        self.assertGreaterEqual(stats["vmem_load_groups"], 2)
        self.assertGreaterEqual(stats["vmem_wait_clusters"], 2)
        self.assertGreaterEqual(stats["mfma_clusters"], 1)
        self.assertEqual(stats["rebound_epilogue_offsets"], 1)
        self.assertLessEqual(stats["new_vgpr"], 252)
        self.assertLessEqual(stats["new_sgpr"], 94)
        # Verify GPR directives updated consistently
        self.assertIn(f".amdhsa_next_free_vgpr {stats['new_vgpr']}", patched)
        self.assertIn(f".set gemm_amd_wp.numbered_sgpr, {stats['new_sgpr']}", patched)
        # Verify SCC save/restore guards around injected probes
        self.assertGreaterEqual(len(re.findall(r"s_cselect_b32\s+s\d+,\s*1,\s*0", patched)), 4)
        self.assertGreaterEqual(len(re.findall(r"s_cmp_lg_u32\s+s\d+,\s*0", patched)), 4)
        # Verify drain loops for VMEM FIFO completion
        self.assertIn(".LWP_AMD_DRAIN_1:", patched)
        self.assertIn(".LWP_AMD_EPDRAIN_", patched)

    def test_names_contract(self) -> None:
        names = build_triton_amd_async_names()
        self.assertEqual(names["async_engines"]["1"], "mem")
        self.assertIn("1", names["slotted_events"])
        self.assertEqual(names["semantics"]["2"], "wait")
        self.assertEqual(names["semantics"]["3"], "mma")


if __name__ == "__main__":
    unittest.main()
