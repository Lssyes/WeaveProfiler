#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Triton 记录器的真实边界行为测试。

NVIDIA 部分直接调用生成器和离线 gate helper，枚举 cap/slot/warp 分支；AMD 部分在当前
gfx942 设备上启动小 kernel，用 sentinel 区域证明越界记录没有落盘。没有对应 GPU 时跳过
设备测试，不把 skipped 当成硬件验收。
"""
from __future__ import annotations

import inspect
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "recorders" / "triton"
sys.path.insert(0, str(FRONTEND_DIR))

try:
    import torch
    import triton
    import triton.language as tl
    import wp_triton as NV
    import wp_triton_amd as AMD
except Exception as exc:  # pragma: no cover - 依赖缺失时明确跳过，不伪造 PASS
    raise unittest.SkipTest(f"Triton recorder tests require torch/triton: {type(exc).__name__}: {exc}")


@triton.jit
def _amd_probe_kernel(time_ptr, tag_ptr, aux_ptr, overflow_ptr,
                      CAP: tl.constexpr, THREADS: tl.constexpr, TIMER: tl.constexpr):
    slot, selected, n = AMD.fi_open(
        time_ptr, tag_ptr, aux_ptr, overflow_ptr,
        ROLE=1, GRAN=0, CAP=CAP, WPC=64, THREADS=THREADS, TIMER=TIMER,
    )
    n = AMD.fi_b(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                 EVENT=0x1234, aux=7, CAP=CAP, THREADS=THREADS, TIMER=TIMER)
    n = AMD.fi_e(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                 EVENT=0x1234, aux=7, CAP=CAP, THREADS=THREADS, TIMER=TIMER)
    n = AMD.fi_close(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                     CAP=CAP, THREADS=THREADS, TIMER=TIMER)


@triton.jit
def _amd_capacity_kernel(time_ptr, tag_ptr, aux_ptr, overflow_ptr,
                         CAP: tl.constexpr, THREADS: tl.constexpr, TIMER: tl.constexpr):
    slot, selected, n = AMD.fi_open(
        time_ptr, tag_ptr, aux_ptr, overflow_ptr,
        ROLE=1, GRAN=0, CAP=CAP, WPC=64, THREADS=THREADS, TIMER=TIMER,
    )
    n = AMD.fi_b(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                 EVENT=1, aux=1, CAP=CAP, THREADS=THREADS, TIMER=TIMER)
    n = AMD.fi_e(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                 EVENT=1, aux=1, CAP=CAP, THREADS=THREADS, TIMER=TIMER)
    n = AMD.fi_b(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                 EVENT=2, aux=2, CAP=CAP, THREADS=THREADS, TIMER=TIMER)
    n = AMD.fi_e(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                 EVENT=2, aux=2, CAP=CAP, THREADS=THREADS, TIMER=TIMER)
    n = AMD.fi_b(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                 EVENT=3, aux=3, CAP=CAP, THREADS=THREADS, TIMER=TIMER)
    n = AMD.fi_e(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                 EVENT=3, aux=3, CAP=CAP, THREADS=THREADS, TIMER=TIMER)
    n = AMD.fi_b(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                 EVENT=4, aux=4, CAP=CAP, THREADS=THREADS, TIMER=TIMER)
    n = AMD.fi_e(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                 EVENT=4, aux=4, CAP=CAP, THREADS=THREADS, TIMER=TIMER)
    n = AMD.fi_close(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                     CAP=CAP, THREADS=THREADS, TIMER=TIMER)


@triton.jit
def _amd_timer_probe_kernel(time_ptr, tag_ptr, aux_ptr, overflow_ptr, n_out,
                            CAP: tl.constexpr, THREADS: tl.constexpr):
    slot, selected, n = AMD.fi_open(
        time_ptr, tag_ptr, aux_ptr, overflow_ptr,
        ROLE=1, GRAN=0, CAP=CAP, WPC=64, THREADS=THREADS, TIMER=1,
    )
    n_slot = n_out + tl.zeros([THREADS], dtype=tl.int32)
    tl.store(n_slot + 0, n, mask=selected)
    n = AMD.fi_b(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                 EVENT=1, aux=11, CAP=CAP, THREADS=THREADS, TIMER=1)
    tl.store(n_slot + 1, n, mask=selected)
    n = AMD.fi_e(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                 EVENT=1, aux=11, CAP=CAP, THREADS=THREADS, TIMER=1)
    tl.store(n_slot + 2, n, mask=selected)
    n = AMD.fi_m(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                 EVENT=2, aux=12, CAP=CAP, THREADS=THREADS, TIMER=1)
    tl.store(n_slot + 3, n, mask=selected)
    n = AMD.fi_close(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                     CAP=CAP, THREADS=THREADS, TIMER=1)
    tl.store(n_slot + 4, n, mask=selected)


def _amd_available() -> bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return str(getattr(props, "gcnArchName", "")).startswith("gfx")


def _ptxas_path() -> Path | None:
    candidate = Path(triton.__file__).resolve().parent / "backends" / "nvidia" / "bin" / "ptxas"
    return candidate if candidate.is_file() else None


def _ptx_module(fragment: str, operands: int) -> str:
    """把 inline-asm 片段包成可交给 ptxas 的最小 PTX 模块。"""
    body = fragment
    for index in range(operands - 1, -1, -1):
        body = body.replace(f"${index}", f"op{index}")
    decls = [f".reg .b64 op{i};" if i in (2, 3) else f".reg .b32 op{i};"
             for i in range(operands)]
    return "\n".join([
        ".version 8.0",
        ".target sm_90",
        ".address_size 64",
        ".visible .entry test()",
        "{",
        *decls,
        body,
        "ret;",
        "}",
        "",
    ])


class AmdTimerContractTests(unittest.TestCase):
    def test_timer_contract_default_and_anchor_metadata(self):
        self.assertEqual(AMD._timer_contract(0), (0, "s_memrealtime"))
        self.assertEqual(AMD._timer_contract(1), (1, "s_memtime+anchors"))
        with self.assertRaisesRegex(ValueError, "timer_mode"):
            AMD._timer_contract(2)

    def test_clock_asm_waits_for_scalar_result_without_draining_vmem(self):
        source = inspect.getsource(AMD._fi_clock.fn)
        self.assertIn("s_memtime $0", source)
        self.assertIn("s_waitcnt lgkmcnt(0)", source)
        self.assertNotIn("s_waitcnt vmcnt(0)", source)

    def test_fibuffer_timer_mode_defaults_to_legacy(self):
        self.assertEqual(inspect.signature(AMD.FiBuffer).parameters["timer_mode"].default, 0)


class NvidiaRecordGateTests(unittest.TestCase):
    def test_dump_preserves_unsigned_timer_words(self):
        buffer = NV.FiBuffer(1, 1, 2, device="cpu")
        buffer.rec[0, 0] = torch.tensor([-1, 0x12345678, 1, 0], dtype=torch.int32)
        buffer.rec[0, 1] = torch.tensor([0, 0x12345679, 2, 0], dtype=torch.int32)
        with tempfile.TemporaryDirectory() as work, patch.object(torch.cuda, "synchronize"):
            path = Path(work) / "clock.bin"
            buffer.dump(path)
            data = path.read_bytes()
        self.assertEqual(struct.unpack_from("<Q", data, len(data) - 32)[0], 0x12345678FFFFFFFF)
        self.assertEqual(struct.unpack_from("<Q", data, len(data) - 16)[0], 0x1234567900000000)

    def test_capacity_gate_is_per_physical_record(self):
        # timer=1 的 OPEN/CLOSE 一次写两条；n=cap-1 时只允许第一条。
        status = NV.record_gate_status(
            n=7, slot=3, warp=2, cap=8, n_slots=16, warps=4, n_records=2,
        )
        self.assertEqual(status["stores"], (True, False))
        self.assertEqual(status["overflow"], NV.FI_OVF_LANE_FULL)

        full = NV.record_gate_status(
            n=8, slot=3, warp=2, cap=8, n_slots=16, warps=4, n_records=2,
        )
        self.assertEqual(full["stores"], (False, False))
        self.assertEqual(full["overflow"], NV.FI_OVF_LANE_FULL)

        fits = NV.record_gate_status(
            n=6, slot=3, warp=2, cap=8, n_slots=16, warps=4, n_records=2,
        )
        self.assertEqual(fits["stores"], (True, True))
        self.assertEqual(fits["overflow"], 0)

    def test_slot_and_warp_oob_block_every_store(self):
        slot_oob = NV.record_gate_status(
            n=0, slot=16, warp=0, cap=8, n_slots=16, warps=4, n_records=1,
        )
        self.assertEqual(slot_oob["stores"], (False,))
        self.assertEqual(slot_oob["overflow"], NV.FI_OVF_SLOT_OOB)

        warp_oob = NV.record_gate_status(
            n=0, slot=4, warp=4, cap=8, n_slots=16, warps=4, n_records=1,
        )
        self.assertEqual(warp_oob["stores"], (False,))
        self.assertEqual(warp_oob["overflow"], NV.FI_OVF_SLOT_OOB)

    def test_generated_ptx_uses_bounds_and_capacity_predicates(self):
        cfg = NV.FiCfg(cap=8, warps=4, n_slots=6, gran=0, timer=1, level=2, threads=128)
        event_asm = cfg.event_asm
        open_asm = cfg.open_asm(0)

        self.assertIn("setp.lt.u32 p_bounds, warp, 4;", event_asm)
        self.assertIn("setp.lt.u32 p_cap, slot, 6;", event_asm)
        self.assertIn("and.pred p_rec, p_rec, p_bounds;", event_asm)
        self.assertIn("setp.lt.u32 p_cap, n, 8;", event_asm)
        self.assertIn("and.pred p_store, p_rec, p_cap;", event_asm)
        self.assertEqual(event_asm.count("@p_store st.global.cs.v4.u32"), 1)
        self.assertNotIn("@p_rec st.global.cs.v4.u32", event_asm)
        self.assertIn("setp.ge.u32 p_ovf, warp, 4;", event_asm)
        self.assertIn("setp.ge.u32 p_ovf, slot, 6;", event_asm)
        self.assertIn("and.b32 tagv, $6,", event_asm)
        self.assertIn("0xF0000FFF", event_asm)

        # timer=1 时 OPEN 的第二条锚点记录在 n=cap-1 必须被挡住。
        self.assertIn("setp.lt.u32 p_cap, n, 8;", open_asm)
        self.assertIn("setp.lt.u32 p_cap, n, 7;", open_asm)
        self.assertEqual(open_asm.count("@p_store st.global.cs.v4.u32"), 2)
        self.assertIn("setp.ge.u32 p_ovf, n, 7;", open_asm)

    def test_write_names_unwraps_constexpr_and_non_string_keys(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "names.json"
            NV.write_names(
                path,
                events={tl.constexpr(1): "mma", 2: {"semantic_id": tl.constexpr("gemm")}},
                roles={tl.constexpr(3): "main"},
                aux={0: tl.constexpr("tile")},
                note=tl.constexpr("masked"),
            )
            doc = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(doc["events"]), {"1", "2"})
            self.assertEqual(doc["events"]["1"], "mma")
            self.assertEqual(doc["events"]["2"]["semantic_id"], "gemm")
            self.assertEqual(doc["roles"], {"3": "main"})
            self.assertEqual(doc["aux"], {"0": "tile"})
            self.assertEqual(doc["note"], "masked")
            self.assertNotIn("constexpr[", path.read_text(encoding="utf-8"))

            with self.assertRaisesRegex(ValueError, "规范化后都是"):
                NV.write_names(Path(td) / "bad.json", events={1: "a", "1": "b"})

    @unittest.skipUnless(_ptxas_path(), "bundled NVIDIA ptxas required")
    def test_generated_ptx_passes_ptxas(self):
        cfg = NV.FiCfg(cap=8, warps=4, n_slots=8, gran=0, timer=1, level=2, threads=128)
        fragments = {
            "open": (cfg.open_asm(0), 6),
            "event": (cfg.event_asm, 8),
            "event_by": (cfg.event_by_asm, 9),
            "close": (cfg.close_asm, 6),
        }
        with tempfile.TemporaryDirectory() as td:
            for name, (fragment, operands) in fragments.items():
                ptx = Path(td) / f"{name}.ptx"
                cubin = Path(td) / f"{name}.cubin"
                ptx.write_text(_ptx_module(fragment, operands), encoding="ascii")
                proc = subprocess.run(
                    [str(_ptxas_path()), "--gpu-name=sm_90", "-o", str(cubin), str(ptx)],
                    text=True, capture_output=True,
                )
                self.assertEqual(proc.returncode, 0,
                                 f"ptxas rejected {name}:\n{proc.stdout}\n{proc.stderr}")


@unittest.skipUnless(_amd_available(), "AMD gfx GPU required for recorder execution tests")
class AmdRecorderExecutionTests(unittest.TestCase):
    def setUp(self):
        self.device = 0

    def test_overflow_storage_keeps_capacity_metadata(self):
        fb = AMD.FiBuffer(n_ctas=2, cap=8, gran=0, num_warps=1, device=self.device)
        self.assertEqual(int(fb.overflow[AMD.OVF_N_SLOTS_INDEX].item()), fb.n_slots)
        fb.reset()
        self.assertEqual(int(fb.overflow[AMD.OVF_N_SLOTS_INDEX].item()), fb.n_slots)

    def test_event_id_is_masked_and_end_flag_survives(self):
        fb = AMD.FiBuffer(n_ctas=1, cap=16, gran=0, num_warps=1, device=self.device)
        fb.reset()
        _amd_probe_kernel[(1,)](
            *fb.ptr_args(), CAP=fb.cap, THREADS=fb.threads,
            TIMER=fb.timer_mode, num_warps=1,
        )
        counts, ovf = fb.read_counts()
        self.assertEqual(ovf, 0)
        tags = fb.tag.cpu().reshape(-1).tolist()
        auxs = fb.aux.cpu().reshape(-1).tolist()
        masked = {tag & 0xFFFFFFFF for tag, aux in zip(tags, auxs) if tag and aux == 7}
        self.assertEqual(masked, {0x00000234, 0x80000234})
        self.assertGreaterEqual(max(counts), 4)

    def test_cap_overflow_never_writes_past_allocated_lane(self):
        cap = 4
        fb = AMD.FiBuffer(n_ctas=1, cap=cap, gran=0, num_warps=1, device=self.device)
        fb.reset()
        n_slots = fb.n_slots
        sentinel = 32
        time = torch.zeros(n_slots * cap + sentinel, dtype=torch.int64, device=self.device)
        tag = torch.zeros(n_slots * cap + sentinel, dtype=torch.int32, device=self.device)
        aux = torch.zeros(n_slots * cap + sentinel, dtype=torch.int32, device=self.device)
        _amd_capacity_kernel[(1,)](
            time, tag, aux, fb.overflow,
            CAP=cap, THREADS=fb.threads, TIMER=fb.timer_mode, num_warps=1,
        )
        flags = int(fb.overflow[AMD.OVF_FLAGS_INDEX].item())
        tag_host = tag.cpu().tolist()
        nonzero = [i for i, value in enumerate(tag_host) if value != 0]
        self.assertTrue(nonzero, "capacity probe wrote no records")
        first = nonzero[0]
        slot = first // cap
        lane_start = slot * cap
        lane_end = lane_start + cap
        self.assertEqual(first, lane_start)
        self.assertTrue(all(lane_start <= i < lane_end for i in nonzero),
                        f"record escaped lane {slot}: {nonzero}")
        self.assertTrue(all(value == 0 for value in tag_host[lane_end:lane_end + sentinel]))
        self.assertTrue(all(value == 0 for value in time.cpu().tolist()[lane_end:lane_end + sentinel]))
        self.assertEqual(flags & AMD.OVF_LANE_FULL, AMD.OVF_LANE_FULL)

    def test_slot_oob_is_flagged_and_never_written(self):
        cap = 16
        fb = AMD.FiBuffer(n_ctas=1, cap=cap, gran=0, num_warps=1, device=self.device)
        fb.reset()
        allocated = fb.n_slots
        sentinel = 64
        total = (allocated + sentinel) * cap
        time = torch.zeros(total, dtype=torch.int64, device=self.device)
        tag = torch.zeros(total, dtype=torch.int32, device=self.device)
        aux = torch.zeros(total, dtype=torch.int32, device=self.device)
        # grid=2，但 host 只分配 1 个 CTA 的槽；第二个 CTA 的 slot 必须被挡住。
        _amd_probe_kernel[(2,)](
            time, tag, aux, fb.overflow,
            CAP=cap, THREADS=fb.threads, TIMER=fb.timer_mode, num_warps=1,
        )
        flags = int(fb.overflow[AMD.OVF_FLAGS_INDEX].item())
        tail = tag.cpu().tolist()[allocated * cap:]
        self.assertEqual(flags & AMD.OVF_SLOT_OOB, AMD.OVF_SLOT_OOB)
        self.assertTrue(all(value == 0 for value in tail),
                        "slot-OOB recorder wrote into the sentinel region")


    def test_timer_one_records_n_anchor_order_and_metadata(self):
        cap = 16
        fb = AMD.FiBuffer(
            n_ctas=1, cap=cap, gran=0, num_warps=1, timer_hz=100_000_000,
            timer_mode=1, device=self.device,
        )
        fb.reset()
        n_out = torch.zeros((5,), dtype=torch.int32, device=self.device)
        _amd_timer_probe_kernel[(1,)](
            *fb.ptr_args(), n_out, CAP=cap, THREADS=fb.threads, num_warps=1,
        )
        counts, ovf = fb.read_counts()
        self.assertEqual(ovf, 0)
        active = [slot for slot, count in enumerate(counts) if count]
        self.assertEqual(len(active), 1, counts)
        slot = active[0]
        self.assertEqual(counts[slot], 7)
        self.assertEqual(n_out.cpu().tolist(), [2, 3, 4, 5, 7])

        tags = (fb.tag.cpu()[slot].to(torch.int64) & 0xFFFFFFFF).tolist()
        auxs = (fb.aux.cpu()[slot].to(torch.int64) & 0xFFFFFFFF).tolist()
        self.assertEqual(tags[:7], [
            AMD.TAG_OPEN | 1,
            AMD.TAG_CLK,
            1,
            AMD.TAG_END | 1,
            AMD.TAG_MARK | 2,
            AMD.TAG_CLOSE,
            AMD.TAG_CLK,
        ])
        self.assertEqual(auxs[:7], [
            auxs[0],  # OPEN 的 aux 是物理 HW_ID，内容随驻留位置变化。
            0,
            11,
            11,
            12,
            0,
            1,
        ])

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "timer1.fi02.bin"
            dump = fb.dump(path)
            blob = path.read_bytes()
            self.assertEqual(blob[:8], b"FILITE02")
            n_slots, dump_cap, wpc, dump_ovf, header_timer, meta_len = struct.unpack_from(
                "<6I", blob, 8,
            )
            self.assertEqual((n_slots, dump_cap, wpc, dump_ovf, header_timer),
                             (fb.n_slots, cap, fb.wpc, 0, 1))
            meta = json.loads(blob[32:32 + meta_len].decode("utf-8"))
            self.assertEqual(meta["timer"], "s_memtime+anchors")
            self.assertEqual(dump["meta"]["timer"], "s_memtime+anchors")

    def test_timer_one_multiwave_closes_without_fault(self):
        # A one-wave probe missed the original SGPR dependency fault.  Eight
        # physical waves on one CTA exercise the multi-wave timer path that the
        # GEMM recorder uses.
        cap = 16
        warps = 8
        fb = AMD.FiBuffer(
            n_ctas=1, cap=cap, gran=0, num_warps=warps, timer_mode=1,
            device=self.device,
        )
        fb.reset()
        _amd_probe_kernel[(1,)](
            *fb.ptr_args(), CAP=cap, THREADS=fb.threads,
            TIMER=fb.timer_mode, num_warps=warps,
        )
        counts, ovf = fb.read_counts()
        active = [int(count) for count in counts if count]
        self.assertEqual(ovf, 0)
        self.assertEqual(len(active), warps, counts)
        self.assertEqual(len(set(active)), 1, active)
        self.assertGreater(active[0], 0)

    def test_timer_zero_default_header_and_meta_are_unchanged(self):
        fb = AMD.FiBuffer(
            n_ctas=1, cap=16, gran=0, num_warps=1, timer_hz=100_000_000,
            device=self.device,
        )
        self.assertEqual(fb.timer_mode, 0)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "timer0.fi02.bin"
            dump = fb.dump(path)
            blob = path.read_bytes()
            _, _, _, _, header_timer, meta_len = struct.unpack_from("<6I", blob, 8)
            meta = json.loads(blob[32:32 + meta_len].decode("utf-8"))
            self.assertEqual(header_timer, 0)
            self.assertEqual(meta["timer"], "s_memrealtime")
            self.assertEqual(dump["meta"]["timer"], "s_memrealtime")

    def test_timer_one_anchor_pair_capacity_boundary(self):
        cap = 5
        fb = AMD.FiBuffer(
            n_ctas=1, cap=cap, gran=0, num_warps=1, timer_mode=1,
            device=self.device,
        )
        fb.reset()
        _amd_probe_kernel[(1,)](
            *fb.ptr_args(), CAP=cap, THREADS=fb.threads,
            TIMER=fb.timer_mode, num_warps=1,
        )
        counts, ovf = fb.read_counts()
        active = [slot for slot, count in enumerate(counts) if count]
        self.assertEqual(len(active), 1, counts)
        slot = active[0]
        self.assertEqual(counts[slot], cap)
        tags = (fb.tag.cpu()[slot].to(torch.int64) & 0xFFFFFFFF).tolist()
        auxs = (fb.aux.cpu()[slot].to(torch.int64) & 0xFFFFFFFF).tolist()
        self.assertEqual(tags, [
            AMD.TAG_OPEN | 1,
            AMD.TAG_CLK,
            0x0234,
            AMD.TAG_END | 0x0234,
            AMD.TAG_CLOSE,
        ])
        self.assertEqual(auxs[1], 0)
        self.assertEqual(ovf & AMD.OVF_LANE_FULL, AMD.OVF_LANE_FULL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
