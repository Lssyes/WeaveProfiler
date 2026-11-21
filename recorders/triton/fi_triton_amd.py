# -*- coding: utf-8 -*-
"""FILITE02 kernel 内打点的 Triton AMD 前端（gfx942 / wave64）。

设计边界
--------
* 设备侧只用 Triton inline asm 读取 ``s_memrealtime``、``s_memtime`` 与 ``HW_ID``；不依赖私改 Triton 编译器。
* ``TIMER=0`` 保持默认兼容路径：每条记录都读 ``s_memrealtime``。``TIMER=1`` 时 OPEN/CLOSE
  各写两条相邻锚点记录：墙钟 ``s_memrealtime`` 后紧跟 ``TAG_CLK`` 的 ``s_memtime``
  （aux=0 开 / 1 关）；中间事件只读 ``s_memtime``，不执行会清空 outstanding VMEM 的
  ``s_waitcnt vmcnt(0)/lgkmcnt(0)``。解码器按 FILITE02 头的 timer=1 做锚点换算。
* wave 粒度（FI_GRAN=0）由每个 wave 的 lane 0 记录，槽号是
  ``cta * 64 + ((simd << 4) | wave)``，保证同一 CTA 内不同 SIMD 的同名 wave 不碰撞。
* SM/CU 粒度（FI_GRAN=1）在 CTA 内先选最小 SIMD_ID，再在该 SIMD 内选最小 WAVE_ID，只由这个
  物理 wave 的 lane 0 记录。Triton 不暴露 threadIdx，故用 HW_ID 选唯一物理记录者；语义上等价于
  每个 CTA 只开一条记录流。
* Triton inline asm 不能像 C++ 宏那样藏一个隐式上下文。调用方显式持有
  ``(slot, selected, n)``，每次 fi_b/fi_e/fi_m 返回递增后的 n。上卡验收（正确性、时延、rocprofv3 计数器）
  由调用方自己的驱动脚本完成；包内的 ``tests/frontend_api_check.py`` 只做离线 API 形状与元数据校验，
  流程见 ``SKILL.md`` 的「检查」「采硬件」两节。
* 设备侧写三块探测缓冲（time/tag/aux），host 在 FiBuffer.dump() 中按既有 FILITE02 的 16B 记录格式打包。
  缓冲清空、读回和 gzip 都在 GPU 锁外/计时区间外。

最小用法（Triton kernel 内）::

    slot, selected, n = fi_open(time_ptr, tag_ptr, aux_ptr, overflow_ptr,
                                ROLE=0, GRAN=FI_GRAN, CAP=FI_CAP,
                                WPC=FI_WPC, THREADS=FI_THREADS, TIMER=FI_TIMER)
    # ...每个语义区间
    n = fi_b(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
             EVENT=1, aux=tile_id, CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
    # ...被观察的工作...
    n = fi_e(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
             EVENT=1, aux=tile_id, CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)
    n = fi_close(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                 CAP=FI_CAP, THREADS=FI_THREADS, TIMER=FI_TIMER)

host 侧::

    fb = FiBuffer(n_ctas=grid, gran=0, cap=64, num_warps=8, timer_mode=1)
    grid = (1,)
    kernel[grid](..., *fb.ptr_args(), ..., FI_CAP=fb.cap, FI_GRAN=fb.gran,
                 FI_WPC=fb.wpc, FI_THREADS=fb.threads, FI_TIMER=fb.timer_mode)
    fb.reset()
    kernel[grid](...)                     # 只采这一拍
    fb.dump("run.fi02.bin", meta={...})
"""
from __future__ import annotations

import ctypes
import ctypes.util
import gzip
import json
import os
import struct
from pathlib import Path

try:  # 允许 host 工具在没有 torch 的机器上 import 并看文档
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

TAG_END = 0x80000000
TAG_OPEN = 0x40000000
TAG_MARK = 0x20000000
TAG_CLK = 0x10000000
TAG_CLOSE = 0x08000000
TAG_IDMASK = 0x0FFF
END_SIGNED = -0x80000000
OVF_LANE_FULL = 1
OVF_SLOT_OOB = 2
OVF_FLAGS_INDEX = 0
OVF_N_SLOTS_INDEX = 1

# hipDeviceAttribute_t from hip_runtime_api.h (ROCm 7.2).  This is not 10016:
# 10016 is hipDeviceAttributeFineGrainSupport, which returns 0/1.
HIP_DEVICE_ATTRIBUTE_WALL_CLOCK_RATE = 10017


def _need_torch():
    if torch is None:
        raise RuntimeError("FiBuffer 需要 torch（ROCm 版）")
    return torch


def _need_triton():
    if tl is None:
        raise RuntimeError("fi_* device helpers 需要 triton")
    return tl


def _wall_clock_hz(device: int = 0) -> int:
    """查询 hipDeviceAttributeWallClockRate；失败时只接受显式 FI_TIMER_HZ。"""
    ov = os.environ.get("FI_TIMER_HZ")
    if ov:
        hz = int(ov)
        if hz <= 0:
            raise ValueError("FI_TIMER_HZ 必须为正整数")
        return hz
    names = ["libamdhip64.so", "libamdhip64.so.7", ctypes.util.find_library("amdhip64")]
    last = None
    lib = None
    for n in names:
        if not n:
            continue
        try:
            lib = ctypes.CDLL(n)
            break
        except OSError as e:
            last = e
    if lib is None:
        raise RuntimeError(f"无法加载 libamdhip64 查询 s_memrealtime 频率: {last}")
    value = ctypes.c_int()
    fn = lib.hipDeviceGetAttribute
    fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    fn.restype = ctypes.c_int
    # hip_runtime_api.h: hipDeviceAttributeWallClockRate = 10017 (kHz)。
    # 不要写成 10016：那是 FineGrainSupport，只会返回 0/1。
    rc = fn(ctypes.byref(value), HIP_DEVICE_ATTRIBUTE_WALL_CLOCK_RATE, int(device))
    if rc != 0 or value.value < 1000:
        raise RuntimeError(f"hipDeviceGetAttribute(WallClockRate) 失败 rc={rc} value={value.value} kHz")
    return int(value.value) * 1000


if tl is not None:
    @tl.core.extern
    def _wp_local_id(dim, _semantic=None):
        return tl.core.extern_elementwise(
            "", "", [dim], {(tl.uint32,): ("__ockl_get_local_id", tl.uint64)},
            is_pure=True, _semantic=_semantic,
        )

    @triton.jit
    def wp_thread_ids(THREADS: tl.constexpr):
        return _wp_local_id(tl.full((THREADS,), 0, tl.uint32)).to(tl.int32)

    @triton.jit
    def _fi_lane(THREADS: tl.constexpr):
        """每个物理 lane 返回 0..63（v_mbcnt 是 wave64 内的前缀计数）。"""
        dummy = tl.arange(0, THREADS)
        return tl.inline_asm_elementwise(
            asm="""
            v_mbcnt_lo_u32_b32 $0, -1, 0
            v_mbcnt_hi_u32_b32 $0, -1, $0
            """,
            constraints="=v,v",
            args=[dummy],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _fi_hw_id(THREADS: tl.constexpr):
        """gfx942 HW_ID/XCC_ID 的 wave/simd/CU/SE/XCC 字段，返回 (wave,simd,cu,se,xcc)。"""
        dummy = tl.arange(0, THREADS)
        return tl.inline_asm_elementwise(
            asm="""
            s_getreg_b32 $0, hwreg(4, 0, 4)
            s_getreg_b32 $1, hwreg(4, 4, 2)
            s_getreg_b32 $2, hwreg(4, 8, 4)
            s_getreg_b32 $3, hwreg(4, 13, 2)
            s_getreg_b32 $4, hwreg(20, 0, 4)
            """,
            constraints="=s,=s,=s,=s,=s,v",
            args=[dummy],
            dtype=(tl.int32, tl.int32, tl.int32, tl.int32, tl.int32),
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _fi_time(THREADS: tl.constexpr):
        """固定频率、跨 CU 同时基的 s_memrealtime（tick，host 用 timer_hz 转 ns）。

        s_memrealtime 写一对 SGPR，属于标量内存/RTC 读取，由 lgkmcnt(0) 等待结果就绪；
        不得在前面插入 s_waitcnt vmcnt(0)，否则每次打点都会排空流水线中的全局内存异步加载。
        """
        zero = tl.zeros([THREADS], dtype=tl.int64)
        return tl.inline_asm_elementwise(
            asm="""
            s_memrealtime $0
            s_waitcnt lgkmcnt(0)
            """,
            constraints="=s",
            args=[],
            dtype=tl.int64,
            is_pure=False,
            pack=1,
        ) + zero

    @triton.jit
    def _fi_clock(THREADS: tl.constexpr):
        """读 s_memtime 周期计数；这是每条泳道的本地时钟，只用 OPEN/CLOSE 墙钟锚点换算。

        s_memtime 写入 SGPR 后必须用 lgkmcnt(0) 等标量写就绪；这里不插入 vmcnt(0)，
        避免每个中间事件都排空 outstanding global memory。
        """
        zero = tl.zeros([THREADS], dtype=tl.int64)
        return tl.inline_asm_elementwise(
            asm="""
            s_memtime $0
            s_waitcnt lgkmcnt(0)
            """,
            constraints="=s",
            args=[],
            dtype=tl.int64,
            is_pure=False,
            pack=1,
        ) + zero

    @triton.jit
    def _fi_record(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                   EVENT: tl.constexpr, aux, flag: tl.constexpr, CAP: tl.constexpr, THREADS: tl.constexpr,
                   TIMER: tl.constexpr, USE_WALL: tl.constexpr, CHECK_OVF: tl.constexpr = 0):
        idx = slot * CAP + n
        ok = selected & (n < CAP)
        if TIMER == 0 or USE_WALL == 1:
            t = _fi_time(THREADS)
        else:
            t = _fi_clock(THREADS)
        tagv_i = (EVENT & 0x0FFF) | flag
        tag = tl.full([THREADS], tagv_i, tl.int32)
        auxv = aux + tl.zeros([THREADS], dtype=tl.int32)
        tl.store(time_ptr + idx, t, mask=ok)
        tl.store(tag_ptr + idx, tag, mask=ok)
        tl.store(aux_ptr + idx, auxv, mask=ok)
        # K 循环热路径默认不发射 atomic_or；n 单调递增，统一在 fi_open/fi_close 汇总上报溢出位。
        if CHECK_OVF == 1:
            tl.atomic_or(overflow_ptr + tl.zeros([THREADS], dtype=tl.int32), 1,
                         mask=selected & (n >= CAP))
        return n + 1

    @triton.jit
    def _fi_record_16b(rec_ptr, overflow_ptr, base_idx, selected, n,
                       EVENT: tl.constexpr, aux, flag: tl.constexpr, CAP: tl.constexpr, THREADS: tl.constexpr,
                       TIMER: tl.constexpr, USE_WALL: tl.constexpr, CHECK_OVF: tl.constexpr = 0, role_value=None):
        """单次 16B AoS 向量化写出（[time_u64, tag_u32|(aux_u32<<32)] -> global_store_dwordx4）。"""
        ok = selected & (n < CAP)
        if TIMER == 0 or USE_WALL == 1:
            t = _fi_time(THREADS)
        else:
            t = _fi_clock(THREADS)
        tag_u64 = tl.full([THREADS], ((EVENT & 0x0FFF) | flag) & 0xFFFFFFFF, tl.uint64)
        if role_value is not None:
            tag_u64 = (tag_u64 & 0xffff0000) | (role_value.to(tl.uint64) & 0xffff)
        aux_u64 = (aux + tl.zeros([THREADS], dtype=tl.int32)).to(tl.uint32).to(tl.uint64)
        hi64 = (tag_u64 | (aux_u64 << 32)).to(tl.int64)
        pair = tl.join(t, hi64)
        offs = (base_idx + n)[:, None] * 2 + tl.arange(0, 2)[None, :]
        tl.store(rec_ptr + offs, pair, mask=ok[:, None])
        if CHECK_OVF == 1:
            tl.atomic_or(overflow_ptr + tl.zeros([THREADS], dtype=tl.int32), 1,
                         mask=selected & (n >= CAP))
        return n + 1

    @triton.jit
    def fi_open(time_ptr, tag_ptr, aux_ptr, overflow_ptr,
                ROLE: tl.constexpr, GRAN: tl.constexpr, CAP: tl.constexpr,
                WPC: tl.constexpr, THREADS: tl.constexpr, TIMER: tl.constexpr = 0):
        lane = _fi_lane(THREADS)
        wave, simd, cu, se, xcc = _fi_hw_id(THREADS)
        cta = tl.program_id(0)
        # 16 bit 物理编号与 weaveprofiler 的 HIP 目标输出约定一致：
        # CU[3:0] | SIMD[5:4] | SE[9:8] | XCC[15:12]。OPEN.aux 沿用既有语义存物理 SM/CU。
        hwid = cu | (simd << 4) | (se << 8) | (xcc << 12)
        if GRAN == 0:
            # WAVE_ID 只有 wave 内的 4 bit；同样编号在不同 SIMD 上会重复，必须把 SIMD 编进槽。
            wave_key = (simd << 4) | wave
            slot = cta * WPC + wave_key
            selected = lane == 0
        else:
            # 先在 CTA 内选最小 SIMD，再只在这个 SIMD 内选最小 WAVE_ID；不能只最小化 wave，
            # 否则多个 SIMD 的 WAVE_ID=0 会同时成为记录者。
            min_simd = tl.min(simd, axis=0)
            in_min_simd = simd == min_simd
            min_wave = tl.min(tl.where(in_min_simd, wave, 0x7FFFFFFF), axis=0)
            slot = cta * WPC + tl.zeros([THREADS], dtype=tl.int32)
            selected = (lane == 0) & in_min_simd & (wave == min_wave)
        # overflow[1] 由 FiBuffer 写入“已分配槽数”。OPEN 是每条泳道的第一条记录，
        # 在这里把 slot 越界永久并入 selected：后续 fi_b/fi_e/fi_m/fi_close 都不会
        # 再对越界地址做 tl.store。不能只依赖 host 的 grid/cap 参数自洽。
        n_slots = tl.load(overflow_ptr + 1)
        slot_ok = slot < n_slots
        lane_selected = selected
        selected = lane_selected & slot_ok
        tl.atomic_or(overflow_ptr + tl.zeros([THREADS], dtype=tl.int32), 2,
                     mask=lane_selected & (~slot_ok))
        n = tl.zeros([THREADS], dtype=tl.int32)
        n = _fi_record(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                       EVENT=0, aux=hwid, flag=0x40000000 | (ROLE & 0xFFFF), CAP=CAP, THREADS=THREADS,
                       TIMER=TIMER, USE_WALL=1, CHECK_OVF=1)
        if TIMER == 1:
            clock_tag: tl.constexpr = 0x10000000
            n = _fi_record(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                           EVENT=0, aux=0, flag=clock_tag, CAP=CAP, THREADS=THREADS,
                           TIMER=TIMER, USE_WALL=0, CHECK_OVF=1)
        return slot, selected, n

    @triton.jit
    def fi_b(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
             EVENT: tl.constexpr, aux, CAP: tl.constexpr, THREADS: tl.constexpr, TIMER: tl.constexpr = 0):
        event_id: tl.constexpr = EVENT & 0x0FFF
        return _fi_record(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                          EVENT=event_id, aux=aux, flag=0, CAP=CAP, THREADS=THREADS,
                          TIMER=TIMER, USE_WALL=0, CHECK_OVF=0)

    @triton.jit
    def fi_e(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
             EVENT: tl.constexpr, aux, CAP: tl.constexpr, THREADS: tl.constexpr, TIMER: tl.constexpr = 0):
        event_id: tl.constexpr = EVENT & 0x0FFF
        return _fi_record(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                          EVENT=event_id, aux=aux, flag=-0x80000000, CAP=CAP, THREADS=THREADS,
                          TIMER=TIMER, USE_WALL=0, CHECK_OVF=0)

    @triton.jit
    def fi_m(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
             EVENT: tl.constexpr, aux, CAP: tl.constexpr, THREADS: tl.constexpr, TIMER: tl.constexpr = 0):
        event_id: tl.constexpr = EVENT & 0x0FFF
        return _fi_record(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                          EVENT=event_id, aux=aux, flag=0x20000000, CAP=CAP, THREADS=THREADS,
                          TIMER=TIMER, USE_WALL=0, CHECK_OVF=0)

    @triton.jit
    def fi_close(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                 CAP: tl.constexpr, THREADS: tl.constexpr, TIMER: tl.constexpr = 0):
        n = _fi_record(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                       EVENT=0, aux=0, flag=0x08000000, CAP=CAP, THREADS=THREADS,
                       TIMER=TIMER, USE_WALL=1, CHECK_OVF=1)
        if TIMER == 1:
            clock_tag: tl.constexpr = 0x10000000
            n = _fi_record(time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n,
                           EVENT=0, aux=1, flag=clock_tag, CAP=CAP, THREADS=THREADS,
                           TIMER=TIMER, USE_WALL=0, CHECK_OVF=1)
        return n

    @triton.jit
    def fi_open_16b(rec_ptr, overflow_ptr,
                    ROLE: tl.constexpr, GRAN: tl.constexpr, CAP: tl.constexpr,
                    WPC: tl.constexpr, THREADS: tl.constexpr, TIMER: tl.constexpr = 0, role_value=None):
        """单块 16B AoS 缓冲版 OPEN：返回 (base_idx, selected, n)，把 slot*CAP 外提出循环。"""
        lane = _fi_lane(THREADS)
        wave, simd, cu, se, xcc = _fi_hw_id(THREADS)
        cta = tl.program_id(0) + tl.num_programs(0) * (tl.program_id(1) + tl.num_programs(1) * tl.program_id(2))
        hwid = cu | (simd << 4) | (se << 8) | (xcc << 12)
        if GRAN == 0:
            wave_key = (simd << 4) | wave
            slot = cta * WPC + wave_key
            selected = lane == 0
        else:
            min_simd = tl.min(simd, axis=0)
            in_min_simd = simd == min_simd
            min_wave = tl.min(tl.where(in_min_simd, wave, 0x7FFFFFFF), axis=0)
            slot = cta * WPC + tl.zeros([THREADS], dtype=tl.int32)
            selected = (lane == 0) & in_min_simd & (wave == min_wave)
        n_slots = tl.load(overflow_ptr + 1)
        slot_ok = slot < n_slots
        if role_value is not None:
            selected = selected & (role_value >= 0)
        lane_selected = selected
        selected = lane_selected & slot_ok
        tl.atomic_or(overflow_ptr + tl.zeros([THREADS], dtype=tl.int32), 2,
                     mask=lane_selected & (~slot_ok))
        base_idx = slot * CAP
        n = tl.zeros([THREADS], dtype=tl.int32)
        n = _fi_record_16b(rec_ptr, overflow_ptr, base_idx, selected, n,
                           EVENT=0, aux=hwid, flag=0x40000000 | (ROLE & 0xFFFF), CAP=CAP, THREADS=THREADS,
                           TIMER=TIMER, USE_WALL=1, CHECK_OVF=1, role_value=role_value)
        if TIMER == 1:
            clock_tag: tl.constexpr = 0x10000000
            n = _fi_record_16b(rec_ptr, overflow_ptr, base_idx, selected, n,
                               EVENT=0, aux=0, flag=clock_tag, CAP=CAP, THREADS=THREADS,
                               TIMER=TIMER, USE_WALL=0, CHECK_OVF=1)
        return base_idx, selected, n

    @triton.jit
    def fi_b_16b(rec_ptr, overflow_ptr, base_idx, selected, n,
                 EVENT: tl.constexpr, aux, CAP: tl.constexpr, THREADS: tl.constexpr, TIMER: tl.constexpr = 0):
        event_id: tl.constexpr = EVENT & 0x0FFF
        return _fi_record_16b(rec_ptr, overflow_ptr, base_idx, selected, n,
                              EVENT=event_id, aux=aux, flag=0, CAP=CAP, THREADS=THREADS,
                              TIMER=TIMER, USE_WALL=0, CHECK_OVF=0)

    @triton.jit
    def fi_e_16b(rec_ptr, overflow_ptr, base_idx, selected, n,
                 EVENT: tl.constexpr, aux, CAP: tl.constexpr, THREADS: tl.constexpr, TIMER: tl.constexpr = 0):
        event_id: tl.constexpr = EVENT & 0x0FFF
        return _fi_record_16b(rec_ptr, overflow_ptr, base_idx, selected, n,
                              EVENT=event_id, aux=aux, flag=-0x80000000, CAP=CAP, THREADS=THREADS,
                              TIMER=TIMER, USE_WALL=0, CHECK_OVF=0)

    @triton.jit
    def fi_m_16b(rec_ptr, overflow_ptr, base_idx, selected, n,
                 EVENT: tl.constexpr, aux, CAP: tl.constexpr, THREADS: tl.constexpr, TIMER: tl.constexpr = 0):
        event_id: tl.constexpr = EVENT & 0x0FFF
        return _fi_record_16b(rec_ptr, overflow_ptr, base_idx, selected, n,
                              EVENT=event_id, aux=aux, flag=0x20000000, CAP=CAP, THREADS=THREADS,
                              TIMER=TIMER, USE_WALL=0, CHECK_OVF=0)

    @triton.jit
    def fi_close_16b(rec_ptr, overflow_ptr, base_idx, selected, n,
                     CAP: tl.constexpr, THREADS: tl.constexpr, TIMER: tl.constexpr = 0):
        n = _fi_record_16b(rec_ptr, overflow_ptr, base_idx, selected, n,
                           EVENT=0, aux=0, flag=0x08000000, CAP=CAP, THREADS=THREADS,
                           TIMER=TIMER, USE_WALL=1, CHECK_OVF=1)
        if TIMER == 1:
            clock_tag: tl.constexpr = 0x10000000
            n = _fi_record_16b(rec_ptr, overflow_ptr, base_idx, selected, n,
                               EVENT=0, aux=1, flag=clock_tag, CAP=CAP, THREADS=THREADS,
                               TIMER=TIMER, USE_WALL=0, CHECK_OVF=1)
        return n


def _timer_contract(timer_mode: int) -> tuple[int, str]:
    """校验 FILITE02 timer 模式，并返回 (头字段, meta.timer)。

    timer=0 是原有的 s_memrealtime 每记录模式；timer=1 只在 OPEN/CLOSE 保存
    s_memrealtime 墙钟锚点，并紧跟 s_memtime 周期锚点，中间记录使用 s_memtime。
    """
    mode = int(timer_mode)
    if mode == 0:
        return 0, "s_memrealtime"
    if mode == 1:
        return 1, "s_memtime+anchors"
    raise ValueError("timer_mode 只能是 0(每记录 s_memrealtime) 或 1(s_memtime+锚点)")


class FiBuffer:
    """设备探测缓冲与 FILITE02 host 打包器。

    gran=0: 每个物理 wave 一个 slot，wpc 默认 64。槽内 wave 编号是 ``(simd << 4) | wave``，
            给每个 SIMD 保留 16 个 WAVE_ID 槽，避免不同 SIMD 的同名 wave 碰撞。
    gran=1: 每个 CTA 一个 slot，wpc=1，只由 CTA 内 ``(最小 SIMD_ID, 该 SIMD 内最小 WAVE_ID)``
            的 lane 0 记录。
    timer_mode=0: 每条记录读 ``s_memrealtime``，保持旧转储语义。
    timer_mode=1: OPEN/CLOSE 写 s_memrealtime+s_memtime 双锚点，中间记录写 s_memtime；
                  kernel 侧必须把同一个值作为 ``TIMER=fb.timer_mode`` 传给 fi_*。
    """

    def __init__(self, n_ctas: int, cap: int, gran: int = 0, num_warps: int = 8,
                 device: int = 0, wpc: int | None = None, timer_hz: int | None = None,
                 timer_mode: int = 0):
        t = _need_torch()
        if gran not in (0, 1):
            raise ValueError("gran 只能是 0(wave) 或 1(SM/CU)")
        if n_ctas <= 0 or cap <= 2 or num_warps <= 0:
            raise ValueError("n_ctas/cap/num_warps 参数非法")
        self.n_ctas = int(n_ctas)
        self.cap = int(cap)
        self.gran = int(gran)
        self.num_warps = int(num_warps)
        self.device = int(device)
        self.threads = self.num_warps * 64
        self.wpc = int(wpc) if wpc is not None else (64 if gran == 0 else 1)
        if gran == 0 and self.wpc < 64:
            raise ValueError("wave 粒度需要 wpc >= 64：槽号包含 SIMD(2 bit) 与 WAVE_ID(4 bit)")
        if gran == 1 and self.wpc != 1:
            raise ValueError("SM/CU 粒度固定 wpc=1（每个 CTA 一个槽）")
        self.n_slots = self.n_ctas * self.wpc
        if self.n_slots <= 0 or self.n_slots >= 2**31:
            raise ValueError(f"n_slots 必须落在 1..2**31-1，收到 {self.n_slots}")
        self.timer_hz = int(timer_hz) if timer_hz else None
        self.timer_mode, self.timer_name = _timer_contract(timer_mode)
        shape = (self.n_slots, self.cap)
        self.time = t.zeros(shape, dtype=t.int64, device=device)
        self.tag = t.zeros(shape, dtype=t.int32, device=device)
        self.aux = t.zeros(shape, dtype=t.int32, device=device)
        # [0]=overflow flags；[1]=host 分配的槽数，供设备侧在 OPEN 时做 slot 上界检查。
        # 多一个只能由 host 写的元数据字，不改变 kernel 实参或 fi_* 调用签名。
        self.overflow = t.zeros((2,), dtype=t.int32, device=device)
        self.overflow[OVF_N_SLOTS_INDEX] = self.n_slots

    def ptr_args(self):
        return self.time, self.tag, self.aux, self.overflow

    def reset(self):
        self.time.zero_()
        self.tag.zero_()
        self.aux.zero_()
        self.overflow[OVF_FLAGS_INDEX].zero_()
        # 容量元数据不是一次采集的状态，reset 不能把它清掉。
        self.overflow[OVF_N_SLOTS_INDEX] = self.n_slots
        return self

    zero = reset

    def synchronize(self):
        _need_torch().cuda.synchronize(self.device)

    def read_counts(self):
        self.synchronize()
        tags = self.tag.cpu()
        counts = (tags != 0).sum(dim=1).to(torch.int64).tolist()
        ovf = int(self.overflow.cpu()[OVF_FLAGS_INDEX].item())
        return counts, ovf

    def dump(self, path, meta=None, extra_meta=None):
        """读回并写 FILITE02；timer 头字段与 meta.timer 由 timer_mode 决定。"""
        timer_mode, timer_name = _timer_contract(self.timer_mode)
        if self.timer_hz is None:
            self.timer_hz = _wall_clock_hz(self.device)
        counts, ovf = self.read_counts()
        times = self.time.cpu().numpy()
        tags = self.tag.cpu().numpy()
        auxs = self.aux.cpu().numpy()
        md = dict(meta or {})
        md.setdefault("timer_hz", int(self.timer_hz))
        md.setdefault("gran", int(self.gran))
        md.setdefault("arch", "gfx942")
        md.setdefault("language", "triton")
        md.setdefault("frontend", "weaveprofiler/triton-amd")
        md.setdefault("timer", timer_name)
        md.setdefault("wpc", int(self.wpc))
        md.setdefault("slot_encoding", "cta*wpc+((simd<<4)|wave)" if self.gran == 0 else "cta")
        md.setdefault("hwid_encoding", "xcc[15:12]|se[9:8]|simd[5:4]|cu[3:0]")
        md.setdefault("n_ctas", int(self.n_ctas))
        md.setdefault("n_slots", int(self.n_slots))
        md.setdefault("overflow_layout", "word0=flags,word1=host_n_slots")
        if extra_meta:
            md.update(extra_meta)
        # timer 表示实际记录格式，不能被附属元数据静默覆盖。
        md["timer"] = timer_name
        mb = json.dumps(md, ensure_ascii=False, sort_keys=True).encode("utf-8")
        path = str(path)
        with open(path, "wb") as f:
            f.write(b"FILITE02")
            f.write(struct.pack("<6I", self.n_slots, self.cap, self.wpc, ovf, timer_mode, len(mb)))
            f.write(mb)
            f.write(struct.pack(f"<{self.n_slots}I", *counts))
            for s, c in enumerate(counts):
                for i in range(min(int(c), self.cap)):
                    time_bits = int(times[s, i]) & 0xFFFFFFFFFFFFFFFF
                    f.write(struct.pack("<QII", time_bits, int(tags[s, i]) & 0xFFFFFFFF, int(auxs[s, i]) & 0xFFFFFFFF))
        return dict(path=path, bytes=os.path.getsize(path), overflow=ovf, records=int(sum(counts)), counts=counts, meta=md)

    def to_perfetto(self, path, names, out, summary=None, allow_errors=False):
        """调用同目录的 fi_lite_decode.py（兼容旧版 FILITE02 转储）。"""
        import subprocess
        import sys
        dump = self.dump(path, meta=(summary or {}).get("meta"))
        dec = Path(__file__).resolve().parents[2] / "weaveprof" / "pipeline" / "validate" / "tools" / "fi_lite_decode.py"
        cmd = [sys.executable, "-I", str(dec), dump["path"], "--names", str(names), "-o", str(out)]
        if summary:
            cmd += ["--summary", str(summary)]
        if allow_errors:
            cmd.append("--allow-errors")
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(p.stdout, end="")
        return dump, p.returncode, p.stdout


class FiPackedBuffer:
    """16B AoS 单块连续缓冲（[n_slots, cap, 2] int64）：设备侧单条 128-bit 写，host 侧零拆包直出 FILITE02。"""

    def __init__(self, n_ctas: int, cap: int, gran: int = 0, num_warps: int = 8,
                 device: int = 0, wpc: int | None = None, timer_hz: int | None = None,
                 timer_mode: int = 0):
        t = _need_torch()
        if gran not in (0, 1):
            raise ValueError("gran 只能是 0(wave) 或 1(SM/CU)")
        if n_ctas <= 0 or cap <= 2 or num_warps <= 0:
            raise ValueError("n_ctas/cap/num_warps 参数非法")
        self.n_ctas = int(n_ctas)
        self.cap = int(cap)
        self.gran = int(gran)
        self.num_warps = int(num_warps)
        self.device = int(device)
        self.threads = self.num_warps * 64
        self.wpc = int(wpc) if wpc is not None else (64 if gran == 0 else 1)
        if gran == 0 and self.wpc < 64:
            raise ValueError("wave 粒度需要 wpc >= 64")
        if gran == 1 and self.wpc != 1:
            raise ValueError("SM/CU 粒度固定 wpc=1")
        self.n_slots = self.n_ctas * self.wpc
        if self.n_slots <= 0 or self.n_slots >= 2**31:
            raise ValueError(f"n_slots 必须落在 1..2**31-1，收到 {self.n_slots}")
        self.timer_hz = int(timer_hz) if timer_hz else None
        self.timer_mode, self.timer_name = _timer_contract(timer_mode)
        self.rec = t.zeros((self.n_slots, self.cap, 2), dtype=t.int64, device=device)
        self.overflow = t.zeros((2,), dtype=t.int32, device=device)
        self.overflow[OVF_N_SLOTS_INDEX] = self.n_slots

    @property
    def records(self):
        return self.rec

    def ptr_args(self):
        return self.rec, self.overflow

    def reset(self):
        self.rec.zero_()
        self.overflow[OVF_FLAGS_INDEX].zero_()
        self.overflow[OVF_N_SLOTS_INDEX] = self.n_slots
        return self

    zero = reset

    def synchronize(self):
        _need_torch().cuda.synchronize(self.device)

    def read_counts(self):
        self.synchronize()
        hi = self.rec[:, :, 1].cpu()
        tags = hi & 0xFFFFFFFF
        counts = (tags != 0).sum(dim=1).to(torch.int64).tolist()
        ovf = int(self.overflow.cpu()[OVF_FLAGS_INDEX].item())
        return counts, ovf

    def dump(self, path, meta=None, extra_meta=None):
        timer_mode, timer_name = _timer_contract(self.timer_mode)
        if self.timer_hz is None:
            self.timer_hz = _wall_clock_hz(self.device)
        counts, ovf = self.read_counts()
        rec_np = self.rec.cpu().numpy()
        md = dict(meta or {})
        md.setdefault("timer_hz", int(self.timer_hz))
        md.setdefault("gran", int(self.gran))
        md.setdefault("arch", "gfx942")
        md.setdefault("language", "triton")
        md.setdefault("frontend", "weaveprofiler/triton-amd-16b")
        md.setdefault("timer", timer_name)
        md.setdefault("wpc", int(self.wpc))
        md.setdefault("slot_encoding", "cta*wpc+((simd<<4)|wave)" if self.gran == 0 else "cta")
        md.setdefault("hwid_encoding", "xcc[15:12]|se[9:8]|simd[5:4]|cu[3:0]")
        md.setdefault("n_ctas", int(self.n_ctas))
        md.setdefault("n_slots", int(self.n_slots))
        md.setdefault("overflow_layout", "word0=flags,word1=host_n_slots")
        if extra_meta:
            md.update(extra_meta)
        md["timer"] = timer_name
        mb = json.dumps(md, ensure_ascii=False, sort_keys=True).encode("utf-8")
        path = str(path)
        with open(path, "wb") as f:
            f.write(b"FILITE02")
            f.write(struct.pack("<6I", self.n_slots, self.cap, self.wpc, ovf, timer_mode, len(mb)))
            f.write(mb)
            f.write(struct.pack(f"<{self.n_slots}I", *counts))
            for s, c in enumerate(counts):
                cnt = min(int(c), self.cap)
                if cnt > 0:
                    f.write(rec_np[s, :cnt, :].astype("<i8", copy=False).tobytes())
        return dict(path=path, bytes=os.path.getsize(path), overflow=ovf, records=int(sum(counts)), counts=counts, meta=md)


_FI_ACTIVE_BUF = None


def fi_attach(buf=None):
    """Attach (or detach when ``None``) a module-level ``FiPackedBuffer``/``FiBuffer``."""
    global _FI_ACTIVE_BUF
    _FI_ACTIVE_BUF = buf
    return buf


def fi_get_attached():
    return _FI_ACTIVE_BUF


class _FiTritonProxy:
    """Same-name transparent AMD Triton launcher proxy (§3.2 2a — 0 Call-Chain Signature Edits)."""

    def __init__(self, l0_fn, inst_fn, buf_getter=fi_get_attached):
        self._l0_fn = l0_fn
        self._inst_fn = inst_fn
        self._buf_getter = buf_getter
        self.__name__ = getattr(l0_fn, "__name__", "triton_amd_kernel")

    def _autotune_kwargs(self, existing_kwargs: dict) -> dict:
        if not hasattr(self._l0_fn, "configs") or hasattr(self._inst_fn, "configs"):
            return existing_kwargs
        best = getattr(self._l0_fn, "best_config", None)
        if best is None:
            cache = getattr(self._l0_fn, "cache", None) or {}
            if cache:
                best = next(reversed(cache.values()))
            elif getattr(self._l0_fn, "configs", None):
                best = self._l0_fn.configs[0]
        if best is None:
            return existing_kwargs
        merged = dict(best.all_kwargs()) if hasattr(best, "all_kwargs") else dict(getattr(best, "kwargs", {}))
        merged.update(existing_kwargs)
        return merged

    def __getitem__(self, grid):
        buf = self._buf_getter()
        if buf is None:
            return self._l0_fn[grid]
        inst_launcher = self._inst_fn[grid]

        def _wrapped(*args, **kwargs):
            call_kw = self._autotune_kwargs(kwargs)
            return inst_launcher(
                *args,
                _fi_rec_ptr=buf.rec,
                _fi_ovf_ptr=buf.overflow,
                _FI_LEVEL=2,
                _FI_GRAN=buf.gran,
                _FI_CAP=buf.cap,
                _FI_WPC=buf.wpc,
                _FI_THREADS=buf.threads,
                _FI_TIMER=buf.timer_mode,
                **call_kw,
            )

        return _wrapped

    def __getattr__(self, name: str):
        buf = self._buf_getter()
        target = self._l0_fn if buf is None else self._inst_fn
        return getattr(target, name)


# Canonical WeaveProfiler (`wp_*` / `Wp*`) aliases (§3.2)
WpPackedBuffer = FiPackedBuffer
if "fi_open" in globals():
    wp_open = fi_open
if "fi_open_16b" in globals():
    wp_open_16b = fi_open_16b
if "fi_b_16b" in globals():
    wp_b_16b = fi_b_16b
if "fi_e_16b" in globals():
    wp_e_16b = fi_e_16b
if "fi_m_16b" in globals():
    wp_m_16b = fi_m_16b
if "fi_close_16b" in globals():
    wp_close_16b = fi_close_16b
wp_attach = fi_attach
wp_get_attached = fi_get_attached
_WpTritonProxy = _FiTritonProxy
wp_wrap = _WpTritonProxy
if "fi_b_16b" in globals():
    WP_B = fi_b_16b
    WP_E = fi_e_16b


