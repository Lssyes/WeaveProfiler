# -*- coding: utf-8 -*-
"""WeaveProfiler (`wp`) canonical Triton recorder frontend for NVIDIA (PTX backend).

Self-contained NVIDIA Triton recorder emitting FILITE02/WPLITE02 dumps decoded
by ``weaveprof/pipeline/validate/tools/wp_decode.py``.

Environment switches:
  WP_LEVEL -> FI_LEVEL   0 = empty, 1 = OPEN/CLOSE only, 2 = all events
  WP_GRAN  -> FI_GRAN    0 = warp lanes, 1 = one lane per CTA
  WP_TIMER -> FI_TIMER   0 = %globaltimer, 1 = %clock64 + OPEN/CLOSE anchors

Host helpers (``write_names``, ``cap_for``, ``WpCfg``/``FiCfg``, ``wp_cfg``,
``WP_LEVEL``/``WP_GRAN``/``WP_TIMER``) import cleanly without Triton; device
JIT helpers raise a clear ImportError when Triton is missing.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import dataclasses
import json
import os
import struct
from pathlib import Path
from typing import Any, Optional, Sequence


def _map_env(canonical: str, legacy: str) -> None:
    value = os.environ.get(canonical)
    if value is None or value == "":
        return
    prior = os.environ.get(legacy)
    if prior not in (None, "") and prior != value:
        raise RuntimeError(f"{canonical}={value!r} conflicts with {legacy}={prior!r}; set only one spelling")
    os.environ[legacy] = value


for _canonical, _legacy in (("WP_LEVEL", "FI_LEVEL"), ("WP_GRAN", "FI_GRAN"), ("WP_TIMER", "FI_TIMER")):
    _map_env(_canonical, _legacy)

WP_LEVEL = int(os.environ.get("FI_LEVEL", "2"))
WP_GRAN = int(os.environ.get("FI_GRAN", "0"))
WP_TIMER = int(os.environ.get("FI_TIMER", "0"))
for _label, _value, _allowed in (
    ("WP_LEVEL", WP_LEVEL, (0, 1, 2)),
    ("WP_GRAN", WP_GRAN, (0, 1)),
    ("WP_TIMER", WP_TIMER, (0, 1)),
):
    if _value not in _allowed:
        raise ValueError(f"{_label} must be one of {_allowed}, got {_value}")

WP_TAG_END = 0x80000000
WP_TAG_OPEN = 0x40000000
WP_TAG_MARK = 0x20000000
WP_TAG_CLK = 0x10000000
WP_TAG_CLOSE = 0x08000000
WP_TAG_IDMASK = 0x0FFF

try:
    import torch
except ImportError as _torch_exc:
    torch = None
    _TORCH_ERROR: BaseException | None = _torch_exc
else:
    _TORCH_ERROR = None

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
    _IMPORT_ERROR: BaseException | None = None
except ImportError as _exc:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False
    _IMPORT_ERROR = _exc

_DEVICE_SYMBOLS = {
    "wp_thread_ids",
    "fi_open", "fi_open_by", "fi_close", "fi_b", "fi_e", "fi_m", "fi_trans",
    "fi_bg", "fi_eg", "fi_mg", "fi_b_by", "fi_e_by", "fi_m_by", "fi_close_by",
    "wp_open", "wp_open_by", "wp_close", "wp_b", "wp_e", "wp_m", "wp_trans",
    "wp_bg", "wp_eg", "wp_mg", "wp_b_by", "wp_e_by", "wp_m_by", "wp_close_by",
    "WP_OPEN", "WP_OPEN_BY", "WP_CLOSE", "WP_B", "WP_E", "WP_M", "WP_TRANS",
}

# ---- tag 位定义：与 recorders/cuda_cpp/wp_lite.cuh 与 weaveprof/pipeline/validate/tools/wp_decode.py 完全一致 ----
FI_TAG_END = 0x80000000     # 区间结束
FI_TAG_OPEN = 0x40000000    # FI_OPEN：低 16 位=角色号，aux=物理 SM/CU 号
FI_TAG_MARK = 0x20000000    # 瞬时事件
FI_TAG_CLK = 0x10000000     # FI_TIMER=1 时紧跟 OPEN/CLOSE 的本地周期锚点，aux=0 开 / 1 关
FI_TAG_CLOSE = 0x08000000   # FI_CLOSE
FI_TAG_IDMASK = 0x0FFF      # 事件号 1..4095（0 保留给“空记录”）
FI_TAG_ROLEMASK = 0xFFFF
FI_TAG_WIREMASK = 0xF0000FFF  # 线上 tag 只允许 4 个高位 flag + 12 位事件号
FI_OVF_LANE_FULL = 1        # overflow bit0：某泳道写满（cap 给小了）
FI_OVF_SLOT_OOB = 2         # overflow bit1：slot 越界（warps 或 CTA 数给小了）

# Triton 3.x 禁止在 @jit 体内读“裸 int 全局变量”（NameError: Cannot access global variable...），
# jit 体内用到的 tag 位必须先实例化成 tl.constexpr（host 侧仍用上面的裸 int，两处值同源）。
FI_TAG_END_C = tl.constexpr(FI_TAG_END) if tl is not None else FI_TAG_END
FI_TAG_MARK_C = tl.constexpr(FI_TAG_MARK) if tl is not None else FI_TAG_MARK
FI_TAG_IDMASK_C = tl.constexpr(FI_TAG_IDMASK) if tl is not None else FI_TAG_IDMASK

FI_VERSION = "fast-instrument-triton"
FI_MAGIC = b"FILITE02"


def _uw(v):
    """从 tl.constexpr 里取出 Python 值。"""
    return v.value if (tl is not None and isinstance(v, tl.constexpr)) else v


@dataclasses.dataclass(frozen=True)
class RecordGatePlan:
    """一次 fi_* 调用的容量门槛；PTX 生成和离线行为测试共用这组定义。"""
    record_cap_limits: tuple
    lane_full_at: int

    def record_fits(self, n: int, index: int) -> bool:
        return index < len(self.record_cap_limits) and int(n) < self.record_cap_limits[index]

    def lane_overflows(self, n: int) -> bool:
        return int(n) >= self.lane_full_at


def _record_gate_plan(cap: int, n_records: int) -> RecordGatePlan:
    if int(cap) <= 0 or int(n_records) <= 0:
        raise ValueError(f"cap/n_records 必须为正整数，收到 {cap}/{n_records}")
    cap = int(cap)
    n_records = int(n_records)
    limits = tuple(max(cap - i, 0) for i in range(n_records))
    return RecordGatePlan(record_cap_limits=limits, lane_full_at=max(cap - n_records + 1, 0))


def record_gate_status(n: int, slot: int, warp: int, cap: int, n_slots: int, warps: int,
                       n_records: int) -> dict:
    """离线复算 _emit_asm 的边界规则，供测试枚举越界/容量分支。

    ``stores[i]`` 对应这次 fi_* 要写的第 i 条记录；``overflow`` 使用与 FILITE02
    相同的 bit0=泳道满、bit1=slot/warp 越界。
    """
    plan = _record_gate_plan(cap, n_records)
    warp_ok = 0 <= int(warp) < int(warps)
    slot_ok = 0 <= int(slot) < int(n_slots)
    bounds_ok = warp_ok and slot_ok
    return {
        "warp_ok": warp_ok,
        "slot_ok": slot_ok,
        "stores": tuple(bounds_ok and plan.record_fits(n, i) for i in range(n_records)),
        "overflow": ((FI_OVF_LANE_FULL if bounds_ok and plan.lane_overflows(n) else 0)
                     | (FI_OVF_SLOT_OOB if not bounds_ok else 0)),
    }


# --------------------------------------------------------------------------
# 后端：只有这一层与 ISA 相关
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class FiAsmBackend:
    """一个后端 = 一组 ISA 片段。片段里可用的占位符：{W}=warps_per_cta 立即数。
    约定寄存器名：t<i>=计时器(.b64)  addr/soff/off=地址(.b64)  slot/pid/n=编号(.b32)
                 lane/warp/sm/tid=id(.b32)  tlo<i>/thi<i>=计时器拆开的两半  p_rec=记录谓词。
    """
    name: str
    isa: str                  # 'ptx' | 'amdgcn'
    validated: bool           # 是否在本项目里真机验证过
    timer_global: str         # 读“全卡墙钟”到 {T}：跨 SM/CU 同一时基，可直接对 nsys / rocprofv3
    timer_local: str          # 读“本地周期计数器”到 {T}（FI_TIMER=1 的事件记录用）
    ids: str                  # 读 lane/warp/sm（/tid）
    slot: str                 # gran=0：slot = pid*{W} + warp
    slot_sm: str              # gran=1：slot = pid
    rec_pred: str             # gran=0 记录谓词：每个 warp 的首 lane
    rec_pred_sm: str          # gran=1 记录谓词：每个 CTA 的 0 号线程
    store16: str              # 16B 流式写，占位 {ADDR} {TLO} {THI} {TAG} {AUX}
    atom_or: str              # 溢出位置位，占位 {PTR} {BIT}
    reg64: str = ".reg .b64 {NAMES};"
    reg32: str = ".reg .b32 {NAMES};"
    pred_reg: str = ".reg .pred {NAMES};"
    note: str = ""


# NVIDIA（PTX）。%globaltimer=ns 级全卡墙钟；%clock64=SM 本地周期；%smid=物理 SM 号。
# 注意：CTA 内 warp 序号必须用 tid.x >> 5（shr.u32 warp, tid, 5），不能直接读物理 SM 槽号 %warpid（0..63），
# 否则同 SM 驻留第 2 个及以上 CTA 时 %warpid >= warps 会触发 p_bounds 假越界并静默丢弃 CTA。
FI_BACKEND_NVIDIA = FiAsmBackend(
    name="nvidia_ptx", isa="ptx", validated=True,
    timer_global="mov.u64 {T}, %globaltimer;",
    timer_local="mov.u64 {T}, %clock64;",
    ids="mov.u32 lane, %laneid;\n    mov.u32 tid, %tid.x;\n    shr.u32 warp, tid, 5;\n    mov.u32 sm, %smid;",
    slot="mul.lo.u32 slot, pid, {W};\n    add.u32 slot, slot, warp;",
    slot_sm="mov.u32 slot, pid;",
    rec_pred="setp.eq.u32 p_rec, lane, 0;",
    rec_pred_sm="setp.eq.u32 p_rec, tid, 0;",
    store16="st.global.cs.v4.u32 [{ADDR}], {{{TLO}, {THI}, {TAG}, {AUX}}};",
    atom_or="atom.global.or.b32 ovfold, [{PTR}], {BIT};",
    reg64=".reg .b64 {NAMES};", reg32=".reg .b32 {NAMES};", pred_reg=".reg .pred {NAMES};",
    note="H20(sm_90) 上 %globaltimer 实测最小步长见解码 summary 的 timer_min_step_ns",
)

# AMD(gfx942) 不在这个文件里：用同目录 wp_triton_amd.py（MI308X 实测通过）。
# 这里不放未实测的 AMDGCN 设计稿，免得被当成可用后端；FiAsmBackend 的字段划分已经给出接入位置，
# 真要在本文件里加 AMDGCN 后端，必须先在 gfx942 上跑通并把 validated 置真。
FI_BACKENDS = {b.name: b for b in (FI_BACKEND_NVIDIA,)}



# --------------------------------------------------------------------------
# 配置：等价 CUDA 的 -DFI_LEVEL / -DFI_TIMER / -DFI_GRAN / -DFI_GROUPS
# --------------------------------------------------------------------------
# 带 a 后缀的架构才是 Triton 在 Hopper/Blackwell 上实际编译的目标（sm_90a/sm_100a/sm_103a/sm_120a）。
_ARCH_A_SUFFIX = {(9, 0), (10, 0), (10, 1), (10, 3), (12, 0)}


def resolve_arch(backend: str, explicit: Optional[str] = None) -> tuple:
    """返回 (arch 标签, 依据文本)。

    此前 meta() 把 NVIDIA 架构写死成 sm_90a，B300（sm_103a）上产出的转储会被打上错误标签，
    事后审计没法信 meta（fi_bugs_b3-sk-ds B03 / fi_bugs_b3-sk-qwen B3）。现在按当前设备现算，
    并允许用 FiCfg(arch=...) 或 dump(meta={"arch": ...}) 覆盖。
    """
    if explicit:
        return str(explicit), "FiCfg.arch 显式给定"
    if backend != "nvidia_ptx":
        return str(backend), "backend 名（非 NVIDIA PTX 后端直接用后端标识）"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() == False")
        cap = torch.cuda.get_device_capability()
        maj, mnr = int(cap[0]), int(cap[1])
    except Exception as e:                                 # 拿不到就如实说明，不猜一个架构
        return "sm_unknown", (f"本机拿不到 torch.cuda 设备能力（{type(e).__name__}: {e}）："
                              "请用 FiCfg(arch='sm_90a') 或 dump(meta={'arch': ...}) 显式指定")
    name = f"sm_{maj}{mnr}" + ("a" if (maj, mnr) in _ARCH_A_SUFFIX else "")
    return name, f"torch.cuda.get_device_capability()={cap}"


@dataclasses.dataclass(frozen=True)
class FiCfg:
    """以 tl.constexpr 传进 kernel（例：FI: tl.constexpr），编译期就把开关烘进 PTX。"""
    cap: int                       # 每泳道最多记录条数（含 OPEN/CLOSE 与锚点）
    warps: int                     # 每 CTA warp 数 = num_warps；gran=1 时填 1（转储头也用 1）
    n_slots: int                   # 泳道总数 = n_ctas * warps（越界检查用）
    gran: int = 0                  # 0=warp 粒度（每 warp 首 lane 记）；1=SM/CU 粒度（每 CTA 0 号线程记）
    timer: int = 0                 # 0=全卡墙钟；1=本地周期 + OPEN/CLOSE 两个锚点
    level: int = 2                 # 0=全空；1=只 OPEN/CLOSE；2=完整事件
    groups: int = 0xFFFFFFFF       # 位掩码：只编进被置位的 FI_BG/FI_EG/FI_MG 组
    threads: int = 256             # 形状载体长度，必须 = 真实线程数 = num_warps*32，且是 2 的幂
    backend: str = "nvidia_ptx"
    arch: Optional[str] = None     # 转储 meta 的架构标签；None 时按当前设备现算
                                   # （NVIDIA 用 torch.cuda.get_device_capability，其它后端用 backend 名）
    fence: int = 0                 # 1=每段打点（Triton 3.6 的                 # 1=每段打点 asm 首尾各加一条 membar.gl（Triton 3.6 的
                                   # inline_asm_elementwise 不支持 ~memory clobber，constraints 里
                                   # 写 ~memory 会让 MLIR 直接 abort）：禁止内存访问跨打点重排，扰动更大

    def __post_init__(self):
        if self.gran not in (0, 1):
            raise ValueError(f"FI_GRAN 只能是 0(warp) 或 1(SM/CU)，收到 {self.gran}")
        if self.timer not in (0, 1):
            raise ValueError(f"FI_TIMER 只能是 0(墙钟) 或 1(本地周期)，收到 {self.timer}")
        if self.level not in (0, 1, 2):
            raise ValueError(f"FI_LEVEL 只能是 0/1/2，收到 {self.level}")
        if self.backend not in FI_BACKENDS:
            raise ValueError(f"backend 只能是 {sorted(FI_BACKENDS)}，收到 {self.backend}")
        if self.warps <= 0:
            raise ValueError(f"warps 必须为正整数，收到 {self.warps}")
        if self.n_slots <= 0 or self.n_slots >= 2**32:
            raise ValueError(f"n_slots 必须落在 1..2**32-1，收到 {self.n_slots}")
        if self.threads <= 0 or (self.threads & (self.threads - 1)):
            raise ValueError(f"threads 必须是 2 的幂（= num_warps*32），收到 {self.threads}")
        if self.gran == 0 and self.threads != self.warps * 32:
            raise ValueError(f"threads({self.threads}) 必须等于 warps*32({self.warps * 32})，"
                             "否则每线程不是恰好执行一次 asm，记录会重复或丢失")
        if self.cap <= 0 or self.cap * 16 >= 2**32:
            raise ValueError(f"cap 非法：{self.cap}（cap*16 必须放得进 32 位立即数）")

    @property
    def backend_obj(self) -> "FiAsmBackend":
        return FI_BACKENDS[self.backend]

    @property
    def warps_per_cta_in_dump(self) -> int:
        """写进 FILITE02 头的 warps_per_cta：gran=1 时每 CTA 只有 1 条泳道。"""
        return 1 if self.gran == 1 else self.warps

    def records_per_open(self) -> int:
        return 2 if self.timer == 1 else 1

    # ---- 设备侧用的 asm 字符串：host 侧构造 cfg 时就生成好，jit 体内只做属性/方法访问 ----
    # （Triton 的依赖哈希器不允许在 @triton.jit 体内调用普通 Python 函数，所以在这里算。）
    @property
    def _asm(self) -> dict:
        c = object.__getattribute__(self, "__dict__").get("_asm_cache")
        if c is None:
            c = {}
            object.__setattr__(self, "_asm_cache", c)
        return c

    def _get(self, key, kind, tag_imm=None, user_pred=False):
        c = self._asm
        if key not in c:
            c[key] = _emit_asm(self, kind, tag_imm, user_pred)
        return c[key]

    @property
    def event_asm(self) -> str:
        return self._get(("event",), "event")[0]

    @property
    def event_cons(self) -> str:
        return self._get(("event",), "event")[1]

    @property
    def event_nr(self) -> int:
        return self._get(("event",), "event")[2]

    @property
    def trans_asm(self) -> str:
        return self._get(("trans",), "trans")[0]

    @property
    def trans_cons(self) -> str:
        return self._get(("trans",), "trans")[1]

    @property
    def trans_nr(self) -> int:
        return self._get(("trans",), "trans")[2]

    def open_asm(self, role: int) -> str:
        return self._get(("open", role), "open", FI_TAG_OPEN | (role & FI_TAG_ROLEMASK))[0]

    def open_cons(self, role: int) -> str:
        return self._get(("open", role), "open", FI_TAG_OPEN | (role & FI_TAG_ROLEMASK))[1]

    def openby_asm(self, role: int) -> str:
        return self._get(("openby", role), "open", FI_TAG_OPEN | (role & FI_TAG_ROLEMASK), True)[0]

    def openby_cons(self, role: int) -> str:
        return self._get(("openby", role), "open", FI_TAG_OPEN | (role & FI_TAG_ROLEMASK), True)[1]

    @property
    def open_nr(self) -> int:
        return self._get(("opennr",), "open", FI_TAG_OPEN)[2]

    @property
    def event_by_asm(self) -> str:
        """带用户谓词的事件 asm（fi_b_by/fi_e_by/fi_m_by 用）。"""
        return self._get(("eventby",), "event", user_pred=True)[0]

    @property
    def event_by_cons(self) -> str:
        return self._get(("eventby",), "event", user_pred=True)[1]

    @property
    def event_by_nr(self) -> int:
        return self._get(("eventby",), "event", user_pred=True)[2]

    @property
    def close_asm(self) -> str:
        return self._get(("close",), "close", FI_TAG_CLOSE)[0]

    @property
    def close_cons(self) -> str:
        return self._get(("close",), "close", FI_TAG_CLOSE)[1]

    @property
    def close_nr(self) -> int:
        return self._get(("close",), "close", FI_TAG_CLOSE)[2]

    @property
    def close_by_asm(self) -> str:
        """带用户谓词的 CLOSE asm（fi_close_by 用）。"""
        return self._get(("closeby",), "close", FI_TAG_CLOSE, True)[0]

    @property
    def close_by_cons(self) -> str:
        return self._get(("closeby",), "close", FI_TAG_CLOSE, True)[1]

    @property
    def close_by_nr(self) -> int:
        return self._get(("closeby",), "close", FI_TAG_CLOSE, True)[2]

    def meta(self, **extra) -> dict:
        """转储 meta JSON：解码器原样透传，只允许加字段（timer_hz/gran/arch 等）。"""
        arch_val, arch_src = resolve_arch(self.backend, self.arch)
        m = dict(fi_version=FI_VERSION, language="triton", triton=(triton.__version__ if triton is not None else "unavailable"),
                 gran=self.gran, timer=self.timer, level=self.level, groups=self.groups,
                 arch=arch_val, arch_source=arch_src,
                 backend=self.backend, backend_isa=self.backend_obj.isa,
                 backend_validated=self.backend_obj.validated, fence=self.fence,
                 cap_per_lane=self.cap, n_slots=self.n_slots,
                 warps_per_cta=self.warps_per_cta_in_dump,
                 timer_hz=(1_000_000_000 if self.timer == 0 else None))
        m.update({k: v for k, v in extra.items() if v is not None})
        return m


# --------------------------------------------------------------------------
# PTX 生成（纯 Python，只在 trace 期执行一次；结果按 cfg 缓存）
# --------------------------------------------------------------------------
def _emit_asm(cfg, kind: str, tag_imm: Optional[int] = None, user_pred: bool = False):
    """生成 (asm, constraints, n_records)。

    kind: 'event'（tag/aux 走寄存器 $6/$7）| 'open'（tag=OPEN|role 立即数，aux=物理 SM 号）
          | 'close'（tag=CLOSE 立即数，aux=0）
    寄存器约定：$0=输出（回填形状载体）  $1=形状载体  $2=rec 基址  $3=ovf 基址
               $4=pid  $5=n  $6=tag（仅 event）  $7=aux（仅 event）
               用户谓词：open_by/close_by 是 $6；event 的 _by 变体因为 $6/$7 已被 tag/aux 占用，谓词排到 $8
    n_records：这次调用写几条记录（FI_TIMER=1 的 OPEN/CLOSE 各多一条锚点）。调用方返回 n+n_records。
    """
    cfg = _uw(cfg)
    bk = cfg.backend_obj
    kind = _uw(kind)
    tag_imm = _uw(tag_imm)
    if bk.isa != "ptx" or not bk.validated:
        # 只交出实测过的后端；AMD 走同目录 wp_triton_amd.py。
        raise NotImplementedError(
            f"backend={bk.name}（isa={bk.isa}, validated={bk.validated}）不可用："
            "本文件只有实测过的 NVIDIA PTX 后端；gfx942/wave64 的 Triton 打点用 recorders/triton/wp_triton_amd.py")

    # 这次调用要写的记录：(计时器, tag 文本, aux 文本, 相对地址偏移)
    if kind == "event":
        # 线上 tag 只保留 4 个高位 flag 与低 12 位事件号。API 边界也会先 mask；
        # 这里再 mask 一次，防止调用方把普通 int/异常常量直接塞进 tag。
        tag_txt, aux_txt = "tagv", "$7"
        recs = [("local" if cfg.timer == 1 else "global", tag_txt, aux_txt, 0)]
    elif kind == "trans":
        # Fused stage transition: 1 timer read + 2 back-to-back 16B stores (END|prev_ev, next_ev) in 1 asm barrier.
        w = "local" if cfg.timer == 1 else "global"
        recs = [(w, "tagv", "$8", 0), ("reuse", "tagv2", "$8", 16)]
    elif kind == "open":
        tag_txt, aux_txt = str(int(tag_imm)), "sm"
        recs = [("global", tag_txt, aux_txt, 0)]
        if cfg.timer == 1:
            recs.append(("local", str(FI_TAG_CLK), "0", 16))
    elif kind == "close":
        tag_txt, aux_txt = str(int(tag_imm)), "0"
        recs = [("global", tag_txt, aux_txt, 0)]
        if cfg.timer == 1:
            recs.append(("local", str(FI_TAG_CLK), "1", 16))
    else:
        raise ValueError(f"未知 kind={kind}")

    # Triton 3.6 的 constraints 解析器不认 clobber（"=r,r,~memory" 会让 MLIR UNREACHABLE abort），
    # fence=1 改用 asm 块内首尾各一条 membar.gl（PTX 级 GPU 范围内存栅栏，效果只会更保守）。
    if kind == "event":
        constraints = "=r,r,l,l,r,r,r,r" + (",r" if user_pred else "")   # out, scratch, rec, ovf, pid, n, tag, aux[, pred]
        pred_op = "$8"
    elif kind == "trans":
        constraints = "=r,r,l,l,r,r,r,r,r"   # out, scratch, rec, ovf, pid, n, prev_tag, next_tag, aux
        pred_op = "$9"
    else:
        constraints = "=r,r,l,l,r,r" + (",r" if user_pred else "")       # out, scratch, rec, ovf, pid, n[, pred]
        pred_op = "$6"

    slot_src = bk.slot_sm if cfg.gran == 1 else bk.slot.replace("{W}", str(cfg.warps))
    pred_src = bk.rec_pred_sm if cfg.gran == 1 else bk.rec_pred
    if user_pred:
        # 记录线程由调用方的谓词决定（gran=1 时再与“0 号线程”取与）。
        # 谓词必须在该 warp/CTA 内恰好对一个线程为真，否则同一 slot 会被写多次。
        pred_src = (pred_src + f"\n    setp.ne.u32 p_user, {pred_op}, 0;\n    and.pred p_rec, p_rec, p_user;"
                    if cfg.gran == 1 else f"setp.ne.u32 p_rec, {pred_op}, 0;")

    names64 = ["addr", "off", "soff"] + [f"t{i}" for i in range(len(recs))]
    names32 = (["lane", "warp", "sm", "tid", "slot", "pid", "n", "ovfold", "tagv", "tagv2"]
               + [x for i in range(len(recs)) for x in (f"tlo{i}", f"thi{i}")])
    lines = ["{"]
    if bk.reg64:
        lines.append("    " + bk.reg64.format(NAMES=", ".join(names64)))
    if bk.reg32:
        lines.append("    " + bk.reg32.format(NAMES=", ".join(names32)))
    if bk.pred_reg:
        preds = ["p_rec", "p_ovf", "p_lane", "p_bounds", "p_cap", "p_store"]
        if user_pred and cfg.gran == 1:
            preds.append("p_user")
        lines.append("    " + bk.pred_reg.format(NAMES=", ".join(preds)))
    if cfg.fence:
        lines.append("    membar.gl;")
    ids_src = bk.ids if kind == "open" else "mov.u32 lane, %laneid;\n    mov.u32 tid, %tid.x;\n    shr.u32 warp, tid, 5;"
    lines += [
        "    mov.u32 pid, $4;",
        "    mov.u32 n, $5;",
        "    " + ids_src,
        "    " + slot_src,
        "    " + pred_src,
        "    and.pred p_lane, p_rec, p_rec;  // original recorder lane for overflow attribution",
        f"    setp.lt.u32 p_bounds, warp, {cfg.warps};",
        f"    setp.lt.u32 p_cap, slot, {cfg.n_slots};",
        "    and.pred p_bounds, p_bounds, p_cap;",
        "    and.pred p_rec, p_rec, p_bounds;  // all stores are gated by this predicate",
        f"    mul.wide.u32 soff, slot, {cfg.cap * 16};",
        "    cvt.u64.u32 off, n;",
        "    mad.lo.u64 off, off, 16, soff;",
        "    add.u64 addr, $2, off;",
    ]
    if kind == "event":
        lines.append(f"    and.b32 tagv, $6, 0x{FI_TAG_WIREMASK:08X};")
    elif kind == "trans":
        lines.append(f"    and.b32 tagv, $6, 0x{FI_TAG_WIREMASK:08X};")
        lines.append(f"    and.b32 tagv2, $7, 0x{FI_TAG_WIREMASK:08X};")
    gate_plan = _record_gate_plan(cfg.cap, len(recs))
    for i, (which, tag_txt, aux_txt, off) in enumerate(recs):
        addr = "addr" if off == 0 else f"addr+{off}"
        # 每条物理记录单独检查自己的下标。FI_TIMER=1 时 OPEN/CLOSE 会写两条，
        # 不能只检查开头那条：n=cap-1 时第二条仍会越过 cap。
        cap_limit = gate_plan.record_cap_limits[i]
        lines.append(f"    setp.lt.u32 p_cap, n, {cap_limit};")
        lines.append("    and.pred p_store, p_rec, p_cap;")
        if which == "reuse":
            tlo_nm, thi_nm = "tlo0", "thi0"
        else:
            src = bk.timer_local if which == "local" else bk.timer_global
            lines.append("    @p_store " + src.replace("{T}", f"t{i}"))
            lines.append(f"    @p_store mov.b64 {{tlo{i}, thi{i}}}, t{i};")
            tlo_nm, thi_nm = f"tlo{i}", f"thi{i}"
        # store16 里 PTX 的向量花括号用 .format() 的双花括号写，避免和占位符混淆
        st = bk.store16.format(ADDR=addr, TLO=tlo_nm, THI=thi_nm, TAG=tag_txt, AUX=aux_txt)
        lines.append("    @p_store " + st)
    full_at = gate_plan.lane_full_at
    lines += [f"    setp.ge.u32 p_ovf, n, {full_at};",
              "    and.pred p_ovf, p_ovf, p_rec;",
              "    @p_ovf " + bk.atom_or.replace("{PTR}", "$3").replace("{BIT}", str(FI_OVF_LANE_FULL)),
              f"    setp.ge.u32 p_ovf, warp, {cfg.warps};",
              "    and.pred p_ovf, p_ovf, p_lane;",
              "    @p_ovf " + bk.atom_or.replace("{PTR}", "$3").replace("{BIT}", str(FI_OVF_SLOT_OOB)),
              f"    setp.ge.u32 p_ovf, slot, {cfg.n_slots};",
              "    and.pred p_ovf, p_ovf, p_lane;",
              "    @p_ovf " + bk.atom_or.replace("{PTR}", "$3").replace("{BIT}", str(FI_OVF_SLOT_OOB))]
    if cfg.fence:
        lines.append("    membar.gl;")
    lines += ["    mov.u32 $0, $1;", "}"]
    return "\n".join(lines), constraints, len(recs)


# --------------------------------------------------------------------------
# 设备侧 API（在 @triton.jit kernel 里调用）。全部返回“写完之后的条数 n”，
# 一律写成 n = fi_xxx(...)：Triton 会把返回值变成 i32 SSA 标量，循环里才会真的递增。
# --------------------------------------------------------------------------


if _TRITON_AVAILABLE:
    @triton.jit
    def wp_thread_ids(THREADS: tl.constexpr):
        return tl.inline_asm_elementwise("mov.u32 $0, %tid.x;", "=r,r", [tl.arange(0, THREADS)],
                                         dtype=tl.int32, is_pure=True, pack=1)


    @triton.jit
    def fi_open(rec_ptr, ovf_ptr, pid, FI: tl.constexpr, role: tl.constexpr):
        """FI_OPEN(role)：kernel 入口调一次（全 CTA 执行），记物理 SM 号与角色；返回 n。"""
        if FI.level >= 1:
            th: tl.constexpr = FI.threads
            scratch = tl.zeros((th,), dtype=tl.int32)
            _ = tl.inline_asm_elementwise(asm=FI.open_asm(role), constraints=FI.open_cons(role),
                                          args=[scratch, rec_ptr, ovf_ptr, pid, 0],
                                          dtype=tl.int32, is_pure=False, pack=1)
            return 0 + FI.open_nr
        return 0


    @triton.jit
    def fi_open_by(rec_ptr, ovf_ptr, pid, FI: tl.constexpr, pred, role: tl.constexpr):
        """FI_OPEN_BY(pred, role)：由 pred（每线程一个 0/1，形状 [FI.threads]）指定写 OPEN 的线程。
        只让 0 号 warp 记录：pred = (tl.arange(0, THREADS) < 32).to(tl.int32)。

        pred 只作用于这一条 OPEN 记录。后面的 fi_b/fi_e/fi_m 仍按 gran 的默认谓词（gran=0 是每个 warp 的
        lane0）写各自泳道，于是会出现“只有事件、没有 OPEN/CLOSE”的残缺泳道，解码器把它们记进 errors
        的未配对项。要整条泳道都由同一个谓词决定，事件和收尾也要用 _by 变体：
        fi_b_by / fi_e_by / fi_m_by / fi_close_by，谓词必须和 fi_open_by 用的那一个完全相同。
        """
        if FI.level >= 1:
            th: tl.constexpr = FI.threads
            scratch = tl.zeros((th,), dtype=tl.int32)
            _ = tl.inline_asm_elementwise(asm=FI.openby_asm(role), constraints=FI.openby_cons(role),
                                          args=[scratch, rec_ptr, ovf_ptr, pid, 0, pred],
                                          dtype=tl.int32, is_pure=False, pack=1)
            return 0 + FI.open_nr
        return 0


    @triton.jit
    def fi_close(rec_ptr, ovf_ptr, pid, n, FI: tl.constexpr):
        """FI_CLOSE()：每个出口都要调（FI_TIMER=1 时它提供第二个锚点，缺了解码器会报错）。"""
        if FI.level >= 1:
            th: tl.constexpr = FI.threads
            scratch = tl.zeros((th,), dtype=tl.int32)
            _ = tl.inline_asm_elementwise(asm=FI.close_asm, constraints=FI.close_cons,
                                          args=[scratch, rec_ptr, ovf_ptr, pid, n],
                                          dtype=tl.int32, is_pure=False, pack=1)
            return n + FI.close_nr
        return n


    @triton.jit
    def _fi_event(rec_ptr, ovf_ptr, pid, n, FI: tl.constexpr, tag, aux):
        """fi_b/fi_e/fi_m 的共同实现：tag 已含 END/MARK 位。"""
        th: tl.constexpr = FI.threads
        scratch = tl.zeros((th,), dtype=tl.int32)
        _ = tl.inline_asm_elementwise(asm=FI.event_asm, constraints=FI.event_cons,
                                      args=[scratch, rec_ptr, ovf_ptr, pid, n, tag, aux],
                                      dtype=tl.int32, is_pure=False, pack=1)
        return n + FI.event_nr


    @triton.jit
    def _fi_event_by(rec_ptr, ovf_ptr, pid, n, FI: tl.constexpr, tag, aux, pred):
        """带用户谓词的事件实现：只有 pred!=0 的线程写记录（必须与 fi_open_by 的谓词一致）。"""
        th: tl.constexpr = FI.threads
        scratch = tl.zeros((th,), dtype=tl.int32)
        _ = tl.inline_asm_elementwise(asm=FI.event_by_asm, constraints=FI.event_by_cons,
                                      args=[scratch, rec_ptr, ovf_ptr, pid, n, tag, aux, pred],
                                      dtype=tl.int32, is_pure=False, pack=1)
        return n + FI.event_by_nr


    @triton.jit
    def fi_b_by(rec_ptr, ovf_ptr, pid, n, FI: tl.constexpr, ev: tl.constexpr, aux, pred):
        """FI_B 的谓词版：整条泳道由 pred 决定，与 fi_open_by 配套用。"""
        if FI.level >= 2:
            event_id: tl.constexpr = ev & FI_TAG_IDMASK_C
            return _fi_event_by(rec_ptr, ovf_ptr, pid, n, FI, event_id, aux, pred)
        return n


    @triton.jit
    def fi_e_by(rec_ptr, ovf_ptr, pid, n, FI: tl.constexpr, ev: tl.constexpr, aux, pred):
        """FI_E 的谓词版。"""
        if FI.level >= 2:
            event_id: tl.constexpr = ev & FI_TAG_IDMASK_C
            return _fi_event_by(rec_ptr, ovf_ptr, pid, n, FI, event_id | FI_TAG_END_C, aux, pred)
        return n


    @triton.jit
    def fi_m_by(rec_ptr, ovf_ptr, pid, n, FI: tl.constexpr, ev: tl.constexpr, aux, pred):
        """FI_M 的谓词版。"""
        if FI.level >= 2:
            event_id: tl.constexpr = ev & FI_TAG_IDMASK_C
            return _fi_event_by(rec_ptr, ovf_ptr, pid, n, FI, event_id | FI_TAG_MARK_C, aux, pred)
        return n


    @triton.jit
    def fi_close_by(rec_ptr, ovf_ptr, pid, n, FI: tl.constexpr, pred):
        """FI_CLOSE 的谓词版：用了 fi_open_by 就要用它收尾，否则该泳道没有 CLOSE。"""
        if FI.level >= 1:
            th: tl.constexpr = FI.threads
            scratch = tl.zeros((th,), dtype=tl.int32)
            _ = tl.inline_asm_elementwise(asm=FI.close_by_asm, constraints=FI.close_by_cons,
                                          args=[scratch, rec_ptr, ovf_ptr, pid, n, pred],
                                          dtype=tl.int32, is_pure=False, pack=1)
            return n + FI.close_by_nr
        return n


    @triton.jit
    def fi_b(rec_ptr, ovf_ptr, pid, n, FI: tl.constexpr, ev: tl.constexpr, aux):
        """FI_B(ev, aux)：区间开始。ev 取 1..4095（0 保留给空记录）。"""
        if FI.level >= 2:
            event_id: tl.constexpr = ev & FI_TAG_IDMASK_C
            return _fi_event(rec_ptr, ovf_ptr, pid, n, FI, event_id, aux)
        return n


    @triton.jit
    def fi_e(rec_ptr, ovf_ptr, pid, n, FI: tl.constexpr, ev: tl.constexpr, aux):
        """FI_E(ev, aux)：区间结束。异步操作（TMA/cp.async/wgmma）要在 wait 返回之后记，不记在发射处。"""
        if FI.level >= 2:
            event_id: tl.constexpr = ev & FI_TAG_IDMASK_C
            return _fi_event(rec_ptr, ovf_ptr, pid, n, FI, event_id | FI_TAG_END_C, aux)
        return n


    @triton.jit
    def fi_trans(rec_ptr, ovf_ptr, pid, n, FI: tl.constexpr, prev_ev: tl.constexpr, next_ev: tl.constexpr, aux):
        """Fused stage transition: closes `prev_ev` and opens `next_ev` with 1 timer read in 1 asm barrier."""
        if FI.level >= 2:
            th: tl.constexpr = FI.threads
            scratch = tl.zeros((th,), dtype=tl.int32)
            prev_tag: tl.constexpr = (prev_ev & FI_TAG_IDMASK_C) | FI_TAG_END_C
            next_tag: tl.constexpr = next_ev & FI_TAG_IDMASK_C
            _ = tl.inline_asm_elementwise(asm=FI.trans_asm, constraints=FI.trans_cons,
                                          args=[scratch, rec_ptr, ovf_ptr, pid, n, prev_tag, next_tag, aux],
                                          dtype=tl.int32, is_pure=False, pack=1)
            return n + FI.trans_nr
        return n


    @triton.jit
    def fi_m(rec_ptr, ovf_ptr, pid, n, FI: tl.constexpr, ev: tl.constexpr, aux):
        """FI_M(ev, aux)：瞬时事件。"""
        if FI.level >= 2:
            event_id: tl.constexpr = ev & FI_TAG_IDMASK_C
            return _fi_event(rec_ptr, ovf_ptr, pid, n, FI, event_id | FI_TAG_MARK_C, aux)
        return n


    @triton.jit
    def fi_bg(rec_ptr, ovf_ptr, pid, n, FI: tl.constexpr, g: tl.constexpr, ev: tl.constexpr, aux):
        """FI_BG(group, ev, aux)：被 FI.groups 屏蔽的组编译期就消失（压开销 / 二分定位开销来源）。"""
        groups: tl.constexpr = FI.groups
        if FI.level >= 2 and ((groups >> (g & 31)) & 1):
            event_id: tl.constexpr = ev & FI_TAG_IDMASK_C
            return _fi_event(rec_ptr, ovf_ptr, pid, n, FI, event_id, aux)
        return n


    @triton.jit
    def fi_eg(rec_ptr, ovf_ptr, pid, n, FI: tl.constexpr, g: tl.constexpr, ev: tl.constexpr, aux):
        """FI_EG(group, ev, aux)。"""
        groups: tl.constexpr = FI.groups
        if FI.level >= 2 and ((groups >> (g & 31)) & 1):
            event_id: tl.constexpr = ev & FI_TAG_IDMASK_C
            return _fi_event(rec_ptr, ovf_ptr, pid, n, FI, event_id | FI_TAG_END_C, aux)
        return n


    @triton.jit
    def fi_mg(rec_ptr, ovf_ptr, pid, n, FI: tl.constexpr, g: tl.constexpr, ev: tl.constexpr, aux):
        """FI_MG(group, ev, aux)。"""
        groups: tl.constexpr = FI.groups
        if FI.level >= 2 and ((groups >> (g & 31)) & 1):
            event_id: tl.constexpr = ev & FI_TAG_IDMASK_C
            return _fi_event(rec_ptr, ovf_ptr, pid, n, FI, event_id | FI_TAG_MARK_C, aux)
        return n

    wp_open = fi_open
    wp_open_by = fi_open_by
    wp_b = fi_b
    wp_e = fi_e
    wp_m = fi_m
    wp_trans = fi_trans
    wp_bg = fi_bg
    wp_eg = fi_eg
    wp_mg = fi_mg
    wp_close = fi_close
    wp_b_by = fi_b_by
    wp_e_by = fi_e_by
    wp_m_by = fi_m_by
    wp_close_by = fi_close_by
    WP_OPEN = fi_open
    WP_OPEN_BY = fi_open_by
    WP_B = fi_b
    WP_E = fi_e
    WP_M = fi_m
    WP_TRANS = fi_trans
    WP_CLOSE = fi_close
else:
    def __getattr__(name: str) -> Any:  # PEP 562 lazy failure for device symbols
        if name in _DEVICE_SYMBOLS:
            raise ImportError(
                f"wp_triton.{name} needs Triton + torch (CUDA); import of triton failed: {_IMPORT_ERROR}"
            )
        raise AttributeError(f"module 'wp_triton' has no attribute {name!r}")


# --------------------------------------------------------------------------
# host 侧：等价 fi_alloc + fi_attach + fi_reset + fi_dump
# --------------------------------------------------------------------------
def default_timer_for_device(device: str = "cuda", explicit: Optional[int] = None) -> int:
    """Select high-resolution timer automatically: sm_86/sm_89 consumer GPUs have 1024 ns
    %globaltimer steps, so default to timer=1 (%clock64 + %globaltimer OPEN/CLOSE anchors,
    ~27 ns step) unless WP_TIMER/FI_TIMER or explicit timer is specified."""
    if explicit is not None:
        return int(explicit)
    for k in ("WP_TIMER", "FI_TIMER"):
        v = os.environ.get(k)
        if v not in (None, ""):
            return int(v)
    try:
        if torch is not None and torch.cuda.is_available():
            dev_idx = torch.device(device).index
            cap = torch.cuda.get_device_capability(dev_idx) if dev_idx is not None else torch.cuda.get_device_capability()
            if tuple(cap) in ((8, 6), (8, 9)):
                return 1
    except Exception:
        pass
    return 0


class FiBuffer:
    """n_ctas: grid 里 CTA 总数；warps_per_cta: num_warps；cap: 每泳道最大记录条数。
    gran=1（SM/CU 粒度）时每 CTA 只有 1 条泳道，转储头里的 warps_per_cta 写 1。
    rec/overflow 两个张量直接当 kernel 实参传（Triton 取 data_ptr）。"""

    def __init__(self, n_ctas: int, warps_per_cta: int, cap: int, device: str = "cuda",
                 gran: int = 0, timer: Optional[int] = None, level: int = 2, groups: int = 0xFFFFFFFF,
                 backend: str = "nvidia_ptx", fence: int = 0):
        self.n_ctas = int(n_ctas)
        self.gran = int(gran)
        self.num_warps = int(warps_per_cta)
        self.warps_per_cta = 1 if self.gran == 1 else self.num_warps
        self.cap = int(cap)
        self.n_slots = self.n_ctas * self.warps_per_cta
        self.device = device
        resolved_timer = default_timer_for_device(device, timer)
        self.rec = torch.zeros((self.n_slots, self.cap, 4), dtype=torch.int32, device=device)
        self.overflow = torch.zeros((1,), dtype=torch.int32, device=device)
        self.cfg = FiCfg(cap=self.cap, warps=self.warps_per_cta, n_slots=self.n_slots, gran=self.gran,
                         timer=resolved_timer, level=level, groups=groups, threads=self.num_warps * 32,
                         backend=backend, fence=fence)

    @property
    def nbytes(self) -> int:
        return self.rec.numel() * self.rec.element_size()

    def reset(self):
        """等价 fi_reset：要采的那次 launch 之前清零（放计时区间外）。"""
        self.rec.zero_()
        self.overflow.zero_()
        return self

    def counts(self):
        """每泳道实际记录条数（tag==0 视为空记录，与 CUDA 版 fi_dump 同一判据）。"""
        tag = self.rec[:, :, 2]
        zero = (tag == 0).to(torch.int32)
        any_zero = zero.any(dim=1)
        first_zero = zero.argmax(dim=1)
        return torch.where(any_zero, first_zero, torch.full_like(first_zero, self.cap))

    def dump(self, path, meta: Optional[dict] = None) -> int:
        """等价 fi_dump：写 FILITE02（magic|6*u32 头|meta JSON|cnt[n_slots]|紧凑记录）。
        返回 overflow 位（>0 表示写满或 slot 越界）。"""
        import numpy as np
        torch.cuda.synchronize()
        arr = self.rec.detach().cpu().numpy()
        ovf = int(self.overflow.detach().cpu().item())
        zero = arr[:, :, 2] == 0
        counts = np.where(zero.any(axis=1), zero.argmax(axis=1), self.cap).astype(np.uint32)
        take = np.arange(self.cap, dtype=np.int32)[None, :] < counts[:, None]
        flat = arr[take]
        recs = np.empty(len(flat), dtype=np.dtype([("t", "<u8"), ("tag", "<u4"), ("aux", "<u4")]))
        # PTX stores two unsigned words in the int32 backing tensor.
        words = flat.view(np.uint32)
        recs["t"] = words[:, 0].astype(np.uint64) | (words[:, 1].astype(np.uint64) << 32)
        recs["tag"] = flat[:, 2].astype(np.uint32)
        recs["aux"] = flat[:, 3].astype(np.uint32)
        mb = json.dumps(self.cfg.meta(**(meta or {})), ensure_ascii=False,
                        separators=(",", ":")).encode("utf-8")
        hdr = struct.pack("<6I", self.n_slots, self.cap, self.warps_per_cta, ovf, self.cfg.timer, len(mb))
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(FI_MAGIC)
                f.write(hdr)
                f.write(mb)
                f.write(counts.astype("<u4").tobytes())
                recs.tofile(f)
        except OSError as e:
            # nsys 下被测进程常常写不了工作目录（errno=13），而 PermissionError 会把整轮采集带走、
            # 只留一行看不出原因的回溯（fi_bugs_h20-sk-qwen B10）。把路径与 errno 讲清楚，并给下一步。
            raise OSError(
                e.errno,
                f"FiBuffer.dump 写不出转储 {path}：{e.strerror}（errno={e.errno}）。"
                "nsys/rocprofv3 下被测进程可能写不了工作目录：改成 /tmp 下的路径，采集结束后再拷回；"
                "或先把目标目录 chmod 0777，并确认 TRITON_CACHE_DIR/HOME 对该 UID 可写") from e
        return ovf


def cap_for(records_per_tile: int, max_tiles_per_cta: int, extra: int = 8, timer: int = 0) -> int:
    """容量公式：每泳道条数 = 每 tile 记录数 * 每 CTA 最多 tile 数 + OPEN/CLOSE(+锚点) + 余量。
    宁大勿小：写满会置 overflow bit0，解码器会报“泳道写满”。"""
    return records_per_tile * max_tiles_per_cta + extra + (4 if timer == 1 else 2)


def _json_name_key(key) -> str:
    """把 tl.constexpr/int 等 key 规范成 JSON 对象 key；事件号写成十进制。"""
    value = _uw(key)
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _json_ready(value):
    """递归展开 tl.constexpr，并把映射 key 统一成字符串，保证 names.json 可序列化。"""
    value = _uw(value)
    if isinstance(value, dict):
        out = {}
        origin = {}
        for key, item in value.items():
            normalized = _json_name_key(key)
            if normalized in out:
                raise ValueError(
                    f"write_names: key {key!r} 与 {origin[normalized]!r} 规范化后都是 {normalized!r}")
            origin[normalized] = key
            out[normalized] = _json_ready(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _normalized_name_map(values, field: str) -> dict:
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise TypeError(f"write_names: {field} 必须是 dict，收到 {type(values).__name__}")
    out = {}
    origin = {}
    for key, value in values.items():
        normalized = _json_name_key(key)
        if normalized in out:
            raise ValueError(
                f"write_names: {field} 的 key {key!r} 与 {origin[normalized]!r} 规范化后都是 {normalized!r}")
        origin[normalized] = key
        out[normalized] = _json_ready(value)
    return out


def write_names(path, events: dict, roles: Optional[dict] = None, aux: Optional[dict] = None,
                group: str = "role", pair: str = "stack", label_aux: bool = True, **extra) -> str:
    """写 names.json（事件名必须是真实语义，不写 ev3/phase2）。

    key 可以是 int、字符串或 ``tl.constexpr(EV_MASK)``；会先取 constexpr 的值再转成
    JSON 字符串。``tl.constexpr(1)`` 因此写成 ``"1"``，不会写成 ``"constexpr[1]"``。
    规范化后撞 key（例如 ``1`` 与 ``"1"`` 同时出现）直接报错，避免静默覆盖。
    """
    obj = dict(events=_normalized_name_map(events, "events"),
               roles=_normalized_name_map(roles or {}, "roles"),
               aux=_normalized_name_map(aux or {}, "aux"),
               group=group, pair=pair, label_aux=label_aux)
    obj.update({key: _json_ready(value) for key, value in extra.items()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


_FI_ACTIVE_BUF: Optional["FiBuffer"] = None


def fi_attach(buf: Optional["FiBuffer"] = None) -> Optional["FiBuffer"]:
    """Attach (or detach when ``None``) a module-level ``FiBuffer`` for zero-signature-edit launches."""
    global _FI_ACTIVE_BUF
    _FI_ACTIVE_BUF = buf
    return buf


def fi_get_attached() -> Optional["FiBuffer"]:
    return _FI_ACTIVE_BUF


class _FiTritonProxy:
    """Same-name transparent Triton launcher proxy (§3.2 2a — 0 Call-Chain Signature Edits).
    Reuses `@triton.autotune` winning `best_config` from `L0` without overflowing or perturbing `do_bench`."""

    def __init__(self, l0_fn, inst_fn, buf_getter=fi_get_attached):
        self._l0_fn = l0_fn
        self._inst_fn = inst_fn
        self._buf_getter = buf_getter
        self.__name__ = getattr(l0_fn, "__name__", "triton_kernel")

    def _find_autotuner(self, fn):
        cur = fn
        for _ in range(4):
            if hasattr(cur, "configs"):
                return cur
            nxt = getattr(cur, "fn", None)
            if nxt is None or nxt is cur:
                break
            cur = nxt
        return None

    def _autotune_kwargs(self, existing_kwargs: dict) -> dict:
        l0_at = self._find_autotuner(self._l0_fn)
        if l0_at is None or self._find_autotuner(self._inst_fn) is not None:
            return existing_kwargs
        best = getattr(l0_at, "best_config", None)
        if best is None:
            cache = getattr(l0_at, "cache", None) or {}
            if cache:
                best = next(reversed(cache.values()))
            elif getattr(l0_at, "configs", None):
                best = l0_at.configs[0]
        if best is None:
            return existing_kwargs
        merged = dict(best.all_kwargs()) if hasattr(best, "all_kwargs") else dict(getattr(best, "kwargs", {}))
        merged.update(existing_kwargs)
        return merged

    def __getitem__(self, grid):
        buf = self._buf_getter()
        if buf is None or buf.cfg.level == 0:
            return self._l0_fn[grid]
        inst_launcher = self._inst_fn[grid]

        def _wrapped(*args, **kwargs):
            call_kw = self._autotune_kwargs(kwargs)
            return inst_launcher(*args, _fi_rec_ptr=buf.rec, _fi_ovf_ptr=buf.overflow,
                                 _FI_CFG=buf.cfg, **call_kw)

        return _wrapped

    def __getattr__(self, name: str):
        buf = self._buf_getter()
        target = self._l0_fn if (buf is None or buf.cfg.level == 0) else self._inst_fn
        return getattr(target, name)


# Canonical WeaveProfiler (`wp_*` / `Wp*`) host aliases
WpCfg = FiCfg
WpBuffer = FiBuffer
wp_attach = fi_attach
wp_get_attached = fi_get_attached
_WpTritonProxy = _FiTritonProxy
wp_wrap = _WpTritonProxy


def wp_cfg(**kwargs: Any) -> FiCfg:
    """Build a ``WpCfg`` (``FiCfg``) whose level/gran/timer default to WP_LEVEL/WP_GRAN/WP_TIMER."""
    kwargs.setdefault("level", WP_LEVEL)
    kwargs.setdefault("gran", WP_GRAN)
    kwargs.setdefault("timer", WP_TIMER)
    return FiCfg(**kwargs)


def wp_available() -> bool:
    return _TRITON_AVAILABLE and torch is not None


__all__ = sorted(
    (_DEVICE_SYMBOLS if _TRITON_AVAILABLE else set())
    | {
        "WP_LEVEL", "WP_GRAN", "WP_TIMER",
        "WP_TAG_END", "WP_TAG_OPEN", "WP_TAG_MARK", "WP_TAG_CLK", "WP_TAG_CLOSE", "WP_TAG_IDMASK",
        "FiCfg", "WpCfg", "FiBuffer", "WpBuffer", "FiAsmBackend", "FI_BACKENDS", "FI_BACKEND_NVIDIA",
        "RecordGatePlan", "record_gate_status", "resolve_arch", "cap_for", "write_names",
        "fi_attach", "wp_attach", "fi_get_attached", "wp_get_attached",
        "_FiTritonProxy", "_WpTritonProxy", "wp_wrap", "wp_cfg", "wp_available",
    }
)
