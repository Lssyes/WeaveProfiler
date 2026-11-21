# -*- coding: utf-8 -*-
"""fast-instrument 检查器：一条命令汇总 C1–C4，写出单一 check.json。NVIDIA 与 AMD 走同一个入口。

平台判定：--platform auto 时按 --arch（gfx* 即 AMD）与二进制内容（.hip_fatbin / AMDGPU ELF）判定。
  NVIDIA：C1 用 cuobjdump -res-usage + ptxas 日志 + 每 SM 驻留 CTA 估算 + fi_binsim（SASS）；
          C4 用 nsys sqlite 对齐（fi_align_nsys.py）。
  AMD：C1 用 AMDGCN code object 的 metadata（VGPR/SGPR/AGPR/scratch/LDS/spill）+ llvm-objdump 反汇编
          的指令类计数 + fi_binsim_hip.py（AMDGCN）；C4 用 rocprofv3 对齐（fi_align_rocprof.py）。
  C4 的“每泳道 clock64 换算频率”只有 FI_TIMER=1 才有证据（OPEN/CLOSE 双锚点）；默认档 FI_TIMER=0
          用 %globaltimer（NVIDIA）或 s_memrealtime（AMD），本来就是 ns 墙钟，该项记为不适用而不是判失败。

C1 资源与同步（NVIDIA）：cuobjdump -res-usage、ptxas spill 日志、每 SM 驻留 CTA 估算、fi_binsim SASS 结果。
   寄存器增幅超预算但“每 SM 驻留 CTA 不降 + spill 不增”时记 warning 不判失败（与 AMD 同口径）：
   FiCtx 常驻 4~8 个寄存器是设计固有代价，低寄存器基线必然让百分比超 10%
   （mail-12 实测 L0=48 -> FI=56、spill 0->0、CTA/SM 4->4 曾被误判 FAIL）。--strict-resources 关闭该降级。
C1 资源与同步（AMD）：见下面“AMD C1 判据”。
C2 输出正确性：调用调用方提供的 randn 全量数组比较脚本（模板 knowledge/templates/compare_raw_arrays.py）。
   要求比较脚本 rc=0、显式 pass=true、键集合完全一致，并把结构化 outputs 的逐项元素数、
   total_elements、checked_elements 与 --expected-elements 对齐；缺字段或只报 1 元素不能通过。
C3 事件完整：FILITE02 解码结构错误、未配对/交叉/负时长、OPEN/CLOSE 锚点、溢出位、未命名事件。
C4 计时：globaltimer/clock64 最小步长、每泳道 clock64 换算 MHz 范围（FI_TIMER=0 时不适用）、
   与 nsys/rocprofv3 kernel 条结束余量，外加对齐后的时间轴自检（软件轨必须落在 kernel 条窗口内）。
   最终门禁还要求 align.ok、metrics_ok、axis_check.ok 严格为 true，errors 为空，对齐进程 rc=0。
Artifacts：直接检查 --trace 最终文件，记录 SHA256，并要求至少一个非 info/nsys_kernel/rocprofv3_kernel 的业务 X。
Logs：调用 fi_logcheck 检查驱动/采集日志与 .excp(.gz)，clean/rc 纳入 overall_pass。

用法（证据项缺失不会崩溃，会明确 FAIL，并在 suggestions 写可执行修法）：
  python3 -I fi_check.py --name ring_pipeline \
    --l0 build/ring_L0 --fi build/ring_L2 --kernel ring_pipeline \
    --l0-build-log logs/ring_L0.log --fi-build-log logs/ring_L2.log \
    --block-threads 128 --dynamic-smem 49152 \
    --dump run_L2c.bin --names names.json \
    --compare-cmd "python3 -I compare_raw_arrays.py --ref ref.npz --candidate fi.npz --atol 0 --rtol 0" \
    --sqlite cap.sqlite --nsys-kernel ring_pipeline \
    --trace out/final.perfetto.json.gz --logs logs/run.log \
    -o check.json
  # AMD（gfx942）：直接给两个 HIP 可执行文件，工具自己抽 code object
  python3 -I fi_check.py --platform amd --arch gfx942 --name ring \
    --l0 build/ring_l0 --fi build/ring_l2g --kernel ring_pipeline \
    --dump run.fi.bin --names names.json --rocprof out/rocprof \
    --trace out/final.perfetto.json.gz --logs logs/run.log -o check.json
  # 注意：只有 --kernel 一个名字参数（正则，同时用于资源挑选与两个 binsim 工具），没有 --kernel-regex
  # AMD 离线回归：没有 ROCm 工具链时，用预先抽好的资源 JSON（schema 见 --l0-amd-res 帮助）
  python3 -I fi_check.py --platform amd --arch gfx942 --l0-amd-res tests/fixtures/amd_c1/ring_l0.json \
    --fi-amd-res tests/fixtures/amd_c1/ring_l2g.json --kernel ring_pipeline -o check.json
  python3 -I fi_check.py --selftest

AMD C1 判据（修掉“正确插桩被判资源扰动”的假阳性）：
  寄存器数在 AMD 上不单独判失败：它只通过“每 SIMD 可驻留 wave 数”起作用，超预算但占用率没塌就记 warning
    （要更严用 --strict-resources 逐项相等，或提高 --min-occupancy-ratio）。
  FAIL（结构性/语义性扰动，插桩绝不该造成）：目标 kernel 在插桩产物里消失；s_barrier 静态条数变化；
    静态 LDS（.group_segment_fixed_size）变化；scratch（.private_segment_fixed_size）增量 > --max-scratch-delta-bytes
    （默认 256 B/work-item，MI308X 上正确插桩实测 0→92，注入 scratch 故障的版本是 0→352）；
    每 SIMD 可驻留 wave 数低于基线的 --min-occupancy-ratio（默认 0.5，即“腰斩”才算崩塌；
    9→8、6→5 这类插桩固有代价不判失败）；反汇编里找不到 FI 探针——探针 = s_memrealtime/s_memtime 读的增量
    加上任一种记录写签名的增量（HIP 头文件的 16B AoS global_store_dwordx3/x4，或 Triton AMD 前端的 SoA
    global_store_dwordx2 时间戳 + global_store_dword tag/aux）。
  warning（插桩固有代价，必须报数但不判失败）：VGPR/SGPR 增长、waves/SIMD 在预算内下降、出现 spill、s_waitcnt 增多。
  gfx942 占用率公式：allocated_vgpr = max(8, ceil(vgpr/8)*8)，寄存器上限 = floor(512/allocated_vgpr)；
    agpr>0 时再与 floor(512/allocated_agpr) 取小（CDNA 的 AGPR 是独立 512 项文件）；
    最后**必须**与硬件并发上限取小：max_waves_per_simd = max_waves_per_cu / simds_per_cu（gfx942 = 32/4 = 8，
    也等于 max_threads_per_cu 2048 / wavefront 64 / 4）。不截断会算出 21、64 这种硬件上不可能的
    waves/SIMD，再把两臂其实相同的占用率判成“崩塌”（fi_bugs_amd-sk-qwen B15：ring_pipeline VGPR 20->60
    被报成 21->8、ratio 0.381，而 48 KB 动态 LDS 早就把每 CU 卡在 1 个 CTA = 1 wave/SIMD，两臂相同）。
    未给 --dynamic-smem 时 LDS 上限只按静态 .group_segment_fixed_size 算，会记 warning（Triton 的 LDS 全是动态的）。

默认阈值适用于“插桩后仍保持正确、事件结构完整、计时可对齐、资源不显著退化”的验收：
  C1：寄存器增幅 <=10%，spill 不增加，smem 不变，每 SM CTA 不减少，sim_strict>=0.60，
      sim_opcode>=0.75，基本块增幅<=300%，至少 1 个新 FI probe，关键指令顺序保持。
  C2：checked_elements>0，非有限值=0，max_abs<=--max-abs，max_rel<=--max-rel（默认 0/0）。
  C3：结构错误=0、溢出=0、锚点错误=0、未命名事件=0。
  C4：最小步长<=128 ns，clock64 换算频率 300–5000 MHz 且相对中位跨度<=5%，nsys 结束余量 0–50 us。
所有阈值都原样写进 JSON，可用命令行参数覆盖；不被静默放宽。
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TOOL = "fast-instrument fi_check"
VERSION = "1.0"
HERE = Path(__file__).resolve().parent
PKG = HERE.parent.parent
DEFAULT_ALIGN = HERE / "fi_align_nsys.py"
DEFAULT_ALIGN_ROCPROF = HERE / "fi_align_rocprof.py"
DEFAULT_BINSIM = HERE / "fi_binsim.py"
DEFAULT_BINSIM_HIP = HERE / "fi_binsim_hip.py"
DEFAULT_DECODER = HERE / "fi_lite_decode.py"
DEFAULT_LOGCHECK = HERE / "fi_logcheck.py"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(argv: Sequence[str], timeout: int, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(list(argv), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(list(argv), 124, e.stdout or "", f"timeout after {timeout}s")
    except FileNotFoundError as e:
        return subprocess.CompletedProcess(list(argv), 127, "", str(e))


def item(passed: bool, metrics: Dict[str, object], thresholds: Dict[str, object],
         evidence: Dict[str, object], suggestions: List[str], note: str = "") -> Dict[str, object]:
    return {"pass": bool(passed), "metrics": metrics, "thresholds": thresholds,
            "evidence": evidence, "suggestions": suggestions if not passed else [], "note": note}


def as_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- resource usage / occupancy

RES_FUNC_RE = re.compile(r"^\s*Function\s+(\S+):")
RES_KV_RE = re.compile(r"\b([A-Z][A-Z0-9_\[\]]*):(\d+)\b")
ARCH_RE = re.compile(r"^\s*arch\s*=\s*(sm_\w+)")


def demangle(names: Iterable[str]) -> Dict[str, str]:
    names = list(dict.fromkeys(names))
    exe = shutil.which("c++filt") or "/usr/bin/c++filt"
    if not names or not os.path.exists(exe):
        return {n: n for n in names}
    p = run([exe, *names], 30)
    vals = p.stdout.splitlines()
    return dict(zip(names, vals)) if len(vals) == len(names) else {n: n for n in names}


def parse_res_usage(text: str, kernel_rx: str, preferred_arch: str = "sm_103a") -> Dict[str, object]:
    rx = re.compile(kernel_rx)
    arch = ""
    candidates = []
    cur_name = None
    cur_arch = ""
    cur = None
    for line in text.splitlines():
        m = ARCH_RE.match(line)
        if m:
            arch = m.group(1)
            continue
        fm = RES_FUNC_RE.match(line)
        if fm:
            if cur:
                candidates.append(cur)
            cur_name = fm.group(1)
            cur_arch = arch
            cur = {"raw_name": cur_name, "arch": cur_arch, "values": {}}
            continue
        if cur:
            for k, v in RES_KV_RE.findall(line):
                cur["values"][k] = int(v)
    if cur:
        candidates.append(cur)
    dm = demangle(x["raw_name"] for x in candidates)
    matched = []
    for x in candidates:
        x["name"] = dm.get(x["raw_name"], x["raw_name"])
        if rx.search(x["raw_name"]) or rx.search(x["name"]):
            matched.append(x)
    if not matched:
        raise ValueError(f"res-usage 中无 /{kernel_rx}/ kernel")
    matched.sort(key=lambda x: (x["arch"] == preferred_arch, len(x["values"])), reverse=True)
    x = matched[0]
    v = x["values"]
    out = {
        "kernel": x["name"], "raw_name": x["raw_name"], "arch": x["arch"],
        "registers_per_thread": v.get("REG"), "stack_bytes": v.get("STACK", 0),
        "static_smem_bytes": v.get("SHARED", 0), "local_bytes": v.get("LOCAL", 0),
        "constant0_bytes": v.get("CONSTANT[0]", 0), "texture_bytes": v.get("TEXTURE", 0),
        "surface_bytes": v.get("SURFACE", 0),
    }
    return out


def parse_build_log(text: str, kernel_rx: str) -> Dict[str, object]:
    rx = re.compile(kernel_rx)
    records = []
    current = None
    pending_props = False
    for line in text.splitlines():
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            current = {"raw_name": m.group(1)}
            records.append(current)
            pending_props = False
            continue
        if re.search(r"Function properties for '([^']+)'", line):
            pending_props = True
            continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and current:
            current.update(stack_frame_bytes=int(m.group(1)), spill_store_bytes=int(m.group(2)),
                           spill_load_bytes=int(m.group(3)))
            pending_props = False
            continue
        m = re.search(r"Used (\d+) registers", line)
        if m and current:
            current["registers_per_thread"] = int(m.group(1))
    dm = demangle(x["raw_name"] for x in records)
    matched = []
    for r in records:
        r["name"] = dm.get(r["raw_name"], r["raw_name"])
        if rx.search(r["raw_name"]) or rx.search(r["name"]):
            matched.append(r)
    if not matched:
        return {"found": False, "records": records[:20]}
    return {"found": True, **matched[0], "matched_count": len(matched)}


def cuda_device_limits(device_index: int = 0) -> Dict[str, object]:
    """driver API 只读查询；不需要创建 context，也不运行 kernel。"""
    for soname in ("libcuda.so.1", "libcuda.so"):
        try:
            lib = ctypes.CDLL(soname)
            break
        except OSError:
            lib = None
    if lib is None:
        raise RuntimeError("无法加载 libcuda.so，不能查询 SM 资源上限；可在 GPU 机器运行或给 --hw-limits-json")
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuInit.restype = ctypes.c_int
    lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.cuDeviceGet.restype = ctypes.c_int
    lib.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    lib.cuDeviceGetAttribute.restype = ctypes.c_int
    rc = lib.cuInit(0)
    if rc:
        raise RuntimeError(f"cuInit rc={rc}")
    dev = ctypes.c_int()
    rc = lib.cuDeviceGet(ctypes.byref(dev), int(device_index))
    if rc:
        raise RuntimeError(f"cuDeviceGet({device_index}) rc={rc}")
    enums = {"sm_count": 16, "max_threads_per_sm": 39,
             "max_shared_memory_per_sm": 81, "max_registers_per_sm": 82,
             "max_blocks_per_sm": 106}
    vals = {}
    for name, enum in enums.items():
        v = ctypes.c_int()
        rc = lib.cuDeviceGetAttribute(ctypes.byref(v), enum, dev)
        if rc:
            raise RuntimeError(f"cuDeviceGetAttribute {name} rc={rc}")
        vals[name] = int(v.value)
    return vals


def occupancy(res: Dict[str, object], block_threads: int, dynamic_smem: int, limits: Dict[str, object]) -> Dict[str, object]:
    regs = int(res.get("registers_per_thread") or 0)
    static_sm = int(res.get("static_smem_bytes") or 0)
    smem = int(static_sm) + int(dynamic_smem)
    # NVIDIA 寄存器分配一般是 8 regs/thread 粒度；共享内存一般 128B 粒度。
    reg_alloc = ((regs + 7) // 8) * 8
    smem_alloc = ((smem + 127) // 128) * 128
    limits_ = {
        "register": int(limits["max_registers_per_sm"]) // (reg_alloc * block_threads) if reg_alloc and block_threads else 0,
        "shared": int(limits["max_shared_memory_per_sm"]) // smem_alloc if smem_alloc else 10**9,
        "threads": int(limits["max_threads_per_sm"]) // block_threads if block_threads else 0,
        "blocks": int(limits["max_blocks_per_sm"]),
    }
    ctas = max(0, min(limits_.values())) if all(v >= 0 for v in limits_.values()) else 0
    return {"ctas_per_sm": ctas, "resource_limits_ctas": limits_, "block_threads": block_threads,
            "static_smem_bytes": static_sm, "dynamic_smem_bytes": int(dynamic_smem),
            "smem_per_block_rounded": smem_alloc, "registers_per_thread_rounded": reg_alloc}


def load_json(path: Optional[str]) -> Optional[object]:
    if not path:
        return None
    p = Path(path)
    raw = p.read_bytes()
    if p.suffix == ".gz":
        import gzip
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def is_sha256_hex(value) -> bool:
    return bool(SHA256_HEX_RE.fullmatch(str(value or "").lower()))


def res_usage_from_artifact(artifact: str, kernel: str, arch: str, timeout: int = 120) -> Tuple[Dict[str, object], str]:
    cu = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    p = run([cu, "-res-usage", artifact], timeout)
    if p.returncode != 0:
        raise RuntimeError(f"cuobjdump -res-usage rc={p.returncode}: {(p.stderr or p.stdout)[:500]}")
    return parse_res_usage(p.stdout, kernel, arch), p.stdout


def parse_dynamic_smem_from_sqlite(sqlite_path: str, kernel_rx: str) -> Dict[str, object]:
    c = sqlite3.connect(sqlite_path)
    c.text_factory = lambda b: b.decode("utf-8", errors="replace")
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        return {"found": False, "error": "CUPTI_ACTIVITY_KIND_KERNEL 表不存在"}
    cols = [r[1] for r in c.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
    wanted = [x for x in ("registersPerThread", "staticSharedMemory", "dynamicSharedMemory", "sharedMemoryExecuted",
                          "blockX", "blockY", "blockZ", "gridX", "gridY", "gridZ") if x in cols]
    rows = []
    for row in c.execute(f"SELECT k.start,k.end,{','.join('k.'+x for x in wanted)},s.value "
                         "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start"):
        name = str(row[-1])
        if re.search(kernel_rx, name):
            rows.append({"start": row[0], "end": row[1], **dict(zip(wanted, row[2:-1])), "name": name})
    if not rows:
        return {"found": False, "error": f"sqlite 中没有匹配 /{kernel_rx}/ 的 kernel"}
    return {"found": True, **rows[-1], "matched_instances": len(rows)}


# ---------------------------------------------------------------- C1

def nvidia_reg_verdict(r0: int, r1: int, strict: bool, max_abs: int, max_pct: float,
                       occ_intact: bool, spill_not_worse: bool) -> Tuple[bool, bool]:
    """NVIDIA C1 的寄存器判据，返回 (是否通过, 是否降级成 warning)。

    默认档：增幅同时满足 <=max_abs 与 <=max_pct 才算在预算内；超预算但“每 SM 驻留 CTA 不降 +
    spill 不增”时降级为 warning，因为 FiCtx 常驻 4~8 个寄存器是设计固有代价，低寄存器基线
    （<80 regs）必然让百分比超 10%（fi_bugs_h20-sk-qwen B5 实测：L0=48 -> FI=56、spill 0->0、
    CTA/SM 4->4、sim_strict=0.703，默认阈值下 C1 被误判 FAIL）。--strict-resources 关闭降级。
    """
    if strict:
        return (r0 == r1), False
    within = r0 > 0 and r1 > 0 and (r1 - r0) <= max_abs and (100.0 * (r1 - r0) / r0) <= max_pct
    if within:
        return True, False
    if r0 > 0 and r1 > 0 and occ_intact and spill_not_worse:
        return True, True
    return False, False


def spill_evidence(l0_log, fi_log, l0_res, fi_res):
    """决定 C1 的 spill 证据档位，返回 (source, spill0_bytes, spill1_bytes, note)。

    三档，档位名如实写进 metrics.spill_bytes_source，绝不拿代理冒充字节数：
      1. "ptxas -v"：两边都有 ptxas -v 文本，字节数精确；
      2. "cuobjdump -res-usage 的 STACK:0/LOCAL:0（下界证据）"：没有 ptxas 文本，但两臂的 STACK 与 LOCAL 都是 0。
         spill 必然落在 local memory，两臂都没分配 local 就等于没有 spill；字节数记 0 是下界不是实测。
         这一档是给 Triton 目标用的——cubin 天生没有 nvcc/ptxas 日志（fi_bugs_b3-sk-qwen B16：
         照 SKILL.md 的示例命令走，spill 记 unavailable、C1 必 FAIL，还给出"插桩引入了 spill"这种反事实建议）；
      3. None：证据不足，ok_spill 判 False（缺证据不判 pass）。
    """
    def _no_local(res):
        return bool(res) and int((res or {}).get("stack_bytes") or 0) == 0 and int((res or {}).get("local_bytes") or 0) == 0

    s0 = int((l0_log or {}).get("spill_store_bytes", 0) or 0) + int((l0_log or {}).get("spill_load_bytes", 0) or 0)
    s1 = int((fi_log or {}).get("spill_store_bytes", 0) or 0) + int((fi_log or {}).get("spill_load_bytes", 0) or 0)
    if (l0_log or {}).get("found") and (fi_log or {}).get("found"):
        return "ptxas -v", s0, s1, None
    if _no_local(l0_res) and _no_local(fi_res):
        return ("cuobjdump -res-usage 的 STACK:0/LOCAL:0（下界证据）", 0, 0,
                "没有 ptxas -v 日志：spill 用两臂 cuobjdump -res-usage 的 STACK:0/LOCAL:0 当下界证据"
                "（spill 必然落在 local memory，两臂都没分配 local 即无 spill），字节数记 0 是下界不是实测。"
                "要精确数字就补 --l0-build-log/--fi-build-log；Triton 目标对编译缓存里的 .ptx 跑 "
                "ptxas -v -arch=<arch> -o /dev/null 就能拿到日志。")
    return None, s0, s1, ("未提供两边 ptxas -v 日志，且 cuobjdump 的 STACK/LOCAL 不是双 0"
                          "（l0 STACK/LOCAL=%s/%s，fi=%s/%s）；spill_bytes=null，无法证明 zero spill"
                          % ((l0_res or {}).get("stack_bytes"), (l0_res or {}).get("local_bytes"),
                             (fi_res or {}).get("stack_bytes"), (fi_res or {}).get("local_bytes")))


def check_c1(a, binsim_mod) -> Dict[str, object]:
    thresholds = {
        "max_register_delta_pct": a.max_reg_delta_pct,
        "max_register_delta_abs": a.max_reg_delta_abs,
        "strict_resources_mode": bool(a.strict_resources),   # 旧名 register_delta_is_fail 严重误导：它只是 --strict-resources
                                                             # 的开关，读成"寄存器增量不判失败"会找错失败原因（fi_bugs_b3-sk-qwen B18）
        "strict_resource_equality": bool(a.strict_resources),
        "register_criterion": "默认档：delta 同时 <= max_register_delta_abs 与 <= max_register_delta_pct 算预算内；"
                              "超预算但每 SM 驻留 CTA 不降且 spill 不增时降级为 warning（降级需要 ptxas -v 日志，"
                              "即 --l0-build-log/--fi-build-log，缺证据不降级）；--strict-resources 改成要求两臂寄存器数完全相等且关闭降级。"
                              "具体哪一项没过看 metrics.failed_subchecks",
        "spill_delta_max_bytes": 0,
        "smem_delta_bytes": 0,
        "min_ctas_per_sm_delta": 0,
        "min_sim_strict": a.min_sim_strict,
        "min_sim_opcode": a.min_sim_opcode,
        "max_basic_block_delta_pct": a.max_bb_delta_pct,
        "min_probe_event_delta": 1,
        "max_probe_events": a.max_probe_events,
        "require_key_instruction_order": True,
        "register_delta_downgrade_rule": "寄存器增幅超预算，但每 SM 驻留 CTA 不降且 spill 不增时记 warning 不判失败；"
                                         "--strict-resources 关闭该降级",
    }
    evidence: Dict[str, object] = {}
    suggestions: List[str] = []
    warnings: List[str] = []
    metrics: Dict[str, object] = {}
    errors = []
    try:
        l0_res = (parse_res_usage(Path(a.l0_resusage).read_text(encoding="utf-8", errors="replace"), a.kernel, a.arch)
                  if a.l0_resusage else res_usage_from_artifact(a.l0, a.kernel, a.arch)[0])
        fi_res = (parse_res_usage(Path(a.fi_resusage).read_text(encoding="utf-8", errors="replace"), a.kernel, a.arch)
                  if a.fi_resusage else res_usage_from_artifact(a.fi, a.kernel, a.arch)[0])
        evidence["resource_usage"] = {"l0": l0_res, "fi": fi_res}
    except Exception as e:
        errors.append(f"资源读取失败: {e}")
        l0_res, fi_res = {}, {}
    l0_log = parse_build_log(Path(a.l0_build_log).read_text(encoding="utf-8", errors="replace"), a.kernel) if a.l0_build_log and Path(a.l0_build_log).exists() else {"found": False}
    fi_log = parse_build_log(Path(a.fi_build_log).read_text(encoding="utf-8", errors="replace"), a.kernel) if a.fi_build_log and Path(a.fi_build_log).exists() else {"found": False}
    evidence["build_logs"] = {"l0": l0_log, "fi": fi_log}
    # SQLite 的 launch 资源优先用于 dynamic smem；它记录的是实际启动值。
    sqlite_res = parse_dynamic_smem_from_sqlite(a.sqlite, a.nsys_kernel or a.kernel) if a.sqlite and Path(a.sqlite).exists() else {"found": False}
    evidence["nsys_launch_resources"] = sqlite_res
    dyn_l0 = int(a.l0_dynamic_smem if a.l0_dynamic_smem is not None else (sqlite_res.get("dynamicSharedMemory", a.dynamic_smem) if sqlite_res.get("found") else a.dynamic_smem))
    dyn_fi = int(a.fi_dynamic_smem if a.fi_dynamic_smem is not None else (sqlite_res.get("dynamicSharedMemory", a.dynamic_smem) if sqlite_res.get("found") else a.dynamic_smem))
    try:
        limits = load_json(a.hw_limits_json) if a.hw_limits_json else None
        if not isinstance(limits, dict):
            limits = cuda_device_limits(a.cuda_device)
        evidence["hardware_limits"] = limits
    except Exception as e:
        errors.append(f"硬件上限查询失败: {e}")
        limits = None
    occ0 = occupancy(l0_res, a.block_threads, dyn_l0, limits) if limits and l0_res else {}
    occ1 = occupancy(fi_res, a.block_threads, dyn_fi, limits) if limits and fi_res else {}
    evidence["occupancy"] = {"l0": occ0, "fi": occ1}
    try:
        bs = binsim_mod.analyze(a.l0, a.fi, a.kernel, a.arch, str(Path(a.work_dir) / "binsim"))
        evidence["binsim"] = bs
    except Exception as e:
        errors.append(f"fi_binsim 失败: {e}")
        bs = {}
    # 值表与变化
    def val(d, k, default=0):
        v = d.get(k)
        return int(v) if v is not None else int(default)
    r0, r1 = val(l0_res, "registers_per_thread"), val(fi_res, "registers_per_thread")
    spill_source, spill0, spill1, spill_note = spill_evidence(l0_log, fi_log, l0_res, fi_res)
    if spill_source is not None and spill_source != "ptxas -v":
        warnings.append(spill_note)          # 用了下界证据就说清楚，别让人以为字节数是实测的
    if spill_source is None:
        # 缺证据时把 SASS 里的 STL/LDL 条数当下限代理写进 evidence（只是条数，不冒充字节 spill）
        ev = bs.get("probes", {})
        evidence["spill_local_ops_proxy"] = {
            side: sum(1 for x in (ev.get(side) or {}).get("fi_buffer_stores", [])
                      if str(x.get("opcode", "")).startswith(("STL", "LDL"))) for side in ("l0", "fi")}
        evidence["spill_proxy_note"] = spill_note
    metrics["spill_bytes_source"] = spill_source or "unavailable"
    reg_delta_abs = r1 - r0
    reg_delta_pct = (100.0 * reg_delta_abs / r0) if r0 else 0.0
    s0 = val(l0_res, "static_smem_bytes") + dyn_l0
    s1 = val(fi_res, "static_smem_bytes") + dyn_fi
    c0 = int((occ0 or {}).get("ctas_per_sm", 0))
    c1 = int((occ1 or {}).get("ctas_per_sm", 0))
    metrics.update({
        "registers_per_thread": {"l0": r0 or None, "fi": r1 or None, "delta": reg_delta_abs, "delta_pct": round(reg_delta_pct, 3)},
        "spill_total_bytes": {"l0": spill0 if spill_source else None, "fi": spill1 if spill_source else None,
                              "delta": (spill1 - spill0) if spill_source else None},
        "smem_bytes": {"l0": s0 or None, "fi": s1 or None, "delta": s1 - s0},
        "ctas_per_sm": {"l0": c0 or None, "fi": c1 or None, "delta": c1 - c0},
        "exact_equal": {"registers": r0 == r1, "spill": bool(spill_source) and spill0 == spill1,
                        "smem": s0 == s1, "ctas_per_sm": c0 == c1},
        "sim_strict": bs.get("sim_strict"), "sim_opcode": bs.get("sim_opcode"),
        "basic_blocks": bs.get("basic_blocks"), "probe_event_delta": bs.get("probes", {}).get("delta", {}).get("probe_events"),
        "probe_events_l0": bs.get("probes", {}).get("l0", {}).get("counts", {}).get("probe_events"),
        "probe_events_fi": bs.get("probes", {}).get("fi", {}).get("counts", {}).get("probe_events"),
        "key_instruction_order_preserved": bs.get("key_instruction_order_preserved"),
    })
    ok_spill = bool(spill_source) and spill1 <= spill0
    ok_smem = bool(l0_res and fi_res) and s1 == s0
    ok_occ = bool(occ0 and occ1) and (c1 == c0 if a.strict_resources else c1 >= c0 and c1 > 0)
    ok_reg, reg_downgraded = nvidia_reg_verdict(r0, r1, bool(a.strict_resources), a.max_reg_delta_abs,
                                                a.max_reg_delta_pct, bool(ok_occ), bool(ok_spill))
    if reg_downgraded:
        # 超预算的寄存器增长若没有压低占用率、也没有引入 spill，就只是 FiCtx 的固有代价：
        # 判 FAIL 会把正确插桩误报成资源扰动（AMD 分支早就这么处理，NVIDIA 分支此前没同步）。
        warnings.append(f"寄存器 {r0}->{r1}（+{reg_delta_abs}，{round(reg_delta_pct, 2)}%）超过预算"
                        f"（+{a.max_reg_delta_abs} 且 {a.max_reg_delta_pct}%），但每 SM 驻留 CTA {c0}->{c1} 未降、"
                        f"spill {spill0}->{spill1} 未增：插桩固有代价，记 warning 不判失败")
    ok_sim = as_float(bs.get("sim_strict")) is not None and bs.get("sim_strict", 0) >= a.min_sim_strict
    ok_op = as_float(bs.get("sim_opcode")) is not None and bs.get("sim_opcode", 0) >= a.min_sim_opcode
    bb_pct = bs.get("basic_blocks", {}).get("delta_pct")
    ok_bb = bb_pct is not None and bb_pct <= a.max_bb_delta_pct
    probe_delta = metrics["probe_event_delta"]
    probe_fi = metrics["probe_events_fi"]
    ok_probe = probe_delta is not None and probe_delta >= 1
    if a.max_probe_events is not None:
        ok_probe = ok_probe and probe_fi is not None and int(probe_fi) <= int(a.max_probe_events)
    ok_key = bs.get("key_instruction_order_preserved") is True
    metrics["register_delta_downgraded_to_warning"] = reg_downgraded
    metrics["warnings"] = warnings
    # 逐项写明"哪个子判据、什么数、什么阈值"没过：以前只能读 ok_reg=False 再去翻源码反推是 9>8 还是 5.5%>10%
    # （fi_bugs_b3-sk-qwen B18：T2 的 C1 只因绝对增量 9>8 挂，百分比 5.521%<10% 是过的）。
    subchecks = [
        ("ok_reg", ok_reg, f"registers_per_thread {r0}->{r1}（+{reg_delta_abs}，{round(reg_delta_pct, 3)}%），"
                           f"预算 +{a.max_reg_delta_abs} 且 {a.max_reg_delta_pct}%，strict={bool(a.strict_resources)}，"
                           f"降级为 warning={reg_downgraded}"),
        ("ok_spill", ok_spill, f"spill_total_bytes {spill0 if l0_log.get('found') else 'n/a'}->"
                               f"{spill1 if fi_log.get('found') else 'n/a'}（证据源 {metrics['spill_bytes_source']}）"),
        ("ok_smem", ok_smem, f"smem_bytes {s0}->{s1}（要求相等）"),
        ("ok_occ", ok_occ, f"ctas_per_sm {c0}->{c1}（要求不降）"),
        ("ok_sim", ok_sim, f"sim_strict {bs.get('sim_strict')}（阈值 >= {a.min_sim_strict}）"),
        ("ok_op", ok_op, f"sim_opcode {bs.get('sim_opcode')}（阈值 >= {a.min_sim_opcode}）"),
        ("ok_bb", ok_bb, f"basic_blocks delta_pct {bb_pct}（阈值 <= {a.max_bb_delta_pct}）"),
        ("ok_probe", ok_probe, f"probe_event_delta {probe_delta}（阈值 >= 1）"),
        ("ok_key", ok_key, f"key_instruction_order_preserved {bs.get('key_instruction_order_preserved')}（要求 true）"),
    ]
    metrics["failed_subchecks"] = [f"{name}: {why}" for name, okk, why in subchecks if not okk]
    passed = not errors and all(okk for _n, okk, _w in subchecks)
    if not ok_reg:
        suggestions.append(
            f"寄存器 {r0}->{r1}（+{reg_delta_abs}，{round(reg_delta_pct, 2)}%）没过 C1 的寄存器判据，先分清是哪种情况："
            "(a) 没给 ptxas -v 日志：降级条件里的 spill 证据缺失，补 --l0-build-log/--fi-build-log 重跑再看；"
            "(b) 每 SM 驻留 CTA 下降或 spill 增加：这是真实资源扰动，按下面减寄存器；"
            f"(c) 目标本身寄存器基数高（例如 163），+9 这类增量在不降占用率、零 spill、实测开销在噪声内时属插桩固有代价，"
            "把 --max-reg-delta-abs 调到能容纳它并在报告里写明理由与四条证据（两臂寄存器数、ctas_per_sm、spill、同口径中位耗时）；"
            "正确性类阈值（--min-sim-strict/--min-sim-opcode）不要跟着放宽。"
            "真要减寄存器：FiCtx 只保留缓冲基址与计数，指针加法用 64-bit 地址寄存器一次完成，或提高 __launch_bounds__ 的 minBlocks。"
            "FI_GRAN=1 也能省寄存器，但它把打点粒度从 warp 换成 SM/CTA——任务要求 warp 粒度时不能用，别把它当唯一退路。")
    if not ok_spill:
        if not (l0_log.get("found") and fi_log.get("found")):
            suggestions.append("缺 ptxas -v 日志，且 cuobjdump 的 STACK/LOCAL 不是双 0，无法证明 zero spill："
                               "CUDA 两臂都用 -Xptxas -v 重编、Triton 对编译缓存里的 .ptx 跑 "
                               "ptxas -v -arch=<arch> -o /dev/null，把日志用 --l0-build-log/--fi-build-log 传进来"
                               "（缺证据不判 pass）。")
        else:
            suggestions.append("插桩后出现/增加 spill：把每 warp 上下文压到 <=3 个寄存器，关闭高频 FI_BG(1)，"
                               "或把循环内事件合并到每个 tile 外层。")
    if not ok_smem:
        suggestions.append("插桩改变了 smem 占用：检查是否把缓冲或状态放进 static __shared__；FI 记录必须写 global buffer。")
    if not ok_occ:
        suggestions.append("每 SM 驻留 CTA 数下降：按占用率约束重编插桩版本，优先降低寄存器/动态 smem；否则该构建不能与 L0 做同口径性能比较。")
    if not ok_sim:
        suggestions.append("SASS 语义相似度低：检查是否插入/删除了业务指令；只应在原有数据流旁增加计时器读和 16B STG，不能替换 MMA/同步。")
    if not ok_op:
        suggestions.append("opcode 相似度低：定位 basic block 中新增的 opcode；把循环内探针移到 FI_BG(1) 并默认关闭，避免分支和地址计算扩张。")
    if not ok_bb:
        suggestions.append("基本块增长过大：把越界检查移出热路径，避免每个事件都产生独立 lane!=0/overflow 分支。")
    if not ok_probe:
        if (a.max_probe_events is not None and probe_fi is not None
                and int(probe_fi) > int(a.max_probe_events)):
            suggestions.append(f"FI probe 数 {probe_fi} 超过预算 {a.max_probe_events}：删除循环内重复探针，把高频事件合并到 FI_BG(1) 或移到 tile/chunk 外层。")
        else:
            suggestions.append("没有检测到新增 FI probe：确认编译了 FI_LEVEL>=2 且 SASS 中确有 SR_GLOBALTIMER/SR_CLOCK/SR_SMID 读和 STG.128；检查 kernel 正则是否选对。")
    if not ok_key:
        suggestions.append("关键指令相对顺序改变：对照 key_instructions_by_class，恢复原 HGMMA/UTC*MMA/UTMALDG/UTMASTG/SYNCS/BAR/LDGSTS/DEPBAR 顺序，只把探针放在这些边界之外。")
    if errors:
        suggestions.append("先修复证据读取错误：" + "；".join(errors))
    return item(passed, metrics, thresholds, evidence, suggestions,
                note="资源严格相等布尔值始终记录；默认 pass 允许不改变占用率的小幅寄存器增长（超预算记 warning），"
                     "--strict-resources 可改为逐项严格相等。"
                     + ("warnings: " + " | ".join(warnings) if warnings else ""))


# ---------------------------------------------------------------- AMD (gfx) C1

AMD_LLVM_CANDIDATES = ("/opt/rocm/lib/llvm/bin", "/opt/rocm-7.2.0/lib/llvm/bin", "/opt/rocm-6.4.0/lib/llvm/bin")
AMD_NOTE_LIST_RE = re.compile(r"^  - (\.[A-Za-z0-9_]+):\s*(.*?)\s*$")
AMD_NOTE_PROP_RE = re.compile(r"^    (\.[A-Za-z0-9_]+):\s*(.*?)\s*$")
AMD_KERNEL_HDR_RE = re.compile(r"^[0-9a-fA-F]+ <([^>]+)>:")
AMD_INSN_RE = re.compile(r"^\s*([A-Za-z0-9_.]+)\b")
# 指令类：类名 -> 匹配反汇编 opcode 的正则（匹配对象是 AMD_INSN_RE 抓到的 opcode 本身，不含操作数）。
# FI 的记录写有**两种**签名，取决于用哪个前端：
#   2_instrument/recorders/hip_cpp/fi_lite_hip.h         -> 16B AoS 一条：global_store_dwordx3/x4（record_store）
#   2_instrument/recorders/triton/fi_triton_amd.py -> SoA 三条 tl.store：时间戳 global_store_dwordx2、
#                                    tag 与 aux 各一条 global_store_dword（record_store_soa_*）
# 只认 AoS 会让 Triton 路线必然多一条“找不到 FI 探针”的结构性 FAIL，而报错还把用户指向
# FI_LEVEL/kernel 正则这种错方向（fi_bugs_amd-sk-qwen B17：实测 base 32 条 global_store_short，
# 插桩后 33 条 global_store_dword + 32 条 short + 12 条 dwordx2、s_memrealtime 读 +13）。
AMD_INSN_CLASSES = {
    "barrier": r"^s_barrier",
    "waitcnt": r"^s_waitcnt",
    "sleep": r"^s_sleep",
    "timer_read": r"^s_mem(realtime|time)$",
    "record_store": r"^(global|buffer|flat)_store_dwordx[34]",
    "record_store_soa_time": r"^(global|buffer|flat)_store_dwordx2$",
    "record_store_soa_word": r"^(global|buffer|flat)_store_dword$",
    "global_load": r"^global_load",
    "lds": r"^ds_(read|write)",
    "mfma": r"^v_mfma_",
    "atomic": r"^(global|buffer|flat|ds)_atomic",
    "cache_flush": r"^(buffer_wbl2|buffer_wbinvl1|s_dcache_wb|s_icache_inv)",
}
# 记录写的三种签名（AoS 一条 + SoA 两条）。SoA 的宽度业务代码里也会出现，所以判定上
# s_memrealtime/s_memtime 读的增量才是强信号，记录写只要求“任一种签名有增量”。
AMD_RECORD_STORE_CLASSES = ("record_store", "record_store_soa_time", "record_store_soa_word")
AMD_RES_KEYS = ("vgpr", "sgpr", "agpr", "scratch", "lds", "vgpr_spill_count", "sgpr_spill_count")


def amd_tools(rocm_llvm: Optional[str] = None) -> Dict[str, object]:
    """解析 AMD 侧需要的 LLVM 工具；缺失项写清缺什么，不静默降级。"""
    base = rocm_llvm or os.environ.get("FI_ROCM_LLVM")
    if not base:
        for c in AMD_LLVM_CANDIDATES:
            if Path(c).is_dir():
                base = c
                break
    base = base or "/opt/rocm/lib/llvm/bin"
    out = {"rocm_llvm": base, "objcopy": shutil.which("objcopy") or "objcopy",
           "file": shutil.which("file") or "file"}
    for name in ("clang-offload-bundler", "llvm-readelf", "llvm-objdump"):
        out[name] = str(Path(base) / name) if Path(base, name).exists() else (shutil.which(name) or "")
    out["missing"] = sorted(k for k in ("clang-offload-bundler", "llvm-readelf", "llvm-objdump") if not out[k])
    return out


def _amd_value(v: str):
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def amd_run(argv: Sequence[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return run(argv, timeout)


EM_X86_64 = 0x3E
EM_AMDGPU = 0xE0          # /usr/include/elf.h：“#define EM_AMDGPU 224 /* AMD GPU */”


def elf_machine_of(path) -> Tuple[bool, Optional[int]]:
    """读 ELF 头的 e_machine，返回 (是否 ELF, machine)。不依赖系统 file(1)。"""
    try:
        with open(path, "rb") as f:
            head = f.read(20)
    except OSError:
        return False, None
    if len(head) < 20 or head[:4] != b"\x7fELF":
        return False, None
    endian = "little" if head[5] == 1 else "big"
    return True, int.from_bytes(head[18:20], endian)


def amd_extract_code_object(path: str, workdir: str, gfx: str, tools: Dict[str, object]) -> Tuple[Path, str, List[str]]:
    """从 HIP 可执行文件/目标文件里抽出 gfx* 的 code object；已经是 AMDGPU ELF 时直接用。"""
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"文件不存在: {p}")
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    # 判据用 ELF 头的 e_machine，不用系统 file(1)：mail-14 上 file 对真实 gfx942 code object 报
    # “ELF 64-bit LSB shared object, unknown arch 0xe0”，按它判定会把真器件 ELF 当成
    # “既不是 AMDGPU ELF 也没有 .hip_fatbin”，C1 直接因证据不足失败（fi_bugs_amd-sk-ds B22）。
    is_elf, machine = elf_machine_of(p)
    readelf = str(tools.get("llvm-readelf") or "")
    head_txt = ""
    if tools.get("file"):
        head_txt = (amd_run([str(tools["file"]), "-b", str(p)], 30).stdout or "").strip()
    if is_elf and machine == EM_AMDGPU:
        notes_txt = ""
        if readelf:
            notes_txt = (amd_run([readelf, "--notes", str(p)], 60).stdout or "").lower()
        if not gfx or gfx in notes_txt:
            return p, "direct-amdgpu-elf", []
        # e_machine 对得上但 notes 里没写这个 gfx：仍按器件 ELF 用，把不一致如实记进过程记录
        return p, "direct-amdgpu-elf", [f"llvm-readelf --notes 里没有 {gfx}（e_machine=0x{machine:x} 是 AMDGPU，仍按器件 ELF 反汇编）"]
    sections = amd_run(["readelf", "-W", "-S", str(p)], 30)
    if ".hip_fatbin" not in (sections.stdout or ""):
        kind = f"e_machine=0x{machine:x}" if is_elf else "不是 ELF"
        raise RuntimeError(f"{p}: 既不是 AMDGPU ELF（EM_AMDGPU=0xe0）也没有 .hip_fatbin，无法抽 code object"
                           f"（{kind}；file 输出：{head_txt[:200] or '（无 file 命令）'}）")
    fat = wd / "hip_fatbin.bin"
    amd_run([str(tools["objcopy"]), "--dump-section", f".hip_fatbin={fat}", str(p), str(wd / "host_copy.o")], 60)
    bundler = str(tools["clang-offload-bundler"])
    listing = amd_run([bundler, "--type=o", "--list", f"--input={fat}"], 30)
    targets = [x.strip() for x in (listing.stdout or "").splitlines() if x.strip()]
    matched = [x for x in targets if x.endswith(f"--{gfx}")]
    if not matched:
        raise RuntimeError(f"{p}: fatbin 里没有 {gfx} target；实际 targets={targets}")
    obj = wd / f"{gfx}.o"
    amd_run([bundler, "--type=o", "--unbundle", f"--input={fat}", f"--output={obj}", f"--targets={matched[0]}"], 60)
    return obj, matched[0], targets


def amd_parse_notes(obj: Path, tools: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    """llvm-readelf --notes 里的 kernel metadata（VGPR/SGPR/AGPR/scratch/LDS/spill）。"""
    p = amd_run([str(tools["llvm-readelf"]), "--notes", str(obj)], 60)
    if p.returncode != 0:
        raise RuntimeError(f"llvm-readelf --notes rc={p.returncode}: {(p.stderr or p.stdout)[:300]}")
    kernels: Dict[str, Dict[str, object]] = {}
    cur: Optional[Dict[str, object]] = None
    for line in (p.stdout or "").splitlines():
        m = AMD_NOTE_LIST_RE.match(line)
        if m:
            if cur is not None:
                kernels[str(cur.get(".name") or cur.get(".symbol") or "<unnamed>")] = cur
            cur = {m.group(1): _amd_value(m.group(2))}
            continue
        m = AMD_NOTE_PROP_RE.match(line)
        if m and cur is not None:
            cur[m.group(1)] = _amd_value(m.group(2))
    if cur is not None:
        kernels[str(cur.get(".name") or cur.get(".symbol") or "<unnamed>")] = cur
    if not kernels:
        raise RuntimeError(f"{obj}: llvm-readelf --notes 里没解析到 kernel metadata")
    return kernels


def amd_parse_disasm(obj: Path, gfx: str, tools: Dict[str, object]) -> Dict[str, Dict[str, int]]:
    """llvm-objdump -d 的指令类计数（barrier/waitcnt/timer_read/record_store/...）。"""
    p = amd_run([str(tools["llvm-objdump"]), "-d", f"--mcpu={gfx}", str(obj)], 180)
    if p.returncode != 0:
        raise RuntimeError(f"llvm-objdump rc={p.returncode}: {(p.stderr or p.stdout)[:300]}")
    rx = {k: re.compile(v) for k, v in AMD_INSN_CLASSES.items()}
    out: Dict[str, Dict[str, int]] = {}
    cur = None
    for line in (p.stdout or "").splitlines():
        m = AMD_KERNEL_HDR_RE.match(line)
        if m:
            cur = m.group(1)
            out[cur] = {k: 0 for k in rx}
            out[cur]["_insns"] = 0
            continue
        if cur is None:
            continue
        m = AMD_INSN_RE.match(line)
        if not m:
            continue
        op = m.group(1)
        out[cur]["_insns"] += 1
        for k, r in rx.items():
            if r.search(op):
                out[cur][k] += 1
                break
    return out


def amd_gather(path: str, workdir: str, gfx: str, tools: Dict[str, object],
               res_json: Optional[str] = None) -> Dict[str, object]:
    """一个产物的 AMD 资源画像。给 --*-amd-res 时直接读 JSON（离线回归/无 ROCm 机器）。

    JSON schema：{"gfx": "gfx942", "source": "<原始路径>", "kernels": {"<kernel 名>": {
        "vgpr": int, "sgpr": int, "agpr": int, "scratch": int, "lds": int,
        "vgpr_spill_count": int, "sgpr_spill_count": int, "instructions": {类名: int}}}}
    """
    if res_json:
        d = load_json(res_json)
        if not isinstance(d, dict) or "kernels" not in d:
            raise RuntimeError(f"{res_json}: 不是 AMD 资源 JSON（需要 gfx/source/kernels 三个键）")
        d.setdefault("gfx", gfx)
        d["provenance"] = f"pre-extracted JSON: {res_json}"
        return d
    if tools.get("missing"):
        raise RuntimeError("缺 AMD LLVM 工具：" + ",".join(tools["missing"]) +
                           f"（预期在 {tools['rocm_llvm']}）；在 AMD 机器上跑，或用 --l0-amd-res/--fi-amd-res 提供预先抽好的资源 JSON")
    obj, target, targets = amd_extract_code_object(path, workdir, gfx, tools)
    notes = amd_parse_notes(obj, tools)
    disasm = amd_parse_disasm(obj, gfx, tools)
    kernels = {}
    for name, meta in notes.items():
        insn = disasm.get(name) or disasm.get(f"{name}.kd") or {}
        kernels[name] = {
            "vgpr": int(meta.get(".vgpr_count", -1)), "sgpr": int(meta.get(".sgpr_count", -1)),
            "agpr": int(meta.get(".agpr_count", 0) or 0),
            "scratch": int(meta.get(".private_segment_fixed_size", -1)),
            "lds": int(meta.get(".group_segment_fixed_size", -1)),
            "vgpr_spill_count": int(meta.get(".vgpr_spill_count", 0) or 0),
            "sgpr_spill_count": int(meta.get(".sgpr_spill_count", 0) or 0),
            "instructions": {k: v for k, v in insn.items()},
            "target": target,
        }
    return {"gfx": gfx, "source": str(path), "targets": targets, "kernels": kernels,
            "code_object": str(obj),
            "provenance": f"llvm-readelf/llvm-objdump on {obj}"}


def amd_device_limits(gfx: str) -> Dict[str, object]:
    """从包内 knowledge/<gfx>.json 读器件上限（LDS/CU、SIMD/CU、wavefront）；读不到就用 gfx942 的实测值。"""
    defaults = {"lds_per_cu": 65536, "simds_per_cu": 4, "wavefront_size": 64,
                "max_waves_per_cu": 32, "max_waves_per_simd": 8, "source": "内置默认（gfx942 实测）"}
    f = PKG / "kb" / f"{gfx}.json"
    if not f.is_file():
        defaults["source"] = f"内置默认（包内没有 knowledge/{gfx}.json）"
        return defaults
    try:
        d = load_json(f)
        dev = d.get("device") or {}
        props = dev.get("hip_device_props") or {}
        out = dict(defaults)
        out["lds_per_cu"] = int(props.get("shared_mem_per_cu") or dev.get("lds_per_cu") or defaults["lds_per_cu"])
        out["simds_per_cu"] = int(dev.get("simds_per_cu") or defaults["simds_per_cu"])
        out["wavefront_size"] = int(dev.get("wavefront_size") or props.get("wavefront_size")
                                    or defaults["wavefront_size"])
        out["max_waves_per_cu"] = int(dev.get("max_waves_per_cu") or defaults["max_waves_per_cu"])
        out["max_waves_per_simd"] = max(1, out["max_waves_per_cu"] // max(1, out["simds_per_cu"]))
        threads = int(props.get("max_threads_per_cu") or 0)
        if threads and out["wavefront_size"]:
            # 线程数上限折算的 wave/SIMD 再取一次小（gfx942：2048/64/4 = 8，与 max_waves_per_cu 口径一致）
            out["max_waves_per_simd"] = max(1, min(out["max_waves_per_simd"],
                                                  threads // out["wavefront_size"] // max(1, out["simds_per_cu"])))
        out["source"] = (f"knowledge/{f.name}: device.max_waves_per_cu / device.simds_per_cu / device.wavefront_size"
                         " / hip_device_props.shared_mem_per_cu / hip_device_props.max_threads_per_cu")
        return out
    except Exception as e:  # noqa: BLE001 - kb 坏了不能让整个检查器退出
        defaults["source"] = f"内置默认（读 knowledge/{gfx}.json 失败：{e}）"
        return defaults


def amd_lds_waves_per_simd(static_lds: int, dynamic_lds: int, waves_per_wg: int,
                           lds_per_cu: int, simds_per_cu: int,
                           max_waves_per_simd: Optional[int] = None) -> Optional[float]:
    """LDS 上限折算成“每 SIMD 可驻留 wave 数”；两边 LDS 都为 0 时返回 None（这一项不起作用）。
    给了 max_waves_per_simd 就与硬件并发上限取小：LDS 很小时 ctas_per_cu 会大到离谱（几百 B 的静态标志
    能算出上千 wave/SIMD），必须截断。返回 0.0 表示 LDS 超配，截断前先判，不能被 clamp 掉。"""
    per_wg = max(0, int(static_lds)) + max(0, int(dynamic_lds))
    if per_wg <= 0 or lds_per_cu <= 0 or simds_per_cu <= 0:
        return None
    ctas = lds_per_cu // per_wg
    if ctas <= 0:
        return 0.0
    w = ctas * waves_per_wg / simds_per_cu
    if max_waves_per_simd and max_waves_per_simd > 0:
        w = min(w, float(max_waves_per_simd))
    return w


def amd_waves_per_simd(vgpr: int, agpr: int = 0, vgpr_file: int = 512,
                       max_waves_per_simd: Optional[int] = None) -> int:
    """gfx942 每 SIMD 512 个 VGPR，按 8 个一组分配；AGPR>0 时与 AGPR 侧取小。
    max_waves_per_simd 是硬件并发上限（gfx942 = 8），给了就一起取小：寄存器文件能装下 64 个 wave，
    硬件一个 SIMD 上也只跑得下 8 个（fi_bugs_amd-sk-qwen B15）。不传就是“只看寄存器文件”的口径。"""
    if vgpr is None or vgpr < 0:
        return -1
    alloc = max(8, ((int(vgpr) + 7) // 8) * 8)
    w = vgpr_file // alloc
    if agpr and agpr > 0:
        alloc_a = max(8, ((int(agpr) + 7) // 8) * 8)
        w = min(w, vgpr_file // alloc_a)
    if max_waves_per_simd and max_waves_per_simd > 0:
        w = min(w, int(max_waves_per_simd))
    return w


def amd_pick_kernel(doc: Dict[str, object], kernel_rx: str) -> Tuple[str, Dict[str, object]]:
    rx = re.compile(kernel_rx)
    ks = doc.get("kernels") or {}
    hit = [k for k in ks if rx.search(k)]
    if not hit:
        raise RuntimeError(f"{doc.get('source')}: 没有匹配 /{kernel_rx}/ 的 kernel；实际有 {sorted(ks)[:8]}")
    hit.sort(key=lambda k: (len(k), k))
    return hit[0], ks[hit[0]]


def check_c1_amd(a, binsim_hip_mod=None) -> Dict[str, object]:
    """AMD 的 C1：结构性扰动判 FAIL，插桩固有代价记 warning（阈值全部写进 JSON）。"""
    require_fi_probe = not bool(getattr(a, "allow_unverified_probe", False))
    thresholds = {
        "platform": "amd", "gfx": a.amd_gfx,
        "max_scratch_delta_bytes": a.max_scratch_delta_bytes,
        "min_occupancy_ratio": a.min_occupancy_ratio,
        "max_register_delta_pct": a.max_reg_delta_pct,
        "max_register_delta_abs": a.max_reg_delta_abs,
        "strict_resource_equality": bool(a.strict_resources),
        "require_fi_probe": require_fi_probe,
        "probe_waiver": bool(getattr(a, "allow_unverified_probe", False)),
        "min_sim_strict": a.min_sim_strict, "min_sim_opcode": a.min_sim_opcode,
        "min_sim_norm": a.min_sim_norm, "min_key_lcs": a.min_key_lcs,
        "similarity_is_fail": bool(a.strict_resources),
        "similarity_policy": "AMD 上 sim_strict/sim_opcode/sim_norm 量的是“指令流长了多少”，而插桩本来就要加"
                             "计时器读 + 16B 记录写，低于阈值记 warning 并把数字报出来；FAIL 只认能直接证明"
                             "业务代码被改坏的证据：probe_removed<1、key_lcs（同义组口径）< --min-key-lcs、"
                             "v_mfma 静态条数减少。--strict-resources 恢复按字面阈值判 FAIL。",
        "occupancy_formula": "寄存器上限：allocated_vgpr=max(8,ceil(vgpr/8)*8); waves_per_simd=floor(512/allocated_vgpr)"
                             "; agpr>0 时再与 floor(512/allocated_agpr) 取小；最后与硬件并发上限 "
                             "max_waves_per_simd=max_waves_per_cu/simds_per_cu 取小（gfx942=8）。"
                             "LDS 上限：ctas_per_cu=floor(lds_per_cu/(静态 LDS+动态 LDS)); "
                             "waves_per_simd_lds=ctas_per_cu*ceil(block_threads/wavefront_size)/simds_per_cu，"
                             "同样按 max_waves_per_simd 截断。"
                             "判定用两者取小后的 effective 值（器件上限来自包内 knowledge/<gfx>.json，可用 --amd-* 覆盖）",
        "fail_conditions": ["kernel 缺失", "s_barrier 条数变化", "静态 LDS 变化",
                            f"scratch 增量 > {a.max_scratch_delta_bytes} B/work-item",
                            f"waves/SIMD < {a.min_occupancy_ratio} × 基线", "找不到 FI 探针",
                            "剔除的探针指令数 < 1", f"key_lcs（同义组）< {a.min_key_lcs}",
                            "v_mfma 静态条数减少"]
                            + (["相似度低于阈值（--strict-resources）"] if a.strict_resources else []),
        "warning_conditions": ["VGPR/SGPR 增长", "waves/SIMD 在预算内下降", "出现 spill", "s_waitcnt 增多",
                               "sim_strict/sim_opcode/sim_norm 低于阈值", "关键指令助记符精确顺序变化"],
    }
    evidence: Dict[str, object] = {}
    suggestions: List[str] = []
    warnings: List[str] = []
    metrics: Dict[str, object] = {}
    errors: List[str] = []
    tools = amd_tools(a.rocm_llvm)
    evidence["tools"] = tools
    l0_doc = fi_doc = {}
    for side, path, res_json in (("l0", a.l0, a.l0_amd_res), ("fi", a.fi, a.fi_amd_res)):
        if not path and not res_json:
            errors.append(f"AMD C1 需要 --{side} 或 --{side}-amd-res")
            continue
        try:
            doc = amd_gather(path or res_json, str(Path(a.work_dir) / f"amd_{side}"), a.amd_gfx, tools, res_json)
            evidence[f"{side}_resources"] = doc
            if side == "l0":
                l0_doc = doc
            else:
                fi_doc = doc
        except Exception as e:
            errors.append(f"{side} 侧 AMD 资源读取失败: {e}")
    k0 = k1 = None
    r0: Dict[str, object] = {}
    r1: Dict[str, object] = {}
    if l0_doc and fi_doc:
        try:
            k0, r0 = amd_pick_kernel(l0_doc, a.kernel)
            k1, r1 = amd_pick_kernel(fi_doc, a.kernel)
            evidence["kernel_match"] = {"l0": k0, "fi": k1}
        except Exception as e:
            errors.append(str(e))
    if errors:
        suggestions.append("先补齐 AMD C1 的证据：" + "；".join(errors))
        return item(False, metrics, thresholds, evidence, suggestions, note="证据不足，未判定。")

    def num(d, k, default=0):
        v = d.get(k)
        return int(v) if isinstance(v, int) or (v is not None and str(v).lstrip("-").isdigit()) else int(default)

    i0 = r0.get("instructions") or {}
    i1 = r1.get("instructions") or {}
    vgpr0, vgpr1 = num(r0, "vgpr", -1), num(r1, "vgpr", -1)
    agpr0, agpr1 = num(r0, "agpr", 0), num(r1, "agpr", 0)
    dev = amd_device_limits(a.amd_gfx)
    lds_per_cu = int(getattr(a, "amd_lds_per_cu", None) or dev["lds_per_cu"])
    simds_per_cu = int(getattr(a, "amd_simds_per_cu", None) or dev["simds_per_cu"])
    wavefront = int(getattr(a, "amd_wavefront_size", None) or dev["wavefront_size"])
    max_wps = int(getattr(a, "amd_max_waves_per_simd", None) or dev["max_waves_per_simd"])
    # 寄存器文件只是上限之一：硬件并发 wave 数另有上限（gfx942 = 32 wave/CU ÷ 4 SIMD = 8）。
    # 不截断就会算出 21、64 这种硬件上不可能的数字，再把两臂其实相同的占用率判成“崩塌”（B15）。
    w0_file = amd_waves_per_simd(vgpr0, agpr0)
    w1_file = amd_waves_per_simd(vgpr1, agpr1)
    w0 = amd_waves_per_simd(vgpr0, agpr0, max_waves_per_simd=max_wps)
    w1 = amd_waves_per_simd(vgpr1, agpr1, max_waves_per_simd=max_wps)
    scratch0, scratch1 = num(r0, "scratch", -1), num(r1, "scratch", -1)
    lds0, lds1 = num(r0, "lds", -1), num(r1, "lds", -1)
    # 动态 LDS 是一条独立的结构性上限：环缓冲这类 kernel 每 CU 只能驻留 1 个 CTA，
    # 此时 VGPR 从 21 waves/SIMD 掉到 10 并不改变真实并发度（fi_bugs_amd-sk-ds B08 的假阳性）。
    block_threads = int(getattr(a, "block_threads", 0) or 256)   # 器件上限已在上面取过
    waves_per_wg = max(1, -(-block_threads // max(1, wavefront)))
    dyn_both = int(getattr(a, "dynamic_smem", 0) or 0)
    dyn0 = getattr(a, "l0_dynamic_smem", None)
    dyn1 = getattr(a, "fi_dynamic_smem", None)
    dyn0 = int(dyn0 if dyn0 is not None else dyn_both)
    dyn1 = int(dyn1 if dyn1 is not None else dyn_both)
    lds_w0 = amd_lds_waves_per_simd(lds0, dyn0, waves_per_wg, lds_per_cu, simds_per_cu, max_wps)
    lds_w1 = amd_lds_waves_per_simd(lds1, dyn1, waves_per_wg, lds_per_cu, simds_per_cu, max_wps)
    lds_note = ""
    eff0, eff1 = (float(w0) if w0 >= 0 else -1.0), (float(w1) if w1 >= 0 else -1.0)
    if lds_w0 == 0.0 or lds_w1 == 0.0:
        lds_note = ("LDS 上限算出 0 个 CTA/CU：静态+动态 LDS 超过每 CU 可用 LDS，"
                    "核对 --dynamic-smem/--l0-dynamic-smem/--fi-dynamic-smem 与 --amd-lds-per-cu；"
                    "这一项不参与判定，占用率仍按 VGPR 上限算")
        lds_w0 = lds_w1 = None
    else:
        if lds_w0 is not None and eff0 >= 0:
            eff0 = min(eff0, lds_w0)
        if lds_w1 is not None and eff1 >= 0:
            eff1 = min(eff1, lds_w1)
    ctas_lds0 = (lds_per_cu // max(1, max(0, lds0) + max(0, dyn0))) if (lds_w0 is not None or lds_note) else None
    ctas_lds1 = (lds_per_cu // max(1, max(0, lds1) + max(0, dyn1))) if (lds_w1 is not None or lds_note) else None
    spill0 = num(r0, "vgpr_spill_count") + num(r0, "sgpr_spill_count")
    spill1 = num(r1, "vgpr_spill_count") + num(r1, "sgpr_spill_count")
    bar0, bar1 = i0.get("barrier"), i1.get("barrier")
    # 探针是否可判：只有反汇编计数里真的有 timer_read / record_store 两类时才能判。
    # 预抽 JSON（例如离线回归夹具）可能只统计了 barrier/waitcnt，此时按“证据缺失”记 warning，不猜。
    probe_measurable = bool(i0) and bool(i1) and "timer_read" in i0 and \
        any(k in i1 for k in AMD_RECORD_STORE_CLASSES)
    timer_delta = store_delta = None
    store_deltas: Dict[str, int] = {}
    if probe_measurable:
        timer_delta = int(i1.get("timer_read", 0)) - int(i0.get("timer_read", 0))
        store_deltas = {k: int(i1.get(k, 0)) - int(i0.get(k, 0)) for k in AMD_RECORD_STORE_CLASSES}
        store_delta = store_deltas["record_store"]
    reg_delta_abs = (vgpr1 - vgpr0) if vgpr0 >= 0 and vgpr1 >= 0 else None
    reg_delta_pct = round(100.0 * reg_delta_abs / vgpr0, 3) if reg_delta_abs is not None and vgpr0 > 0 else None
    metrics.update({
        "kernel": {"l0": k0, "fi": k1},
        "vgpr": {"l0": vgpr0, "fi": vgpr1, "delta": reg_delta_abs, "delta_pct": reg_delta_pct},
        "sgpr": {"l0": num(r0, "sgpr"), "fi": num(r1, "sgpr"), "delta": num(r1, "sgpr") - num(r0, "sgpr")},
        "agpr": {"l0": agpr0, "fi": agpr1},
        "scratch_bytes_per_workitem": {"l0": scratch0, "fi": scratch1,
                                       "delta": (scratch1 - scratch0) if scratch0 >= 0 and scratch1 >= 0 else None},
        "lds_bytes": {"l0": lds0, "fi": lds1, "delta": (lds1 - lds0) if lds0 >= 0 and lds1 >= 0 else None},
        "spill_insns": {"l0": spill0, "fi": spill1, "delta": spill1 - spill0},
        "waves_per_simd": {"l0": w0, "fi": w1, "delta": (w1 - w0) if w0 >= 0 and w1 >= 0 else None,
                           "ratio": round(w1 / w0, 4) if w0 > 0 and w1 >= 0 else None},
        "occupancy_effective": {
            "waves_per_simd": {"l0": round(eff0, 4) if eff0 >= 0 else None,
                               "fi": round(eff1, 4) if eff1 >= 0 else None,
                               "ratio": round(eff1 / eff0, 4) if eff0 > 0 and eff1 >= 0 else None,
                               "meaning": "min(寄存器上限, LDS 上限)；占用率崩塌按这个判"},
            "register_ceiling_waves_per_simd": {"l0": w0, "fi": w1,
                                                "meaning": f"min(VGPR/AGPR 上限, 硬件并发上限 {max_wps} waves/SIMD)"
                                                           "；= metrics.waves_per_simd"},
            "register_file_only_waves_per_simd": {"l0": w0_file, "fi": w1_file,
                                                  "meaning": "只按每 SIMD 512 个 VGPR 的分配粒度算、未按硬件并发上限截断；"
                                                             "低 VGPR 时会给出硬件上不可能的数字，仅供对照，不参与判定"},
            "lds_ceiling_waves_per_simd": {"l0": round(lds_w0, 4) if lds_w0 is not None else None,
                                           "fi": round(lds_w1, 4) if lds_w1 is not None else None},
            "ctas_per_cu_limited_by_lds": {"l0": ctas_lds0, "fi": ctas_lds1},
            "dynamic_lds_bytes": {"l0": dyn0, "fi": dyn1},
            "assumptions": {"lds_per_cu": lds_per_cu, "simds_per_cu": simds_per_cu,
                            "wavefront_size": wavefront, "block_threads": block_threads,
                            "waves_per_workgroup": waves_per_wg,
                            "max_waves_per_simd": max_wps, "max_waves_per_cu": dev["max_waves_per_cu"],
                            "dynamic_lds_given": bool(any(int(getattr(a, k, 0) or 0) for k in
                                                          ("dynamic_smem", "l0_dynamic_smem", "fi_dynamic_smem"))),
                            "limits_source": dev["source"]},
            "note": lds_note or None,
        },
        "instructions": {"l0": i0 or None, "fi": i1 or None, "probe_measurable": probe_measurable,
                         "timer_read_delta": timer_delta, "record_store_delta": store_delta,
                         "record_store_deltas_by_signature": store_deltas or None,
                         "record_store_signatures": {
                             "record_store": "2_instrument/recorders/hip_cpp/fi_lite_hip.h 的 16B AoS 写（global_store_dwordx3/x4）",
                             "record_store_soa_time": "2_instrument/recorders/triton/fi_triton_amd.py 的 SoA 时间戳写（global_store_dwordx2）",
                             "record_store_soa_word": "同前端的 SoA tag/aux 写（global_store_dword）"},
                         "note": "SoA 的 dwordx2/dword 宽度业务代码里也有，强信号是 s_memrealtime/s_memtime 读的增量"},
    })

    # ---- 判据：结构性扰动 = FAIL；插桩固有代价 = warning
    fails: List[str] = []
    if a.strict_resources:
        if not (vgpr0 == vgpr1 and num(r0, "sgpr") == num(r1, "sgpr")):
            fails.append(f"--strict-resources：寄存器必须逐项相等（VGPR {vgpr0}->{vgpr1}，SGPR {num(r0,'sgpr')}->{num(r1,'sgpr')}）")
    elif reg_delta_abs is not None and reg_delta_abs > a.max_reg_delta_abs and (reg_delta_pct or 0) > a.max_reg_delta_pct:
        # AMD 上寄存器数只通过“每 SIMD 可驻留 wave 数”起作用：超预算但没有把占用率压下去就只是代价，
        # 判 FAIL 会把正确插桩误报成资源扰动（mail-14 实测的假阳性就是这么来的）。要更严用 --strict-resources
        # 或提高 --min-occupancy-ratio。
        warnings.append(f"VGPR {vgpr0}->{vgpr1}（+{reg_delta_abs}，{reg_delta_pct}%）超过寄存器预算"
                        f"（+{a.max_reg_delta_abs} 且 {a.max_reg_delta_pct}%），但占用率未崩塌，记 warning 不判失败")
    elif reg_delta_abs:
        warnings.append(f"VGPR {vgpr0}->{vgpr1}（+{reg_delta_abs}，{reg_delta_pct}%）：插桩固有代价，在预算内")
    if num(r1, "sgpr") > num(r0, "sgpr"):
        warnings.append(f"SGPR {num(r0,'sgpr')}->{num(r1,'sgpr')}：插桩需要额外的常量/地址寄存器")

    if i0 and i1 and "mfma" in i0 and "mfma" in i1:
        m0m, m1m = int(i0.get("mfma", 0)), int(i1.get("mfma", 0))
        if m1m < m0m:
            fails.append(f"业务核心指令 v_mfma 静态条数减少：{m0m}->{m1m}（插桩绝不能删业务计算；"
                         "mail-14 实测合法插桩两臂都是 16 条）")
        elif m1m > m0m:
            warnings.append(f"v_mfma 静态条数 {m0m}->{m1m}：核对是否复制了计算分支")

    bar_changed = bar0 is not None and bar1 is not None and bar0 != bar1
    if bar0 is None or bar1 is None:
        warnings.append("未提供反汇编计数，无法核对 s_barrier 条数（用 --l0/--fi 指向真实二进制可查）")
    elif bar_changed:
        fails.append(f"s_barrier 静态条数变化：{bar0}->{bar1}（记录宏绝不能引入/删除同步）")
    if lds0 >= 0 and lds1 >= 0 and lds0 != lds1:
        fails.append(f"静态 LDS 变化：{lds0}->{lds1} B（FI 记录必须写 global buffer，不能占用户 LDS）")
    if scratch0 >= 0 and scratch1 >= 0:
        sd = scratch1 - scratch0
        if sd > a.max_scratch_delta_bytes:
            fails.append(f"scratch 增量超预算：{scratch0}->{scratch1} B/work-item（+{sd}，预算 {a.max_scratch_delta_bytes}）")
        elif sd > 0:
            warnings.append(f"scratch {scratch0}->{scratch1} B/work-item（+{sd}，spill {spill0}->{spill1}）："
                            "插桩把记录状态压出寄存器，在预算内")
    if spill1 > spill0:
        warnings.append(f"spill 指令数 {spill0}->{spill1}")
    if lds_note:
        warnings.append(lds_note)
    if not any(int(getattr(a, k, 0) or 0) for k in ("dynamic_smem", "l0_dynamic_smem", "fi_dynamic_smem")):
        warnings.append("AMD 占用率没拿到动态 LDS（未给 --dynamic-smem/--l0-dynamic-smem/--fi-dynamic-smem）："
                        f"LDS 上限只按 code object 里的静态 .group_segment_fixed_size 算（l0={lds0} B, fi={lds1} B）。"
                        "Triton 与多数 HIP 流水/归约 kernel 的 LDS 全是动态分配（静态部分只是几十字节的握手标志，"
                        "Triton 的 metadata.shared 才是真用量），此时 LDS 上限被严重高估，占用率结论按证据不足看待："
                        "补 --dynamic-smem（Triton 取编译产物的 metadata.shared）后重跑")
    if eff0 > 0 and eff1 >= 0:
        lds_binds = lds_w0 is not None and (lds_w0 <= float(w0 if w0 >= 0 else 0) or lds_w0 <= float(w1 if w1 >= 0 else 0))
        if eff1 < a.min_occupancy_ratio * eff0:
            fails.append(f"每 SIMD 可驻留 wave 数崩塌：{round(eff0, 4)}->{round(eff1, 4)}（低于基线的 {a.min_occupancy_ratio}；"
                         f"寄存器上限 {w0}->{w1}，LDS 上限 {lds_w0}->{lds_w1}）")
        elif eff1 < eff0:
            warnings.append(f"占用率 {round(eff0, 4)}->{round(eff1, 4)} waves/SIMD：插桩固有代价，未崩塌"
                            f"（阈值 {a.min_occupancy_ratio}×基线；寄存器上限 {w0}->{w1}，LDS 上限 {lds_w0}->{lds_w1}）")
        elif w1 < w0 and lds_binds:
            warnings.append(f"VGPR 让寄存器上限从 {w0} 掉到 {w1} waves/SIMD，但动态 LDS 已把每 CU 卡在 "
                            f"{ctas_lds1} 个 CTA（{round(eff1, 4)} waves/SIMD），真实并发度没变，不判崩塌")
    elif w0 > 0 and w1 >= 0:
        if w1 < a.min_occupancy_ratio * w0:
            fails.append(f"每 SIMD 可驻留 wave 数崩塌：{w0}->{w1}（低于基线的 {a.min_occupancy_ratio}）")
        elif w1 < w0:
            warnings.append(f"waves/SIMD {w0}->{w1}：插桩固有代价，未崩塌（阈值 {a.min_occupancy_ratio}×基线）")
    if probe_measurable:
        soa_time_d = store_deltas.get("record_store_soa_time", 0)
        soa_word_d = store_deltas.get("record_store_soa_word", 0)
        if timer_delta < 1:
            fails.append(f"二进制里没找到 FI 计时器读（s_memrealtime/s_memtime 增量 +{timer_delta}，"
                         f"AoS 记录写 +{store_delta}、SoA 时间戳写 +{soa_time_d}、SoA tag/aux 写 +{soa_word_d}）："
                         "确认编译了 FI_LEVEL>=2、AMD Triton 前端确实被 import 并调用，且 kernel 正则选对")
        elif store_delta < 1 and soa_time_d < 1 and soa_word_d < 1:
            fails.append(f"计时器读到了（s_memrealtime/s_memtime +{timer_delta}）但两种记录写签名都没有增量："
                         f"AoS global_store_dwordx3/x4 +{store_delta}、SoA global_store_dwordx2 +{soa_time_d}、"
                         f"SoA global_store_dword +{soa_word_d}。先贴 llvm-objdump -d --mcpu=<gfx> 的 opcode 直方图"
                         "再下结论：记录可能被编译器合并/消除，也可能是前端只插了计时没插记录")
        elif store_delta < 1:
            warnings.append(f"FI 记录写按 Triton AMD 前端的 SoA 签名认到（时间戳 global_store_dwordx2 +{soa_time_d}、"
                            f"tag/aux global_store_dword +{soa_word_d}；AoS dwordx3/x4 +{store_delta}），"
                            f"配合 s_memrealtime/s_memtime 读 +{timer_delta}：探针在二进制里。"
                            "SoA 宽度业务代码也会出现，报告里三个增量都要写出来，不要只写“找到探针”")
    elif i0 and i1:
        probe_missing = "反汇编计数里没有 timer_read/record_store 两类，无法确认 FI 探针落进二进制"
        if require_fi_probe:
            fails.append(probe_missing + "（require_fi_probe=true，证据缺失按 FAIL）")
        else:
            warnings.append(probe_missing + "（显式 --allow-unverified-probe waiver，只放行探针证据，不改变资源/结构判据）")
    else:
        probe_missing = "未提供反汇编计数，无法核对 s_barrier 条数与 FI 探针"
        if require_fi_probe:
            fails.append(probe_missing + "（require_fi_probe=true，证据缺失按 FAIL）")
        else:
            warnings.append(probe_missing + "（显式 --allow-unverified-probe waiver）")
    if i0 and i1 and "waitcnt" in i0 and "waitcnt" in i1:
        w0w, w1w = int(i0.get("waitcnt", 0)), int(i1.get("waitcnt", 0))
        if w1w > w0w:
            warnings.append(f"s_waitcnt {w0w}->{w1w}：探针访存需要排序，属预期")

    # ---- 相似度（fi_binsim_hip，需要 AMDGCN 汇编或可自动反汇编的产物）
    bs: Dict[str, object] = {}
    if binsim_hip_mod is not None and a.l0 and a.fi and not (a.l0_amd_res or a.fi_amd_res):
        # 资源抽取这一步已经解出 gfx*.o 与 ROCm LLVM 目录；把它们交给 fi_binsim_hip，
        # 否则 binsim 会对 HIP host ELF 反汇编，只拿到 x86 host stub，算出假的 sim_strict
        # 并把正确插桩判成相似度不足（fi_bugs_amd-sk-ds B06/B07）。
        l0_bin = str(l0_doc.get("code_object") or a.l0)
        fi_bin = str(fi_doc.get("code_object") or a.fi)
        try:
            if hasattr(binsim_hip_mod, "set_rocm_llvm"):
                binsim_hip_mod.set_rocm_llvm(tools.get("rocm_llvm"))
            bs = binsim_hip_mod.compare(l0_bin, fi_bin, a.kernel, "amdgcn",
                                        str(Path(a.work_dir) / "binsim_hip_slices"),
                                        a.amd_gfx, str(Path(a.work_dir) / "binsim_hip_co"))
            evidence["binsim"] = bs
            evidence["binsim_inputs"] = {"l0": l0_bin, "fi": fi_bin,
                                         "note": "用资源抽取阶段解出的 device code object，不是 HIP host ELF"}
        except Exception as e:
            warnings.append(f"fi_binsim_hip 未跑成：{e}（相似度这一项无证据）")
            suggestions.append(f"相似度证据缺失：{e}")
    if bs:
        ko = bs.get("key_order") or {}
        sim_strict = as_float(bs.get("sim_strict"))
        sim_opcode = as_float(bs.get("sim_opcode"))
        sim_norm = as_float(bs.get("sim_norm"))
        probe_removed = (bs.get("l2") or {}).get("probe_removed")
        # key_lcs 有两个口径：助记符精确匹配（审计用）与语义同义组匹配（判定用）。
        # 基线的 flat_load_dword 在插桩版里被编译器落成 global_load_dwordx2 属同一语义，
        # 精确口径会把它判成“顺序被破坏”（fi_bugs_amd-sk-qwen B06）。
        key_lcs = as_float(ko.get("key_lcs_class"))
        if key_lcs is None:
            key_lcs = as_float(ko.get("key_lcs"))
        metrics.update({"sim_strict": sim_strict, "sim_opcode": sim_opcode, "sim_norm": sim_norm,
                        "probe_removed": probe_removed, "key_lcs": ko.get("key_lcs"),
                        "key_lcs_class": ko.get("key_lcs_class"),
                        "key_order_preserved": ko.get("preserved"),
                        "key_order_preserved_class": ko.get("preserved_class"),
                        "key_order_first_break": ko.get("first_break"),
                        "key_order_first_break_class": ko.get("first_break_class")})
        # 相似度低在 AMD 上首先说明“插桩加了很多指令”，不说明“业务代码被改坏”：mail-14 实测合法插桩
        # （只加计时器读 + 16B 记录写，业务指令一条没删）sim_strict=0.451（device .o 路线）/
        # 0.153（--save-temps .s 路线），sim_norm=0.699/0.536，sim_opcode=0.748/0.625，
        # 默认阈值 0.60/0.75 对任何有密度的插桩都不可达（fi_bugs_amd-sk-qwen B05）。
        if not isinstance(probe_removed, int) or probe_removed < 1:
            fails.append(f"反汇编里没能剔除任何 FI 探针指令（probe_removed={probe_removed}）："
                         "相似度差值无法归因到探针。确认 --kernel 选到被插桩的那个 kernel、编了 FI_LEVEL>=1，"
                         "并且传进来的是 device code object（gfx*.o / --save-temps 的 .s），不是 HIP host 可执行文件")
        if key_lcs is None or key_lcs < a.min_key_lcs:
            fails.append(f"关键指令顺序（同义组口径）key_lcs={key_lcs} 低于 {a.min_key_lcs}："
                         f"业务侧的装载/LDS/MMA/同步先后被改动了（第一处失配：{ko.get('first_break_class')}）")
        if a.strict_resources:
            if sim_strict is None or sim_strict < a.min_sim_strict:
                fails.append(f"--strict-resources：AMDGCN 语义相似度 sim_strict={sim_strict} 低于 {a.min_sim_strict}")
            if sim_opcode is None or sim_opcode < a.min_sim_opcode:
                fails.append(f"--strict-resources：opcode 相似度 sim_opcode={sim_opcode} 低于 {a.min_sim_opcode}")
            if sim_norm is None or sim_norm < a.min_sim_norm:
                fails.append(f"--strict-resources：归一化相似度 sim_norm={sim_norm} 低于 {a.min_sim_norm}")
        else:
            if sim_strict is not None and sim_strict < a.min_sim_strict:
                warnings.append(f"sim_strict={sim_strict} 低于阈值 {a.min_sim_strict}：插桩让指令流变长"
                                f"（剔除探针后 l0={((bs.get('l0') or {}).get('insns'))} -> "
                                f"l2={((bs.get('l2') or {}).get('insns_after'))} 条），"
                                "mail-14 实测合法插桩为 0.451/0.153；要按字面阈值判 FAIL 用 --strict-resources")
            if sim_opcode is not None and sim_opcode < a.min_sim_opcode:
                warnings.append(f"sim_opcode={sim_opcode} 低于阈值 {a.min_sim_opcode}：同上，属插桩固有代价"
                                "（mail-14 实测合法插桩 0.748/0.625）")
            if sim_norm is not None and sim_norm < a.min_sim_norm:
                warnings.append(f"sim_norm={sim_norm} 低于 {a.min_sim_norm}：寄存器归一化后仍有较多新增指令，"
                                "用 --dump-slices 抽查剔除的探针片段是否完整")
        if ko.get("preserved") is False and ko.get("preserved_class") is not False:
            warnings.append(f"关键指令的助记符精确顺序变化但同义组顺序保持（key_lcs={ko.get('key_lcs')} -> "
                            f"key_lcs_class={ko.get('key_lcs_class')}，第一处失配 {ko.get('first_break')}）："
                            "编译器把同一语义落成了不同助记符（flat_load/global_load/buffer_load 由编译器选），"
                            "或调换了分支两臂正文顺序")
        elif ko.get("preserved_class") is False:
            warnings.append("关键指令的同义组顺序也变了：人工核对 mfma/装载/LDS/同步的先后是否仍然成立")

    metrics["warnings"] = warnings
    metrics["fails"] = fails
    passed = not fails
    if fails:
        for f in fails:
            if "s_barrier" in f:
                suggestions.append("记录宏里不能有 __syncthreads()/s_barrier：FI_B/FI_E 只允许“读计时器 + 写 16B 记录”，"
                                   "把任何同步从宏里删掉，改用只由记录线程执行的写法（FI_GRAN=1 时只有 0 号线程记录）。")
            elif "scratch" in f or "spill" in f:
                suggestions.append("scratch 增长过多：把每 wave 的 FI 上下文压到 <=3 个寄存器（缓冲基址 + 计数），"
                                   "关掉循环内高频事件（FI_GROUPS 掩码）或改用 FI_GRAN=1 只让每 CTA 一个线程记录。")
            elif "wave" in f or "崩塌" in f:
                suggestions.append("占用率崩塌：降低插桩版本的 VGPR/AGPR 占用（减少同时活跃的事件、把 aux 计算移出热路径），"
                                   "或用 --min-occupancy-ratio 显式说明可接受的下限并重新论证。")
            elif "LDS" in f:
                suggestions.append("插桩改了静态 LDS：FI 记录必须写 global buffer（global_store_dwordx4），"
                                   "不要把记录缓冲放进 __shared__/LDS。")
            elif "探针" in f:
                suggestions.append("没找到 FI 探针：确认 -DFI_LEVEL=2（或 1）、FI_GROUPS 没有把事件全屏蔽、"
                                   "--kernel 正则匹配到被插桩的那个 kernel。")
            elif "相似度" in f:
                suggestions.append("相似度低：核对是否插入/删除了业务指令；只应在原数据流旁增加计时器读与 16B 记录写，"
                                   "用 --dump-slices 抽查 fi_binsim_hip 剔除的探针片段。")
            elif "key_lcs" in f or "同义组" in f:
                suggestions.append("关键指令顺序被改动：对照 binsim 的 key_order.first_break_class 定位第一处失配，"
                                   "把探针移出装载/LDS/MMA/同步的依赖链，只在这些边界之外记录。")
            elif "probe_removed" in f or "v_mfma" in f:
                suggestions.append("业务指令被动过或探针没识别出来：先确认 --kernel 与 device code object 对得上"
                                   "（fi_binsim_hip 会在找不到 kernel 标签时直接报错），再核对 v_mfma 条数是否两臂相同。")
            else:
                suggestions.append(f)
    return item(passed, metrics, thresholds, evidence, suggestions,
                note="AMD C1 区分“结构性扰动（FAIL）”与“插桩固有代价（warning）”；"
                     "判据与阈值全部写在 thresholds/metrics 里，--strict-resources 可改为逐项严格相等。"
                     + ("warnings: " + " | ".join(warnings) if warnings else ""))



# ---------------------------------------------------------------- C2

def extract_compare_json(stdout: str, out_path: Optional[str]) -> Optional[dict]:
    if out_path and Path(out_path).exists():
        try:
            return json.loads(Path(out_path).read_text(encoding="utf-8"))
        except Exception:
            pass
    for line in reversed(stdout.splitlines()):
        if line.startswith("COMPARE_JSON "):
            try:
                return json.loads(line[len("COMPARE_JSON "):])
            except json.JSONDecodeError:
                pass
    # 最后再接受“整段 stdout 就是 JSON”的脚本。
    try:
        d = json.loads(stdout)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def _c2_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _c2_shape_elements(shape) -> Optional[int]:
    if not isinstance(shape, list) or any(not _c2_int(x) or x < 0 for x in shape):
        return None
    n = 1
    for dim in shape:
        n *= dim
    return n


def _c2_structured_outputs(data: Dict[str, object]) -> Tuple[List[Dict[str, object]], List[str], str]:
    """把比较结果里的全量 output 清单规范化；缺字段不能靠 checked_elements 总数绕过。"""
    errors: List[str] = []
    manifest = data.get("output_manifest") if isinstance(data.get("output_manifest"), dict) else {}
    raw = data.get("outputs")
    source = "outputs"
    if raw is None:
        raw = manifest.get("outputs")
        source = "output_manifest.outputs"
    if raw is None and isinstance(data.get("arrays"), dict):
        raw = []
        source = "arrays"
        for name, row in sorted(data["arrays"].items()):
            if not isinstance(row, dict):
                errors.append(f"arrays[{name!r}] 不是结构化 object")
                continue
            item = dict(row)
            item.setdefault("name", name)
            if "elements" not in item and "checked_elements" in item:
                item["elements"] = item["checked_elements"]
            raw.append(item)
    if not isinstance(raw, list) or not raw:
        errors.append("缺少结构化 outputs（或 output_manifest.outputs）；只有 checked_elements 总数不能证明全量输出")
        return [], errors, source

    outputs: List[Dict[str, object]] = []
    names = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            errors.append(f"{source}[{index}] 不是 object")
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            errors.append(f"{source}[{index}] 缺 name")
            continue
        if name in names:
            errors.append(f"{source} 重复 output name: {name}")
        names.add(name)
        shape = entry.get("shape")
        shape_elements = _c2_shape_elements(shape)
        if shape_elements is None:
            errors.append(f"{source}[{index}] {name}: shape 必须是非负整数数组")
        dtype = str(entry.get("dtype") or "").strip()
        if not dtype:
            errors.append(f"{source}[{index}] {name}: 缺 dtype")
        elements = entry.get("elements")
        if not _c2_int(elements) or elements < 0:
            errors.append(f"{source}[{index}] {name}: elements 必须是非负整数")
            elements_int = None
        else:
            elements_int = int(elements)
            if shape_elements is not None and elements_int != shape_elements:
                errors.append(f"{source}[{index}] {name}: elements={elements_int} 与 shape 元素数 {shape_elements} 不一致")
        checked_row = entry.get("checked_elements")
        if checked_row is not None and (not _c2_int(checked_row) or checked_row != elements_int):
            errors.append(f"{source}[{index}] {name}: checked_elements={checked_row!r} 与 elements={elements_int!r} 不一致")
        layout = entry.get("layout")
        if layout is not None and not str(layout).strip():
            errors.append(f"{source}[{index}] {name}: layout 存在但为空")
        outputs.append({"name": name, "shape": shape, "dtype": dtype,
                        "elements": elements_int, "layout": layout})

    declared_keys = data.get("keys", manifest.get("keys"))
    if not isinstance(declared_keys, list) or not all(isinstance(x, str) for x in declared_keys):
        errors.append("缺少 keys 数组：无法核对 outputs 是否覆盖比较器实际选择的全部键")
    elif set(declared_keys) != names or len(declared_keys) != len(names):
        errors.append(f"keys 与结构化 outputs 名称不一致：keys={sorted(set(declared_keys))} outputs={sorted(names)}")
    if errors:
        return outputs, errors, source
    return outputs, [], source


def check_c2(a) -> Dict[str, object]:
    expected_elements = getattr(a, "expected_elements", None)
    thresholds = {"max_abs": a.max_abs, "max_rel": a.max_rel,
                  "min_checked_elements": expected_elements,
                  "max_nonfinite": 0, "require_all_outputs": True,
                  "require_structured_outputs": True, "require_total_elements_match": True,
                  "require_explicit_pass": True, "require_script_rc_zero": True,
                  "single_element_requires_explicit_expected": True}
    evidence: Dict[str, object] = {}
    suggestions: List[str] = []
    data = None
    errors: List[str] = []
    script_rc: Optional[int] = None
    if a.compare_json:
        try:
            data = load_json(a.compare_json)
        except Exception as e:
            errors.append(f"读取 compare-json 失败: {e}")
        evidence["compare_json"] = a.compare_json
        script_rc = getattr(a, "compare_rc", None)
    elif a.compare_cmd:
        argv = shlex.split(a.compare_cmd)
        p = run(argv, a.compare_timeout)
        evidence["command"] = argv
        evidence["returncode"] = p.returncode
        evidence["command_returncode"] = p.returncode
        script_rc = p.returncode
        evidence["stdout_tail"] = p.stdout[-4000:]
        evidence["stderr_tail"] = p.stderr[-4000:]
        data = extract_compare_json(p.stdout, a.compare_out)
        if p.returncode != 0:
            errors.append(f"比对脚本 rc={p.returncode}，即使输出了合法 JSON 也不能通过: {(p.stderr or p.stdout)[:500]}")
    else:
        errors.append("未提供 --compare-cmd 或 --compare-json，C2 无证据")
    if not isinstance(data, dict):
        errors.append("比对脚本没有输出可解析 JSON（需 COMPARE_JSON 行或 -o 文件）")
        data = {}
    if script_rc is None and isinstance(data, dict):
        for key in ("script_returncode", "returncode", "rc"):
            if _c2_int(data.get(key)):
                script_rc = int(data[key])
                break
    evidence["script_returncode"] = script_rc
    if script_rc is None:
        errors.append("缺少比较脚本 returncode：--compare-json 模式必须显式给 --compare-rc 0（或 JSON 带 returncode=0）")
    elif script_rc != 0:
        errors.append(f"比较脚本 returncode={script_rc}，必须为 0")

    explicit_pass = data.get("pass") is True
    if not explicit_pass:
        errors.append("compare JSON 必须显式包含布尔 pass=true（缺失或 false 都失败）")
    if data.get("error"):
        errors.append(f"compare JSON 带 error: {str(data.get('error'))[:500]}")

    outputs, output_errors, output_source = _c2_structured_outputs(data)
    errors.extend(output_errors)
    outputs_total = sum(int(x["elements"]) for x in outputs if _c2_int(x.get("elements")))
    checked_raw = data.get("checked_elements")
    checked = int(checked_raw) if _c2_int(checked_raw) and checked_raw >= 0 else 0
    if not _c2_int(checked_raw) or checked_raw < 0:
        errors.append("checked_elements 必须是非负整数")
    if outputs and checked != outputs_total:
        errors.append(f"checked_elements={checked} 与结构化 outputs 元素总数 {outputs_total} 不一致")
    if outputs and outputs_total <= 0:
        errors.append("结构化 outputs 的总元素数为 0：空输出不能作为正确性证据")

    total_declared = data.get("total_elements")
    if total_declared is not None:
        if not _c2_int(total_declared) or total_declared < 0:
            errors.append("total_elements 必须是非负整数")
        elif total_declared != outputs_total:
            errors.append(f"total_elements={total_declared} 与结构化 outputs 元素总数 {outputs_total} 不一致")
    if expected_elements is not None:
        if not _c2_int(expected_elements) or expected_elements < 0:
            errors.append("--expected-elements 必须是非负整数")
        elif outputs_total != expected_elements or checked != expected_elements:
            errors.append(f"输出完整性不一致：expected={expected_elements} checked={checked} outputs_total={outputs_total}")
    elif outputs and checked < 2:
        errors.append("只比较了 1 个元素且未给 --expected-elements：禁止静默通过；标量输出请显式传 --expected-elements 1")

    manifest = data.get("output_manifest") if isinstance(data.get("output_manifest"), dict) else {}
    all_outputs_proof = bool(
        data.get("all_outputs_compared") is True
        or (manifest.get("selection") == "all_outputs"
            and (data.get("key_sets_exact") is True or manifest.get("key_sets_exact") is True))
    )
    key_sets_exact = data.get("key_sets_exact") is True or manifest.get("key_sets_exact") is True
    if not key_sets_exact:
        errors.append("key_sets_exact 不是 true：比较器没有证明 reference/candidate 的输出键集合一致")
    if not all_outputs_proof:
        errors.append("没有全输出证据：all_outputs_compared=true（或 output_manifest.selection=all_outputs + key_sets_exact=true）")

    mac = as_float(data.get("max_abs"))
    mrel = as_float(data.get("max_rel"))
    nonfinite_raw = data.get("nonfinite", 0)
    nonfinite = int(nonfinite_raw) if _c2_int(nonfinite_raw) and nonfinite_raw >= 0 else 0
    if not _c2_int(nonfinite_raw) or nonfinite_raw < 0:
        errors.append("nonfinite 必须是非负整数")
    metrics = {"checked_elements": checked, "expected_elements": expected_elements,
               "outputs_total_elements": outputs_total, "total_elements": total_declared,
               "output_count": len(outputs), "outputs_source": output_source,
               "key_sets_exact": key_sets_exact, "all_outputs_compared": all_outputs_proof,
               "max_abs": mac, "max_rel": mrel, "nonfinite": nonfinite,
               "arrays": data.get("arrays"), "first_mismatch": data.get("first_mismatch"),
               "script_verdict": data.get("pass"), "explicit_pass": explicit_pass,
               "script_returncode": script_rc}
    evidence["comparison"] = data
    evidence["structured_outputs"] = outputs
    if mac is None or mrel is None:
        errors.append("max_abs/max_rel 必须是有限数值")
    ok = bool(not errors and mac is not None and mrel is not None
              and nonfinite == 0 and mac <= a.max_abs and mrel <= a.max_rel)
    if not ok:
        suggestions.append("用同一份 torch.randn 输入分别跑 L0 与插桩构建，sync 后保存完整输出张量；确认 shape/dtype/layout 一致，再加紧 atol/rtol 直到 max_abs=max_rel=0（理想情况应逐位一致）。")
        if expected_elements is not None and checked != expected_elements:
            suggestions.append(f"只比较了 {checked} 个元素，要求 {expected_elements}；不要只比 checksum/切片，逐元素覆盖全部输出。")
        if outputs and checked != outputs_total:
            suggestions.append(f"结构化 outputs 合计 {outputs_total} 个元素，但 checked_elements={checked}；比较器必须逐项自洽。")
        if not output_source or not outputs:
            suggestions.append("比较结果必须带 outputs/output_manifest.outputs（每个输出含 name/shape/dtype/elements）；只报一个总元素数不足以验收。")
        if nonfinite:
            suggestions.append("输出出现 inf/NaN：先在未插桩版本复查输入和索引；再检查插桩是否破坏分支/同步或覆盖业务寄存器。")
        if errors:
            suggestions.append("先修复证据命令：" + "；".join(errors))
    return item(ok, metrics, thresholds, evidence, suggestions,
                note="C2 要求比较脚本 rc=0、显式 pass=true、键集合完全一致、结构化 outputs 合计元素数与 checked/expected 三方一致；缺字段不能绕过。")


# ---------------------------------------------------------------- C3

def _load_decoder():
    sys.path.insert(0, str(HERE))
    import fi_lite_decode  # noqa
    return fi_lite_decode


def decode_dump_checks(dump_path: str, names_path: str) -> Dict[str, object]:
    m = _load_decoder()
    d = m.read_dump(dump_path)
    names = json.loads(Path(names_path).read_text(encoding="utf-8"))
    trace, summary = m.decode(d, names)
    anchor = defaultdict(int)
    for slot, rs in d["slots"].items():
        if not rs:
            continue
        opens = sum(1 for _, tag, _ in rs if tag & m.OPEN)
        closes = sum(1 for _, tag, _ in rs if tag & m.CLOSE)
        if opens == 0:
            anchor["缺 OPEN"] += 1
        if opens > 1:
            anchor["多个 OPEN"] += 1
        if closes == 0:
            anchor["缺 CLOSE（早退路径漏 FI_CLOSE）"] += 1
        if closes > 1:
            anchor["多个 CLOSE"] += 1
        if not (rs[0][1] & m.OPEN):
            anchor["OPEN 不是首条"] += 1
        if not (rs[-1][1] & m.CLOSE):
            anchor["CLOSE 不是末条"] += 1
    overflow = int(d.get("overflow", 0) or 0)
    overflow_detail = {"bit0_lane_capacity": bool(overflow & 1), "bit1_slot_bounds": bool(overflow & 2)}
    errs = dict(summary.get("errors", {}))
    errs.update({k: v for k, v in anchor.items() if v})
    total = sum(int(v) for v in errs.values())
    return {"dump": dump_path, "names": names_path, "summary": summary, "trace_events": len(trace.get("traceEvents", [])),
            "structure_errors": errs, "structure_error_count": total, "overflow_raw": overflow,
            "overflow": overflow_detail, "anchor_errors": dict(anchor), "slots": d["n_slots"],
            "records": summary.get("records"), "timer": summary.get("timer")}


def check_c3(a) -> Dict[str, object]:
    thresholds = {"max_structure_errors": 0, "max_overflow": 0, "max_anchor_errors": 0,
                  "max_unnamed_event_ids": 0, "max_records": a.max_records,
                  "require_nonempty_trace": True}
    evidence, suggestions = {}, []
    errors = []
    d = {}
    if not a.dump or not a.names:
        errors.append("需要 --dump 和 --names 才能检查事件完整性")
    elif not Path(a.dump).is_file():
        errors.append(f"dump 不存在: {a.dump}")
    elif not Path(a.names).is_file():
        errors.append(f"names 不存在: {a.names}")
    else:
        try:
            d = decode_dump_checks(a.dump, a.names)
            evidence.update(d)
        except Exception as e:
            errors.append(f"解码失败: {e}")
    s = d.get("summary", {})
    errs = d.get("structure_errors", {})
    unseen = int(thresholds["max_unnamed_event_ids"]) + 1 if s.get("unnamed_event_ids") else 0
    metrics = {"structure_error_count": d.get("structure_error_count"),
               "structure_errors": errs,
               "overflow_raw": d.get("overflow_raw"), "overflow": d.get("overflow"),
               "anchor_error_count": sum(int(v) for v in d.get("anchor_errors", {}).values()) if d else None,
               "unnamed_event_ids": s.get("unnamed_event_ids"),
               "records": d.get("records"), "trace_events": d.get("trace_events")}
    record_count = d.get("records")
    records_ok = (a.max_records is None or
                  (record_count is not None and int(record_count) <= int(a.max_records)))
    ok = (not errors and d.get("structure_error_count") == 0 and d.get("overflow_raw") == 0
          and sum(int(v) for v in d.get("anchor_errors", {}).values()) == 0
          and not s.get("unnamed_event_ids") and int(d.get("trace_events", 0)) > 0
          and records_ok)
    if not ok:
        if d.get("overflow"):
            suggestions.append("FI 缓冲溢出：把每 warp cap 调大到 2*最大事件数+4；高频循环事件移到 FI_BG(1) 并默认关闭；SM 粒度可用 FI_GRAN=1。")
        if any("未配对" in k or "多余" in k or "交叉" in k for k in errs):
            suggestions.append("事件区间配对不完整：每个 FI_B/ FI_BG 在每条路径上都要有且只有一个 FI_E/FI_EG；把循环内事件移入 FI_BG(1)，并给所有提前退出补 FI_E 后 FI_CLOSE。")
        if any("负时长" in k for k in errs):
            suggestions.append("出现负时长：检查是否混用了 globaltimer 与 clock64 转储，或同一泳道发生了记录重排；FI_TIMER 必须全构建一致。")
        if d.get("anchor_errors"):
            suggestions.append("OPEN/CLOSE 锚点不完整：每个 warp/记录线程在入口调用 FI_OPEN，在每个出口（含早退）先 FI_CLOSE；clock64 模式缺 CLOSE 会让换算频率错误。")
        if s.get("unnamed_event_ids"):
            suggestions.append("names.json 缺事件名：补齐所有出现的 event id，禁止用未命名事件发布 trace。")
        if not records_ok:
            suggestions.append(f"转储记录数 {record_count} 超过预算 {a.max_records}：检查是否在循环内重复加 B/E，改用 FI_BG(1) 或每 tile/chunk 只记一次。")
        if errors:
            suggestions.append("先修复解码证据：" + "；".join(errors))
    return item(ok, metrics, thresholds, evidence, suggestions,
                note="结构错误来自 fi_lite_decode 的 stack/aux 配对；另加每泳道 OPEN/CLOSE 锚点和 overflow 位检查。")


# ---------------------------------------------------------------- C4

def align_and_summary(a) -> Tuple[Dict[str, object], Dict[str, object]]:
    evidence: Dict[str, object] = {}
    if a.align_summary:
        d = load_json(a.align_summary)
        if not isinstance(d, dict):
            raise ValueError("align summary 不是 JSON object")
        evidence["summary_path"] = a.align_summary
        evidence["summary_sha256"] = sha256_file(a.align_summary)
        evidence["returncode"] = getattr(a, "align_rc", None)
        evidence["returncode_source"] = "--align-rc" if getattr(a, "align_rc", None) is not None else "align-summary-field"
        return d, evidence
    if platform_of(a) == "amd":
        if not (a.dump and a.names and a.rocprof):
            raise ValueError("AMD 需要 --align-summary，或同时提供 --dump --names --rocprof（rocprofv3 输出目录/JSON）")
        out = Path(a.work_dir) / f"{a.name}.aligned.perfetto.json.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, "-I", str(Path(a.align_rocprof_script)), a.dump, "--names", a.names,
                "--rocprof", a.rocprof, "--kernel", a.nsys_kernel or a.kernel,
                "--instance", str(a.nsys_instance), "-o", str(out), "--max-tail-us", str(a.max_tail_us)]
        if a.pmc:
            argv += ["--pmc", a.pmc]
        if a.info_json:
            argv += ["--info-json", a.info_json]
        p = run(argv, a.align_timeout)
        evidence.update({"command": argv, "returncode": p.returncode, "stdout_tail": p.stdout[-4000:],
                         "stderr_tail": p.stderr[-4000:], "trace": str(out)})
        if Path(out).is_file():
            evidence["trace_sha256"] = sha256_file(str(out))
        summary_path = str(out).replace(".gz", "").replace(".json", "") + ".summary.json"
        if not Path(summary_path).exists():
            raise RuntimeError(f"rocprofv3 对齐未生成 summary={summary_path}; rc={p.returncode}; {(p.stderr or p.stdout)[:500]}")
        d = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        evidence["summary_path"] = summary_path
        evidence["summary_sha256"] = sha256_file(summary_path)
        return d, evidence
    if not (a.dump and a.names and a.sqlite):
        raise ValueError("需要 --align-summary，或同时提供 --dump --names --sqlite")
    out = Path(a.work_dir) / f"{a.name}.aligned.perfetto.json.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-I", str(Path(a.align_script)), a.dump, "--names", a.names,
            "--sqlite", a.sqlite, "--kernel", a.nsys_kernel or a.kernel,
            "--instance", str(a.nsys_instance), "-o", str(out),
            "--max-tail-us", str(a.max_tail_us)]
    if a.info_json:
        argv += ["--info-json", a.info_json]
    p = run(argv, a.align_timeout)
    evidence.update({"command": argv, "returncode": p.returncode, "stdout_tail": p.stdout[-4000:],
                     "stderr_tail": p.stderr[-4000:], "trace": str(out)})
    if Path(out).is_file():
        evidence["trace_sha256"] = sha256_file(str(out))
    summary_path = str(out).replace(".gz", "").replace(".json", "") + ".summary.json"
    if not Path(summary_path).exists():
        raise RuntimeError(f"对齐未生成 summary={summary_path}; rc={p.returncode}; {(p.stderr or p.stdout)[:500]}")
    d = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    evidence["summary_path"] = summary_path
    evidence["summary_sha256"] = sha256_file(summary_path)
    return d, evidence


def _nonempty_errors(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _align_error_evidence(summary: Dict[str, object], align: Dict[str, object]) -> Dict[str, object]:
    axis = align.get("axis_check") if isinstance(align.get("axis_check"), dict) else {}
    values = {
        "summary.errors": summary.get("errors"),
        "align.errors": align.get("errors"),
        "align.metrics_problems": align.get("metrics_problems"),
        "align.axis_check.errors": axis.get("errors"),
    }
    out = {k: v for k, v in values.items() if _nonempty_errors(v)}
    # errors 必须是真正的空集合；null/缺字段/非集合值都不能证明“没有错误”。
    empty_collection = lambda value: isinstance(value, (dict, list, tuple, set)) and not value
    if not any(empty_collection(values[k]) for k in ("summary.errors", "align.errors")):
        out["errors.missing_or_invalid"] = (
            "align summary 未提供空的 errors 集合（必须显式给出 {} 或 []；null/缺字段不算空）"
        )
    return out


def evaluate_timing(summary: Dict[str, object], align: Dict[str, object], thresholds: Dict[str, object],
                    platform: str = "nvidia", align_returncode: Optional[int] = None) -> Dict[str, object]:
    step = as_float(summary.get("timer_min_step_ns"))
    cm = summary.get("clock_mhz_per_lane") or {}
    cmin, cmax = as_float(cm.get("min")), as_float(cm.get("max"))
    cmed = (cmin + cmax) / 2 if cmin is not None and cmax is not None else None
    spread = (100.0 * (cmax - cmin) / cmed) if cmin is not None and cmax is not None and cmed else None
    al = align.get("align", align)
    margin = as_float(al.get("end_margin_us")) if isinstance(al, dict) else None
    if not isinstance(al, dict):
        al = {}
    align_ok = al.get("ok") is True
    metrics_ok = al.get("metrics_ok") is True
    axis = al.get("axis_check")
    axis_ok = isinstance(axis, dict) and axis.get("ok") is True
    align_errors = _align_error_evidence(align, al)
    rc_candidates = [align_returncode, al.get("returncode"), al.get("rc"),
                     align.get("returncode"), align.get("align_returncode")]
    rc_values = [x for x in rc_candidates if x is not None]
    align_rc = rc_values[0] if rc_values else None
    rc_ok = isinstance(align_rc, int) and not isinstance(align_rc, bool) and align_rc == 0
    rc_conflict = any(x != align_rc for x in rc_values)
    errors_empty = not align_errors
    # clock64 换算频率只有 FI_TIMER=1 才有证据（OPEN/CLOSE 双锚点）。默认档 FI_TIMER=0 用
    # %globaltimer（NVIDIA）或 s_memrealtime（AMD），本来就是 ns 墙钟，没有“每泳道换算频率”这个东西：
    # 记为不适用，不因为缺证据判失败（fi_bugs_b3-sk-ds B04 / b3-sk-qwen B10 / h20-sk-qwen B8：
    # 此前只有 AMD 有豁免，NVIDIA 按文档默认档采集时 C4 恒 FAIL、overall_pass 永远 false）。
    # FI_TIMER=1 时照常用锚点频率判定：此时 cm 为空是真缺证据，仍然 FAIL。
    timer_name = str(summary.get("timer") or "")
    wall_clock_timer = timer_name in ("globaltimer", "s_memrealtime")
    clock_na = (not cm) and (platform == "amd" or wall_clock_timer)
    clock_na_reason = ("FI_TIMER=0（%globaltimer / s_memrealtime）是 ns 墙钟，没有 clock64 换算频率"
                       if clock_na else None)
    clock_ok = True if clock_na else (
        cmin is not None and cmax is not None
        and thresholds["min_clock_mhz"] <= cmin <= cmax <= thresholds["max_clock_mhz"]
        and spread is not None and spread <= thresholds["max_clock_spread_pct"])
    timing_ok = (step is not None and step <= thresholds["max_timer_min_step_ns"]
                 and clock_ok
                 and margin is not None and thresholds["min_end_margin_us"] <= margin <= thresholds["max_end_margin_us"])
    passed = bool(timing_ok and align_ok and metrics_ok and axis_ok and errors_empty and rc_ok and not rc_conflict)
    return {"pass": passed, "timer_min_step_ns": step,
            "clock64_mhz_range": {"min": cmin, "max": cmax, "median_pair": cmed, "spread_pct": spread},
            "clock_mhz_not_applicable": clock_na, "clock_mhz_not_applicable_reason": clock_na_reason,
            "timer": timer_name or None,
            "nsys_end_margin_us": margin, "align_ok": align_ok,
            "metrics_ok": metrics_ok, "axis_check_ok": axis_ok,
            "align_errors": align_errors, "align_errors_empty": errors_empty,
            "align_returncode": align_rc, "align_returncode_ok": rc_ok,
            "align_returncode_conflict": rc_conflict,
            "timing_ok": bool(timing_ok)}


def check_c4(a) -> Dict[str, object]:
    thresholds = {"max_timer_min_step_ns": a.max_timer_step_ns,
                  "min_clock_mhz": a.min_clock_mhz, "max_clock_mhz": a.max_clock_mhz,
                  "max_clock_spread_pct": a.max_clock_spread_pct,
                  "min_end_margin_us": 0.0, "max_end_margin_us": a.max_tail_us}
    evidence, suggestions, errors = {}, [], []
    summary: Dict[str, object] = {}
    align: Dict[str, object] = {}
    if a.dump and a.names and Path(a.dump).is_file() and Path(a.names).is_file():
        try:
            dec = decode_dump_checks(a.dump, a.names)
            summary = dec["summary"]
            evidence["decode_summary"] = summary
        except Exception as e:
            errors.append(f"计时 summary 解码失败: {e}")
    else:
        errors.append("缺少 --dump/--names，不能读取 timer summary")
    try:
        align, ae = align_and_summary(a)
        evidence.update(ae)
        evidence["align_summary"] = align
    except Exception as e:
        errors.append(f"软硬件对齐证据读取失败: {e}")
    run_rc = evidence.get("returncode")
    r = evaluate_timing(summary, align, thresholds, platform_of(a), align_returncode=run_rc)
    if not r["pass"]:
        if r["timer_min_step_ns"] is None or r["timer_min_step_ns"] > thresholds["max_timer_min_step_ns"]:
            suggestions.append("计时最小步长过粗：改用 -DFI_TIMER=1 的 clock64 构建，并确保 OPEN/CLOSE  associations 正确；globaltimer 适合跨 SM，clock64 适合细粒度。")
        cr = r["clock64_mhz_range"]
        if r.get("clock_mhz_not_applicable"):
            suggestions.append("FI_TIMER=0（NVIDIA %globaltimer / AMD s_memrealtime）没有 clock64 换算频率，"
                               "该项记为不适用、不判失败；需要周期级分辨率时用 -DFI_TIMER=1 重编"
                               "（clock64/readcyclecounter + OPEN/CLOSE 双锚点）。")
        elif cr["min"] is None or cr["max"] is None:
            suggestions.append("没有 clock64 频率范围：用 FI_TIMER=1 重编；若只能单锚点，meta 必须给 clock_mhz。")
        elif not (thresholds["min_clock_mhz"] <= cr["min"] <= cr["max"] <= thresholds["max_clock_mhz"]) or (cr["spread_pct"] or 0) > thresholds["max_clock_spread_pct"]:
            suggestions.append("clock64 换算频率异常：检查每条泳道是否都有 OPEN/CLOSE 双锚点、是否跨 MPS/不同 SM 时钟域，或 meta.clock_mhz 是否误填；降频范围内两锚点平均频率应稳定。")
        if r["nsys_end_margin_us"] is None or r["nsys_end_margin_us"] < 0 or r["nsys_end_margin_us"] > thresholds["max_end_margin_us"]:
            suggestions.append(("rocprofv3" if platform_of(a) == "amd" else "nsys") + " 结束余量不合格：确认 --kernel/--instance 选到被插桩的那次 launch；nsys 用 --capture-range-end=none 并让最后一条 FI_CLOSE 位于 kernel 结束内；必要时增加 cudaProfilerStart 前的垫 launch。")
        if errors:
            suggestions.append("先修复计时证据：" + "；".join(errors))
        if not r.get("align_ok"):
            suggestions.append("alignment 失败：summary.align.ok 必须严格为 true；任何 tail/metrics 失败都要重采或明确修复，不能写交付 trace。")
        if not r.get("metrics_ok"):
            suggestions.append("硬件指标覆盖失败：summary.align.metrics_ok 必须严格为 true；nsys 检查窗口内逐曲线样本，AMD 检查 PMC/dispatch 证据。")
        if not r.get("axis_check_ok"):
            suggestions.append("时间轴自检失败：summary.align.axis_check.ok 必须严格为 true；核对 launch/instance、timer_hz 与是否同一次采集。")
        if not r.get("align_errors_empty"):
            suggestions.append("对齐 summary 仍带 errors/metrics_problems；必须为空并在重新采集后重跑检查。")
        if not r.get("align_returncode_ok"):
            suggestions.append("对齐进程 rc 不是 0：从 --align-summary 读取时必须给 --align-rc 0 或 summary 内含 returncode=0；直接运行对齐器时以实际进程码为准。")
        if r.get("align_returncode_conflict"):
            suggestions.append("对齐 returncode 多处记录互相冲突，不能采信任何一处。")
    metrics = {"timer_min_step_ns": r["timer_min_step_ns"], "clock64_mhz_range": r["clock64_mhz_range"],
               "clock_mhz_not_applicable": r.get("clock_mhz_not_applicable"),
               "nsys_end_margin_us": r["nsys_end_margin_us"], "equal": {"timer": summary.get("timer")},
               "align_ok": r["align_ok"], "metrics_ok": r["metrics_ok"],
               "axis_check_ok": r["axis_check_ok"], "align_errors": r["align_errors"],
               "align_returncode": r["align_returncode"], "align_returncode_ok": r["align_returncode_ok"],
               "align_returncode_conflict": r["align_returncode_conflict"],
               "align_summary_sha256": evidence.get("summary_sha256"), "platform": platform_of(a),
               "clock_mhz_not_applicable_reason": r.get("clock_mhz_not_applicable_reason"),
               "axis_check": (align.get("align", align) or {}).get("axis_check") if isinstance(align, dict) else None}
    return item(not errors and r["pass"], metrics, thresholds, evidence, suggestions,
                note="最小步长用解码器实测的相邻记录差值；clock64 频率来自每条泳道 OPEN/CLOSE 两锚点换算；"
                     "余量与轴向自检来自 fi_align_nsys（NVIDIA）/ fi_align_rocprof（AMD）。")


# ---------------------------------------------------------------- final artifact / logs

def check_trace_artifact(a) -> Dict[str, object]:
    """检查最终交付 trace，而不是 fi_check 自己生成或检查的中间 dump。"""
    excluded = ["info", "nsys_kernel", "rocprofv3_kernel"]
    thresholds = {"min_business_x_events": 1, "excluded_categories": excluded,
                  "require_sha256": True, "require_final_trace": True}
    evidence: Dict[str, object] = {}
    suggestions: List[str] = []
    errors: List[str] = []
    path = getattr(a, "trace", None)
    expected_sha = getattr(a, "expected_trace_sha256", None)
    artifact: Dict[str, object] = {"path": path, "sha256": None, "business_x_events": None}
    business: List[Dict[str, object]] = []
    category_counts: Dict[str, int] = {}
    trace_events = None
    if not path:
        errors.append("缺少 --trace <final.perfetto.json[.gz]>：最终交付门禁必须直接检查实际 trace")
    elif not Path(path).is_file():
        errors.append(f"最终 trace 不存在: {path}")
    else:
        try:
            artifact["sha256"] = sha256_file(path)
            artifact["bytes"] = Path(path).stat().st_size
            doc = load_json(path)
            if not isinstance(doc, dict):
                raise ValueError("Perfetto JSON 顶层不是 object")
            trace_events = doc.get("traceEvents")
            if not isinstance(trace_events, list):
                raise ValueError("Perfetto JSON 缺 traceEvents 数组")
            other = doc.get("otherData") if isinstance(doc.get("otherData"), dict) else {}
            artifact["schema_version"] = other.get("schema_version")
            artifact["trace_events"] = len(trace_events)
            for index, event in enumerate(trace_events):
                if not isinstance(event, dict) or event.get("ph") != "X":
                    continue
                cat = str(event.get("cat") or "")
                category_counts[cat] = category_counts.get(cat, 0) + 1
                name = str(event.get("name") or "").strip()
                ts, dur = as_float(event.get("ts")), as_float(event.get("dur"))
                if cat in excluded or not name or ts is None or dur is None or dur < 0:
                    continue
                if "pid" not in event or "tid" not in event:
                    continue
                business.append({"index": index, "name": name[:160], "cat": cat,
                                 "ts": ts, "dur": dur, "pid": event.get("pid"), "tid": event.get("tid")})
            artifact["business_x_events"] = len(business)
            artifact["business_x_examples"] = business[:8]
            artifact["x_category_counts"] = category_counts
            if not business:
                errors.append("最终 trace 没有任何真实业务 X 区间（已排除 info/nsys_kernel/rocprofv3_kernel；"
                              "X 还必须有非空 name、数值 ts/dur、pid/tid）")
            if expected_sha:
                if str(expected_sha).lower() != str(artifact["sha256"]).lower():
                    errors.append(f"最终 trace SHA256 不匹配：declared={expected_sha} actual={artifact['sha256']}")
                artifact["expected_sha256"] = str(expected_sha).lower()
                artifact["sha256_verified"] = str(expected_sha).lower() == str(artifact["sha256"]).lower()
            else:
                artifact["sha256_verified"] = None
        except Exception as e:
            errors.append(f"读取最终 Perfetto trace 失败: {e}")
    evidence["artifact"] = artifact
    ok = not errors and trace_events is not None and bool(business)
    if not ok:
        if not path:
            suggestions.append("把最终交付的 .perfetto.json.gz 传给 --trace；不要只让检查器看中间 dump。")
        elif not business:
            suggestions.append("最终 trace 里没有任何业务区间：确认对齐/重命名阶段保留了软件泳道，并排除了只有元数据、info、nsys/rocprof kernel 条的假非空 trace。")
        if expected_sha and artifact.get("sha256_verified") is False:
            suggestions.append("trace 在声明哈希后发生变化：重新生成交付物，并把新的 SHA256 绑定进 check.json。")
        if errors:
            suggestions.append("先修复最终 trace 证据：" + "；".join(errors))
    return item(ok, {"business_x_events": artifact.get("business_x_events"),
                     "trace_events": artifact.get("trace_events"),
                     "sha256": artifact.get("sha256"),
                     "sha256_verified": artifact.get("sha256_verified")},
                thresholds, evidence, suggestions,
                note="Artifacts 直接打开 --trace，记录最终文件 SHA256，并要求至少一个非 info/nsys_kernel/rocprofv3_kernel 的业务 X 区间。")


def check_logs(a) -> Dict[str, object]:
    """调用 fi_logcheck，并把逐文件 clean、命中数与进程 rc 绑定进总门禁。"""
    thresholds = {"max_hits": int(getattr(a, "logcheck_max", 0) or 0), "require_clean_logs": True,
                  "require_logcheck_rc_zero": True, "require_nonempty_files": True,
                  "require_exec_rc_zero_if_given": True}
    paths = list(getattr(a, "logs", None) or [])
    evidence: Dict[str, object] = {"inputs": paths}
    suggestions: List[str] = []
    errors: List[str] = []
    summary: Dict[str, object] = {}
    script = str(getattr(a, "logcheck_script", DEFAULT_LOGCHECK))
    if not paths:
        errors.append("缺少 --logs：最终交付必须检查驱动/采集日志与 .excp(.gz) 异常转储")
    elif not Path(script).is_file():
        errors.append(f"fi_logcheck.py 不存在: {script}")
    else:
        work_dir = Path(getattr(a, "work_dir", None) or Path(getattr(a, "output", "check.json")).parent / "fi_check_evidence")
        work_dir.mkdir(parents=True, exist_ok=True)
        summary_path = work_dir / "logcheck.json"
        try:
            summary_path.unlink()
        except FileNotFoundError:
            pass
        argv = [sys.executable, "-I", script, *paths,
                "--max", str(thresholds["max_hits"]), "--show", "20", "--json-out", str(summary_path)]
        for pattern in list(getattr(a, "log_allow", None) or []):
            argv += ["--allow", pattern]
        if getattr(a, "strict_glog", False):
            argv.append("--strict-glog")
        exec_rc = getattr(a, "log_exec_rc", None)
        if exec_rc is not None:
            argv += ["--exec-rc", str(exec_rc)]
        p = run(argv, int(getattr(a, "logcheck_timeout", 300)))
        evidence.update({"command": argv, "returncode": p.returncode,
                         "stdout_tail": p.stdout[-4000:], "stderr_tail": p.stderr[-4000:],
                         "summary_path": str(summary_path)})
        if p.returncode != 0:
            errors.append(f"fi_logcheck rc={p.returncode}（必须为 0）")
        if not summary_path.is_file():
            errors.append(f"fi_logcheck 未写出汇总 JSON: {summary_path}")
        else:
            try:
                loaded = load_json(str(summary_path))
                if not isinstance(loaded, dict):
                    raise ValueError("summary 顶层不是 object")
                summary = loaded
                evidence["summary_sha256"] = sha256_file(str(summary_path))
            except Exception as e:
                errors.append(f"读取 fi_logcheck 汇总失败: {e}")
    if summary:
        evidence["summary"] = summary
        files = summary.get("files")
        if summary.get("schema") != "fi-logcheck/v1" or summary.get("tool") != "fi-logcheck":
            errors.append("fi_logcheck summary 缺正确 schema/tool 标识")
        if not isinstance(files, list) or not files:
            errors.append("日志门禁没有扫到任何文件：空目录不能证明日志干净")
        else:
            bad_files = [x for x in files if not isinstance(x, dict) or x.get("clean") is not True
                         or x.get("readable") is not True]
            if bad_files:
                errors.append(f"仍有 {len(bad_files)} 个日志/异常转储不干净或不可读")
            seen_paths = set()
            recorded_hits = 0
            for index, record in enumerate(files):
                if not isinstance(record, dict):
                    continue
                path_value = record.get("path")
                if not path_value or str(path_value) in seen_paths:
                    errors.append(f"fi_logcheck files[{index}] path 缺失或重复")
                    continue
                seen_paths.add(str(path_value))
                if not _c2_int(record.get("hits")) or record.get("hits") < 0:
                    errors.append(f"fi_logcheck files[{index}] hits 必须是非负整数")
                else:
                    recorded_hits += int(record["hits"])
                    if record.get("clean") is True and int(record["hits"]) != 0:
                        errors.append(f"fi_logcheck files[{index}] clean=true 但 hits={record['hits']}")
                if not is_sha256_hex(record.get("sha256")):
                    errors.append(f"fi_logcheck files[{index}] 缺有效 SHA256")
                elif not Path(str(path_value)).is_file():
                    errors.append(f"日志文件在检查后消失: {path_value}")
                else:
                    try:
                        actual_sha = sha256_file(str(path_value))
                    except OSError as e:
                        errors.append(f"日志文件在检查后无法复核 SHA256: {path_value}: {e}")
                    else:
                        if actual_sha != str(record.get("sha256")).lower():
                            errors.append(f"日志文件在检查后发生变化: {path_value}")
            if _c2_int(summary.get("total_hits")) and recorded_hits != summary.get("total_hits"):
                errors.append(f"fi_logcheck total_hits={summary.get('total_hits')} 与逐文件 hits={recorded_hits} 不一致")
        if summary.get("clean") is not True:
            errors.append("fi_logcheck summary.clean 不是 true")
        if summary.get("rc") != 0:
            errors.append(f"fi_logcheck summary.rc={summary.get('rc')}（必须为 0）")
        if summary.get("unreadable") != 0:
            errors.append(f"fi_logcheck unreadable={summary.get('unreadable')}（必须为 0）")
        if not _c2_int(summary.get("total_hits")) or summary.get("total_hits") < 0:
            errors.append("fi_logcheck summary.total_hits 缺失或非法")
        elif summary.get("total_hits") > thresholds["max_hits"]:
            errors.append(f"fi_logcheck total_hits={summary.get('total_hits')} 超过上限 {thresholds['max_hits']}")
        if "exec_rc" not in summary or "exec_rc_ok" not in summary:
            errors.append("fi_logcheck summary 缺 exec_rc/exec_rc_ok 字段")
        elif thresholds["require_exec_rc_zero_if_given"] and summary.get("exec_rc") is not None:
            if summary.get("exec_rc_ok") is not True:
                errors.append(f"被测程序/采集器 exec_rc={summary.get('exec_rc')}（必须为 0）")
        evidence["clean"] = summary.get("clean")
        evidence["returncode"] = summary.get("rc")
        evidence["files"] = files
    ok = not errors
    if not ok:
        suggestions.append("先清理 RuntimeError/Traceback/WARNING、rocprofv3 E/F glog 与所有 .excp/.excp.gz；日志不可读也按 FAIL。")
        suggestions.append("driver_template 的 RuntimeError 必须把非 0 rc 写到证据里；不能捕获后返回 0。")
        if errors:
            suggestions.append("日志门禁原因：" + "；".join(errors))
    return item(ok, {"clean": evidence.get("clean"), "rc": evidence.get("returncode"),
                     "files": len(evidence.get("files") or []), "unreadable": summary.get("unreadable"),
                     "total_hits": summary.get("total_hits")},
                thresholds, evidence, suggestions,
                note="Logs 调用同一个 fi_logcheck.py，要求 rc=0、clean=true、至少一个输入文件、逐文件可读且干净，并支持 .excp.gz。")


def summarize_checks(checks: Dict[str, Dict[str, object]]) -> Tuple[Dict[str, int], bool]:
    counts = {"pass": sum(1 for x in checks.values() if x.get("pass") is True),
              "fail": sum(1 for x in checks.values() if x.get("pass") is not True),
              "total": len(checks)}
    return counts, counts["total"] > 0 and counts["fail"] == 0


# ---------------------------------------------------------------- main

def platform_of(a) -> str:
    """auto：--arch 以 gfx 开头即 AMD；否则看二进制是否含 .hip_fatbin/AMDGPU ELF；再否则 NVIDIA。"""
    p = getattr(a, "platform", "auto")
    if p in ("nvidia", "amd"):
        return p
    arch = str(getattr(a, "arch", "") or "").lower()
    if arch.startswith("gfx"):
        return "amd"
    if arch.startswith("sm_"):
        return "nvidia"
    for path in (getattr(a, "fi", None), getattr(a, "l0", None)):
        if path and getattr(a, f"{path and ('fi' if path == getattr(a, 'fi', None) else 'l0')}_amd_res", None):
            return "amd"
    for path in (getattr(a, "fi", None), getattr(a, "l0", None)):
        if not path or not Path(path).is_file():
            continue
        try:
            head = path and Path(path).read_bytes()[:4096]
        except OSError:
            continue
        if head and (b"elf64-amdgpu" in head or b".hip_fatbin" in head or b"AMDGPU" in head):
            return "amd"
    return "nvidia"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="fast-instrument C1–C4 检查器，输出单一 check.json")
    ap.add_argument("--name", default="kernel")
    ap.add_argument("--l0", help="L0 可执行文件/so/cubin/Triton 产物")
    ap.add_argument("--fi", help="插桩可执行文件/so/cubin/Triton 产物")
    ap.add_argument("--kernel", required=False, help="目标 kernel 正则")
    ap.add_argument("--arch", default="sm_103a", help="sm_90/sm_100a/sm_103a 或 gfx942 等；gfx* 自动判为 AMD")
    ap.add_argument("--platform", choices=["auto", "nvidia", "amd"], default="auto",
                    help="auto=按 --arch 与二进制内容判定；也可显式指定")
    ap.add_argument("--amd-gfx", default="gfx942", help="AMD code object 的目标架构（从 fatbin 里挑这个 target）")
    ap.add_argument("--rocm-llvm", help="ROCm LLVM bin 目录（clang-offload-bundler/llvm-readelf/llvm-objdump）")
    ap.add_argument("--amd-lds-per-cu", type=int, default=None,
                    help="AMD：每 CU 可用 LDS 字节；默认取包内 knowledge/<gfx>.json（gfx942=65536）")
    ap.add_argument("--amd-simds-per-cu", type=int, default=None, help="AMD：每 CU 的 SIMD 数；默认取 knowledge/<gfx>.json（gfx942=4）")
    ap.add_argument("--amd-max-waves-per-simd", type=int, default=None,
                    help="AMD：每 SIMD 硬件并发 wave 上限；默认取 knowledge/<gfx>.json 的 max_waves_per_cu/simds_per_cu（gfx942=8）。"
                         "寄存器与 LDS 两个上限都会按它截断，避免算出硬件上不可能的 waves/SIMD")
    ap.add_argument("--amd-wavefront-size", type=int, default=None, help="AMD：wavefront 宽度；默认取 knowledge/<gfx>.json（gfx942=64）")
    ap.add_argument("--l0-amd-res", help="预先抽好的 L0 AMD 资源 JSON（无 ROCm 工具链时用；schema 见 amd_gather 文档）")
    ap.add_argument("--fi-amd-res", help="预先抽好的插桩版 AMD 资源 JSON")
    ap.add_argument("--max-scratch-delta-bytes", type=int, default=256,
                    help="AMD：scratch（private_segment_fixed_size）允许的增量上限，B/work-item")
    ap.add_argument("--min-occupancy-ratio", type=float, default=0.5,
                    help="AMD：waves/SIMD 相对基线的下限比例，低于它才算占用率崩塌")
    ap.add_argument("--work-dir", default=None,
                    help="中间证据（反汇编切片、抽出的 code object、对齐后的 trace）写这里；"
                         "默认写到 -o 输出 JSON 所在目录下的 fi_check_evidence/，"
                         "避免在包根目录运行时把产物落进只读的包里")
    ap.add_argument("--l0-resusage", help="可预先导出的 cuobjdump -res-usage 文本")
    ap.add_argument("--fi-resusage", help="可预先导出的 cuobjdump -res-usage 文本")
    ap.add_argument("--l0-build-log", help="L0 nvcc/ptxas -v 日志")
    ap.add_argument("--fi-build-log", help="插桩 nvcc/ptxas -v 日志")
    ap.add_argument("--block-threads", type=int, default=128)
    ap.add_argument("--dynamic-smem", type=int, default=0, help="两边相同动态 smem 字节")
    ap.add_argument("--l0-dynamic-smem", type=int)
    ap.add_argument("--fi-dynamic-smem", type=int)
    ap.add_argument("--cuda-device", type=int, default=0, help="driver API 物理 GPU 索引（仅查询属性）")
    ap.add_argument("--hw-limits-json", help="cuda_device_limits 格式 JSON；无 GPU 时可用于 occupant 计算")
    ap.add_argument("--strict-resources", action="store_true", help="寄存器/占用必须逐项严格相等")
    ap.add_argument("--max-reg-delta-pct", type=float, default=10.0)
    ap.add_argument("--max-reg-delta-abs", type=int, default=8)
    ap.add_argument("--min-sim-strict", type=float, default=0.60)
    ap.add_argument("--min-sim-opcode", type=float, default=0.75)
    ap.add_argument("--min-sim-norm", type=float, default=0.60,
                    help="寄存器归一化相似度下限（AMD 上默认记 warning，--strict-resources 时判 FAIL）")
    ap.add_argument("--min-key-lcs", type=float, default=0.80,
                    help="关键指令同义组子序列相似度下限；低于它就判 FAIL（业务顺序被改动）")
    ap.add_argument("--max-bb-delta-pct", type=float, default=300.0)
    ap.add_argument("--max-probe-events", type=int,
                    help="SASS 中 FI 探针 STG 的静态条数上限；用于发现循环内额外探针")
    ap.add_argument("--max-abs", type=float, default=0.0)
    ap.add_argument("--max-rel", type=float, default=0.0)
    ap.add_argument("--expected-elements", type=int, help="全量输出元素数；C2 checked_elements 与 outputs 总数必须严格等于此值")
    ap.add_argument("--compare-cmd", help="调用用户全量比较脚本的完整命令")
    ap.add_argument("--compare-json", help="已有比较结果 JSON（与 --compare-cmd 二选一）")
    ap.add_argument("--compare-out", help="比较脚本 -o 输出的文件；若给出则优先读取")
    ap.add_argument("--compare-rc", type=int, default=None,
                    help="--compare-json 模式下的比较脚本退出码；必须显式给 0（JSON 内含 returncode=0 也可）")
    ap.add_argument("--compare-timeout", type=int, default=1800)
    ap.add_argument("--require-script-pass", action="store_true", default=True,
                    help="兼容旧 CLI；C2 现在始终要求显式 pass=true，该参数不能关闭门禁")
    ap.add_argument("--dump", help="FILITE02 转储")
    ap.add_argument("--names", help="names.json")
    ap.add_argument("--max-records", type=int,
                    help="FILITE02 动态记录数上限；用于发现循环内重复探针造成的记录膨胀")
    ap.add_argument("--trace", help="最终交付的 Perfetto JSON(.gz)；Artifacts 直接校验 SHA256 与真实业务 X")
    ap.add_argument("--expected-trace-sha256", help="可选：声明的最终 trace SHA256；给出时实际文件必须完全匹配")
    ap.add_argument("--sqlite", help="同一次 nsys 导出的 sqlite")
    ap.add_argument("--nsys-kernel", help="nsys kernel 名正则；默认用 --kernel")
    ap.add_argument("--nsys-instance", type=int, default=-1)
    ap.add_argument("--align-summary", help="已有 fi_align_nsys summary JSON")
    ap.add_argument("--align-rc", type=int, default=None,
                    help="已有 --align-summary 时，原对齐进程的退出码；C4 要求严格为 0")
    ap.add_argument("--align-script", default=str(DEFAULT_ALIGN))
    ap.add_argument("--align-rocprof-script", default=str(DEFAULT_ALIGN_ROCPROF))
    ap.add_argument("--rocprof", help="AMD：rocprofv3 输出目录或 JSON（kernel-trace / counter_collection）")
    ap.add_argument("--pmc", help="AMD：rocprofv3 计数器 JSON/CSV，默认同 --rocprof")
    ap.add_argument("--align-timeout", type=int, default=600)
    ap.add_argument("--info-json", help="传给 fi_align_nsys 的软件信息 JSON")
    ap.add_argument("--max-tail-us", type=float, default=50.0)
    ap.add_argument("--max-timer-step-ns", type=float, default=None,
                    help="计时最小步长上限；默认 NVIDIA 128 ns（globaltimer），AMD 250 ns"
                         "（s_memrealtime，MI308X 实测相邻记录差 80–160 ns）")
    ap.add_argument("--min-clock-mhz", type=float, default=300.0)
    ap.add_argument("--max-clock-mhz", type=float, default=5000.0)
    ap.add_argument("--max-clock-spread-pct", type=float, default=5.0)
    ap.add_argument("--allow-unverified-probe", action="store_true",
                    help="AMD C1 显式豁免“反汇编看不到 FI 探针”；默认 require_fi_probe=true 时证据缺失即 FAIL")
    ap.add_argument("--logs", action="append", default=[],
                    help="最终驱动/采集日志文件或目录；可重复。最终门禁至少要求一个输入")
    ap.add_argument("--log-allow", action="append", default=[], metavar="REGEX",
                    help="透传给 fi_logcheck --allow；可重复，放行必须写理由")
    ap.add_argument("--logcheck-max", type=int, default=0,
                    help="透传给 fi_logcheck --max，默认 0 条可接受命中")
    ap.add_argument("--log-exec-rc", type=int, default=None,
                    help="被测程序/采集器退出码；给出时非 0 会让 Logs FAIL")
    ap.add_argument("--logcheck-timeout", type=int, default=300)
    ap.add_argument("--logcheck-script", default=str(DEFAULT_LOGCHECK))
    ap.add_argument("--strict-glog", action="store_true",
                    help="透传给 fi_logcheck：连 rocprofv3 的 benign glog 行也判失败")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("-o", "--output", default="check.json")
    return ap


def run_checks(a) -> Dict[str, object]:
    # 导入同目录的 fi_binsim / fi_binsim_hip；-I 模式不会自动加入脚本目录，这里显式加。
    sys.path.insert(0, str(HERE))
    try:
        import fi_binsim
    except Exception:
        fi_binsim = None
    try:
        import fi_binsim_hip
    except Exception:
        fi_binsim_hip = None
    platform = platform_of(a)
    if a.max_timer_step_ns is None:
        a.max_timer_step_ns = 250.0 if platform == "amd" else 128.0
    checks = {}
    if platform == "amd":
        if not ((a.l0 or a.l0_amd_res) and (a.fi or a.fi_amd_res) and a.kernel):
            checks["C1"] = item(False, {}, {"platform": "amd"}, {},
                                ["AMD C1 需要 --kernel，以及 --l0/--fi（HIP 产物）或 --l0-amd-res/--fi-amd-res（预抽 JSON）"])
        else:
            checks["C1"] = check_c1_amd(a, fi_binsim_hip)
    elif fi_binsim is None:
        checks["C1"] = item(False, {}, {}, {}, ["无法 import fi_binsim.py；确认与 fi_check.py 同目录"])
    elif not (a.l0 and a.fi and a.kernel):
        checks["C1"] = item(False, {}, {}, {}, ["C1 需要 --l0、--fi、--kernel"])
    else:
        checks["C1"] = check_c1(a, fi_binsim)
    checks["C2"] = check_c2(a)
    checks["C3"] = check_c3(a)
    checks["C4"] = check_c4(a)
    checks["Artifacts"] = check_trace_artifact(a)
    checks["Logs"] = check_logs(a)
    counts, overall_pass = summarize_checks(checks)
    trace_ev = checks["Artifacts"].get("evidence", {}).get("artifact", {})
    c4_ev = checks["C4"].get("evidence", {})
    log_ev = checks["Logs"].get("evidence", {})
    artifact_binding = {
        "trace": {"path": trace_ev.get("path"), "sha256": trace_ev.get("sha256"),
                  "business_x_events": trace_ev.get("business_x_events")},
        "align_summary": {"path": c4_ev.get("summary_path"), "sha256": c4_ev.get("summary_sha256")},
        "logcheck_summary": {"path": log_ev.get("summary_path"), "sha256": log_ev.get("summary_sha256")},
        "check_output": str(Path(a.output).resolve()) if getattr(a, "output", None) else None,
    }
    binding_sha = hashlib.sha256(json.dumps(artifact_binding, ensure_ascii=False, sort_keys=True,
                                            separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "tool": TOOL, "version": VERSION, "generated_utc": utc_now(), "name": a.name,
        "kernel_regex": a.kernel, "arch": a.arch, "platform": platform,
        "overall_pass": overall_pass, "counts": counts,
        "checks": checks,
        "artifacts": artifact_binding, "artifact_binding_sha256": binding_sha,
        "inputs": {"l0": a.l0, "fi": a.fi, "dump": a.dump, "names": a.names, "sqlite": a.sqlite,
                   "compare_cmd": a.compare_cmd, "compare_json": a.compare_json,
                   "trace": getattr(a, "trace", None), "align_summary": a.align_summary,
                   "align_rc": getattr(a, "align_rc", None), "logs": list(getattr(a, "logs", None) or [])},
        "policy": "fail 项均附可执行建议；未提供的证据不会被当作 pass；默认阈值见各项 thresholds。",
    }


def write_json(obj: object, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.endswith(".gz"):
        import gzip
        with gzip.open(p, "wb") as f:
            f.write(raw)
    else:
        p.write_bytes(raw)


def _mk_dump(path: Path, slots: Dict[int, List[Tuple[int, int, int]]], n_slots: int = 2,
             cap: int = 32, timer: int = 0, overflow: int = 0) -> None:
    import struct
    meta = json.dumps({"fi_check_selftest": True}).encode()
    cnt = [len(slots.get(i, [])) for i in range(n_slots)]
    blob = b"FILITE02" + struct.pack("<6I", n_slots, cap, 1, overflow, timer, len(meta)) + meta + struct.pack(f"<{n_slots}I", *cnt)
    for i in range(n_slots):
        for r in slots.get(i, []):
            blob += struct.pack("<QII", *r)
    path.write_bytes(blob)


def selftest() -> None:
    td = Path(tempfile.mkdtemp(prefix="fi_check_selftest_"))   # 系统临时目录，跑完删除
    # 资源解析 + 占用
    txt = "arch = sm_103a\nFunction _Z4demoPf:\n REG:40 STACK:0 SHARED:128 LOCAL:8\n"
    r = parse_res_usage(txt, "demo")
    assert r["registers_per_thread"] == 40 and r["static_smem_bytes"] == 128
    hw = {"sm_count": 148, "max_threads_per_sm": 2048, "max_shared_memory_per_sm": 232448,
          "max_registers_per_sm": 65536, "max_blocks_per_sm": 32}
    o = occupancy(r, 128, 1024, hw)
    assert o["ctas_per_sm"] > 0
    # 正确 dump：OPEN/B/E/CLOSE
    m = _load_decoder()
    names = td / "names.json"
    names.write_text(json.dumps({"events": {"1": "A"}, "roles": {"0": "r"}}), encoding="utf-8")
    good = td / "good.bin"
    _mk_dump(good, {0: [(1000, m.OPEN, 3), (1100, 1, 0), (1200, m.END | 1, 0), (1300, m.CLOSE, 0)]})
    d = decode_dump_checks(str(good), str(names))
    assert (d["structure_error_count"] == 0
            and d["overflow_raw"] == 0
            and not any(d["overflow"].values())), d
    # 缺一个 E：fi_check 必须报结构错误
    bad = td / "missing_e.bin"
    _mk_dump(bad, {0: [(1000, m.OPEN, 3), (1100, 1, 0), (1300, m.CLOSE, 0)]})
    d2 = decode_dump_checks(str(bad), str(names))
    assert d2["structure_error_count"] > 0 and any("未配对" in k for k in d2["structure_errors"]), d2
    # overflow bit 必须独立报出
    ov = td / "overflow.bin"
    _mk_dump(ov, {0: [(1000, m.OPEN, 3), (1300, m.CLOSE, 0)]}, overflow=1)
    d3 = decode_dump_checks(str(ov), str(names))
    assert d3["overflow_raw"] == 1 and d3["overflow"]["bit0_lane_capacity"], d3
    # C2 JSON 计算阈值
    ns = argparse.Namespace(compare_json=str(td / "cmp.json"), compare_cmd=None, compare_timeout=10,
                            compare_out=None, max_abs=0.0, max_rel=0.0, expected_elements=100,
                            compare_rc=0, require_script_pass=True)
    (td / "cmp.json").write_text(json.dumps({
        "pass": True, "checked_elements": 100, "total_elements": 100, "max_abs": 0.0,
        "max_rel": 0.0, "nonfinite": 0, "key_sets_exact": True, "all_outputs_compared": True,
        "keys": ["out"],
        "outputs": [{"name": "out", "shape": [10, 10], "dtype": "float32", "elements": 100,
                     "checked_elements": 100}],
    }), encoding="utf-8")
    c2 = check_c2(ns)
    assert c2["pass"], c2
    # 旧式“只报 checked_elements=1”没有 pass/outputs/keys/rc，任何字段缺失都不能被当成 PASS。
    (td / "cmp_bad.json").write_text(json.dumps({"checked_elements": 1, "max_abs": 0.0,
                                                  "max_rel": 0.0, "nonfinite": 0}), encoding="utf-8")
    ns_bad = argparse.Namespace(compare_json=str(td / "cmp_bad.json"), compare_cmd=None, compare_timeout=10,
                                compare_out=None, max_abs=0.0, max_rel=0.0, expected_elements=None,
                                compare_rc=None, require_script_pass=True)
    assert not check_c2(ns_bad)["pass"]
    # 显式声明 1 元素标量才允许通过；缺少 expected-elements 时单个元素不能静默拿 PASS。
    (td / "cmp_scalar.json").write_text(json.dumps({
        "pass": True, "checked_elements": 1, "total_elements": 1, "max_abs": 0.0,
        "max_rel": 0.0, "nonfinite": 0, "key_sets_exact": True, "all_outputs_compared": True,
        "keys": ["scalar"],
        "outputs": [{"name": "scalar", "shape": [], "dtype": "float32", "elements": 1,
                     "checked_elements": 1}],
    }), encoding="utf-8")
    ns_scalar = argparse.Namespace(**vars(ns_bad))
    ns_scalar.compare_json = str(td / "cmp_scalar.json")
    ns_scalar.compare_rc = 0
    assert not check_c2(ns_scalar)["pass"]
    ns_scalar.expected_elements = 1
    assert check_c2(ns_scalar)["pass"]
    # C4 计时阈值/频率/余量
    th = {"max_timer_min_step_ns": 128, "min_clock_mhz": 300, "max_clock_mhz": 5000,
          "max_clock_spread_pct": 5, "min_end_margin_us": 0, "max_end_margin_us": 50}
    ev = evaluate_timing({"timer_min_step_ns": 2, "clock_mhz_per_lane": {"min": 2000, "max": 2050}},
                         {"align": {"ok": True, "metrics_ok": True, "axis_check": {"ok": True},
                                    "end_margin_us": 2.5, "returncode": 0}, "errors": {}}, th)
    assert ev["pass"], ev
    ev2 = evaluate_timing({"timer_min_step_ns": 1024, "clock_mhz_per_lane": {"min": 10, "max": 20}},
                          {"align": {"ok": False, "end_margin_us": -1}}, th)
    assert not ev2["pass"], ev2
    # NVIDIA 默认档 FI_TIMER=0（%globaltimer）没有 clock64 锚点：该项记不适用，C4 不该因此 FAIL
    # （fi_bugs_b3-sk-ds B04 / b3-sk-qwen B10 / h20-sk-qwen B8：mail-17 实测 timer_min_step_ns=32、
    #   end_margin_us=1.157，其余三项全 PASS，却因这一项 overall_pass=false）。
    ev3 = evaluate_timing({"timer_min_step_ns": 32.0, "clock_mhz_per_lane": None, "timer": "globaltimer"},
                          {"align": {"ok": True, "metrics_ok": True, "end_margin_us": 1.157,
                                     "axis_check": {"ok": True}, "returncode": 0}, "errors": {}}, th)
    assert ev3["pass"] and ev3["clock_mhz_not_applicable"], ev3
    assert "globaltimer" in (ev3["clock_mhz_not_applicable_reason"] or ""), ev3
    # FI_TIMER=1（clock64）却拿不到锚点：这是真缺证据，必须 FAIL
    ev4 = evaluate_timing({"timer_min_step_ns": 32.0, "clock_mhz_per_lane": None, "timer": "clock64+锚点"},
                          {"align": {"ok": True, "end_margin_us": 1.0}}, th)
    assert not ev4["pass"] and not ev4["clock_mhz_not_applicable"], ev4
    # AMD 的 s_memrealtime 靠 timer 名就能判不适用（不依赖 --platform 传对）
    ev5 = evaluate_timing({"timer_min_step_ns": 80, "clock_mhz_per_lane": None, "timer": "s_memrealtime"},
                          {"align": {"ok": True, "metrics_ok": True, "axis_check": {"ok": True},
                                     "end_margin_us": 2.5, "returncode": 0}, "errors": {}}, th, "nvidia")
    assert ev5["pass"] and ev5["clock_mhz_not_applicable"], ev5
    # NVIDIA C1 寄存器判据：mail-12 实测的假阳性（48->56、占用率不变、spill 不增）必须降级为 warning
    assert nvidia_reg_verdict(48, 56, False, 8, 10.0, True, True) == (True, True)
    assert nvidia_reg_verdict(48, 56, False, 8, 10.0, False, True) == (False, False)   # 占用率掉了就判 FAIL
    assert nvidia_reg_verdict(48, 56, False, 8, 10.0, True, False) == (False, False)   # 出现 spill 就判 FAIL
    assert nvidia_reg_verdict(48, 56, True, 8, 10.0, True, True) == (False, False)     # --strict-resources
    assert nvidia_reg_verdict(128, 136, False, 8, 10.0, True, True) == (True, False)   # 预算内，不降级
    assert nvidia_reg_verdict(128, 200, False, 8, 10.0, True, True) == (True, True)    # 增幅大但占用率没掉
    # 高寄存器基数的目标（fi_bugs_b3-sk-qwen B18，B300 实测）：163->172（+9，5.521%），ctas_per_sm 1->1、
    # spill 0->0，只因绝对增量 9>8 判 FAIL；给了 ptxas -v 日志（spill 证据）就必须降级为 warning。
    assert nvidia_reg_verdict(163, 172, False, 8, 10.0, True, True) == (True, True)
    assert nvidia_reg_verdict(163, 172, False, 8, 10.0, True, False) == (False, False)  # 缺 spill 证据不降级
    assert nvidia_reg_verdict(163, 172, False, 8, 10.0, False, True) == (False, False)  # 占用率掉了不降级
    assert nvidia_reg_verdict(163, 172, False, 12, 10.0, True, True) == (True, False)   # --max-reg-delta-abs 12 时预算内
    # spill 证据三档（fi_bugs_b3-sk-qwen B16）：Triton 的 cubin 没有 ptxas 日志，STACK:0/LOCAL:0 要能当下界证据，
    # 不能再报 unavailable 把 C1 判死；只要有一臂分配了 local 就仍然算缺证据。
    zero_res = {"stack_bytes": 0, "local_bytes": 0, "registers_per_thread": 163}
    src, s0, s1, note = spill_evidence({"found": False}, {"found": False}, zero_res, dict(zero_res))
    assert src.startswith("cuobjdump") and (s0, s1) == (0, 0) and "ptxas -v" in note, (src, s0, s1, note)
    logs = ({"found": True, "spill_store_bytes": 0, "spill_load_bytes": 0},
            {"found": True, "spill_store_bytes": 100, "spill_load_bytes": 100})
    assert spill_evidence(*logs, {}, {})[0] == "ptxas -v" and spill_evidence(*logs, {}, {})[1:3] == (0, 200)
    assert spill_evidence({"found": False}, {"found": False},
                          zero_res, {"stack_bytes": 0, "local_bytes": 8})[0] is None
    assert spill_evidence({"found": False}, {"found": False}, {}, {})[0] is None
    assert "STACK/LOCAL" in spill_evidence({"found": False}, {"found": False}, {}, {})[3]
    # ---- AMD C1 判据：数字取自 mail-14（MI308X/gfx942）amd-checkers 的实测产物 ----
    def amd_ns(vgpr, sgpr, agpr, scratch, lds, spill, bar, waitcnt, timer=0, store=0):
        return {"gfx": "gfx942", "source": "selftest", "kernels": {"_Z4demoPKfPfPjiii": {
            "vgpr": vgpr, "sgpr": sgpr, "agpr": agpr, "scratch": scratch, "lds": lds,
            "vgpr_spill_count": spill, "sgpr_spill_count": 0,
            "instructions": {"barrier": bar, "waitcnt": waitcnt, "sleep": 1, "timer_read": timer,
                             "record_store": store, "global_load": 4, "lds": 6, "mfma": 0,
                             "atomic": 0, "cache_flush": 1, "_insns": 400}}}}
    base = {"vgpr": 53, "sgpr": 29, "agpr": 0, "scratch": 0, "lds": 0, "vgpr_spill_count": 0,
            "sgpr_spill_count": 0,
            "instructions": {"barrier": 2, "waitcnt": 12, "sleep": 1, "timer_read": 0,
                             "record_store": 0, "global_load": 4, "lds": 6, "mfma": 0,
                             "atomic": 0, "cache_flush": 1, "_insns": 400}}

    def amd_case(inst_kernels, l0_kernels=None, binsim=None, **ns_extra):
        ns = argparse.Namespace(platform="amd", arch="gfx942", amd_gfx="gfx942", rocm_llvm=None,
                                l0=None, fi=None, l0_amd_res="inline", fi_amd_res="inline",
                                kernel="demo", work_dir=str(td), max_scratch_delta_bytes=256,
                                min_occupancy_ratio=0.5, max_reg_delta_pct=10.0, max_reg_delta_abs=8,
                                strict_resources=False, min_sim_strict=0.60, min_sim_opcode=0.75,
                                min_sim_norm=0.60, min_key_lcs=0.80, name="selftest")
        for k, v in ns_extra.items():
            setattr(ns, k, v)
        # 直接把两侧画像塞进 amd_gather 的读取路径：用 monkeypatch 避免依赖真实文件
        docs = {"l0": {"gfx": "gfx942", "source": "selftest-l0",
                       "kernels": {"_Z4demoPKfPfPjiii": l0_kernels or base}},
                "fi": {"gfx": "gfx942", "source": "selftest-fi", "kernels": {"_Z4demoPKfPfPjiii": inst_kernels}}}
        orig = globals()["amd_gather"]

        def fake_gather(path, wd, gfx, tools, res_json=None):
            return docs["fi" if ("fi" in str(res_json) or "fi" in str(path)) else "l0"]

        globals()["amd_gather"] = fake_gather
        if binsim is not None:
            # 走真实二进制分支（fi_binsim_hip 会被调用），资源仍用内联画像
            ns.l0, ns.fi = "fake_l0.o", "fake_fi.o"
            ns.l0_amd_res = ns.fi_amd_res = None
        else:
            ns.l0_amd_res = "inline-l0"
            ns.fi_amd_res = "inline-fi"
        try:
            return check_c1_amd(ns, binsim)
        finally:
            globals()["amd_gather"] = orig

    inst = dict(base)
    inst.update({"vgpr": 63, "sgpr": 38, "scratch": 92, "vgpr_spill_count": 42,
                 "instructions": dict(base["instructions"], waitcnt=62, timer_read=6, record_store=6)})
    # ---- AMD 相似度新口径：数字取自 mail-14（MI308X/gfx942）的合法插桩实测
    #      sim_strict=0.451327、sim_norm=0.699115、sim_opcode=0.748420、probe_removed=178、
    #      key_lcs=0.825（精确口径 preserved=False：flat_load_dword 被编译器落成 global_load_dwordx2）
    class FakeBinsimHip:
        ko = {"n_l0_keys": 80, "preserved": False,
              "first_break": {"pos": 24, "expect": "flat_load_dword"}, "key_lcs": 0.825,
              "preserved_class": True, "first_break_class": None, "key_lcs_class": 1.0}
        extra: Dict[str, object] = {}

        @classmethod
        def compare(cls, *a, **k):
            r = {"sim_strict": 0.451327, "sim_norm": 0.699115, "sim_opcode": 0.748420,
                 "l0": {"insns": 830}, "l2": {"insns": 1688, "insns_after": 1510, "probe_removed": 178},
                 "key_order": cls.ko}
            r.update(cls.extra)
            return r

    r_sim = amd_case(inst, binsim=FakeBinsimHip)
    assert r_sim["pass"], r_sim["metrics"]["fails"]      # 旧口径在这里判 FAIL（fi_bugs_amd-sk-qwen B05）
    assert r_sim["metrics"]["sim_strict"] == 0.451327 and r_sim["metrics"]["sim_norm"] == 0.699115, r_sim["metrics"]
    assert r_sim["metrics"]["key_lcs"] == 0.825 and r_sim["metrics"]["key_lcs_class"] == 1.0, r_sim["metrics"]
    assert any("sim_strict=0.451327" in w for w in r_sim["metrics"]["warnings"]), r_sim["metrics"]["warnings"]
    assert any("同义组" in w for w in r_sim["metrics"]["warnings"]), r_sim["metrics"]["warnings"]
    # 业务顺序真被改坏：同义组口径也掉到 0.5 -> FAIL
    class BadOrder(FakeBinsimHip):
        ko = dict(FakeBinsimHip.ko, preserved_class=False, key_lcs_class=0.5,
                  first_break_class={"pos": 3, "expect": "mfma"})
    r_bad = amd_case(inst, binsim=BadOrder)
    assert not r_bad["pass"] and any("key_lcs" in f for f in r_bad["metrics"]["fails"]), r_bad["metrics"]["fails"]
    # 一条探针都没剔除：相似度差值无法归因 -> FAIL
    class NoProbe(FakeBinsimHip):
        extra = {"l2": {"insns": 1688, "insns_after": 1688, "probe_removed": 0}}
    r_np = amd_case(inst, binsim=NoProbe)
    assert not r_np["pass"] and any("probe_removed" in f for f in r_np["metrics"]["fails"]), r_np["metrics"]["fails"]
    # --strict-resources 恢复按字面阈值判相似度
    r_strict = amd_case(inst, binsim=FakeBinsimHip, strict_resources=True)
    assert not r_strict["pass"] and any("sim_strict" in f for f in r_strict["metrics"]["fails"]), r_strict["metrics"]["fails"]
    # 业务核心指令 v_mfma 条数减少 -> FAIL（插桩绝不能删业务计算）
    r_mfma = amd_case(dict(inst, instructions=dict(inst["instructions"], mfma=16)),
                      l0_kernels=dict(base, instructions=dict(base["instructions"], mfma=32)))
    assert not r_mfma["pass"] and any("v_mfma" in f for f in r_mfma["metrics"]["fails"]), r_mfma["metrics"]["fails"]

    r_ok = amd_case(inst)
    assert r_ok["pass"], r_ok  # 正确插桩：必须 PASS（这就是修掉的假阳性）
    # VGPR 53->63 按寄存器文件是 9->8 waves/SIMD，但 gfx942 硬件上限是 8，两臂都是 8：真实占用率没变
    assert r_ok["metrics"]["waves_per_simd"] == {"l0": 8, "fi": 8, "delta": 0, "ratio": 1.0}, r_ok["metrics"]["waves_per_simd"]
    assert r_ok["metrics"]["occupancy_effective"]["register_file_only_waves_per_simd"]["l0"] == 9, r_ok["metrics"]
    assert any("超过寄存器预算" in w for w in r_ok["metrics"]["warnings"]), r_ok["metrics"]["warnings"]
    # 没给 --dynamic-smem：LDS 上限只按静态部分算，必须记 warning（B18：Triton 的 LDS 全是动态的）
    assert any("动态 LDS" in w for w in r_ok["metrics"]["warnings"]), r_ok["metrics"]["warnings"]
    # F2：scratch 0->352（注入的 scratch 故障）必须 FAIL
    r_f2 = amd_case(dict(inst, scratch=352))
    assert not r_f2["pass"] and any("scratch" in f for f in r_f2["metrics"]["fails"]), r_f2["metrics"]["fails"]
    # F6：s_barrier 2->5（记录宏引入同步）必须 FAIL
    r_f6 = amd_case(dict(inst, instructions=dict(inst["instructions"], barrier=5)))
    assert not r_f6["pass"] and any("s_barrier" in f for f in r_f6["metrics"]["fails"]), r_f6["metrics"]["fails"]
    # ring 正确插桩：VGPR 77->84、waves 6->5、scratch/LDS/barrier 不变 -> PASS
    ring_base = dict(base, vgpr=77, sgpr=38, lds=64,
                     instructions=dict(base["instructions"], barrier=8, waitcnt=76))
    ring_inst = dict(ring_base, vgpr=84, sgpr=42,
                     instructions=dict(ring_base["instructions"], waitcnt=95, timer_read=8, record_store=8))
    r_ring = amd_case(ring_inst, l0_kernels=ring_base)
    assert r_ring["pass"], r_ring["metrics"]["fails"]      # ring 正确插桩：VGPR 77->84、waves 6->5 仍是 PASS
    assert r_ring["metrics"]["waves_per_simd"]["ratio"] == round(5 / 6, 4), r_ring["metrics"]["waves_per_simd"]
    # mail-14 的 C1 自检用例 VGPR 7->27（寄存器文件口径 64->16 waves/SIMD）：按硬件上限 8 截断后两臂都是 8，
    # 真实占用率没变，不能再判崩塌（fi_bugs_amd-sk-qwen B15 的假阳性正是这一类）。
    tiny = dict(base, vgpr=7, sgpr=32,
                instructions=dict(base["instructions"], timer_read=0, record_store=0))
    r_col = amd_case(dict(base, vgpr=27, sgpr=36,
                          instructions=dict(base["instructions"], timer_read=4, record_store=4)),
                     l0_kernels=tiny)
    assert r_col["pass"], r_col["metrics"]["fails"]
    assert r_col["metrics"]["waves_per_simd"] == {"l0": 8, "fi": 8, "delta": 0, "ratio": 1.0}, r_col["metrics"]
    assert r_col["metrics"]["occupancy_effective"]["register_file_only_waves_per_simd"] == {
        "l0": 64, "fi": 16, "meaning": r_col["metrics"]["occupancy_effective"]["register_file_only_waves_per_simd"]["meaning"]}
    # 真崩塌仍然要拦住：VGPR 20->200，寄存器上限 8->2（ratio 0.25 < 0.5）
    r_col2 = amd_case(dict(base, vgpr=200, sgpr=36,
                           instructions=dict(base["instructions"], timer_read=4, record_store=4)),
                      l0_kernels=dict(base, vgpr=20, sgpr=32,
                                      instructions=dict(base["instructions"], timer_read=0, record_store=0)))
    assert not r_col2["pass"] and any("崩塌" in f for f in r_col2["metrics"]["fails"]), r_col2["metrics"]["fails"]
    assert r_col2["metrics"]["waves_per_simd"] == {"l0": 8, "fi": 2, "delta": -6, "ratio": 0.25}, r_col2["metrics"]
    # LDS 被插桩占用：必须 FAIL
    r_lds = amd_case(dict(inst, lds=512))
    assert not r_lds["pass"] and any("LDS" in f for f in r_lds["metrics"]["fails"]), r_lds["metrics"]["fails"]
    # 探针没落进二进制：必须 FAIL
    r_np = amd_case(dict(inst, instructions=dict(inst["instructions"], timer_read=0, record_store=0)))
    assert not r_np["pass"] and any("探针" in f or "计时器读" in f for f in r_np["metrics"]["fails"]), \
        r_np["metrics"]["fails"]
    # Triton AMD 前端的 SoA 记录写（数字形状取自 fi_bugs_amd-sk-qwen B17 的实测直方图）：探针必须认得
    soa_l0 = dict(base, instructions=dict(base["instructions"], timer_read=0, record_store=0,
                                          record_store_soa_time=0, record_store_soa_word=0))
    soa_fi = dict(base, vgpr=base["vgpr"] + 6,
                  instructions=dict(base["instructions"], waitcnt=62, timer_read=13, record_store=0,
                                    record_store_soa_time=12, record_store_soa_word=33))
    r_soa = amd_case(soa_fi, l0_kernels=soa_l0)
    assert r_soa["pass"], r_soa["metrics"]["fails"]
    assert any("SoA" in w for w in r_soa["metrics"]["warnings"]), r_soa["metrics"]["warnings"]
    ins = r_soa["metrics"]["instructions"]
    assert ins["record_store_deltas_by_signature"] == {"record_store": 0, "record_store_soa_time": 12,
                                                      "record_store_soa_word": 33}, ins
    # 只有计时器读、两种记录写都没增量：仍然 FAIL，而且报错要把三种签名都报出来
    r_nostore = amd_case(dict(base, instructions=dict(base["instructions"], timer_read=13, record_store=0,
                                                     record_store_soa_time=0, record_store_soa_word=0)),
                         l0_kernels=soa_l0)
    assert not r_nostore["pass"] and any("记录写签名" in f for f in r_nostore["metrics"]["fails"]), \
        r_nostore["metrics"]["fails"]
    # 占用率公式：不给上限就是“只看寄存器文件”的口径（低 VGPR 会算出硬件上不可能的数字）
    assert amd_waves_per_simd(53) == 9 and amd_waves_per_simd(63) == 8 and amd_waves_per_simd(77) == 6
    assert amd_waves_per_simd(84) == 5 and amd_waves_per_simd(7) == 64 and amd_waves_per_simd(27) == 16
    assert amd_waves_per_simd(64, 64) == 8 and amd_waves_per_simd(-1) == -1
    # 给了 gfx942 的硬件并发上限（max_waves_per_cu 32 / simds_per_cu 4 = 8）就必须截断
    assert amd_waves_per_simd(7, max_waves_per_simd=8) == 8 and amd_waves_per_simd(27, max_waves_per_simd=8) == 8
    assert amd_waves_per_simd(20, max_waves_per_simd=8) == 8 and amd_waves_per_simd(200, max_waves_per_simd=8) == 2
    assert amd_device_limits("gfx942")["max_waves_per_simd"] == 8, amd_device_limits("gfx942")
    assert amd_device_limits("gfx942")["max_waves_per_cu"] == 32, amd_device_limits("gfx942")
    assert amd_device_limits("no_such_gfx")["max_waves_per_simd"] == 8      # 读不到 kb 也要有实测默认值
    # 动态 LDS 把每 CU 卡在 1 个 CTA 时，VGPR 造成的 waves/SIMD 21->10 不算崩塌（fi_bugs_amd-sk-ds B08：
    # ring_pipeline 的 3x16 KiB 环 -> 动态 LDS 49152 B）。不传动态 LDS 时仍按寄存器上限判崩塌。
    assert amd_lds_waves_per_simd(48, 49152, 4, 65536, 4) == 1.0
    assert amd_lds_waves_per_simd(0, 0, 4, 65536, 4) is None          # 两边都没用 LDS：这一项不参与
    assert amd_lds_waves_per_simd(0, 99999, 4, 65536, 4) == 0.0       # 超配 -> 0，调用方按“参数可疑”处理
    assert amd_lds_waves_per_simd(0, 99999, 4, 65536, 4, 8) == 0.0    # 截断不许把“超配”吞掉
    # 静态 LDS 只有几十字节时 LDS 上限会算出上千 wave/SIMD，必须按硬件上限截断
    assert amd_lds_waves_per_simd(48, 0, 4, 65536, 4) > 100
    assert amd_lds_waves_per_simd(48, 0, 4, 65536, 4, 8) == 8.0
    dyn_l0 = dict(base, vgpr=24, sgpr=40, lds=48,
                  instructions=dict(base["instructions"], barrier=8, timer_read=0, record_store=0))
    dyn_fi = dict(base, vgpr=48, sgpr=44, lds=48,
                  instructions=dict(base["instructions"], barrier=8, timer_read=4, record_store=21))
    r_dyn = amd_case(dyn_fi, l0_kernels=dyn_l0, block_threads=256, dynamic_smem=49152)
    assert r_dyn["pass"], r_dyn["metrics"]["fails"]
    occ = r_dyn["metrics"]["occupancy_effective"]
    assert occ["ctas_per_cu_limited_by_lds"] == {"l0": 1, "fi": 1}, occ
    assert occ["waves_per_simd"] == {"l0": 1.0, "fi": 1.0, "ratio": 1.0,
                                     "meaning": occ["waves_per_simd"]["meaning"]}, occ
    assert occ["register_ceiling_waves_per_simd"] == {"l0": 8, "fi": 8,
                                                     "meaning": occ["register_ceiling_waves_per_simd"]["meaning"]}, occ
    assert occ["register_file_only_waves_per_simd"]["l0"] == 21 and \
        occ["register_file_only_waves_per_simd"]["fi"] == 10, occ
    assert occ["assumptions"]["max_waves_per_simd"] == 8 and occ["assumptions"]["dynamic_lds_given"] is True, occ
    assert r_dyn["metrics"]["waves_per_simd"] == {"l0": 8, "fi": 8, "delta": 0, "ratio": 1.0}, \
        r_dyn["metrics"]["waves_per_simd"]
    # 不传动态 LDS：LDS 上限按静态 48 B 算（截断后 8 waves/SIMD），两臂相同，不再判崩塌，但要记 warning
    r_nodyn = amd_case(dyn_fi, l0_kernels=dyn_l0)
    assert r_nodyn["pass"], r_nodyn["metrics"]["fails"]
    assert any("动态 LDS" in w for w in r_nodyn["metrics"]["warnings"]), r_nodyn["metrics"]["warnings"]
    assert r_nodyn["metrics"]["occupancy_effective"]["assumptions"]["dynamic_lds_given"] is False
    # AMD C4：FI_TIMER=0 没有 clock64 频率 -> 该项不适用，其余照判
    th_amd = {"max_timer_min_step_ns": 250, "min_clock_mhz": 300, "max_clock_mhz": 5000,
              "max_clock_spread_pct": 5, "min_end_margin_us": 0, "max_end_margin_us": 50}
    ev_amd = evaluate_timing({"timer_min_step_ns": 80, "clock_mhz_per_lane": None},
                             {"align": {"ok": True, "metrics_ok": True, "axis_check": {"ok": True},
                                        "end_margin_us": 1.7, "returncode": 0}, "errors": {}}, th_amd, "amd")
    assert ev_amd["pass"] and ev_amd["clock_mhz_not_applicable"], ev_amd
    ev_amd2 = evaluate_timing({"timer_min_step_ns": 80, "clock_mhz_per_lane": None},
                              {"align": {"ok": False, "end_margin_us": -3}}, th_amd, "amd")
    assert not ev_amd2["pass"], ev_amd2
    shutil.rmtree(td, ignore_errors=True)
    print("SELFTEST PASS: 资源/占用、status-code XML/FILITE02、缺 E、overflow、C2 阈值、C4 计时、"
          "AMD C1 判据（正确插桩 PASS，scratch/barrier/LDS/真崩塌/无探针/业务顺序/v_mfma 减少 FAIL，"
          "占用率按硬件并发上限 8 waves/SIMD 截断，7->27 这类假崩塌改记 warning，缺动态 LDS 记 warning，"
          "相似度按同义组口径判、字面阈值记 warning）、AMD C4 不适用项、"
          "NVIDIA globaltimer 的 C4 豁免与寄存器降级均生效")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if not a.kernel:
        ap.error("需要 --kernel（或用 --selftest）")
    if not a.work_dir:
        # 默认跟着 -o 走：在包根目录下照文档跑命令时，证据不会落进只读的包里
        # （fi_bugs_b3-sk-qwen B12：默认 "fi_check_evidence" 是相对路径，会在包里新建 344 KB 证据）。
        a.work_dir = str(Path(a.output).parent / "fi_check_evidence")
    pkg_root = HERE.parent.parent
    wd_abs = Path(a.work_dir).resolve()
    if wd_abs == pkg_root or pkg_root in wd_abs.parents:
        print(f"提示：--work-dir={a.work_dir} 落在包内（{pkg_root}）。产物请写到包外的工作目录："
              f"给 -o 一个包外路径，或显式 --work-dir <包外目录>。", file=sys.stderr)
    Path(a.work_dir).mkdir(parents=True, exist_ok=True)
    result = run_checks(a)
    write_json(result, a.output)
    print(f"fi_check {a.name}: overall={'PASS' if result['overall_pass'] else 'FAIL'} "
          f"({result['counts']['pass']}/{result['counts']['total']})")
    for k in result["checks"]:
        x = result["checks"][k]
        print(f"{k}: {'PASS' if x['pass'] else 'FAIL'}")
        if not x["pass"]:
            for s in x["suggestions"]:
                print("  修法:", s)
    print("写出", a.output)
    return 0 if result["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
