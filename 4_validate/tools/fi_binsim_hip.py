#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fast-instrument / fi_binsim.py：插桩前后 kernel 二进制的相似度（Xtrace, arXiv 2609.28769 口径）。

流程（AMDGCN 部分，本组件负责；SASS 走同一框架、探针判定规则不同，本机无 N 卡未实测）：
  1) 导出 L0（FI_LEVEL=0，不插桩）与插桩版目标 kernel 的反汇编（hipcc --save-temps 的 .s，或 llvm-objdump -d），
     去掉地址与编码列，只留 “opcode 操作数” 文本。
  2) 识别并剔除探针指令（AMDGCN），分五类：
     - timer/hwid：s_memrealtime、s_getreg_b32 hwreg(HW_REG_HW_ID/HW_REG_XCC_ID) 及纯探针数据搬运链；
     - fi_sym：fi_g 符号地址计算（s_getpc + fi_g@gotpcrel）与字段装载（rec/overflow/cap 指针链）；
     - fi_store：写 FI 缓冲的 global/flat_store（含溢出路径的 atomic_or），及只服务探针的地址计算
       （按“定义-使用”链后向切片：输出只被探针用才收，寄存器复用不误伤业务）；
     - guard/exec 谓词：界限检查与记录线程选择的分支菱形（cmp + s_cbranch + exec 保存/恢复），
       仅当分支到目标之间全是探针时收编；
     - waitcnt：紧邻探针访存的 s_waitcnt（为 FI 字段装载排序而插入；相邻探针必须是访存指令，
       挨着 s_memrealtime 的业务 waitcnt 不动）。
     前向闭包之后还有一次剪枝：只保留位置级后向可达某个 FI 写的闭包成员——业务自读 fi_g 字段
     （如 fi_g.timer_hz 算等待超时）形状与 FI 相同，但数据流不进 FI 写，会被剔除出探针集
     （breakdown.rejected_business_fi_sym）；业务读 s_memrealtime 同理（rejected_business_timer）。
     地址切片、守卫菱形、waitcnt 三类收编走联合不动点（FI 字段装载常既喂地址又喂守卫谓词，
     单边先后顺序会漏收）。
  3) 两侧先做同一规则的死代码清扫（无副作用且目标寄存器无人读的指令迭代剔除，L0 也扫），
     剩余序列做 LCS 对齐，报告 sim_strict / sim_opcode / 基本块数变化 / L0 关键指令相对顺序
     （严格子序列判定 + key_lcs 软比例）。--dump-slices 导出剔除片段原文供人工抽查。

已知局限（如实说明）：守卫菱形收编仍要求分支区间内全是探针，编译器把业务指令调度进 FI 菱形时
整个菱形（exec 保存/恢复 + 分支）保留，体现为 branches 正差值；插桩还可能让编译器调换 if/else 两臂的
正文顺序（实测 demo 即如此），此时 key_order.preserved 为 False 属真实结构变化，看 key_lcs 与
block_delta 更稳。收编方向保守：谓词/地址链都有“只服务探针”验证，不会把业务指令误删成探针。

用法：
  python3 -I fi_binsim.py --l0 l0.s --l2 l2.s --kernel 'ring_pipeline' --isa amdgcn -o binsim.json [--dump-slices dir]
  python3 -I fi_binsim.py --l0 app_l0 --l2 app_l2 --kernel demo --isa amdgcn -o binsim.json   # 可执行文件自动反汇编
  python3 -I fi_binsim.py --selftest
"""
import argparse
import bisect
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

class BinSimError(RuntimeError):
    """可被调用方 `except Exception` 捕获的失败。

    旧实现在 read_asm 里 `raise SystemExit(...)`：SystemExit 继承 BaseException，fi_check 的
    `except Exception` 抓不到，整个检查器直接退出且不写 check.json（fi_bugs_amd-sk-ds B05）。
    """


# ROCm 的 llvm-objdump 常在 <rocm>/lib/llvm/bin 下，不在 <rocm>/bin 下（B05 的实测差异）。
_ROCM_ROOT_GLOBS = ("/opt/rocm*", "/usr/lib/rocm*", "/usr/local/rocm*")
_EXTRA_TOOL_DIRS: list = []          # 调用方（fi_check）解析出的 --rocm-llvm 目录，优先于系统猜测
EM_X86_64 = 0x3E
# EM_AMDGPU = 224 = 0xE0（/usr/include/elf.h：“#define EM_AMDGPU 224 /* AMD GPU */”；mail-14 上
# file(1) 对真实 gfx942 code object 报 “unknown arch 0xe0”，正是这个值）。此前写成 0xE3，
# 真器件 ELF 认不出来（fi_bugs_amd-sk-ds B22）。
EM_AMDGPU = 0xE0


def set_rocm_llvm(bindir) -> None:
    """由调用方注入已解析的 ROCm LLVM bin 目录，避免两边各猜一套路径。"""
    _EXTRA_TOOL_DIRS[:] = [str(bindir)] if bindir else []


def _rocm_bin_dirs():
    dirs = [Path(d) for d in _EXTRA_TOOL_DIRS]
    env = os.environ.get("FI_ROCM_LLVM")
    if env:
        dirs.append(Path(env))
    for pat in _ROCM_ROOT_GLOBS:
        for root in sorted(glob.glob(pat)):
            for sub in ("lib/llvm/bin", "llvm/bin", "bin"):
                dirs.append(Path(root) / sub)
    return dirs


def find_tool(env_var: str, name: str) -> str:
    """按 env -> 调用方给的目录 -> ROCm 安装目录 -> PATH 顺序找一个可执行文件，找不到返回 ""。"""
    v = os.environ.get(env_var)
    if v:
        if Path(v).exists():
            return str(Path(v).resolve())
    for d in _rocm_bin_dirs():
        c = d / name
        if c.is_file() and os.access(c, os.X_OK):
            return str(c.resolve())
    return shutil.which(name) or ""


def _searched_hint(name: str) -> str:
    seen = []
    for d in _rocm_bin_dirs():
        t = str(d)
        if t not in seen:
            seen.append(t)
    return f"找过：env、{'、'.join(seen[:6]) or '（无 ROCm 目录）'}、PATH；都没有 {name}"


def elf_machine(path) -> tuple:
    """读 ELF 头的 e_machine。返回 (是否 ELF, machine)；非 ELF 返回 (False, None)。"""
    try:
        with open(path, "rb") as f:
            head = f.read(20)
    except OSError:
        return False, None
    if len(head) < 20 or head[:4] != b"\x7fELF":
        return False, None
    # EI_DATA：1=小端，2=大端（int.from_bytes 只认 "little"/"big"，不认 struct 的 "<"/">"）。
    endian = "little" if head[5] == 1 else "big"
    return True, int.from_bytes(head[18:20], endian)


def extract_device_code_objects(path, gfx: str = "", work_dir=None) -> tuple:
    """从 HIP host ELF 的 .hip_fatbin 里抽出 device code object，返回 (路径列表, 过程记录)。

    llvm-objdump -d 直接作用于 host ELF 只反汇编 x86 host stub，device kernel 在 .hip_fatbin 里
    （fi_bugs_amd-sk-ds B06/B07：insns=75 而不是 906，l0_kernel_found=false）。
    """
    notes = []
    p = Path(path)
    wd = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="fi_binsim_co_"))
    wd.mkdir(parents=True, exist_ok=True)
    objcopy = find_tool("FI_OBJCOPY", "llvm-objcopy") or shutil.which("objcopy") or ""
    bundler = find_tool("FI_OFFLOAD_BUNDLER", "clang-offload-bundler")
    fat = wd / (p.name + ".hip_fatbin")
    if objcopy:
        r = subprocess.run([objcopy, "--dump-section", f".hip_fatbin={fat}", str(p), str(wd / "host_copy.o")],
                           capture_output=True, text=True, timeout=300)
        notes.append(f"objcopy --dump-section .hip_fatbin rc={r.returncode}")
        if r.returncode != 0 or not fat.is_file():
            notes.append((r.stderr or r.stdout or "").strip()[:300])
            fat = wd / "missing.bin"
    else:
        notes.append("没有 objcopy/llvm-objcopy，无法 dump .hip_fatbin")
    out = []
    if bundler and fat.is_file() and fat.stat().st_size:
        listing = subprocess.run([bundler, "--type=o", "--list", f"--input={fat}"],
                                 capture_output=True, text=True, timeout=300)
        targets = [x.strip() for x in (listing.stdout or "").splitlines() if x.strip()]
        notes.append(f"clang-offload-bundler --list: {targets}")
        want = [t for t in targets if t.endswith(f"--{gfx}")] if gfx else targets
        want = want or targets
        for i, t in enumerate(want):
            co = wd / f"code_object_{i}.o"
            r = subprocess.run([bundler, "--type=o", "--extract", f"--targets={t}",
                                f"--input={fat}", f"--output={co}"],
                               capture_output=True, text=True, timeout=300)
            if r.returncode != 0 and "--extract" in (r.stderr or ""):
                r = subprocess.run([bundler, "--type=o", "--unbundle", f"--targets={t}",
                                    f"--input={fat}", f"--output={co}"],
                                   capture_output=True, text=True, timeout=300)
            if r.returncode == 0 and co.is_file() and co.stat().st_size:
                out.append(co)
            else:
                notes.append(f"extract {t} rc={r.returncode} {(r.stderr or '')[:200]}")
    roc_obj = find_tool("FI_ROC_OBJ", "roc-obj")
    if not out and roc_obj:
        rd = wd / "roc_obj"
        rd.mkdir(parents=True, exist_ok=True)
        subprocess.run([roc_obj, "-o", str(p)], capture_output=True, text=True, cwd=str(rd), timeout=600)
        out = sorted(rd.rglob("*.co"))
        notes.append(f"roc-obj -o: {len(out)} 个 .co")
    return out, notes

# AMDGCN opcode 分类（前缀匹配，小写比较）
AMD_STORE_PREFIX = ("global_store", "flat_store", "buffer_store", "scratch_store", "ds_write",
                    "global_atomic", "flat_atomic", "buffer_atomic")  # atomic_or 溢出路径算 FI 写
AMD_LOAD_PREFIX = ("global_load", "flat_load", "buffer_load", "scratch_load", "ds_read", "s_load")
AMD_BRANCH_PREFIX = ("s_branch", "s_cbranch", "s_setpc", "s_swappc", "s_call", "s_endpgm", "s_barrier")
AMD_NO_DST = ("s_waitcnt", "s_barrier", "s_nop", "s_sleep", "s_endpgm", "s_sendmsg", "v_nop")
# L0 关键指令（结构指令；计时器读 s_memrealtime 不是结构指令，不在此列）
AMD_KEY_RE = re.compile(
    r"^(global_load|buffer_load|flat_load|ds_write|ds_read|s_waitcnt|s_barrier|global_store|flat_store|v_mfma|v_fma)")
SASS_STORE_PREFIX = ("STG", "STS", "ST.", "ATOMG", "ATOM")
SASS_BRANCH_PREFIX = ("BRA", "RET", "EXIT", "JMP", "BAR.", "BSSY", "BSYNC")
SASS_KEY_RE = re.compile(r"^(HGMMA|UTC|UTMALDG|UTMASTG|SYNCS|BAR\.|LDGSTS|DEPBAR|LDG|STG)")

# 函数标签两种形态：llvm-objdump 打 “0000000000207a00 <_Z13ring_pipeline...>:”，
# 汇编器 .s 打 “kernel_demo:”。两种都要认，否则“抽目标 kernel”会静默退化成“用整个文件”
# （fi_bugs_amd-sk-qwen B04d：labels=0、l0_kernel_found=false，多 kernel 产物里相似度被污染）。
OBJDUMP_LABEL_RE = re.compile(r"^([0-9a-fA-F]+)\s+<([^>]+)>:\s*$")
ASM_LABEL_RE = re.compile(r"^([\w$.]+):\s*$")
OBJDUMP_HEADER_RE = re.compile(r"^.*:\s+file format\s+")

# 同义组：同一语义在不同优化档/不同写法下会落成不同助记符（knowledge/cards/mi308x.json limits 第 2 条）。
# mail-14 实测：基线的 flat_load_dword 在插桩版里被编译器落成 global_load_dwordx2（同一次全局读），
# 精确匹配把这种编译器选择判成“关键指令顺序被破坏”（fi_bugs_amd-sk-qwen B06）。
AMD_KEY_CLASSES = (
    ("global_load", "mem_load"), ("flat_load", "mem_load"), ("buffer_load", "mem_load"),
    ("scratch_load", "mem_load"), ("s_load", "const_load"),
    ("global_store", "mem_store"), ("flat_store", "mem_store"), ("buffer_store", "mem_store"),
    ("scratch_store", "mem_store"),
    ("ds_read", "lds_read"), ("ds_write", "lds_write"),
    ("s_waitcnt", "waitcnt"), ("s_barrier", "barrier"),
    ("v_mfma", "mfma"), ("v_fma", "fma"),
)
SASS_KEY_CLASSES = (
    ("HGMMA", "mma"), ("UTC", "mma"),
    ("UTMALDG", "tma_load"), ("UTMASTG", "tma_store"),
    ("LDGSTS", "async_copy"), ("DEPBAR", "depbar"),
    ("SYNCS", "sync"), ("BAR.", "barrier"),
    ("LDG", "mem_load"), ("STG", "mem_store"),
)


def key_class(op: str, isa: str) -> str:
    """把关键指令助记符归到语义同义组；归不进去的原样返回。"""
    table = AMD_KEY_CLASSES if isa == "amdgcn" else SASS_KEY_CLASSES
    o = op.lower() if isa == "amdgcn" else op.upper()
    for prefix, cls in table:
        if o.startswith(prefix.lower() if isa == "amdgcn" else prefix.upper()):
            return cls
    return op


def _is_stub_label(text: str) -> bool:
    """host 侧 __device_stub__ 包装函数不是器件 kernel。

    llvm-objdump 反汇编 HIP host ELF 时，stub 符号名里含 kernel 名，正则会误命中，于是拿
    x86 host 指令流去算 AMDGCN 相似度（fi_bugs_amd-sk-qwen B04b：sim_strict=0.044 却 rc=0）。
    """
    t = text.lower()
    return "__device_stub__" in t or ".cold:" in t


def label_of_line(line: str):
    """取出一行里的函数标签名（objdump 或 .s 形态），不是标签返回 None。"""
    t = line.strip()
    m = OBJDUMP_LABEL_RE.match(t)
    if m:
        return m.group(2) + ":"
    if t.endswith(":"):
        m = ASM_LABEL_RE.match(t)
        if m:
            return m.group(1) + ":"
    return None


def labels_in_lines(lines) -> list:
    """文本里出现过的函数标签（去掉 host stub），用于“找不到 kernel”时给可执行报错。"""
    out = []
    for ln in lines:
        name = label_of_line(ln)
        if name and not _is_stub_label(name) and name not in out:
            out.append(name)
    return out


def has_kernel_label(lines, kernel_re) -> bool:
    """反汇编文本里是否有匹配 kernel_re 的真 kernel 标签（host stub 不算命中）。"""
    rx = re.compile(kernel_re)
    for ln in lines:
        name = label_of_line(ln)
        if name and rx.search(name) and not _is_stub_label(name):
            return True
    return False

REG_TOKEN = re.compile(r"\b([vs])\[(\d+):(\d+)\]|\b([vs])(\d+)\b")
COND_TOKEN = re.compile(r"\b(vcc_lo|vcc_hi|vcc|scc|exec_lo|exec_hi|exec)\b")
COND_REGS = {"scc", "vcc", "exec", "vcc_lo", "vcc_hi", "exec_lo", "exec_hi"}


def _expand_regs(text):
    """把指令文本里的寄存器操作数展开成寄存器集合（v[0:3] -> v0..v3）。含 vcc/scc/exec 条件码。"""
    regs = set(COND_TOKEN.findall(text))
    for m in REG_TOKEN.finditer(text):
        if m.group(1):
            lo, hi = int(m.group(2)), int(m.group(3))
            if hi >= lo and hi - lo <= 32:
                for i in range(lo, hi + 1):
                    regs.add(f"{m.group(1)}{i}")
        else:
            regs.add(f"{m.group(4)}{m.group(5)}")
    return regs


def parse_amdgcn(lines):
    """解析 AMDGCN 汇编文本（.s 或 llvm-objdump -d 输出）。返回 insns=[dict(op,text,dsts,srcs,raw)]。"""
    insns = []
    for ln in lines:
        raw = ln.rstrip()
        t = ln.strip()
        if not t or t.startswith(("//", "#", ";")):
            continue
        if re.match(r"^[0-9a-f]+:\s", t):  # llvm-objdump：去地址列，编码列可有可无
            # _objdump() 传了 --no-show-raw-insn，此时没有编码列；旧写法把编码列当必需，
            # 于是 llvm-objdump 的每一条指令都被丢掉（只剩标签，insns=0）。
            m = re.match(r"^[0-9a-f]+:\s+(?:(?:[0-9a-f]{2} )+\s*)?(.*)$", t)
            if not m:
                continue
            t = m.group(1).strip()
            if not t:
                continue
        t = re.sub(r"\s*(?://|;).*$", "", t).strip()  # .s 行尾注释用 ;，objdump 用 //
        if not t:
            continue
        if t.startswith(("Disassembly of section", "...")) or OBJDUMP_HEADER_RE.match(t):
            continue                                  # llvm-objdump 的段头/文件头/省略号不是指令
        m = OBJDUMP_LABEL_RE.match(t)
        if m:                                         # “<addr> <symbol>:” —— objdump 的函数标签
            insns.append(dict(op="LABEL", text=m.group(2) + ":", dsts=set(), srcs=set(), raw=raw))
            continue
        if t.endswith(":") and re.match(r"^[\w$.]+:$", t):  # 含 .LBB 局部标号（守卫菱形/基本块统计要用）
            insns.append(dict(op="LABEL", text=t, dsts=set(), srcs=set(), raw=raw))
            continue
        parts = t.split(None, 1)
        op = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        if op.startswith("."):
            continue
        opl = op.lower()
        if opl.startswith(("s_cmp", "v_cmp")):
            first = rest.split(",", 1)[0]
            if opl.startswith("s_cmp"):
                dsts = {"scc"}
            elif "_e64" in opl:
                dsts = _expand_regs(first)   # v_cmp_*_e64 写显式 SGPR 对
            else:
                dsts = {"vcc"}               # v_cmp_*_e32 隐式写 vcc
            srcs = _expand_regs(rest[len(first):]) if "," in rest else set()
        elif opl.startswith(AMD_STORE_PREFIX) or opl.startswith(AMD_BRANCH_PREFIX) or opl in AMD_NO_DST:
            dsts, srcs = set(), _expand_regs(rest)
            if opl.startswith("s_cbranch"):  # 条件分支隐式读条件码
                if "scc" in opl:
                    srcs.add("scc")
                elif "exec" in opl:
                    srcs.add("exec")
                elif "vcc" in opl:
                    srcs.add("vcc")
        else:
            first = rest.split(",", 1)[0]
            dsts = _expand_regs(first)
            srcs = _expand_regs(rest[len(first):]) if "," in rest else set()
            # SALU 隐式 scc：加/减/逻辑/移位写 scc；addc/subb 还读 scc；cselect 读 scc
            if re.match(r"^s_(add_u32|sub_u32|addc_u32|subb_u32|and|or|xor|nand|nor|xnor|not|lshl|lshr|ashr)", opl):
                dsts.add("scc")
            if re.match(r"^s_(addc_u32|subb_u32|cselect)", opl):
                srcs.add("scc")
            if "saveexec" in opl:            # saveexec：读 exec，写 exec 与 scc
                dsts |= {"exec", "scc"}
                srcs.add("exec")
        insns.append(dict(op=op, text=f"{op} {rest}".strip(), dsts=dsts, srcs=srcs, raw=raw))
    return insns


def parse_sass(lines):
    """SASS：去地址/编码列，opcode 取首 token。本机无 N 卡，规则按既有文档口径实现，未实测。"""
    insns = []
    for ln in lines:
        raw = ln.rstrip()
        t = ln.strip()
        if not t or t.startswith(("#", ".", "//")):
            continue
        t = re.sub(r"/\*[0-9a-f]+\*/", "", t).strip()
        t = re.sub(r"\s+/\*[^*]*\*/\s*$", "", t).strip()
        if not t:
            continue
        if t.endswith(":"):
            insns.append(dict(op="LABEL", text=t, dsts=set(), srcs=set(), raw=raw))
            continue
        parts = t.split(None, 1)
        op = parts[0].rstrip(";")
        rest = (parts[1] if len(parts) > 1 else "").rstrip(";").strip()
        up = op.upper()
        regs = lambda s: set(re.findall(r"\bR\d+\b|\bUR\d+\b", s))
        if up.startswith(SASS_STORE_PREFIX) or up.startswith(SASS_BRANCH_PREFIX):
            dsts, srcs = set(), regs(rest)
        else:
            first = rest.split(",", 1)[0]
            dsts = regs(first)
            srcs = regs(rest[len(first):])
        insns.append(dict(op=op, text=f"{op} {rest}".strip(), dsts=dsts, srcs=srcs, raw=raw))
    return insns


def is_timer_seed(insn, isa):
    """直接读计时器/硬件号（第一类种子）。"""
    o = insn["op"].lower()
    t = insn["text"].lower()
    if isa == "amdgcn":
        if o in ("s_memrealtime", "s_memtime"):
            return "timer"
        if o == "s_getreg_b32" and ("hw_id" in t or "xcc" in t or "hwreg" in t):
            return "hwid"
        return None
    up = insn["op"].upper()
    if up.startswith(("S2R", "S2UR", "CS2R")) and re.search(r"SR_(GLOBALTIMER|CLOCK|SMID)", insn["text"].upper()):
        return "timer"
    return None


def _is_store(ins, isa):
    opl = ins["op"].lower()
    return opl.startswith(AMD_STORE_PREFIX) if isa == "amdgcn" else ins["op"].upper().startswith(SASS_STORE_PREFIX)


def _fwd_closure(insns, isa, seed_idx):
    """单遍前向数据流引擎。pool=探针派生寄存器池，neutral=常量 0 等中性寄存器池。
    - 种子（seed_idx）：收编，dsts 入 pool；
    - store：任一源 ∈ pool => FI 写，收编（不传播）；
    - 非 store 且有 dst 且有 src：srcs ⊆ pool∪neutral 且至少一个 ∈ pool => 收编，dsts 入 pool；
      srcs ⊆ neutral  => 中性传播（常量搬运链），不收编；
      其余：不收编，dsts 从两池删除（寄存器被业务重定义，派生链死亡）。
    无 src 的 v_mov/s_mov（常量 0/立即数）入 neutral。"""
    pool, neutral, closure, stores = set(), set(), set(), []
    for i, ins in enumerate(insns):
        if i in seed_idx:
            closure.add(i)
            pool |= ins["dsts"] - COND_REGS
            continue
        srcs, dsts = ins["srcs"] - COND_REGS, ins["dsts"] - COND_REGS
        if _is_store(ins, isa):
            if srcs & pool:
                closure.add(i)
                stores.append(i)
            continue
        if not dsts:
            continue
        if srcs and srcs <= (pool | neutral):
            if srcs & pool:
                closure.add(i)
                neutral -= dsts
                pool |= dsts
            else:
                # 中性传播（常量搬运链）：定义先于杀死——pool 里的旧 FI 值已被覆盖
                pool -= dsts
                neutral |= dsts
        elif not srcs:
            if ins["op"].lower().startswith(("v_mov", "s_mov")):
                pool -= dsts
                neutral |= dsts
            else:
                pool -= dsts
                neutral -= dsts
        else:
            pool -= dsts
            neutral -= dsts
    return closure, stores


def _prune_to_stores(insns, closure, stores):
    """闭包剪枝：只保留位置级后向可达某个 FI 写的成员。
    前向闭包会把“恰好也读 fi_g 的业务代码”（比如业务自读 fi_g.timer_hz 算等待超时）
    一起吞进来；这些指令的数据永远流不进 FI 写，从 FI 写出发的位置级后向切片能把它们剔掉。"""
    defs, _reads = _defuse(insns)
    keep = set(stores)
    work = sorted((r, s) for s in stores for r in insns[s]["srcs"] if r not in COND_REGS)
    seen = set()
    while work:
        reg, pos = work.pop()
        if (reg, pos) in seen:
            continue
        seen.add((reg, pos))
        d = _last_def(defs, reg, pos)
        if d is None or d in keep or d not in closure:
            continue
        keep.add(d)
        work.extend(sorted((r, d) for r in insns[d]["srcs"] if r not in COND_REGS))
    return keep


def _defuse(insns):
    """位置级 def-use：defs[r]=写位置列表，reads[r]=读位置列表（程序序）。"""
    defs, reads = defaultdict(list), defaultdict(list)
    for i, ins in enumerate(insns):
        for r in ins["dsts"]:
            defs[r].append(i)
        for r in ins["srcs"]:
            reads[r].append(i)
    return defs, reads


def _last_def(defs, r, p):
    ws = defs.get(r)
    if not ws:
        return None
    j = bisect.bisect_left(ws, p) - 1
    return ws[j] if j >= 0 else None


def _uses_of(defs, reads, r, d):
    """def d 对寄存器 r 的存活区间内的读取位置集合。"""
    ws = defs[r]
    j = bisect.bisect_right(ws, d)
    kill = ws[j] if j < len(ws) else 1 << 60
    # kill 位置的指令本身仍先读旧值（read-modify-write，如 s_add_i32 s10, s10, 1），所以是 <=
    return [i for i in reads.get(r, []) if d < i <= kill]


def identify_probes(insns, isa, max_back=64):
    """返回 (probe set, breakdown dict, stores list)。识别方法见模块 docstring。"""
    n = len(insns)
    probe = set()
    breakdown = defaultdict(int)

    # ---- 0) 种子 ----
    seed_ids = []
    for i, ins in enumerate(insns):
        k1 = is_timer_seed(ins, isa)
        if k1:
            seed_ids.append((i, k1))
            continue
        if isa == "amdgcn" and ("fi_g@" in ins["text"] or "_ZL4fi_g" in ins["text"] or
                                (ins["op"].lower() == "s_getpc_b64" and
                                 any("fi_g@" in insns[j]["text"] for j in range(i + 1, min(n, i + 8))))):
            seed_ids.append((i, "fi_sym"))

    # ---- 1) 种子接受：联合闭包定候选 FI store；timer/hwid 种子须能沿闭包流到 store（防业务自用计时器误收）----
    fi_sym_seeds = {si for si, k in seed_ids if k == "fi_sym"}
    closure_all, stores_cand = _fwd_closure(insns, isa, {si for si, _k in seed_ids})
    reach = set()
    for s in stores_cand:
        reach |= insns[s]["srcs"] - COND_REGS
    changed = True
    while changed:
        changed = False
        for i in closure_all:
            ins = insns[i]
            d_ = ins["dsts"] - COND_REGS
            if d_ and d_ & reach:
                newr = (ins["srcs"] - COND_REGS) - reach
                if newr:
                    reach |= newr
                    changed = True
    acc_seeds = set(fi_sym_seeds)
    for si, kind in seed_ids:
        if kind == "fi_sym":
            continue  # fi_sym 种子计数推迟到剪枝后（业务 fi_g 链会被剔掉）
        if (insns[si]["dsts"] - COND_REGS) & reach:
            breakdown[kind] += 1
            acc_seeds.add(si)
        else:
            breakdown["rejected_business_timer" if kind == "timer" else "rejected_" + kind] += 1
    closure, stores = _fwd_closure(insns, isa, acc_seeds)
    stores = list(dict.fromkeys(stores))
    probe = _prune_to_stores(insns, closure, stores)
    breakdown["fi_sym"] = len(fi_sym_seeds & probe)
    rej_sym = len(fi_sym_seeds - probe)
    if rej_sym:
        breakdown["rejected_business_fi_sym"] = rej_sym

    # ---- 2) 位置级后向切片：探针 store 的地址/数据源 -> 定义者（输出只被探针用才收）----
    defs, reads = _defuse(insns)

    def _backslice(seed_work, extra_known, counter_key):
        add = set()
        work = sorted(seed_work)
        seen = set()
        while work:
            reg, from_i, depth = work.pop()
            if (reg, from_i) in seen or depth > max_back:
                continue
            seen.add((reg, from_i))
            d = _last_def(defs, reg, from_i)
            if d is None or d in probe or d in add or d in extra_known:
                continue
            outs = insns[d]["dsts"]
            if not outs:
                continue
            ok = True
            for or_ in outs:
                for u in _uses_of(defs, reads, or_, d):
                    if u not in probe and u not in add and u not in extra_known:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            add.add(d)
            breakdown[counter_key] += 1
            for r in sorted(insns[d]["srcs"] - COND_REGS):
                work.append((r, d, depth + 1))
        return add

    # ---- 2+3+4) 联合不动点：地址/数据后向切片 ⇄ 守卫菱形 ⇄ 相邻 waitcnt。
    # 三者互相依赖：FI 字段装载（如 fi_g.cap）既喂记录地址又喂守卫谓词（cap!=0），
    # 共享检查的“所有用途都在探针内”只有在守卫也收编后才能满足，所以必须迭代到稳定。
    label_pos = {}
    for i, ins in enumerate(insns):
        if ins["op"] == "LABEL":
            label_pos[ins["text"].rstrip(":")] = i
    while True:
        grew = False
        for s in stores:
            add = _backslice([(r, s, 0) for r in insns[s]["srcs"]], frozenset(), "addr_calc")
            if add - probe:
                probe |= add
                grew = True
        for s in list(stores):
            for j in range(s - 1, max(0, s - 40), -1):
                ins = insns[j]
                if not ins["op"].lower().startswith("s_cbranch"):
                    continue
                tgt = ins["text"].split()[-1]
                ti = label_pos.get(tgt)
                if ti is None or ti <= j:
                    continue
                region = [k for k in range(j + 1, ti) if insns[k]["op"] != "LABEL"]
                if not region:
                    continue
                # 守卫链：往回找产生条件的 cmp
                cmp_i = None
                for m in range(j - 1, max(0, j - 12), -1):
                    if insns[m]["op"].lower().startswith(("s_cmp", "v_cmp")):
                        cmp_i = m
                        break
                tentative = {j}
                if cmp_i is not None:
                    tentative.add(cmp_i)
                    tentative |= _backslice([(r, cmp_i, 0) for r in insns[cmp_i]["srcs"]],
                                            frozenset({j, cmp_i}), "guard_chain")
                # exec 保存/恢复：分支前的 s_*saveexec/xor exec 与目标处的 s_or/s_xor exec 恢复
                for m in range(j - 1, max(0, j - 8), -1):
                    if "saveexec" in insns[m]["op"].lower() or \
                       (insns[m]["op"].lower().startswith(("s_xor", "s_and")) and "exec" in insns[m]["text"]):
                        tentative.add(m)
                        break
                for m in range(ti, min(n, ti + 4)):
                    if insns[m]["op"].lower().startswith(("s_or", "s_xor")) and "exec" in insns[m]["text"]:
                        tentative.add(m)
                        break
                if all(k in probe or k in tentative for k in region):
                    newones = tentative - probe
                    if newones:
                        probe |= newones
                        breakdown["guard_branch"] += len(newones)
                        grew = True
                    break  # 成功才停；否则继续往前找下一个 cbranch
        # waitcnt：紧邻探针访存的 s_waitcnt（为 FI 字段装载/写回排序插入）。
        # 只有相邻探针本身是访存指令时才剔除——s_memrealtime 这类计时读不产生
        # vmcnt/lgkmcnt 依赖，挨着它的 s_waitcnt 是业务自身的，不能删。
        def _is_mem(k):
            o = insns[k]["op"].lower()
            return o.startswith(AMD_STORE_PREFIX) or o.startswith(AMD_LOAD_PREFIX)
        for i, ins in enumerate(insns):
            if i in probe or ins["op"].lower() != "s_waitcnt":
                continue
            prev_p = any(k in probe and _is_mem(k) for k in range(max(0, i - 2), i) if insns[k]["op"] != "LABEL")
            next_p = any(k in probe and _is_mem(k) for k in range(i + 1, min(n, i + 3)) if insns[k]["op"] != "LABEL")
            if prev_p or next_p:
                probe.add(i)
                breakdown["waitcnt"] += 1
                grew = True
        if not grew:
            break

    breakdown["fi_store"] = len(stores)
    breakdown["probe_data"] = len(probe) - len(acc_seeds & probe) - len(stores) - \
        breakdown["addr_calc"] - breakdown["guard_branch"] - breakdown.get("guard_chain", 0) - breakdown["waitcnt"]
    return probe, dict(breakdown), stores


def _sweep_dead(insns):
    """对称死代码清扫（L0 与 L2 过滤后都用同一规则）：无副作用且目标寄存器在存活区间内
    无人读的指令，迭代剔除到不动点。编译器会留下死装载（如 dispatch packet 里未被消费的
    字段），不清扫会把它们当业务差异。exec 相关与分支/存储/屏障/waitcnt 一律保留。"""
    SIDE = AMD_STORE_PREFIX + AMD_BRANCH_PREFIX + AMD_NO_DST
    live = list(insns)
    while True:
        defs, reads = _defuse(live)
        dead = set()
        for i, ins in enumerate(live):
            o = ins["op"].lower()
            if ins["op"] == "LABEL" or not ins["dsts"] or o.startswith(SIDE):
                continue
            if "exec" in ins["text"]:
                continue
            if any(_uses_of(defs, reads, r, i) for r in ins["dsts"]):
                continue
            dead.add(i)
        if not dead:
            return live
        live = [x for k, x in enumerate(live) if k not in dead]


def _norm_regs(text):
    """寄存器名归一化：按首次出现顺序把 v12/s[4:5]/vcc 等映射成 r0,r1,...（立即数/偏移不动）。
    插桩改变寄存器分配后，同一业务的指令文本不再逐字相等，归一化后才可比。"""
    names = []
    seen = {}

    def _sub(m):
        tok = m.group(0)
        if tok not in seen:
            seen[tok] = f"r{len(seen)}"
        return seen[tok]

    NORM_RE = re.compile(r"[vs]\[\d+:\d+\]|\b[vs]\d+\b|\b(?:vcc_lo|vcc_hi|vcc|scc|exec_lo|exec_hi|exec)\b")
    return NORM_RE.sub(_sub, text)


def lcs_len(a, b):
    """滚动数组 LCS 长度。"""
    if len(a) > len(b):
        a, b = b, a
    prev = [0] * (len(a) + 1)
    for j in range(1, len(b) + 1):
        cur = [0] * (len(a) + 1)
        bj = b[j - 1]
        for i in range(1, len(a) + 1):
            if a[i - 1] == bj:
                cur[i] = prev[i - 1] + 1
            else:
                cur[i] = prev[i] if prev[i] >= cur[i - 1] else cur[i - 1]
        prev = cur
    return prev[len(a)]


def count_blocks(insns, isa):
    labels = sum(1 for i in insns if i["op"] == "LABEL")
    if isa == "amdgcn":
        brs = sum(1 for i in insns if i["op"].lower().startswith(("s_cbranch", "s_branch")))
    else:
        brs = sum(1 for i in insns if i["op"].upper().startswith(("BRA", "BSSY")))
    return labels, brs


def key_subsequence(insns, isa, level="exact"):
    """关键指令子序列。level="exact" 按助记符原样；level="class" 归到语义同义组（AMD_KEY_CLASSES）。"""
    rx = AMD_KEY_RE if isa == "amdgcn" else SASS_KEY_RE
    ops = [i["op"] for i in insns if rx.match(i["op"])]
    return [key_class(o, isa) for o in ops] if level == "class" else ops


def first_order_break(l0_keys, l2_keys):
    """l0_keys 是否为 l2_keys 的子序列；不是则返回第一处失配。"""
    j = 0
    for i, k in enumerate(l0_keys):
        while j < len(l2_keys) and l2_keys[j] != k:
            j += 1
        if j == len(l2_keys):
            return dict(pos=i, expect=k, remaining_l2=l2_keys[max(0, j - 3):j + 3])
        j += 1
    return None


def extract_kernel(insns, kernel_re):
    """按函数标签截取目标 kernel；只匹配到 host stub 时算没找到。"""
    rx = re.compile(kernel_re)
    n = len(insns)
    start = None
    for i, ins in enumerate(insns):
        if ins["op"] == "LABEL" and rx.search(ins["text"]) and not _is_stub_label(ins["text"]):
            start = i
            break
    if start is None:
        return insns, False
    end = n
    for i in range(start + 1, n):
        ins = insns[i]
        if ins["op"] == "LABEL" and not ins["text"].startswith((".L", "$")):
            end = i
            break
    return insns[start:end], True


def _objdump(path, objdump):
    return subprocess.run([objdump, "-d", "--no-show-raw-insn", str(path)],
                          capture_output=True, text=True, timeout=600)


def read_asm(path, kernel_re, isa, gfx: str = "", work_dir=None):
    """path 可以是 .s/.asm/.txt/.sass，也可以是可执行文件/代码对象（自动 llvm-objdump，必要时先解 code object）。"""
    p = Path(path)
    # .sass is assembly TEXT here (weaveprof backend names its AMDGCN -S outputs
    # baseline.sass/instrumented.sass); treating it as ELF made llvm-objdump fail.
    if p.suffix in (".s", ".asm", ".txt", ".dis", ".amdgcn", ".sass"):
        return p.read_text(encoding="utf-8", errors="replace").splitlines(), f"asm:{p.name}"
    objdump = find_tool("FI_LLVM_OBJDUMP", "llvm-objdump")
    if not objdump:
        raise BinSimError(f"需要 llvm-objdump 反汇编 {path}，但没找到。{_searched_hint('llvm-objdump')}；"
                          "设 FI_LLVM_OBJDUMP=<路径> 或 FI_ROCM_LLVM=<ROCm LLVM bin 目录>，"
                          "或先用 hipcc --save-temps 拿到 .s 再传进来")
    is_elf, machine = elf_machine(p)
    if is_elf and machine == EM_X86_64:
        # host ELF：device kernel 不在 .text 里，直接反汇编只会拿到 host stub。
        cos, notes = extract_device_code_objects(p, gfx, work_dir)
        lines = []
        for co in cos:
            o = _objdump(co, objdump)
            lines.extend(o.stdout.splitlines())
        if lines and has_kernel_label(lines, kernel_re):
            return lines, f"code-object:{p.name}({len(cos)} co)"
        raise BinSimError(
            f"{p}: 这是 x86-64 host ELF（e_machine=0x{EM_X86_64:x}），device kernel 在 .hip_fatbin 里；"
            f"llvm-objdump -d 只能反汇编到 host stub，会算出假的相似度（fi_bugs_amd-sk-ds B06/B07）。"
            f"抽 code object 结果：{len(cos)} 个（{'; '.join(n for n in notes if n)[:400]}）。"
            f"解法：把 fi_check 预抽好的 gfx*.o 传给 --l0/--l2，或设 FI_ROCM_LLVM 指向含 "
            f"clang-offload-bundler 与 llvm-objcopy 的目录（gfx={gfx or '未指定，用 --gfx 指定'}），"
            f"或用 hipcc --save-temps 直接拿 .s。")
    out = _objdump(p, objdump)
    lines = out.stdout.splitlines()
    # 判据是“有没有匹配的函数标签”，不是“文本里出现过这个名字”：host stub 的符号名里也含 kernel 名，
    # 按后者判会把 x86 host 指令流当成 AMDGCN 去算相似度（fi_bugs_amd-sk-qwen B04b）。
    if not has_kernel_label(lines, kernel_re):
        cos, notes = extract_device_code_objects(p, gfx, work_dir)
        all_lines = []
        for co in cos:
            all_lines.extend(_objdump(co, objdump).stdout.splitlines())
        if all_lines and has_kernel_label(all_lines, kernel_re):
            return all_lines, f"code-object:{p.name}({len(cos)} co)"
        kind = f"e_machine=0x{machine:x}" if is_elf else "不是 ELF"
        have = labels_in_lines(lines) or labels_in_lines(all_lines)
        raise BinSimError(
            f"{path}: 反汇编里没有匹配 {kernel_re!r} 的函数标签（{kind}；{_searched_hint('llvm-objdump')}）。"
            f"文件里的标签：{have[:12] or '（一个都没有）'}。"
            f"kernel 名要用 AMDGCN 里的实际符号（llvm-objdump --syms 或 nm 可查），host 侧 __device_stub__ 不算；"
            f"抽 code object 的过程记录：{'; '.join(str(n) for n in notes if n)[:400]}。"
            f"注意 roc-obj 在 ROCm 7.x 已弃用（mail-14 实测报 “Can't locate File/Which.pm”），"
            f"优先用 clang-offload-bundler：设 FI_ROCM_LLVM 指向含 clang-offload-bundler 与 llvm-objcopy 的目录，"
            f"或用 hipcc --save-temps 直接把 .s 传进来。stderr: {(out.stderr or '')[:300]}")
    return lines, f"llvm-objdump:{p.name}"


def compare(l0_path, l2_path, kernel_re, isa, dump_slices=None, gfx: str = "", work_dir=None):
    parse = parse_amdgcn if isa == "amdgcn" else parse_sass
    l0_lines, src0 = read_asm(l0_path, kernel_re, isa, gfx, work_dir)
    l2_lines, src2 = read_asm(l2_path, kernel_re, isa, gfx, work_dir)
    ins0, ok0 = extract_kernel(parse(l0_lines), kernel_re)
    ins2, ok2 = extract_kernel(parse(l2_lines), kernel_re)
    # 标签没匹配上时 extract_kernel 会退回“用整个文件”，多 kernel 产物里会把无关 kernel 混进相似度
    # 且不吭声（fi_bugs_amd-sk-qwen B04b/B04d：l0_kernel_found=false 但 rc=0，sim_strict=0.044）。
    for side, ok, src, lines in (("l0", ok0, src0, l0_lines), ("l2", ok2, src2, l2_lines)):
        if not ok:
            have = labels_in_lines(lines)
            raise BinSimError(
                f"{side} 侧（{src}）里没有匹配 --kernel {kernel_re!r} 的函数标签，"
                f"不能拿整个文件当目标 kernel。文件里的标签：{have[:12] or '（一个都没有）'}。"
                f"修法：--kernel 传 AMDGCN 里的实际符号名（正则）；两臂 kernel 不同名时"
                f"用能同时匹配两者的正则（如 'ring_pipeline'），或先统一命名。")
    lb0, br0 = count_blocks(ins0, isa)
    lb2, br2 = count_blocks(ins2, isa)
    ins0 = [i for i in ins0 if i["op"] != "LABEL"]
    ins2 = [i for i in ins2 if i["op"] != "LABEL"]
    if not ins0 or not ins2:
        raise BinSimError("Expected nonempty baseline and instrumented instruction sequences")
    raw_lcs = lcs_len([i["op"] for i in ins0], [i["op"] for i in ins2])
    probe, breakdown, stores = identify_probes(ins2, isa)
    ins2f = [i for k, i in enumerate(ins2) if k not in probe]

    ins0s = _sweep_dead(ins0)
    ins2s = _sweep_dead(ins2f)
    t0 = [i["text"] for i in ins0s]
    t2 = [i["text"] for i in ins2s]
    n0 = [_norm_regs(i["text"]) for i in ins0s]
    n2 = [_norm_regs(i["text"]) for i in ins2s]
    o0 = [i["op"] for i in ins0s]
    o2 = [i["op"] for i in ins2s]
    sim_strict = lcs_len(t0, t2) / max(1, len(t0))
    sim_norm = lcs_len(n0, n2) / max(1, len(n0))
    sim_opcode = lcs_len(o0, o2) / max(1, len(o0))
    k0 = key_subsequence(ins0s, isa)
    k2 = key_subsequence(ins2s, isa)
    brk = first_order_break(k0, k2)
    key_lcs = lcs_len([x for x in k0], [x for x in k2]) / max(1, len(k0))
    # 同义组口径：flat_load/global_load/buffer_load 这类“同一语义不同助记符”不算顺序破坏
    kc0 = key_subsequence(ins0s, isa, "class")
    kc2 = key_subsequence(ins2s, isa, "class")
    brk_c = first_order_break(kc0, kc2)
    key_lcs_class = lcs_len(kc0, kc2) / max(1, len(kc0))

    res = dict(kernel=kernel_re, isa=isa,
               sources=dict(l0=src0, l2=src2, l0_kernel_found=ok0, l2_kernel_found=ok2),
               l0=dict(insns=len(ins0), labels=lb0, branches=br0),
               l2=dict(insns=len(ins2), probe_removed=len(probe), insns_after=len(ins2f), labels=lb2, branches=br2),
               probe_breakdown=breakdown, probe_stores=len(stores),
               assembly_change_rate=1 - raw_lcs / len(ins0),
               sim_strict=round(sim_strict, 6), sim_norm=round(sim_norm, 6), sim_opcode=round(sim_opcode, 6),
               block_delta=dict(labels=lb2 - lb0, branches=br2 - br0),
               dead_swept=dict(l0=len(ins0) - len(ins0s), l2_after=len(ins2f) - len(ins2s)),
               key_order=dict(n_l0_keys=len(k0), preserved=brk is None, first_break=brk,
                              key_lcs=round(key_lcs, 6),
                              n_l0_keys_class=len(kc0), preserved_class=brk_c is None,
                              first_break_class=brk_c, key_lcs_class=round(key_lcs_class, 6),
                              class_note="preserved/key_lcs 按助记符精确匹配（留作审计）；"
                                         "preserved_class/key_lcs_class 按语义同义组匹配"
                                         "（global/flat/buffer_load 视为同一语义，见 knowledge/cards/mi308x.json limits）"))
    if dump_slices:
        dd = Path(dump_slices)
        dd.mkdir(parents=True, exist_ok=True)
        idxs = sorted(probe)
        with open(dd / "probe_slices.txt", "w", encoding="utf-8") as f:
            f.write(f"剔除探针指令 {len(idxs)} 条（breakdown={json.dumps(breakdown)}）；每处给前后 2 行上下文\n")
            shown = set()
            for i in idxs:
                if i in shown:
                    continue
                lo, hi = max(0, i - 2), min(len(ins2), i + 3)
                f.write(f"\n--- insn {i} ---\n")
                for k in range(lo, hi):
                    mark = ">>" if k in probe else "  "
                    f.write(f"{mark} {k:6d}: {ins2[k]['raw']}\n")
                    shown.add(k)
        res["slices_dump"] = str(dd / "probe_slices.txt")
    return res


def selftest():
    """合成 AMDGCN 片段：L0 是 7 条普通指令；L2 插入 s_memrealtime/s_getreg + 数据搬运 + 地址计算 + FI 写。"""
    l0 = """
kernel_demo:
\ts_load_dwordx2 s[0:1], s[4:5], 0x0
\tv_mov_b32 v0, 0
\ts_waitcnt vmcnt(0)
\tglobal_load_dword v1, v0, s[0:1]
\tv_add_f32 v1, v1, v1
\tglobal_store_dword v0, v1, s[0:1]
\ts_endpgm
next_func:
\ts_nop 0
"""
    l2 = """
.Ltmp0:
kernel_demo:
\ts_load_dwordx2 s[0:1], s[4:5], 0x0
\tv_mov_b32 v0, 0
\ts_waitcnt vmcnt(0)
\ts_memrealtime s[10:11]
\ts_getreg_b32 s12, hwreg(HW_ID)
\tv_mov_b32 v8, s10
\tv_mov_b32 v9, s11
\tv_mov_b32 v10, s12
\ts_load_dwordx2 s[14:15], s[6:7], 0x0
\ts_lshl_b64 s[16:17], s[14:15], 4
\tglobal_store_dwordx4 s[16:17], v[8:11], off
\tglobal_load_dword v1, v0, s[0:1]
\tv_add_f32 v1, v1, v1
\tglobal_store_dword v0, v1, s[0:1]
\ts_endpgm
"""
    with tempfile.TemporaryDirectory() as td:
        p0, p2 = Path(td) / "l0.s", Path(td) / "l2.s"
        p0.write_text(l0, encoding="utf-8")
        p2.write_text(l2, encoding="utf-8")
        res = compare(str(p0), str(p2), "kernel_demo", "amdgcn")
        # 探针 = s_memrealtime, s_getreg, v_mov×3, s_load_dwordx2 s[14:15], s_lshl_b64, global_store_dwordx4 = 8
        assert res["l2"]["probe_removed"] == 8, res
        assert res["probe_breakdown"]["timer"] == 1 and res["probe_breakdown"]["hwid"] == 1, res
        assert res["probe_breakdown"]["fi_store"] == 1, res
        assert res["sim_strict"] == 1.0 and res["sim_opcode"] == 1.0 and res["sim_norm"] == 1.0, res
        assert res["key_order"]["preserved"] is True, res
        assert res["l0"]["insns"] == 7 and res["l2"]["insns_after"] == 7, res

        # ELF 头识别：host stub（x86-64）与 device code object（AMDGPU）必须分得开。
        def _elf(machine):
            return (b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\x00" * 8
                    + (2).to_bytes(2, "little") + machine.to_bytes(2, "little") + b"\x00" * 64)
        host = Path(td) / "host_exe"
        dev = Path(td) / "device.co"
        host.write_bytes(_elf(EM_X86_64))
        dev.write_bytes(_elf(EM_AMDGPU))
        assert elf_machine(host) == (True, EM_X86_64), elf_machine(host)
        assert elf_machine(dev) == (True, EM_AMDGPU), elf_machine(dev)
        assert elf_machine(p0) == (False, None), elf_machine(p0)   # .s 文本不是 ELF
        # host ELF 必须报“抽 code object”的可执行诊断，并且是 Exception 子类：
        # 旧实现 raise SystemExit，fi_check 的 except Exception 抓不到，检查器直接死掉且不写 check.json。
        # 用假的 llvm-objdump（空输出）把这条路径在任何机器上都变得可测。
        fake = Path(td) / "fake_llvm_bin"
        fake.mkdir(exist_ok=True)
        stub = fake / "llvm-objdump"
        stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        stub.chmod(0o755)
        prev = os.environ.get("FI_LLVM_OBJDUMP")
        os.environ["FI_LLVM_OBJDUMP"] = str(stub)
        try:
            assert find_tool("FI_LLVM_OBJDUMP", "llvm-objdump") == str(stub.resolve())
            try:
                read_asm(str(host), "ring_pipeline", "amdgcn", "gfx942", str(Path(td) / "co"))
                raise AssertionError("host ELF 没有被拦下来")
            except BinSimError as e:
                msg = str(e)
                assert "host ELF" in msg and ".hip_fatbin" in msg, msg
            except Exception as e:                               # noqa: BLE001 - 必须落到 Exception 分支
                raise AssertionError(f"抛出的不是 BinSimError: {type(e).__name__}: {e}")
            assert not isinstance(BinSimError("x"), SystemExit)
            caught = False
            try:
                read_asm(str(host), "ring_pipeline", "amdgcn", "gfx942", str(Path(td) / "co"))
            except Exception:                                    # noqa: BLE001
                caught = True
            assert caught, "fi_check 的 except Exception 必须能抓到"
            # AMDGPU ELF 走正常反汇编路径；kernel 找不到时报的是符号/工具问题，不是 host ELF 那段话。
            try:
                read_asm(str(dev), "ring_pipeline", "amdgcn", "gfx942", str(Path(td) / "co"))
                raise AssertionError("空反汇编没有被拦下来")
            except BinSimError as e:
                assert "没有匹配" in str(e) and "host ELF" not in str(e), str(e)
                assert "e_machine=0xe0" in str(e), str(e)   # 真器件 ELF 的 machine 号必须报出来
        finally:
            if prev is None:
                os.environ.pop("FI_LLVM_OBJDUMP", None)
            else:
                os.environ["FI_LLVM_OBJDUMP"] = prev
        # ---- llvm-objdump 形态：函数标签是 “<addr> <symbol>:”，指令行是 “<addr>: <insn>”
        objdump_txt = [
            "",
            "build/ring_l0.o:	file format elf64-amdgpu",
            "",
            "Disassembly of section .text:",
            "",
            "0000000000207a00 <_Z28__device_stub__ring_pipeline>:",
            "  207a00: s_endpgm",
            "",
            "0000000000207b00 <_Z13ring_pipelinePf>:",
            "  207b00: s_load_dwordx2 s[0:1], s[4:5], 0x0",
            "  207b08: flat_load_dword v1, v0, s[0:1]",
            "  207b10: s_waitcnt vmcnt(0)",
            "  207b14: v_add_f32 v1, v1, v1",
            "  207b18: s_endpgm",
        ]
        ins = parse_amdgcn(objdump_txt)
        labs = [i["text"] for i in ins if i["op"] == "LABEL"]
        assert labs == ["_Z28__device_stub__ring_pipeline:", "_Z13ring_pipelinePf:"], labs
        assert [i["op"] for i in ins if i["op"] != "LABEL"] == [
            "s_endpgm", "s_load_dwordx2", "flat_load_dword", "s_waitcnt", "v_add_f32", "s_endpgm"], ins
        # host stub 不算命中：只有 stub 标签时必须判“找不到”（B04b 的假成功就是这么来的）
        assert has_kernel_label(objdump_txt, "__device_stub__ring_pipeline") is False
        assert has_kernel_label(objdump_txt, "ring_pipeline") is True
        got, ok = extract_kernel(ins, "ring_pipeline")
        assert ok is True and len([i for i in got if i["op"] != "LABEL"]) == 5, (ok, got)
        stub_only = objdump_txt[:8]
        assert has_kernel_label(stub_only, "ring_pipeline") is False, stub_only
        assert extract_kernel(parse_amdgcn(stub_only), "ring_pipeline")[1] is False
        assert labels_in_lines(stub_only) == [], labels_in_lines(stub_only)

        # ---- compare()：kernel 标签缺失必须报错，不能静默用整个文件
        try:
            compare(str(p0), str(p2), "no_such_kernel", "amdgcn")
            raise AssertionError("kernel 标签缺失时 compare() 没有报错")
        except BinSimError as e:
            assert "no_such_kernel" in str(e) and "kernel_demo:" in str(e), str(e)

        # ---- 同义组：基线 flat_load_dword、插桩版 global_load_dwordx2 不算顺序破坏（B06）
        q0, q2 = Path(td) / "syn_l0.s", Path(td) / "syn_l2.s"
        q0.write_text("kernel_demo:\n\tflat_load_dword v1, v0, s[0:1]\n\ts_waitcnt vmcnt(0)\n"
                      "\tv_mfma_f32_16x16x16f16 a[0:3], v1, v1, a[0:3]\n\ts_endpgm\n", encoding="utf-8")
        q2.write_text("kernel_demo:\n\tglobal_load_dword v1, v0, s[0:1]\n\ts_waitcnt vmcnt(0)\n"
                      "\tv_mfma_f32_16x16x16f16 a[0:3], v1, v1, a[0:3]\n\ts_endpgm\n", encoding="utf-8")
        rs = compare(str(q0), str(q2), "kernel_demo", "amdgcn")
        ko = rs["key_order"]
        assert ko["preserved"] is False and ko["first_break"]["expect"] == "flat_load_dword", ko
        assert ko["preserved_class"] is True and ko["key_lcs_class"] == 1.0, ko
        assert key_class("flat_load_dword", "amdgcn") == key_class("global_load_dwordx2", "amdgcn") == "mem_load"
        assert key_class("LDGSTS", "sass") == "async_copy" and key_class("HGMMA.64", "sass") == "mma"
        print("SELFTEST PASS（含 host ELF 识别、objdump 标签、同义组与 BinSimError 可捕获性）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l0", help="L0（FI_LEVEL=0）汇编/反汇编文本或可执行文件")
    ap.add_argument("--l2", help="插桩版汇编/反汇编文本或可执行文件")
    ap.add_argument("--kernel", default=".*", help="kernel 名正则")
    ap.add_argument("--isa", choices=("amdgcn", "sass"), default="amdgcn")
    ap.add_argument("-o", "--out", help="输出 JSON；缺省打印到 stdout")
    ap.add_argument("--dump-slices", help="把剔除的探针指令（带上下文）写到该目录，供人工抽查")
    ap.add_argument("--gfx", default=os.environ.get("FI_AMD_GFX", "gfx942"),
                    help="从 host ELF 的 .hip_fatbin 里挑这个 target 的 code object")
    ap.add_argument("--rocm-llvm", help="ROCm LLVM bin 目录（llvm-objdump/clang-offload-bundler/llvm-objcopy）")
    ap.add_argument("--work-dir", help="抽 code object 的中间产物写这里；默认系统临时目录")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.l0 and a.l2):
        ap.error("需要 --l0 与 --l2")
    if a.rocm_llvm:
        set_rocm_llvm(a.rocm_llvm)
    try:
        res = compare(a.l0, a.l2, a.kernel, a.isa, a.dump_slices, a.gfx, a.work_dir)
    except BinSimError as e:
        print(f"fi_binsim_hip FAIL: {e}", file=sys.stderr)
        return 2
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
        print("写出", a.out)
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
