# -*- coding: utf-8 -*-
"""NVIDIA SASS 相似度与 fast-instrument 探针识别。

输入两个已编译产物（可执行文件、.so、.cubin、.fatbin、.o，或 Triton 编译目录中的此类文件），
自动用 cuobjdump -sass（cubin 可用 nvdisasm）导出匹配 kernel，然后：
  * 用逐指令严格文本 LCS 和 opcode-only LCS 对齐；
  * 统计基本块数变化；
  * 识别 SR_GLOBALTIMER*/SR_CLOCK*/SR_SMID/SR_VIRTUALSMID 的 S2R/S2UR/CS2R；
  * 识别写 FI 缓冲的 128-bit STG，以及从 STG 地址操作数回溯出的地址计算；
  * 比较 HGMMA/UTC*MMA/UTMALDG/UTMASTG/SYNCS/BAR/LDGSTS/DEPBAR 的相对顺序。

用法：
  python3 -I fi_binsim.py L0.out FI.out --kernel ring_pipeline -o binsim.json
  python3 -I fi_binsim.py triton_cache_dir_l0 triton_cache_dir_fi --kernel persistent_mm \
      --arch sm_103a -o binsim.json --disasm-dir disasm
  python3 -I fi_binsim.py --selftest

输出 JSON 是证据，不会替调用方决定总体验收；fi_check.py 用其中 sim_strict、sim_opcode、
probe_counts、basic_blocks 和 key_instruction_order_preserved 做阈值判定。
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TOOL = "fast-instrument fi_binsim"
VERSION = "1.0"
DEFAULT_ARCH = "sm_103a"
ARTIFACT_SUFFIXES = {
    ".so", ".cubin", ".fatbin", ".o", ".a", ".exe", ".elf",
}
RAW_SASS_SUFFIXES = {".sass", ".cuobjdump", ".asm"}
PTX_SUFFIXES = {".ptx"}

INSN_RE = re.compile(r"^\s*/\*(?P<addr>[0-9a-fA-F]+)\*/\s+(?P<body>.*?)\s*;\s*(?:/\*\s*(?P<enc>[0-9a-fA-Fx]+)\s*\*/)?\s*$")
FUNC_RE = re.compile(r"^\s*Function\s*:\s*(\S+)")
ARCH_RE = re.compile(r"^\s*arch\s*=\s*(sm_\w+)")
REG_RE = re.compile(r"\b(?:U?R\d+|P\d+|B\d+|SB\d+)\b")
DEST_RE = re.compile(r"^\s*(?:@!?U?P\d+\s+)?(?:U?R\d+|P\d+|B\d+)\b")
TIMER_SR_RE = re.compile(r"SR_(GLOBALTIMER(?:LO|HI)?|CLOCK(?:LO|HI)?|SMID|VIRTUALSMID)\b", re.I)
STG_RE = re.compile(r"^(?:@!?U?P\d+\s+)?STG\b", re.I)
# FILITE02 的 16B 记录写在源码里是 st.global.cs.v4.u32（recorders/cuda_cpp/fi_lite.cuh、recorders/triton/fi_triton.py），
# 落到 SASS 必然带 .128。旧正则的第二分支“方括号地址 + 末尾源寄存器”会把输出 tile 的
# STG.E.U16/STG.E.64 也算进来：bf16 输出写的源寄存器恰好与计时器污点链重合时就被误报成探针
# （fi_bugs_h20-sk-ds 第 5 条实测：真探针 @P0 STG.E.EF.128 漏掉，两条 STG.E.U16 被当成探针）。
STORE_128_RE = re.compile(r"^(?:@!?U?P\d+\s+)?STG\b.*\.128\b", re.I)
BRANCH_TARGET_OPS = {"BRA", "BRX", "JMP", "CALL", "BSSY"}
FLOW_END_OPS = {"BRA", "BRX", "JMP", "EXIT", "RET", "BSSY"}
KEY_CLASSES = ("HGMMA", "UTC*MMA", "UTMALDG", "UTMASTG", "SYNCS", "BAR", "LDGSTS", "DEPBAR")
TIMER_READ_OPS = {"S2R", "S2UR", "CS2R"}
# 只改寄存器编号的搬运/打包指令。Triton 3.6 在 Hopper/Blackwell 上把 %globaltimer 读出的值先经
# MOV/PRMT/SHF 组装成 16B 记录再写 FI 缓冲，此时 STG.128 的源向量起点已不是计时器寄存器本身，
# 只按“源向量起点 == 计时器目标寄存器”判会漏（fi_bugs_h20-sk-glm 第 3 条）。
TAINT_OPS = {"MOV", "PRMT", "SHF", "SEL", "LOP3", "IADD3", "IMAD", "LEA", "BMSK", "F2F", "I2I"}
TAINT_MAX_HOPS = 8


def _run(argv: Sequence[str], timeout: int = 120, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(list(argv), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as e:
        return subprocess.CompletedProcess(list(argv), 127, "", str(e))
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(list(argv), 124, e.stdout or "", f"timeout after {timeout}s")


def bitset_lcs(a: Sequence[object], b: Sequence[object]) -> int:
    """精确 LCS 长度；用 Python 大整数位并行，避免 O(nm) Python 循环。"""
    if not a or not b:
        return 0
    pos: Dict[object, int] = {}
    for j, token in enumerate(b):
        pos[token] = pos.get(token, 0) | (1 << j)
    s = 0
    for token in a:
        x = pos.get(token, 0) | s
        s = x & ~(x - ((s << 1) | 1))
    return int(s.bit_count())


@dataclass
class Inst:
    offset: int
    offset_text: str
    body: str
    mnemonic: str
    opcode: str
    enc: str = ""
    block: int = -1
    line_no: int = 0

    @property
    def exact_token(self) -> str:
        return re.sub(r"\s+", " ", self.body.strip()).upper()

    @property
    def strict_token(self) -> str:
        # strict 保留 opcode、谓词、寻址模式、偏移和立即数，只把编译器会整体重编号的寄存器族
        # 归一成 R#/UR#/P#。这样同一条语义指令不会仅因插桩多占两个寄存器而被当成不同指令；
        # 原始逐字值另报 sim_exact_raw，供人审计。
        s = self.exact_token
        s = re.sub(r"\bUR\d+\b", "UR#", s)
        s = re.sub(r"\bR\d+\b", "R#", s)
        s = re.sub(r"\bP\d+\b", "P#", s)
        s = re.sub(r"\bSB\d+\b", "SB#", s)
        s = re.sub(r"\bB\d+\b", "B#", s)
        s = re.sub(r"0X([0-9A-F]+)", lambda m: "0x" + m.group(1).lower(), s)
        return s

    @property
    def opcode_token(self) -> str:
        return self.mnemonic.upper()


@dataclass
class Kernel:
    raw_name: str
    name: str
    arch: str
    instructions: List[Inst]
    source: str
    section_index: int = 0
    resources: Dict[str, object] = field(default_factory=dict)
    blocks: List[List[int]] = field(default_factory=list)
    block_terminators: int = 0

    def __post_init__(self) -> None:
        self.compute_blocks()

    @property
    def offsets(self) -> Set[int]:
        return {x.offset for x in self.instructions}

    def compute_blocks(self) -> None:
        if not self.instructions:
            self.blocks = []
            self.block_terminators = 0
            return
        off_to_idx = {x.offset: i for i, x in enumerate(self.instructions)}
        leaders = {self.instructions[0].offset}
        terminators = 0
        for i, ins in enumerate(self.instructions):
            op = ins.opcode
            if op in BRANCH_TARGET_OPS:
                m = re.search(r"\b0x([0-9a-fA-F]+)\b", ins.body)
                if m:
                    target = int(m.group(1), 16)
                    if target in off_to_idx:
                        leaders.add(target)
                if i + 1 < len(self.instructions):
                    leaders.add(self.instructions[i + 1].offset)
                terminators += 1
            elif op in FLOW_END_OPS:
                if i + 1 < len(self.instructions):
                    leaders.add(self.instructions[i + 1].offset)
                terminators += 1
        ordered = sorted(leaders, key=lambda off: off_to_idx[off])
        blocks: List[List[int]] = []
        leader_set = set(ordered)
        current: List[int] = []
        block_id = -1
        for ins in self.instructions:
            if ins.offset in leader_set:
                if current:
                    blocks.append(current)
                current = []
                block_id += 1
            ins.block = block_id
            current.append(ins.offset)
        if current:
            blocks.append(current)
        self.blocks = blocks
        self.block_terminators = terminators


def parse_sass(text: str, source: str) -> List[Kernel]:
    kernels: List[Kernel] = []
    arch = ""
    cur_raw: Optional[str] = None
    cur_arch = ""
    cur_insts: List[Inst] = []
    section = 0

    def flush() -> None:
        nonlocal cur_raw, cur_arch, cur_insts
        if cur_raw is not None and cur_insts:
            kernels.append(Kernel(cur_raw, cur_raw, cur_arch, list(cur_insts), source, section))
        cur_raw = None
        cur_arch = ""
        cur_insts = []

    for line_no, line in enumerate(text.splitlines(), 1):
        am = ARCH_RE.match(line)
        if am:
            arch = am.group(1)
            cur_arch = arch
            continue
        fm = FUNC_RE.match(line)
        if fm:
            flush()
            cur_raw = fm.group(1)
            cur_arch = arch
            section += 1
            continue
        im = INSN_RE.match(line)
        if im and cur_raw is not None:
            body = im.group("body").strip()
            mm = re.match(r"(?:@!?U?P\d+\s+)?([A-Za-z][A-Za-z0-9_.]*)", body)
            if not mm:
                continue
            mnemonic = mm.group(1).upper()
            opcode = mnemonic.split(".", 1)[0]
            cur_insts.append(Inst(int(im.group("addr"), 16), im.group("addr"), body, mnemonic,
                                  opcode, im.group("enc") or "", line_no=line_no))
    flush()
    # 大型 fatbin 可能对同一函数同时含多个架构。先全部保留，resolve 时优先目标 arch。
    return kernels


def demangle_names(names: Iterable[str]) -> Dict[str, str]:
    names = list(dict.fromkeys(names))
    exe = shutil.which("c++filt") or "/usr/bin/c++filt"
    if not names or not os.path.exists(exe):
        return {n: n for n in names}
    p = _run([exe, *names], timeout=30)
    vals = p.stdout.splitlines()
    if len(vals) != len(names):
        return {n: n for n in names}
    return dict(zip(names, vals))


def _sass_from_file(path: Path, out_dir: Path, arch: str, cuobjdump: str, nvdisasm: str) -> Tuple[str, List[str]]:
    notes: List[str] = []
    suffix = path.suffix.lower()
    if suffix in RAW_SASS_SUFFIXES or path.name.endswith(".cuobjdump"):
        return path.read_text(encoding="utf-8", errors="replace"), notes
    tool_path = path
    if suffix in PTX_SUFFIXES:
        ptxas = shutil.which("ptxas") or str(Path(cuobjdump).with_name("ptxas"))
        cubin = out_dir / (path.stem + ".from_ptx.cubin")
        p = _run([ptxas, "-arch=" + arch, str(path), "-o", str(cubin)], timeout=180)
        if p.returncode != 0:
            raise RuntimeError(f"ptxas 失败: {path}: rc={p.returncode}: {(p.stderr or p.stdout)[:800]}")
        notes.append(f"ptxas -arch={arch} {path} -> {cubin}")
        tool_path = cubin
    if cuobjdump and os.path.exists(cuobjdump):
        p = _run([cuobjdump, "-sass", str(tool_path)], timeout=180)
        if p.returncode == 0 and "Function " in p.stdout:
            return p.stdout, notes
        notes.append(f"cuobjdump -sass {tool_path}: rc={p.returncode}: {(p.stderr or p.stdout)[:400]}")
    if nvdisasm and os.path.exists(nvdisasm) and tool_path.suffix.lower() == ".cubin":
        p = _run([nvdisasm, "-c", str(tool_path)], timeout=180)
        if p.returncode != 0:
            raise RuntimeError(f"nvdisasm 失败: {tool_path}: rc={p.returncode}: {(p.stderr or p.stdout)[:800]}")
        notes.append(f"fallback nvdisasm -c {tool_path}")
        return p.stdout, notes
    raise RuntimeError("找不到可用的反汇编结果；" + "；".join(notes[-2:]))


def discover_artifact(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    found: List[Path] = []
    for p in path.rglob("*"):
        if not p.is_file():
            continue
        s = p.suffix.lower()
        if s in ARTIFACT_SUFFIXES | RAW_SASS_SUFFIXES | PTX_SUFFIXES or os.access(p, os.X_OK):
            # Triton 缓存里还有 JSON/LLIR，不会含 SASS，跳过以缩短目录扫描。
            if s in {".json", ".llir", ".ttir", ".ttgir", ".py", ".pyc"}:
                continue
            found.append(p)
    return sorted(found, key=lambda x: (len(x.parts), str(x)))


def resolve_kernel(artifact: Path, kernel_regex: str, arch: str, work_dir: Path,
                   label: str, cuobjdump: str, nvdisasm: str) -> Tuple[Kernel, str, Dict[str, object]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    rx = re.compile(kernel_regex)
    candidates: List[Tuple[Kernel, str, Dict[str, object]]] = []
    errors: List[Dict[str, str]] = []
    files = discover_artifact(artifact)
    if not files:
        raise FileNotFoundError(f"{artifact}: 目录中没找到可反汇编产物")
    for i, f in enumerate(files):
        try:
            sass, notes = _sass_from_file(f, work_dir, arch, cuobjdump, nvdisasm)
        except Exception as e:  # 一个缓存目录里可能有很多非目标文件
            errors.append({"path": str(f), "error": str(e)})
            continue
        ks = parse_sass(sass, str(f))
        if not ks:
            continue
        dmap = demangle_names(k.raw_name for k in ks)
        for k in ks:
            k.name = dmap.get(k.raw_name, k.raw_name)
            if rx.search(k.raw_name) or rx.search(k.name):
                candidates.append((k, sass, {"path": str(f), "notes": notes}))
    if not candidates:
        discovered = []
        for _, sass, meta in candidates:
            discovered.extend([])
        raise RuntimeError(f"{label} 产物 {artifact} 中没有匹配 /{kernel_regex}/ 的 kernel；"
                           f"扫描 {len(files)} 个文件，反汇编失败 {len(errors)} 个")
    # 优先精确目标 arch、指令更多、源码路径排序稳定。
    def score(item: Tuple[Kernel, str, Dict[str, object]]) -> Tuple[int, int, str]:
        k = item[0]
        return (1 if arch and k.arch == arch else 0, len(k.instructions), item[2]["path"])
    candidates.sort(key=score, reverse=True)
    k, sass, meta = candidates[0]
    disasm_path = work_dir / f"{label}.{re.sub(r'[^A-Za-z0-9_.-]+', '_', k.name)[:80]}.sass"
    disasm_path.write_text(sass, encoding="utf-8")
    meta = dict(meta)
    meta["disassembly"] = str(disasm_path)
    meta["matched_candidates"] = len(candidates)
    meta["scan_errors"] = errors[:20]
    return k, sass, meta


def _reg_base(token: str) -> Optional[Tuple[str, int]]:
    m = re.match(r"U?R(\d+)", token, re.I)
    if not m:
        return None
    return ("UR" if token.upper().startswith("UR") else "R", int(m.group(1)))


def _regs_in(text: str) -> Set[str]:
    return {m.group(0).upper().replace("R", "R", 1) for m in re.finditer(r"\bU?R\d+\b", text, re.I)}


def _dest_regs(ins: Inst) -> Set[str]:
    body = re.sub(r"^\s*@!?U?P\d+\s+", "", ins.body)
    m = re.match(r"^([^\s,]+)\s+((?:U?R\d+)(?:\.\w+)?)", body, re.I)
    if not m:
        return set()
    if ins.opcode.startswith(("ST", "STG", "RED", "ATOM")):
        return set()
    reg = m.group(2).split(".", 1)[0].upper()
    out = {reg}
    # 64-bit 写常用 base register 表示一对。
    if re.search(r"\.(?:64|U64|S64)\b", ins.mnemonic) and reg.startswith("R"):
        out.add("R" + str(int(reg[1:]) + 1))
    return out


def _source_regs_in_operand(operand: str) -> Set[str]:
    return _regs_in(operand)


def _timer_taint(k: Kernel) -> Dict[str, Dict[str, object]]:
    """按地址顺序线性扫一遍，跟踪“值来自计时器读”的寄存器。

    只在寄存器到寄存器的搬运/打包指令上传播，任何 load、S2R（非计时器）、分支或函数返回都会
    把目标寄存器的污点清掉，因此业务 STG.128 不会仅因为复用了编号相近的寄存器就被当成探针。
    返回值形如 {"R12": {"kind": "globaltimer", "via": ["MOV R4->R12"]}}，via 是可审计的证据链。
    """
    taint: Dict[str, Dict[str, object]] = {}
    for ins in k.instructions:
        if ins.opcode in TIMER_READ_OPS:
            m = TIMER_SR_RE.search(ins.body)
            dm = re.match(r"^(?:@!?U?P\d+\s+)?(?:S2R|S2UR|CS2R)\s+(U?R\d+)", ins.body, re.I)
            if m and dm:
                sr = m.group(1).upper()
                kind = ("globaltimer" if sr.startswith("GLOBALTIMER")
                        else "clock64" if sr.startswith("CLOCK") else "smid")
                regs = {dm.group(1).upper()}
                if kind != "smid" and dm.group(1).upper().startswith("R"):
                    regs.add("R" + str(int(dm.group(1)[1:]) + 1))
                for r in regs:
                    taint[r] = {"kind": kind, "via": []}
                continue
        dst = _dest_regs(ins)
        if not dst:
            continue
        if "WIDE" in ins.mnemonic.upper():
            # IMAD.WIDE/LEA.WIDE 写寄存器对。
            dst |= {"R" + str(int(d[1:]) + 1) for d in dst if re.match(r"^R\d+$", d)}
        if ins.opcode in TAINT_OPS:
            srcs = _regs_in(ins.body) - dst
            hits = [x for x in sorted(srcs, key=lambda r: _reg_base(r) or ("Z", 9999)) if x in taint]
            if hits:
                base = taint[hits[0]]
                via = list(base["via"]) + [f"{ins.mnemonic} {','.join(hits)}->{','.join(sorted(dst))}"]
                for d in dst:
                    taint[d] = {"kind": base["kind"], "via": via[-TAINT_MAX_HOPS:]}
            else:
                for d in dst:
                    taint.pop(d, None)
        else:
            for d in dst:
                taint.pop(d, None)
    return taint


def identify_probes(k: Kernel) -> Dict[str, object]:
    reads: List[Dict[str, object]] = []
    timer_regs: Dict[str, Set[str]] = {"globaltimer": set(), "clock64": set(), "smid": set()}
    timer_bases: Dict[str, Set[str]] = {"globaltimer": set(), "clock64": set(), "smid": set()}
    for ins in k.instructions:
        m = TIMER_SR_RE.search(ins.body)
        if not m or ins.opcode not in {"S2R", "S2UR", "CS2R"}:
            continue
        sr = m.group(1).upper()
        if sr.startswith("GLOBALTIMER"):
            kind = "globaltimer"
        elif sr.startswith("CLOCK"):
            kind = "clock64"
        else:
            kind = "smid"
        dm = re.match(r"^(?:@!?P\d+\s+)?(?:S2R|S2UR|CS2R)\s+(U?R\d+)", ins.body, re.I)
        regs = {dm.group(1).upper()} if dm else set()
        if kind in {"globaltimer", "clock64"} and dm and dm.group(1).upper().startswith("R"):
            regs.add("R" + str(int(dm.group(1)[1:]) + 1))
        timer_regs[kind].update(regs)
        if dm:
            timer_bases[kind].add(dm.group(1).upper())
        reads.append({"offset": ins.offset_text, "line": ins.line_no, "opcode": ins.mnemonic,
                      "special_register": "SR_" + sr, "kind": kind, "dest": sorted(regs), "block": ins.block})
    stores: List[Dict[str, object]] = []
    rejected_narrow: List[Dict[str, object]] = []
    taint = _timer_taint(k)
    for i, ins in enumerate(k.instructions):
        if not STG_RE.search(ins.body):
            continue
        if not STORE_128_RE.search(ins.body):
            # 不是 128-bit 就不是 16B 记录写。若它的源寄存器确实与计时器链重合，把这次拒绝
            # 写进 rejected_narrow_stores，让“为什么没算探针”可审计。
            tail0 = ins.body.rsplit(",", 1)[-1].strip().upper()
            regs0 = _regs_in(tail0)
            base0 = min((r for r in regs0 if re.match(r"^R\d+$", r)), key=lambda x: int(x[1:]), default="")
            if base0 and (base0 in taint or any(base0 in b for b in timer_bases.values())):
                rejected_narrow.append({"offset": ins.offset_text, "line": ins.line_no, "opcode": ins.mnemonic,
                                        "source_vector": sorted(regs0),
                                        "reason": "非 128-bit STG，不是 FILITE02 的 16B 记录写"})
            continue
        # STG.128 的源向量位于最后一个逗号之后。
        tail = ins.body.rsplit(",", 1)[-1].strip().upper()
        src_regs = _regs_in(tail)
        r_only = [x for x in src_regs if re.match(r"^R\d+$", x)]
        if r_only:
            base = min(int(x[1:]) for x in r_only)
            src_regs |= {"R" + str(base + j) for j in range(4)}
        nearby: Set[str] = set()
        # FILITE02 的 16B 记录以计时器 64-bit 值为源向量起点。用“源向量起点”而不是任意重叠
        # 寄存器判断依赖，避免业务 STG.128 恰好复用了前一探针留下的高位寄存器。
        src_base_reg = min((r for r in src_regs if re.match(r"^R\d+$", r)), key=lambda x: int(x[1:]), default="")
        timer_source = "direct"
        for kind, bases in timer_bases.items():
            if src_base_reg in bases:
                nearby.add(kind)
        if not nearby and src_base_reg in taint:
            # 源向量起点是计时器值经搬运/打包后的寄存器：仍算探针，但在 timer_source 里给出证据链。
            t = taint[src_base_reg]
            nearby.add(str(t["kind"]))
            timer_source = "via " + " ; ".join(str(x) for x in t["via"])
        if not nearby:
            continue
        # 从地址操作数做有界反向切片，报告“专用地址计算”。
        brackets = re.findall(r"\[([^\]]+)\]", ins.body)
        # desc[UR6][R16.64] 有两个方括号；最后一个才是实际数据地址。
        addr_ops = next((x for x in reversed(brackets) if re.search(r"\bR\d+", x, re.I)),
                        brackets[-1] if brackets else "")
        needed = set(re.findall(r"\bU?R\d+\b", addr_ops.upper()))
        # [R16.64] 隐含 R16,R17。
        for r in list(needed):
            if re.match(r"^R\d+$", r):
                needed.add("R" + str(int(r[1:]) + 1))
        addr_slice: List[Dict[str, object]] = []
        for j in range(i - 1, max(-1, i - 24), -1):
            prev = k.instructions[j]
            if prev.block != ins.block:
                break
            dst = _dest_regs(prev)
            hit = dst & needed
            if not hit:
                continue
            addr_slice.append({"offset": prev.offset_text, "opcode": prev.mnemonic,
                               "writes": sorted(hit), "text": prev.body})
            needed -= dst
            needed |= (_regs_in(prev.body) - dst)
            if not needed:
                break
        # OPEN/CLOSE 可能直接复用入口已算好的指针，因此地址切片可为空；仍记录该 store，
        # 并在 evidence 中明确 address_calc_count=0，供人工抽查。
        addr_slice.reverse()
        stores.append({"offset": ins.offset_text, "line": ins.line_no, "opcode": ins.mnemonic,
                       "source_vector": sorted(src_regs, key=lambda x: (x[0], int(x[1:]))),
                       "probe_kinds": sorted(nearby), "timer_source": timer_source,
                       "address_source": addr_ops,
                       "address_calculations": addr_slice,
                       "address_calc_count": len(addr_slice), "block": ins.block})
    return {
        "timer_reads": reads,
        "fi_buffer_stores": stores,
        "rejected_narrow_stores": rejected_narrow[:8],
        "record_store_rule": "只有 128-bit STG 算探针（记录写在源码里是 st.global.cs.v4.u32 = 16B）；"
                             "谓词前缀 @P0/@!P0 不影响识别；非 128-bit 的 STG 即使源寄存器与计时器链重合也拒绝，"
                             "被拒绝的写进 rejected_narrow_stores",
        "counts": {
            "timer_reads_total": len(reads),
            "globaltimer_reads": sum(x["kind"] == "globaltimer" for x in reads),
            "clock64_reads": sum(x["kind"] == "clock64" for x in reads),
            "smid_reads": sum(x["kind"] == "smid" for x in reads),
            "fi_buffer_stores": len(stores),
            "probe_events": len(stores),
            "narrow_stg_rejected": len(rejected_narrow),
            "address_calculations": sum(x["address_calc_count"] for x in stores),
        },
    }


def key_sequence(k: Kernel) -> Dict[str, List[Dict[str, object]]]:
    groups: Dict[str, List[Dict[str, object]]] = {x: [] for x in KEY_CLASSES}
    for ins in k.instructions:
        op = ins.opcode
        cls = None
        if op == "HGMMA":
            cls = "HGMMA"
        elif op.startswith("UTC") and "MMA" in op:
            cls = "UTC*MMA"
        elif op == "UTMALDG":
            cls = "UTMALDG"
        elif op == "UTMASTG":
            cls = "UTMASTG"
        elif op == "SYNCS":
            cls = "SYNCS"
        elif op in {"BAR", "BARRIER"} or op.startswith("BAR"):
            cls = "BAR"
        elif op == "LDGSTS":
            cls = "LDGSTS"
        elif op == "DEPBAR":
            cls = "DEPBAR"
        if cls:
            groups[cls].append({"offset": ins.offset_text, "opcode": ins.mnemonic})
    return groups


def compare_kernels(a: Kernel, b: Kernel) -> Dict[str, object]:
    at = [x.strict_token for x in a.instructions]
    bt = [x.strict_token for x in b.instructions]
    ae = [x.exact_token for x in a.instructions]
    be = [x.exact_token for x in b.instructions]
    ao = [x.opcode_token for x in a.instructions]
    bo = [x.opcode_token for x in b.instructions]
    lcs_s = bitset_lcs(at, bt)
    lcs_e = bitset_lcs(ae, be)
    lcs_o = bitset_lcs(ao, bo)
    n0, n1 = len(at), len(bt)
    sim_s = lcs_s / n0 if n0 else 1.0
    sim_o = lcs_o / n0 if n0 else 1.0
    dice_s = 2 * lcs_s / (n0 + n1) if n0 + n1 else 1.0
    dice_o = 2 * lcs_o / (n0 + n1) if n0 + n1 else 1.0
    k0, k1 = key_sequence(a), key_sequence(b)
    order_by_class = {}
    for cls in KEY_CLASSES:
        s0 = [x["opcode"] for x in k0[cls]]
        s1 = [x["opcode"] for x in k1[cls]]
        order_by_class[cls] = {"l0_count": len(s0), "fi_count": len(s1),
                               "preserved": s0 == s1,
                               "l0_sequence": s0, "fi_sequence": s1}
    # 跨类别也检查一条总的相对顺序；新增 instrumentation 不应插入关键类别。
    flat0 = [(cls, i["opcode"]) for cls in KEY_CLASSES for i in k0[cls]]
    flat1 = [(cls, i["opcode"]) for cls in KEY_CLASSES for i in k1[cls]]
    return {
        "l0_instructions": n0, "fi_instructions": n1,
        "instruction_delta": n1 - n0,
        "strict_lcs": lcs_s, "exact_raw_lcs": lcs_e, "opcode_lcs": lcs_o,
        "assembly_change_rate": 1 - sim_o,
        "sim_strict": round(sim_s, 6), "sim_exact_raw": round(lcs_e / n0 if n0 else 1.0, 6), "sim_opcode": round(sim_o, 6),
        "dice_strict": round(dice_s, 6), "dice_opcode": round(dice_o, 6),
        "l0_basic_blocks": len(a.blocks), "fi_basic_blocks": len(b.blocks),
        "basic_block_delta": len(b.blocks) - len(a.blocks),
        "basic_block_delta_pct": round((len(b.blocks) / len(a.blocks) - 1) * 100, 3) if a.blocks else None,
        "l0_terminators": a.block_terminators, "fi_terminators": b.block_terminators,
        "key_instruction_order_preserved": all(v["preserved"] for v in order_by_class.values()),
        "key_instruction_flat_order_preserved": flat0 == flat1,
        "key_instructions": order_by_class,
    }


def analyze(l0_path: str | os.PathLike, fi_path: str | os.PathLike, kernel: str,
            arch: str = DEFAULT_ARCH, work_dir: Optional[str | os.PathLike] = None,
            cuobjdump: Optional[str] = None, nvdisasm: Optional[str] = None) -> Dict[str, object]:
    cuobjdump = cuobjdump or shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    nvdisasm = nvdisasm or shutil.which("nvdisasm") or "/usr/local/cuda/bin/nvdisasm"
    root = Path(work_dir or ".")
    root.mkdir(parents=True, exist_ok=True)
    k0, _, m0 = resolve_kernel(Path(l0_path), kernel, arch, root, "l0", cuobjdump, nvdisasm)
    k1, _, m1 = resolve_kernel(Path(fi_path), kernel, arch, root, "fi", cuobjdump, nvdisasm)
    p0, p1 = identify_probes(k0), identify_probes(k1)
    cmp = compare_kernels(k0, k1)
    # 人工抽查给一段可直接复核的地址证据。
    sample_store = p1["fi_buffer_stores"][0] if p1["fi_buffer_stores"] else None
    sample_read = p1["timer_reads"][0] if p1["timer_reads"] else None
    manual = {
        "method": [
            "在 fi.sass 中先 grep SR_GLOBALTIMERLO、SR_CLOCKLO、SR_SMID/SR_VIRTUALSMID；每条应是 S2R/S2UR/CS2R。",
            "从一条计时器读向后看同一基本块内的 STG.*.128；其源向量中应包含计时器写出的寄存器（64-bit 计时器占两个 R）。",
            "查看 STG 地址操作数的反向切片：IMAD.WIDE/LEA/IADD3 等应把 lane/slot 序号乘以 16 字节，再加 FI 缓冲基址。",
            "用第 4 步的 dump 解码器抽查同一事件：卡片开始时刻应匹配该 STG 前的计时器读，事件 tag/aux 应匹配 names.json。",
            "改变一个 tile/chunk 输入后重跑，确认 STG 地址按 cap*16 递增而业务数据地址不被沿用到探针地址。",
        ],
        "sample_timer_read": sample_read,
        "sample_buffer_store": sample_store,
        "strict_equality_note": "sim_strict 保留 opcode、谓词、寻址和立即数，并把寄存器编号归一成 R#/UR#；sim_exact_raw 保留原寄存器编号；sim_opcode 只看完整 opcode。",
    }
    out = {
        "tool": TOOL, "version": VERSION,
        "kernel_regex": kernel, "requested_arch": arch,
        "l0": {"artifact": str(l0_path), **m0, "kernel": k0.name, "raw_name": k0.raw_name, "arch": k0.arch},
        "fi": {"artifact": str(fi_path), **m1, "kernel": k1.name, "raw_name": k1.raw_name, "arch": k1.arch},
        "lcs_alignment": cmp,
        "assembly_change_rate": cmp["assembly_change_rate"],
        "sim_strict": cmp["sim_strict"], "sim_opcode": cmp["sim_opcode"],
        "basic_blocks": {"l0": cmp["l0_basic_blocks"], "fi": cmp["fi_basic_blocks"],
                         "delta": cmp["basic_block_delta"], "delta_pct": cmp["basic_block_delta_pct"]},
        "key_instruction_order_preserved": cmp["key_instruction_order_preserved"],
        "key_instructions_by_class": cmp["key_instructions"],
        "probes": {"l0": p0, "fi": p1, "delta": {
            k: p1["counts"][k] - p0["counts"][k] for k in p0["counts"]
        }},
        "manual_spot_check": manual,
    }
    return out


def write_json(obj: object, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if str(p).endswith(".gz"):
        with gzip.open(p, "wb") as f:
            f.write(raw)
    else:
        p.write_bytes(raw)


def selftest() -> None:
    base = r'''
Fatbin elf code:
arch = sm_103a
		Function : _Z4demoPf
        /*0000*/                   S2R R1, SR_TID.X ; /* 0x01 */
        /*0010*/                   LDGSTS.E.BYPASS.128 [R2], [R3] ; /* 0x02 */
        /*0020*/                   BAR.SYNC 0x0 ; /* 0x03 */
        /*0030*/                   HGMMA.16816 R4, R5 ; /* 0x04 */
        /*0040*/                   STG.E.128 [R8.64], R9 ; /* 0x05 */
        /*0050*/                   EXIT ; /* 0x06 */
'''
    fi = r'''
Fatbin elf code:
arch = sm_103a
		Function : _Z4demoPf
        /*0000*/                   S2R R1, SR_TID.X ; /* 01 */
        /*0010*/                   CS2R R10, SR_GLOBALTIMERLO ; /* 02 */
        /*0020*/                   S2R R11, SR_VIRTUALSMID ; /* 03 */
        /*0030*/                   IMAD.WIDE.U32 R14, R12, 0x10, R4 ; /* 04 */
        /*0040*/                   STG.E.EF.128 desc[UR6][R14.64], R10 ; /* 05 */
        /*0050*/                   LDGSTS.E.BYPASS.128 [R2], [R3] ; /* 06 */
        /*0060*/                   BAR.SYNC 0x0 ; /* 07 */
        /*0070*/                   HGMMA.16816 R4, R5 ; /* 08 */
        /*0080*/                   STG.E.128 [R8.64], R9 ; /* 09 */
        /*0090*/                   EXIT ; /* 0a */
'''
    td = Path(tempfile.mkdtemp(prefix="fi_binsim_selftest_"))   # 系统临时目录，跑完删除
    a, b = td / "a.sass", td / "b.sass"
    a.write_text(base, encoding="utf-8")
    b.write_text(fi, encoding="utf-8")
    d = analyze(a, b, "_Z4demoPf", work_dir=td)
    assert d["probes"]["l0"]["counts"]["fi_buffer_stores"] == 0
    assert d["probes"]["fi"]["counts"]["globaltimer_reads"] == 1
    assert d["probes"]["fi"]["counts"]["smid_reads"] == 1
    assert d["probes"]["fi"]["counts"]["fi_buffer_stores"] == 1
    assert d["probes"]["fi"]["fi_buffer_stores"][0]["address_calc_count"] >= 1
    assert d["sim_opcode"] > 0 and d["sim_strict"] > 0
    assert d["key_instruction_order_preserved"]
    # Triton 3.6 在 H20 上生成的谓词化探针写（fi_bugs_h20-sk-glm 第 3 条报的原文形态）：
    #   @!P0 STG.E.EF.128 desc[UR16][R2.64], R12
    # 旧正则锚在 ^STG，谓词前缀让整条被跳过，fi_buffer_stores 报 0；源向量起点又常是计时器值
    # 经 MOV/PRMT 组装后的寄存器，需要同基本块内的污点传播才认得出。
    tri0 = r'''
Fatbin elf code:
arch = sm_103a
		Function : _Z9persistmmPf
        /*0000*/                   S2R R1, SR_TID.X ; /* 0x01 */
        /*0010*/                   BAR.SYNC 0x0 ; /* 0x02 */
        /*0020*/                   @P0 STG.E.128 [R8.64], R40 ; /* 0x03 */
        /*0030*/                   EXIT ; /* 0x04 */
'''
    tri = r'''
Fatbin elf code:
arch = sm_103a
		Function : _Z9persistmmPf
        /*0000*/                   CS2R R4, SR_GLOBALTIMER ; /* 0x01 */
        /*0010*/                   MOV R12, R4 ; /* 0x02 */
        /*0020*/                   PRMT R13, R5, 0x1044, RZ ; /* 0x03 */
        /*0030*/                   @!P0 STG.E.EF.128 desc[UR16][R2.64], R12 ; /* 0x04 */
        /*0040*/                   CS2R R20, SR_GLOBALTIMER ; /* 0x05 */
        /*0050*/                   @!P0 STG.E.EF.128 desc[UR16][R2.64], R20 ; /* 0x06 */
        /*0058*/                   STG.E.U16 [R36.64], R12 ; /* 0x0a */
        /*0060*/                   BAR.SYNC 0x0 ; /* 0x07 */
        /*0070*/                   @P0 STG.E.128 [R8.64], R40 ; /* 0x08 */
        /*0080*/                   EXIT ; /* 0x09 */
'''
    e0, e1 = td / "tri_l0.sass", td / "tri_fi.sass"
    e0.write_text(tri0, encoding="utf-8")
    e1.write_text(tri, encoding="utf-8")
    d3 = analyze(e0, e1, "persistmm", work_dir=td)
    got3 = d3["probes"]["fi"]["fi_buffer_stores"]
    assert d3["probes"]["l0"]["counts"]["fi_buffer_stores"] == 0, d3["probes"]["l0"]["counts"]
    assert d3["probes"]["fi"]["counts"]["fi_buffer_stores"] == 2, got3
    assert any(x["timer_source"] == "direct" for x in got3), got3
    assert any(str(x["timer_source"]).startswith("via MOV R4->R12") for x in got3), got3
    assert all("R40" not in x["source_vector"] for x in got3), got3
    # 谓词化业务写不能因为源向量编号靠近计时器寄存器就被算成探针。
    assert d3["probes"]["fi"]["counts"]["globaltimer_reads"] == 2
    # 非 128-bit 的业务写（bf16 输出 tile 的 STG.E.U16，源寄存器 R12 正好在计时器污点链上）
    # 必须被拒绝，并把拒绝原因留在证据里（fi_bugs_h20-sk-ds 第 5 条）。
    rej = d3["probes"]["fi"]["rejected_narrow_stores"]
    assert d3["probes"]["fi"]["counts"]["narrow_stg_rejected"] == 1, rej
    assert rej and rej[0]["opcode"].upper().startswith("STG.E.U16") and rej[0]["source_vector"] == ["R12"], rej
    # 删除关键 MMA 前后的 BAR 与 HGMMA，顺序必须被报出。
    bad = fi.replace("BAR.SYNC 0x0", "").replace("HGMMA.16816", "NOP")
    c = td / "bad.sass"
    c.write_text(bad, encoding="utf-8")
    d2 = analyze(a, c, "demo", work_dir=td)
    assert not d2["key_instruction_order_preserved"]
    assert d2["basic_blocks"]["fi"] >= 1
    shutil.rmtree(td, ignore_errors=True)
    print("SELFTEST PASS: LCS、基本块、探针（含谓词化 descriptor 写与计时器污点链）、地址切片、关键指令顺序均生效")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="NVIDIA SASS 相似度与 fast-instrument 探针识别")
    ap.add_argument("l0", nargs="?", help="未插桩可执行文件/so/cubin/Triton 构建目录或 .sass")
    ap.add_argument("fi", nargs="?", help="插桩后可执行文件/so/cubin/Triton 构建目录或 .sass")
    ap.add_argument("--kernel", help="目标 kernel 名正则（mangled 或 demangled 均可）")
    ap.add_argument("--arch", default=DEFAULT_ARCH)
    ap.add_argument("-o", "--output", default="binsim.json")
    ap.add_argument("--work-dir", help="反汇编证据写到该目录；默认 JSON 同目录")
    ap.add_argument("--disasm-dir", help="兼容别名，等同 --work-dir")
    ap.add_argument("--cuobjdump")
    ap.add_argument("--nvdisasm")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if not (a.l0 and a.fi and a.kernel):
        ap.error("需要 l0、fi 和 --kernel（或用 --selftest）")
    wd = a.disasm_dir or a.work_dir or str(Path(a.output).resolve().parent / "binsim_disasm")
    try:
        d = analyze(a.l0, a.fi, a.kernel, a.arch, wd, a.cuobjdump, a.nvdisasm)
    except Exception as e:
        print(f"fi_binsim FAIL: {e}", file=sys.stderr)
        return 2
    write_json(d, a.output)
    print(f"KERNEL {d['fi']['kernel']} arch={d['fi']['arch']} instructions={d['lcs_alignment']['l0_instructions']}->{d['lcs_alignment']['fi_instructions']}")
    print(f"LCS sim_strict={d['sim_strict']:.6f} sim_opcode={d['sim_opcode']:.6f} "
          f"basic_blocks={d['basic_blocks']['l0']}->{d['basic_blocks']['fi']} "
          f"key_order_preserved={d['key_instruction_order_preserved']}")
    c0, c1 = d['probes']['l0']['counts'], d['probes']['fi']['counts']
    print(f"PROBES timer_reads={c0['timer_reads_total']}->{c1['timer_reads_total']} "
          f"fi_buffer_stores={c0['fi_buffer_stores']}->{c1['fi_buffer_stores']} "
          f"address_calculations={c0['address_calculations']}->{c1['address_calculations']}")
    print("写出", a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
