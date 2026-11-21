#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMDGCN 插桩前后相似度：strict/opcode LCS、探针剔除、资源与关键指令顺序。

Triton 的 AMDGCN 汇编带 ``.file``/``.loc`` 行表。可以用 ``--strip-file fi_triton.py`` 只剔除
前端探针实现产生的指令，再与 baseline 比较，避免把插入代码本身误算成主循环扰动。

用法：
  python3 -I fi_binsim.py --l0 baseline.s --fi instrumented.s --kernel mm_persistent \
      --strip-file fi_triton.py -o similarity.json
  python3 -I fi_binsim.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from array import array
from pathlib import Path

FILE2 = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]*)"\s+"([^"]*)"')
FILE1 = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]*)"')
LOC = re.compile(r'^\s*\.loc\s+(\d+)\s+(\d+)(?:\s+\d+)?')
LABEL = re.compile(r'^\s*(\.LBB\d+_\d+):')
# 顶层 kernel 标签（行首非空白、以 ':' 结尾）。--kernel 按正则匹配它，
# 与 fi_binsim_hip.py 同口径；旧实现是 startswith(kernel + ":") 的精确前缀匹配，
# 同一个 --kernel 在两个工具里语义不同（fi_bugs_amd-sk-qwen B07）。
TOP_LABEL = re.compile(r"^([A-Za-z_$][\w$.]*)\s*:\s*$")
INST = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*(.*?)\s*(?:;.*)?$')
RES = {
    "vgpr": re.compile(r'^\s*\.vgpr_count:\s*(\d+)', re.M),
    "sgpr": re.compile(r'^\s*\.sgpr_count:\s*(\d+)', re.M),
    "vgpr_spill": re.compile(r'^\s*\.vgpr_spill_count:\s*(\d+)', re.M),
    "sgpr_spill": re.compile(r'^\s*\.sgpr_spill_count:\s*(\d+)', re.M),
    "private_bytes": re.compile(r'^\s*\.private_segment_fixed_size:\s*(\d+)', re.M),
    "group_bytes": re.compile(r'^\s*\.group_segment_fixed_size:\s*(\d+)', re.M),
}


def parse_asm(path: str, kernel: str):
    files = {}
    rx = re.compile(kernel)
    seen_labels: list = []
    inside = False
    current_file = ""
    current_line = 0
    current_label = ""
    insts = []
    blocks = 0
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m2 = FILE2.match(raw)
        if m2:
            files[int(m2.group(1))] = m2.group(3)
            continue
        m1 = FILE1.match(raw)
        if m1:
            files[int(m1.group(1))] = Path(m1.group(2)).name
            continue
        if not inside:
            mt = TOP_LABEL.match(raw)
            if mt and not mt.group(1).startswith((".L", "$")):
                if rx.search(mt.group(1)):
                    inside = True
                else:
                    seen_labels.append(mt.group(1))
            continue
        if raw.startswith(".Lfunc_end"):
            break
        mlabel = LABEL.match(raw)
        if mlabel:
            current_label = mlabel.group(1)
            blocks += 1
            continue
        ml = LOC.match(raw)
        if ml:
            current_file = files.get(int(ml.group(1)), f"file{ml.group(1)}")
            ln = int(ml.group(2))
            if ln:
                current_line = ln
            continue
        mi = INST.match(raw)
        if not mi:
            continue
        op = mi.group(1)
        if op.startswith(".") or op in ("ASMSTART", "ASMEND"):
            continue
        text = (op + (" " + mi.group(2) if mi.group(2) else "")).strip()
        insts.append({
            "text": text,
            "opcode": op,
            "source_file": current_file,
            "source_line": current_line,
            "label": current_label,
        })
    if not inside:
        raise SystemExit(
            f"{path}: 找不到匹配 --kernel {kernel!r} 的 kernel 标签。文件里的顶层标签："
            f"{seen_labels[:12] or '（一个都没有）'}。\n"
            "--kernel 是正则（与 fi_binsim_hip.py 同口径），不是精确前缀；插桩版改过名（例如加 _fi 后缀）时，"
            "要么用能同时匹配两臂的正则（--kernel 'mm_persistent' 同时匹配 mm_persistent 与 mm_persistent_fi），"
            "要么用 --fi-kernel 单独指定插桩臂的名字。")
    return insts, blocks


def lcs_pairs(a, b):
    """返回最长公共子序列长度和配对下标；用整数 id + array 控制 Triton 内核级别内存。"""
    ids = {}
    def enc(seq):
        out = []
        for x in seq:
            if x not in ids:
                ids[x] = len(ids) + 1
            out.append(ids[x])
        return out
    A, B = enc(a), enc(b)
    n, m = len(A), len(B)
    tab = [array("I", [0]) * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row, nxt = tab[i], tab[i + 1]
        ai = A[i]
        for j in range(m - 1, -1, -1):
            row[j] = nxt[j + 1] + 1 if ai == B[j] else max(nxt[j], row[j + 1])
    pairs = []
    i = j = 0
    while i < n and j < m:
        if A[i] == B[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif tab[i + 1][j] >= tab[i][j + 1]:
            i += 1
        else:
            j += 1
    return tab[0][0], pairs


def resources(path):
    s = Path(path).read_text(encoding="utf-8", errors="replace")
    return {k: (int(m.group(1)) if (m := rx.search(s)) else None) for k, rx in RES.items()}


def memrealtime_wait_violations(insts):
    """返回计时读取等待顺序违规的下标及分类。

    正确序列为前置 ``s_waitcnt vmcnt(0)``、``s_memrealtime``、后置
    ``s_waitcnt lgkmcnt(0)``。前置等待清空 outstanding VMEM；后置等待让
    64 位标量读取结果就绪。
    """
    missing_pre = []
    missing_post = []
    for i, ins in enumerate(insts):
        if "s_memrealtime" not in ins["opcode"]:
            continue
        prev = insts[i - 1] if i else None
        nxt = insts[i + 1] if i + 1 < len(insts) else None
        if not (prev and prev["opcode"] == "s_waitcnt" and "vmcnt(0)" in prev["text"]):
            missing_pre.append(i)
        if not (nxt and nxt["opcode"] == "s_waitcnt" and "lgkmcnt(0)" in nxt["text"]):
            missing_post.append(i)
    return {
        "indices": sorted(set(missing_pre) | set(missing_post)),
        "missing_pre_vmcnt0": missing_pre,
        "missing_post_lgkmcnt0": missing_post,
    }




def loop_stats(insts, strip_files=()):
    """找出最外层循环和最内层主循环，统计原始及剔除探针后的指令条数。

    以向后分支定义 natural loop：目标标签位置到分支指令为循环体。循环深度按
    其他循环区间是否严格包含该区间计算。T2 的最内层循环就是 k-tile 主循环。
    """
    strips = {Path(x).name for x in strip_files if x}
    label_at = {}
    for i, ins in enumerate(insts):
        lab = ins.get("label")
        if lab and lab not in label_at:
            label_at[lab] = i
    back = []
    for i, ins in enumerate(insts):
        text = ins["text"]
        if text.startswith("s_branch ") or text.startswith("s_cbranch"):
            target = text.split()[-1]
            if target in label_at and label_at[target] < i:
                back.append((target, label_at[target], i, text))
    if not back:
        return {"k_loop": None, "outer_loop": None, "candidates": []}

    def one(edge):
        target, start, end, latch = edge
        depth = sum(1 for _t, s0, e0, _x in back if s0 < start and e0 > end)
        body = insts[start:end + 1]
        removed = sum(1 for ins in body if Path(ins["source_file"]).name in strips)
        return {
            "label": target,
            "depth": depth,
            "start_index": start + 1,
            "end_index": end + 1,
            "latch": latch,
            "instructions_raw": len(body),
            "probe_instructions_removed": removed,
            "instructions_after_removal": len(body) - removed,
        }

    candidates = [one(x) for x in back]
    # 最深的是 k-tile 主循环；同深度取区间最长者。
    k_loop = sorted(candidates, key=lambda x: (x["depth"], x["instructions_raw"]))[-1]
    # 最外层取 depth=0 中区间最长者（不是循环体内提前退出的条件回边）。
    outer = [x for x in candidates if x["depth"] == 0] or candidates
    outer_loop = sorted(outer, key=lambda x: x["instructions_raw"])[-1]
    return {"k_loop": k_loop, "outer_loop": outer_loop, "candidates": candidates}


def probe_evidence(raw):
    rt = sum("s_memrealtime" in x["opcode"] for x in raw)
    wait_bad = memrealtime_wait_violations(raw)
    return {
        "s_memrealtime": rt,
        "s_getreg_b32": sum(x["opcode"] == "s_getreg_b32" for x in raw),
        "v_mbcnt": sum(x["opcode"].startswith("v_mbcnt_") for x in raw),
        "global_store": sum(x["opcode"].startswith("global_store_") for x in raw),
        "memrealtime_wait_violations": len(wait_bad["indices"]),
        "memrealtime_missing_pre_vmcnt0": len(wait_bad["missing_pre_vmcnt0"]),
        "memrealtime_missing_post_lgkmcnt0": len(wait_bad["missing_post_lgkmcnt0"]),
        "memrealtime_wait_ok": rt > 0 and not wait_bad["indices"],
    }


def analyze(l0_path, fi_path, kernel, fi_kernel, strip_files, key_patterns):
    l0, l0_blocks = parse_asm(l0_path, kernel)
    fi, fi_blocks = parse_asm(fi_path, fi_kernel)
    strips = {Path(x).name for x in strip_files if x}
    stripped = [x for x in fi if Path(x["source_file"]).name not in strips]
    strict, pairs_strict = lcs_pairs([x["text"] for x in l0], [x["text"] for x in stripped])
    opcode, pairs_op = lcs_pairs([x["opcode"] for x in l0], [x["opcode"] for x in stripped])
    from collections import Counter
    available = Counter(x["opcode"] for x in stripped)
    keys = []
    for pat in key_patterns:
        for i, ins in enumerate(l0):
            if pat in ins["opcode"]:
                hit = available[ins["opcode"]] > 0
                if hit:
                    available[ins["opcode"]] -= 1
                keys.append({
                    "pattern": pat,
                    "l0_index": i + 1,
                    "opcode": ins["opcode"],
                    "matched": hit,
                })
    missing = [k for k in keys if not k["matched"]]

    def key_sequence(insts):
        return [x["opcode"] for x in insts if any(p in x["opcode"] for p in key_patterns)]

    key_l0 = key_sequence(l0)
    key_fi = key_sequence(stripped)
    key_lcs, _ = lcs_pairs(key_l0, key_fi)
    c0, cf = Counter(key_l0), Counter(key_fi)
    key_all_present = all(cf[k] >= c0[k] for k in c0)
    ev_raw = probe_evidence(fi)
    ev_strip = probe_evidence(stripped)
    out = {
        "l0": str(l0_path),
        "fi": str(fi_path),
        "kernel": kernel,
        "fi_kernel": fi_kernel,
        "strip_files": sorted(strips),
        "probe_removal_method": "按 AMDGCN .file/.loc 行表剔除指定前端源文件产生的指令；不改输入汇编",
        "l0_instructions": len(l0),
        "fi_instructions_raw": len(fi),
        "fi_probe_instructions_removed": len(fi) - len(stripped),
        "fi_instructions_after_removal": len(stripped),
        "sim_strict": round(strict / len(l0), 6) if l0 else None,
        "sim_opcode": round(opcode / len(l0), 6) if l0 else None,
        "lcs_strict": strict,
        "lcs_opcode": opcode,
        "l0_basic_blocks": l0_blocks,
        "fi_basic_blocks_raw": fi_blocks,
        "basic_block_delta_raw": fi_blocks - l0_blocks,
        "resources_l0": resources(l0_path),
        "resources_fi": resources(fi_path),
        "loops_l0": loop_stats(l0),
        "loops_fi": loop_stats(fi, strips),
        "probe_evidence_raw": ev_raw,
        "probe_evidence_after_strip": ev_strip,
        "probe_removal_verified": not any(ev_strip[k] for k in ("s_memrealtime", "s_getreg_b32", "v_mbcnt")),
        "probe_examples": [
            {"fi_index": i + 1, "text": x["text"], "source": f'{x["source_file"]}:{x["source_line"]}'}
            for i, x in enumerate(fi) if Path(x["source_file"]).name in strips
        ][:8],
        "key_instructions": keys,
        "key_instruction_count": len(keys),
        "key_instructions_matched": len(keys) - len(missing),
        "first_missing_key": missing[0] if missing else None,
        "key_all_present": key_all_present,
        "key_sequence_l0_instructions": len(key_l0),
        "key_sequence_fi_instructions": len(key_fi),
        "key_sequence_lcs": key_lcs,
        "key_order_lcs_ratio": round(key_lcs / len(key_l0), 6) if key_l0 else None,
        "key_relative_order_preserved": key_lcs == len(key_l0),
        "input_sha256": {
            "l0": hashlib.sha256(Path(l0_path).read_bytes()).hexdigest(),
            "fi": hashlib.sha256(Path(fi_path).read_bytes()).hexdigest(),
        },
    }
    return out


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory(prefix=".fi_binsim_", dir=str(Path.cwd())) as td:
        td = Path(td)
        l0 = td / "l0.s"
        fi = td / "fi.s"
        l0.write_text("""
        .file 1 "base.py"
sample:
.LBB0_0:
        .loc 1 1 0
        global_load_dword v1, v0, off
        .loc 1 2 0
        v_mfma_f32_16x16x16_bf16 v1, v2, v3, v1
        .loc 1 3 0
        global_store_dword v1, v0, off
.Lfunc_end0:
        .vgpr_count: 8
        .sgpr_count: 4
        .vgpr_spill_count: 0
        .sgpr_spill_count: 0
        .private_segment_fixed_size: 0
        .group_segment_fixed_size: 0
""", encoding="utf-8")
        fi.write_text("""
        .file 1 "base.py"
        .file 2 "/x/fi_triton.py"
sample:
.LBB0_0:
        .loc 1 1 0
        global_load_dword v1, v0, off
        .loc 2 10 0
        s_waitcnt vmcnt(0)
        s_memrealtime s[2:3]
        s_waitcnt lgkmcnt(0)
        .loc 1 2 0
        v_mfma_f32_16x16x16_bf16 v1, v2, v3, v1
        .loc 1 3 0
        global_store_dword v1, v0, off
.Lfunc_end0:
        .vgpr_count: 10
        .sgpr_count: 6
        .vgpr_spill_count: 0
        .sgpr_spill_count: 0
        .private_segment_fixed_size: 0
        .group_segment_fixed_size: 0
""", encoding="utf-8")
        # --kernel 是正则：'sam' 也要能匹配到 sample；找不到时报出文件里实际有哪些标签
        try:
            analyze(str(l0), fi, "no_such_kernel", "sample", [], [])
            raise AssertionError("kernel 标签缺失时没有报错")
        except SystemExit as e:
            assert "no_such_kernel" in str(e) and "sample" in str(e), str(e)
        r = analyze(l0, fi, "sam", "sample", ["fi_triton.py"], ["global_load", "v_mfma", "global_store"])
        assert r["fi_probe_instructions_removed"] == 3, r
        assert r["sim_strict"] == 1.0 and r["sim_opcode"] == 1.0, r
        assert r["probe_removal_verified"] and r["key_all_present"] and r["key_relative_order_preserved"], r
        assert r["probe_evidence_raw"]["memrealtime_wait_ok"], r
        print("SELFTEST PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l0")
    ap.add_argument("--fi")
    ap.add_argument("--kernel", default="mm_persistent")
    ap.add_argument("--fi-kernel")
    ap.add_argument("--strip-file", action="append", default=[])
    ap.add_argument("--key-pattern", default="global_load_ushort,global_load_dword,ds_write_b64,ds_read2st64_b64,ds_write_b16,ds_read_b64,v_mfma_f32_32x32x8_bf16,global_store_short")
    ap.add_argument("-o", "--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.l0 and a.fi and a.out):
        ap.error("需要 --l0 --fi -o")
    keys = [x for x in a.key_pattern.split(",") if x]
    out = analyze(a.l0, a.fi, a.kernel, a.fi_kernel or a.kernel, a.strip_file, keys)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("l0_instructions", "fi_instructions_raw", "fi_probe_instructions_removed",
                                           "sim_strict", "sim_opcode", "key_instructions_matched",
                                           "key_instruction_count", "probe_removal_verified")},
                     ensure_ascii=False))
    return 0 if out["probe_removal_verified"] and out["key_all_present"] else 2


if __name__ == "__main__":
    sys.exit(main())
