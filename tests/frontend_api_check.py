#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""前端离线检查：语法、公开 API 是否齐全、设备侧关键指令是否在源码里、NVIDIA 前端能否离线生成 PTX。

不需要 GPU。torch/triton 可用时多做一步：真的调 FiCfg 生成 PTX 并核对指令；不可用时按 SKIP 记，不假装通过。
用法：python3 -I tests/frontend_api_check.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent

FRONTENDS = {
    "2_instrument/recorders/triton/fi_triton.py": {
        "symbols": ["FiCfg", "FiBuffer", "FiAsmBackend", "FI_BACKENDS", "fi_open", "fi_open_by", "fi_close",
                    "fi_b", "fi_e", "fi_m", "fi_bg", "fi_eg", "fi_mg", "cap_for", "write_names",
                    "fi_b_by", "fi_e_by", "fi_m_by", "fi_close_by", "resolve_arch"],
        "must_contain": ["%globaltimer", "%clock64", "%smid", "st.global.cs.v4.u32", "atom.global.or.b32",
                         "%laneid", "%warpid"],
        "must_not_contain": ["amd_gfx942"],   # 未实测的 AMD 设计稿后端已移除，AMD 走 fi_triton_amd.py
    },
    "2_instrument/recorders/triton/fi_triton_amd.py": {
        "symbols": ["FiBuffer", "fi_open", "fi_b", "fi_e", "fi_m", "fi_close"],
        # 设备侧读 s_memrealtime（前后各一条 waitcnt），记录写三块探测缓冲（time/tag/aux），host 打包成 FILITE02
        "must_contain": ["s_memrealtime", "s_waitcnt lgkmcnt(0)", "_fi_hw_id", "tl.store", "FILITE02"],
        "must_not_contain": [],
    },
    "2_instrument/recorders/cutedsl/fi_cute.py": {
        "symbols": ["FiBuffer", "fi_open", "fi_b", "fi_e", "fi_close"],
        "must_contain": ["FILITE02"],
        "must_not_contain": [],
    },
}


def top_level_symbols(tree: ast.AST) -> set:
    """模块级可见的名字。fi_triton_amd.py 把设备侧函数放在 `if tl is not None:` 里，
    这种模块级条件块里的定义同样是公开 API，要一并收进来。"""
    out = set()

    def walk(body):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(n.name)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        out.add(t.id)
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                out.add(n.target.id)
            elif isinstance(n, ast.If):
                walk(n.body)
            elif isinstance(n, (ast.Try,)):
                walk(n.body)

    walk(tree.body)
    return out


def main() -> int:
    fails = []
    for rel, spec in FRONTENDS.items():
        p = PKG / rel
        if not p.is_file():
            fails.append(f"{rel}: 文件不存在")
            continue
        src = p.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            fails.append(f"{rel}: 语法错误 {e}")
            continue
        have = top_level_symbols(tree)
        missing = [s for s in spec["symbols"] if s not in have]
        if missing:
            fails.append(f"{rel}: 缺公开 API {missing}")
        for needle in spec["must_contain"]:
            if needle not in src:
                fails.append(f"{rel}: 源码里没有关键指令/标记 {needle!r}")
        for needle in spec["must_not_contain"]:
            if needle in src:
                fails.append(f"{rel}: 不该出现 {needle!r}（未实测的后端/接口不能留在包里）")
        print(f"  {rel}: {len(have & set(spec['symbols']))}/{len(spec['symbols'])} 个 API，"
              f"{len(spec['must_contain'])} 个关键指令标记齐全")

    # NVIDIA Triton 前端：能 import 就真生成 PTX（纯 Python，不需要 GPU）
    try:
        sys.path.insert(0, str(PKG / "2_instrument" / "recorders" / "triton"))
        import torch  # noqa: F401
        import triton  # noqa: F401
        import fi_triton as FT
    except Exception as e:
        print(f"  SKIP PTX 生成（torch/triton 不可用：{type(e).__name__}: {e}）")
    else:
        cfg = FT.FiCfg(cap=64, warps=8, n_slots=8 * 148, gran=0, timer=0, level=2, threads=256)
        open_asm = cfg.open_asm(0)
        ev_asm = cfg.event_asm
        close_asm = cfg.close_asm
        for txt, needle in ((open_asm, "%globaltimer"), (open_asm, "%smid"),
                            (ev_asm, "st.global.cs.v4.u32"), (close_asm, "st.global.cs.v4.u32")):
            if needle not in txt:
                fails.append(f"fi_triton 生成的 PTX 缺 {needle}")
        if cfg.open_nr < 1 or cfg.close_nr < 1 or cfg.event_nr < 1:
            fails.append("fi_triton 记录条数不对")
        # FI_TIMER=1 时 OPEN/CLOSE 各多一条 clock64 锚点
        cfg1 = FT.FiCfg(cap=64, warps=8, n_slots=8, gran=0, timer=1, level=2, threads=256)
        if not (cfg1.open_nr > cfg.open_nr and "%clock64" in cfg1.open_asm(0)):
            fails.append("FI_TIMER=1 的 clock64 锚点没生成")
        # 参数校验必须挡住错误的 threads / gran / timer
        for bad in (dict(cap=64, warps=8, n_slots=8, gran=0, timer=0, level=2, threads=128),
                    dict(cap=64, warps=8, n_slots=8, gran=2, timer=0, level=2, threads=256),
                    dict(cap=64, warps=8, n_slots=8, gran=0, timer=3, level=2, threads=256),
                    dict(cap=64, warps=8, n_slots=8, gran=0, timer=0, level=9, threads=256),
                    dict(cap=64, warps=8, n_slots=8, gran=0, timer=0, level=2, threads=256, backend="amd_gfx942")):
            try:
                FT.FiCfg(**bad)
            except (ValueError, NotImplementedError):
                continue
            fails.append(f"FiCfg 没挡住非法参数 {bad}")
        if FT.cap_for(4, 32) < 4 * 32 + 2:
            fails.append("cap_for 给的 cap 小于 每tile记录数×tile数 + OPEN/CLOSE")
        # meta 必须带上解码器需要的字段
        m = cfg.meta(kernel="demo")
        for k in ("gran", "timer", "level", "arch", "cap_per_lane", "n_slots", "warps_per_cta",
                  "arch_source"):
            if k not in m:
                fails.append(f"fi_triton meta 缺 {k}")
        # 谓词版 API：event 的谓词排到 $8（$6/$7 被 tag/aux 占了），open/close 的谓词是 $6。
        # 只给 fi_open_by 加谓词会产生“只有事件、没有 OPEN/CLOSE”的残缺泳道（fi_bugs_h20-sk-qwen B2）。
        if len(cfg.event_by_cons.split(",")) != 9:
            fails.append(f"event_by 的 constraints 操作数不是 9 个：{cfg.event_by_cons}")
        if len(cfg.close_by_cons.split(",")) != 7:
            fails.append(f"close_by 的 constraints 操作数不是 7 个：{cfg.close_by_cons}")
        if "setp.ne.u32 p_rec, $8, 0;" not in cfg.event_by_asm:
            fails.append("gran=0 的 event_by 没有用 $8 当谓词")
        if "setp.ne.u32 p_rec, $6, 0;" not in cfg.close_by_asm:
            fails.append("gran=0 的 close_by 没有用 $6 当谓词")
        if "setp.ne.u32 p_rec, $6, 0;" not in cfg.openby_asm(1):
            fails.append("gran=0 的 open_by 没有用 $6 当谓词")
        cfg_sm = FT.FiCfg(cap=64, warps=1, n_slots=8, gran=1, timer=0, level=2, threads=256)
        if "setp.ne.u32 p_user, $8, 0;" not in cfg_sm.event_by_asm:
            fails.append("gran=1 的 event_by 没有把用户谓词与 0 号线程取与（$8）")
        if cfg.event_by_nr < 1 or cfg.close_by_nr < 1:
            fails.append("谓词版事件的记录条数不对")
        # 架构标签：显式给定优先，非 NVIDIA 后端用后端名，都不许再写死 sm_90a
        if FT.resolve_arch("nvidia_ptx", "sm_103a")[0] != "sm_103a":
            fails.append("resolve_arch 没有尊重显式 arch")
        if FT.resolve_arch("amd_gfx942", None)[0] != "amd_gfx942":
            fails.append("resolve_arch 对非 NVIDIA 后端没有用后端名")
        cfg_exp = FT.FiCfg(cap=64, warps=8, n_slots=8, gran=0, timer=0, level=2, threads=256,
                           arch="sm_103a")
        if cfg_exp.meta()["arch"] != "sm_103a":
            fails.append("FiCfg(arch=...) 没有写进 meta")
        print(f"  fi_triton: PTX 生成通过（open {cfg.open_nr} 条 / event {cfg.event_nr} 条 / close {cfg.close_nr} 条），"
              f"meta 字段 {len(m)} 个")

    if fails:
        print("FRONTEND CHECK FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("FRONTEND CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
