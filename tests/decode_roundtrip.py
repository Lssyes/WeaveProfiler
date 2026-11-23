#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线端到端：合成 FILITE02 转储 -> weaveprof/pipeline/validate/tools/fi_lite_decode.py -> .perfetto.json.gz -> 校验。

覆盖五种口径（都不需要 GPU）：
  1. NVIDIA warp 粒度 + globaltimer（FI_TIMER=0）
  2. NVIDIA warp 粒度 + clock64 锚点（FI_TIMER=1）
  3. NVIDIA SM 粒度（FI_GRAN=1，每 CTA 一条泳道，按 %smid 归组 + 占用整条）
  4. AMD wave 粒度 + s_memrealtime（meta.timer_hz 换算 tick->ns）
  5. AMD CU 粒度（FI_GRAN=1，OPEN 的 aux 是 HW_ID 编码，按物理 CU 归组）
每种都检查：gzip 可解、JSON 可解、traceEvents 非空、时间轴数字对得上、错误计数为 0。
用法：python3 -I tests/decode_roundtrip.py [--keep-dir DIR]
"""
from __future__ import annotations

import argparse
import gzip
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DECODER = PKG / "weaveprof/pipeline/validate" / "tools" / "fi_lite_decode.py"

END, OPEN, MARK, CLK, CLOSE = 0x80000000, 0x40000000, 0x20000000, 0x10000000, 0x08000000


def build_dump(path: Path, slots: dict, n_slots: int, cap: int, wpc: int, timer: int, meta: dict) -> None:
    blob_meta = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    cnt = [len(slots.get(i, [])) for i in range(n_slots)]
    out = b"FILITE02" + struct.pack("<6I", n_slots, cap, wpc, 0, timer, len(blob_meta)) + blob_meta
    out += struct.pack(f"<{n_slots}I", *cnt)
    for i in range(n_slots):
        for rec in slots.get(i, []):
            out += struct.pack("<QII", *rec)
    path.write_bytes(out)


def hw_aux(xcc: int, se: int, cu: int, simd: int) -> int:
    """gfx942 HW_ID 编码：bits[3:0]=CU, [5:4]=SIMD, [9:8]=SE, [15:12]=XCC。"""
    return (cu & 15) | ((simd & 3) << 4) | ((se & 3) << 8) | ((xcc & 15) << 12)


def decode(dump: Path, names: dict, workdir: Path, extra=()) -> tuple[dict, dict]:
    names_path = workdir / (dump.stem + ".names.json")
    names_path.write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    out = workdir / (dump.stem + ".perfetto.json")
    cmd = [sys.executable, "-I", str(DECODER), str(dump), "--names", str(names_path),
           "-o", str(out), "--write-gz", *extra]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    gz = Path(str(out) + ".gz")
    if not gz.exists():
        raise AssertionError(f"解码未写出 {gz}（rc={p.returncode}）\n{p.stdout}")
    raw = gzip.decompress(gz.read_bytes())
    trace = json.loads(raw.decode("utf-8"))
    summary = json.loads(Path(str(out).replace(".json", "") + ".summary.json").read_text(encoding="utf-8"))
    return trace, summary


NAMES = {"events": {"1": "装载 tile", "4": "计算 tile", "12": "收尾归约"},
         # 标签只供人读；测试显式绑定 canonical ID，避免 "收尾" 同时命中
         # epilogue 与 reduction 时语义漂移。
         "semantics": {"1": "load", "4": "compute", "12": "reduction"},
         "roles": {"0": "链 CTA", "1": "辅助 CTA"},
         "aux": {"1": "tile", "4": "tile"}, "label_aux": True, "group": "role"}


def check_trace(trace: dict, summary: dict, label: str) -> None:
    ev = trace.get("traceEvents")
    assert isinstance(ev, list) and ev, f"{label}: traceEvents 为空"
    xs = [e for e in ev if e.get("ph") == "X" and e.get("cat") != "info"]
    assert xs, f"{label}: 没有区间事件"
    for e in xs:
        assert isinstance(e.get("ts"), (int, float)) and isinstance(e.get("dur"), (int, float)), f"{label}: {e}"
        assert e["dur"] >= 0, f"{label}: 负时长 {e}"
    assert not summary["errors"], f"{label}: 结构错误 {summary['errors']}"
    assert not summary["unnamed_event_ids"], f"{label}: 未命名事件 {summary['unnamed_event_ids']}"
    pn = [e for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"]
    assert pn and all(e["args"]["name"] for e in pn), f"{label}: 进程名缺失"
    tn = [e for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"]
    assert tn and all(e["args"]["name"] for e in tn), f"{label}: 泳道名缺失"
    other = trace.get("otherData") or {}
    assert other.get("observation_start_ns") is not None, f"{label}: 观测起点缺失"
    assert other.get("observation_end_ns", 0) > other.get("observation_start_ns", 0), f"{label}: 观测窗无效"
    assert other.get("observation_start_us") == 0.0, f"{label}: 观测窗未以最早记录为零点"
    assert other.get("observation_span_us", 0) >= (other.get("semantic_span_us") or 0), f"{label}: 观测窗未覆盖语义区间"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-dir", help="保留产物到该目录（默认临时目录，跑完删除）")
    a = ap.parse_args()
    tmp = Path(a.keep_dir) if a.keep_dir else Path(tempfile.mkdtemp(prefix="fi_roundtrip_"))
    tmp.mkdir(parents=True, exist_ok=True)
    cases = []

    # 1) NVIDIA warp 粒度 + globaltimer：记录值就是 ns
    slots = {0: [(1_000_000, OPEN | 0, 5), (1_000_500, 1, 0), (1_002_000, END | 1, 0),
                 (1_002_100, 4, 0), (1_004_000, END | 4, 0), (1_004_200, CLOSE, 0)],
             1: [(1_000_100, OPEN | 1, 5), (1_003_000, MARK | 12, 0), (1_004_500, CLOSE, 0)]}
    d1 = tmp / "nvidia_warp_globaltimer.bin"
    build_dump(d1, slots, 2, 16, 1, 0, {"kernel": "demo", "grid": 2, "gran": 0, "slot": "warp"})
    tr, sm = decode(d1, NAMES, tmp)
    assert sm["timer"] == "globaltimer" and sm["platform"] == "nvidia" and sm["lane_kind"] == "warp", sm["timer"]
    x = {e["name"]: e for e in tr["traceEvents"] if e.get("ph") == "X" and e.get("cat") != "info"}
    load0 = next(event for event in x.values() if event["name"] == "load")
    assert abs(load0["dur"] - 1.5) < 1e-9, load0   # 1500 ns = 1.5 us
    assert sm["records"] == 9 and sm["spans"] == 2 and sm["marks"] == 1, (sm["records"], sm["spans"], sm["marks"])
    assert any(e.get("ph") == "i" and e["name"] == "reduction" for e in tr["traceEvents"]), "瞬时事件缺失"
    check_trace(tr, sm, "nvidia_warp_globaltimer")
    cases.append(("nvidia_warp_globaltimer", len(tr["traceEvents"]), sm["timer_min_step_ns"]))

    # 2) NVIDIA warp 粒度 + clock64 锚点：0.5 ns/周期（2000 MHz）
    slots = {0: [(10_000, OPEN | 0, 5), (0, CLK, 0), (1_000, 1, 0), (3_000, END | 1, 0),
                 (12_000, CLOSE, 0), (4_000, CLK, 1)]}
    d2 = tmp / "nvidia_warp_clock64.bin"
    build_dump(d2, slots, 1, 16, 1, 1, {"kernel": "demo", "gran": 0})
    tr, sm = decode(d2, NAMES, tmp)
    x = [e for e in tr["traceEvents"] if e.get("ph") == "X"][0]
    assert abs(x["ts"] - 0.5) < 1e-9 and abs(x["dur"] - 1.0) < 1e-9, x
    assert sm["clock_mhz_per_lane"] == {"min": 2000.0, "max": 2000.0}, sm["clock_mhz_per_lane"]
    check_trace(tr, sm, "nvidia_warp_clock64")
    cases.append(("nvidia_warp_clock64", len(tr["traceEvents"]), sm["timer_min_step_ns"]))

    # 3) NVIDIA SM 粒度：3 个 CTA，两个落在 SM5（时间重叠 -> 2 条子条），一个落在 SM7
    slots = {0: [(1_000, OPEN | 0, 5), (1_500, 1, 0), (3_000, END | 1, 0), (3_200, CLOSE, 0)],
             1: [(1_200, OPEN | 0, 5), (1_800, 4, 0), (2_800, END | 4, 0), (3_100, CLOSE, 0)],
             2: [(1_100, OPEN | 0, 7), (1_900, 1, 0), (2_500, END | 1, 0), (2_600, CLOSE, 0)]}
    d3 = tmp / "nvidia_sm_gran.bin"
    build_dump(d3, slots, 3, 16, 1, 0, {"kernel": "demo", "grid": 3, "gran": 1, "slot": "cta"})
    tr, sm = decode(d3, NAMES, tmp)
    g = sm["sm_grouping"]
    assert sm["gran"] == 1 and sm["lane_kind"] == "cta" and sm["group"] == "sm_resident", sm["group"]
    assert g["sms_used"] == 2 and g["ctas"] == 3 and g["max_concurrent_resident"] == 2, g
    info = [e for e in tr["traceEvents"] if e.get("cat") == "info"]
    assert len(info) == 2 and all(e["name"] == "SM 占用" for e in info), info
    software = [e for e in tr["traceEvents"] if e.get("ph") == "X" and e.get("cat") != "info"]
    assert software and all(e["tid"] != 0 for e in software), \
        "SM-resident X events must not use Perfetto's default tid=0"
    assert {(e["pid"], e["tid"]) for e in software}.isdisjoint(
        {(e["pid"], e["tid"]) for e in info}), "SM overview and software tracks collide"
    check_trace(tr, sm, "nvidia_sm_gran")
    cases.append(("nvidia_sm_gran", len(tr["traceEvents"]), g["sub_bars_total"]))

    # 4) AMD wave 粒度：s_memrealtime 100 MHz（1 tick = 10 ns）
    slots = {0: [(100, OPEN | 0, hw_aux(0, 1, 1, 2)), (200, 1, 0), (500, END | 1, 0), (600, CLOSE, 0)]}
    d4 = tmp / "amd_wave_memrealtime.bin"
    build_dump(d4, slots, 1, 16, 1, 0, {"timer_hz": 100_000_000, "arch": "gfx942", "gran": 0, "kernel": "demo"})
    tr, sm = decode(d4, NAMES, tmp)
    x = [e for e in tr["traceEvents"] if e.get("ph") == "X"][0]
    assert abs(x["ts"] - 1.0) < 1e-9 and abs(x["dur"] - 3.0) < 1e-9, x   # (200-100)*10ns=1us，(500-200)*10ns=3us
    assert sm["timer"] == "s_memrealtime" and sm["platform"] == "amd" and sm["lane_kind"] == "wave", sm
    assert x["args"]["xcc"] == 0 and x["args"]["se"] == 1 and x["args"]["cu"] == 1 and x["args"]["simd"] == 2, x["args"]
    check_trace(tr, sm, "amd_wave_memrealtime")
    cases.append(("amd_wave_memrealtime", len(tr["traceEvents"]), sm["timer_min_step_ns"]))

    # 4b) AMD 但 meta 没写 arch（老转储）：靠 timer_hz 判定平台
    d4b = tmp / "amd_wave_no_arch.bin"
    build_dump(d4b, slots, 1, 16, 1, 0, {"timer_hz": 100_000_000, "gran": 0})
    tr, sm = decode(d4b, NAMES, tmp)
    assert sm["platform"] == "amd" and sm["platform_source"] == "meta.timer_hz", sm["platform_source"]
    check_trace(tr, sm, "amd_wave_no_arch")
    cases.append(("amd_wave_no_arch", len(tr["traceEvents"]), sm["timer_min_step_ns"]))

    # 5) AMD CU 粒度：两个 CTA 落在同一物理 CU（区间重叠 -> 2 条条带），另一个 CU 1 条
    a1, a2 = hw_aux(2, 1, 5, 0), hw_aux(2, 1, 6, 0)     # CU id = 149 / 150
    slots = {0: [(1_000, OPEN | 0, a1), (2_000, 1, 0), (6_000, END | 1, 0), (7_000, CLOSE, 0)],
             1: [(3_000, OPEN | 0, a1), (4_000, 1, 0), (5_000, END | 1, 0), (6_500, CLOSE, 0)],
             2: [(1_000, OPEN | 0, a2), (2_000, 4, 0), (3_000, END | 4, 0), (4_000, CLOSE, 0)]}
    d5 = tmp / "amd_cu_gran.bin"
    build_dump(d5, slots, 3, 16, 1, 0, {"timer_hz": 1_000_000_000, "arch": "gfx942", "gran": 1})
    tr, sm = decode(d5, NAMES, tmp)
    assert sm["group"] == "cu" and sm["cu"]["count"] == 2 and sm["cu"]["max_bands"] == 2, sm.get("cu")
    tracks = {(e["pid"], e["tid"]) for e in tr["traceEvents"] if e.get("ph") == "X"}
    assert tracks == {(149, 1), (149, 2), (150, 1)}, tracks
    assert all(tid != 0 for _, tid in tracks), "CU-grouped X events must not use Perfetto's default tid=0"
    pn = {e["pid"]: e["args"]["name"] for e in tr["traceEvents"] if e.get("ph") == "M" and e["name"] == "process_name"}
    assert pn[149].startswith("CU149") and "XCC2" in pn[149] and "SE1" in pn[149], pn
    check_trace(tr, sm, "amd_cu_gran")
    cases.append(("amd_cu_gran", len(tr["traceEvents"]), sm["cu"]["max_bands"]))

    # 6) 结构错误必须被拦下：缺一个 E -> 退出码 2 且不写文件
    bad = tmp / "broken_missing_e.bin"
    build_dump(bad, {0: [(1_000, OPEN | 0, 5), (1_500, 1, 0), (3_200, CLOSE, 0)]}, 1, 16, 1, 0, {"gran": 0})
    names_path = tmp / "broken.names.json"
    names_path.write_text(json.dumps(NAMES, ensure_ascii=False), encoding="utf-8")
    p = subprocess.run([sys.executable, "-I", str(DECODER), str(bad), "--names", str(names_path),
                        "-o", str(tmp / "broken.perfetto.json"), "--write-gz"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert p.returncode == 2 and not (tmp / "broken.perfetto.json.gz").exists(), (p.returncode, p.stdout)
    assert "未配对" in p.stdout, p.stdout
    cases.append(("broken_missing_e_blocked", p.returncode, None))

    print(f"{'用例':<30}{'traceEvents/退出码':>20}{'步长或子条数':>16}")
    for name, n, extra in cases:
        print(f"{name:<30}{n:>20}{str(extra):>16}")
    if not a.keep_dir:
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
    else:
        print("产物保留在", tmp)
    print(f"DECODE ROUNDTRIP PASS（{len(cases)} 个用例）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
