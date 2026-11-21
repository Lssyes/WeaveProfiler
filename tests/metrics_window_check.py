#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线回归：nsys GPU metrics 分段采集时，对齐工具必须响亮失败，不许交出没有硬件曲线的 trace。

背景（协调者实测）：`--gpu-metrics-frequency=200000`（5 us 间隔）时 nsys 是**分段**采样的，
H20 约 600 ms 有样本后断约 1100 ms，B300 约 400 ms 有样本后断约 1100 ms。四份盲测 trace
（h20-sk-ds 的 mm 与 ring、h20-sk-qwen 的 ring_pipeline、b3-sk-glm 的 t2）硬件曲线都是 0 条，
而 `summary.align.ok` 却是 true：h20-sk-ds 的 sqlite 只有 -0.8~599.2 ms 的样本、kernel 在 1300 ms；
b3-sk-glm 的 t2 有 5 段采样、31 个 kernel 实例全部落在间隙里。

本用例用合成 sqlite 复刻这两种形态（不需要 GPU、不需要 nsys），断言：
  1) 所选 kernel 窗口内样本不足 -> 退出码 4、不写 trace、summary.align.ok=false 且带覆盖诊断；
  2) 诊断里要能看出"多少个实例被覆盖"与"各采样段的范围"；
  3) 显式 --allow-missing-metrics 时照写 trace，但 align.ok=false、退出码 2；
  4) 正向对照：曲线数与每条样本数都达标时退出码 0、曲线真的进了 trace。

用法：python3 -I tests/metrics_window_check.py
"""
from __future__ import annotations

import gzip
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ALIGN = PKG / "weaveprof/pipeline/validate" / "tools" / "fi_align_nsys.py"
sys.path.insert(0, str(PKG / "weaveprof/pipeline/validate" / "tools"))

import fi_align_nsys as AN            # noqa: E402  复用它的合成转储/合成 sqlite 助手

# FILITE02 的 tag 位（与 recorders/cuda_cpp/fi_lite.cuh、weaveprof/pipeline/validate/tools/fi_lite_decode.py 同源）
END, OPEN, MARK, CLOSE = 0x80000000, 0x40000000, 0x20000000, 0x08000000

# 12 条能被默认 --metrics 正则命中的指标名（正向对照要过"至少 10 条曲线"这一关）
# nsys GPU metrics 在 sqlite 里的指标名是可读文本（TARGET_INFO_GPU_METRICS.metricName）；
# 这 12 条都能被工具默认的 --metrics 正则命中，用来过"至少 10 条曲线"这一关。
CURVE_NAMES = [
    "GPU Throughput [Throughput %]",
    "SMs Active [Throughput %]",
    "SM Issue [Throughput %]",
    "Tensor Active [Throughput %]",
    "Compute Warps in Flight [Throughput %]",
    "DRAM Read Bandwidth [Throughput %]",
    "DRAM Write Bandwidth [Throughput %]",
    "L2 Cache Throughput [Throughput %]",
    "Memory Throughput [Throughput %]",
    "L1/TEX Cache Throughput [Throughput %]",
    "Occupancy [Achieved %]",
    "SM Occupancy [Achieved %]",
]


def _add_metrics(db, k0, metric_names, segments, per_seg, step_ns, seg_stride_ns, offset_ns):
    """写分段采样的 GPU_METRICS：第 s 段的时间戳 = k0 + offset + s*seg_stride + i*step。"""
    c = sqlite3.connect(db)
    if not c.execute("SELECT name FROM sqlite_master WHERE name='TARGET_INFO_GPU_METRICS'").fetchone():
        c.execute("CREATE TABLE TARGET_INFO_GPU_METRICS (metricId INTEGER, metricName TEXT)")
        c.executemany("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?,?)",
                      [(i + 1, n) for i, n in enumerate(metric_names)])
    if not c.execute("SELECT name FROM sqlite_master WHERE name='GPU_METRICS'").fetchone():
        c.execute("CREATE TABLE GPU_METRICS (timestamp INTEGER, metricId INTEGER, value REAL)")
    rows = []
    for s in range(segments):
        for i in range(per_seg):
            ts = int(k0) + offset_ns + s * seg_stride_ns + i * step_ns
            for mid in range(1, len(metric_names) + 1):
                rows.append((ts, mid, 1.0 * mid + i))
    c.executemany("INSERT INTO GPU_METRICS VALUES (?,?,?)", rows)
    c.commit()
    c.close()
    return len(rows)


def _run(db, dump, names, out, extra=()):
    p = subprocess.run([sys.executable, "-I", str(ALIGN), dump, "--names", names, "--sqlite", db,
                        "--kernel", "demo_kernel", "-o", out, *extra],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    return p.returncode, p.stdout


def _summary(out):
    sp = out.replace(".gz", "").replace(".json", "") + ".summary.json"
    if not os.path.exists(sp):
        return None, sp
    with open(sp, encoding="utf-8") as fh:
        return json.load(fh), sp


def main() -> int:
    fails = []
    tmp = tempfile.mkdtemp(prefix="fi_metrics_window_")
    dump = os.path.join(tmp, "demo.fi.bin")
    names = os.path.join(tmp, "names.json")
    with open(names, "w", encoding="utf-8") as fh:
        json.dump(AN.SELFTEST_NAMES, fh, ensure_ascii=False)
    # 软件轴：最早 OPEN=1_000_000 ns，最晚记录=1_006_000 ns -> 跨度 6 us
    AN._selftest_write_dump(dump, {
        0: [(1_000_000, OPEN | 0, 5), (1_000_500, 1, 0), (1_002_000, END | 1, 0), (1_006_000, CLOSE, 0)],
        1: [(1_000_200, OPEN | 0, 6), (1_001_000, 2, 0), (1_004_000, END | 2, 0), (1_005_000, CLOSE, 0)]})
    k0, k1 = 5_000_000_000, 5_000_008_000          # kernel 8 us

    # ---- 1) kernel 落在采样断档里：样本只到 kernel 前 100 ms，两段，中间断 1.1 s ----
    db_gap = os.path.join(tmp, "gap.sqlite")
    AN._selftest_write_db(db_gap, k0, k1, with_metrics=False)
    n = _add_metrics(db_gap, k0, CURVE_NAMES[:2], segments=2, per_seg=30, step_ns=5_000,
                     seg_stride_ns=1_100_000_000, offset_ns=-100_000_000)
    out_gap = os.path.join(tmp, "gap.perfetto.json")
    rc, so = _run(db_gap, dump, names, out_gap, ("--min-metric-curves", "2"))
    if rc != 4:
        fails.append(f"[断档] 退出码 {rc}（期望 4）：{so[-800:]}")
    if os.path.exists(out_gap + ".gz"):
        fails.append("[断档] 不该写出 trace")
    for needle in ("采样段", "gpu-metrics-frequency", "落在断档里"):
        if needle not in so:
            fails.append(f"[断档] 诊断里没有 {needle!r}：{so[-800:]}")
    s, sp = _summary(out_gap)
    if s is None:
        fails.append(f"[断档] 没写出 summary（{sp}）")
    else:
        al = s.get("align", {})
        if al.get("ok") is not False or al.get("metrics_ok") is not False or al.get("tail_ok") is not True:
            fails.append(f"[断档] summary.align 判错：ok={al.get('ok')} metrics_ok={al.get('metrics_ok')} "
                         f"tail_ok={al.get('tail_ok')}")
        if al.get("metrics_in_window") != {}:
            fails.append(f"[断档] metrics_in_window 应为空：{al.get('metrics_in_window')}")
        cov = al.get("metrics_coverage") or {}
        if cov.get("instances_total") != 1 or cov.get("instances_covered") != 0:
            fails.append(f"[断档] 实例覆盖统计不对：{cov.get('instances_total')}/{cov.get('instances_covered')}")
        if len(cov.get("segments") or []) != 2:
            fails.append(f"[断档] 采样段应为 2 段：{cov.get('segments')}")
        if not al.get("metrics_problems"):
            fails.append("[断档] summary 里没有 metrics_problems")
    print(f"  [断档] {n} 行 GPU_METRICS、2 段采样，kernel 在间隙里：rc={rc}，"
          f"匹配 1 个实例、覆盖 0 个，未写 trace —— 符合预期")

    # ---- 2) 显式放行：照写 trace，但 align.ok=false、退出码 2 ----
    rc2, so2 = _run(db_gap, dump, names, out_gap, ("--min-metric-curves", "2", "--allow-missing-metrics"))
    if rc2 != 2 or not os.path.exists(out_gap + ".gz"):
        fails.append(f"[放行] rc={rc2}、trace 存在={os.path.exists(out_gap + '.gz')}（期望 rc=2 且写出 trace）")
    else:
        with gzip.open(out_gap + ".gz", "rb") as fh:
            tr = json.loads(fh.read().decode("utf-8"))
        al2 = tr["otherData"]["align"]
        cs = [e for e in tr["traceEvents"] if e.get("ph") == "C"]
        if al2.get("ok") is not False or al2.get("metrics_ok") is not False:
            fails.append(f"[放行] trace 里 align.ok/metrics_ok 应为 false：{al2.get('ok')}/{al2.get('metrics_ok')}")
        if cs:
            fails.append(f"[放行] 曲线应为 0 条，实得 {len(cs)}")
        if tr["otherData"]["hw_sw_info"]["hw_metrics"].get("metrics_ok") is not False:
            fails.append("[放行] hw_sw_info.hw_metrics.metrics_ok 应为 false")
    print(f"  [放行] --allow-missing-metrics：rc={rc2}，trace 照写但 align.ok=false，曲线 0 条")

    # ---- 3) 正向对照：12 条曲线、每条窗口内 6 个样本 -> rc=0 且曲线进 trace ----
    db_ok = os.path.join(tmp, "ok.sqlite")
    AN._selftest_write_db(db_ok, k0, k1, with_metrics=False)
    _add_metrics(db_ok, k0, CURVE_NAMES, segments=1, per_seg=6, step_ns=1_000,
                 seg_stride_ns=0, offset_ns=0)
    out_ok = os.path.join(tmp, "ok.perfetto.json")
    rc3, so3 = _run(db_ok, dump, names, out_ok)
    if rc3 != 0 or not os.path.exists(out_ok + ".gz"):
        fails.append(f"[正向] rc={rc3}（期望 0）：{so3[-800:]}")
    else:
        with gzip.open(out_ok + ".gz", "rb") as fh:
            tr3 = json.loads(fh.read().decode("utf-8"))
        al3 = tr3["otherData"]["align"]
        cs3 = [e for e in tr3["traceEvents"] if e.get("ph") == "C"]
        if al3.get("metrics_ok") is not True or al3.get("ok") is not True:
            fails.append(f"[正向] metrics_ok/ok 应为 true：{al3.get('metrics_ok')}/{al3.get('ok')}")
        if al3.get("metrics_curves") != len(CURVE_NAMES):
            fails.append(f"[正向] 曲线数 {al3.get('metrics_curves')}，期望 {len(CURVE_NAMES)}")
        if al3.get("metrics_min_samples_per_curve", 0) < 3:
            fails.append(f"[正向] 每条样本数 {al3.get('metrics_min_samples_per_curve')}，期望 >=3")
        if len(cs3) != len(CURVE_NAMES) * 6:
            fails.append(f"[正向] trace 里曲线点数 {len(cs3)}，期望 {len(CURVE_NAMES) * 6}")
        if "metrics_in_window" not in al3 or len(al3["metrics_in_window"]) != len(CURVE_NAMES):
            fails.append(f"[正向] summary.align.metrics_in_window 不全：{len(al3.get('metrics_in_window') or {})}")
        print(f"  [正向] {len(CURVE_NAMES)} 条曲线 × 6 个样本：rc=0，trace 里 {len(cs3)} 个点，"
              f"采样间隔 {al3['metrics_coverage'].get('sample_period_us')} us")

    # ---- 4) 31 个实例全部落在 5 段采样的间隙里（b3-sk-glm 的 t2 形态）----
    db_many = os.path.join(tmp, "many.sqlite")
    AN._selftest_write_db(db_many, k0, k1, with_metrics=False)
    c = sqlite3.connect(db_many)
    for i in range(30):                              # 再加 30 个同名 kernel 实例，跨度 4.2 s
        c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,11,0,7,4,1,1,256,1,1,64,1024,0)",
                  (int(k0) + (i + 1) * 140_000_000, int(k0) + (i + 1) * 140_000_000 + 8_000))
    c.commit()
    c.close()
    _add_metrics(db_many, k0, CURVE_NAMES[:2], segments=5, per_seg=20, step_ns=5_000,
                 seg_stride_ns=900_000_000, offset_ns=-800_000_000)
    out_many = os.path.join(tmp, "many.perfetto.json")
    rc4, so4 = _run(db_many, dump, names, out_many, ("--min-metric-curves", "2", "--instance", "-1"))
    s4, _ = _summary(out_many)
    cov4 = (s4 or {}).get("align", {}).get("metrics_coverage") or {}
    if rc4 != 4:
        fails.append(f"[31 实例] rc={rc4}（期望 4）：{so4[-500:]}")
    if cov4.get("instances_total") != 31:
        fails.append(f"[31 实例] 实例数 {cov4.get('instances_total')}，期望 31")
    if len(cov4.get("segments") or []) != 5:
        fails.append(f"[31 实例] 采样段 {len(cov4.get('segments') or [])}，期望 5")
    print(f"  [31 实例] rc={rc4}，实例 {cov4.get('instances_total')} 个、覆盖 {cov4.get('instances_covered')} 个，"
          f"采样段 {len(cov4.get('segments') or [])} 段，最大断档 {cov4.get('gap_max_us')} us")

    for f in sorted(os.listdir(tmp)):
        try:
            os.remove(os.path.join(tmp, f))
        except OSError:
            pass
    os.rmdir(tmp)
    if fails:
        print("METRICS WINDOW CHECK FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("METRICS WINDOW CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
