# -*- coding: utf-8 -*-
"""把 fi_lite 软件打点对齐到同一次 nsys 采集，合成一个 Perfetto json：上面是每 SM/warp 的软件区间，
下面是 nsys 的 kernel 条和 GPU metrics 曲线（Tensor/SM/HBM 读写……），时间轴统一为“该 kernel 开始 = 0 us”。

对齐方法：
  软件时间是 %globaltimer（FI_TIMER=1 时已由解码器换算到 globaltimer）。nsys 的 kernel 起止来自 CUPTI，属于 nsys 会话时基。
  两者差一个常数偏移：offset = kernel_start(nsys) - 最早一条 FI_OPEN 的 globaltimer。
  这个偏移把“第一个 CTA 开始执行”对到“kernel 开始”，误差 = 第一个 CTA 的派发延迟（通常 ~1 us 量级）。
  校验：对齐后最后一条软件记录必须落在 kernel_end 之前，余量（kernel_end - 最后一条记录）写进 summary；
  余量为负或大于 --max-tail-us 视为对齐失败（找错了 kernel 实例 / 插桩没覆盖到收尾）。

用法：
  python3 -I fi_align_nsys.py <dump.bin> --names names.json --sqlite cap.sqlite --kernel '<kernel 名正则>' \
      [--instance -1] [--metrics 'tensor|sm active|dram|hbm'] -o merged.perfetto.json[.gz]
  --instance：该正则匹配到的第几个 kernel 实例（按开始时间排序，-1=最后一个）。被插桩那次 launch 是哪一次由调用方决定。
"""
import argparse, gzip, hashlib, json, os, re, sqlite3, sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 同目录的 fi_lite_decode（本 skill 自带代码）
from fi_lite_decode import OPEN, decode, print_summary, read_dump, write_json  # noqa: E402
from fi_metric_map import detect_arch, mapping_document, match_metric_names  # noqa: E402

ALIGN_VERSION = "fi-align-nsys"


KCOLS = ["deviceId", "streamId", "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ", "registersPerThread",
         "staticSharedMemory", "dynamicSharedMemory", "localMemoryPerThread", "sharedMemoryExecuted", "clusterX",
         "clusterY", "clusterZ", "maxPotentialClusterSize", "launchType"]
GCOLS = ["name", "chipName", "busLocation", "smCount", "clockRate", "memoryBandwidth", "l2CacheSize", "totalMemory",
         "computeMajor", "computeMinor", "maxWarpsPerSm", "maxRegistersPerSm", "maxShmemPerSm", "maxShmemPerBlockOptin",
         "uuid", "cuDevice"]
ENVKEYS = ["Hostname", "OsDescription", "NvDriverVersion", "CudaDriverVersion", "CpuArchitecture", "CpuCores",
           "KernelVersion", "SupportsGpuMetrics"]


def _cols(c, table):
    return [r[1] for r in c.execute(f"PRAGMA table_info({table})")]


# GPU metrics 覆盖诊断：nsys 在高采样频率下会分段采集（200 kHz 时 H20 约 600 ms 有样本后断约
# 1100 ms），kernel 落在断档里就一个样本都没有，而旧实现只打印一句提示照样出图，
# 于是交出去的 trace 硬件曲线 0 条、summary.align.ok 却是 True（协调者实测的 4 份盲测 trace 都是这个病）。
METRICS_SCAN_LIMIT = 4_000_000       # 只扫一个 metricId 的时间戳；超出就在结论里注明扫描被截断


def _sample_segments(ts, gap_factor: float = 8.0, min_gap_ns: float = 1000.0):
    """把升序时间戳切成连续采样段：相邻间隔超过中位间隔 gap_factor 倍就算断档。"""
    if len(ts) < 2:
        return [{"start": ts[0], "end": ts[0], "samples": 1}] if ts else [], None
    diffs = sorted(b - a for a, b in zip(ts, ts[1:]))
    med = diffs[len(diffs) // 2] or 1
    thr = max(min_gap_ns, gap_factor * med)
    segs, beg, prev, n = [], ts[0], ts[0], 1
    gap_max = 0
    for t in ts[1:]:
        if t - prev > thr:
            segs.append({"start": beg, "end": prev, "samples": n})
            gap_max = max(gap_max, t - prev)
            beg, n = t, 1
        else:
            n += 1
        prev = t
    segs.append({"start": beg, "end": prev, "samples": n})
    return segs, {"median_ns": med, "max_ns": gap_max}


def gpu_metrics_coverage(c, metric_ids, kernels, k0, pad_ns, min_samples, scan_limit=METRICS_SCAN_LIMIT):
    """扫一个 metricId 的全部时间戳（各 metric 的采样时刻相同，扫一个就够），算出采样段与每个
    kernel 实例的覆盖情况。时间一律相对 k0（= trace 的时间零点），单位 us。"""
    import bisect
    out = {"metric_ids_used": sorted(metric_ids), "timestamps_scanned": 0, "scan_truncated": False,
           "segments": [], "sample_period_us": None, "gap_max_us": None,
           "instances_total": len(kernels), "instances_covered": 0, "per_instance": [],
           "min_samples_required": min_samples, "axis": "时间相对所选 kernel 实例开始时刻（us）"}
    if not metric_ids:
        return out
    mid = sorted(metric_ids)[0]
    rows = c.execute("SELECT timestamp FROM GPU_METRICS WHERE metricId=? ORDER BY timestamp LIMIT ?",
                     (int(mid), scan_limit + 1))
    ts = [int(r[0]) for r in rows]
    out["scan_metric_id"] = mid
    out["timestamps_scanned"] = len(ts)
    if len(ts) > scan_limit:
        ts = ts[:scan_limit]
        out["scan_truncated"] = True
        out["scan_limit"] = scan_limit
    if not ts:
        return out
    segs, gaps = _sample_segments(ts)
    out["segments"] = [{"start_us": round((s["start"] - k0) / 1e3, 3), "end_us": round((s["end"] - k0) / 1e3, 3),
                        "samples": s["samples"]} for s in segs]
    if gaps:
        out["sample_period_us"] = round(gaps["median_ns"] / 1e3, 4)
        out["gap_max_us"] = round(gaps["max_ns"] / 1e3, 3)
    for i, (s, e, n) in enumerate(kernels):
        lo, hi = int(s) - pad_ns, int(e) + pad_ns
        cnt = bisect.bisect_right(ts, hi) - bisect.bisect_left(ts, lo)
        ok = cnt >= min_samples
        out["instances_covered"] += 1 if ok else 0
        out["per_instance"].append({"index": i, "selected": (s == k0), "name": str(n)[:120],
                                    "start_us": round((s - k0) / 1e3, 3), "end_us": round((e - k0) / 1e3, 3),
                                    "samples_in_window": cnt, "covered": ok})
    return out


def kernel_launch_info(c, k0, kname):
    """被插桩那次 launch 的软件侧配置：grid/block/cluster/寄存器/共享内存（来自 CUPTI kernel 表）。"""
    have = [x for x in KCOLS if x in _cols(c, "CUPTI_ACTIVITY_KIND_KERNEL")]
    row = c.execute(f"SELECT {','.join(have)} FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start=?", (int(k0),)).fetchone()
    info = dict(zip(have, row)) if row else {}
    info["kernel"] = kname[:300]
    return info


def hw_sw_info(c, dev_id, extra):
    """GPU 型号/SM 数/时钟/带宽、驱动与系统环境、nsys 版本与 GPU metrics 采样配置、调用方给的软件信息（nvcc、编译选项…）。"""
    out = {}
    t = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "TARGET_INFO_GPU" in t:
        have = [x for x in GCOLS if x in _cols(c, "TARGET_INFO_GPU")]
        rows = [dict(zip(have, r)) for r in c.execute(f"SELECT {','.join(have)} FROM TARGET_INFO_GPU")]
        rows = [r for r in rows if r.get("smCount")] or rows
        pick = [r for r in rows if r.get("cuDevice") == dev_id] if dev_id is not None and "cuDevice" in have else []
        g = (pick or rows or [{}])[0]
        if g.get("clockRate"):
            g["clockRate_MHz"] = round(g.pop("clockRate") / 1e6, 1)
        if g.get("memoryBandwidth"):
            g["memoryBandwidth_GBps"] = round(g.pop("memoryBandwidth") / 1e9, 1)
        out["gpu"] = g
    if "TARGET_INFO_SYSTEM_ENV" in t:
        out["system"] = {n: v for n, v in c.execute("SELECT name,value FROM TARGET_INFO_SYSTEM_ENV") if n in ENVKEYS}
    if "META_DATA_EXPORT" in t:
        out["nsys"] = {n: v for n, v in c.execute("SELECT name,value FROM META_DATA_EXPORT") if n == "EXPORT_PRODUCT_VERSION"}
    if "META_DATA_CAPTURE" in t:
        cap = {}
        for n, v in c.execute("SELECT name,value FROM META_DATA_CAPTURE"):
            if "GPU_METRICS" in n.upper() or re.match(r"PROCESS_0:(COMMAND|ARGUMENT_\d+|WORKING_DIR)$", n):
                cap[n] = v
        if cap:
            out["capture"] = cap
    if extra:
        out["user"] = extra
    return out


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_any(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            data = handle.read()
    else:
        with open(path, "rb") as handle:
            data = handle.read()
    try:
        import msgspec
        return msgspec.json.decode(data)
    except ImportError:
        return json.loads(data.decode("utf-8"))


def _validate_decoded_cache(d, dump_path, names_path, trace, summary):
    expected_dump = _sha256_file(dump_path)
    expected_names = _sha256_file(names_path)
    expected_decoder = _sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fi_lite_decode.py"))
    problems = []
    if summary.get("source_dump_sha256") != expected_dump:
        problems.append("source dump hash mismatch")
    if summary.get("names_sha256") != expected_names:
        problems.append("names/semantics hash mismatch")
    if summary.get("decoder_sha256") != expected_decoder:
        problems.append("decoder hash mismatch")
    records = sum(len(rows) for rows in d["slots"].values())
    if int(summary.get("records", -1)) != records:
        problems.append("record count mismatch: summary={} dump={}".format(summary.get("records"), records))
    software_slices = sum(1 for event in trace.get("traceEvents", [])
                          if event.get("ph") == "X" and event.get("cat") != "info")
    if int(summary.get("spans", -1)) != software_slices:
        problems.append("span count mismatch: summary={} trace={}".format(summary.get("spans"), software_slices))
    if (trace.get("otherData") or {}).get("meta") != d["meta"]:
        problems.append("trace metadata does not match dump metadata")
    if problems:
        raise SystemExit("decoded-trace cache rejected: " + "; ".join(problems))
    return expected_dump


def force_gz(path):
    """交付物一律是 Perfetto 可直接打开的 .perfetto.json.gz。"""
    if path.endswith(".gz"):
        return path
    print(f"提示：交付格式固定为 gzip，输出改为 {path}.gz")
    return path + ".gz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="?")
    ap.add_argument("--names")
    ap.add_argument("--sqlite")
    ap.add_argument("--kernel")
    ap.add_argument("--instance", type=int, default=-1)
    ap.add_argument("--metrics", default="tensor|sm active|sms active|issue|warps|occupan|dram|hbm|l2|throughput")
    ap.add_argument("--info-json", help="调用方补充的软件/环境信息（nvcc 版本、编译选项、git 提交…），原样写进 trace")
    ap.add_argument("--pad-us", type=float, default=50.0)
    ap.add_argument("--max-tail-us", type=float, default=50.0)
    ap.add_argument("--arch", default=None, help="强制指标语义映射：gh100/gb10x/gb11x（默认从 GPU 信息识别）")
    ap.add_argument("--metric-map-out", help="额外写出完整跨架构指标映射审计 JSON")
    ap.add_argument("--min-metrics-samples", type=int, default=3,
                    help="每条硬件曲线在所选 kernel 窗口内至少要有的采样点数（默认 3；不足就报错退出）")
    ap.add_argument("--min-metric-curves", type=int, default=10,
                    help="至少要合并几条硬件曲线（默认 10；不足就报错退出，可用 --metrics 放宽正则）")
    ap.add_argument("--allow-missing-metrics", action="store_true",
                    help="显式接受没有硬件曲线的 trace：照写文件，但 summary.align.ok=false 且退出码 2")
    ap.add_argument("--decoded-trace", help="复用已解码的软件 Perfetto JSON(.gz)，避免同一 dump 解码两次")
    ap.add_argument("--decoded-summary", help="--decoded-trace 对应的 decoder summary JSON")
    ap.add_argument("-o", "--out")
    ap.add_argument("--selftest", action="store_true", help="离线自检：合成 sqlite+转储跑一遍对齐")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    missing = [n for n, v in (("dump", a.dump), ("--names", a.names), ("--sqlite", a.sqlite),
                              ("--kernel", a.kernel), ("-o/--out", a.out)) if not v]
    if missing:
        ap.error("缺少参数：" + " ".join(missing) + "（或改用 --selftest 做离线自检）")
    a.out = force_gz(a.out)

    d = read_dump(a.dump)
    g_first = min(t for rs in d["slots"].values() for t, tag, aux in rs if tag & OPEN)
    g_last = max(rs[-1][0] for rs in d["slots"].values())
    if bool(a.decoded_trace) != bool(a.decoded_summary):
        ap.error("--decoded-trace 与 --decoded-summary 必须同时提供")
    if a.decoded_trace:
        trace = _load_json_any(a.decoded_trace)
        summ = _load_json_any(a.decoded_summary)
        cached_dump_sha256 = _validate_decoded_cache(d, a.dump, a.names, trace, summ)
        summ["decode_cache"] = {
            "reused": True,
            "trace": os.path.abspath(a.decoded_trace),
            "summary": os.path.abspath(a.decoded_summary),
            "source_dump_sha256": cached_dump_sha256,
        }
    else:
        trace, summ = decode(d, json.load(open(a.names, encoding="utf-8")))
    print_summary(summ)

    c = sqlite3.connect(a.sqlite)
    c.text_factory = lambda b: b.decode("utf-8", errors="replace")  # Windows 版 nsys 会把本地编码的时区名写进 sqlite
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        print("对齐失败：sqlite 里没有 CUPTI_ACTIVITY_KIND_KERNEL 表（只有 " + ", ".join(sorted(t for t in tables if t.isupper())[:8]) + " …）。\n"
              "常见原因：(1) nsys 在收尾写报告时被 GPU 锁的空转检查结束（看锁日志有无 rc=125）；"
              "(2) --capture-range-end=stop 与 --gpu-metrics-devices 同用；(3) 采集窗口里没有 kernel。\n"
              "修法：按 SKILL.md 的“采硬件”一节重采（--capture-range-end=none，持锁时设 GPULOCK_IDLE_KILL=240 并在日志写明原因）。", file=sys.stderr)
        sys.exit(3)
    ks = sorted((int(s), int(e), str(n)) for s, e, n in c.execute(
        "SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName")
        if re.search(a.kernel, str(n)))
    if not ks:
        raise SystemExit(f"sqlite 里没有匹配 {a.kernel!r} 的 kernel")
    if a.instance < -len(ks) or a.instance >= len(ks):
        raise SystemExit(f"--instance={a.instance} 越界：正则匹配到 {len(ks)} 个 kernel 实例（负索引从 -1 开始）")
    k0, k1, kname = ks[a.instance]
    offset = k0 - g_first
    tail_ns = k1 - (g_last + offset)
    align = dict(kernel=kname[:160], instance=a.instance, matched_instances=len(ks), kernel_us=round((k1 - k0) / 1e3, 3),
                 sw_span_us=round((g_last - g_first) / 1e3, 3), offset_ns=offset, end_margin_us=round(tail_ns / 1e3, 3),
                 ok=0 <= tail_ns <= a.max_tail_us * 1e3)
    print("对齐（软件轴 vs kernel 条）：", json.dumps(align, ensure_ascii=False))

    # 软件区间已是“相对最早一条记录”的 us；最早一条记录可能早于最早 OPEN 吗？不会：OPEN 是每泳道第一条。
    t_shift = (min(t for rs in d["slots"].values() for t, *_ in rs) - g_first) / 1e3  # 一般为 0
    ev = trace["traceEvents"]
    for e in ev:
        if "ts" in e:
            e["ts"] = round(e["ts"] + t_shift, 3)
    # 轴向自检：软件事件必须落在 nsys kernel 条的时间窗内（容差 = --max-tail-us）。
    # 只看结构的验收器发现不了整体错轴（fi_bugs_amd-sk-ds B20），所以这里自己响亮地失败。
    k_us = (k1 - k0) / 1e3
    sw_min = min(e["ts"] for e in ev if "ts" in e)
    sw_max = max(e["ts"] + float(e.get("dur") or 0.0) for e in ev if "ts" in e)
    if sw_min < -a.max_tail_us or sw_max > k_us + a.max_tail_us:
        raise SystemExit(
            f"时间轴自检失败：软件事件在 {sw_min:.3f}~{sw_max:.3f} us，nsys kernel 条在 0~{k_us:.3f} us，"
            f"两者不重叠（容差 ±{a.max_tail_us} us）。\n"
            "常见原因：(1) --instance/--kernel 选到了另一次 launch；(2) 转储 meta 的计时频率与实际不符；"
            "(3) 转储与 sqlite 不是同一次采集。\n"
            f"当前 align：kernel_us={align['kernel_us']}，sw_span_us={align['sw_span_us']}，offset_ns={offset}。")
    align["axis"] = "软件轨 ts 相对第一条 FI_OPEN；nsys kernel 条相对该 kernel 实例开始时刻"
    align["software_min_us"], align["software_max_us"] = round(sw_min, 3), round(sw_max, 3)
    align["axis_check"] = {"tolerance_us": a.max_tail_us, "kernel_us": round(k_us, 3), "ok": True}

    # 遵守渲染契约 R6：把全部软件进程 pid 整体平移到 >= 100，将 pid=1（sort_index=-1000）独占预留给最顶层硬件信息与 [0-100%] 利用率轨，
    # 且把硬件元数据与事件排在 traceEvents 最前面，保证 Perfetto UI 与 TraceProcessor (upid=1) 永远置顶显示在第一行。
    SW_PID_SHIFT = 100
    for e in ev:
        if isinstance(e.get("pid"), int):
            e["pid"] = int(e["pid"]) + SW_PID_SHIFT
            if e.get("ph") == "M" and e.get("name") == "process_sort_index":
                args_m = e.get("args") or {}
                if isinstance(args_m.get("sort_index"), (int, float)):
                    args_m["sort_index"] = int(args_m["sort_index"]) + SW_PID_SHIFT

    HW_PID = 1
    hw_ev = []
    # Perfetto rejects negative timestamps. Emit only the non-negative side of the requested pad window.
    lo, hi = k0, k1 + a.pad_us * 1e3
    nsys_k_rows = sorted(
        (int(s), int(e), str(n))
        for s, e, n in c.execute(
            "SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName "
            "WHERE k.end>=? AND k.start<=?", (int(lo), int(hi))
        )
    )
    # GPU metrics 曲线
    M = {int(i): str(n) for i, n in c.execute("SELECT metricId,metricName FROM TARGET_INFO_GPU_METRICS")} if c.execute(
        "SELECT name FROM sqlite_master WHERE name='TARGET_INFO_GPU_METRICS'").fetchone() else {}
    pick = {i: n for i, n in M.items() if re.search(a.metrics, n, re.I)}
    hw_counts, hw_period_us = {}, None
    hw_c_events = []
    if pick:
        qs = ",".join(str(i) for i in pick)
        tss = []
        for ts, m, v in c.execute(f"SELECT timestamp,metricId,value FROM GPU_METRICS WHERE metricId IN ({qs}) "
                                  "AND timestamp BETWEEN ? AND ? ORDER BY timestamp", (int(lo), int(hi))):
            n = pick[int(m)]
            hw_counts[n] = hw_counts.get(n, 0) + 1
            if not tss or tss[-1] != int(ts):
                tss.append(int(ts))
            hw_c_events.append(dict(ph="C", pid=HW_PID, ts=round((int(ts) - k0) / 1e3, 3), name=n, args=dict(value=float(v))))
        if len(tss) > 2:
            hw_period_us = round(sorted(b - a for a, b in zip(tss, tss[1:]))[len(tss) // 2] / 1e3, 3)
        print(f"硬件曲线：{len(hw_counts)} 条，kernel±{a.pad_us:g}us 窗口内每条 {max(hw_counts.values(), default=0)} 个采样点，"
              f"采样间隔中位数 {hw_period_us} us")
    else:
        print("提示：sqlite 里没有匹配的 GPU metrics（采集时没开 --gpu-metrics-devices？），只合并 kernel 条；"
              "trace 里只有软件事件，没有硬件曲线")

    # ---- 硬件曲线覆盖判定：不足就不许出一张没有曲线的 trace ----
    has_metrics_table = bool(c.execute("SELECT name FROM sqlite_master WHERE name='GPU_METRICS'").fetchone())
    n_metric_rows = int(c.execute("SELECT COUNT(*) FROM GPU_METRICS").fetchone()[0]) if has_metrics_table else 0
    coverage = gpu_metrics_coverage(c, set(pick), ks, k0, int(a.pad_us * 1e3), a.min_metrics_samples)         if has_metrics_table else {}
    curves = len(hw_counts)
    min_samples = min(hw_counts.values()) if hw_counts else 0
    metrics_problems = []
    if not has_metrics_table or n_metric_rows == 0:
        metrics_problems.append("sqlite 里没有 GPU_METRICS 数据：采集时没开 --gpu-metrics-devices，"
                                "或采集在收尾前被掐断（看 GPU 锁日志有无 rc=125）")
    elif not pick:
        metrics_problems.append(f"GPU_METRICS 有 {n_metric_rows} 行，但 --metrics 正则 {a.metrics!r} 一条都没匹配上；"
                                f"sqlite 里可选的指标名有 {len(M)} 个（前 8 个：{sorted(M.values())[:8]}）")
    else:
        if curves < a.min_metric_curves:
            metrics_problems.append(f"硬件曲线只有 {curves} 条，少于要求的 {a.min_metric_curves} 条")
        if min_samples < a.min_metrics_samples:
            metrics_problems.append(f"所选 kernel 窗口内每条曲线最少只有 {min_samples} 个采样点，"
                                    f"少于要求的 {a.min_metrics_samples} 个")
    metrics_ok = not metrics_problems
    align["tail_ok"] = align["ok"]
    align["metrics_in_window"] = hw_counts
    align["metrics_curves"] = curves
    align["metrics_min_samples_per_curve"] = min_samples
    align["metrics_required"] = {"curves": a.min_metric_curves, "samples_per_curve": a.min_metrics_samples}
    align["metrics_coverage"] = coverage or None
    align["metrics_ok"] = metrics_ok
    align["ok"] = bool(align["tail_ok"] and metrics_ok)
    print("对齐结论：", json.dumps({k: align.get(k) for k in
                                    ("kernel", "instance", "kernel_us", "sw_span_us", "end_margin_us",
                                     "tail_ok", "metrics_curves", "metrics_min_samples_per_curve",
                                     "metrics_ok", "ok")}, ensure_ascii=False))
    if not metrics_ok:
        align["metrics_problems"] = metrics_problems
        print("硬件曲线覆盖不足：", file=sys.stderr)
        for m in metrics_problems:
            print("  -", m, file=sys.stderr)
        if coverage:
            segs = coverage.get("segments") or []
            print(f"  采样段 {len(segs)} 段（相对所选 kernel 开始时刻，us）："
                  + "; ".join(f"[{s['start_us']} ~ {s['end_us']}] {s['samples']} 点" for s in segs[:8])
                  + ("…" if len(segs) > 8 else ""), file=sys.stderr)
            print(f"  采样间隔中位数 {coverage.get('sample_period_us')} us，最大断档 {coverage.get('gap_max_us')} us；"
                  f"扫描时间戳 {coverage.get('timestamps_scanned')} 个"
                  + ("（已达扫描上限，结论按前缀）" if coverage.get("scan_truncated") else ""), file=sys.stderr)
            print(f"  匹配到 {coverage.get('instances_total')} 个 kernel 实例，窗口内样本 >= "
                  f"{a.min_metrics_samples} 的有 {coverage.get('instances_covered')} 个", file=sys.stderr)
            for pi in coverage.get("per_instance", [])[:12]:
                print(f"    [{'*' if pi['selected'] else ' '}] #{pi['index']} {pi['name'][:60]} "
                      f"{pi['start_us']}~{pi['end_us']} us 样本 {pi['samples_in_window']}"
                      f"{'' if pi['covered'] else '  <-- 落在断档里'}", file=sys.stderr)
        print("  修法：高采样频率下 nsys 的 GPU metrics 采样缓冲区跟不上，有两种失效形态，都会让 kernel 窗口里"
              "一个样本都没有——(1) **分段**采集：200 kHz 时 H20 上约 600 ms 有样本后断约 1100 ms；"
              "(2) **溢出后停采**：打到约 0.6 s 就 'GPU Metrics [0]: Sampling buffer overflow'，"
              "此后不再采（采样段只有一段、末尾停在 kernel 之前，>1 ms 的空洞列表反而是空的）。"
              "先去 nsys 的日志里搜 Sampling buffer overflow 分辨是哪一种。"
              "优先在 Python/Triton/CUDA 驱动里给目标 steady-state launch 包上 cudaProfilerStart/Stop，"
              "并用 --capture-range=cudaProfilerApi --capture-range-end=stop 把窗口压到亚秒级，继续用 200 kHz"
              "（--gpu-metrics-frequency=200000，5 us 间隔）重采并核查覆盖；"
              "只有在单次稳态 launch 本身超过数百毫秒且仍发生 buffer overflow 时，才退回较低采样率（例如 100000 或 50000）。"
              "确实只要软件时间线、不要硬件曲线时，显式加 --allow-missing-metrics"
              "（summary.align.ok 会记 false，退出码 2）。", file=sys.stderr)
        summ["align"] = align
        write_json(summ, a.out.replace(".gz", "").replace(".json", "") + ".summary.json")
        if not a.allow_missing_metrics:
            print(f"没有写出 {a.out}：禁止静默产出没有硬件曲线的 trace。summary 已写出，align.ok=false。",
                  file=sys.stderr)
            return 4
    # 软硬件信息：既写进 otherData，也作为一条可点开的轨道放在最上面（Perfetto 里点它看 args）
    extra = json.load(open(a.info_json, encoding="utf-8")) if a.info_json else None
    launch = kernel_launch_info(c, k0, kname)
    info = hw_sw_info(c, launch.get("deviceId"), extra)
    info["launch"] = launch
    info["instrument"] = dict(tool="fast-instrument", meta=d["meta"], timer=summ.get("timer"),
                              timer_min_step_ns=summ.get("timer_min_step_ns"), align_version=ALIGN_VERSION)
    # 把“指标名 -> 语义”的证据写进 trace；attrib 的跨架构阈值与这一层保持一致。
    metric_names = sorted(hw_counts)
    arch = a.arch or detect_arch(info.get("gpu", {}))
    metric_mapping = None
    if arch:
        try:
            mm = match_metric_names(metric_names, arch)
            metric_mapping = {
                "schema_version": "fi-metric-map",
                "selected_arch": mm.get("arch"),
                "runtime_verified_on_machine": mm.get("arch") == "gh100" and not a.arch,
                "exact": mm.get("exact", {}),
                "fallback": mm.get("fallback", {}),
                "missing": mm.get("missing", []),
                "unclassified": mm.get("unclassified", []),
                "thresholds": mm.get("thresholds", {}),
                "config_audit": mm.get("config_audit", {}),
                "scope_warning": ("GPU metrics 是整卡聚合/平均；不能定位到单个 SM/warp，"
                                  f"{hw_period_us if hw_period_us else '采样间隔未知'} us 的采样会平滑更短的事件。"),
            }
        except Exception as exc:
            metric_mapping = {"error": f"{type(exc).__name__}: {exc}", "selected_arch": arch}
    info["arch"] = arch
    info["metric_mapping"] = metric_mapping
    info["hw_metrics"] = dict(names=metric_names, samples_in_window=hw_counts, period_us=hw_period_us,
                              scope="整卡（所有 SM 的聚合），不分 SM/warp",
                              metric_mapping_ref="otherData.hw_sw_info.metric_mapping",
                              metrics_ok=metrics_ok, curves=curves,
                              min_samples_per_curve=min_samples,
                              required=dict(curves=a.min_metric_curves, samples_per_curve=a.min_metrics_samples),
                              problems=align.get("metrics_problems") or [],
                              coverage=coverage or None)
    g = info.get("gpu", {})
    gpu_label = g.get("name", "NVIDIA GPU")
    hw_proc_title = (
        f"00 · 硬件信息与利用率 [0–100%]（{gpu_label} · nsys 实测）"
        if hw_c_events else
        f"00 · 硬件信息与 Kernel 时间窗（{gpu_label} · nsys 实测）"
    )
    hw_ev.append(dict(ph="M", pid=HW_PID, name="process_name", args=dict(name=hw_proc_title)))
    hw_ev.append(dict(ph="M", pid=HW_PID, name="process_sort_index", args=dict(sort_index=-1000)))
    hw_ev.append(dict(ph="M", pid=HW_PID, tid=1, name="thread_name", args=dict(name="GPU 硬件规格与配置（点开查看 args）")))
    hw_ev.append(dict(ph="M", pid=HW_PID, tid=1, name="thread_sort_index", args=dict(sort_index=1)))
    title = f"{gpu_label} · {g.get('smCount', '?')} SM · grid {launch.get('gridX', '?')}x{launch.get('gridY', '?')}" \
            f" block {launch.get('blockX', '?')} · reg {launch.get('registersPerThread', '?')}"
    hw_ev.append(dict(ph="X", pid=HW_PID, tid=1, ts=0.0, dur=round((k1 - k0) / 1e3, 3), name=title, cat="info",
                      args=json.loads(json.dumps(info, default=str))))
    # 按非重叠子泳道发射 nsys_kernel 硬件窗口切片（从 tid=2 起，避免 tid=0 碰撞与多流重叠溢出）
    k_free_end = []
    for s, e, n in nsys_k_rows:
        ts_u = round((s - k0) / 1e3, 3)
        end_u = round((e - k0) / 1e3, 3)
        dur_u = round(max(0.0, end_u - ts_u), 3)
        sub_k = next((idx for idx, fe in enumerate(k_free_end) if fe <= ts_u), None)
        if sub_k is None:
            k_free_end.append(ts_u + dur_u)
            sub_k = len(k_free_end) - 1
        else:
            k_free_end[sub_k] = ts_u + dur_u
        ktid = 2 + sub_k
        if sub_k == 0 or not any(x.get("ph") == "M" and x.get("tid") == ktid and x.get("name") == "thread_name" for x in hw_ev):
            hw_ev.append(dict(ph="M", pid=HW_PID, tid=ktid, name="thread_name",
                              args=dict(name="nsys：硬件 Kernel 时间窗" if sub_k == 0 else f"nsys：硬件 Kernel 时间窗 #{sub_k + 1}")))
            hw_ev.append(dict(ph="M", pid=HW_PID, tid=ktid, name="thread_sort_index", args=dict(sort_index=ktid)))
        hw_ev.append(dict(ph="X", pid=HW_PID, tid=ktid, ts=ts_u, dur=dur_u,
                          name=str(n).split("(")[0][-80:], cat="nsys_kernel"))
    hw_ev.extend(hw_c_events)
    trace["traceEvents"] = hw_ev + ev
    trace["otherData"]["schema_version"] = "fi-perfetto"
    trace["otherData"]["hw_sw_info"] = info
    trace["otherData"]["metric_mapping"] = metric_mapping
    summ["hw_sw_info"] = info
    trace["otherData"]["align"] = align
    summ["align"] = align
    trace["otherData"]["time_zero"] = "nsys 中该 kernel 实例开始时刻"
    summ["metric_mapping"] = metric_mapping
    write_json(trace, a.out)
    write_json(summ, a.out.replace(".gz", "").replace(".json", "") + ".summary.json")
    if a.metric_map_out:
        write_json(mapping_document(arch), a.metric_map_out)
        print("写出指标映射审计", a.metric_map_out)
    print("写出", a.out)
    return 0 if align["ok"] and not summ["errors"] else 2


SELFTEST_NAMES = {"events": {"1": "装载 tile", "2": "计算 tile"}, "roles": {"0": "链 CTA"},
                  "aux": {"1": "tile"}, "label_aux": True, "group": "role"}


def _selftest_write_dump(path, slots, n_slots=2, cap=16, timer=0):
    """按 FILITE02 布局写一份合成转储（与 fi_lite 头部导出一致，供自检用）。"""
    import struct
    meta = json.dumps({"kernel": "demo_kernel", "grid": n_slots, "gran": 0, "slot": "warp"}, ensure_ascii=False).encode("utf-8")
    blob = b"FILITE02" + struct.pack("<6I", n_slots, cap, 1, 0, timer, len(meta)) + meta
    blob += struct.pack(f"<{n_slots}I", *[len(slots.get(i, [])) for i in range(n_slots)])
    for i in range(n_slots):
        for rec in slots.get(i, []):
            blob += struct.pack("<QII", *rec)
    with open(path, "wb") as fh:
        fh.write(blob)


def _selftest_write_db(path, k0, k1, with_metrics=True):
    """合成一份 nsys 导出 sqlite：kernel 表 + StringIds + GPU/环境信息 + GPU metrics。"""
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    c.execute("INSERT INTO StringIds VALUES (11, 'demo_kernel(float const*, int)'), (12, 'other_kernel(int)')")
    c.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
        start INTEGER, end INTEGER, demangledName INTEGER, deviceId INTEGER, streamId INTEGER,
        gridX INTEGER, gridY INTEGER, gridZ INTEGER, blockX INTEGER, blockY INTEGER, blockZ INTEGER,
        registersPerThread INTEGER, staticSharedMemory INTEGER, dynamicSharedMemory INTEGER)""")
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,11,0,7,4,1,1,256,1,1,64,1024,0)", (int(k0), int(k1)))
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,12,0,7,1,1,1,64,1,1,24,0,0)",
              (int(k0) + 40_000, int(k0) + 42_000))
    c.execute("CREATE TABLE TARGET_INFO_GPU (deviceId INTEGER, cuDevice INTEGER, name TEXT, chipName TEXT, smCount INTEGER,"
              " clockRate INTEGER, memoryBandwidth INTEGER, l2CacheSize INTEGER, computeMajor INTEGER, computeMinor INTEGER,"
              " maxWarpsPerSm INTEGER)")
    c.execute("INSERT INTO TARGET_INFO_GPU VALUES (0,0,'NVIDIA H20','GH100',78,1980000,4000000000000,603979776,9,0,64)")
    c.execute("CREATE TABLE TARGET_INFO_SYSTEM_ENV (name TEXT, value TEXT)")
    c.executemany("INSERT INTO TARGET_INFO_SYSTEM_ENV VALUES (?,?)",
                  [("Hostname", "selftest"), ("CudaDriverVersion", "13.3"), ("SupportsGpuMetrics", "1")])
    c.execute("CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT)")
    c.execute("INSERT INTO META_DATA_EXPORT VALUES ('EXPORT_PRODUCT_VERSION', 'selftest-nsys')")
    if with_metrics:
        c.execute("CREATE TABLE TARGET_INFO_GPU_METRICS (metricId INTEGER, metricName TEXT)")
        c.executemany("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?,?)",
                      [(1, "dram__bytes_read.sum.pct_of_peak_sustained_elapsed"),
                       (2, "sm__warps_active.avg.pct_of_peak_sustained_active"),
                       (3, "some_unrelated_metric")])
        c.execute("CREATE TABLE GPU_METRICS (timestamp INTEGER, metricId INTEGER, value REAL)")
        rows = []
        for i in range(8):
            ts = int(k0) + i * 1000
            rows += [(ts, 1, 3.5 + i), (ts, 2, 40.0 + i)]
        c.executemany("INSERT INTO GPU_METRICS VALUES (?,?,?)", rows)
    c.commit()
    c.close()


def _selftest_add_segmented_metrics(path, k0):
    """往合成 sqlite 里加分段采集的 GPU metrics：样本只到 kernel 之前 700 ms 就断档，
    复刻 nsys 200 kHz 分段采集把 kernel 落在间隙里的真实形态（协调者实测：H20 约 600 ms 有样本后断约 1100 ms）。"""
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE TARGET_INFO_GPU_METRICS (metricId INTEGER, metricName TEXT)")
    c.executemany("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?,?)",
                  [(1, "dram__bytes_read.sum.pct_of_peak_sustained_elapsed"),
                   (2, "sm__warps_active.avg.pct_of_peak_sustained_active")])
    c.execute("CREATE TABLE GPU_METRICS (timestamp INTEGER, metricId INTEGER, value REAL)")
    rows = []
    base = int(k0) - 700_000_000            # kernel 前 700 ms 开始采
    for seg in range(2):                    # 两段，中间断 1.1 s：kernel 落在断档里
        for i in range(30):
            ts = base + seg * 1_800_000_000 + i * 5_000     # 5 us 间隔 = 200 kHz
            rows += [(ts, 1, 3.5 + i), (ts, 2, 40.0 + i)]
    c.executemany("INSERT INTO GPU_METRICS VALUES (?,?,?)", rows)
    c.commit()
    c.close()


def selftest() -> int:
    """离线自检：合成 nsys sqlite + FILITE02 转储，跑完整 CLI，验证偏移算术、余量校验、
    硬件曲线合并与失败退出码。不需要 nsys、不需要 GPU。"""
    import subprocess
    import tempfile

    END, OPEN, MARK, CLOSE = 0x80000000, 0x40000000, 0x20000000, 0x08000000
    fails = []
    tmp = tempfile.mkdtemp(prefix="fi_align_nsys_selftest_")
    dump = os.path.join(tmp, "demo.bin")
    names = os.path.join(tmp, "names.json")
    db = os.path.join(tmp, "cap.sqlite")
    out = os.path.join(tmp, "merged.perfetto.json")
    with open(names, "w", encoding="utf-8") as fh:
        json.dump(SELFTEST_NAMES, fh, ensure_ascii=False)

    # 软件轴（globaltimer，ns）：最早 OPEN=1_000_000，最晚记录=1_006_000 -> 跨度 6 us
    slots = {0: [(1_000_000, OPEN | 0, 5), (1_000_500, 1, 0), (1_002_000, END | 1, 0), (1_006_000, CLOSE, 0)],
             1: [(1_000_200, OPEN | 0, 6), (1_001_000, 2, 0), (1_004_000, END | 2, 0), (1_005_000, CLOSE, 0)]}
    _selftest_write_dump(dump, slots)

    # kernel 8 us：offset=k0-g_first，尾余量 = 8us - 6us = 2 us -> 对齐成功
    k0, k1 = 5_000_000_000, 5_000_008_000
    _selftest_write_db(db, k0, k1)
    p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), dump, "--names", names,
                        "--sqlite", db, "--kernel", "demo_kernel", "--min-metric-curves", "2", "-o", out],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    if p.returncode != 0:
        fails.append(f"成功用例退出码 {p.returncode}: " + p.stdout)
    elif not os.path.exists(out + ".gz"):
        fails.append("成功用例没写出 .perfetto.json.gz")
    else:
        import gzip
        trace = json.loads(gzip.decompress(open(out + ".gz", "rb").read()).decode("utf-8"))
        al = trace.get("otherData", {}).get("align", {})
        if not al.get("ok"):
            fails.append(f"align.ok={al}")
        if al.get("offset_ns") != k0 - 1_000_000:
            fails.append(f"offset_ns={al.get('offset_ns')}, 期望 {k0 - 1_000_000}")
        if abs(al.get("end_margin_us", -1) - 2.0) > 1e-6:
            fails.append(f"end_margin_us={al.get('end_margin_us')}, 期望 2.0")
        if abs(al.get("kernel_us", -1) - 8.0) > 1e-6 or abs(al.get("sw_span_us", -1) - 6.0) > 1e-6:
            fails.append(f"kernel_us/sw_span_us={al.get('kernel_us')}/{al.get('sw_span_us')}")
        ev = trace["traceEvents"]
        nk = [e for e in ev if e.get("cat") == "nsys_kernel"]
        if len(nk) != 2 or any(abs(e["ts"]) < 1e-9 and abs(e["dur"] - 8.0) > 1e-6 for e in nk if e["ts"] != 0.0):
            fails.append(f"nsys kernel 条不对：{nk}")
        if not any(abs(e["ts"]) < 1e-9 for e in nk):
            fails.append("被插桩 kernel 没被平移到 ts=0")
        cs = [e for e in ev if e.get("ph") == "C"]
        if len(cs) != 16 or {e["name"] for e in cs} != {"dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
                                                        "sm__warps_active.avg.pct_of_peak_sustained_active"}:
            fails.append(f"硬件曲线点数/名字不对：{len(cs)} {sorted({e['name'] for e in cs})}")
        info = trace["otherData"].get("hw_sw_info", {})
        if info.get("launch", {}).get("gridX") != 4 or info.get("gpu", {}).get("smCount") != 78:
            fails.append(f"软硬件信息没带出来：{ {k: info.get(k) for k in ('launch', 'gpu')} }")
        if info.get("nsys", {}).get("EXPORT_PRODUCT_VERSION") != "selftest-nsys":
            fails.append(f"nsys 版本没带出来：{info.get('nsys')}")
        if not info.get("hw_metrics", {}).get("samples_in_window"):
            fails.append(f"hw_metrics 采样计数为空：{info.get('hw_metrics')}")
        sw = [e for e in ev if e.get("ph") == "X" and e.get("cat") not in ("nsys_kernel", "info")]
        if len(sw) != 2 or min(e["ts"] for e in sw) < -1e-9:
            fails.append(f"软件区间不对：{sw}")
        summ_path = out.replace(".json", "") + ".summary.json"   # 与 write_json 的去后缀规则一致
        if not os.path.exists(summ_path):
            fails.append("没写出 summary.json")
        elif json.load(open(summ_path, encoding="utf-8")).get("align", {}).get("ok") is not True:
            fails.append("summary.json 里没有 align.ok=true")

    # 失败用例：kernel 窗口在软件记录结束前就关了 -> 尾余量为负 -> 退出码 2
    db2 = os.path.join(tmp, "cap_bad.sqlite")
    out2 = os.path.join(tmp, "bad.perfetto.json")
    _selftest_write_db(db2, k0, k0 + 3_000)
    p2 = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), dump, "--names", names,
                         "--sqlite", db2, "--kernel", "demo_kernel", "--min-metric-curves", "2", "-o", out2],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    if p2.returncode != 2:
        fails.append(f"尾余量为负时退出码 {p2.returncode}（期望 2）")
    if "end_margin_us" not in p2.stdout:
        fails.append("失败用例没打印对齐摘要")

    # 失败用例：软件跨度远超 kernel 条（时间轴没对上）-> 必须响亮失败，不能出一张错轴的图
    dump_bad = os.path.join(tmp, "demo_bad.bin")
    _selftest_write_dump(dump_bad, {0: [(1_000_000, OPEN | 0, 5), (1_000_500, 1, 0),
                                        (10_000_000_000, END | 1, 0), (10_000_001_000, CLOSE, 0)]})
    p5 = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), dump_bad, "--names", names,
                         "--sqlite", db, "--kernel", "demo_kernel", "--min-metric-curves", "2", "-o", out2],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    if p5.returncode == 0 or "时间轴自检失败" not in p5.stdout:
        fails.append(f"错轴用例退出码 {p5.returncode}，且应打印时间轴自检失败：{p5.stdout[:300]}")

    # 失败用例：GPU metrics 分段采集，kernel 落在断档里 -> 曲线 0 条，必须非 0 退出且不写 trace
    db_gap = os.path.join(tmp, "cap_gap.sqlite")
    out_gap = os.path.join(tmp, "gap.perfetto.json")
    _selftest_write_db(db_gap, k0, k1, with_metrics=False)
    _selftest_add_segmented_metrics(db_gap, k0)
    p6 = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), dump, "--names", names,
                         "--sqlite", db_gap, "--kernel", "demo_kernel",
                         "--min-metric-curves", "2", "-o", out_gap],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    if p6.returncode != 4:
        fails.append(f"kernel 落在采样断档时退出码 {p6.returncode}（期望 4）：{p6.stdout[-600:]}")
    if os.path.exists(out_gap + ".gz"):
        fails.append("kernel 落在采样断档时不该写出 trace")
    if "采样段" not in p6.stdout or "gpu-metrics-frequency" not in p6.stdout:
        fails.append(f"断档用例没打印采样段与降频建议：{p6.stdout[-600:]}")
    sp = out_gap.replace(".json", "") + ".summary.json"
    if not os.path.exists(sp):
        fails.append("断档用例没写出 summary.json")
    else:
        sal = json.load(open(sp, encoding="utf-8")).get("align", {})
        if sal.get("ok") is not False or sal.get("metrics_ok") is not False:
            fails.append(f"断档用例 summary.align.ok/metrics_ok 应为 false：{ {k: sal.get(k) for k in ('ok','metrics_ok','tail_ok')} }")
        if "metrics_in_window" not in sal:
            fails.append("summary.align 里没有 metrics_in_window")
        cov = sal.get("metrics_coverage") or {}
        if cov.get("instances_covered") != 0 or len(cov.get("segments") or []) < 2:
            fails.append(f"断档诊断不对：instances_covered={cov.get('instances_covered')} segments={cov.get('segments')}")
    # 同一份数据显式放行时必须照写文件、退出码 2、ok=false
    p7 = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), dump, "--names", names,
                         "--sqlite", db_gap, "--kernel", "demo_kernel", "--min-metric-curves", "2",
                         "--allow-missing-metrics", "-o", out_gap],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    if p7.returncode != 2 or not os.path.exists(out_gap + ".gz"):
        fails.append(f"--allow-missing-metrics 时应写出 trace 且退出码 2，实得 {p7.returncode}")
    else:
        import gzip as _gz
        al7 = json.loads(_gz.decompress(open(out_gap + ".gz", "rb").read()).decode("utf-8"))["otherData"]["align"]
        if al7.get("ok") is not False or al7.get("metrics_ok") is not False:
            fails.append(f"--allow-missing-metrics 的 trace 里 align.ok 应为 false：{al7.get('ok')}")

    # 失败用例：正则匹配不到 kernel
    p3 = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), dump, "--names", names,
                         "--sqlite", db, "--kernel", "no_such_kernel", "--min-metric-curves", "2", "-o", out2],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    if p3.returncode == 0:
        fails.append("匹配不到 kernel 时不该成功")

    # 失败用例：sqlite 里没有 kernel 表（采集被锁掐掉的典型症状）
    db3 = os.path.join(tmp, "cap_empty.sqlite")
    _c3 = sqlite3.connect(db3)
    _c3.execute("CREATE TABLE FOO (a INTEGER)")
    _c3.commit()
    _c3.close()
    p4 = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), dump, "--names", names,
                         "--sqlite", db3, "--kernel", "demo_kernel", "--min-metric-curves", "2", "-o", out2],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    if p4.returncode != 3 or "CUPTI_ACTIVITY_KIND_KERNEL" not in p4.stdout:
        fails.append(f"缺 kernel 表时退出码 {p4.returncode}，且应给出诊断")

    for f in os.listdir(tmp):
        try:
            os.remove(os.path.join(tmp, f))
        except OSError:
            pass
    os.rmdir(tmp)

    if fails:
        print("[fi_align_nsys SELFTEST] FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("[fi_align_nsys SELFTEST] PASS（合成 sqlite+转储：偏移算术、尾余量校验、kernel 条与 GPU metrics 合并、"
          "软硬件信息、三类失败退出码）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
