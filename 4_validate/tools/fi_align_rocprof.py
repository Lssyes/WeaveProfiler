#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fast-instrument / fi_align_rocprof.py：把 FILITE02 软件打点与 rocprofv3 采集对齐合并为一个 Perfetto JSON。

输出轨道（时间轴统一为“rocprofv3 里被对齐 kernel 实例的开始 = 0 us”）：
  - 软件泳道：fi_lite_decode 解出的每 wave/CTA/CU 语义区间与瞬时点；
  - “rocprofv3：KERNEL_DISPATCH”：时间窗内所有 kernel 条；
  - “硬件：PMC 计数器（每 dispatch 聚合）”：--pmc 采到的计数器，按 dispatch 关联，瞬时点携带全部计数器值；
  - “软硬件信息（点开看 args）”：GPU 型号/ID、grid/block、计数器列表、调用方补充信息。

对齐方法：软件时间（已由解码器统一换算成 ns；FI_TIMER=0 的 s_memrealtime tick 按 meta.timer_hz 换算）与
rocprofv3 的 dispatch 起止同属墙钟时基，相差一个常数偏移：
    offset = kernel_start(rocprofv3) - 最早一条 FI_OPEN 的 ns 时间戳。
校验：对齐后最后一条软件记录必须落在 kernel_end 之前，余量 end_margin_us 写进 summary；
余量为负或大于 --max-tail-us 视为对齐失败（找错 kernel 实例 / 插桩没覆盖收尾），退出码 2。

计数器来源（能采到什么合什么）：
  - rocprofv3 --pmc C0,C1,... 的 counter_collection（JSON 或 CSV 输出），按 dispatch/correlation id 关联到 kernel 条；
  - 本工具不依赖 rocprofv3 的版本细节：递归遍历 JSON，凡带数值 value 与计数器名/ID 的记录都尝试识别；
  - rocprofv3 没有 nsys GPU metrics 那样的整卡周期采样曲线；--pc-sampling 是 PC 采样（beta），不是计数器时间序列。
    需要整卡曲线时用 rocm-smi --show* 手工补采，本工具如实写明“未合入”的原因。

用法：
  python3 -I fi_align_rocprof.py dump.bin --names names.json --rocprof <rocprofv3 输出目录或 JSON> \
      --kernel '<kernel 名正则>' [--instance -1] [--pmc <计数器 JSON/CSV 或目录，默认同 --rocprof>] \
      [--info-json info.json] [--time-unit auto] -o merged.perfetto.json.gz
  python3 -I fi_align_rocprof.py --selftest
"""
import argparse
import csv
import gzip
import json
import re
import struct
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fi_lite_decode import OPEN, decode, print_summary, read_dump, write_json  # noqa: E402

START_KEYS = ("start_timestamp", "start_ns", "start", "begin")
END_KEYS = ("end_timestamp", "end_ns", "end", "stop", "finish")
TAG_KEYS = ("kind", "category", "cat", "domain", "record_type", "type", "event_type")
NAME_KEYS = ("kernel_name", "demangled_name", "mangled_name", "symbol", "function", "function_name", "name")
NESTED_NAME_KEYS = ("kernel", "dispatch", "operation")
UNIT_FACTORS = {"ns": 1, "nsec": 1, "us": 1000, "usec": 1000, "msec": 1000000, "ms": 1000000,
                "sec": 1000000000, "s": 1000000000}
# 计数器记录里可能携带的 dispatch 关联字段与数值字段
LINK_KEYS = ("dispatch_id", "correlation_id", "dispatch", "dispatchId", "correlation")
VALUE_KEYS = ("value", "counter_value", "count", "accumulation", "acc")
CID_KEYS = ("counter_id", "counterId", "id", "handle")
CNAME_KEYS = ("counter_name", "counter", "metric_name", "metric", "name")


def json_files(path):
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.exists():
        raise SystemExit(f"rocprof 路径不存在: {p}")
    return sorted(q for q in p.rglob("*") if q.is_file() and
                  (q.suffix == ".json" or q.name.endswith(".json.gz")))


def load_json(path):
    p = Path(path)
    if p.suffix == ".gz" or p.name.endswith(".json.gz"):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def walk_json(obj, context=()):
    if isinstance(obj, dict):
        yield obj, context
        for k, v in obj.items():
            yield from walk_json(v, context + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_json(v, context + (str(i),))


def _text_ns(value, unit):
    """时间字段 -> 整数 ns。

    rocprofv3 的时间戳是墙钟纪元纳秒（1.79e18 量级），已经超出 float64 能精确表示的整数范围
    （2^53 ≈ 9.0e15）：走 float 会把 kernel 时长量化到 256 ns（同一份数据算出 445.696 us，
    真值 445.760 us），end_margin_us 也带同量级台阶（fi_bugs_amd-sk-qwen B21）。
    所以整数与整数样式的字符串一律走整数算术，只有真带小数点的值才退回 float。
    """
    try:
        f = UNIT_FACTORS[unit]
    except (KeyError, TypeError) as e:
        raise SystemExit(f"未知时间单位: {unit!r}（可用 {sorted(UNIT_FACTORS)}）") from e
    if isinstance(value, bool):
        raise SystemExit(f"rocprof 时间字段不是数字: {value!r} ({unit})")
    if isinstance(value, int):
        return value * f
    if isinstance(value, str):
        s = value.strip()
        try:
            return int(s) * f
        except ValueError:
            pass
    try:
        return int(round(float(value) * f))
    except (TypeError, ValueError) as e:
        raise SystemExit(f"rocprof 时间字段不是数字: {value!r} ({unit})") from e


def _time_pair(rec, unit):
    if unit == "auto":
        # Chrome trace 的 ts/dur 是 us；SDK timestamp / *_ns 字段是 ns。
        if "ts" in rec and not any(k in rec for k in ("start_timestamp", "start_ns", "end_timestamp", "end_ns")):
            unit = "us"
        else:
            unit = "ns"
    for sk, ek in zip(START_KEYS, END_KEYS):
        if sk in rec and ek in rec:
            return _text_ns(rec[sk], unit), _text_ns(rec[ek], unit)
    if "ts" in rec:
        dur = rec.get("dur", rec.get("duration"))
        if dur is not None:
            return _text_ns(rec["ts"], unit), _text_ns(rec["ts"], unit) + _text_ns(dur, unit)
    return None


def _is_kernel_dispatch(rec, context):
    tag = " ".join(str(rec.get(k, "")) for k in TAG_KEYS)
    blob = " ".join(context) + " " + tag
    if "KERNEL_DISPATCH" in blob.upper() or "kernel_dispatch" in blob:
        return True
    return any(k in rec for k in NAME_KEYS[:-1]) or any(k in rec for k in NESTED_NAME_KEYS)


def _record_name(rec):
    for k in NAME_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v and "KERNEL_DISPATCH" not in v.upper():
            return v
    for k in NESTED_NAME_KEYS:
        v = rec.get(k)
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            for nk in NAME_KEYS:
                nv = v.get(nk)
                if isinstance(nv, str) and nv:
                    return nv
    return "<unnamed kernel>"


def _kernel_symbol_names(obj):
    names = {}
    for rec, _context in walk_json(obj):
        if "kernel_id" not in rec:
            continue
        name = (rec.get("formatted_kernel_name") or rec.get("demangled_kernel_name") or
                rec.get("kernel_name") or rec.get("truncated_kernel_name"))
        if isinstance(name, str) and name:
            try:
                names[int(rec["kernel_id"])] = name
            except (TypeError, ValueError):
                pass
    return names


def find_kernels(rocprof_path, pattern, time_unit):
    regex = re.compile(pattern)
    found = []
    files = json_files(rocprof_path)
    if not files:
        raise SystemExit(f"{rocprof_path}: 没找到 JSON 文件")
    for p in files:
        obj = load_json(p)
        symbol_names = _kernel_symbol_names(obj)
        for rec, context in walk_json(obj):
            pair = _time_pair(rec, time_unit)
            if not pair or not _is_kernel_dispatch(rec, context):
                continue
            dispatch_info = rec.get("dispatch_info", {}) if isinstance(rec.get("dispatch_info"), dict) else {}
            kernel_id = dispatch_info.get("kernel_id", rec.get("kernel_id"))
            try:
                kernel_id = int(kernel_id) if kernel_id is not None else None
            except (TypeError, ValueError):
                kernel_id = None
            name = symbol_names.get(kernel_id) if kernel_id is not None else None
            name = name or _record_name(rec)
            if not regex.search(name):
                continue
            grid = dispatch_info.get("grid_size", {}) if isinstance(dispatch_info.get("grid_size"), dict) else {}
            workgroup = dispatch_info.get("workgroup_size", {}) if isinstance(dispatch_info.get("workgroup_size"), dict) else {}
            link = None
            for lk in LINK_KEYS:
                v = rec.get(lk, dispatch_info.get(lk) if isinstance(dispatch_info, dict) else None)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    link = int(v)
                    break
                if isinstance(v, dict):
                    for kk in ("dispatch_id", "id", "handle", "value"):
                        if isinstance(v.get(kk), (int, float)) and not isinstance(v.get(kk), bool):
                            link = int(v[kk])
                            break
                if link is not None:
                    break
            found.append(dict(start=pair[0], end=pair[1], name=name, kernel_id=kernel_id, link=link,
                              grid=grid, workgroup=workgroup, source=str(p)))
    unique = {}
    for k in found:
        unique[(k["start"], k["end"], k["name"])] = k
    return sorted(unique.values(), key=lambda k: (k["start"], k["end"], k["name"])), files


# ---------------- PMC 计数器 ----------------

def _counter_name_maps(obj):
    """从 JSON 里找计数器 id -> 名字 的定义表（metadata.counters / counter_info 等，递归宽容匹配）。"""
    id2name = {}
    for rec, context in walk_json(obj):
        ctx = " ".join(context).lower()
        if "counter" not in ctx and not any("counter" in str(k).lower() for k in rec):
            continue
        name = None
        for nk in CNAME_KEYS:
            v = rec.get(nk)
            if isinstance(v, str) and v:
                name = v
                break
        if not name:
            continue
        cid = rec.get("counter_id", rec.get("id", rec.get("handle")))
        if isinstance(cid, dict):
            cid = cid.get("handle", cid.get("id"))
        if isinstance(cid, (int, float)) and not isinstance(cid, bool):
            id2name[int(cid)] = name
    return id2name


def _link_of(rec):
    """这条记录自己带的 dispatch/correlation id。rocprofv3 的真实结构里它在
    dispatch_data.dispatch_info.dispatch_id 下面，所以要往下钻两层。"""
    for lk in LINK_KEYS:
        v = rec.get(lk)
        if isinstance(v, dict):
            v = v.get("dispatch_id", v.get("id", v.get("handle")))
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    dd = rec.get("dispatch_data")
    if isinstance(dd, dict):
        di = dd.get("dispatch_info")
        if isinstance(di, dict):
            return _link_of(di)
        return _link_of(dd)
    return None


def _kid_of(rec):
    """kernel_id（用来和 kernel_dispatch 侧交叉核对同名 launch 的不同 dispatch）。"""
    for src in (rec, rec.get("dispatch_info") if isinstance(rec.get("dispatch_info"), dict) else {},
                rec.get("dispatch_data") if isinstance(rec.get("dispatch_data"), dict) else {}):
        if not isinstance(src, dict):
            continue
        v = src.get("kernel_id")
        if isinstance(v, dict):
            v = v.get("handle", v.get("id"))
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    di = rec.get("dispatch_data")
    if isinstance(di, dict) and isinstance(di.get("dispatch_info"), dict):
        return _kid_of(di["dispatch_info"])
    return None


def _walk_counters(obj, context=(), link=None, kid=None):
    """带“祖先继承”的遍历：叶子 counter 记录自己没有 dispatch_id，关联号要从祖先
    counter_collection[] 元素上继承下来（fi_bugs_amd-sk-ds B28：旧实现只看叶子键，
    T2 的 104 条 counter 记录一条也关联不上，summary 落下 merged: 0 却仍列出 4 个计数器名，
    trace 只有一个空的 PMC 进程）。"""
    if isinstance(obj, dict):
        l = _link_of(obj)
        k = _kid_of(obj)
        nl = l if l is not None else link
        nk = k if k is not None else kid
        yield obj, context, nl, nk
        for key, v in obj.items():
            yield from _walk_counters(v, context + (str(key),), nl, nk)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_counters(v, context + (str(i),), link, kid)


def _extract_counters_json(path):
    """返回 [(link_dispatch_id 或 None, counter_name, value, source, kernel_id)]。"""
    out = []
    obj = load_json(path)
    id2name = _counter_name_maps(obj)
    for rec, context, link_inherited, kid in _walk_counters(obj):
        ctx = " ".join(context).lower()
        if "counter" not in ctx:
            continue
        val = None
        for vk in VALUE_KEYS:
            v = rec.get(vk)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                val = float(v)
                break
        if val is None:
            continue
        name = None
        for nk in CNAME_KEYS:
            v = rec.get(nk)
            if isinstance(v, str) and v and "collection" not in v.lower():
                name = v
                break
        cid = rec.get("counter_id", rec.get("id", rec.get("handle")))
        if isinstance(cid, dict):
            cid = cid.get("handle", cid.get("id"))
        if name is None and isinstance(cid, (int, float)) and not isinstance(cid, bool):
            name = id2name.get(int(cid))
        if name is None:
            name = f"counter_{cid}" if cid is not None else "counter_unknown"
        # 自己带的关联号优先，没有就用祖先继承下来的
        link = _link_of(rec)
        if link is None:
            link = link_inherited
        out.append((link, name, val, str(path), kid))
    return out


def _extract_counters_csv(path):
    """rocprofv3 counter_collection CSV：列名宽容匹配（Counter_Name/Name、Counter_Value/Value、Dispatch_Id/Correlation_Id）。
    返回的元组形状与 JSON 侧一致（kernel_id 在 CSV 里没有，记 None）。"""
    out = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return out
    hdr = [h.strip().lower() for h in rows[0]]

    def col(*cands):
        for c in cands:
            if c in hdr:
                return hdr.index(c)
        return None
    cn = col("counter_name", "name", "metric", "counter")
    cv = col("counter_value", "value", "accumulation", "acc", "count")
    lk = col("dispatch_id", "correlation_id", "dispatch", "correlation")
    if cn is None or cv is None:
        return out
    for r in rows[1:]:
        try:
            val = float(r[cv])
        except (ValueError, IndexError):
            continue
        link = None
        if lk is not None and lk < len(r):
            try:
                link = int(float(r[lk]))
            except ValueError:
                link = None
        out.append((link, r[cn], val, str(path), None))
    return out


def collect_counters(pmc_path, aggregate="sum"):
    """从文件或目录收集计数器记录。返回 (records, files_checked, notes, stats)。

    同一个 (dispatch 关联号, 计数器名) 常常有多条记录：rocprofv3 一次 --pmc 采集会按趟/按实例
    分开写（盲测在 MI308X 上量到每个 bundle 52 条 counter 记录、4 个计数器名）。
    aggregate="sum" 把它们加总成该 dispatch 的总量（默认，和 rocprofv3 的 per-dispatch 语义一致），
    "first"/"last" 保留一条。合并了多少条如实写进 stats 与 summary，不静默丢数据。"""
    records, notes = [], []
    p = Path(pmc_path) if pmc_path else None
    if p is None or not p.exists():
        return [], [], [f"计数器路径不存在：{pmc_path}"], {"records_seen": 0, "unique_link_name": 0,
                                                          "aggregate": aggregate, "links": [], "kernel_ids": [],
                                                          "max_records_per_counter": 0, "unlinked_records": 0}
    files = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())
    for f in files:
        try:
            if f.suffix == ".csv":
                records.extend(_extract_counters_csv(f))
            elif f.suffix == ".json" or f.name.endswith(".json.gz"):
                records.extend(_extract_counters_json(f))
        except Exception as e:  # 单个文件解析失败不拖垮整体，如实记录
            notes.append(f"{f.name}: 解析失败 {type(e).__name__}: {e}")
    # 按 (link, name) 合并（JSON 父子对象可能重复展开；真实 rocprofv3 一次 dispatch 有多条同名记录）
    agg: Dict[tuple, list] = {}
    order = []
    for link, name, val, src_, kid in records:
        key = (link, name)
        if key not in agg:
            agg[key] = []
            order.append(key)
        agg[key].append((val, src_, kid))
    uniq = []
    for key in order:
        rows = agg[key]
        link, name = key
        if aggregate == "sum":
            val = float(sum(r[0] for r in rows))
        elif aggregate == "last":
            val = rows[-1][0]
        else:
            val = rows[0][0]
        uniq.append((link, name, val, rows[0][1], rows[0][2], len(rows)))
    stats = {"records_seen": len(records), "unique_link_name": len(uniq), "aggregate": aggregate,
             "max_records_per_counter": max((r[-1] for r in uniq), default=0),
             "links": sorted({l for l, _n, _v, _s, _k, _c in uniq if l is not None}),
             "kernel_ids": sorted({k for _l, _n, _v, _s, k, _c in uniq if k is not None}),
             "unlinked_records": sum(1 for l, _n, _v, _s, _k, _c in uniq if l is None)}
    if stats["unlinked_records"]:
        notes.append(f"有 {stats['unlinked_records']} 组计数器记录没有 dispatch/correlation 关联号"
                     "（JSON 里既没有叶子键也没有祖先 dispatch_data.dispatch_info），无法归到某个 dispatch")
    return uniq, files, notes, stats


# ---------------- 主对齐流程 ----------------

def force_gz(path):
    if str(path).endswith(".gz"):
        return str(path)
    print(f"提示：交付格式固定为 gzip，输出改为 {path}.gz")
    return str(path) + ".gz"


def align_dump(dump_path, rocprof_path, names_path, out_path, summary_path,
               kernel_pattern, instance, max_tail_us, pad_us, time_unit, pmc_path=None, info_json=None,
               pmc_aggregate="sum"):
    d = read_dump(dump_path)
    opens = [t for rs in d["slots"].values() for t, tag, aux in rs if tag & OPEN]
    if not opens:
        raise SystemExit(f"{dump_path}: 没有 FI_OPEN 记录")
    g_first, g_last = min(opens), max(rs[-1][0] for rs in d["slots"].values())

    with open(names_path, "r", encoding="utf-8") as f:
        names = json.load(f)
    trace, summary = decode(d, names)
    print_summary(summary)

    kernels, files = find_kernels(rocprof_path, kernel_pattern, time_unit)
    if not kernels:
        raise SystemExit(f"rocprof JSON 中没有匹配 {kernel_pattern!r} 的 KERNEL_DISPATCH；已读取 {len(files)} 个 JSON 文件")
    if not -len(kernels) <= instance < len(kernels):
        raise SystemExit(f"--instance={instance} 超出范围（匹配 {len(kernels)} 个 kernel）")
    k = kernels[instance]
    offset = k["start"] - g_first
    tail_ns = k["end"] - (g_last + offset)
    align = dict(kernel=k["name"][:160], instance=instance, matched_instances=len(kernels),
                 kernel_us=round((k["end"] - k["start"]) / 1000, 3),
                 sw_span_us=round((g_last - g_first) / 1000, 3),
                 offset_ns=offset, end_margin_us=round(tail_ns / 1000, 3),
                 source=k["source"], ok=0 <= tail_ns <= max_tail_us * 1000)

    # decode() 的 ts 相对 dump 最早一条记录（t0）。软件轨与 rocprofv3 kernel 条各自以“自己的起点”为 0：
    # 软件轨对齐到第一条 FI_OPEN（g_first），kernel 条对齐到 k["start"]。器件计时器（AMD s_memrealtime）
    # 与 rocprofv3 的主机时间戳不是同一个时钟域，两者的绝对差值只写进 align.offset_ns 供审计，
    # 不用来平移软件轨 —— 旧写法 shift=(t0-k.start) 把软件事件推到 10^8 us 之外，
    # 与同文件的 kernel 条（0~1826 us）完全不在一条轴上（fi_bugs_amd-sk-ds B19 实测）。
    t0 = min(t for rs in d["slots"].values() for t, tag, aux in rs)
    shift_us = (t0 - g_first) / 1000.0
    events = trace["traceEvents"]
    for e in events:
        if "ts" in e:
            e["ts"] = round(e["ts"] + shift_us, 3)

    # 轴向自检：软件事件必须落在 kernel 条的时间窗内（容差 = --max-tail-us）。
    # 只看结构的验收器（区间嵌套/泳道数/名字）发现不了整体错轴（fi_bugs_amd-sk-ds B20：
    # 错轴 trace 仍被判 PASS），所以这里必须自己响亮地失败。
    k_us = (k["end"] - k["start"]) / 1000.0
    sw_begins = [e["ts"] for e in events if "ts" in e]
    sw_ends = [e["ts"] + float(e.get("dur") or 0.0) for e in events if "ts" in e]
    sw_min, sw_max = min(sw_begins), max(sw_ends)
    if sw_min < -max_tail_us or sw_max > k_us + max_tail_us:
        raise SystemExit(
            f"时间轴自检失败：软件事件在 {sw_min:.3f}~{sw_max:.3f} us，rocprofv3 kernel 条在 0~{k_us:.3f} us，"
            f"两者不重叠（容差 ±{max_tail_us} us）。\n"
            "常见原因：(1) --instance/--kernel 选到了另一次 launch；(2) 转储 meta.timer_hz 与器件实际计时频率不符，"
            "软件轴被整体拉伸或压缩（AMD 上用 rocprofv3 的 kernel 时长交叉标定 timer_hz）；"
            "(3) 转储与 rocprofv3 输出不是同一次采集。\n"
            f"当前 align：kernel_us={k_us:.3f}，sw_span_us={align['sw_span_us']}，offset_ns={offset}。")
    align["axis"] = "软件轨 ts 相对第一条 FI_OPEN；rocprofv3 kernel 条相对该 kernel 实例开始时刻"
    align["software_min_us"], align["software_max_us"] = round(sw_min, 3), round(sw_max, 3)
    align["axis_check"] = {"tolerance_us": max_tail_us, "kernel_us": round(k_us, 3), "ok": True}
    if abs(offset) > 1_000_000_000:
        align["clock_domain_note"] = (
            f"器件计时器与 rocprofv3 主机时间戳不是同一时钟域（offset_ns={offset}，约 {offset / 1e9:.1f} s），"
            "offset_ns 因此不代表 launch 延迟，只作审计记录；对齐质量看 end_margin_us 与 axis_check。")

    # rocprofv3 kernel 条
    p_roc = 1 << 20
    events.append(dict(ph="M", pid=p_roc, name="process_name", args=dict(name="rocprofv3：KERNEL_DISPATCH")))
    events.append(dict(ph="M", pid=p_roc, name="process_sort_index", args=dict(sort_index=-2)))
    lo, hi = k["start"] - pad_us * 1000, k["end"] + pad_us * 1000
    for rk in kernels:
        if rk["end"] < lo or rk["start"] > hi:
            continue
        events.append(dict(ph="X", pid=p_roc, tid=0,
                           ts=round((rk["start"] - k["start"]) / 1000, 3),
                           dur=round((rk["end"] - rk["start"]) / 1000, 3),
                           name=rk["name"].split("(")[0][-120:], cat="rocprofv3_kernel",
                           args=dict(source=rk["source"], grid=rk["grid"], workgroup=rk["workgroup"])))

    # PMC 计数器：按 dispatch 关联；对到被对齐 kernel 的那一份写进 kernel 条 args 与 summary。
    pmc_info = dict(source=pmc_path, merged=0, counters={}, notes=[])
    pmc_recs, pmc_files, pmc_notes, pmc_stats = collect_counters(pmc_path or rocprof_path, pmc_aggregate)
    pmc_info["files_checked"] = len(pmc_files)
    pmc_info["notes"] = pmc_notes
    pmc_info["stats"] = pmc_stats
    if pmc_recs:
        # 与被对齐 kernel 关联：优先 link==k.link；其次只有一条 kernel 时全归它
        mine = [(n, v, cnt) for link, n, v, _s, _kid, cnt in pmc_recs
                if (k["link"] is not None and link == k["link"]) or (len(kernels) == 1 and link is None)]
        for n, v, _cnt in mine:
            pmc_info["counters"][n] = v
        pmc_info["merged"] = len(mine)
        pmc_info["records_per_counter"] = {n: cnt for n, _v, cnt in mine}
        pmc_info["names"] = sorted({n for _l, n, _v, _s, _k, _c in pmc_recs})
        pmc_info["per_dispatch_links"] = sorted({l for l, _n, _v, _s, _k, _c in pmc_recs if l is not None})
        pmc_info["kernel_ids"] = pmc_stats["kernel_ids"]
        # 计数器轨道：每个有关联的 dispatch 一个瞬时点，args 带全部计数器值
        p_pmc = p_roc + 1
        events.append(dict(ph="M", pid=p_pmc, name="process_name",
                           args=dict(name="硬件：PMC 计数器（每 dispatch 聚合，非时间序列采样）")))
        events.append(dict(ph="M", pid=p_pmc, name="process_sort_index", args=dict(sort_index=-1)))
        by_link = defaultdict(dict)
        for link, n, v, _s, _kid, _cnt in pmc_recs:
            if link is not None:
                by_link[link][n] = v
        n_marks = 0
        for rk in kernels:
            if rk["link"] is None or rk["link"] not in by_link:
                continue
            if rk["end"] < lo or rk["start"] > hi:
                continue
            events.append(dict(ph="i", s="t", pid=p_pmc, tid=0,
                               ts=round((rk["start"] - k["start"]) / 1000, 3),
                               name="PMC " + rk["name"].split("(")[0][-60:],
                               args=dict(by_link[rk["link"]])))
            n_marks += 1
        pmc_info["marks_emitted"] = n_marks
        if not n_marks and by_link:
            pmc_info["notes"].append("计数器记录的 dispatch 关联号与 kernel 条对不上，未画 PMC 轨道点")
        if not pmc_info["merged"] and pmc_info["names"]:
            # 列出了计数器名却一条也没合并：这就是 B28 里“只看 names 会误以为已合并”的形态，必须写明
            pmc_info["notes"].append(
                f"识别到 {len(pmc_info['names'])} 个计数器名但 merged=0：这些记录关联到的 dispatch 是 "
                f"{pmc_info['per_dispatch_links']}，被对齐的 kernel 关联号是 {k['link']}。"
                "同名 kernel 有多次 launch 时用 --instance 选对那一次，或核对 kernel_dispatch 与 "
                "counter_collection 两侧的 dispatch_id/kernel_id")
        if pmc_info["merged"]:
            for e in events:
                if e.get("cat") == "rocprofv3_kernel" and abs(e.get("ts", 1e18)) < 1e-9 and e.get("dur") == round((k["end"] - k["start"]) / 1000, 3):
                    e["args"]["pmc"] = pmc_info["counters"]
                    break
    else:
        pmc_info["notes"].append(
            "未从该 rocprofv3 输出里识别到计数器记录。rocprofv3 的 --pmc 输出 per-dispatch 聚合计数器"
            "（counter_collection），没有整卡周期采样曲线（不同于 nsys GPU metrics）；如需计数器请用"
            " rocprofv3 --pmc <计数器> -f json 重采，并把输出目录传给 --pmc。")

    # 软硬件信息轨道
    extra = None
    if info_json:
        with open(info_json, encoding="utf-8") as f:
            extra = json.load(f)
    gpu_info = {}
    for p in json_files(rocprof_path):
        try:
            obj = load_json(p)
        except Exception:
            continue
        for rec, context in walk_json(obj):
            ctx = " ".join(context).lower()
            if "agent" in ctx and isinstance(rec, dict):
                name = rec.get("name") or rec.get("marketing_name")
                if isinstance(name, str) and ("gfx" in name.lower() or "mi3" in name.lower() or "instinct" in name.lower()):
                    gpu_info.update({k2: v2 for k2, v2 in rec.items() if isinstance(v2, (str, int, float))})
    info = dict(tool="fast-instrument fi_align_rocprof",
                gpu=gpu_info or None,
                launch=dict(kernel=k["name"][:300], grid=k["grid"], workgroup=k["workgroup"]),
                instrument=dict(meta=d["meta"], timer=summary.get("timer"),
                                timer_min_step_ns=summary.get("timer_min_step_ns")),
                pmc=pmc_info, user=extra)
    p_info = p_roc + 2
    events.append(dict(ph="M", pid=p_info, name="process_name", args=dict(name="软硬件信息（点开看 args）")))
    events.append(dict(ph="M", pid=p_info, name="process_sort_index", args=dict(sort_index=-3)))
    gsz = k["grid"] or {}
    wgsz = k["workgroup"] or {}
    title = (f"{gpu_info.get('name', 'GPU')} · grid {gsz.get('x', '?')}x{gsz.get('y', '?')}x{gsz.get('z', '?')}"
             f" · wg {wgsz.get('x', '?')} · PMC {len(pmc_info['counters'])} 项")
    events.append(dict(ph="X", pid=p_info, tid=0, ts=0.0, dur=round((k["end"] - k["start"]) / 1000, 3),
                       name=title, cat="info", args=json.loads(json.dumps(info, default=str))))

    trace["otherData"]["tool"] = "fast-instrument"
    trace["otherData"]["align"] = align
    trace["otherData"]["pmc"] = pmc_info
    trace["otherData"]["hw_sw_info"] = info
    trace["otherData"]["time_zero"] = ("rocprofv3 kernel 条：该 kernel 实例开始时刻；"
                                       "软件轨：第一条 FI_OPEN（器件计时器与主机时间戳不同时钟域，"
                                       "绝对差见 otherData.align.offset_ns）")
    summary["align"] = align
    summary["pmc"] = pmc_info
    write_json(trace, str(out_path))
    write_json(summary, str(summary_path))
    print("对齐：", json.dumps(align, ensure_ascii=False))
    print("PMC：", json.dumps({k2: pmc_info.get(k2) for k2 in ("merged", "counters", "notes")}, ensure_ascii=False, default=str))
    print("写出", out_path)
    return trace, summary, align


def _write_dump(path, slots, n_slots, cap, wpc, timer, meta):
    cnt = [len(slots.get(s, [])) for s in range(n_slots)]
    blob = b"FILITE02" + struct.pack("<6I", n_slots, cap, wpc, 0, timer, len(meta)) + meta
    blob += struct.pack(f"<{n_slots}I", *cnt)
    for s in range(n_slots):
        for r in slots.get(s, []):
            blob += struct.pack("<QII", *r)
    Path(path).write_bytes(blob)


def selftest():
    here = Path(".").resolve()
    with tempfile.TemporaryDirectory(prefix=".fi_align_rocprof_test_", dir=str(here)) as td:
        td = Path(td)
        # timer_hz=1e9：1 tick = 1 ns
        slots = {0: [(1000, OPEN | 0, 0), (2000, 1, 0), (4000, 0x80000000 | 1, 0), (5000, 0x08000000, 0)]}
        meta = json.dumps({"timer_hz": 1000000000, "gran": 0, "arch": "gfx942", "kernel": "demo"}).encode()
        _write_dump(td / "dump.bin", slots, 1, 8, 1, 0, meta)
        (td / "names.json").write_text(json.dumps(
            {"events": {"1": "计算"}, "roles": {"0": "链"}}, ensure_ascii=False), encoding="utf-8")
        (td / "rocprof.json").write_text(json.dumps({
            "rocprofiler-sdk-tool": [{
                "metadata": {"kernel_symbols": [
                    {"kernel_id": 11, "formatted_kernel_name": "demo(float*, unsigned int*, int, int, int)"}],
                    "counters": [{"counter_id": 7, "counter_name": "SQ_WAVES"},
                                 {"counter_id": 9, "counter_name": "GRBM_COUNT"}],
                    "agents": [{"id": 1, "name": "AMD Instinct MI308X", "gfx_target_version": "gfx942"}]},
                "buffer_records": {"kernel_dispatch": [{
                    "start_timestamp": 900, "end_timestamp": 6000,
                    "dispatch_info": {"kernel_id": 11, "dispatch_id": 42,
                                      "grid_size": {"x": 10240, "y": 1, "z": 1},
                                      "workgroup_size": {"x": 128, "y": 1, "z": 1}}}],
                    "counter_collection": [
                        {"dispatch_id": 42, "counter_id": 7, "value": 12345.0},
                        {"dispatch_id": 42, "counter_id": 9, "value": 678.0}]}
            }]}, ensure_ascii=False), encoding="utf-8")
        trace, summary, align = align_dump(
            td / "dump.bin", td / "rocprof.json", td / "names.json", td / "merged.json.gz",
            td / "summary.json", "demo", -1, 50, 50, "ns")
        assert align["offset_ns"] == -100, align
        assert align["end_margin_us"] == 1.1, align
        assert align["ok"] is True, align
        assert not summary["errors"], summary
        x = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
        assert any(e.get("cat") == "rocprofv3_kernel" for e in x), x
        assert any(e.get("name") == "计算" for e in x), x
        pmc = summary["pmc"]
        assert pmc["merged"] == 2 and pmc["counters"] == {"SQ_WAVES": 12345.0, "GRBM_COUNT": 678.0}, pmc
        assert pmc["marks_emitted"] == 1, pmc
        merged = json.loads(gzip.open(td / "merged.json.gz", "rt", encoding="utf-8").read())
        assert merged["traceEvents"] and merged["otherData"]["tool"] == "fast-instrument"
        kbar = [e for e in merged["traceEvents"] if e.get("cat") == "rocprofv3_kernel"][0]
        assert kbar["args"].get("pmc", {}).get("SQ_WAVES") == 12345.0, kbar["args"]
        # 轴向自检：软件跨度远大于 kernel 条时必须响亮失败，不能静默出一张错轴的图
        bad_slots = {0: [(1000, OPEN | 0, 0), (2000, 1, 0), (10_000_000_000, 0x80000000 | 1, 0),
                         (10_000_001_000, 0x08000000, 0)]}
        _write_dump(td / "dump_bad.bin", bad_slots, 1, 8, 1, 0, meta)
        try:
            align_dump(td / "dump_bad.bin", td / "rocprof.json", td / "names.json", td / "bad.json.gz",
                       td / "bad.summary.json", "demo", -1, 50, 50, "ns")
            raise AssertionError("错轴没有被拦下来")
        except SystemExit as e:
            assert "时间轴自检失败" in str(e), str(e)
        assert align["axis_check"]["ok"] is True and align["software_min_us"] >= 0.0, align
        assert "clock_domain_note" not in align, align   # 自检里两轴同域，不该出现跨时钟域提示

        # CSV 计数器路径也走通
        (td / "cc.csv").write_text("Dispatch_Id,Counter_Name,Counter_Value\n42,SQ_WAVES,12345\n42,GRBM_COUNT,678\n",
                                   encoding="utf-8")
        recs, files, notes, stats = collect_counters(td / "cc.csv")
        assert {(l, n) for l, n, _v, _s, _k, _c in recs} == {(42, "SQ_WAVES"), (42, "GRBM_COUNT")}, recs
        assert stats["records_seen"] == 2 and stats["aggregate"] == "sum", stats

        # 真实 rocprofv3 的 --pmc JSON：关联号在祖先 counter_collection[].dispatch_data.dispatch_info 上，
        # 叶子 counter 记录自己没有 dispatch_id；同一 (dispatch, 计数器) 有多条记录要加总。
        # 形状照 fi_bugs_amd-sk-ds B28 的原文（T2 两档同名 launch：baseline dispatch 3、插桩 dispatch 12）。
        def bundle(disp, kid, waves, valu):
            return {"dispatch_data": {"dispatch_info": {"dispatch_id": disp, "kernel_id": kid}},
                    "counter": [{"counter_id": 7, "value": waves[0]}, {"counter_id": 11, "value": valu[0]},
                                {"counter_id": 7, "value": waves[1]}, {"counter_id": 11, "value": valu[1]}]}
        (td / "pmc_nested.json").write_text(json.dumps({"rocprofiler-sdk-tool": [{
            "metadata": {"counters": [{"counter_id": 7, "counter_name": "SQ_WAVES"},
                                      {"counter_id": 11, "counter_name": "SQ_INSTS_VALU"}]},
            "buffer_records": {"counter_collection": [bundle(3, 27, (320.0, 320.0), (1.0, 2.0)),
                                                      bundle(12, 196, (320.0, 320.0), (3.0, 4.0))]}}]},
            ensure_ascii=False), encoding="utf-8")
        nrecs, _f2, nnotes, nstats = collect_counters(td / "pmc_nested.json")
        got = {(l, n): v for l, n, v, _s, _k, _c in nrecs}
        assert got == {(3, "SQ_WAVES"): 640.0, (3, "SQ_INSTS_VALU"): 3.0,
                       (12, "SQ_WAVES"): 640.0, (12, "SQ_INSTS_VALU"): 7.0}, got
        assert nstats["links"] == [3, 12] and nstats["kernel_ids"] == [27, 196], nstats
        assert nstats["max_records_per_counter"] == 2 and nstats["unlinked_records"] == 0, nstats
        assert not nnotes, nnotes
        # first/last 两种口径也要能用（写明在 summary 里）
        f_recs, _f3, _n3, _s3 = collect_counters(td / "pmc_nested.json", aggregate="first")
        assert {(l, n): v for l, n, v, _s, _k, _c in f_recs}[(12, "SQ_INSTS_VALU")] == 3.0, f_recs
        # 只有关联号、没有计数器名的记录也不能崩
        (td / "pmc_bare.json").write_text(json.dumps({"buffer_records": {"counter_collection": [
            {"dispatch_data": {"dispatch_info": {"dispatch_id": 5}}, "counter": [{"value": 1.0}]}]}}),
            encoding="utf-8")
        brecs, _f4, _n4, _s4 = collect_counters(td / "pmc_bare.json")
        assert brecs and brecs[0][0] == 5, brecs

        # rocprofv3 7.2 的真实布局（fi_bugs_amd-sk-qwen B22）：定义表在 metadata.counters[]（id={"handle":N}，
        # 每个名字 × 每个 agent 各一条），记录在 callback_records.counter_collection[].records[]，
        # 叶子只有 counter_id 与 value，关联号在祖先 dispatch_data.dispatch_info.dispatch_id 上；
        # 同名计数器有 16 个 SQ instance / 4 个 GRBM instance，必须按 dispatch 加总。
        # 旧代码在这里连环错：关联号取不到 -> 按 (None, name) 去重只留第一条 -> 只有一个 kernel 匹配时
        # 那条"第一个 dispatch 的第一个 instance"被算到被对齐的 kernel 头上（SQ_WAVES 报 0）。
        H_WAVES, H_VALU, H_GRBM = 42949676448, 38654709152, 38654709160
        defs = [{"id": {"handle": h}, "name": n, "agent": ag}
                for ag in range(2)
                for h, n in ((H_WAVES, "SQ_WAVES"), (H_VALU, "SQ_INSTS_VALU"), (H_GRBM, "GRBM_COUNT"))]

        def cc72(disp, kid, waves, valu, grbm):
            recs = ([{"counter_id": {"handle": H_WAVES}, "value": v} for v in waves]
                    + [{"counter_id": {"handle": H_VALU}, "value": v} for v in valu]
                    + [{"counter_id": {"handle": H_GRBM}, "value": v} for v in grbm])
            return {"dispatch_data": {"dispatch_info": {"dispatch_id": disp, "kernel_id": kid}},
                    "records": recs}

        (td / "pmc_rocp72.json").write_text(json.dumps({"rocprofiler-sdk-tool": [{
            "metadata": {"counters": defs,
                         "kernel_symbols": [{"kernel_id": 196,
                                             "formatted_kernel_name": "_Z13ring_pipelinePKfPfS1_Pjii"}]},
            "callback_records": {"counter_collection": [
                cc72(1, 27, [0.0] * 16, [0.0] * 16, [4191.0] * 4),          # rocclr 的填零小 kernel
                cc72(9, 196, [20.0] * 16, [1464883.0] * 16, [3657048.0] * 4),  # 被对齐那次 dispatch
            ]}}]}, ensure_ascii=False), encoding="utf-8")
        r72, _f72, n72, s72 = collect_counters(td / "pmc_rocp72.json")
        g72 = {(l, n): v for l, n, v, _s, _k, _c in r72}
        assert g72[(9, "SQ_WAVES")] == 320.0, g72                 # 16 instance × 20 = 320（=80 CTA × 4 wave）
        assert g72[(9, "SQ_INSTS_VALU")] == 1464883.0 * 16, g72
        assert g72[(9, "GRBM_COUNT")] == 3657048.0 * 4, g72
        assert g72[(1, "SQ_WAVES")] == 0.0 and g72[(1, "GRBM_COUNT")] == 4191.0 * 4, g72
        assert s72["links"] == [1, 9] and s72["kernel_ids"] == [27, 196], s72
        assert s72["unlinked_records"] == 0 and s72["max_records_per_counter"] == 16, s72
        assert not n72, n72

        # 真实 rocprofv3 的墙钟纪元时间戳（1.79e18 ns）：平移方向 + 整数算术的回归。
        # 盲测 fi_bugs_amd-sk-qwen B21 的原文形态：旧写法 shift=(t0-k["start"]) 把软件事件推到
        # ts≈-1.79e15 us，kernel 条却在 0 us，独立验收器直接 FAIL；同时 float 读时间戳会把 kernel
        # 时长量化到 256 ns（算出 445.696 而真值是 445.760）。旧夹具用 900/6000 ns 恰好绕过这两条。
        EPOCH = 1794165000000000000
        (td / "rocprof_epoch.json").write_text(json.dumps({"rocprofiler-sdk-tool": [{
            "metadata": {"kernel_symbols": [{"kernel_id": 11, "formatted_kernel_name": "ring_pipeline"}],
                         "agents": [{"id": 1, "name": "AMD Instinct MI308X", "gfx_target_version": "gfx942"}]},
            "buffer_records": {"kernel_dispatch": [{
                "start_timestamp": EPOCH, "end_timestamp": EPOCH + 445760,
                "dispatch_info": {"kernel_id": 11, "dispatch_id": 9,
                                  "grid_size": {"x": 2560, "y": 1, "z": 1},
                                  "workgroup_size": {"x": 256, "y": 1, "z": 1}}}]}}]},
            ensure_ascii=False), encoding="utf-8")
        ep_slots = {0: [(1000, OPEN | 0, 0), (2000, 1, 0), (440000, 0x80000000 | 1, 0), (442760, 0x08000000, 0)]}
        _write_dump(td / "dump_epoch.bin", ep_slots, 1, 8, 1, 0, meta)
        _tre, _sme, ep = align_dump(td / "dump_epoch.bin", td / "rocprof_epoch.json", td / "names.json",
                                    td / "epoch.json.gz", td / "epoch.summary.json", "ring_pipeline", -1,
                                    50, 50, "ns")
        assert ep["kernel_us"] == 445.76, ep           # 整数算术：走 float 会量化成 445.696
        assert ep["sw_span_us"] == 441.76 and ep["end_margin_us"] == 4.0, ep
        assert ep["ok"] is True and ep["axis_check"]["ok"] is True, ep
        assert ep["offset_ns"] == EPOCH - 1000 and "clock_domain_note" in ep, ep
        assert 0.0 <= ep["software_min_us"] and ep["software_max_us"] <= 445.76 + 50, ep
        ep_ev = json.loads(gzip.open(td / "epoch.json.gz", "rt", encoding="utf-8").read())["traceEvents"]
        ep_span = max(abs(e["ts"]) for e in ep_ev if "ts" in e)
        assert ep_span < 1e4, f"软件轨被推到 {ep_span} us，错轴没有修好"
        print("SELFTEST PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="?")
    ap.add_argument("--names")
    ap.add_argument("--rocprof", help="rocprofv3 输出目录或 JSON 文件")
    ap.add_argument("--kernel", default=".*", help="kernel 名正则，默认匹配所有")
    ap.add_argument("--instance", type=int, default=-1, help="匹配到的第几个实例，-1 表示最后一个")
    ap.add_argument("--pmc", help="计数器 JSON/CSV 或目录；默认同 --rocprof（自动识别 counter_collection）")
    ap.add_argument("--pmc-aggregate", choices=("sum", "first", "last"), default="sum",
                    help="同一 (dispatch, 计数器名) 有多条记录时怎么合并：sum=加总（默认，rocprofv3 的 "
                         "per-dispatch 语义），first/last=只保留一条；合并条数写进 summary 的 records_per_counter")
    ap.add_argument("--info-json", help="调用方补充的软件/环境信息（rocm-smi 时钟、编译选项…），原样写进 trace")
    ap.add_argument("--time-unit", choices=("auto", "ns", "us", "ms", "sec"), default="auto")
    ap.add_argument("--pad-us", type=float, default=50.0)
    ap.add_argument("--max-tail-us", type=float, default=50.0)
    ap.add_argument("-o", "--out")
    ap.add_argument("--summary")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.dump and a.names and a.rocprof and a.out):
        ap.error("需要 dump、--names、--rocprof、-o")
    a.out = force_gz(a.out)
    summary = a.summary or (a.out[:-3].replace(".json", "") + ".summary.json" if a.out.endswith(".gz")
                            else a.out.replace(".json", "") + ".summary.json")
    _, sm, align = align_dump(a.dump, a.rocprof, a.names, a.out, summary, a.kernel,
                              a.instance, a.max_tail_us, a.pad_us, a.time_unit, a.pmc, a.info_json,
                              a.pmc_aggregate)
    return 0 if align["ok"] and not sm["errors"] else 2


if __name__ == "__main__":
    sys.exit(main())
