# -*- coding: utf-8 -*-
"""fast-instrument 归因器：把 FI 软件时间线和 nsys GPU metrics 归因成可审计的 JSON。

核心口径（每个 attrib.json 都写出来）：
    T_kernel = T_min + peak_loss + bubble + unaccounted
其中：
  T_min       = max(有用 FLOPs / 峰值算力, 必需字节 / 峰值带宽)，或调用方显式给的值。
  bubble      = [0,T_kernel] 内没有被任何 load/compute/store/memory 区间覆盖的墙钟时间。
  peak_loss   = max(0, active_span - T_min)；active 时仍有工作，但没达到 T_min 代表的硬件峰值。
  unaccounted = T_kernel - (T_min + peak_loss + bubble)，必须接近 0；若 T_min 高于 active_span，
                会保留有符号残差并标记 model_inconsistent，绝不把差额偷偷塞进某一类。

这是一套“可解释账本”，不是 ncu 式逐 SM/逐 warp 硬件归因。GPU metrics 是全卡平均且通常
4~5 us 采样；软件事件能定位角色/CTA，但角色等待的总和跨多个 warp 并发，不能直接当作墙钟。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
try:
    from weaveprof.pipeline.core.semantics import load_kb as load_semantic_kb
    _SEMANTIC_ROWS = {row["semantic_id"]: row for row in load_semantic_kb()["semantics"]}
except Exception as _semantic_error:  # configuration errors are surfaced by the delivery gate
    _SEMANTIC_ROWS = {}
    _SEMANTIC_ERROR = str(_semantic_error)
else:
    _SEMANTIC_ERROR = None

try:
    from fi_metric_map import (THRESHOLDS as METRIC_THRESHOLDS, amd_interface_document,
                               detect_arch, map_amd_counter_names, mapping_document,
                               match_metric_names, nvidia_arch_document)
except ImportError:  # 允许把 tools 目录加到 PYTHONPATH 后 import
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _HERE)
    from fi_metric_map import (THRESHOLDS as METRIC_THRESHOLDS, amd_interface_document,
                               detect_arch, map_amd_counter_names, mapping_document,
                               match_metric_names, nvidia_arch_document)

SCHEMA_VERSION = "fi-attrib"
TOOL_VERSION = "fi-attrib"
EPS = 1e-7

ACTIVE_KINDS = {"compute", "load", "store", "memory", "control", "other"}
WAIT_TYPES = ("wait_tail", "wait_dependency", "wait_load", "wait_sync", "wait_other")
BUBBLE_PRIORITY = ("wait_tail", "wait_dependency", "wait_load", "wait_sync", "wait_other")
WAIT_TYPE_ZH = {
    "wait_tail": "拖尾等待",
    "wait_dependency": "跨角色/依赖等待",
    "wait_load": "等装载/等数据到达",
    "wait_sync": "栅栏/同步等待",
    "wait_other": "其它等待/空转",
}
RESOURCE_ZH = {
    "compute": "计算", "memory": "内存", "dependency": "依赖", "tail": "拖尾",
    "idle": "空泡", "unknown": "未知", "issue": "指令发射",
}
PRIMARY_ZH = {
    "compute_throughput": "计算效率不足/峰值损失",
    "memory_bandwidth": "内存带宽",
    "dependency_wait": "跨角色依赖等待",
    "tail_imbalance": "CTA/波次拖尾",
    "sync_wait": "同步/栅栏",
    "bubble_idle": "无法归因到具名等待的空泡",
    "unknown": "证据不足，无法归因",
}


# --------------------------------------------------------------------------- 基础工具
def _now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _r(x: Optional[float], nd: int = 6) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if abs(v) < 0.5 * 10 ** (-nd):
        v = 0.0
    return round(v, nd)


def _f(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return default
    if isinstance(x, (int, float)):
        v = float(x)
    else:
        try:
            v = float(str(x).strip())
        except (TypeError, ValueError):
            return default
    return v if math.isfinite(v) else default


def _first_num(d: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for k in keys:
        if k in d:
            v = _f(d[k])
            if v is not None:
                return v
    return None


def load_json(path: str) -> Any:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump(_clean_json(obj), f, ensure_ascii=False, indent=2)
            f.write("\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_clean_json(obj), f, ensure_ascii=False, indent=2)
            f.write("\n")


def _clean_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean_json(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


def sha256_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def _median(v: Sequence[float]) -> Optional[float]:
    return float(statistics.median(v)) if v else None


def _percentile(v: Sequence[float], q: float) -> Optional[float]:
    if not v:
        return None
    a = sorted(float(x) for x in v)
    if len(a) == 1:
        return a[0]
    pos = (len(a) - 1) * max(0.0, min(1.0, q))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return a[lo]
    return a[lo] + (a[hi] - a[lo]) * (pos - lo)


def merge_intervals(intervals: Iterable[Tuple[float, float]], eps: float = EPS) -> List[Tuple[float, float]]:
    iv = sorted((float(a), float(b)) for a, b in intervals if b > a)
    out: List[List[float]] = []
    for a, b in iv:
        if not out or a > out[-1][1] + eps:
            out.append([a, b])
        elif b > out[-1][1]:
            out[-1][1] = b
    return [(a, b) for a, b in out]


def interval_duration(intervals: Iterable[Tuple[float, float]]) -> float:
    return float(sum(b - a for a, b in merge_intervals(intervals)))


def complement_intervals(intervals: Iterable[Tuple[float, float]], start: float, end: float) -> List[Tuple[float, float]]:
    out, cur = [], float(start)
    for a, b in merge_intervals(intervals):
        a, b = max(float(start), a), min(float(end), b)
        if b <= a:
            continue
        if a > cur + EPS:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < end - EPS:
        out.append((cur, float(end)))
    return out


def intersect_interval_sets(a: Iterable[Tuple[float, float]], b: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    aa, bb = merge_intervals(a), merge_intervals(b)
    out, i, j = [], 0, 0
    while i < len(aa) and j < len(bb):
        lo, hi = max(aa[i][0], bb[j][0]), min(aa[i][1], bb[j][1])
        if hi > lo:
            out.append((lo, hi))
        if aa[i][1] <= bb[j][1]:
            i += 1
        else:
            j += 1
    return merge_intervals(out)


def _point_covered(pt: float, merged: Sequence[Tuple[float, float]]) -> bool:
    # 小线性扫描；每个 bubble 元素段只调用几个 subtype。
    return any(a - EPS <= pt <= b + EPS for a, b in merged)


# --------------------------------------------------------------------------- 事件读取与语义
def extract_software_events(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """只取旧版软件事件。nsys kernel 条/信息条位于大 pid（>= 1<<20），明确排除。"""
    out = []
    for e in trace.get("traceEvents", []):
        if e.get("ph") != "X":
            continue
        try:
            pid = int(e.get("pid", 0))
        except (TypeError, ValueError):
            pid = 0
        cat = str(e.get("cat", ""))
        if pid >= (1 << 20) or cat in ("info", "nsys_kernel"):
            continue
        ts, dur = _f(e.get("ts")), _f(e.get("dur"))
        if ts is None or dur is None or dur < 0:
            continue
        args = e.get("args") if isinstance(e.get("args"), dict) else {}
        out.append({
            "name": str(e.get("name", "")),
            "start": ts,
            "end": ts + dur,
            "dur": dur,
            "role": str(args.get("role", "未标角色")),
            "cta": args.get("cta"),
            "warp": args.get("warp"),
            "sm": args.get("sm"),
            "aux": args.get("aux"),
            "cat": cat,
            "semantic_id": args.get("semantic_id"),
            "semantic_source": args.get("semantic_source"),
            "original_label": args.get("original_label", e.get("name", "")),
        })
    return out


def classify_event(name: str, role: str = "", semantic_id: Optional[str] = None) -> Tuple[str, str]:
    """返回 (kind, subtype)。优先使用 trace 内 canonical semantic_id；正则只兼容旧 trace。"""
    if semantic_id and semantic_id != "unknown" and semantic_id in _SEMANTIC_ROWS:
        row = _SEMANTIC_ROWS[semantic_id]
        attribution = str(row.get("attribution") or "other")
        if attribution == "wait":
            subtype = str(row.get("wait_subtype") or "wait_other")
            return subtype, "canonical semantic_id=" + semantic_id
        if attribution in ACTIVE_KINDS:
            return attribution, "canonical semantic_id=" + semantic_id
        return "other", "canonical semantic_id=" + semantic_id
    s = (name or "").lower()
    rs = (role or "").lower()
    is_wait = bool(re.search(r"(^|[\s_])wait([\s_]|$)|等待|等[^同]|自旋|spin|stall|空闲|空转|人为延迟", s))
    # “等”在某些中文名里表示“相等”，例如“等同于”；排除“等于/等同/等于”。
    if re.search(r"等于|等同|等值", s):
        is_wait = False
    if is_wait:
        if re.search(r"拖尾|tail|最后.*cta|收尾.*慢", s):
            return "wait_tail", "cta/角色收尾拖尾"
        if re.search(r"跨\s*sm|依赖|dependency|生产者|producer|就绪|ready|到达|arrival|等.*(?:数据|输出|kv|k/v|权重|stage|scale|q 装载)", s):
            return "wait_dependency", "等上游数据/跨角色依赖"
        if re.search(r"装载|载入|load|权重|tma|copy|预取|prefetch|读取|数据", s):
            return "wait_load", "等装载/等内存数据"
        if re.search(r"屏障|栅栏|barrier|fence|同步|sync|atomic|信号|semaphore|mbarrier", s):
            return "wait_sync", "栅栏/同步"
        return "wait_other", "其它等待/空转"
    # 活跃区间优先识别内存、写回、计算、装载，避免 “TMA 装载角色主阶段” 被算成 compute。
    if re.search(r"hbm|dram|显存|全局.*(?:读|写)|global\s+(?:read|write)|memory\s+(?:copy|stream)", s):
        return "memory", "DRAM/HBM 访存"
    if re.search(r"epilogue|写回|store|输出|归约|reduction", s):
        return "store", "写回/归约"
    if re.search(r"mma|matvec|matmul|gemm|fma|计算|compute|attention|softmax|\bpv\b|qk|norm|卷积|conv|fft", s):
        return "compute", "计算"
    if re.search(r"装载|载入|load|tma|copy|预取|prefetch|读取|生产者|producer", s):
        return "load", "装载/取数"
    if re.search(r"控制器|调度|launcher|launch|emit|取指|指令槽|重排|构造信号量", s):
        return "control", "控制/调度"
    if re.search(r"storer|写|store", rs) and re.search(r"区间|阶段|流水|计算", s):
        return "other", "其它有效工作"
    return "other", "其它有效工作"


def role_kind(role: str) -> str:
    s = (role or "").lower()
    if re.search(r"loader|装载|tma|生产者|producer", s):
        return "load"
    if re.search(r"storer|store|写回|epilogue", s):
        return "store"
    if re.search(r"controller|launcher|调度|控制|辅助", s):
        return "control"
    if re.search(r"consumer|compute|计算|mma|matvec|epilogue", s):
        return "compute"
    return "unknown"


# --------------------------------------------------------------------------- 账本
def _prepare_events(events: List[Dict[str, Any]], T: float) -> List[Dict[str, Any]]:
    recs = []
    for e in events:
        a, b = max(0.0, e["start"]), min(float(T), e["end"])
        if b <= a:
            continue
        kind, subtype = classify_event(e["name"], e["role"], e.get("semantic_id"))
        d = dict(e)
        d["start"], d["end"], d["dur"] = a, b, b - a
        d["kind"], d["subtype"] = kind, subtype
        recs.append(d)
    return recs


def _event_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_name[r["name"]].append(r)
    out = []
    for name, rs in by_name.items():
        durs = [r["dur"] for r in rs]
        kinds = Counter(r["kind"] for r in rs)
        out.append({
            "name": name,
            "count": len(rs),
            "sum_us": _r(sum(durs)),
            "covered_us": _r(interval_duration((r["start"], r["end"]) for r in rs)),
            "median_us": _r(_median(durs)),
            "p95_us": _r(_percentile(durs, 0.95)),
            "max_us": _r(max(durs) if durs else None),
            "roles": sorted({r["role"] for r in rs}),
            "kinds": dict(kinds),
        })
    return {"by_name": out}


# attrib.json 是给人审计的账本，不能把上万个实例名整列铺开：一次 13 事件 × 2048 tile 的运行
# 曾产出 6.1 MB JSON，其中约 3.7 MB 是 roles/role_breakdown 与 dependencies/dependency_breakdown
# 两对逐字节相同的重复字段（fi_bugs_b3-sk-qwen B11）。别名字段已去掉，实例名与明细行数设上限，
# 被截掉的部分一律给出合计与恢复命令，不静默丢数。
DEFAULT_MAX_NAMES = 32
DEFAULT_MAX_DEP_ROWS = 200
# 解码器 label_aux=true 时事件名形如 “等 cp.async 装载落地（wait_group） #1023”，尾部是 aux 实例号。
_INSTANCE_SUFFIX = re.compile(r"\s+#\S+$")


def _base_event_name(name: Any) -> str:
    """去掉 label_aux 加的 " #<aux>" 实例后缀，得到按事件号聚合用的基名。"""
    return _INSTANCE_SUFFIX.sub("", str(name))


def _cap_names(names: Iterable[Any], limit: int) -> Tuple[List[str], bool, int]:
    """返回 (最多 limit 个名字, 是否截断, 名字总数)。limit<=0 表示不截断。"""
    uniq = sorted({str(x) for x in names})
    if limit and limit > 0 and len(uniq) > limit:
        return uniq[:limit], True, len(uniq)
    return uniq, False, len(uniq)


def _cap_rows(rows: List[Dict[str, Any]], limit: int) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """dependencies 明细按 sum_wait_us 降序保留前 limit 行；返回 (保留行, 截断记账/None)。
    建议与主因判定用的是未截断的全量行，这里只影响写进 JSON 的明细。"""
    if not limit or limit <= 0 or len(rows) <= limit:
        return rows, None
    kept, omitted = rows[:limit], rows[limit:]
    return kept, {
        "max_dep_rows": limit,
        "dependency_rows_total": len(rows),
        "dependency_rows_kept": len(kept),
        "omitted_rows": len(omitted),
        "omitted_sum_wait_us": _r(sum(x["sum_wait_us"] or 0.0 for x in omitted)),
        "omitted_max_us": _r(max((x["max_us"] or 0.0) for x in omitted)),
        "note": ("dependencies 是逐 (事件名, aux 实例) 的明细，已按 sum_wait_us 降序保留前 "
                 f"{limit} 行，其余合计写在 omitted_sum_wait_us。要看全部明细加 --max-dep-rows 0；"
                 "要按事件号看主干用 dependency_summary.by_event_base。"),
    }


def _cta_and_role_stats(records: List[Dict[str, Any]],
                        max_names: int = DEFAULT_MAX_NAMES) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    cta_end: Dict[Any, float] = {}
    cta_start: Dict[Any, float] = {}
    cta_records: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("cta") is None:
            continue
        c = str(r["cta"])
        cta_start[c] = min(cta_start.get(c, r["start"]), r["start"])
        cta_end[c] = max(cta_end.get(c, r["end"]), r["end"])
        cta_records[c].append(r)
    ends = sorted(cta_end.values())
    cta_stats: Dict[str, Any] = {"n": len(ends)}
    if ends:
        med = _median(ends) or 0.0
        cta_stats.update({
            "end_min_us": _r(ends[0]), "end_median_us": _r(med), "end_max_us": _r(ends[-1]),
            "tail_us": _r(ends[-1] - med),
            "slowest_cta": max(cta_end, key=cta_end.get),
            "earliest_cta": min(cta_start, key=cta_start.get),
        })
        slow = max(cta_end, key=cta_end.get)
        slow_recs = cta_records[slow]
        cta_stats["slowest_cta_last_events"] = [
            {"name": r["name"], "role": r["role"], "start_us": _r(r["start"]),
             "dur_us": _r(r["dur"]), "kind": r["kind"]}
            for r in sorted(slow_recs, key=lambda x: x["end"])[-8:]
        ]
    role_to_cta_end: Dict[str, Dict[str, float]] = defaultdict(dict)
    role_recs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        role_recs[r["role"]].append(r)
        if r.get("cta") is not None:
            c = str(r["cta"])
            role_to_cta_end[r["role"]][c] = max(role_to_cta_end[r["role"]].get(c, 0.0), r["end"])
    role_rows = []
    for role, rs in role_recs.items():
        ends_r = sorted(role_to_cta_end.get(role, {}).values())
        waits = [r for r in rs if str(r["kind"]).startswith("wait_")]
        works = [r for r in rs if not str(r["kind"]).startswith("wait_")]
        med = _median(ends_r) if ends_r else None
        row = {
            "role": role,
            "kind": role_kind(role),
            "events": len(rs),
            "ctas": len(ends_r),
            "wait_total_us": _r(sum(r["dur"] for r in waits)),
            "work_total_us": _r(sum(r["dur"] for r in works)),
            "wait_covered_us": _r(interval_duration((r["start"], r["end"]) for r in waits)),
            "work_covered_us": _r(interval_duration((r["start"], r["end"]) for r in works)),
            "first_start_us": _r(min(r["start"] for r in rs)),
            "last_end_us": _r(max(r["end"] for r in rs)),
        }
        row["event_names"], row["event_names_truncated"], row["event_names_total"] = _cap_names(
            (r["name"] for r in rs), max_names)
        if ends_r:
            row.update({"end_min_us": _r(ends_r[0]), "end_median_us": _r(med), "end_max_us": _r(ends_r[-1]),
                        "tail_us": _r(ends_r[-1] - (med or 0.0))})
        role_rows.append(row)
    role_rows.sort(key=lambda x: (-(x.get("work_total_us") or 0.0), -(x.get("wait_total_us") or 0.0), x["role"]))
    return cta_stats, role_rows


def _dependency_stats(records: List[Dict[str, Any]], bubble_segments: Sequence[Tuple[float, float]],
                      max_names: int = DEFAULT_MAX_NAMES) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_base: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if str(r["kind"]).startswith("wait_"):
            by_event[r["name"]].append(r)
            by_type[r["kind"]].append(r)
            by_base[(_base_event_name(r["name"]), str(r["kind"]))].append(r)
    # 每个等待类型与空泡的交集，只作证据；不能把多个类型的交集相加。
    type_intersection = {t: interval_duration(intersect_interval_sets(((r["start"], r["end"]) for r in rs), bubble_segments))
                         for t, rs in by_type.items()}
    rows = []
    for name, rs in by_event.items():
        durs = [r["dur"] for r in rs]
        typ = rs[0]["kind"]
        rows.append({
            "event": name,
            "wait_type": typ,
            "wait_type_zh": WAIT_TYPE_ZH.get(typ, typ),
            "roles": sorted({r["role"] for r in rs}),
            "count": len(rs),
            "sum_wait_us": _r(sum(durs)),
            "covered_us": _r(interval_duration((r["start"], r["end"]) for r in rs)),
            "median_us": _r(_median(durs)),
            "p95_us": _r(_percentile(durs, 0.95)),
            "max_us": _r(max(durs) if durs else None),
            "bubble_intersection_us": _r(interval_duration(intersect_interval_sets(((r["start"], r["end"]) for r in rs), bubble_segments))),
        })
    rows.sort(key=lambda x: (-(x["sum_wait_us"] or 0.0), -(x["bubble_intersection_us"] or 0.0), x["event"]))
    # 账本主干：按 (事件基名, 等待类型) 聚合。label_aux 会把 2048 个 tile 拆成 2048 个不同名字，
    # 逐实例明细读不动也审不动，主干必须先按事件号聚合，实例数记在 instances。
    rollup = []
    for (base, typ), rs in by_base.items():
        durs = [r["dur"] for r in rs]
        names, names_cut, names_total = _cap_names((r["name"] for r in rs), 3)
        rollup.append({
            "event_base": base,
            "wait_type": typ,
            "wait_type_zh": WAIT_TYPE_ZH.get(typ, typ),
            "roles": sorted({r["role"] for r in rs}),
            "instances": names_total,
            "instances_truncated_in_examples": names_cut,
            "name_examples": names,
            "count": len(rs),
            "sum_wait_us": _r(sum(durs)),
            "covered_us": _r(interval_duration((r["start"], r["end"]) for r in rs)),
            "median_us": _r(_median(durs)),
            "p95_us": _r(_percentile(durs, 0.95)),
            "max_us": _r(max(durs) if durs else None),
            "bubble_intersection_us": _r(interval_duration(
                intersect_interval_sets(((r["start"], r["end"]) for r in rs), bubble_segments))),
        })
    rollup.sort(key=lambda x: (-(x["sum_wait_us"] or 0.0), x["event_base"]))

    def _by_type_entry(rs):
        names, cut, total = _cap_names((r["name"] for r in rs), max_names)
        return names, cut, total

    by_type_out = {}
    for t, rs in sorted(by_type.items()):
        names, cut, total = _by_type_entry(rs)
        by_type_out[t] = {
            "event_names": names,
            "event_names_truncated": cut,
            "event_names_total": total,
            "count": len(rs),
            "sum_wait_us": _r(sum(r["dur"] for r in rs)),
            "covered_us": _r(interval_duration((r["start"], r["end"]) for r in rs)),
            "bubble_intersection_us": _r(type_intersection.get(t, 0.0)),
        }
    summary = {
        "by_type": by_type_out,
        "by_event_base": rollup,
        "note": ("sum_wait_us 跨并发 warp/CTA 求和，不是墙钟；covered_us 是该角色集合的并集。"
                 "by_event_base 是按事件号聚合的主干（instances=该事件下不同 aux 实例名的个数），"
                 f"event_names 每处最多列 {max_names if max_names and max_names > 0 else '全部'} 个，"
                 "截断时 event_names_truncated=true 且给出 event_names_total；"
                 "roles/dependency_breakdown 这类逐字节重复的别名字段已删除。"),
    }
    return rows, summary


def _partition_bubble(bubble_segments: Sequence[Tuple[float, float]], records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """把每个空泡元素段分给一个等待类型（若有），保证 exclusive_us 可相加。"""
    type_union = {}
    for t in WAIT_TYPES:
        type_union[t] = merge_intervals((r["start"], r["end"]) for r in records if r["kind"] == t)
    accum = defaultdict(float)
    seg_counts = Counter()
    unlabeled = 0.0
    for ba, bb in bubble_segments:
        cuts = {ba, bb}
        for merged in type_union.values():
            for a, b in merged:
                if ba < a < bb:
                    cuts.add(a)
                if ba < b < bb:
                    cuts.add(b)
        cuts = sorted(cuts)
        for a, b in zip(cuts, cuts[1:]):
            if b <= a:
                continue
            mid = 0.5 * (a + b)
            label = next((t for t in BUBBLE_PRIORITY if _point_covered(mid, type_union[t])), None)
            if label:
                accum[label] += b - a
                seg_counts[label] += 1
            else:
                unlabeled += b - a
                seg_counts["idle_without_wait"] += 1
    rows = []
    for t in WAIT_TYPES:
        if accum[t] > EPS:
            rows.append({"kind": t, "label_zh": WAIT_TYPE_ZH[t], "exclusive_us": _r(accum[t]),
                         "segments": seg_counts[t],
                         "fraction_of_bubble": None})
    if unlabeled > EPS:
        rows.append({"kind": "idle_without_wait", "label_zh": "无具名等待覆盖的空洞",
                     "exclusive_us": _r(unlabeled), "segments": seg_counts["idle_without_wait"],
                     "fraction_of_bubble": None})
    total = sum(r["exclusive_us"] or 0.0 for r in rows)
    for r in rows:
        r["fraction_of_bubble"] = _r((r["exclusive_us"] or 0.0) / total, 6) if total > 0 else None
    rows.sort(key=lambda x: -(x["exclusive_us"] or 0.0))
    return rows, {t: accum[t] for t in WAIT_TYPES}


# --------------------------------------------------------------------------- T_min / problem
def resolve_t_min(problem: Optional[Dict[str, Any]], kernel_us: float) -> Dict[str, Any]:
    p = problem or {}
    explicit = _first_num(p, ("t_min_us", "T_min_us", "tmin_us"))
    if explicit is None and isinstance(p.get("t_min"), dict):
        explicit = _first_num(p["t_min"], ("us", "t_min_us"))
    flops = _first_num(p, ("useful_flops", "useful_flop", "work_flops"))
    macs = _first_num(p, ("useful_macs", "macs"))
    if flops is None and macs is not None:
        flops = 2.0 * macs
    req_bytes = _first_num(p, ("required_bytes", "useful_bytes", "bytes", "memory_bytes"))
    peak_tflops = _first_num(p, ("peak_flops_tflops", "peak_tflops", "assumed_peak_tflops_bf16",
                                "peak_bf16_tflops", "peak_compute_tflops"))
    peak_bw_tbps = _first_num(p, ("peak_bandwidth_tbps", "hbm_peak_tbps", "peak_mem_tbps"))
    peak_bw_gbps = _first_num(p, ("peak_bandwidth_gbps", "peak_mem_gbps", "hbm_peak_gbps"))
    if peak_bw_tbps is None and peak_bw_gbps is not None:
        peak_bw_tbps = peak_bw_gbps / 1000.0
    comp_us = mem_us = None
    if flops is not None and peak_tflops and peak_tflops > 0:
        comp_us = flops / (peak_tflops * 1e6)
    if req_bytes is not None and peak_bw_tbps and peak_bw_tbps > 0:
        mem_us = req_bytes / (peak_bw_tbps * 1e6)
    candidates = {"compute": comp_us, "memory": mem_us}
    valid = {k: v for k, v in candidates.items() if v is not None}
    derived = max(valid.values()) if valid else None
    requested_resource = p.get("t_min_resource")
    if explicit is not None:
        tmin = explicit
        if requested_resource:
            res = str(requested_resource)
        elif comp_us is not None and mem_us is not None:
            res = "compute+memory" if abs(comp_us - mem_us) <= EPS else max(candidates, key=candidates.get)
        else:
            res = "explicit"
        confidence = "explicit"
    elif derived is not None:
        tmin = derived
        if requested_resource:
            res = str(requested_resource)
        else:
            res = max(valid, key=valid.get)
            if len(valid) == 2 and abs(comp_us - mem_us) <= max(1e-6, 0.005 * max(comp_us, mem_us)):
                res = "compute+memory"
        confidence = "derived_from_assumed_peaks"
    else:
        tmin, res, confidence = None, "unknown", "missing_inputs"
    ev = []
    if p.get("peak_flops_source"):
        ev.append(str(p["peak_flops_source"]))
    if p.get("peak_bandwidth_source"):
        ev.append(str(p["peak_bandwidth_source"]))
    if p.get("ground_truth_controlled"):
        ev.append("problem 标记为受控真值基准")
    model = {
        "formula": "T_min_us = max(useful_flops/(peak_flops_tflops*1e6), required_bytes/(peak_bandwidth_tbps*1e6))",
        "useful_flops": _r(flops, 3),
        "useful_macs": _r(macs, 3),
        "required_bytes": _r(req_bytes, 3),
        "peak_flops_tflops": _r(peak_tflops, 6),
        "peak_bandwidth_tbps": _r(peak_bw_tbps, 6),
        "compute_bound_us": _r(comp_us),
        "memory_bound_us": _r(mem_us),
        "t_min_us": _r(tmin),
        "t_min_resource": res,
        "confidence": confidence,
        "evidence": ev,
        "is_hardware_measured_peak": bool(p.get("peak_is_measured", False)),
        "ground_truth_controlled": bool(p.get("ground_truth_controlled", False)),
    }
    if tmin is not None and kernel_us > 0 and tmin > 1.05 * kernel_us:
        model["inconsistent_with_kernel"] = True
    return model


# --------------------------------------------------------------------------- 硬件指标
def _series_stats(v: Sequence[float]) -> Dict[str, Any]:
    if not v:
        return {"n": 0}
    return {
        "n": len(v), "mean": _r(sum(v) / len(v)), "median": _r(_median(v)),
        "p10": _r(_percentile(v, 0.10)), "p90": _r(_percentile(v, 0.90)),
        "min": _r(min(v)), "max": _r(max(v)),
    }


def summarize_hardware(trace: Dict[str, Any], T: float, arch_override: Optional[str] = None) -> Dict[str, Any]:
    series: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for e in trace.get("traceEvents", []):
        if e.get("ph") != "C":
            continue
        ts, val = _f(e.get("ts")), None
        args = e.get("args") if isinstance(e.get("args"), dict) else {}
        if isinstance(args, dict):
            val = _f(args.get("value"))
        if ts is None or val is None:
            continue
        if ts < -EPS or ts > T + EPS:
            continue
        series[str(e.get("name", ""))].append((ts, val))
    hw_info = (trace.get("otherData", {}) or {}).get("hw_sw_info", {}) or {}
    gpu = hw_info.get("gpu", {}) if isinstance(hw_info, dict) else {}
    arch = detect_arch(gpu) if gpu else None
    if arch_override:
        arch = detect_arch({"arch": arch_override}) or arch_override
    metric_meta = hw_info.get("hw_metrics", {}) if isinstance(hw_info, dict) else {}
    declared_names = metric_meta.get("names") if isinstance(metric_meta, dict) else None
    names = sorted(series) if series else sorted(declared_names or [])
    status = {
        "present": bool(series), "arch": arch, "gpu": gpu.get("name") if isinstance(gpu, dict) else None,
        "scope": (metric_meta.get("scope") if isinstance(metric_meta, dict) else None) or "未知；通常为整卡聚合",
        "thresholds": dict(METRIC_THRESHOLDS), "period_us": None,
        "n_names": len(names), "available_names": names, "used": [], "missing_semantics": [],
        "mapping_method": None, "runtime_verified_on_machine": False, "warnings": [],
    }
    if not series:
        status["warnings"].append("trace 中没有 ph=C GPU metrics；归因只能依赖软件事件和 T_min 模型。")
        if arch:
            try:
                doc = nvidia_arch_document(arch)
                status["declared_mapping"] = {k: v.get("names") for k, v in doc["semantics"].items()}
                status["runtime_verified_on_machine"] = bool(doc.get("runtime_verified_on_machine"))
            except KeyError:
                status["declared_mapping"] = None
        return status
    # 采样周期：同一时刻常有多条 metric，取所有时间戳的唯一差中位数。
    times = sorted({round(ts, 6) for vs in series.values() for ts, _ in vs})
    if len(times) >= 2:
        gaps = [b - a for a, b in zip(times, times[1:]) if b - a > EPS]
        status["period_us"] = _r(_median(gaps))
    if not arch:
        # 指标名在三代 NVIDIA 上完全相同，不能仅凭名称猜代数；尝试每个 arch 的 exact 匹配。
        candidates = []
        for a in ("gh100", "gb10x", "gb11x"):
            m = match_metric_names(names, a)
            if not m["missing"]:
                candidates.append(a)
        arch = candidates[0] if len(candidates) == 1 else None
        status["arch"] = arch
        if arch is None:
            status["warnings"].append("无法从 trace 的 GPU 信息确定 gh100/gb10x/gb11x；指标语义映射未定型。")
            return status
    try:
        mapping = match_metric_names(names, arch)
    except Exception as exc:  # 不静默吞掉映射错误
        status["warnings"].append(f"指标映射异常：{type(exc).__name__}: {exc}")
        return status
    status["mapping_method"] = "exact_then_regex"
    status["missing_semantics"] = mapping["missing"]
    try:
        adoc = nvidia_arch_document(arch)
        status["runtime_verified_on_machine"] = bool(adoc.get("runtime_verified_on_machine"))
        status["config_verified_on_machine"] = bool(adoc.get("config_verified_on_machine"))
        status["config_sha256"] = adoc.get("observed_config_sha256")
        status["mapping_evidence"] = adoc.get("evidence")
    except KeyError:
        status["runtime_verified_on_machine"] = False
    for sem, m in sorted({**mapping["exact"], **mapping["fallback"]}.items()):
        nm = m["observed_name"]
        vals = [v for _, v in series.get(nm, [])]
        st = _series_stats(vals)
        st.update({
            "semantic": sem, "label_zh": m.get("label_zh"), "observed_name": nm,
            "canonical": m.get("canonical"), "match_method": m.get("match_method"),
            "unit": m.get("unit"), "threshold_high_pct": METRIC_THRESHOLDS["high_pct"],
        })
        status["used"].append(st)
    status["used"].sort(key=lambda x: x["semantic"])
    per_sem_n = [x["n"] for x in status["used"]]
    status["samples_min"] = min(per_sem_n) if per_sem_n else 0
    status["samples_max"] = max(per_sem_n) if per_sem_n else 0
    period = status.get("period_us") or _f(metric_meta.get("period_us") if isinstance(metric_meta, dict) else None) or 0.0
    status["estimated_coverage_fraction"] = _r(min(1.0, max(0.0, (status["samples_min"] - 1) * period / T)), 6) if T > 0 and status.get("samples_min") else 0.0
    if status["samples_min"] < METRIC_THRESHOLDS["min_samples"]:
        status["warnings"].append(f"GPU metrics 内核窗口内最少只有 {status['samples_min']} 个采样；短 kernel 的平均值不稳。")
    if period and period > METRIC_THRESHOLDS["max_period_us"]:
        status["warnings"].append(f"GPU metrics 采样周期 {period:g} us 大于阈值 {METRIC_THRESHOLDS['max_period_us']} us。")
    if T > 0 and status.get("estimated_coverage_fraction", 0) < 0.5:
        status["warnings"].append("GPU metrics 估计覆盖率 <50%；不要把曲线当作逐时刻真值。")
    status["warnings"].append("GPU metrics 是整卡聚合/平均，不能定位到单个 SM 或 warp；5 us 内的事件可能被平滑。")
    return status


def _hw_map(hw: Dict[str, Any]) -> Dict[str, float]:
    return {u["semantic"]: float(u["mean"]) for u in hw.get("used", []) if u.get("mean") is not None}


# --------------------------------------------------------------------------- 归因主逻辑
def _dominant_term(tmin: Optional[float], peak_loss: Optional[float], bubble: Optional[float],
                   tolerance: float) -> str:
    if tmin is None or peak_loss is None or bubble is None:
        return "unavailable"
    vals = {"t_min": tmin, "peak_loss": peak_loss, "bubble": bubble}
    top = max(vals, key=vals.get)
    ordered = sorted(vals.values(), reverse=True)
    if len(ordered) > 1 and ordered[0] - ordered[1] <= tolerance:
        # 两项接近时，若其中一项是 T_min，说明其它项没有形成清晰主因。
        return "balanced"
    return top


def _make_primary(records: List[Dict[str, Any]], cta_stats: Dict[str, Any], T: float,
                  bubble_us: float, bubble_rows: List[Dict[str, Any]], hw: Dict[str, Any],
                  tmin_model: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, Any]], List[str]]:
    hv = _hw_map(hw)
    tail_us = float(cta_stats.get("tail_us") or 0.0)
    tail_fraction = tail_us / T if T > 0 else 0.0
    tail_min = max(10.0, 0.10 * T)
    tail_signal = tail_us >= tail_min and tail_fraction >= 0.10
    bubble_signal = bubble_us >= max(10.0, 0.05 * T) and bubble_us / T >= 0.05 if T > 0 else False
    top_bubble = bubble_rows[0]["kind"] if bubble_rows else None
    active_by_kind: Dict[str, float] = {}
    for kind in ("compute", "load", "store", "memory", "control", "other"):
        active_by_kind[kind] = interval_duration((r["start"], r["end"]) for r in records if r["kind"] == kind)
    evidence: List[str] = []
    notes: List[str] = []
    # 拖尾优先：它既可能表现为尾部空泡，也可能藏在 peak_loss 中（波次未占满 SM）。
    if tail_signal:
        evidence.append(f"CTA 最晚结束比中位晚 {tail_us:.3f} us（{100*tail_fraction:.1f}% kernel）；阈值 {tail_min:.3f} us。")
        return "tail_imbalance", "tail", evidence, notes
    if bubble_signal:
        evidence.append(f"墙钟空泡 {bubble_us:.3f} us（{100*bubble_us/T:.1f}% kernel），最大具名类型 {top_bubble or '无'}。")
        if top_bubble == "wait_tail":
            return "tail_imbalance", "tail", evidence, notes
        if top_bubble in ("wait_dependency", "wait_load"):
            return "dependency_wait", "dependency", evidence, notes
        if top_bubble == "wait_sync":
            return "sync_wait", "dependency", evidence, notes
        return "bubble_idle", "idle", evidence, notes
    # 硬件指标只有在有采样且语义匹配时才参与主判定。
    dram_hi = max(hv.get("dram_read", 0.0), hv.get("dram_write", 0.0))
    if dram_hi >= METRIC_THRESHOLDS["high_pct"]:
        evidence.append(f"DRAM read/write 均值最高 {dram_hi:.1f}%（阈值 {METRIC_THRESHOLDS['high_pct']:.0f}%）。")
        return "memory_bandwidth", "memory", evidence, notes
    tensor_hi = hv.get("tensor_active")
    if tensor_hi is not None and tensor_hi >= METRIC_THRESHOLDS["high_pct"] and active_by_kind.get("compute", 0) > 0:
        evidence.append(f"Tensor Active 均值 {tensor_hi:.1f}% 且时间线有 compute 区间。")
        return "compute_throughput", "compute", evidence, notes
    issue = hv.get("issue")
    if issue is not None and issue >= METRIC_THRESHOLDS["saturated_pct"] and tmin_model.get("t_min_resource") == "compute":
        evidence.append(f"SM Issue 均值 {issue:.1f}% 接近饱和，且 T_min 由计算下界主导。")
        return "compute_throughput", "compute", evidence, notes
    tres = tmin_model.get("t_min_resource")
    if tres in ("compute", "compute+memory"):
        evidence.append(f"T_min 由 {tres} 下界主导（{tmin_model.get('t_min_us')} us），且无显著空泡/拖尾。")
        return "compute_throughput", "compute", evidence, notes
    if tres == "memory":
        evidence.append(f"T_min 由内存下界主导（{tmin_model.get('t_min_us')} us），且无显著空泡/拖尾。")
        return "memory_bandwidth", "memory", evidence, notes
    # 没有 T_min 时，用活跃区间类型兜底，并在 trust 里降级。
    if active_by_kind:
        top = max(active_by_kind, key=active_by_kind.get)
        evidence.append(f"没有 T_min/硬件高利用率证据，按最长活跃类别 {top} 兜底。")
        if top == "memory":
            return "memory_bandwidth", "memory", evidence, notes
        if top in ("compute", "load", "store"):
            return "compute_throughput" if top == "compute" else "memory_bandwidth", "compute" if top == "compute" else "memory", evidence, notes
    notes.append("没有足够证据确定主因。")
    return "unknown", "unknown", evidence, notes


def _secondary(primary: str, records: List[Dict[str, Any]], cta_stats: Dict[str, Any], T: float,
               bubble_rows: List[Dict[str, Any]], hw: Dict[str, Any], dep_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    hv = _hw_map(hw)
    def add(kind: str, resource: str, evidence: str, strength: str = "medium"):
        if kind == primary or any(x["kind"] == kind for x in out):
            return
        out.append({"kind": kind, "resource": resource, "strength": strength, "evidence": evidence})
    tail_us = float(cta_stats.get("tail_us") or 0.0)
    if tail_us >= max(10.0, 0.10 * T):
        add("tail_imbalance", "tail", f"CTA 尾部差 {tail_us:.3f} us")
    if bubble_rows:
        top = bubble_rows[0]
        kind = {"wait_load": "dependency_wait", "wait_dependency": "dependency_wait",
                "wait_sync": "sync_wait", "wait_tail": "tail_imbalance"}.get(top["kind"])
        if kind:
            add(kind, "dependency" if kind != "tail_imbalance" else "tail",
                f"空泡中 {top.get('exclusive_us')} us 由 {top.get('label_zh')} 覆盖", "high" if top.get("fraction_of_bubble", 0) > 0.5 else "medium")
    if hv.get("dram_read", 0) >= METRIC_THRESHOLDS["idle_pct"] or hv.get("dram_write", 0) >= METRIC_THRESHOLDS["idle_pct"]:
        add("memory_bandwidth", "memory", f"DRAM read {hv.get('dram_read', 0):.1f}% / write {hv.get('dram_write', 0):.1f}%")
    if hv.get("tensor_active") is not None and hv["tensor_active"] >= 10.0:
        add("compute_throughput", "compute", f"Tensor Active {hv['tensor_active']:.1f}%")
    dep_entries = dep_summary.get("by_type", {})
    if dep_entries.get("wait_dependency", {}).get("sum_wait_us", 0) > max(100.0, 0.5 * T):
        add("dependency_wait", "dependency",
            f"跨角色依赖等待逐 warp 累计 {dep_entries['wait_dependency']['sum_wait_us']:.1f} us（并发和，不是墙钟）")
    if dep_entries.get("wait_sync", {}).get("sum_wait_us", 0) > max(100.0, 0.5 * T):
        add("sync_wait", "dependency",
            f"栅栏/同步等待逐 warp 累计 {dep_entries['wait_sync']['sum_wait_us']:.1f} us（并发和，不是墙钟）")
    if not out and records:
        waits = sum(r["dur"] for r in records if str(r["kind"]).startswith("wait_"))
        if waits > T:
            add("dependency_wait", "dependency", f"具名等待逐 warp 累计 {waits:.1f} us > kernel {T:.1f} us", "low")
    return out[:4]


# --------------------------------------------------------------------------- 建议来自 configs/validation/fixes.json

FIXES_CACHE: Dict[str, Any] = {}


def default_fixes_path() -> str:
    """包内 configs/validation/fixes.json 的绝对路径（weaveprof/pipeline/validate/tools/ 的上一级目录）。"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "kb", "fixes.json")


def load_fixes_kb(path: Optional[str] = None) -> Dict[str, Any]:
    """读 configs/validation/fixes.json：损失类型 -> 改法（fix/how/cost/verify）。

    建议文本全部来自该文件，代码里不再写死改法。读不到时 available=False，
    调用方必须把这件事显式写进 attrib.json，不允许退化成凭空生成的建议。
    """
    key = os.path.abspath(path or default_fixes_path())
    if key in FIXES_CACHE:
        return FIXES_CACHE[key]
    doc: Dict[str, Any] = {"available": False, "path": key}
    try:
        raw = open(key, "rb").read()
        d = json.loads(raw.decode("utf-8"))
        doc = {"available": True, "path": key, "sha256": hashlib.sha256(raw).hexdigest(),
               "loss_types": list(d.get("loss_types") or []),
               "attrib_kind_map": dict(d.get("attrib_kind_map") or {}),
               "fixes": list(d.get("fixes") or []),
               "observability": list(d.get("observability") or [])}
    except (OSError, ValueError) as e:
        doc["error"] = f"{type(e).__name__}: {e}"
    FIXES_CACHE[key] = doc
    return doc


def fixes_provenance(kb: Dict[str, Any]) -> Dict[str, Any]:
    return {"source": "configs/validation/fixes.json", "path": kb.get("path"), "available": bool(kb.get("available")),
            "sha256": kb.get("sha256"), "error": kb.get("error"),
            "loss_types": list(kb.get("loss_types") or [])}


def _loss_types(primary: str, secondary: Sequence[Dict[str, Any]], kb: Dict[str, Any]) -> List[str]:
    """attrib 的 primary/secondary kind -> kb 的损失类型，按证据强度排序去重。"""
    km = kb.get("attrib_kind_map") or {}
    out: List[str] = []
    for kind in [primary] + [str(s.get("kind")) for s in secondary]:
        lt = km.get(kind)
        if lt and lt not in out:
            out.append(lt)
    return out


def _run_evidence(loss_type: str, ctx: Dict[str, Any]) -> str:
    """本次运行的实测数字（文本由代码拼，改法文本来自 kb）。"""
    T = float(ctx.get("T") or 0.0)
    cta = ctx.get("cta_stats") or {}
    tmin = ctx.get("tmin_model") or {}
    hw = ctx.get("hw") or {}
    rows = ctx.get("bubble_rows") or []
    head = f"本次 kernel {T:.3f} us，主因 {ctx.get('primary')}；损失类型判定为「{loss_type}」。"
    extra = []
    if loss_type == "依赖等待":
        deps = ctx.get("deps") or []
        if deps:
            d0 = deps[0]
            extra.append(f"最大等待事件 {d0.get('event')}（累计 {d0.get('sum_wait_us')} us，跨 warp 求和不是墙钟）")
        for r in rows[:2]:
            if str(r.get("kind")).startswith("wait_"):
                extra.append(f"空泡中 {r.get('label_zh')} 独占 {r.get('exclusive_us')} us")
    elif loss_type == "装载未掩盖":
        hv = _hw_map(hw)
        if hv:
            extra.append(f"DRAM read {hv.get('dram_read', 0.0):.1f}% / write {hv.get('dram_write', 0.0):.1f}%")
        if tmin.get("memory_bound_us") is not None:
            extra.append(f"内存下界 {tmin.get('memory_bound_us')} us")
        for r in rows[:2]:
            if r.get("kind") == "wait_load":
                extra.append(f"wait_load 独占空泡 {r.get('exclusive_us')} us")
    elif loss_type == "拖尾":
        extra.append(f"CTA 尾部差 {cta.get('tail_us')} us，最慢 CTA={cta.get('slowest_cta')}")
        roles = ctx.get("role_rows") or []
        worst = max(roles, key=lambda r: _f(r.get("tail_us")) or 0.0, default=None)
        if worst:
            extra.append(f"角色 {worst.get('role')} 结束中位 {worst.get('end_median_us')} us、最晚 {worst.get('end_max_us')} us")
    elif loss_type == "序言":
        extra.append(f"墙钟空泡 {ctx.get('bubble_us')} us；硬件指标样本 {hw.get('samples_min', 0)} 分钟点")
    elif loss_type == "算得慢":
        extra.append(f"T_min={tmin.get('t_min_us')} us（{tmin.get('t_min_resource')}），peak_loss={ctx.get('peak_loss')} us")
        hv = _hw_map(hw)
        if hv:
            extra.append(f"Tensor Active {hv.get('tensor_active', float('nan')):.1f}% / SM Issue {hv.get('issue', float('nan')):.1f}%")
    return head + ("；".join(extra) if extra else "")


def _obs_applies(when: str, primary: str, hw: Dict[str, Any], short_roles: bool) -> bool:
    """kb 的 observability.when 是受限表达式，只认三种写法，不用 eval。"""
    w = str(when or "").strip()
    m = re.match(r"^primary in \((.*)\)$", w)
    if m:
        vals = {x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()}
        return primary in vals
    if w == "hardware_metrics.present == false":
        return not hw.get("present")
    if w == "short_lived_roles_present":
        return bool(short_roles)
    return False


def _recommendations(primary: str, secondary: List[Dict[str, Any]], ctx: Dict[str, Any],
                     kb: Dict[str, Any], max_items: int = 12) -> List[Dict[str, Any]]:
    """建议全部取自 configs/validation/fixes.json：损失类型由 primary/secondary 映射，改法文本原样引用。"""
    if not kb.get("available"):
        return [{"priority": "high", "loss_type": None,
                 "title": "configs/validation/fixes.json 不可用，无法给出改法",
                 "action": f"确认包内存在 configs/validation/fixes.json（默认路径 {default_fixes_path()}），或用 --fixes 指定后重跑 fi_attrib。",
                 "cost": None, "verify": "重跑 fi_attrib 后 recommendations[].loss_type 非空。",
                 "evidence": f"读取失败：{kb.get('error')}", "symptom": None, "evidence_to_check": []}]
    by_lt = {str(f.get("loss_type")): f for f in (kb.get("fixes") or [])}
    order = _loss_types(primary, secondary, kb)
    out: List[Dict[str, Any]] = []

    def add(priority: str, lt: Optional[str], g: Dict[str, Any], entry: Optional[Dict[str, Any]], evidence: str) -> None:
        out.append({"priority": priority, "loss_type": lt, "title": g.get("fix"),
                    "action": g.get("how"), "cost": g.get("cost"), "verify": g.get("verify"),
                    "evidence": evidence,
                    "symptom": (entry or {}).get("symptom"),
                    "evidence_to_check": list((entry or {}).get("evidence_to_check") or [])})

    for i, lt in enumerate(order):
        entry = by_lt.get(lt)
        if not entry:
            continue
        is_primary = i == 0
        ev = _run_evidence(lt, dict(ctx, primary=primary))
        for j, g in enumerate(entry.get("fixes") or []):
            if is_primary:
                prio = "high" if j < 2 else "medium"
            else:
                prio = "medium" if j == 0 else "low"
            add(prio, lt, g, entry, ev)
    if not order:
        add("high", None, {"fix": f"主因 {primary} 在 configs/validation/fixes.json 的 attrib_kind_map 里没有对应损失类型",
                           "how": "先补齐打点与硬件指标，把空泡拆成具名等待类型，再重跑 fi_attrib；"
                                  "若确认是新的损失类型，把它补进 configs/validation/fixes.json 而不是写在代码里。",
                           "cost": None, "verify": "重跑后 primary 能映射到损失类型，recommendations[].loss_type 非空。"},
            None, f"primary={primary}，attrib_kind_map 未覆盖。")
    role_rows = ctx.get("role_rows") or []
    T = float(ctx.get("T") or 0.0)
    short_roles = [r for r in role_rows
                   if _f(r.get("end_median_us")) is not None and _f(r["end_median_us"]) < 0.05 * T
                   and (r.get("events") or 0) > 0]
    hw = ctx.get("hw") or {}
    for ob in (kb.get("observability") or []):
        if _obs_applies(ob.get("when"), primary, hw, bool(short_roles)):
            add("medium" if ob.get("when", "").startswith("primary") else "low", None, ob, None,
                f"触发条件 when={ob.get('when')}；primary={primary}，硬件指标样本={hw.get('samples_min', 0)}。")
    return out[:max_items]


def analyze(trace: Dict[str, Any], problem: Optional[Dict[str, Any]] = None,
            source_path: Optional[str] = None, problem_path: Optional[str] = None,
            summary: Optional[Dict[str, Any]] = None, arch_override: Optional[str] = None,
            fixes_path: Optional[str] = None, max_names: int = DEFAULT_MAX_NAMES,
            max_dep_rows: int = DEFAULT_MAX_DEP_ROWS) -> Dict[str, Any]:
    trace = trace or {}
    fixes_kb = load_fixes_kb(fixes_path)
    events = extract_software_events(trace)
    other = trace.get("otherData", {}) or {}
    align = other.get("align", {}) or {}
    kernel_us = _f(align.get("kernel_us"))
    spans = [e["end"] - e["start"] for e in events]
    span_max = _f(summary.get("span_us")) if summary else None
    if kernel_us is None:
        # 软件跨度不等于 kernel；必须明确降级，不用它伪装 CUPTI kernel 时长。
        kernel_us = span_max if span_max is not None else (max((e["end"] for e in events), default=0.0) -
                                                             min((e["start"] for e in events), default=0.0))
        kernel_source = "software_span_fallback"
    else:
        kernel_source = "nsys_align.kernel_us"
    kernel_us = max(float(kernel_us or 0.0), 1e-9)
    records = _prepare_events(events, kernel_us)
    active_iv = [(r["start"], r["end"]) for r in records if not str(r["kind"]).startswith("wait_")]
    wait_iv = [(r["start"], r["end"]) for r in records if str(r["kind"]).startswith("wait_")]
    active_union = merge_intervals(active_iv)
    active_span = interval_duration(active_union)
    bubble_segments = complement_intervals(active_union, 0.0, kernel_us)
    bubble_us = interval_duration(bubble_segments)
    wait_union = merge_intervals(wait_iv)
    wait_covered = interval_duration(wait_union)
    wait_in_bubble = interval_duration(intersect_interval_sets(wait_union, bubble_segments))
    # 等待区间被粗粒度 active 区间包住的墙钟部分。它不计入 bubble（同一时刻仍有别的区间在 active），
    # 但足够大时说明打点边界把等待混进了“主阶段”，主因只能记到 peak_loss，必须显式提示读者。
    wait_nested_in_active = max(0.0, wait_covered - wait_in_bubble)
    bubble_rows, _ = _partition_bubble(bubble_segments, records)
    cta_stats, role_rows = _cta_and_role_stats(records, max_names)
    deps, dep_summary = _dependency_stats(records, bubble_segments, max_names)
    # 写进 JSON 的明细可以截断；建议与主因判定仍用全量 deps（见下面的 rec_ctx）。
    deps_out, dep_trunc = _cap_rows(deps, max_dep_rows)
    tmin_model = resolve_t_min(problem, kernel_us)
    tmin = tmin_model.get("t_min_us")
    peak_loss = max(0.0, active_span - float(tmin)) if tmin is not None else None
    reconstructed = (float(tmin) + float(peak_loss) + bubble_us) if tmin is not None and peak_loss is not None else None
    unaccounted = kernel_us - reconstructed if reconstructed is not None else None
    tolerance = max(1.0, 0.005 * kernel_us)
    within = bool(unaccounted is not None and abs(unaccounted) <= tolerance)
    model_inconsistent = bool(tmin is not None and tmin > active_span + tolerance)
    hw = summarize_hardware(trace, kernel_us, arch_override=arch_override)
    primary, resource, evidence, notes = _make_primary(records, cta_stats, kernel_us, bubble_us, bubble_rows, hw, tmin_model)
    secondary = _secondary(primary, records, cta_stats, kernel_us, bubble_rows, hw, dep_summary)
    rec_ctx = {"T": kernel_us, "cta_stats": cta_stats, "role_rows": role_rows, "deps": deps, "hw": hw,
               "tmin_model": tmin_model, "bubble_rows": bubble_rows, "bubble_us": bubble_us,
               "peak_loss": peak_loss}
    recommendations = _recommendations(primary, secondary, rec_ctx, fixes_kb)
    dominant = _dominant_term(tmin, peak_loss, bubble_us, tolerance)
    checks = {
        "traceEvents_nonempty": bool(trace.get("traceEvents")),
        "software_events_nonempty": bool(records),
        "structural_errors_empty": not bool(other.get("errors")),
        "alignment_ok": bool(align.get("ok", False)) if align else None,
        "kernel_duration_from_cupti": kernel_source == "nsys_align.kernel_us",
        "t_min_available": tmin is not None,
        "cta_mapping_available": bool(cta_stats.get("n")),
        "accounting_within_tolerance": within,
        "hardware_metrics_present": bool(hw.get("present")),
        "hardware_metrics_sampled_enough": (hw.get("samples_min", 0) >= METRIC_THRESHOLDS["min_samples"]) if hw.get("present") else False,
    }
    warnings: List[str] = []
    if other.get("errors"):
        warnings.append(f"软件转储结构错误：{other['errors']}；时间线不完整，结论降级。")
    if not align:
        warnings.append("trace 没有 otherData.align；kernel_us 使用软件跨度兜底，包含启动/收尾边界，不能当 CUPTI kernel 时长。")
    elif not align.get("ok"):
        warnings.append(f"nsys 对齐失败：align.ok={align.get('ok')} end_margin_us={align.get('end_margin_us')}；可能找错 kernel 实例。")
    if tmin is None:
        warnings.append("没有 problem 的 T_min 输入（useful_flops/required_bytes/峰值或显式 t_min_us）；峰值损失与账本不可用。")
    elif tmin_model.get("confidence") != "explicit":
        warnings.append("T_min 由 assumed peak 推导，不是本机 microbenchmark 实测峰值；峰值损失包含频率、占空和指令混用等因素。")
    if model_inconsistent:
        warnings.append("T_min 高于 active_span；请核对 shape/flops/峰值假设，账本残差会为非零。")
    if hw.get("warnings"):
        warnings.extend(hw["warnings"])
    if kernel_us > 0 and wait_nested_in_active >= 0.50 * kernel_us and wait_in_bubble <= 0.10 * max(wait_covered, EPS):
        warnings.append(
            f"等待区间与 active 区间高度重叠：wait_covered={wait_covered:.3f} us，其中 {wait_nested_in_active:.3f} us "
            f"落在 active 区间内（{100*wait_nested_in_active/kernel_us:.1f}% kernel）。说明部分角色主阶段把等待包在内部，墙钟 bubble 近零；"
            "主因按 peak_loss 记账，依赖机制请看 dependencies/dependency_summary，不能据此说没有等待。")
    if not hw.get("present"):
        warnings.append("没有 GPU metrics 时，瓶颈资源主要依据软件语义和 T_min；不能证明整卡 Tensor/DRAM 饱和。")
    else:
        warnings.append("GPU metrics 是全卡平均；5 us 采样可能把短等待/尾部平滑掉，必须和软件 CTA/warp 区间交叉判断。")
    if records and any(r["name"].startswith("未命名事件") or re.match(r"^事件\d+$", r["name"]) for r in records):
        warnings.append("存在未命名事件；names.json 语义不完整，角色/依赖归因可信度下降。")
    # 账本置信度：有 CUPTI 对齐、T_min、干净事件、GPU 指标才算 high。
    if (not records or tmin is None or (align and not align.get("ok")) or other.get("errors")
            or model_inconsistent or (unaccounted is not None and not within)):
        trust_level = "low"
    elif hw.get("present") and hw.get("samples_min", 0) >= METRIC_THRESHOLDS["min_samples"] and within:
        trust_level = "high"
    else:
        trust_level = "medium"
    if trust_level == "high" and not tmin_model.get("is_hardware_measured_peak"):
        # 不降级：T_min 的口径已经明确写成 assumed；high 表示“归因证据链完整”，不是“峰值已实测”。
        warnings.append("trust=high 表示对齐/账本/采样证据完整；T_min 峰值仍是模型假设，见 model.confidence。")
    sw_span_start = min((r["start"] for r in records), default=0.0)
    sw_span_end = max((r["end"] for r in records), default=0.0)
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_utc(),
        "tool": {"name": "fast-instrument fi_attrib", "version": TOOL_VERSION,
                 "formula": "T_kernel = T_min + peak_loss + bubble + unaccounted"},
        "source": {
            "trace": os.path.abspath(source_path) if source_path else None,
            "trace_sha256": sha256_file(source_path) if source_path and os.path.exists(source_path) else None,
            "trace_tool": other.get("tool"),
            "trace_time_zero": other.get("time_zero"),
            "kernel_us": _r(kernel_us),
            "kernel_us_source": kernel_source,
            "software_span_start_us": _r(sw_span_start),
            "software_span_end_us": _r(sw_span_end),
            "software_span_us": _r(sw_span_end - sw_span_start),
            "align": align,
            "gpu": (other.get("hw_sw_info", {}) or {}).get("gpu"),
            "launch": (other.get("hw_sw_info", {}) or {}).get("launch"),
            "problem_path": os.path.abspath(problem_path) if problem_path else None,
            "problem_sha256": sha256_file(problem_path) if problem_path and os.path.exists(problem_path) else None,
        },
        "kernel_us": _r(kernel_us),
        "bottleneck_resource": resource,
        "primary_attribution": primary,
        "bottleneck_zh": RESOURCE_ZH.get(resource, resource),
        "primary_attribution_zh": PRIMARY_ZH.get(primary, primary),
        "T_min_us": _r(tmin),
        "t_min_us": _r(tmin),
        "peak_loss_us": _r(peak_loss),
        "bubble_us": _r(bubble_us),
        "dominant_term": dominant,
        "accounting": {
            "model": "T_kernel = T_min + peak_loss + bubble + unaccounted",
            "T_kernel_us": _r(kernel_us),
            "T_min_us": _r(tmin),
            "peak_loss_us": _r(peak_loss),
            "bubble_us": _r(bubble_us),
            "reconstructed_us": _r(reconstructed),
            "unaccounted_us": _r(unaccounted),
            "tolerance_us": _r(tolerance),
            "within_tolerance": within,
            "model_inconsistent": model_inconsistent,
            "active_span_us": _r(active_span),
            "exclusive_idle_us": _r(bubble_us),
            "wait_covered_us": _r(wait_covered),
            "wait_in_bubble_us": _r(wait_in_bubble),
            "wait_nested_in_active_us": _r(wait_nested_in_active),
            "idle_without_named_wait_us": _r(max(0.0, bubble_us - wait_in_bubble)),
            "method": "active intervals=所有非 wait 软件区间；bubble=其在 [0,kernel_us] 的补集；peak_loss=active_span-T_min。",
            "notes": [
                "peak_loss 不是“实测峰值损失”的唯一分量；它包含未达到峰值时的效率/占空损失。",
                "bubble 只计算墙钟无有效区间的时间，不把跨 warp 并发等待错误相加。",
                "wait_nested_in_active_us 是等待区间被粗粒度 active 区间包住的部分；它不计入 bubble，但说明打点边界需要细化。",
            ],
        },
        "model": tmin_model,
        "timeline": {
            "events": len(events), "spans_after_clip": len(records),
            "roles": len({r["role"] for r in records}), "ctas": cta_stats.get("n", 0),
            "active_interval_us": _r(active_span), "active_coverage_fraction": _r(active_span / kernel_us if kernel_us else None),
            "wait_covered_us": _r(wait_covered), "wait_coverage_fraction": _r(wait_covered / kernel_us if kernel_us else None),
            "bubble_us": _r(bubble_us), "bubble_fraction": _r(bubble_us / kernel_us if kernel_us else None),
            "summed_wait_over_lanes_us": _r(sum(r["dur"] for r in records if str(r["kind"]).startswith("wait_"))),
            "active_by_kind": {k: _r(v) for k, v in sorted(((k, interval_duration((r["start"], r["end"]) for r in records if r["kind"] == k))
                                                               for k in ACTIVE_KINDS), key=lambda x: x[1], reverse=True)},
        },
        "roles": role_rows,
        "cta_stats": cta_stats,
        "dependencies": deps_out,
        "dependency_truncation": dep_trunc,
        "dependency_summary": dep_summary,
        "bubble_breakdown": bubble_rows,
        "hardware_metrics": hw,
        "secondary_attributions": secondary,
        "recommendations": recommendations,
        "recommendations_source": fixes_provenance(fixes_kb),
        "trust": {
            "level": trust_level,
            "reason": {
                "high": "CUPTI 对齐、干净语义事件、T_min 和足量 GPU metrics 均可用；仍须遵守整卡平均/采样限制。",
                "medium": "软件账本可用，但 GPU metrics 缺失/采样不足或 T_min 是模型假设；结论需人工核对。",
                "low": "事件缺失、结构错误、对齐失败或 T_min 不可用；不要据此做性能决策。",
            }[trust_level],
            "warnings": list(dict.fromkeys(warnings)),
            "checks": checks,
            "not_proven": [
                "不能把 GPU metrics 的整卡均值定位到某个 SM/warp（除非未来加入 per-SM CUPTI/PM sampling）。",
                "T_min 是基于调用方给定峰值/字节的 roofline 下界，不是 ncu 的实测峰值。",
                "peak_loss 是残差账本中的有效工作未达峰值部分，不等于某个单一微架构事件的独占时长。",
            ],
        },
        "checks": checks,
        "evidence": evidence,
        "notes": notes,
        "summary_zh": "",
    }
    # 账本残差不应被偷偷分摊；在摘要中显式写出来。
    if tmin is not None:
        parts = (f"kernel {kernel_us:.3f} us；主因 {primary}（{RESOURCE_ZH.get(resource, resource)}）；"
                 f"T_min {float(tmin):.3f} + peak_loss {float(peak_loss):.3f} + bubble {bubble_us:.3f} "
                 f"+ unaccounted {float(unaccounted):.3f} us。")
    else:
        parts = f"kernel {kernel_us:.3f} us；主因 {primary}；T_min 输入缺失，peak_loss 账本不可用。"
    if bubble_rows:
        parts += f" 空泡首位：{bubble_rows[0]['label_zh']} {bubble_rows[0]['exclusive_us']:.3f} us。"
    result["summary_zh"] = parts
    return result


# --------------------------------------------------------------------------- 自测
def _mk_trace(events: List[Tuple[float, float, str, str, int]], T: float, metrics: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
    te = []
    for i, (a, b, name, role, cta) in enumerate(events):
        te.append({"ph": "X", "pid": 0 if role.startswith("计算") else 1, "tid": i, "ts": a, "dur": b - a,
                   "name": name, "cat": name, "args": {"role": role, "cta": cta, "warp": 0}})
    if metrics:
        for j, (name, vals) in enumerate(metrics.items()):
            for k, v in enumerate(vals):
                te.append({"ph": "C", "pid": 1 << 20, "ts": k * 5.0, "name": name, "args": {"value": v}})
    return {"traceEvents": te, "otherData": {"align": {"kernel_us": T, "ok": True, "end_margin_us": 0.1},
                                              "errors": {}, "hw_sw_info": {"gpu": {"name": "NVIDIA H20", "chipName": "GH100", "smCount": 78}}}}

def selftest() -> None:
    # 1) 装载等待 -> dependency_wait + bubble
    tr = _mk_trace([(0, 200, "消费者等装载", "计算 warp", 0), (0, 200, "装载线程人为延迟", "装载 warp", 0),
                    (200, 300, "FMA 计算", "计算 warp", 0)], 300)
    r = analyze(tr, {"useful_flops": 1000, "peak_flops_tflops": 1.0, "required_bytes": 0,
                      "ground_truth_controlled": True}, arch_override="gh100")
    assert r["primary_attribution"] == "dependency_wait", r["summary_zh"]
    assert r["bubble_us"] > 190 and r["accounting"]["within_tolerance"]
    # 2) active 全段、没有等待 -> compute / peak_loss
    tr = _mk_trace([(0, 100, "FMA 计算", "计算 warp", 0)], 100)
    r = analyze(tr, {"useful_flops": 10_000_000, "peak_flops_tflops": 1.0, "required_bytes": 0}, arch_override="gh100")
    assert r["primary_attribution"] == "compute_throughput", r
    assert r["peak_loss_us"] > 89 and r["bubble_us"] == 0
    # 3) 单 CTA 收尾拖尾 -> tail
    ev = [(0, 100, "FMA 计算", "计算 warp", c) for c in range(4)]
    ev += [(100, 160, "CTA 拖尾等待", "拖尾 CTA", 0)]
    tr = _mk_trace(ev, 160)
    r = analyze(tr, {"useful_flops": 1000, "peak_flops_tflops": 1.0, "required_bytes": 0}, arch_override="gh100")
    assert r["primary_attribution"] == "tail_imbalance", r["summary_zh"]
    assert r["bubble_us"] > 50
    # 4) 指标全匹配 + 高 DRAM -> memory
    metrics = {"Tensor Active [Throughput %]": [1] * 40, "SMs Active [Throughput %]": [99] * 40,
               "SM Issue [Throughput %]": [20] * 40, "DRAM Read Bandwidth [Throughput %]": [90] * 40,
               "DRAM Write Bandwidth [Throughput %]": [5] * 40,
               "Compute Warps in Flight [Throughput %]": [70] * 40}
    tr = _mk_trace([(0, 200, "HBM 读+写", "计算 warp", 0)], 200, metrics)
    tr["otherData"]["hw_sw_info"]["hw_metrics"] = {"scope": "整卡", "period_us": 5.0}
    r = analyze(tr, {"required_bytes": 800_000_000, "peak_bandwidth_tbps": 4.0}, arch_override="gh100")
    assert r["primary_attribution"] == "memory_bandwidth", r
    assert r["hardware_metrics"]["samples_min"] == 40
    # 5) 大跨度 disjoint 区间排序 + 账本残差
    assert merge_intervals([(2, 3), (1, 2.5), (5, 6)]) == [(1.0, 3.0), (5.0, 6.0)]
    r = analyze(_mk_trace([(0, 10, "FMA 计算", "计算 warp", 0)], 10),
                {"t_min_us": 20}, arch_override="gh100")
    assert r["peak_loss_us"] == 0 and r["accounting"]["unaccounted_us"] < 0
    # 6) 建议必须来自 configs/validation/fixes.json：每条都有损失类型、改法、代价与验证方式，且文本能在 kb 里找到
    kb = load_fixes_kb()
    assert kb["available"], kb.get("error")
    tr = _mk_trace([(0, 200, "消费者等装载", "计算 warp", 0), (0, 200, "装载线程人为延迟", "装载 warp", 0),
                    (200, 300, "FMA 计算", "计算 warp", 0)], 300)
    r = analyze(tr, {"useful_flops": 1000, "peak_flops_tflops": 1.0, "required_bytes": 0}, arch_override="gh100")
    recs = r["recommendations"]
    assert recs and r["recommendations_source"]["available"], r["recommendations_source"]
    kb_texts = {g["fix"] for f in kb["fixes"] for g in f["fixes"]} | {o["fix"] for o in kb["observability"]}
    for x in recs:
        assert x["title"] in kb_texts, x["title"]
        assert x["action"] and x["verify"], x
    assert any(x["loss_type"] == "依赖等待" for x in recs), [x["loss_type"] for x in recs]
    # kb 读不到时必须显式降级，不允许编造建议
    bad = load_fixes_kb("/nonexistent/fixes.json")
    recs2 = _recommendations("dependency_wait", [], {"T": 1.0}, bad)
    assert not bad["available"] and "不可用" in recs2[0]["title"], recs2
    # 7) 输出体积：重复别名字段已删；实例名与明细行数有上限；截断如实记账且可关掉
    ev = []
    for i in range(300):
        ev.append((0, 100, f"消费者等装载 #{i}", "计算 warp", 0))
    ev.append((100, 120, "FMA 计算", "计算 warp", 0))
    tr = _mk_trace(ev, 120)
    prob = {"useful_flops": 1000, "peak_flops_tflops": 1.0, "required_bytes": 0}
    r = analyze(tr, prob, arch_override="gh100")
    assert "role_breakdown" not in r and "dependency_breakdown" not in r, "逐字节重复的别名字段必须去掉"
    base = r["dependency_summary"]["by_event_base"]
    assert base and base[0]["event_base"] == "消费者等装载", base[:1]
    assert base[0]["instances"] == 300 and base[0]["count"] == 300, base[0]
    for row in r["roles"]:
        assert len(row["event_names"]) <= DEFAULT_MAX_NAMES, row["role"]
        assert row["event_names_truncated"] == (row["event_names_total"] > len(row["event_names"]))
    for ent in r["dependency_summary"]["by_type"].values():
        assert len(ent["event_names"]) <= DEFAULT_MAX_NAMES
        assert ent["event_names_total"] == 300, ent["event_names_total"]
    assert len(r["dependencies"]) == DEFAULT_MAX_DEP_ROWS, len(r["dependencies"])
    t = r["dependency_truncation"]
    assert t and t["dependency_rows_total"] == 300 and t["omitted_rows"] == 300 - DEFAULT_MAX_DEP_ROWS, t
    assert t["omitted_sum_wait_us"] > 0
    full = analyze(tr, prob, arch_override="gh100", max_dep_rows=0, max_names=0)
    assert len(full["dependencies"]) == 300 and full["dependency_truncation"] is None
    assert len(full["roles"][0]["event_names"]) == 301 or len(full["roles"][0]["event_names"]) == 300
    print("FI_ATTRIB SELFTEST PASS")


# --------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="fast-instrument 归因：软件时间线 + GPU metrics -> attrib.json")
    ap.add_argument("trace", nargs="?", help="输入 .perfetto.json[.gz]")
    ap.add_argument("--problem", help="problem JSON：useful_flops/required_bytes/峰值或显式 t_min_us")
    ap.add_argument("--summary", help="可选 summary JSON，用来补 span_us")
    ap.add_argument("--arch", help="gh100/gb10x/gb11x/gfx942；默认从 trace GPU 信息识别")
    ap.add_argument("-o", "--out", help="输出 attrib.json（默认 <trace>.attrib.json）")
    ap.add_argument("--fixes", help="configs/validation/fixes.json 路径；默认取包内 configs/validation/fixes.json，建议文本全部来自它")
    ap.add_argument("--metric-map-out", help="额外写出跨架构指标映射审计 JSON")
    ap.add_argument("--max-names", type=int, default=DEFAULT_MAX_NAMES,
                    help=f"roles/dependency_summary 里 event_names 最多列几个（默认 {DEFAULT_MAX_NAMES}，0=全列）")
    ap.add_argument("--max-dep-rows", type=int, default=DEFAULT_MAX_DEP_ROWS,
                    help=f"dependencies 明细最多保留几行，按 sum_wait_us 降序（默认 {DEFAULT_MAX_DEP_ROWS}，0=不截断）")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if not a.trace:
        ap.error("需要 trace；或使用 --selftest")
    try:
        trace = load_json(a.trace)
        problem = load_json(a.problem) if a.problem else None
        summary = load_json(a.summary) if a.summary else None
        out = a.out or (a.trace[:-3] + ".attrib.json" if a.trace.endswith(".gz") else a.trace + ".attrib.json")
        r = analyze(trace, problem, source_path=a.trace, problem_path=a.problem, summary=summary,
                    arch_override=a.arch, fixes_path=a.fixes,
                    max_names=a.max_names, max_dep_rows=a.max_dep_rows)
        write_json(r, out)
        if a.metric_map_out:
            write_json(mapping_document(a.arch), a.metric_map_out)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"fi_attrib 失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if not a.quiet:
        print(r["summary_zh"])
        print(f"主因={r['primary_attribution']} 资源={r['bottleneck_resource']} trust={r['trust']['level']} 写出={out}")
        for w in r["trust"]["warnings"][:8]:
            print(f"  warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
