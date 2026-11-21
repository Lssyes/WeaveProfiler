#!/usr/bin/env python3
"""selftest_hw_merge.py（骨架版）— E25 离线单测：锁死 SW/HW 对齐
(``hw_perfetto/align_sw_hw.py``) 与 ph='C' 硬件曲线合并
(``hw_perfetto/emit_perfetto.merge_hw``)。

流程位置
  由 ``validation/run_offline_selftests.py`` 自动发现并当子进程跑
  (``python3 -B selftest_hw_merge.py``，无 argparse，退出码 0 == PASS)。
  纯 CPU：不需要 GPU、不需要 nsys、不读任何运行产物 —— 解码事件和
  ``gpu-hw-metrics/v1`` 都在进程内合成，然后喂给**真实的**库函数；
  最后用**真实的** trace_processor_shell 导一遍，保证坏掉的 ph='C'
  不可能静默通过。

它到底在防什么
  * 对齐必须是 shift-only（``scale == 1``）：nsys 窗口比软件窗口宽是**结构性**的
    （nsys 量整个 kernel，软件只量第一次打点到最后一次打点），一旦有人用缩放去
    "修正"这个差值，PM 采样之间的间距就被伪造了 —— 这里把 ``span_mismatch_ns``
    直接钉死成断言；
  * 软件零点 = entry rendezvous 的 **release**，没有锚点时退化成最早 start；
  * R6：HW 曲线的 pid 必须严格小于所有软件 pid，且 ``process_sort_index == -1``，
    否则 Perfetto 里 HW 线会掉到 CTA 中间；
  * ph='C' 的 ``args.value`` 必须是有限数值（不能是 bool/None/nan/inf），
    ts 必须 >= 0（靠一次全局 rebase，不许 clamp/丢点/缩放单个事件）；
  * 窗口外的采样必须被裁掉，窗口内的采样数量与间距必须原样保留；
  * 缺 ``kernel_launches`` / 缺 ``nsys_window_ns`` 时必须**硬失败**，不许静默出图。

接新 target 要改的地方
  1. 配置区的 ``ABI_PATH``、``TRACE_PROCESSOR_SHELL_CANDIDATES``；
  2. ``COUNTER_SPECS``：换成你项目真实的 HW 指标名/单位（值随便，只验证形状）；
  3. 窗口/采样常量（``NSYS_*``、``SAMPLE_*``、``OP_WINDOW_*``）按你的量级调，
     所有断言都是**从这些常量算出来的**，不是写死的数字，改了不用改断言。
  事件表**不要**写死：task/phase/ignored 的分层、锚点 task、observer track
  全部从 ABI 派生（见 ``build_events_from_abi``）—— 写死的话 ABI 改了自测还能过。

真实参考实现
  DSA-learn/mk/h20/cutedsl/deepseek_v4_pro/decode/profile/validation/selftest/selftest_hw_merge.py
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve()
_PROFILE_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_PROFILE_ROOT / "hw_perfetto"))
sys.path.insert(0, str(_PROFILE_ROOT / "device_record" / "abi"))

import align_sw_hw  # noqa: E402
import emit_perfetto  # noqa: E402
from abi_loader import load_abi  # noqa: E402

# ======================================================================
# ===== 接新 target 时只需要改这里 (PROJECT CONFIG) =====
# ======================================================================
# 1) ABI：事件分层的唯一来源。换成你项目 extract_abi.py 产出的文件名。
ABI_PATH = _PROFILE_ROOT / "device_record" / "abi" / "record_abi.example.json"

# 2) trace_processor_shell：真实导入校验用。环境变量优先，其次按顺序找。
#    这是唯一一处"机器相关"的路径，找不到会 FAIL（不是 skip）—— 见文件尾说明。
TRACE_PROCESSOR_SHELL_CANDIDATES: Tuple[str, ...] = (
    # ===== PROJECT HOOK: 填你机器上 skill 的 1_scan/get_trace_processor.py 落地的路径 =====
    # （注意不是 .sh：skill 侧脚本名是 get_trace_processor.py，它会把可执行文件放到
    #   <instr_root>/_env/trace_processor_shell）
    "PROJECT_HOOK_REPLACE/_env/trace_processor_shell",
)

# 3) 合成 kernel 的规模（只影响可读性，不影响判定逻辑）
NUM_CTAS = 2                       # 合成几个 CTA（>=2 才能验 CTA 间 skew 与 pid 排序）
CTA_SKEW_NS = 50                   # 第 n 个 CTA 整体晚 n*CTA_SKEW_NS 出发
ANCHOR_START_NS = 900              # entry rendezvous arrive
ANCHOR_END_NS = 1000               # entry rendezvous release == 软件零点
OP_WINDOW_START_NS = 1000          # 算子 task 铺开的窗口
OP_WINDOW_END_NS = 9000
PHASE_PAD_NS = 100                 # phase 相对父 task 内缩多少（保证是嵌套不是部分重叠）
SYNTH_PAYLOAD_VALUE = 0            # tag 里的可选身份负载（expert_idx 之类），0 = 未用

# 4) 合成 nsys/CUPTI 侧窗口与 PM 采样
NSYS_START_NS = 5_000_000
NSYS_END_NS = 5_009_000
SAMPLE_COUNT = 5                   # 窗口内每条曲线的采样点数
SAMPLE_STEP_NS = 2_000             # 采样间隔（= 1/frequency_hz 的量级）
HW_FREQUENCY_HZ = 200_000          # 必须与 check_gate_item 里的采样率门禁一致
HW_DEVICE = 0
HW_DEVICE_NAME = "PROJECT_HOOK_REPLACE (e.g. NVIDIA H20 (Hopper))"
HW_METRICS_SET = "gpu-pm-sampling"
TARGET_KERNEL_NAME = "PROJECT_HOOK_REPLACE: nsys 里目标 kernel 的名字"

# 5) HW 指标定义：换成你项目真实的 metric/单位。**至少两条**才能验多轨道排序；
#    第一条会额外插一个窗口外采样，用来证明裁剪真的发生了。
COUNTER_SPECS: Tuple[Dict[str, Any], ...] = (
    {"metric_id": "sm_active", "name": "SMs Active",
     "raw_name": "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
     "unit": "%", "base": 55.0, "step": 8.0},
    {"metric_id": "dram_read", "name": "DRAM Read Throughput",
     "raw_name": "dram__bytes_read.sum.per_second",
     "unit": "GB/s", "base": 120.0, "step": -6.0},
)

# 6) trace_processor 导入时绝不允许出现的统计项（出现就说明 JSON 被静默丢事件了）
FORBIDDEN_IMPORT_STATS: Tuple[str, ...] = (
    "json_parser_failure", "json_tokenizer_failure",
    "slice_spill_overlapping_complete_event", "slice_drop_overlapping_complete_event",
    "trace_sorter_negative_timestamp_dropped",
)
TRACE_PROCESSOR_TIMEOUT_SEC = 180
# ======================================================================
# ===== 配置区结束 =====
# ======================================================================

# --- 以下全部由配置区推导，改配置不用改断言 ---------------------------- #
MAX_CTA_SKEW_NS = (NUM_CTAS - 1) * CTA_SKEW_NS
EXPECTED_SW_WINDOW = [ANCHOR_START_NS, OP_WINDOW_END_NS + MAX_CTA_SKEW_NS]
EXPECTED_SW_SPAN = EXPECTED_SW_WINDOW[1] - EXPECTED_SW_WINDOW[0]
EXPECTED_NSYS_WINDOW = [NSYS_START_NS, NSYS_END_NS]
EXPECTED_NSYS_SPAN = NSYS_END_NS - NSYS_START_NS
EXPECTED_SPAN_MISMATCH = EXPECTED_NSYS_SPAN - EXPECTED_SW_SPAN
EXPECTED_SHIFT_NS = ANCHOR_END_NS
OUT_OF_WINDOW_SAMPLE_NS = NSYS_END_NS + SAMPLE_STEP_NS * 10
EXPECTED_COUNTER_EVENTS = len(COUNTER_SPECS) * SAMPLE_COUNT
HW_MAP_FORMULA = "ts_us = (sw_start + (nsys_ts - nsys_start) - shift_ns) / 1000"


class SelfTestError(AssertionError):
    pass


# --------------------------------------------------------------------------- #
# PROJECT_HOOK_REPLACE 占位符防护
#   骨架里凡是"接新 target 必须填"的字符串都写成 PROJECT_HOOK_REPLACE。占位符长得
#   像一个合法值，所以**必须在被真正使用的那一刻炸掉**：否则 fixture 会拿着一个
#   假设备名 / 假 kernel 名一路跑绿，自测等于在验一个不存在的 kernel。
# --------------------------------------------------------------------------- #
PROJECT_HOOK_SENTINEL = "PROJECT_HOOK_REPLACE"


def unconfigured(value: Any) -> bool:
    """这一项还是占位符（= 根本没填）吗？"""
    return PROJECT_HOOK_SENTINEL in str(value)


def require_configured(name: str, value: Any, what: str) -> Any:
    """占位符 -> 硬 FAIL 并指明该填什么。绝不允许"占位符也算配好了"。"""
    if unconfigured(value):
        raise SelfTestError(
            "配置区的 {} 仍是 {} 占位符（当前值 {!r}）：{}。接新 target 的第一步就是把本文件"
            "配置区的占位符全部填掉 —— 占位符会让 fixture 变成假的，自测跑绿也没有意义。"
            .format(name, PROJECT_HOOK_SENTINEL, str(value), what))
    return value


def tps_hint() -> str:
    """找不到 trace_processor_shell 时的修复指引。
    这是**外部工具缺失**（不是骨架功能缺口），一律 FAIL，不许 skip。"""
    hint = ("缺外部工具 trace_processor_shell：用 skill 的 1_scan/get_trace_processor.py "
            "装一份，然后 export TRACE_PROCESSOR_SHELL=<可执行文件绝对路径>，"
            "或把绝对路径填进本文件配置区的 TRACE_PROCESSOR_SHELL_CANDIDATES")
    pending = [c for c in TRACE_PROCESSOR_SHELL_CANDIDATES if unconfigured(c)]
    if pending:
        hint += "；当前候选仍是未填的 {} 占位符：{}".format(PROJECT_HOOK_SENTINEL, list(pending))
    return hint


_OK_LINES = 0


def _require(what: str, cond: Any, detail: Any = "") -> None:
    """断言 + 可读输出。失败直接抛，绝不累积成"大部分通过"。"""
    global _OK_LINES
    if cond:
        _OK_LINES += 1
        print("  [ok]   {}".format(what))
        return
    raise SelfTestError("{} (detail={})".format(what, detail))


def _note(msg: str) -> None:
    print("  [note] {}".format(msg))


# --------------------------------------------------------------------------- #
# ABI 派生（不写死任何 event id / track id）
# --------------------------------------------------------------------------- #
def abi_task_index(abi: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    index: Dict[int, Dict[str, Any]] = {}
    for entry in abi.get("tasks") or []:
        index[int(entry["id"])] = entry
    if not index:
        raise SelfTestError("ABI 的 tasks[] 是空的，无法派生事件表：{}".format(ABI_PATH))
    return index


def abi_layer_ids(index: Dict[int, Dict[str, Any]], layer: str) -> List[int]:
    return sorted(tid for tid, entry in index.items()
                  if (entry.get("layer") or "task") == layer)


def abi_track_ids(abi: Dict[str, Any]) -> List[int]:
    tracks = [int(t["track_id"]) for t in (abi.get("tracks") or []) if "track_id" in t]
    if not tracks:
        raise SelfTestError("ABI 的 tracks[] 是空的，无法派生 observer 车道：{}".format(ABI_PATH))
    return sorted(tracks)


def abi_track_roles(abi: Dict[str, Any]) -> Dict[int, str]:
    return {int(t["track_id"]): str(t.get("role") or "track")
            for t in (abi.get("tracks") or []) if "track_id" in t}


def abi_anchor(abi: Dict[str, Any], tracks: Sequence[int]) -> Dict[str, Any]:
    anchor = ((abi.get("sync_anchors") or {}).get("entry_rendezvous") or {})
    task_id = anchor.get("task_id")
    if task_id is None:
        raise SelfTestError(
            "ABI 缺 sync_anchors.entry_rendezvous.task_id；SW/HW 对齐的软件零点就是这个锚点，"
            "没有它 align_sw_hw 只能退化成最早 start。请在 ABI 里补上（参考 "
            "device_record/abi/record_abi.example.json 的 sync_anchors 段）")
    return {"task_id": int(task_id), "task": anchor.get("task"),
            "track": int(anchor.get("track", tracks[0]))}


def _even_slots(start: int, end: int, count: int) -> List[Tuple[int, int]]:
    """把 [start, end] 均分成 count 段；最后一段的右端点严格等于 end
    （这样"最大 end"是可预测的，断言才能从常量算出来）。"""
    if count <= 0 or end <= start:
        return []
    width = (end - start) // count
    if width <= 0:
        raise SelfTestError("窗口 [{}, {}] 装不下 {} 个事件，请调大 OP_WINDOW_* "
                            "或减少 ABI 里的 task 数".format(start, end, count))
    slots = []
    for i in range(count):
        left = start + i * width
        right = end if i == count - 1 else start + (i + 1) * width
        slots.append((left, right))
    return slots


def _ev(block: int, track: int, track_role: str, entry: Dict[str, Any],
        start: int, end: int, depth: int) -> Dict[str, Any]:
    """一条解码后的事件，字段形状与 host_decode/decode_profile.py 的输出一致。
    注意 ``layer`` / ``parent_task_id`` 直接来自 ABI —— 骨架版 emit_perfetto
    没有 load_abi_task_map()，它读的就是事件自带的 layer。"""
    return {"block": block, "track": track, "role": track_role,
            "track_label": "{}{}".format(track_role, track),
            "task_id": int(entry["id"]), "task": entry.get("name"),
            "short_name": entry.get("short_name"), "task_role": entry.get("role"),
            "layer": entry.get("layer") or "task",
            "parent_task_id": entry.get("parent_task_id"),
            "expert_idx": SYNTH_PAYLOAD_VALUE,
            "start_ns": start, "end_ns": end, "dur_ns": end - start, "depth": depth}


def build_events_from_abi(abi: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按 ABI 的分层合成 NUM_CTAS 个 CTA 的事件：
      * 锚点 task 落在 ABI 声明的 track 上，占 [ANCHOR_START, ANCHOR_END]；
      * 其余 layer=="task" 的算子在每条 observer track 上铺满
        [OP_WINDOW_START, OP_WINDOW_END]（每条 track 轮转一下顺序，避免完全雷同）；
      * layer=="phase" 按 parent_task_id 嵌进父 task 内部（内缩 PHASE_PAD_NS）；
      * layer=="ignored" 也照实发出来 —— emit_perfetto 必须把它丢掉。
    """
    index = abi_task_index(abi)
    tracks = abi_track_ids(abi)
    roles = abi_track_roles(abi)
    anchor = abi_anchor(abi, tracks)
    anchor_entry = index[anchor["task_id"]]

    operator_ids = [tid for tid in abi_layer_ids(index, "task") if tid != anchor["task_id"]]
    if not operator_ids:
        operator_ids = abi_layer_ids(index, "task")
    ignored_ids = abi_layer_ids(index, "ignored")
    phases_by_parent: Dict[Optional[int], List[int]] = {}
    for tid in abi_layer_ids(index, "phase"):
        parent = index[tid].get("parent_task_id")
        parent_key = None if parent is None else int(parent)
        phases_by_parent.setdefault(parent_key, []).append(tid)

    events: List[Dict[str, Any]] = []
    for cta in range(NUM_CTAS):
        skew = cta * CTA_SKEW_NS
        anchor_track = anchor["track"] if anchor["track"] in tracks else tracks[0]
        events.append(_ev(cta, anchor_track, roles.get(anchor_track, "track"), anchor_entry,
                          ANCHOR_START_NS + skew, ANCHOR_END_NS + skew, 0))
        for track_pos, track in enumerate(tracks):
            role = roles.get(track, "track")
            rotated = operator_ids[track_pos % len(operator_ids):] + \
                operator_ids[:track_pos % len(operator_ids)]
            slots = _even_slots(OP_WINDOW_START_NS + skew, OP_WINDOW_END_NS + skew, len(rotated))
            for tid, (start, end) in zip(rotated, slots):
                events.append(_ev(cta, track, role, index[tid], start, end, 0))
                children = phases_by_parent.get(tid) or []
                inner = _even_slots(start + PHASE_PAD_NS, end - PHASE_PAD_NS, len(children))
                for child, (cstart, cend) in zip(children, inner):
                    events.append(_ev(cta, track, role, index[child], cstart, cend, 1))
        # layer=="ignored"（整 kernel 锚条 / grid barrier 之类）：整窗铺满，
        # 用来证明 emit_perfetto 真的把它从渲染里剔掉了。
        for tid in ignored_ids:
            events.append(_ev(cta, tracks[0], roles.get(tracks[0], "track"), index[tid],
                              OP_WINDOW_START_NS + skew, OP_WINDOW_END_NS + skew, 0))
    return events


def make_decoded(abi: Dict[str, Any]) -> Dict[str, Any]:
    tracks = abi_track_ids(abi)
    anchor = abi_anchor(abi, tracks)
    events = build_events_from_abi(abi)
    block_map = {str(cta): {"physical_smid": cta, "written": True} for cta in range(NUM_CTAS)}
    anchors = {"entry_rendezvous": {
        "task": anchor["task"], "task_id": anchor["task_id"],
        "ctas_with_anchor": NUM_CTAS, "ctas_missing_anchor_total": 0,
        "global": {"entry_sync_start_ns": ANCHOR_START_NS,
                   "entry_sync_end_ns": ANCHOR_END_NS,
                   "min_start_ns": ANCHOR_START_NS,
                   "max_start_ns": ANCHOR_START_NS + MAX_CTA_SKEW_NS,
                   "min_end_ns": ANCHOR_END_NS,
                   "max_end_ns": ANCHOR_END_NS + MAX_CTA_SKEW_NS,
                   "start_skew_ns": MAX_CTA_SKEW_NS,
                   "end_skew_ns": MAX_CTA_SKEW_NS,
                   "rendezvous_span_ns": ANCHOR_END_NS + MAX_CTA_SKEW_NS - ANCHOR_START_NS}}}
    return {"metadata": {"rank": 0, "device": HW_DEVICE}, "block_map": block_map,
            "anchors": anchors, "events": events}


def make_hw() -> Dict[str, Any]:
    # 占位符必须在**被用进 fixture 之前**炸掉（P2-1）：假设备名/假 kernel 名会让
    # 这份 fixture 与真实 nsys 产物脱钩，自测跑绿也证明不了任何事。
    device_name = require_configured(
        "HW_DEVICE_NAME", HW_DEVICE_NAME,
        "填 nsys 报告里的真实设备名，例 'NVIDIA H20 (Hopper)'")
    kernel_name = require_configured(
        "TARGET_KERNEL_NAME", TARGET_KERNEL_NAME,
        "填 nsys 里目标 kernel 的名字，须与 contract 的 run.target_kernel / "
        "capture.hardware.kernel_name_contains 对得上")
    counters = []
    for position, spec in enumerate(COUNTER_SPECS):
        samples = [[NSYS_START_NS + i * SAMPLE_STEP_NS,
                    float(spec["base"] + i * spec["step"])] for i in range(SAMPLE_COUNT)]
        if position == 0:
            # 故意插一个窗口外采样，证明 merge_hw 真的按 nsys 窗口裁剪。
            samples = samples + [[OUT_OF_WINDOW_SAMPLE_NS, 999.0]]
        counters.append({"metric_id": spec["metric_id"], "name": spec["name"],
                         "raw_name": spec["raw_name"], "unit": spec["unit"],
                         "sample_count": len(samples), "samples": samples})
    return {"schema": "gpu-hw-metrics/v1", "device": HW_DEVICE,
            "device_name": device_name, "metrics_set": HW_METRICS_SET,
            "frequency_hz": HW_FREQUENCY_HZ,
            "kernel_launches": [{"start_ns": NSYS_START_NS, "end_ns": NSYS_END_NS,
                                 "correlation_id": 42, "name": kernel_name,
                                 "device": HW_DEVICE}],
            "counters": counters}


# --------------------------------------------------------------------------- #
# trace_processor_shell
# --------------------------------------------------------------------------- #
def find_tps() -> Optional[str]:
    candidates: List[Optional[str]] = [os.environ.get("TRACE_PROCESSOR_SHELL")]
    candidates.extend(TRACE_PROCESSOR_SHELL_CANDIDATES)
    for candidate in candidates:
        # 占位符不是路径：别让它冒充"找过了"。
        if candidate and not unconfigured(candidate) and Path(candidate).is_file():
            return candidate
    return shutil.which("trace_processor_shell")


def tps_import(tps: str, trace_path: Path) -> Tuple[int, List[str], int, str]:
    forbidden_sql = ",\n  ".join("'{}'".format(name) for name in FORBIDDEN_IMPORT_STATS)
    sql = ("SELECT name\nFROM stats\nWHERE value > 0 AND name IN (\n  " + forbidden_sql
           + "\n)\nGROUP BY name;\nSELECT COUNT(*) AS imported_slices FROM slice;")
    proc = subprocess.run([str(tps), "-Q", sql, str(trace_path)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          timeout=TRACE_PROCESSOR_TIMEOUT_SEC)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    found = [name for name in FORBIDDEN_IMPORT_STATS if name in output]
    slices = -1
    if "imported_slices" in output:
        digits = re.findall(r"(\d+)", output.split("imported_slices", 1)[-1])
        slices = int(digits[0]) if digits else -1
    return proc.returncode, found, slices, output


# --------------------------------------------------------------------------- #
# 用例
# --------------------------------------------------------------------------- #
def case_alignment_shift_only(abi: Dict[str, Any], state: Dict[str, Any]) -> None:
    decoded = state["decoded"]
    hw = state["hw"]
    align = align_sw_hw.compute_alignment(
        {"metadata": decoded["metadata"], "events": decoded["events"],
         "anchors": decoded["anchors"]}, hw)
    state["align"] = align

    _require("schema == sw-hw-alignment/v1", align["schema"] == "sw-hw-alignment/v1", align.get("schema"))
    _require("scale == 1（shift-only，绝不缩放）", align["scale"] == 1, align.get("scale"))
    _require("shift_ns == entry rendezvous release ({})".format(EXPECTED_SHIFT_NS),
             align["shift_ns"] == EXPECTED_SHIFT_NS, align["shift_ns"])
    _require("shift_ns_source == entry_rendezvous.release",
             align["shift_ns_source"] == "entry_rendezvous.release", align["shift_ns_source"])
    _require("shift_ns_by_rank 按 rank 索引", align["shift_ns_by_rank"] == {"0": EXPECTED_SHIFT_NS},
             align["shift_ns_by_rank"])
    _require("sw_window_ns == {}".format(EXPECTED_SW_WINDOW),
             align["sw_window_ns"] == EXPECTED_SW_WINDOW, align["sw_window_ns"])
    _require("sw_span_ns == {}".format(EXPECTED_SW_SPAN),
             align["sw_span_ns"] == EXPECTED_SW_SPAN, align["sw_span_ns"])
    _require("nsys_window_ns == {}".format(EXPECTED_NSYS_WINDOW),
             align["nsys_window_ns"] == EXPECTED_NSYS_WINDOW, align["nsys_window_ns"])
    _require("nsys_span_ns == {}".format(EXPECTED_NSYS_SPAN),
             align["nsys_span_ns"] == EXPECTED_NSYS_SPAN, align["nsys_span_ns"])
    _require("span_mismatch_ns == {}（nsys>sw 是结构性的，不许拉伸）".format(EXPECTED_SPAN_MISMATCH),
             align["span_mismatch_ns"] == EXPECTED_SPAN_MISMATCH, align["span_mismatch_ns"])
    _require("span_mismatch_ns >= 0", align["span_mismatch_ns"] >= 0, align["span_mismatch_ns"])
    _require("chosen_launch ordinal/total == 0/1",
             align["chosen_launch"]["ordinal"] == 0
             and align["chosen_launch"]["total_matched_launches"] == 1,
             align["chosen_launch"])
    residual = align["cross_cta_residual"]
    _require("cross_cta_residual 带 CTA 间 skew（{} CTA, span {}）".format(
        residual.get("ctas_with_anchor"), residual.get("rendezvous_span_ns")),
        residual.get("ctas_with_anchor") == NUM_CTAS
        and residual.get("rendezvous_span_ns") == ANCHOR_END_NS + MAX_CTA_SKEW_NS - ANCHOR_START_NS,
        residual)
    _require("hw_map_formula 写明 shift-only 映射", align["hw_map_formula"] == HW_MAP_FORMULA,
             align.get("hw_map_formula"))


def case_no_anchor_fallback(abi: Dict[str, Any], state: Dict[str, Any]) -> None:
    decoded = state["decoded"]
    fallback = align_sw_hw.compute_alignment(
        {"metadata": {"rank": 0}, "events": decoded["events"], "anchors": {}}, None)
    _require("无锚点时退化成 earliest_software_start（shift={}）".format(fallback["shift_ns"]),
             fallback["shift_ns"] == EXPECTED_SW_WINDOW[0]
             and fallback["shift_ns_source"] == "earliest_software_start",
             (fallback["shift_ns"], fallback["shift_ns_source"]))
    _require("无 hw_metrics 时 nsys_window_ns 为 None（不许编窗口）",
             fallback.get("nsys_window_ns") is None, fallback.get("nsys_window_ns"))


def case_missing_launches_hard_fails(abi: Dict[str, Any], state: Dict[str, Any]) -> None:
    """hw_metrics 没有 kernel_launches 时必须硬失败 —— 不许"没量到就当对齐了"。"""
    broken = dict(state["hw"])
    broken["kernel_launches"] = []
    decoded = state["decoded"]
    try:
        align_sw_hw.compute_alignment(
            {"metadata": decoded["metadata"], "events": decoded["events"],
             "anchors": decoded["anchors"]}, broken)
    except SystemExit as error:
        _require("缺 kernel_launches -> SystemExit", "kernel_launches" in str(error), str(error))
        return
    raise SelfTestError("缺 kernel_launches 时 compute_alignment 居然返回了结果（必须硬失败）")


def case_missing_nsys_window_hard_fails(abi: Dict[str, Any], state: Dict[str, Any]) -> None:
    """alignment 里没有 nsys_window_ns 时 merge_hw 必须拒绝合并。"""
    trace = {"traceEvents": [{"ph": "X", "pid": emit_perfetto.PID_BASE, "tid": 1,
                              "ts": 0.0, "dur": 1.0, "name": "probe", "args": {}}]}
    bad_align = {"schema": "sw-hw-alignment/v1", "scale": 1, "shift_ns": 0,
                 "sw_window_ns": [0, 1], "nsys_window_ns": None}
    try:
        emit_perfetto.merge_hw(trace, state["hw"], bad_align)
    except emit_perfetto.EmitError as error:
        _require("缺 nsys_window_ns -> EmitError", "nsys_window_ns" in str(error), str(error))
        return
    raise SelfTestError("缺 nsys_window_ns 时 merge_hw 居然合并成功了（必须硬失败）")


def case_merge_hw_r6_and_counters(abi: Dict[str, Any], state: Dict[str, Any]) -> None:
    decoded = state["decoded"]
    trace, summary = emit_perfetto.build_perfetto(
        [(decoded, "events_rank0.json")], hw_metrics=state["hw"], alignment=state["align"])
    state["trace"] = trace
    state["summary"] = summary

    _require("软件侧没有部分重叠（渲染契约 R3）", summary["improper_overlaps"] == 0,
             summary.get("improper_overlap_samples"))
    _require("软件车道没有把物理 SM 当成一条 lane（R6 前置）",
             summary["physical_sm_on_software_lane"] == 0,
             summary["physical_sm_on_software_lane"])
    _require("每个 (rank, CTA) 一个 process，共 {} 个".format(NUM_CTAS),
             summary["num_processes"] == NUM_CTAS, summary["num_processes"])

    merged = summary["hw_merge"]
    doc = (trace.get("otherData") or {}).get("hw_merge") or {}
    _require("otherData.hw_merge.schema == hw-merge/v1", doc.get("schema") == "hw-merge/v1",
             doc.get("schema"))
    software_pids = doc.get("software_pids") or []
    _require("software_pids == {}".format([emit_perfetto.PID_BASE + i for i in range(NUM_CTAS)]),
             software_pids == [emit_perfetto.PID_BASE + i for i in range(NUM_CTAS)], software_pids)
    min_sw_pid = min(software_pids)
    hw_pid = merged["hw_counter_pid"]
    _require("R6: hw pid {} < 所有软件 pid（min {}）".format(hw_pid, min_sw_pid),
             hw_pid < min_sw_pid, (hw_pid, software_pids))
    _require("otherData.hw_merge.r6_hw_pid_lt_all_software is True",
             doc.get("r6_hw_pid_lt_all_software") is True, doc.get("r6_hw_pid_lt_all_software"))
    _require("summary.hw_merge.r6_ok is True", merged.get("r6_ok") is True, merged.get("r6_ok"))
    _require("hw pid == PID_BASE-1 == {}（没有发生 pid 平移）".format(emit_perfetto.PID_BASE - 1),
             hw_pid == emit_perfetto.PID_BASE - 1 and doc.get("software_pid_shift_applied") == 0,
             (hw_pid, doc.get("software_pid_shift_applied")))
    _require("alignment_mode 声明 shift-only",
             "shift-only" in str(doc.get("alignment_mode")), doc.get("alignment_mode"))
    _require("hw_merge 回抄 sw/nsys 窗口",
             doc.get("sw_window_ns") == EXPECTED_SW_WINDOW
             and doc.get("nsys_window_ns") == EXPECTED_NSYS_WINDOW,
             (doc.get("sw_window_ns"), doc.get("nsys_window_ns")))

    # HW 曲线必须排在所有软件 process 之前
    sort_indexes = {e["pid"]: e["args"]["sort_index"] for e in trace["traceEvents"]
                    if e.get("ph") == "M" and e.get("name") == "process_sort_index"}
    _require("HW process_sort_index == -1", sort_indexes.get(hw_pid) == -1, sort_indexes)
    _require("HW sort_index 严格小于所有软件 sort_index",
             all(sort_indexes[hw_pid] < sort_indexes[pid] for pid in software_pids),
             sort_indexes)

    counters = [e for e in trace["traceEvents"] if e.get("ph") == "C"]
    _require("窗口内 ph=C 事件数 == {}（{} 条曲线 x {} 点，窗口外的被裁掉）".format(
        EXPECTED_COUNTER_EVENTS, len(COUNTER_SPECS), SAMPLE_COUNT),
        len(counters) == EXPECTED_COUNTER_EVENTS, len(counters))
    _require("summary.hw_merge.counter_events 与实际一致",
             merged.get("counter_events") == EXPECTED_COUNTER_EVENTS, merged.get("counter_events"))

    per_counter = doc.get("counters") or []
    _require("每条曲线都有裁剪台账", len(per_counter) == len(COUNTER_SPECS), per_counter)
    clipped_name = COUNTER_SPECS[0]["name"]
    clipped = next((c for c in per_counter if c["name"] == clipped_name), None)
    _require("{} 窗口内保留 {} 点（多插的窗口外点被裁）".format(clipped_name, SAMPLE_COUNT),
             clipped is not None and clipped.get("samples_in_window") == SAMPLE_COUNT, clipped)

    _require("每个 ph=C args.value 都是有限数值（不是 bool/None/nan/inf）",
             all(isinstance(e["args"]["value"], (int, float))
                 and not isinstance(e["args"]["value"], bool)
                 and math.isfinite(float(e["args"]["value"])) for e in counters),
             [e["args"].get("value") for e in counters[:5]])
    _require("每个 ph=C ts >= 0（靠一次全局 rebase）",
             all(float(e["ts"]) >= 0.0 for e in counters),
             min(float(e["ts"]) for e in counters) if counters else None)
    _require("每个 ph=C 都挂在 device-wide 的 hw pid / tid 0 上",
             all(e["pid"] == hw_pid and e["tid"] == 0 for e in counters),
             sorted({(e["pid"], e["tid"]) for e in counters}))

    expected_names = sorted("{} [{}]".format(spec["name"], spec["unit"]) for spec in COUNTER_SPECS)
    _require("曲线名带真实指标名+单位: {}".format(expected_names),
             sorted({e["name"] for e in counters}) == expected_names,
             sorted({e["name"] for e in counters}))

    monotone = True
    for name in {e["name"] for e in counters}:
        series = [float(e["ts"]) for e in counters if e["name"] == name]
        if any(b < a for a, b in zip(series, series[1:])):
            monotone = False
    _require("同一条曲线的 ph=C 时间戳单调不减", monotone, "")


def case_timeline_rebase_preserves_spacing(abi: Dict[str, Any], state: Dict[str, Any]) -> None:
    trace = state["trace"]
    doc = (trace.get("otherData") or {}).get("hw_merge") or {}
    rebase = doc.get("timeline_rebase") or {}
    _require("timeline_rebase schema == timeline-rebase/v1",
             rebase.get("schema") == "timeline-rebase/v1", rebase.get("schema"))
    _require("rebase 只做一次全局平移，不 clamp/丢点/缩放",
             rebase.get("relative_timing_preserved") is True
             and rebase.get("duration_fields_unchanged") is True,
             rebase)
    _require("rebase 之后没有负时间戳", rebase.get("post_negative_count") == 0, rebase)
    _require("rebase 之后最小 ts == 0", float(rebase.get("post_min_ts_us")) == 0.0,
             rebase.get("post_min_ts_us"))

    # HW 首点映射：完全按 hw_map_formula 推，再叠上 rebase 的全局偏移。
    offset_us = float(rebase.get("offset_us") or 0.0)
    sw_start = EXPECTED_SW_WINDOW[0]
    expected_first = round((sw_start + 0 - EXPECTED_SHIFT_NS) / 1000.0 + offset_us, 9)
    first_name = "{} [{}]".format(COUNTER_SPECS[0]["name"], COUNTER_SPECS[0]["unit"])
    series = sorted(float(e["ts"]) for e in trace["traceEvents"]
                    if e.get("ph") == "C" and e.get("name") == first_name)
    _require("HW 首点 ts == hw_map_formula 推导值 {}".format(expected_first),
             series and series[0] == expected_first, series[:3])
    expected_span_us = (SAMPLE_COUNT - 1) * SAMPLE_STEP_NS / 1000.0
    _require("HW 采样间距原样保留（跨度 {}us）".format(expected_span_us),
             series and round(series[-1] - series[0], 9) == expected_span_us,
             (series[0], series[-1]) if series else None)


def case_ignored_layer_not_rendered(abi: Dict[str, Any], state: Dict[str, Any]) -> None:
    """layer=="ignored"（整 kernel 锚条 / grid barrier）不许出现在渲染里。
    这是覆盖率门禁 G19 的孪生坑：一根整窗宽的条会让人误以为算子铺满了。"""
    index = abi_task_index(abi)
    ignored_ids = abi_layer_ids(index, "ignored")
    spans = [e for e in state["trace"]["traceEvents"] if e.get("ph") == "X"]
    _require("确实产出了 ph=X 软件区间", bool(spans), len(spans))
    if not ignored_ids:
        _note("本 ABI 没有声明 layer==\"ignored\" 的 task，改用显式 layer 覆盖来验通用丢弃规则；"
              "接新 target 时如果你的 ABI 有整 kernel 锚条，请把它标成 ignored")
        probe = dict(state["decoded"])
        events = [dict(e) for e in probe["events"]]
        widest = dict(events[0])
        widest["layer"] = "ignored"
        widest["start_ns"] = OP_WINDOW_START_NS
        widest["end_ns"] = OP_WINDOW_END_NS
        widest["dur_ns"] = OP_WINDOW_END_NS - OP_WINDOW_START_NS
        widest["task"] = "__synthetic_ignored__"
        probe["events"] = events + [widest]
        trace, _summary = emit_perfetto.build_perfetto([(probe, "probe.json")])
        rendered = {e["args"].get("layer") for e in trace["traceEvents"] if e.get("ph") == "X"}
        _require("layer==ignored 的区间被丢弃", "ignored" not in rendered, sorted(map(str, rendered)))
        _require("同一批里 layer==task 的区间仍然渲染", "task" in rendered,
                 sorted(map(str, rendered)))
        return
    rendered_ids = {e["args"].get("task_id") for e in spans}
    _require("ABI 里 layer==ignored 的 task_id {} 全部不渲染".format(ignored_ids),
             not (set(ignored_ids) & rendered_ids), sorted(str(i) for i in rendered_ids))
    _require("ph=X 的 args.layer 里没有 ignored",
             "ignored" not in {e["args"].get("layer") for e in spans},
             sorted({str(e["args"].get("layer")) for e in spans}))


def case_trace_processor_import(abi: Dict[str, Any], state: Dict[str, Any]) -> None:
    """用真实 trace_processor_shell 导一遍。找不到 = FAIL（不是 skip）：
    ph='C' 写坏了只有真实导入器能抓到，跳过就等于放行。"""
    tps = find_tps()
    _require("trace_processor_shell 可用: {}".format(tps),
             bool(tps) and Path(str(tps)).is_file(), tps_hint())
    trace = state["trace"]
    other = trace.setdefault("otherData", {})
    if "provenance" not in other:
        # ===== 已知移植缺口 =====
        # 骨架 emit_perfetto.py 还没写 emit-provenance/v1（真实实现里由
        # build_perfetto(source_events=..., source_abi=...) 落盘，见
        # DSA-learn/.../deepseek_v4_pro/decode/profile/hw_perfetto/emit_perfetto.py）。
        # 交付门禁会查这个字段，这里先补一个占位让导入验证跑通，并把缺口喊出来。
        _note("[todo] emit_perfetto 未产出 otherData.provenance（emit-provenance/v1）；"
              "移植时请参考真实实现补上，交付门禁会查")
        other["provenance"] = {"schema": "emit-provenance/v1", "inputs": [],
                               "note": "placeholder injected by selftest_hw_merge"}
    with tempfile.TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "hw_merge.perfetto.json.gz"
        with gzip.open(str(trace_path), "wt", encoding="utf-8") as handle:
            json.dump(trace, handle, ensure_ascii=False)
        rc, found, slices, output = tps_import(str(tps), trace_path)
        _require("trace_processor 返回码 == 0", rc == 0, output[-400:])
        _require("没有触发任何禁止的导入统计项", not found, found)
        _require("trace_processor 导入了 > 0 个 slice", slices > 0, slices)
        _note("trace_processor_shell = {}".format(tps))
        _note("imported_slices = {}".format(slices))


CASES = (
    ("alignment_shift_only", case_alignment_shift_only),
    ("no_anchor_fallback", case_no_anchor_fallback),
    ("missing_launches_hard_fails", case_missing_launches_hard_fails),
    ("missing_nsys_window_hard_fails", case_missing_nsys_window_hard_fails),
    ("merge_hw_r6_and_counters", case_merge_hw_r6_and_counters),
    ("timeline_rebase_preserves_spacing", case_timeline_rebase_preserves_spacing),
    ("ignored_layer_not_rendered", case_ignored_layer_not_rendered),
    ("trace_processor_import", case_trace_processor_import),
)


def run() -> List[str]:
    if not ABI_PATH.is_file():
        raise SelfTestError("找不到 ABI：{}；先用项目的 extract_abi.py 生成".format(ABI_PATH))
    abi = load_abi(ABI_PATH)
    state: Dict[str, Any] = {"decoded": make_decoded(abi), "hw": make_hw()}
    passed: List[str] = []
    for name, fn in CASES:
        print("[case] {}".format(name))
        fn(abi, state)
        passed.append(name)
    return passed


def main() -> int:
    print("[selftest_hw_merge] profile root = {}".format(_PROFILE_ROOT))
    print("[selftest_hw_merge] abi          = {}".format(ABI_PATH))
    try:
        passed = run()
    except SelfTestError as error:
        print("SELFTEST=FAIL name=selftest_hw_merge reason={}".format(error), file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001 -- 如实报出任何意外失败
        print("SELFTEST=FAIL name=selftest_hw_merge unexpected={}: {}".format(
            type(error).__name__, error), file=sys.stderr)
        return 1
    print("SELFTEST=PASS name=selftest_hw_merge checks={} cases={}".format(
        _OK_LINES, ",".join(passed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
