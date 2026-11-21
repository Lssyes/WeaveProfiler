#!/usr/bin/env python3
"""selftest_decode.py（骨架版）— E25 离线单测：验证 host_decode/decode_profile.py。

流程位置
  由 ``validation/run_offline_selftests.py`` 自动发现并当子进程跑。纯 CPU，不需要
  torch/CUDA，也不读任何运行产物 —— 整个 profiler buffer 都在进程内**按 ABI 合成**。
  退出码 0 == PASS。

它验证什么
  happy path（合成一份完全自洽的 buffer）
    * 配对闭合：records_total == 2 * events_total == records_expected；
    * 所有异常计数器为 0，diagnostic_summary.valid 为 True；
    * block_map 每个 block 一条，logical_cta -> physical_smid 与合成值一致；
    * entry_rendezvous 锚点覆盖每一个 CTA（ctas_missing_anchor_total == 0）；
    * 没有空 track、没有被截断的 track。

  negative paths（解码器必须**抓出来**并把这次解码标成 invalid）
    * buffer 写满            -> truncated_tracks 非空
    * 丢掉一条 end 记录      -> unmatched_starts > 0
    * 事件 id 不在 ABI tasks -> unknown_task_ids > 0
    * tag 里的 block 与槽位不符 -> tag_slot_mismatch > 0
    * timestamp==0 但 tag 非 0  -> zero_timestamp_with_nonzero_tag > 0

为什么 buffer 是"按 ABI 合成"而不是写死一段字节
  合成器只用 ABI 里的布局表达式（header_elems_expr / numel_formula / *_stride_expr /
  slot_formula）和 tag_bitfields 来算下标和打包。这样 ABI 一改（多一个 track、
  换 tag 位宽、加事件），自测的期望值**自动跟着变**；如果把记录数、事件号、
  下标写死，就会出现"ABI 已经变了但自测还过"的假绿灯 —— 那才是最危险的情况。

接新 target 要改的地方
  只有配置区：``SYNTH_NUM_SMS``（合成几个 block，小一点跑得快）、
  ``TAG_FIELD_NAMES``（tag 字段名，必须和 decode_profile.py 实际读的名字一致）、
  ``METADATA_FIELD_NAMES`` / ``BLOCK_MAP_FIELD_NAMES``。
  记录条数、事件表、锚点、strides 全部从 ABI 推，**不要**在这里写死。

真实参考实现
  DSA-learn/mk/h20/cutedsl/deepseek_v4_pro/decode/profile/validation/selftest/selftest_decode.py
  （真实项目是 separate_tensors 布局，用 dp.synthesize_buffer / dp.pack_tag /
   dp.pack_work_id / abi_loader.lane_geometry；骨架是单块扁平 buffer + sentinel，
   所以合成器写在本文件里，见下面的 synthesize()。）
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SELFTEST_DIR = Path(__file__).resolve().parent
_VALIDATION_DIR = _SELFTEST_DIR.parent
_PROFILE_ROOT = _VALIDATION_DIR.parent
for _p in (_PROFILE_ROOT / "host_decode", _PROFILE_ROOT / "device_record" / "abi"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import decode_profile as dp  # noqa: E402
from abi_loader import check_abi_provenance, eval_expr, load_abi  # noqa: E402

# ======================================================================
# ===== 接新 target 时只需要改这里 (PROJECT CONFIG) =====
# ======================================================================
# 1) ABI 路径（相对 profile/）。换成你项目 extract_abi.py 产出的那份。
ABI_PATH = _PROFILE_ROOT / "device_record" / "abi" / "record_abi.example.json"

# 2) 合成几个 block。只影响跑得快不快，判定与数值无关；小值足够覆盖所有分支。
SYNTH_NUM_SMS = 4

# 3) tag / metadata / block_map 的字段名。
#    这些名字必须与 decode_profile.py 里实际读的 key 一致（骨架解码器直接用
#    fields["block"] / fields["track"] / fields["task"] / fields["is_start"] /
#    fields["expert_idx"]）。换 ABI 时如果字段改名，改这里，同时改解码器。
TAG_FIELD_NAMES = {"block": "block", "track": "track", "task": "task",
                   "is_start": "is_start", "payload": "expert_idx"}
METADATA_FIELD_NAMES = {"num_sms": "num_sms", "num_tracks": "num_tracks"}
BLOCK_MAP_FIELD_NAMES = {"logical_cta": "logical_cta", "physical_smid": "physical_smid"}

# 4) 合成记录的时间戳基准与步长（ns）。必须 > 0：timestamp==0 是"未写入"哨兵。
SYNTH_TS_BASE = 1_000_000
SYNTH_TS_STEP = 1_000
# ======================================================================
# ===== 配置区结束；下面全部从 ABI 推导 =====
# ======================================================================


class SelfTestError(AssertionError):
    pass


def _require(name: str, condition: bool, got: Any = None) -> None:
    if not condition:
        raise SelfTestError("{} (got {!r})".format(name, got))


def pack_bits(spec_by_name: Dict[str, Dict[str, Any]], values: Dict[str, int]) -> int:
    """按 ABI 的 {lsb, mask} 把字段打进一个整数。未给的字段一律 0。"""
    word = 0
    for name, spec in spec_by_name.items():
        value = int(values.get(name, 0)) & int(spec["mask"])
        word |= value << int(spec["lsb"])
    return word


def abi_event_plan(abi: Dict[str, Any]) -> List[Tuple[int, int]]:
    """从 ABI 推出一条 (task_id, is_start) 序列：task 层嵌套它的 phase 层。

    顺序保证是合法嵌套：open(task) [open(phase) close(phase)]* close(task)。
    锚点 task 无论 layer 是什么都会被纳入，否则 entry_rendezvous 断言无从验证。
    """
    tasks = [t for t in abi.get("tasks", []) if isinstance(t, dict) and "id" in t]
    by_id = {int(t["id"]): t for t in tasks}
    task_ids = sorted(int(t["id"]) for t in tasks if t.get("layer") == "task")
    anchor = abi.get("sync_anchors", {}).get("entry_rendezvous") or {}
    anchor_id = anchor.get("task_id")
    if anchor_id is not None and int(anchor_id) in by_id and int(anchor_id) not in task_ids:
        task_ids.append(int(anchor_id))          # layer 被标成 ignored 也要记
    if not task_ids:
        raise SelfTestError(
            "ABI 里没有 layer=='task' 的事件，也没有 sync_anchors.entry_rendezvous.task_id；"
            "无法合成任何事件。请先把 ABI 的 tasks[] 填成真实分层（参考 "
            "knowledge/examples/host/extract_abi.py 的产出）")
    phases_of: Dict[int, List[int]] = {}
    for t in tasks:
        if t.get("layer") == "phase" and t.get("parent_task_id") is not None:
            phases_of.setdefault(int(t["parent_task_id"]), []).append(int(t["id"]))
    plan: List[Tuple[int, int]] = []
    for tid in sorted(task_ids):
        plan.append((tid, 1))
        for pid in sorted(phases_of.get(tid, [])):
            plan.append((pid, 1))
            plan.append((pid, 0))
        plan.append((tid, 0))
    return plan


class Layout:
    """把 ABI 的布局表达式解出来，供合成器算元素下标。"""

    def __init__(self, abi: Dict[str, Any], num_sms: int, max_records: int):
        buf = abi["buffer"]
        self.abi = abi
        self.num_sms = int(num_sms)
        self.num_tracks = int(buf["num_tracks"])
        self.max_records = int(max_records)
        self.elems_per_record = int(buf["elems_per_record"])
        env = {"num_sms": self.num_sms, "num_tracks": self.num_tracks,
               "max_records": self.max_records}
        self.env = env
        self.header_elems = eval_expr(buf["header_elems_expr"], env)
        self.numel = eval_expr(buf["numel_formula"], env)
        layout = abi["regions"]["records"]["layout"]
        self.block_stride = eval_expr(layout["block_stride_expr"], env)
        self.track_stride = eval_expr(layout["track_stride_expr"], env)
        self.record_stride = eval_expr(layout["record_stride_expr"], env)
        fields = layout["fields"]
        self.field_index = {f["name"]: int(f["index"]) for f in fields}

    def index(self, block: int, track: int, offset: int, field: str) -> int:
        return (self.header_elems + block * self.block_stride + track * self.track_stride
                + offset * self.record_stride + self.field_index[field])

    def block_map_index(self, block: int) -> int:
        expr = self.abi["regions"]["block_map"]["elem_index_expr"]
        env = dict(self.env)
        env["block"] = block
        return eval_expr(expr, env)


def synthesize(abi: Dict[str, Any], num_sms: int = SYNTH_NUM_SMS,
               extra_slots: int = 2) -> Tuple[List[int], Layout, Dict[str, Any]]:
    """按 ABI 合成一份自洽的扁平 buffer。

    ``extra_slots == 0`` 时 max_records 刚好等于实际记录数，于是每个 track 都会被
    判成 truncated —— 这正是"buffer 写满"的负例，不需要另造一份数据。
    """
    plan = abi_event_plan(abi)
    records_needed = len(plan)
    max_records = records_needed + int(extra_slots)
    lay = Layout(abi, num_sms, max_records)

    ts_field = "timestamp_ns" if "timestamp_ns" in lay.field_index else \
        sorted(lay.field_index, key=lambda n: lay.field_index[n])[0]
    tag_field = "tag" if "tag" in lay.field_index else \
        sorted(lay.field_index, key=lambda n: lay.field_index[n])[-1]

    values = [0] * lay.numel

    # --- metadata：num_sms / num_tracks 按 regions.metadata.fields 打包 ---
    meta_specs = {f["name"]: f for f in abi["regions"]["metadata"]["fields"]}
    values[int(abi["regions"]["metadata"]["elem_index"])] = pack_bits(
        meta_specs, {METADATA_FIELD_NAMES["num_sms"]: lay.num_sms,
                     METADATA_FIELD_NAMES["num_tracks"]: lay.num_tracks})

    # --- block_map：logical_cta -> physical_smid（+1 保证 block 0 的字不是 0）---
    bm_specs = {f["name"]: f for f in abi["regions"]["block_map"]["fields"]}
    expected_block_map: List[Dict[str, Any]] = []
    for block in range(lay.num_sms):
        logical, physical = block, block + 1
        values[lay.block_map_index(block)] = pack_bits(
            bm_specs, {BLOCK_MAP_FIELD_NAMES["logical_cta"]: logical,
                       BLOCK_MAP_FIELD_NAMES["physical_smid"]: physical})
        expected_block_map.append({"logical_cta": logical, "physical_smid": physical,
                                   "written": True})

    # --- records：每个 (block, track) 走同一条嵌套计划，时间戳严格递增 ---
    tag_specs = abi["tag_bitfields"]
    names = TAG_FIELD_NAMES
    for block in range(lay.num_sms):
        for track in range(lay.num_tracks):
            ts = SYNTH_TS_BASE + (block * lay.num_tracks + track) * SYNTH_TS_STEP * 1000
            for offset, (task_id, is_start) in enumerate(plan):
                ts += SYNTH_TS_STEP
                values[lay.index(block, track, offset, ts_field)] = ts
                values[lay.index(block, track, offset, tag_field)] = pack_bits(
                    tag_specs, {names["block"]: block, names["track"]: track,
                                names["task"]: task_id, names["is_start"]: is_start,
                                names["payload"]: 0})

    events_per_track = records_needed // 2
    expected = {
        "plan": plan,
        "records_needed": records_needed,
        "max_records": max_records,
        "events_total": lay.num_sms * lay.num_tracks * events_per_track,
        "records_total": lay.num_sms * lay.num_tracks * records_needed,
        "block_map": expected_block_map,
        "ts_field": ts_field, "tag_field": tag_field,
    }
    return values, lay, expected


def free_task_id(abi: Dict[str, Any]) -> Optional[int]:
    """在 tag 的 task 字段位宽内找一个 ABI 没用过的 id（用来造 unknown 事件）。"""
    mask = int(abi["tag_bitfields"][TAG_FIELD_NAMES["task"]]["mask"])
    used = {int(t["id"]) for t in abi.get("tasks", []) if isinstance(t, dict) and "id" in t}
    for candidate in range(mask, -1, -1):
        if candidate not in used:
            return candidate
    return None


def decode(values: List[int], abi: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    """跑真实解码器。故意不传 num_sms/num_tracks/max_records —— 让解码器自己从
    buffer 的 metadata 头和 numel 里推出来，顺便把那条路径也测掉。"""
    options: Dict[str, Any] = {"abi_path": ABI_PATH, "rank": 0}
    options.update(extra)
    return dp.decode_buffer(values, abi, options)


def _counters(decoded: Dict[str, Any]) -> Dict[str, int]:
    return decoded["metadata"]["counters"]


# ----------------------------------------------------------------------
# 测试用例
# ----------------------------------------------------------------------
def test_happy_path(abi: Dict[str, Any]) -> None:
    values, lay, expected = synthesize(abi)
    decoded = decode(values, abi)
    meta = decoded["metadata"]
    _require("num_sms read from metadata header", meta["num_sms"] == lay.num_sms, meta["num_sms"])
    _require("max_records inferred from numel", meta["max_records"] == expected["max_records"],
             meta["max_records"])
    _require("events_total", meta["events_total"] == expected["events_total"],
             (meta["events_total"], expected["events_total"]))
    _require("records_total==2*events", meta["records_total"] == 2 * meta["events_total"],
             (meta["records_total"], meta["events_total"]))
    _require("records_expected==records_total",
             meta["records_expected"] == meta["records_total"], meta["records_expected"])
    zero = {k: v for k, v in _counters(decoded).items() if v}
    _require("all anomaly counters zero", not zero, zero)
    _require("valid", meta["diagnostic_summary"]["valid"] is True,
             meta["diagnostic_summary"]["failures"])
    _require("no empty tracks", meta["empty_tracks"] == [], meta["empty_tracks"])
    _require("no truncated tracks", meta["truncated_tracks"] == [], meta["truncated_tracks"])
    _require("block_map", decoded["block_map"] == expected["block_map"], decoded["block_map"])
    anchor = decoded["anchors"].get("entry_rendezvous")
    _require("anchor present", anchor is not None,
             "ABI sync_anchors.entry_rendezvous 缺失或 task 名与 tasks[].name 不一致")
    _require("ctas_with_anchor", anchor["ctas_with_anchor"] == lay.num_sms,
             anchor["ctas_with_anchor"])
    _require("ctas_missing_anchor==0", anchor["ctas_missing_anchor_total"] == 0,
             anchor["ctas_missing_anchor_total"])


def test_truncation_detected(abi: Dict[str, Any]) -> None:
    # max_records 刚好等于记录数 -> 每个 track 都写满 -> 必须报截断且判 invalid。
    values, _lay, _expected = synthesize(abi, extra_slots=0)
    decoded = decode(values, abi)
    meta = decoded["metadata"]
    _require("truncated tracks reported", len(meta["truncated_tracks"]) >= 1,
             meta["truncated_tracks"])
    _require("truncation -> invalid", meta["diagnostic_summary"]["valid"] is False,
             meta["diagnostic_summary"])


def test_unpaired_detected(abi: Dict[str, Any]) -> None:
    # 抹掉 (0,0) 的最后一条记录（最外层 task 的 end）-> 它的 begin 变成未配对。
    values, lay, expected = synthesize(abi)
    last = expected["records_needed"] - 1
    values[lay.index(0, 0, last, expected["ts_field"])] = 0
    values[lay.index(0, 0, last, expected["tag_field"])] = 0
    decoded = decode(values, abi)
    meta = decoded["metadata"]
    counters = _counters(decoded)
    _require("unmatched_starts > 0", counters.get("unmatched_starts", 0) >= 1, counters)
    _require("unpaired -> invalid", meta["diagnostic_summary"]["valid"] is False,
             meta["diagnostic_summary"])


def test_unknown_event_detected(abi: Dict[str, Any]) -> None:
    bogus = free_task_id(abi)
    _require("a free task id exists inside the tag field", bogus is not None,
             "ABI 用满了 tag.task 字段的全部取值，无法构造 unknown 事件。"
             "请加宽 tag_bitfields['{}'] 的位宽，或在此改用别的负例。"
             .format(TAG_FIELD_NAMES["task"]))
    values, lay, expected = synthesize(abi)
    plan = expected["plan"]
    # 把最外层 task 的 begin/end 一起改成不存在的 id：这样只触发 unknown，
    # 不会顺带产生 unmatched（断言更精准）。
    first_id = plan[0][0]
    names = TAG_FIELD_NAMES
    for offset, (task_id, is_start) in enumerate(plan):
        if task_id != first_id:
            continue
        values[lay.index(0, 0, offset, expected["tag_field"])] = pack_bits(
            abi["tag_bitfields"], {names["block"]: 0, names["track"]: 0,
                                   names["task"]: bogus, names["is_start"]: is_start,
                                   names["payload"]: 0})
    decoded = decode(values, abi)
    meta = decoded["metadata"]
    counters = _counters(decoded)
    _require("unknown_task_ids > 0", counters.get("unknown_task_ids", 0) >= 1, counters)
    _require("unknown -> invalid", meta["diagnostic_summary"]["valid"] is False,
             meta["diagnostic_summary"])


def test_tag_slot_mismatch_detected(abi: Dict[str, Any]) -> None:
    # tag 里的 block 与它所在的槽位不符 -> 说明写入地址算错了，必须抓出来。
    values, lay, expected = synthesize(abi)
    names = TAG_FIELD_NAMES
    task_id, is_start = expected["plan"][0]
    values[lay.index(0, 0, 0, expected["tag_field"])] = pack_bits(
        abi["tag_bitfields"], {names["block"]: 1, names["track"]: 0,
                               names["task"]: task_id, names["is_start"]: is_start,
                               names["payload"]: 0})
    decoded = decode(values, abi)
    counters = _counters(decoded)
    _require("tag_slot_mismatch > 0", counters.get("tag_slot_mismatch", 0) >= 1, counters)
    _require("mismatch -> invalid", decoded["metadata"]["diagnostic_summary"]["valid"] is False,
             decoded["metadata"]["diagnostic_summary"])


def test_zero_timestamp_detected(abi: Dict[str, Any]) -> None:
    # timestamp==0 但 tag 非 0：既不是合法记录也不是"未写入"哨兵，必须报出来。
    values, lay, expected = synthesize(abi)
    values[lay.index(0, 0, 0, expected["ts_field"])] = 0
    decoded = decode(values, abi)
    counters = _counters(decoded)
    _require("zero_timestamp_with_nonzero_tag > 0",
             counters.get("zero_timestamp_with_nonzero_tag", 0) >= 1, counters)
    _require("zero-ts -> invalid", decoded["metadata"]["diagnostic_summary"]["valid"] is False,
             decoded["metadata"]["diagnostic_summary"])


def run() -> List[str]:
    abi = load_abi(ABI_PATH)
    # 先把 ABI 溯源状态说清楚：source_sha256 与磁盘源码不一致（mismatch）时
    # 解码器会把它升级成硬失败，happy path 必然过不了 —— 那不是自测的错，
    # 是 ABI 过期了。这里提前报出来，省得去查错方向。
    provenance = check_abi_provenance(abi, ABI_PATH)
    stale = {role: entry for role, entry in provenance.items()
             if entry.get("status") == "mismatch"}
    if stale:
        raise SelfTestError(
            "ABI 溯源不一致（source_sha256 与磁盘源码不符）：{}；"
            "请用项目的 extract_abi.py 重新生成 ABI 后再跑自测".format(stale))

    tests = [
        ("happy_path", test_happy_path),
        ("truncation_detected", test_truncation_detected),
        ("unpaired_detected", test_unpaired_detected),
        ("unknown_event_detected", test_unknown_event_detected),
        ("tag_slot_mismatch_detected", test_tag_slot_mismatch_detected),
        ("zero_timestamp_detected", test_zero_timestamp_detected),
    ]
    passed: List[str] = []
    for name, fn in tests:
        fn(abi)
        passed.append(name)
    return passed


def main() -> int:
    try:
        passed = run()
    except SelfTestError as error:
        print("SELFTEST=FAIL name=selftest_decode reason={}".format(error), file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001 -- 如实报出任何意外失败
        print("SELFTEST=FAIL name=selftest_decode unexpected={}: {}".format(
            type(error).__name__, error), file=sys.stderr)
        return 1
    print("SELFTEST=PASS name=selftest_decode cases={}".format(",".join(passed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
