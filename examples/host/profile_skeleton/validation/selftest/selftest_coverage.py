#!/usr/bin/env python3
"""selftest_coverage.py（骨架版）— E25 离线单测：锁死 check_gate_item.gate_g19 里的
"算子 task 覆盖率"子检查。

流程位置
  由 ``validation/run_offline_selftests.py`` 自动发现并当子进程跑。纯 CPU，不碰 GPU，
  不读任何运行产物 —— 事件、ABI、alignment 全在进程内伪造。退出码 0 == PASS。

它到底在防什么（历史坑，必须永久锁住）
  * 只有一根"整 kernel 宽"的条时**不能**算覆盖：那根条在 ABI 里是
    layer=="ignored"，曾经用 min/max 包络判覆盖率，被这根条撑满而假装 100% 通过；
  * 算子只集中在头部（或只在尾部）时必须 FAIL，且报出来的空洞要大于阈值
    （真实 DSV4-Pro decode 交付曾经有 769us 头洞 / 337us 尾洞）；
  * 20 个算子 task 铺满整个 kernel 窗口（只剩 grid_sync 的微小缝）时必须 PASS；
  * scale != 1 时该子检查不适用（空洞语义要求 shift-only 对齐；scale != 1 本身
    已经会让 G19 失败）。

接新 target 要改的地方
  1. ``ABI`` 里的 task/phase/ignored id 划分要换成**你的 ABI 的真实分层**
     （下面给的是"20..39 是算子 task、1/2 是 ignored、10 是 phase"这种形状）；
  2. ``WIN_START`` / ``SPAN`` 换成你 target 的量级即可，判定逻辑与数值无关。
  阈值不要在这里写死：直接从 ``check_gate_item.GATE_PROJECT`` 读，这样调阈值只有
  一个地方，不会出现"门禁放宽了但自测还按老阈值过"的假绿灯。

真实参考实现
  DSA-learn/mk/h20/cutedsl/deepseek_v4_pro/decode/profile/validation/selftest/selftest_coverage.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

_SELFTEST_DIR = Path(__file__).resolve().parent
_VALIDATION_DIR = _SELFTEST_DIR.parent
if str(_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_DIR))

import check_gate_item as cg  # noqa: E402

# ======================================================================
# ===== 接新 target 时只需要改这里 (PROJECT CONFIG) =====
# ======================================================================
WIN_START = 1_000_000_000     # kernel 窗口起点（任意值，只影响可读性）
SPAN = 2_005_760              # kernel 窗口跨度 ns（约 2ms，接近真实 decode 量级）

# 伪造 ABI：只有"算子 task"这一层算覆盖。
#   OPERATOR_TASK_IDS -> layer=="task"
#   IGNORED_IDS       -> layer=="ignored"（整 kernel 锚点 + grid barrier）
#   PHASE_IDS         -> layer=="phase"（算子内部子阶段）
OPERATOR_TASK_IDS = list(range(20, 40))
IGNORED_IDS = [1, 2]
PHASE_IDS = [10]
# ======================================================================
# ===== 配置区结束 =====
# ======================================================================

WIN_END = WIN_START + SPAN

ABI = {"tasks": ([{"id": i, "layer": "task"} for i in OPERATOR_TASK_IDS]
                 + [{"id": i, "layer": "ignored"} for i in IGNORED_IDS]
                 + [{"id": i, "layer": "phase"} for i in PHASE_IDS])}

ALIGN = {"scale": 1, "shift_ns": WIN_END, "sw_span_ns": SPAN,
         "sw_window_ns": [WIN_START, WIN_END]}


class SelfTestError(AssertionError):
    pass


class FakeGC:
    """最小替身，只暴露 _operator_task_coverage 真正用到的东西：
    ``.abi`` / ``.artifacts["sw_events"]`` / ``.present()`` / ``.doc()``。"""

    def __init__(self, events: List[Dict[str, Any]]):
        self.abi = ABI
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"events": events}, handle)
        handle.close()
        self.artifacts = {"sw_events": Path(handle.name)}

    def present(self, *keys: str) -> List[str]:
        return [k for k in keys if not self.artifacts[k].is_file()]

    def doc(self, key: str) -> Optional[Any]:
        path = self.artifacts.get(key)
        if path is None or not path.is_file():
            return None
        return cg.load_json_maybe_gz(path)


def _tile_operators(gap_ns: int = 2000) -> List[Dict[str, Any]]:
    """让所有算子 task 铺满窗口，彼此之间只留 gap_ns 的缝（模拟 grid_sync）。"""
    events: List[Dict[str, Any]] = []
    ids = list(OPERATOR_TASK_IDS)
    step = SPAN // len(ids)
    cursor = WIN_START
    for eid in ids:
        events.append({"event_id": eid, "start_ns": cursor, "end_ns": cursor + step - gap_ns})
        cursor += step
    # 补最后一小截，保证尾部没有大于阈值的残余空洞。
    events.append({"event_id": ids[-1], "start_ns": cursor - step, "end_ns": WIN_END})
    return events


def _cov(events: List[Dict[str, Any]], align: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return cg._operator_task_coverage(FakeGC(events), align if align is not None else ALIGN)


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise SelfTestError("{}: {}".format(name, detail))


def main() -> int:
    passed: List[str] = []
    try:
        # 0) 阈值必须来自 GATE_PROJECT，不是本文件写死的
        check("threshold_sourced_from_gate_config",
              int(cg.GATE_PROJECT["coverage_max_gap_percent"]) > 0
              and int(cg.GATE_PROJECT["coverage_max_gap_floor_ns"]) > 0,
              "GATE_PROJECT 缺少覆盖率阈值配置")
        passed.append("threshold_sourced_from_gate_config")

        # 1) 只有整 kernel 宽的 ignored 大条 -> 覆盖率 0，必须 FAIL
        cov = _cov([{"event_id": IGNORED_IDS[0], "start_ns": WIN_START, "end_ns": WIN_END}])
        check("kernel_bar_only", cov["applicable"] and cov["ok"] is False
              and cov["operator_task_span_events"] == 0,
              "ignored kernel bar must not count as coverage: {}".format(cov))
        passed.append("kernel_bar_only")

        # 2) 只有 phase 层的整宽条 -> 不是 task -> 覆盖率 0
        cov = _cov([{"event_id": PHASE_IDS[0], "start_ns": WIN_START, "end_ns": WIN_END}])
        check("phase_only", cov["ok"] is False and cov["operator_task_span_events"] == 0,
              "phase-layer span must not count: {}".format(cov))
        passed.append("phase_only")

        # 3) 头洞：算子只在后半段 -> FAIL，且空洞大于阈值
        cov = _cov([{"event_id": OPERATOR_TASK_IDS[1],
                     "start_ns": WIN_START + SPAN // 2, "end_ns": WIN_END}])
        check("head_hole", cov["ok"] is False
              and cov["max_uncovered_gap_us"] > cov["gap_threshold_us"],
              "head hole must FAIL: {}".format(cov))
        passed.append("head_hole")

        # 4) 尾洞：算子只在前半段 -> FAIL，且空洞大于阈值
        cov = _cov([{"event_id": OPERATOR_TASK_IDS[1],
                     "start_ns": WIN_START, "end_ns": WIN_START + SPAN // 2}])
        check("tail_hole", cov["ok"] is False
              and cov["max_uncovered_gap_us"] > cov["gap_threshold_us"],
              "tail hole must FAIL: {}".format(cov))
        passed.append("tail_hole")

        # 5) 铺满：算子 task 覆盖整窗（只剩 ~2us 的 grid 缝）-> PASS
        cov = _cov(_tile_operators())
        check("full_tiling", cov["applicable"] and cov["ok"] is True
              and cov["coverage_fraction"] > 0.95,
              "full tiling must PASS: {}".format(cov))
        passed.append("full_tiling")

        # 6) scale != 1 -> 子检查不适用（交给 G19 的 scale 主检查去判）
        align_scaled = dict(ALIGN)
        align_scaled["scale"] = 2
        cov = _cov(_tile_operators(), align=align_scaled)
        check("scale_not_one", cov["applicable"] is False,
              "scale != 1 must be non-applicable: {}".format(cov))
        passed.append("scale_not_one")

        # 7) 事件文件缺失 -> 必须 FAIL，不能当成"没发现空洞"
        gc_missing = FakeGC([])
        gc_missing.artifacts["sw_events"].unlink()
        cov = cg._operator_task_coverage(gc_missing, ALIGN)
        check("missing_events_fails", cov["applicable"] and cov["ok"] is False,
              "missing decoded events must FAIL, not silently pass: {}".format(cov))
        passed.append("missing_events_fails")
    except SelfTestError as error:
        print("SELFTEST=FAIL name=selftest_coverage reason={}".format(error), file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001
        print("SELFTEST=FAIL name=selftest_coverage unexpected={}: {}".format(
            type(error).__name__, error), file=sys.stderr)
        return 1
    print("SELFTEST=PASS name=selftest_coverage cases={}".format(",".join(passed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
