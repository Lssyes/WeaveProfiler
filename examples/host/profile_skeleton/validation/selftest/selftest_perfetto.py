#!/usr/bin/env python3
"""selftest_perfetto.py（骨架版）— E25 离线单测：锁死出图器
``hw_perfetto/emit_perfetto.py`` 的渲染契约。

流程位置
  由 ``validation/run_offline_selftests.py`` 自动发现并当子进程跑
  (``python3 -B selftest_perfetto.py``，无 argparse，退出码 0 == PASS)。
  纯 CPU：不需要 GPU/nsys。它会在临时目录里造一个**完整的 run 目录**
  （CONTRACT.lock.json + sw/events_rank*.json + hw/hw_metrics.json +
  work/alignment.json），然后走**固定 CLI**
  ``emit_perfetto.py --contract <lock> --out <gz>``，再把产物重新打开逐条验：

  1. 契约解析路径本身（events_glob / hw_metrics / alignment 都从 contract 读）；
  2. 交付门禁关心的 Perfetto 形状（displayTimeUnit、producer、ph=X/ph=C 结构、
     R6 计数器置顶、layer==ignored 必须被丢弃）；
  3. 用 **skill 自己的** ``weaveprof/pipeline/scan/layout_contract.py`` 的 ``verify()``
     跑 V1..V6 —— 交付门禁加载的就是这个校验器；
  4. 用**真实的** trace_processor_shell 导一遍（坏 JSON / 交叉区间 / 负时间戳
     只有真实导入器能抓到）；
  5. 10MB 超限接缝：超限时必须么"隐藏所有 phase 后重生成"，要么**硬拒写**，
     绝不允许静默截断事件；
  6. 同一车道上的**部分重叠**（既不相离也不嵌套）必须 rc=3 且不落盘。

为什么事件表要从 ABI 派生（重要，别改回写死）
  事件编号/分层写死在自测里，ABI 一改（比如把某个 phase 提升成 task、
  或者新增算子 id）自测**照样能过** —— 那这条门禁就废了。所以本文件的
  task/phase/ignored 划分、锚点 task、observer track 全部由
  ``selftest_hw_merge.build_events_from_abi`` 从 ABI 现读现算；
  ``allowed_task_ids`` 也是从 ABI 的 ``layer=="task"`` 派生的。

接新 target 要改的地方
  1. 配置区的 ``ABI_PATH``、``RUN_LAYOUT``（产物相对路径约定）；
  2. ``PHYSICAL_WARPS_PER_CTA``：你 kernel 每个 CTA 的物理 warp 数。
     只有当它 > observer 车道数时（Hopper MMA warpgroup / Blackwell epilogue
     warpgroup 这种"一条 observer 代表一组 warp"的情况）才需要顺手填
     ``OBSERVER_REPRESENTATION``，否则留空；
  3. ``SKILL_ROOT_CANDIDATES``（或 ``FAST_INSTRUMENT_SKILL_ROOT`` 环境变量）：
     **必填**。定位不到 skill 就等于 V1..V6 没验，本自测会直接 FAIL；
  4. ``KNOWN_LAYOUT_GAPS`` / ``KNOWN_EMITTER_GAPS`` / ``OPTIONAL_DELIVERY_KEYS``：
     骨架出图器还缺的东西清单。**移植完一项就从这里删一项** —— 删掉之后对应的
     检查立刻从 [todo] 升级成硬 FAIL；
  5. trace_processor_shell 路径与禁止统计项复用 ``selftest_hw_merge.py``
     的配置区（只改一处）。
  fixture 的窗口/采样常量也在 ``selftest_hw_merge.py`` 配置区，同样只改一处。

退出码语义（**没验到 != 验过了**）
  * ``0``  = ``SELFTEST=PASS``：所有 case 都真的验过，一条 todo 都没有。
  * ``3``  = ``SELFTEST=PASS_WITH_TODOS``：断言都没失败，但存在**已登记**的骨架
    缺口，相关 case 被降级为 ``UNVERIFIED``（不计入 passed）。非零退出是故意的：
    "没验全"绝不允许显示成干净的 PASS。
  * ``1``  = ``SELFTEST=FAIL``：断言失败，或**外部依赖缺失**
    （``trace_processor_shell`` 找不到、skill 根目录定位失败、配置区还留着
    ``PROJECT_HOOK_REPLACE`` 占位符）。

  两类缺口必须分开对待，这是本文件最重要的一条设计：
  **外部工具/依赖缺失 -> 硬 FAIL**（``_require_external``），因为那说明环境没配好，
  跳过它就等于放行；**已登记的骨架功能缺口 -> ``[todo]`` + case 降级**
  （``_todo``，未登记的 tag 一律抛异常）。历史教训：曾经有一轮 25 个门禁全绿、
  实际覆盖率只有 44.9%，根因就是"找不到依赖 -> 跳过 -> 仍计入 passed"。

参考实现（dsv4pro/decode，已通过 25 门禁 —— 这是**范本路径**，不是本项目路径）
  DSA-learn/mk/h20/cutedsl/deepseek_v4_pro/decode/profile/validation/selftest/selftest_perfetto.py
"""
from __future__ import annotations

import contextlib
import gzip
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve()
_SELFTEST_DIR = _HERE.parent
_PROFILE_ROOT = _HERE.parents[2]
for _extra in (str(_PROFILE_ROOT / "hw_perfetto"),
               str(_PROFILE_ROOT / "device_record" / "abi"),
               str(_SELFTEST_DIR)):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

import align_sw_hw  # noqa: E402
import emit_perfetto  # noqa: E402
from abi_loader import load_abi  # noqa: E402

# fixture 单一来源：ABI 派生的事件表 / HW 采样 / trace_processor 定位与禁止统计项
# 都在 selftest_hw_merge.py 的配置区，避免两份自测各写一套常量后互相漂移。
import selftest_hw_merge as fixtures  # noqa: E402

# ======================================================================
# ===== 接新 target 时只需要改这里 (PROJECT CONFIG) =====
# ======================================================================
# 1) ABI：事件分层与 allowed_task_ids 的唯一来源。
ABI_PATH = fixtures.ABI_PATH

# 2) run 目录里的产物相对路径约定（与 check_gate_item.GATE_PROJECT
#    ["artifact_relpaths"] 必须一致，否则门禁读不到出图器写的东西）。
RUN_LAYOUT: Dict[str, str] = {
    "events_glob": "sw/events_rank*.json",          # 解码事件（每 rank 一个）
    "events_template": "sw/events_rank{rank}.json",  # 写 fixture 用
    "hw_metrics": "hw/hw_metrics.json",
    "alignment": "work/alignment.json",
    "perfetto": "work/candidate.perfetto.json.gz",
    "contract": "CONTRACT.lock.json",
}
MAX_TRACE_BYTES = emit_perfetto.DEFAULT_MAX_BYTES   # 契约 artifacts 上限：10485760

# 3) warp 拓扑：observer 车道数从 ABI 的 tracks[] 派生；物理 warp 数是 runtime
#    launch 属性，ABI 里没有，只能在这里声明。None == "与 observer 车道数相同"
#    （没有代表例外）。只有 physical > observers 时才需要填 OBSERVER_REPRESENTATION，
#    且 skill 的 layout_contract.validate_contract 只认下面两种 policy。
PHYSICAL_WARPS_PER_CTA: Optional[int] = None
OBSERVER_REPRESENTATION: Tuple[Dict[str, Any], ...] = (
    # 例：{"policy": "hopper_mma_warpgroup", "architecture": "hopper",
    #      "role": "mma warpgroup", "warpgroup": True, "represents": [4, 7]}
)

# 4) skill 侧 layout_contract.py 的位置（校验器与出图器共用的布局契约）。
#    **必填项**：V1..V6 用的就是这个校验器，定位不到 == 这一轮没验 == 硬 FAIL。
#    第一个候选靠 parents[5] 相对定位，只在 profile_skeleton **还住在 skill 里**
#    （examples/host/...）时成立；profile/ 一旦被拷进真实项目，层级就变了，
#    必须靠环境变量或下面那条绝对路径兜底。
SKILL_ROOT_ENV = "FAST_INSTRUMENT_SKILL_ROOT"


def _relative_skill_root_guess() -> Tuple[Path, ...]:
    """`parents[5]` 这条相对定位**层级不够深时会直接 IndexError**。

    例如 profile/ 被拷到 `/tmp/x/validation/selftest` 下，`_HERE.parents` 只有 5 层，
    模块导入期就抛裸 `IndexError: 5`：既印不出 require_skill_layout_checker() 的
    howto，也让"没定位到 skill 校验器"看起来像"脚本坏了"。所以取不到就返回空元组，
    后面照样硬 FAIL —— 只是 FAIL 得能读懂、能照着改。
    """
    parents = _HERE.parents
    if len(parents) <= 5:
        return ()
    return (parents[5],)


_RELATIVE_SKILL_ROOT_AVAILABLE = bool(_relative_skill_root_guess())
SKILL_ROOT_CANDIDATES: Tuple[Path, ...] = _relative_skill_root_guess() + (
    # ===== PROJECT HOOK: 拷进项目后填 skill 的绝对路径 =====
    # 占位符**不会**被当成路径试探（见 skill_root_candidates()），留着不填就等于
    # 只剩 parents[5] 一条相对定位。
    Path("PROJECT_HOOK_REPLACE/.agents/skills/fast-instrument-v2"),
)

# 5) 骨架出图器**已知还缺**的东西 —— 三张登记表。登记过的缺口打 [todo] 并把所在
#    case 降级为 UNVERIFIED（不计入 passed，且让本自测非零退出）；**没登记**的
#    缺口一律硬 FAIL。移植完一项就从表里删一项，对应检查立刻升级成硬检查。
#
#    5a) skill layout_contract.verify() 的失败标签（V1..V6 报出来的 [tag]）。
#        - task_source_measurement_missing: ph=X 的 args 里缺 is_derived=false（V4）
KNOWN_LAYOUT_GAPS: Tuple[str, ...] = (
    "task_source_measurement_missing",
)
#    5b) 出图器**行为**缺口（不是 V 检查，是 emit_perfetto 自己该做没做的事）。
KNOWN_EMITTER_GAPS: Tuple[str, ...] = (
    # write_trace_gz 是「先写后量」，超限时会留下残留 .gz（该改成临时文件 + os.replace）
    "write_trace_gz_non_atomic",
    # 超限只做到「拒写」，缺「隐藏所有 phase 后重生成」分支
    "oversize_hide_all_phases_missing",
    # best_effort 契约被 skill schema 强制放宽的检查项（见 SCHEMA_RELAXED_CHECKS）
    "layout_contract_relaxed_by_skill_schema",
)
#    5c) 交付门禁会查、但出图器还没写的 otherData 键。
#        gap tag 由 key 派生：otherdata_<key>_missing。
OPTIONAL_DELIVERY_KEYS: Tuple[Tuple[str, str, str], ...] = (
    ("layout_contract", "perfetto-layout-contract/v1",
     "把布局契约写进 otherData.layout_contract + 首条 ph=M "
     "perfetto_layout_contract 事件（参考 skill weaveprof/pipeline/scan/layout_contract.py "
     "的 metadata_event()）"),
    ("provenance", "emit-provenance/v1",
     "把 contract/events/abi/hw 的路径+sha256 写进 otherData.provenance"),
)

# 6) best_effort_layout_contract() 里**被迫放宽**的 V 检查项。
#    skill 的 layout_contract.validate_contract() 硬要求这两项为 true
#    （不是 true 就 SystemExit），所以骨架**没法**改成 False —— 只能如实登记成
#    todo 并在文案里写清放宽了什么、真实项目该怎么收紧。
SCHEMA_RELAXED_CHECKS: Tuple[Tuple[str, str], ...] = (
    ("allow_unparented_phase", "没有父 task 的孤儿 phase 区间不会被拦"),
    ("allow_empty_warp_lanes", "完全没有区间的空 warp 泳道不会被拦"),
)
# ======================================================================
# ===== 配置区结束 =====
# ======================================================================

# 参考实现指路（dsv4pro/decode，已通过 25 门禁）。这是**范本路径**，不是本项目
# 路径 —— 移植时照着它改本项目的文件，不要去改它。
_REF_ROOT = "DSA-learn/mk/h20/cutedsl/deepseek_v4_pro/decode/profile"
_REF_EMITTER = "参考实现（dsv4pro/decode，已通过 25 门禁）{}/hw_perfetto/emit_perfetto.py".format(
    _REF_ROOT)

FORBIDDEN_IMPORT_STATS = fixtures.FORBIDDEN_IMPORT_STATS


class SelfTestError(AssertionError):
    pass


# 退出码：0 = 全验过；3 = 断言没失败但没验全；1 = FAIL / 外部依赖缺失。
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_PASS_WITH_TODOS = 3

_OK_LINES = 0
_TODOS: List[Tuple[str, str, str]] = []      # (case, gap_tag, message)
_CURRENT_CASE = "<setup>"


def registered_gaps() -> frozenset:
    """三张登记表的并集。只有登记过的缺口才允许降级成 [todo]。"""
    return (frozenset(KNOWN_LAYOUT_GAPS) | frozenset(KNOWN_EMITTER_GAPS)
            | frozenset("otherdata_{}_missing".format(key)
                        for key, _schema, _how in OPTIONAL_DELIVERY_KEYS))


def _require(what: str, cond: Any, detail: Any = "") -> None:
    global _OK_LINES
    if cond:
        _OK_LINES += 1
        print("  [ok]   {}".format(what))
        return
    raise SelfTestError("{} (detail={})".format(what, detail))


def _require_external(what: str, cond: Any, howto: str) -> None:
    """**外部工具/外部依赖**缺失 -> 硬 FAIL 并给出可执行的修复步骤。

    绝不允许降级成 [todo] 或 skip：环境没配好时跳过检查，等于把门禁变成摆设。
    参考 run_offline_selftests.py 对缺 TRACE_PROCESSOR_SHELL 的处理（rc=1）。
    """
    global _OK_LINES
    if cond:
        _OK_LINES += 1
        print("  [ok]   {}".format(what))
        return
    raise SelfTestError("外部依赖缺失 -> {}；怎么修：{}".format(what, howto))


def _note(msg: str) -> None:
    print("  [note] {}".format(msg))


def _todo(tag: str, msg: str) -> None:
    """**已登记**的骨架功能缺口：允许继续跑，但当前 case 会被降级为 UNVERIFIED
    （不计入 passed），并让 main() 以 EXIT_PASS_WITH_TODOS 非零退出。

    未登记的 tag 一律硬 FAIL —— "找不到东西就跳过、但仍然算通过"是本门链里
    最严重的一类 bug，比检查本身写错更严重。
    """
    if tag not in registered_gaps():
        raise SelfTestError(
            "未登记的缺口 [{}]：{}。要么修好它，要么显式登记进配置区的 "
            "KNOWN_LAYOUT_GAPS / KNOWN_EMITTER_GAPS / OPTIONAL_DELIVERY_KEYS —— "
            "不允许悄悄跳过".format(tag, msg))
    _TODOS.append((_CURRENT_CASE, tag, msg))
    print("  [todo] [{}] {}".format(tag, msg))


# --------------------------------------------------------------------------- #
# run 目录 / CLI 驱动
# --------------------------------------------------------------------------- #
def write_run_dir(base: Path, decoded_list: List[Dict[str, Any]],
                  hw: Optional[Dict[str, Any]] = None,
                  alignment: Optional[Dict[str, Any]] = None,
                  max_bytes: int = MAX_TRACE_BYTES) -> Tuple[Path, Path]:
    """铺一个最小但形状真实的 run 目录，并返回 (contract_path, out_path)。
    契约里全部用**相对路径**，顺带验证 emit_perfetto 的相对路径解析。"""
    for decoded in decoded_list:
        rank = (decoded.get("metadata") or {}).get("rank", 0)
        target = base / RUN_LAYOUT["events_template"].format(rank=rank)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(decoded), encoding="utf-8")

    artifacts: Dict[str, Any] = {"events_glob": RUN_LAYOUT["events_glob"],
                                 "perfetto": RUN_LAYOUT["perfetto"],
                                 "max_bytes": max_bytes}
    if hw is not None:
        hw_path = base / RUN_LAYOUT["hw_metrics"]
        hw_path.parent.mkdir(parents=True, exist_ok=True)
        hw_path.write_text(json.dumps(hw), encoding="utf-8")
        artifacts["hw_metrics"] = RUN_LAYOUT["hw_metrics"]
    if alignment is not None:
        align_path = base / RUN_LAYOUT["alignment"]
        align_path.parent.mkdir(parents=True, exist_ok=True)
        align_path.write_text(json.dumps(alignment), encoding="utf-8")
        artifacts["alignment"] = RUN_LAYOUT["alignment"]

    contract = {"schema": "fast-instrument.contract-lock/v1",
                "project_layout": {"run_root": str(base)},
                "profile_selection": {"abi": str(ABI_PATH)},
                "artifacts": artifacts}
    contract_path = base / RUN_LAYOUT["contract"]
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    (base / RUN_LAYOUT["perfetto"]).parent.mkdir(parents=True, exist_ok=True)
    return contract_path, base / RUN_LAYOUT["perfetto"]


def run_emit(argv: List[str]) -> Tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = emit_perfetto.main(argv)
    return rc, out.getvalue(), err.getvalue()


def load_trace(path: Path) -> Dict[str, Any]:
    with gzip.open(str(path), "rt", encoding="utf-8") as handle:
        return json.load(handle)


def layer_of(event: Dict[str, Any]) -> str:
    args = event.get("args") or {}
    if args.get("layer") in ("task", "phase", "ignored"):
        return str(args["layer"])
    return str(event.get("cat", "")).split("/", 1)[0]


# --------------------------------------------------------------------------- #
# ABI 派生的期望值（不写死任何数字）
# --------------------------------------------------------------------------- #
def fixture_expectations(abi: Dict[str, Any]) -> Dict[str, Any]:
    events = fixtures.build_events_from_abi(abi)
    index = fixtures.abi_task_index(abi)
    rendered = [e for e in events if (e.get("layer") or "task") != "ignored"]
    return {
        "x_spans": len(rendered),
        "task_spans": sum(1 for e in rendered if e.get("layer") == "task"),
        "phase_spans": sum(1 for e in rendered if e.get("layer") == "phase"),
        "ignored_spans": len(events) - len(rendered),
        "allowed_task_ids": fixtures.abi_layer_ids(index, "task"),
        "observer_lanes": len(fixtures.abi_track_ids(abi)),
        "counter_events": fixtures.EXPECTED_COUNTER_EVENTS,
    }


def make_crossing_decoded(abi: Dict[str, Any]) -> Dict[str, Any]:
    """同一车道 (block 0, track 0) 上两个**交叉**区间：A=[1000,5000]、
    B=[3000,7000]。B 在 A 内部打开、在 A 之后关闭 —— 既不相离也不嵌套，
    违反渲染契约 R3/V5，出图器必须抓住并拒写。
    两个 task id 也从 ABI 取（避开 layer=="ignored"，否则会被直接丢掉）。"""
    index = fixtures.abi_task_index(abi)
    candidates = (fixtures.abi_layer_ids(index, "task")
                  + fixtures.abi_layer_ids(index, "phase"))
    if len(candidates) < 2:
        raise SelfTestError("ABI 里非 ignored 的 task 少于 2 个，无法构造交叉用例：{}".format(ABI_PATH))
    tracks = fixtures.abi_track_ids(abi)
    role = fixtures.abi_track_roles(abi).get(tracks[0], "track")
    first = fixtures._ev(0, tracks[0], role, index[candidates[0]], 1000, 5000, 0)
    second = fixtures._ev(0, tracks[0], role, index[candidates[1]], 3000, 7000, 0)
    return {"metadata": {"rank": 0, "device": fixtures.HW_DEVICE},
            "block_map": {"0": {"physical_smid": 0, "written": True}},
            "anchors": {}, "events": [first, second]}


# --------------------------------------------------------------------------- #
# skill 侧 layout_contract.py
# --------------------------------------------------------------------------- #
def skill_root_candidates() -> List[Path]:
    """按优先级列出所有**已配置**的 skill 根候选。

    配置区里还是 PROJECT_HOOK_REPLACE 占位符的条目直接跳过：占位符不是路径，
    让它参与试探只会把"这一项根本没填"伪装成"找过了但没找到"。
    """
    roots: List[Path] = []
    env_root = os.environ.get(SKILL_ROOT_ENV)
    if env_root:
        roots.append(Path(env_root))
    for root in SKILL_ROOT_CANDIDATES:
        if fixtures.unconfigured(root):
            continue
        roots.append(Path(root))
    return roots


def load_skill_layout_checker():
    for root in skill_root_candidates():
        candidate = Path(root) / "1_detect" / "layout_contract.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "fast_instrument_layout_contract_selftest", str(candidate))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module, candidate
    return None, None


def require_skill_layout_checker():
    """定位 skill 的校验器，**找不到就 FAIL**（不是 skip、不是 todo）。

    这里原来是 `_todo(...) + return`：V1..V6 六项断言整轮不跑，case 却照样计入
    passed。而候选里第一条靠 `parents[5]` 相对定位 —— profile/ 被拷进真实项目后
    层级一变就失效，第二条又是 PROJECT_HOOK_REPLACE 假路径，所以那条静默通道
    **必然**会触发。skill 校验器属于外部依赖，缺了只能 fail-closed。
    """
    checker, path = load_skill_layout_checker()
    tried = [str(Path(root) / "1_detect" / "layout_contract.py")
             for root in skill_root_candidates()]
    howto = ("export {}=<fast-instrument-v2 skill 根目录绝对路径>，"
             "或把绝对路径填进本文件配置区的 SKILL_ROOT_CANDIDATES。"
             "已按顺序试过：{}".format(SKILL_ROOT_ENV, tried or "(无任何已配置候选)"))
    pending = [str(root) for root in SKILL_ROOT_CANDIDATES if fixtures.unconfigured(root)]
    if pending:
        howto += "；配置区仍有未填的 {} 占位符（已跳过）：{}".format(
            fixtures.PROJECT_HOOK_SENTINEL, pending)
    if not _RELATIVE_SKILL_ROOT_AVAILABLE:
        howto += ("；注意本次安装路径 {} 层级不足 6 层，parents[5] 那条相对定位"
                  "**整条不可用**（不是没找到，是没法算），只剩环境变量/绝对路径两条"
                  .format(_HERE))
    _require_external(
        "skill 的 weaveprof/pipeline/scan/layout_contract.py 可加载（V1..V6 校验器）",
        checker is not None, howto)
    return checker, path


def best_effort_layout_contract(abi: Dict[str, Any], exp: Dict[str, Any],
                                has_counters: bool, events_path: str) -> Dict[str, Any]:
    """出图器还没把布局契约嵌进 trace 时，按**出图器实际写出的布局**现搭一份，
    好让 skill 的 verify() 现在就能跑起来（不是猜，是照着 build_sw_trace 的
    process_name / thread_name 模板填的）。出图器一旦自己嵌了契约，就以它为准。

    注意 SCHEMA_RELAXED_CHECKS 里那两项：``allow_unparented_phase`` /
    ``allow_empty_warp_lanes`` 被置为 true **不是骨架想放宽**，而是 skill 的
    ``layout_contract.validate_contract()`` 硬要求它们为 true（不是 true 就
    SystemExit）。骨架没法收紧，所以调用方必须把它登记成 todo，让"这两项没验"
    在结果里可见 —— 真实项目要在项目侧另外断言 phase 必须有父 task、observer
    泳道必须非空。
    """
    observers = exp["observer_lanes"]
    physical = PHYSICAL_WARPS_PER_CTA if PHYSICAL_WARPS_PER_CTA is not None else observers
    contract: Dict[str, Any] = {
        "schema": "perfetto-layout-contract/v1",
        "group_by": "rank_block",
        "group_name_format": "rank+CTA {value}",
        "group_sort_index_base": 0,
        "lane_by": ["warp_id", "track"],
        "lane_name_format": "warp{warp_id} (track{track})",
        "cta_field": "block",
        "warp_field": "warp_id",
        "physical_warps_per_cta": int(physical),
        "observer_lanes_per_cta": int(observers),
        "outer_layer": "task",
        "inner_layer": "phase",
        "layer_from_cat_prefix": True,
        "counter_must_be_topmost": bool(has_counters),
        "source_events": events_path,
        "source_abi": str(ABI_PATH),
        "allowed_task_ids": list(exp["allowed_task_ids"]),
        "allowed_task_role": None,
        "task_source_measured_only": True,
        "task_required_phase_kind": None,
        "slice_overlap_policy": "disjoint_or_nested",
        "observer_spans_source_only": True,
    }
    # 被 skill contract schema 强制放宽的项 —— 单一来源在 SCHEMA_RELAXED_CHECKS，
    # 调用方据此打 todo，避免"放宽了但没人知道"。
    for key, _what in SCHEMA_RELAXED_CHECKS:
        contract[key] = True
    if OBSERVER_REPRESENTATION:
        contract["observer_representation"] = [dict(item) for item in OBSERVER_REPRESENTATION]
    return contract


def report_schema_relaxations(contract: Dict[str, Any]) -> None:
    """把 best_effort 契约里放宽掉的 V 检查如实登记成 todo。"""
    relaxed = [(key, what) for key, what in SCHEMA_RELAXED_CHECKS if contract.get(key) is True]
    if not relaxed:
        return
    _todo("layout_contract_relaxed_by_skill_schema",
          "现搭的布局契约里 {} 被置为 true —— **此处放宽了这些检查**：{}。"
          "原因：skill 的 layout_contract.validate_contract() 硬要求这两项为 true"
          "（改成 false 会 SystemExit），骨架无法收紧。**真实项目必须收紧**："
          "让 emit_perfetto 自己嵌 perfetto-layout-contract/v1 并改走 trace 自带契约，"
          "同时在项目侧补断言「每个 phase 都能解析到父 task」「每条 observer 泳道都非空」。"
          .format([key for key, _w in relaxed], [what for _k, what in relaxed]))


def gap_tag(failure: str) -> str:
    """从 verify() 的失败串里抠出 [xxx] 标签，用来跟 KNOWN_LAYOUT_GAPS 对账。"""
    if "[" in failure and "]" in failure:
        return failure.split("[", 1)[1].split("]", 1)[0]
    return failure.split(":", 1)[0].strip()


# --------------------------------------------------------------------------- #
# 用例
# --------------------------------------------------------------------------- #
def case_cli_contract_path(abi: Dict[str, Any], state: Dict[str, Any]) -> None:
    """走固定 CLI（--contract/--out），输入全部由契约解析出来。"""
    base = state["tmp"] / "run_full"
    decoded = state["decoded"]
    alignment = align_sw_hw.compute_alignment(
        {"metadata": decoded["metadata"], "events": decoded["events"],
         "anchors": decoded["anchors"]}, state["hw"])
    contract_path, out_path = write_run_dir(base, [decoded], hw=state["hw"],
                                            alignment=alignment)
    rc, out_text, err_text = run_emit(["--contract", str(contract_path),
                                       "--out", str(out_path)])
    _require("emit rc == 0", rc == 0, (err_text or out_text)[-400:])
    _require("产物落盘: {}".format(out_path.name), out_path.is_file(), str(out_path))
    size = out_path.stat().st_size
    _require("产物大小 {} 在 {} 上限内".format(size, MAX_TRACE_BYTES),
             0 < size <= MAX_TRACE_BYTES, size)

    trace = load_trace(out_path)
    other = trace.get("otherData") or {}
    sources = [str(s) for s in (other.get("sources") or [])]
    _require("otherData.sources 指向契约解析出来的 events 路径（证明 events_glob 生效）",
             any(str(base) in s and s.endswith(".json") for s in sources), sources)
    _require("otherData.hw_merge 存在（证明契约里的 hw_metrics 生效）",
             isinstance(other.get("hw_merge"), dict), sorted(other.keys()))
    _require("shift_ns_by_rank 与 alignment 一致",
             other.get("shift_ns_by_rank") == {"0": alignment["shift_ns"]},
             other.get("shift_ns_by_rank"))
    state["trace"] = trace
    state["out_path"] = out_path
    state["contract_path"] = contract_path
    state["size_full"] = size
    state["run_root"] = base
    state["events_path"] = sources[0] if sources else ""


def case_delivery_schema(abi: Dict[str, Any], state: Dict[str, Any]) -> None:
    trace = state["trace"]
    exp = state["exp"]
    events = trace["traceEvents"]
    other = trace.get("otherData") or {}

    _require("traceEvents 是非空数组", isinstance(events, list) and events, type(events).__name__)
    _require("displayTimeUnit == 'ns'", trace.get("displayTimeUnit") == "ns",
             trace.get("displayTimeUnit"))
    producer = other.get("producer")
    emitter_name = Path(str(emit_perfetto.__file__)).name
    _require("otherData.producer 诚实指向真实出图器 {}".format(emitter_name),
             isinstance(producer, str) and Path(producer).name == emitter_name, producer)

    x_events = [e for e in events if e.get("ph") == "X"]
    c_events = [e for e in events if e.get("ph") == "C"]
    _require("ph=X 区间数 == {}（ABI 派生）".format(exp["x_spans"]),
             len(x_events) == exp["x_spans"], len(x_events))
    _require("task 层区间数 == {}（ABI 派生）".format(exp["task_spans"]),
             sum(1 for e in x_events if layer_of(e) == "task") == exp["task_spans"],
             sum(1 for e in x_events if layer_of(e) == "task"))
    _require("phase 层区间数 == {}（ABI 派生）".format(exp["phase_spans"]),
             sum(1 for e in x_events if layer_of(e) == "phase") == exp["phase_spans"],
             sum(1 for e in x_events if layer_of(e) == "phase"))
    _require("layer==ignored 一个都没渲染（禁止整 kernel 宽的假条）",
             not any(layer_of(e) == "ignored" for e in x_events),
             sorted({layer_of(e) for e in x_events}))
    _require("task 层的 task_id 全部落在 ABI 的 layer==task 集合 {} 内".format(
        exp["allowed_task_ids"]),
        {int((e.get("args") or {}).get("task_id")) for e in x_events
         if layer_of(e) == "task"}.issubset(set(exp["allowed_task_ids"])),
        sorted({(e.get("args") or {}).get("task_id") for e in x_events
                if layer_of(e) == "task"}))
    _require("每个 ph=X 都带 cat='<layer>/<role>' 前缀",
             all("/" in str(e.get("cat", "")) for e in x_events),
             sorted({str(e.get("cat")) for e in x_events})[:4])
    _require("每个 ph=X 都带 start_ns_raw/end_ns_raw（V5 用原始 ns 判重叠）",
             all(isinstance((e.get("args") or {}).get("start_ns_raw"), int)
                 and isinstance((e.get("args") or {}).get("end_ns_raw"), int)
                 for e in x_events), "")

    _require("ph=C 数 == {}".format(exp["counter_events"]),
             len(c_events) == exp["counter_events"], len(c_events))

    def _c_ok(event: Dict[str, Any]) -> bool:
        args = event.get("args")
        if not isinstance(args, dict) or not args:
            return False
        for value in args.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if not math.isfinite(float(value)):
                return False
        ts = event.get("ts")
        return (isinstance(ts, (int, float)) and not isinstance(ts, bool)
                and math.isfinite(float(ts)))

    _require("每个 ph=C 的 args 非空、全是有限数值（不是 bool/None/nan/inf），ts 也有限",
             all(_c_ok(e) for e in c_events), [e.get("args") for e in c_events[:3]])

    x_pids = {e["pid"] for e in x_events}
    c_pids = {e["pid"] for e in c_events}
    _require("R6: HW 计数器 pid {} 严格小于所有软件 pid {}".format(sorted(c_pids), sorted(x_pids)),
             x_pids and c_pids and max(c_pids) < min(x_pids), (sorted(c_pids), sorted(x_pids)))
    pname = {e["pid"]: (e.get("args") or {}).get("name") for e in events
             if e.get("name") == "process_name"}
    psort = {e["pid"]: (e.get("args") or {}).get("sort_index") for e in events
             if e.get("name") == "process_sort_index"}
    hw_pid = min(c_pids)
    _require("HW pid 有 device-wide 的 process_name", "HW" in str(pname.get(hw_pid)),
             pname.get(hw_pid))
    _require("HW process_sort_index == -1 且置顶",
             psort.get(hw_pid) == -1
             and psort[hw_pid] < min(psort[p] for p in x_pids if p in psort), psort)
    _require("每条计数器轨道名都带单位",
             all("[" in n and "]" in n for n in {str(e.get("name")) for e in c_events}),
             sorted({str(e.get("name")) for e in c_events}))
    _require("每个软件 pid 都有 'rank+CTA ...' 形式的 process_name",
             all(str(pname.get(p, "")).startswith("rank+CTA ") for p in x_pids),
             {p: pname.get(p) for p in sorted(x_pids)})
    _require("每条有 X 的泳道都有 thread_name",
             not [(e["pid"], e["tid"]) for e in x_events
                  if (e["pid"], e["tid"]) not in
                  {(m["pid"], m["tid"]) for m in events if m.get("name") == "thread_name"}],
             "")

    # 交付门禁会查、但骨架出图器还没写的字段：登记过的才允许 [todo]（并把本 case
    # 降级为 UNVERIFIED），实现后从 OPTIONAL_DELIVERY_KEYS 删掉即升级为硬检查。
    for key, schema, how in OPTIONAL_DELIVERY_KEYS:
        node = other.get(key)
        if isinstance(node, dict) and node.get("schema") == schema:
            _require("otherData.{} 是 {}".format(key, schema), True, "")
        elif node is None:
            _todo("otherdata_{}_missing".format(key),
                  "emit_perfetto 未产出 otherData.{}（{}）；{}；{}".format(
                      key, schema, how, _REF_EMITTER))
        else:
            raise SelfTestError("otherData.{} 存在但 schema 不对：{}".format(key, node))

    if not any((e.get("args") or {}).get("is_derived") is False
               for e in x_events if layer_of(e) == "task"):
        # tag 不在 KNOWN_LAYOUT_GAPS 里时，_todo 会直接抛 SelfTestError（硬 FAIL）。
        _todo("task_source_measurement_missing",
              "ph=X 的 task 区间缺 args.is_derived=false（skill layout_contract 的 V4 会拒）；"
              "移植时在 emit_perfetto.build_sw_trace 的 args 里补 "
              "is_derived=False（实测区间）/True+aggregation（派生父区间）；{}".format(_REF_EMITTER))
    else:
        _require("每个 task 区间都显式带 args.is_derived is False（V4）",
                 all((e.get("args") or {}).get("is_derived") is False
                     for e in x_events if layer_of(e) == "task"), "")


def case_skill_layout_verify(abi: Dict[str, Any], state: Dict[str, Any]) -> None:
    """用 skill 自己的 weaveprof/pipeline/scan/layout_contract.py 跑 V1..V6。

    定位不到校验器 = 硬 FAIL（见 require_skill_layout_checker）。以前这里会
    `_todo` 一句然后 `return`，六项断言整轮不跑却仍计入 passed —— 那正是本门链
    最危险的静默通过通道。"""
    checker, path = require_skill_layout_checker()
    _note("skill layout_contract.py = {}".format(path))
    trace = state["trace"]
    embedded = checker.contract_from_trace(trace)
    if embedded is not None:
        _require("出图器自己嵌了 perfetto-layout-contract/v1",
                 embedded.get("schema") == checker.SCHEMA, embedded.get("schema"))
        fails = checker.verify(trace, embedded)
        _require("skill verify() 零失败（用 trace 自带的契约）", fails == [], fails[:4])
        return

    _todo("otherdata_layout_contract_missing",
          "出图器没把布局契约嵌进 trace（otherData.layout_contract + 首条 ph=M "
          "perfetto_layout_contract 事件都缺）；下面改用按出图器实际布局现搭的契约来跑 "
          "V1..V6，因此「出图器自证布局」这一环没验；{}".format(_REF_EMITTER))
    contract = best_effort_layout_contract(
        abi, state["exp"], has_counters=True, events_path=state["events_path"])
    report_schema_relaxations(contract)
    checker.validate_contract(contract)   # 契约自身必须自洽，否则是配置区填错了
    _require("现搭的布局契约本身通过 validate_contract", True, "")
    fails = checker.verify(trace, contract)
    unexpected = [f for f in fails if gap_tag(f) not in KNOWN_LAYOUT_GAPS]
    grouped: Dict[str, List[str]] = {}
    for failure in fails:
        tag = gap_tag(failure)
        if tag in KNOWN_LAYOUT_GAPS:
            grouped.setdefault(tag, []).append(failure)
    for tag in sorted(grouped):
        hits = grouped[tag]
        _todo(tag, "已登记移植缺口命中 {} 处，样例：{}".format(len(hits), hits[0][:150]))
    _require("skill verify() 除 KNOWN_LAYOUT_GAPS {} 之外零失败".format(list(KNOWN_LAYOUT_GAPS)),
             not unexpected, unexpected[:4])


def case_trace_processor_import(abi: Dict[str, Any], state: Dict[str, Any]) -> None:
    """真实 trace_processor_shell 导入。找不到 = FAIL（不是 skip）。"""
    tps = fixtures.find_tps()
    _require_external("trace_processor_shell 可用: {}".format(tps),
                      bool(tps) and Path(str(tps)).is_file(), fixtures.tps_hint())
    state["tps"] = tps
    rc, found, slices, output = fixtures.tps_import(str(tps), state["out_path"])
    _require("trace_processor 返回码 == 0", rc == 0, output[-400:])
    _require("没有触发任何禁止的导入统计项", not found, found)
    _require("导入了 > 0 个 slice", slices > 0, slices)
    _require("导入的 slice 数 == ph=X 区间数 {}".format(state["exp"]["x_spans"]),
             slices == state["exp"]["x_spans"], slices)
    _note("imported_slices = {}".format(slices))


def case_oversize_seam(abi: Dict[str, Any], state: Dict[str, Any]) -> None:
    """把上限压到"比刚才的产物小 1 字节"，逼出超限分支。
    合格行为只有两种：(a) 隐藏所有 phase 后重生成（保留全部 task/选中 SM/HW 曲线），
    (b) 硬拒写。绝不允许静默截断事件后还落盘。"""
    base = state["tmp"] / "run_oversize"
    cap = state["size_full"] - 1
    contract_path, out_path = write_run_dir(
        base, [state["decoded"]], hw=state["hw"], max_bytes=cap)
    rc, out_text, err_text = run_emit(["--contract", str(contract_path),
                                       "--out", str(out_path)])
    combined = (out_text or "") + (err_text or "")
    if rc == 0:
        # 出图器实现了重生成分支 -> 立刻按完整语义硬验。
        _require("重生成产物落盘且不超上限（{} <= {}）".format(
            out_path.stat().st_size if out_path.is_file() else -1, cap),
            out_path.is_file() and out_path.stat().st_size <= cap, cap)
        regen = load_trace(out_path)
        other = regen.get("otherData") or {}
        viewer = other.get("semantic_viewer") or {}
        _require("otherData.semantic_viewer.mode == 'hide_all_phases'",
                 viewer.get("mode") == "hide_all_phases", viewer)
        rx = [e for e in regen["traceEvents"] if e.get("ph") == "X"]
        _require("重生成后不留 phase 层区间",
                 all(layer_of(e) != "phase" for e in rx),
                 sorted({layer_of(e) for e in rx}))
        _require("重生成后 task 层区间一个不少（{}）".format(state["exp"]["task_spans"]),
                 sum(1 for e in rx if layer_of(e) == "task") == state["exp"]["task_spans"],
                 sum(1 for e in rx if layer_of(e) == "task"))
        _require("重生成后 HW 曲线一条不少（{}）".format(state["exp"]["counter_events"]),
                 sum(1 for e in regen["traceEvents"] if e.get("ph") == "C")
                 == state["exp"]["counter_events"], "")
        tps = state.get("tps")
        if tps and Path(str(tps)).is_file():
            rc_i, found, slices, output = fixtures.tps_import(str(tps), out_path)
            _require("重生成产物也能被 trace_processor 干净导入",
                     rc_i == 0 and not found and slices > 0, (rc_i, found, slices))
        return
    _require("超限时 rc == 4（拒写）", rc == 4, (rc, combined[-300:]))
    _require("超限原因写进了 stderr",
             "cap" in combined.lower() or "bytes" in combined.lower(), combined[-300:])
    # 真正要守住的性质：绝不能留下一个"看起来合法"的产物（即已被静默截断到上限内的
    # 文件）。骨架 write_trace_gz 是"先写后量"，所以会留下一个**明显超限**的残留文件；
    # 那不会被误当交付物（门禁的尺寸检查会拦），但仍然该改成"先写临时文件再 rename"。
    if out_path.is_file():
        leftover = out_path.stat().st_size
        _require("残留文件明显超限（{} > {}），不可能被误当合法交付物".format(leftover, cap),
                 leftover > cap, leftover)
        _todo("write_trace_gz_non_atomic",
              "emit_perfetto.write_trace_gz 是「先写后量」，超限时会留下残留 .gz；"
              "移植时改成写临时文件 + os.replace()，或在抛 EmitError 前 unlink，"
              "这样阶段失败后 run 目录里不会有半成品；{}".format(_REF_EMITTER))
    else:
        _require("超限时不落盘（fail-closed，绝不静默截断）", True, "")
    _todo("oversize_hide_all_phases_missing",
          "emit_perfetto 超限只做到「拒写」，还缺「隐藏所有 phase 后重生成」分支"
          "（保留全部 task 区间 / 选中 SM 的 CTA 泳道 / HW 曲线，并写 "
          "otherData.semantic_viewer.mode='hide_all_phases'）；"
          "预检 E22 会一直 FAIL 直到补上；{}".format(_REF_EMITTER))


def case_partial_overlap_rejected(abi: Dict[str, Any], state: Dict[str, Any]) -> None:
    crossing = make_crossing_decoded(abi)
    _trace, summary = emit_perfetto.build_perfetto([(crossing, "events_rank0.json")])
    _require("build_perfetto 数出交叉区间（improper_overlaps={}）".format(
        summary["improper_overlaps"]), summary["improper_overlaps"] >= 1,
        summary.get("improper_overlap_samples"))

    base = state["tmp"] / "run_crossing"
    contract_path, out_path = write_run_dir(base, [crossing], hw=None)
    rc, out_text, err_text = run_emit(["--contract", str(contract_path),
                                       "--out", str(out_path)])
    _require("部分重叠时 rc == 3", rc == 3, (rc, (err_text or out_text)[-300:]))
    _require("部分重叠时不落盘（fail-closed）", not out_path.is_file(), str(out_path))
    _require("stderr 说明违反的是 R3「只允许相离或嵌套」",
             "overlap" in (out_text + err_text).lower(), (out_text + err_text)[-300:])

    # V5「交叉重叠必须被报出来」是这条门禁的核心语义之一。以前这里是
    # `if checker is not None:` —— 定位不到 skill 就整段跳过，连 [todo] 都不打，
    # 是一条完全看不见的静默通道。现在无条件加载，找不到就 FAIL。
    checker, _path = require_skill_layout_checker()
    bad_trace, _summary = emit_perfetto.build_perfetto([(crossing, "events_rank0.json")])
    contract = best_effort_layout_contract(
        abi, state["exp"], has_counters=False, events_path="events_rank0.json")
    fails = checker.verify(bad_trace, contract)
    _require("skill verify() 也把交叉区间报成 V5 [slice_partial_overlap]",
             any("slice_partial_overlap" in f for f in fails), fails[:3])


CASES = (
    ("cli_contract_path", case_cli_contract_path),
    ("delivery_schema", case_delivery_schema),
    ("skill_layout_verify", case_skill_layout_verify),
    ("trace_processor_import", case_trace_processor_import),
    ("oversize_seam", case_oversize_seam),
    ("partial_overlap_rejected", case_partial_overlap_rejected),
)


def run(tmp: Path) -> Tuple[List[str], List[str]]:
    """跑完所有 case，返回 (真的验过的 case, 被降级为 UNVERIFIED 的 case)。"""
    global _CURRENT_CASE
    if not ABI_PATH.is_file():
        raise SelfTestError("找不到 ABI：{}；先用项目的 extract_abi.py 生成".format(ABI_PATH))
    abi = load_abi(ABI_PATH)
    exp = fixture_expectations(abi)
    _note("ABI 派生期望：ph=X {} 条（task {} / phase {}，ignored 丢弃 {} 条）、"
          "ph=C {} 点、observer 车道 {} 条、allowed_task_ids={}".format(
              exp["x_spans"], exp["task_spans"], exp["phase_spans"], exp["ignored_spans"],
              exp["counter_events"], exp["observer_lanes"], exp["allowed_task_ids"]))
    state: Dict[str, Any] = {"tmp": tmp, "exp": exp,
                             "decoded": fixtures.make_decoded(abi),
                             "hw": fixtures.make_hw()}
    passed: List[str] = []
    unverified: List[str] = []
    for name, fn in CASES:
        _CURRENT_CASE = name
        print("[case] {}".format(name))
        before = len(_TODOS)
        fn(abi, state)
        gaps = len(_TODOS) - before
        if gaps:
            # 有已登记缺口 == 这个 case 没验全 == 不许计入 passed。
            unverified.append(name)
            print("  [result] {} = UNVERIFIED（{} 项已登记缺口未验）".format(name, gaps))
        else:
            passed.append(name)
            print("  [result] {} = PASSED".format(name))
    _CURRENT_CASE = "<done>"
    return passed, unverified


def main() -> int:
    print("[selftest_perfetto] profile root = {}".format(_PROFILE_ROOT))
    print("[selftest_perfetto] abi          = {}".format(ABI_PATH))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            passed, unverified = run(Path(tmp))
    except (SelfTestError, fixtures.SelfTestError) as error:
        print("SELFTEST=FAIL name=selftest_perfetto case={} reason={}".format(
            _CURRENT_CASE, error), file=sys.stderr)
        return EXIT_FAIL
    except Exception as error:  # noqa: BLE001 -- 如实报出任何意外失败
        print("SELFTEST=FAIL name=selftest_perfetto case={} unexpected={}: {}".format(
            _CURRENT_CASE, type(error).__name__, error), file=sys.stderr)
        return EXIT_FAIL

    if not _TODOS:
        print("SELFTEST=PASS name=selftest_perfetto checks={} todos=0 cases={}".format(
            _OK_LINES, ",".join(passed)))
        return EXIT_PASS

    # 存在 todo == 没验全。绝不允许显示成干净的 PASS，退出码也必须非零 ——
    # 「打印了 [todo] 之后仍然 return 0」正是本文件修掉的那类静默通过。
    print("[selftest_perfetto] {} 项已登记的骨架移植缺口（下列语义这一轮没验）：".format(
        len(_TODOS)))
    for case, tag, msg in _TODOS:
        print("  - [{}] (case={}) {}".format(tag, case, msg))
    marker = ("SELFTEST=PASS_WITH_TODOS name=selftest_perfetto checks={} todos={} "
              "verified_cases={} unverified_cases={}".format(
                  _OK_LINES, len(_TODOS), ",".join(passed) or "-",
                  ",".join(unverified) or "-"))
    print(marker)
    print("{} -- 未验全：{}/{} 个 case 被降级为 UNVERIFIED；补完这 {} 项缺口（并从 "
          "KNOWN_LAYOUT_GAPS / KNOWN_EMITTER_GAPS / OPTIONAL_DELIVERY_KEYS 里删掉）"
          "本自测才会 rc=0".format(marker, len(unverified), len(CASES), len(_TODOS)),
          file=sys.stderr)
    return EXIT_PASS_WITH_TODOS


if __name__ == "__main__":
    sys.exit(main())
