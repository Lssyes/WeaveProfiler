#!/usr/bin/env python3
"""validate_overhead.py（骨架版）— E24 / 门禁 G03：证明"插桩足够便宜"。

流程位置
  编排器 overhead 阶段的固定入口。同一个 kernel、同一份输入，只改插桩编译档位，
  测三次：
    * off         —— 所有记录语句被编译掉（基线开销）
    * entry_only  —— 只记录 entry rendezvous（arrive/release）
    * full        —— 所有 task/phase 事件都记录

  上报的开销比是 contract 定义的 full-vs-entry_only：

      overhead_ratio = (median_full_ns - median_entry_only_ns) / median_entry_only_ns

  必须 <= contract 里的上限。上限按 granularity 分档：task 0.10，phase 0.15
  （phase 记录每个 task 的内部阶段，余量更大；15% 是不可再抬的绝对上限）。
  为什么拿 entry_only 当分母而不是 off：entry rendezvous 是"能对齐时间轴"的最低成本，
  它是插桩的固有代价；真正要卡的是"多记了那么多事件贵不贵"。

  每档都先 warmup 再用 CUDA event 计时 N 次取**中位数**，个别调度抖动不会误判门禁。
  锁 / profiler 计数器的清零发生在**计时区之外**，所以量到的只有 kernel 执行时间，
  而且 full 档不会在一次测量中途把 buffer 写满回绕。

谁来调 / CLI
  python3 -B validate_overhead.py --contract <lock> --out <run_root>/validation/overhead.json
  python3 -B validate_overhead.py --contract <lock> --out <p.json> --dry-run   # 纯 CPU 出计划
  退出码：0 通过 / 5 超预算 / 2 出错（含 GPU 钩子未实现）。

设计约束（照抄，别改）
  torch / cutlass / harness 的 import 全部写在 GPU 函数**内部**，保证 --help /
  --dry-run / py_compile 在 CPU-only 主机上可用。

接新 target 要改的地方
  1. 配置区 PROJECT_OVERHEAD：三个档位名（CONFIG_LEVEL_NAMES 由它生成）、默认上限
     与迭代次数；
  2. ``compile_at_level()``：在指定档位编译一次，返回重复发射所需的一切；
  3. ``reset_launch_state()``：把 grid barrier / profiler 计数器复位。
  钩子未实现时 raise NotImplementedError —— **不要**改成"测不了就当没开销"。

真实参考实现
  DSA-learn/mk/h20/cutedsl/deepseek_v4_pro/decode/profile/validation/validate_overhead.py
    - _compile_at_level(): 第 104-125 行
    - _reset():            第 128-135 行
    - _time_level():       第 138-168 行（本文件已泛化照搬）
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS = Path(__file__).resolve()
_VALIDATION = _THIS.parent
_PROFILE_ROOT = _VALIDATION.parent
_VERSION_ROOT = _PROFILE_ROOT.parent

if str(_PROFILE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROFILE_ROOT))

SCHEMA = "fast-instrument-overhead/v1"

_REAL_IMPL = ("DSA-learn/mk/h20/cutedsl/deepseek_v4_pro/decode/profile/validation/"
              "validate_overhead.py")

# ======================================================================
# ===== 接新 target 时只需要改这里 (PROJECT CONFIG) =====
# ======================================================================
PROJECT_OVERHEAD: Dict[str, Any] = {
    # 1) ABI 兜底路径（相对 profile/）。真正路径优先从 contract 读。
    "abi_relpath": "device_record/abi/record_abi.example.json",

    # 2) 三个编译档位在你项目里的名字。角色 key 固定 off / entry_only / full，
    #    开销公式依赖这三个角色；value 是项目里的字面名字，会原样写进证据 JSON。
    #    骨架：JSON preset 的 off / entry_only / full；
    #    真实项目：cutlass Constexpr 的 PROF_OFF / PROF_ENTRY_ONLY / PROF_FULL。
    "level_names": {"off": "off", "entry_only": "entry_only", "full": "full"},

    # 3) 兜底预算与迭代次数。contract 里 validation.overhead.* 优先；缺省时按
    #    granularity 分档（task 0.10 / phase 0.15，见 _contract_overhead）。
    #    ！！放宽 default_maximum 等于放宽验收标准，改之前先确认是真的测不下去。
    "default_maximum": 0.10,
    "default_maximum_phase": 0.15,
    "default_warmup_iterations": 20,
    "default_measured_iterations": 100,
}

# 三个档位的顺序（证据里的 levels 和 configurations 都按这个顺序）。
CONFIG_LEVEL_NAMES = (PROJECT_OVERHEAD["level_names"]["off"],
                      PROJECT_OVERHEAD["level_names"]["entry_only"],
                      PROJECT_OVERHEAD["level_names"]["full"])
# ======================================================================
# ===== 配置区结束；下面是与项目无关的通用逻辑 =====
# ======================================================================


def load_json(path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dig(node: Any, dotted: str, default: Any = None) -> Any:
    cur = node
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def first(node: Any, dotted_paths, default: Any = None) -> Any:
    for dotted in dotted_paths:
        value = dig(node, dotted, None)
        if value is not None:
            return value
    return default


def contract_paths(contract: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """兼容 frozen lock / materialize 产出 / 真实项目三种 contract 形状。"""
    contract = contract if isinstance(contract, dict) else {}
    abi = first(contract, ("profile_selection.abi.absolute_path", "profile_selection.abi",
                           "profile_selection.abi_path"))
    abi_path = Path(abi) if abi else _PROFILE_ROOT / PROJECT_OVERHEAD["abi_relpath"]
    rank = first(contract, ("run.display_rank", "capture.display_rank", "display_rank"), 0)
    world = first(contract, ("run.world_size", "capture.world_size", "world_size"), 1)
    selected = first(contract, ("capture.software_sm_scope.selected_physical_smids",
                                "capture.selected_physical_smids"), [])
    return {"abi_path": abi_path, "rank": int(rank), "world_size": max(1, int(world)),
            "selected_physical_smids": list(selected)}


def _role_map() -> Dict[str, str]:
    names = PROJECT_OVERHEAD["level_names"]
    return {"instrumentation_disabled": names["off"],
            "entry_anchors_only": names["entry_only"],
            "full_event_recording": names["full"]}


def _contract_overhead(contract: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    granularity = str(dig(contract, "capture.granularity", "") or "").strip().lower()
    fallback = (PROJECT_OVERHEAD["default_maximum_phase"] if granularity == "phase"
                else PROJECT_OVERHEAD["default_maximum"])
    defaults = {
        "maximum": fallback,
        "warmup_iterations": PROJECT_OVERHEAD["default_warmup_iterations"],
        "measured_iterations": PROJECT_OVERHEAD["default_measured_iterations"],
        "formula": "(median_full_ns - median_entry_only_ns) / median_entry_only_ns",
        "configurations": _role_map(),
    }
    if not isinstance(contract, dict):
        return defaults
    over = dig(contract, "validation.overhead", {})
    if isinstance(over, dict):
        for key in ("warmup_iterations", "measured_iterations", "formula"):
            if over.get(key) is not None:
                defaults[key] = over[key]
        # frozen lock 用 maximum，materialize 产出用 max —— 两种都认。
        budget = over.get("maximum", over.get("max"))
        if budget is not None:
            defaults["maximum"] = budget
        if isinstance(over.get("configurations"), dict):
            defaults["configurations"] = over["configurations"]
    return defaults


def _plan(contract: Optional[Dict[str, Any]], info: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _contract_overhead(contract)
    return {
        "schema": SCHEMA,
        "levels": list(CONFIG_LEVEL_NAMES),
        "configuration_names": _role_map(),
        "formula": cfg["formula"],
        "maximum": float(cfg["maximum"]),
        "warmup_iterations": int(cfg["warmup_iterations"]),
        "measured_iterations": int(cfg["measured_iterations"]),
        "rank": int(info["rank"]),
        "world_size": int(info["world_size"]),
        "abi_path": str(info["abi_path"]),
    }


# ----------------------------------------------------------------------
# ===== 项目钩子（GPU 侧）=====
# ----------------------------------------------------------------------
def setup_gpu_imports() -> Dict[str, Any]:
    """把 GPU 侧需要的模块凑成 dict 返回。import 必须写在函数体内。"""
    raise NotImplementedError(
        "接入新 target 时请在此实现 setup_gpu_imports()：import torch 与你的 kernel/"
        "harness 模块并返回 {'torch': torch, ...}。"
        "参考真实实现 {}:178（rp._setup_gpu_imports）".format(_REAL_IMPL))


def compile_at_level(mods: Dict[str, Any], level_name: str,
                     info: Dict[str, Any]) -> Dict[str, Any]:
    """在指定档位编译一次，返回计时循环重复发射所需的一切。

    返回值约定（generic 计时循环依赖这些 key）：
      {"launch": callable(), "state": <任意，交给 reset_launch_state>}
    ``launch()`` 每次调用发射一次 kernel，**不要**在里面 synchronize（计时循环负责）。
    """
    raise NotImplementedError(
        "接入新 target 时请在此实现 compile_at_level(mods, {!r}, info)："
        "返回 {{'launch': <发射一次的可调用>, 'state': <复位所需状态>}}。"
        "参考真实实现 {}:104-125"
        "（harness.load_bundle -> R.alloc -> _make_prof_tensors -> "
        "DecodeMKInstr(..., prof_level=level) -> env_compat.to_cute -> cute.compile）"
        .format(level_name, _REAL_IMPL))


def reset_launch_state(mods: Dict[str, Any], compiled: Dict[str, Any]) -> None:
    """把 grid barrier / profiler 计数器 / buffer 复位，让每次测量的起点一致。

    必须在计时区之外调用：既保证只量到 kernel 执行时间，也保证 full 档不会
    在一次测量中途把 buffer 写满回绕（回绕会让开销数字失真且触发溢出门禁）。
    """
    raise NotImplementedError(
        "接入新 target 时请在此实现 reset_launch_state(mods, compiled)："
        "清零 grid barrier(locks) 与 profiler 的 counter/overflow/buffer。"
        "参考真实实现 {}:128-135".format(_REAL_IMPL))


# ----------------------------------------------------------------------
# 通用计时（与项目无关，别改）
# ----------------------------------------------------------------------
def _time_level(mods: Dict[str, Any], level_name: str, info: Dict[str, Any],
                warmup: int, measured: int) -> Dict[str, Any]:
    """一个档位的中位数（外加 min/mean/max）耗时，单位 ns。"""
    torch = mods["torch"]
    compiled = compile_at_level(mods, level_name, info)
    launch = compiled["launch"]

    for _ in range(max(1, warmup)):
        reset_launch_state(mods, compiled)
        torch.cuda.synchronize()
        launch()
    torch.cuda.synchronize()

    samples_ns: List[float] = []
    for _ in range(max(1, measured)):
        reset_launch_state(mods, compiled)
        torch.cuda.synchronize()          # 复位在计时区之外
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        torch.cuda.synchronize()
        samples_ns.append(float(start.elapsed_time(end)) * 1.0e6)  # ms -> ns

    return {
        "median_ns": statistics.median(samples_ns),
        "mean_ns": statistics.fmean(samples_ns),
        "min_ns": min(samples_ns),
        "max_ns": max(samples_ns),
        "iterations": len(samples_ns),
    }


def run_overhead(contract: Optional[Dict[str, Any]], out_path: Path,
                 info: Dict[str, Any]) -> int:
    cfg = _contract_overhead(contract)
    maximum = float(cfg["maximum"])
    warmup = int(cfg["warmup_iterations"])
    measured = int(cfg["measured_iterations"])
    names = PROJECT_OVERHEAD["level_names"]

    mods = setup_gpu_imports()
    timings = {name: _time_level(mods, name, info, warmup, measured)
               for name in CONFIG_LEVEL_NAMES}

    median_off = timings[names["off"]]["median_ns"]
    median_entry = timings[names["entry_only"]]["median_ns"]
    median_full = timings[names["full"]]["median_ns"]

    # 分母 <= 0 说明计时没量到东西 -> inf -> 必然超预算 -> FAIL。不许当成 0 开销。
    overhead_ratio = ((median_full - median_entry) / median_entry
                      if median_entry > 0 else float("inf"))
    entry_over_off = ((median_entry - median_off) / median_off
                      if median_off > 0 else float("inf"))
    full_over_off = ((median_full - median_off) / median_off
                     if median_off > 0 else float("inf"))
    within_max = overhead_ratio <= maximum

    # 门禁 G03 读 `configurations`（要求 >=3 档）和 `within_max`。
    configurations = {name: timings[name]["median_ns"] for name in CONFIG_LEVEL_NAMES}

    report = dict(_plan(contract, info))
    report.update({
        "within_max": bool(within_max),
        "overhead_ratio": overhead_ratio,
        "overhead_ratio_full_vs_entry_only": overhead_ratio,
        "overhead_ratio_entry_only_vs_off": entry_over_off,
        "overhead_ratio_full_vs_off": full_over_off,
        "configurations": configurations,
        "medians_ns": configurations,
        "timings": timings,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[validate_overhead] wrote {} within_max={} overhead_ratio={:.4f}".format(
        out_path, within_max, overhead_ratio))
    return 0 if within_max else 5


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Overhead gate: three compile levels ({}) timed on the same input."
                    .format(" / ".join(CONFIG_LEVEL_NAMES)))
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--abi", type=Path, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU-only: emit the overhead plan without launching the GPU")
    args = parser.parse_args(argv)

    contract = None
    if args.contract is not None:
        try:
            contract = load_json(args.contract)
        except (OSError, ValueError) as error:
            print("[validate_overhead] cannot read contract {}: {}".format(
                args.contract, error), file=sys.stderr)
            return 2
    info = contract_paths(contract)
    if args.abi is not None:
        info["abi_path"] = args.abi
    if args.rank is not None:
        info["rank"] = args.rank

    if args.dry_run:
        plan = _plan(contract, info)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("[validate_overhead] dry-run plan -> {}".format(args.out))
        return 0

    try:
        return run_overhead(contract, args.out, info)
    except Exception as error:  # noqa: BLE001 -- 如实暴露真实失败原因
        print("[validate_overhead] FAILED: {}: {}".format(
            type(error).__name__, error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
