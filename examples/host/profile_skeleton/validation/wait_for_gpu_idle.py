#!/usr/bin/env python3
"""wait_for_gpu_idle.py（骨架版）— 门禁 G00：等到 contract 允许的 GPU 空闲，并落一份快照。

流程位置
  frozen contract 的 ``execution.gpu_idle_command`` 只带 ``--contract`` 调本文件
  （不给 ``--out``）；之后 gate 适配器读 ``<run_root>/validation/gpu_idle.json``，
  只有 ``idle`` 为 true 才让 G00 过。所以省略 ``--out`` 时本程序会从 contract 的
  ``project_layout.run_root`` 推出**同一个**默认路径 —— 命令和门禁必须指向同一个文件，
  否则会出现"等待成功但门禁读不到快照"的错位。

空闲判据来自 ``execution.gpu_idle_policy``：
  * 某块 GPU 空闲 = 利用率 <= ``utilization_max_percent``，且（设了
    ``memory_max_mib`` 时）已用显存 <= 该上限；
  * 只看 ``execution.allowed_gpu_ids`` ∩ ``CUDA_VISIBLE_DEVICES`` 里的卡；
  * 每 ``poll_interval_sec`` 轮询一次，直到全部空闲或 ``timeout_sec`` 超时。

nvidia-smi 以 ``shell=False`` 调用、按 CSV 解析。**没有 nvidia-smi = GPU 状态不可观测
= BLOCKED，不是 idle。** 此时写 ``idle=false`` + ``probe: "nvidia-smi_absent"`` 并以
专用退出码 3 退出，让上游（run_measurement 的 GPU 阶段、门禁 G00）如实拦住。

  历史坑（本轮修掉的）：这里曾经写 ``idle=true`` + ``probe: "nvidia-smi_absent"``
  并退出 0，理由是"纯 CPU 机器没有 GPU 可争"；而 ``check_gate_item.gate_g00`` 只读
  ``snap["idle"]``，于是 **G00 在任何没有 nvidia-smi 的机器上无条件放行** —— 一个
  门禁白送，还留下一份写着 idle=true 的"证据"。测不到就是测不到，不许自称通过。

谁来调 / CLI
  python3 -B wait_for_gpu_idle.py --contract <lock>
  python3 -B wait_for_gpu_idle.py --contract <lock> --out <p.json> --once
  退出码：0 空闲 / 1 超时仍忙 / 2 出错 / 3 GPU 状态不可观测（没有 nvidia-smi）。
  任何非 0 都会让 GPU 阶段 FAIL；3 只是为了在日志里一眼区分"忙"和"测不到"。

接新 target 要改的地方
  基本可以整份照抄。只有配置区里的默认空闲策略、快照相对路径、probe 标记、以及
  ``contract_paths()`` 读 run_root / allowed_gpu_ids 的候选键需要核对一次。

参考实现（dsv4pro/decode，已通过 25 门禁 —— 这是**范本路径**，不是本项目路径）
  DSA-learn/mk/h20/cutedsl/deepseek_v4_pro/decode/profile/validation/wait_for_gpu_idle.py
  注意：那份实现同样有上面这个 idle=true 的老坑（它跑在有 nvidia-smi 的机器上所以
  从未触发）；照抄时请以本文件为准。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS = Path(__file__).resolve()
_VALIDATION = _THIS.parent
_PROFILE_ROOT = _VALIDATION.parent
_VERSION_ROOT = _PROFILE_ROOT.parent

SCHEMA = "fast-instrument-gpu-idle/v1"

# ======================================================================
# ===== 接新 target 时只需要改这里 (PROJECT CONFIG) =====
# ======================================================================
# 1) 兜底空闲策略。contract 的 execution.gpu_idle_policy 优先。
DEFAULT_POLICY: Dict[str, Any] = {
    "utilization_max_percent": 10,
    "memory_max_mib": None,      # None = 不看显存，只看利用率
    "poll_interval_sec": 5,
    "timeout_sec": 1800,
}

# 2) 快照相对 run_root 的路径。必须与 check_gate_item.py 配置区的
#    artifact_relpaths["gpu_idle"] 完全一致，否则 G00 会读不到。
GPU_IDLE_SNAPSHOT_RELPATH = "validation/gpu_idle.json"

# 3) run_root / allowed_gpu_ids 在 contract 里的候选键（三种 contract 形状都兼容）。
RUN_ROOT_KEYS = ("project_layout.run_root", "artifacts.run_root", "run.run_root")
ALLOWED_GPU_KEYS = ("execution.allowed_gpu_ids", "run.allowed_gpu_ids", "allowed_gpu_ids")

# 4) 快照里 ``probe`` 字段的取值。PROBE_MEASURED = "真的问过 nvidia-smi 了"，
#    PROBE_UNOBSERVABLE = "GPU 状态不可观测"。改了这里，必须同步改
#    check_gate_item.py 配置区的 GATE_PROJECT["gpu_idle_trusted_probes"] ——
#    G00 用**白名单**判 probe，认不出的 probe 一律 FAIL（fail-closed），
#    所以两边不一致的后果是 G00 一直 FAIL（吵，但安全），而不是白送。
PROBE_MEASURED = "nvidia-smi"
PROBE_UNOBSERVABLE = "nvidia-smi_absent"
# ======================================================================
# ===== 配置区结束；下面是与项目无关的通用逻辑 =====
# ======================================================================

EXIT_IDLE = 0
EXIT_BUSY = 1           # 轮询到超时仍然忙
EXIT_ERROR = 2          # 读不动 contract / 未预期异常
EXIT_UNOBSERVABLE = 3   # 没有 nvidia-smi：GPU 状态不可观测 = BLOCKED，不是 idle


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    """只取本文件需要的两项：run_root（决定快照落点）与 allowed_gpu_ids。"""
    contract = contract if isinstance(contract, dict) else {}
    return {"run_root": first(contract, RUN_ROOT_KEYS),
            "allowed_gpu_ids": first(contract, ALLOWED_GPU_KEYS)}


def resolve_policy(contract: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    policy = dict(DEFAULT_POLICY)
    if isinstance(contract, dict):
        raw = dig(contract, "execution.gpu_idle_policy", {})
        if isinstance(raw, dict):
            for key in DEFAULT_POLICY:
                if raw.get(key) is not None:
                    policy[key] = raw[key]
    return policy


def allowed_ids(info: Dict[str, Any]) -> Optional[List[int]]:
    ids = info.get("allowed_gpu_ids")
    contract_allowed = [int(x) for x in ids] if isinstance(ids, list) and ids else None
    # GPU 阶段通常由 acquire.sh 之类的脚本 flock 一块空闲卡，并把
    # CUDA_VISIBLE_DEVICES 导成那块卡的物理号（比如 "2"）。nvidia-smi 的 index 字段
    # 也是物理号，所以把空闲检查缩到"真正分到的那块卡"：这样别的任务占着的卡
    # 不会把本流水线卡死。
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        try:
            visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
        except ValueError:
            visible = []  # UUID 形式（GPU-...），映射不到 nvidia-smi index
        if visible:
            if contract_allowed is not None:
                inter = [g for g in visible if g in contract_allowed]
                return inter or visible
            return visible
    return contract_allowed  # None => 考虑所有可见 GPU


def query_nvidia_smi(smi: str) -> Tuple[List[Dict[str, Any]], int, str]:
    """返回 ([{index,utilization_percent,memory_used_mib,memory_total_mib}], rc, stderr)。"""
    argv = [smi, "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return [], -1, "nvidia-smi invocation failed: {}".format(error)
    rows: List[Dict[str, Any]] = []
    for line in proc.stdout.strip().splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            rows.append({"index": int(parts[0]),
                         "utilization_percent": float(parts[1]),
                         "memory_used_mib": float(parts[2]),
                         "memory_total_mib": float(parts[3])})
        except ValueError:
            continue
    return rows, proc.returncode, proc.stderr.strip()


def evaluate_idle(rows: List[Dict[str, Any]], allowed: Optional[List[int]],
                  policy: Dict[str, Any]
                  ) -> Tuple[bool, List[Dict[str, Any]], List[Dict[str, Any]]]:
    util_max = float(policy["utilization_max_percent"])
    mem_max = policy["memory_max_mib"]
    considered = [r for r in rows if allowed is None or r["index"] in allowed]
    busy: List[Dict[str, Any]] = []
    for r in considered:
        util_ok = r["utilization_percent"] <= util_max
        mem_ok = (mem_max is None) or (r["memory_used_mib"] <= float(mem_max))
        if not (util_ok and mem_ok):
            busy.append(r)
    # bool(considered): 一块都没看到 -> 不算空闲（nvidia-smi 在但查不到卡是异常）。
    idle = bool(considered) and not busy
    return idle, considered, busy


def build_snapshot(idle: bool, **extra: Any) -> Dict[str, Any]:
    snap = {"schema": SCHEMA, "idle": bool(idle), "checked_at_utc": utc_now()}
    snap.update(extra)
    return snap


def write_snapshot(out_path: Optional[Path], snap: Dict[str, Any]) -> None:
    if out_path is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snap, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
                        encoding="utf-8")


def default_out(contract: Optional[Dict[str, Any]], info: Dict[str, Any]) -> Optional[Path]:
    run_root = info.get("run_root")
    if run_root:
        return Path(run_root) / GPU_IDLE_SNAPSHOT_RELPATH
    return None


def run(contract: Optional[Dict[str, Any]], info: Dict[str, Any], out_path: Optional[Path],
        policy: Dict[str, Any], allowed: Optional[List[int]], once: bool) -> int:
    smi = shutil.which("nvidia-smi")
    if not smi:
        # GPU 状态不可观测。以前这里写 idle=true 并 return 0（"纯 CPU 机器没有 GPU
        # 可争"），配上只读 snap["idle"] 的 gate_g00，等于 G00 在所有没有 nvidia-smi
        # 的机器上白送。现在如实写 idle=false 并以专用退出码 3 退出：
        # 快照仍然落盘（保持可审计 + 让 G00 报出具体原因），但绝不自称通过。
        snap = build_snapshot(
            False, probe=PROBE_UNOBSERVABLE, unobservable=True,
            allowed_gpu_ids=allowed, policy=policy,
            note="nvidia-smi not on PATH: GPU state is unobservable; "
                 "this is BLOCKED, not idle",
            howto="在有 GPU 的机器上跑测量阶段；若 nvidia-smi 不在 PATH 上，"
                  "把它所在目录加进 PATH。纯 CPU 的编写机上 GPU 阶段本来就不该判过。")
        write_snapshot(out_path, snap)
        print("[wait_for_gpu_idle] nvidia-smi absent -> GPU state UNOBSERVABLE; "
              "idle=false (BLOCKED, not idle) out={}".format(out_path), file=sys.stderr)
        return EXIT_UNOBSERVABLE

    deadline = time.monotonic() + float(policy["timeout_sec"])
    interval = max(0.0, float(policy["poll_interval_sec"]))
    polls = 0
    last_snap: Dict[str, Any] = build_snapshot(False, note="no poll completed")
    while True:
        polls += 1
        rows, rc, stderr = query_nvidia_smi(smi)
        idle, considered, busy = evaluate_idle(rows, allowed, policy)
        last_snap = build_snapshot(
            idle, probe=PROBE_MEASURED, nvidia_smi_returncode=rc,
            allowed_gpu_ids=allowed, policy=policy, polls=polls,
            considered_gpus=considered, busy_gpus=busy,
            nvidia_smi_stderr=stderr or None)
        if idle:
            write_snapshot(out_path, last_snap)
            print("[wait_for_gpu_idle] idle after {} poll(s); out={}".format(polls, out_path))
            return EXIT_IDLE
        if once or time.monotonic() >= deadline:
            break
        time.sleep(interval)

    last_snap["timed_out"] = True
    write_snapshot(out_path, last_snap)
    print("[wait_for_gpu_idle] NOT idle after {} poll(s) (busy={}); out={}".format(
        polls, [g["index"] for g in last_snap.get("busy_gpus", [])], out_path), file=sys.stderr)
    return EXIT_BUSY


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Wait until the contract's allowed GPUs are idle, "
                    "then emit the gate-G00 snapshot.")
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="snapshot path; defaults to <run_root>/" + GPU_IDLE_SNAPSHOT_RELPATH)
    parser.add_argument("--timeout-sec", type=float, default=None, help="override policy timeout")
    parser.add_argument("--poll-interval-sec", type=float, default=None,
                        help="override policy poll interval")
    parser.add_argument("--once", action="store_true", help="poll a single time (no waiting)")
    args = parser.parse_args(argv)

    contract = None
    if args.contract is not None:
        try:
            contract = load_json(args.contract)
        except (OSError, ValueError) as error:
            print("[wait_for_gpu_idle] cannot read contract {}: {}".format(
                args.contract, error), file=sys.stderr)
            return EXIT_ERROR
    info = contract_paths(contract)
    policy = resolve_policy(contract)
    if args.timeout_sec is not None:
        policy["timeout_sec"] = args.timeout_sec
    if args.poll_interval_sec is not None:
        policy["poll_interval_sec"] = args.poll_interval_sec
    allowed = allowed_ids(info)
    out_path = args.out if args.out is not None else default_out(contract, info)

    try:
        return run(contract, info, out_path, policy, allowed, args.once)
    except Exception as error:  # noqa: BLE001 -- 如实暴露真实失败原因
        print("[wait_for_gpu_idle] FAILED: {}: {}".format(
            type(error).__name__, error), file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
