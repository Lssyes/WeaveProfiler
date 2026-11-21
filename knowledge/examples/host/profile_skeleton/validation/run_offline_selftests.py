#!/usr/bin/env python3
"""run_offline_selftests.py（骨架版）— E25 离线自测聚合器（纯 CPU，不碰 GPU）。

流程位置
  contract 里 offline_selftest 这一阶段的固定入口。它把 ``validation/selftest/`` 下
  所有 ``selftest_*.py`` 逐个当子进程跑掉，汇总成一份
  ``offline-selftest-report/v1`` 报告；preflight E25 会检查这份聚合器存在且覆盖
  decoder / hw_merge / perfetto 三个方面，gate 侧则读它产出的报告。

谁来调 / CLI
  python3 -B run_offline_selftests.py [--contract <lock>] --out <report.json>

约定
  每个 selftest 是独立可执行程序：PASS 退出 0，FAIL 退出非 0，并打印
  ``SELFTEST=PASS name=...`` / ``SELFTEST=FAIL ...`` 标记。新增自测只要按这个命名
  丢进 selftest/ 目录就会被自动发现，不需要改本文件。

接新 target 要改的地方
  基本没有 —— 本文件可以整份照抄。只有 SELFTEST_GLOB / 默认超时这两项在配置区，
  真要改也只改那里。注意"一个 selftest 都没发现"必须判 FAIL（见 all_pass 的
  ``bool(selftests)``）：空目录静默通过等于把 E25 变成摆设。

真实参考实现
  DSA-learn/mk/h20/cutedsl/deepseek_v4_pro/decode/profile/validation/run_offline_selftests.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_VALIDATION_DIR = Path(__file__).resolve().parent
_PROFILE_ROOT = _VALIDATION_DIR.parent

# ======================================================================
# ===== 接新 target 时只需要改这里 (PROJECT CONFIG) =====
# ======================================================================
SELFTEST_DIR = _VALIDATION_DIR / "selftest"   # 自测目录
SELFTEST_GLOB = "selftest_*.py"               # 发现规则
DEFAULT_TIMEOUT_SEC = 300.0                   # 单个自测的超时（秒）
# ======================================================================
# ===== 配置区结束 =====
# ======================================================================


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def discover() -> List[Path]:
    if not SELFTEST_DIR.is_dir():
        return []
    return sorted(p for p in SELFTEST_DIR.glob(SELFTEST_GLOB) if p.is_file())


def run_one(path: Path, timeout_sec: float) -> Dict[str, Any]:
    """把一个自测当子进程跑掉。超时 / 起不来都算 FAIL，绝不当成通过。"""
    started = utc_now()
    try:
        completed = subprocess.run(
            [sys.executable, "-B", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(_PROFILE_ROOT), timeout=timeout_sec, check=False)
        return {"selftest": path.name, "path": str(path), "exit_code": completed.returncode,
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip(),
                "started_at_utc": started, "ended_at_utc": utc_now()}
    except subprocess.TimeoutExpired as error:
        return {"selftest": path.name, "path": str(path), "exit_code": None, "status": "FAIL",
                "stdout": (error.stdout or "").strip() if isinstance(error.stdout, str) else "",
                "stderr": "timed out after {}s".format(timeout_sec),
                "started_at_utc": started, "ended_at_utc": utc_now()}
    except OSError as error:
        return {"selftest": path.name, "path": str(path), "exit_code": None, "status": "FAIL",
                "stdout": "", "stderr": "could not execute: {}".format(error),
                "started_at_utc": started, "ended_at_utc": utc_now()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run all offline (CPU) selftests and aggregate.")
    parser.add_argument("--contract", type=Path, default=None,
                        help="unused; accepted for stage parity")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    args = parser.parse_args(argv)

    selftests = discover()
    results = [run_one(path, args.timeout_sec) for path in selftests]
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] != "PASS")
    # bool(selftests): 一个都没发现 -> FAIL。空目录不许静默通过。
    all_pass = bool(selftests) and failed == 0
    payload = {
        "schema": "offline-selftest-report/v1",
        "generated_at_utc": utc_now(),
        "selftest_dir": str(SELFTEST_DIR),
        "discovered": [p.name for p in selftests],
        "count": len(selftests), "passed": passed, "failed": failed,
        "status": "PASS" if all_pass else "FAIL",
        "results": results,
    }
    if not selftests:
        payload["reason"] = "no selftests discovered under {}".format(SELFTEST_DIR)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for r in results:
        print("[selftest] {} -> {} (exit={})".format(r["selftest"], r["status"], r["exit_code"]))
        if r["status"] != "PASS" and r["stderr"]:
            print("           {}".format(r["stderr"].splitlines()[-1] if r["stderr"] else ""))
    print("OFFLINE_SELFTESTS={} passed={} failed={} count={}".format(
        payload["status"], passed, failed, len(selftests)))
    if args.out is not None:
        print("wrote {}".format(args.out))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
