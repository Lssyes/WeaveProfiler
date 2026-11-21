#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C2 正确性口径离线自检（不需要 GPU，只需要 numpy）。

用同一份 randn 数据造参考/候选张量，走 knowledge/templates/compare_raw_arrays.py 的 CLI，验证：
  1. 逐位一致 -> COMPARE_PASS、pass=true、checked_elements 覆盖全部元素、max_abs=max_rel=0
  2. 改一个元素 -> rc=2、pass=false、first_mismatch 给出确切下标与两边数值
  3. 出现 NaN -> nonfinite>0 且判失败（不能靠 atol 放宽混过去）
  4. shape 不一致 / 缺数组 -> 直接报错退出，不做部分比较
  5. --key 只比指定数组；.npy 单数组也能比；pass 的口径是 max_abs<=atol 且 max_rel<=rtol
  6. 交给 4_validate/tools/fi_check.py --compare-cmd 跑 C2：结果与阈值原样进 check.json，
     --expected-elements 大于实际比较元素数时判 FAIL

用法：python3 -I tests/c2_compare_check.py [--keep-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
COMPARE = PKG / "knowledge" / "templates" / "compare_raw_arrays.py"
CHECK = PKG / "4_validate" / "tools" / "fi_check.py"

SHAPE = (64, 128)


def np():
    import numpy
    return numpy


def run_compare(args, expect_rc=None):
    cmd = [sys.executable, "-I", str(COMPARE), *args]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    if expect_rc is not None and p.returncode != expect_rc:
        raise AssertionError(f"compare rc={p.returncode}，期望 {expect_rc}\n{' '.join(cmd)}\n{p.stdout}")
    payload = None
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("COMPARE_JSON "):
            payload = json.loads(line[len("COMPARE_JSON "):])
            break
    return p, payload


def write_arrays(tmp: Path, name: str, arrays: dict) -> Path:
    n = np()
    path = tmp / name
    n.savez(path, **arrays)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-dir", help="保留产物到该目录（默认临时目录，跑完删除）")
    a = ap.parse_args()
    n = np()
    tmp = Path(a.keep_dir) if a.keep_dir else Path(tempfile.mkdtemp(prefix="fi_c2_"))
    tmp.mkdir(parents=True, exist_ok=True)

    rng = n.random.default_rng(20261109)
    ref_out = rng.standard_normal(SHAPE).astype(n.float32)
    ref_aux = rng.standard_normal((SHAPE[0],)).astype(n.float32)
    ref = write_arrays(tmp, "ref.npz", {"output": ref_out, "aux": ref_aux})
    total = ref_out.size + ref_aux.size

    # 1) 逐位一致
    same = write_arrays(tmp, "same.npz", {"output": ref_out.copy(), "aux": ref_aux.copy()})
    p, r = run_compare(["--ref", str(ref), "--candidate", str(same), "--atol", "0", "--rtol", "0"], expect_rc=0)
    assert "COMPARE_PASS" in p.stdout, p.stdout
    assert r["pass"] is True and r["checked_elements"] == total, r
    assert r["max_abs"] == 0.0 and r["max_rel"] == 0.0 and r["nonfinite"] == 0, r
    assert sorted(r["keys"]) == ["aux", "output"], r["keys"]
    assert r["arrays"]["output"]["shape"] == list(SHAPE), r["arrays"]["output"]
    assert "randn" in r["method"] and "逐元素" in r["method"], r["method"]

    # 2) 改一个元素：下标与数值都要报出来
    bad_out = ref_out.copy()
    bad_out[7, 11] += 1.5e-3
    bad = write_arrays(tmp, "bad.npz", {"output": bad_out, "aux": ref_aux.copy()})
    p, r = run_compare(["--ref", str(ref), "--candidate", str(bad)], expect_rc=2)
    assert "COMPARE_FAIL" in p.stdout, p.stdout
    assert r["pass"] is False and r["checked_elements"] == total, r
    assert r["first_mismatch"]["key"] == "output" and r["first_mismatch"]["index"] == [7, 11], r["first_mismatch"]
    assert abs(r["first_mismatch"]["ref"] - float(ref_out[7, 11])) < 1e-6, r["first_mismatch"]
    assert abs(r["max_abs"] - 1.5e-3) < 1e-6, r["max_abs"]
    assert r["arrays"]["output"]["mismatches"] == 1, r["arrays"]["output"]

    # 3) NaN 必须拦住，atol/rtol 放宽也不行
    nan_out = ref_out.copy()
    nan_out[3, 4] = float("nan")
    nanp = write_arrays(tmp, "nan.npz", {"output": nan_out, "aux": ref_aux.copy()})
    p, r = run_compare(["--ref", str(ref), "--candidate", str(nanp), "--atol", "1e9", "--rtol", "1e9"], expect_rc=2)
    assert r["pass"] is False and r["nonfinite"] >= 1, r
    assert r["first_mismatch"]["index"] == [3, 4], r["first_mismatch"]

    # 4) shape 不一致 / 缺数组：直接退出，不给"部分通过"
    shape_bad = write_arrays(tmp, "shape.npz", {"output": ref_out[:8].copy(), "aux": ref_aux.copy()})
    p, _ = run_compare(["--ref", str(ref), "--candidate", str(shape_bad)])
    assert p.returncode != 0 and "shape 不一致" in p.stdout, p.stdout
    missing = write_arrays(tmp, "missing.npz", {"output": ref_out.copy()})
    p, _ = run_compare(["--ref", str(ref), "--candidate", str(missing), "--key", "output", "--key", "aux"])
    assert p.returncode != 0 and "缺数组" in p.stdout, p.stdout

    # 5) --key 只比指定数组；.npy 单数组也支持
    p, r = run_compare(["--ref", str(ref), "--candidate", str(bad), "--key", "aux"], expect_rc=0)
    assert r["keys"] == ["aux"] and r["checked_elements"] == ref_aux.size and r["pass"] is True, r
    npy_ref = tmp / "ref.npy"
    npy_bad = tmp / "bad.npy"
    n.save(npy_ref, ref_out)
    n.save(npy_bad, bad_out)
    p, r = run_compare(["--ref", str(npy_ref), "--candidate", str(npy_bad)], expect_rc=2)
    assert r["keys"] == ["array"] and r["first_mismatch"]["index"] == [7, 11], r

    # 5b) 阈值口径：pass 要 max_abs<=atol 且 max_rel<=rtol 两条都成立（与 fi_check C2 一致）
    p, r = run_compare(["--ref", str(ref), "--candidate", str(bad), "--atol", "2e-3", "--rtol", "1e-2"], expect_rc=0)
    assert r["pass"] is True and r["thresholds"] == {"atol": 0.002, "rtol": 0.01}, r
    assert r["max_abs"] <= 0.002 and r["max_rel"] <= 0.01, r
    # 只放宽 rtol 不放宽 atol：仍然判失败，阈值原样写进结果供审计
    p, r = run_compare(["--ref", str(ref), "--candidate", str(bad), "--atol", "0", "--rtol", "1e-2"], expect_rc=2)
    assert r["pass"] is False and r["max_abs"] > 0.0, r

    # 6) 交给 fi_check.py 的 C2：同一份命令、同一份阈值
    cmp_out = tmp / "compare.json"
    base_cmd = f"{sys.executable} -I {COMPARE} --ref {ref} --candidate {same} -o {cmp_out}"
    ck = tmp / "check_pass.json"
    p = subprocess.run([sys.executable, "-I", str(CHECK), "--kernel", "demo_kernel",
                        "--compare-cmd", base_cmd, "--compare-out", str(cmp_out),
                        "--expected-elements", str(total), "--max-abs", "0", "--max-rel", "0",
                        "--work-dir", str(tmp / "ev_pass"), "-o", str(ck)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900)
    c2 = json.loads(ck.read_text(encoding="utf-8"))["checks"]["C2"]
    assert c2["pass"] is True, json.dumps(c2, ensure_ascii=False)[:1200]
    assert c2["metrics"]["checked_elements"] == total and c2["metrics"]["script_verdict"] is True, c2["metrics"]
    assert c2["thresholds"]["min_checked_elements"] == total and c2["thresholds"]["require_all_outputs"] is True

    ck2 = tmp / "check_fail.json"
    short_cmd = f"{sys.executable} -I {COMPARE} --ref {ref} --candidate {bad} --key aux"
    p = subprocess.run([sys.executable, "-I", str(CHECK), "--kernel", "demo_kernel",
                        "--compare-cmd", short_cmd,
                        "--expected-elements", str(total),
                        "--work-dir", str(tmp / "ev_fail"), "-o", str(ck2)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900)
    c2 = json.loads(ck2.read_text(encoding="utf-8"))["checks"]["C2"]
    assert c2["pass"] is False, json.dumps(c2, ensure_ascii=False)[:1200]
    assert c2["metrics"]["checked_elements"] == ref_aux.size, c2["metrics"]
    assert any("只比较了" in s for s in c2["suggestions"]), c2["suggestions"]

    cases = ["逐位一致 PASS", "改一个元素 FAIL+下标", "NaN FAIL", "shape/缺数组 拒绝比较",
             "--key/.npy", "atol/rtol 阈值口径", "fi_check C2 PASS", "fi_check C2 覆盖不足 FAIL"]
    print(f"{'用例':<28}{'结果':>8}")
    for c in cases:
        print(f"{c:<28}{'ok':>8}")
    if not a.keep_dir:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("产物保留在", tmp)
    print(f"C2 COMPARE CHECK PASS（{len(cases)} 个用例，比对元素 {total} 个）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
