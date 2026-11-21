#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMD MI308X C1 离线回归：从单份资源描述临时展开 L0/FI 画像并运行 fi_check.py。

回归的是这条已确认的缺陷：旧 C1 把正确插桩判成资源扰动（demo VGPR 53->63、scratch 0->92、waves/SIMD 9->8；
ring VGPR 77->84、waves/SIMD 6->5，两个都是输出逐位正确的构建，却都 FAIL）。
新判据把“结构性扰动”与“插桩固有代价”分开：正确插桩必须 PASS，注入的 scratch 故障与
记录宏加同步必须 FAIL，占用率崩塌（waves/SIMD 64->16）必须 FAIL。

不需要 GPU，也不需要 ROCm 工具链：场景和期望来自
tests/cases/amd_mi308x_resource_profiles.json（MI308X/gfx942 实测资源画像）。

用法：python3 -I tests/amd_c1_regression.py [--json out.json]
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
PROFILES = HERE / "cases" / "amd_mi308x_resource_profiles.json"
FI_CHECK = PKG / "weaveprof/pipeline/validate" / "tools" / "fi_check.py"
SCHEMA = "weaveprofiler.amd-resource-profiles"


def load_profiles(path=PROFILES):
    """读取并验证单文件资源描述，防止场景或资源引用在迁移中丢失。"""
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError("unexpected profile schema: %r" % (doc.get("schema"),))
    if doc.get("platform") != "amd" or doc.get("card") != "mi308x" or doc.get("arch") != "gfx942":
        raise ValueError("resource profile is not AMD MI308X/gfx942")
    targets = doc.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("targets must be an object")
    count = 0
    for target, target_spec in targets.items():
        cases = target_spec.get("cases")
        if not isinstance(cases, dict):
            raise ValueError("%s.cases must be an object" % target)
        for scenario, case in cases.items():
            count += 1
            if not case.get("kernel"):
                raise ValueError("%s_%s has no kernel" % (target, scenario))
            ref = case.get("resource_case")
            if ref is None:
                for side in ("l0", "fi"):
                    side_doc = case.get(side)
                    if not isinstance(side_doc, dict):
                        raise ValueError("%s_%s missing %s" % (target, scenario, side))
                    if case["kernel"] not in side_doc.get("kernels", {}):
                        raise ValueError("%s_%s kernel missing from %s" % (target, scenario, side))
            elif ref not in cases or ref == scenario:
                raise ValueError("%s_%s has invalid resource_case %r" % (target, scenario, ref))
    if count != doc.get("case_count"):
        raise ValueError("case_count %r does not match %d cases" % (doc.get("case_count"), count))
    return doc


def resolve_case(doc, target, scenario):
    """解析 resource_case 引用，但保留当前场景自己的期望。"""
    cases = doc["targets"][target]["cases"]
    if scenario not in cases:
        raise KeyError("%s_%s" % (target, scenario))
    requested = cases[scenario]
    resource_name = requested.get("resource_case", scenario)
    seen = []
    while True:
        if resource_name in seen:
            raise ValueError("resource_case cycle: %s" % " -> ".join(seen + [resource_name]))
        seen.append(resource_name)
        resource = cases[resource_name]
        parent = resource.get("resource_case")
        if parent is None:
            break
        resource_name = parent
    resolved = dict(requested)
    resolved["kernel"] = resource["kernel"]
    resolved["l0"] = resource["l0"]
    resolved["fi"] = resource["fi"]
    resolved["resource_case_resolved"] = resource_name
    return resolved


def iter_cases(doc):
    """按文件中的目标和场景顺序迭代，保证回归报告稳定。"""
    for target, target_spec in doc["targets"].items():
        for scenario in target_spec["cases"]:
            yield target, scenario, resolve_case(doc, target, scenario)


def materialize_resource_pair(doc, target, scenario, workdir):
    """仅在工作目录生成 fi_check 需要的临时 _l0/_fi 文件。"""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    case = resolve_case(doc, target, scenario)
    case_id = "%s_%s" % (target, scenario)
    paths = {}
    for side, suffix in (("l0", "_l0.json"), ("fi", "_fi.json")):
        side_doc = {
            "gfx": doc["arch"],
            "source": case[side]["source"],
            "kernel": case["kernel"],
            "provenance": case[side]["provenance"],
            "kernels": case[side]["kernels"],
        }
        path = workdir / (case_id + suffix)
        path.write_text(json.dumps(side_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths[side] = path
    return paths["l0"], paths["fi"]


def run_case(case_id, kernel, workdir, l0_res, fi_res, extra_args=()):
    out = workdir / (case_id + ".check.json")
    argv = [sys.executable, "-I", str(FI_CHECK), "--platform", "amd", "--arch", "gfx942",
            "--name", case_id, "--kernel", kernel,
            "--l0-amd-res", str(l0_res), "--fi-amd-res", str(fi_res),
            # 这些夹具是预先抽取的 C1 资源画像，instructions 只有 barrier/waitcnt，没有
            # timer_read/record_store 签名。本测试只回归资源/占用判据，因此对“探针可见性”
            # 显式 waiver；生产路径默认 require_fi_probe=true，证据缺失会 FAIL。
            "--allow-unverified-probe",
            "--work-dir", str(workdir / case_id), "-o", str(out)]
    argv.extend(str(x) for x in extra_args)
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
    if not out.exists():
        return {"verdict": "ERROR", "error": "rc=%s %s" % (p.returncode, p.stderr.strip()[:400])}
    d = json.loads(out.read_text(encoding="utf-8"))
    c1 = d["checks"]["C1"]
    metrics = c1["metrics"]
    occupancy = metrics.get("occupancy_effective") or {}
    return {"verdict": "PASS" if c1["pass"] else "FAIL",
            "fails": metrics.get("fails", []),
            "warnings": metrics.get("warnings", []),
            "register_file_only": occupancy.get("register_file_only_waves_per_simd"),
            "register_ceiling": occupancy.get("register_ceiling_waves_per_simd"),
            "waves_per_simd": metrics.get("waves_per_simd"),
            "occupancy_effective": metrics.get("occupancy_effective"),
            "vgpr": metrics.get("vgpr"),
            "scratch": metrics.get("scratch_bytes_per_workitem"),
            "rc": p.returncode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="把结果写成 JSON")
    ap.add_argument("--profiles", default=str(PROFILES), help="单文件资源描述路径")
    a = ap.parse_args()
    profiles = load_profiles(a.profiles)
    rows = []
    with tempfile.TemporaryDirectory(prefix="fi_amd_mi308x_") as td:
        wd = Path(td)
        for target, scenario, case in iter_cases(profiles):
            case_id = "%s_%s" % (target, scenario)
            materialized = wd / "materialized"
            l0_res, fi_res = materialize_resource_pair(profiles, target, scenario, materialized)
            r = run_case(case_id, case["kernel"], wd, l0_res, fi_res,
                         tuple(case.get("extra_args", ())))
            ok = r["verdict"] == case["expect_c1"]
            # 期望里给了 expect_warning_contains 就要求这些字串都出现在 warning 里（判据放宽的同时必须报数）
            missing_w = [s for s in case.get("expect_warning_contains", [])
                         if not any(s in w for w in (r.get("warnings") or []))]
            if missing_w:
                ok = False
                r["error"] = "warning 里缺少：" + "、".join(missing_w)
            rows.append({"case": case_id, "scenario": scenario, "target": target,
                         "expect": case["expect_c1"], "got": r["verdict"], "ok": ok,
                         "old_c1_verdict": case.get("old_c1_verdict"),
                         "old_c1_reasons": case.get("old_c1_reasons"),
                         "why": case["why"], "fails": r.get("fails"),
                         "warnings": r.get("warnings"), "vgpr": r.get("vgpr"),
                         "waves_per_simd": r.get("waves_per_simd"),
                         "occupancy_effective": r.get("occupancy_effective"),
                         "register_file_only_waves_per_simd": r.get("register_file_only"),
                         "register_ceiling_waves_per_simd": r.get("register_ceiling"),
                         "scratch": r.get("scratch"), "error": r.get("error")})
    print("%-52s%7s%6s%6s  判定依据" % ("用例", "旧 C1", "期望", "实得"))
    n_ok = 0
    for r in rows:
        n_ok += bool(r["ok"])
        detail = r.get("error") or ("; ".join(r["fails"][:2]) if r["fails"] else r["why"])
        w = r.get("waves_per_simd") or {}
        extra = ("  VGPR %s->%s waves/SIMD %s->%s" %
                 (r["vgpr"]["l0"], r["vgpr"]["fi"], w.get("l0"), w.get("fi"))) if r.get("vgpr") else ""
        occ = (r.get("occupancy_effective") or {}).get("waves_per_simd") or {}
        if occ.get("l0") is not None and (occ.get("l0"), occ.get("fi")) != (w.get("l0"), w.get("fi")):
            extra += "  有效 %s->%s" % (occ.get("l0"), occ.get("fi"))
        print("%-52s%7s%6s%6s  %s%s" %
              (r["case"], str(r["old_c1_verdict"]), r["expect"], r["got"],
               detail[:110], extra))
    fixed = [r["case"] for r in rows if r["old_c1_verdict"] == "FAIL" and r["expect"] == "PASS" and r["ok"]]
    caught = [r["case"] for r in rows if r["expect"] == "FAIL" and r["ok"]]
    print("\n%d/%d 个用例与期望一致" % (n_ok, len(rows)))
    print("修掉的假阳性（旧 FAIL -> 新 PASS）：%d 个 %s" % (len(fixed), fixed))
    print("仍然拦住的真实扰动：%d 个 %s" % (len(caught), caught))
    result = {"total": len(rows), "matched": n_ok,
              "false_positives_fixed": fixed, "still_caught": caught,
              "rows": rows, "profiles": str(Path(a.profiles).resolve()),
              "profile_schema": profiles["schema"], "card": profiles["card"],
              "arch": profiles["arch"],
              "provenance": profiles["provenance"]}
    if a.json:
        Path(a.json).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("写出", a.json)
    if n_ok != len(rows):
        print("AMD MI308X RESOURCE REGRESSION FAIL")
        return 1
    print("AMD MI308X RESOURCE REGRESSION PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
