#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""正确性证据链离线合同测试。

覆盖 compare_raw_arrays.py 与 validate_correctness.py 的证据约束：
  * 默认输出键集合必须完全相等，不能只比交集；
  * 全部键、全部元素、shape/dtype/元素数与 NaN/Inf 都检查；
  * input/output manifest 的字段、元素数和 SHA256 精确可复核；
  * compare.json 顶层唯一的 PASS 字段是布尔 ``pass``；
  * 失败不复用旧 PASS，结果绑定 contract/input/output SHA256 与 run ID；
  * required output 缺失 fail-closed。

用法：python3 -I tests/test_correctness_contract.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
COMPARE = PKG / "configs" / "templates" / "compare_raw_arrays.py"
VALIDATOR = PKG / "examples" / "host" / "profile_skeleton" / "validation" / "validate_correctness.py"


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_input_sha(entries):
    clean = [{k: e[k] for k in ("name", "shape", "dtype", "layout", "elements", "sha256")}
             for e in sorted(entries, key=lambda item: item["name"])]
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha_bytes(payload.encode("utf-8"))


def run_compare(argv):
    proc = subprocess.run([sys.executable, "-I", str(COMPARE), *map(str, argv)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    payload = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("COMPARE_JSON "):
            payload = json.loads(line[len("COMPARE_JSON "):])
            break
    return proc, payload


def pass_fields(doc):
    return sorted(key for key in doc if "pass" in key.lower())


class CompareEvidenceContractTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.tmp_obj = tempfile.TemporaryDirectory(prefix="weaveprofiler_correctness_")
        self.tmp = Path(self.tmp_obj.name)
        self.rng = np.random.default_rng(20261114)
        self.out = self.rng.standard_normal((8, 8)).astype(np.float32)
        self.aux = self.rng.standard_normal((4,)).astype(np.float32)
        self.ref = self.tmp / "ref.npz"
        np.savez(self.ref, output=self.out, aux=self.aux)
        self.contract = self.tmp / "CONTRACT.lock.json"
        self.contract.write_text(json.dumps({"run": {"run_id": "run-c2"}, "input": {"seed": 7}},
                                            sort_keys=True) + "\n", encoding="utf-8")
        self.contract_sha = sha_bytes(self.contract.read_bytes())
        input_bytes = self.rng.standard_normal((8, 8)).astype(np.float32).tobytes()
        self.input_entry = {
            "name": "x",
            "shape": [8, 8],
            "dtype": "float32",
            "layout": "row_major",
            "elements": 64,
            "sha256": sha_bytes(input_bytes),
        }
        self.input_manifest = self.tmp / "input_manifest.json"
        self.input_manifest.write_text(json.dumps({
            "schema": "weaveprofiler-input-manifest",
            "generator": "randn",
            "seed": 7,
            "rank": 0,
            "world_size": 1,
            "inputs": [self.input_entry],
            "input_sha256": canonical_input_sha([self.input_entry]),
        }, sort_keys=True), encoding="utf-8")

    def tearDown(self):
        self.tmp_obj.cleanup()

    def save_candidate(self, name, **arrays):
        path = self.tmp / name
        self.np.savez(path, **arrays)
        return path

    def test_default_rejects_different_output_key_sets_and_clears_old_pass(self):
        candidate = self.save_candidate("extra_key.npz", output=self.out, unexpected=self.aux)
        out_json = self.tmp / "compare.json"
        out_json.write_text('{"pass": true}\n', encoding="utf-8")
        proc, doc = run_compare(["--ref", self.ref, "--candidate", candidate, "-o", out_json])
        self.assertEqual(proc.returncode, 2, proc.stderr + proc.stdout)
        self.assertIsNotNone(doc)
        self.assertIs(doc["pass"], False)
        self.assertIn("输出键集合不一致", doc["error"])
        self.assertEqual(pass_fields(doc), ["pass"])
        self.assertIs(json.loads(out_json.read_text(encoding="utf-8"))["pass"], False)

    def test_dtype_shape_nan_and_required_outputs_fail_closed(self):
        dtype_bad = self.save_candidate("dtype.npz", output=self.out.astype(self.np.float64), aux=self.aux)
        proc, doc = run_compare(["--ref", self.ref, "--candidate", dtype_bad])
        self.assertEqual(proc.returncode, 2, proc.stdout)
        self.assertIn("dtype 不一致", doc["error"])

        shape_bad = self.save_candidate("shape.npz", output=self.out[:4], aux=self.aux)
        proc, doc = run_compare(["--ref", self.ref, "--candidate", shape_bad])
        self.assertEqual(proc.returncode, 2, proc.stdout)
        self.assertIn("shape 不一致", doc["error"])

        nan_out = self.out.copy()
        nan_out[2, 3] = float("nan")
        nan_bad = self.save_candidate("nan.npz", output=nan_out, aux=self.aux)
        proc, doc = run_compare(["--ref", self.ref, "--candidate", nan_bad,
                                 "--atol", "1e9", "--rtol", "1e9"])
        self.assertEqual(proc.returncode, 2, proc.stdout)
        self.assertIs(doc["pass"], False)
        self.assertGreaterEqual(doc["nonfinite"], 1)

        missing_required = self.save_candidate("missing_required.npz", output=self.out)
        proc, doc = run_compare(["--ref", self.ref, "--candidate", missing_required,
                                 "--required-key", "aux", "--key", "output"])
        self.assertEqual(proc.returncode, 2, proc.stdout)
        self.assertIn("required output", doc["error"])

        aux_bad = self.aux.copy()
        aux_bad[1] += 1.0
        required_bad = self.save_candidate("required_bad.npz", output=self.out.copy(),
                                           aux=aux_bad)
        proc, doc = run_compare(["--ref", self.ref, "--candidate", required_bad,
                                 "--required-key", "aux", "--key", "output"])
        self.assertEqual(proc.returncode, 2, proc.stdout)
        self.assertEqual(doc["first_mismatch"]["key"], "aux")

    def test_full_evidence_binds_manifests_hashes_run_and_contract(self):
        candidate = self.save_candidate("same.npz", output=self.out.copy(), aux=self.aux.copy())
        out_json = self.tmp / "compare.json"
        output_manifest = self.tmp / "output_manifest.json"
        proc, doc = run_compare([
            "--ref", self.ref, "--candidate", candidate,
            "--atol", "0", "--rtol", "0",
            "--contract", self.contract, "--run-id", "run-c2",
            "--rank", "0", "--world-size", "1",
            "--input-manifest", self.input_manifest,
            "--output-manifest", output_manifest,
            "--require-evidence", "-o", out_json,
        ])
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        self.assertIs(doc["pass"], True)
        self.assertIs(doc["evidence_complete"], True)
        self.assertEqual(doc["checked_elements"], self.out.size + self.aux.size)
        self.assertEqual(pass_fields(doc), ["pass"])
        self.assertEqual(doc["run_id"], "run-c2")
        self.assertEqual(doc["rank"], 0)
        self.assertEqual(doc["world_size"], 1)
        self.assertEqual(doc["contract_sha256"], self.contract_sha)
        self.assertEqual(doc["input_sha256"], canonical_input_sha([self.input_entry]))
        self.assertEqual(doc["binding"]["input_sha256"], doc["input_sha256"])
        self.assertEqual(doc["binding"]["reference_output_sha256"], doc["reference_output_sha256"])
        self.assertEqual(doc["binding"]["instrumented_output_sha256"], doc["instrumented_output_sha256"])
        self.assertEqual(doc["reference_output_sha256"], doc["instrumented_output_sha256"])

        manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["rank"], 0)
        self.assertEqual(manifest["world_size"], 1)
        self.assertEqual(manifest["keys"], ["aux", "output"])
        self.assertEqual([item["name"] for item in manifest["outputs"]], ["aux", "output"])
        for item in manifest["outputs"]:
            for arm in ("reference", "instrumented"):
                for field in ("shape", "dtype", "elements", "sha256"):
                    self.assertIn(field, item[arm], (item["name"], arm, field))
            self.assertEqual(item["elements"], item["reference"]["elements"])
            self.assertEqual(item["elements"], item["instrumented"]["elements"])

    def test_input_manifest_missing_layout_fails_and_does_not_reuse_pass(self):
        candidate = self.save_candidate("same2.npz", output=self.out.copy(), aux=self.aux.copy())
        broken = dict(self.input_entry)
        broken.pop("layout")
        broken_manifest = self.tmp / "broken_input_manifest.json"
        broken_manifest.write_text(json.dumps({
            "generator": "randn", "seed": 7, "inputs": [broken],
            "input_sha256": canonical_input_sha([self.input_entry]),
        }), encoding="utf-8")
        out_json = self.tmp / "compare.json"
        out_json.write_text('{"pass": true}\n', encoding="utf-8")
        proc, doc = run_compare(["--ref", self.ref, "--candidate", candidate,
                                 "--contract", self.contract, "--run-id", "run-c2",
                                 "--input-manifest", broken_manifest, "--require-evidence",
                                 "-o", out_json])
        self.assertEqual(proc.returncode, 2, proc.stdout)
        self.assertIs(doc["pass"], False)
        self.assertIn("layout", doc["error"])
        self.assertIs(json.loads(out_json.read_text(encoding="utf-8"))["pass"], False)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "需要 torch 验证默认正确性入口")
class ValidatorEvidenceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("validate_correctness_contract_under_test",
                                                      VALIDATOR)
        cls.module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cls.module)
        import torch
        cls.torch = torch

    def setUp(self):
        self.tmp_obj = tempfile.TemporaryDirectory(prefix="weaveprofiler_validator_")
        self.tmp = Path(self.tmp_obj.name)
        self.contract_path = self.tmp / "CONTRACT.lock.json"
        self.contract = {
            "run": {"run_id": "validator-run", "display_rank": 0, "world_size": 1},
            "input": {
                "seed": 41,
                "tensors": [{"name": "x", "shape": [3, 4], "dtype": "float32",
                             "layout": "row_major", "device": "cpu"}],
                "expected_outputs": [{"name": "out", "shape": [3, 4], "dtype": "float32"}],
            },
            "validation": {"correctness": {"atol": 0, "rtol": 0,
                                            "check_scope": "all_output_elements_all_ranks"}},
        }
        self.contract_path.write_text(json.dumps(self.contract, sort_keys=True) + "\n",
                                      encoding="utf-8")
        torch = self.torch
        self.inputs = {"x": torch.randn((3, 4), dtype=torch.float32)}

    def tearDown(self):
        self.tmp_obj.cleanup()

    def install_hooks(self, candidate_shift=0.0):
        torch = self.torch
        inputs = self.inputs
        module = self.module

        def setup():
            return {"torch": torch, "inputs": inputs}

        def launch(_mods, level, info):
            x = info["inputs"]["x"]
            out = x * 2.0
            if level == "full" and candidate_shift:
                out = out.clone()
                out[1, 2] += candidate_shift
            return {"outputs": {"out": out, "aux": out.sum().reshape(1)}, "inputs": info["inputs"]}

        module.setup_gpu_imports = setup
        module.launch_at_level = launch

    def test_validator_writes_full_manifests_and_replaces_stale_pass_on_failure(self):
        self.install_hooks()
        out = self.tmp / "validation" / "correctness.json"
        rc = self.module.main(["--contract", str(self.contract_path), "--out", str(out)])
        self.assertEqual(rc, 0)
        input_doc = json.loads((out.parent / "input_manifest.json").read_text(encoding="utf-8"))
        output_doc = json.loads((out.parent / "output_manifest.json").read_text(encoding="utf-8"))
        compare_doc = json.loads((out.parent / "compare.json").read_text(encoding="utf-8"))
        self.assertEqual(input_doc["generator"], "randn")
        self.assertEqual(input_doc["seed"], 41)
        self.assertEqual(input_doc["inputs"][0]["name"], "x")
        self.assertEqual(input_doc["inputs"][0]["shape"], [3, 4])
        self.assertEqual(input_doc["inputs"][0]["dtype"], "float32")
        self.assertEqual(input_doc["inputs"][0]["layout"], "row_major")
        self.assertRegex(input_doc["inputs"][0]["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(output_doc["keys"], ["aux", "out"])
        for item in output_doc["outputs"]:
            for field in ("name", "shape", "dtype", "elements"):
                self.assertIn(field, item)
            for arm in ("reference", "instrumented"):
                self.assertIn("sha256", item[arm])
        self.assertIs(compare_doc["pass"], True)
        self.assertEqual(pass_fields(compare_doc), ["pass"])
        self.assertEqual(compare_doc["run_id"], "validator-run")
        self.assertEqual(compare_doc["contract_sha256"], sha_bytes(self.contract_path.read_bytes()))
        self.assertEqual(compare_doc["input_sha256"], input_doc["input_sha256"])
        self.assertEqual(compare_doc["binding"]["input_sha256"], input_doc["input_sha256"])
        self.assertEqual(compare_doc["checked_elements"], 13)

        self.install_hooks(candidate_shift=1.0)
        rc = self.module.main(["--contract", str(self.contract_path), "--out", str(out)])
        self.assertEqual(rc, 4)
        failed = json.loads((out.parent / "compare.json").read_text(encoding="utf-8"))
        self.assertIs(failed["pass"], False)
        self.assertEqual(pass_fields(failed), ["pass"])
        self.assertNotEqual(failed["reference_output_sha256"], failed["instrumented_output_sha256"])

    def test_external_input_manifest_is_rebound_and_verified_against_returned_inputs(self):
        torch = self.torch
        module = self.module
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 11.0
        raw_manifest = module._input_manifest_from_tensors(
            torch, {"x": x}, seed=13, rank=0, world_size=1,
            run_id=None, contract_sha256=None)
        input_manifest = self.tmp / "external_input_manifest.json"
        input_manifest.write_text(json.dumps(raw_manifest), encoding="utf-8")
        module.setup_gpu_imports = lambda: {"torch": torch}
        module.launch_at_level = lambda _mods, _level, _info: {
            "outputs": {"out": x.cos()}, "inputs": {"x": x}}
        contract = {
            "run": {"run_id": "external-inputs-run", "display_rank": 0, "world_size": 1},
            "input": {"seed": 13, "expected_outputs": [{"name": "out"}]},
            "validation": {"correctness": {"atol": 0, "rtol": 0}},
        }
        contract_path = self.tmp / "CONTRACT.external_inputs.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        out = self.tmp / "external_inputs" / "correctness.json"
        rc = module.main(["--contract", str(contract_path), "--out", str(out),
                          "--input-manifest", str(input_manifest)])
        self.assertEqual(rc, 0)
        bound = json.loads((out.parent / "input_manifest.json").read_text(encoding="utf-8"))
        compare_doc = json.loads((out.parent / "compare.json").read_text(encoding="utf-8"))
        self.assertEqual(bound["input_sha256"], raw_manifest["input_sha256"])
        self.assertEqual(bound["run_id"], "external-inputs-run")
        self.assertEqual(bound["contract_sha256"], sha_bytes(contract_path.read_bytes()))
        self.assertEqual(compare_doc["input_sha256"], raw_manifest["input_sha256"])

    def test_launch_returned_inputs_are_accepted_without_prepare_hook(self):
        torch = self.torch
        module = self.module
        module.setup_gpu_imports = lambda: {"torch": torch}

        def launch(_mods, _level, info):
            x = info.get("inputs", {}).get("x")
            if x is None:
                x = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7.0
            return {"outputs": {"out": x.sin()}, "inputs": {"x": x}}

        module.launch_at_level = launch
        contract = {
            "run": {"run_id": "launch-inputs-run", "display_rank": 0, "world_size": 1},
            "input": {"seed": 9, "expected_outputs": [{"name": "out"}]},
            "validation": {"correctness": {"atol": 0, "rtol": 0}},
        }
        contract_path = self.tmp / "CONTRACT.launch_inputs.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        out = self.tmp / "launch_inputs" / "correctness.json"
        rc = module.main(["--contract", str(contract_path), "--out", str(out)])
        self.assertEqual(rc, 0)
        compare_doc = json.loads((out.parent / "compare.json").read_text(encoding="utf-8"))
        self.assertIs(compare_doc["pass"], True)
        input_doc = json.loads((out.parent / "input_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(input_doc["inputs"][0]["name"], "x")
        self.assertEqual(input_doc["inputs"][0]["shape"], [3, 4])
        self.assertEqual(compare_doc["input_sha256"], input_doc["input_sha256"])

    def test_one_sided_output_key_never_passes_and_manifest_retains_both_arms(self):
        torch = self.torch
        inputs = self.inputs
        module = self.module
        module.setup_gpu_imports = lambda: {"torch": torch, "inputs": inputs}

        def launch(_mods, level, info):
            out = info["inputs"]["x"] * 2.0
            outputs = {"out": out}
            if level == "off":
                outputs["aux_only_reference"] = out.sum().reshape(1)
            return {"outputs": outputs, "inputs": info["inputs"]}

        module.launch_at_level = launch
        contract = {
            "run": {"run_id": "one-sided-run", "display_rank": 0, "world_size": 1},
            "input": {"seed": 19, "expected_outputs": [{"name": "out"}]},
            "validation": {"correctness": {"atol": 0, "rtol": 0}},
        }
        contract_path = self.tmp / "CONTRACT.one_sided.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        out = self.tmp / "one_sided" / "correctness.json"
        rc = module.main(["--contract", str(contract_path), "--out", str(out)])
        self.assertEqual(rc, 4)
        compare_doc = json.loads((out.parent / "compare.json").read_text(encoding="utf-8"))
        self.assertIs(compare_doc["pass"], False)
        self.assertIs(compare_doc["key_sets_equal"], False)
        output_doc = json.loads((out.parent / "output_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(output_doc["keys"], ["aux_only_reference", "out"])
        aux = next(item for item in output_doc["outputs"] if item["name"] == "aux_only_reference")
        self.assertIn("reference", aux)
        self.assertNotIn("instrumented", aux)

    def test_required_output_missing_keeps_failure_manifests_and_fails_pass(self):
        self.install_hooks()
        contract = json.loads(json.dumps(self.contract))
        contract["input"]["expected_outputs"].append({"name": "required_missing",
                                                      "shape": [1], "dtype": "float32"})
        contract_path = self.tmp / "CONTRACT.required_missing.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        out = self.tmp / "required_missing" / "correctness.json"
        rc = self.module.main(["--contract", str(contract_path), "--out", str(out)])
        self.assertEqual(rc, 4)
        compare_doc = json.loads((out.parent / "compare.json").read_text(encoding="utf-8"))
        self.assertIs(compare_doc["pass"], False)
        self.assertIn("required_missing", compare_doc["missing_required_tensors"])
        output_doc = json.loads((out.parent / "output_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(output_doc["keys"], ["aux", "out"])
        self.assertTrue((out.parent / "input_manifest.json").is_file())
        self.assertEqual(compare_doc["binding"]["contract"]["path"], str(contract_path))
        self.assertEqual(compare_doc["binding"]["input_manifest"]["path"],
                         str(out.parent / "input_manifest.json"))

    def test_missing_required_output_and_input_evidence_fail_closed(self):
        torch = self.torch
        ref = {"out": torch.ones(2, 2)}
        good = {"out": torch.ones(2, 2)}
        bad_missing = {"different": torch.ones(2, 2)}
        result = self.module._compare_all_elements(
            torch, ref, good, 0.0, 0.0, ["out", "required"], rank=0, world_size=1)
        self.assertIs(result["all_passed"], False)
        self.assertEqual(result["missing_required_tensors"], ["required"])
        scalar_record = self.module._tensor_record(torch, torch.tensor(3.25), "scalar")
        self.assertEqual(scalar_record["shape"], [])
        self.assertEqual(scalar_record["elements"], 1)
        self.assertRegex(scalar_record["sha256"], r"^[0-9a-f]{64}$")
        result = self.module._compare_all_elements(
            torch, ref, bad_missing, 0.0, 0.0, ["out"], rank=0, world_size=1)
        self.assertIs(result["all_passed"], False)
        self.assertEqual(result["only_in_instrumented"], ["different"])
        self.assertEqual(result["only_in_reference"], ["out"])
        shape_result = self.module._compare_all_elements(
            torch, {"out": torch.ones(2, 2)}, {"out": torch.ones(2, 3)}, 0.0, 0.0, ["out"])
        self.assertIs(shape_result["all_passed"], False)
        self.assertIs(shape_result["per_tensor"][0]["shape_match"], False)
        dtype_result = self.module._compare_all_elements(
            torch, {"out": torch.ones(2, 2, dtype=torch.float32)},
            {"out": torch.ones(2, 2, dtype=torch.float64)}, 0.0, 0.0, ["out"])
        self.assertIs(dtype_result["all_passed"], False)
        self.assertIs(dtype_result["per_tensor"][0]["dtype_match"], False)
        nan_out = torch.ones(2, 2)
        nan_out[0, 1] = float("nan")
        nan_result = self.module._compare_all_elements(
            torch, {"out": torch.ones(2, 2)}, {"out": nan_out}, 1e9, 1e9, ["out"])
        self.assertIs(nan_result["all_passed"], False)
        self.assertGreaterEqual(nan_result["per_tensor"][0]["nonfinite_elements"], 1)

        # 无 prepare_inputs、无外部 manifest：error path 仍先删旧 compare，并写本轮 pass=false。
        module = self.module
        module.setup_gpu_imports = lambda: {"torch": torch}
        module.launch_at_level = lambda _mods, _level, _info: {"out": torch.ones(2, 2)}
        no_input_contract = {
            "run": {"run_id": "missing-input-run", "display_rank": 0, "world_size": 1},
            "input": {"seed": 5, "expected_outputs": [{"name": "out"}]},
            "validation": {"correctness": {"atol": 0, "rtol": 0}},
        }
        contract_path = self.tmp / "CONTRACT.no_input.json"
        contract_path.write_text(json.dumps(no_input_contract), encoding="utf-8")
        out = self.tmp / "missing_input" / "correctness.json"
        out.parent.mkdir(parents=True)
        (out.parent / "compare.json").write_text('{"pass": true}\n', encoding="utf-8")
        rc = module.main(["--contract", str(contract_path), "--out", str(out)])
        self.assertEqual(rc, 2)
        failed = json.loads((out.parent / "compare.json").read_text(encoding="utf-8"))
        self.assertIs(failed["pass"], False)
        self.assertIn("输入证据", failed["error"])
        self.assertFalse((out.parent / "output_manifest.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
