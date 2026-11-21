#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""randn 全量输出比对模板（fi_check.py C2 调用）。

被测程序应把参考输出与插桩输出分别保存为 .npy 或 .npz。输入必须是同一份 randn 数据，
输出必须保存完整张量，不能只存切片或汇总值。示例：

  python3 -I compare_raw_arrays.py --ref ref.npz --candidate fi.npz \
      --atol 0 --rtol 0 -o compare.json
  python3 -I compare_raw_arrays.py --ref ref.npy --candidate fi.npy --key output

默认要求两边输出键集合完全相等，并比较全部键、全部 rank/元素、shape、dtype 与元素数。
显式 --key 保留原有的指定键子集调用方式；G02 证据应使用默认的全输出模式。

完整证据模式可额外传入：
  --contract CONTRACT.lock.json --run-id RUN --input-manifest input_manifest.json \
  --require-evidence -o compare.json
此时 compare.json 的 binding 会绑定 contract SHA256、run ID、输入 SHA256 和两臂输出 SHA256。
没有 input manifest 的临时比对仍可运行，但 evidence_complete=false，不能作为交付证据。

输出最后一行是 COMPARE_JSON <单行 json>，同时可写 -o 文件。顶层唯一的 PASS 字段是
布尔值 ``pass``。每次调用先删除 -o 旧文件；失败时只留下本轮的 pass=false 结果，绝不
复用旧 PASS。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

INPUT_MANIFEST_SCHEMA = "weaveprofiler-input-manifest"
OUTPUT_MANIFEST_SCHEMA = "weaveprofiler-output-manifest"
COMPARE_SCHEMA = "weaveprofiler-compare"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array) -> str:
    import numpy as np
    return _sha256_bytes(np.ascontiguousarray(array).tobytes())


def _canonical_digest(records: Iterable[Dict[str, Any]]) -> str:
    ordered = sorted(records, key=lambda item: str(item.get("name", "")))
    payload = json.dumps(ordered, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _unlink(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def load(path: str):
    import numpy as np
    p = Path(path)
    if p.suffix == ".npy":
        return {"array": np.load(p, allow_pickle=False)}
    if p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    raise SystemExit(f"只支持 .npy/.npz：{p}")


def _load_sidecar_meta(path: str) -> Dict[str, Any]:
    p = Path(path)
    meta_path = Path(str(p) + ".meta.json")
    if not meta_path.is_file():
        return {}
    try:
        value = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _logical_dtype(meta: Dict[str, Any], key: str) -> Optional[str]:
    entry = (meta.get("arrays") or {}).get(key)
    if isinstance(entry, dict) and entry.get("logical_dtype"):
        return str(entry["logical_dtype"])
    return None


def _require_sha(value: Any, where: str) -> str:
    text = str(value or "").lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{where} 必须是 64 位小写十六进制 SHA256")
    return text


def _normalize_input_entries(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = manifest.get("inputs", manifest.get("tensors"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("input manifest 必须包含非空 inputs/tensors 数组")
    out: List[Dict[str, Any]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"input manifest inputs[{index}] 不是 object")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ValueError(f"input manifest inputs[{index}] 缺 name")
        shape = entry.get("shape")
        if not isinstance(shape, list) or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in shape):
            raise ValueError(f"input {name}: shape 必须是非负整数数组")
        dtype = str(entry.get("dtype") or entry.get("logical_dtype") or "").strip()
        if not dtype:
            raise ValueError(f"input {name}: 缺 dtype")
        layout = str(entry.get("layout") or "").strip()
        if not layout:
            raise ValueError(f"input {name}: 缺 layout")
        elements = entry.get("elements", entry.get("numel"))
        expected_elements = 1
        for dim in shape:
            expected_elements *= dim
        if not isinstance(elements, int) or isinstance(elements, bool) or elements != expected_elements:
            raise ValueError(f"input {name}: elements={elements!r} 与 shape 元素数 {expected_elements} 不一致")
        out.append({"name": name, "shape": list(shape), "dtype": dtype, "layout": layout,
                    "elements": elements, "sha256": _require_sha(entry.get("sha256"), f"input {name}.sha256")})
    names = [entry["name"] for entry in out]
    if len(names) != len(set(names)):
        raise ValueError("input manifest 的 input 名必须唯一")
    out = sorted(out, key=lambda item: item["name"])
    expected = _canonical_digest([{k: e[k] for k in ("name", "shape", "dtype", "layout", "elements", "sha256")}
                                  for e in out])
    declared = manifest.get("input_sha256")
    if declared is not None and _require_sha(declared, "input manifest input_sha256") != expected:
        raise ValueError("input manifest input_sha256 与 inputs 内容不一致")
    return out


def validate_input_manifest(manifest: Dict[str, Any], contract_sha256: Optional[str] = None,
                            run_id: Optional[str] = None) -> Dict[str, Any]:
    """校验输入 manifest，返回规范化副本。缺 seed/randn/layout/SHA 都 fail-closed。"""
    if not isinstance(manifest, dict):
        raise ValueError("input manifest 必须是 JSON object")
    generator = str(manifest.get("generator") or manifest.get("input_generator") or "").strip()
    if generator != "randn":
        raise ValueError("input manifest generator 必须是 randn")
    seed = manifest.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("input manifest seed 必须是整数")
    rank = manifest.get("rank", 0)
    world_size = manifest.get("world_size", 1)
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError("input manifest rank 必须是非负整数")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1 or rank >= world_size:
        raise ValueError("input manifest world_size 必须是正整数且 rank < world_size")
    entries = _normalize_input_entries(manifest)
    input_sha = _canonical_digest([{k: e[k] for k in ("name", "shape", "dtype", "layout", "elements", "sha256")}
                                   for e in entries])
    doc = {
        "schema": INPUT_MANIFEST_SCHEMA,
        "generator": "randn",
        "seed": seed,
        "rank": rank,
        "world_size": world_size,
        "inputs": entries,
        "input_sha256": input_sha,
    }
    for key in ("run_id", "contract_sha256"):
        if manifest.get(key) is not None:
            doc[key] = str(manifest[key])
    if contract_sha256 is not None:
        contract_sha256 = _require_sha(contract_sha256, "contract_sha256")
        if doc.get("contract_sha256") not in (None, contract_sha256):
            raise ValueError("input manifest contract_sha256 与当前 contract 不一致")
        doc["contract_sha256"] = contract_sha256
    if run_id is not None:
        if doc.get("run_id") not in (None, str(run_id)):
            raise ValueError("input manifest run_id 与当前 run 不一致")
        doc["run_id"] = str(run_id)
    return doc


def _array_record(meta: Dict[str, Any], key: str, array) -> Dict[str, Any]:
    import numpy as np
    arr = np.asarray(array)
    record = {
        "name": key,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "elements": int(arr.size),
        "sha256": _sha256_array(arr),
    }
    logical = _logical_dtype(meta, key)
    if logical:
        record["logical_dtype"] = logical
    return record


def _outputs_digest(records: Sequence[Dict[str, Any]]) -> str:
    return _canonical_digest([{k: record[k] for k in ("name", "shape", "dtype", "elements", "sha256")}
                              for record in records])


def build_output_manifest(ref: Dict[str, Any], got: Dict[str, Any], keys: Sequence[str],
                          *, ref_meta: Optional[Dict[str, Any]] = None,
                          candidate_meta: Optional[Dict[str, Any]] = None,
                          run_id: Optional[str] = None,
                          contract_sha256: Optional[str] = None,
                          input_sha256: Optional[str] = None,
                          rank: Optional[int] = None,
                          world_size: Optional[int] = None) -> Dict[str, Any]:
    """为两臂的全部输出生成精确清单；每个键都带 shape/dtype/elements/SHA256。"""
    ref_meta = ref_meta or {}
    candidate_meta = candidate_meta or {}
    outputs = []
    for key in sorted(keys):
        if key not in ref or key not in got:
            raise ValueError(f"output manifest 生成失败：{key} 未同时出现在 reference 与 instrumented")
        rr = _array_record(ref_meta, key, ref[key])
        ir = _array_record(candidate_meta, key, got[key])
        outputs.append({"name": key, "shape": rr["shape"], "dtype": rr["dtype"],
                        "logical_dtype": rr.get("logical_dtype") or ir.get("logical_dtype") or rr["dtype"],
                        "elements": rr["elements"], "sha256": rr["sha256"],
                        "reference": rr, "instrumented": ir})
    doc = {
        "schema": OUTPUT_MANIFEST_SCHEMA,
        "keys": [entry["name"] for entry in outputs],
        "outputs": outputs,
        "reference_outputs_sha256": _outputs_digest([e["reference"] for e in outputs]),
        "instrumented_outputs_sha256": _outputs_digest([e["instrumented"] for e in outputs]),
    }
    if run_id:
        doc["run_id"] = str(run_id)
    if contract_sha256:
        doc["contract_sha256"] = _require_sha(contract_sha256, "contract_sha256")
    if input_sha256:
        doc["input_sha256"] = _require_sha(input_sha256, "input_sha256")
    if rank is not None:
        doc["rank"] = int(rank)
    if world_size is not None:
        doc["world_size"] = int(world_size)
    return doc


def verify_output_manifest(manifest: Dict[str, Any], arrays: Dict[str, Any], arm: str,
                           meta: Optional[Dict[str, Any]] = None) -> List[str]:
    """可选用外部单臂 output manifest 复核实际数组；任何键/元素数/SHA 差异都返回错误。"""
    if not isinstance(manifest, dict):
        return ["output manifest 不是 object"]
    raw = manifest.get("outputs")
    if not isinstance(raw, list) or not raw:
        return ["output manifest 缺非空 outputs 数组"]
    meta = meta or {}
    actual_keys = set(arrays)
    declared: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict) or not entry.get("name"):
            errors.append(f"output manifest outputs[{index}] 缺 name")
            continue
        name = str(entry["name"])
        if name in declared:
            errors.append(f"output manifest 重复键 {name}")
        declared[name] = entry
    if actual_keys != set(declared):
        errors.append(f"{arm} output manifest 键集合与数组不一致："
                      f"extra={sorted(actual_keys - set(declared))} missing={sorted(set(declared) - actual_keys)}")
    for key in sorted(actual_keys & set(declared)):
        expected = declared[key]
        record = _array_record(meta, key, arrays[key])
        arm_data = expected.get(arm if arm in ("reference", "instrumented") else "instrumented")
        checks = arm_data if isinstance(arm_data, dict) else expected
        if list(expected.get("shape", [])) != record["shape"]:
            errors.append(f"{arm}.{key}: manifest shape 不一致")
        if str(expected.get("dtype", "")) != record["dtype"]:
            errors.append(f"{arm}.{key}: manifest dtype 不一致")
        if int(expected.get("elements", -1)) != record["elements"]:
            errors.append(f"{arm}.{key}: manifest elements 不一致")
        declared_sha = checks.get("sha256", expected.get("sha256"))
        if str(declared_sha or "").lower() != record["sha256"]:
            errors.append(f"{arm}.{key}: manifest SHA256 与实际元素不一致")
    return errors


def _json_scalar(value: Any) -> Any:
    import numpy as np
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.complexfloating, complex)):
        return {"real": _json_scalar(value.real), "imag": _json_scalar(value.imag)}
    return str(value)


def compare(ref: dict, got: dict, keys: Optional[List[str]], atol: float, rtol: float,
            *, required_keys: Optional[Sequence[str]] = None,
            require_exact_keys: Optional[bool] = None,
            ref_meta: Optional[Dict[str, Any]] = None,
            candidate_meta: Optional[Dict[str, Any]] = None,
            run_id: Optional[str] = None,
            contract_sha256: Optional[str] = None,
            input_sha256: Optional[str] = None,
            output_manifest_path: Optional[str] = None,
            rank: Optional[int] = None,
            world_size: Optional[int] = None) -> Dict[str, Any]:
    """全量比较两臂；默认输出键集合必须完全相等，绝不只比交集。"""
    import numpy as np
    ref_meta = ref_meta or {}
    candidate_meta = candidate_meta or {}
    ref_keys = set(ref)
    got_keys = set(got)
    if require_exact_keys is None:
        require_exact_keys = not bool(keys)
    if require_exact_keys and ref_keys != got_keys:
        raise SystemExit("输出键集合不一致；不能只比较交集："
                         f"only_reference={sorted(ref_keys - got_keys)} "
                         f"only_instrumented={sorted(got_keys - ref_keys)}")
    required = list(dict.fromkeys(str(k) for k in (required_keys or ())))
    if keys:
        # required key 即使没在 --key 里，也必须进入本轮逐元素比较，不能只检查存在性。
        selected = list(dict.fromkeys([str(k) for k in keys] + required))
    else:
        selected = sorted(ref_keys)
    missing_ref = sorted((set(selected) | set(required)) - ref_keys)
    missing_got = sorted((set(selected) | set(required)) - got_keys)
    if missing_ref or missing_got:
        raise SystemExit(f"缺数组或 required output：ref={missing_ref} candidate={missing_got}")
    if not selected:
        raise SystemExit("参考与候选没有输出数组；空输出不能作为正确性证据")

    max_abs = 0.0
    max_rel = 0.0
    nonfinite = 0
    checked = 0
    first = None
    rows: Dict[str, Dict[str, Any]] = {}
    all_ok = True
    for key in selected:
        a = np.asarray(ref[key])
        b = np.asarray(got[key])
        if a.shape != b.shape:
            raise SystemExit(f"{key}: shape 不一致 ref={a.shape} candidate={b.shape}；不能做全量比对")
        if a.dtype != b.dtype:
            raise SystemExit(f"{key}: dtype 不一致 ref={a.dtype} candidate={b.dtype}；禁止隐式转换后放行")
        if a.size != b.size:
            raise SystemExit(f"{key}: 元素数不一致 ref={a.size} candidate={b.size}")
        logical_a = _logical_dtype(ref_meta, key)
        logical_b = _logical_dtype(candidate_meta, key)
        if bool(logical_a) != bool(logical_b):
            raise SystemExit(f"{key}: logical dtype 元数据只出现在一边，无法证明两臂类型一致")
        if logical_a and logical_b and logical_a != logical_b:
            raise SystemExit(f"{key}: logical dtype 不一致 ref={logical_a} candidate={logical_b}")
        try:
            af = a.astype(np.float64, copy=False)
            bf = b.astype(np.float64, copy=False)
        except (TypeError, ValueError) as error:
            raise SystemExit(f"{key}: dtype={a.dtype} 不支持数值全量比较：{error}") from error
        finite = np.isfinite(af) & np.isfinite(bf)
        nonfinite_mask = np.logical_not(finite)
        n_nonfinite = int(np.size(a) - np.count_nonzero(finite))
        diff = np.abs(bf - af)
        diff[nonfinite_mask] = np.inf
        rel = diff / np.maximum(np.abs(af), 1e-12)
        rel[nonfinite_mask] = np.inf
        kmax_abs = float(np.max(diff)) if diff.size else 0.0
        kmax_rel = float(np.max(rel)) if rel.size else 0.0
        bad = nonfinite_mask | (diff > (atol + rtol * np.abs(af)))
        nbad = int(np.count_nonzero(bad))
        if nbad and first is None:
            flat = int(np.flatnonzero(bad.reshape(-1))[0])
            index = tuple(int(x) for x in np.unravel_index(flat, a.shape))
            first = {"key": key, "index": list(index),
                     "ref": _json_scalar(a[index]), "candidate": _json_scalar(b[index])}
        max_abs = max(max_abs, kmax_abs)
        max_rel = max(max_rel, kmax_rel)
        nonfinite += n_nonfinite
        checked += int(a.size)
        row = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "elements": int(a.size),
            "checked_elements": int(a.size),
            "sha256_ref": _sha256_array(a),
            "sha256_candidate": _sha256_array(b),
            "max_abs": kmax_abs,
            "max_rel": kmax_rel,
            "nonfinite": n_nonfinite,
            "mismatches": nbad,
        }
        if logical_a or logical_b:
            row["logical_dtype"] = logical_a or logical_b
        rows[key] = row
        all_ok = all_ok and n_nonfinite == 0 and nbad == 0

    passed = bool(all_ok and nonfinite == 0 and checked > 0
                  and max_abs <= atol and max_rel <= rtol)
    manifest = build_output_manifest(ref, got, selected, ref_meta=ref_meta,
                                     candidate_meta=candidate_meta, run_id=run_id,
                                     contract_sha256=contract_sha256, input_sha256=input_sha256,
                                     rank=rank, world_size=world_size)
    manifest["selection"] = "all_outputs" if not keys else "explicit_keys"
    manifest["key_sets_exact"] = ref_keys == got_keys
    result: Dict[str, Any] = {
        "schema": COMPARE_SCHEMA,
        "pass": passed,
        "keys": selected,
        "key_sets_exact": ref_keys == got_keys,
        "checked_elements": checked,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "nonfinite": nonfinite,
        "first_mismatch": first,
        "arrays": rows,
        "thresholds": {"atol": atol, "rtol": rtol},
        "method": "同一 randn 输入；完整 .npy/.npz 张量逐元素比较；全部键/rank/元素；max_rel 分母 max(abs(ref),1e-12)",
        "output_manifest": manifest,
    }
    if rank is not None:
        result["rank"] = int(rank)
        result["ranks_checked"] = [int(rank)]
    if world_size is not None:
        result["world_size"] = int(world_size)
    if run_id:
        result["run_id"] = str(run_id)
    if contract_sha256:
        result["contract_sha256"] = _require_sha(contract_sha256, "contract_sha256")
    if input_sha256:
        result["input_sha256"] = _require_sha(input_sha256, "input_sha256")
    result["reference_output_sha256"] = manifest["reference_outputs_sha256"]
    result["instrumented_output_sha256"] = manifest["instrumented_outputs_sha256"]
    if output_manifest_path:
        result["output_manifest_path"] = output_manifest_path
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="fi_check C2 全量张量比对模板")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--key", action="append", help="只比较该数组（兼容旧调用）；可重复，默认要求两边键集合完全相等")
    ap.add_argument("--required-key", action="append", default=[], help="必须存在的输出键；可重复")
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("-o", "--out")
    ap.add_argument("--contract", help="CONTRACT 文件；只用于记录/核对 SHA256")
    ap.add_argument("--run-id", help="本次运行 ID")
    ap.add_argument("--rank", type=int, help="当前 rank；用于 manifest/证据绑定")
    ap.add_argument("--world-size", type=int, help="world size；用于 manifest/证据绑定")
    ap.add_argument("--input-manifest", help="randn 输入 manifest（seed/shape/dtype/layout/SHA256）")
    ap.add_argument("--output-manifest", help="写出全部输出的精确 manifest")
    ap.add_argument("--ref-output-manifest", help="可选：按实际 ref 数组复核该 manifest")
    ap.add_argument("--candidate-output-manifest", help="可选：按实际 candidate 数组复核该 manifest")
    ap.add_argument("--require-evidence", action="store_true",
                    help="要求 run ID、contract SHA、input manifest 与输出 manifest 全部绑定")
    a = ap.parse_args()

    out_path = Path(a.out) if a.out else None
    default_manifest = Path(str(out_path) + ".output_manifest.json") if out_path else None
    output_manifest_path = Path(a.output_manifest) if a.output_manifest else default_manifest
    if out_path is not None and output_manifest_path is not None and out_path.resolve() == output_manifest_path.resolve():
        _unlink(out_path)
        print("COMPARE_ERROR --out 不能与 --output-manifest 指向同一文件", file=sys.stderr)
        return 2
    # 先删旧证据。即使本轮后续退出，也不会留下上一轮的 PASS。
    _unlink(out_path)
    _unlink(output_manifest_path)

    try:
        contract_sha = _sha256_file(Path(a.contract)) if a.contract else None
        if a.contract:
            if not Path(a.contract).is_file():
                raise ValueError(f"contract 不存在：{a.contract}")
            contract_sha = _sha256_file(Path(a.contract))
        input_sha = None
        input_doc = None
        if a.input_manifest:
            if not Path(a.input_manifest).is_file():
                raise ValueError(f"input manifest 不存在：{a.input_manifest}")
            raw_input = json.loads(Path(a.input_manifest).read_text(encoding="utf-8"))
            input_doc = validate_input_manifest(raw_input, contract_sha, a.run_id)
            input_sha = input_doc["input_sha256"]

        if a.rank is not None and a.rank < 0:
            raise ValueError("--rank 必须是非负整数")
        if a.world_size is not None and a.world_size < 1:
            raise ValueError("--world-size 必须是正整数")
        if a.rank is not None and a.world_size is not None and a.rank >= a.world_size:
            raise ValueError("--rank 必须小于 --world-size")
        ref = load(a.ref)
        got = load(a.candidate)
        ref_meta = _load_sidecar_meta(a.ref)
        candidate_meta = _load_sidecar_meta(a.candidate)
        if a.ref_output_manifest:
            errs = verify_output_manifest(json.loads(Path(a.ref_output_manifest).read_text(encoding="utf-8")),
                                          ref, "reference", ref_meta)
            if errs:
                raise ValueError("; ".join(errs))
        if a.candidate_output_manifest:
            errs = verify_output_manifest(json.loads(Path(a.candidate_output_manifest).read_text(encoding="utf-8")),
                                          got, "instrumented", candidate_meta)
            if errs:
                raise ValueError("; ".join(errs))

        result = compare(ref, got, a.key, a.atol, a.rtol, required_keys=a.required_key,
                         ref_meta=ref_meta, candidate_meta=candidate_meta,
                         run_id=a.run_id, contract_sha256=contract_sha, input_sha256=input_sha,
                         output_manifest_path=str(output_manifest_path) if output_manifest_path else None,
                         rank=a.rank, world_size=a.world_size)
        all_outputs_compared = bool(not a.key and result["key_sets_exact"])
        evidence_complete = bool(a.run_id and contract_sha and input_sha and all_outputs_compared)
        result["all_outputs_compared"] = all_outputs_compared
        result["evidence_complete"] = evidence_complete
        result["binding"] = {
            "run_id": a.run_id,
            "contract": {"path": str(a.contract) if a.contract else None, "sha256": contract_sha},
            "contract_sha256": contract_sha,
            "input_manifest": ({"path": str(a.input_manifest), "sha256": input_sha}
                               if a.input_manifest else None),
            "input_sha256": input_sha,
            "output_manifest": ({"path": str(output_manifest_path),
                                 "reference_outputs_sha256": result["reference_output_sha256"],
                                 "instrumented_outputs_sha256": result["instrumented_output_sha256"]}
                                if output_manifest_path else None),
            "reference_output_sha256": result["reference_output_sha256"],
            "instrumented_output_sha256": result["instrumented_output_sha256"],
        }
        if input_doc is not None:
            result["input_manifest"] = input_doc
        if a.require_evidence and not evidence_complete:
            result["pass"] = False
            result["evidence_error"] = ("require-evidence 需要 --run-id、--contract 与 --input-manifest，"
                                        "且必须比较全部输出而不是 --key 子集")
        if output_manifest_path:
            _atomic_write_json(output_manifest_path, result["output_manifest"])
        if out_path:
            _atomic_write_json(out_path, result)
    except (Exception, SystemExit) as error:  # noqa: BLE001 -- 不留下半截证据或旧 PASS
        error_text = str(error)
        failure = {
            "schema": COMPARE_SCHEMA,
            "pass": False,
            "evidence_complete": False,
            "error": error_text,
            "binding": {"run_id": a.run_id, "contract_sha256": None, "input_sha256": None},
            "thresholds": {"atol": a.atol, "rtol": a.rtol},
            "method": "比较失败：不产生部分通过，不复用旧 compare.json",
        }
        if out_path:
            _atomic_write_json(out_path, failure)
            _unlink(output_manifest_path)
        print("COMPARE_FAIL")
        print("COMPARE_JSON " + json.dumps(failure, ensure_ascii=False, separators=(",", ":")))
        print("COMPARE_ERROR " + error_text, file=sys.stderr)
        return 2

    line = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    print("COMPARE_" + ("PASS" if result["pass"] else "FAIL"))
    print("COMPARE_JSON " + line)
    if result.get("evidence_error"):
        print("COMPARE_ERROR " + result["evidence_error"], file=sys.stderr)
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
