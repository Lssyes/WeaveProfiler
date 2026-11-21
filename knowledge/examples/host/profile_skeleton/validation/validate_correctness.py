#!/usr/bin/env python3
"""validate_correctness.py（骨架版）— E23 / 门禁 G02：证明"插桩不改变计算结果"。

流程位置
  编排器 correctness 阶段的固定入口。它用 contract 里声明的输入生成器和种子造一份
  输入，分别在"插桩全开"和"插桩全关"两个编译档位上跑同一个 kernel，然后逐元素比对
  两边的**全部**输出。check_scope 必须是 all_output_elements_all_ranks，
  ``sample_only`` 必须是 false —— 抽样比对不能作为 G02 的证据。

谁来调 / CLI
  python3 -B validate_correctness.py --contract <lock> --out <run_root>/validation/correctness.json
  python3 -B validate_correctness.py --contract <lock> --out <p.json> --dry-run   # 纯 CPU 出计划
  退出码：0 通过 / 4 比对不通过 / 2 出错（含 contract 读不了、GPU 钩子未实现）。

证据链
  同目录固定写出 input_manifest.json、output_manifest.json、compare.json 与本次 --out。
  input manifest 记录 randn 的 seed、每个输入的 shape/dtype/layout/elements 与 SHA256；
  output manifest 精确列出两臂全部输出键、shape/dtype/elements 与 SHA256；compare.json
  顶层唯一的 PASS 字段是布尔值 ``pass``，并绑定 contract、输入、两臂输出 SHA256 与 run ID。
  每次运行先删除旧 compare.json；失败写出本轮的 pass=false，绝不复用旧结果。缺失任一
  required output、任一输出键只在单边出现、shape/dtype/元素数不一致、NaN/Inf 都会失败。

设计约束（照抄，别改）
  所有 torch / cutlass / 项目 harness 的 import 都必须写在 GPU 函数**内部**，这样
  ``--help`` / ``--dry-run`` / ``py_compile`` / 被别的模块 import 时都不需要 GPU。

接新 target 要改的地方
  1. 配置区 PROJECT_CORRECTNESS：三个编译档位名、默认容差/种子、输出张量名单；
  2. ``setup_gpu_imports()``：把你项目的 torch / kernel / harness 模块凑齐；
  3. ``launch_at_level()``：在指定档位编译并跑一次，返回 {张量名: tensor}。
  推荐返回 ``{"outputs": {...}, "inputs": {...}}``，这样 validator 能复核两臂确实使用
  同一份输入。返回旧式 {张量名: tensor} 仍兼容；若没有 prepare_inputs 钩子，会使用
  contract 中 row-major randn 输入规范生成一份输入并把同一 dict 放进 info["inputs"]。
  这两个钩子都没实现时会 raise NotImplementedError —— 这是故意的，
  **绝不允许**改成"拿不到就 return all_passed=True"。

真实参考实现
  DSA-learn/mk/h20/cutedsl/deepseek_v4_pro/decode/profile/validation/validate_correctness.py
    - _launch_at_level(): 第 77-100 行（harness.load_bundle / R.alloc / _make_prof_tensors /
      DecodeMKInstr(MK.ALL_PHASES, prof_level=level) / env_compat.to_cute / cute.compile）
    - run_correctness(): 第 125-158 行（PROF_OFF 当参考、PROF_FULL 当被测）
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_THIS = Path(__file__).resolve()
_VALIDATION = _THIS.parent
_PROFILE_ROOT = _VALIDATION.parent
_VERSION_ROOT = _PROFILE_ROOT.parent

if str(_PROFILE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROFILE_ROOT))

SCHEMA = "fast-instrument-correctness/v1"
INPUT_MANIFEST_SCHEMA = "weaveprofiler-input-manifest"
OUTPUT_MANIFEST_SCHEMA = "weaveprofiler-output-manifest"
COMPARE_SCHEMA = "weaveprofiler-compare"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REAL_IMPL = ("DSA-learn/mk/h20/cutedsl/deepseek_v4_pro/decode/profile/validation/"
              "validate_correctness.py")

# ======================================================================
# ===== 接新 target 时只需要改这里 (PROJECT CONFIG) =====
# ======================================================================
PROJECT_CORRECTNESS: Dict[str, Any] = {
    # 1) ABI 兜底路径（相对 profile/）。真正的路径优先从 contract 读。
    "abi_relpath": "device_record/abi/record_abi.example.json",

    # 2) 三个编译档位在你项目里的名字。key 固定 off / entry_only / full。
    #    正确性比对只用 off（参考）和 full（被测）。
    #    真实项目是 cutlass Constexpr：PROF_OFF / PROF_ENTRY_ONLY / PROF_FULL。
    "level_names": {"off": "off", "entry_only": "entry_only", "full": "full"},

    # 3) 容差与种子的兜底值。contract 里 validation.correctness.* / input.seed 优先。
    #    这里的默认值抄的是真实项目（atol=rtol=0.01，seed=20240917）。
    "default_atol": 0.01,
    "default_rtol": 0.01,
    "default_seed": 20240917,
    "default_input_generator": "randn",

    # 4) 必须比对的输出张量名。留空 = 比对 launch_at_level 返回的全部张量。
    #    contract 的 input.expected_outputs / validation.correctness.required_output_tensors
    #    也会并入此名单。required output 缺失必须 FAIL，不能靠"交集"绕过。
    "required_output_tensors": (),

    # 5) 是否额外和项目 golden 对一遍。G02 只 gate"插桩前后是否一致"，
    #    kernel 本身的精度（vs golden）记录但不 gate —— 这是真实项目的既定口径。
    "compare_against_golden": False,
}
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


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, where: str) -> str:
    text = str(value or "").lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{where} 必须是 64 位小写十六进制 SHA256")
    return text


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


def _unlink(path: Optional[Path], preserve: Optional[Path] = None) -> None:
    if path is None:
        return
    if preserve is not None and path.resolve() == preserve.resolve():
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _evidence_paths(out_path: Path) -> Dict[str, Path]:
    """固定证据名；--out 通常就是 correctness.json。"""
    if out_path.name in ("compare.json", "input_manifest.json", "output_manifest.json"):
        raise ValueError("--out 不能占用固定证据文件名 compare.json/input_manifest.json/output_manifest.json")
    parent = out_path.parent
    return {
        "correctness": out_path,
        "compare": out_path if out_path.name == "compare.json" else parent / "compare.json",
        "input_manifest": parent / "input_manifest.json",
        "output_manifest": parent / "output_manifest.json",
    }


def _clear_stale_evidence(out_path: Path, preserve_input: Optional[Path] = None) -> Dict[str, Path]:
    paths = _evidence_paths(out_path)
    for name in ("correctness", "compare", "output_manifest"):
        _unlink(paths[name])
    _unlink(paths["input_manifest"], preserve=preserve_input)
    return paths


def contract_paths(contract: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """从 contract 解析出这次比对需要的路径与拓扑；contract 缺项时退回骨架默认。"""
    contract = contract if isinstance(contract, dict) else {}
    abi = first(contract, ("profile_selection.abi.absolute_path", "profile_selection.abi",
                           "profile_selection.abi_path"))
    abi_path = Path(abi) if abi else _PROFILE_ROOT / PROJECT_CORRECTNESS["abi_relpath"]
    rank = first(contract, ("run.display_rank", "capture.display_rank", "display_rank"), 0)
    world = first(contract, ("run.world_size", "capture.world_size", "world_size"), 1)
    selected = first(contract, ("capture.software_sm_scope.selected_physical_smids",
                                "capture.selected_physical_smids"), [])
    run_id = first(contract, ("run.run_id", "artifacts.run_id", "run_id"))
    return {"abi_path": abi_path, "rank": int(rank), "world_size": max(1, int(world)),
            "selected_physical_smids": list(selected), "run_id": run_id}


def _contract_correctness(contract: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    off_name = PROJECT_CORRECTNESS["level_names"]["off"]
    defaults = {"input_generator": PROJECT_CORRECTNESS["default_input_generator"],
                "check_scope": "all_output_elements_all_ranks",
                "atol": PROJECT_CORRECTNESS["default_atol"],
                "rtol": PROJECT_CORRECTNESS["default_rtol"],
                "seed": PROJECT_CORRECTNESS["default_seed"],
                "reference": "{} (instrumentation disabled)".format(off_name)}
    if not isinstance(contract, dict):
        return defaults
    val = dig(contract, "validation.correctness", {})
    if isinstance(val, dict):
        for key in ("input_generator", "check_scope", "atol", "rtol", "reference"):
            if val.get(key) is not None:
                defaults[key] = val[key]
    seed = first(contract, ("input.seed", "run.seed", "validation.correctness.seed"))
    if isinstance(seed, int) and not isinstance(seed, bool):
        defaults["seed"] = seed
    return defaults


def _expected_output_names(contract: Optional[Dict[str, Any]]) -> List[str]:
    names = {str(x) for x in PROJECT_CORRECTNESS.get("required_output_tensors", ()) if str(x)}
    if isinstance(contract, dict):
        groups = [
            dig(contract, "input.expected_outputs", []),
            dig(contract, "validation.correctness.required_output_tensors", []),
            dig(contract, "validation.correctness.expected_outputs", []),
        ]
        for group in groups:
            if isinstance(group, dict):
                if group.get("name"):
                    names.add(str(group["name"]))
                elif isinstance(group.get("tensors"), list) or isinstance(group.get("outputs"), list):
                    nested = group.get("tensors", group.get("outputs"))
                    for item in nested:
                        if isinstance(item, dict) and item.get("name"):
                            names.add(str(item["name"]))
                        elif isinstance(item, str) and item:
                            names.add(item)
                else:
                    names.update(str(name) for name in group if str(name))
                continue
            if not isinstance(group, list):
                continue
            for item in group:
                if isinstance(item, dict) and item.get("name"):
                    names.add(str(item["name"]))
                elif isinstance(item, str) and item:
                    names.add(item)
    return sorted(names)


def _expected_input_names(contract: Optional[Dict[str, Any]]) -> List[str]:
    group = dig(contract, "input.tensors", []) if isinstance(contract, dict) else []
    if isinstance(group, dict):
        return sorted(str(name) for name in group if str(name))
    if not isinstance(group, list):
        return []
    return sorted(str(item["name"]) for item in group
                  if isinstance(item, dict) and item.get("name"))


def _input_manifest_candidates(contract: Optional[Dict[str, Any]],
                               info: Dict[str, Any]) -> List[Path]:
    values = [info.get("input_manifest_path")]
    if isinstance(contract, dict):
        values.extend([
            dig(contract, "validation.correctness.input_manifest"),
            dig(contract, "input.manifest"),
            dig(contract, "input.manifest_path"),
        ])
    out: List[Path] = []
    for value in values:
        if value:
            path = Path(str(value))
            if path not in out:
                out.append(path)
    return out


def _plan(contract: Optional[Dict[str, Any]], info: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _contract_correctness(contract)
    names = PROJECT_CORRECTNESS["level_names"]
    run_id = info.get("run_id")
    contract_sha = info.get("contract_sha256")
    return {
        "schema": SCHEMA,
        "input_generator": cfg["input_generator"],
        # 硬写死 all_*：这是 G02 的验收口径，不允许被 contract 降级成抽样。
        "check_scope": "all_output_elements_all_ranks",
        "reference": cfg["reference"],
        "comparison": ("elementwise instrumented({full}) vs reference({off})"
                       .format(full=names["full"], off=names["off"])),
        "atol": float(cfg["atol"]), "rtol": float(cfg["rtol"]), "seed": int(cfg["seed"]),
        "rank": int(info["rank"]), "world_size": int(info["world_size"]),
        "abi_path": str(info["abi_path"]),
        "run_id": run_id,
        "contract_sha256": contract_sha,
        "required_output_tensors": _expected_output_names(contract),
        "input_manifest_required_fields": ["generator=randn", "seed", "inputs[].name",
                                           "inputs[].shape", "inputs[].dtype", "inputs[].layout",
                                           "inputs[].elements", "inputs[].sha256"],
        "output_manifest_required_fields": ["keys", "outputs[].name", "outputs[].shape",
                                            "outputs[].dtype", "outputs[].elements",
                                            "reference.sha256", "instrumented.sha256"],
        "sample_only": False,
    }


# ----------------------------------------------------------------------
# ===== 项目钩子（GPU 侧）=====
# 这两个函数是"只有你的项目知道怎么做"的部分。没实现就 raise，绝不假装通过。
# ----------------------------------------------------------------------
def setup_gpu_imports() -> Dict[str, Any]:
    """把 GPU 侧需要的模块凑成一个 dict 返回（torch / kernel / harness / ...）。

    import 必须发生在这个函数体内，保证 CPU-only 主机上 --help/--dry-run 仍可用。
    """
    raise NotImplementedError(
        "接入新 target 时请在此实现 setup_gpu_imports()：import torch 与你的 kernel/"
        "harness 模块并返回 {'torch': torch, ...}。"
        "参考真实实现 {}（其 run_correctness 通过 rp._setup_gpu_imports() 取得 "
        "torch/cute/cuda_drv/R/4_validate/harness/env_compat/DecodeMKInstr/MK/wr）:129".format(_REAL_IMPL))


def launch_at_level(mods: Dict[str, Any], level_name: str,
                    info: Dict[str, Any]) -> Dict[str, Any]:
    """在指定编译档位跑一次目标 kernel，返回输出张量字典。

    推荐返回 {"outputs": {名字: tensor}, "inputs": {名字: tensor}}；旧式直接返回
    {名字: tensor} 仍兼容。要点：
      * 使用 info["inputs"]（若存在）作为唯一输入来源，两个档位完全一致；
      * profiler buffer / counter / 溢出标志在计时区之外清零；
      * 跑完必须 synchronize，否则比对到的是未完成的显存。
    """
    raise NotImplementedError(
        "接入新 target 时请在此实现 launch_at_level(mods, {!r}, info)：编译并跑一次目标 "
        "kernel，返回输出张量字典。参考真实实现 {}:77-100"
        "（harness.load_bundle -> R.alloc -> _make_prof_tensors -> "
        "DecodeMKInstr(MK.ALL_PHASES, prof_level=level) -> env_compat.to_cute -> "
        "cute.compile -> zero_() -> fn() -> torch.cuda.synchronize -> R.out_map）"
        .format(level_name, _REAL_IMPL))


# ----------------------------------------------------------------------
# 输入 manifest
# ----------------------------------------------------------------------
def _validate_input_entries(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("input manifest 必须包含非空 inputs/tensors 数组")
    entries: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"input manifest inputs[{index}] 不是 object")
        name = str(item.get("name") or "").strip()
        if not name:
            raise ValueError(f"input manifest inputs[{index}] 缺 name")
        shape = item.get("shape")
        if not isinstance(shape, list) or any(not isinstance(x, int) or isinstance(x, bool) or x < 0
                                              for x in shape):
            raise ValueError(f"input {name}: shape 必须是非负整数数组")
        dtype = str(item.get("dtype") or item.get("logical_dtype") or "").strip()
        if not dtype:
            raise ValueError(f"input {name}: 缺 dtype")
        layout = str(item.get("layout") or "").strip()
        if not layout:
            raise ValueError(f"input {name}: 缺 layout")
        elements = item.get("elements", item.get("numel"))
        expected = 1
        for dim in shape:
            expected *= dim
        if not isinstance(elements, int) or isinstance(elements, bool) or elements != expected:
            raise ValueError(f"input {name}: elements={elements!r} 与 shape 元素数 {expected} 不一致")
        entries.append({"name": name, "shape": list(shape), "dtype": dtype, "layout": layout,
                        "elements": elements,
                        "sha256": _require_sha(item.get("sha256"), f"input {name}.sha256")})
    names = [entry["name"] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("input manifest 的 input 名必须唯一")
    return sorted(entries, key=lambda entry: entry["name"])


def validate_input_manifest(manifest: Dict[str, Any], *, expected_seed: Optional[int] = None,
                            expected_rank: Optional[int] = None,
                            expected_world_size: Optional[int] = None) -> Dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("input manifest 必须是 JSON object")
    generator = str(manifest.get("generator") or manifest.get("input_generator") or "").strip()
    if generator != "randn":
        raise ValueError("input manifest generator 必须是 randn")
    seed = manifest.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("input manifest seed 必须是整数")
    if expected_seed is not None and seed != int(expected_seed):
        raise ValueError(f"input manifest seed={seed} 与 contract seed={expected_seed} 不一致")
    rank = manifest.get("rank", 0)
    world = manifest.get("world_size", 1)
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise ValueError("input manifest rank 必须是整数")
    if not isinstance(world, int) or isinstance(world, bool) or world < 1:
        raise ValueError("input manifest world_size 必须是正整数")
    if expected_rank is not None and rank != int(expected_rank):
        raise ValueError(f"input manifest rank={rank} 与当前 rank={expected_rank} 不一致")
    if expected_world_size is not None and world != int(expected_world_size):
        raise ValueError(f"input manifest world_size={world} 与当前 world_size={expected_world_size} 不一致")
    entries = _validate_input_entries(manifest.get("inputs", manifest.get("tensors")))
    digest = _canonical_digest([{k: e[k] for k in ("name", "shape", "dtype", "layout", "elements", "sha256")}
                                for e in entries])
    if manifest.get("input_sha256") is not None:
        declared = _require_sha(manifest.get("input_sha256"), "input manifest input_sha256")
        if declared != digest:
            raise ValueError("input manifest input_sha256 与 inputs 内容不一致")
    doc = dict(manifest)
    doc.update({"schema": INPUT_MANIFEST_SCHEMA, "generator": "randn", "seed": seed,
                "rank": rank, "world_size": world, "inputs": entries, "input_sha256": digest})
    return doc


def _tensor_layout(tensor: Any) -> str:
    if bool(tensor.is_contiguous()):
        return "row_major"
    return "strided"


def _tensor_record(torch, tensor: Any, name: str) -> Dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name}: 输出/输入必须是 torch.Tensor，实际 {type(tensor).__name__}")
    cpu = tensor.detach().cpu().contiguous()
    byte_source = cpu.reshape(1) if cpu.dim() == 0 else cpu
    raw = byte_source.view(torch.uint8).reshape(-1).numpy().tobytes()
    shape = [int(x) for x in tensor.shape]
    elements = int(tensor.numel())
    record = {
        "name": name,
        "shape": shape,
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "layout": _tensor_layout(tensor),
        "strides": [int(x) for x in tensor.stride()],
        "elements": elements,
        "sha256": _sha256_bytes(raw),
    }
    return record


def _input_manifest_from_tensors(torch, inputs: Mapping[str, Any], *, seed: int, rank: int,
                                 world_size: int, run_id: Optional[str],
                                 contract_sha256: Optional[str]) -> Dict[str, Any]:
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("randn 输入集合为空；不能生成 input manifest")
    entries = [_tensor_record(torch, inputs[name], str(name)) for name in sorted(inputs)]
    digest = _canonical_digest([{k: e[k] for k in ("name", "shape", "dtype", "layout", "elements", "sha256")}
                                for e in entries])
    doc: Dict[str, Any] = {
        "schema": INPUT_MANIFEST_SCHEMA,
        "generator": "randn",
        "seed": int(seed),
        "rank": int(rank),
        "world_size": int(world_size),
        "inputs": entries,
        "input_sha256": digest,
    }
    if run_id:
        doc["run_id"] = str(run_id)
    if contract_sha256:
        doc["contract_sha256"] = _require_sha(contract_sha256, "contract_sha256")
    return doc


def _materialize_contract_inputs(torch, contract: Optional[Dict[str, Any]], seed: int) -> Optional[Dict[str, Any]]:
    """兼容路径：contract 有完整 row-major randn 输入规范时生成一次共享输入。"""
    specs = dig(contract, "input.tensors", []) if isinstance(contract, dict) else []
    if isinstance(specs, dict):
        normalized_specs = []
        for name, spec in specs.items():
            if not isinstance(spec, dict):
                raise ValueError(f"contract input.tensors[{name}] 必须是 object")
            item = dict(spec)
            item.setdefault("name", str(name))
            normalized_specs.append(item)
        specs = normalized_specs
    if not isinstance(specs, list) or not specs:
        return None
    device = first(contract, ("input.device", "run.device"), None)
    if device is None:
        device = "cuda" if bool(getattr(torch, "cuda", None) and torch.cuda.is_available()) else "cpu"
    result: Dict[str, Any] = {}
    torch.manual_seed(seed)
    for spec in specs:
        if not isinstance(spec, dict) or not spec.get("name"):
            raise ValueError("contract input.tensors 每项必须有 name")
        shape = spec.get("shape")
        dtype_name = str(spec.get("dtype") or "")
        layout = str(spec.get("layout") or "row_major")
        if not isinstance(shape, list) or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in shape):
            raise ValueError(f"contract input {spec['name']}: shape 非法")
        if layout not in ("row_major", "contiguous"):
            raise ValueError(f"contract input {spec['name']}: 自动生成只支持 row_major；"
                             "其他 layout 请提供 prepare_inputs 钩子")
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"contract input {spec['name']}: torch 不支持 dtype={dtype_name}")
        result[str(spec["name"])] = torch.randn(tuple(shape), dtype=dtype, device=device)
    return result


def _prepare_inputs(torch, mods: Dict[str, Any], contract: Optional[Dict[str, Any]],
                    info: Dict[str, Any], seed: int) -> Tuple[Optional[Dict[str, Any]], str]:
    """返回 (实际输入, provenance)。外部 manifest 与 tensor 输入至少有一个。"""
    for source in ("inputs", "input_tensors"):
        candidate = info.get(source)
        if isinstance(candidate, Mapping) and candidate:
            return dict(candidate), f"info.{source}"
    for source in ("inputs", "input_tensors"):
        candidate = mods.get(source)
        if isinstance(candidate, Mapping) and candidate:
            return dict(candidate), f"mods.{source}"

    harness = mods.get("harness")
    for source in ("prepare_inputs", "make_inputs"):
        fn = mods.get(source) or getattr(harness, source, None)
        if callable(fn):
            torch.manual_seed(seed)
            candidate = fn(mods, info, seed)
            if not isinstance(candidate, Mapping) or not candidate:
                raise ValueError(f"{source}() 必须返回非空 {{名字: torch.Tensor}}")
            return dict(candidate), f"hook.{source}"

    materialized = _materialize_contract_inputs(torch, contract, seed)
    if materialized:
        return materialized, "contract.input.tensors"
    return None, "external_input_manifest"


# ----------------------------------------------------------------------
# 通用比对
# ----------------------------------------------------------------------
def _compare_all_elements(torch, ref_map, test_map, atol, rtol,
                          required_names: Optional[Sequence[str]] = None, *,
                          rank: Optional[int] = None,
                          world_size: Optional[int] = None) -> Dict[str, Any]:
    """逐元素比对全部输出张量。任何一个缺失/多出/shape/dtype 不符或元素不过，整体不过。"""
    ref_map = dict(ref_map or {})
    test_map = dict(test_map or {})
    per_tensor: List[Dict[str, Any]] = []
    ref_keys = set(ref_map)
    test_keys = set(test_map)
    only_ref = sorted(ref_keys - test_keys)
    only_test = sorted(test_keys - ref_keys)
    required = sorted({str(name) for name in (required_names or ()) if str(name)})
    missing_required = sorted(name for name in required if name not in ref_keys or name not in test_keys)
    all_ok = not (only_ref or only_test or missing_required) and bool(ref_keys) and ref_keys == test_keys

    # 绝不取交集后放行：键集合不等时逐项记录，整体 FAIL。键集合相等时全部参与比较。
    for key in sorted(ref_keys if ref_keys == test_keys else (ref_keys & test_keys)):
        ref = ref_map[key]
        test = test_map[key]
        ref_shape = [int(x) for x in ref.shape]
        test_shape = [int(x) for x in test.shape]
        ref_dtype = str(ref.dtype).replace("torch.", "")
        test_dtype = str(test.dtype).replace("torch.", "")
        ref_numel = int(ref.numel())
        test_numel = int(test.numel())
        shape_ok = ref_shape == test_shape
        dtype_ok = ref_dtype == test_dtype
        numel_ok = ref_numel == test_numel
        finite = False
        max_abs = 0.0
        max_rel = 0.0
        mismatched = ref_numel if not (shape_ok and dtype_ok and numel_ok) else 0
        nonfinite = ref_numel if not (shape_ok and dtype_ok and numel_ok) else 0
        if shape_ok and dtype_ok and numel_ok and ref_numel > 0:
            try:
                ref64 = ref.detach().to(torch.float64)
                test64 = test.detach().to(torch.float64)
                finite = bool(torch.isfinite(ref64).all()) and bool(torch.isfinite(test64).all())
                diff = (ref64 - test64).abs()
                max_abs = float(diff.max().item()) if diff.numel() else 0.0
                denom = ref64.abs().clamp_min(1e-12)
                max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
                nonfinite = int((~torch.isfinite(ref64) | ~torch.isfinite(test64)).sum().item())
                mismatched = int((diff > (atol + rtol * denom)).sum().item())
            except (RuntimeError, TypeError, ValueError) as error:
                shape_ok = False
                max_abs = 0.0
                mismatched = ref_numel
                nonfinite = ref_numel
                per_tensor.append({"name": key, "error": str(error)})
        ok = (shape_ok and dtype_ok and numel_ok and ref_numel > 0 and finite
              and nonfinite == 0 and mismatched == 0)
        all_ok = all_ok and ok
        per_tensor.append({
            "name": key,
            "shape_reference": ref_shape,
            "shape_instrumented": test_shape,
            "dtype_reference": ref_dtype,
            "dtype_instrumented": test_dtype,
            "numel_reference": ref_numel,
            "numel_instrumented": test_numel,
            "shape_match": shape_ok,
            "dtype_match": dtype_ok,
            "numel_match": numel_ok,
            "numel": ref_numel,
            "finite": finite,
            "nonfinite_elements": nonfinite,
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "mismatched_elements": mismatched,
            "passed": ok,
        })
    if not ref_keys:
        all_ok = False
    return {
        "all_passed": bool(all_ok),
        "compared_tensors": len(ref_keys & test_keys),
        "output_keys_reference": sorted(ref_keys),
        "output_keys_instrumented": sorted(test_keys),
        "output_key_sets_equal": ref_keys == test_keys,
        "required_output_tensors": required,
        "missing_required_tensors": missing_required,
        "only_in_reference": only_ref,
        "only_in_instrumented": only_test,
        "rank": rank,
        "world_size": world_size,
        "check_scope": "all_output_elements_all_ranks",
        "sample_only": False,
        "per_tensor": per_tensor,
    }


def _validate_output_manifest(torch, doc: Dict[str, Any], ref_map: Mapping[str, Any],
                              test_map: Mapping[str, Any], required: Sequence[str]) -> None:
    """校验 manifest 结构、两臂实际记录和摘要；required/键集合失败交给 compare PASS 判定。

    失败运行也必须能留下可审计 manifest，所以这里不因 required output 缺失而丢掉文件；
    但每个实际出现过的输出都必须有 shape/dtype/elements/SHA256，任何伪造/错配都会抛错。
    """
    if not isinstance(doc, dict):
        raise ValueError("output manifest 不是 object")
    keys = doc.get("keys")
    outputs = doc.get("outputs")
    if not isinstance(keys, list) or not isinstance(outputs, list):
        raise ValueError("output manifest 必须包含 keys/outputs 数组")
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("output manifest keys 必须唯一且按字典序排列")
    expected_keys = sorted(set(ref_map) | set(test_map))
    if list(keys) != expected_keys:
        raise ValueError("output manifest keys 未精确覆盖两臂实际出现过的全部输出")
    if len(outputs) != len(keys):
        raise ValueError("output manifest outputs 数量与 keys 不一致")
    by_key = {str(item.get("name")): item for item in outputs if isinstance(item, dict)}
    if set(by_key) != set(keys) or len(by_key) != len(outputs):
        raise ValueError("output manifest outputs 的 name 集合与 keys 不一致或存在重复")
    for key in keys:
        item = by_key[key]
        present_arms = [(arm, mapping) for arm, mapping in
                        (("reference", ref_map), ("instrumented", test_map)) if key in mapping]
        if not present_arms:
            raise ValueError(f"output manifest {key} 在两臂中都不存在")
        for arm, mapping in present_arms:
            arm_data = item.get(arm)
            if not isinstance(arm_data, dict):
                raise ValueError(f"output manifest {key}.{arm} 缺失")
            for field in ("shape", "dtype", "elements", "sha256"):
                if field not in arm_data:
                    raise ValueError(f"output manifest {key}.{arm}.{field} 缺失")
            if list(arm_data["shape"]) != [int(x) for x in mapping[key].shape]:
                raise ValueError(f"output manifest {key}.{arm}.shape 与实际张量不一致")
            dtype = str(mapping[key].dtype).replace("torch.", "")
            if str(arm_data["dtype"]).replace("torch.", "") != dtype:
                raise ValueError(f"output manifest {key}.{arm}.dtype 与实际张量不一致")
            if int(arm_data["elements"]) != int(mapping[key].numel()):
                raise ValueError(f"output manifest {key}.{arm}.elements 与实际张量不一致")
            declared_sha = _require_sha(arm_data["sha256"], f"output manifest {key}.{arm}.sha256")
            actual_sha = _tensor_record(torch, mapping[key], key)["sha256"]
            if declared_sha != actual_sha:
                raise ValueError(f"output manifest {key}.{arm}.sha256 与实际张量元素不一致")
        for field in ("shape", "dtype", "elements"):
            if field not in item:
                raise ValueError(f"output manifest {key}.{field} 缺失")
        _require_sha(item.get("sha256"), f"output manifest {key}.sha256")
    field_names = ("name", "shape", "dtype", "elements", "sha256")
    expected_ref = _canonical_digest([
        {k: _tensor_record(torch, ref_map[key], key).get(k) for k in field_names}
        for key in sorted(ref_map)])
    expected_test = _canonical_digest([
        {k: _tensor_record(torch, test_map[key], key).get(k) for k in field_names}
        for key in sorted(test_map)])
    if doc.get("reference_outputs_sha256") != expected_ref:
        raise ValueError("output manifest reference_outputs_sha256 与实际输出不一致")
    if doc.get("instrumented_outputs_sha256") != expected_test:
        raise ValueError("output manifest instrumented_outputs_sha256 与实际输出不一致")


def _output_manifest_from_tensors(torch, ref_map: Mapping[str, Any], test_map: Mapping[str, Any],
                                  *, run_id: Optional[str], contract_sha256: Optional[str],
                                  input_sha256: Optional[str], rank: Optional[int] = None,
                                  world_size: Optional[int] = None) -> Dict[str, Any]:
    union = sorted(set(ref_map) | set(test_map))
    outputs: List[Dict[str, Any]] = []
    for key in union:
        item: Dict[str, Any] = {"name": key}
        if key in ref_map:
            item["reference"] = _tensor_record(torch, ref_map[key], key)
        if key in test_map:
            item["instrumented"] = _tensor_record(torch, test_map[key], key)
        primary = item.get("reference") or item.get("instrumented")
        item["shape"] = primary["shape"]
        item["dtype"] = primary["dtype"]
        item["elements"] = primary["elements"]
        item["sha256"] = primary["sha256"]
        outputs.append(item)
    ref_records = [item["reference"] for item in outputs if "reference" in item]
    test_records = [item["instrumented"] for item in outputs if "instrumented" in item]
    field_names = ("name", "shape", "dtype", "elements", "sha256")
    doc: Dict[str, Any] = {
        "schema": OUTPUT_MANIFEST_SCHEMA,
        "keys": union,
        "outputs": outputs,
        "reference_outputs_sha256": _canonical_digest([{k: r[k] for k in field_names} for r in ref_records]),
        "instrumented_outputs_sha256": _canonical_digest([{k: r[k] for k in field_names} for r in test_records]),
        "key_sets_equal": set(ref_map) == set(test_map),
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


def _normalize_launch_result(result: Any) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], Mapping):
        return dict(result[0]), dict(result[1]) if isinstance(result[1], Mapping) else None
    if isinstance(result, Mapping) and isinstance(result.get("outputs"), Mapping):
        inputs = result.get("inputs")
        return dict(result["outputs"]), dict(inputs) if isinstance(inputs, Mapping) else None
    if isinstance(result, Mapping):
        return dict(result), None
    raise TypeError(f"launch_at_level 必须返回输出 dict，实际 {type(result).__name__}")


def _comparison_doc(comparison: Dict[str, Any], *, run_id: str, contract_sha256: str,
                    input_sha256: str, ref_output_sha: str, test_output_sha: str,
                    required: Sequence[str], output_manifest_path: Path,
                    output_manifest_sha: str) -> Dict[str, Any]:
    max_abs = max((float(t.get("max_abs_err", 0.0) or 0.0) for t in comparison["per_tensor"]), default=0.0)
    max_rel = max((float(t.get("max_rel_err", 0.0) or 0.0) for t in comparison["per_tensor"]), default=0.0)
    nonfinite = sum(int(t.get("nonfinite_elements", 0) or 0) for t in comparison["per_tensor"])
    return {
        "schema": COMPARE_SCHEMA,
        "pass": bool(comparison["all_passed"]),
        "run_id": str(run_id),
        "contract_sha256": _require_sha(contract_sha256, "contract_sha256"),
        "input_sha256": _require_sha(input_sha256, "input_sha256"),
        "reference_output_sha256": _require_sha(ref_output_sha, "reference_output_sha256"),
        "instrumented_output_sha256": _require_sha(test_output_sha, "instrumented_output_sha256"),
        "keys": comparison["output_keys_reference"],
        "key_sets_equal": bool(comparison["output_key_sets_equal"]),
        "required_output_tensors": list(required),
        "missing_required_tensors": comparison["missing_required_tensors"],
        "checked_elements": sum(int(t.get("numel", 0) or 0) for t in comparison["per_tensor"]),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "nonfinite": nonfinite,
        "arrays": {t["name"]: t for t in comparison["per_tensor"]},
        "thresholds": comparison.get("thresholds"),
        "check_scope": "all_output_elements_all_ranks",
        "sample_only": False,
        "rank": comparison.get("rank"),
        "world_size": comparison.get("world_size"),
        "ranks_checked": ([comparison.get("rank")] if comparison.get("rank") is not None else []),
        "binding": {
            "run_id": str(run_id),
            "contract": {"path": None, "sha256": contract_sha256},
            "contract_sha256": contract_sha256,
            "input_manifest": {"path": None, "sha256": input_sha256},
            "input_sha256": input_sha256,
            "reference_output_sha256": ref_output_sha,
            "instrumented_output_sha256": test_output_sha,
            "output_manifest": {"path": str(output_manifest_path), "outputs_sha256": output_manifest_sha},
        },
    }


def run_correctness(contract: Optional[Dict[str, Any]], out_path: Path,
                    info: Dict[str, Any]) -> int:
    cfg = _contract_correctness(contract)
    atol = float(cfg["atol"])
    rtol = float(cfg["rtol"])
    seed = int(cfg["seed"])
    rank = int(info["rank"])
    world_size = int(info["world_size"])
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(f"非法 rank/world_size：rank={rank} world_size={world_size}")
    names = PROJECT_CORRECTNESS["level_names"]
    run_id = info.get("run_id") or first(contract, ("run.run_id", "artifacts.run_id", "run_id"))
    contract_sha = info.get("contract_sha256")
    if not run_id:
        raise ValueError("correctness 证据必须绑定 run ID，但 contract/info 没有 run.run_id")
    if not contract_sha:
        raise ValueError("correctness 证据必须绑定 contract SHA256，但未提供 contract 路径/摘要")

    external_input_paths = _input_manifest_candidates(contract, info)
    external_input_path = external_input_paths[0] if external_input_paths else None
    paths = _clear_stale_evidence(out_path, preserve_input=external_input_path)

    mods = setup_gpu_imports()
    torch = mods["torch"]
    external_manifest = None
    if external_input_path is not None:
        if not external_input_path.is_file():
            raise ValueError(f"input manifest 不存在：{external_input_path}")
        external_manifest = validate_input_manifest(
            load_json(external_input_path), expected_seed=seed,
            expected_rank=rank, expected_world_size=world_size)
        if (external_manifest.get("run_id") not in (None, str(run_id))
                or external_manifest.get("contract_sha256") not in (None, contract_sha)):
            raise ValueError("input manifest 的 run_id/contract_sha256 与当前 run 不一致")

    inputs, input_source = _prepare_inputs(torch, mods, contract, info, seed)
    # 显式 input manifest 优先于 contract 规范的自动物化；否则随机算法/device 差异会
    # 把一份合法的外部证据误判成冲突。若项目另给了真实 tensor，仍会逐元素复核 SHA。
    if external_manifest is not None and input_source == "contract.input.tensors":
        inputs = None
        input_source = "external_input_manifest"
    required = _expected_output_names(contract)
    expected_inputs = _expected_input_names(contract)
    input_manifest = None
    actual_inputs = dict(inputs) if inputs is not None else None
    if inputs is not None:
        missing_inputs = sorted(set(expected_inputs) - set(inputs))
        if missing_inputs:
            raise ValueError(f"randn 输入缺 contract 声明的输入键：{missing_inputs}")
        input_manifest = _input_manifest_from_tensors(
            torch, inputs, seed=seed, rank=rank, world_size=world_size,
            run_id=run_id, contract_sha256=contract_sha)
        if (external_manifest is not None
                and external_manifest["input_sha256"] != input_manifest["input_sha256"]):
            raise ValueError("实际 randn 张量 SHA256 与 input manifest 不一致")
        info = dict(info)
        # 同一 dict 交给两臂；旧式 launch 即使只返回 outputs，也仍使用这份输入。
        info["inputs"] = dict(inputs)
        info["input_seed"] = seed
    elif external_manifest is not None:
        manifest_input_names = {entry["name"] for entry in external_manifest["inputs"]}
        missing_inputs = sorted(set(expected_inputs) - manifest_input_names)
        if missing_inputs:
            raise ValueError(f"input manifest 缺 contract 声明的输入键：{missing_inputs}")
        input_manifest = dict(external_manifest)
        input_manifest["run_id"] = str(run_id)
        input_manifest["contract_sha256"] = _require_sha(contract_sha, "contract_sha256")
        info = dict(info)
        info["input_manifest"] = input_manifest

    torch.manual_seed(seed)
    ref_result = launch_at_level(mods, names["off"], info)        # 参考：插桩全关
    ref_map, ref_returned_inputs = _normalize_launch_result(ref_result)

    # 兼容“launch 自己生成并回传 inputs”的项目：reference 第一跑可提供输入证据，
    # 随后把同一 dict 显式交给 full，并要求 full 回传的输入 SHA256 保持一致。
    if ref_returned_inputs is not None:
        ref_input_manifest = _input_manifest_from_tensors(
            torch, ref_returned_inputs, seed=seed, rank=rank, world_size=world_size,
            run_id=run_id, contract_sha256=contract_sha)
        if (input_manifest is not None
                and input_manifest["input_sha256"] != ref_input_manifest["input_sha256"]):
            raise ValueError("reference 实际输入 SHA256 与输入证据不一致")
        input_manifest = ref_input_manifest
        input_source = "launch.reference.inputs"
        actual_inputs = dict(ref_returned_inputs)
        info = dict(info)
        info["inputs"] = dict(ref_returned_inputs)
        info["input_seed"] = seed
    elif input_manifest is None:
        raise RuntimeError(
            "没有可核验的 randn 输入证据：请实现 mods.prepare_inputs、让 launch_at_level 回传 "
            "inputs，或提供 --input-manifest；不能只记录 seed 就声称输入一致")

    torch.manual_seed(seed)
    test_result = launch_at_level(mods, names["full"], info)      # 被测：插桩全开
    test_map, test_returned_inputs = _normalize_launch_result(test_result)

    # 回传 inputs 时复核两臂确实使用同一份 randn；只回传一边属于证据不完整。
    if (ref_returned_inputs is None) != (test_returned_inputs is None):
        raise ValueError("两臂中只有一边回传 inputs；无法证明使用同一份 randn 输入")
    if test_returned_inputs is not None:
        test_input_manifest = _input_manifest_from_tensors(
            torch, test_returned_inputs, seed=seed, rank=rank, world_size=world_size,
            run_id=run_id, contract_sha256=contract_sha)
        if test_input_manifest["input_sha256"] != input_manifest["input_sha256"]:
            raise ValueError("reference 与 instrumented 的输入 SHA256 不一致；禁止复用正确性结论")
    if actual_inputs is not None:
        final_input_manifest = _input_manifest_from_tensors(
            torch, actual_inputs, seed=seed, rank=rank, world_size=world_size,
            run_id=run_id, contract_sha256=contract_sha)
        if final_input_manifest["input_sha256"] != input_manifest["input_sha256"]:
            raise ValueError("kernel 运行后输入张量 SHA256 发生变化；正确性结果不能绑定到该输入")
    input_sha = _require_sha(input_manifest["input_sha256"], "input_sha256")

    instr_vs_ref = _compare_all_elements(
        torch, ref_map, test_map, atol, rtol, required,
        rank=rank, world_size=world_size)
    instr_vs_ref["thresholds"] = {"atol": atol, "rtol": rtol}

    all_passed = bool(instr_vs_ref["all_passed"])

    # G02 只有一条判据："插桩前后逐元素一致"。但这条判据同时被两件完全不同的事
    # 破坏：插桩真的改变了结果，或者 kernel 自己 run-to-run 就不可重现。只报
    # all_passed=false 分不出是哪一件，会把人送去查插桩，而问题在 baseline。
    # 所以一旦失败就重跑 off 档、拿 off-vs-off 做归因。通过时不跑，零代价。
    baseline_self = None
    structural_outputs_ok = bool(
        instr_vs_ref["output_key_sets_equal"]
        and not instr_vs_ref["missing_required_tensors"]
        and all(t.get("shape_match") and t.get("dtype_match") and t.get("numel_match")
                for t in instr_vs_ref["per_tensor"]))
    if not all_passed and structural_outputs_ok:
        trials = []
        for _ in range(2):
            torch.manual_seed(seed)
            ref_again_result = launch_at_level(mods, names["off"], info)
            ref_again, _ = _normalize_launch_result(ref_again_result)
            trials.append(_compare_all_elements(torch, ref_map, ref_again, atol, rtol, required,
                                                rank=rank, world_size=world_size))
        worst = max((float(t.get("max_abs_err", 0.0) or 0.0)
                     for trial in trials for t in trial["per_tensor"]), default=0.0)
        baseline_self = {
            "trials": trials,
            "bit_exact": worst == 0.0,
            "max_abs_err": worst,
            "all_passed": all(bool(trial["all_passed"]) for trial in trials),
        }

    output_manifest = _output_manifest_from_tensors(
        torch, ref_map, test_map, run_id=run_id, contract_sha256=contract_sha,
        input_sha256=input_sha, rank=rank, world_size=world_size)
    _validate_output_manifest(torch, output_manifest, ref_map, test_map, required)
    compare_doc = _comparison_doc(
        instr_vs_ref, run_id=str(run_id), contract_sha256=contract_sha, input_sha256=input_sha,
        ref_output_sha=output_manifest["reference_outputs_sha256"],
        test_output_sha=output_manifest["instrumented_outputs_sha256"],
        required=required, output_manifest_path=paths["output_manifest"],
        output_manifest_sha=output_manifest["instrumented_outputs_sha256"])
    compare_doc["binding"]["contract"]["path"] = info.get("contract_path")
    compare_doc["binding"]["input_manifest"]["path"] = str(paths["input_manifest"])
    compare_doc["binding"]["output_manifest"]["path"] = str(paths["output_manifest"])
    compare_doc["ranks_checked"] = [rank]
    compare_doc["world_size"] = world_size
    compare_doc["rank_scope"] = "one instrumentation-identical process per rank; aggregate all rank files before G02 delivery"
    # compare.json 顶层唯一的 PASS 字段是 pass；这里强制与全量比较结论一致。
    compare_doc["pass"] = bool(all_passed and instr_vs_ref["output_key_sets_equal"])
    if compare_doc["pass"] is not True:
        all_passed = False

    report = dict(_plan(contract, info))
    report.update({
        "all_passed": all_passed,
        # 硬写 False：G02 明确要求 sample_only 不为 true。
        "sample_only": False,
        "pass": bool(all_passed),
        "instrumented_vs_reference": instr_vs_ref,
        "max_abs_err": max((t.get("max_abs_err", 0.0) for t in instr_vs_ref["per_tensor"]), default=0.0),
        "max_rel_err": max((t.get("max_rel_err", 0.0) for t in instr_vs_ref["per_tensor"]), default=0.0),
        "input_manifest": str(paths["input_manifest"]),
        "output_manifest": str(paths["output_manifest"]),
        "compare_json": str(paths["compare"]),
        "input_sha256": input_sha,
        "reference_output_sha256": output_manifest["reference_outputs_sha256"],
        "instrumented_output_sha256": output_manifest["instrumented_outputs_sha256"],
        "run_id": str(run_id),
        "contract_sha256": contract_sha,
        "input_evidence_source": input_source,
        "binding": compare_doc["binding"],
    })
    if baseline_self is not None:
        report["baseline_self_consistency"] = baseline_self
        report["failure_attribution"] = (
            "instrumentation" if baseline_self["bit_exact"] else "kernel_nondeterministic")
    elif not all_passed:
        report["failure_attribution"] = "output_contract_or_numeric_shape"
    harness = mods.get("harness")
    checker = getattr(harness, "check_all", None) if harness is not None else None
    if PROJECT_CORRECTNESS["compare_against_golden"]:
        if checker is None:
            raise NotImplementedError(
                "compare_against_golden=True 但 mods['harness'].check_all 不存在："
                "请在 setup_gpu_imports() 里提供 golden 比对入口，或把该开关关掉。"
                "参考真实实现 {}:140".format(_REAL_IMPL))
        golden_report = checker(test_map)
        report["instrumented_vs_golden"] = golden_report
        report["instrumented_vs_golden_all_passed"] = bool(golden_report.get("all_passed"))

    # 先原子写 manifests，再写 compare，最后写 correctness；所有摘要都来自本轮内存对象。
    _atomic_write_json(paths["input_manifest"], input_manifest)
    _atomic_write_json(paths["output_manifest"], output_manifest)
    _atomic_write_json(paths["compare"], compare_doc)
    _atomic_write_json(paths["correctness"], report)
    print("[validate_correctness] wrote {} all_passed={}".format(out_path, all_passed))
    return 0 if all_passed else 4


def _write_failure_evidence(out_path: Path, error: BaseException, info: Dict[str, Any]) -> None:
    paths = _evidence_paths(out_path)
    _unlink(paths["output_manifest"])
    failure = {
        "schema": COMPARE_SCHEMA,
        "pass": False,
        "run_id": info.get("run_id"),
        "contract_sha256": info.get("contract_sha256"),
        "input_sha256": None,
        "reference_output_sha256": None,
        "instrumented_output_sha256": None,
        "error": f"{type(error).__name__}: {error}",
        "binding": {
            "run_id": info.get("run_id"),
            "contract_sha256": info.get("contract_sha256"),
            "input_sha256": None,
        },
        "check_scope": "all_output_elements_all_ranks",
        "sample_only": False,
    }
    correctness = {
        "schema": SCHEMA,
        "all_passed": False,
        "pass": False,
        "sample_only": False,
        "check_scope": "all_output_elements_all_ranks",
        "run_id": info.get("run_id"),
        "contract_sha256": info.get("contract_sha256"),
        "compare_json": str(paths["compare"]),
        "error": f"{type(error).__name__}: {error}",
    }
    _atomic_write_json(paths["compare"], failure)
    _atomic_write_json(paths["correctness"], correctness)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Correctness gate: instrumented output must equal the uninstrumented output.")
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--abi", type=Path, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--input-manifest", type=Path, default=None,
                        help="外部 randn 输入 manifest；没有 prepare_inputs 钩子时必须提供")
    parser.add_argument("--run-id", default=None, help="覆盖 contract 的 run ID，仅用于证据绑定")
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU-only: emit the correctness plan without launching the GPU")
    args = parser.parse_args(argv)

    # 任何退出路径都不能留下上一轮 compare PASS。输入 manifest 可能由调用方提供，
    # 这里先不删；run_correctness 会在确认外部路径后清理生成副本。
    try:
        early_paths = _evidence_paths(args.out)
    except ValueError as error:
        if args.out.name in ("compare.json", "output_manifest.json"):
            _unlink(args.out)
        print("[validate_correctness] {}".format(error), file=sys.stderr)
        return 2
    _unlink(early_paths["compare"])
    _unlink(early_paths["output_manifest"])
    if not args.dry_run:
        _unlink(early_paths["correctness"])

    contract = None
    contract_sha = None
    if args.contract is not None:
        try:
            contract_sha = _sha256_file(args.contract)
            contract = load_json(args.contract)
        except (OSError, ValueError) as error:
            print("[validate_correctness] cannot read contract {}: {}".format(
                args.contract, error), file=sys.stderr)
            return 2
    info = contract_paths(contract)
    info["contract_path"] = str(args.contract) if args.contract else None
    info["contract_sha256"] = contract_sha
    if args.run_id is not None:
        info["run_id"] = args.run_id
    if args.input_manifest is not None:
        info["input_manifest_path"] = str(args.input_manifest)
    if args.abi is not None:
        info["abi_path"] = args.abi
    if args.rank is not None:
        info["rank"] = args.rank

    if args.dry_run:
        # dry-run 也只写计划；先删旧 compare，防止上一轮 PASS 被误当成本轮证据。
        paths = _evidence_paths(args.out)
        _unlink(paths["compare"])
        _unlink(paths["output_manifest"])
        plan = _plan(contract, info)
        _atomic_write_json(args.out, plan)
        print("[validate_correctness] dry-run plan -> {}".format(args.out))
        return 0

    try:
        return run_correctness(contract, args.out, info)
    except Exception as error:  # noqa: BLE001 -- 如实暴露真实失败原因
        try:
            _write_failure_evidence(args.out, error, info)
        except Exception as write_error:  # noqa: BLE001
            print("[validate_correctness] failure evidence write failed: {}".format(write_error),
                  file=sys.stderr)
        print("[validate_correctness] FAILED: {}: {}".format(
            type(error).__name__, error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
