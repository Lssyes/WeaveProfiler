"""从 ABI tracks 和一次真实 kernel launch 参数生成 observer warp 拓扑。

该模块故意不读取 ABI 的 reference_config，也不提供默认布局。物理 warp 数
只能来自 runtime.parameters.kNumThreads/32 或 runtime.parameters.num_threads/32。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


SCHEMA = "fast-instrument-warp-topology/v1"
_IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")


class TopologyError(ValueError):
    """输入 ABI/runtime 不满足拓扑契约。"""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TopologyError(f"无法读取 JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TopologyError(f"JSON 解析失败 {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise TopologyError(f"无法读取文件 {path}: {exc}") from exc


def _nonempty(value: Any, name: str) -> Any:
    if value is None or value == "" or value == [] or value == {}:
        raise TopologyError(f"runtime 缺少真实 {name}")
    if isinstance(value, str) and not value.strip():
        raise TopologyError(f"runtime 缺少真实 {name}")
    return value


def _flatten_parameters(parameters: Mapping[str, Any]) -> Dict[str, Any]:
    """允许 launch 参数放在 parameters 的一层或有语义的嵌套 object 中。"""
    result: Dict[str, Any] = {}

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if isinstance(key, str) and _IDENTIFIER.fullmatch(key):
                result.setdefault(key, value)
            if isinstance(value, dict):
                visit(value)

    visit(parameters)
    return result


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise TopologyError(f"{context} 必须是整数")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+]?[0-9]+", value.strip()):
        return int(value, 10)
    raise TopologyError(f"{context} 必须是整数，实际为 {value!r}")


def _eval_expr(expression: Any, values: Mapping[str, Any], context: str) -> int:
    if not isinstance(expression, str) or not expression.strip():
        raise TopologyError(f"{context}.observer_warp_expr 必须是非空表达式")
    # ABI 的表达式是由参数名、整数和基本算术组成的 C++ 常量表达式。
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise TopologyError(f"{context}: 非法 observer_warp_expr {expression!r}") from exc

    def evaluate(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise TopologyError(f"{context}: runtime.parameters 缺少 {node.id!r}")
            return _integer(values[node.id], f"runtime.parameters.{node.id}")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub,
                                                                  ast.Mult, ast.FloorDiv)):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise TopologyError(f"{context}: observer_warp_expr 除零")
            return left // right
        raise TopologyError(f"{context}: 不支持 observer_warp_expr {expression!r}")

    return evaluate(tree.body)


def _runtime_values(runtime: Mapping[str, Any]) -> Tuple[Dict[str, Any], str, str]:
    _nonempty(runtime.get("kernel_signature"), "kernel_signature")
    _nonempty(runtime.get("provenance"), "provenance")
    parameters = runtime.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise TopologyError("runtime.parameters 必须是非空 object")
    values = _flatten_parameters(parameters)
    candidates = [name for name in ("kNumThreads", "num_threads") if name in values]
    if len(candidates) != 1:
        raise TopologyError(
            "runtime.parameters 必须且只能提供 kNumThreads 或 num_threads（用于物理 warp 数）")
    return values, candidates[0], str(runtime["kernel_signature"])


def _track_id(track: Mapping[str, Any], index: int) -> int:
    value = track.get("track_id", track.get("id"))
    if value is None:
        raise TopologyError(f"ABI tracks[{index}] 缺少 track_id")
    return _integer(value, f"ABI tracks[{index}].track_id")


def _representation_policy(track: Mapping[str, Any], abi: Mapping[str, Any]) -> str:
    """Return the only policies allowed to represent multiple physical warps."""
    role = str(track.get("role", track.get("track_role", ""))).strip().lower()
    architecture = str(abi.get("architecture", abi.get("gpu_arch", ""))).lower()
    explicit = track.get("representation_policy", track.get("observer_policy"))
    if isinstance(explicit, str) and explicit:
        explicit = explicit.strip().lower()
        if (explicit == "hopper_mma_warpgroup"
                and "hopper" in architecture
                and (("mma" in role and "warpgroup" in role)
                     or (track.get("warpgroup") is True
                         and ("math" in role or str(track.get("semantic_kind", "")).lower() == "mma")))):
            return explicit
        if (explicit == "blackwell_epilogue_warpgroup"
                and "blackwell" in architecture
                and (("epilogue" in role and "warpgroup" in role)
                     or (track.get("warpgroup") is True and "epilogue" in role))):
            return explicit
        return "physical_warp_observer"
    if "mma" in role and "warpgroup" in role and "hopper" in architecture:
        return "hopper_mma_warpgroup"
    if "epilogue" in role and "warpgroup" in role and "blackwell" in architecture:
        return "blackwell_epilogue_warpgroup"
    return "physical_warp_observer"


def _allows_representation(policy: str) -> bool:
    return policy in {"hopper_mma_warpgroup", "blackwell_epilogue_warpgroup"}


def build_topology(abi: Mapping[str, Any], runtime: Mapping[str, Any],
                   abi_path: Path, runtime_path: Path) -> Dict[str, Any]:
    values, thread_key, kernel_signature = _runtime_values(runtime)
    num_threads = _integer(values[thread_key], f"runtime.parameters.{thread_key}")
    if num_threads <= 0 or num_threads % 32:
        raise TopologyError(f"runtime.parameters.{thread_key} 必须是正的 32 倍数")
    physical = num_threads // 32

    tracks = abi.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise TopologyError("ABI.tracks 必须是非空 list")
    observed: List[Tuple[int, int, Mapping[str, Any]]] = []
    seen_track_ids = set()
    seen_observers = set()
    for index, raw_track in enumerate(tracks):
        if not isinstance(raw_track, dict):
            raise TopologyError(f"ABI tracks[{index}] 必须是 object")
        track_id = _track_id(raw_track, index)
        if track_id in seen_track_ids:
            raise TopologyError(f"ABI tracks 重复 track_id={track_id}")
        seen_track_ids.add(track_id)
        observer = _eval_expr(raw_track.get("observer_warp_expr"),
                              {**values, "track_id": track_id},
                              f"ABI tracks[{index}]")
        if observer < 0:
            raise TopologyError(f"ABI tracks[{index}] observer warp 不能为负数")
        if observer >= physical:
            if raw_track.get("may_be_empty") is True:
                continue
            raise TopologyError(
                f"ABI tracks[{index}] observer warp {observer} 超出 physical warp 范围 0..{physical - 1}")
        if observer in seen_observers:
            raise TopologyError(f"多个 ABI track 映射到 observer warp {observer}")
        seen_observers.add(observer)
        observed.append((observer, track_id, raw_track,
                         _representation_policy(raw_track, abi)))

    if not observed:
        raise TopologyError("没有落在本次 runtime physical warp 范围内的 observer track")
    observed.sort(key=lambda item: item[0])
    lanes: List[Dict[str, Any]] = []
    for index, (observer, track_id, track, policy) in enumerate(observed):
        start = 0 if index == 0 else observer
        end = (observed[index + 1][0] - 1 if index + 1 < len(observed)
               else physical - 1)
        if start > end:
            raise TopologyError(f"observer 边界产生空范围: warp{observer} [{start}, {end}]")
        if end > start and not _allows_representation(policy):
            raise TopologyError(
                f"observer warp {observer} 代表物理 warp {start}-{end} 被禁止；"
                "只有 Hopper MMA warpgroup 或 Blackwell epilogue warpgroup 可代表多个 warp")
        role = track.get("role", track.get("track_role"))
        if not isinstance(role, str) or not role:
            role = f"track{track_id}"
        label = f"warp{observer}" if start == end \
            else f"warp{observer}（代表 warp{start}-{end}）"
        lanes.append({"source_track_id": track_id, "observer_warp": observer,
                      "represents": [start, end], "role": role,
                      "representation_policy": policy,
                      "warpgroup": track.get("warpgroup") is True,
                      "label": label})

    if lanes[0]["represents"][0] != 0 or lanes[-1]["represents"][1] != physical - 1:
        raise TopologyError("observer lane 没有完整覆盖 physical warp 0..physical-1")
    for previous, current in zip(lanes, lanes[1:]):
        if previous["represents"][1] + 1 != current["represents"][0]:
            raise TopologyError("observer lane represents 范围存在空洞或重叠")

    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "abi_path": str(abi_path.resolve()),
        "abi_sha256": _sha256(abi_path),
        "runtime_path": str(runtime_path.resolve()),
        "runtime_sha256": _sha256(runtime_path),
        "kernel_signature": kernel_signature,
        "runtime_provenance": runtime["provenance"],
        "architecture": abi.get("architecture", abi.get("gpu_arch")),
        "physical_warps_per_cta": physical,
        "observer_lane_count": len(lanes),
        "lanes": lanes,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    payload["canonical_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="根据 ABI/runtime 生成 observer warp topology")
    parser.add_argument("--abi", required=True, help="ABI JSON")
    parser.add_argument("--runtime-launch", dest="runtime", required=True,
                        help="真实 kernel launch runtime JSON")
    parser.add_argument("--out", required=True, help="拓扑 JSON 输出路径")
    args = parser.parse_args(list(argv) if argv is not None else None)
    abi_path, runtime_path, out_path = (Path(args.abi).resolve(), Path(args.runtime).resolve(),
                                        Path(args.out).resolve())
    try:
        topology = build_topology(_read_json(abi_path), _read_json(runtime_path),
                                  abi_path, runtime_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(topology, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    except (TopologyError, OSError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(topology, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
