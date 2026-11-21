#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate weaveprofiler knowledge bases and canonical operator semantics.

The arch entry files keep their original six-field schema.  The operator KB is
validated separately so that canonical semantic IDs, evidence status, machine
checks, legacy five-class semantics, and kb references cannot drift apart.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

STD = {"装载", "等待", "计算", "同步", "写回"}
REC = {"记录点", "计时", "物理位置"}
FIELDS = ["semantic", "language", "construct", "isa_mnemonics", "placement_rule", "evidence"]
EV = ["snippet", "machine", "date"]

OPERATOR_IDS = {"gemm", "attention", "reduction", "moe", "convolution", "megakernel"}
EVIDENCE_STATUSES = {"measured", "compiled", "simulated", "unverified"}
REQUIRED_MACHINE_CHECKS = (
    "correctness",
    "overhead",
    "assembly_change_rate",
    "alignment",
    "buffer_completeness",
)
LEGACY_TO_CANONICAL = {
    "装载": {"load", "epilogue"},
    "等待": {"wait"},
    "计算": {"mma", "compute", "epilogue", "reduction", "softmax", "dispatch"},
    "同步": {"barrier", "dispatch", "communication", "wait"},
    "写回": {"store", "epilogue", "communication"},
}
REF_SEP = "::"


def _read_json(path: Path, errs: List[str]) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errs.append(f"缺文件 {path}")
    except json.JSONDecodeError as exc:
        errs.append(f"{path}: JSON 解析失败：{exc}")
    except OSError as exc:
        errs.append(f"{path}: 读取失败：{exc}")
    return None


def check(path: Path, allowed_sem: Set[str], errs: List[str],
          canonical_ids: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Validate one card or recorder entry file.

    The migrated knowledge base keeps the original six evidence fields and adds
    canonical semantic fields.  Both shapes are accepted; entries that declare
    ``semantic_id`` are checked against the canonical naming policy here and
    against ``semantic_ids.json`` by the caller once that file is loaded.
    """
    d = _read_json(path, errs)
    if d is None:
        return {}
    for k in ("arch", "machine", "date", "entries"):
        if k not in d:
            errs.append(f"{path.name}: 缺 arch 级字段 {k}")
    if "warp_size" not in d:
        errs.append(f"{path.name}: 缺 warp_size")
    seen = set()
    for i, e in enumerate(d.get("entries", [])):
        where = f"{path.name}#{i}"
        if not isinstance(e, dict):
            errs.append(f"{where}: 条目必须是对象")
            continue
        for key in ("construct", "isa_mnemonics", "placement_rule", "evidence", "language"):
            if key not in e:
                errs.append(f"{where}: 缺 {key}")
        semantic_id = e.get("semantic_id")
        legacy = e.get("semantic_legacy", e.get("semantic"))
        if semantic_id is not None:
            if not isinstance(semantic_id, str) or not semantic_id.strip():
                errs.append(f"{where}: semantic_id 为空")
            elif semantic_id == "unknown":
                errs.append(f"{where}: 硬件条目不允许 semantic_id=unknown")
            elif canonical_ids is not None and semantic_id not in canonical_ids:
                errs.append(f"{where}: semantic_id={semantic_id!r} 未在 semantic_ids.json 定义")
            if e.get("display_name") != semantic_id:
                errs.append(f"{where}: display_name 必须等于 semantic_id={semantic_id!r}")
            if legacy not in allowed_sem:
                errs.append(f"{where}: semantic_legacy={legacy!r} 不在允许集合")
            if e.get("evidence_status") not in EVIDENCE_STATUSES:
                errs.append(f"{where}: evidence_status 必须属于 {sorted(EVIDENCE_STATUSES)}")
            if not str(e.get("evidence_status_basis", "")).strip():
                errs.append(f"{where}: evidence_status_basis 为空")
        else:
            if legacy not in allowed_sem:
                errs.append(f"{where}: semantic={legacy!r} 不在允许集合")
        if not str(e.get("language", "")).strip():
            errs.append(f"{where}: language 为空")
        if not str(e.get("construct", "")).strip():
            errs.append(f"{where}: construct 为空")
        mn = e.get("isa_mnemonics")
        if not isinstance(mn, list) or not mn:
            errs.append(f"{where}: isa_mnemonics 非空列表")
        elif any(not str(x).strip() for x in mn):
            errs.append(f"{where}: isa_mnemonics 有空项")
        if not str(e.get("placement_rule", "")).strip():
            errs.append(f"{where}: placement_rule 为空")
        ev = e.get("evidence")
        if not isinstance(ev, dict):
            errs.append(f"{where}: evidence 必须是对象")
        else:
            for k in EV:
                if k not in ev:
                    errs.append(f"{where}: evidence 缺 {k}")
                elif not str(ev[k]).strip():
                    errs.append(f"{where}: evidence.{k} 为空")
        key = (e.get("language"), e.get("construct"))
        if key in seen:
            errs.append(f"{where}: 重复条目 {key}")
        seen.add(key)
    return d

def _load_canonical_ids(kb: Path, errs: List[str]) -> Set[str]:
    path = kb / "semantic_ids.json"
    d = _read_json(path, errs)
    if not isinstance(d, dict):
        return set()
    ids: Set[str] = set()
    semantics = d.get("semantics")
    if not isinstance(semantics, list) or not semantics:
        errs.append("semantic_ids.json: semantics 必须是非空列表")
        return ids
    for i, item in enumerate(semantics):
        if not isinstance(item, dict):
            errs.append(f"semantic_ids.json#{i}: 必须是对象")
            continue
        sid = item.get("semantic_id")
        if not isinstance(sid, str) or not sid.strip():
            errs.append(f"semantic_ids.json#{i}: semantic_id 为空")
            continue
        if sid in ids:
            errs.append(f"semantic_ids.json: 重复 semantic_id={sid}")
        ids.add(sid)
    required = set(d.get("required_ids", []))
    missing = required - ids
    if missing:
        errs.append(f"semantic_ids.json: required_ids 未在 semantics 中定义：{sorted(missing)}")
    return ids


def _entry_files(kb: Path) -> List[Path]:
    """Return card profiles and language recorder profiles that carry ISA entries."""
    paths: List[Path] = []
    paths.extend(sorted((kb / "cards").glob("*.json")))
    paths.extend(sorted((kb / "languages").rglob("*.json")))
    return paths


def _entry_refs(kb: Path, errs: List[str]) -> Dict[str, Set[str]]:
    """Map a unique JSON basename to its exact `entries[].construct` values."""
    result: Dict[str, Set[str]] = {}
    owners: Dict[str, Path] = {}
    for path in _entry_files(kb):
        d = _read_json(path, errs)
        if not isinstance(d, dict) or not isinstance(d.get("entries"), list):
            continue
        if path.name in owners:
            # Refs use basenames, as in the curated knowledge base.  A duplicate
            # would make a citation point at two different entries.
            errs.append("entry 文件名重复，kb_ref 无法唯一解析：%s 与 %s" % (owners[path.name], path))
            continue
        owners[path.name] = path
        result[path.name] = {
            item["construct"]
            for item in d["entries"]
            if isinstance(item, dict) and isinstance(item.get("construct"), str)
        }
    # The curated operator KB predates the machine-profile split.  Keep those
    # citations resolvable through explicit aliases while requiring every
    # alias target to be a real, current profile on disk.
    aliases = {
        "sm_90.json": ("h20_sm_90.json",),
        "sm_100.json": ("b300_sm_100a.json", "b300_sm_103a.json"),
        "sm_103.json": ("b300_sm_103a.json",),
        "gfx942.json": ("mi308x_gfx942.json",),
    }
    for alias, targets in aliases.items():
        if alias in result:
            continue
        missing = [name for name in targets if name not in result]
        if missing:
            errs.append(f"entry alias {alias} 缺目标文件：{missing}")
            continue
        merged: Set[str] = set()
        for name in targets:
            merged.update(result[name])
        result[alias] = merged
    return result


def _operator_path(kb: Path) -> Path:
    for path in (kb / "operators.json", kb / "operators" / "operators.json"):
        if path.is_file():
            return path
    return kb / "operators" / "operators.json"


def _fixes_path(kb: Path) -> Path:
    for path in (kb / "fixes.json", kb / "validation" / "fixes.json"):
        if path.is_file():
            return path
    return kb / "validation" / "fixes.json"


def _resolve_ref(ref: str, kb: Path, operators: Dict[str, Any],
                 refs: Dict[str, Set[str]], errs: List[str]) -> None:
    """Resolve `file.json::entry:<construct>` or `operators.json::stage:<id>::<stage>`.

    References are intentionally explicit.  A bare construct or a missing file
    is rejected so a renamed/removed KB entry cannot silently leave a dangling
    citation behind.
    """
    if not isinstance(ref, str) or not ref.strip():
        errs.append("operators.json: kb_refs 含空引用")
        return
    if REF_SEP not in ref:
        errs.append(f"operators.json: kb_ref {ref!r} 缺少显式 file{REF_SEP} 前缀")
        return
    filename, target = ref.split(REF_SEP, 1)
    filename = filename.strip()
    target = target.strip()
    if not filename or not target:
        errs.append(f"operators.json: kb_ref {ref!r} 格式为空")
        return
    if target.startswith("stage:"):
        if filename != "operators.json":
            errs.append(f"operators.json: stage 引用必须指向 operators.json：{ref!r}")
            return
        payload = target[len("stage:"):]
        if REF_SEP not in payload:
            errs.append(f"operators.json: stage 引用应为 operators.json::stage:<semantic_id>{REF_SEP}<stage>，实际 {ref!r}")
            return
        semantic_id, stage_name = payload.split(REF_SEP, 1)
        matches = [
            op for op in operators.get("operators", [])
            if isinstance(op, dict) and op.get("semantic_id") == semantic_id
        ]
        if len(matches) != 1:
            errs.append(f"operators.json: stage 引用 {ref!r} 找不到唯一 operator semantic_id={semantic_id!r}")
            return
        if not any(isinstance(stage, dict) and stage.get("stage") == stage_name
                   for stage in matches[0].get("stages", [])):
            errs.append(f"operators.json: stage 引用 {ref!r} 找不到对应 stage")
        return

    if filename == "operators.json":
        errs.append(f"operators.json: 非 stage 的 operators 引用无效：{ref!r}")
        return
    if filename not in refs:
        errs.append(f"operators.json: kb_ref {ref!r} 指向不存在的 entry 文件")
        return
    if not target.startswith("entry:"):
        errs.append(f"operators.json: entry 引用应为 {filename}{REF_SEP}entry:<construct>，实际 {ref!r}")
        return
    construct = target[len("entry:"):]
    if construct not in refs[filename]:
        errs.append(f"operators.json: kb_ref {ref!r} 在 {filename} 中找不到 construct")


def _validate_operator_contract(kb: Path, canonical_ids: Set[str],
                                errs: List[str], verbose: bool) -> Dict[str, int]:
    path = _operator_path(kb)
    d = _read_json(path, errs)
    stats = {"operators": 0, "stages": 0}
    if not isinstance(d, dict):
        return stats
    if d.get("semantic_taxonomy") is not None and set(d.get("semantic_taxonomy", [])) != STD:
        errs.append(f"operators.json: semantic_taxonomy 必须原样保留五类 {sorted(STD)}")
    elif d.get("semantic_taxonomy") is None:
        errs.append("operators.json: 缺 semantic_taxonomy")

    catalog = d.get("machine_check_catalog")
    if not isinstance(catalog, dict):
        errs.append("operators.json: 缺 machine_check_catalog")
        catalog = {}
    for check_id in REQUIRED_MACHINE_CHECKS:
        item = catalog.get(check_id)
        if not isinstance(item, dict):
            errs.append(f"operators.json: machine_check_catalog 缺 {check_id}")
            continue
        if not item.get("required"):
            errs.append(f"operators.json: machine_check_catalog/{check_id} 必须标记 required=true")
        for field in ("what", "pass_condition"):
            if not str(item.get(field, "")).strip():
                errs.append(f"operators.json: machine_check_catalog/{check_id}.{field} 为空")

    operators = d.get("operators")
    if not isinstance(operators, list) or not operators:
        errs.append("operators.json: operators 必须是非空列表")
        return stats
    refs = _entry_refs(kb, errs)
    seen_ids: Set[str] = set()
    seen_names: Set[str] = set()
    for oi, op in enumerate(operators):
        if not isinstance(op, dict):
            errs.append(f"operators.json#{oi}: operator 必须是对象")
            continue
        name = op.get("name")
        where = f"operators.json/{name or oi}"
        for key in ("name", "stages", "roles", "aliases", "evidence"):
            if key not in op:
                errs.append(f"{where}: 缺 {key}")
        if isinstance(name, str):
            if name in seen_names:
                errs.append(f"{where}: operator name 重复")
            seen_names.add(name)

        sid = op.get("semantic_id")
        if sid not in OPERATOR_IDS:
            errs.append(f"{where}: semantic_id={sid!r} 不在算子 canonical 集 {sorted(OPERATOR_IDS)}")
        elif sid not in canonical_ids:
            errs.append(f"{where}: semantic_id={sid!r} 未在 knowledge/semantic_ids.json 定义")
        elif sid in seen_ids:
            errs.append(f"{where}: semantic_id={sid!r} 在 operators.json 中重复")
        else:
            seen_ids.add(sid)

        if "measurement_status" in op:
            errs.append(f"{where}: 旧 measurement_status 必须替换为 evidence_status + evidence_note")
        status = op.get("evidence_status")
        if status not in EVIDENCE_STATUSES:
            errs.append(f"{where}: evidence_status={status!r} 必须属于 {sorted(EVIDENCE_STATUSES)}")
        if not str(op.get("evidence_note", "")).strip():
            errs.append(f"{where}: evidence_note 为空")
        if not isinstance(op.get("evidence"), list) or not op.get("evidence"):
            errs.append(f"{where}: evidence 必须是非空列表")

        mc = op.get("machine_check")
        if not isinstance(mc, list) or not mc:
            errs.append(f"{where}: machine_check 必须是非空检查 ID 列表")
        else:
            if len(mc) != len(set(mc)):
                errs.append(f"{where}: machine_check 含重复项")
            unknown = set(mc) - set(catalog)
            if unknown:
                errs.append(f"{where}: machine_check 含未定义检查 {sorted(unknown)}")
            missing = set(REQUIRED_MACHINE_CHECKS) - set(mc)
            if missing:
                errs.append(f"{where}: machine_check 缺必需检查 {sorted(missing)}")

        stages = op.get("stages")
        if not isinstance(stages, list) or not stages:
            errs.append(f"{where}: stages 必须是非空列表")
            continue
        stats["operators"] += 1
        seen_stages: Set[str] = set()
        stage_names = {
            st.get("stage") for st in stages
            if isinstance(st, dict) and isinstance(st.get("stage"), str)
        }
        for si, stage in enumerate(stages):
            if not isinstance(stage, dict):
                errs.append(f"{where}/stage#{si}: 必须是对象")
                continue
            stage_name = stage.get("stage")
            swhere = f"{where}/{stage_name or si}"
            stats["stages"] += 1
            for key in ("stage", "depends_on", "typical_role", "semantics", "code_signature",
                        "semantic_ids", "instrument", "phase"):
                if key not in stage:
                    errs.append(f"{swhere}: 缺 {key}")
            if isinstance(stage_name, str):
                if stage_name in seen_stages:
                    errs.append(f"{swhere}: stage 名重复")
                seen_stages.add(stage_name)
            for dep in stage.get("depends_on", []) if isinstance(stage.get("depends_on"), list) else []:
                if dep not in stage_names:
                    errs.append(f"{swhere}: depends_on 指向不存在的 stage {dep!r}")

            legacy = stage.get("semantics")
            if not isinstance(legacy, list) or not legacy:
                errs.append(f"{swhere}: 原 semantics 五类必须是非空列表")
                legacy = []
            bad_legacy = [x for x in legacy if x not in STD]
            if bad_legacy:
                errs.append(f"{swhere}: semantics 含非五类值 {bad_legacy}")

            sids = stage.get("semantic_ids")
            if not isinstance(sids, list) or not sids:
                errs.append(f"{swhere}: semantic_ids 必须是非空列表")
                sids = []
            if len(sids) != len(set(sids)):
                errs.append(f"{swhere}: semantic_ids 含重复项")
            for sem_id in sids:
                if sem_id not in canonical_ids:
                    errs.append(f"{swhere}: semantic_ids 含未知 canonical ID {sem_id!r}")
                elif sem_id == "unknown":
                    errs.append(f"{swhere}: curated stage 不允许写 unknown")
            for legacy_sem in legacy:
                if legacy_sem in LEGACY_TO_CANONICAL and not (set(sids) & LEGACY_TO_CANONICAL[legacy_sem]):
                    errs.append(
                        f"{swhere}: 原 semantics={legacy_sem} 与 semantic_ids={sids} 不兼容；"
                        f"至少需要一个 {sorted(LEGACY_TO_CANONICAL[legacy_sem])}")

            kb_refs = stage.get("kb_refs", [])
            if kb_refs is not None and not isinstance(kb_refs, list):
                errs.append(f"{swhere}: kb_refs 必须是列表")
            elif isinstance(kb_refs, list):
                if len(kb_refs) != len(set(kb_refs)):
                    errs.append(f"{swhere}: kb_refs 含重复项")
                for ref in kb_refs:
                    _resolve_ref(ref, kb, d, refs, errs)

    expected_missing = OPERATOR_IDS - seen_ids
    if expected_missing:
        errs.append(f"operators.json: canonical 算子未覆盖：{sorted(expected_missing)}")
    if verbose:
        print(f"operators.json: {stats['operators']} 个算子, {stats['stages']} 个 stage, "
              f"{len(refs)} 个 entry 文件可引用")
    return stats


def validate_kb(kb: Path, *, verbose: bool = False) -> List[str]:
    """Return all validation errors.  An empty list means PASS."""
    errs: List[str] = []
    total = 0
    canonical_ids = _load_canonical_ids(kb, errs)
    entry_files = _entry_files(kb)
    if not entry_files:
        errs.append("knowledge/cards 与 knowledge/languages 下没有可校验的硬件条目文件")
    for path in entry_files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # check() emits the stable path-specific parse/read diagnostic.
            raw = None
        if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
            continue
        allowed = REC if "recorder" in path.name else STD
        d = check(path, allowed, errs, canonical_ids)
        entries = d.get("entries", []) if isinstance(d, dict) else []
        n = len(entries)
        total += n
        sems: Dict[str, int] = {}
        for e in entries:
            sem = e.get("semantic_id", e.get("semantic_legacy", e.get("semantic")))
            sems[sem] = sems.get(sem, 0) + 1
        if verbose:
            rel = path.relative_to(kb).as_posix()
            print(f"{rel}: {n} 条  语义分布={sems}")

    _validate_operator_contract(kb, canonical_ids, errs, verbose)

    fx_path = _fixes_path(kb)
    if fx_path.exists():
        fx = _read_json(fx_path, errs)
        if isinstance(fx, dict):
            need = {"依赖等待", "装载未掩盖", "拖尾", "序言", "算得慢"}
            got = {x.get("loss_type") for x in fx.get("fixes", []) if isinstance(x, dict)}
            missing = need - got
            if missing:
                errs.append(f"fixes.json 缺损失类型：{sorted(missing)}")
            for f in fx.get("fixes", []):
                if not isinstance(f, dict):
                    errs.append("fixes.json: fixes 含非对象项")
                    continue
                for k in ("loss_type", "symptom", "evidence_to_check", "fixes"):
                    if k not in f:
                        errs.append(f"fixes.json/{f.get('loss_type')}: 缺 {k}")
                for g in f.get("fixes", []):
                    if not isinstance(g, dict):
                        errs.append(f"fixes.json/{f.get('loss_type')}: 改法含非对象项")
                        continue
                    for k in ("fix", "how", "cost", "verify"):
                        if k not in g:
                            errs.append(f"fixes.json/{f.get('loss_type')}: 改法缺 {k}")
            if verbose:
                print(f"fixes.json: {len(fx.get('fixes', []))} 类损失, "
                      f"{sum(len(f['fixes']) for f in fx.get('fixes', []) if isinstance(f, dict))} 条改法")
    else:
        errs.append("缺文件 fixes.json")
    if verbose:
        print(f"TOTAL entries={total}")
    return errs


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("knowledge")
    if root.name == "kb" and not root.exists() and Path("knowledge").is_dir():
        root = Path("knowledge")
    kb = root
    errs = validate_kb(kb, verbose=True)
    if errs:
        print("KB VALIDATE FAIL")
        for err in errs[:80]:
            print("  -", err)
        if len(errs) > 80:
            print(f"  ... 另有 {len(errs) - 80} 项错误")
        return 1
    print("KB VALIDATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
