#!/usr/bin/env python3
"""check_gate_item.py（骨架版）— 项目侧 GATE 判定适配器（G00-G24）。

流程位置
  skill 侧 ``weaveprof/pipeline/validate/validate_all.py`` 是固定入口。它按 frozen contract 的
  ``execution.gate_command_template`` 逐条调起本文件（argv-only、关闭 shell），
  判定顺序与 preflight 完全一致：退出码 -> 证据文件存在/可读 -> 证据 sha256 -> PASS。

谁来调 / CLI
  python3 -B check_gate_item.py --contract <lock> --gate-id G19 --out <p.json>
  一次只判一个门；未知 gate id 退出码 2。

两类门
  * 能力门（G12 buffer ABI / G14 warp 拓扑）：只看 ABI + 仓库状态就能判，纯 CPU
    阶段即可定论，直接复用 ``check_preflight_item`` 里的 E12/E15；
  * 运行时门（其余）：要看测量阶段落在 run_root 下的产物（correctness / overhead /
    raw+sidecar / decoded events / nsys manifest / hw metrics / alignment / candidate）。
    编排器总是先跑完测量阶段再跑 validate_all.py，所以此时产物应当存在；产物还没
    生成时（纯 CPU 编写阶段）这些门必须**如实报缺并 FAIL**，绝不允许假绿灯。

接新 target 要改的地方
  只有下面 ``===== 接新 target 时只需要改这里 =====`` 配置区：
  产物相对路径约定、run_root 兜底、解码 metadata 的异常计数键名映射、覆盖率阈值。
  能力门、覆盖率算法、证据 schema、退出码语义都是通用的，不要动。

参考实现（dsv4pro/decode，已通过 25 门禁 —— 这是**范本路径**，不是本项目路径；
本文件由它泛化而来，对照语义可以，但不要照抄它的配置值）
  DSA-learn/mk/h20/cutedsl/deepseek_v4_pro/decode/profile/validation/check_gate_item.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS = Path(__file__).resolve()
_VALIDATION = _THIS.parent
if str(_VALIDATION) not in sys.path:
    sys.path.insert(0, str(_VALIDATION))
import check_preflight_item as cp  # noqa: E402  (共用 Ctx + 能力检查 + 小工具)

SCHEMA = "fast-instrument-gate-item/v1"
ALL_GATE_IDS = ["G{:02d}".format(index) for index in range(0, 25)]

_REAL_IMPL = ("参考实现（dsv4pro/decode，已通过 25 门禁）DSA-learn/mk/h20/cutedsl/"
              "deepseek_v4_pro/decode/profile/validation/check_gate_item.py")

# ======================================================================
# ===== 接新 target 时只需要改这里 (PROJECT CONFIG) =====
# ======================================================================
GATE_PROJECT: Dict[str, Any] = {
    # 1) run_root 兜底。优先读 contract 的 project_layout.run_root；没有才用这个相对
    #    version_root 的路径。真实项目是 results/profile_runs/v2_dsv4pro_decode。
    "run_root_fallback_relpath": "results/profile_runs/run",

    # 2) 产物相对路径约定（相对 run_root）。``{rank}`` 会被 display_rank 替换。
    #    这是接新 target 最常改的一组：如果你的目录布局不同（比如把 alignment.json
    #    放在 hw/ 而不是 work/），只改这里，所有门自动跟着走。
    "artifact_relpaths": {
        "gpu_idle": "validation/gpu_idle.json",
        "offline_selftest": "validation/offline_selftest.json",
        "correctness": "validation/correctness.json",
        "overhead": "validation/overhead.json",
        "sw_manifest": "sw/stream_manifest.json",
        "sw_events": "sw/events_rank{rank}.json",
        "sw_sidecar": "sw/raw_rank{rank}.bin.json",
        "hw_manifest": "hw/capture_manifest.json",
        "hw_metrics": "hw/hw_metrics.json",
        "alignment": "work/alignment.json",
        "candidate": "work/candidate.perfetto.json.gz",
    },

    # 3) 解码产物里"异常计数"的键名映射。
    #    不同解码器叫法不一样，门禁逻辑不该跟着改，所以在这里做一次翻译：
    #      - 真实项目 decode_profile 直接给 metadata.unpaired_records / unknown_events
    #      - 骨架 decode_profile 给 metadata.counters.{unmatched_starts,unmatched_ends,
    #        unknown_task_ids,...}
    #    语义 = "把所有命中的候选加起来"，因为骨架把 unpaired 拆成了 starts/ends 两个。
    #    一个都没命中 -> None -> 门 FAIL（fail-closed：查不到不等于是 0）。
    "anomaly_counter_paths": {
        "unpaired_records": ("decoded.metadata.unpaired_records",
                             "decoded.metadata.counters.unmatched_starts",
                             "decoded.metadata.counters.unmatched_ends"),
        "unknown_events": ("decoded.metadata.unknown_events",
                           "decoded.metadata.counters.unknown_task_ids"),
        "negative_durations": ("decoded.metadata.counters.negative_durations",),
    },
    # 4) 解码产物里"标量"的键名映射（语义 = 取第一个命中的候选，不求和）。
    "scalar_paths": {
        "events_total": ("decoded.metadata.events_total",),
        "raw_start_count": ("decoded.metadata.raw_start_count",
                            "decoded.metadata.records_total"),
        "raw_end_count": ("decoded.metadata.raw_end_count",),
        "records_total": ("decoded.metadata.records_total",),
        "records_expected": ("decoded.metadata.records_expected",),
        "decode_valid": ("decoded.metadata.diagnostic_summary.valid",
                         "decoded.metadata.diagnostic_valid"),
        "timestamps_monotonic": ("decoded.metadata.diagnostic_summary.timestamps_monotonic",
                                 "decoded.metadata.timestamps_monotonic"),
    },
    # 5) 容量/溢出证据的键名映射。前缀 sidecar. / decoded. / abi. 选择数据源。
    #    真实项目：sidecar 直接给 records_per_lane + overflow_any（独立 prof_ovf 张量）。
    #    骨架扁平 buffer 用 sentinel，没有溢出张量，所以退化成"解码器报的截断 track"。
    "capacity_paths": {
        "records_per_lane": ("sidecar.records_per_lane",
                             "decoded.metadata.record_counts"),
        "overflow_flag": ("sidecar.overflow_any", "sidecar.overflow"),
        "truncated": ("decoded.metadata.truncated_tracks",),
        "max_records_per_lane": ("abi.buffer.max_records_per_lane",
                                 "sidecar.max_records",
                                 "decoded.metadata.max_records"),
    },
    # 6) SM 采集范围（G13）。expected 只允许从"声明侧"推（ABI / contract），
    #    绝不能拿 sidecar 里测出来的集合当期望值 —— 那就成了自己证明自己。
    "sm_scope_paths": {
        "mode": ("abi.capture_scope.mode", "contract.capture.software_sm_scope.mode"),
        "declared_smids": ("abi.capture_scope.selected_physical_smids",
                           "contract.capture.software_sm_scope.selected_physical_smids"),
        "measured_smids": ("sidecar.capture_scope.selected_physical_smids",),
        # mode 表示"全格采集"时的字面量（不同项目写法不同）。
        "all_mode_tokens": ("all", "all_physical_sms", "all_sms"),
        # 全格采集且没有显式列表时，用这里推出 expected = range(num_sms)。
        "num_sms_paths": ("abi.launch.num_sms", "contract.capture.num_sms",
                          "sidecar.num_sms"),
    },

    # 7) G19 算子 task 覆盖率阈值：最大连续未覆盖空洞 <= max(span*百分比, 下限 ns)。
    #    真实项目用 5% / 50us。放宽这两个数等于放宽"插桩完整性"的验收标准，
    #    改之前先想清楚是不是真的有一段 kernel 你不打算插桩。
    "coverage_max_gap_percent": 5,
    "coverage_max_gap_floor_ns": 50_000,

    # 7b) G19 对齐质量硬门禁：shift-only 对齐若 SW/HW 窗口 span 相对误差超过此值，
    #     或端点残差（ns）超过此值，HW counter 会相对 software span 错位超过一个
    #     task 间隔 —— 必须 FAIL 而不是只警告。历史坑：曾有 method/scale 合法但
    #     窗口错位（span 误差 10.5% / 端点残差 1.75ms）的对齐靠 method-only 检查
    #     假绿灯通过；好的实测在 span 误差 <=1.8% / 残差 <=0.5ms。
    "alignment_max_relative_span_error": 0.03,
    "alignment_max_abs_residual_ns": 500_000,

    # 8) HW 采样率 / trace 体积上限。默认沿用 preflight 配置区的同名项，保持单一来源。
    "hw_sample_rate_hz": cp.PROJECT["hw_sample_rate_hz"],
    "max_trace_bytes": cp.PROJECT["max_trace_bytes"],

    # 9) nsys capture manifest 里表示"这次抓取有效"的 verdict 字面量。
    "hw_manifest_pass_verdicts": ("PASS", "OK", "VALID"),

    # 10) G00 允许信任的 gpu_idle 快照 probe 值 —— 必须与 wait_for_gpu_idle.py 配置区的
    #     PROBE_MEASURED 一致。用**白名单**而不是"排除 nvidia-smi_absent"的黑名单：
    #     以后新增任何"探测失败"标记（或者压根没写 probe 字段的旧快照）都会自动被拦，
    #     不需要回来补黑名单。认不出的 probe -> G00 FAIL（吵，但不会白送）。
    "gpu_idle_trusted_probes": ("nvidia-smi",),
}
# ======================================================================
# ===== 配置区结束；下面是与项目无关的通用逻辑 =====
# ======================================================================


def load_json_maybe_gz(path: Path) -> Any:
    """读 JSON，透明支持 .gz（candidate trace 是 gzip 的）。"""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


class GateCtx(cp.Ctx):
    """preflight Ctx + 解析好的 run_root 产物位置。"""

    def __init__(self, contract: Dict[str, Any], contract_path: Path):
        super().__init__(contract, contract_path)
        layout = self.contract.get("project_layout", {})
        run_root = layout.get("run_root") if isinstance(layout, dict) else None
        if not run_root:
            run_root = cp.first(self.contract, ("project_layout.run_root",
                                                "artifacts.run_root", "run.run_root"))
        self.run_root = (Path(run_root) if run_root
                         else self.version_root / GATE_PROJECT["run_root_fallback_relpath"])
        rank = self.display_rank
        self.artifacts: Dict[str, Path] = {}
        for key, relpath in GATE_PROJECT["artifact_relpaths"].items():
            self.artifacts[key] = self.run_root / relpath.format(rank=rank)
        # contract 显式声明的 candidate trace 路径优先（可能落在 run_root 之外）。
        candidate_cfg = cp.ckey(self.contract, "candidate_perfetto")
        if candidate_cfg:
            self.artifacts["candidate"] = Path(candidate_cfg)
        self._cache: Dict[str, Any] = {}

    def present(self, *keys: str) -> List[str]:
        """返回这些 key 里"文件不存在"的那些。"""
        return [key for key in keys if not self.artifacts[key].is_file()]

    def doc(self, key: str) -> Optional[Any]:
        """读一个产物并缓存；不存在或读不动返回 None（由调用方判 FAIL）。"""
        if key in self._cache:
            return self._cache[key]
        path = self.artifacts.get(key)
        value = None
        if path is not None and path.is_file():
            try:
                value = load_json_maybe_gz(path)
            except (OSError, ValueError):
                value = None
        self._cache[key] = value
        return value

    def sources(self) -> Dict[str, Any]:
        """给 pick() 用的数据源表：前缀名 -> 文档。"""
        return {"decoded": self.doc("sw_events") or {},
                "sidecar": self.doc("sw_sidecar") or {},
                "manifest": self.doc("sw_manifest") or {},
                "hw_manifest": self.doc("hw_manifest") or {},
                "hw_metrics": self.doc("hw_metrics") or {},
                "alignment": self.doc("alignment") or {},
                "abi": self.abi,
                "contract": self.contract}


def pick(sources: Dict[str, Any], dotted_paths) -> Any:
    """按 ``<源前缀>.<点路径>`` 取第一个命中的值（None 视为未命中）。"""
    for dotted in dotted_paths:
        head, _, rest = str(dotted).partition(".")
        node = sources.get(head)
        if node is None:
            continue
        value = cp.dig(node, rest, None) if rest else node
        if value is not None:
            return value
    return None


def pick_sum(sources: Dict[str, Any], dotted_paths) -> Optional[int]:
    """把所有命中的候选加起来（骨架把 unpaired 拆成 starts/ends 两个计数）。

    一个都没命中就返回 None —— 让调用方 FAIL，而不是当成 0 放过去。
    """
    total = None
    for dotted in dotted_paths:
        head, _, rest = str(dotted).partition(".")
        node = sources.get(head)
        if node is None:
            continue
        value = cp.dig(node, rest, None) if rest else None
        if value is None:
            continue
        try:
            total = (0 if total is None else total) + int(value)
        except (TypeError, ValueError):
            continue
    return total


def anomaly(gc: GateCtx, logical: str) -> Optional[int]:
    paths = GATE_PROJECT["anomaly_counter_paths"].get(logical, ())
    return pick_sum(gc.sources(), paths)


def scalar(gc: GateCtx, logical: str) -> Any:
    paths = GATE_PROJECT["scalar_paths"].get(logical, ())
    return pick(gc.sources(), paths)


def missing_runtime(gc: GateCtx, *keys: str) -> Dict[str, Any]:
    missing = gc.present(*keys)
    return {"runtime_evidence_missing": [str(gc.artifacts[k]) for k in missing],
            "required_artifacts": [str(gc.artifacts[k]) for k in keys]}


def _counter_hint(logical: str) -> str:
    return ("解码产物里找不到 {!r}：请在 GATE_PROJECT['anomaly_counter_paths'] 里"
            "补上你的 decoder 实际使用的键名（fail-closed，查不到一律 FAIL）"
            .format(logical))


# ----------------------------------------------------------------------
# 能力门（纯 CPU 阶段即可定论）
# ----------------------------------------------------------------------
def gate_g12(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    ok, findings = cp.check_e12(gc)
    findings = dict(findings)
    findings["capability"] = "buffer_abi_paths_and_sha256_consistent"
    return ok, findings


def gate_g14(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    ok, findings = cp.check_e15(gc)
    findings = dict(findings)
    findings["capability"] = "warp_topology_recomputable"
    return ok, findings


# ----------------------------------------------------------------------
# 运行时门（需要测量阶段的产物）
# ----------------------------------------------------------------------
def gate_g00(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    """GPU 空闲。**不是只看 idle 这个布尔值** —— 还要确认它是真探测出来的。

    历史坑（本轮修掉的）：``wait_for_gpu_idle.py`` 在没有 nvidia-smi 的机器上曾经写
    ``idle=true`` + ``probe="nvidia-smi_absent"`` 并退出 0，而这里只读 ``snap["idle"]``，
    于是 **G00 在任何纯 CPU 机器上无条件放行** —— 一个门禁白送，还留下一份写着
    idle=true 的"证据"。GPU 状态探测不到 = 不可观测 = FAIL，不是 idle。

    判法是白名单：``probe`` 必须落在配置区 ``gpu_idle_trusted_probes`` 里，才轮到看
    ``idle``。没写 probe 的旧快照、或任何新的"探测失败"标记，都直接 FAIL。
    """
    miss = missing_runtime(gc, "gpu_idle")
    if miss["runtime_evidence_missing"]:
        return False, miss
    snap = gc.doc("gpu_idle") or {}
    trusted = tuple(GATE_PROJECT["gpu_idle_trusted_probes"])
    probe = snap.get("probe")
    probe_trusted = isinstance(probe, str) and probe in trusted
    idle_flag = bool(snap.get("idle"))
    out: Dict[str, Any] = {"gpu_idle_snapshot": snap, "probe": probe,
                           "trusted_probes": list(trusted),
                           "probe_trusted": probe_trusted, "idle_flag": idle_flag}
    if not probe_trusted:
        out["reason"] = (
            "gpu_idle 快照的 probe={!r} 不在可信探测方式白名单 {} 里：GPU 状态**没有被真的"
            "测到**，因此不算空闲（fail-closed）。若快照写着 idle=true 也一样不算 —— "
            "请在有 nvidia-smi 的机器上重跑 wait_for_gpu_idle.py；如果本项目改用别的探测"
            "方式，把它的 probe 标记加进 GATE_PROJECT['gpu_idle_trusted_probes']。"
            .format(probe, list(trusted)))
        return False, out
    if not idle_flag:
        out["reason"] = "探测成功但 GPU 仍忙（见快照 busy_gpus / timed_out）。"
    return idle_flag, out


def gate_g01(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    bundle = gc.capture.get("information_bundle", "sw+hw")
    needed = ["correctness", "overhead", "sw_manifest", "sw_events"]
    if bundle in ("hw", "sw+hw"):
        needed += ["hw_manifest", "hw_metrics", "alignment", "candidate"]
    miss = missing_runtime(gc, *needed)
    ok = not miss["runtime_evidence_missing"]
    out = {"bundle": bundle}
    out.update(miss)
    return ok, out


def gate_g02(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    miss = missing_runtime(gc, "correctness")
    if miss["runtime_evidence_missing"]:
        return False, miss
    doc = gc.doc("correctness") or {}
    ok = bool(doc.get("all_passed")) and doc.get("sample_only") is not True
    return ok, {"all_passed": doc.get("all_passed"), "check_scope": doc.get("check_scope"),
                "sample_only": doc.get("sample_only"),
                "max_abs_err": doc.get("max_abs_err")}


def gate_g03(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    miss = missing_runtime(gc, "overhead")
    if miss["runtime_evidence_missing"]:
        return False, miss
    doc = gc.doc("overhead") or {}
    configs = doc.get("configurations", doc.get("medians_ns", {}))
    within = doc.get("within_max", doc.get("within_maximum"))
    # 三档（off / entry_only / full）必须都测过，少一档就不可能算出正确的开销比。
    ok = bool(within) and len(configs) >= 3
    return ok, {"within_max": within, "overhead_ratio": doc.get("overhead_ratio"),
                "configurations": configs}


def gate_g04(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    cap_ok, cap = cp.check_e02(gc)
    miss = missing_runtime(gc, "sw_events")
    if miss["runtime_evidence_missing"]:
        out = {"clock_capability_ok": cap_ok}
        out.update(cap)
        out.update(miss)
        return False, out
    valid = scalar(gc, "decode_valid")
    monotonic = scalar(gc, "timestamps_monotonic")
    ok = cap_ok and bool(valid) and monotonic is not False
    return ok, {"clock_capability_ok": cap_ok, "decode_valid": valid,
                "timestamps_monotonic": monotonic}


def gate_g05(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    cap_ok, cap = cp.check_e07(gc)
    miss = missing_runtime(gc, "sw_events")
    if miss["runtime_evidence_missing"]:
        out = {"write_isolation_capability_ok": cap_ok}
        out.update(cap)
        out.update(miss)
        return False, out
    unpaired = anomaly(gc, "unpaired_records")
    ok = cap_ok and unpaired == 0
    out = {"write_isolation_capability_ok": cap_ok, "unpaired_records": unpaired}
    if unpaired is None:
        out["hint"] = _counter_hint("unpaired_records")
    return ok, out


def gate_g06(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    cap_ok, cap = cp.check_e08(gc)
    miss = missing_runtime(gc, "sw_sidecar")
    if miss["runtime_evidence_missing"]:
        out = {"capacity_capability_ok": cap_ok}
        out.update(cap)
        out.update(miss)
        return False, out
    src = gc.sources()
    paths = GATE_PROJECT["capacity_paths"]
    counts = pick(src, paths["records_per_lane"]) or []
    overflow_flag = pick(src, paths["overflow_flag"])
    truncated = pick(src, paths["truncated"])
    maxr = cp.as_int(pick(src, paths["max_records_per_lane"]), 0)
    try:
        peak = max(int(x) for x in counts) if counts else 0
    except (TypeError, ValueError):
        counts, peak = [], -1
    # 三个独立信号，任一说"满了"就 FAIL：显式溢出标志、截断 track、任一 lane 达到上限。
    # maxr<=0 说明既没从 ABI 也没从 sidecar 拿到容量上限 -> 无法证明没溢出 -> FAIL。
    ok = (cap_ok and maxr > 0 and peak >= 0 and peak < maxr
          and overflow_flag is not True
          and not truncated)
    return ok, {"capacity_capability_ok": cap_ok, "overflow_flag": overflow_flag,
                "truncated_tracks": truncated, "max_lane_count": peak,
                "max_records_per_lane": maxr, "lane_count_samples": len(counts)}


def gate_g07(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    cap_ok, cap = cp.check_e09(gc)
    miss = missing_runtime(gc, "sw_events")
    if miss["runtime_evidence_missing"]:
        out = {"entry_sync_capability_ok": cap_ok}
        out.update(cap)
        out.update(miss)
        return False, out
    valid = scalar(gc, "decode_valid")
    ok = cap_ok and bool(valid)
    return ok, {"entry_sync_capability_ok": cap_ok, "decode_valid": valid}


def gate_g08(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    cap_ok, _cap = cp.check_e10(gc)
    dec_ok, _dec = cp.check_e16(gc)
    miss = missing_runtime(gc, "sw_events")
    if miss["runtime_evidence_missing"]:
        out = {"event_semantics_capability_ok": cap_ok, "decoder_selftest_ok": dec_ok}
        out.update(miss)
        return False, out
    unknown = anomaly(gc, "unknown_events")
    unpaired = anomaly(gc, "unpaired_records")
    ok = cap_ok and dec_ok and unknown == 0 and unpaired == 0
    out = {"event_semantics_capability_ok": cap_ok, "decoder_selftest_ok": dec_ok,
           "unknown_events": unknown, "unpaired_records": unpaired}
    if unknown is None or unpaired is None:
        out["hint"] = _counter_hint("unknown_events/unpaired_records")
    return ok, out


def gate_g09(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    miss = missing_runtime(gc, "sw_manifest")
    if miss["runtime_evidence_missing"]:
        return False, miss
    doc = gc.doc("sw_manifest") or {}
    valid = cp.first(doc, ("decode_summary.diagnostic_valid",
                           "decode_summary.valid",
                           "diagnostic_summary.valid"))
    ok = bool(valid)
    out = {"target_kernel": doc.get("target_kernel"), "decode_valid": valid}
    if valid is None:
        out["hint"] = ("stream_manifest.json 里没有 decode_summary.diagnostic_valid："
                       "请让 run_profile.py 在 manifest 里回填解码诊断结论，"
                       "否则 G09 无法证明这次采集自洽（参考真实实现 {}）".format(_REAL_IMPL))
    return ok, out


def gate_g10(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    world = cp.as_int(cp.ckey(gc.contract, "world_size", 1), 1)
    miss = missing_runtime(gc, "sw_manifest")
    if miss["runtime_evidence_missing"]:
        out = {"world_size": world}
        out.update(miss)
        return False, out
    doc = gc.doc("sw_manifest") or {}
    measured = cp.as_int(cp.first(doc, ("world_size", "ranks_total")), -1)
    if measured < 0:
        ranks = doc.get("ranks")
        measured = len(ranks) if isinstance(ranks, list) else -1
    ok = (world == 1) or (measured == world)
    return ok, {"world_size": world, "manifest_world_size": measured,
                "single_rank_launch": world == 1}


def gate_g11(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    miss = missing_runtime(gc, "hw_metrics")
    if miss["runtime_evidence_missing"]:
        return False, miss
    doc = gc.doc("hw_metrics") or {}
    ok = bool(doc.get("samples") or doc.get("metrics") or doc.get("counters"))
    return ok, {"has_samples": ok}


def gate_g13(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    """SM 采集范围必须与"声明"完全一致 —— 不允许事后过滤。

    expected 只从声明侧（ABI capture_scope / contract capture.software_sm_scope）推；
    拿 sidecar 里测出来的集合当期望值就成了自证，必须避免。
    """
    e05_ok, _ = cp.check_e05(gc)
    e06_ok, _ = cp.check_e06(gc)
    cap_ok = e05_ok and e06_ok
    src = gc.sources()
    paths = GATE_PROJECT["sm_scope_paths"]
    mode = str(pick(src, paths["mode"]) or "").lower()
    declared = pick(src, paths["declared_smids"])
    expected: List[int] = []
    basis = "declared_list"
    if isinstance(declared, list) and declared:
        expected = sorted(int(s) for s in declared)
    elif mode in paths["all_mode_tokens"]:
        num_sms = cp.as_int(pick(src, paths["num_sms_paths"]), 0)
        expected = list(range(num_sms))
        basis = "all_mode_range(num_sms={})".format(num_sms)
    miss = missing_runtime(gc, "sw_sidecar")
    findings: Dict[str, Any] = {"sm_scope_capability_ok": cap_ok, "mode": mode,
                                "expected_basis": basis,
                                "expected_physical_smids": expected[:32],
                                "expected_count": len(expected)}
    if miss["runtime_evidence_missing"]:
        findings.update(miss)
        return False, findings
    measured_raw = pick(src, paths["measured_smids"])
    measured = sorted(int(s) for s in measured_raw) if isinstance(measured_raw, list) else None
    findings["measured_count"] = len(measured) if measured is not None else None
    findings["measured_physical_smids"] = (measured or [])[:32]
    if not expected:
        findings["hint"] = ("ABI 与 contract 都没声明 SM 采集范围，无法证明"
                            "「解码出来的 SM 集合 == 声明的集合」。请在 ABI 的 "
                            "capture_scope（mode + selected_physical_smids）或 contract 的 "
                            "capture.software_sm_scope 里声明；只抓子集时必须显式列出 SM 号。"
                            "参考真实实现 {}".format(_REAL_IMPL))
        return False, findings
    ok = cap_ok and measured == expected
    return ok, findings


def gate_g15(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    dec_ok, _dec = cp.check_e16(gc)
    miss = missing_runtime(gc, "sw_manifest", "sw_events")
    if miss["runtime_evidence_missing"]:
        out = {"decoder_selftest_ok": dec_ok}
        out.update(miss)
        return False, out
    unknown = anomaly(gc, "unknown_events")
    unpaired = anomaly(gc, "unpaired_records")
    events_total = cp.as_int(scalar(gc, "events_total"), -1)
    # 每个事件恰好由一条 begin + 一条 end 组成：raw 记录总数必须是事件数的 2 倍。
    # 优先用 raw_start_count+raw_end_count；没有就退回 records_total/records_expected。
    starts = scalar(gc, "raw_start_count")
    ends = scalar(gc, "raw_end_count")
    if starts is not None and ends is not None:
        records_total = cp.as_int(starts, -1) + cp.as_int(ends, -2)
    else:
        records_total = cp.as_int(scalar(gc, "records_total"), -1)
    balanced = (events_total >= 0 and records_total == 2 * events_total)
    ok = (dec_ok and unknown == 0 and unpaired == 0 and balanced)
    out = {"decoder_selftest_ok": dec_ok, "unknown_events": unknown,
           "unpaired_records": unpaired, "raw_start_count": starts,
           "raw_end_count": ends, "records_total_effective": records_total,
           "events_total": events_total, "records_balanced": balanced}
    if unknown is None or unpaired is None:
        out["hint"] = _counter_hint("unknown_events/unpaired_records")
    return ok, out


def _candidate_gate(gc: GateCtx, extra_keys=()) -> Tuple[bool, Dict[str, Any]]:
    keys = ("candidate",) + tuple(extra_keys)
    miss = missing_runtime(gc, *keys)
    if miss["runtime_evidence_missing"]:
        return False, miss
    return True, {"candidate": str(gc.artifacts["candidate"])}


def gate_g16(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    return _candidate_gate(gc, ("sw_manifest",))


def gate_g17(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    cap_ok, _ = cp.check_e10(gc)
    ok, findings = _candidate_gate(gc)
    findings = dict(findings)
    findings["task_phase_capability_ok"] = cap_ok
    return (ok and cap_ok), findings


def gate_g18(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    return _candidate_gate(gc)


def _union_and_gaps(intervals: List[Tuple[int, int]], win_start: int, win_end: int
                    ) -> Tuple[List[Tuple[int, int]], int, List[Tuple[int, int]]]:
    """把区间裁到 [win_start, win_end]、合并重叠，返回 (union, covered_ns, gaps)。

    gaps 是窗口内"没有任何区间覆盖"的极大子区间。这段算法与项目无关，别改。
    """
    clipped: List[Tuple[int, int]] = []
    for start, end in intervals:
        s = max(int(start), win_start)
        e = min(int(end), win_end)
        if e > s:
            clipped.append((s, e))
    clipped.sort()
    union: List[Tuple[int, int]] = []
    for s, e in clipped:
        if union and s <= union[-1][1]:
            if e > union[-1][1]:
                union[-1] = (union[-1][0], e)
        else:
            union.append((s, e))
    covered = sum(e - s for s, e in union)
    gaps: List[Tuple[int, int]] = []
    cursor = win_start
    for s, e in union:
        if s > cursor:
            gaps.append((cursor, s))
        if e > cursor:
            cursor = e
    if cursor < win_end:
        gaps.append((cursor, win_end))
    return union, covered, gaps


def _operator_task_coverage(gc: GateCtx, align_doc: Dict[str, Any]) -> Dict[str, Any]:
    """整个 kernel 窗口的"算子 TASK 覆盖率"（并入 G19）。

    前提：shift-only 对齐（scale == 1）。scale == 1 时 SW 时间轴没有被拉伸，
    所以"窗口里某段没有任何算子 task"就是真的没插桩，而不是缩放假象。

    做法：把所有 **ABI 里 layer == "task"** 的算子区间取并集
    （这样自然排除掉 layer == "ignored" 的整 kernel 锚点 + grid barrier，
    以及 layer == "phase" 的子阶段），要求最大连续未覆盖空洞
    <= max(kernel span * 5%, 50us)。

    为什么必须这么算（历史坑）：曾经只有 attention 是 task，头尾算子
    （quant / GEMM / indexer / logits / top-k / CSA / wo）根本没插桩；而当时用的是
    min/max 包络，那条"整 kernel 宽"的 ignored 大条把包络撑满，于是 100% 覆盖率
    假绿灯通过。所以：**只认 layer == "task"，且按空洞判，不按包络判。**

    注意 task 层级来自 ABI，不写死事件号 —— ABI 改了层级，这里的判定自动跟着改。
    """
    scale = cp.as_int(align_doc.get("scale", 0), 0)
    win = align_doc.get("sw_window_ns")
    span = cp.as_int(align_doc.get("sw_span_ns", 0), 0)
    if scale != 1:
        return {"applicable": False, "scale": scale,
                "reason": "scale != 1; gap semantics require shift-only alignment "
                          "(scale != 1 already fails G19)"}
    if not (isinstance(win, (list, tuple)) and len(win) == 2 and span > 0):
        return {"applicable": True, "ok": False, "scale": scale,
                "reason": "alignment lacks sw_window_ns / sw_span_ns; cannot prove coverage"}
    ev_miss = missing_runtime(gc, "sw_events")
    if ev_miss["runtime_evidence_missing"]:
        out = {"applicable": True, "ok": False, "scale": scale,
               "reason": "decoded sw events missing; cannot prove coverage"}
        out.update(ev_miss)
        return out
    win_start, win_end = int(win[0]), int(win[1])
    task_ids = {int(task["id"]) for task in gc.abi.get("tasks", [])
                if isinstance(task, dict) and task.get("layer") == "task" and "id" in task}
    doc = gc.doc("sw_events") or {}
    intervals: List[Tuple[int, int]] = []
    for ev in doc.get("events", []):
        if not isinstance(ev, dict):
            continue
        raw_id = ev.get("event_id", ev.get("task_id", ev.get("task")))
        try:
            eid = int(raw_id)
        except (TypeError, ValueError):
            continue
        if eid not in task_ids:
            continue
        start = ev.get("start_ns", ev.get("start"))
        end = ev.get("end_ns", ev.get("end"))
        if start is None or end is None:
            continue
        intervals.append((int(start), int(end)))
    _union, covered, gaps = _union_and_gaps(intervals, win_start, win_end)
    max_gap = max((e - s for s, e in gaps), default=0)
    threshold = max(span * int(GATE_PROJECT["coverage_max_gap_percent"]) // 100,
                    int(GATE_PROJECT["coverage_max_gap_floor_ns"]))
    gaps_us = sorted(([round((s - win_start) / 1000.0, 2),
                       round((e - win_start) / 1000.0, 2),
                       round((e - s) / 1000.0, 2)] for s, e in gaps),
                     key=lambda item: item[2], reverse=True)
    return {"applicable": True, "ok": max_gap <= threshold, "scale": scale,
            "kernel_span_us": round(span / 1000.0, 2),
            "operator_task_ids": sorted(task_ids),
            "operator_task_span_events": len(intervals),
            "covered_us": round(covered / 1000.0, 2),
            "coverage_fraction": round(covered / span, 4) if span else 0.0,
            "max_uncovered_gap_us": round(max_gap / 1000.0, 2),
            "gap_threshold_us": round(threshold / 1000.0, 2),
            "uncovered_gaps_us_start_end_dur": gaps_us[:16]}


def gate_g19(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    miss = missing_runtime(gc, "alignment")
    if miss["runtime_evidence_missing"]:
        return False, miss
    doc = gc.doc("alignment") or {}
    scale_ok = cp.as_int(doc.get("scale", 0), 0) == 1
    # 对齐质量硬门禁（阈值见配置区 7b）：method/scale 合法但 SW/HW 窗口本身错位
    # 也能过 method-only 检查，必须按 span 相对误差和端点残差卡死。fail-closed：
    # 字段缺失、非数值或 NaN/inf 都视同超限（查不到不等于合格）。
    span_err = doc.get("relative_span_error")
    max_resid = doc.get("max_abs_residual_ns")
    span_err_limit = float(GATE_PROJECT["alignment_max_relative_span_error"])
    resid_limit = int(GATE_PROJECT["alignment_max_abs_residual_ns"])
    span_err_ok = (isinstance(span_err, (int, float)) and not isinstance(span_err, bool)
                   and span_err <= span_err_limit)
    resid_ok = (isinstance(max_resid, (int, float)) and not isinstance(max_resid, bool)
                and max_resid <= resid_limit)
    quality_ok = span_err_ok and resid_ok
    coverage = _operator_task_coverage(gc, doc)
    coverage_ok = (not coverage.get("applicable", False)) or bool(coverage.get("ok", False))
    ok = scale_ok and quality_ok and coverage_ok
    return ok, {"scale": doc.get("scale"), "shift_ns": doc.get("shift_ns"),
                "scale_ok": scale_ok,
                "relative_span_error": span_err,
                "relative_span_error_limit": span_err_limit,
                "span_error_ok": span_err_ok,
                "max_abs_residual_ns": max_resid,
                "max_abs_residual_ns_limit": resid_limit,
                "residual_ok": resid_ok,
                "alignment_quality_ok": quality_ok,
                "coverage_ok": coverage_ok,
                "operator_task_coverage": coverage}


def gate_g20(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    miss = missing_runtime(gc, "hw_manifest")
    if miss["runtime_evidence_missing"]:
        return False, miss
    doc = gc.doc("hw_manifest") or {}
    verdict = str(doc.get("verdict", "")).upper()
    ok = verdict in GATE_PROJECT["hw_manifest_pass_verdicts"]
    return ok, {"verdict": doc.get("verdict"),
                "accepted_verdicts": list(GATE_PROJECT["hw_manifest_pass_verdicts"])}


def gate_g21(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    miss = missing_runtime(gc, "hw_metrics", "hw_manifest")
    if miss["runtime_evidence_missing"]:
        return False, miss
    manifest = gc.doc("hw_manifest") or {}
    freq = cp.first(manifest, ("frequency_hz", "sample_rate_hz",
                               "capture.frequency_hz", "capture.sample_rate_hz"))
    expected = cp.as_int(cp.ckey(gc.contract, "hw_sample_rate_hz",
                                 GATE_PROJECT["hw_sample_rate_hz"]),
                         GATE_PROJECT["hw_sample_rate_hz"])
    ok = cp.as_int(freq, 0) == expected
    return ok, {"frequency_hz": freq, "expected_frequency_hz": expected}


def gate_g22(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    cap_ok, cap = cp.check_e22(gc)
    ok, findings = _candidate_gate(gc)
    if not ok:
        out = {"browse_capability_ok": cap_ok}
        out.update(cap)
        out.update(findings)
        return False, out
    size = gc.artifacts["candidate"].stat().st_size
    max_bytes = cp.as_int(cp.ckey(gc.contract, "max_trace_bytes",
                                  GATE_PROJECT["max_trace_bytes"]),
                          GATE_PROJECT["max_trace_bytes"])
    # max_bytes 必须是正数：拿不到上限就等于没有上限，那 G22 就白判了。
    ok2 = cap_ok and max_bytes > 0 and size <= max_bytes
    return ok2, {"browse_capability_ok": cap_ok, "candidate_bytes": size,
                 "max_bytes": max_bytes}


def gate_g23(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    return _candidate_gate(gc)


def gate_g24(gc: GateCtx) -> Tuple[bool, Dict[str, Any]]:
    tps = gc.component_path("trace_processor_shell")
    miss = missing_runtime(gc, "candidate")
    findings: Dict[str, Any] = {"trace_processor_shell": tps}
    findings.update(miss)
    if miss["runtime_evidence_missing"]:
        return False, findings
    if not cp.is_file(tps):
        findings["reason"] = "trace_processor_shell not present for real import"
        return False, findings
    return True, findings


GATES = {
    "G00": gate_g00, "G01": gate_g01, "G02": gate_g02, "G03": gate_g03,
    "G04": gate_g04, "G05": gate_g05, "G06": gate_g06, "G07": gate_g07,
    "G08": gate_g08, "G09": gate_g09, "G10": gate_g10, "G11": gate_g11,
    "G12": gate_g12, "G13": gate_g13, "G14": gate_g14, "G15": gate_g15,
    "G16": gate_g16, "G17": gate_g17, "G18": gate_g18, "G19": gate_g19,
    "G20": gate_g20, "G21": gate_g21, "G22": gate_g22, "G23": gate_g23,
    "G24": gate_g24,
}

# 能力门在纯 CPU 编写阶段就能定论；运行时门需要 GPU 编排器产出的测量产物。
CAPABILITY_GATES = {"G12", "G14"}


def write_evidence(out_path: Path, payload: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
                        encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Project gate verdict adapter (one G-gate per invocation).")
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--gate-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    gate_id = str(args.gate_id).strip()
    out_path = Path(args.out).resolve()
    contract_path = Path(args.contract).resolve()

    base = {"schema": SCHEMA, "gate_id": gate_id,
            "gate_class": "capability" if gate_id in CAPABILITY_GATES else "runtime",
            "contract_path": str(contract_path), "generated_at_utc": cp.utc_now()}

    if gate_id not in GATES:
        base.update({"status": "ERROR", "ok": False,
                     "errors": ["unknown gate id {!r}; expected one of {}".format(
                         gate_id, ",".join(ALL_GATE_IDS))]})
        write_evidence(out_path, base)
        print("GATE={} STATUS=ERROR (unknown id)".format(gate_id))
        return 2

    try:
        contract = cp.load_json(contract_path)
    except (OSError, ValueError) as error:
        base.update({"status": "FAIL", "ok": False,
                     "errors": ["cannot read contract: {}".format(error)]})
        write_evidence(out_path, base)
        print("GATE={} STATUS=FAIL (contract unreadable)".format(gate_id))
        return 1

    try:
        gc = GateCtx(contract, contract_path)
        ok, findings = GATES[gate_id](gc)
        errors: List[str] = []
    except Exception as error:  # noqa: BLE001 -- 如实暴露任何门内异常，不吞掉
        ok, findings = False, {}
        errors = ["{}: {}".format(type(error).__name__, error),
                  traceback.format_exc().strip()]

    base.update({"status": "PASS" if ok else "FAIL", "ok": bool(ok),
                 "findings": findings, "errors": errors})
    write_evidence(out_path, base)
    print("GATE={} STATUS={} out={}".format(gate_id, base["status"], out_path))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
