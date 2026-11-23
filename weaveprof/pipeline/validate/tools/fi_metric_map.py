# -*- coding: utf-8 -*-
"""跨架构硬件指标语义映射。

设计边界：
- NVIDIA 名称取自 mail-12 的 nsys 2026.1.3 GpuMetrics 配置（sha256 已登记）；gh100 另在 mail-12 的
  H20 真实 trace 中观察到同名曲线，gb10x/gb11x 只验证了配置文本存在，未在对应 GPU 上运行。
  GpuMetrics 配置目录一律现场从 nsys 的安装位置解析（见 config_dir），本机 nsys 版本不同时
  config_sha256_match 如实写 False 并提示先 --gpu-metrics-set 复核名字，不算失败。
- AMD rocprofv3 只提供接口候选，不在本机 H20 上伪装成已验证。
- 这里只做“名称 -> 语义”的稳定映射；具体阈值由 fi_attrib 使用，并写入 attrib.json。
"""
from __future__ import annotations

import copy
import glob
import hashlib
import os
import re
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple

TOOL_VERSION = "fi-metric-map"

# 所有百分比都是 nsys 的 throughput 百分比（该硬件计数器相对峰值的比例）。
THRESHOLDS = {
    "high_pct": 70.0,
    "saturated_pct": 85.0,
    "idle_pct": 35.0,
    "min_samples": 3,
    "max_period_us": 10.0,
    "max_metric_gap_fraction": 0.50,
}

SEMANTICS: Dict[str, Dict[str, Any]] = {
    "tensor_active": {
        "label_zh": "Tensor 管线活跃比例",
        "canonical_nvidia": "sm__pipe_tensor_cycles_active_realtime",
        "unit_nvidia": "percent_of_peak",
        "use_zh": "判断 Tensor/矩阵管线是否接近饱和；不能单独证明某条 warp 在算。",
    },
    "sm_active": {
        "label_zh": "SM 活跃比例",
        "canonical_nvidia": "sm__cycles_active",
        "unit_nvidia": "percent_of_peak",
        "use_zh": "判断整卡是否有 SM 完全空闲；它是所有 SM 的聚合/平均，不能定位单一 SM。",
    },
    "issue": {
        "label_zh": "SM 指令发射率",
        "canonical_nvidia": "sm__inst_executed_realtime",
        "unit_nvidia": "percent_of_peak",
        "use_zh": "判断指令发射是否是瓶颈；高分也可能来自无效/冗余指令。",
    },
    "dram_read": {
        "label_zh": "DRAM 读活跃比例",
        "canonical_nvidia": "dram__read_throughput",
        "unit_nvidia": "percent_of_peak",
        "use_zh": "判断读带宽是否是瓶颈；这是整卡平均，不能指出哪个 CTA。",
    },
    "dram_write": {
        "label_zh": "DRAM 写活跃比例",
        "canonical_nvidia": "dram__write_throughput",
        "unit_nvidia": "percent_of_peak",
        "use_zh": "判断写带宽是否是瓶颈；这是整卡平均，不能指出哪个 CTA。",
    },
    "warps_in_flight": {
        "label_zh": "Compute warp 在飞数量",
        "canonical_nvidia": "tpc__warps_active_shader_cs_realtime",
        "unit_nvidia": "warp_or_percent",
        "use_zh": "判断是否存在占空/延迟问题；nsys 同时给出 Throughput%、Avg、Avg Warps/Cycle 三种口径。",
    },
}

_NVIDIA_CONFIG_SHA = {
    "ga10x": "",
    "gh100": "535a738b8ed86eb15edf7a7ebf25d809b6b3f520245101898b6d00fafcde41c9",
    "gb10x": "23fe3e0f03189da4e6a01eea036082d1abed45256aeeb2411cd8e323cb22e56e",
    "gb11x": "821c8fe10fd9cd50bb1a4a15ece084c567fb6570ad0dc0ebfcd4c4fff1f9b434",
}
# GpuMetrics 配置目录一律现场解析，不许写死也不许编造兜底路径：nsys 可能在
# /opt/nvidia/nsight-systems/<版本>/，也可能是 CUDA 自带的 /usr/local/cuda-<版本>/NsightSystems-cli-<版本>/，
# 版本号各机器都不同。盲测在 B300 上就是因为兜底路径写死了别的机器的 2026.1.3，
# 报告里出现一个本机根本不存在的文件，--selftest 直接 AssertionError。
# 找不到时如实返回空目录 + 说明，config_path 记 None，sha 校验按“不适用”处理。
_CONFIG_DIR_CACHE: Optional[Tuple[str, str]] = None
_CONFIG_DIR_PATTERNS = (
    "/opt/nvidia/nsight-systems/*/target-linux-x64/GpuMetrics",
    "/opt/nvidia/nsight-systems-*/target-linux-x64/GpuMetrics",
    "/opt/nvidia/nsys*/target-linux-x64/GpuMetrics",
    "/usr/local/cuda*/NsightSystems-cli-*/target-linux-x64/GpuMetrics",
    "/usr/local/cuda*/nsight-systems-*/target-linux-x64/GpuMetrics",
    "/usr/lib/nsight-systems/*/target-linux-x64/GpuMetrics",
    "C:/Program Files/NVIDIA Corporation/Nsight Systems*/target-windows-x64/GpuMetrics",
)


def _is_cfg_dir(d: str) -> bool:
    """目录存在且里面有 *.config 就算命中（不要求特定 arch：B300 上可能只有 gb1*.config）。"""
    if not d or not os.path.isdir(d):
        return False
    try:
        return any(f.endswith(".config") for f in os.listdir(d))
    except OSError:
        return False


def _cfg_dirs_from_nsys() -> List[Tuple[str, str]]:
    """从 nsys 可执行文件反推 GpuMetrics 目录（nsys 常是 symlink，先 realpath）。"""
    out: List[Tuple[str, str]] = []
    for src, label in ((os.environ.get("FI_NSYS_BIN", ""), "环境变量 FI_NSYS_BIN"),
                       (shutil.which("nsys") or "", "PATH 里的 nsys")):
        if not src:
            continue
        real = os.path.realpath(src)
        d = os.path.dirname(real)
        up = os.path.dirname(d)
        for cand, how in ((os.path.join(d, "GpuMetrics"), f"{label} -> {real} 同目录"),
                          (os.path.join(up, "target-linux-x64", "GpuMetrics"), f"{label} -> {real} 的安装根"),
                          (os.path.join(up, "GpuMetrics"), f"{label} -> {real} 的上一级"),
                          (os.path.join(real, "GpuMetrics"), f"{label} -> {real}（nsys 是目录）")):
            out.append((cand, how))
    return out


def config_dir() -> Tuple[str, str]:
    """定位本机 nsys 的 GpuMetrics 配置目录，返回 (目录, 依据)；找不到返回 ("", 说明)。
    顺序：FI_NSYS_GPUMETRICS_DIR -> nsys 可执行文件的安装目录 -> 常见安装位置（取版本最新）。"""
    global _CONFIG_DIR_CACHE
    if _CONFIG_DIR_CACHE is not None:
        return _CONFIG_DIR_CACHE

    env = os.environ.get("FI_NSYS_GPUMETRICS_DIR", "")
    if _is_cfg_dir(env):
        _CONFIG_DIR_CACHE = (env, "环境变量 FI_NSYS_GPUMETRICS_DIR")
        return _CONFIG_DIR_CACHE
    tried: List[str] = []
    for cand, how in _cfg_dirs_from_nsys():
        if _is_cfg_dir(cand):
            _CONFIG_DIR_CACHE = (cand, how)
            return _CONFIG_DIR_CACHE
        tried.append(cand)
    hits = sorted({h for pat in _CONFIG_DIR_PATTERNS for h in glob.glob(pat) if _is_cfg_dir(h)})
    if hits:
        _CONFIG_DIR_CACHE = (hits[-1], f"常见安装位置匹配到 {len(hits)} 处，取版本最新 {hits[-1]}")
        return _CONFIG_DIR_CACHE
    tried += [p for pat in _CONFIG_DIR_PATTERNS for p in (pat,)]
    _CONFIG_DIR_CACHE = ("", "本机没找到 nsys 的 GpuMetrics 配置目录（试过 FI_NSYS_GPUMETRICS_DIR、"
                             "nsys 安装目录、" + "、".join(_CONFIG_DIR_PATTERNS[:2]) + " 等；"
                             f"共试 {len(tried)} 处）：config_path 记 None，sha 校验按不适用处理")
    return _CONFIG_DIR_CACHE

_NVIDIA_EXACT = {
    "ga10x": {
        "tensor_active": ["Tensor Active [Throughput %]"],
        "sm_active": ["SMs Active [Throughput %]"],
        "issue": ["SM Issue [Throughput %]"],
        "dram_read": ["DRAM Read Bandwidth [Throughput %]"],
        "dram_write": ["DRAM Write Bandwidth [Throughput %]"],
        "warps_in_flight": [
            "Compute Warps in Flight [Throughput %]",
            "Compute Warps in Flight [Avg]",
            "Compute Warps in Flight [Avg Warps per Cycle]",
        ],
    },
    "gh100": {
        "tensor_active": ["Tensor Active [Throughput %]"],
        "sm_active": ["SMs Active [Throughput %]"],
        "issue": ["SM Issue [Throughput %]"],
        "dram_read": ["DRAM Read Bandwidth [Throughput %]"],
        "dram_write": ["DRAM Write Bandwidth [Throughput %]"],
        "warps_in_flight": [
            "Compute Warps in Flight [Throughput %]",
            "Compute Warps in Flight [Avg]",
            "Compute Warps in Flight [Avg Warps per Cycle]",
        ],
    },
    "gb10x": {
        "tensor_active": ["Tensor Active [Throughput %]"],
        "sm_active": ["SMs Active [Throughput %]"],
        "issue": ["SM Issue [Throughput %]"],
        "dram_read": ["DRAM Read Bandwidth [Throughput %]"],
        "dram_write": ["DRAM Write Bandwidth [Throughput %]"],
        "warps_in_flight": [
            "Compute Warps in Flight [Throughput %]",
            "Compute Warps in Flight [Avg]",
            "Compute Warps in Flight [Avg Warps per Cycle]",
        ],
    },
    "gb11x": {
        "tensor_active": ["Tensor Active [Throughput %]"],
        "sm_active": ["SMs Active [Throughput %]"],
        "issue": ["SM Issue [Throughput %]"],
        "dram_read": ["DRAM Read Bandwidth [Throughput %]"],
        "dram_write": ["DRAM Write Bandwidth [Throughput %]"],
        "warps_in_flight": [
            "Compute Warps in Flight [Throughput %]",
            "Compute Warps in Flight [Avg]",
            "Compute Warps in Flight [Avg Warps per Cycle]",
        ],
    },
}

# 只用于兼容旧版/不同 nsys 版本的语义兜底；匹配到也会明确标注 match_method=regex_fallback。
_REGEX = {
    "tensor_active": [r"^tensor\s+active(?:\s|\[|$)", r"sm__pipe_tensor_cycles_active"],
    "sm_active": [r"^(?:sms?|sm)\s+active(?:\s|\[|$)", r"sm__cycles_active"],
    "issue": [r"^(?:sm\s+)?(?:inst(?:ruction)?\s+)?issue(?:\s|\[|$)", r"^sm\s+issue(?:\s|\[|$)", r"sm__inst_executed"],
    "dram_read": [r"^(?:dram|hbm)\s+read(?:\s|\[|$)", r"dram__read_throughput"],
    "dram_write": [r"^(?:dram|hbm)\s+write(?:\s|\[|$)", r"dram__write_throughput"],
    "warps_in_flight": [r"^compute\s+warps?\s+in\s+flight(?:\s|\[|$)", r"^warps?\s+in\s+flight(?:\s|\[|$)", r"warps_active_shader_cs"],
}

_NVIDIA_ARCH_INFO = {
    "ga10x": {
        "arch": "ga10x",
        "display": "NVIDIA GA10x / Ampere sm_86 (RTX 30xx / A10 / A40 / A6000)",
        "chips": ["GA102", "GA104", "GA106", "GA107"],
        "metric_set": "ga10x",
        "compute_major": 8,
        "config_verified_on_machine": True,
        "runtime_verified_on_machine": True,
        "verified_host": "local-rtx3050-sm86",
        "evidence": "本地 RTX 3050 (sm_86) Windows nsys 200 kHz trace 实测同名 GPU metrics（Tensor Active / SMs Active / SM Issue / DRAM Read+Write）。",
    },
    "gh100": {
        "arch": "gh100",
        "display": "NVIDIA GH100 / H20",
        "chips": ["GH100"],
        "metric_set": "gh100",
        "compute_major": 9,
        "config_verified_on_machine": True,
        "runtime_verified_on_machine": True,
        "verified_host": "mail-12",
        "evidence": "本机 h20-attrgt/h20-mkcase nsys trace 中观察到同名 GPU metrics；配置 sha256 已登记。",
    },
    "gb10x": {
        "arch": "gb10x",
        "display": "NVIDIA GB10x / B200-class",
        "chips": ["GB100", "GB102"],
        "metric_set": "gb10x",
        "compute_major": 10,
        "config_verified_on_machine": True,
        "runtime_verified_on_machine": False,
        "verified_host": None,
        "evidence": "指标名取自 mail-12 的 nsys 2026.1.3 GpuMetrics/gb10x.config（sha256 已登记）；"
                    "未在 GB10x GPU 上运行，本机 nsys 版本不同就要用 --gpu-metrics-set 复核实际名字。",
    },
    "gb11x": {
        "arch": "gb11x",
        "display": "NVIDIA GB11x / B300-class",
        "chips": ["GB110", "GB112"],
        "metric_set": "gb11x",
        "compute_major": 10,
        "config_verified_on_machine": True,
        "runtime_verified_on_machine": False,
        "verified_host": None,
        "evidence": "指标名取自 mail-12 的 nsys 2026.1.3 GpuMetrics/gb11x.config（sha256 已登记）；"
                    "未在 GB11x/B300 GPU 上运行，本机 nsys 版本不同就要用 --gpu-metrics-set 复核实际名字。",
    },
}

# AMD 接口候选。不同 rocprofv3 / gfx942 计数器集合可能不同，因此 runtime_verified=False。
_AMD_INTERFACE = {
    "arch": "gfx942",
    "vendor": "amd",
    "backend": "rocprofv3",
    "runtime_verified_on_machine": False,
    "status": "interface_only_not_run_on_amd",
    "semantics": {
        "tensor_active": {
            "candidate_counters": ["SQ_INSTS_MFMA"],
            "derived_ratio": "SQ_INSTS_MFMA / max(GRBM_COUNT, 1)",
            "note": "MFMA 指令计数，不是 NVIDIA 的 tensor-pipe 活跃百分比；需要 kernel duration 归一化。",
        },
        "sm_active": {
            "candidate_counters": ["GRBM_GUI_ACTIVE", "GRBM_COUNT"],
            "derived_ratio": "GRBM_GUI_ACTIVE / max(GRBM_COUNT, 1)",
            "note": "辅助判断 CU 活跃；rocprofv3 的实际 PMC 列名需在 AMD 机器上确认。",
        },
        "issue": {
            "candidate_counters": ["SQ_INSTS_VALU", "SQ_INSTS_VMEM"],
            "derived_ratio": "instruction counts / kernel cycles (由调用方提供公式)",
            "note": "只有指令计数，不是发射百分比。",
        },
        "dram_read": {
            "candidate_counters": ["TCC_EA_RDREQ_sum"],
            "derived_ratio": "TCC_EA_RDREQ_sum / elapsed_cycles",
            "note": "EA read request 计数，非 HBM 实际带宽；需与 rocprofv3 字段名核对。",
        },
        "dram_write": {
            "candidate_counters": ["TCC_EA_WRREQ_sum"],
            "derived_ratio": "TCC_EA_WRREQ_sum / elapsed_cycles",
            "note": "EA write request 计数，非 HBM 实际带宽；需与 rocprofv3 字段名核对。",
        },
        "warps_in_flight": {
            "candidate_counters": ["SQ_WAVES"],
            "derived_ratio": "SQ_WAVES / active_CU_count",
            "note": "wave 计数接口；未在 AMD 上实测。",
        },
    },
    "warning": "AMD 名称是 rocprofv3 接口候选，不可作为已验证映射或实测结论。",
}


def _config_sha(arch: str) -> Optional[str]:
    cdir = config_dir()[0]
    if not cdir:
        return None            # 空目录时 join 会得到相对路径，可能误读当前目录下的同名文件
    p = os.path.join(cdir, f"{arch}.config")
    try:
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def normalize_arch(arch: Optional[str]) -> Optional[str]:
    """把 GPU 名、chip 名、sm_xx、metric set 名归一为 ga10x/gh100/gb10x/gb11x。"""
    if not arch:
        return None
    s = str(arch).strip().lower().replace(" ", "")
    if any(x in s for x in ("ga10", "ga102", "ga104", "ga106", "ga107", "rtx30", "rtxa2000", "rtxa4000", "rtxa5000", "rtxa6000", "a10", "a40", "sm86", "sm_86", "compute86", "8.6")):
        return "ga10x"
    if any(x in s for x in ("gh100", "h20", "h100", "sm90", "sm_90", "compute90", "9.0")):
        return "gh100"
    if any(x in s for x in ("gb11", "gb110", "gb112", "b300", "gb300", "sm103", "sm_103", "10.3")):
        return "gb11x"
    if any(x in s for x in ("gb10", "gb100", "gb102", "b200", "gb200", "sm100", "sm_100", "10.0")):
        return "gb10x"
    return None


def detect_arch(gpu: Optional[Dict[str, Any]]) -> Optional[str]:
    if not gpu:
        return None
    # chipName/name 优先，避免把 major=9 的所有设备误判成 H100 以外型号。
    for k in ("chipName", "name", "arch", "metric_set"):
        a = normalize_arch(gpu.get(k))
        if a:
            return a
    major, minor = gpu.get("computeMajor"), gpu.get("computeMinor")
    try:
        major, minor = int(major), int(minor or 0)
    except (TypeError, ValueError):
        major = minor = None
    if major == 8 and minor == 6:
        return "ga10x"
    if major == 9:
        return "gh100"
    if major == 10:
        return "gb11x" if minor and minor >= 3 else "gb10x"
    return None


def nvidia_arch_document(arch: str) -> Dict[str, Any]:
    a = normalize_arch(arch)
    if a not in _NVIDIA_ARCH_INFO:
        raise KeyError(f"不支持的 NVIDIA arch {arch!r}，可选 {sorted(_NVIDIA_ARCH_INFO)}")
    info = copy.deepcopy(_NVIDIA_ARCH_INFO[a])
    cdir, csrc = config_dir()
    # 找不到目录就写 None，不拼一个本机不存在的路径（盲测里那条写死的 2026.1.3 就是这么来的）
    info["config_path"] = os.path.join(cdir, f"{a}.config") if cdir else None
    info["config_dir"] = cdir or None
    info["config_dir_source"] = csrc
    info["declared_config_sha256"] = _NVIDIA_CONFIG_SHA[a]
    observed_sha = _config_sha(a)
    info["observed_config_sha256"] = observed_sha
    # 本机根本没有这份配置时，match 写 None（未知）而不是 False：False 会被读成
    # “本机配置被改过/不可信”（fi_bugs_b3-sk-qwen B2）。AMD 机器、或 nsys 装在别处时都会走到这里。
    info["config_available"] = bool(info["config_path"]) and os.path.isfile(info["config_path"])
    info["config_sha256_match"] = (observed_sha == _NVIDIA_CONFIG_SHA[a]) if info["config_available"] else None
    info["config_audit_note"] = (
        "本机配置与包内登记值一致" if info["config_sha256_match"] else
        ("本机配置存在但 sha256 与登记值不同：多半是 nsys 版本与登记时不同，指标名表可能已变，"
         "用前先 nsys --gpu-metrics-set 列出实际名字（这不是失败，但报告里要照实写）"
         if info["config_available"] else
         f"本机没有 {a}.config（{csrc}）：sha 校验按不适用处理，"
         "声明值仍然写出供交叉核对；这不代表本机配置被改过"))
    info["thresholds"] = copy.deepcopy(THRESHOLDS)
    info["semantics"] = {}
    for sem, spec in SEMANTICS.items():
        d = copy.deepcopy(spec)
        d["names"] = list(_NVIDIA_EXACT[a][sem])
        d["match_method"] = "exact"
        info["semantics"][sem] = d
    return info


def mapping_document(arch: Optional[str] = None) -> Dict[str, Any]:
    """输出给 fi_align_nsys / attrib.json 的可审计映射文档。"""
    requested = normalize_arch(arch) if arch else None
    a = requested or "gh100"
    doc = {
        "schema_version": TOOL_VERSION,
        "requested_arch": arch,
        "selected_arch": a,
        "nvidia": {k: nvidia_arch_document(k) for k in ("gh100", "gb10x", "gb11x")},
        "amd": copy.deepcopy(_AMD_INTERFACE),
        "canonical_semantics": copy.deepcopy(SEMANTICS),
        "thresholds": copy.deepcopy(THRESHOLDS),
        "notes": [
            "NVIDIA 指标名取自 mail-12 的 nsys 2026.1.3 GpuMetrics 配置；gh100 另在 mail-12 的 H20 真实 trace 里观察到同名曲线。",
            "本机 nsys 版本不同时 config_sha256_match=False 属正常：先 nsys --gpu-metrics-set 列出实际名字，再用 match_metric_names 复核。",
            "gb10x/gb11x 仅核对了配置文件文本，没有在对应 GPU 上采集；运行态验证必须另做。",
            "AMD 只有 rocprofv3 接口候选，未在 AMD 机器执行，不能写成已验证。",
        ],
    }
    return doc


def match_metric_names(names: Iterable[str], arch: Optional[str]) -> Dict[str, Any]:
    """把任意 nsys/trace 的指标名匹配到语义类别。

    返回：
      {arch, exact, fallback, missing, unclassified, config_audit}
    每个 match 含 semantic, observed_name, canonical, match_method, unit。
    """
    a = normalize_arch(arch) if arch else None
    if a not in _NVIDIA_ARCH_INFO:
        return {"arch": a, "exact": {}, "fallback": {}, "missing": sorted(SEMANTICS), "unclassified": sorted(map(str, names)),
                "error": "无法从指标名/GPU 信息识别 NVIDIA arch；请显式传 --arch 或在 trace 里提供 chipName"}
    names = [str(x) for x in names]
    exact, fallback = {}, {}
    used = set()
    for sem, candidates in _NVIDIA_EXACT[a].items():
        # 只把候选里实际存在于 trace 的名字填入；优先 Throughput% 版本。
        hit = next((n for n in candidates if n in names), None)
        if hit is not None:
            exact[sem] = {
                "semantic": sem,
                "observed_name": hit,
                "canonical": SEMANTICS[sem]["canonical_nvidia"],
                "match_method": "exact",
                "unit": SEMANTICS[sem]["unit_nvidia"],
                "label_zh": SEMANTICS[sem]["label_zh"],
            }
            used.add(hit)
    for sem, patterns in _REGEX.items():
        if sem in exact:
            continue
        hit = next((n for n in names if any(re.search(p, n, re.I) for p in patterns)), None)
        if hit is not None:
            fallback[sem] = {
                "semantic": sem,
                "observed_name": hit,
                "canonical": SEMANTICS[sem]["canonical_nvidia"],
                "match_method": "regex_fallback",
                "unit": SEMANTICS[sem]["unit_nvidia"],
                "label_zh": SEMANTICS[sem]["label_zh"],
            }
            used.add(hit)
    return {
        "arch": a,
        "exact": exact,
        "fallback": fallback,
        "missing": [s for s in SEMANTICS if s not in exact and s not in fallback],
        "unclassified": sorted(n for n in names if n not in used),
        "config_audit": {k: nvidia_arch_document(a)[k] for k in
                         ("config_path", "config_dir_source", "declared_config_sha256",
                          "observed_config_sha256", "config_sha256_match",
                          "config_available", "config_audit_note")},
        "thresholds": copy.deepcopy(THRESHOLDS),
    }


def amd_interface_document() -> Dict[str, Any]:
    return copy.deepcopy(_AMD_INTERFACE)


def map_amd_counter_names(counter_names: Iterable[str]) -> Dict[str, Any]:
    wanted = [str(x) for x in counter_names]
    out = {}
    for sem, spec in _AMD_INTERFACE["semantics"].items():
        hits = [n for n in wanted if n in spec["candidate_counters"]]
        if hits:
            out[sem] = {"semantic": sem, "observed_name": hits[0], "candidates": list(spec["candidate_counters"]),
                        "derived_ratio": spec["derived_ratio"], "runtime_verified": False,
                        "warning": "未在 AMD 实机验证"}
    return {"arch": _AMD_INTERFACE["arch"], "backend": "rocprofv3", "matches": out,
            "unmatched": [n for n in wanted if not any(n in s["candidate_counters"] for s in _AMD_INTERFACE["semantics"].values())],
            "runtime_verified": False, "warning": _AMD_INTERFACE["warning"]}


def selftest() -> None:
    assert normalize_arch("NVIDIA H20") == "gh100"
    assert normalize_arch("sm_103") == "gb11x"
    assert normalize_arch("sm_100") == "gb10x"
    assert detect_arch({"name": "NVIDIA B300", "computeMajor": 10, "computeMinor": 3}) == "gb11x"
    names = ["Tensor Active [Throughput %]", "SMs Active [Throughput %]", "SM Issue [Throughput %]",
             "DRAM Read Bandwidth [Throughput %]", "DRAM Write Bandwidth [Throughput %]",
             "Compute Warps in Flight [Throughput %]", "Unrelated [Throughput %]"]
    cdir, csrc = config_dir()
    # 找到的目录必须真实存在：不许再出现“报告里写着一个本机没有的路径”（b3-sk-ds/b3-sk-glm 的 AssertionError）
    assert (not cdir) or os.path.isdir(cdir), (cdir, csrc)
    for a in ("gh100", "gb10x", "gb11x"):
        m = match_metric_names(names, a)
        assert not m["missing"], (a, m)
        assert set(m["exact"]) == set(SEMANTICS)
        assert m["unclassified"] == ["Unrelated [Throughput %]"]
        ca = m["config_audit"]
        if ca["config_available"]:
            # 本机有这份配置：路径必须是真文件；sha 一致最好，不一致（nsys 版本不同）要在 note 里讲清楚，
            # 不当失败——否则换一台 nsys 版本不同的机器 selftest 就红（盲测就是这么红的）。
            assert ca["config_path"] and os.path.isfile(ca["config_path"]), (a, ca)
            assert ca["config_sha256_match"] in (True, False), (a, ca)
            if ca["config_sha256_match"] is False:
                assert "nsys 版本" in ca["config_audit_note"] and ca["observed_config_sha256"], (a, ca)
        else:
            # 本机没装 nsys、nsys 装在别处、或在 AMD 机器上跑：sha 校验按“不适用”记 None，不当失败，
            # 也不能写成 False（会被读成“本机配置被改过”，fi_bugs_b3-sk-qwen B2）。
            assert ca["config_path"] is None or not os.path.exists(ca["config_path"]), (a, ca)
            assert ca["observed_config_sha256"] is None and ca["config_sha256_match"] is None, (a, ca)
            assert ca["config_available"] is False and "不适用" in ca["config_audit_note"], (a, ca)
    # 回归：模拟“本机没有 GpuMetrics 目录”，必须如实报 None，不许编路径、不许把 match 写成 False
    global _CONFIG_DIR_CACHE
    saved = _CONFIG_DIR_CACHE
    _CONFIG_DIR_CACHE = ("", "自测：模拟本机没有 GpuMetrics 目录")
    try:
        ca = match_metric_names(names, "gh100")["config_audit"]
        assert ca["config_path"] is None and ca["config_available"] is False, ca
        assert ca["observed_config_sha256"] is None and ca["config_sha256_match"] is None, ca
        assert "不适用" in ca["config_audit_note"] and "没有" in ca["config_audit_note"], ca
        assert ca["declared_config_sha256"] == _NVIDIA_CONFIG_SHA["gh100"], ca
        assert _config_sha("gh100") is None
    finally:
        _CONFIG_DIR_CACHE = saved
    old = match_metric_names(["sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed"], "gh100")
    assert "tensor_active" in old["fallback"], old
    amd = map_amd_counter_names(["SQ_INSTS_MFMA", "TCC_EA_RDREQ_sum", "NO_SUCH"])
    assert "tensor_active" in amd["matches"] and "dram_read" in amd["matches"] and amd["unmatched"] == ["NO_SUCH"]
    sha_state = {a: match_metric_names(names, a)["config_audit"]["config_sha256_match"]
                 for a in ("gh100", "gb10x", "gb11x")}
    print("METRIC_MAP SELFTEST PASS（GpuMetrics 配置：" +
          (f"{cdir}，来源 {csrc}，sha 核对结果 {sha_state}）" if cdir else
           "本机没有，config_path 记 None，sha 校验按不适用处理，声明值仍写进 config_audit）"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    n = ap.parse_args()
    if n.selftest:
        selftest()
    else:
        import json
        d = mapping_document(n.arch)
        print(json.dumps(d, ensure_ascii=False, indent=2))
