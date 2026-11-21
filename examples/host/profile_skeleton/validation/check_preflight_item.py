#!/usr/bin/env python3
"""check_preflight_item.py（骨架版）— 项目侧 preflight 判定适配器（E01-E27）。

流程位置
  skill 侧 ``weaveprof/pipeline/scan/validate_preflight.py`` 是固定入口。它按 frozen contract 的
  ``execution.preflight.check_command_template`` 逐条调起本文件（argv-only、关闭
  shell），并按这个顺序判定：
    1. 进程退出码（非 0 -> FAIL，最先看）；
    2. 声明的证据文件（缺失/不可读 -> BLOCKED）；
    3. 证据 sha256（不一致 -> FAIL）；
    4. 否则 PASS。

谁来调 / CLI
  python3 -B check_preflight_item.py --contract <lock> --check-id E07 --out <p.json>
  一次只判一条；未知 check id 退出码 2。

本文件的唯一职责
  (a) 无论成败都往 ``--out`` 写一份 ``fast-instrument-preflight-item/v1`` 证据；
  (b) 只有能力真的成立时才退出 0。
  它是裁判，不写 contract。**绝对不允许"查不到就当通过"** —— 这是 fail-closed 门禁
  链，一次静默 PASS 会让后面 G00-G24 全部失去意义。

接新 target 要改的地方
  只有下面那个 ``===== 接新 target 时只需要改这里 =====`` 配置区：
  路径常量、三个编译档位名、期望几何、以及每个 check 要在源文件里看到的 token 清单。
  **第一步必须改 ``PROJECT["device_writer_relpaths"]``** —— 骨架里那三条是示例路径、
  实际不存在，不改的话 E01/E05/E08/E09 会带着"device writer 源文件缺失"直接 FAIL
  （这是故意的：定位不到源文件时它们只能对着空字符串下结论）。
  凡是能从 ABI / contract 推出来的东西（事件表、task/phase 层级、时钟、buffer 地址
  公式、HW 采样率、trace 上限、SM 集合……）都**不要**写死在这里：ABI 一改，检查必须
  跟着改判，写死了就会出现"ABI 已经变了但 selftest 还过"的假绿灯。

参考实现（dsv4pro/decode，已通过 25 门禁 —— 这是**范本路径**，不是本项目路径；
本文件由它泛化而来，遇到不确定的语义请对照它，但不要照抄它的配置值）
  DSA-learn/mk/h20/cutedsl/deepseek_v4_pro/decode/profile/validation/check_preflight_item.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS = Path(__file__).resolve()
_VALIDATION = _THIS.parent                       # <project>/profile/validation
_PROFILE_ROOT = _VALIDATION.parent               # <project>/profile
_VERSION_ROOT = _PROFILE_ROOT.parent             # <project>
_HOST_DECODE = _PROFILE_ROOT / "host_decode"
_DEVICE_RECORD = _PROFILE_ROOT / "device_record"
_HW_PERFETTO = _PROFILE_ROOT / "hw_perfetto"

# ======================================================================
# ===== 接新 target 时只需要改这里 (PROJECT CONFIG) =====
# 下面每一项都在本文件里被真正用到；改完这里就能跑，不需要改函数体。
# ======================================================================
PROJECT: Dict[str, Any] = {
    # 1) ABI 文件名（位于 device_record/abi/ 下）。真实项目里叫 profiler_abi.json。
    #    真正的路径优先从 contract 读（见 CONTRACT_KEYS["abi_path"]），这里只是兜底。
    "abi_filename": "record_abi.example.json",

    # 2) 三个插桩编译档位。key 固定为 off / entry_only / full（overhead 公式依赖它），
    #    value 是"这个项目里这个档位叫什么"：
    #      - 骨架/JSON preset 方案：off / entry_only / full（device_record/presets/*.json）
    #      - 真实 DSV4-Pro 方案：PROF_OFF / PROF_ENTRY_ONLY / PROF_FULL（cutlass Constexpr）
    "compile_level_names": {"off": "off", "entry_only": "entry_only", "full": "full"},
    #    档位的实现在哪：JSON preset 相对路径（相对 profile/）。若你的项目用 Python 常量
    #    实现档位，把这里指到那个模块，并在 run_profile.py 里提供 device_constants()。
    "compile_level_files": {
        "off": "device_record/presets/off.json",
        "entry_only": "device_record/presets/entry_only.json",
        "full": "device_record/presets/full.json",
    },

    # 3) device 侧记录源（%smid 过滤 / rec_begin / rec_end / 溢出标志都在这里）。
    #    解析顺序：contract components.device_writer -> ABI source_file -> 下面这份清单。
    #
    #    ★★ 接新 target 的第一步就是改这里 ★★
    #    下面这三条是**骨架里并不存在的示例路径**（骨架的 device_record/ 下只有
    #    abi/ 和 presets/）。不改的后果：device_writer_paths() 恒为空 ->
    #    device_writer_text() 恒为 "" -> E01/E05/E08/E09 拿着一个空字符串做 token
    #    检查，**这四条检查的真实语义一次都没跑过**。现在这四条会显式 FAIL 并让你
    #    来改这一项（见 device_writer_missing()），而不是对着空字符串下结论。
    "device_writer_relpaths": [
        "device_record/device_profiler.cuh",
        "device_record/warp_record.py",
        "device_record/presets/megakernel_instr.py",
    ],

    # 4) host 侧组件相对路径（相对 profile/）。contract components.<key>.absolute_path
    #    存在时优先用 contract 的值，这里是兜底。
    "component_relpaths": {
        "host_harness": "run_profile.py",
        "contract_materializer": "host_decode/materialize_contract.py",
        "decoder": "host_decode/decode_profile.py",
        "hardware_reader": "hw_perfetto/capture_nsys.py",
        "hardware_metrics_reader": "hw_perfetto/extract_gpu_metrics.py",
        "sw_hw_aligner": "hw_perfetto/align_sw_hw.py",
        "perfetto_emitter": "hw_perfetto/emit_perfetto.py",
        "metrics_config_dir": "hw_perfetto/metrics",
        "correctness_validator": "validation/validate_correctness.py",
        "overhead_validator": "validation/validate_overhead.py",
        "offline_selftests": "validation/run_offline_selftests.py",
        "selftest_dir": "validation/selftest",
    },

    # 5) 期望的 launch 几何 / 拓扑。
    #    ★ None **不等于豁免** ★ E14/E15 要求每个几何量都有**两个独立来源**互相印证
    #    （声明侧：contract build_and_launch / topology、ABI；运行侧：runtime launch
    #    描述符 / threads_per_cta//32）。只有一个来源 = 自己证明自己 = 判 FAIL，
    #    证据里的 reason 会写 single_source 并点名该填哪个字段。
    #    历史坑：这四项默认 None 时判据写成 `expected is None or ...`，等于"没填期望值
    #    就不校验"；再叠上 E15 的 `warps == threads // 32`（warps 自己就是 threads//32
    #    兜底算出来的），几何校验整段退化成恒真。
    #    所以：能从 contract/ABI 推出真值的（优先，自动适配不同 target）就把那些字段补齐；
    #    推不出来的，把已知值填在这里当第二来源。真实项目 grid=[78,1,1] / warps=12。
    "expected_grid": None,                    # e.g. [78, 1, 1]
    "expected_threads_per_cta": None,         # e.g. 384
    "expected_physical_warps_per_cta": None,  # e.g. 12
    "expected_observer_lane_count": None,     # e.g. 12

    # 6) buffer 容量 / 溢出。None = 用 ABI+contract 声明值，不写死数字。
    "expected_max_records_per_lane": None,    # 真实项目 8192
    #    独立溢出张量名；单块扁平 buffer（骨架 ABI 那种 sentinel 方案）填 None。
    "overflow_tensor_name": None,             # 真实项目 "prof_ovf"

    # 7) 事件配对的 begin/end record kind。必须与你 frozen contract 的
    #    events.pairing 一致（真实项目 1/0；skill 的 contract_lock_example 是 1/2）。
    "pairing_kinds": {"begin": 1, "end": 0},

    # 8) 交付硬约束。HW 采样率与 trace 体积上限；两者都会和 contract 交叉核对。
    "hw_sample_rate_hz": 200000,
    "max_trace_bytes": 10 * 1024 * 1024,      # 10485760

    # 9) E25 离线自测"有实质内容"的下限。E25 只看**文件内容**，不看文件名。
    #    历史坑：E25 曾把 ``p.name`` 也拼进用于判重的 blob —— 于是 4 个**同名空文件**
    #    （selftest_decode.py / selftest_hw_merge.py / selftest_perfetto.py /
    #    selftest_coverage.py）就能把 decoder/hw_merge/perfetto/coverage 四个 aspect
    #    全部"命中"，E25 直接 PASS。文件名是自由字符串，不是证据。
    #    现在每个自测文件必须：体积 >= selftest_min_bytes、断言点 >= 
    #    selftest_min_assertion_sites、有失败路径、且与其它自测文件**内容不同**
    #    （去注释归一化后 sha256 互不相同，互为副本一律 FAIL）。
    #    骨架现有 4 个自测最小的是 8771 字节 / 11 个断言点，所以下面的默认值留了 4x 余量。
    #    **调低这两个数就是把上面那条假绿灯放回来**，不要为了变绿去动它们。
    "selftest_min_bytes": 2048,
    "selftest_min_assertion_sites": 5,

    # 10) 每个检查要求在对应源文件里出现的 token。
    #    这是最容易需要改的一项：不同项目命名不同（例如骨架 run_profile.py 用
    #    max_records，真实项目用 records_per_lane / overflow_per_lane）。
    #    token 缺失 -> 该 check FAIL 并把缺的 token 列进证据，方便定位。
    "required_tokens": {
        # E05：device 侧必须在"写记录之前"就按 %smid 过滤。
        #   三个 token 一起才能证明"源侧过滤"这件事真的发生了：
        #     smid()        —— 真的读了 %smid（不是注释里提一句 "smid"）
        #     prof_sel      —— 有一张"选中哪些 SM"的表被查
        #     capture_slot  —— 选中的 SM 被映射成 buffer 里的槽位
        #   历史坑：这里曾退化成 ("smid",) —— 任何出现 "smid" 字样（连注释、变量名
        #   smid_unused 都算）的文件都能过，等于没查。单 token 的宽松集合就是假绿灯。
        #   接新 target 可以改名字（比如你的表叫 sm_whitelist），但**必须保持三件事
        #   都被验到**，不要缩成一个词。
        "device_writer_smid": ("smid()", "prof_sel", "capture_slot"),
        # E08：device 侧必须有容量上限与溢出标志。
        #   参考实现（dsv4pro/decode）用的是 ("MAX_RECORDS_PER_LANE", "prof_ovf")；
        #   骨架的扁平 buffer 用 sentinel 而不是独立溢出张量（见 overflow_tensor_name=None），
        #   所以默认只要求容量上限。**如果你的项目有独立溢出张量，把它的名字加进来。**
        "device_writer_capacity": ("max_records",),
        # E09：device 侧必须真的记录入口 rendezvous（每个 CTA 的第一个事件）。
        #   rec_begin / rec_end 证明成对记录真的落在 device 侧，prof_level 证明它受
        #   编译档位控制。历史坑：这里曾退化成 ("profil",) —— 连 "profiler" 这个词
        #   出现在任何注释里都能过。
        "device_writer_anchor": ("rec_begin", "rec_end", "prof_level"),
        # E11：contract materializer 必须暴露 5 个用户选择 + --out + 设备探测
        "materializer": ("--information-bundle", "--granularity", "--display-rank",
                         "--software-sm-scope", "--hardware", "--out"),
        # E13：host capture 必须落 raw + sidecar，并声明几何与 capture scope
        "host_harness": ("sw-capture", "raw_rank", "sidecar", "capture_scope",
                         "max_records", "grid", "block"),
        # E16：解码器必须把异常显式计数（不能吞掉）
        "decoder": ("unknown", "unmatched", "truncated"),
        # E17：nsys 采集必须用 cudaProfilerApi 圈定范围并落 manifest
        "hardware_reader": ("cudaProfilerApi", "capture-range", ".nsys-rep", "sqlite", "manifest"),
        # E19：指标提取必须走 sqlite + 白名单
        "hardware_metrics_reader": ("sqlite", "whitelist"),
        # E21：emitter 必须产出 .perfetto.json.gz 且区分 task / phase
        "perfetto_emitter": (".perfetto.json.gz", "task", "phase"),
        # E22：超限重生成策略（隐藏所有 phase）+ 体积上限
        "perfetto_emitter_oversize": ("hide_all_phases",),
        # E23：正确性对比必须是 randn 输入 + 全元素 + 关插桩参考
        "correctness_validator": ("randn", "all_output_elements_all_ranks"),
    },

    # 11) ABI / contract 里同一语义的多种键名。骨架 ABI 与真实 ABI 段名不同，
    #     这里列候选，代码取第一个命中的；接新 target 时按你 extract_abi.py 的产物补。
    "abi_keys": {
        "clock_intrinsic": ("clock.read_intrinsic", "clock.source"),
        "record_fields": ("buffer.record_tensor.word_fields",
                          "regions.records.layout.fields"),
        "lane_formula": ("buffer.lane_formula", "buffer.slot_formula",
                         "regions.records.layout.slot_formula"),
        "element_index_formula": ("buffer.element_index_formula",
                                  "regions.records.layout.element_index_formula"),
        "max_records_per_lane": ("buffer.max_records_per_lane", "buffer.max_records"),
        "overflow_policy": ("buffer.overflow_policy",),
        "compile_levels": ("compile_switch.levels",),
    },
    #     tag 里"哪些字段唯一确定一条 lane"（E07 写隔离）。真实项目是
    #     capture_slot+warp；骨架扁平 buffer 是 block+track。
    "lane_identity_tag_fields": ("block", "track"),
    #     tag 里事件 id 字段名 / start-end 标志字段名。
    #     ``tag_event_id_field_candidates`` 曾经是**死配置**（声明了但没有任何 check 读它，
    #     看起来"事件 id 字段已经在查了"，其实一次都没查）。现在 E03 会用它要求 tag 里
    #     真的有一个事件 id 位域，且宽度足以编码 ABI tasks[] 里最大的 task id。
    "tag_event_id_field_candidates": ("task", "event_id"),
    "tag_is_start_field": "is_start",

    # 12) 入口 rendezvous 语义 token（ABI sync_anchors.entry_rendezvous）。
    "anchor_semantics": {"start": ("arrive",), "end": ("release",)},
}

# contract 里同一语义的候选路径（点分）。frozen contract（skill 权威形状）优先，
# 后面的是骨架 materialize_contract.py / 真实项目的别名。接新 target 一般不用改。
CONTRACT_KEYS: Dict[str, Tuple[str, ...]] = {
    "version_root": ("project_layout.version_root", "run.repository_root"),
    "run_root": ("project_layout.run_root", "artifacts.run_root"),
    "abi_path": ("profile_selection.abi.absolute_path", "profile_selection.abi",
                 "buffer_abi.abi_file"),
    "abi_sha256": ("profile_selection.abi.sha256", "buffer_abi.abi_sha256"),
    "target_kernel": ("run.target_kernel", "build_and_launch.kernel_name_contains",
                      "capture.hardware.kernel_name_contains"),
    "world_size": ("run.world_size",),
    "display_rank": ("capture.display_rank", "run.rank"),
    "information_bundle": ("capture.information_bundle",),
    "grid": ("build_and_launch.grid",),
    "block": ("build_and_launch.block", "build_and_launch.threads_per_cta"),
    "physical_warps_per_cta": ("build_and_launch.physical_warps_per_cta",
                               "topology.physical_warps_per_cta"),
    "physical_sm_count": ("topology.physical_sm_count",
                          "environment.gpu.physical_sm_count"),
    "observer_lane_count": ("topology.observer_lane_count",),
    "observers": ("topology.observers",),
    "sm_scope_mode": ("capture.software_sm_scope.mode", "capture.software_sm_scope.kind"),
    "sm_scope_selected": ("capture.software_sm_scope.selected_physical_smids",
                          "capture.software_sm_scope.smids"),
    "event_definitions": ("events.event_definitions",),
    "pairing_begin": ("events.pairing.begin_record_kind",),
    "pairing_end": ("events.pairing.end_record_kind",),
    "required_overflow": ("buffer_abi.capacity.required_overflow",),
    "max_records": ("buffer_abi.capacity.records_per_rank", "buffer_abi.max_records"),
    "hw_sample_rate_hz": ("hardware.metrics_config.sample_rate_hz",
                          "capture.hardware.sample_rate_hz",
                          "capture.hardware.frequency_hz"),
    "max_trace_bytes": ("artifacts.maximum_formal_trace_bytes", "artifacts.max_bytes"),
    "oversize_policy": ("artifacts.oversize_policy",),
    "candidate_perfetto": ("artifacts.candidate_perfetto", "artifacts.perfetto"),
    "device_source_combined": ("build_and_launch.device_source_combined_sha256",),
}
# ======================================================================
# ===== 配置区结束；下面是与项目无关的通用逻辑 =====
# ======================================================================

SCHEMA = "fast-instrument-preflight-item/v1"
ALL_CHECK_IDS = ["E{:02d}".format(index) for index in range(1, 28)]
HW_OPTIONAL_IDS = {"E17", "E18", "E19", "E20", "E21", "E22", "E25"}

# 被定位到的源文件必须是"真的那份被插桩的孪生体"，不能是从参考/测试/旧跑目录里
# 捞出来的副本（E01）。注意：本骨架自己就住在 skill 的 examples/demo/ 下，所以在
# 原地跑 E01 会命中 "/examples/demo/" 而 FAIL —— 这是刻意的：先把 profile/ 整棵树
# 拷进你的项目再跑。
EXCLUDED_PATH_TOKENS = ("/reference/", "/examples/demo/", "/testdata/", "/test_data/",
                        "/results/", "/profile_runs/", "/opt_version/", "/__pycache__/")

# run_profile.py 只在模块级 import 标准库（torch/cutlass 都在 GPU 函数内部），
# 所以在 CPU 机器上 import 它是安全的，可以直接拿到项目权威的 launch / observer 助手。
# import 失败不能让本适配器直接崩掉（那样连证据都写不出来 -> BLOCKED），
# 所以这里捕获并记录，具体 check 用到时才判 FAIL。
for _p in (str(_PROFILE_ROOT), str(_HOST_DECODE), str(_DEVICE_RECORD / "abi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
rp = None  # type: Any
RUN_PROFILE_IMPORT_ERROR = None  # type: Optional[str]
try:
    import run_profile as rp  # noqa: E402
except Exception as _error:  # noqa: BLE001 -- 诚实记录，不掩盖
    RUN_PROFILE_IMPORT_ERROR = "{}: {}".format(type(_error).__name__, _error)


# ----------------------------------------------------------------------
# 通用小工具
# ----------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def read_text(path: Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def is_file(path: Optional[str]) -> bool:
    return bool(path) and Path(path).is_file()


def missing_tokens(text: str, tokens) -> List[str]:
    return [token for token in tokens if token not in text]


def dig(node: Any, dotted: str, default: Any = None) -> Any:
    """按点分路径取嵌套值。"""
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def first(node: Any, dotted_paths, default: Any = None) -> Any:
    """取第一个命中的候选路径（值为 None 视为未命中）。"""
    for dotted in dotted_paths:
        value = dig(node, dotted, None)
        if value is not None:
            return value
    return default


def ckey(contract: Dict[str, Any], logical: str, default: Any = None) -> Any:
    return first(contract, CONTRACT_KEYS.get(logical, ()), default)


def akey(abi: Dict[str, Any], logical: str, default: Any = None) -> Any:
    return first(abi, PROJECT["abi_keys"].get(logical, ()), default)


def as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (list, tuple)):
        return as_int(value[0], default) if value else default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def tokens_for(key: str):
    """取某条 check 要求的 token 集合。

    空集合 = "什么都不要求" = missing_tokens() 永远返回 []，也就是静默通过。
    所以这里 fail-loud：token 集合被清空（或者 key 拼错了）直接抛错，让 main()
    把它记成 FAIL —— 想放宽某条检查必须显式改判据，不能靠"把清单删空"。
    """
    tokens = tuple(PROJECT["required_tokens"].get(key, ()))
    if not tokens:
        raise NotImplementedError(
            "PROJECT['required_tokens'][{!r}] 是空的：空 token 集合等于"
            "'什么都不检查'，会让这条 check 静默通过，所以直接判 FAIL。"
            "请在配置区填上本项目源文件里真实出现的标识符（或确认 key 有没有拼错）。"
            .format(key))
    return tokens


def cross_check(logical: str, sources, hint: str) -> Dict[str, Any]:
    """用**两个独立来源**互相印证一个期望值；缺失 / 只有一个来源都判不通过。

    ``sources`` = ``[(来源名, 值)]``，值为 ``None`` 表示"这个来源没声明"。
    调用方有责任只传**互相独立**的来源：把同一个字段算两遍（例如 warps 先由
    ``threads // 32`` 兜底算出来、再拿去和 ``threads // 32`` 比）不是印证，是恒真。

    判定（默认严格）：
      * 0 个来源声明 -> ``missing``   —— 期望值缺失本身就是必须报出来的缺口；
      * 1 个来源声明 -> ``single_source`` —— 自己证明自己，不算校验；
      * >=2 个来源但取值不一致 -> ``mismatch``；
      * >=2 个来源且一致 -> ``ok``。
    """
    declared = [(str(name), value) for name, value in sources if value is not None]
    values: List[Any] = []
    for _name, value in declared:
        norm = [as_int(item) for item in value] \
            if isinstance(value, (list, tuple)) else value
        if norm not in values:
            values.append(norm)
    if not declared:
        reason = "missing"
    elif len(declared) < 2:
        reason = "single_source"
    elif len(values) > 1:
        reason = "mismatch"
    else:
        reason = "ok"
    verdict = {"logical": logical, "ok": reason == "ok", "reason": reason,
               "sources": [{"source": name, "value": value} for name, value in declared],
               "distinct_values": values}
    if reason != "ok":
        verdict["hint"] = hint
    return verdict


class Ctx:
    """一条 check 需要的一切，只解析一次。"""

    def __init__(self, contract: Dict[str, Any], contract_path: Path):
        self.contract = contract if isinstance(contract, dict) else {}
        self.contract_path = contract_path
        # import / 读盘一律信任 __file__ 推出来的根；contract 声明的根只用于交叉核对。
        self.version_root = _VERSION_ROOT
        declared = ckey(self.contract, "version_root")
        self.declared_version_root = Path(declared) if declared else None
        self.profile_root = _PROFILE_ROOT
        declared_abi = ckey(self.contract, "abi_path")
        default_abi = _DEVICE_RECORD / "abi" / PROJECT["abi_filename"]
        self.abi_path = Path(declared_abi) if is_file(declared_abi) else default_abi
        self.abi = load_json(self.abi_path) if self.abi_path.is_file() else {}
        capture = self.contract.get("capture")
        self.capture = capture if isinstance(capture, dict) else {}
        self.display_rank = as_int(ckey(self.contract, "display_rank", 0), 0)
        self.world_size = max(1, as_int(ckey(self.contract, "world_size", 1), 1))
        comps = self.contract.get("components")
        self.components = comps if isinstance(comps, dict) else {}

    # -- 组件定位 ------------------------------------------------------
    def component_path(self, key: str) -> Optional[str]:
        """contract 声明优先，其次 PROJECT["component_relpaths"] 兜底。"""
        entry = self.components.get(key)
        if isinstance(entry, dict) and entry.get("absolute_path"):
            return entry["absolute_path"]
        if isinstance(entry, str):
            return entry
        rel = PROJECT["component_relpaths"].get(key)
        if rel:
            return str(_PROFILE_ROOT / rel)
        return None

    def component(self, key: str) -> Tuple[Path, str]:
        """返回 (path, text)。path 可能不存在，由调用方判 FAIL。"""
        declared = self.component_path(key)
        path = Path(declared) if declared else (_PROFILE_ROOT / "MISSING")
        return path, read_text(path)

    def device_writer_paths(self) -> List[Path]:
        """device 记录源候选：contract -> ABI source_file -> PROJECT 清单。"""
        found: List[Path] = []
        declared = self.component_path("device_writer")
        if is_file(declared):
            found.append(Path(declared))
        for key in ("source_file", "binding_source_file"):
            recorded = self.abi.get(key)
            if not recorded:
                continue
            candidate = Path(recorded)
            if not candidate.is_absolute():
                candidate = self.abi_path.resolve().parent / candidate
            if candidate.is_file():
                found.append(candidate)
        for rel in PROJECT["device_writer_relpaths"]:
            candidate = _PROFILE_ROOT / rel
            if candidate.is_file():
                found.append(candidate)
        unique: List[Path] = []
        for path in found:
            if path not in unique:
                unique.append(path)
        return unique

    def device_writer_text(self) -> str:
        return "\n".join(read_text(path) for path in self.device_writer_paths())


# ----------------------------------------------------------------------
# 项目钩子（无法泛化的部分）—— 一律 fail-loud，绝不静默通过
# ----------------------------------------------------------------------
# 下面这个路径是**参考实现（dsv4pro/decode，已通过 25 门禁）**，不是本项目的路径。
# 报错文案里引用它是为了让你有个能对照的正确样例；千万不要照抄成自己的配置值。
_REAL_IMPL = ("参考实现（dsv4pro/decode，已通过 25 门禁）DSA-learn/mk/h20/cutedsl/"
              "deepseek_v4_pro/decode/profile/validation/check_preflight_item.py")

DEVICE_WRITER_MISSING_REASON = (
    "device writer 源文件缺失，请在 PROJECT.device_writer_relpaths 里填本项目的实际路径"
    "（也可以在 contract 的 components.device_writer 或 ABI 的 source_file 里声明）。"
    "这是接新 target 的第一步。为什么必须 FAIL 而不是继续判：定位不到源文件时 "
    "device_writer_text() 返回空字符串，E01/E05/E08/E09 就变成'对着空字符串查 token'，"
    "结论完全没有意义 —— 而且一旦有人把 token 清单删空，这四条会直接静默变绿。"
)


def device_writer_missing(c: Ctx) -> Optional[Dict[str, Any]]:
    """定位不到任何 device 记录源 -> 返回 FAIL findings；能定位到 -> None。

    E01/E05/E08/E09 都靠 ``device_writer_text()`` 判 device 侧语义，所以四条都要先
    过这道闸：把"源文件根本没找到"和"源文件里少了某个 token"区分开，报错才指得准。
    """
    if c.device_writer_paths():
        return None
    return {
        "device_writer_sources": [],
        "device_writer_relpaths_tried": [str(_PROFILE_ROOT / rel)
                                         for rel in PROJECT["device_writer_relpaths"]],
        "contract_components_device_writer": c.component_path("device_writer"),
        "abi_source_file": c.abi.get("source_file"),
        "abi_binding_source_file": c.abi.get("binding_source_file"),
        "reason": DEVICE_WRITER_MISSING_REASON,
    }


def device_constants(c: Ctx, required=()) -> Tuple[Dict[str, Any], str]:
    """device 侧常量（warp 数 / SM 数 / 选中的物理 SM / 编译档位值）。

    解析顺序：run_profile.device_constants() -> contract/ABI 能推出来的 -> 报错。
    """
    values: Dict[str, Any] = {}
    sources: List[str] = []

    warps = ckey(c.contract, "physical_warps_per_cta")
    block = ckey(c.contract, "block")
    if warps is None and block is not None:
        warps = as_int(block) // 32 or None
    if warps is not None:
        values["WARPS_PER_CTA"] = as_int(warps)
        sources.append("contract")
    sm_count = ckey(c.contract, "physical_sm_count")
    if sm_count is not None:
        values["PHYSICAL_SM_COUNT"] = as_int(sm_count)
        sources.append("contract")

    scope = c.abi.get("capture_scope") if isinstance(c.abi.get("capture_scope"), dict) else {}
    selected = scope.get("selected_physical_smids")
    if selected is None:
        selected = ckey(c.contract, "sm_scope_selected")
    if isinstance(selected, list):
        values["SELECTED_PHYSICAL_SMIDS"] = [int(x) for x in selected]
        sources.append("abi/contract")
    slots = scope.get("capture_slots")
    if slots is not None and values.get("WARPS_PER_CTA"):
        values["NUM_CAPTURE_LANES"] = as_int(slots) * int(values["WARPS_PER_CTA"])
        sources.append("abi")

    levels = akey(c.abi, "compile_levels")
    if isinstance(levels, dict) and levels:
        values.update({str(k): v for k, v in levels.items()})
        sources.append("abi.compile_switch")

    getter = getattr(rp, "device_constants", None) if rp is not None else None
    if callable(getter):
        try:
            values.update(dict(getter()))
            sources.append("run_profile.device_constants")
        except Exception as error:  # noqa: BLE001
            sources.append("run_profile.device_constants FAILED: {}".format(error))

    missing = [key for key in required
               if values.get(key) is None or values.get(key) == [] or values.get(key) == {}]
    if missing:
        raise NotImplementedError(
            "接入新 target 时请提供 device 常量 {}。两种做法任选其一："
            "(1) 在 profile/run_profile.py 里实现 device_constants() -> dict，把 device 模块里的"
            " WARPS_PER_CTA / PHYSICAL_SM_COUNT / NUM_CAPTURE_LANES / SELECTED_PHYSICAL_SMIDS /"
            " 三个编译档位常量原样返回；(2) 让 extract_abi.py 把 capture_scope 与 compile_switch"
            " 写进 ABI，并在 contract 的 topology 段声明 warps/SM 数。"
            "参考真实实现 {}:180 (rp.device_constants() 的用法)".format(missing, _REAL_IMPL))
    return values, "+".join(dict.fromkeys(sources)) or "none"


def device_source_entries(c: Ctx) -> List[Dict[str, Any]]:
    entries = []
    for path in c.device_writer_paths():
        entries.append({"path": str(path), "exists": path.is_file(),
                        "sha256": sha256_file(path)})
    return entries


def generic_combined_sha256(entries: List[Dict[str, Any]]) -> Optional[str]:
    """通用 combined 定义：按声明顺序把各源文件 sha256 的原始字节拼起来再 sha256。

    项目如果用别的定义（真实项目在 run_profile.build_runtime_launch 里算），
    就实现 run_profile.build_runtime_launch()，本函数会被跳过。
    """
    digest = hashlib.sha256()
    seen = False
    for entry in entries:
        if entry.get("sha256"):
            digest.update(bytes.fromhex(entry["sha256"]))
            seen = True
    return digest.hexdigest() if seen else None


# runtime launch 描述符是"从 contract 现推"出来的时候用这个来源名。E14 靠它判断
# 描述符**是不是独立来源**：描述符本身就是 contract 抄过来的，再拿去和 contract 比
# 就是自己和自己比（恒真）。这时必须另外有一个独立来源（PROJECT 的 expected_*，
# 或 run_profile 自己算的 build_runtime_launch）才算真的校验过几何。
GENERIC_LAUNCH_SOURCE = "contract+abi(generic)"


def runtime_launch(c: Ctx) -> Tuple[Dict[str, Any], str]:
    """launch 描述符（grid / threads / rank / device 源 provenance / observers）。"""
    builder = getattr(rp, "build_runtime_launch", None) if rp is not None else None
    if callable(builder):
        return builder(c.version_root, c.abi_path, c.display_rank, c.world_size), \
            "run_profile.build_runtime_launch"

    grid = ckey(c.contract, "grid")
    block = ckey(c.contract, "block")
    geometry = getattr(rp, "resolve_launch_geometry", None) if rp is not None else None
    if callable(geometry) and (grid is None or block is None):
        try:
            resolved = geometry(c.contract)
            grid = grid if grid is not None else resolved.get("grid")
            block = block if block is not None else resolved.get("block")
        except Exception as error:  # noqa: BLE001
            del error
    if grid is None or block is None:
        raise NotImplementedError(
            "接入新 target 时 contract 必须声明 build_and_launch.grid 与 build_and_launch.block"
            "（或 threads_per_cta）；若你的 launch 几何要在运行期算，请在 profile/run_profile.py"
            " 里实现 build_runtime_launch(version_root, abi_path, rank, world_size) -> dict，"
            "返回 grid/threads_per_cta/rank/device_source/device_source_combined_sha256/observers。"
            "参考真实实现 {}:357 (rp.build_runtime_launch)".format(_REAL_IMPL))

    sources = device_source_entries(c)
    topology, _src = observer_topology(c, as_int(block))
    return {
        "grid": list(grid) if isinstance(grid, (list, tuple)) else [as_int(grid), 1, 1],
        "threads_per_cta": as_int(block),
        "rank": c.display_rank,
        "device_source": sources,
        "device_source_combined_sha256": generic_combined_sha256(sources),
        "combined_sha256_definition": "sha256(concat(raw_sha256_bytes_in_declared_order))",
        "observers": topology.get("lanes", []),
    }, GENERIC_LAUNCH_SOURCE


def observer_topology(c: Ctx, threads_per_cta: int) -> Tuple[Dict[str, Any], str]:
    """observer lane 拓扑：一个 lane 代表哪些物理 warp。

    ⚠ 返回的 ``physical_warps_per_cta`` **可能是 threads//32 兜底值**（契约没声明
    ``physical_warps_per_cta`` 时走下面那个 ``or`` 分支），所以它**不是声明值**：
    调用方绝对不能把它拿去和 ``threads // 32`` 比（那是恒真）。要交叉印证 warp 数
    请用 ``declared_warps_sources()``，它把声明侧和运行侧分开列。
    """
    builder = getattr(rp, "observer_topology", None) if rp is not None else None
    if callable(builder):
        return builder(c.abi, threads_per_cta), "run_profile.observer_topology"

    warps = as_int(ckey(c.contract, "physical_warps_per_cta"), 0) or (as_int(threads_per_cta) // 32)
    observers = ckey(c.contract, "observers")
    lanes: List[Dict[str, Any]] = []
    if isinstance(observers, list) and observers:
        for item in observers:
            if not isinstance(item, dict):
                continue
            warp = item.get("observer_warp")
            represents = item.get("represents")
            if not (isinstance(represents, (list, tuple)) and len(represents) == 2):
                represents = [as_int(warp), as_int(warp)]
            lanes.append({"observer_warp": as_int(warp),
                          "represents": [as_int(represents[0]), as_int(represents[1])],
                          "role": item.get("role")})
        source = "contract.topology.observers"
    else:
        for track in c.abi.get("tracks", []) or []:
            if not isinstance(track, dict) or "track_id" not in track:
                continue
            tid = as_int(track["track_id"])
            lanes.append({"observer_warp": tid, "represents": [tid, tid],
                          "role": track.get("role")})
        source = "abi.tracks"
    if not lanes:
        raise NotImplementedError(
            "接入新 target 时无法推出 observer lane 拓扑：contract 的 topology.observers 为空，"
            "ABI 的 tracks 也为空。请让 extract_abi.py 写出 tracks[]（每个记录 warp 一条），"
            "或在 profile/run_profile.py 里实现 observer_topology(abi, threads_per_cta) -> "
            "{physical_warps_per_cta, observer_lane_count, lanes:[{observer_warp, represents}]}。"
            "参考真实实现 {}:377 (rp.observer_topology)".format(_REAL_IMPL))
    lanes.sort(key=lambda item: item["observer_warp"])
    return {"physical_warps_per_cta": warps, "observer_lane_count": len(lanes),
            "lanes": lanes}, source


def declared_warps_sources(c: Ctx, threads_per_cta: int):
    """warp/CTA 的**互相独立**的来源清单：``[(来源名, 值 或 None)]``。

    E15 以前写的是 ``warps == threads // 32``，而 ``warps`` 在契约没声明
    ``physical_warps_per_cta`` 时**本来就是 threads // 32 兜底算出来的** ——
    自己和自己比，恒真。所以这里把两侧拆开：

      * 声明侧：contract ``build_and_launch/topology.physical_warps_per_cta``、
        ABI 的 launch/topology/capture_scope 段。**不做 threads//32 兜底**，
        没声明就是 None（缺口要报出来，不能悄悄补上）。
      * 运行侧：runtime launch 的 threads_per_cta // 32。
      * PROJECT 配置区已知值（接不进契约/ABI 时的第二来源）。
    """
    contract_declared = ckey(c.contract, "physical_warps_per_cta")
    abi_declared = first(c.abi, ("launch.physical_warps_per_cta",
                                 "topology.physical_warps_per_cta",
                                 "capture_scope.physical_warps_per_cta"))
    derived = (as_int(threads_per_cta) // 32) or None
    expected = PROJECT["expected_physical_warps_per_cta"]
    return [
        ("contract.physical_warps_per_cta",
         as_int(contract_declared) if contract_declared is not None else None),
        ("abi.physical_warps_per_cta",
         as_int(abi_declared) if abi_declared is not None else None),
        ("runtime_launch.threads_per_cta//32", derived),
        ("PROJECT.expected_physical_warps_per_cta",
         as_int(expected) if expected is not None else None),
    ]


def observer_lane_count_sources(c: Ctx, topo: Dict[str, Any], topo_source: str):
    """observer lane 条数的**互相独立**的来源清单。

    以前 E15 在 ``contract.topology.observer_lane_count`` 缺失时退回
    ``len(contract.topology.observers)`` —— 而 lane 列表本身就是从这个 observers
    数出来的，于是 ``declared == observer_lane_count`` 恒真（骨架实测 8 == 8）。
    现在按"枚举来源"把同源的那一条剔掉：只有来自**另一个**声明的条数才算印证。
    """
    enumerated = as_int(topo.get("observer_lane_count"))
    scalar = ckey(c.contract, "observer_lane_count")   # 显式标量，永远独立于列表
    observers = ckey(c.contract, "observers")
    observers_count = len(observers) if isinstance(observers, list) and observers else None
    tracks = [t for t in (c.abi.get("tracks") or []) if isinstance(t, dict)]
    tracks_count = len(tracks) or None
    expected = PROJECT["expected_observer_lane_count"]
    sources = [("topology[{}]".format(topo_source), enumerated or None),
               ("contract.topology.observer_lane_count",
                as_int(scalar) if scalar is not None else None),
               ("PROJECT.expected_observer_lane_count",
                as_int(expected) if expected is not None else None)]
    if topo_source != "contract.topology.observers":
        sources.append(("len(contract.topology.observers)", observers_count))
    if topo_source != "abi.tracks":
        sources.append(("len(abi.tracks)", tracks_count))
    return sources


def decoder_selftest(c: Ctx) -> Tuple[bool, List[str], str]:
    """解码器自测：decode_profile.run_self_test() 优先，否则跑 selftest_decode.py 子进程。"""
    try:
        import decode_profile as dp  # noqa: E402
    except Exception as error:  # noqa: BLE001
        return False, ["cannot import decode_profile: {}: {}".format(
            type(error).__name__, error)], "import_failed"
    runner = getattr(dp, "run_self_test", None)
    if callable(runner):
        ok, failures = runner(str(c.abi_path))
        return bool(ok), list(failures or []), "decode_profile.run_self_test"
    script = _VALIDATION / "selftest" / "selftest_decode.py"
    if not script.is_file():
        return False, ["neither decode_profile.run_self_test nor {} exists".format(script)], \
            "missing"
    completed = subprocess.run([sys.executable, "-B", str(script)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               cwd=str(_PROFILE_ROOT), timeout=300, check=False)
    stderr = (completed.stderr or b"").decode("utf-8", "replace").strip()
    ok = completed.returncode == 0
    return ok, ([] if ok else [stderr.splitlines()[-1] if stderr else
                               "exit={}".format(completed.returncode)]), \
        "child:selftest_decode.py"


# ----------------------------------------------------------------------
# E01-E16 : device 记录链 + host 解码链（全部可 CPU 判定）
# ----------------------------------------------------------------------
def check_e01(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    blocked = device_writer_missing(c)
    if blocked is not None:
        return False, blocked
    target = str(ckey(c.contract, "target_kernel") or "")
    needle = target.split(".")[-1] if target else ""
    writers = c.device_writer_paths()
    text = c.device_writer_text()
    excluded = sorted({seg for path in writers for seg in EXCLUDED_PATH_TOKENS
                       if seg in (str(path) + "/").lower()})
    abi_kernel = str(dig(c.abi, "target.kernel") or "")
    has_entry = bool(needle) and needle in text
    ok = (bool(writers) and bool(target) and has_entry and not excluded
          and abi_kernel == target)
    return ok, {"target_kernel": target, "device_writer_sources": [str(p) for p in writers],
                "device_writer_mentions_target": has_entry, "abi_kernel": abi_kernel,
                "abi_kernel_matches_contract": abi_kernel == target,
                "excluded_path_tokens_hit": excluded,
                "hint": ("device 记录源还在 references|results 之类目录下时也算不通过："
                         "先把整棵 profile/ 拷进项目，再更新 "
                         "PROJECT['device_writer_relpaths'] 或 contract "
                         "components.device_writer。" + DEVICE_WRITER_MISSING_REASON)}


def check_e02(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    clock = c.abi.get("clock") if isinstance(c.abi.get("clock"), dict) else {}
    width = clock.get("width_bits")
    intrinsic = str(akey(c.abi, "clock_intrinsic") or "")
    fields = akey(c.abi, "record_fields") or []
    word0 = fields[0] if isinstance(fields, list) and fields else {}
    ts_dtype = str(word0.get("dtype") or "")
    ok = (as_int(width) == 64 and "globaltimer" in intrinsic.lower()
          and as_int(word0.get("index"), -1) == 0
          and ts_dtype in ("uint64", "int64")
          and str(clock.get("unit") or "ns") == "ns")
    return ok, {"clock_width_bits": width, "read_intrinsic": intrinsic,
                "timestamp_field": word0, "clock_unit": clock.get("unit"),
                "element_dtype": dig(c.abi, "buffer.element_dtype")}


def check_e03(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    bitfields = c.abi.get("tag_bitfields") if isinstance(c.abi.get("tag_bitfields"), dict) else {}
    is_start = bitfields.get(PROJECT["tag_is_start_field"], {})
    begin = ckey(c.contract, "pairing_begin")
    end = ckey(c.contract, "pairing_end")
    want = PROJECT["pairing_kinds"]
    layers = {t.get("layer") for t in c.abi.get("tasks", []) or [] if isinstance(t, dict)}
    pairing_ok = (begin is not None and end is not None and begin != end
                  and as_int(begin) == as_int(want["begin"])
                  and as_int(end) == as_int(want["end"]))
    # 事件 id 位域：tag 里必须真的有一个字段能装下 ABI 声明的最大 task id。
    # 宽度期望值从 ABI tasks[] 现推（不写死），ABI 一改这条就跟着改判。
    event_field = None
    for name in PROJECT["tag_event_id_field_candidates"]:
        if isinstance(bitfields.get(name), dict):
            event_field = name
            break
    event_width = as_int((bitfields.get(event_field) or {}).get("width")) if event_field else 0
    task_ids = [as_int(t["id"]) for t in c.abi.get("tasks", []) or []
                if isinstance(t, dict) and "id" in t]
    max_task_id = max(task_ids) if task_ids else -1
    capacity = (1 << event_width) - 1 if event_width > 0 else -1
    event_id_ok = bool(event_field) and event_width >= 1 and max_task_id >= 0 \
        and max_task_id <= capacity
    ok = (as_int(is_start.get("width")) == 1 and pairing_ok and event_id_ok
          and "task" in layers and "phase" in layers)
    return ok, {"is_start_field": is_start,
                "contract_pairing": {"begin_record_kind": begin, "end_record_kind": end},
                "expected_pairing": want, "pairing_ok": pairing_ok,
                "event_id_field": event_field, "event_id_field_width": event_width,
                "event_id_field_candidates": list(PROJECT["tag_event_id_field_candidates"]),
                "abi_max_task_id": max_task_id, "event_id_field_capacity": capacity,
                "event_id_field_ok": event_id_ok,
                "task_layers_present": sorted(x for x in layers if x),
                "hint": ("tag_bitfields 里必须有一个事件 id 位域（候选名见 PROJECT"
                         "['tag_event_id_field_candidates']），且宽度装得下 ABI tasks[] "
                         "里最大的 id —— 装不下就有事件 id 会被截断成别的事件")}


def compile_levels_distinct(files: Dict[str, Any], abi_levels: Any,
                            names: Dict[str, str]) -> Tuple[bool, Dict[str, Any]]:
    """三档编译开关"实现上真的互不相同"——与档位实现形式无关的判法。

    以前写的是 ``three_distinct or bool(abi_levels)``：只要 ABI 提了一句档位，
    三个 preset 就算逐字节相同也照过 —— 那等于 off/entry_only/full 编出同一个
    二进制，overhead 恒为 0、correctness 参考档位其实开着插桩。
    """
    paths = [str(files[role].get("path")) for role in names if role in files]
    detail: Dict[str, Any] = {"level_impl_paths": paths}
    if len(set(paths)) == len(paths) and paths:
        # 三个独立文件：JSON preset 比 compile_defines，其它形式比文件内容 sha256。
        defines = {json.dumps(files[role].get("compile_defines"), sort_keys=True)
                   for role in names if role in files}
        if all(files[role].get("compile_defines") is not None
               for role in names if role in files):
            detail.update({"basis": "compile_defines", "distinct_count": len(defines)})
            return len(defines) == len(paths), detail
        hashes = {sha256_file(Path(p)) for p in paths}
        detail.update({"basis": "file_sha256", "distinct_count": len(hashes)})
        return len(hashes) == len(paths) and None not in hashes, detail
    # 三档共用一个模块（Python 常量方案）：靠 ABI 的三个档位取值 + 模块里三个名字都在。
    shared_text = read_text(Path(paths[0])) if paths else ""
    values = list(abi_levels.values()) if isinstance(abi_levels, dict) else []
    distinct_values = len({json.dumps(v, sort_keys=True) for v in values})
    names_in_text = [names[role] for role in names if names[role] in shared_text]
    detail.update({"basis": "shared_module+abi_level_values",
                   "distinct_abi_level_values": distinct_values,
                   "level_names_found_in_module": names_in_text})
    return distinct_values == len(names) and len(names_in_text) == len(names), detail


def check_e04(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    names = PROJECT["compile_level_names"]
    abi_levels = akey(c.abi, "compile_levels")
    files = {}
    for role, rel in PROJECT["compile_level_files"].items():
        path = _PROFILE_ROOT / rel
        entry: Dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file() and path.suffix == ".json":
            try:
                doc = load_json(path)
                entry["name"] = doc.get("name")
                entry["compile_defines"] = doc.get("compile_defines")
            except ValueError as error:
                entry["error"] = str(error)
        files[role] = entry
    all_present = all(entry["exists"] for entry in files.values())
    names_match = all(str(files[role].get("name") or names[role]) == names[role]
                      for role in names if role in files)
    # ABI 必须**声明**三档；以前这里是 `abi_ok = True`（没声明就不核对），
    # 等于"ABI 里压根没有编译开关"也能过 —— 那三档到底存不存在就没有任何证据。
    abi_declared = isinstance(abi_levels, dict) and bool(abi_levels)
    abi_ok = abi_declared and \
        sorted(str(k) for k in abi_levels) == sorted(names[r] for r in names)
    distinct_ok, distinct_detail = (compile_levels_distinct(files, abi_levels, names)
                                    if all_present else (False, {"basis": "files_missing"}))
    ok = all_present and names_match and distinct_ok and abi_ok
    return ok, {"expected_level_names": names, "level_files": files,
                "abi_compile_switch_levels": abi_levels,
                "abi_declares_compile_levels": abi_declared,
                "three_distinct_compile_defines": distinct_ok,
                "distinctness_detail": distinct_detail,
                "abi_levels_match_project_names": abi_ok,
                "abi_keys_compile_levels": list(PROJECT["abi_keys"]["compile_levels"]),
                "hint": ("(1) 请在 ABI 里声明三档编译开关（{}），三档 = 关闭插桩 / "
                         "仅入口打点 / 完整事件记录；键名要和 PROJECT"
                         "['compile_level_names'] 的取值一一对上。ABI 没声明就 FAIL，"
                         "不再默认放过。(2) 三档的实现必须真的不同（preset 比 "
                         "compile_defines、共用模块比 ABI 档位取值）—— 三档相同 = "
                         "overhead 恒 0 + correctness 参考档位其实开着插桩。"
                         .format("/".join(PROJECT["abi_keys"]["compile_levels"]))) }


def check_e05(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    blocked = device_writer_missing(c)
    if blocked is not None:
        return False, blocked
    scope = c.abi.get("capture_scope") if isinstance(c.abi.get("capture_scope"), dict) else {}
    text = c.device_writer_text()
    missing = missing_tokens(text, tokens_for("device_writer_smid"))
    const, const_src = device_constants(c, required=("PHYSICAL_SM_COUNT",
                                                     "SELECTED_PHYSICAL_SMIDS"))
    selected = list(const["SELECTED_PHYSICAL_SMIDS"])
    sm_count = as_int(const["PHYSICAL_SM_COUNT"])
    mode = str(scope.get("mode") or ckey(c.contract, "sm_scope_mode") or "").lower()
    filter_site = str(scope.get("filter_site") or "").lower()
    abi_selected = [int(x) for x in (scope.get("selected_physical_smids") or [])]
    # 任何 capture 模式都必须：device 源侧按 %smid 过滤 + 明确禁止后处理过滤，
    # 且 device 选中的集合与 ABI 声明一致。
    common = (not missing and "source-side" in filter_site and "%smid" in filter_site
              and scope.get("post_processing_filter_forbidden") is True
              and (not abi_selected or selected == abi_selected))
    if mode in ("all", "all_sms"):
        scope_ok = sm_count > 0 and selected == list(range(sm_count))
    else:
        scope_ok = 0 < len(selected) < sm_count and selected == sorted(set(selected))
    ok = bool(common and scope_ok)
    return ok, {"mode": mode, "selected_physical_smids": selected,
                "physical_sm_count": sm_count, "filter_site": scope.get("filter_site"),
                "device_writer_missing_tokens": missing,
                "post_processing_filter_forbidden": scope.get("post_processing_filter_forbidden"),
                "device_constants_source": const_src,
                "hint": ("ABI 需要 capture_scope 段（mode / selected_physical_smids / "
                         "filter_site 含 'source-side %smid' / post_processing_filter_forbidden）")}


def check_e06(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    scope = c.abi.get("capture_scope") if isinstance(c.abi.get("capture_scope"), dict) else {}
    const, const_src = device_constants(c, required=("WARPS_PER_CTA", "PHYSICAL_SM_COUNT"))
    slots = as_int(scope.get("capture_slots"))
    warps = as_int(const["WARPS_PER_CTA"])
    sm_count = as_int(const["PHYSICAL_SM_COUNT"])
    lanes = as_int(const.get("NUM_CAPTURE_LANES"), slots * warps)
    scope_count = as_int(scope.get("selected_physical_sm_count"), slots)
    _harness_path, harness_text = c.component("host_harness")
    mode = str(scope.get("mode") or ckey(c.contract, "sm_scope_mode") or "").lower()
    was_all = scope.get("source_capture_was_all_sm")
    basis_ok = scope.get("buffer_allocation_basis") == \
        "selected_physical_sm_count_and_cta_capacity"
    lanes_ok = warps > 0 and slots > 0 and lanes == slots * warps
    host_ok = ("capture_scope" in harness_text) and ("max_records" in harness_text
                                                     or "records_per_lane" in harness_text)
    slot_ok = slots == scope_count
    if mode in ("all", "all_sms"):
        scope_ok = slots == sm_count and sm_count > 0 and was_all is True
    else:
        scope_ok = 0 < slots < sm_count and was_all is False
    ok = bool(basis_ok and lanes_ok and host_ok and slot_ok and scope_ok)
    return ok, {"mode": mode, "capture_slots": slots, "warps_per_cta": warps,
                "num_capture_lanes": lanes, "physical_sm_count": sm_count,
                "expected_lanes": slots * warps, "source_capture_was_all_sm": was_all,
                "buffer_allocation_basis": scope.get("buffer_allocation_basis"),
                "host_allocs_by_lane_count": host_ok,
                "device_constants_source": const_src}


def check_e07(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    lane_formula = str(akey(c.abi, "lane_formula") or "")
    elem_formula = str(akey(c.abi, "element_index_formula") or "")
    bitfields = c.abi.get("tag_bitfields") if isinstance(c.abi.get("tag_bitfields"), dict) else {}
    identity = tuple(PROJECT["lane_identity_tag_fields"])
    in_formula = [name for name in identity if name in lane_formula]
    in_tag = [name for name in identity if name in bitfields]
    ok = (bool(lane_formula) and bool(elem_formula)
          and len(in_formula) == len(identity) and len(in_tag) == len(identity)
          and ("index" in elem_formula or "offset" in elem_formula))
    return ok, {"lane_formula": lane_formula, "element_index_formula": elem_formula,
                "lane_identity_fields": list(identity),
                "identity_fields_in_lane_formula": in_formula,
                "identity_fields_in_tag": in_tag,
                "hint": "改 PROJECT['lane_identity_tag_fields'] 成你 tag 里唯一定位 lane 的字段"}


def check_e08(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    blocked = device_writer_missing(c)
    if blocked is not None:
        return False, blocked
    maxr = as_int(akey(c.abi, "max_records_per_lane"), 0) or as_int(ckey(c.contract, "max_records"), 0)
    expected = PROJECT["expected_max_records_per_lane"]
    policy = str(akey(c.abi, "overflow_policy") or "").lower()
    ovf_name = PROJECT["overflow_tensor_name"]
    abi_ovf = dig(c.abi, "buffer.overflow_tensor.name")
    text = c.device_writer_text()
    missing = missing_tokens(text, tokens_for("device_writer_capacity"))
    required_overflow = ckey(c.contract, "required_overflow")
    sentinel = dig(c.abi, "buffer.empty_slot_sentinel")
    maxr_ok = maxr > 0 and (expected is None or maxr == as_int(expected))
    ovf_ok = (abi_ovf == ovf_name) if ovf_name else bool(sentinel)
    policy_ok = ("drop" in policy) or ("stop" in policy)
    overflow_required_zero = (required_overflow is None) or as_int(required_overflow, -1) == 0
    ok = bool(maxr_ok and ovf_ok and policy_ok and not missing and overflow_required_zero)
    return ok, {"max_records_per_lane": maxr, "expected_max_records_per_lane": expected,
                "overflow_policy": policy, "expected_overflow_tensor": ovf_name,
                "abi_overflow_tensor": abi_ovf, "empty_slot_sentinel": sentinel,
                "device_writer_missing_tokens": missing,
                "contract_required_overflow": required_overflow}


def check_e09(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    blocked = device_writer_missing(c)
    if blocked is not None:
        return False, blocked
    anchor = dig(c.abi, "sync_anchors.entry_rendezvous")
    anchor = anchor if isinstance(anchor, dict) else {}
    tasks_by_id = {as_int(t["id"]): t for t in c.abi.get("tasks", []) or []
                   if isinstance(t, dict) and "id" in t}
    anchor_id = anchor.get("task_id")
    entry_task = tasks_by_id.get(as_int(anchor_id, -1), {})
    start_sem = str(anchor.get("start_semantics") or "")
    end_sem = str(anchor.get("end_semantics") or "")
    want = PROJECT["anchor_semantics"]
    start_ok = any(tok in start_sem for tok in want["start"])
    end_ok = any(tok in end_sem for tok in want["end"])
    text = c.device_writer_text()
    missing = missing_tokens(text, tokens_for("device_writer_anchor"))
    ok = bool(anchor_id is not None and entry_task and start_ok and end_ok
              and anchor.get("guaranteed_first_event_per_cta") is True and not missing)
    return ok, {"entry_task_id": anchor_id, "entry_task": entry_task.get("name"),
                "task_id_in_abi_tasks": bool(entry_task),
                "start_semantics": start_sem, "end_semantics": end_sem,
                "start_semantics_ok": start_ok, "end_semantics_ok": end_ok,
                "guaranteed_first_event_per_cta": anchor.get("guaranteed_first_event_per_cta"),
                "device_writer_missing_tokens": missing}


def check_e10(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    abi_view: Dict[int, Tuple[Any, Any]] = {}
    for task in c.abi.get("tasks", []) or []:
        if isinstance(task, dict) and "id" in task:
            abi_view[as_int(task["id"])] = (task.get("name"), task.get("layer"))
    defs = ckey(c.contract, "event_definitions") or []
    contract_view: Dict[int, Tuple[Any, Any]] = {}
    for entry in defs if isinstance(defs, list) else []:
        if isinstance(entry, dict) and "event_id" in entry:
            contract_view[as_int(entry["event_id"])] = (entry.get("name"), entry.get("layer"))
    bad_layer = sorted(tid for tid, (_n, layer) in abi_view.items()
                       if layer not in ("task", "phase", "ignored"))
    phase_without_parent = []
    for task in c.abi.get("tasks", []) or []:
        if not isinstance(task, dict):
            continue
        if task.get("layer") == "phase" and task.get("parent_task_id") in (None, "") \
                and not task.get("parent_resolution"):
            phase_without_parent.append(as_int(task.get("id"), -1))
    ok = bool(abi_view) and bool(contract_view) and abi_view == contract_view \
        and not bad_layer and not phase_without_parent
    return ok, {"abi_event_count": len(abi_view), "contract_event_count": len(contract_view),
                "definitions_match": abi_view == contract_view,
                "bad_layer_ids": bad_layer, "phase_without_parent": phase_without_parent,
                "only_in_abi": sorted(set(abi_view) - set(contract_view)),
                "only_in_contract": sorted(set(contract_view) - set(abi_view)),
                "hint": "contract 的 events.event_definitions 必须与 ABI tasks[] 逐条一致"}


def check_e11(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    path, text = c.component("contract_materializer")
    missing = missing_tokens(text, tokens_for("materializer"))
    has_probe = ("probe_gpu" in text) or ("nvidia-smi" in text)
    # materializer 暴露了 --information-bundle，那 contract 就必须真的落下这个选择。
    # 以前 CONTRACT_KEYS["information_bundle"] 是死配置（列了候选路径但没人读），
    # 于是"用户选了哪个 bundle"这件事从来没被核对过。
    bundle = ckey(c.contract, "information_bundle")
    bundle_ok = isinstance(bundle, str) and "sw" in bundle.lower()
    ok = path.is_file() and not missing and has_probe and bundle_ok
    return ok, {"path": str(path), "exists": path.is_file(),
                "missing_tokens": missing, "has_device_probe": has_probe,
                "contract_information_bundle": bundle,
                "information_bundle_ok": bundle_ok,
                "hint": ("contract 必须声明 capture.information_bundle（如 sw / sw+hw）；"
                         "软件时间线永远不可选，所以取值里必须含 'sw'")}


def check_e12(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    abi_path = c.abi_path
    under_abi_dir = abi_path.parent.name == "abi" and "device_record" in str(abi_path)
    have_buffer = bool(c.abi.get("buffer"))
    have_bits = bool(c.abi.get("tag_bitfields"))
    have_events = bool(c.abi.get("tasks"))
    have_formula = bool(akey(c.abi, "element_index_formula"))
    real_sha = sha256_file(abi_path) if abi_path.is_file() else None
    declared = [s for s in (ckey(c.contract, "abi_sha256"),
                            dig(c.contract, "buffer_abi.abi_sha256")) if s]
    sha_ok = bool(real_sha) and bool(declared) and all(s == real_sha for s in declared)
    ok = bool(abi_path.is_file() and under_abi_dir and have_buffer and have_bits
              and have_events and have_formula and sha_ok)
    return ok, {"abi_path": str(abi_path), "under_device_record_abi": under_abi_dir,
                "has_buffer": have_buffer, "has_bitfields": have_bits,
                "has_event_ids": have_events, "has_address_formula": have_formula,
                "abi_sha256_real": real_sha, "abi_sha256_declared": declared,
                "abi_sha256_consistent": sha_ok}


def check_e13(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    path, text = c.component("host_harness")
    missing = missing_tokens(text, tokens_for("host_harness"))
    import_ok = RUN_PROFILE_IMPORT_ERROR is None
    ok = path.is_file() and not missing and import_ok
    return ok, {"path": str(path), "exists": path.is_file(), "missing_tokens": missing,
                "cpu_importable": import_ok,
                "import_error": RUN_PROFILE_IMPORT_ERROR}


def check_e14(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    runtime, source = runtime_launch(c)
    grid = list(runtime["grid"])
    contract_grid = ckey(c.contract, "grid")
    contract_grid = list(contract_grid) if isinstance(contract_grid, (list, tuple)) else None
    expected_grid = PROJECT["expected_grid"]
    threads = as_int(runtime["threads_per_cta"])
    contract_threads = as_int(ckey(c.contract, "block"), -1)
    expected_threads = PROJECT["expected_threads_per_cta"]
    # 描述符是"从 contract 现推"的时候，它不是独立来源（拿 contract 和 contract 比）。
    # 以前这里写的是 `expected_grid is None or grid == expected_grid`：没填期望值就
    # 整条不校验，几何等于没查。现在缺期望值本身就是缺口，reason 会写 single_source。
    descriptor_independent = source != GENERIC_LAUNCH_SOURCE
    geometry_hint = ("几何必须有**两个独立来源**互相印证。补齐任意一条即可："
                     "(1) contract 声明 build_and_launch.grid / .block；"
                     "(2) 在 PROJECT['expected_grid'] / ['expected_threads_per_cta'] "
                     "填本 target 的已知值；(3) 在 run_profile.py 里实现 "
                     "build_runtime_launch()，让描述符成为独立于 contract 的来源。"
                     "reason=single_source 表示只有一个来源 —— 自己证明自己不算校验。")
    grid_verdict = cross_check("grid", [
        ("runtime_launch[{}]".format(source), grid if descriptor_independent else None),
        ("contract.build_and_launch.grid", contract_grid),
        ("PROJECT.expected_grid",
         list(expected_grid) if expected_grid is not None else None),
    ], geometry_hint)
    threads_verdict = cross_check("threads_per_cta", [
        ("runtime_launch[{}]".format(source), threads if descriptor_independent else None),
        ("contract.build_and_launch.block",
         contract_threads if contract_threads > 0 else None),
        ("PROJECT.expected_threads_per_cta",
         as_int(expected_threads) if expected_threads is not None else None),
    ], geometry_hint)
    grid_ok = bool(grid_verdict["ok"]) and len(grid) == 3 and all(x > 0 for x in
                                                                 (as_int(v) for v in grid))
    threads_ok = bool(threads_verdict["ok"]) and threads > 0
    rank_ok = as_int(runtime["rank"]) == c.display_rank
    sources = runtime.get("device_source") or []
    src_ok = bool(sources) and all(entry.get("sha256") for entry in sources
                                   if entry.get("exists"))
    combined = str(runtime.get("device_source_combined_sha256") or "")
    declared_combined = ckey(c.contract, "device_source_combined")
    combined_ok = len(combined) == 64 and (declared_combined is None
                                           or combined == declared_combined)
    observers_ok = bool(runtime.get("observers"))
    ok = bool(grid_ok and threads_ok and rank_ok and src_ok and combined_ok and observers_ok)
    return ok, {"descriptor_source": source, "grid": grid, "contract_grid": contract_grid,
                "descriptor_independent_of_contract": descriptor_independent,
                "threads_per_cta": threads, "contract_threads_per_cta": contract_threads,
                "rank": runtime["rank"], "grid_ok": grid_ok, "threads_ok": threads_ok,
                "grid_cross_check": grid_verdict,
                "threads_cross_check": threads_verdict,
                "device_source": sources, "device_source_all_hashed": src_ok,
                "device_source_combined_sha256": combined,
                "contract_device_source_combined_sha256": declared_combined,
                "combined_sha_ok": combined_ok,
                "combined_sha256_definition": runtime.get("combined_sha256_definition"),
                "observers": len(runtime.get("observers") or []),
                "hint": ("contract 声明的 combined sha 与本地算法不同时，请在 run_profile.py 里"
                         "实现 build_runtime_launch() 用项目自己的定义。" + geometry_hint)}


def check_e15(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    runtime, _src = runtime_launch(c)
    threads = as_int(runtime["threads_per_cta"])
    topo, source = observer_topology(c, threads)
    lanes = topo["lanes"]
    warps = as_int(topo["physical_warps_per_cta"])
    # 以前：warps_ok = warps > 0 and warps == threads // 32 —— 而 warps 在契约没声明
    # physical_warps_per_cta 时就是 threads//32 兜底算的，等于自己和自己比（恒真）。
    # 现在拿"声明侧（contract/ABI）"和"运行侧（threads//32）"两个独立来源互相印证。
    warps_verdict = cross_check(
        "physical_warps_per_cta", declared_warps_sources(c, threads),
        ("warp 数必须有两个独立来源：契约/ABI 的声明 vs runtime launch 的 "
         "threads_per_cta//32。请在 contract 的 topology.physical_warps_per_cta "
         "（或 build_and_launch.physical_warps_per_cta / ABI 的 launch 段）声明 warp 数，"
         "或在 PROJECT['expected_physical_warps_per_cta'] 填已知值。"
         "reason=missing 表示一个来源都没有，reason=single_source 表示只有一个。"))
    # 以前：declared 缺失时退回 len(observers)，而 lane 列表就是从 observers 数出来的。
    lanes_verdict = cross_check(
        "observer_lane_count", observer_lane_count_sources(c, topo, source),
        ("observer lane 条数必须由**枚举之外**的来源印证：contract 的 "
         "topology.observer_lane_count 标量、ABI 的 tracks[] 条数、或 "
         "PROJECT['expected_observer_lane_count']。拿 observers 列表的长度去印证由它自己"
         "数出来的 lane 数是恒真，已经不再接受。"))
    corroborated_warps = warps_verdict["distinct_values"][0] \
        if warps_verdict["ok"] else None
    warps_ok = bool(warps_verdict["ok"]) and warps > 0 and warps == as_int(corroborated_warps)
    lanes_ok = bool(lanes_verdict["ok"])
    if warps_ok and lanes:
        covers = (lanes[0]["represents"][0] == 0
                  and lanes[-1]["represents"][1] == as_int(corroborated_warps) - 1)
    else:
        covers = False
    ok = bool(warps_ok and lanes_ok and covers)
    return ok, {"topology_source": source, "physical_warps_per_cta": warps,
                "corroborated_physical_warps_per_cta": corroborated_warps,
                "observer_lane_count": topo["observer_lane_count"],
                "contract_observer_lane_count": ckey(c.contract, "observer_lane_count"),
                "warps_cross_check": warps_verdict,
                "observer_lane_count_cross_check": lanes_verdict,
                "warps_ok": warps_ok, "lanes_ok": lanes_ok,
                "covers_full_range": covers,
                "observer_warps": [lane["observer_warp"] for lane in lanes],
                "hint": ("每条 lane 必须连续覆盖 warp 0..corroborated_warps-1；"
                         "warp 数与 lane 数都不接受单来源自证")}


def check_e16(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    path, text = c.component("decoder")
    ok_self, failures, source = decoder_selftest(c)
    missing = missing_tokens(text, tokens_for("decoder"))
    ok = bool(ok_self) and not failures and path.is_file() and not missing
    return ok, {"path": str(path), "self_test_ok": bool(ok_self),
                "self_test_source": source, "self_test_failures": failures,
                "diagnostic_missing_tokens": missing}


# ----------------------------------------------------------------------
# E17-E22, E25 : HW / Perfetto 链 + 离线 selftest（HW 可选）
# ----------------------------------------------------------------------
def check_e17(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    path, text = c.component("hardware_reader")
    missing = missing_tokens(text, tokens_for("hardware_reader"))
    ok = path.is_file() and not missing
    return ok, {"path": str(path), "exists": path.is_file(), "missing_tokens": missing}


def check_e18(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    rel = PROJECT["component_relpaths"]["metrics_config_dir"]
    metrics_dir = _PROFILE_ROOT / rel
    declared = c.component_path("metrics_config")
    if is_file(declared):
        files = [declared]
    else:
        files = sorted(str(p) for p in metrics_dir.glob("*")) if metrics_dir.is_dir() else []
    rate = str(as_int(PROJECT["hw_sample_rate_hz"]))
    hits = [p for p in files if rate in read_text(Path(p))]
    contract_hz = ckey(c.contract, "hw_sample_rate_hz")
    # 采样率同时要在"指标配置"里声明、并与 contract 一致 —— 只有一边对不算。
    ok = bool(hits) and as_int(contract_hz, -1) == as_int(PROJECT["hw_sample_rate_hz"])
    return ok, {"metrics_dir": str(metrics_dir), "metrics_files": files,
                "expected_sample_rate_hz": PROJECT["hw_sample_rate_hz"],
                "files_declaring_rate": hits, "contract_sample_rate_hz": contract_hz}


def check_e19(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    path, text = c.component("hardware_metrics_reader")
    missing = missing_tokens(text, tokens_for("hardware_metrics_reader"))
    ok = path.is_file() and not missing
    return ok, {"path": str(path), "exists": path.is_file(), "missing_tokens": missing}


def check_e20(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    path, text = c.component("sw_hw_aligner")
    low = text.lower()
    has_scale1 = any(tok in low for tok in ("scale == 1", "scale=1", "scale = 1", "shift-only"))
    has_residual = "residual" in low
    has_anchor = any(tok in low for tok in ("rank_release", "entry_rendezvous", "release"))
    contract_scale = dig(c.contract, "alignment.scale")
    scale_ok = (contract_scale is None) or as_int(contract_scale, -1) == 1
    ok = path.is_file() and has_scale1 and has_residual and has_anchor and scale_ok
    return ok, {"path": str(path), "exists": path.is_file(), "scale_is_one": has_scale1,
                "has_residual": has_residual, "uses_entry_release_anchor": has_anchor,
                "contract_alignment_scale": contract_scale}


def check_e21(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    path, text = c.component("perfetto_emitter")
    missing = missing_tokens(text, tokens_for("perfetto_emitter"))
    ok = path.is_file() and not missing
    return ok, {"path": str(path), "exists": path.is_file(), "missing_tokens": missing}


def check_e22(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    path, text = c.component("perfetto_emitter")
    low = text.lower()
    missing = [tok for tok in tokens_for("perfetto_emitter_oversize") if tok not in low]
    cap = as_int(PROJECT["max_trace_bytes"])
    has_cap = (str(cap) in text) or ("max_bytes" in low) or ("10 * 1024 * 1024" in text)
    contract_cap = ckey(c.contract, "max_trace_bytes")
    policy = str(ckey(c.contract, "oversize_policy") or "").lower()
    ok = bool(path.is_file() and not missing and has_cap
              and as_int(contract_cap, -1) == cap and "phase" in policy)
    return ok, {"path": str(path), "missing_oversize_tokens": missing,
                "has_size_cap": has_cap, "expected_max_bytes": cap,
                "contract_max_bytes": contract_cap,
                "oversize_policy_mentions_phase": "phase" in policy,
                "hint": ("emitter 必须实现「超限则隐藏所有 phase 后重生成」（保留全部 task/"
                         "选中 SM/HW 曲线）；骨架 emit_perfetto.py 目前只做到「超限拒写」，"
                         "移植时请补上 hide_all_phases 重生成分支")}


def check_e23(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    path, text = c.component("correctness_validator")
    missing = missing_tokens(text, tokens_for("correctness_validator"))
    off_name = PROJECT["compile_level_names"]["off"]
    has_ref = (off_name in text) or ("reference" in text)
    scope = first(c.contract, ("validation.correctness.check_scope",
                               "validation.correctness.scope"))
    sample_allowed = dig(c.contract, "validation.correctness.sample_only_check_allowed")
    # 以前：scope_ok = (scope is None) or ("all" in str(scope)) —— 契约没声明比对范围
    # 就直接放过，等于"没说要比全部元素"也能过；sample_allowed 的 `is not True` 同理，
    # 键缺失时也算过。缺声明本身就是缺口：必须显式声明全元素比对 + 禁止抽样。
    scope_ok = scope is not None and "all" in str(scope)
    sample_ok = sample_allowed is False
    ok = bool(path.is_file() and not missing and has_ref and scope_ok and sample_ok)
    return ok, {"path": str(path), "exists": path.is_file(), "missing_tokens": missing,
                "compares_against_reference": has_ref,
                "contract_check_scope": scope, "check_scope_ok": scope_ok,
                "contract_sample_only_allowed": sample_allowed,
                "sample_only_explicitly_forbidden": sample_ok,
                "hint": ("contract 必须显式声明 validation.correctness.check_scope"
                         "（要含 'all'，如 all_output_elements_all_ranks）"
                         "和 validation.correctness.sample_only_check_allowed=false。"
                         "两个键**缺失不再等于通过**：没声明就说明没人承诺比全部元素。")}


def check_e24(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    path, text = c.component("overhead_validator")
    names = [PROJECT["compile_level_names"][role] for role in ("off", "entry_only", "full")]
    missing = missing_tokens(text, names)
    maximum = first(c.contract, ("validation.overhead.maximum", "validation.overhead.max"))
    ok = path.is_file() and not missing and maximum is not None
    return ok, {"path": str(path), "exists": path.is_file(),
                "expected_level_names": names, "missing_tokens": missing,
                "contract_overhead_maximum": maximum}


SELFTEST_ASSERTION_TOKENS = ("assert", "_require(", "require(", "check(", "raise ")
SELFTEST_FAILURE_TOKENS = ("sys.exit", "systemexit", "selftest=fail", "return 1", "exit(1")


def selftest_substance(path: Path) -> Dict[str, Any]:
    """一个离线自测文件"有没有实质断言"的证据 —— **只看内容，不看文件名**。

    E25 以前把 ``p.name`` 也拼进用于判 aspect 的 blob，于是 4 个同名空文件
    （名字里带 decode / hw_merge / perfetto / coverage）就能把四个 aspect 全点亮。
    文件名是随便起的字符串，不是证据；这里返回的每一项都取自文件内容。
    """
    text = read_text(path)
    low = text.lower()
    # 归一化：去掉行内注释与空白后再算 sha256，这样"只改了注释的副本"也认得出来。
    code_lines = []
    for line in low.splitlines():
        stripped = line.split("#")[0].strip()
        if stripped:
            code_lines.append(stripped)
    normalized = "\n".join(code_lines)
    assertion_sites = sum(low.count(token) for token in SELFTEST_ASSERTION_TOKENS)
    failure_tokens = [token for token in SELFTEST_FAILURE_TOKENS if token in low]
    return {
        "name": path.name,
        "bytes": len(text.encode("utf-8", "replace")),
        "code_lines": len(code_lines),
        "normalized_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "assertion_sites": assertion_sites,
        "failure_path_tokens": failure_tokens,
        "has_failure_path": bool(failure_tokens),
        "content_lower": low,
    }


def check_e25(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    selftest_dir = _PROFILE_ROOT / PROJECT["component_relpaths"]["selftest_dir"]
    files = sorted(selftest_dir.glob("selftest_*.py")) if selftest_dir.is_dir() else []
    reports = [selftest_substance(path) for path in files]
    # aspect 只按**内容**判定（以前是 name + 内容，导致空文件靠名字就能点亮 aspect）。
    blob = "\n".join(report["content_lower"] for report in reports)
    aspects = {
        "decoder": "decode" in blob,
        "hw_merge": any(tok in blob for tok in ("merge", "hw_metrics", "align")),
        "perfetto": "perfetto" in blob,
        "coverage": "coverage" in blob,
        "real_import": any(tok in blob for tok in ("importlib", "emit_perfetto",
                                                   "trace_processor")),
    }
    min_bytes = as_int(PROJECT["selftest_min_bytes"])
    min_sites = as_int(PROJECT["selftest_min_assertion_sites"])
    too_small = [r["name"] for r in reports if r["bytes"] < min_bytes]
    too_few_assertions = [r["name"] for r in reports if r["assertion_sites"] < min_sites]
    no_failure_path = [r["name"] for r in reports if not r["has_failure_path"]]
    # 互为副本：去注释归一化后内容相同（包括"全都是空文件"这种极端情况）。
    by_hash: Dict[str, List[str]] = {}
    for report in reports:
        by_hash.setdefault(report["normalized_sha256"], []).append(report["name"])
    duplicate_groups = [names for names in by_hash.values() if len(names) > 1]
    # 顺带接上以前的死配置 component_relpaths["offline_selftests"]：聚合器必须存在，
    # 而且真的会用 selftest_* 去发现这个目录 —— 否则这些自测文件是没人跑的孤儿。
    aggregator_path, aggregator_text = c.component("offline_selftests")
    aggregator_ok = aggregator_path.is_file() and ("selftest_" in aggregator_text) \
        and ("glob" in aggregator_text)
    substance_ok = bool(reports) and not too_small and not too_few_assertions \
        and not no_failure_path and not duplicate_groups
    ok = bool(files) and all(aspects.values()) and substance_ok and aggregator_ok
    return ok, {"selftest_dir": str(selftest_dir),
                "selftest_files": [r["name"] for r in reports],
                "aspect_coverage": aspects,
                "aspect_source": "file content only (filenames are NOT evidence)",
                "min_bytes": min_bytes, "min_assertion_sites": min_sites,
                "per_file": [{k: v for k, v in r.items() if k != "content_lower"}
                             for r in reports],
                "files_below_min_bytes": too_small,
                "files_below_min_assertion_sites": too_few_assertions,
                "files_without_failure_path": no_failure_path,
                "duplicate_content_groups": duplicate_groups,
                "substance_ok": substance_ok,
                "aggregator_path": str(aggregator_path),
                "aggregator_discovers_selftests": aggregator_ok,
                "hint": ("离线工具链自测不是「有几个文件」就算存在：每个文件必须有实质断言"
                         "（>= {} 个断言点、>= {} 字节、有失败路径），彼此内容不能相同，"
                         "且 aspect 只从文件内容判定 —— 文件名叫 selftest_decode.py 但内容"
                         "是空的，一律 FAIL。聚合器 run_offline_selftests.py 也必须存在并"
                         "真的用 selftest_* 发现本目录，否则这些自测没人跑。"
                         .format(min_sites, min_bytes))}


# ----------------------------------------------------------------------
# E26 / E27 : 统一门禁入口 + 收尾器（skill 侧 + 项目适配器）
# ----------------------------------------------------------------------
def check_e26(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    gate_enforcer = c.component_path("gate_enforcer")
    preflight_adapter = c.component_path("preflight_adapter") or str(_THIS)
    gate_adapter = c.component_path("gate_adapter") or str(_VALIDATION / "check_gate_item.py")
    enforcer_text = read_text(Path(gate_enforcer)) if is_file(gate_enforcer) else ""
    runs_gates = ("G00" in enforcer_text) or ("gate" in enforcer_text.lower())
    pre_text = read_text(Path(preflight_adapter)) if is_file(preflight_adapter) else ""
    gate_text = read_text(Path(gate_adapter)) if is_file(gate_adapter) else ""
    # 用碎片拼出"注入型子进程开关"这个待查 token，好让本探测器自己的源码
    # （check_preflight_item.py 本身就是 preflight adapter）永远不含这个连续串 ——
    # 否则 E26 会把自己判成"非 argv-only"。
    shell_needle = "shell" + "=" + "True"
    pre_argv_only = ("argparse" in pre_text) and (shell_needle not in pre_text)
    gate_argv_only = ("argparse" in gate_text) and (shell_needle not in gate_text)
    ok = bool(is_file(gate_enforcer) and runs_gates and is_file(preflight_adapter)
              and is_file(gate_adapter) and pre_argv_only and gate_argv_only)
    return ok, {"gate_enforcer": gate_enforcer, "gate_enforcer_runs_gates": runs_gates,
                "preflight_adapter": preflight_adapter,
                "preflight_adapter_argv_only": pre_argv_only,
                "gate_adapter": gate_adapter, "gate_adapter_argv_only": gate_argv_only}


def check_e27(c: Ctx) -> Tuple[bool, Dict[str, Any]]:
    finalizer = c.component_path("finalizer")
    text = read_text(Path(finalizer)) if is_file(finalizer) else ""
    references = all(tok in text for tok in ("PREFLIGHT", "receipt")) if text else False
    ok = is_file(finalizer) and references
    return ok, {"finalizer": finalizer, "exists": is_file(finalizer),
                "references_preflight_and_receipt": references}


CHECKS = {
    "E01": check_e01, "E02": check_e02, "E03": check_e03, "E04": check_e04,
    "E05": check_e05, "E06": check_e06, "E07": check_e07, "E08": check_e08,
    "E09": check_e09, "E10": check_e10, "E11": check_e11, "E12": check_e12,
    "E13": check_e13, "E14": check_e14, "E15": check_e15, "E16": check_e16,
    "E17": check_e17, "E18": check_e18, "E19": check_e19, "E20": check_e20,
    "E21": check_e21, "E22": check_e22, "E23": check_e23, "E24": check_e24,
    "E25": check_e25, "E26": check_e26, "E27": check_e27,
}


def write_evidence(out_path: Path, payload: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
                        encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Project preflight verdict adapter (one E-check per invocation).")
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--check-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    check_id = str(args.check_id).strip()
    out_path = Path(args.out).resolve()
    contract_path = Path(args.contract).resolve()

    base = {"schema": SCHEMA, "check_id": check_id,
            "hw_optional": check_id in HW_OPTIONAL_IDS,
            "contract_path": str(contract_path), "generated_at_utc": utc_now()}

    if check_id not in CHECKS:
        base.update({"status": "ERROR", "ok": False,
                     "errors": ["unknown check id {!r}; expected one of {}".format(
                         check_id, ",".join(ALL_CHECK_IDS))]})
        write_evidence(out_path, base)
        print("CHECK={} STATUS=ERROR (unknown id)".format(check_id))
        return 2

    try:
        contract = load_json(contract_path)
    except (OSError, ValueError) as error:
        base.update({"status": "FAIL", "ok": False,
                     "errors": ["cannot read contract: {}".format(error)]})
        write_evidence(out_path, base)
        print("CHECK={} STATUS=FAIL (contract unreadable)".format(check_id))
        return 1

    try:
        ctx = Ctx(contract, contract_path)
        ok, findings = CHECKS[check_id](ctx)
        errors: List[str] = []
    except Exception as error:  # noqa: BLE001 -- 任何失败都要诚实暴露
        ok, findings = False, {}
        errors = ["{}: {}".format(type(error).__name__, error),
                  traceback.format_exc().strip()]

    base.update({"status": "PASS" if ok else "FAIL", "ok": bool(ok),
                 "findings": findings, "errors": errors})
    write_evidence(out_path, base)
    print("CHECK={} STATUS={} out={}".format(check_id, base["status"], out_path))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
