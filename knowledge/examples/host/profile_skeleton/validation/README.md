# profile/validation/ 骨架 — 接新 target 上手清单

本目录是 **项目侧 validation 适配层** 的可运行骨架。skill 侧入口是固定的
（`1_scan/validate_preflight.py` 跑 E01–E27、`4_validate/validate_all.py` 跑
G00–G24），它们按 frozen contract 里的命令模板 **argv-only、关闭 shell** 调起本目录
下的脚本；判定顺序永远是：**退出码 → 声明的证据文件存在/可读（否则 BLOCKED）→
证据 sha256 → PASS**。

> **铁律：静默返回 PASS 是绝对禁止的。**
> 这是一条 fail-closed 门链。测不了、看不懂、产物还没生成 —— 都必须如实 FAIL 或
> `raise NotImplementedError`。骨架里所有无法泛化的地方都用会 **主动抛异常并打印
> 参考实现位置** 的钩子占位，不要把它们改成 `return True`。
> **没有任何例外。** 早先版本在这里写过"唯一允许的『没测到也算过』是
> `wait_for_gpu_idle.py` 在 nvidia-smi 不存在时输出 `idle=true`" —— 那不是例外，
> 那是 bug：`gate_g00` 只读 `idle` 字段，于是 G00 在任何没有 nvidia-smi 的机器上
> 白送。现在测不到 GPU 状态 = `idle=false` + 退出码 3 + G00 FAIL（见第 3 节）。

参考实现（dsv4pro/decode，已通过 25 门禁 —— 这是**范本路径**，不是本项目路径；
只读，不要改，也不要把它的配置值照抄进你的 `PROJECT`）：
`DSA-learn/mk/h20/cutedsl/deepseek_v4_pro/decode/profile/validation/`
—— 该目录完整过了 E01–E27 与 G00–G25（其中 G00–G24 由本目录的适配器判，G25 由
skill 侧 finalizer 写；适配器对 G25 返回退出码 2 是正确行为），是语义真相来源。

---

## 0. 文件清单：谁调用 / 能否照抄 / 必改配置区

| # | 文件 | 谁调用（contract 键） | 照抄程度 | 必改配置区 |
|---|------|----------------------|----------|-----------|
| 1 | `check_preflight_item.py` | `execution.preflight.check_command_template` / `components.preflight_adapter` | **必改** | `PROJECT` + `CONTRACT_KEYS`（L58–239） |
| 2 | `check_gate_item.py` | `execution.gate_command_template` / `components.gate_adapter` | **必改** | `GATE_PROJECT`（L53–151） |
| 3 | `run_offline_selftests.py` | `execution.stages[0]` / `components.offline_selftests` | **整份照抄** | `SELFTEST_GLOB` / `DEFAULT_TIMEOUT_SEC`（L40–46） |
| 4 | `validate_correctness.py` | `execution.stages[1]` / `components.correctness_validator` | **必改 + 3 个钩子** | `PROJECT_CORRECTNESS`（L54–81） |
| 5 | `validate_overhead.py` | `execution.stages[2]` / `components.overhead_validator` | **必改 + 3 个钩子** | `PROJECT_OVERHEAD`（L68–92） |
| 6 | `wait_for_gpu_idle.py` | `execution.gpu_idle_command` / `components.gpu_idle_waiter` | **近乎照抄** | `DEFAULT_POLICY` / `GPU_IDLE_SNAPSHOT_RELPATH` / `PROBE_*`（L62–88） |
| 7 | `selftest/selftest_coverage.py` | `run_offline_selftests.py` 自动发现 | **整份照抄** | `OPERATOR_TASK_IDS` / `IGNORED_IDS` / `PHASE_IDS`（L44–57） |
| 8 | `selftest/selftest_decode.py` | 同上 | 小改 | `ABI_PATH` / `TAG_FIELD_NAMES` / `SYNTH_NUM_SMS`（L59–80） |
| 9 | `selftest/selftest_hw_merge.py` | 同上 | 小改 | fixture 常量 + TPS 路径 + `COUNTER_SPECS`（L62–114）；**本文件是所有 fixture 的单一来源** |
| 10 | `selftest/selftest_perfetto.py` | 同上 | 小改 | `RUN_LAYOUT` / 三张缺口登记表（L96–175），fixture 从 #9 import |

每个文件顶部都有中文 docstring（流程位置 / 谁调用 / CLI 形状 / 接新 target 改哪里），
配置区都以 `===== 接新 target 时只需要改这里 (PROJECT CONFIG) =====` 起、
`===== 配置区结束 =====` 止。**配置区之外的逻辑是通用的，不要动。**

---

## 1. 有序上手步骤

### 步骤 1 — 拷目录
把整个 `profile_skeleton/validation/`（含 `selftest/`）拷到目标项目的
`<target>/profile/validation/`。除 `README.md` 外一个文件都别删：contract 的
`components.*` 会逐个校验存在性 + sha256。

### 步骤 2 — 填 `PROJECT["device_writer_relpaths"]`（**拷完之后的第一件事**）
`check_preflight_item.py` 配置区第 3 项，填你项目里 **device 侧真正写记录的源文件**
（`%smid` 过滤 / `rec_begin` / `rec_end` / 容量与溢出都在里面）。骨架给的三条
（`device_record/device_profiler.cuh` / `warp_record.py` / `presets/megakernel_instr.py`）
是**示例路径、实际不存在**。

为什么排在 ABI 之前：**E01 / E05 / E08 / E09 四条检查全靠
`Ctx.device_writer_text()` 判 device 侧语义**。一条都定位不到时它返回空字符串，
这四条就变成"对着空字符串查 token"，结论毫无意义。所以现在它们会带着
`device writer 源文件缺失，请在 PROJECT.device_writer_relpaths 里填本项目的实际路径`
直接 FAIL（`device_writer_missing()`），证据里还会列出所有试过的候选路径。
也可以改成从 contract `components.device_writer` 或 ABI `source_file` 声明 —— 三条
路线任选其一即可，但**必须至少有一条能定位到真实文件**。

### 步骤 3 — 先让 ABI 成为真相来源（**最重要**）
骨架刻意 **不硬编码任何 event 编号 / task 名 / track 号**，全部从
`device_record/abi/record_abi.<target>.json` 现读现推。所以先把 `extract_abi.py`
产物补齐，再改 validation：

必须存在的字段（`check_preflight_item.PROJECT["abi_keys"]` 列了候选名，按你的产物补别名）：

- `clock`（intrinsic + `resolution_ns`）
- 记录布局：`record_fields` / `regions` / `elems_per_record` / `header_elems_expr`
- `tag_bitfields`（含 `PROJECT["lane_identity_tag_fields"]` 里的身份字段 +
  `tag_event_id_field_candidates` 里的 task/event id 字段 + `tag_is_start_field`）
- `tasks[]`：每条要有 `task_id` / `name` / `layer ∈ {task, phase, ignored}` /
  `parent_task_id`
- `tracks[]`：每条一个记录 warp（`observer_topology()` 从这里推 lane 拓扑）
- `sync_anchors.entry_rendezvous`（软件零点；覆盖率与对齐都依赖它）
- 有则写：`capture_scope`、`compile_switch.levels`、`max_records_per_lane`、
  `empty_slot_sentinel`、`buffer.slot_formula`

**为什么必须从 ABI 推、不能硬编码**：`selftest/selftest_perfetto.py` 与
`selftest_hw_merge.py` 的期望值（`ph=X` 条数、task/phase/ignored 划分、
`allowed_task_ids`、observer 车道数）全部由 `abi_task_index()` / `abi_layer_ids()` /
`abi_track_ids()` / `abi_anchor()` 现算。如果把 event 号写死，**改了 ABI 之后自测
仍然会绿**，等于自测失效 —— 这条设计务必保留。

### 步骤 4 — 改 `check_preflight_item.py` 的 `PROJECT`
| 键 | 改什么 |
|---|---|
| `abi_filename` | 你的 ABI 文件名（骨架是 `record_abi.example.json`） |
| `compile_level_names` | 三个插桩编译档位名：`off` / `entry_only` / `full` |
| `compile_level_files` | 三个档位对应的 preset json 路径 |
| `device_writer_relpaths` | device 侧写记录的源文件（E01/E05/E08/E09 的判据源）—— **见步骤 2，这是必须最先改的一项** |
| `component_relpaths` | 12 个组件相对路径（host_harness / contract_materializer / decoder / hardware_reader / hardware_metrics_reader / sw_hw_aligner / perfetto_emitter / metrics_config_dir / correctness_validator / overhead_validator / offline_selftests / selftest_dir） |
| `expected_grid` / `expected_threads_per_cta` / `expected_physical_warps_per_cta` / `expected_observer_lane_count` / `expected_max_records_per_lane` | **`None` 不再等于豁免**。E14/E15 现在要求同一个期望值有**两个独立来源**互相印证（声明侧：contract `build_and_launch` / `topology`、ABI；运行侧：launch 描述符、`threads_per_cta//32`）。只有一个来源 → `reason=single_source` → FAIL；一个都没有 → `reason=missing` → FAIL。补法三选一：contract 里声明、这里填死、或在 `run_profile.py` 实现 `build_runtime_launch()` 让描述符成为独立来源 |
| `selftest_min_bytes` / `selftest_min_assertion_sites` | E25 判「离线自测有实质内容」的下限（默认 2048 字节 / 5 个断言点）。骨架四个自测里最小的是 8771 字节 / 11 个断言点，默认值留了约 4x 余量。**调低这两个数就是把「4 个空文件也算自测存在」的假绿灯放回来** |
| `overflow_tensor_name` | 有独立溢出张量就填，没有填 `None` |
| `pairing_kinds` | `begin`/`end` 的取值（真实项目 1/0，骨架示例 1/2） |
| `hw_sample_rate_hz` / `max_trace_bytes` | HW 采样率门禁、trace 体积上限（10485760） |
| `required_tokens` | 11 组「源文件里必须出现的字符串」，按你的实现改（含 `perfetto_emitter_oversize: ("hide_all_phases",)`）。E05 = `("smid()", "prof_sel", "capture_slot")`、E09 = `("rec_begin", "rec_end", "prof_level")`，是参考实现里的严格集合：**可以改名字，但不要缩成一个词**（单 token 子串匹配连注释都算命中，等于没查）。把某组清空会 `raise NotImplementedError` 判 FAIL，**不能靠删清单放行** |
| `abi_keys` | 7 个逻辑名 → 你的 ABI 字段候选路径 |
| `lane_identity_tag_fields` / `tag_event_id_field_candidates` / `tag_is_start_field` / `anchor_semantics` | tag 字段名映射 |
| `CONTRACT_KEYS` | ~27 个逻辑名 → contract 里的候选点号路径。骨架同时兼容 frozen `CONTRACT.lock.json`、`materialize_contract.py` 产物、真实项目三种形状；**新形状只在这里加候选，别去改下面的检查逻辑** |

### 步骤 5 — 改 `check_gate_item.py` 的 `GATE_PROJECT`
| 键 | 改什么 |
|---|---|
| `run_root_fallback_relpath` | contract 里没有 run_root 时的兜底 |
| `artifact_relpaths` | **最常改的一组**：`gpu_idle` / `offline_selftest` / `correctness` / `overhead` / `sw_manifest` / `sw_events` / `sw_sidecar` / `hw_manifest` / `hw_metrics` / `alignment` / `candidate`。目录布局不同（如 `alignment.json` 放 `hw/` 而非 `work/`）只改这里，所有门自动跟着走 |
| `anomaly_counter_paths` | 解码 metadata 的异常计数键名（`unpaired_records` / `unknown_events` / `negative_durations`，**求和语义**：一处都没命中 → 门 FAIL） |
| `scalar_paths` | 7 项首命中标量（`events_total` / `raw_start_count` / `raw_end_count` / `records_total` / `records_expected` / `decode_valid` / `timestamps_monotonic`） |
| `capacity_paths` | 容量四项（`records_per_lane` / `overflow_flag` / `truncated` / `max_records_per_lane`） |
| `sm_scope_paths` | SM 范围五项（含 `all_mode_tokens`） |
| `coverage_max_gap_percent` / `coverage_max_gap_floor_ns` | G19 覆盖率阈值：最大未覆盖空洞 ≤ `max(5% * span, 50_000 ns)`，且仅在 `scale == 1` 时判 |
| `hw_manifest_pass_verdicts` | nsys manifest 里算「过」的判定词 |
| `gpu_idle_trusted_probes` | G00 允许信任的 `gpu_idle.json` `probe` 取值**白名单**，必须包含 `wait_for_gpu_idle.PROBE_MEASURED`。认不出的 probe（含没写 probe 的旧快照、`nvidia-smi_absent`）→ G00 FAIL，**即使快照写着 `idle=true`** |

`hw_sample_rate_hz` / `max_trace_bytes` 直接从 `cp.PROJECT` 转出，保持单一来源；
能力门 `G12`/`G14` 直接委托 `cp.check_e12` / `cp.check_e15`，不要复制一份。

### 步骤 6 — 改 `validate_correctness.py`
配置区 `PROJECT_CORRECTNESS`：`abi_relpath` / `level_names` /
`default_atol=0.01` / `default_rtol=0.01` / `default_seed=20240917` /
`default_input_generator="randn"` / `required_output_tensors` /
`compare_against_golden`。

比对语义是通用的、**不要改**：scope 固定 `all_output_elements_all_ranks`，
`sample_only` 不允许为真；只在一边出现的张量 = 硬错误；一个张量都没比到 = FAIL；
不一致时退出码 4。

### 步骤 7 — 改 `validate_overhead.py`
配置区 `PROJECT_OVERHEAD`：`abi_relpath` / `level_names` /
`default_maximum=0.10` / `default_warmup_iterations=20` /
`default_measured_iterations=100`（`CONFIG_LEVEL_NAMES` 由 `level_names` 自动生成）。

公式与计时协议是通用的、**不要改**：
`(median_full - median_entry_only) / median_entry_only <= maximum`，
CUDA event 计时，取中位数，**buffer/counter 清零必须在计时区之外**。

### 步骤 8 — 核对 `wait_for_gpu_idle.py`
`DEFAULT_POLICY`（`utilization_max_percent=10` / `memory_max_mib` /
`poll_interval_sec=5` / `timeout_sec=1800`）、
`GPU_IDLE_SNAPSHOT_RELPATH="validation/gpu_idle.json"`
（**必须与 `GATE_PROJECT["artifact_relpaths"]["gpu_idle"]` 一致**），
`PROBE_MEASURED` / `PROBE_UNOBSERVABLE`
（**`PROBE_MEASURED` 必须出现在 `GATE_PROJECT["gpu_idle_trusted_probes"]` 白名单里**，
否则 G00 会一直 FAIL），以及 `RUN_ROOT_KEYS` / `ALLOWED_GPU_KEYS` 的候选路径。
其余照抄。

退出码：`0` 空闲 / `1` 超时仍忙 / `2` 出错 / `3` **GPU 状态不可观测**（没有
nvidia-smi）。3 也会让 `run_measurement.py` 的 GPU 阶段 FAIL —— 纯 CPU 机器上
GPU 阶段本来就不该判过。

### 步骤 9 — 核对 `selftest/`
- `selftest_coverage.py`：把 `OPERATOR_TASK_IDS` / `IGNORED_IDS` / `PHASE_IDS`
  改成你 ABI 里的实际取值（这个自测验的是覆盖率算法本身，用合成区间）。
- `selftest_decode.py`：`ABI_PATH`、`TAG_FIELD_NAMES` / `METADATA_FIELD_NAMES` /
  `BLOCK_MAP_FIELD_NAMES`（字段改名只改这里）、`SYNTH_NUM_SMS`。
- `selftest_hw_merge.py`：**fixture 单一来源**。改 `ABI_PATH`、
  `TRACE_PROCESSOR_SHELL_CANDIDATES`（或设 `TRACE_PROCESSOR_SHELL` 环境变量）、
  `HW_DEVICE_NAME` / `TARGET_KERNEL_NAME`（两个 `PROJECT_HOOK_REPLACE` 占位）、
  `COUNTER_SPECS`（曲线名 + 单位）、`HW_FREQUENCY_HZ`（须与门禁采样率一致）。
  所有期望值（`EXPECTED_SW_WINDOW` / `EXPECTED_SHIFT_NS` / `EXPECTED_COUNTER_EVENTS`
  …）都是从这些常量 **推导** 出来的 —— 改常量不用改断言。
- `selftest_perfetto.py`：改 `RUN_LAYOUT`（产物相对路径约定）、
  `PHYSICAL_WARPS_PER_CTA` + `OBSERVER_REPRESENTATION`（Hopper/Blackwell 有物理
  warp 到 lane 的映射时填）、`SKILL_ROOT_CANDIDATES`（或设
  `FAST_INSTRUMENT_SKILL_ROOT`）、以及两个「已知缺口登记表」（见第 3 节）。

### 步骤 10 — 跑验证（见第 4 节）
验收标准：4 个自测全部退出码 0，`selftest_perfetto.py` 打出
`SELFTEST=PASS ... todos=0`。

**刚拷过来的骨架达不到这个标准，这是设计如此**：出图器侧还有已登记缺口（第 3 节），
`selftest_perfetto.py` 会退出 3 并打 `SELFTEST=PASS_WITH_TODOS`，把相关 case 标成
`UNVERIFIED`；`run_offline_selftests.py` 把任何非 0 子进程退出码算作 FAIL，所以
`OFFLINE_SELFTESTS=FAIL`。要变绿必须真的把缺口补掉、再把 tag 从登记表里删掉 ——
**不允许为了变绿去放宽判据、注释断言、或把缺口 tag 加进登记表当挡箭牌**。

---

## 2. 9 个 `NotImplementedError` 钩子：必须实现什么

这些是「只有你的项目知道怎么做」的部分。**它们会抛异常并打印真实实现的文件:行号。**
（行号按当前骨架实测；改过文件后用 `grep -n "raise NotImplementedError" validation/*.py`
重新确认。之前这一节写的是"8 个"，漏了 `validate_correctness.py` 的 golden 比对钩子。）

| 文件:行 | 钩子 | 必须返回 / 做什么 | 也可以这样绕开 |
|---|---|---|---|
| `check_preflight_item.py:513` | `device_constants(c, required)` | `WARPS_PER_CTA` / `PHYSICAL_SM_COUNT` / `NUM_CAPTURE_LANES` / `SELECTED_PHYSICAL_SMIDS` + 三个档位常量 | 在 `run_profile.py` 实现 `device_constants() -> dict`；**或**让 `extract_abi.py` 把 `capture_scope` + `compile_switch` 写进 ABI |
| `check_preflight_item.py:564` | `runtime_launch(c)` | `{grid, threads_per_cta, rank, device_source, device_source_combined_sha256, observers}` | contract 里声明 `build_and_launch.grid` / `.block`；**或**实现 `run_profile.build_runtime_launch(version_root, abi_path, rank, world_size)` |
| `check_preflight_item.py:614` | `observer_topology(c, threads_per_cta)` | `{physical_warps_per_cta, observer_lane_count, lanes:[{observer_warp, represents}]}` | ABI 写出 `tracks[]`；**或**实现 `run_profile.observer_topology(abi, threads_per_cta)` |
| `validate_correctness.py:171` | `setup_gpu_imports()` | `{'torch': torch, ...}`；**import 必须写在函数体内**，保证 CPU-only 主机上 `--help` / `--dry-run` 仍可用 | 无 |
| `validate_correctness.py:187` | `launch_at_level(mods, level_name, info)` | `{输出张量名: tensor}`；同 seed 同输入两档位必须完全一致，跑完必须 `synchronize()` | 无 |
| `validate_correctness.py:265` | `mods['harness'].check_all` | `compare_against_golden=True` 时的 golden 比对入口，返回 `{all_passed, ...}` | 在 `setup_gpu_imports()` 里提供 `harness.check_all`；**或**把 `PROJECT_CORRECTNESS["compare_against_golden"]` 关掉（关掉就等于不比 golden，**要想清楚**） |
| `validate_overhead.py:183` | `setup_gpu_imports()` | 同上 | 无 |
| `validate_overhead.py:197` | `compile_at_level(mods, level_name, info)` | `{"launch": callable, "state": ...}` | 无 |
| `validate_overhead.py:212` | `reset_launch_state(mods, compiled)` | 清零 profiler buffer / counter / 溢出标志；**只能在计时区之外被调用** | 无 |

另有一处**不是钩子**、但同样会抛 `NotImplementedError` 的兜底：
`check_preflight_item.py:351` 的 `tokens_for()` —— `PROJECT["required_tokens"]` 里某组被
清空时直接判 FAIL（空 token 集合 = 什么都不检查 = 静默通过）。它不需要你去"实现"，
它是在拦"靠删配置放行"这条路。

`check_preflight_item.py` 的三个钩子设计成「先尽最大努力从 ABI + contract +
`run_profile.py` 里推，推不出来才抛」。所以多数 target 只要 ABI/contract 补齐，
一个钩子都不用写。`validate_*` 的 6 个 GPU 钩子必须逐个实现 —— 它们要真的编译、
真的发射 kernel，无法泛化。

---

## 3. 已知移植缺口（骨架出图器侧，自测会打 `[todo]`）

`selftest_perfetto.py` 现在打 **8 条 `[todo]`**，退出码 **3**
（`SELFTEST=PASS_WITH_TODOS`），并把命中缺口的 case 标成 `UNVERIFIED`。
这些不是自测的 bug，而是 **骨架 `hw_perfetto/emit_perfetto.py` 相对参考实现还缺的
功能**，移植时要补：

1. **`otherData.layout_contract`（`perfetto-layout-contract/v1`）没写** ——
   要把布局契约写进 `otherData.layout_contract` + 首条 `ph="M"` /
   `name="perfetto_layout_contract"` 事件（照 skill `1_scan/layout_contract.py`
   的 `metadata_event()`）。骨架自测用 `best_effort_layout_contract()` 按出图器
   **实际布局** 现搭一份契约，先把 skill 的 V1..V6 跑起来。
2. **`otherData.provenance`（`emit-provenance/v1`）没写** —— contract / events /
   abi / hw 的路径 + sha256。
3. **`args.is_derived` 缺失** —— skill 的 V4 要求 task 区间显式带
   `is_derived=false`（实测区间）或 `true` + `aggregation`（派生父区间）。
   这一条登记在 `KNOWN_LAYOUT_GAPS = ("task_source_measurement_missing",)` 里，
   **补完出图器后把它从元组里删掉，这个检查就自动升级成硬门禁**。
4. **`write_trace_gz` 是「先写后量」** —— 超限时会留下残留 `.gz`。改成写临时文件 +
   `os.replace()`，或在抛 `EmitError` 前 `unlink()`。自测当前硬断言
   `rc == 4` + 残留文件 **明显超限**（所以不可能被误当合法交付物）。
5. **超限只做到「拒写」，缺 `hide_all_phases` 重生成分支** —— 要保留全部 task 区间 /
   选中 SM 的 CTA 泳道 / HW 曲线，并写 `otherData.semantic_viewer.mode="hide_all_phases"`。
   **预检 E22 会一直 FAIL 直到补上。**
6. **布局契约里 `allow_unparented_phase` / `allow_empty_warp_lanes` 被置为 `true`
   —— 此处放宽了这两项检查**（孤儿 phase 区间、空 warp 泳道不会被拦）。原因是
   skill 的 `layout_contract.validate_contract()` 硬要求这两项为 `true`（改成
   `false` 会 `SystemExit`），骨架无法收紧。**真实项目必须收紧**：出图器保证每个
   phase 都挂在 task 下、每条泳道都有区间之后，去 skill 侧放开这条 schema 约束。
   登记 tag：`layout_contract_relaxed_by_skill_schema`。

三张登记表都在 `selftest_perfetto.py` 配置区：`KNOWN_LAYOUT_GAPS`（V 系列语义缺口）、
`KNOWN_EMITTER_GAPS`（出图器行为缺口，含上面第 4、5、6 条）、
`OPTIONAL_DELIVERY_KEYS`（第 1、2 条）。**补完就从表里删掉，检查自动升级成硬门禁；
反过来，冒出一个没登记的 tag 会直接 FAIL** —— 不许悄悄跳过。

另有两处解码侧可选增强：`abi_loader.lane_geometry`（骨架无，preflight 会退回从
`tracks[]` 推）、`decode_profile.run_self_test`（骨架无，preflight 会退回跑
`selftest_decode.py` 子进程）。

**本轮没能一起修掉的（超出写入范围，下次务必带上）**：

1. **真实实现 `dsv4pro/decode` 的 E25 仍有此缺陷，下次触碰该 target 时需同步收紧** ——
   它的 `check_preflight_item.py:494`（`check_e25`）是
   `blob = "\n".join(p.name + "\n" + read_text(p) for p in files).lower()`，
   **把文件名拌进了 aspect 匹配串**，四个 aspect 全靠子串匹配，没有任何内容下限，
   所以「4 个同名空文件」在那份实现里照样 PASS。骨架这份已改成只看内容
   （见 3.1 第一行），把改动照搬过去即可：`selftest_substance()` + 三道内容闸
   （`selftest_min_bytes` / `selftest_min_assertion_sites` / 失败路径）+ 归一化 sha256 查重
   + 聚合器存在性。**本轮不动它是因为该目录正被别的 agent 使用，不是因为它没问题。**
   已只读核对，**同一缺陷共 5 个 target**：`deepseek_v4_pro/{decode,prefill}`、
   `kimi_k3/prefill`、`llama/{decode,prefill}`。
   另外 3 个（`glm52/prefill`、`kimi_k3/decode`、`minimax_m3/prefill`）的 E25 是另一种写法
   —— **只查文件是否存在**（`run_offline_selftests.py` + skill 侧两个自测），
   连文件名匹配都没有，**同样一行内容都不看**，收紧时要一起带上。
2. **`check_gate_item.py` 里三处死配置仍然没人读**（本轮该文件不在写入范围内）：
   `GATE_PROJECT["artifact_relpaths"]["offline_selftest"]`（约 `:65`，没有任何门读它）、
   `anomaly_counter_paths["negative_durations"]`（约 `:90`，`anomaly(gc, "negative_durations")`
   从未被调用 —— **负时长记录目前根本没有门在查**）、
   `scalar_paths["records_expected"]`（约 `:99`，`scalar(gc, "records_expected")` 从未被调用；
   `:551` 的注释提到它，但 `:557` 只用了 `records_total` —— **"记录条数是否符合预期"这道校验实际不存在**）。
   下次改该文件时：要么接到真实用点（推荐，负时长和 records 期望值都是真问题），
   要么删掉并在此处写明为什么。**留着不读的配置会让人以为已经在查了。**
3. **`CONTRACT_KEYS["run_root"]`** 在 `check_preflight_item.py` 里定义但只有门禁侧消费，
   修正需要同时动 `check_gate_item.py`，本轮同样搁置。
4. **同类 `None` 豁免还剩两处未改**（本轮只按任务点收紧了 E14/E15/E23）：
   `check_e20` 的 `scale_ok = (contract_scale is None) or ...`、
   `check_e08` 的 `overflow_required_zero = (required_overflow is None) or ...`。
   判据形状和 E23 的 `scope is None` 完全一样：**契约不声明就等于免检**。
   下一轮按同样办法收紧（要么从 ABI/契约推真值，要么显式 FAIL 并点名该填哪个字段）。

### 3.1 本轮修掉的静默通过通道（行为变化，务必知道）

以下情形**以前会静默跳过或白送、现在会硬 FAIL**。如果你接 target 时撞上它们，
那是设计如此，请去修配置/环境，**不要把判据改回去**：

| 以前 | 现在 |
|---|---|
| `selftest_perfetto.py` 定位不到 skill 的 `1_scan/layout_contract.py` → 打一条 `[todo]`（或 V5 那处连 `[todo]` 都不打）就 `return`，V1..V6 六条断言整批不跑，case 仍计入 `passed`，`main()` 退出 0 | `SELFTEST=FAIL` + 退出码 **1**，报出试过哪些路径、怎么设 `FAST_INSTRUMENT_SKILL_ROOT` / `SKILL_ROOT_CANDIDATES`。**实测**：把 `profile/` 拷到别的目录深度（`parents[5]` 解析成 `/`）后，修前 `rc=0 / SELFTEST=PASS checks=36 todos=6`，修后 `rc=1 / case=skill_layout_verify` |
| 找不到 `trace_processor_shell` 时相关 case 只打 `[todo]` | 同样硬 FAIL（`_require_external()`），并打印用 skill `1_scan/get_trace_processor.py` 安装 + `export TRACE_PROCESSOR_SHELL=<绝对路径>` 的做法 |
| 打了 `[todo]` 的 case 仍然算 `passed`，`main()` 打 `SELFTEST=PASS` 退出 0 | case 降级为 `UNVERIFIED`，总结分开报 `verified_cases` / `unverified_cases`，打 `SELFTEST=PASS_WITH_TODOS todos=<n>` 并退出 **3**。**有 `[todo]` 就不可能是干净的 PASS** |
| `wait_for_gpu_idle.py` 在没有 nvidia-smi 时写 `idle=true` + 退出 0；`gate_g00` 只读 `idle` → **G00 在任何纯 CPU 机器上白送** | 写 `idle=false` + `probe="nvidia-smi_absent"` + 退出 **3**；`gate_g00` 改成按 `gpu_idle_trusted_probes` **白名单**判 probe。**实测**：拿旧格式快照（`idle=true, probe="nvidia-smi_absent"`）跑，旧逻辑 `True`、新逻辑 `False`；没有 `probe` 字段的旧快照也从 `True` 变 `False`；真探测到的空闲快照仍然 `True` |
| E01/E05/E08/E09 在 `device_writer_relpaths` 全部指向不存在的文件时，对着**空字符串**查 token（这四条的真实语义从未跑过一次） | 先过 `device_writer_missing()` 闸：显式 FAIL 并给出"请在 `PROJECT.device_writer_relpaths` 里填本项目的实际路径"+ 所有试过的候选路径。**实测**：骨架原样跑 E01/E05/E08/E09，四条都 `rc=1 / STATUS=FAIL` 且理由是这句话 |
| E05 的 token 集合退化成 `("smid",)`、E09 退化成 `("profil",)` —— 注释里出现 `smid` / `profiler` 就算过 | 恢复参考实现的严格集合 `("smid()", "prof_sel", "capture_slot")` / `("rec_begin", "rec_end", "prof_level")`。**实测**：一个只在注释里提 `smid` / `profiler` 的诱饵源，旧集合 `missing=[]`（PASS，假绿灯），新集合 `missing` 三项全列（FAIL） |
| `required_tokens` 里某组被清空 → `missing_tokens()` 恒返回 `[]` → 静默 PASS | `tokens_for()` 抛 `NotImplementedError`，该 check FAIL |
| `selftest_hw_merge.py` 的 `HW_DEVICE_NAME` / `TARGET_KERNEL_NAME` 还是 `PROJECT_HOOK_REPLACE` 字面量时，被当成合法 fixture 名照跑 | 真正被使用时 `require_configured()` 抛错：`SELFTEST=FAIL ... 配置区的 HW_DEVICE_NAME 仍是 PROJECT_HOOK_REPLACE 占位符`。**接新 target 必须先把占位符填掉**，否则两个自测在原地 FAIL |
| `SKILL_ROOT_CANDIDATES` 在模块导入期就求值 `_HERE.parents[5]`：`profile/` 被拷到不足 6 层深的路径下时，整个自测在 import 阶段抛裸 `IndexError: 5`，连 fail-closed 的 howto 都印不出来，看起来像"脚本坏了"而不是"这一项没配" | 走 `_relative_skill_root_guess()`：层级不够就整条候选不参与（仍然硬 FAIL，不是放过），并在 howto 里点明"本次安装路径层级不足 6 层，`parents[5]` 那条相对定位整条不可用"。**实测**：`/tmp/x/validation/selftest` 下修前 `IndexError: 5`（rc=1 但无指引），修后 `rc=1 / case=skill_layout_verify` 且带完整指引 |
| **E25 把「文件名」拌进去重 blob** —— aspect（decoder/hw_merge/perfetto/coverage/real_import）在 `文件名 + 内容` 的拼接串上做子串匹配。于是 **4 个 0 字节文件**，只要名字叫 `selftest_decode.py` / `selftest_hw_merge.py` / `selftest_perfetto.py` / `selftest_coverage.py`，五个 aspect 全部"命中"，E25 直接 PASS。**文件名是自由字符串，不是证据** | aspect 只从**文件内容**判定（`aspect_source: "file content only (filenames are NOT evidence)"`），另加三道内容闸：每文件 ≥ `selftest_min_bytes`（2048）、≥ `selftest_min_assertion_sites`（5，数 `assert` / `_require(` / `raise ` 等）、必须有失败路径（`sys.exit` / `SELFTEST=FAIL` / `exit(1)`）；并按**去注释归一化后的 sha256** 查重，互为副本 → `duplicate_content_groups` → FAIL。**实测诱饵**：4 个 0 字节文件（其中一个名为 `selftest_decode_importlib.py` 以点亮 `real_import`）旧 `PASS/rc=0`（证据里五个 aspect 全 `true`），新 `FAIL/rc=1`，四个列表全部列出这 4 个文件 + 一个 4 元素重复组；另一个诱饵「4 份互为副本的真自测」同样旧 PASS → 新 FAIL |
| **E04 的 ABI 交叉校验是 `abi_ok = True`** —— ABI 没声明编译档位时直接置真，等于这一项从不校验；判据还写成 `(three_distinct or bool(abi_levels))`，**只要 ABI 里有档位键就能免掉"三档实现真的不同"** | `abi_ok = abi_declared and 档位键名与 PROJECT['compile_level_names'] 一一对上`，并且 `ok` 要求 `all_present and names_match and distinct_ok and abi_ok`（`or` 改 `and`）。新增 `compile_levels_distinct()`：preset 比 `compile_defines`、非 JSON 文件比 sha256、共用模块要求 ABI 三档取值互不相同**且**三个档位名都在模块正文里出现。hint 明说「请在 ABI 里声明三档编译开关（`compile_switch.levels`），三档 = 关闭插桩 / 仅入口打点 / 完整事件记录」。**实测诱饵**：删掉 ABI 档位声明 → 旧 `PASS/rc=0`，新 `FAIL/rc=1`（`abi_declares_compile_levels: false`）；三档 `compile_defines` 写成完全一样 → 旧 PASS → 新 FAIL（三档相同 = overhead 恒 0、correctness 的参考档位其实开着插桩） |
| **E14 的 `expected_grid` / `expected_threads_per_cta` 默认 `None`，判据在 `None` 上通过** —— 骨架原样、以及任何没填这两项的 target，几何校验整条不跑 | 换成通用 `cross_check()`：`grid` / `threads_per_cta` 必须有**两个独立来源**（contract 声明、PROJECT 填死、runtime launch 描述符）互相印证；`reason` 明确区分 `missing` / `single_source` / `mismatch`，三者都 FAIL。描述符只有在 `run_profile.py` 真的实现了 `build_runtime_launch()` 时才算独立来源（否则 source 是 `contract+abi(generic)`，是 contract 的派生量，不能自证）；`grid_ok` 另外要求三个维度都 > 0。**实测诱饵**：契约声明 grid/block、两个 `expected_*` 保持 `None` → 旧 `PASS/rc=0`，新 `FAIL/rc=1`（`reason: "single_source"`，hint 列出三条补法） |
| **E15 的 `warps == threads // 32` 是恒真式** —— 契约不声明 `physical_warps_per_cta` 时，`observer_topology()` 自己就用 `threads//32` 兜底，再拿这个兜底值去和 `threads // 32` 比。lane 数同理拿 `len(topology.observers)` 和自己比 | warp 数走 `declared_warps_sources()`（contract / ABI / PROJECT / runtime `threads_per_cta//32`，**声明侧不再有 `threads//32` 兜底**），lane 数走 `observer_lane_count_sources()`（只在来源不是 topology 自身时才补 `len(contract.topology.observers)` / `len(abi.tracks)`），两者都过 `cross_check()`；`covers` 只用**被互证过的** warp 数当上界，未互证时直接 `False`。`observer_topology()` 的 docstring 也加了警告：返回的 `physical_warps_per_cta` 可能是兜底值，**调用方绝对不能拿它和 `threads // 32` 比**。**实测诱饵**：契约里删掉 warp 数声明 → 旧 `PASS/rc=0`，新 `FAIL/rc=1`（`reason: "single_source"`，只剩 `runtime_launch.threads_per_cta//32` 一个来源）；lane 数只由 topology 自身提供 → 旧 PASS → 新 FAIL |
| **E23 的 `scope is None` 算通过、`sample_allowed is not True` 也算通过** —— 契约不声明校验范围 / 不声明"禁止只抽样比"，正确性门禁的范围承诺整条不存在 | `scope_ok = scope is not None and "all" in str(scope)`、`sample_ok = sample_allowed is False`：两个键**缺失不再等于通过**，必须显式声明 `validation.correctness.check_scope`（含 `all`）与 `sample_only_check_allowed=false`。**实测诱饵**：契约里删掉这两个键 → 旧 `PASS/rc=0`，新 `FAIL/rc=1` |
| **三处死配置（写了没人读，让人以为在查）**：`tag_event_id_field_candidates`、`component_relpaths["offline_selftests"]`、`CONTRACT_KEYS["information_bundle"]` | 全部接到真实用点：E03 用 `tag_event_id_field_candidates` 定位 event id 位域，校验 `width >= 1` 且 `max(task_id) <= (1<<width)-1`（**位宽装不下 task id 就是解码期必错**）；E25 用 `offline_selftests` 校验聚合器存在且真的用 `selftest_*` + `glob` 发现本目录（**没有聚合器 = 这些自测没人跑**）；E11 用 `information_bundle` 要求 bundle 声明里含 `sw`。**实测诱饵**：把 event id 位域宽度改窄到装不下 task id → 旧 `PASS/rc=0`，新 `FAIL/rc=1`；删掉 `run_offline_selftests.py` → 旧 PASS → 新 FAIL；契约里删掉 bundle 声明 → 旧 PASS → 新 FAIL |

**正控（证明没有过度收紧）**：同一套 fixture 把上面每一项都按新判据填对（契约声明 grid/block、`expected_*` 填死、ABI 三档取值互不相同、8 条 tracks、topology lane 数一致、显式 `check_scope` + `sample_only_check_allowed=false`、四个真自测原样），E03/E04/E11/E14/E15/E23/E25 **新旧两侧都 `PASS/rc=0`**。

---

## 4. 验证命令（3 步，10 个文件 + 6 个适配器 + 4 个自测全过才算数）

```bash
cd <target>/profile

# 1) 语法 / 字节码（10 个文件：6 个主文件 + 4 个 selftest）
python3 -m py_compile validation/*.py validation/selftest/*.py; echo "PY_COMPILE_RC=$?"

# 2) CLI 不炸（6 个带 argparse 的适配器，退出码必须都是 0）
for f in check_preflight_item check_gate_item run_offline_selftests \
         validate_correctness validate_overhead wait_for_gpu_idle; do
  python3 -B validation/$f.py --help >/dev/null 2>&1; echo "HELP_RC $f = $?"
done

# 3) 离线自测全家桶
#    前置条件：必须先有 trace_processor_shell，否则第 3 步一定 rc=1（这是设计，不是环境问题）。
#    本机已装好的一份（写这份 README 时实测可用）：
export TRACE_PROCESSOR_SHELL=/path/to/trace_processor_shell
#    没有的话用 skill 的 1_scan/get_trace_processor.py 装一份（注意不是 .sh）。
python3 -B validation/run_offline_selftests.py --out /tmp/selftest_report.json
```

> **量退出码不要用管道。** `... | tail` 之后 `$?` 是 `tail` 的退出码，永远是 0 ——
> 上一次「4/4 PASS rc=0」的假记录就是这么来的。要么 `> /tmp/x.txt 2>&1; echo $?`，
> 要么把 `echo "RC=$?"` 紧贴在命令后面、放在任何 grep 之前。

骨架自身在本机的实测结果（**本轮 fail-closed 收紧之后重测**，Python 3.11.12）：

```
python3 -m py_compile × 10 个文件               # rc=0
--help × 6 个适配器                              # 全部 rc=0

# 场景 A：原封不动的骨架（配置区还是 PROJECT_HOOK_REPLACE），不设 TRACE_PROCESSOR_SHELL
[selftest] selftest_coverage.py  -> PASS (exit=0)
[selftest] selftest_decode.py    -> PASS (exit=0)
[selftest] selftest_hw_merge.py  -> FAIL (exit=1)   # HW_DEVICE_NAME 仍是占位符
[selftest] selftest_perfetto.py  -> FAIL (exit=1)   # 同上，case=<setup>
OFFLINE_SELFTESTS=FAIL passed=2 failed=2 count=4    # rc=1

# 场景 B：同上，但设了 TRACE_PROCESSOR_SHELL —— 结果**完全一样** rc=1。
#         占位符防护在 TPS 检查之前就拦下了：占位符 fixture 是假的，绿也没意义。

# 场景 C：占位符填完（HW_DEVICE_NAME / TARGET_KERNEL_NAME / 两条路径），TPS 找得到
[selftest] selftest_hw_merge.py  -> PASS (exit=0)
    SELFTEST=PASS name=selftest_hw_merge checks=53 cases=8, imported_slices=12
[selftest] selftest_perfetto.py  -> FAIL (exit=3)
    SELFTEST=PASS_WITH_TODOS checks=41 todos=8
      verified_cases   = cli_contract_path, trace_processor_import, partial_overlap_rejected
      unverified_cases = delivery_schema, skill_layout_verify, oversize_seam
OFFLINE_SELFTESTS=FAIL passed=3 failed=1 count=4    # rc=1

# 场景 D：占位符填完，但 trace_processor_shell 找不到（外部工具缺失）
selftest_hw_merge  -> FAIL (exit=1) reason=trace_processor_shell 可用: None (detail=...get_trace_processor.py...)
selftest_perfetto  -> FAIL (exit=1) case=trace_processor_import 外部依赖缺失
OFFLINE_SELFTESTS=FAIL passed=2 failed=2 count=4    # rc=1

# 场景 E：占位符填完，但 skill 根定位不到（装在浅路径 + SKILL_ROOT_CANDIDATES 未填）
selftest_perfetto  -> FAIL (exit=1) case=skill_layout_verify 外部依赖缺失
    howto 里会点明「本次安装路径层级不足 6 层，parents[5] 那条相对定位整条不可用」
```

**场景 C 的 `rc=3` 是正确的，不要为了变绿去放宽判据。** `PASS_WITH_TODOS`
表示 41 项检查过了、8 项已登记缺口没验、6 个 case 里 3 个被降级为 `UNVERIFIED`；
`run_offline_selftests.py` 没有「部分通过」这档，所以子进程 rc=3 在汇总里显示为
`FAIL`。要 rc=0，唯一的路是把第 3 节那 8 项缺口真的补掉、再从三张登记表里删掉。

**fail-closed 已实测有效**（不是装饰）：
`trace_processor_shell` 缺失 → rc=1（场景 D）；skill 校验器定位失败 → rc=1（场景 E，
以前是静默 rc=0 且 case 照样计入 passed）；配置区占位符未填 → rc=1（场景 A/B）；
`kernel_launches` 缺失 → `SystemExit`；`nsys_window_ns` 缺失 → `EmitError`；
区间部分重叠 → rc=3 且不落产物；体积超限 → rc=4。

### 第 4 步 — 诱饵实测（**收紧判据时这一步不是可选的**）

「判据改严了」这句话本身不是证据。**改严一条判据 = 必须同时交一个诱饵**：
一个**本该被抓住**的输入，证明**改之前 PASS、改之后 FAIL**。
只跑一遍新代码看到 FAIL 说明不了任何事 —— 那可能是别的原因 FAIL 的；
只跑一遍新代码看到 PASS 更说明不了任何事。可复现的做法：

```bash
# 0) 先把改动前的适配器留一份（这是"旧"的那一侧，没有它就无法对比）
cp validation/check_preflight_item.py /tmp/ORIG_check_preflight_item.py

# 1) 每个诱饵造一对 fixture：<case>.old / <case>.new，除了适配器不同，其它字节一致
# 2) 每对只施加一个变异（删掉 ABI 档位声明 / 契约不声明 warp 数 / 4 个空自测文件 ...）
# 3) 用同一个 --check-id 分别跑两侧，比 status 和退出码
#    判定：旧 PASS -> 新 FAIL = 收紧生效；两边都 FAIL = 这个诱饵没打中原来的漏洞，要重造
# 4) 必须另加一个"正控"：把所有项都按新判据填对，要求**新旧两侧都 PASS** ——
#    否则你不是收紧，是把正常输入也一起拦了
```

本轮实测（13 例，`check_preflight_item.py` 新旧对跑）：
**正控 1 例**，E03/E04/E11/E14/E15/E23/E25 七项在新旧两侧**都 `PASS/rc=0`**（没有过度收紧）；
**诱饵 12 例，其中 11 例「旧 `PASS/rc=0` → 新 `FAIL/rc=1`」**，逐条对应 3.1 的新增行。
唯一没打中的一例是**「4 个 0 字节文件，文件名照抄骨架的四个自测名」**：旧代码也 FAIL，
因为 `real_import` 那个 aspect 要 `importlib` / `emit_perfetto` / `trace_processor`，
**没有哪个骨架自测的文件名带这几个词**，所以靠文件名点不亮它。
把其中一个文件改名成 `selftest_decode_importlib.py`，五个 aspect 就全靠文件名点亮了，
旧代码于此确实 `PASS/rc=0` —— **这才是那条假绿灯的最小复现**。
造诱饵时要留意这种「差一个 aspect」的情况：**诱饵没打中不等于漏洞不存在，是诱饵造得不够狠。**

---

## 5. 单一来源 / 不要重复配置的地方

- HW 采样率、trace 体积上限：只在 `check_preflight_item.PROJECT` 定义，
  `check_gate_item.GATE_PROJECT` 从 `cp.PROJECT` 转出。
- 能力门 G12 / G14：`check_gate_item` 委托 `cp.check_e12` / `cp.check_e15`。
- 自测 fixture、`trace_processor_shell` 路径、禁用 import 统计
  （`FORBIDDEN_IMPORT_STATS`）：只在 `selftest_hw_merge.py` 定义，
  `selftest_perfetto.py` 通过 `import selftest_hw_merge as fixtures` 复用。
- 三个编译档位名：`PROJECT["compile_level_names"]`、
  `PROJECT_CORRECTNESS["level_names"]`、`PROJECT_OVERHEAD["level_names"]`
  三处必须一致（contract 里的 `compile_switch.levels` 是最终仲裁）。
- `gpu_idle.json` 相对路径：`wait_for_gpu_idle.GPU_IDLE_SNAPSHOT_RELPATH`
  与 `GATE_PROJECT["artifact_relpaths"]["gpu_idle"]` 必须一致。

## 6. Python 3.8 兼容约束

所有文件都以 `python3 -B` 直接运行，必须 3.8 可用：
文件头 `from __future__ import annotations`；运行期不要用 `X | None`
（注解里可以，因为有 `__future__`）；不要用 `:=` 之后的新语法特性、
`dict |` 合并、`match`。
