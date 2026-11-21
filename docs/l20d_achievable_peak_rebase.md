# L20D（B300 系阉割 SKU）nsys 指标修正：实测可达峰值重归一

适用卡：**NVIDIA L20D**（GB110, CC 10.3, 148 SM, 2032 MHz max, 288GB HBM3e, 18×NVLink5
53.125 GB/s/link, TDP 1100W）。这是 B300（Blackwell Ultra, SM103）的阉割衍生 SKU——SM
从 160 屏到 148、TDP 从 1400W 砍到 1100W，**无官方公开 datasheet**，一切峰值以本机实测为准。

与 H20 ×4 是**两类不同的失真**，不要混用：

| | H20（GH100 系） | L20D（GB110 系） |
|---|---|---|
| 失真类型 | die 归一化：nsys 按整颗 GH100 die 的吞吐峰值归一，H20 TC 只有 1/4 | 理论峰值归一：nsys 按 DRAM/NVLink 的理论速率归一，但该 SKU 持续可达值更低 |
| 指纹 | `Tensor Active` 平台期压在 25 = 100/4（hw_scale_plausibility 门禁检测 100/2^k） | 占用率读数系统性偏低，**无平台期**（Tensor Active 烟测可达 100） |
| 修法 | scale 规则 factor=4 | scale 规则 factor = nsys归一分母 ÷ 本机实测可达 |

## 判定流程（先做，再决定写不写规则）

1. Tensor Active 是 pipe 占空 counter（pct_of_peak_sustained_active），不含速率假设。
   验证：bf16 8192³ dense GEMM 烟测里 Tensor Active max=100、SMs Active max=100 →
   无 die 归一平台期 → **Tensor 不加 scale**。本卡实测 per-clock 速率 = B200 档
   （BF16 8192 / FP8 16384 / FP4 32768 FLOP/clk/SM，三精度独立实测均落在 2 的整幂上，
   效率 92-95%），与 nsys nameplate 口径一致，无需速率修正。
2. DRAM/NVLink 的 % 是 nsys 按理论峰值归一的吞吐：用一次真实采集的 % 与字节数反推
   分母（本卡实测：DRAM 43.7% ↔ 3.57 TB/s ⇒ 分母 8183.8 GB/s；NVLink TX 43.5% ↔
   ~416 GB/s ⇒ 分母 956.25 GB/s/向）。
3. 实测本机可达峰值（GPU 独占干净窗口、全 148 SM、足迹 ≫ L2）：
   - HBM：read 7.33 / write 6.52 / copy(r+w) 6.55 TB/s（全 SM LDG 网格）
   - NVLink 单向：cudaMemcpyPeer 771 / NCCL sendrecv 675 / 8卡 A2A busbw 621 GB/s
   - 算力（ dense GEMM 持续）：FP4 5.34 PF / FP8 2.87 PF / BF16 1.40 PF——
     **1100W 功耗墙**下 SM 时钟从 2032 掉到 1160-1250 MHz（SW Power Cap Active，
     温度仅 56°C），nameplate 9.85 PF 只在毫秒级存在。
4. factor = nsys分母 / 实测可达：
   - DRAM Read ×1.1164（8183.8/7330）、DRAM Write ×1.2552（8183.8/6520）
   - NVLink TX/RX User Data Total ×1.2403（956.25/771；若按 A2A 模式真实可达
     621-675 则 ×1.42-1.54，保守取硬件上限 771）

## FP4 Tensor Active 天花板 ≠ 100（第三类失真：稀疏归一，2026-09-21 实测标定）

第 1 条说"Tensor Active 烟测可达 100"只验证过 **BF16**。对**块缩放 FP4 tcgen05**
（SM103 E2M1+UE4M3），该 counter 的 dense 饱和读数只有 ~48-50%：

| 工作负载 | 交付算力 | Tensor Active (ncu) |
|---|---|---|
| CUTLASS FP4 2SM, FC1 shape 40960×4096×8192（已知打满） | 8.59 PF | **48.3%** |
| CUTLASS FP4 2SM, FC2 shape 40960×8192×2048（薄 K） | 4.31 PF | 31.2% |
| megakernel champion 969.5us 热窗口 | ~5.1 PF 平均 | max 45%（≈dense 天花板的 93%） |

判读规则：**FP4 dense 利用率 = Tensor Active ÷ 48.3%**，不要用 ÷100——直接读会把
"热窗口已 93% 饱和"误判成"还有 55% 算力余量"。疑似 nsys 按稀疏（2×）峰值归一，
与 H20 的 die 归一 ×4、本卡 DRAM/NVLink 的理论峰值归一是第三类失真；whitelists
里不配常数 scale（FC1/FC2 每 % 对应 PF 不同：0.178 vs 0.138），只做判读口径。

另：megakernel 混合负载（计算+通信+调度）实测**不撞功耗墙**——nvidia-smi 旁路
100ms 采样：全程 SM 2032MHz 满频、功耗 ~250W（瞬时峰值 467W），只有 dense GEMM
持续负载才把时钟打到 1160-1250MHz。推论：**dense GEMM 实测峰值（FP4 5.34PF）是
降频口径，对不降频的 kernel 不是适用分母**；满频 shape-aware 口径（FC1 8.59 /
FC2 4.31 PF）才是。报 roofline 占比时必须注明分母口径。排查方法：benchmark 期间
旁路 `nvidia-smi --query-gpu=clocks.sm,power.draw -lms 100`，比重采 HW 便宜得多。

nsys 平台侧另两个已确认异常（进 normalization 审计）：
- GPU metrics 的 GPC Clock Frequency 标注 MHz 而原始值是 9.7e8~2.03e9（实为 Hz）。
- 纯 DTOD memcpy 区段（0 个 kernel）SMs Active 仍报 92-96%——该平台把拷贝引擎
  活动计入 SM 活跃，用 SMs Active 判 kernel 忙碌前先用 kernel 数过滤。

## 坑

- **合并 lane 不能用单一基线缩放**：DRAM Total = Read+Write，用 copy 口径（6.5 TB/s）
  统一缩放会把读主导窗口顶到 >100%（实测 max=115.7%）——读可达 7.33 > copy 6.5。
  必须读/写分向各自 scale，Total 保留原始口径。
- **Tensor Active 不能用常数因子表达 FLOP 利用率**：占空 × per-clock 速率 × 时钟(t)，
  而时钟被功耗墙打成负载相关变量（dense MMA ~57-60% nameplate；混合负载 megakernel
  反推 ~1.9-2.0 GHz）。要么在 trace 里并行采 SM 时钟 lane（NVML 旁路，见下），
  要么报告时注明口径。
- scale 规则的 factor=1 + 空 regex 会被 extract 判 fatal（H20 丢 ×4 的教训），
  每条规则必须 device_name_regex 命中 `L20D`、带 rationale。

## 落点（mk_v2 实例，2026-09-20 验证）

`megamoe_sm103_work/instrument/mk_v2/profile/hw_perfetto/metrics/metrics_whitelist.json`
的 `scale` 数组 + `measured_peaks_20260920` 锚点块；`extract_gpu_metrics.py` 在
combine 之后应用 scale，派生 lane 带 `scale_factor`/`rationale` 落盘，门禁
`hw_scale_plausibility` 通过 `scale_rules` 已生效记录闭环。在 0918 真实 trace
（runs/swhw_atomic_v1）重跑验证：DRAM Read max 75→83.7%、NVLink TX max 68→84.3%。

## 功耗 lane（NVML 旁路）

nsys PM sampling 只有 PerfWorks counter，没有功耗通路。补法：独立进程（避免 GIL/
被测进程干扰）用 pynvml 采 `nvmlDeviceGetPowerUsage` + `nvmlDeviceGetTotalEnergyConsumption`
（+ SM/MEM 时钟），每样本记 (CLOCK_REALTIME, CLOCK_MONOTONIC)。固件有效刷新率仅
~7.7-9.2 次/s（≈100ms 一格），200Hz 采样已远超信息量。对齐用 UTC 锚点：nsys sqlite 的
TARGET_INFO_SESSION_START_TIME 给 (utcEpochNs, systemClockNs)，功耗样本按
realtime - utcEpochNs 落到 nsys 时间轴；**不要用应用侧 MONOTONIC 当锚点**（实测遇到过
运行期间一次性时钟跳变：应用侧 MONO 窗口 23.657s vs nsys 会话 12.511s）。独立校验：
功耗突发 vs nsys SMs Active 突发做互相关，取最大相关 lag 作残差。
