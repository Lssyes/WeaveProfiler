# nsys GPU Metrics 采样为什么会全 0，以及 capture-range 怎么解决

一次真实排查记录。现象是最终 trace 里 hw counter 全是 0，根因不在 GPU、不在 kernel，
在 nsys 的采样缓冲区落在了 NFS 上。

---

## 现象

6 个交付 kernel 里有 3 个的 hw counter 峰值精确为 0：

```
kernel              SMs Active mean    max
minimax/decode          97.646       100.0    OK
glm52/decode            94.784       100.0    OK
llama/prefill           98.021       100.0    OK
minimax/prefill          0.000         0.0    ← 全 0
kimi/decode              0.000         0.0    ← 全 0
kimi/prefill             0.000         0.0    ← 全 0
```

sw 部分完全正常（936 泳道、phase/task 双层、百万级 spans），
hw 时长也匹配（92%~99.7% 覆盖 sw），**只有数值是 0**。

判据要点：**mean 低（3%、5%）可能是 kernel 真实占用低，不该判；
max 精确为 0 才是采集失败** —— kernel 只要在 GPU 上执行，SMs Active 峰值不可能是 0。

---

## 根因：采样缓冲区排空阻塞在 NFS 写上

先看采样间隔的真实分布（`GPU_METRICS` 表，SMs Active 这一条）：

```
采样点总数     2,239,361
间隔中位数         4,992 ns   (4.99 us)  ← 正常，就是 200 kHz
间隔 p99           5,024 ns
最大间隔   1,116,393,005 ns   (1116.4 ms) ← 异常

>1ms 的大 gap    16 个
gap 合计          16.18 秒
采集总跨度        27.37 秒
gap 占比          59.1%
```

16 个 gap 全落在 979~1116 ms 这个窄区间，规律性极强。关键线索是**第一个 gap 的起始时刻**：

```
t0 + 0.499992 秒   ← 几乎精确 0.5 秒
之后：活跃 0.4~0.7 秒 / gap 0.96~1.18 秒 交替
```

0.5 秒正是采样缓冲区被打满的时间。本机 workspace 是**阿里云 extreme NAS 挂载**，
而 `capture_nsys.py` 把 nsys 的 TMPDIR 设在 out-dir 下 —— 也就是落在 NAS 上。
200 kHz × 8 counter 打满缓冲后，nsys 的收集线程阻塞在 NFS 写上，这就是那 1 秒。

于是时间轴变成：

```
├─采 0.5s─┤├─堵 1.0s─┤├─采 0.5s─┤├─堵 1.0s─┤ ...
```

采样只占 41%，停 59%。一次 13.7 ms 的 kernel **有 59% 概率完整落进「堵」的那一秒**，
窗口内一个有效采样点都没有。

### 更上游的问题：采集窗口等于整个进程生命周期

缓冲区为什么会被填满？因为以前的命令是最朴素的用法：

```
nsys profile [options] python3 <harness.py> [args]
```

**没有 `--capture-range`，所以 nsys 从 python3 启动一直采到进程退出。**
采集窗口 = 整个进程生命周期。那 43.6 秒里 GPU 在干什么：

```
Python 启动 + import torch          ~3 s
CuTe DSL cute.compile 编译 kernel   ~6 s
准备输入 (randn)                     ~1 s
跑 torch 参考实现 golden            ~10 s
► launch 目标 kernel                13.8 ms   ← 只有这个是要的
拷回 61 MiB buffer                  ~1 s
数值比对 correctness                ~15 s
写 raw / events 文件                ~5 s
```

目标 kernel 占 **0.03%**，其余 99.97% 的采样数据是编译、golden、数值比对期间的
GPU 状态 —— 全是垃圾。而正是这些垃圾把缓冲区填满，触发写 NAS，堵住采样。

因果是讽刺的：

```
采得越多  →  缓冲区越早满  →  堵得越久  →  真正要的那 13.8 ms 越采不到
```

A 组数据印证：802 次 launch 里 **505 次窗口内采样点为 0**，成功率只有 37%。
跑 800 次不是因为需要 800 次数据，是因为单次成功率太低，只能靠数量凑。

**为什么当初没缩小窗口**：`capture_nsys.py` 里 `--capture-range` 参数一直存在，
但默认值是 `'none'` —— 因为 `nsys profile <command>` 是官方最常见的用法，不需要改被测代码。
旧版本唯一试过一次是在 EP2 多卡上撞了 `dist.barrier()` 死锁，曾把这条路记成「不可用」；
该观察后来由本节的双 rank 对称顺序与 `hw_run9` measured artifact 重新复核。

---

## 解法：用 cudaProfilerApi 圈定采集窗口

让 nsys 只在目标 kernel 执行期间采样。窗口内数据量远小于缓冲区容量，
**压根不触发排空**，也就没有 NFS 阻塞。

### payload 侧

```python
# 1. warmup 必须在窗口外 —— CuTe DSL 首次 launch 要 cute.compile，耗 6 秒
#    编译数据进了窗口就会灌满缓冲，把 gap 请回来
for _ in range(3):
    cnt.zero_()
    mk.launch(inputs, prof=(buf, cnt))
torch.cuda.synchronize(device)

# 2. 开窗
torch.cuda.profiler.start()          # = cudaProfilerStart()

# 3. 窗口内只跑目标 kernel
cnt.zero_()
mk.launch(inputs, prof=(buf, cnt))   # 13.7 ms
torch.cuda.synchronize(device)

# 4. 关窗
torch.cuda.profiler.stop()           # = cudaProfilerStop()
```

### nsys 侧

```
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...
```

`--capture-range=cudaProfilerApi` 让 nsys 启动后**先不采**，等进程调 `cudaProfilerStart()`
才开始，调 `Stop()` 就结束。

### 两个实现细节

- **warmup 一定在 `profiler.start()` 之前**，包含 `cute.compile`。否则 6 秒编译数据
  灌满缓冲，gap 照样出现。
- **每次 launch 前只清 `cnt`，不清 61 MiB 的 `buf`**。清 buf 会显著拖慢；
  cnt 归零就等于记录指针重置，buf 里的旧数据会被覆盖，不影响正确性。

### 为什么缩小窗口就够了：算一下缓冲区占用

```
数据产生速率 = 200 kHz × 8 counter = 每秒 160 万条记录
缓冲区容量  ≈ 80 万条（实测 0.5 秒填满，反推得出）
```

**注意不能按 kernel 时长估算窗口**。实测 `cudaProfilerStart/Stop` 有约 **168 ms 的启停延迟**：
minimax/decode 的 payload 窗口墙钟只有 5.88 ms，但 PM 采样实际跨度是 **174 ms**。
所以要按实际采样跨度算：

```
旧：43.6 秒 × 160 万/秒               = 缓冲区被反复填满几十次
新：约 180 ms × 160 万/秒 ≈ 288,000 条 = 缓冲区的 36%
```

36% 没满，所以零 gap 的结论成立。但**余量不算大** —— 如果 kernel 更长（比如
kimi/prefill 的 28.6 ms）或 counter 更多，这个余量会被吃掉。那时就得靠把
nsys TMPDIR 换到 `/dev/shm`（见下文「独立的改进方向」）。

注意 **TMPDIR 位置没变，还在 NAS 上** —— 变的是整个采集期间一次盘都没写。
只在 nsys 退出时把最终几 MB 的 sqlite 落一次，那点量无所谓。

### warmup 的编译开销因 kernel 而异，差一个数量级

`cute.compile` 的耗时实测：

```
minimax/prefill    ~6 秒
minimax/decode    ~42.3 秒
```

「warmup 必须在 `profiler.start()` 之前」这条约束在 decode 上更要命 ——
42 秒的编译数据灌进窗口，缓冲区必满，gap 立刻回来。

### 一张表看清前后差别

| | 以前（无 capture-range） | 现在（capture-range） |
|---|---|---|
| 采集窗口 | 43.6 秒（整个 python 进程） | 13.7 ms（只有目标 kernel） |
| 目标 kernel 占窗口 | 0.03% | 100% |
| 缓冲区占用 | 反复撑满 → 堵 NAS 写 → 59% 时间不采样 | 2.7%，从不触发写盘，零 gap |
| 单次 launch 采到的概率 | 37%（802 次里 505 次全 0） | ~100%（4/4 重复成功） |
| CUPTI 里的 launch 数 | 802 条（混着 warmup、未插桩 build） | 1 条 |
| 要不要挑 launch | 要，`most-samples` 从 802 里挑，**可能挑错 build** | 不用挑，只有一个 |
| hw 与 sw 是同一次 launch | ❌ 不是（实测选中第 41 次） | ✅ 是 |
| sqlite | 483.68 MB | 4.53 MB |

**「限定范围」同时解决了两个原本看起来无关的问题**：

1. **采样 gap** —— 数据量小到不写盘
2. **选错 build** —— 窗口里只有一次 launch，没有可选的余地

第 2 点尤其值得强调。`--launch-select most-samples` 只看「哪次 launch 采样点最多」，
**完全不看那次是不是同一个 build**。实际翻车过两次：

```
minimax/decode
  hw 采的   runner.py --mode timing → 生产路径 prof=None，走 CUDA graph，0.67 ms
  sw 记录   插桩 build                                              2.8 ms
  → most-samples 选中未插桩那次，贴到 sw 上覆盖率只有 24%

kimi/prefill
  payload 走 run_profile.py --batches 32
  33 次 launch 里只有 4 次是插桩的，其余全是未插桩
  连 --max-records 都不一致（1024 vs sw 的 4096）
```

CUPTI 表是个大杂烩：插桩 build、未插桩 build、warmup、别的 kernel 全在里面，
`most-samples` 分辨不了。**唯一能识别的信号是时长匹配判据（0.5x~2x）** —— 这就是它必须存在的理由。
开了 capture-range 之后，这个风险从「靠判据事后兜底」变成「结构上不可能发生」。

### warmup 无害，别误判

顺带澄清一个容易搞混的点：**warmup launch 进 CUPTI 是无害的**。

实测 A 组 802 条 CUPTI 记录里 warmup 那几条也在，但时长分布是：

```
min 13.690 / median 13.715 / max 13.832 ms
偏离中位数 >50% 的 launch = 0 个
```

warmup 走的是**同一个插桩 build**，而 `cute.compile` 那 6 秒是 host 侧编译开销，
**不计入 kernel 执行时长**。所以选中 warmup 等于选中一次等价 launch，没有问题。
真正的危险是上面说的「混进不同 build」，不是 warmup。

### build 一致性：这次在 3 个 kernel 上都翻过车

「hw 采的和 sw 记录的必须是同一个 build」这条极其容易违反，实测三种形态：

```
minimax/decode   hw 采未插桩 timing build 0.67 ms  vs  sw 插桩 build 2.8 ms
                 → 时长差 4 倍，被时长判据抓住 ✓

kimi/prefill     33 次 launch 里只有 4 次是插桩的，其余全是未插桩
                 且 --max-records 1024  vs  sw 的 4096
                 → 混合体，靠运气

glm52/decode     --max-records 1024  vs  sw 的 16384
                 → 时长差别不到 2 倍，时长判据抓不住 ✗
```

**最后这种是判据的盲区**：buffer 容量不同不一定让 kernel 时长差到 2 倍以外，
但产出的 record 数、溢出行为、清零耗时全都不同，本质上是两个 build。

各项目的插桩开关形态还各不相同，更容易踩：

```
minimax/prefill   host 侧 launch 参数     mk.launch(inputs, prof=(buf, cnt))
minimax/decode    编译期常量 + 环境变量    MM3_PROFILE=1 → prof.PROFILE=True
                                          （且自动强制 DISABLE_GRAPH）
glm52/decode      环境变量                MK_PROF_LEVEL=2 / MK_PROF_MAX_RECORDS
kimi/prefill      构造参数                KdaMegakernel(profiler_enabled=True)
llama/prefill     构造参数                LlamaPrefillMegakernel(..., profiler_enabled=True,
                                                                 profiler_max_records=128)
```

同一个 skill 要面对 5 种接口，每次都得读代码确认，不能照抄上一个项目。

**capture-range 从结构上避免了这个问题**：窗口里只有你亲手用同一份配置跑的那一次 launch，
CUPTI 里就 1~4 条记录，没有「混进别的 build」的空间。这比事后靠判据兜底可靠得多。

写 payload 时建议加拒跑断言（minimax/decode 的 payload 就这么做的）：

```python
assert prof.PROFILE, "插桩开关没打开，会采到未插桩 build"
assert prof.DISABLE_GRAPH, "CUDA graph 没禁用，kernel 时长会对不上"
```

---

## 实验数据

同一个 kernel（minimax/prefill，13.7 ms），GPU 3，四组对照：

| 组 | 方案 | kernel 窗口内采样点 | 覆盖率 | SMs Active mean |
|---|---|---|---|---|
| A | 800 次 launch，无 range | 797,487（802 次合计） | 297/802 次有采样，**505 次全 0** | 99.602 |
| **B** | **1 次 launch + range** | **2,740**（理论上限 2,745） | **99.82%** | 99.590 |
| C | 50 次 launch + range | 137,850（上限 138,071） | 99.84%，50/50 全中 | 99.606 |
| D | 1 次 launch，**无** range | — | **3 次里 2 次全 0** | 0（r3 全程非零占比 = 0） |

三组的 SMs Active mean 逐位对齐（99.590 / 99.602 / 99.606），**capture-range 没有精度损失**。

D 组是因果对照：同样只跑 1 次 launch，唯一区别是关掉 range，就退回全 0。
所以起作用的是 capture-range，不是「采集窗口短所以碰运气中了」。

B 组独立重复 4 次，**4/4 成功**，max gap 只有 5,025 ns（零个 >100 us 的 gap）。

### 6 个生产 kernel 全量重采的实测

实验结论在 6 个真实交付 kernel 上全部复现，无一例外：

| kernel | kernel 时长 | sqlite 缩小 | max gap 前 → 后 | 零采样 launch 占比 | SMs mean 前 → 后 |
|---|---|---|---|---|---|
| minimax/prefill | 13.7 ms | 483.7 → 4.5 MB (107×) | 1116 ms → 5,025 ns | 505/802 (63%) | 99.602 → 99.590 |
| minimax/decode | 2.80 ms | 497.7 → 7.8 MB (63.6×) | 1093 ms → 5,025 ns | **1144/2008 (57.0%)** | 97.646 → 97.746 |
| glm52/decode | 0.88 ms | 1186.6 → 31.9 MB (37.2×) | 1189 ms → 5,025 ns | — | 94.784 → 95.017 |
| kimi/decode | 2.25 ms | **10681 → 7.5 MB (1427×)** | 1246 ms → 5,024 ns | **8/14 (57%)** | 95.753 → 95.797 |
| kimi/prefill | 28.6 ms | 500.6 → 7.4 MB (67.5×) | 1043 ms → 5,025 ns | **113/201 (56%)** | 99.932 → 99.932 |
| llama/prefill | 3.6 ms | 366 → 7.8 MB (47×) | 秒级 → 5,025 ns | — | 98.021 → 98.070 |

两点值得注意：

**一、SMs Active mean 全部逐位对齐**（最大偏差 0.25%），kimi/prefill 甚至完全相同（99.932）。
缩小采集窗口**不损失任何精度**。

**二、三个独立 kernel 的「零采样 launch 占比」都是 56~57%**，
和从采样间隔直方图独立算出的 gap 占比（55.85%~59.1%）高度一致。
两个完全不同的测量路径给出同一个数，交叉印证了「gap 吃掉了约 57% 的采集时间」这个结论。

kimi/decode 的 1427× 是最夸张的：旧方案是 14 个独立进程各采一次，每个都从头采整个进程生命周期，
10.7 GB 的 sqlite 里 99.9% 是垃圾。

### 顺带的收益

```
sqlite          483.68 MB → 4.53 MB   (107×)
extract 输出    134.70 MB → 0.47 MB   (289×)
采集墙钟         40.4 s   → 21.9 s
```

### 最本质的收益

旧方案里 **hw 采的那次 launch 和 sw 记录的那次不是同一次**：sw 记录 harness 跑的那一次，
hw 用 `--launch-select most-samples` 从 802 次里挑采样点最多的（实测选中第 41 次）。
这依赖一个假设 —— 同 build 背靠背连续跑，每次 launch 行为等价。
假设靠时长匹配判据来验证（sw 13.819 ms vs hw 13.827 ms，rel_err 0.05%）。

开 capture-range 之后，窗口里就那一次 launch，**hw 和 sw 是同一次**，假设不再需要。

---

## 适用边界

| 场景 | 结论 |
|---|---|
| 单卡单进程 | **默认开** `--capture-range=cudaProfilerApi --capture-range-end=stop` |
| 多 rank / NCCL（EP2） | **可以开，但必须双 rank 对称验证并由交付闸门验收**；不能再以旧的 600 秒 watchdog 观察作为关闭策略 |

## EP2 双 rank 的已验证顺序（hw_run9）

旧记录把一次未对称的 `dist.barrier()` watchdog 观察概括成“多 rank 保持关闭”。
本次 `hw_run9_200khz_8c_range` 已用双 rank 重新验证，实际顺序是每个 rank 都执行：

```
Start -> barrier -> target kernel launch -> cuda synchronize + barrier
      -> Stop -> barrier
```

`Start` 之前已经完成 input staging、output 预分配和 buffer 清零；JIT/compile、warmup、
计时循环、reference/correctness 以及最终通信均在 capture window 外。该顺序来自
`profile_mega_moe.py` 的 `--cuda-profiler-range` 分支；nsys 的 application log 同时记录
`Capture range started in the application.` 和 `Capture range ended in the application.`，
因此这里的结论是“API 窗口实际生效”，不是只看命令行参数。

本次远端已测 artifact 的关键数值如下（没有在本笔记更新时重新启动 GPU）：

| 证据 | measured 值 |
|---|---|
| rank/设备 | `num_processes=2`，`CUDA_VISIBLE_DEVICES=0,1`，物理 rank/device `0,1`；`run_manifest.correctness.all_passed=true`，rank 0/1 均 `passed=true` |
| range/进程 | `cudaProfilerApi/stop`；nsys profile=0、export=0；capture/application start-stop 均出现；采集后 `busy=false`、foreign processes 为空 |
| PM 配置 | `file:nsys_metrics_ep_moe_hopper.config`，200,000 Hz（周期 5,000 ns），原始配置 8 counters；显式包含 `NVLink TX Requests User Data` |
| GPU metrics | SQLite `GPU_METRICS` 共 521,512 行；裁剪后 8 个配置 counter 各 `sample_count=4556`；目标 launch 窗口内每个配置 counter 3,797 点，其中 `NVLink TX Requests User Data` 也是 3,797 点 |
| 连续性 | 全局及目标窗口最大采样间隔 5,024 ns，满足 `<= 10 * 5,000 = 50,000 ns`；全局 min/median 为 4,990/4,992 ns |
| target kernel | GPU0 的 `sm90_fp8_mega_moe_impl` 唯一 launch：`[33,190,559, 52,174,992] ns`，持续 18.984433 ms；HW clip window `[31,292,116, 54,073,435] ns` 覆盖该 launch |

因此 EP2 的硬条件是“所有 rank 同步执行并正常退出 + 单次目标 launch 落在窗口 + 每个必需
counter 有正样本且连续”，而不是按 rank 数量关闭 capture range。交付时仍必须同时提供
`capture_manifest.json`、`hw_metrics.json` 和 harness `run_manifest.json`；任一 rank、launch、
counter 或 barrier 顺序证据缺失都应由 `deliver_gate.py` 拒绝。

---

## 还有一个独立的改进方向

gap 的根因是 nsys TMPDIR 落在 NAS 上。skill 现在只把 **harness** 的 TMPDIR 放了
`/dev/shm`（为绕开 symm_mem UDS 的 108B sun_path 限制），**nsys 自己的没管**。

把 nsys TMPDIR 也指到本地 tmpfs，能从根上消除 gap，
而且**对多卡场景同样有效**（不依赖 capture-range）。这条尚未实测。

---

## 相关判据（已固化进 skill）

```
deliver_gate.py     SMs Active 峰值为 0 → FAIL (hw_all_zero)
                    只判峰值，不判 mean —— mean 低是 kernel 真实属性
merge_hw_...py      窗口内采样点 < 500 → FAIL (--min-window-samples)
                    hw 跨度不在 sw 的 0.5x~2x → FAIL（采错 build 的铁证）
capture_nsys.py     --gpu-metrics-device 必须与 harness 实际跑的物理卡一致
                    (nsys 按物理序号枚举，不受 CUDA_VISIBLE_DEVICES 影响)
```

最后一条也是这次踩出来的：kimi/prefill 那次 `CUDA_VISIBLE_DEVICES=3` 但
`--gpu-metrics-devices=0`，workload 在物理 GPU 3、counter 采物理 GPU 0 的空闲卡，
CUPTI 窗口看着对，窗口内值全 0。

---

## 原始数据

实验目录：`<项目>/.work/capture_range_exp/`
完整报告：`<项目>/.work/capture_range_exp/RESULT.md`（含每组 sqlite 与逐项统计）
