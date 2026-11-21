# 快速插桩：完整说明

数据全部来自一次真实运行：SM90 DeepGEMM MegaMoE，H20 EP2，qwen3.6 规格，num_tokens=32。
没有估算值。判据的执行入口在 `SKILL.md`，本文是完整规格与理由。

---

## 一句话

**慢的不是插桩，是判断该怎么插桩。**

一次从零开始的任务，40 分钟里 3 个 agent 通读 kernel 烧掉 420 万 token，
最后只换来一个结论 —— 「这项目早就插好桩了，只缺一个解码器」。
这个结论 grep 几个指纹 3 秒能得出。

而且判断错的代价更高：这次差点规划出「改 profiler 扩 tag 位域到 per-warp」这个方案，
它注定被 SASS 逐字节等价门禁毙掉（附录 A3），整个修补方向会白做。

方案只干两件事：把判断脚本化；做好一次之后第二次起免费。
**修补的钱省不掉，判断的钱能全省。**

```
本次(从零)      判断 420 万  +  修补 5400 万 token
下次同项目      0
下次新项目      判断 0  +  修补 仍要大模型
```

---

## 1. 流程四步

```
0  用户显式选择(无默认值)
     target_kernel        脚本列候选, 人选, 禁止靠目录名或历史 trace 猜
     granularity          warp(每物理 warp 的实测 task/phase) / sm(只聚合实测事件与拓扑 metadata)
     information_bundle   sw / hw / hw&sw
     capture_window       full-process / cudaProfilerApi
        |
1  探测存在性       代码, 秒级      -> AND 门
        |
2  冒烟跑 + 验有效性  代码, 几秒     max_records=2, 记录量 0.7%
        |
3  正式采集出图     代码, 分钟级
```

完整性结论按缺失类别分成快/慢两路径（慢路径再分两子类），而不是把所有缺项压成同一条慢路径：

```
fast / complete                   device、语义、host 链全部完整，直接采集
slow / host-bootstrap  device 与 task/phase 语义完整，仅 host 产物缺失；只补 host
slow / device-or-semantics        device 原语或语义边界缺失/错误；才允许改 kernel
```

当前探测器 `0/3` 只是传输信号：`0` 对应 `fast / complete`；`3` 后还要按
`capabilities`/`missing` 分类，区分 slow/host-bootstrap 与 slow/device-or-semantics。

第 0 步不能省，而且**人的显式选择不算大模型介入**。这三个选择必须无默认值 ——
一旦有默认值，大模型就会替人选，那就是在猜意图。

三个选择各自影响判据：

```
target_kernel        决定 profiler 锚点集。多 kernel 仓库里锚点集有多个元素,
                     溯源链可能闭合到错误的那一个 -> 选定它是 2.2 判据成立的前提

granularity          warp: ABI granularity 字段==2, 泳道契约走 R3, 预期 track 数==warp 角色数
                     sm:   ABI granularity 字段==1, 泳道契约走"每物理 SM 恰好一条",
                           physical_sm 泳道禁止携带 cta_id/cluster_id/cluster_rank
                     选错了整组泳道判据用错, 而且不会报错

information_bundle   sw:    不跑 NSYS
                     hw/hw&sw: 必须验 counter 落在 kernel window 内、counter 进程 pid
                            严格小于所有软件泳道(R6)、单次 capture 覆盖目标 rank、
                            所有 displayed ts 有限且非负、PM 原始零值保留且无插值/
                            forward-fill、counter 不专属归因给任一 phase；
                            CUPTI 无 dropped；采样频率 200 kHz，采集 gh100 后按
                            ep_moe_metrics.json 过滤，required 至少包含 SMs Active、
                            SM Issue、NVLink TX/RX Requests User Data

capture_window       full-process: 捕整段 harness，简单但会包含 warmup/采样循环
                     cudaProfilerApi: nsys 只捕最终 full 档落盘 launch 的 host 窗口，
                            排除前置同步和 anchor-only capture；同一个 kernel 内的
                            entry rank sync 仍在窗口内，只能按软件 anchor 后处理裁剪
```

---

## 2. 类 1 代码插桩（8 项）

在 kernel 代码里，**缺了要动 kernel**。

### 2.1 时钟源

**存在**：源码含 `%globaltimer`。**只认这个** —— `clock64()` 命中不算通过。

理由：`clock64()` 是 per-SM 周期计数且随频率变化，两个 SM 的值之间没有可比性，
拿它做跨 SM timeline 是错的。既然结果不可用，「命中就判过、只标个 flag」这个分支
就没有价值 —— 无论如何都得改成 globaltimer，和「完全没有时钟源」处理方式一样。
`%globaltimer` 从 sm_30 就有，不存在兼容问题。

判据保持单一条件（AND 门更硬）；命中 `clock64` 时把它写进 missing 的**诊断理由**，
说明改动量比从零插桩小得多。判据和诊断要分开。

**失效**：读到垃圾寄存器，或读的不是全局计时器。

**有效**：同一 `(block, track)` 内按 offset 递增的记录，**时间戳必须非递减**。

**本项目**：`mega_moe_profiler.cuh:116`
```cuda
asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp_ns) :: "memory");
```

### 2.2 独占槽位

每个 warp 在 global memory 有自己的格子，不和别人抢。

**存在**：两条都要满足 ——
```
正面  乘加式索引  blockIdx.x * A * B + track * B + offset
负面  记录函数内部(含时间戳读取的那个函数体)无 atomicAdd
```

**失效**：两个 warp 写同一地址（竞争），或地址算错。

**有效**：
```
tag_slot_mismatch == 0    地址反算出的 (block,track) 必须等于 tag 里独立编码的那份
开销 < 10%                有 atomic 竞争必然爆
```

第一条为什么硬：槽位公式是
```
slot = (block * 7 * max_records + track * max_records + offset) * 2
```
所以**从数组下标能反算出 `(block, track, offset)`**；而 tag 的 `[31:20]`、`[19:17]`
里**又独立编码了一遍**同样的 block 和 track。同一信息编码两次，必须相等。
竞争、地址算错、`max_records` 传错，都会让两者分道扬镳。

开销 <10% 不是洁癖：超过就说明**观测已显著改变被观测对象**，
测出来的不是 kernel 的真实行为而是被插桩扭曲后的行为，结论作废。

**本项目**：`:123-127`，地址完全由 `(blockIdx, track_id, offset)` 决定，
三者对每个写入者唯一，一行 atomic 都不需要。实测 `tag_slot_mismatch = 0`。

### 2.3 编译期开关

**存在**：`template <bool kX...>` 加方法体全空的 `<false>` 特化；或 `#if` / `#ifdef`；
或 `-D*PROFILER*`。

**失效**：写成运行时 `if` 而非编译期消除，生产构型也背着开销。

**有效**：跑 `profiler_buf=None` 一次，要求 **buffer 全零且时间与无插桩基线一致**。
运行时 `if` 的话开销不会归零。

**本项目**：`:42` 模板声明 + `:46` `MegaMoEProfiler<false, kNumSMs>` 全空特化。

### 2.4 buffer 分配

host 侧要分配一块足够大的、清零的 int64 显存，交给 kernel 往里写时间戳。
这一项查它算得对不对。就一个风险：**开小了**。

**存在**：
```
够大    numel >= 1 + num_sms + num_sms * 7 * max_records * 2
        这个 7 和 max_records 必须和 device 侧算地址用的是同一个值
清零    因为解码器靠"时间戳 == 0"判断空槽
对格式  int64 / 连续 / 在同一张卡上
```

`2.2` 和 `2.4` 是一枚硬币的两面：**2.2 是 device 怎么算地址，2.4 是 host 怎么算总量，
两者必须同源。** 一旦不一致，就是越界写坏别人内存，或者被
`offset >= max_records` 挡掉、静默丢事件。

**有效**：
```
truncated_block_tracks == 0   没有 (block,track) 把格子写满, 即没丢事件
buffer 尾部存在未写区         证明确实开够了, 不是刚好写满
metadata 区的 num_sms / num_tracks 与实际一致
```

**本项目**：`sm90_fp8_mega_moe.hpp:424-439` 要求
`numel >= 1 + num_sms + num_sms*7*max_records*2` 且 `0 < max_records < 2^29`；
`:270` 由 `profiler_buf != nullptr` 自动决定 `kEnableProfiler`。
实测 `truncated_block_tracks = 0`。

### 2.5 rank sync

**这是一项，不是两项。要求是：不插桩的构型有 rank sync，插桩的构型也有。**

理由：rank sync 的目的就是让两个构型可比、提供共同起跑线。只有插桩构型有的话，
它就不是共同起跑线，而是插桩独有的开销 —— 那这一项本身不合格，
不是「合格但口径需注意」。

**存在**：
```
入口区存在 rank 同步 barrier, 且不是业务流程内部的 barrier
  区分方法: 从 kernel 函数体开括号到该行之间不能出现业务代码
           (for/while 循环头、TMA/cp.async/WGMMA、cute::copy|gemm、
            phase 命名调用、atomicAdd)
且该 barrier 不在插桩开关内   <- 不在开关内才意味着 baseline 也执行它
```

**失效**：只有插桩构型有 → 两构型不同起跑线 → 开销测量把整套 rank sync 成本算进去，
CTA 启动抖动没被消除。

**有效**：
```
结构  ctas_missing_anchor == 0     每个 CTA 都有锚点, 不是只有 CTA0
      锚点是该 CTA 的第一个事件     offset == 0
      CTA 间 skew 够小             实测 78 CTA end_skew 192 ns
      min(release) >= max(arrive)  必须在同一时钟域内判

时间  T_anchor - T_none 应接近 0
      两档只差 1092 条记录, 若 rank sync 两边都有, 差值就只是那点记录开销;
      若差值接近 rank sync 时长(实测 56~58 us, 约 2.1%), 就证明 baseline 少做了它
```

「时间」那条把「baseline 到底有没有执行」变成**可测量**，而不只靠扫描 guard 位置推断。

**本项目 FAIL**：`fixed/kernel.cuh:192-220` 整段被 `if constexpr (kEnableProfiler)`
包住（`:202` start、`:205` `nvlink_barrier`、`:212` end、`:218` `grid_sync`），
baseline 完全不执行。而 `:745` 那处不在开关内的 `nvlink_barrier` 被正确排除 ——
因为 `:383` 有 `for (uint32_t i = lane_idx; i < kNumExperts; i += 32)`，
业务循环在它之前，它是 combine 前的业务 barrier 而不是 entry 对齐。

要区分**合格**与**补救**：
```
合格   把 rank sync 挪到插桩开关之外, baseline 也执行  -> 要改 kernel, 慢路径
补救   三档测量, 从测量窗口里剔掉 rank sync            -> 零 kernel 改动, 但这项仍 FAIL
```
探测器该明确报 FAIL 并告知有补救可选，不能默默放过 ——
那等于替人做了「带着这个缺陷继续」的决定。

### 2.6 task / phase / ignored 三态语义

层级由源码语义决定，不由名字关键词或后处理的包含关系决定：

```
task     真实、粗粒度的 operator 区间；start/end 都在 device 的真实边界打点
phase    真实 warp 阶段，如 load/MMA/epilogue/scatter/reduce-store；同样由 device 打点
ignored  整核锚点、grid barrier 等对齐证据或诊断；解码留档，但绝不画成可见 X
```

**persistent megakernel 的粒度铁律：每一个被 `grid_sync` 分隔的计算段各包一条 task，
task 名就是该段的真实算子名。**

单次 launch 的整核里没有 kernel 边界可借，`grid_sync` 边界就是唯一的算子边界。
段内是什么（quant / GEMM / scatter / pool / logits / topk / attention / 输出投影）
决定 task 的名字 —— 不是「这段有意思就展开、其余段只记 barrier」。
barrier 是 `ignored`，task-only 交付会把它整批丢掉，于是除那一段之外时间线全空。
这不是假设，是实测事故，见 2.8。

三态的渲染与折叠语义（同一份 device 源码，两种交付构型）：

```
                       task-only (record_phases=False)   phase (record_phases=True)
task    算子层          全保留，每个算子一条 bar            保留，作为外层
phase   算子内部        编译期整体消失，一条记录都不发       保留，嵌套画在 task 内层
ignored 锚点 / barrier  解码保留，不渲染                   解码保留，不渲染
```

三条约束：

```
1  phase 只被编译期常量 record_phases 门控「记录动作」。被门控的只有 append，
   计算和每一次 grid_sync 在任何构型下照原样执行 —— task-only 仍是同一个 kernel。
2  ignored 在 PROF_ENTRY_ONLY 就必须记：SW/HW 对齐要读它的 entry/release 时间戳，
   开销分母也是这一段。但出图器必须把它丢掉，一条都不许渲染。
3  内层区间只有在「同 lane 上严格被外层算子 task 包含」时才可提升到 task 层，
   且 ABI 要用 nested_within_task_id 写明 —— 这样覆盖率并集不变。
   拿不准就留在 phase：phase 允许消失，task 不允许。
```

**两层收敛示例（attention）**：整段收敛成**一条** `Sparse Attention` task；
内部细节（每 query row、每 KV chunk、KV load、WGMMA、online-softmax rescale、store）
全部降为 phase。于是 task-only 自动折叠成一条 bar，phase 构型才展开内部。
实测这样得到 22 条算子 task、15.7 万 span、99.1% 覆盖率、最大空洞 1.18µs、开销 5.05%。

**命名铁律**：task/phase 的展示名必须是人能直接对回代码的真实算子名 ——
`Quantize X to FP8`、`QKV-A Proj (FP8 GEMM)`、`Indexer Logits (MQA)`、
`Top-k Select`、`Sparse Attention`、`Output-B Proj (FP8 GEMM)`。
`passA` / `passB` / 裸 `token` / 裸 `tile` / `stage1` / `EV_*` 宏名 / `P11`
一律禁止：看时间线的人无法把一条 bar 对应回算子，这条 bar 就没有价值。
统一门禁对每条 `layer == "task"` 的展示名跑黑名单正则，命中直接 FAIL。

CuTe DSL 参考实现索引（`knowledge/examples/cute_dsl/`）：

| 概念 | 文件 |
|---|---|
| 三态记录原语、事件表（id/展示名/layer）、编译期开关 | `profiler.py` |
| 整核由 `grid_sync` 分段、每段一条 task、锚点与边界为 ignored | `megakernel.py` |
| 一条 task + 内部全 phase 的两层收敛，以及 ctx 元组为何不能用闭包 | `attention_task.py` |
| 捕获上下文（源端 `%smid` 过滤）与整核锚点 | `dispatch.py` |
| CTA 内 tile barrier（phase）vs `grid_sync` 边界（ignored） | `barrier.py` |
| 同一份 scatter 在 task 与 phase 两侧，层级由外层结构决定 | `scatter.py` |
| load / MMA / softmax / reduce / store 阶段 phase 打点 | `load.py`、`mma.py`、`softmax.py`、`reduction.py`、`store.py` |

**存在**（纯静态，不用跑）：

```
1  ABI 中每个事件都有唯一 id/name，并显式声明 layer=task|phase|ignored
2  每个可见 task/phase 的名字都在源码被引用，且 start/end 成对
3  task 打点包围真实 operator/GEMM 执行，不从若干 phase 的 min/max 在 host 合成
4  phase 打点对应真实 load/MMA/epilogue 等 warp 阶段
5  observer warp/track 绑定可溯源到源码；绑定只说明谁记录，不授权复制
6  源码里 grid_sync 调用次数 == ignored 层 barrier 事件对数；
   两个 barrier 之间的计算段数 == 源码用到的 task 层 id 数 == ABI 里 layer=="task" 的条数
```

**失效**：

```
语义反转        把 phase 当粗外层、把 load/MMA/store/wait 统称为细 task
段级遗漏        只给某一段打 task，其余 grid_sync 段只记 barrier —— barrier 是 ignored，
                task-only 交付时被整批丢弃，时间线除那一段之外全空（2.8 实测事故）
host 合成 task   从 phase envelope 生成 task，或为强行包含而延长 task
warp 复制        把一个 observer 的区间复制给未观测 warp
死枚举/单边界    声明后零引用，或只有 start/end 一侧
layer 缺失       只能靠名字猜 task/phase/ignored
代号命名        展示名是 passA / passB / 裸 token / 裸 tile / EV_* / P11 这类内部代号
```

**有效**（运行后）：

```
观测 id 属于 ABI 闭集；每个实测区间配平且 end >= start
event.layer、observer warp/track 与 ABI/源码一致
解码事件数与非空记录对账；后处理没有生成、复制或改写 task 时间戳
```

布局只消费真实区间。phase 完整落在 task 内时，以 **task 外层、phase 内层**嵌套；
phase 在 task 结束后开始时保持独立。若同一 lane 两个 X 区间形成 partial overlap
（相交但互不包含），产物硬 FAIL：修 device 边界或语义/泳道归属，禁止延长 task
或修改时间戳来“修图”。

物理 warp 拓扑可以由 metadata 完整声明；没有实测事件的 warp 保持空泳道。
若用户要求每个 warp 都有实测事件，必须回 device 加真实打点，不能在渲染阶段复制。

### 2.7 算子覆盖率判据

粒度对不对不能靠看图，要有一个能算的数。判据只有一句：
**所有 task 层 span 的并集必须铺满整核窗口，不留大洞。**

```
前置    对齐必须是纯平移，scale == 1
        scale != 1 说明 SW 时间轴被拉伸过，空洞长度不再有物理意义
        （scale != 1 本身已经会让 G19 失败，所以这里直接跳过不误报）
窗口    整核窗口 = ignored 层整核锚点的 [entry, release]
        （取 alignment 的 sw_window_ns；没有就退化为全部解码事件的 min/max）
参与    只有 args.layer == "task" 的 ph=="X" 区间参与
        phase 不参与（task-only 交付里它们不存在，判据必须两种构型同解）
        ignored 不参与（否则整核锚点一条就把并集撑满，见 2.8）
时间戳  必须用 args.start_ns_raw / end_ns_raw 这类原始绝对 ns
        rebase 成相对 ts 会把头部空洞抹平成 0
单条上限 任何**一条** task 区间占整核窗口 >= 90% 直接 FAIL，且不再计算覆盖率
        整核窗口严格宽于它内部任何一个 grid_sync 段，所以这种条不可能是算子；
        不拦它，一条冒充算子的锚点就能自己把并集铺满，覆盖率照样报 100%
算法    对参与区间取并集，裁到整核窗口，得到 covered 与若干 gap
门限    max(gap) <= max(整核 span x 5%, 50µs)
```

`90%` 这个单条上限的实测依据（各 target 最宽的单条 task 占整核窗口的比例）：

```
llama/prefill      Causal GQA Flash Attention    28.9%
llama/prefill      QKV Projection (BF16 GEMM)    41.3%
dsv4pro/decode     Sparse Attention              44.4%
dsv4pro/prefill    Sparse Attention              48.3%
--------------------------------------------------------
被禁的 megaphase 包络 phase_span                 51.9%
整核锚点误标成 task 的 kernel                   100.00%
```

真实算子最宽只到 48.3%，锚点是 100.00%，中间留了很大余量。

`50µs` 的绝对下限是给短 kernel 兜底的：窗口只有几百 µs 时 5% 太紧，
`grid_sync` 本身的等待就会误伤。反过来窗口到毫秒级时 5% 才是主判据。
`grid_sync` 边界那种几 µs 的缝是正常的；**整段算子缺失不是**。

四种典型 FAIL 形态，报错必须把空洞 `[起, 止, 时长]` 按时长排序列出来，
这样人一眼能看出是缺了头、缺了尾，还是中间某个段没插桩：

```
只有一条整核宽的条          -> 它是 ignored，不参与并集，于是覆盖率 ~0%，FAIL
整核锚点被误标成 task       -> 单条占 100%，触发单条上限，FAIL（报错点名那条 span）
算子只集中在头部或尾部      -> 另一侧是一整块空洞，远超门限，FAIL
20 段算子铺满、只剩边界缝   -> PASS（实测 99.1%，最大空洞 1.18µs）
```

实测参考值（H20 decode，整核约 2ms）：22 条算子 task、15.7 万 span、
覆盖率 99.1%、最大连续空洞 1.18µs、插桩开销 5.05%。

### 2.8 为什么整核锚点必须是 ignored

整核锚点（`EV_KERNEL` 的 entry/release）和 `grid_sync` 边界都**必须**声明
`layer: "ignored"`：解码数据里保留 —— 软硬件对齐要读它的 entry/release 时间戳，
开销分母也是这一段 —— 但**绝不渲染成 span**。

因为它是一条贯穿整核全宽的区间。一旦渲染，它会被画到每一条泳道上，
于是任何「用 min/max 包络算覆盖率」的判据都被它单条撑满到 100%。

**实测事故**（这一整节存在的唯一理由）：

```
做法      只把 attention 一段用 task 层展开（_rb/_re），
          其余 ~20 个 grid_sync 段只记 barrier（_rbp/_rep，phase 门控 + ignored）
后果      task-only 交付时 barrier 被丢弃，Perfetto 上除 attention 之外全是空的
          头部 769µs、尾部 337µs 没有任何算子
真实覆盖率 44.9%
为何没发现 一条贯穿全宽的整核锚点长条被渲染到每一条泳道，
          把 min/max 覆盖率包络撑满到 100%，25 个门禁全部 PASS
```

三条永久约束，缺一条这个洞就会重开：

```
1  ignored 层一条都不许渲染成 ph=="X"（出图器丢弃 + 离线自测锁死）
2  覆盖率只认 layer=="task" 的并集，禁止用 min/max 包络
3  每个 grid_sync 计算段都必须有自己的 task —— barrier 记录不是算子插桩
```

元教训：**门禁全绿不等于测量正确。** 一个能被单条锚点撑满的判据，
在数学上等价于「不判」。判据自身必须对「只插了一段」这种输入返回 FAIL，
所以离线自测里必须有「只有整核锚点」和「算子只在头部」两个负例，且必须 exit 1。

---

## 3. 类 2 转换器（3 项）

host 侧，**缺了改 Python，不过门禁**。

### 3.1 格式说明文件

**把「这堆数字怎么读」从代码里抄出来，写成一个 JSON 文件。**

里面存：tag 的位域怎么切、槽位公式、20 个 task 的 id 和名字、7 个 track 各是哪个 warp，
加上三份源文件的 sha256。

没有它，「怎么读」这个知识只存在于源码里或某个人脑子里，每次都得有人重读代码。
有了它，解码器读 JSON 就知道怎么解 —— **它是解码器和源码之间的唯一接口**。

**存在**：有 `*_abi.json`，能解析，含 source 溯源字段，
且该字段能解析到本次命中时钟源的那个 profiler 源文件。
仅仅「有个 layout json」不算 —— 本仓库里从别处复制来的 `gptrace4-layout.json`
就没有任何 source 声明字段，判为 uncoupled。

**失效**：源码漂移了不报警 → 零介入失去唯一的安全网。

**有效**（必须是反向的）：
```
5 个负例必须全部 exit 1
   改坏 source_sha256 / 删一个 task / 改 tag 位移 17->16 /
   加多余键 $.decoder_sha256 / 改 source_file 指向别处
--write 重新生成的 JSON 与现有字节一致
   证明抽取确定性, 且现有 JSON 真是抽出来的, 不是手写后漂移过的
```

**正向跑通不算证据**，因为这一项曾经假通过 —— 详见第 5 节那个洞。

### 3.2 解码器

数字 → 事件列表。

**存在**：溯源链闭合 ——
```
候选解码器源码中的 .json 路径字面量
  -> 解析得到 ABI JSON J
  -> 读 J 的 source_file / source / source_path / device_source / sources
  -> 解析后必须落在 profiler 锚点集内
     (锚点集 = 设备扩展名 + 命中时钟源 + 有 profiler 身份证据)
```

**不能用位移匹配**：一个写得好的解码器由 ABI JSON 驱动、不在代码里二次硬编码位域，
于是源码里既没有 `>> 20` 也没有裸位运算 —— 位移判据会漏判好设计。
溯源判据更强，且不惩罚好设计。

**失效**：选中解错 ABI 的解码器 → **输出貌似合理的垃圾数字而不是报错**，
比直接失败危险得多。本仓库里躺着三个文件名含 `decode` 的干扰项
（解 GPTRACE4 的 128B/48B、解 GPUTIML1 的 64B/32B），
而本 kernel 是裸 int64 的 (timestamp, tag) 对。它们被排除的理由可核查：
前两个源码里根本没有 `.json` 路径字面量，第一环就没有；
`gptrace4_host.py:13` 引用的 layout JSON 不含任何 source 声明字段，第二环断掉。

**有效**，分两组：
```
保真(数据搬运没出错)
   解出事件数 == buffer 非空槽位数     既不丢也不造
   unmatched_starts == 0
   unmatched_ends == 0
   stray_slots_after_gap == 0         offset 必须连续

正确(ABI 解读对了)
   tag_slot_mismatch == 0             <- 跨类判据
   unknown_task_ids == 0
   tag_high_bits_set == 0             未使用位必须为 0
   negative_durations == 0            end_ns >= start_ns
   zero_timestamp_with_nonzero_tag == 0   tag 非零则时间戳不能为 0(0 是空槽哨兵)
```

`tag_slot_mismatch` 是**跨类判据**：它同时验证 device 侧槽位公式（2.2）和 host 侧解码器，
因为它要求两者对同一份数据的理解一致。device 写对了而解码器读错了，也会 mismatch。

**本项目**：`decode_megamoe_profiler.py:11 -> megamoe_profiler_abi.json -> source_file
-> mega_moe_profiler.cuh`，链条闭合。实测 152933 事件、9 项异常全为 0。

### 3.3 出图器

事件列表 → 能看的甘特图。

**存在**：与某个已 coupled 的解码器共享 ≥2 个事件 schema 字段名，
**且共享集合必须含事件容器键**（`events` / `traceEvents`）。
容器键这条是必要的 —— 只共享 `start_ns`/`end_ns` 会把
`gate/validate_perfetto_artifact.py` 这类**校验器**误认成 emitter。
另外要声明它遵守哪个硬件 profile 的泳道契约。

**失效**：schema 不匹配跑不出图；或者图跑出来了但**泳道分错，看图的人被误导**。

**有效 —— 保真组**（数据搬运没出错，但这只是保真不是正确）：
```
输出能 json.load
ph == "X" 的数量 == events.json 事件数
dur * 1000 == dur_ns 零例外        纳秒精度没丢
```

**有效 —— 泳道契约组，硬件无关**：
```
R3    一个物理 warp 一条轨道；warp 模式下不得出现 role/task/lane 派生的泳道
      同一 lane 的全部 ph=X 只能 disjoint 或 nested，partial overlap 硬 FAIL
      nested 时必须是粗粒度 task 外层、真实 warp phase 内层
      task 结束后才开始的 phase 可独立存在；不得延长 task 强行包含
R6    counter 进程的 pid 必须严格小于所有软件泳道的 pid
      rank_sync 轨道必须排在最前(rank_sync_first)
身份   physical_sm 泳道禁止携带 cta_id/cluster_id/cluster_rank
      rank_sync/kernel/device_counter 泳道禁止携带 SM/CTA/cluster/warp/role 身份
溯源   禁止根据 CTA 编号、cluster rank 或调度顺序猜测物理 SMID
      泳道名必须能溯源到真实 physical_smid
拓扑   泳道数 == rank × CTA × 有效 track
重叠   同泳道内无非法重叠(允许嵌套, 不允许部分交叠)
时间轴 所有带 ts 的 timed event 必须 finite 且 ts>=0；合法 HW pre-roll 只能对
       全事件做一次统一 display rebase，禁止逐点 clamp/drop/scale
```

**有效 —— 泳道契约组，硬件相关（按 profile 分派）**：
```
R2  仅 B300/SM100: 若 kernel 使用 2-CTA cluster, 必须把同一 CTA-pair 合并分组,
    且 leader/follower 有序。身份来自 cluster_id + cluster_rank, 两者必须同时存在,
    不得从 CTA 编号推测。process 按 (rank, device, cluster_id) 分组,
    校验项 cta_pair_complete
    -> 拆成两条泳道会让人误以为是两个独立执行单元, 图会误导

R5  多 warp 协作只实测 observer warp 时，禁止复制 X 事件；其余物理 warp
    只用 topology metadata 建空泳道。需要逐 warp 实测时回 device 加打点
```

**跨来源印证**（最强的一条，因为它是完全独立的第二条测量链路）：
```
解码算出的 makespan 2836 us  vs  kineto 独立测得 kernel 时间 2704 us
```

**本项目的历史产物**只用于数据保真示例：`ph=X` 数必须等于解码事件数，
`dur*1000 != dur_ns` 必须为零。历史的 5 observer-track 视图不能当成完整物理 warp
拓扑，更不能据此复制区间；新产物按目标 kernel 的 rank × CTA × physical-warp metadata 验收。

---

## 4. 为什么有效性能用代码判，而不是大模型

这是整个方案能成立的关键。不变量只有两种形态，
**两种都不需要知道「正确答案是什么」**。

### 形态一：自洽性检查，靠冗余

同一个信息被编码两次，两个来源必须相等。最典型的是 `tag_slot_mismatch`：
槽位公式能从数组下标反算出 `(block, track)`，而 tag 的 `[31:20]`、`[19:17]`
里又独立编码了一遍。位域切错一位、乘子记错、`max_records` 传错，立刻分道扬镳。
**纯计算，零判断。**

同类：`unmatched_starts/ends == 0`、`negative_durations == 0`、
`tag_high_bits_set == 0`、`stray_slots_after_gap == 0`、
`truncated_block_tracks == 0`、`ctas_missing_anchor == 0`、
以及第 2.6 节的 layer、observer 与源码一致性检查。

### 形态二：独立来源交叉印证

```
解出事件数 == buffer 非空槽位数     两个独立计数
解码 makespan ~= kineto 时间       两条不相干的测量链路   2836 vs 2704 us
改坏 ABI JSON 必须 exit 1          反向验证, 只看退出码
小 shape 相对误差 ~= 大 shape       0.03357 vs 0.03412
empty_tracks == [5,6]              源码注释的预言 vs 实测
```

### 反面才是重点

如果验证需要大模型看着 timeline 判断「这图对不对」，**方案就废了** ——
那退回到不可扩展的人工判断，而且人根本看不出 152933 条事件里哪条位域错了。

所以判据的设计标准只有一条：**必须能用代码判成二值。判不了的就不要当判据。**
覆盖率、间隙、粒度那些阈值全是编的，所以它们**只进报告，不进门**。

本次实证：9 项异常全为 0 完全是脚本算出来的，我一条事件都没看，
也没让任何 agent「审阅数据质量」。而这 9 个 0 就是 ABI 解读正确的证据 ——
位域推错一位它们不可能同时为 0。

---

## 5. 快/慢两路径（慢路径两子类）与大模型入口

```
fast / complete                   不改代码，直接采集与验证
slow / host-bootstrap  只生成/修复 ABI、decoder、emitter、layout contract；
                                不读写 kernel 主体
slow / device-or-semantics        device 原语缺失，或 task/phase 真实边界缺失/错误；
                                读门禁后才修改 kernel
```

第 3 步不变量失败是独立的诊断入口：只检查失败项对应的组件；如果失败项是
host 生成/复制/延长 task 或 partial overlap，先删除这种后处理行为，不能把它包装成
“补全语义”。
其余全是脚本。**每条有效判据都对应一个具体失效模式**，
所以挂了就知道修哪个组件 —— 这是「只看挂掉那条」能成立的前提：

```
tag_slot_mismatch      专抓槽位竞争与地址算错
时间戳单调性            专抓时钟源读错
layer/observer 一致性    专抓语义层或记录 warp 归属错误
truncated_block_tracks 专抓 buffer 开小
dur*1000 == dur_ns     专抓出图器精度损失
R2                      专抓 2-SM 泳道拆错
```

进入 `slow / device-or-semantics` 前必须先读门禁探测结果（附录 A3），确认目标文件是否 immutable
或落在 SASS 镜像树里，否则写完必被打回。

修好后**必须回填**该项目的 ABI JSON 与抽取脚本。
这一步是整套方案能越用越快的前提，不可跳过。

---

## 6. 东西放哪

### 6.1 代码归属

规则一条：**换个 kernel 还能不能用 —— 能用的进 skill，不能用的留项目。**

```
项目代码(不可能放 profile/, 要被 nvcc 编译进 kernel)
  infra/deep_gemm/include/deep_gemm/profiler/mega_moe_profiler.cuh   2.1 2.2 2.3
  fixed/kernel.cuh:169-184                                          track 绑定
  fixed/kernel.cuh:192-220                                          2.5 entry rendezvous
  opt_version/v0/operations/*.cu                                    2.6 各 phase 插桩点
  infra/csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp:424-439         2.4
  infra/deep_gemm/mega/__init__.py:202-234                           2.4 Python 入口

new_struct_mega_moe/profile/        懂本 kernel, 换 kernel 就得换
  abi/megamoe_profiler_abi.json       3.1 格式说明文件
  scripts/extract_abi.py              抽取 + 校验(正则要匹配本项目源码结构)
  scripts/decode_megamoe_profiler.py  3.2 解码器
  4_validate/harness/run_profile.py              采集(要调本项目 API、构造本项目输入)

skill/fast-instrument-v2/           不懂任何具体 kernel, 跨项目通用
  1_scan/                         探测器与 layout contract
  3_capture/build_perfetto.py       3.3 出图器（只吃 events.json，不碰位域）
  3_capture/knowledge/templates/              ABI/解码器/抽取器/harness 参考实现
  4_validate/                       capability、deliver gate 与 selftests
  4_validate/repair/                         慢路径修复路由
  SKILL.md / knowledge/examples/demo/ 当前判据与方法论；_audit/legacy_REPORT.md 仅作历史审计
```

**为什么解码器和出图器分家**：
```
解码器  要知道 tag 的 [31:20] 是 block、槽位公式乘子是 7   -> 换 kernel 全变  -> 留项目
出图器  只看 events.json 里的 start_ns/dur_ns/block/track  -> 换 kernel 不变  -> 进 skill
```

**类 1 贵、类 2 便宜的根本原因不是代码量，是门禁**：`fixed/` immutable、
`opt_version/v0/` 只能是那十个 `.cu`（多一个文件门禁就挂）、
`infra/` 还受 SASS 逐字节等价约束（要同步改仓库外那份 2183 行 monolithic）。
而 `profile/` 是 free zone。

### 6.2 产物归属

一次采集实测产 442 MB，其中 250 MB 是 5 个未压缩 trace（gzip 后各仅 3.5 MB）；
`bundle` 含 `hw` 时 nsys-rep 是 GB 级。必须分三档。

```
A 大中间件 -> .work/profile_runs/<run_id>/       必须在 .gitignore 里
    raw_rank*.pt           8.9 MB/rank   留档(重新解码用), 只保留最近若干次
    events_rank*.json      32 MB/rank    出图后即可删
    trace*.perfetto.json   50 MB         禁止产未压缩版
    *.nsys-rep / *.sqlite  GB 级         验完即删

B 小结论 -> infra/results/ 或项目等价的 results 目录   受版本管理, 单文件 < 1 MB
    correctness.json  overhead.json  coverage_report.json
    INSTRUMENTATION.json  GATES.json
    绝不放 raw buffer / events.json / trace / nsys-rep

C 最终交付 -> trace.perfetto.json.gz  约 3.5 MB
```

硬规定：出图器默认只输出 `.gz`；nsys 证据提取完立刻删 `.nsys-rep`；
写 `results/` 前检查单文件 < 1 MB，超了说明放错档；
临时产物不进 `/tmp`（项目规定）也不写仓库根目录；
探测器与自测的 fixture 必须被探测器自己的排除列表覆盖（附录 A5）。

---

## 7. 安全网：零介入凭什么安全

判定通过后就不再读代码。源码变了怎么办？

`extract_abi.py --check` 对三份文件各存一个 sha256：profiler 源码、
`fixed/kernel.cuh`（track 绑定与 entry rendezvous）、host `.hpp`（buffer 契约）。
任一变更即 `exit 1` 并点名不一致项。实测五个负例 rc 依次为 0/1/1/1/1。

**这网曾有一个洞。** 过滤 sha256 重复项时用了 path **子串**匹配：

```python
diffs = [d for d in diff_json(expected, actual) if "sha256" not in d.split(":")[0]]
```

于是任意深度里 path 含 `sha256` 的不一致被**静默丢弃** ——
不打印、不计入退出码。加一个 `$.decoder_sha256` 键、
改嵌套的 `$.sync_anchors.entry_rendezvous.source_sha256`，修复前都是 `OK + exit 0`。

修法：去重改为按精确 path 集合抑制；退出码与结论**同源于同一个被打印的 failures 列表**；
OK 与 FAIL 都走 stdout（原先 OK 走 stdout、FAIL 走 stderr，
正是「打印 OK 却非零退出」这类矛盾的成因）。

**网破了洞零介入就不成立，所以这条是硬性的。**
也正因如此，3.1 的有效判据必须是反向的 —— 正向跑通不算证据。

---
---

# 附录 A：五个陷阱（都是实测踩到的）

## A1 `%globaltimer` 跨 GPU 不是同一时钟域

实测两块 H20 相差 **3.6168 秒**（`3616875232 ns`）。单卡内所有 SM 同源，**跨卡不同源**。

后果：`min(release) >= max(arrive)` 在原始时间戳上**必然** VIOLATION
（实测 margin `-3616819808 ns`），不知情会误判 kernel 有 bug；
单一全局平移会让 makespan 变成 3.6 秒完全不可用。

正解：按 rank 各自对齐。同 device 时钟内实测 rank0 margin=55424 ns、
rank1 margin=57792 ns，都 PASS。

上游 skill 的 `knowledge/examples/demo/theory/rank-sync-and-time.md` 声称 globaltimer
「天然同源、不需额外校正」—— 同卡内成立，跨卡不成立。

## A2 baseline 缺 rank sync 时噪声压过信号

实测同一份代码两次运行得 `0.93%/3.12%` 与 `-1.85%/4.70%`，计时逻辑一个字没改。
**噪声跨度 6.5 个百分点甚至出现负值，而信号只有约 1%。**

正确口径是三档测量：

```
档1  anchor_off   profiler_buf=None    无 rank sync 无记录       -> T_none
档2  anchor_only  max_records=2        有 rank sync 记录约 0.7%  -> T_anchor, S_anchor
档3  full         max_records=正常值    有 rank sync 完整记录     -> T_full,   S_full

pure_record_overhead_pct = ((T_full - S_full) - (T_anchor - S_anchor))
                           / (T_anchor - S_anchor) * 100        <- 主指标
legacy_overhead_pct      = (T_full - T_none) / T_none * 100     <- 旧口径, 仅对照
```

**档 2 为何成立且零 kernel 改动**：rank sync 锚点是每个 CTA 的第一个事件，
而 profiler 满即丢弃不回绕（`offset >= max_records` 直接 return）。
所以 `max_records=2` 时锚点 track 恰好记满 start+end，其余 track 只记前 2 条，
总量从 154296 条降到 `78×7×2 = 1092` 条（0.7%）；
而 buffer 非空使插桩开关为真，**rank sync 照常执行**。

禁止只报一个未说明口径的开销百分比。

## A3 SASS 镜像树：会让整个修补方向白做

`compile_equivalence` 用同一个 harness 分别以两个 include root 编 cubin，
要求 **SASS 逐字节相同** + ptxas 资源数相同：

```
infra/deep_gemm/include        <->   ../code/deep_gemm/include   (在仓库之外)
```

而 `../code/.../sm90_fp8_mega_moe.cuh` 是 **2183 行 monolithic**，
workspace 的 `fixed/kernel.cuh`(772 行) + `opt_version/v0/` 十个 `.cu`
是它的**重构拆分版**。

所以「扩 tag 位域到 per-warp」不是「两边改一样的代码」，
而是要在**两份结构完全不同的文件**里做语义等价改动并保持 SASS 一致。
**动手前必须先看门禁探测结果**，否则会规划出注定失败的方案。

门禁探测输出六类可写区：
```
free           可自由新建/修改        .work/  profile/
gated          受文件清单约束         opt_version/v0/ 必须恰好那十个 .cu
immutable      碰了必挂门禁           fixed/  opt_version/vN/  infra/verifier/
outside_repo   需额外授权             ../code/deep_gemm/include
artifacts      临时产物唯一合法位置    .work/  (禁止 /tmp)
mirrored_trees 动一侧必须同步另一侧    上面那对, sass_byte_identical
```

## A4 容差从小 shape 继承到大 shape

误差来自 L1 中间激活 FP8 e4m3 量化噪声经 L2 GEMM 传播，
是**正比于输出量级的绝对误差**（相对误差恒为 3.4%），而 `atol` 是常数：

```
官方小 shape   ref_rms ~= 1.48       atol=2.0 ~= 1.35  * ref_rms  -> atol 吃掉全部误差
production    ref_rms ~= 173..177   atol=2.0 ~= 0.011 * ref_rms  -> 等效收紧约 118 倍
```

于是 6.8% 元素违规，而 kernel 完全正确（`calc_diff` 5.8e-4 对项目自身门槛 0.07
有 120 倍余量，每个 expert 的 `cum_stats` 零 mismatch）。

三条独立证据定位到判据而非 kernel：官方 test 在同形状同容差下**同样 FAIL**
（violations 15576/15615 vs harness 15588/15547）；
官方小形状虽 PASS 但其 `fro_rel=0.03357` 与 production 的 `0.03412` **几乎相同**；
用高斯噪声模型预测违规率 6.82~6.88% 对实测 6.78~6.81%，**吻合到 0.1 个百分点**
（局部算错会产生非高斯的集中误差，预测必然偏离）。

修法 `atol_eff = max(floor, 0.5 * rms(y_ref))`。
**修完必须自证只改判据未动数值** —— 实测 `calc_diff` / `max_abs` / `mean_abs` /
`frobenius_relative` 修复前后逐位相同，仅 violations 由 15588 变 0。

由此得一条元判据：**判据自身也要有不变量。**
正确的正确性不变量不是 `violations == 0`，
而是「同一 kernel 在小 shape 与大 shape 下相对误差应当一致」。

## A5 探测器自污染

测试 fixture 若放在被扫描路径内，其常量会泄漏进真实结论。
实测 `profile/.selftest/` 里的 `ALLOWED_TOP` 曾污染真实仓库的 `write_zones`。
探测器必须默认排除自己的草稿区、参考文档目录（`refs`/`references`/`docs`）、
以及从上游复制来的工具链目录。

---

# 附录 B：本项目实测与待办

```
探测   1.6 s 扫 237 文件
采集   81 s EP2 双 rank, raw 8.9 MB x 2
解码   152933 事件(77148 + 75785), 9 项异常全为 0
出图   3.5 MB gzip, 780 条历史 observer 泳道, 17 种历史 event name
rank sync   78 CTA, start_skew 544/512 ns, end_skew 192 ns
时钟域      dev0 与 dev1 相差 3616875232 ns
makespan    解码 2836 us  vs  kineto 2704 us
开销        0.93%/3.12% 与 -1.85%/4.70% (口径待修正为三档)
正确性      calc_diff 5.8e-4 (门槛 0.07), cum_stats mismatch 0
```

已知问题：

```
2.5 rank sync         只在插桩构型有 -> 按本规格这一项 FAIL
2.6 历史 catalog     旧口径把 wait/barrier 计作可见 phase；新口径归为 ignored，
                      不再用 20 - 1 - 2 = 17 推断 task/phase 语义完整性
拓扑                  历史 5 observer-track 只代表真实记录者，不得复制成全 warp 事件
```

待办：

```
1  2.5 两项合并为一项且变阻塞项  -> 本项目 verdict 会从 complete 变 FAIL
2  2.6 加进探测器                layer + 真实边界 + observer 归属检测
3  第 0 步三个必填参数 + discover 列候选   现在探测器只有 --root
4  3.3 判据换成泳道契约, 按硬件 profile 分派
5  build_perfetto 的 --clock-domain 回归  makespan 显示 3.6 s 而非 2836 us
6  三档开销测量落地
```
