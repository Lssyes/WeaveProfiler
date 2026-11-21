# 慢路径：插桩缺失时才走这里

只有快/慢路由判为 `slow / device-or-semantics` 时才进入本文件：device 时钟/槽位/
buffer/sync 等原语缺失，或 task/phase 的真实语义边界缺失、错误。
若 device 与语义完整、只缺 ABI/decoder/emitter/contract，应走
`slow / host-bootstrap`，禁止在这里改 kernel。

完整方法论在这个 skill 之外，不要复制过来，直接读：

```
$PROFILE_GPU_KERNEL_TIMELINE/SKILL.md
```

那份文档约 29KB，包含语义 catalog、fixed-slot 物化、单次 NSYS 采集时序、
Perfetto 渲染合同、独立审计与 finalize 门禁。它的 device 侧真源与硬件后端在：

```
$PROFILE_GPU_KERNEL_TIMELINE/assets/instrumentation/gpu_timeline_fixed_slot.cuh
$PROFILE_GPU_KERNEL_TIMELINE/assets/instrumentation/counted-rank-sync.cuh
$PROFILE_GPU_KERNEL_TIMELINE/backends/hardware/generic_sm_90/runtime/gpu_timeline_hw.cuh
$PROFILE_GPU_KERNEL_TIMELINE/backends/hardware/b300/runtime/gpu_timeline_hw.cuh
```

按 `backends/hardware/<profile>.json` 选后端，SM90 用 `generic_sm_90`（`sm_90a`、
`%globaltimer`、fixed-slot ABI、开销上限 10%）。

`assets/instrumentation/gpu_timeline_trace.cuh` 是被门禁明令禁止的旧
GPUTIML1 staging 路径，不要引用。

## 语义硬约束

- `task` 是 device 真实打点的粗粒度 operator/GEMM；`phase` 是真实
  load/MMA/epilogue 等 warp 阶段；wait/barrier/sync 默认是 `ignored`。
- 后处理禁止生成、复制或延长 task。observer 事件只属于实际记录的 warp；
  未观测 warp 只用 topology metadata 表示。
- 同 lane 的 `ph=X` 只允许 disjoint/nested，partial overlap 硬 FAIL；
  phase 在 task 结束后开始时可独立存在，不得延长 task 强行包含。

## 读之前先做的两件事

1. 读 `GATES.json`，确认插桩点所在文件是否 `immutable` 或落在 `mirrored_trees` 里。
   如果是，先和用户确认授权，否则写完必被门禁打回。
2. 只读探测器已经定位到的行号附近，不做全文通读。

## 插桩落地后必须回填

新插桩跑通后，产出该项目的 ABI JSON 与 `extract_abi.py`。若 host 链也齐全，
下一次应路由为 `fast / complete`；若只剩 host 产物缺失，则路由为
`slow / host-bootstrap`。这一步是整套方案能越用越快的前提，不可跳过。
