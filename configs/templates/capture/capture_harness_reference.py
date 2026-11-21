#!/usr/bin/env python3
"""项目侧硬件采集窗口模板。

这里没有假设任何通用 kernel，也不会自行创建 CUDA tensor。调用方必须传入本项目
自己的 callable：warmup、基于 randn 的全量正确性检查、目标 kernel，以及采集器的
Start/Stop 和 rank barrier。warmup 与正确性检查在采集窗口外执行；窗口内目标 kernel
只调用一次。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


LIFECYCLE_SCHEMA = 'rank-lifecycle/v1'
EventFn = Callable[[], Any]


def _now_ns() -> int:
    return time.monotonic_ns()


def _record(events: List[Dict[str, Any]], name: str) -> None:
    events.append({'name': name, 'time_ns': _now_ns()})


def run_warmup_and_correctness(warmup: EventFn, correctness: EventFn) -> None:
    """在 Start 之前完成 warmup 和 randn 全量正确性检查。

    correctness 必须由项目传入的 callable 完成真实的 randn 输入、完整输出比较和
    失败抛错；本模板不伪造通用 reference，也不只检查几个元素。
    """
    warmup()
    correctness()


def capture_target_once(rank: int,
                        warmup: EventFn,
                        correctness: EventFn,
                        target_kernel: EventFn,
                        start_capture: EventFn,
                        stop_capture: EventFn,
                        barrier: EventFn,
                        lifecycle_path: Path,
                        metadata: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
    """执行一次 rank 的采集窗口，并写出 rank_lifecycle.json。

    事件顺序严格为 Start -> barrier -> target_kernel_once -> barrier -> Stop。
    start_capture/stop_capture 应绑定项目实际的采集 API（例如 cudaProfilerStart/Stop
    或外部采集器控制）；barrier 应绑定项目的 torch.distributed barrier 或等价实现。
    """
    if not isinstance(rank, int) or rank < 0:
        raise ValueError('rank 必须是非负整数')
    events: List[Dict[str, Any]] = []
    document: Dict[str, Any] = {
        'schema': LIFECYCLE_SCHEMA,
        'rank': rank,
        'status': 'running',
        'target_kernel_calls': 0,
        'events': events,
    }
    if metadata:
        document['metadata'] = dict(metadata)

    started = False
    try:
        run_warmup_and_correctness(warmup, correctness)
        start_capture()
        started = True
        _record(events, 'Start')

        barrier()
        _record(events, 'barrier_after_start')

        target_kernel()
        document['target_kernel_calls'] = 1
        _record(events, 'target_kernel_once')

        barrier()
        _record(events, 'barrier_before_stop')
        stop_capture()
        started = False
        _record(events, 'Stop')
    except BaseException as error:
        # 只有真正异常时才写 error；正常文档没有“残留中的异常/采集状态”字段。
        document['status'] = 'failed'
        document['error'] = '{}: {}'.format(type(error).__name__, error)
        if started:
            try:
                stop_capture()
                _record(events, 'Stop_after_error')
            except BaseException as stop_error:
                document['stop_error'] = '{}: {}'.format(type(stop_error).__name__, stop_error)
        _write_lifecycle(lifecycle_path, document)
        raise

    document['status'] = 'completed'
    document['start_stop_balanced'] = True
    document['warmup_outside_window'] = True
    document['correctness_outside_window'] = True
    _write_lifecycle(lifecycle_path, document)
    return document


def _write_lifecycle(path: Path, document: Dict[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def lifecycle_schema_example() -> Dict[str, Any]:
    """返回正常退出时的 JSON 结构示例，便于项目实现自己的 writer/validator。"""
    return {
        'schema': LIFECYCLE_SCHEMA,
        'rank': 0,
        'status': 'completed',
        'target_kernel_calls': 1,
        'events': [
            {'name': 'Start', 'time_ns': 0},
            {'name': 'barrier_after_start', 'time_ns': 1},
            {'name': 'target_kernel_once', 'time_ns': 2},
            {'name': 'barrier_before_stop', 'time_ns': 3},
            {'name': 'Stop', 'time_ns': 4},
        ],
        'start_stop_balanced': True,
        'warmup_outside_window': True,
        'correctness_outside_window': True,
    }
