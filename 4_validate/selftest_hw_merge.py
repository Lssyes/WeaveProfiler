#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROFILE_DIR = SCRIPT_DIR.parent / '3_capture'
CAPTURE = PROFILE_DIR / 'capture_nsys.py'
EXTRACT = PROFILE_DIR / 'extract_gpu_metrics.py'
MERGE = PROFILE_DIR / 'merge_hw_into_perfetto.py'

INTERVAL_NS = 5000
KERNEL_NEEDLE = 'sm90_fp8_mega_moe_impl'
METRICS = [
    (0, 'DRAM Total Throughput [Throughput %]'),
    (1, 'SM FMA Pipe Throughput [Throughput %]'),
    (2, 'Tensor Active [Throughput %]'),
    (3, 'NVLink RX Requests User Data [Throughput %]'),
    (4, 'NVLink RX Responses User Data [Throughput %]'),
    (5, 'NVLink TX Requests User Data [Throughput %]'),
    (6, 'NVLink TX Responses User Data [Throughput %]'),
    (7, 'SMs Active [Throughput %]'),
    (8, 'SM Issue [Throughput %]'),
]
# nsys 时间基准（ns）的两次目标 kernel launch；launch1 的 span 更接近 sw_span
LAUNCH0 = (1_000_000, 1_000_000 + 2_800_000)
LAUNCH1 = (5_000_000, 5_000_000 + 2_834_000)
GPU_METRICS_RANGE = (500_000, 8_500_000)
# 软件（device %globaltimer 原值）窗口，span 对齐需求实测 makespan 2834048 ns
SW_T0 = 1787113039011611200
SW_SPAN = 2834048
# Visible SW uses entry_release as zero; raw events include a legal 60 us kernel pre-roll.
SHIFT_NS = SW_T0 + 60000


class Failure(Exception):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise Failure(msg)
    print(f'  [ok] {msg}')


def build_mini_sqlite(path: Path) -> Dict[str, int]:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute('CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)')
    cur.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL '
                '(correlationId INTEGER, deviceId INTEGER, start INTEGER, end INTEGER, '
                'demangledName INTEGER, shortName INTEGER)')
    cur.execute('CREATE TABLE TARGET_INFO_GPU_METRICS '
                '(typeId INTEGER, sourceId INTEGER, typeName TEXT, metricId INTEGER, metricName TEXT)')
    cur.execute('CREATE TABLE GPU_METRICS '
                '(rawTimestamp INTEGER, timestamp INTEGER, typeId INTEGER, metricId INTEGER, value INTEGER)')

    target_name = f'void {KERNEL_NEEDLE}<128,256>(params)'
    other_name = 'void some_other_kernel<float>(int)'
    cur.execute('INSERT INTO StringIds VALUES (?,?)', (1, target_name))
    cur.execute('INSERT INTO StringIds VALUES (?,?)', (2, other_name))

    type_id = 281479271677952
    cur.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?)',
                (101, 0, LAUNCH0[0], LAUNCH0[1], 1, 1))
    cur.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?)',
                (102, 0, LAUNCH1[0], LAUNCH1[1], 1, 1))
    # 一个别的 kernel（不应匹配）+ 一个 device 1 的目标 kernel（device 过滤应排除）
    cur.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?)',
                (103, 0, 200_000, 300_000, 2, 2))
    cur.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?)',
                (104, 1, LAUNCH0[0], LAUNCH0[1], 1, 1))

    for mid, name in METRICS:
        cur.execute('INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?,?,?,?,?)',
                    (type_id, type_id, '', mid, name))
    # nsys 内部 EndTimestamp：不同 typeId、metricId 与 SMs Active 撞（这里撞 METRICS[0]），应被排除
    end_type_id = type_id + 256
    cur.execute('INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?,?,?,?,?)',
                (end_type_id, type_id, '', 0, 'EndTimestamp'))
    for k in range(5):
        cur.execute('INSERT INTO GPU_METRICS VALUES (?,?,?,?,?)',
                    (LAUNCH1[0] + k * INTERVAL_NS, LAUNCH1[0] + k * INTERVAL_NS,
                     end_type_id, 0, 999))

    total_rows = 0
    ts = GPU_METRICS_RANGE[0]
    while ts <= GPU_METRICS_RANGE[1]:
        for mid, _name in METRICS:
            value = int((ts // INTERVAL_NS + mid * 7) % 101)
            cur.execute('INSERT INTO GPU_METRICS VALUES (?,?,?,?,?)',
                        (ts, ts, type_id, mid, value))
            total_rows += 1
        ts += INTERVAL_NS
    conn.commit()
    conn.close()
    return {'gpu_metrics_rows': total_rows}


def write_manifest(path: Path) -> None:
    path.write_text(json.dumps({
        'schema': 'nsys-hw-capture/v1',
        'gpu_metrics_device': 0,
        'metrics_set': 'file:selftest-gh100-hbm-fma-alu.config',
        'frequency_hz': 200000,
    }, ensure_ascii=False), encoding='utf-8')


def build_software_trace(path: Path) -> Dict[str, int]:
    events: List[Dict[str, Any]] = [
        {'name': 'process_name', 'ph': 'M', 'pid': 1, 'tid': 0,
         'args': {'name': 'rank0 (dev0)'}},
        {'name': 'process_sort_index', 'ph': 'M', 'pid': 1, 'tid': 0,
         'args': {'sort_index': 0}},
        {'name': 'thread_name', 'ph': 'M', 'pid': 1, 'tid': 1,
         'args': {'name': 'CTA0/SM64 track0 dispatch'}},
    ]
    x_count = 40
    visible_span = SW_SPAN - (SHIFT_NS - SW_T0)
    for i in range(x_count):
        raw = SHIFT_NS + int(i * visible_span / x_count)
        events.append({'name': 'phase', 'cat': 'math', 'ph': 'X', 'pid': 1, 'tid': 1,
                       'ts': (raw - SHIFT_NS) / 1000.0, 'dur': 1.0,
                       'args': {'block': 0, 'start_ns_raw': raw}})
    doc = {'traceEvents': events, 'displayTimeUnit': 'ns',
           'otherData': {'producer': 'selftest', 'shift_ns': SHIFT_NS, 'align': 'entry_release'}}
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        json.dump(doc, handle, ensure_ascii=False)
    return {'x_count': x_count}


def build_events(path: Path, rank: int = 0, device: int = 0,
                 t0: int = SW_T0, span: int = SW_SPAN) -> None:
    events = []
    n = 30
    for i in range(n):
        start = t0 + int(i * span / n)
        end = start + 500
        events.append({'block': 0, 'track': 3, 'role': 'math',
                       'start_ns': start, 'end_ns': end})
    events[0]['start_ns'] = t0
    events[-1]['end_ns'] = t0 + span
    path.write_text(json.dumps({
        'metadata': {'rank': rank, 'device': device},
        'events': events,
    }, ensure_ascii=False), encoding='utf-8')


def build_per_rank_software_trace(src: Path, dst: Path) -> None:
    with gzip.open(src, 'rt', encoding='utf-8') as handle:
        doc = json.load(handle)
    doc['otherData']['shift_ns'] = None
    doc['otherData']['shift_ns_by_rank'] = {'0': SHIFT_NS, '1': SHIFT_NS + 123456}
    doc['otherData']['alignment_mode'] = 'per_rank'
    doc['traceEvents'].extend([
        {'name': 'process_name', 'ph': 'M', 'pid': 2, 'tid': 0,
         'args': {'name': 'rank1 (dev1)'}},
        {'name': 'process_sort_index', 'ph': 'M', 'pid': 2, 'tid': 0,
         'args': {'sort_index': 1}},
    ])
    with gzip.open(dst, 'wt', encoding='utf-8') as handle:
        json.dump(doc, handle, ensure_ascii=False)


def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def expected_samples_in(lo: int, hi: int) -> int:
    count = 0
    ts = GPU_METRICS_RANGE[0]
    while ts <= GPU_METRICS_RANGE[1]:
        if lo <= ts <= hi:
            count += 1
        ts += INTERVAL_NS
    return count


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description='fast-instrument 硬件 counter 三脚本自测（合成数据，无需 GPU）')
    parser.add_argument('--work-dir', default=None,
                        help='临时工作目录，默认当前目录下 .selftest_hw_merge（结束自动清理）')
    parser.add_argument('--keep', action='store_true', help='保留工作目录用于排查')
    args = parser.parse_args(argv)

    work = Path(args.work_dir).expanduser().resolve() if args.work_dir else \
        (Path.cwd() / '.selftest_hw_merge').resolve()
    work.mkdir(parents=True, exist_ok=True)
    failures: List[str] = []
    try:
        print('[1] capture_nsys.py --help 秒回')
        t0 = time.time()
        proc = run([sys.executable, '-B', str(CAPTURE), '--help'])
        dt = time.time() - t0
        check(proc.returncode == 0, f'--help 退出码 0（实际 {proc.returncode}）')
        check(dt < 5.0, f'--help 用时 {dt:.2f}s < 5s（防 import 期阻塞回归）')

        print('[2] extract_gpu_metrics.py 合成 sqlite 提取')
        sqlite_path = work / 'mini.sqlite'
        info = build_mini_sqlite(sqlite_path)
        write_manifest(work / 'capture_manifest.json')
        hw_path = work / 'hw_metrics.json'
        proc = run([sys.executable, '-B', str(EXTRACT), '--sqlite', str(sqlite_path),
                    '--kernel-name-contains', KERNEL_NEEDLE, '--out', str(hw_path),
                    '--keep-all'])
        check(proc.returncode == 0, f'extract 退出码 0（实际 {proc.returncode}）\n{proc.stdout}\n{proc.stderr}')
        hw = json.loads(hw_path.read_text())
        check(hw['schema'] == 'gpu-hw-metrics/v1', 'hw schema 正确')
        check(hw['device'] == 0, 'device 从 manifest 读到 0')
        check(len(hw['kernel_launches']) == 2,
              f'目标 kernel launch 数 == 2（device 0，排除 device1 与他核；实际 {len(hw["kernel_launches"])}）')
        check(len(hw['counters']) == len(METRICS), f'counter 数 == {len(METRICS)}')
        check(all(c['name'] != 'EndTimestamp' for c in hw['counters']),
              'EndTimestamp 伪 metric（不同 typeId）被正确排除')
        check(hw['sampling_interval_ns']['median'] == INTERVAL_NS,
              f'采样中位间隔 == {INTERVAL_NS} ns（实际 {hw["sampling_interval_ns"]["median"]}）')
        before = hw['clip_report']['samples_before_clip']
        after = hw['clip_report']['samples_after_clip']
        check(before == info['gpu_metrics_rows'], f'裁剪前样本数 == 全表 {info["gpu_metrics_rows"]}')
        check(after < before, f'裁剪后 {after} < 裁剪前 {before}（窗口外样本被剔除）')
        names = {c['name'] for c in hw['counters']}
        check('DRAM Total Throughput' in names, 'counter 名解析出 "DRAM Total Throughput"')
        check(all(c['unit'] == 'Throughput %' for c in hw['counters']), 'unit 解析为 "Throughput %"')

        print('[3] merge_hw_into_perfetto.py 合并（含 R6 与对齐断言）')
        trace_path = work / 'sw_trace.json.gz'
        events_path = work / 'events_rank0.json'
        tinfo = build_software_trace(trace_path)
        build_events(events_path)
        out_path = work / 'merged.json.gz'
        proc = run([sys.executable, '-B', str(MERGE), '--trace', str(trace_path),
                    '--hw-metrics', str(hw_path), '--events', str(events_path),
                    '--out', str(out_path), '--launch-select', 'closest-span'])
        check(proc.returncode == 0, f'merge 退出码 0（实际 {proc.returncode}）\n{proc.stdout}\n{proc.stderr}')
        with gzip.open(out_path, 'rt', encoding='utf-8') as handle:
            merged = json.load(handle)
        ev = merged['traceEvents']
        x_after = sum(1 for e in ev if e.get('ph') == 'X')
        c_after = sum(1 for e in ev if e.get('ph') == 'C')
        check(x_after == tinfo['x_count'], f'ph==X 不变 == {tinfo["x_count"]}（实际 {x_after}）')
        check(c_after > 0, f'ph==C > 0（实际 {c_after}）')
        hm = merged['otherData']['hw_merge']
        sw_pids = sorted({e['pid'] for e in ev if e.get('ph') in ('X', 'C', 'M')})
        hw_pid = hm['hw_counter_pid']
        sw_only = [p for p in sw_pids if p != hw_pid]
        check(hw_pid < min(sw_only), f'R6：counter pid {hw_pid} < 所有软件 pid {sw_only}')
        check(hm['r6_hw_pid_lt_all_software'] is True, 'manifest 标记 R6 成立')
        chosen = hm['chosen_launch']
        check(chosen['ordinal'] == 1, f'closest-span 选中 launch1（ordinal 1，实际 {chosen["ordinal"]}）')
        rel = hm['alignment']['relative_error']
        check(rel < 0.10, f'对齐相对误差 {rel * 100:.3f}% < 10%')
        check(hm['alignment']['sw_span_ns'] == SW_SPAN, f'sw_span == {SW_SPAN}')
        exp_c = expected_samples_in(LAUNCH1[0], LAUNCH1[1]) * len(METRICS)
        check(c_after == exp_c, f'ph==C 数 == 选定 launch 窗口内样本×metric == {exp_c}（实际 {c_after}）')
        pname = next((e['args']['name'] for e in ev
                      if e.get('ph') == 'M' and e.get('name') == 'process_name' and e['pid'] == hw_pid), None)
        check(pname == 'HW dev0 (device-wide, full-kernel sampled)',
              f'HW process 名标明设备级/全 kernel 采样（实际 {pname!r}）')
        check('CTA' in hm['note'] and ('设备级' in hm['note'] or 'device-wide' in hm['note'])
              and 'phase' in hm['note'],
              'otherData 含设备级/不可归因到 CTA、warp、单一 phase 的说明')
        check(hm['sampling_scope'].startswith('continuous PM samples'),
              'metadata 明确 counter 连续覆盖所选 full-kernel 窗口')
        check(hm['phase_attribution'].startswith('none'),
              'metadata 禁止把 device-wide counter 专属归因给 Dispatch 等 phase')
        check('zeros preserved' in hm['sample_value_policy']
              and 'no interpolation' in hm['sample_value_policy']
              and 'forward-fill' in hm['sample_value_policy'],
              'metadata 明确保留原始零值且不插值/forward-fill')
        check(any('Scatter' in item for item in hm['known_nvlink_contributors']),
              'metadata 列出 rank-mapped Scatter 这一 NVLink 来源')
        rebase = hm['timeline_rebase']
        check(rebase['applied'] is True and abs(rebase['offset_us'] - 60.0) < 1e-9,
              f'合法 pre-roll 触发一次全局 60us rebase（实际 {rebase}）')
        check(rebase['relative_timing_preserved'] is True
              and rebase['duration_fields_unchanged'] is True,
              'rebase 只做全局平移，不 clamp/drop/rescale')
        check(merged['otherData']['timeline_rebase'] == rebase,
              '顶层同步记录 timeline_rebase，避免下游遗漏')
        check('logical software coordinates' in
              merged['otherData']['timeline_coordinate_systems']['summary_and_barrier_ns'],
              'metadata 区分 raw/logical/displayed 三套坐标')
        check('sw_span / nsys_span' not in hm['alignment']['formula'],
              'alignment metadata 公式与 shift-only 实现一致，不再伪写缩放项')
        x_ts = [e['ts'] for e in ev if e.get('ph') == 'X']
        c_ts = [e['ts'] for e in ev if e.get('ph') == 'C']
        check(min(x_ts + c_ts) >= 0.0, '最终所有 X/C displayed ts >= 0')
        check(all(abs(t) < 1e7 for t in c_ts), 'counter ts 为有限的 μs 值（落在软件时间轴量级）')

        # Broken SW input must hard-fail before merge and leave no new artifact.
        bad_sw = work / 'sw_trace_negative.json.gz'
        with gzip.open(trace_path, 'rt', encoding='utf-8') as handle:
            bad_doc = json.load(handle)
        next(e for e in bad_doc['traceEvents'] if e.get('ph') == 'X')['ts'] = -0.001
        with gzip.open(bad_sw, 'wt', encoding='utf-8') as handle:
            json.dump(bad_doc, handle, ensure_ascii=False)
        bad_out = work / 'merged_from_negative_input.json.gz'
        bad_proc = run([sys.executable, '-B', str(MERGE), '--trace', str(bad_sw),
                        '--hw-metrics', str(hw_path), '--events', str(events_path),
                        '--out', str(bad_out), '--launch-select', 'closest-span'])
        check(bad_proc.returncode == 4 and not bad_out.exists(),
              f'负 SW ts 必须 rc4 且写前无产物（rc={bad_proc.returncode}）')
        check('负显示时间戳' in bad_proc.stdout,
              f'负 SW ts FAIL 原因明确（stdout={bad_proc.stdout!r}）')

        print('[4] 双 rank per-rank shift + 可重复 --events')
        per_rank_trace = work / 'sw_trace_per_rank.json.gz'
        build_per_rank_software_trace(trace_path, per_rank_trace)
        events_rank1 = work / 'events_rank1.json'
        build_events(events_rank1, rank=1, device=1, t0=SW_T0 + 123456)
        per_rank_out = work / 'merged_per_rank.json.gz'
        proc = run([sys.executable, '-B', str(MERGE), '--trace', str(per_rank_trace),
                    '--hw-metrics', str(hw_path),
                    '--events', str(events_path), '--events', str(events_rank1),
                    '--events-rank', '0', '--out', str(per_rank_out),
                    '--launch-select', 'closest-span'])
        check(proc.returncode == 0,
              f'双 rank merge 退出码 0（实际 {proc.returncode}）\n{proc.stdout}\n{proc.stderr}')
        with gzip.open(per_rank_out, 'rt', encoding='utf-8') as handle:
            per_rank_merged = json.load(handle)
        prm = per_rank_merged['otherData']['hw_merge']
        check(prm['aligned_software_rank'] == 0, '双 rank merge 选择 rank0 对齐')
        check(prm['shift_ns'] == SHIFT_NS, '从 shift_ns_by_rank[0] 取 shift')
        check('shift_ns_by_rank[0]' in prm['shift_ns_source'],
              f'记录 per-rank shift 来源（实际 {prm["shift_ns_source"]!r}）')
        check(len(prm['events_sources']) == 2, '保留两个 --events 来源')
        check(prm['hardware_coverage_scope'] == 'single device-wide counter set; not per-rank',
              '明确 HW 是单 device-wide counter set，不伪称逐 rank 覆盖')

        print('[5] NVLink RX/TX Request+Response 逐点累加并画独立方向曲线')
        nvlink_hw_path = work / 'hw_metrics_nvlink_totals.json'
        proc = run([sys.executable, '-B', str(EXTRACT), '--sqlite', str(sqlite_path),
                    '--kernel-name-contains', KERNEL_NEEDLE, '--out', str(nvlink_hw_path),
                    '--metrics-allowlist', str(PROFILE_DIR / 'ep_moe_metrics.json')])
        check(proc.returncode == 0,
              f'NVLink totals extract 退出码 0（实际 {proc.returncode}）\n{proc.stdout}\n{proc.stderr}')
        nvlink_hw = json.loads(nvlink_hw_path.read_text())
        counters_by_name = {counter['name']: counter for counter in nvlink_hw['counters']}
        direction_sources = {
            'RX': ['NVLink RX Requests User Data', 'NVLink RX Responses User Data'],
            'TX': ['NVLink TX Requests User Data', 'NVLink TX Responses User Data'],
        }
        for direction, sources in direction_sources.items():
            target = f'NVLink {direction} User Data Total'
            check(all(source in counters_by_name for source in sources),
                  f'{direction} Request/Response 两条原始曲线均保留')
            check(target in counters_by_name, f'自动生成 {target} 派生曲线')
            derived = counters_by_name[target]
            check(derived.get('combined_from') == sources,
                  f'{target} combined_from 元数据严格绑定 Request+Response')
            request_samples = counters_by_name[sources[0]]['samples']
            response_samples = counters_by_name[sources[1]]['samples']
            total_samples = derived['samples']
            check(len(request_samples) == len(response_samples) == len(total_samples),
                  f'{target} 与两条源曲线样本数完全一致')
            exact = all(
                req[0] == resp[0] == total[0] and total[1] == req[1] + resp[1]
                for req, resp, total in zip(request_samples, response_samples, total_samples)
            )
            check(exact, f'{target} 每个相同时间戳严格等于 Request+Response')
            check(max(sample[1] for sample in total_samples) > 100,
                  f'{target} 不把合法逐点和强行截断到 100%')

        nvlink_trace_path = work / 'merged_nvlink_totals.json.gz'
        proc = run([sys.executable, '-B', str(MERGE), '--trace', str(trace_path),
                    '--hw-metrics', str(nvlink_hw_path), '--events', str(events_path),
                    '--out', str(nvlink_trace_path), '--launch-select', 'most-samples'])
        check(proc.returncode == 0,
              f'NVLink totals merge 退出码 0（实际 {proc.returncode}）\n{proc.stdout}\n{proc.stderr}')
        with gzip.open(nvlink_trace_path, 'rt', encoding='utf-8') as handle:
            nvlink_trace = json.load(handle)
        expected_direction_samples = expected_samples_in(LAUNCH1[0], LAUNCH1[1])
        for direction in direction_sources:
            track = f'NVLink {direction} User Data Total [Throughput %]'
            actual = sum(1 for event in nvlink_trace['traceEvents']
                         if event.get('ph') == 'C' and event.get('name') == track)
            check(actual == expected_direction_samples,
                  f'Perfetto 画出 {track}，样本数 {actual} == {expected_direction_samples}')

        print('[6] extract 白名单过滤 + merge most-samples 策略')
        allowlist_path = work / 'keep_tensor.json'
        allowlist_path.write_text(json.dumps({
            'schema': 'ep-moe-metrics/v1',
            'keep': ['Tensor Active'],
            'required': ['Tensor Active'],
        }, ensure_ascii=False), encoding='utf-8')
        hw_filtered = work / 'hw_metrics_filtered.json'
        proc = run([sys.executable, '-B', str(EXTRACT), '--sqlite', str(sqlite_path),
                    '--kernel-name-contains', KERNEL_NEEDLE, '--out', str(hw_filtered),
                    '--metrics-allowlist', str(allowlist_path)])
        check(proc.returncode == 0, f'带白名单的 extract 退出码 0（实际 {proc.returncode}）\n{proc.stderr}')
        hwf = json.loads(hw_filtered.read_text())
        check([c['name'] for c in hwf['counters']] == ['Tensor Active'],
              f'白名单过滤后只剩 Tensor Active（实际 {[c["name"] for c in hwf["counters"]]}）')
        mf = hwf['metrics_filter']
        check(mf['enabled'] is True, '白名单过滤应标记 enabled=True')
        check('Tensor Active' in mf['matched'], 'matched 应含 Tensor Active')
        check('DRAM Total Throughput' in mf['dropped'] and 'SM FMA Pipe Throughput' in mf['dropped'],
              f'dropped 应含未命中的两个指标（实际 {mf["dropped"]}）')
        ms_out = work / 'merged_most_samples.json.gz'
        proc = run([sys.executable, '-B', str(MERGE), '--trace', str(trace_path),
                    '--hw-metrics', str(hw_filtered), '--events', str(events_path),
                    '--out', str(ms_out), '--launch-select', 'most-samples'])
        check(proc.returncode == 0, f'most-samples merge 退出码 0（实际 {proc.returncode}）\n{proc.stderr}')
        with gzip.open(ms_out, 'rt', encoding='utf-8') as handle:
            ms_merged = json.load(handle)
        ms_hm = ms_merged['otherData']['hw_merge']['chosen_launch']
        check(ms_hm['launch_select'] == 'most-samples', 'chosen_launch 记录 launch_select=most-samples')
        check(ms_hm['samples_in_window'] > 0,
              f'most-samples 选中的 launch 窗口内应有采样点（实际 {ms_hm["samples_in_window"]}）')
        check(ms_hm['coverage'] is not None, 'most-samples 应报告覆盖率')
        ms_c = sum(1 for e in ms_merged['traceEvents'] if e.get('ph') == 'C')
        check(ms_c > 0, f'most-samples 合并后应有 counter 事件（实际 {ms_c}）')

    except Failure as exc:
        failures.append(str(exc))
        print(f'  [FAIL] {exc}')
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)

    print()
    if failures:
        print(f'SELFTEST FAIL（{len(failures)} 项）')
        return 1
    print('SELFTEST PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
