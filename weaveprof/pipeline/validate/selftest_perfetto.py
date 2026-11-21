from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
PROFILE_DIR = SCRIPTS_DIR.parents[2]
REPO_DIR = PROFILE_DIR.parent
WORK_DIR = REPO_DIR / '.work' / 'selftest' / 'perfetto'
ABI_PATH = WORK_DIR / 'fixture_abi.json'
TOPOLOGY_PATH = WORK_DIR / 'fixture_warp_topology.json'
CAPTURE_PLAN_PATH = WORK_DIR / 'capture_scope.json'
BUILDER = PROFILE_DIR / 'weaveprof/pipeline/capture' / 'build_perfetto.py'
LAYOUT_CHECKER = PROFILE_DIR / 'weaveprof/pipeline/scan' / 'layout_contract.py'
DELIVER_GATE = SCRIPTS_DIR / 'deliver_gate.py'

from deliver_gate import check_self_evident, check_task_claim_contract
from validate_capability import find_partial_overlaps, validate_task_phase_semantics

TRACK_DISPATCH = 0
TRACK_LOADER_A = 1
TRACK_LOADER_B = 2
TRACK_MATH_WG0 = 3
TRACK_MATH_WG1 = 4
ENTRY_TASK_ID = 19

TASK_NAMES = {
    0: 'kDispatchRoutePublish',
    1: 'kDispatchNvlinkPull',
    5: 'kLoaderAPipeline',
    6: 'kLoaderBPipeline',
    7: 'kMathL1GemmPipeline',
    8: 'kMathL1SwiGLUQuantize',
    10: 'kMathL2GemmPipeline',
    11: 'kMathL2Bf16Scatter',
    15: 'kMathWaitFullBarrier',
    19: 'kEntryRankSync',
}

SHORT_NAMES = {
    0: 'route_publish',
    1: 'nvlink_pull',
    5: 'loader_a_pipe',
    6: 'loader_b_pipe',
    7: 'l1_gemm',
    8: 'l1_epilogue',
    10: 'l2_gemm',
    11: 'l2_scatter',
    15: 'wait_full_barrier',
    19: 'entry_rank_sync',
}

TASK_ROLES = {0: 'dispatch', 1: 'dispatch', 5: 'loader', 6: 'loader',
              7: 'math', 8: 'math', 10: 'math', 11: 'math', 15: 'sync', 19: 'sync'}

def expected_lane_counts(*docs: Dict[str, Any]) -> Dict[str, Any]:
    measured_x = 0
    processes = 0
    group_thread_counts: List[int] = []
    for doc in docs:
        per_lane: Dict[tuple, int] = {}
        visible_per_lane: Dict[tuple, int] = {}
        for item in doc['events']:
            key = (item['block'], item['track'])
            per_lane[key] = per_lane.get(key, 0) + 1
            if item['layer'] != 'ignored':
                visible_per_lane[key] = visible_per_lane.get(key, 0) + 1
        measured_x += sum(visible_per_lane.values())
        blocks: Dict[int, int] = {}
        for block, track in per_lane:
            blocks[block] = 8
        processes += len(blocks)
        group_thread_counts.extend(blocks.values())
    return {
        'measured_x': measured_x,
        'replica_x': 0,
        'total_x': measured_x,
        'processes': processes,
        'threads_per_group': sorted(set(group_thread_counts)),
    }


def event(block: int, track: int, role: str, task_id: int, start_ns: int, end_ns: int,
          layer: str, semantic_kind: str, expert_idx: int = 0, depth: int = 0,
          offset: int = 0, task_role: str | None = None) -> Dict[str, Any]:
    return {
        'block': block,
        'depth': depth,
        'dur_ns': end_ns - start_ns,
        'end_ns': end_ns,
        'end_offset': offset + 1,
        'expert_idx': expert_idx,
        'is_derived': False,
        'layer': layer,
        'role': role,
        'semantic_kind': semantic_kind,
        'short_name': SHORT_NAMES[task_id],
        'start_ns': start_ns,
        'start_offset': offset,
        'task': TASK_NAMES[task_id],
        'task_id': task_id,
        'task_role': task_role,
        'track': track,
        'track_label': role,
    }


def rank_events(base_ns: int, arrive_offsets: Dict[int, int],
                release_offsets: Dict[int, int]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for block in sorted(arrive_offsets):
        arrive = base_ns + arrive_offsets[block]
        release = base_ns + release_offsets[block]
        events.append(event(block, TRACK_DISPATCH, 'sync', ENTRY_TASK_ID, arrive, release,
                            'ignored', 'sync'))
        events.append(event(block, TRACK_DISPATCH, 'dispatch', 0,
                            release + 100, release + 900, 'phase', 'dispatch', depth=1))
        events.append(event(block, TRACK_DISPATCH, 'dispatch', 1,
                            release + 1000, release + 5000, 'phase', 'dispatch', depth=1))
        events.append(event(block, TRACK_LOADER_A, 'loader', 5,
                            release + 200, release + 4800, 'phase', 'load', depth=1))
        events.append(event(block, TRACK_LOADER_B, 'loader', 6,
                            release + 250, release + 4700, 'phase', 'load', depth=1))
        gemm0_task = (release + 300, release + 5600)
        gemm0_mma = (release + 300, release + 5200)
        events.append(event(block, TRACK_MATH_WG0, 'math', 7, *gemm0_task,
                            'task', 'gemm', expert_idx=block % 8, task_role='math'))
        events.append(event(block, TRACK_MATH_WG0, 'math', 7, *gemm0_mma,
                            'phase', 'mma', expert_idx=block % 8, depth=1))
        events.append(event(block, TRACK_MATH_WG0, 'math', 8,
                            release + 5300, release + 5600, 'phase', 'epilogue',
                            expert_idx=block % 8, depth=1))
        events.append(event(block, TRACK_MATH_WG0, 'sync', 15,
                            release + 400, release + 1200, 'ignored', 'barrier',
                            expert_idx=block % 8, depth=1))
        gemm1_task = (release + 350, release + 5500)
        gemm1_mma = (release + 350, release + 4900)
        events.append(event(block, TRACK_MATH_WG1, 'math', 10, *gemm1_task,
                            'task', 'gemm', expert_idx=block % 8, task_role='math'))
        events.append(event(block, TRACK_MATH_WG1, 'math', 10, *gemm1_mma,
                            'phase', 'mma', expert_idx=block % 8, depth=1))
        events.append(event(block, TRACK_MATH_WG1, 'math', 11,
                            release + 5000, release + 5500, 'phase', 'scatter',
                            expert_idx=block % 8, depth=1))
    return events


def real_anchors(base_ns: int, arrive_offsets: Dict[int, int],
                 release_offsets: Dict[int, int]) -> Dict[str, Any]:
    blocks = sorted(arrive_offsets)
    starts = [base_ns + arrive_offsets[block] for block in blocks]
    ends = [base_ns + release_offsets[block] for block in blocks]
    per_cta = [{'block': block,
                'dur_ns': release_offsets[block] - arrive_offsets[block],
                'entry_sync_end_ns': base_ns + release_offsets[block],
                'entry_sync_start_ns': base_ns + arrive_offsets[block],
                'occurrences': 1} for block in blocks]
    return {
        'entry_rendezvous': {
            'ctas_missing_anchor': [],
            'ctas_missing_anchor_total': 0,
            'ctas_with_anchor': len(blocks),
            'end_semantics': 'rank_release',
            'global': {
                'end_skew_ns': max(ends) - min(ends),
                'entry_sync_end_ns': max(ends),
                'entry_sync_start_ns': min(starts),
                'max_end_ns': max(ends),
                'max_start_ns': max(starts),
                'min_end_ns': min(ends),
                'min_start_ns': min(starts),
                'rendezvous_span_ns': max(ends) - min(starts),
                'start_skew_ns': max(starts) - min(starts),
            },
            'mechanism': {'between_start_and_end': 'comm::nvlink_barrier across ranks',
                          'profile_only': True},
            'per_cta': per_cta,
            'start_semantics': 'rank_arrive',
            'task': 'kEntryRankSync',
            'task_id': ENTRY_TASK_ID,
        }
    }


def make_rank_doc(rank: int, device: int, base_ns: int, arrive_offsets: Dict[int, int],
                  release_offsets: Dict[int, int],
                  unwritten: Sequence[int] = ()) -> Dict[str, Any]:
    blocks = sorted(arrive_offsets)
    events = rank_events(base_ns, arrive_offsets, release_offsets)
    return {
        'anchors': real_anchors(base_ns, arrive_offsets, release_offsets),
        'block_map': [{'logical_cta': block, 'physical_smid': 40 + block * 3,
                       'written': block not in unwritten} for block in blocks],
        'events': events,
        'metadata': {
            'device': device,
            'events_total': len(events),
            'max_records': 64,
            'num_sms': len(blocks),
            'num_tracks': 7,
            'rank': rank,
            'records_total': len(events),
            'time_window': {'max_end_ns': max(item['end_ns'] for item in events),
                            'min_start_ns': min(item['start_ns'] for item in events)},
        },
    }


def flatten_anchors(doc: Dict[str, Any]) -> Dict[str, Any]:
    flat = json.loads(json.dumps(doc))
    nested = doc['anchors']['entry_rendezvous']
    flat['anchors'] = {
        'entry_sync_start_ns': nested['global']['max_start_ns'],
        'entry_sync_end_ns': nested['global']['min_end_ns'],
        'per_cta': [{'logical_cta': entry['block'],
                     'entry_sync_start_ns': entry['entry_sync_start_ns'],
                     'entry_sync_end_ns': entry['entry_sync_end_ns']}
                    for entry in nested['per_cta']],
    }
    return flat


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return path


def run_builder(arguments: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, '-B', str(BUILDER)] + arguments
                          + ['--warp-topology', str(TOPOLOGY_PATH), '--abi', str(ABI_PATH),
                             '--capture-plan', str(CAPTURE_PLAN_PATH)],
                          capture_output=True, text=True, timeout=300)


def load_trace(path: Path) -> Dict[str, Any]:
    if path.name.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding='utf-8'))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def metadata_names(trace: Dict[str, Any], kind: str) -> List[str]:
    return [item['args']['name'] for item in trace['traceEvents']
            if item['ph'] == 'M' and item['name'] == kind]


def rank_shift(trace: Dict[str, Any], item: Dict[str, Any]) -> int:
    return trace['otherData']['shift_ns_by_rank'][str(item['args']['rank'])]


def x_of(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [item for item in trace['traceEvents'] if item.get('ph') == 'X']


def semantic_gate_regressions() -> None:
    def span(pid: int, tid: int, name: str, layer: str,
             start_ns: int, end_ns: int, **extra) -> Dict[str, Any]:
        args = {"layer": layer, "start_ns_raw": start_ns, "end_ns_raw": end_ns,
                "depth": 0 if layer == "task" else 1}
        args.update(extra)
        return {"ph": "X", "pid": pid, "tid": tid, "name": name,
                "cat": f"{layer}/{args.get('semantic_kind', 'compute')}",
                "ts": start_ns / 1000.0, "dur": (end_ns - start_ns) / 1000.0,
                "args": args}

    task = span(1, 0, "GEMM0", "task", 1000, 3000,
                task_id=7, role="math", task_role="math", is_derived=False)
    mma = span(1, 0, "MMA", "phase", 1000, 3000,
               task_id=7, role="math", semantic_kind="mma", is_derived=False)
    late_phase = span(1, 1, "Epilogue", "phase", 4000, 5000,
                      task_id=8, role="epilogue", semantic_kind="epilogue",
                      is_derived=False)
    legal = [task, mma, late_phase]
    semantic_fails, _ = validate_task_phase_semantics(legal)
    check(not semantic_fails,
          f'task 结束后另一 lane 的真实 phase 应合法: {semantic_fails}')
    check(find_partial_overlaps(legal)[0] == 0,
          '相等区间(task+MMA)与另一 lane 后续 phase 不应算 partial overlap')

    replica = json.loads(json.dumps(late_phase))
    replica["args"]["is_replica"] = True
    semantic_fails, _ = validate_task_phase_semantics([task, mma, replica])
    check(any(fail_id == "observer_replica_visible" for fail_id, _ in semantic_fails),
          f'observer replica X 必须硬 FAIL: {semantic_fails}')

    a = span(9, 9, "A", "phase", 5536, 51136,
             task_id=5, role="loader", semantic_kind="load", is_derived=False)
    b = span(9, 9, "B", "phase", 47552, 55296,
             task_id=8, role="epilogue", semantic_kind="store", is_derived=False)
    crossing = legal + [a, b]
    count, samples = find_partial_overlaps(crossing)
    check(count >= 1 and samples,
          'A=[5536,51136], B=[47552,55296] 必须命中 partial overlap')
    gate_fails, _ = check_self_evident({"traceEvents": crossing})
    check(any(fail_id == "slice_partial_overlap" for fail_id, _ in gate_fails),
          f'deliver gate 必须硬 FAIL partial overlap: {gate_fails}')

    negative_x = json.loads(json.dumps(legal))
    negative_x[0]["ts"] = -0.001  # raw ns stays positive on purpose
    gate_fails, _ = check_self_evident({"traceEvents": negative_x})
    check(any(fail_id == "timeline_negative_timestamp" for fail_id, _ in gate_fails),
          f'deliver gate 必须硬 FAIL negative X ts even with positive raw ns: {gate_fails}')
    negative_c = json.loads(json.dumps(legal))
    negative_c.append({"ph": "C", "pid": 0, "tid": 0, "name": "SMs Active",
                       "ts": -0.001, "args": {"value": 1}})
    gate_fails, _ = check_self_evident({"traceEvents": negative_c})
    check(any(fail_id == "timeline_negative_timestamp" for fail_id, _ in gate_fails),
          f'deliver gate 必须硬 FAIL negative C ts: {gate_fails}')

    # Regression for the rejected 2026-08-26 trace: Perfetto emits one parser
    # failure for each boolean counter arg.  Python bool is an int subclass, so
    # the gate must reject it explicitly rather than accepting isinstance(v, int).
    bad_counter = json.loads(json.dumps(legal))
    bad_counter.append({"ph": "C", "pid": 0, "tid": 1, "name": "SMs Active",
                        "ts": 1.0,
                        "args": {"value": 12.5, "sampled": True,
                                 "zero_filled": False}})
    gate_fails, _ = check_self_evident({
        "traceEvents": bad_counter,
        "otherData": {"producer": "profile/scripts/build_perfetto.py"},
    })
    bad_values = [message for fail_id, message in gate_fails
                  if fail_id == "delivery_perfetto_counter_value"]
    check(len(bad_values) == 2,
          f'two boolean counter args must produce two hard failures: {gate_fails}')

    derived = json.loads(json.dumps(task))
    derived["args"]["is_derived"] = True
    semantic_fails, _ = validate_task_phase_semantics([derived, mma])
    check(any(fail_id == "derived_task_visible" for fail_id, _ in semantic_fails),
          f'derived task 必须硬 FAIL: {semantic_fails}')

    fake = json.loads(json.dumps(task))
    fake["args"].update({"task_id": 5, "role": "loader", "task_role": "loader"})
    semantic_fails, _ = validate_task_phase_semantics([fake, mma])
    check(any(fail_id == "invalid_task_provenance" for fail_id, _ in semantic_fails),
          f'非 GEMM0/GEMM1 task 必须硬 FAIL: {semantic_fails}')


def write_topology_fixtures() -> None:
    tasks = []
    for task_id, name in TASK_NAMES.items():
        tasks.append({'id': task_id, 'name': name, 'short_name': SHORT_NAMES[task_id],
                      'layer': 'task' if task_id in (7, 10) else 'phase',
                      'semantic_kind': 'gemm' if task_id in (7, 10) else 'phase',
                      'viewer_policy': 'exact',
                      'parent_task_id': 7 if task_id in (8,) else (10 if task_id in (11,) else None),
                      'is_entry_sync': task_id == ENTRY_TASK_ID})
    tracks = [
        {'track_id': 0, 'observer_warp_expr': '0', 'role': 'dispatch'},
        {'track_id': 1, 'observer_warp_expr': '1', 'role': 'load'},
        {'track_id': 2, 'observer_warp_expr': '2', 'role': 'load'},
        {'track_id': 3, 'observer_warp_expr': '3', 'role': 'math'},
        {'track_id': 4, 'observer_warp_expr': '4', 'role': 'math',
         'representation_policy': 'hopper_mma_warpgroup', 'warpgroup': True},
        {'track_id': 5, 'observer_warp_expr': '8', 'role': 'hopper_mma_warpgroup',
         'representation_policy': 'hopper_mma_warpgroup', 'warpgroup': True},
        {'track_id': 6, 'observer_warp_expr': '12', 'role': 'hopper_mma_warpgroup',
         'representation_policy': 'hopper_mma_warpgroup', 'warpgroup': True},
        {'track_id': 7, 'observer_warp_expr': '16', 'role': 'hopper_mma_warpgroup',
         'representation_policy': 'hopper_mma_warpgroup', 'warpgroup': True},
    ]
    abi = {'schema': 'fixture-abi/v1', 'architecture': 'Hopper',
           'tasks': tasks, 'tracks': tracks}
    runtime = {'kernel_signature': 'fixture_kernel(int)',
               'provenance': {'source': 'selftest fixture', 'launch': 'fixture'},
               'parameters': {'kNumThreads': 640, 'math_warp_start': 4}}
    write_json(ABI_PATH, abi)
    runtime_path = WORK_DIR / 'fixture_runtime.json'
    write_json(runtime_path, runtime)
    detector = PROFILE_DIR / 'weaveprof/pipeline/capture' / 'detect_warp_topology.py'
    result = subprocess.run([sys.executable, '-B', str(detector), '--abi', str(ABI_PATH),
                             '--runtime', str(runtime_path), '--out', str(TOPOLOGY_PATH)],
                            capture_output=True, text=True, timeout=60)
    check(result.returncode == 0, f'topology fixture 生成失败:\n{result.stdout}\n{result.stderr}')
    topology = json.loads(TOPOLOGY_PATH.read_text(encoding='utf-8'))
    check(topology['physical_warps_per_cta'] == 20, 'fixture physical warp 必须为 20')
    check(topology['observer_lane_count'] == 8, 'fixture observer lane 必须为 8')
    expected_labels = ['warp0', 'warp1', 'warp2', 'warp3',
                       'warp4（代表 warp4-7）', 'warp8（代表 warp8-11）',
                       'warp12（代表 warp12-15）', 'warp16（代表 warp16-19）']
    check([lane['label'] for lane in topology['lanes']] == expected_labels,
          f'fixture lane labels 异常: {topology["lanes"]}')
    check(all('track' not in lane['label'] for lane in topology['lanes']),
          'topology label 不得包含内部 track 文本')


def _write_topology_fixtures_v2() -> None:
    """Create the H100 physical-observer/warpgroup fixture used by topology regression."""
    tasks = []
    for task_id, name in TASK_NAMES.items():
        tasks.append({
            'id': task_id, 'name': name, 'short_name': SHORT_NAMES[task_id],
            'layer': 'task' if task_id in (7, 10) else 'phase',
            'semantic_kind': 'gemm' if task_id in (7, 10) else 'phase',
            'viewer_policy': 'exact',
            'parent_task_id': 7 if task_id == 8 else (10 if task_id == 11 else None),
            'is_entry_sync': task_id == ENTRY_TASK_ID,
        })
    tracks = [
        {'track_id': 0, 'observer_warp_expr': '0', 'role': 'dispatch'},
        {'track_id': 1, 'observer_warp_expr': '1', 'role': 'load'},
        {'track_id': 2, 'observer_warp_expr': '2', 'role': 'load'},
        {'track_id': 3, 'observer_warp_expr': '3', 'role': 'math'},
        {'track_id': 4, 'observer_warp_expr': '4', 'role': 'math',
         'representation_policy': 'hopper_mma_warpgroup', 'warpgroup': True},
        {'track_id': 5, 'observer_warp_expr': '8', 'role': 'hopper_mma_warpgroup',
         'representation_policy': 'hopper_mma_warpgroup', 'warpgroup': True},
        {'track_id': 6, 'observer_warp_expr': '12', 'role': 'hopper_mma_warpgroup',
         'representation_policy': 'hopper_mma_warpgroup', 'warpgroup': True},
        {'track_id': 7, 'observer_warp_expr': '16', 'role': 'hopper_mma_warpgroup',
         'representation_policy': 'hopper_mma_warpgroup', 'warpgroup': True},
    ]
    abi = {'schema': 'fixture-abi/v1', 'architecture': 'Hopper',
           'tasks': tasks, 'tracks': tracks}
    runtime = {
        'kernel_signature': 'fixture_kernel(int)',
        'provenance': {'source': 'selftest fixture', 'launch': 'fixture'},
        'parameters': {'kNumThreads': 640},
    }
    write_json(ABI_PATH, abi)
    runtime_path = WORK_DIR / 'fixture_runtime.json'
    write_json(runtime_path, runtime)
    detector = PROFILE_DIR / 'weaveprof/pipeline/capture' / 'detect_warp_topology.py'
    result = subprocess.run([
        sys.executable, '-B', str(detector), '--abi', str(ABI_PATH),
        '--runtime', str(runtime_path), '--out', str(TOPOLOGY_PATH)],
        capture_output=True, text=True, timeout=60)
    check(result.returncode == 0, f'topology fixture generation failed:\n{result.stdout}\n{result.stderr}')
    topology = json.loads(TOPOLOGY_PATH.read_text(encoding='utf-8'))
    check(topology['physical_warps_per_cta'] == 20, 'fixture must have 20 physical warps')
    labels = [
        'warp0', 'warp1', 'warp2', 'warp3',
        'warp4\uff08\u4ee3\u8868 warp4-7\uff09',
        'warp8\uff08\u4ee3\u8868 warp8-11\uff09',
        'warp12\uff08\u4ee3\u8868 warp12-15\uff09',
        'warp16\uff08\u4ee3\u8868 warp16-19\uff09',
    ]
    check(topology['observer_lane_count'] == len(labels), 'fixture must have 8 observer lanes')
    check([lane['label'] for lane in topology['lanes']] == labels,
          f'fixture lane labels are wrong: {topology["lanes"]}')
    check(all('track' not in lane['label'] for lane in topology['lanes']),
          'topology labels must not expose internal track text')


write_topology_fixtures = _write_topology_fixtures_v2


def main() -> int:
    semantic_gate_regressions()
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    write_json(CAPTURE_PLAN_PATH, {
        'schema': 'fast-instrument-capture-plan/v1',
        'mode': 'all',
        'capture_scope': 'all_physical_sms',
        'selected_physical_smids': [],
        'kernel_execution_scope': 'full_grid',
        'source_capture_was_all_sm': True,
        'source_physical_smid_filter_required': False,
        'compact_buffer_required': False,
        'postprocess_smid_filter_allowed': False,
    })
    write_topology_fixtures()

    arrive = {0: 500, 1: 900, 2: 1500}
    release = {0: 2000, 1: 2050, 2: 2100}
    rank0 = make_rank_doc(0, 0, 1_000_000_000, arrive, release)
    rank1 = make_rank_doc(1, 1, 1_000_000_300, arrive, release)
    rank0_path = write_json(WORK_DIR / 'events_rank0.json', rank0)
    rank1_path = write_json(WORK_DIR / 'events_rank1.json', rank1)
    global0 = rank0['anchors']['entry_rendezvous']['global']
    global1 = rank1['anchors']['entry_rendezvous']['global']

    out_path = WORK_DIR / 'trace.perfetto.json'
    result = run_builder([str(rank0_path), str(rank1_path), '--out', str(out_path)])
    check(result.returncode == 0,
          f'build_perfetto.py 退出码 {result.returncode}\nstdout:\n{result.stdout}\n'
          f'stderr:\n{result.stderr}')
    check(out_path.exists() and out_path.stat().st_size > 0, f'{out_path} 未生成或为空')
    trace = load_trace(out_path)

    check(isinstance(trace.get('traceEvents'), list), 'traceEvents 必须是列表')
    embedded_contract = [item for item in trace['traceEvents']
                         if item.get('name') == 'perfetto_layout_contract']
    check(len(embedded_contract) == 1, 'builder 必须内嵌唯一 layout contract metadata')
    layout_proc = subprocess.run([sys.executable, '-B', str(LAYOUT_CHECKER), str(out_path)],
                                 capture_output=True, text=True, timeout=60)
    check(layout_proc.returncode == 0,
          f'生成 trace 必须通过同源 layout checker: {layout_proc.stdout}\n'
          f'{layout_proc.stderr}')
    x_events = [item for item in trace['traceEvents'] if item.get('ph') == 'X']
    m_events = [item for item in trace['traceEvents'] if item.get('ph') == 'M']
    expected = expected_lane_counts(rank0, rank1)
    check(len(x_events) == expected['total_x'],
          f'X 事件数 {len(x_events)} != {expected["total_x"]}（只允许实测 observer）')
    check(len(trace['traceEvents']) == len(x_events) + len(m_events),
          'traceEvents 只应包含 ph=X 与 ph=M')
    summary_counts = trace['otherData']['summary']
    check(summary_counts['num_x_events'] == len(x_events)
          and summary_counts['num_metadata_events'] == len(m_events),
          f'summary 里的事件计数与实际不符: {summary_counts["num_x_events"]}/'
          f'{summary_counts["num_metadata_events"]} vs {len(x_events)}/{len(m_events)}')
    check(summary_counts['num_x_measured'] == expected['measured_x']
          and summary_counts['num_x_replica'] == expected['replica_x'],
          f'实测/复制事件数不符: {summary_counts["num_x_measured"]}/'
          f'{summary_counts["num_x_replica"]} vs {expected["measured_x"]}/{expected["replica_x"]}')

    process_names = [name for name in metadata_names(trace, 'process_name')
                     if name.startswith('rank+CTA')]
    expected_processes = [f'rank+CTA {rank}:{block} / SM{40 + block * 3}'
                          for rank in (0, 1) for block in (0, 1, 2)]
    check(process_names == expected_processes, f'process_name 异常: {process_names}')
    check(len(process_names) == expected['processes'],
          f'process 数 {len(process_names)} != {expected["processes"]}（应为每个 (rank,CTA) 一个）')
    sort_indices = [item['args']['sort_index'] for item in trace['traceEvents']
                    if item['ph'] == 'M' and item['name'] == 'process_sort_index'
                    and item['pid'] >= trace['otherData']['pid_base']]
    check(sort_indices == list(range(6)), f'process_sort_index 异常: {sort_indices}')

    thread_names = [item['args']['name'] for item in trace['traceEvents']
                    if item['ph'] == 'M' and item['name'] == 'thread_name'
                    and item['pid'] >= trace['otherData']['pid_base']]
    expected_labels = {'warp0', 'warp1', 'warp2', 'warp3',
                       'warp4（代表 warp4-7）', 'warp8（代表 warp8-11）',
                       'warp12（代表 warp12-15）', 'warp16（代表 warp16-19）'}
    expected_labels = {
        'warp0', 'warp1', 'warp2', 'warp3',
        'warp4\uff08\u4ee3\u8868 warp4-7\uff09',
        'warp8\uff08\u4ee3\u8868 warp8-11\uff09',
        'warp12\uff08\u4ee3\u8868 warp12-15\uff09',
        'warp16\uff08\u4ee3\u8868 warp16-19\uff09',
    }
    check(set(thread_names) == expected_labels,
          f'thread_name 必须只含 8 个 topology label: {thread_names}')
    check(all('track' not in name for name in thread_names),
          f'thread_name 不得暴露内部 track: {thread_names}')
    # 每个 CTA 只声明 8 条 observer lane，不生成物理空 warp lane。
    expected_threads = expected['processes'] * 8
    check(len(thread_names) == expected_threads,
          f'泳道数 {len(thread_names)} != {expected_threads}')
    check(summary_counts['threads_per_cta_group']['values'] == expected['threads_per_group'],
          f'每 CTA 泳道组 thread 数分布异常: {summary_counts["threads_per_cta_group"]}')

    pids = sorted({item['pid'] for item in x_events
                   if item['pid'] >= trace['otherData']['pid_base']})
    check(len(pids) == expected['processes'], f'期望 {expected["processes"]} 个 process，实得 {pids}')
    check(min(pids) >= trace['otherData']['pid_base'],
          f'软件 pid 必须 >= pid_base={trace["otherData"]["pid_base"]}: {pids}')
    replica_x = [item for item in x_events if item['args'].get('is_replica')]
    check(not replica_x, f'任何 observer ph=X 都不得复制: {replica_x[:3]}')
    for item in x_events:
        if item['args'].get('scope') == 'global_trace':
            continue
        for key in ('task_id', 'expert_idx', 'depth', 'block', 'physical_smid',
                    'warp_id', 'layer', 'semantic_kind',
                    'is_derived'):
            check(key in item['args'], f'事件 args 缺少 {key}: {item}')
        check(not any(key in item['args'] for key in ('track', 'is_replica', 'represented_by')),
              f'事件 args 泄漏内部 topology 字段: {item}')
        check(isinstance(item['ts'], float) and isinstance(item['dur'], float),
              f'ts / dur 必须是浮点微秒: {item}')
        check(abs(item['dur'] * 1000.0 - item['args']['dur_ns']) < 1e-6,
              f'dur 与 dur_ns 不一致: {item}')
        check(abs(item['ts'] * 1000.0
                  - (item['args']['start_ns_raw'] - rank_shift(trace, item))) < 1e-6,
              f'ts 丢失了纳秒精度: {item}')
    depths = {item['args']['depth'] for item in x_events
              if item['args'].get('scope') != 'global_trace'}
    check(depths == {0, 1}, f'depth 应原样透传: {depths}')

    shift = trace['otherData']['shift_ns']
    expected_shift = min(global0['entry_sync_end_ns'], global1['entry_sync_end_ns'])
    check(shift == expected_shift, f'shift_ns {shift} != {expected_shift}')
    check(trace['otherData']['align_field'] == 'entry_sync_end_ns',
          f'entry_release 应使用 entry_sync_end_ns: {trace["otherData"]["align_field"]}')

    check(trace['otherData']['clock_domain_requested'] == 'auto',
          f'默认应请求 auto: {trace["otherData"]["clock_domain_requested"]}')
    check(trace['otherData']['clock_domain'] == 'shared',
          f'同时钟域输入下 auto 必须判定为 shared: {trace["otherData"]["clock_domain"]}')
    check(trace['otherData']['shift_ns_by_rank'] == {'0': expected_shift, '1': expected_shift},
          f'shared 时两个 rank 必须共用同一 shift: {trace["otherData"]["shift_ns_by_rank"]}')
    check(trace['otherData']['clock_domain_warning'] is None,
          f'shared 模式不应带跨 rank 不可比警告: {trace["otherData"]["clock_domain_warning"]}')
    check(trace['otherData']['summary']['clock_domain'] == 'shared',
          f'summary 里的 clock_domain 应与 otherData 一致: {trace["otherData"]["summary"]}')
    shared_detail = trace['otherData']['clock_domain_detail']
    check(shared_detail['raw_barrier']['holds'] is True,
          f'shared 判定应基于原始时间戳上判据成立: {shared_detail["raw_barrier"]}')
    check(shared_detail['max_offset']['abs_offset_ns']
          == abs(global0['entry_sync_end_ns'] - global1['entry_sync_end_ns']),
          f'两两基准偏移量应被算出: {shared_detail["max_offset"]}')
    check('=== clock domain ===' in result.stdout, f'stdout 缺少 clock domain 分段:\n{result.stdout}')
    check('requested=auto resolved=shared' in result.stdout,
          f'stdout 必须打印 auto 的判定结果:\n{result.stdout}')
    for token in (str(global0['entry_sync_end_ns']), str(global1['entry_sync_end_ns']),
                  '基准偏移 rank0 - rank1'):
        check(token in result.stdout, f'clock domain 判定必须可解释，缺少 {token}:\n{result.stdout}')
    check('切换到 per_rank' not in result.stdout,
          f'同时钟域输入不应切换到 per_rank:\n{result.stdout}')
    entry_events = [item for item in x_events if item['args'].get('task_id') == ENTRY_TASK_ID]
    check(not entry_events, 'layer=ignored 的 kEntryRankSync 不得画成 ph=X')
    check(min(item['ts'] for item in x_events) >= 0.0,
          'ignored arrive/release 不画 X；首个可见 phase 应位于 release 之后')

    for entry in trace['otherData']['summary']['ranks']:
        anchors = entry['anchors']
        check(anchors['sources']['entry_sync_end_ns']
              == 'anchors.entry_rendezvous.global.entry_sync_end_ns',
              f'release 锚点应来自真实嵌套路径: {anchors["sources"]}')
        check(anchors['sources']['max_start_ns']
              == 'anchors.entry_rendezvous.global.max_start_ns',
              f'max_start_ns 应来自真实嵌套路径: {anchors["sources"]}')
        check(anchors['per_cta_source'] == 'anchors.entry_rendezvous.per_cta',
              f'per-CTA 锚点应来自真实嵌套路径: {anchors["per_cta_source"]}')
        check(anchors['num_per_cta'] == 3, f'per-CTA 锚点数应为 3: {anchors["num_per_cta"]}')
        check(anchors['missing_paths'] == [],
              f'真实 schema 下不应有缺失字段: {anchors["missing_paths"]}')
        for field in ('min_start_ns', 'max_start_ns', 'min_end_ns', 'max_end_ns',
                      'start_skew_ns', 'end_skew_ns'):
            check(anchors['fields'][field] is not None, f'{field} 应被解析出来: {anchors["fields"]}')

    barrier = trace['otherData']['barrier_check']
    expected_margin = (min(global0['min_end_ns'], global1['min_end_ns'])
                       - max(global0['max_start_ns'], global1['max_start_ns']))
    check(barrier['available'] and barrier['holds'],
          f'合法输入的 barrier 校验应通过: {barrier}')
    check(barrier['criterion'] == 'min(rank.min_end_ns) >= max(rank.max_start_ns)',
          f'barrier 判据描述异常: {barrier["criterion"]}')
    check(barrier['margin_ns'] == expected_margin,
          f'barrier margin {barrier["margin_ns"]} != {expected_margin}')
    check(all(entry['holds'] for entry in barrier['intra_rank']),
          f'每个 rank 自身时钟内的 barrier 应成立: {barrier["intra_rank"]}')
    check(barrier['clock_normalized']['available'] and barrier['clock_normalized']['holds'],
          f'归一化 barrier 校验应通过: {barrier["clock_normalized"]}')
    check(barrier['per_cta']['available'] and barrier['per_cta']['holds'],
          f'per-CTA barrier 校验应通过: {barrier["per_cta"]}')
    check(f'{barrier["criterion"]}: PASS' in result.stdout,
          f'stdout 缺少 barrier PASS 摘要:\n{result.stdout}')
    check(f'保守 margin = {expected_margin}ns' in result.stdout,
          f'stdout 缺少保守 margin:\n{result.stdout}')
    check('=== roles ===' in result.stdout and 'per-CTA core' in result.stdout,
          f'stdout 缺少 role / core 摘要:\n{result.stdout}')

    summary = trace['otherData']['summary']
    role_names = {entry['role'] for entry in summary['roles']}
    check(role_names == {'dispatch', 'loader', 'math'}, f'role 集合异常: {role_names}')
    total_pct = sum(entry['pct_of_event_time'] for entry in summary['roles'])
    check(abs(total_pct - 100.0) < 1e-6, f'role 占比之和 {total_pct} != 100')
    check(summary['makespan_ns'] > 0, 'makespan 必须为正')
    check(summary['improper_overlaps'] == 0,
          f'嵌套区间不应被判为部分重叠: {summary["improper_overlaps"]}')

    # Builder hard gate: crossing same-lane slices must return rc=4 before writing.
    partial_doc = make_rank_doc(0, 0, 0, {0: 0}, {0: 1})
    partial_doc['events'] = [
        event(0, TRACK_DISPATCH, 'sync', ENTRY_TASK_ID, 0, 1,
              'ignored', 'sync'),
        event(0, TRACK_DISPATCH, 'dispatch', 0, 5536, 51136,
              'phase', 'dispatch', depth=1),
        event(0, TRACK_DISPATCH, 'dispatch', 1, 47552, 55296,
              'phase', 'dispatch', depth=1),
    ]
    partial_path = write_json(WORK_DIR / 'events_partial_overlap.json', partial_doc)
    partial_out = WORK_DIR / 'trace_partial_overlap.json'
    result_partial = run_builder([str(partial_path), '--out', str(partial_out),
                                  '--align', 'none'])
    check(result_partial.returncode == 4,
          f'partial overlap 应在写文件前 rc=4，实得 {result_partial.returncode}: '
          f'{result_partial.stderr}')
    check(not partial_out.exists(), 'partial overlap 失败后不得留下 trace 文件')
    check('partial overlap' in result_partial.stderr,
          f'partial overlap rc4 应给出明确原因: {result_partial.stderr}')

    # ABI task metadata may supply omitted semantic fields.  The entry sync
    # starts before the release anchor, so one global translation must keep it.
    missing_semantics = json.loads(json.dumps(rank0))
    missing_semantics['events'][0].pop('layer')
    missing_semantics_path = write_json(WORK_DIR / 'events_missing_layer.json',
                                        missing_semantics)
    missing_semantics_out = WORK_DIR / 'trace_missing_layer.json'
    result_missing_semantics = run_builder(
        [str(missing_semantics_path), '--out', str(missing_semantics_out)])
    check(result_missing_semantics.returncode == 0,
          f'ABI fallback + global rebase 应成功 rc=0: {result_missing_semantics.stderr}')
    rebased = load_trace(missing_semantics_out)
    timed = [float(item['ts']) for item in rebased['traceEvents']
             if isinstance(item.get('ts'), (int, float))]
    check(timed and min(timed) >= 0.0, 'global rebase 后所有 numeric ts 必须非负')
    rebase_report = rebased['otherData'].get('timeline_rebase') or {}
    check(rebase_report.get('policy') == 'single_global_translation'
          and rebase_report.get('shift_us', 0) > 0,
          f'必须记录一次全局平移: {rebase_report}')


    derived_input = json.loads(json.dumps(rank0))
    derived_task = next(item for item in derived_input['events'] if item['layer'] == 'task')
    derived_task['is_derived'] = True
    derived_path = write_json(WORK_DIR / 'events_derived_task.json', derived_input)
    derived_out = WORK_DIR / 'trace_derived_task.json'
    result_derived = run_builder([str(derived_path), '--out', str(derived_out)])
    check(result_derived.returncode == 3,
          f'derived task 必须 schema rc=3: {result_derived.stderr}')
    check(not derived_out.exists(), 'derived task 失败时不得写 trace')

    no_contract = json.loads(json.dumps(trace))
    no_contract['traceEvents'] = [item for item in no_contract['traceEvents']
                                  if item.get('name') != 'perfetto_layout_contract']
    no_contract_path = write_json(WORK_DIR / 'trace_no_layout_contract.json', no_contract)
    deliver_proc = subprocess.run(
        [sys.executable, '-B', str(DELIVER_GATE), str(PROFILE_DIR),
         str(no_contract_path), '--skip-validate'],
        capture_output=True, text=True, timeout=60)
    check(deliver_proc.returncode == 1 and 'layout_contract_missing' in deliver_proc.stdout,
          f'deliver gate 必须拒绝缺 layout metadata: rc={deliver_proc.returncode} '
          f'{deliver_proc.stdout} {deliver_proc.stderr}')
    cores = summary['cta_cores']
    check(len(cores) == 6, f'cta_cores 应有 2 rank x 3 CTA = 6 条，实得 {len(cores)}')
    core_of = {(core['rank'], core['block']): core for core in cores}
    check(core_of[(0, 2)]['core_begin_ns'] == 0,
          f'对齐后 rank0/CTA2（最晚 release）的 core 起点应为 0: {core_of[(0, 2)]}')
    check(core_of[(0, 0)]['core_begin_ns'] == release[0] - release[2],
          f'rank0/CTA0 的 core 起点应为 {release[0] - release[2]}: {core_of[(0, 0)]}')
    for core in cores:
        check(core['core_end_ns'] > core['core_begin_ns'], f'core 区间非正: {core}')

    gz_path = WORK_DIR / 'trace.perfetto.json.gz'
    result_gz = run_builder([str(rank0_path), str(rank1_path), '--out', str(gz_path)])
    check(result_gz.returncode == 0, f'gzip 输出失败:\n{result_gz.stderr}')
    check(load_trace(gz_path)['traceEvents'], 'gzip trace 内容为空')

    result_none = run_builder([str(rank0_path), str(rank1_path), '--out',
                               str(WORK_DIR / 'trace_none.json'), '--align', 'none'])
    check(result_none.returncode == 0, f'--align none 失败:\n{result_none.stderr}')
    trace_none = load_trace(WORK_DIR / 'trace_none.json')
    check(trace_none['otherData']['shift_ns'] == 0, '--align none 时 shift 必须为 0')
    none_min_ts = min(item['ts'] for item in trace_none['traceEvents'] if item.get('ph') == 'X')
    check(none_min_ts > 0.0, '--align none 时不应出现负时间戳')
    check(none_min_ts != min(item['ts'] for item in x_events),
          '--align none 与 entry_release 的时间轴必须不同')

    none_per_rank_out = WORK_DIR / 'trace_none_per_rank.json'
    result_none_pr = run_builder([str(rank0_path), str(rank1_path), '--out',
                                  str(none_per_rank_out), '--align', 'none',
                                  '--clock-domain', 'per_rank'])
    check(result_none_pr.returncode == 0,
          f'--align none 与 --clock-domain per_rank 同时给出应正常出图:\n{result_none_pr.stderr}')
    none_pr_other = load_trace(none_per_rank_out)['otherData']
    check(none_pr_other['shift_ns'] == 0 and none_pr_other['shift_ns_by_rank'] == {'0': 0, '1': 0},
          f'--align none 时所有 shift 必须为 0: {none_pr_other["shift_ns_by_rank"]}')
    check(none_pr_other['clock_domain_detail']['applicable'] is False,
          f'--align none 时 clock-domain 应标为不适用: {none_pr_other["clock_domain_detail"]}')
    check(none_pr_other['clock_domain_requested'] == 'per_rank',
          f'请求值仍应被记录: {none_pr_other["clock_domain_requested"]}')
    check(none_pr_other['clock_domain_warning'] is None,
          f'--align none 没有归零动作，不应发 per_rank 警告: '
          f'{none_pr_other["clock_domain_warning"]}')
    check('不适用' in result_none_pr.stdout,
          f'--align none 时必须显式说明 --clock-domain 不适用:\n{result_none_pr.stdout}')

    result_arrive = run_builder([str(rank0_path), str(rank1_path), '--out',
                                 str(WORK_DIR / 'trace_arrive.json'), '--align', 'entry_arrive'])
    check(result_arrive.returncode == 0, f'--align entry_arrive 失败:\n{result_arrive.stderr}')
    trace_arrive = load_trace(WORK_DIR / 'trace_arrive.json')
    expected_arrive_shift = min(global0['entry_sync_start_ns'], global1['entry_sync_start_ns'])
    check(trace_arrive['otherData']['shift_ns'] == expected_arrive_shift,
          f'entry_arrive shift 异常: {trace_arrive["otherData"]["shift_ns"]}')
    check(trace_arrive['otherData']['align_field'] == 'entry_sync_start_ns',
          f'entry_arrive 应使用 entry_sync_start_ns: {trace_arrive["otherData"]["align_field"]}')

    clock_skew_ns = 3_616_875_232
    skew0 = make_rank_doc(0, 0, 1_000_000_000, arrive, release)
    skew1 = make_rank_doc(1, 1, 1_000_000_000 + clock_skew_ns, arrive, release)
    skew0_path = write_json(WORK_DIR / 'events_skew_rank0.json', skew0)
    skew1_path = write_json(WORK_DIR / 'events_skew_rank1.json', skew1)
    skew_global0 = skew0['anchors']['entry_rendezvous']['global']
    skew_global1 = skew1['anchors']['entry_rendezvous']['global']
    basis0 = skew_global0['entry_sync_end_ns']
    basis1 = skew_global1['entry_sync_end_ns']
    check(basis1 - basis0 == clock_skew_ns, '构造的时钟偏移与预期不符')
    visible_skew0 = [item for item in skew0['events'] if item['layer'] != 'ignored']
    true_span_ns = (max(item['end_ns'] for item in visible_skew0)
                    - min(item['start_ns'] for item in visible_skew0))

    auto_out = WORK_DIR / 'trace_skew_auto.json'
    result_auto = run_builder([str(skew0_path), str(skew1_path), '--out', str(auto_out)])
    check(result_auto.returncode == 0, f'跨时钟域输入应正常出图:\n{result_auto.stderr}')
    trace_auto = load_trace(auto_out)
    auto_other = trace_auto['otherData']
    check(auto_other['clock_domain_requested'] == 'auto',
          f'默认请求应为 auto: {auto_other["clock_domain_requested"]}')
    check(auto_other['clock_domain'] == 'per_rank',
          f'跨时钟域输入下 auto 必须切换到 per_rank: {auto_other["clock_domain"]}')
    check(auto_other['shift_ns'] is None,
          f'per_rank 下不存在单一全局 shift，应为 None: {auto_other["shift_ns"]}')
    check(auto_other['shift_ns_by_rank'] == {'0': basis0, '1': basis1},
          f'per_rank 下每个 rank 应以自身基准为 0: {auto_other["shift_ns_by_rank"]}')
    check(auto_other['clock_domain_warning'] is not None
          and '不可直接比较' in auto_other['clock_domain_warning'],
          f'per_rank 必须在 otherData 里警告跨 rank 先后不可比: '
          f'{auto_other["clock_domain_warning"]}')
    auto_detail = auto_other['clock_domain_detail']
    check(auto_detail['raw_barrier']['holds'] is False,
          f'auto 切换 per_rank 的依据应是原始时间戳判据不成立: {auto_detail["raw_barrier"]}')
    check(auto_detail['max_offset']['abs_offset_ns'] == clock_skew_ns,
          f'实测偏移量应为 {clock_skew_ns}: {auto_detail["max_offset"]}')
    check(auto_detail['jitter_reference']['span_ns'] > 0,
          f'抖动参考应可用: {auto_detail["jitter_reference"]}')
    check(auto_detail['offset_over_jitter'] > 10.0,
          f'偏移量应远超抖动参考: {auto_detail["offset_over_jitter"]}')
    check(auto_other['summary']['makespan_ns'] == true_span_ns,
          f'per_rank 的 makespan 应回到真实值 {true_span_ns}: '
          f'{auto_other["summary"]["makespan_ns"]}')
    for token in ('requested=auto resolved=per_rank', str(basis0), str(basis1),
                  f'基准偏移 rank0 - rank1 = {basis0 - basis1}ns',
                  f'最大基准偏移 = {clock_skew_ns}ns', '合理调度抖动量级参考',
                  '切换到 per_rank', str(clock_skew_ns), '不可直接比较'):
        check(token in result_auto.stdout,
              f'auto 判定必须可解释，stdout 缺少 {token}:\n{result_auto.stdout}')
    check('远超合理调度抖动' in result_auto.stdout,
          f'偏移量远超抖动时应明确这么说:\n{result_auto.stdout}')
    check('不可直接比较' in result_auto.stderr,
          f'per_rank 的跨 rank 不可比警告也要进 stderr:\n{result_auto.stderr}')
    auto_x = x_of(trace_auto)
    for item in auto_x:
        if item['args'].get('scope') == 'global_trace':
            continue
        check(abs(item['ts'] * 1000.0
                  - (item['args']['start_ns_raw'] - rank_shift(trace_auto, item))) < 1e-6,
              f'per_rank 下 ts 必须能用本 rank 的 shift 还原: {item}')
        check(abs(item['dur'] * 1000.0 - item['args']['dur_ns']) < 1e-6,
              f'per_rank 下 dur 必须保持纳秒精度: {item}')

    shared_out = WORK_DIR / 'trace_skew_shared.json'
    result_shared = run_builder([str(skew0_path), str(skew1_path), '--out', str(shared_out),
                                '--clock-domain', 'shared'])
    check(result_shared.returncode == 0, f'--clock-domain shared 失败:\n{result_shared.stderr}')
    trace_shared = load_trace(shared_out)
    shared_other = trace_shared['otherData']
    check(shared_other['clock_domain'] == 'shared',
          f'显式 shared 不应被自动改写: {shared_other["clock_domain"]}')
    check(shared_other['shift_ns'] == min(basis0, basis1),
          f'shared 应使用单一全局 shift: {shared_other["shift_ns"]}')
    check(shared_other['shift_ns_by_rank'] == {'0': min(basis0, basis1),
                                              '1': min(basis0, basis1)},
          f'shared 下两个 rank 的 shift 必须相同: {shared_other["shift_ns_by_rank"]}')
    check(shared_other['clock_domain_warning'] is None,
          f'shared 不应带 per_rank 警告: {shared_other["clock_domain_warning"]}')
    check(shared_other['summary']['makespan_ns'] == true_span_ns + clock_skew_ns,
          f'shared 下 makespan 应被时钟偏移撑开到 {true_span_ns + clock_skew_ns}: '
          f'{shared_other["summary"]["makespan_ns"]}')
    shared_barrier = shared_other['barrier_check']
    check(shared_barrier['holds'] is False,
          f'shared 下全局 barrier 判据必须报 VIOLATION: {shared_barrier}')
    check(all(entry['holds'] for entry in shared_barrier['intra_rank']),
          f'每 rank 自身时钟内的判据仍应 PASS: {shared_barrier["intra_rank"]}')
    check(shared_barrier['clock_normalized']['holds'] is True,
          f'归一化判据仍应 PASS: {shared_barrier["clock_normalized"]}')
    check('VIOLATION' in result_shared.stdout,
          f'shared 下应打印 VIOLATION:\n{result_shared.stdout}')

    per_rank_out = WORK_DIR / 'trace_skew_per_rank.json'
    result_per_rank = run_builder([str(skew0_path), str(skew1_path), '--out', str(per_rank_out),
                                   '--clock-domain', 'per_rank'])
    check(result_per_rank.returncode == 0,
          f'--clock-domain per_rank 失败:\n{result_per_rank.stderr}')
    trace_per_rank = load_trace(per_rank_out)
    per_rank_other = trace_per_rank['otherData']
    check(per_rank_other['clock_domain'] == 'per_rank',
          f'显式 per_rank 应生效: {per_rank_other["clock_domain"]}')
    check(per_rank_other['clock_domain_requested'] == 'per_rank',
          f'requested 应记录 per_rank: {per_rank_other["clock_domain_requested"]}')
    check(per_rank_other['shift_ns_by_rank'] == auto_other['shift_ns_by_rank'],
          'per_rank 与 auto 的 per-rank shift 必须一致')
    check(per_rank_other['summary']['makespan_ns'] == auto_other['summary']['makespan_ns']
          and per_rank_other['summary']['window_ns'] == auto_other['summary']['window_ns'],
          'per_rank 与 auto 的时间窗必须一致')
    check([item['ts'] for item in x_of(trace_per_rank)] == [item['ts'] for item in auto_x],
          'per_rank 与 auto 的全部 ts 必须逐条一致')

    skew_expected_x = expected_lane_counts(skew0, skew1)['total_x']
    for label, candidate in (('auto', trace_auto), ('shared', trace_shared),
                             ('per_rank', trace_per_rank)):
        count = len(x_of(candidate))
        check(count == skew_expected_x,
              f'{label} 模式下 ph=X 事件数 {count} != {skew_expected_x}，不能丢事件')

    flat0 = flatten_anchors(rank0)
    flat1 = flatten_anchors(rank1)
    flat0_path = write_json(WORK_DIR / 'events_flat_rank0.json', flat0)
    flat1_path = write_json(WORK_DIR / 'events_flat_rank1.json', flat1)
    flat_out = WORK_DIR / 'trace_flat.json'
    result_flat = run_builder([str(flat0_path), str(flat1_path), '--out', str(flat_out)])
    check(result_flat.returncode == 0, f'旧扁平 anchors 解析失败:\n{result_flat.stderr}')
    trace_flat = load_trace(flat_out)
    check(trace_flat['otherData']['shift_ns'] == min(flat0['anchors']['entry_sync_end_ns'],
                                                     flat1['anchors']['entry_sync_end_ns']),
          f'扁平布局的 shift_ns 异常: {trace_flat["otherData"]["shift_ns"]}')
    for entry in trace_flat['otherData']['summary']['ranks']:
        anchors = entry['anchors']
        check(anchors['sources']['entry_sync_end_ns'] == 'anchors.entry_sync_end_ns',
              f'扁平 release 锚点来源异常: {anchors["sources"]}')
        check(anchors['per_cta_source'] == 'anchors.per_cta',
              f'扁平 per_cta 来源异常: {anchors["per_cta_source"]}')
        check(anchors['sources']['min_end_ns'].startswith('derived-from:anchors.per_cta'),
              f'扁平布局的 min_end_ns 应由 per_cta 推导: {anchors["sources"]}')
        check('anchors.entry_rendezvous.global.min_end_ns' in anchors['missing_paths'],
              f'扁平布局应报告缺失的完整路径: {anchors["missing_paths"]}')
    check(trace_flat['otherData']['barrier_check']['holds'],
          f'扁平布局的 barrier 应通过: {trace_flat["otherData"]["barrier_check"]}')

    bad_rank1 = make_rank_doc(1, 1, 1_000_000_000, {0: 0, 1: 40, 2: 80}, {0: 100, 1: 140, 2: 180})
    bad_path = write_json(WORK_DIR / 'events_rank1_violation.json', bad_rank1)
    violation_out = WORK_DIR / 'trace_violation.json'
    result_bad = run_builder([str(rank0_path), str(bad_path), '--out', str(violation_out)])
    check(result_bad.returncode == 0, f'违规输入仍应正常出图:\n{result_bad.stderr}')
    trace_bad = load_trace(violation_out)
    bad_barrier = trace_bad['otherData']['barrier_check']
    check(bad_barrier['holds'] is False, 'min(min_end) < max(max_start) 时 holds 必须为 False')
    check(bad_barrier['margin_ns'] < 0, f'违规时 margin 必须为负: {bad_barrier["margin_ns"]}')
    check('VIOLATION' in result_bad.stdout and 'WARNING' in result_bad.stdout,
          f'违规时 stdout 必须显式告警:\n{result_bad.stdout}')
    check(str(bad_barrier['min_release']['ns_raw']) in result_bad.stdout
          and str(bad_barrier['max_arrive']['ns_raw']) in result_bad.stdout,
          f'违规时必须打印两个具体纳秒值:\n{result_bad.stdout}')
    check('WARNING' in result_bad.stderr, f'违规时 stderr 也必须告警:\n{result_bad.stderr}')
    bad_domain = trace_bad['otherData']['clock_domain_detail']
    check(trace_bad['otherData']['clock_domain'] == 'per_rank',
          f'原始判据不成立时 auto 必须切到 per_rank: {trace_bad["otherData"]["clock_domain"]}')
    check(bad_domain['offset_over_jitter'] < 10.0,
          f'该用例的偏移量与抖动同量级，比值应很小: {bad_domain["offset_over_jitter"]}')
    check('远超合理调度抖动' not in result_bad.stdout,
          f'偏移量与抖动同量级时不得声称远超抖动:\n{result_bad.stdout}')
    check('同量级' in result_bad.stdout and '也可能是真实的跨 rank 滞后' in result_bad.stdout,
          f'同量级时必须如实说明另一种可能:\n{result_bad.stdout}')

    unwritten_doc = make_rank_doc(0, 0, 1_000_000_000, arrive, release, unwritten=(1,))
    unwritten_path = write_json(WORK_DIR / 'events_unwritten.json', unwritten_doc)
    unwritten_out = WORK_DIR / 'trace_unwritten.json'
    result_unwritten = run_builder([str(unwritten_path), '--out', str(unwritten_out)])
    check(result_unwritten.returncode == 0, f'written=false 用例失败:\n{result_unwritten.stderr}')
    unwritten_processes = metadata_names(load_trace(unwritten_out), 'process_name')
    check('rank+CTA 0:1 / SM43(unwritten)' in unwritten_processes,
          f'written=false 的 CTA 必须在 process_name 里标注: {unwritten_processes}')
    check('written=false' in result_unwritten.stdout,
          f'stdout 应提示 written=false 的 CTA:\n{result_unwritten.stdout}')

    missing = json.loads(json.dumps(rank0))
    for item in missing['events']:
        item.pop('end_ns')
        item.pop('dur_ns')
    missing_path = write_json(WORK_DIR / 'events_missing_end.json', missing)
    result_missing = run_builder([str(missing_path), '--out', str(WORK_DIR / 'trace_missing.json')])
    check(result_missing.returncode == 3, f'缺字段应返回 3，实得 {result_missing.returncode}')
    check('缺少结束时间' in result_missing.stderr and 'end_ns' in result_missing.stderr,
          f'缺字段报错必须点明缺哪个字段:\n{result_missing.stderr}')
    check('自动降级为 schema probe' in result_missing.stderr
          and '顶层键名' in result_missing.stderr,
          f'缺字段时应自动降级并 dump 真实结构:\n{result_missing.stderr}')

    no_anchor = {
        'metadata': {'rank': 0, 'device': 0},
        'block_map': [{'logical_cta': 0, 'physical_smid': 7, 'written': True}],
        'anchors': {},
        'events': [
            event(0, TRACK_MATH_WG0, 'math', 7, 600, 1600,
                  'task', 'gemm', task_role='math'),
            event(0, TRACK_MATH_WG0, 'math', 7, 600, 1600,
                  'phase', 'mma', depth=1),
        ],
    }
    no_anchor_path = write_json(WORK_DIR / 'events_no_anchor.json', no_anchor)
    result_no_anchor = run_builder([str(no_anchor_path), '--out',
                                   str(WORK_DIR / 'trace_no_anchor.json')])
    check(result_no_anchor.returncode == 3,
          f'锚点缺失时应返回 3，实得 {result_no_anchor.returncode}')
    check('anchors.entry_rendezvous.global.entry_sync_end_ns' in result_no_anchor.stderr,
          f'锚点缺失必须打印完整路径:\n{result_no_anchor.stderr}')
    check('anchor entry_sync_end_ns: 取不到' in result_no_anchor.stderr,
          f'降级 probe 必须点明取不到的字段:\n{result_no_anchor.stderr}')
    result_no_anchor_none = run_builder([str(no_anchor_path), '--out',
                                        str(WORK_DIR / 'trace_no_anchor_none.json'),
                                        '--align', 'none'])
    check(result_no_anchor_none.returncode == 0,
          f'--align none 应容忍锚点缺失:\n{result_no_anchor_none.stderr}')
    check('锚点不可用' in result_no_anchor_none.stdout,
          f'--align none 时应说明锚点不可用:\n{result_no_anchor_none.stdout}')

    events_flag_out = WORK_DIR / 'trace_events_flag.json'
    result_flag = run_builder(['--events', str(rank0_path), str(rank1_path),
                               '--out', str(events_flag_out)])
    check(result_flag.returncode == 0, f'--events 显式写法失败:\n{result_flag.stderr}')
    check(events_flag_out.exists(), '--events 不能被当成 --events-schema-probe 前缀缩写吃掉，'
                                    '必须真的出图')
    trace_flag = load_trace(events_flag_out)
    check(sum(1 for item in trace_flag['traceEvents'] if item['ph'] == 'X') == expected['total_x'],
          '--events 与 positional 写法的产物必须一致')
    check('顶层键名' not in result_flag.stdout,
          f'--events 不应静默降级成 schema probe:\n{result_flag.stdout}')

    probe = run_builder([str(rank0_path), '--events-schema-probe'])
    check(probe.returncode == 0, f'--events-schema-probe 失败:\n{probe.stderr}')
    for token in ('顶层键名', 'metadata 键名', 'block_map[0] 键名', 'start_ns',
                  'anchors.entry_rendezvous 键名',
                  'anchors.entry_rendezvous.per_cta[0] 键名',
                  'anchor min_end_ns: '):
        check(token in probe.stdout, f'probe 输出缺少 {token}:\n{probe.stdout}')

    aliased = {
        'meta': {'rank_idx': 7, 'device_idx': 3},
        'block_map': {'0': 11},
        'records': [
            {'logical_cta': 0, 'track_id': 3, 'taskId': ENTRY_TASK_ID, 'begin_ns': 10,
             'duration_ns': 40, 'layer': 'ignored', 'semantic_kind': 'sync', 'is_derived': False},
            {'logical_cta': 0, 'track_id': 3, 'taskId': 7, 'begin_ns': 60, 'duration_ns': 100, 'layer': 'task', 'semantic_kind': 'gemm', 'is_derived': False, 'task_role': 'math'},
            {'logical_cta': 0, 'track_id': 3, 'taskId': 7, 'begin_ns': 60, 'duration_ns': 100, 'layer': 'phase', 'semantic_kind': 'mma', 'is_derived': False},
            {'logical_cta': 1, 'track_id': 1, 'taskId': 5, 'begin_ns': 70, 'duration_ns': 80, 'layer': 'phase', 'semantic_kind': 'load', 'is_derived': False,
             'smid': 99},
        ],
    }
    aliased_path = write_json(WORK_DIR / 'events_aliased.json', aliased)
    aliased_out = WORK_DIR / 'trace_aliased.json'
    result_alias = run_builder([str(aliased_path), '--out', str(aliased_out)])
    check(result_alias.returncode == 0, f'别名字段读取失败:\n{result_alias.stderr}')
    trace_alias = load_trace(aliased_out)
    alias_processes = [name for name in metadata_names(trace_alias, 'process_name')
                       if name.startswith('rank+CTA')]
    check(alias_processes == ['rank+CTA 7:0 / SM11', 'rank+CTA 7:1 / SM99'],
          f'别名 rank/device/block_map 解析失败: {metadata_names(trace_alias, "process_name")}')
    alias_threads = metadata_names(trace_alias, 'thread_name')
    check('warp4（代表 warp4-7）' in alias_threads,
          f'别名 block_map 解析失败: {alias_threads}')
    check('warp2' in alias_threads,
          f'block_map 缺项时应回退到事件自带 smid: {alias_threads}')
    alias_rank = trace_alias['otherData']['summary']['ranks'][0]
    check(alias_rank['anomalies']['derived_end_from_dur'] == 4, 'end_ns 应由 duration_ns 推导')
    check(alias_rank['anchors']['per_cta_source'] == 'events(ABI task is_entry_sync)',
          f'无 anchors 时应从事件推导 per_cta: {alias_rank["anchors"]["per_cta_source"]}')

    coarse_role = {
        'metadata': {'rank': 0, 'device': 0},
        'block_map': [{'logical_cta': 0, 'physical_smid': 7, 'written': True}],
        'events': [
            {'block': 0, 'track': 0, 'role': 'dispatch', 'task_id': ENTRY_TASK_ID,
             'task': 'kEntryRankSync', 'start_ns': 100, 'end_ns': 500, 'layer': 'ignored', 'semantic_kind': 'sync', 'is_derived': False},
            {'block': 0, 'track': 3, 'role': 'math', 'task_id': 7,
             'task': 'kMathL1GemmPipeline', 'start_ns': 600, 'end_ns': 1600, 'layer': 'task', 'semantic_kind': 'gemm', 'is_derived': False, 'task_role': 'math'},
            {'block': 0, 'track': 3, 'role': 'math', 'task_id': 7,
             'task': 'kMathL1GemmPipeline', 'start_ns': 600, 'end_ns': 1600, 'layer': 'phase', 'semantic_kind': 'mma', 'is_derived': False},
            {'block': 0, 'track': 4, 'role': 'math', 'task_id': 10,
             'task': 'kMathL2GemmPipeline', 'start_ns': 700, 'end_ns': 1700, 'layer': 'task', 'semantic_kind': 'gemm', 'is_derived': False, 'task_role': 'math'},
            {'block': 0, 'track': 4, 'role': 'math', 'task_id': 10,
             'task': 'kMathL2GemmPipeline', 'start_ns': 700, 'end_ns': 1700, 'layer': 'phase', 'semantic_kind': 'mma', 'is_derived': False},
        ],
    }
    coarse_path = write_json(WORK_DIR / 'events_coarse_role.json', coarse_role)
    coarse_out = WORK_DIR / 'trace_coarse_role.json'
    result_coarse = run_builder([str(coarse_path), '--out', str(coarse_out)])
    check(result_coarse.returncode == 0, f'笼统 role 用例失败:\n{result_coarse.stderr}')
    trace_coarse = load_trace(coarse_out)
    coarse_threads = metadata_names(trace_coarse, 'thread_name')
    check('warp4（代表 warp4-7）' in coarse_threads
          and 'warp8（代表 warp8-11）' in coarse_threads,
          f'role=math 时泳道名仍应带 warpgroup/warp 身份: {coarse_threads}')
    coarse_roles = {entry['role']: entry for entry in trace_coarse['otherData']['summary']['roles']}
    check(coarse_roles['math']['lanes'] == 2,
          f'role 汇总应按实际 math observer 泳道计数: {coarse_roles}')

    coarse_spans = [item for item in trace_coarse['traceEvents']
                    if item.get('ph') == 'X']
    coarse_claim_fails, _ = check_task_claim_contract(trace_coarse, coarse_spans)
    check(coarse_claim_fails,
          '旧 coarse trace 缺少 scheduler claim identity 时必须被 task-claim contract 拒绝')
    coarse_self_fails, _ = check_self_evident(trace_coarse)
    check(any(fail_id in {
                  'task_instance_missing', 'scheduler_claim_audit_missing',
                  'scheduler_claim_task_mismatch'}
              for fail_id, _ in coarse_self_fails),
          f'旧 coarse trace 必须经 deliver_gate 主路径失败: {coarse_self_fails}')

    print('selftest_perfetto: semantic gates PASS '
          '(source GEMM task, independent phase, partial overlap, global ts rebase)')
    print(f'selftest_perfetto: PASS ({expected["total_x"]} X 事件 = '
          f'{expected["measured_x"]} 实测 + 0 X复制, 输出目录 {WORK_DIR})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
