import argparse
import gzip
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple

PID_BASE = 1000

# Logical task records are observer-warp source observations.  They are not
# rendered on every CTA lane: the post-processor emits one derived envelope
# for each logical GEMM over the complete, already-aligned trace.
SKILL_ROOT = Path(__file__).resolve().parents[3]
LAYOUT_CHECKER_PATH = SKILL_ROOT / 'weaveprof/pipeline/scan' / 'layout_contract.py'
CAPTURE_SCOPE_PATH = Path(__file__).resolve().parent / 'capture_scope.py'


def load_layout_checker():
    spec = importlib.util.spec_from_file_location('fast_instrument_layout_contract',
                                                  LAYOUT_CHECKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load layout checker: {LAYOUT_CHECKER_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LAYOUT_CHECKER = load_layout_checker()


def load_capture_scope_checker():
    spec = importlib.util.spec_from_file_location('fast_instrument_capture_scope',
                                                  CAPTURE_SCOPE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load capture scope checker: {CAPTURE_SCOPE_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAPTURE_SCOPE = load_capture_scope_checker()


EVENT_LIST_KEYS = ('events', 'intervals', 'records', 'spans')
BLOCK_KEYS = ('block', 'logical_cta', 'cta', 'block_idx', 'blockIdx', 'logical_block')
TRACK_KEYS = ('track', 'track_id', 'track_idx', 'trackId')
START_KEYS = ('start_ns', 'start', 'begin_ns', 'ts_start_ns', 'start_timestamp_ns')
END_KEYS = ('end_ns', 'end', 'stop_ns', 'ts_end_ns', 'end_timestamp_ns')
DUR_KEYS = ('dur_ns', 'duration_ns', 'dur', 'duration')
TASK_ID_KEYS = ('task_id', 'taskId', 'task_index', 'task_idx')
TASK_NAME_KEYS = ('task', 'task_name', 'taskName')
SHORT_NAME_KEYS = ('short_name', 'shortName', 'label')
ROLE_KEYS = ('role', 'role_name', 'lane', 'track_role')
TRACK_LABEL_KEYS = ('track_label', 'trackLabel', 'lane_label')
DEPTH_KEYS = ('depth', 'nest_depth', 'level')
EXPERT_KEYS = ('expert_idx', 'expert', 'expert_id', 'expertIdx')
RANK_KEYS = ('rank', 'rank_idx', 'rank_id', 'ep_rank')
DEVICE_KEYS = ('device', 'device_idx', 'device_id', 'gpu', 'gpu_index')
SMID_KEYS = ('physical_smid', 'smid', 'sm', 'sm_id', 'physical_sm')
LAYER_KEYS = ('layer',)
SEMANTIC_KIND_KEYS = ('semantic_kind',)
IS_DERIVED_KEYS = ('is_derived',)
TASK_ROLE_KEYS = ('task_role',)
WRITTEN_KEYS = ('written', 'valid', 'has_record')
CTA_ARRIVE_KEYS = ('entry_sync_start_ns', 'entry_rank_sync_start_ns', 'rank_arrive_ns',
                   'arrive_ns', 'start_ns')
CTA_RELEASE_KEYS = ('entry_sync_end_ns', 'entry_rank_sync_end_ns', 'rank_release_ns',
                    'release_ns', 'end_ns')
ANCHOR_CONTAINERS = ('entry_rendezvous', 'entry_rank_sync', 'entry_sync', 'rendezvous', '')


class EventsSchemaError(Exception):
    pass


def anchor_paths(*fields: str) -> Tuple[str, ...]:
    paths: List[str] = []
    for container in ANCHOR_CONTAINERS:
        base = 'anchors' if not container else f'anchors.{container}'
        for field in fields:
            paths.append(f'{base}.global.{field}')
            paths.append(f'{base}.{field}')
    return tuple(paths)


ANCHOR_FIELD_PATHS: Dict[str, Tuple[str, ...]] = {
    'entry_sync_start_ns': anchor_paths('entry_sync_start_ns', 'entry_rank_sync_start_ns',
                                        'rank_arrive_ns', 'arrive_ns'),
    'entry_sync_end_ns': anchor_paths('entry_sync_end_ns', 'entry_rank_sync_end_ns',
                                      'rank_release_ns', 'release_ns'),
    'min_start_ns': anchor_paths('min_start_ns'),
    'max_start_ns': anchor_paths('max_start_ns'),
    'min_end_ns': anchor_paths('min_end_ns'),
    'max_end_ns': anchor_paths('max_end_ns'),
    'start_skew_ns': anchor_paths('start_skew_ns'),
    'end_skew_ns': anchor_paths('end_skew_ns'),
}
PER_CTA_PATHS = anchor_paths('per_cta', 'per_block', 'ctas', 'blocks', 'cta_anchors')
ALIGN_FIELD = {'entry_release': 'entry_sync_end_ns', 'entry_arrive': 'entry_sync_start_ns'}
BARRIER_ARRIVE_FIELD = 'max_start_ns'
BARRIER_RELEASE_FIELD = 'min_end_ns'
CLOCK_DOMAIN_CHOICES = ('auto', 'shared', 'per_rank')
CLOCK_DOMAIN_CRITERION = (f'barrier 判据 min(rank.{BARRIER_RELEASE_FIELD}) >= '
                          f'max(rank.{BARRIER_ARRIVE_FIELD}) 在原始时间戳上是否成立')
CLOCK_SKEW_RATIO_THRESHOLD = 10.0
PER_RANK_CLOCK_WARNING = (
    'per_rank 模式下每个 rank 的时间轴已各自以自身对齐基准归零，'
    '跨 rank 的绝对先后关系不可直接比较；只能比较同一 rank 内部的相对时序与时长')


def pick(mapping: Any, keys: Sequence[str]) -> Tuple[Optional[str], Any]:
    if not isinstance(mapping, dict):
        return None, None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return key, mapping[key]
    return None, None


def lookup_path(root: Any, paths: Sequence[str]) -> Tuple[Optional[str], Any]:
    for path in paths:
        node: Any = root
        for part in path.split('.'):
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(part)
            if node is None:
                break
        if node is not None:
            return path, node
    return None, None


def require(mapping: Any, keys: Sequence[str], context: str) -> Any:
    key, value = pick(mapping, keys)
    if key is None:
        seen = sorted(mapping) if isinstance(mapping, dict) else type(mapping).__name__
        raise EventsSchemaError(
            f'{context}: 缺少字段（完整路径 {context}.{keys[0]}，兼容别名 {list(keys[1:])}）。'
            f' 实际看到的字段: {seen}')
    return value


def as_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise EventsSchemaError(f'{context}: 期望整数，得到 bool')
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value, 10)
        except ValueError:
            pass
    raise EventsSchemaError(f'{context}: 期望整数，得到 {value!r} ({type(value).__name__})')


def load_json(path: Path) -> Any:
    try:
        if path.name.endswith('.gz'):
            with gzip.open(path, 'rt', encoding='utf-8') as handle:
                return json.load(handle)
        else:
            with path.open('r', encoding='utf-8') as handle:
                return json.load(handle)
    except OSError as error:
        raise EventsSchemaError(f'{path}: 无法读取: {error}') from error
    except json.JSONDecodeError as error:
        raise EventsSchemaError(f'{path}: JSON 解析失败: {error}') from error


# SPEC-1: one canonical delivery filename, including the hardware card token.
CARD_TARGETS = {
    'h20': {'platform': 'nv', 'arches': frozenset(('sm_90',))},
    'b300': {'platform': 'nv', 'arches': frozenset(('sm_100a', 'sm_103a'))},
    'mi308x': {'platform': 'amd', 'arches': frozenset(('gfx942',))},
    'sm86': {'platform': 'nv', 'arches': frozenset(('sm_86',))},
}
BUNDLE_TOKENS = {'sw': 'sw', 'hw': 'hw', 'sw+hw': 'sw_hw'}
TRACE_FILENAME_RE = re.compile(
    r'^trace__(?P<model>[a-z0-9]+)_(?P<stage>[a-z0-9]+)'
    r'__card-(?P<card>h20|b300|mi308x|sm86)'
    r'__bs(?P<batch>\d+)_sq(?P<seqlen_q>\d+)(?:_kv(?P<seqlen_kv>\d+))?'
    r'__(?P<dtype>[a-z0-9]+)'
    r'__grid(?P<grid>\d+)_warp(?P<warp>\d+)_ph(?P<phases>\d+)'
    r'__(?P<bundle>sw|hw|sw_hw)\.perfetto\.json\.gz$')


def _slug_token(value: Any, context: str) -> str:
    if value is None:
        raise EventsSchemaError(f'{context}: 缺少值，无法生成规范 trace 文件名')
    token = re.sub(r'[^a-z0-9]', '', str(value).lower())
    if not token:
        raise EventsSchemaError(f'{context}={value!r}: 无法规范化为 [a-z0-9]+')
    return token


def _int_token(value: Any, context: str) -> str:
    if isinstance(value, bool):
        raise EventsSchemaError(f'{context}: 期望正整数，得到 bool')
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and float(value).is_integer():
        number = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip(), 10)
    elif isinstance(value, (list, tuple)) and value:
        product = 1
        for item in value:
            token = _int_token(item, context)
            product *= int(token)
        number = product
    else:
        raise EventsSchemaError(f'{context}: 期望正整数，得到 {value!r}')
    if number <= 0:
        raise EventsSchemaError(f'{context}: 必须大于 0，得到 {number}')
    return str(number)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def contract_field(contract: Any, section: Optional[str], *keys: str) -> Any:
    if not isinstance(contract, dict):
        return None
    if section is not None:
        sub = contract.get(section)
        if isinstance(sub, dict):
            _, value = pick(sub, keys)
            if value is not None:
                return value
    _, value = pick(contract, keys)
    return value


def resolve_bundle_token(bundle: Any) -> str:
    if bundle is None:
        return 'sw'
    token = str(bundle).strip().lower()
    token = BUNDLE_TOKENS.get(token)
    if token is None:
        raise EventsSchemaError(
            f'information_bundle={bundle!r} 非法；只接受 sw、hw、sw+hw')
    return token


def validate_contract_target(contract: Any) -> Dict[str, str]:
    """Return normalized target identity or fail closed.

    h20 => nv/sm_90, b300 => nv/sm_100a|sm_103a, mi308x => amd/gfx942.
    """
    target = contract.get('target') if isinstance(contract, dict) else None
    if not isinstance(target, dict):
        raise EventsSchemaError('contract.target 必须是 object 且声明 card/platform/arch')
    raw_card = target.get('card')
    card = raw_card.strip().lower() if isinstance(raw_card, str) else None
    if card not in CARD_TARGETS:
        raise EventsSchemaError(
            f'contract.target.card={raw_card!r} 非法；只接受 h20、b300、mi308x')
    raw_platform = target.get('platform')
    platform = raw_platform.strip().lower() if isinstance(raw_platform, str) else None
    platform = {'nvidia': 'nv', 'cuda': 'nv', 'rocm': 'amd', 'hip': 'amd'}.get(
        platform, platform)
    if platform is None or platform not in ('nv', 'amd'):
        raise EventsSchemaError(
            f'contract.target.platform={raw_platform!r} 非法；只接受 nv、amd')
    raw_arch = target.get('arch')
    arch = raw_arch.strip().lower() if isinstance(raw_arch, str) else None
    if arch is None or not arch:
        raise EventsSchemaError('contract.target.arch 必须是非空字符串')
    expected = CARD_TARGETS[card]
    if platform != expected['platform']:
        raise EventsSchemaError(
            f'card {card} 必须对应 platform={expected["platform"]}，得到 {platform}')
    if arch not in expected['arches']:
        raise EventsSchemaError(
            f'card {card} 只接受 arch={"|".join(sorted(expected["arches"]))}，得到 {arch}')
    normalized = dict(target)
    normalized.update({'card': card, 'platform': platform, 'arch': arch})
    return normalized


def build_trace_filename(contract: Any, bundle: str = 'sw', grid: Any = None,
                         warps_per_cta: Any = None,
                         num_phases: Any = None) -> str:
    """Build the one canonical SPEC-1 filename from contract metadata."""
    target = validate_contract_target(contract)
    model = _slug_token(contract_field(contract, 'target', 'model', 'model_name', 'name'),
                        'contract.target.model')
    stage = _slug_token(contract_field(contract, 'target', 'stage', 'phase', 'mode'),
                        'contract.target.stage')
    batch = _int_token(contract_field(contract, 'input', 'batch', 'batch_size', 'bs'),
                       'contract.input.batch')
    seqlen_q = _int_token(contract_field(contract, 'input',
                                         'seqlen_q', 'seq_len_q', 'seqlen', 'sq'),
                          'contract.input.seqlen_q')
    dtype = _slug_token(contract_field(contract, 'input',
                                       'dtype', 'data_type', 'precision'),
                        'contract.input.dtype')
    seqlen_kv = contract_field(contract, 'input', 'seqlen_kv', 'seq_len_kv',
                               'kv_len', 'kv')
    kv_part = ''
    if seqlen_kv is not None:
        kv_part = '_kv' + _int_token(seqlen_kv, 'contract.input.seqlen_kv')
    grid_token = _int_token(_first_present(
        contract_field(contract, 'build_and_launch', 'grid', 'grid_size',
                       'num_blocks', 'num_ctas', 'grid_dim'),
        contract_field(contract, 'runtime_launch', 'grid', 'grid_size',
                       'num_blocks', 'num_ctas', 'grid_dim'),
        contract_field(contract, 'launch', 'grid', 'grid_size', 'num_blocks'),
        contract.get('grid') if isinstance(contract, dict) else None,
        grid), 'grid')
    warp_token = _int_token(_first_present(
        contract_field(contract, 'build_and_launch', 'warps_per_block',
                       'warps_per_cta', 'num_warps'),
        contract_field(contract, 'runtime_launch', 'warps_per_block',
                       'warps_per_cta', 'num_warps'),
        contract.get('warps_per_block') if isinstance(contract, dict) else None,
        warps_per_cta), 'warps_per_block')
    phase_token = _int_token(_first_present(
        contract_field(contract, 'build_and_launch', 'phases', 'num_phases'),
        contract_field(contract, 'runtime_launch', 'phases', 'num_phases'),
        contract.get('phases') if isinstance(contract, dict) else None,
        num_phases), 'phases')
    filename = (
        f'trace__{model}_{stage}__card-{target["card"]}__bs{batch}_sq{seqlen_q}'
        f'{kv_part}__{dtype}__grid{grid_token}_warp{warp_token}_ph{phase_token}'
        f'__{resolve_bundle_token(bundle)}.perfetto.json.gz')
    if TRACE_FILENAME_RE.match(filename) is None:
        raise EventsSchemaError(f'生成的 trace 文件名不符合 SPEC-1: {filename}')
    return filename


def validate_explicit_trace_filename(path: Path, contract: Any = None) -> None:
    """Explicit staging names may be non-delivery, but trace__ names are strict."""
    if not path.name.startswith('trace__'):
        return
    match = TRACE_FILENAME_RE.match(path.name)
    if match is None:
        raise EventsSchemaError(
            f'显式 trace 文件名不符合 SPEC-1（必须包含 card-h20|b300|mi308x）: {path.name}')
    if match.group('bundle') != 'sw':
        raise EventsSchemaError(
            f'build_perfetto 只产出 __sw.perfetto.json.gz，显式文件名却是 __{match.group("bundle")}')
    if contract is None:
        raise EventsSchemaError(
            'trace__ 交付文件名必须同时提供 --contract；'
            '没有 target.card 的契约不能生成可交付 trace')
    target = validate_contract_target(contract)
    if match.group('card') != target['card']:
        raise EventsSchemaError(
            f'文件名 card={match.group("card")} 与契约 target.card={target["card"]} 不符')


def load_abi_tasks(path: Path) -> Dict[int, Dict[str, Any]]:
    abi = load_json(path)
    raw_tasks = abi.get('tasks') if isinstance(abi, dict) else None
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise EventsSchemaError(f'{path}: ABI.tasks 必须是非空列表')
    result: Dict[int, Dict[str, Any]] = {}
    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, dict) or not isinstance(raw.get('id'), int):
            raise EventsSchemaError(f'{path}: tasks[{index}] 缺少整数 id')
        task_id = raw['id']
        if task_id in result:
            raise EventsSchemaError(f'{path}: tasks 重复 id={task_id}')
        if not isinstance(raw.get('name'), str) or not raw['name']:
            raise EventsSchemaError(f'{path}: tasks[{index}] 缺少 name')
        if not isinstance(raw.get('short_name'), str) or not raw['short_name']:
            raise EventsSchemaError(f'{path}: tasks[{index}] 缺少 short_name')
        result[task_id] = dict(raw)
    return result


def find_event_list(doc: Any, path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    if not isinstance(doc, dict):
        raise EventsSchemaError(f'{path}: 顶层必须是 JSON object，实际是 {type(doc).__name__}')
    for key in EVENT_LIST_KEYS:
        value = doc.get(key)
        if isinstance(value, list):
            if value and not isinstance(value[0], dict):
                raise EventsSchemaError(f'{path}: "{key}" 必须是 object 列表，元素类型是 '
                                        f'{type(value[0]).__name__}')
            return key, value
    raise EventsSchemaError(
        f'{path}: 找不到事件列表，尝试过的完整路径 {[k for k in EVENT_LIST_KEYS]}；'
        f'顶层实际键名: {sorted(doc)}')


def probe_dict_tree(label: str, node: Any, stream: TextIO, depth: int = 0) -> None:
    if depth > 2 or not isinstance(node, dict):
        return
    print(f'  {label} 键名: {sorted(node)}', file=stream)
    for key, value in sorted(node.items()):
        child = f'{label}.{key}'
        if isinstance(value, dict):
            probe_dict_tree(child, value, stream, depth + 1)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            print(f'  {child}[0] 键名: {sorted(value[0])}（共 {len(value)} 项）', file=stream)


def probe_document(path: Path, doc: Any, stream: TextIO) -> None:
    print(f'--- {path}', file=stream)
    if not isinstance(doc, dict):
        print(f'  顶层类型: {type(doc).__name__}（期望 object）', file=stream)
        return
    print(f'  顶层键名: {sorted(doc)}', file=stream)
    for key in ('metadata', 'meta'):
        value = doc.get(key)
        if isinstance(value, dict):
            print(f'  {key} 键名: {sorted(value)}', file=stream)
    probe_dict_tree('anchors', doc.get('anchors'), stream)
    for field, paths in ANCHOR_FIELD_PATHS.items():
        found, value = lookup_path(doc, paths)
        if found is None:
            print(f'  anchor {field}: 取不到，尝试过的完整路径 {list(paths)}', file=stream)
        else:
            print(f'  anchor {field}: {value} <- {found}', file=stream)
    found, _ = lookup_path(doc, PER_CTA_PATHS)
    if found is None:
        print(f'  anchor per_cta: 取不到，尝试过的完整路径 {list(PER_CTA_PATHS)}', file=stream)
    block_map = doc.get('block_map')
    if isinstance(block_map, list) and block_map:
        first = block_map[0]
        print(f'  block_map[0] 键名: '
              f'{sorted(first) if isinstance(first, dict) else type(first).__name__}'
              f'（共 {len(block_map)} 项）', file=stream)
    elif isinstance(block_map, dict) and block_map:
        print(f'  block_map 是 dict，共 {len(block_map)} 项', file=stream)
    else:
        print('  block_map: 缺失，thread_name 将退化成 SM?', file=stream)
    for key in EVENT_LIST_KEYS:
        events = doc.get(key)
        if isinstance(events, list):
            field_counts: Dict[str, int] = {}
            for event in events:
                if isinstance(event, dict):
                    for field in event:
                        field_counts[field] = field_counts.get(field, 0) + 1
            print(f'  "{key}" 共 {len(events)} 条，字段出现次数: '
                  f'{dict(sorted(field_counts.items()))}', file=stream)
            if events and isinstance(events[0], dict):
                print(f'  "{key}"[0] = {json.dumps(events[0], ensure_ascii=False, sort_keys=True)}',
                      file=stream)
            break


def parse_block_map(doc: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    raw = doc.get('block_map')
    mapping: Dict[int, Dict[str, Any]] = {}
    entries: List[Tuple[Any, Any]] = []
    if isinstance(raw, list):
        entries = list(enumerate(raw))
    elif isinstance(raw, dict):
        entries = list(raw.items())
    for identifier, entry in entries:
        context = f'block_map[{identifier}]'
        if isinstance(entry, dict):
            block_key, block = pick(entry, BLOCK_KEYS)
            block_id = as_int(block, f'{context}.{block_key}') if block_key is not None \
                else as_int(identifier, f'{context} key')
            smid_key, smid = pick(entry, SMID_KEYS)
            if smid_key is None:
                continue
            written_key, written = pick(entry, WRITTEN_KEYS)
            mapping[block_id] = {
                'physical_smid': as_int(smid, f'{context}.{smid_key}'),
                'written': True if written_key is None else bool(written),
            }
        elif entry is not None:
            block_id = as_int(identifier, f'{context} key')
            mapping[block_id] = {'physical_smid': as_int(entry, context), 'written': True}
    return mapping


def rank_from_path(path: Path) -> Optional[int]:
    match = re.search(r'rank[_-]?(\d+)', path.name)
    return int(match.group(1)) if match else None


def normalize_events(events_key: str, events: List[Dict[str, Any]], path: Path,
                     task_specs: Dict[int, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    normalized: List[Dict[str, Any]] = []
    anomalies = {'negative_duration': 0, 'derived_end_from_dur': 0}
    for index, event in enumerate(events):
        context = f'{path}: {events_key}[{index}]'
        if not isinstance(event, dict):
            raise EventsSchemaError(f'{context}: 期望 object，实际 {type(event).__name__}')
        block = as_int(require(event, BLOCK_KEYS, context), f'{context}.block')
        track = as_int(require(event, TRACK_KEYS, context), f'{context}.track')
        start = as_int(require(event, START_KEYS, context), f'{context}.start_ns')
        end_key, end_value = pick(event, END_KEYS)
        if end_key is None:
            dur_key, dur_value = pick(event, DUR_KEYS)
            if dur_key is None:
                raise EventsSchemaError(
                    f'{context}: 缺少结束时间，完整路径 {context}.end_ns 和 {context}.dur_ns 都不存在'
                    f'（兼容别名 {list(END_KEYS[1:])} / {list(DUR_KEYS[1:])}）。'
                    f' 实际字段: {sorted(event)}')
            end = start + as_int(dur_value, f'{context}.{dur_key}')
            anomalies['derived_end_from_dur'] += 1
        else:
            end = as_int(end_value, f'{context}.{end_key}')
        if end < start:
            anomalies['negative_duration'] += 1
            end = start

        task_key, task_value = pick(event, TASK_ID_KEYS)
        task_id = as_int(task_value, f'{context}.{task_key}') if task_key is not None else None
        spec = task_specs.get(task_id) if task_id is not None else None
        if task_id is not None and spec is None:
            raise EventsSchemaError(f'{context}: task_id={task_id} 不在 ABI.tasks 中')
        task_name = None if spec is None else spec['name']
        short_name = None if spec is None else spec['short_name']
        _, role = pick(event, ROLE_KEYS)
        if not isinstance(role, str):
            role = f'observer_{track}'
        _, track_label = pick(event, TRACK_LABEL_KEYS)
        if not isinstance(track_label, str):
            track_label = None

        layer_key, layer_value = pick(event, LAYER_KEYS)
        layer = layer_value if layer_key is not None else (spec or {}).get('layer')
        if layer is None:
            raise EventsSchemaError(f'{context}: 缺少 layer，ABI.tasks 也未提供 layer')
        if layer not in ('task', 'phase', 'ignored'):
            raise EventsSchemaError(
                f'{context}.layer: expected task/phase/ignored, got {layer!r}')
        semantic_key, semantic_value = pick(event, SEMANTIC_KIND_KEYS)
        semantic_kind = semantic_value if semantic_key is not None \
            else (spec or {}).get('semantic_kind')
        if semantic_kind is None:
            raise EventsSchemaError(f'{context}: 缺少 semantic_kind，ABI.tasks 也未提供 semantic_kind')
        if not isinstance(semantic_kind, str) or not semantic_kind:
            raise EventsSchemaError(
                f'{context}.semantic_kind: expected non-empty string, got {semantic_kind!r}')
        is_derived = require(event, IS_DERIVED_KEYS, context)
        if not isinstance(is_derived, bool):
            raise EventsSchemaError(
                f'{context}.is_derived: expected bool, got {is_derived!r}')
        _, task_role = pick(event, TASK_ROLE_KEYS)
        if task_role is not None and not isinstance(task_role, str):
            raise EventsSchemaError(
                f'{context}.task_role: expected string when present, got {task_role!r}')
        if layer == 'task' and is_derived:
            raise EventsSchemaError(
                f'{context}: layer=task requires source measurement (is_derived=false)')
        if layer == 'task' and task_id is None:
            raise EventsSchemaError(f'{context}: layer=task requires explicit task_id')

        depth_key, depth_value = pick(event, DEPTH_KEYS)
        depth = as_int(depth_value, f'{context}.{depth_key}') if depth_key is not None else None
        expert_key, expert_value = pick(event, EXPERT_KEYS)
        expert_idx = as_int(expert_value, f'{context}.{expert_key}') \
            if expert_key is not None else None
        source_args = event.get('args') if isinstance(event.get('args'), dict) else {}
        smid_key, smid_value = pick(event, SMID_KEYS)
        if smid_key is None:
            smid_key, smid_value = pick(source_args, SMID_KEYS)
        smid = as_int(smid_value, f'{context}.{smid_key}') if smid_key is not None else None
        name = short_name or task_name or role
        claim_fields = (
            'task_instance_id', 'claim_kind', 'claim_ticket',
            'claim_success', 'claim_instance_count', 'group', 'm_block',
            'n_block', 'first_grouped_row', 'rows', 'token')
        claim_data = {
            key: event[key] if key in event else source_args[key]
            for key in claim_fields if key in event or key in source_args
        }

        normalized.append({
            'block': block, 'track': track, 'role': role, 'name': name,
            'task_id': task_id, 'task': task_name, 'short_name': short_name,
            'task_spec': spec,
            'viewer_policy': None if spec is None else spec.get('viewer_policy'),
            'track_label': track_label, 'depth': depth,
            'layer': layer, 'semantic_kind': semantic_kind,
            'is_derived': is_derived, 'task_role': task_role,
            'expert_idx': expert_idx, 'physical_smid': smid,
            'start_ns': start, 'end_ns': end, 'dur_ns': end - start,
            **claim_data,
        })
    return normalized, anomalies


def entry_sync_from_events(events: List[Dict[str, Any]]) -> Dict[int, Dict[str, int]]:
    per_cta: Dict[int, Dict[str, int]] = {}
    for event in events:
        spec = event.get('task_spec') or {}
        is_entry = spec.get('is_entry_sync') is True
        if not is_entry:
            continue
        current = per_cta.get(event['block'])
        if current is None:
            per_cta[event['block']] = {'arrive_ns': event['start_ns'],
                                       'release_ns': event['end_ns']}
        else:
            current['arrive_ns'] = min(current['arrive_ns'], event['start_ns'])
            current['release_ns'] = max(current['release_ns'], event['end_ns'])
    return per_cta


def resolve_per_cta(doc: Dict[str, Any],
                    events: List[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, int]], str]:
    found, raw = lookup_path(doc, PER_CTA_PATHS)
    per_cta: Dict[int, Dict[str, int]] = {}
    if found is not None:
        entries: List[Tuple[Any, Any]] = []
        if isinstance(raw, list):
            entries = list(enumerate(raw))
        elif isinstance(raw, dict):
            entries = list(raw.items())
        for identifier, entry in entries:
            if not isinstance(entry, dict):
                continue
            context = f'{found}[{identifier}]'
            block_key, block = pick(entry, BLOCK_KEYS)
            block_id = as_int(block, f'{context}.{block_key}') if block_key is not None \
                else as_int(identifier, f'{found} key')
            arrive_key, arrive = pick(entry, CTA_ARRIVE_KEYS)
            release_key, release = pick(entry, CTA_RELEASE_KEYS)
            if arrive_key is None or release_key is None:
                continue
            per_cta[block_id] = {
                'arrive_ns': as_int(arrive, f'{context}.{arrive_key}'),
                'release_ns': as_int(release, f'{context}.{release_key}'),
            }
        if per_cta:
            return per_cta, found
    per_cta = entry_sync_from_events(events)
    return per_cta, 'events(ABI task is_entry_sync)' if per_cta else ''


def per_cta_derived(per_cta: Dict[int, Dict[str, int]]) -> Dict[str, int]:
    if not per_cta:
        return {}
    starts = [entry['arrive_ns'] for entry in per_cta.values()]
    ends = [entry['release_ns'] for entry in per_cta.values()]
    return {
        'entry_sync_start_ns': min(starts),
        'entry_sync_end_ns': max(ends),
        'min_start_ns': min(starts),
        'max_start_ns': max(starts),
        'min_end_ns': min(ends),
        'max_end_ns': max(ends),
    }


def resolve_anchors(doc: Dict[str, Any], events: List[Dict[str, Any]],
                    path: Path) -> Dict[str, Any]:
    per_cta, per_cta_source = resolve_per_cta(doc, events)
    fields: Dict[str, Optional[int]] = {}
    sources: Dict[str, Optional[str]] = {}
    missing: List[str] = []
    for field, paths in ANCHOR_FIELD_PATHS.items():
        found, value = lookup_path(doc, paths)
        if found is None:
            fields[field] = None
            sources[field] = None
            missing.append(paths[0])
        else:
            fields[field] = as_int(value, f'{path}: {found}')
            sources[field] = found
    for field, value in per_cta_derived(per_cta).items():
        if fields[field] is None:
            fields[field] = value
            sources[field] = f'derived-from:{per_cta_source}'
    for skew, low, high in (('start_skew_ns', 'min_start_ns', 'max_start_ns'),
                            ('end_skew_ns', 'min_end_ns', 'max_end_ns')):
        if fields[skew] is None and fields[low] is not None and fields[high] is not None:
            fields[skew] = fields[high] - fields[low]
            sources[skew] = f'derived-from:{high}-{low}'
    if fields['entry_sync_start_ns'] is None or fields['entry_sync_end_ns'] is None:
        raise EventsSchemaError(
            f'{path}: 解析不到 kEntryRankSync 锚点。缺失字段的完整路径（首选路径）: '
            f'{[ANCHOR_FIELD_PATHS[f][0] for f in ("entry_sync_start_ns", "entry_sync_end_ns") if fields[f] is None]}；'
            f' 全部候选路径: entry_sync_start_ns -> {list(ANCHOR_FIELD_PATHS["entry_sync_start_ns"])}, '
            f'entry_sync_end_ns -> {list(ANCHOR_FIELD_PATHS["entry_sync_end_ns"])}；'
            f' per_cta 候选路径 -> {list(PER_CTA_PATHS)}；'
            f' 事件里也没有 ABI 标记的 entry sync 记录。可改用 --align none 跳过对齐。')
    return {
        'fields': fields,
        'sources': sources,
        'per_cta': per_cta,
        'per_cta_source': per_cta_source,
        'missing_paths': missing,
    }


def load_rank_document(path: Path, fallback_rank: int,
                       task_specs: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    doc = load_json(path)
    events_key, raw_events = find_event_list(doc, path)
    if not raw_events:
        raise EventsSchemaError(f'{path}: 完整路径 {path.name}:{events_key} 为空，没有可视化内容')
    events, anomalies = normalize_events(events_key, raw_events, path, task_specs)
    # Reuse normalized events in the document so the original raw event list
    # can be released instead of keeping two copies alive.
    doc[events_key] = events
    metadata = doc.get('metadata') if isinstance(doc.get('metadata'), dict) else {}
    if not metadata and isinstance(doc.get('meta'), dict):
        metadata = doc['meta']

    rank_key, rank_value = pick(metadata, RANK_KEYS)
    rank_source = f'metadata.{rank_key}'
    if rank_key is None:
        rank_key, rank_value = pick(doc, RANK_KEYS)
        rank_source = f'top-level.{rank_key}'
    if rank_key is None:
        derived = rank_from_path(path)
        rank_value = derived if derived is not None else fallback_rank
        rank_source = 'filename' if derived is not None else 'argument-order'
    rank = as_int(rank_value, f'{path}: rank')

    device_key, device_value = pick(metadata, DEVICE_KEYS)
    device: Optional[int]
    if device_key is None:
        device = None
    else:
        try:
            device = as_int(device_value, f'{path}: metadata.{device_key}')
        except EventsSchemaError:
            device = None
    return {
        'path': path,
        'raw': doc,
        'rank': rank,
        'rank_source': rank_source,
        'device': device,
        'metadata': metadata,
        'block_map': parse_block_map(doc),
        'events': events,
        'events_key': events_key,
        'anomalies': anomalies,
        'anchors': None,
        'anchors_error': None,
    }


def raw_barrier_verdict(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    arrivals: List[Dict[str, Any]] = []
    releases: List[Dict[str, Any]] = []
    unavailable: List[str] = []
    for doc in docs:
        anchors = doc['anchors']
        if anchors is None:
            unavailable.append(f'rank{doc["rank"]}: 锚点解析失败')
            continue
        arrive = anchors['fields'].get(BARRIER_ARRIVE_FIELD)
        release = anchors['fields'].get(BARRIER_RELEASE_FIELD)
        if arrive is None:
            unavailable.append(f'rank{doc["rank"]}: '
                               f'{ANCHOR_FIELD_PATHS[BARRIER_ARRIVE_FIELD][0]} 取不到')
        if release is None:
            unavailable.append(f'rank{doc["rank"]}: '
                               f'{ANCHOR_FIELD_PATHS[BARRIER_RELEASE_FIELD][0]} 取不到')
        if arrive is None or release is None:
            continue
        arrivals.append({'rank': doc['rank'], 'ns_raw': arrive,
                         'source': anchors['sources'][BARRIER_ARRIVE_FIELD]})
        releases.append({'rank': doc['rank'], 'ns_raw': release,
                         'source': anchors['sources'][BARRIER_RELEASE_FIELD]})
    if not arrivals or not releases:
        return {'available': False,
                'reason': '；'.join(unavailable) or '没有可用的 kEntryRankSync 锚点',
                'unavailable_notes': unavailable}
    max_arrive = max(arrivals, key=lambda item: item['ns_raw'])
    min_release = min(releases, key=lambda item: item['ns_raw'])
    return {
        'available': True,
        'criterion': f'min(rank.{BARRIER_RELEASE_FIELD}) >= max(rank.{BARRIER_ARRIVE_FIELD})',
        'unavailable_notes': unavailable,
        'max_arrive': max_arrive,
        'min_release': min_release,
        'holds': min_release['ns_raw'] >= max_arrive['ns_raw'],
        'margin_ns': min_release['ns_raw'] - max_arrive['ns_raw'],
    }


def intra_rank_jitter(docs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    widest: Optional[Dict[str, Any]] = None
    for doc in docs:
        anchors = doc['anchors']
        if anchors is None:
            continue
        fields = anchors['fields']
        low = fields.get('min_start_ns')
        high = fields.get('max_end_ns')
        if low is None or high is None:
            low = fields.get(BARRIER_ARRIVE_FIELD)
            high = fields.get(BARRIER_RELEASE_FIELD)
        if low is None or high is None:
            continue
        span = high - low
        if widest is None or span > widest['span_ns']:
            widest = {'rank': doc['rank'], 'span_ns': span,
                      'basis': 'max_end_ns-min_start_ns'}
    return widest


def decide_clock_domain(docs: List[Dict[str, Any]], field: str, request: str,
                        per_rank: List[Dict[str, Any]],
                        raw_barrier: Dict[str, Any]) -> Dict[str, Any]:
    ordered = sorted(per_rank, key=lambda entry: (entry['rank'], entry['basis_ns']))
    pairwise: List[Dict[str, Any]] = []
    for index, high in enumerate(ordered):
        for low in ordered[index + 1:]:
            offset = high['basis_ns'] - low['basis_ns']
            pairwise.append({'rank_a': high['rank'], 'rank_b': low['rank'],
                             'offset_ns': offset, 'abs_offset_ns': abs(offset)})
    max_offset = max(pairwise, key=lambda item: item['abs_offset_ns']) if pairwise else None
    jitter = intra_rank_jitter(docs)
    ratio: Optional[float] = None
    if max_offset is not None and jitter is not None and jitter['span_ns'] > 0:
        ratio = max_offset['abs_offset_ns'] / jitter['span_ns']

    if request == 'shared':
        resolved = 'shared'
        reason = (f'显式指定 --clock-domain shared：所有 rank 共用一个时钟域，'
                  f'用单一全局 shift = min(各 rank 的 {field})')
    elif request == 'per_rank':
        resolved = 'per_rank'
        reason = (f'显式指定 --clock-domain per_rank：每个 rank 以自身的 {field} 为 0 平移')
    elif not raw_barrier.get('available'):
        resolved = 'shared'
        reason = (f'--clock-domain auto 判定所需的{CLOCK_DOMAIN_CRITERION}不可用'
                  f'（{raw_barrier.get("reason")}），保守退回 shared')
    elif raw_barrier['holds']:
        resolved = 'shared'
        reason = (f'{CLOCK_DOMAIN_CRITERION}：成立（margin={raw_barrier["margin_ns"]}ns >= 0），'
                  f'各 rank 处于同一时钟域，判定为 shared')
    else:
        resolved = 'per_rank'
        if ratio is not None and ratio >= CLOCK_SKEW_RATIO_THRESHOLD:
            reason = (f'因为偏移量 {max_offset["abs_offset_ns"]} ns 远超合理调度抖动 '
                      f'{jitter["span_ns"]} ns（{ratio:.1f}x），判定为不同时钟域，切换到 per_rank')
        elif max_offset is not None and jitter is not None:
            reason = (f'因为{CLOCK_DOMAIN_CRITERION}不成立'
                      f'（margin={raw_barrier["margin_ns"]}ns < 0），判定为不同时钟域，'
                      f'切换到 per_rank；最大基准偏移 {max_offset["abs_offset_ns"]}ns 与同 device '
                      f'时钟内抖动参考 {jitter["span_ns"]}ns 同量级'
                      f'（{"n/a" if ratio is None else f"{ratio:.1f}x"}），'
                      f'也可能是真实的跨 rank 滞后而非时钟域差异')
        else:
            reason = (f'因为{CLOCK_DOMAIN_CRITERION}不成立'
                      f'（margin={raw_barrier["margin_ns"]}ns < 0），判定为不同时钟域，'
                      f'切换到 per_rank')

    return {
        'requested': request,
        'resolved': resolved,
        'applicable': True,
        'criterion': CLOCK_DOMAIN_CRITERION,
        'basis_field': field,
        'raw_barrier': raw_barrier,
        'bases': [{'rank': entry['rank'], 'basis_ns': entry['basis_ns'],
                   'source': entry['source']} for entry in ordered],
        'pairwise_offsets': pairwise,
        'max_offset': max_offset,
        'jitter_reference': jitter,
        'offset_over_jitter': ratio,
        'reason': reason,
        'warning': PER_RANK_CLOCK_WARNING if resolved == 'per_rank' else None,
    }


def resolve_alignment(docs: List[Dict[str, Any]], mode: str,
                      clock_domain: str) -> Dict[str, Any]:
    raw_barrier = raw_barrier_verdict(docs)
    if mode == 'none':
        # --align none performs no cross-rank synchronization: every rank keeps its
        # own raw device clock, so the alignment shift is exactly 0.  Making the
        # display timeline nonnegative is a separate downstream concern handled by
        # rebase_trace_timestamps; it must not be folded into the alignment shift,
        # which G19 and the contract read as the cross-rank offset.
        for doc in docs:
            doc['shift_ns'] = 0
        shift_by_rank = {str(doc['rank']): 0 for doc in docs}
        domain = {
            'requested': clock_domain,
            'resolved': 'shared',
            'applicable': False,
            'criterion': CLOCK_DOMAIN_CRITERION,
            'basis_field': None,
            'raw_barrier': raw_barrier,
            'bases': [],
            'pairwise_offsets': [],
            'max_offset': None,
            'jitter_reference': intra_rank_jitter(docs),
            'offset_over_jitter': None,
            'reason': (f'--align none 不做任何平移，--clock-domain {clock_domain} 不适用，'
                       f'所有 rank 的 shift_ns 都是 0'),
            'warning': None,
            'shift_ns_by_rank': shift_by_rank,
        }
        return {'mode': mode, 'field': None, 'shift_ns': 0,
                'global_shift_ns': 0,
                'per_rank': [], 'basis_spread_ns': 0, 'clock_domain': domain,
                'shift_ns_by_rank': shift_by_rank}
    field = ALIGN_FIELD[mode]
    per_rank: List[Dict[str, Any]] = []
    for doc in docs:
        anchors = doc['anchors']
        value = None if anchors is None else anchors['fields'].get(field)
        if value is None:
            raise EventsSchemaError(
                f'{doc["path"]}: --align {mode} 需要 {field}，但下面这些完整路径都取不到: '
                f'{list(ANCHOR_FIELD_PATHS[field])}。可改用 --align none。')
        per_rank.append({'rank': doc['rank'], 'basis_ns': value,
                         'source': anchors['sources'][field]})
    bases = [entry['basis_ns'] for entry in per_rank]
    global_shift = min(bases)
    domain = decide_clock_domain(docs, field, clock_domain, per_rank, raw_barrier)
    for doc, entry in zip(docs, per_rank):
        shift = entry['basis_ns'] if domain['resolved'] == 'per_rank' else global_shift
        doc['shift_ns'] = shift
        entry['shift_ns'] = shift
        entry['aligned_ns'] = entry['basis_ns'] - shift
    shift_by_rank = {str(entry['rank']): entry['shift_ns'] for entry in per_rank}
    domain['shift_ns_by_rank'] = shift_by_rank
    return {'mode': mode, 'field': field,
            'shift_ns': None if domain['resolved'] == 'per_rank' else global_shift,
            'global_shift_ns': global_shift, 'per_rank': per_rank,
            'basis_spread_ns': max(bases) - min(bases), 'clock_domain': domain,
            'shift_ns_by_rank': shift_by_rank}


def smid_label(info: Optional[Dict[str, Any]]) -> str:
    if info is None or info.get('physical_smid') is None:
        return 'SM?'
    if info.get('written') is False:
        return f'SM{info["physical_smid"]}(unwritten)'
    return f'SM{info["physical_smid"]}'


def load_warp_topology(path: Path) -> Dict[str, Any]:
    try:
        topology = load_json(path)
    except EventsSchemaError:
        raise
    if not isinstance(topology, dict) or topology.get('schema') != 'fast-instrument-warp-topology/v1':
        raise EventsSchemaError(f'{path}: 不是 fast-instrument-warp-topology/v1')
    physical = topology.get('physical_warps_per_cta')
    observer_count = topology.get('observer_lane_count')
    lanes = topology.get('lanes')
    if (not isinstance(physical, int) or physical < 1 or
            not isinstance(observer_count, int) or observer_count < 1 or
            not isinstance(lanes, list) or len(lanes) != observer_count):
        raise EventsSchemaError(f'{path}: topology physical/observer/lanes 字段无效')
    expected_sha = topology.get('canonical_payload_sha256')
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise EventsSchemaError(f'{path}: 缺少 canonical_payload_sha256')
    canonical = dict(topology)
    canonical.pop('canonical_payload_sha256', None)
    actual_sha = __import__('hashlib').sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                   separators=(',', ':')).encode('utf-8')).hexdigest()
    if actual_sha != expected_sha:
        raise EventsSchemaError(f'{path}: canonical_payload_sha256 不匹配')
    by_track: Dict[int, Dict[str, Any]] = {}
    previous_end = -1
    for index, lane in enumerate(lanes):
        if not isinstance(lane, dict):
            raise EventsSchemaError(f'{path}: lanes[{index}] 必须是 object')
        source_track = lane.get('source_track_id')
        observer = lane.get('observer_warp')
        represents = lane.get('represents')
        label = lane.get('label')
        if (not isinstance(source_track, int) or isinstance(source_track, bool) or
                not isinstance(observer, int) or isinstance(observer, bool) or
                not isinstance(represents, list) or len(represents) != 2 or
                not all(isinstance(item, int) and not isinstance(item, bool)
                        for item in represents) or not isinstance(label, str)):
            raise EventsSchemaError(f'{path}: lanes[{index}] 字段无效')
        if source_track in by_track:
            raise EventsSchemaError(f'{path}: source_track_id={source_track} 重复')
        start, end = represents
        if start != previous_end + 1 or end < start or end >= physical:
            raise EventsSchemaError(f'{path}: lanes[{index}].represents 不是连续物理覆盖')
        policy = lane.get('representation_policy', 'physical_warp_observer')
        role_text = str(lane.get('role', '')).lower()
        architecture = str(topology.get('architecture', '')).lower()
        warpgroup = lane.get('warpgroup') is True
        policy_valid = (
            policy == 'hopper_mma_warpgroup' and 'hopper' in architecture
            and ((('mma' in role_text and 'warpgroup' in role_text)
                  or (warpgroup and 'math' in role_text)))
        ) or (
            policy == 'blackwell_epilogue_warpgroup' and 'blackwell' in architecture
            and ((('epilogue' in role_text and 'warpgroup' in role_text)
                  or (warpgroup and 'epilogue' in role_text)))
        )
        if end > start and not policy_valid:
            raise EventsSchemaError(
                f'{path}: lanes[{index}] 只能由 Hopper MMA warpgroup 或 '
                'Blackwell epilogue warpgroup 代表多个 physical warp')
        previous_end = end
        if not re.fullmatch(r'warp\d+(（代表 warp\d+-\d+）)?', label):
            raise EventsSchemaError(f'{path}: lanes[{index}].label 不符合拓扑命名契约')
        by_track[source_track] = lane
    if previous_end != physical - 1:
        raise EventsSchemaError(f'{path}: represents 未覆盖到 physical warp {physical - 1}')
    topology['_by_source_track'] = by_track
    topology['_path'] = str(path.resolve())
    return topology


def phase_parent_task_id(event: Dict[str, Any]) -> Optional[int]:
    spec = event.get('task_spec') or {}
    parent = spec.get('parent_task_id')
    return parent if isinstance(parent, int) else None


def local_task_parent_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold each source task occurrence with its adjacent child phases locally."""
    # Device-recorded task boundaries are authoritative.  Never expand or
    # synthesize a task envelope from child phases; containment is validated
    # later against the unmodified source intervals.
    return list(events)

    source_by_id: Dict[int, List[Dict[str, Any]]] = {}
    for event in events:
        if event.get('layer') == 'task' and isinstance(event.get('task_id'), int):
            source_by_id.setdefault(event['task_id'], []).append(event)
    for values in source_by_id.values():
        values.sort(key=lambda item: (item['start_ns'], item['end_ns']))

    children: Dict[int, List[Dict[str, Any]]] = {
        id(event): [] for values in source_by_id.values() for event in values}
    for phase in events:
        parent_id = phase_parent_task_id(phase) if phase.get('layer') == 'phase' else None
        if parent_id is None:
            continue
        candidates = [source for source in source_by_id.get(parent_id, [])
                      if source['start_ns'] <= phase['start_ns']
                      and (source.get('expert_idx') is None
                           or phase.get('expert_idx') is None
                           or source.get('expert_idx') == phase.get('expert_idx'))]
        if candidates:
            children[id(candidates[-1])].append(phase)

    result: List[Dict[str, Any]] = []
    for event in events:
        phases = children.get(id(event), [])
        if event.get('layer') != 'task' or not phases:
            result.append(event)
            continue
        derived = dict(event)
        derived_start = min([event['start_ns']] + [phase['start_ns'] for phase in phases])
        derived_end = max([event['end_ns']] + [phase['end_ns'] for phase in phases])
        derived.update({
            'start_ns': derived_start,
            'end_ns': derived_end,
            'dur_ns': derived_end - derived_start,
            'is_derived': True,
            'local_parent': True,
            'source_start_ns_raw': event['start_ns'],
            'source_end_ns_raw': event['end_ns'],
            'source_event_count': 1 + len(phases),
            'source_task_event_count': 1,
            'source_phase_ids': sorted({phase.get('task_id') for phase in phases
                                        if phase.get('task_id') is not None}),
        })
        result.append(derived)
    return result


def build_trace(docs: List[Dict[str, Any]], align: str, alignment: Dict[str, Any],
                summary_max_ctas: int, warp_topology: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    clock_domain = alignment['clock_domain']
    trace_events: List[Dict[str, Any]] = []
    role_totals: Dict[str, Dict[str, Any]] = {}
    lanes_per_role: Dict[str, int] = {}
    cta_cores: List[Dict[str, Any]] = []
    overlap_count = 0
    global_start: Optional[int] = None
    global_end: Optional[int] = None
    ts_extremes: List[float] = []
    process_index = 0
    measured_leader_threads = 0
    topology_only_threads = 0
    num_x_measured = 0
    num_x_replica = 0
    group_thread_counts: List[int] = []
    audit_lane_maps: List[Dict[Tuple[int, int], Tuple[int, int]]] = []

    ordered = sorted(docs, key=lambda item: (item['rank'], str(item['path'])))
    topology_by_track = warp_topology['_by_source_track']
    for doc in ordered:
        unmapped = sorted({event['track'] for event in doc['events']
                           if event['track'] not in topology_by_track})
        if unmapped:
            raise EventsSchemaError(
                f'{doc["path"]}: source track {unmapped} 没有 topology lane 映射')
    # Task and phase events stay in the CTA observer lane.  Cross-CTA global
    # envelopes are intentionally not constructed or emitted.
    task_process_count = 0
    cta_sort_index_base = task_process_count
    for doc in ordered:
        shift_ns = doc['shift_ns']
        device_label = 'dev?' if doc['device'] is None else f'dev{doc["device"]}'

        lanes: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for event in doc['events']:
            lanes.setdefault((event['block'], event['track']), []).append(event)

        smid_of_block: Dict[int, Dict[str, Any]] = {block: dict(info)
                                                    for block, info in doc['block_map'].items()}
        for event in doc['events']:
            if event['physical_smid'] is not None:
                smid_of_block.setdefault(event['block'],
                                         {'physical_smid': event['physical_smid'],
                                          'written': True})

        blocks = sorted({block for block, _track in lanes})
        audit_lane_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for block in blocks:
            pid = PID_BASE + process_index
            sort_index = cta_sort_index_base + process_index
            process_index += 1
            for lane_index, topology_lane in enumerate(warp_topology['lanes']):
                audit_lane_map[(block, topology_lane['source_track_id'])] = (pid, lane_index + 1)
            info = smid_of_block.get(block)
            smid = None if info is None else info.get('physical_smid')
            trace_events.append({'name': 'process_name', 'ph': 'M', 'pid': pid, 'tid': 0,
                                 'args': {'name': f'rank+CTA {doc["rank"]}:{block} / '
                                                  f'{smid_label(info)}'}})
            trace_events.append({'name': 'process_sort_index', 'ph': 'M', 'pid': pid, 'tid': 0,
                                 'args': {'sort_index': sort_index}})
            trace_events.append({'name': 'process_labels', 'ph': 'M', 'pid': pid, 'tid': 0,
                                 'args': {'labels': f'{doc["path"].name} {device_label} '
                                                    f'rank{doc["rank"]} CTA{block}'}})

            threads: List[Dict[str, Any]] = []
            for topology_lane in warp_topology['lanes']:
                track = topology_lane['source_track_id']
                lane_events = sorted(
                    lanes.get((block, track), []),
                    key=lambda item: (item['start_ns'], -item['dur_ns'],
                                      item['task_id'] if item['task_id'] is not None else -1))
                lane_events = local_task_parent_events(lane_events)
                role = topology_lane.get('role') or f'track{track}'
                threads.append({'track': track, 'role': role,
                                'lane_label': topology_lane['label'],
                                'warp_id': topology_lane['observer_warp'],
                                'leader_warp': topology_lane['observer_warp'],
                                'is_leader': True, 'events': lane_events})
            for track in sorted(t for b, t in lanes if b == block):
                if track in topology_by_track:
                    continue
                lane_events = sorted(
                    lanes[(block, track)],
                    key=lambda item: (item['start_ns'], -item['dur_ns'],
                                      item['task_id'] if item['task_id'] is not None else -1))
                role_counts: Dict[str, int] = {}
                for event in lane_events:
                    role_counts[event['role']] = role_counts.get(event['role'], 0) + 1
                role = max(sorted(role_counts), key=lambda name: role_counts[name]) \
                    if role_counts else f'observer_{track}'
                raise EventsSchemaError(f'source track {track} 未映射 topology lane')

            threads.sort(key=lambda item: (item['warp_id'] if item['warp_id'] is not None
                                           else 1 << 30, item['track'],
                                           0 if item['is_leader'] else 1))
            group_thread_counts.append(len(threads))
            for tid_index, thread in enumerate(threads):
                tid = tid_index + 1
                track = thread['track']
                role = thread['role']
                lane_label = thread['lane_label']
                warp_id = thread['warp_id']
                is_leader = thread['is_leader']
                thread_name = thread['lane_label']
                trace_events.append({'name': 'thread_name', 'ph': 'M', 'pid': pid, 'tid': tid,
                                     'args': {'name': thread_name}})
                trace_events.append({'name': 'thread_sort_index', 'ph': 'M', 'pid': pid,
                                     'tid': tid, 'args': {'sort_index': tid_index}})
                if is_leader:
                    measured_leader_threads += 1
                    lanes_per_role[role] = lanes_per_role.get(role, 0) + 1
                if not is_leader:
                    topology_only_threads += 1
                    continue

                open_stack: List[int] = []
                for event in thread['events']:
                    if event['layer'] == 'ignored':
                        continue
                    start = event['start_ns'] - shift_ns
                    end = event['end_ns'] - shift_ns
                    if is_leader:
                        while open_stack and open_stack[-1] <= start:
                            open_stack.pop()
                        if open_stack and end > open_stack[-1]:
                            overlap_count += 1
                        open_stack.append(end)
                        global_start = start if global_start is None else min(global_start, start)
                        global_end = end if global_end is None else max(global_end, end)
                        event_role = event['role']
                        bucket = role_totals.setdefault(event_role, {'total_ns': 0, 'count': 0})
                        bucket['total_ns'] += event['dur_ns']
                        bucket['count'] += 1
                        ts_extremes.append(start / 1000.0)
                        num_x_measured += 1
                    else:
                        num_x_replica += 1
                    args = {
                        'task_id': event['task_id'],
                        'task': event['task'],
                        'expert_idx': event['expert_idx'],
                        'depth': event['depth'],
                        'block': block,
                        'rank_block': f'{doc["rank"]}:{block}',
                        'physical_smid': smid,
                        'smid_written': None if info is None else info.get('written'),
                        'role': event['role'],
                        'layer': event['layer'],
                        'semantic_kind': event['semantic_kind'],
                        'is_derived': event['is_derived'],
                        'rank': doc['rank'],
                        'warp_id': warp_id,
                        'dur_ns': event['dur_ns'],
                        'start_ns_raw': event['start_ns'],
                        'end_ns_raw': event['end_ns'],
                    }
                    for key in (
                            'task_instance_id', 'claim_kind', 'claim_ticket',
                            'claim_success', 'claim_instance_count',
                            'group', 'm_block', 'n_block',
                            'first_grouped_row', 'rows', 'token'):
                        if key in event:
                            args[key] = event[key]
                    if event.get('local_parent'):
                        args.update({
                            'source_start_ns_raw': event['source_start_ns_raw'],
                            'source_end_ns_raw': event['source_end_ns_raw'],
                            'aggregation': 'per_observer_lane_minmax',
                            'source_event_count': event['source_event_count'],
                            'source_task_event_count': event['source_task_event_count'],
                            'source_phase_ids': event['source_phase_ids'],
                        })
                    if event['task_role'] is not None:
                        args['task_role'] = event['task_role']
                    parent_task_id = phase_parent_task_id(event) \
                        if event['layer'] == 'phase' else None
                    if parent_task_id is not None:
                        # Parent closure is local to this (pid, tid): the ABI
                        # parent id must have a containing source task interval.
                        parent_candidates = [candidate for candidate in thread['events']
                                             if candidate['layer'] == 'task'
                                             and candidate['task_id'] == parent_task_id
                                             and candidate['start_ns'] <= event['start_ns']
                                             and candidate['end_ns'] >= event['end_ns']]
                        if not parent_candidates:
                            raise EventsSchemaError(
                                f'phase task_id={event.get("task_id")} 的 ABI parent_task_id='
                                f'{parent_task_id} 在同一 observer lane 没有时间包含的 task parent')
                        parent = min(parent_candidates,
                                     key=lambda candidate: candidate['end_ns'] - candidate['start_ns'])
                        args['parent_task_key'] = parent['short_name'] or parent['task']
                        args['parent_task_id'] = parent_task_id
                    trace_events.append({
                        'name': event['name'],
                        'cat': f"{event['layer']}/{event['semantic_kind']}",
                        'ph': 'X',
                        'pid': pid,
                        'tid': tid,
                        'ts': start / 1000.0,
                        'dur': event['dur_ns'] / 1000.0,
                        'args': args,
                    })

        audit_lane_maps.append(audit_lane_map)

        anchors = doc['anchors'] or {}
        per_cta = anchors.get('per_cta') or {}
        rank_release = (anchors.get('fields') or {}).get('entry_sync_end_ns')
        last_end_of_block: Dict[int, int] = {}
        for event in doc['events']:
            previous = last_end_of_block.get(event['block'])
            if previous is None or event['end_ns'] > previous:
                last_end_of_block[event['block']] = event['end_ns']
        for block in sorted(last_end_of_block):
            core_begin = per_cta.get(block, {}).get('release_ns')
            if core_begin is None:
                core_begin = rank_release
            last_end = last_end_of_block[block]
            info = smid_of_block.get(block)
            cta_cores.append({
                'rank': doc['rank'],
                'block': block,
                'physical_smid': None if info is None else info.get('physical_smid'),
                'smid_written': None if info is None else info.get('written'),
                'core_begin_ns': None if core_begin is None else core_begin - shift_ns,
                'core_end_ns': last_end - shift_ns,
                'core_dur_ns': None if core_begin is None else last_end - core_begin,
            })

    # No auxiliary TASK process: task parents are source measurements in CTA
    # observer lanes, alongside their contained phase events.
    task_pid: Optional[int] = None

    makespan_ns = 0 if global_start is None or global_end is None else global_end - global_start
    total_event_ns = sum(bucket['total_ns'] for bucket in role_totals.values())
    roles_summary = []
    for role in sorted(role_totals):
        bucket = role_totals[role]
        lane_count = lanes_per_role.get(role, 0)
        roles_summary.append({
            'role': role,
            'events': bucket['count'],
            'lanes': lane_count,
            'total_ns': bucket['total_ns'],
            'pct_of_event_time': (0.0 if total_event_ns == 0
                                  else bucket['total_ns'] / total_event_ns * 100.0),
            'pct_of_lane_makespan': (0.0 if makespan_ns == 0 or lane_count == 0
                                     else bucket['total_ns'] / (makespan_ns * lane_count) * 100.0),
        })

    ts_quantum_ns = 0.0
    if ts_extremes:
        ts_quantum_ns = max(math.ulp(min(ts_extremes)), math.ulp(max(ts_extremes))) * 1000.0

    threads_per_group = {
        'min': min(group_thread_counts) if group_thread_counts else 0,
        'max': max(group_thread_counts) if group_thread_counts else 0,
        'values': sorted(set(group_thread_counts)),
    }
    task_span_count = sum(1 for item in trace_events
                          if item.get('ph') == 'X'
                          and (item.get('args') or {}).get('layer') == 'task')
    phase_span_count = sum(1 for item in trace_events
                           if item.get('ph') == 'X'
                           and (item.get('args') or {}).get('layer') == 'phase')

    summary = {
        'align': align,
        'alignment': alignment,
        'clock_domain': clock_domain['resolved'],
        'clock_domain_requested': clock_domain['requested'],
        'shift_ns': alignment['shift_ns'],
        'shift_ns_by_rank': alignment['shift_ns_by_rank'],
        'makespan_ns': makespan_ns,
        'window_ns': [global_start, global_end],
        'total_event_ns': total_event_ns,
        'num_trace_events': len(trace_events),
        'num_x_events': sum(1 for item in trace_events if item['ph'] == 'X'),
        'num_metadata_events': sum(1 for item in trace_events if item['ph'] == 'M'),
        'num_x_measured': num_x_measured,
        'num_x_replica': num_x_replica,
        'num_processes': process_index + task_process_count,
        'num_cta_processes': process_index,
        'num_task_processes': task_process_count,
        'task_pid': task_pid,
        'task_spans': task_span_count,
        'phase_spans': phase_span_count,
        'pid_base': PID_BASE,
        'measured_leader_threads': measured_leader_threads,
        'topology_only_threads': topology_only_threads,
        'threads_per_cta_group': threads_per_group,
        'physical_warps_per_cta': warp_topology['physical_warps_per_cta'],
        'observer_lane_count': warp_topology['observer_lane_count'],
        'architecture': warp_topology.get('architecture'),
        'warp_topology': warp_topology,
        'warp_topology_sha256': warp_topology['canonical_payload_sha256'],
        'ts_quantum_ns': ts_quantum_ns,
        'roles': roles_summary,
        'cta_cores': cta_cores,
        'cta_cores_truncated_at': summary_max_ctas,
        'improper_overlaps': overlap_count,
        'ranks': [{'rank': doc['rank'], 'path': str(doc['path']), 'device': doc['device'],
                   'events': len(doc['events']), 'rank_source': doc['rank_source'],
                   'shift_ns': doc['shift_ns'],
                   'anomalies': doc['anomalies'],
                   'unwritten_blocks': sorted(block for block, info in doc['block_map'].items()
                                              if info.get('written') is False),
                   'anchors': None if doc['anchors'] is None else {
                       'fields': doc['anchors']['fields'],
                       'sources': doc['anchors']['sources'],
                       'per_cta_source': doc['anchors']['per_cta_source'],
                       'num_per_cta': len(doc['anchors']['per_cta']),
                       'missing_paths': doc['anchors']['missing_paths'],
                   },
                   'anchors_error': doc['anchors_error']}
                  for doc in ordered],
    }
    scheduler_audits = []
    for doc_index, doc in enumerate(ordered):
        raw = doc.get('raw')
        other = raw.get('otherData') if isinstance(raw, dict) else None
        audit = other.get('scheduler_claim_audit') if isinstance(other, dict) else None
        if audit is not None:
            for item in (audit if isinstance(audit, list) else [audit]):
                if not isinstance(item, dict):
                    scheduler_audits.append(item)
                    continue
                mapped = dict(item)
                key = (item.get('pid'), item.get('tid'))
                mapped_pid_tid = audit_lane_maps[doc_index].get(key)
                if mapped_pid_tid is not None:
                    mapped['pid'], mapped['tid'] = mapped_pid_tid
                scheduler_audits.append(mapped)
    trace = {
        'traceEvents': trace_events,
        'displayTimeUnit': 'ns',
        'otherData': {
            'producer': 'profile/scripts/build_perfetto.py',
            'align': align,
            'align_field': alignment['field'],
            'clock_domain': clock_domain['resolved'],
            'clock_domain_requested': clock_domain['requested'],
            'clock_domain_reason': clock_domain['reason'],
            'clock_domain_warning': clock_domain['warning'],
            'clock_domain_detail': clock_domain,
            'shift_ns': alignment['shift_ns'],
            'shift_ns_by_rank': alignment['shift_ns_by_rank'],
            'pid_base': PID_BASE,
            'num_processes': process_index + task_process_count,
            'num_cta_processes': process_index,
            'num_task_processes': task_process_count,
            'task_pid': task_pid,
            'task_process_sort_index': None,
            'task_spans': task_span_count,
            'phase_spans': phase_span_count,
            'warp_topology': {key: value for key, value in warp_topology.items()
                              if not key.startswith('_')},
            'physical_warps_per_cta': warp_topology['physical_warps_per_cta'],
            'scheduler_claim_audit': scheduler_audits,
            'observer_lane_count': warp_topology['observer_lane_count'],
            'warp_topology_sha256': warp_topology['canonical_payload_sha256'],
            'sources': [str(doc['path']) for doc in ordered],
        },
    }
    return trace, summary


def rebase_trace_timestamps(trace: Dict[str, Any],
                            summary: Dict[str, Any]) -> Dict[str, Any]:
    """Apply one global translation so every finite numeric Perfetto ts is nonnegative."""
    timed = [event for event in trace.get('traceEvents', [])
             if isinstance(event, dict) and isinstance(event.get('ts'), (int, float))]
    if not timed:
        raise EventsSchemaError('trace has no numeric timestamps')
    for event in timed:
        if not math.isfinite(float(event['ts'])):
            raise EventsSchemaError(
                f'non-finite timestamp before write: {event.get("name")!r} ts={event["ts"]!r}')
    minimum_us = min(float(event['ts']) for event in timed)
    shift_us = -minimum_us if minimum_us < 0.0 else 0.0
    if shift_us:
        for event in timed:
            event['ts'] = float(event['ts']) + shift_us
        shift_ns = shift_us * 1000.0
        window = summary.get('window_ns')
        if isinstance(window, list) and len(window) == 2:
            summary['window_ns'] = [None if value is None else value + shift_ns
                                    for value in window]
        for cta in summary.get('cta_cores', []):
            for key in ('core_begin_ns', 'core_end_ns'):
                if isinstance(cta.get(key), (int, float)):
                    cta[key] += shift_ns
    invalid = [event for event in timed
               if not math.isfinite(float(event['ts'])) or float(event['ts']) < 0.0]
    if invalid:
        sample = [{'name': event.get('name'), 'ts': event.get('ts')}
                  for event in invalid[:4]]
        raise EventsSchemaError(f'invalid timestamps after global rebase: {sample}')
    report = {
        'policy': 'single_global_translation',
        'minimum_ts_before_us': minimum_us,
        'shift_us': shift_us,
        'shift_ns': shift_us * 1000.0,
        'numeric_timestamp_count': len(timed),
    }
    summary['timeline_rebase'] = report
    trace.setdefault('otherData', {})['timeline_rebase'] = report
    return report


def default_layout_contract(docs: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
    task_ids = sorted({event['task_id'] for doc in docs for event in doc['events']
                       if event.get('layer') == 'task' and event.get('task_id') is not None})
    task_roles = sorted({event.get('task_role') or event.get('role')
                         for doc in docs for event in doc['events']
                         if event.get('layer') == 'task'})
    phase_kinds = sorted({event.get('semantic_kind') for doc in docs for event in doc['events']
                          if event.get('layer') == 'phase' and event.get('semantic_kind')})
    return {
        'schema': LAYOUT_CHECKER.SCHEMA,
        'group_by': 'rank_block',
        'group_name_format': 'rank+CTA {value}',
        'group_sort_index_base': 0,
        'lane_by': ['rank_block', 'warp_id'],
        'lane_name_format': 'rank+CTA {} warp{}',
        'cta_field': 'rank_block',
        'warp_field': 'warp_id',
        'physical_warps_per_cta': summary['physical_warps_per_cta'],
        'observer_lanes_per_cta': summary['observer_lane_count'],
        'observer_representation': [
            {'represents': lane['represents'],
             'policy': lane.get('representation_policy', 'physical_warp_observer'),
             'role': lane.get('role'),
             'architecture': summary.get('architecture'),
             'warpgroup': lane.get('warpgroup') is True}
            for lane in summary.get('warp_topology', {}).get('lanes', [])
            if lane['represents'][1] > lane['represents'][0]
        ],
        'outer_layer': 'task',
        'inner_layer': 'phase',
        'layer_from_cat_prefix': True,
        # The builder creates the SW trace. HW merge preserves this metadata;
        # counter ordering is separately enforced by the HW merge selftest.
        'counter_must_be_topmost': False,
        'source_events': [str(doc['path']) for doc in docs],
        'source_abi': None,
        'allowed_task_ids': task_ids,
        'allowed_task_role': task_roles[0] if len(task_roles) == 1 else None,
        'task_source_measured_only': True,
        'task_required_phase_kind': 'mma' if 'mma' in phase_kinds else None,
        'allow_unparented_phase': True,
        'slice_overlap_policy': 'disjoint_or_nested',
        'observer_spans_source_only': True,
        'allow_empty_warp_lanes': True,
        'local_task_parent': {
            'enabled': True,
            'scope': 'rank_cta_observer_warp',
            'parent_id_field': 'task_id',
            'phase_parent_id_field': 'parent_task_id',
            'containment': 'same_pid_tid_time_interval',
        },
    }


def resolve_layout_contract(path: Optional[str], docs: List[Dict[str, Any]],
                            summary: Dict[str, Any]) -> Dict[str, Any]:
    if path:
        raw = load_json(Path(path).resolve())
        if not isinstance(raw, dict):
            raise EventsSchemaError(f'{path}: layout contract must be a JSON object')
        contract = raw
    else:
        contract = default_layout_contract(docs, summary)
    contract.setdefault('local_task_parent', {
        'enabled': True,
        'scope': 'rank_cta_observer_warp',
        'parent_id_field': 'task_id',
        'phase_parent_id_field': 'parent_task_id',
        'containment': 'same_pid_tid_time_interval',
    })
    try:
        LAYOUT_CHECKER.validate_contract(contract)
    except SystemExit as error:
        raise EventsSchemaError(f'layout contract invalid: {error}') from error
    return contract


def barrier_check(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw = raw_barrier_verdict(docs)
    if not raw.get('available'):
        return {'available': False, 'reason': raw.get('reason')}
    unavailable = raw['unavailable_notes']
    shift_of_rank = {doc['rank']: doc['shift_ns'] for doc in docs}
    max_arrive = dict(raw['max_arrive'])
    min_release = dict(raw['min_release'])
    max_arrive['ns'] = max_arrive['ns_raw'] - shift_of_rank.get(max_arrive['rank'], 0)
    min_release['ns'] = min_release['ns_raw'] - shift_of_rank.get(min_release['rank'], 0)

    intra_rank: List[Dict[str, Any]] = []
    normalized: List[Dict[str, int]] = []
    for doc in docs:
        anchors = doc['anchors']
        if anchors is None:
            continue
        arrive = anchors['fields'].get(BARRIER_ARRIVE_FIELD)
        release = anchors['fields'].get(BARRIER_RELEASE_FIELD)
        basis = anchors['fields'].get('entry_sync_end_ns')
        if arrive is None or release is None:
            continue
        intra_rank.append({'rank': doc['rank'], 'max_start_ns': arrive, 'min_end_ns': release,
                           'holds': release >= arrive, 'margin_ns': release - arrive})
        if basis is not None:
            normalized.append({'rank': doc['rank'], 'arrive_ns': arrive - basis,
                               'release_ns': release - basis})
    clock_normalized: Dict[str, Any] = {'available': False,
                                        'reason': '缺少 entry_sync_end_ns 归一化基准'}
    if normalized:
        worst = max(normalized, key=lambda item: item['arrive_ns'])
        best = min(normalized, key=lambda item: item['release_ns'])
        clock_normalized = {
            'available': True,
            'basis_field': 'entry_sync_end_ns',
            'max_arrive': {'rank': worst['rank'], 'ns': worst['arrive_ns']},
            'min_release': {'rank': best['rank'], 'ns': best['release_ns']},
            'holds': best['release_ns'] >= worst['arrive_ns'],
            'margin_ns': best['release_ns'] - worst['arrive_ns'],
        }

    cta_arrivals: List[Dict[str, Any]] = []
    cta_releases: List[Dict[str, Any]] = []
    for doc in docs:
        if doc['anchors'] is None:
            continue
        doc_shift = doc['shift_ns']
        for block, entry in doc['anchors']['per_cta'].items():
            cta_arrivals.append({'rank': doc['rank'], 'block': block,
                                 'ns_raw': entry['arrive_ns'],
                                 'ns': entry['arrive_ns'] - doc_shift})
            cta_releases.append({'rank': doc['rank'], 'block': block,
                                 'ns_raw': entry['release_ns'],
                                 'ns': entry['release_ns'] - doc_shift})
    per_cta_result: Dict[str, Any] = {'available': False,
                                      'reason': 'per-CTA 锚点为空'}
    if cta_arrivals and cta_releases:
        worst_arrive = max(cta_arrivals, key=lambda item: item['ns_raw'])
        best_release = min(cta_releases, key=lambda item: item['ns_raw'])
        per_cta_result = {
            'available': True,
            'max_arrive': worst_arrive,
            'min_release': best_release,
            'holds': best_release['ns_raw'] >= worst_arrive['ns_raw'],
            'margin_ns': best_release['ns_raw'] - worst_arrive['ns_raw'],
        }
    return {
        'available': True,
        'criterion': f'min(rank.{BARRIER_RELEASE_FIELD}) >= max(rank.{BARRIER_ARRIVE_FIELD})',
        'unavailable_notes': unavailable,
        'max_arrive': max_arrive,
        'min_release': min_release,
        'holds': min_release['ns_raw'] >= max_arrive['ns_raw'],
        'margin_ns': min_release['ns_raw'] - max_arrive['ns_raw'],
        'intra_rank': intra_rank,
        'clock_normalized': clock_normalized,
        'per_cta': per_cta_result,
    }


def print_clock_domain(domain: Dict[str, Any]) -> None:
    print('=== clock domain ===')
    print(f'  requested={domain["requested"]} resolved={domain["resolved"]}')
    print(f'  判定依据: {domain["criterion"]}')
    raw = domain.get('raw_barrier') or {}
    if raw.get('available'):
        verdict = 'PASS' if raw['holds'] else 'VIOLATION'
        print(f'  原始时间戳上的全局判据 {raw["criterion"]}: {verdict} '
              f'margin={raw["margin_ns"]}ns')
        print(f'    最晚 arrive = rank{raw["max_arrive"]["rank"]} '
              f'{raw["max_arrive"]["ns_raw"]}ns')
        print(f'    最早 release = rank{raw["min_release"]["rank"]} '
              f'{raw["min_release"]["ns_raw"]}ns')
    else:
        print(f'  原始时间戳上的全局判据不可用: {raw.get("reason")}')
    for entry in domain.get('bases') or []:
        print(f'    rank{entry["rank"]} 对齐基准原始值 basis_ns={entry["basis_ns"]} '
              f'<- {entry["source"]}')
    pairwise = domain.get('pairwise_offsets') or []
    for pair in pairwise:
        print(f'    基准偏移 rank{pair["rank_a"]} - rank{pair["rank_b"]} = '
              f'{pair["offset_ns"]}ns (abs={pair["abs_offset_ns"]}ns)')
    if not pairwise:
        print('    note 只有一个 rank，没有两两基准偏移量可比')
    max_offset = domain.get('max_offset')
    if max_offset is not None:
        print(f'    最大基准偏移 = {max_offset["abs_offset_ns"]}ns '
              f'(rank{max_offset["rank_a"]} vs rank{max_offset["rank_b"]})')
    jitter = domain.get('jitter_reference')
    if jitter is not None:
        print(f'    合理调度抖动量级参考（同一 device 时钟内 rendezvous 跨度最大值 '
              f'{jitter["basis"]}）= {jitter["span_ns"]}ns (rank{jitter["rank"]})')
    if domain.get('offset_over_jitter') is not None:
        print(f'    最大基准偏移 / 抖动参考 = {domain["offset_over_jitter"]:.1f}x')
    print(f'  结论: {domain["reason"]}')
    for rank, shift in sorted((domain.get('shift_ns_by_rank') or {}).items()):
        print(f'    rank{rank} shift_ns={shift}')
    if domain.get('warning'):
        print(f'  WARNING {domain["warning"]}')
        print(f'WARNING {domain["warning"]}', file=sys.stderr)


def print_barrier(barrier: Dict[str, Any]) -> None:
    print('=== barrier (kEntryRankSync) ===')
    if not barrier.get('available'):
        message = f'  WARNING 跨 rank barrier 校验不可用: {barrier.get("reason")}'
        print(message)
        print(message.strip(), file=sys.stderr)
        return
    arrive = barrier['max_arrive']
    release = barrier['min_release']
    verdict = 'PASS' if barrier['holds'] else 'VIOLATION'
    print(f'  判据 {barrier["criterion"]}: {verdict}')
    print(f'  最晚 arrive  = rank{arrive["rank"]} {arrive["ns_raw"]}ns '
          f'(对齐后 {arrive["ns"]}ns) <- {arrive["source"]}')
    print(f'  最早 release = rank{release["rank"]} {release["ns_raw"]}ns '
          f'(对齐后 {release["ns"]}ns) <- {release["source"]}')
    print(f'  保守 margin = {barrier["margin_ns"]}ns')
    if not barrier['holds']:
        message = (f'WARNING kEntryRankSync barrier 判据不成立: 最早 release '
                   f'rank{release["rank"]}@{release["ns_raw"]}ns < 最晚 arrive '
                   f'rank{arrive["rank"]}@{arrive["ns_raw"]}ns，缺口 '
                   f'{-barrier["margin_ns"]}ns；跨 rank 时间轴不可直接比较'
                   f'（各 device 的 globaltimer 未同步时必然如此）')
        print(f'  {message}')
        print(message, file=sys.stderr)
    for note in barrier.get('unavailable_notes') or []:
        print(f'  note {note}')
    for entry in barrier.get('intra_rank') or []:
        verdict = 'PASS' if entry['holds'] else 'VIOLATION'
        print(f'  rank{entry["rank"]} 同 device 时钟内 min_end_ns({entry["min_end_ns"]}) >= '
              f'max_start_ns({entry["max_start_ns"]}): {verdict} margin={entry["margin_ns"]}ns')
        if not entry['holds']:
            message = (f'WARNING rank{entry["rank"]} 自身时钟内 barrier 判据不成立: '
                       f'min_end_ns={entry["min_end_ns"]} < max_start_ns={entry["max_start_ns"]}')
            print(f'  {message}')
            print(message, file=sys.stderr)
    normalized = barrier.get('clock_normalized') or {}
    if normalized.get('available'):
        verdict = 'PASS' if normalized['holds'] else 'VIOLATION'
        print(f'  去掉 device 时钟偏移后（各 rank 以自身 {normalized["basis_field"]} 为 0）: {verdict} '
              f'最晚 arrive=rank{normalized["max_arrive"]["rank"]}@'
              f'{normalized["max_arrive"]["ns"]}ns 最早 release='
              f'rank{normalized["min_release"]["rank"]}@{normalized["min_release"]["ns"]}ns '
              f'margin={normalized["margin_ns"]}ns')
        if not normalized['holds']:
            message = (f'WARNING 归一化后 barrier 判据仍不成立: 最早 release '
                       f'rank{normalized["min_release"]["rank"]}@{normalized["min_release"]["ns"]}ns '
                       f'< 最晚 arrive rank{normalized["max_arrive"]["rank"]}@'
                       f'{normalized["max_arrive"]["ns"]}ns')
            print(f'  {message}')
            print(message, file=sys.stderr)
    else:
        print(f'  note 归一化判据不可用: {normalized.get("reason")}')
    per_cta = barrier.get('per_cta') or {}
    if per_cta.get('available'):
        verdict = 'PASS' if per_cta['holds'] else 'VIOLATION'
        print(f'  per-CTA 交叉校验: {verdict} '
              f'最晚 arrive=rank{per_cta["max_arrive"]["rank"]}/CTA'
              f'{per_cta["max_arrive"]["block"]}@{per_cta["max_arrive"]["ns_raw"]}ns '
              f'最早 release=rank{per_cta["min_release"]["rank"]}/CTA'
              f'{per_cta["min_release"]["block"]}@{per_cta["min_release"]["ns_raw"]}ns '
              f'margin={per_cta["margin_ns"]}ns')
        if not per_cta['holds']:
            print('  WARNING per-CTA barrier 判据不成立')
    else:
        print(f'  note per-CTA 交叉校验不可用: {per_cta.get("reason")}')


def print_summary(summary: Dict[str, Any], barrier: Dict[str, Any], out_path: Path,
                  out_bytes: int, summary_max_ctas: int) -> None:
    print('=== ranks ===')
    for entry in summary['ranks']:
        print(f'  rank{entry["rank"]} dev={entry["device"]} events={entry["events"]} '
              f'rank_source={entry["rank_source"]} shift_ns={entry["shift_ns"]}')
        anchors = entry['anchors']
        if anchors is None:
            print(f'    WARNING 锚点不可用: {entry["anchors_error"]}')
        else:
            fields = anchors['fields']
            print(f'    anchors arrive={fields["entry_sync_start_ns"]} '
                  f'release={fields["entry_sync_end_ns"]} '
                  f'start=[{fields["min_start_ns"]}, {fields["max_start_ns"]}] '
                  f'skew={fields["start_skew_ns"]}ns '
                  f'end=[{fields["min_end_ns"]}, {fields["max_end_ns"]}] '
                  f'skew={fields["end_skew_ns"]}ns')
            print(f'    anchors 来源 release<-{anchors["sources"]["entry_sync_end_ns"]} '
                  f'arrive<-{anchors["sources"]["entry_sync_start_ns"]} '
                  f'per_cta={anchors["num_per_cta"]}<-{anchors["per_cta_source"]}')
            if anchors['missing_paths']:
                print(f'    note 这些完整路径缺失（已由 per_cta/skew 推导补齐）: '
                      f'{anchors["missing_paths"]}')
        if entry['unwritten_blocks']:
            print(f'    WARNING block_map 里 written=false 的 CTA: {entry["unwritten_blocks"]}'
                  f'（thread_name 标为 SMxx(unwritten)）')
        if entry['anomalies'].get('negative_duration'):
            print(f'    WARNING {entry["anomalies"]["negative_duration"]} 条事件 end < start，'
                  f'已夹到 0 长度')
        if entry['anomalies'].get('derived_end_from_dur'):
            print(f'    note {entry["anomalies"]["derived_end_from_dur"]} 条事件的 end_ns '
                  f'由 dur_ns 推出')
        print(f'    source={entry["path"]}')

    print('=== alignment ===')
    alignment = summary['alignment']
    domain = alignment['clock_domain']
    shift_text = 'per-rank' if alignment['shift_ns'] is None else str(alignment['shift_ns'])
    print(f'  mode={summary["align"]} field={alignment["field"]} '
          f'clock_domain={domain["resolved"]} shift_ns={shift_text} '
          f'window_ns={summary["window_ns"]} makespan_ns={summary["makespan_ns"]}')
    for entry in alignment['per_rank']:
        print(f'    rank{entry["rank"]} basis={entry["basis_ns"]}ns '
              f'shift={entry["shift_ns"]}ns -> {entry["aligned_ns"]}ns <- {entry["source"]}')
    if alignment['per_rank']:
        print(f'    跨 rank 基准跨度 = {alignment["basis_spread_ns"]}ns')
    if summary['ts_quantum_ns'] > 1.0:
        print(f'  WARNING ts 的 float 量化约 {summary["ts_quantum_ns"]:.1f}ns，'
              f'纳秒精度只保留在 args.start_ns_raw / args.dur_ns（--align none 的固有限制）')

    print_clock_domain(domain)
    print_barrier(barrier)

    print('=== roles ===')
    for role in summary['roles']:
        print(f'  {role["role"]:<12} events={role["events"]:<7} lanes={role["lanes"]:<5} '
              f'total={role["total_ns"] / 1000.0:14.3f}us '
              f'{role["pct_of_event_time"]:6.2f}% of event-time '
              f'{role["pct_of_lane_makespan"]:6.2f}% of lane makespan')
    print(f'  total_event_ns={summary["total_event_ns"]} '
          f'makespan={summary["makespan_ns"] / 1000.0:.3f}us')
    if summary['improper_overlaps']:
        print(f'  WARNING {summary["improper_overlaps"]} 对同 lane 事件部分重叠（非嵌套），'
              f'Perfetto 里可能显示异常')

    print('=== lanes (CTA observer process/thread with local task parents) ===')
    groups = summary['threads_per_cta_group']
    print(f'  processes(CTA 泳道组)={summary["num_cta_processes"]} '
          f'local task parent processes={summary["num_task_processes"]} '
          f'total={summary["num_processes"]} pid_base={summary["pid_base"]}'
          f'（TASK pid={summary["task_pid"]}，CTA pid 从 {summary["pid_base"]} 起；'
          f'硬件 counter 继续使用更小 pid）')
    print(f'  每 CTA 泳道组 thread 数 min/max={groups["min"]}/{groups["max"]} '
          f'取值={groups["values"]}')
    print(f'  observer thread={summary["measured_leader_threads"]} '
          f'空 observer thread={summary["topology_only_threads"]} '
          f'physical_warps_per_cta={summary["physical_warps_per_cta"]} '
          f'observer_lane_count={summary["observer_lane_count"]}')
    print(f'  ph=X 实测={summary["num_x_measured"]} 复制={summary["num_x_replica"]} '
          f'合计={summary["num_x_events"]} task_spans={summary["task_spans"]} '
          f'phase_spans={summary["phase_spans"]}')
    print(f'  warp_topology_sha256={summary["warp_topology_sha256"]}')

    print('=== per-CTA core (entry_sync_end -> last event end) ===')
    cores = summary['cta_cores']
    shown = cores if summary_max_ctas <= 0 else cores[:summary_max_ctas]
    for core in shown:
        begin = 'n/a' if core['core_begin_ns'] is None else f'{core["core_begin_ns"]}'
        dur = 'n/a' if core['core_dur_ns'] is None else f'{core["core_dur_ns"] / 1000.0:.3f}us'
        label = smid_label({'physical_smid': core['physical_smid'],
                            'written': core['smid_written']})
        print(f'  rank{core["rank"]} CTA{core["block"]}/{label} '
              f'core=[{begin}, {core["core_end_ns"]}]ns dur={dur}')
    if summary_max_ctas > 0 and len(cores) > summary_max_ctas:
        print(f'  ... 共 {len(cores)} 个 CTA，已截断到 {summary_max_ctas}'
              f'（--summary-max-ctas 0 打印全部）')
    print(f'=== 输出 {out_path} ({out_bytes} bytes, {summary["num_trace_events"]} 条 traceEvents '
          f'= {summary["num_x_events"]} 条 ph=X + {summary["num_metadata_events"]} 条 ph=M) ===')


def write_trace(trace: Dict[str, Any], out_path: Path, use_gzip: bool, pretty: bool,
                gzip_level: int = 3) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: Dict[str, Any] = {'ensure_ascii': False}
    if pretty:
        kwargs['indent'] = 2
    else:
        kwargs['separators'] = (',', ':')
    partial = out_path.with_name(out_path.name + '.partial')
    if use_gzip:
        with gzip.open(partial, 'wt', encoding='utf-8', compresslevel=gzip_level) as handle:
            json.dump(trace, handle, **kwargs)
    else:
        with partial.open('w', encoding='utf-8') as handle:
            json.dump(trace, handle, **kwargs)
    partial.replace(out_path)
    return out_path.stat().st_size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='把解码后的 MegaMoE events.json 转成 Chrome/Perfetto trace JSON',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False)
    parser.add_argument('events', nargs='*', help='一个或多个 events.json（每个 rank 一个）')
    parser.add_argument('--events', dest='events_option', nargs='+', default=[],
                        metavar='EVENTS',
                        help='与 positional events 等价的显式写法，可混用')
    parser.add_argument('--out', type=str, default=None,
                        help='输出 trace 路径；以 .gz 结尾时自动 gzip；'
                             '若指向目录，必须同时给 --contract，并按 SPEC-1 自动拼'
                             ' trace__<model>_<stage>__card-<h20|b300|mi308x>__...perfetto.json.gz')
    parser.add_argument('--contract', default=None,
                        help='锁定契约 JSON；--out 是目录时必填。文件名从 target.card、'
                             'target.model/stage、input 与 build_and_launch/runtime_launch 生成')
    parser.add_argument('--run-tag', type=str, default=None,
                        help='仅写入 otherData.run_tag，不再参与交付文件名拼接')
    parser.add_argument('--gzip', action='store_true', help='强制 gzip 压缩输出')
    parser.add_argument('--align', choices=('entry_release', 'entry_arrive', 'none'),
                        default='entry_release',
                        help='跨 rank 时间对齐基准：kEntryRankSync 的 release / arrive / 不对齐')
    parser.add_argument('--clock-domain', dest='clock_domain', choices=CLOCK_DOMAIN_CHOICES,
                        default='auto',
                        help='shared=所有 rank 共用一个时钟域，用单一全局 shift；'
                             'per_rank=每个 rank 各自以自己的对齐基准为 0 平移；'
                             'auto=按 barrier 判据在原始时间戳上是否成立自动判定'
                             '（成立 shared，不成立 per_rank）')
    parser.add_argument('--events-schema-probe', action='store_true',
                        help='只打印实际看到的字段结构后退出')
    parser.add_argument('--summary-max-ctas', type=int, default=0,
                        help='stdout 里最多打印多少个 CTA 的 core 区间，0 表示全部')
    parser.add_argument('--warp-topology', required=True,
                        help='fast-instrument-warp-topology/v1 JSON')
    parser.add_argument('--layout-contract', default=None,
                        help='显式 layout contract JSON；缺省时按实际 topology 生成并内嵌')
    parser.add_argument('--pretty', action='store_true', help='缩进输出（文件更大）')
    parser.add_argument('--gzip-level', type=int, choices=range(1, 10), default=3,
                        help='gzip 压缩级别；默认 3，优先后处理速度')
    parser.add_argument('--abi', required=True, help='包含 tasks 元数据的 ABI JSON')
    parser.add_argument('--capture-plan', required=True,
                        help='第 0 步强制弹窗后生成的 capture plan JSON')
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    paths = [Path(item).resolve() for item in list(args.events) + list(args.events_option)]
    if not paths:
        parser.error('至少要给一个 events.json（positional 或 --events）')

    if args.events_schema_probe:
        for path in paths:
            try:
                probe_document(path, load_json(path), sys.stdout)
            except EventsSchemaError as error:
                print(f'{error}', file=sys.stderr)
                return 3
        return 0

    if not args.out:
        parser.error('--out 是必填项（除非只做 --events-schema-probe）')

    contract_doc = None
    if args.contract:
        try:
            contract_doc = load_json(Path(args.contract).resolve())
        except EventsSchemaError as error:
            print(f'ERROR {error}', file=sys.stderr)
            return 3
        if not isinstance(contract_doc, dict):
            print('ERROR --contract 必须是 JSON object', file=sys.stderr)
            return 3

    try:
        capture_plan = CAPTURE_SCOPE.load_plan(Path(args.capture_plan).resolve())
        task_specs = load_abi_tasks(Path(args.abi).resolve())
        docs = [load_rank_document(path, index, task_specs)
                for index, path in enumerate(paths)]
        CAPTURE_SCOPE.validate_decoded_objects(
            capture_plan, [(doc['path'], doc['raw']) for doc in docs])
        for doc in docs:
            try:
                doc['anchors'] = resolve_anchors(doc['raw'], doc['events'], doc['path'])
            except EventsSchemaError as error:
                if args.align != 'none':
                    raise
                doc['anchors'] = None
                doc['anchors_error'] = str(error)
        alignment = resolve_alignment(docs, args.align, args.clock_domain)
    except (EventsSchemaError, CAPTURE_SCOPE.CaptureScopeError) as error:
        print(f'ERROR {error}', file=sys.stderr)
        # Schema probing is an explicit mode (--events-schema-probe).  Never
        # reopen large event files after a failed formal build.
        return 3

    ranks = [doc['rank'] for doc in docs]
    if len(set(ranks)) != len(ranks):
        print(f'WARNING 多个输入文件解析出重复的 rank {ranks}，进程泳道会按文件名区分',
              file=sys.stderr)

    try:
        warp_topology = load_warp_topology(Path(args.warp_topology).resolve())
    except EventsSchemaError as error:
        print(f'ERROR {error}', file=sys.stderr)
        return 3

    trace, summary = build_trace(docs, args.align, alignment, args.summary_max_ctas,
                                 warp_topology)
    try:
        rebase_trace_timestamps(trace, summary)
    except EventsSchemaError as error:
        print(f'ERROR {error}', file=sys.stderr)
        return 4
    if summary['improper_overlaps']:
        print(f'ERROR same-lane ph=X partial overlap='
              f'{summary["improper_overlaps"]}; trace not written',
              file=sys.stderr)
        return 4
    try:
        layout_contract = resolve_layout_contract(args.layout_contract, docs, summary)
    except EventsSchemaError as error:
        print(f'ERROR {error}', file=sys.stderr)
        return 3
    trace['traceEvents'].insert(0, LAYOUT_CHECKER.metadata_event(layout_contract))
    summary['num_trace_events'] += 1
    summary['num_metadata_events'] += 1
    trace['otherData']['layout_contract'] = layout_contract
    layout_fails = LAYOUT_CHECKER.verify(trace, layout_contract)
    if layout_fails:
        print('ERROR layout contract verification failed before write:', file=sys.stderr)
        for failure in layout_fails:
            print(f'  - {failure}', file=sys.stderr)
        return 4
    barrier = barrier_check(docs)
    trace['otherData']['barrier_check'] = barrier
    trace['otherData']['summary'] = summary
    trace['otherData']['capture_scope'] = CAPTURE_SCOPE.trace_metadata(capture_plan)
    if args.run_tag:
        trace['otherData']['run_tag'] = args.run_tag
    if contract_doc is not None:
        try:
            trace['otherData']['target'] = validate_contract_target(contract_doc)
        except EventsSchemaError as error:
            print(f'ERROR {error}', file=sys.stderr)
            return 3

    out_path = Path(args.out).resolve()
    if out_path.is_dir():
        if contract_doc is None:
            print('ERROR --out 指向目录时必须提供 --contract；'
                  '禁止回退到不含 card 的旧文件名', file=sys.stderr)
            return 3
        try:
            grid_fb, warp_fb, phase_fb = decoded_topology(docs, summary)
            filename = build_trace_filename(
                contract_doc, bundle='sw', grid=grid_fb,
                warps_per_cta=warp_fb, num_phases=phase_fb)
        except EventsSchemaError as error:
            print(f'ERROR {error}', file=sys.stderr)
            return 3
        out_path = out_path / filename
    else:
        try:
            validate_explicit_trace_filename(out_path, contract_doc)
        except EventsSchemaError as error:
            print(f'ERROR {error}', file=sys.stderr)
            return 3
    use_gzip = args.gzip or out_path.name.endswith('.gz')
    out_bytes = write_trace(trace, out_path, use_gzip, args.pretty, args.gzip_level)
    print_summary(summary, barrier, out_path, out_bytes, args.summary_max_ctas)
    return 0


if __name__ == '__main__':
    sys.exit(main())
