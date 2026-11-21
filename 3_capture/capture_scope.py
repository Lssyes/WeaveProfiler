#!/usr/bin/env python3
"""Persist and validate the physical-SM source-capture contract."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


SCHEMA = 'fast-instrument-capture-plan/v1'
LEGACY_SM01 = (0, 1)
EVENT_KEYS = ('events', 'intervals', 'records', 'spans')
SMID_KEYS = ('physical_smid', 'smid', 'physical_sm', 'sm_id')


class CaptureScopeError(ValueError):
    pass


def _selected_smids(values: Sequence[Any]) -> List[int]:
    selected = [_integer(value, 'selected_physical_smids') for value in values]
    if not selected:
        raise CaptureScopeError('selected_physical_smids must be non-empty')
    if any(value < 0 for value in selected):
        raise CaptureScopeError('selected_physical_smids must be non-negative')
    if len(set(selected)) != len(selected):
        raise CaptureScopeError('selected_physical_smids must be unique')
    return selected


def make_plan(mode: str, selected_physical_smids: Sequence[Any] = ()) -> Dict[str, Any]:
    if mode == 'sm01':
        selected = list(LEGACY_SM01)
    elif mode == 'selected':
        selected = _selected_smids(selected_physical_smids)
    elif mode == 'all':
        selected = []
    else:
        raise CaptureScopeError("capture mode must be 'selected', 'all', or legacy 'sm01'")
    subset = mode != 'all'
    return {
        'schema': SCHEMA,
        'mode': mode,
        'capture_scope': 'physical_sm_subset' if subset else 'all_physical_sms',
        'selected_physical_smids': selected,
        'kernel_execution_scope': 'full_grid',
        'source_capture_was_all_sm': not subset,
        'source_physical_smid_filter_required': subset,
        'compact_buffer_required': subset,
        'postprocess_smid_filter_allowed': False,
    }


def _read_json(path: Path) -> Any:
    opener = gzip.open if path.name.endswith('.gz') else open
    try:
        with opener(path, 'rt', encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError) as error:
        raise CaptureScopeError(f'cannot read JSON {path}: {error}') from error


def validate_plan(plan: Any) -> Dict[str, Any]:
    if not isinstance(plan, dict) or plan.get('schema') != SCHEMA:
        raise CaptureScopeError(f'capture plan schema must be {SCHEMA!r}')
    expected = make_plan(plan.get('mode'), plan.get('selected_physical_smids', ()))
    for key, value in expected.items():
        if plan.get(key) != value:
            raise CaptureScopeError(
                f'capture plan {key}={plan.get(key)!r}, expected {value!r}')
    return dict(plan)


def load_plan(path: Path) -> Dict[str, Any]:
    return validate_plan(_read_json(path))


def write_plan(path: Path, mode: str,
               selected_physical_smids: Sequence[Any] = ()) -> Dict[str, Any]:
    plan = make_plan(mode, selected_physical_smids)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return plan


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise CaptureScopeError(f'{context} must be an integer, got bool')
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise CaptureScopeError(f'{context} must be an integer, got {value!r}') from error
    if isinstance(value, float) and not value.is_integer():
        raise CaptureScopeError(f'{context} must be an integer, got {value!r}')
    return result


def _smid(row: Any) -> Any:
    if isinstance(row, Mapping):
        for key in SMID_KEYS:
            if row.get(key) is not None:
                return row[key]
    return None


def _events(doc: Mapping[str, Any], path: Path) -> Sequence[Mapping[str, Any]]:
    for key in EVENT_KEYS:
        rows = doc.get(key)
        if isinstance(rows, list):
            if any(not isinstance(row, dict) for row in rows):
                raise CaptureScopeError(f'{path}:{key} must contain objects')
            return rows
    raise CaptureScopeError(f'{path}: no event list {EVENT_KEYS}')


def _block_map(doc: Mapping[str, Any]) -> Dict[int, int]:
    raw = doc.get('block_map')
    entries: Iterable[Any]
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        entries = ({'logical_cta': key, 'physical_smid': value}
                   if not isinstance(value, dict)
                   else dict(value, logical_cta=key) for key, value in raw.items())
    else:
        return {}
    result: Dict[int, int] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            continue
        block = entry.get('logical_cta', entry.get('block', entry.get('cta', index)))
        smid = _smid(entry)
        if block is not None and smid is not None:
            result[_integer(block, 'block_map.logical_cta')] = _integer(
                smid, 'block_map.physical_smid')
    return result


def decoded_smids(doc: Mapping[str, Any], path: Path) -> Set[int]:
    by_block = _block_map(doc)
    observed: Set[int] = set()
    unresolved = 0
    for index, event in enumerate(_events(doc, path)):
        value = _smid(event)
        if value is None:
            block = event.get('logical_cta', event.get('block', event.get('cta')))
            if block is not None:
                value = by_block.get(_integer(block, f'{path}:events[{index}].block'))
        if value is None:
            unresolved += 1
        else:
            observed.add(_integer(value, f'{path}:events[{index}].physical_smid'))
    if unresolved:
        raise CaptureScopeError(f'{path}: {unresolved} events lack physical_smid')
    if not observed:
        raise CaptureScopeError(f'{path}: no physical SM was observed')
    return observed


def _compact_metadata(plan: Mapping[str, Any], doc: Mapping[str, Any], path: Path) -> None:
    meta = doc.get('capture_scope')
    if not isinstance(meta, Mapping) and isinstance(doc.get('metadata'), Mapping):
        meta = doc['metadata'].get('capture_scope')
    if not isinstance(meta, Mapping):
        raise CaptureScopeError(f'{path}: missing decoded capture_scope metadata')
    required = {
        'capture_scope': 'physical_sm_subset',
        'selected_physical_smids': list(plan['selected_physical_smids']),
        'source_capture_was_all_sm': False,
        'buffer_allocation_basis': 'selected_physical_sm_count_and_cta_capacity',
        'selected_physical_sm_count': len(plan['selected_physical_smids']),
        'capture_overflow': False,
    }
    for key, expected in required.items():
        if meta.get(key) != expected:
            raise CaptureScopeError(
                f'{path}: capture_scope.{key}={meta.get(key)!r}, expected {expected!r}')
    if (_integer(meta.get('capture_slots'), f'{path}:capture_slots')
            < len(plan['selected_physical_smids'])):
        raise CaptureScopeError(f'{path}: compact buffer cannot cover selected physical SMs')


def validate_decoded_objects(
        plan: Mapping[str, Any], documents: Sequence[Tuple[Path, Mapping[str, Any]]]
) -> Dict[str, Any]:
    checked = validate_plan(plan)
    files: List[Dict[str, Any]] = []
    for path, doc in documents:
        observed = decoded_smids(doc, path)
        if checked['mode'] != 'all':
            _compact_metadata(checked, doc, path)
            expected = set(checked['selected_physical_smids'])
            if observed != expected:
                raise CaptureScopeError(
                    f'{path}: observed SMIDs {sorted(observed)}, expected {sorted(expected)}')
        files.append({'path': str(path), 'observed_physical_smids': sorted(observed)})
    return {'ok': True, 'mode': checked['mode'], 'files': files}


def validate_decoded(plan: Mapping[str, Any], paths: Sequence[Path]) -> Dict[str, Any]:
    documents: List[Tuple[Path, Mapping[str, Any]]] = []
    for path in paths:
        doc = _read_json(path)
        if not isinstance(doc, dict):
            raise CaptureScopeError(f'{path}: top-level JSON must be an object')
        documents.append((path, doc))
    return validate_decoded_objects(plan, documents)


def trace_metadata(plan: Mapping[str, Any]) -> Dict[str, Any]:
    checked = validate_plan(plan)
    return {
        'schema': SCHEMA,
        'mode': checked['mode'],
        'capture_scope': checked['capture_scope'],
        'selected_physical_smids': checked['selected_physical_smids'],
        'kernel_execution_scope': 'full_grid',
        'source_capture_was_all_sm': checked['source_capture_was_all_sm'],
        'postprocess_smid_filter_applied': False,
    }


def validate_trace_object(plan: Mapping[str, Any], trace: Mapping[str, Any],
                          path: Path) -> Dict[str, Any]:
    checked = validate_plan(plan)
    if (trace.get('otherData') or {}).get('capture_scope') != trace_metadata(checked):
        raise CaptureScopeError(f'{path}: trace capture_scope does not match plan')
    observed = {_integer(value, f'{path}:physical_smid')
                for event in trace.get('traceEvents', []) if isinstance(event, Mapping)
                for value in [_smid(event.get('args'))] if value is not None}
    expected = set(checked['selected_physical_smids'])
    if checked['mode'] != 'all' and observed != expected:
        raise CaptureScopeError(
            f'{path}: trace observed SMIDs {sorted(observed)}, expected {sorted(expected)}')
    return {'ok': True, 'mode': checked['mode'],
            'observed_physical_smids': sorted(observed), 'path': str(path)}


def validate_trace(plan: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    trace = _read_json(path)
    if not isinstance(trace, dict):
        raise CaptureScopeError(f'{path}: trace must be an object')
    return validate_trace_object(plan, trace, path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(allow_abbrev=False)
    commands = result.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create')
    create.add_argument('--scope', choices=('selected', 'all', 'sm01'), required=True)
    create.add_argument('--physical-smids', default=None,
                        help='comma-separated physical SM IDs; required with --scope selected')
    create.add_argument('--out', type=Path, required=True)
    decoded = commands.add_parser('validate-decoded')
    decoded.add_argument('--plan', type=Path, required=True)
    decoded.add_argument('events', type=Path, nargs='+')
    trace = commands.add_parser('validate-trace')
    trace.add_argument('--plan', type=Path, required=True)
    trace.add_argument('--trace', type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == 'create':
            selected = []
            if args.physical_smids:
                selected = [token.strip() for token in args.physical_smids.split(',')
                            if token.strip()]
            if args.scope == 'selected' and not args.physical_smids:
                raise CaptureScopeError('--scope selected requires --physical-smids')
            if args.scope != 'selected' and args.physical_smids:
                raise CaptureScopeError('--physical-smids is only valid with --scope selected')
            result = {'ok': True, 'plan': write_plan(
                args.out.resolve(), args.scope, selected)}
        elif args.command == 'validate-decoded':
            result = validate_decoded(load_plan(args.plan.resolve()),
                                      [path.resolve() for path in args.events])
        else:
            result = validate_trace(load_plan(args.plan.resolve()), args.trace.resolve())
    except CaptureScopeError as error:
        print(f'ERROR {error}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
