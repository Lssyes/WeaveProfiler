#!/usr/bin/env python3
"""Hard gate for user-selected physical-SM source filtering and compact buffers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


CAPTURE_SCOPE_PATH = Path(__file__).resolve().parent.parent / '3_capture' / 'capture_scope.py'


def _load_capture_scope():
    spec = importlib.util.spec_from_file_location('fast_instrument_capture_scope',
                                                  CAPTURE_SCOPE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {CAPTURE_SCOPE_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAPTURE_SCOPE = _load_capture_scope()


def _read(paths: Sequence[Path]) -> str:
    chunks: List[str] = []
    for path in paths:
        try:
            chunks.append(path.read_text(encoding='utf-8', errors='replace'))
        except OSError as error:
            raise ValueError(f'cannot read source {path}: {error}') from error
    return '\n'.join(chunks)


def validate(plan: Dict[str, Any], device_paths: Sequence[Path],
             host_paths: Sequence[Path]) -> Dict[str, Any]:
    checked = CAPTURE_SCOPE.validate_plan(plan)
    if checked['mode'] == 'all':
        return {'ok': True, 'mode': 'all', 'checks': {'subset_gates_required': False}}
    device = _read(device_paths)
    host = _read(host_paths)
    checks = {
        'reads_physical_smid': bool(re.search(r'%{1,2}smid|physical_smid', device)),
        'reads_selected_physical_smids': bool(
            re.search(r'selected_physical_smids', device, re.I)),
        'separates_capture_slot_from_logical_cta': bool(
            re.search(r'capture_slot', device) and re.search(r'logical_cta', device)),
        'has_explicit_overflow': bool(re.search(r'overflow', device, re.I)),
        'host_allocates_compact_slots': bool(
            re.search(r'capture_slots|capture_capacity', host)
            and re.search(r'selected_physical_sm_count|selected_physical_smids', host, re.I)),
        'host_rejects_overflow': bool(re.search(r'overflow', host, re.I)),
        'host_emits_scope_metadata': bool(
            re.search(r'source_capture_was_all_sm', host)
            and re.search(r'buffer_allocation_basis', host)),
    }
    missing = sorted(name for name, present in checks.items() if not present)
    return {'ok': not missing, 'mode': checked['mode'],
            'selected_physical_smids': checked['selected_physical_smids'],
            'checks': checks, 'missing': missing,
            'device_sources': [str(path) for path in device_paths],
            'host_sources': [str(path) for path in host_paths]}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--capture-plan', type=Path, required=True)
    parser.add_argument('--device-source', type=Path, action='append', default=[])
    parser.add_argument('--host-source', type=Path, action='append', default=[])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    try:
        plan = CAPTURE_SCOPE.load_plan(args.capture_plan.resolve())
        if plan['mode'] != 'all' and (not args.device_source or not args.host_source):
            raise ValueError('a selected physical-SM scope requires explicit '
                             '--device-source and --host-source evidence')
        report = validate(plan, [path.resolve() for path in args.device_source],
                          [path.resolve() for path in args.host_source])
    except (ValueError, CAPTURE_SCOPE.CaptureScopeError) as error:
        print(f'ERROR {error}', file=sys.stderr)
        return 2
    payload = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.out:
        args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.out.resolve().write_text(payload, encoding='utf-8')
    print(payload, end='')
    return 0 if report['ok'] else 3


if __name__ == '__main__':
    sys.exit(main())
