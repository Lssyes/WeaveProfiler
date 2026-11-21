#!/usr/bin/env python3
"""The only supported fast-instrument-v2 delivery entry point.

The finalizer validates the complete ALL_GATES report before invoking
deliver_gate.py.  A receipt is accepted only after the real
trace_processor_shell import has passed and the final trace hash is checked
again after the receipt has been atomically rewritten.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
GATE = HERE / 'deliver_gate.py'
ALL_GATES_IDS = tuple('G%02d' % index for index in range(25))
PREFLIGHT_IDS = tuple('E%02d' % index for index in range(1, 28))
SHA256_RE = re.compile(r'^[0-9a-f]{64}$')

# Defect 3: the skill hardcodes which IDs may be declared not-applicable for
# each information bundle.  A contract only selects a bundle; it can never pick
# N/A IDs freely.  sw+hw is the fully-required bundle, so its N/A sets stay
# empty and its behaviour is byte-identical to the previous rigid policy.
# The gate N/A set is the single authoritative copy shared with
# weaveprof/pipeline/validate/validate_all.py (SW_NOT_APPLICABLE): only a software-only bundle
# may mark exactly {G11,G19,G20,G21} not-applicable; hw and sw+hw keep every
# gate required.  Keeping both files identical guarantees the ALL_GATES report
# produced by validate_all.py always matches the partition finalize expects.
BUNDLE_NA_GATE_IDS = {
    'sw': ('G11', 'G19', 'G20', 'G21'),
    'hw': (),
    'sw+hw': (),
}
BUNDLE_NA_CHECK_IDS = {
    'sw': ('E17', 'E18', 'E19', 'E20'),
    'hw': ('E02', 'E03', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E12',
           'E13', 'E15', 'E16', 'E20'),
    'sw+hw': (),
}


def _bundle_partition(bundle):
    """Return (required_gates, na_gates, required_checks, na_checks) for a bundle.

    Unknown/absent bundles fall back to the legacy fully-required partition
    (no N/A IDs) so callers that never opt in keep the original behaviour."""
    na_gates = set(BUNDLE_NA_GATE_IDS.get(bundle, ()))
    na_checks = set(BUNDLE_NA_CHECK_IDS.get(bundle, ()))
    required_gates = [gate_id for gate_id in ALL_GATES_IDS if gate_id not in na_gates]
    required_checks = [check_id for check_id in PREFLIGHT_IDS
                       if check_id not in na_checks]
    return required_gates, sorted(na_gates), required_checks, sorted(na_checks)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path, label):
    try:
        with path.open('r', encoding='utf-8') as handle:
            return json.load(handle)
    except Exception as error:
        raise ValueError('%s cannot be read as JSON: %s' % (label, error))


def _hash_field(payload, *names):
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _contract_gate_partition(contract):
    """Read both gate partitions, allowing the documented legacy locations."""
    if not isinstance(contract, dict):
        return None, None
    required = None
    not_applicable = None
    for container in (contract.get('execution'), contract.get('validation'), contract):
        if not isinstance(container, dict):
            continue
        if required is None and 'required_gate_ids' in container:
            required = container['required_gate_ids']
        if not_applicable is None and 'not_applicable_gate_ids' in container:
            not_applicable = container['not_applicable_gate_ids']
    if (not isinstance(required, list) or
            not all(isinstance(item, str) for item in required)):
        required = None
    if (not isinstance(not_applicable, list) or
            not all(isinstance(item, str) for item in not_applicable)):
        not_applicable = None
    return required, not_applicable


def validate_all_gates_payload(payload, expected_contract_sha256,
                               expected_trace_sha256, required_gate_ids=None,
                               not_applicable_gate_ids=None):
    """Validate the canonical in-memory ALL_GATES schema."""
    errors = []
    if not isinstance(payload, dict):
        return ['ALL_GATES root must be an object']
    if payload.get('schema_version') != 'all_gates.v1':
        errors.append('schema_version must be all_gates.v1')
    if payload.get('status') != 'PASS':
        errors.append('top-level status is not PASS')
    if required_gate_ids is None:
        required_gate_ids = list(ALL_GATES_IDS)
    if not_applicable_gate_ids is None:
        not_applicable_gate_ids = [gate_id for gate_id in ALL_GATES_IDS
                                   if gate_id not in required_gate_ids]
    # The caller supplies the expected partition (already checked against the
    # information bundle in main()).  These checks confirm it is a well-formed,
    # complete G00-G24 split; they no longer force required==all/NA==empty.
    if (not isinstance(required_gate_ids, list) or
            any(item not in ALL_GATES_IDS for item in required_gate_ids) or
            len(set(required_gate_ids)) != len(required_gate_ids)):
        errors.append('required gate IDs must be valid unique G00-G24 IDs')
        required_gate_ids = []
    if (not isinstance(not_applicable_gate_ids, list) or
            any(item not in ALL_GATES_IDS for item in not_applicable_gate_ids) or
            len(set(not_applicable_gate_ids)) != len(not_applicable_gate_ids)):
        errors.append('not-applicable gate IDs must be valid unique G00-G24 IDs')
        not_applicable_gate_ids = []
    if (set(required_gate_ids).intersection(not_applicable_gate_ids) or
            set(required_gate_ids).union(not_applicable_gate_ids) != set(ALL_GATES_IDS)):
        errors.append('required/not-applicable gate IDs must partition G00-G24')
    declared_required = payload.get('required_gate_ids')
    not_applicable = payload.get('not_applicable_gate_ids')
    if not isinstance(declared_required, list):
        errors.append('required_gate_ids must be a list')
        declared_required = []
    elif not all(isinstance(item, str) for item in declared_required):
        errors.append('required_gate_ids must contain strings')
        declared_required = [item for item in declared_required
                             if isinstance(item, str)]
    if not isinstance(not_applicable, list):
        errors.append('not_applicable_gate_ids must be a list')
        not_applicable = []
    elif not all(isinstance(item, str) for item in not_applicable):
        errors.append('not_applicable_gate_ids must contain strings')
        not_applicable = [item for item in not_applicable
                          if isinstance(item, str)]
    if (len(declared_required) != len(required_gate_ids) or
            set(declared_required) != set(required_gate_ids)):
        errors.append('ALL_GATES required_gate_ids do not match the contract')
    if (len(not_applicable) != len(not_applicable_gate_ids) or
            set(not_applicable) != set(not_applicable_gate_ids)):
        errors.append('ALL_GATES not_applicable_gate_ids do not match the contract')
    partition = declared_required + not_applicable
    if any(item not in ALL_GATES_IDS for item in partition):
        errors.append('required/not_applicable contains unknown gate IDs')
    if len(set(partition)) != len(partition):
        errors.append('required/not_applicable gate IDs are not unique')
    if set(partition) != set(ALL_GATES_IDS):
        errors.append('required/not_applicable gate ID partition is incomplete')
    if not isinstance(payload.get('input_errors'), list):
        errors.append('input_errors must be a list')
    elif payload['input_errors']:
        errors.append('input_errors must be empty')
    if not isinstance(payload.get('partition_errors'), list):
        errors.append('partition_errors must be a list')
    elif payload['partition_errors']:
        errors.append('partition_errors must be empty')
    summary = payload.get('summary')
    if not isinstance(summary, dict):
        errors.append('summary must be an object')
        summary = {}
    if summary.get('all_required_pass') is not True:
        errors.append('summary.all_required_pass must be true')
    if payload.get('integrity_errors') != []:
        errors.append('integrity_errors must be empty')
    if summary.get('counts_consistent') is not True:
        errors.append('summary.counts_consistent must be true')
    if payload.get('contract_sha256_after') != payload.get('contract_sha256'):
        errors.append('contract_sha256_after does not match contract_sha256')
    if payload.get('trace_sha256_after') != payload.get('trace_sha256'):
        errors.append('trace_sha256_after does not match trace_sha256')
    expected_counts = {
        'required_count': len(required_gate_ids),
        'pass': len(required_gate_ids),
        'fail': 0,
        'blocked': 0,
    }
    for key, expected in expected_counts.items():
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            errors.append('summary.%s is inconsistent' % key)
    gates = payload.get('gates')
    if not isinstance(gates, list):
        errors.append('gates must be a list')
        return errors

    actual_contract = _hash_field(payload, 'contract_sha256',
                                  'CONTRACT_LOCK_SHA256')
    actual_trace = _hash_field(payload, 'trace_sha256', 'TRACE_SHA256')
    if not isinstance(actual_contract, str) or not SHA256_RE.match(actual_contract):
        errors.append('contract_sha256 must be a lowercase 64-character SHA256')
    elif actual_contract != expected_contract_sha256:
        errors.append('contract_sha256 does not match the supplied contract')
    if not isinstance(actual_trace, str) or not SHA256_RE.match(actual_trace):
        errors.append('trace_sha256 must be a lowercase 64-character SHA256')
    elif actual_trace != expected_trace_sha256:
        errors.append('trace_sha256 does not match the candidate trace')

    seen = []
    required_set = set(required_gate_ids)
    na_set = set(not_applicable_gate_ids)
    for index, gate in enumerate(gates):
        if not isinstance(gate, dict):
            errors.append('gate[%d] must be an object' % index)
            continue
        gate_id = gate.get('id', gate.get('gate_id'))
        status = gate.get('status')
        if not isinstance(gate_id, str):
            errors.append('gate[%d] is missing id' % index)
            continue
        seen.append(gate_id)
        if gate_id in na_set:
            # Defect 3: a not-applicable gate is recorded as N/A and never run;
            # any other status would mean the gate actually executed.
            if status != 'N/A':
                errors.append('%s status is %r (required N/A)' % (gate_id, status))
        elif status != 'PASS':
            errors.append('%s status is %r (required PASS)' % (gate_id, status))

    duplicates = sorted(set(item for item in seen if seen.count(item) > 1))
    if duplicates:
        errors.append('duplicate gate IDs: %s' % ', '.join(duplicates))
    missing = [gate_id for gate_id in required_gate_ids if gate_id not in seen]
    missing_na = [gate_id for gate_id in not_applicable_gate_ids
                  if gate_id not in seen]
    unknown = sorted(set(gate_id for gate_id in seen
                         if gate_id not in required_set and gate_id not in na_set))
    if missing:
        errors.append('missing gate IDs: %s' % ', '.join(missing))
    if missing_na:
        errors.append('missing not-applicable gate IDs: %s' % ', '.join(missing_na))
    if unknown:
        errors.append('unknown gate IDs: %s' % ', '.join(unknown))
    return errors


def validate_all_gates(path, contract_sha256, trace_sha256, required_gate_ids=None,
                       not_applicable_gate_ids=None):
    payload = _load_json(path, 'ALL_GATES')
    errors = validate_all_gates_payload(payload, contract_sha256, trace_sha256,
                                        required_gate_ids, not_applicable_gate_ids)
    if errors:
        raise ValueError('ALL_GATES validation failed: ' + '; '.join(errors))
    return payload


def _contract_preflight_partition(contract):
    execution = contract.get('execution') if isinstance(contract, dict) else None
    preflight = execution.get('preflight') if isinstance(execution, dict) else None
    if not isinstance(preflight, dict):
        return None, None
    required = preflight.get('required_check_ids')
    not_applicable = preflight.get('not_applicable_check_ids')
    if (not isinstance(required, list) or
            not all(isinstance(item, str) for item in required)):
        required = None
    if (not isinstance(not_applicable, list) or
            not all(isinstance(item, str) for item in not_applicable)):
        not_applicable = None
    return required, not_applicable


def validate_preflight_payload(payload, expected_contract_sha256,
                               required_check_ids, not_applicable_check_ids):
    """Validate the canonical preflight.v1 report in memory."""
    errors = []
    if not isinstance(payload, dict):
        return ['PREFLIGHT root must be an object']
    if payload.get('schema_version') != 'preflight.v1':
        errors.append('schema_version must be preflight.v1')
    if payload.get('status') != 'PASS':
        errors.append('top-level status is not PASS')
    if not isinstance(required_check_ids, list) or not all(
            isinstance(item, str) and item in PREFLIGHT_IDS
            for item in required_check_ids):
        errors.append('contract required preflight IDs are invalid')
        required_check_ids = []
    if not isinstance(not_applicable_check_ids, list) or not all(
            isinstance(item, str) and item in PREFLIGHT_IDS
            for item in not_applicable_check_ids):
        errors.append('contract not-applicable preflight IDs are invalid')
        not_applicable_check_ids = []
    if len(set(required_check_ids)) != len(required_check_ids):
        errors.append('contract required preflight IDs are not unique')
    if len(set(not_applicable_check_ids)) != len(not_applicable_check_ids):
        errors.append('contract not-applicable preflight IDs are not unique')
    if (set(required_check_ids).intersection(not_applicable_check_ids) or
            set(required_check_ids).union(not_applicable_check_ids) != set(PREFLIGHT_IDS)):
        errors.append('contract preflight IDs must partition E01-E27')
    declared_required = payload.get('required_check_ids')
    declared_na = payload.get('not_applicable_check_ids')
    if not isinstance(declared_required, list):
        errors.append('required_check_ids must be a list')
        declared_required = []
    elif not all(isinstance(item, str) for item in declared_required):
        errors.append('required_check_ids must contain strings')
        declared_required = [item for item in declared_required
                             if isinstance(item, str)]
    if not isinstance(declared_na, list):
        errors.append('not_applicable_check_ids must be a list')
        declared_na = []
    elif not all(isinstance(item, str) for item in declared_na):
        errors.append('not_applicable_check_ids must contain strings')
        declared_na = [item for item in declared_na if isinstance(item, str)]
    if (len(declared_required) != len(required_check_ids) or
            set(declared_required) != set(required_check_ids)):
        errors.append('PREFLIGHT required_check_ids do not match the contract')
    if (len(declared_na) != len(not_applicable_check_ids) or
            set(declared_na) != set(not_applicable_check_ids)):
        errors.append('PREFLIGHT not_applicable_check_ids do not match the contract')
    partition = declared_required + declared_na
    if any(item not in PREFLIGHT_IDS for item in partition):
        errors.append('preflight partition contains unknown IDs')
    if len(set(partition)) != len(partition):
        errors.append('preflight partition IDs are not unique')
    if set(partition) != set(PREFLIGHT_IDS):
        errors.append('preflight E01-E27 partition is incomplete')

    actual_contract = _hash_field(payload, 'contract_sha256',
                                  'CONTRACT_LOCK_SHA256')
    if not isinstance(actual_contract, str) or not SHA256_RE.match(actual_contract):
        errors.append('contract_sha256 must be a lowercase 64-character SHA256')
    elif actual_contract != expected_contract_sha256:
        errors.append('preflight contract_sha256 does not match the contract')
    if not isinstance(payload.get('input_errors'), list):
        errors.append('preflight input_errors must be a list')
    elif payload['input_errors']:
        errors.append('preflight input_errors must be empty')
    if not isinstance(payload.get('partition_errors'), list):
        errors.append('preflight partition_errors must be a list')
    elif payload['partition_errors']:
        errors.append('preflight partition_errors must be empty')
    summary = payload.get('summary')
    if not isinstance(summary, dict):
        errors.append('preflight summary must be an object')
        summary = {}
    if summary.get('all_required_pass') is not True:
        errors.append('preflight summary.all_required_pass must be true')
    contract_validation = payload.get('contract_validation')
    if not isinstance(contract_validation, dict):
        errors.append('preflight contract_validation must be an object')
    else:
        if contract_validation.get('status') != 'PASS':
            errors.append('preflight contract_validation.status must be PASS')
        if contract_validation.get('errors') != []:
            errors.append('preflight contract_validation.errors must be empty')
    if 'contract_sha256_after' in payload and (
            payload.get('contract_sha256_after') != payload.get('contract_sha256')):
        errors.append('preflight contract_sha256_after does not match contract_sha256')
    if 'integrity_errors' in payload and payload.get('integrity_errors') != []:
        errors.append('preflight integrity_errors must be empty')
    expected_counts = {
        'required_count': len(required_check_ids),
        'pass': len(required_check_ids),
        'fail': 0,
        'blocked': 0,
    }
    for key, expected in expected_counts.items():
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            errors.append('preflight summary.%s is inconsistent' % key)

    checks = payload.get('checks')
    if not isinstance(checks, list):
        errors.append('checks must be a list')
        return errors
    seen = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            errors.append('check[%d] must be an object' % index)
            continue
        check_id = check.get('check_id', check.get('id'))
        if not isinstance(check_id, str):
            errors.append('check[%d] is missing check_id' % index)
            continue
        seen.append(check_id)
        if check.get('status') != 'PASS':
            errors.append('%s status is not PASS' % check_id)
    if len(set(seen)) != len(seen):
        errors.append('preflight check IDs are not unique')
    missing = [item for item in required_check_ids if item not in seen]
    unknown = [item for item in seen if item not in required_check_ids]
    if missing:
        errors.append('missing preflight checks: %s' % ', '.join(missing))
    if unknown:
        errors.append('unexpected preflight checks: %s' % ', '.join(sorted(set(unknown))))
    return errors


def validate_preflight(path, contract_sha256, required_check_ids,
                       not_applicable_check_ids):
    payload = _load_json(path, 'PREFLIGHT')
    errors = validate_preflight_payload(payload, contract_sha256,
                                        required_check_ids,
                                        not_applicable_check_ids)
    if errors:
        raise ValueError('PREFLIGHT validation failed: ' + '; '.join(errors))
    return payload


def _is_example_contract(path, payload):
    name = path.name.lower()
    if 'example' in name:
        return True
    try:
        text = json.dumps(payload, sort_keys=True).upper()
    except Exception:
        return False
    def has_explicit_marker(value):
        if isinstance(value, dict):
            if value.get('example_only') is True:
                return True
            expected = value.get('production_validation_expected')
            if isinstance(expected, str) and expected.upper() == 'REJECT':
                return True
            return any(has_explicit_marker(item) for item in value.values())
        if isinstance(value, list):
            return any(has_explicit_marker(item) for item in value)
        return False
    if has_explicit_marker(payload):
        return True
    return ('EXAMPLE_ONLY' in text or 'EXAMPLE CONTRACT' in text or
            'EXAMPLE_CONTRACT' in text or 'NOT_A_REAL_SHA256' in text or
            'PLACEHOLDER' in text or 'REPLACE_ME' in text)


def _atomic_write_json(path, payload):
    """Write beside the receipt and replace it atomically; never use /tmp."""
    partial = path.with_name(path.name + '.partial')
    encoded = json.dumps(payload, indent=2, sort_keys=True) + '\n'
    try:
        with partial.open('w', encoding='utf-8', newline='\n') as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(partial), str(path))
    except Exception:
        try:
            if partial.exists():
                partial.unlink()
        except OSError:
            pass
        raise


def _timing_summary_projection(path):
    """Read a run_measurement TIMING.json and return a compact receipt summary.

    Defect 1: the finalizer receives the per-stage timing file and embeds a
    projection into the receipt.  This never raises -- any failure produces a
    structured ``available: False`` entry so timing cannot block a valid
    delivery, and the projection is purely additive so hash closure is intact."""
    try:
        data = _load_json(path, 'timing summary')
    except Exception as error:
        return {'available': False, 'error': str(error)}
    if not isinstance(data, dict):
        return {'available': False, 'error': 'timing summary is not an object'}
    phases = []
    raw_phases = data.get('phases')
    if isinstance(raw_phases, list):
        for phase in raw_phases:
            if not isinstance(phase, dict):
                continue
            entry = {}
            for key in ('name', 'kind', 'status', 'wall_seconds',
                        'execute_seconds', 'gpu_wait_seconds', 'bound_gpu'):
                if key in phase:
                    entry[key] = phase[key]
            phases.append(entry)
    projection = {'available': True, 'source_path': str(path), 'phases': phases}
    for key in ('schema', 'bundle', 'status', 'started_at', 'generated_at',
                'wall_seconds_total', 'gpu_wait_seconds_total', 'phase_count'):
        if key in data:
            projection[key] = data[key]
    projection.setdefault('phase_count', len(phases))
    return projection


def _selftest():
    contract_hash = 'a' * 64
    trace_hash = 'b' * 64
    good = {
        'schema_version': 'all_gates.v1',
        'status': 'PASS',
        'contract_sha256': contract_hash,
        'trace_sha256': trace_hash,
        'contract_sha256_after': contract_hash,
        'trace_sha256_after': trace_hash,
        'integrity_errors': [],
        'required_gate_ids': list(ALL_GATES_IDS),
        'not_applicable_gate_ids': [],
        'input_errors': [],
        'partition_errors': [],
        'gates': [{'gate_id': gate_id, 'status': 'PASS'}
                  for gate_id in ALL_GATES_IDS],
        'summary': {
            'required_count': len(ALL_GATES_IDS),
            'pass': len(ALL_GATES_IDS),
            'fail': 0,
            'blocked': 0,
            'all_required_pass': True,
            'counts_consistent': True,
        },
    }
    assert validate_all_gates_payload(good, contract_hash, trace_hash,
                                      list(ALL_GATES_IDS)) == []

    preflight_required = list(PREFLIGHT_IDS)
    preflight_na = []
    preflight_good = {
        'schema_version': 'preflight.v1',
        'status': 'PASS',
        'contract_sha256': contract_hash,
        'contract_sha256_after': contract_hash,
        'integrity_errors': [],
        'contract_validation': {'status': 'PASS', 'errors': []},
        'required_check_ids': preflight_required,
        'not_applicable_check_ids': preflight_na,
        'input_errors': [],
        'partition_errors': [],
        'checks': [{'check_id': check_id, 'status': 'PASS'}
                   for check_id in PREFLIGHT_IDS],
        'summary': {
            'required_count': len(PREFLIGHT_IDS),
            'pass': len(PREFLIGHT_IDS), 'fail': 0, 'blocked': 0,
            'all_required_pass': True,
        },
    }
    assert validate_preflight_payload(preflight_good, contract_hash,
                                      preflight_required, preflight_na) == []
    bad_preflight_na = json.loads(json.dumps(preflight_good))
    bad_preflight_na['not_applicable_check_ids'].append('E01')
    assert validate_preflight_payload(bad_preflight_na, contract_hash,
                                      preflight_required, preflight_na)
    bad_preflight_hash = json.loads(json.dumps(preflight_good))
    bad_preflight_hash['contract_sha256'] = 'c' * 64
    assert validate_preflight_payload(bad_preflight_hash, contract_hash,
                                      preflight_required, preflight_na)
    bad_preflight_type = json.loads(json.dumps(preflight_good))
    bad_preflight_type['not_applicable_check_ids'] = [[]]
    assert validate_preflight_payload(bad_preflight_type, contract_hash,
                                      preflight_required, preflight_na)
    bad_preflight_contract = json.loads(json.dumps(preflight_good))
    bad_preflight_contract['contract_validation']['status'] = 'FAIL'
    assert validate_preflight_payload(bad_preflight_contract, contract_hash,
                                      preflight_required, preflight_na)
    bad_preflight_integrity = json.loads(json.dumps(preflight_good))
    bad_preflight_integrity['integrity_errors'] = ['forged']
    assert validate_preflight_payload(bad_preflight_integrity, contract_hash,
                                      preflight_required, preflight_na)

    bad_status = json.loads(json.dumps(good))
    bad_status['gates'][5]['status'] = 'BLOCKED'
    assert validate_all_gates_payload(bad_status, contract_hash, trace_hash)

    bad_ids = json.loads(json.dumps(good))
    bad_ids['gates'][0]['id'] = 'G01'
    assert validate_all_gates_payload(bad_ids, contract_hash, trace_hash)

    bad_hash = json.loads(json.dumps(good))
    bad_hash['trace_sha256'] = 'c' * 64
    assert validate_all_gates_payload(bad_hash, contract_hash, trace_hash,
                                      list(ALL_GATES_IDS))

    partial = json.loads(json.dumps(good))
    partial['required_gate_ids'] = ['G00']
    partial['not_applicable_gate_ids'] = [gate_id for gate_id in ALL_GATES_IDS
                                          if gate_id != 'G00']
    partial['gates'] = ([{'gate_id': 'G00', 'status': 'PASS'}]
                        + [{'gate_id': gate_id, 'status': 'N/A'}
                           for gate_id in ALL_GATES_IDS if gate_id != 'G00'])
    partial['summary'] = {
        'required_count': 1, 'pass': 1, 'fail': 0, 'blocked': 0,
        'all_required_pass': True, 'counts_consistent': True,
    }
    # Defect 3: a partial partition is accepted when the declared not-applicable
    # set matches the caller-supplied expectation.
    assert validate_all_gates_payload(
        partial, contract_hash, trace_hash, ['G00'],
        [gate_id for gate_id in ALL_GATES_IDS if gate_id != 'G00']) == []

    bad_na = json.loads(json.dumps(partial))
    bad_na['not_applicable_gate_ids'][0] = 'G00'
    assert validate_all_gates_payload(bad_na, contract_hash, trace_hash,
                                      ['G00'])
    bad_partition_type = json.loads(json.dumps(good))
    bad_partition_type['not_applicable_gate_ids'] = [{}]
    assert validate_all_gates_payload(bad_partition_type, contract_hash,
                                      trace_hash)

    bad_input = json.loads(json.dumps(good))
    bad_input['input_errors'] = ['forged input error']
    assert validate_all_gates_payload(bad_input, contract_hash, trace_hash)

    bad_summary = json.loads(json.dumps(good))
    bad_summary['summary']['all_required_pass'] = False
    assert validate_all_gates_payload(bad_summary, contract_hash, trace_hash)
    bad_integrity = json.loads(json.dumps(good))
    bad_integrity['integrity_errors'] = ['forged']
    assert validate_all_gates_payload(bad_integrity, contract_hash, trace_hash)
    bad_after_hash = json.loads(json.dumps(good))
    bad_after_hash['trace_sha256_after'] = 'c' * 64
    assert validate_all_gates_payload(bad_after_hash, contract_hash, trace_hash)

    # Defect 3: bundle-aware partitions.  The hardcoded per-bundle N/A sets are
    # the single source of truth; a contract can only select a bundle.
    sw_req_g, sw_na_g, sw_req_e, sw_na_e = _bundle_partition('sw')
    assert set(sw_na_g) == {'G11', 'G19', 'G20', 'G21'}
    assert set(sw_na_e) == {'E17', 'E18', 'E19', 'E20'}
    swhw_partition = _bundle_partition('sw+hw')
    assert swhw_partition[1] == [] and swhw_partition[3] == []

    sw_gates = json.loads(json.dumps(good))
    sw_gates['required_gate_ids'] = sw_req_g
    sw_gates['not_applicable_gate_ids'] = sw_na_g
    # validate_all.py emits PASS records for required gates and N/A records for
    # the not-applicable ones; the finalizer must accept exactly that shape.
    sw_gates['gates'] = ([{'gate_id': gate_id, 'status': 'PASS'}
                         for gate_id in sw_req_g]
                        + [{'gate_id': gate_id, 'status': 'N/A'}
                           for gate_id in sw_na_g])
    sw_gates['summary'] = {
        'required_count': len(sw_req_g), 'pass': len(sw_req_g), 'fail': 0,
        'blocked': 0, 'all_required_pass': True, 'counts_consistent': True,
    }
    assert validate_all_gates_payload(sw_gates, contract_hash, trace_hash,
                                      sw_req_g, sw_na_g) == []
    sw_gates_bad = json.loads(json.dumps(sw_gates))
    sw_gates_bad['not_applicable_gate_ids'] = []
    assert validate_all_gates_payload(sw_gates_bad, contract_hash, trace_hash,
                                      sw_req_g, sw_na_g)
    # Negative control: a not-applicable gate reported as PASS means it actually
    # ran, which must be rejected even though the partition metadata is correct.
    sw_gates_ran_na = json.loads(json.dumps(sw_gates))
    for gate in sw_gates_ran_na['gates']:
        if gate['gate_id'] in set(sw_na_g):
            gate['status'] = 'PASS'
    assert validate_all_gates_payload(sw_gates_ran_na, contract_hash, trace_hash,
                                      sw_req_g, sw_na_g)

    sw_pre = json.loads(json.dumps(preflight_good))
    sw_pre['required_check_ids'] = sw_req_e
    sw_pre['not_applicable_check_ids'] = sw_na_e
    sw_pre['checks'] = [{'check_id': check_id, 'status': 'PASS'}
                       for check_id in sw_req_e]
    sw_pre['summary'] = {
        'required_count': len(sw_req_e), 'pass': len(sw_req_e), 'fail': 0,
        'blocked': 0, 'all_required_pass': True,
    }
    assert validate_preflight_payload(sw_pre, contract_hash,
                                      sw_req_e, sw_na_e) == []
    sw_pre_bad = json.loads(json.dumps(sw_pre))
    sw_pre_bad['required_check_ids'] = [c for c in sw_req_e if c != 'E01']
    sw_pre_bad['not_applicable_check_ids'] = ['E01'] + list(sw_na_e)
    sw_pre_bad['checks'] = [{'check_id': c, 'status': 'PASS'}
                           for c in sw_pre_bad['required_check_ids']]
    assert validate_preflight_payload(sw_pre_bad, contract_hash,
                                      sw_req_e, sw_na_e)

    example = {'finalizer': {'sha256': 'EXAMPLE_ONLY_NOT_A_REAL_SHA256'}}
    assert _is_example_contract(Path('contract_lock_example.json'), example)
    assert _is_example_contract(Path('CONTRACT.lock.json'), {'example_only': True})
    assert _is_example_contract(Path('CONTRACT.lock.json'),
                                {'production_validation_expected': 'REJECT'})
    assert not _is_example_contract(Path('CONTRACT.lock.json'),
                                    {'contract_version': '1'})
    print('SELFTEST=PASS')
    return 0



def _skill_manifest_binding():
    """Pin the receipt to the exact skill tree that produced it.

    Delegates to weaveprof/pipeline/validate/harness/manifest.py verify, whose anchor is the git object
    store.  Raises ValueError when the working tree has drifted from the
    committed manifest: delivering with modified gate code is worse than no
    delivery at all.
    """
    skill_root = Path(__file__).resolve().parents[3]
    manifest_py = skill_root / 'weaveprof/pipeline/validate' / 'harness' / 'manifest.py'
    if not manifest_py.is_file():
        raise ValueError('harness manifest tool missing: %s' % manifest_py)
    proc = subprocess.run(
        [sys.executable, '-B', str(manifest_py), 'verify',
         '--skill-root', str(skill_root), '--json'],
        capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise ValueError('skill manifest verification failed (commit the skill '
                         'tree and rebuild MANIFEST.json): %s'
                         % (proc.stdout or proc.stderr).strip()[:500])
    report = json.loads(proc.stdout)
    return {'skill_root': str(skill_root),
            'anchor': report.get('anchor'),
            'root_sha256': report.get('root_sha256'),
            'file_count': report.get('file_count')}


def main():
    # --selftest is intentionally a dependency-free mode, so bypass the
    # delivery parser before enforcing the normal required arguments.
    if len(sys.argv) == 2 and sys.argv[1] == '--selftest':
        return _selftest()
    parser = argparse.ArgumentParser(
        description='Finalize a machine-verifiable Perfetto delivery')
    parser.add_argument('project', nargs='?')
    parser.add_argument('trace', nargs='?')
    parser.add_argument('--selftest', action='store_true',
                        help='run in-memory good/bad ALL_GATES validation tests')
    parser.add_argument('--trace-processor', default=None,
                        help='absolute trace_processor_shell path')
    parser.add_argument('--receipt', default=None)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--all-gates', required=True)
    parser.add_argument('--preflight', required=True)
    parser.add_argument('--information-bundle', choices=('sw', 'hw', 'sw+hw'),
                        default='sw+hw')
    parser.add_argument('--capture-manifest', default=None)
    parser.add_argument('--hw-metrics', default=None)
    parser.add_argument('--compact-exact', default=None)
    parser.add_argument('--abi', default=None)
    parser.add_argument('--timing-summary', default=None,
                        help='path to run_measurement TIMING.json to embed in the receipt')
    args = parser.parse_args()

    if args.selftest:
        parser.error('--selftest must be used by itself')
    for option, value in (('--trace-processor', args.trace_processor),):
        if not value:
            parser.error('%s is required unless --selftest is used' % option)
    if not args.project or not args.trace:
        parser.error('project and trace are required unless --selftest is used')

    trace = Path(args.trace).expanduser().resolve()
    tool = Path(args.trace_processor).expanduser().resolve()
    contract = Path(args.contract).expanduser().resolve()
    all_gates = Path(args.all_gates).expanduser().resolve()
    preflight = Path(args.preflight).expanduser().resolve()
    if (args.information_bundle in ('hw', 'sw+hw') and
            (not args.capture_manifest or not args.hw_metrics)):
        print('[fatal] hw and sw+hw require --capture-manifest and --hw-metrics',
              file=sys.stderr)
        return 2
    receipt = (Path(args.receipt).expanduser().resolve() if args.receipt
               else Path(str(trace) + '.delivery_receipt.json'))
    for path, label in ((trace, 'trace'), (tool, 'trace_processor_shell'),
                        (contract, 'contract'), (all_gates, 'ALL_GATES'),
                        (preflight, 'PREFLIGHT')):
        if not path.is_file():
            print('[fatal] %s does not exist: %s' % (label, path), file=sys.stderr)
            return 2

    delivery_inputs = {}
    input_args = (
        ('capture_manifest', args.capture_manifest),
        ('hw_metrics', args.hw_metrics),
        ('compact_exact', args.compact_exact),
        ('abi', args.abi),
    )
    for name, value in input_args:
        if not value:
            continue
        input_path = Path(value).expanduser().resolve()
        if not input_path.is_file():
            print('[fatal] delivery input does not exist: %s' % input_path,
                  file=sys.stderr)
            return 2
        delivery_inputs[name] = {
            'path': str(input_path),
            'sha256': sha256(input_path),
        }
    tool_digest = sha256(tool)

    try:
        contract_payload = _load_json(contract, 'contract')
        if _is_example_contract(contract, contract_payload):
            raise ValueError('example contract is not accepted')
        required_gate_ids, not_applicable_gate_ids = _contract_gate_partition(
            contract_payload)
        if required_gate_ids is None or not_applicable_gate_ids is None:
            raise ValueError('contract gate partition is missing or invalid')
        required_check_ids, not_applicable_check_ids = _contract_preflight_partition(
            contract_payload)
        if required_check_ids is None or not_applicable_check_ids is None:
            raise ValueError('contract preflight partition is missing or invalid')
        # Layer 1: the information bundle fully determines the allowed partition.
        # The contract may only select a bundle, so its declared required/N/A
        # IDs must match the hardcoded bundle sets exactly.
        exp_req_gates, exp_na_gates, exp_req_checks, exp_na_checks = _bundle_partition(
            args.information_bundle)
        if (set(required_gate_ids) != set(exp_req_gates) or
                set(not_applicable_gate_ids) != set(exp_na_gates)):
            raise ValueError('contract gate partition does not match the %s bundle'
                             % args.information_bundle)
        if (set(required_check_ids) != set(exp_req_checks) or
                set(not_applicable_check_ids) != set(exp_na_checks)):
            raise ValueError('contract preflight partition does not match the %s bundle'
                             % args.information_bundle)
        contract_digest = sha256(contract)
        trace_digest = sha256(trace)
        preflight_digest = sha256(preflight)
        validate_preflight(preflight, contract_digest, required_check_ids,
                           not_applicable_check_ids)
        validate_all_gates(all_gates, contract_digest, trace_digest,
                           required_gate_ids, not_applicable_gate_ids)
    except ValueError as error:
        print('[fatal] %s' % error, file=sys.stderr)
        return 2

    command = [sys.executable, '-B', str(GATE), str(Path(args.project).resolve()),
               str(trace), '--trace-processor', str(tool), '--receipt', str(receipt),
               '--contract', str(contract),
               '--information-bundle', args.information_bundle]
    for flag, value in (
        ('--capture-manifest', delivery_inputs.get('capture_manifest', {}).get('path')),
        ('--hw-metrics', delivery_inputs.get('hw_metrics', {}).get('path')),
        ('--compact-exact', delivery_inputs.get('compact_exact', {}).get('path')),
        ('--abi', delivery_inputs.get('abi', {}).get('path')),
    ):
        if value:
            command.extend((flag, value))
    result = subprocess.run(command)
    if result.returncode != 0:
        return result.returncode

    try:
        payload = _load_json(receipt, 'PASS receipt')
        observed = sha256(trace)
        if payload.get('status') != 'PASS' or payload.get('trace_sha256') != observed:
            raise ValueError('deliver_gate receipt status/hash does not match final trace')
        if sha256(contract) != contract_digest:
            raise ValueError('contract changed during delivery validation')
        if sha256(tool) != tool_digest:
            raise ValueError('trace_processor_shell changed during delivery validation')
        for name, record in delivery_inputs.items():
            input_path = Path(record['path'])
            if not input_path.is_file() or sha256(input_path) != record['sha256']:
                raise ValueError('delivery input changed: %s' % input_path)
        if sha256(preflight) != preflight_digest:
            raise ValueError('preflight changed during delivery validation')
        validate_preflight(preflight, contract_digest, required_check_ids,
                           not_applicable_check_ids)
        # Re-read the evidence after the delegated validator returns.  This
        # closes the same hash binding used before invocation.
        validate_all_gates(all_gates, contract_digest, observed,
                           required_gate_ids, not_applicable_gate_ids)
        all_gates_digest = sha256(all_gates)
        skill_manifest = _skill_manifest_binding()
        payload.update({
            'skill_manifest': skill_manifest,
            'trace_path': str(trace),
            'contract_path': str(contract),
            'contract_sha256': contract_digest,
            'all_gates_path': str(all_gates),
            'all_gates_sha256': all_gates_digest,
            'preflight_path': str(preflight),
            'preflight_sha256': preflight_digest,
            'information_bundle': args.information_bundle,
            'trace_processor_path': str(tool),
            'trace_processor_sha256': tool_digest,
            'delivery_inputs': delivery_inputs,
            'g25_finalization': {
                'gate_id': 'G25',
                'status': 'PASS',
                'trace_path': str(trace),
                'trace_sha256': observed,
            },
        })
        # Defect 1: embed the per-stage timing summary additively.  When
        # --timing-summary is not supplied the receipt is byte-identical to the
        # previous output, and the final verification below never reads this key
        # so the receipt hash closure is unaffected.
        if args.timing_summary:
            payload['timing_summary'] = _timing_summary_projection(
                Path(args.timing_summary).expanduser().resolve())
        _atomic_write_json(receipt, payload)
        final_trace_digest = sha256(trace)
        final_payload = _load_json(receipt, 'final receipt')
        if (final_trace_digest != observed or
                final_payload.get('status') != 'PASS' or
                final_payload.get('trace_path') != str(trace) or
                final_payload.get('trace_sha256') != final_trace_digest or
                final_payload.get('information_bundle') != args.information_bundle or
                final_payload.get('trace_processor_path') != str(tool) or
                final_payload.get('trace_processor_sha256') != tool_digest or
                final_payload.get('delivery_inputs') != delivery_inputs or
                final_payload.get('contract_path') != str(contract) or
                final_payload.get('contract_sha256') != contract_digest or
                final_payload.get('all_gates_path') != str(all_gates) or
                final_payload.get('all_gates_sha256') != all_gates_digest or
                final_payload.get('preflight_path') != str(preflight) or
                final_payload.get('preflight_sha256') != preflight_digest or
                final_payload.get('skill_manifest') != skill_manifest or
                final_payload.get('g25_finalization', {}).get('gate_id') != 'G25' or
                final_payload.get('g25_finalization', {}).get('status') != 'PASS' or
                final_payload.get('g25_finalization', {}).get('trace_path') != str(trace) or
                final_payload.get('g25_finalization', {}).get('trace_sha256') != final_trace_digest):
            raise ValueError('final receipt/hash verification failed')
    except Exception as error:
        print('[fatal] PASS receipt cannot be finalized: %s' % error, file=sys.stderr)
        return 3

    print('FINALIZED_TRACE=%s' % trace)
    print('DELIVERY_RECEIPT=%s' % receipt)
    print('TRACE_SHA256=%s' % final_trace_digest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
