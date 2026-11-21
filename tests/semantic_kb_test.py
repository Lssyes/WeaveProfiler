#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contract tests for the canonical operator semantic KB.

Runs offline and is deliberately compatible with the default Python 3.6 on the
shared host.  Invalid copies are mutated only inside TemporaryDirectory.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
KB = ROOT / 'knowledge'
OPERATORS = KB / 'operators' / 'operators.json'
FIXES = KB / 'validation' / 'fixes.json'

EXPECTED_OPERATOR_IDS = {
    'gemm', 'attention', 'reduction', 'moe', 'convolution', 'megakernel',
}
REQUIRED_STAGE_IDS = {
    'load', 'wait', 'mma', 'compute', 'epilogue',
    'store', 'barrier', 'dispatch', 'communication',
}
EVIDENCE_STATUSES = {'measured', 'compiled', 'simulated', 'unverified'}
MACHINE_CHECKS = {
    'correctness', 'overhead', 'assembly_change_rate', 'alignment', 'buffer_completeness',
}
LEGACY_TO_CANONICAL = {
    '装载': {'load', 'epilogue'},
    '等待': {'wait'},
    '计算': {'mma', 'compute', 'epilogue', 'reduction', 'softmax', 'dispatch'},
    '同步': {'barrier', 'dispatch', 'communication', 'wait'},
    '写回': {'store', 'epilogue', 'communication'},
}
OPERATOR_MARKERS = {
    'gemm': {'mma', 'load', 'store'},
    'attention': {'mma', 'softmax', 'store'},
    'reduction': {'reduction', 'barrier', 'store'},
    'moe': {'dispatch', 'load', 'store', 'communication'},
    'convolution': {'mma', 'load', 'store'},
    'megakernel': {'dispatch', 'wait', 'communication'},
}


def load_json(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def load_validator():
    path = TESTS / 'validate_kb.py'
    spec = importlib.util.spec_from_file_location('weaveprofiler_validate_kb', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CanonicalMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.operators_doc = load_json(OPERATORS)
        cls.semantic_doc = load_json(KB / 'semantic_ids.json')
        cls.canonical_ids = {
            item['semantic_id'] for item in cls.semantic_doc['semantics']
        }
        cls.operators = cls.operators_doc['operators']

    def test_every_operator_has_one_unique_canonical_id(self):
        ids = [op.get('semantic_id') for op in self.operators]
        self.assertEqual(len(ids), len(set(ids)), 'operator semantic_id 必须唯一')
        self.assertEqual(set(ids), EXPECTED_OPERATOR_IDS)
        self.assertTrue(set(ids) <= self.canonical_ids)

    def test_every_stage_has_valid_canonical_mapping(self):
        coverage = set()
        for op in self.operators:
            self.assertIn('semantic_id', op)
            for stage in op['stages']:
                where = '%s/%s' % (op['semantic_id'], stage['stage'])
                semantic_ids = stage.get('semantic_ids')
                self.assertIsInstance(semantic_ids, list, where)
                self.assertTrue(semantic_ids, where)
                self.assertEqual(len(semantic_ids), len(set(semantic_ids)), where)
                self.assertNotIn('unknown', semantic_ids, where)
                self.assertTrue(set(semantic_ids) <= self.canonical_ids,
                                '%s 含未知 ID: %r' % (where, semantic_ids))
                coverage.update(semantic_ids)

                legacy = stage.get('semantics')
                self.assertIsInstance(legacy, list, where)
                self.assertTrue(legacy, where)
                self.assertTrue(set(legacy) <= set(LEGACY_TO_CANONICAL), where)
                for legacy_id in legacy:
                    self.assertTrue(
                        set(semantic_ids) & LEGACY_TO_CANONICAL[legacy_id],
                        '%s: legacy=%s 没有兼容的 canonical ID' % (where, legacy_id),
                    )
        self.assertTrue(REQUIRED_STAGE_IDS <= coverage,
                        'stage 映射未覆盖：%r' % sorted(REQUIRED_STAGE_IDS - coverage))

    def test_operator_specific_semantics_are_not_flat_aliases(self):
        for op in self.operators:
            stage_ids = {
                semantic_id
                for stage in op['stages']
                for semantic_id in stage['semantic_ids']
            }
            expected = OPERATOR_MARKERS[op['semantic_id']]
            self.assertTrue(expected <= stage_ids,
                            '%s 缺关键语义：%r' % (op['semantic_id'], sorted(expected - stage_ids)))
        self.assertEqual(self.operators_doc['semantic_taxonomy'],
                         ['装载', '等待', '计算', '同步', '写回'])

    def test_machine_checks_and_evidence_status_are_structured(self):
        catalog = self.operators_doc.get('machine_check_catalog', {})
        self.assertTrue(MACHINE_CHECKS <= set(catalog))
        for check_id in MACHINE_CHECKS:
            self.assertTrue(catalog[check_id].get('required'))
            self.assertTrue(catalog[check_id].get('pass_condition'))

        for op in self.operators:
            where = op['semantic_id']
            self.assertNotIn('measurement_status', op, where)
            self.assertIn(op.get('evidence_status'), EVIDENCE_STATUSES, where)
            self.assertTrue(op.get('evidence_note'), where)
            self.assertTrue(MACHINE_CHECKS <= set(op.get('machine_check', [])), where)

    def test_every_kb_ref_resolves_to_a_real_target(self):
        ref_count = 0
        for op in self.operators:
            for stage in op['stages']:
                refs = stage.get('kb_refs', [])
                self.assertIsInstance(refs, list)
                for ref in refs:
                    ref_count += 1
                    self.assertIn('::', ref)
                    self.assertTrue(
                        ref.startswith('operators.json::stage:')
                        or (ref.split('::', 1)[0].endswith('.json')
                            and ref.split('::', 1)[1].startswith('entry:')),
                        ref,
                    )
        self.assertGreater(ref_count, 0)
        self.assertEqual(load_validator().validate_kb(KB), [])


class InvalidKBMutationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='weaveprofiler-kb-test-')
        self.kb = Path(self.temp.name) / 'kb'
        shutil.copytree(str(KB), str(self.kb))

    def tearDown(self):
        self.temp.cleanup()

    def mutate_and_validate(self, mutator):
        path = self.kb / 'operators' / 'operators.json'
        doc = load_json(path)
        mutator(doc)
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return load_validator().validate_kb(self.kb)

    def test_invalid_operator_id_fails(self):
        def mutate(doc):
            doc['operators'][0]['semantic_id'] = 'not_a_canonical_operator'
        errors = self.mutate_and_validate(mutate)
        self.assertTrue(any('not_a_canonical_operator' in e for e in errors), errors)

    def test_duplicate_operator_id_fails(self):
        def mutate(doc):
            doc['operators'][1]['semantic_id'] = doc['operators'][0]['semantic_id']
        errors = self.mutate_and_validate(mutate)
        self.assertTrue(any('semantic_id=' in e and '重复' in e for e in errors), errors)

    def test_invalid_stage_semantic_id_fails(self):
        def mutate(doc):
            doc['operators'][0]['stages'][0]['semantic_ids'] = ['invented_stage']
        errors = self.mutate_and_validate(mutate)
        self.assertTrue(any('invented_stage' in e for e in errors), errors)

    def test_dangling_kb_ref_fails(self):
        def mutate(doc):
            doc['operators'][0]['stages'][0]['kb_refs'] = [
                'sm_90.json::entry:this construct does not exist'
            ]
        errors = self.mutate_and_validate(mutate)
        self.assertTrue(any('this construct does not exist' in e for e in errors), errors)

    def test_invalid_evidence_status_fails(self):
        def mutate(doc):
            doc['operators'][0]['evidence_status'] = 'probably_ran'
        errors = self.mutate_and_validate(mutate)
        self.assertTrue(any('probably_ran' in e for e in errors), errors)

    def test_missing_machine_check_fails(self):
        def mutate(doc):
            doc['operators'][0]['machine_check'] = ['correctness']
        errors = self.mutate_and_validate(mutate)
        self.assertTrue(any('machine_check 缺必需检查' in e for e in errors), errors)

    def test_cli_returns_nonzero_for_invalid_id(self):
        path = self.kb / 'operators' / 'operators.json'
        doc = load_json(path)
        doc['operators'][0]['semantic_id'] = 'bad_cli_id'
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        env = dict(os.environ)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        proc = subprocess.run(
            [sys.executable, '-B', str(TESTS / 'validate_kb.py'), str(self.kb)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env,
        )
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn('bad_cli_id', proc.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
