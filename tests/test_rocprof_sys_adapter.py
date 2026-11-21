#!/usr/bin/env python3
"""Offline tests for the MI308X ROCm collector enhancement.

No test launches a GPU workload.  Executable probes use temporary shell
scripts, and capture-path tests mock GPU inspection plus payload execution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import time
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO / 'weaveprof/pipeline/capture'
if str(PROFILE_DIR) not in sys.path:
    sys.path.insert(0, str(PROFILE_DIR))

import capture_rocm  # noqa: E402
import rocprof_sys_adapter as adapter  # noqa: E402

SYS_HELP = '\n'.join(adapter.ROCPROF_SYS_FLAGS_USED) + '\n'
V3_HELP = '\n'.join(capture_rocm.ROCPROF_FLAGS_USED) + '\n'


def write_script(path: Path, help_text: str, *,
                 version: str = 'fake-version',
                 version_rc: int = 0) -> Path:
    path.write_text(
        '#!/bin/sh\n'
        'if [ "$1" = "--version" ]; then\n'
        f'  printf "%s\\n" "{version}"\n'
        '  printf "%s\\n" "probe stderr" >&2\n'
        f'  exit {version_rc}\n'
        'fi\n'
        'if [ "$1" = "--help" ]; then\n'
        "  cat <<'HELP'\n"
        f'{help_text}'
        "HELP\n"
        '  exit 0\n'
        'fi\n'
        'exit 0\n',
        encoding='utf-8')
    path.chmod(0o755)
    return path


class AdapterSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix='weaveprofiler-test-')
        self.root = Path(self.temp.name)
        self.sys_main = write_script(self.root / 'rocprof-sys', SYS_HELP,
                                     version='rocprof-sys fake-version')
        self.sys_run = write_script(self.root / 'rocprof-sys-run', SYS_HELP,
                                    version='rocprof-sys-run fake-version')
        self.sys_record = write_script(self.root / 'rocprof-sys-record', SYS_HELP,
                                       version='rocprof-sys-record fake-version')
        self.v3 = write_script(self.root / 'rocprofv3', V3_HELP,
                               version='rocprofv3 fake-version')

    def tearDown(self) -> None:
        self.temp.cleanup()

    def which_map(self, *names: str):
        paths = {
            'rocprof-sys': self.sys_main,
            'rocprof-sys-run': self.sys_run,
            'rocprof-sys-record': self.sys_record,
            'rocprofv3': self.v3,
        }
        return lambda name: str(paths[name]) if name in names else None

    def test_alias_priority_is_canonical_then_run_then_record(self) -> None:
        all_three = adapter.select_collector(
            which=self.which_map('rocprof-sys', 'rocprof-sys-run',
                                 'rocprof-sys-record', 'rocprofv3'))
        self.assertEqual(all_three['family'], adapter.ROCPROF_SYS_FAMILY)
        self.assertEqual(all_three['binary'], 'rocprof-sys')

        run_and_record = adapter.select_collector(
            which=self.which_map('rocprof-sys-run', 'rocprof-sys-record',
                                 'rocprofv3'))
        self.assertEqual(run_and_record['binary'], 'rocprof-sys-run')

        record_only = adapter.select_collector(
            which=self.which_map('rocprof-sys-record', 'rocprofv3'))
        self.assertEqual(record_only['binary'], 'rocprof-sys-record')

    def test_explicit_rocprof_sys_path_wins(self) -> None:
        selected = adapter.select_collector(
            rocprof_sys=str(self.sys_run), which=lambda _name: None)
        self.assertEqual(selected['family'], adapter.ROCPROF_SYS_FAMILY)
        self.assertEqual(selected['path'], str(self.sys_run.resolve()))
        self.assertFalse(selected['fallback']['used'])

    def test_missing_rocprof_sys_falls_back_with_reason(self) -> None:
        selected = adapter.select_collector(
            which=self.which_map('rocprofv3'))
        self.assertEqual(selected['family'], adapter.ROCPROFV3_FAMILY)
        self.assertEqual(selected['binary'], 'rocprofv3')
        self.assertTrue(selected['fallback']['used'])
        self.assertEqual(selected['fallback']['trigger'], 'not-found')
        self.assertIn('rocprof-sys 不可用', selected['fallback']['reason'])
        self.assertEqual(selected['version_probe']['argv'],
                         [str(self.v3.resolve()), '--version'])

    def test_incompatible_rocprof_sys_cli_fails_preflight_then_falls_back(self) -> None:
        incompatible = write_script(
            self.root / 'rocprof-sys-incompatible',
            '--trace\n--use-rocm\n--output\n--help\n--version\n',
            version='rocprof-sys-record v-legacy')
        selected = adapter.select_collector(
            rocprof_sys=str(incompatible),
            which=self.which_map('rocprofv3'))
        self.assertEqual(selected['family'], adapter.ROCPROFV3_FAMILY)
        self.assertEqual(selected['fallback']['trigger'], 'preflight')
        self.assertIn('--perfetto-backend', selected['fallback']['reason'])

    def test_explicit_rocprofv3_does_not_probe_rocprof_sys(self) -> None:
        selected = adapter.select_collector(
            requested=adapter.ROCPROFV3_FAMILY,
            rocprofv3=str(self.v3), which=lambda _name: None)
        self.assertEqual(selected['family'], adapter.ROCPROFV3_FAMILY)
        self.assertFalse(selected['fallback']['used'])

    def test_version_probe_failure_is_recorded_without_fake_version(self) -> None:
        bad = write_script(self.root / 'bad-sys', SYS_HELP,
                           version='candidate', version_rc=7)
        selected = adapter.select_collector(
            rocprof_sys=str(bad), which=lambda _name: None)
        self.assertEqual(selected['family'], adapter.ROCPROF_SYS_FAMILY)
        self.assertIsNone(selected['version'])
        self.assertFalse(selected['version_probe']['success'])
        self.assertEqual(selected['version_probe']['exit_code'], 7)
        self.assertEqual(selected['version_probe']['argv'],
                         [str(bad.resolve()), '--version'])

    def test_pmc_request_routes_to_v3_and_declares_nonperiodic_semantics(self) -> None:
        selected = adapter.select_collector(
            hw_requested=True,
            which=self.which_map('rocprof-sys-run', 'rocprofv3'))
        self.assertEqual(selected['family'], adapter.ROCPROFV3_FAMILY)
        self.assertTrue(selected['fallback']['used'])
        self.assertEqual(selected['fallback']['trigger'], 'capability')
        self.assertIn('--pmc', selected['fallback']['reason'])
        sampling = selected['capabilities']['sampling']
        self.assertEqual(sampling['counter_granularity'], 'per-dispatch')
        self.assertFalse(sampling['periodic_pm_sampling'])
        self.assertFalse(sampling['equivalent_to_nsys_200khz'])

    def test_rocprof_sys_timeline_capabilities_are_honest(self) -> None:
        selected = adapter.select_collector(
            which=self.which_map('rocprof-sys-run'))
        sampling = selected['capabilities']['sampling']
        self.assertEqual(sampling['mode'], 'event-trace')
        self.assertEqual(sampling['counter_granularity'], 'event-boundary')
        self.assertFalse(sampling['pmc_counters'])
        self.assertFalse(sampling['periodic_pm_sampling'])
        self.assertFalse(sampling['equivalent_to_nsys_200khz'])
        self.assertIn('不是周期 GPU PM 采样', ' '.join(sampling['notes']))


class CommandAndEvidenceTests(unittest.TestCase):
    def test_build_command_preserves_argv_and_shell_quoting(self) -> None:
        payload = ['/bin/echo', 'a b', "quote's", '$HOME', '$(not-run)']
        domains = ('kernel_dispatch', 'memory_copy')
        events = adapter.DEFAULT_ROCM_EVENTS
        cmd = adapter.build_rocprof_sys_cmd(
            '/opt/rocm/bin/rocprof-sys-run', Path('/tmp/out'), payload,
            buffer_kb=4096, rocm_domains=domains, rocm_events=events)
        self.assertEqual(cmd[:3], ['/opt/rocm/bin/rocprof-sys-run',
                                   '--trace', '--use-rocm'])
        domain_start = cmd.index('--rocm-domains')
        event_start = cmd.index('--rocm-events')
        output_start = cmd.index('--output')
        self.assertEqual(cmd[domain_start + 1:event_start], list(domains))
        self.assertEqual(cmd[event_start + 1:output_start], list(events))
        self.assertEqual(cmd[-len(payload):], payload)
        self.assertIn('--perfetto-backend', cmd)
        self.assertEqual(shlex.split(adapter.command_line(cmd)), cmd)
        self.assertIn("'$HOME'", adapter.command_line(cmd))

    def test_build_command_omits_rocm_options_when_not_requested(self) -> None:
        cmd = adapter.build_rocprof_sys_cmd(
            '/opt/rocm/bin/rocprof-sys-run', Path('/tmp/out'), ['/bin/true'])
        self.assertNotIn('--rocm-domains', cmd)
        self.assertNotIn('--rocm-events', cmd)

    def test_rocm_event_defaults_are_the_six_same_run_tracks(self) -> None:
        self.assertIn('--rocm-domains', adapter.ROCPROF_SYS_FLAGS_USED)
        self.assertIn('--rocm-events', adapter.ROCPROF_SYS_FLAGS_USED)
        self.assertEqual(adapter.DEFAULT_ROCM_DOMAINS, ('kernel_dispatch',))
        self.assertEqual(adapter.DEFAULT_ROCM_EVENTS, (
            'SQ_VALU_MFMA_BUSY_CYCLES',
            'GRBM_GUI_ACTIVE',
            'SIMD_UTILIZATION',
            'GPU_UTIL',
            'TCC_EA0_RDREQ_DRAM',
            'TCC_EA0_WRREQ_DRAM',
        ))
        domains, events = capture_rocm.resolve_rocm_options(None, [])
        self.assertEqual(domains, list(adapter.DEFAULT_ROCM_DOMAINS))
        self.assertEqual(events, list(adapter.DEFAULT_ROCM_EVENTS))
        self.assertEqual(capture_rocm.resolve_rocm_options(None, None), ([], []))
        args = capture_rocm.parse_args(['--rocm-events', '--payload', '/bin/true'])
        domains, events = capture_rocm.resolve_rocm_options(
            args.rocm_domains, args.rocm_events)
        self.assertEqual(domains, list(adapter.DEFAULT_ROCM_DOMAINS))
        self.assertEqual(events, list(adapter.DEFAULT_ROCM_EVENTS))

    def test_artifact_discovery_and_validation(self) -> None:
        with tempfile.TemporaryDirectory(prefix='weaveprofiler-artifacts-') as td:
            root = Path(td)
            nested = root / '2026-11-14_12.00'
            nested.mkdir()
            trace = nested / 'capture-perfetto-trace-123.proto'
            trace.write_bytes(b'proto\x00my_test_kernel\x00')
            metadata = nested / 'capture-metadata-123.json'
            metadata.write_text(json.dumps({
                'rocprofiler-systems': {'metadata': {'info': {
                    'ROCPROFSYS_VERSION': '1.3.0'}}},
            }), encoding='utf-8')
            discovery = adapter.discover_rocprof_sys_artifacts(root)
            self.assertEqual(discovery['selected_trace'], str(trace))
            self.assertEqual(discovery['selected_metadata'], str(metadata))
            report = adapter.inspect_rocprof_sys_artifacts(
                discovery, 'my_test_kernel')
            self.assertTrue(report['valid'])
            self.assertTrue(report['kernel']['observed'])
            self.assertFalse(report['kernel']['structural_verification'])


class PersistentGuardTests(unittest.TestCase):
    def test_persistent_true_rejects_pmc(self) -> None:
        failures = capture_rocm.persistent_guard(
            {'persistent': True, 'loaded': True}, True)
        self.assertTrue(failures)
        self.assertIn('禁止挂 PMC', failures[0])

    def test_missing_or_unloaded_contract_rejects_pmc(self) -> None:
        missing = capture_rocm.persistent_guard(
            {'persistent': None, 'loaded': False}, True)
        unset = capture_rocm.persistent_guard(
            {'persistent': None, 'loaded': True}, True)
        self.assertTrue(missing)
        self.assertTrue(unset)
        self.assertIn('未提供有效 --contract', missing[0])
        self.assertIn('缺少 platform.persistent', unset[0])

    def test_explicit_false_allows_pmc_and_timeline_is_always_safe(self) -> None:
        self.assertEqual(capture_rocm.persistent_guard(
            {'persistent': False, 'loaded': True}, True), [])
        self.assertEqual(capture_rocm.persistent_guard(
            {'persistent': None, 'loaded': False}, False), [])


class ManifestIntegrationTests(unittest.TestCase):
    def _gpu_report(self):
        return {
            'amd_smi_available': True,
            'target_indices': [0],
            'memory_threshold_mib': 1024,
            'utilization_threshold_pct': 10,
            'gpus': [{'index': 0, 'memory_used_mib': 0,
                      'utilization_pct': 0, 'over_memory': False,
                      'over_utilization': False}],
            'foreign_processes': [],
            'shared_system_pids': [],
            'busy': False,
            'reasons': [],
        }

    def test_manifest_records_rocprof_sys_probes_argv_and_semantics(self) -> None:
        with tempfile.TemporaryDirectory(prefix='weaveprofiler-manifest-sys-') as td:
            root = Path(td)
            out_dir = root / 'out'
            sys_bin = write_script(root / 'rocprof-sys-run', SYS_HELP,
                                   version='rocprof-sys-run fake-version')
            contract = root / 'CONTRACT.lock.json'
            contract.write_text(json.dumps({
                'platform': {'persistent': False},
            }), encoding='utf-8')

            def fake_run(cmd, cwd, log_path, timeout, env):
                self.assertEqual(cmd[0], str(sys_bin.resolve()))
                self.assertIn('--trace', cmd)
                domain_start = cmd.index('--rocm-domains')
                event_start = cmd.index('--rocm-events')
                output_start = cmd.index('--output')
                self.assertEqual(cmd[domain_start + 1:event_start],
                                 ['kernel_dispatch'])
                self.assertEqual(cmd[event_start + 1:output_start],
                                 list(adapter.DEFAULT_ROCM_EVENTS))
                log_path.write_text('$ fake rocprof-sys\ncapture completed\n',
                                    encoding='utf-8')
                artifacts = Path(cwd) / 'rocprof_out' / '2026-11-14_12.00'
                artifacts.mkdir(parents=True)
                (artifacts / 'capture-perfetto-trace-1.proto').write_bytes(
                    b'perfetto\x00target_kernel\x00')
                (artifacts / 'capture-metadata-1.json').write_text(json.dumps({
                    'rocprofiler-systems': {'metadata': {'info': {}}},
                }), encoding='utf-8')
                return 0, '$ fake rocprof-sys\ncapture completed\n'

            argv = ['--root', str(root), '--out-dir', str(out_dir),
                    '--amd-smi', str(root / 'amd-smi'),
                    '--rocprof-sys', str(sys_bin),
                    '--kernel-name-contains', 'target_kernel',
                    '--contract', str(contract),
                    '--rocm-domains', 'kernel_dispatch',
                    '--rocm-events', *adapter.DEFAULT_ROCM_EVENTS,
                    '--payload', '/bin/true']
            with mock.patch.object(capture_rocm, 'inspect_gpus',
                                   return_value=self._gpu_report()), \
                 mock.patch.object(capture_rocm, 'run_capture', side_effect=fake_run):
                rc = capture_rocm.main(argv)
            self.assertEqual(rc, 0)
            manifest = json.loads(
                (out_dir / 'capture_manifest.json').read_text(encoding='utf-8'))
            collector = manifest['collector']
            self.assertEqual(collector['family'], adapter.ROCPROF_SYS_FAMILY)
            self.assertEqual(collector['path'], str(sys_bin.resolve()))
            self.assertEqual(collector['version_probe']['argv'],
                             [str(sys_bin.resolve()), '--version'])
            self.assertEqual(collector['argv'][0], str(sys_bin.resolve()))
            self.assertEqual(collector['argv'][-1], '/bin/true')
            self.assertEqual(shlex.split(collector['argv_shell']),
                             collector['argv'])
            self.assertEqual(collector['rocm_domains'], ['kernel_dispatch'])
            self.assertEqual(collector['rocm_events'],
                             list(adapter.DEFAULT_ROCM_EVENTS))
            self.assertFalse(collector['fallback']['used'])
            self.assertIn('--rocm-domains',
                          collector['capability_probe']['flags_checked'])
            self.assertIn('--rocm-events',
                          collector['capability_probe']['flags_checked'])
            self.assertFalse(collector['capabilities']['sampling']
                             ['periodic_pm_sampling'])
            self.assertTrue(collector['capabilities']['raw_rocm_counters'])
            self.assertEqual(collector['capabilities']['sampling']
                             ['counter_granularity'], 'per-dispatch')
            self.assertEqual(manifest['rocm_domains'], ['kernel_dispatch'])
            self.assertEqual(manifest['rocm_events'],
                             list(adapter.DEFAULT_ROCM_EVENTS))
            self.assertEqual(manifest['capture_contract']['rocm_domains'],
                             ['kernel_dispatch'])
            self.assertEqual(manifest['capture_contract']['rocm_events'],
                             list(adapter.DEFAULT_ROCM_EVENTS))
            self.assertTrue(manifest['rocm_counter_semantics']['applied'])
            self.assertFalse(manifest['rocm_counter_semantics']
                             ['periodic_pm_sampling'])
            self.assertEqual(manifest['exit_codes'],
                             {adapter.ROCPROF_SYS_FAMILY: 0})
            self.assertEqual(manifest['actual_collector'],
                             adapter.ROCPROF_SYS_FAMILY)
            self.assertEqual(manifest['target_card'], 'mi308x')
            self.assertEqual(manifest['target_arch'], 'gfx942')
            self.assertEqual(manifest['collector_argv'],
                             collector['argv'])
            self.assertEqual(manifest['verdict'], 'PASS')

    def test_manifest_records_explicit_fallback_to_rocprofv3(self) -> None:
        with tempfile.TemporaryDirectory(prefix='weaveprofiler-manifest-v3-') as td:
            root = Path(td)
            out_dir = root / 'out'
            v3 = write_script(root / 'rocprofv3', V3_HELP,
                              version='rocprofv3 fake-version')

            def fake_run(cmd, cwd, log_path, timeout, env):
                self.assertEqual(cmd[0], str(v3.resolve()))
                log_path.write_text('$ fake rocprofv3\ncapture completed\n',
                                    encoding='utf-8')
                artifact_dir = Path(cwd) / 'rocprof_out'
                artifact_dir.mkdir(parents=True, exist_ok=True)
                (artifact_dir / 'capture_results.pftrace').write_bytes(b'pftrace')
                (artifact_dir / 'capture_results.json').write_text(json.dumps({
                    'rocprofiler-sdk-tool': [{
                        'kernel_symbols': [],
                        'buffer_records': {'kernel_dispatch': []},
                    }],
                }), encoding='utf-8')
                return 0, '$ fake rocprofv3\ncapture completed\n'

            def no_sys(name):
                return None if name.startswith('rocprof-sys') else None

            argv = ['--root', str(root), '--out-dir', str(out_dir),
                    '--amd-smi', str(root / 'amd-smi'),
                    '--rocprof', str(v3),
                    '--payload', '/bin/true']
            with mock.patch.object(capture_rocm.shutil, 'which', side_effect=no_sys), \
                 mock.patch.object(capture_rocm, 'inspect_gpus',
                                   return_value=self._gpu_report()), \
                 mock.patch.object(capture_rocm, 'run_capture', side_effect=fake_run):
                rc = capture_rocm.main(argv)
            self.assertEqual(rc, 0)
            manifest = json.loads(
                (out_dir / 'capture_manifest.json').read_text(encoding='utf-8'))
            collector = manifest['collector']
            self.assertEqual(collector['family'], adapter.ROCPROFV3_FAMILY)
            self.assertTrue(collector['fallback']['used'])
            self.assertEqual(collector['fallback']['trigger'], 'not-found')
            self.assertIn('rocprof-sys 不可用',
                          collector['fallback']['reason'])
            self.assertEqual(manifest['exit_codes'],
                             {adapter.ROCPROFV3_FAMILY: 0})
            self.assertEqual(manifest['verdict'], 'PASS')

    def test_persistent_pmc_rejection_happens_before_any_gpu_action(self) -> None:
        with tempfile.TemporaryDirectory(prefix='weaveprofiler-guard-') as td:
            root = Path(td)
            out_dir = root / 'out'
            v3 = write_script(root / 'rocprofv3', V3_HELP,
                              version='rocprofv3 fake-version')
            contract = root / 'CONTRACT.lock.json'
            contract.write_text(json.dumps({
                'platform': {'persistent': True},
            }), encoding='utf-8')
            argv = ['--root', str(root), '--out-dir', str(out_dir),
                    '--collector', adapter.ROCPROFV3_FAMILY,
                    '--rocprof', str(v3), '--amd-smi', str(root / 'amd-smi'),
                    '--contract', str(contract), '--pmc', 'SQ_WAVES',
                    '--payload', '/bin/true']
            with mock.patch.object(capture_rocm, 'inspect_gpus') as inspect_mock, \
                 mock.patch.object(capture_rocm, 'run_capture') as run_mock:
                rc = capture_rocm.main(argv)
            self.assertEqual(rc, 1)
            inspect_mock.assert_not_called()
            run_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
