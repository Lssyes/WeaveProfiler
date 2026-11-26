#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Characterization tests for `agent_installer.main`, `guard_hook.status_report`, and `atrex_plugin` (B4-G2, B4-G3)."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zlib

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.agent_installer import main as installer_main  # noqa: E402
from weaveprof.atrex_plugin import apply as apply_atrex_plugin  # noqa: E402
from weaveprof.guard_hook import register_seal, status_report, supervisor_env  # noqa: E402
from weaveprof.harness import EpisodeSealer, HarnessPolicy, unregister_lock_provider  # noqa: E402
from weaveprof.workspace import digest as file_sha256  # noqa: E402


FIXED_HMAC_KEY = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"

INSTALL_WS_EXPECTED_KEYS = {
    "agent",
    "files_written",
    "launch",
    "notes",
    "ok",
    "protected",
    "session_dir",
    "session_id",
    "workspace",
}

STATUS_CLI_INSTALLED_KEYS = {
    "blockers",
    "incumbent",
    "installed",
    "live_shas",
    "matched_seal_sha256",
    "ok",
    "session_dir",
    "stop_blocks",
    "valid_pass_seals",
    "workspace",
}

GUARD_EXEC_EXPECTED_KEYS = {
    "agent_returncode",
    "bwrap",
    "completion",
    "elapsed_s",
    "installed",
    "ok",
    "reverts",
}

STATUS_REPORT_INSTALLED_KEYS = {
    "arch",
    "bench_cmd",
    "card",
    "edit_unlocked",
    "incumbent",
    "installed",
    "max_overhead",
    "next_step",
    "protected",
    "require_final_seal",
    "seals_recorded",
    "session_id",
    "stop_blockers",
    "stop_unlocked",
    "valid_pass_seals",
    "workspace",
}

STATUS_REPORT_PROTECTED_ENTRY_KEYS = {
    "baseline_sha256",
    "latest_sealed_sha256",
    "live_sha256",
    "matches_latest_seal",
    "modified_since_baseline",
    "new_probe_tokens",
    "path",
    "rel",
}

IDENTIFY_EXPECTED_KEYS = {
    "arch",
    "card",
    "kernel_name",
    "language",
    "platform",
    "probes",
    "roles",
    "source_path",
    "source_sha256",
}

VERIFY_RESULT_EXPECTED_KEYS = {
    "checks",
    "enclosing_loop",
    "failed",
    "fires",
    "ledger_path",
    "metrics",
    "seal_path",
    "seal_sha256",
    "status",
}

AGENT_REPORT_EXPECTED_KEYS = {
    "bottleneck_hint",
    "caliber",
    "causal_opportunities",
    "counterfactual_upper_bound_us",
    "failed",
    "fidelity_gates",
    "global_hw_regime",
    "hw_summary",
    "kernel_us",
    "ledger_path",
    "overhead_pct",
    "overlap_recommendations",
    "perfetto_json_gz",
    "phases",
    "preview_png",
    "refinement_advice",
    "round_diff",
    "schema",
    "seal_errors",
    "seal_path",
    "seal_sha256",
    "seal_verified",
    "stage_breakdown",
    "status",
    "structural_blockers",
    "trace_path",
    "verdict",
}

DOCTOR_EXPECTED_KEYS = {
    "gpus",
    "krex_lock",
    "llm",
    "platform",
    "python",
    "ready_for_gpu_capture",
    "ready_for_offline_tests",
    "schema",
    "tools",
}


def _write_valid_png(path: Path) -> None:
    """Write a minimal valid 800x600 PNG satisfying `guard_hook._is_valid_png`."""
    def chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return len(payload).to_bytes(4, "big") + tag + payload + crc.to_bytes(4, "big")

    width, height = 800, 600
    ihdr = width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"
    raw_rows = b"\x00" * (height * (1 + width * 3))
    compressed = zlib.compress(raw_rows, level=1)
    png_bytes = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)


def _write_synth_bench_script(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sqlite3
import struct

rdir = Path(os.environ["WEAVEPROF_ROUND_DIR"])
names = json.loads(Path(os.environ["WEAVEPROF_NAMES"]).read_text(encoding="utf-8"))
ovh_mode = os.environ.get("WP_CHAR_BENCH_MODE", "pass")

ops = ["MOV", "LDG", "FFMA", "FADD", "STG", "EXIT"]
sass = "Function : demo_kernel\\n" + "".join(f"/*{16 * i:04x}*/ {op} ;\\n" for i, op in enumerate(ops))
(rdir / "l0.sass").write_text(sass, encoding="utf-8")
(rdir / "inst.sass").write_text(sass, encoding="utf-8")
(rdir / "l0.bin").write_bytes(b"\\x01\\x02\\x03\\x04" * 8)
(rdir / "inst.bin").write_bytes(b"\\x01\\x02\\x03\\x04" * 8)

END = 1 << 31
OPEN = 1 << 30
CLOSE = 1 << 27
t0 = 1_000_000
recs = [(t0, OPEN, 0)]
t = t0 + 200
for k_str in sorted(names.get("probe_ids", {}), key=lambda x: int(x)):
    if ovh_mode == "no_events_high_ovh":
        break
    eid = int(k_str)
    recs.append((t, eid, 0))
    recs.append((t + 20000, END | eid, 0))
    t += 21000
recs.append((t + 200, CLOSE, 0))

meta = json.dumps({"gran": 0, "arch": "sm_90a", "backend": "nvidia_ptx"}).encode("utf-8")
blob = bytearray(b"FILITE02")
blob.extend(struct.pack("<6I", 1, len(recs), 1, 0, 0, len(meta)))
blob.extend(meta)
blob.extend(struct.pack("<1I", len(recs)))
for rec in recs:
    blob.extend(struct.pack("<QII", *rec))
(rdir / "capture.fi02.bin").write_bytes(blob)

db = sqlite3.connect(str(rdir / "capture.sqlite"))
db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, demangledName INTEGER)")
db.execute("INSERT INTO StringIds VALUES (1, 'demo_kernel')")
db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 1)", (t0 - 500, t + 1000))
db.commit()
db.close()

inst_ns = (
    [54000, 54000, 54000, 54000]
    if ovh_mode in ("high_ovh", "no_events_high_ovh")
    else [41000, 41000, 41000, 41000]
)
rt = {
    "kernel_name": "demo_kernel",
    "baseline_sass": "l0.sass",
    "instrumented_sass": "inst.sass",
    "baseline_spill_bytes": 0,
    "instrumented_spill_bytes": 0,
    "extra_smem_bytes": 0,
    "baseline_output": "l0.bin",
    "instrumented_output": "inst.bin",
    "dump": "capture.fi02.bin",
    "nsys_sqlite": "capture.sqlite",
    "warmup": 2,
    "timing_protocol": "ABBA",
    "baseline_ns": [40000, 40000, 40000, 40000],
    "instrumented_ns": inst_ns,
}
(rdir / "runtime.json").write_text(json.dumps(rt), encoding="utf-8")
""",
        encoding="utf-8",
    )


class _IsolatedEnvCase(unittest.TestCase):
    _ENV_KEYS = (
        "HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "MPLCONFIGDIR",
        "WEAVEPROF_SESSIONS_DIR",
        "WEAVEPROF_CACHE",
        "WEAVEPROF_LOCK_DIR",
        "WEAVEPROF_WORKDIR",
        "WEAVEPROF_SUPERVISOR_HMAC_KEY",
        "WEAVEPROF_SUPERVISOR_KEY_FILE",
        "QWEN_PROJECT_DIR",
        "QWEN_SESSION_ID",
        "WEAVEPROF_LOCK_PROVIDER",
        "WEAVEPROF_KEEP_SHADOW",
        "FI_SAMPLE",
        "WP_SAMPLE",
        "WEAVEPROF_SAMPLE",
        "DASHSCOPE_API_KEY",
        "BAILIAN_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "PYTHONDONTWRITEBYTECODE",
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls._saved_env = {k: os.environ.get(k) for k in cls._ENV_KEYS}
        cls._root = Path(tempfile.mkdtemp(prefix="wp-r11-g2g3-"))
        for key, rel in (
            ("HOME", "home"),
            ("XDG_CACHE_HOME", "xdg_cache"),
            ("XDG_CONFIG_HOME", "xdg_config"),
            ("MPLCONFIGDIR", "mplconfig"),
            ("WEAVEPROF_SESSIONS_DIR", "sessions"),
            ("WEAVEPROF_CACHE", "wp_cache"),
            ("WEAVEPROF_LOCK_DIR", "wp_locks"),
            ("WEAVEPROF_WORKDIR", "wp_runs"),
        ):
            target = cls._root / rel
            target.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(target)
        os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = FIXED_HMAC_KEY
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        for clear_key in (
            "WEAVEPROF_SUPERVISOR_KEY_FILE",
            "QWEN_PROJECT_DIR",
            "QWEN_SESSION_ID",
            "WEAVEPROF_LOCK_PROVIDER",
            "WEAVEPROF_KEEP_SHADOW",
            "FI_SAMPLE",
            "WP_SAMPLE",
            "WEAVEPROF_SAMPLE",
            "DASHSCOPE_API_KEY",
            "BAILIAN_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
        ):
            os.environ.pop(clear_key, None)

    @classmethod
    def tearDownClass(cls) -> None:
        for k, v in cls._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(cls._root, ignore_errors=True)

    def _make_case_dir(self, label: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=f"{label}-", dir=str(self._root)))

    def _run_installer_cli(self, argv: list[str]) -> tuple[int, str, str]:
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            rc = installer_main(argv)
        return rc, out_buf.getvalue(), err_buf.getvalue()

    def _seal_workspace_kernel(self, session_dir: Path, kernel_path: Path, run_dir: Path, *, kernel_us: float) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        fb_path = run_dir / "feedback.json"
        ver_path = run_dir / "verify.json"
        png_path = run_dir / "overview.png"
        fb_path.write_text('{"status": "PASS"}\n', encoding="utf-8")
        ver_path.write_text('{"status": "PASS"}\n', encoding="utf-8")
        _write_valid_png(png_path)
        policy = HarnessPolicy(max_overhead_limit=0.10)
        result = {
            "status": "PASS",
            "checks": {"assembly": True, "output": True, "events": True, "overhead": True},
            "failed": [],
            "metrics": {"overhead_ratio": 0.025},
        }
        with supervisor_env(session_dir):
            seal_path, _doc = EpisodeSealer.seal_round(
                round_dir=run_dir,
                round_number=1,
                source_path=kernel_path,
                source_sha256=file_sha256(kernel_path),
                probe_path=None,
                instrumented_path=None,
                names_path=None,
                feedback_path=fb_path,
                verify_path=ver_path,
                trace_path=png_path,
                failed_trace_path=None,
                result=result,
                policy=policy,
                limit=0.10,
                bench_cmd=["python", "bench.py"],
                fsm_state="PROMOTED",
                region_lock={"gpu_id": "h20_0", "provider": "local_file"},
                extra_artifacts={"overview_png": png_path},
            )
        reg = register_seal(session_dir, seal_path, kernel_us=kernel_us, tool="unit_test")
        self.assertIs(reg["registered"], True, reg)
        self.assertIs(reg["covers_protected"], True, reg)
        return seal_path


class AgentInstallerMainAndStatusReportTests(_IsolatedEnvCase):
    def test_installer_main_workspace_and_global_modes(self) -> None:
        cdir = self._make_case_dir("inst")
        ws = cdir / "ws"
        ws.mkdir()
        sroot = cdir / "sessions"
        kernel = ws / "kernel.cu"
        kernel.write_text("__global__ void k(float* x) { x[0] = 2.0f; }\n", encoding="utf-8")

        rc, out_txt, err_txt = self._run_installer_cli([
            "install",
            "--workspace",
            str(ws),
            "--protect",
            "kernel.cu",
            "--agent",
            "all",
            "--bench",
            "python bench.py",
            "--card",
            "h20",
            "--arch",
            "sm_90a",
            "--max-overhead",
            "0.08",
            "--sessions-root",
            str(sroot),
        ])
        self.assertEqual(rc, 0, err_txt)
        self.assertEqual(err_txt, "")
        out = json.loads(out_txt)
        self.assertEqual(set(out.keys()), INSTALL_WS_EXPECTED_KEYS)
        self.assertIs(out["ok"], True)
        self.assertEqual(out["agent"], "all")
        self.assertEqual(len(out["protected"]), 1)
        self.assertEqual(out["protected"][0]["rel"], "kernel.cu")

        sd = Path(out["session_dir"])
        sess_doc = json.loads((sd / "session.json").read_text(encoding="utf-8"))
        self.assertEqual(sess_doc["card"], "h20")
        self.assertEqual(sess_doc["arch"], "sm_90a")
        self.assertEqual(sess_doc["max_overhead"], 0.08)
        self.assertEqual(sess_doc["bench_cmd"], ["python", "bench.py"])
        self.assertIs(sess_doc["require_final_seal"], True)
        self.assertIs(sess_doc["png_only"], True)

        key_before = (sd / ".supervisor.key").read_bytes()
        rc_reset, out_reset_txt, _ = self._run_installer_cli([
            "install",
            "--workspace",
            str(ws),
            "--protect",
            "kernel.cu",
            "--no-require-final-seal",
            "--allow-gz",
            "--reset",
            "--sessions-root",
            str(sroot),
        ])
        self.assertEqual(rc_reset, 0)
        out_reset = json.loads(out_reset_txt)
        self.assertIs(out_reset["ok"], True)
        sess_reset = json.loads((sd / "session.json").read_text(encoding="utf-8"))
        self.assertIs(sess_reset["require_final_seal"], False)
        self.assertIs(sess_reset["png_only"], False)
        self.assertNotEqual((sd / ".supervisor.key").read_bytes(), key_before)

        custom_home = cdir / "isolated_home"
        custom_home.mkdir()
        rc_glob, out_glob_txt, err_glob = self._run_installer_cli([
            "install",
            "--global",
            "--agent",
            "all",
            "--home",
            str(custom_home),
            "--sessions-root",
            str(sroot),
        ])
        self.assertEqual(rc_glob, 0, err_glob)
        out_glob = json.loads(out_glob_txt)
        self.assertEqual(set(out_glob.keys()), {"ok", "mode", "agent", "files_written"})
        self.assertIs(out_glob["ok"], True)
        self.assertEqual(out_glob["mode"], "global")
        self.assertEqual(out_glob["agent"], "all")
        self.assertEqual(len(out_glob["files_written"]), 2)
        qwen_cfg = json.loads((custom_home / ".qwen" / "settings.json").read_text(encoding="utf-8"))
        claude_cfg = json.loads((custom_home / ".claude" / "settings.json").read_text(encoding="utf-8"))
        self.assertIs(qwen_cfg["disableAllHooks"], False)
        self.assertIn("PreToolUse", claude_cfg["hooks"])

        rc_bad_glob, out_bad_glob, err_bad_glob = self._run_installer_cli([
            "install",
            "--global",
            "--agent",
            "codex",
            "--home",
            str(custom_home),
        ])
        self.assertEqual(rc_bad_glob, 2)
        self.assertEqual(out_bad_glob, "")
        self.assertEqual(
            err_bad_glob,
            "weaveprof-agent install error: ValueError: global install supports agent in: qwen, claude, all\n",
        )

    def test_installer_main_status_guard_exec_and_guard_hook_status_report(self) -> None:
        cdir = self._make_case_dir("guard")
        sroot = cdir / "sessions"
        unprov_ws = cdir / "unprovisioned_ws"
        unprov_ws.mkdir()

        rc_unprov, out_unprov_txt, _ = self._run_installer_cli([
            "status",
            "--workspace",
            str(unprov_ws),
            "--sessions-root",
            str(sroot),
        ])
        self.assertEqual(rc_unprov, 1)
        out_unprov = json.loads(out_unprov_txt)
        self.assertEqual(set(out_unprov.keys()), {"ok", "installed", "session_dir", "blockers"})
        self.assertIs(out_unprov["ok"], False)
        self.assertIs(out_unprov["installed"], False)
        self.assertTrue(out_unprov["blockers"][0].startswith("SESSION_MISSING:"))

        sr_missing = status_report(cdir / "missing_session_id")
        self.assertEqual(set(sr_missing.keys()), {"installed", "session_id", "hint"})
        self.assertIs(sr_missing["installed"], False)
        self.assertEqual(sr_missing["session_id"], "missing_session_id")

        ws = cdir / "ws"
        ws.mkdir()
        kernel = ws / "kernel.cu"
        baseline_src = "__global__ void k(float* x) { x[0] = 3.0f; }\n"
        kernel.write_text(baseline_src, encoding="utf-8")

        rc_ge_unsealed, out_ge_unsealed_txt, err_ge_unsealed = self._run_installer_cli([
            "guard-exec",
            "--workspace",
            str(ws),
            "--protect",
            "kernel.cu",
            "--card",
            "h20",
            "--arch",
            "sm_90a",
            "--bench",
            "python bench.py",
            "--sessions-root",
            str(sroot),
            "--bwrap",
            "never",
            "--",
            sys.executable,
            "-c",
            "import sys; sys.exit(0)",
        ])
        self.assertEqual(rc_ge_unsealed, 1, err_ge_unsealed)
        ge_unsealed = json.loads(out_ge_unsealed_txt)
        self.assertEqual(set(ge_unsealed.keys()), GUARD_EXEC_EXPECTED_KEYS)
        self.assertIs(ge_unsealed["ok"], False)
        self.assertEqual(ge_unsealed["agent_returncode"], 0)
        self.assertIs(ge_unsealed["bwrap"], False)
        self.assertIs(ge_unsealed["installed"], True)
        self.assertEqual(ge_unsealed["reverts"], [])
        self.assertIs(ge_unsealed["completion"]["ok"], False)
        self.assertEqual(ge_unsealed["completion"]["valid_pass_seals"], 0)

        sd = Path(ge_unsealed["completion"]["session_dir"])
        sr_unsealed = status_report(sd)
        self.assertEqual(set(sr_unsealed.keys()), STATUS_REPORT_INSTALLED_KEYS)
        self.assertEqual(set(sr_unsealed["protected"][0].keys()), STATUS_REPORT_PROTECTED_ENTRY_KEYS)
        self.assertIs(sr_unsealed["installed"], True)
        self.assertEqual(sr_unsealed["seals_recorded"], 0)
        self.assertEqual(sr_unsealed["valid_pass_seals"], 0)
        self.assertIs(sr_unsealed["edit_unlocked"], False)
        self.assertIs(sr_unsealed["stop_unlocked"], False)
        self.assertIsNone(sr_unsealed["incumbent"])
        self.assertEqual(sr_unsealed["card"], "h20")
        self.assertEqual(sr_unsealed["arch"], "sm_90a")
        self.assertIs(sr_unsealed["protected"][0]["modified_since_baseline"], False)
        self.assertIs(sr_unsealed["protected"][0]["matches_latest_seal"], False)
        self.assertEqual(sr_unsealed["protected"][0]["new_probe_tokens"], [])

        self._seal_workspace_kernel(sd, kernel, cdir / "seal_run_1", kernel_us=412.5)

        rc_st_ok, out_st_ok_txt, _ = self._run_installer_cli([
            "status",
            "--workspace",
            str(ws),
            "--sessions-root",
            str(sroot),
        ])
        self.assertEqual(rc_st_ok, 0)
        st_ok = json.loads(out_st_ok_txt)
        self.assertEqual(set(st_ok.keys()), STATUS_CLI_INSTALLED_KEYS)
        self.assertIs(st_ok["ok"], True)
        self.assertIs(st_ok["installed"], True)
        self.assertEqual(st_ok["valid_pass_seals"], 1)
        self.assertEqual(st_ok["blockers"], [])
        self.assertEqual(st_ok["incumbent"]["kernel_us"], 412.5)

        sr_sealed = status_report(sd)
        self.assertEqual(sr_sealed["seals_recorded"], 1)
        self.assertEqual(sr_sealed["valid_pass_seals"], 1)
        self.assertIs(sr_sealed["edit_unlocked"], True)
        self.assertIs(sr_sealed["stop_unlocked"], True)
        self.assertEqual(sr_sealed["stop_blockers"], [])
        self.assertEqual(sr_sealed["incumbent"]["kernel_us"], 412.5)
        self.assertIs(sr_sealed["protected"][0]["matches_latest_seal"], True)
        self.assertTrue(sr_sealed["next_step"].startswith("Edits unlocked."))

        proc_ge_ok = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "weaveprof" / "agent_installer.py"),
                "guard-exec",
                "--workspace",
                str(ws),
                "--sessions-root",
                str(sroot),
                "--bwrap",
                "never",
                "--",
                sys.executable,
                "-c",
                "import sys; sys.exit(0)",
            ],
            text=True,
            capture_output=True,
            cwd=str(cdir),
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            timeout=60,
            check=False,
        )
        self.assertEqual(proc_ge_ok.returncode, 0, proc_ge_ok.stderr)
        ge_ok = json.loads(proc_ge_ok.stdout)
        self.assertEqual(set(ge_ok.keys()), GUARD_EXEC_EXPECTED_KEYS)
        self.assertIs(ge_ok["ok"], True)
        self.assertEqual(ge_ok["agent_returncode"], 0)
        self.assertIs(ge_ok["bwrap"], False)
        self.assertIs(ge_ok["installed"], False)
        self.assertEqual(ge_ok["reverts"], [])
        self.assertIs(ge_ok["completion"]["ok"], True)

        rc_ge_nz, out_ge_nz_txt, _ = self._run_installer_cli([
            "guard-exec",
            "--workspace",
            str(ws),
            "--sessions-root",
            str(sroot),
            "--bwrap",
            "never",
            "--",
            sys.executable,
            "-c",
            "import sys; sys.exit(5)",
        ])
        self.assertEqual(rc_ge_nz, 1)
        ge_nz = json.loads(out_ge_nz_txt)
        self.assertIs(ge_nz["ok"], False)
        self.assertEqual(ge_nz["agent_returncode"], 5)
        self.assertIs(ge_nz["completion"]["ok"], True)

        kernel.write_text(
            "__global__ void k(float* x) { WP_OPEN(0); x[0] = 9.0f; }\n",
            encoding="utf-8",
        )
        sr_polluted = status_report(sd)
        self.assertIs(sr_polluted["edit_unlocked"], True)
        self.assertIs(sr_polluted["stop_unlocked"], False)
        self.assertIs(sr_polluted["protected"][0]["modified_since_baseline"], True)
        self.assertIs(sr_polluted["protected"][0]["matches_latest_seal"], False)
        self.assertEqual(sr_polluted["protected"][0]["new_probe_tokens"], ["WP_OPEN"])
        self.assertEqual(len(sr_polluted["stop_blockers"]), 2)
        self.assertTrue(sr_polluted["stop_blockers"][0].startswith("PROBE_POLLUTION:"))
        self.assertTrue(sr_polluted["stop_blockers"][1].startswith("UNSEALED_FINAL_KERNEL:"))


class AtrexPluginMethodsCharacterizationTests(_IsolatedEnvCase):
    def test_all_ten_delegated_methods_and_registered_tool_handler(self) -> None:
        cdir = self._make_case_dir("atrex")
        provided: dict[str, object] = {}
        registered_tools: dict[str, tuple[str, object]] = {}

        class HostContextStub:
            def provide(self, key: str, service: object) -> None:
                provided[key] = service

            def register_tool(self, *, name: str, description: str, handler: object) -> None:
                registered_tools[name] = (description, handler)

        @contextlib.contextmanager
        def custom_lock_factory(gpu_id: str, *, timeout_s: float, label: str, cpu_cores: object):
            yield {"gpu_id": gpu_id, "label": label, "wait_ms": 0.0}

        try:
            host_ctx = HostContextStub()
            svc = apply_atrex_plugin(
                host_ctx,
                {
                    "default_limit": 0.10,
                    "model": "qwen4-max",
                    "card": "h20",
                    "max_rounds": 1,
                    "lock_provider": "r11_test_provider",
                    "lock_provider_factory": custom_lock_factory,
                },
            )
            self.assertIs(provided["intra_kernel_verifier_profiler"], svc)
            self.assertEqual(set(registered_tools.keys()), {"weaveprof_profile_for_agent"})

            # 1) critical_region (custom provider configured on AutoProfService + explicit local_file)
            with svc.critical_region("h20_0", label="atrex_lease") as lease_custom:
                self.assertEqual(lease_custom.gpu_id, "h20_0")
                self.assertEqual(lease_custom.provider, "r11_test_provider")
                self.assertEqual(lease_custom.label, "atrex_lease")
            with svc.critical_region("h20_1", label="local_lease", lock_provider="local_file") as lease_local:
                self.assertEqual(lease_local.gpu_id, "h20_1")
                self.assertEqual(lease_local.provider, "local_file")
                self.assertEqual(lease_local.label, "local_lease")

            # Reset lock_provider to default local_file for downstream verify/profile_for_agent
            svc_local = apply_atrex_plugin({}, {"default_limit": 0.10, "model": "qwen4-max", "card": "h20", "max_rounds": 1})

            # 2) doctor(check_gpu=False)
            doc_out = svc_local.doctor(check_gpu=False)
            self.assertEqual(set(doc_out.keys()), DOCTOR_EXPECTED_KEYS)
            self.assertEqual(doc_out["schema"], "weaveprof.doctor/v1")
            self.assertEqual(doc_out["gpus"], {"checked": False, "nvidia": None, "amd": None})
            self.assertIs(doc_out["ready_for_offline_tests"], True)
            self.assertIs(doc_out["ready_for_gpu_capture"], False)
            self.assertEqual(doc_out["krex_lock"]["active_provider"], "local_file")
            self.assertIs(doc_out["krex_lock"]["lock_dir_writable"], True)
            self.assertIs(doc_out["llm"]["api_key_detected"], False)

            # 3) plan_context
            pc = svc_local.plan_context(
                "__global__ void demo_kernel(float* x) { x[threadIdx.x] = 1.0f; }\n",
                arch="sm_90a",
                block_threads=256,
                l0_regs=64,
                l0_smem_bytes=16384,
            )
            self.assertEqual(
                set(pc.keys()),
                {
                    "arch",
                    "compile_defines",
                    "context_mode",
                    "fallback_arms",
                    "has_grid_sync",
                    "has_role_switch_macro",
                    "has_setmaxnreg",
                    "placement_search_space",
                    "reasons",
                    "recommended_granularity",
                    "reg_budget_per_thread",
                    "reg_headroom",
                    "smem_free_bytes",
                },
            )
            self.assertEqual(pc["arch"], "sm_90a")
            self.assertEqual(pc["context_mode"], "WP_CTX_REG")
            self.assertEqual(pc["compile_defines"], ["-DWP_CTX_REG"])
            self.assertEqual(pc["reg_headroom"], 191)
            self.assertEqual(pc["smem_free_bytes"], 216064)

            # 4) analyze_causality
            perf_doc = {
                "otherData": {"arch": "sm_103a", "gpu": "B300", "alignment": {"kernel_duration_us": 100.0}},
                "traceEvents": (
                    [{"ph": "C", "name": "dram_read_bw", "ts": float(i * 10), "args": {"pct": 85.0}} for i in range(10)]
                    + [{"ph": "C", "name": "tensor_active", "ts": float(i * 10), "args": {"pct": 4.0}} for i in range(10)]
                    + [{"ph": "C", "name": "sm_active", "ts": float(i * 10), "args": {"pct": 99.0}} for i in range(10)]
                    + [
                        {
                            "ph": "X",
                            "pid": 100001,
                            "tid": 0,
                            "ts": 0.0,
                            "dur": 98.0,
                            "name": "Rec_GDN",
                            "args": {"cta": c, "sm": c, "semantic_id": "compute"},
                        }
                        for c in range(16)
                    ]
                ),
            }
            ac = svc_local.analyze_causality(perf_doc, arch="sm_103a", total_sms=140)
            self.assertEqual(ac["schema"], "weaveprof.causal-report/v1")
            self.assertEqual(ac["verdict"], "CAUSE_EXPLAINED")
            self.assertEqual(ac["global_hw_regime"], "DRAM_READ_BOUND")
            self.assertEqual(ac["kernel_us"], 100.0)
            self.assertEqual(ac["total_sms"], 140)

            # 5) build_agent_report
            prev_causal = {"kernel_us": 200.0, "verdict": "CAUSE_EXPLAINED", "global_hw_regime": "DRAM_READ_BOUND"}
            built = svc_local.build_agent_report(
                {
                    "status": "PASS",
                    "checks": {"assembly": True, "output": True, "events": True, "overhead": True},
                    "metrics": {"overhead_ratio": 0.045},
                },
                causal_input=perf_doc,
                prev_causal_report=prev_causal,
            )
            self.assertEqual(set(built.keys()), AGENT_REPORT_EXPECTED_KEYS)
            self.assertEqual(built["schema"], "weaveprof.agent-report/v1")
            self.assertEqual(built["status"], "PASS")
            self.assertEqual(built["overhead_pct"], 4.5)
            self.assertEqual(built["kernel_us"], 100.0)
            self.assertEqual(built["global_hw_regime"], "DRAM_READ_BOUND")
            self.assertIs(built["round_diff"]["improved"], True)
            self.assertEqual(built["round_diff"]["latency_delta_us"], -100.0)

            # 6) identify (offline_heuristic=True)
            proj = cdir / "proj"
            proj.mkdir()
            src = proj / "kernel.cu"
            old_src_text = (
                "__global__ void demo_kernel(float* out, int n) {\n"
                "    for (int i = 0; i < n; ++i) {\n"
                "        float v = out[i];\n"
                "        v = v * 2.0f + 1.0f;\n"
                "        out[i] = v;\n"
                "    }\n"
                "}\n"
            )
            src.write_text(old_src_text, encoding="utf-8")
            spec = svc_local.identify(src, offline_heuristic=True)
            self.assertEqual(set(spec.keys()), IDENTIFY_EXPECTED_KEYS)
            self.assertEqual(spec["card"], "h20")
            self.assertEqual(spec["arch"], "sm_90a")
            self.assertEqual(spec["platform"], "nv")
            self.assertEqual(spec["language"], "cuda_cpp")
            self.assertEqual(spec["kernel_name"], "demo_kernel")
            self.assertEqual(len(spec["probes"]), 1)
            self.assertEqual(spec["probes"][0]["id"], "compute")
            self.assertEqual(spec["probes"][0]["begin"]["line"], 5)

            # 7) remap_probe_0token (clean shift + broken anchor)
            shifted_src_text = "// inserted header\n// second line\n" + old_src_text
            remapped, stats_ok = svc_local.remap_probe_0token(spec, old_src_text, shifted_src_text)
            self.assertIsNotNone(remapped)
            self.assertEqual(
                set(stats_ok.keys()),
                {"reused_zero_token", "probes_total", "endpoints_shifted", "failed_probes", "new_source_sha256"},
            )
            self.assertIs(stats_ok["reused_zero_token"], True)
            self.assertEqual(stats_ok["probes_total"], 1)
            self.assertEqual(stats_ok["endpoints_shifted"], 2)
            self.assertEqual(stats_ok["failed_probes"], [])
            self.assertEqual(remapped["probes"][0]["begin"]["line"], 7)
            self.assertEqual(remapped["probes"][0]["end"]["line"], 7)

            remapped_fail, stats_fail = svc_local.remap_probe_0token(
                spec, old_src_text, "__global__ void completely_rewritten() {}\n"
            )
            self.assertIsNone(remapped_fail)
            self.assertIs(stats_fail["reused_zero_token"], False)
            self.assertEqual(stats_fail["failed_probes"], ["compute"])

            # 8) verify + 9) verify_seal
            bench_py = cdir / "bench_synth.py"
            _write_synth_bench_script(bench_py)
            ver_out_trace = cdir / "verify_out" / "trace.perfetto-trace"
            vres = svc_local.verify(
                spec,
                source=src,
                bench=[sys.executable, "-B", str(bench_py)],
                out=ver_out_trace,
            )
            self.assertEqual(set(vres.keys()), VERIFY_RESULT_EXPECTED_KEYS)
            self.assertEqual(vres["status"], "PASS")
            self.assertEqual(
                vres["checks"],
                {"assembly": True, "output": True, "event": True, "events": True, "overhead": True},
            )
            self.assertEqual(vres["failed"], [])
            self.assertEqual(vres["fires"], {"compute": 1})

            vs_res = svc_local.verify_seal(vres["seal_path"], require_pass=True, max_overhead_limit=0.10)
            self.assertEqual(set(vs_res.keys()), {"ok", "errors", "seal"})
            self.assertIs(vs_res["ok"], True)
            self.assertEqual(vs_res["errors"], [])
            self.assertEqual(vs_res["seal"]["status"], "PASS")

            # Tamper a copy of seal.json to pin fail-closed verification via AutoProfService.verify_seal
            bad_seal = cdir / "tampered.seal.json"
            bad_doc = json.loads(Path(vres["seal_path"]).read_text(encoding="utf-8"))
            bad_doc["seal_hmac"] = "0" * 64
            bad_seal.write_text(json.dumps(bad_doc), encoding="utf-8")
            vs_bad = svc_local.verify_seal(bad_seal, require_pass=True)
            self.assertIs(vs_bad["ok"], False)
            self.assertIn("Seal supervisor HMAC mismatch (possible forgery)", vs_bad["errors"])

            # 10) profile_for_agent (direct method + registered `weaveprof_profile_for_agent` tool handler)
            pfa_dir = cdir / "pfa_pass"
            pfa_dir.mkdir()
            pfa_pass = svc_local.profile_for_agent(
                src,
                bench=[sys.executable, "-B", str(bench_py)],
                run_dir=str(pfa_dir),
                out=str(pfa_dir / "trace.perfetto-trace"),
            )
            self.assertEqual(set(pfa_pass.keys()), AGENT_REPORT_EXPECTED_KEYS)
            self.assertEqual(pfa_pass["schema"], "weaveprof.agent-report/v1")
            self.assertEqual(pfa_pass["status"], "PASS")
            self.assertIs(pfa_pass["seal_verified"], True)
            self.assertEqual(pfa_pass["overhead_pct"], 2.5)
            self.assertEqual(pfa_pass["failed"], [])

            _desc, tool_handler = registered_tools["weaveprof_profile_for_agent"]
            pfa_fail_dir = cdir / "pfa_fail"
            pfa_fail_dir.mkdir()
            prev_mode = os.environ.get("WP_CHAR_BENCH_MODE")
            os.environ["WP_CHAR_BENCH_MODE"] = "no_events_high_ovh"
            try:
                pfa_fail = tool_handler(  # type: ignore[misc]
                    src,
                    bench=[sys.executable, "-B", str(bench_py)],
                    run_dir=str(pfa_fail_dir),
                    out=str(pfa_fail_dir / "trace.perfetto-trace"),
                )
            finally:
                if prev_mode is None:
                    os.environ.pop("WP_CHAR_BENCH_MODE", None)
                else:
                    os.environ["WP_CHAR_BENCH_MODE"] = prev_mode
            self.assertEqual(pfa_fail["schema"], "weaveprof.agent-report/v1")
            self.assertEqual(pfa_fail["status"], "FAIL")
            self.assertIs(pfa_fail["seal_verified"], True)
            self.assertEqual(pfa_fail["overhead_pct"], 35.0)
            self.assertEqual(pfa_fail["failed"], ["pairing", "overhead"])
        finally:
            unregister_lock_provider("r11_test_provider")


if __name__ == "__main__":
    unittest.main()
