#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Characterization tests for `backend.assess` instruction-order comparison (`lcs_ratio`, `order_change_ratio`, `checks['assembly']`)."""
from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

# ---------------------------------------------------------------------------
# In-process zero-external-network socket guard (installed BEFORE importing weaveprof)
# ---------------------------------------------------------------------------
_ORIG_SOCKET_CONNECT = socket.socket.connect
_ORIG_SOCKET_CONNECT_EX = socket.socket.connect_ex
_ORIG_CREATE_CONNECTION = socket.create_connection
_ORIG_GETADDRINFO = socket.getaddrinfo
_NON_LOOPBACK_ATTEMPTS: list[str] = []


def _is_loopback_host(host: object) -> bool:
    return isinstance(host, str) and host.strip() in ("127.0.0.1", "::1", "localhost")


def _is_ip_literal(host: object) -> bool:
    if not isinstance(host, str):
        return False
    try:
        ipaddress.ip_address(host.strip())
        return True
    except ValueError:
        return False


def _guarded_connect(self: socket.socket, address: object) -> object:
    if isinstance(address, tuple) and address and not _is_loopback_host(address[0]):
        _NON_LOOPBACK_ATTEMPTS.append(f"connect:{address!r}")
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _ORIG_SOCKET_CONNECT(self, address)


def _guarded_connect_ex(self: socket.socket, address: object) -> int:
    if isinstance(address, tuple) and address and not _is_loopback_host(address[0]):
        _NON_LOOPBACK_ATTEMPTS.append(f"connect_ex:{address!r}")
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _ORIG_SOCKET_CONNECT_EX(self, address)


def _guarded_create_connection(address: tuple[str, int], *args: object, **kwargs: object) -> socket.socket:
    host = address[0] if isinstance(address, tuple) and address else ""
    if not _is_loopback_host(host):
        _NON_LOOPBACK_ATTEMPTS.append(f"create_connection:{address!r}")
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _ORIG_CREATE_CONNECTION(address, *args, **kwargs)


def _guarded_getaddrinfo(host: object, port: object, *args: object, **kwargs: object) -> list[tuple]:
    if not (_is_loopback_host(host) or _is_ip_literal(host)):
        _NON_LOOPBACK_ATTEMPTS.append(f"getaddrinfo:{host!r}:{port!r}")
        raise OSError(f"blocked by test guard: DNS lookup for {host!r}")
    return _ORIG_GETADDRINFO(host, port, *args, **kwargs)


socket.socket.connect = _guarded_connect  # type: ignore[assignment]
socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[assignment]
socket.create_connection = _guarded_create_connection  # type: ignore[assignment]
socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.backend import assess  # noqa: E402

FIXED_HMAC_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


def _format_nvidia_sass(kernel: str, ops: list[str]) -> str:
    return f"Function : {kernel}\n" + "".join(f"/*{16 * idx:04x}*/ {op} ;\n" for idx, op in enumerate(ops))


def _format_amd_gcn(kernel: str, ops: list[str]) -> str:
    body = "\n".join(f"    {op}" for op in ops)
    return f"    .text\n    .globl {kernel}\n{kernel}:\n{body}\n"


def _seed_round_dir(
    directory: Path,
    *,
    kernel: str,
    base_asm: str,
    inst_asm: str,
    amd: bool = False,
) -> tuple[dict, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    ext = ".s" if amd else ".sass"
    (directory / f"l0{ext}").write_text(base_asm, encoding="utf-8")
    (directory / f"inst{ext}").write_text(inst_asm, encoding="utf-8")
    (directory / "l0.bin").write_bytes(b"\x11\x22\x33\x44" * 8)
    (directory / "inst.bin").write_bytes(b"\x11\x22\x33\x44" * 8)

    # Valid 2-probe x 4-iteration FILITE02 dump so pairing/output/overhead gates pass cleanly
    tag_open, tag_end, tag_close = 0x40000000, 0x80000000, 0x08000000
    t0 = 1_000_000
    recs = [(t0, tag_open, 0)]
    tick = t0 + 200
    for _ in range(4):
        for eid in (1, 3):
            recs.append((tick, eid, 0))
            recs.append((tick + 15_000, tag_end | eid, 0))
            tick += 15_200
    recs.append((tick + 300, tag_close, 0))

    meta = json.dumps(
        {
            "gran": 0,
            "arch": "gfx942" if amd else "sm_90a",
            "backend": "amd_hip" if amd else "nvidia_ptx",
        }
    ).encode("utf-8")
    blob = bytearray(b"FILITE02")
    blob.extend(struct.pack("<6I", 1, len(recs), 1, 0, 0, len(meta)))
    blob.extend(meta)
    blob.extend(struct.pack("<1I", len(recs)))
    for r in recs:
        blob.extend(struct.pack("<QII", *r))
    (directory / "capture.fi02.bin").write_bytes(blob)
    (directory / "names.json").write_text(
        json.dumps(
            {
                "events": {"1": "load", "3": "mma"},
                "probe_ids": {"1": "load", "3": "mma"},
                "roles": {"0": "worker"},
            }
        ),
        encoding="utf-8",
    )

    runtime = {
        "kernel_name": kernel,
        "platform": "amd" if amd else "nvidia",
        "baseline_sass": f"l0{ext}",
        "instrumented_sass": f"inst{ext}",
        "baseline_spill_bytes": 0,
        "instrumented_spill_bytes": 0,
        "extra_smem_bytes": 0,
        "baseline_output": "l0.bin",
        "instrumented_output": "inst.bin",
        "warmup": 2,
        "baseline_ns": [50_000, 50_000, 50_000, 50_000],
        "instrumented_ns": [50_500, 50_500, 50_500, 50_500],
    }
    (directory / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
    spec = {
        "kernel_name": kernel,
        "platform": "amd" if amd else "nvidia",
        "card": "mi308x" if amd else "h20",
        "probes": [{"id": "load"}, {"id": "mma"}],
    }
    return spec, runtime


class WpProfileAsmOrderCharacterizationTests(unittest.TestCase):
    _ENV_SAVE_KEYS = (
        "HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "MPLCONFIGDIR",
        "WEAVEPROF_SESSIONS_DIR",
        "WEAVEPROF_CACHE",
        "WEAVEPROF_LOCK_DIR",
        "WEAVEPROF_WORKDIR",
        "TMPDIR",
        "WEAVEPROF_SUPERVISOR_HMAC_KEY",
        "WEAVEPROF_SUPERVISOR_KEY_FILE",
        "QWEN_PROJECT_DIR",
        "QWEN_SESSION_ID",
        "WEAVEPROF_LOCK_PROVIDER",
        "WEAVEPROF_KEEP_SHADOW",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "PYTHONDONTWRITEBYTECODE",
    )

    @classmethod
    def setUpClass(cls) -> None:
        _NON_LOOPBACK_ATTEMPTS.clear()
        cls._saved_cwd = os.getcwd()
        cls._saved_tempdir = tempfile.tempdir
        cls._saved_env = {k: os.environ.get(k) for k in cls._ENV_SAVE_KEYS}
        cls._iso_root = Path(tempfile.mkdtemp(prefix="wp_r14_asm_iso_")).resolve()
        for sub in ("home", "xdg_cache", "xdg_config", "mpl", "sessions", "cache", "locks", "workdir", "tmp", "cwd"):
            (cls._iso_root / sub).mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = str(cls._iso_root / "home")
        os.environ["XDG_CACHE_HOME"] = str(cls._iso_root / "xdg_cache")
        os.environ["XDG_CONFIG_HOME"] = str(cls._iso_root / "xdg_config")
        os.environ["MPLCONFIGDIR"] = str(cls._iso_root / "mpl")
        os.environ["WEAVEPROF_SESSIONS_DIR"] = str(cls._iso_root / "sessions")
        os.environ["WEAVEPROF_CACHE"] = str(cls._iso_root / "cache")
        os.environ["WEAVEPROF_LOCK_DIR"] = str(cls._iso_root / "locks")
        os.environ["WEAVEPROF_WORKDIR"] = str(cls._iso_root / "workdir")
        os.environ["TMPDIR"] = str(cls._iso_root / "tmp")
        os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = FIXED_HMAC_KEY
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        for k in (
            "WEAVEPROF_SUPERVISOR_KEY_FILE",
            "QWEN_PROJECT_DIR",
            "QWEN_SESSION_ID",
            "WEAVEPROF_LOCK_PROVIDER",
            "WEAVEPROF_KEEP_SHADOW",
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        ):
            os.environ.pop(k, None)
        tempfile.tempdir = str(cls._iso_root / "tmp")
        os.chdir(cls._iso_root / "cwd")

    @classmethod
    def tearDownClass(cls) -> None:
        os.chdir(cls._saved_cwd)
        tempfile.tempdir = cls._saved_tempdir
        for k, val in cls._saved_env.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
        shutil.rmtree(cls._iso_root, ignore_errors=True)
        if _NON_LOOPBACK_ATTEMPTS:
            raise AssertionError(f"Unexpected non-loopback network attempts: {_NON_LOOPBACK_ATTEMPTS}")

    def _new_case_dir(self, tag: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=f"{tag}_", dir=str(self._iso_root / "tmp"))).resolve()

    def test_asm_adjacent_swap_noncore_core_and_pipeline_interleave(self) -> None:
        """Group 1: swapping two adjacent instructions (`A B C D` vs `B A C D`) yields `lcs_ratio=0.75, order_change_ratio=0.25`."""
        # 1a) 4 non-core ops: baseline = [MOV, IADD3, IMAD, EXIT], inst = [IADD3, MOV, IMAD, EXIT]
        # Hand calculation:
        #   lcs = 3, len = 4, multiset intersection mbag = 4, core_seqs = [[], []]
        #   lcs_ratio = 3 / 4 = 0.75; assembly_change_rate = 1 - 0.75 = 0.25
        #   order_change_ratio = 1.0 - 3 / 4 = 0.25
        #   core_retention_ratio = 0.75; core_multiset_ratio = 0.75
        #   non-core gate: ratio >= 0.85 -> False -> checks["assembly"] == False
        d_noncore = self._new_case_dir("swap_noncore")
        spec, rt = _seed_round_dir(
            d_noncore,
            kernel="kern_swap_nc",
            base_asm=_format_nvidia_sass("kern_swap_nc", ["MOV", "IADD3", "IMAD", "EXIT"]),
            inst_asm=_format_nvidia_sass("kern_swap_nc", ["IADD3", "MOV", "IMAD", "EXIT"]),
        )
        checks_nc, metrics_nc, errors_nc, *_ = assess(d_noncore, spec, 0.10, rt)
        self.assertEqual(errors_nc, {})
        self.assertEqual(metrics_nc["lcs_ratio"], 0.75)
        self.assertEqual(metrics_nc["assembly_change_rate"], 0.25)
        self.assertEqual(metrics_nc["order_change_ratio"], 0.25)
        self.assertEqual(metrics_nc["core_retention_ratio"], 0.75)
        self.assertEqual(metrics_nc["core_multiset_ratio"], 0.75)
        self.assertIs(checks_nc["assembly"], False)

        # 1b) 4 core ops: baseline = [LDGSTS, HMMA, FFMA, UTMASTG], inst = [HMMA, LDGSTS, FFMA, UTMASTG]
        # Hand calculation:
        #   lcs = 3, len = 4, mbag = 4, core_seqs len = 4
        #   lcs_ratio = 0.75; order_change_ratio = 0.25; core_retention_ratio = 0.75; core_multiset_ratio = 1.0
        #   core gate (`backend.py:140`): core_ratio 0.75 < 0.85 -> False -> checks["assembly"] == False
        d_core4 = self._new_case_dir("swap_core4")
        spec4, rt4 = _seed_round_dir(
            d_core4,
            kernel="kern_swap_c4",
            base_asm=_format_nvidia_sass("kern_swap_c4", ["LDGSTS", "HMMA", "FFMA", "UTMASTG"]),
            inst_asm=_format_nvidia_sass("kern_swap_c4", ["HMMA", "LDGSTS", "FFMA", "UTMASTG"]),
        )
        checks_c4, metrics_c4, errors_c4, *_ = assess(d_core4, spec4, 0.10, rt4)
        self.assertEqual(errors_c4, {})
        self.assertEqual(metrics_c4["lcs_ratio"], 0.75)
        self.assertEqual(metrics_c4["assembly_change_rate"], 0.25)
        self.assertEqual(metrics_c4["order_change_ratio"], 0.25)
        self.assertEqual(metrics_c4["core_retention_ratio"], 0.75)
        self.assertEqual(metrics_c4["core_multiset_ratio"], 1.0)
        self.assertIs(checks_c4["assembly"], False)

        # 1c) 8 core ops with 1 adjacent swap: lcs = 7, len = 8, mbag = 8
        # Hand calculation:
        #   lcs_ratio = 7 / 8 = 0.875; assembly_change_rate = 1 - 0.875 = 0.125; order_change_ratio = 1 - 7/8 = 0.125
        #   core_retention_ratio = 0.875 (>= 0.85), core_multiset_ratio = 1.0 (>= 0.98), ratio 0.875 >= 0.60 -> True!
        base8 = ["UTMALDG", "LDGSTS", "HMMA", "GMMA", "WGMMA", "FFMA", "HFMA2", "UTMASTG"]
        inst8 = ["LDGSTS", "UTMALDG", "HMMA", "GMMA", "WGMMA", "FFMA", "HFMA2", "UTMASTG"]
        d_core8 = self._new_case_dir("swap_core8")
        spec8, rt8 = _seed_round_dir(
            d_core8,
            kernel="kern_swap_c8",
            base_asm=_format_nvidia_sass("kern_swap_c8", base8),
            inst_asm=_format_nvidia_sass("kern_swap_c8", inst8),
        )
        checks_c8, metrics_c8, errors_c8, *_ = assess(d_core8, spec8, 0.10, rt8)
        self.assertEqual(errors_c8, {})
        self.assertEqual(metrics_c8["lcs_ratio"], 0.875)
        self.assertEqual(metrics_c8["assembly_change_rate"], 0.125)
        self.assertEqual(metrics_c8["order_change_ratio"], 0.125)
        self.assertEqual(metrics_c8["core_retention_ratio"], 0.875)
        self.assertEqual(metrics_c8["core_multiset_ratio"], 1.0)
        self.assertIs(checks_c8["assembly"], True)

    def test_asm_pure_insertion_preserves_order(self) -> None:
        """Group 2: inserting 1 instruction without changing relative order (`A B C D` vs `A B X C D`)."""
        # Hand calculation:
        #   baseline = [LDGSTS, HMMA, UTMASTG, EXIT] (len = 4; core len = 3)
        #   inst     = [LDGSTS, HMMA, CS2R, UTMASTG, EXIT] (len = 5; CS2R is non-core; core len = 3)
        #   lcs = 4, len(seq0) = 4, mbag = 4
        #   lcs_ratio = 4 / 4 = 1.0; assembly_change_rate = 0.0; order_change_ratio = 1.0 - 4 / 4 = 0.0
        #   core_retention_ratio = 3 / 3 = 1.0; core_multiset_ratio = 3 / 3 = 1.0 -> checks["assembly"] == True
        d_ins = self._new_case_dir("insert_one")
        spec, rt = _seed_round_dir(
            d_ins,
            kernel="kern_ins",
            base_asm=_format_nvidia_sass("kern_ins", ["LDGSTS", "HMMA", "UTMASTG", "EXIT"]),
            inst_asm=_format_nvidia_sass("kern_ins", ["LDGSTS", "HMMA", "CS2R", "UTMASTG", "EXIT"]),
        )
        checks, metrics, errors, *_ = assess(d_ins, spec, 0.10, rt)
        self.assertEqual(errors, {})
        self.assertEqual(metrics["lcs_ratio"], 1.0)
        self.assertEqual(metrics["assembly_change_rate"], 0.0)
        self.assertEqual(metrics["order_change_ratio"], 0.0)
        self.assertEqual(metrics["core_retention_ratio"], 1.0)
        self.assertEqual(metrics["core_multiset_ratio"], 1.0)
        self.assertIs(checks["assembly"], True)

        # Also insert 1 CORE instruction (`DFMA`) without reordering:
        #   core_seqs[0] len = 4, core_seqs[1] len = 5 -> core_lcs = 4 -> core_retention_ratio = 1.0,
        #   core_multiset_ratio = 4 / max(4, 5) = 0.8, order_change_ratio = 0.0, checks["assembly"] == True
        d_ins_core = self._new_case_dir("insert_core")
        spec_c, rt_c = _seed_round_dir(
            d_ins_core,
            kernel="kern_ins_c",
            base_asm=_format_nvidia_sass("kern_ins_c", ["LDGSTS", "HMMA", "FFMA", "UTMASTG"]),
            inst_asm=_format_nvidia_sass("kern_ins_c", ["LDGSTS", "HMMA", "DFMA", "FFMA", "UTMASTG"]),
        )
        checks_c, metrics_c, errors_c, *_ = assess(d_ins_core, spec_c, 0.10, rt_c)
        self.assertEqual(errors_c, {})
        self.assertEqual(metrics_c["lcs_ratio"], 1.0)
        self.assertEqual(metrics_c["assembly_change_rate"], 0.0)
        self.assertEqual(metrics_c["order_change_ratio"], 0.0)
        self.assertEqual(metrics_c["core_retention_ratio"], 1.0)
        self.assertEqual(metrics_c["core_multiset_ratio"], 0.8)
        self.assertIs(checks_c["assembly"], True)

    def test_asm_full_reversal_nvidia_sass_and_amd_gcn(self) -> None:
        """Group 3: identical instruction multiset in completely reversed order (`A B C D` vs `D C B A`)."""
        # 3a) NVIDIA SASS 4 distinct core ops reversed:
        #   lcs = 1, len = 4, mbag = 4
        #   lcs_ratio = 1 / 4 = 0.25; assembly_change_rate = 0.75; order_change_ratio = 1.0 - 1 / 4 = 0.75
        #   core_retention_ratio = 0.25; core_multiset_ratio = 1.0 -> checks["assembly"] == False
        d_rev_nv = self._new_case_dir("rev_nv")
        spec_nv, rt_nv = _seed_round_dir(
            d_rev_nv,
            kernel="kern_rev_nv",
            base_asm=_format_nvidia_sass("kern_rev_nv", ["LDGSTS", "HMMA", "FFMA", "UTMASTG"]),
            inst_asm=_format_nvidia_sass("kern_rev_nv", ["UTMASTG", "FFMA", "HMMA", "LDGSTS"]),
        )
        checks_nv, metrics_nv, errors_nv, *_ = assess(d_rev_nv, spec_nv, 0.10, rt_nv)
        self.assertEqual(errors_nv, {})
        self.assertEqual(metrics_nv["lcs_ratio"], 0.25)
        self.assertEqual(metrics_nv["assembly_change_rate"], 0.75)
        self.assertEqual(metrics_nv["order_change_ratio"], 0.75)
        self.assertEqual(metrics_nv["core_retention_ratio"], 0.25)
        self.assertEqual(metrics_nv["core_multiset_ratio"], 1.0)
        self.assertIs(checks_nv["assembly"], False)

        # 3b) AMD `.s` 4 distinct non-core AMDGCN ops reversed:
        #   lcs = 1, len = 4, mbag = 4, core_seqs = [[], []]
        #   lcs_ratio = 0.25; assembly_change_rate = 0.75; order_change_ratio = 0.75
        #   core_retention_ratio = 0.25; core_multiset_ratio = 0.25 -> checks["assembly"] == False
        d_rev_amd = self._new_case_dir("rev_amd")
        spec_amd, rt_amd = _seed_round_dir(
            d_rev_amd,
            kernel="kern_rev_amd",
            base_asm=_format_amd_gcn("kern_rev_amd", ["s_mov_b32", "v_mov_b32", "s_waitcnt", "s_endpgm"]),
            inst_asm=_format_amd_gcn("kern_rev_amd", ["s_endpgm", "s_waitcnt", "v_mov_b32", "s_mov_b32"]),
            amd=True,
        )
        checks_amd, metrics_amd, errors_amd, *_ = assess(d_rev_amd, spec_amd, 0.10, rt_amd)
        self.assertEqual(errors_amd, {})
        self.assertEqual(metrics_amd["lcs_ratio"], 0.25)
        self.assertEqual(metrics_amd["assembly_change_rate"], 0.75)
        self.assertEqual(metrics_amd["order_change_ratio"], 0.75)
        self.assertEqual(metrics_amd["core_retention_ratio"], 0.25)
        self.assertEqual(metrics_amd["core_multiset_ratio"], 0.25)
        self.assertIs(checks_amd["assembly"], False)


if __name__ == "__main__":
    unittest.main()
