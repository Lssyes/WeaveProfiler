#!/usr/bin/env python3
"""Build compact, sanitized golden fixtures from recorded ``wp.profile`` H20 runs.

Each generated case under ``tests/golden/cases/<case>/`` follows the replay layout:

    <case>/case.json                        CLI options + sanitized provenance + digests
    <case>/kernel/handoff.cu                uninstrumented CUDA kernel source
    <case>/kernel/probe.json                input probe specification for that run
    <case>/kernel/<capture_bench>           benchmark script used during capture
    <case>/recorded/baseline_source.sha256  SHA-256 of baseline source in round-1/L0
    <case>/recorded/instrumented_source.sha256 SHA-256 of injected source in round-1/L_inst
    <case>/recorded/runtime.json            recorded ABBA timing and spill/smem metrics
    <case>/recorded/l0.sass                 recorded baseline SASS disassembly
    <case>/recorded/inst.sass               recorded instrumented SASS disassembly
    <case>/recorded/baseline.bin            recorded baseline output tensor bytes
    <case>/recorded/instrumented.bin        recorded instrumented output tensor bytes
    <case>/recorded/capture.fi02.bin.xz     same-run software recorder dump (xz)
    <case>/recorded/capture.sqlite.xz       trimmed, sanitized Nsight Systems SQLite (xz)

Machine-specific identifiers (hostnames, local/remote run directory paths, GPU UUIDs,
PCI bus locations, and encoded process IDs) are replaced with deterministic placeholders,
and ``fi_merge_perfetto.hardware_window`` equivalence is verified before writing.

Usage:
    python3 -B tests/golden/make_wpprofile_fixture.py --result-dir <capture_result_dir>
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import lzma
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.pipeline.validate.tools import fi_merge_perfetto  # noqa: E402

PAD_NS = 2_000_000  # keep GPU_METRICS rows within +-2 ms of profiled kernel window

CASE_SPECS: tuple[dict, ...] = (
    {
        "case": "wpprofile_h20_handoff_round1_fail",
        "run_tag": "A1",
        "base_commit": "2933fb1548806549ba934844b758d58cd565c7ad",
        "entry_source": "handoff.cu",
        "kernel_src_rel": "kernel/handoff.cu",
        "probe_src_rel": "kernel/probe.original.json",
        "bench_src_rel": "kernel/cuda_bench.k8192.py",
        "bench_dst_name": "cuda_bench.py",
        "card": "h20",
        "arch": "sm_90a",
        "k_steps": 8192,
        "hw_counters": False,
        "suspected_bugs_pinned": [],
        "description": (
            "Initial inner-loop mma probe (L23..26) at K=8192: "
            "FAIL on assembly and overhead checks"
        ),
    },
    {
        "case": "wpprofile_h20_handoff_collapse_fail",
        "run_tag": "A1p",
        "base_commit": "2933fb1548806549ba934844b758d58cd565c7ad",
        "entry_source": "handoff.cu",
        "kernel_src_rel": "kernel/handoff.cu",
        "probe_src_rel": "kernel/probe.revised.json",
        "bench_src_rel": "kernel/cuda_bench.k8192.py",
        "bench_dst_name": "cuda_bench.py",
        "card": "h20",
        "arch": "sm_90a",
        "k_steps": 8192,
        "hw_counters": False,
        "suspected_bugs_pinned": [],
        "description": (
            "Hoisted mma probe (L22..27) at K=8192: "
            "FAIL on pairing check via granularity gate SINGLE_STAGE_LOOP_COLLAPSE"
        ),
    },
    {
        "case": "wpprofile_h20_handoff_no_window_samples",
        "run_tag": "B_9245b14",
        "base_commit": "9245b14543f4ba03122ebe31510abadf3d1e9134",
        "entry_source": "handoff.cu",
        "kernel_src_rel": "kernel/handoff.cu",
        "probe_src_rel": "kernel/probe.revised.json",
        "bench_src_rel": "kernel/cuda_bench_hw.py",
        "bench_dst_name": "cuda_bench_hw.py",
        "card": "h20",
        "arch": "sm_90a",
        "k_steps": 4096,
        "hw_counters": True,
        "suspected_bugs_pinned": ["A4", "A6"],
        "description": (
            "Hoisted mma probe (L22..27) at K=4096 with 200 kHz GPU_METRICS sampling "
            "buffer overflow (0 in-window samples): PASS via preview.py fallback "
            "(pins current behavior of suspected bugs A4 and A6)"
        ),
    },
    {
        "case": "wpprofile_h20_handoff_round2_pass",
        "run_tag": "B_9245b14_r2",
        "base_commit": "9245b14543f4ba03122ebe31510abadf3d1e9134",
        "entry_source": "handoff.cu",
        "kernel_src_rel": "kernel/handoff.cu",
        "probe_src_rel": "kernel/probe.revised.json",
        "bench_src_rel": "kernel/cuda_bench_hw.py",
        "bench_dst_name": "cuda_bench_hw.py",
        "card": "h20",
        "arch": "sm_90a",
        "k_steps": 4096,
        "hw_counters": True,
        "suspected_bugs_pinned": [],
        "description": (
            "Hoisted mma probe (L22..27) at K=4096 with 360 in-window 200 kHz "
            "GPU_METRICS samples: PASS with 4-panel viz.py overview"
        ),
    },
)

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_PCI_BUS_RE = re.compile(r"\b[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]\b")
_GENERIC_PATH_PARTS = frozenset(
    {
        "/",
        "runs",
        "result",
        "out",
        "round1",
        "kernel",
        "p1d",
        "tmp",
        "var",
        "opt",
        "documents",
        "repos",
    }
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _compress_xz(payload: bytes) -> bytes:
    return lzma.compress(
        payload,
        format=lzma.FORMAT_XZ,
        check=lzma.CHECK_CRC64,
        preset=9 | lzma.PRESET_EXTREME,
    )


def _load_sha256sums(result_dir: Path) -> dict[str, str]:
    manifest = result_dir / "SHA256SUMS.txt"
    if not manifest.is_file():
        return {}
    sums: dict[str, str] = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            sums[parts[1].strip()] = parts[0].strip().lower()
    return sums


def _verify_input_file(result_dir: Path, rel_path: str, sums: dict[str, str]) -> Path:
    path = result_dir / rel_path
    if not path.is_file():
        raise SystemExit(f"missing required input file: {path}")
    if rel_path in sums:
        actual = _sha256_file(path)
        expected = sums[rel_path]
        if actual != expected:
            raise SystemExit(
                f"SHA-256 mismatch for {rel_path}: expected {expected}, got {actual}"
            )
    return path


def _discover_machine_tokens(
    result_dir: Path,
) -> tuple[list[tuple[str, str]], set[str]]:
    """Extract dynamic hostnames and directory prefixes from capture metadata."""
    replacements: set[tuple[str, str]] = {
        (str(result_dir), "<RESULT_DIR>"),
        (str(result_dir.resolve()), "<RESULT_DIR>"),
    }
    forbidden_tokens: set[str] = set()

    def _add_path_parts(p: Path) -> None:
        parts = p.parts
        if len(parts) >= 2 and parts[0] == "/":
            forbidden_tokens.add(f"/{parts[1]}/")
        for part in parts[2:]:
            low = part.lower()
            if (
                len(part) >= 4
                and low not in _GENERIC_PATH_PARTS
                and not part.startswith("2026-")
            ):
                forbidden_tokens.add(part)

    _add_path_parts(result_dir.resolve())

    for verify_path in sorted(result_dir.glob("runs/*/out/verify.json")):
        try:
            doc = json.loads(verify_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = doc.get("metrics") or {}
        reg_lock = metrics.get("region_lock") or {}
        host_val = reg_lock.get("host")
        if isinstance(host_val, str) and host_val:
            replacements.add((host_val, "<CAPTURE_HOST>"))
            forbidden_tokens.add(host_val)
        for candidate in (
            doc.get("failed_trace"),
            metrics.get("preview_png"),
            reg_lock.get("lock_path"),
        ):
            if isinstance(candidate, str) and "/runs/" in candidate:
                remote_root = candidate.split("/runs/", 1)[0]
                if remote_root:
                    replacements.add((remote_root, "<REMOTE_RUN_DIR>"))
                    _add_path_parts(Path(remote_root))

    ordered = sorted(
        (pair for pair in replacements if pair[0]),
        key=lambda item: (-len(item[0]), item[1]),
    )
    return ordered, forbidden_tokens


def _scrub_text(text: str, replacements: list[tuple[str, str]]) -> str:
    out = text
    for raw, placeholder in replacements:
        if raw in out:
            out = out.replace(raw, placeholder)
    out = _UUID_RE.sub("<UUID>", out)
    out = _PCI_BUS_RE.sub("<PCI_BUS>", out)
    return out


def _sanitize_sass(raw_sass: str, replacements: list[tuple[str, str]]) -> str:
    lines = []
    for line in raw_sass.splitlines():
        if line.lstrip().startswith(("Fatbin file", "file =")):
            prefix = line.split(":", 1)[0] if ":" in line else line.split("=", 1)[0]
            sep = ":" if ":" in line else "="
            line = f"{prefix}{sep} <BINARY>"
        lines.append(_scrub_text(line, replacements))
    return "\n".join(lines) + "\n"


def _sanitize_runtime_json(raw_runtime: dict, replacements: list[tuple[str, str]]) -> dict:
    cleaned: dict = {}
    for key, val in sorted(raw_runtime.items()):
        if key in (
            "baseline_output",
            "instrumented_output",
            "baseline_sass",
            "instrumented_sass",
        ) and isinstance(val, str):
            cleaned[key] = Path(val).name
        elif isinstance(val, str):
            cleaned[key] = _scrub_text(val, replacements)
        else:
            cleaned[key] = val
    return cleaned


def _build_sanitized_sqlite(
    src_sqlite_bytes: bytes,
    kernel_substr: str,
    replacements: list[tuple[str, str]],
) -> bytes:
    """Trim (within +-2 ms) and scrub machine identifiers into a fresh VACUUMed SQLite DB."""
    with tempfile.TemporaryDirectory(prefix="wp-make-fixture-sql-") as td:
        td_path = Path(td)
        src_path = td_path / "src.sqlite"
        dst_path = td_path / "capture.sqlite"
        src_path.write_bytes(src_sqlite_bytes)

        src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
        dst = sqlite3.connect(str(dst_path))
        try:
            dst.execute("PRAGMA page_size = 4096")
            dst.execute("PRAGMA auto_vacuum = NONE")
            dst.execute("PRAGMA journal_mode = MEMORY")
            dst.execute("PRAGMA synchronous = OFF")

            tables = {
                r[0]
                for r in src.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            kt = "CUPTI_ACTIVITY_KIND_KERNEL"
            if kt not in tables or "StringIds" not in tables:
                raise SystemExit("SQLite missing CUPTI_ACTIVITY_KIND_KERNEL or StringIds")

            kcols = [r[1] for r in src.execute(f'PRAGMA table_info("{kt}")')]
            name_col = next(
                c for c in ("demangledName", "shortName", "mangledName") if c in kcols
            )
            k_window_rows = src.execute(
                f'SELECT k.start, k."end" FROM "{kt}" k '
                f'JOIN StringIds s ON s.id = k."{name_col}" '
                "WHERE s.value LIKE ? ORDER BY k.start, k.rowid",
                (f"%{kernel_substr}%",),
            ).fetchall()
            if not k_window_rows:
                raise SystemExit(f"no kernel matching {kernel_substr!r} in SQLite")
            t_lo = min(int(r[0]) for r in k_window_rows) - PAD_NS
            t_hi = max(int(r[1]) for r in k_window_rows) + PAD_NS

            # 1. CUPTI_ACTIVITY_KIND_KERNEL (zero out globalPid to strip host process IDs).
            k_ddl = src.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (kt,),
            ).fetchone()[0]
            dst.execute(k_ddl)
            col_sql = ", ".join(f'"{c}"' for c in kcols)
            raw_krows = src.execute(
                f'SELECT {col_sql} FROM "{kt}" '
                f'WHERE "{name_col}" IN (SELECT id FROM StringIds WHERE value LIKE ?) '
                "ORDER BY start, rowid",
                (f"%{kernel_substr}%",),
            ).fetchall()
            clean_krows = []
            for row in raw_krows:
                vals = list(row)
                for idx, col in enumerate(kcols):
                    if col == "globalPid" and vals[idx] is not None:
                        vals[idx] = 0
                    elif isinstance(vals[idx], str):
                        vals[idx] = _scrub_text(vals[idx], replacements)
                clean_krows.append(tuple(vals))
            placeholders = ", ".join("?" for _ in kcols)
            dst.executemany(f'INSERT INTO "{kt}" VALUES ({placeholders})', clean_krows)

            # 2. StringIds referenced by kept kernel rows.
            str_ids: set[int] = set()
            for col in ("demangledName", "shortName", "mangledName"):
                if col in kcols:
                    for (sid,) in dst.execute(f'SELECT DISTINCT "{col}" FROM "{kt}"'):
                        if sid is not None:
                            str_ids.add(int(sid))
            s_ddl = src.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='StringIds'"
            ).fetchone()[0]
            dst.execute(s_ddl)
            for sid in sorted(str_ids):
                srow = src.execute(
                    "SELECT id, value FROM StringIds WHERE id = ?", (sid,)
                ).fetchone()
                if srow is not None:
                    dst.execute(
                        "INSERT INTO StringIds VALUES (?, ?)",
                        (int(srow[0]), _scrub_text(str(srow[1]), replacements)),
                    )

            # 3. TARGET_INFO_GPU with physical UUIDs, PCI bus IDs, and device paths scrubbed.
            if "TARGET_INFO_GPU" in tables:
                g_ddl = src.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='TARGET_INFO_GPU'"
                ).fetchone()[0]
                dst.execute(g_ddl)
                gcols = [
                    r[1] for r in src.execute('PRAGMA table_info("TARGET_INFO_GPU")')
                ]
                gcol_sql = ", ".join(f'"{c}"' for c in gcols)
                order_col = '"id"' if "id" in gcols else "rowid"
                raw_grows = src.execute(
                    f'SELECT {gcol_sql} FROM "TARGET_INFO_GPU" ORDER BY {order_col}, rowid'
                ).fetchall()
                id_idx = gcols.index("id") if "id" in gcols else None
                clean_grows = []
                for row_num, row in enumerate(raw_grows):
                    vals = list(row)
                    gpu_id = (
                        int(vals[id_idx])
                        if (id_idx is not None and vals[id_idx] is not None)
                        else row_num
                    )
                    for idx, col in enumerate(gcols):
                        if col == "uuid":
                            vals[idx] = f"<GPU_UUID_{gpu_id}>"
                        elif col == "busLocation":
                            vals[idx] = f"<PCI_BUS_{gpu_id}>"
                        elif col in ("ctxswDevPath", "ctrlDevPath"):
                            vals[idx] = ""
                        elif col == "luid":
                            vals[idx] = 0
                        elif col == "vmId" and vals[idx] is not None:
                            vals[idx] = 0
                        elif isinstance(vals[idx], str):
                            vals[idx] = _scrub_text(vals[idx], replacements)
                    clean_grows.append(tuple(vals))
                g_ph = ", ".join("?" for _ in gcols)
                dst.executemany(
                    f'INSERT INTO "TARGET_INFO_GPU" VALUES ({g_ph})', clean_grows
                )

            # 4. TARGET_INFO_GPU_METRICS and GPU_METRICS (within +-2 ms window).
            if "TARGET_INFO_GPU_METRICS" in tables:
                m_ddl = src.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='TARGET_INFO_GPU_METRICS'"
                ).fetchone()[0]
                dst.execute(m_ddl)
                mcols = [
                    r[1]
                    for r in src.execute(
                        'PRAGMA table_info("TARGET_INFO_GPU_METRICS")'
                    )
                ]
                mcol_sql = ", ".join(f'"{c}"' for c in mcols)
                raw_mrows = src.execute(
                    f'SELECT {mcol_sql} FROM "TARGET_INFO_GPU_METRICS" ORDER BY metricId, rowid'
                ).fetchall()
                clean_mrows = [
                    tuple(
                        _scrub_text(v, replacements) if isinstance(v, str) else v
                        for v in row
                    )
                    for row in raw_mrows
                ]
                m_ph = ", ".join("?" for _ in mcols)
                dst.executemany(
                    f'INSERT INTO "TARGET_INFO_GPU_METRICS" VALUES ({m_ph})',
                    clean_mrows,
                )

            if "GPU_METRICS" in tables:
                gm_ddl = src.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='GPU_METRICS'"
                ).fetchone()[0]
                dst.execute(gm_ddl)
                gmcols = [
                    r[1] for r in src.execute('PRAGMA table_info("GPU_METRICS")')
                ]
                gmcol_sql = ", ".join(f'"{c}"' for c in gmcols)
                gm_rows = src.execute(
                    f'SELECT {gmcol_sql} FROM "GPU_METRICS" '
                    "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp, metricId, rowid",
                    (t_lo, t_hi),
                ).fetchall()
                gm_ph = ", ".join("?" for _ in gmcols)
                dst.executemany(
                    f'INSERT INTO "GPU_METRICS" VALUES ({gm_ph})', gm_rows
                )

            dst.commit()
            dst.execute("VACUUM")
            dst.commit()
        finally:
            dst.close()
            src.close()

        # Verify fi_merge_perfetto.hardware_window equivalence before and after sanitization.
        ref_dir = td_path / "ref_round"
        got_dir = td_path / "got_round"
        ref_dir.mkdir()
        got_dir.mkdir()
        (ref_dir / "capture.sqlite").write_bytes(src_sqlite_bytes)
        shutil.copyfile(dst_path, got_dir / "capture.sqlite")
        rt_probe = {"kernel_name": kernel_substr}
        ref_win = fi_merge_perfetto.hardware_window(ref_dir, rt_probe)
        got_win = fi_merge_perfetto.hardware_window(got_dir, rt_probe)
        if tuple(ref_win) != tuple(got_win) or getattr(
            ref_win, "gpu_name", None
        ) != getattr(got_win, "gpu_name", None):
            raise SystemExit(
                f"sanitized SQLite altered hardware_window for {kernel_substr!r}"
            )

        return dst_path.read_bytes()


def _audit_case_hygiene(case_dir: Path, forbidden_tokens: set[str]) -> None:
    """Verify generated fixture files (including decompressed .xz payloads) have zero machine leaks."""
    for path in sorted(case_dir.rglob("*")):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        payloads = [(path.name, raw)]
        if path.suffix == ".xz":
            payloads.append((f"{path.name}(decompressed)", lzma.decompress(raw)))
        for label, blob in payloads:
            text = blob.decode("latin1")
            low = text.lower()
            for tok in sorted(forbidden_tokens):
                if tok.lower() in low:
                    raise SystemExit(
                        f"machine token {tok!r} leaked into {case_dir.name}/{label}"
                    )
            uuid_hit = _UUID_RE.search(text)
            if uuid_hit:
                raise SystemExit(
                    f"UUID {uuid_hit.group(0)!r} leaked into {case_dir.name}/{label}"
                )
            pci_hit = _PCI_BUS_RE.search(text)
            if pci_hit:
                raise SystemExit(
                    f"PCI bus ID {pci_hit.group(0)!r} leaked into {case_dir.name}/{label}"
                )


def build_one_case(
    spec: dict,
    result_dir: Path,
    out_root: Path,
    sums: dict[str, str],
    replacements: list[tuple[str, str]],
    forbidden_tokens: set[str],
) -> dict:
    case_name = spec["case"]
    tag = spec["run_tag"]
    entry_name = spec["entry_source"]

    kernel_src_p = _verify_input_file(result_dir, spec["kernel_src_rel"], sums)
    probe_src_p = _verify_input_file(result_dir, spec["probe_src_rel"], sums)
    bench_src_p = _verify_input_file(result_dir, spec["bench_src_rel"], sums)

    r1_rel = f"runs/{tag}/round1"
    l0_src_p = _verify_input_file(result_dir, f"{r1_rel}/L0_{entry_name}", sums)
    l_inst_src_p = _verify_input_file(result_dir, f"{r1_rel}/L_inst_{entry_name}", sums)
    base_sha_p = _verify_input_file(
        result_dir, f"{r1_rel}/baseline_source.sha256", sums
    )
    inst_sha_p = _verify_input_file(
        result_dir, f"{r1_rel}/instrumented_source.sha256", sums
    )
    rt_p = _verify_input_file(result_dir, f"{r1_rel}/runtime.json", sums)
    l0_sass_p = _verify_input_file(result_dir, f"{r1_rel}/l0.sass", sums)
    inst_sass_p = _verify_input_file(result_dir, f"{r1_rel}/inst.sass", sums)
    base_bin_p = _verify_input_file(result_dir, f"{r1_rel}/baseline.bin", sums)
    inst_bin_p = _verify_input_file(result_dir, f"{r1_rel}/instrumented.bin", sums)
    fi02_xz_p = _verify_input_file(result_dir, f"{r1_rel}/capture.fi02.bin.xz", sums)

    sqlite_xz_rel = f"{r1_rel}/capture.sqlite.xz"
    sqlite_raw_rel = f"{r1_rel}/capture.sqlite"
    if (result_dir / sqlite_xz_rel).is_file():
        sqlite_in_p = _verify_input_file(result_dir, sqlite_xz_rel, sums)
        src_sqlite_bytes = lzma.decompress(sqlite_in_p.read_bytes())
    else:
        sqlite_in_p = _verify_input_file(result_dir, sqlite_raw_rel, sums)
        src_sqlite_bytes = sqlite_in_p.read_bytes()

    kernel_bytes = kernel_src_p.read_bytes()
    if _sha256_bytes(kernel_bytes) != _sha256_file(l0_src_p):
        raise SystemExit(f"{case_name}: kernel/{entry_name} != {r1_rel}/L0_{entry_name}")

    base_sha = base_sha_p.read_text(encoding="utf-8").strip().lower()
    inst_sha = inst_sha_p.read_text(encoding="utf-8").strip().lower()
    if base_sha != _sha256_bytes(kernel_bytes):
        raise SystemExit(f"{case_name}: baseline_source.sha256 mismatch")
    if inst_sha != _sha256_file(l_inst_src_p):
        raise SystemExit(f"{case_name}: instrumented_source.sha256 mismatch")

    probe_doc = json.loads(probe_src_p.read_text(encoding="utf-8"))
    probe_bytes = (
        json.dumps(probe_doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    bench_bytes = _scrub_text(
        bench_src_p.read_text(encoding="utf-8"), replacements
    ).encode("utf-8")

    raw_rt = json.loads(rt_p.read_text(encoding="utf-8"))
    clean_rt = _sanitize_runtime_json(raw_rt, replacements)
    rt_bytes = (
        json.dumps(clean_rt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    l0_sass_bytes = _sanitize_sass(
        l0_sass_p.read_text(encoding="utf-8"), replacements
    ).encode("utf-8")
    inst_sass_bytes = _sanitize_sass(
        inst_sass_p.read_text(encoding="utf-8"), replacements
    ).encode("utf-8")
    base_bin_bytes = base_bin_p.read_bytes()
    inst_bin_bytes = inst_bin_p.read_bytes()

    fi02_raw_bytes = lzma.decompress(fi02_xz_p.read_bytes())
    fi02_xz_bytes = _compress_xz(fi02_raw_bytes)

    kernel_name = str(clean_rt.get("kernel_name") or "warp_handoff")
    sanitized_sqlite_bytes = _build_sanitized_sqlite(
        src_sqlite_bytes, kernel_name, replacements
    )
    sqlite_xz_bytes = _compress_xz(sanitized_sqlite_bytes)

    meta = {
        "arch": spec["arch"],
        "capture_bench": spec["bench_dst_name"],
        "card": spec["card"],
        "case": case_name,
        "description": spec["description"],
        "entry_source": entry_name,
        "inputs_sha256": {
            "baseline.bin": _sha256_bytes(base_bin_bytes),
            "baseline_source": base_sha,
            "capture.fi02.bin": _sha256_bytes(fi02_raw_bytes),
            "capture.sqlite(sanitized)": _sha256_bytes(sanitized_sqlite_bytes),
            "capture_bench": _sha256_bytes(bench_bytes),
            "entry_source": _sha256_bytes(kernel_bytes),
            "inst.sass": _sha256_bytes(inst_sass_bytes),
            "instrumented.bin": _sha256_bytes(inst_bin_bytes),
            "instrumented_source": inst_sha,
            "l0.sass": _sha256_bytes(l0_sass_bytes),
            "probe.json(source)": _sha256_file(probe_src_p),
            "runtime.json(sanitized)": _sha256_bytes(rt_bytes),
        },
        "k_steps": spec["k_steps"],
        "limit": 0.10,
        "max_rounds": 1,
        "out_name": "overview.png",
        "probe": "probe.json",
        "provenance": {
            "base_commit": spec["base_commit"],
            "capture_host": "<CAPTURE_HOST>",
            "capture_run_dir": f"<REMOTE_RUN_DIR>/runs/{tag}",
            "gpu_model": "NVIDIA H20",
            "hw_counters_enabled": spec["hw_counters"],
            "run_tag": tag,
        },
        "sizes": {
            "capture_fi02_raw_bytes": len(fi02_raw_bytes),
            "capture_fi02_xz_bytes": len(fi02_xz_bytes),
            "capture_sqlite_sanitized_bytes": len(sanitized_sqlite_bytes),
            "capture_sqlite_xz_bytes": len(sqlite_xz_bytes),
        },
        "suspected_bugs_pinned": spec["suspected_bugs_pinned"],
    }
    case_json_bytes = (
        json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    out_root.mkdir(parents=True, exist_ok=True)
    target_dir = (out_root / case_name).resolve()
    existing_expected = (
        (target_dir / "expected.json").read_bytes()
        if (target_dir / "expected.json").is_file()
        else None
    )

    stage_dir = Path(tempfile.mkdtemp(prefix=f".stage-{case_name}-", dir=str(out_root)))
    try:
        (stage_dir / "kernel").mkdir(parents=True)
        (stage_dir / "recorded").mkdir(parents=True)
        (stage_dir / "case.json").write_bytes(case_json_bytes)
        (stage_dir / "kernel" / entry_name).write_bytes(kernel_bytes)
        (stage_dir / "kernel" / "probe.json").write_bytes(probe_bytes)
        (stage_dir / "kernel" / spec["bench_dst_name"]).write_bytes(bench_bytes)

        rec = stage_dir / "recorded"
        (rec / "baseline_source.sha256").write_bytes((base_sha + "\n").encode("utf-8"))
        (rec / "instrumented_source.sha256").write_bytes(
            (inst_sha + "\n").encode("utf-8")
        )
        (rec / "runtime.json").write_bytes(rt_bytes)
        (rec / "l0.sass").write_bytes(l0_sass_bytes)
        (rec / "inst.sass").write_bytes(inst_sass_bytes)
        (rec / "baseline.bin").write_bytes(base_bin_bytes)
        (rec / "instrumented.bin").write_bytes(inst_bin_bytes)
        (rec / "capture.fi02.bin.xz").write_bytes(fi02_xz_bytes)
        (rec / "capture.sqlite.xz").write_bytes(sqlite_xz_bytes)

        if existing_expected is not None:
            (stage_dir / "expected.json").write_bytes(existing_expected)

        _audit_case_hygiene(stage_dir, forbidden_tokens)

        if target_dir.exists():
            shutil.rmtree(target_dir)
        stage_dir.rename(target_dir)
    except BaseException:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        raise

    total_bytes = sum(p.stat().st_size for p in target_dir.rglob("*") if p.is_file())
    return {
        "case": case_name,
        "run_tag": tag,
        "fixture_bytes": total_bytes,
        **meta["sizes"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build sanitized wp.profile H20 replay fixtures from capture results."
    )
    ap.add_argument(
        "--result-dir",
        required=True,
        type=Path,
        help="Capture result root containing kernel/, runs/<tag>/round1/, and SHA256SUMS.txt",
    )
    ap.add_argument(
        "--case",
        default="all",
        choices=["all", *[s["case"] for s in CASE_SPECS]],
        help="Specific case name to build, or 'all' (default)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=HERE / "cases",
        help="Output cases directory (defaults to tests/golden/cases)",
    )
    args = ap.parse_args(argv)

    result_dir = args.result_dir.resolve()
    if not result_dir.is_dir():
        raise SystemExit(f"result directory not found: {result_dir}")

    sums = _load_sha256sums(result_dir)
    replacements, forbidden_tokens = _discover_machine_tokens(result_dir)
    selected = [
        s for s in CASE_SPECS if args.case in ("all", s["case"])
    ]
    reports = [
        build_one_case(
            spec,
            result_dir,
            args.out.resolve(),
            sums,
            replacements,
            forbidden_tokens,
        )
        for spec in selected
    ]
    print(json.dumps({"built": reports}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
