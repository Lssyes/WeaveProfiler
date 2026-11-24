#!/usr/bin/env python3
"""Build a compact golden fixture from one real megakernel-audit capture.

A fixture is everything Workflow B of ``run_megakernel_audit`` needs, made small enough for git:

    <case>/workspace/<entry>          clean production source (the audit hashes it)
    <case>/names.json                 event / role names used for the capture
    <case>/runtime.json               bench runtime metrics; the multi-MB output files are replaced by their
                                      sha256 (``baseline_output_sha256`` / ``instrumented_output_sha256``)
    <case>/capture.fi02.bin.xz        same-run recorder dump (xz)
    <case>/nsys.sqlite.xz             nsys SQLite trimmed to the tables / rows extract_nsys_sqlite_fast reads
    <case>/case.json                  audit arguments + provenance

The trimmed SQLite is accepted only if ``extract_nsys_sqlite_fast`` returns the same result on it as on the
full report (``sqlite`` path and ``elapsed_ms`` excluded). Run on the machine that holds the capture:

    python3 -B tests/golden/make_fixture.py --case h20_v6 --run-dir <job dir> --entry matmul-v6.cu \
        --card h20 --arch sm_90a --total-sms 78 --title "..." --provenance "..."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from weaveprof.megakernel import extract_nsys_sqlite_fast  # noqa: E402

PAD_NS = 2_000_000  # keep GPU_METRICS rows within +-2 ms of the profiled kernels


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _xz(src: Path, dst: Path) -> None:
    with open(src, "rb") as fi, lzma.open(dst, "wb", preset=9 | lzma.PRESET_EXTREME) as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)


def trim_sqlite(full: Path, out: Path, kernel_substr: str) -> None:
    """Copy only what extract_nsys_sqlite_fast reads, limited to the kernel windows."""
    src = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    if out.exists():
        out.unlink()
    dst = sqlite3.connect(str(out))
    tables = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def copy(table: str, where: str = "", params=()):
        if table not in tables:
            return
        ddl = src.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
        dst.execute(ddl)
        cols = [r[1] for r in src.execute(f'PRAGMA table_info("{table}")')]
        q = f'SELECT {", ".join(chr(34) + c + chr(34) for c in cols)} FROM "{table}" {where} ORDER BY rowid'
        rows = src.execute(q, params).fetchall()
        dst.executemany(f'INSERT INTO "{table}" VALUES ({", ".join("?" for _ in cols)})', rows)

    kt = "CUPTI_ACTIVITY_KIND_KERNEL"
    kcols = {r[1] for r in src.execute(f'PRAGMA table_info("{kt}")')}
    name_col = next(c for c in ("demangledName", "shortName", "mangledName") if c in kcols)
    k_rows = src.execute(
        f'SELECT k.start, k."end", k."{name_col}" FROM "{kt}" k JOIN StringIds s ON s.id = k."{name_col}" '
        "WHERE s.value LIKE ?", (f"%{kernel_substr}%",)).fetchall()
    if not k_rows:
        raise SystemExit(f"no kernel matching {kernel_substr!r} in {full}")
    t_lo = min(r[0] for r in k_rows) - PAD_NS
    t_hi = max(r[1] for r in k_rows) + PAD_NS
    copy(kt, f'WHERE "{name_col}" IN (SELECT id FROM StringIds WHERE value LIKE ?)', (f"%{kernel_substr}%",))
    str_ids = {r[0] for r in dst.execute(f'SELECT DISTINCT "{name_col}" FROM "{kt}"')}
    for c in ("demangledName", "shortName", "mangledName"):
        if c in kcols and c != name_col:
            str_ids |= {r[0] for r in dst.execute(f'SELECT DISTINCT "{c}" FROM "{kt}"')}
    ddl = src.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='StringIds'").fetchone()[0]
    dst.execute(ddl)
    for sid in sorted(i for i in str_ids if i is not None):
        row = src.execute("SELECT id, value FROM StringIds WHERE id = ?", (sid,)).fetchone()
        if row:
            dst.execute("INSERT INTO StringIds VALUES (?, ?)", row)
    copy("TARGET_INFO_GPU")
    copy("TARGET_INFO_GPU_METRICS")
    copy("GPU_METRICS", "WHERE timestamp BETWEEN ? AND ?", (t_lo, t_hi))
    dst.commit()
    dst.execute("VACUUM")
    dst.close()
    src.close()


def _comparable(hw: dict) -> dict:
    return {k: v for k, v in hw.items() if k not in ("sqlite", "elapsed_ms")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--run-dir", required=True, type=Path, help="job dir with workspace/ and audit_out/")
    ap.add_argument("--entry", required=True)
    ap.add_argument("--card", default="h20")
    ap.add_argument("--arch", default="sm_90a")
    ap.add_argument("--total-sms", type=int, default=78)
    ap.add_argument("--title", default="")
    ap.add_argument("--provenance", default="")
    ap.add_argument("--out", type=Path, default=HERE / "cases")
    args = ap.parse_args()

    run = args.run_dir.resolve()
    a = run / "audit_out"
    rm_path = a / "bench" / "runtime.json"
    rm = json.loads(rm_path.read_text())
    case = (args.out / args.case).resolve()
    if case.exists():
        shutil.rmtree(case)
    (case / "workspace").mkdir(parents=True)
    shutil.copy2(run / "workspace" / args.entry, case / "workspace" / args.entry)
    shutil.copy2(run / "workspace" / "names.json", case / "names.json")

    # Output evidence by digest instead of 16-64 MB files (megakernel._gate_output supports both).
    pairs = rm.pop("output_files", None) or []
    if pairs:
        base, inst = (Path(p) if Path(p).is_absolute() else rm_path.parent / p for p in pairs[0])
        rm["baseline_output_sha256"] = _sha256(base)
        rm["instrumented_output_sha256"] = _sha256(inst)
        rm["output_bytes"] = base.stat().st_size
    (case / "runtime.json").write_text(json.dumps(rm, indent=1) + "\n")

    dump = a / "nsys_manual" / "capture.fi02.bin"
    _xz(dump, case / "capture.fi02.bin.xz")

    full_sqlite = a / "nsys_manual" / "cap.sqlite"
    kernel_substr = str(rm.get("kernel_name") or "")
    with tempfile.TemporaryDirectory() as td:
        trimmed = Path(td) / "nsys.sqlite"
        trim_sqlite(full_sqlite, trimmed, kernel_substr)
        ref = _comparable(extract_nsys_sqlite_fast(full_sqlite, kernel_substr=kernel_substr))
        got = _comparable(extract_nsys_sqlite_fast(trimmed, kernel_substr=kernel_substr))
        if json.dumps(ref, sort_keys=True) != json.dumps(got, sort_keys=True):
            raise SystemExit("trimmed SQLite changes extract_nsys_sqlite_fast output; widen PAD_NS")
        _xz(trimmed, case / "nsys.sqlite.xz")
        trimmed_bytes = trimmed.stat().st_size

    meta = {
        "case": args.case,
        "entry_source": args.entry,
        "card": args.card,
        "arch": args.arch,
        "total_sms": args.total_sms,
        "limit": 0.10,
        "overview_title": args.title,
        "provenance": args.provenance,
        "inputs_sha256": {
            "capture.fi02.bin": _sha256(dump),
            "nsys.sqlite(full)": _sha256(full_sqlite),
            "runtime.json(original)": _sha256(rm_path),
        },
        "sizes": {"dump_bytes": dump.stat().st_size, "sqlite_full_bytes": full_sqlite.stat().st_size,
                  "sqlite_trimmed_bytes": trimmed_bytes},
    }
    (case / "case.json").write_text(json.dumps(meta, indent=1) + "\n")
    total = sum(p.stat().st_size for p in case.rglob("*") if p.is_file())
    print(json.dumps({"case": args.case, "fixture_bytes": total, **meta["sizes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
