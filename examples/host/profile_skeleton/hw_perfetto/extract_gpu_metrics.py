#!/usr/bin/env python3
"""extract_gpu_metrics.py (E19) — nsys sqlite + metric whitelist -> raw PM samples.

FULLY IMPLEMENTED generic logic, adapted from examples/host/extract_gpu_metrics.py.
Reads the nsys export sqlite, finds the target kernel launches by name, clips the
device-wide GPU_METRICS samples to those launch windows (+/- margin), applies the
whitelist under ../metrics/, and writes a `gpu-hw-metrics/v1` document. duckdb is
used when available (50-100x faster range-join); otherwise a sqlite3 sweep-line
fallback runs. Both backends produce byte-equivalent output.

The ONLY project-specific input is the target kernel name substring (needle):
that identifies YOUR kernel among all launches in the capture. It is a PROJECT
HOOK supplied via --kernel-name-contains or the contract; there is no safe default.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "gpu-hw-metrics/v1"
UNIT_RE = re.compile(r"\[([^\]]+)\]\s*$")
# ===== PROJECT HOOK: default metric whitelist lives next to this file =====
# The generic default is metrics/metrics_whitelist.json (see E18). Fill that file
# with YOUR kernel's SM/DRAM/NVLink metrics; the loader below reads keep/required/
# combine/scale from it.
DEFAULT_ALLOWLIST = Path(__file__).resolve().parent / "metrics" / "metrics_whitelist.json"


def list_tables(conn: sqlite3.Connection) -> List[str]:
    return [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [row[1] for row in conn.execute('PRAGMA table_info("{}")'.format(table))]


def pick_table(tables: List[str], want: str) -> Optional[str]:
    for name in tables:
        if name.upper() == want.upper():
            return name
    return None


def split_name_unit(metric_name: str) -> Tuple[str, str]:
    match = UNIT_RE.search(metric_name)
    if match:
        return metric_name[:match.start()].strip(), match.group(1).strip()
    return metric_name.strip(), ""


def normalize_metric(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def load_allowlist(path: Path) -> Tuple[List[str], List[str], Dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit("[fatal] metrics whitelist must be a JSON object: {}".format(path))
    keep = config.get("keep")
    required = config.get("required")
    # WHY these are hard failures instead of `or []`: an empty/absent `keep` used to
    # fall through to "filtering disabled", i.e. every PM metric in the capture gets
    # published — and an empty/absent `required` used to make validate_required_metrics()
    # return valid=True while checking nothing at all. Both look like a clean pass.
    # There is no safe default: only the project knows which counters its kernel needs.
    if not isinstance(keep, list) or not keep:
        raise SystemExit(
            "[fatal] whitelist 'keep' is missing/empty in {}: an empty keep list silently "
            "disables filtering and publishes every PM metric in the capture. Run this script "
            "once with --keep-all against a real capture to dump the available metric names, "
            "then list the SM/DRAM/NVLink counters your kernel actually needs.".format(path))
    if not all(isinstance(item, str) for item in keep):
        raise SystemExit("[fatal] whitelist 'keep' must be a list of strings: {}".format(path))
    if not isinstance(required, list) or not required:
        raise SystemExit(
            "[fatal] whitelist 'required' is missing/empty in {}: with an empty required list "
            "the metric validation passes without checking anything, so a capture that recorded "
            "zero samples for the counter you care about still reports success. List the counters "
            "whose absence must fail the run (at minimum the SM-activity counter and the DRAM "
            "components you quote).".format(path))
    if not all(isinstance(item, str) for item in required):
        raise SystemExit("[fatal] whitelist 'required' must be a list of strings: {}".format(path))
    placeholders = [name for name in list(keep) + list(required)
                    if "REPLACE_WITH" in name.upper() or "PROJECT_HOOK" in name.upper()]
    if placeholders:
        raise SystemExit(
            "[fatal] whitelist {} still contains placeholder metric names {}: replace them with "
            "the exact names nsys reports for THIS capture.".format(path, placeholders))
    return list(keep), list(required), config


def match_keep(name: str, keep_norm: List[Tuple[str, str]]) -> Optional[str]:
    normalized = normalize_metric(name)
    for original, keep in keep_norm:
        if normalized == keep or keep in normalized or normalized in keep:
            return original
    return None


def apply_combine_rules(counters: List[Dict[str, Any]], rules: List[Dict[str, Any]]) -> None:
    """Materialize each `combine` rule as a summed counter. Every failure is fatal.

    WHY no soft skip: a combine rule exists because the components alone are not the
    number a reader quotes. DRAM read and write are both "fraction of cycles the shared
    interface was active" on ONE physical interface, so only the per-sample sum is the
    roofline figure. A delivery once shipped the two components with no total because
    this function printed "source missing, skip" and returned success — the missing
    curve looked exactly like a curve nobody asked for. If a rule cannot be applied,
    either the source names are wrong (fix them) or the component is genuinely not
    captured (delete the rule deliberately) — never both silently.
    """
    for rule in rules or []:
        target = rule.get("name")
        sources = rule.get("sum_of") or []
        if not target or not sources:
            raise SystemExit(
                "[fatal] combine rule {!r} has no name and/or empty sum_of. A rule that "
                "cannot name its target and sources cannot be applied; fill it or delete "
                "the entry.".format(rule))
        found = []
        for source_name in sources:
            hit = next((c for c in counters if c.get("name") == source_name), None)
            if hit is None:
                raise SystemExit(
                    "[fatal] combine rule {!r}: source counter {!r} is not present among the "
                    "kept counters {}. Either the name does not match what this nsys build "
                    "reports (fix sum_of / keep in the whitelist), or that component was not "
                    "captured — in which case delete this combine entry on purpose. Silently "
                    "skipping it would ship the split components with no summed total, which "
                    "is exactly the roofline number that went missing once already."
                    .format(target, source_name, sorted(c.get("name") for c in counters)))
            found.append(hit)
        lengths = {len(c["samples"]) for c in found}
        if len(lengths) != 1:
            raise SystemExit(
                "[fatal] combine rule {!r}: sources have different sample counts {} so they "
                "cannot be summed per sample. The components must come from the same PM sampling "
                "stream and the same clip window; check that all sources are in `keep` and that "
                "no source was added by a scale rule.".format(target, lengths))
        merged = []
        for i in range(next(iter(lengths))):
            ts_set = {c["samples"][i][0] for c in found}
            if len(ts_set) != 1:
                raise SystemExit(
                    "[fatal] combine rule {!r}: sample {} has mismatched timestamps {} across "
                    "sources. Summing misaligned samples would fabricate a curve; the sources "
                    "must be sampled on the same tick.".format(target, i, sorted(ts_set)))
            merged.append([found[0]["samples"][i][0], sum(c["samples"][i][1] for c in found)])
        counters.append({"metric_id": None, "name": target, "raw_name": target,
                         "unit": rule.get("unit") or found[0].get("unit", ""),
                         "sample_count": len(merged), "samples": merged,
                         "combined_from": [c["name"] for c in found]})
        if rule.get("drop_sources"):
            drop = set(sources)
            counters[:] = [c for c in counters if c.get("name") not in drop]


def apply_scale_rules(counters: List[Dict[str, Any]], rules: List[Dict[str, Any]],
                      device_name: Optional[str]) -> List[Dict[str, Any]]:
    """Publish a DERIVED corrected lane for each `scale` rule. Unfillable rules are fatal.

    WHY: percentage PM counters on cut-down SKUs are normalized against the FULL die, so
    the curve plateaus at exactly 100/2^k instead of 100. A real H20 delivery shipped
    Tensor Active 4x too low because this function treated `factor: 1` + empty regex as
    "identity rule, nothing to do" and continued. Nothing failed, every gate passed, and
    the whole tensor-utilisation story in the delivery was wrong by 4x.

    So: an unfillable rule now stops the run. "I decided no correction is needed" is
    expressed by DELETING the rule, not by leaving an identity in place — a deleted rule
    is visibly absent, an identity rule is invisible.

    How to decide a factor, once, against a real capture: compare each percentage
    counter's peak against a counter you know saturates (SM Active normally reaches a
    true 100). If a counter pins at exactly 100/2, 100/4, 100/8 or 100/16 over several
    samples, its factor is that divisor and device_name_regex is the SKU that behaves
    this way.
    """
    reports: List[Dict[str, Any]] = []
    derived: List[Dict[str, Any]] = []
    for rule in rules or []:
        target = rule.get("name")
        factor = rule.get("factor")
        pattern = rule.get("device_name_regex")
        report = {"source_name": target, "factor": factor, "device_name_regex": pattern,
                  "device_name": device_name, "applied": False}
        reports.append(report)
        if not target or "REPLACE_WITH" in str(target).upper():
            raise SystemExit(
                "[fatal] scale rule {!r} has no real counter name. The template ships the "
                "placeholder REPLACE_WITH_COUNTER_NAME_OR_DELETE_THIS_ENTRY on purpose: either "
                "name the counter that is normalized against the full die, or delete the whole "
                "entry. Do not leave it half-filled.".format(rule))
        if factor is None:
            raise SystemExit(
                "[fatal] scale rule for {!r} has no factor. Decide it from a real capture: the "
                "factor is the divisor k in a measured plateau of exactly 100/k (compare against "
                "a counter that reaches a true 100, normally SM Active).".format(target))
        if factor == 1:
            raise SystemExit(
                "[fatal] scale rule for {!r} has factor 1, which applies nothing while reading "
                "like a decision. If this counter needs no correction, DELETE the entry; if it "
                "does, set the measured divisor (a counter pinned at 25 against a sibling that "
                "reaches 100 means factor 4). A shipped H20 delivery understated tensor "
                "utilisation 4x because factor 1 was copied verbatim.".format(target))
        if not isinstance(pattern, str) or not pattern or "REPLACE_WITH" in pattern.upper():
            raise SystemExit(
                "[fatal] scale rule for {!r} has no usable device_name_regex ({!r}). The "
                "correction is SKU-specific, so it must state which GPU it applies to — put a "
                "substring of the nvidia-smi name here (e.g. 'H20'). An empty regex silently "
                "disables the rule on every machine.".format(target, pattern))
        if not isinstance(device_name, str) or not device_name.strip():
            raise SystemExit(
                "[fatal] scale rule for {!r} cannot be evaluated: this run has no GPU device "
                "name, so there is no way to tell whether the correction applies. The name comes "
                "from capture_manifest.json gpu_check (written by capture_nsys.py); pass "
                "--manifest explicitly or re-run the capture on a host where nvidia-smi works. "
                "Continuing would drop the correction silently — the exact failure that shipped "
                "a 4x-low Tensor Active curve.".format(target))
        if re.search(pattern, device_name) is None:
            # The ONLY legitimate non-fatal outcome: the device name is known and this
            # rule belongs to a different SKU. Still shout, so nobody discovers later
            # that the correction they expected never ran.
            report["reason"] = "device_name_mismatch"
            print("[scale] NOT APPLIED to {!r}: device_name {!r} does not match rule regex {!r}. "
                  "If this GPU also needs the correction, widen the regex; if it does not, that "
                  "is fine — but verify the raw curve is not pinned at 100/2^k on this SKU too."
                  .format(target, device_name, pattern))
            continue
        hit = next((c for c in counters if c.get("name") == target), None)
        if hit is None:
            raise SystemExit(
                "[fatal] scale rule for {!r} matched device {!r} but that counter is not among "
                "the kept counters {}. The rule name must match a `keep` entry exactly as nsys "
                "reports it, otherwise the correction is computed for nothing."
                .format(target, device_name, sorted(c.get("name") for c in counters)))
        clone = dict(hit)
        clone["samples"] = [[ts, v * factor] for ts, v in hit["samples"]]
        clone["name"] = rule.get("derived_name") or "{} (corrected x{})".format(
            hit.get("name", target), int(factor) if factor == int(factor) else factor)
        clone["derived_from"] = hit.get("name", target)
        clone["is_derived"] = True
        clone["scale_factor"] = factor
        derived.append(clone)
        report.update({"applied": True, "reason": "device_name_matched", "derived_name": clone["name"]})
    counters.extend(derived)
    return reports


def validate_required_metrics(counters: List[Dict[str, Any]], required: List[str]) -> Dict[str, Any]:
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for counter in counters:
        by_name.setdefault(normalize_metric(counter.get("name", "")), []).append(counter)
    missing = [n for n in required if not by_name.get(normalize_metric(n))]
    zero = [n for n in required if by_name.get(normalize_metric(n))
            and all(int(c.get("sample_count", 0)) == 0 for c in by_name[normalize_metric(n)])]
    failures = []
    if missing:
        failures.append("missing required metrics: {}".format(missing))
    if zero:
        failures.append("required metrics with zero samples: {}".format(zero))
    return {"required": list(required), "missing": missing, "zero_samples": zero,
            "failures": failures, "valid": not failures}


def load_manifest(path: Path) -> Dict[str, Any]:
    # A silently-empty manifest is dangerous: device_name comes from here, and without a
    # device name every `scale` rule becomes undecidable. Absence is reported by the
    # caller (it may legitimately be a manual sqlite), but a corrupt file is a real error
    # and must not be swallowed into {}.
    if not path.is_file():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(
            "[fatal] capture manifest {} exists but cannot be read/parsed ({}). Returning an "
            "empty manifest would drop the GPU device name and silently disable every metric "
            "scale rule; re-run capture_nsys.py or pass a good --manifest.".format(path, error))
    if not isinstance(manifest, dict):
        raise SystemExit("[fatal] capture manifest {} must be a JSON object".format(path))
    return manifest


def manifest_device_name(manifest: Dict[str, Any], device: Optional[int]) -> Optional[str]:
    gpu_check = manifest.get("gpu_check")
    if not isinstance(gpu_check, dict):
        return None
    for gpu in gpu_check.get("gpus", []) or []:
        try:
            if device is not None and int(gpu.get("index")) == int(device):
                name = gpu.get("name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
        except (TypeError, ValueError):
            continue
    return None


def pick_device_type_id(conn: sqlite3.Connection) -> Optional[int]:
    row = conn.execute(
        "SELECT typeId, COUNT(*) c FROM GPU_METRICS GROUP BY typeId ORDER BY c DESC LIMIT 1").fetchone()
    return int(row[0]) if row else None


def metric_id_names(conn: sqlite3.Connection, tables: List[str], device_type_id: int) -> Dict[int, str]:
    table = pick_table(tables, "TARGET_INFO_GPU_METRICS")
    if table is None:
        raise SystemExit("[fatal] sqlite missing TARGET_INFO_GPU_METRICS (metricId->name)")
    mapping: Dict[int, str] = {}
    for metric_id, name in conn.execute(
            'SELECT metricId, metricName FROM "{}" WHERE typeId = ?'.format(table), (device_type_id,)):
        mapping[int(metric_id)] = name
    return mapping


def find_kernels(conn: sqlite3.Connection, tables: List[str], needle: str,
                 device: Optional[int]) -> Tuple[str, List[Dict[str, Any]]]:
    ktable = pick_table(tables, "CUPTI_ACTIVITY_KIND_KERNEL")
    if ktable is None:
        raise SystemExit("[fatal] sqlite missing CUPTI_ACTIVITY_KIND_KERNEL")
    if pick_table(tables, "StringIds") is None:
        raise SystemExit("[fatal] sqlite missing StringIds")
    columns = table_columns(conn, ktable)
    name_col = "demangledName" if "demangledName" in columns else ("shortName" if "shortName" in columns else None)
    if name_col is None:
        raise SystemExit("[fatal] kernel table has no name column: {}".format(columns))
    sql = ('SELECT k.correlationId, k.deviceId, k.start, k.end, s.value '
           'FROM "{}" k JOIN StringIds s ON s.id = k."{}" WHERE s.value LIKE ?'.format(ktable, name_col))
    params: List[Any] = ["%{}%".format(needle)]
    if device is not None:
        sql += " AND k.deviceId = ?"
        params.append(device)
    sql += " ORDER BY k.start"
    launches = [{"correlation_id": int(c) if c is not None else None, "device": int(d),
                 "start_ns": int(s), "end_ns": int(e), "name": n}
                for c, d, s, e, n in conn.execute(sql, params)]
    return sql, launches


def merge_windows(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for lo, hi in ordered[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def _interval_from_timestamps(ts: List[int], sample_metric_id: int) -> Dict[str, Any]:
    if len(ts) < 2:
        return {"available": False, "count": len(ts)}
    deltas = sorted(ts[i] - ts[i - 1] for i in range(1, len(ts)))
    n = len(deltas)

    def q(frac: float) -> int:
        return deltas[min(n - 1, max(0, int(round(frac * (n - 1)))))]

    return {"available": True, "sample_metric_id": sample_metric_id, "count": len(ts),
            "median": q(0.5), "p95": q(0.95), "min": deltas[0], "max": deltas[-1], "mean": sum(deltas) / n}


def read_gpu_metrics_duckdb(sqlite_path: Path, device_type_id: int, metric_ids: List[int],
                            windows: List[Tuple[int, int]]):
    import duckdb
    con = duckdb.connect(":memory:")
    try:
        con.execute("INSTALL sqlite; LOAD sqlite;")
        con.execute("ATTACH '{}' AS s (TYPE sqlite, READ_ONLY);".format(sqlite_path))
        try:
            con.execute("SET threads TO 8;")
        except Exception:
            pass
        wanted = sorted(int(m) for m in metric_ids)
        samples = {m: [] for m in wanted}
        before = {m: 0 for m in wanted}
        wset = set(wanted)
        for mid, cnt in con.execute(
                "SELECT metricId, COUNT(*) FROM s.GPU_METRICS WHERE typeId = ? GROUP BY metricId",
                [device_type_id]).fetchall():
            if int(mid) in wset:
                before[int(mid)] = int(cnt)
        con.execute("CREATE OR REPLACE TABLE _wins(lo BIGINT, hi BIGINT)")
        if windows:
            con.executemany("INSERT INTO _wins VALUES (?, ?)", windows)
        placeholders = ",".join(str(m) for m in wanted) or "NULL"
        sql = ("SELECT g.metricId, g.timestamp, g.value FROM s.GPU_METRICS g, _wins w "
               "WHERE g.typeId = {} AND g.metricId IN ({}) AND g.timestamp BETWEEN w.lo AND w.hi "
               "ORDER BY g.metricId, g.timestamp".format(int(device_type_id), placeholders))
        for mid, ts_ns, val in con.execute(sql).fetchall():
            if int(mid) in samples:
                samples[int(mid)].append([int(ts_ns), int(val)])
        sample_metric_id = wanted[0] if wanted else None
        if sample_metric_id is None:
            interval = {"available": False}
        else:
            rows = con.execute("SELECT timestamp FROM s.GPU_METRICS WHERE typeId = ? AND metricId = ? ORDER BY timestamp",
                               [device_type_id, sample_metric_id]).fetchall()
            interval = _interval_from_timestamps([int(r[0]) for r in rows], sample_metric_id)
        return samples, before, interval
    finally:
        con.close()


def read_gpu_metrics_sqlite(conn: sqlite3.Connection, device_type_id: int, metric_ids: List[int],
                            windows: List[Tuple[int, int]]):
    try:
        conn.execute("PRAGMA cache_size = -524288")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 1073741824")
    except sqlite3.Error:
        pass
    wanted = sorted(int(m) for m in metric_ids)
    samples = {m: [] for m in wanted}
    before = {m: 0 for m in wanted}
    wset = set(wanted)
    for mid, cnt in conn.execute(
            "SELECT metricId, COUNT(*) FROM GPU_METRICS WHERE typeId = ? GROUP BY metricId", (device_type_id,)):
        if int(mid) in wset:
            before[int(mid)] = int(cnt)
    ordered = sorted(windows)
    if ordered and wanted:
        idx = 0
        cur_lo, cur_hi = ordered[0]
        cursor = conn.execute(
            "SELECT metricId, timestamp, value FROM GPU_METRICS WHERE typeId = ? ORDER BY timestamp", (device_type_id,))
        for mid, ts_ns, val in cursor:
            ts_ns = int(ts_ns)
            while cur_hi is not None and ts_ns > cur_hi:
                idx += 1
                if idx >= len(ordered):
                    cur_lo, cur_hi = None, None
                    break
                cur_lo, cur_hi = ordered[idx]
            if cur_hi is None:
                break
            if ts_ns >= cur_lo and int(mid) in samples:
                samples[int(mid)].append([ts_ns, int(val)])
        for mid in samples:
            samples[mid].sort(key=lambda item: item[0])
    sample_metric_id = wanted[0] if wanted else None
    if sample_metric_id is None:
        interval = {"available": False}
    else:
        rows = conn.execute("SELECT timestamp FROM GPU_METRICS WHERE metricId = ? AND typeId = ? ORDER BY timestamp",
                            (sample_metric_id, device_type_id)).fetchall()
        interval = _interval_from_timestamps([int(r[0]) for r in rows], sample_metric_id)
    return samples, before, interval


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract target-kernel-window GPU PM samples from an nsys sqlite.")
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--contract", default=None, help="read kernel needle / device from the contract capture section")
    parser.add_argument("--kernel-name-contains", default=None,
                        help="PROJECT HOOK: substring identifying YOUR target kernel among all launches")
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--margin-frac", type=float, default=0.10)
    parser.add_argument("--metrics-allowlist", default=None)
    parser.add_argument("--keep-metric", action="append", default=None)
    parser.add_argument("--keep-all", action="store_true")
    parser.add_argument("--backend", choices=("auto", "duckdb", "sqlite"), default="auto")
    return parser.parse_args(argv)


def resolve_kernel_needle(args: argparse.Namespace) -> str:
    if args.kernel_name_contains:
        return args.kernel_name_contains
    if args.contract:
        contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
        needle = (((contract.get("capture") or {}).get("hardware") or {}).get("kernel_name_contains")
                  or (contract.get("build_and_launch") or {}).get("kernel_name_contains"))
        if needle:
            return needle
    # ===== PROJECT HOOK: target kernel name substring =====
    raise NotImplementedError(
        "PROJECT HOOK: pass --kernel-name-contains <substr> (or set "
        "capture.hardware.kernel_name_contains in the contract). This substring picks "
        "YOUR kernel out of every launch in the nsys capture; there is no safe default.")


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    needle = resolve_kernel_needle(args)
    sqlite_path = Path(args.sqlite).expanduser().resolve()
    if not sqlite_path.is_file():
        print("[fatal] sqlite not found: {}".format(sqlite_path))
        return 1
    out_path = Path(args.out).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest \
        else sqlite_path.parent / "capture_manifest.json"
    manifest = load_manifest(manifest_path)
    device = args.device if args.device is not None else manifest.get("gpu_metrics_device")
    device_name = manifest_device_name(manifest, device)
    metrics_set = manifest.get("metrics_set", "unknown")
    frequency_hz = manifest.get("frequency_hz")

    keep_list: List[str] = []
    required_list: List[str] = []
    allowlist_source: Optional[str] = None
    combine_rules: List[Dict[str, Any]] = []
    scale_rules: List[Dict[str, Any]] = []
    if not args.keep_all:
        allowlist_path = Path(args.metrics_allowlist).expanduser().resolve() if args.metrics_allowlist \
            else (DEFAULT_ALLOWLIST if DEFAULT_ALLOWLIST.is_file() else None)
        if allowlist_path and allowlist_path.is_file():
            keep_list, required_list, config = load_allowlist(allowlist_path)
            allowlist_source = str(allowlist_path)
            combine_rules = config.get("combine") or []
            scale_rules = config.get("scale") or []
    if args.keep_metric:
        keep_list = list(keep_list) + list(args.keep_metric)
    filtering = (not args.keep_all) and bool(keep_list)
    keep_norm = [(item, normalize_metric(item)) for item in keep_list]

    conn = sqlite3.connect("file:{}?mode=ro".format(sqlite_path), uri=True)
    try:
        tables = list_tables(conn)
        if pick_table(tables, "GPU_METRICS") is None:
            print("[fatal] sqlite has no GPU_METRICS table")
            return 1
        device_type_id = pick_device_type_id(conn)
        if device_type_id is None:
            print("[fatal] GPU_METRICS is empty")
            return 1
        id_to_name = metric_id_names(conn, tables, device_type_id)
        kernel_sql, launches = find_kernels(conn, tables, needle, device)
        print('[kernel] needle="{}" device={} matched {} launch(es)'.format(needle, device, len(launches)))
        if not launches:
            print("[fatal] no launch matched the target kernel needle")
            return 3

        durations = [lc["end_ns"] - lc["start_ns"] for lc in launches]
        windows = merge_windows([(lc["start_ns"] - int(max(d, 0) * args.margin_frac),
                                  lc["end_ns"] + int(max(d, 0) * args.margin_frac))
                                 for lc, d in zip(launches, durations)])

        all_ids = sorted(id_to_name)
        matched_keep: set = set()
        dropped: List[str] = []
        if filtering:
            kept = []
            for mid in all_ids:
                base, _u = split_name_unit(id_to_name[mid])
                hit = match_keep(base, keep_norm)
                if hit is not None:
                    kept.append(mid)
                    matched_keep.add(hit)
                else:
                    dropped.append(base)
            if not kept:
                # WHY fatal instead of "keeping all": matching nothing means the names in
                # `keep` do not exist in THIS capture (different nsys build, different
                # metrics set, different SKU). The old fallback published every PM metric
                # and still exited 0, so a whitelist that was wrong end-to-end looked
                # exactly like a whitelist that was right — and `required` was then
                # satisfied by accident from the unfiltered dump.
                raise SystemExit(
                    "[fatal] whitelist matched none of the {} metrics in this capture. keep={} "
                    "but nsys reports {}. Re-run this script with --keep-all to dump the exact "
                    "names this build/SKU exposes, then fix `keep` in the whitelist. Keeping all "
                    "metrics instead would silently publish an unfiltered dump and let `required` "
                    "pass by accident.".format(
                        len(all_ids), keep_list,
                        sorted({split_name_unit(id_to_name[m])[0] for m in all_ids})))
            metric_ids = sorted(kept)
        else:
            metric_ids = all_ids

        backend_used = "sqlite"
        backend_error = None
        started = time.time()
        if args.backend in ("auto", "duckdb"):
            try:
                samples, before, interval = read_gpu_metrics_duckdb(sqlite_path, device_type_id, metric_ids, windows)
                backend_used = "duckdb"
            except ImportError as exc:
                backend_error = "duckdb unavailable ({})".format(exc)
                if args.backend == "duckdb":
                    print("[fatal] forced duckdb import failed: {}".format(exc))
                    return 1
                samples, before, interval = read_gpu_metrics_sqlite(conn, device_type_id, metric_ids, windows)
            except Exception as exc:
                backend_error = "duckdb runtime error ({}): {}".format(type(exc).__name__, exc)
                if args.backend == "duckdb":
                    print("[fatal] forced duckdb failed: {}".format(exc))
                    return 1
                samples, before, interval = read_gpu_metrics_sqlite(conn, device_type_id, metric_ids, windows)
        else:
            samples, before, interval = read_gpu_metrics_sqlite(conn, device_type_id, metric_ids, windows)
        elapsed = time.time() - started
        total_before = sum(before.get(m, 0) for m in metric_ids)
        total_after = sum(len(v) for v in samples.values())
        print("[backend] {} elapsed {:.2f}s samples {} -> {}{}".format(
            backend_used, elapsed, total_before, total_after,
            " ({})".format(backend_error) if backend_error and backend_used == "sqlite" else ""))

        counters = []
        for mid in metric_ids:
            name, unit = split_name_unit(id_to_name[mid])
            counters.append({"metric_id": mid, "name": name, "raw_name": id_to_name[mid], "unit": unit,
                             "sample_count": len(samples[mid]), "sample_count_before_clip": before.get(mid, 0),
                             "samples": samples[mid]})
    finally:
        conn.close()

    apply_combine_rules(counters, combine_rules)
    scale_report = apply_scale_rules(counters, scale_rules, device_name)
    for counter in counters:
        counter.setdefault("sample_count", len(counter.get("samples", [])))
        counter["sample_count"] = len(counter["samples"])
    required_validation = validate_required_metrics(counters, required_list)

    result = {"schema": SCHEMA, "source_sqlite": str(sqlite_path),
              "source_manifest": str(manifest_path) if manifest else None,
              "device": device, "device_name": device_name, "device_type_id": device_type_id,
              "metrics_set": metrics_set, "frequency_hz": frequency_hz, "scale_rules": scale_report,
              "kernel_name_contains": needle, "margin_frac": args.margin_frac,
              "metrics_filter": {"enabled": filtering, "allowlist_source": allowlist_source,
                                 "keep": keep_list, "required": required_list,
                                 "matched": sorted(matched_keep), "dropped": dropped},
              "clip_windows_ns": [[lo, hi] for lo, hi in windows],
              "sampling_interval_ns": interval, "kernel_launches": launches,
              "counters": counters, "required_validation": required_validation}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print("[out] {} bytes={} counters={}".format(out_path, out_path.stat().st_size, len(counters)))
    if required_validation["failures"]:
        print("[out] FAIL required metric validation: {}".format(required_validation["failures"]))
    return 0 if required_validation["valid"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
