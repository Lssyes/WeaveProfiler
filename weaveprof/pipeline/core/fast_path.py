#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast instrumentation build/smoke path for weaveprofiler.

This is deliberately not a delivery-evidence path.  It builds the two artifacts
needed while iterating on instrumentation -- instrumentation disabled (off) and
fully enabled (full) -- in parallel, caches successful builds by source and
command identity, then runs a short smoke sequence.  Correctness sweeps, entry
only timing, hardware replay and delivery gates belong to the slower check and
delivery paths.

Usage:
    python3 -B run.py fast --job fast_job.json
    python3 -B run.py template
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "weaveprofiler-fast-job/v1"
RESULT_SCHEMA = "weaveprofiler-fast-result/v1"
CACHE_SCHEMA = "weaveprofiler-fast-cache/v1"
TOOL_ID = "weaveprofiler-fast/v1"
SKILL_ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN_TOKENS = {"--shell", "--skip"}
PRUNE_PARTS = {".git", ".hg", ".svn", "__pycache__", ".weaveprofiler", ".mypy_cache", ".pytest_cache"}


class FastPathError(RuntimeError):
    def __init__(self, message: str, *, records: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message)
        self.records = records or []


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(partial), str(path))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FastPathError("cannot read JSON {}: {}".format(path, exc))


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _positive_number(value: Any, where: str, default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise FastPathError("{}.timeout_sec must be a positive number".format(where))
    return float(value)


def _validate_argv(argv: Any, where: str) -> List[str]:
    if not isinstance(argv, list) or not argv or not all(isinstance(v, str) and v for v in argv):
        raise FastPathError("{} must be a non-empty argv string list".format(where))
    for token in argv:
        low = token.strip().lower()
        if low in FORBIDDEN_TOKENS or low.startswith("--skip") or low == "--shell":
            raise FastPathError("{} contains forbidden shell/skip token {!r}".format(where, token))
    return list(argv)


def _command_spec(raw: Any, project_root: Path, where: str, *, outputs_required: bool = True) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise FastPathError("{} must be an object".format(where))
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise FastPathError("{}.name is required".format(where))
    argv = _validate_argv(raw.get("argv"), where + ".argv")
    cwd = _resolve(str(raw.get("cwd", project_root)), project_root)
    env_raw = raw.get("env", {})
    if not isinstance(env_raw, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                 for k, v in env_raw.items()):
        raise FastPathError("{}.env must map strings to strings".format(where))
    outputs_raw = raw.get("outputs", [])
    if not isinstance(outputs_raw, list) or not all(isinstance(v, str) and v for v in outputs_raw):
        raise FastPathError("{}.outputs must be a string list".format(where))
    outputs = [_resolve(v, cwd) for v in outputs_raw]
    if outputs_required and not outputs:
        raise FastPathError("{} must declare at least one output".format(where))
    timeout = _positive_number(raw.get("timeout_sec"), where, 3600.0)
    return {"name": name.strip(), "argv": argv, "cwd": cwd, "env": dict(env_raw),
            "outputs": outputs, "timeout_sec": timeout, "raw": raw}


def _iter_glob(project_root: Path, pattern: str) -> Iterable[Path]:
    path = Path(pattern)
    if path.is_absolute():
        matches = [path]
    else:
        matches = list(project_root.glob(pattern))
    seen = set()
    for match in matches:
        if not match.is_file():
            continue
        resolved = match.resolve()
        rel_parts = resolved.relative_to(project_root).parts if _inside(resolved, project_root) else resolved.parts
        if any(part in PRUNE_PARTS for part in rel_parts):
            continue
        if resolved not in seen:
            seen.add(resolved)
            yield resolved


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _collect_sources(project_root: Path, job: Dict[str, Any]) -> List[Path]:
    raw_files = job.get("source_files")
    raw_globs = job.get("source_globs")
    if raw_files is not None and (not isinstance(raw_files, list) or not all(isinstance(v, str) and v for v in raw_files)):
        raise FastPathError("source_files must be a string list")
    if raw_globs is not None and (not isinstance(raw_globs, list) or not all(isinstance(v, str) and v for v in raw_globs)):
        raise FastPathError("source_globs must be a string list")
    paths: List[Path] = []
    for raw in raw_files or []:
        path = _resolve(raw, project_root)
        if not path.is_file():
            raise FastPathError("source file does not exist: {}".format(path))
        paths.append(path)
    for pattern in raw_globs or []:
        paths.extend(_iter_glob(project_root, pattern))
    unique = sorted({p.resolve() for p in paths})
    if not unique:
        raise FastPathError("job must select at least one existing source file via source_files or source_globs")
    outside = [str(p) for p in unique if not _inside(p, project_root)]
    if outside:
        raise FastPathError("source files must stay under project_root: {}".format(", ".join(outside[:3])))
    return unique


def _source_digest(project_root: Path, files: Sequence[Path], cache_path: Path) -> Dict[str, Any]:
    """Hash selected source files, reusing hashes only when size and mtime match.

    This stat cache is intentionally limited to the fast path.  Delivery paths
    re-read and hash every evidence input.
    """
    previous: Dict[str, Any] = {}
    if cache_path.is_file():
        try:
            doc = json.loads(cache_path.read_text(encoding="utf-8"))
            if doc.get("schema") == "weaveprofiler-fast-file-hash/v1":
                previous = doc.get("files", {})
        except (OSError, ValueError):
            previous = {}
    records: Dict[str, Any] = {}
    hashes: List[str] = []
    for path in files:
        rel = str(path.relative_to(project_root)).replace(os.sep, "/")
        stat = path.stat()
        # Always hash selected sources.  mtime-only reuse can silently miss a
        # content change after checkout/rsync preserving timestamps; hashing a
        # handful of selected files is cheap next to compiling them.
        sha = _sha256_file(path)
        records[rel] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha}
        hashes.append(rel + "\0" + sha)
    if records != previous:
        _atomic_write_json(cache_path, {"schema": "weaveprofiler-fast-file-hash/v1", "files": records})
    return {"digest": _sha256_bytes("\n".join(hashes).encode("utf-8")), "files": records}


def _normalized_target(target: Dict[str, Any]) -> Dict[str, Any]:
    return {"platform": target.get("platform"), "card": target.get("card"),
            "arch": target.get("arch"), "architectures": target.get("architectures"),
            "language": target.get("language")}


def _toolchain_env() -> Dict[str, Optional[str]]:
    return {name: os.environ.get(name) for name in
            ("PATH", "CC", "CXX", "NVCC", "HIPCC", "CUDA_HOME", "ROCM_PATH")}


def _base_key(job: Dict[str, Any], target: Dict[str, Any], source: Dict[str, Any]) -> str:
    payload = {
        "tool": TOOL_ID,
        "tool_sha256": _sha256_file(Path(__file__).resolve()),
        "toolchain_env": _toolchain_env(),
        "job_schema": job.get("schema"),
        "project_root": str(Path(job["_project_root"]).resolve()),
        "target": _normalized_target(target),
        "source_digest": source["digest"],
        "recipe_sha256": job.get("recipe_sha256"),
        "cache_salt": job.get("cache_salt"),
        "builds": [spec["raw"] for spec in job["_builds"]],
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _build_cache_key(base_key: str, spec: Dict[str, Any]) -> str:
    payload = {"base": base_key, "name": spec["name"], "argv": spec["argv"], "cwd": str(spec["cwd"]),
               "env": spec["env"], "outputs": [str(p) for p in spec["outputs"]]}
    return _sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _artifact_name(index: int, output: Path) -> str:
    return "{:02d}_{}".format(index, output.name or "output")


def _try_restore_build(entry: Path, spec: Dict[str, Any], project_root: Path) -> Optional[Dict[str, Any]]:
    manifest_path = entry / "build.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if manifest.get("schema") != CACHE_SCHEMA or manifest.get("key") != entry.name:
        return None
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(spec["outputs"]):
        return None
    for item, output in zip(artifacts, spec["outputs"]):
        source = entry / "artifacts" / str(item.get("name"))
        if not source.is_file() or _sha256_file(source) != item.get("sha256"):
            return None
        if output.is_file() and _sha256_file(output) == item.get("sha256"):
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + ".weaveprofiler-cache.partial")
        shutil.copy2(str(source), str(partial))
        os.replace(str(partial), str(output))
    return manifest


def _store_build(entry: Path, cache_key: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    artifacts_dir = entry / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for index, output in enumerate(spec["outputs"]):
        if not output.is_file():
            raise FastPathError("build {} did not produce output {}".format(spec["name"], output))
        name = _artifact_name(index, output)
        dest = artifacts_dir / name
        partial = dest.with_name(dest.name + ".partial")
        shutil.copy2(str(output), str(partial))
        os.replace(str(partial), str(dest))
        artifacts.append({"output": str(output), "name": name, "sha256": _sha256_file(output)})
    manifest = {"schema": CACHE_SCHEMA, "key": cache_key, "name": spec["name"],
                "argv": spec["argv"], "cwd": str(spec["cwd"]), "env": spec["env"],
                "artifacts": artifacts, "created_at": _utc_now()}
    _atomic_write_json(entry / "build.json", manifest)
    return manifest


def _run_process(spec: Dict[str, Any], log_path: Path, process_table: Dict[str, subprocess.Popen],
                 table_lock: threading.Lock) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(spec["env"])
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace", newline="\n") as log:
        log.write("argv={}\ncwd={}\nenv_overrides={}\n\n".format(
            json.dumps(spec["argv"], ensure_ascii=False), spec["cwd"], json.dumps(spec["env"], sort_keys=True)))
        log.flush()
        try:
            proc = subprocess.Popen(spec["argv"], cwd=str(spec["cwd"]), env=env,
                                    stdout=log, stderr=subprocess.STDOUT, text=True)
        except OSError as exc:
            return {"name": spec["name"], "status": "FAIL", "exit_code": None,
                    "duration_seconds": time.monotonic() - started, "log": str(log_path),
                    "error": str(exc)}
        with table_lock:
            process_table[spec["name"]] = proc
        timed_out = False
        try:
            rc = proc.wait(timeout=spec["timeout_sec"])
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            rc = proc.wait()
        finally:
            with table_lock:
                process_table.pop(spec["name"], None)
    duration = time.monotonic() - started
    status = "PASS" if rc == 0 and not timed_out else "FAIL"
    return {"name": spec["name"], "status": status, "exit_code": rc,
            "timed_out": timed_out, "duration_seconds": round(duration, 3), "log": str(log_path)}


def _terminate_other_processes(process_table: Dict[str, subprocess.Popen], table_lock: threading.Lock,
                               keep: Optional[str] = None) -> None:
    with table_lock:
        procs = list(process_table.items())
    for name, proc in procs:
        if name == keep:
            continue
        if proc.poll() is None:
            proc.terminate()
    for name, proc in procs:
        if name == keep or proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _execute_builds(specs: Sequence[Dict[str, Any]], project_root: Path, run_dir: Path,
                    cache_root: Path, base_key: str, max_parallel: int) -> List[Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    pending: List[Tuple[Dict[str, Any], Path, str]] = []
    for spec in specs:
        cache_key = _build_cache_key(base_key, spec)
        entry = cache_root / cache_key
        restored = _try_restore_build(entry, spec, project_root)
        if restored is not None:
            results[spec["name"]] = {"name": spec["name"], "status": "PASS", "cached": True,
                                     "cache_key": cache_key, "duration_seconds": 0.0,
                                     "outputs": [{"path": str(p), "sha256": _sha256_file(p)}
                                                 for p in spec["outputs"]],
                                     "cache_entry": str(entry)}
        else:
            pending.append((spec, entry, cache_key))

    process_table: Dict[str, subprocess.Popen] = {}
    table_lock = threading.Lock()
    errors: List[BaseException] = []

    def worker(spec: Dict[str, Any], entry: Path, cache_key: str) -> Dict[str, Any]:
        log_path = run_dir / "logs" / "build__{}.log".format(spec["name"])
        record = _run_process(spec, log_path, process_table, table_lock)
        record["cached"] = False
        if record["status"] != "PASS":
            _terminate_other_processes(process_table, table_lock, keep=spec["name"])
            return record
        entry.mkdir(parents=True, exist_ok=True)
        try:
            manifest = _store_build(entry, cache_key, spec)
        except Exception as exc:
            record.update({"status": "FAIL", "error": str(exc)})
            return record
        record.update({"cached": False, "cache_key": cache_key,
                       "outputs": [{"path": a["output"], "sha256": a["sha256"]}
                                   for a in manifest["artifacts"]],
                       "cache_entry": str(entry)})
        return record

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_parallel, len(pending))) as pool:
            future_map = {pool.submit(worker, spec, entry, key): spec["name"]
                          for spec, entry, key in pending}
            for future in concurrent.futures.as_completed(future_map):
                name = future_map[future]
                try:
                    results[name] = future.result()
                except BaseException as exc:
                    errors.append(exc)
                    results[name] = {"name": name, "status": "FAIL", "cached": False,
                                     "error": str(exc), "duration_seconds": 0.0}
    ordered = [results[spec["name"]] for spec in specs]
    failed = [r["name"] for r in ordered if r.get("status") != "PASS"]
    if failed:
        raise FastPathError("build failed: {}".format(", ".join(failed)), records=ordered)
    return ordered


def _execute_smoke(specs: Sequence[Dict[str, Any]], run_dir: Path, parallel: bool) -> List[Dict[str, Any]]:
    def run_one(spec: Dict[str, Any]) -> Dict[str, Any]:
        log_path = run_dir / "logs" / "smoke__{}.log".format(spec["name"])
        table: Dict[str, subprocess.Popen] = {}
        lock = threading.Lock()
        record = _run_process(spec, log_path, table, lock)
        if record["status"] == "PASS":
            missing = [str(p) for p in spec["outputs"] if not p.is_file()]
            if missing:
                record.update({"status": "FAIL", "error": "missing smoke outputs: " + ", ".join(missing)})
            else:
                record["outputs"] = [{"path": str(p), "sha256": _sha256_file(p)} for p in spec["outputs"]]
        return record

    if parallel and len(specs) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as pool:
            records = list(pool.map(run_one, specs))
    else:
        records = [run_one(spec) for spec in specs]
    failed = [r["name"] for r in records if r.get("status") != "PASS"]
    if failed:
        raise FastPathError("smoke failed: {}".format(", ".join(failed)))
    return records


def load_job(path: Path) -> Dict[str, Any]:
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise FastPathError("fast job must be a JSON object")
    if raw.get("schema") != SCHEMA:
        raise FastPathError("fast job schema must be {}".format(SCHEMA))
    project_raw = raw.get("project_root")
    if not isinstance(project_raw, str) or not project_raw:
        raise FastPathError("project_root is required")
    project_root = _resolve(project_raw, path.parent)
    if not project_root.is_dir():
        raise FastPathError("project_root is not a directory: {}".format(project_root))
    builds_raw = raw.get("build")
    if not isinstance(builds_raw, list) or not builds_raw:
        raise FastPathError("build must be a non-empty list")
    builds = [_command_spec(item, project_root, "build[{}]".format(i)) for i, item in enumerate(builds_raw)]
    names = [b["name"] for b in builds]
    if len(names) != len(set(names)):
        raise FastPathError("build names must be unique")
    output_owners: Dict[Path, str] = {}
    for spec in builds:
        for output in spec["outputs"]:
            if output in output_owners:
                raise FastPathError("output {} is produced by both {} and {}".format(
                    output, output_owners[output], spec["name"]))
            output_owners[output] = spec["name"]
    smoke_raw = raw.get("smoke", [])
    if not isinstance(smoke_raw, list):
        raise FastPathError("smoke must be a list")
    smoke = [_command_spec(item, project_root, "smoke[{}]".format(i), outputs_required=False)
             for i, item in enumerate(smoke_raw)]
    smoke_names = [s["name"] for s in smoke]
    if len(smoke_names) != len(set(smoke_names)):
        raise FastPathError("smoke names must be unique")
    # Resolve and validate selected sources now; source hashes happen later.
    sources = _collect_sources(project_root, raw)
    max_parallel = raw.get("max_parallel", 2)
    if isinstance(max_parallel, bool) or not isinstance(max_parallel, int) or max_parallel < 1:
        raise FastPathError("max_parallel must be a positive integer")
    target_raw = raw.get("target")
    if not isinstance(target_raw, dict):
        raise FastPathError("target must be an object with card, arch and language")
    return {**raw, "_path": path.resolve(), "_project_root": project_root,
            "_sources": sources, "_builds": builds, "_smoke": smoke,
            "_max_parallel": max_parallel, "_target_raw": target_raw}


def _resolve_target(job: Dict[str, Any]) -> Dict[str, Any]:
    if str(SKILL_ROOT) not in sys.path:
        sys.path.insert(0, str(SKILL_ROOT))
    try:
        from weaveprof.pipeline.core import registry
    except Exception as exc:
        raise FastPathError("cannot import target registry: {}".format(exc))
    target_raw = job["_target_raw"]
    try:
        route = registry.resolve(card=target_raw.get("card"), arch=target_raw.get("arch"),
                                 platform=target_raw.get("platform"),
                                 language=target_raw.get("language"))
    except Exception as exc:
        raise FastPathError("target routing failed: {}".format(exc))
    target = route.get("target") or {}
    if target.get("card") == "b300" and not target.get("arch"):
        raise FastPathError("B300 requires an exact arch: sm_100a or sm_103a")
    return route


def run_fast(job: Dict[str, Any], out_path: Optional[Path] = None, *, skip_smoke: bool = False) -> Dict[str, Any]:
    started = time.monotonic()
    started_at = _utc_now()
    project_root = job["_project_root"]
    route = _resolve_target(job)
    sources = job.get("_sources") or _collect_sources(project_root, job)
    cache_root = _resolve(str(job.get("cache_dir", ".weaveprofiler/cache/fast")), project_root)
    hash_cache = cache_root / "file_hashes.json"
    source = _source_digest(project_root, sources, hash_cache)
    base_key = _base_key(job, route, source)
    run_root = _resolve(str(job.get("run_root", ".weaveprofiler/runs")), project_root)
    run_dir = run_root / ("fast_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + base_key[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    result: Dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "RUNNING",
        "fast_path": True,
        "delivery_evidence": False,
        "notice": "FAST RESULT ONLY: not correctness, overhead, hardware or delivery evidence",
        "run_id": run_dir.name,
        "project_root": str(project_root),
        "target": _normalized_target(route),
        "route": route,
        "source": source,
        "cache_key": base_key,
        "builds": [],
        "smoke": [],
        "started_at": started_at,
    }
    try:
        result["builds"] = _execute_builds(job["_builds"], project_root, run_dir,
                                            cache_root, base_key, job["_max_parallel"])
        if skip_smoke:
            result["smoke_skipped"] = True
        else:
            result["smoke"] = _execute_smoke(job["_smoke"], run_dir, bool(job.get("parallel_smoke", False)))
        result["status"] = "PASS"
    except FastPathError as exc:
        result["status"] = "FAIL"
        result["error"] = str(exc)
        if exc.records:
            result["builds"] = exc.records
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = "internal error: {}".format(exc)
    result["ended_at"] = _utc_now()
    result["wall_seconds"] = round(time.monotonic() - started, 3)
    result_path = (out_path.resolve() if out_path else run_dir / "FAST.json")
    result["result_path"] = str(result_path)
    _atomic_write_json(result_path, result)
    _atomic_write_json(run_dir / "FAST.json", result)
    return result


def template() -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "project_root": "/absolute/path/to/project",
        "target": {"card": "h20", "arch": "sm_90", "language": "cuda_cpp"},
        "source_files": ["kernels/target.cu"],
        "source_globs": ["profile/device_record/**/*.cuh"],
        "recipe_sha256": None,
        "cache_salt": None,
        "cache_dir": ".weaveprofiler/cache/fast",
        "run_root": ".weaveprofiler/runs",
        "max_parallel": 2,
        "parallel_smoke": False,
        "build": [
            {"name": "off", "argv": ["bash", "build.sh", "off"],
             "cwd": ".", "env": {"FI_LEVEL": "0"},
             "outputs": ["build/off/libkernel.so"], "timeout_sec": 1800},
            {"name": "full", "argv": ["bash", "build.sh", "full"],
             "cwd": ".", "env": {"FI_LEVEL": "2", "FI_TIMER": "1"},
             "outputs": ["build/full/libkernel.so"], "timeout_sec": 1800},
        ],
        "smoke": [
            {"name": "off_smoke", "argv": ["bash", "smoke.sh", "off"], "cwd": ".",
             "env": {}, "timeout_sec": 600},
            {"name": "full_smoke", "argv": ["bash", "smoke.sh", "full"], "cwd": ".",
             "env": {}, "timeout_sec": 600},
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="weaveprofiler fast build/smoke path")
    sub = parser.add_subparsers(dest="command", required=True)
    fast = sub.add_parser("fast", help="run fast off/full parallel build and smoke")
    fast.add_argument("--job", required=True, help="fast job JSON")
    fast.add_argument("--out", help="result JSON path (default: run_dir/FAST.json)")
    fast.add_argument("--no-smoke", action="store_true", help="build only; skip smoke commands")
    sub.add_parser("template", help="print a fast job template")
    args = parser.parse_args(argv)
    if args.command == "template":
        print(json.dumps(template(), indent=2, ensure_ascii=False))
        return 0
    try:
        job = load_job(Path(args.job).expanduser().resolve())
        result = run_fast(job, Path(args.out).expanduser() if args.out else None,
                          skip_smoke=bool(args.no_smoke))
    except FastPathError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": result["status"], "result": result["result_path"],
                      "wall_seconds": result["wall_seconds"], "cache_key": result["cache_key"],
                      "delivery_evidence": False}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
