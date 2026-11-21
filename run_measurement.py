#!/usr/bin/env python3
"""Fail-closed, contract-driven measurement pipeline."""
from __future__ import print_function

import argparse
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


SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_STAGES = {
    "sw": ("offline_selftest", "correctness", "overhead", "sw_capture", "perfetto"),
    "hw": ("offline_selftest", "correctness", "overhead", "hw_capture", "perfetto"),
    "sw+hw": ("offline_selftest", "correctness", "overhead", "sw_capture",
              "hw_capture", "perfetto"),
}
STAGE_KINDS = {"offline_selftest": "cpu", "correctness": "gpu", "overhead": "gpu",
               "sw_capture": "gpu", "hw_capture": "gpu", "perfetto": "cpu"}
FORBIDDEN_PREFIXES = ("--shell", "--skip")
# Defaults for the optional allowed_gpu_ids whitelist idle policy (defect 2).
# When a contract does not declare allowed_gpu_ids these are never consulted and
# behaviour is byte-identical to the original whole-machine gpu_idle_command wait.
GPU_IDLE_DEFAULT_UTIL_MAX = 10.0
GPU_IDLE_DEFAULT_POLL_INTERVAL = 5.0
# O7: default idle wait is two minutes (wait-only, never kills); contracts
# may still set gpu_idle_policy.timeout_sec explicitly.
GPU_IDLE_DEFAULT_TIMEOUT = 120.0


class PipelineError(Exception):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as error:
        raise PipelineError("cannot read JSON {}: {}".format(path, error))


def _write_json_atomic(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    data = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(partial), str(path))


def _absolute(value, base):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    return path.resolve()


def _inside(path, directory):
    try:
        return os.path.commonpath([str(Path(path).resolve()), str(Path(directory).resolve())]) == str(Path(directory).resolve())
    except ValueError:
        return False


def _check_argv(argv, where):
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise PipelineError("{} must be a non-empty argv string list".format(where))
    for token in argv:
        lowered = token.strip().lower()
        if lowered in ("shell", "skip") or any(lowered.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            raise PipelineError("{} contains forbidden shell/skip token".format(where))
    return list(argv)


def _timeout(value, where):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise PipelineError("{}.timeout_sec must be a positive number".format(where))
    return float(value)


def _command_config(config, where, default_cwd, default_env=None):
    if isinstance(config, list):
        argv, options = config, {}
    elif isinstance(config, dict):
        if any("shell" in str(key).lower() or "skip" in str(key).lower() for key in config):
            raise PipelineError("{} contains forbidden shell/skip configuration".format(where))
        argv, options = config.get("argv", config.get("command")), config
    else:
        raise PipelineError("{} must be an argv list or command object".format(where))
    argv = _check_argv(argv, where + ".argv")
    cwd = options.get("cwd", default_cwd)
    if not isinstance(cwd, (str, os.PathLike)):
        raise PipelineError("{}.cwd must be a path string".format(where))
    raw_env = options.get("env", {})
    if not isinstance(raw_env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in raw_env.items()):
        raise PipelineError("{}.env must map strings to strings".format(where))
    env = dict(default_env or {})
    env.update(raw_env)
    return {"argv": argv, "cwd": _absolute(cwd, default_cwd), "env": env,
            "env_overrides": dict(raw_env), "timeout_sec": _timeout(options.get("timeout_sec"), where)}


def _normalize_stage_name(raw):
    name = raw.strip().lower().replace("-", "_").replace("+", "_")
    return {"sw_capture_sw": "sw_capture", "sw_capture_sw_hw": "sw_capture",
            "hw_capture_hw": "hw_capture", "hw_capture_sw_hw": "hw_capture"}.get(name, name)


def _stage_name(spec):
    raw = spec.get("name", spec.get("id", spec.get("stage")))
    if not isinstance(raw, str) or not raw.strip():
        raise PipelineError("each execution stage requires name")
    return _normalize_stage_name(raw)


def _output_paths(spec, base):
    outputs = spec.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise PipelineError("stage {} must declare non-empty outputs".format(spec.get("name", "unknown")))
    result = []
    for item in outputs:
        value = item if isinstance(item, str) else item.get("path", item.get("absolute_path")) if isinstance(item, dict) else None
        if not isinstance(value, str) or not value:
            raise PipelineError("stage {} has an invalid output path".format(spec.get("name", "unknown")))
        result.append(_absolute(value, base))
    return result


def _validate_contract(contract, default_cwd=None):
    if not isinstance(contract, dict):
        raise PipelineError("contract must be a JSON object")
    if contract.get("example_only") is True or contract.get("production_validation_expected") == "REJECT":
        raise PipelineError("example-only or explicitly rejected contract cannot be measured")
    capture = contract.get("capture")
    if not isinstance(capture, dict) or capture.get("information_bundle") not in BUNDLE_STAGES:
        raise PipelineError("capture.information_bundle must be sw, hw, or sw+hw")
    bundle = capture["information_bundle"]
    execution = contract.get("execution")
    if not isinstance(execution, dict):
        raise PipelineError("contract.execution is required")
    stages = execution.get("stages")
    if not isinstance(stages, list):
        raise PipelineError("execution.stages must be a list")
    normalized, seen = [], set()
    for index, spec in enumerate(stages):
        if not isinstance(spec, dict):
            raise PipelineError("execution.stages[{}] must be an object".format(index))
        if any("shell" in str(key).lower() or "skip" in str(key).lower() for key in spec):
            raise PipelineError("execution.stages[{}] contains forbidden shell/skip field".format(index))
        name = _stage_name(spec)
        if name in seen:
            raise PipelineError("duplicate stage: {}".format(name))
        seen.add(name)
        if name not in STAGE_KINDS:
            raise PipelineError("unknown stage: {}".format(name))
        command = _command_config(spec, "stage {}".format(name), spec.get("cwd", default_cwd or "."), os.environ)
        kind = spec.get("kind")
        if kind != STAGE_KINDS[name]:
            raise PipelineError("stage {}.kind must be {}".format(name, STAGE_KINDS[name]))
        command["outputs"] = spec.get("outputs")
        _output_paths(spec, command["cwd"])
        inputs = _input_paths(spec, command["cwd"], name)
        normalized.append({"name": name, "spec": spec, "command": command, "kind": kind, "inputs": inputs})
    expected = BUNDLE_STAGES[bundle]
    actual = tuple(item["name"] for item in normalized)
    if actual != expected:
        raise PipelineError("stage order for {} must be {}; got {}".format(bundle, " -> ".join(expected), " -> ".join(actual)))
    for item in normalized:
        required = item["spec"].get("gpus_required")
        if required is not None:
            if isinstance(required, bool) or not isinstance(required, int) or required < 1:
                raise PipelineError("stage {}.gpus_required must be a positive integer".format(item["name"]))
            item["gpus_required"] = required
    parallel_groups = _parallel_groups(execution.get("parallel_groups"), normalized)
    allowed_gpu_ids, _policy = _allowed_gpu_config(execution)
    _gpu_lease_reuse_config(execution)
    idle = execution.get("gpu_idle_command")
    if allowed_gpu_ids is None:
        _command_config(idle, "execution.gpu_idle_command", ".", {})
    elif idle is not None:
        _command_config(idle, "execution.gpu_idle_command", ".", {})
    return execution, bundle, normalized, expected, parallel_groups


def _input_paths(spec, base, name):
    """Optional stage.inputs: files the stage actually consumes (scripts, ABI,
    input data).  Stages without the field keep the legacy outputs-only reuse."""
    raw = spec.get("inputs")
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) and item for item in raw):
        raise PipelineError("stage {} inputs must be a non-empty list of path strings".format(name))
    return [_absolute(item, base) for item in raw]


def _hash_inputs(paths):
    return {str(Path(path).resolve()): (_sha256(path) if Path(path).is_file() else None) for path in paths}


def _inputs_match(record, paths):
    recorded = record.get("input_sha256") if isinstance(record, dict) else None
    if not isinstance(recorded, dict):
        return False
    for path in paths:
        key = str(Path(path).resolve())
        expected = recorded.get(key)
        if not isinstance(expected, str) or not Path(key).is_file():
            return False
        try:
            if _sha256(key) != expected:
                return False
        except OSError:
            return False
    return True


def _stage_inputs_reusable(item, prior):
    declared = item.get("inputs")
    if declared is None:
        return True
    return _inputs_match(prior, declared)


# Component-key mapping used to derive default stage inputs for legacy
# contracts that carry a components registry but no stage inputs.  Shared
# runtime pieces (harness, entrypoint, ABI, symmetric-memory runtime) map to
# every GPU stage that executes the workload; stage-specific tools map to
# their own stage only.  Unknown component keys map to nothing.
_STAGE_COMPONENT_KEYS = {
    "offline_selftest": ("offline_selftests", "materialized_abi", "template_abi"),
    "correctness": ("correctness_validator", "measurement_harness", "profile_entrypoint",
                    "symmetric_memory_bypass", "materialized_abi", "template_abi"),
    "overhead": ("overhead_validator", "measurement_harness", "profile_entrypoint",
                 "symmetric_memory_bypass", "materialized_abi", "template_abi"),
    "sw_capture": ("profile_entrypoint", "measurement_harness", "decoder", "decoder_entrypoint",
                   "symmetric_memory_bypass", "materialized_abi", "template_abi"),
    "hw_capture": ("hw_capture", "measurement_harness", "metrics_definition",
                   "symmetric_memory_bypass", "materialized_abi", "template_abi"),
    "perfetto": ("perfetto_pipeline", "perfetto_emitter", "sw_hw_aligner", "hw_extract",
                 "metrics_definition"),
}


def _default_stage_inputs(name, command, contract):
    """Default inputs for legacy contracts without stage inputs: the stage
    script (first .py argv token) plus the components mapped to this stage.
    Returns None when nothing can be determined (caller must rerun)."""
    inputs = []
    script = next((token for token in command["argv"] if token.endswith(".py")), None)
    if script:
        inputs.append(str(_absolute(script, command["cwd"])))
    components = contract.get("components") if isinstance(contract, dict) else None
    if isinstance(components, dict):
        for key in _STAGE_COMPONENT_KEYS.get(name, ()):
            entry = components.get(key)
            if isinstance(entry, dict):
                path = entry.get("absolute_path")
                if isinstance(path, str) and path:
                    inputs.append(str(_absolute(path, command["cwd"])))
    seen, result = set(), []
    for path in inputs:
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result or None


def _stage_record_inputs(item, contract):
    """input_sha256 recorded at execution time: declared inputs when present,
    otherwise the inferred defaults (recorded as evidence for future inherit)."""
    declared = item.get("inputs")
    paths = declared if declared is not None else (_default_stage_inputs(item["name"], item["command"], contract) or [])
    return _hash_inputs(paths)


def _resume_invalidation_reason(item, prior):
    if not isinstance(prior, dict) or prior.get("status") not in ("PASS", "REUSED"):
        return "previous stage not reusable"
    if not _outputs_match(prior):
        return "previous output hash mismatch"
    return "previous input hash mismatch"


def _build_inherit_ctx(old_state, old_contract, old_specs, old_run_dir, old_contract_path):
    """Evidence for --inherit: sha256 records that can prove an input file is
    byte-identical to what the old run consumed, checked in priority order
    (a) old state input_sha256, (b) old contract code registry, (c) old state
    output_sha256 (upstream products)."""
    code_entries = {}

    def visit(node):
        if isinstance(node, dict):
            path = node.get("absolute_path")
            if not isinstance(path, str):
                path = node.get("path")
            sha = node.get("sha256")
            if isinstance(path, str) and isinstance(sha, str) and os.path.isabs(path):
                code_entries.setdefault(str(Path(path).resolve()), sha)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(old_contract)
    input_entries, output_entries = {}, {}
    old_rows = old_state.get("stages") if isinstance(old_state, dict) else None
    for row in (old_rows or {}).values():
        if not isinstance(row, dict):
            continue
        for key, value in (row.get("input_sha256") or {}).items():
            if isinstance(value, str):
                input_entries[key] = value
        for key, value in (row.get("output_sha256") or {}).items():
            if isinstance(value, str):
                output_entries[key] = value
    return {"old_run_dir": old_run_dir, "old_contract_path": old_contract_path,
            "old_specs": {item["name"]: item for item in old_specs}, "old_state": old_state,
            "evidence_inputs": input_entries, "evidence_code": code_entries,
            "evidence_outputs": output_entries}


def _try_inherit(item, row, ctx, contract, run_dir, contract_path):
    """Reuse one stage from an older run dir whose contract hash differs.

    Returns True only when argv/cwd/env/outputs match after run_dir/contract
    normalization and every input file is proven byte-identical via recorded
    evidence; then hard-links (falls back to copy2) the old outputs into the
    new run dir.  Fail-closed: any unproven input returns False (stage and its
    successors rerun); a recorded-but-mismatching old output raises."""
    name = item["name"]
    old_item = ctx["old_specs"].get(name)
    if old_item is None or old_item["kind"] != item["kind"]:
        return False
    old_rows = ctx["old_state"].get("stages") or {}
    old_row = old_rows.get(name)
    if not isinstance(old_row, dict) or old_row.get("status") not in ("PASS", "REUSED"):
        return False
    old_run_dir, old_contract_path = ctx["old_run_dir"], ctx["old_contract_path"]

    def normalize(token):
        token = token.replace(str(old_contract_path), str(contract_path))
        return token.replace(str(old_run_dir), str(run_dir))

    old_command, command = old_item["command"], item["command"]
    if [normalize(token) for token in old_command["argv"]] != list(command["argv"]):
        return False
    if normalize(str(old_command["cwd"])) != str(command["cwd"]):
        return False
    old_env = {key: normalize(value) for key, value in old_command.get("env_overrides", {}).items()}
    if old_env != dict(command.get("env_overrides", {})):
        return False
    if old_command.get("timeout_sec") != command.get("timeout_sec"):
        return False
    old_outputs = _output_paths(old_item["spec"], old_command["cwd"])
    new_outputs = _output_paths(item["spec"], command["cwd"])
    if [normalize(str(path)) for path in old_outputs] != [str(path) for path in new_outputs]:
        return False
    inputs = item.get("inputs")
    if inputs is None:
        inputs = _default_stage_inputs(name, command, contract)
    if not inputs:
        return False
    evidence, input_hashes = [], {}
    for path in inputs:
        path = str(Path(path).resolve())
        if not Path(path).is_file():
            return False
        current = _sha256(path)
        candidates = [path]
        if _inside(path, run_dir):
            relative = Path(path).relative_to(Path(run_dir).resolve())
            candidates.append(str((Path(old_run_dir) / relative).resolve()))
        hit = None
        for key in candidates:
            for source, entries in (("state_input", ctx["evidence_inputs"]),
                                    ("contract_code", ctx["evidence_code"]),
                                    ("state_output", ctx["evidence_outputs"])):
                if key in entries:
                    hit = (entries[key], source, key)
                    break
            if hit is not None:
                break
        if hit is None or hit[0].lower() != current.lower():
            return False
        evidence.append({"input": path, "sha256": current, "source": hit[1], "evidence_path": hit[2]})
        input_hashes[path] = current
    old_hashes = old_row.get("output_sha256")
    if not isinstance(old_hashes, dict) or not old_hashes:
        return False
    ordered_old = []
    for old_path in old_outputs:
        key = str(old_path)
        recorded = old_hashes.get(key)
        if not isinstance(recorded, str):
            return False
        if not Path(key).is_file():
            return False
        if _sha256(key) != recorded:
            raise PipelineError("inherit evidence tampered: {} content differs from recorded output sha256".format(key))
        ordered_old.append((key, recorded))
    for (old_file, recorded), new_path in zip(ordered_old, new_outputs):
        new_path = Path(new_path)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(old_file, str(new_path))
        except OSError:
            shutil.copy2(old_file, str(new_path))
        if _sha256(new_path) != recorded:
            raise PipelineError("inherited output verification failed for {}".format(new_path))
    row.update({"status": "REUSED", "reused": True, "inherited_from": str(old_run_dir),
                "output_sha256": {str(Path(path).resolve()): recorded for (old_file, recorded), path in zip(ordered_old, new_outputs)},
                "input_sha256": input_hashes, "inherit_evidence": evidence,
                "duration_seconds": old_row.get("duration_seconds"),
                "started_at": old_row.get("started_at"), "ended_at": old_row.get("ended_at"),
                "exit_code": 0, "log": old_row.get("log")})
    return True


def _candidate_trace(contract, execution, run_dir):
    value = execution.get("candidate_trace")
    if value is None:
        artifacts = contract.get("artifacts")
        value = artifacts.get("candidate_perfetto") if isinstance(artifacts, dict) else None
    if not isinstance(value, str) or not value:
        raise PipelineError("execution.candidate_trace or artifacts.candidate_perfetto is required")
    candidate = _absolute(value, run_dir)
    if not _inside(candidate, run_dir):
        raise PipelineError("candidate trace must be inside run_dir")
    return candidate


def _hash_outputs(paths):
    return {str(Path(path).resolve()): (_sha256(path) if Path(path).is_file() else None) for path in paths}


def _outputs_match(record):
    expected = record.get("output_sha256", {}) if isinstance(record, dict) else {}
    if not expected:
        return False
    try:
        return all(value is not None and Path(path).is_file() and
                   _sha256(path) == value for path, value in expected.items())
    except OSError:
        return False


def _log_path(run_dir, label):
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in label)
    return Path(run_dir).resolve() / "logs" / (safe + ".log")


def _run_command(command, log_path):
    start, clock = _now(), time.time()
    log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"argv": list(command["argv"]), "cwd": str(command["cwd"]),
              "env_overrides": dict(command.get("env_overrides", {})), "started_at": start,
              "timeout_sec": command.get("timeout_sec"), "log": str(log_path)}
    try:
        # Stream child output directly to disk.  The previous subprocess.run
        # path retained the entire compiler/profiler log in RAM and only then
        # copied it to disk, adding a second large allocation and delaying
        # fail-closed detection.  Only this child is terminated on timeout.
        with log_path.open("wb") as log_handle:
            process = subprocess.Popen(command["argv"], cwd=str(command["cwd"]), env=command["env"],
                                       stdout=log_handle, stderr=subprocess.STDOUT)
            try:
                record["exit_code"] = process.wait(timeout=command.get("timeout_sec"))
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log_handle.write(b"\nTIMEOUT\n")
                record.update({"exit_code": None, "timed_out": True})
    except Exception as error:
        log_path.write_text("{}: {}\n".format(type(error).__name__, error), encoding="utf-8")
        record.update({"exit_code": None, "exception": str(error)})
    record.update({"ended_at": _now(), "duration_seconds": time.time() - clock})
    return record


def _query_gpus_nvidia_smi(timeout=30.0):
    """Return a per-GPU snapshot via nvidia-smi.  Read-only; never kills."""
    argv = ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    except Exception as error:
        raise PipelineError("cannot run nvidia-smi: {}".format(error))
    if result.returncode != 0:
        raise PipelineError("nvidia-smi query failed with exit code {}".format(result.returncode))
    return _parse_nvidia_smi((result.stdout or b"").decode("utf-8", "replace"))


def _parse_nvidia_smi(text):
    gpus = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpus.append({"index": int(parts[0]), "memory_used_mib": float(parts[1]),
                         "utilization_percent": float(parts[2])})
        except (TypeError, ValueError):
            continue
    return gpus


def _gpu_is_free(gpu, util_max, mem_max):
    """A GPU is idle when utilisation is within threshold; memory is only used
    when the contract opts in (GPUs commonly hold cached memory while idle)."""
    utilization = gpu.get("utilization_percent")
    if utilization is None or float(utilization) > float(util_max):
        return False
    if mem_max is not None and float(gpu.get("memory_used_mib", 0.0)) > float(mem_max):
        return False
    return True


def _pick_free_gpu(gpus, allowed_ids, util_max, mem_max):
    by_index = {gpu.get("index"): gpu for gpu in gpus}
    for gpu_id in allowed_ids:
        gpu = by_index.get(gpu_id)
        if gpu is not None and _gpu_is_free(gpu, util_max, mem_max):
            return gpu_id
    return None


def _pick_free_gpus(gpus, allowed_ids, count, util_max, mem_max, reserved):
    by_index = {gpu.get("index"): gpu for gpu in gpus}
    chosen = []
    for gpu_id in allowed_ids:
        if gpu_id in reserved:
            continue
        gpu = by_index.get(gpu_id)
        if gpu is not None and _gpu_is_free(gpu, util_max, mem_max):
            chosen.append(gpu_id)
            if len(chosen) == count:
                return chosen
    return None


def _wait_for_whitelisted_gpus(allowed_ids, policy, log_path, count, reservation, query_fn=None, sleep_fn=None):
    """Wait (never kill) until count disjoint whitelisted GPUs are idle.

    reservation = {"lock": threading.Lock(), "taken": set()} is shared by one
    parallel group so concurrently scheduled stages never bind the same card.
    Cards stay reserved for the whole group; nothing is released early."""
    query_fn = query_fn or _query_gpus_nvidia_smi
    sleep_fn = sleep_fn or time.sleep
    start, clock = _now(), time.time()
    log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"mode": "allowed_gpu_ids", "allowed_gpu_ids": list(allowed_ids),
              "policy": dict(policy), "gpus_required": count, "started_at": start, "polls": []}
    lines, deadline, chosen = [], clock + policy["timeout_sec"], None
    while True:
        try:
            gpus = query_fn()
        except Exception as error:
            gpus = []
            lines.append("{} query error: {}".format(_now(), error))
        snapshot = [{"index": g.get("index"), "memory_used_mib": g.get("memory_used_mib"),
                     "utilization_percent": g.get("utilization_percent")}
                    for g in gpus if g.get("index") in allowed_ids]
        record["polls"].append({"at": _now(), "whitelisted": snapshot})
        lines.append("{} whitelisted={}".format(_now(), snapshot))
        with reservation["lock"]:
            chosen = _pick_free_gpus(gpus, allowed_ids, count, policy["utilization_max_percent"], policy["memory_max_mib"], reservation["taken"])
            if chosen is not None:
                reservation["taken"].update(chosen)
        if chosen is not None:
            lines.append("selected GPUs {} (never killed any process)".format(chosen))
            record["exit_code"] = 0
            record["chosen_gpus"] = list(chosen)
            record["chosen_gpu"] = chosen[0]
            break
        if time.time() >= deadline:
            lines.append("TIMEOUT: {} disjoint whitelisted GPUs did not become idle within {}s".format(count, policy["timeout_sec"]))
            record.update({"exit_code": None, "timed_out": True})
            break
        sleep_fn(policy["poll_interval_sec"])
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record.update({"log": str(log_path), "ended_at": _now(),
                   "duration_seconds": time.time() - clock})
    return record


def _parallel_groups(raw, normalized):
    """Validate optional execution.parallel_groups: lists of two or more known
    GPU stage names, each stage in at most one group."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise PipelineError("execution.parallel_groups must be a list of stage name lists")
    by_name = {item["name"]: item for item in normalized}
    groups, seen = [], set()
    for group in raw:
        if not isinstance(group, list) or len(group) < 2 or not all(isinstance(name, str) for name in group):
            raise PipelineError("each execution.parallel_groups entry must list at least two stage names")
        names = []
        for raw_name in group:
            name = _normalize_stage_name(raw_name)
            if name not in by_name:
                raise PipelineError("parallel group references unknown stage: {}".format(raw_name))
            if by_name[name]["kind"] != "gpu":
                raise PipelineError("parallel group stage {} is not a GPU stage".format(name))
            if name in seen:
                raise PipelineError("stage {} appears in multiple parallel groups".format(name))
            seen.add(name)
            names.append(name)
        groups.append(tuple(names))
    return groups


def _active_parallel_groups(parallel_groups, spec_by_name, allowed_gpu_ids):
    """A group runs concurrently only when every member declares a positive
    gpus_required and the whitelist can give each member a disjoint card set
    (sum of gpus_required <= whitelist size); otherwise it silently falls back
    to the serial path.  Rank counts are never guessed here: contracts declare
    gpus_required per stage."""
    if not parallel_groups:
        return [], "no parallel_groups declared"
    if allowed_gpu_ids is None or len(allowed_gpu_ids) < 2:
        return [], "allowed_gpu_ids missing or has fewer than two cards"
    active, reasons = [], []
    for group in parallel_groups:
        required = []
        for name in group:
            value = spec_by_name[name].get("gpus_required")
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                required = None
                break
            required.append(value)
        if required is None:
            reasons.append("group {} lacks gpus_required".format(list(group)))
            continue
        if sum(required) > len(allowed_gpu_ids):
            reasons.append("group {} needs {} GPUs but whitelist has {}".format(list(group), sum(required), len(allowed_gpu_ids)))
            continue
        active.append(group)
    if active:
        return active, "active"
    return [], "; ".join(reasons) or "no groups declared"


def _group_member_worker(item, index, row, run_dir, allowed_gpu_ids, gpu_idle_policy, reservation, errors, contract):
    """Thread body for one member of an active parallel group.  Never raises;
    failures land in row + errors and are enforced after the join in stage
    order with the same BLOCKED semantics as the serial path."""
    name, command = item["name"], item["command"]
    try:
        if not command["cwd"].is_dir():
            raise PipelineError("stage {} cwd does not exist".format(name))
        idle_log = _log_path(run_dir, "{:02d}_{}_gpu_idle".format(index, name))
        idle_record = _wait_for_whitelisted_gpus(allowed_gpu_ids, gpu_idle_policy, idle_log,
                                                 item["gpus_required"], reservation)
        # Concurrent members share one gpu_idle.json path; serialise the atomic
        # write so two threads never race the same .partial file.
        with reservation["lock"]:
            _write_whitelist_gpu_idle_evidence(run_dir, idle_record)
        chosen = idle_record.get("chosen_gpus") or []
        if chosen:
            visible = ",".join(str(gpu) for gpu in chosen)
            command["env"] = dict(command["env"])
            command["env"]["CUDA_VISIBLE_DEVICES"] = visible
            command["env_overrides"] = dict(command.get("env_overrides", {}))
            command["env_overrides"]["CUDA_VISIBLE_DEVICES"] = visible
            row["bound_gpu"] = chosen[0]
            row["bound_gpus"] = list(chosen)
        row["gpu_idle"] = idle_record
        if idle_record.get("exit_code") != 0 or idle_record.get("timed_out"):
            row["status"] = "FAIL"
            row["reason"] = "GPU idle check failed"
            errors[name] = "stage {} GPU idle check failed".format(name)
            return
        row["input_sha256"] = _stage_record_inputs(item, contract)
        result = _run_command(command, _log_path(run_dir, "{:02d}_{}".format(index, name)))
        row.update(result)
        row["output_sha256"] = _hash_outputs(_output_paths(item["spec"], command["cwd"]))
        missing = [path for path, value in row["output_sha256"].items() if value is None]
        if result.get("exit_code") != 0 or result.get("timed_out") or missing:
            row["status"] = "FAIL"
            row["reason"] = "timeout" if result.get("timed_out") else ("missing outputs" if missing else "non-zero exit")
            errors[name] = "stage {} failed".format(name)
            return
        row["status"] = "PASS"
    except Exception as error:
        row["status"] = "FAIL"
        row["reason"] = str(error)
        errors[name] = "stage {} failed".format(name)


def _run_parallel_group(members, stage_specs, state, state_path, stage_order, run_dir,
                        allowed_gpu_ids, gpu_idle_policy, contract):
    reservation = {"lock": threading.Lock(), "taken": set()}
    errors = {}
    index_of = {item["name"]: idx for idx, item in enumerate(stage_specs)}
    threads = []
    for item in members:
        row = state["stages"][item["name"]]
        thread = threading.Thread(target=_group_member_worker,
                                  args=(item, index_of[item["name"]], row, run_dir,
                                        allowed_gpu_ids, gpu_idle_policy, reservation, errors, contract))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    _write_json_atomic(state_path, state)
    for item in members:
        name, row = item["name"], state["stages"][item["name"]]
        if row.get("status") == "FAIL":
            first_index = index_of[members[0]["name"]]
            _set_blocked(state, stage_order[first_index + 1:], "previous stage failed: {}".format(name))
            _write_json_atomic(state_path, state)
            raise PipelineError(errors.get(name) or "stage {} failed".format(name))


def _allowed_gpu_config(execution):
    """Validate and normalise execution.allowed_gpu_ids and gpu_idle_policy.

    Returns (None, {}) when the contract does not opt in, so the original
    whole-machine gpu_idle_command behaviour is preserved unchanged."""
    raw = execution.get("allowed_gpu_ids")
    if raw is None:
        return None, {}
    if not isinstance(raw, list) or not raw:
        raise PipelineError("execution.allowed_gpu_ids must be a non-empty list of GPU indices")
    allowed = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise PipelineError("execution.allowed_gpu_ids entries must be non-negative integers")
        if item not in allowed:
            allowed.append(item)
    policy_raw = execution.get("gpu_idle_policy", {})
    if not isinstance(policy_raw, dict):
        raise PipelineError("execution.gpu_idle_policy must be an object")
    util_max = policy_raw.get("utilization_max_percent", GPU_IDLE_DEFAULT_UTIL_MAX)
    if isinstance(util_max, bool) or not isinstance(util_max, (int, float)) or util_max < 0:
        raise PipelineError("execution.gpu_idle_policy.utilization_max_percent must be a non-negative number")
    mem_max = policy_raw.get("memory_max_mib")
    if mem_max is not None and (isinstance(mem_max, bool) or not isinstance(mem_max, (int, float)) or mem_max < 0):
        raise PipelineError("execution.gpu_idle_policy.memory_max_mib must be a non-negative number or null")

    def _positive(name, default):
        value = policy_raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise PipelineError("execution.gpu_idle_policy.{} must be a positive number".format(name))
        return float(value)

    policy = {"utilization_max_percent": float(util_max),
              "memory_max_mib": (float(mem_max) if mem_max is not None else None),
              "poll_interval_sec": _positive("poll_interval_sec", GPU_IDLE_DEFAULT_POLL_INTERVAL),
              "timeout_sec": _positive("timeout_sec", GPU_IDLE_DEFAULT_TIMEOUT)}
    return allowed, policy


def _gpu_lease_reuse_config(execution):
    """Return whether consecutive serial GPU stages may share one lease.

    Reuse is a pipeline-run optimisation and is enabled by default only when
    allowed_gpu_ids selected the card.  Existing contracts can opt out with
    ``execution.reuse_gpu_lease: false`` (or the equivalent
    ``execution.gpu_lease_reuse: false`` spelling).  A contract that already
    declares one of these fields must not silently change meaning.
    """
    declarations = []
    for field in ("reuse_gpu_lease", "gpu_lease_reuse"):
        if field not in execution:
            continue
        raw = execution[field]
        if isinstance(raw, dict):
            raw = raw.get("enabled")
        if not isinstance(raw, bool):
            raise PipelineError("execution.{} must be a boolean or an object with boolean enabled".format(field))
        declarations.append((field, raw))
    if declarations and len({value for _field, value in declarations}) != 1:
        raise PipelineError("conflicting execution GPU lease reuse declarations")
    if not declarations:
        return True, "default enabled for consecutive serial GPU stages"
    field, enabled = declarations[0]
    return enabled, "explicit {}={}".format(field, str(enabled).lower())


def _wait_for_whitelisted_gpu(allowed_ids, policy, log_path, query_fn=None, sleep_fn=None):
    """Wait (never kill) until at least one whitelisted GPU is idle and bind to it.

    The returned record mirrors _run_command's timing fields so the same timing
    accounting treats this as the GPU-idle wait for the stage."""
    query_fn = query_fn or _query_gpus_nvidia_smi
    sleep_fn = sleep_fn or time.sleep
    start, clock = _now(), time.time()
    log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"mode": "allowed_gpu_ids", "allowed_gpu_ids": list(allowed_ids),
              "policy": dict(policy), "started_at": start, "polls": []}
    lines, deadline, chosen = [], clock + policy["timeout_sec"], None
    while True:
        try:
            gpus = query_fn()
        except Exception as error:
            gpus = []
            lines.append("{} query error: {}".format(_now(), error))
        snapshot = [{"index": g.get("index"), "memory_used_mib": g.get("memory_used_mib"),
                     "utilization_percent": g.get("utilization_percent")}
                    for g in gpus if g.get("index") in allowed_ids]
        record["polls"].append({"at": _now(), "whitelisted": snapshot})
        lines.append("{} whitelisted={}".format(_now(), snapshot))
        chosen = _pick_free_gpu(gpus, allowed_ids, policy["utilization_max_percent"], policy["memory_max_mib"])
        if chosen is not None:
            lines.append("selected GPU {} (never killed any process)".format(chosen))
            record["exit_code"] = 0
            record["chosen_gpu"] = chosen
            break
        if time.time() >= deadline:
            lines.append("TIMEOUT: no whitelisted GPU became idle within {}s".format(policy["timeout_sec"]))
            record.update({"exit_code": None, "timed_out": True})
            break
        sleep_fn(policy["poll_interval_sec"])
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record.update({"log": str(log_path), "ended_at": _now(),
                   "duration_seconds": time.time() - clock})
    return record


def _write_whitelist_gpu_idle_evidence(run_dir, idle_record, stage_label=None):
    """Persist the G00 snapshot normally emitted by the project idle command."""
    polls = idle_record.get("polls") if isinstance(idle_record.get("polls"), list) else []
    lease = idle_record.get("lease") if isinstance(idle_record.get("lease"), dict) else {}
    acquisition_snapshot = lease.get("acquisition_snapshot")
    if polls:
        last_poll = polls[-1]
    elif isinstance(acquisition_snapshot, dict):
        last_poll = acquisition_snapshot
    else:
        last_poll = {}
    considered = last_poll.get("whitelisted") if isinstance(last_poll, dict) else []
    chosen = idle_record.get("chosen_gpu")
    idle = idle_record.get("exit_code") == 0 and chosen is not None
    snapshot = {
        "schema": "fast-instrument-gpu-idle/v1",
        "idle": idle,
        "checked_at_utc": idle_record.get("ended_at") or _now(),
        "probe": "run_measurement_allowed_gpu_ids",
        "allowed_gpu_ids": list(idle_record.get("allowed_gpu_ids") or []),
        "chosen_gpu": chosen,
        "policy": dict(idle_record.get("policy") or {}),
        "polls": len(polls),
        "query_count": lease.get("query_count", len(polls)),
        "lease": dict(lease),
        "stage": stage_label,
        "considered_gpus": considered if isinstance(considered, list) else [],
        "busy_gpus": [] if idle else (considered if isinstance(considered, list) else []),
        "note": ("reused the pipeline-run GPU lease; no new probe and no process was killed"
                 if lease.get("reused") else
                 "selected an idle whitelisted GPU; no process was killed"),
    }
    validation = Path(run_dir).resolve() / "validation"
    path = validation / "gpu_idle.json"
    _write_json_atomic(path, snapshot)
    if stage_label:
        safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(stage_label))
        _write_json_atomic(validation / "gpu_idle" / (safe + ".json"), snapshot)
    return path


def _stage_gpu_count(item):
    count = item.get("gpus_required", item.get("spec", {}).get("gpus_required", 1))
    if count is None:
        return 1
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise PipelineError("stage {} has an invalid gpus_required count".format(item.get("name", "unknown")))
    return count


def _bind_gpu_selection(command, row, idle_record):
    chosen = idle_record.get("chosen_gpus") or []
    if not chosen and idle_record.get("chosen_gpu") is not None:
        chosen = [idle_record["chosen_gpu"]]
    if not chosen:
        return
    visible = ",".join(str(gpu) for gpu in chosen)
    command["env"] = dict(command["env"])
    command["env"]["CUDA_VISIBLE_DEVICES"] = visible
    command["env_overrides"] = dict(command.get("env_overrides", {}))
    command["env_overrides"]["CUDA_VISIBLE_DEVICES"] = visible
    row["bound_gpus"] = list(chosen)
    row["bound_gpu"] = chosen[0] if len(chosen) == 1 else list(chosen)


def _new_sequential_gpu_lease(run_dir, allowed_ids, policy, enabled, reason, active_groups):
    lease_reason = reason
    if allowed_ids is None:
        enabled, lease_reason = False, "allowed_gpu_ids is not configured"
    elif not enabled:
        lease_reason = reason
    elif active_groups:
        enabled, lease_reason = False, "active parallel_groups require disjoint per-stage reservations"
    return {
        "schema": "fast-instrument-gpu-lease/v1",
        "scope": "pipeline_run",
        "run_dir": str(Path(run_dir).resolve()),
        "allowed_gpu_ids": list(allowed_ids or []),
        "policy": dict(policy or {}),
        "enabled": bool(enabled),
        "reason": lease_reason,
        "id": "gpu-lease-{}".format(hashlib.sha256(str(Path(run_dir).resolve()).encode("utf-8")).hexdigest()[:16]),
        "chosen_gpus": None,
        "gpus_required": None,
        "acquired_by": None,
        "acquired_at": None,
        "acquisition_record": None,
        "query_count": 0,
        "acquisition_count": 0,
        "reused_by": [],
        "last_stage_index": None,
        "reservation": {"lock": threading.Lock(), "taken": set()},
    }


def _release_sequential_gpu_lease(lease, reason):
    if not lease or not lease.get("chosen_gpus"):
        if lease:
            lease["chosen_gpus"] = None
            lease["gpus_required"] = None
        return
    released_gpus = list(lease["chosen_gpus"])
    reservation = lease.get("reservation") or {}
    with reservation.get("lock", threading.Lock()):
        reservation.get("taken", set()).difference_update(released_gpus)
    lease.update({"chosen_gpus": None, "gpus_required": None, "acquired_by": None,
                  "acquired_at": None, "acquisition_record": None,
                  "last_released_gpus": released_gpus, "release_reason": reason})


def _sequential_gpu_lease_state(lease):
    if not lease:
        return {"schema": "fast-instrument-gpu-lease/v1", "enabled": False,
                "reason": "no GPU lease context"}
    keys = ("schema", "scope", "run_dir", "allowed_gpu_ids", "enabled", "reason", "id",
            "chosen_gpus", "gpus_required", "acquired_by", "acquired_at", "query_count",
            "acquisition_count", "reused_by", "last_released_gpus", "release_reason")
    result = {key: lease.get(key) for key in keys if key in lease}
    result["held"] = bool(lease.get("chosen_gpus"))
    return result


def _reuse_sequential_gpu_record(lease, item, stage_label):
    acquisition = lease.get("acquisition_record") or {}
    polls = acquisition.get("polls") if isinstance(acquisition.get("polls"), list) else []
    acquisition_snapshot = polls[-1] if polls else {}
    now = _now()
    record = {
        "mode": "allowed_gpu_ids",
        "allowed_gpu_ids": list(lease.get("allowed_gpu_ids") or []),
        "policy": dict(lease.get("policy") or {}),
        "gpus_required": _stage_gpu_count(item),
        "started_at": now,
        "ended_at": now,
        "duration_seconds": 0.0,
        "polls": [],
        "exit_code": 0,
        "chosen_gpu": lease["chosen_gpus"][0],
        "chosen_gpus": list(lease["chosen_gpus"]),
        "stage": stage_label,
        "lease": {
            "schema": lease["schema"],
            "id": lease["id"],
            "scope": "pipeline_run",
            "reused": True,
            "acquired_by": lease.get("acquired_by"),
            "acquired_at": lease.get("acquired_at"),
            "query_count": 0,
            "acquisition_polls": len(polls),
            "acquisition_snapshot": acquisition_snapshot,
        },
    }
    return record


def _write_gpu_lease_reuse_log(log_path, record):
    log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lease = record.get("lease") or {}
    lines = [
        "{} reused {} from stage {} (scope=pipeline_run)".format(
            _now(), lease.get("id"), lease.get("acquired_by")),
        "chosen_gpus={} required={}; no GPU probe, no process killed".format(
            record.get("chosen_gpus"), record.get("gpus_required")),
        "reused stage={}".format(record.get("stage")),
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record["log"] = str(log_path)


def _acquire_sequential_stage_gpus(item, index, row, run_dir, allowed_gpu_ids,
                                   gpu_idle_policy, lease):
    """Select GPUs for a serial GPU stage, reusing one run-scoped lease.

    The lease is reusable only for adjacent serial GPU stages requesting the
    same GPU count.  A non-GPU stage, an active parallel group, a changed
    requirement, a resume boundary without an acquired lease, or an explicit
    opt-out forces the original wait path.  Reuse never broadens allowed_gpu_ids
    and never spans pipeline runs.
    """
    name, count = item["name"], _stage_gpu_count(item)
    stage_label = "{:02d}_{}".format(index, name)
    idle_log = _log_path(run_dir, "{}_gpu_idle".format(stage_label))
    reused = bool(
        lease and lease.get("enabled")
        and lease.get("chosen_gpus")
        and lease.get("gpus_required") == count
        and lease.get("last_stage_index") == index - 1
    )
    if reused:
        if not set(lease["chosen_gpus"]) <= set(allowed_gpu_ids):
            raise PipelineError("stage {} GPU lease is outside allowed_gpu_ids".format(name))
        idle_record = _reuse_sequential_gpu_record(lease, item, stage_label)
        _write_gpu_lease_reuse_log(idle_log, idle_record)
        lease["reused_by"].append(name)
        lease["last_stage_index"] = index
    else:
        if lease and lease.get("enabled"):
            _release_sequential_gpu_lease(lease, "new lease for stage {}".format(name))
            reservation = lease["reservation"]
            idle_record = _wait_for_whitelisted_gpus(
                allowed_gpu_ids, gpu_idle_policy, idle_log, count, reservation)
            lease["query_count"] += len(idle_record.get("polls") or [])
            lease["acquisition_count"] += 1
            lease["last_stage_index"] = index
            if idle_record.get("exit_code") == 0:
                chosen = idle_record.get("chosen_gpus") or []
                if not chosen or not set(chosen) <= set(allowed_gpu_ids):
                    raise PipelineError("stage {} GPU lease selection escaped allowed_gpu_ids".format(name))
                lease.update({
                    "chosen_gpus": list(chosen),
                    "gpus_required": count,
                    "acquired_by": name,
                    "acquired_at": idle_record.get("started_at"),
                    "acquisition_record": idle_record,
                })
                lease.pop("release_reason", None)
            idle_record["lease"] = {
                "schema": lease["schema"],
                "id": lease["id"],
                "scope": "pipeline_run",
                "reused": False,
                "acquired_by": name,
                "acquired_at": idle_record.get("started_at"),
                "query_count": len(idle_record.get("polls") or []),
            }
        else:
            reservation = {"lock": threading.Lock(), "taken": set()}
            idle_record = _wait_for_whitelisted_gpus(
                allowed_gpu_ids, gpu_idle_policy, idle_log, count, reservation)
            idle_record["lease"] = {
                "schema": "fast-instrument-gpu-lease/v1",
                "scope": "stage",
                "reused": False,
                "query_count": len(idle_record.get("polls") or []),
            }
        idle_record["stage"] = stage_label
    row["gpu_idle"] = idle_record
    _bind_gpu_selection(item["command"], row, idle_record)
    _write_whitelist_gpu_idle_evidence(run_dir, idle_record, stage_label)
    return idle_record


def _prepare_serial_stage_gpus(item, index, row, run_dir, allowed_gpu_ids,
                               gpu_idle_policy, lease, idle_command):
    if allowed_gpu_ids is not None:
        return _acquire_sequential_stage_gpus(
            item, index, row, run_dir, allowed_gpu_ids, gpu_idle_policy, lease)
    idle_log = _log_path(run_dir, "{:02d}_{}_gpu_idle".format(index, item["name"]))
    idle_record = _run_command(idle_command, idle_log)
    idle_record["stage"] = "{:02d}_{}".format(index, item["name"])
    row["gpu_idle"] = idle_record
    return idle_record


def _note_stage_for_sequential_gpu_lease(lease, index, kind, name):
    if not lease or not lease.get("enabled"):
        return
    if kind != "gpu":
        _release_sequential_gpu_lease(lease, "non-GPU stage {} ended the serial GPU lane".format(name))
        lease["last_stage_index"] = None


def _phase_timing(name, kind, row):
    """Build one TIMING.json phase entry, separating GPU wait from execution."""
    row = row or {}
    execute = row.get("duration_seconds")
    started, ended = row.get("started_at"), row.get("ended_at")
    gpu_wait = None
    if kind == "gpu":
        idle = row.get("gpu_idle") or {}
        gpu_wait = idle.get("duration_seconds")
        if idle.get("started_at"):
            started = idle.get("started_at")
    if isinstance(gpu_wait, (int, float)) and isinstance(execute, (int, float)):
        wall = gpu_wait + execute
    else:
        wall = execute
    entry = {"name": name, "kind": kind, "status": row.get("status"),
             "started_at": started, "ended_at": ended, "wall_seconds": wall,
             "execute_seconds": execute, "gpu_wait_seconds": gpu_wait,
             "exit_code": row.get("exit_code"), "timed_out": bool(row.get("timed_out"))}
    chosen = (row.get("gpu_idle") or {}).get("chosen_gpu")
    if kind == "gpu" and chosen is not None:
        entry["bound_gpu"] = chosen
    return entry


def _build_timing(state, stage_order):
    phases = []
    pre = state.get("post_stages", {}).get("preflight")
    if pre:
        phases.append(_phase_timing("preflight", "cpu", pre))
    for name in stage_order:
        row = state.get("stages", {}).get(name)
        if row:
            phases.append(_phase_timing(name, STAGE_KINDS.get(name, "cpu"), row))
    gates = state.get("post_stages", {}).get("validate_all")
    if gates:
        phases.append(_phase_timing("gates", "cpu", gates))
    fin = state.get("post_stages", {}).get("finalize")
    if fin:
        phases.append(_phase_timing("finalizer", "cpu", fin))
    total = sum(phase["wall_seconds"] for phase in phases
                if isinstance(phase.get("wall_seconds"), (int, float)))
    gpu_wait_total = sum(phase["gpu_wait_seconds"] for phase in phases
                         if isinstance(phase.get("gpu_wait_seconds"), (int, float)))
    return {"schema": "fast-instrument-v2-timing/v1", "run_dir": state.get("run_dir"),
            "bundle": state.get("bundle"), "started_at": state.get("started_at"),
            "generated_at": _now(), "status": state.get("status"),
            "wall_seconds_total": total, "gpu_wait_seconds_total": gpu_wait_total,
            "phase_count": len(phases), "phases": phases}


def _write_timing(run_dir, state, stage_order):
    timing = _build_timing(state, stage_order)
    try:
        _write_json_atomic(Path(run_dir).resolve() / "TIMING.json", timing)
    except Exception:
        pass
    return timing


def _record_base(name, command, outputs):
    return {"name": name, "status": "PENDING", "argv": list(command["argv"]), "cwd": str(command["cwd"]),
            "env_overrides": dict(command.get("env_overrides", {})), "timeout_sec": command.get("timeout_sec"),
            "outputs": [str(path) for path in outputs], "output_sha256": {}, "input_sha256": {}, "reused": False}


def _set_blocked(state, stage_names, reason):
    targets = list(stage_names) if stage_names else list(state.get("stages", {}))
    for name in targets:
        row = state["stages"].get(name)
        if row and row.get("status") in ("PENDING", "BLOCKED"):
            row.update({"status": "BLOCKED", "reason": reason})
    for name in ("preflight", "validate_all", "finalize"):
        row = state.get("post_stages", {}).get(name)
        if row and row.get("status") in ("PENDING", "BLOCKED"):
            row.update({"status": "BLOCKED", "reason": reason})


def _force_flags(argv, values):
    result, skip_next = [], False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        matched = next((key for key in values if token == key or token.startswith(key + "=")), None)
        if matched:
            skip_next = token == matched
            continue
        result.append(token)
    if skip_next:
        raise PipelineError("forced flag has no value")
    for key, value in values.items():
        result.extend((key, str(value)))
    return result


def _flag_value(argv, flag):
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def _assert_contract_unchanged(contract_path, expected_hash, state, where):
    try:
        observed = _sha256(contract_path)
    except OSError as error:
        state["contract_changed_at"] = where
        _set_blocked(state, (), "contract became unreadable during pipeline")
        raise PipelineError("cannot hash contract at {}: {}".format(where, error))
    if observed != expected_hash:
        state["contract_changed_at"] = where
        _set_blocked(state, (), "contract changed during pipeline")
        raise PipelineError("contract SHA256 changed at {}".format(where))


def _check_preflight_payload(path, contract_hash):
    payload = _load_json(path)
    if payload.get("status") != "PASS" or payload.get("contract_sha256") != contract_hash:
        raise PipelineError("preflight status or contract hash is invalid")
    return payload


def _candidate_unchanged(candidate, expected_hash, where):
    try:
        observed = _sha256(candidate)
    except OSError as error:
        raise PipelineError("cannot hash candidate trace at {}: {}".format(where, error))
    if observed != expected_hash:
        raise PipelineError("candidate trace SHA256 changed at {}".format(where))


def _find_receipt(contract, execution, run_dir):
    configs = [execution.get("finalize"), execution.get("finalize_args"), contract.get("artifacts")]
    for config in configs:
        if isinstance(config, dict):
            for key in ("receipt", "delivery_receipt"):
                if isinstance(config.get(key), str):
                    return _absolute(config[key], run_dir)
            argv = config.get("argv", config.get("command", config.get("args", config.get("finalize_args"))))
        else:
            argv = config if isinstance(config, list) else None
        value = _flag_value(argv, "--receipt") if isinstance(argv, list) else None
        if value:
            return _absolute(value, run_dir)
    return Path(run_dir).resolve() / "delivery" / "delivery_receipt.json"


def _finalizer_args(execution):
    config = execution.get("finalize")
    if config is None:
        args = execution.get("finalize_args")
    elif isinstance(config, dict):
        if "argv" in config or "command" in config:
            raise PipelineError("execution.finalize cannot replace the fixed finalizer script")
        args = config.get("args", config.get("finalize_args"))
    else:
        args = config
    args = _check_argv(args, "execution.finalize.args")
    protected = ("--contract", "--all-gates", "--preflight", "--trace-processor",
                 "--timing-summary")
    if any(token == flag or token.startswith(flag + "=") for token in args for flag in protected):
        raise PipelineError("execution.finalize.args cannot provide protected enforcer flags")
    return args


def _build_finalizer(execution, candidate, contract_path, all_gates, receipt, bundle, trace_processor, run_dir, script_root, timing_path):
    args = _finalizer_args(execution)
    candidate_tokens = [token for token in args if not token.startswith("-")]
    if not any(token == str(candidate) or token == candidate.as_posix() or _absolute(token, run_dir) == candidate for token in candidate_tokens):
        raise PipelineError("finalizer args must provide the candidate trace")
    if _flag_value(args, "--receipt") is None or _flag_value(args, "--information-bundle") != bundle:
        raise PipelineError("finalizer args must provide receipt and information bundle")
    if len([token for token in args if not token.startswith("-")]) < 2:
        raise PipelineError("finalizer args must provide project and trace")
    argv = [sys.executable, str(Path(script_root) / "4_validate" / "finalize_delivery.py")] + args
    return _force_flags(argv, {"--contract": contract_path, "--all-gates": all_gates,
                               "--trace-processor": trace_processor, "--preflight": run_dir / "PREFLIGHT.json",
                               "--timing-summary": timing_path})


def run_pipeline(contract_path, run_dir, trace_processor, resume=False, script_root=None, inherit=None):
    contract_path, run_dir, trace_processor = Path(contract_path).expanduser().resolve(), Path(run_dir).expanduser().resolve(), Path(trace_processor).expanduser().resolve()
    if contract_path != (run_dir / "CONTRACT.lock.json").resolve():
        raise PipelineError("contract must be run_dir/CONTRACT.lock.json")
    if not contract_path.is_file() or not trace_processor.is_file():
        raise PipelineError("contract and trace processor must exist")
    if resume and inherit:
        raise PipelineError("--resume and --inherit are mutually exclusive")
    contract_hash = _sha256(contract_path)
    contract = _load_json(contract_path)
    execution, bundle, stage_specs, stage_order, parallel_groups = _validate_contract(contract, run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate = _candidate_trace(contract, execution, run_dir)
    perfetto_outputs = _output_paths(next(item["spec"] for item in stage_specs if item["name"] == "perfetto"),
                                     next(item["command"]["cwd"] for item in stage_specs if item["name"] == "perfetto"))
    if candidate not in perfetto_outputs:
        raise PipelineError("perfetto stage outputs must include candidate trace")
    state_path = run_dir / "PIPELINE_STATE.json"
    if not resume and state_path.exists():
        raise PipelineError("existing PIPELINE_STATE.json requires --resume")
    previous = None
    if resume:
        if not state_path.is_file():
            raise PipelineError("resume requires existing PIPELINE_STATE.json")
        previous = _load_json(state_path)
        if previous.get("contract_sha256") != contract_hash or previous.get("candidate_trace") != str(candidate):
            raise PipelineError("resume rejected: contract hash or candidate trace path mismatch")
    state = {"schema": "fast-instrument-v2-pipeline-state/v3", "status": "RUNNING", "contract": str(contract_path),
             "contract_sha256": contract_hash, "bundle": bundle, "run_dir": str(run_dir), "candidate_trace": str(candidate),
             "candidate_trace_sha256": None, "started_at": _now(), "stages": {}}
    for item in stage_specs:
        outputs = _output_paths(item["spec"], item["command"]["cwd"])
        if any(not _inside(path, run_dir) for path in outputs):
            raise PipelineError("all stage outputs must be inside run_dir")
        if not resume and any(path.exists() for path in outputs):
            raise PipelineError("existing stage output requires --resume")
        state["stages"][item["name"]] = _record_base(item["name"], item["command"], outputs)
    if not resume and candidate.exists():
        raise PipelineError("existing candidate trace requires --resume")
    state["post_stages"] = {"preflight": {"name": "preflight", "status": "PENDING", "output_sha256": {}},
                             "validate_all": {"name": "validate_all", "status": "PENDING", "output_sha256": {}},
                             "finalize": {"name": "finalize", "status": "PENDING", "output_sha256": {}}}
    _write_json_atomic(state_path, state)
    inherit_ctx = None
    if inherit:
        old_run_dir = Path(inherit).expanduser().resolve()
        if old_run_dir == run_dir:
            raise PipelineError("--inherit needs a different run directory; identical run dir should use --resume")
        old_state_path = old_run_dir / "PIPELINE_STATE.json"
        old_contract_path = old_run_dir / "CONTRACT.lock.json"
        if not old_state_path.is_file() or not old_contract_path.is_file():
            raise PipelineError("--inherit requires an old run dir holding PIPELINE_STATE.json and CONTRACT.lock.json")
        old_state = _load_json(old_state_path)
        old_contract_hash = _sha256(old_contract_path)
        if old_state.get("contract_sha256") != old_contract_hash:
            raise PipelineError("inherit rejected: old run state does not match old contract hash")
        if old_contract_hash == contract_hash:
            raise PipelineError("inherit rejected: contract hash unchanged; use --resume")
        old_contract = _load_json(old_contract_path)
        try:
            _old_execution, _old_bundle, old_specs, _old_order, _old_groups = _validate_contract(old_contract, old_run_dir)
        except PipelineError as error:
            raise PipelineError("inherit rejected: old contract is invalid: {}".format(error))
        previous = old_state
        inherit_ctx = _build_inherit_ctx(old_state, old_contract, old_specs, old_run_dir, old_contract_path)
        state["inherit_from"] = str(old_run_dir)
        _write_json_atomic(state_path, state)
    previous_rows, previous_post = (previous or {}).get("stages", {}), (previous or {}).get("post_stages", {})
    invalidated = False
    script_root = Path(script_root or SCRIPT_DIR).resolve()
    try:
        preflight_path = run_dir / "PREFLIGHT.json"
        pre = _command_config([sys.executable, str(script_root / "1_scan" / "validate_preflight.py")], "fixed preflight runner", run_dir, os.environ)
        pre["argv"] = _force_flags(pre["argv"], {"--contract": contract_path, "--run-dir": run_dir, "--out": preflight_path})
        old_pre = previous_post.get("preflight") if isinstance(previous_post, dict) else None
        pre_row = state["post_stages"]["preflight"]
        pre_row.update(_run_command(pre, _log_path(run_dir, "preflight"))); pre_row["argv"] = pre["argv"]; pre_row["output_sha256"] = _hash_outputs([preflight_path])
        payload = _load_json(preflight_path) if preflight_path.is_file() else {}
        if pre_row.get("exit_code") != 0 or pre_row.get("timed_out") or payload.get("status") != "PASS" or payload.get("contract_sha256") != contract_hash:
            pre_row["status"] = "FAIL"; _set_blocked(state, stage_order, "preflight failed"); _write_json_atomic(state_path, state)
            raise PipelineError("preflight failed: {}".format(payload.get("input_errors") or payload.get("summary") or pre_row))
        pre_row["status"] = "PASS"
        _assert_contract_unchanged(contract_path, contract_hash, state, "after preflight")
        pre_hash = _sha256(preflight_path)
        state["preflight"] = str(preflight_path); state["preflight_sha256"] = pre_hash; _write_json_atomic(state_path, state)

        allowed_gpu_ids, gpu_idle_policy = _allowed_gpu_config(execution)
        reuse_gpu_lease, gpu_lease_reason = _gpu_lease_reuse_config(execution)
        idle = None
        if allowed_gpu_ids is None:
            idle = _command_config(execution["gpu_idle_command"], "execution.gpu_idle_command", run_dir, os.environ)
        perfetto_reused = False
        spec_by_name = {item["name"]: item for item in stage_specs}
        active_groups, parallel_reason = _active_parallel_groups(parallel_groups, spec_by_name, allowed_gpu_ids)
        gpu_lease = _new_sequential_gpu_lease(
            run_dir, allowed_gpu_ids, gpu_idle_policy,
            reuse_gpu_lease, gpu_lease_reason, active_groups)
        state["gpu_lease"] = _sequential_gpu_lease_state(gpu_lease)
        _write_json_atomic(state_path, state)
        if parallel_groups:
            state["parallel_groups"] = {"declared": [list(group) for group in parallel_groups],
                                        "active": [list(group) for group in active_groups],
                                        "reason": parallel_reason}
            _write_json_atomic(state_path, state)
        if not active_groups:
            for index, item in enumerate(stage_specs):
                name, row, prior = item["name"], state["stages"][item["name"]], previous_rows.get(item["name"])
                _assert_contract_unchanged(contract_path, contract_hash, state, "before stage {}".format(name))
                if resume and not invalidated and isinstance(prior, dict) and prior.get("status") in ("PASS", "REUSED") and _outputs_match(prior) and _stage_inputs_reusable(item, prior):
                    row.update(prior); row["status"] = "REUSED"; row["reused"] = True
                    if name == "perfetto":
                        perfetto_reused = True
                    _note_stage_for_sequential_gpu_lease(gpu_lease, index, item["kind"], name)
                    state["gpu_lease"] = _sequential_gpu_lease_state(gpu_lease)
                    _write_json_atomic(state_path, state); continue
                if resume and not invalidated and prior:
                    invalidated = True; row["resume_invalidation"] = _resume_invalidation_reason(item, prior)
                if inherit and not invalidated:
                    if _try_inherit(item, row, inherit_ctx, contract, run_dir, contract_path):
                        if name == "perfetto":
                            perfetto_reused = True
                        _note_stage_for_sequential_gpu_lease(gpu_lease, index, item["kind"], name)
                        state["gpu_lease"] = _sequential_gpu_lease_state(gpu_lease)
                        _write_json_atomic(state_path, state); continue
                    invalidated = True; row["inherit_invalidation"] = "inherit evidence incomplete"
                command = item["command"]
                if not command["cwd"].is_dir():
                    raise PipelineError("stage {} cwd does not exist".format(name))
                if item["kind"] == "gpu":
                    idle_record = _prepare_serial_stage_gpus(
                        item, index, row, run_dir, allowed_gpu_ids,
                        gpu_idle_policy, gpu_lease, idle)
                    state["gpu_lease"] = _sequential_gpu_lease_state(gpu_lease)
                    if idle_record.get("exit_code") != 0 or idle_record.get("timed_out"):
                        row["status"] = "FAIL"; row["reason"] = "GPU idle check failed"; _set_blocked(state, stage_order[index + 1:], "previous stage failed: {}".format(name)); _write_json_atomic(state_path, state); raise PipelineError("stage {} GPU idle check failed".format(name))
                row["input_sha256"] = _stage_record_inputs(item, contract)
                result = _run_command(command, _log_path(run_dir, "{:02d}_{}".format(index, name))); row.update(result); row["output_sha256"] = _hash_outputs(_output_paths(item["spec"], command["cwd"]))
                missing = [path for path, value in row["output_sha256"].items() if value is None]
                if result.get("exit_code") != 0 or result.get("timed_out") or missing:
                    row["status"] = "FAIL"; row["reason"] = "timeout" if result.get("timed_out") else ("missing outputs" if missing else "non-zero exit"); _set_blocked(state, stage_order[index + 1:], "previous stage failed: {}".format(name)); _write_json_atomic(state_path, state); raise PipelineError("stage {} failed".format(name))
                row["status"] = "PASS"
                _note_stage_for_sequential_gpu_lease(gpu_lease, index, item["kind"], name)
                state["gpu_lease"] = _sequential_gpu_lease_state(gpu_lease)
                _write_json_atomic(state_path, state)
        else:
            executed_groups = set()
            group_of = {}
            for group in active_groups:
                for member in group:
                    group_of[member] = group
            for index, item in enumerate(stage_specs):
                name, row, prior = item["name"], state["stages"][item["name"]], previous_rows.get(item["name"])
                if name in group_of and group_of[name] in executed_groups:
                    continue
                _assert_contract_unchanged(contract_path, contract_hash, state, "before stage {}".format(name))
                if resume and not invalidated and isinstance(prior, dict) and prior.get("status") in ("PASS", "REUSED") and _outputs_match(prior) and _stage_inputs_reusable(item, prior):
                    row.update(prior); row["status"] = "REUSED"; row["reused"] = True
                    if name == "perfetto":
                        perfetto_reused = True
                    _note_stage_for_sequential_gpu_lease(gpu_lease, index, item["kind"], name)
                    state["gpu_lease"] = _sequential_gpu_lease_state(gpu_lease)
                    _write_json_atomic(state_path, state); continue
                if resume and not invalidated and prior:
                    invalidated = True; row["resume_invalidation"] = _resume_invalidation_reason(item, prior)
                if inherit and not invalidated:
                    if _try_inherit(item, row, inherit_ctx, contract, run_dir, contract_path):
                        if name == "perfetto":
                            perfetto_reused = True
                        _note_stage_for_sequential_gpu_lease(gpu_lease, index, item["kind"], name)
                        state["gpu_lease"] = _sequential_gpu_lease_state(gpu_lease)
                        _write_json_atomic(state_path, state); continue
                    invalidated = True; row["inherit_invalidation"] = "inherit evidence incomplete"
                if name in group_of:
                    group = group_of[name]
                    executed_groups.add(group)
                    members = [entry for entry in stage_specs if entry["name"] in group]
                    _run_parallel_group(members, stage_specs, state, state_path, stage_order, run_dir,
                                        allowed_gpu_ids, gpu_idle_policy, contract)
                    continue
                command = item["command"]
                if not command["cwd"].is_dir():
                    raise PipelineError("stage {} cwd does not exist".format(name))
                if item["kind"] == "gpu":
                    idle_record = _prepare_serial_stage_gpus(
                        item, index, row, run_dir, allowed_gpu_ids,
                        gpu_idle_policy, gpu_lease, idle)
                    state["gpu_lease"] = _sequential_gpu_lease_state(gpu_lease)
                    if idle_record.get("exit_code") != 0 or idle_record.get("timed_out"):
                        row["status"] = "FAIL"; row["reason"] = "GPU idle check failed"; _set_blocked(state, stage_order[index + 1:], "previous stage failed: {}".format(name)); _write_json_atomic(state_path, state); raise PipelineError("stage {} GPU idle check failed".format(name))
                row["input_sha256"] = _stage_record_inputs(item, contract)
                result = _run_command(command, _log_path(run_dir, "{:02d}_{}".format(index, name))); row.update(result); row["output_sha256"] = _hash_outputs(_output_paths(item["spec"], command["cwd"]))
                missing = [path for path, value in row["output_sha256"].items() if value is None]
                if result.get("exit_code") != 0 or result.get("timed_out") or missing:
                    row["status"] = "FAIL"; row["reason"] = "timeout" if result.get("timed_out") else ("missing outputs" if missing else "non-zero exit"); _set_blocked(state, stage_order[index + 1:], "previous stage failed: {}".format(name)); _write_json_atomic(state_path, state); raise PipelineError("stage {} failed".format(name))
                row["status"] = "PASS"
                _note_stage_for_sequential_gpu_lease(gpu_lease, index, item["kind"], name)
                state["gpu_lease"] = _sequential_gpu_lease_state(gpu_lease)
                _write_json_atomic(state_path, state)

        _assert_contract_unchanged(contract_path, contract_hash, state, "before validate_all")
        if not candidate.is_file():
            raise PipelineError("perfetto stage did not produce candidate trace")
        candidate_hash = _sha256(candidate)
        previous_candidate_hash = previous.get("candidate_trace_sha256") if isinstance(previous, dict) else None
        if (resume or inherit) and perfetto_reused and (not previous_candidate_hash or candidate_hash != previous_candidate_hash):
            raise PipelineError("candidate trace hash changed after measurement stages")
        state["candidate_trace_sha256"] = candidate_hash
        _write_json_atomic(state_path, state)
        _candidate_unchanged(candidate, candidate_hash, "before validate_all")
        all_gates = run_dir / "ALL_GATES.json"
        val_row = state["post_stages"]["validate_all"]
        val_cfg = _command_config([sys.executable, str(script_root / "4_validate" / "validate_all.py")], "fixed validate_all runner", run_dir, os.environ)
        val_cfg["argv"] = _force_flags(val_cfg["argv"], {"--contract": contract_path, "--run-dir": run_dir, "--trace": candidate, "--out": all_gates})
        val_row.update(_run_command(val_cfg, _log_path(run_dir, "validate_all"))); val_row["argv"] = val_cfg["argv"]; val_row["output_sha256"] = _hash_outputs([all_gates])
        if val_row.get("exit_code") != 0 or val_row.get("timed_out") or not all_gates.is_file():
            val_row["status"] = "FAIL"; _set_blocked(state, (), "validate_all failed"); _write_json_atomic(state_path, state); raise PipelineError("validate_all failed")
        val_row["status"] = "PASS"
        _assert_contract_unchanged(contract_path, contract_hash, state, "after validate_all")
        _candidate_unchanged(candidate, candidate_hash, "after validate_all")
        all_gates_hash = _sha256(all_gates); state["all_gates"] = str(all_gates); state["all_gates_sha256"] = all_gates_hash; _write_json_atomic(state_path, state)

        _assert_contract_unchanged(contract_path, contract_hash, state, "before finalize")
        _candidate_unchanged(candidate, candidate_hash, "before finalize")
        receipt = _find_receipt(contract, execution, run_dir)
        if not _inside(receipt, run_dir):
            raise PipelineError("receipt must be inside run_dir")
        _write_timing(run_dir, state, stage_order)
        final_argv = _build_finalizer(execution, candidate, contract_path, all_gates, receipt, bundle, trace_processor, run_dir, script_root, run_dir / "TIMING.json")
        final_cfg = _command_config(final_argv, "fixed finalizer runner", run_dir, os.environ)
        final_cfg["argv"] = final_argv
        fin_row = state["post_stages"]["finalize"]
        fin_row.update(_run_command(final_cfg, _log_path(run_dir, "finalize_delivery"))); fin_row["argv"] = final_argv; fin_row["receipt"] = str(receipt); fin_row["output_sha256"] = _hash_outputs([receipt])
        payload = _load_json(receipt) if receipt.is_file() else {}
        required_hashes = {"contract_sha256": contract_hash, "all_gates_sha256": all_gates_hash, "trace_sha256": candidate_hash}
        receipt_preflight = payload.get("preflight_path")
        preflight_path_matches = isinstance(receipt_preflight, str) and _absolute(receipt_preflight, run_dir) == preflight_path
        post_pre_hash = _sha256(preflight_path) if preflight_path.is_file() else None
        post_gates_hash = _sha256(all_gates) if all_gates.is_file() else None
        receipt_files_match = post_pre_hash == pre_hash and post_gates_hash == all_gates_hash
        if fin_row.get("exit_code") != 0 or fin_row.get("timed_out") or payload.get("status") != "PASS" or any(payload.get(key) != value for key, value in required_hashes.items()) or not preflight_path_matches or payload.get("preflight_sha256") != pre_hash or not receipt_files_match:
            fin_row["status"] = "FAIL"; _write_json_atomic(state_path, state); raise PipelineError("finalizer did not produce a matching PASS receipt")
        fin_row["status"] = "PASS"; state["receipt"] = str(receipt); state["receipt_sha256"] = _sha256(receipt); _write_json_atomic(state_path, state)
        _write_timing(run_dir, state, stage_order)
        _assert_contract_unchanged(contract_path, contract_hash, state, "after finalize")
        _candidate_unchanged(candidate, candidate_hash, "after finalize")
        _assert_contract_unchanged(contract_path, contract_hash, state, "before completion")
        _candidate_unchanged(candidate, candidate_hash, "before completion")
        state.update({"receipt": str(receipt), "receipt_sha256": _sha256(receipt), "status": "PASS", "ended_at": _now()}); _write_json_atomic(state_path, state); _write_timing(run_dir, state, stage_order); return state
    except PipelineError:
        _set_blocked(state, (), "pipeline aborted"); state["status"] = "FAIL"; state["ended_at"] = _now(); _write_json_atomic(state_path, state); _write_timing(run_dir, state, stage_order); raise


def _gpu_probe(allowed_ids=None, util_max=GPU_IDLE_DEFAULT_UTIL_MAX, mem_max=None):
    """Read-only classification of every visible GPU (never kills a process).

    Exposes the exact free/busy predicate the whitelist idle wait uses so an
    operator can confirm which cards the pipeline would bind to right now."""
    gpus = _query_gpus_nvidia_smi()
    report = [{"index": gpu.get("index"), "memory_used_mib": gpu.get("memory_used_mib"),
               "utilization_percent": gpu.get("utilization_percent"),
               "free": _gpu_is_free(gpu, util_max, mem_max)} for gpu in gpus]
    chosen = _pick_free_gpu(gpus, allowed_ids, util_max, mem_max) if allowed_ids else None
    return {"utilization_max_percent": util_max, "memory_max_mib": mem_max,
            "allowed_gpu_ids": allowed_ids, "chosen_gpu": chosen, "gpus": report}


def _selftest():
    """Test bundle order, preflight, rejection, fail-closed delivery, and resume."""
    with tempfile.TemporaryDirectory(prefix="runner_selftest_", dir=str(SCRIPT_DIR)) as temp:
        root = Path(temp).resolve(); (root / "1_scan").mkdir(); (root / "4_validate").mkdir()
        helper = root / "helper.py"
        helper.write_text('import gzip, json, pathlib, sys, time\nmode = sys.argv[1]\nTILES = 20\nBASE = 1000000000\nSPAN = 100000000\nSTEP = SPAN // TILES\n\n\ndef uid(i, a, b):\n    return {"rank": 0, "lane": 0, "start_offset": a, "end_offset": b, "event_id": 1, "work_id": i}\n\n\ndef instance(i):\n    a = i * 10\n    b = a + 5\n    return {"event_id": 1, "instance_key": {"tile": i}, "is_derived": False,\n            "canonical_recorder": {"warp": 0, "thread": 0},\n            "start_offset": a, "end_offset": b, "source_event_uid": uid(i, a, b)}\n\n\ndef trace_events():\n    events = []\n    for i in range(TILES):\n        args = {"layer": "task", "start_ns_raw": BASE + i * STEP, "end_ns_raw": BASE + (i + 1) * STEP,\n                "event_id": 1, "instance_key": {"tile": i}, "is_derived": False,\n                "canonical_recorder": {"warp": 0, "thread": 0},\n                "source_event_uid": instance(i)["source_event_uid"]}\n        events.append({"name": "Operator {} Proj".format(i), "ph": "X", "pid": 1000, "tid": 1,\n                       "ts": (BASE + i * STEP) / 1000.0, "dur": STEP / 1000.0, "args": args})\n    return {"traceEvents": events, "otherData": {"hw_merge": {"sw_window_ns": [BASE, BASE + SPAN]}}}\n\n\ndef write_side_artifacts(p):\n    run = p.parent\n    if p.name == "correctness.out":\n        val = run / "validation"\n        val.mkdir(parents=True, exist_ok=True)\n        (val / "correctness.json").write_text(json.dumps({"all_passed": True, "sample_only": False,\n                                                          "per_tensor": [{"max_abs_err": 0.0}]}), encoding="utf-8")\n    if p.name == "sw_capture.out":\n        sw = run / "sw"\n        sw.mkdir(parents=True, exist_ok=True)\n        (sw / "raw_pairs_rank0.json").write_text(json.dumps({"raw_pairs": [instance(i) for i in range(TILES)]}), encoding="utf-8")\n        (sw / "events_rank0.json").write_text(json.dumps({"events": [instance(i) for i in range(TILES)]}), encoding="utf-8")\n    if p.name == "hw_capture.out":\n        hw = run / "hw"\n        hw.mkdir(parents=True, exist_ok=True)\n        counters = [\n            {"name": "DRAM Read Bandwidth", "unit": "GB/s", "sample_count": 2, "samples": [[0, 900.0], [5000, 910.0]]},\n            {"name": "DRAM Write Bandwidth", "unit": "GB/s", "sample_count": 2, "samples": [[0, 410.0], [5000, 420.0]]},\n            {"name": "DRAM Bandwidth Total", "unit": "GB/s", "sample_count": 2, "samples": [[0, 1310.0], [5000, 1330.0]]},\n        ]\n        (hw / "hw_metrics.json").write_text(json.dumps({"counters": counters, "scale_rules": []}), encoding="utf-8")\n\n\ndef write_output(path):\n    p = pathlib.Path(path)\n    p.parent.mkdir(parents=True, exist_ok=True)\n    if p.name == "candidate.perfetto.json.gz":\n        p.write_bytes(gzip.compress(json.dumps(trace_events()).encode("utf-8")))\n    else:\n        p.write_text("ok-" + p.name)\n    write_side_artifacts(p)\n\n\nif mode == "fail":\n    sys.exit(7)\nif mode == "write":\n    write_output(sys.argv[2])\n    pathlib.Path(sys.argv[3]).open("a").write(pathlib.Path(sys.argv[2]).name + "\\n")\nif mode == "evidence":\n    p = pathlib.Path(sys.argv[3])\n    p.parent.mkdir(parents=True, exist_ok=True)\n    p.write_text(sys.argv[2])\nif mode == "idle":\n    pathlib.Path(sys.argv[2]).write_text("idle")\nif mode == "psleep":\n    write_output(sys.argv[2])\n    pathlib.Path(sys.argv[3]).open("a").write(pathlib.Path(sys.argv[2]).name + "\\n")\n    t0 = time.time()\n    time.sleep(1.0)\n    t1 = time.time()\n    rec = pathlib.Path(sys.argv[4])\n    rec.parent.mkdir(parents=True, exist_ok=True)\n    rec.open("a").write(json.dumps({"name": sys.argv[5], "start": t0, "end": t1}) + "\\n")\n', encoding="utf-8")
        preflight = root / "1_scan" / "validate_preflight.py"
        preflight.write_bytes((SCRIPT_DIR / "1_scan" / "validate_preflight.py").read_bytes())
        # validate_preflight imports semantic_contract from its own directory.
        # Keep the selftest hermetic when it runs from a temporary script root.
        semantic = root / "1_scan" / "semantic_contract.py"
        semantic.write_bytes((SCRIPT_DIR / "1_scan" / "semantic_contract.py").read_bytes())
        validate = root / "4_validate" / "validate_all.py"
        validate.write_bytes((SCRIPT_DIR / "4_validate" / "validate_all.py").read_bytes())
        (root / "4_validate" / "semantic_contract.py").write_bytes(
            (SCRIPT_DIR / "1_scan" / "semantic_contract.py").read_bytes())
        # validate_all imports deliver_gate (timeline audit), which imports
        # validate_capability and semantic_contract; keep the temporary script
        # root complete so the selftest exercises the real validators.
        for extra_validator in ("deliver_gate.py", "validate_capability.py", "validate_run.py"):
            (root / "4_validate" / extra_validator).write_bytes(
                (SCRIPT_DIR / "4_validate" / extra_validator).read_bytes())
        (root / "1_scan" / "layout_contract.py").write_bytes(
            (SCRIPT_DIR / "1_scan" / "layout_contract.py").read_bytes())
        finalize = root / "4_validate" / "finalize_delivery.py"
        finalize.write_text("import argparse,hashlib,json,pathlib\np=argparse.ArgumentParser();p.add_argument('--contract');p.add_argument('--all-gates');p.add_argument('--trace-processor');p.add_argument('--preflight');p.add_argument('--receipt');p.add_argument('--information-bundle');p.add_argument('--timing-summary');p.add_argument('pos',nargs='*');a=p.parse_args();h=lambda x:hashlib.sha256(pathlib.Path(x).read_bytes()).hexdigest();trace=a.pos[1];tj=json.loads(pathlib.Path(a.timing_summary).read_text()) if (a.timing_summary and pathlib.Path(a.timing_summary).is_file()) else None;ts={'available':True,'schema':tj.get('schema'),'phase_count':len(tj.get('phases',[]))} if tj else {'available':False};pathlib.Path(a.receipt).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(a.receipt).write_text(json.dumps({'status':'PASS','contract_sha256':h(a.contract),'all_gates_sha256':h(a.all_gates),'trace_sha256':h(trace),'preflight_path':str(pathlib.Path(a.preflight).resolve()),'preflight_sha256':h(a.preflight),'timing_summary':ts}))\n", encoding="utf-8")
        processor = root / "trace_processor_shell"; processor.write_text("selftest", encoding="utf-8")

        # The current preflight validator requires a source-locked semantic
        # contract even for runner selftests.  Build a one-instance fixture
        # here so the selftest exercises the real validator instead of
        # weakening production validation.
        semantic_version_root = root / "version_root"
        semantic_version_root.mkdir(parents=True, exist_ok=True)
        semantic_source = semantic_version_root / "semantic_source.py"
        semantic_source.write_text(
            "semantic_task_begin\nsemantic_task_sync\nsemantic_task_end\n",
            encoding="utf-8")
        task_contract = {
            "classification": "task",
            "semantic_unit": "selftest_tile",
            "instance_key": ["tile"],
            "instance_domain": {"axes": {"tile": list(range(20))}},
            "expected_cardinality": 20,
            "collaborating_warps": {"values": [0]},
            "recording_representative": {"warp": 0, "thread": 0},
            "tile_shape": {"rows": 1, "columns": 1},
            "source_boundary": {
                "begin": {"path": "semantic_source.py", "line": 1, "anchor": "semantic_task_begin"},
                "completion_sync": {"path": "semantic_source.py", "line": 2, "anchor": "semantic_task_sync"},
                "end": {"path": "semantic_source.py", "line": 3, "anchor": "semantic_task_end"},
            },
        }
        task_descriptor = {"id": 1, "name": "Selftest Tile", "display_name": "Selftest Tile",
                           "layer": "task", "task_contract": task_contract}
        semantic = {
            "schema": "fast-instrument-semantic-analysis/v1",
            "model_reviewed": True,
            "source_closure": [{"path": "semantic_source.py",
                                "sha256": _sha256(semantic_source)}],
        }

        def semantic_hash(document):
            analysis = document["semantic_analysis"]
            locked = {key: value for key, value in analysis.items()
                      if key not in {"conclusion_sha256", "semantic_conclusion_sha256",
                                     "abi", "abi_path", "abi_sha256", "bijection_evidence", "evidence"}}
            tasks = [{"id": task_descriptor["id"], "name": task_descriptor["name"],
                      "display_name": task_descriptor["display_name"], "layer": "task",
                      "task_contract": task_contract}]
            payload = {"semantic_analysis": locked, "tasks": tasks}
            return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                             separators=(",", ":")).encode("utf-8")).hexdigest()

        abi = {"semantic_analysis": semantic, "tasks": [task_descriptor]}
        semantic["conclusion_sha256"] = semantic_hash(abi)
        abi_path = semantic_version_root / "semantic_abi.json"
        abi_path.write_text(json.dumps(abi, sort_keys=True), encoding="utf-8")
        abi_sha256 = _sha256(abi_path)

        whitelist_run = root / "whitelist_evidence"
        evidence_path = _write_whitelist_gpu_idle_evidence(whitelist_run, {
            "exit_code": 0,
            "chosen_gpu": 2,
            "allowed_gpu_ids": [2, 3],
            "policy": {"utilization_max_percent": 10.0},
            "polls": [{"at": _now(), "whitelisted": [
                {"index": 2, "memory_used_mib": 34.0, "utilization_percent": 0.0}
            ]}],
            "ended_at": _now(),
        })
        evidence = _load_json(evidence_path)
        if (not evidence_path.is_file() or evidence.get("idle") is not True
                or evidence.get("chosen_gpu") != 2
                or evidence.get("allowed_gpu_ids") != [2, 3]):
            raise PipelineError("selftest whitelist scheduler did not emit canonical G00 evidence")

        required_checks = ["E{:02d}".format(index) for index in range(1, 28)]
        required_gates = ["G{:02d}".format(index) for index in range(25)]
        # Defect 3: mirror the skill's hardcoded per-bundle N/A partitions so the
        # sw contract satisfies the (now bundle-aware) preflight and validate_all
        # validators; sw+hw stays fully required (byte-identical to before).
        bundle_na_checks = {"sw": ["E17", "E18", "E19", "E20"],
                            "hw": ["E02", "E03", "E05", "E06", "E07", "E08",
                                   "E09", "E10", "E12", "E13", "E15", "E16", "E20"],
                            "sw+hw": []}
        bundle_na_gates = {"sw": ["G11", "G19", "G20", "G21"], "hw": [], "sw+hw": []}

        def make(bundle, name, fail=False, extra=None, stage_inputs=None, components=None,
                 allowed=None, parallel=None, gpus=None, stage_modes=None, stage_scripts=None):
            version_root = semantic_version_root
            run = version_root / "results" / "profile_runs" / name
            run.mkdir(parents=True, exist_ok=True)
            candidate = run / "candidate.perfetto.json.gz"; order = run / "order.log"; names = list(BUNDLE_STAGES[bundle]);
            if extra: names.insert(-1, extra)
            stages = []
            for stage in names:
                output = candidate if stage == "perfetto" else run / (stage + ".out"); mode = "fail" if fail and stage == "correctness" else (stage_modes or {}).get(stage, "write"); script = (stage_scripts or {}).get(stage, helper)
                if mode == "psleep":
                    argv = [sys.executable, str(script), mode, str(output), str(order), str(run / "parallel_timing.jsonl"), stage]
                elif mode == "fail":
                    argv = [sys.executable, str(script), mode]
                else:
                    argv = [sys.executable, str(script), mode, str(output), str(order)]
                side = []
                if stage == "correctness":
                    side = [str(run / "validation" / "correctness.json")]
                elif stage == "sw_capture":
                    side = [str(run / "sw" / "raw_pairs_rank0.json"), str(run / "sw" / "events_rank0.json")]
                elif stage == "hw_capture":
                    side = [str(run / "hw" / "hw_metrics.json")]
                entry = {"name": stage, "argv": argv, "cwd": str(run), "env": {}, "outputs": [str(output)] + side, "kind": "gpu" if stage in ("sw_capture", "hw_capture", "correctness", "overhead") else "cpu"}
                if stage_inputs and stage in stage_inputs:
                    entry["inputs"] = list(stage_inputs[stage])
                if gpus and stage in gpus:
                    entry["gpus_required"] = gpus[stage]
                stages.append(entry)
            receipt = run / "delivery" / "delivery_receipt.json"
            preflight_template = {"argv": [sys.executable, str(helper), "evidence", "{check_id}", "preflight_evidence/{check_id}.json"], "evidence": ["preflight_evidence/{check_id}.json"]}
            gate_template = {"argv": [sys.executable, str(helper), "evidence", "{gate_id}", "gate_evidence/{gate_id}.json"], "evidence": ["gate_evidence/{gate_id}.json"]}
            na_checks = list(bundle_na_checks.get(bundle, [])); req_checks = [c for c in required_checks if c not in na_checks]
            na_gates = list(bundle_na_gates.get(bundle, [])); req_gates = [g for g in required_gates if g not in na_gates]
            contract = {"lock": {"state": "frozen", "allow_in_place_edit": False}, "project_layout": {"version_root": str(version_root), "run_root": str(run)}, "run": {"target_kernel": "selftest_kernel"}, "capture": {"information_bundle": bundle, "display_rank": 0, "granularity": "task", "software_sm_scope": {"mode": "all", "selected_physical_smids": []}}, "input": {"generator": "randn"}, "semantic_analysis": {**json.loads(json.dumps(semantic)), "abi": {"path": str(abi_path), "sha256": abi_sha256}}, "events": {"event_definitions": [json.loads(json.dumps(task_descriptor))]}, "execution": {"candidate_trace": str(candidate), "gpu_idle_command": {"argv": [sys.executable, str(helper), "idle", str(run / "idle")], "cwd": str(run), "timeout_sec": 10}, "stages": stages, "preflight": {"required_check_ids": req_checks, "not_applicable_check_ids": na_checks, "check_command_template": preflight_template}, "required_gate_ids": req_gates, "not_applicable_gate_ids": na_gates, "gate_command_template": gate_template, "finalize": {"args": [str(root / "project"), str(candidate), "--receipt", str(receipt), "--information-bundle", bundle]}}, "artifacts": {"candidate_perfetto": str(candidate)}}
            if components is not None:
                contract["components"] = components
            if allowed is not None:
                contract["execution"]["allowed_gpu_ids"] = list(allowed)
                contract["execution"]["gpu_idle_policy"] = {"poll_interval_sec": 0.2, "timeout_sec": 10}
            if parallel is not None:
                contract["execution"]["parallel_groups"] = [list(group) for group in parallel]
            path = run / "CONTRACT.lock.json"; path.write_text(json.dumps(contract), encoding="utf-8"); return path

        good_path = make("sw+hw", "good")
        good_contract = _load_json(good_path)
        if good_contract["execution"]["preflight"]["required_check_ids"] != required_checks or good_contract["execution"]["preflight"]["not_applicable_check_ids"] != [] or good_contract["execution"]["required_gate_ids"] != required_gates or good_contract["execution"]["not_applicable_gate_ids"] != []:
            raise PipelineError("selftest gate contract is incomplete")
        run_pipeline(good_path, good_path.parent, processor, script_root=root)
        expected_order = [name + ".out" if name != "perfetto" else "candidate.perfetto.json.gz" for name in BUNDLE_STAGES["sw+hw"]]
        if (good_path.parent / "order.log").read_text().splitlines() != expected_order: raise PipelineError("selftest stage order failed")
        timing = _load_json(good_path.parent / "TIMING.json")
        timing_names = [phase.get("name") for phase in timing.get("phases", [])]
        if timing.get("schema") != "fast-instrument-v2-timing/v1" or "preflight" not in timing_names or "finalizer" not in timing_names or "sw_capture" not in timing_names:
            raise PipelineError("selftest TIMING.json missing expected phases")
        gpu_phases = [phase for phase in timing["phases"] if phase.get("kind") == "gpu"]
        if not gpu_phases or any(not isinstance(phase.get("gpu_wait_seconds"), (int, float)) for phase in gpu_phases):
            raise PipelineError("selftest TIMING.json missing gpu wait accounting")
        if not isinstance(timing.get("wall_seconds_total"), (int, float)) or timing["wall_seconds_total"] < 0:
            raise PipelineError("selftest TIMING.json total wall is invalid")
        receipt_payload = _load_json(good_path.parent / "delivery" / "delivery_receipt.json")
        receipt_timing = receipt_payload.get("timing_summary")
        if not isinstance(receipt_timing, dict) or receipt_timing.get("available") is not True or receipt_timing.get("schema") != "fast-instrument-v2-timing/v1" or not isinstance(receipt_timing.get("phase_count"), int) or receipt_timing["phase_count"] <= 0:
            raise PipelineError("selftest receipt missing timing_summary projection")
        sw_contract = make("sw", "sw_ok")
        run_pipeline(sw_contract, sw_contract.parent, processor, script_root=root)
        if "hw_capture.out" in (sw_contract.parent / "order.log").read_text().splitlines(): raise PipelineError("selftest sw bundle ran hw_capture")
        bad_bundle_path = make("sw", "bad_bundle", extra="hw_capture")
        for path, bad_name in ((bad_bundle_path, "bad_bundle"),):
            try:
                run_pipeline(path, path.parent, processor, script_root=root)
            except PipelineError:
                pass
            else:
                raise PipelineError("selftest accepted wrong bundle stages")
        example = json.loads(good_path.read_text()); example["example_only"] = True; example_run = root / "example"; example_run.mkdir(); example_path = example_run / "CONTRACT.lock.json"; example_path.write_text(json.dumps(example), encoding="utf-8")
        try:
            run_pipeline(example_path, example_path.parent, processor, script_root=root)
        except PipelineError:
            pass
        else:
            raise PipelineError("selftest accepted example contract")
        bad_path = make("sw", "bad", fail=True)
        try:
            run_pipeline(bad_path, bad_path.parent, processor, script_root=root)
        except PipelineError:
            pass
        else:
            raise PipelineError("selftest accepted failed stage")
        if (bad_path.parent / "delivery" / "delivery_receipt.json").exists(): raise PipelineError("failed run produced receipt")
        tampered = good_path.parent / "correctness.out"; tampered.write_text("tampered", encoding="utf-8"); before = len((good_path.parent / "order.log").read_text().splitlines()); run_pipeline(good_path, good_path.parent, processor, resume=True, script_root=root); after = (good_path.parent / "order.log").read_text().splitlines()
        if len(after) <= before or after[before] != "correctness.out": raise PipelineError("resume did not rerun from tampered output")
        if GPU_IDLE_DEFAULT_TIMEOUT != 120.0:
            raise PipelineError("selftest gpu idle default timeout must be 120s")

        # O1: resume with declared stage inputs reuses unchanged stages; a
        # changed input reruns exactly that stage and its successors.
        extra_input = semantic_version_root / "sw_capture_extra_input.txt"
        extra_input.write_text("extra-v1", encoding="utf-8")
        declared_inputs = {stage: [str(helper)] for stage in BUNDLE_STAGES["sw"]}
        declared_inputs["sw_capture"] = [str(helper), str(extra_input)]
        inputs_path = make("sw", "inputs_ok", stage_inputs=declared_inputs)
        run_pipeline(inputs_path, inputs_path.parent, processor, script_root=root)
        inputs_state = _load_json(inputs_path.parent / "PIPELINE_STATE.json")
        helper_key = str(Path(helper).resolve())
        if any(helper_key not in (row.get("input_sha256") or {}) for row in inputs_state["stages"].values()):
            raise PipelineError("selftest stage input hashes were not recorded")
        before = len((inputs_path.parent / "order.log").read_text().splitlines())
        run_pipeline(inputs_path, inputs_path.parent, processor, resume=True, script_root=root)
        after = (inputs_path.parent / "order.log").read_text().splitlines()
        if len(after) != before:
            raise PipelineError("selftest resume with unchanged inputs reran stages")
        inputs_state = _load_json(inputs_path.parent / "PIPELINE_STATE.json")
        if any(row.get("status") != "REUSED" for row in inputs_state["stages"].values()):
            raise PipelineError("selftest resume with unchanged inputs did not reuse all stages")
        extra_input.write_text("extra-v2", encoding="utf-8")
        run_pipeline(inputs_path, inputs_path.parent, processor, resume=True, script_root=root)
        after = (inputs_path.parent / "order.log").read_text().splitlines()
        if after[before:] != ["sw_capture.out", "candidate.perfetto.json.gz"]:
            raise PipelineError("selftest input change did not rerun exactly the affected stages")
        inputs_state = _load_json(inputs_path.parent / "PIPELINE_STATE.json")
        if [row.get("status") for row in inputs_state["stages"].values()] != ["REUSED", "REUSED", "REUSED", "PASS", "PASS"]:
            raise PipelineError("selftest input-change cascade statuses are wrong")

        # O1 --inherit: legacy contracts (no stage inputs) carrying a
        # components registry; per-stage scripts are the registered component
        # files so old-contract code entries alone can prove identity.
        tool_dir = root / "toolchain"
        tool_dir.mkdir()
        component_files = {}
        for key in ("offline_selftests", "correctness_validator", "overhead_validator",
                    "profile_entrypoint", "measurement_harness", "decoder", "decoder_entrypoint",
                    "hw_capture", "metrics_definition", "symmetric_memory_bypass", "template_abi",
                    "perfetto_pipeline", "perfetto_emitter", "sw_hw_aligner", "hw_extract"):
            suffix = ".json" if key in ("metrics_definition", "template_abi") else ".py"
            tool_file = tool_dir / (key + suffix)
            if suffix == ".py":
                tool_file.write_bytes(helper.read_bytes())
            else:
                tool_file.write_text(key + "-v1", encoding="utf-8")
            component_files[key] = tool_file

        def run_dir_for(name):
            return semantic_version_root / "results" / "profile_runs" / name

        def make_components(run):
            comps = {}
            for key, tool_file in component_files.items():
                comps[key] = {"absolute_path": str(tool_file), "role": key, "sha256": _sha256(tool_file)}
            abi_file = run / "abi" / "selftest_abi.materialized.json"
            abi_file.parent.mkdir(parents=True, exist_ok=True)
            abi_file.write_text("selftest-abi-v1", encoding="utf-8")
            comps["materialized_abi"] = {"absolute_path": str(abi_file), "role": "run-specific semantic ABI",
                                         "sha256": _sha256(abi_file)}
            return comps

        script_map = {"offline_selftest": component_files["offline_selftests"],
                      "correctness": component_files["correctness_validator"],
                      "overhead": component_files["overhead_validator"],
                      "sw_capture": component_files["profile_entrypoint"],
                      "hw_capture": component_files["hw_capture"],
                      "perfetto": component_files["perfetto_pipeline"]}
        stage_names_hw = list(BUNDLE_STAGES["sw+hw"])

        def stage_statuses(state):
            return {name: row.get("status") for name, row in state["stages"].items()}

        old_run = run_dir_for("inherit_old")
        old_path = make("sw+hw", "inherit_old", components=make_components(old_run), stage_scripts=script_map)
        run_pipeline(old_path, old_run, processor, script_root=root)
        if _load_json(old_run / "PIPELINE_STATE.json").get("status") != "PASS":
            raise PipelineError("selftest inherit base run failed")

        # inherit with every input unchanged: all stages REUSED (zero stage
        # executions), outputs hard-linked, candidate hash check satisfied.
        new_path_b = make("sw+hw", "inherit_all", components=make_components(run_dir_for("inherit_all")),
                          stage_scripts=script_map)
        run_pipeline(new_path_b, new_path_b.parent, processor, inherit=old_run, script_root=root)
        state_b = _load_json(new_path_b.parent / "PIPELINE_STATE.json")
        if state_b.get("status") != "PASS" or any(row.get("status") != "REUSED" for row in state_b["stages"].values()):
            raise PipelineError("selftest inherit did not reuse every stage")
        if any(not row.get("inherited_from") or not row.get("inherit_evidence") for row in state_b["stages"].values()):
            raise PipelineError("selftest inherit evidence chain missing")
        if (new_path_b.parent / "order.log").exists():
            raise PipelineError("selftest inherit run executed a stage")
        for stage in stage_names_hw:
            filename = "candidate.perfetto.json.gz" if stage == "perfetto" else stage + ".out"
            if not os.path.samefile(str(old_run / filename), str(new_path_b.parent / filename)):
                raise PipelineError("selftest inherit outputs are not hard links")
        sources_b = {entry.get("source") for row in state_b["stages"].values() for entry in row.get("inherit_evidence", [])}
        if "state_input" not in sources_b:
            raise PipelineError("selftest inherit did not use state input evidence")

        # stripping state input evidence must fall back to the old contract
        # code registry and still reuse every stage.
        old_state = _load_json(old_run / "PIPELINE_STATE.json")
        saved_inputs = {name: row.get("input_sha256") for name, row in old_state["stages"].items()}
        for row in old_state["stages"].values():
            row.pop("input_sha256", None)
        _write_json_atomic(old_run / "PIPELINE_STATE.json", old_state)
        new_path_d = make("sw+hw", "inherit_code", components=make_components(run_dir_for("inherit_code")),
                          stage_scripts=script_map)
        run_pipeline(new_path_d, new_path_d.parent, processor, inherit=old_run, script_root=root)
        state_d = _load_json(new_path_d.parent / "PIPELINE_STATE.json")
        if any(row.get("status") != "REUSED" for row in state_d["stages"].values()):
            raise PipelineError("selftest inherit via contract code evidence failed")
        sources_d = {entry.get("source") for row in state_d["stages"].values() for entry in row.get("inherit_evidence", [])}
        if "contract_code" not in sources_d:
            raise PipelineError("selftest inherit did not use contract code evidence")
        for name, value in saved_inputs.items():
            if value is not None:
                old_state["stages"][name]["input_sha256"] = value
        _write_json_atomic(old_run / "PIPELINE_STATE.json", old_state)

        # the target scenario: a perfetto-stage tool (emit_perfetto analog) is
        # fixed, the contract is re-materialized into a new run dir, and
        # --inherit reuses the five measurement stages while perfetto and
        # everything after it reruns.
        component_files["perfetto_emitter"].write_text("# emit_perfetto fixed\n", encoding="utf-8")
        new_path_a = make("sw+hw", "inherit_tool_fix", components=make_components(run_dir_for("inherit_tool_fix")),
                          stage_scripts=script_map)
        run_pipeline(new_path_a, new_path_a.parent, processor, inherit=old_run, script_root=root)
        state_a = _load_json(new_path_a.parent / "PIPELINE_STATE.json")
        expected_a = {"offline_selftest": "REUSED", "correctness": "REUSED", "overhead": "REUSED",
                      "sw_capture": "REUSED", "hw_capture": "REUSED", "perfetto": "PASS"}
        if stage_statuses(state_a) != expected_a:
            raise PipelineError("selftest inherit tool-fix cascade statuses are wrong")
        if (new_path_a.parent / "order.log").read_text().splitlines() != ["candidate.perfetto.json.gz"]:
            raise PipelineError("selftest inherit tool-fix reran more than perfetto")

        # a tampered old output must fail closed instead of being inherited.
        (old_run / "correctness.out").write_text("tampered-after-run", encoding="utf-8")
        new_path_c = make("sw+hw", "inherit_tamper", components=make_components(run_dir_for("inherit_tamper")),
                          stage_scripts=script_map)
        try:
            run_pipeline(new_path_c, new_path_c.parent, processor, inherit=old_run, script_root=root)
        except PipelineError:
            pass
        else:
            raise PipelineError("selftest accepted tampered inherit evidence")

        # declared stage inputs inherit: perfetto declares the upstream
        # sw_capture output; with its state input evidence stripped the
        # upstream output_sha256 record must prove identity.
        e_old = run_dir_for("inherit_decl_old")
        e_new = run_dir_for("inherit_decl_new")
        e_inputs_old = {"offline_selftest": [str(component_files["offline_selftests"])],
                        "correctness": [str(component_files["correctness_validator"])],
                        "overhead": [str(component_files["overhead_validator"])],
                        "sw_capture": [str(component_files["profile_entrypoint"])],
                        "hw_capture": [str(component_files["hw_capture"])],
                        "perfetto": [str(component_files["perfetto_pipeline"]), str(e_old / "sw_capture.out")]}
        e_path_old = make("sw+hw", "inherit_decl_old", components=make_components(e_old),
                          stage_inputs=e_inputs_old, stage_scripts=script_map)
        run_pipeline(e_path_old, e_old, processor, script_root=root)
        e_state = _load_json(e_old / "PIPELINE_STATE.json")
        e_state["stages"]["perfetto"].pop("input_sha256", None)
        _write_json_atomic(e_old / "PIPELINE_STATE.json", e_state)
        e_inputs_new = dict(e_inputs_old)
        e_inputs_new["perfetto"] = [str(component_files["perfetto_pipeline"]), str(e_new / "sw_capture.out")]
        e_path_new = make("sw+hw", "inherit_decl_new", components=make_components(e_new),
                          stage_inputs=e_inputs_new, stage_scripts=script_map)
        run_pipeline(e_path_new, e_new, processor, inherit=e_old, script_root=root)
        state_e = _load_json(e_new / "PIPELINE_STATE.json")
        if any(row.get("status") != "REUSED" for row in state_e["stages"].values()):
            raise PipelineError("selftest declared-input inherit failed")
        perfetto_sources = {entry.get("source") for entry in state_e["stages"]["perfetto"].get("inherit_evidence", [])}
        if "state_output" not in perfetto_sources:
            raise PipelineError("selftest inherit did not use upstream output evidence")

        # O6: parallel groups run GPU stages concurrently only when the
        # whitelist can give every member a disjoint card set; otherwise the
        # group silently falls back to the serial path.
        def fake_gpus(timeout=30.0):
            return [{"index": idx, "memory_used_mib": 0.0, "utilization_percent": 0.0} for idx in range(4)]

        original_query = globals()["_query_gpus_nvidia_smi"]
        globals()["_query_gpus_nvidia_smi"] = fake_gpus
        try:
            par_modes = {"correctness": "psleep", "hw_capture": "psleep"}
            par_path = make("sw+hw", "parallel_ok", allowed=[0, 1, 2, 3],
                            parallel=[["correctness", "hw_capture"]],
                            gpus={"correctness": 2, "hw_capture": 2}, stage_modes=par_modes)
            run_pipeline(par_path, par_path.parent, processor, script_root=root)
            par_state = _load_json(par_path.parent / "PIPELINE_STATE.json")
            if par_state.get("status") != "PASS":
                raise PipelineError("selftest parallel run failed")
            if (par_state.get("parallel_groups") or {}).get("active") != [["correctness", "hw_capture"]]:
                raise PipelineError("selftest parallel group was not activated")
            spans = {rec["name"]: (rec["start"], rec["end"]) for rec in
                     (json.loads(line) for line in (par_path.parent / "parallel_timing.jsonl").read_text().splitlines() if line.strip())}
            if not (spans["correctness"][0] < spans["hw_capture"][1] and spans["hw_capture"][0] < spans["correctness"][1]):
                raise PipelineError("selftest parallel stages did not overlap")
            bound_c = set(par_state["stages"]["correctness"].get("bound_gpus") or [])
            bound_h = set(par_state["stages"]["hw_capture"].get("bound_gpus") or [])
            if len(bound_c) != 2 or len(bound_h) != 2 or bound_c & bound_h or not (bound_c | bound_h) <= {0, 1, 2, 3}:
                raise PipelineError("selftest parallel GPU sets are not disjoint whitelist subsets")
            serial_path = make("sw+hw", "parallel_noop", allowed=[1, 3],
                               parallel=[["correctness", "hw_capture"]],
                               gpus={"correctness": 2, "hw_capture": 2}, stage_modes=par_modes)
            run_pipeline(serial_path, serial_path.parent, processor, script_root=root)
            serial_state = _load_json(serial_path.parent / "PIPELINE_STATE.json")
            if serial_state.get("status") != "PASS":
                raise PipelineError("selftest parallel fallback run failed")
            if (serial_state.get("parallel_groups") or {}).get("active") != []:
                raise PipelineError("selftest parallel fallback should stay serial")
            spans = {rec["name"]: (rec["start"], rec["end"]) for rec in
                     (json.loads(line) for line in (serial_path.parent / "parallel_timing.jsonl").read_text().splitlines() if line.strip())}
            if spans["hw_capture"][0] < spans["correctness"][1]:
                raise PipelineError("selftest parallel fallback overlapped stages")
        finally:
            globals()["_query_gpus_nvidia_smi"] = original_query

        bad_group_path = make("sw", "bad_parallel")
        bad_group = _load_json(bad_group_path)
        bad_group["execution"]["parallel_groups"] = [["correctness", "nonsense"]]
        bad_group_path.write_text(json.dumps(bad_group), encoding="utf-8")
        try:
            run_pipeline(bad_group_path, bad_group_path.parent, processor, script_root=root)
        except PipelineError:
            pass
        else:
            raise PipelineError("selftest accepted a parallel group with an unknown stage")
        bad_kind_path = make("sw", "bad_parallel_kind")
        bad_kind = _load_json(bad_kind_path)
        bad_kind["execution"]["parallel_groups"] = [["offline_selftest", "correctness"]]
        bad_kind_path.write_text(json.dumps(bad_kind), encoding="utf-8")
        try:
            run_pipeline(bad_kind_path, bad_kind_path.parent, processor, script_root=root)
        except PipelineError:
            pass
        else:
            raise PipelineError("selftest accepted a parallel group with a CPU stage")
    print("SELFTEST PASS: preflight, bundle order, rejection, fail-closed delivery, resume hash invalidation, stage input reuse, inherit evidence chain, parallel groups")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="fast-instrument-v2 measurement runner")
    parser.add_argument("--selftest", action="store_true", help="run no-GPU selftest")
    parser.add_argument("--gpu-probe", action="store_true", help="classify GPUs free/busy via nvidia-smi and exit")
    parser.add_argument("--allowed-gpu-ids", help="comma-separated GPU indices for --gpu-probe whitelist selection")
    parser.add_argument("--contract"); parser.add_argument("--run-dir"); parser.add_argument("--trace-processor"); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inherit", help="reuse byte-identical stages from an older run dir whose contract hash differs")
    args = parser.parse_args(argv)
    if args.selftest: return _selftest()
    if args.gpu_probe:
        allowed = None
        if args.allowed_gpu_ids:
            try:
                allowed = [int(token) for token in args.allowed_gpu_ids.split(",") if token.strip() != ""]
            except ValueError:
                parser.error("--allowed-gpu-ids must be comma-separated integers")
        try:
            print(json.dumps(_gpu_probe(allowed), indent=2)); return 0
        except PipelineError as error:
            print("[fatal] {}".format(error), file=sys.stderr); return 2
    if not args.contract or not args.run_dir or not args.trace_processor: parser.error("formal measurement requires --contract --run-dir --trace-processor")
    try:
        state = run_pipeline(args.contract, args.run_dir, args.trace_processor, args.resume, inherit=args.inherit); print("PIPELINE_STATE={}".format(Path(args.run_dir).resolve() / "PIPELINE_STATE.json")); print("STATUS={}".format(state["status"])); return 0
    except PipelineError as error:
        print("[fatal] {}".format(error), file=sys.stderr); return 2


if __name__ == "__main__":
    sys.exit(main())
