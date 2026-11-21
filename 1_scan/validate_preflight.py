#!/usr/bin/env python3
"""Contract-driven E01-E27 preflight runner.

All applicable checks are collected in one run.  Commands are argv-only and
are always launched with shell=False.  The output is atomically replaced only
after the complete result has been assembled.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

from semantic_contract import (semantic_conclusion_sha256, sha256_file as semantic_sha256_file,
                               validate_abi_lock, validate_semantic_contract)


ALL_CHECK_IDS = ["E{:02d}".format(index) for index in range(1, 28)]

# Defect 3: the skill hardcodes which preflight checks may be declared
# not-applicable per information bundle.  A contract selects only a bundle and
# can never freely pick N/A IDs.  sw+hw keeps an empty N/A set, so its behaviour
# is byte-identical to the previous "all E01-E27 required" policy.  When the
# bundle is unknown or absent the legacy no-N/A behaviour is used.
BUNDLE_NA_CHECK_IDS = {
    "sw": ("E17", "E18", "E19", "E20"),
    "hw": ("E02", "E03", "E05", "E06", "E07", "E08", "E09", "E10", "E12",
           "E13", "E15", "E16", "E20"),
    "sw+hw": (),
}


def _bundle_na_check_ids(bundle):
    if bundle in BUNDLE_NA_CHECK_IDS:
        return set(BUNDLE_NA_CHECK_IDS[bundle])
    return None


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def absolute_path(value, base):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def safe_component(value):
    text = str(value)
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in text)


def contains_example_placeholder(value):
    if isinstance(value, str):
        return "EXAMPLE_ONLY" in value
    if isinstance(value, dict):
        return any(contains_example_placeholder(key) or contains_example_placeholder(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(contains_example_placeholder(item) for item in value)
    return False


def is_absolute_contract_path(value):
    if not isinstance(value, str) or not value:
        return False
    return (value.startswith("/") or value.startswith("\\\\") or
            (len(value) >= 3 and value[1] == ":" and value[2] in ("/", "\\")))


def path_is_below_profile_runs(run_root, version_root):
    normalized_run = run_root.replace("\\", "/")
    normalized_version = version_root.replace("\\", "/")
    if normalized_run.startswith("/") and normalized_version.startswith("/"):
        child = PurePosixPath(normalized_run)
        parent = PurePosixPath(normalized_version) / "results" / "profile_runs"
    else:
        child = PureWindowsPath(run_root)
        parent = PureWindowsPath(version_root) / "results" / "profile_runs"
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_contract_schema(contract):
    errors = []
    if not isinstance(contract, dict):
        return ["contract root must be an object"]
    if contains_example_placeholder(contract):
        errors.append("EXAMPLE_ONLY placeholder is forbidden anywhere in contract")
    if contract.get("example_only") is True:
        errors.append("example_only contract is rejected")
    if contract.get("production_validation_expected") == "REJECT":
        errors.append("production_validation_expected=REJECT")

    lock = contract.get("lock")
    if not isinstance(lock, dict):
        errors.append("lock must be an object")
    else:
        if lock.get("state") != "frozen":
            errors.append("lock.state must be frozen")
        if lock.get("allow_in_place_edit") is not False:
            errors.append("lock.allow_in_place_edit must be false")

    layout = contract.get("project_layout")
    if not isinstance(layout, dict):
        errors.append("project_layout must be an object")
    else:
        version_root = layout.get("version_root")
        run_root = layout.get("run_root")
        if not is_absolute_contract_path(version_root):
            errors.append("project_layout.version_root must be absolute")
        if not is_absolute_contract_path(run_root):
            errors.append("project_layout.run_root must be absolute")
        if is_absolute_contract_path(version_root) and is_absolute_contract_path(run_root):
            if not path_is_below_profile_runs(run_root, version_root):
                errors.append("project_layout.run_root must be below version_root/results/profile_runs")

    run = contract.get("run")
    if not isinstance(run, dict) or not isinstance(run.get("target_kernel"), str) or not run.get("target_kernel").strip():
        errors.append("run.target_kernel must be a non-empty string")

    capture = contract.get("capture")
    if not isinstance(capture, dict):
        errors.append("capture must be an object")
    else:
        if capture.get("information_bundle") not in ("sw", "hw", "sw+hw"):
            errors.append("capture.information_bundle must be sw, hw, or sw+hw")
        display_rank = capture.get("display_rank")
        if isinstance(display_rank, bool) or not isinstance(display_rank, int) or display_rank < 0:
            errors.append("capture.display_rank must be a non-negative integer")
        granularity = capture.get("granularity", capture.get("display_granularity"))
        if granularity not in ("task", "phase"):
            errors.append("capture granularity must be task or phase")
        sm_scope = capture.get("software_sm_scope")
        if not isinstance(sm_scope, dict) or sm_scope.get("mode") not in ("all", "selected"):
            errors.append("capture.software_sm_scope.mode must be all or selected")
        elif sm_scope.get("mode") == "selected":
            selected = sm_scope.get("selected_physical_smids")
            if not isinstance(selected, list) or not selected:
                errors.append("selected software_sm_scope requires a non-empty list")

    input_config = contract.get("input")
    if not isinstance(input_config, dict) or input_config.get("generator") != "randn":
        errors.append("input.generator must be randn")
    execution = contract.get("execution")
    if not isinstance(execution, dict):
        errors.append("execution must be an object")
    elif not isinstance(execution.get("preflight"), dict):
        errors.append("execution.preflight must be an object")

    semantic_errors = validate_semantic_contract(contract, base_dir=".", verify_sources=True)
    errors.extend("semantic contract: {}".format(error) for error in semantic_errors)
    abi_errors, _abi = validate_abi_lock(contract, base_dir=".")
    errors.extend("semantic ABI lock: {}".format(error) for error in abi_errors)
    return errors


def validate_partition(preflight, bundle=None):
    required = preflight.get("required_check_ids")
    not_applicable = preflight.get("not_applicable_check_ids")
    errors = []
    if not isinstance(required, list):
        errors.append("required_check_ids must be a list")
        required = []
    if not isinstance(not_applicable, list):
        errors.append("not_applicable_check_ids must be a list")
        not_applicable = []

    def inspect(name, values):
        non_strings = [value for value in values if not isinstance(value, str)]
        string_values = [value for value in values if isinstance(value, str)]
        duplicates = sorted({value for value in string_values if string_values.count(value) > 1})
        unknown = sorted({value for value in string_values if value not in ALL_CHECK_IDS})
        if non_strings:
            errors.append("{} must contain strings only: {}".format(
                name, ", ".join(json.dumps(value, sort_keys=True, ensure_ascii=True) for value in non_strings)))
        if duplicates:
            errors.append("{} contains duplicates: {}".format(name, ", ".join(map(str, duplicates))))
        if unknown:
            errors.append("{} contains unknown IDs: {}".format(name, ", ".join(map(str, unknown))))

    inspect("required_check_ids", required)
    inspect("not_applicable_check_ids", not_applicable)
    required_strings = {value for value in required if isinstance(value, str)}
    not_applicable_strings = {value for value in not_applicable if isinstance(value, str)}
    overlap = sorted(required_strings.intersection(not_applicable_strings))
    if overlap:
        errors.append("required/not_applicable overlap: {}".format(", ".join(overlap)))
    missing = sorted(set(ALL_CHECK_IDS).difference(required_strings.union(not_applicable_strings)))
    if missing:
        errors.append("partition missing IDs: {}".format(", ".join(missing)))
    na_bundle = _bundle_na_check_ids(bundle)
    if na_bundle is None:
        na_bundle = set()
    expected_required = set(ALL_CHECK_IDS) - na_bundle
    if (required_strings == expected_required and not_applicable_strings == na_bundle and
            len(required) == len(expected_required) and len(not_applicable) == len(na_bundle)):
        pass
    elif na_bundle:
        errors.append("policy requires the {} bundle partition: required must be E01-E27 "
                      "minus {}, not_applicable must be exactly {}".format(
                          bundle, sorted(na_bundle), sorted(na_bundle)))
    else:
        errors.append("policy requires all E01-E27 required and no not_applicable IDs")
    return required, not_applicable, errors


def substitute_template(value, token, check_id):
    if isinstance(value, str):
        return value.replace(token, check_id)
    if isinstance(value, list):
        return [substitute_template(item, token, check_id) for item in value]
    if isinstance(value, dict):
        return {key: (item if key in ("sha256", "expected_sha256") else substitute_template(item, token, check_id))
                for key, item in value.items()}
    return value


def parse_command(commands, check_id, template=None):
    if isinstance(commands, dict) and check_id in commands:
        value = commands[check_id]
    elif template is not None:
        template_argv = template if isinstance(template, list) else (
            template.get("argv") if isinstance(template, dict) else None)
        if not isinstance(template_argv, list) or not any(
            isinstance(item, str) and "{check_id}" in item for item in template_argv
        ):
            return None, "check_command_template argv must contain {check_id}"
        value = substitute_template(template, "{check_id}", check_id)
    else:
        return None, "missing command"
    if isinstance(value, list):
        argv = value
        options = {}
    elif isinstance(value, dict):
        argv = value.get("argv")
        options = value
        if "shell" in value:
            return None, "shell field is forbidden"
    else:
        return None, "command must be an argv list"
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        return None, "command must be an argv list of strings"
    if any(not item for item in argv):
        return None, "command argv contains an empty argument"
    if any(item.startswith("--shell") or item.startswith("--skip") for item in argv):
        return None, "shell and skip argv options are forbidden"
    if isinstance(options, dict) and any(key.startswith("skip") for key in options):
        return None, "skip fields are forbidden"
    return {"argv": list(argv), "options": options}, None


def evidence_records(spec, run_dir):
    values = spec.get("options", {}).get("evidence", [])
    if isinstance(values, (str, dict)):
        values = [values]
    if not isinstance(values, list):
        return [], "evidence must be a path or list"
    records = []
    for item in values:
        expected = None
        value = item
        if isinstance(item, dict):
            value = item.get("path", item.get("absolute_path"))
            expected = item.get("sha256")
        if not isinstance(value, str) or not value:
            return [], "evidence entries must contain a path"
        path = absolute_path(value, run_dir)
        record = {"path": str(path), "exists": path.is_file()}
        if isinstance(expected, str):
            record["expected_sha256"] = expected
        if path.is_file():
            try:
                record["sha256"] = sha256_file(path)
            except OSError as error:
                record["read_error"] = str(error)
            if isinstance(expected, str) and "sha256" in record:
                record["sha256_match"] = record["sha256"] == expected
        records.append(record)
    return records, None


def write_log(path, content):
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    Path(path).write_text(content or "", encoding="utf-8", errors="replace")


def base_record(check_id, run_dir):
    log_dir = run_dir / "preflight_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_component(check_id)
    return {
        "check_id": check_id,
        "status": "BLOCKED",
        "command": None,
        "cwd": None,
        "timeout_sec": None,
        "started_at_utc": utc_now(),
        "ended_at_utc": None,
        "exit_code": None,
        "stdout_log": str((log_dir / (stem + ".stdout.log")).resolve()),
        "stderr_log": str((log_dir / (stem + ".stderr.log")).resolve()),
        "evidence": [],
        "reason": None,
    }


def finish(record, status, reason=None):
    record["status"] = status
    record["reason"] = reason
    record["ended_at_utc"] = utc_now()
    return record


def run_check(check_id, spec, parse_error, run_dir):
    record = base_record(check_id, run_dir)
    if spec is None:
        write_log(record["stdout_log"], "")
        write_log(record["stderr_log"], parse_error or "missing command")
        return finish(record, "BLOCKED", parse_error or "missing command")

    options = spec["options"]
    record["command"] = list(spec["argv"])
    cwd = options.get("cwd", str(run_dir))
    record["cwd"] = str(absolute_path(cwd, run_dir))
    env_value = options.get("env", {})
    if env_value is None:
        env_value = {}
    if not isinstance(env_value, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in env_value.items()
    ):
        reason = "env must be a string-to-string mapping"
        write_log(record["stdout_log"], "")
        write_log(record["stderr_log"], reason)
        return finish(record, "BLOCKED", reason)

    timeout_sec = options.get("timeout_sec")
    if timeout_sec is not None and (
        isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float)) or timeout_sec <= 0
    ):
        reason = "timeout_sec must be a positive number"
        write_log(record["stdout_log"], "")
        write_log(record["stderr_log"], reason)
        return finish(record, "BLOCKED", reason)
    record["timeout_sec"] = timeout_sec
    evidence, evidence_error = evidence_records(spec, run_dir)
    if evidence_error:
        write_log(record["stdout_log"], "")
        write_log(record["stderr_log"], evidence_error)
        return finish(record, "BLOCKED", evidence_error)

    try:
        child_env = os.environ.copy()
        child_env.update(env_value)
        completed = subprocess.run(
            spec["argv"], cwd=record["cwd"], env=child_env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, shell=False, check=False, timeout=timeout_sec,
        )
        record["exit_code"] = completed.returncode
        write_log(record["stdout_log"], completed.stdout)
        write_log(record["stderr_log"], completed.stderr)
    except subprocess.TimeoutExpired as error:
        write_log(record["stdout_log"], error.stdout)
        write_log(record["stderr_log"], error.stderr)
        record["timed_out"] = True
        return finish(record, "FAIL", "command timed out after {} seconds".format(timeout_sec))
    except (OSError, ValueError) as error:
        reason = "command could not be executed: {}".format(error)
        write_log(record["stdout_log"], "")
        write_log(record["stderr_log"], reason)
        return finish(record, "BLOCKED", reason)

    evidence, evidence_error = evidence_records(spec, run_dir)
    record["evidence"] = evidence
    missing = [item["path"] for item in evidence if not item["exists"] or "read_error" in item]
    mismatched = [item["path"] for item in evidence if item.get("sha256_match") is False]
    if record["exit_code"] != 0:
        return finish(record, "FAIL", "command exited with non-zero status")
    if evidence_error or missing:
        return finish(record, "BLOCKED", evidence_error or "required evidence is missing")
    if mismatched:
        return finish(record, "FAIL", "evidence SHA256 mismatch")
    return finish(record, "PASS")


def atomic_dump(payload, out):
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(out) + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(partial), str(out))


def validate(args):
    contract_path = Path(args.contract).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    contract = {}
    contract_hash = None
    try:
        contract_hash = sha256_file(contract_path)
        contract = load_json(contract_path)
        if not isinstance(contract, dict):
            errors.append("contract root must be an object")
            contract = {}
    except Exception as error:
        errors.append("contract error: {}".format(error))

    schema_errors = validate_contract_schema(contract)
    errors.extend(schema_errors)

    execution = contract.get("execution") if isinstance(contract, dict) else {}
    if not isinstance(execution, dict):
        execution = {}
    preflight = execution.get("preflight")
    if not isinstance(preflight, dict):
        preflight = {}
    capture = contract.get("capture") if isinstance(contract, dict) else {}
    if not isinstance(capture, dict):
        capture = {}
    bundle = capture.get("information_bundle")
    required, not_applicable, partition_errors = validate_partition(preflight, bundle)
    errors.extend(partition_errors)
    commands = preflight.get("check_commands")
    command_template = preflight.get("check_command_template")

    if partition_errors or schema_errors or errors:
        records = []
        for check_id in required:
            record = base_record(check_id, run_dir)
            reason = "run-level validation failed; check not executed"
            write_log(record["stdout_log"], "")
            write_log(record["stderr_log"], reason)
            records.append(finish(record, "BLOCKED", reason))
    else:
        specs = {}
        parse_errors = {}
        for check_id in required:
            specs[check_id], parse_errors[check_id] = parse_command(commands, check_id, command_template)
        with ThreadPoolExecutor(max_workers=max(1, len(required))) as pool:
            futures = {
                pool.submit(run_check, check_id, specs[check_id], parse_errors[check_id], run_dir): check_id
                for check_id in required
            }
            completed = {futures[future]: future.result() for future in as_completed(futures)}
        records = [completed[check_id] for check_id in required]

    contract_hash_after = None
    integrity_errors = []
    if contract_path.is_file():
        try:
            contract_hash_after = sha256_file(contract_path)
        except Exception as error:
            integrity_errors.append("contract post-run hash error: {}".format(error))
    if contract_hash_after != contract_hash:
        integrity_errors.append("contract changed during check execution")
    errors.extend(integrity_errors)

    counts = {status: sum(1 for record in records if record["status"] == status)
              for status in ("PASS", "FAIL", "BLOCKED")}
    all_pass = not errors and counts["FAIL"] == 0 and counts["BLOCKED"] == 0
    payload = {
        "schema_version": "preflight.v1",
        "status": "PASS" if all_pass else "FAIL",
        "generated_at_utc": utc_now(),
        "contract_path": str(contract_path),
        "contract_sha256": contract_hash,
        "contract_sha256_after": contract_hash_after,
        "integrity_errors": integrity_errors,
        "contract_validation": {
            "status": "PASS" if not schema_errors else "FAIL",
            "errors": schema_errors,
        },
        "run_dir": str(run_dir),
        "out_path": str(out_path),
        "required_check_ids": required,
        "not_applicable_check_ids": not_applicable,
        "partition_errors": partition_errors,
        "input_errors": errors,
        "checks": records,
        "summary": {
            "required_count": len(required),
            "pass": counts["PASS"],
            "fail": counts["FAIL"],
            "blocked": counts["BLOCKED"],
            "all_required_pass": all_pass,
        },
    }
    try:
        atomic_dump(payload, out_path)
    except Exception as error:
        print("fatal: cannot write {}: {}".format(out_path, error), file=sys.stderr)
        return 2
    print("PREFLIGHT={}".format(out_path))
    print("STATUS={}".format(payload["status"]))
    return 0 if all_pass else 1


def selftest():
    root = Path(__file__).resolve().parent / ".validate_preflight_selftest"
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run_", dir=str(root)))
    try:
        py = sys.executable
        evidence = work / "evidence.json"
        write_code = "from pathlib import Path; Path('evidence.json').write_text('ok')"
        required = list(ALL_CHECK_IDS)
        not_applicable = []
        passing = {
            check_id: {"argv": [py, "-c", write_code],
                       "evidence": ["evidence.json"] if check_id == "E01" else []}
            for check_id in ALL_CHECK_IDS
        }
        version_root = work / "version"
        version_root.mkdir(parents=True, exist_ok=True)
        semantic_source = version_root / "kernel.py"
        semantic_source.write_text(
            "semantic_task_begin()\nsemantic_task_sync()\nsemantic_task_end()\n",
            encoding="utf-8")
        task_contract = {
            "classification": "task",
            "semantic_unit": "one selftest tile",
            "instance_key": ["rank", "tile"],
            "instance_domain": {"axes": {"rank": [0], "tile": [0]}},
            "expected_cardinality": 1,
            "collaborating_warps": [0],
            "recording_representative": {"warp": 0, "lane": 0, "thread_idx": 0},
            "tile_shape": {"rows": 1, "columns": 1},
            "source_boundary": {
                "begin": {"path": "kernel.py", "line": 1, "anchor": "semantic_task_begin"},
                "completion_sync": {"path": "kernel.py", "line": 2, "anchor": "semantic_task_sync"},
                "end": {"path": "kernel.py", "line": 3, "anchor": "semantic_task_end"},
            },
        }
        semantic = {
            "schema": "fast-instrument-semantic-analysis/v1",
            "model_reviewed": True,
            "source_closure": [{"path": "kernel.py",
                                "sha256": semantic_sha256_file(semantic_source)}],
        }
        descriptor = {"id": 1, "name": "Selftest Tile", "display_name": "Selftest Tile",
                      "layer": "task", "task_contract": task_contract}
        abi = {"semantic_analysis": semantic, "tasks": [descriptor]}
        semantic["conclusion_sha256"] = semantic_conclusion_sha256(abi)
        abi_path = version_root / "profile" / "device_record" / "abi" / "selftest.json"
        abi_path.parent.mkdir(parents=True, exist_ok=True)
        abi_path.write_text(json.dumps(abi, sort_keys=True), encoding="utf-8")
        abi_sha256 = semantic_sha256_file(abi_path)

        def contract_for(commands, bundle="sw+hw", required_ids=None, na_ids=None, **extra):
            run_root = version_root / "results" / "profile_runs" / "run"
            contract_semantic = json.loads(json.dumps(semantic))
            contract_semantic["abi"] = {"path": str(abi_path), "sha256": abi_sha256}
            payload = {
                "lock": {"state": "frozen", "allow_in_place_edit": False},
                "project_layout": {"version_root": str(version_root), "run_root": str(run_root)},
                "run": {"target_kernel": "selftest_kernel"},
                "capture": {"information_bundle": bundle, "display_rank": 0,
                             "granularity": "task", "software_sm_scope": {"mode": "all"}},
                "input": {"generator": "randn"},
                "semantic_analysis": contract_semantic,
                "events": {"event_definitions": [json.loads(json.dumps(descriptor))]},
                "execution": {"preflight": {
                    "required_check_ids": required if required_ids is None else required_ids,
                    "not_applicable_check_ids": not_applicable if na_ids is None else na_ids,
                    "check_commands": commands,
                }},
            }
            payload.update(extra)
            return payload

        def invoke(name, payload, expect_zero):
            contract_path = work / (name + ".contract.json")
            out_path = work / (name + ".PREFLIGHT.json")
            contract_path.write_text(json.dumps(payload), encoding="utf-8")
            code = validate(argparse.Namespace(contract=str(contract_path), run_dir=str(work), out=str(out_path)))
            if (code == 0) != expect_zero or not out_path.is_file():
                raise AssertionError(name + " returned unexpected result")
            return load_json(out_path), contract_path, out_path

        result, contract_path, out_path = invoke("all_pass", contract_for(passing), True)
        assert result["checks"][0]["status"] == "PASS"
        assert len(result["contract_sha256"]) == 64
        assert result["checks"][0]["evidence"][0].get("sha256")

        # Defect 3: the sw bundle allows exactly its hardcoded HW-only N/A set.
        sw_na = ["E17", "E18", "E19", "E20"]
        sw_required = [check_id for check_id in ALL_CHECK_IDS if check_id not in sw_na]
        sw_result, _, _ = invoke("sw_bundle_ok", contract_for(
            passing, bundle="sw", required_ids=sw_required, na_ids=sw_na), True)
        assert sorted(sw_result["not_applicable_check_ids"]) == sw_na
        assert not sw_result["partition_errors"]
        # Negative: marking a mandatory SW check (E01) N/A under sw is rejected.
        sw_bad_na = [check_id for check_id in sw_required if check_id != "E01"]
        sw_bad_result, _, _ = invoke("sw_bundle_bad_na", contract_for(
            passing, bundle="sw", required_ids=sw_bad_na, na_ids=["E01"] + sw_na), False)
        assert sw_bad_result["partition_errors"]

        template_payload = contract_for({})
        template_payload["execution"]["preflight"]["check_command_template"] = {
            "argv": [py, "-c", "from pathlib import Path; Path('{check_id}.marker').write_text('ok')"],
            "evidence": [{"path": "{check_id}.marker"}],
        }
        template_result, _, _ = invoke("template_all", template_payload, True)
        assert all(check["status"] == "PASS" for check in template_result["checks"])

        missing_partition = contract_for(passing)
        missing_partition["execution"]["preflight"]["required_check_ids"] = required[:-1]
        invoke("missing_id", missing_partition, False)

        invalid_element = contract_for(passing)
        invalid_element["execution"]["preflight"]["required_check_ids"] = [["E01"]] + required[1:]
        result, _, _ = invoke("invalid_partition_element", invalid_element, False)
        assert result["partition_errors"]

        result, _, _ = invoke("missing_command", contract_for({}), False)
        assert result["checks"][0]["status"] == "BLOCKED"

        failing = {"E01": {"argv": [py, "-c", "raise SystemExit(3)"]}}
        result, _, _ = invoke("nonzero", contract_for(failing), False)
        assert result["checks"][0]["status"] == "FAIL"

        timeout = {"E01": {"argv": [py, "-c", "import time; time.sleep(1)"], "timeout_sec": 0.05}}
        result, _, _ = invoke("timeout", contract_for(timeout), False)
        assert result["checks"][0]["status"] == "FAIL"
        assert result["checks"][0].get("timed_out") is True
        assert Path(result["checks"][0]["stdout_log"]).is_file()
        assert Path(result["checks"][0]["stderr_log"]).is_file()

        absent = {"E01": {"argv": [py, "-c", "pass"], "evidence": ["absent.bin"]}}
        result, _, _ = invoke("missing_evidence", contract_for(absent), False)
        assert result["checks"][0]["status"] == "BLOCKED"

        result, _, _ = invoke("example_rejected", contract_for(
            passing, example_only=True,
            metadata={"placeholder": "EXAMPLE_ONLY_NOT_A_REAL_VALUE"},
        ), False)
        assert result["checks"][0]["status"] == "BLOCKED"
        assert result["input_errors"]

        shell_field = dict(passing)
        shell_field["E01"] = {"argv": [py, "-c", "pass"], "shell": False}
        result, _, _ = invoke("shell_field", contract_for(shell_field), False)
        assert result["checks"][0]["status"] == "BLOCKED"
        skip_argv = dict(passing)
        skip_argv["E01"] = {"argv": [py, "--skip-check"]}
        result, _, _ = invoke("skip_argv", contract_for(skip_argv), False)
        assert result["checks"][0]["status"] == "BLOCKED"

        mutation_path = work / "contract_mutation.contract.json"
        mutation_out = work / "contract_mutation.PREFLIGHT.json"
        mutation_code = "from pathlib import Path; p=Path({}); p.write_text(p.read_text() + '\\n')".format(
            repr(str(mutation_path)))
        mutation_commands = dict(passing)
        mutation_commands["E01"] = {"argv": [py, "-c", mutation_code]}
        mutation_path.write_text(json.dumps(contract_for(mutation_commands)), encoding="utf-8")
        code = validate(argparse.Namespace(contract=str(mutation_path), run_dir=str(work), out=str(mutation_out)))
        assert code != 0
        mutation_result = load_json(mutation_out)
        assert "contract changed during check execution" in mutation_result["integrity_errors"]
        assert mutation_result["contract_sha256_after"] != mutation_result["contract_sha256"]

        print("SELFTEST=PASS")
        return 0
    except Exception as error:
        print("SELFTEST=FAIL: {}".format(error), file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(str(work), ignore_errors=True)
        try:
            root.rmdir()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Run contract-driven E01-E27 preflight checks")
    parser.add_argument("--contract")
    parser.add_argument("--run-dir")
    parser.add_argument("--out")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    missing = [name for name in ("contract", "run_dir", "out") if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: {}".format(", ".join("--" + name.replace("_", "-") for name in missing)))
    return validate(args)


if __name__ == "__main__":
    sys.exit(main())
