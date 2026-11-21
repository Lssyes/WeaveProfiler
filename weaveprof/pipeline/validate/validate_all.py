#!/usr/bin/env python3
"""Run all contract-declared validation gates and emit one atomic ALL_GATES.json.

The validator deliberately uses argv-only subprocesses and runs every applicable
gate (including after a failure).  It is Python 3.8 compatible and has no
dependency on the project being profiled.
"""

import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


DETECT_DIR = Path(__file__).resolve().parent.parent / "scan"
if str(DETECT_DIR) not in sys.path:
    sys.path.insert(0, str(DETECT_DIR))

from semantic_contract import (audit_semantic_task_bijection,
                               semantic_analysis,
                               selftest_semantic_task_bijection)


ALL_GATE_IDS = ["G{:02d}".format(i) for i in range(25)]

# SPEC-4: under a software-only information bundle these four hardware gates
# cannot produce evidence, so they are declared not-applicable (status "N/A")
# rather than being run, passed, failed, or blocked.
SW_NOT_APPLICABLE = ("G11", "G19", "G20", "G21")


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def absolute_path(value, base):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_gate_lists(contract):
    """Accept the current execution location and the older validation location."""
    execution = contract.get("execution")
    if not isinstance(execution, dict):
        execution = {}
    validation = contract.get("validation")
    if not isinstance(validation, dict):
        validation = {}

    required = execution.get("required_gate_ids")
    if required is None:
        required = validation.get("required_gate_ids")
    if required is None:
        required = contract.get("required_gate_ids")
    not_applicable = execution.get("not_applicable_gate_ids")
    if not_applicable is None:
        not_applicable = validation.get("not_applicable_gate_ids")
    if not_applicable is None:
        not_applicable = contract.get("not_applicable_gate_ids")
    return required, not_applicable


def information_bundle(contract):
    """SPEC-4: contract.information_bundle in {sw, hw, sw+hw}, normalized.

    Real CONTRACT.lock.json files carry the bundle under ``capture`` (the same
    location validate_preflight.py reads), so that container is searched in
    addition to the legacy top-level/execution/target locations."""
    if not isinstance(contract, dict):
        return None
    for container in (contract, contract.get("capture"),
                      contract.get("execution"), contract.get("target")):
        if isinstance(container, dict):
            value = container.get("information_bundle")
            if isinstance(value, str):
                return value.strip().lower()
    return None


def validate_partition(contract, bundle=None):
    required, not_applicable = get_gate_lists(contract)
    errors = []
    if not isinstance(required, list):
        errors.append("required_gate_ids must be a list")
        required = []
    if not isinstance(not_applicable, list):
        errors.append("not_applicable_gate_ids must be a list")
        not_applicable = []

    def check_ids(name, ids):
        string_ids = [item for item in ids if isinstance(item, str)]
        non_strings = [item for item in ids if not isinstance(item, str)]
        duplicates = sorted({item for item in string_ids if string_ids.count(item) > 1})
        unknown = sorted({item for item in string_ids if item not in ALL_GATE_IDS})
        if non_strings:
            errors.append("{} must contain strings only: {}".format(
                name, ", ".join(json.dumps(item, sort_keys=True, ensure_ascii=True) for item in non_strings)))
        if duplicates:
            errors.append("{} contains duplicates: {}".format(name, ", ".join(duplicates)))
        if unknown:
            errors.append("{} contains unknown IDs: {}".format(name, ", ".join(map(str, unknown))))

    check_ids("required_gate_ids", required)
    check_ids("not_applicable_gate_ids", not_applicable)
    required_strings = {item for item in required if isinstance(item, str)}
    not_applicable_strings = {item for item in not_applicable if isinstance(item, str)}
    overlap = sorted(required_strings.intersection(not_applicable_strings))
    if overlap:
        errors.append("required/not_applicable overlap: {}".format(", ".join(overlap)))
    covered = required_strings.union(not_applicable_strings)
    missing = sorted(set(ALL_GATE_IDS).difference(covered))
    if missing:
        errors.append("partition missing IDs: {}".format(", ".join(missing)))
    # SPEC-4: only a software-only bundle may mark exactly {G11,G19,G20,G21} as
    # not-applicable; hw / sw+hw / an unset bundle keep the strict all-required
    # policy so N/A can never be reached for a hardware delivery.
    sw_na = set(SW_NOT_APPLICABLE)
    if bundle == "sw":
        expected_required = set(ALL_GATE_IDS) - sw_na
        if (required_strings != expected_required
                or len(required) != len(expected_required)
                or not_applicable_strings != sw_na
                or len(not_applicable) != len(sw_na)):
            errors.append("policy(sw bundle) requires required==G00-G24 minus "
                          "{G11,G19,G20,G21} and not_applicable=={G11,G19,G20,G21}")
    elif required_strings != set(ALL_GATE_IDS) or len(required) != len(ALL_GATE_IDS) or not_applicable_strings:
        errors.append("policy requires all G00-G24 required and no not_applicable IDs")
    return required, not_applicable, errors


def expected_trace_hash(contract):
    """Return an explicitly declared expected trace hash, if the contract has one."""
    candidates = []
    for container in (contract, contract.get("execution"), contract.get("artifacts")):
        if isinstance(container, dict):
            for key in ("trace_sha256", "candidate_trace_sha256", "candidate_perfetto_sha256"):
                value = container.get(key)
                if isinstance(value, str):
                    candidates.append(value)
            candidate = container.get("candidate_perfetto")
            if isinstance(candidate, dict) and isinstance(candidate.get("sha256"), str):
                candidates.append(candidate["sha256"])
    return candidates[0] if candidates else None


def substitute_template(value, token, item_id):
    if isinstance(value, str):
        return value.replace(token, item_id)
    if isinstance(value, list):
        return [substitute_template(item, token, item_id) for item in value]
    if isinstance(value, dict):
        return {key: (item if key in ("sha256", "expected_sha256") else substitute_template(item, token, item_id))
                for key, item in value.items()}
    return value


def command_spec(commands, gate_id, template=None):
    if isinstance(commands, dict) and gate_id in commands:
        spec = commands[gate_id]
    elif template is not None:
        template_argv = template if isinstance(template, list) else (
            template.get("argv") if isinstance(template, dict) else None)
        if not isinstance(template_argv, list) or not any(
            isinstance(item, str) and "{gate_id}" in item for item in template_argv
        ):
            return None, "gate_command_template argv must contain {gate_id}"
        spec = substitute_template(template, "{gate_id}", gate_id)
    else:
        return None, "missing command"
    if isinstance(spec, list):
        argv = spec
        options = {}
    elif isinstance(spec, dict):
        options = spec
        argv = spec.get("argv")
        # "command" is accepted as the per-gate field name, but remains argv-only.
        if argv is None:
            argv = spec.get("command")
        if "shell" in spec:
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


def evidence_items(spec, run_dir):
    values = spec.get("options", {}).get("evidence", [])
    if isinstance(values, (str, dict)):
        values = [values]
    if not isinstance(values, list):
        return [], "evidence must be a path or list"
    result = []
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
        result.append(record)
    return result, None


def base_gate_record(gate_id, run_dir):
    log_dir = run_dir / "validation_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return {
        "gate_id": gate_id,
        "status": "BLOCKED",
        "started_at_utc": None,
        "ended_at_utc": None,
        "exit_code": None,
        "stdout_log": str((log_dir / (gate_id + ".stdout.log")).resolve()),
        "stderr_log": str((log_dir / (gate_id + ".stderr.log")).resolve()),
        "evidence": [],
        "reason": None,
    }


def write_log(path, content):
    path_obj = Path(path)
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    path_obj.write_text(content or "", encoding="utf-8", errors="replace")


def na_gate_record(gate_id, run_dir, bundle):
    """SPEC-4: a not-applicable gate is recorded as N/A and never executed."""
    record = base_gate_record(gate_id, run_dir)
    record["status"] = "N/A"
    record["command"] = None
    record["reason"] = ("information_bundle={} 为软件交付，该硬件门禁不适用 (N/A)"
                        .format(bundle))
    write_log(record["stdout_log"], "")
    write_log(record["stderr_log"], (record["reason"] or "") + "\n")
    return record


def run_gate(gate_id, spec, run_dir):
    record = base_gate_record(gate_id, run_dir)
    record["command"] = list(spec["argv"]) if spec else None
    if spec is None:
        record["reason"] = "missing or invalid command"
        write_log(record["stdout_log"], "")
        write_log(record["stderr_log"], record["reason"] + "\n")
        return record

    options = spec["options"]
    cwd = options.get("cwd", str(run_dir))
    cwd_path = absolute_path(cwd, run_dir)
    record["cwd"] = str(cwd_path)
    env_value = options.get("env", {})
    if env_value is None:
        env_value = {}
    if not isinstance(env_value, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in env_value.items()
    ):
        record["reason"] = "env must be a string-to-string mapping"
        write_log(record["stdout_log"], "")
        write_log(record["stderr_log"], record["reason"] + "\n")
        return record
    timeout_sec = options.get("timeout_sec")
    if timeout_sec is not None and (
        isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float)) or timeout_sec <= 0
    ):
        record["reason"] = "timeout_sec must be a positive number"
        write_log(record["stdout_log"], "")
        write_log(record["stderr_log"], record["reason"] + "\n")
        return record
    if timeout_sec is not None:
        record["timeout_sec"] = timeout_sec
    evidence, evidence_error = evidence_items(spec, run_dir)
    if evidence_error:
        record["reason"] = evidence_error
        write_log(record["stdout_log"], "")
        write_log(record["stderr_log"], evidence_error + "\n")
        return record
    record["evidence"] = evidence
    record["started_at_utc"] = utc_now()
    try:
        child_env = os.environ.copy()
        child_env.update(env_value)
        completed = subprocess.run(
            spec["argv"], cwd=str(cwd_path), env=child_env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, shell=False, check=False,
            timeout=timeout_sec,
        )
        record["exit_code"] = completed.returncode
        write_log(record["stdout_log"], completed.stdout)
        write_log(record["stderr_log"], completed.stderr)
    except subprocess.TimeoutExpired as error:
        # subprocess.run terminated the timed-out child.
        # Preserve any partial stdout and stderr produced before termination.
        record["status"] = "FAIL"
        record["timed_out"] = True
        record["reason"] = "command timed out after {} seconds".format(timeout_sec)
        write_log(record["stdout_log"], error.stdout)
        write_log(record["stderr_log"], error.stderr)
        record["ended_at_utc"] = utc_now()
        return record
    except (OSError, ValueError) as error:
        record["reason"] = "command could not be executed: {}".format(error)
        write_log(record["stdout_log"], "")
        write_log(record["stderr_log"], record["reason"] + "\n")
        record["ended_at_utc"] = utc_now()
        return record
    record["ended_at_utc"] = utc_now()

    # Re-stat and hash after the command: gates commonly create their evidence.
    evidence, evidence_error = evidence_items(spec, run_dir)
    record["evidence"] = evidence
    missing = [item["path"] for item in evidence if not item["exists"] or "read_error" in item]
    hash_mismatch = [item["path"] for item in evidence if item.get("sha256_match") is False]
    if record["exit_code"] != 0:
        record["status"] = "FAIL"
        record["reason"] = "command exited with non-zero status"
    elif evidence_error or missing:
        record["status"] = "BLOCKED"
        record["reason"] = evidence_error or "required evidence is missing"
    elif hash_mismatch:
        record["status"] = "FAIL"
        record["reason"] = "evidence SHA256 mismatch"
    else:
        record["status"] = "PASS"
    return record


def atomic_dump(payload, out):
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(out) + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(partial), str(out))


# ---------------------------------------------------------------------------
# Skill-authoritative mandatory audits.
#
# The per-target gate adapter is a thin shim (checks a file exists, a boolean
# flag, or a frequency).  These audits run inside the fixed skill entry so they
# apply to every target and cannot be skipped or softened by a contract:
#   1. overhead ceiling  — two-tier hard ceiling: task granularity stays at
#      10%, phase granularity may go up to 15%; the 15% allowance applies only
#      when the delivered trace actually carries phase-layer intervals, so
#      declaring ``phase`` while shipping a task-only timeline is still held
#      to 10%.  The ceiling itself can never be raised further.
#   2. full-grid SM base  — the SM baseline is the full grid; any subset must be
#      a halving of it (78 -> 39 -> ... -> 1), never an arbitrary count.
#   3. HW DRAM coverage    — an HW bundle must retain DRAM read+write bandwidth.
#   4. lane purity         — only genuine task/phase slices may be drawn; no
#      phase_span / megaphase / envelope containers.
#   5. operator-task coverage — the delivered timeline must cover the whole
#      kernel window with task-layer operator spans.  A min/max envelope (or a
#      full-width kernel bar) can hide the fact that only one operator was
#      instrumented, so coverage is measured as a union with explicit gaps.
#   6. HW normalization plateau — a percentage counter pinned at exactly
#      100/2^k while a sibling counter reaches 100 means a die-normalization
#      correction is missing; require an applied scale rule.
#   7. split-resource totals — when one physical resource is delivered as
#      component curves (DRAM read+write, NVLink request+response) the summed
#      total curve must also be present; the total is the roofline number.
#   8. semantic naming — task display names must be human-readable operator
#      names, never internal codenames (passA/passB/token/chunk/EV_*).
#   9. trace filename — the delivered filename must satisfy the SPEC-1 schema
#      (model/stage/shape/dtype/grid/warp/ph/bundle all present) and its bundle
#      and grid fields must agree with the locked contract.
# ---------------------------------------------------------------------------
OVERHEAD_HARD_MAX = 0.10
# Phase granularity records every task's internal stages, so it is allowed
# a higher hard ceiling.  Anything above this is never acceptable.
OVERHEAD_HARD_MAX_PHASE = 0.15
_BANNED_LANE_ROLES = ("megaphase", "envelope", "span_container", "global_envelope")
_BANNED_LANE_SHORT = ("phase", "megaphase", "phase_span")

# Audit 5: the largest continuous un-instrumented gap allowed inside the kernel
# window.  Barrier-scale gaps (a few us) are normal; a whole missing operator
# stage is not.
COVERAGE_GAP_FRACTION = 5           # percent of the kernel span
COVERAGE_GAP_FLOOR_NS = 50_000      # 50 us absolute floor
# Measured widest real operator segment across four delivered targets: 41.3% /
# 44.4% / 48.3% / 28.9% of the entry->release window.  A mislabelled whole-kernel
# anchor measures 100.00%, and the banned megaphase envelope 51.9%.
COVERAGE_SINGLE_SPAN_MAX_PERCENT = 90

# Audit 6: GPU PM percentage counters are normalized against a peak that on some
# SKUs belongs to the full die rather than the shipped part, so the curve tops
# out at exactly 100/2^k instead of 100.  H20 measured evidence: raw
# "Tensor Active" plateaus at 25 (= 100/4) through dense WGMMA stretches while
# "SMs Active" on the very same capture reaches a true 100.
_NORMALIZATION_DIVISORS = (2, 4, 8, 16)
_PLATEAU_TOLERANCE = 0.02           # relative tolerance when matching 100/2^k
# A die-normalization ceiling is a *plateau*, not just a maximum: the raw H20
# "Tensor Active" peak of 25 is held by 5%-64% of all samples.  An
# integer-quantized counter reaches its own maximum a handful of times without
# any ceiling being involved (measured false positives: 0.02%-0.23% of samples),
# so the peak must be held by a meaningful share of the capture, not merely
# repeated.
_PLATEAU_MIN_HITS = 2               # never fire on a single-sample spike
_PLATEAU_MIN_HIT_FRACTION = 0.02    # and >=2% of samples must sit on it exactly

# Audit 7: (family label, component matchers, total matcher, suggested name).
# Each component matcher is a tuple of substrings that must all appear in a
# counter name.
_SPLIT_RESOURCE_FAMILIES = (
    ("DRAM 读写带宽", (("dram", "read"), ("dram", "write")), ("dram", "total"),
     "DRAM Bandwidth Total"),
    ("NVLink RX 用户数据", (("nvlink", "rx", "request"), ("nvlink", "rx", "response")),
     ("nvlink", "rx", "total"), "NVLink RX User Data Total"),
    ("NVLink TX 用户数据", (("nvlink", "tx", "request"), ("nvlink", "tx", "response")),
     ("nvlink", "tx", "total"), "NVLink TX User Data Total"),
)

# Audit 8: internal codenames that carry no operator meaning.  These are the
# names that actually shipped and had to be renamed by hand.
_CODENAME_PATTERNS = (
    (r"pass\s*[ab]\b", "passA/passB 是内部代号，不是算子名"),
    (r"^(token|chunk|tile|block|iter|step|stage|phase|slice|part)\s*\d*$",
     "裸结构词没有算子语义"),
    (r"^ev_[a-z0-9_]+$", "EV_* 是原始事件宏名"),
    (r"^p\d+[a-z]?$", "P<n> 是内部段号"),
    (r"^[a-z]+_[a-z]$", "形如 foo_a 的缩写没有语义"),
)
_CODENAME_MIN_CHARS = 3


def _first_present(container, paths):
    """Return the first value reachable by any dotted key path, else None."""
    for path in paths:
        node = container
        ok = True
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                ok = False
                break
        if ok:
            return node
    return None


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _halving_ladder(total):
    """Full-grid halving sequence: total, ceil(total/2), ... , 1."""
    ladder = []
    value = int(total)
    while value >= 1:
        ladder.append(value)
        if value == 1:
            break
        value = (value + 1) // 2
    return ladder


def _load_json_quiet(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _load_trace_quiet(path):
    try:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _is_production_contract(contract):
    """A real measurement contract, as opposed to a minimal selftest contract."""
    if not isinstance(contract, dict):
        return False
    return any(_first_present(contract, [(key,)]) is not None
               for key in ("capture", "validation", "topology", "build_and_launch"))


def _trace_has_phase_layer(trace):
    """True when the delivered trace actually draws phase-layer intervals.

    The 15% phase ceiling pays for real phase detail; a contract that declares
    ``phase`` but ships a task-only timeline gets the task ceiling instead."""
    if not isinstance(trace, dict):
        return False
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        return False
    for event in events:
        if not isinstance(event, dict) or event.get("ph") not in ("X", "B"):
            continue
        args = event.get("args")
        if isinstance(args, dict) and args.get("layer") == "phase":
            return True
    return False


def _overhead_ceiling(contract, trace):
    """(ceiling, granularity): phase granularity earns 15%, everything else 10%.

    A missing/unreadable granularity falls back to the stricter task ceiling --
    the contract is required to state ``capture.granularity`` explicitly, and
    the audit never widens a limit on ambiguous input."""
    granularity = _first_present(contract or {}, [("capture", "granularity")])
    granularity = str(granularity).strip().lower() if granularity is not None else ""
    if granularity == "phase" and _trace_has_phase_layer(trace):
        return OVERHEAD_HARD_MAX_PHASE, "phase"
    return OVERHEAD_HARD_MAX, granularity or "task"


def _audit_overhead(contract, run_dir, trace):
    """Defect #1: two-tier hard ceiling.  task stays < 10%; phase may go up to
    < 15% because it records every task's internal stages -- but only when the
    delivered trace actually contains phase-layer intervals (a ``phase``
    contract shipping a task-only timeline is still held to 10%).  The ceiling
    itself is never negotiable; the only remedies are dropping to task
    granularity or halving the SM set."""
    failures = []
    ceiling, granularity = _overhead_ceiling(contract, trace)
    ceiling_pct = "{:.0f}%".format(ceiling * 100.0)
    phase_without_phases = (granularity == "phase" and ceiling == OVERHEAD_HARD_MAX)
    maximum = _as_number(_first_present(contract, [
        ("validation", "overhead", "maximum"),
        ("overhead", "maximum"),
        ("acceptance", "overhead", "maximum"),
    ]))
    if maximum is not None and maximum > ceiling + 1e-9:
        failures.append(
            "overhead.maximum={:.4f} 超过 {} 档硬上限 {:.2f}；禁止抬高阈值，"
            "只能取消 phase / 折半 SM 降开销".format(maximum, granularity, ceiling))
    overhead_doc = _load_json_quiet(Path(run_dir) / "validation" / "overhead.json")
    if overhead_doc is None:
        if _first_present(contract, [("validation", "overhead"), ("overhead",)]) is not None:
            failures.append("缺少实测 overhead.json，无法验证真实插桩开销")
        return failures
    ratio = _as_number(overhead_doc.get("overhead_ratio"))
    if ratio is None:
        ratio = _as_number(overhead_doc.get("overhead_ratio_full_vs_entry_only"))
    if ratio is None:
        medians = overhead_doc.get("medians_ns") or overhead_doc.get("configurations") or {}
        entry = _as_number(medians.get("PROF_ENTRY_ONLY"))
        full = _as_number(medians.get("PROF_FULL"))
        if entry and full and entry > 0:
            ratio = (full - entry) / entry
    if ratio is None:
        failures.append("overhead.json 缺少可判定的 full-vs-entry 开销比值")
    elif ratio >= ceiling:
        message = ("实测插桩开销 {:.2f}% >= {}（{} 档上限）；先取消 phase 只测 task，"
                   "仍超再从满 grid 折半 SM，严禁抬阈值".format(
                       ratio * 100.0, ceiling_pct, granularity))
        if phase_without_phases:
            message += ("；注意：契约声明了 phase 但交付 trace 没有任何 phase 层区间，"
                        "15% 的 phase 余量不适用，按 task 档 10% 判定")
        failures.append(message)
    return failures


def _audit_sm_scope(contract):
    """Defect #2: SM baseline is the full grid; any subset must be a halving of it."""
    failures = []
    scope = _first_present(contract, [("capture", "software_sm_scope"), ("software_sm_scope",)])
    if not isinstance(scope, dict):
        return failures
    if str(scope.get("mode", "")).lower() != "selected":
        return failures
    selected = scope.get("selected_physical_smids")
    if not isinstance(selected, list) or not selected:
        failures.append("software_sm_scope.mode=selected 但缺少 selected_physical_smids")
        return failures
    total = _as_number(_first_present(contract, [
        ("topology", "physical_sm_count"),
        ("environment", "gpu", "physical_sm_count"),
        ("build_and_launch", "num_ctas"),
    ]))
    if total is None:
        grid = _first_present(contract, [("build_and_launch", "grid")])
        if isinstance(grid, list) and grid:
            total = _as_number(grid[0])
    if total is None:
        failures.append("无法确定满 grid SM 数，无法校验 SM 折半基线")
        return failures
    ladder = _halving_ladder(total)
    count = len(selected)
    if count not in ladder:
        failures.append(
            "选取 {} 个 SM 不在从满 grid {} 起的折半阶梯 {}；SM 缩减只能从满 grid 折半(如 78->39->...->1)".format(
                count, int(total), ladder))
    return failures


def _audit_hw_dram(contract, run_dir, bundle):
    """Defect #3: an HW bundle must retain DRAM read+write bandwidth counters."""
    failures = []
    if "hw" not in (bundle or ""):
        return failures
    metrics = _load_json_quiet(Path(run_dir) / "hw" / "hw_metrics.json")
    if metrics is None:
        if _first_present(contract, [("capture",), ("hardware",), ("hw",)]) is not None:
            failures.append("HW bundle 缺少 hw_metrics.json，无法验证 DRAM 带宽指标")
        return failures
    counters = metrics.get("counters") or []

    def _has(kind):
        for counter in counters:
            name = str(counter.get("name", "")).lower()
            samples = _as_number(counter.get("sample_count"))
            if "dram" in name and kind in name and samples and samples > 0:
                return True
        return False

    if not _has("read"):
        failures.append("HW 指标缺少有样本的 DRAM 读带宽(DRAM Read Bandwidth)")
    if not _has("write"):
        failures.append("HW 指标缺少有样本的 DRAM 写带宽(DRAM Write Bandwidth)")
    return failures


def _audit_lane_purity(trace):
    """Defect #4: only genuine task/phase slices may be drawn on a lane."""
    if trace is None:
        return ["无法读取候选 trace，无法验证泳道纯净性"]
    events = trace.get("traceEvents", trace if isinstance(trace, list) else [])
    banned = []
    for event in events:
        if event.get("ph") not in ("X", "B"):
            continue
        args = event.get("args") or {}
        name = str(event.get("name", "")).strip().lower()
        short = str(args.get("short_name", "")).strip().lower()
        role = str(args.get("role", "")).strip().lower()
        layer = args.get("layer")
        reason = None
        if "phase_span" in name or "megaphase" in name:
            reason = "name={}".format(name)
        elif args.get("task_envelope") is True:
            reason = "task_envelope=true"
        elif role in _BANNED_LANE_ROLES:
            reason = "role={}".format(role)
        elif layer == "task" and short in _BANNED_LANE_SHORT:
            reason = "task-layer 容器 short_name={}".format(short)
        elif isinstance(layer, str) and layer not in ("task", "phase", "ignored"):
            reason = "layer={}".format(layer)
        if reason:
            banned.append({"name": event.get("name"), "why": reason})
    if banned:
        return ["泳道出现 {} 个非 task/phase 真实切片(如 phase_span/megaphase envelope): {}；泳道只允许 task 和 phase".format(
            len(banned), banned[:8])]
    return []


def _union_and_gaps(intervals, win_start, win_end):
    """Clip intervals to [win_start, win_end], merge overlaps, and return
    (union, covered_ns, gaps).  Gaps are the maximal sub-intervals of the window
    covered by no interval."""
    clipped = []
    for start, end in intervals:
        s = max(int(start), win_start)
        e = min(int(end), win_end)
        if e > s:
            clipped.append((s, e))
    clipped.sort()
    union = []
    for s, e in clipped:
        if union and s <= union[-1][1]:
            if e > union[-1][1]:
                union[-1] = (union[-1][0], e)
        else:
            union.append((s, e))
    covered = sum(e - s for s, e in union)
    gaps = []
    cursor = win_start
    for s, e in union:
        if s > cursor:
            gaps.append((cursor, s))
        if e > cursor:
            cursor = e
    if cursor < win_end:
        gaps.append((cursor, win_end))
    return union, covered, gaps


def _kernel_window_ns(run_dir, trace):
    """Absolute-ns kernel window plus the alignment scale, or (None, None, why).

    The window must come from the device entry/release anchor, never from the
    extent of the drawn spans: an envelope derived from the spans themselves
    cannot reveal that whole operator stages were never instrumented.
    """
    align = _load_json_quiet(Path(run_dir) / "work" / "alignment.json")
    if isinstance(align, dict):
        window = align.get("sw_window_ns")
        if isinstance(window, (list, tuple)) and len(window) == 2:
            scale = _as_number(align.get("scale"))
            return (int(window[0]), int(window[1])), scale, None
    other = (trace or {}).get("otherData") or {}
    window = _first_present(other, [("hw_merge", "sw_window_ns"), ("sw_window_ns",)])
    if isinstance(window, (list, tuple)) and len(window) == 2:
        return (int(window[0]), int(window[1])), 1.0, None
    # Software-only bundles have no SW/HW alignment.  Fall back to the full
    # extent of the decoded events, which still includes the ignored-layer
    # whole-kernel anchor and therefore is the real kernel window.
    lo, hi = None, None
    for path in sorted(glob.glob(str(Path(run_dir) / "sw" / "events_rank*.json"))):
        doc = _load_json_quiet(path)
        for event in (doc or {}).get("events", []):
            start, end = event.get("start_ns"), event.get("end_ns")
            if start is None or end is None:
                continue
            lo = int(start) if lo is None else min(lo, int(start))
            hi = int(end) if hi is None else max(hi, int(end))
    if lo is not None and hi is not None and hi > lo:
        return (lo, hi), 1.0, None
    return None, None, "既没有 work/alignment.json 也没有 sw/events_rank*.json，无法确定整核窗口"


def _audit_operator_task_coverage(run_dir, trace, bundle):
    """Defect #5: task-layer operator spans must cover the whole kernel window.

    Precondition: shift-only alignment (scale == 1).  Under scale == 1 the SW
    timeline is never stretched, so a stretch of the kernel window with no
    operator task is genuinely un-instrumented compute rather than a scaling
    artifact.  This catches the historical hole where only attention was a task
    and every head/tail operator (quant / GEMM / indexer / logits / top-k /
    sparse-attention indices / output projection) was missing -- a min/max
    envelope inflated by a full-width kernel bar wrongly reported 100%.
    """
    if trace is None:
        return ["无法读取候选 trace，无法验证算子 task 覆盖率"]
    window, scale, why = _kernel_window_ns(run_dir, trace)
    if window is None:
        return [why]
    if scale is not None and abs(scale - 1.0) > 1e-9:
        # scale != 1 means the timeline was stretched, so gap semantics are not
        # trustworthy.  A non-unit scale already fails G19 on its own.
        return []
    win_start, win_end = window
    span = win_end - win_start
    if span <= 0:
        return ["整核窗口长度 {} ns 非正，无法验证覆盖率".format(span)]
    events = trace.get("traceEvents", trace if isinstance(trace, list) else [])
    intervals = []
    task_spans = 0
    missing_raw = 0
    widest = (0, None)
    for event in events:
        if event.get("ph") != "X":
            continue
        args = event.get("args") or {}
        if args.get("layer") != "task":
            continue
        task_spans += 1
        start, end = args.get("start_ns_raw"), args.get("end_ns_raw")
        if start is None or end is None:
            missing_raw += 1
            continue
        start, end = int(start), int(end)
        intervals.append((start, end))
        if end - start > widest[0]:
            widest = (end - start, event.get("name"))
    if task_spans == 0:
        return ["候选 trace 里没有任何 layer=task 的区间；算子必须以 task 层交付"]
    if not intervals:
        return ["task 区间缺少 args.start_ns_raw/end_ns_raw 绝对时间戳({}/{} 条)；"
                "rebase 后的相对 ts 会把头部空洞抹平，emitter 必须保留原始 ns".format(
                    missing_raw, task_spans)]
    # A single span this wide cannot be one grid_sync-delimited segment: the
    # entry->release window is strictly wider than any segment inside it.  Left
    # unchecked, such a span fills the union by itself and the coverage number
    # becomes meaningless -- which is exactly how the historical hole reported
    # 100% while only attention was instrumented.
    if widest[0] * 100 >= span * COVERAGE_SINGLE_SPAN_MAX_PERCENT:
        return ["单条 layer=task 区间 {!r} 占整核窗口的 {:.2f}%(>= {}%)，它不可能是一个 grid_sync 计算段。"
                "要么整核锚点被误标成 layer=task(必须改成 layer=ignored，只用于对齐、不渲染)，"
                "要么整核确实只有一段算子(那就必须按内部阶段拆成更细的 task)。"
                "在修好之前覆盖率数字无意义，不予采信".format(
                    widest[1], widest[0] * 100.0 / span, COVERAGE_SINGLE_SPAN_MAX_PERCENT)]
    _union, covered, gaps = _union_and_gaps(intervals, win_start, win_end)
    max_gap = max((e - s for s, e in gaps), default=0)
    threshold = max(span * COVERAGE_GAP_FRACTION // 100, COVERAGE_GAP_FLOOR_NS)
    if max_gap <= threshold:
        return []
    ranked = sorted(([round((s - win_start) / 1000.0, 2),
                      round((e - win_start) / 1000.0, 2),
                      round((e - s) / 1000.0, 2)] for s, e in gaps),
                    key=lambda item: item[2], reverse=True)
    return ["算子 task 只覆盖整核窗口 {:.2f}us 的 {:.1f}%，最大连续未插桩空洞 {:.2f}us > 阈值 {:.2f}us；"
            "空洞[起,止,时长]us(取前 8 段): {}；每个 grid_sync 计算段都必须有自己的 task".format(
                span / 1000.0, covered * 100.0 / span, max_gap / 1000.0,
                threshold / 1000.0, ranked[:8])]


def _counter_values(counter):
    """Numeric sample values of an hw_metrics counter, whatever the encoding."""
    values = []
    for sample in counter.get("samples") or []:
        if isinstance(sample, (list, tuple)):
            value = _as_number(sample[1]) if len(sample) > 1 else None
        elif isinstance(sample, dict):
            value = _as_number(sample.get("value"))
        else:
            value = _as_number(sample)
        if value is not None:
            values.append(value)
    return values


def _audit_hw_scale_plausibility(contract, run_dir, bundle):
    """Defect #6: a percentage counter pinned at exactly 100/2^k is a missing
    die-normalization correction, not a real ceiling.

    H20 measured evidence: raw "Tensor Active" tops out at 25 (= 100/4) with
    many samples pinned there, while "SMs Active" from the same capture reaches
    a true 100.  Delivering the raw curve understates tensor utilisation 4x.
    Either an applied scale rule must publish a corrected lane, or the contract
    must record an explicit "reviewed, no correction needed" decision.
    """
    if "hw" not in (bundle or ""):
        return []
    metrics = _load_json_quiet(Path(run_dir) / "hw" / "hw_metrics.json")
    if not isinstance(metrics, dict):
        return []
    counters = metrics.get("counters") or []
    rules = metrics.get("scale_rules") or []
    corrected = set()
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        source = str(rule.get("source_name", "")).strip().lower()
        if not source:
            continue
        if rule.get("applied") is True:
            corrected.add(source)
        elif str(rule.get("reason", "")).strip():
            # Explicit reviewed-and-rejected decision is also a decision.
            corrected.add(source)
    for counter in counters:
        origin = str(counter.get("derived_from", "")).strip().lower()
        if origin:
            corrected.add(origin)
    reviewed = _first_present(contract, [
        ("capture", "hardware", "normalization_plateau_reviewed"),
        ("hardware", "normalization_plateau_reviewed"),
    ])
    if isinstance(reviewed, dict):
        for name, note in reviewed.items():
            if str(note).strip():
                corrected.add(str(name).strip().lower())
    failures = []
    for counter in counters:
        if counter.get("is_derived") is True:
            continue
        unit = str(counter.get("unit", ""))
        if "%" not in unit and "percent" not in unit.lower():
            continue
        name = str(counter.get("name", "")).strip()
        values = _counter_values(counter)
        if not values:
            continue
        peak = max(values)
        if peak <= 0:
            continue
        for divisor in _NORMALIZATION_DIVISORS:
            expected = 100.0 / divisor
            if abs(peak - expected) > expected * _PLATEAU_TOLERANCE:
                continue
            hits = sum(1 for value in values if abs(value - peak) < 1e-9)
            if hits < _PLATEAU_MIN_HITS:
                continue
            if hits < len(values) * _PLATEAU_MIN_HIT_FRACTION:
                continue
            if name.lower() in corrected:
                break
            failures.append(
                "HW 指标 {!r} 的峰值 {:g} 恰好钉在 100/{}（{}/{} 个样本命中，"
                "{:.1f}%），这是按整颗 die 峰值归一化、缺少 x{} 修正的典型症状；"
                "要么在 metrics whitelist 的 scale 里发布一条修正后的派生曲线，"
                "要么在契约 capture.hardware.normalization_plateau_reviewed 里写明"
                "为何确认无需修正".format(name, peak, divisor, hits, len(values),
                                       100.0 * hits / len(values), divisor))
            break
    return failures


# ----------------------------------------------------------------------
# tile granularity (mandatory): declaration + size cap + set reconciliation
# ----------------------------------------------------------------------
TILE_MAX_AXIS_DEFAULT = 512


def _tile_declared_events(abi):
    """Collect tile-declared events from the ABI.  Two conventions:

    * new schema: entry["tile_domain"] with a "passes" list (machine-checkable
      per-pass domains, set-reconciled by this audit);
    * legacy schema (llama): entry["task_contract"]["tile_domain"] -- those are
      task-layer events whose exact per-instance reconciliation is already
      enforced by the semantic_task_bijection audit; here they only need to
      exist.
    """
    new_schema = []
    legacy = []
    for entry in abi.get("tasks", []) if isinstance(abi, dict) else []:
        if not isinstance(entry, dict):
            continue
        event_id = entry.get("id", entry.get("event_id"))
        if isinstance(event_id, bool) or not isinstance(event_id, int):
            continue
        name = str(entry.get("name", ""))
        domain = entry.get("tile_domain")
        if isinstance(domain, dict) and isinstance(domain.get("passes"), list):
            new_schema.append((event_id, name, domain))
            continue
        contract = entry.get("task_contract")
        if isinstance(contract, dict) and isinstance(contract.get("tile_domain"), dict):
            legacy.append((event_id, name))
    return new_schema, legacy


def _tile_work_packing(abi):
    fields = ((abi.get("work_id_packing") or {}).get("fields") or {})
    sl = fields.get("slice", {}); ch = fields.get("chunk", {})
    return (int(sl.get("lsb", 8)), int(ch.get("lsb", 0)))


def _tile_expected_set(passes, captured_ctas, grid, slice_lsb, chunk_lsb):
    """Expected (logical_cta, work_id) pairs for one tile event, restricted to
    CTAs that actually appear in the captured stream."""
    expected = set()
    for spec in passes:
        if not isinstance(spec, dict):
            continue
        ctas = spec.get("ctas", [0, grid])
        c0, c1 = int(ctas[0]), int(ctas[1])
        slice_v = int(spec.get("slice", 0))
        kind = str(spec.get("kind", "grid_stride"))
        base = slice_v << slice_lsb
        if kind == "grid_stride":
            num_tiles = int(spec.get("num_tiles", 0))
            for c in range(c0, c1):
                if c not in captured_ctas:
                    continue
                tile = c
                while tile < num_tiles:
                    expected.add((c, base | tile))
                    tile += grid
        elif kind == "cta_map":
            chunk = int(spec.get("chunk", 0))
            for c in range(c0, c1):
                if c in captured_ctas:
                    expected.add((c, base | chunk))
        elif kind == "per_cta_range":
            n = int(spec.get("range", 0))
            for c in range(c0, c1):
                if c not in captured_ctas:
                    continue
                for j in range(n):
                    expected.add((c, base | j))
    return expected


def _tile_free_claim(abi):
    """Explicit tile-free proof for kernels with no tile structure at all.

    The claim lives in ABI ``semantic_analysis.tile_free`` so it is locked by
    ``conclusion_sha256`` together with every other semantic conclusion: the
    model must write a source-level justification (which loop nest / launch
    mapping proves there is no tile axis), and any later source or conclusion
    drift invalidates the hash.  Silence is never a proof."""
    analysis = semantic_analysis(abi)
    if not isinstance(analysis, dict):
        return None
    claim = analysis.get("tile_free")
    if not isinstance(claim, dict):
        return None
    justification = claim.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        return None
    return justification.strip()


def _audit_tile_granularity(contract, run_dir, bundle):
    """Tile granularity is a MANDATORY delivery level, orthogonal to the
    task/phase layer taxonomy.  Three teeth, all fail-closed:

    1. declaration: the ABI must declare at least one tile-level event stream
       (``tile_domain``), or the kernel must carry an explicit hash-locked
       tile-free proof (``semantic_analysis.tile_free.justification``);
    2. size cap: every declared tile axis must be <= ``tile_max_axis``
       (contract ``validation.tile_granularity.max_axis``, default 512) so a
       giant pseudo-tile cannot masquerade as fine granularity;
    3. set reconciliation: the recorded (logical_cta, work_id) instance set of
       every declared tile event must EXACTLY equal the set derived from the
       declared per-pass domain formulas (grid-stride / cta_map /
       per_cta_range), restricted to the captured CTAs -- one missing, extra
       or duplicated tile fails the run.

    Evidence comes from the decoded event stream
    (``sw/events_rank<display_rank>.json``), whose integrity is gated upstream,
    never from the rendered trace (the emitter may drop work_id from args).
    """
    if "sw" not in str(bundle or ""):
        return []
    failures = []
    abi_path = _first_present(contract or {}, [
        ("profile_selection", "abi", "absolute_path"),
        ("buffer_abi", "abi_file"),
    ])
    abi = _load_json_quiet(abi_path) if abi_path else None
    if not isinstance(abi, dict):
        return ["tile 粒度审计无法读取 ABI（{}）；无法确认 tile 声明，拒绝放行".format(abi_path)]
    new_schema, legacy = _tile_declared_events(abi)
    if not new_schema and not legacy:
        tile_free = _tile_free_claim(abi)
        if tile_free is not None:
            return []
        return ["ABI 没有声明任何 tile 级事件（缺 tile_domain）：tile 粒度是强制交付标准，"
                "与 task/phase 分层无关。请在 ABI 为每个 per-tile 事件流声明 tile_domain"
                "（含每 pass 的域公式与 tile_axes）；确无 tile 结构的 kernel 必须在 ABI "
                "semantic_analysis.tile_free.justification 写入源码级证明（随 "
                "conclusion_sha256 哈希锁定），沉默不等于证明"]
    max_axis = TILE_MAX_AXIS_DEFAULT
    tile_cfg = _first_present(contract or {}, [
        ("validation", "tile_granularity", "max_axis"),
    ])
    if isinstance(tile_cfg, (int, float)) and not isinstance(tile_cfg, bool) and tile_cfg > 0:
        max_axis = int(tile_cfg)
    grid = 78
    bl = (contract or {}).get("build_and_launch", {})
    if isinstance(bl, dict) and isinstance(bl.get("grid"), list) and bl["grid"]:
        try:
            grid = int(bl["grid"][0])
        except (TypeError, ValueError):
            pass
    rank = 0
    capture = (contract or {}).get("capture", {})
    if isinstance(capture, dict) and isinstance(capture.get("display_rank"), int):
        rank = capture["display_rank"]
    events_path = Path(run_dir) / "sw" / "events_rank{}.json".format(rank)
    doc = _load_json_quiet(events_path)
    events = doc.get("events") if isinstance(doc, dict) else None
    if not isinstance(events, list):
        return ["tile 粒度审计缺少解码事件流 {}；tile 证据必须来自 gated 解码器输出，"
                "不能用渲染后的 trace 代替".format(events_path)]
    captured_ctas = {e.get("logical_cta") for e in events
                     if isinstance(e, dict) and isinstance(e.get("logical_cta"), int)}
    slice_lsb, chunk_lsb = _tile_work_packing(abi)
    for event_id, name, domain in new_schema:
        passes = domain.get("passes") or []
        for spec in passes:
            axes = spec.get("tile_axes") if isinstance(spec, dict) else None
            if isinstance(axes, list):
                for axis in axes:
                    if isinstance(axis, int) and not isinstance(axis, bool) and axis > max_axis:
                        failures.append(
                            "tile 事件 {} {!r} 的 tile 轴 {} 超过上限 {}：tile 不允许特别大，"
                            "超上限的 tile 会把 per-tile 粒度退化成大段。请缩小 tile 或在契约 "
                            "validation.tile_granularity.max_axis 显式提高并说明理由".format(
                                event_id, name, axis, max_axis))
        expected = _tile_expected_set(passes, captured_ctas, grid, slice_lsb, chunk_lsb)
        actual = {(e.get("logical_cta"), e.get("work_id")) for e in events
                  if isinstance(e, dict) and e.get("event_id") == event_id
                  and isinstance(e.get("logical_cta"), int)
                  and e.get("logical_cta") in captured_ctas}
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            failures.append(
                "tile 事件 {} {!r} 的实例集合与声明域公式不对账：缺 {} 个、多 {} 个"
                "（缺失样例 {}，多出样例 {}）。每个声明 tile 必须恰好记录一次".format(
                    event_id, name, len(missing), len(extra),
                    sorted(missing)[:3], sorted(extra)[:3]))
    return failures


def _audit_hw_required_totals(run_dir, bundle):
    """Defect #7: when one physical resource ships as component curves, the
    summed total curve must ship too.

    DRAM read and write throughput are both "fraction of cycles the shared DRAM
    interface was active", so their per-sample sum is the number to compare
    against the roofline.  Delivering only the components forces the reader to
    eyeball two stacked curves and silently loses the peak.
    """
    if "hw" not in (bundle or ""):
        return []
    metrics = _load_json_quiet(Path(run_dir) / "hw" / "hw_metrics.json")
    if not isinstance(metrics, dict):
        return []
    names = [str(counter.get("name", "")).strip().lower()
             for counter in metrics.get("counters") or []]
    if not names:
        return []

    def _match(needles):
        return [name for name in names if all(needle in name for needle in needles)]

    failures = []
    for label, components, total, suggested in _SPLIT_RESOURCE_FAMILIES:
        present = [_match(component) for component in components]
        if not all(present):
            continue
        if _match(total):
            continue
        failures.append(
            "{} 的分量曲线都在（{}）但缺少求和总曲线；总量才是能对 roofline 的数，"
            "请在 metrics whitelist 的 combine 里加一条 sum_of 这些分量、"
            "drop_sources=false 的 {!r} 曲线".format(
                label, " + ".join(found[0] for found in present), suggested))
    return failures


def _audit_semantic_names(trace):
    """Defect #8: task display names must be operator names, not codenames.

    Names like "passA" / "passB" / "token" / "EV_ATTN_CHUNK" describe the
    instrumentation's internal structure and are useless on a timeline that a
    human reads to find which operator is slow.
    """
    if trace is None:
        return ["无法读取候选 trace，无法验证 task 语义命名"]
    events = trace.get("traceEvents", trace if isinstance(trace, list) else [])
    offenders = {}
    for event in events:
        if event.get("ph") != "X":
            continue
        args = event.get("args") or {}
        if args.get("layer") != "task":
            continue
        display = str(event.get("name", "")).strip()
        if display in offenders:
            continue
        probe = display.lower()
        reason = None
        if len(probe) < _CODENAME_MIN_CHARS:
            reason = "名字短于 {} 个字符".format(_CODENAME_MIN_CHARS)
        elif not any(char.isalpha() for char in probe):
            reason = "名字里没有字母"
        else:
            for pattern, why in _CODENAME_PATTERNS:
                if re.search(pattern, probe):
                    reason = why
                    break
        if reason:
            offenders[display] = reason
    if not offenders:
        return []
    listed = ", ".join("{!r}({})".format(name, why)
                       for name, why in sorted(offenders.items())[:8])
    return ["{} 个 task 显示名是内部代号而不是算子语义名: {}；"
            "时间线上的名字必须让人直接看出是哪个算子".format(len(offenders), listed)]


def _sibling_module(filename, module_name):
    """Import a sibling validator module for its canonical predicates."""
    import importlib.util
    here = Path(__file__).resolve().parent
    # deliver_gate.py imports its sibling validate_capability at module scope, so
    # the directory has to be importable even when validate_all is loaded as a
    # library instead of being run as a script.
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    spec = importlib.util.spec_from_file_location(module_name, str(here / filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _deliver_gate_module():
    return _sibling_module("deliver_gate.py", "fast_instrument_deliver_gate")


def _audit_trace_filename(contract, trace_path):
    """Defect #13: the delivered filename must be self-describing AND truthful.

    Two teeth, both fail-closed:

    1. syntax: the candidate trace filename must satisfy the canonical SPEC-1
       schema (trace__<model>_<stage>__card-<h20|b300|mi308x>__bs<B>_sq<S>
       [_kv<K>]__<dtype>__grid<G>_warp<W>_ph<P>__<bundle>.perfetto.json.gz)
       -- model, stage, card, shape and HW/SW bundle must be identifiable
       without opening the file.  The check reuses
       deliver_gate.check_trace_filename_schema so the two enforcement points
       can never drift apart.
    2. reconciliation: fields parsed from the name must agree with the locked
       contract -- <card> with target.card, <bundle> with
       capture.information_bundle (sw/hw/sw_hw), and grid<G> with
       build_and_launch.grid[0].  target.card must agree with target.platform
       and target.arch: h20=>nv/sm_90, b300=>nv/sm_100a|sm_103a,
       mi308x=>amd/gfx942.  A syntactically valid name that lies about the run
       still fails.  bs/sq/kv/dtype have no machine-comparable contract fields
       (flattened shapes, mixed input dtypes), so their presence is enforced by
       tooth 1 and their values stay declarative; ph<P> is a renderer-chosen
       phase count, likewise presence-only.
    """
    failures = []
    gate = _deliver_gate_module()
    # A missing contract is not a syntax-only invitation: the card and target
    # identity are mandatory, so pass an empty object into the canonical gate
    # and let it report target_missing instead of silently skipping the check.
    name_fails, _notes = gate.check_trace_filename_schema(
        str(trace_path), contract if isinstance(contract, dict) else {})
    for fail_id, message in name_fails:
        failures.append("{}: {}".format(fail_id, message))
    if failures:
        return failures
    name = Path(str(trace_path)).name
    parsed = gate.parse_trace_filename(name)
    if parsed is None:
        failures.append("trace_filename_schema: 合法文件名解析失败")
        return failures
    name_grid = int(parsed["grid"])
    name_bundle = parsed["bundle"]
    bundle_token = {"sw": "sw", "hw": "hw", "sw+hw": "sw_hw"}
    contract_bundle = _first_present(contract or {}, [("capture", "information_bundle")])
    if isinstance(contract_bundle, str) and contract_bundle in bundle_token:
        if bundle_token[contract_bundle] != name_bundle:
            failures.append(
                "文件名 bundle={!r} 与契约 capture.information_bundle={!r} 不符："
                "文件名必须如实反映本次采集的 HW/SW 内容".format(name_bundle, contract_bundle))
    grid = _first_present(contract or {}, [("build_and_launch", "grid")])
    if isinstance(grid, (list, tuple)) and grid:
        try:
            contract_grid = int(grid[0])
        except (TypeError, ValueError):
            contract_grid = None
        if contract_grid is not None and contract_grid != name_grid:
            failures.append(
                "文件名 grid{} 与契约 build_and_launch.grid[0]={} 不符："
                "文件名必须如实反映本次 launch 的 grid".format(name_grid, contract_grid))
    return failures


def _audit_timeline_structure(trace):
    """Defect #9: G18's own predicates only ran inside the G25 finalizer.

    deliver_gate.py owns the timeline-structure verdict, but the pipeline only
    reaches it through finalize_delivery.py.  A project-side G18 adapter that is
    just a file-existence stub therefore reported PASS while nothing about the
    structure had been looked at, and the real verdict arrived one stage later
    with no gate to attribute it to.  Running the same predicates here keeps the
    G18 row honest and fails closed before delivery.
    """
    if trace is None:
        return ["无法读取候选 trace，无法验证时间线结构"]
    events = trace.get("traceEvents", trace if isinstance(trace, list) else [])
    if not events:
        return ["候选 trace 没有任何事件，时间线结构无从验证"]
    gate = _deliver_gate_module()
    problems = []
    negative_ts = gate.check_nonnegative_timestamps(events)
    if negative_ts:
        problems.append("{} 条事件的显示时间戳 ts 为负数(首条 {!r} ts={})；"
                        "rebase 之后所有显示时间必须 >= 0".format(
                            len(negative_ts), negative_ts[0].get("name"),
                            negative_ts[0].get("ts")))
    negative_dur = [event for event in events
                    if event.get("ph") == "X"
                    and isinstance(event.get("dur"), (int, float))
                    and not isinstance(event.get("dur"), bool)
                    and float(event["dur"]) < 0]
    if negative_dur:
        problems.append("{} 条区间的时长 dur 为负数(首条 {!r} dur={})；"
                        "end 早于 start 说明配对或时钟读取出错".format(
                            len(negative_dur), negative_dur[0].get("name"),
                            negative_dur[0].get("dur")))
    crossings = gate.check_no_partial_overlap(events)
    if crossings:
        listed = ["{!r} [{:.2f}, {:.2f}]us 越过了同泳道上一层的 {:.2f}us".format(
            name, start, end, outer) for _lane, name, start, end, outer in crossings[:5]]
        problems.append("{} 处同泳道区间部分交叉(既不相离也不严格嵌套): {}；"
                        "Perfetto 导入时会把这些区间丢弃或溢出".format(
                            len(crossings), listed))
    return problems


def _audit_correctness_attribution(run_dir):
    """A failing G02 has to name which side is broken.

    The project-side G02 adapter is a bare pass/fail on all_passed, so a failed
    run reports a number and nothing else.  But that one number is broken by two
    unrelated things: the instrumentation really changed the output, or the kernel
    is not reproducible run-to-run and no instrumentation could ever pass.  The
    first sends someone into the profiler for hours; the second means the kernel
    is what needs fixing.  Reporting them identically wastes the whole debug pass.
    """
    doc = _load_json_quiet(Path(run_dir) / "validation" / "correctness.json")
    if doc is None:
        return ["缺少实测 correctness.json，无法验证插桩前后逐元素一致"]
    failures = []
    if doc.get("sample_only") is True:
        failures.append("correctness.sample_only=true：G02 要求逐一比较全部输出元素，禁止抽样")
    if doc.get("all_passed") is not True:
        module = _sibling_module("validate_run.py", "fast_instrument_validate_run")
        failures.append(module.attribute_correctness_failure(doc))
    return failures


def skill_mandatory_audits(contract, run_dir, trace_path, bundle):
    """Run the skill-authoritative product-quality audits (fail-closed).

    Each audit is isolated: one raising must not stop the others, otherwise
    a single unrelated exception collapses the whole report into one message and
    every remaining defect goes unreported.
    """
    results = []
    try:
        trace = _load_trace_quiet(trace_path)
    except Exception as error:
        return [("mandatory_audit_error", "候选 trace 读取失败: {}".format(error))]
    audits = (
        ("overhead_ceiling", lambda: _audit_overhead(contract, run_dir, trace)),
        ("sm_full_grid_baseline", lambda: _audit_sm_scope(contract)),
        ("hw_dram_coverage", lambda: _audit_hw_dram(contract, run_dir, bundle)),
        ("lane_purity", lambda: _audit_lane_purity(trace)),
        ("timeline_structure", lambda: _audit_timeline_structure(trace)),
        ("correctness_attribution", lambda: _audit_correctness_attribution(run_dir)),
        ("operator_task_coverage",
         lambda: _audit_operator_task_coverage(run_dir, trace, bundle)),
        ("hw_scale_plausibility",
         lambda: _audit_hw_scale_plausibility(contract, run_dir, bundle)),
        ("tile_granularity_mandatory",
         lambda: _audit_tile_granularity(contract, run_dir, bundle)),
        ("hw_required_totals", lambda: _audit_hw_required_totals(run_dir, bundle)),
        ("semantic_naming", lambda: _audit_semantic_names(trace)),
        ("semantic_task_bijection",
         lambda: audit_semantic_task_bijection(contract, run_dir, trace)),
        ("trace_filename_schema",
         lambda: _audit_trace_filename(contract, trace_path)),
    )
    for audit_id, run_audit in audits:
        try:
            results.extend((audit_id, msg) for msg in run_audit())
        except Exception as error:
            results.append((audit_id, "审计执行异常({}): {}".format(
                type(error).__name__, error)))
    return results


def validate(args):
    contract_path = Path(args.contract).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    trace_path = Path(args.trace).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    contract_hash = None
    trace_hash = None
    contract = {}
    contract_rejected = False
    try:
        contract_hash = sha256_file(contract_path)
        contract = load_json(contract_path)
        if not isinstance(contract, dict):
            errors.append("contract root must be an object")
            contract = {}
        else:
            if contract.get("example_only") is True:
                errors.append("example_only contract is rejected for production validation")
                contract_rejected = True
            if contract.get("production_validation_expected") == "REJECT":
                errors.append("contract production_validation_expected=REJECT")
                contract_rejected = True
    except Exception as error:
        errors.append("contract error: {}".format(error))
    if trace_path.is_file():
        try:
            trace_hash = sha256_file(trace_path)
        except Exception as error:
            errors.append("trace hash error: {}".format(error))
    else:
        errors.append("trace does not exist: {}".format(trace_path))
    expected = expected_trace_hash(contract)
    if expected is not None and trace_hash != expected:
        errors.append("trace SHA256 mismatch: expected {}, observed {}".format(expected, trace_hash))

    bundle = information_bundle(contract)
    required, not_applicable, partition_errors = validate_partition(contract, bundle)
    errors.extend(partition_errors)
    mandatory_audits = []
    if _is_production_contract(contract) and not contract_rejected:
        for message in selftest_mandatory_audits():
            errors.append("mandatory_audit_selftest: {}".format(message))
        mandatory_audits = skill_mandatory_audits(contract, run_dir, trace_path, bundle)
        for audit_id, message in mandatory_audits:
            errors.append("mandatory_audit[{}]: {}".format(audit_id, message))
    execution = contract.get("execution") if isinstance(contract, dict) else {}
    if not isinstance(execution, dict):
        execution = {}
    commands = execution.get("gate_commands")
    if commands is None:
        commands = contract.get("gate_commands") if isinstance(contract, dict) else None
    command_template = execution.get("gate_command_template")

    records = []
    # A partition error invalidates the run and prevents accidental partial execution.
    if not partition_errors and not contract_rejected:
        specs = {}
        for gate_id in required:
            parsed, parse_error = command_spec(commands, gate_id, command_template)
            if parse_error:
                specs[gate_id] = None
            else:
                specs[gate_id] = parsed
        with ThreadPoolExecutor(max_workers=max(1, len(required))) as pool:
            futures = {
                pool.submit(run_gate, gate_id, specs[gate_id], run_dir): gate_id
                for gate_id in required
            }
            completed = {gate_id: future.result() for future, gate_id in ((f, futures[f]) for f in as_completed(futures))}
        gate_records = [completed[gate_id] for gate_id in required]
        for gate_id, parse_error in ((gate_id, command_spec(commands, gate_id, command_template)[1]) for gate_id in required):
            if parse_error:
                gate_records[required.index(gate_id)]["reason"] = parse_error
    else:
        gate_records = []
        for gate_id in required:
            record = base_gate_record(gate_id, run_dir)
            record["reason"] = "run-level validation failed; gate not executed"
            write_log(record["stdout_log"], "")
            write_log(record["stderr_log"], record["reason"] + "\n")
            gate_records.append(record)

    # SPEC-4: record the not-applicable gates as N/A (never executed) so the
    # emitted ALL_GATES.json still accounts for every gate ID.
    na_gate_ids = [gate_id for gate_id in ALL_GATE_IDS
                   if gate_id in {item for item in not_applicable if isinstance(item, str)}]
    na_records = [na_gate_record(gate_id, run_dir, bundle) for gate_id in na_gate_ids]
    gate_records.extend(na_records)

    contract_hash_after = None
    trace_hash_after = None
    integrity_errors = []
    if contract_path.is_file():
        try:
            contract_hash_after = sha256_file(contract_path)
        except Exception as error:
            integrity_errors.append("contract post-run hash error: {}".format(error))
    if trace_path.is_file():
        try:
            trace_hash_after = sha256_file(trace_path)
        except Exception as error:
            integrity_errors.append("trace post-run hash error: {}".format(error))
    if contract_hash_after != contract_hash:
        integrity_errors.append("contract changed during gate execution")
    if trace_hash_after != trace_hash:
        integrity_errors.append("trace changed during gate execution")
    errors.extend(integrity_errors)

    counts = {status: sum(1 for record in gate_records if record["status"] == status)
              for status in ("PASS", "FAIL", "BLOCKED", "N/A")}
    counts_consistent = (
        len(gate_records) == len(required) + len(na_records) and
        sum(counts.values()) == len(gate_records) and
        all(record.get("status") in ("PASS", "FAIL", "BLOCKED", "N/A") for record in gate_records)
    )
    if not counts_consistent:
        errors.append("gate status counts are inconsistent with gate records")
    # SPEC-4: N/A never blocks the finalizer; only FAIL/BLOCKED do.
    all_pass = not errors and counts["FAIL"] == 0 and counts["BLOCKED"] == 0
    payload = {
        "schema_version": "all_gates.v1",
        "status": "PASS" if all_pass else "FAIL",
        "generated_at_utc": utc_now(),
        "contract_path": str(contract_path),
        "contract_sha256": contract_hash,
        "contract_sha256_after": contract_hash_after,
        "trace_path": str(trace_path),
        "trace_sha256": trace_hash,
        "trace_sha256_after": trace_hash_after,
        "integrity_errors": integrity_errors,
        "trace_expected_sha256": expected,
        "run_dir": str(run_dir),
        "information_bundle": bundle,
        "required_gate_ids": required,
        "not_applicable_gate_ids": not_applicable,
        "partition_errors": partition_errors,
        "input_errors": errors,
        "mandatory_audit_failures": [{"audit": audit_id, "reason": message}
                                     for audit_id, message in mandatory_audits],
        "gates": gate_records,
        "summary": {
            "required_count": len(required),
            "not_applicable_count": len(na_records),
            "pass": counts["PASS"],
            "fail": counts["FAIL"],
            "blocked": counts["BLOCKED"],
            "not_applicable": counts["N/A"],
            "counts_consistent": counts_consistent,
            "all_required_pass": all_pass,
        },
    }
    try:
        atomic_dump(payload, out_path)
    except Exception as error:
        print("[fatal] cannot write {}: {}".format(out_path, error), file=sys.stderr)
        return 2
    print("ALL_GATES={}".format(out_path))
    print("STATUS={}".format(payload["status"]))
    return 0 if all_pass else 1


def make_selftest_contract(path, required, commands, expected_trace=None, not_applicable=None):
    if not_applicable is None:
        not_applicable = [gate_id for gate_id in ALL_GATE_IDS if gate_id not in required]
    payload = {
        "execution": {
            "required_gate_ids": required,
            "not_applicable_gate_ids": not_applicable,
            "gate_commands": commands,
        }
    }
    if expected_trace is not None:
        payload["trace_sha256"] = expected_trace
    path.write_text(json.dumps(payload), encoding="utf-8")


_SELFTEST_BASE_NS = 1_788_000_000_000_000_000
_SELFTEST_SPAN_NS = 2_000_000


def _selftest_stage(directory, spans, counters=None, rules=None):
    """Materialize a synthetic run dir + trace for the audit selftest."""
    directory = Path(directory)
    (directory / "hw").mkdir(parents=True, exist_ok=True)
    events = []
    for name, start, end in spans:
        events.append({
            "name": name, "ph": "X", "pid": 1000, "tid": 1,
            "ts": start / 1000.0, "dur": (end - start) / 1000.0,
            "args": {"layer": "task",
                     "start_ns_raw": _SELFTEST_BASE_NS + start,
                     "end_ns_raw": _SELFTEST_BASE_NS + end},
        })
    trace = {
        "traceEvents": events,
        "otherData": {"hw_merge": {"sw_window_ns": [
            _SELFTEST_BASE_NS, _SELFTEST_BASE_NS + _SELFTEST_SPAN_NS]}},
    }
    with open(str(directory / "hw" / "hw_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({"counters": counters or [], "scale_rules": rules or []}, handle)
    return trace


def selftest_mandatory_audits():
    """Lock in the direction of the four evidence-based audits (no GPU needed).

    Every case asserts both halves: the defect is caught, and the correct
    artifact is left alone.  Without this, a refactor could silently turn any
    audit into a no-op and the whole gate suite would still report PASS.
    """
    span = _SELFTEST_SPAN_NS
    good_spans = [("Operator {} Proj".format(i), i * (span // 20), (i + 1) * (span // 20))
                  for i in range(20)]
    pinned = [[i * 5000, 25 if i % 3 else 0] for i in range(60)]
    raw_pair = [
        {"name": "Tensor Active", "unit": "Throughput %", "samples": pinned},
        {"name": "SMs Active", "unit": "Throughput %",
         "samples": [[i * 5000, 100] for i in range(60)]},
    ]
    corrected_pair = raw_pair + [
        {"name": "Tensor Active (corrected x4)", "unit": "Throughput %",
         "is_derived": True, "derived_from": "Tensor Active", "samples": pinned},
    ]
    # An integer-quantized counter whose own maximum lands on 100/4 for two
    # samples out of a thousand.  That is a spike, not a die-normalization
    # ceiling, and used to be reported as one.
    quantized_spike = [
        {"name": "L2 Hit Rate", "unit": "Throughput %",
         "samples": [[i * 5000, 25 if i in (137, 842) else i % 20]
                     for i in range(1000)]},
    ]
    dram_split = [
        {"name": "DRAM Read Bandwidth", "unit": "Throughput %", "samples": [[0, 44]]},
        {"name": "DRAM Write Bandwidth", "unit": "Throughput %", "samples": [[0, 21]]},
    ]
    dram_total = dram_split + [
        {"name": "DRAM Bandwidth Total", "unit": "Throughput %", "samples": [[0, 45]]},
    ]
    # (label, spans, counters, rules, audit, expect_failure, must_mention)
    cases = (
        ("只有 attention 一段 task 时必须报头尾空洞",
         [("Sparse Attention", 769_000, 1_668_000)], None, None,
         "coverage", True, "769"),
        ("整核锚点被误标成 task 时不得靠它凑满覆盖率",
         [("Whole Kernel Execution", 0, span),
          ("Sparse Attention", 769_000, 1_668_000)], None, None,
         "coverage", True, "Whole Kernel Execution"),
        ("20 段算子铺满窗口必须放行", good_spans, None, None, "coverage", False, None),
        ("passA/passB 代号名必须被拒",
         [("passA", 0, span // 2), ("passB", span // 2, span)], None, None,
         "naming", True, "passA"),
        ("人类可读算子名必须放行", good_spans, None, None, "naming", False, None),
        ("百分比指标钉在 100/4 且无修正必须被拒",
         good_spans, raw_pair, None, "scale", True, "100/4"),
        ("已发布 x4 修正派生曲线后必须放行",
         good_spans, corrected_pair,
         [{"source_name": "Tensor Active", "factor": 4, "applied": True}],
         "scale", False, None),
        ("整数量化指标偶然两次触顶 100/4 不算平台期，必须放行",
         good_spans, quantized_spike, None, "scale", False, None),
        ("DRAM 只有读写分量、缺总曲线必须被拒",
         good_spans, dram_split, None, "totals", True, "DRAM Bandwidth Total"),
        ("DRAM 补上总曲线后必须放行", good_spans, dram_total, None, "totals", False, None),
        ("同泳道区间部分交叉必须被拒",
         [("Q K V Projection", 0, 600_000), ("Attention Score", 300_000, 900_000)],
         None, None, "structure", True, "部分交叉"),
        ("end 早于 start 的负时长区间必须被拒",
         [("Output Projection", 900_000, 300_000)], None, None,
         "structure", True, "dur 为负数"),
        ("相离的 20 段区间结构必须放行", good_spans, None, None, "structure", False, None),
    )
    failures = []
    failures.extend(selftest_semantic_task_bijection())
    temp = tempfile.mkdtemp(prefix="mandatory_audit_selftest_")
    try:
        for index, case in enumerate(cases):
            label, spans, counters, rules, audit, expect_failure, mention = case
            stage = Path(temp) / "case{}".format(index)
            trace = _selftest_stage(stage, spans, counters, rules)
            if audit == "coverage":
                found = _audit_operator_task_coverage(stage, trace, "sw+hw")
            elif audit == "naming":
                found = _audit_semantic_names(trace)
            elif audit == "structure":
                found = _audit_timeline_structure(trace)
            elif audit == "scale":
                found = _audit_hw_scale_plausibility({"capture": {}}, stage, "sw+hw")
            else:
                found = _audit_hw_required_totals(stage, "sw+hw")
            if expect_failure and not found:
                failures.append("{}：审计没有报错(退化成空操作)".format(label))
            elif not expect_failure and found:
                failures.append("{}：审计误报 {}".format(label, found[:1]))
            elif expect_failure and mention and not any(mention in msg for msg in found):
                failures.append("{}：报错信息里没有 {!r}，缺少可定位证据".format(label, mention))
        # The attribution audit is driven by a correctness.json fixture instead of
        # spans/counters, so it gets its own loop.  The last case is the trap taken
        # from a real run: both off-vs-off trials stay inside the tolerance and
        # report all_passed=true, yet the two runs are not bit-identical.  Judging
        # by the tolerance would blame the instrumentation for a kernel defect.
        bit_exact = [{"all_passed": True, "per_tensor": [{"max_abs_err": 0.0}]}] * 2
        tolerant_but_not_exact = [
            {"all_passed": True, "per_tensor": [{"max_abs_err": 0.015625}]},
            {"all_passed": True, "per_tensor": [{"max_abs_err": 0.0}]},
        ]
        attribution_cases = (
            ("插桩前后逐元素一致时不得报错",
             {"all_passed": True, "sample_only": False}, False, None),
            ("缺少 correctness.json 必须被拒", None, True, "correctness.json"),
            ("失败但没有 off-vs-off 证据必须被拒",
             {"all_passed": False, "sample_only": False}, True,
             "baseline_self_consistency.trials"),
            ("未插桩两次逐位相同 -> 必须归因到插桩",
             {"all_passed": False, "sample_only": False,
              "baseline_self_consistency": {"trials": bit_exact}}, True, "要修的是插桩"),
            ("未插桩两次落在容差内但不逐位相同 -> 必须归因到 kernel",
             {"all_passed": False, "sample_only": False,
              "baseline_self_consistency": {"trials": tolerant_but_not_exact}}, True,
             "run-to-run 不可重现"),
        )
        for index, case in enumerate(attribution_cases):
            label, doc, expect_failure, mention = case
            stage = Path(temp) / "attribution{}".format(index)
            (stage / "validation").mkdir(parents=True, exist_ok=True)
            if doc is not None:
                (stage / "validation" / "correctness.json").write_text(
                    json.dumps(doc), encoding="utf-8")
            found = _audit_correctness_attribution(stage)
            if expect_failure and not found:
                failures.append("{}：审计没有报错(退化成空操作)".format(label))
            elif not expect_failure and found:
                failures.append("{}：审计误报 {}".format(label, found[:1]))
            elif expect_failure and mention and not any(mention in msg for msg in found):
                failures.append("{}：报错信息里没有 {!r}，缺少可定位证据".format(label, mention))
        # Tile-granularity audit fixtures (v2: declaration + size cap + set
        # reconciliation).  Tiny grid=4; grid_stride pass slice0 num_tiles=8
        # expects {(c, c+4k)}; both directions asserted.
        def _tile_fixture(stage, abi_tasks, events, grid=4, abi_extra=None):
            stage = Path(stage)
            (stage / "sw").mkdir(parents=True, exist_ok=True)
            abi_path = stage / "abi.json"
            abi_doc = {
                "tasks": abi_tasks,
                "work_id_packing": {"fields": {"slice": {"lsb": 8}, "chunk": {"lsb": 0}}},
            }
            if isinstance(abi_extra, dict):
                abi_doc.update(abi_extra)
            abi_path.write_text(json.dumps(abi_doc), encoding="utf-8")
            if events is not None:
                (stage / "sw" / "events_rank0.json").write_text(
                    json.dumps({"events": events}), encoding="utf-8")
            return {"capture": {"display_rank": 0},
                    "build_and_launch": {"grid": [grid, 1, 1]},
                    "profile_selection": {"abi": {"absolute_path": str(abi_path)}}}

        good_pass = {"slice": 0, "kind": "grid_stride", "num_tiles": 8,
                     "ctas": [0, 4], "tile_axes": [64, 128]}
        good_events = [{"event_id": 100, "logical_cta": c, "work_id": c + 4 * k}
                       for c in range(4) for k in range(2)]
        big_pass = dict(good_pass, tile_axes=[64, 1024])
        tile_cases = (
            ("域公式与记录集合精确相等 + tile 轴不超标 必须放行",
             _tile_fixture(Path(temp) / "tile_ok",
                           [{"id": 100, "name": "Tile Load",
                             "tile_domain": {"passes": [good_pass]}}], good_events),
             False, None),
            ("缺一个 tile 必须被拒",
             _tile_fixture(Path(temp) / "tile_missing",
                           [{"id": 100, "name": "Tile Load",
                             "tile_domain": {"passes": [good_pass]}}],
                           good_events[:-1]),
             True, "不对账"),
            ("多一个 tile 必须被拒",
             _tile_fixture(Path(temp) / "tile_extra",
                           [{"id": 100, "name": "Tile Load",
                             "tile_domain": {"passes": [good_pass]}}],
                           good_events + [{"event_id": 100, "logical_cta": 0, "work_id": 99}]),
             True, "不对账"),
            ("tile 轴 1024 超过 512 上限必须被拒",
             _tile_fixture(Path(temp) / "tile_oversize",
                           [{"id": 100, "name": "Tile Load",
                             "tile_domain": {"passes": [big_pass]}}], good_events),
             True, "超过上限"),
            ("ABI 完全没有 tile_domain 声明必须被拒",
             _tile_fixture(Path(temp) / "tile_undeclared",
                           [{"id": 20, "name": "GEMM"}], good_events),
             True, "tile_domain"),
            ("声明了 tile 事件但解码事件流缺失必须被拒",
             _tile_fixture(Path(temp) / "tile_noevents",
                           [{"id": 100, "name": "Tile Load",
                             "tile_domain": {"passes": [good_pass]}}], None),
             True, "events_rank0"),
            ("legacy task_contract.tile_domain 声明（llama 惯例）必须放行",
             _tile_fixture(Path(temp) / "tile_legacy",
                           [{"id": 120, "name": "Tile MMA",
                             "task_contract": {"tile_domain": {"num_tiles": 8}}}],
                           good_events),
             False, None),
            ("无 tile 结构 + 哈希锁定的 tile_free 源码级证明 必须放行",
             _tile_fixture(Path(temp) / "tile_free_ok",
                           [{"id": 20, "name": "Reduce"}], good_events,
                           abi_extra={"semantic_analysis": {
                               "tile_free": {"justification":
                                   "kernel 只做逐元素归约：launch 为一 CTA 一输出元素，"
                                   "源码无任何 tile 循环或 tile 映射常量"}}}),
             False, None),
            ("无 tile 结构但 tile_free.justification 为空 必须被拒",
             _tile_fixture(Path(temp) / "tile_free_empty",
                           [{"id": 20, "name": "Reduce"}], good_events,
                           abi_extra={"semantic_analysis": {
                               "tile_free": {"justification": "  "}}}),
             True, "tile_free"),
        )
        for label, contract, expect_failure, mention in tile_cases:
            run_dir = Path(contract["profile_selection"]["abi"]["absolute_path"]).parent
            found = _audit_tile_granularity(contract, run_dir, "sw+hw")
            if expect_failure and not found:
                failures.append("{}：审计没有报错(退化成空操作)".format(label))
            elif not expect_failure and found:
                failures.append("{}：审计误报 {}".format(label, found[:1]))
            elif expect_failure and mention and not any(mention in msg for msg in found):
                failures.append("{}：报错信息里没有 {!r}，缺少可定位证据".format(label, mention))

        # Overhead-ceiling audit fixtures: two-tier hard ceiling (task 10% /
        # phase 15%), and the phase allowance only counts when the delivered
        # trace really carries phase-layer intervals.  Both directions asserted.
        def _overhead_fixture(stage, granularity, ratio, with_phases,
                              maximum=None):
            stage = Path(stage)
            (stage / "validation").mkdir(parents=True, exist_ok=True)
            (stage / "validation" / "overhead.json").write_text(
                json.dumps({"overhead_ratio": ratio}), encoding="utf-8")
            overhead = {"maximum": maximum} if maximum is not None else {}
            contract = {"capture": {"granularity": granularity},
                        "validation": {"overhead": overhead}}
            trace = {"traceEvents": [
                {"ph": "X", "name": "Quant GEMM", "ts": 0, "dur": 900,
                 "pid": 0, "tid": 1, "args": {"layer": "task"}}]}
            if with_phases:
                trace["traceEvents"].append(
                    {"ph": "X", "name": "Tile Load", "ts": 0, "dur": 400,
                     "pid": 0, "tid": 1, "args": {"layer": "phase"}})
            return contract, stage, trace

        overhead_cases = (
            ("task 档 12% 开销必须被拒",
             _overhead_fixture(Path(temp) / "ov_task_fail", "task", 0.12, False),
             True, "10%"),
            ("task 档 9% 开销必须放行",
             _overhead_fixture(Path(temp) / "ov_task_ok", "task", 0.09, False),
             False, None),
            ("phase 档 12% 开销且交付含 phase 区间 必须放行",
             _overhead_fixture(Path(temp) / "ov_phase_ok", "phase", 0.12, True),
             False, None),
            ("phase 档 16% 开销必须被拒",
             _overhead_fixture(Path(temp) / "ov_phase_fail", "phase", 0.16, True),
             True, "15%"),
            ("声明 phase 但交付 trace 无 phase 区间，12% 必须按 10% 被拒",
             _overhead_fixture(Path(temp) / "ov_phase_sham", "phase", 0.12, False),
             True, "phase 层区间"),
            ("契约 maximum=0.15 用于 task 档必须被拒",
             _overhead_fixture(Path(temp) / "ov_task_max15", "task", 0.05, False,
                               maximum=0.15),
             True, "0.10"),
            ("契约 maximum=0.15 用于 phase 档且开销合规 必须放行",
             _overhead_fixture(Path(temp) / "ov_phase_max15", "phase", 0.12, True,
                               maximum=0.15),
             False, None),
        )
        for label, (contract, run_dir, trace), expect_failure, mention in overhead_cases:
            found = _audit_overhead(contract, run_dir, trace)
            if expect_failure and not found:
                failures.append("{}：审计没有报错(退化成空操作)".format(label))
            elif not expect_failure and found:
                failures.append("{}：审计误报 {}".format(label, found[:1]))
            elif expect_failure and mention and not any(mention in msg for msg in found):
                failures.append("{}：报错信息里没有 {!r}，缺少可定位证据".format(label, mention))

        # Trace-filename audit fixtures: SPEC-1 syntax tooth plus the
        # bundle/grid reconciliation against the locked contract.  The audit
        # only inspects the file NAME, so no real file needs to exist.
        def _name_fixture(name, information_bundle="sw+hw", grid=78,
                          card="h20", platform="nv", arch="sm_90"):
            contract = {"target": {"card": card, "platform": platform, "arch": arch},
                        "capture": {"information_bundle": information_bundle},
                        "build_and_launch": {"grid": [grid, 1, 1]}}
            return contract, Path(temp) / name

        good_name = ("trace__dsv4pro_prefill__card-h20__bs1_sq4096_kv16384__bf16"
                     "__grid78_warp12_ph13__sw_hw.perfetto.json.gz")
        b300_name = good_name.replace("card-h20", "card-b300")
        mi308x_name = good_name.replace("card-h20", "card-mi308x")
        name_cases = (
            ("H20 模型/size/dtype/grid/bundle 齐全的合法文件名必须放行",
             _name_fixture(good_name), False, None),
            ("B300 + sm_100a 必须放行",
             _name_fixture(b300_name, card="b300", arch="sm_100a"),
             False, None),
            ("B300 + sm_103a 必须放行",
             _name_fixture(b300_name, card="b300", arch="sm_103a"),
             False, None),
            ("MI308X + gfx942 必须放行",
             _name_fixture(mi308x_name, card="mi308x", platform="amd",
                           arch="gfx942"), False, None),
            ("缺 bundle 段的文件名必须被拒",
             _name_fixture("trace__dsv4pro_prefill__card-h20__bs1_sq4096__bf16"
                           "__grid78_warp12_ph13.perfetto.json.gz"),
             True, "schema"),
            ("旧的无 card 文件名必须被拒",
             _name_fixture("trace__dsv4pro_prefill__bs1_sq4096_kv16384__bf16"
                           "__grid78_warp12_ph13__sw_hw.perfetto.json.gz"),
             True, "schema"),
            ("未知卡型必须被拒",
             _name_fixture(good_name.replace("card-h20", "card-h100")),
             True, "schema"),
            ("文件名 card 与契约 target.card 不符必须被拒",
             _name_fixture(good_name, card="b300", arch="sm_103a"),
             True, "target.card"),
            ("H20 搭配 sm_100a 必须被拒",
             _name_fixture(good_name, arch="sm_100a"),
             True, "target_arch_mismatch"),
            ("B300 搭配 sm_90 必须被拒",
             _name_fixture(b300_name, card="b300", arch="sm_90"),
             True, "target_arch_mismatch"),
            ("MI308X 搭配 nv 必须被拒",
             _name_fixture(mi308x_name, card="mi308x", platform="nv",
                           arch="gfx942"), True, "target_platform_mismatch"),
            ("文件名 sw_hw 但契约是 sw-only 必须被拒",
             _name_fixture(good_name, information_bundle="sw"),
             True, "sw_hw"),
            ("文件名 grid39 但契约 grid=78 必须被拒",
             _name_fixture(good_name.replace("grid78", "grid39")),
             True, "grid39"),
        )
        for label, (contract, trace_path), expect_failure, mention in name_cases:
            found = _audit_trace_filename(contract, trace_path)
            if expect_failure and not found:
                failures.append("{}：审计没有报错(退化成空操作)".format(label))
            elif not expect_failure and found:
                failures.append("{}：审计误报 {}".format(label, found[:1]))
            elif expect_failure and mention and not any(mention in msg for msg in found):
                failures.append("{}：报错信息里没有 {!r}，缺少可定位证据".format(label, mention))
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return failures


def selftest():
    root = Path(__file__).resolve().parent / ".validate_all_selftest"
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run_", dir=str(root)))
    try:
        audit_failures = selftest_mandatory_audits()
        if audit_failures:
            raise AssertionError("mandatory audit selftest failed: {}".format(audit_failures))
        trace = work / "trace.json.gz"
        trace.write_bytes(b"trace")
        evidence = work / "evidence.json"
        py = sys.executable
        write_code = "from pathlib import Path; Path('evidence.json').write_text('ok')"
        required = list(ALL_GATE_IDS)
        na = []
        pass_command = {
            gate_id: {"argv": [py, "-c", write_code], "evidence": ["evidence.json"]}
            if gate_id == "G00" else {"argv": [py, "-c", "pass"]}
            for gate_id in ALL_GATE_IDS
        }

        def invoke(name, contract, expect_zero):
            contract_path = work / (name + ".contract.json")
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            out = work / (name + ".ALL_GATES.json")
            code = validate(argparse.Namespace(contract=str(contract_path), run_dir=str(work), trace=str(trace), out=str(out)))
            if (code == 0) != expect_zero or not out.is_file():
                raise AssertionError(name + " returned unexpected result")
            return load_json(out)

        contract = {"execution": {"required_gate_ids": required, "not_applicable_gate_ids": na,
                                   "gate_commands": pass_command}}
        result = invoke("all_pass", contract, True)
        assert result["gates"][0]["status"] == "PASS"

        # SPEC-4: a software-only bundle marks exactly {G11,G19,G20,G21} N/A.
        sw_required = [gate_id for gate_id in ALL_GATE_IDS if gate_id not in SW_NOT_APPLICABLE]
        sw_na = list(SW_NOT_APPLICABLE)
        sw_result = invoke("sw_bundle", {"execution": {
            "information_bundle": "sw", "required_gate_ids": sw_required,
            "not_applicable_gate_ids": sw_na, "gate_commands": pass_command}}, True)
        sw_status = {gate["gate_id"]: gate["status"] for gate in sw_result["gates"]}
        assert all(sw_status[gate_id] == "N/A" for gate_id in SW_NOT_APPLICABLE), sw_status
        assert all(sw_status[gate_id] == "PASS" for gate_id in sw_required), sw_status
        assert sw_result["summary"]["not_applicable"] == len(SW_NOT_APPLICABLE)
        assert sw_result["summary"]["all_required_pass"] is True
        assert sw_result["status"] == "PASS"
        assert sw_result["summary"]["counts_consistent"] is True
        # SPEC-4: the very same not-applicable partition is illegal for hardware.
        invoke("sw_na_rejected_for_hw", {"execution": {
            "information_bundle": "sw+hw", "required_gate_ids": sw_required,
            "not_applicable_gate_ids": sw_na, "gate_commands": pass_command}}, False)

        template_contract = {"execution": {
            "required_gate_ids": required, "not_applicable_gate_ids": na,
            "gate_commands": {},
            "gate_command_template": {
                "argv": [py, "-c", "print('{gate_id}')"],
            },
        }}
        template_result = invoke("template_all", template_contract, True)
        assert all(gate["status"] == "PASS" for gate in template_result["gates"])

        example_result = invoke("example_rejected", {
            "example_only": True,
            "production_validation_expected": "REJECT",
            "execution": {"required_gate_ids": required, "not_applicable_gate_ids": na,
                           "gate_commands": pass_command},
        }, False)
        assert len(example_result["input_errors"]) == 2
        assert example_result["gates"][0]["status"] == "BLOCKED"

        invoke("missing_gate", {"execution": {"required_gate_ids": required[:-1],
            "not_applicable_gate_ids": na, "gate_commands": pass_command}}, False)

        fail_command = {"G00": {"argv": [py, "-c", "raise SystemExit(3)"]}}
        invoke("nonzero", {"execution": {"required_gate_ids": required, "not_applicable_gate_ids": na,
                                             "gate_commands": fail_command}}, False)

        missing_evidence = {"G00": {"argv": [py, "-c", "pass"], "evidence": ["absent.bin"]}}
        result = invoke("missing_evidence", {"execution": {"required_gate_ids": required,
            "not_applicable_gate_ids": na, "gate_commands": missing_evidence}}, False)
        assert result["gates"][0]["status"] == "BLOCKED"

        timeout_command = {"G00": {"argv": [py, "-c", "import time; print('before'); time.sleep(1)"],
                                    "timeout_sec": 0.05}}
        result = invoke("timeout", {"execution": {"required_gate_ids": required,
            "not_applicable_gate_ids": na, "gate_commands": timeout_command}}, False)
        assert result["gates"][0]["status"] == "FAIL"
        assert result["gates"][0].get("timed_out") is True
        assert Path(result["gates"][0]["stdout_log"]).is_file()
        assert Path(result["gates"][0]["stderr_log"]).is_file()

        # Simulate an evidence file that exists but cannot be read, including an
        # expected SHA256, to ensure the record never indexes a missing sha256 key.
        read_error_command = {"G00": {"argv": [py, "-c", "pass"],
                                       "evidence": [{"path": "evidence.json", "sha256": "0" * 64}]}}
        original_sha256_file = globals()["sha256_file"]
        def fail_evidence_hash(path):
            if Path(path).resolve() == evidence.resolve():
                raise OSError("selftest unreadable evidence")
            return original_sha256_file(path)
        globals()["sha256_file"] = fail_evidence_hash
        try:
            result = invoke("evidence_read_error", {"execution": {
                "required_gate_ids": required, "not_applicable_gate_ids": na,
                "gate_commands": read_error_command}}, False)
        finally:
            globals()["sha256_file"] = original_sha256_file
        assert result["gates"][0]["status"] == "BLOCKED"
        assert result["gates"][0]["evidence"][0].get("read_error")

        invoke("bad_trace_hash", {"trace_sha256": "0" * 64, "execution": {
            "required_gate_ids": required, "not_applicable_gate_ids": na, "gate_commands": pass_command}}, False)
        bad_partition = {"execution": {"required_gate_ids": ["G00", "G00"],
            "not_applicable_gate_ids": na, "gate_commands": pass_command}}
        invoke("bad_partition", bad_partition, False)

        mutation_path = work / "contract_mutation.contract.json"
        mutation_out = work / "contract_mutation.ALL_GATES.json"
        mutation_code = "from pathlib import Path; p=Path(r'{}'); p.write_text(p.read_text() + '\\n')".format(mutation_path)
        mutation_commands = dict(pass_command)
        mutation_commands["G00"] = {"argv": [py, "-c", mutation_code]}
        mutation_contract = {"execution": {"required_gate_ids": required,
            "not_applicable_gate_ids": na, "gate_commands": mutation_commands}}
        mutation_path.write_text(json.dumps(mutation_contract), encoding="utf-8")
        mutation_code_result = validate(argparse.Namespace(contract=str(mutation_path), run_dir=str(work),
                                                            trace=str(trace), out=str(mutation_out)))
        assert mutation_code_result != 0
        mutation_result = load_json(mutation_out)
        assert "contract changed during gate execution" in mutation_result["integrity_errors"]
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
    parser = argparse.ArgumentParser(description="Run all contract-declared gates")
    parser.add_argument("--contract")
    parser.add_argument("--run-dir")
    parser.add_argument("--trace")
    parser.add_argument("--out")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    missing = [name for name in ("contract", "run_dir", "trace", "out") if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: {}".format(", ".join("--" + name.replace("_", "-") for name in missing)))
    return validate(args)


if __name__ == "__main__":
    sys.exit(main())
