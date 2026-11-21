#!/usr/bin/env python3
"""Verify a weaveprofiler delivery: receipt + gate evidence + skill manifest.

This is the single choke point every enforcement layer shares: the Stop hook,
the PreToolUse upload guard, a parent agent's acceptance check, or an OSS
upload wrapper all call this and honor its exit code.

Checks (fail-closed, all errors reported at once):
  1. receipt exists, status=PASS, G25 finalization block present and PASS
  2. every hash bound in the receipt recomputes from the live files
  3. ALL_GATES.json: schema all_gates.v1, every required gate PASS, no
     FAIL/BLOCKED, error lists empty, gate partition covers G00..G24
  4. PREFLIGHT.json: every required check PASS, partition covers E01..E27
  5. trace filename matches SPEC-1 and reconciles with the contract bundle
  6. receipt is bound to the exact skill tree via skill_manifest.root_sha256,
     and the working tree still matches the git-committed manifest

Exit codes: 0 PASS | 1 FAIL | 2 usage/internal | 4 not a delivery
(exploratory__ namespace, verification not applicable).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

GATE_IDS = ["G%02d" % i for i in range(25)]
CHECK_IDS = ["E%02d" % i for i in range(1, 28)]
CARD_TARGETS = {
    "h20": {"platform": "nv", "arches": frozenset(("sm_90",))},
    "b300": {"platform": "nv", "arches": frozenset(("sm_100a", "sm_103a"))},
    "mi308x": {"platform": "amd", "arches": frozenset(("gfx942",))},
}
SPEC1 = re.compile(
    r"^trace__(?P<model>[a-z0-9]+)_(?P<stage>[a-z0-9]+)"
    r"__card-(?P<card>h20|b300|mi308x)"
    r"__bs(?P<batch>\d+)_sq(?P<seqlen_q>\d+)(?:_kv(?P<seqlen_kv>\d+))?"
    r"__(?P<dtype>[a-z0-9]+)"
    r"__grid(?P<grid>\d+)_warp(?P<warp>\d+)_ph(?P<phases>\d+)"
    r"__(?P<bundle>sw|hw|sw_hw)\.perfetto\.json\.gz$")
BUNDLE_TO_DISK = {"sw": "sw", "hw": "hw", "sw+hw": "sw_hw"}
EXPLORATORY_PREFIX = "exploratory__"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path, label, errors):
    try:
        with open(str(path), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        errors.append("%s unreadable: %s (%s)" % (label, path, error))
        return None


def check_hash(errors, label, path_str, want):
    if not path_str or not want:
        errors.append("receipt missing %s path/hash binding" % label)
        return
    path = Path(path_str)
    if not path.is_file():
        errors.append("%s missing: %s" % (label, path))
        return
    got = sha256_file(path)
    if got != want:
        errors.append("%s hash mismatch: %s (receipt %s != live %s)"
                      % (label, path, want, got))


def check_gates(errors, path, receipt):
    doc = load_json(path, "ALL_GATES.json", errors)
    if doc is None:
        return
    if doc.get("schema_version") != "all_gates.v1":
        errors.append("ALL_GATES schema_version != all_gates.v1")
    if doc.get("status") != "PASS":
        errors.append("ALL_GATES status=%s" % doc.get("status"))
    summary = doc.get("summary") or {}
    if summary.get("all_required_pass") is not True:
        errors.append("ALL_GATES summary.all_required_pass is not true")
    for key in ("input_errors", "integrity_errors", "partition_errors"):
        if doc.get(key):
            errors.append("ALL_GATES %s non-empty: %s" % (key, doc[key]))
    if doc.get("mandatory_audit_failures"):
        errors.append("ALL_GATES mandatory_audit_failures: %s"
                      % doc["mandatory_audit_failures"])
    for key, receipt_key in (("contract_sha256", "contract_sha256"),
                             ("trace_sha256", "trace_sha256")):
        if doc.get(key) != receipt.get(receipt_key):
            errors.append("ALL_GATES %s does not match receipt %s" % (key, receipt_key))
    required = set(doc.get("required_gate_ids") or [])
    na = set(doc.get("not_applicable_gate_ids") or [])
    if required | na != set(GATE_IDS) or required & na:
        errors.append("ALL_GATES gate partition does not cover G00..G24 cleanly")
    seen = {}
    for record in doc.get("gates") or []:
        seen[record.get("gate_id")] = record.get("status")
    for gate_id in sorted(required):
        if seen.get(gate_id) != "PASS":
            errors.append("required gate %s status=%s" % (gate_id, seen.get(gate_id)))
    for gate_id in sorted(na):
        if seen.get(gate_id) not in (None, "N/A"):
            errors.append("N/A gate %s reported status=%s" % (gate_id, seen.get(gate_id)))


def check_preflight(errors, path):
    doc = load_json(path, "PREFLIGHT.json", errors)
    if doc is None:
        return
    summary = doc.get("summary") or {}
    if summary.get("all_required_pass") is not True:
        errors.append("PREFLIGHT summary.all_required_pass is not true")
    required = set(doc.get("required_check_ids") or [])
    na = set(doc.get("not_applicable_check_ids") or [])
    if required | na != set(CHECK_IDS) or required & na:
        errors.append("PREFLIGHT check partition does not cover E01..E27 cleanly")
    seen = {}
    for record in doc.get("checks") or []:
        seen[record.get("check_id")] = record.get("status")
    for check_id in sorted(required):
        if seen.get(check_id) != "PASS":
            errors.append("required preflight check %s status=%s"
                          % (check_id, seen.get(check_id)))


def _normalize_card(value):
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    return token if token in CARD_TARGETS else None


def _normalize_platform(value):
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    token = {"nvidia": "nv", "cuda": "nv", "rocm": "amd", "hip": "amd"}.get(
        token, token)
    return token if token in ("nv", "amd") else None


def check_target_card(errors, contract):
    """Fail closed on missing/unknown card or a card/platform/arch contradiction."""
    target = contract.get("target") if isinstance(contract, dict) else None
    if not isinstance(target, dict):
        errors.append("contract target missing: target.card/platform/arch are required")
        return None
    raw_card = target.get("card")
    card = _normalize_card(raw_card)
    if raw_card is None:
        errors.append("contract target.card missing (h20|b300|mi308x)")
    elif card is None:
        errors.append("contract target.card invalid: %s" % raw_card)

    raw_platform = target.get("platform")
    platform = _normalize_platform(raw_platform)
    if raw_platform is None:
        errors.append("contract target.platform missing (nv|amd)")
    elif platform is None:
        errors.append("contract target.platform invalid: %s" % raw_platform)

    raw_arch = target.get("arch")
    arch = raw_arch.strip().lower() if isinstance(raw_arch, str) else None
    if raw_arch is None:
        errors.append("contract target.arch missing")
    elif not arch:
        errors.append("contract target.arch invalid: %s" % raw_arch)

    if card is not None:
        expected = CARD_TARGETS[card]
        if platform is not None and platform != expected["platform"]:
            errors.append("card %s requires platform=%s, contract has %s"
                          % (card, expected["platform"], platform))
        if arch is not None and arch not in expected["arches"]:
            errors.append("card %s requires arch=%s, contract has %s"
                          % (card, "|".join(sorted(expected["arches"])), arch))
    return card


def check_filename(errors, trace_path, contract):
    match = SPEC1.match(Path(trace_path).name)
    if not match:
        errors.append("trace filename violates SPEC-1: %s" % Path(trace_path).name)
        return
    parsed = match.groupdict()
    contract_card = check_target_card(errors, contract)
    if contract_card is not None and contract_card != parsed["card"]:
        errors.append("filename card %s contradicts contract target.card %s"
                      % (parsed["card"], contract_card))
    bundle = (contract.get("capture") or {}).get("information_bundle")
    want = BUNDLE_TO_DISK.get(bundle)
    if want is None:
        errors.append("contract capture.information_bundle missing/invalid: %s" % bundle)
    elif parsed["bundle"] != want:
        errors.append("filename bundle %s contradicts contract bundle %s"
                      % (parsed["bundle"], bundle))
    grid = ((contract.get("build_and_launch") or {}).get("grid") or [None])[0]
    if grid is not None and int(parsed["grid"]) != int(grid):
        errors.append("filename grid%s contradicts contract grid %s"
                      % (parsed["grid"], grid))


def check_manifest(errors, skill_root, receipt):
    binding = receipt.get("skill_manifest") or {}
    want = binding.get("root_sha256")
    if not want:
        errors.append("receipt has no skill_manifest binding "
                      "(receipt must bind the skill tree)")
        return
    manifest_py = Path(skill_root) / "weaveprof/pipeline/validate" / "harness" / "manifest.py"
    if not manifest_py.is_file():
        errors.append("harness manifest tool missing: %s" % manifest_py)
        return
    proc = subprocess.run(
        [sys.executable, "-B", str(manifest_py), "verify",
         "--skill-root", str(skill_root), "--json"],
        capture_output=True, text=True, timeout=120)
    if proc.returncode not in (0, 1, 3):
        errors.append("manifest verify crashed (rc=%d): %s"
                      % (proc.returncode, proc.stderr.strip()[:300]))
        return
    if proc.returncode == 3:
        errors.append("skill manifest anchor unavailable: %s" % proc.stdout.strip())
        return
    if proc.returncode != 0:
        errors.append("skill tree differs from committed manifest: %s"
                      % proc.stdout.strip()[:600])
        return
    try:
        got = json.loads(proc.stdout).get("root_sha256")
    except ValueError:
        errors.append("manifest verify output unparseable: %s" % proc.stdout[:200])
        return
    if got != want:
        errors.append("skill manifest drifted since delivery: receipt %s != live %s"
                      % (want, got))


def verify(receipt_path, skill_root):
    errors = []
    receipt = load_json(receipt_path, "receipt", errors)
    if receipt is None:
        return errors
    if receipt.get("status") != "PASS":
        errors.append("receipt status=%s" % receipt.get("status"))
    g25 = receipt.get("g25_finalization") or {}
    if g25.get("gate_id") != "G25" or g25.get("status") != "PASS":
        errors.append("receipt missing PASS G25 finalization block")
    if g25.get("trace_sha256") and g25.get("trace_sha256") != receipt.get("trace_sha256"):
        errors.append("G25 trace hash does not match receipt trace hash")

    check_hash(errors, "trace", receipt.get("trace_path"), receipt.get("trace_sha256"))
    check_hash(errors, "contract", receipt.get("contract_path"),
               receipt.get("contract_sha256"))
    check_hash(errors, "all_gates", receipt.get("all_gates_path"),
               receipt.get("all_gates_sha256"))
    check_hash(errors, "preflight", receipt.get("preflight_path"),
               receipt.get("preflight_sha256"))
    check_hash(errors, "trace_processor", receipt.get("trace_processor_path"),
               receipt.get("trace_processor_sha256"))
    for name, record in (receipt.get("delivery_inputs") or {}).items():
        check_hash(errors, "delivery_input[%s]" % name,
                   record.get("path"), record.get("sha256"))

    if receipt.get("all_gates_path") and Path(receipt["all_gates_path"]).is_file():
        check_gates(errors, receipt["all_gates_path"], receipt)
    if receipt.get("preflight_path") and Path(receipt["preflight_path"]).is_file():
        check_preflight(errors, receipt["preflight_path"])

    contract = None
    if receipt.get("contract_path") and Path(receipt["contract_path"]).is_file():
        contract = load_json(receipt["contract_path"], "contract", errors) or {}
    if receipt.get("trace_path"):
        contract = contract or {}
        bundle = (contract.get("capture") or {}).get("information_bundle")
        if bundle and receipt.get("information_bundle") != bundle:
            errors.append("receipt bundle %s contradicts contract bundle %s"
                          % (receipt.get("information_bundle"), bundle))
        check_filename(errors, receipt["trace_path"], contract)

    check_manifest(errors, skill_root, receipt)
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--receipt", help="path to *.delivery_receipt.json")
    source.add_argument("--trace", help="path to trace__*.perfetto.json.gz")
    parser.add_argument(
        "--skill-root",
        default=os.environ.get("FI_SKILL_ROOT")
        or str(Path(__file__).resolve().parents[4]))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.trace:
        name = Path(args.trace).name
        if name.startswith(EXPLORATORY_PREFIX):
            print("NOT_A_DELIVERY (exploratory namespace): %s" % name)
            return 4
        if not SPEC1.match(name):
            print("FAIL: filename matches neither SPEC-1 nor exploratory__: %s" % name)
            return 1
        receipt_path = str(args.trace) + ".delivery_receipt.json"
    else:
        receipt_path = args.receipt
    if not Path(receipt_path).is_file():
        print("FAIL: no receipt at %s -- this trace never went through "
              "run_measurement.py + finalize_delivery.py" % receipt_path)
        return 1

    errors = verify(receipt_path, Path(args.skill_root).resolve())
    if not errors:
        try:
            from hooklib import issue_permits
            issue_permits(receipt_path, load_json(receipt_path, "receipt", []) or {})
        except Exception:
            pass  # permit issuance failure never turns a PASS into a FAIL
    if args.json:
        print(json.dumps({"status": "FAIL" if errors else "PASS",
                          "receipt": receipt_path, "errors": errors},
                         ensure_ascii=False, indent=2, sort_keys=True))
    else:
        if errors:
            print("VERIFY=FAIL receipt=%s" % receipt_path)
            for error in errors:
                print("  - %s" % error)
        else:
            print("VERIFY=PASS receipt=%s" % receipt_path)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
