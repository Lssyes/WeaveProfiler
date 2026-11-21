#!/usr/bin/env python3
"""Hermetic selftest for the weaveprofiler harness.

Builds a miniature skill tree + git anchor + a fully consistent synthetic
delivery, then asserts the verifier accepts it and rejects every tampered
variant.  Also exercises the hook scripts end to end over their stdin/stdout
wire protocol.  No GPU, no network, no shared state.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
VERIFY = HARNESS / "verify_receipt.py"
MANIFEST = HARNESS / "manifest.py"
PRETOOL = HARNESS / "hooks" / "fi_pretool_guard.py"
STOPGATE = HARNESS / "hooks" / "fi_stop_gate.py"
MARKER = HARNESS / "hooks" / "fi_session_marker.py"
V2_INCIDENT_RAW = os.environ.get("WEAVEPROFILER_V2_INCIDENT_TRACE")
V2_INCIDENT = Path(V2_INCIDENT_RAW) if V2_INCIDENT_RAW else None

FAILURES = []


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def run(script, args=(), stdin=None, cwd=None):
    return subprocess.run([sys.executable, "-B", str(script)] + list(args),
                          input=stdin, capture_output=True, text=True,
                          cwd=cwd, timeout=300)


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print("[%s] %s%s" % (status, name, (" -- " + detail) if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def build_fixture(work):
    skill = work / "fake_skill"
    (skill / "weaveprof/pipeline/validate" / "harness").mkdir(parents=True)
    (skill / "gate.py").write_text("# gate\n")
    (skill / "weaveprof/pipeline/validate" / "harness" / "tool.py").write_text("# tool\n")
    shutil.copy2(str(MANIFEST), str(skill / "weaveprof/pipeline/validate" / "harness" / "manifest.py"))
    subprocess.run(["git", "init", "-q"], cwd=str(skill), check=True)
    run(MANIFEST, ["build", "--skill-root", str(skill)])
    subprocess.run(["git", "add", "-A"], cwd=str(skill), check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "init"], cwd=str(skill), check=True)

    run_dir = work / "run"
    run_dir.mkdir()
    trace = run_dir / ("trace__demo_moe__card-h20__bs1_sq8__fp8e4m3__grid78_warp12_ph0__sw"
                       ".perfetto.json.gz")
    trace.write_bytes(b'{"traceEvents": []}')
    contract = run_dir / "CONTRACT.lock.json"
    contract.write_text(json.dumps({
        "target": {"card": "h20", "platform": "nv", "arch": "sm_90"},
        "capture": {"information_bundle": "sw"},
        "build_and_launch": {"grid": [78, 1, 1]}}))
    preflight = run_dir / "PREFLIGHT.json"
    preflight.write_text(json.dumps({
        "required_check_ids": ["E%02d" % i for i in range(1, 28)],
        "not_applicable_check_ids": [],
        "checks": [{"check_id": "E%02d" % i, "status": "PASS"}
                   for i in range(1, 28)],
        "summary": {"all_required_pass": True}}))
    na = ["G11", "G19", "G20", "G21"]
    required = ["G%02d" % i for i in range(25) if "G%02d" % i not in na]
    all_gates = run_dir / "ALL_GATES.json"
    all_gates.write_text(json.dumps({
        "schema_version": "all_gates.v1", "status": "PASS",
        "contract_sha256": sha256(contract), "trace_sha256": sha256(trace),
        "required_gate_ids": required, "not_applicable_gate_ids": na,
        "input_errors": [], "integrity_errors": [], "partition_errors": [],
        "mandatory_audit_failures": [],
        "gates": ([{"gate_id": g, "status": "PASS"} for g in required]
                  + [{"gate_id": g, "status": "N/A"} for g in na]),
        "summary": {"all_required_pass": True}}))
    manifest_report = run(MANIFEST, ["verify", "--skill-root", str(skill), "--anchor", "tree", "--json"])
    root_sha = json.loads(manifest_report.stdout)["root_sha256"]
    receipt = Path(str(trace) + ".delivery_receipt.json")
    receipt.write_text(json.dumps({
        "status": "PASS",
        "trace_path": str(trace), "trace_sha256": sha256(trace),
        "contract_path": str(contract), "contract_sha256": sha256(contract),
        "all_gates_path": str(all_gates), "all_gates_sha256": sha256(all_gates),
        "preflight_path": str(preflight), "preflight_sha256": sha256(preflight),
        "trace_processor_path": "/bin/true", "trace_processor_sha256": sha256("/bin/true"),
        "information_bundle": "sw", "delivery_inputs": {},
        "g25_finalization": {"gate_id": "G25", "status": "PASS",
                             "trace_path": str(trace),
                             "trace_sha256": sha256(trace)},
        "skill_manifest": {"skill_root": str(skill), "anchor": "tree",
                           "root_sha256": root_sha}}))
    return skill, trace, receipt


def mutate_json(path, fn):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    fn(doc)
    Path(path).write_text(json.dumps(doc), encoding="utf-8")


def main():
    work = Path(tempfile.mkdtemp(prefix="fi_harness_selftest_",
                                 dir=str(HARNESS / "testdata")))
    state = work / "state"
    state.mkdir()
    os.environ["FI_STATE_DIR"] = str(state)
    try:
        skill, trace, receipt = build_fixture(work)

        proc = run(VERIFY, ["--trace", str(trace), "--skill-root", str(skill)])
        check("good fixture verifies PASS", proc.returncode == 0, proc.stdout[-300:])

        legacy_trace = trace.with_name(trace.name.replace("__card-h20", ""))
        proc = run(VERIFY, ["--trace", str(legacy_trace), "--skill-root", str(skill)])
        check("legacy no-card trace filename FAILs",
              proc.returncode == 1 and "SPEC-1" in proc.stdout, proc.stdout[:300])

        mismatch_trace = trace.with_name(trace.name.replace("card-h20", "card-mi308x"))
        mismatch_trace.write_bytes(trace.read_bytes())
        mismatch_receipt = Path(str(mismatch_trace) + ".delivery_receipt.json")
        mismatch_doc = json.loads(receipt.read_text())
        mismatch_doc["trace_path"] = str(mismatch_trace)
        mismatch_doc["g25_finalization"]["trace_path"] = str(mismatch_trace)
        mismatch_receipt.write_text(json.dumps(mismatch_doc))
        proc = run(VERIFY, ["--trace", str(mismatch_trace), "--skill-root", str(skill)])
        check("filename card vs contract target.card mismatch FAILs",
              proc.returncode == 1 and "target.card" in proc.stdout,
              proc.stdout[:500])
        mismatch_trace.unlink()
        mismatch_receipt.unlink()

        original = trace.read_bytes()
        trace.write_bytes(original + b"tampered")
        proc = run(VERIFY, ["--trace", str(trace), "--skill-root", str(skill)])
        check("tampered trace FAILs", proc.returncode == 1)
        trace.write_bytes(original)

        mutate_json(receipt, lambda d: d.pop("skill_manifest"))
        proc = run(VERIFY, ["--trace", str(trace), "--skill-root", str(skill)])
        check("receipt without skill_manifest FAILs",
              proc.returncode == 1 and "skill_manifest" in proc.stdout)
        mutate_json(receipt, lambda d: d.__setitem__("skill_manifest",
                 {"root_sha256": json.loads(run(MANIFEST, ["verify", "--skill-root",
                  str(skill), "--anchor", "tree", "--json"]).stdout)["root_sha256"]}))

        gates_path = json.loads(receipt.read_text())["all_gates_path"]
        mutate_json(gates_path, lambda d: d["gates"][0].__setitem__("status", "FAIL"))
        proc = run(VERIFY, ["--trace", str(trace), "--skill-root", str(skill)])
        check("gate FAIL inside ALL_GATES FAILs verification", proc.returncode == 1)
        mutate_json(gates_path, lambda d: d["gates"][0].__setitem__("status", "PASS"))
        mutate_json(receipt, lambda d: d.__setitem__(
            "all_gates_sha256", sha256(gates_path)))

        (skill / "gate.py").write_text("# gate tampered\n")
        proc = run(VERIFY, ["--trace", str(trace), "--skill-root", str(skill)])
        check("skill tree drift FAILs verification",
              proc.returncode == 1 and "differs from committed manifest" in proc.stdout)
        (skill / "gate.py").write_text("# gate\n")

        proc = run(VERIFY, ["--trace", str(trace).replace("trace__", "exploratory__")])
        check("nonexistent exploratory name -> NOT_A_DELIVERY", proc.returncode == 4)

        if V2_INCIDENT is not None and V2_INCIDENT.is_file():
            proc = run(VERIFY, ["--trace", str(V2_INCIDENT)])
            check("v2 incident artifact FAILs (legacy no-card name)",
                  proc.returncode == 1 and "SPEC-1" in proc.stdout,
                  proc.stdout[:300])

        payload = {"hook_event_name": "PreToolUse", "session_id": "t1",
                   "cwd": str(work), "tool_name": "Bash",
                   "tool_input": {"command": "echo hello"}}
        proc = run(PRETOOL, stdin=json.dumps(payload))
        check("guard allows plain command", proc.returncode == 0 and not proc.stdout)

        unverified = work / "unverified_upload"
        unverified.mkdir()
        unverified_trace = unverified / trace.name
        # Deliberately different bytes: permits are content-hash bound, so a copy
        # of the already verified trace must not masquerade as verified evidence.
        unverified_trace.write_bytes(b"unverified measurement bytes")
        payload["tool_input"]["command"] = ("ossutil cp %s oss://bucket/x/"
                                            % unverified_trace)
        proc = run(PRETOOL, stdin=json.dumps(payload))
        decision = json.loads(proc.stdout) if proc.stdout else {}
        check("guard denies unverified OSS upload",
              "deny" in json.dumps(decision), proc.stdout[:200])
        unverified_trace.unlink()
        unverified.rmdir()

        payload["tool_input"]["command"] = ("cp a.gz "
                                            "trace__x_y__card-h20__bs1_sq8__fp8__grid1_warp1_ph0__sw.perfetto.json.gz")
        proc = run(PRETOOL, stdin=json.dumps(payload))
        check("guard denies renaming into SPEC-1 namespace",
              "deny" in (proc.stdout or ""))

        payload["tool_input"]["command"] = "cp a.gz exploratory__x.perfetto.json.gz"
        proc = run(PRETOOL, stdin=json.dumps(payload))
        check("guard allows exploratory__ namespace",
              proc.returncode == 0 and not proc.stdout)

        payload["tool_input"]["command"] = "echo x > ALL_GATES.json"
        proc = run(PRETOOL, stdin=json.dumps(payload))
        check("guard denies evidence overwrite", "deny" in (proc.stdout or ""))

        # Regression cases from the live probe: python-inline writers and
        # heredoc patches must not slip past R2; reads and remediation must pass.
        payload["tool_input"]["command"] = (
            "python3 -c \"open('trace__x_y__card-h20__bs1_sq8__fp8__grid1_warp1_ph0__sw"
            ".perfetto.json.gz','w').write('x')\"")
        proc = run(PRETOOL, stdin=json.dumps(payload))
        check("guard denies python-inline SPEC-1 writer", "deny" in (proc.stdout or ""))

        payload["tool_input"]["command"] = (
            "cat trace__x_y__card-h20__bs1_sq8__fp8__grid1_warp1_ph0__sw.perfetto.json.gz | zcat | head")
        proc = run(PRETOOL, stdin=json.dumps(payload))
        check("guard allows read-only inspection", not proc.stdout)

        payload["tool_input"]["command"] = (
            "mv trace__x_y__card-h20__bs1_sq8__fp8__grid1_warp1_ph0__sw.perfetto.json.gz "
            "exploratory__x.perfetto.json.gz")
        proc = run(PRETOOL, stdin=json.dumps(payload))
        check("guard allows rename OUT to exploratory__", not proc.stdout)

        payload["tool_input"]["command"] = (
            "rm trace__x_y__card-h20__bs1_sq8__fp8__grid1_warp1_ph0__sw.perfetto.json.gz")
        proc = run(PRETOOL, stdin=json.dumps(payload))
        check("guard allows deleting fake artifacts", not proc.stdout)

        os.environ["FI_SKILL_ROOT"] = str(skill)
        marker_payload = {"hook_event_name": "SessionStart", "session_id": "t-stop",
                          "cwd": str(work)}
        proc = run(MARKER, stdin=json.dumps(marker_payload))
        check("session marker hook exits 0 and writes marker",
              proc.returncode == 0
              and (state / "session_t-stop.json").is_file(),
              proc.stderr[:200])
        stop_payload = {"hook_event_name": "Stop", "session_id": "t-stop",
                        "stop_hook_active": False}
        rogue = work / "rogue"
        rogue.mkdir()
        (rogue / ("trace__r_y__card-h20__bs1_sq8__fp8__grid1_warp1_ph0__sw.perfetto.json.gz")
         ).write_bytes(b"rogue")
        config = json.loads((HARNESS / "config.json").read_text())
        config["scan_roots"] = [str(work)]
        config_path = HARNESS / "config.json"
        original_config = config_path.read_text()
        config_path.write_text(json.dumps(config))
        try:
            proc = run(STOPGATE, stdin=json.dumps(stop_payload))
            check("stop gate blocks unverified delivery",
                  '"decision": "block"' in (proc.stdout or ""), proc.stdout[:200])
            receipt_path = json.loads(receipt.read_text())
            shutil.copy2(str(trace), str(rogue))  # verified trace must pass
            shutil.copy2(str(receipt), str(rogue) + "/" + receipt.name)
            (rogue / ("trace__r_y__card-h20__bs1_sq8__fp8__grid1_warp1_ph0__sw.perfetto.json.gz")
             ).unlink()
            proc = run(STOPGATE, stdin=json.dumps(stop_payload))
            check("stop gate passes verified delivery",
                  proc.returncode == 0 and not proc.stdout, proc.stdout[:1200])
        finally:
            config_path.write_text(original_config)
        # ---- armed-session model -------------------------------------
        arm_payload = {"hook_event_name": "PreToolUse", "session_id": "t-armed",
                       "cwd": str(work),
                       "tool_name": "Bash",
                       "tool_input": {"command": "python3 -B run_measurement.py "
                                      "--contract x/CONTRACT.lock.json"}}
        proc = run(PRETOOL, stdin=json.dumps(arm_payload))
        check("orchestrator command allowed and arms the session",
              proc.returncode == 0 and not proc.stdout
              and (state / "armed_t-armed.json").is_file(), proc.stdout[:200])

        arm_payload["tool_input"]["command"] = (
            "python3 -c \"open('foo.perfetto.json.gz','w').write('x')\"")
        proc = run(PRETOOL, stdin=json.dumps(arm_payload))
        check("armed: artifact write outside orchestrator denied",
              "deny" in (proc.stdout or ""))

        arm_payload["tool_input"]["command"] = (
            "python3 -c \"open('exploratory__foo.perfetto.json.gz','w').write('x')\"")
        proc = run(PRETOOL, stdin=json.dumps(arm_payload))
        check("armed: exploratory__ artifact write allowed", not proc.stdout)

        work2 = work / "armed_run"
        work2.mkdir()
        loose = work2 / "foo.perfetto.json.gz"
        loose.write_bytes(b"manual capture")
        arm_payload["tool_input"]["command"] = (
            "ossutil cp %s oss://bucket/x/" % loose)
        proc = run(PRETOOL, stdin=json.dumps(arm_payload))
        check("R1: unverifiable measurement file upload denied (any name)",
              "deny" in (proc.stdout or ""))

        # permit issued by a PASS verification unblocks the upload
        run(VERIFY, ["--trace", str(trace), "--skill-root", str(skill)])
        up_payload = {"hook_event_name": "PreToolUse", "session_id": "t-up",
                      "cwd": str(work), "tool_name": "Bash",
                      "tool_input": {"command": "ossutil cp %s oss://bucket/x/"
                                     % trace}}
        proc = run(PRETOOL, stdin=json.dumps(up_payload))
        check("R1: permit from PASS receipt allows upload", not proc.stdout,
              proc.stdout[:200])

        # armed stop: empty exit blocked, BLOCKED evidence accepted
        run(MARKER, stdin=json.dumps({"hook_event_name": "SessionStart",
                                      "session_id": "t-armed",
                                      "cwd": str(work2)}))
        stop2 = {"hook_event_name": "Stop", "session_id": "t-armed",
                 "stop_hook_active": False}
        proc = run(STOPGATE, stdin=json.dumps(stop2))
        check("armed stop: no receipt and no BLOCKED evidence -> block",
              '"decision": "block"' in (proc.stdout or ""), proc.stdout[:300])
        # remediation: loose artifact renounced to exploratory__ first
        loose.rename(work2 / "exploratory__foo.perfetto.json.gz")
        (work2 / "PREFLIGHT.json").write_text(json.dumps(
            {"status": "FAIL", "summary": {"all_required_pass": False}}))
        proc = run(STOPGATE, stdin=json.dumps(stop2))
        check("armed stop: honest BLOCKED evidence allows exit",
              proc.returncode == 0 and not proc.stdout, proc.stdout[:300])

    finally:
        shutil.rmtree(str(work), ignore_errors=True)

    print("SELFTEST=%s (%d failures)" % ("FAIL" if FAILURES else "PASS", len(FAILURES)))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
