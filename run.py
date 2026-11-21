#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""weaveprofiler command entry.

The public workflow is:
    scan -> instrument -> capture -> validate

Fast development falls back to ``fast``; the complete receipt path is ``deliver``.
Every internal tool remains callable directly, but this file gives all supported
cards one stable command surface.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
CARDS = ("h20", "b300", "mi308x")


def _load(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_python(relative: str, argv: Sequence[str], *, cwd: Optional[Path] = None) -> int:
    cmd = [sys.executable, "-B", str(ROOT / relative), *argv]
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _scan(argv: Sequence[str]) -> int:
    return _load("weaveprof/pipeline/scan/scan.py", "weaveprofiler_scan").main(list(argv))


def _instrument(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Generate and optionally apply SHA-bound instrumentation edits")
    parser.add_argument("--scan", help="scan JSON from `run.py scan`")
    parser.add_argument("--selection", help="explicit event selection JSON")
    parser.add_argument("--plan", help="patch-plan path to write, or an existing plan with --apply/check")
    parser.add_argument("--apply", action="store_true", help="apply the validated patch plan to source files")
    parser.add_argument("--dry-run", action="store_true", help="validate only; never modify source")
    parser.add_argument("--diff-output", help="write the validated unified diff here")
    args = parser.parse_args(list(argv))

    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run are mutually exclusive")
    plan = Path(args.plan).expanduser().resolve() if args.plan else None
    if args.scan or args.selection:
        if not (args.scan and args.selection):
            parser.error("generation mode requires both --scan and --selection")
        if plan is None:
            parser.error("generation mode requires --plan PATH")
        gen_argv = ["--scan", str(Path(args.scan).expanduser().resolve()),
                    "--selection", str(Path(args.selection).expanduser().resolve()),
                    "--out", str(plan)]
        rc = _run_python("weaveprof/pipeline/instrument/generate_edits.py", gen_argv)
        if rc != 0:
            return rc
    elif plan is None:
        parser.error("provide --scan/--selection to generate, or --plan to inspect/apply an existing plan")

    if not plan.is_file():
        _json_print({"ok": False, "error": {"code": "plan_missing", "path": str(plan)}})
        return 2
    apply_argv: List[str] = [str(plan)]
    if not args.apply:
        apply_argv.append("--check")
    if args.diff_output:
        apply_argv.extend(["--diff-output", str(Path(args.diff_output).expanduser().resolve())])
    rc = _run_python("weaveprof/pipeline/instrument/apply_plan.py", apply_argv)
    if rc != 0:
        return rc
    _json_print({
        "ok": True,
        "plan": str(plan),
        "names": str(Path(str(plan) + ".names.json")),
        "applied": bool(args.apply),
        "dry_run": not bool(args.apply),
    })
    return 0


def _parse_deadline(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        value = dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("--deadline must be ISO 8601, for example 2026-11-14T18:30:00Z: {}".format(exc))
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.timestamp()


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                       encoding="utf-8")
    os.replace(str(partial), str(path))


def _kill_group(proc: subprocess.Popen) -> None:
    if hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            proc.terminate()
    else:
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if hasattr(os, "killpg"):
            try:
                os.killpg(proc.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
        elif os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=10)
        else:
            proc.kill()
        proc.wait()


def _run_observed(cmd: Sequence[str], *, cwd: Path, deadline: Optional[float],
                  heartbeat: Optional[Path], card: str, phase: str) -> int:
    if heartbeat:
        _atomic_json(heartbeat, {"schema": "weaveprofiler-heartbeat/v1", "status": "STARTING",
                                 "phase": phase, "card": card, "pid": None,
                                 "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    proc = subprocess.Popen(list(cmd), cwd=str(cwd), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)
    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    thread = threading.Thread(target=reader, name="weaveprofiler-output", daemon=True)
    thread.start()
    started = time.monotonic()
    status = "RUNNING"
    rc: Optional[int] = None
    while True:
        rc = proc.poll()
        now = time.monotonic()
        if heartbeat:
            _atomic_json(heartbeat, {"schema": "weaveprofiler-heartbeat/v1", "status": status,
                                     "phase": phase, "card": card, "pid": proc.pid,
                                     "elapsed_seconds": round(now - started, 3),
                                     "deadline_epoch": deadline,
                                     "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()})
        if rc is not None:
            break
        if deadline is not None and time.time() >= deadline:
            status = "DEADLINE_EXCEEDED"
            _kill_group(proc)
            rc = 124
            break
        time.sleep(0.5)
    thread.join(timeout=2)
    if heartbeat:
        final_status = ("PASS" if rc == 0 else
                        "DEADLINE_EXCEEDED" if status == "DEADLINE_EXCEEDED" else
                        "FAIL")
        _atomic_json(heartbeat, {"schema": "weaveprofiler-heartbeat/v1",
                                 "status": final_status,
                                 "phase": phase, "card": card, "pid": proc.pid,
                                 "exit_code": rc, "elapsed_seconds": round(time.monotonic() - started, 3),
                                 "deadline_epoch": deadline,
                                 "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    return int(rc)


def _capture(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Route one capture to nsys or rocprof-sys/rocprofv3",
        epilog="Arguments after --payload are passed verbatim to the selected capture tool.")
    parser.add_argument("--card", required=True, choices=CARDS)
    parser.add_argument("--arch")
    parser.add_argument("--platform", choices=("nv", "nvidia", "amd", "rocm"))
    parser.add_argument("--project", "--root", dest="root")
    parser.add_argument("--out", "--out-dir", dest="out_dir")
    parser.add_argument("--collector", choices=("auto", "nsys", "rocprof-sys", "rocprofv3"))
    parser.add_argument("--deadline", help="absolute ISO 8601 UTC deadline")
    parser.add_argument("--heartbeat", help="heartbeat JSON path")
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv))
    if args.card == "b300" and not args.arch:
        parser.error("B300 requires --arch sm_100a or sm_103a")
    try:
        from weaveprof.pipeline.core.registry import resolve_target, UnsupportedTarget
    except Exception as exc:
        _json_print({"ok": False, "error": "cannot import registry: {}".format(exc)})
        return 2
    try:
        route = resolve_target(card=args.card, arch=args.arch, platform=args.platform)
    except UnsupportedTarget as exc:
        _json_print(exc.as_dict())
        return 2
    platform = str(route.get("platform"))
    root = Path(args.root).expanduser().resolve() if args.root else None
    if root is None:
        parser.error("--project/--root is required")
    if not root.is_dir():
        parser.error("--project/--root is not a directory: {}".format(root))
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if out_dir is not None and (out_dir == root or root not in out_dir.parents):
        parser.error("--out/--out-dir must be inside --project/--root and must not equal it")
    try:
        deadline = _parse_deadline(args.deadline)
    except ValueError as exc:
        parser.error(str(exc))
    rest = list(args.rest)
    if rest and rest[0] == "--":
        rest = rest[1:]

    if platform == "nv":
        if args.collector not in (None, "auto", "nsys"):
            parser.error("H20/B300 use nsys; collector={!r} is invalid".format(args.collector))
        script = ROOT / "weaveprof/pipeline/capture" / "capture_nsys.py"
        tool_argv: List[str] = ["--root", str(root)]
        if out_dir is not None:
            tool_argv.extend(["--out-dir", str(out_dir)])
        tool_argv.extend(rest)
    else:
        if args.collector == "nsys":
            parser.error("MI308X uses rocprof-sys with rocprofv3 fallback; nsys is invalid")
        script = ROOT / "weaveprof/pipeline/capture" / "capture_rocm.py"
        tool_argv = ["--root", str(root)]
        if out_dir is not None:
            tool_argv.extend(["--out-dir", str(out_dir)])
        if args.collector and args.collector != "auto":
            tool_argv.extend(["--collector", args.collector])
        tool_argv.extend(rest)
    heartbeat = Path(args.heartbeat).expanduser().resolve() if args.heartbeat else None
    return _run_observed([sys.executable, "-B", str(script), *tool_argv], cwd=root,
                         deadline=deadline, heartbeat=heartbeat, card=args.card,
                         phase="capture")


def _find_single(pattern: str, run_dir: Path) -> Optional[Path]:
    found = sorted(path for path in run_dir.rglob(pattern) if path.is_file())
    return found[0] if len(found) == 1 else None


def _validate(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Run all contract-declared validation gates")
    parser.add_argument("--contract")
    parser.add_argument("--run-dir")
    parser.add_argument("--trace")
    parser.add_argument("--out")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(list(argv))
    if args.selftest:
        return _run_python("weaveprof/pipeline/validate/validate_all.py", ["--selftest"])
    if not args.run_dir:
        parser.error("--run-dir is required")
    run_dir = Path(args.run_dir).expanduser().resolve()
    contract = Path(args.contract).expanduser().resolve() if args.contract else run_dir / "CONTRACT.lock.json"
    trace = Path(args.trace).expanduser().resolve() if args.trace else (
        _find_single("*.perfetto.json.gz", run_dir) or _find_single("*.pftrace", run_dir))
    out = Path(args.out).expanduser().resolve() if args.out else run_dir / "validation" / "validate_all.json"
    missing = [str(path) for path, label in ((contract, "contract"), (trace, "trace"))
               if path is None or not path.is_file()]
    if missing or trace is None:
        _json_print({"ok": False, "error": "missing or ambiguous inputs",
                     "missing": missing, "run_dir": str(run_dir)})
        return 2
    return _run_python("weaveprof/pipeline/validate/validate_all.py",
                       ["--contract", str(contract), "--run-dir", str(run_dir),
                        "--trace", str(trace), "--out", str(out)])


def _check(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Quick run check; never issues a delivery receipt")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--trace")
    parser.add_argument("--json")
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv))
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        parser.error("--run-dir is not a directory: {}".format(run_dir))
    trace = Path(args.trace).expanduser().resolve() if args.trace else _find_single("*.perfetto.json.gz", run_dir)
    if trace is None or not trace.is_file():
        _json_print({"ok": False, "error": "trace is required or must identify one file",
                     "run_dir": str(run_dir)})
        return 2
    out = Path(args.json).expanduser().resolve() if args.json else run_dir / "validation" / "check.json"
    rest = list(args.rest)
    if rest and rest[0] == "--":
        rest = rest[1:]
    return _run_python("weaveprof/pipeline/validate/validate_run.py",
                       [str(run_dir), str(trace), *rest, "--json", str(out)])


def _deliver(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/pipeline/core/run_measurement.py", list(argv))


def _fast(argv: Sequence[str]) -> int:
    # weaveprof/pipeline/core/fast_path.py owns a ``fast`` subcommand; run.py already dispatched that
    # token.  Reinsert it so the public ``run.py fast --job ...`` entry is
    # identical to the standalone CLI instead of forwarding a bare --job.
    return _load("weaveprof/pipeline/core/fast_path.py", "weaveprofiler_fast_path").main(
        ["fast", *list(argv)])


def _gemm_probe(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/pipeline/capture/gemm_probe.py", argv)


def _auto_gemm(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/pipeline/capture/auto_gemm.py", argv)


def _probe_place(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/pipeline/scan/llm_probe_placer.py", argv)


def _probe_reuse(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/pipeline/instrument/incremental_reuse.py", argv)


def _probe_inject(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/pipeline/instrument/rule_engine.py", argv)


def _auto_profile(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/pipeline/validate/closed_loop_verifier.py", argv)


def _install_agent(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/agent_installer.py", ["install", *list(argv)])


def _guard_exec(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/agent_installer.py", ["guard-exec", *list(argv)])


def _agent_status(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/agent_installer.py", ["status", *list(argv)])


def _mcp_server(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/mcp_server.py", argv)


def _template(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Print a command template")
    parser.add_argument("--mode", choices=("fast", "scan", "instrument", "capture", "validate", "check", "deliver",
                                            "gemm-probe", "auto-gemm", "probe-place", "probe-reuse", "probe-inject",
                                            "auto-profile", "install-agent", "guard-exec"),
                        default="fast")
    args = parser.parse_args(list(argv))
    if args.mode == "fast":
        _json_print(_load("weaveprof/pipeline/core/fast_path.py", "weaveprofiler_fast_path").template())
        return 0
    python_example = "/opt/venv/python3.14.4/bin/python3"
    examples = {
        "install-agent": {"argv": ["python3", "-B", "run.py", "install-agent", "--workspace", "KERNEL_DIR",
                                    "--protect", "kernel.cu", "--card", "h20"]},
        "guard-exec": {"argv": ["python3", "-B", "run.py", "guard-exec", "--workspace", "KERNEL_DIR",
                                 "--", "codex", "exec", "optimize the protected kernel with WeaveProfiler"]},
        "auto-profile": {"argv": ["python3", "-B", "run.py", "auto-profile", "--source", "gemm.cu",
                                   "--card", "h20", "--arch", "sm_90a", "--model", "qwen4-max", "--out-dir", "AUTO_PROFILE"]},
        "probe-place": {"argv": ["python3", "-B", "run.py", "probe-place", "--source", "gemm.cu",
                                  "--card", "h20", "--arch", "sm_90a", "--model", "qwen4-max", "--out", "probe_spec.json"]},
        "probe-reuse": {"argv": ["python3", "-B", "run.py", "probe-reuse", "--prev-source", "v0.cu",
                                  "--prev-spec", "probe_spec.json", "--new-source", "v1.cu", "--out-spec", "probe_spec_v1.json"]},
        "probe-inject": {"argv": ["python3", "-B", "run.py", "probe-inject", "--source", "gemm.cu",
                                   "--spec", "probe_spec.json", "--out-source", "inst_gemm.cu"]},
        "auto-gemm": {"argv": ["python3", "-B", "run.py", "auto-gemm", "--source", "gemm.cu",
                                "--card", "h20", "--arch", "sm_90a", "--phase", "prep", "--out-dir", "AUTO_RUN"]},
        "gemm-probe": {"argv": [python_example, "-B", "run.py", "gemm-probe", "--card", "h20",
                                 "--arch", "sm_90", "--profile", "full",
                                 "--gpu", "auto", "--out", "RUN", "--deadline-sec", "1800"],
                        "quick": [python_example, "-B", "run.py", "gemm-probe", "--card", "h20",
                                  "--arch", "sm_90", "--profile", "quick",
                                  "--gpu", "auto", "--out", "QUICK_RUN", "--deadline-sec", "300"]},
        "scan": {"argv": ["python3", "-B", "run.py", "scan", "--project", "P", "--card", "h20",
                           "--arch", "sm_90", "--language", "cuda_cpp", "--source", "F", "--out", "SCAN.json"]},
        "instrument": {"argv": ["python3", "-B", "run.py", "instrument", "--scan", "SCAN.json",
                                 "--selection", "SELECTION.json", "--plan", "EDITS.json"],
                       "apply": ["python3", "-B", "run.py", "instrument", "--plan", "EDITS.json", "--apply"]},
        "capture": {"argv": ["python3", "-B", "run.py", "capture", "--card", "h20", "--arch", "sm_90",
                              "--project", "P", "--out", "RUN", "--deadline", "2026-11-14T20:00:00Z",
                              "--heartbeat", "RUN/heartbeat.json", "--payload", "python3", "app.py"]},
        "validate": {"argv": ["python3", "-B", "run.py", "validate", "--run-dir", "RUN"]},
        "check": {"argv": ["python3", "-B", "run.py", "check", "--run-dir", "RUN"]},
        "deliver": {"argv": ["python3", "-B", "run.py", "deliver", "--contract", "RUN/CONTRACT.lock.json",
                              "--run-dir", "RUN", "--trace-processor", "TRACE_PROCESSOR"]},
    }
    _json_print({"schema": "weaveprofiler-template/v1", "mode": args.mode, **examples[args.mode]})
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        return 0
    command, rest = args[0], args[1:]
    handlers = {"scan": _scan, "instrument": _instrument, "capture": _capture,
                "validate": _validate, "check": _check, "fast": _fast,
                "deliver": _deliver, "gemm-probe": _gemm_probe,
                "auto-gemm": _auto_gemm,
                "probe-place": _probe_place, "probe-reuse": _probe_reuse,
                "probe-inject": _probe_inject, "auto-profile": _auto_profile,
                "install-agent": _install_agent, "guard-exec": _guard_exec,
                "agent-status": _agent_status, "mcp-server": _mcp_server,
                "template": _template}
    handler = handlers.get(command)
    if handler is None:
        print("unknown command {!r}; use {}".format(command, ", ".join(handlers)), file=sys.stderr)
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
