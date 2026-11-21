"""Capture one unmodified Triton launch and measure its two shadow builds on NVIDIA."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import importlib.util
import math
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import threading
from unittest.mock import patch

from .backend import run_cmd
from .workspace import read_json, write_json


_capture_lock = threading.Lock()


def _arguments(args, kwargs, device):
    """Snapshot whole allocations once, preserving tensor views and pointer aliasing."""
    import torch
    buffers, indices = [], {}

    def encode(value):
        if isinstance(value, torch.Tensor):
            if value.device != device or value.layout != torch.strided:
                raise ValueError("Replay requires strided tensors on the current CUDA device")
            storage = value.untyped_storage()
            key = storage.data_ptr(), storage.nbytes()
            if key not in indices:
                indices[key] = len(buffers)
                buffers.append(torch.empty(0, device=device, dtype=torch.uint8).set_(
                    storage, 0, (storage.nbytes(),), (1,)))
            return dict(storage=indices[key], offset=value.storage_offset(), dtype=value.dtype,
                        size=tuple(value.size()), stride=tuple(value.stride()))
        if value is None or isinstance(value, (int, float, bool, str)):
            return value
        raise TypeError(f"Unsupported Triton launch argument {type(value).__name__}")

    positional = [encode(v) for v in args]
    keywords = {k: encode(v) for k, v in kwargs.items()}
    if not buffers:
        raise ValueError("The target launch has no tensor arguments to verify")
    return positional, keywords, buffers


@contextmanager
def capture(source, workspace, spec, prepared):
    import torch
    from triton.runtime.jit import JITFunction

    if spec.get("language") != "triton" or torch.version.hip:
        raise ValueError("Automatic launch replay currently requires an NVIDIA Triton kernel")
    thread = threading.get_ident()
    original = JITFunction.run
    seen = False
    target = spec.get("kernel_name")

    def intercept(fn, *args, grid, warmup, **kwargs):
        nonlocal seen
        if (thread != threading.get_ident() or warmup or
                Path(fn.fn.__code__.co_filename).resolve() != source or
                (target and fn.fn.__name__ != target)):
            return original(fn, *args, grid=grid, warmup=warmup, **kwargs)
        if seen:
            raise ValueError("The profiling context contains multiple target launches")
        device = torch.device("cuda", torch.cuda.current_device())
        positional, keywords, buffers = _arguments(args, kwargs, device)
        snapshot = dict(kernel=fn.fn.__name__, module=fn.fn.__module__, device=str(device),
                        args=positional, kwargs=keywords, before=[v.cpu() for v in buffers])

        def resolve_grid(meta):
            snapshot["grid"] = tuple(grid(meta))
            return snapshot["grid"]

        if not callable(grid):
            snapshot["grid"] = tuple(grid)
        result = original(fn, *args, grid=resolve_grid if callable(grid) else snapshot["grid"],
                          warmup=False, **kwargs)
        snapshot["num_warps"] = result.metadata.num_warps
        snapshot["after"] = [v.cpu() for v in buffers]
        torch.save(snapshot, workspace / "launch.pt")
        seen = True
        return result

    if not _capture_lock.acquire(blocking=False):
        raise RuntimeError("Another Triton launch capture is active in this process")
    try:
        with patch.object(JITFunction, "run", intercept):
            yield
        if not seen:
            raise ValueError(f"No target Triton launch captured from {source}")
    finally:
        _capture_lock.release()


def measure(workspace, prepared):
    directory, base, inst = prepared
    env = workspace.environment(directory, base, inst)
    env["TRITON_ALWAYS_COMPILE"] = "1"
    worker = [sys.executable, "-B", "-m", "weaveprof.replay"]
    run_cmd([*worker, "measure", str(base), str(inst), str(directory)], directory, env, directory / "replay.log")
    # One dispatched instrumented kernel supplies both the hardware interval and FILITE02.
    env.pop("TRITON_ALWAYS_COMPILE")
    run_cmd(["nsys", "profile", "--sample=none", "--cpuctxsw=none", "--trace=cuda",
             "-o", str(directory / "capture"), *worker, "record", str(base), str(inst), str(directory)],
            directory, env, directory / "nsys.log")
    run_cmd(["nsys", "export", "--type=sqlite", "-o", str(directory / "capture.sqlite"),
             str(directory / "capture.nsys-rep")], directory, env, directory / "nsys_export.log")


def _load_kernel(path, snapshot, instrumented):
    # Preserve the package context for relative imports from the shadow tree.
    name = snapshot["module"]
    package = name.rpartition(".")[0]
    alias = ".".join(filter(None, (package, "_weaveprof_" + path.stem + ("_inst" if instrumented else "_l0"))))
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module, getattr(module, snapshot["kernel"])


def _bytes_file(path, buffers):
    with path.open("wb") as out:
        for buf in buffers:
            buf.cpu().numpy().tofile(out)


def _worker(mode, base, inst, directory):
    import torch
    from triton import knobs

    snapshot = torch.load(directory.parent / "launch.pt", weights_only=True, map_location="cpu")
    torch.cuda.set_device(snapshot["device"])
    initial = [v.to(snapshot["device"]) for v in snapshot["before"]]
    buffers = [v.clone() for v in initial]

    def restore(arg):
        if not isinstance(arg, dict):
            return arg
        tensor = torch.empty(0, dtype=arg["dtype"], device=snapshot["device"])
        return tensor.set_(buffers[arg["storage"]].untyped_storage(), arg["offset"], arg["size"], arg["stride"])

    args = [restore(v) for v in snapshot["args"]]
    kwargs = {k: restore(v) for k, v in snapshot["kwargs"].items()}
    grid = snapshot["grid"]
    mod, instrumented = _load_kernel(inst, snapshot, True)
    recorder = mod.FiAutoBuffer(math.prod(grid), snapshot["num_warps"], 256, device=snapshot["device"])
    mod.fi_attach(recorder)
    if mode == "record":
        instrumented[grid](*args, **kwargs)
        recorder.dump(directory / "capture.fi02.bin")
        return
    _, baseline = _load_kernel(base, snapshot, False)
    kernels = {"baseline": baseline[grid], "instrumented": instrumented[grid]}
    _bytes_file(directory / "expected.bin", snapshot["after"])
    metrics = dict(warmup=8, expected_output="expected.bin")
    shared = {}
    graphs = {}
    knobs.nvidia.dump_ptxas_log = True
    for variant, launch in kernels.items():
        for dst, src in zip(buffers, initial):
            dst.copy_(src)
        log = directory / f"{variant}.ptxas.log"
        with log.open("w", encoding="utf-8") as stream, redirect_stdout(stream):
            compiled = launch(*args, **kwargs)
        _bytes_file(directory / f"{variant}.bin", buffers)
        cubin = directory / f"{variant}.cubin"
        cubin.write_bytes(compiled.asm["cubin"])
        with (directory / f"{variant}.sass").open("wb") as out:
            subprocess.run(["cuobjdump", "--dump-sass", str(cubin)], stdout=out, check=True)
        spill = re.findall(r"(\d+) bytes spill stores,\s*(\d+) bytes spill loads", log.read_text())
        if len(spill) != 1:
            raise ValueError(f"Expected the target kernel's ptxas spill report in {log}")
        shared[variant] = compiled.metadata.shared
        metrics.update({variant + "_spill_bytes": sum(map(int, spill[0])),
                        variant + "_kernel": compiled.metadata.name, variant + "_output": variant + ".bin",
                        variant + "_sass": variant + ".sass", variant + "_ns": []})
        # External event nodes measure GPU execution within a pre-built single-launch graph.
        start, end = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            start.record()
            launch(*args, **kwargs)
            end.record()
        graphs[variant] = graph, start, end
    # Strict ABBA timing blocks (baseline, instrumented, instrumented, baseline)
    # cancel linear GPU clock/thermal drift across the measurement window.
    n_blocks = 16  # 32 paired samples per variant after warmup
    for index in range(metrics["warmup"]):
        for variant in ("baseline", "instrumented"):
            for dst, src in zip(buffers, initial):
                dst.copy_(src)
            graph, start, end = graphs[variant]
            graph.replay()
            end.synchronize()
    from .harness import abba_schedule
    for variant in abba_schedule(n_blocks):
        for dst, src in zip(buffers, initial):
            dst.copy_(src)
        graph, start, end = graphs[variant]
        graph.replay()
        end.synchronize()
        metrics[variant + "_ns"].append(round(start.elapsed_time(end) * 1e6))
    metrics["timing_protocol"] = "ABBA"
    metrics.update(kernel_name=metrics["instrumented_kernel"],
                   extra_smem_bytes=shared["instrumented"] - shared["baseline"])
    write_json(directory / "runtime.json", metrics)


def _main():
    mode, *args = sys.argv[1:]
    if mode in ("measure", "record"):
        _worker(mode, *map(Path, args))
    elif mode == "collect":
        source, directory = map(Path, args[:2])
        argv = args[2:]
        while argv and argv[0] in ("-B", "-u"):
            argv = argv[1:]
        with capture(source, directory.parent, read_json(directory / "probe.json"), None):
            if argv[:1] == ["-m"]:
                sys.argv = argv[1:]
                runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
            else:
                sys.argv = argv
                sys.path.insert(0, str(Path(argv[0]).resolve().parent))
                runpy.run_path(argv[0], run_name="__main__")
    else:
        raise ValueError(f"Unknown replay action {mode}")


if __name__ == "__main__":
    _main()
