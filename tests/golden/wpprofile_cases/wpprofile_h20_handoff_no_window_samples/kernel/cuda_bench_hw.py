"""Hopper replay adapter. Run inside weaveprof profile, or pass --baseline-dir."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

import torch


CTAS = 2
K = 4096
WARMUP = 8
SAMPLES = 30
FIXTURE = Path(__file__).resolve().parent
WRAPPER = """
extern "C" int run_kernel(const uint32_t *x, float *y) {
    warp_handoff<<<2, 160>>>(x, y, 4096);
    return cudaGetLastError();
}
#if FI_LEVEL == 2
static WpBuf capture_buffer;
extern "C" int attach_buffer() { return wp_host_init(&capture_buffer, 2, 5, 16400); }
extern "C" int dump_buffer(const char *path) { return wp_host_dump(&capture_buffer, path); }
#endif
"""


def compile_kernel(work, name, source):
    instrumented = name == "inst"
    wrapper = work / (name + ".cu")
    wrapper.write_text(source.read_text(encoding="utf-8") + "\n" + WRAPPER, encoding="utf-8")
    lib = work / (name + ".so")
    cmd = ["nvcc", "-O3", "-arch=sm_90a", "-shared", "-Xcompiler=-fPIC",
           "-Xlinker=-Bsymbolic", "-Xptxas=-v", "-DFI_LEVEL=" + ("2" if instrumented else "0")]
    if instrumented:
        headers = Path(os.environ["WEAVEPROF_ROOT"]) / "recorders/cuda_cpp"
        cmd += ["-I", str(headers)]
    comp = subprocess.run([*cmd, str(wrapper), "-o", str(lib)], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    (work / (name + ".ptxas.log")).write_text(comp.stdout)
    if comp.returncode:
        raise RuntimeError(comp.stdout)
    with (work / (name + ".sass")).open("w") as output:
        subprocess.run(["cuobjdump", "--function", "warp_handoff", "--dump-sass", str(lib)],
                       stdout=output, check=True)
    spills = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", comp.stdout)
    smem = re.search(r"(\d+) bytes smem", comp.stdout)
    if spills is None or smem is None:
        raise ValueError("Expected ptxas resource evidence in " + str(work / (name + ".ptxas.log")))
    return sum(map(int, spills.groups())), int(smem[1])


def check(status):
    if status:
        raise RuntimeError("CUDA status " + str(status))


def load_kernel(work, name):
    lib = ctypes.CDLL(str(work / (name + ".so")))
    lib.run_kernel.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    if name == "inst":
        check(lib.attach_buffer())
    return lib


def make_input(work):
    generator = torch.Generator().manual_seed(47361)
    x = torch.randn((CTAS, 256), dtype=torch.float16, generator=generator)
    (work / "randn.f16").write_bytes(x.numpy().tobytes())
    return x.cuda()


def output_tensor():
    return torch.full((CTAS, 128, 4), float("nan"), dtype=torch.float32, device="cuda")


def launch(lib, x, out):
    check(lib.run_kernel(x.data_ptr(), out.data_ptr()))


def capture_one(work):
    x = torch.frombuffer(bytearray((work / "randn.f16").read_bytes()), dtype=torch.float16).cuda()
    out = output_tensor()
    lib = load_kernel(work, "inst")
    launch(lib, x, out)
    torch.cuda.synchronize()
    check(lib.dump_buffer(os.fsencode(work / "capture.fi02.bin")))


def bench(work, baseline_only):
    work.mkdir(exist_ok=True, parents=True)
    sources = {"l0": FIXTURE / "handoff.cu" if baseline_only else Path(os.environ["WEAVEPROF_BASELINE_SOURCE"])}
    if not baseline_only:
        sources["inst"] = Path(os.environ["WEAVEPROF_INSTRUMENTED_SOURCE"])
    resources = {name: compile_kernel(work, name, file) for name, file in sources.items()}
    libs = {name: load_kernel(work, name) for name in sources}
    x = make_input(work)
    outputs = {name: output_tensor() for name in sources}
    times = {name: [] for name in sources}
    events = {name: [torch.cuda.Event(enable_timing=True) for _ in range(2)] for name in sources}
    for i in range(WARMUP + SAMPLES):
        order = list(sources)
        if i % 2:
            order.reverse()
        for name in order:
            start, end = events[name]
            start.record()
            launch(libs[name], x, outputs[name])
            end.record()
            end.synchronize()
            if i >= WARMUP:
                times[name].append(round(start.elapsed_time(end) * 1e6))
    paths = {"l0": "baseline.bin", "inst": "instrumented.bin"}
    for name, output in outputs.items():
        value = output.cpu()
        if not torch.isfinite(value).all() or not torch.count_nonzero(value):
            raise ValueError("Incomplete or non-finite output in " + name)
        (work / paths[name]).write_bytes(value.numpy().tobytes())
    metrics = {"kernel_name": "warp_handoff", "warmup": WARMUP, "baseline_ns": times["l0"],
               "baseline_output": paths["l0"], "baseline_sass": "l0.sass"}
    if not baseline_only:
        nsys = os.environ["WP_NSYS"]                       # 10.1 预检得到的真实路径
        run_as = os.environ["WP_RUN_AS"]                   # 普通用户名（id -un）
        D = os.environ["WP_RUN_DIR"]                       # 运行目录
        assert not Path(nsys).resolve().is_relative_to(Path(D).resolve()), nsys
        gpu = os.environ["CUDA_VISIBLE_DEVICES"]
        assert gpu.isdigit(), "need exactly one locked GPU, got " + gpu
        child_env = {
            "PYTHONPATH": os.environ["PYTHONPATH"], "WP_AUDIT_LOG": os.environ["WP_AUDIT_LOG"],
            "CUDA_VISIBLE_DEVICES": gpu, "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": D + "/child-home", "XDG_CACHE_HOME": D + "/child-home/.cache",
            "XDG_CONFIG_HOME": D + "/child-home/.config", "TMPDIR": D + "/child-tmp",
            "CUDA_CACHE_PATH": D + "/child-home/nv-cache", "TORCH_HOME": D + "/child-home/torch",
            "TRITON_CACHE_DIR": D + "/child-home/triton", "WEAVEPROF_CACHE": D + "/cache",
            "PATH": os.environ["PATH"],
        }
        if "LD_LIBRARY_PATH" in os.environ:
            child_env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
        envs = [k + "=" + v for k, v in child_env.items()]
        assert not any("," in e for e in envs), envs     # 分隔写法以 10.1 第 5 条核实的为准
        prof = ["sudo", nsys, "profile", "--run-as=" + run_as,
                "--sample=none", "--cpuctxsw=none", "--trace=cuda", "--force-overwrite=true",
                "--gpu-metrics-devices=" + gpu, "--gpu-metrics-frequency=200000",
                "--env-var=" + ",".join(envs), "-o", str(work / "capture"),
                sys.executable, "-B", __file__, "--capture-one", str(work)]
        print("SUDO_CMD:", " ".join(prof), flush=True)      # 4.3③ 命令原文
        subprocess.run(prof, check=True, stdin=subprocess.DEVNULL)
        export = [nsys, "export", "--type=sqlite", "--force-overwrite=true",
                  "-o", str(work / "capture.sqlite"), str(work / "capture.nsys-rep")]
        first = subprocess.run(export, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if first.returncode:                               # 4.3⑨：先不用 sudo
            print("EXPORT_WITHOUT_SUDO_FAILED rc=%d" % first.returncode, flush=True)
            print(first.stdout, flush=True)                # 报错原样进 bench.log，作为"写明原因"
            sudo_export = ["sudo"] + export
            print("SUDO_CMD:", " ".join(sudo_export), flush=True)
            subprocess.run(sudo_export, check=True, stdin=subprocess.DEVNULL)
        metrics.update(instrumented_ns=times["inst"], instrumented_output=paths["inst"],
                       instrumented_sass="inst.sass", baseline_spill_bytes=resources["l0"][0],
                       instrumented_spill_bytes=resources["inst"][0],
                       extra_smem_bytes=resources["inst"][1] - resources["l0"][1])
    (work / "runtime.json").write_text(json.dumps(metrics, indent=2))
    print({name: statistics.median(values) for name, values in times.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--capture-one", type=Path)
    args = parser.parse_args()
    if args.capture_one:
        capture_one(args.capture_one)
    else:
        bench((args.baseline_dir or Path(os.environ["WEAVEPROF_ROUND_DIR"])).resolve(), bool(args.baseline_dir))
