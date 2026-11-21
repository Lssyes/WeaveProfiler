# -*- coding: utf-8 -*-
"""被测 kernel 的 Python 驱动模板：用 torch / ctypes 调自写 .so，或调 Triton / vllm / sglang / flashinfer 的 kernel。

先填 TODO 再用；--dry-run 不需要 GPU、不 import torch，用来验证模板骨架（延迟 import、日志捕获、日志门禁）本身没坏。

用法：
  python3 -I driver_template.py --arm ours_so --log run_ours.log --so ./libkernel.so --out result_ours.json
  python3 -I driver_template.py --arm vllm    --log run_vllm.log --out result_vllm.json
  python3 -I driver_template.py --dry-run --log local_test/driver_dry.log             # 骨架自检，无需 GPU
  python3 -I driver_template.py --dry-run --log x.log --inject-warning                # 验证门禁真的会拦：应退出码 1、不写结果文件
上卡前先跑 4_validate/tools/fi_preflight.py --gpu <id>（卡被别人占着就等，不要 kill）。

====================  三条规则（每条的理由与落点）  ====================

(a) 被测臂不需要的库，一律在用到的函数里延迟 import，顶层不 import。
    理由：import vllm / sglang / flashinfer / torch 本身就会初始化 CUDA 上下文、改 allocator 状态、往 stderr 吐一大批日志。
    如果 A 臂（自写 .so）的进程顶层 import 了 vllm，A 臂的计时和日志就被 B 臂的库污染了，对比不再公平。
    落点：本文件顶层只 import 标准库；每个臂函数（arm_xxx）里才通过 import_quiet() 引入自己需要的库；
         --dry-run 末尾检查 sys.modules，任何重库被提前加载就报错（防止有人在顶层偷偷加 import）。

(b) 必须 import 的大库，先把它的日志压到 ERROR，再 import，然后只打印一行“已导入 X，日志级别压到 ERROR”。
    顺序不能反：多数库在 import 时就读取环境变量（如 vllm 的 VLLM_LOGGING_LEVEL）并配置自己的 logger，import 之后再设就晚了。
    落点：import_quiet(name) = 设环境变量（LIB_ENV）+ logging.getLogger(name).setLevel(ERROR) + importlib.import_module
         + import 后再压一次 logger（部分库在 import 时会重设自己的 logger）+ 打印一行。
    注意：这一行本身含 ERROR 字样，会被 fi_logcheck 的 `ERROR` 规则命中，所以第 (c) 步调用门禁时必须
         用 --allow 精确放行这一行（DEFAULT_ALLOW），而不是放宽整体规则。
    注意：warnings 模块发出的告警（如 UserWarning）不受 logger 级别影响，会进日志并被门禁抓住 —— 这是故意的：
         要么修掉它，要么用 --allow 写明放行哪一类、为什么。

(c) 结尾自动调用 fi_logcheck 检查自己写的日志文件；不干净就 sys.exit(1) 且不写结果文件。
    理由：带着隐藏报错（CUDA 调用失败、No module named、Traceback）跑完的计时数据不可信，不能让它落成“结果”。
    落点：本脚本把自己的输出写进 --log：logging（FileHandler 式的 StreamHandler）+ contextlib.redirect_stderr（Python 层 stderr，
         含 warnings）+ os.dup2（fd 层 stderr，能抓到 .so / CUDA / CUPTI 用 printf、fprintf 写的内容）。
         跑完关闭日志，subprocess 调 4_validate/tools/fi_logcheck.py；返回非 0 就不写 --out，退出码 1。
         若 --out 指向的文件是上次遗留的，脚本只会提示“这不是本次结果”，不会替你删除它。

    局限：进程被 C 层 abort / segfault 直接杀死时门禁来不及运行（实测：本机 Windows conda 的 torch 在 import 时报
         `OMP: Error #15`（重复 libiomp5md.dll）后 abort，退出码 3）。此时退出码非 0、不写结果文件，fd 层捕获已把报错写进 --log，
         用 `python3 -I 4_validate/tools/fi_logcheck.py <log>` 查看即可；不要为了让它跑过而放宽门禁。

退出码：0=日志干净且（若给了 --out）结果已写；1=驱动运行失败或日志不干净（不写结果文件）。
"""
import argparse
import contextlib
import importlib
import json
import logging
import os
import subprocess
import sys
import time
import warnings

# Windows 控制台常是 GBK：统一把 stdout/stderr 改成 UTF-8 + replace，打印中文不崩（没有 reconfigure 的旧版本不调用）
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOGCHECK = os.path.normpath(os.path.join(HERE, "..", "tools", "fi_logcheck.py"))

# ---- 规则 (a)：顶层不 import 的重库名单（--dry-run 会检查它们都没被加载）
HEAVY = ("torch", "vllm", "sglang", "flashinfer", "triton")

# 每个被测臂需要哪些库（只在该臂的函数里 import）
ARM_LIBS = {
    "ours_so": ["torch"],                 # ctypes 调自写 .so，torch 只负责分配张量 / 计时
    "triton": ["torch", "triton"],
    "vllm": ["torch", "vllm"],
    "sglang": ["torch", "sglang"],
    "flashinfer": ["torch", "flashinfer"],
}

# ---- 规则 (b)：import 前要设置的环境变量（按库名）。只写有把握的；其它库按其文档补充，拿不准的不写，靠 logger 级别兜底
LIB_ENV = {
    "vllm": {"VLLM_LOGGING_LEVEL": "ERROR"},
    "torch": {"TORCH_CPP_LOG_LEVEL": "ERROR"},
}

# ---- 规则 (c)：调用 fi_logcheck 时放行的行。只放行规则 (b) 自己打印的那一行，别的 ERROR / WARNING 一律拦
DEFAULT_ALLOW = [r"已导入 \S+，日志级别压到 ERROR"]

PROBE_PY = "dry-run：python 层 stderr 捕获探针"
PROBE_FD = "dry-run：fd 层 stderr 捕获探针"

STATE = {"dry_run": False}
_QUIET_DONE = set()


def say(msg):
    """既写进日志（经 logging），也打印到控制台一行。"""
    logging.getLogger("driver").info(msg)
    print(msg)


def import_quiet(name):
    """规则 (b)：先压日志、后 import、再只打印一行。dry-run 时走完同样的步骤但不真 import（所以不需要装这些库）。"""
    for k, v in LIB_ENV.get(name, {}).items():
        os.environ[k] = v                                    # 必须在 import 之前：库在 import 时读取
    logging.getLogger(name).setLevel(logging.ERROR)
    if name in _QUIET_DONE:
        return sys.modules.get(name)
    mod = None
    if not STATE["dry_run"]:
        mod = importlib.import_module(name)
        logging.getLogger(name).setLevel(logging.ERROR)      # 部分库在 import 时会重设自己的 logger，再压一次
    _QUIET_DONE.add(name)
    say("已导入 %s，日志级别压到 ERROR%s" % (name, "（dry-run：未实际 import）" if STATE["dry_run"] else ""))
    return mod


@contextlib.contextmanager
def capture_stderr(logf):
    """把 stderr 同时在 Python 层（redirect_stderr，含 warnings）和 fd 层（dup2，含 C/C++ 库的 fprintf）导进日志文件。"""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = None
    try:
        saved = os.dup(2)
        os.dup2(logf.fileno(), 2)
    except OSError:                                           # 没有可用的 fd 2（如 pythonw）：只做 Python 层
        saved = None
    try:
        with contextlib.redirect_stderr(logf):
            yield
    finally:
        logf.flush()
        if saved is not None:
            os.dup2(saved, 2)
            os.close(saved)


# ---------------------------------------------------------------- 计时辅助
def bench_cuda(fn, warmup=10, iters=20):
    """cudaEvent 计时，返回每次的 us 列表（升序）。SKILL 要求 >=20 次取中位。注意：本函数未在 GPU 上实测过，用前自查。"""
    torch = import_quiet("torch")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)                    # ms -> us
    return sorted(ts)


def summarize(ts):
    n = len(ts)
    return dict(n=n, median_us=ts[n // 2] if n % 2 else 0.5 * (ts[n // 2 - 1] + ts[n // 2]), min_us=ts[0], max_us=ts[-1])


# ---------------------------------------------------------------- 各被测臂（kernel 调用留 TODO）
# 每个臂函数返回一个可 JSON 化的 dict（会进 --out 结果文件）。重库只在函数里经 import_quiet() 引入（规则 a）。
# 所有臂共同的 TODO（来自 SKILL / 项目规则）：
#   * 输入用 torch.randn（真实形状 + randn），不要用全 0 / 全 1 / arange 这类合成输入；
#   * 正确性要把输出与参考实现逐元素全量比较（报 max_abs_err、不匹配元素个数），不要只比一两个值；
#   * 计时用 bench_cuda（>=20 次取中位），插桩构建与 FI_LEVEL=0 构建的开销对比也走这里。
def arm_ours_so(args):
    torch = import_quiet("torch")
    import ctypes
    # TODO(1) lib = ctypes.CDLL(args.so)；设置每个导出函数的 argtypes / restype
    # TODO(2) x = torch.randn(..., device="cuda")；分配输出与参考输出
    # TODO(3) 调 lib.run_kernel(x.data_ptr(), ...)；torch.cuda.synchronize()；全量比较输出
    # TODO(4) ts = bench_cuda(lambda: lib.run_kernel(...), args.warmup, args.iters)；return dict(**summarize(ts), max_abs_err=...)
    raise NotImplementedError("驱动模板：请在 arm_ours_so 里填写 kernel 调用（见上面 TODO）")


def arm_triton(args):
    torch = import_quiet("torch")
    triton = import_quiet("triton")
    # TODO 导入你的 @triton.jit kernel 并按上面的共同 TODO 调用、比较、计时
    raise NotImplementedError("驱动模板：请在 arm_triton 里填写 kernel 调用")


def arm_vllm(args):
    vllm = import_quiet("vllm")
    # TODO 调 vllm 的对应算子 / 引擎接口，作为对照臂；注意 vllm 内部可能自己调 cudaProfilerStop，配合 nsys 时见 SKILL 第 5 节
    raise NotImplementedError("驱动模板：请在 arm_vllm 里填写调用")


def arm_sglang(args):
    sglang = import_quiet("sglang")
    # TODO 调 sglang 的对应算子 / 接口，作为对照臂
    raise NotImplementedError("驱动模板：请在 arm_sglang 里填写调用")


def arm_flashinfer(args):
    flashinfer = import_quiet("flashinfer")
    # TODO 调 flashinfer 的对应算子，作为对照臂
    raise NotImplementedError("驱动模板：请在 arm_flashinfer 里填写调用")


ARMS = {"ours_so": arm_ours_so, "triton": arm_triton, "vllm": arm_vllm, "sglang": arm_sglang, "flashinfer": arm_flashinfer}
assert set(ARMS) == set(ARM_LIBS)


# ---------------------------------------------------------------- dry-run：不需要 GPU，不 import 任何重库
def run_dry(args):
    for lib in ARM_LIBS[args.arm]:
        import_quiet(lib)                                     # 走完规则 (b) 的步骤（设环境变量、压 logger、打印一行），但不真 import
    loaded = [m for m in HEAVY if m in sys.modules]
    if loaded:
        raise RuntimeError("规则 (a) 被破坏：顶层或依赖链已提前加载重库 %s，请把它们改成函数内延迟 import" % loaded)
    say("规则 (a) 检查通过：%s 均未被加载（顶层只有标准库）" % " / ".join(HEAVY))
    print(PROBE_PY, file=sys.stderr)                          # 经 redirect_stderr 进日志
    os.write(2, (PROBE_FD + "\n").encode("utf-8"))            # 绕过 Python 直接写 fd 2，验证 dup2 捕获
    return dict(arm=args.arm, dry_run=True, note="dry-run：未调用 kernel")


def run(args):
    logging.getLogger("driver").info("开始：arm=%s dry_run=%s", args.arm, args.dry_run)
    if args.inject_warning:
        warnings.warn("driver_template 注入的测试告警（--inject-warning）")
    return run_dry(args) if args.dry_run else ARMS[args.arm](args)


def verify_capture(log_path):
    """dry-run 专用：确认两个探针都真的落进了日志文件（证明 Python 层和 fd 层的 stderr 都被捕获）。"""
    with open(log_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return dict(python层=PROBE_PY in text, fd层=PROBE_FD in text)


def run_logcheck(logcheck, log_path, allow):
    """规则 (c)：subprocess 调 fi_logcheck.py，转发它的报告，返回退出码。找不到脚本按失败处理（没有门禁就不能证明干净）。"""
    if not os.path.isfile(logcheck):
        print("找不到日志门禁脚本 %s（用 --logcheck 指定）—— 无法证明日志干净，按失败处理" % logcheck)
        return 2
    cmd = [sys.executable, "-I", logcheck, log_path]
    for a in allow:
        cmd += ["--allow", a]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(p.stdout.decode("utf-8", "replace"), end="")
    if p.stderr:
        print(p.stderr.decode("utf-8", "replace"), end="")
    return p.returncode


def parse_args():
    ap = argparse.ArgumentParser(description="被测 kernel 的 Python 驱动模板")
    ap.add_argument("--arm", choices=sorted(ARMS), default="ours_so", help="被测臂（默认 ours_so）")
    ap.add_argument("--log", required=True, help="本脚本自己的输出写到这个日志文件（每次运行覆盖）")
    ap.add_argument("--out", help="结果 JSON 路径；日志不干净时不会写")
    ap.add_argument("--so", help="ours_so 臂：自写 .so 的路径")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20, help="计时次数，SKILL 要求 >=20")
    ap.add_argument("--allow", action="append", default=[], metavar="REGEX", help="额外放行的日志行正则（传给 fi_logcheck --allow），可重复")
    ap.add_argument("--logcheck", default=DEFAULT_LOGCHECK, help="fi_logcheck.py 的路径（默认 ../4_validate/tools/fi_logcheck.py）")
    ap.add_argument("--dry-run", action="store_true", help="不需要 GPU、不 import torch/任何重库，只验证骨架")
    ap.add_argument("--inject-warning", action="store_true", help="仅用于验证门禁：往日志注入一条 UserWarning，应被拦住")
    return ap.parse_args()


def main():
    args = parse_args()
    STATE["dry_run"] = args.dry_run
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)

    failed, result = False, None
    logf = open(args.log, "w", encoding="utf-8", newline="\n", buffering=1)
    handler = logging.StreamHandler(logf)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        with capture_stderr(logf):
            try:
                result = run(args)
            except Exception:                                 # 失败时 Traceback 进日志，下面的门禁会抓住它
                failed = True
                logging.getLogger("driver").exception("驱动运行失败")
    finally:
        root.removeHandler(handler)
        handler.flush()
        logf.close()

    if args.dry_run and not failed:
        cap = verify_capture(args.log)
        print("stderr 捕获自检：" + "，".join("%s=%s" % (k, "通过" if v else "未捕获") for k, v in cap.items()))
        failed = not all(cap.values())

    rc = run_logcheck(args.logcheck, args.log, DEFAULT_ALLOW + args.allow)
    if rc != 0 or failed:
        print("日志不干净或驱动运行失败：不写结果文件，退出码 1（日志：%s）" % args.log)
        if args.out and os.path.exists(args.out):
            print("注意：%s 是上次运行遗留的文件，不是本次结果，请勿使用" % args.out)
        sys.exit(1)

    if args.out:
        doc = dict(arm=args.arm, dry_run=args.dry_run, log=os.path.abspath(args.log), logcheck="PASS",
                   time=time.strftime("%Y-%m-%dT%H:%M:%S"), result=result)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print("结果已写入 " + args.out)
    print("完成：日志干净（%s）" % args.log)


if __name__ == "__main__":
    main()
