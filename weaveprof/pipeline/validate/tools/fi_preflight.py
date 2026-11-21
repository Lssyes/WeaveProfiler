# -*- coding: utf-8 -*-
"""fast-instrument 插桩前环境自检：每项一行“结论 + 修法”，结尾 PREFLIGHT PASS / PREFLIGHT FAIL（退出码 0 / 1）。

为什么要有它：插桩构建和 nsys 对齐都很贵（nsys 启动 20~25 s、占着 GPU 锁），环境问题（nvcc 不认 arch、卡被别人占着、
锁目录没指到共享目录导致锁外占卡）应在上卡前就暴露，而不是跑完一轮才发现数据作废。

用法：
  python3 -I weaveprof/pipeline/validate/tools/fi_preflight.py --card h20 --arch sm_90 --language cuda_cpp --gpu 0 --json out.json
  python3 -I weaveprof/pipeline/validate/tools/fi_preflight.py --card b300 --arch sm_103a --language triton --json out.json
  python3 -I weaveprof/pipeline/validate/tools/fi_preflight.py --card mi308x --arch gfx942 --language hip_cpp --gpu 0 [--rocm-llvm /opt/rocm/llvm/bin]
  python3 -I weaveprof/pipeline/validate/tools/fi_preflight.py --inspect-host             # 显式允许读本机工具补全缺失目标
  python3 -I weaveprof/pipeline/validate/tools/fi_preflight.py --selftest                 # 假命令自检，不依赖本机工具/GPU

目标：card/arch/platform 必须显式给出，或显式加 --inspect-host。默认不按本机工具猜平台，也不默认 NVIDIA；
路由统一走 registry.py。AMD 分支查 hipcc / rocminfo 架构 / llvm-objdump / rocprofv3 / rocm-smi，
完全不查 nvcc、nsys、nvidia-smi。

检查项（状态 OK / WARN / FAIL / SKIP；只要有一项 FAIL 就整体 FAIL）：
  python        Python >= 3.8                                          （不满足 FAIL）
  stdlib        json / gzip / sqlite3 标准库                             （缺任一 FAIL）
  matplotlib    概览 PNG 用，可选                                        （缺只 WARN）
  nvcc          在 PATH 且能报版本                                       （缺 FAIL）
  nvcc-arch     --arch 能被 nvcc 接受（--list-gpu-arch / --list-gpu-code；带 a/f 后缀的架构用 --dryrun 确认）
  nvcc-host     只在 --dryrun 暴露 host 编译器缺失（如 Windows 没有 cl.exe）时出现   （WARN）
  nsys          可选；加 --need-nsys 后缺失变 FAIL
  nvidia-smi    可用且能列出 GPU                                         （缺 FAIL）
  gpu-list      GPU 列表（编号、型号、利用率、显存）
  gpu-busy      --gpu 指定的卡 utilization.gpu > --busy-util(默认 5)% 或 memory.used > --busy-mem-mib(默认 1024) MiB 判“被占用”（FAIL）
  lock-dir      --lock-dir 存在、是目录、可写（用 os.access，不创建探针文件，避免干扰锁目录里的文件）
  lock-dir-match  与 --expect-lock-dir 规范化后比较，不一致 FAIL

AMD 分支（--platform amd 或 --arch gfx*）：
  python/stdlib/matplotlib  同上
  hipcc         在 PATH 且能报版本                                       （缺 FAIL）
  hip-arch      rocminfo 报的 ISA 里有 --arch 给的 gfx 号                  （rocminfo 不可用则 SKIP，以真实编译为准）
  llvm-objdump  C1 回查二进制要用；按 FI_LLVM_OBJDUMP / --rocm-llvm / /opt/rocm*/lib/llvm/bin / PATH 顺序找（缺 FAIL）
  rocprofv3     硬件计数器采集用，可选                                     （缺 WARN）
  gpu-list/busy rocm-smi --showuse / --showmemuse / --showpids             （rocm-smi 缺或输出解析不出来则 SKIP）

静默规则：默认（--quiet）所有子进程的 stderr 一律捕获、不透传，控制台只有结论行；排查时加 --verbose 才把捕获到的 stderr 缩进打印。
兼容性：本脚本只用标准库，语法兼容 Python 3.6+，这样 Python 太旧时也能跑出“Python 版本不够”的结论而不是语法错误。
"""
import argparse
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

# ``-I`` does not add the script directory to sys.path.  The backend registry
# is the sole source of card/architecture/language/profiler routing.
SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if SKILL_ROOT not in sys.path:
    sys.path.insert(0, SKILL_ROOT)

import weaveprof.pipeline.core.registry as backend_registry

# Windows 控制台常是 GBK：统一把 stdout/stderr 改成 UTF-8 + replace，打印中文不崩（没有 reconfigure 的旧版本不调用）
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"


def R(name, status, msg, fix="", stderr=""):
    """一条检查结果：name 是固定的 ASCII 短名，msg 是结论，fix 是修法（仅 WARN/FAIL 时打印），stderr 是捕获到的子进程 stderr 开头。"""
    return dict(name=name, status=status, msg=msg, fix=fix, stderr=stderr)


def first_line(text, n=160):
    for ln in (text or "").splitlines():
        if ln.strip():
            return ln.strip()[:n]
    return ""


def run_cmd(argv, timeout=30):
    """运行子进程，stdout/stderr 全部捕获、不透传。返回 (rc, stdout, stderr)；
    程序不存在 / 超时 / 无法启动时 rc=None，stderr 里放原因。"""
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, timeout=timeout)
    except FileNotFoundError as e:
        return None, "", "找不到程序：%s" % e
    except subprocess.TimeoutExpired:
        return None, "", "超时 %d s" % timeout
    except OSError as e:
        return None, "", "无法启动：%s" % e
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


# ---------------------------------------------------------------- Python 环境
def check_python():
    v = sys.version_info
    s = "%d.%d.%d" % (v[0], v[1], v[2])
    if v >= (3, 8):
        return R("python", OK, "Python %s (%s)" % (s, sys.executable))
    return R("python", FAIL, "Python %s 低于 3.8" % s, "改用 python3.8 以上（新建 conda 环境需要你审核后再建，或用系统自带的更高版本）")


def check_stdlib():
    missing = []
    for m in ("json", "gzip", "sqlite3"):
        try:
            importlib.import_module(m)
        except Exception as e:  # ImportError 或 _sqlite3 动态库加载失败
            missing.append("%s(%s)" % (m, first_line(str(e), 60)))
    if missing:
        return R("stdlib", FAIL, "标准库缺失：" + "、".join(missing),
                 "该 Python 构建不完整（常见于缺 libsqlite3 的精简 Python）；换用带 sqlite3 的发行版 / conda 的 python")
    return R("stdlib", OK, "json / gzip / sqlite3 均可导入")


def check_matplotlib():
    try:
        spec = importlib.util.find_spec("matplotlib")
    except Exception:
        spec = None
    if spec is None:
        return R("matplotlib", WARN, "未安装 matplotlib（可选）", "概览 PNG 需要 matplotlib，pip install matplotlib 或 --no-png")
    ver = ""
    try:
        import importlib.metadata as md  # Python 3.8+
        ver = md.version("matplotlib")
    except Exception:
        pass
    return R("matplotlib", OK, "matplotlib %s 可用（概览 PNG）" % (ver or "(版本未知)"))


# ---------------------------------------------------------------- nvcc / 架构
def check_nvcc():
    """返回 (nvcc 路径或 None, 结果)。"""
    path = shutil.which("nvcc")
    if not path:
        return None, R("nvcc", FAIL, "nvcc 不在 PATH",
                       "安装 CUDA Toolkit 并把 <cuda>/bin 加进 PATH（如 export PATH=/usr/local/cuda/bin:$PATH）；只解码已有 .bin 可忽略")
    rc, out, err = run_cmd([path, "--version"])
    m = re.search(r"release\s+([\d.]+),\s*(V[\d.]+)", out)
    if rc != 0 or not m:
        return path, R("nvcc", FAIL, "nvcc --version 失败（rc=%s）：%s" % (rc, first_line(err or out, 80)),
                       "检查 CUDA 安装是否完整、PATH 里是不是别的同名程序", stderr=first_line(err, 300))
    return path, R("nvcc", OK, "CUDA release %s（%s），%s" % (m.group(1), m.group(2), path))


def _parse_arch_list(text):
    return set(re.findall(r"\b((?:sm|compute|lto)_[0-9]+[a-z]*)\b", text))


def _natural(s):
    m = re.search(r"(\d+)", s)
    return (int(m.group(1)) if m else 0, s)


def dryrun_arch(nvcc, arch):
    """用 nvcc --dryrun 探测 -arch=<arch> 是否被接受（不编译、不写文件，输入用空设备）。
    返回 (结论, 输出开头)：结论 'unsupported'=nvcc 明确说不支持；'unknown'=nvcc 不认 --dryrun 或根本没跑起来；'accepted'=没有架构类报错。
    nvcc 先校验架构、后找 host 编译器，所以即便缺 cl.exe，只要报的不是 Unsupported gpu architecture 就说明架构本身被接受。"""
    rc, out, err = run_cmd([nvcc, "--dryrun", "-arch=" + arch, "-x", "cu", "-c", os.devnull])
    text = (out or "") + "\n" + (err or "")
    if rc is None or re.search(r"Unknown option|Unrecognized option|unknown option", text):
        return "unknown", first_line(text)
    if re.search(r"Unsupported gpu architecture", text, re.I):
        return "unsupported", first_line(text)
    return "accepted", first_line(text)


def check_arch_one(nvcc, arch, comp, code):
    """单个 --arch 的结果列表。arch 形如 sm_103a / sm_90 / compute_100 / 103a；comp/code 是 --list-gpu-arch / --list-gpu-code 解析出的集合。
    判定顺序：
      1) 有 list 时先查基础架构（如 compute_103、sm_103）在不在 list 里，不在直接 FAIL 并列出该 nvcc 支持的架构；
      2) 再用 nvcc --dryrun 探一次（不编译、不写文件）：报 Unsupported gpu architecture 则 FAIL。
         之所以必须有第 2 步：--list-* 只列基础架构，不列 sm_100a / sm_103a 这类架构特化（a/f 后缀）版本，带后缀的只能靠 dryrun 确认；
      3) dryrun 同时能暴露 host 编译器缺失（如 Windows 没有 cl.exe），单独给一条 nvcc-host 警告。
    没有 --list-gpu-arch 子命令的旧 nvcc 只靠第 2 步；第 2 步也不可用才 SKIP。"""
    m = re.match(r"^(?:(sm|compute)_)?([0-9]+)([af]?)$", arch)
    if not m:
        return [R("nvcc-arch", FAIL, "--arch %r 格式不对" % arch, "写成 sm_103a / sm_90 / compute_100 这种形式")]
    kind, num, suf = m.group(1), m.group(2), m.group(3)
    full = "%s_%s%s" % (kind or "sm", num, suf)
    have_list = bool(comp or code)
    if have_list:
        in_sm = ("sm_" + num) in code or ("compute_" + num) in comp
        in_cp = ("compute_" + num) in comp
        base_ok = in_cp if kind == "compute" else (in_sm if kind == "sm" else (in_sm or in_cp))
        if not base_ok:
            sup = sorted({x for x in (comp | code) if x.startswith("sm_")} or comp, key=_natural)
            return [R("nvcc-arch", FAIL, "%s 不在该 nvcc 的支持列表里（基础架构 %s 缺失）" % (full, num),
                      "升级 CUDA Toolkit 或改用受支持架构；该 nvcc 支持：" + " ".join(sup))]
    verdict, text = dryrun_arch(nvcc, full)
    if verdict == "unsupported":
        return [R("nvcc-arch", FAIL, "%s 被 nvcc --dryrun 判为不支持（%s）" % (full, text[:80]),
                  "升级 CUDA Toolkit 或改用受支持架构" + ("；基础架构在 list 里而后缀版本不支持，也可以去掉后缀 %s" % suf if suf and have_list else ""))]
    res = []
    if verdict == "unknown":
        if not have_list:
            return [R("nvcc-arch", SKIP, "该 nvcc 没有 --list-gpu-arch 子命令，且 --dryrun 也无法使用，跳过 %s 校验（以真实编译为准）" % full)]
        if suf:
            return [R("nvcc-arch", WARN, "%s：基础架构在 list 里，但后缀 %s 不在 list 输出中且 --dryrun 不可用，无法确认" % (full, suf),
                      "直接编译一个空 kernel 确认；或去掉后缀")]
        res.append(R("nvcc-arch", OK, "%s 在 --list-gpu-arch / --list-gpu-code 中（--dryrun 不可用，未二次确认）" % full))
    elif not have_list:
        res.append(R("nvcc-arch", OK, "%s 被接受（该 nvcc 没有 --list-gpu-arch，改用 nvcc --dryrun 判断）" % full))
    elif suf:
        res.append(R("nvcc-arch", OK, "%s：后缀 %s 不在 --list 输出中，已用 nvcc --dryrun 确认被接受" % (full, suf)))
    else:
        res.append(R("nvcc-arch", OK, "%s 在 --list-gpu-arch / --list-gpu-code 中，nvcc --dryrun 也接受" % full))
    if re.search(r"Cannot find compiler|cannot find compiler", text):
        res.append(R("nvcc-host", WARN, "nvcc --dryrun 提示找不到 host 编译器：" + text[:80],
                     "Windows 需在“x64 Native Tools Command Prompt”里运行或把 cl.exe 加进 PATH；Linux 检查 gcc/g++ 版本是否在 nvcc 支持范围"))
    return res


def check_arch(nvcc, arch_arg):
    if not arch_arg:
        return [R("nvcc-arch", SKIP, "未给 --arch，跳过架构校验")]
    if not nvcc:
        return [R("nvcc-arch", SKIP, "nvcc 不可用，无法校验 --arch " + arch_arg)]
    rc1, o1, _ = run_cmd([nvcc, "--list-gpu-arch"])
    rc2, o2, _ = run_cmd([nvcc, "--list-gpu-code"])
    comp = _parse_arch_list(o1) if rc1 == 0 else set()
    code = _parse_arch_list(o2) if rc2 == 0 else set()
    out, host_seen = [], False
    for a in [x.strip() for x in arch_arg.split(",") if x.strip()]:
        for r in check_arch_one(nvcc, a, comp, code):
            if r["name"] == "nvcc-host":      # host 编译器缺失只报一次，不随架构个数重复
                if host_seen:
                    continue
                host_seen = True
            out.append(r)
    return out


# ---------------------------------------------------------------- nsys
def check_nsys(need):
    bad = FAIL if need else WARN
    path = shutil.which("nsys")
    if not path:
        return R("nsys", bad, "nsys 不在 PATH" + ("（--need-nsys 要求必须有）" if need else "（可选）"),
                 "装 Nsight Systems 并加进 PATH；没有 nsys 时仍可只用 fi_lite_decode.py 出软件时间线，但不能用 fi_align_nsys.py 对齐")
    rc, out, err = run_cmd([path, "--version"])
    if rc != 0:
        return R("nsys", bad, "nsys --version 失败（rc=%s）：%s" % (rc, first_line(err or out, 80)), "检查 Nsight Systems 安装", stderr=first_line(err, 300))
    return R("nsys", OK, "%s，%s" % (first_line(out, 80), path))


# ---------------------------------------------------------------- nvidia-smi / GPU
def _num(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):  # "[N/A]" / "[Not Supported]"
        return None


def query_gpus(smi):
    """返回 (GPU 列表或 None, 失败原因, 捕获的 stderr)。每个 GPU 是 dict(index, uuid, name, driver, util, mem_used, mem_total)。"""
    rc, out, err = run_cmd([smi, "--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"])
    if rc != 0:
        return None, first_line(err or out, 120) or "rc=%s" % rc, first_line(err, 300)
    gpus = []
    for ln in out.splitlines():
        if not ln.strip():
            continue
        p = [x.strip() for x in ln.split(",")]
        if len(p) < 7:
            continue
        gpus.append(dict(index=p[0], uuid=p[1], name=",".join(p[2:-4]), driver=p[-4], util=_num(p[-3]),
                         mem_used=_num(p[-2]), mem_total=_num(p[-1])))
    if not gpus:
        return None, "nvidia-smi 没有列出任何 GPU", ""
    return gpus, "", ""


def query_procs(smi, uuid):
    """某张卡上的计算进程 [(pid, 名字)]；查不到返回 []。"""
    rc, out, _ = run_cmd([smi, "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    procs = []
    if rc == 0:
        for ln in out.splitlines():
            p = [x.strip() for x in ln.split(",")]
            if len(p) >= 4 and p[0] == uuid:
                procs.append((p[1], os.path.basename(",".join(p[2:-1]).replace("\\", "/"))))
    return procs


def _fmt_gpu(g):
    u = "N/A" if g["util"] is None else "%d%%" % g["util"]
    m = "N/A" if g["mem_used"] is None else "%d" % g["mem_used"]
    return "%s=%s(util %s, mem %s/%s MiB)" % (g["index"], g["name"], u, m, g["mem_total"])


def check_gpu(gpu_id, busy_util, busy_mem):
    smi = shutil.which("nvidia-smi")
    if not smi:
        return [R("nvidia-smi", FAIL, "nvidia-smi 不在 PATH，无法判断 GPU 是否可用 / 是否被占用",
                  "确认在有 NVIDIA 驱动的 GPU 机器上运行（本机无 GPU 或驱动时需换到 GPU 机器）；驱动装好后 nvidia-smi 通常在 PATH 里"),
                R("gpu-list", SKIP, "依赖 nvidia-smi"), R("gpu-busy", SKIP, "依赖 nvidia-smi")]
    gpus, why, serr = query_gpus(smi)
    if gpus is None:
        return [R("nvidia-smi", FAIL, "nvidia-smi 不能正常查询：" + why,
                  "驱动没加载 / 容器没挂 GPU / 权限不足；先在 shell 里直接跑 nvidia-smi 看原因", stderr=serr),
                R("gpu-list", SKIP, "依赖 nvidia-smi"), R("gpu-busy", SKIP, "依赖 nvidia-smi")]
    res = [R("nvidia-smi", OK, "%s，驱动 %s" % (smi, gpus[0]["driver"])),
           R("gpu-list", OK, "共 %d 张：" % len(gpus) + "；".join(_fmt_gpu(g) for g in gpus))]
    if gpu_id is None:
        res.append(R("gpu-busy", SKIP, "未给 --gpu，跳过占用检查"))
        return res
    g = next((x for x in gpus if x["index"] == str(gpu_id)), None)
    if g is None:
        res.append(R("gpu-busy", FAIL, "GPU %s 不存在（nvidia-smi 编号 %s）" % (gpu_id, ",".join(x["index"] for x in gpus)),
                     "改 --gpu；注意这里是 nvidia-smi 编号，与 CUDA_VISIBLE_DEVICES 重排后的编号可能不同"))
        return res
    busy_by = []
    if g["util"] is not None and g["util"] > busy_util:
        busy_by.append("util %d%% > %d%%" % (g["util"], busy_util))
    if g["mem_used"] is not None and g["mem_used"] > busy_mem:
        busy_by.append("mem %d MiB > %d MiB" % (g["mem_used"], busy_mem))
    procs = query_procs(smi, g["uuid"])
    ptxt = "、".join("pid %s(%s)" % p for p in procs[:5])
    if busy_by:
        idle = [x["index"] for x in gpus if x["index"] != g["index"] and (x["util"] or 0) <= busy_util and (x["mem_used"] or 0) <= busy_mem]
        res.append(R("gpu-busy", FAIL, "GPU %s 被占用（%s）%s" % (g["index"], "；".join(busy_by), "，占用进程：" + ptxt if ptxt else ""),
                     "不要 kill 别人的进程；走 GPU 锁排队等空闲，或换空闲卡（当前看起来空闲的：%s）" % (",".join(idle) or "无")))
        return res
    note = ""
    if procs:
        note = "；卡上有 %d 个计算进程（%s）但未超阈值" % (len(procs), ptxt)
    elif g["util"] is None:
        note = "；利用率读不到，仅按显存判断"
    res.append(R("gpu-busy", OK, "GPU %s 空闲（util %s，mem %s MiB；阈值 util>%d%% 或 mem>%d MiB）%s" % (
        g["index"], "N/A" if g["util"] is None else "%d%%" % g["util"], g["mem_used"], busy_util, busy_mem, note)))
    return res


# ---------------------------------------------------------------- GPU 锁目录
def norm_path(p):
    """规范化路径用于比较：展开 ~、解析软链接、统一分隔符、Windows 下忽略大小写。"""
    return os.path.normcase(os.path.realpath(os.path.expanduser(p)))


def show_path(p):
    """消息里显示用：解析软链接但保留大小写，便于人看。"""
    return os.path.realpath(os.path.expanduser(p))


def check_lock(lock_dir, expect):
    res = []
    if lock_dir is None:
        if expect is not None:
            return [R("lock-dir", FAIL, "给了 --expect-lock-dir 但没给 --lock-dir，无法比较", "同时给出 --lock-dir <你实际要用的锁目录>")]
        return [R("lock-dir", SKIP, "未给 --lock-dir，跳过锁目录检查")]
    if not os.path.isdir(lock_dir):
        res.append(R("lock-dir", FAIL, "锁目录不存在或不是目录：" + lock_dir, "确认路径（共享盘是否已挂载）；不要自己 mkdir 一个新目录顶替共享锁目录"))
    elif not os.access(lock_dir, os.W_OK | os.X_OK):
        res.append(R("lock-dir", FAIL, "锁目录不可写：" + lock_dir, "检查权限 / 共享盘是否只读挂载"))
    else:
        res.append(R("lock-dir", OK, "存在且可写（os.access）：" + lock_dir))
    if expect is not None:
        if norm_path(lock_dir) == norm_path(expect):
            res.append(R("lock-dir-match", OK, "与期望共享锁目录一致：" + show_path(expect)))
        else:
            res.append(R("lock-dir-match", FAIL, "锁目录 %s 与期望 %s 不一致" % (show_path(lock_dir), show_path(expect)),
                         "锁目录没指到共享目录会导致“锁外占卡”（别人看不到你的锁）；把 --lock-dir 改成期望的共享目录"))
    return res



# ---------------------------------------------------------------- AMD（gfx942 / ROCm）
def is_amd_arch(arch):
    return bool(arch) and re.search(r"gfx[0-9a-f]+", str(arch), re.I) is not None


def platform_of(arch=None, platform=None):
    """Legacy host probe used only behind --inspect-host.

    It never defaults to NVIDIA: when neither the target nor host tools provide
    evidence, it returns ``None`` so the registry can fail closed.
    """
    if platform and platform != "auto":
        return platform
    if is_amd_arch(arch):
        return "amd"
    if arch and re.search(r"(sm_|compute_)?[0-9]{2,3}[af]?$", str(arch).strip()):
        return "nvidia"
    if shutil.which("hipcc") or shutil.which("rocm-smi"):
        return "amd"
    return None


def _argument_given(value):
    """True only for a real user-supplied target value."""
    return value is not None and str(value).strip().lower() not in (
        "", "auto", "none", "unknown")


def _host_card_from_name(name):
    text = str(name or "").upper()
    if re.search(r"(^|[^A-Z0-9])H20([^A-Z0-9]|$)", text):
        return "h20"
    if re.search(r"(^|[^A-Z0-9])B300([^A-Z0-9]|$)", text):
        return "b300"
    return None


def _host_arch_from_compute_cap(value):
    match = re.search(r"\b(\d+)\.(\d+)\b", str(value or ""))
    if not match:
        return None
    return {
        (9, 0): "sm_90",
        (10, 0): "sm_100a",
        (10, 3): "sm_103a",
    }.get((int(match.group(1)), int(match.group(2))))


def _query_nvidia_host():
    """Best-effort explicit host inspection; never used unless --inspect-host."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"platform": "nv", "card": None, "arch": None,
                "evidence": ["nvidia-smi not found"]}
    last_error = ""
    for fields in ("name,compute_cap", "name"):
        rc, out, err = run_cmd([smi, "--query-gpu=" + fields,
                                "--format=csv,noheader,nounits"], timeout=30)
        if rc != 0:
            last_error = first_line(err or out, 120)
            continue
        line = next((item.strip() for item in out.splitlines() if item.strip()), "")
        if not line:
            last_error = "nvidia-smi returned no GPU rows"
            continue
        fields_out = [item.strip() for item in line.split(",")]
        name = fields_out[0] if fields_out else ""
        cap = fields_out[1] if len(fields_out) > 1 else ""
        card = _host_card_from_name(name)
        arch = _host_arch_from_compute_cap(cap)
        if card == "h20" and not arch:
            arch = "sm_90"
        return {"platform": "nv", "card": card, "arch": arch,
                "gpu_name": name, "compute_cap": cap,
                "evidence": ["nvidia-smi --query-gpu=" + fields]}
    return {"platform": "nv", "card": None, "arch": None,
            "evidence": ["nvidia-smi query failed: " + (last_error or "unknown")]}


def _query_amd_host():
    """Best-effort explicit host inspection; never used unless --inspect-host."""
    rocminfo = shutil.which("rocminfo")
    if not rocminfo:
        return {"platform": "amd", "card": None, "arch": None,
                "evidence": ["rocminfo not found"]}
    rc, out, err = run_cmd([rocminfo], timeout=60)
    if rc != 0:
        return {"platform": "amd", "card": None, "arch": None,
                "evidence": ["rocminfo failed: " + first_line(err or out, 120)]}
    archs = sorted(set(match.group(1).lower()
                       for match in re.finditer(r"\b(gfx[0-9a-f]+)\b", out, re.I)))
    marketing = ""
    match = re.search(r"Marketing Name:\s*([^\r\n]+)", out, re.I)
    if match:
        marketing = match.group(1).strip()
    arch = "gfx942" if "gfx942" in archs else (archs[0] if len(archs) == 1 else None)
    card = "mi308x" if "MI308X" in marketing.upper() or arch == "gfx942" else None
    return {"platform": "amd", "card": card, "arch": arch,
            "marketing_name": marketing, "architectures_seen": archs,
            "evidence": ["rocminfo"]}


def inspect_host_target(platform_hint=None):
    """Legacy host auto-detection, now reachable only via --inspect-host."""
    platform = backend_registry.normalize_platform(platform_hint)
    if platform is None:
        platform = backend_registry.normalize_platform(platform_of(None, None))
    if platform is None:
        raise backend_registry.UnsupportedTarget(
            "host inspection could not identify a supported GPU platform; "
            "pass --card/--arch explicitly",
            kind="host_platform_unresolved")
    record = _query_nvidia_host() if platform == "nv" else _query_amd_host()
    record["requested_platform"] = platform
    return record


def resolve_preflight_target(args, inspect_host=None):
    """Resolve one registry route for preflight, or fail closed.

    No host command or default platform is consulted unless the caller supplied
    --inspect-host.  In inspect mode, host evidence fills only missing card/arch
    fields; explicit values remain authoritative and conflicts are rejected by
    the registry.
    """
    card = getattr(args, "card", None)
    arch = getattr(args, "arch", None)
    platform = getattr(args, "platform", None)
    language = getattr(args, "language", None)
    do_inspect = bool(getattr(args, "inspect_host", False)
                      if inspect_host is None else inspect_host)
    card_given = _argument_given(card)
    arch_given = _argument_given(arch)
    platform_given = _argument_given(platform)
    if not card_given and not arch_given and not do_inspect:
        raise backend_registry.UnsupportedTarget(
            "target is unspecified; pass --card or --arch (and optionally --language), "
            "or explicitly opt into host inspection with --inspect-host",
            kind="target_required",
            details={"card": card, "arch": arch, "platform": platform})

    host = {}
    if do_inspect and not (card_given and arch_given):
        host = inspect_host_target(platform if platform_given else None)
        if not card_given and not arch_given and not (host.get("card") or host.get("arch")):
            raise backend_registry.UnsupportedTarget(
                "host inspection did not identify a supported card or architecture; "
                "pass --card or --arch explicitly",
                kind="host_target_unresolved", details=host)
        card = card if _argument_given(card) else host.get("card")
        arch = arch if _argument_given(arch) else host.get("arch")
        platform = platform if _argument_given(platform) else host.get("platform")

    card = card if _argument_given(card) else None
    arch = arch if _argument_given(arch) else None
    platform = platform if _argument_given(platform) else None

    target_preview = backend_registry.resolve_target(
        card=card, arch=arch, platform=platform)
    language_given = _argument_given(language)
    selected_language = language if language_given else \
        backend_registry.CARD_SPECS[target_preview["card"]]["languages"][0]
    route = backend_registry.resolve(
        card=card, arch=arch, platform=platform, language=selected_language)
    route["language_resolution"] = "explicit" if language_given else "card_default"
    route["host_inspection"] = host or None
    setattr(args, "_host_inspection", host or None)
    return route


def check_hipcc():
    path = shutil.which("hipcc")
    if not path:
        return None, R("hipcc", FAIL, "hipcc 不在 PATH",
                       "装 ROCm 并把 /opt/rocm/bin 加进 PATH；或 export HIPCC=/path/to/hipcc 后重试")
    rc, out, err = run_cmd([path, "--version"])
    if rc != 0:
        return path, R("hipcc", FAIL, "hipcc --version 失败（rc=%s）：%s" % (rc, first_line(err or out, 80)),
                       "检查 ROCm 安装是否完整", stderr=first_line(err, 300))
    ver = first_line(out, 80) or first_line(err, 80)
    return path, R("hipcc", OK, "%s，%s" % (ver, path))


def check_hip_arch(arch):
    """rocminfo 给出本机 agent 的 ISA 名（形如 gfx942:sramecc+:xnack-）；没有 rocminfo 就 SKIP，以真实编译为准。"""
    if not arch:
        return [R("hip-arch", SKIP, "未给 --arch，跳过架构校验")]
    want = set()
    for x in str(arch).split(","):
        m = re.search(r"(gfx[0-9a-f]+)", x.strip(), re.I)
        if m:
            want.add(m.group(1).lower())
    if not want:
        return [R("hip-arch", FAIL, "--arch %r 不是 gfx 号" % arch, "AMD 侧写成 gfx942 这种形式；NVIDIA 侧用 --platform nvidia")]
    ri = shutil.which("rocminfo")
    if not ri:
        return [R("hip-arch", SKIP, "rocminfo 不在 PATH，无法核对本机 ISA 是否包含 %s（以真实编译为准）" % ",".join(sorted(want)),
                  "装 ROCm 后 rocminfo 在 /opt/rocm/bin；也可以直接编一个空 kernel 验证 --offload-arch")]
    rc, out, err = run_cmd([ri], timeout=60)
    if rc != 0:
        return [R("hip-arch", SKIP, "rocminfo 运行失败（rc=%s），跳过架构校验" % rc,
                  " rocminfo 需要能访问 /dev/kfd 与 /dev/dri；被锁或权限不足时先看 dmesg", stderr=first_line(err, 300))]
    have = set(m.group(1).lower() for m in re.finditer(r"\b(gfx[0-9a-f]+)", out))
    if not have:
        return [R("hip-arch", SKIP, "rocminfo 输出里没有 gfx ISA 名（可能只有 CPU agent），跳过架构校验",
                  "看 rocminfo 全文里 Agent 的 Name 字段")]
    miss = sorted(want - have)
    if miss:
        return [R("hip-arch", FAIL, "--arch %s 不在本机 rocminfo 的 ISA 列表里（本机有 %s）" % (",".join(miss), ",".join(sorted(have))),
                  "改用本机真实 ISA；跨架构只能编不能跑")]
    return [R("hip-arch", OK, "%s 在本机 rocminfo 的 ISA 列表里（%s）" % (",".join(sorted(want)), ",".join(sorted(have))))]


def find_llvm_objdump(rocm_llvm=None):
    """找 llvm-objdump，返回 (路径, 来源说明)；找不到时路径是 None。

    顺序是 FI_LLVM_OBJDUMP -> --rocm-llvm -> /opt/rocm*/lib/llvm/bin -> /opt/rocm*/bin ->
    /usr/lib/llvm-*/bin -> PATH（写死 /opt/rocm-<版本>/bin/llvm-objdump 在多数机器上都是错的，
    真实位置常在 lib/llvm/bin）。

    **显式指定过（环境变量或 --rocm-llvm）就只认它**：那里没有可执行的 llvm-objdump 就直接返回 None，
    不再静默回退到别处的另一个 llvm-objdump——回退会让人以为用的是自己指的那个，版本/目标架构可能完全不同。
    只有在什么都没指的时候才按安装位置与 PATH 回退。
    （fi_bugs_b3-sk-r2ds BUG-001：mail-17 上 /usr/lib/llvm-* 里有 llvm-objdump，
    于是"显式指到空目录"也返回 OK，--selftest 的那条断言在别的机器上就挂了。）
    """
    import glob as _glob
    explicit = []
    env = os.environ.get("FI_LLVM_OBJDUMP")
    if env:
        explicit.append((env, "环境变量 FI_LLVM_OBJDUMP"))
    if rocm_llvm:
        explicit.append((os.path.join(rocm_llvm, "llvm-objdump"), "--rocm-llvm " + rocm_llvm))
        explicit.append((rocm_llvm, "--rocm-llvm（直接指到可执行文件）"))
    fallback = []
    for pat in ("/opt/rocm*/lib/llvm/bin/llvm-objdump", "/opt/rocm*/llvm/bin/llvm-objdump",
                "/opt/rocm*/bin/llvm-objdump", "/usr/lib/llvm-*/bin/llvm-objdump"):
        for h in sorted(_glob.glob(pat)):
            fallback.append((h, "安装位置 " + pat))
    w = shutil.which("llvm-objdump")
    if w:
        fallback.append((w, "PATH"))
    for path, why in (explicit or fallback):
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path, why
    tried = "、".join(dict((w2, 1) for _, w2 in (explicit or fallback)).keys())
    if explicit:
        return None, ("显式指定的位置里没有可执行的 llvm-objdump（试过：%s）；不再回退到别处的另一个。"
                      "确认路径对不对，或去掉 FI_LLVM_OBJDUMP / --rocm-llvm 让它按安装位置与 PATH 自己找" % tried)
    return None, ("试过：%s" % tried) if (explicit or fallback) else "没有候选路径"


def check_llvm_objdump(rocm_llvm=None):
    path, why = find_llvm_objdump(rocm_llvm)
    if not path:
        return R("llvm-objdump", FAIL, "找不到 llvm-objdump（%s）" % why,
                 "C1 回查二进制需要它：export FI_LLVM_OBJDUMP=/opt/rocm/lib/llvm/bin/llvm-objdump，或给 --rocm-llvm 指到 llvm 的 bin 目录")
    rc, out, err = run_cmd([path, "--version"], timeout=30)
    tgt = first_line(out, 200)
    if rc != 0:
        return R("llvm-objdump", WARN, "%s 存在但 --version 失败（rc=%s）" % (path, rc), "换 ROCm 自带的 llvm-objdump", stderr=first_line(err, 200))
    return R("llvm-objdump", OK, "%s（%s）：%s" % (path, why, (tgt or "版本未知")[:80]))


def check_rocprof(need=False):
    bad = FAIL if need else WARN
    for name in ("rocprofv3", "rocprof"):
        path = shutil.which(name)
        if path:
            rc, out, err = run_cmd([path, "--version"], timeout=30)
            txt = first_line(out, 80) or first_line(err, 80)
            if rc == 0:
                return R("rocprofv3", OK, "%s，%s" % (txt, path))
            return R("rocprofv3", OK, "%s 在 PATH（--version rc=%s，不影响使用）：%s" % (name, rc, path))
    return R("rocprofv3", bad, "rocprofv3 / rocprof 都不在 PATH" + ("（--need-nsys 要求必须有）" if need else "（可选）"),
             "没有它就只能出软件时间线（fi_lite_decode.py），不能做 C4 的硬件计数器关联；装 ROCm 后在 /opt/rocm/bin")


def _amd_smi_parse(text, gpu_id):
    """从 rocm-smi 输出里挑出该卡的 GPU 利用率与显存占用。格式跨 ROCm 版本差别很大，
    解析不出来就返回 None（调用方记 SKIP，不猜）。"""
    use = mem = None
    for ln in (text or "").splitlines():
        if gpu_id is not None and not re.search(r"(\b|card|GPU\[?)0*%s\b" % gpu_id, ln, re.I):
            continue
        m = re.search(r"GPU use \(%\):\s*([0-9]+)", ln) or re.search(r"GPU utilization[^0-9]*([0-9]+)\s*%", ln, re.I) \
            or re.search(r"\b([0-9]{1,3})\s*%\b", ln)
        if m and use is None and re.search(r"use|util", ln, re.I):
            use = int(m.group(1))
        m2 = re.search(r"VRAM Total Used Memory \(B\):\s*([0-9]+)", ln) or re.search(r"Used Memory[^0-9]*([0-9]+)", ln, re.I)
        if m2 and mem is None:
            mem = int(m2.group(1))
    return use, mem


def check_gpu_amd(gpu_id, busy_util):
    smi = shutil.which("rocm-smi")
    if not smi:
        return [R("gpu-list", SKIP, "rocm-smi 不在 PATH，无法列 GPU", "装 ROCm 后 rocm-smi 在 /opt/rocm/bin；也可以直接看锁目录判断卡是否被占"),
                R("gpu-busy", SKIP, "rocm-smi 不可用，跳过占用判断", "以 GPU 锁为准：持锁后再上卡，绝不 kill 别人的进程")]
    rc, out, err = run_cmd([smi, "--showuse", "--showmeminfo", "vram"], timeout=60)
    if rc != 0:
        return [R("gpu-list", SKIP, "rocm-smi --showuse 失败（rc=%s）：%s" % (rc, first_line(err or out, 80)),
                  "换 rocm-smi --showuse --showmeminfo vram 手工看一眼", stderr=first_line(err, 300)),
                R("gpu-busy", SKIP, "rocm-smi 输出不可用，跳过占用判断", "以 GPU 锁为准")]
    use, mem = _amd_smi_parse(out, gpu_id)
    if use is None:
        return [R("gpu-list", WARN, "rocm-smi 输出没能解析出 GPU 利用率（格式与预期不同）", 
                  "把 rocm-smi --showuse 的原文贴进报告，用 GPU 锁判断占用", stderr=first_line(out, 300)),
                R("gpu-busy", SKIP, "解析不出利用率，跳过占用判断", "以 GPU 锁为准")]
    mem_mib = int(mem / (1024 * 1024)) if mem else None
    busy = use > busy_util
    pids = ""
    rc2, out2, _ = run_cmd([smi, "--showpids"], timeout=60)
    if rc2 == 0:
        rows = [ln.strip() for ln in out2.splitlines() if re.search(r"^[0-9]+\s+", ln.strip())]
        pids = "；在跑的 PID：%s" % ", ".join(r.split()[0] for r in rows[:6]) if rows else ""
    if busy:
        return [R("gpu-list", OK, "rocm-smi 可用（util %s%%，mem %s MiB）" % (use, mem_mib if mem_mib is not None else "?")),
                R("gpu-busy", FAIL, "GPU %s 看起来被占用（util %s%% > %d%%%s）" % (gpu_id, use, busy_util, pids),
                  "排队等 GPU 锁，绝不 kill 别人的进程；确认没人用再上卡")]
    return [R("gpu-list", OK, "rocm-smi 可用（util %s%%，mem %s MiB）" % (use, mem_mib if mem_mib is not None else "?")),
            R("gpu-busy", OK, "GPU %s 空闲（util %s%%，阈值 %d%%%s）" % (gpu_id, use, busy_util, pids))]


# ---------------------------------------------------------------- 汇总
def run_all(a, route):
    target = route["target"]
    plat = target["platform"]
    arch = target.get("arch")
    profiler = route.get("profiler") or {}
    res = [R("target-route", OK,
             "card=%s platform=%s arch=%s language=%s recorder=%s profiler=%s evidence=%s" % (
                 target.get("card"), plat,
                 arch or ("set:" + ",".join(target.get("architectures") or [])),
                 route.get("language"), route.get("recorder"),
                 profiler.get("primary"), route.get("evidence_status"))),
           check_python(), check_stdlib(), check_matplotlib()]
    if plat == "amd":
        _hipcc, r = check_hipcc()
        res.append(r)
        res += check_hip_arch(arch)
        res.append(check_llvm_objdump(getattr(a, "rocm_llvm", None)))
        res.append(check_rocprof(a.need_nsys))
        res += check_gpu_amd(a.gpu, a.busy_util)
    else:
        nvcc, r = check_nvcc()
        res.append(r)
        res += check_arch(nvcc, arch)
        res.append(check_nsys(a.need_nsys))
        res += check_gpu(a.gpu, a.busy_util, a.busy_mem)
    res += check_lock(a.lock_dir, a.expect_lock_dir)
    return res


def fmt_line(r):
    s = "[%-4s] %-14s %s" % (r["status"], r["name"], r["msg"])
    if r["fix"] and r["status"] in (WARN, FAIL):
        s += " | 修法: " + r["fix"]
    return s


def verdict_of(results):
    return "FAIL" if any(r["status"] == FAIL for r in results) else "PASS"


def report(results, verbose):
    for r in results:
        print(fmt_line(r))
        if verbose and r["stderr"]:
            print("       [捕获的 stderr] " + r["stderr"])
    cnt = dict((s, sum(1 for r in results if r["status"] == s)) for s in (OK, WARN, FAIL, SKIP))
    print("汇总：OK %(OK)d，WARN %(WARN)d，FAIL %(FAIL)d，SKIP %(SKIP)d" % cnt)
    v = verdict_of(results)
    print("PREFLIGHT " + v)
    return v, cnt


def selftest():
    """用假的 run_cmd / which 验证解析与判定逻辑（不依赖本机有无 nvcc / GPU；输入是手工构造的命令输出，不是测量数据）。"""
    g = globals()
    real_run, real_which = g["run_cmd"], shutil.which

    def fake(table):
        def run(argv, timeout=30):
            key = " ".join(argv[1:])
            for k, v in table:
                if k in key:
                    return v
            return (None, "", "找不到程序")
        return run

    try:
        # 1. nvcc 版本与架构
        shutil.which = lambda n: "/fake/" + n
        nvcc_tbl = [("--version", (0, "Cuda compilation tools, release 12.8, V12.8.93\n", "")),
                    ("--list-gpu-arch", (0, "compute_80\ncompute_90\ncompute_100\ncompute_103\n", "")),
                    ("--list-gpu-code", (0, "sm_80\nsm_90\nsm_100\nsm_103\n", "")),
                    ("-arch=sm_104a", (1, "", "nvcc fatal   : Unsupported gpu architecture 'sm_104a'")),
                    ("--dryrun", (1, "", "nvcc fatal   : Cannot find compiler 'cl.exe' in PATH"))]
        g["run_cmd"] = fake(nvcc_tbl)
        p, r = check_nvcc()
        assert r["status"] == OK and "12.8" in r["msg"] and "V12.8.93" in r["msg"], r
        arch = lambda a: check_arch(p, a)
        sts = lambda rs: [(x["name"], x["status"]) for x in rs]
        # 假 nvcc 的 --dryrun 一律回“找不到 cl.exe”（架构被接受 + host 编译器缺失），所以每个可用架构都带一条 nvcc-host 警告
        assert sts(arch("sm_103")) == [("nvcc-arch", OK), ("nvcc-host", WARN)], arch("sm_103")
        assert sts(arch("compute_90"))[0] == ("nvcc-arch", OK) and sts(arch("103"))[0] == ("nvcc-arch", OK)
        r103a = arch("sm_103a")                       # 带后缀：基础在 list，dryrun 不报 Unsupported -> OK
        assert sts(r103a) == [("nvcc-arch", OK), ("nvcc-host", WARN)] and "后缀 a" in r103a[0]["msg"], r103a
        r104a = arch("sm_104a")                       # list 里没有 104 -> 直接 FAIL（修法里列出支持的架构）
        assert sts(r104a) == [("nvcc-arch", FAIL)] and "sm_103" in r104a[0]["fix"], r104a
        assert arch("sm_999")[0]["status"] == FAIL and arch("garbage")[0]["status"] == FAIL
        assert sts(arch("sm_80,sm_90")) == [("nvcc-arch", OK), ("nvcc-host", WARN), ("nvcc-arch", OK)]  # host 警告只报一次
        # list 说有、dryrun 说不支持（带后缀的新架构）-> FAIL
        g["run_cmd"] = fake([("--version", (0, "release 12.8, V12.8.93", "")), ("-arch=sm_103a", (1, "", "nvcc fatal : Unsupported gpu architecture 'sm_103a'")),
                             ("--list-gpu-arch", (0, "compute_103\n", "")), ("--list-gpu-code", (0, "sm_103\n", ""))])
        assert sts(check_arch("/fake/nvcc", "sm_103a")) == [("nvcc-arch", FAIL)]
        assert sts(check_arch("/fake/nvcc", "sm_103")) == [("nvcc-arch", OK)]  # 此表里 dryrun 对 sm_103 无匹配 -> unknown -> 仍按 list 判 OK
        # dryrun 不可用但 list 可用：不带后缀按 list 判 OK；带后缀无法确认 -> WARN
        g["run_cmd"] = fake([("--list-gpu-arch", (0, "compute_103\n", "")), ("--list-gpu-code", (0, "sm_103\n", "")), ("--dryrun", (1, "", "Unknown option '--dryrun'"))])
        assert sts(check_arch("/fake/nvcc", "sm_103")) == [("nvcc-arch", OK)] and sts(check_arch("/fake/nvcc", "sm_103a")) == [("nvcc-arch", WARN)]
        # 没有 --list-gpu-arch 子命令：改用 dryrun；dryrun 也不可用则 SKIP
        old = [("--list-gpu-arch", (1, "", "nvcc fatal : Unknown option '--list-gpu-arch'")), ("--list-gpu-code", (1, "", "Unknown option"))]
        g["run_cmd"] = fake(old + [("-arch=sm_70", (0, "", "")), ("-arch=sm_999", (1, "", "Unsupported gpu architecture 'sm_999'"))])
        assert check_arch("/fake/nvcc", "sm_70")[0]["status"] == OK and check_arch("/fake/nvcc", "sm_999")[0]["status"] == FAIL
        g["run_cmd"] = fake(old + [("--dryrun", (1, "", "nvcc fatal : Unknown option '--dryrun'"))])
        assert check_arch("/fake/nvcc", "sm_70")[0]["status"] == SKIP
        assert check_arch(None, "sm_70")[0]["status"] == SKIP and check_arch("/fake/nvcc", None)[0]["status"] == SKIP
        shutil.which = lambda n: None
        assert check_nvcc()[1]["status"] == FAIL

        # 2. nsys：缺失只警告；--need-nsys 变失败
        assert check_nsys(False)["status"] == WARN and check_nsys(True)["status"] == FAIL
        shutil.which = lambda n: "/fake/" + n
        g["run_cmd"] = fake([("--version", (0, "NVIDIA Nsight Systems version 2025.3.1.90\n", ""))])
        assert check_nsys(True)["status"] == OK

        # 3. nvidia-smi：空闲 / 被占用（利用率）/ 被占用（显存）/ 读不到利用率 / 卡号不存在 / 命令失败 / 不存在
        csv = ("0, GPU-aaa, NVIDIA B300, 580.1, 0, 12, 183359\n"
               "1, GPU-bbb, NVIDIA B300, 580.1, 87, 4096, 183359\n"
               "2, GPU-ccc, NVIDIA B300, 580.1, 0, 20480, 183359\n"
               "3, GPU-ddd, NVIDIA GeForce RTX 3050 Laptop GPU, 580.1, [N/A], 9, 4096\n")
        procs = "GPU-bbb, 4242, /usr/bin/python3, 4000\nGPU-bbb, 4243, /opt/x/train.py, 96\n"
        g["run_cmd"] = fake([("--query-gpu", (0, csv, "")), ("--query-compute-apps", (0, procs, ""))])
        st = lambda i: [x for x in check_gpu(i, 5, 1024) if x["name"] == "gpu-busy"][0]
        assert "共 4 张" in check_gpu(0, 5, 1024)[1]["msg"]
        assert st(0)["status"] == OK
        b = st(1)
        assert b["status"] == FAIL and "util 87%" in b["msg"] and "pid 4242" in b["msg"] and "不要 kill" in b["fix"] and "0" in b["fix"], b
        assert st(2)["status"] == FAIL and "mem 20480" in st(2)["msg"]
        assert st(3)["status"] == OK and "仅按显存" in st(3)["msg"]
        assert st(9)["status"] == FAIL and "不存在" in st(9)["msg"]
        assert st(None)["status"] == SKIP
        g["run_cmd"] = fake([("--query-gpu", (9, "", "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver\n"))])
        x = check_gpu(0, 5, 1024)
        assert x[0]["status"] == FAIL and "failed" in x[0]["msg"] and x[0]["stderr"], x
        shutil.which = lambda n: None
        x = check_gpu(0, 5, 1024)
        assert [y["status"] for y in x] == [FAIL, SKIP, SKIP]

        # 4. 锁目录
        here = os.path.dirname(os.path.abspath(__file__))
        assert check_lock(None, None)[0]["status"] == SKIP
        assert check_lock(None, here)[0]["status"] == FAIL
        assert [x["status"] for x in check_lock(here, here)] == [OK, OK]
        assert [x["status"] for x in check_lock(here + os.sep, here)] == [OK, OK]          # 末尾分隔符不影响
        assert [x["status"] for x in check_lock(os.path.dirname(here), here)] == [OK, FAIL]  # 目录存在但不是期望的共享目录
        assert check_lock(os.path.join(here, "不存在的锁目录"), None)[0]["status"] == FAIL
        assert check_lock(here, None)[0]["status"] == OK

        # 4b. AMD 分支：hipcc / rocminfo / llvm-objdump / rocprofv3 / rocm-smi；且不碰 nvcc、nvidia-smi、nsys
        import tempfile
        tdir = tempfile.mkdtemp(prefix="fi_preflight_selftest_")
        fake_objdump = os.path.join(tdir, "llvm-objdump")
        with open(fake_objdump, "w", encoding="utf-8") as fh:
            fh.write("#!/bin/sh\necho 'LLVM (http://llvm.org/):\n  LLVM version 20.0.0.git\n'\n")
        os.chmod(fake_objdump, 0o755)

        amd_which = {"hipcc": "/opt/rocm/bin/hipcc", "rocminfo": "/opt/rocm/bin/rocminfo",
                     "rocprofv3": "/opt/rocm/bin/rocprofv3", "rocm-smi": "/opt/rocm/bin/rocm-smi",
                     "rocprof": None, "nvcc": None, "nvidia-smi": None, "nsys": None, "llvm-objdump": None}
        shutil.which = lambda n: amd_which.get(n)
        g["run_cmd"] = fake([
            ("--showuse --showmeminfo vram", (0, "GPU[0]\t\t: GPU use (%): 3\nGPU[0]\t\t: VRAM Total Used Memory (B): 1073741824\n", "")),
            ("--showpids", (0, "PID\tNAME\n", "")),
            ("--version", (0, "HIP version: 7.2.0-abcdef\n", "")),
            ("", (0, "Agent 2:\n  Name:                    gfx942\n  Marketing Name:          AMD Instinct MI308X\n", "")),
        ])
        assert platform_of("gfx942") == "amd" and platform_of("sm_90") == "nvidia" and platform_of(None, "amd") == "amd"
        assert platform_of(None) == "amd", "本机只有 hipcc/rocm-smi 时应判 amd"
        hipcc, r_hip = check_hipcc()
        assert r_hip["status"] == OK and "7.2.0" in r_hip["msg"], r_hip
        assert [x["status"] for x in check_hip_arch("gfx942")] == [OK], check_hip_arch("gfx942")
        bad = check_hip_arch("gfx999")
        assert bad[0]["status"] == FAIL and "gfx942" in bad[0]["msg"], bad
        assert check_hip_arch(None)[0]["status"] == SKIP
        os.environ["FI_LLVM_OBJDUMP"] = fake_objdump
        r_obj = check_llvm_objdump(None)
        assert r_obj["status"] == OK and r_obj["msg"].startswith(fake_objdump), r_obj
        assert "环境变量" in r_obj["msg"], r_obj
        del os.environ["FI_LLVM_OBJDUMP"]
        r_obj2 = check_llvm_objdump(fake_objdump)                 # --rocm-llvm 直接指到可执行文件
        assert r_obj2["status"] == OK and "--rocm-llvm" in r_obj2["msg"], r_obj2
        r_obj3 = check_llvm_objdump(tdir)                          # 指到目录：自己拼 <dir>/llvm-objdump
        assert r_obj3["status"] == OK and "--rocm-llvm" in r_obj3["msg"], r_obj3
        empty = tempfile.mkdtemp(prefix="fi_preflight_empty_")
        assert check_llvm_objdump(empty)["status"] == FAIL, "目录里没有 llvm-objdump 就该 FAIL"
        # 显式指定必须严格：就算 PATH 上另有一个 llvm-objdump，也不许静默回退过去
        # （mail-17/B300 上 /usr/lib/llvm-* 里就有，fi_bugs_b3-sk-r2ds BUG-001 的断言因此在别的机器上挂）
        empty2 = tempfile.mkdtemp(prefix="fi_preflight_empty2_")
        amd_which["llvm-objdump"] = fake_objdump
        r_strict = check_llvm_objdump(empty2)
        assert r_strict["status"] == FAIL and "不再回退" in r_strict["msg"], r_strict
        assert check_llvm_objdump(None)["status"] == OK, "什么都没指时才该用 PATH 上的那个"
        amd_which["llvm-objdump"] = None
        shutil.rmtree(empty, ignore_errors=True)
        shutil.rmtree(empty2, ignore_errors=True)
        r_prof = check_rocprof(False)
        assert r_prof["status"] == OK and "rocprofv3" in r_prof["msg"], r_prof
        assert check_rocprof(True)["status"] == OK
        amd_which["rocprofv3"] = None
        assert check_rocprof(False)["status"] == WARN and check_rocprof(True)["status"] == FAIL
        amd_which["rocprofv3"] = "/opt/rocm/bin/rocprofv3"
        x = check_gpu_amd(0, 5)
        assert [y["status"] for y in x] == [OK, OK] and "3" in x[1]["msg"], x
        g["run_cmd"] = fake([("--showuse", (0, "GPU[0]\t\t: GPU use (%): 97\nGPU[0]\t\t: VRAM Total Used Memory (B): 15032385536\n", "")),
                             ("--showpids", (0, "PID\tNAME\n1234\tpython\n", "")),
                             ("--version", (0, "HIP version: 7.2.0\n", "")),
                             ("", (0, "  Name:                    gfx942\n", ""))])
        x = check_gpu_amd(0, 5)
        assert [y["status"] for y in x] == [OK, FAIL] and "1234" in x[1]["msg"], x
        g["run_cmd"] = fake([("--showuse", (0, "完全看不懂的输出\n", "")), ("--showpids", (1, "", "")),
                             ("--version", (0, "HIP version: 7.2.0\n", "")), ("", (0, "  Name:  gfx942\n", ""))])
        x = check_gpu_amd(0, 5)
        assert [y["status"] for y in x] == [WARN, SKIP], x     # 解析不出来不猜，记 SKIP
        amd_which["rocm-smi"] = None
        x = check_gpu_amd(0, 5)
        assert [y["status"] for y in x] == [SKIP, SKIP], x
        shutil.rmtree(tdir, ignore_errors=True)

        # 5. 汇总判定
        assert verdict_of([R("a", OK, ""), R("b", WARN, ""), R("c", SKIP, "")]) == "PASS"
        assert verdict_of([R("a", OK, ""), R("b", FAIL, "")]) == "FAIL"
        assert "修法" in fmt_line(R("x", FAIL, "m", "f")) and "修法" not in fmt_line(R("x", OK, "m", "f"))
    finally:
        g["run_cmd"], shutil.which = real_run, real_which
    print("SELFTEST PASS")


def _write_json(path, document):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=1)


def _route_document(route):
    target = route["target"]
    return dict(
        target=target,
        language=route["language"],
        language_resolution=route.get("language_resolution"),
        recorder=route["recorder"],
        profiler=route["profiler"],
        evidence_status=route["evidence_status"],
        host_inspection=route.get("host_inspection"),
        route=route,
    )


def main():
    ap = argparse.ArgumentParser(description="weaveprofiler 插桩前环境自检")
    ap.add_argument("--card", help="目标卡：h20、b300 或 mi308x")
    ap.add_argument("--arch", help="目标架构，如 sm_90、sm_100a、sm_103a、gfx942；可逗号分隔多个")
    ap.add_argument("--language", help="目标语言：cuda_cpp、hip_cpp、triton 或 cutedsl；省略时用该卡的主语言")
    ap.add_argument("--gpu", help="要用的 GPU 编号（NVIDIA 用 nvidia-smi 编号，AMD 用 rocm-smi 编号）")
    ap.add_argument("--busy-util", type=int, default=5, help="利用率超过该百分比判被占用（默认 5）")
    ap.add_argument("--busy-mem-mib", dest="busy_mem", type=int, default=1024, help="显存占用超过该 MiB 判被占用（默认 1024）")
    ap.add_argument("--lock-dir", help="实际要用的 GPU 锁目录")
    ap.add_argument("--expect-lock-dir", help="期望的共享锁目录；与 --lock-dir 不一致则失败")
    ap.add_argument("--need-nsys", action="store_true",
                    help="要求主 profiler 可用：NVIDIA 查 nsys，AMD 查 rocprofv3/rocprof")
    ap.add_argument("--platform", default=None,
                    help="目标平台：nv/nvidia/cuda 或 amd/rocm/hip；不填时由 --card/--arch 路由")
    ap.add_argument("--inspect-host", action="store_true",
                    help="显式允许读取本机 nvidia-smi/rocminfo 来补全缺失的卡和架构；默认绝不按主机工具猜测")
    ap.add_argument("--rocm-llvm", help="AMD：ROCm 的 llvm bin 目录（或 llvm-objdump 全路径），C1 回查二进制要用")
    ap.add_argument("--json", metavar="OUT.json", help="把结果写成 JSON")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quiet", action="store_true", help="（默认）子进程 stderr 捕获不透传，只输出结论行")
    g.add_argument("--verbose", action="store_true", help="额外缩进打印捕获到的子进程 stderr，便于排查")
    ap.add_argument("--selftest", action="store_true", help="用假命令输出自检解析逻辑后退出")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    try:
        route = resolve_preflight_target(a)
    except backend_registry.UnsupportedTarget as exc:
        target = {
            "card": backend_registry.normalize_card(a.card),
            "platform": backend_registry.normalize_platform(a.platform),
            "arch": backend_registry.normalize_arch(a.arch),
        }
        result = R("target-route", FAIL, "目标路由失败：%s" % exc,
                   "给 --card h20|b300|mi308x 或 --arch sm_90|sm_100a|sm_103a|gfx942；"
                   "只有显式加 --inspect-host 才允许读本机工具补全目标")
        v, cnt = report([result], a.verbose)
        if a.json:
            doc = dict(verdict=v, counts=cnt, results=[result],
                       python=sys.version.split()[0], os_platform=sys.platform,
                       target=target, language=a.language, recorder=None,
                       profiler=None, evidence_status="unverified",
                       target_route_error=exc.as_dict(),
                       args=dict(card=a.card, arch=a.arch, language=a.language,
                                 gpu=a.gpu, lock_dir=a.lock_dir,
                                 expect_lock_dir=a.expect_lock_dir,
                                 need_nsys=a.need_nsys, platform=a.platform,
                                 inspect_host=a.inspect_host,
                                 rocm_llvm=a.rocm_llvm))
            try:
                _write_json(a.json, doc)
                print("结果已写入 " + a.json)
            except OSError as error:
                print("无法写 --json 文件 %s：%s" % (a.json, error))
        return 1

    target = route["target"]
    target_arch = target.get("arch") or ",".join(target.get("architectures") or []) or "(unresolved)"
    print("目标：card=%s platform=%s arch=%s" % (
        target.get("card"), target.get("platform"), target_arch))
    print("路由：language=%s recorder=%s profiler=%s evidence_status=%s" % (
        route.get("language"), route.get("recorder"),
        (route.get("profiler") or {}).get("primary"), route.get("evidence_status")))
    results = run_all(a, route)
    v, cnt = report(results, a.verbose)
    if a.json:
        doc = dict(verdict=v, counts=cnt, results=results,
                   python=sys.version.split()[0], os_platform=sys.platform,
                   args=dict(card=a.card, arch=a.arch, language=a.language,
                             gpu=a.gpu, lock_dir=a.lock_dir,
                             expect_lock_dir=a.expect_lock_dir,
                             need_nsys=a.need_nsys, platform=a.platform,
                             inspect_host=a.inspect_host,
                             rocm_llvm=a.rocm_llvm))
        doc.update(_route_document(route))
        try:
            _write_json(a.json, doc)
            print("结果已写入 " + a.json)
        except OSError as e:
            print("无法写 --json 文件 %s：%s" % (a.json, e))
            return 1
    return 0 if v == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
