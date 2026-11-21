# -*- coding: utf-8 -*-
"""跑完后的日志干净度门禁：日志里出现告警 / 报错 / Traceback 就不放行（LOGCHECK FAIL，退出码 1），否则 LOGCHECK PASS（退出码 0）。

为什么要有它：被测库（vllm / sglang / flashinfer / torch）import 和运行时会往 stderr 吐大量 WARNING，混在里面的真正报错
（CUDA 调用失败、No module named、Traceback）很容易被淹没；而一次“带着隐藏报错跑完”的测量数据是不可信的。
所以每次采集后先过这道门，门不过，数据不进结论。

用法：
  python3 -I fi_logcheck.py <log 文件或目录...> [--allow <正则> ...] [--max 0] [--show 5] \
      [--exec-rc 0] [--json-out logcheck.json]
  python3 -I fi_logcheck.py --selftest

给目录时自动扫描里面的 *.log / *.txt / *.err / *.out / *.excp（含 .gz 与 .excp.gz），
不用自己拼文件列表；.excp/.excp.gz 是 Triton 在进程 CWD 留下的异常转储，文件存在本身就算一条命中。

匹配规则（逐行 re.search，大小写敏感，按下面写的形式匹配）：
  WARNING | Warning | warning: | Traceback | Error | ERROR | error: | CUDA .* failed | No module named
  英文 error 只认 `error:`、`Error`、`ERROR` 三种形式；小写的 `errors={}`、`error_count=0`、`no error found` 不算；
  中文“错误：无”本来就不含英文关键字，不会命中。
  注意 `Error` 是子串匹配，所以 RuntimeError / ValueError / ModuleNotFoundError 都会命中（这正是想要的）；
  代价是 `Errors: 0`、`MeanSquaredError` 这类正常文本也会命中 —— 这种情况用 --allow 精确放行，并在注释里写明放行理由。
  rocprofv3 用 glog 风格前缀打严重级别（`W20261109 02:07:38.787238 <tid> simple_timer.cpp:55] ...`），
  整行不含 WARNING/Error 字样，只按上面那些词判会一条也看不见（fi_bugs_amd-sk-ds B26）。所以另认
  `^[WEF]<8 位日期>`：W/E/F 一律命中，只有两类 rocprofv3 正常行按内置白名单放行并单独计数——
  simple_timer.cpp / tool.cpp 的初始化与计时行，以及 output_stream.cpp 的 `Opened result file`
  （它是 E 前缀但内容是成功信息）。要连这两类也当命中，加 --strict-glog；其余 rocprofv3 的 W/E/F 一律不放行。
--allow：可重复；命中行只要匹配任一 --allow 正则（re.search）就不计入命中，但会在报告里单独数出“放行 K 行”，方便审计。
--max：总命中行数上限（默认 0），总数 > --max 判 FAIL。
文件读不了（不存在 / 无权限）也判 FAIL：无法证明干净，不能当作干净。.gz 文件自动解压读取。
--json-out 写逐文件 clean/命中数/SHA256 以及总 rc；--exec-rc 非 0 同样判 FAIL。
退出码：0=PASS；1=FAIL；2=用法错误（如正则写错、没给文件）。
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import tempfile
import time

# Windows 控制台常是 GBK：统一把 stdout/stderr 改成 UTF-8 + replace，打印中文不崩（没有 reconfigure 的旧版本不调用）
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PATTERN = re.compile(r"WARNING|Warning|warning:|Traceback|Error|ERROR|error:|CUDA .* failed|No module named")
# rocprofv3 / glog 的严重级别前缀：W=warning、E=error、F=fatal，后面跟 8 位日期。
GLOG_RE = re.compile(r"^[WEF]\d{8}\b")
# 只有这两类 glog 行算正常（实测：一次成功的 rocprofv3 采集会打十几条 simple_timer/tool 的 W 行，
# 外加一条写着 E 前缀的 "Opened result file" 成功信息）。白名单必须窄：写成“放行所有 E 行”就会漏真错误。
GLOG_BENIGN_RE = (
    re.compile(r"^[WEF]\d{8}\s+[\d:.]+\s+\d+\s+(simple_timer|tool)\.cpp:\d+\]"),
    re.compile(r"^[WEF]\d{8}\s+[\d:.]+\s+\d+\s+output_stream\.cpp:\d+\]\s+Opened result file"),
)
MAX_SHOW_CHARS = 200  # 每条命中行最多显示的字符数
# 给目录时要扫的后缀。.excp 必须在列：Triton 每次预期失败（例如 num_stages 超资源的
# OutOfResources）都会在进程 CWD 落一个 <时间戳>_<pid>_<tid>.excp，只看 *.log 会漏掉它
# （fi_bugs_amd-sk-ds B21：12 个 .excp 落在运行目录根部，门禁却没报）。
DIR_SCAN_SUFFIXES = (".log", ".txt", ".err", ".out", ".excp",
                     ".log.gz", ".txt.gz", ".err.gz", ".out.gz", ".excp.gz")


def expand_paths(paths, max_files=4000):
    """把参数里的目录展开成需要门禁的文件；文件原样保留。返回 (文件列表, 展开说明)。"""
    import os
    files, notes = [], []
    for p in paths:
        if os.path.isdir(p):
            picked = []
            for root, dirs, names in os.walk(p):
                dirs.sort()
                for n in sorted(names):
                    if n.endswith(DIR_SCAN_SUFFIXES):
                        picked.append(os.path.join(root, n))
            picked.sort()
            notes.append("目录 %s：扫到 %d 个日志/异常转储文件" % (p, len(picked)))
            if len(picked) > max_files:
                notes.append("目录 %s：文件数超过 %d，只扫前 %d 个（其余请分批传）" % (p, max_files, max_files))
                picked = picked[:max_files]
            files.extend(picked)
        else:
            files.append(p)
    return files, notes


def glog_benign(s, strict_glog=False):
    """这行是不是 rocprofv3 的 glog 正常行（初始化/计时/Opened result file）。--strict-glog 时一律 False。"""
    if strict_glog or not GLOG_RE.match(s):
        return False
    return any(r.match(s) for r in GLOG_BENIGN_RE)


def scan_lines(lines, allow, strict_glog=False):
    """扫描可迭代的文本行。返回 (命中列表[(行号, 文本)], 被 --allow 放行的行数, glog 正常行数)。"""
    hits, allowed, benign = [], 0, 0
    for i, line in enumerate(lines, 1):
        s = line.rstrip("\r\n")
        hit = bool(PATTERN.search(s)) or bool(GLOG_RE.match(s))
        if not hit:
            continue
        if glog_benign(s, strict_glog):
            benign += 1
        elif any(a.search(s) for a in allow):
            allowed += 1
        else:
            hits.append((i, s))
    return hits, allowed, benign


def scan_file(path, allow, strict_glog=False):
    """读文件并扫描。UTF-8 解码，坏字节用替换符，不会因为日志里混了二进制而崩。"""
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        hits, allowed, benign = scan_lines(f, allow, strict_glog)
    if path.endswith(".excp") or path.endswith(".excp.gz"):
        # 异常转储文件存在本身就说明有进程异常退出过，正文没有关键字也要记一条命中。
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            first = next((ln.strip() for ln in f if ln.strip()), "（空文件）")
        synth = "异常转储文件（.excp）存在，说明有进程异常退出过：%s" % first[:MAX_SHOW_CHARS]
        if any(a.search(synth) for a in allow):
            allowed += 1
        else:
            hits.append((0, synth))
    return hits, allowed, benign


def format_file_report(path, hits, allowed, show, benign=0):
    head = "文件 %s：命中 %d 行（--allow 放行 %d 行）" % (path, len(hits), allowed)
    if benign:
        head += "，另有 %d 行 rocprofv3 glog 正常行（simple_timer/tool 初始化计时、Opened result file）" % benign
    out = [head]
    for i, s in hits[:show]:
        out.append("  L%d: %s" % (i, s if len(s) <= MAX_SHOW_CHARS else s[:MAX_SHOW_CHARS] + "..."))
    if len(hits) > show:
        out.append("  ...（另有 %d 条未列出）" % (len(hits) - show))
    return out


def decide(total, mx, unreadable, exec_rc=None):
    """门禁判定：总命中数 > 上限、有读不了的文件，或显式给出的执行码非 0，判 FAIL。"""
    return "FAIL" if (total > mx or unreadable > 0 or exec_rc not in (None, 0)) else "PASS"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_write_json(path, value):
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def selftest():
    """用手工构造的日志行验证“应命中 / 不应命中 / --allow 放行 / 展示条数 / 判定”。"""
    no_allow = []
    should_hit = [
        "WARNING 11-07 12:00:01 [vllm] something is deprecated",
        "/x/y.py:12: UserWarning: TypedStorage is deprecated",
        'demo.cu(12): warning: variable "x" was declared but never referenced',
        "Traceback (most recent call last):",
        "RuntimeError: CUDA out of memory",
        "ModuleNotFoundError: No module named 'vllm'",
        "  No module named flashinfer",            # 只靠 No module named 命中
        "[ERROR] bad arg",
        "demo.cu(5): error: identifier \"foo\" is undefined",
        "CUDA call cudaMalloc failed with code 2",
        "CUDA launch of fi_kernel failed",         # 只靠 CUDA .* failed 命中
        "总计 Errors: 3",                            # `Error` 是子串匹配，Errors 也命中（按规则）
        # rocprofv3 的 glog 前缀：不含 WARNING/Error 字样，也必须命中（fi_bugs_amd-sk-ds B26）
        "E20261109 02:07:41.194387 139851752005504 output_stream.cpp:111] failed to flush queue",
        "W20261109 02:07:38.787238 139851752005504 hsa_activity.cpp:99] counter overflow",
        "F20261109 02:07:38.787238 139851752005504 rocprofv3.cpp:42] fatal: cannot open output",
    ]
    # glog 正常行：内置白名单放行，单独计数（--strict-glog 时改成命中）
    glog_ok = [
        "W20261109 02:07:38.787238 139851752005504 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.000561 sec",
        "W20261109 02:07:38.800000 139851752005504 tool.cpp:321] [rocprofv3] HSA activity enabled",
        "E20261109 02:07:41.194387 139851752005504 output_stream.cpp:111] Opened result file: /x/1450588_results.json",
    ]
    should_not_hit = [
        "errors={}",
        "错误：无",
        "结构错误计数 = 0",
        "error_count=0",
        "no error found",
        "0 errors, 0 warnings",
        "warnings=0",
        "kernel 时长 12.3 us",
        "CUDA graph captured",
        "CUDA kernel finished",                    # 有 CUDA 但没有 failed
        "retry failed once",                       # 有 failed 但没有 CUDA
        "cuda call failed",                        # 小写 cuda 不是 `CUDA .* failed`
        "",
    ]
    for ln in should_hit:
        h, a, b = scan_lines([ln], no_allow)
        assert len(h) == 1 and a == 0 and b == 0, ("应命中却没命中", ln)
    for ln in should_not_hit:
        h, a, b = scan_lines([ln], no_allow)
        assert not h and a == 0 and b == 0, ("不应命中却命中了", ln)
    for ln in glog_ok:
        h, a, b = scan_lines([ln], no_allow)
        assert not h and a == 0 and b == 1, ("glog 正常行被误判", ln, h)
        h2, a2, b2 = scan_lines([ln], no_allow, strict_glog=True)
        assert len(h2) == 1 and b2 == 0, ("--strict-glog 没把正常行算成命中", ln)

    # --allow：精确放行驱动模板自己打印的那一行（含 ERROR 字样），其余仍命中；放行行数单独计
    allow = [re.compile(r"已导入 \S+，日志级别压到 ERROR")]
    lines = ["已导入 vllm，日志级别压到 ERROR", "WARNING real problem", "已导入 torch，日志级别压到 ERROR", "Traceback (most recent call last):"]
    h, a, _b = scan_lines(lines, allow)
    assert [x[0] for x in h] == [2, 4] and a == 2, (h, a)
    h2, a2, _b2 = scan_lines(lines, [])
    assert len(h2) == 4 and a2 == 0
    # 多个 --allow 取并集；allow 也能放行 warning: 这类行
    h3, a3, _b3 = scan_lines(lines + ["foo.cu(1): warning: unused"], allow + [re.compile(r"unused")])
    assert [x[0] for x in h3] == [2, 4] and a3 == 3

    # 行号从 1 开始，CRLF 行尾被去掉
    h4, _, _ = scan_lines(["ok\r\n", "ERROR x\r\n"], [])
    assert h4 == [(2, "ERROR x")], h4

    # 报告：前 5 条 + “另有 N 条”，超长行被截断；glog 正常行单独写在标题行里
    many = [(i, "ERROR %d" % i) for i in range(1, 9)]
    rep = format_file_report("a.log", many, 0, 5)
    assert rep[0] == "文件 a.log：命中 8 行（--allow 放行 0 行）" and len(rep) == 1 + 5 + 1 and "另有 3 条" in rep[-1], rep
    assert len(format_file_report("a.log", many[:5], 0, 5)) == 1 + 5
    assert format_file_report("a.log", [(1, "E" * 500)], 0, 5)[1].endswith("...")
    assert "glog 正常行" in format_file_report("a.log", [], 0, 5, benign=14)[0]

    # 判定：总数 > max 才 FAIL；读不了文件或显式执行码非 0 一律 FAIL
    assert decide(0, 0, 0) == "PASS" and decide(1, 0, 0) == "FAIL"
    assert decide(2, 2, 0) == "PASS" and decide(3, 2, 0) == "FAIL"
    assert decide(0, 0, 1) == "FAIL"
    assert decide(0, 0, 0, 0) == "PASS" and decide(0, 0, 0, 7) == "FAIL"

    # 目录展开 + .excp：Triton 失败时落在 CWD 的异常转储必须被门禁抓到（fi_bugs_amd-sk-ds B21）
    import os
    import shutil
    import tempfile
    td = tempfile.mkdtemp(prefix="fi_logcheck_selftest_")
    try:
        os.makedirs(os.path.join(td, "logs"))
        open(os.path.join(td, "logs", "clean.log"), "w", encoding="utf-8").write("kernel 时长 12.3 us\n")
        open(os.path.join(td, "logs", "dirty.log"), "w", encoding="utf-8").write("RuntimeError: boom\n")
        open(os.path.join(td, "1794159045.7072983_1212676_140226434412992.excp"), "w",
             encoding="utf-8").write("triton.runtime.errors.OutOfResources: out of resource: shared memory\n")
        files, notes = expand_paths([td])
        assert len(files) == 3, (files, notes)
        assert any(x.endswith(".excp") for x in files), files
        assert notes and "扫到 3 个" in notes[0], notes
        h, _a, _b = scan_file(files[[i for i, x in enumerate(files) if x.endswith(".excp")][0]], [])
        assert len(h) == 1 and "异常转储文件" in h[0][1], h
        h2, a2, b2 = scan_file(os.path.join(td, "logs", "clean.log"), [])
        assert not h2 and a2 == 0 and b2 == 0, h2
        # --allow 也能放行 .excp 这条合成命中（写明理由后才允许）
        h3, a3, _b3 = scan_file([x for x in files if x.endswith(".excp")][0], [re.compile(r"OutOfResources")])
        assert not h3 and a3 == 1, (h3, a3)
        # .excp.gz 也必须在目录扫描里出现，并且文件存在本身就命中。
        with gzip.open(os.path.join(td, "compressed.excp.gz"), "wt", encoding="utf-8") as f:
            f.write("triton.runtime.errors.OutOfResources: compressed dump\n")
        files2, notes2 = expand_paths([td])
        excp_gz = [x for x in files2 if x.endswith(".excp.gz")]
        assert len(files2) == 4 and len(excp_gz) == 1, (files2, notes2)
        hgz, agz, _bgz = scan_file(excp_gz[0], [])
        assert len(hgz) == 1 and "异常转储文件" in hgz[0][1], hgz
        # 一次成功的 rocprofv3 采集：14 条 glog 正常行放行，夹一条真 E 行必须命中
        open(os.path.join(td, "logs", "rocprof.log"), "w", encoding="utf-8").write(
            "\n".join(glog_ok * 4 + ["E20261109 02:07:42.000000 139851752005504 output_stream.cpp:9] write failed"]) + "\n")
        h4, a4, b4 = scan_file(os.path.join(td, "logs", "rocprof.log"), [])
        assert len(h4) == 1 and a4 == 0 and b4 == 12, (h4, a4, b4)
        assert h4[0][1].endswith("write failed"), h4
        # 只传文件时行为不变
        f2, n2 = expand_paths([os.path.join(td, "logs", "clean.log")])
        assert len(f2) == 1 and not n2, (f2, n2)
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print("SELFTEST PASS（应命中 %d 条、不应命中 %d 条、glog 正常行 %d 条按白名单单独计数、"
          "allow/展示/判定/目录展开/.excp/.excp.gz 均符合预期）"
          % (len(should_hit), len(should_not_hit), len(glog_ok)))


def main():
    ap = argparse.ArgumentParser(description="日志干净度门禁")
    ap.add_argument("logs", nargs="*", help="要检查的日志文件（可多个，.gz 自动解压）")
    ap.add_argument("--allow", action="append", default=[], metavar="REGEX", help="命中行若匹配该正则则放行；可重复")
    ap.add_argument("--max", type=int, default=0, help="总命中行数上限（默认 0；总数 > 上限则 FAIL）")
    ap.add_argument("--show", type=int, default=5, help="每个文件最多列出几条命中行（默认 5）")
    ap.add_argument("--strict-glog", action="store_true",
                    help="连 rocprofv3 的 glog 正常行（simple_timer/tool 初始化计时、Opened result file）也当命中")
    ap.add_argument("--exec-rc", type=int, default=None,
                    help="可选：被测程序或采集器的退出码；非 0 时日志门禁直接 FAIL 并写进 JSON")
    ap.add_argument("--json-out", help="把逐文件 clean/rc/SHA256 汇总写成 JSON")
    ap.add_argument("--selftest", action="store_true", help="运行内置单元测试后退出")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.logs:
        ap.error("需要至少一个日志文件（或用 --selftest）")
    if a.json_out:
        try:
            os.unlink(a.json_out)
        except FileNotFoundError:
            pass
    try:
        allow = [re.compile(x) for x in a.allow]
    except re.error as e:
        print("--allow 正则写错了：%s" % e, file=sys.stderr)
        sys.exit(2)

    files, notes = expand_paths(a.logs)
    for n in notes:
        print(n)
    total, unreadable, benign_total = 0, 0, 0
    records = []
    for path in files:
        try:
            hits, allowed, benign = scan_file(path, allow, a.strict_glog)
        except OSError as e:
            unreadable += 1
            print("文件 %s：无法读取（%s）—— 无法证明干净，按失败处理" % (path, e.strerror or e))
            records.append({"path": str(path), "readable": False, "clean": False,
                            "error": e.strerror or str(e), "hits": None})
            continue
        total += len(hits)
        benign_total += benign
        records.append({
            "path": str(path),
            "readable": True,
            "clean": not hits,
            "sha256": sha256_file(path),
            "hits": len(hits),
            "allowed_hits": allowed,
            "benign_glog_lines": benign,
            "exception_dump": str(path).endswith(".excp") or str(path).endswith(".excp.gz"),
            "hit_samples": [{"line": ln, "text": text[:MAX_SHOW_CHARS]} for ln, text in hits[:a.show]],
        })
        print("\n".join(format_file_report(path, hits, allowed, a.show, benign)))
    v = decide(total, a.max, unreadable, a.exec_rc)
    print("合计命中 %d 行（上限 --max %d），无法读取的文件 %d 个" % (total, a.max, unreadable))
    if a.exec_rc is not None:
        print("执行码 --exec-rc %d" % a.exec_rc)
    if benign_total:
        print("另有 %d 行 rocprofv3 glog 正常行按内置白名单放行（--strict-glog 可改成命中）" % benign_total)
    print("LOGCHECK " + v)
    rc = 0 if v == "PASS" else 1
    if a.json_out:
        summary = {
            "schema": "fi-logcheck/v1",
            "tool": "fi-logcheck",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rc": rc,
            "clean": v == "PASS",
            "exec_rc": a.exec_rc,
            "exec_rc_ok": None if a.exec_rc is None else a.exec_rc == 0,
            "total_hits": total,
            "max_hits": a.max,
            "unreadable": unreadable,
            "benign_glog_lines": benign_total,
            "files": records,
            "inputs": [str(x) for x in a.logs],
        }
        _atomic_write_json(a.json_out, summary)
        print("写出", a.json_out)
    sys.exit(rc)


if __name__ == "__main__":
    main()
