# -*- coding: utf-8 -*-
"""fi_overview：从 Perfetto/Chrome trace json 画概览 PNG，并做“图与数据一致”的覆盖率自检。

为什么不能随手画：嵌套区间（外层“prep 装载”包住内层“等 conv”）如果先画内层、再画外层，或者都画满高度，
外层会盖住内层，图上看着“等 conv”很少，实际它占 prep 的 85%。本脚本的画法与自检：
  1) 每条泳道按“最内层独占”着色：任一时刻只归属当时最内层的那个区间（同深度取最后开始的）。
     这样每种事件在图上的面积 = 它的独占时间，不会互相遮盖；外层区间另用泳道上方的细线标出范围。
  2) 同一泳道里互相交叉（不是嵌套）的区间说明有并发实例，脚本会报错并提示先跑 fi_relane.py 拆子泳道
     （加 --auto-relane 自动拆）。
  3) 自检：把渲染好的 PNG 读回来，按颜色数像素，比较“每种事件的像素占比”和“数据里的独占时间占比”，
     任何一项差超过 --tol 个百分点（默认 3）就判 FAIL（退出码 2），不出图结论。
     同时检查图例覆盖了数据里出现的所有事件名。
  4) 下方条形图按进程（通常是角色）给出每种事件的独占时间占比和包含时间占比数字，直接标在图上。

用法：
  python3 -I fi_overview.py in.perfetto.json[.gz] -o out.png [--max-lanes 48] [--t0 us --t1 us] [--auto-relane] [--key prep_instance]
  python3 -I fi_overview.py --selftest
依赖 matplotlib（缺失时给出安装提示并退出 1）。
"""
import argparse, gzip, json, os, sys, warnings
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

EPS = 1e-9
PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
           "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    return d if isinstance(d, dict) else {"traceEvents": d}


_SUFFIX = None
LANE_ROLE = {}  # 解码器在 args.role 里写了 warp 角色时，按它分组


def base_name(n):
    """去掉解码器加的实例后缀（"名字 #123"），按事件种类归类；不同 tile/chunk 号不算不同事件。"""
    global _SUFFIX
    if _SUFFIX is None:
        import re
        _SUFFIX = re.compile(r"\s*#\d+$")
    return _SUFFIX.sub("", n)


def lanes_of(d, t0=None, t1=None):
    ev = d["traceEvents"]
    pname = {e.get("pid"): e.get("args", {}).get("name") for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
    tname = {(e.get("pid"), e.get("tid")): e.get("args", {}).get("name") for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
    lanes = defaultdict(list)
    global LANE_ROLE
    LANE_ROLE = {}
    for e in ev:
        if e.get("ph") != "X" or e.get("cat") in ("nsys_kernel", "info"):
            continue
        r = e.get("args", {}).get("role")
        if r is not None:
            LANE_ROLE.setdefault((e.get("pid"), e.get("tid")), str(r))
        b, en = float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0))
        if t0 is not None:
            b = max(b, t0)
        if t1 is not None:
            en = min(en, t1)
        if en <= b:
            continue
        lanes[(e.get("pid"), e.get("tid"))].append((b, en, base_name(e.get("name", "?"))))
    return lanes, pname, tname


def exclusive(spans):
    """返回 (段列表 [(b, e, name)] 表示最内层独占归属, 交叉数, 外层区间列表 [(b,e,name,depth)])。"""
    sp = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
    cross = 0
    stack = []  # (end, name, depth)
    depth_of = []
    for b, e, n in sp:
        while stack and stack[-1][0] <= b + EPS:
            stack.pop()
        if stack and e > stack[-1][0] + EPS:
            cross += 1
        dep = len(stack)
        depth_of.append((b, e, n, dep))
        stack.append((e, n, dep))
    # 扫描线：事件点上取当前活跃集合里深度最大的（同深度取开始最晚的）
    pts = sorted({x for b, e, n, dep in depth_of for x in (b, e)})
    segs = []
    if not pts:
        return segs, cross, depth_of
    import bisect
    starts = sorted(range(len(depth_of)), key=lambda k: depth_of[k][0])
    active = []
    si = 0
    for a, z in zip(pts[:-1], pts[1:]):
        while si < len(starts) and depth_of[starts[si]][0] <= a + EPS:
            active.append(starts[si])
            si += 1
        active = [k for k in active if depth_of[k][1] > a + EPS]
        if not active:
            continue
        k = max(active, key=lambda k: (depth_of[k][3], depth_of[k][0]))
        n = depth_of[k][2]
        if segs and segs[-1][2] == n and abs(segs[-1][1] - a) < EPS:
            segs[-1] = (segs[-1][0], z, n)
        else:
            segs.append((a, z, n))
    return segs, cross, depth_of


def pick_font():
    from matplotlib import font_manager
    want = ["Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Source Han Sans SC",
            "Microsoft YaHei", "SimHei", "PingFang SC", "AR PL UMing CN"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for w in want:
        if w in have:
            return w
    return None


def set_y_ticks(ax, positions, labels=None, fontsize=None):
    """设 y 轴刻度与标签。

    不用 ax.set_yticks/set_yticklabels：matplotlib 3.9 在需要新建刻度对象时会 deepcopy 已有刻度的
    MarkerStyle，其中的 Path.__deepcopy__ 在 Python 3.13+ 上无限递归（RecursionError）。
    直接换 locator/formatter 效果相同，各版本都稳。
    """
    from matplotlib.ticker import FixedFormatter, FixedLocator
    ax.yaxis.set_major_locator(FixedLocator([float(x) for x in positions]))
    if labels is not None:
        ax.yaxis.set_major_formatter(FixedFormatter([str(x) for x in labels]))
    if fontsize is not None:
        ax.tick_params(axis="y", labelsize=fontsize)


def fix_mpl_path_deepcopy():
    """绕开 matplotlib 3.9 及更早版本的 Path.__deepcopy__：它写的是 copy.deepcopy(super(), memo)，
    在 Python 3.13+ 上无限递归（RecursionError），任何需要新建刻度/图例对象的绘制都会崩。
    先探测，确实有这个 bug 才换成逐字段深拷贝；新版 matplotlib 不受影响。返回是否打了补丁。"""
    import copy
    from matplotlib.path import Path
    probe = Path([[0.0, 0.0], [1.0, 1.0]], [Path.MOVETO, Path.LINETO])
    try:
        copy.deepcopy(probe)
        return False
    except RecursionError:
        pass

    def __deepcopy__(self, memo=None):
        p = Path(copy.deepcopy(self.vertices, memo), copy.deepcopy(self.codes, memo),
                 copy.deepcopy(self._interpolation_steps, memo))
        p._readonly = False
        return p

    Path.__deepcopy__ = __deepcopy__
    Path.deepcopy = __deepcopy__
    return True


def render(d, out, max_lanes=48, t0=None, t1=None, tol=3.0, title=None, dpi=110, group_regex=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("缺 matplotlib/numpy：pip install matplotlib（或不出 PNG，只看 Perfetto json）", file=sys.stderr)
        return 1, None
    import logging
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    if fix_mpl_path_deepcopy():
        print("提示：本机 matplotlib 的 Path.__deepcopy__ 与该 Python 版本不兼容，已就地换成逐字段深拷贝后继续出图")
    lanes, pname, tname = lanes_of(d, t0, t1)
    if not lanes:
        print("trace 里没有区间事件（ph=X）", file=sys.stderr)
        return 1, None
    tot_incl = defaultdict(float)
    for sp in lanes.values():
        for b, e, n in sp:
            tot_incl[n] += e - b
    if len(tot_incl) > len(PALETTE):  # 颜色不够时把包含时间最小的尾部合并成“其它”，否则颜色重复会让像素自检失去意义
        keep = set(sorted(tot_incl, key=lambda n: -tot_incl[n])[:len(PALETTE) - 1])
        other = "其它" if pick_font() else "other"
        for lk in list(lanes):
            lanes[lk] = [(b, e, n if n in keep else other) for b, e, n in lanes[lk]]
        print(f"提示：事件种类 {len(tot_incl)} 个超过调色板 {len(PALETTE)} 色，尾部 {len(tot_incl) - len(keep)} 个合并为“{other}”")
    import re
    relaned = bool(d.get("otherData", {}).get("relane"))
    unit_of = (lambda k: (k[0], (k[1] // 64) if isinstance(k[1], int) else k[1])) if relaned else (lambda k: k)
    rx = re.compile(group_regex) if group_regex else None
    if rx is None and not LANE_ROLE and len({k[0] for k in lanes}) == 1:  # 只有一个进程时，试着从线程名里的 (角色) 自动分组
        cand = re.compile(r"[\(（]([^\)）]+)[\)）]")
        if all(cand.search(tname.get(k) or "") for k in lanes):
            rx = cand

    use_role = rx is None and group_regex is None and LANE_ROLE and all(k in LANE_ROLE for k in lanes)

    def group_of(k):
        if use_role:
            return LANE_ROLE[k]
        if rx is not None:
            m = rx.search(tname.get(k) or "")
            if m:
                return m.group(1) if m.groups() else m.group(0)
        g = pname.get(k[0], k[0])
        return str(g)
    font = pick_font()
    if font:
        matplotlib.rcParams["font.sans-serif"] = [font]
        matplotlib.rcParams["axes.unicode_minus"] = False
    # 统计（全部泳道）
    excl = defaultdict(float)
    incl = defaultdict(float)
    excl_by_p = defaultdict(lambda: defaultdict(float))
    incl_by_p = defaultdict(lambda: defaultdict(float))
    lane_span_by_p = defaultdict(float)
    segs_of, cross_tot, cross_lanes = {}, 0, 0
    for lk, sp in lanes.items():
        segs, cross, _ = exclusive(sp)
        segs_of[lk] = segs
        if cross:
            cross_tot += cross
            cross_lanes += 1
        g = group_of(lk)
        for b, e, n in segs:
            excl[n] += e - b
            excl_by_p[g][n] += e - b
        for b, e, n in sp:
            incl[n] += e - b
            incl_by_p[g][n] += e - b
        lane_span_by_p[g] += max(e for b, e, n in sp) - min(b for b, e, n in sp)
    if cross_lanes:
        print(f"错误：{cross_lanes} 条泳道里有 {cross_tot} 处区间交叉（不是嵌套），说明同一泳道有并发实例；"
              f"先运行 fi_relane.py 拆子泳道，或加 --auto-relane", file=sys.stderr)
        return 2, None
    names = sorted(excl, key=lambda n: -excl[n])
    ascii_map = {}
    if not font:
        ascii_map = {n: f"E{i + 1}" for i, n in enumerate(names)}
        print("提示：本机没有中文字体，图上用 E1/E2… 代替事件名，对照表：" + "，".join(f"{v}={k}" for k, v in ascii_map.items()))
    lab = lambda n: ascii_map.get(n, n)
    color = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}
    # 选泳道：以“单元”（原始泳道，拆过子泳道的把同一原始泳道的子泳道当一个单元）为单位，按组分层均匀抽样，
    # 每组至少 2 个单元，同一单元的子泳道全部显示，总行数不超过 max_lanes（单个单元超过时也至少显示一个单元）。
    skey = lambda k: (str(k[0]), k[1] if isinstance(k[1], (int, float)) else str(k[1]))
    keys = sorted(lanes, key=skey)
    units = defaultdict(list)
    for k in keys:
        units[unit_of(k)].append(k)
    ukeys_by_g = defaultdict(list)
    for u in sorted(units, key=lambda u: (str(u[0]), u[1] if isinstance(u[1], (int, float)) else str(u[1]))):
        ukeys_by_g[group_of(units[u][0])].append(u)
    rows_per_unit = max(1, round(sum(len(v) for v in units.values()) / len(units)))
    budget_units = max(len(ukeys_by_g) * 2, max_lanes // rows_per_unit)
    tot_units = len(units)
    shown = []
    for g, us in ukeys_by_g.items():
        q = max(2, round(budget_units * len(us) / tot_units))
        q = min(q, len(us))
        step = len(us) / q
        for i in range(q):
            shown.extend(units[us[int(i * step)]])
    shown_units = len({unit_of(k) for k in shown})
    tmin = min(b for k in keys for b, e, n in lanes[k])
    tmax = max(e for k in keys for b, e, n in lanes[k])
    pids = sorted(lane_span_by_p, key=str)
    fig_h = 2.0 + 0.16 * len(shown) + 0.5 * len(pids) + 1.0
    fig = plt.figure(figsize=(16, fig_h))
    fig.subplots_adjust(left=0.1, right=0.86)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.16 * len(shown) + 1.2, 0.5 * len(pids) + 0.9], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    for row, k in enumerate(shown):
        y = len(shown) - 1 - row
        by = defaultdict(list)
        for b, e, n in segs_of[k]:
            by[n].append((b, e - b))
        for n, xr in by.items():
            ax.broken_barh(xr, (y - 0.42, 0.84), facecolors=color[n], linewidth=0, antialiased=False)
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(-0.6, len(shown) - 0.4)
    yl = [(tname.get(k) or f"{pname.get(k[0], k[0])}/{k[1]}") for k in shown]
    if len(shown) <= 64:
        set_y_ticks(ax, range(len(shown)), [lab(x) if not font else x for x in reversed(yl)], fontsize=6)
    ax.set_xlabel("时间（us）" if font else "time (us)")
    sub = (f"显示 {shown_units}/{len(units)} 个原始泳道（{len(shown)} 行，按组分层均匀抽样），按最内层独占时间着色" if font
           else f"{shown_units}/{len(units)} lanes ({len(shown)} rows), innermost-exclusive coloring")
    ax.set_title((title or "") + ("  " if title else "") + sub, fontsize=10)
    from matplotlib.patches import Patch
    tot_ex = sum(excl.values())
    ax.legend(handles=[Patch(facecolor=color[n], label=f"{lab(n)}  {100 * excl[n] / tot_ex:.1f}%" if font else
                             f"{lab(n)} excl {100 * excl[n] / tot_ex:.1f}%") for n in names],
              loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False,
              title=("独占时间占比\n（全部泳道）" if font else "exclusive share\n(all lanes)"), title_fontsize=8)
    # 条形图：每个进程的独占占比（相对该进程所有泳道的跨度之和）
    bx = fig.add_subplot(gs[1])
    for i, p in enumerate(pids):
        x = 0.0
        span = lane_span_by_p[p] or 1.0
        for n in names:
            v = 100.0 * excl_by_p[p].get(n, 0.0) / span
            if v <= 0:
                continue
            bx.barh(i, v, left=x, color=color[n], linewidth=0)
            if v >= 4:
                bx.text(x + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=7, color="white")
            x += v
        idle = max(0.0, 100.0 - x)
        if idle > 0.05:
            bx.barh(i, idle, left=x, color="#eeeeee", linewidth=0)
    short = lambda x: x if len(x) <= 24 else x[:22] + "…"
    set_y_ticks(bx, range(len(pids)), [short(str(p)) for p in pids], fontsize=8)
    bx.set_xlim(0, 100)
    bx.set_xlabel("占该组泳道时间的百分比（独占；灰色=泳道内没有任何区间）" if font else "% of lane time (exclusive; grey = no span)")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error" if font else "ignore")
        fig.savefig(out, dpi=dpi, bbox_inches=None)
    # ---- 自检：读回像素 ----
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    bb = ax.get_window_extent()
    H = buf.shape[0]
    x0, x1 = int(np.ceil(bb.x0)) + 1, int(np.floor(bb.x1)) - 1
    y0, y1 = int(np.ceil(H - bb.y1)) + 1, int(np.floor(H - bb.y0)) - 1
    reg = buf[y0:y1, x0:x1].reshape(-1, 3)
    import matplotlib.colors as mc
    pix = {}
    for n in names:
        rgb = (np.array(mc.to_rgb(color[n])) * 255).round().astype(np.int16)
        pix[n] = int((np.abs(reg.astype(np.int16) - rgb).sum(axis=1) <= 6).sum())
    plt.close(fig)
    # 期望：显示泳道上的独占时间占比
    ex_shown = defaultdict(float)
    for k in shown:
        for b, e, n in segs_of[k]:
            ex_shown[n] += e - b
    tot_s = sum(ex_shown.values()) or 1.0
    tot_p = sum(pix.values()) or 1
    rows, bad = [], []
    for n in names:
        want = 100.0 * ex_shown.get(n, 0.0) / tot_s
        got = 100.0 * pix[n] / tot_p
        rows.append((n, want, got))
        if abs(want - got) > tol:
            bad.append(n)
    check = dict(ok=not bad, tol_pp=tol, lanes_total=len(keys), lanes_shown=len(shown), font=font,
                 rows=[dict(event=n, expect_pct=round(w, 2), pixel_pct=round(g, 2)) for n, w, g in rows],
                 exclusive_us={n: round(excl[n], 3) for n in names}, inclusive_us={n: round(incl[n], 3) for n in names},
                 by_group={str(p): {n: dict(excl_pct=round(100 * excl_by_p[p].get(n, 0) / (lane_span_by_p[p] or 1), 2),
                                                          incl_pct=round(100 * incl_by_p[p].get(n, 0) / (lane_span_by_p[p] or 1), 2))
                                                  for n in names if incl_by_p[p].get(n)} for p in pids})
    return (0 if not bad else 2), check


def print_check(c):
    print(f"覆盖率自检（图上像素占比 vs 数据独占时间占比，容差 {c['tol_pp']} 个百分点）：{'PASS' if c['ok'] else 'FAIL'}")
    print("事件                      数据占比%   像素占比%   独占us        包含us")
    for r in c["rows"]:
        n = r["event"]
        print(f"{n:<22}{r['expect_pct']:>10.2f}{r['pixel_pct']:>11.2f}{c['exclusive_us'][n]:>13.1f}{c['inclusive_us'][n]:>14.1f}")
    for g, v in c["by_group"].items():
        print(f"[{g}] " + "；".join(f"{n} 独占 {x['excl_pct']}% / 包含 {x['incl_pct']}%" for n, x in v.items()))


PIXEL_AUDIT_MARKER = "fi_overview.pixel_audit"


def pixel_audit_path(out_png, explicit=None):
    """像素自检（图与数据是否一致）的落盘路径，默认 `<stem>.overview_check.json`。

    不能再默认写 `<stem>.check.json`：fi_check 的交付物就叫 `out/<name>.check.json`，而本工具 `-o` 的
    惯例是 `out/<name>.png`，splitext 之后两者同名——按 SKILL.md 把产物都放同一个 out/ 的人，
    出一次图就把 C1-C4 结果换成像素自检（fi_bugs_b3-sk-qwen B15 实测损失过一份交付物）。
    """
    return explicit or (os.path.splitext(out_png)[0] + ".overview_check.json")


def write_pixel_audit(c, out_png, explicit=None):
    """写出像素自检；目标已存在且不是本工具写的（没有 tool 标记）就拒绝覆盖并非 0 退出。"""
    path = pixel_audit_path(out_png, explicit)
    if os.path.exists(path):
        mine = False
        try:
            with open(path, "r", encoding="utf-8") as f:
                old = json.load(f)
            mine = isinstance(old, dict) and old.get("tool") == PIXEL_AUDIT_MARKER
        except Exception:
            mine = False
        if not mine:
            raise SystemExit(
                f"{path} 已存在，且不是本工具写的像素自检（缺 tool={PIXEL_AUDIT_MARKER!r} 标记），拒绝覆盖。\n"
                "fi_check 的交付物默认就叫 <stem>.check.json，被覆盖就丢了 C1-C4 结果。"
                "要把像素自检写到别处用 --pixel-audit-out <路径>，或先自己把那个文件挪走。")
    payload = dict(c)
    payload["tool"] = PIXEL_AUDIT_MARKER
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return path


def selftest():
    import shutil
    import tempfile
    X = lambda ts, dur, name, tid=0: dict(ph="X", pid=1, tid=tid, ts=ts, dur=dur, name=name)
    ev = [dict(ph="M", pid=1, name="process_name", args=dict(name="链"))]
    for t in range(6):
        # 外层 prep 0..100，内层等待 5..90（85%），之后计算 100..130
        ev += [X(0, 100, "prep", t), X(5, 85, "等待", t), X(100, 30, "计算", t)]
    d = {"traceEvents": ev}
    td = tempfile.mkdtemp(prefix=".fi_overview_", dir=os.getcwd())
    try:
        out = os.path.join(td, "t.png")
        rc, c = render(d, out)
        assert rc == 0 and c["ok"], c
        assert abs(c["exclusive_us"]["等待"] - 6 * 85) < 1e-6 and abs(c["exclusive_us"]["prep"] - 6 * 15) < 1e-6, c["exclusive_us"]
        exp = {r["event"]: r["expect_pct"] for r in c["rows"]}
        assert abs(exp["等待"] - 100 * 85 / 130) < 0.01, exp
        # 像素自检的落盘名不能撞 fi_check 的交付物，且不许覆盖别人写的文件（fi_bugs_b3-sk-qwen B15）
        assert pixel_audit_path(os.path.join(td, "t1.png")) == os.path.join(td, "t1.overview_check.json")
        p1 = write_pixel_audit(c, os.path.join(td, "t1.png"))
        with open(p1, encoding="utf-8") as f:
            assert json.load(f)["tool"] == PIXEL_AUDIT_MARKER, p1
        write_pixel_audit(c, os.path.join(td, "t1.png"))          # 自己写的可以重写
        foreign = os.path.join(td, "t2.check.json")               # 别人（fi_check）的交付物
        with open(foreign, "w", encoding="utf-8") as f:
            json.dump({"checks": {"C1": {"passed": True}}}, f)
        try:
            write_pixel_audit(c, os.path.join(td, "t2.png"), foreign)
            raise AssertionError("没有拒绝覆盖 fi_check 的 check.json")
        except SystemExit as e:
            assert "拒绝覆盖" in str(e), str(e)
        with open(foreign, encoding="utf-8") as f:
            assert json.load(f)["checks"]["C1"]["passed"] is True, "foreign file was modified"
        squat = os.path.join(td, "t3.overview_check.json")        # 默认名上有个非本工具的文件
        with open(squat, "w", encoding="utf-8") as f:
            f.write("not json at all")
        try:
            write_pixel_audit(c, os.path.join(td, "t3.png"))
            raise AssertionError("默认名上的陌生文件被覆盖了")
        except SystemExit:
            pass
        # 交叉区间应报错
        rc2, _ = render({"traceEvents": [X(0, 10, "a"), X(5, 10, "a")]}, out)
        assert rc2 == 2
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print("SELFTEST PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--max-lanes", type=int, default=48)
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--title")
    ap.add_argument("--auto-relane", action="store_true")
    ap.add_argument("--key", help="--auto-relane 时按此 args 字段分子泳道")
    ap.add_argument("--pixel-audit-out", dest="pixel_audit_out",
                    help="像素自检（图与数据是否一致）的**输出**路径，默认 <stem>.overview_check.json。"
                         "这是输出不是输入：本工具不读 fi_check 的结果，也不会往图上标注它")
    ap.add_argument("--check-json", dest="pixel_audit_out", help=argparse.SUPPRESS)  # 旧名字，等价于 --pixel-audit-out
    ap.add_argument("--group-regex", help="按线程名的正则分组（取第一个捕获组）；默认按进程，只有一个进程且线程名都带(角色)时自动按角色")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.inp and a.out):
        ap.error("需要输入 json 和 -o out.png")
    d = load(a.inp)
    if a.auto_relane:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from fi_relane import relane
        d, _ = relane(d, a.key)
    rc, c = render(d, a.out, a.max_lanes, a.t0, a.t1, a.tol, a.title, group_regex=a.group_regex)
    if c is None:
        sys.exit(rc)
    print_check(c)
    audit = write_pixel_audit(c, a.out, a.pixel_audit_out)
    if rc:
        print("自检 FAIL：图与数据不一致，不要用这张图下结论", file=sys.stderr)
    else:
        print("写出", a.out)
    print("像素自检", audit)
    sys.exit(rc)


if __name__ == "__main__":
    main()
