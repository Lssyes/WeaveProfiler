# -*- coding: utf-8 -*-
"""fi_relane：把一份 Chrome trace / Perfetto json 里“同一泳道内互相交叉（不是严格嵌套）的区间”拆到子泳道。

为什么需要：同一个 warp/CTA 同时在途多个实例（例如 5 个 prep 块的 TMA 同时发出）时，它们的区间互相交叉，
Perfetto 会把它们画在同一条泳道里叠成一团，看不出谁是谁。本工具对每条 (pid, tid) 泳道做“嵌套感知的贪心分配”：
按开始时间升序（同时开始的长者优先）扫描，放进第一条“能严格嵌套”的子泳道（该子泳道当前没有在途区间，
或者在途栈顶区间完全包住它且名字不同——同名区间视为并发实例，不互相嵌套）；放不下就开新子泳道。严格嵌套的区间（外层“块计算”包内层“装载”）留在同一子泳道。

用法：
  python3 -I fi_relane.py in.perfetto.json[.gz] -o out.perfetto.json[.gz] [--key instance]
  --key：如果区间 args 里有这个字段（例如 prep_instance），优先按它的值分子泳道（同值同泳道），其余仍用贪心。
  python3 -I fi_relane.py --selftest
输出泳道 tid = 原 tid * 64 + 子泳道号，线程名后缀 “· 实例k”。摘要打印到 stdout（每条原泳道拆成几条）。
"""
import argparse, gzip, json, os, sys
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

EPS = 1e-9


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    return d if isinstance(d, dict) else {"traceEvents": d}


def save(d, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = json.dumps(d, ensure_ascii=False).encode("utf-8")
    with (gzip.open(path, "wb") if path.endswith(".gz") else open(path, "wb")) as f:
        f.write(data)


def assign(spans, key=None):
    """spans: list of dict(ph=X)。返回同长度的子泳道号列表。"""
    n = len(spans)
    sub = [0] * n
    order = sorted(range(n), key=lambda k: (spans[k]["ts"], -spans[k].get("dur", 0)))
    keyed = {}
    if key:
        vals = sorted({spans[k].get("args", {}).get(key) for k in range(n)} - {None}, key=str)
        keyed = {v: j for j, v in enumerate(vals)}
    stacks = []  # 每条子泳道一个栈：[(end, name)]
    base_for_keyed = None
    if keyed:
        # 带 key 的区间：子泳道 = keyed[值]；不带 key 的区间放在 keyed 之后，用贪心
        base_for_keyed = 0
        stacks = [[] for _ in keyed]
    for k in order:
        s = spans[k]
        b, e = s["ts"], s["ts"] + s.get("dur", 0)
        kv = s.get("args", {}).get(key) if key else None
        if kv is not None and kv in keyed:
            sub[k] = keyed[kv]
            st = stacks[sub[k]]
            while st and st[-1][0] <= b + EPS:
                st.pop()
            st.append((e, s.get("name")))
            continue
        start_j = len(keyed)
        placed = None
        for j in range(start_j, len(stacks)):
            st = stacks[j]
            while st and st[-1][0] <= b + EPS:
                st.pop()
            if not st or (e <= st[-1][0] + EPS and st[-1][1] != s.get("name")):
                placed = j
                break
        if placed is None:
            stacks.append([])
            placed = len(stacks) - 1
        stacks[placed].append((e, s.get("name")))
        sub[k] = placed
    return sub


def normalize_ids(d):
    """别的工具（如 Proton）用字符串作 pid/tid（"kernel Core0 CTA6" / "warp 0 (line 0)"）。
    统一换成整数，并把原字符串写成 process_name/thread_name，Perfetto 显示不变。"""
    ev = d["traceEvents"]
    pmap, tmap, meta = {}, {}, []
    for e in ev:
        p, t = e.get("pid", 0), e.get("tid", 0)
        if not isinstance(p, int):
            if p not in pmap:
                pmap[p] = 1000000 + len(pmap)
                meta.append(dict(ph="M", pid=pmap[p], name="process_name", args=dict(name=str(p))))
            e["pid"] = pmap[p]
        if not isinstance(t, int):
            k = (e["pid"], t)
            if k not in tmap:
                tmap[k] = len([1 for q in tmap if q[0] == e["pid"]]) + 1
                meta.append(dict(ph="M", pid=e["pid"], tid=tmap[k], name="thread_name", args=dict(name=str(t))))
            e["tid"] = tmap[k]
    ev.extend(meta)
    return bool(pmap or tmap)


def relane(d, key=None):
    normalize_ids(d)
    ev = d["traceEvents"]
    lanes = defaultdict(list)
    for i, e in enumerate(ev):
        if e.get("ph") == "X":
            lanes[(e.get("pid"), e.get("tid"))].append(i)
    tname = {(e.get("pid"), e.get("tid")): e.get("args", {}).get("name", "") for e in ev
             if e.get("ph") == "M" and e.get("name") == "thread_name"}
    split = {}
    for lk, idx in lanes.items():
        spans = [ev[i] for i in idx]
        sub = assign(spans, key)
        nsub = max(sub) + 1 if sub else 1
        split[lk] = nsub
        for i, j in zip(idx, sub):
            ev[i]["tid"] = ev[i].get("tid", 0) * 64 + j
            if nsub > 1:
                ev[i].setdefault("args", {})["sublane"] = j
    # 非 X 事件（瞬时事件、计数器）留在 0 号子泳道
    for e in ev:
        if e.get("ph") in ("i", "I", "B", "E") and (e.get("pid"), e.get("tid")) in split:
            e["tid"] = e.get("tid", 0) * 64
    out_meta = [e for e in ev if not (e.get("ph") == "M" and e.get("name") in ("thread_name", "thread_sort_index"))]
    for (p, t), nsub in sorted(split.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        base = tname.get((p, t), f"tid {t}")
        for j in range(nsub):
            nt = (t or 0) * 64 + j
            out_meta.append(dict(ph="M", pid=p, tid=nt, name="thread_name", args=dict(name=base + (f" · 实例{j}" if nsub > 1 else ""))))
            out_meta.append(dict(ph="M", pid=p, tid=nt, name="thread_sort_index", args=dict(sort_index=nt)))
    d["traceEvents"] = out_meta
    d.setdefault("otherData", {})["relane"] = dict(tool="fi_relane", key=key, lanes=len(split),
                                                    lanes_split=sum(1 for v in split.values() if v > 1),
                                                    max_sublanes=max(split.values()) if split else 0)
    return d, split


def selftest():
    X = lambda ts, dur, name, tid=0, **a: dict(ph="X", pid=1, tid=tid, ts=ts, dur=dur, name=name, args=a)
    # 外层块计算 0..100 包住 3 个交叉的装载实例；另一个独立区间 120..130
    d = {"traceEvents": [X(0, 100, "块计算"), X(1, 30, "装载", inst=0), X(1, 33, "装载", inst=1), X(2, 31, "装载", inst=2),
                         X(120, 10, "收尾"), dict(ph="M", pid=1, tid=0, name="thread_name", args=dict(name="CTA0"))]}
    out, split = relane(json.loads(json.dumps(d)))
    xs = sorted([e for e in out["traceEvents"] if e["ph"] == "X"], key=lambda e: (e["ts"], -e["dur"]))
    subs = [e["tid"] % 64 for e in xs]
    # 块计算在 0；第一个装载(1..34 长者优先)嵌套在块计算里也在 0；另两个交叉 → 1、2；收尾回到 0
    assert subs == [0, 0, 1, 2, 0], subs
    assert split[(1, 0)] == 3
    names = [e["args"]["name"] for e in out["traceEvents"] if e["ph"] == "M" and e["name"] == "thread_name"]
    assert names == ["CTA0 · 实例0", "CTA0 · 实例1", "CTA0 · 实例2"], names
    out2, _ = relane(json.loads(json.dumps(d)), key="inst")
    xs2 = sorted([e for e in out2["traceEvents"] if e["ph"] == "X"], key=lambda e: (e["ts"], -e["dur"]))
    subs2 = {(e["name"], e["args"].get("inst")): e["tid"] % 64 for e in xs2}
    assert subs2[("装载", 0)] == 0 and subs2[("装载", 1)] == 1 and subs2[("装载", 2)] == 2 and subs2[("块计算", None)] == 3, subs2
    # 严格嵌套不拆
    d3 = {"traceEvents": [X(0, 100, "外"), X(10, 20, "中"), X(12, 5, "内"), X(40, 10, "中")]}
    _, s3 = relane(d3)
    assert s3[(1, 0)] == 1, s3
    print("SELFTEST PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--key")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.inp and a.out):
        ap.error("需要输入文件和 -o")
    d, split = relane(load(a.inp), a.key)
    r = d["otherData"]["relane"]
    print(f"泳道 {r['lanes']} 条，其中 {r['lanes_split']} 条有交叉区间被拆开，单条最多拆成 {r['max_sublanes']} 条子泳道")
    hist = defaultdict(int)
    for v in split.values():
        hist[v] += 1
    print("子泳道数分布：" + "，".join(f"{k} 条子泳道 × {v}" for k, v in sorted(hist.items())))
    save(d, a.out)
    print("写出", a.out)


if __name__ == "__main__":
    main()
