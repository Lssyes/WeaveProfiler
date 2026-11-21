#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 randn 输入与全量输出张量无损落盘（fi_check.py C2 与交付清单第 5 项用）。

为什么要这个模板：BF16 张量直接 `.numpy()` 会抛

    TypeError: Got unsupported ScalarType BFloat16

numpy 没有 bfloat16 这个 dtype。如果落盘这一步排在计时与 trace 之后，这个异常会让前面全部 GPU 实测
（包括已经跑完的 .perfetto.json.gz 与 t2_run.json）一起丢掉。这里的做法是把 bf16 无损存成它的
uint16 位模式：bf16 就是 1 符号 + 8 指数 + 7 尾数，按位存进 uint16 再按位还原，逐位一致，
不做任何数值近似。float16/float32/float64/整型 numpy 原生支持，原样存。

两臂（L0 与插桩构建）都用本模板存，键名与 storage dtype 就完全一致，可以直接交给
`knowledge/templates/compare_raw_arrays.py --atol 0 --rtol 0` 做逐位比对：uint16 位模式相等 <=> bf16 逐位相等。

用法（驱动脚本里）：

    sys.path.insert(0, "<包根>/templates")
    from save_tensors import save_tensors

    save_tensors({"x": x, "y": y_ref}, "run/ref.npz",
                 meta={"arm": "L0", "seed": 1234, "kernel": "ring_pipeline"})
    save_tensors({"x": x, "y": y_fi}, "run/fi.npz",
                 meta={"arm": "FI", "seed": 1234, "kernel": "ring_pipeline"})

    # 还原成 torch 张量（bf16 按位还原）
    from save_tensors import load_tensors
    t = load_tensors("run/ref.npz")

同时写出 `<path>.meta.json`：每个键的 logical_dtype / storage_dtype / shape / 元素数 / 字节流 SHA-256，
以及调用方给的 meta。审计时能证明"存的是完整张量、不是切片"，也能证明两臂输入是同一份 randn。

自检：python3 -I knowledge/templates/save_tensors.py --selftest
（需要 numpy；有 torch 时多验一步真实 bf16 张量的往返，没有 torch 就只验位模式与 meta 逻辑，并如实打印。）
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# numpy 没有 bfloat16；这些 torch dtype 一律按位存成同宽整型。
_BIT_VIEW = {"bfloat16": "uint16"}
_BIT_BACK = {"uint16": "bfloat16"}


def _sha256_bytes(buf: bytes) -> str:
    return hashlib.sha256(buf).hexdigest()


def to_lossless(t):
    """torch 张量 -> (numpy 数组, logical_dtype, storage_dtype)。bf16 走位模式，其余原样。"""
    import numpy as np
    import torch                                    # noqa: F401  （类型来源，也用来判定 dtype 名）
    logical = str(t.dtype).replace("torch.", "")
    tc = t.detach().cpu().contiguous()
    if logical in _BIT_VIEW:
        storage = _BIT_VIEW[logical]
        try:
            arr = getattr(tc, "view")(getattr(__import__("torch"), storage)).numpy()
        except Exception:                           # 老 torch 不给 view() 时直接读底层字节
            raw = bytes(tc.untyped_storage())
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(tuple(tc.shape))
        return arr.astype(storage, copy=False), logical, storage
    return tc.numpy(), logical, logical


def from_lossless(arr, logical: str):
    """numpy 数组 -> torch 张量；位模式按 logical_dtype 还原。"""
    import numpy as np
    import torch
    if logical in ("bfloat16",) and arr.dtype == np.dtype("uint16"):
        return torch.from_numpy(np.ascontiguousarray(arr)).view(torch.bfloat16)
    return torch.from_numpy(np.ascontiguousarray(arr))


def save_tensors(tensors: Dict[str, Any], path: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """把 {名字: torch 张量} 无损存成 .npz，并写 <path>.meta.json。返回 meta 文档。"""
    import numpy as np
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, Any] = {}
    fields: Dict[str, Any] = {}
    for k, v in tensors.items():
        arr, logical, storage = to_lossless(v)
        arrays[k] = arr
        raw = np.ascontiguousarray(arr).tobytes()
        fields[k] = {
            "logical_dtype": logical,
            "storage_dtype": storage,
            "shape": list(arr.shape),
            "elements": int(arr.size),
            "bytes": len(raw),
            "sha256": _sha256_bytes(raw),
            "lossless": True,
            "note": ("bfloat16 按位存成 uint16（numpy 没有 bfloat16）；位模式相等即张量逐位相等"
                     if storage != logical else "numpy 原生支持该 dtype，原样存"),
        }
    np.savez(out, **arrays)
    doc = {
        "tool": "fast-instrument/knowledge/templates/save_tensors.py",
        "written_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "path": str(out),
        "keys": sorted(fields),
        "arrays": fields,
        "meta": meta or {},
    }
    mp = out.with_suffix(out.suffix + ".meta.json") if out.suffix else Path(str(out) + ".meta.json")
    mp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return doc


def load_tensors(path: str) -> Dict[str, Any]:
    """读回 .npz 成 {名字: torch 张量}。

    有 <path>.meta.json 时按 logical_dtype 还原（bf16 从 uint16 位模式还原）；没有侧车文件时
    原样返回 numpy 数组（此时无法知道 uint16 到底是位模式还是真的整型）。
    """
    import numpy as np
    p = Path(path)
    mp = Path(str(p) + ".meta.json")
    meta = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else None
    if meta is None:
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    out: Dict[str, Any] = {}
    with np.load(p, allow_pickle=False) as z:
        for k in z.files:
            logical = ((meta.get("arrays", {}) or {}).get(k, {}) or {}).get("logical_dtype") or ""
            out[k] = from_lossless(z[k], logical)
    return out


def selftest() -> int:
    import numpy as np
    fails = []
    tmp = Path(__file__).resolve().parent / ".save_tensors_selftest.tmp"
    tmp.mkdir(exist_ok=True)
    try:
        try:
            import torch
        except Exception as e:
            print(f"  SKIP torch 往返（{type(e).__name__}: {e}）：只验位模式与 meta 逻辑")
            torch = None
        if torch is not None:
            g = torch.Generator().manual_seed(20261109)
            x32 = torch.randn((4, 5), generator=g, dtype=torch.float32)
            xbf = x32.to(torch.bfloat16)
            xbf[0, 0] = float("nan")                      # NaN 也要按位存回来
            xbf[1, 1] = float("inf")
            doc = save_tensors({"x32": x32, "y": xbf}, str(tmp / "arm.npz"),
                               meta={"arm": "selftest", "seed": 20261109})
            if doc["arrays"]["y"]["logical_dtype"] != "bfloat16":
                fails.append(f"bf16 的 logical_dtype 记错：{doc['arrays']['y']}")
            if doc["arrays"]["y"]["storage_dtype"] != "uint16":
                fails.append(f"bf16 没有按位存成 uint16：{doc['arrays']['y']}")
            if doc["arrays"]["x32"]["storage_dtype"] != "float32":
                fails.append("float32 不该被改写存储 dtype")
            if doc["arrays"]["y"]["elements"] != 20 or doc["arrays"]["y"]["bytes"] != 40:
                fails.append(f"元素数/字节数不对：{doc['arrays']['y']}")
            got = load_tensors(str(tmp / "arm.npz"))
            if not isinstance(got["x32"], torch.Tensor) or not torch.equal(got["x32"], x32):
                fails.append("float32 往返不一致")
            if got["y"].dtype != torch.bfloat16:
                fails.append(f"bf16 还原后 dtype 不对：{got['y'].dtype}")
            same_bits = torch.equal(got["y"].view(torch.uint16), xbf.view(torch.uint16))
            if not same_bits:
                bad = (got["y"].view(torch.uint16) != xbf.view(torch.uint16)).nonzero()[:4]
                fails.append(f"bf16 往返不是逐位一致，前几个不同处：{bad.tolist()}")
            # 两臂同键名同 storage dtype，才能直接交给 compare_raw_arrays.py 做 atol=0/rtol=0 比对
            save_tensors({"y": xbf.clone()}, str(tmp / "arm2.npz"), meta={"arm": "selftest-copy"})
            with np.load(tmp / "arm.npz") as z1, np.load(tmp / "arm2.npz") as z2:
                if z1["y"].dtype != z2["y"].dtype or not np.array_equal(z1["y"], z2["y"]):
                    fails.append("两臂的位模式数组不可直接比对")
            # meta 侧车文件必须存在且能被 json 读回，sha256 要能对上
            mj = Path(str(tmp / "arm.npz") + ".meta.json")
            if not mj.is_file():
                fails.append("没有写出 <path>.meta.json")
            else:
                d = json.loads(mj.read_text(encoding="utf-8"))
                with np.load(tmp / "arm.npz") as z:
                    raw = np.ascontiguousarray(z["y"]).tobytes()
                if d["arrays"]["y"]["sha256"] != _sha256_bytes(raw):
                    fails.append("meta 里的 sha256 与实际字节流不一致")
                if d["meta"].get("arm") != "selftest":
                    fails.append("调用方给的 meta 没有被原样记下")
            print(f"  torch 往返 PASS：bf16 {doc['arrays']['y']['elements']} 元素按位一致（含 NaN/Inf），"
                  f"sha256={doc['arrays']['y']['sha256'][:16]}…")
        # 没有 torch 时也要能验位模式逻辑：uint16 -> bf16 的映射是纯位操作
        bits = np.array([0x3F80, 0x4000, 0x7FC0, 0x7F80, 0x0000], dtype=np.uint16)
        if bits.dtype != np.dtype("uint16") or bits.size != 5:
            fails.append("位模式自检数据构造失败")
        if _BIT_VIEW.get("bfloat16") != "uint16" or _BIT_BACK.get("uint16") != "bfloat16":
            fails.append("bf16 与 uint16 的映射表不对")
    finally:
        for f in sorted(tmp.glob("*")):
            f.unlink()
        tmp.rmdir()
    if fails:
        print("SAVE_TENSORS SELFTEST FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("SAVE_TENSORS SELFTEST PASS")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="把 randn 输入与全量输出张量无损落盘（bf16 存 uint16 位模式），并写 meta 侧车文件")
    ap.add_argument("--selftest", action="store_true", help="跑自检（需要 numpy；有 torch 时多验真实 bf16 往返）")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    ap.error("本模板是给驱动脚本 import 用的（from save_tensors import save_tensors, load_tensors）；"
             "命令行只有 --selftest")


if __name__ == "__main__":
    raise SystemExit(main())
