# -*- coding: utf-8 -*-
"""WeaveProfiler (`wp`) canonical CuTeDSL recorder frontend.

Thin first-class facade over ``fi_cute.py`` (same device code, same FILITE02 dump).

Environment switches (canonical spelling first; mapped onto the legacy names
*before* ``fi_cute`` is imported because ``fi_cute`` reads them at import time):

  WP_LEVEL -> FI_LEVEL   0 disables writes, 1 records OPEN/CLOSE only, 2 records events
  WP_GRAN  -> FI_GRAN    0 one lane per warp, 1 one lane per CTA (linear thread 0)

Device API (value-style, CuTeDSL SSA)::

    ctx = wp_open(buf.records_cute, buf.overflow_cute, role, warps_per_cta)
    ctx = wp_b(ctx, EV_LOAD, tile)
    ctx = wp_e(ctx, EV_LOAD, tile)
    ctx = wp_close(ctx)

Pure-host helpers (``write_names``, ``open_perfetto``, ``WP_LEVEL``/``WP_GRAN``)
work without CUTLASS installed; device symbols raise a clear ImportError on first
use when ``cutlass`` is missing, instead of breaking metadata-only imports.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
import subprocess
from typing import Any, Mapping

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _map_env(canonical: str, legacy: str) -> None:
    value = os.environ.get(canonical)
    if value is None or value == "":
        return
    prior = os.environ.get(legacy)
    if prior not in (None, "") and prior != value:
        raise RuntimeError(f"{canonical}={value!r} conflicts with {legacy}={prior!r}; set only one spelling")
    os.environ[legacy] = value


for _canonical, _legacy in (("WP_LEVEL", "FI_LEVEL"), ("WP_GRAN", "FI_GRAN")):
    _map_env(_canonical, _legacy)

WP_LEVEL = int(os.environ.get("FI_LEVEL", "2"))
WP_GRAN = int(os.environ.get("FI_GRAN", "0"))
if WP_LEVEL not in (0, 1, 2):
    raise ValueError(f"WP_LEVEL must be 0, 1 or 2, got {WP_LEVEL}")
if WP_GRAN not in (0, 1):
    raise ValueError(f"WP_GRAN must be 0 or 1, got {WP_GRAN}")

TAG_END = 0x80000000
TAG_OPEN = 0x40000000
TAG_MARK = 0x20000000
TAG_CLOSE = 0x08000000
TAG_IDMASK = 0x0FFF
WP_TAG_END, WP_TAG_OPEN, WP_TAG_MARK, WP_TAG_CLOSE, WP_TAG_IDMASK = TAG_END, TAG_OPEN, TAG_MARK, TAG_CLOSE, TAG_IDMASK

_DEVICE_EXPORTS = {
    "WpCtx": "FiCtx",
    "WpBuffer": "FiBuffer",
    "wp_open": "fi_open",
    "wp_open_by": "fi_open_by",
    "wp_b": "fi_b",
    "wp_e": "fi_e",
    "wp_m": "fi_m",
    "wp_close": "fi_close",
    "wp_attach": "fi_attach",
    "wp_get_attached": "fi_get_attached",
    "wp_wrap": "_FiCuteProxy",
    "wp_descriptor": "_FiCuteDescriptor",
    "WP_OPEN": "fi_open",
    "WP_OPEN_BY": "fi_open_by",
    "WP_B": "fi_b",
    "WP_E": "fi_e",
    "WP_M": "fi_m",
    "WP_CLOSE": "fi_close",
}

try:
    import fi_cute as _fi  # noqa: E402  (env mapping must happen first)
    _IMPORT_ERROR: BaseException | None = None
except ImportError as _exc:  # CUTLASS / CuTeDSL not installed: keep host helpers importable
    _fi = None
    _IMPORT_ERROR = _exc


def _name_key(value: Any) -> str:
    raw = getattr(value, "value", value)
    if isinstance(raw, bool):
        raise TypeError("names.json keys must be integers, not bool")
    return str(int(raw))


if _fi is not None:
    for _name, _target in _DEVICE_EXPORTS.items():
        globals()[_name] = getattr(_fi, _target)
    write_names = _fi.write_names
else:
    def __getattr__(name: str) -> Any:  # PEP 562 lazy failure for device symbols
        if name in _DEVICE_EXPORTS:
            raise ImportError(
                f"wp_cute.{name} needs the CUTLASS CuTeDSL runtime (pip install nvidia-cutlass-dsl); "
                f"import of fi_cute failed: {_IMPORT_ERROR}"
            )
        raise AttributeError(f"module 'wp_cute' has no attribute {name!r}")

    def write_names(
        path: str,
        events: Mapping[Any, str],
        roles: Mapping[Any, str] | None = None,
        aux: Mapping[Any, str] | None = None,
        group: str = "role",
        pair: str = "stack",
        label_aux: bool | None = None,
        **extra: Any,
    ) -> str:
        """Host-only names.json writer (same document shape as fi_cute.write_names)."""
        if group not in ("role", "sm", "cu", "cta"):
            raise ValueError(f"group must be role/sm/cu/cta, got {group!r}")
        if pair not in ("stack", "aux"):
            raise ValueError(f"pair must be stack/aux, got {pair!r}")
        obj: dict[str, Any] = {
            "events": {_name_key(k): str(v) for k, v in (events or {}).items()},
            "roles": {_name_key(k): str(v) for k, v in (roles or {}).items()},
            "group": group,
            "pair": pair,
        }
        if aux:
            obj["aux"] = {_name_key(k): str(v) for k, v in aux.items()}
        if label_aux is not None:
            obj["label_aux"] = bool(label_aux)
        obj.update(extra)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return path


def open_perfetto(dump_path: str, names_path: str, out_path: str) -> dict[str, Any]:
    """Decode a CuTeDSL dump with the canonical package decoder (4_validate/tools/wp_decode.py)."""
    decoder = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "4_validate", "tools", "wp_decode.py"))
    if not os.path.isfile(decoder):
        raise FileNotFoundError(f"decoder not found: {decoder}")
    proc = subprocess.run(
        [sys.executable, "-B", decoder, dump_path, "--names", names_path, "-o", out_path],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    summary_path = out_path.replace(".gz", "").replace(".json", "") + ".summary.json"
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def wp_available() -> bool:
    """True when the CuTeDSL device frontend (fi_cute + cutlass) imported successfully."""
    return _fi is not None


__all__ = sorted((set(_DEVICE_EXPORTS) if _fi is not None else set()) | {
    "WP_LEVEL", "WP_GRAN", "WP_TAG_END", "WP_TAG_OPEN", "WP_TAG_MARK", "WP_TAG_CLOSE", "WP_TAG_IDMASK",
    "write_names", "open_perfetto", "wp_available",
})
