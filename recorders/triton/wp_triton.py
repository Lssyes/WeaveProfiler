# -*- coding: utf-8 -*-
"""WeaveProfiler (`wp`) canonical Triton recorder frontend for NVIDIA (PTX backend).

First-class facade over ``fi_triton.py``: same device code, same FILITE02 dump,
decoded by ``weaveprof/pipeline/validate/tools/wp_decode.py``.

Environment switches (canonical spelling mapped onto the legacy names on import):

  WP_LEVEL -> FI_LEVEL   0 = empty, 1 = OPEN/CLOSE only, 2 = all events
  WP_GRAN  -> FI_GRAN    0 = warp lanes, 1 = one lane per CTA
  WP_TIMER -> FI_TIMER   0 = %globaltimer, 1 = %clock64 + OPEN/CLOSE anchors

Triton bakes these into PTX through the ``FiCfg``/``WpCfg`` constexpr; ``wp_cfg()``
builds one whose defaults come from the switches above::

    WP = wp_cfg(cap=buf.cap, warps=num_warps, n_slots=buf.n_slots, threads=num_warps * 32)
    n = wp_open(rec, ovf, pid, WP, 0)
    n = wp_b(rec, ovf, pid, n, WP, EV_LOAD, tile)
    n = wp_e(rec, ovf, pid, n, WP, EV_LOAD, tile)
    n = wp_close(rec, ovf, pid, n, WP)

Host helpers (``write_names``, ``WP_LEVEL``/``WP_GRAN``/``WP_TIMER``) import without
Triton; device symbols raise a clear ImportError on first use when Triton is missing.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
from typing import Any, Mapping, Optional

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


for _canonical, _legacy in (("WP_LEVEL", "FI_LEVEL"), ("WP_GRAN", "FI_GRAN"), ("WP_TIMER", "FI_TIMER")):
    _map_env(_canonical, _legacy)

WP_LEVEL = int(os.environ.get("FI_LEVEL", "2"))
WP_GRAN = int(os.environ.get("FI_GRAN", "0"))
WP_TIMER = int(os.environ.get("FI_TIMER", "0"))
for _label, _value, _allowed in (("WP_LEVEL", WP_LEVEL, (0, 1, 2)), ("WP_GRAN", WP_GRAN, (0, 1)),
                                 ("WP_TIMER", WP_TIMER, (0, 1))):
    if _value not in _allowed:
        raise ValueError(f"{_label} must be one of {_allowed}, got {_value}")

WP_TAG_END = 0x80000000
WP_TAG_OPEN = 0x40000000
WP_TAG_MARK = 0x20000000
WP_TAG_CLK = 0x10000000
WP_TAG_CLOSE = 0x08000000
WP_TAG_IDMASK = 0x0FFF

_DEVICE_EXPORTS = {
    "WpCfg": "FiCfg",
    "WpBuffer": "FiBuffer",
    "wp_open": "fi_open",
    "wp_open_by": "fi_open_by",
    "wp_b": "fi_b",
    "wp_e": "fi_e",
    "wp_m": "fi_m",
    "wp_bg": "fi_bg",
    "wp_eg": "fi_eg",
    "wp_mg": "fi_mg",
    "wp_close": "fi_close",
    "wp_b_by": "fi_b_by",
    "wp_e_by": "fi_e_by",
    "wp_m_by": "fi_m_by",
    "wp_close_by": "fi_close_by",
    "wp_attach": "fi_attach",
    "wp_get_attached": "fi_get_attached",
    "wp_wrap": "_FiTritonProxy",
    "WP_OPEN": "fi_open",
    "WP_OPEN_BY": "fi_open_by",
    "WP_B": "fi_b",
    "WP_E": "fi_e",
    "WP_M": "fi_m",
    "WP_CLOSE": "fi_close",
}

try:
    import fi_triton as _fi  # noqa: E402
    _IMPORT_ERROR: BaseException | None = None
except ImportError as _exc:  # Triton / torch missing: keep host helpers importable
    _fi = None
    _IMPORT_ERROR = _exc


def _name_key(value: Any) -> str:
    raw = getattr(value, "value", value)
    if isinstance(raw, bool):
        raise TypeError("names.json keys must be integers, not bool")
    return str(int(raw))


def _write_names_host(path, events: Mapping[Any, str], roles: Optional[Mapping[Any, str]] = None,
                      aux: Optional[Mapping[Any, str]] = None, group: str = "role", pair: str = "stack",
                      label_aux: Optional[bool] = None, **extra: Any) -> str:
    if group not in ("role", "sm", "cu", "cta"):
        raise ValueError(f"group must be role/sm/cu/cta, got {group!r}")
    if pair not in ("stack", "aux"):
        raise ValueError(f"pair must be stack/aux, got {pair!r}")
    obj: dict = {
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
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


if _fi is not None:
    for _name, _target in _DEVICE_EXPORTS.items():
        globals()[_name] = getattr(_fi, _target)
    write_names = _fi.write_names
else:
    write_names = _write_names_host

    def __getattr__(name: str) -> Any:  # PEP 562 lazy failure for device symbols
        if name in _DEVICE_EXPORTS:
            raise ImportError(f"wp_triton.{name} needs Triton + torch (CUDA); import of fi_triton failed: {_IMPORT_ERROR}")
        raise AttributeError(f"module 'wp_triton' has no attribute {name!r}")


def wp_cfg(**kwargs: Any):
    """Build a ``WpCfg`` (``FiCfg``) whose level/gran/timer default to WP_LEVEL/WP_GRAN/WP_TIMER."""
    if _fi is None:
        raise ImportError(f"wp_cfg needs Triton; import of fi_triton failed: {_IMPORT_ERROR}")
    kwargs.setdefault("level", WP_LEVEL)
    kwargs.setdefault("gran", WP_GRAN)
    kwargs.setdefault("timer", WP_TIMER)
    return _fi.FiCfg(**kwargs)


def wp_available() -> bool:
    return _fi is not None


__all__ = sorted((set(_DEVICE_EXPORTS) if _fi is not None else set()) | {
    "WP_LEVEL", "WP_GRAN", "WP_TIMER", "WP_TAG_END", "WP_TAG_OPEN", "WP_TAG_MARK", "WP_TAG_CLK",
    "WP_TAG_CLOSE", "WP_TAG_IDMASK", "write_names", "wp_cfg", "wp_available",
})
