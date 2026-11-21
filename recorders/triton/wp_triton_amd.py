# -*- coding: utf-8 -*-
"""WeaveProfiler (`wp`) canonical Triton recorder frontend for AMD (gfx942 / wave64, MI308X).

First-class facade over ``fi_triton_amd.py``: same device code, same FILITE02 dump,
decoded by ``weaveprof/pipeline/validate/tools/wp_decode.py``.

Environment switches (canonical spelling mapped onto the legacy names on import):

  WP_LEVEL -> FI_LEVEL   0 = empty, 1 = OPEN/CLOSE only, 2 = all events
  WP_GRAN  -> FI_GRAN    0 = wave lanes, 1 = one lane per CTA
  WP_TIMER -> FI_TIMER   0 = s_memrealtime, 1 = s_memtime + OPEN/CLOSE anchors

``wp_buffer()`` builds a ``WpBuffer`` whose ``gran``/``timer_mode`` default to the
switches above.  On AMD the recording wave is chosen from HW_ID inside ``wp_open``
(Triton exposes no threadIdx), so ``wp_open_by`` is not available on this backend.
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
    "wp_open": "fi_open",
    "wp_b": "fi_b",
    "wp_e": "fi_e",
    "wp_m": "fi_m",
    "wp_close": "fi_close",
    "wp_open_16b": "fi_open_16b",
    "wp_b_16b": "fi_b_16b",
    "wp_e_16b": "fi_e_16b",
    "wp_m_16b": "fi_m_16b",
    "wp_close_16b": "fi_close_16b",
    "WP_OPEN": "fi_open",
    "WP_B": "fi_b",
    "WP_E": "fi_e",
    "WP_M": "fi_m",
    "WP_CLOSE": "fi_close",
}
_HOST_EXPORTS = {
    "WpBuffer": "FiBuffer",
    "WpPackedBuffer": "FiPackedBuffer",
    "wp_attach": "fi_attach",
    "wp_get_attached": "fi_get_attached",
    "wp_wrap": "_FiTritonProxy",
}

try:
    import fi_triton_amd as _fi  # noqa: E402  (imports without torch/triton; device fns need triton)
    _IMPORT_ERROR: BaseException | None = None
except ImportError as _exc:
    _fi = None
    _IMPORT_ERROR = _exc

_MISSING: dict = {}
for _table in (_HOST_EXPORTS, _DEVICE_EXPORTS):
    for _name, _target in _table.items():
        if _fi is not None and hasattr(_fi, _target):
            globals()[_name] = getattr(_fi, _target)
        else:
            _MISSING[_name] = _target


def __getattr__(name: str) -> Any:  # PEP 562 lazy failure for symbols that need triton
    if name in _MISSING:
        raise ImportError(
            f"wp_triton_amd.{name} needs Triton (ROCm) for device code; "
            f"fi_triton_amd.{_MISSING[name]} is unavailable ({_IMPORT_ERROR or 'triton not importable'})"
        )
    raise AttributeError(f"module 'wp_triton_amd' has no attribute {name!r}")


def wp_open_by(*_args: Any, **_kwargs: Any):
    """Not supported on AMD Triton: the recording wave is selected from HW_ID inside ``wp_open``."""
    raise NotImplementedError(
        "wp_open_by is not available on the AMD Triton backend: wp_open selects the recording wave "
        "from HW_ID (lowest SIMD/wave per CTA for WP_GRAN=1, lane 0 of every wave for WP_GRAN=0)"
    )


WP_OPEN_BY = wp_open_by


def _name_key(value: Any) -> str:
    raw = getattr(value, "value", value)
    if isinstance(raw, bool):
        raise TypeError("names.json keys must be integers, not bool")
    return str(int(raw))


def write_names(path, events: Mapping[Any, str], roles: Optional[Mapping[Any, str]] = None,
                aux: Optional[Mapping[Any, str]] = None, group: str = "cu", pair: str = "stack",
                label_aux: Optional[bool] = None, **extra: Any) -> str:
    """Write names.json (int/constexpr keys normalized to JSON strings); AMD default group is ``cu``."""
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


def wp_buffer(n_ctas: int, cap: int, **kwargs: Any):
    """Build a ``WpBuffer`` whose ``gran``/``timer_mode`` default to WP_GRAN/WP_TIMER."""
    if _fi is None:
        raise ImportError(f"wp_buffer needs fi_triton_amd: {_IMPORT_ERROR}")
    kwargs.setdefault("gran", WP_GRAN)
    kwargs.setdefault("timer_mode", WP_TIMER)
    return _fi.FiBuffer(n_ctas=n_ctas, cap=cap, **kwargs)


def wp_available() -> bool:
    return _fi is not None and not any(name in _MISSING for name in _DEVICE_EXPORTS)


__all__ = sorted({n for n in (*_HOST_EXPORTS, *_DEVICE_EXPORTS) if n not in _MISSING} | {
    "WP_LEVEL", "WP_GRAN", "WP_TIMER", "WP_TAG_END", "WP_TAG_OPEN", "WP_TAG_MARK", "WP_TAG_CLK",
    "WP_TAG_CLOSE", "WP_TAG_IDMASK", "wp_open_by", "WP_OPEN_BY", "write_names", "wp_buffer", "wp_available",
})
