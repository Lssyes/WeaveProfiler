#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin CLI/API forwarder to ``weaveprof/pipeline/instrument/incremental_reuse.py``."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parents[1] / "instrument" / "incremental_reuse.py"
_SPEC = importlib.util.spec_from_file_location("wp_incremental_reuse_impl", _TARGET)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

reuse_probes_incremental = _MOD.reuse_probes_incremental
BuildArtifactCache = _MOD.BuildArtifactCache
main = _MOD.main

if __name__ == "__main__":
    raise SystemExit(main())
