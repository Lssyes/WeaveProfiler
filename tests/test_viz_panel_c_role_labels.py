"""Regression tests for Panel C/D long warp/CTA-slot role labels in ``weaveprof.viz``.

Synthetic 4-panel traces are modeled on ``tests/test_wp_megakernel.py:1194-1222``
(``test_async_engine_depth_pins_4panel_and_triton_ptx_pass`` item #3: hardware
utilization curves + dual MEM/MMA async engines + deterministic micro-zoom),
rendered at the default ``dpi`` used by ``run_megakernel_audit``.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path
import shutil
import tempfile
import unittest

from matplotlib.axes import Axes

from weaveprof import viz
from weaveprof.viz import render_swhw_overview_png

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
BASE_ROLE = "Consumer WG1/WG2 (ping-pong WGMMA)"
ROLE_FILL = " (ping-pong WGMMA) Consumer WG1/WG2"


def _role_for_tick_len(tick_len: int, prefix_len: int, kind: str) -> str:
    body_len = tick_len - prefix_len
    assert body_len > 0, (tick_len, prefix_len)
    if kind == "normal":
        return (BASE_ROLE + ROLE_FILL * 8)[:body_len]
    if kind == "all_W":
        return "W" * body_len
    raise ValueError(f"unknown kind: {kind}")


def _build_synth_4panel_trace(role_str: str, *, cta_slots: bool) -> dict:
    evs = [
        {"ph": "C", "pid": 90000, "name": "Tensor Active [Throughput %]", "ts": 0.0, "args": {"value": 72.0}},
        {"ph": "C", "pid": 90000, "name": "Tensor Active [Throughput %]", "ts": 200.0, "args": {"value": 68.0}},
        {"ph": "C", "pid": 90000, "name": "DRAM Read Bandwidth [Throughput %]", "ts": 0.0, "args": {"value": 40.0}},
        {"ph": "C", "pid": 90000, "name": "DRAM Write Bandwidth [Throughput %]", "ts": 0.0, "args": {"value": 20.0}},
        {"ph": "C", "pid": 90000, "name": "DRAM Read Bandwidth [Throughput %]", "ts": 200.0, "args": {"value": 35.0}},
        {"ph": "C", "pid": 90000, "name": "DRAM Write Bandwidth [Throughput %]", "ts": 200.0, "args": {"value": 15.0}},
        {"ph": "X", "pid": 90000, "tid": 0, "ts": 0.0, "dur": 200.0, "name": "dual_async_gemm", "cat": "hardware", "args": {"kind": "dispatch"}},
    ]
    if cta_slots:
        for wave in range(3):
            for sub in range(2):
                cta = wave * 2 + sub
                base = wave * 60.0 + sub * 2.0
                for k in range(10):
                    t = base + k * 4.5
                    evs.append({"ph": "X", "pid": 100000, "tid": 1, "ts": t, "dur": 3.2, "name": "load",
                                "args": {"sm": 0, "warp": 0, "cta": cta, "role": role_str, "semantic_id": "load", "async_engine": "mem"}})
                    evs.append({"ph": "X", "pid": 100000, "tid": 1, "ts": t + 0.2, "dur": 2.4, "name": "mma",
                                "args": {"sm": 0, "warp": 0, "cta": cta, "role": role_str, "semantic_id": "mma", "async_engine": "mma"}})
                    w_dur = 0.7 if cta == 2 else 0.1
                    evs.append({"ph": "X", "pid": 100000, "tid": 1, "ts": t + 0.1, "dur": w_dur, "name": "wait",
                                "args": {"sm": 0, "warp": 0, "cta": cta, "role": role_str, "semantic_id": "wait"}})
                    evs.append({"ph": "X", "pid": 100000, "tid": 1, "ts": t + 0.1 + w_dur, "dur": 0.8, "name": "compute",
                                "args": {"sm": 0, "warp": 0, "cta": cta, "role": role_str, "semantic_id": "compute"}})
    else:
        for cta in range(5):
            base = cta * 40.0
            for k in range(10):
                t = base + k * 3.6
                evs.append({"ph": "X", "pid": 100000, "tid": 1, "ts": t, "dur": 3.2, "name": "load",
                            "args": {"sm": 0, "warp": 0, "cta": cta, "role": "Producer (TMA issue)", "semantic_id": "load", "async_engine": "mem"}})
                evs.append({"ph": "X", "pid": 100000, "tid": 2, "ts": t + 0.2, "dur": 2.4, "name": "mma",
                            "args": {"sm": 0, "warp": 4, "cta": cta, "role": role_str, "semantic_id": "mma", "async_engine": "mma"}})
                w_dur = 0.7 if cta == 2 else 0.1
                evs.append({"ph": "X", "pid": 100000, "tid": 2, "ts": t + 0.1, "dur": w_dur, "name": "wait",
                            "args": {"sm": 0, "warp": 4, "cta": cta, "role": role_str, "semantic_id": "wait"}})
                evs.append({"ph": "X", "pid": 100000, "tid": 2, "ts": t + 0.1 + w_dur, "dur": 0.8, "name": "compute",
                            "args": {"sm": 0, "warp": 4, "cta": cta, "role": role_str, "semantic_id": "compute"}})
    return {
        "otherData": {"platform": "nv", "arch": "sm_90", "meta": {"card": "h20", "arch": "sm_90"}},
        "traceEvents": evs,
    }


class PanelCRoleLabelsVizTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="wp_viz_panelc_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _render_and_capture_ticks(self, trace: dict, name: str) -> tuple[Path, list[list[str]]]:
        calls: list[tuple[Axes, list[str]]] = []
        orig = Axes.set_yticklabels

        def _spy(ax: Axes, labels, *args, **kwargs):
            if ax in ax.figure.axes[2:]:
                calls.append((ax, [str(x) for x in labels]))
            return orig(ax, labels, *args, **kwargs)

        Axes.set_yticklabels = _spy
        try:
            out_png = render_swhw_overview_png(trace, self.tmp / name)
        finally:
            Axes.set_yticklabels = orig
        self.assertTrue(out_png.is_file())
        raw = out_png.read_bytes()
        self.assertTrue(raw.startswith(PNG_MAGIC))
        self.assertGreater(len(raw), 1024)
        by_ax: dict[int, list[str]] = {}
        ordered_axes: list[int] = []
        for ax, lbls in calls:
            aid = id(ax)
            if aid not in by_ax:
                ordered_axes.append(aid)
            by_ax[aid] = lbls
        return out_png, [by_ax[aid] for aid in ordered_axes]

    def test_1_within_threshold_does_not_widen_or_redraw(self) -> None:
        # Measured non-trigger maxima on synthetic 4-panel trace at default dpi:
        #   W04 · <role>    -> normal: 37 chars, all_W: 23 chars
        #   Slot 0 · <role> -> normal: 40 chars, all_W: 27 chars (x0 ~= 2.46 px in [2.0, 3.0))
        cases = (
            (False, "normal", 37, "W04 · "),
            (False, "all_W", 23, "W04 · "),
            (True, "normal", 40, "Slot 0 · "),
            (True, "all_W", 27, "Slot 0 · "),
        )
        for cta_slots, kind, tick_len, prefix in cases:
            with self.subTest(cta_slots=cta_slots, kind=kind, tick_len=tick_len):
                role = _role_for_tick_len(tick_len, len(prefix), kind)
                expected_tick = prefix + role
                self.assertEqual(len(expected_tick), tick_len)
                _, panel_ticks = self._render_and_capture_ticks(
                    _build_synth_4panel_trace(role, cta_slots=cta_slots),
                    f"ok_{cta_slots}_{kind}_{tick_len}.png",
                )
                self.assertEqual(render_swhw_overview_png.last_left_rescue_px, 0.0)
                self.assertEqual(render_swhw_overview_png.last_extra_draws, 0)
                self.assertFalse(render_swhw_overview_png.last_panel_c_title_truncated)
                self.assertEqual(len(panel_ticks), 2)
                self.assertEqual(panel_ticks[0], panel_ticks[1])
                self.assertIn(expected_tick, panel_ticks[0])
                self.assertFalse(any(s.endswith("\u2026") for s in panel_ticks[0]))

    def test_2_over_threshold_widens_left_margin_without_truncating(self) -> None:
        # Measured first-trigger lengths (+ 60-char normal cap check):
        #   W04 · <role>    -> normal: 38 & 60 chars, all_W: 24 chars
        #   Slot 0 · <role> -> normal: 41 chars, all_W: 28 chars
        cases = (
            (False, "normal", 38, "W04 · "),
            (False, "normal", 60, "W04 · "),
            (False, "all_W", 24, "W04 · "),
            (True, "normal", 41, "Slot 0 · "),
            (True, "all_W", 28, "Slot 0 · "),
        )
        for cta_slots, kind, tick_len, prefix in cases:
            with self.subTest(cta_slots=cta_slots, kind=kind, tick_len=tick_len):
                role = _role_for_tick_len(tick_len, len(prefix), kind)
                expected_tick = prefix + role
                _, panel_ticks = self._render_and_capture_ticks(
                    _build_synth_4panel_trace(role, cta_slots=cta_slots),
                    f"widen_{cta_slots}_{kind}_{tick_len}.png",
                )
                self.assertGreater(render_swhw_overview_png.last_left_rescue_px, 0.0)
                self.assertEqual(render_swhw_overview_png.last_extra_draws, 1)
                self.assertFalse(render_swhw_overview_png.last_panel_c_title_truncated)
                self.assertEqual(len(panel_ticks), 2)
                self.assertEqual(panel_ticks[0], panel_ticks[1])
                self.assertIn(expected_tick, panel_ticks[0])
                self.assertFalse(any(s.endswith("\u2026") for s in panel_ticks[0]))

    def test_3_120_char_labels_hit_fallback_truncation_and_keep_prefix(self) -> None:
        cases = (
            (False, "normal", 120, "W04 · ", True),
            (False, "all_W", 120, "W04 · ", True),
            (True, "normal", 120, "Slot 0 · ", False),
            (True, "all_W", 120, "Slot 0 · ", False),
        )
        for cta_slots, kind, tick_len, prefix, expect_title_cut in cases:
            with self.subTest(cta_slots=cta_slots, kind=kind, tick_len=tick_len):
                role = _role_for_tick_len(tick_len, len(prefix), kind)
                _, panel_ticks = self._render_and_capture_ticks(
                    _build_synth_4panel_trace(role, cta_slots=cta_slots),
                    f"trunc_{cta_slots}_{kind}_{tick_len}.png",
                )
                self.assertGreater(render_swhw_overview_png.last_left_rescue_px, 0.0)
                self.assertEqual(render_swhw_overview_png.last_extra_draws, 2)
                self.assertEqual(render_swhw_overview_png.last_panel_c_title_truncated, expect_title_cut)
                self.assertEqual(len(panel_ticks), 2)
                self.assertEqual(panel_ticks[0], panel_ticks[1])
                matching = [s for s in panel_ticks[0] if s.startswith(prefix)]
                self.assertTrue(matching, panel_ticks[0])
                for s in matching:
                    self.assertTrue(s.endswith("\u2026"), s)
                    self.assertLess(len(s), tick_len)
                    self.assertGreater(len(s), len(prefix) + 1)

    def test_4_extra_draws_never_exceed_hard_cap(self) -> None:
        self.assertEqual(viz.LEFT_RESCUE_MAX_EXTRA_DRAWS, 2)
        self.assertEqual(viz.LEFT_M_MAX, 0.35)
        for cta_slots, prefix in ((False, "W04 · "), (True, "Slot 0 · ")):
            for kind in ("normal", "all_W"):
                with self.subTest(cta_slots=cta_slots, kind=kind):
                    role = _role_for_tick_len(200, len(prefix), kind)
                    self._render_and_capture_ticks(
                        _build_synth_4panel_trace(role, cta_slots=cta_slots),
                        f"cap_{cta_slots}_{kind}_200.png",
                    )
                    self.assertGreaterEqual(render_swhw_overview_png.last_extra_draws, 1)
                    self.assertLessEqual(
                        render_swhw_overview_png.last_extra_draws,
                        viz.LEFT_RESCUE_MAX_EXTRA_DRAWS,
                    )
                    self.assertLessEqual(render_swhw_overview_png.last_extra_draws, 2)


if __name__ == "__main__":
    unittest.main()
