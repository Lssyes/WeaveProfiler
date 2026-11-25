#!/usr/bin/env python3
"""Wall-clock-free characterization tests for KREX lock hand-off polling.

Pins ``weaveprof.harness.gpu_critical_region`` (default ``local_file`` OS flock
provider) without calling ``time.sleep`` or reading real wall-clock elapsed time:

  1. Contended 2-worker hand-off (default ``poll_interval_s=0.02``): an
     uncontended holder acquires in ``0`` polls; a contending waiter polls at
     ``0.02 s`` steps while the lock is held and acquires on the very next probe
     once the holder exits (``0`` post-release polls, ``<= MAX_POST_RELEASE_POLLS``).
  2. Tick-boundary and staggered-arrival hand-offs: whether the holder releases
     on an exact ``poll_interval_s`` multiple, ``1 us`` after a poll probe, or
     before the waiter arrives, post-release polls never exceed ``1`` and
     hand-off lag never exceeds one poll interval (``20.0 ms``).
  3. 4-worker concurrent off-region + serialized in-region schedule (deterministic
     counterpart of ``test_krex_region_lock_concurrency_and_zero_overlap``) plus
     independent multi-GPU zero-poll concurrency.
  4. ``timeout_s`` minimum clamp (``max(0.05, float(timeout_s))``), exact ``>=``
     deadline comparison, owner metadata inclusion in ``TimeoutError``, and
     explicit ``poll_interval_s`` override.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Callable
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import weaveprof.harness as harness  # noqa: E402
from weaveprof.harness import RegionLease, gpu_critical_region  # noqa: E402

# Upper bound on additional sleep polls a waiting worker may initiate AFTER the
# previous holder has exited gpu_critical_region. Because unlock happens in the
# holder's finally block before the next poll probe runs, a waiter waking from an
# in-flight poll acquires immediately (0 post-release polls); even if a waiter
# sampled _try_lock_file at the exact instant before unlock and entered sleep
# right after unlock, at most 1 post-release sleep can occur before acquisition.
MAX_POST_RELEASE_POLLS = 1


@dataclass(frozen=True)
class PollRecord:
    worker_id: int
    call_time_s: float
    interval_s: float
    wake_time_s: float
    release_seq_at_call: int


class _DeterministicVirtualClock:
    """Single-stepping discrete-event virtual clock over real OS worker threads.

    All participating workers enter an initial barrier at ``t = 0.0`` and run
    strictly one at a time in deterministic ``(wake_time_s, priority, worker_id)``
    order. Real OS ``fcntl.flock`` / ``msvcrt.locking`` calls execute on distinct
    open file descriptors per thread, while ``harness.time`` advances purely via
    virtual events with zero real sleeping and zero OS scheduling races.
    """

    _BASE_EPOCH = 1_700_000_000.0
    _MAX_SPIN_QUERIES = 5_000
    _MAX_ZERO_SLEEPS = 2_000

    def __init__(self, num_workers: int) -> None:
        self._num_workers = int(num_workers)
        self._cond = threading.Condition()
        self.now_s: float = 0.0
        self.release_seq: int = 0
        self.release_times_s: list[tuple[int, float]] = []
        self.polls: list[PollRecord] = []
        self._thread_to_worker: dict[int, int] = {}
        self._active: set[int] = set(range(self._num_workers))
        self._parked: dict[int, tuple[float, int]] = {}
        self._started_count: int = 0
        self._current_runner: int | None = None
        self._aborted: BaseException | None = None
        self._spin_queries_at_tick: int = 0
        self._zero_sleeps_at_tick: int = 0

    def _schedule_next_locked(self) -> None:
        if self._aborted is not None or not self._active:
            self._current_runner = None
            self._cond.notify_all()
            return
        if len(self._parked) < len(self._active):
            self._cond.notify_all()
            return
        next_wid = min(
            self._parked,
            key=lambda w: (self._parked[w][0], self._parked[w][1], w),
        )
        wake_at, _prio = self._parked.pop(next_wid)
        if wake_at > self.now_s:
            self.now_s = round(wake_at, 9)
            self._spin_queries_at_tick = 0
            self._zero_sleeps_at_tick = 0
        self._current_runner = next_wid
        self._cond.notify_all()

    def _wait_turn_locked(self, wid: int) -> None:
        ok = self._cond.wait_for(
            lambda: self._aborted is not None or self._current_runner == wid,
            timeout=10.0,
        )
        if not ok:
            self._aborted = RuntimeError(f"VirtualClock deadlock waiting for worker {wid}")
            self._cond.notify_all()
        if self._aborted is not None:
            raise self._aborted

    def enter_worker(self, wid: int, *, initial_delay_s: float = 0.0, priority: int = 1) -> None:
        ident = threading.get_ident()
        with self._cond:
            self._thread_to_worker[ident] = wid
            self._parked[wid] = (round(float(initial_delay_s), 9), int(priority))
            self._started_count += 1
            if self._started_count == self._num_workers and self._current_runner is None:
                self._schedule_next_locked()
            self._wait_turn_locked(wid)

    def finish_worker(self, wid: int) -> None:
        with self._cond:
            self._active.discard(wid)
            self._parked.pop(wid, None)
            if self._current_runner == wid:
                self._current_runner = None
            self._schedule_next_locked()

    def abort(self, exc: BaseException) -> None:
        with self._cond:
            if self._aborted is None:
                self._aborted = exc
            self._cond.notify_all()

    def mark_released(self, wid: int) -> None:
        with self._cond:
            self.release_seq += 1
            self.release_times_s.append((wid, round(self.now_s, 9)))

    def advance_work(self, duration_s: float, *, priority: int = 0) -> None:
        """Advance simulated off-region or in-region GPU work for the current worker."""
        ident = threading.get_ident()
        with self._cond:
            wid = self._thread_to_worker[ident]
            dt = max(0.0, float(duration_s))
            wake_at = round(self.now_s + dt, 9)
            self._parked[wid] = (wake_at, int(priority))
            self._current_runner = None
            self._schedule_next_locked()
            self._wait_turn_locked(wid)

    def perf_counter(self) -> float:
        with self._cond:
            self._spin_queries_at_tick += 1
            if self._spin_queries_at_tick > self._MAX_SPIN_QUERIES:
                err = RuntimeError(
                    "Busy-spin detected: > 5000 perf_counter() calls without virtual sleep"
                )
                self._aborted = err
                self._cond.notify_all()
                raise err
            return round(self.now_s, 9)

    def time(self) -> float:
        with self._cond:
            return round(self._BASE_EPOCH + self.now_s, 6)

    def sleep(self, seconds: float) -> None:
        """Replacement for ``weaveprof.harness.time.sleep`` recording lock poll steps."""
        ident = threading.get_ident()
        with self._cond:
            wid = self._thread_to_worker[ident]
            dt = float(seconds)
            if dt <= 0.0:
                self._zero_sleeps_at_tick += 1
                if self._zero_sleeps_at_tick > self._MAX_ZERO_SLEEPS:
                    err = RuntimeError(
                        f"Non-advancing poll sleep({dt}) exceeded {self._MAX_ZERO_SLEEPS} iterations"
                    )
                    self._aborted = err
                    self._cond.notify_all()
                    raise err
                wake_at = self.now_s
            else:
                wake_at = round(self.now_s + dt, 9)
            self.polls.append(
                PollRecord(
                    worker_id=wid,
                    call_time_s=round(self.now_s, 9),
                    interval_s=round(dt, 9),
                    wake_time_s=wake_at,
                    release_seq_at_call=self.release_seq,
                )
            )
            # Priority 1 ensures that if a lock holder finishes holding at the exact
            # same virtual timestamp (priority 0), the holder's unlock completes first.
            self._parked[wid] = (wake_at, 1)
            self._current_runner = None
            self._schedule_next_locked()
            self._wait_turn_locked(wid)

    def polls_for(self, wid: int) -> list[PollRecord]:
        return [p for p in self.polls if p.worker_id == wid]


class _VirtualTimeProxy:
    """Drop-in proxy for ``weaveprof.harness.time`` backed by ``_DeterministicVirtualClock``."""

    def __init__(self, clock: _DeterministicVirtualClock, real_time_mod) -> None:
        self._clock = clock
        self._real_time = real_time_mod

    def perf_counter(self) -> float:
        return self._clock.perf_counter()

    def time(self) -> float:
        return self._clock.time()

    def sleep(self, seconds: float) -> None:
        self._clock.sleep(seconds)

    def __getattr__(self, name: str):
        return getattr(self._real_time, name)


def _run_virtual_workers(
    worker_specs: list[tuple[float, int, Callable[[_DeterministicVirtualClock, int], object]]],
) -> tuple[_DeterministicVirtualClock, list[object]]:
    """Run ``worker_specs`` of ``(initial_delay_s, initial_priority, fn)`` under virtual time."""
    clock = _DeterministicVirtualClock(len(worker_specs))
    results: list[object] = [None] * len(worker_specs)
    errors: list[BaseException] = []
    err_lock = threading.Lock()

    def _entry(wid: int, init_delay: float, init_prio: int, fn) -> None:
        try:
            clock.enter_worker(wid, initial_delay_s=init_delay, priority=init_prio)
            results[wid] = fn(clock, wid)
        except BaseException as exc:
            with err_lock:
                errors.append(exc)
            clock.abort(exc)
        finally:
            clock.finish_worker(wid)

    real_time = harness.time
    proxy = _VirtualTimeProxy(clock, real_time)
    with mock.patch.object(harness, "time", proxy):
        threads = [
            threading.Thread(
                target=_entry,
                args=(idx, delay_s, prio, fn),
                name=f"krex-virtual-worker-{idx}",
                daemon=True,
            )
            for idx, (delay_s, prio, fn) in enumerate(worker_specs)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)
            if t.is_alive():
                raise RuntimeError(f"Worker thread {t.name} did not terminate")

    if errors:
        raise errors[0]
    return clock, results


class KrexHandoffPollsCharacterizationTest(unittest.TestCase):
    _ENV_KEYS = (
        "WEAVEPROF_LOCK_DIR",
        "WEAVEPROF_CACHE",
        "WEAVEPROF_LOCK_PROVIDER",
        "XDG_CACHE_HOME",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls._saved_env = {k: os.environ.get(k) for k in cls._ENV_KEYS}
        cls._root = Path(tempfile.mkdtemp(prefix="wp-p1c-r28-"))
        os.environ["WEAVEPROF_LOCK_DIR"] = str(cls._root / "default_locks")
        os.environ["WEAVEPROF_CACHE"] = str(cls._root / "cache")
        os.environ["XDG_CACHE_HOME"] = str(cls._root / "xdg_cache")
        os.environ.pop("WEAVEPROF_LOCK_PROVIDER", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)

    @classmethod
    def tearDownClass(cls) -> None:
        for k, v in cls._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(cls._root, ignore_errors=True)

    def _assert_under_temp_root(self, target_path: str | Path) -> None:
        resolved = Path(target_path).resolve()
        temp_root = self._root.resolve()
        self.assertTrue(
            resolved.is_relative_to(temp_root),
            f"Lock artifact escaped temp root: {resolved} vs {temp_root}",
        )

    def test_single_handoff_acquires_within_one_poll_after_holder_release(self) -> None:
        lock_dir = self._root / "single_handoff_locks"
        owner_snapshots: list[dict] = []

        def holder_fn(clock: _DeterministicVirtualClock, wid: int) -> RegionLease:
            with gpu_critical_region(
                gpu_id="h20_0",
                timeout_s=15.0,
                lock_dir=lock_dir,
                label="holder_w0",
            ) as lease:
                owner_file = lock_dir / "h20_0.owner.json"
                owner_snapshots.append(json.loads(owner_file.read_text(encoding="utf-8")))
                clock.advance_work(0.05, priority=0)
            clock.mark_released(wid)
            return lease

        def waiter_fn(clock: _DeterministicVirtualClock, wid: int) -> RegionLease:
            with gpu_critical_region(
                gpu_id="h20_0",
                timeout_s=15.0,
                lock_dir=lock_dir,
                label="waiter_w1",
            ) as lease:
                owner_file = lock_dir / "h20_0.owner.json"
                owner_snapshots.append(json.loads(owner_file.read_text(encoding="utf-8")))
                clock.advance_work(0.05, priority=0)
            clock.mark_released(wid)
            return lease

        clock, raw_leases = _run_virtual_workers(
            [
                (0.0, 0, holder_fn),
                (0.0, 1, waiter_fn),
            ]
        )
        lease_h, lease_w = raw_leases  # type: ignore[misc]
        self._assert_under_temp_root(lease_h.lock_path)
        self._assert_under_temp_root(lease_w.lock_path)

        # Holder acquires immediately with 0 polls; owner file is written then cleaned on exit.
        self.assertEqual(clock.polls_for(0), [])
        self.assertEqual(lease_h.gpu_id, "h20_0")
        self.assertEqual(lease_h.label, "holder_w0")
        self.assertEqual(lease_h.provider, "local_file")
        self.assertEqual(lease_h.wait_ms, 0.0)
        self.assertEqual(lease_h.hold_ms, 50.0)
        self.assertEqual([snap["label"] for snap in owner_snapshots], ["holder_w0", "waiter_w1"])
        self.assertFalse((lock_dir / "h20_0.owner.json").exists())
        self.assertTrue((lock_dir / "h20_0.lock").is_file())
        self.assertEqual(clock.release_times_s, [(0, 0.05), (1, 0.11)])

        # Waiter polls at default poll_interval_s=0.02 while held (t=0.00, 0.02, 0.04),
        # initiates 0 additional sleeps after holder releases at t=0.05, and acquires on
        # the 1st poll wakeup at/after t=0.05 (t=0.06 <= 1 post-release wakeup).
        holder_rel_s = clock.release_times_s[0][1]
        w_polls = clock.polls_for(1)
        pre_release_polls = [p for p in w_polls if p.release_seq_at_call == 0]
        post_release_polls = [p for p in w_polls if p.release_seq_at_call >= 1]
        post_release_wakeups = [p for p in w_polls if p.wake_time_s >= holder_rel_s]

        self.assertEqual(len(pre_release_polls), 3)
        self.assertEqual([p.call_time_s for p in pre_release_polls], [0.0, 0.02, 0.04])
        self.assertEqual([p.interval_s for p in pre_release_polls], [0.02, 0.02, 0.02])
        self.assertLessEqual(len(post_release_polls), MAX_POST_RELEASE_POLLS)
        self.assertEqual(len(post_release_polls), 0)
        self.assertLessEqual(len(post_release_wakeups), MAX_POST_RELEASE_POLLS)
        self.assertEqual(len(post_release_wakeups), 1)

        self.assertEqual(lease_w.label, "waiter_w1")
        self.assertEqual(lease_w.provider, "local_file")
        self.assertEqual(lease_w.wait_ms, 60.0)
        self.assertEqual(lease_w.hold_ms, 50.0)
        handoff_lag_ms = round(lease_w.wait_ms - lease_h.hold_ms, 3)
        self.assertEqual(handoff_lag_ms, 10.0)
        self.assertLessEqual(handoff_lag_ms, 20.0)
        self.assertEqual(clock.now_s, 0.11)

    def test_tick_boundary_and_post_release_arrival_handoff_bounds(self) -> None:
        # Subcase A: Holder releases at exact poll_interval_s multiple (t = 0.06s = 3 * 0.02s).
        lock_dir_a = self._root / "boundary_exact_locks"

        def holder_exact(clock: _DeterministicVirtualClock, wid: int) -> RegionLease:
            with gpu_critical_region(gpu_id="h20_0", lock_dir=lock_dir_a, label="h_exact") as lease:
                clock.advance_work(0.06, priority=0)
            clock.mark_released(wid)
            return lease

        def waiter_exact(clock: _DeterministicVirtualClock, wid: int) -> RegionLease:
            with gpu_critical_region(gpu_id="h20_0", lock_dir=lock_dir_a, label="w_exact") as lease:
                clock.advance_work(0.02, priority=0)
            clock.mark_released(wid)
            return lease

        clock_a, (lh_a, lw_a) = _run_virtual_workers([(0.0, 0, holder_exact), (0.0, 1, waiter_exact)])
        self._assert_under_temp_root(lh_a.lock_path)
        w_polls_a = clock_a.polls_for(1)
        post_a = [p for p in w_polls_a if p.release_seq_at_call >= 1]
        wake_a = [p for p in w_polls_a if p.wake_time_s >= clock_a.release_times_s[0][1]]
        self.assertEqual(len(w_polls_a), 3)
        self.assertLessEqual(len(post_a), MAX_POST_RELEASE_POLLS)
        self.assertEqual(len(post_a), 0)
        self.assertLessEqual(len(wake_a), MAX_POST_RELEASE_POLLS)
        self.assertEqual(len(wake_a), 1)
        self.assertEqual(lh_a.hold_ms, 60.0)
        self.assertEqual(lw_a.wait_ms, 60.0)

        # Subcase B: Waiter probes at t = 0.06s just 1 us BEFORE holder unlocks at t = 0.060001s;
        # waiter enters at most one extra sleep (0.06 -> 0.08s) and acquires immediately at 0.08s.
        lock_dir_b = self._root / "boundary_just_after_locks"

        def holder_late(clock: _DeterministicVirtualClock, wid: int) -> RegionLease:
            with gpu_critical_region(gpu_id="h20_0", lock_dir=lock_dir_b, label="h_late") as lease:
                clock.advance_work(0.060001, priority=0)
            clock.mark_released(wid)
            return lease

        def waiter_late(clock: _DeterministicVirtualClock, wid: int) -> RegionLease:
            with gpu_critical_region(gpu_id="h20_0", lock_dir=lock_dir_b, label="w_late") as lease:
                clock.advance_work(0.02, priority=0)
            clock.mark_released(wid)
            return lease

        clock_b, (lh_b, lw_b) = _run_virtual_workers([(0.0, 0, holder_late), (0.0, 1, waiter_late)])
        self._assert_under_temp_root(lh_b.lock_path)
        w_polls_b = clock_b.polls_for(1)
        post_b = [p for p in w_polls_b if p.release_seq_at_call >= 1]
        wake_b = [p for p in w_polls_b if p.wake_time_s >= clock_b.release_times_s[0][1]]
        self.assertEqual(len(w_polls_b), 4)
        self.assertEqual([p.call_time_s for p in w_polls_b], [0.0, 0.02, 0.04, 0.06])
        self.assertLessEqual(len(post_b), MAX_POST_RELEASE_POLLS)
        self.assertEqual(len(post_b), 0)
        self.assertLessEqual(len(wake_b), MAX_POST_RELEASE_POLLS)
        self.assertEqual(len(wake_b), 1)
        self.assertEqual(lh_b.hold_ms, 60.001)
        self.assertEqual(lw_b.wait_ms, 80.0)
        self.assertLessEqual(round(lw_b.wait_ms - lh_b.hold_ms, 3), 20.0)

        # Subcase C: Second caller arrives at t = 0.05s right after first holder released -> 0 polls.
        lock_dir_c = self._root / "staggered_arrival_locks"

        def first_holder(clock: _DeterministicVirtualClock, wid: int) -> RegionLease:
            with gpu_critical_region(gpu_id="h20_0", lock_dir=lock_dir_c, label="h_first") as lease:
                clock.advance_work(0.05, priority=0)
            clock.mark_released(wid)
            return lease

        def post_release_arriver(clock: _DeterministicVirtualClock, wid: int) -> RegionLease:
            with gpu_critical_region(gpu_id="h20_0", lock_dir=lock_dir_c, label="w_after") as lease:
                clock.advance_work(0.03, priority=0)
            clock.mark_released(wid)
            return lease

        clock_c, (_lh_c, lw_c) = _run_virtual_workers(
            [(0.0, 0, first_holder), (0.05, 1, post_release_arriver)]
        )
        self._assert_under_temp_root(lw_c.lock_path)
        self.assertEqual(clock_c.polls_for(1), [])
        self.assertEqual(lw_c.wait_ms, 0.0)
        self.assertEqual(lw_c.hold_ms, 30.0)

    def test_four_worker_off_region_parallelism_and_chained_handoff_polls(self) -> None:
        """Wall-clock-free counterpart of test_krex_region_lock_concurrency_and_zero_overlap."""
        lock_dir = self._root / "four_workers_locks"

        def make_worker(idx: int):
            def _worker(clock: _DeterministicVirtualClock, wid: int) -> tuple[int, float, float, RegionLease]:
                # Off-region work (0.12s) runs concurrently across all 4 workers (t = 0.00 -> 0.12s).
                clock.advance_work(0.12, priority=1)
                with gpu_critical_region(
                    gpu_id="h20_0",
                    timeout_s=15.0,
                    lock_dir=lock_dir,
                    label=f"w{idx}",
                ) as lease:
                    t_start = clock.now_s
                    clock.advance_work(0.05, priority=0)
                    t_end = clock.now_s
                clock.mark_released(wid)
                return idx, t_start, t_end, lease

            return _worker

        clock, raw_out = _run_virtual_workers([(0.0, 1, make_worker(i)) for i in range(4)])
        records = list(raw_out)  # type: ignore[arg-type]
        for _i, _s, _e, lease in records:
            self._assert_under_temp_root(lease.lock_path)

        intervals = [(s, e) for (_i, s, e, _lease) in records]
        self.assertEqual(
            intervals,
            [
                (0.12, 0.17),
                (0.18, 0.23),
                (0.24, 0.29),
                (0.30, 0.35),
            ],
        )
        for (_, prev_end), (next_start, _) in zip(intervals, intervals[1:]):
            self.assertGreaterEqual(next_start, prev_end)

        # Total virtual span is 0.35s (vs 4 * 0.17s = 0.68s if off-region were serialized).
        self.assertEqual(clock.now_s, 0.35)
        self.assertEqual([r[3].wait_ms for r in records], [0.0, 60.0, 120.0, 180.0])
        self.assertEqual([r[3].hold_ms for r in records], [50.0, 50.0, 50.0, 50.0])
        self.assertEqual([len(clock.polls_for(i)) for i in range(4)], [0, 3, 6, 9])

        # For every chained hand-off k-1 -> k, worker k enters 0 sleeps after worker k-1
        # releases and acquires on the 1st poll wakeup at/after worker k-1's release time.
        for k in range(1, 4):
            prev_rel_s = clock.release_times_s[k - 1][1]
            polls_after_prev_release = [
                p for p in clock.polls_for(k) if p.release_seq_at_call >= k
            ]
            wakeups_after_prev_release = [
                p for p in clock.polls_for(k) if p.wake_time_s >= prev_rel_s
            ]
            self.assertLessEqual(len(polls_after_prev_release), MAX_POST_RELEASE_POLLS)
            self.assertEqual(len(polls_after_prev_release), 0)
            self.assertLessEqual(len(wakeups_after_prev_release), MAX_POST_RELEASE_POLLS)
            self.assertEqual(len(wakeups_after_prev_release), 1)

        # Distinct GPU IDs never block each other (0 polls on both workers).
        multi_gpu_dir = self._root / "multi_gpu_locks"

        def gpu_worker(gpu_name: str):
            def _fn(clock: _DeterministicVirtualClock, wid: int) -> RegionLease:
                with gpu_critical_region(gpu_id=gpu_name, lock_dir=multi_gpu_dir, label=gpu_name) as lease:
                    clock.advance_work(0.05, priority=0)
                clock.mark_released(wid)
                return lease

            return _fn

        clock_mg, (l0, l1) = _run_virtual_workers(
            [(0.0, 0, gpu_worker("h20_0")), (0.0, 0, gpu_worker("h20_1"))]
        )
        self._assert_under_temp_root(l0.lock_path)
        self._assert_under_temp_root(l1.lock_path)
        self.assertEqual(clock_mg.polls, [])
        self.assertEqual((l0.wait_ms, l1.wait_ms), (0.0, 0.0))
        self.assertEqual(clock_mg.now_s, 0.05)

    def test_timeout_deadline_clamp_and_custom_poll_interval_bound(self) -> None:
        # 1. timeout_s=0.01 is clamped to 0.05s via max(0.05, float(timeout_s));
        #    with poll_interval_s=0.025, waiter polls at t=0.00 and t=0.025 (2 polls),
        #    reaches t=0.050 >= 0.050, and raises TimeoutError with owner[:200] excerpt.
        timeout_dir = self._root / "timeout_locks"

        def long_holder(clock: _DeterministicVirtualClock, wid: int) -> RegionLease:
            with gpu_critical_region(
                gpu_id="h20_0",
                timeout_s=15.0,
                lock_dir=timeout_dir,
                label="holder_timeout_probe",
            ) as lease:
                clock.advance_work(0.20, priority=0)
            clock.mark_released(wid)
            return lease

        def short_waiter(_clock: _DeterministicVirtualClock, _wid: int) -> tuple[str, dict]:
            with self.assertRaises(TimeoutError) as err_ctx:
                with gpu_critical_region(
                    gpu_id="h20_0",
                    timeout_s=0.01,
                    lock_dir=timeout_dir,
                    label="waiter_short",
                    poll_interval_s=0.025,
                ):
                    pass
            live_owner = json.loads((timeout_dir / "h20_0.owner.json").read_text(encoding="utf-8"))
            return str(err_ctx.exception), live_owner

        clock_to, (lh_to, (timeout_msg, live_owner)) = _run_virtual_workers(
            [(0.0, 0, long_holder), (0.0, 1, short_waiter)]
        )
        self._assert_under_temp_root(lh_to.lock_path)
        self.assertEqual(lh_to.hold_ms, 200.0)
        w_to_polls = clock_to.polls_for(1)
        self.assertEqual(len(w_to_polls), 2)
        self.assertEqual([p.call_time_s for p in w_to_polls], [0.0, 0.025])
        self.assertEqual([p.interval_s for p in w_to_polls], [0.025, 0.025])
        prefix, sep, owner_excerpt = timeout_msg.partition("; current owner=")
        self.assertEqual(sep, "; current owner=")
        self.assertTrue(
            prefix.startswith("Timed out after 0.0s waiting for KREX GPU critical region lock on h20_0 (")
        )
        self.assertTrue(prefix.endswith("h20_0.lock)"))
        self.assertEqual(len(owner_excerpt), 200)
        self.assertEqual(owner_excerpt.splitlines()[:2], ["{", '  "gpu_id": "h20_0",'])
        self.assertEqual(live_owner["gpu_id"], "h20_0")
        self.assertEqual(live_owner["label"], "holder_timeout_probe")
        self.assertEqual(live_owner["wait_ms"], 0.0)
        self.assertEqual(live_owner["released_at"], 0.0)
        self.assertEqual(live_owner["provider"], "local_file")
        self.assertFalse((timeout_dir / "h20_0.owner.json").exists())

        # 2. Custom poll_interval_s=0.025 with holder holding 0.04s -> 2 polls (0.00, 0.025),
        #    acquires at t=0.050 (wait_ms=50.0, 0 post-release polls), using default lock_dir env.
        def custom_holder(clock: _DeterministicVirtualClock, wid: int) -> RegionLease:
            with gpu_critical_region(label="env_default_holder", poll_interval_s=0.025) as lease:
                clock.advance_work(0.04, priority=0)
            clock.mark_released(wid)
            return lease

        def custom_waiter(clock: _DeterministicVirtualClock, wid: int) -> RegionLease:
            with gpu_critical_region(label="env_default_waiter", poll_interval_s=0.025) as lease:
                clock.advance_work(0.01, priority=0)
            clock.mark_released(wid)
            return lease

        clock_cu, (lh_cu, lw_cu) = _run_virtual_workers(
            [(0.0, 0, custom_holder), (0.0, 1, custom_waiter)]
        )
        self._assert_under_temp_root(lh_cu.lock_path)
        self.assertEqual(lh_cu.gpu_id, "gpu_0")
        self.assertEqual(Path(lh_cu.lock_path).parent, (self._root / "default_locks").resolve())
        w_cu_polls = clock_cu.polls_for(1)
        self.assertEqual(len(w_cu_polls), 2)
        self.assertEqual([p.interval_s for p in w_cu_polls], [0.025, 0.025])
        self.assertEqual([p for p in w_cu_polls if p.release_seq_at_call >= 1], [])
        self.assertEqual(
            len([p for p in w_cu_polls if p.wake_time_s >= clock_cu.release_times_s[0][1]]),
            1,
        )
        self.assertEqual(lw_cu.wait_ms, 50.0)

        # 3. XDG_CACHE_HOME fallback (when WEAVEPROF_LOCK_DIR is unset) also stays in temp root.
        saved_lock_dir_env = os.environ.pop("WEAVEPROF_LOCK_DIR", None)
        try:
            def xdg_worker(clock: _DeterministicVirtualClock, wid: int) -> RegionLease:
                with gpu_critical_region(label="xdg_fallback_probe") as lease:
                    clock.advance_work(0.01, priority=0)
                clock.mark_released(wid)
                return lease

            _clock_xdg, (lh_xdg,) = _run_virtual_workers([(0.0, 0, xdg_worker)])
            self._assert_under_temp_root(lh_xdg.lock_path)
            self.assertEqual(
                Path(lh_xdg.lock_path).parent,
                (self._root / "xdg_cache" / "weaveprof" / "krex_locks").resolve(),
            )
            self.assertFalse((Path(lh_xdg.lock_path).parent / "gpu_0.owner.json").exists())
        finally:
            if saved_lock_dir_env is not None:
                os.environ["WEAVEPROF_LOCK_DIR"] = saved_lock_dir_env


if __name__ == "__main__":
    unittest.main()
