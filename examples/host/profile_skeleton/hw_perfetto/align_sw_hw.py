#!/usr/bin/env python3
"""align_sw_hw.py (E20) — put software events and hardware PM samples on one zero.

FULLY IMPLEMENTED generic logic. Produces a `sw-hw-alignment/v1` document that
emit_perfetto.py consumes. The alignment is SHIFT-ONLY (scale == 1): the device
%globaltimer that stamps the software events and the CUPTI kernel-launch clock
that stamps the PM samples advance at the same rate, so the only free parameter
is a translation. Any residual span mismatch (nsys_span > sw_span) is STRUCTURAL
(nsys measures the whole kernel; software spans first-punch..last-punch) and must
NOT be corrected by stretching — that would falsify the spacing between PM samples.

Common zero:
  * software zero (shift_ns)  = per-rank entry-rendezvous *release* (device side),
    falling back to the earliest software start when no anchor exists.
  * HW mapping window (nsys)  = the chosen target kernel launch [start,end] from
    the nsys sqlite (via extract_gpu_metrics.py -> hw_metrics.json).
The HW display mapping (applied later in emit_perfetto.merge_hw) is:
    ts_us = (sw_start + (nsys_ts - nsys_start) - shift_ns) / 1000

No project code hook: the launch-selection needle is data (contract capture),
and the entry anchor is data (ABI sync_anchors).

CLI:
  python3 -B align_sw_hw.py --events events_rank0.json [--hw-metrics hw/hw_metrics.json] \
      --out work/alignment.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RESIDUAL_MAX_NS_DEFAULT = 5000
# 5000 ns is a STARTING POINT, not a measured hardware limit, and main() now enforces it.
# The authoritative value is the frozen contract's alignment.residual_max_ns -- pass it with
# --residual-max-ns so the budget in the artifact is the budget you decided. Replace this
# number once you have seen the real cross-CTA rendezvous end skew for your grid on your GPU
# (it is reported as residual_measured_ns): a budget looser than the measured skew proves
# nothing, and one tighter than the hardware can hold will fail every run.


def load_json_maybe_gz(path: Path) -> Any:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def software_window(decoded: Dict[str, Any]) -> Tuple[int, int, int]:
    events = decoded.get("events") if isinstance(decoded, dict) else decoded
    if not isinstance(events, list):
        raise SystemExit("[align] events doc has no 'events' list")
    starts = [int(e["start_ns"]) for e in events if "start_ns" in e]
    ends = [int(e["end_ns"]) for e in events if "end_ns" in e]
    if not starts or not ends:
        raise SystemExit("[align] events doc has no start_ns/end_ns")
    return min(starts), max(ends), len(events)


def entry_release_ns(decoded: Dict[str, Any]) -> Optional[int]:
    anchors = decoded.get("anchors") or {}
    rendezvous = anchors.get("entry_rendezvous") or {}
    glob = rendezvous.get("global") or {}
    release = glob.get("entry_sync_end_ns")
    if isinstance(release, (int, float)) and not isinstance(release, bool):
        return int(release)
    return None


def cross_cta_residual(decoded: Dict[str, Any]) -> Dict[str, Any]:
    """Report how tightly the per-CTA entry rendezvous lines up (the real residual)."""
    anchors = decoded.get("anchors") or {}
    rendezvous = anchors.get("entry_rendezvous") or {}
    glob = rendezvous.get("global") or {}
    return {"start_skew_ns": glob.get("start_skew_ns"), "end_skew_ns": glob.get("end_skew_ns"),
            "rendezvous_span_ns": glob.get("rendezvous_span_ns"),
            "ctas_with_anchor": rendezvous.get("ctas_with_anchor"),
            "ctas_missing_anchor_total": rendezvous.get("ctas_missing_anchor_total")}


def count_samples_in_window(sample_ts_sorted: List[int], start: int, end: int) -> int:
    import bisect
    return bisect.bisect_right(sample_ts_sorted, end) - bisect.bisect_left(sample_ts_sorted, start)


def choose_launch(launches: List[Dict[str, Any]], sample_ts_sorted: List[int],
                  ordinal: Optional[int]) -> Tuple[int, Dict[str, Any]]:
    def _checked(index: int) -> Tuple[int, Dict[str, Any]]:
        launch = launches[index]
        # WHY this cannot be a warning: the chosen window IS the HW mapping window
        # (ts_us = sw_start + (nsys_ts - nsys_start) - shift_ns). Zero PM samples inside it
        # means merge_hw clips every sample away and the delivery ships software spans with
        # no counters underneath them, while alignment.json still looks perfectly valid.
        # The old code returned the shortest launch when every count was 0, and never counted
        # at all when there was exactly one launch.
        found = count_samples_in_window(sample_ts_sorted, launch["start_ns"], launch["end_ns"])
        if found <= 0:
            raise SystemExit(
                "[align] the chosen kernel launch (ordinal {} of {}, name={!r}, window "
                "[{}, {}]) contains 0 of the {} PM samples in this capture. Do not align "
                "against it. Check, in this order: (1) --capture-range=cudaProfilerApi "
                "actually brackets this launch (cudaProfilerStart/Stop around it); "
                "(2) contract capture.hardware.kernel_name_contains matches the kernel you "
                "profiled and not a warmup launch; (3) GPU PM sampling was enabled and "
                "produced rows (extract_gpu_metrics.py reported non-empty counters); "
                "(4) the launch and the samples come from the same GPU/device index."
                .format(index, len(launches), launch.get("name"),
                        launch["start_ns"], launch["end_ns"], len(sample_ts_sorted)))
        return index, launch

    if ordinal is not None:
        if not (0 <= ordinal < len(launches)):
            raise SystemExit("[align] --launch-ordinal {} out of range (have {})".format(ordinal, len(launches)))
        return _checked(ordinal)
    if len(launches) == 1:
        return _checked(0)
    counts = [count_samples_in_window(sample_ts_sorted, lc["start_ns"], lc["end_ns"]) for lc in launches]
    best = max(counts) if counts else 0
    cands = [i for i, c in enumerate(counts) if c == best]
    chosen = min(cands, key=lambda i: (launches[i]["end_ns"] - launches[i]["start_ns"], i))
    return _checked(chosen)


def compute_alignment(decoded: Dict[str, Any], hw_metrics: Optional[Dict[str, Any]] = None,
                      launch_ordinal: Optional[int] = None,
                      residual_max_ns: int = RESIDUAL_MAX_NS_DEFAULT) -> Dict[str, Any]:
    meta = decoded.get("metadata") or {}
    rank = meta.get("rank")
    sw_start, sw_end, sw_events = software_window(decoded)
    sw_span = sw_end - sw_start
    release = entry_release_ns(decoded)
    shift_ns = release if release is not None else sw_start

    result: Dict[str, Any] = {
        "schema": "sw-hw-alignment/v1",
        "scale": 1,
        "alignment_mode": "shift-only (scale=1; nsys_span > sw_span is structural, never stretched)",
        "rank": rank,
        "shift_ns": int(shift_ns),
        "shift_ns_source": "entry_rendezvous.release" if release is not None else "earliest_software_start",
        "shift_ns_by_rank": {str(rank if rank is not None else 0): int(shift_ns)},
        "sw_window_ns": [sw_start, sw_end],
        "sw_span_ns": sw_span,
        "sw_event_count": sw_events,
        "cross_cta_residual": cross_cta_residual(decoded),
        "residual_max_ns_budget": residual_max_ns,
        "nsys_window_ns": None,
    }

    if hw_metrics is not None:
        launches = hw_metrics.get("kernel_launches") or []
        if not launches:
            raise SystemExit("[align] hw_metrics has no kernel_launches; cannot align HW")
        sample_ts = sorted({int(ts) for counter in hw_metrics.get("counters", [])
                            for ts, _v in counter.get("samples", [])})
        idx, chosen = choose_launch(launches, sample_ts, launch_ordinal)
        nsys_start, nsys_end = int(chosen["start_ns"]), int(chosen["end_ns"])
        nsys_span = nsys_end - nsys_start
        result.update({
            "nsys_window_ns": [nsys_start, nsys_end],
            "nsys_span_ns": nsys_span,
            "device": hw_metrics.get("device"),
            "chosen_launch": {"ordinal": idx, "total_matched_launches": len(launches),
                              "correlation_id": chosen.get("correlation_id"),
                              "name": chosen.get("name"), "device": chosen.get("device")},
            "span_mismatch_ns": nsys_span - sw_span,
            "span_note": ("nsys_span - sw_span >= 0 is expected and structural; only the shift is "
                          "applied, spacing between PM samples is preserved"),
            "hw_map_formula": "ts_us = (sw_start + (nsys_ts - nsys_start) - shift_ns) / 1000",
        })
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compute shift-only SW/HW alignment (scale=1).")
    parser.add_argument("--events", type=Path, required=True, help="decode_profile.py events doc for the display rank")
    parser.add_argument("--hw-metrics", type=Path, default=None, help="extract_gpu_metrics.py hw_metrics.json")
    parser.add_argument("--launch-ordinal", type=int, default=None)
    parser.add_argument("--residual-max-ns", type=int, default=RESIDUAL_MAX_NS_DEFAULT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    decoded = load_json_maybe_gz(args.events)
    # WHY an explicitly passed --hw-metrics may not silently vanish: the old
    # `if args.hw_metrics and args.hw_metrics.is_file() else None` turned a wrong path or a
    # failed nsys capture into a SW-only alignment (nsys_window_ns=None) and still exited 0.
    # Omitting the flag is the legitimate sw_only path; passing a path that is not there is a
    # broken run that must not produce an alignment document.
    if args.hw_metrics is not None and not args.hw_metrics.is_file():
        raise SystemExit(
            "[align] --hw-metrics {} does not exist. Run extract_gpu_metrics.py first, or drop "
            "the flag if this is a software-only capture. Falling back to a SW-only alignment "
            "here would hand emit_perfetto an alignment with nsys_window_ns=null and ship a "
            "delivery with no HW counters at all.".format(args.hw_metrics))
    hw_metrics = load_json_maybe_gz(args.hw_metrics) if args.hw_metrics is not None else None
    alignment = compute_alignment(decoded, hw_metrics, args.launch_ordinal, args.residual_max_ns)

    # WHY this block exists: residual_max_ns_budget used to be written into alignment.json and
    # compared against nothing. A number sitting in an artifact reads as "checked" -- to the
    # reviewer, and to every later gate that quotes it. Either it is enforced here or the
    # document must say it was not.
    measured_skew = (alignment.get("cross_cta_residual") or {}).get("end_skew_ns")
    if isinstance(measured_skew, (int, float)) and not isinstance(measured_skew, bool):
        alignment["residual_max_ns_enforced"] = True
        alignment["residual_measured_ns"] = int(measured_skew)
        if int(measured_skew) > int(args.residual_max_ns):
            raise SystemExit(
                "[align] cross-CTA entry-rendezvous end skew is {} ns, over the "
                "residual_max_ns budget of {} ns. One shift cannot serve every CTA this far "
                "apart: the shared zero is off by up to that much for the worst CTA. Either "
                "the rendezvous is not actually synchronising all CTAs (check "
                "ctas_missing_anchor_total={}), or the budget in the contract "
                "(alignment.residual_max_ns) was set tighter than this hardware can hold and "
                "must be re-decided deliberately -- do not widen it just to make this pass."
                .format(int(measured_skew), int(args.residual_max_ns),
                        (alignment.get("cross_cta_residual") or {}).get("ctas_missing_anchor_total")))
        print("[align] residual check PASS: cross-CTA end skew {} ns <= budget {} ns".format(
            int(measured_skew), int(args.residual_max_ns)))
    else:
        alignment["residual_max_ns_enforced"] = False
        alignment["residual_not_enforced_reason"] = (
            "no entry-rendezvous end skew in the decoded anchors (shift_ns_source={}), so there "
            "is nothing to compare the budget against".format(alignment.get("shift_ns_source")))
        print("[align] WARNING residual budget {} ns was NOT enforced: {}. The shared zero is "
              "the earliest software start, whose per-CTA error is unmeasured -- add the entry "
              "rendezvous to the ABI (sync_anchors.entry_rendezvous) to get a real residual."
              .format(args.residual_max_ns, alignment["residual_not_enforced_reason"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(alignment, indent=2) + "\n", encoding="utf-8")
    print("[align] rank={} shift_ns={} ({}) sw_span_ns={} nsys_window={}".format(
        alignment.get("rank"), alignment["shift_ns"], alignment["shift_ns_source"],
        alignment["sw_span_ns"], alignment.get("nsys_window_ns")))
    if alignment.get("nsys_window_ns"):
        print("[align] chosen launch ordinal={}/{} span_mismatch_ns={}".format(
            alignment["chosen_launch"]["ordinal"], alignment["chosen_launch"]["total_matched_launches"],
            alignment["span_mismatch_ns"]))
    print("[align] wrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
