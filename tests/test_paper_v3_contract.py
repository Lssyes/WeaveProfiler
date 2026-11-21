"""Binary fixtures and source placements from Figures 4(b) and 6 (CPU tests)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weaveprof.contract import compact_probe_spec, compile_role, loop_spans, normalize_probe_spec, revise_probe_spec
from weaveprof.backend import assess


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


engine = load("2_instrument/rule_engine.py", "paper3_engine")
decoder = load("4_validate/tools/fi_lite_decode.py", "paper3_decoder")
verifier = load("4_validate/closed_loop_verifier.py", "paper3_verifier")
placer = load("1_scan/llm_probe_placer.py", "paper3_placer")


def paper_source():
    lines = [""] * 73
    for number, code in {
        1: "__global__ void pipeline(float* out) {",
        2: "    int s = 0;",
        41: "    if (threadIdx.x / warpSize == 4) {",
        42: "        tma_load_async(s);",
        43: "    }",
        54: "    if (threadIdx.x / warpSize < 4) {",
        55: "        full_bar[s].wait();",
        58: "        for (int k=0; k<8192; ++k) {",
        60: "            // misleading } while (--k) {",
        61: "            float x = out[k];",
        62: "            x *= x;",
        63: "            out[k] = x;",
        70: "        }",
        71: "    }",
        73: "}",
    }.items():
        lines[number - 1] = code
    return "\n".join(lines) + "\n"


HANDOFF = {"roles": {"dma": "warp 4", "math": "warps 0-3"},
           "probes": [{"id": "load", "begin": {"role": "dma", "line": 42},
                       "end": {"role": "math", "line": 55}, "slot": "s"},
                      {"id": "mma", "begin": {"role": "math", "line": 61},
                       "end": {"role": "math", "line": 63}}]}


def binary_decode(lanes, names, wpc=5):
    meta = json.dumps({"gran": 0, "arch": "sm_90", "backend": "nvidia_ptx"}).encode()
    counts = [len(records) for records in lanes]
    blob = bytearray(b"FILITE02")
    blob.extend(struct.pack("<6I", len(lanes), max(counts), wpc, 0, 0, len(meta)))
    blob.extend(meta)
    blob.extend(struct.pack(f"<{len(lanes)}I", *counts))
    for records in lanes:
        for record in records:
            blob.extend(struct.pack("<QII", *record))
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as work:
        path = Path(work) / "events.bin"
        path.write_bytes(blob)
        return decoder.decode(decoder.read_dump(path), names)


class PaperContractTests(unittest.TestCase):
    def test_spill_delta_with_nonzero_baseline(self):
        src = paper_source()
        # Eight distinct baseline opcodes, one replacement in the instrumented build.
        ops = ["MOV", "LDG", "IADD", "IMAD", "MUL", "STG", "FADD", "EXIT"]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as work:
            directory = Path(work)
            for name, sequence in (("l0", ops), ("inst", [*ops[:-1], "NOP"])):
                sass = "Function : test_spill\n" + "".join(
                    f"/*{16 * i:04x}*/ {op} ;\n" for i, op in enumerate(sequence))
                (directory / (name + ".sass")).write_text(sass, encoding="utf-8")
            for inst, status, failed in ((64, "PASS", []), (80, "FAIL", ["assembly"]), (48, "PASS", [])):
                with self.subTest(instrumented_spill_bytes=inst):
                    runtime = {"kernel_name": "test_spill", "baseline_sass": "l0.sass",
                               "instrumented_sass": "inst.sass", "baseline_spill_bytes": 64,
                               "instrumented_spill_bytes": inst, "extra_smem_bytes": 0}
                    checks, metrics, errors, *_ = assess(directory, HANDOFF, 0.1, runtime)
                    self.assertNotIn("assembly", errors)
                    self.assertEqual(checks["assembly"], not failed)
                    self.assertEqual(metrics["assembly_change_rate"], 0.125)
                    self.assertEqual(metrics["delta_spill_bytes"], inst - 64)
                    resources = {"l0": {"spill_total_bytes": 64}, "inst": {"spill_total_bytes": inst},
                                 "delta_smem_bytes": 0}
                    evaluation = verifier.evaluate_four_gates(
                        source_text=src, probe_spec=HANDOFF,
                        preflight={"ok": True, "binsim": {"sim_opcode": 0.875}, "resources": resources},
                        runtime_metrics={"ok": True, "mismatches_total": 0, "nonfinite_total": 0, "overhead_pct": 0},
                        completeness={"ok": True, "fires": {"load": 1, "mma": 1}})
                    self.assertEqual(evaluation["metrics"]["assembly_change_rate"], 0.125)
                    self.assertEqual(evaluation["metrics"]["delta_spill_bytes"], inst - 64)
                    feedback = verifier.build_feedback_json(evaluation, HANDOFF, src)
                    self.assertEqual((feedback["status"], feedback["failed"]), (status, failed))

    def test_role_compilation_and_lossless_v1_metadata(self):
        self.assertIn("warpSize", compile_role("warp 4", "hip_cpp"))
        self.assertEqual(compile_role("warp_id == 0"), "warp_id == 0")
        old = {"schema": "weaveprofiler.probe-spec/v1", "custom": [9],
               "warp_roles": [{"name": "lane", "role_id": 7, "predicate": "warp_id == 0", "owner": "user"}],
               "probes": [{"probe_id": "custom_stage", "semantic_id": "load", "event_id": 291,
                           "begin_line": 3, "end_line": 8, "role_id": 7, "aux_expr": "slot[i]",
                           "notes": "retained"}]}
        normal = normalize_probe_spec(old)
        restored = normalize_probe_spec(json.loads(json.dumps(compact_probe_spec(normal))))
        self.assertEqual(normal, restored)
        self.assertEqual(restored["probes"][0]["slot"], "slot[i]")
        self.assertEqual(restored["warp_roles"][0]["owner"], "user")
        for bad in ({"role": "unknown", "line": 1}, {"role": "dma", "line": 74},
                    {"role": "dma", "line": True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize_probe_spec({**HANDOFF, "probes": [{**HANDOFF["probes"][0], "begin": bad}]}, paper_source())

    def test_paper_frontend_reads_two_levels_and_binds_endpoints(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as work:
            path = Path(work) / "pipeline.cu"
            original = paper_source().replace("\n", "\r\n").encode()
            path.write_bytes(original)
            def caller(model, messages, **kwargs):
                self.assertEqual(model, "qwen4-max")
                for fragment in ("Level 1", "Level 2", "CTA warp-role", "full_bar[s].wait()"):
                    self.assertIn(fragment, messages[1]["content"])
                return {"elapsed_s": 0, "content": json.dumps(HANDOFF), "usage": {}}
            spec = placer.identify_source(path, card="h20", llm_caller=caller)
            self.assertEqual(spec["probes"], HANDOFF["probes"])
            self.assertEqual(spec["source_sha256"], placer.sha256_file(path))
            self.assertEqual(path.read_bytes(), original)

    def test_figure_4b_injection_and_multicast_fifo(self):
        result = engine.inject_probes(paper_source(), HANDOFF)
        self.assertTrue(result["ok"], result["diagnostics"])
        code, names = result["instrumented_source"], result["names"]
        self.assertIn("wp_g", code)
        self.assertIn("wp_lite.cuh", code)
        self.assertLess(code.index("WP_B("), code.index("tma_load_async(s)"))
        self.assertLess(code.index("full_bar[s].wait()"), code.index("WP_E("))
        b_role = next(int(r) for r,n in names["roles"].items() if n == "dma")
        e_role = next(int(r) for r,n in names["roles"].items() if n == "math")
        self.assertEqual(names["cross_warp"]["1"], {"id": "load", "begin_role": b_role,
                                                   "end_role": e_role, "slot": "s"})
        lanes = []
        for w in range(4):
            lanes.append([(900, decoder.OPEN | e_role, 0),
                          (1250 + 10*w, decoder.END | 1, 7), (1300, 3, 0),
                          (1400, decoder.END | 3, 0), (2500 + 10*w, decoder.END | 1, 7),
                          (3000, decoder.CLOSE, 0)])
        lanes.append([(900, decoder.OPEN | b_role, 0), (1000, 1, 7),
                      (2000, 1, 7), (3000, decoder.CLOSE, 0)])
        trace, summary = binary_decode(lanes, names)
        self.assertEqual(summary["errors"], {})
        self.assertEqual(summary["fires"], {"mma": 1, "load": 2})
        events = [e for e in trace["traceEvents"] if e["ph"] == "X" and e["name"] == "load"]
        self.assertEqual(len(events), 8)
        for w in range(4):
            intervals = [e for e in events if e["args"]["warp"] == w]
            self.assertEqual(sorted(round(e["dur"]*1000) for e in intervals), [250 + 10*w, 500 + 10*w])
            self.assertTrue(all((e["args"]["handoff_from"], e["args"]["slot"]) == ("dma", 7)
                                for e in intervals))
        for records in lanes[:4]:
            records.pop(-2)
        _, missing = binary_decode(lanes, names)
        self.assertEqual(missing["errors"], {"unpaired_begin": 1})

    def test_figure_6_measured_counter_reduction_and_hoist(self):
        src = paper_source()
        doc = {"roles": {"math": "warps 0-3"}, "probes": [HANDOFF["probes"][1]]}
        injected = engine.inject_probes(src, doc)
        self.assertTrue(injected["ok"], injected["diagnostics"])
        names = injected["names"]
        def feedback(placement, counts, similarity, slowdown):
            lanes = []
            for count in counts:
                records = [(0, decoder.OPEN, 0)]
                for k in range(count):
                    records.extend(((2*k+1, 3, 0), (2*k+2, decoder.END | 3, 0)))
                lanes.append(records + [(2*count+4, decoder.CLOSE, 0)])
            _, summary = binary_decode(lanes, names, wpc=len(counts))
            preflight = {"ok": True, "binsim": {"sim_opcode": similarity},
                         "resources": {"l0": {"spill_total_bytes": 0}, "inst": {"spill_total_bytes": 0},
                                       "delta_smem_bytes": 0}}
            runtime = {"ok": True, "mismatches_total": 0, "nonfinite_total": 0, "overhead_pct": slowdown}
            evaluation = verifier.evaluate_four_gates(source_text=src, probe_spec=placement,
                            preflight=preflight, runtime_metrics=runtime,
                            completeness={"ok": not summary["errors"], "fires": summary["fires"]})
            return verifier.build_feedback_json(evaluation, placement, src)
        first = feedback(doc, (8192, 4096), 0.64, 38.4)
        self.assertEqual(first, {"status": "FAIL", "failed": ["assembly", "overhead"],
                                 "fires": {"mma": 8192}, "enclosing_loop": [58, 70]})
        revised = revise_probe_spec(doc, first, src)
        self.assertEqual([revised["probes"][0][s]["line"] for s in ("begin", "end")], [58, 70])
        code = engine.inject_probes(src, revised)["instrumented_source"]
        self.assertLess(code.index("WP_B("), code.index("for (int k=0"))
        self.assertGreater(code.index("WP_E("), code.index("}", code.index("out[k] = x;")))
        second = feedback(revised, (1, 1), 0.98, 1.8)
        self.assertEqual(second, {"status": "PASS", "failed": [], "fires": {"mma": 1}, "enclosing_loop": None})
        self.assertEqual(loop_spans("for k in range(8):\n    while k:\n        step()\n", "triton"),
                         [(1, 3), (2, 3)])

    def test_dynamic_multislot_async_load_and_mma(self):
        """Arbitrary dynamic slots (e.g. 5-slot cross-warp load + 2-slot same-warp async wgmma)
        pair cleanly in FIFO order without false LIFO cross-nesting, while catching slot overwrite."""
        spec = {
            "roles": {"dma": "warp 0", "math": "warp 1"},
            "probes": [
                {"id": "load", "begin": {"role": "dma", "line": 42}, "end": {"role": "math", "line": 55}, "slot": "k % 5"},
                {"id": "mma", "begin": {"role": "math", "line": 61}, "end": {"role": "math", "line": 63}, "slot": "k % 2"},
            ],
        }
        names = engine.inject_probes(paper_source(), spec)["names"]
        self.assertIn("3", names["slotted_events"])
        b_role = next(int(r) for r, n in names["roles"].items() if n == "dma")
        e_role = next(int(r) for r, n in names["roles"].items() if n == "math")
        # DMA issues 5 concurrent dynamic load slots (slots 0..4); Math consumes slots 0..4 AND runs 2 overlapping async MMA slots (0, 1)
        dma_lane = [(900, decoder.OPEN | b_role, 0),
                    *( (1000 + 40 * s, 1, s) for s in range(5) ),
                    (3000, decoder.CLOSE, 0)]
        math_lane = [(900, decoder.OPEN | e_role, 0),
                     (1300, decoder.END | 1, 0), (1310, 3, 0),           # MMA slot 0 begins (t=1310)
                     (1400, decoder.END | 1, 1), (1410, 3, 1),           # MMA slot 1 begins (t=1410) BEFORE slot 0 ends!
                     (1550, decoder.END | 3, 0),                         # MMA slot 0 ends (t=1550, FIFO!)
                     (1600, decoder.END | 1, 2), (1610, 3, 0),           # MMA slot 0 re-issued cleanly after release
                     (1700, decoder.END | 3, 1),                         # MMA slot 1 ends (t=1700)
                     (1800, decoder.END | 1, 3),
                     (1900, decoder.END | 1, 4),
                     (1950, decoder.END | 3, 0),                         # MMA slot 0 2nd gen ends
                     (3000, decoder.CLOSE, 0)]
        trace, summary = binary_decode([dma_lane, math_lane], names, wpc=2)
        self.assertEqual(summary["errors"], {})
        self.assertEqual(summary["fires"], {"load": 5, "mma": 3})
        mma_evs = [e for e in trace["traceEvents"] if e["ph"] == "X" and e["name"] == "mma"]
        self.assertEqual(len(mma_evs), 3)
        # Overlapping MMA slots 0 and 1 must be split onto distinct sublanes (0 and 1)
        self.assertGreaterEqual(len({e["args"]["instance"] for e in mma_evs}), 2)

        # Now verify Trace Verifier catches intra-slot overwrite (re-issuing MMA slot 0 BEFORE prior MMA slot 0 ended!)
        bad_math = [(900, decoder.OPEN | e_role, 0),
                    (1310, 3, 0), (1350, 3, 0), (1550, decoder.END | 3, 0), (1650, decoder.END | 3, 0),
                    (3000, decoder.CLOSE, 0)]
        _, bad_sum = binary_decode([[(900, decoder.OPEN | b_role, 0), (3000, decoder.CLOSE, 0)], bad_math], names, wpc=2)
        self.assertEqual(bad_sum["errors"].get("slot_reentry_overwrite"), 1)

    def test_triton_autotune_helper_diag_and_cutedsl_two_tier_launcher(self):
        # 1. Triton @autotune stripped on _fi_inst_ + do_not_specialize injected + proxy forwards best_config
        tri_src = (
            "import triton, triton.language as tl\n"
            "@triton.autotune(configs=[triton.Config({'BLOCK_M': 128}, num_warps=4)], key=['M'])\n"
            "@triton.jit\n"
            "def mm_k(A, B, M, BLOCK_M: tl.constexpr):\n"
            "    x = tl.load(A)\n"
            "    tl.store(B, x)\n"
        )
        tri_spec = {"language": "triton", "card": "h20", "roles": {"w": "all"},
                    "probes": [{"id": "load", "begin": {"role": "w", "line": 5}, "end": {"role": "w", "line": 5}}]}
        tri_res = engine.inject_probes(tri_src, tri_spec)
        self.assertTrue(tri_res["ok"], tri_res["diagnostics"])
        inst_py = tri_res["instrumented_source"]
        l0_pos = inst_py.index("def _fi_l0_mm_k")
        inst_pos = inst_py.index("def _fi_inst_mm_k")
        self.assertIn("@triton.autotune", inst_py[:l0_pos])
        self.assertNotIn("@triton.autotune", inst_py[l0_pos:inst_pos])
        self.assertIn("do_not_specialize=['_fi_rec_ptr', '_fi_ovf_ptr']", inst_py[l0_pos:inst_pos])

        # 2. Inner @triton.jit helper callee static diagnostic points to root call site
        helper_src = (
            "import triton, triton.language as tl\n"
            "@triton.jit\n"
            "def inner_helper(ptr):\n"
            "    return tl.load(ptr)\n"
            "@triton.jit\n"
            "def entry_k(A, B):\n"
            "    v = inner_helper(A)\n"
            "    tl.store(B, v)\n"
        )
        helper_spec = {"language": "triton", "card": "h20", "roles": {"w": "all"},
                       "probes": [{"id": "load", "begin": {"role": "w", "line": 4}, "end": {"role": "w", "line": 4}}]}
        h_res = engine.inject_probes(helper_src, helper_spec)
        self.assertFalse(h_res["ok"])
        self.assertEqual(h_res["diagnostics"][0]["code"], "INNER_JIT_HELPER_PROBE")
        self.assertEqual(h_res["diagnostics"][0]["enclosing_call_lines"], [7, 7])

        # 3. Two-tier CuTeDSL @cute.jit host launcher -> @cute.kernel device kernel threading
        cute_src = (
            "import cutlass, cutlass.cute as cute\n"
            "@cute.kernel\n"
            "def dev_gemm(A, B):\n"
            "    x = A[0]\n"
            "    B[0] = x\n"
            "@cute.jit\n"
            "def host_launch(A, B, stream):\n"
            "    dev_gemm(A, B).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)\n"
        )
        cute_spec = {"language": "cutedsl", "card": "b300", "roles": {"w": "all"},
                     "probes": [{"id": "load", "begin": {"role": "w", "line": 4}, "end": {"role": "w", "line": 5}}]}
        c_res = engine.inject_probes(cute_src, cute_spec)
        self.assertTrue(c_res["ok"], c_res["diagnostics"])
        c_py = c_res["instrumented_source"]
        self.assertIn("def _fi_inst_host_launch(A, B, stream, _fi_rec=None, _fi_ovf=None):", c_py)
        self.assertIn("_fi_inst_dev_gemm(A, B, _fi_rec=_fi_rec, _fi_ovf=_fi_ovf).launch(", c_py)
        self.assertIn("host_launch = _WpCuteProxy(_fi_l0_host_launch, _fi_inst_host_launch, fi_get_attached)", c_py)


if __name__ == "__main__":
    unittest.main()
