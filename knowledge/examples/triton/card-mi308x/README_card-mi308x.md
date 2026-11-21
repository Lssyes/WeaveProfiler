# MI308X Triton GEMM reference

This directory contains a directly runnable, self-written FP16 GEMM that records
every physical wave64 separately. The canonical output semantics are `load`,
`mma`, `store`, and the single `gemm` envelope. On MI308X, `tl.dot` lowers to
`v_mfma`.

`FI_LEVEL` and `FI_SAMPLE` are independent settings:

- `FI_LEVEL` is recorder verbosity only: `0=off`, `1=entry-only`, and
  `2=full-events`.
- `FI_SAMPLE` is K-block event density: `0=all K blocks` (full audit) and
  `1=one representative K block` (fast sampled feedback). It is meaningful with
  `FI_LEVEL=2`.
- Fast sampled feedback always requires `FI_LEVEL=2 FI_SAMPLE=1`.
- Full audit always requires `FI_LEVEL=2 FI_SAMPLE=0`.
- `FI_LEVEL=1` is entry-only. It records OPEN/CLOSE and optional timer anchors,
  not K-block events, and must never be labeled or used as full-audit evidence.

Run the sampled path for development feedback:

```bash
env ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  WEAVEPROFILER_ROOT=/path/to/weaveprofiler \
  python3 -B gemm_triton_instrumented_card-mi308x.py \
  --m 128 --n 128 --k 2048 --fi-level 2 --fi-sample 1 --fi-timer 1 \
  --output-dir fi/fast_sample_card-mi308x
```

It records one representative K block per physical wave plus the complete
`gemm` envelope. The MI308X reference measured 14 raw records per wave,
`0.975x` steady-state ratio, and passed the 10% gate. This is development
evidence with `delivery_evidence=false`; it does not certify full-audit event
coverage.

Run the full audit explicitly:

```bash
env ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  WEAVEPROFILER_ROOT=/path/to/weaveprofiler \
  python3 -B gemm_triton_instrumented_card-mi308x.py \
  --m 128 --n 128 --k 2048 --fi-level 2 --fi-sample 0 --fi-timer 1 \
  --output-dir fi/full_audit_card-mi308x
```

This records every K block on every physical wave. The historical 390-record
full-K reference measured a `1.703x` steady-state ratio; it predates the added
GEMM envelope pair. The corrected full-audit layout has a 392-record contract at
64 K blocks with two timer anchors. That `1.703x` full-audit ratio failed the 10%
gate, so the result is an explicit performance failure. Correctness and overflow
checks passing cannot turn that failed gate into a pass.

An entry-only diagnostic uses `--fi-level 1`; its metadata records
`fi_sample: "not applicable"`. Entry-only is a recorder-level baseline and is
not a substitute for either sampled or full-audit evidence.

Correctness and performance are separate contracts. Sampling changes event
density only: every correctness run still compares every element of every
output against an independent reference. The 10% gate uses steady-state kernel
time only, never event-record count, trace size, or collector duration. Keep
the fast `0.975x` result and the full-audit `1.703x` failure as separate
evidence, and report the SASS/ISA delta for each capture.

Every output filename includes the literal `card-mi308x` and the selected
configuration token: `fi-l2-s1` for fast sampled feedback or `fi-l2-s0` for
full audit. Decode the raw FILITE02 dump with
`4_validate/tools/fi_lite_decode.py` and the generated names sidecar. Do not
rename semantics to source labels such as `A load` or `MMA`; trace display names
must remain the canonical IDs.

The H20 and B300 routes keep their existing `scan -> instrument -> capture ->
validate` workflow. MI308X HIP C++ and Triton share the same capture, export,
and validation entry points; there is no HIP-only workflow.
