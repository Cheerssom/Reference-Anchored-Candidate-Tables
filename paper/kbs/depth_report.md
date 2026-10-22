# KBS depth extension — completed local evidence

2026-10-22. This report separates measured evidence from a journal acceptance prediction.

## New evidence

- Three datasets, 9 dense Tip-Adapter partition runs (sum and class-mean, each native and independently calibrated), matched to the existing encoder, fitting pools and support.
- 81 shot/calibration configurations: three datasets × three class partitions × three shot budgets × three correlated trials, each with two orders and five arrivals. Aggregate trials/orders inside partitions; three partition means supply the SD.
- Exact empirical retention-envelope derivation; descriptive matched-test sweeps remain distinct from the validation-only policy.
- 648 append checks per normalization variant: no restricted candidate-ranking changes for fixed references; dynamic references change rankings on each dataset.
- Measured optimized readout latency/storage on the same RTX 4070 Laptop; image encoding and common backbone memory excluded.

At the ImageNet-R 95% budget, retained novel accuracy is 84.48±0.72%, versus 81.58±1.72% for class-mean Tip-Adapter. Average incremental gain over the dense cache is 1.48±0.35 percentage points; final gain is 1.21±0.46 points. Across repeated five-shot calibration trials, the paired average gain over recalibrated removal is 2.14±0.37 points on ImageNet-R, 1.22±1.18 on CIFAR and 0.78±1.03 on Pets. Accuracy rankings vary by dataset, support budget and operating constraint; all comparisons are retained, including cases favoring another readout. No universal superiority or significance claim is made.

## Validation and persistence

Three numerical depth tests pass. Five earlier evidence tests also pass. Cache-free reconstruction verifies all 48,510 new metric records and their feasible thresholds from test and validation sufficient statistics; the original trial-zero/five-shot four-table results have maximum difference 0.0. Live pause/resume preserves a completed session's hash and modification time. All 90 units complete. Existing feature extraction remains resumable at batch shards; new readouts need no encoder pass.

The resumed invocation reports 59.6 s for CIFAR, 36.9 s for Pets and 80.3 s for ImageNet-R; the CIFAR number excludes work completed before the intentional pause. These are invocation durations, not full reproducible end-to-end wall times. The largest recorded new-run peak PyTorch allocation is 227,623,936 bytes, excluding desktop/driver and cached memory. No paid compute was used.

## Manuscript and review status

Current PDFs: 23-page main, 22-page supplement, 186-word abstract. Source and cross-reference checks pass; current submission sources and bundle are rebuilt. The fresh GPT-6-Astra claim-audit invocation failed before producing output because of a model usage limit; its local ERROR artifact is preserved. Local deterministic checks do not replace a fresh scientific review. This report is not an independent referee panel or a guarantee of Q1/Q2 acceptance. No journal submission has occurred.

Protocols: experiments/kbs_depth_protocol.md. Reproduction: experiments/kbs_strengthening/README.md. Raw outcomes, selection trials and source snapshots: experiments/kbs_depth/.
