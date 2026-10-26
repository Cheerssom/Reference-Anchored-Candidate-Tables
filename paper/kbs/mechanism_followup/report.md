# Frozen-model mechanism diagnostics

Post-hoc diagnostic analysis; ten class partitions on fixed images, order0 primary. No new encoding or image download. Partition0 is a prespecified illustration, not the best-performing partition. The main plot zooms into the retention region; full trajectories of both orders are in operating_frontiers_all_full.pdf. Connecting lines guide the eye and are not interpolation used in matching. Hollow markers show reference fallback, which can lie outside the replaced readout's trajectory because it uses the reference classifier. All frontier quantities use test labels descriptively; none are deployed thresholds.

| Encoder/data | Selected gap on matched-feasible set pp | At baseline actual base, retained minus baseline selected pp | Both envelopes at that target pp | Feasible | Delta AUROC | Margin AUROC gain |
|---|---:|---:|---:|---:|---:|---:|
| l14/CIFAR-100 | +12.98 [+2.05,+28.83] | +13.06 [+2.46,+29.23] | +12.02 [+2.07,+27.35] | 6/10 | 0.353 | +0.1288 |
| l14/Pets | +8.81 [-1.73,+28.54] | +0.46 [-3.77,+4.66] | -9.52 [-26.21,+0.28] | 9/10 | 0.374 | +0.0246 |
| l14/ImageNet-R | +4.85 [+2.77,+6.93] | +3.23 [+1.48,+5.02] | +3.10 [+1.37,+4.88] | 10/10 | 0.409 | +0.0536 |
| b32/CIFAR-100 | +12.79 [+1.24,+25.79] | +10.17 [-4.68,+25.02] | +8.04 [-4.79,+20.88] | 6/10 | 0.424 | +0.1270 |
| b32/Pets | +35.45 (n=2; no interval) | +15.24 (n=2; no interval) | -23.02 (n=2; no interval) | 2/10 | 0.421 | +0.1307 |
| b32/ImageNet-R | +18.70 [+11.56,+27.01] | +19.24 [+12.51,+26.94] | +17.53 [+11.78,+23.67] | 7/10 | 0.451 | +0.0481 |

## Decomposition at the original common95% minimum requirement

| Encoder/data | Policy gap pp | Envelope gap pp | Retained oracle headroom pp | Baseline oracle headroom pp | Paired feasible | Both selected meet target |
|---|---:|---:|---:|---:|---:|---:|
| l14/CIFAR-100 | +35.40 [+16.61,+54.61] | +3.68 [+1.08,+6.38] | +5.56 [+4.14,+7.17] | +37.28 [+20.02,+55.34] | 10/10 | 10/10 |
| l14/Pets | +15.88 [-1.27,+40.52] | -2.12 [-4.07,-0.49] | +4.40 [+2.56,+6.69] | +22.40 [+3.61,+49.38] | 10/10 | 10/10 |
| l14/ImageNet-R | +4.85 [+2.77,+6.93] | +0.29 [-0.55,+1.16] | +2.32 [+1.69,+2.94] | +6.88 [+4.87,+8.98] | 10/10 | 10/10 |
| b32/CIFAR-100 | +26.38 [+11.60,+41.41] | +3.50 [-0.47,+7.94] | +6.12 [+4.08,+8.35] | +29.00 [+15.40,+42.81] | 9/10 | 9/9 |
| b32/Pets | +42.92 [+15.13,+66.79] | +1.17 [-5.91,+7.38] | +11.31 [+6.24,+16.89] | +53.06 [+34.03,+70.75] | 8/10 | 8/8 |
| b32/ImageNet-R | +26.15 [+15.85,+37.54] | +10.40 [+7.31,+14.64] | +3.16 [+2.50,+3.72] | +18.91 [+7.89,+30.86] | 9/10 | 8/9 |

This is a descriptive identity: policy gap = common-target envelope gap + baseline headroom - retained headroom. Headroom is oracle frontier minus the selected novel accuracy; it is nonnegative only if that selected policy meets this test target. It includes finite candidate grids, sampling, conservatism and other policy-selection effects; it is not uniquely attributable to statistical certification. The common95% minimum requirement and the baseline-realized-accuracy target in the first table are different comparisons and must not be substituted for one another.

## Original retention grid, exploratory same-model recomputation

| Encoder/data | Retention | Envelope gap pp | Jointly feasible |
|---|---:|---:|---:|
| l14/CIFAR-100 | 80% | -0.09 [-0.97,+0.66] | 10/10 |
| l14/CIFAR-100 | 85% | -0.03 [-1.00,+0.82] | 10/10 |
| l14/CIFAR-100 | 90% | +0.41 [-0.88,+1.59] | 10/10 |
| l14/CIFAR-100 | 95% | +3.68 [+1.08,+6.38] | 10/10 |
| l14/CIFAR-100 | 99% | +8.14 (n=2; no interval) | 2/10 |
| l14/Pets | 80% | -0.46 [-1.32,+0.36] | 10/10 |
| l14/Pets | 85% | -0.64 [-1.66,+0.29] | 10/10 |
| l14/Pets | 90% | -1.14 [-2.36,-0.13] | 10/10 |
| l14/Pets | 95% | -2.12 [-4.07,-0.49] | 10/10 |
| l14/Pets | 99% | -3.64 [-8.46,+0.03] | 8/10 |
| l14/ImageNet-R | 80% | -0.16 [-0.59,+0.33] | 10/10 |
| l14/ImageNet-R | 85% | -0.37 [-0.80,+0.15] | 10/10 |
| l14/ImageNet-R | 90% | -0.36 [-0.82,+0.12] | 10/10 |
| l14/ImageNet-R | 95% | +0.29 [-0.55,+1.16] | 10/10 |
| l14/ImageNet-R | 99% | +6.06 [+1.50,+12.35] | 5/10 |
| b32/CIFAR-100 | 80% | +0.03 [-1.02,+1.01] | 10/10 |
| b32/CIFAR-100 | 85% | +0.26 [-1.13,+1.58] | 10/10 |
| b32/CIFAR-100 | 90% | +0.59 [-1.41,+2.48] | 10/10 |
| b32/CIFAR-100 | 95% | +3.50 [-0.47,+7.94] | 9/10 |
| b32/CIFAR-100 | 99% | -0.92 (n=4; no interval) | 4/10 |
| b32/Pets | 80% | -0.32 [-3.07,+2.50] | 10/10 |
| b32/Pets | 85% | -0.18 [-3.50,+3.14] | 10/10 |
| b32/Pets | 90% | +0.48 [-3.78,+4.59] | 10/10 |
| b32/Pets | 95% | +1.17 [-5.91,+7.38] | 8/10 |
| b32/Pets | 99% | -15.12 (n=3; no interval) | 3/10 |
| b32/ImageNet-R | 80% | +0.14 [-0.31,+0.56] | 10/10 |
| b32/ImageNet-R | 85% | +0.50 [+0.23,+0.79] | 10/10 |
| b32/ImageNet-R | 90% | +1.27 [+0.80,+1.75] | 10/10 |
| b32/ImageNet-R | 95% | +10.40 [+7.31,+14.64] | 9/10 |
| b32/ImageNet-R | 99% | -- | 0/10 |

These five ratios are from the original published grid, but this recomputation is a post-hoc descriptive follow-up. Feasible sets may differ across ratios. These are test-label oracle capacities, not newly certified99%-retention deployment results.

Intervals resample partition means and condition on fixed images; no interval is displayed for fewer than five partitions. The three gain columns use the same matched-feasible set within a row. All-ten fallback-inclusive selected gaps are separately recorded in summary.json. Matched feasible sets differ between conditions; see counts. In particular, B/32 Pets has only two matched-feasible partitions and supplies a selected-set description, not evidence of a population-wide effect. Conditional admission counts remain in summary.json. Matching is conservative (at least baseline base accuracy), without interpolation. The envelope difference concerns complete fixed readouts, including fusion/normalization, not an isolated classifier-weight causal effect.

A_ref is identical for all methods in a partition. D is split into losses to novel labels and losses to other base labels. Retained G=0 and D_to_base=0 were checked everywhere. A signed envelope-minus-selected adjustment is not necessarily calibration regret: a selected point may not satisfy the target chosen from the other method.

Delta separation is exploratory and not required by the algebra. AUROC uses fixed novel-positive orientation and class-macro weights. Compare the combined margin AUROC, not Delta alone. Removing centering is recalibrated on the same independent images; fusion weights and query scales are fixed, so it is not a fully retuned alternative architecture.

Verification: 120 partition/order records; 590 new threshold tests independently reconstructed; all recorded source hashes checked; max algebraic float error 6.23e-05. Original calibration archives preserve cached batch rounding differences; max reconstructed difference 0.000139, within their recorded5e-4 witness tolerance.

These diagnostics are included in Supplement P with a corrected main-text attribution. They have no new completed fresh-agent integrity audit: the reviewer stopped at a model usage limit before the final semantic report. Partial independent deterministic receipts are retained privately. The preceding manuscript audit does not automatically cover this new analysis.
