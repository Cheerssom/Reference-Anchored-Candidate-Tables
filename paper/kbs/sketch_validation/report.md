# ImageNet-Sketch natural-shift validation

New target-domain benchmark evaluation; frozen source models/support/tuning; order0 five arrivals; exact deduplication; no guaranteed pretraining novelty

{"acquired": 10152, "unique": 6445, "excluded": 3707, "calibration_reserved": 2492, "test": 3953, "minimum_unique_class": 22, "maximum_unique_class": 50, "matched_calibration_total": 2492}

| Encoder | Readout | Rule | Calibration | Finite | Met | Base | Novel | Conditional novel |
|---|---|---|---|---:|---:|---:|---:|---:|
| l14 | retained | protection | source_cal | 10/10 | 10/10 | 92.23 | 82.83 | 82.83 |
| l14 | retained | protection | target_cal | 10/10 | 10/10 | 90.81 | 86.73 | 86.73 |
| l14 | anchored_tp | protection | source_cal | 6/10 | 10/10 | 93.16 | 38.21 | 63.69 |
| l14 | anchored_tp | protection | target_cal | 10/10 | 10/10 | 91.96 | 78.68 | 78.68 |
| l14 | hycal | protection | source_cal | 0/10 | 10/10 | 93.77 | 0.00 | -- |
| l14 | hycal | protection | target_cal | 1/10 | 10/10 | 93.41 | 4.50 | 45.00 |
| l14 | retained | paired_accuracy | source_cal | 10/10 | 10/10 | 92.21 | 82.99 | 82.99 |
| l14 | retained | paired_accuracy | target_cal | 10/10 | 10/10 | 90.76 | 86.82 | 86.82 |
| l14 | anchored_tp | paired_accuracy | source_cal | 10/10 | 10/10 | 92.24 | 75.43 | 75.43 |
| l14 | anchored_tp | paired_accuracy | target_cal | 10/10 | 10/10 | 91.56 | 83.34 | 83.34 |
| l14 | hycal | paired_accuracy | source_cal | 0/10 | 10/10 | 93.77 | 0.00 | -- |
| l14 | hycal | paired_accuracy | target_cal | 2/10 | 9/10 | 93.04 | 10.47 | 52.34 |
| b32 | retained | protection | source_cal | 10/10 | 10/10 | 80.10 | 52.34 | 52.34 |
| b32 | retained | protection | target_cal | 10/10 | 10/10 | 79.10 | 58.23 | 58.23 |
| b32 | anchored_tp | protection | source_cal | 0/10 | 10/10 | 81.32 | 0.00 | -- |
| b32 | anchored_tp | protection | target_cal | 0/10 | 10/10 | 81.32 | 0.00 | -- |
| b32 | hycal | protection | source_cal | 0/10 | 10/10 | 81.32 | 0.00 | -- |
| b32 | hycal | protection | target_cal | 0/10 | 10/10 | 81.32 | 0.00 | -- |
| b32 | retained | paired_accuracy | source_cal | 10/10 | 10/10 | 79.71 | 54.87 | 54.87 |
| b32 | retained | paired_accuracy | target_cal | 10/10 | 10/10 | 78.87 | 58.96 | 58.96 |
| b32 | anchored_tp | paired_accuracy | source_cal | 4/10 | 10/10 | 80.39 | 11.19 | 27.99 |
| b32 | anchored_tp | paired_accuracy | target_cal | 3/10 | 10/10 | 80.82 | 8.35 | 27.84 |
| b32 | hycal | paired_accuracy | source_cal | 0/10 | 10/10 | 81.32 | 0.00 | -- |
| b32 | hycal | paired_accuracy | target_cal | 0/10 | 10/10 | 81.32 | 0.00 | -- |

Both calibration domains use equal counts within every base class. Source fit, support and score coefficients are unchanged. Source-calibrated outcomes diagnose transport under shift; the source-law guarantee does not cover the changed target law. Sketch-calibrated guarantees retain the class-conditional independence assumptions. Exact/dihedral hashes do not establish absence of all near-duplicate dependence. Intervals condition on the fixed target images and class assignments, and confidence statements apply separately to each fixed method/partition/calibration/policy over five prespecified arrivals.


## Common-target capacity

| Encoder | Retained-minus-dense capacity pp | Jointly feasible |
|---|---:|---:|
| l14 | +0.001 [-0.64,+0.69] | 10/10 |
| b32 | +5.776 [+4.03,+7.55] | 9/10 |

These use the same jointly feasible partitions. L/14 selected-policy gains do not imply a common-floor recognition-capacity advantage. Full curves are in operating_frontiers_all.pdf and frontiers.npz; partition0 is an index-based illustration, not a best-case selection.
