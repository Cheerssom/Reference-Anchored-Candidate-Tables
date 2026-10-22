# Independent calibration coverage and label budgets

Fixed existing readouts, tuning parameters, support and test archives; only independent calibration coverage changes. Caps 2/5/10/20/40 use nested class-wise subsets. Three subset seeds are averaged within each partition; cap 40 uses the identical full set once. The primary setting is cap 40, 95% retention and 95% five-arrival confidence for each prespecified method/order/partition.

The zero-loss necessary effective count is log(0.01)/log(0.95)=89.78. Original CIFAR counts are 80 and Pets 45--60, so no method could pass even at zero observed protection loss under the original bound. This identifies a certificate-feasibility bottleneck, not an empirical failure of the retained scoring mechanism.

| Encoder | Dataset | Queries | Certified expansion | Test met | Novel % |
|---|---|---:|---:|---:|---:|
| l14 | CIFAR-100 | 1600--1600 | 10/10 | 10/10 | 72.76 |
| l14 | Pets | 600--600 | 10/10 | 10/10 | 88.51 |
| l14 | ImageNet-R | 3636--3778 | 10/10 | 10/10 | 80.74 |
| b32 | CIFAR-100 | 1600--1600 | 10/10 | 10/10 | 52.54 |
| b32 | Pets | 600--600 | 10/10 | 10/10 | 61.94 |
| b32 | ImageNet-R | 3636--3778 | 10/10 | 10/10 | 49.67 |

Budget is a per-class cap; ImageNet-R availability can truncate it. Actual totals and minimum counts are in the CSV. Every operational mean includes base-only fallback with zero novel accuracy. Test attainment including fallback must not be equated with useful certified admission.

All six methods use identical calibration subsets and the same original reference. The sufficient protection criterion conservatively ignores recovery of reference errors by replacement readouts. Population confidence relies on independent class-conditional sampling and prespecified arrivals; this is an adaptive revision of previously examined benchmarks, not a new blind test.

Full tables, paired conditional bootstrap intervals, trajectory outcomes and all caps remain in calibration_budget_summary.json/CSV and calibration_budget_paired.csv. Paired policy gains include all ten partitions; jointly finite sets and their conditional mean gains are additionally identified. Supplemental identities and content hashes exclude fitting, tuning, support and test images. Frozen features are reused wherever possible; only new calibration images are encoded.

At the primary cap40 setting all final retained-table partitions admit expansion. All 20 test trajectories per dataset/encoder meet their targets; Pets has 17/20 L/14 and 16/20 B/32 paths with finite admission at every arrival, with explicit early-session fallback on the remaining paths. This distinction is retained rather than counting fallback as useful admission.

Verification completed 2026-10-23: 93,600 metric records, 88,216 reconstructed threshold tests, zero protected-pool overlap, frozen model/test hashes unchanged. Seven scientific tests pass across risk-control and budget suites. A real rerun preserves SHA-256 and modification time of 1,996 checkpoint files; extension storage is 198.9 MiB. No model weights or images were copied.

The figure-quality review prompted explicit final-arrival/order scope, correlated cell denominators, clipped SD bands and readable type (minimum 8.26 pt when placed at 5 inches). It is a figure review, separate from a full scientific manuscript audit. Main20 pages, Supplement54 pages, abstract153 words.

A second figure-only check of the compiled manuscript and supplement passed: no remaining significant readability or caption-scope issues. The optional fallback wording was clarified.
