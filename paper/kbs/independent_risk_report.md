# Independent risk calibration results

All 60 refitted partition runs and 600 arrival sessions completed; 21,600 metric records reconstructed independently. The original fitting/validation pools are repartitioned before all ridge, fusion and cache selection. Three disjoint pools share indices across encoders and readouts. No new images, model weights or feature caches are copied.

Primary: 95% population macro retention, confidence 95% across the five arrivals of each prespecified method/order/partition, conditional on the stated class-conditional sampling and model assumptions. This is not a simultaneous guarantee across all methods or benchmarks, a guarantee under distribution shift, or a guarantee for every finite test sample. The test benchmark has been used previously in research development.

Cert. exp. counts finite certified thresholds. When no such threshold exists, the shared original reference classifier serves base classes and novel accuracy is zero. All-ten test attainment including fallback is not evidence of useful expansion. The certificate uses a common sufficient protection loss; it conservatively ignores recovery of reference errors by replacement readouts.

| Encoder | Dataset | Readout | Empirical test met | Empirical novel % | Certified expansion | Risk-policy test met | Risk-policy novel % |
|---|---|---|---:|---:|---:|---:|---:|
| l14 | CIFAR-100 | Retained | 7/10 | 76.28 | 0/10 | 10/10 | 0.00 |
| l14 | CIFAR-100 | Recalibrated removal | 6/10 | 35.32 | 0/10 | 10/10 | 0.00 |
| l14 | CIFAR-100 | Dense prototypes | 6/10 | 71.92 | 0/10 | 10/10 | 0.00 |
| l14 | CIFAR-100 | Raw fusion | 5/10 | 34.46 | 0/10 | 10/10 | 0.00 |
| l14 | CIFAR-100 | Fixed temperature | 8/10 | 71.36 | 0/10 | 10/10 | 0.00 |
| l14 | CIFAR-100 | Class-mean cache | 4/10 | 65.73 | 0/10 | 10/10 | 0.00 |
| l14 | Pets | Retained | 1/10 | 93.07 | 0/10 | 10/10 | 0.00 |
| l14 | Pets | Recalibrated removal | 4/10 | 93.29 | 0/10 | 10/10 | 0.00 |
| l14 | Pets | Dense prototypes | 0/10 | 95.08 | 0/10 | 10/10 | 0.00 |
| l14 | Pets | Raw fusion | 4/10 | 86.90 | 0/10 | 10/10 | 0.00 |
| l14 | Pets | Fixed temperature | 3/10 | 91.87 | 0/10 | 10/10 | 0.00 |
| l14 | Pets | Class-mean cache | 1/10 | 92.50 | 0/10 | 10/10 | 0.00 |
| l14 | ImageNet-R | Retained | 7/10 | 83.45 | 10/10 | 10/10 | 77.64 |
| l14 | ImageNet-R | Recalibrated removal | 4/10 | 57.36 | 0/10 | 10/10 | 0.00 |
| l14 | ImageNet-R | Dense prototypes | 6/10 | 82.47 | 5/10 | 10/10 | 24.12 |
| l14 | ImageNet-R | Raw fusion | 3/10 | 64.80 | 4/10 | 10/10 | 26.42 |
| l14 | ImageNet-R | Fixed temperature | 5/10 | 81.28 | 10/10 | 10/10 | 65.69 |
| l14 | ImageNet-R | Class-mean cache | 4/10 | 79.37 | 2/10 | 10/10 | 6.20 |
| b32 | CIFAR-100 | Retained | 1/10 | 60.64 | 0/10 | 10/10 | 0.00 |
| b32 | CIFAR-100 | Recalibrated removal | 2/10 | 47.79 | 0/10 | 10/10 | 0.00 |
| b32 | CIFAR-100 | Dense prototypes | 2/10 | 52.74 | 0/10 | 10/10 | 0.00 |
| b32 | CIFAR-100 | Raw fusion | 2/10 | 50.33 | 0/10 | 10/10 | 0.00 |
| b32 | CIFAR-100 | Fixed temperature | 5/10 | 55.06 | 0/10 | 10/10 | 0.00 |
| b32 | CIFAR-100 | Class-mean cache | 2/10 | 57.79 | 0/10 | 10/10 | 0.00 |
| b32 | Pets | Retained | 1/10 | 75.10 | 0/10 | 10/10 | 0.00 |
| b32 | Pets | Recalibrated removal | 10/10 | 28.49 | 0/10 | 10/10 | 0.00 |
| b32 | Pets | Dense prototypes | 4/10 | 54.67 | 0/10 | 10/10 | 0.00 |
| b32 | Pets | Raw fusion | 6/10 | 53.57 | 0/10 | 10/10 | 0.00 |
| b32 | Pets | Fixed temperature | 5/10 | 64.90 | 0/10 | 10/10 | 0.00 |
| b32 | Pets | Class-mean cache | 7/10 | 42.00 | 0/10 | 10/10 | 0.00 |
| b32 | ImageNet-R | Retained | 2/10 | 56.51 | 10/10 | 10/10 | 45.30 |
| b32 | ImageNet-R | Recalibrated removal | 9/10 | 3.58 | 0/10 | 10/10 | 0.00 |
| b32 | ImageNet-R | Dense prototypes | 4/10 | 43.48 | 0/10 | 10/10 | 0.00 |
| b32 | ImageNet-R | Raw fusion | 9/10 | 3.75 | 0/10 | 10/10 | 0.00 |
| b32 | ImageNet-R | Fixed temperature | 2/10 | 52.45 | 7/10 | 10/10 | 27.28 |
| b32 | ImageNet-R | Class-mean cache | 5/10 | 33.38 | 0/10 | 10/10 | 0.00 |

Full confidence-level and safety-margin outcomes are in independent_risk_summary.csv. Exact threshold test traces and raw predictions remain in experiments/kbs_independent_risk. Methods use a smaller refitted training pool than the earlier manuscript comparisons; their numerical results must not be mixed as if the fitted models were unchanged.

Reproduce: scripts/run_kbs_independent_risk.ps1; python src/experiments/kbs_verify_independent_risk.py; python paper/kbs/build_independent_risk_results.py. Shared PAUSE: experiments/kbs_strengthening/PAUSE. Each completed session and partition is saved atomically.

Validation: four scientific edge-case tests passed, including exhaustive paired-outcome implication and exact enumeration of a small heterogeneous null. A full rerun preserved SHA-256 and modification times of all 1,320 checkpoint files. Run storage is approximately 184.5 MiB; existing features were reused.

For empirical controls, infeasible calibration targets also use the shared reference fallback. Thus every operational mean and test-attainment count includes all partitions for both empirical and risk policies; the full CSV reports empirical feasibility through its expanded/fallback fields.
