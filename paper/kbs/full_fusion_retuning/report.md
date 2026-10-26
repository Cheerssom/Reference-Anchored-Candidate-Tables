# Fully retuned fusion comparisons

Equal fusion-search budget on base-only pseudo episodes; original common reference; orders averaged within partition; test-label capacities descriptive

| Encoder/data | Query-none capacity gap pp | Query-fixed capacity gap pp | Retained-dense capacity gap pp |
|---|---:|---:|---:|
| l14/CIFAR-100 | +4.71 [+2.39,+6.92] (n=10) | +4.32 [+1.26,+6.95] (n=10) | +3.31 [+0.31,+6.07] (n=10) |
| l14/Pets | -2.61 [-8.33,+2.31] (n=10) | -2.83 [-6.73,+0.65] (n=10) | -2.62 [-9.24,+4.39] (n=10) |
| l14/ImageNet-R | +3.60 [+2.55,+4.74] (n=10) | +2.84 [+2.42,+3.28] (n=10) | +0.25 [-0.46,+0.92] (n=10) |
| b32/CIFAR-100 | +1.02 [-2.18,+3.36] (n=9) | +3.55 [-0.30,+7.28] (n=10) | +4.15 [+1.30,+7.44] (n=9) |
| b32/Pets | +12.49 [-4.08,+30.02] (n=8) | +5.93 [+0.68,+10.63] (n=10) | +3.92 [-8.88,+15.76] (n=8) |
| b32/ImageNet-R | +5.29 [+3.74,+7.25] (n=10) | +4.65 [+3.57,+5.71] (n=10) | +9.69 [+8.08,+11.21] (n=10) |

All schemes independently select both fusion coefficients and a tuning offset under the same3636-setting budget. Ridge regularization, fitting pools and encoder are held fixed. Dense prototypes have separate base/novel coefficients. The original retained reference and fallback remain common; retuned fusion can recover original-reference errors, so G=0 is not asserted here. Oracle capacities and independently selected policies are different metrics. Selection intervals condition on fixed images and jointly feasible partitions. No natural-shift results are included in this study.
