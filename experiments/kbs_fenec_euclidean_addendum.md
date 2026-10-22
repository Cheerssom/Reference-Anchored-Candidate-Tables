# FeNeC Euclidean-distance addendum, 2026-10-22

The original Mahalanobis FeNeC runs on CIFAR/Pets have completed and are retained
unchanged. Before evaluating ImageNet-R, add the Euclidean-distance variant
described in the FeNeC paper's distance ablation. This checks whether the
covariance-estimation requirement influences the five-shot comparison and gives
a computationally lighter member of the same recent baseline family.

Use the identical permitted data, signed frozen features (identity Tukey),
K in {1,3,5}, neighbor counts in {1,3,5}, and pseudo-episode harmonic selection
over offsets [-4,4], step 0.25. Euclidean distance is unsquared, matching the
author `EuclideanMetric`; vote weights are inverse distance. Report native
zero-offset and independently offset-calibrated forms. All selection is base-only.
Do not replace or discard the Mahalanobis results. Use separate checkpoint and
result files and record this addendum's hash. No other grid is changed.
