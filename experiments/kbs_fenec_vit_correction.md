# FeNeC ViT configuration correction, 2026-10-22

Reviewing the full author paper after the first runs identified that its ViT
experiments apply covariance shrinkage once, while double shrinkage is used for
ResNet. The first transferred readout used double shrinkage and a small
representative budget. Preserve those runs, but do not use them as the final
recent-baseline evidence.

For the final frozen-CLIP transfer, use single shrinkage, no extra sample or
centroid normalization (the input CLIP vectors are already unit-normalized), and
identity Tukey transformation for signed features. Select K in {1,5,25}, neighbor
count in {1,3,5}, and paired gamma settings {(1,1),(6.12,8.10),(9.98,0)} exclusively
on the same base-only pseudo episode. The latter two gamma pairs come from the
author's reported ViT settings. Classes still use min(K,n_class) representatives;
single-sample covariance uses identity. K-means remains Euclidean, seed 42,
at most 100 iterations. The Euclidean variant uses the same representative and
neighbor grids, with no covariance.

This is a transparent post-inspection correction, not a claim that this final
baseline grid was preregistered before all test results. It does not change the
retained table, its controls, image pools, original metrics or class orders.
Store corrected checkpoints and records in `fenec_vit/` separately. Report native
and group-offset-calibrated forms and retain sparse global-neighbor votes as the
baseline's score interface; an infeasible offset target must not be interpreted
as a bound on a different dense or groupwise classifier.
