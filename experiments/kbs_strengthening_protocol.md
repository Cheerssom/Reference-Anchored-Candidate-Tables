# Locked KBS evidence extension, 2026-10-22

This plan is written before evaluating the new dataset or FeNeC baseline. Prior
CIFAR/Pets results remain immutable and are not used to tune this extension.

## Claims and intended evidence

1. Measure novel accuracy under the same absolute original-class accuracy target.
2. Quantify the paired contribution of retaining the original learned classifier
   across five successive arrivals, against removal, recalibration, prototype
   replacement and a recent frozen-feature incremental baseline.

The extension adds **ImageNet-R, 200 classes**, emphasizing larger vocabulary
and rendition-domain variation. It is not described as ImageNet-1K or as a larger
image-count dataset than CIFAR-100. Source: https://github.com/hendrycks/imagenet-r .

## ImageNet-R access and split

- Use the authors' public 30,000-image archive, record its hash, and exclude
  repeated byte-identical images before assigning pools. Save relative image
  paths, class identities and content hashes.
- For each class, permute unique images with seed `2026 + class_id`; use 20%
  (floor, minimum 5) for test, next 20% (floor, minimum 5) for base validation,
  next five for reserved novel support, and the remainder for the fitting pool.
  Exact duplicate bytes cannot cross pools. These are task-specific splits,
  not a claim of an official train/test partition.
- Downsample the fitting pool by global class rank with desired counts
  `round(200 * 100**(-class_id/199))`, capped by available fitting images and
  floored at 2. Save achieved class counts and imbalance rather than claiming
  an exact achieved ratio of 100 after capping.
- Frozen, locally cached OpenAI CLIP ViT-L/14, no task-adapted encoder or historical
  LoRA checkpoint. RGB 224-square bicubic image resize and the same CLIP channel
  normalization used in the Pets pipeline. Use the existing eight prompt strings
  and official ImageNet synset-to-name metadata. Record all preprocessing choices.
- Class partitions: seeds 0, 1, 2; stratify the 200 class IDs into thirds by
  desired fitting frequency, select 100 base classes using the existing exact
  allocation rule, and use the remaining 100 as novel. No held-out negative group.
- Within each base set, randomly select 50 pseudo-base classes with seed 900+p;
  the remaining 50 are pseudo-novel. All calibration uses base validation only.
- Ridge grid {0.01,0.1,1,10}; branch/offset grids and retained/removal/prototype
  configurations match the existing retention-value protocol.
- Five arrivals of 20 novel classes; two arrival orders using seeds 1000+p and
  1100+p, averaged within partition before the SD across three partitions.

## Recent matched baseline: FeNeC (KBS 2026)

Source paper: https://arxiv.org/abs/2503.14301 ; author code:
https://github.com/gmum/FeNeC , pinned commit
`1575ea019fd583149c47c068f7f5ccc102ea4580`.

Use the nearest-neighbor variant with per-class covariance, double shrinkage,
correlation normalization, Euclidean k-means and inverse-Mahalanobis-distance
votes. Signed CLIP features use identity Tukey transform. Select centroid budget
K in {1,3,5}, neighbor count in {1,3,5}, and diagonal shrinkage gamma1 in {1,10}
(gamma2=1) by macro harmonic accuracy on the same pseudo episode. Small classes
use min(K,n_class) centroids; a class with one fitting sample uses identity
covariance. These two few-shot/long-tail adaptations must be named explicitly.
The baseline receives all permitted base fitting features and exactly the same
five novel support features as other methods. It gets no semantic anchors,
consistent with the published visual readout; an auxiliary semantic fusion
should not be silently substituted for the method.

Convert positive class votes to log votes (floor 1e-30), which preserves the
zero-offset classifier. Select a novel-group offset on the same pseudo episode
using the existing offset grid [-4,4], step 0.25. Report the unmodified zero-offset
baseline in addition to this explicitly labeled calibrated baseline. Recompute
global neighbor votes over only the currently active class representatives at
each arrival. No score-preservation claim is attached to this global kNN baseline.

Run FeNeC on CIFAR and Pets as well as ImageNet-R under their existing splits,
support and cached features. Validate covariance, clustering and voting against
the pinned upstream implementation on a synthetic fixture; independently verify
archived score/prediction statistics and common-retention margins.

## Reporting

- Keep fixed-gate accuracy trajectories, 80/85/90/95/99% common matched-test
  retention frontiers, and a separate validation-only threshold policy.
- Every method uses the retained table's same per-partition closed-base target.
  Infeasible targets are missing with counts, never replaced by zero.
- Paired differences use jointly feasible partitions. No test-selected branch
  hyperparameters, no significance claim based on three partitions.
- Save source snapshots/hashes, selected hyperparameters, active class IDs,
  labels, group margins and predictions. Keep all outcomes, regardless of ranking.
- Add only measured findings to the KBS manuscript. New evidence may narrow or
  change the emphasized operating regime; it must not be invented to support
  the existing abstract.

Compute: reuse the existing local RTX 4070 Laptop / Python environment. No paid
GPU service is authorized or needed. Dataset download and frozen extraction
precede the final comparisons. Record actual elapsed time and peak memory.
