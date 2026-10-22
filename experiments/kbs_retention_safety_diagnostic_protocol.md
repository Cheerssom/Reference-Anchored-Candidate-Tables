# Retention safety diagnostic (exploratory)

Date: 2026-10-22. Written before this diagnostic is executed.

Purpose: quantify the trade-off between a stricter validation target and held-out
test retention in the existing fixed models. This diagnoses the review concern;
it does not establish a deployment guarantee or select a final published policy.

- Inputs: immutable ten-partition/two-encoder prediction archives, order 0,
  session 5. Both orders have the same complete final vocabulary; count once.
- Datasets: CIFAR-100, Pets, ImageNet-R. Encoders: L/14 and B/32.
- Readouts: retained, recalibrated removal, dense anchored prototypes, and
  class-mean semantic cache, with their original locked calibration parameters.
- Validation target fractions of the retained closed-base reference:
  0.95, 0.97, 0.99, 1.00. Report every value; do not choose one from test results.
- Test success always uses the original 0.95 test-reference target.
- Report feasibility, attained test target count, base/novel macro accuracy,
  signed base-accuracy gap to target, sample sizes, and paired novel-accuracy
  cost relative to the original policy on jointly feasible partitions.
- Infeasibility remains explicit, with no silent fallback or excluded failures.
- Sanity: reproduce all original 0.95 validation-policy metrics and test flags.
- CPU only, one atomic checkpoint per encoder/dataset/partition, shared PAUSE
  marker, SHA-256 input/code/protocol guards on resume. No feature extraction.

The original validation labels also selected ridge regularization, pseudo-episode
fusion and cache settings. Results from these archives are exploratory and cannot
be upgraded to independent-calibration certificates by adding confidence bounds.
Ten partitions share images; test-attainment counts are descriptive, not ten
independent Bernoulli trials.

Next confirmatory protocol must separate fitting, hyperparameter selection,
risk calibration and testing before recomputing fitted readouts. It must define
macro accuracy, reference uncertainty, confidence level, threshold search and
simultaneous control across arrivals. A population-risk guarantee is distinct
from guaranteed attainment on every finite test sample. A policy with no certified
nontrivial threshold must explicitly decline expansion and report zero novel
admission, rather than advertise a useful certified expansion.
