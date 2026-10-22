# Validation-budget sensitivity — fixed before the new subset evaluations

Date: 2026-10-22. Purpose: quantify how many old-class validation labels the
admission-threshold policy needs, separate from fusion calibration and from the
descriptive matched-test frontier. Existing results are immutable.

## Design

- Reuse only the archived test/validation sufficient statistics from the original
  trial-zero, five-shot `kbs_depth/{dataset}/robust/p{p}_r0_k5` and corresponding
  `tip/p{p}` runs. The four table readouts and summed/class-mean calibrated
  Tip-Adapter are frozen; no model, feature, mixture or cache setting is retuned.
- Datasets: CIFAR-100, Pets, ImageNet-R. Original partitions p in {0,1,2}; arrival
  order zero; all five sessions. The final active vocabulary is identical across
  arrival orders, so the experiment does not duplicate the final stage as an
  independent order sample.
- Within every base class, retain max(1,floor(fraction*n_class)) validation
  images for nominal fractions {0.25,0.5,1}. Record actual counts and actual
  fractions; never describe the nominal fraction as exact. Classes with one
  image necessarily keep it.
- For the reduced fractions, subset seeds {0,1,2}. For class c the permutation
  seed is 51000 + 1000*p + 100*subset_seed + c. Use the same class permutation at
  25% and 50%, so subsets are nested. Use all validation rows once at 100%, with
  subset seed zero; do not count identical full-data subsets as extra trials.
- Ratios {0.95,0.99}. Compute the retained classifier's closed-base macro accuracy
  on the selected validation subset. All six readouts receive the same absolute
  target: ratio times that closed-base accuracy. The selected offset is the
  largest feasible offset under the existing canonical class-ID tie rule.
- Only selected base-validation labels/within-group predictions/margins enter
  threshold selection. Evaluate test old/novel accuracy only after selection.
  The test criterion ratio*closed-base-test-reference is used ONLY to report
  achieved retention and target attainment, never to change the selected offset.
- An infeasible validation target has no offset or test metrics. Report counts.
  Do not insert a zero result, discard adverse feasible trials, alter the grid,
  or select an experiment subset based on test performance.

## Reporting

For each dataset/fraction/ratio/readout, average the subset repeats within each
partition and then report the mean and sample SD of the available partition
means. Report number of feasible partition/subset combinations: /9 for reduced
fractions and /3 for full validation. Report test target attainment as a count
among those feasible combinations, not a binomial confidence interval or
independent-trial significance claim. At the final arrival, report actual sample
count range, old and novel test accuracy, and achieved test-retention ratio.
Keep all five session records for reproducibility. This is threshold-label
sensitivity conditional on the original full-data fusion calibration; it is not
a claim of reducing total training or calibration labels.

## Cost, persistence and validation

CPU only, no images, feature extraction, GPU allocation or additional downloads.
Save each session atomically; use the existing shared PAUSE flag. Manifest the
protocol, script and all original archive hashes before reusing a saved unit.
Two numerical tests: compare threshold selection with exhaustive boundary
enumeration, and verify deterministic class-stratified nested sampling including
singleton classes. Reconstruct saved policies independently from archived
predictions and verify that the full-budget outputs reproduce the original
base-validation policy, including infeasible targets. Verify checkpoint reuse
by saving one unit, rerunning and comparing hash and modification time.

Expected units: 3 datasets * 3 partitions * (3+3+1 subsets) * 5 sessions = 315;
expected metric rows: 315 * 6 readouts * 2 retention ratios = 3780.
