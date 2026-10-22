# Independent calibration coverage and label-budget experiment

Locked 2026-10-22 before extracting supplementary features or evaluating this
experiment. This extends a previously examined benchmark; it is not a new blind
test or evidence that a single-paper acceptance probability has been calibrated.

## Frozen comparison

Reuse the exact fitting/tuning indices, selected readouts, class partitions,
five-shot support, encoders and two five-arrival orders of kbs_independent_risk.
Do not change fitted weights, pseudo-episode mixtures, temperatures, cache
hyperparameters, candidate offset families, or test pools. Test prediction
archives are reused, not copied or retuned. Compare retained, recalibrated
removal, dense prototypes, raw fusion, fixed temperature and class-mean cache.

## Independent calibration pool

For each class, first use the existing independent calibration observations.
If fewer than 40 remain, supplement from the original dataset's unused training
images, excluding every fitting, tuning, support and test image by identity and
content hash. Calibration selection uses only class membership, fixed random
seeds and availability, never image scores, calibration errors or test outcomes.
Randomly order existing calibration images first and supplemental candidates
second with seed 9032026, retaining at most 40 unique observations per class.
Keep input identities, pixel/content hashes, cached feature indices and protected
pool hashes in an immutable manifest. Use the same manifest for both encoders.
No novel query labels enter model or threshold selection within a partition.

Limits are per-class CAPS, not guaranteed exact counts. ImageNet-R has classes
with fewer than 40 available images; retain all available images and report
actual per-class and total query counts. CIFAR/Pets availability auditing found
at least 40 candidates per class before the content-hash exclusion audit.

Reuse old calibration features where available. Encode only supplemental images
with the pinned existing CLIP weights and identical dataset preprocessing.
Check fresh re-embeddings against existing cached calibration features and save
small batch-resumable feature shards. Do not copy image arrays or model weights.

## Curves and controls

Caps: 2,5,10,20,40. Subset seeds: 0,1,2, using RNG seed 771000+seed.
For each seed permute the fixed maximum pool within each class and take nested
prefixes min(cap, available). All readouts/encoders share identical subsets.
Primary largest-cap result is prespecified as 40; do not choose per-dataset caps
using test outcomes. At the largest cap all three seeds select the same maximum
set; count that set ONCE, not as three independent calibration repetitions.

Primary protection: r=.95 and delta=.05 across five arrivals for each prespecified
method/order/partition/cap/subset. Apply the unchanged weighted protection loss,
class-macro weights, fixed candidate family, fixed-sequence testing, delta/5 per
session, and base-only fallback. Also report empirical 95% threshold selection
on the exact same independent subsets. No joint guarantee across all budget
choices, subset seeds or methods is claimed. Coverage guarantees rely on the
stated independent class-conditional sampling assumptions; prior adaptive use
of these benchmarks remains disclosed.

Increasing a cap changes both the number and the class coverage of calibration
labels. Fitting/tuning labels remain constant. Compare original long-tailed
calibration with capped allocation as a coverage diagnostic, not as a pure causal
effect of total query count. All comparators get identical label access.

## Reporting and integrity

For all datasets/encoders/methods/caps report: actual total queries, minimum class
count, useful finite certification rate, fallback rate, achieved test base/novel
accuracy, 95% test-target attainment and violation size, and all-session trajectory
attainment. All operational means include fallback with zero novel accuracy.
Final vocabulary counts order 0 once; trajectories use both orders. Average
subset replicates within partition, then report ten-partition variation. Pair
method gains on identical partitions; 95% percentile bootstrap uses 20,000 paired
partition resamples, seed 2026. Do not call overlapping partitions or nested
subsets independent datasets/trials. Preserve all outcomes, including negative
comparisons or no useful certification.

Use two CPU threads, one shared-lock local GPU, initial inference batch 16 with
OOM halving, atomic feature/session checkpoints and shared PAUSE. Save only new
calibration sufficient statistics and decisions; reference existing test archives.
Verify exclusion hashes, original model/selection hashes, test archive hashes,
all decision traces/metrics, and actual checkpoint reuse before publication.
