# Reviewer-directed replication and encoder extension

Locked before new score evaluation, 2026-10-22. Existing three-partition runs
remain immutable. The supplied 6/10 report is an external informal assessment,
not an editorial decision or a completed independent audit of this revision.

## Questions

1. Do fixed per-query references add recognition value beyond raw-score fusion
   and a simpler fixed branch temperature under identical calibration grids?
2. Do paired retained-classifier gains persist over ten distinct class partitions
   and a second frozen CLIP architecture?
3. What happens under validation-selected admission, including strong dense
   semantic/prototype baselines and explicit feasibility/test-target accounting?

## Frozen design

- Datasets/pools: the existing CIFAR-100, Pets and ImageNet-R image pools, names,
  prompts, preprocessing, base/novel counts and five-shot reserve. No new dataset,
  reallocation of pools or use of adapted historical features.
- Encoders: original OpenAI CLIP ViT-L/14 and locally cached OpenAI CLIP
  ViT-B/32. The latter uses cached snapshot
  `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; hash config/weights/tokenizer files.
  Image processing matches each original dataset, and both encoders use the same
  eight text templates. Do not compare absolute classifier targets across
  encoders: each encoder's retained closed-base reference defines its targets.
- Partition seeds 0..9: preserve the existing class memberships and pseudo splits
  for 0..2. Seeds 3..9 use the same stratified allocation, with pseudo-class
  permutations 100+p on CIFAR/Pets and random half-base seed 900+p on ImageNet-R.
  Partitions are distinct random class assignments on shared images, not ten
  independent datasets. Full and pseudo ridge parameters are selected from the
  existing grid on their corresponding base validation classes.
- Five arrivals and two orders, seeds 1000+p and 1100+p. Five-shot support seed
  remains 7. All mixture/cache parameters are selected once on the base-only
  pseudo episode, then remain fixed throughout arrivals.

## Compared readouts

Retained reference table, fixed-gate removal, independently recalibrated removal,
dense anchored text/prototypes, summed and class-mean Tip-Adapter with offset,
raw fusion and fixed-temperature fusion. Tip-Adapter uses the previously locked
beta/alpha/offset grids and the same access; it is a controlled single-view
readout transfer, not an augmented-cache published-protocol reproduction.

Raw fusion uses cosine text scores, raw base ridge outputs and cosine novel
prototypes. Fixed-temperature fusion divides each of those branches by a single
positive scalar sigma_e, with no candidate-wise or per-query normalization.
Estimate sigma_e as the mean per-query population standard deviation across the
fixed (pseudo-)base reference columns on base validation features, floored at
1e-6. Estimate these temperatures once for the pseudo episode and once for the
full base model before evaluation. They use base data only and stay fixed as
novel classes arrive. The two simple readouts use the same learned/prototype
coefficient and offset grids and harmonic pseudo-episode objective as retained.
Both simple controls also preserve old scores on append; do not claim that
score preservation is unique to reference anchoring or a substantive theorem by
itself. Their purpose is recognition/calibration isolation.

## Outcomes and statistics

Save fixed-gate old/new/all-class/harmonic accuracy; matched-test frontiers and
base-validation threshold policies for ratios 0.80/0.85/0.90/0.95/0.99. Every
readout gets the retained model's same absolute reference target within an
encoder/partition. Infeasibility has no fabricated test metrics. Report achieved
test-retention ratio and test target attainment separately from validation
feasibility. Test labels never select mixtures or the operational threshold.

Average two orders within each partition. Report ten partition means and sample
SD; paired percentile bootstrap 95% CIs resample the ten paired partition values
20,000 times with seed 2026. These CIs characterize random class-partition
variation conditional on fixed datasets, pools, support and encoder. They do
not estimate independent-image, pretraining or population uncertainty. Pair only
jointly feasible identical partition/order/session records, never unpaired
available means. Report counts, budget-dependent gains and unchanged/unfavorable
results as well. No claim of general superiority or standard FSCIL SOTA.

Use frequency-group per-class outcomes to describe the observed regime; the
study does not vary the underlying imbalance schedule and cannot establish
causal effects of a different imbalance ratio.

## Runtime and audit

Warm-reuse .aris/compute/local.md without dependency changes or paid compute.
One GPU process, two CPU threads, existing 75% CUDA allocator cap. B/32 image
batches start at 16, halve on OOM; features saved per batch with input/source
hashes and contiguous-index guards. Verify 8 fresh image embeddings against the
completed cache per dataset. Feature extraction and all per-session metric
archives support the shared cooperative PAUSE flag and atomic writes.

Validate raw/fixed-temperature equations on scalar fixtures, numerical
append preservation and original-seed L/14 agreement. Independently reconstruct
saved test and validation metrics and feasible offsets from prediction archives.
Then update the manuscript, comparison tables and a review-response ledger.
Show main and supplement together in a combined review PDF; no journal response
is sent or represented as an official revision submission.
