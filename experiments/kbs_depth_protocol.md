# KBS evidence depth extension — locked before these new evaluations

Date: 2026-10-22. Existing three-dataset results remain unchanged. The purpose is
to strengthen two claims: novel recognition at common base accuracy, and paired
classifier value over five arrivals. No journal-quartile or acceptance guarantee
is inferred from experiment counts.

## B1: Dense semantic-cache comparator

Implement training-free Tip-Adapter's published score
`100 * image @ text.T + alpha * exp(-beta * (1-image @ cache.T)) @ one_hot`.
Primary sources: ECCV 2022 paper https://arxiv.org/abs/2207.09519 and author code
https://github.com/gaopengcuhk/Tip-Adapter at commit
`d0e2d6f8c5feb8b6ce937b757810761f7155d4d5` (main.py lines 25–58).
All readouts share existing frozen single-view CLIP caches, text prompts, fitting,
validation and support access. This is a controlled transfer, not reproduction
of the paper's augmented-cache or fine-tuned Tip-Adapter-F experiments.

Use all allowed base fitting examples and five novel support examples as keys.
Report the original summed retrieval and a clearly labeled class-mean retrieval
control (divide each class cache score by its key count). The latter tests the
long-tail count confound. Beta grid {0.1,1,2,5,10,20,50}, alpha grid
{0.1,1,5,10,20,50,100,200}. Jointly select beta/alpha and group offset from
[-40,40] step 2.5 using ONLY each original base pseudo episode. The offset grid
has the same 33 entries as table calibration; its scale reflects raw CLIP logits.
Also select a native zero-offset beta/alpha pair independently on that episode.
No true novel query labels or test labels enter selection. After selection,
cache contributions are dense per class and independent of appended classes.

Three existing partitions/dataset, two orders and five sessions. Use precisely
the original retained table's reference targets for matched-test and validation
threshold policies at ratios {0.8,0.85,0.9,0.95,0.99}. Include feasible counts and
all outcomes. Accuracy and table feasibility decide conclusions, not prechosen
ranking. Validate retrieval against a direct scalar loop before launch.

## B2: Calibration and shot robustness, with mechanism controls

Three datasets × three existing class partitions × shots {1,2,5} × three
predeclared calibration/support trials = 81 configurations. Trial 0 uses the
original pseudo split; trial 1/2 split base classes with seeds 1701+p / 2701+p
and pseudo-base size floor(B/2). Support seeds are 7,8,9 respectively. Five-shot
trials reuse the same reserved five examples; their variation comes from the
pseudo split, not independent five-shot support pools. The other trials jointly
vary calibration and subset choice; do not attribute variance to either alone.

Fit pseudo classifiers with ridge selected on pseudo-base validation; select
coefficients using pseudo-base/pseudo-novel validation only. Compare retained,
removed-fixed, independently recalibrated removal, dense text+prototype, and
retained dynamic normalization. Dynamic uses current-candidate text statistics
and current novel prototype statistics, leaving the learned branch unchanged;
select its own mixture/offset using exactly the same grid as retained. Its
purpose is to isolate the fixed-reference mechanism, not a published baseline.

Evaluate two arrival orders and five sessions, frozen settings within sequence.
Reference budgets use the retained readout's closed-base classifier per
configuration, shared by every comparator. Save sufficient statistics for both
test and base-validation queries; archive source/protocol/cache hashes. No
test-selected coefficients. Record maximum previous-row score drift and
restricted previous-vocabulary prediction changes after each append.

Aggregate orders and trials WITHIN each partition, then report three partition
means and sample SD. Matched comparisons average only jointly feasible trials;
report trial and partition counts. No p-values or significance claims from 81
correlated configurations. Operational results report achieved test base
accuracy separately from validation feasibility and target attainment.

## B3: Measured cost

On the existing RTX 4070 Laptop, measure readout-only median/p90 latency on 512
cached queries, 5 warm-ups and 20 timed repetitions, excluding image encoding.
Compare retained and dense prototypes against both retrieval variants using
identical GPU, float32 and batches. Count persistent readout tensor bytes
explicitly, excluding common query/image encoder and temporary workspaces.
Do not call this full-model memory or end-to-end deployment latency.

## Execution, decisions and persistence

Warm-reuse `.aris/compute/local.md`; no new environment or paid compute. Use
existing OS GPU lock, 2 CPU threads, 75% CUDA allocator cap, query chunks 256 for
cache retrieval. Save selection and each session atomically. Shared PAUSE flag
stops at saved units. Resume verifies immutable source/protocol/cache manifests
before reusing files. Estimate: 1–4 GPU hours worst case; measure actual duration.

Run scalar numerical tests → B1 all three datasets → B2 → archive verifier →
result-to-claim analysis → measured manuscript integration → clean compilation.
If comparisons reverse a claim, revise the claim and retain the results. Do not
increase grids or choose new partitions after inspecting outcomes in this round.
