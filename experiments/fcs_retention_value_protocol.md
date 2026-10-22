# Retention-constrained recognition and classifier value

Locked before this extension's first result, 2026-10-21.

Question 1: how much novel recognition is available at a common base accuracy
constraint? Question 2: what incremental predictive value does the retained
base-trained classifier provide over five consecutive class-arrival sessions?

Use the existing frozen features, three actual partitions and disjoint five-shot
supports on CIFAR-100 and Pets. Two arrival orders per partition use seeds
1000+p and 1100+p; five sessions admit 8 or 3 classes respectively. The original
base group remains the reference group throughout.

Compare retained text+learned / text+prototype fusion, the same fusion with its
learned contribution removed and all other gates fixed, separately calibrated
text+novel-prototype fusion with no learned columns, and dense anchored
text+prototype fusion. Every model uses the same image features, reference banks,
class partition, full base fitting access and arriving support. Dense prototype
fusion uses all base training images for its base prototypes.

Gate calibration uses only the existing pseudo-base/pseudo-novel episode.
For retained fusion a∈{0,.25,.5,1,2}; prototype mixtures use
u∈{0,.25,.5,1,2,4}; novel offsets use [-4,4] step .25. Select harmonic macro
accuracy with deterministic tie order. Calibration occurs once per partition,
and all branch coefficients remain fixed across arrival sessions.

Common budgets: 80%, 85%, 90%, 95% and 99% of the retained model's closed-base
accuracy in the same partition. Every compared method receives the SAME
absolute threshold; methods never normalize targets by their own ceiling.

Two distinct evaluations:

1. **Descriptive matched-test-retention frontier**: freeze calibrated branch
   coefficients and sweep only the group admission threshold. Use macro-weighted
   correct-base margin order statistics to find the largest admissible offset.
   This retrospective analysis measures recognition capacity at a common TEST
   constraint and does not select a deployed operating point. Record achieved
   base accuracy, novel accuracy, threshold, feasible status and target overshoot.
   An infeasible constraint is missing, never a zero novel accuracy. Joint-feasible
   comparisons use paired partitions and report their count.
2. **Base-validation-calibrated operation**: set the same retained-closed-base
   fraction on actual base validation examples at each session, with no actual
   novel validation/test labels. Apply the selected admission threshold to test
   queries and report both achieved groups. These test accuracies need not match.
   This policy recalibrates the group offset; it does not update encoder or base
   classifier parameters. Previously appended novel scores can change through
   the offset. The fixed-offset append-only theorem remains a separate mechanism.

Also report fixed-gate sequential macro-all/base/novel accuracy and paired
retained-minus-removed as well as retained-minus-recalibrated-no-learned gains.
At a matched base budget, report paired novel gains separately. Each partition
averages the two arrival orders before computing means and SAMPLE SD across
partitions. Final session orders share all novel rows and are not extra seeds.

Preserve original primary and review results. New records live in each dataset's
`retention_value/` directory with byte-exact source snapshots. No test-driven
choice of gate, support subset, class partition or headline budget is allowed.
