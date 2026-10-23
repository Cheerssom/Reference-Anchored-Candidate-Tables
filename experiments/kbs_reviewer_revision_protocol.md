# Locked five-block reviewer revision

Locked 2026-10-23 before these new outputs. Existing test benchmarks have informed
development; this is a controlled revision, not a newly blind external test.

Primary claims: retained discrimination provides admission capacity at a common
base-accuracy target; query-dependent scaling contributes independently of the
frozen score-frame invariant. No universal prototype-replacement advantage.

1. Normalization factorial: query/fixed centering crossed with query/fixed scale.
   Freeze the independently fitted ridge, reference bank and retained a/u gates.
   Fixed moments use only base tuning queries. Compare fixed gate, descriptive
   matched95 capacity and independently calibrated policies over all five arrivals.
2. Mechanism diagnosis: per-class fitting counts, learned/reference and prototype
   closed-base correctness, prototype cohesion and classifier/prototype
   disagreements. Frequency thirds and correlations are descriptive, conditional
   on fixed data; no causal dataset-property claim from correlations alone.
3. Calibration fairness: same frozen six existing readouts, cap40 images,
   candidate thresholds and five-arrival delta=.05. Compare original sufficient
   protection test with a paired accuracy e-value test of A_C >= .95 A_R that
   credits recovered reference errors. Fixed mixture lambda grid, no selection
   on calibration or test outcomes. FST stops first failure at delta/5.
   Report finite admission, conditional admitted accuracy, fallback-inclusive
   accuracy, test target attainment and destroyed/recovered reference correctness.
4. Recent baseline: author-code-controlled HyCal transfer on both existing CLIP
   encoders, identical permitted fitting images and five-shot support, same
   reference target and calibration access. Sum-fused image/text prototypes,
   class covariance, Gaussian-CDF score calibration and sample-count weighting.
   Class ridge regularization candidates1e-4/1e-3/1e-2 selected on base-only pseudo
   expansion. n_min_p=5, scale=1, identity shrinkage as official default (real=False).
   The existing common eight-prompt text anchors are shared by all methods;
   this is a controlled transfer, not a reproduction of published ViT-B/16 numbers.
5. Distribution shift: CIFAR image Gaussian noise sigma=.08 and Gaussian blur
   radius1.25; no transformed outputs examined before locking. Encode all4000
   independent calibration images and20 test images/class, selected with seed
   23102026 before feature extraction. Noise seed depends only on role/image ID,
   seed23102026 and corruption. Keep all fitted models, gates, supports and clean
   tuning threshold families fixed. Compare clean-cal/shift-test against
   shift-cal/shift-test. No guarantee under unmatched calibration/test shift.
   This is covariate-shift stress testing, not an untouched external dataset.

All core comparisons: three datasets, two encoders, ten class partitions, two
arrival orders, five arrivals; macro accuracies and partition-unit conditional
paired20k bootstrap intervals(seed2026). Fallback original reference has zero
novel accuracy and is never counted as finite admission. Statistical confidence
is per prespecified method/order/partition/calibration policy under independent
class-conditional sampling; not simultaneous across the experimental grid.

Compute: reuse F:/Anaconda/python.exe and existing weights/features, one serial
GPU stage, two CPU threads, batch16 with OOM halving, shared PAUSE marker,
atomic checkpoints. No paid compute or new model downloads. New artifacts live
under experiments/kbs_reviewer_revision; previous runs remain immutable.
