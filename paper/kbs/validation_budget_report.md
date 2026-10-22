# Validation-query budget sensitivity

Completed 2026-10-22. Purpose: evaluate the amount of base-validation query data needed by admission-threshold selection, conditional on frozen original classifiers and full-data fusion calibration.

Design: three datasets, three existing class partitions, one fixed arrival order and five sessions. Nominal class-stratified query fractions are 25%, 50% and 100%; actual counts are reported after rounding and a one-query-per-class minimum. Reduced budgets use three nested subset permutations; full validation uses one subset. Six frozen readouts and both 95%/99% targets produce 315 saved units and 3,780 metric records. These are correlated subset trials, not independent class partitions.

On ImageNet-R, the nominal 25% budget retains 696–731 old-class validation queries versus 2,938–3,087 with full validation. Under the 95% validation policy, the retained table gives 85.85±0.38% base and 84.98±0.38% novel test accuracy; full validation gives 86.01±0.59% and 84.73±0.71%. Meeting the empirical validation constraint is distinct from test retention: the separate 95% test target is attained in 2/9 reduced-budget cases and 0/3 full-budget partitions. These counts do not imply a statistical guarantee or monotonic dependence on sample count. All targets and comparator outcomes are retained in validation_budget_summary.csv.

Two numerical tests pass. Independent reconstruction validates all 3,780 rows, source/input hashes, maximal feasible offsets, test metrics and attainment flags. The full-validation policies reproduce 540 original rows exactly (maximum delta 0.0), including feasibility. Checkpoint reuse preserves a saved unit's SHA-256 and modification time. This experiment uses CPU archived arrays only; no feature extraction, GPU stage or paid compute is needed.

The main manuscript includes a concise result paragraph; Supplementary Section H contains the full protocol and two tables. Current PDFs: 23-page main and 24-page supplement. No journal submission or new fresh-agent review has occurred. The earlier fresh-review invocation remains unavailable; local numerical checks are distinct from an independent scientific review.

Commands: python src/experiments/kbs_validation_budget.py ; python src/experiments/kbs_verify_validation_budget.py ; python paper/kbs/build_validation_budget_results.py . PowerShell: scripts/run_kbs_strengthening.ps1 -Stage ValidationBudget . The shared PAUSE flag stops between atomic session units.
