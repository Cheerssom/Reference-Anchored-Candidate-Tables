# Locked mechanism follow-up analysis

Date2026-10-23. This is a locked post-hoc analysis plan, written before new diagnostic outcomes. Existing benchmark results have already been examined; this is not prospective preregistration or an untouched evaluation.

## Frozen basis and scope

Reuse all six dataset/encoder conditions, ten partitions, both arrival orders, final session5 only. Models, fitting/tuning splits, five supports, selected coefficients, original cap40 calibration and original test prediction archives remain unchanged. Primary summaries use the first prespecified order0, matching the published81.61/76.76 comparison. Order1 is a separately reported sensitivity analysis. CPU threads2, existing local GPU only, no model/image downloads, atomic per-partition outputs and shared PAUSE marker.

## 1. Realized operating points and discrete envelopes

Read retained/dense exact score margins and predictions from the independent-risk archives. Enumerate exact weighted-accuracy changes including canonical class-ID ties, without interpolation. Overlay both protection and direct calibration points from immutable cap40 fairness records. Save all curves, plus a prespecified partition0 illustration and every-partition plots.

At each baseline's actual base accuracy, compute each readout's maximum novel accuracy at base accuracy at least that target. Report achieved accuracy mismatch, infeasibility and paired partition effects. Also compute envelopes at the original common0.95-reference target. These are descriptive test-label oracle readout comparisons, not new deployment thresholds or uniquely causal separation of classifier and calibration effects. Any signed envelope-minus-policy quantity is not called regret unless that policy itself satisfies the chosen test target.

Use one common reference for every readout. Check A_B=A_ref-D+G; split D into reference-correct errors to novel and to another base class. For retained scores require G=0 and D_to_base=0. Replacement G can be zero or positive.

## 2. Effective-bias identity and exploratory conditional distributions

For the retained readout use query-specific branch standard deviations with the original epsilon floor. Text centering is common to all candidates and cancels. Reconstruct uncentered scaled base/novel scores and Delta=a*mu_L/sigma_L-u*mu_P/sigma_P, so centered margin equals uncentered margin minus Delta in exact arithmetic. Compare numerical reconstruction and frozen archive predictions, reporting floating-point tolerances without modifying the original arrays.

Compute macro-weighted old/novel ECDFs of Delta for each partition. Primary figure uses prespecified partition0. Compute AUROC with novel as positive for Delta, minus uncentered margin, and minus centered margin; orientation is fixed and never flipped after looking at results. Conditional separation of Delta alone is neither necessary nor sufficient for accuracy benefit. Repeated-image partitions are not independent dataset samples; intervals resample ten partition means conditional on fixed images.

Remove query centering while retaining query scaling as a diagnostic intervention, reselecting direct-risk thresholds from the same tuning-derived candidate quantiles and same cap40 calibration images. This compares adaptive bias to a constant-offset family after calibration; it is not an independently retuned fusion architecture. Current fixed-center/query-scale controls are distinct because fixed nonzero centers divided by query-dependent scales can still produce a dynamic offset.

## Reporting

No preferred sign, monotonic gain, positive interaction, causal dataset explanation or publication-score improvement is assumed. Publish adverse/zero outcomes alongside favorable ones. Do not rewrite the accepted frozen numerical archives. Keep new analysis artifacts separate from the journal manuscript until findings and scope are checked. New natural-shift data acquisition, a readout-by-normalization factorial with fair retuning, and controlled degradation are separate stages, not implied completed by these two diagnostics.
