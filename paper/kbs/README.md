# Risk-certified class expansion through reference-anchored evidence alignment

Target journal: Knowledge-Based Systems. No journal submission has been performed.

Current review-format manuscript:30 pages,5 figures and3 tables. Supplement:87 pages, Sections A–R. The 120-word abstract and keywords fit on the first page. Read review_complete.pdf for the full paper and supplement; main.pdf and supplementary.pdf are separate upload files.

The paper connects vocabulary-invariant alignment of asymmetric evidence with finite-sample retention certification. Its central analysis relates adaptive admission margins, preserved error structure and the gap between recognition capacity and selected policies. The main text includes the ridge mean identity, protection coverage condition, direct accuracy test, paired-score variance identity and empirical policy decomposition. Descriptive test-label capacity is separate from independently selected policies. Confidence is conditional on the stated class-conditional sampling assumptions, per fixed configuration over five prespecified arrivals; bootstrap intervals describe class-assignment variation on fixed images.

The paired-score identities are checked without retraining by `python paper/kbs/verify_error_structure.py`:8855 joint-probability checks,480 saved policy reconstructions and24800 class-conditional checks. At matched reference and expanded accuracies, the variance contribution of recovery is exactly `2*r*G`. This conditional identity does not assert a universal ordering of calibration power. The original experimental results are unchanged.

## New controlled evidence

- Section Q independently selects both fusion coefficients and a tuning offset for six readout/normalization schemes under equal3636-setting budgets.360 searches produce28800 metrics and26755 reconstructed calibration tests. This retunes fusion, with ridge fits and encoder held fixed.1338 checkpoint hashes/timestamps survive a real restart.
- Section R transfers original frozen source-selected readouts to6445 deduplicated Sketch images (2492 calibration,3953 test), excluding3624 source-byte overlaps and83 other target duplicates/equivalents.2700 metrics and4899 calibration tests reconstruct;1049 feature/evaluation checkpoints survive restart. Target labels do not select source fusion parameters, prototypes, model fits or candidate families. Exact hashes do not eliminate every possible near-duplicate dependency or establish pretraining novelty.
- See full_fusion_retuning/report.md and sketch_validation/report.md. Numerical reconstruction is distinct from independent semantic peer review; no new completed fresh semantic-audit verdict is claimed.

## Rebuild the manuscript

The saved tables and figures suffice to compile the paper without retraining. From the project root:

```powershell
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/kbs/main.tex
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/kbs/supplementary.tex
python paper/kbs/validate_submission.py
python paper/kbs/build_submission_materials.py
python paper/kbs/build_release.py
```

Compile main first because the supplement imports its labels. The source ZIPs use a flat directory; the supplement-only archive includes main.aux. build_release.py also checks compilation from clean-extracted source archives.

## Resume the experiments

```powershell
powershell -File scripts/run_kbs_full_fusion_retuning.ps1 -Python python
powershell -File scripts/run_kbs_sketch_validation.ps1 -Python python
```

These reuse the original cached source features, local CLIP snapshots and source selection records. Reconstruct these inputs with the repository's original data/feature instructions. The runtime serializes GPU jobs, uses two PyTorch CPU threads, caps process GPU memory and honors experiments/kbs_strengthening/PAUSE. Each feature batch and per-session result is atomic. Remove the PAUSE marker to resume the same invocation. ImageNet-Sketch acquisition uses a pinned author-linked ZIP index and requests only mapped class ranges; images live separately under ../data/imagenet-sketch-r200. Images, pretrained weights and PT feature caches are excluded from public releases.

The source-hashed old evidence ZIP and the separate Q/R extension evidence ZIP preserve their respective scopes. Highlights, cover letter, declarations and title page are regenerated with the submission-materials builder. Literature/template provenance remains under literature/ and template_provenance.json.

## Submission freeze — 2026-10-26

The final cohort/conditioning/attribution clarifications and presentation cleanup are incorporated. Figure 1 uses asymmetric-readout terminology; Section 4.2 combines retention frontiers with query-adaptive alignment; Tables 1 and 2 use upright encoder names; the introduction connects the task to SimpleCIL, RanPAC, BiMC and HyCal. `submission_freeze_manifest.json` records the frozen artifacts; `python paper/kbs/verify_submission_freeze.py` checks them without rebuilding. See `submission_check.md` for the upload map and the distinct status of local checks versus journal-specific rules. Current direct access to the KBS guide failed, so anonymity and length are not marked as officially verified. No journal submission has occurred.
