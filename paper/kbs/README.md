# Retention-constrained class expansion — KBS manuscript

Current target: **Knowledge-Based Systems**, original Research Article.
The earlier FCS manuscript remains under `../fcs/` as a historical venue version.
Neither version has been submitted to a journal by this workflow.

Start with `main.pdf`, `supplementary.pdf`, `Highlights.docx` and
`cover_letter.docx`. The main text retains the two research questions:
novel recognition under a common base-accuracy constraint, and the paired value
of the retained classifier across successive arrivals.

The current revision has a 26-page main manuscript and 68-page supplement. It adds the 200-class ImageNet-R task and corrected ViT-transfer FeNeC-M/FeNeC-E comparisons.

Build from this directory:

```bash
pip install pymupdf python-docx
python build_depth_results.py
python build_validation_budget_results.py
python build_revision_results.py
python build_independent_risk_results.py
python build_concise_tables.py
python build_calibration_budget_results.py
python build_kbs_presentation.py
python build_reviewer_revision_results.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
python validate_submission.py
python build_submission_materials.py
```

Build the main document first: the supplement imports its equation and
proposition labels using `xr`. A clean extracted supplementary-source archive
contains `main.aux` for standalone compilation. All submission-source files are
at one directory level for Editorial Manager.

The original numerical macros and legacy tables remain byte-identical to the checked FCS retention-focus revision. New KBS tables and plots are generated from `experiments/kbs_strengthening/`, using the corrected `fenec_vit/` records. Initial probes remain separate. `experiments/kbs_strengthening/README.md` describes the real pause/resume test, bounded memory and serial GPU runner. The source protocols, correction ledger and source-hashed margin archives preserve the exact comparison conditions.

`literature/reading_notes.md` records the seven added KBS papers and the actual
reading scope. `submission_check.md` records authoritative links and the
remaining limitation in directly accessing the current journal guide.

The depth extension adds dense summed/class-mean Tip-Adapter readouts, 81 shot/calibration configurations, 648 append diagnostics and optimized readout measurements. See `depth_report.md`. Local archive checks pass; the fresh-agent claim-audit invocation failed on a model usage limit, and its ERROR record is preserved. It is not a completed external review.

The threshold-label extension evaluates nominal 25/50/100% base-validation query budgets with frozen fusion settings. Supplementary Section H distinguishes validation feasibility from attained test retention, reports actual query counts, and keeps all six readouts and both targets. See `validation_budget_report.md`.

The primary comparison now has ten class partitions and two frozen CLIP architectures. Validation-selected strong-baseline results are in the main text, with conditional paired bootstrap intervals and raw/fixed-temperature controls. Earlier three-partition studies remain explicitly scoped in Supplementary A–H; Section I contains the expanded comparison. See `revision_evidence_report.md`. The bookmarked `review_complete.pdf` combines the main and full supplement for review.

## User-supplied Elsevier template

The manuscript and supplement use elsarticle 3.4 (2024/04/04), generated from the supplied Elsarticle bundle with `latex elsarticle.ins`. Both use the numeric starter, `preprint,review,12pt`, class-default margins and review line spacing. The template archive hash is recorded in `template_provenance.json`; editable releases include the class source, installer, starter and license notice. Main: 22 pages; supplement: 54 pages; combined review PDF: 76 pages. Template conversion and prose compression preserve all numerical result tables.

## Independent risk calibration: original allocation

Section J and `independent_risk_report.md` describe 60 refitted partition runs with disjoint fitting, tuning and risk-calibration indices. The 21,600 metric records and 16,404 threshold tests reconstruct independently. At 95% trajectory confidence and 95% retention, ImageNet-R admits nontrivial expansion in all ten partitions with both encoders (novel accuracy 77.64% L/14, 45.30% B/32). CIFAR and Pets select the explicit base-only fallback, with zero novel recognition. The sufficient protection bound is conditional on the stated sampling assumptions, and conservative for replacement readouts that recover reference errors. It is not a guarantee under distribution shift or on every finite test set.

Run `scripts/run_kbs_independent_risk.ps1`, then `python src/experiments/kbs_verify_independent_risk.py` and `python paper/kbs/build_independent_risk_results.py`. The shared PAUSE marker is respected. A real rerun preserved hashes and modification times of all 1,320 completed checkpoint files. New experiment storage is approximately 184.5 MiB.

## Concise narrative revision

The main text now has 22 review-format pages and a 114-word abstract. It centers on retained discrimination as class-expansion capacity, with all original numerical artifacts preserved. Full final-session comparisons and normalization controls are in Supplement I, independent risk calibration in J and implementation detail in K. See `concision_report.md`.

## Independent calibration coverage (current main evidence)

The frozen-model coverage experiment and all six readouts appear in main Figure 2,
Table 2 and Supplement L. It supersedes the original allocation as the primary
operational result while retaining every original record in Section J. At a cap
of 40 images per base class (availability-truncated for ImageNet-R), all six
dataset/encoder conditions certify expansion in all ten final-session partitions
and attain the 95% test-retention target. The necessary zero-loss effective
count is 89.78; this explains original CIFAR/Pets fallback under the bound.

See `calibration_budget_report.md`, `calibration_budget_summary.json/CSV` and
`calibration_budget_paired.csv`. The last distinguishes policy gains including
fallback from differences on jointly admitted partitions. A full reconstruction
checks 93,600 results and 88,216 threshold tests; real resume preserves 1,996
checkpoint hashes and timestamps. Storage for this extension is about 198.9 MiB.

Run `scripts/run_kbs_calibration_budget.ps1`, then the verifier and builder:

```bash
python src/experiments/kbs_verify_calibration_budget.py
python src/experiments/test_kbs_calibration_budget.py
python paper/kbs/build_calibration_budget_results.py
```

The main has 22 pages; Supplement A-L has 54 pages and the combined PDF 76.
Population confidence is per method/order/partition/budget under the stated
class-conditional assumptions. Pets includes early-session fallback on three
L/14 and four B/32 paths out of twenty; all final retained-table paths admit.

## Main-figure and table revision (2026-10-23)

Current main:22 review-format pages,3 figures,2 tables,114-word abstract and10
KBS references. The shared-retention capacity curve is restored to the main text.
Its L/14 y-axis is50-100% and B/32 is30-90%; error bars stop at panel limits.
The supplement has54 pages; the combined review PDF has76.

`build_kbs_presentation.py` generates grouped method-by-dataset paired gains,
calibration cost/admission/accuracy columns and complete baseline policy panels.
All18 paired comparisons,6 primary calibration conditions and36 baseline rows
are preserved. Table2 adds the existing partition SDs. See
`presentation_revision_report.md` and `literature/reading_notes.md` for the
three added citations, inspected author-version tables and access scope.
The new citations are related work, not unrun experimental baselines.

## First-page abstract (2026-10-23)

The current abstract has106 words, covering the mechanism, direct calibration comparison and query-shift scope. Both the complete abstract and keywords fit on page1 using the existing class fonts, margins and review spacing. Validator checks their first-page placement.

## Five-point reviewer revision

See [revision response](five_point_revision_response.md), the new summary and Supplement M-O. All24,000 new records and54,960 threshold tests reconstruct. The direct-risk proof has a conditional same-family review; the completed experiment audit has a same-family/provisional PASS, with no material issue and one corrected fixture-label wording. The previous interrupted run remains in private history; the public status receipt is experiment_audit_status.json. New data/model/PT feature files are not redistributed.

For fresh HyCal feature computation, obtain the upstream author code separately (not bundled) before running the fixture:

```powershell
git clone https://github.com/EJLEE5826/HyCal-CIL paper/kbs/literature/reviewer_baselines/hycal
git -C paper/kbs/literature/reviewer_baselines/hycal checkout 4558fb7fa6aa389310c9e9dc7db7c2cbebc84a41
powershell -File scripts/run_kbs_reviewer_revision.ps1 -Python python
```

Run those commands from the repository root after constructing the original dataset/feature caches. Reproduction on new software/hardware can differ in floating-point details; pinned source hashes identify the recorded run. Readout manifests bind source/protocol; the additional provenance file is explicitly a post-run binding, with cache numerical witnesses and raw-pixel separation checks. Completed session shards and feature batches support resume; PAUSE uses experiments/kbs_strengthening/PAUSE. No additional model download or paid compute was needed for the original local run.
