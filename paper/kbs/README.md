# Retention-constrained class expansion — KBS manuscript

Current target: **Knowledge-Based Systems**, original Research Article.
The earlier FCS manuscript remains under `../fcs/` as a historical venue version.
Neither version has been submitted to a journal by this workflow.

Start with `main.pdf`, `supplementary.pdf`, `Highlights.docx` and
`cover_letter.docx`. The main text retains the two research questions:
novel recognition under a common base-accuracy constraint, and the paired value
of the retained classifier across successive arrivals.

The current revision has a 34-page main manuscript and 42-page supplement. It adds the 200-class ImageNet-R task and corrected ViT-transfer FeNeC-M/FeNeC-E comparisons.

Build from this directory:

```bash
pip install pymupdf python-docx
python build_depth_results.py
python build_validation_budget_results.py
python build_revision_results.py
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

The manuscript and supplement use elsarticle 3.4 (2024/04/04), generated from the supplied Elsarticle bundle with `latex elsarticle.ins`. Both use the numeric starter, `preprint,review,12pt`, class-default margins and review line spacing. The template archive hash is recorded in `template_provenance.json`; editable releases include the class source, installer, starter and license notice. Main: 34 pages; supplement: 42 pages; combined review PDF: 76 pages. Template conversion and prose compression preserve all numerical result tables.

## Independent risk calibration

Section J and `independent_risk_report.md` describe 60 refitted partition runs with disjoint fitting, tuning and risk-calibration indices. The 21,600 metric records and 16,404 threshold tests reconstruct independently. At 95% trajectory confidence and 95% retention, ImageNet-R admits nontrivial expansion in all ten partitions with both encoders (novel accuracy 77.64% L/14, 45.30% B/32). CIFAR and Pets select the explicit base-only fallback, with zero novel recognition. The sufficient protection bound is conditional on the stated sampling assumptions, and conservative for replacement readouts that recover reference errors. It is not a guarantee under distribution shift or on every finite test set.

Run `scripts/run_kbs_independent_risk.ps1`, then `python src/experiments/kbs_verify_independent_risk.py` and `python paper/kbs/build_independent_risk_results.py`. The shared PAUSE marker is respected. A real rerun preserved hashes and modification times of all 1,320 completed checkpoint files. New experiment storage is approximately 184.5 MiB.
