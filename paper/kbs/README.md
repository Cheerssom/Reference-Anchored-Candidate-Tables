# Retention-constrained class expansion — KBS manuscript

Current target: **Knowledge-Based Systems**, original Research Article.
The earlier FCS manuscript remains under `../fcs/` as a historical venue version.
Neither version has been submitted to a journal by this workflow.

Start with `main.pdf`, `supplementary.pdf`, `Highlights.docx` and
`cover_letter.docx`. The main text retains the two research questions:
novel recognition under a common base-accuracy constraint, and the paired value
of the retained classifier across successive arrivals.

The current revision has a 22-page main manuscript and 16-page supplement. It adds the 200-class ImageNet-R task and corrected ViT-transfer FeNeC-M/FeNeC-E comparisons.

Build from this directory:

```bash
pip install pymupdf python-docx
python build_strengthening_results.py
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

`literature/reading_notes.md` records the six added KBS papers and the actual
reading scope. `submission_check.md` records authoritative links and the
remaining limitation in directly accessing the current journal guide.
