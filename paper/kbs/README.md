# Retention-constrained class expansion — KBS manuscript

Current target: **Knowledge-Based Systems**, original Research Article.
The earlier FCS manuscript remains under `../fcs/` as a historical venue version.
Neither version has been submitted to a journal by this workflow.

Start with `main.pdf`, `supplementary.pdf`, `Highlights.docx` and
`cover_letter.docx`. The main text retains the two research questions:
novel recognition under a common base-accuracy constraint, and the paired value
of the retained classifier across successive arrivals.

Build from this directory:

```bash
pip install pymupdf python-docx
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
python validate_submission.py
python build_submission_materials.py
```

Build the main document first: the supplement imports its equation and
proposition labels using `xr`. A clean extracted supplementary-source archive
contains `main.aux` for standalone compilation. All submission-source files are
at one directory level for Editorial Manager.

The numerical macros and table inputs are byte-identical copies of the checked
FCS retention-focus revision. To regenerate results, use the existing repository
scripts under `paper/fcs/` and `src/experiments/`; no experiment protocol or raw
result was changed for this venue migration. The figure files were copied here
without changing their contents; the architecture file is named `framework.pdf`.

`literature/reading_notes.md` records the six added KBS papers and the actual
reading scope. `submission_check.md` records authoritative links and the
remaining limitation in directly accessing the current journal guide.
