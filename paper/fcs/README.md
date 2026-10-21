# Manuscript sources

`main.pdf` is the compiled research manuscript. `main.tex`, `references.bib`,
the official FCS class/style, generated tables and vector figures are provided.
The manuscript is not a published or accepted article.

See the repository-root README for feature extraction, experiments and table
generation. Compile here with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`figures/fig1_framework_fcs.drawio` is the editable architecture figure.
