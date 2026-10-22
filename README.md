# Reference-Anchored Candidate Tables

Research code and manuscript for **Retention-constrained class expansion with
reference-anchored candidate tables**.

**Junyi Zhang, Shandong University**  
Contact: cheerssom0876@gmail.com

This repository accompanies a research manuscript.

Current submission target: **Knowledge-Based Systems (KBS)**.
The current manuscript is [paper/kbs/main.pdf](paper/kbs/main.pdf), with
[supplementary material](paper/kbs/supplementary.pdf) and
[editable Highlights](paper/kbs/Highlights.docx).
The earlier `paper/fcs/` version is retained as a historical venue draft.
No journal submission has been performed by this workflow.

## Method

Each class carries the evidence available to it: a text anchor, a base-trained
classifier column, or a few-shot visual prototype. Fixed base reference banks
normalize each branch independently of the growing candidate vocabulary.
Appending a row preserves existing scores in real arithmetic under fixed
evidence, encoder, references and gates. New classes compete through new scores.

![Architecture](paper/fcs/figures/fig1_framework_fcs.png)

- [Manuscript PDF](paper/fcs/main.pdf)
- [Editable architecture](paper/fcs/figures/fig1_framework_fcs.drawio)
- [Three-slide highlights](paper/fcs/highlights_retention.pptx)

## Matched-retention novel recognition

Frozen CLIP ViT-L/14; mean accuracy percentages across three distinct class
partitions. Complete partition sample standard deviations and all baseline
results are in the manuscript and CSV files.

| Dataset / common budget | Readout | Base | Novel |
|---|---|---:|---:|
| CIFAR-100 / 95% | Retained classifier | 83.48 | 75.79 |
| CIFAR-100 / 95% | Learned branch removed, recalibrated | 83.48 | 61.36 |
| CIFAR-100 / 95% | Anchored text + prototypes | 83.48 | 74.43 |
| Oxford-IIIT Pets / 99% | Retained classifier | 96.84 | 90.99 |
| Oxford-IIIT Pets / 99% | Anchored text + prototypes | 96.85 | 84.00 |

Budgets are fractions of the SAME retained model's closed-base accuracy within
each partition. These are descriptive matched-test-retention frontiers: branch
mixtures use base-only calibration, and the subsequent threshold sweep aligns
the base test requirement. A separate base-validation-only threshold policy
reports deployable operating points and their actual achieved test accuracies.
All five budgets (80/85/90/95/99%), feasible partition counts, paired sample SD
and comparisons are retained in the records. Maximum discrete overshoot is
0.023 pp on CIFAR and 0.076 pp on Pets.

## Value across five class arrivals

With gates and offsets fixed after base-only calibration, CIFAR average
incremental accuracy is 82.02% with retained columns and 79.84% after removal and
recalibration. The paired gain is **2.18 ± 0.24 pp**; final accuracy is
79.31% versus 77.56%, a **1.75 ± 0.23 pp** paired gain. The full comparison also
includes fixed-gate removal, dense base prototype replacement, and Pets.
Two arrival orders are averaged within each of three partitions before SD.

![Matched-retention recognition](paper/fcs/figures/matched_retention.png)
![Sequential classifier value](paper/fcs/figures/sequential_value.png)

## Install and prepare public assets

Python 3.10+ is recommended. The exact locally tested package versions are in
`tested_environment.json`; install a PyTorch build appropriate for your machine.

```bash
python -m pip install -r requirements.txt
python -c "from torchvision.datasets import CIFAR100; CIFAR100(root='data', train=True, download=True); CIFAR100(root='data', train=False, download=True)"
python -c "from transformers import CLIPModel, CLIPTokenizer; CLIPModel.from_pretrained('openai/clip-vit-large-patch14'); CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')"
```

Run commands from the repository root. The default data directory is `data/`;
override it with `JLT_DATA` if necessary. Oxford-IIIT Pets is downloaded from its
official source by the loader. Dataset images and model weights are not stored
in this repository. Checkpoints must be downloaded before the Pets loader,
which uses the local Hugging Face cache. Feature extraction is GPU-intensive;
the recorded experiments used an RTX 4070 Laptop GPU.

## Protocol checks and experiments

```bash
python src/experiments/test_fcs_protocol.py
python src/experiments/test_fcs_retention.py
python src/experiments/fcs_expansion.py
python src/experiments/fcs_expansion.py --dataset pets
python src/experiments/fcs_review_experiments.py --dataset cifar
python src/experiments/fcs_review_experiments.py --dataset pets
python src/experiments/fcs_retention_value.py --dataset cifar
python src/experiments/fcs_retention_value.py --dataset pets
```

Running experiments overwrites their result directories. Preserve a checkout of
the released JSON/snapshots to compare a rerun. If the CIFAR frozen cache is
absent, the primary runner builds it from frozen CLIP without a LoRA checkpoint.
Run each primary dataset experiment before its supplemental controls.

Base classifier fitting, ridge selection and gate calibration use base classes
only. Actual novel classes supply their names and reserved support images.
At five shots all five reserved images are used; support seeds are not
independent repetitions. Statistics average subsets within each class partition
and use the three distinct partitions as independent units.

`experiments/fcs*/review_controls/` contains offset-calibrated controls, 95%/99%
validation retention targets, five-session arrival tests and a frozen
2000-dimensional RanPAC-style analytic adaptation. The latter uses base
validation rather than original within-task MSE selection; it is not a full
reproduction of published RanPAC. Validation targets do not constrain test
retention. Infeasible validation partitions are explicitly flagged.

`experiments/fcs*/retention_value/` contains the main common-budget and paired
sequential comparisons: 1320 records per dataset, byte-exact run source, and
30 compact margin archives per dataset. The locked protocol is in
`experiments/fcs_retention_value_protocol.md`. Verify the records without data,
feature caches or a GPU:

```bash
python src/experiments/fcs_verify_retention.py
```

## Regenerate tables and compile the paper

Regenerate directly from the released records without downloading any data:

```bash
python paper/fcs/build_results.py
python paper/fcs/build_review_results.py
python paper/fcs/build_retention_value.py
cd paper/fcs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The original run source snapshots and hashes remain in the result directories.
The public copy of `src/config.py` replaces machine-specific default paths with
paths relative to this checkout; experimental source snapshots are unchanged.
Supporting legacy modules are included because the runner imports their dataset
split helper; the commands above are the FCS evaluation entry points.

## Availability

Datasets and pretrained weights retain their original licenses. The repository
contains implementation, manuscript, editable figure, numeric records and small
prediction archives. Downloaded third-party papers, local model/feature caches,
and internal editorial reports are excluded.

## Expanded KBS evidence

The current revision adds a 200-class ImageNet-R task, recent FeNeC-M/FeNeC-E
readouts and source-hashed prediction margins. The corrected FeNeC transfer uses
the author's ViT single-shrinkage setting and broader representative budget;
initial probes remain in the records with a separate correction ledger.

On ImageNet-R, retaining the classifier adds 2.22 +/- 0.40 percentage points in
average incremental accuracy and 1.75 +/- 0.48 points in final accuracy over
the recalibrated removal control. Matched-retention analysis and validation-only
threshold selection remain separate.

See `experiments/kbs_strengthening/README.md` for pause/resume and resource
controls. The PowerShell runner defaults to `python` on PATH; use `-Python`
to choose an installed environment. Dataset images and all `.pt` model/feature
caches are obtained locally and are not distributed in this repository.

```bash
python src/experiments/kbs_verify_strengthening.py
python src/experiments/test_kbs_strengthening.py
python paper/kbs/build_strengthening_results.py
```

## Calibration and dense-cache extension

The current manuscript includes summed and class-mean Tip-Adapter comparisons,
81 shot/calibration configurations, 648 append diagnostics per normalization
variant, and measured readout costs on the same laptop. Trials/orders are
averaged within each of three class partitions; 81 is not an independent seed count.

On ImageNet-R, the retained readout gives 84.48 +/- 0.72% novel accuracy at the
95% shared retention budget versus 81.58 +/- 1.72% for class-mean Tip-Adapter.
Its paired average incremental gain over that cache is 1.48 +/- 0.35 percentage
points. Across repeated five-shot calibration, the paired average gain over
recalibrated removal is 2.14 +/- 0.37 points. All outcomes and operating scopes
are retained. See [the completion report](paper/kbs/depth_report.md).

```bash
python src/experiments/test_kbs_depth.py
python src/experiments/kbs_verify_depth.py
python src/experiments/kbs_depth_cost.py
python paper/kbs/build_depth_results.py
```

## Admission-threshold validation-query budgets

The additional CPU-only study fixes all original five-shot readouts and fusion
settings, then varies base-validation query subsets at nominal 25/50/100%
budgets for 95/99% retention targets. Its 315 saved session units contain
3,780 metric records. Full-validation outputs reproduce the original policy
exactly; reduced-budget repeats are averaged inside each original class
partition. Validation feasibility and attained test retention are reported
separately. See [the protocol and report](paper/kbs/validation_budget_report.md).

```bash
python src/experiments/kbs_validation_budget.py
python src/experiments/kbs_verify_validation_budget.py
python paper/kbs/build_validation_budget_results.py
```
