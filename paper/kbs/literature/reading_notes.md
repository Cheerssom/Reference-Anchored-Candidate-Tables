# KBS references added for the venue revision

Checked: 2026-10-22. Reading scope: publisher-indexed abstracts, highlights and
available introduction excerpts; this is not a claim of full-text access to all
six papers. Citation statements below stay within those accessible passages.
Complete author lists, issue years, volumes, article numbers and DOIs are in
`added_kbs_references.json`. The first five records were also returned by
Crossref; a console encoding error interrupted saving the raw response. A later
bulk retry was rate-limited, so no raw Crossref-response artifact is claimed.

| Paper | Publisher source | Relevant contribution | Use in this manuscript |
|---|---|---|---|
| Xu et al., MFS3, KBS 265 (2023) 110394 | https://www.sciencedirect.com/science/article/abs/pii/S0950705123001442 | Combines base and incremental feature spaces to refine similarity-based decisions. | Introduction and related work: situates heterogeneous stored/new evidence. |
| Chen et al., SAGG, KBS 299 (2024) 112030 | https://www.sciencedirect.com/science/article/pii/S0950705124006646 | Uses sharpness-aware optimization and prototype calibration for FSCIL. | Introduction and related work: distinguishes adaptation from retained-classifier value at a common accuracy constraint. |
| Han et al., Adaptive exploration, KBS 330 (2025) 114496 | https://www.sciencedirect.com/science/article/abs/pii/S0950705125015357 | Reward-guided adaptation balances retention and acquisition across incremental tasks. | Related work: identifies the alternative of updating the classifier through task adaptation. |
| He et al., REAL, KBS 332 (2026) 114901 | https://www.sciencedirect.com/science/article/pii/S0950705125019392 | Combines complementary base representations and feature fusion for analytic incremental learning. | Related work: connects the retained evidence frame to analytic readouts without asserting equivalent training protocols. |
| Fang et al., BPC-CLIP, KBS 342 (2026) 115862 | https://www.sciencedirect.com/science/article/pii/S0950705126005885 | Uses attribute-enriched text, visual/text prompt fusion and prompt selection for FSCIL. | Introduction and related work: relates multimodal evidence fusion to the fixed-reference score frame. |
| Yan et al., CLIP-guided discovery, KBS 310 (2025) 112920 | https://www.sciencedirect.com/science/article/abs/pii/S0950705124015545 | Transfers CLIP information through semi-supervised prototype adaptation for continual novel-class discovery. | Related work explicitly distinguishes unlabeled discovery from named classes with labeled support. |

These are related-work citations. No experimental performance claim or direct
baseline result is attributed to an implementation that was not run. No numerical
results were imported from these papers into the experiment tables.

The publication year follows the issue record, not the year embedded in the DOI:
REAL is a 2026 article with a 2025 DOI; CLIP-guided discovery is a 2025 article
with a 2024 DOI. DELTA, whose issue is dated 2026-10-25, was not needed for the
six-paper addition and is not added.

Editorial lesson applied: introduce the decision problem and concrete gain early;
name the evidence sources and their contribution; distinguish a retained-readout
analysis from representation adaptation. Existing experimental conditions and
descriptive-frontier qualifications remain visible.

## Experimental baseline extension (2026-10-22)

FeNeC, KBS 338 (2026) 115479, DOI 10.1016/j.knosys.2026.115479, is now also an experimental baseline. Publisher record: https://www.sciencedirect.com/science/article/pii/S0950705126002224 . The full accessible author preprint https://arxiv.org/html/2503.14301v2 and pinned implementation commit 1575ea019fd583149c47c068f7f5ccc102ea4580 were inspected. Its ViT setting uses single covariance shrinkage; the initial double-shrinkage probe is preserved separately. The corrected transfer includes K=25 representatives and published ViT gamma pairs, selected with the same base-only pseudo episode. The source and correction ledger explain the frozen-CLIP/five-shot adaptations and sparse-vote score interface. Native and offset-calibrated Mahalanobis and Euclidean forms are reported. Author-output fixtures verify covariance, clustering and voting without bundling unlicensed upstream code into the public repository.
