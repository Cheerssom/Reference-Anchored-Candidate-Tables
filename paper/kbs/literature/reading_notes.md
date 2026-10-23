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

## Reviewer-requested positioning: BiMC and HyCal

Primary BiMC metadata verified from the CVF CVPR2025 page (pp.9881–9890); full paper read from the author's mirror https://www.tianyuding.com/papers/bimc-cvpr25.pdf , especially Sections3.2–3.3. It uses descriptive text enrichment, base-guided prototype calibration, multimodal fusion, evolving global covariance and masked inference.

HyCal full text read at https://arxiv.org/html/2604.15678v2 ; author implementation https://github.com/EJLEE5826/HyCal-CIL and university announcement confirm the work's CVPR2026 context. The bibliographic entry deliberately cites the accessible arXiv version instead of inventing proceedings pages. Sections4.3–4.4 define fused prototype summaries and hybrid cosine/covariance scoring; its cross-discipline variable-shot benchmark differs from our fixed-pool class partitions.

Both are related work, not executed baselines in this revision. No published numbers are mixed with the controlled readout tables.

## Additional KBS positioning and table exemplars (2026-10-23)

Three further related-work references were verified against publisher-indexed
records and Crossref. Raw Crossref responses are preserved in
`recent_kbs_crossref.json`; author order, title, journal, volume, year and
article number match the new BibTeX records. The manuscript now cites ten KBS
papers. These additions position existing representation-adaptation approaches;
no unrun experimental results are attributed to them.

| Added paper | Primary record | Reading scope and use |
|---|---|---|
| Li, Wu and Ji, MoTE, KBS324(2025)113795 | https://doi.org/10.1016/j.knosys.2025.113795 | Publisher abstract and full author version https://arxiv.org/html/2506.11038 read; expert filtering and joint inference distinguish representation adaptation from fixed-readout admission. |
| Dong et al., Dynamic expansion orthogonal network, KBS341(2026)115868 | https://doi.org/10.1016/j.knosys.2026.115868 | Publisher-indexed abstract/highlights, Crossref metadata; orthogonal feature-space constraints position architectural expansion. No full-text access or table inspection claimed for this paper. |
| Yuan et al., Counterfactual distribution intervention, KBS351(2026)116754 | https://doi.org/10.1016/j.knosys.2026.116754 | Publisher-indexed abstract/highlights and Crossref metadata; representation invariance through counterfactual intervention. No full-text access or table inspection claimed. |

### Actual table-layout inspection

Publisher article requests returned403. Lawful publicly posted author versions
of three papers were inspected as PDF pages, not inferred from abstracts. Their
layout is used as an exemplar, not represented as an official KBS requirement
or a verified final typeset version. Local PDFs and screenshots are private
reading assets under `table_exemplars/`, excluded from public export.

- MoTE, https://arxiv.org/pdf/2506.11038, PDFp21 Table1 and HTMLTable3: dataset/protocol column groups, paired metric subheaders with direction arrows, uncertainty alongside means, boldface conventions explained in the caption; ablation table separate.
- FeNeC, https://arxiv.org/pdf/2503.14301v2, PDFp8 Tables3-4: compact method-by-dataset matrices, average/final metrics separated into panels, means with uncertainty and special-case markers defined in captions.
- REAL, https://arxiv.org/pdf/2403.13522, PDFp8 Table1: backbone groups in rows, metric and phase-count groups in columns, explicitly scoped bold/underline conventions.

Applied to this manuscript: Table1 is a method-by-dataset matrix split into
encoder panels, retaining all18 paired gains and CIs; bold estimates mean the
conditional interval excludes zero, not a best-method ranking. Table2 separates
calibration cost, finite admission, test attainment and base/novel accuracy,
adding existing partition SDs. Supplement L groups empirical/certified policies
and bolds the largest displayed certified novel mean, with the rule stated.
No performance values are imported from any exemplar. Avoiding excessive columns
keeps our tables readable without reproducing tiny-font layouts.

## 2026-10-23 reviewer revision

HyCal CVPR2026 official proceedings: https://openaccess.thecvf.com/content/CVPR2026/html/Lee_HyCal_A_Training-Free_Prototype_Calibration_Method_for_Cross-Discipline_Few-Shot_Class-Incremental_CVPR_2026_paper.html , pp29462–29471. Author repository https://github.com/EJLEE5826/HyCal-CIL commit4558fb7fa6aa389310c9e9dc7db7c2cbebc84a41. Read author method/implementation; shared eight-prompt anchors and L/14/B/32 are controlled-transfer choices, rather than the published B/16/single-prompt setting. Fixtures and scope in Supplement M. No third-party full text or code is redistributed.

BiMC CVPR2025: https://www.tianyuding.com/papers/bimc-cvpr25.pdf ; https://github.com/yychen016/BiMC commitb4ef2ea01b268098631f75ace4e716daff75db6b. Inspected for comparison relevance; not implemented.

Waudby-Smith and Ramdas (2024), Estimating means of bounded random variables by betting, JRSSB86(1),1–27, DOI10.1093/jrsssb/qkad009: https://academic.oup.com/jrsssb/article/86/1/1/7043257 . Citation supplies betting context; our unequal-stratum fixed-sample proof is given self-contained and is not attributed as an existing theorem from this paper.
