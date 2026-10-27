# KBS submission preparation — frozen manuscript

Checked on 2026-10-27. Target: Knowledge-Based Systems, original Research Article. No journal submission has occurred.

## Current files

Main manuscript: 29 pages in the official elsarticle preprint/review/12pt layout, including references and declarations; 5 figures and 3 tables. Supplement: 88 pages, Sections A–R. The 123-word abstract and six keywords fit on page 1. Five editable Highlights contain 81, 77, 63, 76 and 81 characters, respectively.

The revised narrative leads with asymmetric evidence alignment, query-adaptive admission and recognition across successive arrivals. Three main tables present matched readout recognition, controlled query-centering ablation and admission under the common direct accuracy test. Stability and Sketch-transfer figures move into the main paper. The abstract and Highlights use the matched HyCal comparison and separately retuned normalization results. Statistical attribution remains adjacent to the test statements; conditional variance analysis and detailed fairness results remain in the supplement. New presentation values are reaggregated from frozen prediction archives; no model outputs or experimental inputs are changed.

## Checks and rule sources

| Item | Status | Basis |
|---|---|---|
| Main and supplementary compilation | Passed | No undefined references, overfull boxes or unresolved citations; embedded fonts checked. |
| Clean extracted source compilation | Passed | Main and supplement compile from the flat source archive; supplement also compiles from its separate archive. |
| Abstract and keywords on page 1 | Passed | Extracted PDF text and visual check. |
| Highlights | Passed | Five bullets, each within 85 characters; editable DOCX. Official guidance: https://www.elsevier.support/publishing/answer/how-do-i-include-highlights-with-my-manuscript |
| LaTeX package | Passed | Flat archives with figure/table dependencies, BibTeX and class/style files. Official instructions: https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions |
| Author and declarations | Consistent | Junyi Zhang; Shandong University, China; cheerssom0876@gmail.com; no funding or competing interests; CRediT, data availability and AI assistance supplied. |
| KBS-specific author guide | Direct verification unavailable | ScienceDirect and the Elsevier guide URL returned 403; browser retrieval timed out. Official guide: https://www.sciencedirect.com/journal/knowledge-based-systems/publish/guide-for-authors |
| Review anonymity | Prepared with author information; current official rule not directly verified | A March 2026 third-party transcription of the KBS guide states single-anonymized review. The existing author-bearing manuscript and separate title page are retained. Confirm the portal's current instruction before upload. |
| Length | Recorded; current official limit not directly verified | The same transcription states a preference for no more than 20 double-line-spaced pages including tables/figures. This 29-page review-format manuscript must not be described as meeting that preference or a verified hard limit. |
| Address fields | Author-provided institution and country included | Department, campus, city, postcode and street address were not supplied. Enter the actual corresponding-author address if required by the submission form. |

The third-party transcription is provenance only, not authoritative confirmation of the current KBS requirements: https://www.studypool.com/discuss/82267917/scopus-article-writing-knowledge-system-journal . Local file checks are complete; anonymity, length and any additional affiliation fields require checking against the live journal submission instructions.

## Upload map

| Purpose | File |
|---|---|
| Main manuscript | main.pdf |
| Supplementary material | supplementary.pdf |
| Highlights | Highlights.docx |
| Cover letter | cover_letter.docx or cover_letter.md |
| Separate title page, if requested | title_page.docx |
| Declarations, if requested | declarations.docx |
| Editable manuscript source | candidate_tables_kbs_source.zip |
| Editable supplement source, if requested | candidate_tables_kbs_supplement_source.zip |
| Local combined reading copy | review_complete.pdf |

Extract candidate_tables_kbs_submission.zip and upload the individual files required by the portal. The outer delivery ZIP, local checks and combined reading copy are not additional journal attachments unless requested. Verify the portal-generated PDF before final submission.

## Frozen version

submission_freeze_manifest.json identifies the exact deliverable and source hashes. Run `python paper/kbs/verify_submission_freeze.py` to verify them without changing files. The local submission_freeze_20261027_asymmetric/files folder preserves the delivery files. Any later manuscript edit requires an explicitly refreshed freeze manifest and submission bundle.
