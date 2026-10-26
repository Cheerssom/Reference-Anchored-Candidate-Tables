# KBS submission preparation — frozen manuscript

Checked on 2026-10-26. Target: Knowledge-Based Systems, original Research Article. No journal submission has occurred.

## Current files

Main manuscript: 30 pages in the official elsarticle preprint/review/12pt layout, including references and declarations; 5 figures and 3 tables. Supplement: 87 pages, Sections A–R. The 120-word abstract and six keywords fit on page 1. Five editable Highlights contain 77, 66, 75, 79 and 71 characters, respectively.

The three final clarifications distinguish original frozen readouts from independently retuned fusion in the abstract, specify within-class accuracy matching and shared calibration counts for the stratified variance comparison, and attribute the statistical constructions immediately before the theorems. The cover letter uses the same experiment distinction. The presentation cleanup unifies Figure 1 labels with the asymmetric-readout terminology, merges the frontier and alignment subsections, uses upright encoder names in Tables 1 and 2, and adds representative-method citations to the introduction. Experimental values, figure data, architecture layout and matched-capacity plot ranges are unchanged.

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
| Length | Recorded; current official limit not directly verified | The same transcription states a preference for no more than 20 double-line-spaced pages including tables/figures. This 30-page review-format manuscript must not be described as meeting that preference or a verified hard limit. |
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

submission_freeze_manifest.json identifies the exact deliverable and source hashes. Run `python paper/kbs/verify_submission_freeze.py` to verify them without changing files. The local submission_freeze_20261026_presentation/files folder preserves the delivery files. Any later manuscript edit requires an explicitly refreshed freeze manifest and submission bundle.
