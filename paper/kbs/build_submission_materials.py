"""Prepare editable KBS submission attachments and reproducible source archive."""
from pathlib import Path
import hashlib,json,re,zipfile
from docx import Document
from docx.shared import Inches,Pt

H=Path(__file__).resolve().parent
title='Retention-constrained class expansion with reference-anchored candidate tables'
highlights=[
'Novel recognition is measured under a common original-class accuracy constraint.',
'Reference-anchored tables reuse trained classifiers during class expansion.',
'CIFAR novel accuracy is 75.79% at a matched base accuracy of 83.48%.',
'ImageNet-R novel accuracy is 84.48% versus 81.58% for a dense semantic cache.',
'Repeated calibration retains a 2.14-point ImageNet-R gain over classifier removal.'
]
if (H/'revision_summary.json').exists():
 revision=json.loads((H/'revision_summary.json').read_text())
 highlights=[
  'Novel recognition is measured under a common original-class accuracy constraint.',
  'Reference-anchored tables reuse trained classifiers during class expansion.',
  'Two CLIP encoders and ten class partitions test retained classifier value.',
  'ImageNet-R gains over dense prototypes are 0.71 points on L/14 and 0.90 on B/32.',
  'Validation policies report feasibility and achieved test retention separately.'
 ]
assert 3<=len(highlights)<=5 and max(map(len,highlights))<=85
(H/'Highlights.txt').write_text('\n'.join('- '+v for v in highlights)+'\n',encoding='utf-8')
letter=f'''Dear Editors of Knowledge-Based Systems,

Please consider our manuscript, "{title}," as an original Research Article.

An expanding recognition system must incorporate new categories while preserving an explicit accuracy requirement for its existing classes. This paper asks how much novel recognition a retained classifier enables under the same original-class accuracy constraint, and how much predictive value it contributes over successive class arrivals. Reference-anchored candidate tables combine learned class columns, semantic anchors and sparse visual prototypes in a consistent score system.

With frozen CLIP features on CIFAR-100, Oxford-IIIT Pets and ImageNet-R, three class partitions and disjoint five-shot support, the matched-retention analysis gives 75.79% novel accuracy at approximately 83.48% CIFAR base accuracy. Removing the learned columns and independently recalibrating the remaining branches gives 61.36% novel accuracy at the same base constraint. Across five CIFAR arrivals, retained columns add a paired 2.18 +/- 0.24 percentage points in average incremental accuracy over that recalibrated alternative. On the 200-class ImageNet-R extension, the corresponding paired gain is 2.22 +/- 0.40 points, with a 1.75 +/- 0.48-point final-accuracy gain. Dense Tip-Adapter cache readouts and recent FeNeC readouts share the encoder and permitted data. At the ImageNet-R 95% budget, retained columns give 84.48% novel accuracy versus 81.58% for class-mean Tip-Adapter. Across five arrivals, the paired average gain over that cache control is 1.48 +/- 0.35 points. An 81-configuration shot-and-calibration study retains a five-shot ImageNet-R gain of 2.14 +/- 0.37 points over recalibrated classifier removal. Complete frontiers, prototype-replacement controls and a separate validation-selected threshold policy distinguish recognition capacity from operational threshold selection.

The work contributes to the journal's machine-learning and knowledge-based computer-vision scope by organizing heterogeneous prior and arriving evidence for a constrained decision problem. We position it against relevant KBS studies of multi-feature similarity supplementation, sharpness-aware prototype calibration, adaptive exploration, CLIP prompt fusion, novel-class discovery and representation-enhanced analytic learning. These are literature comparisons; we do not present unrun methods as experimental baselines.

The manuscript is original, has not been formally published and is not under consideration elsewhere. I approve its submission to Knowledge-Based Systems and declare no competing interests. The research received no external funding. AI assistance is disclosed in the manuscript. Code and source-hashed experimental records are public at https://github.com/Cheerssom/Reference-Anchored-Candidate-Tables .

Sincerely,
Junyi Zhang
Shandong University, China
cheerssom0876@gmail.com
'''
# Keep editor-facing narrative focused on the paper, without process commentary.
letter=letter.replace(' These are literature comparisons; we do not present unrun methods as experimental baselines.','')
if (H/'revision_summary.json').exists():
 a=letter.index('With frozen CLIP features');b=letter.index("The work contributes",a)
 letter=letter[:a]+'''Experiments cover CIFAR-100, Oxford-IIIT Pets and 200-class ImageNet-R with ten distinct class partitions, two arrival orders and frozen OpenAI CLIP ViT-L/14 and ViT-B/32. On ImageNet-R, the ViT-L/14 retained readout gives 84.31% novel accuracy at approximately 86.42% base accuracy in a descriptive matched-test-retention analysis. Across five arrivals, retaining the classifier adds 0.71 percentage points [conditional paired 95% CI: 0.46, 0.92] in average incremental accuracy over dense prototype fusion, and 1.42 points [1.15, 1.62] over class-mean semantic-cache retrieval. The dense-prototype comparison transfers to ViT-B/32 with a 0.90-point gain [0.69, 1.11]. These intervals describe class-assignment variation conditional on the fixed image pools and encoders. A main-paper validation-selected comparison reports achieved test group accuracies, feasibility and test-target attainment separately. Raw-score and fixed-temperature controls isolate reference normalization; the full tables retain regime-dependent outcomes, including comparisons favoring an alternative.\n\n'''+letter[b:]
(H/'cover_letter.md').write_text(letter,encoding='utf-8')

def doc(name,heading,paragraphs,bullets=False):
 d=Document();section=d.sections[0]
 section.top_margin=section.bottom_margin=Inches(.8)
 section.left_margin=section.right_margin=Inches(.85)
 normal=d.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(11)
 normal.paragraph_format.space_after=Pt(7)
 if heading:d.add_heading(heading,level=1)
 for text in paragraphs:d.add_paragraph(text,style='List Bullet' if bullets else None)
 d.save(H/name)

doc('Highlights.docx','Highlights',highlights,True)
doc('cover_letter.docx','',letter.strip().split('\n\n'))
doc('title_page.docx',title,['Junyi Zhang','Shandong University, China','Corresponding author: Junyi Zhang','Email: cheerssom0876@gmail.com'])
doc('declarations.docx','Declarations',[
'CRediT authorship contribution statement: Junyi Zhang: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing - original draft, Writing - review and editing.',
'Funding: This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.',
'Declaration of competing interest: The author declares that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.',
'Data availability: Public datasets and the CLIP checkpoint are obtained separately. Code and experimental records: https://github.com/Cheerssom/Reference-Anchored-Candidate-Tables .',
'AI assistance: OpenAI Codex assisted literature discovery, manuscript drafting and revision, experiment and verification scripts, and LaTeX and table preparation. The author is responsible for the accuracy, interpretation and originality of the submitted work and for reviewing and approving the final manuscript and supporting materials.'
])

sources=[H/n for n in ['main.tex','main.bbl','supplementary.tex','supplementary.bbl','references.bib','elsarticle.cls','elsarticle-num.bst','numbers.tex','retention_numbers.tex','strengthening_results.tex','depth_results.tex'] if (H/n).exists()]
sources += [H/n for n in ['elsarticle.dtx','elsarticle.ins','template_README','template_manifest.txt','elsarticle-template-num.tex','template_provenance.json']]
sources+=list(H.glob('table_*.tex'))
sources+=[H/'validation_budget_results.tex']
sources+=[p for p in H.glob('*.pdf') if p.name!='logo.pdf']
sources=[p for p in sources if p.name not in ['main.pdf','supplementary.pdf','review_complete.pdf','cover_letter.pdf','Highlights.pdf','title_page.pdf','declarations.pdf']]
sources+=list(H.glob('*.docx'))+[H/'Highlights.txt']
sources=sorted(set(sources))
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
with zipfile.ZipFile(H/'candidate_tables_kbs_source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sources:z.write(p,p.name)
 z.writestr('file_hashes.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(H/'candidate_tables_kbs_source.zip') as z:
 assert z.testzip() is None
 assert all('/' not in n for n in z.namelist())
with zipfile.ZipFile(H/'candidate_tables_kbs_supplement_source.zip','w',zipfile.ZIP_DEFLATED) as z:
 supplement_manifest={}
 for p in sources:
  if p.name not in ['main.tex','main.bbl']:
   z.write(p,p.name);supplement_manifest[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 z.write(H/'main.aux','main.aux')
 supplement_manifest['main.aux']=hashlib.sha256((H/'main.aux').read_bytes()).hexdigest()
 z.writestr('BUILD.txt','Compile supplementary.tex. main.aux supplies references to the main manuscript.\n')
 z.writestr('file_hashes.json',json.dumps(supplement_manifest,indent=2))
print(json.dumps({'highlight_characters':list(map(len,highlights)),'flat_source_files':len(sources)},indent=2))
