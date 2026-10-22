"""Prepare editable KBS submission attachments and reproducible source archive."""
from pathlib import Path
import hashlib,json,re,zipfile
from docx import Document
from docx.shared import Inches,Pt

H=Path(__file__).resolve().parent
title='Retention-constrained class expansion with reference-anchored candidate tables'
highlights=[
 'Retained discrimination supplies capacity for protected class expansion.',
 'Fixed references separate existing discrimination from novel admission.',
 'Classifier value is measured at matched retention and across repeated arrivals.',
 'ImageNet-R gains over dense prototypes persist with two frozen CLIP encoders.',
 'Independent class coverage supports certified admission across three datasets.'
]
assert 3<=len(highlights)<=5 and max(map(len,highlights))<=85
(H/'Highlights.txt').write_text('\n'.join('- '+v for v in highlights)+'\n',encoding='utf-8')
letter=f"""Dear Editors of Knowledge-Based Systems,

Please consider our manuscript, "{title}," as an original Research Article.

The paper investigates how a trained classifier's discrimination can support new-class recognition under an explicit original-class accuracy requirement. Reference-anchored candidate tables preserve the existing score frame while a separate threshold controls novel admission. Matched-retention frontiers and paired arrival sequences quantify the retained classifier's expansion capacity and continuing predictive value.

Experiments cover three datasets, ten class partitions and two frozen CLIP encoders. On ImageNet-R, retained classifiers improve average incremental accuracy over dense prototypes by 0.71 points with ViT-L/14 and 0.90 with ViT-B/32. Independent protection-risk calibration with up to 40 images per base class certifies final-session expansion in all ten partitions of every dataset and encoder. ViT-L/14 novel accuracy reaches 72.76% on CIFAR-100, 88.51% on Pets and 80.74% on ImageNet-R, with the 95% test-retention target met in each partition. A coverage condition explains certification feasibility; all readouts share calibration access and the manuscript reports fallback separately from useful admission.

The manuscript is original, has not been formally published and is not under consideration elsewhere. I approve submission to Knowledge-Based Systems, declare no competing interests, and received no external funding. AI assistance is disclosed. Code and source-hashed records are available at https://github.com/Cheerssom/Reference-Anchored-Candidate-Tables .

Sincerely,
Junyi Zhang
Shandong University, China
cheerssom0876@gmail.com
"""
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
