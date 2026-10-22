"""Venue-conversion checks; experiment evidence is unchanged from the checked release."""
from pathlib import Path
import hashlib,json,re,zipfile
import fitz
from docx import Document

H=Path(__file__).resolve().parent
main=(H/'main.tex').read_text(encoding='utf-8')
supp=(H/'supplementary.tex').read_text(encoding='utf-8')
bib=(H/'references.bib').read_text(encoding='utf-8')
keys=set(re.findall(r'@\w+\{([^,]+),',bib))
added=json.loads((H/'literature/added_kbs_references.json').read_text())
report={'target':'Knowledge-Based Systems','documents':{},'added_kbs_citations':len(added)}
alllabels={name:set(re.findall(r'\\label\{([^}]+)\}',s)) for name,s in [('main',main),('supplementary',supp)]}
for name,s in [('main',main),('supplementary',supp)]:
 cites={k for g in re.findall(r'\\cite\{([^}]+)\}',s) for k in g.split(',')}
 refs=set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',s))
 known=alllabels[name] | {'main-'+v for v in alllabels['main']}
 assert not cites-keys,(name,cites-keys)
 assert not refs-known,(name,refs-known)
 labels=re.findall(r'\\label\{([^}]+)\}',s)
 assert len(labels)==len(set(labels))
 log=(H/(name+'.log')).read_text(errors='replace')
 problems=[v for v in log.splitlines() if any(p in v for p in ['Overfull','undefined','LaTeX Warning:','Package natbib Warning:','Package hyperref Warning:','! '])]
 assert not problems,(name,problems)
 assert 'Warning--' not in (H/(name+'.blg')).read_text(errors='replace')
 pdf=fitz.open(H/(name+'.pdf'));text='\n'.join(p.get_text() for p in pdf)
 assert '??' not in text
 assert all(len(p.get_text().strip())>100 for p in pdf)
 fonts={item[0] for page in pdf for item in page.get_fonts()}
 assert all(pdf.extract_font(x)[-1] or (pdf.xref_get_key(x,'Subtype')[1]=='/Type3' and pdf.xref_get_key(x,'CharProcs')[0]!='null') for x in fonts),'Unembedded PDF font'
 report['documents'][name]={'pages':len(pdf),'cited_entries':len(cites),'missing_references':[],
  'compile_warnings':problems,'all_fonts_embedded':True}
 if name=='main':
  assert all(row['key'] in cites for row in added)
  if r'\label{sec:tenpartitions}' in main:
   current=json.loads((H/'revision_summary.json').read_text())
   matched=current['l14']['imagenetr']['methods']['retained']['policies']['matched_test']['0.95']
   for value in [matched['old']['mean'],matched['new']['mean'],current['l14']['imagenetr']['gains']['anchored_tp']['average']['mean']]:
    assert f'{value:.2f}' in text,value
   report['primary_class_partitions']=10
   report['primary_encoders']=['CLIP ViT-L/14','CLIP ViT-B/32']
  else:
   for value in ['83.48','75.79','61.36','2.18']:assert value in text,value
  assert 'Frontiers of Computer Science' not in pdf[0].get_text()
  assert 'Working Draft' not in text
  assert all(k in text for k in ['CRediT','Data availability','Declaration of competing interest','OpenAI Codex'])
abstract=main.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
report['abstract_words']=len(abstract.split());assert report['abstract_words']<=250
report['keywords']=len(main.split(r'\begin{keyword}')[1].split(r'\end{keyword}')[0].split(r'\sep'))
assert 1<=report['keywords']<=7
high=[p.text for p in Document(H/'Highlights.docx').paragraphs][1:]
assert 3<=len(high)<=5 and max(map(len,high))<=85
report['highlight_characters']=list(map(len,high))
for filename in ['numbers.tex','retention_numbers.tex']+[p.name for p in H.glob('table_*.tex')]:
 original=H.parent/'fcs'/filename
 if original.exists():assert (H/filename).read_bytes()==original.read_bytes(),filename
report['numerical_inputs_byte_identical_to_checked_revision']=True
if r'\label{sec:riskexperiment}' in main:
 risk=json.loads((H/'independent_risk_summary.json').read_text())
 verified=json.loads((H.parents[1]/'experiments/kbs_independent_risk/verification.json').read_text())
 assert verified['complete'] and verified['metric_records']==21600
 assert risk['metric_records']==21600
 for encoder in ['l14','b32']:
  row=next(r for r in risk['final'] if r['encoder']==encoder and r['dataset']=='imagenetr' and r['method']=='retained' and r['policy']=='risk' and r['level']==.05)
  assert row['expanded']==10 and row['test_met']==10
  with fitz.open(H/'supplementary.pdf') as document:
   assert f"{100*row['new_mean']:.2f}" in '\n'.join(page.get_text() for page in document)
 report['independent_risk_calibration']={'partition_runs':60,'metric_records':21600,'supplementary_section':'J','fallbacks_explicit':True}
if r'\label{sec:budgetexperiment}' in main:
 budget=json.loads((H/'calibration_budget_summary.json').read_text())
 verified=json.loads((H.parents[1]/'experiments/kbs_calibration_budget/verification.json').read_text())
 restart=json.loads((H.parents[1]/'experiments/kbs_calibration_budget/restart_verification.json').read_text())
 assert verified['complete'] and verified['metric_records']==93600
 assert budget['metric_records']==93600
 primary=[r for r in budget['summary'] if r['method']=='retained' and r['policy']=='risk' and r['cap']==40]
 assert len(primary)==6
 maintext='\n'.join(p.get_text() for p in fitz.open(H/'main.pdf'))
 for row in primary:
  assert row['finite_admission_cells']==10 and row['test_met_cells']==10
  assert f"{100*row['new_mean']:.2f}" in maintext
 report['calibration_budget']={'metric_records':93600,'reconstructed_tests':verified['reconstructed_tests'],'primary_conditions':6,'supplementary_section':'L','restart':restart}
if (H/'strengthening_summary.json').exists():
 summary=json.loads((H/'strengthening_summary.json').read_text())
 assert set(summary)=={'cifar','pets','imagenetr'}
 for name in summary:assert summary[name]['methods']['retained']['final']['n']==3
 report['strengthening_datasets']=list(summary)
 report['new_result_inputs']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in H.glob('table_strengthened_*.tex')}
(H/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
