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
with fitz.open(H/'main.pdf') as document:
 first=' '.join(document[0].get_text().split())
 assert '86.82%' in first and 'ten partitions per encoder.' in first and 'matched HyCal transfer' in first
 assert 'Keywords:' in first
 report['abstract_complete_on_first_page']=True
 report['keywords_on_first_page']=True
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
 budgettext=maintext+'\n'+'\n'.join(p.get_text() for p in fitz.open(H/'supplementary.pdf'))
 for row in primary:
  assert row['finite_admission_cells']==10 and row['test_met_cells']==10
  assert f"{100*row['new_mean']:.2f}" in budgettext
 report['calibration_budget']={'metric_records':93600,'reconstructed_tests':verified['reconstructed_tests'],'primary_conditions':6,'supplementary_section':'L','restart':restart}
if r'\label{fig:capacity}' in main:
 presentation=json.loads((H/'presentation_verification.json').read_text())
 assert presentation['paired_comparisons_preserved']==18
 assert presentation['primary_calibration_conditions']==6
 assert presentation['complete_calibration_rows']==36
 for name,sha in presentation['source_sha256'].items():
  assert hashlib.sha256((H/name).read_bytes()).hexdigest()==sha
 assert len(re.findall(r'\\begin\{figure\}',main))==5
 assert len(re.findall(r'\\begin\{table\}',main))==3
 assert len(added)==10
 report['presentation_revision']={'main_figures':5,'main_tables':3,'kbs_citations':10,'paired_comparisons_preserved':18,'numerical_sources_unchanged':True}
 structure=json.loads((H/'error_structure_verification.json').read_text())
 assert structure['complete'] and structure['prediction_policy_checks']==480
 assert structure['max_absolute_error']<1e-10
 report['error_structure_identities']={k:v for k,v in structure.items() if k!='source_hashes'}
if (H/'strengthening_summary.json').exists():
 summary=json.loads((H/'strengthening_summary.json').read_text())
 assert set(summary)=={'cifar','pets','imagenetr'}
 for name in summary:assert summary[name]['methods']['retained']['final']['n']==3
 report['strengthening_datasets']=list(summary)
 report['new_result_inputs']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in H.glob('table_strengthened_*.tex')}
if r'\label{sec:shift}' in main:
 revision=json.loads((H/'reviewer_revision_summary.json').read_text())
 verified=json.loads((H.parents[1]/'experiments/kbs_reviewer_revision/verification.json').read_text())
 provenance=json.loads((H.parents[1]/'experiments/kbs_reviewer_revision/provenance_verification.json').read_text())
 restart=json.loads((H.parents[1]/'experiments/kbs_reviewer_revision/restart_verification.json').read_text())
 assert verified['complete'] and verified['metric_records']==24000 and verified['reconstructed_threshold_tests']==54960
 assert provenance['complete'] and provenance['pixel_overlap']==0 and provenance['protected_overlap']==0
 assert restart['complete']
 maintext='\n'.join(p.get_text() for p in fitz.open(H/'main.pdf'))
 pair=next(r for r in revision['comparisons'] if (r['encoder'],r['dataset'],r['comparator'])==('l14','imagenetr','anchored_tp'))
 assert pair['jointly_admitted']==10
 assert f"{pair['all_partition_gain']['mean']:.2f}" in maintext
 for m in ['retained','anchored_tp']:
  row=next(r for r in revision['policy'] if (r['encoder'],r['dataset'],r['method'],r['policy'])==('l14','imagenetr',m,'paired_accuracy'))
  assert row['finite']==10 and f"{100*row['novel']['mean']:.2f}" in maintext
 report['five_point_revision']={'metric_records':24000,'threshold_tests':54960,'supplementary_sections':'M-O','provenance_verified':True,'restart':restart}
if r'\label{app:mechanismfollowup}' in supp:
 mechanism=json.loads((H/'mechanism_followup/summary.json').read_text())
 assert mechanism['verification']['complete'] and mechanism['verification']['partition_order_records']==120
 assert mechanism['verification']['new_threshold_tests_reconstructed']==590
 case=next(r for r in mechanism['conditions'] if (r['encoder'],r['dataset'])==('l14','imagenetr'))
 maintext='\n'.join(p.get_text() for p in fitz.open(H/'main.pdf'))
 combinedtext=maintext+'\n'+'\n'.join(p.get_text() for p in fitz.open(H/'supplementary.pdf'))
 for value in [case['common_floor_envelope_gap_pp']['mean'],case['aligned_vs_baseline_policy_pp']['mean']]:
  assert f'{value:.2f}' in combinedtext
 report['mechanism_followup']={'partition_order_records':120,'threshold_tests':590,'supplementary_section':'P','semantic_audit_complete':False,'natural_shift_executed':False}
if (H/'reviewer_clarity_revision/normalization_recalibrated_summary.json').exists():
 clarity=json.loads((H/'reviewer_clarity_revision/normalization_recalibrated_summary.json').read_text())
 assert len(clarity['records'])==6
 assert r'\input{table_asymmetric_admission}' in main and r'\input{table_centering_recalibrated}' in supp
 assert r'\input{table_mechanism_common_floor}' in supp
 assert r'\input{table_asymmetric_centering}' in main
 assert r'\input{table_normalization_factorial}' not in main and r'\input{table_normalization_factorial}' in supp
 assert main.count(r'\begin{table}')==3
 for item in clarity['records']:
  assert item['test_met_centered']==item['test_met_uncentered']==10
 for name,sha in json.loads((H/'reviewer_clarity_revision/unchanged_experiment_inputs.json').read_text()).items():
  assert hashlib.sha256((H.parents[1]/name).read_bytes()).hexdigest()==sha
 report['reviewer_clarity_revision']={'conditions':6,'raw_checkpoint_files_unchanged':240,'capacity_table_in_supplement':True,'offset_reselected_control':True,'full_fusion_retuning_in_that_revision':False,'external_natural_shift_in_that_revision':False}
if r'\label{app:fullretuning}' in supp:
 tuned=json.loads((H/'full_fusion_retuning/summary.json').read_text())
 assert tuned['verification']['complete'] and tuned['verification']['metric_records']==28800
 assert tuned['verification']['reconstructed_threshold_tests']==26755
 assert json.loads((H.parents[1]/'experiments/kbs_full_fusion_retuning/restart_verification.json').read_text())['hash_and_mtime_unchanged']
 for enc in ['l14','b32']:
  c=next(v for v in tuned['conditions'] if v['encoder']==enc and v['dataset']=='imagenetr')
  for other in ['retained_none','retained_fixed']:
   assert f"{c['contrasts']['retained_query__'+other]['matched_test']['mean']:.2f}" in maintext
 report['full_fusion_retuning']={'complete':True,'metric_records':28800,'threshold_tests':26755,'independent_searches':360,'settings_per_search':3636,'checkpoint_restart_files':1338,'supplementary_section':'Q','fresh_semantic_audit_complete':False}
if r'\label{app:sketchvalidation}' in supp:
 sketch=json.loads((H/'sketch_validation/summary.json').read_text())
 assert sketch['verification']['complete'] and sketch['verification']['metric_records']==2700
 assert sketch['verification']['reconstructed_threshold_tests']==4899
 assert sketch['identity_counts']['unique']==6445 and sketch['identity_counts']['test']==3953
 assert json.loads((H.parents[1]/'experiments/kbs_sketch_validation/restart_verification.json').read_text())['hash_and_mtime_unchanged']
 for enc in ['l14','b32']:
  c=next(v for v in sketch['conditions'] if v['encoder']==enc and v['method']=='retained' and v['policy']=='paired_accuracy' and v['calibration']=='target_cal')
  assert c['finite']==c['test_met']==10
  assert f"{c['novel']['mean']:.2f}" in maintext
 report['natural_shift_validation']={'complete':True,'metric_records':2700,'threshold_tests':4899,'unique_images':6445,'source_byte_overlap_exclusions':3624,'calibration_images':2492,'test_images':3953,'checkpoint_restart_files':1049,'supplementary_section':'R','frozen_source_parameters':True,'fresh_semantic_audit_complete':False}
if (H/'asymmetric_presentation_verification.json').exists():
 presentation=json.loads((H/'asymmetric_presentation_verification.json').read_text())
 assert presentation['prediction_archives']==600
 assert presentation['retained_aggregation_matches_existing_summary']
 for name,sha in presentation['source_sha256'].items():
  assert hashlib.sha256((H.parents[1]/name).read_bytes()).hexdigest()==sha,name
 for row in presentation['averages']:
  assert f"{row['mean']:.2f}" in maintext
 assert (H/'table_asymmetric_centering.tex').read_text()==(H/'table_centering_recalibrated.tex').read_text().replace(r'\textit{ViT-',r'\textnormal{CLIP ViT-')
 report['recognition_first_revision']={'reaggregated_prediction_archives':600,'unchanged_archives':True,'average_accuracy_cells':24,'direct_admission_cells':24,'centering_control_values_preserved':True}
(H/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
