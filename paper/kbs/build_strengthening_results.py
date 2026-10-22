"""Generate the expanded KBS tables/plots directly from complete saved runs."""
from pathlib import Path
import csv,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent;ROOT=H.parents[1]
sys.path.insert(0,str(ROOT/'src/experiments'))
from kbs_verify_strengthening import records
DATASETS=['cifar','pets','imagenetr']
NAMES={'cifar':'CIFAR-100','pets':'Pets','imagenetr':'ImageNet-R'}
METHODS=['retained','removed_fixed','no_learned_tuned','anchored_tp','tip_sum_calibrated','tip_mean_calibrated']
LABELS={'retained':'Retained classifier','removed_fixed':'Classifier removed (fixed gates)',
 'no_learned_tuned':'No classifier (recalibrated)','anchored_tp':'Anchored text + prototypes',
 'fenec_calibrated':'FeNeC-M + offset','fenec_euclidean_calibrated':'FeNeC-E + offset',
 'fenec_native':'FeNeC-M (native)','fenec_euclidean_native':'FeNeC-E (native)'}
LABELS.update({'tip_sum_calibrated':'Tip-Adapter sum + offset','tip_mean_calibrated':'Tip-Adapter mean + offset','tip_sum_native':'Tip-Adapter sum (native)','tip_mean_native':'Tip-Adapter mean (native)'})
ALL=METHODS+['fenec_calibrated','fenec_euclidean_calibrated','fenec_native','fenec_euclidean_native','tip_sum_native','tip_mean_native']
rows={}
for name in DATASETS:
 folder=ROOT/'experiments/kbs_strengthening'/name
 assert all((folder/f'p{p}/euclidean_complete.json').exists() for p in range(3)),name
 corrected=folder/'fenec_vit'
 assert all((corrected/f'p{p}/complete.json').exists() for p in range(3)),name
 extra=[r for path in corrected.glob('p*/*_o*_t*.json') for r in json.loads(path.read_text())['records']]
 rows[name]=[r for r in records(name) if not r['method'].startswith('fenec')]+extra
 assert len(rows[name])==2640
 depth=ROOT/'experiments/kbs_depth'/name/'tip'
 assert len(list(depth.glob('p*/complete.json')))==3
 rows[name]+=[r for path in depth.glob('p*/o*_t*.json') for r in json.loads(path.read_text())['records']]
 assert len(rows[name])==3960

def select(ds,method,evaluation,ratio=None,session=5):
 return [r for r in rows[ds] if r['method']==method and r['evaluation']==evaluation and (session is None or r['session']==session)
         and (ratio is None or r['ratio']==ratio) and r.get('feasible',True)]

def partition_values(r,key):
 return {p:float(np.mean([x[key] for x in r if x['part']==p]))*100 for p in range(3) if any(x['part']==p for x in r)}

def stat(r,key):
 v=list(partition_values(r,key).values())
 return {'mean':float(np.mean(v)) if v else None,'sd':float(np.std(v,ddof=1)) if len(v)>1 else None,'n':len(v)}

def cell(r,key):
 s=stat(r,key)
 if s['mean'] is None:return '--'
 return f"{s['mean']:.2f}"+(f" $\\pm$ {s['sd']:.2f}" if s['sd'] is not None else '')

def paired(a,b,key):
 aa=partition_values(a,key);bb=partition_values(b,key);v=[aa[p]-bb[p] for p in aa.keys()&bb.keys()]
 return {'mean':float(np.mean(v)) if v else None,'sd':float(np.std(v,ddof=1)) if len(v)>1 else None,'n':len(v)}

summary={};flat=[]
for ds in DATASETS:
 summary[ds]={'methods':{},'gains':{}}
 for method in ALL:
  info={'final':stat(select(ds,method,'fixed_gate'),'overall'),
        'incremental_average':stat(select(ds,method,'fixed_gate',session=None),'overall'),
        'matched':{str(r):{k:stat(select(ds,method,'matched_test',r),k) for k in ['old','new']} for r in [.8,.85,.9,.95,.99]}}
  summary[ds]['methods'][method]=info
  for ratio in [.8,.85,.9,.95,.99]:
   rr=select(ds,method,'matched_test',ratio)
   flat.append({'dataset':ds,'method':method,'ratio':ratio,'base':stat(rr,'old')['mean'],'novel':stat(rr,'new')['mean'],'feasible':stat(rr,'old')['n']})
  summary[ds]['gains'][method]={
   'incremental_average':paired(select(ds,'retained','fixed_gate',session=None),select(ds,method,'fixed_gate',session=None),'overall'),
   'final':paired(select(ds,'retained','fixed_gate'),select(ds,method,'fixed_gate'),'overall'),
   'matched95_novel':paired(select(ds,'retained','matched_test',.95),select(ds,method,'matched_test',.95),'new')}
(H/'strengthening_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
with (H/'strengthening_summary.csv').open('w',newline='',encoding='utf-8') as f:
 writer=csv.DictWriter(f,fieldnames=flat[0]);writer.writeheader();writer.writerows(flat)

lines=[r'\begin{tabular}{lllrrr}',r'\toprule',r'Data & Budget & Readout & Base & Novel & Feasible \\',r'\midrule']
for ds in DATASETS:
 for method in METHODS:
  r=select(ds,method,'matched_test',.95)
  lines.append(f"{NAMES[ds]} & 95\\% & {LABELS[method]} & {cell(r,'old')} & {cell(r,'new')} & {stat(r,'old')['n']}/3 \\\\")
 lines.append(r'\midrule')
lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
(H/'table_strengthened_matched.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Data & Readout & Incremental avg. & Final & $\Delta$ avg. & $\Delta$ final \\',r'\midrule']
for ds in DATASETS:
 for method in METHODS:
  average=select(ds,method,'fixed_gate',session=None);final=select(ds,method,'fixed_gate')
  def gain_cell(key):
   s=summary[ds]['gains'][method][key];return f"{s['mean']:.2f} $\\pm$ {s['sd']:.2f}"
  lines.append(f"{NAMES[ds]} & {LABELS[method]} & {cell(average,'overall')} & {cell(final,'overall')} & {gain_cell('incremental_average')} & {gain_cell('final')} \\\\")
 lines.append(r'\midrule')
lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
(H/'table_strengthened_sequential.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

lines=[r'\begin{tabular}{llrrr}',r'\toprule',r'Budget & Readout & Base & Novel & Feasible \\',r'\midrule']
for ratio in [.9,.95,.99]:
 for method in METHODS:
  r=select('imagenetr',method,'matched_test',ratio)
  lines.append(f"{100*ratio:.0f}\\% & {LABELS[method]} & {cell(r,'old')} & {cell(r,'new')} & {stat(r,'old')['n']}/3 \\\\")
 lines.append(r'\midrule')
lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
(H/'table_imagenetr_retention.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Readout & Base test & Novel test & Validation feasible \\',r'\midrule']
for method in METHODS:
 r=select('imagenetr',method,'base_validation',.95)
 lines.append(f"{LABELS[method]} & {cell(r,'old')} & {cell(r,'new')} & {stat(r,'old')['n']}/3 \\\\")
lines.extend([r'\bottomrule',r'\end{tabular}'])
(H/'table_imagenetr_validation.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

ir=summary['imagenetr']['methods']['retained'];gain=summary['imagenetr']['gains']['no_learned_tuned']
baseline=summary['imagenetr']['methods']['no_learned_tuned']['matched']['0.95']['new']
def measured(s):return f"{s['mean']:.2f}"+(f" $\\pm$ {s['sd']:.2f}" if s['sd'] is not None else '')
paragraph=f"At the ImageNet-R 95\\% budget, the retained table reaches {measured(ir['matched']['0.95']['old'])}\\% base accuracy and {measured(ir['matched']['0.95']['new'])}\\% novel accuracy, with all three partitions feasible (Table~\\ref{{tab:matchedretention}}). "
if baseline['n']==3:
 paragraph+=f"The recalibrated classifier-free readout gives {measured(baseline)}\\% novel accuracy, a paired novel-accuracy gain of {measured(gain['matched95_novel'])} points for the retained classifier. "
else:paragraph+=f"The recalibrated classifier-free readout meets this budget in {baseline['n']} of three partitions; the table identifies the corresponding feasible sets. "
paragraph+=f"Across the five 20-class arrivals, retaining the classifier adds {measured(gain['incremental_average'])} percentage points in average incremental accuracy and {measured(gain['final'])} points in final accuracy over the recalibrated removal (Table~\\ref{{tab:sequentialvalue}}).\n\n"
paragraph+='The FeNeC comparisons use the same fitting and support pools on all three datasets. Their native and group-offset-calibrated trajectories are reported separately in Supplementary Section G; matching the test-retention target reselects the full offset and therefore yields the same frontier for both forms. Supplementary Section F reports the full ImageNet-R budgets and the independent base-validation policy.\n'
(H/'strengthening_results.tex').write_text(paragraph,encoding='utf-8')

plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(7.2,3.1),constrained_layout=True)
colors=['#155e75','#a16207','#c2410c','#4d7c0f','#7c3aed','#475569']
for ax,ds in zip(axes,DATASETS):
 for method,color in zip(METHODS,colors):
  x=[];y=[];error=[]
  for ratio in [.8,.85,.9,.95,.99]:
   s=stat(select(ds,method,'matched_test',ratio),'new')
   if s['n']==3:x.append(100*ratio);y.append(s['mean']);error.append(s['sd'])
  if x:ax.errorbar(x,y,yerr=error,color=color,marker='o',markersize=3,linewidth=1.1,capsize=2,label=LABELS[method])
 ax.set_title(NAMES[ds]);ax.set_xticks([80,85,90,95,99]);ax.set_xlabel('Retention budget (%)');ax.grid(alpha=.18)
axes[0].set_ylabel('Novel accuracy (%)')
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False,fontsize=8)
for suffix in ['pdf','png']:fig.savefig(H/('matched_retention_kbs.'+suffix),dpi=220)
plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(7.2,4.4),constrained_layout=True)
for col,ds in enumerate(DATASETS):
 for method,color in [('no_learned_tuned','#c2410c'),('anchored_tp','#4d7c0f')]:
  for row,evaluation in enumerate(['matched_test','fixed_gate']):
   means=[];sds=[]
   for session in range(1,6):
    a=select(ds,'retained',evaluation,.95 if evaluation=='matched_test' else None,session)
    b=select(ds,method,evaluation,.95 if evaluation=='matched_test' else None,session)
    s=paired(a,b,'new' if row==0 else 'overall');means.append(s['mean'] if s['mean'] is not None else np.nan);sds.append(s['sd'] if s['sd'] is not None else np.nan)
   axes[row,col].errorbar(range(1,6),means,yerr=sds,color=color,marker='o',markersize=3,capsize=2,label=LABELS[method])
 for row in range(2):
  axes[row,col].axhline(0,color='#777',linewidth=.7);axes[row,col].set_xticks(range(1,6));axes[row,col].grid(alpha=.15)
 axes[0,col].set_title(NAMES[ds]);axes[1,col].set_xlabel('Arrival session')
axes[0,0].set_ylabel('Novel gain at 95% budget (pp)');axes[1,0].set_ylabel('Fixed-gate all-class gain (pp)')
axes[1,0].legend(frameon=False,fontsize=8)
for suffix in ['pdf','png']:fig.savefig(H/('sequential_value_kbs.'+suffix),dpi=220)
plt.close(fig)
print(json.dumps(summary['imagenetr'],indent=2))
