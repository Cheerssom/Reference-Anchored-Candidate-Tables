"""Natural-shift summary; requires completed independent policy reconstruction."""
from pathlib import Path
import json
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/experiments'))
import fcs_retention_value as rv
import kbs_mechanism_followup as mechanism
H=Path(__file__).resolve().parent;ROOT=H.parents[1];OUT=ROOT/'experiments/kbs_sketch_validation';DEST=H/'sketch_validation';DEST.mkdir(exist_ok=True)
NAMES={'retained':'Retained','anchored_tp':'Dense prototypes','hycal':'HyCal'}
def stat(v):
 a=np.array(v,float);rng=np.random.default_rng(20261027)
 if not len(a):return dict(n=0,mean=None,ci95=None)
 boots=a[rng.integers(0,len(a),(20000,len(a)))].mean(1)*100
 return dict(n=len(a),mean=100*float(a.mean()),ci95=np.quantile(boots,[.025,.975]).tolist() if len(a)>=5 else None,values=(100*a).tolist())
def main():
 verification=json.loads((OUT/'verification.json').read_text());assert verification['complete']
 identity=json.loads((OUT/'identity_split.json').read_text());budget=json.loads((OUT/'calibration_budget.json').read_text());allrows=[]
 for p in (OUT/'evaluation').glob('*/p*/o0_t*.json'):allrows.extend(json.loads(p.read_text())['records'])
 assert len(allrows)==2700
 summary=[];pairs=[]
 for enc in ['l14','b32']:
  for policy in ['protection','paired_accuracy']:
   table=[r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',r'Readout & Calibration & Finite & Met & Base (\%) & Novel (\%) \\',r'\midrule']
   for method in NAMES:
    for cal in ['source_cal','target_cal']:
     rows=[r for r in allrows if (r['encoder'],r['method'],r['policy'],r.get('calibration'))==(enc,method,policy,cal)];final=sorted([r for r in rows if r['session']==5],key=lambda x:x['part']);assert len(final)==10
     finite=[r for r in final if not r['fallback']]
     rec=dict(encoder=enc,method=method,policy=policy,calibration=cal,finite=len(finite),test_met=sum(r['test_target_met'] for r in final),base=stat([r['old'] for r in final]),novel=stat([r['new'] for r in final]),reference=stat([r['reference_closed_test'] for r in final]),conditional_novel=stat([r['new'] for r in finite]),all_sessions_met_paths=sum(all(r['test_target_met'] for r in rows if r['part']==p) for p in range(10)),all_sessions_finite_paths=sum(all(not r['fallback'] for r in rows if r['part']==p) for p in range(10)),calibration_queries=sorted(set(r['calibration_queries'] for r in final)),test_retention_ratio_mean=float(np.mean([r['old']/r['reference_closed_test'] for r in final])))
     summary.append(rec);name='Source' if cal=='source_cal' else 'Sketch';table.append(f"{NAMES[method]} & {name} & {rec['finite']}/10 & {rec['test_met']}/10 & {rec['base']['mean']:.2f} & {rec['novel']['mean']:.2f} "+r'\\')
    if method!='hycal':table.append(r'\midrule')
   table += [r'\bottomrule',r'\end{tabular}'];(H/f'table_sketch_{enc}_{policy}.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
   for cal in ['source_cal','target_cal']:
    a=[r for r in allrows if (r['encoder'],r['method'],r['policy'],r.get('calibration'),r['session'])==(enc,'retained',policy,cal,5)]
    b=[r for r in allrows if (r['encoder'],r['method'],r['policy'],r.get('calibration'),r['session'])==(enc,'anchored_tp',policy,cal,5)]
    pairs.append(dict(encoder=enc,calibration=cal,policy=policy,retained_minus_dense_novel=stat([next(r['new'] for r in a if r['part']==p)-next(r['new'] for r in b if r['part']==p) for p in range(10)])))
 counts=dict(acquired=10152,unique=len(identity['records']),excluded=len(identity['excluded']),calibration_reserved=sum(r['role']=='cal' for r in identity['records']),test=sum(r['role']=='test' for r in identity['records']),minimum_unique_class=min(r['unique'] for r in identity['class_counts']),maximum_unique_class=max(r['unique'] for r in identity['class_counts']),matched_calibration_total=len(budget['target_indices']))
 reasons={reason:sum(r['reason']==reason for r in identity['excluded']) for reason in sorted({r['reason'] for r in identity['excluded']})}
 capacity=[];capacity_pairs=[];curves={}
 for enc in ['l14','b32']:
  tab=[r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',r'Readout & Feasible & Base (\%) & Novel capacity (\%) \\',r'\midrule']
  for method in NAMES:
   r=[v for v in allrows if v['encoder']==enc and v['method']==method and v['policy']=='matched_test' and v['session']==5 and v['retention']==.95]
   feasible=[v for v in r if v['feasible']];record=dict(encoder=enc,method=method,finite=len(feasible),novel=stat([v['new'] for v in feasible]),base=stat([v['old'] for v in feasible]));capacity.append(record)
   base='--' if not feasible else f"{record['base']['mean']:.2f}";novel='--' if not feasible else f"{record['novel']['mean']:.2f}"
   tab.append(f"{NAMES[method]} & {len(feasible)}/10 & {base} & {novel} "+r'\\')
  tab += [r'\bottomrule',r'\end{tabular}'];(H/f'table_sketch_capacity_{enc}.tex').write_text('\n'.join(tab)+'\n',encoding='utf-8')
  aa=[v for v in allrows if (v['encoder'],v['method'],v['policy'],v['session'])==(enc,'retained','matched_test',5) and v['retention']==.95]
  bb=[v for v in allrows if (v['encoder'],v['method'],v['policy'],v['session'])==(enc,'anchored_tp','matched_test',5) and v['retention']==.95];values=[]
  for part in range(10):
   a=next(v for v in aa if v['part']==part);b=next(v for v in bb if v['part']==part)
   if a['feasible'] and b['feasible']:values.append(a['new']-b['new'])
  capacity_pairs.append(dict(encoder=enc,retained_minus_dense=stat(values)))
  for part in range(10):
   raw=OUT/f'evaluation/{enc}/p{part}/o0_t5.npz'
   with np.load(raw) as z:
    for method in NAMES:
     score_base=z['test_'+method+'_base_prediction'];score_new=z['test_'+method+'_novel_prediction'];y=z['test_labels'];base=z['base_classes'];new=z['novel_classes'];w=np.zeros(len(y))
     for group in [base,new]:
      for c in group:w[y==c]=1/(len(group)*np.sum(y==c))
     c=dict(margin=z['test_'+method+'_margin'],base_prediction=score_base,novel_prediction=score_new,base_correct=np.isin(y,base)&(score_base==y),novel_correct=np.isin(y,new)&(score_new==y),weights=w,base_mask=np.isin(y,base),novel_mask=np.isin(y,new),nb=len(base),nn=len(new))
     curve=mechanism.envelope(c);curves[enc,part,method]=curve
     original=next(v for v in allrows if (v['encoder'],v['part'],v['method'],v['policy'],v['session'])==(enc,part,method,'matched_test',5) and v['retention']==.95)
     eligible=curve['base']>=.95*original['reference_closed_test']-1e-10
     assert bool(eligible.any())==original['feasible']
     if eligible.any():assert abs(float(curve['novel'][eligible].max())-original['new'])<1e-9
 plot_frontiers(curves,allrows)
 result=dict(scope='New target-domain benchmark evaluation; frozen source models/support/tuning; order0 five arrivals; exact deduplication; no guaranteed pretraining novelty',verification=verification,identity_counts=counts,exclusion_reasons=reasons,conditions=summary,paired_comparisons=pairs,capacities=capacity,capacity_comparisons=capacity_pairs)
 (DEST/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 lines=['# ImageNet-Sketch natural-shift validation','',result['scope'],'',json.dumps(counts),'','| Encoder | Readout | Rule | Calibration | Finite | Met | Base | Novel | Conditional novel |','|---|---|---|---|---:|---:|---:|---:|---:|']
 for r in summary:
  cond='--' if r['conditional_novel']['mean'] is None else f"{r['conditional_novel']['mean']:.2f}"
  lines.append(f"| {r['encoder']} | {r['method']} | {r['policy']} | {r['calibration']} | {r['finite']}/10 | {r['test_met']}/10 | {r['base']['mean']:.2f} | {r['novel']['mean']:.2f} | {cond} |")
 lines += ['','Both calibration domains use equal counts within every base class. Source fit, support and score coefficients are unchanged. Source-calibrated outcomes diagnose transport under shift; the source-law guarantee does not cover the changed target law. Sketch-calibrated guarantees retain the class-conditional independence assumptions. Exact/dihedral hashes do not establish absence of all near-duplicate dependence. Intervals condition on the fixed target images and class assignments, and confidence statements apply separately to each fixed method/partition/calibration/policy over five prespecified arrivals.']
 lines += ['', '## Common-target capacity', '', '| Encoder | Retained-minus-dense capacity pp | Jointly feasible |', '|---|---:|---:|']
 for r in capacity_pairs:
  v=r['retained_minus_dense'];lines.append(f"| {r['encoder']} | {v['mean']:+.3f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}] | {v['n']}/10 |")
 lines += ['', 'These use the same jointly feasible partitions. L/14 selected-policy gains do not imply a common-floor recognition-capacity advantage. Full curves are in operating_frontiers_all.pdf and frontiers.npz; partition0 is an index-based illustration, not a best-case selection.']
 (DEST/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines))
def plot_frontiers(curves,allrows):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 from matplotlib.backends.backend_pdf import PdfPages
 colors={'retained':'#1B4F72','anchored_tp':'#D55E00','hycal':'#009E73'}
 plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 def panel(ax,enc,part,zoom):
  rows=[r for r in allrows if r['encoder']==enc and r['part']==part and r['session']==5]
  ref=rows[0]['reference_closed_test'];target=.95*ref
  for m in NAMES:
   curve=curves[enc,part,m];ax.plot(100*curve['base'],100*curve['novel'],color=colors[m],lw=1.4,label=NAMES[m])
   for cal in ['source_cal','target_cal']:
    r=next(v for v in rows if v['method']==m and v['policy']=='paired_accuracy' and v['calibration']==cal)
    ax.scatter(100*r['old'],100*r['new'],marker='x' if r['fallback'] else 'o',s=35,color=colors[m],facecolors='none' if cal=='source_cal' and not r['fallback'] else colors[m],zorder=5)
  ax.axvline(100*target,ls=':',color='#777777',lw=1)
  ax.set_xlim(max(0,100*target-8) if zoom else 0,min(100,100*max(ref,max(curves[enc,part,m]['base'][0] for m in NAMES))+1) if zoom else 100)
  ax.set_ylim(-2,102);ax.set_xlabel('Actual base accuracy (%)');ax.set_ylabel('Novel accuracy (%)');ax.grid(axis='y',alpha=.15);ax.set_title(('L/14' if enc=='l14' else 'B/32')+f', partition {part}')
 fig,axes=plt.subplots(1,2,figsize=(10.8,4.3))
 for enc,ax in zip(['l14','b32'],axes):panel(ax,enc,0,True)
 hh,ll=axes[0].get_legend_handles_labels();hh += [Line2D([],[],color='#555555',marker='o',markerfacecolor='none',ls='',label='Source calibration'),Line2D([],[],color='#555555',marker='o',ls='',label='Sketch calibration'),Line2D([],[],color='#555555',marker='x',ls='',label='Reference fallback')]
 fig.legend(handles=hh,ncol=3,loc='lower center',frameon=False);fig.tight_layout(rect=[0,.15,1,1])
 for ext in ['pdf','png']:fig.savefig(H/('sketch_operating_frontiers.'+ext),dpi=300)
 plt.close(fig)
 with PdfPages(DEST/'operating_frontiers_all.pdf') as pdf:
  for enc in ['l14','b32']:
   fig,axes=plt.subplots(2,5,figsize=(15,7))
   for part,ax in enumerate(axes.flat):panel(ax,enc,part,False)
   fig.tight_layout();pdf.savefig(fig);plt.close(fig)
 np.savez_compressed(DEST/'frontiers.npz',**{f'{e}_p{p}_{m}_{k}':v for (e,p,m),c in curves.items() for k,v in c.items()})

if __name__=='__main__':main()
