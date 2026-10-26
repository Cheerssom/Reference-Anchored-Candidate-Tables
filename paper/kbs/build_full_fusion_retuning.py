"""Summarize equal-budget retuning only after archive reconstruction passes."""
from pathlib import Path
import json
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];OUT=ROOT/'experiments/kbs_full_fusion_retuning';DEST=H/'full_fusion_retuning';DEST.mkdir(exist_ok=True)
METHODS=[h+'_'+m for h in ['retained','dense'] for m in ['query','none','fixed']]
DATA={'cifar':'CIFAR-100','pets':'Pets','imagenetr':'ImageNet-R'}

def stats(values):
 a=np.asarray(values,float);rng=np.random.default_rng(20261027)
 if not len(a):return dict(n=0,mean=None,ci95=None)
 boots=a[rng.integers(0,len(a),(20000,len(a)))].mean(1)*100
 return dict(n=len(a),mean=100*float(a.mean()),ci95=np.quantile(boots,[.025,.975]).tolist() if len(a)>=5 else None,values=(100*a).tolist())
def fmt(v):
 if v['mean'] is None:return '--'
 return f"{v['mean']:+.2f}"+(f" [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]" if v['ci95'] is not None else '')
def main():
 verification=json.loads((OUT/'verification.json').read_text());assert verification['complete']
 allrows=[]
 for p in OUT.glob('*/*/p*/o*_t*.json'):allrows.extend(json.loads(p.read_text())['records'])
 assert len(allrows)==28800
 conditions=[]
 for e in ['l14','b32']:
  table=[r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',r'Data / scheme & Readout & Finite & Met & Base (\%) & Novel (\%) \\',r'\midrule']
  for d in DATA:
   rr=[r for r in allrows if r['encoder']==e and r['dataset']==d];methods={}
   for m in METHODS:
    rows=[r for r in rr if r['method']==m];detail={}
    for pol in ['fixed_gate','protection','paired_accuracy']:
     selected=[r for r in rows if r['policy']==pol]
     final=[r for r in selected if r['session']==5];first=[r for r in final if r['order']==0]
     meanpart=lambda key:stats([np.mean([r[key] for r in selected if r['part']==p]) for p in range(10)])
     finalpart=lambda key:stats([np.mean([r[key] for r in final if r['part']==p]) for p in range(10)])
     detail[pol]=dict(average_incremental=meanpart('overall'),final_novel=finalpart('new'),final_base=finalpart('old'),finite_final_paths=sum(not r['fallback'] for r in final),empirical_met_final_paths=sum(r.get('test_target_met',False) for r in final),first_order=dict(finite=sum(not r['fallback'] for r in first),met=sum(r.get('test_target_met',False) for r in first),base=100*np.mean([r['old'] for r in first]),novel=100*np.mean([r['new'] for r in first]),given_admission_novel=100*np.mean([r['new'] for r in first if not r['fallback']]) if any(not r['fallback'] for r in first) else None))
    cap=[r for r in rows if r['policy']=='matched_test' and r['session']==5 and r['retention']==.95];valid=[p for p in range(10) if all(r['feasible'] for r in cap if r['part']==p)]
    detail['common_floor_capacity']=stats([np.mean([r['new'] for r in cap if r['part']==p]) for p in valid]);methods[m]=detail
    f=detail['paired_accuracy']['first_order'];head,mode=m.split('_');table.append(f"{DATA[d]} / {mode} & {head} & {f['finite']}/10 & {f['met']}/10 & {f['base']:.2f} & {f['novel']:.2f} "+r'\\')
   contrasts={}
   for a,b in [('retained_query','retained_none'),('retained_query','retained_fixed'),('retained_query','dense_query')]:
    paired={}
    for policy in ['matched_test','paired_accuracy','protection']:
     aa=[r for r in rr if r['method']==a and r['policy']==policy and r['session']==5 and (policy!='matched_test' or r['retention']==.95)]
     bb=[r for r in rr if r['method']==b and r['policy']==policy and r['session']==5 and (policy!='matched_test' or r['retention']==.95)]
     values=[];parts=[]
     for p in range(10):
      x=[r for r in aa if r['part']==p];y=[r for r in bb if r['part']==p]
      assert len(x)==len(y)==2
      if policy=='matched_test' and not all(r['feasible'] for r in x+y):continue
      values.append(np.mean([r['new'] for r in x])-np.mean([r['new'] for r in y]));parts.append(p)
     paired[policy]=dict(**stats(values),partitions=parts)
    contrasts[a+'__'+b]=paired
   conditions.append(dict(encoder=e,dataset=d,methods=methods,contrasts=contrasts))
   if d!='imagenetr':table.append(r'\midrule')
  table += [r'\bottomrule',r'\end{tabular}'];(H/f'table_full_retuning_{e}.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
 tab=[r'\begin{tabular}{@{}llrrr@{}}',r'\toprule',r'Encoder & Data & Query $-$ none & Query $-$ fixed & Retained $-$ dense \\',r'\midrule']
 for c in conditions:
  vals=[c['contrasts'][a]['matched_test'] for a in ['retained_query__retained_none','retained_query__retained_fixed','retained_query__dense_query']]
  def cell(v):
   if v['mean'] is None:return '--'
   lo,hi=v['ci95']
   return r'\shortstack{'+f"{v['mean']:+.2f} ({v['n']})"+r'\\{\scriptsize '+f'[{lo:.2f},{hi:.2f}]'+r'}}'
  tab.append(('L/14' if c['encoder']=='l14' else 'B/32')+' & '+DATA[c['dataset']]+' & '+' & '.join(cell(v) for v in vals)+r' \\')
 tab += [r'\bottomrule',r'\end{tabular}'];(H/'table_full_retuning_contrasts.tex').write_text('\n'.join(tab)+'\n',encoding='utf-8')
 report=dict(scope='Equal fusion-search budget on base-only pseudo episodes; original common reference; orders averaged within partition; test-label capacities descriptive',verification=verification,conditions=conditions)
 (DEST/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 lines=['# Fully retuned fusion comparisons','',report['scope'],'','| Encoder/data | Query-none capacity gap pp | Query-fixed capacity gap pp | Retained-dense capacity gap pp |','|---|---:|---:|---:|']
 for c in conditions:
  vals=[c['contrasts'][a]['matched_test'] for a in ['retained_query__retained_none','retained_query__retained_fixed','retained_query__dense_query']]
  lines.append('| '+c['encoder']+'/'+DATA[c['dataset']]+' | '+' | '.join(fmt(v)+f" (n={v['n']})" for v in vals)+' |')
 lines += ['','All schemes independently select both fusion coefficients and a tuning offset under the same3636-setting budget. Ridge regularization, fitting pools and encoder are held fixed. Dense prototypes have separate base/novel coefficients. The original retained reference and fallback remain common; retuned fusion can recover original-reference errors, so G=0 is not asserted here. Oracle capacities and independently selected policies are different metrics. Selection intervals condition on fixed images and jointly feasible partitions. No natural-shift results are included in this study.']
 (DEST/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines))
if __name__=='__main__':main()
