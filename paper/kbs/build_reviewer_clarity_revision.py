"""Generate reviewer-facing summaries from unchanged frozen-model records."""
from pathlib import Path
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent; ROOT=H.parents[1]
OUT=H/'reviewer_clarity_revision';OUT.mkdir(exist_ok=True)
DATA={'cifar':'CIFAR-100','pets':'Pets','imagenetr':'ImageNet-R'}

def main():
 summary=json.loads((H/'mechanism_followup/summary.json').read_text())
 manifest=json.loads((OUT/'unchanged_experiment_inputs.json').read_text())
 for name,sha in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
 records=[json.loads(p.read_text()) for p in (ROOT/'experiments/kbs_mechanism_followup').glob('*/*/p*/o0.json')]
 assert len(records)==60
 lines=[r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',r'Data & Centering & Finite & Base (\%) & Novel (\%) & Gain (pp) \\',r'\midrule']
 results=[]
 for s in summary['conditions']:
  e,d=s['encoder'],s['dataset'];rr=sorted([v for v in records if (v['encoder'],v['dataset'])==(e,d)],key=lambda v:v['part'])
  c=[next(v for v in r['rows'] if v['method']=='retained' and v['policy']=='paired_accuracy') for r in rr];u=[r['uncentered_direct'] for r in rr]
  gain=100*np.array([a['novel']-b['new'] for a,b in zip(c,u)])
  oldc=100*np.mean([v['base'] for v in c]);oldu=100*np.mean([v['old'] for v in u]);nc=100*np.mean([v['novel'] for v in c]);nu=100*np.mean([v['new'] for v in u])
  assert np.allclose(gain,s['centered_minus_uncentered_direct_pp']['values'])
  fc=sum(not v['fallback'] for v in c);fu=sum(not v['fallback'] for v in u)
  mc=sum(v['base']>=v['target']-1e-10 for v in c);mu=sum(v['test_target_met'] for v in u)
  for a,v in zip(c,u):assert bool(v['old']>=a['target']-1e-10)==v['test_target_met']
  item=dict(encoder=e,dataset=d,finite_centered=fc,finite_uncentered=fu,test_met_centered=mc,test_met_uncentered=mu,base_centered=oldc,base_uncentered=oldu,novel_centered=nc,novel_uncentered=nu,paired_gain=s['centered_minus_uncentered_direct_pp'],conditional_novel_centered=100*np.mean([v['novel'] for v in c if not v['fallback']]),conditional_novel_uncentered=100*np.mean([v['new'] for v in u if not v['fallback']]))
  results.append(item)
  lo,hi=item['paired_gain']['ci95'];enc='L/14' if e=='l14' else 'B/32'
  if d=='cifar':
   if e=='b32':lines.append(r'\midrule')
   lines.append(r'\multicolumn{6}{@{}l}{\textit{ViT-'+enc+r'}} \\')
  lines.append(f"{DATA[d]} & Query & {fc}/10 & {oldc:.2f} & {nc:.2f} & +{gain.mean():.2f} [{lo:.2f},{hi:.2f}] "+r'\\')
  lines.append(f" & None & {fu}/10 & {oldu:.2f} & {nu:.2f} & "+r'\\')
 lines += [r'\bottomrule',r'\end{tabular}']
 (H/'table_centering_recalibrated.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (OUT/'normalization_recalibrated_summary.json').write_text(json.dumps(dict(scope='Final session, first order, ten fixed-image partitions; fixed weights/scales, independently recalibrated offset; not full hyperparameter retuning',records=results),indent=2),encoding='utf-8')
 # Copy already verified common-floor estimates with human-readable encoder names.
 tab=(H/'table_mechanism_common_floor.tex').read_text().replace('l14 &','L/14 &').replace('b32 &','B/32 &')
 (H/'table_policy_capacity_main.tex').write_text(tab,encoding='utf-8')
 print(json.dumps({'source_checkpoint_files_unchanged':len(manifest),'normalization_conditions':len(results),'all_centered_and_uncentered_final_test_targets_met':all(v['test_met_centered']==v['test_met_uncentered']==10 for v in results)},indent=2))
if __name__=='__main__':main()
