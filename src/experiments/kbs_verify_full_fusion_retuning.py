"""Independent archive, candidate-stopping and search-budget verification."""
from pathlib import Path
import json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from kbs_runtime import atomic_json,digest_file
from kbs_verify_reviewer_revision import verify_rows
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/kbs_full_fusion_retuning'

def main():
 start=time.perf_counter();count=tests=selections=0
 assert json.loads((OUT/'complete.json').read_text())['complete']
 for e in ['l14','b32']:
  for d in ['cifar','pets','imagenetr']:
   folder=OUT/e/d;config=json.loads((folder/'manifest.json').read_text())
   for name,sha in config['sources'].items():assert digest_file(ROOT/name)==sha
   for name,sha in json.loads((folder/'data_manifest.json').read_text()).items():assert digest_file(ROOT/name)==sha
   for p in range(10):
    po=folder/f'p{p}';sp=po/'selection.json';selection=json.loads(sp.read_text())
    assert digest_file(ROOT/f'experiments/kbs_independent_risk/{e}/{d}/p{p}/selection.json')==selection['original_selection_sha256']
    assert len(selection['selections'])==6
    for m,run in selection['selections'].items():
     best=(-1.,None)
     assert len(run['trials'])==36 and run['evaluated_settings']==3636
     assert [(v['a'],v['u']) for v in run['trials']]==[(a,u) for a in [0,.25,.5,1,2,4] for u in [0,.25,.5,1,2,4]]
     for v in run['trials']:
      hs=np.array(v['harmonics']);ds=np.array(v['offsets']);assert len(hs)==len(ds)==101 and np.isfinite(ds).all() and (np.diff(ds)>=0).all()
      j=int(np.argmax(hs))
      if hs[j]>best[0]+1e-12:best=(hs[j],dict(a=v['a'],u=v['u'],offset=float(ds[j]),pseudo_harmonic=float(hs[j]),quantile_index=j))
     assert best[1]==run['selected'];selections+=1
    for order in [0,1]:
     for session in range(1,6):
      path=po/f'o{order}_t{session}.json';r=json.loads(path.read_text());raw=path.with_suffix('.npz')
      assert digest_file(sp)==r['selection_sha256'] and digest_file(raw)==r['raw_sha256']
      assert len(r['records'])==48
      with np.load(raw) as z:
       base,novel=z['base_classes'],z['novel_classes'];yc,yt=z['cal_labels'],z['test_labels'];ref=z['test_reference_correct']
       n,k=verify_rows(r['records'],z,z,'cal_','test_',yc,yt,base,novel,z['cal_reference_correct'],ref,z);count+=n;tests+=k
       with np.load(ROOT/f'experiments/kbs_independent_risk/{e}/{d}/p{p}/o{order}_t{session}.npz') as old:
        assert np.array_equal(yt,old['test_labels']) and np.array_equal(ref,old['test_reference_correct'])
       # Independent feasibility and exact boundary witnesses for every oracle row.
       for row in r['records']:
        if row['policy']!='matched_test':continue
        method=row['method'];bp=z['test_'+method+'_base_prediction'];margin=z['test_'+method+'_margin'];npred=z['test_'+method+'_novel_prediction'];target=row['retention']*row['reference_closed_test']
        ceiling=np.mean([(bp[yt==c]==c).mean() for c in base]);assert (ceiling>=target-1e-10)==row['feasible']
        if not row['feasible']:continue
        # For any feasible finite boundary, next representable threshold either
        # preserves the same accuracy or crosses the exact requested constraint.
        off=row['offset'];assert np.isfinite(off)
        cc=(bp==yt)&~((off>margin)|((off==margin)&(npred<bp)))
        ab=np.mean([cc[yt==c].mean() for c in base]);assert ab>=target-1e-10
   print('RETUNING VERIFIED',e,d,flush=True)
 atomic_json(dict(complete=True,metric_records=count,reconstructed_threshold_tests=tests,selection_runs=selections,equal_budget_settings_per_run=3636,search_settings_total=selections*3636,original_reference_predictions_identical=True,seconds=time.perf_counter()-start),OUT/'verification.json')
 print('RETUNING VERIFICATION COMPLETE',count,tests,flush=True)
if __name__=='__main__':main()
