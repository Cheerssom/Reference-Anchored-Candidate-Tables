"""Cache-free reconstruction of the extended records and paired data provenance."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
METHODS=['retained','removed_fixed','no_learned_tuned','anchored_tp','fenec_native','fenec_calibrated']

def records(dataset):
 folder=ROOT/'experiments/kbs_strengthening'/dataset
 return [r for p in sorted(folder.glob('p*/o*_t*.json')) for r in json.loads(p.read_text())['records']]

def verify(dataset):
 folder=ROOT/'experiments/kbs_strengthening'/dataset
 manifest=json.loads((folder/'run_manifest.json').read_text())
 assert hashlib.sha256((ROOT/'experiments/kbs_strengthening_protocol.md').read_bytes()).hexdigest()==manifest['protocol_sha256']
 for name,sha in manifest['dependencies'].items():
  if name.startswith('kbs_'):
   snapshot=folder/(Path(name).stem+'_snapshot.py')
   if snapshot.exists():assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==sha
 rows=records(dataset);assert len(rows)==1980,(dataset,len(rows))
 checked=0;groups={};oldrows={}
 if dataset in ['cifar','pets']:
  prior=ROOT/'experiments'/('fcs' if dataset=='cifar' else 'fcs_pets')/'retention_value/results.json'
  oldrows={(r['part'],r['order'],r['session'],r['method'],r['evaluation'],r.get('ratio')):r for r in json.loads(prior.read_text())['records']}
 max_prior_delta=0.
 for part in range(3):
  assert json.loads((folder/f'p{part}/complete.json').read_text())['complete']
  for order in [0,1]:
   for session in range(1,6):
    selected=[r for r in rows if (r['part'],r['order'],r['session'])==(part,order,session)]
    with np.load(folder/f'p{part}/o{order}_t{session}.npz') as z:
     y=z['labels'];base=z['base_classes'];novel=z['novel_classes']
     w=np.zeros(len(y))
     for group in [base,novel]:
      for c in group:w[y==c]=1./(len(group)*np.sum(y==c))
     for method in METHODS:
      assert np.allclose(w,z[method+'_weights'])
      assert np.array_equal(z[method+'_base_correct'],np.isin(y,base)&(z[method+'_base_prediction']==y))
      assert np.array_equal(z[method+'_novel_correct'],np.isin(y,novel)&(z[method+'_novel_prediction']==y))
     for r in selected:
      key=(part,order,session,r['method'],r['evaluation'],r.get('ratio'))
      if key in oldrows:
       old=oldrows[key]
       assert r.get('feasible')==old.get('feasible')
       for metric in ['old','new','overall','harmonic']:
        if metric in r:max_prior_delta=max(max_prior_delta,abs(r[metric]-old[metric]))
      if r['evaluation']!='fixed_gate':
       group=(part,order,session,r['evaluation'],r['ratio'])
       if group in groups:assert abs(groups[group]-r['target'])<1e-12
       groups[group]=r['target']
       if not r['feasible']:
        assert 'new' not in r and r['offset'] is None
        assert r['selection_base_ceiling']<r['target']+1e-10
        continue
       assert r['selection_base_accuracy']>=r['target']-1e-10
      m=r['method'];offset=r['offset'];margin=z[m+'_margin']
      new=(offset>margin)|((offset==margin)&(z[m+'_novel_prediction']<z[m+'_base_prediction']))
      ab=float(w[z[m+'_base_correct']&~new].sum());an=float(w[z[m+'_novel_correct']&new].sum())
      assert abs(ab-r['old'])<1e-10 and abs(an-r['new'])<1e-10
      assert abs((ab*len(base)+an*len(novel))/(len(base)+len(novel))-r['overall'])<1e-10
      if r['evaluation']=='matched_test':assert ab>=r['target']-1e-10
      checked+=1
 assert max_prior_delta<1e-8,(dataset,max_prior_delta)
 return {'records':len(rows),'recomputed_points':checked,'shared_target_groups':len(groups),
         'max_prior_result_delta':max_prior_delta,'source_and_protocol_hashes_valid':True}

def verify_euclidean(dataset):
 folder=ROOT/'experiments/kbs_strengthening'/dataset
 manifest=json.loads((folder/'euclidean_manifest.json').read_text())
 assert hashlib.sha256((folder/'kbs_fenec_euclidean_snapshot.py').read_bytes()).hexdigest()==manifest['source_sha256']
 assert hashlib.sha256((ROOT/'experiments/kbs_fenec_euclidean_addendum.md').read_bytes()).hexdigest()==manifest['addendum_sha256']
 checked=0;total=0
 for part in range(3):
  assert (folder/f'p{part}/euclidean_complete.json').exists()
  for order in [0,1]:
   for session in range(1,6):
    prefix=folder/f'p{part}/euclidean_o{order}_t{session}'
    rows=json.loads(prefix.with_suffix('.json').read_text())['records'];total+=len(rows)
    original=json.loads((folder/f'p{part}/o{order}_t{session}.json').read_text())['records']
    with np.load(prefix.with_suffix('.npz')) as z:
     y=z['labels'];base=z['base_classes'];novel=z['novel_classes'];w=np.zeros(len(y))
     for group in [base,novel]:
      for c in group:w[y==c]=1./(len(group)*np.sum(y==c))
     for r in rows:
      m=r['method']
      assert np.allclose(z[m+'_weights'],w)
      assert np.array_equal(z[m+'_base_correct'],np.isin(y,base)&(z[m+'_base_prediction']==y))
      assert np.array_equal(z[m+'_novel_correct'],np.isin(y,novel)&(z[m+'_novel_prediction']==y))
      if r['evaluation']!='fixed_gate':
       target=next(v['target'] for v in original if v['evaluation']==r['evaluation'] and v.get('ratio')==r['ratio'])
       assert abs(target-r['target'])<1e-12
       if not r['feasible']:
        assert 'new' not in r and r['offset'] is None
        assert r['selection_base_ceiling']<target+1e-10
        continue
       assert r['selection_base_accuracy']>=target-1e-10
      margin=z[m+'_margin'];offset=r['offset']
      pick=(offset>margin)|((offset==margin)&(z[m+'_novel_prediction']<z[m+'_base_prediction']))
      a=float(w[z[m+'_base_correct']&~pick].sum());n=float(w[z[m+'_novel_correct']&pick].sum())
      assert abs(a-r['old'])<1e-10 and abs(n-r['new'])<1e-10
      assert abs((a*len(base)+n*len(novel))/(len(base)+len(novel))-r['overall'])<1e-10
      if r['evaluation']=='matched_test':assert a>=r['target']-1e-10
      checked+=1
 assert total==660
 return {'records':total,'recomputed_points':checked,'source_and_addendum_hashes_valid':True}

def verify_vit(dataset):
 parent=ROOT/'experiments/kbs_strengthening'/dataset;folder=parent/'fenec_vit'
 manifest=json.loads((folder/'manifest.json').read_text())
 assert hashlib.sha256((folder/'source_snapshot.py').read_bytes()).hexdigest()==manifest['source_sha256']
 assert hashlib.sha256((ROOT/'experiments/kbs_fenec_vit_correction.md').read_bytes()).hexdigest()==manifest['correction_sha256']
 assert hashlib.sha256((parent/'run_manifest.json').read_bytes()).hexdigest()==manifest['primary_manifest_sha256']
 checked=0;total=0
 for part in range(3):
  assert (folder/f'p{part}/complete.json').exists()
  for variant in ['mahalanobis','euclidean']:
   for order in [0,1]:
    for session in range(1,6):
     prefix=folder/f'p{part}/{variant}_o{order}_t{session}'
     rows=json.loads(prefix.with_suffix('.json').read_text())['records'];total+=len(rows)
     original=json.loads((parent/f'p{part}/o{order}_t{session}.json').read_text())['records']
     with np.load(prefix.with_suffix('.npz')) as z:
      y=z['labels'];base=z['base_classes'];novel=z['novel_classes'];w=np.zeros(len(y))
      for group in [base,novel]:
       for c in group:w[y==c]=1./(len(group)*np.sum(y==c))
      for r in rows:
       m=r['method'];assert np.allclose(z[m+'_weights'],w)
       assert np.array_equal(z[m+'_base_correct'],np.isin(y,base)&(z[m+'_base_prediction']==y))
       assert np.array_equal(z[m+'_novel_correct'],np.isin(y,novel)&(z[m+'_novel_prediction']==y))
       if r['evaluation']!='fixed_gate':
        target=next(v['target'] for v in original if v['evaluation']==r['evaluation'] and v.get('ratio')==r['ratio'])
        assert abs(target-r['target'])<1e-12
        if not r['feasible']:
         assert 'new' not in r and r['offset'] is None
         assert r['selection_base_ceiling']<target+1e-10
         continue
        assert r['selection_base_accuracy']>=target-1e-10
       offset=r['offset'];margin=z[m+'_margin']
       new=(offset>margin)|((offset==margin)&(z[m+'_novel_prediction']<z[m+'_base_prediction']))
       a=float(w[z[m+'_base_correct']&~new].sum());n=float(w[z[m+'_novel_correct']&new].sum())
       assert abs(a-r['old'])<1e-10 and abs(n-r['new'])<1e-10
       assert abs((a*len(base)+n*len(novel))/(len(base)+len(novel))-r['overall'])<1e-10
       if r['evaluation']=='matched_test':assert a>=r['target']-1e-10
       checked+=1
 assert total==1320
 return {'records':total,'recomputed_points':checked,'corrected_vit_source_and_manifest_hashes_valid':True}

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--datasets',nargs='+',default=['cifar','pets','imagenetr']);args=ap.parse_args()
 report={dataset:{'primary':verify(dataset),'euclidean':verify_euclidean(dataset),'corrected_vit':verify_vit(dataset)} for dataset in args.datasets}
 (ROOT/'experiments/kbs_strengthening'/('verification_'+'_'.join(args.datasets)+'.json')).write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
