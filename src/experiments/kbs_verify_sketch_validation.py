"""Independent natural-shift policy and identity reconstruction."""
from pathlib import Path
import json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from kbs_runtime import atomic_json,digest_file
from kbs_verify_reviewer_revision import verify_rows
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/kbs_sketch_validation'
def main():
 start=time.perf_counter();assert json.loads((OUT/'evaluation_complete.json').read_text())['complete']
 identity=json.loads((OUT/'identity_split.json').read_text());rows=identity['records'];source=json.loads((OUT/'source_identity_hashes.json').read_text());budget=json.loads((OUT/'calibration_budget.json').read_text())
 assert len(identity['class_counts'])==200 and all(r['unique']>=2 for r in identity['class_counts'])
 assert not ({r['raw_sha256'] for r in rows}&set(source['raw_hashes']))
 assert not ({r['pixel224_sha256'] for r in rows}&set(source['pixel224_hashes']))
 for key in ['raw_sha256','pixel224_sha256','dihedral224_sha256']:assert len({r[key] for r in rows})==len(rows)
 cal=set(budget['target_indices']);test={i for i,r in enumerate(rows) if r['role']=='test'};assert not cal&test and all(rows[i]['role']=='cal' for i in cal)
 count=tests=0
 for enc in ['l14','b32']:
  folder=OUT/'evaluation'/enc;meta=json.loads((folder/'manifest.json').read_text())
  for name,sha in meta['inputs'].items():assert digest_file(ROOT/name)==sha
  feature=OUT/f'features/{enc}/verification.json';fv=json.loads(feature.read_text());assert fv['min_source_witness_cosine']>.999 and fv['manifest_sha256']==digest_file(OUT/'identity_split.json')
  assert digest_file(OUT/f'features/{enc}/features.pt')==fv['feature_sha256']
  for p in range(10):
   for t in range(1,6):
    path=folder/f'p{p}/o0_t{t}.json';r=json.loads(path.read_text());raw=path.with_suffix('.npz');assert digest_file(raw)==r['raw_sha256']
    sp=ROOT/f'experiments/kbs_independent_risk/{enc}/imagenetr/p{p}/selection.json';hp=ROOT/f'experiments/kbs_reviewer_revision/readouts/{enc}/imagenetr/p{p}/selection.json'
    assert digest_file(sp)==r['source_selection_sha256'] and digest_file(hp)==r['hycal_selection_sha256']
    with np.load(raw) as z,np.load(ROOT/f'experiments/kbs_independent_risk/{enc}/imagenetr/p{p}/o0_t{t}.npz') as old,np.load(ROOT/f'experiments/kbs_reviewer_revision/readouts/{enc}/imagenetr/p{p}/o0_t{t}.npz') as hy:
     assert np.array_equal(z['base_classes'],old['base_classes']) and np.array_equal(z['novel_classes'],old['novel_classes'])
     for method in ['retained','anchored_tp','hycal']:assert np.array_equal(z[method+'_candidates'],hy['hycal_candidates'] if method=='hycal' else old[method+'_candidates'])
     for name in ['source_cal','target_cal']:
      selected=[v for v in r['records'] if v.get('calibration')==name]
      n,k=verify_rows(selected,z,z,name+'_','test_',z[name+'_labels'],z['test_labels'],z['base_classes'],z['novel_classes'],z[name+'_reference_correct'],z['test_reference_correct'],z);count+=n;tests+=k
     for c in z['base_classes']:assert np.sum(z['source_cal_labels']==c)==np.sum(z['target_cal_labels']==c)
     selected=[v for v in r['records'] if v['policy']=='matched_test']
     n,k=verify_rows(selected,z,z,'target_cal_','test_',z['target_cal_labels'],z['test_labels'],z['base_classes'],z['novel_classes'],z['target_cal_reference_correct'],z['test_reference_correct'],z);count+=n
  print('SKETCH VERIFIED',enc,flush=True)
 assert count==2700
 atomic_json(dict(complete=True,metric_records=count,reconstructed_threshold_tests=tests,mapped_classes=200,unique_target_images=len(rows),excluded_target_images=len(identity['excluded']),source_image_paths_checked=source['source_paths'],source_target_raw_and_pixel_overlap=0,target_dihedral_groups_disjoint=True,equal_source_target_class_calibration_counts=True,sampling_scope='exact hash separation; residual near-duplicate dependence and pretraining exposure unknown',seconds=time.perf_counter()-start),OUT/'verification.json')
 print('SKETCH VERIFICATION COMPLETE',count,tests,flush=True)
if __name__=='__main__':main()
