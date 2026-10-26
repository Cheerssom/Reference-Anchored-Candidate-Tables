"""Frozen-source natural-shift validation with equal calibration counts."""
from pathlib import Path
import argparse,gc,json,sys,time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fcs_expansion as core
import fcs_retention_value as rv
import kbs_reviewer_revision as previous
import kbs_calibration_fairness as fairness
import kbs_risk_control as risk
from kbs_runtime import configure,check_pause,atomic_json,atomic_npz,digest_file
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/kbs_sketch_validation';METHODS=['retained','anchored_tp','hycal']

def prepare_budget():
 identity=OUT/'identity_split.json';value=json.loads(identity.read_text());pool=ROOT/'experiments/kbs_calibration_budget/pools/imagenetr.json';source=json.loads(pool.read_text())['rows']
 config=dict(identity_split_sha256=digest_file(identity),source_pool_sha256=digest_file(pool),protocol_sha256=digest_file(ROOT/'experiments/kbs_sketch_execution_protocol_v1.md'),source_sha256=digest_file(Path(__file__)))
 path=OUT/'calibration_budget.json'
 if path.exists():old=json.loads(path.read_text());assert old['inputs']==config;return old
 rng=np.random.default_rng(27102026);si=[];ti=[];counts=[]
 for c in range(200):
  a=[i for i,r in enumerate(source) if r['class_id']==c];b=[i for i,r in enumerate(value['records']) if r['class_id']==c and r['role']=='cal'];n=min(len(a),len(b));assert n>=1
  si.extend(rng.permutation(a)[:n].tolist());ti.extend(rng.permutation(b)[:n].tolist());counts.append(dict(class_id=c,count=n,available_source=len(a),reserved_target=len(b)))
 result=dict(inputs=config,seed=27102026,source_indices=sorted(si),target_indices=sorted(ti),class_counts=counts);atomic_json(result,path);return result

@torch.no_grad()
def run(enc,device):
 assert json.loads((OUT/'features_complete.json').read_text())['complete'];identity=json.loads((OUT/'identity_split.json').read_text());budget=prepare_budget()
 folder=OUT/'evaluation'/enc;folder.mkdir(parents=True,exist_ok=True)
 paths=[Path(__file__),ROOT/'experiments/kbs_sketch_execution_protocol_v1.md',OUT/'identity_split.json',OUT/'calibration_budget.json',OUT/f'features/{enc}/features.pt']
 paths += [ROOT/('experiments/kbs_imagenetr/frozen_imagenetr.pt' if enc=='l14' else 'experiments/kbs_revision/b32_features/imagenetr/frozen.pt'),ROOT/f'experiments/kbs_calibration_budget/features/{enc}/imagenetr/extra.pt',ROOT/f'experiments/kbs_revision/{enc}/imagenetr/partitions.json']
 paths += [Path(m.__file__) for m in [core,rv,previous,fairness,risk,previous.readouts]]
 config=dict(inputs={p.relative_to(ROOT).as_posix():digest_file(p) for p in paths});mf=folder/'manifest.json'
 if mf.exists():assert json.loads(mf.read_text())==config
 else:atomic_json(config,mf)
 if (folder/'complete.json').exists():return
 d,cache=previous.load_data('imagenetr',enc);target=torch.load(OUT/f'features/{enc}/features.pt',map_location='cpu',weights_only=False)
 si=np.array(budget['source_indices']);ti=np.array(budget['target_indices']);test=np.array([i for i,r in enumerate(identity['records']) if r['role']=='test']);assert not set(ti)&set(test)
 all_y=target['labels'];assert np.array_equal(all_y,np.array([r['class_id'] for r in identity['records']]))
 # HyCal uses source-only fit/support statistics, source-tune and source-cal queries,
 # and target-query scores. No target labels enter statistics or score selection.
 hycal_data={k:d[k] for k in ['E_fit','fit_y','E_support','support_y','T','E_tune']};hycal_data['E_cal']=d['E_cal'][si];hycal_data['E_test']=target['features']
 hycal=previous.prepare_hycal(hycal_data,folder,device)
 entries=json.loads((ROOT/f'experiments/kbs_revision/{enc}/imagenetr/partitions.json').read_text())
 Ef,Es,T=[d[k].to(device) for k in ['E_fit','E_support','T']];yf,ys=d['fit_y'],d['support_y']
 for entry in entries:
  check_pause();p=entry['seed'];po=folder/f'p{p}';po.mkdir(exist_ok=True)
  if (po/'complete.json').exists():continue
  source_selection=ROOT/f'experiments/kbs_independent_risk/{enc}/imagenetr/p{p}/selection.json';sel=json.loads(source_selection.read_text())
  hycal_selection=ROOT/f'experiments/kbs_reviewer_revision/readouts/{enc}/imagenetr/p{p}/selection.json';hs=json.loads(hycal_selection.read_text())
  base=np.array(entry['base']);new=np.array(entry['novel']);W=core.fit_ridge(Ef,yf,base,sel['ridge_lambda']);Pb=core.means(Ef,yf,base)
  source_y=d['cal_y'][si];source_mask=np.isin(source_y,base);target_y=all_y[ti];target_mask=np.isin(target_y,base)
  queries={'source_cal':d['E_cal'][si][source_mask].to(device),'target_cal':target['features'][ti][target_mask].to(device),'test':target['features'][test].to(device)}
  labels={'source_cal':source_y[source_mask],'target_cal':target_y[target_mask],'test':all_y[test]}
  assert all(np.sum(labels['source_cal']==c)==np.sum(labels['target_cal']==c) for c in base)
  a=sel['parameters']['retained'][0]
  def ref(E,y):
   s=core.z(E@T[base].T)+a*core.z(core.linear(E,W));return base[s.argmax(1).cpu().numpy()]==y
  references={key:ref(q,labels[key]) for key,q in queries.items()};closed=float(np.mean([references['test'][labels['test']==c].mean() for c in base]))
  ordered=np.random.default_rng(1000+p).permutation(new);support=np.array(sel['support_indices'])
  for session in range(1,6):
   check_pause();dest=po/f'o0_t{session}.json'
   if dest.exists():continue
   novel=ordered[:len(new)*session//5];ids=np.r_[base,novel];Pn=core.means(Es[support],ys[support],novel)
   systems={};archive=dict(base_classes=base,novel_classes=novel,test_labels=labels['test'],test_reference_correct=references['test']);records=[]
   raw_hycal_source=hycal(hs['hycal']['regularization'],'cal',base,novel)[source_mask]
   raw_hycal_target=hycal(hs['hycal']['regularization'],'test',base,novel)
   hys={'source_cal':raw_hycal_source,'target_cal':raw_hycal_target[ti][target_mask],'test':raw_hycal_target[test]}
   for key,q in queries.items():
    comp=rv.arrays(q,T[ids],Pb,Pn,W)
    for method in METHODS:
     score=hys[key] if method=='hycal' else rv.score(comp,method,sel['parameters'][method]);c=rv.compact(score,ids,labels[key],base,novel);systems[key,method]=c
     for name in ['margin','base_prediction','novel_prediction']:archive[key+'_'+method+'_'+name]=c[name] if name=='margin' else c[name].astype(np.int16)
    if key!='test':archive[key+'_labels']=labels[key];archive[key+'_reference_correct']=references[key]
   old_candidates=ROOT/f'experiments/kbs_independent_risk/{enc}/imagenetr/p{p}/o0_t{session}.npz';hy_candidates=ROOT/f'experiments/kbs_reviewer_revision/readouts/{enc}/imagenetr/p{p}/o0_t{session}.npz'
   with np.load(old_candidates) as oc,np.load(hy_candidates) as hc:
    for method in METHODS:
     candidates=hc['hycal_candidates'] if method=='hycal' else oc[method+'_candidates'];archive[method+'_candidates']=candidates
     tc=systems['test',method];common=dict(encoder=enc,dataset='imagenet_sketch_r200',part=p,order=0,session=session,method=method,reference_closed_test=closed)
     for key in ['source_cal','target_cal']:
      cc=systems[key,method];weights=risk.weights(labels[key],base)
      for policy in ['protection','paired_accuracy']:
       result=fairness.evaluate_policy(cc,references[key],weights,candidates,tc,references['test'],closed,policy)
       records.append(dict(**common,calibration=key,calibration_queries=len(labels[key]),**result))
     for ratio in [.80,.85,.90,.95,.99]:
      off,ceiling=rv.threshold(tc,ratio*closed);records.append(dict(**common,policy='matched_test',retention=ratio,offset=off,feasible=off is not None,ceiling=ceiling,**(rv.evaluate(tc,off) if off is not None else {})))
   raw=dest.with_suffix('.npz');atomic_npz(raw,**archive)
   atomic_json(dict(source_selection_sha256=digest_file(source_selection),hycal_selection_sha256=digest_file(hycal_selection),source_candidates_sha256=digest_file(old_candidates),hycal_candidates_sha256=digest_file(hy_candidates),raw_sha256=digest_file(raw),records=records),dest)
  atomic_json(dict(complete=True,part=p,reference_closed_test=closed),po/'complete.json');print('SKETCH EVALUATION',enc,p,flush=True)
 atomic_json(dict(complete=True,partitions=10),folder/'complete.json')
 del hycal,d,target,Ef,Es,T;gc.collect();torch.cuda.empty_cache()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare-budget',action='store_true');args=ap.parse_args()
 if args.prepare_budget:prepare_budget();return
 device=configure()
 for enc in ['l14','b32']:run(enc,device)
 atomic_json(dict(complete=True,encoders=2,partitions=20),OUT/'evaluation_complete.json')
if __name__=='__main__':main()
