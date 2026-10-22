"""Resumable author-described Euclidean FeNeC variant on the same data pools."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import fcs_retention_value as rv
import fcs_expansion as core
from kbs_strengthen import load,joined_features,append_records
from kbs_fenec import centroids,log_votes
from kbs_runtime import configure,atomic_json,atomic_npz,check_pause,digest_file

ROOT=Path(__file__).resolve().parents[2]

def representative_distances(x,y,classes,E,k,device):
 c=[];labels=[]
 for label in classes:
  cc=centroids(x[y==label].to(device),k);c.append(cc);labels.extend([int(label)]*len(cc))
 allc=torch.cat(c);d=[]
 for i in range(0,len(E),128):d.append(torch.cdist(E[i:i+128].to(device),allc).clamp_min(1e-12).cpu())
 return torch.cat(d),np.array(labels)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['cifar','pets','imagenetr'],required=True);args=ap.parse_args()
 device=configure();d,entries,_=load(args.dataset)
 parent=ROOT/'experiments/kbs_strengthening'/args.dataset
 config={'source_sha256':digest_file(__file__),'addendum_sha256':digest_file(ROOT/'experiments/kbs_fenec_euclidean_addendum.md')}
 path=parent/'euclidean_manifest.json'
 if path.exists():assert json.loads(path.read_text())==config
 else:atomic_json(config,path);(parent/'kbs_fenec_euclidean_snapshot.py').write_bytes(Path(__file__).read_bytes())
 for p in range(3):
  out=parent/f'p{p}';done=out/'euclidean_complete.json'
  if done.exists():continue
  check_pause();info=json.loads((out/'partition.json').read_text());entry=info['entry']
  base,new,pb,pn=[np.array(entry[k]) for k in ['base','novel','pseudo_base','pseudo_novel']]
  mask=np.isin(d['val_y'],base);ids=np.r_[pb,pn]
  x,y=joined_features(d,pb,pn);best=(-1.,None);trials=[]
  for k in [1,3,5]:
   dist,cl=representative_distances(x,y,ids,d['E_val'][mask],k,device)
   for neighbors in [1,3,5]:
    scores=log_votes(dist,cl,ids,neighbors);c=rv.compact(scores,ids,d['val_y'][mask],pb,pn)
    for offset in rv.OFFSETS:
     value=rv.evaluate(c,offset)['harmonic'];params={'clusters':k,'neighbors':neighbors,'offset':offset,'calibration_harmonic':value}
     trials.append(params)
     if value>best[0]+1e-12:best=(value,params)
  fp=best[1];atomic_json({'selected':fp,'trials':trials},out/'euclidean_selection.json')
  x,y=joined_features(d,base,new);allids=np.r_[base,new]
  E=torch.cat([d['E_test'],d['E_val'][mask]])
  distances,cl=representative_distances(x,y,allids,E,fp['clusters'],device)
  td=distances[:len(d['E_test'])];vd=distances[len(d['E_test']):]
  for order in [0,1]:
   ordered=np.random.default_rng(1000+p+100*order).permutation(new)
   for session in range(1,6):
    check_pause();file=out/f'euclidean_o{order}_t{session}.json'
    if file.exists():continue
    novel=ordered[:len(new)*session//5];active=np.r_[base,novel]
    ts=log_votes(td,cl,active,fp['neighbors']);vs=log_votes(vd,cl,active,fp['neighbors'])
    result={'records':[]};archive={};common={'part':p,'order':order,'session':session,'added':len(novel)}
    for method,offset in [('fenec_euclidean_native',0.),('fenec_euclidean_calibrated',fp['offset'])]:
     append_records(result,archive,method,ts,vs,[fp['clusters'],fp['neighbors'],offset],active,
       d['test_y'],d['val_y'][mask],base,novel,common,info['reference_closed_test'],info['reference_closed_validation'])
    atomic_npz(out/f'euclidean_o{order}_t{session}.npz',active_classes=active,base_classes=base,novel_classes=novel,labels=d['test_y'],**archive)
    atomic_json(result,file)
  atomic_json({'complete':True,'part':p},done);print('EUCLIDEAN COMPLETE',args.dataset,p,fp,flush=True)

if __name__=='__main__':main()
