"""Corrected author-described ViT shrinkage and stronger representative budgets."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from kbs_runtime import configure,atomic_json,atomic_torch,atomic_npz,digest_file,check_pause
from kbs_fenec import centroids,log_votes,distances
from kbs_strengthen import load,joined_features,append_records
import fcs_retention_value as rv
ROOT=Path(__file__).resolve().parents[2]
GAMMAS=[(1.,1.),(6.12,8.10),(9.98,0.)]

def inverse_once(x,g1,g2):
 d=x.shape[1];eye=torch.eye(d,device=x.device)
 if len(x)<2:return eye
 c=torch.cov(x.T);diag=c.diagonal();v1=diag.mean();v2=(c.sum()-diag.sum())/(d*(d-1))
 c=c+g1*v1*eye+g2*v2*(1-eye)
 std=c.diagonal().clamp_min(1e-20).sqrt();c=c/(std[:,None]*std[None,:])
 factor,info=torch.linalg.cholesky_ex(c)
 return torch.cholesky_inverse(factor) if int(info)==0 else torch.linalg.pinv(c,hermitian=True)

def state(x,y,classes,k,gamma,device,folder):
 folder.mkdir(exist_ok=True);cdir=folder/f'centers_k{k}';cdir.mkdir(exist_ok=True)
 if gamma is not None:
  idir=folder/f'inverse_{gamma[0]:g}_{gamma[1]:g}';idir.mkdir(exist_ok=True)
 result={'classes':{}}
 for c in classes:
  check_pause();cp=cdir/f'{c}.pt';xx=x[y==c].to(device)
  if cp.exists():center=torch.load(cp,map_location='cpu',weights_only=False)
  else:center=centroids(xx,k).cpu();atomic_torch(center,cp)
  if gamma is not None:
   ip=idir/f'{c}.pt'
   if ip.exists():inv=torch.load(ip,map_location='cpu',weights_only=False)
   else:inv=inverse_once(xx,*gamma).cpu();atomic_torch(inv,ip)
  else:inv=None
  result['classes'][str(c)]={'centroids':center,'inverse':inv}
 return result

def distance(s,E,classes,device,folder,euclidean=False):
 if not euclidean:return distances(s,E,classes,device,cache_dir=folder)
 folder.mkdir(exist_ok=True);blocks=[];labels=[]
 for c in classes:
  check_pause();path=folder/f'{c}.pt';center=s['classes'][str(c)]['centroids']
  if path.exists():d=torch.load(path,map_location='cpu',weights_only=False)
  else:
   cc=center.to(device);d=torch.cat([torch.cdist(E[i:i+128].to(device),cc).clamp_min(1e-12).cpu() for i in range(0,len(E),128)])
   atomic_torch(d,path)
  blocks.append(d);labels.extend([int(c)]*len(center))
 return torch.cat(blocks,1),np.array(labels)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['cifar','pets','imagenetr'],required=True);args=ap.parse_args()
 device=configure();d,_,_=load(args.dataset);parent=ROOT/'experiments/kbs_strengthening'/args.dataset
 root=parent/'fenec_vit';root.mkdir(exist_ok=True)
 manifest={'source_sha256':digest_file(__file__),'correction_sha256':digest_file(ROOT/'experiments/kbs_fenec_vit_correction.md'),
           'primary_manifest_sha256':digest_file(parent/'run_manifest.json')}
 if (root/'manifest.json').exists():assert json.loads((root/'manifest.json').read_text())==manifest
 else:atomic_json(manifest,root/'manifest.json');(root/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
 for p in range(3):
  out=root/f'p{p}';out.mkdir(exist_ok=True)
  if (out/'complete.json').exists():continue
  info=json.loads((parent/f'p{p}/partition.json').read_text());entry=info['entry']
  base,new,pb,pn=[np.array(entry[k]) for k in ['base','novel','pseudo_base','pseudo_novel']]
  mask=np.isin(d['val_y'],base);ids=np.r_[pb,pn];x,y=joined_features(d,pb,pn)
  selected={}
  for variant in ['mahalanobis','euclidean']:
   choice=out/(variant+'_selection.json')
   if choice.exists():selected[variant]=json.loads(choice.read_text())['selected'];continue
   best=(-1.,None);trials=[]
   for gamma in (GAMMAS if variant=='mahalanobis' else [None]):
    for k in [1,5,25]:
     check_pause();tag=f"cal_{variant}_{k}_{gamma}"
     s=state(x,y,ids,k,gamma,device,out/'cal_models')
     dist,cl=distance(s,d['E_val'][mask],ids,device,out/tag,variant=='euclidean');del s
     for neighbors in [1,3,5]:
      scores=log_votes(dist,cl,ids,neighbors);c=rv.compact(scores,ids,d['val_y'][mask],pb,pn)
      for offset in rv.OFFSETS:
       h=rv.evaluate(c,offset)['harmonic'];params={'clusters':k,'neighbors':neighbors,'gamma':gamma,'offset':offset,'calibration_harmonic':h}
       trials.append(params)
       if h>best[0]+1e-12:best=(h,params)
     print('VIT BASELINE calibration',args.dataset,p,variant,k,gamma,'best',round(best[0],4),flush=True)
   selected[variant]=best[1];atomic_json({'selected':best[1],'trials':trials},choice)
  x,y=joined_features(d,base,new);allids=np.r_[base,new];E=torch.cat([d['E_test'],d['E_val'][mask]])
  for variant,fp in selected.items():
   s=state(x,y,allids,fp['clusters'],fp['gamma'],device,out/('final_'+variant))
   dist,cl=distance(s,E,allids,device,out/('query_'+variant),variant=='euclidean');del s
   td=dist[:len(d['E_test'])];vd=dist[len(d['E_test']):]
   for order in [0,1]:
    ordered=np.random.default_rng(1000+p+100*order).permutation(new)
    for session in range(1,6):
     check_pause();path=out/f'{variant}_o{order}_t{session}.json'
     if path.exists():continue
     novel=ordered[:len(new)*session//5];active=np.r_[base,novel]
     ts=log_votes(td,cl,active,fp['neighbors']);vs=log_votes(vd,cl,active,fp['neighbors'])
     result={'records':[]};archive={};common={'part':p,'order':order,'session':session,'added':len(novel)}
     stem='fenec' if variant=='mahalanobis' else 'fenec_euclidean'
     for method,offset in [(stem+'_native',0.),(stem+'_calibrated',fp['offset'])]:
      append_records(result,archive,method,ts,vs,[fp['clusters'],fp['neighbors'],offset],active,
       d['test_y'],d['val_y'][mask],base,novel,common,info['reference_closed_test'],info['reference_closed_validation'])
     atomic_npz(path.with_suffix('.npz'),active_classes=active,base_classes=base,novel_classes=novel,labels=d['test_y'],**archive)
     atomic_json(result,path)
  atomic_json({'complete':True,'part':p,'selected':selected},out/'complete.json')
  print('VIT BASELINE COMPLETE',args.dataset,p,selected,flush=True)

if __name__=='__main__':main()
