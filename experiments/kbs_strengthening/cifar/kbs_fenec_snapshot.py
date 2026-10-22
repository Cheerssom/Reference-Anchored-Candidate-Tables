"""FeNeC frozen-feature readout, with explicit few-shot safeguards.

Equations follow Ksiazek et al. and author commit 1575ea019fd583149c47c068f7f5ccc102ea4580.
Independent implementation; third-party source is not bundled in the public code.
"""
import numpy as np
import torch
from kbs_runtime import check_pause

def covariance_inverse(x,gamma1=1.,gamma2=1.):
 """Per-class double shrinkage and correlation normalization from author code."""
 d=x.shape[1]
 if len(x)<2:return torch.eye(d,device=x.device,dtype=x.dtype)
 c=torch.cov(x.T)
 eye=torch.eye(d,device=x.device,dtype=x.dtype)
 for _ in range(2):
  diag=c.diagonal();v1=diag.mean();v2=(c.sum()-diag.sum())/(d*(d-1))
  c=c+gamma1*v1*eye+gamma2*v2*(1-eye)
 std=c.diagonal().clamp_min(1e-20).sqrt()
 c=c/(std[:,None]*std[None,:])
 # Equivalent to the upstream pseudoinverse for positive-definite shrinkage.
 factor,info=torch.linalg.cholesky_ex(c)
 if int(info)==0:return torch.cholesky_inverse(factor)
 return torch.linalg.pinv(c,hermitian=True)

def centroids(x,k,seed=42,max_iter=100):
 """Euclidean k-means++, matching the author algorithm on nonempty clusters."""
 k=min(k,len(x));generator=torch.Generator(device=x.device).manual_seed(seed)
 centers=[x[torch.randint(len(x),(1,),device=x.device,generator=generator)]]
 for _ in range(k-1):
  dist=torch.cdist(x,torch.cat(centers)).min(1).values.square()
  if float(dist.sum())==0:break
  centers.append(x[torch.multinomial(dist,1,generator=generator)])
 c=torch.cat(centers)
 for _ in range(max_iter):
  labels=torch.cdist(x,c).argmin(1)
  nxt=torch.stack([x[labels==j].mean(0) if (labels==j).any() else c[j] for j in range(len(c))])
  if torch.equal(nxt,c):break
  c=nxt
 return c

def fit(x,y,classes,k,gamma1,device,checkpoint=None):
 """Store fitted classes on CPU; move one class to GPU during prediction."""
 from pathlib import Path
 from kbs_runtime import atomic_torch
 checkpoint=Path(checkpoint) if checkpoint else None
 state={}
 if checkpoint and checkpoint.exists():state=torch.load(checkpoint,map_location='cpu',weights_only=False)
 expected={'classes':np.asarray(classes).tolist(),'k':k,'gamma1':gamma1}
 if state:assert state['configuration']==expected
 else:state={'configuration':expected,'classes':{}}
 if checkpoint:
  rowdir=checkpoint.with_suffix('');rowdir.mkdir(exist_ok=True)
 for c in classes:
  check_pause();key=str(c)
  if key in state['classes']:continue
  rowpath=rowdir/(key+'.pt') if checkpoint else None
  if rowpath and rowpath.exists():
   state['classes'][key]=torch.load(rowpath,map_location='cpu',weights_only=False);continue
  xx=x[y==c].to(device).float();assert len(xx)>0
  state['classes'][key]={'centroids':centroids(xx,k).cpu(),'inverse':covariance_inverse(xx,gamma1).cpu(),
                         'samples':len(xx),'single_sample_fallback':len(xx)<2}
  del xx
  if checkpoint:
   atomic_torch(state['classes'][key],rowpath)
   atomic_torch({'configuration':expected,'classes':{}},checkpoint)
 return state

def distances(state,E,classes,device,chunk=128,cache_dir=None):
 """Squared class-specific Mahalanobis distances, keeping VRAM bounded."""
 from pathlib import Path
 from kbs_runtime import atomic_torch
 cache_dir=Path(cache_dir) if cache_dir else None
 if cache_dir:cache_dir.mkdir(exist_ok=True)
 blocks=[];labels=[]
 for c in classes:
  check_pause();row=state['classes'][str(c)]
  cached=cache_dir/(str(c)+'.pt') if cache_dir else None
  if cached and cached.exists():
   block=torch.load(cached,map_location='cpu',weights_only=False)
   assert block.shape==(len(E),len(row['centroids']))
   blocks.append(block);labels.extend([int(c)]*len(row['centroids']));continue
  cent=row['centroids'].to(device);inv=row['inverse'].to(device)
  ans=[];cross=inv@cent.T;constant=(cent@inv*cent).sum(1)
  for start in range(0,len(E),chunk):
   q=E[start:start+chunk].to(device).float()
   dist=(q@inv*q).sum(1)[:,None]-2*q@cross+constant[None,:]
   # Direct quadratic form is used near cancellation, not arbitrary zero distance.
   near=dist<1e-5
   if near.any():
    ix,jx=near.nonzero(as_tuple=True);delta=q[ix]-cent[jx]
    dist[ix,jx]=(delta@inv*delta).sum(1)
   ans.append(dist.clamp_min(1e-12).cpu())
  block=torch.cat(ans)
  if cached:atomic_torch(block,cached)
  blocks.append(block);labels.extend([int(c)]*len(cent))
  del cent,inv
 return torch.cat(blocks,1),np.asarray(labels)

def log_votes(distance,representative_labels,active,neighbors):
 """Global kNN inverse-distance votes; log preserves zero-offset predictions."""
 keep=np.isin(representative_labels,active)
 d=distance[:,keep];lab=torch.as_tensor(representative_labels[keep])
 ix=torch.topk(d,min(neighbors,d.shape[1]),dim=1,largest=False,sorted=False).indices
 cls={int(c):j for j,c in enumerate(active)}
 labels=torch.as_tensor([cls[int(c)] for c in lab])
 votes=torch.zeros(len(d),len(active))
 votes.scatter_add_(1,labels[ix],1./d.gather(1,ix))
 return votes.clamp_min(1e-30).log()
