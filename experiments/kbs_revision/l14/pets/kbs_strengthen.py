"""Resumable ImageNet-R extension and recent FeNeC comparisons.

Run the same command to resume. Create experiments/kbs_strengthening/PAUSE to
stop cooperatively between saved units; remove it before resuming.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import fcs_expansion as core
import fcs_retention_value as rv
from fcs_review_experiments import predict
from kbs_runtime import configure,atomic_json,atomic_torch,atomic_npz,check_pause,digest_file
import kbs_fenec as fenec

ROOT=Path(__file__).resolve().parents[2]

def load(dataset):
 if dataset=='cifar':
  from lt_dataset import make_lt_datasets
  from train import split_train_val
  training,support,_,testing,_=make_lt_datasets(rho=100);fit,val=split_train_val(training)
  folder=ROOT/'experiments/fcs';path=ROOT/'src/expand_cache.pt'
  if not path.exists():path=folder/'frozen_cifar.pt'
  d=torch.load(path,map_location='cpu',weights_only=False)
  if 'E_support' not in d:d['E_support']=torch.load(folder/'frozen_support.pt',weights_only=False)['features']
  for key,y in zip(['fit','val','support','test'],[fit.fine,val.fine,support.fine,testing.fine]):d[key+'_y']=y
 elif dataset=='pets':
  folder=ROOT/'experiments/fcs_pets';path=folder/'pets_frozen.pt';d=torch.load(path,map_location='cpu',weights_only=False)
 else:
  folder=ROOT/'experiments/kbs_imagenetr';path=folder/'frozen_imagenetr.pt';d=torch.load(path,map_location='cpu',weights_only=False)
 d={k:v for k,v in d.items() if k.startswith('E_') or k.endswith('_y') or k in ['T100','meta']}
 entries=json.loads((folder/'manifest.json').read_text())['partitions'] if dataset!='imagenetr' else None
 return d,entries,path

def make_entries(d,device,out):
 path=out/'partitions.json'
 if path.exists():return json.loads(path.read_text())
 buckets=np.array(['head']*67+['medium']*66+['tail']*67)
 entries=[];Ef,Ev=d['E_fit'].to(device),d['E_val'].to(device);yf,yv=d['fit_y'],d['val_y']
 for p in [0,1,2]:
  check_pause();base,new,_=core.partition(buckets,p,n_base=100,n_negative=0)
  pb=np.sort(np.random.default_rng(900+p).choice(base,50,replace=False));pn=np.setdiff1d(base,pb)
  _,lam=core.select_ridge(Ef,yf,Ev,yv,base);_,plam=core.select_ridge(Ef,yf,Ev,yv,pb)
  entries.append({'seed':p,'base':base.tolist(),'novel':new.tolist(),'pseudo_base':pb.tolist(),
    'pseudo_novel':pn.tolist(),'ridge_lambda':lam,'pseudo_ridge_lambda':plam})
 atomic_json(entries,path);return entries

def joined_features(d,base,new):
 mask=np.isin(d['fit_y'],base);si=core.support_rows(d['support_y'],new,5,7)
 return torch.cat([d['E_fit'][mask],d['E_support'][si]]),np.r_[d['fit_y'][mask],d['support_y'][si]]

def fenec_scores(x,y,classes,E,k,gamma,device,path):
 state=fenec.fit(x,y,classes,k,gamma,device,path)
 return fenec.distances(state,E,classes,device,cache_dir=path.with_name(path.stem+'_query'))

def choose_fenec(d,pb,pn,device,out):
 path=out/'fenec_selection.json'
 if path.exists():return json.loads(path.read_text())
 ids=np.r_[pb,pn];x,y=joined_features(d,pb,pn)
 mask=np.isin(d['val_y'],ids);E=d['E_val'][mask].to(device);labels=d['val_y'][mask]
 best=(-1.,None);trials=[]
 for gamma in [1.,10.]:
  for k in [1,3,5]:
   check_pause();name=f'cal_k{k}_g{gamma:g}';saved=out/(name+'_dist.pt')
   if saved.exists():v=torch.load(saved,map_location='cpu',weights_only=False);dist,cl=v['distances'],v['labels']
   else:
    dist,cl=fenec_scores(x,y,ids,E,k,gamma,device,out/(name+'.pt'))
    atomic_torch({'distances':dist,'labels':cl},saved)
   for neighbors in [1,3,5]:
    scores=fenec.log_votes(dist,cl,ids,neighbors)
    c=rv.compact(scores,ids,labels,pb,pn)
    for offset in rv.OFFSETS:
     h=rv.evaluate(c,offset)['harmonic']
     params={'clusters':k,'gamma1':gamma,'neighbors':neighbors,'offset':offset,'calibration_harmonic':h}
     trials.append(params)
     if h>best[0]+1e-12:best=(h,params)
   print('FeNeC calibration',out.name,'k',k,'gamma',gamma,'best',round(best[0],4),flush=True)
 atomic_json({'selected':best[1],'trials':trials},path)
 return {'selected':best[1],'trials':trials}

def append_records(result,archive,method,scores_test,scores_val,params,ids,yt,yv,base,novel,common,ref_test,ref_val):
 tc=rv.compact(scores_test,ids,yt,base,novel);vc=rv.compact(scores_val,ids,yv,base,novel)
 for key in ['margin','base_prediction','novel_prediction','base_correct','novel_correct','weights']:
  archive[method+'_'+key]=tc[key]
 offset=float(params[-1])
 result['records'].append({**common,'method':method,'parameters':params,'evaluation':'fixed_gate',
                          'offset':offset,**rv.evaluate(tc,offset)})
 for ratio in rv.RATIOS:
  for evaluation,c,target in [('matched_test',tc,ratio*ref_test),('base_validation',vc,ratio*ref_val)]:
   threshold,ceiling=rv.threshold(c,target)
   row={**common,'method':method,'parameters':params,'evaluation':evaluation,'ratio':ratio,
     'target':target,'selection_base_ceiling':ceiling,'feasible':threshold is not None,'offset':threshold}
   if threshold is not None:
    row.update(rv.evaluate(tc,threshold));row['selection_base_accuracy']=rv.evaluate(c,threshold)['old']
    row['selection_overshoot']=row['selection_base_accuracy']-target
   result['records'].append(row)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['cifar','pets','imagenetr'],required=True)
 ap.add_argument('--part',type=int,choices=[0,1,2]);args=ap.parse_args();device=configure()
 out=ROOT/'experiments/kbs_strengthening'/args.dataset;out.mkdir(parents=True,exist_ok=True)
 dependencies={p.name:digest_file(p) for p in [Path(__file__),Path(fenec.__file__),Path(rv.__file__),Path(core.__file__)]}
 protocol_hash=digest_file(ROOT/'experiments/kbs_strengthening_protocol.md')
 check_pause();d,entries,cachepath=load(args.dataset)
 manifest=out/'run_manifest.json';configuration={'dependencies':dependencies,'protocol_sha256':protocol_hash,'dataset':args.dataset,
   'cache_sha256':digest_file(cachepath)}
 if manifest.exists():assert json.loads(manifest.read_text())==configuration,'Source/protocol changed; preserve the old run and use a new run directory.'
 else:
  atomic_json(configuration,manifest)
  for source in [Path(__file__),Path(fenec.__file__)]: (out/(source.stem+'_snapshot.py')).write_bytes(source.read_bytes())
 if entries is None:entries=make_entries(d,device,out)
 Ef,Ev,Es,Et,T=[d[k].to(device) for k in ['E_fit','E_val','E_support','E_test','T100']]
 yf,yv,ys,yt=[d[k+'_y'] for k in ['fit','val','support','test']]
 start=time.monotonic()
 for entry in entries:
  p=entry['seed']
  if args.part is not None and args.part!=p:continue
  po=out/f'p{p}';po.mkdir(exist_ok=True);done=po/'complete.json'
  if done.exists():print('Already complete:',args.dataset,p,flush=True);continue
  base,new,pb,pn=[np.array(entry[k]) for k in ['base','novel','pseudo_base','pseudo_novel']]
  ids=np.r_[base,new];mask=np.isin(yv,base)
  W=core.fit_ridge(Ef,yf,base,entry['ridge_lambda']);Wp=core.fit_ridge(Ef,yf,pb,entry['pseudo_ridge_lambda'])
  Pb=core.means(Ef,yf,base);Ppb=core.means(Ef,yf,pb)
  si=core.support_rows(ys,new,5,7);psi=core.support_rows(ys,pn,5,7)
  Ppn=core.means(Es[psi],ys[psi],pn)
  cal=rv.arrays(Ev[mask],T[np.r_[pb,pn]],Ppb,Ppn,Wp)
  parameters={m:rv.calibrate(cal,m,np.r_[pb,pn],yv[mask],pb,pn) for m in ['retained','no_learned_tuned','anchored_tp']}
  l,u,offset=parameters['retained'];parameters['removed_fixed']=(0.,u,offset)
  def closed(E,y):
   score=core.z(E@T[base].T)+l*core.z(core.linear(E,W));pred=predict(score,base)
   return float(np.mean([np.mean(pred[y==c]==c) for c in base]))
  ref_test=closed(Et,yt);ref_val=closed(Ev[mask],yv[mask])
  selection=choose_fenec(d,pb,pn,device,po);fp=selection['selected']
  x,y=joined_features(d,base,new)
  saved=po/'final_distances.pt'
  if saved.exists():cache=torch.load(saved,map_location='cpu',weights_only=False);td,vd,cl=cache['test'],cache['val'],cache['labels']
  else:
   state=fenec.fit(x,y,ids,fp['clusters'],fp['gamma1'],device,po/'final_model.pt')
   td,cl=fenec.distances(state,Et,ids,device,cache_dir=po/'test_distances')
   vd,_=fenec.distances(state,Ev[mask],ids,device,cache_dir=po/'validation_distances')
   atomic_torch({'test':td,'val':vd,'labels':cl},saved)
  atomic_json({'part':p,'entry':entry,'table_parameters':parameters,'fenec':fp,'reference_closed_test':ref_test,
               'reference_closed_validation':ref_val,'cache_sha256':digest_file(cachepath)},po/'partition.json')
  for order in [0,1]:
   ordered=np.random.default_rng(1000+p+100*order).permutation(new)
   for session in range(1,6):
    check_pause();path=po/f'o{order}_t{session}.json'
    if path.exists():continue
    novel=ordered[:len(new)*session//5];active=np.r_[base,novel]
    Pn=core.means(Es[si],ys[si],novel)
    ta=rv.arrays(Et,T[active],Pb,Pn,W);va=rv.arrays(Ev[mask],T[active],Pb,Pn,W)
    result={'records':[]};archive={};common={'part':p,'order':order,'session':session,'added':len(novel)}
    for method in rv.METHODS:
     append_records(result,archive,method,rv.score(ta,method,parameters[method]),rv.score(va,method,parameters[method]),
                    parameters[method],active,yt,yv[mask],base,novel,common,ref_test,ref_val)
    ts=fenec.log_votes(td,cl,active,fp['neighbors']);vs=fenec.log_votes(vd,cl,active,fp['neighbors'])
    for name,offset in [('fenec_native',0.),('fenec_calibrated',fp['offset'])]:
     append_records(result,archive,name,ts,vs,[fp['clusters'],fp['neighbors'],fp['gamma1'],offset],
                    active,yt,yv[mask],base,novel,common,ref_test,ref_val)
    atomic_npz(po/f'o{order}_t{session}.npz',active_classes=active,base_classes=base,novel_classes=novel,labels=yt,**archive)
    atomic_json(result,path)
    print('SAVED',args.dataset,p,order,session,flush=True)
  atomic_json({'part':p,'complete':True,'elapsed_seconds':time.monotonic()-start},done)
  print('PART COMPLETE',args.dataset,p,'seconds',round(time.monotonic()-start),flush=True)
 print('RUN COMPLETE',args.dataset,flush=True)

if __name__=='__main__':main()
