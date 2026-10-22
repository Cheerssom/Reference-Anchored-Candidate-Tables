"""Ten class partitions, two encoders, simple fusion controls and policies."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fcs_expansion as core
import fcs_retention_value as rv
import kbs_depth as depth
import kbs_strengthen as prior
from kbs_runtime import configure,check_pause,atomic_json,digest_file

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/kbs_revision'
METHODS=['retained','removed_fixed','no_learned_tuned','anchored_tp','raw','fixed_temperature','tip_sum_calibrated','tip_mean_calibrated']


def temperatures(E,Tbase,Pbase,W):
    def scale(x):return float(x.std(1,correction=0).mean().clamp_min(1e-6))
    return [scale(E@Tbase.T),scale(core.linear(E,W)),scale(E@Pbase.T)]


def simple_arrays(E,T,Pb,Pn,W,scales):
    return {'text':E@T.T/scales[0],'learned':core.linear(E,W)/scales[1],
            'prototype':E@torch.cat([Pb,Pn]).T/scales[2],'nb':len(Pb)}


def entries_for(d,dataset,original,device):
    if dataset=='imagenetr':buckets=np.array(['head']*67+['medium']*66+['tail']*67);nbase,nneg=100,0
    elif dataset=='cifar':
        from lt_dataset import make_lt_datasets
        _,_,_,_,meta=make_lt_datasets(rho=100);buckets=meta['buckets'];nbase,nneg=40,20
    else:buckets=d['meta']['buckets'];nbase,nneg=15,7
    entries=[];seen=set()
    for p in range(10):
        if p<3:base,new,pb,pn=[np.array(original[p][k]) for k in ['base','novel','pseudo_base','pseudo_novel']]
        else:
            base,new,_=core.partition(buckets,p,nbase,nneg)
            if dataset=='imagenetr':pb=np.sort(np.random.default_rng(900+p).choice(base,len(base)//2,replace=False));pn=np.setdiff1d(base,pb)
            else:
                pseudo=np.random.default_rng(100+p).permutation(base);pb=np.sort(pseudo[:len(base)//2]);pn=np.sort(pseudo[len(base)//2:])
        assert tuple(base) not in seen;seen.add(tuple(base))
        entries.append({'seed':p,'base':base.tolist(),'novel':new.tolist(),'pseudo_base':pb.tolist(),'pseudo_novel':pn.tolist()})
    return entries


def load(dataset,encoder):
    if dataset=='imagenetr':original=json.loads((ROOT/'experiments/kbs_strengthening/imagenetr/partitions.json').read_text())
    else:original=json.loads((ROOT/'experiments'/('fcs' if dataset=='cifar' else 'fcs_pets')/'manifest.json').read_text())['partitions']
    if encoder=='l14':d,_,cache=prior.load(dataset)
    else:
        cache=OUT/'b32_features'/dataset/'frozen.pt'
        assert (cache.parent/'verification.json').exists(),'Extract and validate the B/32 cache first.'
        d=torch.load(cache,map_location='cpu',weights_only=False)
    return d,original,cache


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['cifar','pets','imagenetr'],required=True)
    ap.add_argument('--encoder',choices=['l14','b32'],required=True);args=ap.parse_args()
    device=configure();check_pause();start=time.perf_counter();d,original,cache=load(args.dataset,args.encoder)
    out=OUT/args.encoder/args.dataset;out.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),Path(core.__file__),Path(rv.__file__),Path(depth.__file__),Path(prior.__file__),ROOT/'src/experiments/kbs_runtime.py']
    config={'dataset':args.dataset,'encoder':args.encoder,'cache_sha256':digest_file(cache),
            'sources':{p.name:digest_file(p) for p in sources},'protocol_sha256':digest_file(ROOT/'experiments/kbs_revision_protocol.md')}
    manifest=out/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==config,'Run source or inputs changed.'
    else:
        atomic_json(config,manifest)
        for source in sources:(out/source.name).write_bytes(source.read_bytes())
    entries=entries_for(d,args.dataset,original,device);atomic_json(entries,out/'partitions.json')
    Ef,Ev,Es,Et,T=[d[k].to(device) for k in ['E_fit','E_val','E_support','E_test','T100']]
    yf,yv,ys,yt=[d[k+'_y'] for k in ['fit','val','support','test']]
    for entry in entries:
        check_pause();p=entry['seed'];po=out/f'p{p}';po.mkdir(exist_ok=True)
        if (po/'complete.json').exists():continue
        base,new,pb,pn=[np.array(entry[k]) for k in ['base','novel','pseudo_base','pseudo_novel']]
        ids=np.r_[base,new];ci=np.r_[pb,pn];mask=np.isin(yv,base);vE=Ev[mask];vy=yv[mask]
        W,lam=core.select_ridge(Ef,yf,Ev,yv,base);Wp,plam=core.select_ridge(Ef,yf,Ev,yv,pb)
        Pb=core.means(Ef,yf,base);Ppb=core.means(Ef,yf,pb)
        si=core.support_rows(ys,new,5,7);psi=core.support_rows(ys,pn,5,7)
        Ppn=core.means(Es[psi],ys[psi],pn)
        path=po/'selection.json'
        if path.exists():selected=json.loads(path.read_text());parameters=selected['parameters'];full_scales=selected['full_temperatures']
        else:
            cal=rv.arrays(vE,T[ci],Ppb,Ppn,Wp)
            parameters={m:rv.calibrate(cal,m,ci,vy,pb,pn) for m in ['retained','no_learned_tuned','anchored_tp']}
            l,u,offset=parameters['retained'];parameters['removed_fixed']=(0.,u,offset)
            pseudo_scales=temperatures(vE,T[pb],Ppb,Wp);full_scales=temperatures(vE,T[base],Pb,W)
            for method,scales in [('raw',[1.,1.,1.]),('fixed_temperature',pseudo_scales)]:
                a=simple_arrays(vE,T[ci],Ppb,Ppn,Wp,scales)
                parameters[method]=rv.calibrate(a,'retained',ci,vy,pb,pn)
            selected={'parameters':parameters,'full_temperatures':full_scales,'pseudo_temperatures':pseudo_scales,
                      'ridge_lambda':lam,'pseudo_ridge_lambda':plam,'support_indices':si.tolist(),'entry':entry}
            atomic_json(selected,path)
        tip=depth.select_tip(d,pb,pn,po/'tip',device)['selected']
        l=parameters['retained'][0]
        def closed(E,y):
            prediction=prior.predict(core.z(E@T[base].T)+l*core.z(core.linear(E,W)),base)
            return float(np.mean([np.mean(prediction[y==c]==c) for c in base]))
        refs=(closed(Et,yt),closed(vE,vy));keys,labels=depth.make_keys(d,base,new);keys=keys.to(device)
        chosen={m:tip[m] for m in ['tip_sum_calibrated','tip_mean_calibrated']}
        caches={}
        for beta in sorted({v['beta'] for v in chosen.values()}):
            check_pause();caches[beta]=(depth.retrieval(Et,keys,labels,ids,beta),depth.retrieval(vE,keys,labels,ids,beta))
        counts=torch.tensor([np.sum(labels==c) for c in ids],device=device)
        for order in [0,1]:
            ordered=np.random.default_rng(1000+p+100*order).permutation(new)
            for session in range(1,6):
                check_pause();path=po/f'o{order}_t{session}.json'
                if path.exists():
                    result=json.loads(path.read_text());changed=False
                    for row in result['records']:
                        if row['evaluation']!='fixed_gate' and row['feasible'] and 'test_target_met' not in row:
                            row['test_reference']=refs[0];row['test_retention_ratio']=row['old']/refs[0]
                            row['test_target_met']=row['old']>=row['ratio']*refs[0]-1e-10;changed=True
                    if changed:atomic_json(result,path)
                    continue
                novel=ordered[:len(new)*session//5];active=np.r_[base,novel]
                Pn=core.means(Es[si],ys[si],novel)
                ta=rv.arrays(Et,T[active],Pb,Pn,W);va=rv.arrays(vE,T[active],Pb,Pn,W)
                systems={m:(rv.score(ta,m,parameters[m]),rv.score(va,m,parameters[m]),parameters[m]) for m in rv.METHODS}
                for method,scales in [('raw',[1.,1.,1.]),('fixed_temperature',full_scales)]:
                    systems[method]=(rv.score(simple_arrays(Et,T[active],Pb,Pn,W,scales),'retained',parameters[method]),
                                     rv.score(simple_arrays(vE,T[active],Pb,Pn,W,scales),'retained',parameters[method]),parameters[method])
                ix=np.array([np.flatnonzero(ids==c)[0] for c in active]);text=100*(Et@T[active].T);vtext=100*(vE@T[active].T)
                for method,settings in chosen.items():
                    ct,cv=caches[settings['beta']];ct,cv=ct[:,ix],cv[:,ix]
                    if '_mean_' in method:ct,cv=ct/counts[ix][None],cv/counts[ix][None]
                    params=[settings[k] for k in ['beta','alpha','offset']]
                    systems[method]=(text+settings['alpha']*ct,vtext+settings['alpha']*cv,params)
                depth.save_session(path,systems,active,base,novel,yt,vy,
                                   {'part':p,'order':order,'session':session,'shots':5,'encoder':args.encoder,'dataset':args.dataset},refs)
                # Test target accounting is outcome-only; never used to choose the offset.
                result=json.loads(path.read_text())
                for row in result['records']:
                    if row['evaluation']!='fixed_gate' and row['feasible']:
                        row['test_reference']=refs[0];row['test_retention_ratio']=row['old']/refs[0]
                        row['test_target_met']=row['old']>=row['ratio']*refs[0]-1e-10
                atomic_json(result,path)
        atomic_json({'complete':True,'part':p,'encoder':args.encoder,'reference_closed_test':refs[0],
                     'reference_closed_validation':refs[1]},po/'complete.json')
        print('REVISION COMPLETE',args.encoder,args.dataset,p,flush=True)
    atomic_json({'complete':True,'partitions':10,'encoder':args.encoder,'dataset':args.dataset,
                 'seconds_this_invocation':time.perf_counter()-start,
                 'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0},out/'complete.json')


if __name__=='__main__':main()
