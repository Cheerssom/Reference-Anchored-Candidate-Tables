"""Locked, resumable dense-cache comparison and calibration/shot robustness.

No image encoder is loaded. New runs live separately from immutable prior runs.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fcs_expansion as core
import fcs_retention_value as rv
import kbs_strengthen as prior
from kbs_runtime import configure,check_pause,atomic_json,atomic_npz,digest_file

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/kbs_depth'
ALPHAS=(.1,1.,5.,10.,20.,50.,100.,200.)
BETAS=(.1,1.,2.,5.,10.,20.,50.)
TIP_OFFSETS=tuple(float(v) for v in np.arange(-40,40.01,2.5))
ROBUST_METHODS=('retained','removed_fixed','no_learned_tuned','anchored_tp','dynamic')


def retrieval(E,keys,labels,ids,beta,mean=False,chunk=256):
    """Published dense exponential cache; optional explicit class-mean control."""
    counts=np.array([np.sum(labels==c) for c in ids])
    assert (counts>0).all()
    # ids may be base-then-novel rather than sorted.
    index={int(c):i for i,c in enumerate(ids)}
    target=torch.tensor([index[int(c)] for c in labels],device=keys.device)
    values=F.one_hot(target,len(ids)).to(keys.dtype)
    if mean:values/=torch.as_tensor(counts,device=keys.device,dtype=keys.dtype)[None]
    out=[]
    for i in range(0,len(E),chunk):
        affinity=E[i:i+chunk]@keys.T
        out.append(torch.exp(-beta*(1-affinity))@values)
    return torch.cat(out)


def make_keys(d,base,new,shot=5,seed=7):
    mask=np.isin(d['fit_y'],base)
    si=core.support_rows(d['support_y'],new,shot,seed)
    return torch.cat([d['E_fit'][mask],d['E_support'][si]]),np.r_[d['fit_y'][mask],d['support_y'][si]]


def dynamic_arrays(E,T,Pb,Pn,W):
    return {'text':core.z(E@T.T),'learned':core.z(core.linear(E,W)),
            'prototype':torch.cat([E@Pb.T,core.z(E@Pn.T)],1),'nb':len(Pb)}


def method_score(a,method,params):
    return rv.score(a,'retained' if method=='dynamic' else method,params)


def save_session(path,systems,ids,base,novel,yt,yv,common,reference):
    """Archive test AND selection sufficient statistics for independent checks."""
    records=[];archive={'active_classes':ids,'base_classes':base,'novel_classes':novel,
                        'labels':yt,'validation_labels':yv}
    for method,(ts,vs,params) in systems.items():
        tc=rv.compact(ts,ids,yt,base,novel);vc=rv.compact(vs,ids,yv,base,novel)
        for prefix,c in [('',tc),('validation_',vc)]:
            for key in ['margin','base_prediction','novel_prediction','base_correct','novel_correct','weights']:
                archive[prefix+method+'_'+key]=c[key]
        fixed=float(params[-1]);meta={**common,'method':method,'parameters':params}
        records.append({**meta,'evaluation':'fixed_gate','offset':fixed,**rv.evaluate(tc,fixed)})
        for ratio in rv.RATIOS:
            for evaluation,c,ref in [('matched_test',tc,reference[0]),('base_validation',vc,reference[1])]:
                target=ratio*ref;offset,ceiling=rv.threshold(c,target)
                row={**meta,'evaluation':evaluation,'ratio':ratio,'target':target,'feasible':offset is not None,
                     'offset':offset,'selection_base_ceiling':ceiling}
                if offset is not None:
                    row.update(rv.evaluate(tc,offset));row['selection_base_accuracy']=rv.evaluate(c,offset)['old']
                records.append(row)
    atomic_npz(path.with_suffix('.npz'),**archive)
    atomic_json({'records':records},path)


def manifest(folder,dataset,cache):
    paths=[Path(__file__),Path(core.__file__),Path(rv.__file__),Path(prior.__file__),
           ROOT/'src/experiments/kbs_runtime.py',ROOT/'src/experiments/fcs_review_experiments.py']
    config={'dataset':dataset,'cache_sha256':digest_file(cache),'sources':{p.name:digest_file(p) for p in paths},
            'protocol_sha256':digest_file(ROOT/'experiments/kbs_depth_protocol.md'),
            'primary_manifest_sha256':digest_file(ROOT/f'experiments/kbs_strengthening/{dataset}/run_manifest.json')}
    p=folder/'manifest.json'
    if p.exists():assert json.loads(p.read_text())==config,'Immutable run changed; use a new directory.'
    else:
        atomic_json(config,p)
        for source in paths:(folder/source.name).write_bytes(source.read_bytes())


def select_tip(d,pb,pn,folder,device):
    path=folder/'selection.json'
    if path.exists():return json.loads(path.read_text())
    mask=np.isin(d['val_y'],np.r_[pb,pn]);E=d['E_val'][mask].to(device);y=d['val_y'][mask]
    ids=np.r_[pb,pn];keys,labels=make_keys(d,pb,pn);keys=keys.to(device)
    text=100*(E@d['T100'][ids].to(device).T)
    names=['tip_sum_native','tip_sum_calibrated','tip_mean_native','tip_mean_calibrated']
    selected={name:{'harmonic':-1.} for name in names};trials=[]
    for beta in BETAS:
        check_pause()
        raw=retrieval(E,keys,labels,ids,beta)
        counts=torch.tensor([np.sum(labels==c) for c in ids],device=device)
        for style,cache in [('sum',raw),('mean',raw/counts[None])]:
            for alpha in ALPHAS:
                c=rv.compact(text+alpha*cache,ids,y,pb,pn)
                for native,offsets in [(True,(0.,)),(False,TIP_OFFSETS)]:
                    name=f'tip_{style}_'+('native' if native else 'calibrated')
                    best=max(((rv.evaluate(c,offset)['harmonic'],offset) for offset in offsets),key=lambda x:x[0])
                    row={'method':name,'beta':beta,'alpha':alpha,'offset':best[1],'harmonic':best[0]}
                    trials.append(row)
                    if row['harmonic']>selected[name]['harmonic']+1e-12:selected[name]=row
        print('TIP CAL',folder,'beta',beta,flush=True)
    value={'selected':selected,'trials':trials};atomic_json(value,path);return value


def benchmark(E,T,Pb,Pn,W,params,keys,labels,ids,selected,device):
    E=E[:512];nb=len(Pb)
    def table():return rv.score(rv.arrays(E,T,Pb,Pn,W),'retained',params)
    def dense():return rv.score(rv.arrays(E,T,Pb,Pn,W),'anchored_tp',(0.,1.,0.))
    jobs={'retained':table,'anchored_tp':dense}
    for style in ['sum','mean']:
        p=selected[f'tip_{style}_calibrated']
        jobs['tip_'+style]=lambda p=p,style=style:100*(E@T.T)+p['alpha']*retrieval(E,keys,labels,ids,p['beta'],style=='mean')
    rows={}
    for name,fn in jobs.items():
        for _ in range(5):fn()
        times=[]
        for _ in range(20):
            if device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();fn()
            if device=='cuda':torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
        # Persistent weights, references and class-index vector; common encoder excluded.
        tensors=[T,Pb,W] if name=='retained' else [T,Pb,Pn] if name=='anchored_tp' else [T,keys]
        elements=sum(v.numel()*v.element_size() for v in tensors)
        if name=='retained':elements+=Pn.numel()*Pn.element_size()
        if name.startswith('tip'):elements+=len(labels)*8+len(ids)*8
        rows[name]={'median_ms':float(np.median(times)),'p90_ms':float(np.percentile(times,90)),
                    'persistent_tensor_bytes':elements}
    return {'device':torch.cuda.get_device_name() if device=='cuda' else 'cpu','queries':len(E),
            'warmups':5,'repetitions':20,'dtype':'float32','includes_image_encoding':False,'readouts':rows}


def run_tip(d,entries,dataset,folder,device):
    Ef,Ev,Et,T=[d[k].to(device) for k in ['E_fit','E_val','E_test','T100']]
    yf,yv,yt=[d[k+'_y'] for k in ['fit','val','test']]
    for entry in entries:
        p=entry['seed'];out=folder/f'p{p}';out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        check_pause();base,new,pb,pn=[np.array(entry[k]) for k in ['base','novel','pseudo_base','pseudo_novel']]
        selected=select_tip(d,pb,pn,out,device)['selected'];allids=np.r_[base,new]
        keys,labels=make_keys(d,base,new);keys=keys.to(device)
        mask=np.isin(yv,base);vE=Ev[mask];vy=yv[mask]
        old=json.loads((ROOT/f'experiments/kbs_strengthening/{dataset}/p{p}/partition.json').read_text())
        references=(old['reference_closed_test'],old['reference_closed_validation'])
        # Compute each chosen beta once, all classes, then select columns per session.
        caches={}
        for beta in sorted({v['beta'] for v in selected.values()}):
            check_pause()
            caches[beta]=(retrieval(Et,keys,labels,allids,beta),retrieval(vE,keys,labels,allids,beta))
        counts=torch.tensor([np.sum(labels==c) for c in allids],device=device)
        for order in [0,1]:
            ordered=np.random.default_rng(1000+p+100*order).permutation(new)
            for session in range(1,6):
                check_pause();path=out/f'o{order}_t{session}.json'
                if path.exists():continue
                novel=ordered[:len(new)*session//5];ids=np.r_[base,novel]
                ix=np.array([np.flatnonzero(allids==c)[0] for c in ids])
                text=100*(Et@T[ids].T);vtext=100*(vE@T[ids].T);systems={}
                for name,setting in selected.items():
                    ct,cv=caches[setting['beta']];ct,cv=ct[:,ix],cv[:,ix]
                    if '_mean_' in name:ct,cv=ct/counts[ix][None],cv/counts[ix][None]
                    params=[setting[k] for k in ['beta','alpha','offset']]
                    systems[name]=(text+setting['alpha']*ct,vtext+setting['alpha']*cv,params)
                save_session(path,systems,ids,base,novel,yt,vy,
                             {'part':p,'order':order,'session':session,'shots':5},references)
                print('TIP SAVED',dataset,p,order,session,flush=True)
        if p==0:
            W=core.fit_ridge(Ef,yf,base,entry['ridge_lambda']);Pb=core.means(Ef,yf,base)
            si=core.support_rows(d['support_y'],new,5,7);Pn=core.means(d['E_support'][si].to(device),d['support_y'][si],new)
            atomic_json(benchmark(Et,T[allids],Pb,Pn,W,old['table_parameters']['retained'],keys,labels,allids,selected,device),folder/'benchmark.json')
        atomic_json({'complete':True,'part':p},out/'complete.json')


def run_robust(d,entries,dataset,folder,device):
    Ef,Ev,Es,Et,T=[d[k].to(device) for k in ['E_fit','E_val','E_support','E_test','T100']]
    yf,yv,ys,yt=[d[k+'_y'] for k in ['fit','val','support','test']]
    for entry in entries:
        p=entry['seed'];base,new=[np.array(entry[k]) for k in ['base','novel']]
        W=core.fit_ridge(Ef,yf,base,entry['ridge_lambda']);Pb=core.means(Ef,yf,base)
        mask=np.isin(yv,base);vE=Ev[mask];vy=yv[mask]
        for trial in range(3):
            if trial==0:pb,pn=[np.array(entry[k]) for k in ['pseudo_base','pseudo_novel']]
            else:
                pb=np.sort(np.random.default_rng(1701+p+1000*(trial-1)).choice(base,len(base)//2,replace=False));pn=np.setdiff1d(base,pb)
            check_pause();Wp,_=core.select_ridge(Ef,yf,Ev,yv,pb);Ppb=core.means(Ef,yf,pb);ci=np.r_[pb,pn]
            for shots in [1,2,5]:
                out=folder/f'p{p}_r{trial}_k{shots}';out.mkdir(parents=True,exist_ok=True)
                if (out/'complete.json').exists():continue
                check_pause();selected=out/'selection.json';seed=7+trial
                si=core.support_rows(ys,new,shots,seed);psi=core.support_rows(ys,pn,shots,seed)
                if selected.exists():selection=json.loads(selected.read_text());params=selection['parameters']
                else:
                    Ppn=core.means(Es[psi],ys[psi],pn)
                    cal=rv.arrays(vE,T[ci],Ppb,Ppn,Wp)
                    params={m:rv.calibrate(cal,m,ci,vy,pb,pn) for m in ['retained','no_learned_tuned','anchored_tp']}
                    dyn=dynamic_arrays(vE,T[ci],Ppb,Ppn,Wp)
                    params['dynamic']=rv.calibrate(dyn,'retained',ci,vy,pb,pn)
                    l,u,offset=params['retained'];params['removed_fixed']=(0.,u,offset)
                    selection={'parameters':params,'pseudo_base':pb.tolist(),'pseudo_novel':pn.tolist(),
                               'support_indices':si.tolist(),'pseudo_support_indices':psi.tolist(),'support_seed':seed}
                    atomic_json(selection,selected)
                l=params['retained'][0]
                def closed(E,y):
                    pred=prior.predict(core.z(E@T[base].T)+l*core.z(core.linear(E,W)),base)
                    return float(np.mean([np.mean(pred[y==c]==c) for c in base]))
                references=(closed(Et,yt),closed(vE,vy))
                # Recompute drift deterministically even when numerical sessions were saved.
                drift=[]
                for order in [0,1]:
                    ordered=np.random.default_rng(1000+p+100*order).permutation(new);previous={};oldids=None
                    for session in range(1,6):
                        check_pause();novel=ordered[:len(new)*session//5];ids=np.r_[base,novel]
                        Pn=core.means(Es[si],ys[si],novel)
                        ta=rv.arrays(Et,T[ids],Pb,Pn,W);va=rv.arrays(vE,T[ids],Pb,Pn,W)
                        dt=dynamic_arrays(Et,T[ids],Pb,Pn,W);dv=dynamic_arrays(vE,T[ids],Pb,Pn,W)
                        systems={m:(method_score(dt if m=='dynamic' else ta,m,params[m]),
                                    method_score(dv if m=='dynamic' else va,m,params[m]),params[m]) for m in ROBUST_METHODS}
                        for m in ['retained','dynamic']:
                            current=systems[m][0]
                            if oldids is not None:
                                ix=np.array([np.flatnonzero(ids==c)[0] for c in oldids]);restricted=current[:,ix]
                                delta=(restricted-previous[m]).abs()
                                drift.append({'order':order,'session':session,'method':m,
                                    'maximum_score_drift':float(delta.max()),
                                    'mean_absolute_score_drift':float(delta.mean()),
                                    'restricted_prediction_changes':int(np.sum(prior.predict(restricted,oldids)!=prior.predict(previous[m],oldids)))})
                            previous[m]=current
                        oldids=ids.copy();path=out/f'o{order}_t{session}.json'
                        if not path.exists():
                            save_session(path,systems,ids,base,novel,yt,vy,
                                {'part':p,'trial':trial,'shots':shots,'order':order,'session':session},references)
                atomic_json(drift,out/'drift.json');atomic_json({'complete':True},out/'complete.json')
                print('ROBUST COMPLETE',dataset,p,trial,shots,flush=True)


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['cifar','pets','imagenetr'],required=True)
    ap.add_argument('--stage',choices=['tip','robust','all'],default='all');args=ap.parse_args()
    device=configure();check_pause();start=time.perf_counter();d,entries,cache=prior.load(args.dataset)
    if entries is None:entries=json.loads((ROOT/'experiments/kbs_strengthening/imagenetr/partitions.json').read_text())
    folder=OUT/args.dataset;folder.mkdir(parents=True,exist_ok=True);manifest(folder,args.dataset,cache)
    for stage,fn in [('tip',run_tip),('robust',run_robust)]:
        if args.stage in [stage,'all']:fn(d,entries,args.dataset,folder/stage,device)
    atomic_json({'dataset':args.dataset,'stage':args.stage,'elapsed_seconds_this_invocation':time.perf_counter()-start,
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None},folder/f'{args.stage}_invocation.json')
    print('DEPTH COMPLETE',args.dataset,args.stage,flush=True)


if __name__=='__main__':main()
