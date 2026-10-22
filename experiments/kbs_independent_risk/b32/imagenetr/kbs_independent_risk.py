"""Refitted disjoint-pool risk experiment on cached frozen features."""
from pathlib import Path
import argparse, gc, json, sys, time, shutil
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fcs_expansion as core
import fcs_retention_value as rv
import kbs_strengthen as prior
import kbs_depth as depth
import kbs_revision as revision
import kbs_risk_control as risk
from kbs_runtime import configure, check_pause, atomic_json, atomic_npz, digest_file

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/kbs_independent_risk'
PROTOCOL=ROOT/'experiments/kbs_independent_risk_protocol.md'
METHODS=('retained','no_learned_tuned','anchored_tp','raw','fixed_temperature','tip_mean_calibrated')


def split_indices(y):
    rng=np.random.default_rng(662026);groups={k:[] for k in ['fit','tune','cal']}
    counts=[]
    for c in np.unique(y):
        ix=rng.permutation(np.flatnonzero(y==c));n=len(ix)
        nc=max(1,int(.4*n));nt=max(1,int(.2*n));assert n-nc-nt>=2
        groups['cal'].extend(ix[:nc]);groups['tune'].extend(ix[nc:nc+nt]);groups['fit'].extend(ix[nc+nt:])
        counts.append(dict(class_id=int(c),fit=n-nc-nt,tune=nt,cal=nc))
    groups={k:np.sort(v).tolist() for k,v in groups.items()}
    assert sorted(sum(groups.values(),[]))==list(range(len(y)))
    return dict(indices=groups,class_counts=counts,label_sha256=core.digest(y))


def select_cache(d,pb,pn,device):
    mask=np.isin(d['val_y'],np.r_[pb,pn]);E=d['E_val'][mask].to(device);y=d['val_y'][mask]
    ids=np.r_[pb,pn];keys,labels=depth.make_keys(d,pb,pn);keys=keys.to(device)
    counts=torch.tensor([np.sum(labels==c) for c in ids],device=device)
    text=100*(E@d['T100'][ids].to(device).T);best={'harmonic':-1.};trials=[]
    for beta in depth.BETAS:
        check_pause();cache=depth.retrieval(E,keys,labels,ids,beta)/counts[None]
        for alpha in depth.ALPHAS:
            c=rv.compact(text+alpha*cache,ids,y,pb,pn)
            h,offset=max(((rv.evaluate(c,o)['harmonic'],o) for o in depth.TIP_OFFSETS),key=lambda x:x[0])
            row=dict(beta=beta,alpha=alpha,offset=offset,harmonic=h);trials.append(row)
            if h>best['harmonic']+1e-12:best=row
    return dict(selected=best,trials=trials)


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['cifar','pets','imagenetr'],required=True)
    ap.add_argument('--encoder',choices=['l14','b32'],required=True)
    ap.add_argument('--parts',type=int,default=10);args=ap.parse_args()
    assert 1<=args.parts<=10
    assert shutil.disk_usage(ROOT).free>3*1024**3,'Less than 3 GiB free; stop before writing.'
    device=configure();check_pause();torch.manual_seed(662026);start=time.perf_counter()
    if args.encoder=='l14':d,_,cache=prior.load(args.dataset)
    else:
        cache=ROOT/f'experiments/kbs_revision/b32_features/{args.dataset}/frozen.pt'
        d=torch.load(cache,map_location='cpu',weights_only=False)
    partitions=ROOT/f'experiments/kbs_revision/{args.encoder}/{args.dataset}/partitions.json'
    entries=json.loads(partitions.read_text())
    sources=[Path(__file__),Path(risk.__file__),Path(core.__file__),Path(rv.__file__),
             Path(depth.__file__),Path(prior.__file__),Path(revision.__file__),
             ROOT/'src/experiments/kbs_runtime.py']
    config=dict(encoder=args.encoder,dataset=args.dataset,cache_sha256=digest_file(cache),
                partitions_sha256=digest_file(partitions),protocol_sha256=digest_file(PROTOCOL),
                sources={f.name:digest_file(f) for f in sources})
    out=OUT/args.encoder/args.dataset;out.mkdir(parents=True,exist_ok=True)
    mf=out/'manifest.json'
    if mf.exists():assert json.loads(mf.read_text())==config,'Changed source or inputs: use a new run directory.'
    else:
        atomic_json(config,mf)
        for f in sources:(out/f.name).write_bytes(f.read_bytes())
    y=np.r_[d['fit_y'],d['val_y']];E=torch.cat([d['E_fit'],d['E_val']])
    split=split_indices(y);sp=OUT/'splits'/f'{args.dataset}.json'
    if sp.exists():assert json.loads(sp.read_text())==split
    else:atomic_json(split,sp)
    d={k:v for k,v in d.items() if k in ['E_support','support_y','E_test','test_y','T100','meta']}
    for dst,src in [('fit','fit'),('val','tune'),('cal','cal')]:
        ix=split['indices'][src];d['E_'+dst]=E[ix];d[dst+'_y']=y[ix]
    del E;gc.collect()
    Ef,Es,T=[d[k].to(device) for k in ['E_fit','E_support','T100']]
    yf,ys=d['fit_y'],d['support_y']
    for entry in entries[:args.parts]:
        check_pause();p=entry['seed'];po=out/f'p{p}';po.mkdir(exist_ok=True)
        if (po/'complete.json').exists():continue
        base,new,pb,pn=[np.array(entry[k]) for k in ['base','novel','pseudo_base','pseudo_novel']]
        masks={pool:np.isin(d[pool+'_y'],base) for pool in ['val','cal']}
        Ev=d['E_val'][masks['val']].to(device);yv=d['val_y'][masks['val']]
        Ec=d['E_cal'][masks['cal']].to(device);yc=d['cal_y'][masks['cal']]
        Et=d['E_test'].to(device);yt=d['test_y']
        W,lam=core.select_ridge(Ef,yf,Ev,yv,base);Wp,plam=core.select_ridge(Ef,yf,Ev,yv,pb)
        Pb=core.means(Ef,yf,base);Ppb=core.means(Ef,yf,pb)
        si=core.support_rows(ys,new,5,7);psi=core.support_rows(ys,pn,5,7)
        Ppn=core.means(Es[psi],ys[psi],pn);ci=np.r_[pb,pn]
        selected_path=po/'selection.json'
        if selected_path.exists():selected=json.loads(selected_path.read_text())
        else:
            a=rv.arrays(Ev,T[ci],Ppb,Ppn,Wp)
            params={m:rv.calibrate(a,m,ci,yv,pb,pn) for m in METHODS[:3]}
            ps=revision.temperatures(Ev,T[pb],Ppb,Wp)
            fs=revision.temperatures(Ev,T[base],Pb,W)
            for m,scales in [('raw',[1.,1.,1.]),('fixed_temperature',ps)]:
                params[m]=rv.calibrate(revision.simple_arrays(Ev,T[ci],Ppb,Ppn,Wp,scales),'retained',ci,yv,pb,pn)
            selected=dict(parameters=params,full_temperatures=fs,ridge_lambda=lam,
                          pseudo_ridge_lambda=plam,cache=select_cache(d,pb,pn,device),
                          split_sha256=digest_file(sp),support_indices=si.tolist())
            atomic_json(selected,selected_path)
        assert selected['ridge_lambda']==lam and selected['pseudo_ridge_lambda']==plam
        params=selected['parameters'];l=params['retained'][0]
        def ref(Eq):return prior.predict(core.z(Eq@T[base].T)+l*core.z(core.linear(Eq,W)),base)
        ref_c,ref_t=ref(Ec)==yc,ref(Et)==yt
        wc=risk.weights(yc,base);ref_c_acc=float(wc@ref_c)
        ref_t_acc=float(np.mean([ref_t[yt==c].mean() for c in base]))
        keys,labels=depth.make_keys(d,base,new);keys=keys.to(device);fullids=np.r_[base,new]
        tip=selected['cache']['selected'];counts=torch.tensor([np.sum(labels==c) for c in fullids],device=device)
        cache_scores={pool:depth.retrieval(q,keys,labels,fullids,tip['beta'])/counts[None]
                      for pool,q in [('tune',Ev),('cal',Ec),('test',Et)]}
        for order in [0,1]:
            ordered=np.random.default_rng(1000+p+100*order).permutation(new)
            for session in range(1,6):
                check_pause();path=po/f'o{order}_t{session}.json'
                if path.exists():continue
                novel=ordered[:len(new)*session//5];active=np.r_[base,novel]
                Pn=core.means(Es[si],ys[si],novel)
                ix=[int(np.flatnonzero(fullids==c)[0]) for c in active]
                systems={}
                for pool,q,yq in [('tune',Ev,yv),('cal',Ec,yc),('test',Et,yt)]:
                    a=rv.arrays(q,T[active],Pb,Pn,W)
                    systems[pool]={m:rv.compact(rv.score(a,m,params[m]),active,yq,base,novel) for m in METHODS[:3]}
                    for m,scales in [('raw',[1.,1.,1.]),('fixed_temperature',selected['full_temperatures'])]:
                        a=revision.simple_arrays(q,T[active],Pb,Pn,W,scales)
                        systems[pool][m]=rv.compact(rv.score(a,'retained',params[m]),active,yq,base,novel)
                    score=100*(q@T[active].T)+tip['alpha']*cache_scores[pool][:,ix]
                    systems[pool]['tip_mean_calibrated']=rv.compact(score,active,yq,base,novel)
                archive=dict(base_classes=base,novel_classes=novel,cal_labels=yc,test_labels=yt,
                             cal_reference_correct=ref_c,test_reference_correct=ref_t)
                rows=[]
                for m in METHODS:
                    cc,tc=systems['cal'][m],systems['test'][m]
                    candidates=np.unique(np.quantile(systems['tune'][m]['margin'],np.linspace(0,1,101)))
                    archive[m+'_candidates']=candidates
                    for pool,c in [('cal',cc),('test',tc)]:
                        for k in ['margin','base_prediction','novel_prediction']:
                            arr=c[k];archive[pool+'_'+m+'_'+k]=arr if k=='margin' else arr.astype(np.int16)
                    for policy,value in [('empirical',.95),('empirical',.97),('empirical',.99),
                                         ('risk',.10),('risk',.05),('risk',.01)]:
                        row=dict(method=m,policy=policy,level=value,part=p,order=order,session=session,
                                 reference_closed_cal=ref_c_acc,reference_closed_test=ref_t_acc)
                        if policy=='risk':
                            detail=risk.certify(cc,ref_c,wc,candidates,delta=value)
                            offset=detail['offset'];row.update(detail);row['feasible']=offset is not None
                        else:
                            offset,ceiling=rv.threshold(cc,value*ref_c_acc)
                            row.update(offset=offset,feasible=offset is not None,validation_ceiling=ceiling,
                                       fallback=offset is None,certified_expansion=False)
                        if offset is None:
                            result=dict(old=ref_t_acc,new=0.,overall=ref_t_acc*len(base)/len(active),harmonic=0.)
                        else:result=rv.evaluate(tc,offset)
                        row.update(result,test_target_met=result['old']>=.95*ref_t_acc-1e-10,
                                   target_gap_pp=100*(result['old']-.95*ref_t_acc),
                                   test_retention_ratio=result['old']/ref_t_acc)
                        rows.append(row)
                atomic_npz(path.with_suffix('.npz'),**archive)
                atomic_json(dict(records=rows,encoder=args.encoder,dataset=args.dataset),path)
        atomic_json(dict(complete=True,encoder=args.encoder,dataset=args.dataset,part=p,
                         reference_closed_test=ref_t_acc,reference_closed_cal=ref_c_acc,
                         calibration_queries=len(yc),tuning_queries=len(yv),
                         fitting_queries=int(np.isin(yf,base).sum()),
                         calibration_max_weight=float(wc.max())),po/'complete.json')
        print('RISK COMPLETE',args.encoder,args.dataset,p,'cal',len(yc),flush=True)
    if args.parts==10:
        atomic_json(dict(complete=True,partitions=10,seconds_this_invocation=time.perf_counter()-start,
                         peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0),out/'complete.json')


if __name__=='__main__':main()
