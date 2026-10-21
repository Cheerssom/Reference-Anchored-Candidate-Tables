"""Additional review-driven controls; original result files are never changed.

Fixed before this extension's first run: k=5, original partitions/support IDs,
offset grid -4..4 step .25, prototype weights 0,.25,.5,1,2,4;
retention targets .95/.99 of the reference readout's pseudo-base closed accuracy;
RanPAC frozen phase-2 adaptation, M=2000 and base-only ridge selection.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,hashlib,json,time
import numpy as np
import torch
import torch.nn.functional as F
import fcs_expansion as core
from lt_dataset import make_lt_datasets
from train import split_train_val

ROOT=Path(__file__).resolve().parents[2]
OFFSETS=tuple(float(x) for x in np.arange(-4,4.01,.25))
MIXES=(0.,.25,.5,1.,2.,4.)
RETENTION=(.95,.99)
RP_DIM=2000
RIDGES=tuple(float(10.**x) for x in range(-8,9))

def predict(sc,ids):
    # Canonical global-label tie breaking makes arrival-order tests well defined.
    order=np.argsort(ids)
    return np.asarray(ids)[order][sc[:,order].argmax(1).cpu().numpy()]

def base_arrays(E,T,Pb,Pn,W,mode):
    if mode in ('reference','dynamic'):
        B=core.branches(E,T,Pb,Pn,W,mode)
        return B
    text=E@T.T
    proto=E@torch.cat([Pb,Pn]).T
    if mode=='text_bias':
        return core.z(text,text[:,:len(Pb)]),None
    if mode=='anchored_tp':
        return core.z(text,text[:,:len(Pb)]),core.z(proto,E@Pb.T)
    raise ValueError(mode)

def control_score(A,mode,param,nb):
    if mode in ('reference','dynamic'):return core.score(A,param)
    u,d=param
    sc=A[0].clone()
    if A[1] is not None:sc+=u*A[1]
    sc[:,nb:]+=d
    return sc

def choose(candidates,ids,y,nb,target=None):
    evaluated=[]
    for param,sc in candidates:
        met=core.metrics(predict(sc,ids),y,ids[:nb],ids[nb:])
        evaluated.append((param,met))
    if target is None:
        return max(evaluated,key=lambda x:x[1]['harmonic']),True
    feasible=[x for x in evaluated if x[1]['old']>=target-1e-12]
    if feasible:return max(feasible,key=lambda x:x[1]['new']),True
    return max(evaluated,key=lambda x:(x[1]['old'],x[1]['new'])),False

def rp_stats(H,y,ids):
    yy=torch.tensor(np.array([np.where(ids==c)[0][0] for c in y]),device=H.device)
    Y=F.one_hot(yy,len(ids)).double()
    return H.T@H,H.T@Y

def rp_solve(G,Q,lam):
    return torch.linalg.solve(G+lam*torch.eye(len(G),device=G.device,dtype=G.dtype),Q)

def rp_select(H,y,Hv,yv,ids):
    G,Q=rp_stats(H,y,ids)
    eig,U=torch.linalg.eigh(G)
    eig=eig.clamp_min(0)
    projected=Hv@U
    cross=U.T@Q
    best=(-1,None)
    for lam in RIDGES:
        pp=predict(projected@(cross/(eig[:,None]+lam)),ids)
        acc=np.mean([np.mean(pp[yv==c]==c) for c in ids])
        if acc>best[0]+1e-12:best=(acc,lam)
    return G,Q,best[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['cifar','pets'],required=True)
    args=ap.parse_args();torch.set_num_threads(4);dev='cuda' if torch.cuda.is_available() else 'cpu'
    start=time.perf_counter();folder=ROOT/'experiments'/('fcs' if args.dataset=='cifar' else 'fcs_pets')
    out=folder/'review_controls';out.mkdir(exist_ok=True)
    manifest=json.loads((folder/'manifest.json').read_text())
    previous=json.loads((folder/'results.json').read_text())
    if args.dataset=='pets':
        c=torch.load(folder/'pets_frozen.pt',map_location='cpu',weights_only=False)
        yf,yv,yt,ys=[c[k+'_y'] for k in ('fit','val','test','support')]
        Es=c['E_support'].to(dev)
    else:
        train,sup,_,test,_=make_lt_datasets(rho=100);fit,val=split_train_val(train)
        yf,yv,yt,ys=fit.fine,val.fine,test.fine,sup.fine
        c=torch.load(ROOT/'src/expand_cache.pt',map_location='cpu',weights_only=False)
        Es=torch.load(folder/'frozen_support.pt',map_location='cpu',weights_only=False)['features'].to(dev)
    Ef,Ev,Et,T=[c[k].to(dev) for k in ('E_fit','E_val','E_test','T100')]
    results={'dataset':args.dataset,'controls':[],'retention':[],'sequential':[],
             'ranpac':[],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'offsets':OFFSETS,'mixes':MIXES,'retention_ratios':RETENTION,
             'rp_dimension':RP_DIM,'ridge_grid':RIDGES}
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for entry in manifest['partitions']:
        p=entry['seed'];base=np.array(entry['base']);new=np.array(entry['novel'])
        pb=np.array(entry['pseudo_base']);pn=np.array(entry['pseudo_novel'])
        ids=np.r_[base,new];ci=np.r_[pb,pn]
        keep=np.isin(yv,base);Ecal=Ev[keep];ycal=yv[keep]
        W=core.fit_ridge(Ef,yf,base,entry['ridge_lambda'])
        Wp=core.fit_ridge(Ef,yf,pb,entry['pseudo_ridge_lambda'])
        Pb=core.means(Ef,yf,base);Ppb=core.means(Ef,yf,pb)
        si=core.support_rows(ys,new,5,7);psi=core.support_rows(ys,pn,5,7)
        Pn=core.means(Es[si],ys[si],new);Ppn=core.means(Es[psi],ys[psi],pn)
        gate={m:next(r['gates'] for r in previous['records'] if r['part']==p and r['support_seed']==7 and r['k']==5 and r['method']==m)
              for m in ('reference','dynamic')}
        closed_cal=core.z(Ecal@T[pb].T)+gate['reference'][0]*core.z(core.linear(Ecal,Wp))
        pcm=predict(closed_cal,pb)
        closed_acc=np.mean([np.mean(pcm[ycal==c]==c) for c in pb])
        for mode in ('text_bias','anchored_tp','reference','dynamic'):
            ac=base_arrays(Ecal,T[ci],Ppb,Ppn,Wp,mode)
            at=base_arrays(Et,T[ids],Pb,Pn,W,mode)
            params=([(gate[mode][0],gate[mode][1],d) for d in OFFSETS] if mode in gate else
                    [(u,d) for u in ((0.,) if mode=='text_bias' else MIXES) for d in OFFSETS])
            candidates=[(param,control_score(ac,mode,param,len(pb))) for param in params]
            if mode in ('text_bias','anchored_tp'):
                (param,vm),_=choose(candidates,ci,ycal,len(pb))
                sc=control_score(at,mode,param,len(base))
                results['controls'].append({'part':p,'method':mode,'parameter':param,
                    'validation':vm,**core.metrics(predict(sc,ids),yt,base,new)})
            for ratio in RETENTION:
                target=ratio*closed_acc
                (param,vm),feasible=choose(candidates,ci,ycal,len(pb),target)
                pred=predict(control_score(at,mode,param,len(base)),ids)
                results['retention'].append({'part':p,'method':mode,'ratio':ratio,
                    'validation_base_target':target,'validation_feasible':feasible,
                    'parameter':param,'validation':vm,**core.metrics(pred,yt,base,new)})
        # Growing vocabulary across five sessions; two arrival orders share rows.
        for arrival in (0,1):
            ordered=np.random.default_rng(1000+p+100*arrival).permutation(new)
            previous_sc=None;previous_ids=None
            for step in range(1,6):
                nn=ordered[:len(new)*step//5]
                pp=core.means(Es[si],ys[si],nn)
                active=np.r_[base,nn]
                sc=core.score(core.branches(Et,T[active],Pb,pp,W,'reference'),gate['reference'])
                old_drift=0. if previous_sc is None else float((sc[:,:len(previous_ids)]-previous_sc).abs().max())
                assert old_drift<1e-4
                pred=predict(sc,active)
                results['sequential'].append({'part':p,'order':arrival,'session':step,
                    'added':len(nn),'max_existing_score_drift':old_drift,
                    **core.metrics(pred,yt,base,nn)})
                previous_sc,previous_ids=sc,active
            if arrival==0:final_pred=pred
            else:assert np.array_equal(final_pred,pred),'Arrival-order dependence'
        # RanPAC phase-2 equations with Gaussian random projection and ReLU.
        # Encoder stays fixed, M is bounded to 2000, and lambda is selected on
        # base validation (rather than recalibrating using novel labels).
        gen=torch.Generator(device=dev).manual_seed(2000+p)
        projection=torch.randn(Ef.shape[1],RP_DIM,generator=gen,device=dev)
        Hf=F.relu(Ef@projection).double();Hv=F.relu(Ev@projection).double()
        Hs=F.relu(Es@projection).double();Ht=F.relu(Et@projection).double()
        def fit_rp(old,novel,shotids):
            mask=np.isin(yf,old);v=np.isin(yv,old)
            G,Q,lam=rp_select(Hf[mask],yf[mask],Hv[v],yv[v],old)
            fullids=np.r_[old,novel]
            Qall=torch.zeros((RP_DIM,len(fullids)),device=dev,dtype=torch.double);Qall[:,:len(old)]=Q
            yg=ys[shotids];Y=F.one_hot(torch.tensor(np.array([np.where(fullids==c)[0][0] for c in yg]),device=dev),len(fullids)).double()
            G=G+Hs[shotids].T@Hs[shotids];Qall=Qall+Hs[shotids].T@Y
            return rp_solve(G,Qall,lam),lam
        Rcal,lamcal=fit_rp(pb,pn,psi)
        Rtest,lam=fit_rp(base,new,si)
        rcal=core.z(Hv[keep]@Rcal);rtest=core.z(Ht@Rtest)
        cs=[]
        for d in OFFSETS:
            sc=rcal.clone();sc[:,len(pb):]+=d;cs.append((d,sc))
        (offset,vm),_=choose(cs,ci,ycal,len(pb))
        rtest[:,len(base):]+=offset
        results['ranpac'].append({'part':p,'method':'ranpac_frozen_2000',
             'ridge':lam,'pseudo_ridge':lamcal,'offset':offset,
             'validation':vm,**core.metrics(predict(rtest,ids),yt,base,new)})
        results['elapsed_seconds']=time.perf_counter()-start
        (out/'results.json').write_text(json.dumps(results,indent=2))
        print(args.dataset,p,results['controls'][-2:],results['ranpac'][-1],flush=True)
    print(f'FINISHED {args.dataset}: {time.perf_counter()-start:.1f}s',flush=True)

if __name__=='__main__':main()
