"""Equal-budget base-only fusion retuning with resumable local evaluation."""
from pathlib import Path
import argparse,gc,json,sys,time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fcs_expansion as core
import fcs_retention_value as rv
import kbs_reviewer_revision as previous
import kbs_reviewer_readouts as readouts
import kbs_calibration_fairness as fairness
import kbs_risk_control as risk
import kbs_mechanism_followup as mechanism
from kbs_runtime import configure,check_pause,atomic_json,atomic_npz,digest_file
ROOT=Path(__file__).resolve().parents[2]
GRID=(0.,.25,.5,1.,2.,4.)
MODES=('query','none','fixed')
METHODS=tuple(f'{head}_{mode}' for head in ('retained','dense') for mode in MODES)

@torch.no_grad()
def components(E,T,Pb,Pn,W,moments,mode):
    nb=len(Pb);result={}
    for i,(name,values,reference) in enumerate([
        ('text',E@T.T,E@T[:nb].T),
        ('learned',core.linear(E,W),None),
        ('prototype',E@torch.cat([Pb,Pn]).T,E@Pb.T)]):
        if reference is None:reference=values
        mu=reference.mean(1,keepdim=True) if mode=='query' else (0. if mode=='none' else moments[i][0])
        sig=reference.std(1,correction=0,keepdim=True).clamp_min(1e-6) if mode!='fixed' else moments[i][1]
        result[name]=(values-mu)/sig
    result['nb']=nb
    return result

def scores(comp,head,a,u):
    nb=comp['nb'];out=comp['text'].clone()
    out[:,:nb]+=a*(comp['learned'] if head=='retained' else comp['prototype'][:,:nb])
    out[:,nb:]+=u*comp['prototype'][:,nb:]
    return out

def tuning_scores(c,candidates):
    ch=(candidates[:,None]>c['margin'][None])|((candidates[:,None]==c['margin'][None])&(c['novel_prediction'][None]<c['base_prediction'][None]))
    ab=(~ch)@(c['weights']*c['base_correct']);an=ch@(c['weights']*c['novel_correct'])
    return np.divide(2*ab*an,ab+an,out=np.zeros_like(ab),where=(ab+an)>0)

def select(comp,head,ids,y,base,new):
    best=(-1.,None);trials=[]
    for a in GRID:
        for u in GRID:
            check_pause();c=rv.compact(scores(comp,head,a,u),ids,y,base,new)
            offsets=np.quantile(c['margin'],np.linspace(0,1,101));hs=tuning_scores(c,offsets)
            j=int(np.argmax(hs));trials.append(dict(a=a,u=u,offsets=offsets.tolist(),harmonics=hs.tolist()))
            if float(hs[j])>best[0]+1e-12:best=(float(hs[j]),dict(a=a,u=u,offset=float(offsets[j]),pseudo_harmonic=float(hs[j]),quantile_index=j))
    assert len(trials)==36
    return dict(selected=best[1],trials=trials,evaluated_settings=3636)

def pack(c):
    return {k:(c[k].astype(np.int16) if k.endswith('prediction') else c[k]) for k in ['margin','base_prediction','novel_prediction']}

@torch.no_grad()
def run(enc,ds,device,out,parts):
    start=time.perf_counter();folder=out/enc/ds;folder.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),Path(core.__file__),Path(rv.__file__),Path(previous.__file__),Path(readouts.__file__),Path(fairness.__file__),Path(risk.__file__),Path(mechanism.__file__),ROOT/'src/experiments/kbs_paired_accuracy.py',ROOT/'src/experiments/kbs_runtime.py']
    config=dict(protocol_sha256=digest_file(ROOT/'experiments/kbs_full_fusion_retuning_protocol.md'),sources={p.relative_to(ROOT).as_posix():digest_file(p) for p in sources})
    mf=folder/'manifest.json'
    if mf.exists():assert json.loads(mf.read_text())==config,'Source changed; use a new directory.'
    else:atomic_json(config,mf)
    if (folder/'complete.json').exists():return
    d,cache=previous.load_data(ds,enc)
    data_paths=[cache,ROOT/f'experiments/kbs_independent_risk/splits/{ds}.json',ROOT/f'experiments/kbs_calibration_budget/pools/{ds}.json',ROOT/f'experiments/kbs_calibration_budget/features/{enc}/{ds}/extra.pt',ROOT/f'experiments/kbs_revision/{enc}/{ds}/partitions.json']
    inputs={p.relative_to(ROOT).as_posix():digest_file(p) for p in data_paths}
    dp=folder/'data_manifest.json'
    if dp.exists():assert json.loads(dp.read_text())==inputs
    else:atomic_json(inputs,dp)
    entries=json.loads(data_paths[-1].read_text());Ef,Es,T=[d[k].to(device) for k in ['E_fit','E_support','T']];yf,ys=d['fit_y'],d['support_y']
    for entry in entries[:parts]:
        check_pause();p=entry['seed'];po=folder/f'p{p}';po.mkdir(exist_ok=True)
        if (po/'complete.json').exists():continue
        original=ROOT/f'experiments/kbs_independent_risk/{enc}/{ds}/p{p}/selection.json';sel=json.loads(original.read_text());oldhash=digest_file(original)
        base,new,pb,pn=[np.asarray(entry[k]) for k in ['base','novel','pseudo_base','pseudo_novel']]
        W=core.fit_ridge(Ef,yf,base,sel['ridge_lambda']);Wp=core.fit_ridge(Ef,yf,pb,sel['pseudo_ridge_lambda']);Pb=core.means(Ef,yf,base);Ppb=core.means(Ef,yf,pb)
        mask=np.isin(d['tune_y'],base);Ev=d['E_tune'][mask].to(device);yv=d['tune_y'][mask]
        psi=core.support_rows(ys,pn,5,7);Ppn=core.means(Es[psi],ys[psi],pn);pseudo_ids=np.r_[pb,pn]
        ps=po/'selection.json'
        if ps.exists():selection=json.loads(ps.read_text());assert selection['original_selection_sha256']==oldhash
        else:
            pm=readouts.fixed_moments(Ev,T[pb],Ppb,Wp);fullmom=readouts.fixed_moments(Ev,T[base],Pb,W)
            selections={}
            for mode in MODES:
                comp=components(Ev,T[pseudo_ids],Ppb,Ppn,Wp,pm,mode)
                for head in ['retained','dense']:
                    method=head+'_'+mode;selections[method]=select(comp,head,pseudo_ids,yv,pb,pn)
            selection=dict(encoder=enc,dataset=ds,part=p,original_selection_sha256=oldhash,full_moments=fullmom,pseudo_moments=pm,selections=selections,selection_uses='base-only pseudo-expansion tuning queries; no calibration/test inputs')
            atomic_json(selection,ps)
        # Selection is atomically frozen before any calibration/test score evaluation.
        maskc=np.isin(d['cal_y'],base);queries={'tune':Ev,'cal':d['E_cal'][maskc].to(device),'test':d['E_test'].to(device)}
        labels={'tune':yv,'cal':d['cal_y'][maskc],'test':d['test_y']}
        original_a=sel['parameters']['retained'][0]
        def reference(E,y):
            values=core.z(E@T[base].T)+original_a*core.z(core.linear(E,W))
            pred=base[values.argmax(1).cpu().numpy()]
            return pred==y
        refc=reference(queries['cal'],labels['cal']);reft=reference(queries['test'],labels['test'])
        closed=float(np.mean([reft[labels['test']==c].mean() for c in base]));wc=risk.weights(labels['cal'],base)
        si=np.array(sel['support_indices']);selected_hash=digest_file(ps);count=0
        for order in [0,1]:
            ordered=np.random.default_rng(1000+p+100*order).permutation(new)
            for session in range(1,6):
                check_pause();dest=po/f'o{order}_t{session}.json'
                if dest.exists():continue
                novel=ordered[:len(new)*session//5];ids=np.r_[base,novel];Pn=core.means(Es[si],ys[si],novel)
                systems={};archive=dict(base_classes=base,novel_classes=novel,cal_labels=labels['cal'],test_labels=labels['test'],cal_reference_correct=refc,test_reference_correct=reft);results=[]
                for mode in MODES:
                    for pool in ['tune','cal','test']:
                        comp=components(queries[pool],T[ids],Pb,Pn,W,selection['full_moments'],mode)
                        for head in ['retained','dense']:
                            method=head+'_'+mode;param=selection['selections'][method]['selected']
                            c=rv.compact(scores(comp,head,param['a'],param['u']),ids,labels[pool],base,novel)
                            systems[pool,method]=c
                            if pool!='tune':archive.update({pool+'_'+method+'_'+k:v for k,v in pack(c).items()})
                for method in METHODS:
                    tc=systems['test',method];cc=systems['cal',method];tune=systems['tune',method]
                    param=selection['selections'][method]['selected'];candidates=np.unique(np.quantile(tune['margin'],np.linspace(0,1,101)))
                    archive[method+'_candidates']=candidates
                    common=dict(encoder=enc,dataset=ds,part=p,order=order,session=session,method=method,reference_closed_test=closed)
                    results.append(dict(**common,policy='fixed_gate',offset=param['offset'],fallback=False,**rv.evaluate(tc,param['offset'])))
                    for pol in ['protection','paired_accuracy']:
                        detail=fairness.evaluate_policy(cc,refc,wc,candidates,tc,reft,closed,pol)
                        assert abs(detail['old']-(closed-detail['destroyed_test']+detail['recovered_test']))<1e-9
                        results.append(dict(**common,**detail))
                    for ratio in [.80,.85,.90,.95,.99]:
                        off,ceiling=rv.threshold(tc,ratio*closed)
                        results.append(dict(**common,policy='matched_test',retention=ratio,feasible=off is not None,ceiling=ceiling,offset=off,**(rv.evaluate(tc,off) if off is not None else {})))
                raw=dest.with_suffix('.npz');atomic_npz(raw,**archive)
                atomic_json(dict(selection_sha256=selected_hash,raw_sha256=digest_file(raw),records=results),dest);count+=len(results)
        atomic_json(dict(complete=True,selection_sha256=selected_hash,part=p,reference_closed_test=closed),po/'complete.json')
        print('RETUNING COMPLETE',enc,ds,p,'seconds',round(time.perf_counter()-start,1),flush=True)
    if parts==10:atomic_json(dict(complete=True,partitions=10,seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0),folder/'complete.json')
    del d,Ef,Es,T;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--encoder',choices=['l14','b32']);ap.add_argument('--dataset',choices=['cifar','pets','imagenetr']);ap.add_argument('--parts',type=int,default=10);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    device=configure();check_pause();torch.manual_seed(27102026)
    out=ROOT/('experiments/kbs_full_fusion_retuning_smoke' if args.smoke else 'experiments/kbs_full_fusion_retuning')
    for e in ([args.encoder] if args.encoder else ['l14','b32']):
        for d in ([args.dataset] if args.dataset else ['cifar','pets','imagenetr']):run(e,d,device,out,args.parts)
    if args.parts==10 and not args.encoder and not args.dataset:atomic_json(dict(complete=True,conditions=6,partitions=60),out/'complete.json')
if __name__=='__main__':main()
