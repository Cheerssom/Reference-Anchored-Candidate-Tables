"""Frozen-model operating-point and effective-bias diagnostics; no new encoding."""
from pathlib import Path
import os,sys,json,gc,time
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0')
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fcs_expansion as core
import fcs_retention_value as rv
import kbs_reviewer_revision as oldrun
import kbs_calibration_budget as budget
import kbs_calibration_fairness as fairness
import kbs_risk_control as risk
from kbs_runtime import configure,check_pause,atomic_json,atomic_npz,digest_file
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/kbs_mechanism_followup'
METHODS=['retained','anchored_tp']

def choose(c,d):
    return (d>c['margin'])|((d==c['margin'])&(c['novel_prediction']<c['base_prediction']))

def envelope(c):
    # Exact prediction-change events, including equality ties and adjacent floats.
    relevant=c['base_correct']|c['novel_correct']
    m=c['margin'][relevant].copy()
    first=c['novel_prediction'][relevant]<c['base_prediction'][relevant]
    events=np.where(first,m,np.nextafter(m,np.inf));order=np.argsort(events,kind='stable')
    events=events[order]
    db=-c['weights'][relevant]*c['base_correct'][relevant]
    dn=c['weights'][relevant]*c['novel_correct'][relevant]
    ends=np.r_[np.flatnonzero(events[1:]!=events[:-1]),len(events)-1]
    offsets=np.r_[-np.inf,events[ends]]
    ab=np.r_[c['weights'][c['base_correct']].sum(),c['weights'][c['base_correct']].sum()+np.cumsum(db[order])[ends]]
    an=np.r_[0.,np.cumsum(dn[order])[ends]]
    assert np.all(np.diff(ab)<=1e-12) and np.all(np.diff(an)>=-1e-12)
    # Independently evaluate deterministic witnesses using the original evaluator.
    for i in np.unique(np.linspace(0,len(offsets)-1,19,dtype=int)):
        actual=rv.evaluate(c,float(offsets[i]));assert abs(actual['old']-ab[i])<1e-10 and abs(actual['new']-an[i])<1e-10
    return dict(offset=offsets,base=ab,novel=an)

def capacity(curve,c,target):
    eligible=np.flatnonzero(curve['base']>=target-1e-10)
    if not len(eligible):return dict(feasible=False)
    # Last eligible event is most permissive and maximizes novel accuracy.
    j=eligible[-1];value=rv.evaluate(c,float(curve['offset'][j]))
    off,_=rv.threshold(c,target);assert off is not None
    original=rv.evaluate(c,off)
    assert abs(original['new']-value['new'])<1e-9
    return dict(feasible=True,base=value['old'],novel=value['new'],overshoot=value['old']-target)

def error_terms(c,reference,d,closed):
    ch=np.zeros(len(reference),bool) if d is None else choose(c,d)
    correct=reference.copy() if d is None else c['base_correct']&~ch
    b=c['base_mask'];n=c['novel_mask'];w=c['weights']
    Dn=float(w[b]@(reference[b]&~correct[b]&ch[b]));Do=float(w[b]@(reference[b]&~correct[b]&~ch[b]))
    G=float(w[b]@(~reference[b]&correct[b]));ab=float(w[b]@correct[b])
    an=0. if d is None else float(w[n]@(c['novel_correct'][n]&ch[n]))
    assert abs(ab-(closed-Dn-Do+G))<1e-10
    return dict(base=ab,novel=an,destroyed_to_novel=Dn,destroyed_to_base=Do,recovered=G,
        false_novelty_all_base=float(w[b]@ch[b]),novel_to_base=float(w[n]@(~ch[n])))

@torch.no_grad()
def bias_arrays(E,T,Pb,Pn,W,parameters):
    nb=len(Pb);a,u,_=parameters
    qT=E@T.T;qL=core.linear(E,W);qP=E@torch.cat([Pb,Pn]).T
    sigT=qT[:,:nb].std(1,correction=0).clamp_min(1e-6)
    sigL=qL.std(1,correction=0).clamp_min(1e-6)
    sigP=qP[:,:nb].std(1,correction=0).clamp_min(1e-6)
    muL=qL.mean(1);muP=qP[:,:nb].mean(1)
    delta=(a*muL.double()/sigL.double()-u*muP.double()/sigP.double()).cpu().numpy()
    # Compare the algebra with an independently formed float64 score matrix.
    raw=qT.double()/sigT[:,None].double()
    raw[:,:nb]+=a*qL.double()/sigL[:,None].double()
    raw[:,nb:]+=u*qP[:,nb:].double()/sigP[:,None].double()
    raw_margin=(raw[:,:nb].max(1).values-raw[:,nb:].max(1).values).cpu().numpy()
    centered=rv.score(rv.arrays(E,T,Pb,Pn,W),'retained',parameters)
    centered_margin=(centered[:,:nb].max(1).values.double()-centered[:,nb:].max(1).values.double()).cpu().numpy()
    error=float(np.max(np.abs(centered_margin-(raw_margin-delta))))
    assert error<1e-4,error
    return delta,raw_margin,centered,dict(max_algebra_margin_error=error)

def aucs(c,delta,raw_margin):
    mask=c['base_mask']|c['novel_mask'];y=c['novel_mask'][mask].astype(int);w=c['weights'][mask]
    return {name:float(roc_auc_score(y,score[mask],sample_weight=w)) for name,score in
        [('delta',delta),('raw_margin',-raw_margin),('centered_margin',-c['margin'])]}

@torch.no_grad()
def run_suite(enc,ds,device):
    folder=OUT/enc/ds;folder.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/f'experiments/kbs_revision/{enc}/{ds}/partitions.json',
        ROOT/f'experiments/kbs_independent_risk/splits/{ds}.json',
        ROOT/'experiments/kbs_mechanism_followup_protocol.md',Path(__file__)]
    manifest=dict(inputs={p.relative_to(ROOT).as_posix():digest_file(p) for p in paths})
    mf=folder/'manifest.json'
    if mf.exists():assert json.loads(mf.read_text())==manifest
    else:atomic_json(manifest,mf)
    if (folder/'complete.json').exists():return
    d,cache=oldrun.load_data(ds,enc);manifest_input=dict(cache_sha256=digest_file(cache))
    entries=json.loads(paths[0].read_text())
    Ef,Es,T=[d[k].to(device) for k in ['E_fit','E_support','T']]
    yf,ys=d['fit_y'],d['support_y']
    for entry in entries:
        check_pause();p=entry['seed'];po=folder/f'p{p}';po.mkdir(exist_ok=True)
        if (po/'complete.json').exists():continue
        old=ROOT/f'experiments/kbs_independent_risk/{enc}/{ds}/p{p}'
        sp=old/'selection.json';sel=json.loads(sp.read_text());parameters=sel['parameters']['retained']
        base=np.array(entry['base']);novel=np.array(entry['novel']);ids=np.r_[base,novel]
        W=core.fit_ridge(Ef,yf,base,sel['ridge_lambda']);Pb=core.means(Ef,yf,base)
        si=np.array(sel['support_indices']);Pn=core.means(Es[si],ys[si],novel)
        delta={};rawmargin={};rebuild={};witness={}
        for key in ['test','tune','cal']:
            labels=d[key+'_y'];mask=np.isin(labels,base) if key!='test' else np.ones(len(labels),bool)
            E=d['E_'+key][mask].to(device)
            de,raw,score,check=bias_arrays(E,T[ids],Pb,Pn,W,parameters)
            delta[key]=de;rawmargin[key]=raw;witness[key]=check
            rebuild[key]=rv.compact(score,ids,labels[mask],base,novel)
            del E,score
        for order in [0,1]:
            dest=po/f'o{order}.json'
            if dest.exists():continue
            tp=old/f'o{order}_t5.npz';cp=ROOT/f'experiments/kbs_calibration_budget/{enc}/{ds}/p{p}/o{order}_t5.npz'
            fp=ROOT/f'experiments/kbs_reviewer_revision/fairness/{enc}/{ds}/p{p}/o{order}_t5.json'
            with np.load(tp) as z0,np.load(cp) as z1:
                z={k:z0[k] for k in z0.files};zc={k:z1[k] for k in z1.files}
            y=z['test_labels'];ref=z['test_reference_correct'];closed=float(np.mean([ref[y==cl].mean() for cl in base]))
            assert np.array_equal(y,d['test_y'])
            assert np.array_equal(zc['labels'],d['cal_y'][np.isin(d['cal_y'],base)])
            c={m:budget.compact(z,m,y,base,novel,'test_') for m in METHODS}
            # Frozen scores are authoritative, reconstruction only checks identity.
            for key in ['base_prediction','novel_prediction']:
                assert np.array_equal(c['retained'][key],rebuild['test'][key]),(enc,ds,p,key)
            archive_error=float(np.max(abs(c['retained']['margin']-rebuild['test']['margin'])))
            assert archive_error<1e-4,archive_error
            curves={m:envelope(c[m]) for m in METHODS}
            policies=[r for r in json.loads(fp.read_text())['records'] if r['method'] in METHODS]
            rows=[];comparisons=[]
            for r in policies:
                m=r['method'];terms=error_terms(c[m],ref,r['offset'],closed)
                assert abs(terms['base']-r['old'])<1e-10 and abs(terms['novel']-r['new'])<1e-10
                if m=='retained':assert terms['recovered']==0 and terms['destroyed_to_base']==0
                rows.append(dict(method=m,policy=r['policy'],fallback=r['fallback'],offset=r['offset'],
                    reference=closed,target=.95*closed,slack=terms['base']-.95*closed,**terms))
            for pol in ['protection','paired_accuracy']:
                rr={r['method']:r for r in rows if r['policy']==pol};baseline=rr['anchored_tp'];ours=rr['retained']
                target=baseline['base'];matched={m:capacity(curves[m],c[m],target) for m in METHODS}
                common={m:capacity(curves[m],c[m],.95*closed) for m in METHODS}
                row=dict(policy=pol,baseline_realized_base=target,policy_gain=ours['novel']-baseline['novel'],
                    both_admitted=not ours['fallback'] and not baseline['fallback'],matched=matched,common_target_capacity=common)
                if all(v['feasible'] for v in matched.values()):
                    co,cb=matched['retained'],matched['anchored_tp'];go=co['novel']-ours['novel'];gb=cb['novel']-baseline['novel']
                    row.update(matched_retained_minus_baseline_policy=co['novel']-baseline['novel'],
                        envelope_gain=co['novel']-cb['novel'],ours_alignment_adjustment=go,baseline_alignment_adjustment=gb,
                        ours_selected_feasible_at_baseline=ours['base']>=target-1e-10)
                    assert abs(row['policy_gain']-(row['envelope_gain']-go+gb))<1e-10
                comparisons.append(row)
            oldcal=budget.compact(zc,'retained',zc['labels'],base,novel)
            assert np.max(abs(oldcal['margin']-rebuild['cal']['margin']))<1e-4
            # Remove centering, preserving query scales and group rankings.
            unc={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in c['retained'].items()}
            uncal={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in oldcal.items()}
            unc['margin']=c['retained']['margin']+delta['test'];uncal['margin']=oldcal['margin']+delta['cal']
            tuning_margin=rebuild['tune']['margin']+delta['tune']
            candidates=np.unique(np.quantile(tuning_margin,np.linspace(0,1,101)))
            unpolicy=fairness.evaluate_policy(uncal,zc['reference_correct'],risk.weights(zc['labels'],base),candidates,unc,ref,closed,'paired_accuracy')
            unterms=error_terms(unc,ref,unpolicy['offset'],closed);assert unterms['recovered']==0 and unterms['destroyed_to_base']==0
            active=c['retained']['base_mask']|c['retained']['novel_mask']
            rawout=po/f'o{order}.npz'
            arrays={f'{m}_{key}':value for m,cc in curves.items() for key,value in cc.items()}
            arrays.update(delta=delta['test'][active],uncentered_margin=rawmargin['test'][active],centered_margin=c['retained']['margin'][active],
                novel_mask=c['retained']['novel_mask'][active],weights=c['retained']['weights'][active],labels=y[active],
                uncentered_candidates=candidates,uncentered_cal_margin=uncal['margin'],cal_delta=delta['cal'])
            atomic_npz(rawout,**arrays)
            atomic_json(dict(encoder=enc,dataset=ds,part=p,order=order,session=5,rows=rows,comparisons=comparisons,
                bias_auc=aucs(c['retained'],delta['test'],rawmargin['test']),uncentered_direct=unpolicy,
                witness=witness,archive_margin_error=archive_error,source_hashes={str(q.relative_to(ROOT)):digest_file(q) for q in [tp,cp,fp,sp]},
                **manifest_input,raw_sha256=digest_file(rawout)),dest)
        atomic_json(dict(complete=True,part=p),po/'complete.json');print('MECHANISM',enc,ds,p,flush=True)
    del d,Ef,Es,T;gc.collect();torch.cuda.empty_cache()
    atomic_json(dict(complete=True,parts=10),folder/'complete.json')

def main():
    start=time.perf_counter();device=configure()
    for enc in ['l14','b32']:
        for ds in ['cifar','pets','imagenetr']:run_suite(enc,ds,device)
    atomic_json(dict(complete=True,seconds=time.perf_counter()-start,conditions=6,parts=10,orders=2,sessions=[5]),OUT/'complete.json')

if __name__=='__main__':main()
