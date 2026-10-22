"""Common base-accuracy budgets and paired classifier-value measurements.

See experiments/fcs_retention_value_protocol.md for the locked evaluation.
Matched-test frontiers are descriptive. Deployable offsets use base validation.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,hashlib,json,time
import numpy as np
import torch
import fcs_expansion as core
from fcs_review_experiments import predict
from lt_dataset import make_lt_datasets
from train import split_train_val

ROOT=Path(__file__).resolve().parents[2]
RATIOS=(.80,.85,.90,.95,.99)
MIXES=(0.,.25,.5,1.,2.,4.)
LEARNED=(0.,.25,.5,1.,2.)
OFFSETS=tuple(float(v) for v in np.arange(-4,4.01,.25))
METHODS=('retained','removed_fixed','no_learned_tuned','anchored_tp')

def compact(scores,ids,y,base,new):
    """Sufficient statistics for a group-offset sweep; label ties are canonical."""
    nb=len(base)
    bp=predict(scores[:,:nb],ids[:nb]);npred=predict(scores[:,nb:],ids[nb:])
    margin=scores[:,:nb].max(1).values.double().cpu().numpy()-scores[:,nb:].max(1).values.double().cpu().numpy()
    weights=np.zeros(len(y),dtype=float)
    for group in (base,new):
        for c in group:
            count=np.sum(y==c)
            if count:weights[y==c]=1./(len(group)*count)
    bmask=np.isin(y,base);nmask=np.isin(y,new)
    return {'margin':margin,'base_prediction':bp,'novel_prediction':npred,
      'base_correct':bmask&(bp==y),'novel_correct':nmask&(npred==y),
      'base_mask':bmask,'novel_mask':nmask,'weights':weights,
      'nb':nb,'nn':len(new)}

def evaluate(c,offset):
    # This comparison avoids adding a tiny double offset to float32 score arrays.
    choose_new=(offset>c['margin'])|((offset==c['margin'])&(c['novel_prediction']<c['base_prediction']))
    ab=float(c['weights'][c['base_correct']&~choose_new].sum())
    an=float(c['weights'][c['novel_correct']&choose_new].sum())
    return {'old':ab,'new':an,'overall':(ab*c['nb']+an*c['nn'])/(c['nb']+c['nn']),
            'harmonic':2*ab*an/max(ab+an,1e-12)}

def threshold(c,target):
    """Largest feasible threshold at a common macro-base constraint.

    Scores are fixed; only the base/novel admission offset varies. Return None
    when the method's within-base ceiling cannot meet the requested constraint.
    """
    ix=np.flatnonzero(c['base_correct'])
    ceiling=float(c['weights'][ix].sum())
    if target>ceiling+1e-10:return None,ceiling
    if target<=0:return float('inf'),ceiling
    order=ix[np.argsort(-c['margin'][ix],kind='stable')]
    cum=np.cumsum(c['weights'][order])
    j=min(int(np.searchsorted(cum,target-1e-12)),len(order)-1)
    boundary=float(c['margin'][order[j]])
    offset=boundary if evaluate(c,boundary)['old']>=target-1e-10 else float(np.nextafter(boundary,-np.inf))
    assert evaluate(c,offset)['old']>=target-1e-10
    return offset,ceiling

def arrays(E,T,Pb,Pn,W):
    nb=len(Pb);old=E@T[:nb].T
    return {'text':torch.cat([core.z(old),core.z(E@T[nb:].T,old)],1),
            'learned':core.z(core.linear(E,W)),
            'prototype':core.z(E@torch.cat([Pb,Pn]).T,E@Pb.T),'nb':nb}

def score(a,method,parameters):
    out=a['text'].clone();l,u,_=parameters;nb=a['nb']
    if method=='retained':out[:,:nb]+=l*a['learned']
    if method=='anchored_tp':out+=u*a['prototype']
    else:out[:,nb:]+=u*a['prototype'][:,nb:]
    return out

def calibrate(a,method,ids,y,base,new):
    best=(-1.,None)
    for l in (LEARNED if method=='retained' else (0.,)):
        for u in MIXES:
            c=compact(score(a,method,(l,u,0.)),ids,y,base,new)
            for d in OFFSETS:
                h=evaluate(c,d)['harmonic']
                if h>best[0]+1e-12:best=(h,(l,u,d))
    return best[1]

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['cifar','pets'],required=True)
    args=ap.parse_args();torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    folder=ROOT/'experiments'/('fcs' if args.dataset=='cifar' else 'fcs_pets')
    out=folder/'retention_value';out.mkdir(exist_ok=True)
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    manifest=json.loads((folder/'manifest.json').read_text())
    if args.dataset=='pets':
        cache=torch.load(folder/'pets_frozen.pt',map_location='cpu',weights_only=False)
        yf,yv,ys,yt=[cache[k+'_y'] for k in ('fit','val','support','test')]
        Es=cache['E_support'].to(device)
    else:
        training,support,_,testing,_=make_lt_datasets(rho=100);fit,val=split_train_val(training)
        yf,yv,ys,yt=fit.fine,val.fine,support.fine,testing.fine
        cache_path=ROOT/'src/expand_cache.pt'
        if not cache_path.exists():cache_path=folder/'frozen_cifar.pt'
        cache=torch.load(cache_path,map_location='cpu',weights_only=False)
        Es=(cache['E_support'] if 'E_support' in cache else
            torch.load(folder/'frozen_support.pt',map_location='cpu',weights_only=False)['features']).to(device)
    Ef,Ev,Et,T=[cache[k].to(device) for k in ('E_fit','E_val','E_test','T100')]
    result={'protocol':'common-retention-classifier-value-v1','dataset':args.dataset,
      'source_sha256':source_hash,'ratios':RATIOS,'mixes':MIXES,'learned_grid':LEARNED,
      'calibration_offsets':OFFSETS,'partitions':[],'records':[]}
    start=time.perf_counter()
    for entry in manifest['partitions']:
        p=entry['seed'];base=np.array(entry['base']);new=np.array(entry['novel'])
        pb=np.array(entry['pseudo_base']);pn=np.array(entry['pseudo_novel']);ci=np.r_[pb,pn]
        W=core.fit_ridge(Ef,yf,base,entry['ridge_lambda']);Wp=core.fit_ridge(Ef,yf,pb,entry['pseudo_ridge_lambda'])
        Pb=core.means(Ef,yf,base);Ppb=core.means(Ef,yf,pb)
        si=core.support_rows(ys,new,5,7);psi=core.support_rows(ys,pn,5,7)
        Ppn=core.means(Es[psi],ys[psi],pn)
        mask=np.isin(yv,base);cal=arrays(Ev[mask],T[ci],Ppb,Ppn,Wp)
        parameters={method:calibrate(cal,method,ci,yv[mask],pb,pn)
                    for method in ('retained','no_learned_tuned','anchored_tp')}
        l,u,d=parameters['retained'];parameters['removed_fixed']=(0.,u,d)
        def closed(E,y):
            scores=core.z(E@T[base].T)+l*core.z(core.linear(E,W))
            pred=predict(scores,base)
            return float(np.mean([np.mean(pred[y==c]==c) for c in base]))
        ref_test=closed(Et,yt);ref_val=closed(Ev[mask],yv[mask])
        result['partitions'].append({'part':p,'parameters':parameters,
           'reference_closed_test':ref_test,'reference_closed_validation':ref_val,
           'base':base.tolist(),'novel':new.tolist()})
        for order in (0,1):
            ordered=np.random.default_rng(1000+p+100*order).permutation(new)
            for session in range(1,6):
                novel=ordered[:len(new)*session//5];ids=np.r_[base,novel]
                Pn=core.means(Es[si],ys[si],novel)
                ta=arrays(Et,T[ids],Pb,Pn,W);va=arrays(Ev[mask],T[ids],Pb,Pn,W)
                archive={}
                for method in METHODS:
                    tc=compact(score(ta,method,parameters[method]),ids,yt,base,novel)
                    vc=compact(score(va,method,parameters[method]),ids,yv[mask],base,novel)
                    for key in ['margin','base_prediction','novel_prediction','base_correct','novel_correct','weights']:
                        archive[method+'_'+key]=tc[key]
                    common={'part':p,'order':order,'session':session,'added':len(novel),
                            'method':method,'parameters':parameters[method]}
                    result['records'].append({**common,'evaluation':'fixed_gate',
                        'offset':parameters[method][2],**evaluate(tc,parameters[method][2])})
                    for ratio in RATIOS:
                        for evaluation,c,target in [('matched_test',tc,ratio*ref_test),('base_validation',vc,ratio*ref_val)]:
                            offset,ceiling=threshold(c,target)
                            row={**common,'evaluation':evaluation,'ratio':ratio,'target':target,
                                'selection_base_ceiling':ceiling,'feasible':offset is not None,'offset':offset}
                            if offset is not None:
                                row.update(evaluate(tc,offset))
                                row['selection_base_accuracy']=evaluate(c,offset)['old']
                                row['selection_overshoot']=row['selection_base_accuracy']-target
                            result['records'].append(row)
                np.savez_compressed(out/f'margins_p{p}_o{order}_t{session}.npz',
                                    active_classes=ids,base_classes=base,novel_classes=novel,labels=yt,**archive)
        result['elapsed_seconds']=time.perf_counter()-start
        (out/'results.json').write_text(json.dumps(result,indent=2))
        print(args.dataset,p,'gates',parameters,'closed',ref_test,ref_val,flush=True)
    print('FINISHED',args.dataset,len(result['records']),round(time.perf_counter()-start,2),flush=True)

if __name__=='__main__':main()
