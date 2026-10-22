"""Coverage/label-budget curves with the original independently fitted models."""
from pathlib import Path
import argparse,gc,json,sys,time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fcs_expansion as core
import fcs_retention_value as rv
import kbs_strengthen as prior
import kbs_depth as depth
import kbs_revision as revision
import kbs_independent_risk as independent
import kbs_risk_control as risk
import kbs_calibration_pool as pools
from kbs_runtime import configure,check_pause,atomic_json,atomic_npz,digest_file

ROOT=Path(__file__).resolve().parents[2]
OUT=pools.OUT
CAPS=(2,5,10,20,40)
METHODS=independent.METHODS


def nested_subsets(rows,seed):
    rng=np.random.default_rng(771000+seed)
    y=np.array([r['class_id'] for r in rows]);order={}
    for c in np.unique(y):order[c]=rng.permutation(np.flatnonzero(y==c))
    return {cap:np.sort(np.concatenate([ix[:min(cap,len(ix))] for ix in order.values()])) for cap in CAPS}


def compact(z,method,labels,base,novel,prefix=''):
    c={k:z[prefix+method+'_'+k] for k in ['margin','base_prediction','novel_prediction']}
    bm,nm=np.isin(labels,base),np.isin(labels,novel)
    w=np.zeros(len(labels))
    for group in [base,novel]:
        for cl in group:
            ix=labels==cl
            if ix.any():w[ix]=1/(len(group)*ix.sum())
    c.update(base_correct=bm&(c['base_prediction']==labels),novel_correct=nm&(c['novel_prediction']==labels),
             base_mask=bm,novel_mask=nm,weights=w,nb=len(base),nn=len(novel))
    return c


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--encoder',choices=['l14','b32'],required=True)
    ap.add_argument('--dataset',choices=['cifar','pets','imagenetr'],required=True)
    ap.add_argument('--parts',type=int,default=10);args=ap.parse_args()
    device=configure();check_pause();torch.manual_seed(662026);start=time.perf_counter()
    oldroot=ROOT/f'experiments/kbs_independent_risk/{args.encoder}/{args.dataset}'
    oldmanifest=json.loads((oldroot/'manifest.json').read_text())
    if args.encoder=='l14':d,_,feature_path=prior.load(args.dataset)
    else:
        feature_path=ROOT/f'experiments/kbs_revision/b32_features/{args.dataset}/frozen.pt'
        d=torch.load(feature_path,map_location='cpu',weights_only=False)
    assert digest_file(feature_path)==oldmanifest['cache_sha256']
    pool_path=OUT/'pools'/f'{args.dataset}.json';pool=json.loads(pool_path.read_text())
    extra_path=OUT/f'features/{args.encoder}/{args.dataset}/extra.pt'
    verified=json.loads(extra_path.with_name('verification.json').read_text());assert verified['complete']
    assert digest_file(extra_path)==verified['feature_sha256']
    split_path=ROOT/f'experiments/kbs_independent_risk/splits/{args.dataset}.json'
    split=json.loads(split_path.read_text());assert digest_file(split_path)==pool['split_sha256']
    sources=[Path(__file__),Path(pools.__file__),Path(independent.__file__),Path(risk.__file__),
             Path(core.__file__),Path(rv.__file__),Path(depth.__file__),Path(revision.__file__),Path(prior.__file__)]
    config=dict(encoder=args.encoder,dataset=args.dataset,old_manifest_sha256=digest_file(oldroot/'manifest.json'),
                pool_sha256=digest_file(pool_path),extra_feature_sha256=digest_file(extra_path),
                protocol_sha256=digest_file(pools.PROTOCOL),sources={p.name:digest_file(p) for p in sources})
    out=OUT/args.encoder/args.dataset;out.mkdir(parents=True,exist_ok=True);mf=out/'manifest.json'
    if mf.exists():assert json.loads(mf.read_text())==config,'Immutable budget sources changed.'
    else:
        atomic_json(config,mf)
        for f in sources:(out/f.name).write_bytes(f.read_bytes())
    y=np.r_[d['fit_y'],d['val_y']];E=torch.cat([d['E_fit'],d['E_val']])
    extra=torch.load(extra_path,map_location='cpu',weights_only=False)['features'];j=0;blocks=[]
    for r in pool['rows']:
        if 'cached_index' in r:blocks.append(E[r['cached_index']]);assert y[r['cached_index']]==r['class_id']
        else:blocks.append(extra[j]);j+=1
    assert j==len(extra);Epool=torch.stack(blocks);ypool=np.array([r['class_id'] for r in pool['rows']])
    Ef=E[split['indices']['fit']].to(device);yf=y[split['indices']['fit']]
    Es,T=d['E_support'].to(device),d['T100'].to(device);ys=d['support_y']
    all_subsets={s:nested_subsets(pool['rows'],s) for s in [0,1,2]}
    del E,extra,blocks;gc.collect()
    entries=json.loads((ROOT/f'experiments/kbs_revision/{args.encoder}/{args.dataset}/partitions.json').read_text())
    for entry in entries[:args.parts]:
        check_pause();p=entry['seed'];po=out/f'p{p}';po.mkdir(exist_ok=True)
        selection_path=oldroot/f'p{p}/selection.json';selection_sha=digest_file(selection_path)
        if (po/'complete.json').exists():
            assert json.loads((po/'complete.json').read_text())['selection_sha256']==selection_sha;continue
        selection=json.loads(selection_path.read_text());params=selection['parameters']
        base,new=[np.array(entry[k]) for k in ['base','novel']]
        W=core.fit_ridge(Ef,yf,base,selection['ridge_lambda']);Pb=core.means(Ef,yf,base)
        rowids=np.flatnonzero(np.isin(ypool,base));Ec=Epool[rowids].to(device);yc=ypool[rowids]
        reference=prior.predict(core.z(Ec@T[base].T)+params['retained'][0]*core.z(core.linear(Ec,W)),base)==yc
        keys,labels=depth.make_keys(dict(E_fit=Ef,fit_y=yf,E_support=Es,support_y=ys),base,new)
        keys=keys.to(device);ids=np.r_[base,new];tip=selection['cache']['selected']
        counts=torch.tensor([np.sum(labels==c) for c in ids],device=device)
        cache=depth.retrieval(Ec,keys,labels,ids,tip['beta'])/counts[None]
        si=np.array(selection['support_indices']);closed=json.loads((oldroot/f'p{p}/complete.json').read_text())['reference_closed_test']
        oldcal=np.array(split['indices']['cal']);oldcal=oldcal[np.isin(y[oldcal],base)]
        lookup={int(idx):j for j,idx in enumerate(oldcal)}
        cached=[(i,lookup[pool['rows'][int(g)]['cached_index']]) for i,g in enumerate(rowids) if 'cached_index' in pool['rows'][int(g)]]
        ca=np.array([a for a,b in cached]);cb=np.array([b for a,b in cached])
        for order in [0,1]:
            ordered=np.random.default_rng(1000+p+100*order).permutation(new)
            for session in range(1,6):
                check_pause();path=po/f'o{order}_t{session}.json';rawpath=path.with_suffix('.npz')
                original=oldroot/f'p{p}/o{order}_t{session}.npz'
                if path.exists():
                    saved=json.loads(path.read_text());assert saved['original_sha256']==digest_file(original)
                    assert saved['calibration_archive_sha256']==digest_file(rawpath);continue
                with np.load(original) as z:old={k:z[k] for k in z.files}
                novel=ordered[:len(new)*session//5];active=np.r_[base,novel]
                if rawpath.exists():
                    with np.load(rawpath) as z:archive={k:z[k] for k in z.files}
                else:
                    Pn=core.means(Es[si],ys[si],novel);a=rv.arrays(Ec,T[active],Pb,Pn,W)
                    scores={m:rv.score(a,m,params[m]) for m in METHODS[:3]}
                    for m,scales in [('raw',[1.,1.,1.]),('fixed_temperature',selection['full_temperatures'])]:
                        scores[m]=rv.score(revision.simple_arrays(Ec,T[active],Pb,Pn,W,scales),'retained',params[m])
                    ix=[int(np.flatnonzero(ids==c)[0]) for c in active]
                    scores['tip_mean_calibrated']=100*(Ec@T[active].T)+tip['alpha']*cache[:,ix]
                    reference[ca]=old['cal_reference_correct'][cb]
                    archive=dict(pool_row_indices=rowids,labels=yc,base_classes=base,novel_classes=novel,
                                 reference_correct=reference)
                    maxdelta=0.
                    for m,score in scores.items():
                        c=rv.compact(score,active,yc,base,novel)
                        delta=float(np.max(np.abs(c['margin'][ca]-old['cal_'+m+'_margin'][cb])))
                        assert delta<5e-4,(m,delta);maxdelta=max(maxdelta,delta)
                        for k in ['margin','base_prediction','novel_prediction']:
                            # Reuse exact cached scores, avoiding batch-size rounding drift.
                            arr=c[k].copy();arr[ca]=old['cal_'+m+'_'+k][cb]
                            archive[m+'_'+k]=arr if k=='margin' else arr.astype(np.int16)
                    archive['cached_margin_max_delta']=np.array(maxdelta)
                    atomic_npz(rawpath,**archive)
                rows=[]
                test_compacts={m:compact(old,m,old['test_labels'],base,novel,'test_') for m in METHODS}
                full_compacts={m:compact(archive,m,yc,base,novel) for m in METHODS}
                for cap in CAPS:
                    for seed in ([0] if cap==40 else [0,1,2]):
                        take=np.flatnonzero(np.isin(rowids,all_subsets[seed][cap]));ss=yc[take]
                        w=risk.weights(ss,base);ref=archive['reference_correct'][take]
                        refacc=float(w@ref);ncs=[int(np.sum(ss==c)) for c in base]
                        common=dict(part=p,order=order,session=session,cap=cap,subset_seed=seed,
                                    calibration_queries=len(take),minimum_class_count=min(ncs),
                                    maximum_class_count=max(ncs),subset_sha256=core.digest(rowids[take]),
                                    reference_closed_cal=refacc,reference_closed_test=closed)
                        for m in METHODS:
                            cc={k:(v[take] if isinstance(v,np.ndarray) else v) for k,v in full_compacts[m].items()}
                            cc['weights']=w
                            tc=test_compacts[m]
                            for policy in ['empirical','risk']:
                                row=dict(**common,method=m,policy=policy)
                                if policy=='risk':row.update(risk.certify(cc,ref,w,old[m+'_candidates'],delta=.05));offset=row['offset']
                                else:
                                    offset,ceiling=rv.threshold(cc,.95*refacc)
                                    row.update(offset=offset,feasible=offset is not None,fallback=offset is None,calibration_ceiling=ceiling)
                                result=dict(old=closed,new=0.,overall=closed*len(base)/len(active),harmonic=0.) if offset is None else rv.evaluate(tc,offset)
                                row.update(result,test_target_met=result['old']>=.95*closed-1e-10,
                                           target_gap_pp=100*(result['old']-.95*closed),test_retention_ratio=result['old']/closed)
                                rows.append(row)
                atomic_json(dict(encoder=args.encoder,dataset=args.dataset,original_sha256=digest_file(original),
                                 calibration_archive_sha256=digest_file(rawpath),selection_sha256=selection_sha,records=rows),path)
        atomic_json(dict(complete=True,part=p,selection_sha256=selection_sha),po/'complete.json')
        print('BUDGET COMPLETE',args.encoder,args.dataset,p,flush=True)
    if args.parts==10:atomic_json(dict(complete=True,partitions=10,seconds_this_invocation=time.perf_counter()-start),out/'complete.json')


if __name__=='__main__':main()
