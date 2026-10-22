"""Independent reconstruction of pool exclusions, budget decisions and metrics."""
from pathlib import Path
import gc,hashlib,json,math
import numpy as np
import torch
import kbs_calibration_pool as pools

ROOT=pools.ROOT
OUT=pools.OUT


def macro_w(y,classes):
    w=np.zeros(len(y))
    for c in classes:
        ix=y==c;assert ix.any();w[ix]=1/(len(classes)*ix.sum())
    return w


def subsets(rows,seed):
    rng=np.random.default_rng(771000+seed);y=np.array([r['class_id'] for r in rows]);order=[]
    for c in sorted(set(y)):order.append(rng.permutation(np.flatnonzero(y==c)))
    return {cap:np.sort(np.concatenate([a[:cap] for a in order])) for cap in [2,5,10,20,40]}


def main():
    pool_reports={}
    for ds in ['cifar','pets','imagenetr']:
        pool=json.loads((OUT/'pools'/f'{ds}.json').read_text());rows=pool['rows']
        original,y,protected,_,_,sources=pools.identities(ds)
        split=json.loads((ROOT/f'experiments/kbs_independent_risk/splits/{ds}.json').read_text())
        excluded=set(protected)|{original[i]['sha256'] for k in ['fit','tune'] for i in split['indices'][k]}
        assert excluded==set(pool['protected_hashes']) and sources==pool['sources']
        assert len(rows)==len({r['sha256'] for r in rows}) and not excluded&{r['sha256'] for r in rows}
        oldcal=set(split['indices']['cal'])
        for r in rows:
            if 'cached_index' in r:
                idx=r['cached_index'];assert idx in oldcal
                assert original[idx]['class_id']==r['class_id'] and original[idx]['sha256']==r['sha256']
        # Check supplemental source pixels independently of their saved feature files.
        extra=[r for r in rows if 'cached_index' not in r]
        if ds=='imagenetr':
            for r in extra:
                assert pools.digest_file(pools.CFG.data_root/'imagenet-r'/r['identity'])==r['sha256']
        else:
            pixels=pools.pixels(ds,extra)
            for r,im in zip(extra,pixels):assert pools.pixel_hash(im)==r['sha256']
            del pixels
        gc.collect()
        pool_reports[ds]=dict(rows=len(rows),supplemental=len(extra),excluded_overlap=0,
                              minimum_class_count=min(r['available_selected'] for r in pool['class_counts']))
    checked=tests=parts=0
    for encoder in ['l14','b32']:
        for ds in ['cifar','pets','imagenetr']:
            root=OUT/encoder/ds;manifest=json.loads((root/'manifest.json').read_text())
            for name,sha in manifest['sources'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
            pool_path=OUT/'pools'/f'{ds}.json';assert pools.digest_file(pool_path)==manifest['pool_sha256']
            pool=json.loads(pool_path.read_text());draw={s:subsets(pool['rows'],s) for s in [0,1,2]}
            fv=OUT/f'features/{encoder}/{ds}/verification.json';verified=json.loads(fv.read_text())
            assert verified['complete'] and verified['min_witness_cosine']>.999
            assert pools.digest_file(fv.with_name('extra.pt'))==verified['feature_sha256']
            for p in range(10):
                po=root/f'p{p}';completion=json.loads((po/'complete.json').read_text());assert completion['complete'];parts+=1
                oldpo=ROOT/f'experiments/kbs_independent_risk/{encoder}/{ds}/p{p}'
                assert pools.digest_file(oldpo/'selection.json')==completion['selection_sha256']
                closed=json.loads((oldpo/'complete.json').read_text())['reference_closed_test']
                for order in [0,1]:
                    for session in range(1,6):
                        path=po/f'o{order}_t{session}';saved=json.loads(path.with_suffix('.json').read_text())
                        original=oldpo/f'o{order}_t{session}.npz'
                        assert pools.digest_file(original)==saved['original_sha256']
                        assert pools.digest_file(path.with_suffix('.npz'))==saved['calibration_archive_sha256']
                        with np.load(original) as z:old={k:z[k] for k in z.files}
                        with np.load(path.with_suffix('.npz')) as z:raw={k:z[k] for k in z.files}
                        base,novel,labels,rows=[raw[k] for k in ['base_classes','novel_classes','labels','pool_row_indices']]
                        assert all(pool['rows'][int(i)]['class_id']==int(c) for i,c in zip(rows,labels))
                        assert np.isin(labels,base).all();wt=macro_w(old['test_labels'],base);wn=macro_w(old['test_labels'],novel)
                        assert abs(float(wt@old['test_reference_correct'])-closed)<1e-12
                        assert float(raw['cached_margin_max_delta'])<5e-4
                        def prediction(z,m,d,prefix=''):
                            margin=z[prefix+m+'_margin'];bp=z[prefix+m+'_base_prediction'];npred=z[prefix+m+'_novel_prediction']
                            new=(d>margin)|((d==margin)&(npred<bp));return np.where(new,npred,bp)
                        assert len(saved['records'])==156
                        for r in saved['records']:
                            checked+=1;cap,seed,m=r['cap'],r['subset_seed'],r['method']
                            assert cap!=40 or seed==0
                            ix=np.flatnonzero(np.isin(rows,draw[seed][cap]));y=labels[ix]
                            assert len(ix)==r['calibration_queries']
                            assert hashlib.sha256(np.ascontiguousarray(rows[ix]).tobytes()).hexdigest()==r['subset_sha256']
                            ns=[int((y==c).sum()) for c in base]
                            assert min(ns)==r['minimum_class_count'] and max(ns)==r['maximum_class_count']
                            w=macro_w(y,base);ref=raw['reference_correct'][ix].astype(bool)
                            assert abs(float(w@ref)-r['reference_closed_cal'])<1e-12
                            if r['policy']=='risk':
                                accepted=[];trace=r['trace'];candidates=old[m+'_candidates']
                                for j,t in enumerate(trace):
                                    assert t['offset']==candidates[j]
                                    correct=prediction(raw,m,t['offset'])[ix]==y
                                    q=float(w@((ref&~correct).astype(float)+.05*(~ref)))
                                    if q>=.05:pvalue=1.
                                    else:
                                        kl=(q*math.log(q/.05) if q else 0)+(1-q)*math.log((1-q)/.95)
                                        pvalue=math.exp(-max(0.,kl)/w.max())
                                    assert abs(q-t['loss'])<1e-12 and abs(pvalue-t['pvalue'])<1e-9
                                    assert t['accepted']==(pvalue<=.01)
                                    if t['accepted']:accepted.append(t['offset'])
                                    else:assert j==len(trace)-1
                                    tests+=1
                                assert r['offset']==(accepted[-1] if accepted else None)
                            else:
                                ceiling=float(w@(raw[m+'_base_prediction'][ix]==y));target=.95*r['reference_closed_cal']
                                assert r['feasible']==(ceiling>=target-1e-10)
                                if r['offset'] is not None:
                                    assert float(w@(prediction(raw,m,r['offset'])[ix]==y))>=target-1e-10
                                    assert float(w@(prediction(raw,m,np.nextafter(r['offset'],np.inf))[ix]==y))<target+1e-10
                            assert r['fallback']==(r['offset'] is None)
                            if r['offset'] is None:a,b=closed,0.
                            else:
                                correct=prediction(old,m,r['offset'],'test_')==old['test_labels']
                                a,b=float(wt@correct),float(wn@correct)
                            assert abs(a-r['old'])<1e-12 and abs(b-r['new'])<1e-12
                            assert r['test_target_met']==(a>=.95*closed-1e-10)
                            assert abs(r['target_gap_pp']-100*(a-.95*closed))<1e-10
    assert parts==60 and checked==93600
    report=dict(complete=True,partition_runs=parts,metric_records=checked,reconstructed_tests=tests,
                pools=pool_reports,protected_pool_overlap=0,original_models_and_test_archives_unchanged=True)
    pools.atomic_json(report,OUT/'verification.json');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
