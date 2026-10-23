"""Independently reconstruct decisions, accuracy, calibration traces and scopes."""
from pathlib import Path
import json,math,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from kbs_runtime import digest_file,atomic_json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/kbs_reviewer_revision'
GRID=np.array([.01,.025,.05,.1,.2,.3,.4,.5,.65,.8,.95,1.])


def decision(z,prefix,method,labels,offset):
    bp=z[prefix+method+'_base_prediction'];npred=z[prefix+method+'_novel_prediction'];m=z[prefix+method+'_margin']
    choose=(offset>m)|((offset==m)&(npred<bp))
    return (bp==labels)&~choose,npred==labels,choose


def direct_p(ref,c,w):
    # Alternative four-state grouping, independent of production helper.
    values=c.astype(float)-.95*ref;levels=np.c_[w/w.max(),values]
    pairs,count=np.unique(levels,axis=0,return_counts=True)
    logs=np.log1p(GRID[:,None]*pairs[:,0][None]*pairs[:,1][None])@count
    le=float(np.logaddexp.reduce(logs)-math.log(len(GRID)))
    return math.exp(-max(0,le))


def protection_p(ref,c,w):
    q=float(w@((ref&~c).astype(float)+.05*(~ref)))
    if q>=.05:return 1.
    kl=(q*math.log(q/.05) if q else 0)+(1-q)*math.log((1-q)/.95)
    return math.exp(-max(0,kl)/w.max())


def verify_rows(rows,zcal,ztest,cp,tp,clabel,tlabel,base,novel,cref,tref,candidate_source):
    n=0;tests=0
    wc=np.array([1/(len(base)*np.sum(clabel==c)) for c in clabel])
    closed=np.mean([tref[tlabel==c].mean() for c in base])
    for row in rows:
        m=row['method'];policy=row['policy'];offset=row['offset'];n+=1
        if policy in ['protection','paired_accuracy']:
            trace=row['trace'];candidates=candidate_source[m+'_candidates']
            assert len(trace)<=len(candidates)
            accepted=None
            for i,step in enumerate(trace):
                assert step['offset']==float(candidates[i])
                correct=decision(zcal,cp,m,clabel,step['offset'])[0]
                p=(protection_p if policy=='protection' else direct_p)(cref,correct,wc);tests+=1
                assert np.isclose(p,step['pvalue'],atol=1e-10,rtol=1e-8),(m,policy,p,step)
                passed=p<=.01;assert passed==step['accepted']
                if passed:accepted=step['offset']
                else:assert i==len(trace)-1
            assert accepted==offset
            assert (offset is None)==row['fallback']
        if offset is None:
            if policy=='matched_test':continue
            ab,an=closed,0.
        else:
            corr,ncorr,choose=decision(ztest,tp,m,tlabel,offset)
            ab=float(np.mean([corr[tlabel==c].mean() for c in base]))
            an=float(np.mean([(ncorr&choose)[tlabel==c].mean() for c in novel]))
        assert np.isclose(ab,row['old'],atol=1e-10) and np.isclose(an,row['new'],atol=1e-10)
        if 'test_target_met' in row:assert (ab>=.95*closed-1e-10)==row['test_target_met']
    return n,tests


def main():
    start=time.perf_counter();count=tests=0;source_hashes={}
    for encoder in ['l14','b32']:
        for dataset in ['cifar','pets','imagenetr']:
            old=ROOT/f'experiments/kbs_independent_risk/{encoder}/{dataset}'
            meta=json.loads((old/'manifest.json').read_text())
            if encoder=='b32':cache=ROOT/f'experiments/kbs_revision/b32_features/{dataset}/frozen.pt'
            else:
                cache=ROOT/({'cifar':'src/expand_cache.pt','pets':'experiments/fcs_pets/pets_frozen.pt','imagenetr':'experiments/kbs_imagenetr/frozen_imagenetr.pt'}[dataset])
            source_hashes[f'{encoder}/{dataset}']=digest_file(cache);assert source_hashes[f'{encoder}/{dataset}']==meta['cache_sha256']
            for part in range(10):
                for order in [0,1]:
                    for t in range(1,6):
                        tail=f'{encoder}/{dataset}/p{part}/o{order}_t{t}'
                        fp=OUT/'fairness'/(tail+'.json');fv=json.loads(fp.read_text())
                        calpath=ROOT/'experiments/kbs_calibration_budget'/(tail+'.npz');testpath=ROOT/'experiments/kbs_independent_risk'/(tail+'.npz')
                        assert digest_file(calpath)==fv['inputs']['calibration_sha256'];assert digest_file(testpath)==fv['inputs']['test_sha256']
                        with np.load(calpath) as c,np.load(testpath) as z:
                            n,k=verify_rows(fv['records'],c,z,'','test_',c['labels'],z['test_labels'],c['base_classes'],c['novel_classes'],c['reference_correct'],z['test_reference_correct'],z)
                            count+=n;tests+=k
                        rp=OUT/'readouts'/(tail+'.json');record=json.loads(rp.read_text());raw=rp.with_suffix('.npz')
                        assert digest_file(raw)==record['raw_sha256']
                        assert digest_file(old/f'p{part}/selection.json')==record['frozen_selection_sha256']
                        with np.load(raw) as z:
                            clean=[r for r in record['records'] if 'corruption' not in r]
                            n,k=verify_rows(clean,z,z,'cal_','test_',z['cal_labels'],z['test_labels'],z['base_classes'],z['novel_classes'],z['cal_reference_correct'],z['test_reference_correct'],z)
                            count+=n;tests+=k
                            with np.load(testpath) as original:
                                assert np.array_equal(z['test_labels'],original['test_labels'])
                                assert np.array_equal(z['test_reference_correct'],original['test_reference_correct'])
                            shifted=[r for r in record['records'] if 'corruption' in r]
                            for corruption in ['noise08','blur125']:
                                for calibration in ['clean','matched_shift']:
                                    rows=[r for r in shifted if r['corruption']==corruption and r['calibration']==calibration]
                                    if not rows:continue
                                    ck='cal' if calibration=='clean' else 'cal_'+corruption;tk='test_'+corruption
                                    n,k=verify_rows(rows,z,z,ck+'_',tk+'_',z[ck+'_labels'],z[tk+'_labels'],z['base_classes'],z['novel_classes'],z[ck+'_reference_correct'],z[tk+'_reference_correct'],z)
                                    count+=n;tests+=k
            print('VERIFIED REVIEWER REVISION',encoder,dataset,flush=True)
    # Explicitly verify transformed calibration/test image identities remain disjoint.
    shift=json.loads((OUT/'shift_features/manifest.json').read_text())
    assert len(shift['calibration_raw_ids'])==4000 and len(shift['test_raw_ids'])==2000
    assert len(set(shift['test_raw_ids']))==2000
    atomic_json(dict(complete=True,metric_records=count,reconstructed_threshold_tests=tests,original_feature_hashes=source_hashes,
                     original_references_unchanged=True,transformed_queries=24000,
                     seconds=time.perf_counter()-start),OUT/'verification.json')
    print('VERIFIED',count,'records',tests,'tests',flush=True)


if __name__=='__main__':main()
