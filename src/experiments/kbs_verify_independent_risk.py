"""Reconstruct independent-calibration decisions from compressed predictions."""
from pathlib import Path
import hashlib,json,math
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/kbs_independent_risk'


def macro_w(y,classes):
    out=np.zeros(len(y))
    for c in classes:
        mask=y==c;assert mask.any();out[mask]=1/(len(classes)*mask.sum())
    return out


def verify():
    count=0;decisions=0;parts=0
    for dataset in ['cifar','pets','imagenetr']:
        split=json.loads((OUT/'splits'/f'{dataset}.json').read_text())
        a,b,c=[set(split['indices'][k]) for k in ['fit','tune','cal']]
        assert not(a&b or a&c or b&c)
        assert sorted(a|b|c)==list(range(len(a)+len(b)+len(c)))
        assert all(r['fit']>=2 and r['tune']>=1 and r['cal']>=1 for r in split['class_counts'])
    for encoder in ['l14','b32']:
        for dataset in ['cifar','pets','imagenetr']:
            folder=OUT/encoder/dataset;manifest=json.loads((folder/'manifest.json').read_text())
            for name,sha in manifest['sources'].items():
                assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha
            assert hashlib.sha256((ROOT/'experiments/kbs_independent_risk_protocol.md').read_bytes()).hexdigest()==manifest['protocol_sha256']
            for part in range(10):
                po=folder/f'p{part}';completion=json.loads((po/'complete.json').read_text());assert completion['complete'];parts+=1
                selection=json.loads((po/'selection.json').read_text())
                assert selection['split_sha256']==hashlib.sha256((OUT/'splits'/f'{dataset}.json').read_bytes()).hexdigest()
                for order in [0,1]:
                    for session in range(1,6):
                        path=po/f'o{order}_t{session}';record=json.loads(path.with_suffix('.json').read_text())
                        with np.load(path.with_suffix('.npz')) as raw:z={k:raw[k] for k in raw.files}
                        base,novel,y,yc=[z[k] for k in ['base_classes','novel_classes','test_labels','cal_labels']]
                        assert not set(base)&set(novel) and np.isin(yc,base).all()
                        wc=macro_w(yc,base);wt=macro_w(y,base);wn=macro_w(y,novel)
                        refc=z['cal_reference_correct'].astype(bool);reft=z['test_reference_correct'].astype(bool)
                        ref_cal=float(wc@refc);ref_test=float(wt@reft)
                        assert abs(ref_cal-completion['reference_closed_cal'])<1e-12
                        assert abs(ref_test-completion['reference_closed_test'])<1e-12
                        def correctness(pool,m,d):
                            yy=yc if pool=='cal' else y
                            margin=z[pool+'_'+m+'_margin'];bp=z[pool+'_'+m+'_base_prediction'];npred=z[pool+'_'+m+'_novel_prediction']
                            choose=(d>margin)|((d==margin)&(npred<bp))
                            return np.where(choose,npred,bp)==yy
                        for r in record['records']:
                            count+=1;m=r['method'];offset=r['offset']
                            if r['policy']=='risk':
                                trace=r['trace'];candidates=z[m+'_candidates']
                                assert len(trace)<=len(candidates)
                                accepted=[]
                                for j,t in enumerate(trace):
                                    assert t['offset']==candidates[j]
                                    correct=correctness('cal',m,t['offset'])
                                    loss=(refc&~correct).astype(float)+.05*(~refc)
                                    q=float(wc@loss)
                                    if q>=.05:p=1.
                                    else:
                                        kl=(q*math.log(q/.05) if q else 0)+(1-q)*math.log((1-q)/.95)
                                        p=math.exp(-max(0.,kl)/wc.max())
                                    assert abs(q-t['loss'])<1e-12 and abs(p-t['pvalue'])<1e-10
                                    passed=p<=r['level']/5
                                    assert passed==t['accepted']
                                    if not passed:assert j==len(trace)-1
                                    else:accepted.append(t['offset'])
                                    decisions+=1
                                expected=accepted[-1] if accepted else None
                                assert expected==offset and r['fallback']==(expected is None)
                                assert r['certified_expansion']==(expected is not None)
                                assert abs(r['per_session_delta']-r['level']/5)<1e-12
                            else:
                                bp=z['cal_'+m+'_base_prediction'];ceiling=float(wc@(bp==yc));target=r['level']*ref_cal
                                assert r['feasible']==(ceiling>=target-1e-10)
                                if offset is not None:
                                    assert float(wc@correctness('cal',m,offset))>=target-1e-10
                                    assert float(wc@correctness('cal',m,np.nextafter(offset,np.inf)))<target+1e-10
                            if offset is None:old,new=ref_test,0.
                            else:
                                correct=correctness('test',m,offset);old,new=float(wt@correct),float(wn@correct)
                            assert abs(old-r['old'])<1e-12 and abs(new-r['new'])<1e-12
                            assert abs((len(base)*old+len(novel)*new)/(len(base)+len(novel))-r['overall'])<1e-12
                            assert r['test_target_met']==(old>=.95*ref_test-1e-10)
                            assert abs(r['target_gap_pp']-100*(old-.95*ref_test))<1e-10
    result=dict(complete=True,partition_runs=parts,metric_records=count,reconstructed_tests=decisions,
                disjoint_pool_indices='passed',independent_numeric_reconstruction='passed')
    assert parts==60 and count==21600
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':verify()
