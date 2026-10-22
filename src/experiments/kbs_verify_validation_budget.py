"""Independent reconstruction of validation-budget policies from raw archives."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def verify(dataset):
    folder=ROOT/'experiments/kbs_validation_budget'/dataset
    manifest=json.loads((folder/'manifest.json').read_text())
    for rel,sha in manifest['input_sha256'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha
    for name,sha in manifest['source_sha256'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha
    rows_checked=0;feasible=0;full_matches=0;max_delta=0.;units=0
    for part in range(3):
        for session in range(1,6):
            z={};original=[]
            for relative in [f'robust/p{part}_r0_k5/o0_t{session}',f'tip/p{part}/o0_t{session}']:
                prefix=ROOT/'experiments/kbs_depth'/dataset/relative
                with np.load(prefix.with_suffix('.npz')) as raw:
                    for key in raw.files:
                        if key in z:assert np.array_equal(z[key],raw[key])
                        else:z[key]=raw[key]
                original+=json.loads(prefix.with_suffix('.json').read_text())['records']
            base,novel,y,vy=[z[k] for k in ['base_classes','novel_classes','labels','validation_labels']]
            def weights(labels,groups):
                w=np.zeros(len(labels))
                for group in groups:
                    for c in group:
                        n=np.sum(labels==c)
                        if n:w[labels==c]=1/(len(group)*n)
                return w
            wt=weights(y,[base,novel]);test_ref=float(wt[np.isin(y,base)&(z['retained_base_prediction']==y)].sum())
            for path in sorted(folder.glob(f'p{part}_f*_s*_t{session}.json')):
                unit=json.loads(path.read_text());units+=1;indices=np.array(unit['selection_indices'])
                assert len(indices)==len(set(indices));labels=vy[indices]
                wv=weights(labels,[base]);rows=unit['records'];assert len(rows)==12
                reference=float(wv[z['validation_retained_base_prediction'][indices]==labels].sum())
                assert unit['validation_count']==len(indices) and unit['full_validation_count']==len(vy)
                assert abs(unit['actual_fraction']-len(indices)/len(vy))<1e-12
                for c in base:
                    assert np.sum(labels==c)==max(1,int(np.floor(rows[0]['fraction']*np.sum(vy==c))))
                for r in rows:
                    rows_checked+=1;m=r['method'];ratio=r['ratio'];offset=r['offset']
                    original_row=next(a for a in original if a['method']==m and a['evaluation']=='base_validation' and a['ratio']==ratio)
                    assert r['parameters']==original_row['parameters']
                    assert abs(r['validation_reference']-reference)<1e-12 and abs(r['target']-ratio*reference)<1e-12
                    assert abs(r['test_reference']-test_ref)<1e-12
                    prefix='validation_'+m+'_';bp=z[prefix+'base_prediction'][indices];npred=z[prefix+'novel_prediction'][indices]
                    correct=bp==labels;margin=z[prefix+'margin'][indices]
                    ceiling=float(wv[correct].sum());assert abs(ceiling-r['validation_ceiling'])<1e-12
                    if not r['feasible']:
                        assert offset is None and 'old' not in r and ceiling<r['target']+1e-10
                    else:
                        feasible+=1
                        def base_accuracy(d):
                            choose=(d>margin)|((d==margin)&(npred<bp))
                            return float(wv[correct&~choose].sum())
                        selected=base_accuracy(offset)
                        assert selected>=r['target']-1e-10 and abs(selected-r['validation_base'])<1e-12
                        assert base_accuracy(np.nextafter(float(offset),np.inf))<r['target']+1e-10
                        prefix=m+'_';margin=z[prefix+'margin'];bp=z[prefix+'base_prediction'];npred=z[prefix+'novel_prediction']
                        pick_new=(offset>margin)|((offset==margin)&(npred<bp));prediction=np.where(pick_new,npred,bp)
                        old=float(wt[(prediction==y)&np.isin(y,base)].sum());new=float(wt[(prediction==y)&np.isin(y,novel)].sum())
                        assert abs(old-r['old'])<1e-12 and abs(new-r['new'])<1e-12
                        assert abs(r['test_retention_ratio']-old/test_ref)<1e-12
                        assert r['test_target_met']==(old>=ratio*test_ref-1e-10)
                    if r['fraction']==1:
                        assert r['feasible']==original_row['feasible'];full_matches+=1
                        for key in ['old','new','offset']:
                            if r['feasible']:max_delta=max(max_delta,abs(r[key]-original_row[key]))
    assert units==105 and rows_checked==1260 and max_delta<1e-10
    assert json.loads((folder/'complete.json').read_text())['complete']
    return {'units':units,'records':rows_checked,'feasible_policies_reconstructed':feasible,
            'full_budget_original_matches':full_matches,'full_budget_max_delta':max_delta,'source_input_hashes_valid':True}


if __name__=='__main__':
    result={ds:verify(ds) for ds in ['cifar','pets','imagenetr']}
    before=json.loads((ROOT/'experiments/kbs_validation_budget/restart_before.json').read_text())
    p=ROOT/before['path'];restart={'saved_hash_unchanged':hashlib.sha256(p.read_bytes()).hexdigest()==before['sha256'],
                                 'saved_mtime_unchanged':p.stat().st_mtime_ns==before['mtime_ns']}
    assert all(restart.values());result['restart_reuse']=restart
    (ROOT/'experiments/kbs_validation_budget/verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
