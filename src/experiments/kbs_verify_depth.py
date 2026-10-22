"""Independently reconstruct depth metrics and thresholds from archived margins."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def weights(y,base,novel):
    out=np.zeros(len(y))
    for group in [base,novel]:
        for c in group:
            n=np.sum(y==c)
            if n:out[y==c]=1/(len(group)*n)
    return out


def accuracy(z,method,offset,w,prefix=''):
    name=prefix+method+'_';margin=z[name+'margin']
    new=(offset>margin)|((offset==margin)&(z[name+'novel_prediction']<z[name+'base_prediction']))
    return (float(w[z[name+'base_correct']&~new].sum()),float(w[z[name+'novel_correct']&new].sum()))


def verify(dataset):
    folder=ROOT/'experiments/kbs_depth'/dataset;manifest=json.loads((folder/'manifest.json').read_text())
    for name,sha in manifest['sources'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha
    assert hashlib.sha256((ROOT/'experiments/kbs_depth_protocol.md').read_bytes()).hexdigest()==manifest['protocol_sha256']
    assert hashlib.sha256((ROOT/f'experiments/kbs_strengthening/{dataset}/run_manifest.json').read_bytes()).hexdigest()==manifest['primary_manifest_sha256']
    expected={'tip':3,'robust':27};result={}
    for stage,n in expected.items():
        dirs=sorted(p.parent for p in (folder/stage).glob('*/complete.json'))
        assert len(dirs)==n,(dataset,stage,len(dirs),n)
        checked=0;rows_checked=0;matching_max=0.;selection_points=0
        for out in dirs:
            files=sorted(out.glob('o*_t*.json'));assert len(files)==10
            for path in files:
                rows=json.loads(path.read_text())['records'];targets={}
                with np.load(path.with_suffix('.npz')) as stored:
                    z={k:stored[k] for k in stored.files}
                    base,novel=z['base_classes'],z['novel_classes'];ids=z['active_classes'];y=z['labels'];vy=z['validation_labels']
                    assert np.array_equal(ids,np.r_[base,novel]);assert not np.isin(vy,novel).any()
                    w=weights(y,base,novel);vw=weights(vy,base,novel)
                    methods=set(r['method'] for r in rows)
                    for m in methods:
                        for prefix,labels,ww in [('',y,w),('validation_',vy,vw)]:
                            assert np.allclose(z[prefix+m+'_weights'],ww,atol=1e-14)
                            assert np.array_equal(z[prefix+m+'_base_correct'],np.isin(labels,base)&(z[prefix+m+'_base_prediction']==labels))
                            assert np.array_equal(z[prefix+m+'_novel_correct'],np.isin(labels,novel)&(z[prefix+m+'_novel_prediction']==labels))
                    for row in rows:
                        rows_checked+=1;m=row['method'];evaluation=row['evaluation']
                        if evaluation!='fixed_gate':
                            key=(evaluation,row['ratio'])
                            if key in targets:assert abs(targets[key]-row['target'])<1e-12
                            targets[key]=row['target']
                            prefix='' if evaluation=='matched_test' else 'validation_';ww=w if prefix=='' else vw
                            ceiling=float(ww[z[prefix+m+'_base_correct']].sum())
                            assert abs(ceiling-row['selection_base_ceiling'])<1e-12
                            if not row['feasible']:
                                assert 'new' not in row and row['offset'] is None and ceiling<row['target']+1e-10
                                continue
                            selected=accuracy(z,m,row['offset'],ww,prefix)[0]
                            assert selected>=row['target']-1e-10
                            assert abs(selected-row['selection_base_accuracy'])<1e-12
                            # Immediately increasing the boundary must violate the target.
                            bigger=np.nextafter(float(row['offset']),np.inf)
                            assert accuracy(z,m,bigger,ww,prefix)[0]<row['target']+1e-10
                            selection_points+=1
                        old,new=accuracy(z,m,row['offset'],w)
                        assert abs(old-row['old'])<1e-12 and abs(new-row['new'])<1e-12
                        assert abs((old*len(base)+new*len(novel))/len(ids)-row['overall'])<1e-12
                        assert abs(2*old*new/max(old+new,1e-12)-row['harmonic'])<1e-12
                        checked+=1
                # Trial 0/five-shot is a reconstruction of the old four table readouts.
                if stage=='robust' and '_r0_k5' in out.name:
                    p=rows[0]['part'];oldpath=ROOT/f'experiments/kbs_strengthening/{dataset}/p{p}'/path.name
                    oldrows=json.loads(oldpath.read_text())['records']
                    for row in rows:
                        if row['method']=='dynamic':continue
                        other=next(r for r in oldrows if (r['method'],r['evaluation'],r.get('ratio'))==(row['method'],row['evaluation'],row.get('ratio')))
                        assert row.get('feasible')==other.get('feasible')
                        for key in ['old','new','overall','harmonic']:
                            if key in row:matching_max=max(matching_max,abs(row[key]-other[key]))
        assert matching_max<1e-7
        result[stage]={'complete_configurations':n,'records':rows_checked,'reconstructed_metrics':checked,
                       'maximal_feasible_thresholds_verified':selection_points,'original_five_shot_max_delta':matching_max}
    return result


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--datasets',nargs='+',default=['cifar','pets','imagenetr']);args=ap.parse_args()
    result={name:verify(name) for name in args.datasets}
    (ROOT/'experiments/kbs_depth/verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
