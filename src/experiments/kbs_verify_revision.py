"""Reconstruct both encoder suites and old L/14 seeds from raw predictions."""
from pathlib import Path
import json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def verify(encoder,dataset):
    root=ROOT/'experiments/kbs_revision'/encoder/dataset
    manifest=json.loads((root/'manifest.json').read_text())
    for name,sha in manifest['sources'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
    assert hashlib.sha256((ROOT/'experiments/kbs_revision_protocol.md').read_bytes()).hexdigest()==manifest['protocol_sha256']
    partitions=json.loads((root/'partitions.json').read_text());assert len(partitions)==10
    assert len({tuple(p['base']) for p in partitions})==10
    count=0;checked=0;max_old_delta=0.;original_matches=0
    for p in range(10):
        out=root/f'p{p}';complete=json.loads((out/'complete.json').read_text());assert complete['complete']
        for order in [0,1]:
            for session in range(1,6):
                prefix=out/f'o{order}_t{session}';rows=json.loads(prefix.with_suffix('.json').read_text())['records']
                assert len(rows)==88
                with np.load(prefix.with_suffix('.npz')) as raw:z={k:raw[k] for k in raw.files}
                base,novel,y,vy=[z[k] for k in ['base_classes','novel_classes','labels','validation_labels']]
                assert not set(base)&set(novel) and np.isin(vy,base).all()
                def weights(labels):
                    w=np.zeros(len(labels))
                    for group in [base,novel]:
                        for c in group:
                            n=np.sum(labels==c)
                            if n:w[labels==c]=1/(len(group)*n)
                    return w
                wt,wv=weights(y),weights(vy)
                for m in {r['method'] for r in rows}:
                    for pre,labels,w in [('',y,wt),('validation_',vy,wv)]:
                        assert np.allclose(z[pre+m+'_weights'],w)
                        for group,name in [(base,'base'),(novel,'novel')]:
                            assert np.array_equal(z[pre+m+'_'+name+'_correct'],np.isin(labels,group)&(z[pre+m+'_'+name+'_prediction']==labels))
                targets={}
                for r in rows:
                    count+=1;m=r['method'];evaluation=r['evaluation']
                    if evaluation!='fixed_gate':
                        key=(evaluation,r['ratio'])
                        if key in targets:assert abs(targets[key]-r['target'])<1e-12
                        targets[key]=r['target']
                        pre='' if evaluation=='matched_test' else 'validation_';w=wt if pre=='' else wv
                        ref=complete['reference_closed_test' if pre=='' else 'reference_closed_validation']
                        assert abs(r['target']-r['ratio']*ref)<1e-12
                        correct=z[pre+m+'_base_correct'];ceiling=float(w[correct].sum())
                        assert abs(ceiling-r['selection_base_ceiling'])<1e-12
                        if not r['feasible']:
                            assert ceiling<r['target']+1e-10 and r['offset'] is None and 'old' not in r
                            continue
                        margin=z[pre+m+'_margin'];bp=z[pre+m+'_base_prediction'];npred=z[pre+m+'_novel_prediction']
                        def old_acc(d):
                            new=(d>margin)|((d==margin)&(npred<bp));return float(w[correct&~new].sum())
                        assert old_acc(r['offset'])>=r['target']-1e-10
                        assert abs(old_acc(r['offset'])-r['selection_base_accuracy'])<1e-12
                        assert old_acc(np.nextafter(float(r['offset']),np.inf))<r['target']+1e-10
                    d=r['offset'];margin=z[m+'_margin'];bp=z[m+'_base_prediction'];npred=z[m+'_novel_prediction']
                    new=(d>margin)|((d==margin)&(npred<bp));pred=np.where(new,npred,bp)
                    old=float(wt[(pred==y)&np.isin(y,base)].sum());nov=float(wt[(pred==y)&np.isin(y,novel)].sum())
                    assert abs(old-r['old'])<1e-12 and abs(nov-r['new'])<1e-12
                    assert abs((len(base)*old+len(novel)*nov)/(len(base)+len(novel))-r['overall'])<1e-12
                    if evaluation!='fixed_gate':
                        ref=complete['reference_closed_test'];assert abs(r['test_retention_ratio']-old/ref)<1e-12
                        assert r['test_target_met']==(old>=r['ratio']*ref-1e-10)
                    checked+=1
                if encoder=='l14' and p<3:
                    paths=[ROOT/f'experiments/kbs_strengthening/{dataset}/p{p}'/prefix.with_suffix('.json').name,
                           ROOT/f'experiments/kbs_depth/{dataset}/tip/p{p}'/prefix.with_suffix('.json').name]
                    original=[r for path in paths for r in json.loads(path.read_text())['records']]
                    for r in rows:
                        if r['method'] in ['raw','fixed_temperature']:continue
                        old=next(a for a in original if (a['method'],a['evaluation'],a.get('ratio'))==(r['method'],r['evaluation'],r.get('ratio')))
                        assert r.get('feasible')==old.get('feasible');original_matches+=1
                        for key in ['old','new','overall','harmonic']:
                            if key in r:max_old_delta=max(max_old_delta,abs(r[key]-old[key]))
    assert count==8800 and max_old_delta<1e-7,(encoder,dataset,max_old_delta)
    assert json.loads((root/'complete.json').read_text())['complete']
    return {'partitions':10,'records':count,'feasible_metrics_reconstructed':checked,'original_rows_matched':original_matches,
            'original_max_delta':max_old_delta,'source_and_protocol_hashes_valid':True}


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--encoders',nargs='+',default=['l14','b32']);ap.add_argument('--datasets',nargs='+',default=['cifar','pets','imagenetr']);args=ap.parse_args()
    result={e:{d:verify(e,d) for d in args.datasets} for e in args.encoders}
    (ROOT/'experiments/kbs_revision/verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
