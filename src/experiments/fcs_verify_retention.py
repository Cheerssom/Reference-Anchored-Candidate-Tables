"""Cache-free verification of retention/value records from archived margins."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def verify(folder):
    folder=Path(folder);d=json.loads((folder/'results.json').read_text())
    assert hashlib.sha256((folder/'source_snapshot.py').read_bytes()).hexdigest()==d['source_sha256']
    assert len(d['records'])==1320
    targets={};checked=0;maxovershoot=0.
    for part in range(3):
        for order in (0,1):
            for session in range(1,6):
                with np.load(folder/f'margins_p{part}_o{order}_t{session}.npz') as z:
                    nb=len(z['base_classes']);nn=len(z['novel_classes'])
                    expected_weights=np.zeros(len(z['labels']),dtype=float)
                    for group in (z['base_classes'],z['novel_classes']):
                        for c in group:expected_weights[z['labels']==c]=1./(len(group)*np.sum(z['labels']==c))
                    for method in ['retained','removed_fixed','no_learned_tuned','anchored_tp']:
                        assert np.allclose(expected_weights,z[method+'_weights'])
                        assert np.array_equal(z[method+'_base_correct'],np.isin(z['labels'],z['base_classes'])&(z[method+'_base_prediction']==z['labels']))
                        assert np.array_equal(z[method+'_novel_correct'],np.isin(z['labels'],z['novel_classes'])&(z[method+'_novel_prediction']==z['labels']))
                    rows=[r for r in d['records'] if (r['part'],r['order'],r['session'])==(part,order,session)]
                    for r in rows:
                        if r['evaluation']!='fixed_gate':
                            key=(part,order,session,r['evaluation'],r['ratio'])
                            if key in targets:assert abs(targets[key]-r['target'])<1e-12
                            targets[key]=r['target']
                            if not r['feasible']:
                                assert r['offset'] is None and 'new' not in r
                                assert r['selection_base_ceiling']<r['target']+1e-10
                                continue
                            assert r['selection_base_accuracy']>=r['target']-1e-10
                            if r['evaluation']=='matched_test':
                                assert r['old']>=r['target']-1e-10
                                maxovershoot=max(maxovershoot,r['old']-r['target'])
                        m=r['method'];margin=z[m+'_margin'];offset=r['offset']
                        choose_new=(offset>margin)|((offset==margin)&(z[m+'_novel_prediction']<z[m+'_base_prediction']))
                        weights=z[m+'_weights']
                        base=float(weights[z[m+'_base_correct']&~choose_new].sum())
                        new=float(weights[z[m+'_novel_correct']&choose_new].sum())
                        assert abs(base-r['old'])<1e-10 and abs(new-r['new'])<1e-10
                        assert abs((nb*base+nn*new)/(nb+nn)-r['overall'])<1e-10
                        checked+=1
    return {'records':len(d['records']),'margin_recomputed_points':checked,
            'shared_target_groups':len(targets),'source_hash_valid':True,
            'max_matched_base_overshoot_pp':100*maxovershoot}

if __name__=='__main__':
    print(json.dumps({name:verify(ROOT/'experiments'/name/'retention_value')
                      for name in ['fcs','fcs_pets']},indent=2))
