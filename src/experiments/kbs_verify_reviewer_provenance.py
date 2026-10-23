"""Post-run input binding, image-role separation and cached HyCal score witnesses.

This additional record binds current inputs; it is not a retrospective claim
that the original runner manifest bound every input at experiment start.
"""
from pathlib import Path
import gc,json,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import kbs_reviewer_revision as run
import kbs_reviewer_readouts as ro
import kbs_calibration_pool as pool
from kbs_runtime import atomic_json,digest_file
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/kbs_reviewer_revision'

def main():
    torch.set_num_threads(2)
    from lt_dataset import _load_split
    tr,yt,_=_load_split('train');te,ye,_=_load_split('test')
    sm=json.loads((OUT/'shift_features/manifest.json').read_text())
    pm=json.loads((ROOT/'experiments/kbs_calibration_budget/pools/cifar.json').read_text())
    ch=[pool.pixel_hash(tr[i]) for i in sm['calibration_raw_ids']]
    th=[pool.pixel_hash(te[i]) for i in sm['test_raw_ids']]
    assert ch==[r['sha256'] for r in pm['rows']]
    assert len(set(ch))==4000 and len(set(th))==2000
    assert not set(ch)&set(th)
    assert not set(ch)&set(pm['protected_hashes'])
    assert np.array_equal(yt[sm['calibration_raw_ids']],[r['class_id'] for r in pm['rows']])
    assert np.all(np.bincount(ye[sm['test_raw_ids']],minlength=100)==20)
    del tr,te;gc.collect()
    bindings={};witnesses=[]
    for enc in ['l14','b32']:
        for ds in ['cifar','pets','imagenetr']:
            d,cache=run.load_data(ds,enc)
            paths=[cache,ROOT/f'experiments/kbs_independent_risk/splits/{ds}.json',
                ROOT/f'experiments/kbs_revision/{enc}/{ds}/partitions.json',
                ROOT/f'experiments/kbs_calibration_budget/pools/{ds}.json',
                ROOT/f'experiments/kbs_calibration_budget/features/{enc}/{ds}/extra.pt',
                ROOT/'experiments/kbs_reviewer_revision_protocol.md']
            if ds=='cifar':paths += [OUT/f'shift_features/{enc}/{c}/features.pt' for c in ['noise08','blur125']]
            for p in paths:bindings[p.relative_to(ROOT).as_posix()]=digest_file(p)
            q=d['E_tune'][:8];T=d['T'];classes=[0,len(T)-1]
            for reg in ro.HYCAL_LAMBDAS:
                for role in ['base','novel']:
                    p=OUT/f'readouts/{enc}/{ds}/hycal_{role}_{reg:g}.pt'
                    v=torch.load(p,map_location='cpu',weights_only=False)
                    err=0.
                    for c in classes:
                        x=d['E_fit' if role=='base' else 'E_support'][d['fit_y' if role=='base' else 'support_y']==c]
                        state=ro.hycal_statistics(x,T[c],reg)
                        co,md=ro.hycal_distances(q,T[c],state)
                        assert len(x)==v['counts'][c]
                        assert torch.allclose(co,v['cosine'][:8,c],atol=2e-5,rtol=2e-5)
                        assert torch.allclose(-md,v['negative_distance'][:8,c],atol=2e-2,rtol=2e-5)
                        err=max(err,float((md+v['negative_distance'][:8,c]).abs().max()))
                    witnesses.append(dict(encoder=enc,dataset=ds,role=role,regularization=reg,queries=8,classes=classes,max_distance_error=err))
                    bindings[p.relative_to(ROOT).as_posix()]=digest_file(p)
                    del v
            del d;gc.collect()
            print('PROVENANCE VERIFIED',enc,ds,flush=True)
    for p in (ROOT/'src/experiments').glob('kbs_*reviewer*.py'):bindings[p.relative_to(ROOT).as_posix()]=digest_file(p)
    atomic_json(dict(complete=True,record_type='post-run current-input binding with numerical witnesses',
        cifar_calibration_unique=4000,cifar_test_unique=2000,pixel_overlap=0,protected_overlap=0,
        input_sha256=bindings,hycal_cache_witnesses=witnesses),OUT/'provenance_verification.json')

if __name__=='__main__':main()
