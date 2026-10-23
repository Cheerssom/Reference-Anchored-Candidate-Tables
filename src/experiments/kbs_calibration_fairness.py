"""Recalibrate immutable cap40 archives with recovery-aware paired tests."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fcs_retention_value as rv
import kbs_calibration_budget as budget
import kbs_risk_control as risk
import kbs_paired_accuracy as paired
from kbs_runtime import atomic_json,check_pause,digest_file

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/kbs_reviewer_revision/fairness'


def evaluate_policy(c,reference,w,candidates,tc,test_reference,closed,policy):
    detail=(risk if policy=='protection' else paired).certify(c,reference,w,candidates)
    offset=detail['offset']
    result=dict(old=closed,new=0.,overall=closed*tc['nb']/(tc['nb']+tc['nn']),harmonic=0.) if offset is None else rv.evaluate(tc,offset)
    mask=tc['base_mask'];tw=tc['weights'][mask]
    corr=(test_reference if offset is None else tc['base_correct']&~risk.choose_new(tc,offset))[mask]
    ref=test_reference[mask]
    result.update(destroyed_test=float(tw@(ref&~corr)),recovered_test=float(tw@(~ref&corr)))
    return dict(policy=policy,**detail,**result,test_target_met=result['old']>=.95*closed-1e-10,
                target_gap_pp=100*(result['old']-.95*closed))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--encoder',choices=['l14','b32']);ap.add_argument('--dataset',choices=['cifar','pets','imagenetr']);args=ap.parse_args()
    started=time.perf_counter()
    for e in ([args.encoder] if args.encoder else ['l14','b32']):
        for d in ([args.dataset] if args.dataset else ['cifar','pets','imagenetr']):
            for p in range(10):
                for order in [0,1]:
                    for t in range(1,6):
                        check_pause();out=OUT/e/d/f'p{p}/o{order}_t{t}.json'
                        calpath=ROOT/f'experiments/kbs_calibration_budget/{e}/{d}/p{p}/o{order}_t{t}.npz'
                        testpath=ROOT/f'experiments/kbs_independent_risk/{e}/{d}/p{p}/o{order}_t{t}.npz'
                        config=dict(calibration_sha256=digest_file(calpath),test_sha256=digest_file(testpath),
                                    code_sha256=digest_file(Path(paired.__file__)),runner_sha256=digest_file(Path(__file__)))
                        if out.exists():assert json.loads(out.read_text())['inputs']==config;continue
                        with np.load(calpath) as z:cal={k:z[k] for k in z.files}
                        with np.load(testpath) as z:test={k:z[k] for k in z.files}
                        base,new=cal['base_classes'],cal['novel_classes'];y=cal['labels'];yt=test['test_labels']
                        w=risk.weights(y,base);reference=cal['reference_correct'];tref=test['test_reference_correct']
                        closed=float(np.mean([tref[yt==c].mean() for c in base]));rows=[]
                        for m in budget.METHODS:
                            c=budget.compact(cal,m,y,base,new);tc=budget.compact(test,m,yt,base,new,'test_')
                            for policy in ['protection','paired_accuracy']:
                                detail=evaluate_policy(c,reference,w,test[m+'_candidates'],tc,tref,closed,policy)
                                rows.append(dict(encoder=e,dataset=d,part=p,order=order,session=t,method=m,
                                                 reference_closed_test=closed,**detail))
                        atomic_json(dict(inputs=config,records=rows),out)
                print('FAIRNESS COMPLETE',e,d,p,flush=True)
    atomic_json(dict(complete=True,seconds=time.perf_counter()-started),OUT/'complete.json')


if __name__=='__main__':main()
