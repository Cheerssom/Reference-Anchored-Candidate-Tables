"""Check paired-error identities from saved predictions; no fitting or selection."""
from pathlib import Path
import hashlib
import json
import numpy as np

H=Path(__file__).resolve().parent
ROOT=H.parents[1]

def variance_formula(ar, ac, g, r):
    return ar*(1-r)**2+(ar-ac)*(2*r-1)+2*r*g-(ac-r*ar)**2

def check_joint_probabilities():
    # Enumerate all four-event probability vectors with denominator 20.
    checked=0;error=0.
    for a in range(21):
        for b in range(21-a):
            for c in range(21-a-b):
                probs=np.array([a,b,c,20-a-b-c],float)/20
                ref=np.array([0.,0.,1.,1.]);correct=np.array([0.,1.,0.,1.])
                ar=probs@ref;ac=probs@correct;g=probs[1]
                for r in [.1,.5,.8,.95,.99]:
                    z=correct-r*ref;direct=probs@(z-z@probs)**2
                    error=max(error,abs(direct-variance_formula(ar,ac,g,r)))
                    loss=ref*(1-correct)+(1-r)*(1-ref)
                    assert abs((1-r)-probs@loss-(ac-r*ar-g))<1e-12
                    if ac<=ar+1e-12:
                        zero_g=np.array([1-ar,0.,ar-ac,ac])
                        v0=zero_g@(z-z@zero_g)**2
                        assert abs(direct-v0-2*r*g)<1e-12
                    checked+=1
    assert error<1e-12
    return checked,error

def check_archives():
    records=0;classes=0;max_error=0.;hashes={}
    for filename in sorted((ROOT/'experiments/kbs_mechanism_followup').glob('*/*/p*/o[01].json')):
        data=json.loads(filename.read_text())
        source=next(k for k in data['source_hashes'] if 'kbs_independent_risk' in k and k.endswith('.npz'))
        source_path=ROOT/Path(source.replace('\\','/'))
        sha=hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert sha==data['source_hashes'][source]
        hashes[str(filename.relative_to(ROOT))]=hashlib.sha256(filename.read_bytes()).hexdigest()
        with np.load(source_path) as z:
            labels=z['test_labels'];base=z['base_classes'];mask=np.isin(labels,base)
            y=labels[mask];ref=z['test_reference_correct'][mask].astype(float)
            weights=np.zeros(len(y))
            for cl in base:weights[y==cl]=1/(len(base)*np.sum(y==cl))
            for row in data['rows']:
                name=row['method'];r=.95
                if row['fallback']:
                    correct=ref.copy()
                else:
                    bp=z[f'test_{name}_base_prediction'][mask]
                    npred=z[f'test_{name}_novel_prediction'][mask]
                    margin=z[f'test_{name}_margin'][mask];d=row['offset']
                    admit=(d>margin)|((d==margin)&(npred<bp))
                    correct=((bp==y)&~admit).astype(float)
                ar=float(weights@ref);ac=float(weights@correct)
                g=float(weights@((ref==0)&(correct==1)))
                destruction=float(weights@((ref==1)&(correct==0)))
                assert abs(ac-row['base'])<1e-10 and abs(g-row['recovered'])<1e-10
                assert abs(destruction-row['destroyed_to_base']-row['destroyed_to_novel'])<1e-10
                score=correct-r*ref;mean=float(weights@score)
                v=float(weights@((score-mean)**2))
                max_error=max(max_error,abs(v-variance_formula(ar,ac,g,r)))
                loss=ref*(1-correct)+(1-r)*(1-ref)
                assert abs((1-r)-weights@loss-(ac-r*ar-g))<1e-10
                if name=='retained':assert g==0 and row['destroyed_to_base']==0
                for cl in base:
                    ix=y==cl
                    rc=ref[ix];cc=correct[ix];gc=float(np.mean((rc==0)&(cc==1)))
                    max_error=max(max_error,abs(np.var(cc-r*rc)-variance_formula(rc.mean(),cc.mean(),gc,r)))
                    classes+=1
                records+=1
    assert records==480 and max_error<1e-10
    return records,classes,max_error,hashes

if __name__=='__main__':
    count,e1=check_joint_probabilities()
    records,classes,e2,hashes=check_archives()
    report=dict(complete=True,probability_grid_checks=count,prediction_policy_checks=records,
                conditional_class_checks=classes,max_absolute_error=max(e1,e2),
                interpretation='Exact identities and conditional coupling comparison; no certification-power ordering claimed.',
                scope='Original frozen source models, final arrival, both orders; not the retuned fusion study.',
                source_hashes=hashes)
    (H/'error_structure_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'},indent=2))
