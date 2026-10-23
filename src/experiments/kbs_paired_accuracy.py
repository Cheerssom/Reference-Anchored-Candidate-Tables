"""Fixed-time paired accuracy certification for independent macro strata.

E-values test the exact population requirement A_candidate >= ratio*A_reference.
The fixed lambda mixture credits recovery of reference mistakes. It is not an
optional-stopping e-process over a calibration stream.
"""
import math
import numpy as np
from kbs_risk_control import choose_new

LAMBDAS=np.array([.01,.025,.05,.1,.2,.3,.4,.5,.65,.8,.95,1.0])


def log_evalue(reference,correct,w,ratio=.95):
    reference=np.asarray(reference,bool);correct=np.asarray(correct,bool);w=np.asarray(w,float)
    assert reference.shape==correct.shape==w.shape
    assert (w>0).all() and np.isclose(w.sum(),1.) and 0<ratio<1
    assert (LAMBDAS>=0).all() and (LAMBDAS<1/ratio).all()
    # Group equal macro weights, making each candidate test inexpensive.
    levels,groups=np.unique(w/w.max(),return_inverse=True)
    h=np.bincount(groups[reference&~correct],minlength=len(levels))
    g=np.bincount(groups[~reference&correct],minlength=len(levels))
    k=np.bincount(groups[reference&correct],minlength=len(levels))
    a=LAMBDAS[:,None]*levels[None]
    logs=np.log1p(-ratio*a)@h+np.log1p(a)@g+np.log1p((1-ratio)*a)@k
    return float(np.logaddexp.reduce(logs)-math.log(len(LAMBDAS)))


def pvalue(reference,correct,w,ratio=.95):
    le=log_evalue(reference,correct,w,ratio)
    p=float(math.exp(-max(0.,le)))
    return p,dict(log_evalue=le,accuracy_surplus=float(w@(correct.astype(float)-ratio*reference)),
                  destroyed=float(w@(reference&~correct)),recovered=float(w@(~reference&correct)))


def certify(c,reference,w,candidates,ratio=.95,delta=.05,sessions=5):
    candidates=np.asarray(candidates,float)
    assert np.isfinite(candidates).all() and (np.diff(candidates)>0).all()
    offset=None;trace=[]
    for d in candidates:
        correct=c['base_correct']&~choose_new(c,d)
        p,detail=pvalue(reference,correct,w,ratio)
        passed=p<=delta/sessions
        trace.append(dict(offset=float(d),pvalue=p,accepted=passed,**detail))
        if not passed:break
        offset=float(d)
    return dict(offset=offset,fallback=offset is None,certified_expansion=offset is not None,
                delta=delta,per_session_delta=delta/sessions,trace=trace)
