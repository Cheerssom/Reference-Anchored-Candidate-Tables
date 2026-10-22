"""Stratified macro protection risk and fixed-sequence calibration."""
import math
import numpy as np


def weights(labels, classes):
    w = np.zeros(len(labels), dtype=float)
    for c in classes:
        ix = labels == c
        if not ix.any():
            raise ValueError(f'Missing calibration class {c}')
        w[ix] = 1. / (len(classes)*ix.sum())
    assert np.isclose(w.sum(), 1.)
    return w


def bernoulli_kl(q, p):
    if not 0 <= q <= 1 or not 0 < p < 1:
        raise ValueError('Invalid Bernoulli parameters')
    a = 0. if q == 0 else q*math.log(q/p)
    b = 0. if q == 1 else (1-q)*math.log((1-q)/(1-p))
    return max(0., a+b)


def protection_loss(reference_correct, expanded_correct, ratio=.95):
    ref = np.asarray(reference_correct, dtype=bool)
    correct = np.asarray(expanded_correct, dtype=bool)
    return (ref & ~correct).astype(float)+(1-ratio)*(~ref)


def pvalue(loss, w, alpha=.05):
    loss, w = np.asarray(loss, float), np.asarray(w, float)
    assert loss.shape == w.shape and (loss >= 0).all() and (loss <= 1).all()
    assert (w > 0).all() and np.isclose(w.sum(), 1.)
    q = float(w @ loss)
    return (math.exp(-bernoulli_kl(q, alpha)/float(w.max())) if q < alpha else 1.), q


def choose_new(c, offset):
    return ((offset > c['margin']) |
            ((offset == c['margin']) & (c['novel_prediction'] < c['base_prediction'])))


def certify(c, reference_correct, w, candidates, delta=.05, sessions=5, ratio=.95):
    candidates = np.asarray(candidates, float)
    assert np.isfinite(candidates).all() and (np.diff(candidates) > 0).all()
    accepted = None
    trace = []
    for offset in candidates:
        expanded = c['base_correct'] & ~choose_new(c, offset)
        p, q = pvalue(protection_loss(reference_correct, expanded, ratio), w, 1-ratio)
        passed = p <= delta/sessions
        trace.append(dict(offset=float(offset), pvalue=p, loss=q, accepted=passed))
        if not passed:
            break
        accepted = float(offset)
    return dict(offset=accepted, certified_expansion=accepted is not None,
                fallback=accepted is None, delta=delta, per_session_delta=delta/sessions,
                calibration_max_weight=float(w.max()), trace=trace)
