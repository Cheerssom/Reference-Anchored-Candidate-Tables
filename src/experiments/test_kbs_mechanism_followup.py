"""Scientific boundary checks, not recognition experiments."""
import unittest
import numpy as np
import torch
import torch.nn.functional as F
import fcs_expansion as core
import fcs_retention_value as rv
from kbs_mechanism_followup import envelope,capacity,error_terms,bias_arrays

class MechanismTests(unittest.TestCase):
    def test_discrete_gap_is_not_interpolated(self):
        c=dict(margin=np.zeros(4),base_prediction=np.array([0,1,0,1]),novel_prediction=np.array([2,3,2,3]),
            base_correct=np.array([1,1,0,0],bool),novel_correct=np.array([0,0,1,1],bool),
            base_mask=np.array([1,1,0,0],bool),novel_mask=np.array([0,0,1,1],bool),weights=np.full(4,.5),nb=2,nn=2)
        curve=envelope(c);matched=capacity(curve,c,.5)
        self.assertEqual(matched['base'],1.);self.assertEqual(matched['novel'],0.)
        self.assertEqual(rv.evaluate(c,0.)['old'],1.)
        self.assertEqual(rv.evaluate(c,np.nextafter(0.,np.inf))['new'],1.)

    def test_replacement_destruction_has_two_destinations(self):
        c=dict(margin=np.array([0.,2.,2.,2.,0.,0.]),base_prediction=np.array([0,1,1,0,0,1]),
            novel_prediction=np.array([2,3,2,3,2,3]),base_correct=np.array([1,0,1,0,0,0],bool),
            novel_correct=np.array([0,0,0,0,1,1],bool),base_mask=np.array([1,1,1,1,0,0],bool),
            novel_mask=np.array([0,0,0,0,1,1],bool),weights=np.array([.25,.25,.25,.25,.5,.5]),nb=2,nn=2)
        ref=np.array([1,1,0,0,0,0],bool);v=error_terms(c,ref,1.,.5)
        self.assertEqual(v['destroyed_to_novel'],.25);self.assertEqual(v['destroyed_to_base'],.25)
        self.assertEqual(v['recovered'],.25);self.assertEqual(v['base'],.25)

    def test_unregularized_intercept_fixes_learned_row_mean(self):
        torch.manual_seed(27102026);y=np.repeat(np.arange(4),[37,23,12,8])
        E=F.normalize(torch.randn(len(y),9),dim=1);W=core.fit_ridge(E,y,np.arange(4),.2)
        self.assertTrue(torch.allclose(W[:-1].sum(1),torch.zeros(9),atol=1e-6))
        self.assertAlmostEqual(float(W[-1].sum()),1.,places=6)
        q=F.normalize(torch.randn(17,9),dim=1)
        self.assertTrue(torch.allclose(core.linear(q,W).mean(1),torch.full((17,),.25),atol=1e-6))

    def test_effective_bias_preserves_group_order(self):
        torch.manual_seed(17);K=7;D=9;nb=4
        E=F.normalize(torch.randn(31,D),dim=1);T=F.normalize(torch.randn(K,D),dim=1)
        P=F.normalize(torch.randn(K,D),dim=1);W=torch.randn(D+1,nb)
        delta,raw,sc,w=bias_arrays(E,T,P[:nb],P[nb:],W,(.5,1.,0.))
        m=(sc[:,:nb].max(1).values.double()-sc[:,nb:].max(1).values.double()).numpy()
        self.assertLess(np.max(abs(m-(raw-delta))),1e-5)

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
