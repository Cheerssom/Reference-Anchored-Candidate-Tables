"""Behavior checks for matched macro-retention and paired group competition."""
import unittest
import numpy as np
import torch
from fcs_retention_value import compact,evaluate,threshold

class RetentionTests(unittest.TestCase):
    def stats(self):
        return {'margin':np.array([4.,3.,2.,1.,-1.,1.5,2.5,5.]),
          'base_prediction':np.array([0]*4+[0]*4),
          'novel_prediction':np.array([2]*8),
          'base_correct':np.array([True]*4+[False]*4),
          'novel_correct':np.array([False]*4+[True]*4),
          'weights':np.full(8,.25),'nb':2,'nn':2}
    def test_threshold_maximizes_admission_under_common_constraint(self):
        c=self.stats();d,ceiling=threshold(c,.75)
        self.assertEqual(ceiling,1.)
        self.assertEqual(d,2.)  # Canonical base label wins this boundary tie.
        self.assertAlmostEqual(evaluate(c,d)['old'],.75)
        self.assertAlmostEqual(evaluate(c,d)['new'],.5)
        self.assertLess(evaluate(c,2.1)['old'],.75)
    def test_infeasible_is_missing_not_zero_novel_accuracy(self):
        d,ceiling=threshold(self.stats(),1.01)
        self.assertIsNone(d);self.assertEqual(ceiling,1.)
    def test_tied_base_margins_keep_constraint_feasible(self):
        c=self.stats();c['margin'][1]=2.
        d,_=threshold(c,.5)
        self.assertGreaterEqual(evaluate(c,d)['old'],.5)
    def test_threshold_never_uses_novel_query_correctness(self):
        c=self.stats();first,_=threshold(c,.75)
        c['novel_correct'][:]=False
        second,_=threshold(c,.75)
        self.assertEqual(first,second)
    def test_macro_weights_and_global_label_tie_break(self):
        ids=np.array([3,5,0,1]);y=np.array([3,3,3,5,0,1])
        scores=torch.tensor([[2.,0.,2.,0.],[2.,0.,0.,0.],[2.,0.,0.,0.],
                             [0.,2.,0.,0.],[0.,0.,2.,0.],[0.,0.,0.,2.]])
        c=compact(scores,ids,y,ids[:2],ids[2:])
        self.assertAlmostEqual(evaluate(c,0.)['old'],5/6)
        self.assertAlmostEqual(evaluate(c,0.)['new'],1.)
        self.assertAlmostEqual(c['weights'][:3].sum(),.5)

if __name__=='__main__':unittest.main()
