"""Boundary witnesses for the new equal-budget fusion search."""
import unittest
import numpy as np
import torch
import kbs_full_fusion_retuning as f
import fcs_expansion as core
import fcs_retention_value as rv

class FusionTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(123);self.E=torch.nn.functional.normalize(torch.randn(40,8),dim=1);self.T=torch.nn.functional.normalize(torch.randn(6,8),dim=1)
  self.P=torch.nn.functional.normalize(torch.randn(6,8),dim=1);self.W=torch.randn(9,3);self.m=[(.1,.7),(.2,.9),(.3,.4)]
 def test_query_retained_matches_original(self):
  c=f.components(self.E,self.T,self.P[:3],self.P[3:],self.W,self.m,'query')
  expected=rv.score(rv.arrays(self.E,self.T,self.P[:3],self.P[3:],self.W),'retained',(.5,2,0))
  torch.testing.assert_close(f.scores(c,'retained',.5,2),expected)
 def test_tied_dense_matches_original(self):
  c=f.components(self.E,self.T,self.P[:3],self.P[3:],self.W,self.m,'query')
  expected=rv.score(rv.arrays(self.E,self.T,self.P[:3],self.P[3:],self.W),'anchored_tp',(0,2,0))
  torch.testing.assert_close(f.scores(c,'dense',2,2),expected)
 def test_uncentered_explicit_equation(self):
  c=f.components(self.E,self.T,self.P[:3],self.P[3:],self.W,self.m,'none')
  qt=self.E@self.T.T;ql=core.linear(self.E,self.W);qp=self.E@self.P.T
  expected=qt/qt[:,:3].std(1,correction=0)[:,None]
  expected[:,:3]+=.5*ql/ql.std(1,correction=0)[:,None]
  expected[:,3:]+=2*qp[:,3:]/qp[:,:3].std(1,correction=0)[:,None]
  torch.testing.assert_close(f.scores(c,'retained',.5,2),expected)
 def test_search_ties_and_extremes(self):
  ids=np.array([0,1,2,3]);y=np.tile(ids,10);s=torch.zeros(40,4);c=rv.compact(s,ids,y,ids[:2],ids[2:]);offsets=np.array([-np.inf,0,np.nextafter(0,np.inf),np.inf])
  np.testing.assert_allclose(f.tuning_scores(c,offsets),[rv.evaluate(c,float(x))['harmonic'] for x in offsets],atol=1e-12)
if __name__=='__main__':unittest.main()
