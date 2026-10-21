"""Behavior checks for the scientific protocol; no model downloads or training."""
import unittest
import numpy as np
import torch
from fcs_expansion import partition,branches,score,metrics,support_rows
from fcs_review_experiments import predict,choose

class ProtocolTests(unittest.TestCase):
    def test_ties_use_global_class_id_across_arrival_orders(self):
        scores=torch.tensor([[1.,1.,0.],[0.,1.,1.]])
        ids=np.array([8,2,5]);perm=np.array([2,0,1])
        np.testing.assert_array_equal(predict(scores,ids),np.array([2,2]))
        np.testing.assert_array_equal(predict(scores[:,perm],ids[perm]),predict(scores,ids))

    def test_retention_selects_feasible_point_and_flags_fallback(self):
        ids=np.array([0,1]);y=np.array([0,0,1,1])
        good_old=torch.tensor([[2.,0.],[2.,0.],[0.,2.],[2.,0.]])
        good_new=torch.tensor([[2.,0.],[0.,2.],[0.,2.],[0.,2.]])
        selected,feasible=choose([('old',good_old),('new',good_new)],ids,y,1,.9)
        self.assertTrue(feasible);self.assertEqual(selected[0],'old')
        selected,feasible=choose([('new',good_new)],ids,y,1,.9)
        self.assertFalse(feasible);self.assertEqual(selected[0],'new')

    def test_membership_changes_with_seed_and_is_exhaustive(self):
        for sizes,nb,nn in [([35,35,30],40,20),([0,24,13],15,7)]:
            buckets=sum(([name]*n for name,n in zip(('head','medium','tail'),sizes)),[])
            results=[partition(buckets,s,nb,nn) for s in (0,1,2)]
            self.assertEqual(len({tuple(b) for b,n,e in results}),3)
            for base,new,neg in results:
                self.assertEqual((len(base),len(new),len(neg)),(nb,len(buckets)-nb-nn,nn))
                self.assertEqual(len(set(base)|set(new)|set(neg)),len(buckets))
                self.assertFalse(set(base)&set(new))

    def test_existing_rows_survive_appends_including_earlier_novel_rows(self):
        g=torch.Generator().manual_seed(100)
        def random(n,d):return torch.nn.functional.normalize(torch.randn(n,d,generator=g),dim=1)
        E,T,Pb,Pn=random(27,16),random(15,16),random(7,16),random(8,16)
        W=torch.randn(17,7,generator=g)
        gate=(.5,1.,.5)
        first=score(branches(E,T[:9],Pb,Pn[:2],W,'reference'),gate)
        expanded=score(branches(E,T,Pb,Pn,W,'reference'),gate)
        torch.testing.assert_close(first,expanded[:,:9],rtol=1e-5,atol=1e-6)
        # Competing new rows may win, but cannot create an old-to-old change.
        before=first.argmax(1);after=expanded.argmax(1)
        self.assertTrue(torch.all((after==before)|(after>=9)))

    def test_group_metrics_ignore_unused_negative_classes(self):
        y=np.array([0,0,1,1,2,2,3,3,9])
        p=np.array([0,0,1,0,2,0,3,0,9])
        m=metrics(p,y,np.array([0,1]),np.array([2,3]))
        self.assertAlmostEqual(m['old'],.75)
        self.assertAlmostEqual(m['new'],.5)
        self.assertAlmostEqual(m['overall'],.625)
        self.assertAlmostEqual(m['harmonic'],.6)

    def test_support_budget_and_nested_replication(self):
        y=np.repeat(np.arange(4),5)
        ids=np.array([1,3])
        for k in (1,2,5):
            rows=support_rows(y,ids,k,7)
            self.assertEqual(len(rows),2*k)
            self.assertTrue(set(y[rows])==set(ids))
        # Five-shot support subsets are not independent across seeds.
        self.assertEqual(set(support_rows(y,ids,5,7)),set(support_rows(y,ids,5,8)))

if __name__=='__main__':unittest.main()
