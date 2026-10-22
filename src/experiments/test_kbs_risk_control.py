"""Checks scientific edge cases of the new protection rule."""
import unittest
import numpy as np
import kbs_risk_control as rc


class RiskTests(unittest.TestCase):
    def test_protection_implies_ratio_even_with_recoveries(self):
        # Exhaustive paired outcome counts, including comparator recoveries.
        for both in range(5):
            for damage in range(5):
                for recover in range(5):
                    for neither in range(1,5):
                        ref = np.array([1]*both+[1]*damage+[0]*recover+[0]*neither, bool)
                        pred = np.array([1]*both+[0]*damage+[1]*recover+[0]*neither, bool)
                        for r in [.8,.95]:
                            loss = rc.protection_loss(ref,pred,r).mean()
                            if loss <= 1-r+1e-12:
                                self.assertGreaterEqual(pred.mean()+1e-12,r*ref.mean())

    def test_unequal_strata_have_macro_weights(self):
        y=np.array([0,0,1,1,1,1]);w=rc.weights(y,[0,1])
        self.assertAlmostEqual(w[y==0].sum(),.5)
        self.assertAlmostEqual(w[y==1].sum(),.5)
        # More copies within a stratum cannot change its macro contribution.
        self.assertAlmostEqual(w @ (y==0),.5)

    def test_stop_on_first_failed_test_and_power(self):
        n=5000;ref=np.ones(n,bool);w=np.ones(n)/n
        c=dict(margin=np.linspace(0,1,n),base_correct=ref,
               base_prediction=np.zeros(n,int),novel_prediction=np.ones(n,int))
        out=rc.certify(c,ref,w,[-1.,.01,.9,.99])
        self.assertEqual(out['offset'],.01)
        self.assertEqual(len(out['trace']),3)
        self.assertFalse(out['trace'][-1]['accepted'])
        small={k:v[:20] for k,v in c.items()}
        out=rc.certify(small,ref[:20],np.ones(20)/20,[-1.,.01])
        self.assertTrue(out['fallback'])

    def test_weighted_chernoff_controls_small_exact_null(self):
        # Exact enumeration under heterogeneous independent Bernoulli strata.
        import itertools
        w=np.array([.5,.25,.25]);means=np.array([.2,.8,.8]);mu=w@means
        for level in [.01,.05,.2]:
            rejection=0.
            for x in itertools.product([0,1],repeat=3):
                x=np.array(x);prob=np.prod(np.where(x,means,1-means))
                p,_=rc.pvalue(x,w,mu)
                rejection+=prob*(p<=level)
            self.assertLessEqual(rejection,level+1e-12)


if __name__=='__main__':unittest.main()
