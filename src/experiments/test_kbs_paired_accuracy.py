"""Scientific checks: weighted non-iid nulls, recovery credit and FST."""
import itertools,unittest
import numpy as np
from kbs_paired_accuracy import LAMBDAS,log_evalue,pvalue,certify


class TestPairedAccuracy(unittest.TestCase):
    def test_nonidentical_weighted_null(self):
        # Enumerate all4^3 real binary correctness outcomes under heterogeneous laws.
        outcomes=np.array([[0,0],[0,1],[1,0],[1,1]])
        probs=np.array([[.1,.1,.7,.1],[.3,.15,.4,.15],[.1,.05,.75,.1]])
        w=np.array([.2,.3,.5]);mu=(probs@(outcomes[:,1]-.95*outcomes[:,0]))@w
        self.assertLess(mu,0);expectation=0.
        for ids in itertools.product(range(4),repeat=3):
            x=outcomes[list(ids)];prob=np.prod(probs[np.arange(3),ids])
            expectation+=prob*np.exp(log_evalue(x[:,0],x[:,1],w))
        self.assertLessEqual(expectation,1+1e-12)

    def test_recovers_reference_errors(self):
        r=np.r_[np.ones(4000),np.zeros(1000)].astype(bool)
        c=np.r_[np.ones(3600),np.zeros(400),np.ones(1000)].astype(bool)
        p,detail=pvalue(r,c,np.full(5000,1/5000))
        self.assertLess(p,.01);self.assertAlmostEqual(detail['destroyed'],.08)
        self.assertAlmostEqual(detail['recovered'],.2)
        self.assertGreater(detail['destroyed'],.05*r.mean())

    def test_composite_boundary_with_positive_individual_mean(self):
        q=np.array([7/10,1/10,35/78]);w=np.array([.5,.25,.25])
        self.assertAlmostEqual(float(w@(1.95*q-.95)),0.)
        self.assertGreater(1.95*q[0]-.95,0.)
        outcomes=[];expectation=0.
        for recovered in itertools.product([False,True],repeat=3):
            c=np.array(recovered,bool);r=~c;prob=float(np.prod(np.where(c,q,1-q)))
            loge=log_evalue(r,c,w);expectation+=prob*np.exp(loge)
            outcomes.append((pvalue(r,c,w)[0],prob))
        self.assertAlmostEqual(expectation,.9550611942312288)
        for level in [.01,.05,.1,.25,.5,.75,.9]:
            self.assertLessEqual(sum(prob for p,prob in outcomes if p<=level),level+1e-12)

    def test_direct_product_matches_grouped(self):
        r=np.array([1,1,0,0,1],bool);c=np.array([0,1,1,0,1],bool);w=np.array([.1,.1,.2,.3,.3])
        z=c.astype(float)-.95*r
        exact=np.mean([np.prod(1+l*(w/w.max())*z) for l in LAMBDAS])
        self.assertAlmostEqual(np.exp(log_evalue(r,c,w)),exact)

    def test_fst_stops_first_failure(self):
        n=1000;r=np.ones(n,bool);w=np.full(n,1/n)
        comp=dict(margin=np.zeros(n),base_prediction=np.zeros(n),novel_prediction=np.ones(n),base_correct=r)
        out=certify(comp,r,w,[-1.,1.,2.]);self.assertEqual(out['offset'],-1.)
        self.assertEqual(len(out['trace']),2)


if __name__=='__main__':unittest.main()
