"""Checks for finite-sample threshold boundaries and stratified budgets."""
import unittest
import numpy as np
from kbs_validation_budget import select_threshold,select_rows,accuracy


class BudgetTests(unittest.TestCase):
    def test_threshold_matches_exhaustive_boundaries(self):
        rng=np.random.default_rng(109)
        for _ in range(50):
            margin=rng.integers(-2,3,16).astype(float)
            bp=rng.integers(0,4,16);npred=rng.integers(0,4,16)
            correct=rng.random(16)>.25;w=np.ones(16)/16
            c={'margin':margin,'base_prediction':bp,'novel_prediction':npred,
               'base_correct':correct,'novel_correct':~correct,'weights':w}
            target=.5;offset,ceiling=select_threshold(c,target)
            candidates=sorted(set([float(v) for v in margin]+[float(np.nextafter(v,-np.inf)) for v in margin]))
            feasible=[v for v in candidates if accuracy(c,v)[0]>=target-1e-10]
            if feasible:
                self.assertEqual(offset,max(feasible));self.assertAlmostEqual(ceiling,w[correct].sum())
            else:self.assertIsNone(offset)

    def test_nested_stratified_deterministic_subsets(self):
        y=np.array([4]*9+[7]*3+[12]);base=np.array([4,7,12])
        small=select_rows(y,base,.25,1,2);large=select_rows(y,base,.5,1,2)
        self.assertTrue(set(small)<=set(large));self.assertEqual(len(small),4)
        self.assertTrue(np.array_equal(small,select_rows(y,base,.25,1,2)))
        self.assertEqual(set(y[small]),set(base));self.assertEqual(select_rows(y,base,1.,1,0).tolist(),list(range(len(y))))


if __name__=='__main__':unittest.main()
