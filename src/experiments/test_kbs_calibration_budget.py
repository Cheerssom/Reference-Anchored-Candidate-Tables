"""Coverage-budget scientific checks, beyond reproducing displayed rows."""
import unittest
import numpy as np
import kbs_calibration_budget as budget
import kbs_risk_control as risk


class BudgetTests(unittest.TestCase):
    def test_nested_subsets_and_capped_availability(self):
        rows=[dict(class_id=c) for c,n in [(0,40),(1,14),(2,30)] for _ in range(n)]
        for seed in [0,1,2]:
            subsets=budget.nested_subsets(rows,seed);last=set()
            for cap,ix in subsets.items():
                self.assertTrue(last<=set(ix));self.assertEqual(len(ix),len(set(ix)))
                for c,n in [(0,40),(1,14),(2,30)]:
                    self.assertEqual(sum(rows[i]['class_id']==c for i in ix),min(cap,n))
                last=set(ix)
            self.assertEqual(last,set(range(len(rows))))

    def test_largest_cap_is_one_identical_set(self):
        rows=[dict(class_id=c) for c in range(3) for _ in range(40)]
        for seed in [1,2]:
            np.testing.assert_array_equal(budget.nested_subsets(rows,0)[40],budget.nested_subsets(rows,seed)[40])
        self.assertFalse(np.array_equal(budget.nested_subsets(rows,0)[5],budget.nested_subsets(rows,1)[5]))

    def test_coverage_not_total_labels_drives_conservative_bound(self):
        unbalanced=np.array([0]*2+[1]*398);balanced=np.array([0]*200+[1]*200)
        pu,_=risk.pvalue(np.zeros(400),risk.weights(unbalanced,[0,1]))
        pb,_=risk.pvalue(np.zeros(400),risk.weights(balanced,[0,1]))
        self.assertGreater(pu,.01);self.assertLess(pb,.01)


if __name__=='__main__':unittest.main()
