"""Numerical checks for semantic-cache retrieval and mechanism isolation."""
import unittest
import numpy as np
import torch
import kbs_depth as depth


class DepthTests(unittest.TestCase):
    def test_cache_matches_scalar_published_equation(self):
        torch.manual_seed(10)
        E=torch.nn.functional.normalize(torch.randn(7,8),dim=1)
        keys=torch.nn.functional.normalize(torch.randn(6,8),dim=1)
        labels=np.array([4,4,4,2,2,9]);ids=np.array([4,2,9]);beta=2.3
        actual=depth.retrieval(E,keys,labels,ids,beta,chunk=2)
        expected=torch.tensor([[sum(float(torch.exp(-beta*(1-(x*key).sum())))
                    for key,label in zip(keys,labels) if label==c) for c in ids] for x in E])
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
        mean=depth.retrieval(E,keys,labels,ids,beta,mean=True,chunk=3)
        self.assertTrue(torch.allclose(mean,expected/torch.tensor([3,2,1]),atol=1e-6))

    def test_cache_existing_columns_survive_append(self):
        torch.manual_seed(22);E=torch.randn(9,8);keys=torch.randn(6,8)
        labels=np.array([0,0,1,1,2,2])
        old=depth.retrieval(E,keys[:4],labels[:4],[0,1],1)
        expanded=depth.retrieval(E,keys,labels,[0,1,2],1)
        self.assertTrue(torch.allclose(old,expanded[:,:2],atol=1e-5))

    def test_dynamic_control_changes_only_reference_statistics(self):
        torch.manual_seed(31);E=torch.randn(10,8);T=torch.randn(7,8)
        Pb=torch.randn(3,8);Pn=torch.randn(4,8);W=torch.randn(9,3)
        a=depth.dynamic_arrays(E,T,Pb,Pn,W)
        l,t,p=depth.core.branches(E,T,Pb,Pn,W,'dynamic')
        self.assertTrue(torch.equal(a['text'],t));self.assertTrue(torch.equal(a['learned'],l))
        self.assertTrue(torch.equal(a['prototype'][:,3:],p))


if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
