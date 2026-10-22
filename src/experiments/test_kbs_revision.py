"""Simple fusion controls: equations and fixed-bank append behavior."""
import unittest
import numpy as np
import torch
from kbs_revision import simple_arrays,temperatures
import fcs_retention_value as rv


class RevisionTests(unittest.TestCase):
    def test_raw_scalar_equation(self):
        torch.manual_seed(8);E=torch.randn(4,8);T=torch.randn(5,8);Pb=torch.randn(3,8);Pn=torch.randn(2,8);W=torch.randn(9,3)
        a=simple_arrays(E,T,Pb,Pn,W,[1.,1.,1.]);sc=rv.score(a,'retained',(2.,.5,0.))
        expected=E@T.T;expected[:,:3]+=2*(E@W[:-1]+W[-1]);expected[:,3:]+=.5*(E@Pn.T)
        self.assertTrue(torch.allclose(sc,expected,atol=1e-6))

    def test_fixed_temperatures_preserve_existing_columns(self):
        torch.manual_seed(9);E=torch.randn(6,8);T=torch.randn(6,8);Pb=torch.randn(3,8);Pn=torch.randn(3,8);W=torch.randn(9,3)
        scale=temperatures(E,T[:3],Pb,W);self.assertTrue(all(v>0 for v in scale))
        old=rv.score(simple_arrays(E,T[:5],Pb,Pn[:2],W,scale),'retained',(1.,2.,0.))
        full=rv.score(simple_arrays(E,T,Pb,Pn,W,scale),'retained',(1.,2.,0.))
        self.assertTrue(torch.allclose(old,full[:,:5],atol=1e-6))


if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
