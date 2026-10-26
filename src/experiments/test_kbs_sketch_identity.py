import unittest
import numpy as np
from kbs_sketch_prepare import dihedral_hash
class IdentityTests(unittest.TestCase):
 def test_dihedral_equivalents_share_one_unit(self):
  x=np.random.default_rng(27102026).integers(0,256,(13,13,3),dtype=np.uint8);expected=dihedral_hash(x)
  for k in range(4):
   self.assertEqual(expected,dihedral_hash(np.rot90(x,k)))
   self.assertEqual(expected,dihedral_hash(np.fliplr(np.rot90(x,k))))
 def test_distinct_patterns_are_not_collapsed(self):
  x=np.random.default_rng(2).integers(0,256,(13,13,3),dtype=np.uint8);y=x.copy();y[0,0,0]^=1
  self.assertNotEqual(dihedral_hash(x),dihedral_hash(y))
if __name__=='__main__':unittest.main()
