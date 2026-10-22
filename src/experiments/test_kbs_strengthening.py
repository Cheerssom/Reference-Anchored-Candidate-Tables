"""Numerical and restart behavior tests for the evidence extension."""
from pathlib import Path
import sys,tempfile,unittest,json
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from kbs_fenec import covariance_inverse,centroids,log_votes,fit,distances
from kbs_runtime import atomic_torch,atomic_json

class EvidenceTests(unittest.TestCase):
 def test_atomic_checkpoint(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'state.pt';atomic_torch({'done':torch.arange(3)},p)
   self.assertEqual(torch.load(p,weights_only=False)['done'].tolist(),[0,1,2])
   atomic_torch({'done':torch.arange(5)},p)
   self.assertEqual(torch.load(p,weights_only=False)['done'].tolist(),list(range(5)))
   self.assertFalse(p.with_name('state.pt.tmp').exists())
 def test_resume_class_fit(self):
  torch.manual_seed(91);x=torch.randn(24,8);y=np.repeat([0,1],12)
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'fit.pt';a=fit(x,y,[0,1],3,1.,'cpu',p)
   a_path=p.with_suffix('')/'0.pt';stamp=a_path.stat().st_mtime_ns
   b=fit(x,y,[0,1],3,1.,'cpu',p)
   self.assertEqual(stamp,a_path.stat().st_mtime_ns)
   self.assertTrue(torch.equal(a['classes']['0']['inverse'],b['classes']['0']['inverse']))
 def test_five_shot_and_single_sample(self):
  torch.manual_seed(1);x=torch.randn(6,8);y=np.array([0]+[1]*5)
  state=fit(x,y,[0,1],5,1.,'cpu')
  d,lab=distances(state,x,[0,1],'cpu')
  self.assertTrue(torch.isfinite(d).all())
  self.assertEqual(state['classes']['0']['centroids'].shape[0],1)
  self.assertTrue(state['classes']['0']['single_sample_fallback'])
  score=log_votes(d,lab,[0,1],3)
  self.assertTrue(torch.isfinite(score).all())
 def test_saved_author_fixture(self):
  from kbs_fenec_vit import inverse_once
  p=Path(__file__).resolve().parents[2]/'experiments/kbs_strengthening/fenec_parity_fixture.json'
  d=json.loads(p.read_text());x=torch.tensor(d['input'])
  self.assertTrue(torch.allclose(covariance_inverse(x),torch.tensor(d['inverse_double']),atol=1e-5,rtol=1e-4))
  self.assertTrue(torch.allclose(inverse_once(x,6.12,8.10),torch.tensor(d['inverse_single']),atol=1e-5,rtol=1e-4))
  self.assertTrue(torch.allclose(centroids(x,3),torch.tensor(d['centroids']),atol=1e-6))
 @unittest.skipUnless((Path(__file__).resolve().parents[2]/'paper/kbs/literature/fenec_official/fenec.py').exists(),'Optional live author-code fixture; the saved author-output fixture is checked separately.')
 def test_upstream_parity(self):
  upstream=Path(__file__).resolve().parents[2]/'paper/kbs/literature/fenec_official'
  sys.path.insert(0,str(upstream))
  from metrics import MahalanobisMetric,EuclideanMetric
  from kmeans import KMeans
  from fenec import FeNeC
  torch.manual_seed(13);x=torch.randn(20,8)
  metric=MahalanobisMetric(shrinkage=2,gamma_1=1,gamma_2=1,normalization=True)
  metric.preprocess([x]);ours=covariance_inverse(x)
  self.assertTrue(torch.allclose(metric.inv_cov_matrix[0],ours,atol=1e-5,rtol=1e-4))
  upstream_cent=KMeans(3,metric=EuclideanMetric(),seed=42).fit_predict(x[None])[0]
  self.assertTrue(torch.allclose(upstream_cent,centroids(x,3),atol=1e-6))
  d=torch.tensor([[[1.,2.,3.],[4.,5.,6.]],[[4.,5.,6.],[1.,2.,3.]]])
  prediction=FeNeC(3,metric=EuclideanMetric()).model_predict(d)
  actual=log_votes(d.reshape(2,-1),np.repeat([0,1],3),[0,1],3).argmax(1)
  self.assertTrue(torch.equal(prediction,actual))

if __name__=='__main__':
 torch.set_num_threads(2);unittest.main()
