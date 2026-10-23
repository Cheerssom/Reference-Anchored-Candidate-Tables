"""Controlled normalization and author-code-controlled HyCal feature transfer."""
from pathlib import Path
import ast,math
import numpy as np
import torch
import torch.nn.functional as F
import fcs_expansion as core

HYCAL_LAMBDAS=(1e-4,1e-3,1e-2)


def fixed_moments(E,T,P,W):
    result=[]
    for values in [E@T.T,core.linear(E,W),E@P.T]:
        result.append((float(values.mean()),float(values.std(1,correction=0).mean().clamp_min(1e-6))))
    return result


def normalization_scores(E,T,Pb,Pn,W,parameters,moments,center='query',scale='query',learned_factor=1.):
    nb=len(Pb);out=[]
    for i,(values,reference) in enumerate([(E@T.T,E@T[:nb].T),(core.linear(E,W),None),
                                          (E@torch.cat([Pb,Pn]).T,E@Pb.T)]):
        if reference is None:reference=values
        mu=reference.mean(1,keepdim=True) if center=='query' else moments[i][0]
        sig=reference.std(1,correction=0,keepdim=True).clamp_min(1e-6) if scale=='query' else moments[i][1]
        out.append((values-mu)/sig)
    a,u,_=parameters;scores=out[0].clone()
    scores[:,:nb]+=a*learned_factor*out[1];scores[:,nb:]+=u*out[2][:,nb:]
    return scores


def hycal_statistics(E,T,regularization):
    fused=F.normalize(E+T[None],dim=1);mu=F.normalize(fused.mean(0),dim=0)
    A=(fused-mu).double();n=len(A);assert n>1
    # Woodbury: exact regularized precision, stored in sample rather than feature space.
    L=torch.linalg.cholesky(A@A.T+regularization*(n-1)*torch.eye(n,device=E.device,dtype=torch.float64))
    B=torch.linalg.solve_triangular(L,A,upper=False).float()
    return dict(mu=mu,B=B,n=n,regularization=regularization)


def hycal_distances(E,T,state):
    z=F.normalize(E+T[None],dim=1);diff=z-state['mu']
    cos=z@state['mu'];dist=((diff*diff).sum(1)-(diff@state['B'].T).square().sum(1))/state['regularization']
    return cos,dist.clamp_min(0.)


def cdf_normalize(values):
    z=(values-values.mean(1,keepdim=True))/values.std(1,correction=1,keepdim=True).clamp_min(1e-6)
    return .5*(1+torch.erf(z/math.sqrt(2)))


def hycal_scores(cosine,negative_distances,counts):
    cosine=cdf_normalize(cosine);maha=cdf_normalize(negative_distances)
    n=torch.as_tensor(counts,device=cosine.device,dtype=cosine.dtype)
    weight=torch.where(n>=5,torch.sigmoid(n-5),torch.zeros_like(n))
    # The official shared-background Mahalanobis term cancels in per-query z scores.
    return (1-weight[None])*cosine+weight[None]*maha


def official_fixtures(root):
    """Execute only author precision/CDF methods, without model loading or installation."""
    torch.manual_seed(23102026);E=F.normalize(torch.randn(9,12),dim=1);T=F.normalize(torch.randn(12),dim=0)
    Q=F.normalize(torch.randn(17,12),dim=1);reg=.001
    private=Path(root)/'paper/kbs/literature/reviewer_baselines/hycal/hycal'
    def isolated(file,cls,methods):
        parsed=ast.parse((private/file).read_text(encoding='utf-8'))
        node=next(n for n in parsed.body if isinstance(n,ast.ClassDef) and n.name==cls)
        node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in methods]
        env={'torch':torch,'math':math,'Optional':__import__('typing').Optional}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(private/file),'exec'),env)
        return env[cls]()
    model=isolated('hycal_model.py','HyCalModel',['_compute_regularized_precision'])
    model.lambda_reg=reg;model.device='cpu';model.gamma=1
    X=F.normalize(E+T,dim=1);mu=F.normalize(X.mean(0),dim=0);A=X-mu;cov=A.T@A/(len(X)-1)
    P=model._compute_regularized_precision(cov,real=False)
    official=((F.normalize(Q+T,dim=1)-mu)@P*(F.normalize(Q+T,dim=1)-mu)).sum(1)
    own=hycal_distances(Q,T,hycal_statistics(E,T,reg))[1]
    relative=float(((own-official).abs()/official.abs().clamp_min(1e-6)).max())
    assert relative<5e-4,relative
    evaluator=isolated('hycal_evaluator.py','HyCalEvaluator',['_gaussian_cdf','_normalize_rmd_scores'])
    fixture=torch.randn(17,11);cdf_delta=float((cdf_normalize(fixture)-evaluator._normalize_rmd_scores(fixture)).abs().max())
    assert cdf_delta<1e-7
    return dict(precision_max_relative_error=relative,cdf_max_absolute_error=cdf_delta,
                official_commit='4558fb7fa6aa389310c9e9dc7db7c2cbebc84a41')
