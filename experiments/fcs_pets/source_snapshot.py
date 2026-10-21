"""Frozen-CLIP expansion with base-only calibration and auditable sample IDs.

Run from any directory: python src/experiments/fcs_expansion.py
Protocol fixed before first evaluation, 2026-10-21. Historical LoRA features are
never read. Existing frozen E_* features are checked against re-embedded images.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from lt_dataset import make_lt_datasets
from train import split_train_val

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments' / 'fcs'
PARTS, SUPPORTS, SHOTS = (0, 1, 2), (7, 8, 9), (1, 2, 5)
GATES = [(a, c, d) for a in (0., .25, .5, 1., 2.)
         for c in (0., .25, .5, 1., 2.)
         for d in (-2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2.)]
LAMBDAS = (.01, .1, 1., 10.)


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def partition(buckets, seed, n_base=40, n_negative=20):
    """Exact stratified allocation, randomizing members before selection."""
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(np.flatnonzero(np.asarray(buckets) == b))
             for b in ('head', 'medium', 'tail')]
    def allocate(n, sizes):
        ideal = n * np.asarray(sizes, float) / sum(sizes)
        q = np.floor(ideal).astype(int)
        for i in np.argsort(-(ideal - q), kind='stable')[:n-int(q.sum())]:
            q[i] += 1
        return q
    q = allocate(n_base, [len(p) for p in pools])
    base = np.sort(np.concatenate([p[:v] for p, v in zip(pools, q)]))
    rest = [p[v:] for p, v in zip(pools, q)]
    qn = allocate(n_negative, [len(p) for p in rest])
    neg = np.sort(np.concatenate([p[:v] for p, v in zip(rest, qn)]))
    new = np.sort(np.concatenate([p[v:] for p, v in zip(rest, qn)]))
    assert len(base) == n_base and len(neg) == n_negative
    assert not set(base) & set(new) and len(base)+len(new)+len(neg)==len(buckets)
    return base, new, neg


def norm(x):
    return F.normalize(x, dim=1)


def z(s, reference=None):
    r = s if reference is None else reference
    return (s-r.mean(1, keepdim=True))/r.std(1, correction=0, keepdim=True).clamp_min(1e-6)


def means(E, y, ids):
    return norm(torch.stack([E[y == c].mean(0) for c in ids]))


def support_rows(y, ids, k, seed):
    rng = np.random.default_rng(seed)
    sel = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:k] for c in ids])
    assert all(np.sum(y[sel] == c) == k for c in ids)
    return sel


def fit_ridge(E, y, ids, lam):
    keep = np.isin(y, ids)
    x = E[keep].double()
    yy = np.searchsorted(ids, y[keep])
    counts = np.bincount(yy, minlength=len(ids))
    w = torch.as_tensor(1./counts[yy], device=E.device, dtype=torch.float64)
    target = F.one_hot(torch.tensor(yy, device=E.device), len(ids)).double()
    # Unregularized intercept; slope regularization avoids inverse covariance
    # assumptions and is fitted using base-class training examples only.
    x = torch.cat([x, torch.ones(len(x), 1, device=x.device)], 1)
    reg = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)*lam
    reg[-1, -1] = 0
    return torch.linalg.solve(x.T@(x*w[:, None])+reg, x.T@(target*w[:, None])).float()


def linear(E, W):
    return E@W[:-1]+W[-1]


def select_ridge(Ef, yf, Ev, yv, ids):
    mask = np.isin(yv, ids)
    best = (-1., None, None)
    for lam in LAMBDAS:
        W = fit_ridge(Ef, yf, ids, lam)
        pred = ids[linear(Ev[mask], W).argmax(1).cpu().numpy()]
        acc = np.mean([np.mean(pred[yv[mask] == c] == c) for c in ids])
        if acc > best[0]:
            best = (acc, W, lam)
    return best[1:]


def metrics(pred, y, base, new):
    old = float(np.mean([np.mean(pred[y == c] == c) for c in base]))
    novel = float(np.mean([np.mean(pred[y == c] == c) for c in new]))
    return {'old': old, 'new': novel,
            'overall': (len(base)*old+len(new)*novel)/(len(base)+len(new)),
            'harmonic': 2*old*novel/max(old+novel, 1e-12)}


def branches(E, T, Pbase, Pnew, W, mode):
    b = len(Pbase)
    ts = E@T.T
    ls = z(linear(E, W))
    ps = E@Pnew.T
    if mode == 'reference':
        # Compute the reference block separately, so changing GEMM output width
        # cannot perturb the old-row scores through floating-point kernels.
        old_ts = E@T[:b].T
        tz = torch.cat([z(old_ts), z(E@T[b:].T, old_ts)], 1)
        pz = z(ps, E@Pbase.T)
    elif mode == 'dynamic':
        tz, pz = z(ts), z(ps)
    elif mode == 'raw':
        tz, ls, pz = ts, linear(E, W), ps
    else:
        raise ValueError(mode)
    return ls, tz, pz


def score(B, gate):
    ls, ts, ps = B
    a, c, d = gate
    out = ts.clone()
    out[:, :ls.shape[1]] += a*ls
    out[:, ls.shape[1]:] += c*ps+d
    return out


def choose_gate(B, y, base, new):
    ids = np.r_[base, new]
    best = (-1., None)
    for gate in GATES:
        pred = ids[score(B, gate).argmax(1).cpu().numpy()]
        value = metrics(pred, y, base, new)['harmonic']
        if value > best[0]+1e-12:
            best = (value, gate)
    return best[1], best[0]


@torch.no_grad()
def embed_support(support, fit, cache, device):
    from transformers import CLIPModel
    path = OUT/'frozen_support.pt'
    if path.exists():
        d = torch.load(path, map_location='cpu', weights_only=False)
        assert d['support_image_hash'] == digest(support.x)
        assert d['fit_image_hash'] == digest(fit.x)
        return d['features'].to(device), d['checks']
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14',
                                     local_files_only=True).to(device).eval()
    mean = torch.tensor([.48145466, .4578275, .40821073],device=device)[None,:,None,None]
    std = torch.tensor([.26862954, .26130258, .27577711],device=device)[None,:,None,None]
    def run(X):
        out = []
        for i in range(0, len(X), 32):
            x = torch.from_numpy(X[i:i+32]).to(device).float()/255
            x = F.interpolate(x, size=(224,224), mode='bicubic', align_corners=False)
            with torch.autocast(device, enabled=device=='cuda', dtype=torch.float16):
                v = model.vision_model(pixel_values=(x-mean)/std)
                e = model.visual_projection(v.pooler_output)
            out.append(norm(e.float()).cpu())
        return torch.cat(out)
    idx = np.array([0, 1, 17, 103, 999, len(fit)-1])
    actual = run(fit.x[idx])
    cosine = F.cosine_similarity(actual, cache['E_fit'][idx], dim=1)
    checks = {'fit_indices': idx.tolist(), 'cache_check_cosines':cosine.tolist()}
    assert cosine.min() > .999, checks
    features = run(support.x)
    torch.save({'features':features, 'checks':checks,
                'support_image_hash':digest(support.x), 'fit_image_hash':digest(fit.x)},path)
    del model
    if device == 'cuda': torch.cuda.empty_cache()
    return features.to(device), checks


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dataset',choices=['cifar','pets'],default='cifar')
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    t0 = time.perf_counter()
    dev = args.device
    if args.dataset=='pets':
        OUT=ROOT/'experiments'/'fcs_pets'
        from fcs_pets_data import prepare
        cache,fit,val,support,test,meta,cache_path=prepare(OUT,dev)
        Es=cache['E_support'].to(dev)
        checks={'feature_source':'fresh frozen CLIP extraction; no historical cache'}
        nbase,nneg=15,7
    else:
        OUT.mkdir(parents=True,exist_ok=True)
        train, support, _, test, meta = make_lt_datasets(rho=100)
        fit, val = split_train_val(train)
        cache_path = ROOT/'src'/'expand_cache.pt'
        if cache_path.exists():
            cache = torch.load(cache_path,map_location='cpu',weights_only=False)
            Es, checks = embed_support(support,fit,cache,dev)
        else:
            cache_path=OUT/'frozen_cifar.pt'
            if cache_path.exists():
                cache=torch.load(cache_path,map_location='cpu',weights_only=False)
            else:
                from fcs_frozen_features import prepare
                cache=prepare(cache_path,{'fit':fit,'val':val,'support':support,'test':test},dev)
            Es=cache['E_support'].to(dev)
            checks={'feature_source':'fresh frozen CLIP extraction; no historical cache'}
        nbase,nneg=40,20
    for key, ds in [('E_fit',fit),('E_val',val),('E_test',test)]:
        assert len(cache[key]) == len(ds)
    Ef, Ev, Et, T = [cache[k].to(dev) for k in ('E_fit','E_val','E_test','T100')]
    yf,yv,yt,ys = fit.fine,val.fine,test.fine,support.fine
    # Check disjointness directly by image hash rather than relying on order.
    pools = [{digest(x) for x in ds.x} for ds in (fit,val,support,test)]
    assert all(not pools[i]&pools[j] for i in range(3) for j in range(i+1,3))
    manifest = {'protocol':'fcs-frozen-clip-base-only-v1','dataset':args.dataset, 'created':'2026-10-21',
                'parts':PARTS,'support_seeds':SUPPORTS,'shots':SHOTS,'gates':GATES,
                'ridge_lambdas':LAMBDAS,'calibration_objective':'macro harmonic old/new',
                'novel_calibration_labels':0, 'expansion_gradient_steps':0,
                'cache_sha256':hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'feature_checks':checks,'fit_hash':digest(fit.x),
                'val_hash':digest(val.x),'support_hash':digest(support.x),
                'test_hash':digest(test.x),'partitions':[]}
    (OUT/'source_snapshot.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    if args.dataset=='pets':
        helper=Path(__file__).with_name('fcs_pets_data.py')
        manifest['data_loader_sha256']=hashlib.sha256(helper.read_bytes()).hexdigest()
        (OUT/'data_loader_snapshot.py').write_text(helper.read_text(encoding='utf-8'),encoding='utf-8')
    records, invariance, efficiency, vocabulary, frontier = [],[],[],[],[]
    seen_partitions = set()
    for part in PARTS:
        base,new,neg = partition(meta['buckets'],part,nbase,nneg)
        vocab_sizes=(max(1,len(new)//4),len(new)//2,len(new))
        assert tuple(base) not in seen_partitions
        seen_partitions.add(tuple(base))
        rng = np.random.default_rng(100+part)
        pseudo = rng.permutation(base)
        cut=len(base)//2
        pb,pn = np.sort(pseudo[:cut]),np.sort(pseudo[cut:])
        W,lam = select_ridge(Ef,yf,Ev,yv,base)
        Wp,lamp = select_ridge(Ef,yf,Ev,yv,pb)
        Pbase,Ppb = means(Ef,yf,base),means(Ef,yf,pb)
        vmask = np.isin(yv,base)
        Ecal, ycal = Ev[vmask],yv[vmask]
        ids,calids = np.r_[base,new],np.r_[pb,pn]
        entry = {'seed':part,'base':base.tolist(),'novel':new.tolist(),'negative':neg.tolist(),
                 'pseudo_base':pb.tolist(),'pseudo_novel':pn.tolist(),
                 'ridge_lambda':lam,'pseudo_ridge_lambda':lamp,'supports':[]}
        manifest['partitions'].append(entry)
        for ss in SUPPORTS:
            for k in SHOTS:
                si = support_rows(ys,new,k,ss)
                ci = support_rows(ys,pn,k,ss)
                Pnew = means(Es[si],ys[si],new)
                Ppn = means(Es[ci],ys[ci],pn)
                entry['supports'].append({'seed':ss,'k':k,'novel_support_indices':si.tolist(),
                                          'pseudo_support_indices':ci.tolist()})
                gate_info = {}
                for mode in ('reference','dynamic','raw'):
                    Bcal = branches(Ecal,T[calids],Ppb,Ppn,Wp,mode)
                    gate,value = choose_gate(Bcal,ycal,pb,pn)
                    Btest = branches(Et,T[ids],Pbase,Pnew,W,mode)
                    scores = score(Btest,gate)
                    pred = ids[scores.argmax(1).cpu().numpy()]
                    r = {'part':part,'support_seed':ss,'k':k,'method':mode,
                         'gates':gate,'calibration_harmonic':value,
                         **metrics(pred,yt,base,new)}
                    records.append(r)
                    gate_info[mode] = gate
                    if mode == 'reference' and ss==7 and k==5:
                        # Scores for old rows must be exactly vocabulary invariant.
                        closed = z(Et@T[base].T)+gate[0]*z(linear(Et,W))
                        drift = float((closed-scores[:,:len(base)]).abs().max())
                        oldmask = np.isin(yt,base)
                        p0 = base[closed.argmax(1).cpu().numpy()]
                        invariance.append({'part':part,'max_old_score_drift':drift,
                            'old_prediction_churn':float(np.mean(p0[oldmask]!=pred[oldmask])),
                            'correct_to_wrong':float(np.mean((p0[oldmask]==yt[oldmask])&(pred[oldmask]!=yt[oldmask]))),
                            'wrong_to_correct':float(np.mean((p0[oldmask]!=yt[oldmask])&(pred[oldmask]==yt[oldmask]))),
                            'old_closed_accuracy':float(np.mean(p0[oldmask]==yt[oldmask]))})
                        assert drift < 1e-5
                        np.savez_compressed(OUT/f'predictions_p{part}_s{ss}_k{k}.npz',
                            y=yt,expanded=pred,closed=p0,base=base,novel=new)
                    if ss==7 and k==5 and mode in ('reference','dynamic'):
                        closed = z(Et@T[base].T)+gate[0]*z(linear(Et,W))
                        oldmask=np.isin(yt,base)
                        for nadd in vocab_sizes:
                            ids_n=np.r_[base,new[:nadd]]
                            sn=score(branches(Et,T[ids_n],Pbase,Pnew[:nadd],W,mode),gate)
                            before=base[closed.argmax(1).cpu().numpy()]
                            within=base[sn[:,:len(base)].argmax(1).cpu().numpy()]
                            after=ids_n[sn.argmax(1).cpu().numpy()]
                            vocabulary.append({'part':part,'method':mode,'added':nadd,
                                'max_old_score_drift':float((closed-sn[:,:len(base)]).abs().max()),
                                'old_rank_churn':float(np.mean(before[oldmask]!=within[oldmask])),
                                'old_prediction_churn':float(np.mean(before[oldmask]!=after[oldmask])),
                                **metrics(after,yt,base,new[:nadd])})
                        # A descriptive operating curve, not test-set selection.
                        for offset in (-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,2.):
                            pp=ids[score(Btest,(gate[0],gate[1],offset)).argmax(1).cpu().numpy()]
                            frontier.append({'part':part,'method':mode,'offset':offset,
                                             **metrics(pp,yt,base,new)})
                    # Within the same run, mechanism ablations use the selected
                    # gate unchanged, isolating evidence removal.
                    if mode == 'reference':
                        for ab,ag in [('no_learned',(0.,gate[1],gate[2])),
                                      ('no_prototype',(gate[0],0.,gate[2])),
                                      ('no_offset',(gate[0],gate[1],0.))]:
                            pp = ids[score(Btest,ag).argmax(1).cpu().numpy()]
                            records.append({'part':part,'support_seed':ss,'k':k,
                                            'method':ab,**metrics(pp,yt,base,new)})
                # Text/prototype baseline, with its mixing weight chosen on the
                # SAME pseudo-novel validation episode (including alpha=0).
                pfull = torch.cat([Pbase,Pnew])
                pcal = torch.cat([Ppb,Ppn])
                best = (-1.,0.)
                for alpha in (0.,.25,.5,1.,2.,4.):
                    pp = calids[(Ecal@T[calids].T+alpha*(Ecal@pcal.T)).argmax(1).cpu().numpy()]
                    h = metrics(pp,ycal,pb,pn)['harmonic']
                    if h>best[0]:best=(h,alpha)
                for name,sc in [('text',Et@T[ids].T),('prototype',Et@pfull.T),
                               ('text_prototype',Et@T[ids].T+best[1]*(Et@pfull.T))]:
                    pp=ids[sc.argmax(1).cpu().numpy()]
                    records.append({'part':part,'support_seed':ss,'k':k,'method':name,
                                    **metrics(pp,yt,base,new)})
                # Tuned training-free Tip-Adapter with a balanced k-shot cache
                # for ALL candidates; no favorable old/new cache imbalance.
                allsi = support_rows(ys,ids,k,ss)
                calsi = support_rows(ys,calids,k,ss)
                def tip(E, rows, classids, beta):
                    labels = torch.tensor(np.array([np.where(classids==yy)[0][0] for yy in ys[rows]]),device=dev)
                    onehot = F.one_hot(labels,len(classids)).float()
                    return torch.exp(beta*(E@Es[rows].T-1))@onehot
                best=(-1.,0.,1.)
                for beta in (1.,2.,5.,10.):
                    cachecal=tip(Ecal,calsi,calids,beta)
                    for alpha in (0.,.5,1.,2.,4.):
                        pp=calids[(100*(Ecal@T[calids].T)+alpha*cachecal).argmax(1).cpu().numpy()]
                        h=metrics(pp,ycal,pb,pn)['harmonic']
                        if h>best[0]:best=(h,alpha,beta)
                pp=ids[(100*(Et@T[ids].T)+best[1]*tip(Et,allsi,ids,best[2])).argmax(1).cpu().numpy()]
                records.append({'part':part,'support_seed':ss,'k':k,'method':'tip_adapter',
                                'alpha':best[1],'beta':best[2],**metrics(pp,yt,base,new)})
                print(f'part={part} support={ss} k={k} '+
                      ' '.join(f'{r["method"]}:{r["harmonic"]:.4f}' for r in records[-10:] if r.get('k')==k),flush=True)
        # Feature-space scoring latency (not encoder latency), fixed batch.
        X=Et[:256]
        for n in vocab_sizes:
            pnsub=Pnew[:n]
            tsub=T[np.r_[base,new[:n]]]
            for _ in range(5):score(branches(X,tsub,Pbase,pnsub,W,'reference'),gate_info['reference'])
            if dev=='cuda':torch.cuda.synchronize()
            start=time.perf_counter()
            for _ in range(50):score(branches(X,tsub,Pbase,pnsub,W,'reference'),gate_info['reference'])
            if dev=='cuda':torch.cuda.synchronize()
            efficiency.append({'part':part,'candidates':len(base)+n,'batch':256,
                               'milliseconds':(time.perf_counter()-start)*1000/50})
        (OUT/'results.json').write_text(json.dumps({'records':records,'invariance':invariance,
                    'efficiency':efficiency,'vocabulary':vocabulary,'frontier':frontier,
                    'elapsed_seconds':time.perf_counter()-t0},indent=2))
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'FINISHED {time.perf_counter()-t0:.1f}s',flush=True)


if __name__=='__main__':main()
