"""Optimized readout microbenchmark with numerical parity and explicit storage.

The initial depth benchmark used shared array helpers that compute unused
branches. This separate measurement caches static retrieval values and skips
unused branches for every readout; no accuracy result or selection is changed.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import kbs_depth as depth
from kbs_runtime import configure,atomic_json,digest_file


@torch.no_grad()
def main():
    device=configure();results={}
    for ds in ['cifar','pets','imagenetr']:
        d,entries,_=depth.prior.load(ds)
        if entries is None:entries=json.loads((depth.ROOT/'experiments/kbs_strengthening/imagenetr/partitions.json').read_text())
        entry=entries[0];base,new=[np.array(entry[k]) for k in ['base','novel']];ids=np.r_[base,new]
        Ef,E,T=[d[k].to(device) for k in ['E_fit','E_test','T100']];E=E[:512];T=T[ids]
        W=depth.core.fit_ridge(Ef,d['fit_y'],base,entry['ridge_lambda']);Pb=depth.core.means(Ef,d['fit_y'],base)
        si=depth.core.support_rows(d['support_y'],new,5,7)
        Pn=depth.core.means(d['E_support'][si].to(device),d['support_y'][si],new);P=torch.cat([Pb,Pn])
        saved=json.loads((depth.ROOT/f'experiments/kbs_strengthening/{ds}/p0/partition.json').read_text())
        rp=saved['table_parameters']['retained'];dp=saved['table_parameters']['anchored_tp'];nb=len(base)
        def text():
            old=E@T[:nb].T
            return torch.cat([depth.core.z(old),depth.core.z(E@T[nb:].T,old)],1)
        def retained():
            out=text();out[:,:nb]+=rp[0]*depth.core.z(depth.core.linear(E,W))
            out[:,nb:]+=rp[1]*depth.core.z(E@Pn.T,E@Pb.T)+rp[2];return out
        def dense():
            out=text()+dp[1]*depth.core.z(E@P.T,E@Pb.T);out[:,nb:]+=dp[2];return out
        a=depth.rv.arrays(E,T,Pb,Pn,W)
        parity={}
        for name,fn,params in [('retained',retained,rp),('anchored_tp',dense,dp)]:
            reference=depth.rv.score(a,name,params);reference[:,nb:]+=params[2]
            parity[name]=float((fn()-reference).abs().max());assert parity[name]<1e-4
        jobs={'retained':retained,'anchored_tp':dense}
        sizes={'retained':sum(x.numel()*x.element_size() for x in [T,W,Pb,Pn]),
               'anchored_tp':sum(x.numel()*x.element_size() for x in [T,P])}
        keys,labels=depth.make_keys(d,base,new);keys=keys.to(device)
        mapping={int(c):i for i,c in enumerate(ids)}
        index=torch.tensor([mapping[int(c)] for c in labels],device=device)
        onehot=F.one_hot(index,len(ids)).float()
        selected=json.loads((depth.OUT/ds/'tip/p0/selection.json').read_text())['selected']
        # Bind cache tensors in each closure; construction is excluded from query latency.
        for style in ['sum','mean']:
            params=selected[f'tip_{style}_calibrated'];values=onehot.clone()
            if style=='mean':values/=torch.bincount(index,minlength=len(ids))[None]
            def query(values=values,params=params):
                blocks=[]
                for i in range(0,len(E),256):
                    x=E[i:i+256]
                    blocks.append(100*(x@T.T)+params['alpha']*(torch.exp(-params['beta']*(1-x@keys.T))@values))
                out=torch.cat(blocks);out[:,nb:]+=params['offset'];return out
            jobs['tip_'+style]=query
            sizes['tip_'+style]=sum(x.numel()*x.element_size() for x in [T,keys,values])
            reference=100*(E@T.T)+params['alpha']*depth.retrieval(E,keys,labels,ids,params['beta'],style=='mean')
            reference[:,nb:]+=params['offset']
            parity['tip_'+style]=float((query()-reference).abs().max());assert parity['tip_'+style]<1e-3
        rows={}
        for name,fn in jobs.items():
            for _ in range(5):fn()
            timings=[]
            for _ in range(20):
                if device=='cuda':torch.cuda.synchronize()
                start=time.perf_counter();fn()
                if device=='cuda':torch.cuda.synchronize()
                timings.append((time.perf_counter()-start)*1000)
            rows[name]={'median_ms':float(np.median(timings)),'p90_ms':float(np.percentile(timings,90)),
                        'persistent_tensor_bytes':sizes[name],'score_parity_max_abs':parity[name]}
        results[ds]={'queries':len(E),'cache_keys':len(keys),'active_classes':len(ids),'readouts':rows}
    atomic_json({'source_sha256':digest_file(Path(__file__)),'device':torch.cuda.get_device_name() if device=='cuda' else 'cpu',
                 'warmups':5,'repetitions':20,'image_encoding_included':False,'precision':'float32','results':results},depth.OUT/'cost.json')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
