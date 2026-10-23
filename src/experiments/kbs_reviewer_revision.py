"""Five-block revision readouts, diagnostics and corruption-policy comparison.

GPU work is serialized with the existing runtime. Every per-session result is
atomic and resumable. Prior experiments and frozen model selection are reused.
"""
from pathlib import Path
import argparse,gc,json,sys,time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fcs_expansion as core
import fcs_retention_value as rv
import kbs_strengthen as prior
import kbs_calibration_budget as budget
import kbs_reviewer_readouts as readouts
import kbs_calibration_fairness as fairness
import kbs_risk_control as risk
from kbs_runtime import configure,check_pause,atomic_json,atomic_torch,atomic_npz,digest_file

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/kbs_reviewer_revision/readouts'
VARIANTS={'query_center_fixed_scale':('query','fixed',1.),
          'fixed_center_query_scale':('fixed','query',1.),
          'fixed_center_fixed_scale':('fixed','fixed',1.),
          'classifier_half':('query','query',.5),'classifier_zero':('query','query',0.)}


def load_data(dataset,encoder):
    if encoder=='l14':d,_,path=prior.load(dataset)
    else:
        path=ROOT/f'experiments/kbs_revision/b32_features/{dataset}/frozen.pt'
        d=torch.load(path,map_location='cpu',weights_only=False)
    split=json.loads((ROOT/f'experiments/kbs_independent_risk/splits/{dataset}.json').read_text())
    E=torch.cat([d['E_fit'],d['E_val']]);y=np.r_[d['fit_y'],d['val_y']]
    pool=json.loads((ROOT/f'experiments/kbs_calibration_budget/pools/{dataset}.json').read_text())
    extras=torch.load(ROOT/f'experiments/kbs_calibration_budget/features/{encoder}/{dataset}/extra.pt',map_location='cpu',weights_only=False)['features']
    pieces=[];j=0
    for r in pool['rows']:
        if 'cached_index' in r:pieces.append(E[r['cached_index']])
        else:pieces.append(extras[j]);j+=1
    result=dict(E_fit=E[split['indices']['fit']],fit_y=y[split['indices']['fit']],
                E_tune=E[split['indices']['tune']],tune_y=y[split['indices']['tune']],
                E_cal=torch.stack(pieces),cal_y=np.array([r['class_id'] for r in pool['rows']]),
                E_test=d['E_test'],test_y=d['test_y'],E_support=d['E_support'],support_y=d['support_y'],T=d['T100'])
    if dataset=='cifar':
        for corruption in ['noise08','blur125']:
            p=ROOT/f'experiments/kbs_reviewer_revision/shift_features/{encoder}/{corruption}/features.pt'
            value=torch.load(p,map_location='cpu',weights_only=False);n=value['calibration_count'];ids=value['test_ids']
            result['E_cal_'+corruption]=value['features'][:n];result['cal_'+corruption+'_y']=value['labels'][:n]
            result['E_test_'+corruption]=value['features'][n:];result['test_'+corruption+'_y']=value['labels'][n:]
            result['shift_test_ids']=ids
    return result,path


@torch.no_grad()
def prepare_hycal(d,out,device):
    qnames=['tune','cal','test']+(['cal_noise08','test_noise08','cal_blur125','test_blur125'] if 'E_cal_noise08' in d else [])
    lengths=[len(d['E_'+k]) for k in qnames];Q=torch.cat([d['E_'+k] for k in qnames]).to(device)
    offsets=np.r_[0,np.cumsum(lengths)];slices={k:slice(int(a),int(b)) for k,a,b in zip(qnames,offsets[:-1],offsets[1:])}
    T=d['T'].to(device);K=len(T);result={};counts={}
    for reg in readouts.HYCAL_LAMBDAS:
        result[reg]={}
        for role in ['base','novel']:
            check_pause();path=out/f'hycal_{role}_{reg:g}.pt'
            if path.exists():value=torch.load(path,map_location='cpu',weights_only=False)
            else:
                E=d['E_fit' if role=='base' else 'E_support'].to(device)
                y=d['fit_y' if role=='base' else 'support_y']
                cosine=torch.empty((len(Q),K));maha=torch.empty_like(cosine);ncs=[]
                for c in range(K):
                    check_pause();x=E[y==c];ncs.append(len(x));state=readouts.hycal_statistics(x,T[c],reg)
                    for a in range(0,len(Q),1024):
                        co,di=readouts.hycal_distances(Q[a:a+1024],T[c],state)
                        cosine[a:a+len(co),c]=co.cpu();maha[a:a+len(di),c]=-di.cpu()
                value=dict(cosine=cosine,negative_distance=maha,counts=np.array(ncs))
                atomic_torch(value,path);print('HYCAL RAW',out,role,reg,flush=True)
            result[reg][role]=value
    del Q;gc.collect()
    def score(reg,key,base,novel):
        a,b=result[reg]['base'],result[reg]['novel'];sl=slices[key]
        co=torch.cat([a['cosine'][sl][:,base],b['cosine'][sl][:,novel]],1).to(device)
        md=torch.cat([a['negative_distance'][sl][:,base],b['negative_distance'][sl][:,novel]],1).to(device)
        return readouts.hycal_scores(co,md,np.r_[a['counts'][base],b['counts'][novel]])
    return score


def make_record(c,tc,reference,w,candidates,tref,closed,common,fixed):
    rows=[dict(**common,policy='fixed_gate',offset=float(fixed),fallback=False,**rv.evaluate(tc,fixed))]
    off,ceiling=rv.threshold(tc,.95*closed)
    rows.append(dict(**common,policy='matched_test',offset=off,fallback=off is None,ceiling=ceiling,
                     **(rv.evaluate(tc,off) if off is not None else {})))
    for policy in ['protection','paired_accuracy']:
        rows.append(dict(**common,**fairness.evaluate_policy(c,reference,w,candidates,tc,tref,closed,policy)))
    return rows


@torch.no_grad()
def run(dataset,encoder,device,parts):
    out=OUT/encoder/dataset;out.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),Path(readouts.__file__),ROOT/'src/experiments/kbs_paired_accuracy.py',Path(fairness.__file__)]
    config=dict(sources={p.name:digest_file(p) for p in sources},
                protocol_sha256=digest_file(ROOT/'experiments/kbs_reviewer_revision_protocol.md'))
    mf=out/'manifest.json'
    if mf.exists():assert json.loads(mf.read_text())==config,'New revision inputs changed.'
    else:atomic_json(config,mf)
    if (out/'complete.json').exists():return
    d,path=load_data(dataset,encoder);score_hycal=prepare_hycal(d,out,device)
    entries=json.loads((ROOT/f'experiments/kbs_revision/{encoder}/{dataset}/partitions.json').read_text())
    Ef,Es,T=[d[k].to(device) for k in ['E_fit','E_support','T']];yf,ys=d['fit_y'],d['support_y']
    for entry in entries[:parts]:
        check_pause();p=entry['seed'];po=out/f'p{p}';po.mkdir(exist_ok=True)
        if (po/'complete.json').exists():continue
        oldroot=ROOT/f'experiments/kbs_independent_risk/{encoder}/{dataset}/p{p}'
        selectpath=oldroot/'selection.json';sel=json.loads(selectpath.read_text());parameters=sel['parameters']
        base,new,pb,pn=[np.array(entry[k]) for k in ['base','novel','pseudo_base','pseudo_novel']]
        W=core.fit_ridge(Ef,yf,base,sel['ridge_lambda']);Pb=core.means(Ef,yf,base)
        qkeys=['tune','cal','test']+(['cal_noise08','test_noise08','cal_blur125','test_blur125'] if dataset=='cifar' else [])
        query={};labels={}
        for key in qkeys:
            y=d[key+'_y'];mask=np.isin(y,base) if key.startswith('cal') or key=='tune' else np.ones(len(y),bool)
            query[key]=d['E_'+key][mask].to(device);labels[key]=y[mask]
        moments=readouts.fixed_moments(query['tune'],T[base],Pb,W)
        ps=po/'selection.json'
        if ps.exists():selected=json.loads(ps.read_text())
        else:
            best=(-1.,None);trials=[];ids=np.r_[pb,pn];ym=d['tune_y'];mask=np.isin(ym,base)
            for reg in readouts.HYCAL_LAMBDAS:
                cc=rv.compact(score_hycal(reg,'tune',pb,pn)[mask],ids,ym[mask],pb,pn)
                for off in np.arange(-1,1.001,.05):
                    h=rv.evaluate(cc,float(off))['harmonic'];trials.append(dict(regularization=reg,offset=float(off),harmonic=h))
                    if h>best[0]+1e-12:best=(h,dict(regularization=reg,offset=float(off)))
            selected=dict(hycal=best[1],trials=trials,fixed_moments=moments,
                          frozen_selection_sha256=digest_file(selectpath));atomic_json(selected,ps)
        reg=selected['hycal']['regularization']
        def ref(key):return prior.predict(core.z(query[key]@T[base].T)+parameters['retained'][0]*core.z(core.linear(query[key],W)),base)==labels[key]
        references={key:ref(key) for key in qkeys};ref_test=references['test'];yt=labels['test']
        closed=float(np.mean([ref_test[yt==c].mean() for c in base]))
        # Per-class diagnostics, fixed reference and original dense comparison.
        rawold=np.load(oldroot/'o0_t5.npz');dense=rawold['test_anchored_tp_base_prediction'];reference_pred=prior.predict(core.z(query['test']@T[base].T)+parameters['retained'][0]*core.z(core.linear(query['test'],W)),base)
        prototype_pred=prior.predict(query['test']@Pb.T,base);diagnostics=[]
        for c in base:
            mask=yt==c;x=Ef[yf==c];idx=int(np.flatnonzero(base==c)[0]);dis=reference_pred[mask]!=dense[mask]
            diagnostics.append(dict(class_id=int(c),fit_count=len(x),reference_accuracy=float(ref_test[mask].mean()),
                 dense_accuracy=float((dense[mask]==c).mean()),prototype_accuracy=float((prototype_pred[mask]==c).mean()),
                 prototype_cohesion=float((x@Pb[idx]).mean()),disagreement_fraction=float(dis.mean()),
                 reference_correct_on_disagreement=float((reference_pred[mask][dis]==c).mean()) if dis.any() else None,
                 dense_correct_on_disagreement=float((dense[mask][dis]==c).mean()) if dis.any() else None))
        atomic_json(dict(encoder=encoder,dataset=dataset,part=p,classes=diagnostics),po/'diagnostics.json')
        for order in [0,1]:
            ordered=np.random.default_rng(1000+p+100*order).permutation(new)
            for session in range(1,6):
                check_pause();dest=po/f'o{order}_t{session}.json'
                if dest.exists():continue
                novel=ordered[:len(new)*session//5];ids=np.r_[base,novel];Pn=core.means(Es[np.array(sel['support_indices'])],ys[np.array(sel['support_indices'])],novel)
                systems={};archive=dict(base_classes=base,novel_classes=novel)
                for key in qkeys:
                    E=query[key];y=labels[key];systems[key]={};archive[key+'_labels']=y;archive[key+'_reference_correct']=references[key]
                    for name,(center,scale,factor) in VARIANTS.items():
                        score=readouts.normalization_scores(E,T[ids],Pb,Pn,W,parameters['retained'],moments,center,scale,factor)
                        systems[key][name]=rv.compact(score,ids,y,base,novel)
                    hy=score_hycal(reg,key,base,novel)
                    if key=='tune' or key.startswith('cal'):hy=hy[np.isin(d[key+'_y'],base)]
                    systems[key]['hycal']=rv.compact(hy,ids,y,base,novel)
                    if dataset=='cifar':
                        a=rv.arrays(E,T[ids],Pb,Pn,W)
                        for m in ['retained','anchored_tp']:systems[key][m]=rv.compact(rv.score(a,m,parameters[m]),ids,y,base,novel)
                    for m,c in systems[key].items():
                        for k in ['margin','base_prediction','novel_prediction']:archive[key+'_'+m+'_'+k]=c[k]
                rows=[];wc=risk.weights(labels['cal'],base)
                for m in [*VARIANTS,'hycal']:
                    c,tc=systems['cal'][m],systems['test'][m]
                    candidates=np.unique(np.quantile(systems['tune'][m]['margin'],np.linspace(0,1,101)))
                    archive[m+'_candidates']=candidates;fixed=selected['hycal']['offset'] if m=='hycal' else parameters['retained'][2]
                    common=dict(encoder=encoder,dataset=dataset,part=p,order=order,session=session,method=m,reference_closed_test=closed)
                    rows.extend(make_record(c,tc,references['cal'],wc,candidates,ref_test,closed,common,fixed))
                # Shift evaluation is prespecified for order0 and three representative readouts.
                if dataset=='cifar' and order==0:
                    for corruption in ['noise08','blur125']:
                        key='test_'+corruption;stref=references[key];sy=labels[key]
                        sclosed=float(np.mean([stref[sy==c].mean() for c in base]))
                        for m in ['retained','anchored_tp','hycal']:
                            cand=np.unique(np.quantile(systems['tune'][m]['margin'],np.linspace(0,1,101)))
                            archive[m+'_candidates']=cand
                            for calibration in ['clean','matched_shift']:
                                ck='cal' if calibration=='clean' else 'cal_'+corruption
                                c=systems[ck][m];w=risk.weights(labels[ck],base)
                                for policy in ['protection','paired_accuracy']:
                                    detail=fairness.evaluate_policy(c,references[ck],w,cand,systems[key][m],stref,sclosed,policy)
                                    rows.append(dict(encoder=encoder,dataset=dataset,part=p,order=order,session=session,
                                                     method=m,corruption=corruption,calibration=calibration,reference_closed_test=sclosed,**detail))
                rawpath=dest.with_suffix('.npz');atomic_npz(rawpath,**archive)
                atomic_json(dict(records=rows,raw_sha256=digest_file(rawpath),frozen_selection_sha256=digest_file(selectpath)),dest)
        atomic_json(dict(complete=True,part=p),po/'complete.json');print('REVIEWER READOUTS COMPLETE',encoder,dataset,p,flush=True)
    if parts==10:atomic_json(dict(complete=True,parts=10),out/'complete.json')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['cifar','pets','imagenetr']);ap.add_argument('--encoder',choices=['l14','b32']);ap.add_argument('--parts',type=int,default=10);args=ap.parse_args()
    device=configure();check_pause();start=time.perf_counter()
    atomic_json(readouts.official_fixtures(ROOT),OUT.parent/'hycal_author_fixture.json')
    for encoder in ([args.encoder] if args.encoder else ['l14','b32']):
        for dataset in ([args.dataset] if args.dataset else ['cifar','pets','imagenetr']):run(dataset,encoder,device,args.parts)
    atomic_json(dict(complete=True,seconds=time.perf_counter()-start),OUT/'complete.json')


if __name__=='__main__':main()
