"""CPU-only label-budget experiment on immutable score archives."""
from pathlib import Path
import argparse,json,time
import numpy as np
from kbs_runtime import atomic_json,check_pause,digest_file

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/kbs_validation_budget'
METHODS=('retained','removed_fixed','no_learned_tuned','anchored_tp','tip_sum_calibrated','tip_mean_calibrated')
KEYS=('margin','base_prediction','novel_prediction','base_correct','novel_correct','weights')


def accuracy(c,offset):
    new=(offset>c['margin'])|((offset==c['margin'])&(c['novel_prediction']<c['base_prediction']))
    return float(c['weights'][c['base_correct']&~new].sum()),float(c['weights'][c['novel_correct']&new].sum())


def select_threshold(c,target):
    correct=np.flatnonzero(c['base_correct']);ceiling=float(c['weights'][correct].sum())
    if target>ceiling+1e-10:return None,ceiling
    if target<=0:return float('inf'),ceiling
    ordered=correct[np.argsort(-c['margin'][correct],kind='stable')]
    cumulative=np.cumsum(c['weights'][ordered])
    index=min(int(np.searchsorted(cumulative,target-1e-12)),len(ordered)-1)
    boundary=float(c['margin'][ordered[index]])
    offset=boundary if accuracy(c,boundary)[0]>=target-1e-10 else float(np.nextafter(boundary,-np.inf))
    return offset,ceiling


def select_rows(y,base,fraction,part,seed):
    if fraction==1:return np.arange(len(y))
    chosen=[]
    for c in base:
        indices=np.flatnonzero(y==c);assert len(indices)>0
        permutation=np.random.default_rng(51000+1000*part+100*seed+int(c)).permutation(indices)
        chosen.extend(permutation[:max(1,int(np.floor(fraction*len(indices))))])
    return np.sort(chosen)


def macro_weights(y,base,novel=()):
    w=np.zeros(len(y))
    for group in [base,novel]:
        for c in group:
            n=np.sum(y==c)
            if n:w[y==c]=1/(len(group)*n)
    return w


def archives(dataset,part,session):
    root=ROOT/'experiments/kbs_depth'/dataset
    return [root/f'robust/p{part}_r0_k5/o0_t{session}.npz',root/f'tip/p{part}/o0_t{session}.npz']


def load_archives(paths):
    merged={}
    for path in paths:
        with np.load(path) as z:
            for name in z.files:
                value=z[name]
                if name in merged:assert np.array_equal(merged[name],value),(path,name)
                else:merged[name]=value
    return merged


def compact(z,method,validation=False,indices=None):
    prefix=('validation_' if validation else '')+method+'_'
    result={key:z[prefix+key] for key in KEYS}
    if indices is not None:result={key:arr[indices] for key,arr in result.items()}
    return result


def prepare_manifest(dataset):
    out=OUT/dataset;out.mkdir(parents=True,exist_ok=True)
    inputs=[p for part in range(3) for session in range(1,6) for p in archives(dataset,part,session)]
    inputs += [p.with_suffix('.json') for p in inputs]
    sources=[Path(__file__),ROOT/'src/experiments/kbs_runtime.py',ROOT/'experiments/kbs_validation_budget_protocol.md']
    value={'dataset':dataset,'input_sha256':{p.relative_to(ROOT).as_posix():digest_file(p) for p in inputs},
           'source_sha256':{p.name:digest_file(p) for p in sources}}
    manifest=out/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==value,'Source or archive changed; use a new experiment directory.'
    else:
        atomic_json(value,manifest)
        for p in sources:(out/p.name).write_bytes(p.read_bytes())
    return out


def run_dataset(dataset,limit=None):
    check_pause();out=prepare_manifest(dataset);new_units=0;start=time.perf_counter()
    for part in range(3):
        for session in range(1,6):
            check_pause();paths=archives(dataset,part,session);z=load_archives(paths)
            base,novel,y,vy=[z[key] for key in ['base_classes','novel_classes','labels','validation_labels']]
            assert np.isin(vy,base).all() and not np.isin(vy,novel).any()
            test_weights=macro_weights(y,base,novel)
            ref_test=float(test_weights[z['retained_base_correct']].sum())
            params={r['method']:r['parameters'] for p in paths for r in json.loads(p.with_suffix('.json').read_text())['records'] if r['evaluation']=='fixed_gate'}
            for fraction in [.25,.5,1.]:
                for seed in ([0] if fraction==1 else [0,1,2]):
                    check_pause();path=out/f'p{part}_f{int(fraction*100)}_s{seed}_t{session}.json'
                    if path.exists():continue
                    indices=select_rows(vy,base,fraction,part,seed);selected_y=vy[indices]
                    weights=macro_weights(selected_y,base)
                    ref_val=float(weights[z['validation_retained_base_correct'][indices]].sum())
                    counts={int(c):int(np.sum(selected_y==c)) for c in base};records=[]
                    for method in METHODS:
                        # Selection inputs contain base validation queries only.
                        vc=compact(z,method,True,indices);vc['weights']=weights
                        for ratio in [.95,.99]:
                            target=ratio*ref_val;offset,ceiling=select_threshold(vc,target)
                            row={'dataset':dataset,'part':part,'session':session,'order':0,'fraction':fraction,'subset_seed':seed,
                                 'method':method,'parameters':params[method],'ratio':ratio,'target':target,
                                 'validation_reference':ref_val,'validation_ceiling':ceiling,'feasible':offset is not None,
                                 'offset':offset,'test_reference':ref_test,'test_evaluation_target':ratio*ref_test}
                            if offset is not None:
                                row['validation_base']=accuracy(vc,offset)[0]
                                tc=compact(z,method);tc['weights']=test_weights
                                old,new=accuracy(tc,offset)
                                row.update({'old':old,'new':new,'overall':(len(base)*old+len(novel)*new)/(len(base)+len(novel)),
                                            'test_retention_ratio':old/ref_test,'test_target_met':old>=ratio*ref_test-1e-10})
                            records.append(row)
                    atomic_json({'selection_indices':indices.tolist(),'validation_count':len(indices),
                                 'full_validation_count':len(vy),'actual_fraction':len(indices)/len(vy),
                                 'per_class_selected_counts':counts,'records':records},path)
                    new_units+=1
                    if limit is not None and new_units>=limit:
                        print('STOPPED after saved unit limit',dataset,new_units,flush=True);return
            print('BUDGET SAVED',dataset,part,session,flush=True)
    atomic_json({'complete':True,'units':105,'records':1260,'new_units_this_invocation':new_units,
                 'seconds_this_invocation':time.perf_counter()-start,'device':'cpu','image_encoder_loaded':False},out/'complete.json')
    print('BUDGET COMPLETE',dataset,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--datasets',nargs='+',default=['cifar','pets','imagenetr']);ap.add_argument('--max-new-units',type=int)
    args=ap.parse_args()
    for ds in args.datasets:
        assert ds in ['cifar','pets','imagenetr'];run_dataset(ds,args.max_new_units)
