"""Small, identity-audited calibration extensions; no fit/test images admitted."""
from pathlib import Path
import argparse,gc,hashlib,json,sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import CFG
from kbs_runtime import configure,check_pause,atomic_json,atomic_torch,digest_file

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/kbs_calibration_budget'
PROTOCOL=ROOT/'experiments/kbs_calibration_budget_protocol.md'
REVISIONS={'l14':'32bd64288804d66eefd0ccbe215aa642df71cc41',
           'b32':'3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'}


def pixel_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def image_pixels(path):
    with Image.open(path) as im:
        return np.array(im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC),copy=True).transpose(2,0,1)


def identities(dataset):
    """Return current pooled identities and unused raw candidates, without scores."""
    if dataset=='cifar':
        from lt_dataset import make_lt_datasets,_load_split
        from train import split_train_val
        train,support,_,test,_=make_lt_datasets(rho=100);fit,val=split_train_val(train)
        x=np.concatenate([fit.x,val.x]);y=np.r_[fit.fine,val.fine]
        rows=[dict(class_id=int(c),identity='cifar/train/'+pixel_hash(im),sha256=pixel_hash(im),cached_index=i)
              for i,(im,c) in enumerate(zip(x,y))]
        protected=[pixel_hash(im) for array in [support.x,test.x] for im in array]
        raw,labels,_=_load_split('train')
        candidates={int(c):np.flatnonzero(labels==c).tolist() for c in np.unique(labels)}
        def candidate(i):
            sha=pixel_hash(raw[i]);return dict(class_id=int(labels[i]),identity=f'cifar/train-index/{i}',
                                             sha256=sha,raw_index=int(i))
        sources={'train_file_sha256':digest_file(CFG.cifar_root/'train'),
                 'test_file_sha256':digest_file(CFG.cifar_root/'test')}
    elif dataset=='pets':
        from torchvision.datasets import OxfordIIITPet
        old=torch.load(ROOT/'experiments/fcs_pets/pets_frozen.pt',map_location='cpu',weights_only=False)
        train=OxfordIIITPet(root=CFG.data_root/'oxford_pets',split='trainval',target_types='category',download=False)
        ids=np.array(old['train_sample_ids']);yy=np.array(train._labels)[ids]
        rng=np.random.RandomState(1);vm=np.zeros(len(ids),bool)
        for c in np.unique(yy):
            ix=np.flatnonzero(yy==c);rng.shuffle(ix);n=5 if len(ix)>=20 else (2 if len(ix)>=6 else 1);vm[ix[:n]]=True
        combined_ids=np.r_[ids[~vm],ids[vm]];y=np.r_[old['fit_y'],old['val_y']]
        assert np.array_equal(y,np.array(train._labels)[combined_ids])
        rows=[]
        for pool,pool_ids in [('fit',ids[~vm]),('val',ids[vm])]:
            for j,raw_id in enumerate(pool_ids):
                path=Path(train._images[int(raw_id)]);sha=pixel_hash(old[pool+'_x'][j])
                rows.append(dict(class_id=int(old[pool+'_y'][j]),identity=path.name,
                                 sha256=sha,cached_index=len(rows),path=str(path)))
        protected=[pixel_hash(im) for pool in ['support','test'] for im in old[pool+'_x']]
        forbidden_ids=set(ids.tolist())|set(old['support_sample_ids'])
        candidates={int(c):[int(i) for i in np.flatnonzero(np.array(train._labels)==c) if i not in forbidden_ids]
                    for c in np.unique(y)}
        def candidate(i):
            path=Path(train._images[i]);sha=pixel_hash(image_pixels(path))
            return dict(class_id=int(train._labels[i]),identity=path.name,sha256=sha,path=str(path))
        sources={'old_cache_sha256':digest_file(ROOT/'experiments/fcs_pets/pets_frozen.pt')}
        del old;gc.collect()
    else:
        path=ROOT/'experiments/kbs_imagenetr/data_manifest.json';meta=json.loads(path.read_text())
        rows=[]
        for pool in ['fit','val']:
            for r in meta['pools'][pool]:
                rows.append(dict(class_id=r['class'],identity=r['path'],sha256=r['sha256'],
                                 cached_index=len(rows),path=str(CFG.data_root/'imagenet-r'/r['path'])))
        y=np.array([r['class_id'] for r in rows])
        protected=[r['sha256'] for pool in ['support','test'] for r in meta['pools'][pool]]
        used={r['path'] for pool in meta['pools'].values() for r in pool}
        used.update(r['path'] for r in meta['duplicates_removed'])
        candidates={c:[str(f) for f in sorted((CFG.data_root/'imagenet-r'/name).iterdir())
                       if f.is_file() and not f.name.startswith('.') and f.relative_to(CFG.data_root/'imagenet-r').as_posix() not in used]
                    for c,name in enumerate(meta['synsets'])}
        def candidate(f):
            path=Path(f);identity=path.relative_to(CFG.data_root/'imagenet-r').as_posix()
            return dict(class_id=meta['synsets'].index(path.parent.name),identity=identity,
                        sha256=digest_file(path),path=str(path))
        sources={'original_data_manifest_sha256':digest_file(path)}
    return rows,y,protected,candidates,candidate,sources


def build_manifest(dataset):
    target=OUT/'pools'/f'{dataset}.json'
    if target.exists():return json.loads(target.read_text())
    split_path=ROOT/f'experiments/kbs_independent_risk/splits/{dataset}.json'
    split=json.loads(split_path.read_text());rows,y,protected,candidates,candidate,sources=identities(dataset)
    assert pixel_hash(y)==split['label_sha256']
    excluded=set(protected)|{rows[i]['sha256'] for k in ['fit','tune'] for i in split['indices'][k]}
    rng=np.random.default_rng(9032026);chosen=[];seen=set();counts=[]
    cal=split['indices']['cal']
    for c in np.unique(y):
        existing=[rows[i] for i in rng.permutation([i for i in cal if rows[i]['class_id']==c])]
        picks=[]
        for row in existing:
            if row['sha256'] not in excluded|seen:
                picks.append(row);seen.add(row['sha256'])
                if len(picks)==40:break
        if len(picks)<40:
            for j in rng.permutation(len(candidates[int(c)])):
                row=candidate(candidates[int(c)][j])
                if row['sha256'] in excluded|seen:continue
                picks.append(row);seen.add(row['sha256'])
                if len(picks)==40:break
        assert len(picks)>=2,(dataset,int(c),len(picks))
        counts.append(dict(class_id=int(c),available_selected=len(picks),
                           reused=sum('cached_index' in r for r in picks)))
        chosen.extend(picks)
    assert len(chosen)==len({r['sha256'] for r in chosen})
    assert not excluded&{r['sha256'] for r in chosen}
    # Save portable paths: reconstruct actual image roots at runtime.
    for row in chosen:row.pop('path',None)
    result=dict(dataset=dataset,rows=chosen,class_counts=counts,
                protected_hashes=sorted(excluded),sources=sources,
                split_sha256=digest_file(split_path),protocol_sha256=digest_file(PROTOCOL),
                source_sha256=digest_file(Path(__file__)))
    atomic_json(result,target);print('POOL',dataset,len(chosen),'new',sum('cached_index' not in r for r in chosen),flush=True)
    return result


def pixels(dataset,rows):
    if dataset=='cifar':
        from lt_dataset import _load_split
        raw,_,_=_load_split('train')
        return np.stack([raw[r['raw_index']] for r in rows])
    folder=CFG.data_root/('imagenet-r' if dataset=='imagenetr' else 'oxford_pets/oxford-iiit-pet/images')
    return np.stack([image_pixels(folder/r['identity']) for r in rows])


@torch.inference_mode()
def prepare(dataset,encoder,device):
    manifest=build_manifest(dataset);mp=OUT/'pools'/f'{dataset}.json'
    out=OUT/'features'/encoder/dataset;out.mkdir(parents=True,exist_ok=True)
    revision=REVISIONS[encoder];name='large-patch14' if encoder=='l14' else 'base-patch32'
    model_path=Path.home()/f'.cache/huggingface/hub/models--openai--clip-vit-{name}/snapshots'/revision
    weight='model.safetensors' if encoder=='l14' else 'pytorch_model.bin'
    config=dict(pool_sha256=digest_file(mp),source_sha256=digest_file(Path(__file__)),
                model_revision=revision,model_files={n:digest_file(model_path/n) for n in ['config.json',weight]})
    cf=out/'manifest.json'
    if cf.exists():assert json.loads(cf.read_text())==config,'Feature inputs changed.'
    else:atomic_json(config,cf)
    target=out/'extra.pt';verification=out/'verification.json'
    if target.exists() and verification.exists():
        assert json.loads(verification.read_text())['feature_sha256']==digest_file(target);return
    from transformers import CLIPModel
    model=CLIPModel.from_pretrained(str(model_path),local_files_only=True,
                                    use_safetensors=encoder=='l14').to(device).eval()
    mean=torch.tensor([.48145466,.4578275,.40821073],device=device)[None,:,None,None]
    std=torch.tensor([.26862954,.26130258,.27577711],device=device)[None,:,None,None]
    def encode(rows):
        x=torch.from_numpy(pixels(dataset,rows)).to(device).float()/255
        if dataset=='cifar':x=F.interpolate(x,size=(224,224),mode='bicubic',align_corners=False)
        with torch.autocast(device,enabled=device=='cuda',dtype=torch.float16):
            features=model.visual_projection(model.vision_model(pixel_values=(x-mean)/std).pooler_output)
        return F.normalize(features.float(),dim=1).cpu()
    extra=[r for r in manifest['rows'] if 'cached_index' not in r];shards=out/'batches';shards.mkdir(exist_ok=True)
    blocks=[];done=0
    for f in sorted(shards.glob('batch_*.pt')):
        b=torch.load(f,map_location='cpu',weights_only=False);assert b['start']==done and b['manifest_sha256']==digest_file(cf)
        blocks.append(b['features']);done+=len(blocks[-1])
    batch=16
    while done<len(extra):
        check_pause()
        try:block=encode(extra[done:done+batch])
        except torch.cuda.OutOfMemoryError:
            gc.collect();torch.cuda.empty_cache()
            if batch==1:raise
            batch=max(1,batch//2);continue
        atomic_torch(dict(start=done,features=block,manifest_sha256=digest_file(cf)),shards/f'batch_{done:06d}.pt')
        blocks.append(block);done+=len(block)
        if done%160==0 or done==len(extra):print('CAL FEATURES',encoder,dataset,done,'/',len(extra),flush=True)
    # Witness uses CIFAR raw identities too, recovered through a content-hash map.
    witness=[r.copy() for r in manifest['rows'] if 'cached_index' in r][:8]
    if dataset=='cifar':
        from lt_dataset import _load_split
        raw,_,_=_load_split('train');lookup={pixel_hash(x):i for i,x in enumerate(raw)}
        for r in witness:r['raw_index']=lookup[r['sha256']]
    import kbs_strengthen as prior
    if encoder=='l14':d,_,_=prior.load(dataset)
    else:d=torch.load(ROOT/f'experiments/kbs_revision/b32_features/{dataset}/frozen.pt',map_location='cpu',weights_only=False)
    original=torch.cat([d['E_fit'],d['E_val']]);actual=encode(witness)
    cosine=F.cosine_similarity(actual,original[[r['cached_index'] for r in witness]])
    assert float(cosine.min())>.999,cosine.tolist()
    features=torch.cat(blocks) if blocks else torch.empty((0,original.shape[1]))
    atomic_torch(dict(features=features,manifest_sha256=digest_file(cf)),target)
    atomic_json(dict(complete=True,feature_sha256=digest_file(target),extra_images=len(extra),
                     min_witness_cosine=float(cosine.min()),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                     cached_indices=[r['cached_index'] for r in witness]),verification)
    del model,d;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--audit-only',action='store_true');args=ap.parse_args()
    if args.audit_only:
        for dataset in ['cifar','pets','imagenetr']:build_manifest(dataset)
    else:
        device=configure()
        for encoder in ['l14','b32']:
            for dataset in ['cifar','pets','imagenetr']:check_pause();prepare(dataset,encoder,device)
