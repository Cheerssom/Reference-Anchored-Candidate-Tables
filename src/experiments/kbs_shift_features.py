"""Locked CIFAR corruption stress test: atomic, local-weights-only feature shards."""
from pathlib import Path
import gc,hashlib,json,sys,time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageFilter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import kbs_calibration_pool as pools
from kbs_runtime import configure,check_pause,atomic_json,atomic_torch,digest_file

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/kbs_reviewer_revision/shift_features'
CORRUPTIONS={'noise08':{'gaussian_noise_sigma':.08},'blur125':{'gaussian_blur_radius':1.25}}


def corrupt(x,name,identity):
    if name=='clean':return x.copy()
    if name=='noise08':
        seed=int(hashlib.sha256(('23102026/'+identity).encode()).hexdigest()[:16],16)
        rng=np.random.default_rng(seed)
        return np.clip(np.rint(x.astype(float)+rng.normal(0,.08*255,x.shape)),0,255).astype(np.uint8)
    im=Image.fromarray(x.transpose(1,2,0)).filter(ImageFilter.GaussianBlur(1.25))
    return np.asarray(im).transpose(2,0,1).copy()


@torch.inference_mode()
def main():
    from lt_dataset import _load_split
    device=configure();check_pause();start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    train,ytrain,_=_load_split('train');test,ytest,_=_load_split('test')
    poolpath=ROOT/'experiments/kbs_calibration_budget/pools/cifar.json';pool=json.loads(poolpath.read_text())
    lookup={pools.pixel_hash(x):i for i,x in enumerate(train)}
    calids=np.array([lookup[r['sha256']] for r in pool['rows']]);caly=ytrain[calids]
    rng=np.random.default_rng(23102026);testids=np.sort(np.concatenate([rng.permutation(np.flatnonzero(ytest==c))[:20] for c in np.unique(ytest)]))
    rows=[('cal',int(i)) for i in calids]+[('test',int(i)) for i in testids]
    labels=np.r_[caly,ytest[testids]]
    config=dict(calibration_pool_sha256=digest_file(poolpath),calibration_raw_ids=calids.tolist(),
                test_raw_ids=testids.tolist(),corruptions=CORRUPTIONS,source_sha256=digest_file(Path(__file__)),
                protocol_sha256=digest_file(ROOT/'experiments/kbs_reviewer_revision_protocol.md'))
    mf=OUT/'manifest.json'
    if mf.exists():assert json.loads(mf.read_text())==config
    else:atomic_json(config,mf)
    for encoder in ['l14','b32']:
        revision=pools.REVISIONS[encoder];name='large-patch14' if encoder=='l14' else 'base-patch32'
        modelpath=Path.home()/f'.cache/huggingface/hub/models--openai--clip-vit-{name}/snapshots'/revision
        from transformers import CLIPModel
        model=CLIPModel.from_pretrained(str(modelpath),local_files_only=True,use_safetensors=encoder=='l14').to(device).eval()
        mean=torch.tensor([.48145466,.4578275,.40821073],device=device)[None,:,None,None]
        std=torch.tensor([.26862954,.26130258,.27577711],device=device)[None,:,None,None]
        def encode(selected,corruption):
            images=[]
            for role,i in selected:images.append(corrupt((train if role=='cal' else test)[i],corruption,f'{role}/{i}'))
            x=torch.from_numpy(np.stack(images)).to(device).float()/255
            x=F.interpolate(x,(224,224),mode='bicubic',align_corners=False)
            with torch.autocast(device,enabled=device=='cuda',dtype=torch.float16):
                z=model.visual_projection(model.vision_model(pixel_values=(x-mean)/std).pooler_output)
            return F.normalize(z.float(),dim=1).cpu()
        # Verify clean preprocessing against the exact cached test-feature space.
        import kbs_strengthen as prior
        if encoder=='l14':d,_,_=prior.load('cifar')
        else:d=torch.load(ROOT/'experiments/kbs_revision/b32_features/cifar/frozen.pt',map_location='cpu',weights_only=False)
        witness=F.cosine_similarity(encode([('test',int(i)) for i in testids[:8]],'clean'),d['E_test'][testids[:8]])
        assert float(witness.min())>.999,witness.tolist();del d;gc.collect()
        for corruption in CORRUPTIONS:
            folder=OUT/encoder/corruption;folder.mkdir(parents=True,exist_ok=True)
            target=folder/'features.pt';vf=folder/'verification.json'
            if target.exists() and vf.exists():
                assert digest_file(target)==json.loads(vf.read_text())['feature_sha256'];continue
            blocks=[];done=0;batch=16
            for p in sorted(folder.glob('batch_*.pt')):
                value=torch.load(p,map_location='cpu',weights_only=False)
                assert value['start']==done and value['manifest_sha256']==digest_file(mf)
                blocks.append(value['features']);done+=len(blocks[-1])
            while done<len(rows):
                check_pause()
                try:z=encode(rows[done:done+batch],corruption)
                except torch.cuda.OutOfMemoryError:
                    gc.collect();torch.cuda.empty_cache()
                    if batch==1:raise
                    batch=max(1,batch//2);continue
                atomic_torch(dict(start=done,features=z,manifest_sha256=digest_file(mf)),folder/f'batch_{done:06d}.pt')
                blocks.append(z);done+=len(z)
                if done%320==0 or done==len(rows):print('SHIFT FEATURES',encoder,corruption,done,'/',len(rows),flush=True)
            atomic_torch(dict(features=torch.cat(blocks),labels=labels,calibration_count=len(calids),test_ids=testids,
                              model_revision=revision,manifest_sha256=digest_file(mf)),target)
            atomic_json(dict(complete=True,feature_sha256=digest_file(target),min_clean_witness_cosine=float(witness.min()),
                             images=len(rows),peak_cuda_bytes=torch.cuda.max_memory_allocated()),vf)
        del model;gc.collect();torch.cuda.empty_cache()
    atomic_json(dict(complete=True,seconds=time.perf_counter()-start),OUT/'complete.json')


if __name__=='__main__':main()
