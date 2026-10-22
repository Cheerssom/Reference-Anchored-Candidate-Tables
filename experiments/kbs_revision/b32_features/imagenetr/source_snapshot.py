"""Batch-resumable B/32 extraction on the exact existing image pools."""
from pathlib import Path
import argparse,json,pickle,sys,time,gc
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import CFG
from kbs_runtime import configure,check_pause,atomic_json,atomic_torch,digest_file
from fcs_pets_data import TEMPLATES

ROOT=Path(__file__).resolve().parents[2]
MODEL=Path.home()/'.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
OUT=ROOT/'experiments/kbs_revision/b32_features'


def inputs(dataset):
    if dataset=='cifar':
        from lt_dataset import make_lt_datasets
        from train import split_train_val
        training,support,_,test,meta=make_lt_datasets(rho=100);fit,val=split_train_val(training)
        pools={name:(ds.x,ds.fine) for name,ds in [('fit',fit),('val',val),('support',support),('test',test)]}
        with (CFG.cifar_root/'meta').open('rb') as f:names=pickle.load(f,encoding='latin1')['fine_label_names']
        names=[n.replace('_',' ') for n in names]
        identities={name:{'shape':list(x.shape),'images_sha256':__import__('hashlib').sha256(x.tobytes()).hexdigest(),
                          'labels_sha256':__import__('hashlib').sha256(y.tobytes()).hexdigest()} for name,(x,y) in pools.items()}
    elif dataset=='pets':
        old=torch.load(ROOT/'experiments/fcs_pets/pets_frozen.pt',map_location='cpu',weights_only=False)
        pools={name:(old[name+'_x'],old[name+'_y']) for name in ['fit','val','support','test']}
        names=old['meta']['names'];meta=old['meta']
        identities={name:{'shape':list(x.shape),'images_sha256':__import__('hashlib').sha256(x.tobytes()).hexdigest(),
                          'labels_sha256':__import__('hashlib').sha256(y.tobytes()).hexdigest()} for name,(x,y) in pools.items()}
    else:
        manifest=ROOT/'experiments/kbs_imagenetr/data_manifest.json';saved=json.loads(manifest.read_text())
        names=saved['class_names'];meta={k:v for k,v in saved.items() if k!='pools'}
        pools={name:(rows,np.array([r['class'] for r in rows])) for name,rows in saved['pools'].items()}
        identities={'data_manifest_sha256':digest_file(manifest)}
    return pools,names,meta,identities


def pixels(dataset,rows,start,count,device):
    if dataset=='imagenetr':
        images=[]
        for row in rows[start:start+count]:
            with Image.open(CFG.data_root/'imagenet-r'/row['path']) as im:
                images.append(np.array(im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC),copy=True).transpose(2,0,1))
        x=torch.from_numpy(np.stack(images)).to(device).float()/255
    else:
        x=torch.from_numpy(rows[start:start+count]).to(device).float()/255
        if dataset=='cifar':x=F.interpolate(x,size=(224,224),mode='bicubic',align_corners=False)
    return x


@torch.inference_mode()
def prepare(dataset,device):
    out=OUT/dataset;out.mkdir(parents=True,exist_ok=True);path=out/'frozen.pt'
    pools,names,meta,identities=inputs(dataset)
    model_files=['config.json','pytorch_model.bin','tokenizer_config.json','vocab.json','merges.txt']
    configuration={'dataset':dataset,'model':'openai/clip-vit-base-patch32','revision':MODEL.name,
                   'model_files_sha256':{name:digest_file(MODEL/name) for name in model_files},'inputs':identities,
                   'source_sha256':digest_file(Path(__file__)),'protocol_sha256':digest_file(ROOT/'experiments/kbs_revision_protocol.md')}
    manifest=out/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==configuration,'Immutable feature inputs changed.'
    else:
        atomic_json(configuration,manifest);(out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    if path.exists() and (out/'verification.json').exists():
        print('B32 cache already complete',dataset,flush=True);return
    from transformers import CLIPModel,CLIPTokenizer
    model=CLIPModel.from_pretrained(str(MODEL),use_safetensors=False,local_files_only=True).to(device).eval()
    tokenizer=CLIPTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    mean=torch.tensor([.48145466,.4578275,.40821073],device=device)[None,:,None,None]
    std=torch.tensor([.26862954,.26130258,.27577711],device=device)[None,:,None,None]
    def encode(name,start,count):
        x=pixels(dataset,pools[name][0],start,count,device)
        with torch.autocast(device,enabled=device=='cuda',dtype=torch.float16):
            result=model.visual_projection(model.vision_model(pixel_values=(x-mean)/std).pooler_output)
        return F.normalize(result.float(),dim=1).cpu()
    data={};start_time=time.perf_counter()
    for name,(rows,labels) in pools.items():
        shards=out/('embedding_'+name);shards.mkdir(exist_ok=True);blocks=[];done=0;manifest_hash=digest_file(manifest)
        for shard in sorted(shards.glob('batch_*.pt')):
            saved=torch.load(shard,map_location='cpu',weights_only=False)
            assert saved['start']==done and saved['manifest_sha256']==manifest_hash
            blocks.append(saved['features']);done+=len(blocks[-1])
        batch=16
        while done<len(labels):
            check_pause()
            try:block=encode(name,done,min(batch,len(labels)-done))
            except torch.cuda.OutOfMemoryError:
                gc.collect();torch.cuda.empty_cache()
                if batch==1:raise
                batch=max(1,batch//2);print('B32 OOM batch',batch,flush=True);continue
            atomic_torch({'start':done,'features':block,'manifest_sha256':manifest_hash},shards/f'batch_{done:06d}.pt')
            blocks.append(block);done+=len(block)
            if done%320==0 or done==len(labels):print('B32',dataset,name,done,'/',len(labels),flush=True)
        data['E_'+name]=torch.cat(blocks);data[name+'_y']=labels
    text=[]
    for template in TEMPLATES:
        check_pause();batch=tokenizer([template.format(n) for n in names],padding=True,return_tensors='pt').to(device)
        text.append(F.normalize(model.text_projection(model.text_model(**batch).pooler_output).float(),dim=1).cpu())
    data['T100']=F.normalize(torch.stack(text).mean(0),dim=1);data['meta']=meta
    data['feature_manifest_sha256']=digest_file(manifest)
    atomic_torch(data,path)
    # Re-encode cached test images in a new call, and compare with stored features.
    witness=encode('test',0,8);cos=F.cosine_similarity(witness,data['E_test'][:8])
    assert cos.min()>.999
    atomic_json({'complete':True,'cache_sha256':digest_file(path),'minimum_reembedding_cosine':float(cos.min()),
                 'feature_shapes':{k:list(v.shape) for k,v in data.items() if isinstance(v,torch.Tensor)},
                 'elapsed_seconds_this_invocation':time.perf_counter()-start_time,
                 'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0},out/'verification.json')
    print('B32 COMPLETE',dataset,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--datasets',nargs='+',default=['cifar','pets','imagenetr']);args=ap.parse_args()
    device=configure()
    for ds in args.datasets:
        assert ds in ['cifar','pets','imagenetr'];prepare(ds,device)
