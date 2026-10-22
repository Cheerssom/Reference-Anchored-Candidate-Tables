"""Disjoint ImageNet-R pools and frozen CLIP cache for the locked KBS extension."""
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import requests,torch
import torch.nn.functional as F
from PIL import Image
from config import CFG
from fcs_pets_data import TEMPLATES
from kbs_runtime import atomic_json,atomic_torch,check_pause,configure

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/kbs_imagenetr'
CLASS_INDEX='https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'

def pools(root):
 out=OUT;out.mkdir(exist_ok=True)
 mapping_path=out/'imagenet_class_index.json'
 if not mapping_path.exists():
  r=requests.get(CLASS_INDEX,timeout=30);r.raise_for_status();mapping_path.write_bytes(r.content)
 mapping={v[0]:v[1].replace('_',' ') for v in json.loads(mapping_path.read_text()).values()}
 dirs=sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith('n'))
 assert len(dirs)==200,len(dirs)
 samples=[];seen=set();duplicates=[];counts=[]
 split_rows={k:[] for k in ['fit','val','support','test']}
 for c,folder in enumerate(dirs):
  unique=[]
  for path in sorted(folder.iterdir()):
   if not path.is_file() or path.name.startswith('.'):continue
   digest=hashlib.sha256(path.read_bytes()).hexdigest()
   if digest in seen:duplicates.append({'path':path.relative_to(root).as_posix(),'sha256':digest});continue
   seen.add(digest)
   row={'path':path.relative_to(root).as_posix(),'class':c,'sha256':digest}
   unique.append(row);samples.append(row)
  order=np.random.default_rng(2026+c).permutation(len(unique));n=max(5,len(unique)//5)
  assert len(order)-2*n-5>=2,(c,len(unique))
  desired=max(2,int(np.rint(200*100**(-c/199))))
  for name,ix in [('test',order[:n]),('val',order[n:2*n]),('support',order[2*n:2*n+5]),
                 ('fit',order[2*n+5:2*n+5+desired])]:
   split_rows[name].extend(unique[int(i)] for i in ix)
  counts.append(min(desired,len(order)-2*n-5))
 all_hash=[r['sha256'] for rows in split_rows.values() for r in rows]
 assert len(all_hash)==len(set(all_hash))
 meta={'dataset':'ImageNet-R','classes':200,'source':'https://github.com/hendrycks/imagenet-r',
  'class_index_source':CLASS_INDEX,'class_names':[mapping[p.name] for p in dirs],
  'synsets':[p.name for p in dirs],'desired_counts':[max(2,int(np.rint(200*100**(-c/199)))) for c in range(200)],
  'fit_counts':counts,'raw_unique_images':len(samples),'duplicates_removed':duplicates,
  'pools':split_rows,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 atomic_json(meta,out/'data_manifest.json')
 return meta

def prepare(root,device):
 OUT.mkdir(exist_ok=True);path=OUT/'frozen_imagenetr.pt'
 if path.exists():return torch.load(path,map_location='cpu',weights_only=False)
 from transformers import CLIPModel,CLIPTokenizer
 meta=pools(root)
 model=CLIPModel.from_pretrained('openai/clip-vit-large-patch14',local_files_only=True).to(device).eval()
 tok=CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14',local_files_only=True)
 mean=torch.tensor([.48145466,.4578275,.40821073],device=device)[None,:,None,None]
 std=torch.tensor([.26862954,.26130258,.27577711],device=device)[None,:,None,None]
 data={};start=time.monotonic()
 with torch.inference_mode():
  for name,rows in meta['pools'].items():
   checkpoint=OUT/('embedding_'+name);checkpoint.mkdir(exist_ok=True);features=[];done=0
   for shard in sorted(checkpoint.glob('batch_*.pt')):
    partial=torch.load(shard,map_location='cpu',weights_only=False)
    assert partial['data_manifest_sha256']==hashlib.sha256((OUT/'data_manifest.json').read_bytes()).hexdigest()
    assert partial['start']==done
    features.append(partial['features']);done+=len(features[-1])
   i=done;batch_size=16
   while i<len(rows):
    check_pause()
    images=[]
    for row in rows[i:i+batch_size]:
     with Image.open(root/row['path']) as im:
      images.append(np.array(im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC),copy=True).transpose(2,0,1))
    x=torch.from_numpy(np.stack(images)).to(device).float()/255.
    try:
     with torch.autocast(device,enabled=device=='cuda',dtype=torch.float16):
      e=model.visual_projection(model.vision_model(pixel_values=(x-mean)/std).pooler_output)
    except torch.cuda.OutOfMemoryError:
     del x
     torch.cuda.empty_cache()
     if batch_size<=1:raise
     batch_size=max(1,batch_size//2)
     print('Reducing embedding batch to',batch_size,flush=True);continue
    block=F.normalize(e.float(),dim=1).cpu();features.append(block)
    atomic_torch({'features':block,'start':i,'data_manifest_sha256':hashlib.sha256((OUT/'data_manifest.json').read_bytes()).hexdigest()},checkpoint/f'batch_{i:06d}.pt')
    i+=len(images)
    if i%160==0 or i==len(rows):
     print('ImageNet-R embedding',name,i,'/',len(rows),'batch',batch_size,'elapsed',round(time.monotonic()-start),flush=True)
   data['E_'+name]=torch.cat(features);data[name+'_y']=np.array([r['class'] for r in rows])
   assert len(data['E_'+name])==len(rows)
  text=[]
  for template in TEMPLATES:
   batch=tok([template.format(n) for n in meta['class_names']],padding=True,return_tensors='pt').to(device)
   text.append(F.normalize(model.text_projection(model.text_model(**batch).pooler_output).float(),dim=1).cpu())
  data['T100']=F.normalize(torch.stack(text).mean(0),dim=1)
 data['meta']={k:v for k,v in meta.items() if k!='pools'}
 data['data_manifest_sha256']=hashlib.sha256((OUT/'data_manifest.json').read_bytes()).hexdigest()
 data['elapsed_seconds']=time.monotonic()-start
 data['peak_cuda_memory_bytes']=torch.cuda.max_memory_allocated() if device=='cuda' else 0
 atomic_torch(data,path)
 print('CACHE COMPLETE',[(k,list(v.shape)) for k,v in data.items() if isinstance(v,torch.Tensor)],flush=True)
 return data

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=CFG.data_root/'imagenet-r');args=ap.parse_args()
 prepare(args.root,configure())
