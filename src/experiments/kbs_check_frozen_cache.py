"""Re-embed a fixed small sample to check ImageNet-R cache/input agreement."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from config import CFG
from kbs_runtime import configure,atomic_json

def main():
 from transformers import CLIPModel
 device=configure();root=Path(__file__).resolve().parents[2];folder=root/'experiments/kbs_imagenetr'
 cache=torch.load(folder/'frozen_imagenetr.pt',map_location='cpu',weights_only=False)
 data=json.loads((folder/'data_manifest.json').read_text())
 model=CLIPModel.from_pretrained('openai/clip-vit-large-patch14',local_files_only=True).to(device).eval()
 selected=[];images=[];saved=[]
 for name in ['fit','val','support','test']:
  rows=data['pools'][name]
  for i in np.random.default_rng(20261022).choice(len(rows),2,replace=False):
   row=rows[int(i)]
   import hashlib
   assert hashlib.sha256((CFG.data_root/'imagenet-r'/row['path']).read_bytes()).hexdigest()==row['sha256']
   with Image.open(CFG.data_root/'imagenet-r'/row['path']) as im:
    images.append(np.array(im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC),copy=True).transpose(2,0,1))
   selected.append({'pool':name,'row':int(i),'path':row['path']});saved.append(cache['E_'+name][int(i)])
 mean=torch.tensor([.48145466,.4578275,.40821073],device=device)[None,:,None,None]
 std=torch.tensor([.26862954,.26130258,.27577711],device=device)[None,:,None,None]
 x=torch.from_numpy(np.stack(images)).to(device).float()/255.
 with torch.inference_mode(),torch.autocast(device,enabled=device=='cuda',dtype=torch.float16):
  emb=model.visual_projection(model.vision_model(pixel_values=(x-mean)/std).pooler_output)
 actual=F.normalize(emb.float(),dim=1).cpu();cos=(actual*torch.stack(saved)).sum(1)
 assert float(cos.min())>.9999
 report={'reembedded_images':len(selected),'minimum_cosine_agreement':float(cos.min()),'samples':selected,
         'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0}
 atomic_json(report,folder/'reembedding_validation.json');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
