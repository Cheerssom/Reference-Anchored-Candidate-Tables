"""Build the CIFAR frozen-CLIP cache without any task-adapted checkpoint."""
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from config import CFG
from fcs_pets_data import TEMPLATES

def prepare(path,datasets,device):
    from transformers import CLIPModel,CLIPTokenizer
    model=CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device).eval()
    tokenizer=CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    mean=torch.tensor([.48145466,.4578275,.40821073],device=device)[None,:,None,None]
    std=torch.tensor([.26862954,.26130258,.27577711],device=device)[None,:,None,None]
    data={}
    with torch.no_grad():
        for name,ds in datasets.items():
            ans=[]
            for i in range(0,len(ds),32):
                x=torch.from_numpy(ds.x[i:i+32]).to(device).float()/255
                x=F.interpolate(x,size=(224,224),mode='bicubic',align_corners=False)
                with torch.autocast(device,enabled=device=='cuda',dtype=torch.float16):
                    pooled=model.vision_model(pixel_values=(x-mean)/std).pooler_output
                    e=model.visual_projection(pooled)
                ans.append(F.normalize(e.float(),dim=1).cpu())
                if i%320==0:print(f'CIFAR embedding {name}: {i}/{len(ds)}',flush=True)
            data['E_'+name]=torch.cat(ans)
        with open(CFG.cifar_root/'meta','rb') as f:
            names=pickle.load(f,encoding='latin1')['fine_label_names']
        text=[]
        for template in TEMPLATES:
            b=tokenizer([template.format(s.replace('_',' ')) for s in names],padding=True,return_tensors='pt').to(device)
            pooled=model.text_model(**b).pooler_output
            text.append(F.normalize(model.text_projection(pooled).float(),dim=1).cpu())
        data['T100']=F.normalize(torch.stack(text).mean(0),dim=1)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(data,path)
    del model
    if device=='cuda':torch.cuda.empty_cache()
    return data
