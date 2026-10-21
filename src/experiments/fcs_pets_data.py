"""Public Oxford-IIIT Pets extension; shares the locked CIFAR scoring grids."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import OxfordIIITPet
from transformers import CLIPModel, CLIPTokenizer
from lt_dataset import IndexedCifar,bucket_of
from config import CFG

TEMPLATES=['a photo of a {}.','a bad photo of a {}.','a cropped photo of a {}.',
 'a close-up photo of a {}.','a bright photo of a {}.','a good photo of a {}.',
 'a photo of the {}.','a photo of a {}.']

def prepare(out,device):
    from train import split_train_val
    out.mkdir(parents=True,exist_ok=True)
    path=out/'pets_frozen.pt'
    if path.exists():
        data=torch.load(path,map_location='cpu',weights_only=False)
        fit=IndexedCifar(data['fit_x'],data['fit_y'],data['fit_y'])
        val=IndexedCifar(data['val_x'],data['val_y'],data['val_y'])
        support=IndexedCifar(data['support_x'],data['support_y'],data['support_y'])
        test=IndexedCifar(data['test_x'],data['test_y'],data['test_y'])
        return data,fit,val,support,test,data['meta'],path
    root=CFG.data_root/'oxford_pets'
    trainset=OxfordIIITPet(root=root,split='trainval',target_types='category',download=True)
    testset=OxfordIIITPet(root=root,split='test',target_types='category',download=True)
    names=[s.replace('_',' ').lower() for s in trainset.classes]
    y=np.asarray(trainset._labels)
    rng=np.random.default_rng(0)
    counts=np.round(90*10**(-np.arange(37)/36)).astype(int)
    fitpool,support_ids=[],[]
    for c in range(37):
        idx=rng.permutation(np.flatnonzero(y==c))
        support_ids.extend(idx[:5]);fitpool.extend(idx[5:5+counts[c]])
    fitpool=np.asarray(fitpool);rng.shuffle(fitpool)
    def load_images(ds,ids):
        xx=[]
        for i in ids:
            with Image.open(ds._images[int(i)]) as im:
                xx.append(np.asarray(im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC)).transpose(2,0,1))
        return np.stack(xx)
    support_ids=np.asarray(support_ids)
    xx=load_images(trainset,fitpool)
    training=IndexedCifar(xx,y[fitpool],y[fitpool])
    fit,val=split_train_val(training)
    support=IndexedCifar(load_images(trainset,support_ids),y[support_ids],y[support_ids])
    test_y=np.asarray(testset._labels)
    test=IndexedCifar(load_images(testset,np.arange(len(testset))),test_y,test_y)
    model=CLIPModel.from_pretrained('openai/clip-vit-large-patch14',local_files_only=True).to(device).eval()
    tok=CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14',local_files_only=True)
    mean=torch.tensor([.48145466,.4578275,.40821073],device=device)[None,:,None,None]
    std=torch.tensor([.26862954,.26130258,.27577711],device=device)[None,:,None,None]
    @torch.no_grad()
    def embed(ds,label):
        ans=[]
        for i in range(0,len(ds),32):
            x=torch.from_numpy(ds.x[i:i+32]).to(device).float()/255
            with torch.autocast(device,enabled=device=='cuda',dtype=torch.float16):
                v=model.vision_model(pixel_values=(x-mean)/std)
                e=model.visual_projection(v.pooler_output)
            ans.append(F.normalize(e.float(),dim=1).cpu())
            if i%320==0:print(f'Pets embedding {label}: {i}/{len(ds)}',flush=True)
        return torch.cat(ans)
    data={}
    for label,ds in [('fit',fit),('val',val),('support',support),('test',test)]:
        data['E_'+label]=embed(ds,label)
        data[label+'_x']=ds.x;data[label+'_y']=ds.fine
    anchors=[]
    with torch.no_grad():
        for template in TEMPLATES:
            batch=tok([template.format(n) for n in names],padding=True,return_tensors='pt').to(device)
            h=model.text_model(**batch).pooler_output
            anchors.append(F.normalize(model.text_projection(h).float(),dim=1).cpu())
    data['T100']=F.normalize(torch.stack(anchors).mean(0),dim=1)
    data['meta']={'cls_num_list':counts.tolist(),'buckets':[bucket_of(n) for n in counts],
                  'rho':10,'names':names,'dataset':'Oxford-IIIT Pets'}
    data['train_sample_ids']=fitpool.tolist();data['support_sample_ids']=support_ids.tolist()
    torch.save(data,path)
    (out/'data_manifest.json').write_text(json.dumps({'dataset':'Oxford-IIIT Pets',
       'class_names':names,'trainval_ids':fitpool.tolist(),'support_ids':support_ids.tolist(),
       'class_counts':counts.tolist(),'fit_size':len(fit),'val_size':len(val),
       'test_size':len(test),'source':'https://www.robots.ox.ac.uk/~vgg/data/pets/'},indent=2))
    del model
    if device=='cuda':torch.cuda.empty_cache()
    return data,fit,val,support,test,data['meta'],path
