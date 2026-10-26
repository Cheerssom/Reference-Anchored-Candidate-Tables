"""Outcome-blind identity split and local-only resumable Sketch embeddings."""
from pathlib import Path
import argparse,gc,hashlib,io,json,sys,time
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
from kbs_runtime import atomic_json,digest_file,check_pause
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/kbs_sketch_validation';DATA=ROOT.parent/'data/imagenet-sketch-r200';SOURCE=ROOT.parent/'data/imagenet-r'

def pixels(raw):
 with Image.open(io.BytesIO(raw)) as im:return np.array(im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC),copy=True)
def sha(x):return hashlib.sha256(x).hexdigest()
def phash(x):return sha(np.ascontiguousarray(x.transpose(2,0,1)).tobytes())
def dihedral_hash(x):
 return min(sha(np.ascontiguousarray(v).tobytes()) for k in range(4) for v in [np.rot90(x,k),np.fliplr(np.rot90(x,k))])

def source_hashes():
 OUT.mkdir(exist_ok=True);meta=ROOT/'experiments/kbs_imagenetr/data_manifest.json';pool=ROOT/'experiments/kbs_calibration_budget/pools/imagenetr.json'
 config=dict(manifest_sha256=digest_file(meta),calibration_pool_sha256=digest_file(pool),protocol_sha256=digest_file(ROOT/'experiments/kbs_sketch_execution_protocol_v1.md'),source_sha256=digest_file(Path(__file__)))
 dest=OUT/'source_identity_hashes.json'
 if dest.exists():
  value=json.loads(dest.read_text());assert value['inputs']==config;return value
 m=json.loads(meta.read_text());paths={SOURCE/r['path']:r['sha256'] for rows in m['pools'].values() for r in rows}
 for r in json.loads(pool.read_text())['rows']:paths[SOURCE/r['identity']]=r['sha256']
 rawset=set();pixelset=set()
 for i,(p,expected) in enumerate(sorted(paths.items(),key=lambda x:str(x[0]))):
  check_pause();assert p.resolve().is_relative_to(SOURCE.resolve());raw=p.read_bytes();actual=sha(raw);assert actual==expected
  rawset.add(actual);pixelset.add(phash(pixels(raw)))
  if (i+1)%1000==0:print('SKETCH SOURCE IDENTITIES',i+1,'/',len(paths),flush=True)
 value=dict(inputs=config,source_paths=len(paths),raw_hashes=sorted(rawset),pixel224_hashes=sorted(pixelset));atomic_json(value,dest);return value

def split():
 assert json.loads((ROOT/'experiments/kbs_sketch_acquisition/download_complete.json').read_text())['complete']
 source=source_hashes();source_raw=set(source['raw_hashes']);source_pixel=set(source['pixel224_hashes'])
 mapping=json.loads((ROOT/'experiments/kbs_sketch_protocol_draft/expected_class_mapping.json').read_text())['classes']
 config=dict(protocol_sha256=digest_file(ROOT/'experiments/kbs_sketch_execution_protocol_v1.md'),mapping_sha256=digest_file(ROOT/'experiments/kbs_sketch_protocol_draft/expected_class_mapping.json'),acquisition_sha256=digest_file(ROOT/'experiments/kbs_sketch_acquisition/download_manifest.json'),source_identity_sha256=digest_file(OUT/'source_identity_hashes.json'),source_sha256=digest_file(Path(__file__)))
 dest=OUT/'identity_split.json'
 if dest.exists():value=json.loads(dest.read_text());assert value['inputs']==config;return value
 seen_raw=set();seen_pixel=set();seen_d4=set();retained=[];excluded=[];counts=[];rng=np.random.default_rng(27102026)
 for cl in mapping:
  check_pause();folder=DATA/cl['synset'];receipt=json.loads((folder/'acquisition.json').read_text());unique=[]
  for im in sorted(receipt['images'],key=lambda v:v['name']):
   path=folder/im['name'];raw=path.read_bytes();rawsha=sha(raw);assert rawsha==im['sha256'];x=pixels(raw);ps=phash(x);d4=dihedral_hash(x)
   reason=('source_raw' if rawsha in source_raw else 'source_pixel224' if ps in source_pixel else 'target_raw' if rawsha in seen_raw else 'target_pixel224' if ps in seen_pixel else 'target_dihedral224' if d4 in seen_d4 else None)
   row=dict(path=path.relative_to(DATA).as_posix(),class_id=cl['source_task_class'],synset=cl['synset'],raw_sha256=rawsha,pixel224_sha256=ps,dihedral224_sha256=d4)
   if reason:excluded.append(dict(**row,reason=reason));continue
   seen_raw.add(rawsha);seen_pixel.add(ps);seen_d4.add(d4);unique.append(row)
  n=len(unique);assert n>=2,(cl,n)
  ncal=min(20,max(1,int(.4*n)));perm=rng.permutation(n);cal=set(perm[:ncal].tolist())
  for i,row in enumerate(unique):retained.append(dict(**row,role='cal' if i in cal else 'test'))
  counts.append(dict(class_id=cl['source_task_class'],synset=cl['synset'],acquired=len(receipt['images']),unique=n,calibration=ncal,test=n-ncal))
  print('SKETCH SPLIT',cl['source_task_class'],'unique',n,flush=True)
 value=dict(inputs=config,seed=27102026,records=retained,excluded=excluded,class_counts=counts,scope='Outcome-blind exact-hash/dihedral deduplication; residual near-duplicates and pretraining exposure unknown');atomic_json(value,dest);return value

def encode():
 import torch
 import torch.nn.functional as F
 from kbs_runtime import configure,atomic_torch
 from kbs_calibration_pool import REVISIONS
 from transformers import CLIPModel
 device=configure();value=split();rows=value['records'];manifest=OUT/'identity_split.json';manifest_sha=digest_file(manifest);source_meta=json.loads((ROOT/'experiments/kbs_imagenetr/data_manifest.json').read_text())
 for enc in ['l14','b32']:
  folder=OUT/'features'/enc;folder.mkdir(parents=True,exist_ok=True);dest=folder/'features.pt';receipt=folder/'verification.json'
  if dest.exists() and receipt.exists():assert digest_file(dest)==json.loads(receipt.read_text())['feature_sha256'];continue
  name='large-patch14' if enc=='l14' else 'base-patch32';path=Path.home()/f'.cache/huggingface/hub/models--openai--clip-vit-{name}/snapshots'/REVISIONS[enc]
  model=CLIPModel.from_pretrained(str(path),local_files_only=True,use_safetensors=enc=='l14').to(device).eval()
  mean=torch.tensor([.48145466,.4578275,.40821073],device=device)[None,:,None,None];std=torch.tensor([.26862954,.26130258,.27577711],device=device)[None,:,None,None]
  @torch.inference_mode()
  def batch(paths):
   x=torch.from_numpy(np.stack([pixels(p.read_bytes()).transpose(2,0,1).copy() for p in paths])).to(device).float()/255.
   with torch.autocast(device,enabled=device=='cuda',dtype=torch.float16):z=model.visual_projection(model.vision_model(pixel_values=(x-mean)/std).pooler_output)
   return F.normalize(z.float(),dim=1).cpu()
  cache=ROOT/('experiments/kbs_imagenetr/frozen_imagenetr.pt' if enc=='l14' else 'experiments/kbs_revision/b32_features/imagenetr/frozen.pt')
  old=torch.load(cache,map_location='cpu',weights_only=False);witness=F.cosine_similarity(batch([SOURCE/r['path'] for r in source_meta['pools']['test'][:8]]),old['E_test'][:8]);assert float(witness.min())>.999,witness.tolist();del old;gc.collect()
  blocks=[];done=0;bs=16
  for p in sorted(folder.glob('batch_*.pt')):
   v=torch.load(p,map_location='cpu',weights_only=False);assert v['start']==done and v['manifest_sha256']==manifest_sha;blocks.append(v['features']);done+=len(blocks[-1])
  while done<len(rows):
   check_pause()
   try:z=batch([DATA/r['path'] for r in rows[done:done+bs]])
   except torch.cuda.OutOfMemoryError:
    gc.collect();torch.cuda.empty_cache()
    if bs==1:raise
    bs=max(1,bs//2);continue
   atomic_torch(dict(start=done,features=z,manifest_sha256=manifest_sha),folder/f'batch_{done:06d}.pt');blocks.append(z);done+=len(z)
   if done%320==0 or done==len(rows):print('SKETCH FEATURES',enc,done,'/',len(rows),flush=True)
  atomic_torch(dict(features=torch.cat(blocks),labels=np.array([r['class_id'] for r in rows]),manifest_sha256=manifest_sha,model_revision=REVISIONS[enc]),dest)
  atomic_json(dict(complete=True,feature_sha256=digest_file(dest),manifest_sha256=manifest_sha,min_source_witness_cosine=float(witness.min()),images=len(rows)),receipt)
  del model;gc.collect();torch.cuda.empty_cache()
 atomic_json(dict(complete=True,encoders=2,images=len(rows)),OUT/'features_complete.json')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['source','split','encode','all'],default='all');args=ap.parse_args()
 if args.phase=='source':source_hashes()
 elif args.phase=='split':split()
 elif args.phase=='encode':encode()
 else:source_hashes();split();encode()
if __name__=='__main__':main()
