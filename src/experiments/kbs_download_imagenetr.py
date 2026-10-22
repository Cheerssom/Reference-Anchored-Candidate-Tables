"""Resumable ranged download of the authors' public ImageNet-R archive."""
from pathlib import Path
import concurrent.futures,hashlib,json,os,tarfile,time,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import CFG
import requests

ROOT=CFG.data_root
URL='https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
TOTAL=2191079936

def main():
 root=ROOT;target=root/'imagenet-r.tar';prefix=root/'imagenet-r.tar.part'
 destination=root/'imagenet-r'
 if (destination/'download_provenance.json').exists() and len(list(destination.glob('n*/*')))==30000:
  print('ImageNet-R already downloaded and extracted; reusing completed stage.',flush=True);return
 folder=root/'imagenet-r.download';folder.mkdir(exist_ok=True)
 start=time.monotonic()
 if not target.exists():
  initial=prefix.stat().st_size if prefix.exists() else 0
  planpath=folder/'ranges.json'
  if planpath.exists():
   plan=json.loads(planpath.read_text());assert initial==plan['prefix_bytes']
  else:
   step=32*1024*1024
   plan={'source':URL,'total':TOTAL,'prefix_bytes':initial,
         'ranges':[[s,min(s+step,TOTAL)-1] for s in range(initial,TOTAL,step)]}
   planpath.write_text(json.dumps(plan,indent=2))
  def get(pair):
   a,b=pair;p=folder/f'{a}-{b}.part'
   if p.exists() and p.stat().st_size==b-a+1:return p
   for attempt in range(4):
    try:
     r=requests.get(URL,headers={'Range':f'bytes={a}-{b}'},stream=True,timeout=(25,90));r.raise_for_status()
     assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {a}-{b}/')
     with p.open('wb') as f:
      for chunk in r.iter_content(1024*1024):
       if chunk:f.write(chunk)
     assert p.stat().st_size==b-a+1
     print(f'range {a}-{b} complete',flush=True);return p
    except (requests.RequestException,AssertionError):
     if attempt==3:raise
     time.sleep(2*(attempt+1))
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
   paths=list(ex.map(get,plan['ranges']))
  import shutil
  merged=folder/'complete.tar'
  with merged.open('wb') as out:
   if initial:
    with prefix.open('rb') as f:shutil.copyfileobj(f,out)
   for p in paths:
    with p.open('rb') as f:shutil.copyfileobj(f,out)
  assert merged.stat().st_size==TOTAL
  merged.replace(target)
 print('download complete; extracting',flush=True)
 destination=root/'imagenet-r'
 with tarfile.open(target) as t:
  members=t.getmembers();base=root.resolve()
  assert all((base/m.name).resolve().is_relative_to(base) for m in members)
  t.extractall(root,filter='data')
 metadata={'source':URL,'archive_bytes':target.stat().st_size,
  'archive_sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),
  'elapsed_seconds':time.monotonic()-start,'members':len(members)}
 (destination/'download_provenance.json').write_text(json.dumps(metadata,indent=2))
 print(json.dumps(metadata),flush=True)

if __name__=='__main__':main()
