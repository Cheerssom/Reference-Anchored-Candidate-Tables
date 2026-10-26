"""Author-linked, pinned, CRC-checked partial ZIP acquisition for200 classes."""
from pathlib import Path
import concurrent.futures,hashlib,json,shutil,struct,sys,time,zlib
import requests
sys.path.insert(0,str(Path(__file__).resolve().parent))
from kbs_runtime import atomic_json,check_pause,digest_file
ROOT=Path(__file__).resolve().parents[2];META=ROOT/'experiments/kbs_sketch_acquisition'
DATA=ROOT.parent/'data/imagenet-sketch-r200'

def main():
 index=json.loads((META/'remote_zip_index.json').read_text());mapping=json.loads((ROOT/'experiments/kbs_sketch_protocol_draft/expected_class_mapping.json').read_text())
 wanted={x['synset'] for x in mapping['classes']};entries=sorted(index['entries'],key=lambda x:x['offset']);plans=[]
 for syn in sorted(wanted):
  selected=[x for x in entries if x['name'].startswith('sketch/'+syn+'/') and not x['name'].endswith('/')]
  assert selected and all(len(x['name'].split('/'))==3 for x in selected)
  a=min(x['offset'] for x in selected);last=max(x['offset'] for x in selected)
  end=min(x['offset'] for x in entries if x['offset']>last)
  assert all(x['name'].startswith('sketch/'+syn+'/') for x in entries if a<=x['offset']<end)
  plans.append(dict(synset=syn,start=a,end=end-1,entries=selected))
 assert shutil.disk_usage(DATA.parent).free>10*1024**3
 DATA.mkdir(exist_ok=True);shards=DATA/'ranges';shards.mkdir(exist_ok=True)
 config=dict(source_url=index['url'],archive_size=index['archive_size'],index_sha256=digest_file(META/'remote_zip_index.json'),protocol_sha256=digest_file(ROOT/'experiments/kbs_sketch_execution_protocol_v1.md'),classes=200,images=sum(len(x['entries']) for x in plans),range_bytes=sum(x['end']-x['start']+1 for x in plans),source_sha256=digest_file(Path(__file__)))
 mf=META/'download_manifest.json'
 if mf.exists():assert json.loads(mf.read_text())==config
 else:atomic_json(config,mf)
 def acquire(plan):
  check_pause();syn=plan['synset'];folder=DATA/syn;folder.mkdir(exist_ok=True);receipt=folder/'acquisition.json'
  if receipt.exists():
   old=json.loads(receipt.read_text());assert old['manifest_sha256']==digest_file(mf)
   assert all(digest_file(folder/x['name'])==x['sha256'] for x in old['images']);return
  a,b=plan['start'],plan['end'];dest=shards/(syn+'.range');tmp=shards/(syn+'.partial')
  if not dest.exists():
   for attempt in range(4):
    try:
     done=tmp.stat().st_size if tmp.exists() else 0
     if done==b-a+1:tmp.replace(dest);break
     start=a+done;url=index['url']+'?download=true&range_start='+str(start)
     with requests.get(url,headers={'Range':f'bytes={start}-{b}'},stream=True,timeout=(20,60)) as response:
      assert response.status_code==206,response.status_code
      assert response.headers.get('content-range')==f"bytes {start}-{b}/{index['archive_size']}"
      with tmp.open('ab') as f:
       for block in response.iter_content(1024*1024):
        check_pause()
        if block:f.write(block)
     assert tmp.stat().st_size==b-a+1;tmp.replace(dest);break
    except (requests.RequestException,AssertionError):
     if attempt==3:raise
     time.sleep(attempt+1)
  assert dest.stat().st_size==b-a+1
  blob=dest.read_bytes();images=[]
  for e in plan['entries']:
   offset=e['offset']-a;header=struct.unpack_from('<4s5H3L2H',blob,offset);assert header[0]==b'PK\x03\x04';start=offset+30+header[-2]+header[-1]
   compressed=blob[start:start+e['compressed']]
   raw=zlib.decompress(compressed,-15) if e['compression']==8 else compressed
   assert len(raw)==e['size'] and zlib.crc32(raw)==e['crc']
   name=e['name'].split('/')[-1];path=folder/name;assert path.resolve().is_relative_to(DATA.resolve())
   temp=path.with_suffix(path.suffix+'.tmp');temp.write_bytes(raw);temp.replace(path)
   images.append(dict(name=name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),crc32=e['crc']))
  atomic_json(dict(synset=syn,manifest_sha256=digest_file(mf),range_sha256=digest_file(dest),images=images),receipt)
  print('SKETCH ACQUIRED',syn,len(images),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:list(executor.map(acquire,plans))
 atomic_json(dict(complete=True,classes=200,images=config['images'],manifest_sha256=digest_file(mf)),META/'download_complete.json')
if __name__=='__main__':main()
