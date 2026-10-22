"""Atomic checkpoints, cooperative pause and conservative local GPU settings."""
from pathlib import Path
import hashlib,json,os
import torch

ROOT=Path(__file__).resolve().parents[2]
CONTROL=ROOT/'experiments/kbs_strengthening'
_lock_handle=None

def configure():
 global _lock_handle
 CONTROL.mkdir(parents=True,exist_ok=True)
 # The OS releases this lock on crash/reboot. Keep only one GPU job active.
 if _lock_handle is None:
  handle=(CONTROL/'gpu.lock').open('a+b');handle.seek(0)
  if handle.read(1)==b'':handle.write(b'0');handle.flush()
  handle.seek(0)
  try:
   if os.name=='nt':
    import msvcrt
    msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
   else:
    import fcntl
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except OSError:
   handle.close();raise RuntimeError('Another KBS GPU stage is running; wait for it to finish before resuming this stage.')
  _lock_handle=handle
 torch.set_num_threads(int(os.environ.get('KBS_CPU_THREADS','2')))
 if torch.cuda.is_available():
  torch.cuda.set_per_process_memory_fraction(float(os.environ.get('KBS_GPU_FRACTION','.75')))
 return 'cuda' if torch.cuda.is_available() else 'cpu'

def check_pause():
 if (CONTROL/'PAUSE').exists():
  print('PAUSED at a saved checkpoint. Remove experiments/kbs_strengthening/PAUSE and rerun the same command to resume.',flush=True)
  raise SystemExit(0)

def atomic_json(value,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp')
 with tmp.open('w',encoding='utf-8') as f:
  json.dump(value,f,indent=2,ensure_ascii=True);f.flush();os.fsync(f.fileno())
 json.loads(tmp.read_text(encoding='utf-8'));tmp.replace(path)

def atomic_torch(value,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp');torch.save(value,tmp)
 # Reopen before replacing the last good checkpoint.
 torch.load(tmp,map_location='cpu',weights_only=False)
 tmp.replace(path)

def digest_file(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def atomic_npz(path,**arrays):
 import numpy as np
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
 with np.load(tmp) as z:
  for k in z.files:assert z[k].dtype!=object
 tmp.replace(path)
