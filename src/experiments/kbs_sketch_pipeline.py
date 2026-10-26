"""Serialized Sketch stages; safe restart reuses existing atomic checkpoints."""
from pathlib import Path
import argparse,subprocess,sys,time
from kbs_runtime import check_pause
ROOT=Path(__file__).resolve().parents[2]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--wait-acquisition',action='store_true');args=ap.parse_args()
 def run(name,*flags):
  check_pause();subprocess.run([sys.executable,str(ROOT/'src/experiments'/name),*flags],cwd=ROOT,check=True)
 if args.wait_acquisition:
  started=time.monotonic()
  while not (ROOT/'experiments/kbs_sketch_acquisition/download_complete.json').exists():
   check_pause()
   if time.monotonic()-started>6*3600:raise TimeoutError('Acquisition incomplete; resume after checking its log.')
   time.sleep(5)
 else:run('kbs_sketch_acquire.py')
 run('kbs_sketch_prepare.py','--phase','split')
 run('kbs_sketch_evaluate.py','--prepare-budget')
 run('kbs_sketch_prepare.py','--phase','encode')
 run('kbs_sketch_evaluate.py')
 run('kbs_verify_sketch_validation.py')
if __name__=='__main__':main()
