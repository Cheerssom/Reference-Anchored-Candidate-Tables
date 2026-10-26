from pathlib import Path
import hashlib,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/kbs_full_fusion_retuning'
def records():
 paths=list(OUT.glob('*/*/*.json'))+list(OUT.glob('*/*/p*/*.json'))+list(OUT.glob('*/*/p*/*.npz'))
 return {p.relative_to(ROOT).as_posix():dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),mtime_ns=p.stat().st_mtime_ns) for p in paths}
def main():
 before=records();start=time.perf_counter();subprocess.run([sys.executable,str(ROOT/'src/experiments/kbs_full_fusion_retuning.py')],check=True,cwd=ROOT);after=records();assert before==after
 value=dict(complete=True,checkpoint_files=len(before),hash_and_mtime_unchanged=True,aggregate_root_marker_excluded=True,seconds=time.perf_counter()-start)
 (OUT/'restart_verification.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
if __name__=='__main__':main()
