from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/kbs_sketch_validation'
def snapshot():
 excluded={OUT/'evaluation_complete.json',OUT/'features_complete.json',OUT/'verification.json',OUT/'restart_verification.json'}
 return {p.relative_to(ROOT).as_posix():[hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns] for p in OUT.rglob('*') if p.is_file() and p not in excluded}
def main():
 before=snapshot();start=time.perf_counter();env=dict(os.environ,USE_TF='0',USE_FLAX='0')
 for name,args in [('kbs_sketch_prepare.py',['--phase','encode']),('kbs_sketch_evaluate.py',[])]:subprocess.run([sys.executable,str(ROOT/'src/experiments'/name),*args],cwd=ROOT,env=env,check=True)
 after=snapshot();assert before==after
 value=dict(complete=True,checkpoint_files=len(before),hash_and_mtime_unchanged=True,aggregate_root_markers_excluded=True,feature_and_evaluation_resume_checked=True,seconds=time.perf_counter()-start)
 (OUT/'restart_verification.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
if __name__=='__main__':main()
