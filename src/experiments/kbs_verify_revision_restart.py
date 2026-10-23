"""Real completed-checkpoint reuse test for three locked revision runners."""
from pathlib import Path
import json,subprocess,sys,time
from kbs_runtime import digest_file,atomic_json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/kbs_reviewer_revision'

def main():
    # Top-level timing markers/author fixture are regenerated deliberately.
    paths=[p for folder in ['fairness','readouts','shift_features'] for p in (OUT/folder).rglob('*')
        if p.is_file() and p.suffix in ['.json','.npz','.pt'] and p.name not in ['complete.json']]
    before={p.relative_to(ROOT).as_posix():(digest_file(p),p.stat().st_mtime_ns) for p in paths}
    started=time.perf_counter()
    for script in ['kbs_calibration_fairness.py','kbs_shift_features.py','kbs_reviewer_revision.py']:
        subprocess.run([sys.executable,str(ROOT/'src/experiments'/script)],cwd=ROOT,check=True)
    for p in paths:assert before[p.relative_to(ROOT).as_posix()]==(digest_file(p),p.stat().st_mtime_ns),p
    atomic_json(dict(complete=True,checkpoints_unchanged=len(paths),seconds=time.perf_counter()-started,
        total_revision_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
        marker_files_excluded=['complete.json','hycal_author_fixture.json']),OUT/'restart_verification.json')
    print('RESTART VERIFIED',len(paths),'immutable checkpoint hashes and timestamps')

if __name__=='__main__':main()
