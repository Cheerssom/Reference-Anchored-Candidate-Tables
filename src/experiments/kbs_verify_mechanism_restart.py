"""Verify real reuse of completed mechanism diagnostic checkpoints."""
from pathlib import Path
import json,subprocess,sys,time
from kbs_runtime import digest_file,atomic_json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/kbs_mechanism_followup'
paths=list(OUT.glob('*/*/p*/o*.json'))+list(OUT.glob('*/*/p*/o*.npz'))+list(OUT.glob('*/*/manifest.json'))
assert len(paths)==246
before={str(p.relative_to(ROOT)):(digest_file(p),p.stat().st_mtime_ns) for p in paths}
initial=json.loads((OUT/'complete.json').read_text())
history=OUT/'history/first_completed_run.json'
if not history.exists():atomic_json(initial,history)
start=time.perf_counter()
subprocess.run([sys.executable,str(ROOT/'src/experiments/kbs_mechanism_followup.py')],cwd=ROOT,check=True)
for p in paths:assert before[str(p.relative_to(ROOT))]==(digest_file(p),p.stat().st_mtime_ns)
atomic_json(dict(complete=True,immutable_checkpoints=len(paths),sha_and_mtime_unchanged=True,
    first_completed_run=json.loads(history.read_text()),restart_seconds=time.perf_counter()-start,
    scope='unchanged-input reuse, not an input-mutation rejection test'),OUT/'restart_verification.json')
print('MECHANISM RESTART VERIFIED',len(paths),'checkpoint hashes and timestamps')
