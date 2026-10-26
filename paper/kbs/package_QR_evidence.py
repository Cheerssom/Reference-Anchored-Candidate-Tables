"""Package the Q/R extension without images, weights or PT feature caches."""
from pathlib import Path
import hashlib,json,zipfile
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
def main():
 files=[]
 for folder in ['kbs_full_fusion_retuning','kbs_sketch_acquisition','kbs_sketch_protocol_draft','kbs_sketch_validation']:
  files += [p for p in (ROOT/'experiments'/folder).rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.md','.py']]
 files += list((ROOT/'src/experiments').glob('kbs_*.py'))
 files += [ROOT/'src/experiments'/n for n in ['fcs_expansion.py','fcs_retention_value.py','test_kbs_full_fusion_retuning.py','test_kbs_sketch_identity.py']]
 files += [ROOT/'scripts'/n for n in ['run_kbs_full_fusion_retuning.ps1','run_kbs_sketch_validation.ps1']]
 files += [ROOT/'experiments'/n for n in ['kbs_full_fusion_retuning_protocol.md','kbs_sketch_prospective_protocol_DRAFT.md','kbs_sketch_execution_protocol_v1.md']]
 files += [H/n for n in ['build_full_fusion_retuning.py','build_sketch_validation.py','validate_submission.py','validation.json','QR_experiments_结论说明.md','README.md']]
 for folder in ['full_fusion_retuning','sketch_validation']:
  files += [p for p in (H/folder).glob('*') if p.is_file() and p.suffix in ['.json','.md','.npz','.pdf']]
 files += list(H.glob('table_full_retuning*.tex'))+list(H.glob('table_sketch*.tex'))+[H/'sketch_operating_frontiers.pdf']
 files=sorted(set(files));assert all(p.exists() for p in files)
 manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 target=H/'candidate_tables_kbs_QR_evidence.zip'
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,p.relative_to(ROOT).as_posix())
  z.writestr('file_hashes.json',json.dumps(manifest,indent=2))
 with zipfile.ZipFile(target) as z:assert z.testzip() is None
 report=dict(complete=True,files=len(files),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),scope='Sections Q/R extension, separate from historical evidence archive',images_weights_and_pt_features_excluded=True)
 (H/'QR_evidence_package.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
