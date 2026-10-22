"""Rebuild a concrete submission bundle and clean-extracted source checks."""
from pathlib import Path
import argparse,hashlib,json,subprocess,tempfile,zipfile
import fitz

H=Path(__file__).resolve().parent


def clean_compile():
    report={}
    for archive,targets in [('candidate_tables_kbs_source.zip',['main','supplementary']),
                            ('candidate_tables_kbs_supplement_source.zip',['supplementary'])]:
        with tempfile.TemporaryDirectory(prefix='kbs-source-check-') as tmp:
            root=Path(tmp)
            with zipfile.ZipFile(H/archive) as z:
                assert z.testzip() is None
                assert all('/' not in name and '\\' not in name for name in z.namelist())
                z.extractall(root)
                hashes=json.loads(z.read('file_hashes.json'))
                for name,sha in hashes.items():
                    if (root/name).exists():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
            info={}
            for name in targets:
                result=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',name+'.tex'],cwd=root,
                                      stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
                assert result.returncode==0,result.stdout[-4000:]
                log=(root/(name+'.log')).read_text(errors='replace')
                assert not any(value in log for value in ['Overfull','undefined','LaTeX Warning:','Package natbib Warning:','! '])
                with fitz.open(root/(name+'.pdf')) as pdf:info[name]=len(pdf)
            report[archive]=info
    (H/'package_validation.json').write_text(json.dumps(report,indent=2))
    return report


def bundle():
    names=['main.pdf','supplementary.pdf','Highlights.docx','cover_letter.docx','title_page.docx','declarations.docx',
           'candidate_tables_kbs_source.zip','candidate_tables_kbs_supplement_source.zip']
    report=json.loads((H/'validation.json').read_text())
    combined=fitz.open()
    for name in ['main.pdf','supplementary.pdf']:
        with fitz.open(H/name) as document:combined.insert_pdf(document)
    combined.set_toc([[1,'Main manuscript',1],[1,'Supplementary material A-J',report['documents']['main']['pages']+1]])
    combined.save(H/'review_complete.pdf',garbage=4,deflate=True);combined.close()
    names+=['review_complete.pdf']
    instruction=(f"Prepared for Knowledge-Based Systems. Main: {report['documents']['main']['pages']} review-format pages; "
                 f"supplement: {report['documents']['supplementary']['pages']} pages; abstract: {report['abstract_words']} words.\n"
                 "No journal submission has been performed. Use the portal's current instructions for upload fields, "
                 "article length and required affiliation/mailing details. Direct access to the journal-specific guide "
                 "was unavailable; see submission_check.md in the project.\n"
                 "The fresh-agent numeric audit could not start due to its model usage limit; no review PASS is claimed. "
                 "Separate local archive reconstruction, numerical tests and clean source compilation are recorded.\n"
                 "The source archive contains editable LaTeX sources; Highlights and letter are editable DOCX files.\n")
    hashes={name:hashlib.sha256((H/name).read_bytes()).hexdigest() for name in names}
    target=H/'candidate_tables_kbs_submission.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(H/name,name)
        z.writestr('UPLOAD_INSTRUCTIONS.txt',instruction)
        z.writestr('file_hashes.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name,sha in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha
    return {'submission_bundle_bytes':target.stat().st_size,'files':len(names),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--skip-clean-compile',action='store_true');args=ap.parse_args()
    if not args.skip_clean_compile:print(json.dumps(clean_compile(),indent=2))
    print(json.dumps(bundle(),indent=2))
