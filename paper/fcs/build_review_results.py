"""Build review-control tables from separate, source-hashed experiment records."""
from pathlib import Path
import csv, hashlib, json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
labels = {'text_bias': 'Text + group offset', 'anchored_tp': 'Anchored text + prototypes',
          'reference': 'Reference-anchored table', 'dynamic': 'Candidate-normalized table',
          'ranpac_frozen_2000': 'Frozen RP-ridge (2000)'}
datasets = {}
summary = []
def cell(rows, key):
    x = np.array([r[key] for r in rows]) * 100
    return f'{x.mean():.2f} $\\pm$ {x.std(ddof=1):.2f}'

for name, title in [('fcs', 'CIFAR-100'), ('fcs_pets', 'Pets')]:
    folder = ROOT / 'experiments' / name / 'review_controls'
    d = json.loads((folder / 'results.json').read_text())
    assert hashlib.sha256((folder/'source_snapshot.py').read_bytes()).hexdigest() == d['source_sha256']
    assert len(d['controls']) == 6 and len(d['retention']) == 24 and len(d['sequential']) == 30 and len(d['ranpac']) == 3
    datasets[name] = d
    lines = [r'\begin{tabular}{lrrr}', r'\toprule', r'Readout & Base & Novel & All \\', r'\midrule']
    for method in ['text_bias', 'anchored_tp', 'ranpac_frozen_2000']:
        rows = [r for r in d['controls']+d['ranpac'] if r['method'] == method]
        lines.append(labels[method]+' & '+' & '.join(cell(rows,k) for k in ['old','new','overall'])+r' \\')
        for k in ['old','new','overall']:
            x=np.array([r[k] for r in rows])*100
            summary.append([title,method,'harmonic',k,x.mean(),x.std(ddof=1),3])
    lines += [r'\bottomrule', r'\end{tabular}']
    (HERE/f'table_controls_{name}.tex').write_text('\n'.join(lines))

lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Data & Readout & Base & Novel & All & Feasible \\',r'\midrule']
for name,title in [('fcs','CIFAR'),('fcs_pets','Pets')]:
    for method in ['text_bias','anchored_tp','dynamic','reference']:
        rows=[r for r in datasets[name]['retention'] if r['method']==method and r['ratio']==.95]
        feasible=sum(r['validation_feasible'] for r in rows)
        lines.append(title+' & '+labels[method]+' & '+' & '.join(cell(rows,k) for k in ['old','new','overall'])+f' & {feasible}/3'+r' \\')
    for ratio in [.95,.99]:
        for method in ['text_bias','anchored_tp','dynamic','reference']:
            rows=[r for r in datasets[name]['retention'] if r['method']==method and r['ratio']==ratio]
            for k in ['old','new','overall']:
                x=np.array([r[k] for r in rows])*100
                summary.append([title,method,str(ratio),k,x.mean(),x.std(ddof=1),sum(r['validation_feasible'] for r in rows)])
lines += [r'\bottomrule',r'\end{tabular}']
(HERE/'table_retention.tex').write_text('\n'.join(lines))
with (HERE/'review_summary.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['dataset','method','selection','metric','mean_percent','sample_sd_percent','feasible_partitions']);w.writerows(summary)
report={name:{'controls':len(d['controls']),'retention':len(d['retention']),
    'sessions':len(d['sequential']),'rp_readouts':len(d['ranpac']),
    'max_sequential_score_drift':max(r['max_existing_score_drift'] for r in d['sequential']),
    'source_hash_valid':True} for name,d in datasets.items()}
(HERE/'review_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
