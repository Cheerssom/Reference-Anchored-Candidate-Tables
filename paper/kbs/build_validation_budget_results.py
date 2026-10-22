"""Summarize label-budget policies at the original partition statistical level."""
from pathlib import Path
import csv,json
import numpy as np

H=Path(__file__).resolve().parent;ROOT=H.parents[1]
NAMES={'cifar':'CIFAR-100','pets':'Pets','imagenetr':'ImageNet-R'}
LABELS={'retained':'Retained classifier','removed_fixed':'Fixed-gate removal','no_learned_tuned':'Recalibrated removal',
        'anchored_tp':'Dense text + prototypes','tip_sum_calibrated':'Tip-Adapter sum','tip_mean_calibrated':'Tip-Adapter mean'}


def statistic(rows,key):
    values=[np.mean([r[key] for r in rows if r['part']==p])*100 for p in range(3) if any(r['part']==p for r in rows)]
    return {'mean':float(np.mean(values)) if values else None,'sd':float(np.std(values,ddof=1)) if len(values)>1 else None,'n':len(values)}


def cell(value):
    if value['mean'] is None:return '--'
    return f"{value['mean']:.2f}"+(f" $\\pm$ {value['sd']:.2f}" if value['sd'] is not None else '')


summary={};flat=[]
for ds in NAMES:
    folder=ROOT/'experiments/kbs_validation_budget'/ds
    assert json.loads((folder/'complete.json').read_text())['complete']
    units=[json.loads(p.read_text()) for p in folder.glob('p*_t5.json')]
    summary[ds]={}
    for fraction in [.25,.5,1.]:
        selected=[u for u in units if u['records'][0]['fraction']==fraction]
        info={'validation_count_range':[min(u['validation_count'] for u in selected),max(u['validation_count'] for u in selected)],
              'actual_fraction_range':[min(u['actual_fraction'] for u in selected),max(u['actual_fraction'] for u in selected)],'policies':{}}
        total=9 if fraction<1 else 3
        for ratio in [.95,.99]:
            info['policies'][str(ratio)]={}
            for method in LABELS:
                rows=[r for u in selected for r in u['records'] if r['method']==method and r['ratio']==ratio and r['feasible']]
                result={'old':statistic(rows,'old'),'new':statistic(rows,'new'),'retention':statistic(rows,'test_retention_ratio'),
                        'feasible':len(rows),'possible':total,'test_target_met':sum(r['test_target_met'] for r in rows)}
                info['policies'][str(ratio)][method]=result
                flat.append({'dataset':ds,'nominal_fraction':fraction,'ratio':ratio,'method':method,
                             'validation_count_min':info['validation_count_range'][0],'validation_count_max':info['validation_count_range'][1],
                             'base_mean':result['old']['mean'],'base_sd':result['old']['sd'],'novel_mean':result['new']['mean'],'novel_sd':result['new']['sd'],
                             'test_retention_percent':result['retention']['mean'],'feasible':result['feasible'],'possible':total,
                             'test_target_met':result['test_target_met'],'available_partitions':result['old']['n']})
        summary[ds][str(fraction)]=info
(H/'validation_budget_summary.json').write_text(json.dumps(summary,indent=2))
with (H/'validation_budget_summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=flat[0]);writer.writeheader();writer.writerows(flat)

lines=[r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Data & Nominal budget & Queries & Target & Base test & Novel test & Val. feasible & Test met \\',r'\midrule']
for ds in NAMES:
    for fraction in [.25,.5,1.]:
        info=summary[ds][str(fraction)];low,high=info['validation_count_range'];count=str(low) if low==high else f'{low}--{high}'
        for ratio in [.95,.99]:
            v=info['policies'][str(ratio)]['retained']
            line=[NAMES[ds],f'{fraction*100:.0f}\\%',count,f'{ratio*100:.0f}\\%',cell(v['old']),cell(v['new']),f"{v['feasible']}/{v['possible']}",f"{v['test_target_met']}/{v['feasible']}" if v['feasible'] else '--']
            lines.append(' & '.join(line)+r' \\')
    lines.append(r'\midrule')
lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
(H/'table_validation_budget_retained.tex').write_text('\n'.join(lines)+'\n')

lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Data & Readout & Base test & Novel test & Val. feasible & Test met \\',r'\midrule']
for ds in NAMES:
    for method,label in LABELS.items():
        v=summary[ds]['0.25']['policies']['0.95'][method]
        lines.append(' & '.join([NAMES[ds],label,cell(v['old']),cell(v['new']),f"{v['feasible']}/9",f"{v['test_target_met']}/{v['feasible']}" if v['feasible'] else '--'])+r' \\')
    lines.append(r'\midrule')
lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
(H/'table_validation_budget_comparators.tex').write_text('\n'.join(lines)+'\n')

low=summary['imagenetr']['0.25']['policies']['0.95']['retained'];full=summary['imagenetr']['1.0']['policies']['0.95']['retained']
paragraph=(
    'We also reduce the base-validation queries used for threshold selection while keeping every fitted readout and fusion setting fixed. '
    f"On ImageNet-R, the nominal 25\\% budget uses 696--731 queries and gives {cell(low['old'])}\\% base and {cell(low['new'])}\\% novel test accuracy, "
    f"compared with {cell(full['old'])}\\% and {cell(full['new'])}\\% using all 2,938--3,087 base queries. "
    'Validation feasibility and achieved test retention are measured separately; Supplementary Section H reports both targets, '
    'subset variation, actual query counts and all readouts.\n'
)
(H/'validation_budget_results.tex').write_text(paragraph)
print(json.dumps({ds:summary[ds]['0.25']['policies']['0.95']['retained'] for ds in NAMES},indent=2))
