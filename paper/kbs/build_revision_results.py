"""Ten-partition estimates and conditional paired CIs from the new raw records."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent;ROOT=H.parents[1]
DATA=['cifar','pets','imagenetr'];NAMES={'cifar':'CIFAR-100','pets':'Pets','imagenetr':'ImageNet-R'}
METHODS=['retained','removed_fixed','no_learned_tuned','anchored_tp','raw','fixed_temperature','tip_sum_calibrated','tip_mean_calibrated']
LABELS={'retained':'Retained references','removed_fixed':'Fixed-gate removal','no_learned_tuned':'Recalibrated removal',
        'anchored_tp':'Dense text + prototypes','raw':'Raw-score fusion','fixed_temperature':'Fixed temperatures',
        'tip_sum_calibrated':'Tip-Adapter sum','tip_mean_calibrated':'Tip-Adapter mean'}
MAIN=['retained','no_learned_tuned','anchored_tp','tip_mean_calibrated','raw','fixed_temperature']


def rows_for(encoder,ds):
    folder=ROOT/'experiments/kbs_revision'/encoder/ds
    assert all((folder/f'p{p}/complete.json').exists() for p in range(10))
    return [r for p in range(10) for f in (folder/f'p{p}').glob('o*_t*.json') for r in json.loads(f.read_text())['records']]


def select(rows,method,evaluation,ratio=None,session=5):
    return [r for r in rows if r['method']==method and r['evaluation']==evaluation and (ratio is None or r.get('ratio')==ratio)
            and (session is None or r['session']==session)]


def partition_values(rows,key):
    return {p:float(np.mean([r[key] for r in rows if r['part']==p and r.get('feasible',True)]))*100
            for p in range(10) if any(r['part']==p and r.get('feasible',True) for r in rows)}


def statistic(rows,key):
    values=partition_values(rows,key);a=np.array(list(values.values()))
    return {'mean':float(a.mean()) if len(a) else None,'sd':float(a.std(ddof=1)) if len(a)>1 else None,'n':len(a),'partition_values':values}


def paired(a,b,key):
    aa={(r['part'],r['order'],r['session']):r for r in a if r.get('feasible',True)}
    bb={(r['part'],r['order'],r['session']):r for r in b if r.get('feasible',True)}
    joint=sorted(aa.keys()&bb.keys());values={p:float(np.mean([aa[k][key]-bb[k][key] for k in joint if k[0]==p]))*100 for p in range(10) if any(k[0]==p for k in joint)}
    v=np.array(list(values.values()));n=len(v)
    if n:
        rng=np.random.default_rng(2026);bootstrap=v[rng.integers(0,n,size=(20000,n))].mean(1)
        ci=np.quantile(bootstrap,[.025,.975]).tolist()
    else:ci=[None,None]
    return {'mean':float(v.mean()) if n else None,'sd':float(v.std(ddof=1)) if n>1 else None,'ci95':ci,'n':n,'joint_cells':len(joint),'partition_values':values}


def cell(s):
    if s['mean'] is None:return '--'
    return f"{s['mean']:.2f}"+(f" $\\pm$ {s['sd']:.2f}" if s['sd'] is not None else '')


def ci_cell(s):
    if s['mean'] is None:return '--'
    return f"{s['mean']:.2f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]"


summary={};rawrows={};flat=[]
for encoder in ['l14','b32']:
    summary[encoder]={};rawrows[encoder]={}
    for ds in DATA:
        rows=rows_for(encoder,ds);rawrows[encoder][ds]=rows;info={'methods':{},'gains':{}}
        for m in METHODS:
            one={'average':statistic(select(rows,m,'fixed_gate',session=None),'overall'),
                 'final':statistic(select(rows,m,'fixed_gate'),'overall'),'policies':{}}
            for evaluation in ['matched_test','base_validation']:
                one['policies'][evaluation]={}
                for ratio in [.8,.85,.9,.95,.99]:
                    r=select(rows,m,evaluation,ratio);feasible=[v for v in r if v['feasible']]
                    # Final order-zero cases are the unique final vocabulary per partition.
                    unique=[v for v in feasible if v['order']==0]
                    value={'old':statistic(r,'old'),'new':statistic(r,'new'),'feasible_partitions':len(unique),
                           'test_met_partitions':sum(v['test_target_met'] for v in unique),
                           'retention':statistic(r,'test_retention_ratio')}
                    one['policies'][evaluation][str(ratio)]=value
                    flat.append({'encoder':encoder,'dataset':ds,'method':m,'evaluation':evaluation,'ratio':ratio,
                                 'base_mean':value['old']['mean'],'novel_mean':value['new']['mean'],
                                 'feasible_partitions':len(unique),'test_met_partitions':value['test_met_partitions']})
            info['methods'][m]=one
            info['gains'][m]={'average':paired(select(rows,'retained','fixed_gate',session=None),select(rows,m,'fixed_gate',session=None),'overall'),
                             'final':paired(select(rows,'retained','fixed_gate'),select(rows,m,'fixed_gate'),'overall'),
                             'matched95_novel':paired(select(rows,'retained','matched_test',.95),select(rows,m,'matched_test',.95),'new')}
        summary[encoder][ds]=info
(H/'revision_summary.json').write_text(json.dumps(summary,indent=2))
with (H/'revision_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=flat[0]);w.writeheader();w.writerows(flat)

for encoder in ['l14','b32']:
    for evaluation,short in [('base_validation','validation'),('matched_test','matched')]:
        lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Data & Readout & Base & Novel & Feasible & Test met \\',r'\midrule']
        for ds in DATA:
            for m in (MAIN if encoder=='l14' and short=='validation' else METHODS):
                v=summary[encoder][ds]['methods'][m]['policies'][evaluation]['0.95']
                lines.append(' & '.join([NAMES[ds],LABELS[m],cell(v['old']),cell(v['new']),str(v['feasible_partitions'])+'/10',
                              str(v['test_met_partitions'])+'/'+str(v['feasible_partitions']) if v['feasible_partitions'] else '--'])+r' \\')
            lines.append(r'\midrule')
        lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
        (H/f'table_revision_{short}_{encoder}.tex').write_text('\n'.join(lines)+'\n')

    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Data & Readout & Base & Novel & Feasible & Test met \\',r'\midrule']
    for ds in DATA:
        for m in METHODS:
            v=summary[encoder][ds]['methods'][m]['policies']['base_validation']['0.95']
            lines.append(' & '.join([NAMES[ds],LABELS[m],cell(v['old']),cell(v['new']),str(v['feasible_partitions'])+'/10',
                          str(v['test_met_partitions'])+'/'+str(v['feasible_partitions']) if v['feasible_partitions'] else '--'])+r' \\')
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
    (H/f'table_revision_validation_all_{encoder}.tex').write_text('\n'.join(lines)+'\n')

lines=[r'\begin{tabular}{lllrr}',r'\toprule',r'Encoder & Data & Comparator & Paired avg. gain [95\% CI] & Paired final gain [95\% CI] \\',r'\midrule']
for encoder in ['l14','b32']:
    for ds in DATA:
        for m in ['no_learned_tuned','anchored_tp','tip_mean_calibrated']:
            g=summary[encoder][ds]['gains'][m]
            lines.append(' & '.join(['L/14' if encoder=='l14' else 'B/32',NAMES[ds],LABELS[m],ci_cell(g['average']),ci_cell(g['final'])])+r' \\')
    lines.append(r'\midrule')
lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
(H/'table_revision_paired.tex').write_text('\n'.join(lines)+'\n')

lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Data & Fusion & Fixed-gate avg. & Matched base & Matched novel & Feasible \\',r'\midrule']
for ds in DATA:
    for m in ['retained','raw','fixed_temperature']:
        v=summary['l14'][ds]['methods'][m];r=v['policies']['matched_test']['0.95']
        lines.append(' & '.join([NAMES[ds],LABELS[m],cell(v['average']),cell(r['old']),cell(r['new']),str(r['feasible_partitions'])+'/10'])+r' \\')
    lines.append(r'\midrule')
lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
(H/'table_revision_simple.tex').write_text('\n'.join(lines)+'\n')

plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(7.2,4.7),constrained_layout=True)
for row,encoder in enumerate(['l14','b32']):
    for col,ds in enumerate(DATA):
        ax=axes[row,col]
        for m,color in [('retained','#155e75'),('anchored_tp','#4d7c0f'),('tip_mean_calibrated','#a16207'),('fixed_temperature','#7c3aed')]:
            x=[];y=[];sd=[]
            for ratio in [.8,.85,.9,.95,.99]:
                v=summary[encoder][ds]['methods'][m]['policies']['matched_test'][str(ratio)]
                if v['feasible_partitions']==10:x.append(ratio*100);y.append(v['new']['mean']);sd.append(v['new']['sd'])
            if x:ax.errorbar(x,y,yerr=sd,marker='o',capsize=2,ms=3,color=color,label=LABELS[m])
        ax.set_title(NAMES[ds]+(' (L/14)' if encoder=='l14' else ' (B/32)'));ax.set_xticks([80,85,90,95,99]);ax.grid(alpha=.15)
        if col==0:ax.set_ylabel('Novel accuracy (%)')
        if row==1:ax.set_xlabel('Shared base-retention budget (%)')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='outside upper center',frameon=False)
for ext in ['pdf','png']:fig.savefig(H/f'revision_frontiers.{ext}',dpi=220)
plt.close(fig)

print(json.dumps({e:{ds:{m:summary[e][ds]['gains'][m]['average'] for m in ['no_learned_tuned','anchored_tp','tip_mean_calibrated','raw','fixed_temperature']} for ds in DATA} for e in summary},indent=2))
