"""Build the robustness, dense-baseline and mechanism evidence without selection."""
from pathlib import Path
import json,csv,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent;ROOT=H.parents[1]
sys.path.insert(0,str(H))
import build_strengthening_results as main


def summarize(values):
    a=np.array(list(values),float)
    return {'mean':float(a.mean()) if len(a) else None,'sd':float(a.std(ddof=1)) if len(a)>1 else None,'n':len(a)}


def cell(s):
    return '--' if s['mean'] is None else f"{s['mean']:.2f}"+(f" $\\pm$ {s['sd']:.2f}" if s['sd'] is not None else '')


def group(rows,key):
    return {p:float(np.mean([r[key] for r in rows if r['part']==p]))*100 for p in range(3) if any(r['part']==p for r in rows)}


def fixed(rows,m,k,final=False):
    return [r for r in rows if r['method']==m and r['shots']==k and r['evaluation']=='fixed_gate' and (not final or r['session']==5)]


summary={};flat=[];all_drift={}
for ds in main.DATASETS:
    root=ROOT/'experiments/kbs_depth'/ds
    assert len(list((root/'robust').glob('*/complete.json')))==27
    rows=[r for path in (root/'robust').glob('*/o*_t*.json') for r in json.loads(path.read_text())['records']]
    summary[ds]={}
    for k in [1,2,5]:
        info={'readouts':{},'paired_gain':{}}
        for m in ['retained','removed_fixed','no_learned_tuned','anchored_tp','dynamic']:
            avg=group(fixed(rows,m,k),'overall');final=group(fixed(rows,m,k,True),'overall')
            matched=[r for r in rows if r['method']==m and r['shots']==k and r['session']==5 and r['order']==0 and r['evaluation']=='matched_test' and r['ratio']==.95 and r['feasible']]
            operational=[r for r in rows if r['method']==m and r['shots']==k and r['session']==5 and r['order']==0 and r['evaluation']=='base_validation' and r['ratio']==.95 and r['feasible']]
            info['readouts'][m]={'average':summarize(avg.values()),'final':summarize(final.values()),
                'matched95_base':summarize(group(matched,'old').values()),'matched95_novel':summarize(group(matched,'new').values()),
                'matched95_feasible_trials':len(matched),'validation95_base':summarize(group(operational,'old').values()),
                'validation95_novel':summarize(group(operational,'new').values()),'validation95_feasible_trials':len(operational)}
            for metric,reference,comparator in [('average',group(fixed(rows,'retained',k),'overall'),avg),('final',group(fixed(rows,'retained',k,True),'overall'),final)]:
                info['paired_gain'].setdefault(m,{})[metric]=summarize([reference[p]-comparator[p] for p in reference.keys()&comparator.keys()])
            # Match trial IDs before averaging a constrained comparison, not just partitions.
            reference={(r['part'],r['trial']):r for r in rows if r['method']=='retained' and r['shots']==k and r['session']==5 and r['order']==0 and r['evaluation']=='matched_test' and r['ratio']==.95 and r['feasible']}
            paired=[{**r,'gain':100*(reference[(r['part'],r['trial'])]['new']-r['new'])} for r in matched if (r['part'],r['trial']) in reference]
            info['paired_gain'][m]['matched95_novel']=summarize([np.mean([r['gain'] for r in paired if r['part']==p]) for p in range(3) if any(r['part']==p for r in paired)])
            info['paired_gain'][m]['matched95_joint_trials']=len(paired)
            flat.append({'dataset':ds,'shots':k,'method':m,**{metric:info['readouts'][m][metric]['mean'] for metric in ['average','final','matched95_base','matched95_novel']},'matched95_feasible_trials':len(matched)})
        summary[ds][str(k)]=info
    drift=[r for p in (root/'robust').glob('*/drift.json') for r in json.loads(p.read_text())]
    all_drift[ds]={m:{'max_score_drift':max(r['maximum_score_drift'] for r in drift if r['method']==m),
                          'max_restricted_prediction_changes':max(r['restricted_prediction_changes'] for r in drift if r['method']==m),
                          'append_checks':sum(r['method']==m for r in drift)} for m in ['retained','dynamic']}

(H/'depth_summary.json').write_text(json.dumps({'robustness':summary,'score_drift':all_drift},indent=2))
with (H/'depth_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=flat[0]);w.writeheader();w.writerows(flat)

lines=[r'\begin{tabular}{llrrrrr}',r'\toprule',r'Data & Shots & Retained avg. & $\Delta$ recalibrated & $\Delta$ dense & $\Delta$ dynamic & Retained final \\',r'\midrule']
for ds in main.DATASETS:
    for k in [1,2,5]:
        s=summary[ds][str(k)]
        cells=[main.NAMES[ds],str(k),cell(s['readouts']['retained']['average'])]
        cells += [cell(s['paired_gain'][m]['average']) for m in ['no_learned_tuned','anchored_tp','dynamic']]
        cells += [cell(s['readouts']['retained']['final'])]
        lines.append(' & '.join(cells)+r' \\')
    lines.append(r'\midrule')
lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
(H/'table_depth_robustness.tex').write_text('\n'.join(lines)+'\n')

# Full readout comparisons retain recent baselines and separately selected native forms.
lines=[r'\begin{tabular}{llrrr}',r'\toprule',r'Data & Readout & Incremental avg. & Final & Paired avg. gain \\',r'\midrule']
for ds in main.DATASETS:
    for method in main.ALL:
        s=main.summary[ds];v=s['methods'][method]
        lines.append(' & '.join([main.NAMES[ds],main.LABELS[method],cell(v['incremental_average']),cell(v['final']),cell(s['gains'][method]['incremental_average'])])+r' \\')
    lines.append(r'\midrule')
lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
(H/'table_depth_all_readouts.tex').write_text('\n'.join(lines)+'\n')

lines=[r'\begin{tabular}{lllrrr}',r'\toprule',r'Data & Budget & Readout & Base & Novel & Feasible \\',r'\midrule']
for ds in main.DATASETS:
    for ratio in [.95,.99]:
        for method in ['fenec_calibrated','fenec_euclidean_calibrated']:
            rows=main.select(ds,method,'matched_test',ratio)
            lines.append(' & '.join([main.NAMES[ds],f'{ratio*100:.0f}\\%',main.LABELS[method],main.cell(rows,'old'),main.cell(rows,'new'),str(main.stat(rows,'old')['n'])+'/3'])+r' \\')
    lines.append(r'\midrule')
lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
(H/'table_depth_fenec.tex').write_text('\n'.join(lines)+'\n')

# One table per dataset keeps all shot/readout operating policies readable.
for ds in main.DATASETS:
    lines=[r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Shots & Readout & Match base & Match novel & Feasible & Val. base & Val. novel & Feasible \\',r'\midrule']
    for k in [1,2,5]:
        for method in ['retained','no_learned_tuned','anchored_tp','dynamic']:
            v=summary[ds][str(k)]['readouts'][method]
            cells=[str(k),main.LABELS.get(method,'Dynamic reference'),cell(v['matched95_base']),cell(v['matched95_novel']),str(v['matched95_feasible_trials'])+'/9',cell(v['validation95_base']),cell(v['validation95_novel']),str(v['validation95_feasible_trials'])+'/9']
            lines.append(' & '.join(cells)+r' \\')
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
    (H/f'table_depth_policy_{ds}.tex').write_text('\n'.join(lines)+'\n')

cost=json.loads((ROOT/'experiments/kbs_depth/cost.json').read_text())
lines=[r'\begin{tabular}{llrrr}',r'\toprule',r'Data & Readout & Median (ms) & P90 (ms) & Stored tensors (MiB) \\',r'\midrule']
for ds in main.DATASETS:
    for m,v in cost['results'][ds]['readouts'].items():
        label={'retained':'Retained classifier','anchored_tp':'Dense text + prototypes','tip_sum':'Tip-Adapter sum','tip_mean':'Tip-Adapter mean'}[m]
        lines.append(f"{main.NAMES[ds]} & {label} & {v['median_ms']:.3f} & {v['p90_ms']:.3f} & {v['persistent_tensor_bytes']/2**20:.3f}"+r' \\')
    lines.append(r'\midrule')
lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
(H/'table_depth_cost.tex').write_text('\n'.join(lines)+'\n')

ir=main.summary['imagenetr'];cf=main.summary['cifar']
paragraph=(
    'Dense semantic caches supply a direct alternative to prototype summaries. '
    f"At the CIFAR 95\\% budget, the retained table gives {cell(cf['methods']['retained']['matched']['0.95']['new'])}\\% novel accuracy, "
    f"compared with {cell(cf['methods']['tip_mean_calibrated']['matched']['0.95']['new'])}\\% for class-mean Tip-Adapter. "
    f"On ImageNet-R, the corresponding values are {cell(ir['methods']['retained']['matched']['0.95']['new'])}\\% and "
    f"{cell(ir['methods']['tip_mean_calibrated']['matched']['0.95']['new'])}\\%, with all three partitions feasible for both readouts. "
    f"Across its five arrivals, retaining the classifier adds {cell(ir['gains']['tip_mean_calibrated']['incremental_average'])} points in average incremental accuracy "
    f"and {cell(ir['gains']['tip_mean_calibrated']['final'])} points in final accuracy over this dense semantic-cache control.\n\n"
    'We further evaluate 81 configurations spanning three datasets, three class partitions, '
    'one/two/five shots and three predeclared pseudo-split/support trials. Orders and trials are averaged within each class partition. '
    f"At five shots, the paired average gain over recalibrated removal is {cell(summary['imagenetr']['5']['paired_gain']['no_learned_tuned']['average'])} points on ImageNet-R, "
    f"{cell(summary['cifar']['5']['paired_gain']['no_learned_tuned']['average'])} on CIFAR and "
    f"{cell(summary['pets']['5']['paired_gain']['no_learned_tuned']['average'])} on Pets. "
    'The support-budget and readout rankings vary across operating conditions; Supplementary Section G gives every configuration family and both threshold policies. '
    'Across 648 append checks, fixed references preserve all restricted pre-offset candidate rankings, while dynamic references change rankings on each dataset. '
    'These checks isolate the reference mechanism separately from its accuracy trade-offs.\n'
)
(H/'depth_results.tex').write_text(paragraph)

plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(7.2,2.7),constrained_layout=True)
for ax,ds in zip(axes,main.DATASETS):
    for m,color,label in [('no_learned_tuned','#c2410c','Recalibrated removal'),('anchored_tp','#4d7c0f','Dense prototypes'),('dynamic','#7c3aed','Dynamic reference')]:
        values=[summary[ds][str(k)]['paired_gain'][m]['average'] for k in [1,2,5]]
        ax.errorbar([1,2,5],[v['mean'] for v in values],yerr=[v['sd'] for v in values],color=color,marker='o',capsize=2,label=label)
    ax.axhline(0,color='#777',lw=.7);ax.set_title(main.NAMES[ds]);ax.set_xticks([1,2,5]);ax.set_xlabel('Support images / novel class');ax.grid(alpha=.15)
axes[0].set_ylabel('Retained average gain (pp)');handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=3,loc='outside upper center',frameon=False,fontsize=8)
for ext in ['pdf','png']:fig.savefig(H/f'depth_robustness.{ext}',dpi=220)
plt.close(fig)

print(json.dumps({'robust_five_shot':{ds:summary[ds]['5']['paired_gain']['no_learned_tuned']['average'] for ds in main.DATASETS},'score_drift':all_drift},indent=2))
