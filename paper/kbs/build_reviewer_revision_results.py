"""Claim-linked tables/plots for the five reviewer-requested evidence blocks."""
from pathlib import Path
from collections import defaultdict
import csv,json
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
sys.path.insert(0,str(ROOT/'src/experiments'));sys.path.insert(0,str(ROOT/'src'))
import fcs_retention_value as rv
import kbs_calibration_budget as budget

OUT=ROOT/'experiments/kbs_reviewer_revision'
DATA={'cifar':'CIFAR-100','pets':'Pets','imagenetr':'ImageNet-R'}
NAMES={'retained':'Retained','anchored_tp':'Dense prototypes','no_learned_tuned':'Recalibrated removal',
       'raw':'Raw fusion','fixed_temperature':'Fixed temperature','tip_mean_calibrated':'Class-mean cache','hycal':'HyCal transfer'}
VARIANTS=['retained','query_center_fixed_scale','fixed_center_query_scale','fixed_center_fixed_scale']
LABELS={'retained':'Query / query','query_center_fixed_scale':'Query / fixed','fixed_center_query_scale':'Fixed / query','fixed_center_fixed_scale':'Fixed / fixed',
        'classifier_half':'Half learned weight','classifier_zero':'Zero learned weight','hycal':'HyCal transfer'}
RNG=np.random.default_rng(2026);BOOT=RNG.integers(0,10,(20000,10))


def stats(values):
    a=np.asarray(values,float);return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)) if len(a)>1 else 0.,n=len(a),values=a.tolist())


def paired(a,b):
    delta=100*(np.asarray(a)-np.asarray(b));lo,hi=np.quantile(delta[BOOT].mean(1),[.025,.975])
    return dict(mean=float(delta.mean()),ci95=[float(lo),float(hi)],n=10,values=delta.tolist())


def mean_sd(a):return f"{100*a['mean']:.2f}"+r'\,\ensuremath{\pm}\,'+f"{100*a['sd']:.2f}"


def write_table(name,lines):
    (H/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    verified=json.loads((OUT/'verification.json').read_text());assert verified['complete']
    records=[];diagnostics=[]
    for folder in ['fairness','readouts']:
        for p in (OUT/folder).glob('*/*/p*/o*_t*.json'):records.extend(json.loads(p.read_text())['records'])
    for p in (OUT/'readouts').glob('*/*/p*/diagnostics.json'):
        value=json.loads(p.read_text());diagnostics.extend([dict(encoder=value['encoder'],dataset=value['dataset'],part=value['part'],**r) for r in value['classes']])
    # Supply original retained fixed-gate/matched counterparts from the frozen archives.
    for e in ['l14','b32']:
        for d in DATA:
            for p in range(10):
                old=ROOT/f'experiments/kbs_independent_risk/{e}/{d}/p{p}'
                params=json.loads((old/'selection.json').read_text())['parameters']
                closed=json.loads((old/'complete.json').read_text())['reference_closed_test']
                for o in [0,1]:
                    for t in range(1,6):
                        with np.load(old/f'o{o}_t{t}.npz') as z:
                            c=budget.compact(z,'retained',z['test_labels'],z['base_classes'],z['novel_classes'],'test_')
                            common=dict(encoder=e,dataset=d,part=p,order=o,session=t,method='retained')
                            records.append(dict(**common,policy='fixed_gate',fallback=False,**rv.evaluate(c,params['retained'][2])))
                            off,ceiling=rv.threshold(c,.95*closed)
                            records.append(dict(**common,policy='matched_test',fallback=off is None,ceiling=ceiling,
                                                **(rv.evaluate(c,off) if off is not None else {})))
    clean=[r for r in records if 'corruption' not in r]
    policy=[];bykey={}
    for e in ['l14','b32']:
        for d in DATA:
            methods=sorted({r['method'] for r in clean if r['encoder']==e and r['dataset']==d})
            for m in methods:
                for pol in ['protection','paired_accuracy']:
                    a=sorted([r for r in clean if (r['encoder'],r['dataset'],r['method'],r['policy'],r['order'],r['session'])==(e,d,m,pol,0,5)],key=lambda r:r['part'])
                    assert len(a)==10,(e,d,m,pol,len(a));finite=[r for r in a if not r['fallback']]
                    row=dict(encoder=e,dataset=d,method=m,policy=pol,finite=len(finite),test_met=sum(r['test_target_met'] for r in a),
                             admitted_test_met=sum(r['test_target_met'] for r in finite),novel=stats([r['new'] for r in a]),
                             base=stats([r['old'] for r in a]),conditional_novel=stats([r['new'] for r in finite]) if finite else None,
                             conditional_base=stats([r['old'] for r in finite]) if finite else None,
                             destroyed=stats([r['destroyed_test'] for r in a]),recovered=stats([r['recovered_test'] for r in a]))
                    policy.append(row);bykey[e,d,m,pol]=row
    normalization=[];aia={};capacity={};contrasts=[]
    for e in ['l14','b32']:
        for d in DATA:
            for m in VARIANTS+['classifier_half','classifier_zero','hycal']:
                rows=[r for r in clean if (r['encoder'],r['dataset'],r['method'])==(e,d,m)]
                av=[np.mean([r['overall'] for r in rows if r['part']==p and r['policy']=='fixed_gate']) for p in range(10)]
                aia[e,d,m]=av
                cap=[r for r in rows if r['policy']=='matched_test' and r['session']==5 and not r['fallback']]
                pvals={p:[r['new'] for r in cap if r['part']==p] for p in range(10)}
                vals=[np.mean(v) for v in pvals.values() if len(v)==2];capacity[e,d,m]=dict(feasible=len(vals),novel=stats(vals) if vals else None)
                normalization.append(dict(encoder=e,dataset=d,method=m,average=stats(av),matched95=capacity[e,d,m]))
            for name,a,b in [('query_scale_given_query_center','retained','query_center_fixed_scale'),
                             ('query_scale_given_fixed_center','fixed_center_query_scale','fixed_center_fixed_scale'),
                             ('query_center_given_query_scale','retained','fixed_center_query_scale'),
                             ('half_learned_weight','retained','classifier_half'),('zero_learned_weight','retained','classifier_zero')]:
                contrasts.append(dict(encoder=e,dataset=d,contrast=name,**paired(aia[e,d,a],aia[e,d,b])))
    mechanism=[]
    for e in ['l14','b32']:
        for d in DATA:
            rows=[r for r in diagnostics if r['encoder']==e and r['dataset']==d];per=defaultdict(list)
            for r in rows:per[r['class_id']].append(r)
            classes=[]
            for c,rr in per.items():
                classes.append(dict(class_id=c,**{k:float(np.mean([r[k] for r in rr])) for k in ['fit_count','reference_accuracy','dense_accuracy','prototype_accuracy','prototype_cohesion']}))
            classes.sort(key=lambda r:(-r['fit_count'],r['class_id']));groups=[]
            for name,indices in zip(['head','middle','tail'],np.array_split(np.arange(len(classes)),3)):
                rr=[classes[i] for i in indices];groups.append(dict(group=name,classes=len(rr),
                    fit_count_mean=float(np.mean([r['fit_count'] for r in rr])),reference_minus_dense_pp=float(np.mean([100*(r['reference_accuracy']-r['dense_accuracy']) for r in rr]))))
            gain=np.array([r['reference_accuracy']-r['dense_accuracy'] for r in classes])
            mechanism.append(dict(encoder=e,dataset=d,classes=classes,frequency_groups=groups,
                reference_minus_dense_closed_pp=float(100*gain.mean()),
                fit_count_gain_spearman=float(spearmanr([r['fit_count'] for r in classes],gain).statistic),
                prototype_quality_gain_spearman=float(spearmanr([r['prototype_accuracy'] for r in classes],gain).statistic),
                cohesion_gain_spearman=float(spearmanr([r['prototype_cohesion'] for r in classes],gain).statistic)))
    comparisons=[]
    for e in ['l14','b32']:
        for d in DATA:
            for m in ['anchored_tp','fixed_temperature','hycal']:
                a=bykey[e,d,'retained','paired_accuracy'];b=bykey[e,d,m,'paired_accuracy']
                rr=[r for r in clean if (r['encoder'],r['dataset'],r['policy'],r['order'],r['session'])==(e,d,'paired_accuracy',0,5) and r['method'] in ['retained',m]]
                aa={r['part']:r for r in rr if r['method']=='retained'};bb={r['part']:r for r in rr if r['method']==m}
                joint=[p for p in range(10) if not aa[p]['fallback'] and not bb[p]['fallback']]
                comparisons.append(dict(encoder=e,dataset=d,comparator=m,all_partition_gain=paired(a['novel']['values'],b['novel']['values']),
                    jointly_admitted=len(joint),conditional_gain_pp=float(np.mean([100*(aa[p]['new']-bb[p]['new']) for p in joint])) if joint else None))
    shifts=[]
    for e in ['l14','b32']:
        for corruption in ['noise08','blur125']:
            for cal in ['clean','matched_shift']:
                for m in ['retained','anchored_tp','hycal']:
                    for pol in ['protection','paired_accuracy']:
                        rr=[r for r in records if (r['encoder'],r.get('corruption'),r.get('calibration'),r['method'],r['policy'])==(e,corruption,cal,m,pol)]
                        assert len(rr)==50
                        final=sorted([r for r in rr if r['session']==5],key=lambda r:r['part']);finite=[r for r in final if not r['fallback']]
                        shifts.append(dict(encoder=e,corruption=corruption,calibration=cal,method=m,policy=pol,finite=len(finite),
                             test_met=sum(r['test_target_met'] for r in final),novel=stats([r['new'] for r in final]),base=stats([r['old'] for r in final]),
                             conditional_novel=stats([r['new'] for r in finite]) if finite else None,
                             all_sessions_test_met=sum(all(r['test_target_met'] for r in rr if r['part']==p) for p in range(10)),
                             all_sessions_admit=sum(all(not r['fallback'] for r in rr if r['part']==p) for p in range(10))))
    summary=dict(verification=verified,policy=policy,normalization=normalization,normalization_contrasts=contrasts,
                 mechanism=mechanism,comparisons=comparisons,shifts=shifts)
    (H/'reviewer_revision_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    flat=[]
    for r in policy:
        flat.append({k:v for k,v in r.items() if not isinstance(v,dict)}|dict(novel_mean=r['novel']['mean'],novel_sd=r['novel']['sd'],base_mean=r['base']['mean'],
                    conditional_novel_mean=r['conditional_novel']['mean'] if r['conditional_novel'] else None))
    with (H/'reviewer_revision_policy.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in flat))));w.writeheader();w.writerows(flat)
    # Frozen-gate factorial table in the main manuscript.
    lines=[r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',r'Center / scale & CIFAR-100 & Pets & ImageNet-R \\',r'\midrule']
    for e,enc in [('l14','ViT-L/14'),('b32','ViT-B/32')]:
        lines.append(r'\multicolumn{4}{@{}l}{\textit{'+enc+r'}} \\')
        for m in VARIANTS:
            lines.append(' & '.join([LABELS[m]]+[mean_sd(stats(aia[e,d,m])) for d in DATA])+r' \\')
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}');write_table('table_normalization_factorial.tex',lines)
    # Full fairness tables, one dataset/encoder per table, both procedures included.
    for e in ['l14','b32']:
        for d in DATA:
            lines=[r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',r'Readout & Test & Finite & Met & Novel & Given admission & Base \\',r'\midrule']
            for m,name in NAMES.items():
                for pol,label in [('protection','Protect'),('paired_accuracy','Paired')]:
                    r=bykey[e,d,m,pol];cond='--' if r['conditional_novel'] is None else f"{100*r['conditional_novel']['mean']:.2f}"
                    lines.append(' & '.join([name if pol=='protection' else '',label,f"{r['finite']}/10",f"{r['test_met']}/10",f"{100*r['novel']['mean']:.2f}",cond,f"{100*r['base']['mean']:.2f}"])+r' \\')
                lines.append(r'\addlinespace[2pt]')
            lines+=[r'\bottomrule',r'\end{tabular}'];write_table(f'table_fairness_{e}_{d}.tex',lines)
    lines=[r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',r'Encoder & Data & Retained avg. & HyCal avg. & Retained $N_{.95}$ & HyCal $N_{.95}$ & HyCal feasible \\',r'\midrule']
    for e in ['l14','b32']:
        for d in DATA:
            a=capacity[e,d,'retained'];b=capacity[e,d,'hycal'];n='--' if b['novel'] is None else f"{100*b['novel']['mean']:.2f}"
            lines.append(' & '.join([e,DATA[d],f"{100*np.mean(aia[e,d,'retained']):.2f}",f"{100*np.mean(aia[e,d,'hycal']):.2f}",f"{100*a['novel']['mean']:.2f}" if a['novel'] else '--',n,f"{b['feasible']}/10"])+r' \\')
    lines+=[r'\bottomrule',r'\end{tabular}'];write_table('table_hycal_transfer.tex',lines)
    lines=[r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',r'Encoder & Data & Closed gap & Count $\rho$ & Prototype $\rho$ & Cohesion $\rho$ & Classes \\',r'\midrule']
    for r in mechanism:
        lines.append(' & '.join([r['encoder'],DATA[r['dataset']],f"{r['reference_minus_dense_closed_pp']:+.2f}",f"{r['fit_count_gain_spearman']:.2f}",f"{r['prototype_quality_gain_spearman']:.2f}",f"{r['cohesion_gain_spearman']:.2f}",str(len(r['classes']))])+r' \\')
    lines+=[r'\bottomrule',r'\end{tabular}'];write_table('table_classifier_diagnostics.tex',lines)
    for e in ['l14','b32']:
        for pol in ['protection','paired_accuracy']:
            lines=[r'\begin{tabular}{@{}lllrrrr@{}}',r'\toprule',r'Shift & Calibration & Readout & Finite & Met & Novel & Admitted novel \\',r'\midrule']
            for r in shifts:
                if r['encoder']!=e or r['policy']!=pol:continue
                cond='--' if r['conditional_novel'] is None else f"{100*r['conditional_novel']['mean']:.2f}"
                lines.append(' & '.join([r['corruption'],r['calibration'].replace('_',' '),NAMES[r['method']],f"{r['finite']}/10",f"{r['test_met']}/10",f"{100*r['novel']['mean']:.2f}",cond])+r' \\')
            lines+=[r'\bottomrule',r'\end{tabular}'];write_table(f'table_shift_{e}_{pol}.tex',lines)
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':13.5,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#1B4F72','#D55E00','#009E73','#777777','#0072B2','#CC79A7','#8C564B'];markers=['o','s','v','x','D','^','P']
    fig,axes=plt.subplots(2,3,figsize=(7.5,5.8))
    for i,e in enumerate(['l14','b32']):
        for j,d in enumerate(DATA):
            ax=axes[i,j]
            for k,m in enumerate(NAMES):
                yy=[100*bykey[e,d,m,pol]['novel']['mean'] for pol in ['protection','paired_accuracy']]
                ax.plot([0,1],yy,marker=markers[k],color=colors[k],lw=2.3 if m=='retained' else 1.2,label=NAMES[m],ms=4)
            ax.set_ylim(-3,103);ax.set_yticks([0,50,100]);ax.set_xticks([0,1],['Protect','Direct']);ax.set_xlim(-.12,1.12)
            ax.get_xticklabels()[0].set_ha('left')
            ax.get_xticklabels()[1].set_ha('right')
            if j>0:ax.set_yticklabels([])
            if j==0:ax.set_ylabel(('L/14' if i==0 else 'B/32')+'\nNovel accuracy (%)')
            ax.set_xlabel(DATA[d]);ax.grid(axis='y',alpha=.15)
    ha,la=axes[0,0].get_legend_handles_labels();fig.legend(ha,la,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.01),frameon=False,fontsize=12.5)
    fig.subplots_adjust(left=.13,right=.99,top=.98,bottom=.28,wspace=.13,hspace=.5)
    for ext in ['pdf','png']:fig.savefig(H/('calibration_fairness.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    # Readout-quality diagnosis: classes are descriptive repeated benchmark units.
    fig,axes=plt.subplots(2,3,figsize=(7.5,5.8))
    for i,e in enumerate(['l14','b32']):
        for j,d in enumerate(DATA):
            r=next(r for r in mechanism if r['encoder']==e and r['dataset']==d);cc=r['classes'];ax=axes[i,j]
            ax.scatter([100*c['prototype_accuracy'] for c in cc],[100*(c['reference_accuracy']-c['dense_accuracy']) for c in cc],s=12,alpha=.5,color='#1B4F72')
            ax.axhline(0,color='#777777',lw=.7);ax.set_xlim(0,100);ax.set_ylim(-40,40);ax.set_xticks([0,50,100]);ax.set_yticks([-40,0,40])
            ax.set_xlabel(DATA[d]+'\nPrototype accuracy (%)')
            if j>0:ax.set_yticklabels([])
            if j==0:ax.set_ylabel(('L/14' if i==0 else 'B/32')+'\nClosed-base gain (pp)')
    fig.subplots_adjust(left=.13,right=.99,top=.98,bottom=.17,wspace=.13,hspace=.6)
    for ext in ['pdf','png']:fig.savefig(H/('classifier_quality_diagnosis.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    print('REVIEWER SUMMARY',len(records),'records',len(policy),'policy groups',len(shifts),'shift groups',flush=True)
    for r in mechanism:print('MECHANISM',r['encoder'],r['dataset'],r['reference_minus_dense_closed_pp'],r['prototype_quality_gain_spearman'])
    for r in contrasts:
        if r['dataset']=='imagenetr':print('CONTRAST',r['encoder'],r['contrast'],r['mean'],r['ci95'])


if __name__=='__main__':main()
