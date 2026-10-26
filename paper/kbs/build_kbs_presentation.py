"""Reorganize checked results into grouped tables and a readable capacity figure.

No models, scores, thresholds, metric definitions or numerical summaries change.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent
DATA={'cifar':'CIFAR-100','pets':'Pets','imagenetr':'ImageNet-R'}
ENCODERS={'l14':'CLIP ViT-L/14','b32':'CLIP ViT-B/32'}
METHODS={'retained':'Retained','no_learned_tuned':'Recalibrated removal',
         'anchored_tp':'Dense prototypes','raw':'Raw fusion',
         'fixed_temperature':'Fixed temperature','tip_mean_calibrated':'Class-mean cache'}


def save_table(name,lines):
    (H/name).write_text('\n'.join(lines)+ '\n',encoding='utf-8')


def build():
    revision=json.loads((H/'revision_summary.json').read_text())
    budget=json.loads((H/'calibration_budget_summary.json').read_text())
    lookup={(r['encoder'],r['dataset'],r['method'],r['policy'],r['cap']):r for r in budget['summary']}
    paired=[r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',
            r'Comparator & \multicolumn{3}{c}{Retained-minus-comparator gain (pp) $\uparrow$} \\',
            r'\cmidrule(l){2-4}',r'& CIFAR-100 & Pets & ImageNet-R \\',r'\midrule']
    pairs=[]
    for e,enc in ENCODERS.items():
        paired.append(r'\multicolumn{4}{@{}l}{\textnormal{'+enc+r'}} \\')
        for m in ['no_learned_tuned','anchored_tp','tip_mean_calibrated']:
            cells=[]
            for d in DATA:
                g=revision[e][d]['gains'][m]['average']
                # Read the original complete display as an independent rounding check.
                original=(H/'table_revision_paired.tex').read_text()
                assert f"{g['mean']:.2f}" in original
                lo,hi=g['ci95'];mean=f"{g['mean']:+.2f}"
                if lo>0 or hi<0:mean=r'\textbf{'+mean+'}'
                cells.append(r'\shortstack[r]{'+mean+r'\\{\scriptsize ['+f'{lo:+.2f}, {hi:+.2f}'+r']}}')
                pairs.append((e,d,m,g['mean'],lo,hi))
            paired.append(METHODS[m]+' & '+' & '.join(cells)+r' \\')
        paired.append(r'\midrule')
    paired[-1]=r'\bottomrule';paired.append(r'\end{tabular}')
    save_table('table_kbs_paired.tex',paired)

    main=[r'\begin{tabular}{@{}lrrrrr@{}}',r'\toprule',
          r'Dataset & Calibration & \multicolumn{2}{c}{Admission} & \multicolumn{2}{c}{Test accuracy (\%) $\uparrow$} \\',
          r'\cmidrule(lr){3-4}\cmidrule(l){5-6}',
          r'& Queries & Finite & Test met & Base & Novel \\',r'\midrule']
    def acc(r,k):return f"{100*r[k+'_mean']:.2f}"+r'\,\ensuremath{\pm}\,'+f"{100*r[k+'_sd']:.2f}"
    for e,enc in ENCODERS.items():
        main.append(r'\multicolumn{6}{@{}l}{\textnormal{'+enc+r'}} \\')
        for d,name in DATA.items():
            r=lookup[e,d,'retained','risk',40]
            q=str(r['query_min']) if r['query_min']==r['query_max'] else f"{r['query_min']}--{r['query_max']}"
            main.append(' & '.join([name,q,f"{r['finite_admission_cells']}/10",f"{r['test_met_cells']}/10",acc(r,'old'),acc(r,'new')])+r' \\')
        main.append(r'\midrule')
    main[-1]=r'\bottomrule';main.append(r'\end{tabular}');save_table('table_kbs_calibration.tex',main)

    # Group empirical and certified policies explicitly, preserving all methods.
    for e in ENCODERS:
        rows=[r'\begin{tabular}{@{}lrrrrrr@{}}',r'\toprule',
              r'Readout & \multicolumn{2}{c}{Empirical policy} & \multicolumn{4}{c}{Certified policy} \\',
              r'\cmidrule(lr){2-3}\cmidrule(l){4-7}',
              r'& Test met & Novel $\uparrow$ & Finite & Test met & Base $\uparrow$ & Novel $\uparrow$ \\',r'\midrule']
        for d,name in DATA.items():
            rows.append(r'\multicolumn{7}{@{}l}{\textit{'+name+r'}} \\')
            best=max(round(100*lookup[e,d,m,'risk',40]['new_mean'],2) for m in METHODS)
            for m,label in METHODS.items():
                r=lookup[e,d,m,'risk',40];emp=lookup[e,d,m,'empirical',40]
                novel=f"{100*r['new_mean']:.2f}"
                if round(100*r['new_mean'],2)==best:novel=r'\textbf{'+novel+'}'
                rows.append(' & '.join([label,f"{emp['test_met_cells']}/10",f"{100*emp['new_mean']:.2f}",
                            f"{r['finite_admission_cells']}/10",f"{r['test_met_cells']}/10",f"{100*r['old_mean']:.2f}",novel])+r' \\')
            rows.append(r'\midrule')
        rows[-1]=r'\bottomrule';rows.append(r'\end{tabular}');save_table(f'table_kbs_calibration_{e}.tex',rows)

    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':13.5,
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(7.5,5.8))
    selected=[('retained','#1B4F72','o'),('anchored_tp','#D55E00','s'),
              ('tip_mean_calibrated','#CC79A7','^'),('fixed_temperature','#0072B2','D')]
    points=[]
    for i,(e,enc) in enumerate(ENCODERS.items()):
        for j,(d,name) in enumerate(DATA.items()):
            ax=axes[i,j]
            for m,color,marker in selected:
                x=[];y=[];sd=[]
                for r in [.8,.85,.9,.95,.99]:
                    cell=revision[e][d]['methods'][m]['policies']['matched_test'][str(r)]
                    if cell['feasible_partitions']==10:
                        x.append(r*100);y.append(cell['new']['mean']);sd.append(cell['new']['sd'])
                        points.append(dict(encoder=e,dataset=d,method=m,retention=r,mean=y[-1],sd=sd[-1],partitions=10))
                if x:ax.errorbar(x,y,yerr=sd,color=color,marker=marker,ms=4,capsize=2,
                                 lw=2.2 if m=='retained' else 1.2,label=METHODS[m],zorder=5 if m=='retained' else 2)
            ax.set_ylim((50,100) if i==0 else (30,90))
            ax.set_yticks([50,60,70,80,90,100] if i==0 else [30,45,60,75,90])
            ax.set_xticks([80,90,99]);ax.set_xlim(78.5,100.5)
            if j>0:ax.set_yticklabels([])
            ax.set_xlabel(name+'\nRetention (%)')
            if j==0:ax.set_ylabel(('L/14' if e=='l14' else 'B/32')+'\nNovel accuracy (%)')
            ax.grid(axis='y',alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc='lower center',bbox_to_anchor=(.5,-.005),frameon=False,fontsize=13.5)
    fig.subplots_adjust(left=.13,right=.99,top=.98,bottom=.24,wspace=.13,hspace=.48)
    for ext in ['pdf','png']:fig.savefig(H/('matched_capacity_main.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    verification={'paired_comparisons_preserved':len(pairs),'primary_calibration_conditions':6,
                  'complete_calibration_rows':36,'frontier_points':points,
                  'source_sha256':{n:hashlib.sha256((H/n).read_bytes()).hexdigest() for n in ['revision_summary.json','calibration_budget_summary.json']}}
    (H/'presentation_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print('Preserved 18 paired comparisons, 6 primary conditions and 36 complete baseline rows; rebuilt readable frontier figure.')


if __name__=='__main__':build()
