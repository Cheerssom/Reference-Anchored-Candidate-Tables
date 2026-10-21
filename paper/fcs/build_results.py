"""Generate every new manuscript number, table and plot from experiment JSON."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
D=json.loads((PROJECT/'experiments/fcs/results.json').read_text())
R=D['records']
FIG=ROOT/'figures'
FIG.mkdir(exist_ok=True)
LABELS={'text':'CLIP text','prototype':'Visual prototypes',
 'text_prototype':'Text + prototypes','tip_adapter':'Tip-Adapter (tuned)',
 'raw':'Raw evidence fusion','dynamic':'Candidate-normalized table',
 'reference':'Reference-anchored table','no_learned':'Without learned evidence',
 'no_prototype':'Without novel prototypes','no_offset':'Without offset'}

def stat(method,k,key):
    # Partitions are independent units; support subsets are nested within them.
    # At k=5 all five reserved images are used, so support seed is not replication.
    x=[np.mean([r[key] for r in R if r['part']==p and r['k']==k and r['method']==method]) for p in (0,1,2)]
    return np.mean(x)*100,np.std(x,ddof=1)*100

def cell(method,k,key):
    a,b=stat(method,k,key)
    return f'{a:.2f} $\\pm$ {b:.2f}'

def table(methods,k):
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',
           r'Readout & Base & Novel & All & Harmonic \\',r'\midrule']
    lines += [LABELS[m]+' & '+' & '.join(cell(m,k,key) for key in ('old','new','overall','harmonic'))+r' \\' for m in methods]
    lines += [r'\bottomrule',r'\end{tabular}']
    return '\n'.join(lines)

(ROOT/'table_expansion.tex').write_text(table(['text','prototype','text_prototype','tip_adapter','raw','dynamic','reference'],5))
(ROOT/'table_ablation.tex').write_text(table(['reference','no_learned','no_prototype','no_offset'],5))
lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Shots & Base & Novel & All & Harmonic \\',r'\midrule']
for k in (1,2,5):
    lines.append(str(k)+' & '+' & '.join(cell('reference',k,key) for key in ('old','new','overall','harmonic'))+r' \\')
lines += [r'\bottomrule',r'\end{tabular}']
(ROOT/'table_shots.tex').write_text('\n'.join(lines))

macros=[]
for method,prefix in [('reference','Ref'),('dynamic','Dynamic'),('text_prototype','Simple'),('text','Text')]:
    for key,suffix in [('old','Base'),('new','Novel'),('overall','All'),('harmonic','Harmonic')]:
        a,b=stat(method,5,key)
        macros += [f'\\newcommand{{\\{prefix}{suffix}}}{{{a:.2f}}}',
                   f'\\newcommand{{\\{prefix}{suffix}SD}}{{{b:.2f}}}']
for key,suffix in [('correct_to_wrong','ErrorFlip'),('old_prediction_churn','Churn')]:
    vals=np.array([x[key] for x in D['invariance']])*100
    macros += [f'\\newcommand{{\\Ref{suffix}}}{{{vals.mean():.2f}}}',
               f'\\newcommand{{\\Ref{suffix}SD}}{{{vals.std(ddof=1):.2f}}}']
for n,name in [(50,'Fifty'),(60,'Sixty'),(80,'Eighty')]:
    ms=np.mean([x['milliseconds'] for x in D['efficiency'] if x['candidates']==n])
    macros.append(f'\\newcommand{{\\Latency{name}}}{{{ms:.2f}}}')
(ROOT/'numbers.tex').write_text('\n'.join(macros)+'\n')

with (ROOT/'results_summary.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['method','k','metric','partition_mean_percent','partition_sample_sd_percent'])
    for k in (1,2,5):
        for method in LABELS:
            for key in ('old','new','overall','harmonic'):
                w.writerow([method,k,key,*stat(method,k,key)])

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                    'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
blue,orange,green='#27627E','#C56837','#638749'
fig,axs=plt.subplots(1,3,figsize=(10.5,2.9))
for method,color in [('reference',blue),('dynamic',orange)]:
    for ax,key,scale in zip(axs,['max_old_score_drift','old_rank_churn','old_prediction_churn'],[1,100,100]):
        means=[];std=[]
        for n in (10,20,40):
            vals=np.array([x[key] for x in D['vocabulary'] if x['method']==method and x['added']==n])*scale
            means.append(vals.mean());std.append(vals.std(ddof=1))
        ax.errorbar([10,20,40],means,yerr=std,marker='o',ms=4,capsize=3,color=color,label=LABELS[method])
        ax.set_xlabel('Appended classes');ax.set_xticks([10,20,40]);ax.grid(alpha=.15)
axs[0].set_ylabel('Maximum base-score change')
axs[1].set_ylabel('Base-only rank churn (%)')
axs[2].set_ylabel('Expanded prediction churn (%)')
axs[0].set_title('(a) Score preservation',loc='left')
axs[1].set_title('(b) Existing-class order',loc='left')
axs[2].set_title('(c) New-class competition',loc='left')
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.07))
fig.tight_layout();fig.savefig(FIG/'stability.pdf',bbox_inches='tight');fig.savefig(FIG/'stability.png',dpi=180,bbox_inches='tight');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(7.2,2.9))
for m,color in [('reference',blue),('dynamic',orange),('text_prototype',green)]:
    for ax,key in zip(axs,['old','new']):
        vals=[stat(m,k,key) for k in (1,2,5)]
        ax.errorbar([1,2,5],[v[0] for v in vals],yerr=[v[1] for v in vals],color=color,marker='o',capsize=3,label=LABELS[m])
        ax.set_xticks([1,2,5]);ax.set_xlabel('Novel support images per class');ax.set_ylabel('Accuracy (%)');ax.grid(alpha=.15)
axs[0].set_title('(a) Base classes',loc='left');axs[1].set_title('(b) Novel classes',loc='left')
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=1,frameon=False,bbox_to_anchor=(.5,-.23))
fig.tight_layout();fig.savefig(FIG/'shots.pdf',bbox_inches='tight');fig.savefig(FIG/'shots.png',dpi=180,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(5.2,3.3))
for m,col in [('reference',blue),('dynamic',orange)]:
    pts=[]
    for off in sorted(set(x['offset'] for x in D['frontier'])):
        rr=[x for x in D['frontier'] if x['method']==m and x['offset']==off]
        pts.append((100*np.mean([x['old'] for x in rr]),100*np.mean([x['new'] for x in rr])))
    ax.plot(*np.array(pts).T,'o-',ms=4,color=col,label=LABELS[m]+' (offset sweep)')
for m,col,mark in [('reference',blue,'*'),('text_prototype',green,'s'),('tip_adapter','#8261A4','D'),('text','#666666','^')]:
    x=stat(m,5,'old')[0];y=stat(m,5,'new')[0]
    ax.scatter(x,y,color=col,marker=mark,s=95 if mark=='*' else 38,zorder=4,label=LABELS[m]+' (selected)' if mark=='*' else LABELS[m])
ax.set_xlabel('Base accuracy (%)');ax.set_ylabel('Novel accuracy (%)');ax.grid(alpha=.15)
ax.legend(frameon=False,fontsize=7,loc='lower left')
fig.tight_layout();fig.savefig(FIG/'tradeoff.pdf',bbox_inches='tight');fig.savefig(FIG/'tradeoff.png',dpi=180,bbox_inches='tight');plt.close(fig)

print((ROOT/'table_expansion.tex').read_text())
print('Generated tables, macros, CSV, and three vector figures.')

pets_path=PROJECT/'experiments/fcs_pets/results.json'
if pets_path.exists():
    cifar_records=R
    pets=json.loads(pets_path.read_text())
    R=pets['records']
    methods=['text','prototype','text_prototype','tip_adapter','raw','dynamic','reference']
    (ROOT/'table_pets.tex').write_text(table(methods,5))
    extra=[]
    for method,prefix in [('reference','PetRef'),('dynamic','PetDynamic'),('text_prototype','PetSimple'),('text','PetText')]:
        for key,suffix in [('old','Base'),('new','Novel'),('overall','All'),('harmonic','Harmonic')]:
            a,b=stat(method,5,key)
            extra += [f'\\newcommand{{\\{prefix}{suffix}}}{{{a:.2f}}}',
                      f'\\newcommand{{\\{prefix}{suffix}SD}}{{{b:.2f}}}']
    with (ROOT/'numbers.tex').open('a') as f:f.write('\n'.join(extra)+'\n')
    with (ROOT/'pets_summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','k','metric','partition_mean_percent','partition_sample_sd_percent'])
        for k in (1,2,5):
            for method in LABELS:
                for key in ('old','new','overall','harmonic'):w.writerow([method,k,key,*stat(method,k,key)])
    print((ROOT/'table_pets.tex').read_text())
    print('Pets max reference drift',max(x['max_old_score_drift'] for x in pets['vocabulary'] if x['method']=='reference'))
    R=cifar_records
