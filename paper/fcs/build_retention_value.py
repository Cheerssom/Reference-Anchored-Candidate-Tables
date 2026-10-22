"""Generate common-budget comparisons and paired sequential classifier gains."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LABELS={'retained':'Retained classifier','removed_fixed':'Classifier removed (fixed gates)',
        'no_learned_tuned':'No classifier (recalibrated)','anchored_tp':'Anchored text + prototypes'}
METHODS=list(LABELS)
DATA={n:json.loads((ROOT/'experiments'/f/'retention_value/results.json').read_text())
      for n,f in [('CIFAR','fcs'),('Pets','fcs_pets')]}
report={};csv_rows=[]

def stats(rows,key):
    vals=[np.mean([r[key] for r in rows if r['part']==p]) for p in sorted({r['part'] for r in rows})]
    return (100*np.mean(vals),100*np.std(vals,ddof=1) if len(vals)>1 else None,len(vals)) if vals else (None,None,0)
def cell(rows,key):
    m,s,n=stats(rows,key)
    if not n:return '--'
    return f'{m:.2f}' if s is None else f'{m:.2f} $\\pm$ {s:.2f}'
def selected(d,ev,method,session=5,ratio=None):
    return [r for r in d['records'] if r['evaluation']==ev and r['method']==method
            and r['session']==session and (ratio is None or r.get('ratio')==ratio)]
def paired(d,ev,other,session=None,ratio=None,key='overall'):
    rows=[r for r in d['records'] if r['evaluation']==ev and (session is None or r['session']==session)
          and (ratio is None or r.get('ratio')==ratio)]
    idx={(r['part'],r['order'],r['session'],r['method']):r for r in rows}
    result=[]
    for p in range(3):
        values=[]
        for k,r in idx.items():
            if k[0]!=p or k[3]!='retained':continue
            comp=idx.get((*k[:3],other))
            if comp is None:continue
            if ev!='fixed_gate' and not (r['feasible'] and comp['feasible']):continue
            values.append(r[key]-comp[key])
        expected=2*(5 if session is None else 1)
        if len(values)==expected:result.append({'part':p,'value':float(np.mean(values))})
    return result

lines=[r'\begin{tabular}{lllrrr}',r'\toprule',r'Data & Budget & Readout & Base & Novel & Feasible \\',r'\midrule']
sequential=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Data & Readout & Incremental avg. & Final & $\Delta$ avg. & $\Delta$ final \\',r'\midrule']
validation=[r'\begin{tabular}{lllrrr}',r'\toprule',r'Data & Budget & Readout & Base & Novel & Feasible \\',r'\midrule']
for name,d in DATA.items():
    folder=ROOT/'experiments'/('fcs' if name=='CIFAR' else 'fcs_pets')/'retention_value'
    assert hashlib.sha256((folder/'source_snapshot.py').read_bytes()).hexdigest()==d['source_sha256']
    assert len(d['records'])==1320
    feasible=[r for r in d['records'] if r['evaluation']=='matched_test' and r['feasible']]
    max_overshoot=max(r['selection_overshoot'] for r in feasible)*100
    for r in feasible:assert r['old']>=r['target']-1e-10
    for ratio in (.90,.95,.99):
        for method in METHODS:
            rows=[r for r in selected(d,'matched_test',method,ratio=ratio) if r['feasible']]
            n=len({r['part'] for r in rows})
            lines.append(name+f' & {100*ratio:.0f}\\% & '+LABELS[method]+' & '+cell(rows,'old')+' & '+cell(rows,'new')+f' & {n}/3'+r' \\')
            rows=[r for r in selected(d,'base_validation',method,ratio=ratio) if r['feasible']]
            n=len({r['part'] for r in rows})
            validation.append(name+f' & {100*ratio:.0f}\\% & '+LABELS[method]+' & '+cell(rows,'old')+' & '+cell(rows,'new')+f' & {n}/3'+r' \\')
    summary={'records':len(d['records']),'maximum_matched_base_overshoot_pp':max_overshoot,
             'fixed_gate':{},'matched':{},'validation':{}}
    for method in METHODS:
        rows=[r for r in d['records'] if r['evaluation']=='fixed_gate' and r['method']==method]
        final=selected(d,'fixed_gate',method)
        means={k:stats(rows,k)[0] for k in ['old','new','overall']}
        means['final_overall']=stats(final,'overall')[0]
        summary['fixed_gate'][method]=means
        gains=paired(d,'fixed_gate',method)
        final_gains=paired(d,'fixed_gate',method,5)
        sequential.append(name+' & '+LABELS[method]+' & '+cell(rows,'overall')+' & '+cell(final,'overall')
                          +' & '+cell(gains,'value')+' & '+cell(final_gains,'value')+r' \\')
    for ev,target_name in [('matched_test','matched'),('base_validation','validation')]:
        for ratio in d['ratios']:
            summary[target_name][str(ratio)]={}
            for method in METHODS:
                rows=[r for r in selected(d,ev,method,ratio=ratio) if r['feasible']]
                values={k:stats(rows,k)[0] for k in ['old','new','overall']}
                values['feasible_partitions']=len({r['part'] for r in rows})
                diff=paired(d,ev,method,5,ratio,key='new')
                values['paired_novel_gain_from_retaining_pp']=stats(diff,'value')[0]
                values['paired_partitions']=len(diff)
                summary[target_name][str(ratio)][method]=values
    for r in d['records']:
        csv_rows.append([name]+[r.get(k,'') for k in ['evaluation','part','order','session','added','method','ratio','target','feasible','offset','old','new','overall','selection_overshoot']])
    report[name]=summary
lines += [r'\bottomrule',r'\end{tabular}']
sequential += [r'\bottomrule',r'\end{tabular}']
validation += [r'\bottomrule',r'\end{tabular}']
(HERE/'table_matched_retention.tex').write_text('\n'.join(lines))
(HERE/'table_sequential_value.tex').write_text('\n'.join(sequential))
(HERE/'table_validation_value.tex').write_text('\n'.join(validation))
(HERE/'retention_value_summary.json').write_text(json.dumps(report,indent=2))
macros=[]
headline={
 'CFMatchedBase':report['CIFAR']['matched']['0.95']['retained']['old'],
 'CFMatchedNovel':report['CIFAR']['matched']['0.95']['retained']['new'],
 'CFTunedNovel':report['CIFAR']['matched']['0.95']['no_learned_tuned']['new'],
 'CFDenseNovel':report['CIFAR']['matched']['0.95']['anchored_tp']['new'],
 'CFMatchedGain':report['CIFAR']['matched']['0.95']['no_learned_tuned']['paired_novel_gain_from_retaining_pp'],
 'CFDenseGain':report['CIFAR']['matched']['0.95']['anchored_tp']['paired_novel_gain_from_retaining_pp'],
 'PetStrictBase':report['Pets']['matched']['0.99']['retained']['old'],
 'PetStrictNovel':report['Pets']['matched']['0.99']['retained']['new'],
 'PetStrictDenseNovel':report['Pets']['matched']['0.99']['anchored_tp']['new'],
 'PetStrictGain':report['Pets']['matched']['0.99']['anchored_tp']['paired_novel_gain_from_retaining_pp']}
headline['CFMatchedGainSD']=stats(paired(DATA['CIFAR'],'matched_test','no_learned_tuned',5,.95,key='new'),'value')[1]
headline['CFDenseGainSD']=stats(paired(DATA['CIFAR'],'matched_test','anchored_tp',5,.95,key='new'),'value')[1]
headline['PetStrictGainSD']=stats(paired(DATA['Pets'],'matched_test','anchored_tp',5,.99,key='new'),'value')[1]
for name,d in DATA.items():
    prefix='CF' if name=='CIFAR' else 'Pet'
    for other,suffix in [('no_learned_tuned','Tuned'),('anchored_tp','Dense'),('removed_fixed','Removed')]:
        for session,ending in [(None,'AverageGain'),(5,'FinalGain')]:
            m,s,_=stats(paired(d,'fixed_gate',other,session),'value')
            headline[prefix+suffix+ending]=m
            headline[prefix+suffix+ending+'SD']=s
for name,value in headline.items():macros.append(f'\\newcommand{{\\{name}}}{{{value:.2f}}}')
(HERE/'retention_numbers.tex').write_text('\n'.join(macros)+'\n')
with (HERE/'retention_value_records.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['dataset','evaluation','part','order','session','added','method','ratio','target','feasible','offset','old','new','overall','selection_overshoot']);w.writerows(csv_rows)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                    'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
colors={'retained':'#27627E','removed_fixed':'#C56837','no_learned_tuned':'#98649A','anchored_tp':'#638749'}
fig,axes=plt.subplots(1,2,figsize=(10,3.35))
for ax,(name,d) in zip(axes,DATA.items()):
    for method in METHODS:
        ys=[];errors=[]
        for ratio in d['ratios']:
            rows=[r for r in selected(d,'matched_test',method,ratio=ratio) if r['feasible']]
            m,s,n=stats(rows,'new');ys.append(m if n==3 else np.nan);errors.append(s if n==3 else np.nan)
        ax.errorbar(np.array(d['ratios'])*100,ys,yerr=errors,marker='o',ms=3,capsize=2,
                    color=colors[method],label=LABELS[method])
    ax.set_title(name);ax.set_xticks(np.array(d['ratios'])*100)
    ax.set_xlabel('Required base accuracy / retained closed-base accuracy (%)',fontsize=8)
    ax.set_ylabel('Novel accuracy (%)');ax.grid(alpha=.15)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.08),ncol=2,frameon=False,fontsize=8)
fig.tight_layout()
for fmt in ('pdf','png'):fig.savefig(HERE/'figures'/('matched_retention.'+fmt),bbox_inches='tight',dpi=220)
plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(9.5,5.6))
for col,(name,d) in enumerate(DATA.items()):
    added=np.arange(1,6)*(8 if name=='CIFAR' else 3)
    for row,(ev,key,ratio) in enumerate([('matched_test','new',.95),('fixed_gate','overall',None)]):
        ax=axes[row,col]
        for method in ['no_learned_tuned','anchored_tp']:
            values=[stats(paired(d,ev,method,t,ratio,key=key),'value') for t in range(1,6)]
            ax.errorbar(added,[v[0] for v in values],yerr=[v[1] for v in values],marker='o',ms=3,capsize=2,
                        color=colors[method],label='vs '+LABELS[method])
        ax.axhline(0,color='#888888',lw=.8);ax.set_xticks(added)
        ax.set_xlabel('Cumulative novel classes');ax.grid(alpha=.15)
        ax.set_ylabel('Novel gain at 95% budget (pp)' if row==0 else 'Fixed-gate all-class gain (pp)',fontsize=8)
        if row==0:ax.set_title(name)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.04),ncol=2,frameon=False,fontsize=8)
fig.tight_layout()
for fmt in ('pdf','png'):fig.savefig(HERE/'figures'/('sequential_value.'+fmt),bbox_inches='tight',dpi=220)
plt.close(fig)
print(json.dumps({'headline':headline,'records':{k:v['records'] for k,v in report.items()}},indent=2))
