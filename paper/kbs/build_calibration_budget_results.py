"""Tables and publication figures for independent calibration coverage curves."""
from pathlib import Path
from collections import defaultdict
import csv,json,math,os
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent
ROOT=H.parents[1]
OUT=ROOT/'experiments/kbs_calibration_budget'
NAMES={'retained':'Retained','no_learned_tuned':'Recalibrated removal',
       'anchored_tp':'Dense prototypes','raw':'Raw fusion',
       'fixed_temperature':'Fixed temperature','tip_mean_calibrated':'Class-mean cache'}
DATA={'cifar':'CIFAR-100','pets':'Pets','imagenetr':'ImageNet-R'}
CAPS=[2,5,10,20,40]


def build():
    assert json.loads((OUT/'verification.json').read_text())['complete']
    groups=defaultdict(list);raw_count=0
    for encoder in ['l14','b32']:
        for dataset in DATA:
            for p in range(10):
                for o in [0,1]:
                    for t in range(1,6):
                        f=OUT/encoder/dataset/f'p{p}/o{o}_t{t}.json'
                        for r in json.loads(f.read_text())['records']:
                            groups[(encoder,dataset,r['method'],r['policy'],r['cap'])].append(r);raw_count+=1
    summaries=[];trajectories=[];partition_means={}
    for key,rows in groups.items():
        e,d,m,policy,cap=key;final=[r for r in rows if r['order']==0 and r['session']==5]
        assert len(final)==(10 if cap==40 else 30)
        s=dict(encoder=e,dataset=d,method=m,policy=policy,cap=cap,partitions=10,
               subsets_per_partition=1 if cap==40 else 3,final_cells=len(final),
               finite_admission_cells=sum(not r['fallback'] for r in final),
               test_met_cells=sum(r['test_target_met'] for r in final),
               fallback_cells=sum(r['fallback'] for r in final))
        per={p:{k:float(np.mean([r[k] for r in final if r['part']==p]))
                for k in ['old','new','overall','target_gap_pp','calibration_queries','minimum_class_count']}
             for p in range(10)}
        for k in per[0]:
            vals=[per[p][k] for p in range(10)];s[k+'_mean']=float(np.mean(vals));s[k+'_sd']=float(np.std(vals,ddof=1))
        s['worst_shortfall_pp']=max(0.,-min(r['target_gap_pp'] for r in final))
        s['minimum_class_count_all']=min(r['minimum_class_count'] for r in final)
        s['query_min']=min(r['calibration_queries'] for r in final);s['query_max']=max(r['calibration_queries'] for r in final)
        summaries.append(s);partition_means[key]=[per[p]['new'] for p in range(10)]
        paths=defaultdict(list)
        for r in rows:paths[(r['part'],r['order'],r['subset_seed'])].append(r)
        assert all(len(v)==5 for v in paths.values())
        trajectories.append(dict(encoder=e,dataset=d,method=m,policy=policy,cap=cap,paths=len(paths),
                                 all_sessions_test_met=sum(all(r['test_target_met'] for r in v) for v in paths.values()),
                                 all_sessions_admit=sum(all(not r['fallback'] for r in v) for v in paths.values())))
    paired=[]
    indices=np.random.default_rng(2026).integers(0,10,(20000,10))
    for e in ['l14','b32']:
        for d in DATA:
            a=np.array(partition_means[(e,d,'retained','risk',40)])
            for m in NAMES:
                if m=='retained':continue
                diff=100*(a-np.array(partition_means[(e,d,m,'risk',40)]));boot=diff[indices].mean(1)
                lo,hi=np.quantile(boot,[.025,.975])
                ra={r['part']:r for r in groups[(e,d,'retained','risk',40)] if r['order']==0 and r['session']==5}
                rb={r['part']:r for r in groups[(e,d,m,'risk',40)] if r['order']==0 and r['session']==5}
                joint=[p for p in range(10) if not ra[p]['fallback'] and not rb[p]['fallback']]
                jd=100*np.array([ra[p]['new']-rb[p]['new'] for p in joint])
                paired.append(dict(encoder=e,dataset=d,comparator=m,novel_gain_pp=float(diff.mean()),
                                   ci_low=float(lo),ci_high=float(hi),joint_finite_partitions=len(joint),
                                   joint_finite_gain_pp=float(jd.mean()) if len(jd) else None,
                                   partition_values=diff.tolist()))
    bound=[]
    threshold=math.log(.05/5)/math.log(.95)
    for d in DATA:
        old=[json.loads(f.read_text()) for f in (ROOT/f'experiments/kbs_independent_risk/l14/{d}').glob('p*/complete.json')]
        counts=[1/r['calibration_max_weight'] for r in old]
        bound.append(dict(dataset=d,original_effective_min=min(counts),original_effective_max=max(counts),
                          zero_loss_necessary_effective_count=threshold))
    result=dict(metric_records=raw_count,summary=summaries,trajectories=trajectories,paired=paired,zero_loss_bound=bound)
    (H/'calibration_budget_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,records in [('calibration_budget_summary.csv',summaries),('calibration_budget_paired.csv',paired)]:
        with (H/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lookup={(r['encoder'],r['dataset'],r['method'],r['policy'],r['cap']):r for r in summaries}
    def get(e,d,m='retained',policy='risk',cap=40):return lookup[e,d,m,policy,cap]
    # Retained headline, all datasets and encoders, with explicit costs/outcomes.
    lines=[r'\begin{tabular}{llrrrrr}',r'\toprule',
           r'Encoder & Dataset & Queries & Exp. & Test met & Base & Novel \\',r'\midrule']
    for e in ['l14','b32']:
        for d in DATA:
            r=get(e,d);enc='L/14' if e=='l14' else 'B/32'
            queries=str(r['query_min']) if r['query_min']==r['query_max'] else f"{r['query_min']}--{r['query_max']}"
            lines.append(f"{enc} & {DATA[d]} & {queries} & {r['finite_admission_cells']}/10 & {r['test_met_cells']}/10 & {100*r['old_mean']:.2f} & {100*r['new_mean']:.2f} \\\\")
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
    (H/'table_calibration_budget_main.tex').write_text('\n'.join(lines)+'\n')
    for e in ['l14','b32']:
        lines=[r'\begin{tabular}{llrrrrrr}',r'\toprule',
               r'Dataset & Readout & Emp. met & Emp. novel & Cert. exp. & Test met & Base & Novel \\',r'\midrule']
        for d in DATA:
            for m,n in NAMES.items():
                r=get(e,d,m);emp=get(e,d,m,'empirical')
                lines.append(f"{DATA[d]} & {n} & {emp['test_met_cells']}/10 & {100*emp['new_mean']:.2f} & {r['finite_admission_cells']}/10 & {r['test_met_cells']}/10 & {100*r['old_mean']:.2f} & {100*r['new_mean']:.2f} \\\\")
            lines.append(r'\midrule')
        lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
        (H/f'table_calibration_budget_{e}.tex').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':13.5,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors=['#1B4F72','#009E73','#D55E00','#777777','#0072B2','#CC79A7']
    markers=['o','v','s','x','D','^']
    for metric,stem in [('new','calibration_budget_curve'),('finite','calibration_certification_curve')]:
        fig,axes=plt.subplots(2,3,figsize=(7.5,5.8),sharey=True)
        for i,e in enumerate(['l14','b32']):
            for j,d in enumerate(DATA):
                ax=axes[i,j]
                for k,m in enumerate(NAMES):
                    values=[];sd=[]
                    for cap in CAPS:
                        r=get(e,d,m,cap=cap)
                        values.append(100*r['new_mean'] if metric=='new' else 100*r['finite_admission_cells']/r['final_cells'])
                        sd.append(100*r['new_sd'])
                    ax.plot(CAPS,values,color=colors[k],marker=markers[k],ms=4,
                            lw=2.3 if m=='retained' else 1.3,label=NAMES[m],zorder=5 if m=='retained' else 2)
                    if metric=='new' and m=='retained':
                        ax.fill_between(CAPS,np.maximum(0,np.array(values)-sd),np.minimum(100,np.array(values)+sd),
                                        color=colors[k],alpha=.12,linewidth=0)
                bcount={'cifar':40,'pets':15,'imagenetr':100}[d]
                minimum=math.ceil(threshold/bcount)
                if minimum>=2:ax.axvline(minimum,color='#999999',ls=':',lw=1,zorder=0)
                ax.set_xscale('log',base=2);ax.set_xticks(CAPS,labels=list(map(str,CAPS)))
                ax.set_ylim(-3,103);ax.set_yticks([0,50,100]);ax.tick_params(labelsize=13.5)
                ax.set_xlabel(DATA[d]+' cap',fontsize=13.5)
                if j==0:ax.set_ylabel(('L/14' if e=='l14' else 'B/32')+'\n'+('Novel accuracy (%)' if metric=='new' else 'Finite admission (%)'))
                ax.grid(axis='y',alpha=.15)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.005),frameon=False,fontsize=13.5)
        fig.subplots_adjust(left=.13,right=.99,top=.98,bottom=.24,wspace=.13,hspace=.48)
        fig.savefig(H/(stem+'.pdf'),bbox_inches='tight');fig.savefig(H/(stem+'.png'),dpi=300,bbox_inches='tight');plt.close(fig)
    lines=['# Independent calibration coverage and label budgets','',
           'Fixed existing readouts, tuning parameters, support and test archives; only independent calibration coverage changes. Caps 2/5/10/20/40 use nested class-wise subsets. Three subset seeds are averaged within each partition; cap 40 uses the identical full set once. The primary setting is cap 40, 95% retention and 95% five-arrival confidence for each prespecified method/order/partition.','',
           'The zero-loss necessary effective count is log(0.01)/log(0.95)=89.78. Original CIFAR counts are 80 and Pets 45--60, so no method could pass even at zero observed protection loss under the original bound. This identifies a certificate-feasibility bottleneck, not an empirical failure of the retained scoring mechanism.','',
           '| Encoder | Dataset | Queries | Certified expansion | Test met | Novel % |',
           '|---|---|---:|---:|---:|---:|']
    for e in ['l14','b32']:
        for d in DATA:
            r=get(e,d);lines.append(f"| {e} | {DATA[d]} | {r['query_min']}--{r['query_max']} | {r['finite_admission_cells']}/10 | {r['test_met_cells']}/10 | {100*r['new_mean']:.2f} |")
    lines+=['','Budget is a per-class cap; ImageNet-R availability can truncate it. Actual totals and minimum counts are in the CSV. Every operational mean includes base-only fallback with zero novel accuracy. Test attainment including fallback must not be equated with useful certified admission.','',
            'All six methods use identical calibration subsets and the same original reference. The sufficient protection criterion conservatively ignores recovery of reference errors by replacement readouts. Population confidence relies on independent class-conditional sampling and prespecified arrivals; this is an adaptive revision of previously examined benchmarks, not a new blind test.','',
            'Full tables, paired conditional bootstrap intervals, trajectory outcomes and all caps remain in calibration_budget_summary.json/CSV and calibration_budget_paired.csv. Paired policy gains include all ten partitions; jointly finite sets and their conditional mean gains are additionally identified. Supplemental identities and content hashes exclude fitting, tuning, support and test images. Frozen features are reused wherever possible; only new calibration images are encoded.','',
            'At the primary cap40 setting all final retained-table partitions admit expansion. All 20 test trajectories per dataset/encoder meet their targets; Pets has 17/20 L/14 and 16/20 B/32 paths with finite admission at every arrival, with explicit early-session fallback on the remaining paths. This distinction is retained rather than counting fallback as useful admission.','']
    (H/'calibration_budget_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps([r for r in summaries if r['method']=='retained' and r['policy']=='risk' and r['cap']==40],indent=2))


if __name__=='__main__':build()
