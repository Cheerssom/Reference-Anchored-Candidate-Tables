"""Build complete independent-calibration tables without changing old results."""
from pathlib import Path
import csv,json
import numpy as np

H=Path(__file__).resolve().parent
ROOT=H.parents[1]
OUT=ROOT/'experiments/kbs_independent_risk'
NAMES={'retained':'Retained','no_learned_tuned':'Recalibrated removal',
       'anchored_tp':'Dense prototypes','raw':'Raw fusion',
       'fixed_temperature':'Fixed temperature','tip_mean_calibrated':'Class-mean cache'}
DATA={'cifar':'CIFAR-100','pets':'Pets','imagenetr':'ImageNet-R'}


def build():
    assert json.loads((OUT/'verification.json').read_text())['complete']
    rows=[]
    for encoder in ['l14','b32']:
        for dataset in DATA:
            for part in range(10):
                for order in [0,1]:
                    for t in range(1,6):
                        f=OUT/encoder/dataset/f'p{part}'/f'o{order}_t{t}.json'
                        rows += [dict(encoder=encoder,dataset=dataset,**r) for r in json.loads(f.read_text())['records']]
    summaries=[];trajectories=[]
    for encoder in ['l14','b32']:
        for dataset in DATA:
            for method in NAMES:
                for policy,level in [('empirical',.95),('empirical',.97),('empirical',.99),
                                     ('risk',.10),('risk',.05),('risk',.01)]:
                    group=[r for r in rows if (r['encoder'],r['dataset'],r['method'],r['policy'],r['level']) ==
                           (encoder,dataset,method,policy,level)]
                    final=[r for r in group if r['order']==0 and r['session']==5]
                    assert len(final)==10
                    expanded=[r for r in final if not r['fallback']]
                    s=dict(encoder=encoder,dataset=dataset,method=method,policy=policy,level=level,
                           partitions=10,expanded=len(expanded),fallbacks=10-len(expanded),
                           test_met=sum(r['test_target_met'] for r in final),
                           expanded_test_met=sum(r['test_target_met'] for r in expanded))
                    for k in ['old','new','overall','target_gap_pp','test_retention_ratio']:
                        s[k+'_mean']=float(np.mean([r[k] for r in final]))
                    s['worst_shortfall_pp']=max(0.,-min(r['target_gap_pp'] for r in final))
                    s['expanded_novel_mean']=float(np.mean([r['new'] for r in expanded])) if expanded else None
                    summaries.append(s)
                    all_met=all_expand=0;av=[]
                    for p in range(10):
                        for o in [0,1]:
                            g=[r for r in group if r['part']==p and r['order']==o];assert len(g)==5
                            all_met+=all(r['test_target_met'] for r in g)
                            all_expand+=all(not r['fallback'] for r in g)
                            av.append(np.mean([r['overall'] for r in g]))
                    trajectories.append(dict(encoder=encoder,dataset=dataset,method=method,policy=policy,level=level,
                                             paths=20,all_sessions_test_met=all_met,
                                             all_sessions_expand=all_expand,average_incremental=float(np.mean(av))))
    result=dict(final=summaries,trajectories=trajectories,metric_records=len(rows))
    (H/'independent_risk_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (H/'independent_risk_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    def get(e,d,m,p='risk',l=.05):
        return next(s for s in summaries if (s['encoder'],s['dataset'],s['method'],s['policy'],s['level'])==(e,d,m,p,l))
    def table(encoder,methods,caption=False):
        lines=[r'\begin{tabular}{llrrrrrr}',r'\toprule',
               r'Dataset & Readout & Emp. met & Emp. novel & Cert. exp. & Test met & Cert. base & Cert. novel \\',r'\midrule']
        for d in DATA:
            for m in methods:
                a=get(encoder,d,m,'empirical',.95);b=get(encoder,d,m)
                lines.append(f"{DATA[d]} & {NAMES[m]} & {a['test_met']}/10 & {100*a['new_mean']:.2f} & {b['expanded']}/10 & {b['test_met']}/10 & {100*b['old_mean']:.2f} & {100*b['new_mean']:.2f} \\\\")
            lines.append(r'\midrule')
        lines[-1]=r'\bottomrule';lines += [r'\end{tabular}']
        return '\n'.join(lines)+'\n'
    for e in ['l14','b32']:
        (H/f'table_independent_risk_{e}.tex').write_text(table(e,list(NAMES)))
    (H/'table_independent_risk_main.tex').write_text(table('l14',['retained','anchored_tp','fixed_temperature','tip_mean_calibrated']))
    lines=['# Independent risk calibration results','',
           'All 60 refitted partition runs and 600 arrival sessions completed; 21,600 metric records reconstructed independently. The original fitting/validation pools are repartitioned before all ridge, fusion and cache selection. Three disjoint pools share indices across encoders and readouts. No new images, model weights or feature caches are copied.','',
           'Primary: 95% population macro retention, confidence 95% across the five arrivals of each prespecified method/order/partition, conditional on the stated class-conditional sampling and model assumptions. This is not a simultaneous guarantee across all methods or benchmarks, a guarantee under distribution shift, or a guarantee for every finite test sample. The test benchmark has been used previously in research development.','',
           'Cert. exp. counts finite certified thresholds. When no such threshold exists, the shared original reference classifier serves base classes and novel accuracy is zero. All-ten test attainment including fallback is not evidence of useful expansion. The certificate uses a common sufficient protection loss; it conservatively ignores recovery of reference errors by replacement readouts.','',
           '| Encoder | Dataset | Readout | Empirical test met | Empirical novel % | Certified expansion | Risk-policy test met | Risk-policy novel % |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for e in ['l14','b32']:
        for d in DATA:
            for m in NAMES:
                a=get(e,d,m,'empirical',.95);b=get(e,d,m)
                lines.append(f"| {e} | {DATA[d]} | {NAMES[m]} | {a['test_met']}/10 | {100*a['new_mean']:.2f} | {b['expanded']}/10 | {b['test_met']}/10 | {100*b['new_mean']:.2f} |")
    lines += ['','Full confidence-level and safety-margin outcomes are in independent_risk_summary.csv. Exact threshold test traces and raw predictions remain in experiments/kbs_independent_risk. Methods use a smaller refitted training pool than the earlier manuscript comparisons; their numerical results must not be mixed as if the fitted models were unchanged.','',
              'Reproduce: scripts/run_kbs_independent_risk.ps1; python src/experiments/kbs_verify_independent_risk.py; python paper/kbs/build_independent_risk_results.py. Shared PAUSE: experiments/kbs_strengthening/PAUSE. Each completed session and partition is saved atomically.','']
    (H/'independent_risk_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps([s for s in summaries if s['method']=='retained' and s['policy']=='risk' and s['level']==.05],indent=2))


if __name__=='__main__':build()
