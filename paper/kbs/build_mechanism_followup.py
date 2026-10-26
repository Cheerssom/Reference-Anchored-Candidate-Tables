"""Separate diagnostic figures/report, generated only after frozen-run checks."""
from pathlib import Path
import csv,json,hashlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
H=Path(__file__).resolve().parent;ROOT=H.parents[1];OUT=ROOT/'experiments/kbs_mechanism_followup'
FIG=H/'mechanism_followup';FIG.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'src/experiments'))
from kbs_verify_reviewer_revision import direct_p
DATA={'cifar':'CIFAR-100','pets':'Pets','imagenetr':'ImageNet-R'}
COL={'retained':'#1B4F72','anchored_tp':'#D55E00'}
NAME={'retained':'Retained','anchored_tp':'Dense prototypes'}

def weighted_auc(y,s,w):
    # Independent weighted Mann-Whitney calculation with half credit for ties.
    order=np.argsort(s,kind='stable');s=s[order];y=y[order];w=w[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    pos=np.add.reduceat(w*y,starts);neg=np.add.reduceat(w*(1-y),starts)
    before=np.r_[0,np.cumsum(neg)[:-1]]
    return float(np.sum(pos*(before+.5*neg))/(pos.sum()*neg.sum()))

def summarize(v,scale=100):
    a=np.asarray(v,float);rng=np.random.default_rng(20261023)
    if not len(a):return dict(n=0,mean=None,ci95=None)
    b=a[rng.integers(0,len(a),(20000,len(a)))].mean(1)*scale
    return dict(n=len(a),mean=float(scale*a.mean()),ci95=np.quantile(b,[.025,.975]).tolist() if len(a)>=5 else None,
        values=(scale*a).tolist(),small_n=len(a)<5)

def verify(records):
    tests=0
    for path,r in records:
        raw=path.with_suffix('.npz');assert hashlib.sha256(raw.read_bytes()).hexdigest()==r['raw_sha256']
        for name,sha in r['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
        with np.load(raw) as z:
            for row in r['rows']:
                assert abs(row['base']-(row['reference']-row['destroyed_to_novel']-row['destroyed_to_base']+row['recovered']))<1e-10
                if row['method']=='retained':assert row['recovered']==0 and row['destroyed_to_base']==0
                if not row['fallback']:
                    m=row['method'];ix=np.searchsorted(z[m+'_offset'],row['offset'],side='right')-1
                    assert abs(z[m+'_base'][ix]-row['base'])<1e-10 and abs(z[m+'_novel'][ix]-row['novel'])<1e-10
            for comparison in r['comparisons']:
                for m,v in comparison['matched'].items():
                    take=z[m+'_base']>=comparison['baseline_realized_base']-1e-10
                    assert bool(take.any())==v['feasible']
                    if take.any():assert abs(float(z[m+'_novel'][take].max())-v['novel'])<1e-10
            for name,score in [('delta',z['delta']),('raw_margin',-z['uncentered_margin']),('centered_margin',-z['centered_margin'])]:
                assert abs(weighted_auc(z['novel_mask'].astype(int),score,z['weights'])-r['bias_auc'][name])<1e-10
            # Rebuild the new uncentered final-session calibration trace.
            cp=next(ROOT/n for n in r['source_hashes'] if 'kbs_calibration_budget' in n)
            with np.load(cp) as cal:
                y=cal['labels'];base=cal['base_classes'];w=np.array([1/(len(base)*np.sum(y==c)) for c in y])
                reference=cal['reference_correct'];bp=cal['retained_base_prediction'];npred=cal['retained_novel_prediction']
                accepted=None
                for i,step in enumerate(r['uncentered_direct']['trace']):
                    assert step['offset']==z['uncentered_candidates'][i]
                    off=step['offset'];ch=(off>z['uncentered_cal_margin'])|((off==z['uncentered_cal_margin'])&(npred<bp))
                    correct=(bp==y)&~ch;p=direct_p(reference,correct,w);tests+=1
                    assert abs(p-step['pvalue'])<1e-8
                    if p<=.01:accepted=off
                    else:assert i==len(r['uncentered_direct']['trace'])-1
                assert accepted==r['uncentered_direct']['offset']
            tp=next(ROOT/n for n in r['source_hashes'] if 'kbs_independent_risk' in n and n.endswith('.npz'))
            with np.load(tp) as old:
                active=np.isin(old['test_labels'],np.r_[old['base_classes'],old['novel_classes']])
                bp=old['test_retained_base_prediction'][active];npred=old['test_retained_novel_prediction'][active]
                if accepted is not None:
                    margin=z['centered_margin']+z['delta'];ch=(accepted>margin)|((accepted==margin)&(npred<bp))
                    b=~z['novel_mask'];n=z['novel_mask'];w=z['weights'];y=z['labels']
                    ab=float(w[b]@((bp[b]==y[b])&~ch[b]));an=float(w[n]@((npred[n]==y[n])&ch[n]))
                    assert abs(ab-r['uncentered_direct']['old'])<1e-10 and abs(an-r['uncentered_direct']['new'])<1e-10
    return dict(complete=True,partition_order_records=len(records),new_threshold_tests_reconstructed=tests,
        weighted_auc_independently_reconstructed=True,retained_G_zero=True,retained_destroyed_to_base_zero=True)

def panel(ax,path,r,zoom=True,legend=False):
    with np.load(path.with_suffix('.npz')) as z:
        for m in COL:
            ax.plot(100*z[m+'_base'],100*z[m+'_novel'],color=COL[m],lw=1.2,marker='.',ms=1.1,label=NAME[m])
    for row in r['rows']:
        ax.scatter(100*row['base'],100*row['novel'],marker='o' if row['policy']=='paired_accuracy' else '^',
            s=27,color=COL[row['method']],facecolors='none' if row['fallback'] else COL[row['method']],zorder=4)
    target=r['rows'][0]['target'];ax.axvline(100*target,color='#777777',ls=':',lw=1)
    if zoom:
        ceiling=max(row['reference'] for row in r['rows'])
        with np.load(path.with_suffix('.npz')) as z:ceiling=max(ceiling,z['anchored_tp_base'][0])
        ax.set_xlim(max(0,100*target-3),min(100,100*ceiling+1))
    else:ax.set_xlim(0,100)
    ax.set_ylim(-2,102);ax.set_yticks([0,50,100]);ax.grid(axis='y',alpha=.15)
    ax.set_xlabel('Actual base accuracy (%)');ax.set_ylabel('Novel accuracy (%)')
    ax.set_title(('L/14' if r['encoder']=='l14' else 'B/32')+' '+DATA[r['dataset']])

def ecdf(x,w):
    order=np.argsort(x);return x[order],np.cumsum(w[order])/w.sum()

def main():
    assert json.loads((OUT/'complete.json').read_text())['complete']
    records=[(p,json.loads(p.read_text())) for p in OUT.glob('*/*/p*/o*.json')]
    assert len(records)==120
    verification=verify(records);(FIG/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    primary=[(p,r) for p,r in records if r['order']==0];summary=[];error_rows=[];retention_grid=[]
    for e in ['l14','b32']:
        for d in DATA:
            rr=sorted([r for _,r in primary if r['encoder']==e and r['dataset']==d],key=lambda r:r['part'])
            comp=[next(c for c in r['comparisons'] if c['policy']=='paired_accuracy') for r in rr]
            admitted=[c for c in comp if c['both_admitted']]
            feasible=[c for c in comp if 'envelope_gain' in c]
            floor_cases=[]
            for r,c in zip(rr,comp):
                cc=c['common_target_capacity']
                if not all(v['feasible'] for v in cc.values()):continue
                actual={v['method']:v for v in r['rows'] if v['policy']=='paired_accuracy'}
                ho=cc['retained']['novel']-actual['retained']['novel']
                hb=cc['anchored_tp']['novel']-actual['anchored_tp']['novel']
                gap=cc['retained']['novel']-cc['anchored_tp']['novel']
                assert abs(c['policy_gain']-(gap+hb-ho))<1e-10
                floor_cases.append(dict(envelope_gap=gap,retained_headroom=ho,baseline_headroom=hb,
                    headroom_difference=hb-ho,policy_gain=c['policy_gain'],
                    both_selected_feasible=all(v['base']>=v['target']-1e-10 for v in actual.values())))
            row=dict(encoder=e,dataset=d,both_admitted=len(admitted),matched_feasible=len(feasible),
                selected_gain_pp=summarize([c['policy_gain'] for c in comp]),
                selected_gain_on_matched_set_pp=summarize([c['policy_gain'] for c in feasible]),
                aligned_vs_baseline_policy_pp=summarize([c['matched_retained_minus_baseline_policy'] for c in feasible]),
                envelope_difference_at_baseline_base_pp=summarize([c['envelope_gain'] for c in feasible]),
                selected_ours_meets_baseline_base=sum(c['ours_selected_feasible_at_baseline'] for c in feasible),
                delta_auc=summarize([r['bias_auc']['delta'] for r in rr],1),
                raw_margin_auc=summarize([r['bias_auc']['raw_margin'] for r in rr],1),
                centered_margin_auc=summarize([r['bias_auc']['centered_margin'] for r in rr],1),
                centering_auc_gain=summarize([r['bias_auc']['centered_margin']-r['bias_auc']['raw_margin'] for r in rr],1),
                uncentered_direct_admitted=sum(not r['uncentered_direct']['fallback'] for r in rr),
                uncentered_direct_novel_pp=summarize([r['uncentered_direct']['new'] for r in rr]),
                centered_minus_uncentered_direct_pp=summarize([next(v['novel'] for v in r['rows'] if v['method']=='retained' and v['policy']=='paired_accuracy')-r['uncentered_direct']['new'] for r in rr]))
            row.update(common_floor_parts=len(floor_cases),common_floor_both_selected_feasible=sum(v['both_selected_feasible'] for v in floor_cases),
                common_floor_envelope_gap_pp=summarize([v['envelope_gap'] for v in floor_cases]),
                common_floor_policy_gain_pp=summarize([v['policy_gain'] for v in floor_cases]),
                common_floor_retained_headroom_pp=summarize([v['retained_headroom'] for v in floor_cases]),
                common_floor_baseline_headroom_pp=summarize([v['baseline_headroom'] for v in floor_cases]),
                common_floor_headroom_difference_pp=summarize([v['headroom_difference'] for v in floor_cases]))
            summary.append(row)
            for r in rr:
                for v in r['rows']:error_rows.append(dict(encoder=e,dataset=d,part=r['part'],**v))
            # Original published retention grid, recomputed descriptively on these
            # independently fitted models. No new policy/confidence claim at .99.
            for ratio in [.80,.85,.90,.95,.99]:
                vals=[]
                for path,r in primary:
                    if r['encoder']!=e or r['dataset']!=d:continue
                    target=ratio*r['rows'][0]['reference'];cc={}
                    with np.load(path.with_suffix('.npz')) as z:
                        for m in COL:
                            ix=np.flatnonzero(z[m+'_base']>=target-1e-10)
                            if len(ix):cc[m]=float(z[m+'_novel'][ix[-1]])
                    if len(cc)==2:vals.append(cc['retained']-cc['anchored_tp'])
                retention_grid.append(dict(encoder=e,dataset=d,retention=ratio,paired_feasible=len(vals),gain_pp=summarize(vals)))
    report=dict(scope='post-hoc frozen-model diagnostic; first order primary; ten fixed-image class partitions',
        verification=verification,conditions=summary,max_algebra_error=max(max(v['max_algebra_margin_error'] for v in r['witness'].values()) for _,r in records),
        max_test_archive_error=max(r['archive_margin_error'] for _,r in records),max_cal_archive_error=max(r['cal_archive_margin_error'] for _,r in records),
        exploratory_original_retention_grid=retention_grid)
    (FIG/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (FIG/'operating_points.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(error_rows[0]));wr.writeheader();wr.writerows(error_rows)
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(10.8,7.2))
    for i,e in enumerate(['l14','b32']):
        for j,d in enumerate(DATA):
            p,r=next((p,r) for p,r in primary if r['encoder']==e and r['dataset']==d and r['part']==0)
            panel(axes[i,j],p,r)
    handles,labels=axes[0,0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([],[],marker='o',color='#444444',ls='',label='Direct test'),Line2D([],[],marker='^',color='#444444',ls='',label='Protection test'),Line2D([],[],color='#777777',ls=':',label='95% target'),Line2D([],[],marker='o',color='#444444',markerfacecolor='none',ls='',label='Reference fallback')]
    fig.legend(handles=handles,ncol=3,loc='lower center',frameon=False)
    fig.tight_layout(rect=[0,.10,1,1])
    for ext in ['pdf','png']:fig.savefig(FIG/('operating_frontiers_p0.'+ext),dpi=300)
    plt.close(fig)
    with PdfPages(FIG/'operating_frontiers_all_full.pdf') as pdf:
        for order in [0,1]:
            for e in ['l14','b32']:
                for d in DATA:
                    fig,axes=plt.subplots(2,5,figsize=(15,6.8))
                    for part,ax in enumerate(axes.flat):
                        p,r=next((p,r) for p,r in records if (r['encoder'],r['dataset'],r['part'],r['order'])==(e,d,part,order))
                        panel(ax,p,r,zoom=False);ax.set_title(f'Partition {part}')
                    fig.suptitle(f'{e} {DATA[d]}, final session, order {order}; full discrete trajectories')
                    fig.tight_layout(rect=[0,0,1,.95]);pdf.savefig(fig);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10.8,7.2))
    for i,e in enumerate(['l14','b32']):
        for j,d in enumerate(DATA):
            p,r=next((p,r) for p,r in primary if (r['encoder'],r['dataset'],r['part'])==(e,d,0))
            ax=axes[i,j]
            with np.load(p.with_suffix('.npz')) as z:
                for mask,name,color in [(~z['novel_mask'],'Base','#1B4F72'),(z['novel_mask'],'Novel','#D55E00')]:
                    xx,yy=ecdf(z['delta'][mask],z['weights'][mask]);ax.step(xx,yy,where='post',color=color,label=name)
            ax.set_title(('L/14' if e=='l14' else 'B/32')+' '+DATA[d]);ax.set_xlabel(r'Effective bias term $\Delta d(x)$');ax.set_ylabel('Class-macro ECDF')
            ax.set_ylim(0,1.02);ax.grid(alpha=.12);ax.text(.04,.90,f"AUROC(Delta) = {r['bias_auc']['delta']:.3f}",transform=ax.transAxes,fontsize=9)
    hh,ll=axes[0,0].get_legend_handles_labels();fig.legend(hh,ll,ncol=2,loc='lower center',frameon=False)
    fig.tight_layout(rect=[0,.05,1,1])
    for ext in ['pdf','png']:fig.savefig(FIG/('effective_bias_ecdf_p0.'+ext),dpi=300)
    plt.close(fig)
    lines=['# Frozen-model mechanism diagnostics','',
        'Post-hoc diagnostic analysis; ten class partitions on fixed images, order0 primary. No new encoding or image download. Partition0 is a prespecified illustration, not the best-performing partition. The main plot zooms into the retention region; full trajectories of both orders are in operating_frontiers_all_full.pdf. Connecting lines guide the eye and are not interpolation used in matching. Hollow markers show reference fallback, which can lie outside the replaced readout\'s trajectory because it uses the reference classifier. All frontier quantities use test labels descriptively; none are deployed thresholds.','',
        '| Encoder/data | Selected gap on matched-feasible set pp | At baseline actual base, retained minus baseline selected pp | Both envelopes at that target pp | Feasible | Delta AUROC | Margin AUROC gain |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def form(v):
        if v['mean'] is None:return '--'
        if v['ci95'] is None:return f"{v['mean']:+.2f} (n={v['n']}; no interval)"
        return f"{v['mean']:+.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]"
    for r in summary:
        lines.append(f"| {r['encoder']}/{DATA[r['dataset']]} | {form(r['selected_gain_on_matched_set_pp'])} | {form(r['aligned_vs_baseline_policy_pp'])} | {form(r['envelope_difference_at_baseline_base_pp'])} | {r['matched_feasible']}/10 | {r['delta_auc']['mean']:.3f} | {r['centering_auc_gain']['mean']:+.4f} |")
    lines += ['', '## Decomposition at the original common95% minimum requirement', '',
        '| Encoder/data | Policy gap pp | Envelope gap pp | Retained oracle headroom pp | Baseline oracle headroom pp | Paired feasible | Both selected meet target |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['encoder']}/{DATA[r['dataset']]} | {form(r['common_floor_policy_gain_pp'])} | {form(r['common_floor_envelope_gap_pp'])} | {form(r['common_floor_retained_headroom_pp'])} | {form(r['common_floor_baseline_headroom_pp'])} | {r['common_floor_parts']}/10 | {r['common_floor_both_selected_feasible']}/{r['common_floor_parts']} |")
    lines += ['', 'This is a descriptive identity: policy gap = common-target envelope gap + baseline headroom - retained headroom. Headroom is oracle frontier minus the selected novel accuracy; it is nonnegative only if that selected policy meets this test target. It includes finite candidate grids, sampling, conservatism and other policy-selection effects; it is not uniquely attributable to statistical certification. The common95% minimum requirement and the baseline-realized-accuracy target in the first table are different comparisons and must not be substituted for one another.',
        '', '## Original retention grid, exploratory same-model recomputation', '',
        '| Encoder/data | Retention | Envelope gap pp | Jointly feasible |', '|---|---:|---:|---:|']
    for r in retention_grid:
        lines.append(f"| {r['encoder']}/{DATA[r['dataset']]} | {100*r['retention']:.0f}% | {form(r['gain_pp'])} | {r['paired_feasible']}/10 |")
    lines += ['', 'These five ratios are from the original published grid, but this recomputation is a post-hoc descriptive follow-up. Feasible sets may differ across ratios. These are test-label oracle capacities, not newly certified99%-retention deployment results.']
    lines += ['', 'Intervals resample partition means and condition on fixed images; no interval is displayed for fewer than five partitions. The three gain columns use the same matched-feasible set within a row. All-ten fallback-inclusive selected gaps are separately recorded in summary.json. Matched feasible sets differ between conditions; see counts. In particular, B/32 Pets has only two matched-feasible partitions and supplies a selected-set description, not evidence of a population-wide effect. Conditional admission counts remain in summary.json. Matching is conservative (at least baseline base accuracy), without interpolation. The envelope difference concerns complete fixed readouts, including fusion/normalization, not an isolated classifier-weight causal effect.',
        '', 'A_ref is identical for all methods in a partition. D is split into losses to novel labels and losses to other base labels. Retained G=0 and D_to_base=0 were checked everywhere. A signed envelope-minus-selected adjustment is not necessarily calibration regret: a selected point may not satisfy the target chosen from the other method.',
        '', 'Delta separation is exploratory and not required by the algebra. AUROC uses fixed novel-positive orientation and class-macro weights. Compare the combined margin AUROC, not Delta alone. Removing centering is recalibrated on the same independent images; fusion weights and query scales are fixed, so it is not a fully retuned alternative architecture.',
        '',f"Verification: {len(records)} partition/order records; {verification['new_threshold_tests_reconstructed']} new threshold tests independently reconstructed; all recorded source hashes checked; max algebraic float error {report['max_algebra_error']:.3g}. Original calibration archives preserve cached batch rounding differences; max reconstructed difference {report['max_cal_archive_error']:.3g}, within their recorded5e-4 witness tolerance.",
        '', 'These diagnostics are included in Supplement P with a corrected main-text attribution. They have no new completed fresh-agent integrity audit: the reviewer stopped at a model usage limit before the final semantic report. Partial independent deterministic receipts are retained privately. The preceding manuscript audit does not automatically cover this new analysis.']
    (FIG/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report|dict(conditions=[{k:v for k,v in r.items() if k in ['encoder','dataset','selected_gain_pp','aligned_vs_baseline_policy_pp','envelope_difference_at_baseline_base_pp','delta_auc','centering_auc_gain','centered_minus_uncentered_direct_pp']} for r in summary]),indent=2))

if __name__=='__main__':main()
