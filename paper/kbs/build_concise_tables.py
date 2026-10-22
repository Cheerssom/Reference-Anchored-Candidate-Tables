"""Concise displays derived from the complete, checked result artifacts."""
from pathlib import Path
import json

H=Path(__file__).resolve().parent


def main():
    output=[];rows=0
    for line in (H/'table_revision_paired.tex').read_text().splitlines():
        if line.startswith(r'\begin{tabular}'):
            output.append(r'\begin{tabular}{lllr}')
        elif line.startswith('Encoder &'):
            output.append(r'Encoder & Dataset & Comparator & Avg. gain [95\% CI] \\')
        elif line.count('&')==4:
            output.append(' & '.join(line.split(' & ')[:4])+r' \\');rows+=1
        else:output.append(line)
    assert rows==18
    (H/'table_concise_paired.tex').write_text('\n'.join(output)+'\n')
    summary=json.loads((H/'independent_risk_summary.json').read_text())['final']
    names={'retained':'Retained','no_learned_tuned':'Recalibrated removal',
           'anchored_tp':'Dense prototypes','raw':'Raw fusion',
           'fixed_temperature':'Fixed temperature','tip_mean_calibrated':'Class-mean cache'}
    output=[r'\begin{tabular}{llrrrr}',r'\toprule',
            r'Encoder & Readout & Exp. & Test met & Base & Novel \\',r'\midrule']
    for encoder in ['l14','b32']:
        for method,name in names.items():
            row=next(r for r in summary if
                     (r['encoder'],r['dataset'],r['method'],r['policy'],r['level'])==
                     (encoder,'imagenetr',method,'risk',.05))
            enc='L/14' if encoder=='l14' else 'B/32'
            output.append(f"{enc} & {name} & {row['expanded']}/10 & {row['test_met']}/10 & "
                          f"{100*row['old_mean']:.2f} & {100*row['new_mean']:.2f} \\\\")
        output.append(r'\midrule')
    output[-1]=r'\bottomrule';output.append(r'\end{tabular}')
    (H/'table_concise_risk.tex').write_text('\n'.join(output)+'\n')
    print('Derived 18 paired rows and 12 risk rows; complete artifacts preserved.')


if __name__=='__main__':main()
