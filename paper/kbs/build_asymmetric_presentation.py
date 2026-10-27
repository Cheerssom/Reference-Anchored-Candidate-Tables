"""Build recognition-first main tables from frozen prediction archives.

No fitting, support selection, calibration or prediction is rerun. The first
table reaggregates the original fixed-gate predictions for all four readouts.
The admission table uses the same direct paired certificate for every method.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
DATA = {'cifar': 'CIFAR-100', 'pets': 'Pets', 'imagenetr': 'ImageNet-R'}
ENC = {'l14': 'CLIP ViT-L/14', 'b32': 'CLIP ViT-B/32'}
METHODS = {'retained': 'Retained readout', 'anchored_tp': 'Dense prototypes',
           'fixed_temperature': 'Fixed temperature', 'hycal': 'HyCal transfer'}


def evaluate(z, method, offset):
    y = z['test_labels']
    base, novel = z['base_classes'], z['novel_classes']
    prefix = 'test_' + method + '_'
    margin = z[prefix + 'margin']
    bp, npred = z[prefix + 'base_prediction'], z[prefix + 'novel_prediction']
    choose = (offset > margin) | ((offset == margin) & (npred < bp))
    prediction = np.where(choose, npred, bp)
    return float(np.mean([np.mean(prediction[y == c] == c)
                          for c in np.r_[base, novel]]))


def build():
    source = H / 'reviewer_revision_summary.json'
    revision = json.loads(source.read_text())
    averages, hashes = {}, {source.relative_to(ROOT).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest()}
    for e in ENC:
        for d in DATA:
            values = {m: [] for m in METHODS}
            for part in range(10):
                folder = ROOT / f'experiments/kbs_independent_risk/{e}/{d}/p{part}'
                selection = folder / 'selection.json'
                params = json.loads(selection.read_text())['parameters']
                hashes[selection.relative_to(ROOT).as_posix()] = hashlib.sha256(selection.read_bytes()).hexdigest()
                cells = {m: [] for m in METHODS if m != 'hycal'}
                for order in [0, 1]:
                    for session in range(1, 6):
                        path = folder / f'o{order}_t{session}.npz'
                        hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
                        with np.load(path) as z:
                            for method in cells:
                                cells[method].append(evaluate(z, method, params[method][2]))
                for method in cells:
                    values[method].append(float(np.mean(cells[method])))
            for method in ['retained', 'hycal']:
                original = next(r['average'] for r in revision['normalization']
                                if (r['encoder'], r['dataset'], r['method']) == (e, d, method))
                if method == 'retained':
                    assert np.allclose(values[method], original['values'], atol=1e-12)
                else:
                    values[method] = original['values']
            for method, v in values.items():
                averages[e, d, method] = {'mean': float(100 * np.mean(v)),
                                          'sd': float(100 * np.std(v, ddof=1)),
                                          'partition_values': v}
            print(f'Checked fixed-gate recognition: {e}/{d}', flush=True)

    lines = [r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
             r'Readout & CIFAR-100 & Pets & ImageNet-R \\', r'\midrule']
    for e, encoder in ENC.items():
        lines.append(r'\multicolumn{4}{@{}l}{\textnormal{' + encoder + r'}} \\')
        for m, label in METHODS.items():
            cells = []
            for d in DATA:
                row = averages[e, d, m]
                value = f"{row['mean']:.2f}"
                if round(row['mean'], 2) == max(round(averages[e, d, mm]['mean'], 2) for mm in METHODS):
                    value = r'\textbf{' + value + '}'
                cells.append(value + r'\,\ensuremath{\pm}\,' + f"{row['sd']:.2f}")
            lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    (H / 'table_asymmetric_recognition.tex').write_text('\n'.join(lines) + '\n')

    policy = {(r['encoder'], r['dataset'], r['method']): r for r in revision['policy']
              if r['policy'] == 'paired_accuracy'}
    lines = [r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
             r'Readout & CIFAR-100 & Pets & ImageNet-R \\', r'\midrule']
    for e, encoder in ENC.items():
        lines.append(r'\multicolumn{4}{@{}l}{\textnormal{' + encoder + r'}} \\')
        for m, label in METHODS.items():
            cells = []
            for d in DATA:
                row = policy[e, d, m]
                value = f"{100 * row['novel']['mean']:.2f}"
                if round(100 * row['novel']['mean'], 2) == max(round(100 * policy[e, d, mm]['novel']['mean'], 2) for mm in METHODS):
                    value = r'\textbf{' + value + '}'
                cells.append(value + f" ({row['finite']}/10)")
            lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    (H / 'table_asymmetric_admission.tex').write_text('\n'.join(lines) + '\n')
    centering = (H / 'table_centering_recalibrated.tex').read_text()
    (H / 'table_asymmetric_centering.tex').write_text(
        centering.replace(r'\textit{ViT-', r'\textnormal{CLIP ViT-'))
    report = {'scope': 'Original independently fitted source models; identical fixed-gate aggregation and common direct-test admission.',
              'prediction_archives': 600, 'averages': [{'encoder': e, 'dataset': d, 'method': m, **v}
                  for (e, d, m), v in averages.items()], 'source_sha256': hashes,
              'retained_aggregation_matches_existing_summary': True,
              'admission_conditions': len(ENC) * len(DATA) * len(METHODS)}
    (H / 'asymmetric_presentation_verification.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    build()
