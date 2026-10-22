"""Exploratory validation-target tightening; no deployment certificate claimed."""
from pathlib import Path
import json
import numpy as np
import fcs_retention_value as rv
from kbs_runtime import atomic_json, check_pause, digest_file

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'experiments/kbs_revision'
OUT = ROOT / 'experiments/kbs_retention_safety_diagnostic'
RATIOS = (.95, .97, .99, 1.)
METHODS = ('retained', 'no_learned_tuned', 'anchored_tp', 'tip_mean_calibrated')
PROTOCOL = ROOT / 'experiments/kbs_retention_safety_diagnostic_protocol.md'


def compact(z, method, validation=False):
    prefix = ('validation_' if validation else '') + method + '_'
    y = z['validation_labels' if validation else 'labels']
    base, novel = z['base_classes'], z['novel_classes']
    c = {k: z[prefix+k] for k in ('margin', 'base_prediction', 'novel_prediction',
                                'base_correct', 'novel_correct', 'weights')}
    c.update(base_mask=np.isin(y, base), novel_mask=np.isin(y, novel),
             nb=len(base), nn=len(novel))
    return c


def main():
    for encoder in ('l14', 'b32'):
        for dataset in ('cifar', 'pets', 'imagenetr'):
            for part in range(10):
                check_pause()
                source = INPUT / encoder / dataset / f'p{part}'
                files = [source/'o0_t5.npz', source/'o0_t5.json', source/'complete.json']
                signature = {str(p.relative_to(ROOT)): digest_file(p)
                             for p in files + [Path(__file__), Path(rv.__file__), PROTOCOL]}
                target = OUT / encoder / dataset / f'p{part}.json'
                if target.exists():
                    assert json.loads(target.read_text())['input_hashes'] == signature
                    continue
                original = json.loads(files[1].read_text())['records']
                reference = json.loads(files[2].read_text())
                ref_val = reference['reference_closed_validation']
                ref_test = reference['reference_closed_test']
                rows = []
                with np.load(files[0]) as z:
                    n_val = len(z['validation_labels'])
                    for method in METHODS:
                        tc, vc = compact(z, method), compact(z, method, True)
                        previous = None
                        for ratio in RATIOS:
                            offset, ceiling = rv.threshold(vc, ratio*ref_val)
                            row = dict(encoder=encoder, dataset=dataset, part=part,
                                       method=method, validation_ratio=ratio,
                                       test_ratio=.95, feasible=offset is not None,
                                       validation_queries=n_val,
                                       validation_ceiling=ceiling)
                            if offset is not None:
                                test = rv.evaluate(tc, offset)
                                assert rv.evaluate(vc, offset)['old'] >= ratio*ref_val-1e-10
                                if previous is not None:
                                    # Tightening the old-class constraint cannot increase
                                    # novel admission or decrease base accuracy.
                                    assert offset <= previous['offset']
                                    assert test['old'] >= previous['old']-1e-10
                                    assert test['new'] <= previous['new']+1e-10
                                row.update(offset=offset, **test,
                                           test_target_met=test['old'] >= .95*ref_test-1e-10,
                                           test_retention_ratio=test['old']/ref_test,
                                           target_gap_pp=100*(test['old']-.95*ref_test))
                                previous = row
                            if ratio == .95:
                                old = next(r for r in original if r['method'] == method
                                           and r['evaluation'] == 'base_validation'
                                           and r['ratio'] == .95)
                                assert old['feasible'] == row['feasible']
                                if row['feasible']:
                                    for key in ('old', 'new', 'offset'):
                                        assert abs(old[key]-row[key]) < 1e-10
                                    assert old['test_target_met'] == row['test_target_met']
                            rows.append(row)
                atomic_json(dict(input_hashes=signature, records=rows), target)
    rows = []
    for encoder in ('l14', 'b32'):
        for dataset in ('cifar', 'pets', 'imagenetr'):
            for part in range(10):
                rows += json.loads((OUT/encoder/dataset/f'p{part}.json').read_text())['records']
    summary = []
    for encoder in ('l14', 'b32'):
        for dataset in ('cifar', 'pets', 'imagenetr'):
            for method in METHODS:
                group = [r for r in rows if (r['encoder'], r['dataset'], r['method']) ==
                         (encoder, dataset, method)]
                original = {r['part']: r for r in group if r['validation_ratio'] == .95}
                for ratio in RATIOS:
                    selected = [r for r in group if r['validation_ratio'] == ratio and r['feasible']]
                    paired = [r for r in selected if original[r['part']]['feasible']]
                    s = dict(encoder=encoder, dataset=dataset, method=method,
                             validation_ratio=ratio, partitions=10, feasible=len(selected),
                             test_met=sum(r['test_target_met'] for r in selected),
                             paired_partitions=len(paired))
                    for key in ('old', 'new', 'test_retention_ratio', 'target_gap_pp'):
                        s[key+'_mean'] = float(np.mean([r[key] for r in selected])) if selected else None
                    s['paired_novel_delta_pp'] = float(np.mean([
                        100*(r['new']-original[r['part']]['new']) for r in paired])) if paired else None
                    summary.append(s)
    atomic_json(dict(status='exploratory_not_a_certificate', records=len(rows),
                     original_policy_reconstruction='passed',
                     monotonicity_checks='passed', summary=summary), OUT/'summary.json')
    print(json.dumps([r for r in summary if r['method'] == 'retained'], indent=2))


if __name__ == '__main__':
    main()
