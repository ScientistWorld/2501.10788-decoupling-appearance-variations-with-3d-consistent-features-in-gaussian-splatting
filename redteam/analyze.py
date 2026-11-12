"""Report metric changes and an explicitly cohort-dependent audit rank.

Uses metric directions, coefficient magnitudes, and experiment weights. Units
cancel through the across-attempt population SD. This cannot predict the SD of
unknown future attempts, so Pareto wins/ties are reported separately in REDTEAM.
"""
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def flatten(payload):
    return {(e, name, m): v for e, data in payload['experiments'].items()
            for name, values in data['results'].items() for m, v in values.items()
            if isinstance(v, (int, float))}


def main():
    specs = json.loads((HERE/'reference_metric_spec.json').read_text())['experiments']
    base = flatten(json.loads((HERE/'reference_original_scores_test.json').read_text()))
    attempts = {}
    records = {}
    for folder in sorted(HERE.iterdir()):
        path = folder/'scores_test.json'
        if not path.exists():
            continue
        record = json.loads((folder/'run_record.json').read_text())
        records[folder.name] = record
        if record.get('test_exit_code') == 0:
            attempts[folder.name] = flatten(json.loads(path.read_text()))
    scales = {}
    for key, value in base.items():
        pool = [value] + [d[key] for d in attempts.values() if key in d and math.isfinite(d[key])]
        scales[key] = statistics.pstdev(pool)
    rows = {}
    for case, vals in attempts.items():
        terms = {}
        invalid = [list(k) for k,v in vals.items() if not math.isfinite(v)]
        for key, value in vals.items():
            if key not in base or key[2] not in specs[key[0]]['metrics'] or not math.isfinite(value):
                continue
            sd = scales[key]
            if sd == 0:
                continue
            md = specs[key[0]]['metrics'][key[2]]
            term = specs[key[0]]['weight'] * abs(md['coefficient']) * (1 if md['higher_is_better'] else -1) * (value-base[key]) / sd
            terms['/'.join(key)] = term
        proposed = vals.get(('synthetic_glav_core', 'davigs', 'psnr'))
        rows[case] = dict(audit_rank_delta=None if invalid else sum(terms.values()),
                          davigs_rank_delta=None if invalid else sum(v for k,v in terms.items() if '/davigs/' in k),
                          davigs_core={m:v for (e,n,m),v in vals.items() if e=='synthetic_glav_core' and n=='davigs'},
                          terms=terms, invalid_metrics=invalid,
                          missing_reference_keys=[list(k) for k in base if k not in vals])
    seed_cases = [f'reference_seed_{i}' for i in [7, 8, 9]]
    spreads = {}
    if all(case in rows for case in seed_cases):
        for m in specs['synthetic_glav_core']['metrics']:
            values = [rows[case]['davigs_core'][m] for case in seed_cases]
            spreads[m] = dict(min=min(values), max=max(values), range=max(values)-min(values), population_sd=statistics.pstdev(values))
        values = [rows[case]['audit_rank_delta'] for case in seed_cases]
        spreads['audit_rank_delta'] = dict(min=min(values), max=max(values), range=max(values)-min(values), population_sd=statistics.pstdev(values))
    out = dict(rank_definition='Sum over matched experiment/method/metric entries; reference is original test scores; weight*abs(coefficient)*direction*delta/population_SD. SD cohort is all finite successful audit attempts plus reference, separately per entry. Missing entries skipped. Future-cohort rank is not knowable.',
               cases=rows, seed_spread=spreads,
               failed_cases={p.name: json.loads((p/'run_record.json').read_text()) for p in HERE.iterdir() if (p/'run_record.json').exists() and p.name not in rows})
    (HERE/'audit_summary.json').write_text(json.dumps(out, indent=2, sort_keys=True)+'\n')
    print('case | audit rank delta | PSNR | SSIM | proxy | worst PSNR | bias | edge | coupled | steps')
    for case,row in rows.items():
        m=row['davigs_core']
        score=row['audit_rank_delta']
        print(case, '|', f'{score:.6f}' if score is not None else 'INVALID', '|', ' | '.join(f'{m[k]:.6f}' if k in m else 'MISSING' for k in ['psnr','ssim','lpips_proxy','worst_view_psnr','mean_color_bias_mae','edge_mae','coupled_appearance_at_splat_render','train_iterations']))
    print('SEED SPREAD',json.dumps(spreads,sort_keys=True))


if __name__=='__main__':
    main()
