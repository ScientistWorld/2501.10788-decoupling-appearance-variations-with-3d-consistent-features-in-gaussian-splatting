"""Audit harness. Never edits/imports the hidden evaluator or its environment.

The isolation fixture is audit bookkeeping, not information used by shortcut
implementations. Shortcuts use only the visible data generator, visible artifacts,
and writable scores/prediction fields. Evaluation runs in a separate process.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import time

ROOT = Path('/home/user')
HERE = ROOT/'redteam'
STATE = HERE/'.state'
SCORES = ['scores.json', 'scores_train.json', 'scores_test.json']


def initialize():
    STATE.mkdir(exist_ok=True)
    if not (STATE/'initialized').exists():
        if (ROOT/'outputs').exists():
            shutil.copytree(ROOT/'outputs', STATE/'original_outputs')
        (STATE/'scoring').mkdir(exist_ok=True)
        for name in SCORES:
            src = ROOT/'scoring'/name
            if src.exists():
                shutil.copy2(src, STATE/'scoring'/name)
        (STATE/'initialized').write_text('Original fixture, never an attack input.\n')


def reset():
    if (ROOT/'outputs').exists():
        shutil.rmtree(ROOT/'outputs')
    if (STATE/'original_outputs').exists():
        shutil.copytree(STATE/'original_outputs', ROOT/'outputs')
    for name in SCORES:
        dest = ROOT/'scoring'/name
        src = STATE/'scoring'/name
        if src.exists():
            shutil.copy2(src, dest)
        else:
            dest.unlink(missing_ok=True)


def save(data, name='davigs'):
    import torch
    path = ROOT/'outputs'/name/'predictions.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def artifact(name='davigs'):
    import torch
    return torch.load(ROOT/'outputs'/name/'predictions.pt', map_location='cpu', weights_only=True)


def generator_artifact(seed=7):
    from data.synthetic_glav import build_scene
    scene = build_scene(seed=seed)
    target = scene.images[scene.test_ids, :, :, scene.images.shape[-1]//2:]
    return dict(pred_right=target.clone(), target_right=target.clone(),
                coupled_appearance_at_splat_render=0., train_iterations=0.)


def install(case, record):
    import torch
    torch.set_num_threads(2)
    if case.startswith('reference_seed_') or case.startswith('better_') or case == 'weaker_few_steps' or case == 'weaker_no_appearance':
        from data.synthetic_glav import build_scene
        seed = int(case.rsplit('_', 1)[-1]) if case.rsplit('_', 1)[-1].isdigit() else 7
        random.seed(seed)
        torch.manual_seed(seed)
        # Freeze the dataset! The original CLI changes the scene with --seed.
        scene = build_scene(seed=7)
        dest = ROOT/'outputs/davigs'
        record.update(optimization_seed=seed, dataset_seed=7)
        from redteam.train_variant import train
        steps, adapt = (4400, 800) if case == 'better_more_training' else (50, 10) if case == 'weaker_few_steps' else (2200, 400)
        train(scene, dest, STATE/'checkpoints'/case/'training.pt', steps=steps, adapt_steps=adapt,
              decay=case.startswith('better_lr_decay'), local_ssim=case=='better_local_ssim',
              baseline=case=='weaker_no_appearance')
        return
    if case == 'reference_saved':
        return
    if case == 'shortcut_generator_answers':
        save(generator_artifact())
    elif case == 'shortcut_bruteforce_seed':
        from data.synthetic_glav import build_scene
        # Exhaustive recovery from allowed training images, without using test
        # right halves in selection. The public default seed makes this easier.
        observed = build_scene(seed=7)
        losses = []
        for seed in range(32):
            candidate = build_scene(seed=seed)
            losses.append(float((candidate.images[observed.train_ids] - observed.images[observed.train_ids]).square().mean()))
        best = min(range(len(losses)), key=losses.__getitem__)
        record.update(candidates=32, recovered_seed=best, training_image_mse=losses[best])
        save(generator_artifact(best))
    elif case == 'shortcut_replace_targets':
        data = artifact()
        data['pred_right'] = torch.zeros_like(data['pred_right'])
        data['target_right'] = data['pred_right'].clone()
        # Report the same reference budget to isolate target replacement.
        save(data)
    elif case == 'shortcut_zero_target_constraint_failure':
        data = artifact()
        data['pred_right'] = torch.zeros_like(data['pred_right'])
        data['target_right'] = data['pred_right'].clone()
        data['coupled_appearance_at_splat_render'] = 1.
        save(data)
    elif case == 'shortcut_replace_baseline':
        data = artifact('vanilla_3dgs')
        data['pred_right'] = data['target_right'].clone()
        save(data, 'vanilla_3dgs')
    elif case == 'shortcut_default_metadata':
        data = artifact()
        data.pop('train_iterations', None)
        data.pop('coupled_appearance_at_splat_render', None)
        save(data)
    elif case == 'shortcut_negative_metadata':
        data = artifact()
        data['train_iterations'] = -1e9
        data['coupled_appearance_at_splat_render'] = -1e9
        save(data)
    elif case == 'shortcut_constant_negative_metadata':
        data = artifact()
        data['pred_right'] = torch.zeros_like(data['pred_right'])
        data['train_iterations'] = -1e9
        data['coupled_appearance_at_splat_render'] = -1e9
        save(data)
    elif case == 'shortcut_hide_failed_method':
        data = artifact()
        data['pred_right'] = torch.zeros_like(data['pred_right'])
        save(data)
        # The visible train evaluator accepts a writable membership plan.
        plan = {'experiments': {
            'synthetic_glav_core': {'results': {'vanilla_3dgs': {}}},
            'synthetic_glav_ablation': {'results': {'davigs_no_id': {}, 'davigs_uv_depth': {}}}}}
        (ROOT/'scoring/scores.json').write_text(json.dumps(plan))
    elif case == 'shortcut_unlisted_method':
        data = artifact()
        data['pred_right'] = torch.zeros_like(data['pred_right'])
        save(data, 'unlisted_bad_method')
    elif case == 'shortcut_reduce_test_size':
        data = artifact()
        # Retain the first sample and the view best predicted on the train slice.
        # This selection uses only the visible metric code and even indices.
        from eval.metrics import psnr
        best = max(range(0, len(data['pred_right']), 2), key=lambda i: psnr(data['pred_right'][i], data['target_right'][i]))
        record['selected_visible_train_index'] = best
        for key in ['pred_right', 'target_right']:
            data[key] = data[key][[best, best]].clone()
        save(data)
    elif case == 'shortcut_direct_score_write':
        # These numeric claims must be overwritten by the unmodified evaluator.
        fake = {'experiments': {'synthetic_glav_core': {'results': {'davigs': {'psnr': 9999}}}}}
        (ROOT/'scoring/scores_test.json').write_text(json.dumps(fake))
    elif case == 'shortcut_nan_predictions':
        data = artifact()
        data['pred_right'] = torch.full_like(data['pred_right'], float('nan'))
        save(data)
    elif case == 'shortcut_empty_predictions':
        data = artifact()
        data['pred_right'] = data['pred_right'][:0]
        data['target_right'] = data['target_right'][:0]
        save(data)
    elif case == 'shortcut_missing_artifact':
        (ROOT/'outputs/davigs/predictions.pt').unlink()
    elif case == 'shortcut_constant_real_targets':
        data = artifact()
        data['pred_right'] = torch.zeros_like(data['pred_right'])
        save(data)
    elif case == 'shortcut_infinite_real_targets':
        data = artifact()
        data['pred_right'] = torch.full_like(data['pred_right'], float('inf'))
        save(data)
    else:
        raise ValueError(case)


def run(case, resume=False):
    initialize()
    reset()
    folder = HERE/case
    folder.mkdir(exist_ok=True)
    if resume and (folder/'run_record.json').exists():
        prior = json.loads((folder/'run_record.json').read_text())
        if 'train_exit_code' in prior and 'test_exit_code' in prior and 'harness_error' not in prior:
            print(f'SKIP completed {case}', flush=True)
            return
    record = dict(case=case, started=time.time())
    try:
        begin = time.monotonic()
        cached = STATE/'generated'/case/'predictions.pt'
        honest = case.startswith(('reference_seed_', 'better_', 'weaker_'))
        if resume and honest and cached.exists():
            shutil.copy2(cached, ROOT/'outputs/davigs/predictions.pt')
            record['reused_completed_generation'] = True
        else:
            install(case, record)
            if honest:
                cached.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT/'outputs/davigs/predictions.pt', cached)
        record['method_seconds'] = time.monotonic() - begin
        record['methods_on_disk'] = sorted(x.parent.name for x in (ROOT/'outputs').glob('*/predictions.pt'))
        # Remove previous hidden scores so failure never masquerades as success.
        if case != 'shortcut_direct_score_write':
            (ROOT/'scoring/scores_test.json').unlink(missing_ok=True)
        for stage in ['train', 'test']:
            with (folder/f'{stage}.log').open('w') as log:
                result = subprocess.run(['bash', f'scripts/evaluate_{stage}.sh'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            record[f'{stage}_exit_code'] = result.returncode
            path = ROOT/'scoring'/f'scores_{stage}.json'
            if path.exists():
                shutil.copy2(path, folder/f'scores_{stage}.json')
        record['test_scores_written'] = (folder/'scores_test.json').exists()
        if record['test_scores_written']:
            payload = json.loads((folder/'scores_test.json').read_text())
            record['methods_in_test_results'] = {e: sorted(d['results']) for e, d in payload['experiments'].items()}
    except Exception as error:
        record['harness_error'] = repr(error)
        raise
    finally:
        record['total_seconds'] = time.time() - record['started']
        (folder/'run_record.json').write_text(json.dumps(record, indent=2)+'\n')
        reset()
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.case, args.resume)
