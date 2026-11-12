"""Checkpointed honest variants, preserving the reference's exact protocol."""
from pathlib import Path
import math
import torch
import torch.nn.functional as F
from method.davigs_model import AppearanceModule, GaussianSplatScene, identity_regularizer, reconstruction_loss
from method.run_experiment import lambda_id, left_half, right_half


def local_loss(pred, target):
    x, y = pred[None], target[None]
    kernel = torch.ones((3, 1, 7, 7), dtype=x.dtype, device=x.device) / 49
    avg = lambda v: F.conv2d(v, kernel, padding=3, groups=3)
    mx, my = avg(x), avg(y)
    vx, vy, cov = avg(x*x)-mx*mx, avg(y*y)-my*my, avg(x*y)-mx*my
    sim = ((2*mx*my+0.01**2)*(2*cov+0.03**2))/((mx*mx+my*my+0.01**2)*(vx+vy+0.03**2))
    return 0.8*(pred-target).abs().mean() + 0.2*(1-sim).mean()/2


def train(scene, out: Path, checkpoint: Path, steps=2200, adapt_steps=400,
          decay=False, local_ssim=False, baseline=False):
    gs = GaussianSplatScene(scene.init_points, scene.init_colors)
    app = None if baseline else AppearanceModule(num_views=len(scene.cameras))
    params = list(gs.parameters()) + ([] if baseline else list(app.parameters()))
    opt = torch.optim.Adam(params, lr=0.025 if baseline else 0.018)
    loss_fn = local_loss if local_ssim else reconstruction_loss
    state = dict(phase='train', step=0, view=0, adapt_step=0, preds=[], targets=[])
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        gs.load_state_dict(saved['gs'])
        if app is not None:
            app.load_state_dict(saved['app'])
        opt.load_state_dict(saved['opt'])
        torch.set_rng_state(saved['rng'])
        state = saved['state']
        print(f'RESUME {checkpoint.parent.name} phase={state["phase"]} step={state["step"]} view={state["view"]} adapt={state["adapt_step"]}', flush=True)

    def persist(emb_opt=None):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        saved = dict(gs=gs.state_dict(), app=None if app is None else app.state_dict(),
                     opt=opt.state_dict(), rng=torch.get_rng_state(), state=state,
                     emb_opt=None if emb_opt is None else emb_opt.state_dict())
        temp = checkpoint.with_suffix('.tmp')
        torch.save(saved, temp)
        temp.replace(checkpoint)

    if state['phase'] == 'train':
        for step in range(state['step'], steps):
            if decay:
                lr = 0.018 * (0.1+0.9*(1+math.cos(math.pi*step/steps))/2)
                for group in opt.param_groups:
                    group['lr'] = lr
            vid = scene.train_ids[step % len(scene.train_ids)]
            raw, depth = gs(scene.cameras[vid])
            if app is None:
                loss = loss_fn(raw, scene.images[vid])
            else:
                pred, mats = app(raw, depth, scene.cameras[vid], vid, cell_size=8)
                loss = loss_fn(pred, scene.images[vid]) + lambda_id(step, steps)*identity_regularizer(mats)
            opt.zero_grad()
            loss.backward()
            opt.step()
            state['step'] = step+1
            if (step+1) % 100 == 0 or step+1 == steps:
                persist()
                print(f'TRAIN {checkpoint.parent.name} {step+1}/{steps} loss={float(loss.detach()):.6f}', flush=True)
        state['phase'] = 'adapt'
        persist()

    for module in [gs] + ([] if app is None else [app.grid, app.mlp]):
        for param in module.parameters():
            param.requires_grad_(False)
    for view_index in range(state['view'], len(scene.test_ids)):
        vid = scene.test_ids[view_index]
        with torch.no_grad():
            raw, depth = gs(scene.cameras[vid])
        if app is not None:
            emb = app.embeddings.weight
            opt_emb = torch.optim.Adam([emb], lr=0.03)
            if state['adapt_step']:
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                opt_emb.load_state_dict(saved['emb_opt'])
            for a in range(state['adapt_step'], adapt_steps):
                pred, mats = app(raw, depth, scene.cameras[vid], vid, cell_size=8)
                loss = loss_fn(left_half(pred), left_half(scene.images[vid])) + 0.01*identity_regularizer(mats)
                opt_emb.zero_grad()
                loss.backward()
                emb.grad[:min(scene.test_ids)] = 0
                opt_emb.step()
                state['adapt_step'] = a+1
                if (a+1) % 100 == 0:
                    persist(opt_emb)
            with torch.no_grad():
                pred, _ = app(raw, depth, scene.cameras[vid], vid, cell_size=8)
        else:
            pred = raw
        state['preds'].append(right_half(pred).detach().clone())
        state['targets'].append(right_half(scene.images[vid]).clone())
        state['view'] = view_index+1
        state['adapt_step'] = 0
        persist()
        print(f'ADAPT {checkpoint.parent.name} view={view_index+1}/{len(scene.test_ids)}', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(pred_right=torch.stack(state['preds']), target_right=torch.stack(state['targets']),
                    coupled_appearance_at_splat_render=0., train_iterations=float(steps)), out/'predictions.pt')
