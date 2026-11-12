"""Honest optimization variants; same fixed data and half-image protocol."""
from pathlib import Path
import math
import torch
from method.davigs_model import AppearanceModule, GaussianSplatScene, identity_regularizer, reconstruction_loss
from method.run_experiment import lambda_id, left_half, right_half


def train_decay(scene, out: Path, steps=2200, adapt_steps=400):
    gs = GaussianSplatScene(scene.init_points, scene.init_colors)
    app = AppearanceModule(num_views=len(scene.cameras))
    opt = torch.optim.Adam(list(gs.parameters()) + list(app.parameters()), lr=0.018)
    for step in range(steps):
        # A genuine optimizer change, with the reference's exact loss/model/data.
        lr = 0.018 * (0.1 + 0.9 * (1 + math.cos(math.pi * step / steps)) / 2)
        for group in opt.param_groups:
            group['lr'] = lr
        vid = scene.train_ids[step % len(scene.train_ids)]
        raw, depth = gs(scene.cameras[vid])
        pred, mats = app(raw, depth, scene.cameras[vid], vid, cell_size=8)
        loss = reconstruction_loss(pred, scene.images[vid]) + lambda_id(step, steps) * identity_regularizer(mats)
        opt.zero_grad()
        loss.backward()
        opt.step()
    for module in [gs, app.grid, app.mlp]:
        for param in module.parameters():
            param.requires_grad_(False)
    preds, targets = [], []
    for vid in scene.test_ids:
        # Rendering is frozen during embedding adaptation; cache identical inputs.
        with torch.no_grad():
            raw, depth = gs(scene.cameras[vid])
        emb = app.embeddings.weight
        opt_emb = torch.optim.Adam([emb], lr=0.03)
        for _ in range(adapt_steps):
            pred, mats = app(raw, depth, scene.cameras[vid], vid, cell_size=8)
            loss = reconstruction_loss(left_half(pred), left_half(scene.images[vid])) + 0.01 * identity_regularizer(mats)
            opt_emb.zero_grad()
            loss.backward()
            emb.grad[:min(scene.test_ids)] = 0
            opt_emb.step()
        with torch.no_grad():
            pred, _ = app(raw, depth, scene.cameras[vid], vid, cell_size=8)
            preds.append(right_half(pred))
            targets.append(right_half(scene.images[vid]))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(pred_right=torch.stack(preds), target_right=torch.stack(targets),
                    coupled_appearance_at_splat_render=0., train_iterations=float(steps)), out/'predictions.pt')
