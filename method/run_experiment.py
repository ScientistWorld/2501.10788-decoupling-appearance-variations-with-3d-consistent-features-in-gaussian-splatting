#!/usr/bin/env python3
"""Run the reduced-scale DAVIGS reproduction and save predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch

from data.synthetic_glav import build_scene
from method.davigs_model import AppearanceModule, GaussianSplatScene, identity_regularizer, reconstruction_loss


def right_half(x: torch.Tensor) -> torch.Tensor:
    return x[..., x.shape[-1] // 2 :]


def left_half(x: torch.Tensor) -> torch.Tensor:
    return x[..., : x.shape[-1] // 2]


def lambda_id(step: int, total: int) -> float:
    warm = min(1.0, step / max(1, int(0.25 * total)))
    decay = 0.2 + 0.1 * 0.5 * (1.0 + torch.cos(torch.tensor(step / max(1, total) * 3.14159265))).item()
    return warm * decay


def train_baseline(scene, steps: int, out_dir: Path) -> None:
    model = GaussianSplatScene(scene.init_points, scene.init_colors)
    opt = torch.optim.Adam(model.parameters(), lr=0.025)
    for step in range(steps):
        vid = scene.train_ids[step % len(scene.train_ids)]
        pred, _ = model(scene.cameras[vid])
        loss = reconstruction_loss(pred, scene.images[vid])
        opt.zero_grad()
        loss.backward()
        opt.step()
    preds, targets = [], []
    with torch.no_grad():
        for vid in scene.test_ids:
            pred, _ = model(scene.cameras[vid])
            preds.append(right_half(pred))
            targets.append(right_half(scene.images[vid]))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pred_right": torch.stack(preds).cpu(),
            "target_right": torch.stack(targets).cpu(),
            "coupled_appearance_at_splat_render": 0.0,
            "train_iterations": float(steps),
        },
        out_dir / "predictions.pt",
    )


def train_davigs(scene, steps: int, adapt_steps: int, out_dir: Path) -> None:
    gs = GaussianSplatScene(scene.init_points, scene.init_colors)
    app = AppearanceModule(num_views=len(scene.cameras))
    opt = torch.optim.Adam(list(gs.parameters()) + list(app.parameters()), lr=0.018)
    for step in range(steps):
        vid = scene.train_ids[step % len(scene.train_ids)]
        raw, depth = gs(scene.cameras[vid])
        pred, mats = app(raw, depth, scene.cameras[vid], vid, cell_size=8)
        loss = reconstruction_loss(pred, scene.images[vid]) + lambda_id(step, steps) * identity_regularizer(mats)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Match the paper's test protocol at small scale: optimize only the test
    # view's appearance embedding on the left half, then score the right half.
    for param in gs.parameters():
        param.requires_grad_(False)
    for param in app.grid.parameters():
        param.requires_grad_(False)
    for param in app.mlp.parameters():
        param.requires_grad_(False)

    preds, targets = [], []
    for vid in scene.test_ids:
        emb_param = app.embeddings.weight
        opt_emb = torch.optim.Adam([emb_param], lr=0.03)
        for _ in range(adapt_steps):
            raw, depth = gs(scene.cameras[vid])
            pred, mats = app(raw, depth, scene.cameras[vid], vid, cell_size=8)
            loss = reconstruction_loss(left_half(pred), left_half(scene.images[vid])) + 0.05 * identity_regularizer(mats)
            opt_emb.zero_grad()
            loss.backward()
            emb_param.grad[: min(scene.test_ids)] = 0
            opt_emb.step()
        with torch.no_grad():
            raw, depth = gs(scene.cameras[vid])
            pred, _ = app(raw, depth, scene.cameras[vid], vid, cell_size=8)
            preds.append(right_half(pred))
            targets.append(right_half(scene.images[vid]))

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pred_right": torch.stack(preds).cpu(),
            "target_right": torch.stack(targets).cpu(),
            "coupled_appearance_at_splat_render": 0.0,
            "train_iterations": float(steps),
        },
        out_dir / "predictions.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--adapt-steps", type=int, default=80)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--only", choices=["all", "baseline", "davigs"], default="all")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    scene = build_scene(seed=args.seed)
    if args.only in ("all", "baseline"):
        train_baseline(scene, max(100, args.steps // 2), args.out / "vanilla_3dgs")
    if args.only in ("all", "davigs"):
        train_davigs(scene, args.steps, args.adapt_steps, args.out / "davigs")


if __name__ == "__main__":
    main()
