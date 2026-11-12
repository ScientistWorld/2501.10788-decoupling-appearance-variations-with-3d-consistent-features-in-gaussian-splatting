"""Per-scene optimisation for vanilla 3DGS and DAVIGS.

Protocol (paper, "Metrics"): the left half of every held-out capture is used
during training to optimise appearance-related parameters only (embedding,
hash grids, MLP); Gaussians never receive gradients from held-out captures.
The right half is predicted; the appearance-free render is also returned for
the average-appearance (colour-corrected) evaluation.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from method.appearance import AppearanceModule, identity_regularizer, lambda_id
from method.gaussians import Camera, GaussianModel

LAMBDA_DSSIM = 0.2


def _gauss_window(device) -> torch.Tensor:
    x = torch.arange(11, dtype=torch.float32, device=device) - 5
    g = torch.exp(-(x ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(3, 1, 11, 11).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    a, b = a[None], b[None]
    mu_a = F.conv2d(a, win, padding=5, groups=3)
    mu_b = F.conv2d(b, win, padding=5, groups=3)
    s_aa = F.conv2d(a * a, win, padding=5, groups=3) - mu_a ** 2
    s_bb = F.conv2d(b * b, win, padding=5, groups=3) - mu_b ** 2
    s_ab = F.conv2d(a * b, win, padding=5, groups=3) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return m.mean()


def photometric_loss(pred, gt, win):
    return (1 - LAMBDA_DSSIM) * (pred - gt).abs().mean() + LAMBDA_DSSIM * (1 - ssim(pred, gt, win))


def train_scene(scene: dict, cfg: dict, seed: int, device: str, log=print) -> dict:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    h, w = int(scene["height"]), int(scene["width"])
    half = w // 2
    train_imgs = scene["train_images"].float().div(255.0).to(device)
    held_left = scene["heldout_left"].float().div(255.0).to(device)
    n_tr, n_ho = train_imgs.shape[0], held_left.shape[0]
    cams_tr = [Camera(scene["train_R"][i], scene["train_t"][i], scene["train_K"][i], h, w, device) for i in range(n_tr)]
    cams_ho = [Camera(scene["heldout_R"][i], scene["heldout_t"][i], scene["heldout_K"][i], h, w, device) for i in range(n_ho)]
    centers = torch.stack([c.center for c in cams_tr])
    extent = float((centers - centers.mean(0)).norm(dim=1).max()) * 1.1

    steps = int(cfg["steps"])
    gm = GaussianModel(scene["points"], scene["colors"], extent, device, max_gaussians=cfg["max_gaussians"])
    davigs = cfg["kind"] == "davigs"
    app = app_opt = None
    if davigs:
        pts = scene["points"].float()
        lo, hi = torch.quantile(pts, 0.01, dim=0), torch.quantile(pts, 0.99, dim=0)
        pad = 0.1 * (hi - lo)
        app = AppearanceModule(
            n_tr,
            lo - pad,
            hi + pad,
            depth_scale=2.0 * extent,
            feature_mode=cfg["feature_mode"],
            emb_dim=cfg["emb_dim"],
            cell=cfg["cell_size"],
        ).to(device)
        app_opt = torch.optim.Adam(app.param_groups(cfg["lr_emb"], cfg["lr_mlp"], cfg["lr_grid"]), betas=(0.9, 0.99), eps=1e-15)
        base_lrs = [g["lr"] for g in app_opt.param_groups]
        # Held-out captures get their own embeddings, optimised on their left
        # halves only (each a separate parameter, so untouched rows never move).
        ho_embs = [torch.nn.Parameter(torch.randn(cfg["emb_dim"], device=device)) for _ in range(n_ho)]
        ho_opt = torch.optim.Adam(ho_embs, lr=cfg["lr_emb"], betas=(0.9, 0.99), eps=1e-15)
    use_id = davigs and cfg["use_id"]
    win = _gauss_window(device)
    bg = torch.zeros(3, device=device)

    # 3DGS schedule (30k iterations: densify from 500 to 15k every 100, opacity
    # reset every 3k), scaled proportionally to the step budget.
    densify_from = max(1, round(steps * 500 / 30000))
    densify_until = steps // 2
    densify_every = max(10, round(steps * 100 / 30000))
    reset_every = max(50, round(steps * 3000 / 30000))
    sh_every = max(1, steps // 30)
    heldout_every = max(1, round(n_tr / n_ho)) if (davigs and cfg["heldout_adapt"]) else 0

    order: list[int] = []
    ho_count = 0
    t0 = time.time()
    last_loss = float("nan")
    for step in range(steps):
        gm.set_position_lr(step / max(1, steps - 1))
        if davigs and cfg["module_cosine"]:
            factor = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / steps))
            for g, lr in zip(app_opt.param_groups, base_lrs):
                g["lr"] = lr * factor
            for g in ho_opt.param_groups:
                g["lr"] = cfg["lr_emb"] * factor
        sh_degree = min(3, step // sh_every)
        if not order:
            order = torch.randperm(n_tr, generator=gen).tolist()
        vid = order.pop()
        cam = cams_tr[vid]
        tracking = step < densify_until
        image, depth, _, info = gm.render(cam, sh_degree, bg, track=tracking)
        if davigs:
            pred, mats = app(image, depth, cam, vid)
            loss = photometric_loss(pred, train_imgs[vid], win)
            if use_id:
                loss = loss + lambda_id(step, steps) * identity_regularizer(mats)
        else:
            loss = photometric_loss(image, train_imgs[vid], win)
        loss.backward()
        last_loss = float(loss.detach())
        with torch.no_grad():
            if tracking:
                gm.add_densify_stats(info)
            gm.opt.step()
            gm.opt.zero_grad(set_to_none=True)
            if davigs:
                app_opt.step()
                app_opt.zero_grad(set_to_none=True)
            if densify_from <= step < densify_until and (step + 1) % densify_every == 0:
                gm.densify_and_prune(0.0002, 0.005, prune_big=step >= reset_every)
            if step < densify_until and (step + 1) % reset_every == 0:
                gm.reset_opacity()

        if heldout_every and (step + 1) % heldout_every == 0:
            k = ho_count % n_ho
            ho_count += 1
            with torch.no_grad():
                img_h, depth_h, _, _ = gm.render(cams_ho[k], sh_degree, bg)
            pred, mats = app(img_h, depth_h, cams_ho[k], 0, embedding=ho_embs[k])
            loss_h = photometric_loss(pred[..., :half], held_left[k], win)
            if use_id:
                loss_h = loss_h + lambda_id(step, steps) * identity_regularizer(mats)
            (ho_embs[k].grad,) = torch.autograd.grad(loss_h, [ho_embs[k]])
            ho_opt.step()
            ho_opt.zero_grad(set_to_none=True)

        if (step + 1) % max(1, steps // 6) == 0 or step + 1 == steps:
            el = time.time() - t0
            log(f"step {step + 1}/{steps} loss={last_loss:.4f} gaussians={gm.n} {1000 * el / (step + 1):.1f}ms/step")

    final_degree = min(3, (steps - 1) // sh_every)
    embeddings = {}
    if davigs:
        for k in range(n_ho):
            # Without held-out adaptation the capture gets the mean training embedding.
            embeddings[k] = ho_embs[k].detach() if cfg["heldout_adapt"] else app.emb.weight.detach().mean(0)
    if davigs and cfg["post_adapt_steps"] > 0:
        for p in app.parameters():
            p.requires_grad_(False)
        for k in range(n_ho):
            with torch.no_grad():
                img_h, depth_h, _, _ = gm.render(cams_ho[k], final_degree, bg)
            emb = embeddings[k].clone().requires_grad_(True)
            opt = torch.optim.Adam([emb], lr=cfg["lr_emb"])
            for _ in range(int(cfg["post_adapt_steps"])):
                pred, mats = app(img_h, depth_h, cams_ho[k], 0, embedding=emb)
                loss_h = photometric_loss(pred[..., :half], held_left[k], win)
                if use_id:
                    loss_h = loss_h + lambda_id(steps, steps) * identity_regularizer(mats)
                opt.zero_grad(set_to_none=True)
                loss_h.backward()
                opt.step()
            embeddings[k] = emb.detach()

    novel, canonical = [], []
    with torch.no_grad():
        for k in range(n_ho):
            img_h, depth_h, _, _ = gm.render(cams_ho[k], final_degree, bg)
            if davigs:
                out, _ = app(img_h, depth_h, cams_ho[k], 0, embedding=embeddings[k])
            else:
                out = img_h
            novel.append(out.clamp(0, 1)[..., half:].cpu())
            canonical.append(img_h.clamp(0, 1)[..., half:].cpu())
        train_psnr = []
        for i in range(0, n_tr, max(1, n_tr // 6)):
            img, depth, _, _ = gm.render(cams_tr[i], final_degree, bg)
            if davigs:
                img, _ = app(img, depth, cams_tr[i], i)
            mse = float(((img.clamp(0, 1) - train_imgs[i]) ** 2).mean())
            train_psnr.append(-10 * math.log10(max(mse, 1e-10)))
    return {
        "novel_view": torch.stack(novel).float(),
        "average_appearance": torch.stack(canonical).float(),
        "log": {
            "seconds": time.time() - t0,
            "gaussians": gm.n,
            "final_loss": last_loss,
            "train_view_psnr": sum(train_psnr) / len(train_psnr),
        },
    }
