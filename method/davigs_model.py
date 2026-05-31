"""Reduced-scale DAVIGS implementation with differentiable Gaussian splats."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from data.synthetic_glav import Camera, backproject_grid, splat_render


class GaussianSplatScene(nn.Module):
    def __init__(self, init_points: torch.Tensor, init_colors: torch.Tensor):
        super().__init__()
        self.points = nn.Parameter(init_points.clone())
        self.color_logits = nn.Parameter(torch.logit(init_colors.clamp(1e-4, 1 - 1e-4)))
        self.opacity_logits = nn.Parameter(torch.full((init_points.shape[0],), 0.15))
        self.log_scale = nn.Parameter(torch.full((init_points.shape[0],), math.log(0.03)))

    def forward(self, camera: Camera) -> tuple[torch.Tensor, torch.Tensor]:
        colors = torch.sigmoid(self.color_logits)
        opacity = torch.sigmoid(self.opacity_logits)
        scale = self.log_scale.exp().clamp(0.012, 0.08)
        image, depth, _ = splat_render(self.points, colors, opacity, scale, camera)
        return image, depth


class MultiResHashGrid(nn.Module):
    def __init__(self, levels: int = 4, features: int = 4, table_size: int = 1024):
        super().__init__()
        self.levels = levels
        self.features = features
        self.table_size = table_size
        self.resolutions = [8, 13, 21, 34]
        self.tables = nn.Parameter(torch.empty(levels, table_size, features).uniform_(-1e-4, 1e-4))
        self.register_buffer("bbox_min", torch.tensor([-1.25, -0.9, -1.0]))
        self.register_buffer("bbox_max", torch.tensor([1.25, 0.9, 0.9]))

    def hash_index(self, corner: torch.Tensor) -> torch.Tensor:
        primes = torch.tensor([1, 2654435761, 805459861], dtype=torch.long, device=corner.device)
        return ((corner.long() * primes).sum(dim=-1) % self.table_size).long()

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        orig_shape = xyz.shape[:-1]
        x = ((xyz.reshape(-1, 3) - self.bbox_min) / (self.bbox_max - self.bbox_min)).clamp(0, 1)
        feats = []
        offsets = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
            dtype=torch.long,
            device=xyz.device,
        )
        for li, res in enumerate(self.resolutions[: self.levels]):
            pos = x * (res - 1)
            base = torch.floor(pos).long()
            frac = pos - base.float()
            corner = base[:, None, :] + offsets[None]
            idx = self.hash_index(corner)
            val = self.tables[li, idx]
            w = torch.where(offsets[None].bool(), frac[:, None], 1.0 - frac[:, None]).prod(dim=-1)
            feats.append((val * w[..., None]).sum(dim=1))
        return torch.cat(feats, dim=-1).reshape(*orig_shape, -1)


class AppearanceModule(nn.Module):
    def __init__(self, num_views: int, appearance_dim: int = 8, levels: int = 4, features: int = 4):
        super().__init__()
        self.embeddings = nn.Embedding(num_views, appearance_dim)
        self.grid = MultiResHashGrid(levels=levels, features=features)
        in_dim = levels * features + appearance_dim
        self.mlp = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 12))
        nn.init.zeros_(self.mlp[-1].weight)
        ident = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float32)
        self.mlp[-1].bias.data.copy_(ident)

    def query_matrices(self, xyz: torch.Tensor, view_id: int) -> torch.Tensor:
        feat = self.grid(xyz)
        emb = self.embeddings(torch.tensor(view_id, device=xyz.device)).expand(*feat.shape[:-1], -1)
        return self.mlp(torch.cat([feat, emb], dim=-1)).reshape(*feat.shape[:-1], 3, 4)

    def forward(self, image: torch.Tensor, depth: torch.Tensor, camera: Camera, view_id: int, cell_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = image.shape[-2:]
        pooled_depth = F.avg_pool2d(depth[None, None], kernel_size=cell_size, stride=cell_size, ceil_mode=True)[0, 0]
        ch, cw = pooled_depth.shape
        ys = (torch.arange(ch, dtype=depth.dtype, device=depth.device) + 0.5) * cell_size - 0.5
        xs = (torch.arange(cw, dtype=depth.dtype, device=depth.device) + 0.5) * cell_size - 0.5
        yy, xx = torch.meshgrid(ys.clamp(0, h - 1), xs.clamp(0, w - 1), indexing="ij")
        x = (xx - (w - 1) / 2.0) * pooled_depth / camera.focal
        y = -(yy - (h - 1) / 2.0) * pooled_depth / camera.focal
        cam = torch.stack([x, y, pooled_depth], dim=-1)
        xyz = (cam - camera.translation.to(depth.device)) @ camera.rotation.to(depth.device)
        coarse = self.query_matrices(xyz, view_id).permute(2, 3, 0, 1).reshape(1, 12, ch, cw)
        mats = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).reshape(h, w, 3, 4)
        rgb1 = torch.cat([image.permute(1, 2, 0), torch.ones(h, w, 1, dtype=image.dtype, device=image.device)], dim=-1)
        transformed = torch.einsum("hwk,hwck->hwc", rgb1, mats).permute(2, 0, 1).clamp(0, 1)
        return transformed, mats


def dssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu_x = pred.mean(dim=(-1, -2), keepdim=True)
    mu_y = target.mean(dim=(-1, -2), keepdim=True)
    vx = ((pred - mu_x) ** 2).mean(dim=(-1, -2), keepdim=True)
    vy = ((target - mu_y) ** 2).mean(dim=(-1, -2), keepdim=True)
    vxy = ((pred - mu_x) * (target - mu_y)).mean(dim=(-1, -2), keepdim=True)
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * vxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (vx + vy + c2))
    return ((1.0 - ssim) / 2.0).mean()


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, lam: float = 0.2) -> torch.Tensor:
    return (1 - lam) * (pred - target).abs().mean() + lam * dssim_loss(pred, target)


def identity_regularizer(mats: torch.Tensor) -> torch.Tensor:
    ident = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=mats.dtype, device=mats.device)
    return (mats - ident).abs().mean()
