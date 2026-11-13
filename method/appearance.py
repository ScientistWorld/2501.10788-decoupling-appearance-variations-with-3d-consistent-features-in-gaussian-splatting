"""DAVIGS decoupled appearance module (Lin et al., AAAI 2025).

For a rendered image I^r and its rendered mean-depth map D, the image is divided
into non-overlapping cells of ``cell`` x ``cell`` pixels. Each cell's mean depth
back-projects the cell centre to a 3D point x, which queries multi-resolution
hash grids (Mueller et al. 2022) for 3D-consistent features {f_i}. These are
concatenated with the view's appearance embedding l and decoded by an MLP
(hidden widths 128 and 64) into a 3x4 affine colour transform per cell. Pixel
transforms are bilinearly interpolated from neighbouring cells and applied to
[r, g, b, 1]. The module is used only during optimisation; final rendering
discards it.

``feature_mode="uv_depth"`` is the paper's "xyz -> uv + depth" ablation: the
3D hash features are replaced by positional encodings of the pixel
coordinates and depth.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

IDENTITY_3x4 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])


class HashGrid(nn.Module):
    def __init__(self, bbox_min, bbox_max, levels=16, features=2, log2_table=15, base_res=16, max_res=512):
        super().__init__()
        growth = math.exp((math.log(max_res) - math.log(base_res)) / (levels - 1))
        self.res = [int(math.floor(base_res * growth ** level)) for level in range(levels)]
        self.table_size = 2 ** log2_table
        self.tables = nn.Parameter(torch.empty(levels, self.table_size, features).uniform_(-1e-4, 1e-4))
        self.register_buffer("bmin", bbox_min.float().clone())
        self.register_buffer("bmax", bbox_max.float().clone())
        corners = [[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)]
        self.register_buffer("offsets", torch.tensor(corners, dtype=torch.long))
        self.out_dim = levels * features

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        shape = xyz.shape[:-1]
        x = ((xyz.reshape(-1, 3) - self.bmin) / (self.bmax - self.bmin)).clamp(0.0, 1.0)
        mask = self.offsets.bool()[None]
        feats = []
        for level, res in enumerate(self.res):
            pos = (x * res).clamp(0.0, res - 1e-4)
            base = torch.floor(pos)
            frac = pos - base
            corners = base.long()[:, None, :] + self.offsets[None]
            if (res + 1) ** 3 <= self.table_size:
                idx = corners[..., 0] + corners[..., 1] * (res + 1) + corners[..., 2] * (res + 1) ** 2
            else:
                idx = (corners[..., 0] ^ (corners[..., 1] * 2654435761) ^ (corners[..., 2] * 805459861)) % self.table_size
            weights = torch.where(mask, frac[:, None, :], 1.0 - frac[:, None, :]).prod(dim=-1)
            feats.append((self.tables[level][idx] * weights[..., None]).sum(dim=1))
        return torch.cat(feats, dim=-1).reshape(*shape, -1)


class AppearanceModule(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        depth_scale: float,
        feature_mode: str = "xyz",
        emb_dim: int = 32,
        cell: int = 1,
        hash_max_res: int = 512,
    ):
        super().__init__()
        if feature_mode not in ("xyz", "uv_depth"):
            raise ValueError(feature_mode)
        self.feature_mode = feature_mode
        self.cell = int(cell)
        self.depth_scale = float(depth_scale)
        self.emb = nn.Embedding(num_embeddings, emb_dim)
        if feature_mode == "xyz":
            self.grid = HashGrid(bbox_min, bbox_max, max_res=int(hash_max_res))
            feat_dim = self.grid.out_dim
        else:
            self.grid = None
            feat_dim = 3 + 3 * 2 * 5
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + emb_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 12)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        with torch.no_grad():
            self.mlp[-1].bias.copy_(IDENTITY_3x4.reshape(-1))

    def param_groups(self, lr_emb: float, lr_mlp: float, lr_grid: float) -> list[dict]:
        groups = [
            {"params": list(self.emb.parameters()), "lr": lr_emb, "name": "emb"},
            {"params": list(self.mlp.parameters()), "lr": lr_mlp, "name": "mlp"},
        ]
        if self.grid is not None:
            groups.append({"params": [self.grid.tables], "lr": lr_grid, "name": "grid"})
        return groups

    def forward(self, image, depth, cam, view_index: int, embedding: torch.Tensor | None = None):
        h, w = image.shape[-2:]
        s = self.cell
        d = F.avg_pool2d(depth[None, None], s, s)[0, 0] if s > 1 else depth
        hc, wc = d.shape
        ys = (torch.arange(hc, device=depth.device, dtype=depth.dtype) + 0.5) * s - 0.5
        xs = (torch.arange(wc, device=depth.device, dtype=depth.dtype) + 0.5) * s - 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        if self.feature_mode == "xyz":
            pc = torch.stack([(xx - cam.cx) / cam.fx * d, (yy - cam.cy) / cam.fy * d, d], dim=-1)
            xyz = (pc - cam.t) @ cam.R
            feat = self.grid(xyz)
        else:
            uvd = torch.stack([xx / max(1, w - 1), yy / max(1, h - 1), d / self.depth_scale], dim=-1)
            freqs = (2.0 ** torch.arange(5, device=depth.device, dtype=depth.dtype)) * math.pi
            ang = (uvd[..., None] * freqs).flatten(-2)
            feat = torch.cat([uvd, torch.sin(ang), torch.cos(ang)], dim=-1)
        emb = self.emb.weight[view_index] if embedding is None else embedding
        out = self.mlp(torch.cat([feat, emb.expand(hc, wc, -1)], dim=-1))
        if s > 1:
            out = F.interpolate(out.permute(2, 0, 1)[None], size=(h, w), mode="bilinear", align_corners=False)[0]
            out = out.permute(1, 2, 0)
        mats = out.reshape(h, w, 3, 4)
        rgb1 = torch.cat([image.permute(1, 2, 0), torch.ones(h, w, 1, device=image.device, dtype=image.dtype)], dim=-1)
        transformed = torch.einsum("hwck,hwk->hwc", mats, rgb1).permute(2, 0, 1)
        return transformed, mats


def identity_regularizer(mats: torch.Tensor) -> torch.Tensor:
    return (mats - IDENTITY_3x4.to(mats.device, mats.dtype)).abs().mean()


def lambda_id(step: int, total: int, peak: float = 0.3, final: float = 0.2) -> float:
    """Linear warm-up 0 -> 0.3 over the first sixth of training (5k of 30k in the
    paper), then cosine annealing to 0.2 at the end."""
    warm = max(1, total // 6)
    if step < warm:
        return peak * step / warm
    t = (step - warm) / max(1, total - warm)
    return final + (peak - final) * 0.5 * (1.0 + math.cos(math.pi * t))
