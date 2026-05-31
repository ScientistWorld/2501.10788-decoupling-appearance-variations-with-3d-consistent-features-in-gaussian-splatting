"""Procedural data for a small GLAV-like appearance-variation benchmark.

The paper's GLAV captures are not included with the workspace. This module
creates a compact scene with 3D surfaces, multi-view cameras, and view-specific
global plus local appearance changes. It is deliberately method-agnostic:
methods receive images, cameras, and a sparse point cloud, but no DAVIGS code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Camera:
    rotation: torch.Tensor
    translation: torch.Tensor
    focal: float
    height: int
    width: int


@dataclass
class SyntheticScene:
    cameras: list[Camera]
    images: torch.Tensor
    canonical_images: torch.Tensor
    depths: torch.Tensor
    train_ids: list[int]
    test_ids: list[int]
    init_points: torch.Tensor
    init_colors: torch.Tensor


def look_at(eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    forward = target - eye
    forward = forward / forward.norm()
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    right = torch.cross(forward, up, dim=0)
    right = right / right.norm()
    true_up = torch.cross(right, forward, dim=0)
    return torch.stack([right, true_up, forward], dim=0)


def make_cameras(num_views: int, height: int, width: int) -> list[Camera]:
    cameras = []
    for i in range(num_views):
        theta = 2.0 * math.pi * i / num_views
        eye = torch.tensor([2.6 * math.sin(theta), 0.25 * math.sin(2 * theta), -2.6 * math.cos(theta)])
        rotation = look_at(eye, torch.zeros(3))
        translation = -(rotation @ eye)
        cameras.append(Camera(rotation, translation, focal=52.0, height=height, width=width))
    return cameras


def make_true_points(seed: int = 7) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    n_wall, n_plane, n_blob = 70, 70, 40
    wall_xy = torch.rand((n_wall, 2), generator=gen) * 2.0 - 1.0
    wall = torch.stack([wall_xy[:, 0], wall_xy[:, 1] * 0.75, torch.full((n_wall,), 0.55)], dim=-1)
    plane_xz = torch.rand((n_plane, 2), generator=gen) * 2.0 - 1.0
    plane = torch.stack([plane_xz[:, 0], torch.full((n_plane,), -0.72), plane_xz[:, 1] * 0.75], dim=-1)
    blob = torch.randn((n_blob, 3), generator=gen) * 0.23 + torch.tensor([0.25, -0.1, -0.05])
    points = torch.cat([wall, plane, blob], dim=0)

    colors = torch.empty_like(points)
    colors[:, 0] = 0.45 + 0.35 * torch.sin(2.7 * points[:, 0] + 1.2)
    colors[:, 1] = 0.45 + 0.35 * torch.sin(2.2 * points[:, 1] - 0.4)
    colors[:, 2] = 0.45 + 0.35 * torch.cos(2.4 * points[:, 2] + 0.7)
    colors = colors.clamp(0.05, 0.95)
    opacity = torch.full((points.shape[0],), 0.55)
    scale = torch.full((points.shape[0],), 0.028)
    return points, colors, opacity, scale


def project(points: torch.Tensor, camera: Camera) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cam = points @ camera.rotation.T + camera.translation
    z = cam[:, 2].clamp_min(0.05)
    u = camera.focal * cam[:, 0] / z + (camera.width - 1) / 2.0
    v = -camera.focal * cam[:, 1] / z + (camera.height - 1) / 2.0
    valid = (cam[:, 2] > 0.05) & (u > -6) & (u < camera.width + 6) & (v > -6) & (v < camera.height + 6)
    return torch.stack([u, v], dim=-1), z, valid


def splat_render(
    points: torch.Tensor,
    colors: torch.Tensor,
    opacity: torch.Tensor,
    scale: torch.Tensor,
    camera: Camera,
    background: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.arange(camera.height, dtype=points.dtype, device=points.device),
        torch.arange(camera.width, dtype=points.dtype, device=points.device),
        indexing="ij",
    )
    xy, z, valid = project(points, camera)
    sigma = camera.focal * scale / z
    sigma = sigma.clamp(0.7, 5.0)
    dist2 = (xx[None] - xy[:, 0, None, None]) ** 2 + (yy[None] - xy[:, 1, None, None]) ** 2
    weights = opacity[:, None, None] * torch.exp(-0.5 * dist2 / (sigma[:, None, None] ** 2)) * valid[:, None, None]
    depth_weights = weights / z[:, None, None].clamp_min(0.05)
    acc = weights.sum(dim=0).clamp_min(1e-6)
    rgb = (weights[:, None] * colors[:, :, None, None]).sum(dim=0) / acc[None]
    alpha = (1.0 - torch.exp(-weights.sum(dim=0))).clamp(0.0, 1.0)
    image = rgb * alpha[None] + background * (1.0 - alpha[None])
    depth = (depth_weights * z[:, None, None]).sum(dim=0) / depth_weights.sum(dim=0).clamp_min(1e-6)
    return image.clamp(0.0, 1.0), depth, alpha


def backproject_grid(depth: torch.Tensor, camera: Camera) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(camera.height, dtype=depth.dtype, device=depth.device),
        torch.arange(camera.width, dtype=depth.dtype, device=depth.device),
        indexing="ij",
    )
    x = (xx - (camera.width - 1) / 2.0) * depth / camera.focal
    y = -(yy - (camera.height - 1) / 2.0) * depth / camera.focal
    cam = torch.stack([x, y, depth], dim=-1)
    return (cam - camera.translation.to(depth.device)) @ camera.rotation.to(depth.device)


def apply_view_appearance(canonical: torch.Tensor, xyz: torch.Tensor, view_id: int, num_views: int) -> torch.Tensor:
    phase = 2.0 * math.pi * view_id / num_views
    gains = torch.tensor(
        [1.0 + 0.48 * math.sin(phase), 1.0 + 0.42 * math.cos(phase + 0.4), 1.0 + 0.46 * math.sin(phase + 1.1)],
        dtype=canonical.dtype,
        device=canonical.device,
    )
    bias = torch.tensor(
        [0.13 * math.cos(phase), -0.11 * math.sin(phase + 0.3), 0.12 * math.cos(phase + 1.2)],
        dtype=canonical.dtype,
        device=canonical.device,
    )
    spot_center = torch.tensor([0.28 * math.sin(phase), -0.18, 0.18 * math.cos(phase)], dtype=canonical.dtype, device=canonical.device)
    dist = ((xyz - spot_center) ** 2).sum(dim=-1)
    local = torch.exp(-dist / 0.16)[None]
    tint = torch.tensor([0.68, 0.42, -0.32], dtype=canonical.dtype, device=canonical.device)[:, None, None]
    shadow = torch.exp(-((xyz[..., 0] + 0.35) ** 2 + (xyz[..., 2] - 0.2) ** 2) / 0.12)[None]
    image = canonical * gains[:, None, None] + bias[:, None, None] + local * tint - 0.36 * shadow
    return image.clamp(0.0, 1.0)


def build_scene(num_views: int = 16, height: int = 64, width: int = 64, seed: int = 7) -> SyntheticScene:
    cameras = make_cameras(num_views, height, width)
    points, colors, opacity, scale = make_true_points(seed)
    canonical, varied, depths = [], [], []
    for vid, cam in enumerate(cameras):
        img, depth, _ = splat_render(points, colors, opacity, scale, cam)
        xyz = backproject_grid(depth, cam)
        canonical.append(img)
        depths.append(depth)
        varied.append(apply_view_appearance(img, xyz, vid, num_views))
    gen = torch.Generator().manual_seed(seed + 1)
    init_points = points + 0.035 * torch.randn(points.shape, generator=gen)
    init_colors = (colors + 0.12 * torch.randn(colors.shape, generator=gen)).clamp(0.05, 0.95)
    test_ids = [3, 7, 11, 15]
    train_ids = [i for i in range(num_views) if i not in test_ids]
    return SyntheticScene(
        cameras=cameras,
        images=torch.stack(varied),
        canonical_images=torch.stack(canonical),
        depths=torch.stack(depths),
        train_ids=train_ids,
        test_ids=test_ids,
        init_points=init_points,
        init_colors=init_colors,
    )
