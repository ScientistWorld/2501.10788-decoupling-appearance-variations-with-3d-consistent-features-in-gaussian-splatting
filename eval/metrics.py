"""Evaluation metrics independent of method implementations."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2).item()
    return 99.0 if mse <= 1e-12 else -10.0 * math.log10(mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.clamp(0, 1).unsqueeze(0)
    target = target.clamp(0, 1).unsqueeze(0)
    c1, c2 = 0.01**2, 0.03**2
    kernel = torch.ones((3, 1, 7, 7), dtype=pred.dtype, device=pred.device) / 49.0
    mu_x = F.conv2d(pred, kernel, padding=3, groups=3)
    mu_y = F.conv2d(target, kernel, padding=3, groups=3)
    sig_x = F.conv2d(pred * pred, kernel, padding=3, groups=3) - mu_x**2
    sig_y = F.conv2d(target * target, kernel, padding=3, groups=3) - mu_y**2
    sig_xy = F.conv2d(pred * target, kernel, padding=3, groups=3) - mu_x * mu_y
    val = ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2))
    return float(val.mean().item())


def lpips_proxy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """A deterministic perceptual proxy used because LPIPS weights are external."""
    pred = pred.clamp(0, 1).unsqueeze(0)
    target = target.clamp(0, 1).unsqueeze(0)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device) / 8.0
    sobel_y = sobel_x.T
    kernels = torch.stack([sobel_x, sobel_y]).unsqueeze(1).repeat(3, 1, 1, 1)
    grad_p = F.conv2d(pred, kernels, padding=1, groups=3)
    grad_t = F.conv2d(target, kernels, padding=1, groups=3)
    return float((0.65 * (pred - target).abs().mean() + 0.35 * (grad_p - grad_t).abs().mean()).item())
