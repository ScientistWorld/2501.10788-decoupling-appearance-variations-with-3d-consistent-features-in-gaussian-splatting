#!/usr/bin/env python3
"""Evaluate saved predictions and write scoring/scores.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from eval.metrics import lpips_proxy, psnr, ssim


CORE_METRICS = {
    "psnr",
    "ssim",
    "lpips_proxy",
    "worst_view_psnr",
    "mean_color_bias_mae",
    "edge_mae",
    "coupled_appearance_at_splat_render",
    "train_iterations",
}
ABLATION_METRICS = {
    "psnr",
    "ssim",
    "lpips_proxy",
    "worst_view_psnr",
    "mean_color_bias_mae",
    "edge_mae",
}
CELL_METRICS = {
    "psnr",
    "ssim",
    "lpips_proxy",
    "worst_view_psnr",
    "mean_color_bias_mae",
    "edge_mae",
    "cell_size",
    "cell_queries_per_view",
    "train_iterations",
}


def artifact_methods(outputs: Path) -> list[str]:
    return sorted(path.parent.name for path in outputs.glob("*/predictions.pt") if not path.parent.name.startswith("_"))


def score_plan(reference: Path) -> dict[str, list[str]]:
    if not reference.exists():
        return {}
    data = json.loads(reference.read_text())
    experiments = data.get("experiments", data)
    plan: dict[str, list[str]] = {}
    for exp_name, exp_data in experiments.items():
        if exp_name.startswith("_") or not isinstance(exp_data, dict):
            continue
        results = exp_data.get("results", {})
        if isinstance(results, dict):
            plan[exp_name] = sorted(name for name in results if not name.startswith("_"))
    return plan


def edge_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.clamp(0, 1).unsqueeze(0)
    target = target.clamp(0, 1).unsqueeze(0)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device) / 8.0
    sobel_y = sobel_x.T
    kernels = torch.stack([sobel_x, sobel_y]).unsqueeze(1).repeat(3, 1, 1, 1)
    grad_pred = F.conv2d(pred, kernels, padding=1, groups=3)
    grad_target = F.conv2d(target, kernels, padding=1, groups=3)
    return float((grad_pred - grad_target).abs().mean().item())


def summarize(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    vals = {"psnr": [], "ssim": [], "lpips_proxy": [], "edge_mae": []}
    for i in range(pred.shape[0]):
        vals["psnr"].append(psnr(pred[i], target[i]))
        vals["ssim"].append(ssim(pred[i], target[i]))
        vals["lpips_proxy"].append(lpips_proxy(pred[i], target[i]))
        vals["edge_mae"].append(edge_mae(pred[i], target[i]))
    metrics = {k: float(sum(v) / len(v)) for k, v in vals.items()}
    metrics["worst_view_psnr"] = float(min(vals["psnr"]))
    pred_bias = pred.clamp(0, 1).mean(dim=(-1, -2))
    target_bias = target.clamp(0, 1).mean(dim=(-1, -2))
    metrics["mean_color_bias_mae"] = float((pred_bias - target_bias).abs().mean().item())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores.json"))
    parser.add_argument("--reference", type=Path, default=Path("scoring/reference.json"))
    args = parser.parse_args()

    results = {}
    for name in artifact_methods(args.outputs):
        artifact = args.outputs / name / "predictions.pt"
        data = torch.load(artifact, map_location="cpu", weights_only=False)
        metrics = summarize(data["pred_right"], data["target_right"])
        metrics["coupled_appearance_at_splat_render"] = float(data.get("coupled_appearance_at_splat_render", 0.0))
        metrics["train_iterations"] = float(data.get("train_iterations", 0.0))
        if "cell_size" in data:
            metrics["cell_size"] = float(data["cell_size"])
        if "cell_queries_per_view" in data:
            metrics["cell_queries_per_view"] = float(data["cell_queries_per_view"])
        results[name] = metrics

    if not results:
        raise SystemExit(f"No predictions found under {args.outputs}")

    plan = score_plan(args.reference)
    ablation_ref = set(plan.get("synthetic_glav_ablation", []))
    core_ref = set(plan.get("synthetic_glav_core", []))
    cell_ref = set(plan.get("synthetic_glav_cell_size", []))
    ablation_only = ablation_ref - core_ref
    cell_only = cell_ref - core_ref - ablation_ref
    core_names = sorted(name for name in results if name not in ablation_only and name not in cell_only)
    ablation_names = sorted(name for name in ablation_ref if name in results)
    cell_names = sorted(name for name in cell_ref if name in results)

    core = {
        name: {metric: value for metric, value in results[name].items() if metric in CORE_METRICS}
        for name in core_names
    }
    ablation = {
        name: {metric: value for metric, value in results[name].items() if metric in ABLATION_METRICS}
        for name in ablation_names
    }
    cell_size = {
        name: {metric: value for metric, value in results[name].items() if metric in CELL_METRICS}
        for name in cell_names
    }
    experiments = {}
    if core:
        experiments["synthetic_glav_core"] = {"results": core}
    if len(ablation) > 1:
        experiments["synthetic_glav_ablation"] = {"results": ablation}
    if len(cell_size) > 1:
        experiments["synthetic_glav_cell_size"] = {"results": cell_size}
    payload = {"experiments": experiments}
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
