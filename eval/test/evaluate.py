#!/usr/bin/env python3
"""Held-out test-slice scorer for saved GLAV prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from eval.metrics import lpips_proxy, psnr, ssim


SLICE = "test"

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


def score_plan_from_train(scores_train: Path) -> dict[str, list[str]]:
    if not scores_train.exists():
        return {}
    data = json.loads(scores_train.read_text())
    experiments = data.get("experiments", data)
    plan: dict[str, list[str]] = {}
    for exp_name, exp_data in experiments.items():
        if exp_name.startswith("_") or not isinstance(exp_data, dict):
            continue
        results = exp_data.get("results", exp_data)
        if isinstance(results, dict):
            plan[exp_name] = sorted(name for name in results if not name.startswith("_"))
    return plan


def test_indices(num_samples: int) -> list[int]:
    if num_samples < 2:
        raise ValueError("train/test split requires at least two held-out prediction samples")
    return list(range(1, num_samples, 2))


def edge_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.clamp(0, 1).unsqueeze(0)
    target = target.clamp(0, 1).unsqueeze(0)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device) / 8.0
    sobel_y = sobel_x.T
    kernels = torch.stack([sobel_x, sobel_y]).unsqueeze(1).repeat(3, 1, 1, 1)
    grad_pred = F.conv2d(pred, kernels, padding=1, groups=3)
    grad_target = F.conv2d(target, kernels, padding=1, groups=3)
    return float((grad_pred - grad_target).abs().mean().item())


def summarize(pred: torch.Tensor, target: torch.Tensor, indices: list[int]) -> dict[str, float]:
    vals = {"psnr": [], "ssim": [], "lpips_proxy": [], "edge_mae": []}
    for i in indices:
        vals["psnr"].append(psnr(pred[i], target[i]))
        vals["ssim"].append(ssim(pred[i], target[i]))
        vals["lpips_proxy"].append(lpips_proxy(pred[i], target[i]))
        vals["edge_mae"].append(edge_mae(pred[i], target[i]))
    metrics = {key: float(sum(values) / len(values)) for key, values in vals.items()}
    metrics["worst_view_psnr"] = float(min(vals["psnr"]))
    pred_bias = pred[indices].clamp(0, 1).mean(dim=(-1, -2))
    target_bias = target[indices].clamp(0, 1).mean(dim=(-1, -2))
    metrics["mean_color_bias_mae"] = float((pred_bias - target_bias).abs().mean().item())
    return metrics


def load_metrics(outputs: Path, method_name: str) -> dict[str, float]:
    artifact = outputs / method_name / "predictions.pt"
    if not artifact.exists():
        raise FileNotFoundError(f"Missing prediction artifact for {method_name}: {artifact}")

    data = torch.load(artifact, map_location="cpu", weights_only=False)
    pred = data["pred_right"]
    target = data["target_right"]
    metrics = summarize(pred, target, test_indices(pred.shape[0]))
    metrics["coupled_appearance_at_splat_render"] = float(data.get("coupled_appearance_at_splat_render", 0.0))
    metrics["train_iterations"] = float(data.get("train_iterations", 0.0))
    if "cell_size" in data:
        metrics["cell_size"] = float(data["cell_size"])
    if "cell_queries_per_view" in data:
        metrics["cell_queries_per_view"] = float(data["cell_queries_per_view"])
    return metrics


def build_payload(outputs: Path, scores_train: Path) -> dict:
    plan = score_plan_from_train(scores_train)
    if plan:
        method_names = sorted({name for names in plan.values() for name in names})
    else:
        method_names = artifact_methods(outputs)

    if not method_names:
        raise SystemExit(f"No predictions found under {outputs}")

    per_method = {name: load_metrics(outputs, name) for name in method_names}
    experiments = {}

    core_names = plan.get("synthetic_glav_core", method_names)
    core = {
        name: {metric: value for metric, value in per_method[name].items() if metric in CORE_METRICS}
        for name in core_names
    }
    if core:
        experiments["synthetic_glav_core"] = {"results": core}

    if "synthetic_glav_ablation" in plan:
        ablation_names = plan["synthetic_glav_ablation"]
    else:
        ablation_names = [name for name in per_method if name != "vanilla_3dgs"]
    ablation = {
        name: {metric: value for metric, value in per_method[name].items() if metric in ABLATION_METRICS}
        for name in ablation_names
    }
    if ablation:
        experiments["synthetic_glav_ablation"] = {"results": ablation}

    if "synthetic_glav_cell_size" in plan:
        cell_names = plan["synthetic_glav_cell_size"]
        cell_size = {
            name: {metric: value for metric, value in per_method[name].items() if metric in CELL_METRICS}
            for name in cell_names
            if name in per_method
        }
        if cell_size:
            experiments["synthetic_glav_cell_size"] = {"results": cell_size}

    return {"slice": SLICE, "experiments": experiments}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores_test.json"))
    parser.add_argument("--scores-train", type=Path, default=Path("scoring/scores_train.json"))
    args = parser.parse_args()

    payload = build_payload(args.outputs, args.scores_train)
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
