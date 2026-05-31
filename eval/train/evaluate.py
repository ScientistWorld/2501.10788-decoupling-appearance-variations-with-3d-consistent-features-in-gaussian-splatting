#!/usr/bin/env python3
"""Visible train-slice scorer for saved GLAV prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eval.metrics import lpips_proxy, psnr, ssim


SLICE = "train"

CORE_METRICS = {
    "psnr",
    "ssim",
    "lpips_proxy",
    "coupled_appearance_at_splat_render",
    "train_iterations",
}
ABLATION_METRICS = {
    "psnr",
    "ssim",
    "lpips_proxy",
    "uses_3d_hash_features",
    "uses_identity_regularizer",
}


def artifact_methods(outputs: Path) -> list[str]:
    return sorted(path.parent.name for path in outputs.glob("*/predictions.pt") if not path.parent.name.startswith("_"))


def train_indices(num_samples: int) -> list[int]:
    if num_samples < 2:
        raise ValueError("train/test split requires at least two held-out prediction samples")
    return list(range(max(1, num_samples // 2)))


def summarize(pred: torch.Tensor, target: torch.Tensor, indices: list[int]) -> dict[str, float]:
    vals = {"psnr": [], "ssim": [], "lpips_proxy": []}
    for i in indices:
        vals["psnr"].append(psnr(pred[i], target[i]))
        vals["ssim"].append(ssim(pred[i], target[i]))
        vals["lpips_proxy"].append(lpips_proxy(pred[i], target[i]))
    return {key: float(sum(values) / len(values)) for key, values in vals.items()}


def load_metrics(outputs: Path, method_name: str) -> dict[str, float]:
    artifact = outputs / method_name / "predictions.pt"
    if not artifact.exists():
        raise FileNotFoundError(f"Missing prediction artifact for {method_name}: {artifact}")

    data = torch.load(artifact, map_location="cpu", weights_only=False)
    pred = data["pred_right"]
    target = data["target_right"]
    metrics = summarize(pred, target, train_indices(pred.shape[0]))
    metrics["coupled_appearance_at_splat_render"] = float(data.get("coupled_appearance_at_splat_render", 0.0))
    metrics["train_iterations"] = float(data.get("train_iterations", 0.0))
    if "uses_3d_hash_features" in data:
        metrics["uses_3d_hash_features"] = float(data["uses_3d_hash_features"])
    if "uses_identity_regularizer" in data:
        metrics["uses_identity_regularizer"] = float(data["uses_identity_regularizer"])
    return metrics


def build_payload(outputs: Path) -> dict:
    methods = artifact_methods(outputs)
    if not methods:
        raise SystemExit(f"No predictions found under {outputs}")

    per_method = {name: load_metrics(outputs, name) for name in methods}
    core = {
        name: {metric: value for metric, value in metrics.items() if metric in CORE_METRICS}
        for name, metrics in per_method.items()
    }
    ablation = {
        name: {metric: value for metric, value in metrics.items() if metric in ABLATION_METRICS}
        for name, metrics in per_method.items()
        if {"uses_3d_hash_features", "uses_identity_regularizer"} & set(metrics)
    }

    experiments = {}
    if core:
        experiments["synthetic_glav_core"] = {"results": core}
    if ablation:
        experiments["synthetic_glav_ablation"] = {"results": ablation}
    return {"slice": SLICE, "experiments": experiments}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores_train.json"))
    args = parser.parse_args()

    payload = build_payload(args.outputs)
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
