#!/usr/bin/env python3
"""Held-out test-slice scorer for saved GLAV prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eval.metrics import lpips_proxy, psnr, ssim


SLICE = "test"

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
    start = max(1, num_samples // 2)
    return list(range(start, num_samples))


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
    metrics = summarize(pred, target, test_indices(pred.shape[0]))
    metrics["coupled_appearance_at_splat_render"] = float(data.get("coupled_appearance_at_splat_render", 0.0))
    metrics["train_iterations"] = float(data.get("train_iterations", 0.0))
    if "uses_3d_hash_features" in data:
        metrics["uses_3d_hash_features"] = float(data["uses_3d_hash_features"])
    if "uses_identity_regularizer" in data:
        metrics["uses_identity_regularizer"] = float(data["uses_identity_regularizer"])
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
        ablation_names = [
            name
            for name, metrics in per_method.items()
            if {"uses_3d_hash_features", "uses_identity_regularizer"} & set(metrics)
        ]
    ablation = {
        name: {metric: value for metric, value in per_method[name].items() if metric in ABLATION_METRICS}
        for name in ablation_names
    }
    if ablation:
        experiments["synthetic_glav_ablation"] = {"results": ablation}

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
