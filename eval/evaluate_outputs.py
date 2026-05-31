#!/usr/bin/env python3
"""Evaluate saved predictions and write scoring/scores.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eval.metrics import lpips_proxy, psnr, ssim


def summarize(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    vals = {"psnr": [], "ssim": [], "lpips_proxy": []}
    for i in range(pred.shape[0]):
        vals["psnr"].append(psnr(pred[i], target[i]))
        vals["ssim"].append(ssim(pred[i], target[i]))
        vals["lpips_proxy"].append(lpips_proxy(pred[i], target[i]))
    return {k: float(sum(v) / len(v)) for k, v in vals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()

    results = {}
    for name in ["vanilla_3dgs", "davigs"]:
        artifact = args.outputs / name / "predictions.pt"
        if not artifact.exists():
            continue
        data = torch.load(artifact, map_location="cpu", weights_only=False)
        metrics = summarize(data["pred_right"], data["target_right"])
        metrics["coupled_appearance_at_splat_render"] = float(data.get("coupled_appearance_at_splat_render", 0.0))
        metrics["train_iterations"] = float(data.get("train_iterations", 0.0))
        results[name] = metrics

    if not results:
        raise SystemExit(f"No predictions found under {args.outputs}")

    payload = {"experiments": {"synthetic_glav_core": {"results": results}}}
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
