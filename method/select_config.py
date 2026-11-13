#!/usr/bin/env python3
"""Choose the appearance-module hyperparameters the paper does not specify.

The paper gives the MLP widths, the identity-regulariser schedule, lambda_1 and
the cell size, but not the hash grid's finest resolution or the module learning
rates. ``python -m method.run_experiment --plan select`` trains each candidate
in ``run_experiment.CANDIDATES`` on the three development scenes; this script
scores them with the public development-scene scorer (``eval/train``, whose
targets are visible) and writes the best one to ``method/selected_config.json``.

Selection criterion (development scenes only): the gym's own weighting,
sum over experiments and metrics of weight * |coefficient| * (value in the
better direction) / scale, with scales PSNR 1 dB, SSIM 0.02, LPIPS 0.02.

The winner's development-scene caches are copied into ``outputs/davigs`` so
the reference run does not retrain them (identical configuration and seed).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from eval.train.evaluate import LPIPSAlex, load_prediction_file, score
from method.run_experiment import CANDIDATES, SELECTED, cfg_hash, resolve

ROOT = Path(__file__).resolve().parents[1]
SCALES = {"psnr": 1.0, "ssim": 0.02, "lpips": 0.02}


def main() -> None:
    torch.set_num_threads(2)
    ref = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
    answers = torch.load(ROOT / "eval" / "train" / "answers.pt", map_location="cpu", weights_only=True)
    lpips = LPIPSAlex(ROOT / "data" / "downloads" / "lpips")
    table = {}
    for cand in CANDIDATES:
        path = ROOT / "outputs" / "_select" / cand / "davigs" / "predictions.pt"
        pred = load_prediction_file(path, answers)
        summary, _ = score(pred, answers, lpips)
        total = 0.0
        for exp, spec in ref.items():
            for m, mdef in spec["metrics"].items():
                sign = 1.0 if mdef["higher_is_better"] else -1.0
                total += spec["weight"] * abs(mdef["coefficient"]) * sign * summary[exp][m] / SCALES[m]
        table[cand] = {"criterion": total, **{f"{exp}.{m}": v for exp, s in summary.items() for m, v in s.items()}}
        print(f"[select] {cand}: criterion={total:.3f} " + " ".join(f"{k}={v:.4f}" for k, v in table[cand].items() if k != "criterion"), flush=True)
    best = max(table, key=lambda c: table[c]["criterion"])
    steps = resolve("davigs", f"cand_{best}", None)["steps"]
    out = {**CANDIDATES[best], "chosen": best, "candidates_dev_scores": table,
           "note": "chosen on the development scenes by method/select_config.py; see its docstring"}
    SELECTED.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[select] chosen {best}: {CANDIDATES[best]} -> {SELECTED}", flush=True)

    # Reuse the winner's development-scene runs for the reference (same config, seed and code).
    src_dir = ROOT / "outputs" / "_select" / best / "davigs" / "scenes"
    dst_dir = ROOT / "outputs" / "davigs" / "scenes"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in src_dir.glob("*.pt"):
        cached = torch.load(f, map_location="cpu", weights_only=True)
        h = cfg_hash(cached["config"], int(cached["seed"]))
        if f.name.endswith(f"-{h}.pt") and cached["config"]["steps"] == steps:
            shutil.copy2(f, dst_dir / f.name)
            print(f"[select] reuse {f.name}", flush=True)


if __name__ == "__main__":
    main()
