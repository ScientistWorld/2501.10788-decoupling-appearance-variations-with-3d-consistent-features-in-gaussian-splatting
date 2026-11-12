#!/usr/bin/env python3
"""Maintainer tool: freeze the reference comparison predictions for this slice.

Copies ``outputs/<name>/predictions.pt`` for every comparison method in
``roster.json`` (vanilla 3DGS and the paper's ablations, produced by the
reference pipeline) into evaluator-owned ``frozen/<name>.pt``, keeping only this
slice's scenes. The evaluator scores comparisons only from these files.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    roster = json.loads((HERE / "roster.json").read_text())
    answers = torch.load(HERE / "answers.pt", map_location="cpu", weights_only=True)
    out_dir = HERE / "frozen"
    out_dir.mkdir(exist_ok=True)
    for name in roster["comparisons"]:
        src = ROOT / "outputs" / name / "predictions.pt"
        obj = torch.load(src, map_location="cpu", weights_only=True)
        frozen = {
            section: {sid: obj[section][sid].float().clamp(0, 1).to(torch.float16) for sid in answers["scenes"]}
            for section in ("novel_view", "average_appearance")
        }
        torch.save(frozen, out_dir / f"{name}.pt")
        print(f"[freeze:{HERE.name}] {src} -> {out_dir / (name + '.pt')}")


if __name__ == "__main__":
    main()
