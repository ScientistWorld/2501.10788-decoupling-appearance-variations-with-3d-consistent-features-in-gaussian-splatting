#!/usr/bin/env python3
"""Maintainer tool: freeze the reference-run predictions this slice compares against.

Copies ``outputs/<name>/predictions.pt`` for every comparison method in
``roster.json`` (vanilla 3DGS and the paper's ablations) and the gate's
reference reconstruction (the paper's method, ``outputs/davigs`` from the
reference run, stored as ``frozen/<gate.reference_file>.pt``) into
evaluator-owned ``frozen/``, keeping only this slice's scenes. The evaluator
scores comparisons and the gate only from these files, never from outputs/.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GATE_SOURCE = "davigs"


def main() -> None:
    roster = json.loads((HERE / "roster.json").read_text())
    answers = torch.load(HERE / "answers.pt", map_location="cpu", weights_only=True)
    out_dir = HERE / "frozen"
    out_dir.mkdir(exist_ok=True)
    jobs = {name: name for name in roster["comparisons"]}
    jobs[roster["gate"]["reference_file"]] = GATE_SOURCE
    for dest, src_name in jobs.items():
        src = ROOT / "outputs" / src_name / "predictions.pt"
        obj = torch.load(src, map_location="cpu", weights_only=True)
        frozen = {
            section: {sid: obj[section][sid].float().clamp(0, 1).to(torch.float16) for sid in answers["scenes"]}
            for section in ("novel_view", "average_appearance")
        }
        torch.save(frozen, out_dir / f"{dest}.pt")
        print(f"[freeze:{HERE.name}] {src} -> {out_dir / (dest + '.pt')}")


if __name__ == "__main__":
    main()
