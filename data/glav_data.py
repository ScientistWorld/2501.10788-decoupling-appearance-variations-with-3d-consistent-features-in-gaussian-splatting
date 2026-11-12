"""Loader for the GLAV-style appearance-variation benchmark inputs.

Each scene file in ``data/glav_syn/`` holds only what a method may use:
  * ``train_images``  uint8 [V_train, 3, H, W] captures with per-image appearance,
  * ``train_R/t/K``   world-to-camera rotation, translation and intrinsics
                      (OpenCV convention: x right, y down, z forward),
  * ``heldout_left``  uint8 [V_heldout, 3, H, W // 2] left halves of the held-out
                      captures (for test-time appearance fitting only),
  * ``heldout_R/t/K`` cameras of the held-out captures,
  * ``points``/``colors`` a sparse, noisy SfM-like point cloud.

The right halves of held-out captures and the average-appearance targets are
owned by the evaluator and are not stored here.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parent / "glav_syn"


def load_manifest() -> dict:
    return json.loads((DATA_DIR / "manifest.json").read_text())


def scene_ids(split: str | None = None) -> list[str]:
    """Scene ids, optionally filtered by split ("dev" or "test")."""
    scenes = load_manifest()["scenes"]
    return [s["id"] for s in scenes if split in (None, "all", s["split"])]


def load_scene(scene_id: str) -> dict:
    return torch.load(DATA_DIR / f"{scene_id}.pt", map_location="cpu", weights_only=True)
