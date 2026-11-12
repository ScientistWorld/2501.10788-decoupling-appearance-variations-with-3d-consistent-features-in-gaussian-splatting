# Evaluation Targets

## Primary: glav_novel_view

Held-out novel-view quality under global and local appearance variation (paper Table 1 protocol: fit appearance on the left half of a held-out capture, score the right half).

- **Metrics:** `psnr` (primary) and `ssim` are higher-is-better; `lpips` (v0.1, AlexNet) is lower-is-better.
- **Paper direction reproduced:** the method beats vanilla 3DGS on all three metrics (see `scoring/scores_test.json`).
- **Run:** `bash scripts/reproduce.sh`, then `bash scripts/evaluate_train.sh` (development scenes) / `bash scripts/evaluate_test.sh` (hidden scenes).

## Constraint and secondary: glav_average_appearance

The scene rendered without per-capture appearance state, colour-corrected by the evaluator to the camera's average appearance (paper Table 4 protocol). It captures the paper's claim that the appearance module can be discarded after optimisation without floaters or colour distortions.

- **Metrics:** `psnr` (primary), `ssim` higher-is-better; `lpips` lower-is-better.
- **Constraint (gate):** a method more than 1.0 dB PSNR below the frozen vanilla 3DGS render here fails the run (worst values everywhere).
- **Ablations:** `davigs_uv_depth` (3D hash features → positional encoding of pixel coordinates and depth) and `davigs_no_id` (no identity regulariser) are frozen comparison points, following paper Table 4.

## Not scored

The paper's efficiency claims (training time, VRAM, FPS) depend on hardware and wall-clock time, and the PhotoTourism and plug-and-play experiments (Tables 2 and 3) need real captures and other renderers. None of these is scored. Their reported values are kept under `_paper_reported_unscored` in `scoring/reference.json`.
