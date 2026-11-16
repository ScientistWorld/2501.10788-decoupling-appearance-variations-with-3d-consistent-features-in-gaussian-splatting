# Evaluation Targets

## Primary: glav_novel_view (weight 0.85)

Held-out novel-view quality under global and local appearance variation (paper Table 1 protocol: fit appearance on the left half of a held-out capture, score the right half). 36 held-out captures over six hidden test scenes.

- **Metrics:** `psnr` (primary) and `ssim` are higher-is-better; `lpips` (v0.1, AlexNet) is lower-is-better.
- **Paper direction reproduced:** the reference beats vanilla 3DGS on all three metrics on the hidden test (PSNR 22.87 vs 18.70, SSIM 0.729 vs 0.714, LPIPS 0.158 vs 0.209) and beats the no-identity-regulariser ablation on all three. Against the uv+depth ablation it wins PSNR (22.87 vs 20.86) and LPIPS (0.158 vs 0.175) but not SSIM (0.729 vs 0.751).
- **Run:** `bash scripts/reproduce.sh`, then `bash scripts/evaluate_train.sh` (development scenes) / `bash scripts/evaluate_test.sh` (hidden scenes).

## Constraint and secondary: glav_average_appearance (weight 0.15)

The submitted 3D Gaussian scene, rendered by the evaluator with no appearance model at six canonical-only cameras per scene, colour-corrected to the camera's average appearance over the training captures (paper Table 4 protocol), scored on the full frame. It captures the paper's claim that the appearance module can be discarded after optimisation without floaters or colour distortions.

- **Metrics:** `psnr` (primary), `ssim` higher-is-better; `lpips` lower-is-better.
- **Constraint (gate):** a method more than 1.0 dB PSNR below the evaluator's frozen reference reconstruction (the reference run of the paper's method) fails the run (worst values everywhere).
- **Ablations (paper Table 4):** the reference beats `davigs_no_id` here (26.27 vs 24.25 dB). It does not beat `davigs_uv_depth` (28.11 dB) or vanilla 3DGS (27.37 dB); at this scale the full method's scene renders less cleanly than those two. This is a part of the paper's claim that the reproduction does not show, and it is room for a better method.

## Not scored

The paper's efficiency claims (training time, VRAM, FPS) depend on hardware and wall-clock time, and the PhotoTourism and plug-and-play experiments (Tables 2 and 3) need real captures and other renderers. None of these is scored. Their reported values are kept under `_paper_reported_unscored` in `scoring/reference.json`.
