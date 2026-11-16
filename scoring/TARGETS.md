# Evaluation Targets

All targets are scored by `bash scripts/evaluate_train.sh` (3 development scenes → `scoring/scores_train.json`; `scripts/evaluate.sh` writes the same to `scoring/scores.json`) and by the hidden test `bash scripts/evaluate_test.sh` (6 test scenes → `scoring/scores_test.json`). The reference reproduction is regenerated with `bash scripts/reproduce.sh`. The numbers below are hidden-test values from `scoring/scores_test.json`.

## Primary: glav_novel_view (weight 0.85)

What the paper optimises (Table 1 protocol): render a held-out capture of a scene whose training photos vary in global and local appearance. The capture's appearance may be inferred from its left half, and its right half is scored. 36 held-out captures over six hidden scenes.

- **Metrics:** `psnr` (primary, higher is better), `ssim` (higher is better), `lpips` (v0.1 AlexNet, lower is better).
- **Reference (`davigs`):** PSNR 22.87, SSIM 0.729, LPIPS 0.158.
- **Baseline (`vanilla_3dgs`):** PSNR 18.70, SSIM 0.714, LPIPS 0.209. The reference beats it on all three metrics, which is the paper's direction.

## Constraint: glav_average_appearance (weight 0.15, plus a hard gate)

What the paper must maintain: the reconstructed scene itself stays clean when rendered without any appearance model, so the appearance module can be discarded after optimisation. The evaluator renders each method's submitted 3D Gaussian scene at six canonical-only cameras per scene, colour-corrects each render to the camera's average appearance over the training captures (Table 4 protocol), and scores the full frame.

- **Metrics:** `psnr` (primary, higher is better), `ssim` (higher is better), `lpips` (lower is better).
- **Gate:** a run more than 1.0 dB PSNR below the evaluator's frozen reference reconstruction fails, with worst values in every metric. A scientist who corrupts the scene to win novel-view scores fails here.
- **Reference:** PSNR 26.27, SSIM 0.737, LPIPS 0.222.
- **Vanilla 3DGS:** PSNR 27.37, SSIM 0.797, LPIPS 0.147. At this scale the reference's scene is less clean than vanilla's, which leaves room for a better method.

## Ablations (paper Table 4), scored in both experiments

What breaks when a key idea is removed. Both ablations are fixed comparison entries in `scores_*.json` for both experiments.

- **`davigs_no_id`, no identity regulariser on the affine maps.**
  - Novel view: PSNR 22.78, SSIM 0.707, LPIPS 0.173.
  - Scene: PSNR 24.25, SSIM 0.647, LPIPS 0.253.
  - Worse than the reference on all six metrics, as the paper reports.
- **`davigs_uv_depth`, 3D hash-grid features replaced by encodings of pixel coordinates and depth.**
  - Novel view: PSNR 20.86, SSIM 0.751, LPIPS 0.175. Worse than the reference in PSNR and LPIPS, better in SSIM.
  - Scene: PSNR 28.11, SSIM 0.824, LPIPS 0.124. Better than the reference.
  - The paper's claim that 3D features win is reproduced on the main protocol's PSNR and LPIPS only.

## Not scored

The paper's efficiency claims (training time, VRAM, FPS) depend on hardware and wall-clock time, and the PhotoTourism and plug-and-play experiments (Tables 2 and 3) need real captures and other renderers. None of these is scored. Their reported values are kept under `_paper_reported_unscored` in `scoring/reference.json`.
