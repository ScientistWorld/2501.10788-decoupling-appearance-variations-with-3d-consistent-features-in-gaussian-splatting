# Progress

## What Works

- Audited the inherited workspace and found no real implementation in `method/` or `eval/`.
- Implemented a reduced-scale DAVIGS reproduction with differentiable Gaussian splatting, 3D back-projection, multi-resolution hash-grid features, per-view appearance embeddings, MLP-decoded 3x4 affine color transforms, cell-based querying, and identity regularization.
- Added an independent evaluator that reads saved prediction artifacts and writes `scoring/scores.json`.
- Added a deterministic synthetic GLAV-style benchmark because the paper's captured GLAV dataset is not included.
- `python validate.py --reference-only` passes.

## Results

No reproduced scores have been claimed yet. Job `49cf5a13-8ce` generated both prediction artifacts but failed inside the evaluator before writing `scoring/scores.json`.

## Remaining

- Run the compute job and inspect whether DAVIGS improves PSNR/SSIM and perceptual proxy over vanilla 3DGS.
- If the gap is weak, tune the reduced benchmark or training schedule while keeping the DAVIGS computation intact.
- After a successful run, update the milestone and compare with `python validate.py --compare`.

## Issues

- The previous workspace was mostly template files.
- The original GLAV dataset and official code were not present. The project page found via web search did not expose a code repository in the visible page text.
- Job `49cf5a13-8ce` failed because the LPIPS-proxy Sobel kernel had shape `[3, 2, 3, 3]` while grouped convolution expected one input channel per group. Fixed by constructing `[6, 1, 3, 3]` depthwise filters.
- The job also warned that NumPy was absent from the container overlay. Added `numpy` to `environment/setup.sh`.

## Deviations from Paper

- **Dataset:** The paper evaluates on captured GLAV and PhotoTourism scenes. This reproduction uses a deterministic procedural GLAV-style scene because the captured GLAV data is not bundled and no direct public download was available in the workspace. The synthetic data preserves the key phenomenon: multi-view Gaussian reconstruction under global and local appearance variation.
- **Scale:** The paper trains full 3DGS scenes for 30k iterations at high resolution. This reproduction uses 64x64 images, a small Gaussian set, and hundreds of iterations so it can run within the benchmark budget while preserving the same appearance-model computation.
- **Perceptual metric:** The paper reports LPIPS. This environment uses a deterministic gradient-weighted LPIPS proxy to avoid external pretrained metric downloads.
