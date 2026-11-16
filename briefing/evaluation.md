# Evaluation

## What a method submits

`outputs/<method>/predictions.pt`, a `torch.save`d dict of plain tensors (loaded with `weights_only=True`):

```
{"novel_view": {scene_id: float [6, 3, 96, 64]},          # right halves of the 6 held-out captures, values in [0, 1]
 "scene":      {scene_id: {"means":   float [N, 3],        # the reconstructed 3D Gaussians (world frame)
                           "scales":  float [N, 3],        #   standard deviations along the local axes, > 0
                           "quats":   float [N, 4],        #   rotation (w, x, y, z), normalised by the evaluator
                           "opacity": float [N],           #   in [0, 1]
                           "sh":      float [N, 16, 3]}}}  #   degree-3 spherical-harmonic colour (3DGS convention: rgb = SH + 0.5)
```

for every scene in `data/glav_syn/manifest.json`, held-out captures in file order, 1 ≤ N ≤ 60,000 Gaussians per scene. Nothing else in the file or directory is read: no self-reported metrics, flags, targets, images of the scene, or iteration counts.

## Experiments and metrics

- **glav_novel_view** (weight 0.7): PSNR, SSIM (11x11 Gaussian window) and LPIPS (v0.1, AlexNet) of the predicted right halves against the captured right halves. A held-out capture's appearance may be inferred only from its left half.
- **glav_average_appearance** (weight 0.3): the evaluator renders the submitted Gaussian scene itself, with its own 3D Gaussian Splatting rasteriser (EWA projection, depth-sorted alpha compositing, black background, no appearance model of any kind), at six canonical-only cameras per scene (`canonical_R/t/K` in each scene file). No capture exists at these viewpoints. Each render is colour-corrected by the evaluator (the mip-NeRF 360 per-image quadratic colour warp) to that camera's average appearance over the training capture conditions, then scored with PSNR, SSIM and LPIPS on the full frame. This measures whether the reconstructed scene itself is clean.

Each metric is the mean over the slice's held-out captures (novel view) or canonical views (average appearance). Higher PSNR/SSIM and lower LPIPS are better.

## Constraint

The appearance-free scene is also a gate. If a method's `glav_average_appearance` PSNR is more than 1 dB below that of the evaluator's frozen reference reconstruction (the reference run of the paper's method), the run counts as failed.

## Failure handling

A missing, unreadable or malformed `predictions.pt` fails the run. So does a missing scene, a wrong shape, non-finite values, novel-view values outside [0, 1], an invalid Gaussian scene (opacity outside [0, 1], non-positive scales, zero quaternions, more than 60,000 Gaussians, footprints too large to render) or a failed constraint. A failed run receives the worst value of every metric in every experiment: PSNR 0, SSIM −1, LPIPS 1.

## Baselines and comparisons

The plain Gaussian-splatting baseline (`vanilla_3dgs`) and two controlled degraded variants (`davigs_uv_depth`, `davigs_no_id`) are fixed comparison points in both experiments. Their scores come from frozen predictions owned by the evaluator, so files under `outputs/` with those names are ignored.

## Slices

- `bash scripts/evaluate_train.sh` (and `scripts/evaluate.sh`) score the three development scenes, whose targets are visible. They write `scoring/scores_train.json` (`scoring/scores.json`).
- The hidden test (`scripts/evaluate_test.sh`, no arguments) scores the six test scenes with the same metrics and writes `scoring/scores_test.json`. Both evaluators score every method directory under `outputs/`.
