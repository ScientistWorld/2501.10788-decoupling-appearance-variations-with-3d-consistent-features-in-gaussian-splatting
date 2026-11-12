# Evaluation

## What a method submits

`outputs/<method>/predictions.pt`, a `torch.save`d dict of plain tensors:

```
{"novel_view":         {scene_id: float [6, 3, 96, 64]},    # right halves of the 6 held-out captures
 "average_appearance": {scene_id: float [6, 3, 96, 128]}}   # each held-out camera rendered without any per-capture appearance state
```

for every scene in `data/glav_syn/manifest.json`, values in [0, 1], held-out captures in file order. Nothing else in the file or directory is read: no self-reported metrics, flags, targets or iteration counts.

## Experiments and metrics

- **glav_novel_view** (weight 0.7): PSNR, SSIM (11x11 Gaussian window) and LPIPS (v0.1, AlexNet) of the predicted right halves against the captured right halves. A held-out capture's appearance may be inferred only from its left half.
- **glav_average_appearance** (weight 0.3): the appearance-free render of each held-out camera is colour-corrected by the evaluator (the mip-NeRF 360 per-image quadratic colour warp) to that camera's average appearance over the training capture conditions, then scored with PSNR, SSIM and LPIPS on the full image. This measures whether the reconstructed scene itself is clean.

Each metric is the mean over all held-out captures of the slice. Higher PSNR/SSIM and lower LPIPS are better.

## Constraint

The appearance-free render is also a gate. If a method's `glav_average_appearance` PSNR is more than 1 dB below that of the evaluator's frozen vanilla Gaussian-splatting reconstruction, the run counts as failed.

## Failure handling

A missing, unreadable or malformed `predictions.pt` fails the run. So does a missing scene, a wrong shape, non-finite values, values outside [0, 1] or a failed constraint. A failed run receives the worst value of every metric in every experiment: PSNR 0, SSIM −1, LPIPS 1.

## Baselines and comparisons

The plain Gaussian-splatting baseline (`vanilla_3dgs`) and two controlled degraded variants (`davigs_uv_depth`, `davigs_no_id`) are fixed comparison points. Their scores come from frozen predictions owned by the evaluator, so files under `outputs/` with those names are ignored.

## Slices

- `bash scripts/evaluate_train.sh` (and `scripts/evaluate.sh`) score the three development scenes, whose targets are visible. They write `scoring/scores_train.json` (`scoring/scores.json`).
- The hidden test (`scripts/evaluate_test.sh`, no arguments) scores the six test scenes with the same metrics and writes `scoring/scores_test.json`. Both evaluators score every method directory under `outputs/`.
