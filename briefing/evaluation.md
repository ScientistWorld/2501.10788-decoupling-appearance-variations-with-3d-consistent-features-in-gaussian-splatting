# Evaluation

## Metrics

The main quality metrics are PSNR and SSIM, where higher is better, and a perceptual distance metric, where lower is better. The original paper reports LPIPS; this reduced environment uses a deterministic gradient-weighted LPIPS proxy to avoid external pretrained metric downloads while retaining a perceptual constraint.

The main constraint metric is whether extra per-view correction machinery is required during final rendering. A value of zero means the learned scene can be rendered without persistent per-image appearance state. Training iterations are also tracked as a resource constraint.

## Baselines and Targets

The primary baseline is a plain scene representation optimized directly against appearance-varying images. A successful submission should outperform that baseline on the held-out right halves of test images while avoiding persistent per-view correction at final rendering.

Paper reference values in `scoring/reference.json` come from the GLAV dataset results reported in Table 1. The local reproduction uses a deterministic synthetic GLAV-style benchmark because the paper's captured GLAV dataset is not distributed in this workspace.

## Evaluation Protocol

Evaluation reads prediction artifacts from `outputs/<method>/predictions.pt` and computes metrics independently in `eval/`. The evaluator never imports from `method/`.

For the reduced benchmark, methods train on a fixed procedural multi-view scene with global and local appearance variation. Test-view appearance fitting may use the left half of each test image, and metrics are computed only on the held-out right half, following the paper's half-image test adaptation protocol.
