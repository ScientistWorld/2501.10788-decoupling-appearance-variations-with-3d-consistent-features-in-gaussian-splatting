# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-31 12:05] - core_claim
- Completed a reduced-scale DAVIGS run on the synthetic GLAV-style benchmark.
- DAVIGS improved held-out PSNR, SSIM, and LPIPS-proxy over vanilla Gaussian Splatting while keeping the appearance module decoupled from final splat rendering.
- Ablation artifacts for 3D-aware features and identity regularization exist and pass local validation after the evaluator schema fix, but a clean container job is being submitted before claiming `secondary_claims`.
