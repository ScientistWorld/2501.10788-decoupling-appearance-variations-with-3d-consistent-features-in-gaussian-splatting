# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-05-31 12:05] - core_claim
- Completed a reduced-scale DAVIGS run on the synthetic GLAV-style benchmark.
- DAVIGS improved held-out PSNR, SSIM, and LPIPS-proxy over vanilla Gaussian Splatting while keeping the appearance module decoupled from final splat rendering.
- Ablation artifacts for 3D-aware features and identity regularization exist and pass local validation after the evaluator schema fix, but a clean container job is being submitted before claiming `secondary_claims`.

### [2026-05-31 12:07] - core_claim_plus
- Audited the inherited implementation and confirmed it contains the paper's actual DAVIGS computation at reduced scale: Gaussian splat rendering, depth back-projection, 3D hash-grid features, per-view appearance embeddings, MLP affine transforms, cell-based querying, and identity regularization.
- Confirmed `eval/` is method-independent and `python validate.py --compare` passes locally on the current artifacts.
- Keeping the milestone at the reset `core_claim_plus` while submitting a clean post-fix container job to validate the ablation outputs before restoring `secondary_claims`.
