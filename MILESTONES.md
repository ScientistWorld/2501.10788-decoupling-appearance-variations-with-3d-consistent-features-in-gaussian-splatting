# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-05-31 12:05] - core_claim
- Completed a reduced-scale DAVIGS run on the synthetic GLAV-style benchmark.
- DAVIGS improved held-out PSNR, SSIM, and LPIPS-proxy over vanilla Gaussian Splatting while keeping the appearance module decoupled from final splat rendering.
- Ablation artifacts for 3D-aware features and identity regularization exist and pass local validation after the evaluator schema fix, but a clean container job is being submitted before claiming `secondary_claims`.

### [2026-05-31 12:07] - core_claim_plus
- Audited the inherited implementation and confirmed it contains the paper's actual DAVIGS computation at reduced scale: Gaussian splat rendering, depth back-projection, 3D hash-grid features, per-view appearance embeddings, MLP affine transforms, cell-based querying, and identity regularization.
- Confirmed `eval/` is method-independent and `python validate.py --compare` passes locally on the current artifacts.
- Keeping the milestone at the reset `core_claim_plus` while submitting a clean post-fix container job to validate the ablation outputs before restoring `secondary_claims`.

### [2026-05-31 12:46] - secondary_claims
- Job `d1a7d3c1-6e7` completed successfully in the container and passed `python validate.py --compare`.
- Restored `secondary_claims`: full DAVIGS outperformed vanilla 3DGS on the reduced GLAV-style core benchmark, and the full model outperformed both ablations for 3D hash features and identity regularization.
- Stopping here because higher milestones would require substantially broader reproduction of unavailable captured GLAV/PhotoTourism experiments or additional full Gaussian Splatting baselines beyond the validated reduced benchmark.
