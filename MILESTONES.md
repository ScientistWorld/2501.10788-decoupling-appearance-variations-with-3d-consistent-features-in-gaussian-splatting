# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-05-31 11:53] - method_runs
- Implemented reduced-scale DAVIGS with differentiable Gaussian splatting, rendered depth back-projection, 3D multi-resolution hash-grid features, per-view appearance embeddings, MLP-decoded affine color transforms, cell-based query, and identity regularization.
- Ran the method and evaluator end to end on the synthetic GLAV-style benchmark.

### [2026-05-31 11:53] - core_claim
- Reproduced the core claim on a reduced GLAV-style appearance-variation benchmark.
- DAVIGS improved held-out right-half test PSNR over vanilla 3DGS from 22.64 to 24.50 while preserving the decoupled rendering constraint.
- `python validate.py --compare` passes with independent evaluation from saved prediction artifacts.

### [2026-05-31 12:10] - core_claim
- Continuing-agent audit found the core implementation is a real reduced DAVIGS reproduction rather than an input-output surrogate.
- Added secondary ablation targets for the paper's 3D-aware feature and identity-regularization claims; submitted a job to generate those scores.

### [2026-05-31 12:25] - secondary_claims
- Job `bf35e825-b46` ran the core benchmark plus ablations and produced valid artifacts.
- Fixed evaluator schema filtering so each experiment writes only metrics declared in `reference.json`.
- The full DAVIGS variant outperformed both ablations on held-out PSNR, SSIM, and LPIPS-proxy, supporting the paper's secondary claims about 3D-aware features and identity regularization.
