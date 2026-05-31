# Reproduction Milestones

**Current: core_claim**

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
