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

### [2026-05-31 13:10] - secondary_claims
- Tightened the standard evaluator to emit all diagnostics declared in `reference.json`.
- Updated `scripts/run.sh` to regenerate the full reproduction, train/test slice scores, and run `python validate.py --compare` in the container before any done signal.

### [2026-05-31 13:16] - secondary_claims
- Final verification passed after reverting an unvalidated cell-size extension from the active scoring contract.
- Confirmed `python validate.py --compare` passes with the validated core and ablation experiments.

### [2026-11-16] - secondary_claims (hardening, iteration 11)
- Reproduction restored toward the paper's settings: per-pixel cell query (the paper's 8 px at 1080p is sub-pixel at 96 px), hash-grid finest resolution and module learning rates (unspecified in the paper) chosen on development scenes.
- Hidden test: DAVIGS beats vanilla 3DGS on all three novel-view metrics (PSNR 22.87 vs 18.70, SSIM 0.729 vs 0.714, LPIPS 0.158 vs 0.209) and the no-L_ID ablation on all six metrics; the uv+depth ablation keeps a cleaner appearance-free scene (not reproduced at this scale).
- Appearance-free experiment now renders each method's submitted Gaussian scene inside the evaluator at canonical-only cameras; gate relative to the frozen reference reconstruction. See HARDENING.md.

## Stop Justification

Final milestone: **secondary_claims**. Iteration 11 (hardening) ended with the GPU budget exhausted (16.3 GPU-hours) after one full run and two smoke runs.

**Reproduced on the hidden test (6 scenes, 36 held-out captures):**
- **Paper Table 1 direction:** DAVIGS beats vanilla 3D Gaussian Splatting on all three novel-view metrics (PSNR 22.87 vs 18.70, SSIM 0.729 vs 0.714, LPIPS 0.158 vs 0.209).
- **Paper Table 4, identity regulariser:** removing it degrades every metric in both experiments.

**Not reproduced:**
- **3D features vs uv+depth (Table 4):** reproduced only in novel-view PSNR and LPIPS. With uv+depth features, the scene rendered without the appearance module is cleaner (AA 28.11 vs 26.27 dB).
- **"Fewer floaters than vanilla":** the reproduced DAVIGS scene renders less cleanly than vanilla's under the colour-corrected protocol.
- **Likely cause:** the scaled budget (6k instead of 30k iterations, 96x128 images, 6,000-Gaussian cap). A 1.5x-steps variant improves all six metrics, including +1.31 dB on the scene, but a rerun at a larger budget no longer fit the compute.
- **Not scored:** efficiency (training time, VRAM, FPS), PhotoTourism and plug-and-play (Tables 2-3). They need real captures, other renderers or wall-clock measurements, so they stop the milestone at secondary_claims.

**The gym** rewards better research: two honest variants beat the reference beyond seed noise, weaker ones score lower, and all REDTEAM shortcuts are closed. One scene-provenance shortcut remains for the grader. See HARDENING.md.
