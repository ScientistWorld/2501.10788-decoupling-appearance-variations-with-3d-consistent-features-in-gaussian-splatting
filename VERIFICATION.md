# Verification Report

## Initial Assessment

The workspace entered final verification at `secondary_claims`. The claimed milestone is justified: the reduced DAVIGS implementation runs on a deterministic GLAV-style benchmark, improves held-out quality over the vanilla Gaussian Splatting baseline, and includes controlled degraded variants that score worse than the full reproduction.

`python validate.py --compare` passed at the start of verification, and the required train/test split artifacts were present.

## Issues Found

- **Minor:** `scoring/EXPERIMENTS.md`, `scoring/TARGETS.md`, and `scoring/CONSTRAINTS.md` leaked method-component details into files that should be method-agnostic or method-safe.
- **Minor:** `scoring/reference.json` gave positive coefficient weight to component-presence flags. Those flags reward a particular implementation choice rather than scientific output, so they were poor gym incentives.
- **Minor:** `scoring/scores.json` did not include the diagnostic metrics declared in `reference.json`, even though the train/test score files already contained them.
- **Minor:** `eval/evaluate_outputs.py` was tied to the current output directory names and would not automatically score a future scientist's new method artifact.

## Fixes Applied

- Reworded method-agnostic and method-safe scoring docs to describe ablations as controlled degraded variants without naming architecture or loss details.
- Removed component-presence flags from scored metrics and score artifacts; the ablation is now scored by image quality and robustness metrics only.
- Added `worst_view_psnr`, `mean_color_bias_mae`, and `edge_mae` to `scoring/scores.json` using the existing validated train/test artifacts.
- Updated the standard evaluator to discover `outputs/*/predictions.pt` and use the scoring plan to group methods, so new method outputs can be evaluated without editing `eval/`.
- Re-ran `python validate.py --compare`; all checks passed.

## Final Judgment

The workspace is ready to serve as a gym, with a clear limitation: it is a reduced synthetic benchmark rather than the full captured GLAV/PhotoTourism reproduction. Problem/solution separation is acceptable, no committed code references `shared/`, `scripts/download.sh` is self-contained with no external assets, and train/test evaluators are independent and lightweight.

- Milestone: secondary_claims
- Ready for gym use: yes
- Confidence: medium
- Key limitation: synthetic reduced-scale data only, with simplified Gaussian Splatting rather than full paper-scale 3DGS.
