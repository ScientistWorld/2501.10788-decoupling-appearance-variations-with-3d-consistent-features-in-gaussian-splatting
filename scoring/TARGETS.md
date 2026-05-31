# Evaluation Targets

## Primary: synthetic_glav_core

Tests whether a reconstruction method improves held-out novel-view image quality in the presence of global and local appearance variation.

- **Metrics:** `psnr` and `ssim` are higher better; `lpips_proxy` is lower better.
- **Run:** `bash scripts/reproduce.sh`, then `bash scripts/evaluate.sh`.

## Constraint: synthetic_glav_core

Tests whether quality gains are achieved without requiring a appearance model coupled into Gaussian splat rendering.

- **Metric:** `coupled_appearance_at_splat_render` is lower better; zero means appearance handling is not coupled into the Gaussian splat renderer.
- **Run:** `bash scripts/evaluate.sh` on method artifacts.

## Resource: synthetic_glav_core

Tracks the amount of optimization used in the reduced benchmark so improvements cannot be claimed solely by increasing training budget.

- **Metric:** `train_iterations` is lower better when quality is comparable.
- **Run:** `bash scripts/evaluate.sh` on method artifacts.
