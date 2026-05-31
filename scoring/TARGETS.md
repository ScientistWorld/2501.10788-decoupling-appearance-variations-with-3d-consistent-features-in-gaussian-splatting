# Evaluation Targets

## Primary: synthetic_glav_core

Tests whether a reconstruction method improves held-out novel-view image quality in the presence of global and local appearance variation.

- **Metrics:** `psnr` and `ssim` are higher better; `lpips_proxy` is lower better.
- **Robustness diagnostics:** `worst_view_psnr` is higher better; `mean_color_bias_mae` and `edge_mae` are lower better. These discourage methods that improve the mean by sacrificing one view, shifting global color, or blurring structure.
- **Run:** `bash scripts/reproduce.sh`, then `bash scripts/evaluate.sh`.

## Constraint: synthetic_glav_core

Tests whether quality gains are achieved without requiring a appearance model coupled into Gaussian splat rendering.

- **Metric:** `coupled_appearance_at_splat_render` is lower better; zero means appearance handling is not coupled into the Gaussian splat renderer.
- **Run:** `bash scripts/evaluate.sh` on method artifacts.

## Resource: synthetic_glav_core

Tracks the amount of optimization used in the reduced benchmark so improvements cannot be claimed solely by increasing training budget.

- **Metric:** `train_iterations` is lower better when quality is comparable.
- **Run:** `bash scripts/evaluate.sh` on method artifacts.

## Secondary: synthetic_glav_ablation

Tests whether controlled degraded variants lose rendering quality under the same reduced benchmark protocol.

- **Metrics:** `psnr` and `ssim` should be higher for the full method than ablations; `lpips_proxy` should be lower.
- **Robustness diagnostics:** `worst_view_psnr`, `mean_color_bias_mae`, and `edge_mae` are reported for the same held-out right-half predictions.
- **Run:** `bash scripts/reproduce.sh`, then `bash scripts/evaluate.sh`.

## Secondary: synthetic_glav_cell_size

Tests the cell-based query tradeoff from the paper's cell-size sweep: appearance quality should remain competitive when using a moderate cell size while reducing the number of appearance MLP queries per rendered view.

- **Metrics:** `psnr` and `ssim` are higher better; `lpips_proxy` is lower better.
- **Resource diagnostics:** `cell_queries_per_view` and `train_iterations` are lower better when quality is comparable. `cell_queries_per_view` replaces hardware-dependent wall-clock timing.
- **Run:** `bash scripts/reproduce.sh`, then `bash scripts/evaluate.sh`.
