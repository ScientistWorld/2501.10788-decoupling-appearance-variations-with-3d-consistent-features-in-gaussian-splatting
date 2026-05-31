# Train/Test Evaluation Design Notes

## Checked

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The test wrapper invokes `eval.test.evaluate`, writes `scoring/scores_test.json`, and carries a test-specific output path.
- Checked `eval/train/` and `eval/test/` for cross-imports. Both slices use only neutral metric utilities from top-level `eval.metrics`; neither imports from the other slice.
- Checked for hardcoded method defaults. The train evaluator discovers every `outputs/*/predictions.pt` artifact, and the test evaluator follows the train score plan when present or discovers artifacts directly.
- Checked `scoring/reference.json` for paper-agreement metrics. The active metrics describe reconstruction quality, appearance constraints, components, or resource use; none score closeness to published values.
- Confirmed no `scoring/scores_train.json` or `scoring/scores_test.json` stubs are present.
- Ran `bash -n` on both train/test shell wrappers.

## Changed

- Replaced the contiguous held-out artifact split with an alternating split: train scores even-indexed held-out predictions and test scores odd-indexed held-out predictions. With the current four held-out views this balances camera/appearance phases better than scoring the first two views against the last two.
- Added `worst_view_psnr` to both slices. This makes a method that improves the average by sacrificing one held-out view easier to detect.
- Added `mean_color_bias_mae` to both slices. This targets global appearance shortcuts that can look acceptable under average pixel metrics while drifting in color.
- Added `edge_mae` to both slices. This adds a structure-sensitive check against blur or texture shortcuts beyond PSNR/SSIM.
- Updated `scoring/reference.json` so every active metric has `higher_is_better`, `coefficient`, and `description`, and so the new metrics are declared for both existing experiments.

## Runtime Concerns For The Runner Iteration

- The split is constrained by the existing artifact format: each method currently saves only four held-out right-half predictions. The alternating split improves balance but still leaves only two views per slice, so train/test variance may be noticeable.
- The added metrics are computed from existing `pred_right` and `target_right` tensors and should not require re-running the method. The runner should verify that all future method artifacts keep those tensor keys and shape conventions.
- `validate.py` appears to reject score methods not listed under `reference.json` results. I did not edit it because it is outside this iteration's allowed files, but future improve-mode runners should check whether arbitrary new method names are accepted by the scoring harness.
