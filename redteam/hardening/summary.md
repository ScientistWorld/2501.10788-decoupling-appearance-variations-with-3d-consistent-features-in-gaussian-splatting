| run | NV PSNR | NV SSIM | NV LPIPS | AA PSNR | AA SSIM | AA LPIPS | dR |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference_s0 | 17.824 | 0.5864 | 0.5518 | 23.740 | 0.5734 | 0.4565 | +0.000 |
| cell1_s0 | 0.000 | -1.0000 | 1.0000 | 0.000 | -1.0000 | 1.0000 | -172.732 |
| cell1_s1 | 0.000 | -1.0000 | 1.0000 | 0.000 | -1.0000 | 1.0000 | -172.732 |
| cell1_s2 | 0.000 | -1.0000 | 1.0000 | 0.000 | -1.0000 | 1.0000 | -172.732 |
| long_s0 | 18.185 | 0.5982 | 0.4875 | 23.617 | 0.5674 | 0.4307 | +0.533 |
| long_s1 | 0.000 | -1.0000 | 1.0000 | 0.000 | -1.0000 | 1.0000 | -172.732 |
| long_s2 | 0.000 | -1.0000 | 1.0000 | 0.000 | -1.0000 | 1.0000 | -172.732 |
| module_cosine_s0 | 19.024 | 0.5927 | 0.5798 | 23.816 | 0.5813 | 0.4560 | +0.932 |
| module_cosine_s1 | 18.991 | 0.5969 | 0.5793 | 23.837 | 0.5800 | 0.4472 | +1.052 |
| module_cosine_s2 | 19.134 | 0.5997 | 0.5692 | 23.769 | 0.5780 | 0.4516 | +0.923 |
| noadapt_s0 | 14.324 | 0.5097 | 0.5520 | 23.746 | 0.5730 | 0.4590 | -2.579 |
| reference_s1 | 18.052 | 0.5892 | 0.5452 | 23.774 | 0.5763 | 0.4479 | +0.501 |
| reference_s2 | 17.006 | 0.5800 | 0.5531 | 23.709 | 0.5739 | 0.4528 | -0.547 |
| short_s0 | 18.078 | 0.5186 | 0.6220 | 23.967 | 0.5770 | 0.5074 | -0.820 |
| short_s1 | 0.000 | -1.0000 | 1.0000 | 0.000 | -1.0000 | 1.0000 | -172.732 |
| short_s2 | 0.000 | -1.0000 | 1.0000 | 0.000 | -1.0000 | 1.0000 | -172.732 |
| vanilla_3dgs (frozen comparison) | 17.449 | 0.5430 | 0.6394 | 24.445 | 0.6115 | 0.5274 | - | dR if submitted: +1.683
| davigs_uv_depth (frozen comparison) | 17.583 | 0.5576 | 0.6105 | 23.828 | 0.5935 | 0.4594 | - | dR if submitted: -0.000
| davigs_no_id (frozen comparison) | 17.736 | 0.5910 | 0.5228 | 23.696 | 0.5713 | 0.4450 | - | dR if submitted: +0.167

| shortcut | davigs status | NV PSNR | AA PSNR | dR |
|---|---|---:|---:|---:|
| reference_copy | ok | 17.824 | 23.740 | +0.000 |
| metadata_injection | ok | 17.824 | 23.740 | +0.000 |
| replace_targets | ok | 7.895 | 23.740 | -14.469 |
| replace_baseline_and_gate | ok | 17.824 | 23.740 | +0.000 |
| hide_failed_method | failed (InvalidPrediction: missing predictions.pt) | 0.000 | 0.000 | -172.732 |
| unlisted_method | ok | 17.824 | 23.740 | +0.000 |
| reduce_test_size | failed (InvalidPrediction: novel_view: missing scene test_01) | 0.000 | 0.000 | -172.732 |
| direct_score_write | ok | 17.824 | 23.740 | +0.000 |
| nan_predictions | failed (InvalidPrediction: novel_view/test_00: non-finite values) | 0.000 | 0.000 | -172.732 |
| inf_predictions | failed (InvalidPrediction: novel_view/test_00: non-finite values) | 0.000 | 0.000 | -172.732 |
| empty_predictions | failed (InvalidPrediction: novel_view: missing scene test_00) | 0.000 | 0.000 | -172.732 |
| missing_artifact | failed (InvalidPrediction: missing predictions.pt) | 0.000 | 0.000 | -172.732 |
| out_of_range | failed (InvalidPrediction: novel_view/test_00: values outside [0, 1]) | 0.000 | 0.000 | -172.732 |
| pickle_payload | failed (InvalidPrediction: unreadable predictions.pt (UnpicklingError)) | 0.000 | 0.000 | -172.732 |
| gate_violation_flat_canonical | failed (InvalidPrediction: constraint failed: glav_average_appearance.psnr = 20.230 < 22) [unscored: NV 17.824, AA 20.230] | 0.000 | 0.000 | -172.732 |
| canonical_from_vanilla | ok | 17.824 | 24.445 | +3.236 |
| canonical_via_module | ok | 17.824 | 25.185 | +8.195 |
| nearest_view_copy | failed (InvalidPrediction: constraint failed: glav_average_appearance.psnr = 22.700 < 22) [unscored: NV 16.149, AA 22.700] | 0.000 | 0.000 | -172.732 |
| 2d_color_transfer | failed (InvalidPrediction: constraint failed: glav_average_appearance.psnr = 22.700 < 22) [unscored: NV 19.112, AA 22.700] | 0.000 | 0.000 | -172.732 |
| 2d_color_transfer_vanilla_canonical | ok | 19.112 | 24.445 | +4.167 |
