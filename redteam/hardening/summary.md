| run | NV PSNR | NV SSIM | NV LPIPS | AA PSNR | AA SSIM | AA LPIPS | dR |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference_s0 | 22.866 | 0.7294 | 0.1579 | 26.274 | 0.7372 | 0.2219 | +0.000 |
| reference_s1 | 22.741 | 0.7242 | 0.1606 | 26.643 | 0.7540 | 0.2179 | +0.059 |
| reference_s2 | 22.242 | 0.7232 | 0.1715 | 26.601 | 0.7472 | 0.2233 | -0.630 |
| module_cosine_s0 | 24.112 | 0.7407 | 0.1478 | 25.713 | 0.7036 | 0.2310 | +0.595 |
| module_cosine_s1 | 23.968 | 0.7390 | 0.1500 | 25.344 | 0.7039 | 0.2416 | +0.118 |
| long_s0 | 23.548 | 0.7449 | 0.1459 | 27.584 | 0.7792 | 0.1996 | +2.180 |
| short_s0 | 0.000 | -1.0000 | 1.0000 | 0.000 | -1.0000 | 1.0000 | -108.870 |
| noadapt_s0 | 19.646 | 0.6976 | 0.1903 | 26.215 | 0.7357 | 0.2207 | -3.286 |
| module_cosine_s2 | 23.871 | 0.7360 | 0.1571 | 25.706 | 0.7150 | 0.2326 | +0.165 |
| vanilla_3dgs (frozen comparison) | 18.704 | 0.7140 | 0.2086 | 27.367 | 0.7967 | 0.1474 | - | dR if submitted: -1.979
| davigs_uv_depth (frozen comparison) | 20.863 | 0.7506 | 0.1747 | 28.112 | 0.8244 | 0.1239 | - | dR if submitted: +1.712
| davigs_no_id (frozen comparison) | 22.783 | 0.7068 | 0.1725 | 24.251 | 0.6466 | 0.2531 | - | dR if submitted: -2.859

| shortcut | davigs status | NV PSNR | AA PSNR | dR |
|---|---|---:|---:|---:|
| reference_copy | ok | 22.866 | 26.274 | +0.000 |
| metadata_injection | ok | 22.866 | 26.274 | +0.000 |
| replace_targets | ok | 7.895 | 26.274 | -43.931 |
| replace_baseline_and_gate | ok | 22.866 | 26.274 | +0.000 |
| hide_failed_method | failed (InvalidPrediction: missing predictions.pt) | 0.000 | 0.000 | -108.870 |
| unlisted_method | ok | 22.866 | 26.274 | +0.000 |
| reduce_test_size | failed (InvalidPrediction: novel_view/test_01: missing) | 0.000 | 0.000 | -108.870 |
| direct_score_write | ok | 22.866 | 26.274 | +0.000 |
| nan_predictions | failed (InvalidPrediction: novel_view/test_00: non-finite values) | 0.000 | 0.000 | -108.870 |
| nan_scene | failed (InvalidPrediction: scene/test_00: 'means' has non-finite values) | 0.000 | 0.000 | -108.870 |
| inf_predictions | failed (InvalidPrediction: novel_view/test_00: non-finite values) | 0.000 | 0.000 | -108.870 |
| empty_predictions | failed (InvalidPrediction: novel_view/test_00: missing) | 0.000 | 0.000 | -108.870 |
| missing_artifact | failed (InvalidPrediction: missing predictions.pt) | 0.000 | 0.000 | -108.870 |
| out_of_range | failed (InvalidPrediction: novel_view/test_00: values outside [0, 1]) | 0.000 | 0.000 | -108.870 |
| pickle_payload | failed (InvalidPrediction: unreadable predictions.pt (UnpicklingError)) | 0.000 | 0.000 | -108.870 |
| images_instead_of_scene | failed (InvalidPrediction: missing 'scene' section) | 0.000 | 0.000 | -108.870 |
| too_many_gaussians | failed (InvalidPrediction: scene/test_00: 65956 Gaussians (allowed 1..60000)) | 0.000 | 0.000 | -108.870 |
| gate_violation_transparent_scene | failed (InvalidPrediction: constraint failed: glav_average_appearance.psnr = 17.690 < 25) [unscored: NV 22.866, AA 17.690] | 0.000 | 0.000 | -108.870 |
| canonical_from_vanilla | ok | 22.866 | 27.368 | +1.826 |
| canonical_via_module_images | ok | 22.866 | 26.274 | +0.000 |
| nearest_view_copy_points_scene | failed (InvalidPrediction: constraint failed: glav_average_appearance.psnr = 23.240 < 25) [unscored: NV 16.149, AA 23.240] | 0.000 | 0.000 | -108.870 |
| 2d_color_transfer_points_scene | failed (InvalidPrediction: constraint failed: glav_average_appearance.psnr = 23.240 < 25) [unscored: NV 19.112, AA 23.240] | 0.000 | 0.000 | -108.870 |
| 2d_color_transfer_vanilla_scene | ok | 19.112 | 27.368 | -12.841 |
