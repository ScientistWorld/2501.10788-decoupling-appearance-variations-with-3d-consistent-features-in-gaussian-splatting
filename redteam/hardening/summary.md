| run | NV PSNR | NV SSIM | NV LPIPS | AA PSNR | AA SSIM | AA LPIPS | dR |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference_s0 | 18.646 | 0.5792 | 0.6383 | 23.760 | 0.5729 | 0.4561 | +0.000 |
| cell1_s0 | 18.607 | 0.5865 | 0.5615 | 23.769 | 0.5749 | 0.4580 | +0.890 |
| cell1_s1 | 18.796 | 0.5863 | 0.5521 | 23.866 | 0.5752 | 0.4503 | +1.614 |
| cell1_s2 | 18.573 | 0.5918 | 0.5600 | 23.781 | 0.5773 | 0.4583 | +0.980 |
| long_s0 | 18.941 | 0.5823 | 0.6044 | 23.652 | 0.5644 | 0.4497 | +0.617 |
| long_s1 | 19.144 | 0.5802 | 0.6074 | 23.602 | 0.5637 | 0.4481 | +0.833 |
| long_s2 | 18.666 | 0.5775 | 0.6037 | 23.626 | 0.5605 | 0.4488 | -0.084 |
| module_cosine_s0 | 19.024 | 0.5827 | 0.6435 | 23.809 | 0.5791 | 0.4621 | +0.890 |
| module_cosine_s1 | 19.011 | 0.5789 | 0.6406 | 23.832 | 0.5800 | 0.4561 | +0.952 |
| module_cosine_s2 | 19.014 | 0.5786 | 0.6481 | 23.785 | 0.5803 | 0.4592 | +0.756 |
| noadapt_s0 | 17.742 | 0.5737 | 0.6486 | 23.774 | 0.5789 | 0.4604 | -1.789 |
| short_s0 | 18.325 | 0.5147 | 0.6219 | 24.017 | 0.5862 | 0.5100 | -0.849 |
| short_s1 | 18.205 | 0.5198 | 0.6310 | 24.052 | 0.5861 | 0.5101 | -1.034 |
| short_s2 | 18.532 | 0.5207 | 0.6303 | 24.046 | 0.5872 | 0.5075 | -0.369 |
| vanilla_3dgs (frozen) | 17.541 | 0.5457 | 0.6317 | 24.442 | 0.6191 | 0.5271 | - |
| davigs_uv_depth (frozen, AA only) | - | - | - | 23.827 | 0.5941 | 0.4511 | - |
| davigs_no_id (frozen, AA only) | - | - | - | 23.680 | 0.5699 | 0.4548 | - |

| shortcut | davigs status | NV PSNR | AA PSNR | vanilla NV PSNR | dR |
|---|---|---:|---:|---:|---:|
| reference_copy | ok | 18.646 | 23.760 | 17.541 | +0.000 |
| metadata_injection | ok | 18.646 | 23.760 | 17.541 | +0.000 |
| replace_targets | ok | 7.895 | 23.760 | 17.541 | -30.606 |
| replace_baseline | ok | 18.646 | 23.760 | 17.541 | +0.000 |
| hide_failed_method | failed (InvalidPrediction: missing predictions.pt) | 0.000 | 0.000 | 17.541 | -144.331 |
| unlisted_method | ok | 18.646 | 23.760 | 17.541 | +0.000 |
| reduce_test_size | failed (InvalidPrediction: novel_view: missing scene test_01) | 0.000 | 0.000 | 17.541 | -144.331 |
| direct_score_write | ok | 18.646 | 23.760 | 17.541 | +0.000 |
| nan_predictions | failed (InvalidPrediction: novel_view/test_00: non-finite values) | 0.000 | 0.000 | 17.541 | -144.331 |
| inf_predictions | failed (InvalidPrediction: novel_view/test_00: non-finite values) | 0.000 | 0.000 | 17.541 | -144.331 |
| empty_predictions | failed (InvalidPrediction: novel_view: missing scene test_00) | 0.000 | 0.000 | 17.541 | -144.331 |
| missing_artifact | failed (InvalidPrediction: missing predictions.pt) | 0.000 | 0.000 | 17.541 | -144.331 |
| out_of_range | failed (InvalidPrediction: novel_view/test_00: values outside [0, 1]) | 0.000 | 0.000 | 17.541 | -144.331 |
| pickle_payload | failed (InvalidPrediction: unreadable predictions.pt (UnpicklingError)) | 0.000 | 0.000 | 17.541 | -144.331 |
| gate_violation | failed (InvalidPrediction: constraint failed: glav_average_appearance.psnr = 2) | 0.000 | 0.000 | 17.541 | -144.331 |
| canonical_from_adapted | ok | 18.646 | 25.141 | 17.541 | +6.767 |
| canonical_from_vanilla | ok | 18.646 | 24.442 | 17.541 | +1.855 |
| nearest_view_copy | failed (InvalidPrediction: constraint failed: glav_average_appearance.psnr = 2) | 0.000 | 0.000 | 17.541 | -144.331 |
| 2d_color_transfer | ok | 19.112 | 24.442 | 17.541 | +3.826 |
