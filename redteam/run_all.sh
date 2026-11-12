#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
bash redteam/shortcut_generator_answers/run.sh --resume
bash redteam/shortcut_bruteforce_seed/run.sh --resume
bash redteam/shortcut_replace_targets/run.sh --resume
bash redteam/shortcut_zero_target_constraint_failure/run.sh --resume
bash redteam/shortcut_replace_baseline/run.sh --resume
bash redteam/shortcut_default_metadata/run.sh --resume
bash redteam/shortcut_negative_metadata/run.sh --resume
bash redteam/shortcut_constant_negative_metadata/run.sh --resume
bash redteam/shortcut_hide_failed_method/run.sh --resume
bash redteam/shortcut_unlisted_method/run.sh --resume
bash redteam/shortcut_reduce_test_size/run.sh --resume
bash redteam/shortcut_direct_score_write/run.sh --resume
bash redteam/shortcut_nan_predictions/run.sh --resume
bash redteam/shortcut_empty_predictions/run.sh --resume
bash redteam/shortcut_missing_artifact/run.sh --resume
bash redteam/shortcut_constant_real_targets/run.sh --resume
bash redteam/shortcut_infinite_real_targets/run.sh --resume
bash redteam/reference_saved/run.sh --resume
bash redteam/reference_seed_7/run.sh --resume
bash redteam/reference_seed_8/run.sh --resume
bash redteam/reference_seed_9/run.sh --resume
bash redteam/better_more_training/run.sh --resume
bash redteam/better_lr_decay_7/run.sh --resume
bash redteam/better_lr_decay_8/run.sh --resume
bash redteam/better_lr_decay_9/run.sh --resume
bash redteam/better_local_ssim/run.sh --resume
bash redteam/weaker_few_steps/run.sh --resume
bash redteam/weaker_no_appearance/run.sh --resume
python -m redteam.analyze | tee redteam/results_table.txt
