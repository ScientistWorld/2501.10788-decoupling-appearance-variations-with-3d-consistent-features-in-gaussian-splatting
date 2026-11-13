#!/bin/bash
# Job script: build the benchmark if absent/outdated, choose the two appearance-module
# hyperparameters the paper leaves open on the development scenes, run the reference
# reproduction (vanilla 3DGS, DAVIGS, its two ablations), freeze the evaluator-owned
# comparison/gate predictions, score both slices, then the hardening audit (extra
# seeds, honest variants, shortcut cases through the hidden-test evaluator) and
# validation. Per-scene results are cached, so a resubmission resumes.
# SMOKE=1 runs every stage with a tiny optimisation budget (pipeline check).
set -uo pipefail
cd /home/user
SMOKE="${SMOKE:-1}"  # SMOKE TEST JOB: set back to 0 for the full run
STEPS_ARG=""
if [ "$SMOKE" = "1" ]; then STEPS_ARG="--steps 150"; fi

echo "== environment $(date)"
python -c "import torch, os; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'cpus', len(os.sched_getaffinity(0)))"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null || true

set -e
echo "== build benchmark $(date)"
python -m eval.test.build_benchmark
echo "== select module hyperparameters on the development scenes $(date)"
python -m method.run_experiment --plan select $STEPS_ARG --strict
python -m method.select_config
cat method/selected_config.json
echo "== reference methods $(date)"
python -m method.run_experiment --plan reference $STEPS_ARG --strict
echo "== freeze comparison and gate predictions $(date)"
python -m eval.test.freeze
python -m eval.train.freeze
echo "== evaluate $(date)"
bash scripts/evaluate.sh > /dev/null
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
set +e
echo "== audit runs (seeds and honest variants on the test scenes) $(date)"
python -m method.run_experiment --plan audit $STEPS_ARG
echo "== audit evaluation $(date)"
python -m redteam.hardening.audit all
echo "== validate $(date)"
python validate.py
echo "== done $(date)"
exit 0
