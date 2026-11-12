#!/bin/bash
# Job script: rebuild-if-needed, reference reproduction, frozen comparisons,
# both evaluation slices, then the hardening audit (seeds, honest variants,
# shortcut cases through the hidden-test evaluator) and validation.
# SMOKE=1 runs every stage with a tiny optimisation budget (pipeline check).
set -uo pipefail
cd /home/user
BUILD_ARGS="${BUILD_ARGS:---force}"  # one-time rebuild: answers now store right halves only
SMOKE="${SMOKE:-0}"
STEPS_ARG=""
if [ "$SMOKE" = "1" ]; then STEPS_ARG="--steps 150"; fi

echo "== environment"
python -c "import torch, os; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'cpus', len(os.sched_getaffinity(0)))"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

set -e
echo "== build benchmark $(date)"
python -m eval.test.build_benchmark ${BUILD_ARGS:-}
echo "== reference methods $(date)"
python -m method.run_experiment --plan reference $STEPS_ARG --strict
echo "== freeze comparison methods"
python -m eval.test.freeze
python -m eval.train.freeze
echo "== evaluate"
bash scripts/evaluate.sh > /dev/null
bash scripts/evaluate_train.sh > /dev/null
bash scripts/evaluate_test.sh
set +e
echo "== audit runs (seeds and honest variants on the test scenes) $(date)"
python -m method.run_experiment --plan audit $STEPS_ARG
echo "== audit evaluation $(date)"
python -m redteam.hardening.audit all
echo "== validate $(date)"
python validate.py
exit 0
