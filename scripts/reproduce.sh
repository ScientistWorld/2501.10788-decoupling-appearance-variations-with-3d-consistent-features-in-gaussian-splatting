#!/bin/bash
# End-to-end reproduction: (re)build the benchmark if absent, run vanilla 3DGS,
# DAVIGS and the paper's two ablations on all scenes, freeze the comparison
# methods and the gate reference into evaluator-owned storage, and score both
# slices. DAVIGS uses the module hyperparameters in method/selected_config.json;
# SELECT=1 re-runs that development-scene selection first.
set -euo pipefail
cd /home/user
if [ -f eval/test/build_benchmark.py ]; then
  python -m eval.test.build_benchmark
fi
if [ "${SELECT:-0}" = "1" ]; then
  python -m method.run_experiment --plan select --strict
  python -m method.select_config
fi
python -m method.run_experiment --plan reference --strict
if [ -f eval/test/freeze.py ]; then
  python -m eval.test.freeze
fi
python -m eval.train.freeze
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
if [ -d eval/test ]; then
  bash scripts/evaluate_test.sh
fi
