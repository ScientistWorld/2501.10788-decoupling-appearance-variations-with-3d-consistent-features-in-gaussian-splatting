#!/bin/bash
# End-to-end reproduction: (re)build the benchmark if absent, run vanilla 3DGS,
# DAVIGS and the paper's two ablations on all scenes, freeze the comparison
# methods into evaluator-owned storage, and score both slices.
set -euo pipefail
cd /home/user
if [ -f eval/test/build_benchmark.py ]; then
  python -m eval.test.build_benchmark
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
