#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

python -m method.run_experiment --only davigs --steps "${DAVIGS_STEPS:-2200}" --adapt-steps "${DAVIGS_ADAPT_STEPS:-400}" --out outputs
bash scripts/evaluate.sh
