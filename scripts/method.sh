#!/bin/bash
# The paper's method (DAVIGS on 3DGS) on every benchmark scene, then evaluation.
# Usage: bash scripts/method.sh
set -e
cd /home/user
python -m method.run_experiment --plan proposed --strict
bash scripts/evaluate.sh
