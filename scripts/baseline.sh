#!/bin/bash
# Vanilla 3D Gaussian Splatting baseline on every benchmark scene, then evaluation.
# Usage: bash scripts/baseline.sh
set -e
cd /home/user
python -m method.run_experiment --plan baseline --strict
bash scripts/evaluate.sh
