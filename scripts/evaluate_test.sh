#!/bin/bash
# Held-out test-slice evaluator for saved method outputs.

set -e

cd /home/user

mkdir -p scoring

python -m eval.test.evaluate --outputs outputs --scores scoring/scores_test.json --scores-train scoring/scores_train.json
