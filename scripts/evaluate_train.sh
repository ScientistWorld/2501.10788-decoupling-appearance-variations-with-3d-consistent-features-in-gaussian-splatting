#!/bin/bash
# Visible train-slice evaluator for saved method outputs.

set -e

cd /home/user

mkdir -p scoring

python -m eval.train.evaluate --outputs outputs --scores scoring/scores_train.json
