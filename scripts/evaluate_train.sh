#!/bin/bash
# Visible train-slice evaluator (development scenes). Scores every method
# directory under outputs/ and writes scoring/scores_train.json.
set -e
cd /home/user
mkdir -p scoring
python -m eval.train.evaluate --outputs outputs --scores scoring/scores_train.json
