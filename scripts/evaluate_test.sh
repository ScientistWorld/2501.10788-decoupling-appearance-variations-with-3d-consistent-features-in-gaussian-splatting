#!/bin/bash
# Held-out test-slice evaluator. Runs with no arguments and scores every method
# directory under outputs/ (plus the roster's required methods) against the
# hidden test scenes; writes scoring/scores_test.json.
set -e
cd /home/user
mkdir -p scoring
python -m eval.test.evaluate --outputs outputs --scores scoring/scores_test.json
