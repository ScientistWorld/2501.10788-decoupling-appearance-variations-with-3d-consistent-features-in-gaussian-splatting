#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

export DAVIGS_STEPS="${DAVIGS_STEPS:-700}"
export DAVIGS_ADAPT_STEPS="${DAVIGS_ADAPT_STEPS:-80}"

bash scripts/reproduce.sh
python validate.py --compare
