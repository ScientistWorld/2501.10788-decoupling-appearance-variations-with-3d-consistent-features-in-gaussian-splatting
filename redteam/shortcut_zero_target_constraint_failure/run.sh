#!/bin/bash
set -euo pipefail
cd /home/user
python -u -m redteam.run_case shortcut_zero_target_constraint_failure "${@}"
