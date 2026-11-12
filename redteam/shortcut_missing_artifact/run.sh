#!/bin/bash
set -euo pipefail
cd /home/user
python -u -m redteam.run_case shortcut_missing_artifact "${@}"
