#!/usr/bin/env bash
set -euo pipefail
# Valid range check: s1 in [w_min, w_max]
python3 /workspace/main.py --value-kind s1_weight --value 0.60
python3 /workspace/main.py --value-kind s1_weight --value 1.50
