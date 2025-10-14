#!/usr/bin/env bash
set -euo pipefail
# Invalid range check: s1 < w_min or > w_max should fail with code 2
set +e
python3 /workspace/main.py --value-kind s1_weight --value -0.01; echo "exit=$?"
python3 /workspace/main.py --value-kind s1_weight --value 2.01; echo "exit=$?"
set -e
