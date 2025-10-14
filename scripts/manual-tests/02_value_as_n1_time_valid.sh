#!/usr/bin/env bash
set -euo pipefail
# Valid range check: n1 in [0, 5000]
python3 /workspace/main.py --value-kind n1_spike_time --value 115
python3 /workspace/main.py --value-kind n1_spike_time --value 150
