#!/usr/bin/env bash
set -euo pipefail
# Invalid range check: n1 < 0 and > 5000 should fail with code 2
set +e
python3 /workspace/main.py --value-kind n1_spike_time --value -1; echo "exit=$?"
python3 /workspace/main.py --value-kind n1_spike_time --value 6000; echo "exit=$?"
set -e
