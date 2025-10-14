#!/usr/bin/env bash
set -euo pipefail
# Expect output spike and S0 LTP; S1 LTD because N1 arrives after post
python3 /workspace/main.py --n0-spike-time 100 --n1-spike-time 115 --s0-weight 1.10 --s1-weight 0.60 --delay 5 --no-plot
# Expect no output spike if s0 < threshold; acceptance still passes
python3 /workspace/main.py --n0-spike-time 100 --n1-spike-time 115 --s0-weight 0.90 --s1-weight 0.60 --delay 5 --no-plot
