#!/usr/bin/env bash
set -euo pipefail
mm_activate /usr/workspace/wsb/kirchenb/matrix-torch-cu126-demo
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
assessment=artifacts/tuolumne/tar-rio/20261119T1620Z/source/matrix-assessment
mode=${1:?inventory or gpu}
if [[ "$mode" == inventory ]]; then
    python "$assessment/probe.py" --output "$assessment/environment-inventory.json"
elif [[ "$mode" == gpu ]]; then
    python "$assessment/probe.py" --gpu --output "$assessment/gpu-probe.json"
else
    exit 2
fi
