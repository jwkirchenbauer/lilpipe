#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-train-rocm72-v4
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
run=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio-cuda/20261121T0610Z
while true; do
    python "$run/ops/watch.py"
    if grep -q '"all_complete": true' "$run/status/poll.json"; then
        break
    fi
    sleep 900
done
