#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1
exec python "$(dirname "${BASH_SOURCE[0]}")/peer_allocation_probe.py"
