#!/usr/bin/env bash
set -euo pipefail
source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=matrix_env.sh
source "$source_dir/matrix_env.sh"
output="${OUTPUT_DIR:?Use launch_matrix.py}/results-$SLURM_JOB_ID"
mkdir "$output"
record_failure() {
    if (( $1 != 0 )) && [[ ! -f "$output/result.json" ]]; then
        printf '{"state":"failed","exit_code":%d}\n' "$1" > "$output/worker-failure.json"
    fi
}
trap 'record_failure "$?"' EXIT
/usr/bin/time -v python "$source_dir/pilot.py" "$output"
