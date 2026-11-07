#!/bin/bash
#SBATCH --job-name=capability-models
#SBATCH --output=artifacts/tamia/logs/eval_capability_models-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=03:00:00
set -euo pipefail

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
# Run from examples/tamper-resistance. The existing capability launcher owns
# offline cache/environment setup; bash reuses this allocation for every model.
output_root=${1:?missing evaluation root}
shift
if (( $# == 0 )); then
    echo "Provide at least one model checkpoint path" >&2
    exit 2
fi
for model in "$@"; do
    output="$output_root/$(basename "$model")"
    CAPABILITY_TASKS=mmlu,truthfulqa_mc1,arc_easy,hellaswag NUM_FEWSHOT=0 MAX_BATCH_SIZE=128 \
      bash scripts/slurm/tamia/eval_capabilities.sbatch "$model" "$output/capabilities-4gpu"
    CAPABILITY_TASKS=gsm8k NUM_FEWSHOT=5 MAX_BATCH_SIZE=32 \
      bash scripts/slurm/tamia/eval_capabilities.sbatch "$model" "$output/gsm8k-5shot"
    if [[ ${INCLUDE_HARMFULNESS:-0} == 1 ]]; then
        HARMFULNESS_ONLY=1 bash scripts/slurm/tamia/eval_harmfulness_and_capabilities.sbatch "$model" "$output"
    fi
done
