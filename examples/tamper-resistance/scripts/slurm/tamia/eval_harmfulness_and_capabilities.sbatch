#!/bin/bash
#SBATCH --job-name=eval-harmfulness-capabilities
#SBATCH --output=artifacts/tamia/logs/eval_harmfulness_and_capabilities-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=03:00:00
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
export UV_LINK_MODE=copy

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
# Usage: [--checkpoints] MODEL OUTPUT [MODEL OUTPUT ...]
# --checkpoints evaluates steps 50..250, then reuses step 250 for the identical final model.
checkpoints=false
if [[ ${1:-} == --checkpoints ]]; then
    checkpoints=true
    shift
fi
if (( $# < 2 || $# % 2 != 0 )); then
    echo "Usage: $0 [--checkpoints] MODEL OUTPUT [MODEL OUTPUT ...]" >&2
    exit 2
fi
# Keep the authors' harmfulness dependencies separate from the project lm-eval env.
seam_env=${SEAM_ENV:-$SCRATCH/lilpipe/seam-official/venv}
harmfulness_script=../seam/scripts/eval_harmfulness.py
test -x "$seam_env/bin/python"
test -f "$harmfulness_script"

uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

evaluate_model() {
    local model=$1 output=$2
    mkdir -p "$output"
    echo "Evaluating $model at $(date -Is)"
    accelerate launch --multi_gpu --num_processes 4 --num_machines 1 \
      --num_cpu_threads_per_process 8 -m lm_eval run \
      --model hf \
      --model_args "pretrained=$model,dtype=bfloat16,trust_remote_code=True" \
      --tasks mmlu,truthfulqa_mc1,arc_easy,hellaswag \
      --num_fewshot 0 \
      --batch_size auto:4 \
      --max_batch_size 128 \
      --output_path "$output/capabilities"

    HF_DATASETS_CACHE="$XDG_CACHE_HOME/huggingface-seam-datasets-3.3.1" \
    TOKENIZERS_PARALLELISM=false \
    "$seam_env/bin/python" -m accelerate.commands.launch \
      --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision no \
      --num_cpu_threads_per_process 8 \
      "$harmfulness_script" "$model" \
      --batch-size 16 \
      --output "$output/harmfulness.json"
}

while (( $# )); do
    model=$1
    output=$2
    shift 2
    if [[ "$checkpoints" == false ]]; then
        evaluate_model "$model" "$output"
        continue
    fi
    # Confirm that the final model and checkpoint-250 are interchangeable before
    # using a single evaluation for both. Include tokenizer and generation config.
    "$seam_env/bin/python" - "$model" <<'PYTHON'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoint = root / "checkpoint-250"
index = json.loads((root / "model.safetensors.index.json").read_text())
assert index == json.loads((checkpoint / "model.safetensors.index.json").read_text())
files = set(index["weight_map"].values()) | {
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json",
}
for filename in files:
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest(root / filename) == digest(checkpoint / filename), filename
print(f"Final model matches checkpoint-250: {root}", flush=True)
PYTHON
    for step in 50 100 150 200 250; do
        evaluate_model "$model/checkpoint-$step" "$output/checkpoint-$step"
    done
    "$seam_env/bin/python" - "$model" "$output" <<'PYTHON'
import json
import shutil
import sys
from pathlib import Path

model, output = sys.argv[1], Path(sys.argv[2])
shutil.copytree(output / "checkpoint-250/capabilities", output / "capabilities")
report = json.loads((output / "checkpoint-250/harmfulness.json").read_text())
report["evaluated_model"] = report["model"]
report["model"] = model
report["reused_from"] = "checkpoint-250/harmfulness.json"
(output / "harmfulness.json").write_text(json.dumps(report, indent=2) + "\n")
(output / "final-evaluation.json").write_text(json.dumps({
    "model": model, "reused_from": "checkpoint-250",
    "reason": "Identical model, tokenizer, and generation configuration (SHA-256 verified)",
}, indent=2) + "\n")
PYTHON
done
