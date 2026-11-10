#!/usr/bin/env bash
set -euo pipefail
study=${1:?study directory required}
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3 HIP_VISIBLE_DEVICES=0,1,2,3
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1 MPLCONFIGDIR="$study/cache/matplotlib"
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
trap 'result=$?; if (( result != 0 )); then printf "{\"stage\":\"early-evaluations\",\"state\":\"failed\",\"exit_code\":%d}\n" "$result" > "$study/status/early-evaluations.json"; fi' EXIT
for step in 1 2 4 8; do
    accelerate launch --multi_gpu --num_processes 4 --num_machines 1 \
        --num_cpu_threads_per_process 8 -m lm_eval run \
        --model hf --model_args "pretrained=$study/models/attack/checkpoint-$step,dtype=bfloat16,trust_remote_code=True" \
        --include_path "/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-k16-repro/20261109T1830Z/source/lm_eval_tasks" \
        --tasks wmdp_bio_robust,mmlu_no_bio --num_fewshot 0 \
        --batch_size 32 --output_path "$study/evals/attack/checkpoint-$step" \
        2>&1 | tee "$study/logs/early-evaluation-$step.log"
done
python - "$study" <<'PY'
import csv
import json
import sys
from pathlib import Path
study = Path(sys.argv[1])
rows = []
for step in (1, 2, 4, 8):
    matches = list((study / f"evals/attack/checkpoint-{step}").rglob("results*.json"))
    assert len(matches) == 1, matches
    document = json.loads(matches[0].read_text())
    assert document["config"]["limit"] is None
    for task in ("wmdp_bio_robust", "mmlu_no_bio"):
        score = 100 * document["results"][task]["acc,none"]
        assert 0 <= score <= 100
        rows.append({"learning_rate": "1e-6", "attack_step": step, "task": task, "jwk_percent": score})
with (study / "early-comparison.csv").open("w") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
PY
printf '{"stage":"early-evaluations","state":"complete"}\n' > "$study/status/early-evaluations.json"
