# Commands

Run commands from `examples/tamper-resistance`.

## Hyperparameter optimization for q0.6b

```sh
# 1. Generate and run all defenses and their evaluations.
python scripts/hpo.py generate configs/stages-beavertails/0_alignment/qwen3-0.6b-beavertail--align-fft-seam/sweep.yml
lilpipe artifacts/generated/hpo/qwen3-0.6b-beavertail--align-fft-seam--hpo/experiment.yml

# 2. Generate and run all relearning candidates (only after the defense sweep is done).
python scripts/hpo.py generate configs/stages-beavertails/1_relearn/qwen3-0.6b-beavertail--align-fft-seam/sweep.yml
lilpipe artifacts/generated/hpo/qwen3-0.6b-beavertail--align-fft-seam--relearn--hpo/experiment.yml

# 3. Uncomment the desired configs in the plot script, then plot after evaluations finish.
.venv-eval/bin/python scripts/analysis/relearning-trajectory/plot_trajectory_beavertails.py
```
