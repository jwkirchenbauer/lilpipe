# Analysis scripts

Scripts are grouped by experiment under `scripts/analysis/<experiment-name>/`,
with underscore filenames beginning with `get_` or `plot_`. Separate 2D and 3D
plotters remain available. Arguments, defaults, and output paths are unchanged.
Slurm wrapper filenames remain unchanged.

Historical retirement inventories use the former names listed below.
All paths in this table are relative to `scripts/analysis/`.

| Former filename | Current path |
| --- | --- |
| `gen_ll.py` | `loss-landscapes/get_ll.py` |
| `plot_ll_2d.py` | `loss-landscapes/plot_ll_2d.py` |
| `plot_ll_3d.py` | `loss-landscapes/plot_ll_3d.py` |
| `get_param_norm.py` | `parameter-norms/get_param_norm.py` |
| `plot_layerwise_norms.py` | `parameter-norms/plot_layerwise_norms.py` |
| `gen_seam_toy.py` | `toy-experiments/get_seam_toy.py` |
| `plot_seam_toy.py` | `toy-experiments/plot_seam_toy.py` |
| `gen_locking_toy.py` | `toy-experiments/get_locking_toy.py` |
| `plot_locking_toy.py` | `toy-experiments/plot_locking_toy.py` |
| `gen_mnist_basin.py` | `toy-experiments/get_mnist_basin.py` |
| `plot_mnist_basin.py` | `toy-experiments/plot_mnist_basin.py` |
| `train_mnist.py` | `toy-experiments/get_mnist.py` |
| `plot_trajectory_wmdp.py` | `relearning-trajectory/plot_trajectory_wmdp.py` |
| `plot_trajectory_beavertails.py` | `relearning-trajectory/plot_trajectory_beavertails.py` |

Model-distance, smoothness, and perturbed-unlearning analysis implementations
were removed along with their dedicated model-distance and smoothness tests.
Historical configurations and results remain preserved.

## CIFAR-10 defense and attack toy

Four scripts under `scripts/analysis/toy-experiments/` run and plot the comparison:

- `run_cifar_defence_and_attack.py`: random initialization, BaSFT, TAR, or SAM,
  followed by an ordinary cross-entropy attack. Saves `train.pt`, `relearn.pt`,
  and `results.json`. Checkpoints contain model weights, not optimizer state.
- `get_cifar_landscapes.py`: evaluates shared orthonormal random weight slices
  around the initial and defended models, writing JSON grids.
- `plot_cifar_relearn_trajectory.py`: plots attack LR as rows and metrics as columns.
- `plot_cifar_landscapes.py`: plots radius as rows and method as columns.

Run from this example directory using an environment with torch, torchvision,
numpy, pandas, seaborn, and matplotlib. Set `run_root` to a scratch directory.
The plotters produce PDF only.

```bash
run_root="$SCRATCH/lilpipe/cifar10-tar-sweep"
python scripts/analysis/toy-experiments/run_cifar_defence_and_attack.py \
  --method random --output "$run_root/random"
python scripts/analysis/toy-experiments/run_cifar_defence_and_attack.py \
  --method tar --attack-steps 16 --output "$run_root/tar"
python scripts/analysis/toy-experiments/run_cifar_defence_and_attack.py \
  --method sam-nll --rho 0.01 --output "$run_root/sam-nll-rho0.01"
python scripts/analysis/toy-experiments/run_cifar_defence_and_attack.py \
  --method sam-ent --rho 0.01 --output "$run_root/sam-ent-rho0.01"
```

Use TAR folders `tar-k1`, `tar-k8`, and `tar-k32` with the corresponding
`--attack-steps` values. BaSFT uses `--method basft --beta VALUE`.
Every run defaults to seed 0, 600 defense steps, 600 attack steps, batch size 128,
Adam LR 0.001, and a fixed 1,024-image test subset. Attack loss and gradient norm
are measured on the current batch every 10 steps. SAM (nll) minimizes negative
cross-entropy; SAM (ent) and TAR minimize negative entropy. TAR uses a fresh Adam
attacker and a first-order gradient through its endpoint. SAM uses global-L2
perturbations and a detached perturbation for its outer gradient.

To attack an existing defense with a larger LR, skip defense with `--checkpoint`:

```bash
python scripts/analysis/toy-experiments/run_cifar_defence_and_attack.py \
  --method tar --checkpoint "$run_root/tar/train.pt" \
  --relearn-learning-rate 0.01 --output "$run_root/attack-lr0.01/tar"
```

Repeat for each method, using the same folder names under `attack-lr0.01`.
For the random baseline omit `--checkpoint` and use `--method random`.

```bash
python scripts/analysis/toy-experiments/plot_cifar_relearn_trajectory.py --root "$run_root"
python scripts/analysis/toy-experiments/get_cifar_landscapes.py \
  --root "$run_root" --radii 1000 5000 --points 9
python scripts/analysis/toy-experiments/plot_cifar_landscapes.py \
  --input artifacts/figures/resnet18-cifar10-tar-loss-landscapes.json
```

Landscapes use a fixed 1,024-image training subset and seed-0 random initialization.
`--resume` reuses saved grids with matching settings and checkpoint paths; use
immutable checkpoint paths or omit `--resume` after replacing checkpoint contents.
