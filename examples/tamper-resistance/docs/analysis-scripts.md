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
