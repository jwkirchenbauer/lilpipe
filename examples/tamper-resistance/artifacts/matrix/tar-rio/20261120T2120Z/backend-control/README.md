# Paired K4 training backend control

This repeats the original K4 defense for 125 outer updates and its standard
1000-update attack on CUDA. The completed ROCm K4 run remains unchanged.
Only output paths, run labels, and the attack's defended checkpoint path differ
in the scientific configurations. Fresh CUDA evaluations compare both runs.

The measurement study uses the frozen Matrix source hashes and optional
CPU storage for landscape anchors. Only the attack is in `evaluation_cases`:
its checkpoint zero evaluates the defended endpoint, so a separate defense
benchmark job is unnecessary. Landscape and early-update profile jobs retain
the original probes, physical directions, radii, and validation gates.

The first control submission failed before model loading because case output
directories were absent. Its log is preserved under
`../failures/backend-control-missing-dirs/516132.out`. Slurm canceled its
dependent requests and did not permit requeue. The ledger retains these
attempts as superseded. New requests use the same configurations after adding
the `status`, `logs`, `cache`, `models`, `geometry`, and `surfaces` directories,
plus the attack's `models/defense` link to the defense output. The submission
helper now checks case directories before requesting an allocation.

Raw model files, optimizer states, and basis arrays stay in the project
filesystem. The final report compares the paired scores in
`../backend-control-drift.csv` and states the training and evaluation backends.
