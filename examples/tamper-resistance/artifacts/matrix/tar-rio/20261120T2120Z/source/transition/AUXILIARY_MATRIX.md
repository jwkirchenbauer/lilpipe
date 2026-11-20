# Matrix auxiliary parity and precision jobs

These are fresh Matrix jobs, separate from the original pending Flux requests.
The parity worker uses an isolated `source/parity-matrix` snapshot of the
frozen matched-cache-v2 Python sources and the current Matrix runtime wrapper;
it does not import the legacy `source/src` trainer implementation. Its startup
AST preflight checks that `trainers.tr_tar` resolves through the matched-cache-v2
source link and that `torch.clear_autocast_cache()` is present in that imported
module. It reuses the original forward-matched defender, attack trace, prepared
dataset, and archived RIO K16 reference. The config preserves the full
1000-step learning-rate schedule and the existing callback stop at update 16.
Same-backend simulated-versus-genuine checks and zero loss-error requirements
remain strict.

Run the parity job with four H100s. After it completes with exact parity,
launch the precision audit on one H100 with an `afterok` dependency on that
Matrix job ID. The audit requires the exact native defender file used as the
pilot's `base_model`, its saved post-reload CUDA anchor, simulated states at
1/2/4/8/16, and the archived AMD checkpoints. It gates on exact CUDA
simulated-versus-genuine checks and all 16 zero loss errors. The archived RIO
v2 pair is labeled AMD-versus-AMD; the new forward-matched pair is labeled
CUDA-versus-AMD. Both reference comparisons are diagnostics and neither gates
same-backend parity.

From `examples/tamper-resistance`, review each dry run before submitting:

```bash
bash -ic 'set -e; mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-train-cu126-v1; python /p/vast1/kirchenb/llnl-tools/launch_matrix.py --run_name=tar-rio-aux-new-anchor-parity --partition=pbatch --account=verival --nodes=1 --tasks_per_node=1 --gpus_per_node=4 --exclusive=false --torchrun=false --launch_once_per_node=true --environment=/usr/workspace/wsb/kirchenb/lilpipe-tar-train-cu126-v1 --env_act_style=mm_activate --cuda_module= --minutes=45 --output_dir=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio/20261120T2120Z/launches --sub_output_dir_name=aux-new-anchor-parity --pass_run_name=false --quiet_launch=true --squonce=false --dryrun -- bash -i /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio/20261120T2120Z/source/transition/auxiliary_parity.sh'
```

After reviewing the rendered script, remove `--dryrun` for parity submission.
Record its Matrix job ID, then dry-run the audit using that ID as the
`afterok` dependency:

```bash
bash -ic 'set -e; mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1; python /p/vast1/kirchenb/llnl-tools/launch_matrix.py --run_name=tar-rio-aux-precision-audit --partition=pbatch --account=verival --nodes=1 --tasks_per_node=1 --gpus_per_node=1 --exclusive=false --torchrun=false --launch_once_per_node=true --environment=/usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1 --env_act_style=mm_activate --cuda_module= --minutes=45 --dependency=afterok:<PARITY_JOB_ID> --output_dir=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio/20261120T2120Z/launches --sub_output_dir_name=aux-precision-audit --pass_run_name=false --quiet_launch=true --squonce=false --dryrun -- bash -i /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio/20261120T2120Z/source/precision-audit/worker.sh'
```

Replace `<PARITY_JOB_ID>` with the completed parity job ID, review the rendered
script, then remove `--dryrun` for audit submission.
