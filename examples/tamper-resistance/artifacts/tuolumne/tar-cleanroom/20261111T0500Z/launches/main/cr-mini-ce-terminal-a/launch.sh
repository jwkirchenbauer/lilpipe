#!/bin/bash
#flux: --output=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-cleanroom/20261111T0500Z/launches/main/cr-mini-ce-terminal-a/logs/flux-{{id}}.out
#flux: --error=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-cleanroom/20261111T0500Z/launches/main/cr-mini-ce-terminal-a/logs/flux-{{id}}.out

# example of node allocation
# flux alloc -q pdebug --bank=verival --job-name=cr-mini-ce-terminal-a -t55 -N1 -n1 -g1 -c96 -ofastload -o mpibind=off --exclusive --unbuffered --label-io
# then
# bash /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-cleanroom/20261111T0500Z/launches/main/cr-mini-ce-terminal-a/launch.sh
# Example of batch allocation
# flux batch -q pdebug --bank=verival --job-name=cr-mini-ce-terminal-a -t55 -N1 -n1 -g1 -c96 -ofastload -o mpibind=off --label-io --exclusive --unbuffered /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-cleanroom/20261111T0500Z/launches/main/cr-mini-ce-terminal-a/launch.sh

export FLUX_JOB_NNODES=1
export FLUX_JOB_SIZE=1

export MASTER_ADDR=$(flux getattr hostlist | /bin/hostlist -n 1)
PORT_PYTHON=$(command -v python || command -v python3)
export MASTER_PORT=$("$PORT_PYTHON" -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])")

# Dirs
export OUTPUT_DIR=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-cleanroom/20261111T0500Z/launches/main/cr-mini-ce-terminal-a
export LOGDIR=$OUTPUT_DIR/logs
mkdir -p "$LOGDIR"

echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "Logging to $LOGDIR"

export RUN_NAME="cr-mini-ce-terminal-a"

echo $(date -u) "Calling flux run..."

flux run -N1 -n1 -g1 -c96 -ofastload -o mpibind=off --exclusive --unbuffered --label-io \
    "$OUTPUT_DIR/subrunner.sh"

echo $(date -u) "Job execution finished."
