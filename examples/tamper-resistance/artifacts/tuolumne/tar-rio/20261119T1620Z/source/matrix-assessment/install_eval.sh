#!/usr/bin/env bash

# Create a Micromamba prefix and install the locked Matrix Python environment.
set -euo pipefail
mm_activate /usr/workspace/wsb/kirchenb/matrix-torch-cu126-demo

PROJECT_DIR=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/runtime/matrix/eval
WRKSPC="${WRKSPC:?WRKSPC must be set}"
ENV_PREFIX="$WRKSPC/${1:-lilpipe-tar-eval-cu126-v1}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$WRKSPC/micromamba}"
MICROMAMBA="${MICROMAMBA:-$MAMBA_ROOT_PREFIX/bin/micromamba}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
UV_VERSION="${UV_VERSION:-0.12.5}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$WRKSPC/.cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [[ ! -f "$PROJECT_DIR/uv.lock" ]]; then
    echo "Missing $PROJECT_DIR/uv.lock. Run the documented uv lock workflow for $PROJECT_DIR." >&2
    exit 1
fi
if [[ ! -x "$MICROMAMBA" ]]; then
    echo "Micromamba is unavailable at $MICROMAMBA. Run install_micromamba.sh first." >&2
    exit 1
fi
if [[ -e "$ENV_PREFIX" || -L "$ENV_PREFIX" ]]; then
    echo "Environment already exists at $ENV_PREFIX. Choose a new name for a clean build." >&2
    exit 1
fi

"$MICROMAMBA" create \
    --root-prefix "$MAMBA_ROOT_PREFIX" \
    --prefix "$ENV_PREFIX" \
    --override-channels --channel conda-forge --always-copy \
    "python=$PYTHON_VERSION" libstdcxx-ng "uv=$UV_VERSION" --yes

# Keep uv as the sole owner of Python distributions in the prefix.
"$MICROMAMBA" remove \
    --root-prefix "$MAMBA_ROOT_PREFIX" --prefix "$ENV_PREFIX" \
    --force pip setuptools wheel packaging --yes

mm_activate "$ENV_PREFIX"
export UV_SYSTEM_CERTS=true
UV_PROJECT_ENVIRONMENT="$ENV_PREFIX" \
    "$ENV_PREFIX/bin/uv" sync --project "$PROJECT_DIR" --locked

"$ENV_PREFIX/bin/python" -c 'import torch; print(f"torch={torch.__version__}, CUDA={torch.version.cuda}"); assert torch.version.cuda == "12.6"'
printf 'Activate with mm_activate %q\n' "$ENV_PREFIX"
