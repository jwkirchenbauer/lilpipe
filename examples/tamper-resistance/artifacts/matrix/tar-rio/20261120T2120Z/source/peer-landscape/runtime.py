"""Matrix runtime validation and provenance for the two-GPU landscape."""

import hashlib
import os
import socket
import sys
from pathlib import Path


SOURCE = Path(__file__).resolve().parent
BASELINE = SOURCE.parent / "misc/tar_dynamics.py"
PEER = SOURCE / "misc/tar_dynamics.py"
EVAL_PREFIX = Path("/usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = [
        BASELINE,
        PEER,
        BASELINE.with_name("trace_runner.py"),
        *(SOURCE / name for name in ("analysis.sh", "matrix_env.sh", "runtime.py")),
    ]
    return {str(path): sha256(path) for path in paths}


def matrix_runtime():
    import torch

    assert os.environ["SLURM_CLUSTER_NAME"].lower() == "matrix"
    assignment = os.environ["TAR_GPU_ASSIGNMENT_MODE"]
    step_gpus = os.environ.get("SLURM_STEP_GPUS")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    inherited = os.environ.get("TAR_SLURM_INHERITED_CUDA_VISIBLE_DEVICES")
    assert visible == inherited
    if step_gpus:
        assert assignment == "slurm_step_gpus"
    else:
        assert assignment == "slurm_cgroup"
    assert Path(sys.prefix).samefile(EVAL_PREFIX)
    assert torch.__version__ == "2.12.1+cu126"
    assert torch.version.cuda == "12.6" and torch.version.hip is None
    assert torch.cuda.device_count() == 2, "Expected exactly two visible Matrix GPUs"
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    devices = [torch.cuda.get_device_properties(i) for i in range(2)]
    assert all("H100" in device.name for device in devices)
    assert devices[0].uuid != devices[1].uuid
    environment = (
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_ID",
        "SLURM_STEP_ID",
        "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK",
        "TAR_GPU_ASSIGNMENT_MODE",
        "CONDA_PREFIX",
        "TAR_RUNTIME_BACKEND",
        "TAR_LANDSCAPE_PEER",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TORCH_BLAS_PREFER_CUBLASLT",
        "TORCHAO_FORCE_SKIP_LOADING_SO_FILES",
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "PYTHONUNBUFFERED",
        "MPLBACKEND",
        "NVIDIA_TF32_OVERRIDE",
    )
    return {
        "host": socket.gethostname(),
        "python": sys.version,
        "python_executable": sys.executable,
        "environment": {
            **{name: os.environ[name] for name in environment},
            "SLURM_STEP_GPUS": step_gpus,
            "CUDA_VISIBLE_DEVICES": visible,
            "TAR_SLURM_INHERITED_CUDA_VISIBLE_DEVICES": inherited,
        },
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "devices": [str(device) for device in devices],
        "peer_access_0_to_1": torch.cuda.can_device_access_peer(0, 1),
        "peer_access_1_to_0": torch.cuda.can_device_access_peer(1, 0),
    }
