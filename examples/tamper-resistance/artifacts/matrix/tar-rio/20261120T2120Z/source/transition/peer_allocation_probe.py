"""Record native Matrix GPU assignment and device cgroup evidence."""

import json
import os
from pathlib import Path
import subprocess
import sys

import torch

result = {
    "python": sys.executable,
    "torch": str(torch.__version__),
    "environment": {
        key: value
        for key, value in os.environ.items()
        if key.startswith("SLURM_")
        or key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
    },
    "cgroup": Path("/proc/self/cgroup").read_text(),
}
result["device_cgroups"] = {}
for line in result["cgroup"].splitlines():
    _, controllers, location = line.split(":", 2)
    if "devices" in controllers.split(","):
        path = Path("/sys/fs/cgroup/devices") / location.lstrip("/") / "devices.list"
        result["device_cgroups"][str(path)] = (
            path.read_text() if path.exists() else "missing"
        )
for name, command in {
    "gpus": [
        "nvidia-smi",
        "--query-gpu=index,uuid,minor_number",
        "--format=csv,noheader",
    ],
    "topology": ["nvidia-smi", "topo", "-m"],
}.items():
    proc = subprocess.run(command, capture_output=True, text=True)
    result[name] = dict(
        returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr
    )
count = torch.cuda.device_count()
result["visible_cuda_count"] = count
result["visible_cuda_devices"] = [
    str(torch.cuda.get_device_properties(i)) for i in range(count)
]
if count == 2:
    result["tensor_checks"] = [
        torch.ones(1, device=f"cuda:{i}").item() for i in range(2)
    ]
    result["state"] = "complete"
else:
    result["state"] = "assignment_mismatch"
output = Path(os.environ["OUTPUT_DIR"]) / (
    "allocation-" + os.environ["SLURM_JOB_ID"] + ".json"
)
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
assert count == 2, "Expected exactly two visible devices with explicit gpus_per_task=2"
