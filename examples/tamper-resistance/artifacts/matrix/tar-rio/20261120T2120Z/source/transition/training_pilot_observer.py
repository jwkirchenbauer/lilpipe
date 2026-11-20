"""Observe device memory after the existing schedule-preserving stop callback."""

import json
from pathlib import Path

import torch
from transformers import TrainerCallback
from tar_dynamics_training import RecordedTARMETrainer


class PeakMemory(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        rows = []
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)
            rows.append(
                {
                    "device": index,
                    "name": torch.cuda.get_device_name(index),
                    "allocated_peak_bytes": torch.cuda.max_memory_allocated(index),
                    "reserved_peak_bytes": torch.cuda.max_memory_reserved(index),
                    "total_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
            )
        target = Path(args.output_dir) / "matrix-runtime.json"
        target.write_text(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "updates": state.global_step,
                    "gpus": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print("Matrix pilot device memory:", json.dumps(rows), flush=True)


class MatrixPilotTrainer(RecordedTARMETrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(PeakMemory())
