"""Shared trainer utilities."""

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from safetensors import safe_open

try:
    from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
except ImportError:  # pragma: no cover - older torch without FSDP2
    DTensor = ()
    Replicate = ()
    distribute_tensor = None


def load_reference_parameters(parameters, model_name_or_path):
    """Load reference weights and shard them like the corresponding parameters."""
    model_path = Path(model_name_or_path)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(repo_id=model_name_or_path, local_files_only=True)
        )

    index_path = model_path / "model.safetensors.index.json"
    with index_path.open() as handle:
        weight_map = json.load(handle)["weight_map"]

    checkpoint_names = {
        name: name.replace("._checkpoint_wrapped_module", "").replace(
            "._fsdp_wrapped_module", ""
        )
        for name in parameters
    }
    missing = {
        checkpoint_name
        for checkpoint_name in checkpoint_names.values()
        if checkpoint_name not in weight_map
    }
    if missing:
        names = ", ".join(sorted(missing)[:5])
        raise RuntimeError(f"anchor checkpoint is missing parameters: {names}")

    names_by_shard = defaultdict(list)
    for name, checkpoint_name in checkpoint_names.items():
        names_by_shard[weight_map[checkpoint_name]].append(name)

    reference_parameters = {}
    for shard_name, names in names_by_shard.items():
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as shard:
            for name in names:
                parameter = parameters[name]
                reference_parameter = shard.get_tensor(checkpoint_names[name]).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                if isinstance(parameter, DTensor):
                    reference_parameter = distribute_tensor(
                        reference_parameter,
                        parameter.device_mesh,
                        parameter.placements,
                        src_data_rank=None,
                    )
                reference_parameters[name] = reference_parameter.detach()
    return reference_parameters


def global_l2_norm(tensors):
    """Return the full-model norm for ordinary or FSDP2-sharded tensors."""
    tensors = tuple(tensors)
    if not tensors:
        raise ValueError("global_l2_norm requires at least one tensor")

    uses_dtensor = any(isinstance(tensor, DTensor) for tensor in tensors)
    world_size = dist.get_world_size() if uses_dtensor else 1
    norm_device = tensors[0].device
    local_squared_norm = torch.zeros((), device=norm_device, dtype=torch.float32)
    for tensor in tensors:
        if isinstance(tensor, DTensor):
            local_tensor = tensor.to_local()
            is_replicated = all(
                isinstance(placement, Replicate) for placement in tensor.placements
            )
            divisor = world_size if is_replicated else 1
        else:
            local_tensor = tensor
            divisor = 1
        squared_norm = local_tensor.detach().float().square().sum() / divisor
        local_squared_norm += squared_norm.to(norm_device)

    if uses_dtensor:
        if not dist.is_initialized():
            raise RuntimeError("DTensor gradients require distributed initialization")
        dist.all_reduce(local_squared_norm, op=dist.ReduceOp.SUM)
    return local_squared_norm.sqrt()


def gradient_dict_l2_norm(gradients):
    """Return the norm of the non-None gradients in a parameter mapping."""
    active = [gradient for gradient in gradients.values() if gradient is not None]
    if not active:
        raise RuntimeError("loss term produced no parameter gradients")
    return global_l2_norm(active)


class MetadataCollator:
    """Collate token fields and encode metadata as integer source IDs."""

    def __init__(self, collator, metadata_to_idx):
        self.collator = collator
        self.metadata_to_idx = metadata_to_idx

    def __call__(self, features, **kwargs):
        metadata_ids = [self.metadata_to_idx[row["metadata"]] for row in features]
        batch = self.collator(
            [
                {key: value for key, value in row.items() if key != "metadata"}
                for row in features
            ],
            **kwargs,
        )
        batch["metadata_ids"] = torch.tensor(metadata_ids, dtype=torch.long)
        return batch


class BalancedSourceSampler(torch.utils.data.Sampler[int]):
    """Shuffle sources and contribute equally to each DataLoader batch."""

    def __init__(self, dataset, batch_size, metadata_to_idx, seed=42):
        groups = defaultdict(list)
        for index, metadata in enumerate(dataset["metadata"]):
            groups[metadata_to_idx(metadata)].append(index)
        if not groups or batch_size < 1 or batch_size % len(groups):
            raise ValueError("batch size must be divisible by the number of sources")
        self.groups = list(groups.values())
        self.batch_size = batch_size
        self.per_source_batch_size = batch_size // len(groups)
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        groups = [group.copy() for group in self.groups]
        for group in groups:
            rng.shuffle(group)
        usable = len(self) // self.batch_size * self.per_source_batch_size
        for start in range(0, usable, self.per_source_batch_size):
            batch = [
                index
                for group in groups
                for index in group[start : start + self.per_source_batch_size]
            ]
            rng.shuffle(batch)
            yield from batch

    def __len__(self):
        return (
            min(map(len, self.groups)) // self.per_source_batch_size * self.batch_size
        )


class BalancedSourceSamplerv2(torch.utils.data.Sampler[int]):
    """Shuffle sources and take the configured number of rows from each."""

    def __init__(self, dataset, metadata_to_idx, metadata_to_batch_size, seed=42):
        metadata = dataset["metadata"]
        self.meta_ids = [metadata_to_idx[value] for value in metadata]
        groups = defaultdict(list)
        for index, source in enumerate(metadata):
            groups[source].append(index)
        if not groups or groups.keys() != metadata_to_batch_size.keys():
            raise ValueError("batch sizes must cover exactly the dataset sources")
        if any(
            type(size) is not int or size < 1
            for size in metadata_to_batch_size.values()
        ):
            raise ValueError("per-source batch sizes must be positive integers")
        self.groups = [
            (groups[source], size) for source, size in metadata_to_batch_size.items()
        ]
        self.batch_size = sum(metadata_to_batch_size.values())
        self.num_batches = min(len(indices) // size for indices, size in self.groups)
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        groups = [(indices.copy(), size) for indices, size in self.groups]
        for indices, _ in groups:
            rng.shuffle(indices)
        for step in range(self.num_batches):
            batch = [
                index
                for indices, size in groups
                for index in indices[step * size : (step + 1) * size]
            ]
            rng.shuffle(batch)
            yield from batch

    def __len__(self):
        return self.num_batches * self.batch_size


class MixedSourceSampler(torch.utils.data.Sampler[int]):
    """Shuffle, truncate, and mix both sources without replacement."""

    def __init__(self, dataset, batch_size: int, seed: int):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("mixed source sampling requires an even batch size")

        sources = dataset["is_forget"]
        self.forget = [index for index, is_forget in enumerate(sources) if is_forget]
        self.retain = [
            index for index, is_forget in enumerate(sources) if not is_forget
        ]
        if not self.forget or not self.retain:
            raise ValueError("mixed source sampling requires forget and retain rows")

        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    @property
    def source_size(self):
        half_batch = self.batch_size // 2
        return min(len(self.forget), len(self.retain)) // half_batch * half_batch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        forget = self.forget.copy()
        retain = self.retain.copy()
        rng.shuffle(forget)
        rng.shuffle(retain)

        half_batch = self.batch_size // 2
        for start in range(0, self.source_size, half_batch):
            batch = (
                forget[start : start + half_batch] + retain[start : start + half_batch]
            )
            rng.shuffle(batch)
            yield from batch

    def __len__(self):
        return 2 * self.source_size


class AuthorsMixedSourceSampler(torch.utils.data.Sampler[int]):
    """Shuffle forget rows and sample matching retain rows with replacement."""

    def __init__(self, dataset, batch_size: int, seed: int):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("mixed source sampling requires an even batch size")
        sources = dataset["is_forget"]
        self.forget = [index for index, value in enumerate(sources) if value]
        self.retain = [index for index, value in enumerate(sources) if not value]
        if not self.forget or not self.retain:
            raise ValueError("mixed source sampling requires forget and retain rows")
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        forget = self.forget.copy()
        rng.shuffle(forget)
        half_batch = self.batch_size // 2
        usable = len(forget) // half_batch * half_batch
        for start in range(0, usable, half_batch):
            batch = forget[start : start + half_batch]
            batch += [rng.choice(self.retain) for _ in range(half_batch)]
            rng.shuffle(batch)
            yield from batch

    def __len__(self):
        half_batch = self.batch_size // 2
        return 2 * (len(self.forget) // half_batch * half_batch)


class OptmlMixedSourceSampler(torch.utils.data.Sampler[int]):
    """Match OPTML's shuffled forget rows and dynamic retain draws."""

    def __init__(self, dataset, batch_size: int):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("mixed source sampling requires an even batch size")
        sources = dataset["is_forget"]
        self.forget = [index for index, value in enumerate(sources) if value]
        self.retain = [index for index, value in enumerate(sources) if not value]
        if not self.forget or not self.retain:
            raise ValueError("mixed source sampling requires forget and retain rows")
        self.batch_size = batch_size

    def __iter__(self):
        # OPTML wraps the forget dataset in PyTorch's default RandomSampler.
        # Its paired dataset then draws one retain row from Python's global RNG
        # each time a forget row is fetched.
        order = torch.utils.data.RandomSampler(self.forget)
        half_batch = self.batch_size // 2
        batch = []
        for position in order:
            batch.append(self.forget[position])
            if len(batch) == half_batch:
                yield from batch
                yield from (random.choice(self.retain) for _ in range(half_batch))
                batch.clear()

    def __len__(self):
        half_batch = self.batch_size // 2
        return 2 * (len(self.forget) // half_batch * half_batch)
