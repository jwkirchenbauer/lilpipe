"""BF16 relearning with harmful/benign gradient-cosine logging."""

from __future__ import annotations

import torch

from data.seam import BENIGN, HARMFUL
from trainers.rl_bf16 import BF16Trainer
from trainers.tr_seam import gradient_dict_dot, named_gradients


class HarmfulBenignSampler(torch.utils.data.Sampler[int]):
    """Arrange the merged dataset into half-harmful, half-benign batches."""

    def __init__(self, dataset, batch_size, seed):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("cosine tracking requires an even batch size")

        roles = dataset["seam_role"]
        self.harmful_indices = [
            index for index, role in enumerate(roles) if role == HARMFUL
        ]
        self.benign_indices = [
            index for index, role in enumerate(roles) if role == BENIGN
        ]
        if not self.harmful_indices or not self.benign_indices:
            raise ValueError("cosine tracking requires harmful and benign rows")

        self.rows_per_role = batch_size // 2
        available_rows = min(len(self.harmful_indices), len(self.benign_indices))
        self.rows_per_epoch = available_rows // self.rows_per_role * self.rows_per_role
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices_harmful = torch.randperm(
            len(self.harmful_indices), generator=generator
        ).tolist()
        indices_benign = torch.randperm(
            len(self.benign_indices), generator=generator
        ).tolist()
        self.epoch += 1

        for start in range(0, self.rows_per_epoch, self.rows_per_role):
            stop = start + self.rows_per_role
            yield from (
                self.harmful_indices[index] for index in indices_harmful[start:stop]
            )
            yield from (
                self.benign_indices[index] for index in indices_benign[start:stop]
            )

    def __len__(self):
        return 2 * self.rows_per_epoch


class CosineTrackedBF16Trainer(BF16Trainer):
    """Optimize harmful CE while measuring its cosine with benign CE gradients."""

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        seed = (
            self.args.data_seed if self.args.data_seed is not None else self.args.seed
        )
        return HarmfulBenignSampler(dataset, self._train_batch_size, seed)

    def training_step(self, model, inputs, num_items_in_batch=None):
        del num_items_in_batch
        model.train()
        inputs = self._prepare_inputs(inputs)
        roles = inputs.pop("seam_role")
        inputs.pop("source_index")
        fields = ("input_ids", "attention_mask", "labels")
        inputs_harmful = {field: inputs[field][roles == HARMFUL] for field in fields}
        inputs_benign = {field: inputs[field][roles == BENIGN] for field in fields}

        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        divisor = self.current_gradient_accumulation_steps
        loss_harmful = model(**inputs_harmful).loss
        gradients_harmful = named_gradients(loss_harmful, parameters, divisor)

        loss_benign = model(**inputs_benign).loss
        gradients_benign = named_gradients(loss_benign, parameters, divisor)

        norm_harmful = gradient_dict_dot(gradients_harmful, gradients_harmful).sqrt()
        norm_benign = gradient_dict_dot(gradients_benign, gradients_benign).sqrt()
        cosine = gradient_dict_dot(gradients_harmful, gradients_benign) / (
            norm_harmful * norm_benign + 1e-8
        )

        # Only the harmful gradients are accumulated for the optimizer step.
        with torch.no_grad():
            for name, parameter in parameters.items():
                if parameter.grad is None:
                    parameter.grad = gradients_harmful[name]
                else:
                    parameter.grad.add_(gradients_harmful[name])

        self.log(
            {
                "loss_harmful": loss_harmful.detach().item(),
                "loss_benign": loss_benign.detach().item(),
                "loss_sd": cosine.detach().item(),
                "norm_harmful": norm_harmful.detach().item(),
                "norm_benign": norm_benign.detach().item(),
            }
        )
        return (loss_harmful.detach() / divisor).to(self.args.device)
