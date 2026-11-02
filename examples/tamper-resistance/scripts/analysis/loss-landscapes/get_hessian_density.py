"""Estimate the forget-loss Hessian spectrum with PyHessian and CPU Lanczos vectors."""

import argparse
import gc
import hashlib
import json
import os
import os.path as osp
import time

import numpy as np
import torch
from pyhessian import hessian
from transformers import AutoModelForCausalLM, AutoTokenizer

from get_ll import get_dataloader


class LanguageModelHessian(hessian):
    """Use PyHessian's density routine with batchwise HVPs on a sharded model."""

    def __init__(self, model, batches):
        self.model = model.eval()
        self.data = batches
        self.full_dataset = True
        self.device = "cpu"
        self.parameters = tuple(p for p in model.parameters() if p.requires_grad)
        # PyHessian only needs shapes here; its Lanczos vectors live on CPU.
        self.params = [torch.empty_like(p, device="meta") for p in self.parameters]
        self.hvp_count = 0

    def dataloader_hv_product(self, vector):
        start = time.monotonic()
        result = [torch.zeros_like(v, device="cpu") for v in vector]
        device_vector = [v.to(p.device) for v, p in zip(vector, self.parameters)]
        device = self.model.get_input_embeddings().weight.device
        mean_loss = 0.0
        for batch in self.data:
            inputs = {name: value.to(device) for name, value in batch.items()}
            loss = self.model(**inputs).loss / len(self.data)
            mean_loss += loss.detach().item()
            gradient = torch.autograd.grad(loss, self.parameters, create_graph=True)
            product = torch.autograd.grad(
                gradient, self.parameters, grad_outputs=device_vector
            )
            for total, value in zip(result, product):
                total.add_(value.detach().cpu())
            del loss, gradient, product, inputs
        del device_vector
        rayleigh = sum((hv * v).sum() for hv, v in zip(result, vector)).item()
        self.hvp_count += 1
        print(
            f"HVP {self.hvp_count}: loss={mean_loss:.8f}, "
            f"vHv={rayleigh:.6g}, elapsed={time.monotonic() - start:.1f}s",
            flush=True,
        )
        return rayleigh, result


def smooth_density(eigenvalues, weights, points=1024, bandwidth_fraction=0.03):
    """Smooth the weighted quadrature nodes, averaging over random probes."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(eigenvalues).all() or not np.isfinite(weights).all():
        raise ValueError("PyHessian returned nonfinite nodes or weights")
    if (weights < 0).any() or not np.allclose(weights.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Quadrature weights must be nonnegative and sum to one")
    low, high = eigenvalues.min(), eigenvalues.max()
    bandwidth = max(high - low, abs(low), abs(high), 1e-12) * bandwidth_fraction
    grid = np.linspace(low - 4 * bandwidth, high + 4 * bandwidth, points)
    distance = (grid[:, None, None] - eigenvalues[None, :, :]) / bandwidth
    kernels = np.exp(-0.5 * distance**2) / (bandwidth * np.sqrt(2 * np.pi))
    density = (kernels * weights[None, :, :]).sum(axis=2).mean(axis=1)
    return grid, density, bandwidth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="cais/wmdp-bio-forget-corpus")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--lanczos-steps", type=int, default=10)
    parser.add_argument("--probes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    if args.lanczos_steps < 2 or args.probes < 1:
        parser.error("Use at least two Lanczos steps and one probe")
    os.makedirs(args.outdir, exist_ok=False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    batches = list(
        get_dataloader(
            args.dataset, tokenizer, args.max_seq_len, args.batch_size, args.limit
        )
    )
    if not batches:
        raise ValueError("No documents available for Hessian estimation")
    digest = hashlib.sha256()
    for batch in batches:
        for name in sorted(batch):
            digest.update(name.encode())
            digest.update(str(tuple(batch[name].shape)).encode())
            digest.update(batch[name].numpy().tobytes())
    print("Loading checkpoint in FP32 with eager attention", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        device_map="balanced",
        attn_implementation="eager",
    )
    model.config.use_cache = False
    estimator = LanguageModelHessian(model, batches)
    result = {
        **vars(args),
        "dtype": "float32",
        "objective": "mean of gen_ll.py batch cross-entropies; padding labels masked",
        "documents": sum(len(batch["input_ids"]) for batch in batches),
        "batch_sha256": digest.hexdigest(),
        "parameter_count": sum(p.numel() for p in estimator.parameters),
        "pyhessian_revision": "de8cf239ace21c7754ee599255897ebf3eed19cb",
        "eigenvalues": [],
        "weights": [],
    }
    for probe in range(args.probes):
        torch.manual_seed(args.seed + probe)
        print(f"Probe {probe + 1}/{args.probes}", flush=True)
        nodes, weights = estimator.density(iter=args.lanczos_steps, n_v=1)
        result["eigenvalues"].append(np.asarray(nodes[0]).real.astype(float).tolist())
        result["weights"].append(np.asarray(weights[0]).real.astype(float).tolist())
        gc.collect()
        torch.cuda.empty_cache()
        with open(osp.join(args.outdir, "quadrature.json"), "w") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
    grid, density, bandwidth = smooth_density(result["eigenvalues"], result["weights"])
    np.savetxt(
        osp.join(args.outdir, "density.csv"),
        np.column_stack([grid, density]),
        delimiter=",",
        header="eigenvalue,density",
        comments="",
    )
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(grid, density)
    axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
    axis.set(
        xlabel="Hessian eigenvalue",
        ylabel="Estimated spectral density",
        title=f"TAR archived checkpoint · {args.lanczos_steps} steps × {args.probes} probes",
    )
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(osp.join(args.outdir, f"spectral-density.{extension}"), dpi=200)
    print(f"Saved density estimate; Gaussian bandwidth={bandwidth:.6g}", flush=True)
