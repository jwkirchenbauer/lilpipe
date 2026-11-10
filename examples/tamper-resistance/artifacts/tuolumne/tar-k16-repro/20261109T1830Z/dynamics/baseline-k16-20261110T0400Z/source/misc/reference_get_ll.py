import argparse
import itertools
import os
import os.path as osp
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


def dict_unary_op(d, fn):
    return {k: fn(v) for k, v in d.items()}


def dict_bin_op(a, b, fn):
    return {k: fn(a[k], b[k]) for k in a}


def get_directions(
    model, seed, dataloader=None, use_grad_directions=False, grad_lr=5e-6
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    directions = []

    if use_grad_directions:  # take gradients @ steps 1 & 2 as directions
        params = dict(model.named_parameters())
        theta_0 = dict_unary_op(params, lambda p: p.detach().to("cpu", copy=True))
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in next(iter(dataloader)).items()}
        optimizer = torch.optim.SGD(model.parameters(), lr=grad_lr)
        try:
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                model(**batch).loss.backward()
                directions.append(
                    dict_unary_op(
                        params,
                        lambda p: (
                            p.grad.detach().to("cpu", copy=True)
                            if p.grad is not None
                            else torch.zeros_like(p, device="cpu")
                        ),
                    )
                )
                if step == 0:
                    optimizer.step()
        finally:
            with torch.no_grad():
                for n, p in params.items():
                    p.copy_(theta_0[n])
            optimizer.zero_grad(set_to_none=True)
        del theta_0
        directions = [dict_unary_op(d, lambda t: t.to(device)) for d in directions]
    else:  # random directions
        for _ in range(2):
            d = dict()
            for n, p in model.named_parameters():
                d[n] = torch.randn_like(p)
            directions.append(d)

    # normalize so each direction has unit global L2 norm
    normalized = []
    for d in directions:
        norm = sum(t.pow(2).sum() for t in d.values()).sqrt()
        normalized.append(dict_unary_op(d, lambda t: t / norm))
    return normalized


def get_dataloader(dataset_name, tokenizer, max_seq_len, batch_size, limit):
    ds = load_dataset(dataset_name, split="train")
    ds = ds.filter(lambda row: len(row["text"]) > 50)
    ds = ds.select(range(min(limit, len(ds))))

    ds = ds.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
        ),
        batched=True,
        remove_columns=ds.column_names,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    return dl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trajectory", nargs="+", default=[])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.05])
    parser.add_argument("--n-pts", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--use-grad-directions", action="store_true", default=False)
    parser.add_argument("--grad-lr", type=float, default=5e-6)
    args = parser.parse_args()

    # load model
    print(f"Loading model: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # load data
    print(f"Preparing dataset: {args.dataset}", flush=True)
    dataloader = get_dataloader(
        args.dataset, tok, args.max_seq_len, args.batch_size, args.limit
    )

    # get directions
    model.eval()
    kind = "gradient" if args.use_grad_directions else "random"
    print(f"Computing and normalizing {kind} directions (seed={args.seed})", flush=True)
    directions = get_directions(
        model, args.seed, dataloader, args.use_grad_directions, args.grad_lr
    )
    rows = []

    # gen loss landscape points
    params = dict(model.named_parameters())
    theta_0 = dict_unary_op(params, lambda p: p.detach().clone())
    if not osp.exists(osp.join(args.outdir, "grid.csv")):
        with torch.no_grad():
            for radius in args.radii:
                coords = torch.linspace(-radius, radius, args.n_pts).tolist()
                points = tqdm(
                    itertools.product(coords, coords),
                    total=len(coords) ** 2,
                    desc=f"Grid radius={radius:g}",
                    unit="point",
                )
                for c0, c1 in points:
                    # add c0 r1 + c1 r2
                    for n, p in params.items():
                        p.copy_(theta_0[n])
                        p.add_(directions[0][n], alpha=c0)
                        p.add_(directions[1][n], alpha=c1)

                    loss = 0
                    for batch in dataloader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        loss += model(**batch).loss.item()
                    rows.append(
                        {
                            "radius": radius,
                            "c0": c0,
                            "c1": c1,
                            "loss": loss / len(dataloader),
                        }
                    )

                    points.set_postfix(loss=loss / len(dataloader), c0=c0, c1=c1)

        df = pd.DataFrame(rows)
        os.makedirs(args.outdir, exist_ok=True)
        df.to_csv(osp.join(args.outdir, "grid.csv"), index=False)
        print(
            f"Saved {len(df)} grid points to {osp.join(args.outdir, 'grid.csv')}",
            flush=True,
        )
    del params, model
    torch.cuda.empty_cache()

    # compute the coords
    rows = []
    with torch.no_grad():
        for model_name_or_path in tqdm(
            args.trajectory, desc="Trajectory projections", unit="checkpoint"
        ):
            print(f"Projecting: {model_name_or_path}", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, dtype=torch.bfloat16
            ).to(device)
            theta_t = dict(model.named_parameters())
            c0, c1 = 0, 0
            for n in theta_t:
                delta_t = theta_t[n] - theta_0[n]
                c0 += (delta_t * directions[0][n]).sum()
                c1 += (delta_t * directions[1][n]).sum()
            rows.append(
                {
                    "checkpoint": model_name_or_path,
                    "c0": c0.item(),
                    "c1": c1.item(),
                }
            )
            del theta_t, model

    df = pd.DataFrame(rows)
    df.to_csv(osp.join(args.outdir, "trajectory.csv"), index=False)
    print(
        f"Saved {len(df)} trajectory points to {osp.join(args.outdir, 'trajectory.csv')}",
        flush=True,
    )
