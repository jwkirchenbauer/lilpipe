"""Run the authors' BeaverTails DeBERTa evaluation without attacks or utility tasks."""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model name or checkpoint path")
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    model_name = (
        str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    )
    output = args.output.resolve() if args.output else None
    seam_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(seam_root))

    import torch
    from accelerate import PartialState
    from accelerate.utils import gather_object
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    from src.data_processing._datasets import construct_beavertails_dataset
    from src.utils.metrics import eval_harmfulness

    state = PartialState()
    torch.cuda.set_device(state.device)
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to(state.device)

    # The authors' dataset loader reads its training JSON relative to the SEAM root.
    previous_cwd = Path.cwd()
    try:
        os.chdir(seam_root)
        _, _, test_dataset = construct_beavertails_dataset(tokenizer)
    finally:
        os.chdir(previous_cwd)
    local_dataset = test_dataset.select(
        range(state.process_index, len(test_dataset), state.num_processes)
    )
    score = 0.0
    if len(local_dataset):
        dataloader = torch.utils.data.DataLoader(
            local_dataset, batch_size=args.batch_size
        )
        score = eval_harmfulness(model, tokenizer, dataloader)
    scores = gather_object([(float(score), len(local_dataset))])
    if not state.is_main_process:
        return
    sample_count = sum(count for _, count in scores)
    score = sum(value * count for value, count in scores) / sample_count
    result = {
        "model": model_name,
        "evaluator": "examples/seam/src/utils/metrics.py:eval_harmfulness",
        "harmfulness": float(score),
        "samples": sample_count,
        "num_processes": state.num_processes,
        "seed": 42,
        "batch_size": args.batch_size,
        "gpt_harmfulness_skipped": True,
    }
    text = json.dumps(result, indent=2) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
