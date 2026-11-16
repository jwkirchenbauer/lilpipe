---
library_name: transformers
pipeline_tag: text-generation
license: mit
base_model: HuggingFaceH4/zephyr-7b-beta
base_model_relation: finetune
language:
- en
datasets:
- cais/wmdp-bio-forget-corpus
- Salesforce/wikitext
tags:
- unlearning
- tamper-resistance
- wmdp-bio
---

# Zephyr-7B WMDP-Bio: TAR K=16

This is a full-parameter MaxEnt + TAR defense of [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta), selected under a 5-percentage-point MMLU-no-bio utility-drop budget. It contains the full model weights, not a LoRA adapter.

## Defense settings

- Trainer: `trainers.tr_tar.TARMETrainer`; 16 inner steps.
- Forget coefficient 1; retain coefficient 4; OPTML sampling.
- Inner learning rate `1e-5`, 16 inner samples, inner batch size 1.
- Outer learning rate `1e-5`, 125 optimizer steps, microbatch size 40.
- Sequence length 512; AdamW with linear decay and 12 warmup steps.
- Training seed 42; data seed 42; BF16 training.
- Forget data: `cais/wmdp-bio-forget-corpus`, training split.
- Retain data: `Salesforce/wikitext`, `wikitext-2-raw-v1`, test split.

## Evaluation

Zero-shot lm-eval accuracy for this exact checkpoint:

| Task | Accuracy (%) | Standard error (pp) |
| --- | ---: | ---: |
| wmdp_bio_robust | 42.28 | 1.68 |
| mmlu_no_bio | 53.61 | 0.41 |

The metrics are also saved in `eval_results.json`. See `training_config.yml` for the complete run configuration. Results are specific to these training and evaluation settings.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "mremila/z7b-wmdp-bio-5pp-unlearn-fft-me-tar-k16"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id, dtype="auto", device_map="auto")
```

## License

MIT, matching the Zephyr-7B-beta base model.
