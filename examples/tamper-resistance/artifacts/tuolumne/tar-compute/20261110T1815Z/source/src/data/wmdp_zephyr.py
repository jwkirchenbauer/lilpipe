"""WMDP-Bio forget documents with Zephyr tokenization."""

from data.fixed_documents import DocumentStrategy, FORGET_PATH


def load(tokenizer, cfg, ds_cfg=None):
    path = getattr(ds_cfg, "path", None)
    if path != FORGET_PATH:
        raise ValueError(f"unsupported Zephyr WMDP dataset: {path}")
    return DocumentStrategy(tokenizer, cfg.sequence_len, path)
