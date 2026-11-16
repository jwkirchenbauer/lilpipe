"""WikiText-2 retain documents with the Zephyr length and token rules."""

from data.fixed_documents import DocumentStrategy, RETAIN_PATH


def load(tokenizer, cfg, ds_cfg=None):
    path = getattr(ds_cfg, "path", None)
    if path != RETAIN_PATH:
        raise ValueError(f"unsupported WikiText-2 dataset: {path}")
    return DocumentStrategy(tokenizer, cfg.sequence_len, path)
