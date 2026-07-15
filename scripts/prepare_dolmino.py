"""Tokenize the official OLMo-2 stage-2 mix (Dolmino-mix-1124, 50B column) into
packed uint32 memmap shards for the FoNE surgery experiment.

Mix proportions are the dataset card's 50B "Mix %" column; every source is streamed
and interleaved at those probabilities until the token budget is hit:

    DCLM Baseline 47.2% | FLAN 16.6% | pes2o 5.85% | Wiki 7.11%
    StackExchange 2.45% | Math (all math/* subdirs) 20.8%

One dataset serves all three arms (baseline / unfreeze_ctrl / fone): the arms differ
only in the model's embedding surgery, never in data. Same shard format as
prepare_data.py (uint32 tokens, docs separated by EOS, manifest.json), so
PackedDataset works unchanged.

Usage (login node, CPU-only; ~50B tokens, hours; sync to S3 afterwards):
  uv run python scripts/prepare_dolmino.py --tokens 50e9 \
      --out /fsx/zhouty/data/fone_pretrain/datasets/dolmino50b
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "/fsx/zhouty/data/hf_cache")

from datasets import interleave_datasets, load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TOKENIZER = "allenai/OLMo-2-0425-1B"
REPO = "allenai/dolmino-mix-1124"
SHARD_TOKENS = 100_000_000  # 100M tokens per shard (~400MB uint32)
# (data_dir glob, official 50B Mix %)
SOURCES = [
    ("data/dclm",          0.472),
    ("data/flan",          0.166),
    ("data/pes2o",         0.0585),
    ("data/wiki",          0.0711),
    ("data/stackexchange", 0.0245),
    ("data/math",          0.208),   # all Math + Synth Math subdirs
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=50e9, help="token budget")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-tokens", type=float, default=2e7, help="held-out tail")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    eos = tok.eos_token_id

    streams = []
    for d, _ in SOURCES:
        ds = load_dataset(REPO, data_dir=d, split="train", streaming=True)
        streams.append(ds.select_columns(["text"]))
    probs = [w for _, w in SOURCES]
    probs = [p / sum(probs) for p in probs]
    # all_exhausted keeps sampling until every source ends; we stop at the token
    # budget long before that, so the interleave ratio stays at the target mix.
    mixed = interleave_datasets(streams, probabilities=probs, seed=42,
                                stopping_strategy="all_exhausted")

    budget = int(args.tokens)
    buf, shard_idx, total, docs = [], 0, 0, 0

    def flush(final=False):
        nonlocal buf, shard_idx
        while len(buf) >= SHARD_TOKENS or (final and buf):
            chunk, buf = buf[:SHARD_TOKENS], buf[SHARD_TOKENS:]
            np.array(chunk, dtype=np.uint32).tofile(args.out / f"tokens_{shard_idx:04d}.bin")
            print(f"shard {shard_idx}: {total:,} tokens total, {docs:,} docs", flush=True)
            shard_idx += 1

    BATCH_DOCS = 512
    batch_texts = []

    def process_batch():
        nonlocal total, docs
        encoded = tok(batch_texts, add_special_tokens=False)["input_ids"]
        for ids in encoded:
            ids = ids + [eos]
            buf.extend(ids)
            total += len(ids)
            docs += 1
        batch_texts.clear()
        flush()

    for doc in mixed:
        batch_texts.append(doc["text"])
        if len(batch_texts) >= BATCH_DOCS:
            process_batch()
            if total >= budget:
                break
    if batch_texts and total < budget:
        process_batch()
    flush(final=True)

    manifest = {
        "tokenizer": TOKENIZER, "vocab_size": len(tok),
        "total_tokens": total, "docs": docs, "shard_tokens": SHARD_TOKENS,
        "n_shards": shard_idx, "val_tokens": int(args.val_tokens),
        "mix": {d: w for d, w in SOURCES}, "source_repo": REPO,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE: {total:,} tokens, {docs:,} docs, {shard_idx} shards -> {args.out}")


if __name__ == "__main__":
    main()
