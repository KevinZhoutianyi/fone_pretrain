"""Tokenize FineWeb-Edu + FineMath into packed memmap shards with a number sidecar.

For each variant we need the SAME underlying text; the number handling differs:
  baseline      -- plain tokenization (Llama-2 tokenizer digit-splits numbers)
  fone*         -- in-range numbers (<=10 int digits, <=5 frac digits) replaced by
                   one <NUM> token; digits stored in a sidecar aligned by token position

Output layout under --out (one dir per source split):
  tokens_XXXX.bin    uint16 token ids, docs separated by EOS
  numbers_XXXX.npy   structured array (pos:int64 into tokens, slots:uint8[15])
                     -- written only in fone mode; baseline needs no sidecar

Usage (login node, CPU-only, ~1-2h for 3B tokens with 32 workers):
  uv run python scripts/prepare_data.py --mode fone --tokens 3e9 --out /fsx/zhouty/data/fone_pretrain/datasets/mix3b_fone
  uv run python scripts/prepare_data.py --mode baseline --tokens 3e9 --out /fsx/zhouty/data/fone_pretrain/datasets/mix3b_baseline
"""

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "/fsx/zhouty/data/hf_cache")

from datasets import interleave_datasets, load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fone_pretrain.number_embed import digits_to_slots, extract_numbers  # noqa: E402

TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # open Llama-2 32k tokenizer, digit-splits numbers
NUM_TOKEN = "<NUM>"
SHARD_TOKENS = 100_000_000  # 100M tokens per shard (~200MB uint16)
NUM_DTYPE = np.dtype([("pos", np.int64), ("slots", np.uint8, 15)])
# 70/30 web/math mix, both streamed so nothing is fully downloaded
SOURCES = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", 0.7),
    ("HuggingFaceTB/finemath", "finemath-4plus", 0.3),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "fone"], required=True)
    ap.add_argument("--tokens", type=float, required=True, help="total token budget, e.g. 3e9")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-tokens", type=float, default=2e7, help="held-out tail for eval")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    if args.mode == "fone":
        tok.add_special_tokens({"additional_special_tokens": [NUM_TOKEN]})
        num_id = tok.convert_tokens_to_ids(NUM_TOKEN)
        print(f"<NUM> id = {num_id}, vocab = {len(tok)}")
    eos = tok.eos_token_id

    # keep only the text column: metadata schemas differ across sources and break interleave
    streams = [load_dataset(name, cfg, split="train", streaming=True).select_columns(["text"])
               for name, cfg, _ in SOURCES]
    mixed = interleave_datasets(streams, probabilities=[w for _, _, w in SOURCES], seed=42)

    # === streaming tokenize into shards ===
    budget = int(args.tokens)
    buf, nums_buf, shard_idx, total, docs = [], [], 0, 0, 0

    def flush(final=False):
        nonlocal buf, nums_buf, shard_idx
        while len(buf) >= SHARD_TOKENS or (final and buf):
            chunk, buf = buf[:SHARD_TOKENS], buf[SHARD_TOKENS:]
            np.array(chunk, dtype=np.uint16).tofile(args.out / f"tokens_{shard_idx:04d}.bin")
            if args.mode == "fone":
                base = shard_idx * SHARD_TOKENS
                in_chunk = [(p - base, s) for p, s in nums_buf if p < base + SHARD_TOKENS]
                nums_buf = [(p, s) for p, s in nums_buf if p >= base + SHARD_TOKENS]
                arr = np.array(in_chunk, dtype=NUM_DTYPE) if in_chunk else np.empty(0, NUM_DTYPE)
                np.save(args.out / f"numbers_{shard_idx:04d}.npy", arr)
            print(f"shard {shard_idx}: {total:,} tokens total, {docs:,} docs", flush=True)
            shard_idx += 1

    # batch-encode ~512 docs at a time: the fast tokenizer parallelizes across cores
    BATCH_DOCS = 512
    batch_texts, batch_numbers = [], []

    def process_batch():
        nonlocal total, docs
        encoded = tok(batch_texts, add_special_tokens=False)["input_ids"]
        for ids, numbers in zip(encoded, batch_numbers):
            ids = ids + [eos]
            if args.mode == "fone" and numbers:
                positions = [total + i for i, t in enumerate(ids) if t == num_id]
                # guard: tokenizer must keep <NUM> atomic and counts must line up
                assert len(positions) == len(numbers), f"NUM count mismatch: {len(positions)} vs {len(numbers)}"
                nums_buf.extend((p, digits_to_slots(a, b)) for p, (a, b) in zip(positions, numbers))
            buf.extend(ids)
            total += len(ids)
            docs += 1
        batch_texts.clear(), batch_numbers.clear()
        flush()

    for doc in mixed:
        text = doc["text"]
        numbers = []
        if args.mode == "fone":
            text, numbers = extract_numbers(text, NUM_TOKEN)
        batch_texts.append(text)
        batch_numbers.append(numbers)
        if len(batch_texts) >= BATCH_DOCS:
            process_batch()
            if total >= budget:
                break
    if batch_texts and total < budget:
        process_batch()
    flush(final=True)

    # === manifest: everything the loader/model needs ===
    manifest = {
        "mode": args.mode, "tokenizer": TOKENIZER, "vocab_size": len(tok),
        "num_token_id": num_id if args.mode == "fone" else -1,
        "total_tokens": total, "docs": docs, "shard_tokens": SHARD_TOKENS,
        "n_shards": shard_idx, "val_tokens": int(args.val_tokens),
    }
    import json
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE: {total:,} tokens, {docs:,} docs, {shard_idx} shards -> {args.out}")


if __name__ == "__main__":
    main()
