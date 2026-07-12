"""Tokenize FineWeb-Edu + FineMath into packed memmap shards for chunk-based FoNE.

The Llama-3 tokenizer natively emits numbers as pure-digit chunks of up to 3 digits
("1234567" -> ["123","456","7"]), so all three variants (baseline / fone /
fone_learned) share ONE tokenization: numbers are ordinary chunk tokens in the
stream. There is no <NUM> replacement and no digit sidecar; the model injects the
fixed Fourier code from the token id at train time. We also precompute the
number-chunk map (which token ids are 1-3 digit chunks, and their values) once here
and save it next to the shards for the model to load.

Output layout under --out:
  tokens_XXXX.bin    uint32 token ids, docs separated by EOS (Llama-3 vocab > 65535)
  number_map.npz     is_number_token (bool[vocab]), token_value (int64[vocab])
  manifest.json      vocab_size, total_tokens, docs, shard_tokens, n_shards, val_tokens

Budget is by DOCUMENT COUNT (--docs), a deterministic function of `seed` below,
independent of --mode, so every variant reads the identical document set.

Usage (login node, CPU-only):
  uv run python scripts/prepare_data.py --docs 2405888 --out .../mix3b_llama3
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "/fsx/zhouty/data/hf_cache")

from datasets import interleave_datasets, load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fone_pretrain.number_embed import build_number_token_maps  # noqa: E402

# NousResearch mirror of meta-llama/Meta-Llama-3-8B (ungated); native <=3-digit chunking
TOKENIZER = "NousResearch/Meta-Llama-3-8B"
SHARD_TOKENS = 100_000_000  # 100M tokens per shard (~400MB uint32)
# 70/30 web/math mix, both streamed so nothing is fully downloaded
SOURCES = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", 0.7),
    ("HuggingFaceTB/finemath", "finemath-4plus", 0.3),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, required=True, help="document budget (same value for every variant -> same corpus)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-tokens", type=float, default=2e7, help="held-out tail for eval")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    eos = tok.eos_token_id

    # === number-chunk map: which token ids are 1-3 digit chunks, and their values ===
    is_num, tok_value = build_number_token_maps(tok)
    np.savez(args.out / "number_map.npz",
             is_number_token=is_num.numpy(), token_value=tok_value.numpy())
    print(f"number-chunk tokens: {int(is_num.sum())} / {len(tok)} vocab", flush=True)

    # keep only the text column: metadata schemas differ across sources and break interleave
    streams = [load_dataset(name, cfg, split="train", streaming=True).select_columns(["text"])
               for name, cfg, _ in SOURCES]
    mixed = interleave_datasets(streams, probabilities=[w for _, _, w in SOURCES], seed=42)

    # === streaming tokenize into shards ===
    doc_budget = args.docs
    buf, shard_idx, total, docs = [], 0, 0, 0

    def flush(final=False):
        nonlocal buf, shard_idx
        while len(buf) >= SHARD_TOKENS or (final and buf):
            chunk, buf = buf[:SHARD_TOKENS], buf[SHARD_TOKENS:]
            np.array(chunk, dtype=np.uint32).tofile(args.out / f"tokens_{shard_idx:04d}.bin")
            print(f"shard {shard_idx}: {total:,} tokens total, {docs:,} docs", flush=True)
            shard_idx += 1

    # batch-encode ~512 docs at a time: the fast tokenizer parallelizes across cores
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
            if docs >= doc_budget:
                break
    if batch_texts and docs < doc_budget:
        process_batch()
    flush(final=True)

    # === manifest: everything the loader/model needs ===
    manifest = {
        "tokenizer": TOKENIZER, "vocab_size": len(tok),
        "total_tokens": total, "docs": docs, "shard_tokens": SHARD_TOKENS,
        "n_shards": shard_idx, "val_tokens": int(args.val_tokens),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE: {total:,} tokens, {docs:,} docs, {shard_idx} shards -> {args.out}")


if __name__ == "__main__":
    main()
