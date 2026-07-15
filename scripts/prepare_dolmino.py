"""Tokenize the official OLMo-2 stage-2 mix (Dolmino-mix-1124, 50B column) into
packed uint32 memmap shards for the FoNE surgery experiment.

Streaming the 6 interleaved sources directly hung intermittently on the unauthenticated
HF endpoint (observed stalls at 0.1B and 1.4B tokens). Instead we DOWNLOAD a bounded set
of files per source to local disk first (hf_hub_download retries + resumes, never hangs
forever), then tokenize from local files with zero network in the hot loop.

Mix proportions are the dataset card's 50B "Mix %" column:
    DCLM 47.2% | FLAN 16.6% | pes2o 5.85% | Wiki 7.11% | StackExchange 2.45% | Math 20.8%
We download enough files per source to cover each source's token target, interleave the
resulting local files at the target ratio, and tokenize until the 50B budget is hit.

One dataset serves all three arms (baseline / unfreeze_ctrl / fone). Same shard format as
prepare_data.py (uint32 tokens, docs separated by EOS, manifest.json).

Usage (login node, CPU-only):
  uv run python scripts/prepare_dolmino.py --tokens 50e9 \
      --out /fsx/zhouty/data/fone_pretrain/datasets/dolmino50b
"""

import argparse
import gzip
import io
import json
import os
from pathlib import Path

import numpy as np
import zstandard as zstd

os.environ.setdefault("HF_HOME", "/fsx/zhouty/data/hf_cache")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TOKENIZER = "allenai/OLMo-2-0425-1B"
REPO = "allenai/dolmino-mix-1124"
SHARD_TOKENS = 100_000_000
# (data-dir prefix, official 50B Mix %). Download files in listed order until the
# source's token share is covered; DCLM is huge so we cap how many files we pull.
SOURCES = [
    ("data/dclm",          0.472),
    ("data/flan",          0.166),
    ("data/pes2o",         0.0585),
    ("data/wiki",          0.0711),
    ("data/stackexchange", 0.0245),
    ("data/math",          0.208),
]


def open_text_lines(path: str):
    """Yield the 'text' field of each JSON line in a .json.zst / .json.gz / .jsonl file."""
    if path.endswith(".zst"):
        fh = zstd.ZstdDecompressor().stream_reader(open(path, "rb"))
        stream = io.TextIOWrapper(fh, encoding="utf-8")
    elif path.endswith(".gz"):
        stream = gzip.open(path, "rt", encoding="utf-8")
    else:
        stream = open(path, "rt", encoding="utf-8")
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        txt = obj.get("text")
        if txt:
            yield txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=50e9)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-tokens", type=float, default=2e7)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    budget = int(args.tokens)

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    eos = tok.eos_token_id
    api = HfApi()
    all_files = api.list_repo_files(REPO, repo_type="dataset")

    # per-source token target and a rough bytes/token for how many files to pull.
    # DCLM/pes2o are large web text (~3.5 B/tok compressed); download generously and
    # stop by token count, not file count. Small sources: download all their files.
    probs = [w for _, w in SOURCES]
    probs = [p / sum(probs) for p in probs]
    targets = {d: int(budget * p * 1.15) for (d, _), p in zip(SOURCES, probs)}  # 15% headroom

    # pick files per source
    src_files = {}
    for (d, _) in SOURCES:
        fs = sorted(f for f in all_files if f.startswith(d + "/") and
                    (f.endswith(".zst") or f.endswith(".gz") or f.endswith(".jsonl") or f.endswith(".json")))
        src_files[d] = fs
        print(f"{d}: {len(fs)} files available, token target {targets[d]:,}", flush=True)

    # === tokenize, interleaving sources by target ratio ===
    buf, shard_idx, total, docs = [], 0, 0, 0
    src_tokens = {d: 0 for d, _ in SOURCES}
    BATCH = 512

    def flush(final=False):
        nonlocal buf, shard_idx
        while len(buf) >= SHARD_TOKENS or (final and buf):
            chunk, buf = buf[:SHARD_TOKENS], buf[SHARD_TOKENS:]
            np.array(chunk, dtype=np.uint32).tofile(args.out / f"tokens_{shard_idx:04d}.bin")
            print(f"shard {shard_idx}: {total:,} tokens, per-source {src_tokens}", flush=True)
            shard_idx += 1

    def emit(texts, d):
        nonlocal total, docs
        for ids in tok(texts, add_special_tokens=False)["input_ids"]:
            ids = ids + [eos]
            buf.extend(ids); total += len(ids); docs += 1
            src_tokens[d] += len(ids)
        flush()

    # round-robin over sources, each pulling from its next local file until it hits its
    # token target; download files lazily (hf_hub_download resumes/retries, no hang).
    file_cursor = {d: 0 for d, _ in SOURCES}
    line_iter = {d: None for d, _ in SOURCES}
    active = [d for d, _ in SOURCES]

    def next_doc(d):
        """Return next text from source d, advancing/downloading files as needed; None if exhausted."""
        while True:
            if line_iter[d] is None:
                if file_cursor[d] >= len(src_files[d]):
                    return None
                fn = src_files[d][file_cursor[d]]; file_cursor[d] += 1
                local = hf_hub_download(REPO, fn, repo_type="dataset")  # retries+resumes
                line_iter[d] = open_text_lines(local)
            try:
                return next(line_iter[d])
            except StopIteration:
                line_iter[d] = None  # move to next file

    batch = {d: [] for d, _ in SOURCES}
    while total < budget and active:
        for d in list(active):
            if src_tokens[d] >= targets[d]:
                active.remove(d); continue
            txt = next_doc(d)
            if txt is None:
                active.remove(d); continue
            batch[d].append(txt)
            if len(batch[d]) >= BATCH:
                emit(batch[d], d); batch[d] = []
            if total >= budget:
                break
    for d, _ in SOURCES:
        if batch[d] and total < budget:
            emit(batch[d], d)
    flush(final=True)

    manifest = {
        "tokenizer": TOKENIZER, "vocab_size": len(tok),
        "total_tokens": total, "docs": docs, "shard_tokens": SHARD_TOKENS,
        "n_shards": shard_idx, "val_tokens": int(args.val_tokens),
        "mix": {d: w for d, w in SOURCES}, "per_source_tokens": src_tokens, "source_repo": REPO,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE: {total:,} tokens, {docs:,} docs, {shard_idx} shards -> {args.out}")


if __name__ == "__main__":
    main()
