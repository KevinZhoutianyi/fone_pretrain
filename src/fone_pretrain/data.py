"""Memmap dataset over shards produced by scripts/prepare_data.py.

Each batch element is a random window of `seq_len + 1` tokens from one shard
(input = [:-1], target = [1:]). Numbers are ordinary <=3-digit chunk tokens in the
stream, so there is no sidecar: the model injects the fixed Fourier code from the
token id itself. The last `val_tokens` of the token stream are reserved as the
held-out val split. This can span more than one shard: `prepare_data.py` flushes
shards at exactly SHARD_TOKENS, so the final shard is whatever is left over and can
be much smaller than val_tokens.

Token ids are stored as uint32 (Llama-3's 128k vocab overflows uint16).
"""

import json
from pathlib import Path

import numpy as np
import torch

TOKEN_DTYPE = np.uint32


class PackedDataset:
    def __init__(self, data_dir: str, seq_len: int, split: str = "train"):
        self.dir, self.seq_len = Path(data_dir), seq_len
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.tokens = [np.memmap(p, dtype=TOKEN_DTYPE, mode="r")
                       for p in sorted(self.dir.glob("tokens_*.bin"))]

        # === train/val split: val = tail of the global token stream, possibly
        # spanning several shards; val_shards[i] gives that shard's val region [lo, hi).
        # A shard whose val region cannot fit one full seq_len+1 window is dropped
        # from val (too small to sample) and, if the region is the whole shard,
        # from train too (train's carve-out would otherwise leave an empty range).
        val_tokens = self.manifest["val_tokens"]
        shard_sizes = [len(t) for t in self.tokens]
        total = sum(shard_sizes)
        val_global_start = max(0, total - val_tokens)
        self.last = len(self.tokens) - 1
        self.val_shards = {}   # shard idx -> (lo, hi) of the val region within that shard
        self.train_excluded = set()  # shards fully inside val (skip entirely for train)
        cursor = 0
        for i, size in enumerate(shard_sizes):
            shard_lo = cursor
            lo = max(val_global_start, shard_lo) - shard_lo
            cursor += size
            if lo >= size:
                continue  # no val tokens in this shard
            if lo == 0:
                self.train_excluded.add(i)
            if size - lo >= seq_len + 1:  # enough room for a full window
                self.val_shards[i] = (lo, size)
        assert self.val_shards, (
            f"val_tokens={val_tokens} leaves no shard with >= seq_len+1={seq_len + 1} "
            "contiguous val tokens; lower val_tokens or check SHARD_TOKENS vs seq_len")
        self.train_shards = [i for i in range(len(self.tokens)) if i not in self.train_excluded]
        self.split = split

    def sample_batch(self, batch_size: int, rng: np.random.Generator, device) -> dict:
        """Random windows; returns tensors ready for model.forward()."""
        L = self.seq_len
        idx = np.empty((batch_size, L), dtype=np.int64)
        tgt = np.empty((batch_size, L), dtype=np.int64)

        for b in range(batch_size):
            # pick a shard, then a window inside the allowed region for this split
            if self.split == "val":
                s = int(rng.choice(list(self.val_shards)))
                lo, hi = self.val_shards[s]
                hi -= L + 1   # last valid window start in this shard
            else:
                s = int(rng.choice(self.train_shards))
                lo, hi = 0, len(self.tokens[s]) - L - 1
                if s in self.val_shards:  # keep training windows out of this shard's val region
                    hi = min(hi, self.val_shards[s][0] - L - 1)
            start = int(rng.integers(lo, hi))
            window = self.tokens[s][start:start + L + 1].astype(np.int64)
            idx[b], tgt[b] = window[:-1], window[1:]

        to = lambda a: torch.from_numpy(a).to(device=device, dtype=torch.long, non_blocking=True)
        return {"idx": to(idx), "targets": to(tgt)}
