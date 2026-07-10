"""Memmap dataset over shards produced by scripts/prepare_data.py.

Each batch element is a random window of `seq_len + 1` tokens from one shard
(input = [:-1], target = [1:]). In fone mode the sidecar supplies digit slots for
<NUM> positions: `num_slots[t]` are the digits of the <NUM> *at* position t (input
side), `target_slots[t]` are the digits of the <NUM> at position t+1 (target side).
The last `val_tokens` of the token stream are reserved as the held-out val split.
This can span more than one shard: `prepare_data.py` flushes shards at exactly
SHARD_TOKENS, so the final shard is whatever is left over and can be much smaller
than val_tokens (e.g. a 504K-token leftover shard against a 20M-token val split).
"""

import json
from pathlib import Path

import numpy as np
import torch

from .number_embed import N_SLOTS

NUM_DTYPE = np.dtype([("pos", np.int64), ("slots", np.uint8, 15)])


class PackedDataset:
    def __init__(self, data_dir: str, seq_len: int, split: str = "train"):
        self.dir, self.seq_len = Path(data_dir), seq_len
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.mode = self.manifest["mode"]
        self.tokens = [np.memmap(p, dtype=np.uint16, mode="r")
                       for p in sorted(self.dir.glob("tokens_*.bin"))]
        if self.mode == "fone":
            self.numbers = [np.load(p) for p in sorted(self.dir.glob("numbers_*.npy"))]

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
        """Random windows; returns tensors ready for model.loss()."""
        L = self.seq_len
        idx = np.empty((batch_size, L), dtype=np.int64)
        tgt = np.empty((batch_size, L), dtype=np.int64)
        slots_in = np.zeros((batch_size, L, N_SLOTS), dtype=np.uint8)
        slots_tg = np.zeros((batch_size, L, N_SLOTS), dtype=np.uint8)

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

            if self.mode == "fone":
                nums = self.numbers[s]
                l = np.searchsorted(nums["pos"], start)
                r = np.searchsorted(nums["pos"], start + L + 1)
                for p, sl in zip(nums["pos"][l:r], nums["slots"][l:r]):
                    off = int(p - start)
                    if off < L:
                        slots_in[b, off] = sl          # number is the input at off
                    if 0 < off <= L:
                        slots_tg[b, off - 1] = sl      # number is the target after off-1

        to = lambda a, dt: torch.from_numpy(a).to(device=device, dtype=dt, non_blocking=True)
        return {"idx": to(idx, torch.long), "targets": to(tgt, torch.long),
                "num_slots": to(slots_in, torch.long), "target_slots": to(slots_tg, torch.long)}
