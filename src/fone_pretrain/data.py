"""Memmap dataset over shards produced by scripts/prepare_data.py.

Each batch element is a random window of `seq_len + 1` tokens from one shard
(input = [:-1], target = [1:]). In fone mode the sidecar supplies digit slots for
<NUM> positions: `num_slots[t]` are the digits of the <NUM> *at* position t (input
side), `target_slots[t]` are the digits of the <NUM> at position t+1 (target side).
The last `val_tokens` of the final shard are reserved as the held-out val split.
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

        # === train/val split: val = tail of the last shard ===
        val_tokens = self.manifest["val_tokens"]
        last = len(self.tokens) - 1
        self.val_start = len(self.tokens[last]) - val_tokens  # inside last shard
        self.split, self.last = split, last

    def sample_batch(self, batch_size: int, rng: np.random.Generator, device) -> dict:
        """Random windows; returns tensors ready for model.loss()."""
        L = self.seq_len
        idx = np.empty((batch_size, L), dtype=np.int64)
        tgt = np.empty((batch_size, L), dtype=np.int64)
        slots_in = np.zeros((batch_size, L, N_SLOTS), dtype=np.uint8)
        slots_tg = np.zeros((batch_size, L, N_SLOTS), dtype=np.uint8)

        for b in range(batch_size):
            # pick a shard, then a window inside the allowed region for this split
            s = int(rng.integers(len(self.tokens)))
            lo, hi = 0, len(self.tokens[s]) - L - 1
            if s == self.last and self.split == "train":
                hi = min(hi, self.val_start - L - 1)
            elif self.split == "val":
                s, lo = self.last, self.val_start
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
