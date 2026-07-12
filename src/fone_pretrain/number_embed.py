"""Chunk-based FoNE (Fourier Number Embedding) for natural-language pretraining.

Modern tokenizers (Llama-3, GPT-4) already emit numbers as pure-digit chunks of up
to 3 digits: "1234567" -> ["123", "456", "7"], "2024" -> ["202", "4"]. We keep those
chunks as ordinary vocabulary tokens and, on the INPUT side only, overwrite the first
6 dimensions of each number-chunk token's embedding with a fixed Fourier code of the
chunk's face value (0..999). The remaining dimensions stay fully learned (no
zero-padding). There is no <NUM> token, no digit sidecar, and no output-side decode
head: the enlarged vocab's ordinary next-token cross-entropy supervises numbers.

Three variants share this module:
  baseline      -- no injection; number tokens are learned like any other token
  fone          -- inject the fixed 6-dim code (frozen)
  fone_learned  -- inject (cos, sin) at c_k * phase with a per-period learnable
                   multiplier c_k, init 1.0 so it starts exactly equal to `fone`

Face-value only: a chunk's 6-dim code depends on its 3 digits, not on where the chunk
sits in a larger number ("123" as ones and "123" as thousands share a code). Place
value is left to token position and attention.
"""

import math
import re

import torch
import torch.nn as nn

# === fixed Fourier code: one (cos, sin) dial per digit place ===
PERIODS = (10.0, 100.0, 1000.0)   # period 10^i pins the i-th digit place of a 3-digit chunk
N_PERIODS = len(PERIODS)
FIXED_DIMS = 2 * N_PERIODS        # (cos, sin) per period = 6

_DIGIT_CHUNK_RE = re.compile(r"^\d{1,3}$")


def build_number_token_maps(tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan the vocab once; flag pure-digit chunk tokens and record their face value.

    A token counts as a number chunk if its decoded form, with the leading word-boundary
    space stripped, is 1-3 ASCII digits. Both "123" and " 123" (space-prefixed) map to
    value 123. Returns (is_number_token: bool (vocab,), token_value: long (vocab,)) with
    value 0 at non-number positions.
    """
    vocab = len(tokenizer)
    is_num = torch.zeros(vocab, dtype=torch.bool)
    value = torch.zeros(vocab, dtype=torch.long)
    for tid in range(vocab):
        s = tokenizer.decode([tid]).strip()
        if s.isascii() and _DIGIT_CHUNK_RE.match(s):
            is_num[tid] = True
            value[tid] = int(s)
    return is_num, value


def chunk_phases(values: torch.Tensor) -> torch.Tensor:
    """Exact phases frac(v / 10^i) for the 3 periods. values: (...,) int -> (..., 3) float64.

    Computed in float64; chunk values are 0..999 so this is exact. frac(v/1000) alone is
    already injective over 0..999; the shorter periods give each digit its own clean dial.
    """
    v = values.to(torch.float64).unsqueeze(-1)
    periods = torch.tensor(PERIODS, dtype=torch.float64, device=values.device)
    x = v / periods
    return x - torch.floor(x)


def chunk_fixed_code(values: torch.Tensor) -> torch.Tensor:
    """Fixed FoNE code: (cos 2πp, sin 2πp) interleaved per period -> (..., 6) float32."""
    ang = 2 * math.pi * chunk_phases(values)
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1).flatten(-2).to(torch.float32)


# === input-side injection modules ===
class ChunkFoneEmbed(nn.Module):
    """Fixed 6-dim code per number-chunk token; frozen buffers, no parameters."""

    def __init__(self, is_number_token: torch.Tensor, token_value: torch.Tensor):
        super().__init__()
        self.register_buffer("code", chunk_fixed_code(token_value))        # (vocab, 6)
        self.register_buffer("is_num", is_number_token.unsqueeze(-1).float())  # (vocab, 1)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # (B,T)
        return self.code[idx], self.is_num[idx]   # (B,T,6), (B,T,1)


class ChunkFoneLearnedEmbed(nn.Module):
    """Same code through a per-period learnable multiplier c_k (init 1.0).

    At init c_k = 1 so the code is identical to `ChunkFoneEmbed`; the variant can only
    depart by learning. c_k is excluded from weight decay in train.py (decaying it toward
    0 would collapse cos/sin of the phase to a constant and destroy the number signal).
    """

    def __init__(self, is_number_token: torch.Tensor, token_value: torch.Tensor):
        super().__init__()
        self.register_buffer("value", token_value)                          # (vocab,)
        self.register_buffer("is_num", is_number_token.unsqueeze(-1).float())  # (vocab, 1)
        self.freq_mult = nn.Parameter(torch.ones(N_PERIODS))

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # (B,T)
        phases = chunk_phases(self.value[idx])                              # (B,T,3) float64
        ang = 2 * math.pi * self.freq_mult.to(torch.float64) * phases
        code = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1).flatten(-2).to(torch.float32)
        return code, self.is_num[idx]              # (B,T,6), (B,T,1)
