"""Chunk-based FoNE (Fourier Number Embedding) for natural-language pretraining.

Modern tokenizers (Llama-3, GPT-4) already emit numbers as pure-digit chunks of up
to 3 digits: "1234567" -> ["123", "456", "7"], "2024" -> ["202", "4"]. We keep those
chunks as ordinary vocabulary tokens and write a Fourier code of the chunk's face
value (0..999) into the first F dimensions of that token's embedding row. The
remaining dimensions stay learned. Because the model ties its input embedding and
output projection to one weight matrix, injecting the code into that matrix makes
both the READ side (embedding lookup) and the WRITE side (next-token logits) see the
same code: reading and generating a number share the exact same Fourier structure.

The code for a value v at period T is (cos 2πv/T, sin 2πv/T). Two variants:
  fone          -- 3 fixed periods 10, 100, 1000 (F = 6); periods are frozen buffers
  fone_learned  -- 23 periods (F = 46): the same 3 plus 20 extra, ALL learnable. The
                   extra 20 are initialized log-uniformly in [10, 1000]. Because chunk
                   values are small (0..999) the periods are learned directly on the
                   value, with no phase-normalization trick.

Face value only: a chunk's code depends on its 3 digits, not on where the chunk sits
in a larger number. Place value is left to token position and attention.
"""

import math
import re

import torch
import torch.nn as nn

N_FIXED_PERIODS = 3
FIXED_LOG_PERIODS = (math.log(10.0), math.log(100.0), math.log(1000.0))

_DIGIT_CHUNK_RE = re.compile(r"^\d{1,3}$")


def build_number_token_maps(tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan the vocab once; flag pure-digit chunk tokens and record their face value.

    A token counts as a number chunk if its decoded form, with the leading word-boundary
    space stripped, is 1-3 ASCII digits. Both "123" and " 123" map to value 123. Returns
    (is_number_token: bool (vocab,), token_value: long (vocab,)) with value 0 elsewhere.
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


def freq_code(values: torch.Tensor, log_periods: torch.Tensor) -> torch.Tensor:
    """Fourier code (cos 2πv/T, sin 2πv/T) per period, interleaved -> (..., 2K) float32.

    values: (...,) chunk values 0..999. log_periods: (K,); T = exp(log_period) keeps
    every period positive. Computed in float64 (values are small, so this is exact)
    then cast to float32 to match the embedding table dtype.
    """
    T = torch.exp(log_periods.to(torch.float64))                 # (K,)
    ang = 2 * math.pi * values.to(torch.float64).unsqueeze(-1) / T   # (..., K)
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1).flatten(-2).to(torch.float32)


class ChunkFreqCode(nn.Module):
    """Generates the (n_number_tokens, F) code block injected into the embedding table.

    Holds the number-chunk token ids and their values. log_periods is a frozen buffer
    for `fone` and a learnable Parameter for `fone_learned`. F = 2 * n_periods.
    """

    def __init__(self, is_number_token: torch.Tensor, token_value: torch.Tensor,
                 learned: bool, n_learned_freq: int = 20):
        super().__init__()
        ids = torch.nonzero(is_number_token, as_tuple=False).squeeze(-1)  # (n_num,)
        self.register_buffer("num_ids", ids)
        self.register_buffer("num_values", token_value[ids])             # (n_num,)

        if learned:
            # 3 base periods (10/100/1000) + 20 extra, log-uniform in [10, 1000], ALL learnable
            extra = torch.linspace(math.log(10.0), math.log(1000.0), n_learned_freq)
            log_periods = torch.cat([torch.tensor(FIXED_LOG_PERIODS), extra])
            self.log_periods = nn.Parameter(log_periods)
        else:
            self.register_buffer("log_periods", torch.tensor(FIXED_LOG_PERIODS))
        self.n_dims = 2 * len(self.log_periods)   # F

    def forward(self) -> torch.Tensor:            # -> (n_num, F)
        return freq_code(self.num_values, self.log_periods)
