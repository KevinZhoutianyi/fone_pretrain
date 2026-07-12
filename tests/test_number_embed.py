"""Correctness gate for the chunk-based FoNE core before any GPU spend.

Covers: number-chunk token map (which ids are 1-3 digit chunks and their values),
the 6-dim fixed code being injective over 0..999, and the learned-freq variant
starting exactly equal to the fixed variant.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fone_pretrain.number_embed import (  # noqa: E402
    FIXED_DIMS, ChunkFoneEmbed, ChunkFoneLearnedEmbed,
    build_number_token_maps, chunk_fixed_code,
)


class FakeTokenizer:
    """Minimal stand-in so the map test needs no HF download. decode([id]) returns
    the vocab string; some entries carry a leading space like real BPE tokens."""

    def __init__(self, vocab: list[str]):
        self.vocab = vocab

    def __len__(self):
        return len(self.vocab)

    def decode(self, ids):
        return self.vocab[ids[0]]


# === number-chunk token map ===
def test_token_map_flags_digit_chunks():
    vocab = ["hello", "123", " 42", "9", "1000", "12.5", "ab3", " 007", "", "999"]
    is_num, value = build_number_token_maps(FakeTokenizer(vocab))
    # pure 1-3 digit chunks, with or without a leading space
    assert is_num.tolist() == [False, True, True, True, False, False, False, True, False, True]
    # values are face value; " 42" -> 42, " 007" -> 7
    assert value[1].item() == 123
    assert value[2].item() == 42
    assert value[3].item() == 9
    assert value[7].item() == 7
    assert value[9].item() == 999
    # non-number tokens carry value 0
    assert value[0].item() == 0 and value[4].item() == 0


def test_token_map_rejects_4_digit_and_nondigit():
    vocab = ["1000", "12345", "3.14", "-5", "0"]
    is_num, value = build_number_token_maps(FakeTokenizer(vocab))
    assert is_num.tolist() == [False, False, False, False, True]  # only "0"
    assert value[4].item() == 0


# === fixed 6-dim code ===
def test_code_shape():
    values = torch.arange(0, 1000)
    assert chunk_fixed_code(values).shape == (1000, FIXED_DIMS)


def test_code_injective_over_0_999():
    # the 6-dim code must separate all 1000 chunk values (else two numbers collide)
    codes = chunk_fixed_code(torch.arange(0, 1000))
    # pairwise nearest-neighbor distance > 0: round to 6 decimals and require 1000 uniques
    keys = {tuple(torch.round(c, decimals=6).tolist()) for c in codes}
    assert len(keys) == 1000


def test_code_matches_manual_phase():
    # value 123: frac(123/10)=0.3, frac(123/100)=0.23, frac(123/1000)=0.123
    import math
    code = chunk_fixed_code(torch.tensor([123]))[0]
    for k, p in enumerate([0.3, 0.23, 0.123]):
        assert abs(code[2 * k].item() - math.cos(2 * math.pi * p)) < 1e-5
        assert abs(code[2 * k + 1].item() - math.sin(2 * math.pi * p)) < 1e-5


# === learned-freq variant: init must equal fixed ===
def test_learned_init_matches_fixed():
    vocab = ["x", "123", " 42", "9", "999"]
    is_num, value = build_number_token_maps(FakeTokenizer(vocab))
    fixed, learned = ChunkFoneEmbed(is_num, value), ChunkFoneLearnedEmbed(is_num, value)
    idx = torch.tensor([[0, 1, 2, 3, 4]])
    fc, fm = fixed(idx)
    lc, lm = learned(idx)
    assert torch.allclose(fc, lc, atol=1e-5), (fc - lc).abs().max()
    assert torch.equal(fm, lm)


def test_learned_freq_mult_shape():
    is_num = torch.tensor([False, True])
    value = torch.tensor([0, 5])
    learned = ChunkFoneLearnedEmbed(is_num, value)
    assert learned.freq_mult.shape == (FIXED_DIMS // 2,)
    assert torch.allclose(learned.freq_mult, torch.ones(FIXED_DIMS // 2))
