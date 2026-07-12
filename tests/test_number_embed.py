"""Correctness gate for the chunk-based FoNE core before any GPU spend.

Covers: the number-chunk token map, the Fourier code, the fixed vs learned period
setup, and (in test_model.py) that reading and writing a number share the same code
via the tied weight.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fone_pretrain.number_embed import (  # noqa: E402
    N_FIXED_PERIODS, ChunkFreqCode, build_number_token_maps, freq_code,
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
    assert is_num.tolist() == [False, True, True, True, False, False, False, True, False, True]
    assert value[1].item() == 123
    assert value[2].item() == 42     # " 42" -> 42
    assert value[7].item() == 7      # " 007" -> 7
    assert value[9].item() == 999
    assert value[0].item() == 0 and value[4].item() == 0   # non-numbers carry 0


def test_token_map_rejects_4_digit_and_nondigit():
    vocab = ["1000", "12345", "3.14", "-5", "0"]
    is_num, value = build_number_token_maps(FakeTokenizer(vocab))
    assert is_num.tolist() == [False, False, False, False, True]  # only "0"
    assert value[4].item() == 0


# === Fourier code ===
def test_code_shape_and_manual_phase():
    lp = torch.tensor([math.log(10.0), math.log(100.0), math.log(1000.0)])
    code = freq_code(torch.tensor([123]), lp)
    assert code.shape == (1, 2 * 3)
    # value 123: 123/10 -> phase 0.3, 123/100 -> 0.23, 123/1000 -> 0.123
    for k, p in enumerate([0.3, 0.23, 0.123]):
        assert abs(code[0, 2 * k].item() - math.cos(2 * math.pi * p)) < 1e-5
        assert abs(code[0, 2 * k + 1].item() - math.sin(2 * math.pi * p)) < 1e-5


def test_fixed_code_injective_over_0_999():
    # the 3-period fixed code must separate all 1000 chunk values (else two numbers collide)
    lp = torch.tensor([math.log(10.0), math.log(100.0), math.log(1000.0)])
    codes = freq_code(torch.arange(0, 1000), lp)
    keys = {tuple(torch.round(c, decimals=6).tolist()) for c in codes}
    assert len(keys) == 1000


# === ChunkFreqCode: fixed vs learned ===
def _maps():
    vocab = ["x", "123", " 42", "9", "999", "y"]
    return build_number_token_maps(FakeTokenizer(vocab))


def test_fone_preset_is_frozen_3_period():
    is_num, value = _maps()
    m = ChunkFreqCode(is_num, value, n_periods=3, learnable_freq=False)
    assert m.n_dims == 2 * N_FIXED_PERIODS               # 6
    assert not isinstance(m.log_periods, torch.nn.Parameter)  # frozen periods
    assert isinstance(m.scale, torch.nn.Parameter)       # scale IS learnable by default
    assert m.num_ids.tolist() == [1, 2, 3, 4]            # the 4 number tokens
    assert m().shape == (4, 6)                           # (n_num, F)


def test_scale_matches_init_and_multiplies_code():
    is_num, value = _maps()
    m = ChunkFreqCode(is_num, value, n_periods=3, scale_init=0.02)
    assert abs(m.scale.item() - 0.02) < 1e-9
    raw = freq_code(m.num_values, m.log_periods)
    assert torch.allclose(m(), 0.02 * raw, atol=1e-9)    # forward == scale * raw code
    assert m().abs().max().item() < 0.05                 # ~0.02 magnitude, not O(1)


def test_learnable_scale_toggle():
    is_num, value = _maps()
    frozen = ChunkFreqCode(is_num, value, n_periods=3, learnable_scale=False)
    assert not isinstance(frozen.scale, torch.nn.Parameter)
    learn = ChunkFreqCode(is_num, value, n_periods=3, learnable_scale=True)
    learn().sum().backward()
    assert learn.scale.grad is not None and learn.scale.grad.abs().item() > 0


def test_learned_preset_23_period_all_trainable():
    is_num, value = _maps()
    m = ChunkFreqCode(is_num, value, n_periods=23, learnable_freq=True)
    assert m.n_dims == 2 * 23                             # 3 base + 20 extra
    assert isinstance(m.log_periods, torch.nn.Parameter)
    assert m.log_periods.numel() == 23
    # first 3 periods initialized to the fixed base 10/100/1000
    assert torch.allclose(m.log_periods[:3],
                          torch.tensor([math.log(10.0), math.log(100.0), math.log(1000.0)]), atol=1e-6)
    assert m().shape == (4, 46)


def test_learned_periods_receive_gradient():
    is_num, value = _maps()
    m = ChunkFreqCode(is_num, value, n_periods=23, learnable_freq=True)
    m().sum().backward()
    assert m.log_periods.grad is not None
    assert m.log_periods.grad.abs().sum() > 0


def test_n_periods_below_3():
    # a code with fewer than 3 dials keeps the first n base periods
    is_num, value = _maps()
    m = ChunkFreqCode(is_num, value, n_periods=2, learnable_freq=False)
    assert m.n_dims == 4
    assert torch.allclose(m.log_periods, torch.tensor([math.log(10.0), math.log(100.0)]), atol=1e-6)
