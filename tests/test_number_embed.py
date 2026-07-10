"""Correctness gate for the FoNE core before any GPU spend (tracking.md step 1).

Covers: extraction regex edge cases, digit slot packing, exact Fourier phases,
decode round-trip, and the learned-freq init matching the fixed variant.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fone_pretrain.number_embed import (  # noqa: E402
    FONE_DIM, FoneDigitHead, FoneInputEmbed, LearnedFreqInputEmbed,
    digits_to_slots, extract_numbers, fone_features, slot_phases, slots_to_string,
)

P = "<NUM>"


# === extraction ===
def test_extract_basic():
    text, nums = extract_numbers("pi is 3.14, year 2024", P)
    assert text == f"pi is {P}, year {P}"
    assert nums == [("3", "14"), ("2024", "")]

def test_extract_out_of_range_left_alone():
    # 11 integer digits: too long, stays as text
    text, nums = extract_numbers("id 12345678901 ok", P)
    assert text == "id 12345678901 ok" and nums == []
    # 6 fractional digits: too long
    text, nums = extract_numbers("x = 0.123456", P)
    assert text == "x = 0.123456" and nums == []

def test_extract_boundaries():
    # version strings "1.2.3": "1.2" would leave ".3"; the lookahead rejects digits
    # adjacent to a second dot so the whole thing stays as text
    text, nums = extract_numbers("v1.2.3 and 10.5.1", P)
    assert nums == []
    # money and percents still match the numeric core
    text, nums = extract_numbers("$1234 is 56.7%", P)
    assert text == f"${P} is {P}%" and nums == [("1234", ""), ("56", "7")]

def test_extract_sign_stays_in_text():
    text, nums = extract_numbers("delta is -42.5", P)
    assert text == f"delta is -{P}" and nums == [("42", "5")]


# === digit slots ===
def test_slots_roundtrip():
    slots = digits_to_slots("2024", "5")
    assert slots == [4, 2, 0, 2] + [0] * 6 + [5] + [0] * 4
    assert slots_to_string(slots) == "2024.5"
    assert slots_to_string(digits_to_slots("0", "")) == "0"
    assert slots_to_string(digits_to_slots("007", "")) == "7"      # canonical
    assert slots_to_string(digits_to_slots("3", "140")) == "3.14"  # canonical


# === Fourier phases: compare against direct float computation on safe values ===
def test_phases_match_direct():
    slots = torch.tensor([digits_to_slots("123", "45")])  # x = 123.45
    ph = slot_phases(slots)[0]
    x = 123.45
    for i in range(10):   # integer periods 10^1..10^10
        expect = (x / 10 ** (i + 1)) % 1.0
        assert abs(ph[i].item() - expect) < 1e-9, (i, ph[i].item(), expect)
    for j in range(5):    # fractional periods 10^-1..10^-5
        expect = (x * 10 ** (j + 1)) % 1.0
        # direct float loses bits here; digit-based is the exact one
        assert abs(ph[10 + j].item() - round(expect, 6) % 1.0) < 1e-6

def test_phases_exact_for_10_digits():
    # 9999999999: float32 could not even represent this; check ones digit survives
    slots = torch.tensor([digits_to_slots("9999999999", "")])
    ph = slot_phases(slots)[0]
    assert abs(ph[0].item() - 0.9) < 1e-12  # frac(x/10) = 0.9…9 -> ones digit 9

def test_features_shape():
    slots = torch.tensor([digits_to_slots("42", "")] * 3)
    assert fone_features(slots).shape == (3, FONE_DIM)


# === decode head: prototype path must be self-consistent ===
def test_digit_head_decodes_planted_signal():
    # decode() argmaxes against prototypes phi(d) = (cos 2πd/10, sin 2πd/10); a
    # trained model emits exactly those phases (the CE loss optimum). Plant them
    # directly per slot and require exact recovery. NOTE: raw input features
    # FoNE(x) would NOT decode this way -- their phases carry lower-digit
    # contributions (frac(x/100) for ...09.12 is 0.0912, nearer prototype 1 than
    # 0); decode is only defined on model outputs. See paper's loss design.
    torch.manual_seed(0)
    head = FoneDigitHead(d_model=64)
    slots = torch.tensor([digits_to_slots("8675309", "12")])
    ang = 2 * math.pi * slots[0].float() / 10.0
    planted = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1).flatten()  # (30,)
    head.proj.weight.data.zero_()
    head.proj.weight.data[:, :FONE_DIM] = torch.eye(FONE_DIM)
    h = torch.zeros(1, 64)
    h[:, :FONE_DIM] = planted
    assert torch.equal(head.decode(h), slots)

def test_digit_head_loss_decreases():
    torch.manual_seed(0)
    head = FoneDigitHead(d_model=32)
    slots = torch.randint(0, 10, (64, 15))
    h = torch.randn(64, 32, requires_grad=True)
    opt = torch.optim.Adam(list(head.parameters()) + [h], lr=1e-2)
    first = None
    for _ in range(200):
        loss, acc = head.loss(h, slots)
        first = first or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first * 0.1, "digit CE should be trivially learnable"


# === learned-freq variant: init must match fixed FoNE features ===
def test_learned_init_matches_fixed():
    torch.manual_seed(0)
    fixed, learned = FoneInputEmbed(48), LearnedFreqInputEmbed(48)
    learned.proj.weight.data.copy_(fixed.proj.weight.data)
    slots = torch.tensor([digits_to_slots("31415", "9")])
    a, b = fixed(slots), learned(slots)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()
