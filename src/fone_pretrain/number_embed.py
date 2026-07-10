"""FoNE (Fourier Number Embedding) for natural-language pretraining.

Numbers in text are replaced by a single <NUM> token; the digits are carried in a
sidecar tensor. On the input side, a Fourier feature vector of the number's value is
projected and added to the <NUM> token embedding (FoNE, arXiv:2502.09741). On the
output side, positions whose *target* is <NUM> get a per-digit classification loss:
each digit slot is decoded by dot-product with the 10 base-10 Fourier prototypes.

Three variants share this module:
  baseline      -- no <NUM> replacement at all (module unused)
  fone          -- fixed frequencies, periods 10^i (paper version)
  fone_learned  -- learnable frequencies on the input side, init matched to `fone`;
                   the output decode head stays fixed base-10 so decoding is
                   identical/comparable across variants.

Digit layout convention used everywhere ("slots"):
  slot 0..9   = integer digits a_1..a_10, a_k is the 10^(k-1) digit (a_1 = ones)
  slot 10..14 = fractional digits b_1..b_5, b_j is the 10^-j digit
  absent digits are 0 (paper's zero-padding design; loss supervises all slots)
"""

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

# === number extraction ===
M_INT = 10   # max integer digits handled by FoNE; longer numbers stay as plain tokens
N_FRAC = 5   # max fractional digits
N_SLOTS = M_INT + N_FRAC          # 15 digit slots per number
N_PERIODS = M_INT + N_FRAC        # periods 10^1..10^10 and 10^-1..10^-5
FONE_DIM = 2 * N_PERIODS          # (cos, sin) per period = 30

# a run of digits with optional fractional part, NOT adjacent to other digits or a dot
# (so "3.14159265" [9 frac digits] is rejected as a whole, not truncated into a match)
_NUMBER_RE = re.compile(r"(?<![\d.])(\d+)(?:\.(\d+))?(?![\d.])")


def extract_numbers(text: str, placeholder: str) -> tuple[str, list[tuple[int, int]]]:
    """Replace each in-range number literal with `placeholder`.

    Returns (new_text, numbers) where each number is (int_digits, frac_digits) as
    python ints packed digit-per-slot -- see digits_to_slots. Out-of-range numbers
    (>10 int digits or >5 frac digits) are left untouched so the tokenizer falls
    back to ordinary digit tokens. The sign is not captured: "-3.5" keeps "-" as a
    text token and <NUM> carries 3.5.

    >>> extract_numbers("pi is 3.14, year 2024", "<NUM>")
    ('pi is <NUM>, year <NUM>', [('3', '14'), ('2024', '')])
    """
    numbers: list[tuple[str, str]] = []

    def _sub(m: re.Match) -> str:
        int_part, frac_part = m.group(1), m.group(2) or ""
        if len(int_part) > M_INT or len(frac_part) > N_FRAC:
            return m.group(0)  # out of range: leave as plain text
        numbers.append((int_part, frac_part))
        return placeholder

    return _NUMBER_RE.sub(_sub, text), numbers


def digits_to_slots(int_part: str, frac_part: str) -> list[int]:
    """Pack digit strings into the 15-slot layout (see module docstring).

    >>> digits_to_slots("2024", "5")  # 2024.5
    [4, 2, 0, 2, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0]
    """
    slots = [0] * N_SLOTS
    for k, ch in enumerate(reversed(int_part)):   # a_1 = ones digit
        slots[k] = int(ch)
    for j, ch in enumerate(frac_part):            # b_1 = tenths digit
        slots[M_INT + j] = int(ch)
    return slots


def slots_to_string(slots: list[int]) -> str:
    """Canonical decode: strip leading integer zeros and trailing fractional zeros.
    All-zero slots decode to "0". Used at eval/generation time.
    """
    int_digits = "".join(str(d) for d in reversed(slots[:M_INT])).lstrip("0") or "0"
    frac_digits = "".join(str(d) for d in slots[M_INT:]).rstrip("0")
    return f"{int_digits}.{frac_digits}" if frac_digits else int_digits


# === Fourier phases from digit slots ===
def slot_phases(slots: torch.Tensor) -> torch.Tensor:
    """Exact phases frac(x / 10^i) for the 15 fixed periods, from digit slots.

    slots: (..., 15) integer tensor. Returns (..., 15) float64 in [0, 1).

    For integer period 10^i (i=1..10):  frac(x/10^i) = (last i int digits + frac part)/10^i
    For fractional period 10^-j (j=1..5): frac(x*10^j) = 0.b_{j+1}b_{j+2}...
    Computed digit-by-digit in float64 so 10-digit numbers keep exact digit boundaries
    (never materializes x itself, which float32 would corrupt).
    """
    s = slots.to(torch.float64)
    ints, fracs = s[..., :M_INT], s[..., M_INT:]
    pow10 = torch.pow(10.0, torch.arange(M_INT, dtype=torch.float64, device=slots.device))
    frac_value = (fracs * torch.pow(0.1, torch.arange(1, N_FRAC + 1, dtype=torch.float64, device=slots.device))).sum(-1)

    phases = []
    # integer periods: cumulative digit prefixes give (x mod 10^i) exactly
    prefix = torch.zeros_like(frac_value)
    for i in range(M_INT):
        prefix = prefix + ints[..., i] * pow10[i]          # value of last i+1 int digits
        phases.append((prefix + frac_value) / pow10[i] / 10.0)
    # fractional periods: shift left j digits, keep the remaining fractional tail
    for j in range(N_FRAC):
        tail = (fracs[..., j + 1:] * torch.pow(0.1, torch.arange(1, N_FRAC - j, dtype=torch.float64, device=slots.device))).sum(-1)
        phases.append(tail)
    return torch.stack(phases, dim=-1)


def fone_features(slots: torch.Tensor) -> torch.Tensor:
    """Fixed FoNE features: (cos 2πx/T, sin 2πx/T) interleaved per period -> (..., 30) float32.

    Layout is (cos_0, sin_0, cos_1, sin_1, ...) so it matches FoneDigitHead, which
    reads the projected hidden state as (cos, sin) pairs per digit slot.
    """
    ang = 2 * math.pi * slot_phases(slots)
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1).flatten(-2).to(torch.float32)


# === input-side embedding modules ===
class FoneInputEmbed(nn.Module):
    """Projects fixed FoNE features into the model dim; added to the <NUM> embedding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(FONE_DIM, d_model, bias=False)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:  # (N, 15) -> (N, d)
        return self.proj(fone_features(slots))


class LearnedFreqInputEmbed(nn.Module):
    """Learned-frequency variant: per-period frequency multipliers on the exact phases.

    The naive TabFM-style form cos(2π f_k x) with raw x is numerically ill-posed here:
    x spans 10 orders of magnitude, so any float error in a learned f_k gets amplified
    by x and scrambles the phase entirely (verified by test; TabFM operates on
    normalized tabular floats, not exact 10-digit integers). Instead we keep the
    digit-exact phases p_k in [0,1) and learn a multiplier c_k on each dial:
    feature_k = (cos 2π c_k p_k, sin 2π c_k p_k), c_k init 1.0 so at init this equals
    the fixed variant exactly and can only depart by learning.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.freq_mult = nn.Parameter(torch.ones(N_PERIODS))
        self.proj = nn.Linear(FONE_DIM, d_model, bias=False)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:  # (N, 15) -> (N, d)
        ang = 2 * math.pi * self.freq_mult.to(torch.float64) * slot_phases(slots)
        feats = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1).flatten(-2).to(torch.float32)
        return self.proj(feats)


# === output-side per-digit decode head (shared by both FoNE variants) ===
class FoneDigitHead(nn.Module):
    """Per-digit classification via dot products with base-10 Fourier prototypes.

    h -> Linear -> 30 dims read as (cos, sin) pairs per slot; logits for digit j of
    slot i are <(h_2i, h_2i+1), (cos 2πj/10, sin 2πj/10)> -- the paper's decode rule
    with a learned projection in front.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, FONE_DIM, bias=False)
        ang = 2 * math.pi * torch.arange(10) / 10.0
        self.register_buffer("proto", torch.stack([torch.cos(ang), torch.sin(ang)]))  # (2, 10)

    def digit_logits(self, h: torch.Tensor) -> torch.Tensor:  # (N, d) -> (N, 15, 10)
        pairs = self.proj(h).view(*h.shape[:-1], N_PERIODS, 2)
        return torch.einsum("...sc,cd->...sd", pairs, self.proto)

    def loss(self, h: torch.Tensor, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean per-digit CE over all 15 slots (zero-padded design), plus digit accuracy."""
        logits = self.digit_logits(h)                              # (N, 15, 10)
        loss = F.cross_entropy(logits.reshape(-1, 10), slots.reshape(-1).long())
        acc = (logits.argmax(-1) == slots).float().mean()
        return loss, acc

    @torch.no_grad()
    def decode(self, h: torch.Tensor) -> torch.Tensor:  # (N, d) -> (N, 15) predicted slots
        return self.digit_logits(h).argmax(-1)
