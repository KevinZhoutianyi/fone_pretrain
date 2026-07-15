"""Correctness gate for the OLMo surgery module (surgery.py) before any GPU spend.

Uses a small fake pretrained model (same interface as HF: get_input_embeddings /
get_output_embeddings / model(inputs_embeds=...)) so no download is needed.

Properties under test:
  - fone arm touches ONLY the 20 code dims of number rows (glue + rest = pretrained)
  - unfreeze_ctrl arm is bit-identical to pretrained at init
  - code scale is matched to the replaced dims' RMS (no magnitude blow-up), per matrix
  - gradients: frozen segments get none; glue/periods/scale do; non-number rows do
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fone_pretrain.number_embed import build_number_token_maps  # noqa: E402
from fone_pretrain.surgery import N_GLUE, SurgeredMatrix, pick_surgery_dims  # noqa: E402


class FakeTokenizer:
    def __init__(self, vocab): self.vocab = vocab
    def __len__(self): return len(self.vocab)
    def decode(self, ids): return self.vocab[ids[0]]


VOCAB = ["<eos>", "the", "1", "42", "123", "999", "100", "7"] + [f"w{i}" for i in range(56)]
D = 128  # d_model for tests (>= 52 surgery dims)


def _setup(arm, scale_mult=1.0):
    torch.manual_seed(0)
    weight = torch.randn(len(VOCAB), D) * 0.1 * scale_mult
    is_num, value = build_number_token_maps(FakeTokenizer(VOCAB))
    sm = SurgeredMatrix(weight, is_num, value, arm)
    ids = torch.nonzero(is_num).squeeze(-1)
    return weight, sm, ids


def test_fone_touches_only_code_dims():
    weight, sm, ids = _setup("fone")
    W = sm.effective_weight()
    # non-number rows identical
    non = [i for i in range(len(VOCAB)) if i not in ids.tolist()]
    assert torch.equal(W[non], weight[non])
    # number rows: everything except code dims equals pretrained
    mask = torch.ones(D, dtype=torch.bool)
    mask[sm.code_dims] = False
    assert torch.allclose(W[ids][:, mask], weight[ids][:, mask])
    # code dims differ from pretrained (the code is there)
    assert not torch.allclose(W[ids][:, sm.code_dims], weight[ids][:, sm.code_dims])


def test_unfreeze_ctrl_identity_at_init():
    weight, sm, ids = _setup("unfreeze_ctrl")
    assert torch.allclose(sm.effective_weight(), weight)


def test_scale_matched_to_replaced_dims():
    # the injected code's RMS must match the RMS of the dims it replaced, per matrix
    for mult in [1.0, 9.0]:   # OLMo's emb vs lm_head differ ~9x; both must self-match
        weight, sm, ids = _setup("fone", scale_mult=mult)
        replaced_rms = weight[ids][:, sm.code_dims].pow(2).mean().sqrt()
        W = sm.effective_weight()
        code_rms = W[ids][:, sm.code_dims].pow(2).mean().sqrt()
        ratio = (code_rms / replaced_rms).item()
        assert 0.5 < ratio < 2.0, f"code RMS {code_rms:.4f} vs replaced {replaced_rms:.4f} (x{mult})"


def test_gradient_routing_fone():
    weight, sm, ids = _setup("fone")
    W = sm.effective_weight()
    W.sum().backward()
    # frozen buffer has no grad by construction; main's number rows got zero grad
    assert sm.main.grad is not None
    assert torch.allclose(sm.main.grad[ids], torch.zeros_like(sm.main.grad[ids]))
    # non-number rows of main DO get grad
    non = [i for i in range(len(VOCAB)) if i not in ids.tolist()]
    assert sm.main.grad[non].abs().sum() > 0
    # glue, periods, scale all flow
    assert sm.glue.grad is not None and sm.glue.grad.abs().sum() > 0
    assert sm.num_code.log_periods.grad is not None
    assert sm.num_code.scale.grad is not None


def test_gradient_routing_ctrl():
    weight, sm, ids = _setup("unfreeze_ctrl")
    sm.effective_weight().sum().backward()
    assert sm.patch.grad is not None and sm.patch.grad.abs().sum() > 0
    assert sm.glue.grad is not None and sm.glue.grad.abs().sum() > 0
    assert torch.allclose(sm.main.grad[ids], torch.zeros_like(sm.main.grad[ids]))


def test_dim_selection_is_low_variance():
    weight, _, _ = _setup("fone")
    is_num, value = build_number_token_maps(FakeTokenizer(VOCAB))
    ids = torch.nonzero(is_num).squeeze(-1)
    code_dims, glue_dims = pick_surgery_dims(weight, ids)
    assert len(code_dims) == 20 and len(glue_dims) == N_GLUE
    assert set(code_dims.tolist()).isdisjoint(set(glue_dims.tolist()))
    var = weight[ids].var(dim=0)
    # every chosen code dim has variance <= every non-chosen, non-glue dim
    chosen = set(code_dims.tolist()) | set(glue_dims.tolist())
    rest_min = min(var[i].item() for i in range(D) if i not in chosen)
    assert var[code_dims].max().item() <= rest_min + 1e-9
