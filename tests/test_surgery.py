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
from fone_pretrain.surgery import SurgeredMatrix, pick_surgery_dims  # noqa: E402


class FakeTokenizer:
    def __init__(self, vocab): self.vocab = vocab
    def __len__(self): return len(self.vocab)
    def decode(self, ids): return self.vocab[ids[0]]


VOCAB = ["<eos>", "the", "1", "42", "123", "999", "100", "7"] + [f"w{i}" for i in range(56)]
D = 128  # d_model for tests (>= 20 code dims)


def _setup(scale_mult=1.0, mode="fone"):
    torch.manual_seed(0)
    weight = torch.randn(len(VOCAB), D) * 0.1 * scale_mult
    is_num, value = build_number_token_maps(FakeTokenizer(VOCAB))
    sm = SurgeredMatrix(weight, is_num, value, mode=mode)
    ids = torch.nonzero(is_num).squeeze(-1)
    return weight, sm, ids


def test_fone_touches_only_code_dims_at_init():
    # arm A: only the 20 code dims of number rows differ from pretrained; everything
    # else (other dims of number rows, all non-number rows) is unchanged at init.
    weight, sm, ids = _setup()
    W = sm.effective_weight()
    non = [i for i in range(len(VOCAB)) if i not in ids.tolist()]
    assert torch.equal(W[non], weight[non])                       # non-number rows untouched
    mask = torch.ones(D, dtype=torch.bool); mask[sm.code_dims] = False
    assert torch.allclose(W[ids][:, mask], weight[ids][:, mask])  # non-code dims untouched
    assert not torch.allclose(W[ids][:, sm.code_dims], weight[ids][:, sm.code_dims])  # code present


def test_scale_matched_to_replaced_dims():
    # the injected code's RMS must match the RMS of the dims it replaced, per matrix
    for mult in [1.0, 9.0]:   # OLMo's emb vs lm_head differ ~9x; both must self-match
        weight, sm, ids = _setup(scale_mult=mult)
        replaced_rms = weight[ids][:, sm.code_dims].pow(2).mean().sqrt()
        code_rms = sm.effective_weight()[ids][:, sm.code_dims].pow(2).mean().sqrt()
        ratio = (code_rms / replaced_rms).item()
        assert 0.5 < ratio < 2.0, f"code RMS {code_rms:.4f} vs replaced {replaced_rms:.4f} (x{mult})"


def test_gradient_routing_fone():
    weight, sm, ids = _setup()
    sm.effective_weight().sum().backward()
    # code dims of number rows are overwritten -> never read -> zero grad (frozen)
    assert sm.main.grad is not None
    assert torch.allclose(sm.main.grad[ids][:, sm.code_dims],
                          torch.zeros_like(sm.main.grad[ids][:, sm.code_dims]))
    # NON-code dims of number rows DO train (arm A: rest of the row is not frozen)
    mask = torch.ones(D, dtype=torch.bool); mask[sm.code_dims] = False
    assert sm.main.grad[ids][:, mask].abs().sum() > 0
    # non-number rows train
    non = [i for i in range(len(VOCAB)) if i not in ids.tolist()]
    assert sm.main.grad[non].abs().sum() > 0
    # code periods + scale flow
    assert sm.num_code.log_periods.grad is not None and sm.num_code.log_periods.grad.abs().sum() > 0
    assert sm.num_code.scale.grad is not None


def test_dim_selection_is_low_variance():
    weight, _, _ = _setup()
    is_num, value = build_number_token_maps(FakeTokenizer(VOCAB))
    ids = torch.nonzero(is_num).squeeze(-1)
    code_dims, _ = pick_surgery_dims(weight, ids)
    assert len(code_dims) == 20
    var = weight[ids].var(dim=0)
    rest_min = min(var[i].item() for i in range(D) if i not in set(code_dims.tolist()))
    assert var[code_dims].max().item() <= rest_min + 1e-9


def test_config_driven_bandwidth():
    # high-bandwidth variant: n_periods/n_glue are configurable; the code must occupy
    # exactly 2*n_periods dims and still touch only those dims of number rows at init.
    torch.manual_seed(0)
    weight = torch.randn(len(VOCAB), D) * 0.1
    is_num, value = build_number_token_maps(FakeTokenizer(VOCAB))
    sm = SurgeredMatrix(weight, is_num, value, n_periods=23, n_glue=32)
    ids = torch.nonzero(is_num).squeeze(-1)
    assert len(sm.code_dims) == 46                                    # 2 * 23
    assert sm.num_code().shape[1] == 46
    W = sm.effective_weight()
    non = [i for i in range(len(VOCAB)) if i not in ids.tolist()]
    assert torch.equal(W[non], weight[non])                          # non-number rows untouched
    mask = torch.ones(D, dtype=torch.bool); mask[sm.code_dims] = False
    assert torch.allclose(W[ids][:, mask], weight[ids][:, mask])     # non-code dims untouched


def test_fone_forced_numbers_differ_only_in_code():
    # forced arm: every number row is the frozen shared mean + code, so two number rows
    # differ ONLY in the code dims -- the model must read the code to tell numbers apart.
    weight, sm, ids = _setup(mode="fone_forced")
    W = sm.effective_weight()
    non = [i for i in range(len(VOCAB)) if i not in ids.tolist()]
    assert torch.equal(W[non], weight[non])                          # non-number rows untouched
    # non-code dims are identical across ALL number rows (the shared frozen mean)
    mask = torch.ones(D, dtype=torch.bool); mask[sm.code_dims] = False
    nc = W[ids][:, mask]
    assert torch.allclose(nc, nc[0].expand_as(nc))                   # every number row shares them
    # and that shared value IS the mean of the pretrained number rows
    assert torch.allclose(nc[0], weight[ids].mean(dim=0)[mask], atol=1e-6)
    # code dims DO differ across number values (identity carried by the code)
    cd = W[ids][:, sm.code_dims]
    assert not torch.allclose(cd, cd[0].expand_as(cd))


def test_fone_forced_gradient_routing():
    # number rows are never read from `main` (they are rebuilt from the frozen base), so
    # main gets zero grad on number rows; the code periods/scale carry the number signal.
    weight, sm, ids = _setup(mode="fone_forced")
    sm.effective_weight().sum().backward()
    assert torch.allclose(sm.main.grad[ids], torch.zeros_like(sm.main.grad[ids]))  # frozen
    non = [i for i in range(len(VOCAB)) if i not in ids.tolist()]
    assert sm.main.grad[non].abs().sum() > 0                         # non-number rows train
    assert sm.num_code.log_periods.grad.abs().sum() > 0              # code trains
    assert sm.num_code.scale.grad is not None


def test_mean_ctrl_collapses_numbers_no_code():
    # control arm: every number row is the frozen shared mean, no code -> all number
    # tokens share one identical embedding, and there is no num_code module.
    weight, sm, ids = _setup(mode="mean_ctrl")
    assert not hasattr(sm, "num_code")
    W = sm.effective_weight()
    rows = W[ids]
    assert torch.allclose(rows, rows[0].expand_as(rows))             # all number rows identical
    assert torch.allclose(rows[0], weight[ids].mean(dim=0), atol=1e-6)
    non = [i for i in range(len(VOCAB)) if i not in ids.tolist()]
    assert torch.equal(W[non], weight[non])                          # non-number rows untouched
