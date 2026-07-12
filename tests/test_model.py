"""The load-bearing property of chunk-based FoNE: because input embedding and output
projection are tied, injecting the Fourier code into the effective weight makes both
the READ side (embedding) and the WRITE side (logits) use the same code. Also checks
baseline is untouched, all three variants run, and the overwritten weight entries stay
frozen while the learned tail and periods train.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fone_pretrain.model import FonePretrainModel, ModelConfig  # noqa: E402
from fone_pretrain.number_embed import build_number_token_maps, freq_code  # noqa: E402


class FakeTokenizer:
    def __init__(self, vocab): self.vocab = vocab
    def __len__(self): return len(self.vocab)
    def decode(self, ids): return self.vocab[ids[0]]


VOCAB = ["<eos>", "the", "1", "2", "42", "123", "999", "100", "7"] + [f"w{i}" for i in range(8)]


def _model(mode, **kw):
    is_num, value = build_number_token_maps(FakeTokenizer(VOCAB))
    cfg = ModelConfig(vocab_size=len(VOCAB), n_layer=2, n_head=2, d_model=64, d_ff=128,
                      max_seq_len=32, embed_mode=mode, **kw)
    torch.manual_seed(0)
    return FonePretrainModel(cfg, is_num, value), is_num, value


def test_effective_weight_injects_code_into_number_rows():
    m, is_num, value = _model("fone")
    W = m.effective_weight()
    F_ = m.num_code.n_dims
    for tid in torch.nonzero(is_num).squeeze(-1).tolist():
        expect = freq_code(value[tid:tid + 1], m.num_code.log_periods)[0]
        assert torch.allclose(W[tid, :F_], expect, atol=1e-5), tid
    # a non-number row is untouched (equals the raw table)
    non = int(torch.nonzero(~is_num).squeeze(-1)[0])
    assert torch.allclose(W[non], m.tok_emb.weight[non])


def test_read_and_write_share_the_code():
    # the whole point: input embedding row == output projection row == code, via tie.
    m, is_num, _ = _model("fone")
    W = m.effective_weight()
    tid = int(torch.nonzero(is_num).squeeze(-1)[0])
    read = torch.nn.functional.embedding(torch.tensor([tid]), W)[0]  # input side
    write = W[tid]                                                   # output side (h @ W.T uses this row)
    assert torch.equal(read, write)


def test_baseline_weight_unchanged():
    m, _, _ = _model("baseline")
    assert torch.equal(m.effective_weight(), m.tok_emb.weight)
    assert not hasattr(m, "num_code")


def test_all_variants_forward_backward():
    idx = torch.randint(0, len(VOCAB), (2, 8))
    tgt = torch.randint(0, len(VOCAB), (2, 8))
    for mode in ["baseline", "fone", "fone_learned"]:
        m, _, _ = _model(mode)
        out = m(idx, targets=tgt)
        assert set(out) == {"lm_loss", "loss"}
        out["loss"].backward()   # must not raise


def test_overwritten_dims_get_no_gradient():
    # the first F dims of number rows are replaced in the effective weight, so those
    # entries of tok_emb.weight are never used -> zero gradient -> effectively frozen.
    m, is_num, _ = _model("fone")
    idx = torch.randint(0, len(VOCAB), (2, 8))
    m(idx, targets=idx.clone())["loss"].backward()
    F_ = m.num_code.n_dims
    tid = int(torch.nonzero(is_num).squeeze(-1)[0])
    g = m.tok_emb.weight.grad
    assert torch.allclose(g[tid, :F_], torch.zeros(F_)), "code dims of number rows must be frozen"
    # the learned tail of the same row DOES receive gradient (it is used)
    assert g[tid, F_:].abs().sum() > 0


def test_learned_periods_train():
    m, _, _ = _model("fone_learned", n_learned_freq=20)
    idx = torch.randint(0, len(VOCAB), (2, 8))
    m(idx, targets=idx.clone())["loss"].backward()
    lp = m.num_code.log_periods
    assert lp.grad is not None and lp.grad.abs().sum() > 0
