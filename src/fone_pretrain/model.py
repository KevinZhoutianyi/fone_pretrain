"""nanoGPT-style Llama decoder with chunk-based FoNE number embedding.

Architecture: RMSNorm, RoPE, SwiGLU, no biases, tied input/output embeddings.
The only novelty is the effective weight matrix: if `embed_mode` is a FoNE variant,
the first F dimensions of pure-digit chunk tokens (values 0..999) carry a Fourier
code of the chunk's value; all other dimensions stay learned. Because input embedding
and output projection share one weight, injecting the code makes both the read side
(lookup) and the write side (logits) see the same code. There is no <NUM> token, no
digit sidecar, and no output-side decode head -- ordinary next-token cross-entropy
over the vocab supervises numbers.
"""

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .number_embed import ChunkFreqCode


@dataclass
class ModelConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    d_model: int
    d_ff: int
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    embed_mode: str = "baseline"   # baseline | fone | fone_learned
    n_learned_freq: int = 20       # extra learnable periods for fone_learned


# === building blocks ===
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cache(seq_len: int, head_dim: int, theta: float, device) -> torch.Tensor:
    """Returns (seq_len, head_dim/2) complex rotations for RoPE."""
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    t = torch.arange(seq_len, device=device).float()
    return torch.polar(torch.ones(seq_len, head_dim // 2, device=device), torch.outer(t, inv_freq))


def apply_rope(x: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    # x: (B, n_head, T, head_dim) -> rotate pairs as complex numbers
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * rot).flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.d_model // cfg.n_head
        self.wqkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, rot):
        B, T, C = x.shape
        q, k, v = self.wqkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, rot[:T]), apply_rope(k, rot[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)   # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)   # up
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.ln2, self.mlp = RMSNorm(cfg.d_model), SwiGLU(cfg)

    def forward(self, x, rot):
        x = x + self.attn(self.ln1(x), rot)
        return x + self.mlp(self.ln2(x))


# === the model ===
class FonePretrainModel(nn.Module):
    def __init__(self, cfg: ModelConfig,
                 is_number_token: torch.Tensor | None = None,
                 token_value: torch.Tensor | None = None):
        """is_number_token/token_value are (vocab,) maps from build_number_token_maps;
        required for fone modes, unused for baseline."""
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.d_model)
        # output is tied to tok_emb: forward() projects with effective_weight() (the
        # tied table with the FoNE code injected), so no separate lm_head is needed.

        if cfg.embed_mode != "baseline":
            assert is_number_token is not None and token_value is not None, \
                "fone modes need is_number_token/token_value"
            self.num_code = ChunkFreqCode(is_number_token, token_value,
                                          learned=(cfg.embed_mode == "fone_learned"),
                                          n_learned_freq=cfg.n_learned_freq)
            assert self.num_code.n_dims <= cfg.d_model, \
                f"FoNE code needs {self.num_code.n_dims} dims but d_model={cfg.d_model}"

        self.apply(self._init)
        n_params = sum(p.numel() for p in self.parameters())
        if os.environ.get("RANK", "0") == "0":  # once, not per DDP rank
            print(f"model: {cfg.n_layer}L/{cfg.d_model}d/{cfg.n_head}h  embed_mode={cfg.embed_mode}  params={n_params/1e6:.1f}M")

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def effective_weight(self) -> torch.Tensor:
        """The tied weight with the Fourier code written into number-chunk rows.

        For each number-chunk token, the first F dims are replaced by its code (a
        function of the periods) and the remaining dims keep the learned values. Used
        for BOTH the input lookup and the output logits, so reading and generating a
        number share the same code. The overwritten entries of tok_emb.weight receive
        no gradient (they are never used), so they are effectively frozen; the learned
        tail and the learnable periods (fone_learned) train normally.
        """
        W = self.tok_emb.weight
        if self.cfg.embed_mode == "baseline":
            return W
        F_ = self.num_code.n_dims
        ids = self.num_code.num_ids
        code = self.num_code().to(W.dtype)                       # (n_num, F)
        rows = torch.cat([code, W[ids, F_:]], dim=1)             # (n_num, d_model)
        return W.index_copy(0, ids, rows)                        # out-of-place, autograd-safe

    def hidden(self, idx, W):
        rot = rope_cache(self.cfg.max_seq_len, self.cfg.d_model // self.cfg.n_head,
                         self.cfg.rope_theta, idx.device)
        x = F.embedding(idx, W)
        for blk in self.blocks:
            x = blk(x, rot)
        return self.ln_f(x)

    def forward(self, idx, targets=None):
        """With targets: LM loss dict (DDP-safe: loss goes through forward so the reducer
        sees it). Without: final hidden states.

        idx/targets: (B, T) shifted-by-one token ids (numbers are ordinary chunk tokens).
        """
        W = self.effective_weight()
        if targets is None:
            return self.hidden(idx, W)
        h = self.hidden(idx, W)
        logits = F.linear(h, W)                                   # tied output, same code
        lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),
                                  ignore_index=-100)
        return {"lm_loss": lm_loss, "loss": lm_loss}
