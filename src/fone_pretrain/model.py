"""nanoGPT-style Llama decoder with pluggable number embedding.

Architecture: RMSNorm, RoPE, SwiGLU, no biases, tied input/output embeddings.
The only novelty is in embed()/loss(): if `embed_mode` is a FoNE variant, <NUM>
positions get a Fourier value embedding added on the input side, and positions
whose *target* is <NUM> get a per-digit loss from FoneDigitHead on the output side.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .number_embed import FoneDigitHead, FoneInputEmbed, LearnedFreqInputEmbed


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
    num_token_id: int = -1         # id of <NUM>; required for fone modes
    num_loss_weight: float = 1.0   # λ on the per-digit loss


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
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied

        if cfg.embed_mode == "fone":
            self.num_in = FoneInputEmbed(cfg.d_model)
        elif cfg.embed_mode == "fone_learned":
            self.num_in = LearnedFreqInputEmbed(cfg.d_model)
        if cfg.embed_mode != "baseline":
            assert cfg.num_token_id >= 0, "fone modes need num_token_id"
            self.num_head = FoneDigitHead(cfg.d_model)

        self.apply(self._init)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"model: {cfg.n_layer}L/{cfg.d_model}d/{cfg.n_head}h  embed_mode={cfg.embed_mode}  params={n_params/1e6:.1f}M")

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def embed(self, idx: torch.Tensor, num_slots: torch.Tensor | None) -> torch.Tensor:
        """Token embeddings; at <NUM> positions add the projected Fourier value embedding.
        num_slots: (B, T, 15) digit slots, zeros at non-number positions.
        """
        x = self.tok_emb(idx)
        if self.cfg.embed_mode != "baseline":
            mask = idx == self.cfg.num_token_id                      # (B, T)
            if mask.any():
                x = x.clone()
                x[mask] = x[mask] + self.num_in(num_slots[mask]).to(x.dtype)
        return x

    def hidden(self, idx, num_slots=None):
        rot = rope_cache(self.cfg.max_seq_len, self.cfg.d_model // self.cfg.n_head,
                         self.cfg.rope_theta, idx.device)
        x = self.embed(idx, num_slots)
        for blk in self.blocks:
            x = blk(x, rot)
        return self.ln_f(x)

    def forward(self, idx, targets=None, num_slots=None, target_slots=None):
        """With targets: combined pretraining loss dict (DDP-safe: loss goes through
        forward so the reducer sees it). Without: final hidden states.

        idx/targets: (B, T) shifted-by-one token ids (numbers already as <NUM>).
        num_slots:   (B, T, 15) digits for <NUM> tokens in the *input*.
        target_slots:(B, T, 15) digits for positions whose *target* is <NUM>.
        """
        if targets is None:
            return self.hidden(idx, num_slots)
        h = self.hidden(idx, num_slots)
        logits = self.lm_head(h)
        lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),
                                  ignore_index=-100)
        out = {"lm_loss": lm_loss, "num_loss": torch.zeros_like(lm_loss),
               "digit_acc": torch.ones((), device=idx.device)}
        if self.cfg.embed_mode != "baseline":
            tmask = targets == self.cfg.num_token_id
            if tmask.any():
                num_loss, digit_acc = self.num_head.loss(h[tmask], target_slots[tmask])
                out["num_loss"], out["digit_acc"] = num_loss, digit_acc
        out["loss"] = out["lm_loss"] + self.cfg.num_loss_weight * out["num_loss"]
        return out
