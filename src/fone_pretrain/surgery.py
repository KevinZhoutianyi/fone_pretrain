"""FoNE embedding surgery on a pretrained (untied) HF causal LM, e.g. OLMo-2-1B.

For each pure-digit chunk token (0..999, found by build_number_token_maps), the row of
BOTH the input embedding and the lm_head is rebuilt as three segments:

    [ 20-dim FoNE code | 32 glue dims | remaining dims FROZEN at pretrained values ]
      cos/sin(2pi v/T_k)  pretrained
      10 periods T_k      values,
      LEARNABLE           trainable

  - code dims REPLACE the 20 lowest-variance dims (across number rows) of that matrix;
    glue = the next 32 lowest-variance dims, kept at pretrained values but trainable.
    Low-variance dims carry the least information, so the surgery destroys the minimum.
  - each matrix gets its own learnable code scale, initialized to match the RMS of the
    dims it replaces (measured on the checkpoint: the input embedding and lm_head of
    OLMo-2-1B differ by ~9x in scale, so a shared or unit scale would blow up logits).
  - everything outside the number rows trains normally; within a number row, only the
    code parameters (periods, scale) and the glue dims receive gradient.

Arms:
  baseline      -- no surgery at all (official stage-2 replica)
  unfreeze_ctrl -- same 52-dim trainable / rest-frozen pattern, but NO code: separates
                   "the code helps" from "the freeze pattern helps"
  fone          -- the full surgery above

Implementation: the pretrained weight tensor is kept as a frozen buffer; trainable
parts (glue values, code periods/scale, and for baseline-arm compatibility the whole
matrix) are separate Parameters, and an effective weight is assembled each forward via
index_copy (same static-shape pattern as model.py's effective_weight). Freezing is by
construction: frozen segments live only in the buffer, so they receive no gradient.
"""

import math

import torch
import torch.nn as nn

from .number_embed import ChunkFreqCode

N_CODE_PERIODS = 10          # 20 code dims
N_GLUE = 32                  # trainable glue dims


def pick_surgery_dims(weight: torch.Tensor, num_ids: torch.Tensor,
                      n_code: int = 2 * N_CODE_PERIODS, n_glue: int = N_GLUE):
    """Choose which dims to replace (code) and which to unfreeze (glue).

    Ranks dims by variance across the number rows; the n_code lowest-variance dims are
    replaced by the code, the next n_glue become glue. Returns (code_dims, glue_dims),
    each a sorted LongTensor of column indices.
    """
    rows = weight[num_ids]                       # (n_num, d)
    order = rows.var(dim=0).argsort()            # low variance first
    code_dims = order[:n_code].sort().values
    glue_dims = order[n_code:n_code + n_glue].sort().values
    return code_dims, glue_dims


class SurgeredMatrix(nn.Module):
    """One weight matrix (input embedding OR lm_head) with FoNE code overwrite (arm A).

    The full matrix is a normal trainable Parameter (`main`); everything trains as usual
    EXCEPT the code_dims of number rows, which are overwritten each forward with the FoNE
    code. Those overwritten entries are never read from `main`, so they get zero gradient
    (frozen by construction); all other dims of number rows and all non-number rows train
    normally. Thus the model is identical to the untouched pretrained model except the 20
    code dims of number rows carry the learnable-frequency Fourier code -- the only
    difference from the baseline arm.

    Trainable: main (full matrix, code dims of number rows excepted) + num_code
    (ChunkFreqCode periods + per-matrix scale).
    """

    def __init__(self, weight: torch.Tensor, is_number_token: torch.Tensor,
                 token_value: torch.Tensor,
                 n_periods: int = N_CODE_PERIODS, n_glue: int = N_GLUE):
        super().__init__()
        num_ids = torch.nonzero(is_number_token, as_tuple=False).squeeze(-1)
        self.register_buffer("num_ids", num_ids)
        code_dims, _ = pick_surgery_dims(weight, num_ids, n_code=2 * n_periods, n_glue=n_glue)
        self.register_buffer("code_dims", code_dims)

        # scale init: match the RMS of the dims the code replaces; cos/sin RMS = 1/sqrt(2)
        replaced_rms = weight[num_ids][:, code_dims].pow(2).mean().sqrt().item()
        scale_init = replaced_rms * math.sqrt(2.0)

        self.main = nn.Parameter(weight.clone())
        self.num_code = ChunkFreqCode(is_number_token, token_value,
                                      n_periods=n_periods, learnable_freq=True,
                                      learnable_scale=True, scale_init=scale_init)

    def effective_weight(self) -> torch.Tensor:
        """Overwrite the code_dims of number rows with the FoNE code; the rest of `main`
        (all other dims of number rows, all non-number rows) trains normally."""
        rows = self.main[self.num_ids].clone()                        # (n_num, d) trained rows
        rows[:, self.code_dims] = self.num_code().to(rows.dtype)      # (n_num, 20) code
        return self.main.index_copy(0, self.num_ids, rows)


class SurgeredLM(nn.Module):
    """Wraps an HF causal LM (untied embeddings). arm="fone" overwrites the 20 code dims
    of number rows on both the input embedding and the lm_head with the learnable-freq
    FoNE code; all other weights train normally. arm="baseline" wraps without touching
    anything (faithful stage-2 replica). The only fone-vs-baseline difference is the code.

    forward(idx, targets) mirrors our FonePretrainModel loss interface so train code and
    PackedDataset batches work unchanged.
    """

    def __init__(self, hf_model, tokenizer, arm: str,
                 n_periods: int = N_CODE_PERIODS, n_glue: int = N_GLUE):
        super().__init__()
        from .number_embed import build_number_token_maps
        self.lm = hf_model
        self.arm = arm
        if arm != "baseline":
            is_num, value = build_number_token_maps(tokenizer)
            vocab = self.lm.get_input_embeddings().weight.shape[0]
            if len(is_num) < vocab:   # config vocab (padded) > tokenizer vocab
                pad = vocab - len(is_num)
                is_num = torch.cat([is_num, torch.zeros(pad, dtype=torch.bool)])
                value = torch.cat([value, torch.zeros(pad, dtype=torch.long)])
            emb_w = self.lm.get_input_embeddings().weight.data
            head_w = self.lm.get_output_embeddings().weight.data
            self.emb_surgery = SurgeredMatrix(emb_w, is_num, value, n_periods, n_glue)
            self.head_surgery = SurgeredMatrix(head_w, is_num, value, n_periods, n_glue)
            # the HF module's own embedding/lm_head weights are replaced per-forward by
            # the surgery `main` params; drop the originals so they are not double-trained.
            self.lm.get_input_embeddings().weight.requires_grad_(False)
            self.lm.get_output_embeddings().weight.requires_grad_(False)

    def forward(self, idx, targets=None, z_loss_mult=0.0):
        import torch.nn.functional as F
        if self.arm == "baseline":
            out = self.lm(input_ids=idx)
            logits = out.logits
        else:
            W_in = self.emb_surgery.effective_weight()
            W_out = self.head_surgery.effective_weight()
            x = F.embedding(idx, W_in)
            # run the transformer body on inputs_embeds; project with surgered head
            body = self.lm.model(inputs_embeds=x)
            logits = F.linear(body.last_hidden_state, W_out)
        if targets is None:
            return logits
        flat = logits.view(-1, logits.size(-1)).float()
        lm_loss = F.cross_entropy(flat, targets.reshape(-1), ignore_index=-100)
        out = {"lm_loss": lm_loss, "loss": lm_loss}
        if z_loss_mult:  # OLMo-2 softmax auxiliary (z-)loss: penalize logsumexp drift
            mask = (targets.reshape(-1) != -100)
            z = torch.logsumexp(flat, dim=-1)[mask]
            z_loss = z_loss_mult * (z ** 2).mean()
            out["z_loss"] = z_loss
            out["loss"] = lm_loss + z_loss
        return out
