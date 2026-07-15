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
    """One weight matrix (input embedding OR lm_head) with FoNE row surgery.

    arm:
      "fone"          -- code + glue trainable, rest of number rows frozen
      "unfreeze_ctrl" -- code_dims + glue dims trainable AT PRETRAINED VALUES (no code),
                         rest of number rows frozen
    (the baseline arm never constructs this module)

    Trainable parameters:
      main       -- full-matrix Parameter for all NON-number rows (trains normally)
      glue       -- (n_num, n_glue) pretrained values of the glue dims
      patch      -- (n_num, n_code) pretrained values of the code dims (unfreeze_ctrl only)
      num_code   -- ChunkFreqCode periods + scale (fone only)
    The frozen remainder of number rows lives in the `frozen` buffer.
    """

    def __init__(self, weight: torch.Tensor, is_number_token: torch.Tensor,
                 token_value: torch.Tensor, arm: str):
        super().__init__()
        assert arm in ("fone", "unfreeze_ctrl")
        self.arm = arm
        num_ids = torch.nonzero(is_number_token, as_tuple=False).squeeze(-1)
        self.register_buffer("num_ids", num_ids)
        code_dims, glue_dims = pick_surgery_dims(weight, num_ids)
        self.register_buffer("code_dims", code_dims)
        self.register_buffer("glue_dims", glue_dims)

        # scale init: match the RMS of the dims the code replaces; cos/sin RMS = 1/sqrt(2)
        replaced_rms = weight[num_ids][:, code_dims].pow(2).mean().sqrt().item()
        scale_init = replaced_rms * math.sqrt(2.0)

        # the full matrix trains normally for non-number rows; number rows in `main`
        # are overwritten by index_copy each forward, so they never contribute and
        # receive zero gradient (frozen by construction, like model.py).
        self.main = nn.Parameter(weight.clone())
        self.register_buffer("frozen", weight[num_ids].clone())   # (n_num, d) pretrained rows
        self.glue = nn.Parameter(weight[num_ids][:, glue_dims].clone())
        if arm == "fone":
            self.num_code = ChunkFreqCode(is_number_token, token_value,
                                          n_periods=N_CODE_PERIODS, learnable_freq=True,
                                          learnable_scale=True, scale_init=scale_init)
        else:  # unfreeze_ctrl: trainable pretrained values where the code would go
            self.patch = nn.Parameter(weight[num_ids][:, code_dims].clone())

    def effective_weight(self) -> torch.Tensor:
        """Assemble [code | glue | frozen] number rows into the full matrix."""
        rows = self.frozen.clone()                                    # (n_num, d) frozen base
        rows[:, self.glue_dims] = self.glue
        if self.arm == "fone":
            rows[:, self.code_dims] = self.num_code().to(rows.dtype)  # (n_num, 20)
        else:
            rows[:, self.code_dims] = self.patch
        return self.main.index_copy(0, self.num_ids, rows)


class SurgeredLM(nn.Module):
    """Wraps an HF causal LM (untied embeddings) with row surgery on both the input
    embedding and the lm_head. arm="baseline" wraps without touching anything.

    forward(idx, targets) mirrors our FonePretrainModel loss interface so train code
    and PackedDataset batches work unchanged.
    """

    def __init__(self, hf_model, tokenizer, arm: str):
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
            self.emb_surgery = SurgeredMatrix(emb_w, is_num, value, arm)
            self.head_surgery = SurgeredMatrix(head_w, is_num, value, arm)
            # the HF module's own embedding/lm_head weights are replaced per-forward;
            # drop their Parameters so they are not trained or double-counted.
            self.lm.get_input_embeddings().weight.requires_grad_(False)
            self.lm.get_output_embeddings().weight.requires_grad_(False)

    def forward(self, idx, targets=None):
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
        lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                  targets.reshape(-1), ignore_index=-100)
        return {"lm_loss": lm_loss, "loss": lm_loss}
