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
    """One weight matrix (input embedding OR lm_head), surgered per `mode`.

    The full matrix is a normal trainable Parameter (`main`). What happens to the number
    rows depends on mode; non-number rows always come from `main` and train normally.

      mode="fone"        -- the code_dims of number rows are overwritten each forward with
        the FoNE code; all OTHER dims of number rows still train (from `main`). The model
        is the untouched pretrained model except the code dims carry the Fourier code.
        Problem: the free non-code dims let the model keep its pretrained number identity
        and ignore the code (observed: code scale decays toward 0).

      mode="fone_forced" -- every number row is a FROZEN shared base (the mean over
        pretrained number rows) with the code_dims overwritten by the FoNE code. Two
        number tokens differ ONLY in the code, so the model cannot distinguish numbers
        without reading it. This is the "force the model to use FoNE" arm.

      mode="mean_ctrl"   -- every number row is the frozen shared base, no code: all
        number tokens share one embedding. Control that isolates "collapse number rows to
        their shared mean" from "add the FoNE code back on top of that base".

    Freezing is by construction: in the forced/ctrl modes `main`'s number rows are never
    read, so they get zero gradient; the shared base is a buffer. Trainable in fone /
    fone_forced: main (non-number rows; plus non-code number dims in fone) + num_code.
    """

    def __init__(self, weight: torch.Tensor, is_number_token: torch.Tensor,
                 token_value: torch.Tensor,
                 n_periods: int = N_CODE_PERIODS, n_glue: int = N_GLUE, mode: str = "fone"):
        super().__init__()
        self.mode = mode
        num_ids = torch.nonzero(is_number_token, as_tuple=False).squeeze(-1)
        self.register_buffer("num_ids", num_ids)
        code_dims, _ = pick_surgery_dims(weight, num_ids, n_code=2 * n_periods, n_glue=n_glue)
        self.register_buffer("code_dims", code_dims)

        self.main = nn.Parameter(weight.clone())

        if mode in ("fone_forced", "mean_ctrl"):
            # shared, frozen base for every number row: the mean over pretrained number
            # rows. Keeps the embedding norm/direction in-distribution while erasing the
            # per-token number identity, which then lives only in the code (or nowhere).
            self.register_buffer("base_row", weight[num_ids].mean(dim=0))   # (d,)

        if mode != "mean_ctrl":
            # scale init: match the RMS of the dims the code replaces; cos/sin RMS = 1/sqrt(2)
            replaced_rms = weight[num_ids][:, code_dims].pow(2).mean().sqrt().item()
            scale_init = replaced_rms * math.sqrt(2.0)
            self.num_code = ChunkFreqCode(is_number_token, token_value,
                                          n_periods=n_periods, learnable_freq=True,
                                          learnable_scale=True, scale_init=scale_init)

    def effective_weight(self) -> torch.Tensor:
        """Assemble the number rows per mode; non-number rows come from `main` unchanged."""
        if self.mode == "fone":
            rows = self.main[self.num_ids].clone()                    # (n_num, d) trained rows
            rows[:, self.code_dims] = self.num_code().to(rows.dtype)  # code
        else:
            # fone_forced / mean_ctrl: frozen shared base for every number row
            rows = self.base_row.unsqueeze(0).expand(self.num_ids.numel(), -1).clone()
            if self.mode == "fone_forced":
                rows[:, self.code_dims] = self.num_code().to(rows.dtype)
        return self.main.index_copy(0, self.num_ids, rows)


class SurgeredLM(nn.Module):
    """Wraps an HF causal LM (untied embeddings). The arm selects the number-row surgery:

      baseline      -- no surgery (faithful stage-2 replica).
      fone          -- code overwrites the code_dims of number rows on BOTH the input
                       embedding and the lm_head; other number dims still train, so the
                       model can route around the code.
      fone_forced   -- INPUT embedding only: number rows are a frozen shared mean + FoNE
                       code, so the code is the ONLY thing distinguishing numbers on input,
                       forcing the model to read it. The lm_head trains normally (surgering
                       the output projection too would collapse every number token's output
                       direction to the same vector and cripple number generation).
      mean_ctrl     -- INPUT embedding only: number rows are the frozen shared mean, no
                       code (control for fone_forced: isolates the code from the
                       mean-collapse). The lm_head trains normally.

    So fone surgers both matrices; fone_forced / mean_ctrl surger only the input embedding
    and leave the lm_head as an ordinary trainable weight, identical across the two arms
    and the baseline. forward(idx, targets) mirrors our FonePretrainModel loss interface
    so train code and PackedDataset batches work unchanged.
    """

    def __init__(self, hf_model, tokenizer, arm: str,
                 n_periods: int = N_CODE_PERIODS, n_glue: int = N_GLUE):
        super().__init__()
        from .number_embed import build_number_token_maps
        self.lm = hf_model
        self.arm = arm
        # fone ties input+output surgery; the input-forcing arms surger the embedding only.
        self.head_surgery = None
        if arm != "baseline":
            is_num, value = build_number_token_maps(tokenizer)
            vocab = self.lm.get_input_embeddings().weight.shape[0]
            if len(is_num) < vocab:   # config vocab (padded) > tokenizer vocab
                pad = vocab - len(is_num)
                is_num = torch.cat([is_num, torch.zeros(pad, dtype=torch.bool)])
                value = torch.cat([value, torch.zeros(pad, dtype=torch.long)])
            emb_w = self.lm.get_input_embeddings().weight.data
            self.emb_surgery = SurgeredMatrix(emb_w, is_num, value, n_periods, n_glue, mode=arm)
            # the HF module's input embedding is replaced per-forward by emb_surgery.main;
            # drop the original so it is not double-trained.
            self.lm.get_input_embeddings().weight.requires_grad_(False)
            if arm == "fone":
                head_w = self.lm.get_output_embeddings().weight.data
                self.head_surgery = SurgeredMatrix(head_w, is_num, value, n_periods, n_glue, mode=arm)
                self.lm.get_output_embeddings().weight.requires_grad_(False)
            # fone_forced / mean_ctrl: lm_head stays a normal trainable HF weight.

    def forward(self, idx, targets=None, z_loss_mult=0.0):
        import torch.nn.functional as F
        if self.arm == "baseline":
            out = self.lm(input_ids=idx)
            logits = out.logits
        else:
            W_in = self.emb_surgery.effective_weight()
            # fone: surgered head; forced/mean_ctrl: ordinary (trainable) lm_head weight.
            W_out = (self.head_surgery.effective_weight() if self.head_surgery is not None
                     else self.lm.get_output_embeddings().weight)
            x = F.embedding(idx, W_in)
            # run the transformer body on inputs_embeds; project with the head weight
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
