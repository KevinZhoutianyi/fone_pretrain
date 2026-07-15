# OLMo-2-1B Stage-2 FoNE embedding surgery (GSM8K)

**Goal (paper.md §3).** Test whether a FoNE Fourier code injected into the number-token
embeddings of a pretrained open model improves math capability (GSM8K), measured at the
training stage where that capability is actually formed.

## Why this design

Our from-scratch line (125M/350M) shows only a weak, non-significant FoNE edge on
number-magnitude comparison and 0% on arithmetic (scale floor). Training a from-scratch
model to GSM8K-capable scale is months of compute. Instead we stand on a fully open,
reproducible model: OLMo-2-1B (AI2 publishes data, code, hparams, and every checkpoint).

Verified fact: OLMo-2-1B scores GSM8K **3.3 after the full 4T-token stage 1** and
**43.8 after just 50B tokens of the math-heavy Dolmino stage-2 mid-training**. The entire
GSM8K capability window is inside stage-2. So we replicate stage-2 twice (with and without
FoNE) and read off the effect where math ability is born, at ~1 day/run instead of months.

## The surgery

For each of the 1110 pure-digit chunk tokens (values 0-999), the row of BOTH the input
embedding and the (untied) lm_head becomes three segments:

| segment | width | trained? | init |
|---|---|---|---|
| FoNE code | 20 (10 periods, cos/sin) | periods + per-matrix scale learnable | periods 10/100/1000 + 7 log-uniform; scale = RMS of replaced dims |
| glue | 32 | trainable | pretrained values of the 32 next-lowest-variance dims |
| frozen | rest (~1996) | frozen | pretrained values |

The code REPLACES the 20 lowest-variance dims (least information destroyed). The learnable
scale is matched per matrix to the RMS of the dims it replaces (the OLMo-2 input embedding
and lm_head number rows differ ~9x in scale, 0.21 vs 0.023, so a shared/unit scale would
blow up logits). Everything outside number rows trains normally.

## Three arms x 3 seeds = 9 runs

| arm | number rows | isolates |
|---|---|---|
| baseline | untouched, fully trainable (official stage-2 replica) | reference |
| unfreeze_ctrl | same 52 dims trainable, rest of number row frozen, NO code | code vs the freeze pattern |
| fone | 20-dim code + 32 glue, rest frozen | the method |

## Setup (verified official OLMo-2-0425-1B stage-2 hparams)

| knob | value |
|---|---|
| init | OLMo-2-0425-1B stage1-step1907359-tokens4001B (4T-token stage-1 end) |
| data | Dolmino-mix-1124 official 50B mix (DCLM 47.2%, FLAN 16.6%, Math 20.8%, ...) |
| tokens | 50B (23852 steps x 2,097,152 tok/step = 512 sequences x 4096) |
| LR | 7.4487e-5, linear decay to 0, no warmup |
| optimizer | AdamW betas (0.9, 0.95) eps 1e-8, weight decay 0.1, grad clip 1.0 |
| aux | softmax auxiliary (z-)loss, multiplier 1e-5 |
| hardware | 2 nodes (16xH100) per run via torchrun over EFA, grad checkpointing; ~3 days |

## Eval

GSM8K 8-shot exact match (our own greedy decode, no vLLM) on each final model, plus the
official OLMo-2-0425-1B stage-2 base as an external reference and the 3 official
stage2-ingredient checkpoints as a "did our baseline replica land in the official band"
sanity check. Also our number suite (compare/add/sub by digit length).

Contamination note: Dolmino contains the GSM8K train split, so the absolute 43.8 is not
clean held-out; our comparison is relative (all arms share the same data), and we
supplement with an uncontaminated arithmetic suite.

## Status

| check | result |
|---|---|
| surgery unit tests (dim pick, grad routing, scale match, arm identity) | 6/6 green |
| forward parity on real checkpoint (baseline arm) | bit-identical to HF (diff 0.0) |
| single-node smoke, 3 arms | 3/3 no OOM, ~92k tok/s/node, scales stable |
| 2-node smoke (rendezvous + EFA NCCL) | OK, 0 errors |
| Dolmino 50B prep | running (~700M/50B) |
| training wave | pending data |
