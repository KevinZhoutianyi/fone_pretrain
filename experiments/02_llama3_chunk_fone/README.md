# 02 — Llama-3 chunk-based FoNE, 125M three-variant

**Goal (paper.md §1).** Show the chunk-based FoNE pipeline trains stably: LM loss
converges for all three variants, and the Fourier code injected into number-chunk
tokens does not destabilize training.

## What changed from experiment 01

Experiment 01 collapsed each whole number into one `<NUM>` token with a 15-digit
sidecar and an output-side digit head. That design is retired. The new design keeps
numbers as the tokenizer's own pure-digit chunks (up to 3 digits, values 0-999) and
writes a Fourier code of the chunk's face value into the first F dimensions of that
token's embedding row. The other dimensions stay learned. No `<NUM>` token, no sidecar,
no digit head, no auxiliary loss.

Because the model ties its input embedding and output projection to one weight matrix,
the code is injected into that matrix (an effective weight recomputed each step) and
used on BOTH sides: the same code serves when the model reads a number (lookup) and
when it generates one (logits). The overwritten entries never receive gradient, so they
are effectively frozen; the learned tail of each row and, for the learned variant, the
periods train normally.

## Setup

| knob | value |
|---|---|
| variants | baseline / fone / fone_learned (`configs/llama3_*.yaml`) |
| tokenizer | `NousResearch/Meta-Llama-3-8B` (ungated mirror), vocab 128256, native <=3-digit chunking |
| model | 12L / 768d / 12h, SwiGLU d_ff 2048, seq 2048; ~124M transformer + ~98M tied 128k embedding |
| data | mix3b_llama3: FineWeb-Edu 70% + FineMath-4plus 30%; ONE dataset shared by all three variants |
| variants | fone: 3 fixed periods (10/100/1000), F=6 dims, frozen. fone_learned: 3 base + 20 extra learnable periods (log-uniform in [10,1000]), F=46 dims, all periods trainable |
| budget | 6000 steps × 524k tokens = ~3.1B tokens |
| optimizer | AdamW lr 6e-4, cosine to 10%, warmup 120, bf16, grad clip 1.0; log_periods (fone_learned) excluded from weight decay |
| hardware | one 8×H100 node per variant, DDP |

All three variants read the identical token stream: tokenization does not depend on the
variant, only the model's embedding does. This is a cleaner control than experiment 01,
where baseline and FoNE consumed different tokenizations.

The code encodes face value only: for a chunk value v and period T the code is
(cos 2πv/T, sin 2πv/T). A chunk's place in a larger number (ones vs thousands) is left
to token position and attention, not the embedding. The fixed variant's 3 periods
already separate all 1000 values; the learned variant's 20 extra periods let it discover
its own resolution, and the two differ in dimension (6 vs 46) by design, so they do NOT
start from the same code (the old exp-01 "identical at init" property no longer holds).

The 128k Llama-3 vocab is stored as uint32 (past uint16's 65535 ceiling) and makes the
tied embedding table ~98M params, so total parameter count is larger than 01's TinyLlama
runs. The three variants are still matched to each other.

Init-loss caveat: injecting the code makes the number rows of the effective weight
larger in norm than the 0.02-scale learned rows, so at step 0 the FoNE variants show a
higher LM loss than baseline (fone > baseline, fone_learned highest with 46 dims). This
is a transient starting-point effect, not divergence; the § 1 check is that it falls and
converges, not that it starts low.

## How to run

```bash
# data (login node, CPU): one dataset for all three variants
uv run python scripts/prepare_data.py --docs <N> --out /fsx/zhouty/data/fone_pretrain/datasets/mix3b_llama3

# smoke then formal (inside an 8xH100 allocation):
sbatch scripts/train.sbatch configs/llama3_baseline.yaml
sbatch scripts/train.sbatch configs/llama3_fone.yaml
sbatch scripts/train.sbatch configs/llama3_fone_learned.yaml
# eval after training:
srun --gres=gpu:1 ... python -m fone_pretrain.eval_numbers \
    /fsx/zhouty/data/fone_pretrain/checkpoints/llama3_fone/ckpt_final.pt --out results_fone.json
```

## Status / observations

| check | result |
|---|---|
| unit tests (token map, code injectivity over 0-999, fixed/learned period setup, uint32 round-trip) | 17/17 green |
| read == write share the code (tied weight) | verified: number row of input embedding equals output-projection row equals the code |
| overwritten code dims get zero gradient (frozen); learned tail + log_periods train | verified at d_model=768 |
| CPU forward/backward, all 3 variants | pass |
| log_periods decay-exclusion finds it by name (`num_code.log_periods`) | verified |
| data prep | not yet run |
| 20-step smoke on GPU | not yet run |
