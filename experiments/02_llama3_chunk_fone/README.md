# 02 — Llama-3 chunk-based FoNE, 125M three-variant

**Goal (paper.md §1).** Show the chunk-based FoNE pipeline trains stably: LM loss
converges for all three variants, and the fixed 6-dim Fourier code injected into
number-chunk tokens does not destabilize training.

## What changed from experiment 01

Experiment 01 collapsed each whole number into one `<NUM>` token with a 15-digit
sidecar and an output-side digit head. That design is retired. The new design keeps
numbers as the tokenizer's own pure-digit chunks (up to 3 digits, values 0-999) and,
on the input side only, overwrites the first 6 dimensions of each number-chunk token's
embedding with a fixed Fourier code of the chunk's face value. The other dimensions
stay learned. No `<NUM>` token, no sidecar, no digit head, no auxiliary loss.

## Setup

| knob | value |
|---|---|
| variants | baseline / fone / fone_learned (`configs/llama3_*.yaml`) |
| tokenizer | `NousResearch/Meta-Llama-3-8B` (ungated mirror), vocab 128256, native <=3-digit chunking |
| model | 12L / 768d / 12h, SwiGLU d_ff 2048, seq 2048; ~124M transformer + ~98M tied 128k embedding |
| data | mix3b_llama3: FineWeb-Edu 70% + FineMath-4plus 30%; ONE dataset shared by all three variants |
| budget | 6000 steps × 524k tokens = ~3.1B tokens |
| optimizer | AdamW lr 6e-4, cosine to 10%, warmup 120, bf16, grad clip 1.0; freq_mult excluded from weight decay |
| hardware | one 8×H100 node per variant, DDP |

All three variants read the identical token stream: tokenization does not depend on the
variant, only the model's embedding does. This is a cleaner control than experiment 01,
where baseline and FoNE consumed different tokenizations.

The fixed 6-dim code encodes face value only: periods 10, 100, 1000 pin the three digits
of a chunk. A chunk's place in a larger number (ones vs thousands) is left to token
position and attention, not the embedding.

The 128k Llama-3 vocab is stored as uint32 (past uint16's 65535 ceiling) and makes the
tied embedding table ~98M params, so total parameter count is larger than 01's TinyLlama
runs. The three variants are still matched to each other.

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
| unit tests (token map, 6-dim code injectivity, learned-init parity, uint32 round-trip) | 11/11 green |
| CPU forward/backward, all 3 variants | pass; fone == fone_learned loss bit-identical at init |
| freq_mult gradient flows; decay-exclusion finds it by name | verified (grad norm ~0.05, 1 param in no-decay group) |
| data prep | not yet run |
| 20-step smoke on GPU | not yet run |
