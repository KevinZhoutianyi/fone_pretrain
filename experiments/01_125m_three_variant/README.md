# 01 — 125M three-variant pipeline validation

**Goal (paper.md §1).** Show the FoNE pretraining pipeline trains stably: mixed
LM + per-digit loss converges for all three variants and digit accuracy rises
well above chance (0.1).

## Setup

| knob | value |
|---|---|
| variants | baseline / fone / fone_learned (`configs/125m_*.yaml`) |
| model | 12L / 768d / 12h, SwiGLU d_ff 2048, seq 2048, ~124M params + 25M tied emb |
| data | mix3b: FineWeb-Edu 70% + FineMath-4plus 30%, TinyLlama 32k tokenizer |
| budget | 6000 steps × 524k tokens = ~3.1B tokens, identical for every variant |
| optimizer | AdamW lr 6e-4, cosine to 10%, warmup 120, bf16, grad clip 1.0 |
| hardware | one 8×H100 node per variant, DDP |

Baseline consumes the `mix3b_baseline` tokenization (numbers digit-split by the
tokenizer); both FoNE variants consume `mix3b_fone` (in-range numbers as one
`<NUM>` token + digit sidecar). Equal token budget means FoNE sees slightly more
text per step; docs-seen is logged for the record.

## How to run

```bash
sbatch scripts/train.sbatch configs/125m_baseline.yaml
sbatch scripts/train.sbatch configs/125m_fone.yaml
sbatch scripts/train.sbatch configs/125m_fone_learned.yaml
# eval after training:
srun --gres=gpu:1 ... python -m fone_pretrain.eval_numbers \
    /fsx/zhouty/data/fone_pretrain/checkpoints/125m_fone/ckpt_final.pt --out results_fone.json
```

## Status / observations

(pending — smoke test in progress)
