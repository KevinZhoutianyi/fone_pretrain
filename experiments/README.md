# Experiments

Each numbered folder is a self-contained experiment with its own `run.py`, results, and logs.

| # | Name | Status | Description |
|---|------|--------|-------------|
| 01 | 125m_three_variant | running smoke | pipeline validation: 3 variants, 3B tokens, 8xH100 each |

## Convention
- Create new numbered folders (`01_xxx/`, `02_xxx/`, ...) for new experiments — don't edit old ones.
- Each folder contains: `run.py`, `args/`, `ckpts/`, `figures/`, and `README.md` (observations).
- This README should only contain brief descriptions of each experiment. Detailed setup, results, and observations belong in each experiment's own `README.md`.
- Large outputs (checkpoints, datasets) go in `/data/fone_pretrain/`, not here.
