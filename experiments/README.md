# Experiments

Each folder is a self-contained experiment with its own `README.md` (setup, observations).

| Name | Status | Description |
|------|--------|-------------|
| chunk_fone | data prep running | chunk-based FoNE: 3 variants (baseline / fone / fone_learned), 125M, ~3B tokens, one 8xH100 node each |

## Convention
- Configs live in `configs/`, code in `src/fone_pretrain/`; the folder holds the write-up.
- This README lists experiments briefly; detailed setup and results live in each folder's `README.md`.
- Large outputs (checkpoints, datasets) go in `/fsx/zhouty/data/fone_pretrain/`, not here.
