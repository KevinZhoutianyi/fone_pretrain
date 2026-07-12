#!/bin/bash
# Launch the chunk-FoNE design sweep: 9 runs (3 headline + 6 ablations), one 8xH100
# node each, all on the shared mix3b_llama3 dataset. Pass --smoke to run each as a
# 20-step smoke first (do this before the real wave).
#
# Usage:
#   bash scripts/launch_sweep.sh --smoke   # 9 quick smokes, ~5 min each
#   bash scripts/launch_sweep.sh           # the real 9-run wave, ~3B tokens each
#
# The design questions each run answers are in experiments/chunk_fone/README.md.
set -euo pipefail
cd /fsx/zhouty/fone_pretrain

SMOKE="${1:-}"
EXCLUDE="--exclude=ip-10-4-120-250"   # node with a stray GPU-hogging process (see tracking.md)

CONFIGS=(
  llama3_baseline               # 1. control
  llama3_fone                   # 2. fixed FoNE, 3 periods (6d)
  llama3_fone_learned           # 3. learned freq, 23 periods (46d)
  sweep_fone_12d                # 4. more fixed dims (6 periods, 12d)
  sweep_fone_learned_43         # 5. more learned freqs (43 periods, 86d)
  sweep_fone_3p_learnfreq       # 6. learn freq at equal dim to #2 (isolates learn-freq)
  sweep_fone_fixedscale         # 7. fone with scale frozen (isolates learnable scale)
  sweep_fone_learned_fixedscale # 8. learned with scale frozen
  sweep_fone_seed2              # 9. variance band on #2 (seed 2024)
)

for cfg in "${CONFIGS[@]}"; do
  sbatch --job-name="fone_${cfg}" $EXCLUDE scripts/train.sbatch "configs/${cfg}.yaml" $SMOKE
done
echo "submitted ${#CONFIGS[@]} runs${SMOKE:+ (smoke)}"
squeue -u "$USER"
