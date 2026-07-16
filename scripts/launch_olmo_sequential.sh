#!/bin/bash
# Sequential OLMo-2 stage-2 wave: 2 arms (baseline/fone) x 2 seeds (1337, 2024) = 4 runs,
# 4 nodes each but ONE AT A TIME (resources are tight). Each run is chained on the prior
# via --dependency=afterany, so only 4 nodes are ever occupied. Each run's GSM8K eval is
# chained on its own training job.
#
# Usage: bash scripts/launch_olmo_sequential.sh [FIRST_DEP_JOBID]
#   FIRST_DEP_JOBID: if given, the first queued run waits for that job (e.g. the already
#   -running baseline_s1337) before starting. Omit to start the chain immediately.
set -euo pipefail
cd /fsx/zhouty/fone_pretrain
CKPT=/fsx/zhouty/data/fone_pretrain/checkpoints

# runs to queue, in order. baseline_s1337 is assumed already running (passed as dep).
RUNS=(olmo_baseline_s2024 olmo_fone_s1337 olmo_fone_s2024)

prev="${1:-}"   # job id the first run waits on (the currently-running baseline_s1337)
for run in "${RUNS[@]}"; do
  cfg="configs/olmo_stage2/${run}.yaml"
  dep=""
  [ -n "$prev" ] && dep="--dependency=afterany:$prev"
  tjid=$(sbatch $dep --job-name="$run" scripts/train_olmo_4node.sbatch "$cfg" 2>/dev/null | grep -oE "[0-9]+$")
  # GSM8K eval for this run, after its own training completes
  sbatch --job-name="ev_${run}" --dependency=afterok:$tjid \
    scripts/eval_gsm8k.sbatch "$CKPT/$run/ckpt_final.pt" "$run" >/dev/null 2>&1
  echo "queued $run as $tjid (waits for ${prev:-nothing})"
  prev=$tjid
done
echo "=== queue ==="
squeue -u "$USER" -h -o "%.6i %.9T %.20j %R" | grep olmo
