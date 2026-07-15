#!/bin/bash
# Launch the 9-run OLMo-2 stage-2 surgery wave (3 arms x 3 seeds), 2 nodes each, once
# the Dolmino 50B dataset is ready. Chains a GSM8K eval per run via afterok.
#
# Usage: bash scripts/launch_olmo_wave.sh          # waits for data, then submits
#        bash scripts/launch_olmo_wave.sh --now     # skip the wait (data already ready)
set -euo pipefail
cd /fsx/zhouty/fone_pretrain

DATA=/fsx/zhouty/data/fone_pretrain/datasets/dolmino50b
CKPT=/fsx/zhouty/data/fone_pretrain/checkpoints

if [ "${1:-}" != "--now" ]; then
  echo "waiting for $DATA/manifest.json (Dolmino 50B prep) ..."
  while [ ! -f "$DATA/manifest.json" ]; do sleep 60; done
  echo "dataset ready:"; cat "$DATA/manifest.json"
fi

for arm in baseline unfreeze_ctrl fone; do
  for s in 1337 2024 777; do
    run="olmo_${arm}_s${s}"
    cfg="configs/olmo_stage2/${run}.yaml"
    tjid=$(sbatch --job-name="$run" scripts/train_olmo_2node.sbatch "$cfg" 2>/dev/null | grep -oE "[0-9]+$")
    sbatch --job-name="ev_${run}" --dependency=afterok:$tjid \
      scripts/eval_gsm8k.sbatch "$CKPT/$run/ckpt_final.pt" "$run" >/dev/null 2>&1
    echo "launched $run (train $tjid + chained GSM8K eval)"
  done
done
echo "=== queue ==="
squeue -u "$USER" -h -o "%.6i %.9T %.22j %D" | grep -E "olmo_|ev_olmo" | head -20
