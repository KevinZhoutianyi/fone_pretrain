#!/bin/bash
# compare-only eval (add/sub are ~0 at this scale, skip). Usage: eval_compare.sbatch CKPT NAME
#SBATCH --job-name=evcmp
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=0:30:00
#SBATCH --exclude=ip-10-4-120-250,ip-10-4-102-38
#SBATCH --output=/fsx/zhouty/fone_pretrain/logs/%x-%j.out
set -euo pipefail
CKPT=$1
NAME=$2
cd /fsx/zhouty/fone_pretrain
source .venv/bin/activate
export HF_HOME=/fsx/zhouty/data/hf_cache
srun --ntasks=1 python -m fone_pretrain.eval_numbers "$CKPT" \
  --tasks compare --digits 2,4,6,8,10 --n 200 \
  --out /fsx/zhouty/fone_pretrain/outputs/eval/${NAME}.json
