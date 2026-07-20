#!/bin/bash
# Usage: sbatch scripts/eval_gsm8k.sbatch <ckpt-or-hf> <name> [--hf]
#   ckpt mode:  sbatch scripts/eval_gsm8k.sbatch /path/ckpt_final.pt fone_s1337
#   hf mode:    sbatch scripts/eval_gsm8k.sbatch allenai/OLMo-2-0425-1B official --hf
#SBATCH --job-name=gsm8k
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=1:30:00
#SBATCH --exclude=ip-10-4-120-250,ip-10-4-102-38
#SBATCH --output=/fsx/zhouty/fone_pretrain/logs/%x-%j.out
set -euo pipefail
TARGET=$1
NAME=$2
MODE=${3:-}
N=${GSM8K_N:-200}  # override with GSM8K_N=1319 for the full test set
cd /fsx/zhouty/fone_pretrain
source .venv/bin/activate
export HF_HOME=/fsx/zhouty/data/hf_cache
OUT=/fsx/zhouty/fone_pretrain/outputs/eval/gsm8k_${NAME}.json
if [ "$MODE" = "--hf" ]; then
  srun --ntasks=1 python scripts/eval_gsm8k.py --hf "$TARGET" --n "$N" --out "$OUT"
else
  srun --ntasks=1 python scripts/eval_gsm8k.py --ckpt "$TARGET" --n "$N" --out "$OUT"
fi
