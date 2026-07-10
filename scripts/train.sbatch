#!/bin/bash
# Usage: sbatch scripts/train.sbatch configs/125m_fone.yaml [--smoke]
# One 8xH100 node per run; checkpoints auto-sync to S3 every 10 min and on exit.
#SBATCH --job-name=fone
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --output=/fsx/zhouty/fone_pretrain/logs/%x-%j.out

set -euo pipefail
CONFIG=$1
EXTRA=${2:-}
cd /fsx/zhouty/fone_pretrain
source .venv/bin/activate
export HF_HOME=/fsx/zhouty/data/hf_cache

RUN_DIR=$(python -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['run_dir'])")
S3_DEST=s3://tianyizhoubucket/fone_pretrain/$(basename "$RUN_DIR")
mkdir -p "$RUN_DIR" logs

# === background checkpoint sync to S3 (survives preemption via trap) ===
( while true; do sleep 600; aws s3 sync "$RUN_DIR" "$S3_DEST" --region us-west-2 --no-progress || true; done ) &
SYNC_PID=$!
cleanup() { kill "$SYNC_PID" 2>/dev/null || true
            aws s3 sync "$RUN_DIR" "$S3_DEST" --region us-west-2 --no-progress || true; }
trap cleanup EXIT

echo "=== job $SLURM_JOB_ID on $(hostname): $CONFIG $EXTRA ==="
# --ntasks=1 is load-bearing: without it slurm derives ntasks from cores/cpus-per-task
# and launches N parallel copies whose torchrun rendezvous collide (caught in smoke).
srun --ntasks=1 torchrun --standalone --nproc_per_node=8 -m fone_pretrain.train "$CONFIG" $EXTRA
