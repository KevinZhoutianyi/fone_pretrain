#!/bin/bash
# Usage: sbatch scripts/train_olmo.sbatch configs/olmo_stage2/fone_s1337.yaml [--smoke]
# One 8xH100 node per run; checkpoints + dataset auto-sync to S3.
#SBATCH --job-name=olmo
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --exclude=ip-10-4-120-250,ip-10-4-102-38
#SBATCH --output=/fsx/zhouty/fone_pretrain/logs/%x-%j.out

set -euo pipefail
CONFIG=$1
shift
EXTRA=("$@")
cd /fsx/zhouty/fone_pretrain
source .venv/bin/activate
export HF_HOME=/fsx/zhouty/data/hf_cache

RUN_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['run_dir'])")
S3_DEST=s3://tianyizhoubucket/fone_pretrain/$(basename "$RUN_DIR")
mkdir -p "$RUN_DIR" logs

# === background checkpoint sync to S3 (survives preemption via trap) ===
( while true; do sleep 600; aws s3 sync "$RUN_DIR" "$S3_DEST" --region us-west-2 --no-progress || true; done ) &
SYNC_PID=$!
# cleanup on EXIT and on SIGTERM/SIGINT (scancel/preemption): kill the sync loop, then
# kill any torchrun/python workers of THIS run still on the node -- torchrun can leave
# GPU-holding grandchildren when the outer script dies, turning the node into a
# slurm-idle-but-GPU-busy trap for the next job.
cleanup() { kill "$SYNC_PID" 2>/dev/null || true
            pkill -P $$ 2>/dev/null || true
            pkill -f "fone_pretrain.train_olmo $CONFIG" 2>/dev/null || true
            aws s3 sync "$RUN_DIR" "$S3_DEST" --region us-west-2 --no-progress || true; }
trap cleanup EXIT TERM INT

echo "=== job $SLURM_JOB_ID on $(hostname): $CONFIG ${EXTRA[*]} ==="
srun --ntasks=1 torchrun --standalone --nproc_per_node=8 -m fone_pretrain.train_olmo "$CONFIG" "${EXTRA[@]}"
