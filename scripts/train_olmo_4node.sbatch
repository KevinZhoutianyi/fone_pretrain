#!/bin/bash
# Multi-node (4x8=32 GPU) OLMo-2 stage-2 surgery run.
# Usage: sbatch scripts/train_olmo_4node.sbatch configs/olmo_stage2/olmo_fone_s1337.yaml [--smoke]
#
# Global batch stays the official 512 sequences x 4096 = 2,097,152 tok/step:
# bs 2 x ga 8 x 32 GPU = 512 sequences (grad_accum overridden to 8 for 4 nodes).
#SBATCH --job-name=olmo4n
#SBATCH --partition=dev
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
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
export FI_PROVIDER=efa
export NCCL_DEBUG=WARN

RUN_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['run_dir'])")
S3_DEST=s3://tianyizhoubucket/fone_pretrain/$(basename "$RUN_DIR")
mkdir -p "$RUN_DIR" logs

( while true; do sleep 600; aws s3 sync "$RUN_DIR" "$S3_DEST" --region us-west-2 --no-progress || true; done ) &
SYNC_PID=$!
cleanup() { kill "$SYNC_PID" 2>/dev/null || true
            pkill -P $$ 2>/dev/null || true
            aws s3 sync "$RUN_DIR" "$S3_DEST" --region us-west-2 --no-progress || true; }
trap cleanup EXIT TERM INT

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR
export MASTER_PORT=29501
echo "=== job $SLURM_JOB_ID: $SLURM_NNODES nodes, master $MASTER_ADDR, $CONFIG ${EXTRA[*]} ==="

srun --ntasks=4 --ntasks-per-node=1 bash -c '
  torchrun --nnodes=4 --nproc_per_node=8 \
    --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    -m fone_pretrain.train_olmo '"$CONFIG"' --grad-accum 8 '"${EXTRA[*]}"'
'
