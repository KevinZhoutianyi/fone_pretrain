#!/bin/bash
# Multi-node (2x8=16 GPU) OLMo-2 stage-2 surgery run, ~2x throughput vs single node.
# Usage: sbatch scripts/train_olmo_2node.sbatch configs/olmo_stage2/fone_s1337.yaml [--smoke]
#
# Global batch must stay 512 sequences x 4096 = the official 2,097,152 tok/step. With
# 16 GPUs, use bs 2 x ga 16 x 16 = 512 sequences (config's grad_accum is overridden to
# 16 via GA env below so tokens/step is invariant to node count).
#SBATCH --job-name=olmo2n
#SBATCH --partition=dev
#SBATCH --nodes=2
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

# EFA/NCCL for fast cross-node RDMA (this cluster exposes provider=efa)
export FI_PROVIDER=efa
export NCCL_DEBUG=WARN

RUN_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['run_dir'])")
S3_DEST=s3://tianyizhoubucket/fone_pretrain/$(basename "$RUN_DIR")
mkdir -p "$RUN_DIR" logs

# S3 sync only from the head node (rank-0 node writes checkpoints)
( while true; do sleep 600; aws s3 sync "$RUN_DIR" "$S3_DEST" --region us-west-2 --no-progress || true; done ) &
SYNC_PID=$!
cleanup() { kill "$SYNC_PID" 2>/dev/null || true
            pkill -P $$ 2>/dev/null || true
            aws s3 sync "$RUN_DIR" "$S3_DEST" --region us-west-2 --no-progress || true; }
trap cleanup EXIT TERM INT

# rendezvous: head node's hostname, from the SLURM nodelist
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR
export MASTER_PORT=29501
echo "=== job $SLURM_JOB_ID: $SLURM_NNODES nodes, master $MASTER_ADDR, $CONFIG ${EXTRA[*]} ==="

# one torchrun per node (srun fans out to both), 8 procs each, GA halved to keep
# global batch = 512 sequences (2 nodes -> grad_accum 16 instead of the 1-node 32).
srun --ntasks=2 --ntasks-per-node=1 bash -c '
  torchrun --nnodes=2 --nproc_per_node=8 \
    --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    -m fone_pretrain.train_olmo '"$CONFIG"' --grad-accum 16 '"${EXTRA[*]}"'
'
