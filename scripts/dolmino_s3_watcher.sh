#!/bin/bash
DATA=/fsx/zhouty/data/fone_pretrain/datasets/dolmino50b
DEST=s3://tianyizhoubucket/fone_pretrain/datasets/dolmino50b
echo "[$(date '+%H:%M:%S')] waiting for $DATA/manifest.json ..."
while [ ! -f "$DATA/manifest.json" ]; do sleep 120; done
echo "[$(date '+%H:%M:%S')] manifest ready, syncing dataset to S3"
aws s3 sync "$DATA" "$DEST" --region us-west-2 --no-progress
echo "[$(date '+%H:%M:%S')] dolmino50b synced to $DEST"
