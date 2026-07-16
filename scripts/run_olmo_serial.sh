#!/bin/bash
# Drive the OLMo-2 stage-2 runs strictly serially WITHOUT slurm dependencies: wait for
# the current training job to leave the queue, then submit the next one (plus its GSM8K
# eval). This is more controllable than --dependency chains (a preempted/failed link
# won't silently strand the rest, and we can see each launch happen).
#
# Usage: nohup bash scripts/run_olmo_serial.sh [FIRST_RUNNING_JOBID] > logs/olmo_serial.log 2>&1 &
#   FIRST_RUNNING_JOBID: a job already running to wait on before the first queued run
#   (e.g. baseline_s1337). Omit to submit the first run immediately.
set -uo pipefail
cd /fsx/zhouty/fone_pretrain
CKPT=/fsx/zhouty/data/fone_pretrain/checkpoints

# 2 arms x 2 seeds; baseline_s1337 is already running (passed as $1), so queue the rest.
RUNS=(olmo_baseline_s2024 olmo_fone_s1337 olmo_fone_s2024)

wait_done () {  # block until slurm job $1 leaves the queue
  local jid=$1
  [ -z "$jid" ] && return
  echo "[$(date '+%m-%d %H:%M')] waiting for job $jid to finish ..."
  while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 120; done
  echo "[$(date '+%m-%d %H:%M')] job $jid done"
}

launch () {  # submit train for run $1, then its eval on afterok; echo the train jobid
  local run=$1
  local tjid
  tjid=$(sbatch --job-name="$run" scripts/train_olmo_4node.sbatch "configs/olmo_stage2/${run}.yaml" 2>/dev/null | grep -oE "[0-9]+$")
  sbatch --job-name="ev_${run}" --dependency=afterok:$tjid \
    scripts/eval_gsm8k.sbatch "$CKPT/$run/ckpt_final.pt" "$run" >/dev/null 2>&1
  echo "[$(date '+%m-%d %H:%M')] launched $run as train job $tjid (+ chained eval)"
  echo "$tjid"
}

wait_done "${1:-}"
for run in "${RUNS[@]}"; do
  tjid=$(launch "$run" | tail -1)
  wait_done "$tjid"
done
echo "[$(date '+%m-%d %H:%M')] all runs submitted and finished"
