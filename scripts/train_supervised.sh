#!/usr/bin/env bash
# Auto-resuming training supervisor. Runs a training command; if the process
# exits before reaching the target epoch (RSS-limit safety stop, or any other
# crash), automatically relaunches with --resume-from the latest periodic
# checkpoint. Loops until the run actually reaches its final epoch.
#
# Usage: scripts/train_supervised.sh <run_name> <target_epochs> -- <train.py args...>
# Example:
#   scripts/train_supervised.sh mlp_full_fullres_v4 400 -- \
#     --mode real --model mlp --task full --epochs 400 --full-resolution \
#     --sims-per-batch 4 --hidden 256,256,256,256,256 --rss-limit-gb 40 --checkpoint-every 10
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

RUN_NAME="$1"
TARGET_EPOCH=$(( $2 - 1 ))
shift 2
if [ "$1" != "--" ]; then
  echo "expected -- before train.py args" >&2
  exit 1
fi
shift

RESUME_CKPT="checkpoints/${RUN_NAME}_resume.pt"
LOSSES_CSV="runs/${RUN_NAME}/losses.csv"

while true; do
  ARGS=("$@" "--run-name" "$RUN_NAME")
  if [ -f "$RESUME_CKPT" ]; then
    echo "$(date): resuming from $RESUME_CKPT"
    ARGS+=("--resume-from" "$RESUME_CKPT")
  else
    echo "$(date): starting fresh (no resume checkpoint found yet)"
  fi

  python3 -m src.train "${ARGS[@]}"
  EXIT_CODE=$?

  LAST_EPOCH=$(tail -1 "$LOSSES_CSV" 2>/dev/null | cut -d, -f1)
  echo "$(date): training process exited (code $EXIT_CODE), last logged epoch: ${LAST_EPOCH:-none}"

  if [ "$LAST_EPOCH" == "$TARGET_EPOCH" ]; then
    echo "$(date): reached target epoch $TARGET_EPOCH, done."
    break
  fi

  echo "$(date): relaunching in 10s..."
  sleep 10
done
