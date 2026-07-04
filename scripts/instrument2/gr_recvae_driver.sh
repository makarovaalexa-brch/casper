#!/usr/bin/env bash
# Chunked RESUME driver for gr_recvae.py: trains one latent-d to early-stop via
# bounded <=9.5-min chunks (honours the "chunked <=600000ms, RESUME" discipline),
# then the final chunk runs the TEST eval and writes {tag}_TEST.json.
# Usage: bash gr_recvae_driver.sh <latent> <nsub> <batch>
set -e
D=${1:-256}; NSUB=${2:-40000}; BATCH=${3:-1000}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
TESTJSON=".cache/instrument2/gr_recvae_d${D}_TEST.json"
CKPT=".cache/instrument2/gr_recvae_d${D}.pt"
# first chunk: fresh (no --resume) unless a ckpt already exists
FIRST="--resume"; [ -f "$CKPT" ] || FIRST=""
i=0
while [ ! -f "$TESTJSON" ]; do
  i=$((i+1))
  echo "=== [driver d=$D] chunk $i ($([ -z "$FIRST" ] && echo fresh || echo resume)) ==="
  python scripts/instrument2/gr_recvae.py --latent "$D" --nsub "$NSUB" --batch "$BATCH" \
      --epochs 60 --patience 8 --max_minutes 9.3 $FIRST
  FIRST="--resume"
  # if the run early-stopped, it wrote TEST.json and we exit; else loop resumes.
done
echo "=== [driver d=$D] COMPLETE -> $TESTJSON ==="
