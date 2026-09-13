#!/usr/bin/env bash
# The three policy-B static orders on Goodreads validation, then the two test evaluations.
# Sequential: one heavy job at a time (HARD RULE 11d). combined FIRST -- items and concepts
# transplant their seed pass from its search heap.
set -u
cd /c/dev/phd/casper
export OMP_NUM_THREADS=4 CASPER_DATASET=goodreads
L=experiments/goodreads
for ch in combined items concepts; do
  echo "=== $(date +%H:%M:%S) policyB order: $ch ==="
  python -u src/instrument/greedy_static_policyB_channels.py --channel $ch \
      >> $L/policyB_${ch}.out 2>&1 || { echo "FAILED: $ch"; exit 1; }
done
echo "=== $(date +%H:%M:%S) ALL THREE ORDERS DONE ==="
