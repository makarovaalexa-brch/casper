#!/usr/bin/env bash
# M3 -- grow the six forest teachers, SEQUENTIALLY (one heavy job on the box at a time).
# Design sheet: docs/design/DS_M3_FOREST_TREES.md
# Each tree: cohort = validation users who STATE the anchor; anchor folded free at the root and
# masked from the pool; 16 questions, same pool/branch-rule/min-leaf as the cold tree.
# Resumable: ndcg_tree.py persists a contract-signed partial + per-node heaps, so a kill costs at
# most the lazy re-evals since the last pick.
set -u
cd /c/dev/phd/casper || exit 1
PY="C:/Program Files/Python312/python.exe"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6

# anchor:concept-id, smallest cohort first so a failure surfaces cheaply
for spec in "arthouse:68" "dialogue:275" "marvel:581" "gore:417" "teenagers:922" "inspiring:503"; do
  name="${spec%%:*}"; cid="${spec##*:}"
  echo "=================================================================="
  echo "[m3] $(date '+%H:%M:%S') growing teacher '${name}' (concept ${cid})"
  echo "=================================================================="
  "$PY" src/instrument/ndcg_tree.py \
      --label "m3_${name}" \
      --cohort_concept "${cid}" \
      --total_L 16 \
      --full_threads
  echo "[m3] $(date '+%H:%M:%S') teacher '${name}' exited with status $?"
done
echo "[m3] ALL TEACHERS DONE $(date '+%H:%M:%S')"
