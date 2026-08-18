#!/usr/bin/env bash
# M3 under STOPPING RULE v2 -- regrow BOTH sides so the M5 comparison stays symmetric.
#
#   1. the COLD TREE (incumbent), 10,000 validation users, no seed  -> label coldtree_v2
#   2. the six FOREST TEACHERS, anchor-seeded cohorts               -> label m3v2_<name>
#
# Both use the identical recipe; the ONLY difference is the cohort and the free seeded anchor.
# Rule v2: split iff >=2 children are non-empty and >=1 reaches min_leaf; children below the floor
# become leaves inheriting the parent's tail, so nothing is ever fitted below the floor.
# Resumable: contract-signed partials + per-node CELF heaps.
set -u
cd /c/dev/phd/casper || exit 1
PY="C:/Program Files/Python312/python.exe"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6

echo "=================================================================="
echo "[m3v2] $(date '+%H:%M:%S') regrowing the COLD TREE under rule v2 (the incumbent)"
echo "=================================================================="
"$PY" src/instrument/ndcg_tree.py --label coldtree_v2 --total_L 16 --full_threads
st=$?; echo "[m3v2] $(date '+%H:%M:%S') cold tree exited with status $st"

for spec in "arthouse:68" "dialogue:275" "marvel:581" "gore:417" "teenagers:922" "inspiring:503"; do
  name="${spec%%:*}"; cid="${spec##*:}"
  echo "=================================================================="
  echo "[m3v2] $(date '+%H:%M:%S') growing teacher '${name}' (concept ${cid})"
  echo "=================================================================="
  "$PY" src/instrument/ndcg_tree.py \
      --label "m3v2_${name}" \
      --cohort_concept "${cid}" \
      --total_L 16 \
      --full_threads
  st=$?; echo "[m3v2] $(date '+%H:%M:%S') teacher '${name}' exited with status $st"
done
echo "[m3v2] ALL DONE $(date '+%H:%M:%S')"
