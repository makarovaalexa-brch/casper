#!/usr/bin/env bash
# M3 v2b -- EXTEND the banked v1 tree under stopping rule v2, then grow the six teachers.
# The cold tree is NOT rebuilt: coldtree_v2_partial.json was seeded from ndcgtree_partial.json
# (22 nodes, depth 5) with all 15 leaves re-decided from their recorded child_sizes, and the 22
# CELF heaps copied over. Only the children of the 12 newly-splitting leaves need searching.
# total_L=15 throughout, matching the v1 node table.
set -u
cd /c/dev/phd/casper || exit 1
PY="C:/Program Files/Python312/python.exe"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6

echo "=================================================================="
echo "[v2b] $(date '+%H:%M:%S') EXTENDING the cold tree under rule v2 (resumes 22 banked nodes)"
echo "=================================================================="
"$PY" src/instrument/ndcg_tree.py --label coldtree_v2 --total_L 15 --full_threads
echo "[v2b] $(date '+%H:%M:%S') cold tree exited with status $?"

for spec in "arthouse:68" "dialogue:275" "marvel:581" "gore:417" "teenagers:922" "inspiring:503"; do
  name="${spec%%:*}"; cid="${spec##*:}"
  echo "=================================================================="
  echo "[v2b] $(date '+%H:%M:%S') growing teacher '${name}' (concept ${cid})"
  echo "=================================================================="
  "$PY" src/instrument/ndcg_tree.py --label "m3v2_${name}" --cohort_concept "${cid}" \
        --total_L 15 --full_threads
  echo "[v2b] $(date '+%H:%M:%S') teacher '${name}' exited with status $?"
done
echo "[v2b] ALL DONE $(date '+%H:%M:%S')"
