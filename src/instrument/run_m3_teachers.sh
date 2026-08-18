#!/usr/bin/env bash
# M3 teachers only -- the cold tree (coldtree_v2.json, depth 11, 164 nodes, 109 leaves) is banked.
# FAIL-FAST: a crashed teacher stops the run. The previous driver reported "status 0" for every
# teacher because `$?` was expanded AFTER a $(date) substitution and therefore reported date's
# status -- three crashed runs were reported as successes.
set -u
cd /c/dev/phd/casper || exit 1
PY="C:/Program Files/Python312/python.exe"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6

for spec in "arthouse:68" "dialogue:275" "marvel:581" "gore:417" "teenagers:922" "inspiring:503"; do
  name="${spec%%:*}"; cid="${spec##*:}"
  echo "=================================================================="
  echo "[teach] $(date '+%H:%M:%S') growing teacher '${name}' (concept ${cid})"
  echo "=================================================================="
  "$PY" src/instrument/ndcg_tree.py --label "m3v2_${name}" --cohort_concept "${cid}" \
        --total_L 15 --full_threads
  st=$?
  echo "[teach] $(date '+%H:%M:%S') teacher '${name}' exited with status ${st}"
  if [ "${st}" -ne 0 ]; then
    echo "[teach] ABORTING -- '${name}' failed. Nothing downstream is trustworthy."
    exit "${st}"
  fi
done
echo "[teach] ALL TEACHERS DONE $(date '+%H:%M:%S')"
