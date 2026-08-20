#!/usr/bin/env bash
# Refit every group on 2,000 TRAIN users with MIN-LEAF stopping and no depth cap (author, 2026-08-20),
# matching the cold tree's rule. arthouse is already running; this picks up the other five.
# Each artifact is tagged by fitting condition so nothing can clobber a validation-fitted tree.
set -u
cd /c/dev/phd/casper || exit 1
PY="C:/Program Files/Python312/python.exe"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
# wait for the arthouse fit to finish before starting -- one heavy job at a time (HARD RULE 11d)
for i in $(seq 1 720); do
  grep -q "^\[.*\] \[done\]" experiments/battery/m6k_arthouse_tr2k_v2.out && break
  tasklist 2>/dev/null | grep -qi python.exe || break
  sleep 60
done
echo "[all2k] arthouse finished at $(date '+%H:%M:%S'); starting the rest"
OK=""; BAD=""
for g in marvel gore dialogue teenagers inspiring; do
  echo "=================================================================="
  echo "[all2k] START ${g} at $(date '+%H:%M:%S')"
  "$PY" src/instrument/m6k_cluster_tree.py --i_have_author_goahead --threads 6 \
        --cluster "${g}" --split train --slice_hi 34000 --n_users 2000 \
        --depth 6 --min_leaf 200 > "experiments/battery/m6k_${g}_tr2k.out" 2>&1
  st=$?
  echo "[all2k] ${g} exited ${st} at $(date '+%H:%M:%S')"
  if [ "${st}" -ne 0 ]; then BAD="${BAD} ${g}"; else OK="${OK} ${g}"; fi
  grep -E "\[done\]" "experiments/battery/m6k_${g}_tr2k.out" | tail -1
done
echo "[all2k] DONE $(date '+%H:%M:%S')  OK:${OK:-none}  FAILED:${BAD:-none}"
