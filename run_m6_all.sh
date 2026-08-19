#!/usr/bin/env bash
# M6 overnight: fit an opener-state tree for every remaining group, then compare four arms.
# Cheapest cohorts first, so a failure late in the night still leaves the most groups done.
# NOT fail-fast ACROSS groups (they are independent and a night is expensive) but every exit
# status is captured accurately -- the M3 driver once reported "status 0" for three crashed runs
# because $? was expanded after a $(date) substitution.
set -u
cd /c/dev/phd/casper || exit 1
PY="C:/Program Files/Python312/python.exe"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
OK=""; BAD=""
for g in arthouse gore marvel dialogue inspiring; do
  echo "=================================================================="
  echo "[m6] START fit ${g} at $(date '+%H:%M:%S')"
  "$PY" src/instrument/m6k_cluster_tree.py --i_have_author_goahead --threads 6 \
        --cluster "${g}" --depth 1 > "experiments/battery/m6k_${g}.out" 2>&1
  st=$?
  echo "[m6] fit ${g} exited ${st} at $(date '+%H:%M:%S')"
  if [ "${st}" -ne 0 ]; then BAD="${BAD} ${g}(fit)"; continue; fi
  "$PY" src/instrument/m6l_compare.py --i_have_author_goahead --threads 6 \
        --cluster "${g}" > "experiments/battery/m6l_${g}.out" 2>&1
  st2=$?
  echo "[m6] compare ${g} exited ${st2} at $(date '+%H:%M:%S')"
  if [ "${st2}" -ne 0 ]; then BAD="${BAD} ${g}(cmp)"; else OK="${OK} ${g}"; fi
  grep -E "K=15" "experiments/battery/m6l_${g}.out" | tail -1
done
echo "=================================================================="
echo "[m6] DONE $(date '+%H:%M:%S')  OK:${OK:-none}  FAILED:${BAD:-none}"
