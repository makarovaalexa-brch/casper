#!/usr/bin/env bash
# M6 stage 2: waits for the tree fits, then distils STATE-ONLY and runs generalisation + mix.
# VALIDATION ONLY throughout -- the author's instruction is explicit that the test set stays sealed.
set -u
cd /c/dev/phd/casper || exit 1
PY="C:/Program Files/Python312/python.exe"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6

# 1. wait for the fits, but never forever, and never on a dead process
for i in $(seq 1 900); do
  grep -q "^\[m6\] DONE" experiments/battery/m6_all.out && break
  tasklist 2>/dev/null | grep -qi python.exe || { echo "[chain] fits died; ABORT"; exit 1; }
  sleep 60
done
grep -q "^\[m6\] DONE" experiments/battery/m6_all.out || { echo "[chain] fits timed out; ABORT"; exit 1; }
echo "[chain] fits complete at $(date '+%H:%M:%S')"
ls experiments/battery/m6/m6k_*.json | sed 's/.*m6k_//;s/\.json//' | tr '\n' ' '
GROUPS=$(ls experiments/battery/m6/m6k_*.json | sed 's/.*m6k_//;s/\.json//' | tr '\n' ',' | sed 's/,$//')
echo "[chain] groups with a fitted tree: ${GROUPS}"

# 2. trajectories on TRAIN users, openers from film-pool half A
"$PY" src/instrument/m6c_distill_gen.py --i_have_author_goahead --threads 6 --half A \
      > experiments/battery/m6c_gen.out 2>&1
grep -q "\[done\] all slices banked" experiments/battery/m6c_gen.out || { echo "[chain] GEN FAILED"; exit 1; }
echo "[chain] generation done at $(date '+%H:%M:%S')"

# 3. distil, state-only (there is no token to announce: the opener differs per user)
"$PY" src/instrument/m4b_policy.py --i_have_author_goahead --threads 6 --epochs 3 \
      --anchor_encoding none --no_cold_all \
      --traj_dir experiments/battery/m6/traj --trees "cold,${GROUPS}" \
      --tag m6 > experiments/battery/m6_train.out 2>&1
[ -f experiments/battery/m6/m6_policy.pt ] || cp experiments/battery/m4/m6_policy.pt experiments/battery/m6/ 2>/dev/null
[ -f experiments/battery/m6/m6_policy.pt ] || { echo "[chain] TRAIN FAILED"; exit 1; }
echo "[chain] distillation done at $(date '+%H:%M:%S')"

# 4. generalisation on UNSEEN films (half B), then the group mix with the do-no-harm cell
"$PY" src/instrument/m6d_eval.py --i_have_author_goahead --threads 6 --mode generalise --half B \
      > experiments/battery/m6d_generalise.out 2>&1
echo "[chain] generalise exited $? at $(date '+%H:%M:%S')"
"$PY" src/instrument/m6d_eval.py --i_have_author_goahead --threads 6 --mode mix --half B \
      > experiments/battery/m6e_mix.out 2>&1
echo "[chain] mix exited $? at $(date '+%H:%M:%S')"
echo "[chain] M6 PIPELINE DONE $(date '+%H:%M:%S')"
