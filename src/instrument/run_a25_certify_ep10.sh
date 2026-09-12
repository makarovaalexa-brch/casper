#!/usr/bin/env bash
# Certify cvg_ep10 as the Goodreads concept checkpoint: the epoch the project's own pre-registered
# selection rule picks when it is actually applied. A17 used ep12 (the run's last epoch) by mistake.
# Seeds 777/1777/2777 -- independent of the 4242/5242 set used in the A23 sweep.
set -u
cd /c/dev/phd/casper
export OMP_NUM_THREADS=4
T=.cache/instrument/t2i26g_ep53.pt
C=.cache/instrument/cvg_ep10.pt
run () { local n=$1; shift
  echo "=== $(date +%H:%M:%S) $n ==="
  CASPER_DATASET=goodreads python -u src/instrument/a22_concept_curve_paired.py \
    --snapshot $T --ckpt $C --seeds 3 --skip_g1 --seed0 777 "$@" 2>&1 | grep -E "AGG|G2|Traceback"
}
run "ep10 broad"             --tag c10B  --ask broad
run "ep10 answers only"      --tag c10NR --ask broad --no_refusals
run "ep10 equal-answer"      --tag c10L  --ask broad --ms 0,8,16,22,32
echo "=== $(date +%H:%M:%S) A25 COMPLETE ==="
