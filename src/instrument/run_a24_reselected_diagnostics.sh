#!/usr/bin/env bash
# run_a24_reselected_diagnostics.sh -- re-run A22's provisional diagnostics on the RE-SELECTED epochs.
#
# A23 showed the concept channel ports (ML-25M +0.0015, Goodreads +0.0016 averaged over all 12 epochs),
# and that A22's Addenda 1-2 had run every diagnostic on cvg_ep12 / cv3_ep04 -- both chosen by the bent
# monitor, and both weak (fold +0.0014 and +0.0006). Their measurements are sound; reading them as
# DATASET properties is not. These are the same diagnostics on cvg_ep01 (+0.0028/+0.0035) and
# cv3_ep11 (+0.0024/+0.0020).
#
# Seeds: --seed0 777, INDEPENDENT of the 4242/5242 set that selected these checkpoints.
# One job at a time (HARD RULE 11d) -- this box has 4 physical cores.

set -u
cd /c/dev/phd/casper
export OMP_NUM_THREADS=4
L=experiments/goodreads
mkdir -p "$L"

GR_TOWER=.cache/instrument/t2i26g_ep53.pt
GR_CKPT=.cache/instrument/cvg_ep01.pt
ML_TOWER=models/i26c_ep31/t2i26c_ep31.pt
ML_CKPT=.cache/instrument/cv3_ep11.pt

run () {  # run <logname> <env> <args...>
  local name=$1; shift
  local ds=$1; shift
  echo "=== $(date +%H:%M:%S) $name ==="
  CASPER_DATASET=$ds python -u src/instrument/a22_concept_curve_paired.py \
      --seeds 3 --skip_g1 --seed0 777 "$@" 2>&1 | grep -E "AGG|G2|ERROR|Traceback"
}

# 1. refusals-vs-answers -- the most striking Addendum-1 claim was that on Goodreads the ANSWERS buy
#    nothing and the REFUSALS carry the lift. Baselines (no flag) come from A23 but are re-run here so
#    the contrast is on identical packs.
run "GR ep01 broad"              goodreads --snapshot $GR_TOWER --ckpt $GR_CKPT --tag g1B   --ask broad
run "GR ep01 broad no_refusals"  goodreads --snapshot $GR_TOWER --ckpt $GR_CKPT --tag g1NR  --ask broad --no_refusals
run "ML ep11 broad"              ml-25m    --snapshot $ML_TOWER --ckpt $ML_CKPT --tag m11B  --ask broad
run "ML ep11 broad no_refusals"  ml-25m    --snapshot $ML_TOWER --ckpt $ML_CKPT --tag m11NR --ask broad --no_refusals

# 2. equal-ANSWER curve -- Addendum 2's decisive test, which gave -0.0001 at m=22 on the weak ep12.
run "GR ep01 equal-answer"       goodreads --snapshot $GR_TOWER --ckpt $GR_CKPT --tag g1L   --ask broad --ms 0,8,16,22,32

# 3. ask-set arm -- the member-count hypothesis. topN:61 gave -0.0077 on ep12.
run "GR ep01 topN:61"            goodreads --snapshot $GR_TOWER --ckpt $GR_CKPT --tag g1T   --ask topN:61
run "GR ep01 band:61-200"        goodreads --snapshot $GR_TOWER --ckpt $GR_CKPT --tag g1D   --ask band:61-200

echo "=== $(date +%H:%M:%S) A24 COMPLETE ==="
