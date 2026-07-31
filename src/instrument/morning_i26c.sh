#!/usr/bin/env bash
# morning_i26c.sh -- the go/no-go battery for the i26 certification run, in one command.
#
# WHY A SCRIPT. The author's instruction (2026-07-31) was: "check ASAP if it beats our i25 certified
# chkp, if not, all other work wasted". That check must not depend on anyone reconstructing five
# commands from a chat log at 6 a.m., and every step must run on the SELECTED checkpoint rather than on
# whichever file happens to be called _best. Selection happens first, from VAL only, by the
# pre-registered rule; nothing here can feed a test number back into the choice.
#
#   bash src/instrument/morning_i26c.sh
#
# Writes everything under experiments/ and prints a comparison block at the end. Safe to re-run.
set -u                      # NOT -e: one failing gate must not prevent the others from producing data
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=6
CK=.cache/instrument
SEL=$CK/t2i26c_SELECTED.pt
OUT=experiments/baselines
mkdir -p $OUT

echo "############ 0. SELECT (val only, pre-registered rule: max sel + val_full) ############"
python src/instrument/select_i26c.py --tag t2i26c --copy 2>&1 | tee experiments/instrument/t2i26c_selection.out
if [ ! -f "$SEL" ]; then echo "!! NO SELECTED CHECKPOINT -- stopping."; exit 1; fi

echo
echo "############ 1. G1/G4  full profile + arm A/N on the TEST ruler ############"
# arm_n_tower prints A_full (directly comparable to the certified 0.3482), the pool-change row, the
# arm-N rows, and the k-truncated cold rows -- G1 and G4 in one pass.
python -u src/baselines/arm_n_tower.py --arch i26 --snapshot $SEL 2>&1 | tee $OUT/arm_n_tower_i26c.out

echo
echo "############ 2. G3  paired bootstrap vs RecVAE, both arms ############"
# Paired, not unpaired: the two models score the SAME 10,000 users. The chapter's old CI 0.0069 was
# 1.96*sqrt(2)*SE and overstated the interval by ~4.6x.
python -u src/baselines/arm_n_paired.py --arch i26 --snapshot $SEL --arm A --rival recvae 2>&1 | tee $OUT/paired_i26c_armA.out
python -u src/baselines/arm_n_paired.py --arch i26 --snapshot $SEL --arm N --rival recvae 2>&1 | tee $OUT/paired_i26c_armN.out

echo
echo "############ 3. G2  interview table, k=8, uniform floor, elicitation baselines ############"
# The uniform floor (no per-model exemption) is the fair contract: no-floor destroys models with no
# prior (rbmf scores 0.0011 on pure_entropy, i.e. random), floor slightly penalises models that can
# read unknowns (golbandi 0.1948 -> 0.1899). Report this as primary, no-floor as robustness.
python -u src/baselines/run_interview_table.py --arch i26 --snapshot $SEL \
    --only ours,golbandi_leaf,rbmf,belief_mf --budgets 8 --label k8_i26c 2>&1 | tee $OUT/interview_k8_i26c.out

echo
echo "############ 4. the same table, no floor (robustness) ############"
python -u src/baselines/run_interview_table.py --arch i26 --snapshot $SEL \
    --only ours,golbandi_leaf,rbmf,belief_mf --budgets 8 --no_floor --label k8_i26c_NOFLOOR 2>&1 \
    | tee $OUT/interview_k8_i26c_nofloor.out

echo
echo "############ 5. interview curve, ours only, all budgets ############"
python -u src/baselines/run_interview_table.py --arch i26 --snapshot $SEL \
    --only ours --budgets 1,2,4,8,16 --label curve_i26c 2>&1 | tee $OUT/interview_curve_i26c.out

echo
echo "################################ VERDICT BLOCK ################################"
echo "REFERENCE (canonical Liang ML-25M TEST ruler):"
echo "  RecVAE                       0.3540 / 0.2497"
echo "  i25 CERTIFIED                0.3482 / 0.2462     interview entropy0 k=8 (uniform floor) 0.1935"
echo "  i26 run1 ep20 (2-hop)        0.3416 / 0.2455     interview entropy0 k=8                 0.2078"
echo "  step-0 of run2, untrained    0.3510 val          <- what training must not destroy"
echo
echo "THIS RUN:"
grep -E "A_full|N_full" $OUT/arm_n_tower_i26c.out 2>/dev/null
grep -E "ours .*entropy0|ours .*helf" $OUT/interview_k8_i26c.out 2>/dev/null
echo
echo "DECISION: does it beat i25 certified on the BALANCE (full profile vs interview)?"
echo "  If yes -> i26 becomes the instrument; proceed to concepts, then belief, then EDDI/TaNP."
echo "  If it matches run1 ep20 -> take THIS one anyway: same balance, one-hop provenance."
echo "  If both lose to i25 on the balance -> the retrain line is dead, Paper A ships on i25."
