# TODO 1.3(b): the THREE-OUTCOME EDDI response model at CONTRACT SCALE.
#
# The only artifact for `eddi_ig3` today is a 200-user smoke stamped SMOKE_NOT_CITABLE
# (`SMOKE_adaptive_table_SNAP3_smoke.json`, budgets <= 8, one consumer, no intervals). This is the
# same cells on the chapter's standard contract: certified pair (t2i26c_ep31 + concept_cv3_ep08),
# policy-B candidate pool (arm_n.build_pool, the harness default since 2026-07-30), NO-FLOOR,
# behavioural answers, learned decoder bias, ALL 10,000 test users, no sampling anywhere.
#
# WHAT IS IN THE SPEC LIST AND WHY:
#   entropy0-static  the G2 cross-run gate. Must reproduce act1_per_user_vectors.npz per user at
#                    0.0e+00 or this run's vectors may not be differenced against anything banked.
#   eddi_ig          the FREE CONTRACT CHECK. It is the two-outcome row already in tab:battery
#                    (.1537/.0336 .. .1917/.0749); re-running it here re-derives those cells inside
#                    the run that produces the three-outcome ones, so the comparison the write-up
#                    makes is in-run and not cross-run.
#   eddi_ig3_zero    three-outcome, dislike branch = zero (conservative; the third outcome enters
#                    only through the answerability weight a_i).
#   eddi_ig3         three-outcome, dislike branch = signed (a DOCUMENTED APPROXIMATION: the frozen
#                    Partial-VAE has no learned token for a disliked observation).
# Cheapest first, so the two gates land in minutes rather than hours.
#
# No new code: both selectors have been wired at adaptive_harness.py:750-759 since 2026-08-05.
# Per-cell persistence + per-user npz after every cell is the harness's own (HARD RULE 11a).
$env:OMP_NUM_THREADS = "8"
$env:OPENBLAS_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"
$py = "C:\Program Files\Python312\python.exe"
& $py src\baselines\adaptive_harness.py `
  --selectors "entropy0-static,eddi_ig,eddi_ig3_zero,eddi_ig3" `
  --budgets 1,2,4,8,16 `
  --only ours `
  --arch i26 `
  --snapshot models\i26c_ep31\t2i26c_ep31.pt `
  --concept_ckpt models\i26c_ep31\concept_cv3_ep08.pt `
  --no_floor `
  --per_user `
  --label eddi3_policyB
