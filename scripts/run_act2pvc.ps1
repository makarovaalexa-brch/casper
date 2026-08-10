# Stage 1 of DS_ACT2_CONCEPT_BANK_PV.md -- the act2pvc harness run.
# ONE process, fresh label, per-cell persistence (HARD RULE 11a/e).
# Conventions are the design sheet's and are NOT optional:
#   --concept_neg_cap 0        the CERTIFIED-ERA convention concept_cv3_ep08 was trained under
#   --keyterm_repr uniform_member_mean
#                              the value ALL FOUR carried-forward Act-1 concept-asker cells used
#                              (cfold_uniform_negcap0 for conlinucb_mcr/bs_static, convbandit_uniform_v2
#                              for conucb, dpe_v1 for dpe_concept) -- read off the run JSONs, not guessed.
# Spec ORDER is cheapest-and-most-load-bearing first: fixedorder:entropy0 is the PV reproduction gate
# and must land in minutes, and conucb is last because its Eq.8 cost is the one unbounded term.
$env:OMP_NUM_THREADS = "8"
$env:OPENBLAS_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"
$py = "C:\Users\Tech\AppData\Local\pypoetry\Cache\virtualenvs\casper-rAThf-mq-py3.12\Scripts\python.exe"
& $py src\baselines\adaptive_harness.py `
  --selectors "fixedorder:entropy0,fixedorder:greedy_items,fixedorder:greedy_concepts,dpe_concept,conlinucb_bs_static,conlinucb_mcr,conucb" `
  --budgets 1,2,4,8 `
  --only ours `
  --arch i26 `
  --snapshot models\i26c_ep31\t2i26c_ep31.pt `
  --concept_ckpt models\i26c_ep31\concept_cv3_ep08.pt `
  --no_floor `
  --per_user `
  --label act2pvc `
  --concept_neg_cap 0 `
  --keyterm_repr uniform_member_mean
