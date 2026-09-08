# Chain 2, launched 2026-09-09 after chain 1's ConUCB points.
#   (a) A11 with the consumers that actually exist on this box.
#   (b) DS-A10 grid EXTENSION. The pre-registered grids were monotone at their edge: GLM-UCB rose to
#       alpha=4.0 (0.21859 vs 0.21281 published) and Thompson rose as the draw shrank (scale=0.5,
#       0.21171 vs 0.20991). Extending in the improving direction can only STRENGTHEN a baseline, i.e.
#       it works against this paper's claim, so it is the conservative direction to extend and is
#       recorded here as an extension rather than presented as the original grid.
$ErrorActionPreference = "Continue"
Set-Location "C:\dev\phd\casper"
$master = "experiments\baselines\interview\a11_a10_chain.out"
function Step($name, $argv) {
    "[$(Get-Date -Format HH:mm:ss)] START $name" | Out-File -Append $master
    & python src\baselines\adaptive_harness.py @argv 2>&1 | Out-File -Append "experiments\baselines\interview\$name.out"
    "[$(Get-Date -Format HH:mm:ss)] END   $name exit=$LASTEXITCODE" | Out-File -Append $master
}
$common = @("--arch","i26","--no_floor","--concept_neg_cap","0")

# ---- A11. EXCLUDED for want of a checkpoint on this box, not by choice:
#      tanp  (.cache/baselines/tanp_ml25m_liang.pt), eddi (.cache/baselines/eddi_ml25m_liang.pt).
#      Both loaders read the epoch off the checkpoint before fitting, so neither can be produced
#      without a training run. Reported as unavailable.
Step "a11_tree_consumers" ($common + @("--selectors","replay:ndcgtree16test,fixedorder_json:policyB_combined",
    "--budgets","1,2,4,8,16","--only","ours,recvae_likes,golbandi_leaf,belief_mf",
    "--per_user","--label","a11_tree_consumers"))

foreach ($a in @("8.0","16.0")) {
    Step "a10_val_glmucb_a$a" ($common + @("--cohort","validation","--selectors","glm_ucb","--budgets","16",
        "--only","ours","--ucb_alpha",$a,"--label","a10_glmucb_a$a"))
}
foreach ($s in @("0.25","0.1")) {
    Step "a10_val_thompson_s$s" ($common + @("--cohort","validation","--selectors","thompson","--budgets","16",
        "--only","ours","--ts_scale",$s,"--label","a10_thompson_s$s"))
}
"[$(Get-Date -Format HH:mm:ss)] CHAIN 2 DONE" | Out-File -Append $master
