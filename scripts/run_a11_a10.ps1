# DS-A11 then DS-A10, sequentially (one heavy job at a time). Launched detached 2026-09-08.
$ErrorActionPreference = "Continue"
Set-Location "C:\dev\phd\casper"
$master = "experiments\baselines\interview\a11_a10_chain.out"
function Step($name, $argv) {
    # NOTE: the parameter must not be called $args -- that is PowerShell's automatic variable and the
    # first launch (2026-09-08 22:14) silently dropped every flag, running i25 defaults on every step.
    "[$(Get-Date -Format HH:mm:ss)] START $name" | Out-File -Append $master
    & python src\baselines\adaptive_harness.py @argv 2>&1 | Out-File -Append "experiments\baselines\interview\$name.out"
    "[$(Get-Date -Format HH:mm:ss)] END   $name exit=$LASTEXITCODE" | Out-File -Append $master
}
$common = @("--arch","i26","--no_floor","--concept_neg_cap","0")

# ---- A11: the tree and its comparator replayed through every consumer, test cohort, per-user vectors
Step "a11_tree_consumers" ($common + @("--selectors","replay:ndcgtree16test,fixedorder_json:policyB_combined",
    "--budgets","1,2,4,8,16","--only","ours,recvae_likes,tanp,golbandi_leaf,belief_mf,eddi",
    "--per_user","--label","a11_tree_consumers"))

# ---- A10: constants chosen on VALIDATION, K=16, consumer ours only
foreach ($a in @("0.25","0.5","1.0","2.0","4.0")) {
    Step "a10_val_glmucb_a$a" ($common + @("--cohort","validation","--selectors","glm_ucb","--budgets","16",
        "--only","ours","--ucb_alpha",$a,"--label","a10_glmucb_a$a"))
}
foreach ($s in @("0.5","1.0","2.0")) {
    Step "a10_val_thompson_s$s" ($common + @("--cohort","validation","--selectors","thompson","--budgets","16",
        "--only","ours","--ts_scale",$s,"--label","a10_thompson_s$s"))
}
foreach ($l in @("0.1","0.5","2.0")) {
    Step "a10_val_conucb_l$l" ($common + @("--cohort","validation","--selectors","conucb","--budgets","16",
        "--only","ours","--conucb_lam",$l,"--label","a10_conucb_l$l"))
}
"[$(Get-Date -Format HH:mm:ss)] CHAIN DONE" | Out-File -Append $master
