# DS-A10 final: the reference the grid is read against, then the CHOSEN constants on TEST.
# Chosen on validation, K=16, consumer = certified tower:
#   GLM-UCB alpha = 4.0   (0.21859 vs 0.21281 published; interior maximum, falls at 8.0 and 16.0)
#   Thompson scale = 0.5  (0.21171 vs 0.20991 published; interior maximum, falls at 0.25 and 1.0)
#   ConUCB  lambda = 0.5  = the published value; no test re-run needed.
$ErrorActionPreference = "Continue"
Set-Location "C:\dev\phd\casper"
$master = "experiments\baselines\interview\a11_a10_chain.out"
function Step($name, $argv) {
    "[$(Get-Date -Format HH:mm:ss)] START $name" | Out-File -Append $master
    & python src\baselines\adaptive_harness.py @argv 2>&1 | Out-File -Append "experiments\baselines\interview\$name.out"
    "[$(Get-Date -Format HH:mm:ss)] END   $name exit=$LASTEXITCODE" | Out-File -Append $master
}
$common = @("--arch","i26","--no_floor","--concept_neg_cap","0")

# (1) the static comparator on the SAME validation cohort the grid was chosen on
Step "a10_val_static" ($common + @("--cohort","validation","--selectors","fixedorder_json:policyB_combined",
    "--budgets","16","--only","ours","--label","a10_static"))

# (2) the chosen constants, on TEST, full budget ladder
Step "a10_test_glmucb_a4" ($common + @("--selectors","glm_ucb","--budgets","1,2,4,8,16","--only","ours",
    "--ucb_alpha","4.0","--per_user","--label","a10_test_glmucb_a4"))
Step "a10_test_thompson_s05" ($common + @("--selectors","thompson","--budgets","1,2,4,8,16","--only","ours",
    "--ts_scale","0.5","--per_user","--label","a10_test_thompson_s05"))
"[$(Get-Date -Format HH:mm:ss)] A10 FINAL DONE" | Out-File -Append $master
