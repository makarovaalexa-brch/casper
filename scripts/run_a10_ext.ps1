# A10 grid extension, launched after A11 finishes. One job at a time.
$ErrorActionPreference = "Continue"
Set-Location "C:\dev\phd\casper"
$master = "experiments\baselines\interview\a11_a10_chain.out"
function Step($name, $argv) {
    "[$(Get-Date -Format HH:mm:ss)] START $name" | Out-File -Append $master
    & python src\baselines\adaptive_harness.py @argv 2>&1 | Out-File -Append "experiments\baselines\interview\$name.out"
    "[$(Get-Date -Format HH:mm:ss)] END   $name exit=$LASTEXITCODE" | Out-File -Append $master
}
$common = @("--arch","i26","--no_floor","--concept_neg_cap","0")
foreach ($a in @("8.0","16.0")) {
    Step "a10_val_glmucb_a$a" ($common + @("--cohort","validation","--selectors","glm_ucb","--budgets","16",
        "--only","ours","--ucb_alpha",$a,"--label","a10_glmucb_a$a"))
}
foreach ($s in @("0.25","0.1")) {
    Step "a10_val_thompson_s$s" ($common + @("--cohort","validation","--selectors","thompson","--budgets","16",
        "--only","ours","--ts_scale",$s,"--label","a10_thompson_s$s"))
}
"[$(Get-Date -Format HH:mm:ss)] EXT DONE" | Out-File -Append $master
