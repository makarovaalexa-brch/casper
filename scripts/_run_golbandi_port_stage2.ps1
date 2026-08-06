# TASK #13 STAGE 2: the two port cells, on the canonical Liang ML-25M ruler.
#   (P) the method as published  = selector golbandi_tree  x consumer golbandi_leaf
#   (X) the Act-1 cell           = selector golbandi_tree  x consumer ours (certified tower i26c ep31)
# Both share ONE set of interviews, so the selector is identical across the two cells by construction.
#
# Matched static comparators run through golbandi_tree_selector.StaticOrderTreeSelector itself
# (treestatic:*), and `entropy0-static` is included both to tie these rows to the published
# ACT1_ADAPTIVE_VS_STATIC table and as a measured identity check against treestatic:entropy0.
#
# The tree is LOADED from the stage-1 checkpoint (contract-signed); this stage never re-grows it.
$ErrorActionPreference = "Stop"
Set-Location C:\dev\phd\casper
$py = "C:\Program Files\Python312\python.exe"
$log = "experiments\baselines\interview\gtport13.log"

Add-Content $log "[launcher] STAGE 2 harness $(Get-Date -Format o)"
& $py src\baselines\adaptive_harness.py `
    --selectors golbandi_tree,treestatic:popularity,treestatic:helf,treestatic:entropy0,entropy0-static `
    --budgets 1,2,4,8 `
    --only ours,golbandi_leaf `
    --golbandi_lam 200 `
    --golbandi_depth 8 `
    --golbandi_tree_cache .cache\snap5\tree_ml25m_liang_d8.json `
    --arch i26 --snapshot models\i26c_ep31\t2i26c_ep31.pt `
    --no_floor --per_user --label gtport13 *>> $log
Add-Content $log "[launcher] STAGE 2 exit=$LASTEXITCODE $(Get-Date -Format o)"
