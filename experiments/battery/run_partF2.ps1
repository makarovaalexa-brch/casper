# Part F chain, stage 2+. Detached so it survives the session; one heavy job at a time.
# Waits for the outcome-head process to exit, then recovers the metric suite if the run was
# stopped before it, then the lift head + combined policy, then the shuffle control.
param([int]$WaitPid = 0)
$ErrorActionPreference = "Continue"
Set-Location C:\dev\phd\casper
$py = "C:\Program Files\Python312\python.exe"
$B  = "experiments\battery"

if ($WaitPid -gt 0) {
    try { Wait-Process -Id $WaitPid -ErrorAction Stop } catch { }
}
"=== stage 1 process gone at $(Get-Date -Format HH:mm:ss) ===" | Out-File -Append $B\chain2.log

# Recovery: the head-vs-popularity suite lives after the epoch loop, so a run stopped early has a
# checkpoint but no comparison. --eval_only scores it from the checkpoint on disk.
if ((Test-Path "$B\outcome_head_main_both.pt") -and -not (Test-Path "$B\outcome_head_main_both.json")) {
    "=== recovering the metric suite with --eval_only ===" | Out-File -Append $B\chain2.log
    & $py src/instrument/outcome_head.py --arm main --eval_only --batch 1024 --d_k 64 `
        --threads 8 --full_threads --i_have_author_goahead *> $B\outcome_main_evalonly.log
    "=== eval_only exit=$LASTEXITCODE ===" | Out-File -Append $B\chain2.log
}

if (Test-Path "$B\outcome_head_main_both.pt") {
    & $py src/instrument/lift_head.py --threads 8 --full_threads `
        --i_have_author_goahead *> $B\lift_head.log
    "=== lift_head exit=$LASTEXITCODE ===" | Out-File -Append $B\chain2.log
} else {
    "=== LIFT SKIPPED: no outcome-head checkpoint ===" | Out-File -Append $B\chain2.log
}

& $py src/instrument/outcome_head.py --arm shuffle --epochs 1 --batch 1024 --d_k 64 `
    --threads 8 --full_threads --i_have_author_goahead *> $B\outcome_shuffle.log
"=== shuffle exit=$LASTEXITCODE ===" | Out-File -Append $B\chain2.log
"=== CHAIN COMPLETE $(Get-Date -Format HH:mm:ss) ===" | Out-File -Append $B\chain2.log
