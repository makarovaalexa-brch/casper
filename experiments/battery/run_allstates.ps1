# Queued behind phase 3 (HARD RULE 11d: one heavy job at a time). Waits for the sweep PID to exit,
# re-checks that no other heavy job took the box, then refits the value head on all banked states.
param([int]$WaitPid = 0)
$ErrorActionPreference = "Continue"
Set-Location C:\dev\phd\casper
$py = "C:\Program Files\Python312\python.exe"
$B  = "experiments\battery"
if ($WaitPid -gt 0) { try { Wait-Process -Id $WaitPid -ErrorAction Stop } catch { } }
"=== phase 3 (pid $WaitPid) gone at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" |
    Out-File -Append $B\allstates_chain.log
# do not start if another training/sweep job is live
$busy = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*policy_labels.py*" -or $_.CommandLine -like "*outcome_head.py*" }
if ($busy) {
    "=== REFUSING: another heavy job is live ($($busy.ProcessId -join ',')) ===" |
        Out-File -Append $B\allstates_chain.log
    exit 1
}
& $py src/instrument/lift_head.py --allstates --lift_seeds 3 --lift_epochs 60 --lift_lr 1e-4 `
    --threads 8 --full_threads --i_have_author_goahead *> $B\allstates.log
"=== allstates exit=$LASTEXITCODE at $(Get-Date -Format 'HH:mm:ss') ===" |
    Out-File -Append $B\allstates_chain.log
