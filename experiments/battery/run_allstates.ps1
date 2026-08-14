# Queued behind phase 3 (HARD RULE 11d: one heavy job at a time).
#
# NOT a one-shot wait on a PID: phase 3 is a RESUMABLE sweep and relaunches under a new PID at each
# resume boundary (observed 25108 -> 27440), so waiting on a single PID would fire mid-sweep. This
# polls for a SUSTAINED idle box - no policy_labels/outcome_head/lift_head process for N consecutive
# checks - before starting, and re-checks once more immediately before launching.
param([int]$IdleChecks = 3, [int]$IntervalSec = 60, [int]$MaxHours = 14)
$ErrorActionPreference = "Continue"
Set-Location C:\dev\phd\casper
$py  = "C:\Program Files\Python312\python.exe"
$B   = "experiments\battery"
$log = "$B\allstates_chain.log"
function Heavy {
    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -like "*policy_labels.py*" -or
        $_.CommandLine -like "*outcome_head.py*"  -or
        $_.CommandLine -like "*lift_head.py*" }
}
"=== queued $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'); polling for a sustained idle box ===" |
    Out-File -Append $log
$idle = 0
$deadline = (Get-Date).AddHours($MaxHours)
while ((Get-Date) -lt $deadline) {
    $h = Heavy
    if ($h) {
        if ($idle -gt 0) { "  busy again ($($h.ProcessId -join ',')), idle counter reset" |
            Out-File -Append $log }
        $idle = 0
    } else {
        $idle++
        "  idle $idle/$IdleChecks at $(Get-Date -Format 'HH:mm:ss')" | Out-File -Append $log
        if ($idle -ge $IdleChecks) { break }
    }
    Start-Sleep -Seconds $IntervalSec
}
if (Heavy) { "=== REFUSING at launch: box busy again ===" | Out-File -Append $log; exit 1 }
if ((Get-Date) -ge $deadline) { "=== GAVE UP: deadline reached ===" | Out-File -Append $log; exit 1 }
"=== STARTING allstates $(Get-Date -Format 'HH:mm:ss') ===" | Out-File -Append $log
& $py src/instrument/lift_head.py --allstates --lift_seeds 3 --lift_epochs 60 --lift_lr 1e-4 `
    --threads 8 --full_threads --i_have_author_goahead *> $B\allstates.log
"=== allstates exit=$LASTEXITCODE at $(Get-Date -Format 'HH:mm:ss') ===" | Out-File -Append $log
