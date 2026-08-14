# Queued behind phase 3 (HARD RULE 11d: one heavy job at a time), in PRIORITY order:
#   (a) the nearest-banked-state CONDITIONAL value -- settles the live question, ~12 min
#   (b) the 245-state refit                        -- answers label starvation, ~2 h
#
# NOT a one-shot wait on a PID: phase 3 is a RESUMABLE sweep and relaunches under a new PID at each
# resume boundary (observed 25108 -> 27440), so waiting on a single PID would fire mid-sweep. This
# polls for a SUSTAINED idle box and re-checks immediately before each launch.
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
function WaitIdle {
    $idle = 0
    $deadline = (Get-Date).AddHours($MaxHours)
    while ((Get-Date) -lt $deadline) {
        if (Heavy) { $idle = 0 } else {
            $idle++
            "  idle $idle/$IdleChecks at $(Get-Date -Format 'HH:mm:ss')" | Out-File -Append $log
            if ($idle -ge $IdleChecks) { return $true }
        }
        Start-Sleep -Seconds $IntervalSec
    }
    return $false
}
"=== queued $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'); polling for a sustained idle box ===" |
    Out-File -Append $log
if (-not (WaitIdle)) { "=== GAVE UP: deadline ===" | Out-File -Append $log; exit 1 }
if (Heavy) { "=== REFUSING: busy again at launch ===" | Out-File -Append $log; exit 1 }

"=== (a) STARTING conditional-value arm $(Get-Date -Format 'HH:mm:ss') ===" | Out-File -Append $log
& $py src/instrument/lift_head.py --meanvec --no_ceiling --label _cond `
    --threads 8 --full_threads --i_have_author_goahead *> $B\conditional.log
"=== (a) exit=$LASTEXITCODE at $(Get-Date -Format 'HH:mm:ss') ===" | Out-File -Append $log

if (-not (WaitIdle)) { "=== GAVE UP before (b) ===" | Out-File -Append $log; exit 1 }
"=== (b) STARTING 245-state refit $(Get-Date -Format 'HH:mm:ss') ===" | Out-File -Append $log
& $py src/instrument/lift_head.py --allstates --lift_seeds 3 --lift_epochs 60 --lift_lr 1e-4 `
    --threads 8 --full_threads --i_have_author_goahead *> $B\allstates.log
"=== (b) exit=$LASTEXITCODE at $(Get-Date -Format 'HH:mm:ss') ===" | Out-File -Append $log
