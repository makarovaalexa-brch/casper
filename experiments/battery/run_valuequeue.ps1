# Queued strictly BEHIND phase 3 (priority: phase 3 control + fine-tunes, then these).
#   A. --directgain  : one head predicting the measured gain directly, warm-started from E1
#   B. --statecurve  : policy score vs number of measured states (parameter-free, no training)
#   C. --allstates   : the 245+-state refit of the factorised head
# Threads 6 (standing setting: 8 on 4 physical cores cost ~2.5x on 2026-08-12).
param([int]$IdleChecks = 3, [int]$IntervalSec = 60, [int]$MaxHours = 20)
$ErrorActionPreference = "Continue"
Set-Location C:\dev\phd\casper
$py  = "C:\Program Files\Python312\python.exe"
$B   = "experiments\battery"
$log = "$B\valuequeue.log"
function Heavy {
    # Name filter FIRST: matching command-line text alone also matches any shell that merely MENTIONS
    # these scripts, including the diagnostics used to inspect this queue, which deadlocks the wait.
    Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object {
        $_.CommandLine -like "*policy_labels.py*" -or $_.CommandLine -like "*policy_deep.py*" -or
        $_.CommandLine -like "*value_chain.py*"   -or $_.CommandLine -like "*outcome_head.py*" -or
        $_.CommandLine -like "*lift_head.py*" }
}
function WaitIdle {
    $idle = 0; $deadline = (Get-Date).AddHours($MaxHours)
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
function RunStage($name, $argv, $out) {
    if (-not (WaitIdle)) { "=== GAVE UP before $name ===" | Out-File -Append $log; exit 1 }
    if (Heavy) { "=== REFUSING $name : busy at launch ===" | Out-File -Append $log; exit 1 }
    "=== $name STARTING $(Get-Date -Format 'HH:mm:ss') ===" | Out-File -Append $log
    & $py src/instrument/lift_head.py @argv *> $out
    "=== $name exit=$LASTEXITCODE at $(Get-Date -Format 'HH:mm:ss') ===" | Out-File -Append $log
}
"=== value queue armed $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'); phase 3 has priority ===" |
    Out-File -Append $log
RunStage "A directgain" @("--directgain","--lift_seeds","3","--lift_epochs","60","--lift_lr","1e-4",
                          "--threads","6","--full_threads","--i_have_author_goahead") "$B\directgain.log"
RunStage "B statecurve" @("--statecurve","--curve_draws","3","--threads","6","--full_threads",
                          "--i_have_author_goahead") "$B\statecurve.log"
RunStage "C allstates"  @("--allstates","--lift_seeds","3","--lift_epochs","60","--lift_lr","1e-4",
                          "--threads","6","--full_threads","--i_have_author_goahead") "$B\allstates.log"
"=== value queue COMPLETE $(Get-Date -Format 'HH:mm:ss') ===" | Out-File -Append $log
