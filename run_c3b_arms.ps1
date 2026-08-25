# C3B overnight driver — both adapter arms sequentially, then the epoch-2 early read.
# Detached from any Claude session on purpose: two session-side kills (2026-08-24/25) cost wall-clock.
# Logs: experiments/paperC/c3b/driver_run.log (+ the trainer's own jsonl). Exit codes recorded per step.
Set-Location $PSScriptRoot
$log = "experiments/paperC/c3b/driver_run.log"
function Step($name, $argv) {
    Add-Content $log "[$(Get-Date -Format o)] START $name"
    & "C:\Program Files\Python312\python.exe" @argv *>> $log
    Add-Content $log "[$(Get-Date -Format o)] EXIT $name code=$LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { Add-Content $log "ABORTING: $name failed"; exit $LASTEXITCODE }
}
Step "train-true"      @("src/instrument/c3b_adapter.py","train","--arm","true","--threads","4","--epochs","10")
Step "train-scrambled" @("src/instrument/c3b_adapter.py","train","--arm","scrambled","--threads","4","--epochs","10")
Step "earlyread"       @("src/instrument/c3b_adapter.py","earlyread","--epoch","2")
Add-Content $log "[$(Get-Date -Format o)] DRIVER DONE"
