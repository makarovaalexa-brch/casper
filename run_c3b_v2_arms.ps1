# C3B_V2 driver -- both adapter arms on QUESTION targets, sequentially, then the epoch-2 early read.
# Design sheet: docs/design/DS_C3B_V2_QUESTIONS.md (PRE-REGISTERED 2026-08-30, BINDING).
# Detached from any Claude session on purpose: session background bash has been reaped twice.
# Log: experiments/paperC/c3b_v2/driver_run.log (+ the trainer's own jsonl). Exit codes per step.
# 6 epochs x 2 arms; the smoke probe measured 3.18 s/step, 437 steps/epoch => ~25 min/epoch/arm,
# so ~2.5 h per arm and ~5 h end to end. Threads 4 (4 PHYSICAL cores).
Set-Location $PSScriptRoot
$log = "experiments/paperC/c3b_v2/driver_run.log"
function Step($name, $argv) {
    Add-Content $log "[$(Get-Date -Format o)] START $name"
    & "C:\Program Files\Python312\python.exe" @argv *>> $log
    Add-Content $log "[$(Get-Date -Format o)] EXIT $name code=$LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { Add-Content $log "ABORTING: $name failed"; exit $LASTEXITCODE }
}
Add-Content $log "[$(Get-Date -Format o)] DRIVER START pid=$PID rev=$(git rev-parse HEAD)"
Step "train-true"      @("src/instrument/c3b_adapter.py","train","--targets","questions","--arm","true","--threads","4","--epochs","6")
Step "train-scrambled" @("src/instrument/c3b_adapter.py","train","--targets","questions","--arm","scrambled","--threads","4","--epochs","6")
Step "earlyread"       @("src/instrument/c3b_adapter.py","earlyread","--targets","questions","--epoch","2")
Add-Content $log "[$(Get-Date -Format o)] DRIVER DONE"
