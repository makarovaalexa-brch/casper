# pb_watchdog.ps1 -- zero-token overnight watchdog for the Paper B replication pipeline.
# Registered via schtasks to run every ~15 min. If the pipeline is DONE, self-unregisters. If neither
# the orchestrator nor its training/interview child is alive AND not done, relaunches the orchestrator
# (which RESUMES pb_train_enc from enc_state.pt best-val checkpoint, then runs the two-model interview).
$ErrorActionPreference = 'SilentlyContinue'
$ROOT = 'C:\dev\phd\casper'
$B = "$ROOT\experiments\paper2_repl"
$done = "$B\pb_ORCH_DONE.txt"
$wlog = "$B\pb_watchdog.log"
function W($m) { Add-Content $wlog "$(Get-Date -Format o) $m" }

if (Test-Path $done) {
    W "DONE marker present ($((Get-Content $done -Raw).Trim())) -- unregistering watchdog"
    schtasks /Delete /TN PBReplWatchdog /F | Out-Null
    exit 0
}
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'pb_run_rest|pb_train_enc|pb_interview' }
if ($procs) {
    W "alive: $((($procs.ProcessId) -join ','))"
    exit 0
}
# not done and nothing running -> resume
try {
    Start-Process -FilePath 'python' -ArgumentList 'scripts/paper2_repl/pb_run_rest.py' `
        -WorkingDirectory $ROOT -RedirectStandardOutput "$B\orch_stdout.log" `
        -RedirectStandardError "$B\orch_stderr.log" -WindowStyle Hidden
    W "RESTARTED orchestrator (resume from best-val checkpoint)"
} catch {
    W "RESTART FAILED: $_"
}
