# i26_watchdog.ps1 -- overnight keep-alive for the interview-native retrain (author directive
# 2026-07-30: "make sure the job doesn't die overnight ... just make sure best chkp is always written").
#
# Every run: if the DONE marker is absent and the trainer is not alive (dead PID or stale heartbeat),
# relaunch it DETACHED. train_i26.py RESUMES from t2i26_last.pt (encoder + optimiser + epoch + best),
# so a relaunch continues rather than starting over. Never double-launches: a live PID with a fresh
# heartbeat is left alone.
#
# The best checkpoint is written ATOMICALLY (temp + os.replace) by the trainer itself, so a crash --
# including a crash during the write -- cannot corrupt it.
#
# Install (runs every 15 min, survives logout):
#   schtasks /Create /TN i26_watchdog /SC MINUTE /MO 15 /TR "powershell -NoProfile -ExecutionPolicy Bypass -File C:\dev\phd\casper\src\instrument\i26_watchdog.ps1" /F
# Remove:
#   schtasks /Delete /TN i26_watchdog /F

$ErrorActionPreference = 'SilentlyContinue'
$root   = 'C:\dev\phd\casper'
$out    = Join-Path $root 'experiments\instrument'
$done   = Join-Path $out  't2i26_DONE.marker'
$state  = Join-Path $out  't2i26_state.json'
$runlog = Join-Path $out  't2i26_run.out'
$log    = Join-Path $out  'i26_watchdog.log'
$py     = 'C:\Program Files\Python312\python.exe'
$script = Join-Path $root 'src\instrument\train_i26.py'

function Log($m) { "$([DateTime]::Now.ToString('s')) $m" | Out-File -FilePath $log -Append -Encoding utf8 }

if (Test-Path $done) { Log 'DONE marker present -> nothing to do'; exit 0 }

$alive = $false
if (Test-Path $state) {
    try {
        $s   = Get-Content $state -Raw | ConvertFrom-Json
        $cpid = [int]$s.pid
        $age  = [double][DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - [double]$s.heartbeat
        $proc = Get-Process -Id $cpid -ErrorAction SilentlyContinue
        # 1800 s: an epoch-boundary evaluation (canonical val + empty-set + k=1) can take several
        # minutes with no heartbeat, so the staleness threshold must comfortably exceed it.
        if ($proc -and ($proc.ProcessName -like 'python*') -and ($age -lt 1800)) {
            $alive = $true
            Log ("alive: pid=$cpid ep=" + $s.epoch + " step=" + $s.step + " best=" + $s.best_val + " hb_age=" + [int]$age + "s")
        } else {
            Log ("DEAD or STALE: pid=$cpid hb_age=" + [int]$age + "s proc=" + ($proc -ne $null))
        }
    } catch { Log 'state file unreadable -> treating as dead'; $alive = $false }
} else {
    Log 'no state file yet -> treating as dead'
}

if (-not $alive) {
    $env:OMP_NUM_THREADS = '6'
    $env:OPENBLAS_NUM_THREADS = '1'
    Log 'RELAUNCHING train_i26.py (detached; resumes from t2i26_last.pt)'
    Start-Process -FilePath $py `
        -ArgumentList @('-u', $script, '--epochs', '14', '--tag', 't2i26', '--steps_per_epoch', '2200') `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput $runlog -RedirectStandardError ($runlog + '.err')
}
