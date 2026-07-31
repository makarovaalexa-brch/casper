# i26c_watchdog.ps1 -- overnight keep-alive for the i26 CERTIFICATION run (tag t2i26c).
#
# SEPARATE FILE from i26_watchdog.ps1 ON PURPOSE. That one is pinned to tag t2i26 with --epochs 14 and
# NONE of the certification flags. If it ever fired at this run it would relaunch with a DIFFERENT
# CONFIG -- warm_lr_scale 0.1 instead of 0.01, the 45/45 mixture instead of 30/60, and a warm start from
# t2final_best instead of RecVAE -- and the resumed epochs would be silently incomparable to the earlier
# ones. A checkpoint is only as good as the config that produced it, so the relaunch command here is
# byte-identical to the launch command, and the old watchdog must NOT be running at the same time.
#
# LAUNCH COMMAND OF RECORD (2026-07-31 22:38, commit 5f5d2dc):
#   python -u src/instrument/train_i26.py --init recvae --warm_lr_scale 0.01 \
#          --mix 0.30,0.60,0.05,0.05 --epochs 24 --tag t2i26c --steps_per_epoch 2200
#
# RESUME SEMANTICS: train_i26.py reads t2i26c_last.pt, which carries encoder + optimiser state + epoch +
# best + hist, and is written atomically at the END of every epoch. So a relaunch continues from the last
# completed epoch; the most that a crash can cost is the epoch in flight. --init recvae is harmless on a
# resume (the checkpoint is loaded over the init) but is passed anyway so the command stays identical.
#
# NEVER DELETES ANYTHING. --fresh is deliberately NOT passed: it would archive the run so far.
#
# Install (detached loop -- the primary mechanism; see the note at the bottom of the old watchdog about
# schtasks reporting LastResult=0 while demonstrably never writing its log):
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',
#     'C:\dev\phd\casper\src\instrument\i26c_watchdog.ps1','-Loop' -WindowStyle Hidden

param([switch]$Loop, [int]$IntervalSec = 300)
$ErrorActionPreference = 'SilentlyContinue'
$root   = 'C:\dev\phd\casper'
$out    = Join-Path $root 'experiments\instrument'
$done   = Join-Path $out  't2i26c_DONE.marker'
$state  = Join-Path $out  't2i26c_state.json'
$runlog = Join-Path $out  't2i26c_run.out'
$log    = Join-Path $out  'i26c_watchdog.log'
$py     = 'C:\Program Files\Python312\python.exe'
$script = Join-Path $root 'src\instrument\train_i26.py'

# 2700 s = 45 min. An epoch is ~18 min and its END-OF-EPOCH evaluation (canonical val full-profile,
# empty-set, k=1, plus two simulated interviews at k=2 and k=8) is several minutes during which the
# heartbeat does not advance. The threshold must comfortably exceed a slow epoch or the watchdog will
# kill-and-relaunch a perfectly healthy run and lose the epoch in flight.
$STALE_SEC = 2700

function Log($m) { "$([DateTime]::Now.ToString('s')) $m" | Out-File -FilePath $log -Append -Encoding utf8 }

function Check {
    if (Test-Path $done) { Log 'DONE marker present -> standing down'; return $true }

    $alive = $false
    if (Test-Path $state) {
        try {
            $s    = Get-Content $state -Raw | ConvertFrom-Json
            $cpid = [int]$s.pid
            $age  = [double][DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - [double]$s.heartbeat
            $proc = Get-Process -Id $cpid -ErrorAction SilentlyContinue
            if ($proc -and ($proc.ProcessName -like 'python*') -and ($age -lt $STALE_SEC)) {
                $alive = $true
                Log ("alive: pid=$cpid ep=" + $s.epoch + " step=" + $s.step + " best=" + $s.best_val + " hb_age=" + [int]$age + "s")
            } else {
                Log ("DEAD or STALE: pid=$cpid hb_age=" + [int]$age + "s proc_exists=" + ($null -ne $proc))
            }
        } catch { Log 'state file unreadable -> treating as dead'; $alive = $false }
    } else {
        Log 'no state file yet -> treating as dead'
    }

    if (-not $alive) {
        $env:OMP_NUM_THREADS = '6'
        $env:OPENBLAS_NUM_THREADS = '1'
        Log 'RELAUNCHING train_i26.py with the IDENTICAL certification config (resumes from t2i26c_last.pt)'
        Start-Process -FilePath $py `
            -ArgumentList @('-u', $script,
                            '--init', 'recvae',
                            '--warm_lr_scale', '0.01',
                            '--mix', '0.30,0.60,0.05,0.05',
                            '--epochs', '24',
                            '--tag', 't2i26c',
                            '--steps_per_epoch', '2200') `
            -WorkingDirectory $root -WindowStyle Hidden `
            -RedirectStandardOutput ($runlog + '.relaunch') -RedirectStandardError ($runlog + '.relaunch.err')
    }
    return $false
}

if ($Loop) {
    Log "watchdog LOOP started (pid=$PID, interval=${IntervalSec}s, stale=${STALE_SEC}s, tag=t2i26c)"
    while ($true) {
        if (Check) { Log 'loop exiting: DONE'; break }
        Start-Sleep -Seconds $IntervalSec
    }
} else {
    Check | Out-Null
}
