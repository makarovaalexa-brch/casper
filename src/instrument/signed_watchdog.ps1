# signed_watchdog.ps1 -- Task-Scheduler watchdog for the SIGNED overnight queue (15 min, zero-token).
# 2026-07-25 hardening (incident: scheduled runs 11:47-12:32 never revived the dead runner while a
# manual run did): relaunch on (no process match) OR (heartbeat older than 20 min). The runner now
# carries a single-instance lock, so an over-eager relaunch is safe (the duplicate exits itself).
$root  = 'C:\dev\phd\casper'
$state = "$root\experiments\battery\signed_queue_state.json"
$wdlog = "$root\experiments\battery\signed_watchdog.log"
if (-not (Test-Path $state)) { exit 0 }
try { $s = Get-Content $state -Raw | ConvertFrom-Json } catch { exit 0 }
if ($s.stage -eq 'done') { exit 0 }
$now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$hbAge = $now - [long]$s.heartbeat
$alive = $null
try {
    $alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop |
             Where-Object { $_.CommandLine -match 'signed_queue' }
} catch { $alive = $null }
if ($alive -and $hbAge -lt 1200) { exit 0 }
Add-Content $wdlog "$(Get-Date -Format s) runner dead-or-stale (hbAge=${hbAge}s, procMatch=$([bool]$alive)) at stage $($s.stage) -> relaunch"
Start-Process -FilePath 'C:\Program Files\Python312\python.exe' `
    -ArgumentList '-u', "$root\src\instrument\signed_queue.py" `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput "$root\experiments\battery\signed_queue_wdout.log" `
    -RedirectStandardError  "$root\experiments\battery\signed_queue_wdout.err"
