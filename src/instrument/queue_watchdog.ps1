# queue_watchdog.ps1 -- Task-Scheduler watchdog for the overnight queue runner (every 15 min,
# zero-token). If queue_state.json exists, stage != done, and no python process matches
# 'queue_runner' -> relaunch the runner (it resumes from the state file). Restarts logged.
$root  = 'C:\dev\phd\casper'
$state = "$root\experiments\battery\queue_state.json"
$wdlog = "$root\experiments\battery\watchdog.log"
if (-not (Test-Path $state)) { exit 0 }                       # queue never started
try { $s = Get-Content $state -Raw | ConvertFrom-Json } catch { exit 0 }
if ($s.stage -eq 'done') { exit 0 }
$alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
         Where-Object { $_.CommandLine -match 'queue_runner' }
if ($alive) { exit 0 }
Add-Content $wdlog "$(Get-Date -Format s) runner DEAD at stage $($s.stage) (hb age $([int]((Get-Date)-(Get-Date '1970-01-01').AddSeconds($s.heartbeat).ToLocalTime()).TotalSeconds)s) -> relaunch"
Start-Process -FilePath 'C:\Program Files\Python312\python.exe' `
    -ArgumentList '-u', "$root\src\instrument\queue_runner.py" `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$root\experiments\battery\queue_runner_wd.log" `
    -RedirectStandardError  "$root\experiments\battery\queue_runner_wd.err" `
    -WindowStyle Hidden
