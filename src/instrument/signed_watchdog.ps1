# signed_watchdog.ps1 -- Task-Scheduler watchdog for the SIGNED overnight queue (15 min, zero-token).
$root  = 'C:\dev\phd\casper'
$state = "$root\experiments\battery\signed_queue_state.json"
$wdlog = "$root\experiments\battery\signed_watchdog.log"
if (-not (Test-Path $state)) { exit 0 }
try { $s = Get-Content $state -Raw | ConvertFrom-Json } catch { exit 0 }
if ($s.stage -eq 'done') { exit 0 }
$alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
         Where-Object { $_.CommandLine -match 'signed_queue' }
if ($alive) { exit 0 }
Add-Content $wdlog "$(Get-Date -Format s) signed runner DEAD at stage $($s.stage) -> relaunch"
Start-Process -FilePath 'C:\Program Files\Python312\python.exe' `
    -ArgumentList '-u', "$root\src\instrument\signed_queue.py" `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$root\experiments\battery\signed_queue_wd.log" `
    -RedirectStandardError  "$root\experiments\battery\signed_queue_wd.err" `
    -WindowStyle Hidden
