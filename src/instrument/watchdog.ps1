# watchdog.ps1 -- zero-token schtasks watchdog for the answer-decomposition chain (author overnight
# directive 2026-07-25). Every ~15 min: if not DONE and the chain is not alive (dead process OR stale
# heartbeat), relaunch chain_runner.py detached. It RESUMES via artifact checks; never double-launches
# a live chain (chain_runner has its own singleton guard as a second line of defense).
$ErrorActionPreference = 'SilentlyContinue'
$root  = 'C:\dev\phd\casper'
$bat   = Join-Path $root 'experiments\battery'
$done  = Join-Path $bat 'chain_DONE.marker'
$state = Join-Path $bat 'chain_state.json'
$runner = Join-Path $root 'src\instrument\chain_runner.py'
$log   = Join-Path $bat 'watchdog.log'
function Log($m) { "$([DateTime]::Now.ToString('s')) $m" | Out-File -FilePath $log -Append -Encoding utf8 }

if (Test-Path $done) { Log 'DONE marker present -> nothing to do'; exit 0 }

$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = 'python' }

$alive = $false
if (Test-Path $state) {
    try {
        $s = Get-Content $state -Raw | ConvertFrom-Json
        $cpid = [int]$s.pid
        $hb = [double]$s.heartbeat
        $now = [double][DateTimeOffset]::UtcNow.ToUnixTimeSeconds()   # true UTC unix (PS %s is local-buggy)
        $age = $now - $hb
        $proc = Get-Process -Id $cpid -ErrorAction SilentlyContinue
        if ($proc -and ($proc.ProcessName -like 'python*') -and ($age -lt 1200)) { $alive = $true }
    } catch { $alive = $false }
}

if ($alive) {
    Log "chain alive (pid $cpid, heartbeat ${age}s) -> ok"
} else {
    Log 'chain NOT alive -> relaunching chain_runner (resume via artifacts)'
    Start-Process -FilePath $py -ArgumentList '-u', $runner `
        -RedirectStandardOutput (Join-Path $bat 'chain_runner_wd.out') `
        -RedirectStandardError  (Join-Path $bat 'chain_runner_wd.err') `
        -WindowStyle Hidden -WorkingDirectory $root
}
