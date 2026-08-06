# TASK #13 launcher: wait out any live heavy job, then grow the tree, then run the two port cells.
# Detached on purpose (Start-Process) so it survives the agent session; every stage logs to a
# UNIQUELY LABELLED file so it cannot collide with another run's output (HARD RULE 11e).
param([int]$WaitPid = 0)

$ErrorActionPreference = "Stop"
Set-Location C:\dev\phd\casper
$py = "C:\Program Files\Python312\python.exe"
$log = "experiments\baselines\interview\gtport13.log"

if ($WaitPid -gt 0) {
  while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 }
  Add-Content $log "[launcher] PID $WaitPid exited $(Get-Date -Format o); box is free"
}
$os = Get-CimInstance Win32_OperatingSystem
Add-Content $log ("[launcher] free RAM {0} MB at start" -f [int]($os.FreePhysicalMemory/1KB))

# ---- STAGE 1: grow + checkpoint the tree (the only expensive, resumable step)
Add-Content $log "[launcher] STAGE 1 grow $(Get-Date -Format o)"
& $py scripts\baselines\golbandi_grow_ml25m.py --depth 8 `
    --out .cache\snap5\tree_ml25m_liang_d8.json *>> $log
Add-Content $log "[launcher] STAGE 1 exit=$LASTEXITCODE $(Get-Date -Format o)"
