# Session-independent launcher for the overnight distillation DOWNSTREAM chain (D1/D2/D3).
# Waits on the POC done-marker (concept_distill_train_DONE.marker), selects the best distilled fold,
# runs D1->D2->D3 idempotently, split logs, state + heartbeat + done-marker, watchdog resume-on-death.
param(
  [string]$WaitMarker = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $WaitMarker) { $WaitMarker = Join-Path $root "experiments\battery\concept_distill_train_DONE.marker" }
$out = Join-Path $root "experiments\battery\distill_chain.out"
$err = Join-Path $root "experiments\battery\distill_chain.err"
$py  = "python"
$argline = "-u src/instrument/distill_downstream_chain.py --run --full_threads --wait_marker `"$WaitMarker`""

$p = Start-Process -FilePath $py -ArgumentList $argline -WorkingDirectory $root `
     -RedirectStandardOutput $out -RedirectStandardError $err -PassThru -WindowStyle Hidden
Write-Output "[launch] distill chain pid=$($p.Id) wait_marker=$WaitMarker -> $out"

$marker = Join-Path $root "experiments\battery\distill_chain_DONE.marker"
$wd = @"
`$marker = '$marker'
if (Test-Path `$marker) { schtasks /Delete /TN CasperDistillChainWatchdog /F 2>`$null; exit }
`$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { `$_.CommandLine -like '*distill_downstream_chain.py --run*' }
if (-not `$running) {
  Start-Process -FilePath '$py' -ArgumentList '$argline' -WorkingDirectory '$root' ``
    -RedirectStandardOutput '$out.resume' -RedirectStandardError '$err.resume' -WindowStyle Hidden
}
"@
$wdPath = Join-Path $root "scripts\_distill_chain_watchdog.ps1"
Set-Content -Path $wdPath -Value $wd -Encoding UTF8
schtasks /Create /TN CasperDistillChainWatchdog /TR "powershell -NonInteractive -ExecutionPolicy Bypass -File `"$wdPath`"" /SC MINUTE /MO 20 /F | Out-Null
Write-Output "[watchdog] CasperDistillChainWatchdog registered (20 min; auto-deletes on DONE marker)"
