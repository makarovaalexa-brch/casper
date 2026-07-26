# Session-independent launcher for the concept-fold DISTILLATION single-seed POC.
# Runs concept_distill_train.py --poc detached (survives the agent session), split logs, and
# registers a schtasks watchdog that resumes-from-checkpoint on death (the --poc mode skips
# already-recorded configs). Heartbeat files: experiments/battery/<tag>.heartbeat;
# done marker: experiments/battery/concept_distill_train_DONE.marker.
param(
  [int]$Epochs = 5,
  [int]$Phase2Epochs = 3
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "experiments\battery\concept_distill_train.out"
$err = Join-Path $root "experiments\battery\concept_distill_train.err"
$py  = "python"
$argline = "-u src/instrument/concept_distill_train.py --poc --full_threads --epochs $Epochs --phase2_epochs $Phase2Epochs"

# launch detached
$p = Start-Process -FilePath $py -ArgumentList $argline -WorkingDirectory $root `
     -RedirectStandardOutput $out -RedirectStandardError $err -PassThru -WindowStyle Hidden
Write-Output "[launch] concept_distill POC pid=$($p.Id) epochs=$Epochs -> $out"

# watchdog: every 20 min, if no DONE.marker and no running python on the poc, relaunch (resume-safe)
$marker = Join-Path $root "experiments\battery\concept_distill_train_DONE.marker"
$wd = @"
`$marker = '$marker'
if (Test-Path `$marker) { schtasks /Delete /TN CasperDistillWatchdog /F 2>`$null; exit }
`$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { `$_.CommandLine -like '*concept_distill_train.py --poc*' }
if (-not `$running) {
  Start-Process -FilePath '$py' -ArgumentList '$argline' -WorkingDirectory '$root' ``
    -RedirectStandardOutput '$out.resume' -RedirectStandardError '$err.resume' -WindowStyle Hidden
}
"@
$wdPath = Join-Path $root "scripts\_distill_watchdog.ps1"
Set-Content -Path $wdPath -Value $wd -Encoding UTF8
schtasks /Create /TN CasperDistillWatchdog /TR "powershell -NonInteractive -ExecutionPolicy Bypass -File `"$wdPath`"" /SC MINUTE /MO 20 /F | Out-Null
Write-Output "[watchdog] CasperDistillWatchdog registered (20 min; auto-deletes on DONE marker)"
