# L2a chain runner (Part L.2). One heavy stage at a time, in order, stopping on the first failure.
# Validation only. Every stage stamps test_touched=false.
$py = "C:\Program Files\Python312\python.exe"
$root = "C:\dev\phd\casper"
Set-Location $root
$stages = @(
  @{name = "composition"; args = @("--mode", "composition") },
  @{name = "states";      args = @("--mode", "states") },
  @{name = "fit_main";    args = @("--mode", "fit", "--arm", "main") },
  @{name = "eval_main";   args = @("--mode", "evaluate", "--arm", "main") },
  @{name = "fit_shuffle"; args = @("--mode", "fit", "--arm", "shuffle") },
  @{name = "eval_shuffle"; args = @("--mode", "evaluate", "--arm", "shuffle") }
)
foreach ($s in $stages) {
  $log = "experiments/battery/forest_$($s.name).log"
  $err = "experiments/battery/forest_$($s.name).err"
  $a = $s.args + @("--threads", "6", "--i_have_author_goahead")
  Write-Output "=== stage $($s.name) starting $(Get-Date -Format 'HH:mm:ss')"
  $p = Start-Process -FilePath $py -ArgumentList (@("src/instrument/forest_policy.py") + $a) `
    -WorkingDirectory $root -RedirectStandardOutput $log -RedirectStandardError $err `
    -NoNewWindow -PassThru -Wait
  if ($p.ExitCode -ne 0) {
    Write-Output "=== stage $($s.name) FAILED with exit code $($p.ExitCode); chain stops here"
    break
  }
  Write-Output "=== stage $($s.name) done $(Get-Date -Format 'HH:mm:ss')"
}
