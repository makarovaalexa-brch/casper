# Waits for the A11 harness process to exit, then runs the A10 extension. No compute of its own.
Set-Location "C:\dev\phd\casper"
for ($i = 0; $i -lt 480; $i++) {
    $busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
              Where-Object { $_.CommandLine -and $_.CommandLine.Contains("adaptive_harness") })
    if ($busy.Count -eq 0) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File "C:\dev\phd\casper\scripts\run_a10_ext.ps1"
        exit 0
    }
    Start-Sleep -Seconds 30
}
