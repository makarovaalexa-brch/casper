# Waits for chain 1 to write CHAIN DONE, then starts chain 2. Detached; no compute of its own.
Set-Location "C:\dev\phd\casper"
$log = "experiments\baselines\interview\a11_a10_chain.out"
for ($i = 0; $i -lt 720; $i++) {          # up to 6 h at 30 s
    if ((Test-Path $log) -and ((Get-Content $log -Raw) -match "CHAIN DONE")) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File "C:\dev\phd\casper\scripts\run_a11_a10b.ps1"
        exit 0
    }
    Start-Sleep -Seconds 30
}
"[waiter] gave up after 6 h" | Out-File -Append $log
