# Waits for RAM to free up (the dense 18359^2 filter needs ~3GB peak), then runs the Turbo-CF
# val sweep on the canonical Liang ML-25M split. Launched detached so it survives the session.
$ErrorActionPreference = "Stop"
Set-Location C:\dev\phd\casper
$log = "C:\dev\phd\casper\experiments\baselines\ml25m_liang\turbocf_queued.log"
# 4GB is the working threshold: the dense filter is P (1.35GB) plus F (1.35GB) with the polynomial
# computed row-blocked, so ~2.7GB peak plus evaluation batches.
"[{0}] waiting for >=4GB free RAM" -f (Get-Date -Format "HH:mm:ss") | Out-File $log -Encoding utf8
while ($true) {
    $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    if ($free -ge 4) { break }
    Start-Sleep -Seconds 300
}
"[{0}] {1:N1}GB free -- launching turbocf sweep" -f (Get-Date -Format "HH:mm:ss"), $free |
    Out-File $log -Append -Encoding utf8
$env:OMP_NUM_THREADS = "6"
& "C:\Program Files\Python312\python.exe" src/baselines/run_queue2.py --only turbocf --max_minutes 180 *>&1 |
    Out-File $log -Append -Encoding utf8
"[{0}] done" -f (Get-Date -Format "HH:mm:ss") | Out-File $log -Append -Encoding utf8
