# run_goodreads_tower.ps1 -- the i26 tower on the Goodreads arena, launch command of record.
#
# WHY A SCRIPT AND NOT A COMMAND IN A LOG. The i26c watchdog's own header records the lesson: a
# relaunch that differs from the launch produces epochs that are silently incomparable to the earlier
# ones, and "a checkpoint is only as good as the config that produced it". So the command lives here,
# once, and both the launch and any resume use it.
#
# LAUNCH COMMAND OF RECORD for the ML-25M certified tower (t2i26c, from i26c_watchdog.ps1):
#     python -u src/instrument/train_i26.py --init recvae --warm_lr_scale 0.01 `
#            --mix 0.25,0.30,0.35,0.05,0.05 --epochs 60 --tag t2i26c --steps_per_epoch 800
#
# ONE DELIBERATE DIFFERENCE FROM THAT LINE, and it is a correction, not a change of recipe:
#   --warm_lr_scale is NOT passed, so it takes its 0.1 default. The watchdog comment says 0.01, but
#   the certified run's OWN training log records warm_lr_scale=0.1 at every resume segment
#   (experiments/instrument/t2i26c_train.log:7 and t2i26c_run.out:9,110,142,209). The log is primary
#   evidence of what actually ran; the comment is not. Replicating the comment would NOT replicate
#   the certified tower.
#
# THE OTHER DIFFERENCES ARE NAMESPACE ONLY:
#   --tag t2i26g          so Goodreads checkpoints never collide with ML-25M's t2i26c
#   CASPER_DATASET        selects the Goodreads proc / raw / recvae / genome paths in train_tower_t2
#
# RESUME SEMANTICS: train_i26.py writes <tag>_last.pt atomically at the end of every epoch, carrying
# encoder + optimiser + epoch + best + history, so a relaunch continues from the last completed
# epoch and a crash costs at most the epoch in flight. --fresh is deliberately NOT passed: it would
# archive the run so far. NEVER DELETES ANYTHING.
#
#   powershell -File src/instrument/run_goodreads_tower.ps1

$ErrorActionPreference = "Continue"
Set-Location "C:\dev\phd\casper"
$env:CASPER_DATASET = "goodreads"
$env:OMP_NUM_THREADS = "4"

$out = "experiments\goodreads\t2i26g_run.out"
$err = "experiments\goodreads\t2i26g_run.err"
New-Item -ItemType Directory -Force "experiments\goodreads" | Out-Null

# refuse to start beside another heavy job (HARD RULE 11d)
$busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match "train_i26|a20_goodreads_recvae|recvae" }
if ($busy) {
    "REFUSING TO LAUNCH: a heavy python job is already running -"
    $busy | Select-Object ProcessId, CommandLine | Format-List
    exit 1
}
$free = [Math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB, 2)
"[launch] free RAM ${free} GB; CASPER_DATASET=$($env:CASPER_DATASET)"

$p = Start-Process -FilePath "python" -ArgumentList @(
        "-u", "src/instrument/train_i26.py",
        "--init", "recvae",
        "--mix", "0.25,0.30,0.35,0.05,0.05",
        "--epochs", "60",
        "--tag", "t2i26g",
        "--steps_per_epoch", "800"
     ) -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
"[launch] PID=$($p.Id)  ->  $out"
