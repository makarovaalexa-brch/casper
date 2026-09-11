# run_goodreads_concept.ps1 -- the concept channel on Goodreads, launch command of record.
#
# RECIPE SOURCE. The certified ML-25M concept channel is `concept_cv3_ep08.pt`. There is no v3
# script: cv3 was produced by concept_train_v2.py with a non-default mixture, recoverable only from
# its own training log, experiments/instrument/cv3_train.out:1 --
#     [cv3] mixture interview=0.55 dense=0.3 kc0=0.05 kc1=0.1
# `--mix` for THIS script takes FOUR weights: p_interview,p_dense,p_kc0,p_kc1 -> 0.55,0.3,0.05,0.1.
# NOTE the argparse help string in concept_train_v2.py is STALE: it documents five weights
# ("p_broad,p_dense,p_interview,p_kc0,p_kc1"), which is the TOWER's interview_curriculum.set_mixture
# signature, not the concept channel's concept_curriculum_v2.set_mixture(p_interview, p_dense,
# p_kc0, p_kc1). Passing five raises TypeError -- which is how this was caught. Everything else is
# concept_train_v2's own
# default: hidden 256, epochs 12, steps_per_epoch 400, batch 256, lr 1e-3, seed 98765, keep_meh off.
# The certified checkpoint was epoch 8 of that 12-epoch run.
#
# DIFFERENCES FROM THE ML-25M LINE, both namespace only:
#   --tag cvg              so Goodreads checkpoints cannot collide with cv3
#   --snapshot ...t2i26g_ep53.pt   the tower selected by the pre-registered rule on THIS dataset
#   CASPER_DATASET         selects the Goodreads proc / raw / genome paths
#
#   powershell -File src/instrument/run_goodreads_concept.ps1

$ErrorActionPreference = "Continue"
Set-Location "C:\dev\phd\casper"
$env:CASPER_DATASET = "goodreads"
$env:OMP_NUM_THREADS = "4"

$snap = ".cache\instrument\t2i26g_ep53.pt"
if (-not (Test-Path $snap)) { "MISSING TOWER: $snap"; exit 1 }

# refuse to start beside another heavy job (HARD RULE 11d)
$busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match "train_i26|concept_train|recvae|run_battery" }
if ($busy) { "REFUSING TO LAUNCH: heavy python job already running"; $busy | Select-Object ProcessId, CommandLine | Format-List; exit 1 }
$free = [Math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB, 2)
"[launch] free RAM ${free} GB; snapshot $snap"

$p = Start-Process -FilePath "python" -ArgumentList @(
        "-u", "src/instrument/concept_train_v2.py",
        "--tag", "cvg",
        "--snapshot", $snap,
        "--mix", "0.55,0.3,0.05,0.1"
     ) -RedirectStandardOutput "experiments\goodreads\cvg_train.out" `
       -RedirectStandardError "experiments\goodreads\cvg_train.err" -WindowStyle Hidden -PassThru
"[launch] PID=$($p.Id)  ->  experiments\goodreads\cvg_train.out"
