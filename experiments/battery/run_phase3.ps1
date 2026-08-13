# WORKSTREAM E PHASE 3 (design sheet G.B) -- detached chain, ONE heavy job at a time.
#
# HARD RULE 11(d): nothing here launches until the Part F stage-2 watcher is gone. Pass its PID as
# -WaitPid and this script blocks on it first. The box is a 4-PHYSICAL-core i7-7820HQ reporting 8
# logical, so every step runs at --threads 6, not 8: on the night of 2026-08-13 eight compute threads
# on four physical cores cost ~2.5x. That number is a property of THIS box on THIS night.
param([int]$WaitPid = 0)
$ErrorActionPreference = "Continue"
Set-Location C:\dev\phd\casper
$py = "C:\Program Files\Python312\python.exe"
$B  = "experiments\battery"
$L  = "$B\phase3.log"
$NT = 6

function Say($m) { "$(Get-Date -Format HH:mm:ss) $m" | Out-File -Append $L }

if ($WaitPid -gt 0) {
    Say "=== waiting for the Part F chain (pid $WaitPid) ==="
    try { Wait-Process -Id $WaitPid -ErrorAction Stop } catch { }
}
# Belt and braces: refuse to start while ANY of the phase-2 jobs is still alive.
while ($true) {
    $busy = Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
            Where-Object { $_.CommandLine -match 'outcome_head|lift_head|policy_' }
    if (-not $busy) { break }
    Say "=== box still busy: $($busy.ProcessId -join ',') -- waiting 120s ==="
    Start-Sleep -Seconds 120
}
Say "=== BOX FREE. Phase 3 starts. ==="

# --- STEP 1: the exact measured-NDCG sweep at the 42 states one level below the floor. ~3.3 h -----
if (-not (Test-Path "$B\policy_labels_deep.json")) {
    Say "--- step 1: policy_labels.py --mode sweep --states deep (projection: 42 states, 41,834 evals, 3.3 h)"
    & $py src/instrument/policy_labels.py --mode sweep --states deep --label deep `
        --threads $NT *> $B\phase3_sweep.log
    Say "--- step 1 exit=$LASTEXITCODE"
} else { Say "--- step 1 SKIPPED: policy_labels_deep.json already on disk" }

# --- STEP 2: (leaf, branch) -> question. Zero scoring cost, all gates. ----------------------------
Say "--- step 2: policy_deep.py --mode labels"
& $py src/instrument/policy_deep.py --mode labels *> $B\phase3_labels.log
Say "--- step 2 exit=$LASTEXITCODE"

# --- STEP 3: ★ THE DEEPER-TREE CONTROL. Mandatory. -----------------------------------------------
Say "--- step 3: policy_deep.py --mode control (the deeper tree, its shuffle control, order-invariance)"
& $py src/instrument/policy_deep.py --mode control --threads $NT *> $B\phase3_control.log
Say "--- step 3 exit=$LASTEXITCODE"

# --- STEP 4: fine-tune the policy on the deep labels MIXED with the tree labels. ------------------
Say "--- step 4: policy_net.py --arm deep (fine-tune from the E1 anchor, replay on)"
& $py src/instrument/policy_net.py --arm deep --tag phase3 `
    --init_ckpt $B\policy_main_equivariant_both.pt --epochs 3 --lr 2e-4 `
    --threads $NT --i_have_author_goahead *> $B\phase3_deep.log
Say "--- step 4 exit=$LASTEXITCODE"

# --- STEP 5: the HARD RULE 5 control -- the same 42 questions, permuted across branches. ----------
Say "--- step 5: policy_net.py --arm deepshuffle (the shuffle control)"
& $py src/instrument/policy_net.py --arm deepshuffle --tag phase3 `
    --init_ckpt $B\policy_main_equivariant_both.pt --epochs 3 --lr 2e-4 `
    --threads $NT --i_have_author_goahead *> $B\phase3_deepshuffle.log
Say "--- step 5 exit=$LASTEXITCODE"

Say "=== PHASE 3 CHAIN COMPLETE ==="
