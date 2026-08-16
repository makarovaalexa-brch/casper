# PART K PILOT (design sheet docs/design/PAPER_B_POLICY_DESIGN.md K.4) -- the chain, BUILT NOT LAUNCHED.
#
# ★ NOTHING IN THIS FILE HAS BEEN RUN. The pilot runs on the author's explicit go and nothing else in
# Part K runs without one. Every python step below carries --i_have_author_goahead precisely so that
# an accidental invocation of a step does nothing.
#
# THE CHAIN
#   0  selftests            pure-tensor / tiny-synthetic. No data, no environment, no fit.
#   1  labels (pilot)       1,000 FIT users, one state each, all 1,000 candidates    ~27 min
#   2  composition check    the mandatory K.4 print + the .f32 record arithmetic     ~seconds
#   3  train                warm start from E1, listwise soft targets + replay       ~minutes
#   4  shuffle control      HARD RULE 5. Same data, association destroyed            ~minutes
#   5  holdout evaluation   K.5 gates on the 2,000 HOLDOUT users                     ~20-40 min
#
# HARD RULE 11(d): ONE heavy job on the box. The box is a 4-PHYSICAL-core i7-7820HQ reporting 8
# logical, so every step runs at --threads 6, not 8: on 2026-08-13 eight compute threads on four
# physical cores cost ~2.5x. That number is a property of THIS box on THIS night, not of any method.
#
# LOG ROTATION (the phase-3 lesson, `verify-the-artifact-not-a-proxy`): a dead traceback at the tail
# of a log gets read as a live one. Every start ARCHIVES the previous attempt's step logs into
# pu_attic_prerestart/ and stamps a visible === CHAIN START === boundary, so a reader tailing a log
# can never mistake a previous attempt's failure for this one's.
param([int]$WaitPid = 0)
$ErrorActionPreference = "Continue"
Set-Location C:\dev\phd\casper
$py = "C:\Program Files\Python312\python.exe"
$B  = "experiments\battery"
$L  = "$B\pu_pilot.log"
$NT = 6

function Say($m) { "$(Get-Date -Format HH:mm:ss) $m" | Out-File -Append $L }

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$attic = "$B\pu_attic_prerestart"
New-Item -ItemType Directory -Force -Path $attic | Out-Null
foreach ($n in @("pu_pilot","pu_selftest","pu_labels","pu_compose","pu_train","pu_shuffle","pu_eval")) {
    if (Test-Path "$B\$n.log") { Move-Item "$B\$n.log" "$attic\$n.PRIOR_$stamp.log" -Force }
}
"=== CHAIN START $stamp -- any earlier step logs are in pu_attic_prerestart/ ===" |
    Out-File -Append $L

# A failed step ABORTS the chain. On 2026-08-14 a step-1 refusal cascaded through steps 2-5, each
# failing on the artifact the previous one never produced, and buried the real error five logs deep.
function Need($code, $what) {
    if ($code -ne 0) { Say "=== ABORT: $what exited $code. Later steps depend on it and are NOT run. ==="; exit 1 }
}

if ($WaitPid -gt 0) {
    Say "=== waiting for pid $WaitPid ==="
    try { Wait-Process -Id $WaitPid -ErrorAction Stop } catch { }
}
# HARD RULE 11(d)+(e): refuse to start while another instrument job is alive, and never let two runs
# share an output path.
while ($true) {
    $busy = Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
            Where-Object { $_.CommandLine -match 'policy_|pu_labels|pu_policy|lift_head|outcome_head|value_dataset' }
    if (-not $busy) { break }
    Say "=== box still busy: $($busy.ProcessId -join ',') -- waiting 120s ==="
    Start-Sleep -Seconds 120
}
Say "=== BOX FREE. Part K pilot starts. ==="

# --- STEP 0: the selftests. Pure. If these do not pass, nothing below is worth running. -----------
Say "--- step 0: selftests (pure-tensor / tiny-synthetic; no data, no environment, no fit)"
& $py src/instrument/pu_labels.py --selftest  *> $B\pu_selftest.log
Need $LASTEXITCODE "pu_labels --selftest"
& $py src/instrument/pu_policy.py --selftest  *>> $B\pu_selftest.log
Say "--- step 0 exit=$LASTEXITCODE"; Need $LASTEXITCODE "pu_policy --selftest"

# --- the cost projection, printed into the chain log before a minute is spent --------------------
& $py src/instrument/pu_labels.py --cost_only *>> $L

# --- STEP 1: the labelling pass. 1,000 FIT users x 1,000 candidates. ~27 min. --------------------
# Resumable: a kill costs ONE candidate. A resume across a changed contract is REFUSED, not merged.
Say "--- step 1: pu_labels.py --pilot (projection: 1,001 evals at ~1.66 s/eval = ~27 min)"
& $py src/instrument/pu_labels.py --pilot --threads $NT --i_have_author_goahead *> $B\pu_labels.log
Say "--- step 1 exit=$LASTEXITCODE"; Need $LASTEXITCODE "the pilot labelling pass"

# --- STEP 2: the composition check. Mandatory (K.4), and it also checks the record arithmetic. ----
Say "--- step 2: pu_labels.py --mode compose (state-type mix, prefix histogram, per-user nnz)"
& $py src/instrument/pu_labels.py --mode compose *> $B\pu_compose.log
Say "--- step 2 exit=$LASTEXITCODE"; Need $LASTEXITCODE "the composition check"

# --- STEP 3: the trainer. Warm start from E1, listwise soft targets + tree-label replay. ---------
Say "--- step 3: pu_policy.py --mode fit --arm main"
& $py src/instrument/pu_policy.py --mode fit --arm main --tag pilot `
    --threads $NT --i_have_author_goahead *> $B\pu_train.log
Say "--- step 3 exit=$LASTEXITCODE"; Need $LASTEXITCODE "the fit"

# --- STEP 4: ★ THE CONTROL (HARD RULE 5). Same gains, association destroyed. Not optional. -------
Say "--- step 4: pu_policy.py --mode fit --arm shuffle (the HARD RULE 5 control)"
& $py src/instrument/pu_policy.py --mode fit --arm shuffle --tag pilot `
    --threads $NT --i_have_author_goahead *> $B\pu_shuffle.log
Say "--- step 4 exit=$LASTEXITCODE"; Need $LASTEXITCODE "the shuffle control"

# --- STEP 5: the K.5 gates, on the 2,000 HOLDOUT users. Never labelled, never trained on. --------
Say "--- step 5: pu_policy.py --mode evaluate (spanning, behavioural win, redundancy, anti-collapse)"
& $py src/instrument/pu_policy.py --mode evaluate --tag pilot `
    --ckpt $B\pu_labels\pu_main_pilot.pt --threads $NT --i_have_author_goahead *> $B\pu_eval.log
Say "--- step 5 exit=$LASTEXITCODE"; Need $LASTEXITCODE "the holdout evaluation"

Say "=== PART K PILOT CHAIN COMPLETE ==="
Say "=== READ, IN THIS ORDER: pu_compose.log (composition), pu_train.log vs pu_shuffle.log (does"
Say "=== the trained arm beat the control on held-out FIT states?), then pu_gates_pilot.json."
Say "=== If there is no trainable signal, the design is DEAD at pilot price and is recorded dead (K.4)."
