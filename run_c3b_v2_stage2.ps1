# C3B_V2 stage 2 -- 4 arms x 2 decodings over the 3,880 held-out questions.
# Design sheet: docs/design/DS_C3B_V2_QUESTIONS.md (PRE-REGISTERED 2026-08-30, BINDING).
# Detached on purpose: session background bash has been reaped twice.
#
# The 2026-08-30 14:05 attempt was killed after ~4 h with no arm finished, for two reasons, both
# now fixed in src/instrument/c3b_adapter.py + c4c_decoding.py:
#   1. the decoder re-ran the whole prefix every token (no KV cache), so cost was quadratic in the
#      32-token questions budget and best-of-8 paid it eight times per point. With the cache a
#      best-of-8 pass measures 31.6 rows/min => ~2.0 h/arm (~11x faster at gen_batch 64), and the
#      cached decoder is CHECKED byte-identical to the uncached one before the run proceeds;
#   2. it banked only at the very end. Every (decoding, arm) generation set and every SBERT
#      selection set is now written to its own contract-signed sidecar under
#      experiments/paperC/c3b_v2/gen/ the moment it completes, and resumed on an exact contract
#      match -- so an interruption (this box has gone to battery-sleep twice in ten days) costs at
#      most the one unit in flight. Re-running this script IS the resume.
#
# Nothing about what is measured changes: same adapters (true ep5 / scrambled ep5), same 4 arms,
# best-of-8 by LM likelihood primary with greedy beside it, same readings 1-5.
Set-Location $PSScriptRoot
$log = "experiments/paperC/c3b_v2/stage2_run.log"
Add-Content $log "[$(Get-Date -Format o)] STAGE2 START pid=$PID rev=$(git rev-parse HEAD)"
& "C:\Program Files\Python312\python.exe" src/instrument/c3b_adapter.py stage2 `
    --targets questions --threads 4 --gen_batch 64 *>> $log
Add-Content $log "[$(Get-Date -Format o)] STAGE2 EXIT code=$LASTEXITCODE"
