#!/usr/bin/env bash
# Autonomous ML-25M I2 post-training driver: wait for RecVAE [saved], then run W2 + gates.
set -u
cd /c/dev/phd/casper
LOG=.cache/instrument2/ml25m_recvae_d512_run.log
echo "[driver] waiting for training [saved] marker..."
until grep -q "\[saved\]" "$LOG" 2>/dev/null; do
  if ! tasklist 2>/dev/null | grep -qi python.exe; then
    if ! grep -q "\[saved\]" "$LOG" 2>/dev/null; then
      echo "[driver] python exited before [saved]; aborting W2/gates"; exit 2
    fi
  fi
  sleep 60
done
echo "[driver] training done. running W2..."
python scripts/instrument2/ml25m_w2.py --eta 16 > .cache/instrument2/ml25m_w2_run.log 2>&1
echo "[driver] W2 exit $?. running gates..."
python scripts/instrument2/ml25m_gates.py > .cache/instrument2/ml25m_gates_run.log 2>&1
echo "[driver] gates exit $?. DONE_ALL"
