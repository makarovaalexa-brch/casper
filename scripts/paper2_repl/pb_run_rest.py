"""pb_run_rest.py -- detached orchestrator for the Paper B faithful single-run replication.
Chains: wait-for-prep -> pb_train_enc (val-selected best ckpt) -> pb_interview (curves + controls).
Session-independent (launched via Start-Process). Writes split logs + a done marker.

Steps:
  1. wait until .cache/paper2_repl/pb_data.npz exists AND is size-stable; abort if it never appears.
  2. leak check: fold-in intersect target == empty for every test user (structural, but verified).
  3. pb_train_enc.py  (EP_MAX epochs, best-val checkpoint on the 10k val cohort -> enc.pt/Qp.npy/Ec.npy)
  4. pb_interview.py   (T=8 item+concept, static-pop + EIG, geometric answers, full+tail NDCG@10 on 10k
                        test users; canonical-snap full-profile + answer-permutation control inside it)
  5. write DONE marker with status.
"""
import os, sys, time, subprocess, numpy as np

ROOT = 'C:/dev/phd/casper'; OUT = f'{ROOT}/.cache/paper2_repl'; RESD = f'{ROOT}/experiments/paper2_repl'
PY = sys.executable
os.makedirs(RESD, exist_ok=True)
LOG = open(f'{RESD}/pb_run_rest.log', 'a', buffering=1, encoding='utf-8')
def log(m): LOG.write(f"[{time.strftime('%H:%M:%S')}] {m}\n"); LOG.flush()

NPZ = f'{OUT}/pb_data.npz'
DONE = f'{RESD}/pb_ORCH_DONE.txt'
try:
    os.remove(DONE)
except OSError:
    pass

# ---------------------------------------------------------------- 1. wait for prep
log("orchestrator start; waiting for pb_data.npz")
deadline = time.time() + 3 * 3600
last = -1
while True:
    if os.path.exists(NPZ):
        sz = os.path.getsize(NPZ)
        if sz > 0 and sz == last:   # size-stable => fully written
            log(f"pb_data.npz present and stable ({sz} bytes)"); break
        last = sz
    if time.time() > deadline:
        log("ABORT: pb_data.npz never appeared within 3h"); open(DONE, 'w').write("ABORT: no npz\n"); sys.exit(1)
    time.sleep(15)

# ---------------------------------------------------------------- 2. leak check
Z = np.load(NPZ)
fu, fs = Z['foldin_uid'], Z['foldin_sid']; tu, ts = Z['target_uid'], Z['target_sid']
fold = {}; tgt = {}
for u, s in zip(fu, fs): fold.setdefault(int(u), set()).add(int(s))
for u, s in zip(tu, ts): tgt.setdefault(int(u), set()).add(int(s))
overlap = sum(len(fold.get(u, set()) & tgt.get(u, set())) for u in tgt)
log(f"LEAK CHECK: total fold-in/target overlaps across test users = {overlap} (must be 0)")
if overlap != 0:
    log("ABORT: fold-in/target leak detected"); open(DONE, 'w').write(f"ABORT: leak {overlap}\n"); sys.exit(1)
log(f"leak check OK; n_test_fold={len(fold)} n_test_tgt={len(tgt)} nc={int(Z['Ac'].shape[0])} ni={int(Z['ni'])}")

# ---------------------------------------------------------------- 3. train encoder (best-val ckpt)
env = dict(os.environ)
env['PYTHONIOENCODING'] = 'utf-8'; env['PYTHONUNBUFFERED'] = '1'
env['EP_CHUNK'] = env.get('EP_MAX', '12'); env['EP_MAX'] = env.get('EP_MAX', '12')
log(f"launching pb_train_enc.py (EP_MAX={env['EP_MAX']})")
with open(f'{RESD}/pb_train_enc.out', 'w', buffering=1) as o, open(f'{RESD}/pb_train_enc.err', 'w', buffering=1) as e:
    r = subprocess.run([PY, f'{ROOT}/scripts/paper2_repl/pb_train_enc.py'], stdout=o, stderr=e, env=env, cwd=ROOT)
if r.returncode != 0 or not os.path.exists(f'{OUT}/enc.pt'):
    log(f"ABORT: pb_train_enc failed rc={r.returncode} (enc.pt exists={os.path.exists(f'{OUT}/enc.pt')})")
    open(DONE, 'w').write(f"ABORT: train rc={r.returncode}\n"); sys.exit(1)
log("pb_train_enc DONE; enc.pt/Qp.npy/Ec.npy written")

# ---------------------------------------------------------------- 4. interview + controls
log("launching pb_interview.py (T=8, item+concept, static-pop + EIG, controls)")
with open(f'{RESD}/pb_interview.out', 'w', buffering=1) as o, open(f'{RESD}/pb_interview.err', 'w', buffering=1) as e:
    r = subprocess.run([PY, f'{ROOT}/scripts/paper2_repl/pb_interview.py'], stdout=o, stderr=e, env=env, cwd=ROOT)
if r.returncode != 0 or not os.path.exists(f'{RESD}/pb_results.json'):
    log(f"ABORT: pb_interview failed rc={r.returncode}")
    open(DONE, 'w').write(f"ABORT: interview rc={r.returncode}\n"); sys.exit(1)

log("pb_interview DONE; pb_results.json written")
open(DONE, 'w').write("OK\n")
log("ORCHESTRATOR COMPLETE")
