"""
DECISIVE DIAGNOSTIC + PROTOTYPE: distill a clairvoyant oracle into a policy
that sees only the realizable belief state (privileged-information / asymmetric
imitation).

WHY: heuristics + PPO plateau at ~0.72 AUAC (held-out) while a clairvoyant
attribute-routing oracle reaches ~0.796. Question: is that gap REALIZABLE?
If the oracle's per-user action is predictable from the observable belief
state, a distilled student should approach 0.796 -> the adaptive edge is
capturable (we have a method). If the student stalls at ~0.72, the edge is
information-limited (the observable state cannot support the routing) -> we
prove an impossibility / pivot.

Teacher: per-turn pick the ATTRIBUTE maximizing held-out accuracy under the
instrument given the user's TRUE answers (privileged). Student: MLP on the
env's dual-belief state, trained by masked cross-entropy on teacher actions
(behavior cloning). Reports: teacher AUAC, student AUAC, action-match top-1,
greedy-info-gain action-match (predictability baseline), student adaptivity.

Usage: DATASET_NAME=ml_stratified poetry run python scripts/paper2/oracle_distill.py
"""
import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1'); sys.path.insert(0, 'scripts/paper2')
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from test_instrument_lib import load_instrument_by_name
from env import ElicitationEnv

NAME = os.environ.get('DATASET_NAME', 'ml_stratified')
NPZ = {'ml_stratified': 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz',
       'yelp_multicity': 'C:/dev/phd/casper/data/yelp/yelp_multicity_profiles.npz'}[NAME]
N_TURNS = 15
N_TRAIN = int(os.environ.get('DISTILL_TRAIN_USERS', 800))
N_TEST = 300
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
OUT = f'C:/dev/phd/casper/experiments/paper2/distill_{NAME}.pt'


def make_env(w):
    belief, _ = load_instrument_by_name(f'instrument_{NAME}')
    return ElicitationEnv(w, w.n_items if hasattr(w, 'n_items') else None, N_TURNS,
                          belief_instrument=belief, state_mode='dual', heldout=True)


def oracle_action(env, prof, nm, attr_lo):
    """Clairvoyant: attribute maximizing held-out accuracy given true answers."""
    asked = env.asked
    cands = [q for q in range(attr_lo, env.n_items) if q not in asked]
    if not cands:
        cands = [q for q in range(env.n_items) if q not in asked]
    sets = []
    for q in cands:
        v = prof[q]
        sets.append(env.revealed + ([(q, 1.0 if v >= 0.5 else 0.0)] if not np.isnan(v) else []))
    preds = env.instrument.predict_batch(sets)
    gt = prof[:nm]; base = ~np.isnan(gt)
    excl_base = np.array([q for q in asked if q < nm], dtype=int)
    filt = base.copy()
    if len(excl_base): filt[excl_base] = False
    y = gt[filt]
    if filt.sum() == 0:
        return cands[0]
    acc = ((preds[:, filt] > 0.5) == y).mean(1)
    return cands[int(np.argmax(acc))]


def _entropy(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def greedy_action(env, nm, attr_lo, p_rated):
    """Realizable myopic info-gain over ATTRIBUTES (expected entropy reduction
    under the instrument's beliefs; no true labels). Fast (attr-only)."""
    asked = env.asked
    cands = [q for q in range(attr_lo, env.n_items) if q not in asked]
    if not cands:
        cands = [q for q in range(env.n_items) if q not in asked]
    cur = env.meas
    excl = np.array([q for q in asked if q < nm], dtype=int)
    base = np.ones(nm, bool)
    if len(excl):
        base[excl] = False
    h_now = _entropy(cur[:nm])[base].sum()
    sets = []
    for e in cands:
        sets.append(env.revealed + [(e, 1.0)])
        sets.append(env.revealed + [(e, 0.0)])
    preds = env.instrument.predict_batch(sets)            # [2*ncand, nm]
    best_e, best_eig = cands[0], -np.inf
    for j, e in enumerate(cands):
        h_l = _entropy(preds[2 * j])[base].sum()
        h_d = _entropy(preds[2 * j + 1])[base].sum()
        p_l = float(cur[e])
        eig = p_rated[e] * (p_l * (h_now - h_l) + (1 - p_l) * (h_now - h_d))
        if eig > best_eig:
            best_eig, best_e = eig, e
    return int(best_e)


def rollout_collect(env, profiles, uids, nm, attr_lo, chooser):
    import time
    X, A, aucs = [], [], []
    t0 = time.time()
    for n, uid in enumerate(uids):
        s = env.reset(profiles[uid]); accs = [env.episode_accuracy()]
        for _ in range(N_TURNS):
            a = chooser(env, profiles[uid])
            X.append(s.copy()); A.append(a)
            s, _, _ = env.step(a); accs.append(env.episode_accuracy())
        aucs.append(np.nanmean(accs))
        if (n + 1) % 50 == 0:
            print(f"    rollout {n+1}/{len(uids)} users, running teacher AUAC "
                  f"{np.mean(aucs):.4f} ({time.time()-t0:.0f}s)", flush=True)
    return np.array(X, np.float32), np.array(A, np.int64), float(np.mean(aucs))


def eval_policy(net, env, profiles, uids, nm):
    net.eval(); aucs = []; seqs = set(); first = Counter(); br = {}
    with torch.no_grad():
        for uid in uids:
            s = env.reset(profiles[uid]); accs = [env.episode_accuracy()]; qs = []; ans = []
            for _ in range(N_TURNS):
                logits = net(torch.from_numpy(s).unsqueeze(0))[0]
                mask = torch.full((env.n_items,), float('-inf'))
                rem = [i for i in range(env.n_items) if i not in env.asked]
                mask[rem] = 0.0
                a = int(torch.argmax(logits + mask).item())
                qs.append(a); ans.append('liked' if (not np.isnan(profiles[uid][a]) and profiles[uid][a] >= .5)
                                          else ('unknown' if np.isnan(profiles[uid][a]) else 'disliked'))
                s, _, _ = env.step(a); accs.append(env.episode_accuracy())
            aucs.append(np.nanmean(accs)); seqs.add(tuple(qs)); first[qs[0]] += 1
            if len(qs) >= 2: br.setdefault(ans[0], Counter())[qs[1]] += 1
    branch = len({c.most_common(1)[0][0] for c in br.values()}) > 1 if br else False
    return float(np.mean(aucs)), len(seqs), len(first), branch


def main():
    w, _ = load_instrument_by_name(f'instrument_{NAME}')
    d = np.load(NPZ, allow_pickle=True)
    train, test = d['train'], d['test']
    nm = int(d['n_targets']); n_items = train.shape[1]; attr_lo = nm
    belief, _ = load_instrument_by_name(f'instrument_{NAME}')
    env = ElicitationEnv(w, n_items, N_TURNS, belief_instrument=belief,
                         state_mode='dual', heldout=True)
    sd = env.state_dim
    rng = np.random.default_rng(SEED)
    tr_idx = rng.choice(len(train), min(N_TRAIN, len(train)), replace=False)
    tr_prof = {int(i): train[i] for i in tr_idx}
    te_prof = {i: test[i] for i in range(min(N_TEST, len(test)))}

    # teacher: 'oracle' (clairvoyant, privileged) or 'greedy' (realizable info-gain)
    import policies as P
    TEACHER = os.environ.get('TEACHER', 'oracle')
    items = [tuple(x) for x in d['items'].tolist()]
    p_rated = (~np.isnan(train)).mean(0)
    if TEACHER == 'greedy':
        chooser = lambda env, prof: greedy_action(env, nm, attr_lo, p_rated)
    else:
        chooser = lambda env, prof: oracle_action(env, prof, nm, attr_lo)

    print(f"{NAME}: collecting {TEACHER} rollouts on {len(tr_prof)} train users (state_dim={sd})...", flush=True)
    X, A, teacher_auac = rollout_collect(env, tr_prof, list(tr_prof), nm, attr_lo, chooser)
    print(f"  teacher ({TEACHER}) AUAC={teacher_auac:.4f}; {len(X)} state-action pairs", flush=True)

    # split for action-prediction accuracy
    n = len(X); perm = rng.permutation(n); cut = int(0.9 * n)
    Xtr, Atr, Xva, Ava = X[perm[:cut]], A[perm[:cut]], X[perm[cut:]], A[perm[cut:]]

    from equivariant_actor import EquivariantActor
    net = EquivariantActor(n_items, 'dual')
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.CrossEntropyLoss()
    Xtr_t, Atr_t = torch.from_numpy(Xtr), torch.from_numpy(Atr)
    best_top1, best_state = -1, None
    for ep in range(60):
        net.train(); idx = torch.randperm(len(Xtr_t))
        for s0 in range(0, len(idx), 256):
            b = idx[s0:s0+256]
            opt.zero_grad(); loss = lossf(net(Xtr_t[b]), Atr_t[b]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(torch.from_numpy(Xva))
            top1 = (pv.argmax(1).numpy() == Ava).mean()
            top5 = np.mean([Ava[i] in pv[i].topk(5).indices.numpy() for i in range(len(Ava))])
        if top1 > best_top1:
            best_top1, best_state = top1, {k: v.clone() for k, v in net.state_dict().items()}
        if (ep+1) % 20 == 0:
            print(f"  ep{ep+1} BC val action top1={top1:.3f} top5={top5:.3f}", flush=True)
    net.load_state_dict(best_state)

    out = f'C:/dev/phd/casper/experiments/paper2/distill_{NAME}_{TEACHER}.pt'
    student_auac, seqs, t1, branch = eval_policy(net, env, te_prof, list(te_prof), nm)
    torch.save({'policy_state_dict': net.state_dict(), 'state_mode': 'dual',
                'state_dim': sd, 'n_items': n_items, 'arch': 'equivariant',
                'teacher': TEACHER, 'teacher_auac': teacher_auac,
                'student_auac': student_auac, 'bc_top1': float(best_top1)}, out)
    print(f"\n===== DISTILLATION RESULT ({TEACHER} teacher, held-out, test users) =====")
    print(f"  teacher ({TEACHER})     AUAC = {teacher_auac:.4f}")
    print(f"  student (distilled)    AUAC = {student_auac:.4f}   "
          f"distinct_seqs={seqs}/{len(te_prof)} distinct_t1={t1} branches={branch}")
    print(f"  BC teacher-action top1 = {best_top1:.3f}")
    print(f"  reference: greedy 0.7157, scpr 0.7210, popularity 0.7006, PPO-static 0.7192, oracle ~0.796")
    print(f"  saved {out}")


if __name__ == '__main__':
    main()
