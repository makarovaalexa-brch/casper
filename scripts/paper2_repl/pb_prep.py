"""pb_prep.py -- FAITHFUL OLD Paper B (June 2026 ML-1M reconstruction-encoder line) ported to the
CANONICAL ML-25M Liang ruler. Single-run replication (NOT the 2x2 in DESIGN_PAPERB_REPLICATION.md).

STEP 1/3: data recovery in CANONICAL index space + genome concepts + biased-SVD.

Why we recover explicit ratings: the faithful method is inherently EXPLICIT-rating based (biased-SVD,
residual tokens r-mu-bi, POS/NEG geometric-answer constants from liked vs disliked mean residual).
The canonical proc/*.csv are BINARY (rating>3.5 likes only). So we reproduce the exact Liang identity
maps (deterministic, seed 98765) and pull ALL-BAND explicit ratings from the raw ML-25M ratings.csv,
restricted to the canonical 18,359-item vocab -- exactly as the archived ML-25M port
(scripts/_archive/paper2_old/ml25m_build_svd_full.py) trained its biased-SVD on train users' full
explicit ratings (LIKE=4.0). The RULER (fold-in=test_tr, targets=test_te, vocab, head mask) is untouched
and taken verbatim from data/ml-25m/proc/.

Biased-SVD recipe VERBATIM from ml25m_build_svd_full.py: D=64, LAMF=0.05, LR=0.01, EP=15, minibatch SGD
(16384), mu = mean of train-users' all-band ratings. Concepts VERBATIM from ml25m_concepts_full.py:
genome relevance>0.5 membership on the vocab, keep concepts with >=30 catalog members, Ac = mean Q over
members. POS/NEG from train-users' liked (>=4) vs disliked (<4) mean residual.

Outputs -> .cache/paper2_repl/pb_data.npz  (+ asserts vocab identity with proc/unique_sid.txt).
"""
import os, time, json, numpy as np, pandas as pd

ROOT = 'C:/dev/phd/casper'
RAW = f'{ROOT}/data/movielens/ratings.csv'
GENOME = f'{ROOT}/data/movielens/genome-scores.csv'
GTAGS = f'{ROOT}/data/movielens/genome-tags.csv'
PROC = f'{ROOT}/data/ml-25m/proc'
OUT = f'{ROOT}/.cache/paper2_repl'
os.makedirs(OUT, exist_ok=True)

# Liang constants (verbatim, seed 98765) -- must reproduce liang_split.py exactly.
SEED = 98765; MIN_UC = 5; N_HELDOUT = 10000; RATING_GT = 3.5
CAT_MIN = 20; CAT_N = 18430
# biased-SVD recipe (verbatim ml25m_build_svd_full.py)
D = 64; LAMF = 0.05; LR = 0.01; EP = 15; LIKE = 4.0; MINMEMBERS = 30
rng = np.random.default_rng(0)
t0 = time.time()


def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


# ------------------------------------------------------------------ reproduce Liang identity maps
log(f"loading raw {RAW}")
raw = pd.read_csv(RAW, usecols=['userId', 'movieId', 'rating'],
                  dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32})
log(f"raw interactions: {len(raw):,}")

icnt = raw.groupby('movieId').size()
catalog = set(icnt[icnt >= CAT_MIN].index.tolist())
assert len(catalog) == CAT_N, f"catalog {len(catalog)} != {CAT_N}"
raw_cat = raw[raw['movieId'].isin(catalog)]
log(f"catalog-restricted (>= {CAT_MIN} total ratings): {len(catalog)} movies, {len(raw_cat):,} interactions")

# like matrix (Liang binarize) -> filter_triplets(min_uc=5) -> permute -> cohorts (verbatim)
raw_like = raw_cat[raw_cat['rating'] > RATING_GT]
uc = raw_like[['userId']].groupby('userId', as_index=False).size()
keep_u = uc[uc['size'] >= MIN_UC]['userId']
raw_like = raw_like[raw_like['userId'].isin(keep_u)]
uact = raw_like[['userId']].groupby('userId', as_index=False).size()
unique_uid = uact['userId'].values
np.random.seed(SEED)
unique_uid = unique_uid[np.random.permutation(unique_uid.size)]
n_users = unique_uid.size
tr_users = unique_uid[:(n_users - N_HELDOUT * 2)]
vd_users = unique_uid[(n_users - N_HELDOUT * 2):(n_users - N_HELDOUT)]
te_users = unique_uid[(n_users - N_HELDOUT):]
log(f"cohorts: train={len(tr_users):,} val={len(vd_users):,} test={len(te_users):,}")

train_plays = raw_like[raw_like['userId'].isin(tr_users)]
unique_sid = pd.unique(train_plays['movieId'])           # vocab from TRAIN users only, ordered
ni = len(unique_sid)
log(f"item vocab (train users only): {ni}")

# --- FIDELITY ASSERT: vocab must match the canonical proc/unique_sid.txt byte-for-byte (order+values)
proc_sid = np.loadtxt(f'{PROC}/unique_sid.txt', dtype=np.int64)
assert ni == len(proc_sid), f"vocab size {ni} != proc {len(proc_sid)}"
assert np.array_equal(unique_sid.astype(np.int64), proc_sid), "vocab order/values differ from proc!"
meta = json.load(open(f'{PROC}/meta.json'))
assert ni == meta['n_items'] and len(tr_users) == meta['n_train_users'], "cohort sizes differ from meta"
log(f"FIDELITY OK: vocab + cohort sizes identical to canonical proc/ (n_items={ni}, train={len(tr_users)})")

show2id = {int(m): i for i, m in enumerate(unique_sid)}
profile2id = {int(u): i for i, u in enumerate(unique_uid)}

# ------------------------------------------------------------------ popularity / head mask from proc/train.csv (binary)
# compute cnt + head mask IDENTICALLY to src/baselines/run_ml25m_liang.compute_head_mask (canonical metric)
tr_csv = pd.read_csv(f'{PROC}/train.csv')
cnt = np.bincount(tr_csv['sid'].values, minlength=ni).astype(np.float64)   # per-item train like count
order_pop = np.argsort(-cnt)
cumfrac = np.cumsum(cnt[order_pop]) / cnt.sum()
headmask = np.zeros(ni, bool)
headmask[order_pop[:np.searchsorted(cumfrac, 0.33) + 1]] = True
log(f"head/tail: {int(headmask.sum())} head items (top-33% train like-mass) / {int((~headmask).sum())} tail")

# ------------------------------------------------------------------ recover ALL-BAND explicit ratings (vocab items)
raw_vocab = raw_cat[raw_cat['movieId'].isin(show2id)].copy()
raw_vocab['sid'] = raw_vocab['movieId'].map(show2id).astype(np.int32)
# --- TRAIN users' full-band ratings -> biased-SVD training set
tr_all = raw_vocab[raw_vocab['userId'].isin(tr_users)].copy()
tr_all['uidx'] = tr_all['userId'].map(profile2id).astype(np.int32)
ru = tr_all['uidx'].values.astype(np.int32)
ri = tr_all['sid'].values.astype(np.int32)
rr = tr_all['rating'].values.astype(np.float32)
mu = float(rr.mean())
log(f"train explicit ratings (all bands, vocab): {len(rr):,}  mu={mu:.4f}  "
    f"(likes r>=4: {int((rr>=LIKE).sum()):,})")

# ------------------------------------------------------------------ biased SVD (verbatim recipe)
truniq = np.unique(ru); trmap = np.zeros(ru.max() + 1, np.int32); trmap[truniq] = np.arange(len(truniq))
ruD = trmap[ru]; ntr = len(truniq)
bu = np.zeros(ntr, np.float32); bi = np.zeros(ni, np.float32)
P = (0.1 * rng.standard_normal((ntr, D))).astype(np.float32)
Q = (0.1 * rng.standard_normal((ni, D))).astype(np.float32)
idx = np.arange(len(rr))
log(f"training biased-SVD: {len(rr):,} ratings, D={D}, EP={EP}")
for ep in range(EP):
    rng.shuffle(idx)
    for b0 in range(0, len(idx), 16384):
        bidx = idx[b0:b0 + 16384]; us = ruD[bidx]; it = ri[bidx]
        pred = mu + bu[us] + bi[it] + np.sum(P[us] * Q[it], 1)
        e = (rr[bidx] - pred).astype(np.float32)
        np.add.at(bu, us, LR * (e - LAMF * bu[us])); np.add.at(bi, it, LR * (e - LAMF * bi[it]))
        gP = LR * (e[:, None] * Q[it] - LAMF * P[us]); gQ = LR * (e[:, None] * P[us] - LAMF * Q[it])
        np.add.at(P, us, gP); np.add.at(Q, it, gQ)
    if (ep + 1) % 5 == 0 or ep == 0:
        tr_rmse = np.sqrt(np.mean((rr - (mu + bu[ruD] + bi[ri] + np.sum(P[ruD] * Q[ri], 1))) ** 2))
        log(f"  ep{ep+1} train RMSE={tr_rmse:.4f}")
Q = Q.astype(np.float32); bi = bi.astype(np.float32)

# POS/NEG geometric-answer constants (train users' liked vs disliked mean residual)
res_all = (rr - mu - bi[ri]).astype(np.float32)
POS = float(res_all[rr >= LIKE].mean()); NEG = float(res_all[rr < LIKE].mean())
log(f"POS(mean liked resid)={POS:.4f}  NEG(mean disliked resid)={NEG:.4f}")

# ------------------------------------------------------------------ genome concepts (verbatim ml25m_concepts_full)
log("building genome concepts (relevance>0.5, >=30 members)")
tagitems = {}
with open(GENOME) as f:
    next(f)
    for line in f:
        a = line.split(','); m = int(a[0])
        if m in show2id and float(a[2]) > 0.5:
            tagitems.setdefault(int(a[1]), []).append(show2id[m])
ctags = [t for t, its in tagitems.items() if len(its) >= MINMEMBERS]
nc = len(ctags)
Ac = np.stack([Q[np.array(tagitems[t])].mean(0) for t in ctags]).astype(np.float32)
citems = [np.array(tagitems[t], np.int32) for t in ctags]
citems_off = np.zeros(nc + 1, np.int64)
for k in range(nc): citems_off[k + 1] = citems_off[k] + len(citems[k])
citems_flat = np.concatenate(citems) if nc else np.zeros(0, np.int32)
ccount = np.array([len(c) for c in citems])                # #members (for conc_pop selector)
log(f"concepts: {nc} (vs OLD ML-1M 761); mean members/concept={ccount.mean():.0f}")

# ------------------------------------------------------------------ train CSR (for encoder trainer)
o = np.argsort(ru, kind='stable')
uu_s = ru[o]; ii_s = ri[o]; rr_s = rr[o]
res_s = (rr_s - mu - bi[ii_s]).astype(np.float32)
off = np.zeros(ntr + 1, np.int64); np.add.at(off, uu_s + 1, 1); off = np.cumsum(off)
# NOTE: uu_s is in ruD? No -- ru is uidx (canonical). Rebuild offsets over canonical train uids present.
# Use contiguous train-user index (ruD space) so off indexes 0..ntr-1.
ruo = ruD[o]
off = np.zeros(ntr + 1, np.int64); np.add.at(off, ruo + 1, 1); off = np.cumsum(off)
# store the canonical uid for each contiguous train index (not needed downstream but kept for trace)
tr_uid_of_idx = np.zeros(ntr, np.int32); tr_uid_of_idx[ruo] = uu_s  # canonical uid per contiguous idx

# ------------------------------------------------------------------ TEST interview data (canonical fold-in/target)
# fold-in = proc/test_tr.csv sids; enrich with explicit ratings (join on uid,sid). targets = proc/test_te.csv sids.
te_tr = pd.read_csv(f'{PROC}/test_tr.csv')
te_te = pd.read_csv(f'{PROC}/test_te.csv')
te_all = raw_vocab[raw_vocab['userId'].isin(te_users)].copy()
te_all['uidx'] = te_all['userId'].map(profile2id).astype(np.int32)
# rating lookup (uid,sid)->rating for test users
rat_lookup = {(int(u), int(s)): float(r)
              for u, s, r in zip(te_all['uidx'].values, te_all['sid'].values, te_all['rating'].values)}
miss = 0
foldin_uid = te_tr['uid'].values.astype(np.int32); foldin_sid = te_tr['sid'].values.astype(np.int32)
foldin_rat = np.empty(len(te_tr), np.float32)
for k in range(len(te_tr)):
    v = rat_lookup.get((int(foldin_uid[k]), int(foldin_sid[k])))
    if v is None: miss += 1; v = LIKE
    foldin_rat[k] = v
log(f"test fold-in rows: {len(te_tr):,} (rating-join misses={miss})")
target_uid = te_te['uid'].values.astype(np.int32); target_sid = te_te['sid'].values.astype(np.int32)

# VAL cohort (10k) -- for encoder best-checkpoint selection (HARD RULE 9), same enrichment as test.
va_tr = pd.read_csv(f'{PROC}/validation_tr.csv'); va_te = pd.read_csv(f'{PROC}/validation_te.csv')
va_all = raw_vocab[raw_vocab['userId'].isin(vd_users)].copy()
va_all['uidx'] = va_all['userId'].map(profile2id).astype(np.int32)
vrat_lookup = {(int(u), int(s)): float(r)
               for u, s, r in zip(va_all['uidx'].values, va_all['sid'].values, va_all['rating'].values)}
vfold_uid = va_tr['uid'].values.astype(np.int32); vfold_sid = va_tr['sid'].values.astype(np.int32)
vfold_rat = np.array([vrat_lookup.get((int(vfold_uid[k]), int(vfold_sid[k])), LIKE) for k in range(len(va_tr))], np.float32)
vtarget_uid = va_te['uid'].values.astype(np.int32); vtarget_sid = va_te['sid'].values.astype(np.int32)
log(f"val fold-in rows: {len(va_tr):,}")

np.savez(f'{OUT}/pb_data.npz',
         Q=Q, bi=bi, mu=mu, cnt=cnt, headmask=headmask, POS=POS, NEG=NEG,
         Ac=Ac, ctags=np.array(ctags), citems_flat=citems_flat, citems_off=citems_off, ccount=ccount,
         # train CSR for encoder
         uu_s=ruo.astype(np.int32), ii_s=ii_s.astype(np.int32), rr_s=rr_s, res_s=res_s, off=off, ntr=ntr, ni=ni,
         tr_uid_of_idx=tr_uid_of_idx,
         # test interview
         foldin_uid=foldin_uid, foldin_sid=foldin_sid, foldin_rat=foldin_rat,
         target_uid=target_uid, target_sid=target_sid,
         vfold_uid=vfold_uid, vfold_sid=vfold_sid, vfold_rat=vfold_rat,
         vtarget_uid=vtarget_uid, vtarget_sid=vtarget_sid)
log(f"SAVED {OUT}/pb_data.npz")
print("DONE")
