"""sasrec.py -- SASRec (Kang & McAuley, ICDM 2018, "Self-Attentive Sequential Recommendation").

Faithful minimal port of the self-attention sequential recommender, adapted to the CASPER Liang-ML-25M
strong-generalization ruler. SASRec is the one baseline here that consumes ITEM ORDER, so it needs
TIMESTAMPS -- which the Liang proc CSVs discard. We therefore rebuild per-user chronological sequences
from the raw ratings file, restricted to the canonical split.

SEQUENCE CONSTRUCTION (documented, no data caps):
  * Users: exactly the TRAIN users of the canonical split, reconstructed by REPLAYING the deterministic
    liang_split recipe (same catalog restriction -> rating>3.5 -> filter_triplets(min_uc=5) -> seed-98765
    permutation -> first n-20000 users). We import scripts/baselines/liang_split.py so the logic cannot
    drift from the split that built data/ml-25m/proc/.
  * Items: the 18,359-item train vocab, read from proc/unique_sid.txt (show2id = line index).
  * Each train user's sequence = ALL of their kept interactions (rating>3.5 AND movie in vocab) in
    ascending TIMESTAMP order (ties broken by movieId for determinism). No truncation of the corpus;
    only the model's fixed context window `maxlen` keeps the most-recent maxlen items per step (the
    standard SASRec windowing, not a data reduction).

MODEL (paper architecture): item + learned positional embedding -> `num_blocks` Transformer blocks with
  CAUSAL (lower-triangular) self-attention + point-wise FFN, residual + LayerNorm, dropout. Next-item
  objective with 1 sampled negative per position (paper's binary cross-entropy; `sigmoid(pos)-sigmoid(neg)`
  BCE). Adam, lr=1e-3, beta2=0.98.

HYPERPARAMETERS (paper ML-1M config, scaled sensibly for 18k items / 140k users -- documented choices,
  NOT a verbatim published-number snap; SASRec has no ML-20M/25M Liang-split number to match):
    hidden=128 (paper ML-1M=50; raised for the larger catalog), num_blocks=2, num_heads=2 (paper=1),
    maxlen=200 (paper ML-1M), dropout=0.2, lr=1e-3, batch=128, l2=0.0, num_neg=1.
  Exposed on the CLI; the used values are recorded in the output `hp`.

EVAL ADAPTATION -- fold-in for strong generalization (documented; REAL per-user timestamps):
  The paper evaluates next-item on a held-out LAST item of a known sequence. Our ruler is the SET-based
  metrics.evaluate: predict(X_csr) receives the fold-in ITEM SET for each held-out user and must score
  the whole catalog for NDCG@10 over the 20% target set. We adapt SASRec as a fold-in scorer:
    - each held-out user's fold-in items are ordered by THAT USER'S OWN raw timestamps
      (data/movielens/ratings.csv), oldest-first, most-recent `maxlen` kept; fed as the sequence; the
      FINAL position's hidden state scores all items by inner product with the item-embedding table ->
      dense (batch x n_items). ("sequence -> next-item logits over the catalog = full-profile prediction",
      the standard fold-in adaptation.)
  TIMESTAMP-CARRYING EVAL PATH (build_eval_order): the proc CSVs drop timestamps, so we rebuild the
  per-row ordered fold-in sequences by (a) replaying the deterministic liang_split user permutation to
  map raw userId -> proc uid, (b) reading validation_tr.csv / test_tr.csv (the EXACT fold-in rows the
  ruler scores), and (c) joining each (uid, sid) with its raw timestamp. metrics.evaluate slices users
  in ascending-row batches, so predict carries a batch CURSOR per split (predict.set_split resets it);
  every row is verified by an exact SET-EQUALITY assertion between the incoming fold-in columns and the
  stored ordered sequence -- any misalignment raises instead of silently mis-ordering. The set-based
  ruler itself is untouched; order matters only inside sasrec's own predict.

Checkpoint/resume + early stop on val FULL NDCG@10 (the primary metric), same convention as multvae.py.

Interface: fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print) -> predict_fn
CLI: python src/baselines/sasrec.py --smoke   # SYNTHETIC sequences, code-path only (NOT the real split)
     python src/baselines/sasrec.py           # full train+eval on the Liang ML-25M split (later night)
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

torch.manual_seed(98765)
np.random.seed(98765)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RATINGS = os.path.join(_ROOT, "data", "movielens", "ratings.csv")


def _defaults():
    return argparse.Namespace(
        hidden=128, num_blocks=2, num_heads=2, maxlen=200, dropout=0.2,
        lr=1e-3, beta2=0.98, batch=128, l2=0.0, num_neg=1,
        epochs=200, patience=15, max_minutes=1e9, resume=True)


# --------------------------------------------------------------------------- model
class PointWiseFFN(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.c1 = nn.Conv1d(d, d, 1); self.c2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout); self.relu = nn.ReLU()

    def forward(self, x):                          # x: (B, L, d)
        h = x.transpose(1, 2)
        h = self.drop(self.c2(self.drop(self.relu(self.c1(h)))))
        return h.transpose(1, 2) + x


class SASRec(nn.Module):
    def __init__(self, n_items, hidden, num_blocks, num_heads, maxlen, dropout):
        super().__init__()
        self.n_items = n_items
        self.maxlen = maxlen
        # id 0 = padding; items are 1..n_items (shifted by +1 from sid)
        self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden)
        self.emb_drop = nn.Dropout(dropout)
        self.attn_ln = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_blocks)])
        self.attn = nn.ModuleList([nn.MultiheadAttention(hidden, num_heads, dropout=dropout,
                                                          batch_first=True) for _ in range(num_blocks)])
        self.ffn_ln = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_blocks)])
        self.ffn = nn.ModuleList([PointWiseFFN(hidden, dropout) for _ in range(num_blocks)])
        self.last_ln = nn.LayerNorm(hidden)

    def seq_repr(self, seq):                       # seq: (B, L) long, 0=pad
        B, L = seq.shape
        h = self.item_emb(seq) * (self.item_emb.embedding_dim ** 0.5)
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        h = self.emb_drop(h + self.pos_emb(pos))
        pad = (seq == 0)                           # (B, L) True where padding
        h = h * (~pad).unsqueeze(-1).float()
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), diagonal=1)
        for i in range(len(self.attn)):
            q = self.attn_ln[i](h)
            a, _ = self.attn[i](q, h, h, attn_mask=causal, need_weights=False)
            h = q + a
            h = self.ffn[i](self.ffn_ln[i](h))
            h = h * (~pad).unsqueeze(-1).float()
        return self.last_ln(h)                      # (B, L, d)

    def forward(self, seq, pos, neg):
        """Training: seq (B,L) inputs, pos/neg (B,L) next-item pos/neg targets. Return pos/neg logits."""
        h = self.seq_repr(seq)                      # (B, L, d)
        pe = self.item_emb(pos); ne = self.item_emb(neg)
        pos_logit = (h * pe).sum(-1)
        neg_logit = (h * ne).sum(-1)
        return pos_logit, neg_logit

    def score_last(self, seq):                      # (B,L) -> (B, n_items) catalog scores
        h = self.seq_repr(seq)[:, -1, :]            # (B, d) final position
        W = self.item_emb.weight[1:]                # (n_items, d), drop padding row
        return h @ W.T


# --------------------------------------------------------------------------- fold-in predict
def _make_predict(model, eval_order, maxlen):
    """predict(X_csr) -> dense (B, n_items), ordering each row's fold-in items by the USER'S OWN
    timestamps.

    eval_order: {"val": [ordered sid array per row], "test": [...]} -- row-aligned with the proc
    val/test matrices (build_eval_order). metrics.evaluate consumes users in ascending-row batches, so
    a per-split CURSOR tracks which rows this batch covers; predict.set_split(name) selects the split
    AND resets the cursor (call it before every full evaluate pass). Every row is checked by an exact
    set-equality assertion (incoming fold-in columns vs the stored sequence) -- a cursor desync or a
    wrong split raises immediately instead of silently mis-ordering."""
    state = {"split": None, "cursor": 0}

    def set_split(name):
        if name not in eval_order:
            raise KeyError(f"[sasrec] unknown eval split '{name}' (have {list(eval_order)})")
        state["split"] = name
        state["cursor"] = 0

    def predict(X_csr):
        if state["split"] is None:
            raise RuntimeError("[sasrec] predict.set_split('val'|'test') must be called before evaluate")
        ordered = eval_order[state["split"]]
        model.eval()
        Xc = X_csr.tocsr()
        B = Xc.shape[0]
        if state["cursor"] + B > len(ordered):
            raise RuntimeError(f"[sasrec] cursor overrun: {state['cursor']}+{B} > {len(ordered)} "
                               f"(missing set_split reset?)")
        seq = np.zeros((B, maxlen), dtype=np.int64)
        for r in range(B):
            cols = Xc.indices[Xc.indptr[r]:Xc.indptr[r + 1]]
            s_user = ordered[state["cursor"] + r]              # user's fold-in sids, oldest->newest
            if set(cols.tolist()) != set(s_user.tolist()):     # HARD alignment check, every row
                raise RuntimeError(f"[sasrec] fold-in set mismatch at split={state['split']} "
                                   f"row={state['cursor'] + r}: eval row has {len(cols)} items, "
                                   f"stored order has {len(s_user)} -- cursor/split desync")
            s = s_user[-maxlen:]                               # keep most-recent maxlen
            seq[r, maxlen - len(s):] = s + 1                   # +1 (0 is padding); right-aligned
        state["cursor"] += B
        with torch.no_grad():
            out = model.score_last(torch.from_numpy(seq))
        return out.cpu().numpy().astype(np.float32)

    predict.set_split = set_split
    return predict


# --------------------------------------------------------------------------- training
def _train_batches(seqs, maxlen, batch, n_items, rng):
    """Yield (seq, pos, neg) int64 arrays. seqs = list of np arrays of sids (chronological)."""
    order = rng.permutation(len(seqs))
    for st in range(0, len(seqs), batch):
        idx = order[st:st + batch]
        bs = len(idx)
        seq = np.zeros((bs, maxlen), dtype=np.int64)
        pos = np.zeros((bs, maxlen), dtype=np.int64)
        neg = np.zeros((bs, maxlen), dtype=np.int64)
        for j, u in enumerate(idx):
            items = seqs[u]                                     # sids, chronological
            if len(items) < 2:
                continue
            inp = items[:-1][-maxlen:]                          # inputs = all-but-last, last-maxlen window
            nxt = items[1:][-maxlen:]                           # targets = shifted by one, same window
            L = len(inp)                                        # == len(nxt)
            seq[j, maxlen - L:] = inp + 1
            pos[j, maxlen - L:] = nxt + 1
            seen = set(items.tolist())
            for k in range(maxlen - L, maxlen):
                t = rng.randint(0, n_items)
                while t in seen:
                    t = rng.randint(0, n_items)
                neg[j, k] = t + 1
        yield seq, pos, neg


def fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print,
        seqs=None, eval_order=None):
    """seqs: precomputed chronological training sequences (list of sid arrays).
    eval_order: {"val": [...], "test": [...]} row-aligned ordered fold-in sequences (build_eval_order).
    If None, both are rebuilt from the raw split (real-data path).
    NOTE: the returned predict is split-aware -- call predict.set_split('val'|'test') before every
    evaluate pass (the evaluator you pass in must do this for 'val')."""
    a = args or _defaults()
    if seqs is None or eval_order is None:
        rp = _replay_split(n_items, log=log)                   # one raw-CSV replay serves both builds
        if seqs is None:
            seqs = build_sequences(n_items, log=log, _replayed=rp)
        if eval_order is None:
            eval_order = build_eval_order(n_items, log=log, _replayed=rp)
    model = SASRec(n_items, a.hidden, a.num_blocks, a.num_heads, a.maxlen, a.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, a.beta2), weight_decay=a.l2)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.RandomState(98765)
    state = {"epoch": 0, "best": -1.0, "best_state": None, "best_epoch": 0, "history": []}
    if ckpt and getattr(a, "resume", True) and os.path.exists(ckpt):
        blob = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"]); state = blob["state"]
        log(f"[sasrec] RESUMED ep{state['epoch']} best={state['best']:.4f}")
    predict = _make_predict(model, eval_order, a.maxlen)
    t0 = time.time()
    for epoch in range(state["epoch"], a.epochs):
        model.train(); ep_loss = 0.0; nb = 0
        for seq, pos, neg in _train_batches(seqs, a.maxlen, a.batch, n_items, rng):
            seq_t = torch.from_numpy(seq); pos_t = torch.from_numpy(pos); neg_t = torch.from_numpy(neg)
            mask = (pos_t != 0).float()
            opt.zero_grad()
            pl, nl = model(seq_t, pos_t, neg_t)
            loss = (bce(pl, torch.ones_like(pl)) * mask + bce(nl, torch.zeros_like(nl)) * mask).sum() \
                / mask.sum().clamp(min=1)
            loss.backward(); opt.step()
            ep_loss += float(loss); nb += 1
        state["epoch"] = epoch + 1
        sc, _ = evaluator(predict) if evaluator is not None else (-1.0, {})
        el = (time.time() - t0) / 60.0
        state["history"].append({"epoch": epoch + 1, "loss": ep_loss / max(nb, 1),
                                 "val": sc, "min": round(el, 1)})
        log(f"[sasrec] ep{epoch+1:3d} loss {ep_loss/max(nb,1):.4f} val {sc:.4f} {el:.1f}m")
        if sc > state["best"]:
            state["best"] = sc; state["best_epoch"] = epoch + 1
            state["best_state"] = {k: v.clone() for k, v in model.state_dict().items()}
        if ckpt:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                        "args": vars(a)}, ckpt)
        if evaluator is not None and (epoch + 1) - state["best_epoch"] >= a.patience:
            log(f"[sasrec] early stop @ep{epoch+1} best {state['best']:.4f} @ep{state['best_epoch']}")
            break
        if el >= a.max_minutes:
            log(f"[sasrec] wall budget {a.max_minutes}m hit @ep{epoch+1}; re-run to resume")
            break
    if state.get("best_state") is not None:
        model.load_state_dict(state["best_state"])
    predict.best_val = state["best"]; predict.best_epoch = state["best_epoch"]
    predict.hp = dict(hidden=a.hidden, num_blocks=a.num_blocks, num_heads=a.num_heads,
                      maxlen=a.maxlen, dropout=a.dropout, lr=a.lr, batch=a.batch, num_neg=a.num_neg,
                      best_val=state["best"], best_epoch=state["best_epoch"],
                      role="SASRec (Kang&McAuley 2018) fold-in adaptation; per-user timestamp order")
    return predict


# --------------------------------------------------------------------------- real sequence build
def _replay_split(n_items, log=print):
    """Replay the deterministic liang_split recipe on the raw 25M ratings (REAL-DATA path, ~1 min).
    Returns (kept, unique_uid, LS): `kept` = DataFrame of all kept interactions of vocab items with
    columns userId, sid, timestamp; `unique_uid` = the seed-98765-permuted userId array whose slices
    define train/val/test users AND the proc uid = position in this array (liang_split's profile2id)."""
    import pandas as pd
    sys.path.insert(0, os.path.join(_ROOT, "scripts", "baselines"))
    import liang_split as LS

    # show2id from the canonical vocab (line index == sid), so we never diverge from the split.
    proc = os.path.join(_ROOT, "data", "ml-25m", "proc")
    with open(os.path.join(proc, "unique_sid.txt")) as f:
        vocab = [int(x) for x in f.read().split()]
    show2id = {m: i for i, m in enumerate(vocab)}
    assert len(vocab) == n_items, f"vocab {len(vocab)} != n_items {n_items}"

    log(f"[sasrec] loading raw ratings {RATINGS} (real-data path)")
    raw = pd.read_csv(RATINGS)
    # replay split filters: catalog restriction -> rating>3.5 -> filter_triplets(min_uc=5)
    icnt = raw.groupby("movieId").size()
    catalog = set(icnt[icnt >= LS.CATALOG_MIN_RATINGS].index.tolist())
    if len(catalog) != LS.CATALOG_N_EXPECTED:
        raise SystemExit(f"[sasrec] CATALOG MISMATCH {len(catalog)} != {LS.CATALOG_N_EXPECTED}")
    raw = raw[raw["movieId"].isin(catalog)]
    raw = raw[raw["rating"] > LS.RATING_GT]
    raw, user_activity, _ = LS.filter_triplets(raw)
    # deterministic seed-98765 user permutation (== liang_split; uid = position in this array)
    unique_uid = user_activity["userId"].values
    np.random.seed(LS.SEED)
    unique_uid = unique_uid[np.random.permutation(unique_uid.size)]
    # keep vocab items only; map movieId -> sid
    kept = raw[raw["movieId"].isin(show2id)]
    kept = kept.assign(sid=kept["movieId"].map(show2id))[["userId", "sid", "timestamp"]]
    return kept, unique_uid, LS


def build_sequences(n_items, log=print, _replayed=None):
    """TRAIN-user chronological sid sequences from raw timestamps (list of int64 arrays)."""
    kept, unique_uid, LS = _replayed if _replayed is not None else _replay_split(n_items, log=log)
    tr_users = set(unique_uid[:(unique_uid.size - LS.N_HELDOUT * 2)].tolist())
    log(f"[sasrec] train users={len(tr_users)} (replayed split)")
    sub = kept[kept["userId"].isin(tr_users)]
    sub = sub.sort_values(["userId", "timestamp", "sid"], kind="stable")
    seqs = [g["sid"].to_numpy(dtype=np.int64) for _, g in sub.groupby("userId", sort=False)]
    log(f"[sasrec] built {len(seqs)} train sequences; median len="
        f"{int(np.median([len(s) for s in seqs]))}")
    return seqs


def build_eval_order(n_items, log=print, _replayed=None):
    """Per-ROW ordered fold-in sequences for the val/test matrices, from each user's OWN timestamps.

    For split in {val, test}: read proc/<split>_tr.csv (the EXACT fold-in rows metrics.evaluate
    scores), map uid -> raw userId via the replayed permutation (uid = position in unique_uid, row =
    uid - first_uid_of_split, identical to metrics._load_tr_te_data's start_idx offset), join each
    (userId, sid) with its raw timestamp, and sort that user's fold-in sids oldest->newest (ties by
    sid, matching build_sequences). Returns {"val": [np.int64 array per row], "test": [...]}.
    Every (uid, sid) fold-in pair MUST find a timestamp (it exists in raw by construction); asserted."""
    import pandas as pd
    kept, unique_uid, LS = _replayed if _replayed is not None else _replay_split(n_items, log=log)
    n_users = unique_uid.size
    proc = os.path.join(_ROOT, "data", "ml-25m", "proc")
    bounds = {"val": (n_users - 2 * LS.N_HELDOUT, n_users - LS.N_HELDOUT),
              "test": (n_users - LS.N_HELDOUT, n_users)}
    out = {}
    for split, (lo, hi) in bounds.items():
        split_users = unique_uid[lo:hi]
        uid_of_user = {u: lo + i for i, u in enumerate(split_users)}     # raw userId -> proc uid
        sub = kept[kept["userId"].isin(set(split_users.tolist()))]
        sub = sub.assign(uid=sub["userId"].map(uid_of_user))
        ts = {(u, s): t for u, s, t in zip(sub["uid"].values, sub["sid"].values,
                                           sub["timestamp"].values)}
        # the Liang split writes validation_tr.csv / validation_te.csv, not val_*
        _fname = {"val": "validation"}.get(split, split)
        tr = pd.read_csv(os.path.join(proc, f"{_fname}_tr.csv"))
        assert int(tr["uid"].min()) == lo, f"[sasrec] {split} uid offset {tr['uid'].min()} != {lo}"
        ordered = [np.empty(0, dtype=np.int64)] * (hi - lo)
        for uid, g in tr.groupby("uid"):
            sids = g["sid"].to_numpy(dtype=np.int64)
            t = np.array([ts.get((uid, s), np.nan) for s in sids])
            assert not np.isnan(t).any(), f"[sasrec] missing timestamp for uid={uid} ({split})"
            order = np.lexsort((sids, t))                                # by ts, ties by sid
            ordered[uid - lo] = sids[order]
        out[split] = ordered
        log(f"[sasrec] {split}: ordered fold-in for {len(ordered)} rows "
            f"(median len={int(np.median([len(s) for s in ordered]))})")
    return out


# --------------------------------------------------------------------------- CLI / smoke
def _head_mask(train, n_items):
    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    order = np.argsort(-cnt)
    cumfrac = np.cumsum(cnt[order]) / cnt.sum()
    head = np.zeros(n_items, dtype=bool)
    head[order[:np.searchsorted(cumfrac, 0.33) + 1]] = True
    return head


def _smoke():
    print("[sasrec][SMOKE] synthetic sequences (code-path only, NOT the Liang split)")
    from scipy import sparse
    rng = np.random.RandomState(0)
    n_items = 130
    # synthetic chronological sequences + a synthetic held-out set
    seqs = [np.array(sorted(rng.choice(n_items, size=rng.randint(3, 12), replace=False)), dtype=np.int64)
            for _ in range(200)]
    te_tr = (sparse.random(30, n_items, density=0.1, random_state=rng,
                           data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    te_te = (sparse.random(30, n_items, density=0.1, random_state=rng,
                           data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    # synthetic per-user ordered fold-in sequences: each row's items in a random (fake-timestamp) order
    def _rows_order(Xc):
        Xc = Xc.tocsr()
        return [rng.permutation(Xc.indices[Xc.indptr[r]:Xc.indptr[r + 1]]).astype(np.int64)
                for r in range(Xc.shape[0])]
    eval_order = {"val": _rows_order(te_tr), "test": _rows_order(te_tr)}
    a = _defaults(); a.hidden = 32; a.maxlen = 20; a.epochs = 2; a.batch = 32; a.patience = 99
    hm = _head_mask(te_tr, n_items)

    def ev(pred):
        pred.set_split("val")                         # split-aware predict: reset cursor per pass
        m = M.evaluate(pred, te_tr, te_te, batch_size=10, head_mask=hm)
        return m["ndcg@10"], m
    pr = fit(None, n_items, evaluator=ev, args=a, ckpt=None, log=print, seqs=seqs,
             eval_order=eval_order)
    pr.set_split("test")
    res = M.evaluate(pr, te_tr, te_te, batch_size=10, head_mask=hm)
    # negative control: a desynced cursor / wrong rows MUST raise (alignment assertion)
    try:
        pr.set_split("test")
        pr(te_tr[5:15])                               # rows 5..14 presented as rows 0..9 -> mismatch
        raise SystemExit("[sasrec][SMOKE] FAIL: misaligned rows did NOT raise")
    except RuntimeError:
        print("[sasrec][SMOKE] alignment assertion fires on misaligned rows (control OK)")
    print(f"[sasrec][SMOKE] OK full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
          f"best_val={pr.best_val:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--max_minutes", type=float, default=1e9)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke(); return
    proc = os.path.join(_ROOT, "data", "ml-25m", "proc")
    meta = M.load_meta(proc); n_items = meta["n_items"]
    va_tr, va_te = M.load_val(n_items, proc)
    te_tr, te_te = M.load_test(n_items, proc)
    hm = _head_mask(M.load_train(n_items, proc), n_items)
    a = _defaults(); a.epochs = args.epochs; a.max_minutes = args.max_minutes
    ckpt = os.path.join(_ROOT, ".cache", "baselines", "sasrec_ml25m_liang.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    def ev(pred):
        pred.set_split("val")                         # split-aware predict: reset cursor per pass
        m = M.evaluate(pred, va_tr, va_te, batch_size=500, head_mask=hm)
        return m["ndcg@10"], m
    t0 = time.time()
    predict = fit(None, n_items, evaluator=ev, args=a, ckpt=ckpt)
    predict.set_split("test")
    res = M.evaluate(predict, te_tr, te_te, batch_size=500, head_mask=hm)
    res["seconds"] = time.time() - t0; res["hp"] = predict.hp
    print(f"[sasrec] test full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
          f"ndcg@100={res['ndcg@100']:.4f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
