r"""adaptive_concept_arms.py -- U3b: THE decisive row for the author's concept ruling (2026-07-25):
do concepts carry realizable value under ADAPTIVE selection? The static generic bank is a SELECTION
problem (generic answers are mostly weak-valued; the per-answer strength needs the user's strong
concepts FOUND). Three concept-question arms, per-question currency, q=2/4/8/16, SIGNED retrained
modules (sclite/scfull), same COLD_SEED 10k users:

  (a) ADAPTIVE-REALIZABLE greedy: opener = the globally best train concept (max train mean |signed v|
      over answerable cells); each next question maximizes expected |signed response| =
      |<d_c, z_current>| x train std|v|(c) -- uses ONLY answers observed so far (via the rung's own
      belief z) + train-population statistics. Nothing user-privileged.
  (b) ADAPTIVE-ORACLE ceiling: next = the unasked concept with the true largest |signed v| for the
      user (PRIVILEGED read of the eval answer table -- upper bound, labeled).
  (c) STATIC bank floor: the existing global member-mass order.
  (d) items-pop comparator (popularity-ranked item questions through the same rung's tower).

Per-question currency everywhere: an asked concept burns budget; answered iff structural (>=2 rated
members); folded per the signed four-band rule (refuse band burns without folding). DECISION
QUANTITY: the fraction of the (oracle - static) gap the realizable greedy closes, full and tail,
per q. Output: experiments/battery/adaptive_concept_arms.json + a labeled table.

Usage:
  python src/instrument/adaptive_concept_arms.py --smoke
  python src/instrument/adaptive_concept_arms.py [--full_threads] [--rungs sclite,scfull]
"""
import os
import sys
_FULL = "--full_threads" in sys.argv
_NT = str(os.cpu_count()) if _FULL else "4"
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)
os.environ.setdefault("MKL_NUM_THREADS", _NT)
import json
import time
import argparse
import numpy as np
import torch
from scipy import sparse

torch.set_num_threads(int(_NT))

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)

import metrics as M
from train_tower_t2 import log, I25Encoder
import run_battery_phaseA as PA
from run_battery_phaseA import (build_real_ctx, build_smoke_ctx, ndcg10_from_scores, bootstrap_ci,
                                OUTDIR, SEED)
from tradeoff_ledger import build_shared, Rung

assert not hasattr(sys.modules[__name__], "load_answerer")

BUDGETS = (2, 4, 8, 16)
CONCSTATS = os.path.join(_ROOT, ".cache", "instrument", "signed_train_concstats.npz")


def train_concept_stats(ctx, Mm, pexp, smoke=False, chunk=20000):
    """Train-population per-concept stats of the SIGNED value over answerable cells:
    mean|v| (the opener score) and std(v) (the spread term). Cached."""
    if smoke:
        rng = np.random.RandomState(2)
        C = Mm.shape[1]
        return rng.rand(C).astype(np.float32), (0.2 + rng.rand(C)).astype(np.float32)
    if os.path.exists(CONCSTATS):
        z = np.load(CONCSTATS)
        return z["mean_abs"], z["std"]
    from signed_answers import load_prereg, _components, TAU_REF
    import pandas as pd
    prereg, item_mean = load_prereg()
    df = ctx.raw[ctx.raw["userId"].isin(ctx.tr_set)]
    sid = df["movieId"].map(ctx.show2id); ok = sid.notna()
    sid = sid[ok].astype(np.int64).values
    stars = df.loc[ok, "rating"].values.astype(np.float32)
    uid, _ = pd.factorize(df.loc[ok, "userId"].values)
    n_u = int(uid.max()) + 1
    resid = (stars - item_mean[sid]).astype(np.float32)
    Xb = sparse.csr_matrix((np.ones(len(sid), np.float32), (uid, sid)), shape=(n_u, ctx.ni))
    Xr = sparse.csr_matrix((resid, (uid, sid)), shape=(n_u, ctx.ni))
    C = Mm.shape[1]
    s_n = np.zeros(C); s_v = np.zeros(C); s_v2 = np.zeros(C); s_av = np.zeros(C)
    for st in range(0, n_u, chunk):
        SEL, NPMI, VAL, E, nc = _components(Xb[st:st + chunk], Xr[st:st + chunk], Mm, pexp)
        v = np.clip(NPMI + prereg["w_val"] * VAL, -1, 1)
        m = (nc >= 2) & (E >= TAU_REF)
        s_n += m.sum(0); s_v += np.where(m, v, 0).sum(0)
        s_v2 += np.where(m, v * v, 0).sum(0); s_av += np.where(m, np.abs(v), 0).sum(0)
    n = np.maximum(s_n, 1)
    mean_abs = (s_av / n).astype(np.float32)
    std = np.sqrt(np.maximum(s_v2 / n - (s_v / n) ** 2, 1e-6)).astype(np.float32)
    np.savez(CONCSTATS, mean_abs=mean_abs, std=std, n=s_n)
    log(f"[stats] train concept stats cached: opener = concept {int(mean_abs.argmax())} "
        f"(mean|v|={mean_abs.max():.3f})")
    return mean_abs, std


def score_users(rung, item_seqs, conc_lists, rows, mask_items=None):
    """Rung fold -> per-user full/tail NDCG@10 with credit-neutral masking of revealed ITEMS."""
    ctx = rung.ctx
    Z = rung.z_batch(item_seqs, conc_lists)
    full = np.full(len(rows), np.nan); tail = np.full(len(rows), np.nan)
    for st in range(0, len(rows), 500):
        ch = rows[st:st + 500]
        S = (Z[st:st + len(ch)] @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
        mask = ctx.va_tr[ch].copy()
        te = ctx.va_te[ch].tolil()
        for j in range(len(ch)):
            s = item_seqs[st + j][0]
            if len(s):
                extra = sparse.csr_matrix((np.ones(len(s), np.float32), (np.zeros(len(s)), s)),
                                          shape=(1, ctx.ni))
                mask[j] = mask[j] + extra
                for i in np.asarray(s).tolist():
                    te[j, i] = 0
        mask.data[:] = 1.0
        te = te.tocsr(); te.eliminate_zeros()
        f, t = ndcg10_from_scores(S, mask.tocsr(), te, ctx.head_mask)
        full[st:st + len(ch)] = f; tail[st:st + len(ch)] = t
    return full, tail, Z


def run_arm(rung, rows, arm, mean_abs, std_v, budgets=BUDGETS):
    """One concept-question arm walked to q16 with snapshots. Returns {q: (full, tail, answered)}."""
    ctx, sh = rung.ctx, rung.sh
    C = sh["d_c"].shape[0]
    qmax = max(budgets)
    empty_i = (np.empty(0, np.int64), np.empty(0, np.int64))
    D = sh["d_c"].numpy().astype(np.float32)                  # (C, d) selection geometry
    Vs = sh["conc_value_signed"]; Fs = sh["conc_fold_signed"]
    answerable = sh["conc_answerable"]
    asked = np.zeros((len(rows), C), bool)
    folded = [[] for _ in rows]
    Z = torch.zeros(len(rows), ctx.d)
    out = {}
    if arm == "static":
        orders = None
    elif arm == "oracle":
        prio = np.where(answerable & Fs, np.abs(Vs), -1.0)    # PRIVILEGED: true |signed v|
    for step in range(1, qmax + 1):
        if arm == "static":
            picks = []
            for j, r in enumerate(rows):
                nxt = next((int(c) for c in sh["bank_conc"] if not asked[j, int(c)]), -1)
                picks.append(nxt)
        elif arm == "oracle":
            pr = prio[np.asarray(rows)].copy()
            pr[asked] = -2.0
            picks = pr.argmax(1).tolist()
        else:                                                 # greedy realizable
            if step == 1:
                base = np.tile(mean_abs, (len(rows), 1))      # global opener score
            else:
                base = np.abs(Z.numpy() @ D.T) * std_v[None, :]
            base[asked] = -np.inf
            picks = base.argmax(1).tolist()
        changed = False
        for j, r in enumerate(rows):
            c = int(picks[j])
            if c < 0:
                continue
            asked[j, c] = True
            if answerable[r, c] and Fs[r, c]:
                folded[j].append((c, float(Vs[r, c])))
                changed = True
        if arm == "greedy" and changed and step < qmax:       # belief update for the next pick
            Z = rung.z_batch([empty_i] * len(rows), [list(f) for f in folded])
        if step in budgets:
            full, tail, _ = score_users(rung, [empty_i] * len(rows),
                                        [list(f) for f in folded], rows)
            out[step] = {"full@10": float(np.nanmean(full)), "tail@10": float(np.nanmean(tail)),
                         "mean_answered": float(np.mean([len(f) for f in folded])),
                         "_fv": full, "_tv": tail}
    return out


def run_items_pop(rung, rows, budgets=BUDGETS):
    ctx, sh = rung.ctx, rung.sh
    qmax = max(budgets)
    out = {}
    seqs_at = {q: [] for q in budgets}
    for r in rows:
        d = sh["lvl_lookup"][r]
        items = []
        nq = 0
        for i in sh["order_pop"]:
            i = int(i); nq += 1
            if i in d:
                items.append((i, d[i]))
            for q in budgets:
                if nq == q:
                    seqs_at[q].append((np.asarray([s for s, _ in items], np.int64),
                                       np.asarray([l for _, l in items], np.int64)))
            if nq >= qmax:
                break
    for q in budgets:
        full, tail, _ = score_users(rung, seqs_at[q], [[] for _ in rows], rows)
        out[q] = {"full@10": float(np.nanmean(full)), "tail@10": float(np.nanmean(tail)),
                  "mean_answered": float(np.mean([len(s[0]) for s in seqs_at[q]])),
                  "_fv": full, "_tv": tail}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--rungs", default="sclite,scfull")
    ap.add_argument("--sclite_ckpt", default=os.path.join(_ROOT, ".cache", "instrument",
                                                          "cfold_signed_best.pt"))
    ap.add_argument("--scfull_ckpt", default=os.path.join(_ROOT, ".cache", "instrument",
                                                          "cfull_signed_best.pt"))
    ap.add_argument("--full_threads", action="store_true")
    args = ap.parse_args()
    t00 = time.time()
    ctx = build_smoke_ctx() if args.smoke else build_real_ctx(args.snapshot)
    sh = build_shared(ctx)
    if args.smoke and not sh.get("signed_available"):
        # fabricate signed arrays for the code-path check (labeled)
        rng = np.random.RandomState(4)
        C = sh["d_c"].shape[0]
        sh["conc_value_signed"] = (rng.rand(ctx.n, C).astype(np.float32) * 2 - 1)
        sh["conc_fold_signed"] = rng.rand(ctx.n, C) > 0.15
        sh["signed_available"] = True
    assert sh.get("signed_available"), "signed prereg cache required (U1 computes it)"
    Mm = sparse.csr_matrix((np.ones(sum(len(sh["members"][t]) for t in sh["tags"]), np.float32),
                            (np.concatenate([sh["members"][t] for t in sh["tags"]]),
                             np.concatenate([np.full(len(sh["members"][t]), i)
                                             for i, t in enumerate(sh["tags"])]))),
                           shape=(ctx.ni, len(sh["tags"])))
    pexp = (ctx.cnt @ np.asarray(Mm.todense())) / max(ctx.cnt.sum(), 1e-9)
    mean_abs, std_v = train_concept_stats(ctx, Mm, pexp, smoke=args.smoke)
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    out = {"analysis": "adaptive_concept_arms_U3b", "n_users": len(rows), "budgets": list(BUDGETS),
           "greedy_rule": "opener = argmax train mean|v|; then argmax |<d_c, z>| * train std|v| "
                          "(answers-so-far + train stats ONLY)",
           "oracle_rule": "argmax true |signed v| per user (PRIVILEGED ceiling)",
           "rungs": {}}
    for name in args.rungs.split(","):
        name = name.strip()
        ckpt = args.sclite_ckpt if name == "sclite" else args.scfull_ckpt
        if args.smoke:
            from concept_fold import ConceptFoldNet
            net = ConceptFoldNet(len(sh["tags"]), d=ctx.d, h=32, conc_init=sh["d_c"].numpy())
            with torch.no_grad():
                for p in net.mlp[-1].parameters():
                    p.add_(torch.randn_like(p) * 0.05)
            net.eval()
            rung = Rung(name, ctx, sh, clite_net=net, val_source="signed")
            if name == "scfull":
                continue                                     # smoke: one rung suffices
        elif not os.path.exists(ckpt):
            out["rungs"][name] = "PENDING (no ckpt)"; continue
        elif name == "sclite":
            from concept_fold import ConceptFoldNet
            blob = torch.load(ckpt, map_location="cpu")
            net = ConceptFoldNet(len(blob["tags"]), d=ctx.d, h=blob["hidden"])
            net.load_state_dict(blob["net"]); net.eval()
            rung = Rung(name, ctx, sh, clite_net=net, val_source="signed")
        else:
            blob = torch.load(ckpt, map_location="cpu")
            native = ctx.enc._native[0] if hasattr(ctx.enc, "_native") else None
            enc_sf = I25Encoder(ctx.ni, native, d_lat=ctx.d, token_mode="film",
                                n_concepts=len(sh["tags"]))
            enc_sf.load_state_dict(blob["enc"]); enc_sf.eval()
            rung = Rung(name, ctx, sh, cfull_enc=enc_sf, val_source="signed")
        log(f"=== RUNG {name} (adaptive arms) ===")
        res = {}
        arms = {"static": None, "greedy": None, "oracle": None}
        for arm in arms:
            t0 = time.time()
            cur = run_arm(rung, rows, arm, mean_abs, std_v)
            res[arm] = cur
            log(f"[{name} {arm:>7}] " + " ".join(
                f"q{q}={cur[q]['full@10']:.4f}/{cur[q]['tail@10']:.4f}(a={cur[q]['mean_answered']:.1f})"
                for q in BUDGETS) + f" ({(time.time()-t0)/60:.1f}m)")
        res["items_pop"] = run_items_pop(rung, rows)
        log(f"[{name} items-pop] " + " ".join(
            f"q{q}={res['items_pop'][q]['full@10']:.4f}" for q in BUDGETS))
        # DECISION QUANTITY: gap closure per q, full + tail, with paired CI on greedy-static @q8
        gap = {}
        for q in BUDGETS:
            go = res["oracle"][q]["full@10"] - res["static"][q]["full@10"]
            gg = res["greedy"][q]["full@10"] - res["static"][q]["full@10"]
            got = res["oracle"][q]["tail@10"] - res["static"][q]["tail@10"]
            ggt = res["greedy"][q]["tail@10"] - res["static"][q]["tail@10"]
            gap[str(q)] = {"oracle_minus_static_full": go, "greedy_closes_full":
                           (gg / go if abs(go) > 1e-9 else float("nan")),
                           "oracle_minus_static_tail": got, "greedy_closes_tail":
                           (ggt / got if abs(got) > 1e-9 else float("nan"))}
        d8 = bootstrap_ci(res["greedy"][8]["_fv"] - res["static"][8]["_fv"])
        rout = {"arms": {a: {str(q): {k: v for k, v in res[a][q].items() if not k.startswith("_")}
                             for q in BUDGETS} for a in res},
                "gap_closure": gap,
                "greedy_minus_static_q8_full": {"mean": d8[0], "ci95": d8[1]}}
        out["rungs"][name] = rout
    out["seconds"] = round(time.time() - t00, 1)
    outdir = ctx.outdir if args.smoke else OUTDIR
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "adaptive_concept_arms.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    log(f"[out] -> {path}")
    print("\n=== U3b ADAPTIVE CONCEPT ARMS (the decisive row) ===")
    for name, r in out["rungs"].items():
        if isinstance(r, str):
            print(f"[{name}] {r}"); continue
        for a in ("static", "greedy", "oracle", "items_pop"):
            lbl = {"static": "STATIC bank (floor)", "greedy": "ADAPTIVE-REALIZABLE greedy",
                   "oracle": "ADAPTIVE-ORACLE (PRIVILEGED ceiling)",
                   "items_pop": "items-pop comparator"}[a]
            c = r["arms"][a]
            print(f"[{name}] {lbl:>36}: " + " ".join(
                f"q{q}={c[str(q)]['full@10']:.4f}/{c[str(q)]['tail@10']:.4f}" for q in BUDGETS))
        g = r["gap_closure"]
        print(f"[{name}] gap closed by greedy (full): " + " ".join(
            f"q{q}={g[str(q)]['greedy_closes_full']:.0%}" for q in BUDGETS) +
            " | (tail): " + " ".join(f"q{q}={g[str(q)]['greedy_closes_tail']:.0%}" for q in BUDGETS))


if __name__ == "__main__":
    main()
