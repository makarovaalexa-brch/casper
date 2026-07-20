"""rung1_checks.py -- TWO training-free diagnostics on the Rung-I answer-encoder pilot.

CHECK 1  Learned vs inherited: decompose the 0.340 val-NDCG into
         (a) NATIVE-ONLY  z = enc_items(revealed liked items) alone (rho + tokens bypassed),
         (b) TOKENS-ONLY  z = fold with native dropped (keepA=0): member-bags + implicit only,
         (c) FULL         z = native_z + rho(pool),
         (d) rho-CONTRIB  = full - native-only,
         reported at interview lengths k in {2,4,8,full} + the mixed-default context.

CHECK 2  Is graded VALUE informative for ranking under the FROZEN RecVAE decoder? Bypass the
         inert trained gamma; build a hard-wired signed-value belief
             z = native_z(revealed LIKES) + eta * SUM_expl-tok s(value) * memberbag_emb
         s = centered valence band map {hated:-1, meh:0, liked:+0.5, loved:+1} (thresholds at the
         CENTERED_FOLD midpoints -2/3, 0, +2/3). Compare NDCG@10 of the SIGNED-VALUE belief vs a
         VALUE-BLIND belief (s==0 -> native alone = training-free value_delta) across TWO contexts:
             (i)  ON-PROFILE / positive  (default interview, mostly liked reveals),
             (ii) DISLIKE-HEAVY / adversarial (reveal the user's disliked items+regions; held set
                  includes disliked items so ranking must push them DOWN).
         Sweep eta in {2,4,8}. On (ii) also run a fold-level IG2 percentile check: are the held
         DISLIKED items down-ranked (and held LIKES up-ranked) by the signed term?

NO TRAINING. NO DATA REDUCTION. Same held-out val cohort as the pilot (2k/1k, seed 0). The 173/300
study users are quarantined (excluded upstream by the trU firewall in load_train_profiles).
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")

import rung1_encoder as R
import i25_lib as L
from i25_fold_v3_sampler import (KIND_IMPL, KIND_EXPL, LVL_ROUGH, LVL_KW, LVL_NEG,
                                  FID_DATA, FID_EASE, FID_LLM, TYPE_ITEM, TYPE_CONCEPT,
                                  TYPE_ATTR, TYPE_ENTITY, CENTERED_FOLD)

OUT = ".cache/rung1"
BEST = f"{OUT}/pilot_best.pt"


# ---------- helpers ----------
def s_band(v):
    """Signed valence from a (possibly continuous) centered value. Thresholds = midpoints between
    the four CENTERED_FOLD levels (-1,-1/3,1/3,1): -2/3, 0, 2/3."""
    if v <= -2.0 / 3.0:
        return -1.0       # hated
    if v <= 0.0:
        return 0.0        # meh
    if v <= 2.0 / 3.0:
        return 0.5        # liked
    return 1.0            # loved


def ndcg_of(FR, z, rec):
    return L.ndcg10(FR, np.asarray(z, np.float64), rec["held_likes"], set(rec["known"].keys()))


def native_only_batch(FR, native_lists):
    """z = enc_items(revealed liked items) alone."""
    return FR.enc_items(native_lists).numpy().astype(np.float64)


def fold_mu_chunked(FR, model, tok_lists, native_lists, keepA=1.0, max_elems=2_000_000, cap=96):
    """Length-BUCKETED micro-batched deterministic-mu fold (memory-safe; NO data reduction/cap of
    users or tokens). Sort by token count so padding is bounded, then form DYNAMIC chunks so that
    (chunk_size * max_len_in_chunk) <= max_elems (bounds the padded (B,K,d) tensor). Original order
    restored."""
    B = len(tok_lists)
    if B == 0:
        return np.zeros((0, FR.W.shape[1]))
    order = sorted(range(B), key=lambda i: len(tok_lists[i]))
    Z = np.zeros((B, FR.W.shape[1]), dtype=np.float64)
    i = 0
    while i < B:
        j = i
        while j < B:
            idx = order[i:j + 1]
            klen = max(1, len(tok_lists[idx[-1]]))       # sorted ascending -> last is longest
            if len(idx) * klen > max_elems and j > i:    # keep at least one user per chunk
                break
            if len(idx) >= cap:
                j += 1; break
            j += 1
        idx = order[i:j]
        tl = [tok_lists[k] for k in idx]; nl = [native_lists[k] for k in idx]
        Zc = R.fold_mu_batch(FR, model, tl, nl, keepA=keepA)
        for k, oi in enumerate(idx):
            Z[oi] = Zc[k]
        i = j
    return Z


def signed_belief(FR, native_ids, toks_stripped, eta, blind=False):
    """z = native_z + eta * SUM over EXPLICIT tokens [ s(value) * memberbag_emb ]. blind -> s==0."""
    z = FR.enc_items([list(native_ids)])[0].numpy().astype(np.float64)
    if blind:
        return z
    acc = np.zeros_like(z)
    for (typ, kind, lvl, fid, val, emb) in toks_stripped:
        if kind != KIND_EXPL:
            continue
        s = s_band(float(val))
        if s != 0.0:
            acc += s * emb.astype(np.float64)
    return z + eta * acc


def _dedup_expl(toks7):
    """Strip V3 7-field tokens to Rung1 6-field and dedup (reuse R._dedup)."""
    return R._dedup([R._strip(t) for t in toks7])


def _region_percentile(FR, z, members, profset):
    """Mean rank-percentile (1=top) of member items in the full decode ranking (profile masked)."""
    s = FR.decode_np(np.asarray(z, np.float64)[None, :])[0].copy()
    if profset:
        s[list(profset)] = -1e18
    order = np.argsort(-s)
    rankfrac = np.empty(len(s)); rankfrac[order] = 1.0 - np.arange(len(s)) / len(s)
    m = [j for j in members if j < len(s)]
    return float(np.mean(rankfrac[m])) if m else float("nan")


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


# ---------- dislike-heavy reveal builder (reuses V3Sampler primitives) ----------
def dislike_heavy_reveal(S, rec, rng):
    """Adversarial dislike-heavy interview: ask the user's DISLIKED items + disliked concept/attr/
    entity regions (real negative aggregate value), plus a minority of likes (mixed). Values come
    from the user's real ratings via S.emit->region_value; embeddings are leak-free member-bags.
    Returns (tokens7, native_liked_ids_asked)."""
    known = rec["known"]; cache = rec["cache"]
    ks = cache["ks"]; kset = cache["kset"]
    r = np.array([known[j] for j in ks], float); mu = float(r.mean())
    cr = {int(j): float(known[j] - mu) for j in ks}
    trait = S.user_traits(rng)
    toks = []
    disliked = [int(j) for j in ks if known[int(j)] <= 2]
    liked = [int(j) for j in ks if known[int(j)] >= 4]
    # ask ALL disliked items + a MINORITY of likes -> dislike-heavy but mixed
    ask_likes = liked[: max(1, len(disliked) // 2)] if liked else []
    item_asked_like = []
    for j in disliked + ask_likes:
        toks += S.emit(known, cr, mu, TYPE_ITEM, int(j), trait, rng, force_level=LVL_KW)
        if known[int(j)] >= 4:
            item_asked_like.append(int(j))
    # disliked concept/attr/entity regions: emit only where the user's aggregate value is negative
    for c in cache["concept_top"]:
        v, fid = S.region_value(known, cr, mu, TYPE_CONCEPT, int(c), LVL_KW, rng)
        if v is not None and v < 0:
            toks += S.emit(known, cr, mu, TYPE_CONCEPT, int(c), trait, rng, force_level=LVL_KW)
    for ak in cache["attr_hit"]:
        v, fid = S.region_value(known, cr, mu, TYPE_ATTR, ak, LVL_KW, rng)
        if v is not None and v < 0:
            toks += S.emit(known, cr, mu, TYPE_ATTR, ak, trait, rng, force_level=LVL_KW)
    for eid in cache["ent_hit"]:
        v, fid = S.region_value(known, cr, mu, TYPE_ENTITY, eid, LVL_KW, rng)
        if v is not None and v < 0:
            toks += S.emit(known, cr, mu, TYPE_ENTITY, eid, trait, rng, force_level=LVL_KW)
    rng.shuffle(toks)
    return toks, item_asked_like


# ============================================================ main
def main():
    t0 = time.time()
    print("[checks] loading frozen RecVAE + sampler ...", flush=True)
    D = L.G.load_data(); FR = L.Frozen(D); S = R.V3Sampler(D, FR)

    # ---- SAME held-out val cohort as the pilot (seed 0) ----
    seed = 0
    tr, va, _te = R.load_split(D, n_train=2000, n_val=1000, n_test=0, seed=seed + 1234)
    val_recs = R.prep_users(va, np.random.default_rng(seed + 1), S)
    print(f"[checks] val cohort {len(val_recs)} users  [{time.time()-t0:.0f}s]", flush=True)

    model = R.Rung1Encoder()
    model.load_state_dict(torch.load(BEST, map_location="cpu")["model"]); model.eval()

    results = {"n_val_users": len(val_recs)}

    # ======================================================== CHECK 1: decomposition
    print("\n[CHECK 1] learned-vs-inherited decomposition ...", flush=True)
    dec_rows = []
    contexts = [("k2", "interview", 2), ("k4", "interview", 4),
                ("k8", "interview", 8), ("full", "clean", 0)]
    for cname, mode, budget in contexts:
        rng = np.random.default_rng(1000 + budget)
        tls, nls, recs2 = [], [], []
        for rec in val_recs:
            toks, native = S.build_reveal(rec["known"], mode, budget, rng, cache=rec["cache"])
            if toks:
                tls.append(toks); nls.append(native); recs2.append(rec)
        # (a) native-only
        Zn = native_only_batch(FR, nls)
        # (b) tokens-only (native dropped)
        Zt = fold_mu_chunked(FR, model, tls, nls, keepA=0.0)
        # (c) full
        Zf = fold_mu_chunked(FR, model, tls, nls, keepA=1.0)
        na, to, fu = [], [], []
        n_native_empty = 0
        for i, rec in enumerate(recs2):
            if len(nls[i]) == 0:
                n_native_empty += 1
            a = ndcg_of(FR, Zn[i], rec); b = ndcg_of(FR, Zt[i], rec); c = ndcg_of(FR, Zf[i], rec)
            if None not in (a, b, c):
                na.append(a); to.append(b); fu.append(c)
        row = dict(context=cname, n=len(fu),
                   native_only=float(np.mean(na)), tokens_only=float(np.mean(to)),
                   full=float(np.mean(fu)), rho_contrib=float(np.mean(fu) - np.mean(na)),
                   mean_native_ids=float(np.mean([len(x) for x in nls])),
                   frac_native_empty=float(n_native_empty / max(len(recs2), 1)))
        dec_rows.append(row)
        print(f"  {cname:5s} n={row['n']:4d} native={row['native_only']:.4f} "
              f"tokens={row['tokens_only']:.4f} full={row['full']:.4f} "
              f"rho+={row['rho_contrib']:+.4f} (native_ids~{row['mean_native_ids']:.1f}, "
              f"empty {row['frac_native_empty']*100:.0f}%)", flush=True)

    # mixed-default context (reproduce pilot 0.340 full / 0.271 tokens-only)
    rng = np.random.default_rng(seed + 21)
    tls, nls, recs2 = [], [], []
    for rec in val_recs:
        toks, native = R.sample_reveal(S, rec, 0.3, rng)
        if toks:
            tls.append(toks); nls.append(native); recs2.append(rec)
    Zn = native_only_batch(FR, nls)
    Zt = fold_mu_chunked(FR, model, tls, nls, keepA=0.0)
    Zf = fold_mu_chunked(FR, model, tls, nls, keepA=1.0)
    na, to, fu = [], [], []
    for i, rec in enumerate(recs2):
        a = ndcg_of(FR, Zn[i], rec); b = ndcg_of(FR, Zt[i], rec); c = ndcg_of(FR, Zf[i], rec)
        if None not in (a, b, c):
            na.append(a); to.append(b); fu.append(c)
    mixed = dict(context="mixed_default", n=len(fu),
                 native_only=float(np.mean(na)), tokens_only=float(np.mean(to)),
                 full=float(np.mean(fu)), rho_contrib=float(np.mean(fu) - np.mean(na)))
    dec_rows.append(mixed)
    print(f"  mixed n={mixed['n']:4d} native={mixed['native_only']:.4f} "
          f"tokens={mixed['tokens_only']:.4f} full={mixed['full']:.4f} "
          f"rho+={mixed['rho_contrib']:+.4f}", flush=True)
    results["check1_decomposition"] = dec_rows

    # ======================================================== CHECK 2: value informativeness
    print("\n[CHECK 2] hard-wired signed-value belief (training-free) ...", flush=True)
    etas = [2.0, 4.0, 8.0]
    ctx_specs = {"on_profile": "interview", "dislike_heavy": "adversarial"}
    check2 = {}
    for ctx_name, kind in ctx_specs.items():
        rng = np.random.default_rng(2000 + hash(ctx_name) % 999)
        prepared = []  # (rec, dedup_expl_tokens, native_ids)
        n_expl, n_neg_expl = 0, 0
        for rec in val_recs:
            if kind == "interview":
                toks, native = S.build_reveal(rec["known"], "interview", 8, rng, cache=rec["cache"])
            else:
                toks, native = dislike_heavy_reveal(S, rec, rng)
            if not toks:
                continue
            ded = _dedup_expl(toks)
            expl = [t for t in ded if t[1] == KIND_EXPL]
            n_expl += len(expl)
            n_neg_expl += sum(1 for t in expl if s_band(float(t[4])) < 0)
            prepared.append((rec, ded, native))
        # blind baseline (native alone)
        blind = [ndcg_of(FR, signed_belief(FR, nat, ded, 0.0, blind=True), rec)
                 for (rec, ded, nat) in prepared]
        blind = [x for x in blind if x is not None]
        blind_mean = float(np.mean(blind))
        eta_rows = []
        for eta in etas:
            deltas, signed_vals = [], []
            for (rec, ded, nat) in prepared:
                zc = signed_belief(FR, nat, ded, eta, blind=False)
                zb = signed_belief(FR, nat, ded, eta, blind=True)   # == native alone
                sc = ndcg_of(FR, zc, rec); sb = ndcg_of(FR, zb, rec)
                if None not in (sc, sb):
                    deltas.append(sc - sb); signed_vals.append(sc)
            ci = boot_ci(deltas, seed=7)
            eta_rows.append(dict(eta=eta, signed_ndcg=float(np.mean(signed_vals)),
                                 value_delta=float(np.mean(deltas)), ci=list(ci),
                                 helps=bool(ci[0] > 0)))
            print(f"  {ctx_name:13s} eta={eta:.0f} blind={blind_mean:.4f} "
                  f"signed={np.mean(signed_vals):.4f} dNDCG={np.mean(deltas):+.4f} "
                  f"CI[{ci[0]:+.4f},{ci[1]:+.4f}] helps={ci[0]>0}", flush=True)
        check2[ctx_name] = dict(blind_native_only=blind_mean, n=len(blind),
                                mean_expl_per_user=float(n_expl / max(len(prepared), 1)),
                                frac_negative_expl=float(n_neg_expl / max(n_expl, 1)),
                                by_eta=eta_rows)

    # ---- fold-level IG2 on dislike-heavy: are held disliked items down-ranked / likes up-ranked? ----
    print("\n[CHECK 2b] fold-level IG2 percentile on dislike-heavy ...", flush=True)
    rng = np.random.default_rng(4242)
    eta_ig = 4.0
    dis_dn, lik_up = [], []       # signed - blind percentile shift
    n_used = 0
    for rec in val_recs:
        toks, native = dislike_heavy_reveal(S, rec, rng)
        if not toks:
            continue
        ded = _dedup_expl(toks)
        held_dis = [int(j) for j, r in rec["held"].items() if r <= 2]
        held_lik = [int(j) for j in rec["held_likes"]]
        if not held_dis:
            continue
        prof = set(rec["known"].keys())
        z_signed = signed_belief(FR, native, ded, eta_ig, blind=False)
        z_blind = signed_belief(FR, native, ded, eta_ig, blind=True)
        p_dis_s = _region_percentile(FR, z_signed, held_dis, prof)
        p_dis_b = _region_percentile(FR, z_blind, held_dis, prof)
        dis_dn.append(p_dis_s - p_dis_b)   # NEGATIVE = pushed down (good)
        if held_lik:
            p_lik_s = _region_percentile(FR, z_signed, held_lik, prof)
            p_lik_b = _region_percentile(FR, z_blind, held_lik, prof)
            lik_up.append(p_lik_s - p_lik_b)  # POSITIVE = pushed up (good)
        n_used += 1
    ci_dis = boot_ci(dis_dn, seed=9); ci_lik = boot_ci(lik_up, seed=10)
    ig2 = dict(n_users=n_used, eta=eta_ig,
               held_disliked_pctile_shift=float(np.mean(dis_dn)), ci_disliked=list(ci_dis),
               disliked_pushed_down=bool(ci_dis[1] < 0),
               held_liked_pctile_shift=float(np.mean(lik_up)) if lik_up else float("nan"),
               ci_liked=list(ci_lik),
               liked_pushed_up=bool(ci_lik[0] > 0) if lik_up else False)
    results["check2_value"] = check2
    results["check2b_ig2_dislike_heavy"] = ig2
    print(f"  held-disliked pctile shift {ig2['held_disliked_pctile_shift']:+.4f} "
          f"CI[{ci_dis[0]:+.4f},{ci_dis[1]:+.4f}] down={ig2['disliked_pushed_down']}", flush=True)
    print(f"  held-liked   pctile shift {ig2['held_liked_pctile_shift']:+.4f} "
          f"CI[{ci_lik[0]:+.4f},{ci_lik[1]:+.4f}] up={ig2['liked_pushed_up']}", flush=True)

    # ---- verdicts ----
    d_full = mixed["rho_contrib"]
    best_dislike_delta = max((r["value_delta"] for r in check2["dislike_heavy"]["by_eta"]))
    best_onprofile_delta = max((r["value_delta"] for r in check2["on_profile"]["by_eta"]))
    results["verdicts"] = dict(
        rho_contribution_mixed=float(d_full),
        best_value_delta_on_profile=float(best_onprofile_delta),
        best_value_delta_dislike_heavy=float(best_dislike_delta),
        decisive_number_value_dislike_heavy=float(best_dislike_delta),
    )
    results["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(results, open(f"{OUT}/checks.json", "w"), indent=2, default=float)
    print(f"\n[checks] wrote {OUT}/checks.json  [{results['minutes']}m]", flush=True)
    return results


if __name__ == "__main__":
    main()
