"""arena_falsification.py -- COUNT-vs-CONTENT falsification of the belief fold's per-turn NDCG curve.

Author-directed diagnostic. Tests whether the current FOLD-V3.1 belief (arena_core) fakes the
monotone NDCG-vs-turns curve via TOKEN COUNT (cardinality) rather than answer CONTENT.

The fold is z = native_z + rho([pool, native_z, log1p(ntok)]) (i25_fold_v31.FoldV31.forward):
pool = sum_k phi(token_k), ntok = number of tokens. Both grow with the NUMBER of tokens
independent of information content -> a suspected count-confound of monotonicity.

Three curves on the SAME ~400 seed-123 devtest users, NDCG@10 (endpoint metric):
  1. REAL      : fold real answers to first m answered questions, m=0,1,2,4,8 (normal per-turn curve).
  2. DUPLICATE : take the user's first k=4 answered questions' tokens, fold once (x1), then the SAME
                 tokens duplicated x2/x3/x4 (identical CONTENT, zero new information). Any change from
                 x1 = pure count-confound. HEADLINE.
  3. PADDING   : from cold (0 real answers), fold m taste-NEUTRAL no-clue/refusal tokens, m=0,1,2,4,8.
                 SOFT control (no-clue carries a weak 'not-in-world' signal via anti-surprise).

DEV/synthetic ONLY. $0. NO LLM. Fold calls only (no training). The 173 study users are NEVER touched
(make_cohorts excludes study ids; population trU only).

Run:  python scripts/arena_falsification.py
"""
import os, sys, json, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import warnings
warnings.filterwarnings("ignore")

import arena_core as AC
from arena_core import ndcg_at_k

K = 10
N_USERS = 400          # devtest sample size
N_ANS = 8              # answered questions to collect per user (need >=8)
N_REF = 8              # refused questions to collect per user (need >=8)
K_DUP = 4              # duplicate-test base: first 4 answered questions
REAL_M = [0, 1, 2, 4, 8]
PAD_M = [0, 1, 2, 4, 8]
DUP_FACTORS = [1, 2, 3, 4]
OUT_JSON = ".cache/arena/falsification_test.json"
BUILD_MD = "experiments/ARENA_BUILD.md"


def collect_asked(ar, rec, ctx):
    """Deterministic per-user ask order; classify questions into answered (know>=1) vs refused
    (know==0). Return (answered_qidx, refused_qidx), first N_ANS / N_REF each."""
    uid = rec["u"]
    rng = np.random.default_rng(AC.SEED * 7919 + int(uid))
    order = rng.permutation(ar.nQ)
    ans, ref = [], []
    for qi in order:
        qi = int(qi)
        if ar.answered(uid, qi, ctx):
            if len(ans) < N_ANS:
                ans.append(qi)
        else:
            if len(ref) < N_REF:
                ref.append(qi)
        if len(ans) >= N_ANS and len(ref) >= N_REF:
            break
    return ans, ref


def toks_native_for(ar, uid, ctx, qids):
    """Concatenate FOLD-V3.1 tokens for a list of asked questions + native fold-in items (liked
    answered bank items), exactly as arena_core.belief_z does."""
    toks, nat = [], []
    for qi in qids:
        toks += ar.tokens_for(uid, qi, ctx)
        if ar.answered(uid, qi, ctx) and ar.is_liked_item(uid, qi, ctx):
            nat.append(int(ar.uni.bank[qi - ar.off_item]))
    return toks, nat


def ndcg_batch(ar, tok_lists, nat_lists, held_list, prof_list):
    Z = ar.belief_z_batch(tok_lists, nat_lists)
    out = []
    for b in range(Z.shape[0]):
        v = ndcg_at_k(ar.FR, Z[b], held_list[b], prof_list[b], K)
        out.append(v)
    return out


def mean_ok(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    return float(a.mean()) if len(a) else float("nan"), len(a)


def main():
    t0 = time.time()
    ar = AC.Arena()
    # canonical main-run cohort prefix (n_train=1000, n_devval=80); n_devtest=400 keeps the first 160
    # identical to the canonical devtest by the prefix property of make_cohorts.
    coh = AC.make_cohorts(ar, n_train=1000, n_devval=80, n_devtest=N_USERS)
    cfg = coh["cfg"]
    ar.prefill_answers(coh["devtest"], "devtest_falsif", cfg, verbose=True)

    # ---- select users with >= N_ANS answered AND >= N_REF refused (fully paired across all curves) --
    users = []          # dict(uid, ctx, held, prof, ans, ref)
    for rec in coh["devtest"]:
        ctx = ar.user_ctx(rec)
        ans, ref = collect_asked(ar, rec, ctx)
        if len(ans) >= N_ANS and len(ref) >= N_REF:
            users.append(dict(uid=rec["u"], ctx=ctx, held=rec["held"],
                              prof=set(rec["known"].keys()), ans=ans, ref=ref))
    print(f"[falsif] {len(users)}/{len(coh['devtest'])} devtest users usable "
          f"(>= {N_ANS} answered & >= {N_REF} refused)  [{time.time()-t0:.0f}s]", flush=True)
    held = [u["held"] for u in users]
    prof = [u["prof"] for u in users]

    # =============================================================== CURVE 1: REAL informative
    real = {}
    for m in REAL_M:
        tls, nls = [], []
        for u in users:
            toks, nat = toks_native_for(ar, u["uid"], u["ctx"], u["ans"][:m])
            tls.append(toks); nls.append(nat)
        v = ndcg_batch(ar, tls, nls, held, prof)
        mu, n = mean_ok(v)
        real[m] = mu
        print(f"[falsif] REAL      m={m:<2d} answered  NDCG@10 {mu:.4f}  (n={n})", flush=True)

    # =============================================================== CURVE 2: DUPLICATE (headline)
    # base = tokens/native for first K_DUP answered questions; duplicate the TOKENS x1..x4 (native
    # held FIXED -> isolates the token-count / pool-magnitude confound in rho([pool,native_z,ntok])).
    dup = {}
    base_toks, base_nats = [], []
    ntok_base = []
    for u in users:
        toks, nat = toks_native_for(ar, u["uid"], u["ctx"], u["ans"][:K_DUP])
        base_toks.append(toks); base_nats.append(nat); ntok_base.append(len(toks))
    for f in DUP_FACTORS:
        tls = [bt * f for bt in base_toks]
        v = ndcg_batch(ar, tls, base_nats, held, prof)
        mu, n = mean_ok(v)
        dup[f] = mu
        print(f"[falsif] DUPLICATE x{f} ({f}x identical content)  NDCG@10 {mu:.4f}  (n={n})", flush=True)
    mean_ntok = float(np.mean(ntok_base))

    # =============================================================== CURVE 3: UNINFORMATIVE padding
    pad = {}
    for m in PAD_M:
        tls, nls = [], []
        for u in users:
            toks, nat = toks_native_for(ar, u["uid"], u["ctx"], u["ref"][:m])  # refusals -> no-clue toks
            tls.append(toks); nls.append(nat)                                   # nat empty (refusals)
        v = ndcg_batch(ar, tls, nls, held, prof)
        mu, n = mean_ok(v)
        pad[m] = mu
        print(f"[falsif] PADDING   m={m:<2d} no-clue   NDCG@10 {mu:.4f}  (n={n})", flush=True)

    # =============================================================== deltas + confound fractions
    real_rise = real[8] - real[0]
    dup_rise = dup[4] - dup[1]
    pad_rise = pad[8] - pad[0]
    frac_dup = dup_rise / real_rise if abs(real_rise) > 1e-9 else float("nan")
    frac_pad = pad_rise / real_rise if abs(real_rise) > 1e-9 else float("nan")

    R = dict(
        meta=dict(K=K, n_users=len(users), n_devtest=len(coh["devtest"]),
                  n_answered=N_ANS, n_refused=N_REF, k_dup=K_DUP, seed=AC.SEED,
                  fold_ckpt=AC.FOLD_CKPT, fold_val=ar.fold_state.get("best_val"),
                  mean_base_tokens_for_dup=mean_ntok),
        curve1_real={str(m): real[m] for m in REAL_M},
        curve2_duplicate={f"x{f}": dup[f] for f in DUP_FACTORS},
        curve3_padding={str(m): pad[m] for m in PAD_M},
        deltas=dict(real_rise_0to8=real_rise, dup_rise_x1tox4=dup_rise, pad_rise_0to8=pad_rise,
                    frac_real_rise_explained_by_dup=frac_dup,
                    frac_real_rise_explained_by_pad=frac_pad),
    )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(R, open(OUT_JSON, "w"), indent=1, default=float)
    print(f"\n[falsif] DUPLICATE-TEST delta (x4 - x1): {dup_rise:+.4f}  "
          f"(identical content, {mean_ntok:.1f} -> {4*mean_ntok:.1f} tokens)", flush=True)
    print(f"[falsif] PADDING rise (m0 -> m8):        {pad_rise:+.4f}", flush=True)
    print(f"[falsif] REAL rise   (m0 -> m8):         {real_rise:+.4f}", flush=True)
    print(f"[falsif] frac of real rise = dup {frac_dup:+.2%} | pad {frac_pad:+.2%}", flush=True)

    _append_md(R)
    print(f"[falsif] wrote {OUT_JSON} + appended {BUILD_MD}  [{time.time()-t0:.0f}s]", flush=True)
    return R


def _append_md(R):
    m = R["meta"]; d = R["deltas"]
    c1, c2, c3 = R["curve1_real"], R["curve2_duplicate"], R["curve3_padding"]
    confounded = abs(d["dup_rise_x1tox4"]) >= 0.005
    o = []
    o.append("\n\n## FALSIFICATION: count vs content\n\n")
    o.append(f"Diagnostic `scripts/arena_falsification.py`: does the FOLD-V3.1 belief "
             f"(`{m['fold_ckpt']}`, val {m['fold_val']:.4f}) fake the monotone NDCG-vs-turns curve via "
             f"TOKEN COUNT rather than answer CONTENT? The fold is "
             f"`z = native_z + rho([pool, native_z, log1p(ntok)])` -- pool (token sum) and ntok both "
             f"grow with the NUMBER of tokens. NDCG@{m['K']}, {m['n_users']} seed-{m['seed']} devtest "
             f"users (>= {m['n_answered']} answered & >= {m['n_refused']} refused each), $0, no LLM, "
             f"fold calls only. The 173 study users untouched.\n\n")

    o.append("**Curve 1 -- REAL informative (reference):** real answers to first m answered "
             "questions.\n\n")
    o.append("| m answered | 0 | 1 | 2 | 4 | 8 |\n|---|--:|--:|--:|--:|--:|\n")
    o.append(f"| NDCG@{m['K']} | {c1['0']:.4f} | {c1['1']:.4f} | {c1['2']:.4f} | {c1['4']:.4f} | "
             f"{c1['8']:.4f} |\n\n")

    o.append(f"**Curve 2 -- DUPLICATE TEST (headline):** first {m['k_dup']} answered questions' tokens "
             f"(~{m['mean_base_tokens_for_dup']:.1f} tokens), folded once then duplicated x2/x3/x4 "
             f"(IDENTICAL content, zero new information; native fold-in held fixed). A content-driven "
             f"fold is FLAT here by construction; any rise = pure count-confound.\n\n")
    o.append("| duplication | x1 | x2 | x3 | x4 | delta (x4-x1) |\n|---|--:|--:|--:|--:|--:|\n")
    o.append(f"| NDCG@{m['K']} | {c2['x1']:.4f} | {c2['x2']:.4f} | {c2['x3']:.4f} | {c2['x4']:.4f} | "
             f"{d['dup_rise_x1tox4']:+.4f} |\n\n")

    o.append("**Curve 3 -- UNINFORMATIVE padding (soft control):** from cold, m taste-neutral "
             "no-clue/refusal tokens (caveat: no-clue carries a weak 'not-in-world' anti-surprise "
             "signal, so this is softer than the duplicate test). A genuine efficiency curve is "
             "~flat; a rise = count-confound.\n\n")
    o.append("| m no-clue | 0 | 1 | 2 | 4 | 8 |\n|---|--:|--:|--:|--:|--:|\n")
    o.append(f"| NDCG@{m['K']} | {c3['0']:.4f} | {c3['1']:.4f} | {c3['2']:.4f} | {c3['4']:.4f} | "
             f"{c3['8']:.4f} |\n\n")

    o.append("**Deltas:**\n\n")
    o.append(f"- REAL rise (m0->m8): {d['real_rise_0to8']:+.4f}\n")
    o.append(f"- DUPLICATE rise (x1->x4, identical content): {d['dup_rise_x1tox4']:+.4f} "
             f"= {d['frac_real_rise_explained_by_dup']:+.1%} of the real rise\n")
    o.append(f"- PADDING rise (m0->m8, no-clue): {d['pad_rise_0to8']:+.4f} "
             f"= {d['frac_real_rise_explained_by_pad']:+.1%} of the real rise\n\n")

    verdict = ("COUNT-CONFOUNDED" if confounded else "NOT count-confounded (content-driven)")
    o.append(f"**VERDICT: {verdict}.** Duplicating identical content moves NDCG@{m['K']} by "
             f"{d['dup_rise_x1tox4']:+.4f} ({d['frac_real_rise_explained_by_dup']:+.1%} of the real "
             f"per-turn rise); uninformative no-clue padding moves it {d['pad_rise_0to8']:+.4f} "
             f"({d['frac_real_rise_explained_by_pad']:+.1%}). The bulk of the real per-turn rise is "
             f"{'NOT ' if not confounded else ''}explained by token count alone.\n")
    open(BUILD_MD, "a", encoding="utf-8").write("".join(o))
    assert os.path.exists(BUILD_MD)


if __name__ == "__main__":
    main()
