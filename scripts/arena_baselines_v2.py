"""arena_baselines_v2.py -- STANDARD cold-start elicitation baselines on the SYNTHETIC population,
per-turn NDCG@10 (turns 1..10, headline turn 8), on a large 14k/3k/3k split. DEV/synthetic ONLY;
the 173/300 LLM-judged users are NEVER touched. NO LLM calls. $0. Deterministic (seed 123).

Reuses the GATED v2.1 arena world (arena_core.Arena), fold-v3.1 belief, the 2,428-question universe,
and the audited run_policy / ndcg_at_k_batch machinery.

BASELINES (all per-turn NDCG@10 on TEST):
  COLD          - turn 0 (ask nothing); reference.
  RANDOM        - uniform over unasked questions each turn; 5 seeds, mean +/- std.
  POPULARITY    - static order by catalog popularity mass q_popmass (most-popular first); deterministic.
  INFO-GAIN     - static order by EXPECTED taste NDCG@10 gain per question (single-pass myopic
                  expected-gain over a TRAIN subsample; the Paper-B EIG idea, non-greedy). Deterministic.
                  [Port note below.]
  FULL-PROFILE  - fold ALL known ratings (ceiling anchor).

INFO-GAIN PORT NOTE: Paper B's deployable info-gain (encoder_realizable.py 'infogain_EIG') ranks a
candidate by the EXPECTED taste-coverage after folding its (unknown) answer, marginalized over the
belief's predicted like/dislike -- i.e. expected reduction in taste-posterior uncertainty, NOT
answerability entropy. That policy operates over a user's own profile items; it does not port
verbatim to the arena's 2,428-question universe + fold belief. Per the task's stated fallback we
implement expected-taste-information-gain DIRECTLY as the expected 1-question NDCG@10 gain of each
question, estimated cold on a TRAIN subsample (arena_policies.prescreen_gains), and order questions
by it descending. This is the myopic (non-greedy) expected-taste-gain ranking. NOT answerability
entropy (that is b1, excluded).
"""
import os, sys, json, time, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings
warnings.filterwarnings("ignore")

import arena_core as AC
import arena_policies as AP
from arena_core import ndcg_at_k_batch
from arena_policies import StaticSeq, B0Cold, Policy, run_policy, train_pop_prior
from i25_fold_v3_sampler import (TYPE_ITEM, KIND_IMPL, KIND_EXPL, LVL_ROUGH, LVL_KW, FID_DATA)

MD = "experiments/ARENA_BUILD.md"
Tmax = 10
K = 10


# ============================================================ RANDOM policy (uniform over unasked)
class RandomPolicy(Policy):
    """Uniform over unasked questions each turn; deterministic per (seed, user)."""
    def __init__(self, seed):
        self.name = f"random_s{seed}"; self.seed = seed
    def start(self, arena, recs, ctxs):
        self.nQ = arena.nQ
        rng = np.random.default_rng(self.seed * 100003 + 7)
        self.perm = [rng.permutation(self.nQ) for _ in range(len(recs))]
        self.ptr = [0] * len(recs)
    def pick(self, arena, i, uid, view, asked, ansf, used, tokens, natives, z_cur, t):
        p = self.perm[i]
        while self.ptr[i] < len(p):
            qi = int(p[self.ptr[i]]); self.ptr[i] += 1
            if qi not in used:
                return qi
        return None


# ============================================================ FULL-PROFILE anchor (fold all known)
def full_profile_ndcg(arena, recs, chunk=400):
    """Fold the ENTIRE known profile of each user (every rated item as an item-question with
    know_well + real centered rating; liked items also enter the native path) and score NDCG@10.
    Surprise field set to 0 for the anchor (non-critical; documented). The ceiling reference."""
    FR = arena.FR
    tok_lists, nat_lists, held, prof = [], [], [], []
    for r in recs:
        known = r["known"]
        mu_known = float(np.mean(list(known.values()))) if known else 3.5
        toks = []; nat = []
        for j, rr in known.items():
            j = int(j)
            if not (0 <= j < FR.Wn.shape[0]):
                continue
            emb = FR.Wn[j].numpy().astype(np.float32)
            cr = float(rr - mu_known)
            toks.append((TYPE_ITEM, KIND_IMPL, LVL_KW, 0.0, FID_DATA, 0.0, emb, 0.0))
            toks.append((TYPE_ITEM, KIND_EXPL, LVL_ROUGH, 0.0, FID_DATA, cr, emb, 0.0))
            if rr >= 4:
                nat.append(j)
        tok_lists.append(toks); nat_lists.append(nat)
        held.append(r["held"]); prof.append(set(known.keys()))
    vals = []
    for s in range(0, len(recs), chunk):
        e = min(s + chunk, len(recs))
        Z = arena.belief_z_batch(tok_lists[s:e], nat_lists[s:e])
        vv = ndcg_at_k_batch(FR, Z, held[s:e], prof[s:e], K)
        vals += [v for v in vv]
    arr = np.array([v for v in vals if v is not None], float)
    return float(np.nanmean(arr)), len(arr)


# ============================================================ per-turn curve extraction
def curve_means(out):
    c = out["curves"][K]
    return [float(np.nanmean(c[:, t])) for t in range(Tmax + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=14000)
    ap.add_argument("--n_devval", type=int, default=3000)
    ap.add_argument("--n_devtest", type=int, default=3000)
    ap.add_argument("--n_train_sub", type=int, default=800,
                    help="TRAIN subsample for priors + info-gain prescreen (compute deviation)")
    ap.add_argument("--random_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    t0 = time.time()

    ar = AC.Arena()
    coh = AC.make_cohorts(ar, n_train=a.n_train, n_devval=a.n_devval, n_devtest=a.n_devtest)
    train, dv, dt = coh["train"], coh["devval"], coh["devtest"]
    print(f"[split] train={len(train)} devval={len(dv)} devtest={len(dt)} (seed {AC.SEED}; "
          f"study ids excluded by construction)", flush=True)
    cfg = coh["cfg"]

    # ---- TRAIN subsample for priors + info-gain prescreen (documented compute deviation) ----
    train_sub = train[:a.n_train_sub]
    ar.prefill_answers(train_sub, "trainsub", dict(**cfg, sub=a.n_train_sub))
    ar.set_pop_prior(train_pop_prior(ar, train_sub))

    # ---- TEST answers (headline cohort) ----
    ar.prefill_answers(dt, "devtest", cfg)

    # ---- baseline orders ----
    pop_order = [int(q) for q in np.argsort(-ar.q_popmass)]                       # POPULARITY
    ig_gain = AP.prescreen_gains(ar, train_sub, K=K, tag="baselines_v2")          # INFO-GAIN
    ig_order = [int(q) for q in np.argsort(-ig_gain)]
    # ENTROPY (classic Rashid/Golbandi): rank by population RATING-ENTROPY of each question's
    # elicited value distribution across TRAIN-sub users. Answered values live in user_table["val"]
    # (int8 per question; -1 = no clue/refused, 0..3 = rating bins). Stacked exactly the way the pop
    # prior stacks user_table["know"]; per-question Shannon entropy (base 2) over the answered bins
    # (val>=0); high entropy = most divisive = informative. Static order, deterministic.
    Vtab = np.stack([ar.user_table(r["u"], r["known"])["val"] for r in train_sub])  # (n_sub, nQ)
    ent = np.zeros(ar.nQ, float)
    for q in range(ar.nQ):
        vq = Vtab[:, q]
        vq = vq[vq >= 0].astype(int)
        if vq.size >= 2:
            counts = np.bincount(vq, minlength=4).astype(float)
            pr = counts[counts > 0] / counts.sum()
            ent[q] = float(-(pr * np.log2(pr)).sum())
    ent_order = [int(q) for q in np.argsort(-ent)]                                # ENTROPY
    print(f"[info-gain] top-5 questions by expected NDCG@10 gain: "
          f"{[ar.q_names[q][:40] for q in ig_order[:5]]}", flush=True)
    print(f"[popularity] top-5 by popmass: {[ar.q_names[q][:40] for q in pop_order[:5]]}", flush=True)
    print(f"[entropy] top-5 by rating-entropy: {[ar.q_names[q][:40] for q in ent_order[:5]]}",
          flush=True)

    # ---- run per-turn arms on TEST ----
    print("\n=== running baselines on TEST ===", flush=True)
    res = {}
    r_cold = run_policy(ar, dt, B0Cold(), Tmax, Ks=(K,)); res["cold"] = curve_means(r_cold)
    print(f"    cold done [{time.time()-t0:.0f}s]", flush=True)
    r_pop = run_policy(ar, dt, StaticSeq("popularity", pop_order), Tmax, Ks=(K,))
    res["popularity"] = curve_means(r_pop)
    print(f"    popularity done [{time.time()-t0:.0f}s]", flush=True)
    r_ig = run_policy(ar, dt, StaticSeq("infogain", ig_order), Tmax, Ks=(K,))
    res["infogain"] = curve_means(r_ig)
    print(f"    info-gain done [{time.time()-t0:.0f}s]", flush=True)
    r_ent = run_policy(ar, dt, StaticSeq("entropy", ent_order), Tmax, Ks=(K,))
    res["entropy"] = curve_means(r_ent)
    print(f"    entropy done [{time.time()-t0:.0f}s]", flush=True)

    rand_curves = []
    for sd in a.random_seeds:
        rr = run_policy(ar, dt, RandomPolicy(sd), Tmax, Ks=(K,))
        rand_curves.append(curve_means(rr))
        print(f"    random seed {sd} done [{time.time()-t0:.0f}s]", flush=True)
    rand_curves = np.array(rand_curves)                                          # (nseed, Tmax+1)
    res["random_mean"] = rand_curves.mean(0).tolist()
    res["random_std"] = rand_curves.std(0).tolist()

    fp_val, fp_n = full_profile_ndcg(ar, dt)
    res["full_profile"] = fp_val
    print(f"    full-profile anchor NDCG@10 = {fp_val:.4f} (n={fp_n}) [{time.time()-t0:.0f}s]",
          flush=True)

    cold0 = res["cold"][0]

    # ---- write table to ARENA_BUILD.md ----
    lines = []
    lines.append("\n\n## BASELINES V2 (test, per-turn @10)\n\n")
    lines.append(f"Standard cold-start elicitation baselines on the SYNTHETIC population, per-turn "
                 f"NDCG@10 (full catalog). Split (seed {AC.SEED}, disjoint, the 173/300 LLM-judged "
                 f"users excluded by construction): TRAIN {len(train)} / VAL {len(dv)} / TEST "
                 f"{len(dt)}. All numbers on TEST. NO LLM calls; $0; gated v2.1 world + fold-v3.1 "
                 f"belief + 2,428-question universe.\n\n")
    lines.append(f"- **COLD** (turn 0, ask nothing) = **{cold0:.4f}** (reference).\n")
    lines.append(f"- **FULL-PROFILE** anchor (fold all known ratings) = **{fp_val:.4f}** (ceiling; "
                 f"n={fp_n}).\n")
    lines.append("- RANDOM = uniform over unasked, mean +/- std over "
                 f"{len(a.random_seeds)} seeds {a.random_seeds}.\n")
    lines.append("- POPULARITY = static order by catalog popularity mass (most-popular first).\n")
    lines.append("- INFO-GAIN = static order by EXPECTED taste NDCG@10 gain per question (myopic, "
                 "non-greedy; Paper-B EIG idea; prescreen on a "
                 f"{a.n_train_sub}-user TRAIN subsample). NOT answerability entropy.\n")
    lines.append("- ENTROPY = static order by RATING-ENTROPY of each question's elicited value "
                 "distribution over the TRAIN subsample (classic Rashid/Golbandi; Shannon base-2 "
                 "over answered bins; most-divisive first).\n\n")
    lines.append("| turn | random (mean +/- std) | popularity | info-gain | entropy |\n")
    lines.append("|--:|--:|--:|--:|--:|\n")
    for t in range(1, Tmax + 1):
        rm = res["random_mean"][t]; rs = res["random_std"][t]
        tag = "  **<- headline (turn 8)**" if t == 8 else ""
        lines.append(f"| {t} | {rm:.4f} +/- {rs:.4f} | {res['popularity'][t]:.4f} | "
                     f"{res['infogain'][t]:.4f} | {res['entropy'][t]:.4f} |{tag}\n")
    lines.append(f"\n(COLD turn-0 = {cold0:.4f}; FULL-PROFILE ceiling = {fp_val:.4f} -- both anchors, "
                 "not per-turn.)\n")
    # lift over cold at turn 8
    l8 = {"random": res["random_mean"][8] - cold0, "popularity": res["popularity"][8] - cold0,
          "infogain": res["infogain"][8] - cold0, "entropy": res["entropy"][8] - cold0}
    best = max(l8, key=l8.get)
    lines.append(f"\n**Turn-8 lift over COLD:** random {l8['random']:+.4f}, popularity "
                 f"{l8['popularity']:+.4f}, info-gain {l8['infogain']:+.4f}, entropy "
                 f"{l8['entropy']:+.4f}. Strongest = **{best}**. "
                 f"Random across-seed std at turn 8 = {res['random_std'][8]:.4f}.\n")
    open(MD, "a", encoding="utf-8").write("".join(lines))

    json.dump({"cfg": dict(n_train=len(train), n_devval=len(dv), n_devtest=len(dt),
                           n_train_sub=a.n_train_sub, random_seeds=a.random_seeds),
               "results": res, "cold0": cold0, "turn8_lift": l8},
              open(f"{AC.CACHE_DIR}/baselines_v2.json", "w"), indent=1)

    print("\n" + "=" * 64, flush=True)
    print("BASELINES V2 (TEST, per-turn NDCG@10)", flush=True)
    print(f"  COLD(t0) = {cold0:.4f}   FULL-PROFILE = {fp_val:.4f}", flush=True)
    print(f"  {'turn':>4} {'random':>18} {'popularity':>11} {'info-gain':>10} {'entropy':>10}",
          flush=True)
    for t in range(1, Tmax + 1):
        print(f"  {t:>4} {res['random_mean'][t]:>10.4f}+/-{res['random_std'][t]:.4f} "
              f"{res['popularity'][t]:>11.4f} {res['infogain'][t]:>10.4f} {res['entropy'][t]:>10.4f}"
              f"{'  <-turn8' if t == 8 else ''}", flush=True)
    print(f"\n  turn-8 lift vs cold: {l8}", flush=True)
    print(f"  strongest baseline: {best}", flush=True)
    print(f"\n[done] wrote {MD} + {AC.CACHE_DIR}/baselines_v2.json [{(time.time()-t0)/60:.1f} min]",
          flush=True)


if __name__ == "__main__":
    main()
