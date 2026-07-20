"""kmap_ceiling.py -- THE CEILING CHECK (Fable, 2026-07-08).

Question: is blind discovery capped by the K-map's ESTIMATION (fixable with more/better evidence) or
by the INTRINSIC predictability of 'which popular items did this user happen to rate' (not fixable by
any belief)? Method: give the K-map the MAXIMUM possible evidence -- split-half over the 160-item
probe bank (~80 observed outcomes, vs the interview's <=24 turns) -- and measure held-out structural
AUC. If the asymptote sits near the t=16 number (0.683) and the privileged feature ceiling (~0.695),
the ceiling is intrinsic: rated-ness beyond popularity is mostly idiosyncratic chance, and no
in-interview belief can find a user's rated items efficiently => open recall is information-
theoretically forced, not a UX preference. No LLM calls.

Run: python scripts/kmap_ceiling.py
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
import kmap_validate as V
import llm_answerability_gate as G

def main():
    D = G.load_data()
    split = G.build_split(D)
    KM = V.Kmap(D)

    # rebuild BANK exactly as kmap_validate.main does
    import collections
    gate = json.load(open(V.GATE_GRID))["users"]
    cover = collections.Counter()
    for us in gate:
        u = int(us)
        if u not in split:
            continue
        kn, ho = split[u]; rat = dict(D["rat_by_u"][u])
        known = {j: rat[j] for j in kn if j in rat}
        like = [j for j in kn if rat.get(j, 0) >= 4]
        tlike = set(j for j in ho if rat.get(j, 0) >= 4)
        if not like or not tlike or len(known) < 4:
            continue
        for j in known:
            cover[j] += 1
    BANK = np.array([j for j, c in cover.most_common() if c >= 3][:160], dtype=np.int64)
    V.BANK = BANK

    users = V.build_cohort(D, split)
    rng = np.random.default_rng(0)

    auc_full, auc_pop = [], []
    for rec in users:
        lab = {int(j): (1 if int(j) in rec["known"] else 0) for j in BANK}
        J = np.array(sorted(lab.keys()), dtype=np.int64)
        y = np.array([lab[int(j)] for j in J])
        if y.min() == y.max():
            continue
        perm = rng.permutation(len(J))
        aucs_u, pops_u = [], []
        for half in (perm[: len(J) // 2], perm[len(J) // 2:]):
            hold = np.setdiff1d(np.arange(len(J)), half)
            J_ev, y_ev = J[half], y[half]
            Jh, yh = J[hold], y[hold]
            if yh.min() == yh.max():
                continue
            alpha, k = KM.infer(J_ev, y_ev)
            a1 = V.user_auc(yh, KM.score(Jh, alpha, k, True, True))
            a0 = V.user_auc(yh, KM.b_of(Jh))
            if a1 is not None and a0 is not None:
                aucs_u.append(a1); pops_u.append(a0)
        if aucs_u:
            auc_full.append(np.mean(aucs_u)); auc_pop.append(np.mean(pops_u))

    auc_full = np.array(auc_full); auc_pop = np.array(auc_pop)
    m = float(auc_full.mean()); lo, hi = V.boot_mean(auc_full)
    mp = float(auc_pop.mean()); lop, hip = V.boot_mean(auc_pop)
    d = auc_full - auc_pop
    md = float(d.mean()); lod, hid = V.boot_mean(d)
    out = dict(n=len(auc_full),
               asymptote_full=dict(mean=m, ci=[lo, hi]),
               popularity=dict(mean=mp, ci=[lop, hip]),
               delta=dict(mean=md, ci=[lod, hid]),
               reference=dict(t16_full=0.683, priv_pmodel_true=0.695, t8_full=0.669))
    print(f"[ceiling] n={len(auc_full)} users")
    print(f"[ceiling] K-map ASYMPTOTE (split-half, ~80 events): AUC {m:.3f} [{lo:.3f},{hi:.3f}]")
    print(f"[ceiling] popularity-only same protocol:            AUC {mp:.3f} [{lop:.3f},{hip:.3f}]")
    print(f"[ceiling] user-term delta at asymptote:             {md:+.3f} [{lod:+.3f},{hid:+.3f}]")
    print(f"[ceiling] references: full@t8 0.669, full@t16 0.683, privileged pmodel-true ~0.695")
    json.dump(out, open("experiments/kmap_ceiling.json", "w"), indent=1)
    print("[ceiling] saved experiments/kmap_ceiling.json")

if __name__ == "__main__":
    main()
