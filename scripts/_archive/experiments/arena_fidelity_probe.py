"""arena_fidelity_probe.py -- DIRECT FIDELITY-TRUST PROBE on the deployed fold-v3.1 (author-driven).

For ~200 population val users: from an identical warm prior (the user's liked items), fold ONE explicit
token that differs ONLY in fidelity class (data / ease / llm-style) at a FIXED stated value (loved).
Measure belief-shift magnitude |delta z| and region-score change per class, with bootstrap CIs.

Design expectation: data >= ease > llm-style (the fold trusts high-fidelity answers more). If the classes
shift EQUALLY, FLAG loudly -- policies must then not reason about answer quality via the fold, and
vividness-aware selection loses its instrument-side mechanism. Writes .cache/arena/fidelity_probe.json.
NO LLM calls.
"""
import os, sys, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings
warnings.filterwarnings("ignore")

import arena_core as AC
import i25_fold_v31 as FV31
from arena_core import (TYPE_CONCEPT, KIND_EXPL, KIND_IMPL, LVL_ROUGH, LVL_KW,
                        FID_DATA, FID_EASE, FID_LLM)

FIDS = [("data", FID_DATA), ("ease", FID_EASE), ("llm", FID_LLM)]


def _ci(x, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(3000)]
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main(n_users=200):
    ar = AC.Arena()
    coh = AC.make_cohorts(ar, n_train=50, n_devval=n_users, n_devtest=10)
    users = coh["devval"][:n_users]
    dz = {k: [] for k, _ in FIDS}
    dsc = {k: [] for k, _ in FIDS}
    n = 0
    for rec in users:
        known = rec["known"]
        ks = list(known.keys())
        liked = [int(j) for j in ks if known[j] >= 4][:6]
        if len(liked) < 2:
            continue
        # top concept by revealed member count = the region we craft a value token about
        ind = np.zeros(ar.uni.ni, np.float64)
        for j in ks:
            if 0 <= int(j) < ar.uni.ni:
                ind[int(j)] = 1.0
        mass = np.asarray(ar.uni.tagM.dot(ind)).ravel()
        ctag = int(np.argmax(mass))
        members = ar.region_members(ctag)                    # concept qidx == tag rid (layout)
        if len(members) == 0:
            continue
        emb = ar.Qemb[ctag]
        native = liked
        # identical warm prior: native liked items, NO explicit token
        z0 = FV31.fold_np(ar.FR, ar.model, [], native)
        base = float(np.mean(ar.FR.decode_np(z0[None, :])[0][members]))
        for name, fid in FIDS:
            tok = [(TYPE_CONCEPT, KIND_EXPL, LVL_ROUGH, 0.0, fid, 1.0, emb, 0.0)]  # fixed value=loved
            z = FV31.fold_np(ar.FR, ar.model, tok, native)
            dz[name].append(float(np.linalg.norm(z - z0)))
            dsc[name].append(float(np.mean(ar.FR.decode_np(z[None, :])[0][members]) - base))
        n += 1
    out = {"n_users": n, "value_token": "loved(+1)", "region": "top concept", "prior": "native liked items"}
    print(f"\n=== FIDELITY-TRUST PROBE (n={n} population val users; fixed value=loved, top-concept region) ===",
          flush=True)
    print(f"{'fid':8s} {'mean|dz|':>12s} {'95% CI':>26s}   {'region-score shift':>20s} {'95% CI':>26s}",
          flush=True)
    for name, _ in FIDS:
        m, lo, hi = _ci(dz[name]); ms, los, his = _ci(dsc[name], seed=1)
        out[name] = dict(dz_mean=m, dz_ci=[lo, hi], dscore_mean=ms, dscore_ci=[los, his])
        print(f"{name:8s} {m:12.4f} [{lo:+.4f},{hi:+.4f}]   {ms:20.4f} [{los:+.4f},{his:+.4f}]",
              flush=True)
    # verdict: is data >= ease > llm, or are they equal?
    d, e, l = out["data"]["dz_mean"], out["ease"]["dz_mean"], out["llm"]["dz_mean"]
    spread = max(d, e, l) - min(d, e, l)
    rel = spread / (np.mean([d, e, l]) + 1e-9)
    monotone = d >= e >= l
    out["spread"] = spread; out["rel_spread"] = float(rel); out["monotone_data_ge_ease_ge_llm"] = bool(monotone)
    if rel < 0.05:
        out["verdict"] = ("FLAG: fidelity classes shift ~EQUALLY (rel spread %.1f%%) -- the fold does "
                          "NOT trust high-fidelity answers more; policies must not reason about answer "
                          "quality via the fold; vividness-aware selection has no instrument-side "
                          "mechanism." % (100 * rel))
    elif monotone:
        out["verdict"] = ("OK: data >= ease > llm-style shift magnitudes (rel spread %.1f%%) -- the fold "
                          "trusts high-fidelity answers more, as designed." % (100 * rel))
    else:
        out["verdict"] = ("PARTIAL: fidelity classes differ (rel spread %.1f%%) but NOT monotone "
                          "data>=ease>=llm (order d=%.4f e=%.4f l=%.4f)." % (100 * rel, d, e, l))
    print("\n" + out["verdict"], flush=True)
    json.dump(out, open(f"{AC.CACHE_DIR}/fidelity_probe.json", "w"), indent=1)
    print(f"\n[probe] wrote {AC.CACHE_DIR}/fidelity_probe.json", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
