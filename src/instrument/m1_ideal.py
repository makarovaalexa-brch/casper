r"""M1 Stage A2 -- the IDEAL (perfect-volunteering) opener family.

Design sheet: `docs/design/DS_M1_STATE_MATCHING.md`. VALIDATION ONLY.

WHAT THIS IS
------------
`z_full` (Stage A) DEFINES the taste partition -- it is never an interview state, because a fully
folded profile is satiated and has nothing left to ask. The ideal family's interview instead starts
from ONE PERFECT VOLUNTEERED FACT: the single thing a user could say that places them in their taste
cluster most sharply. That makes it comparable with the realistic family (both are cold + a little
evidence) and it is deliberately an ORACLE -- the fact is chosen knowing the user's cluster, which no
deployed system knows at ask time. It is the CEILING arm and is labelled as such everywhere.

Author, 2026-08-17: *"that is why we call it perfect volunteering and expect it to separate people in
maximal way"*.

WHY IDENTIFICATION, NOT DISTANCE
--------------------------------
A one-fact fold and a sixty-item fold sit at very different norms, so "nearest centroid" would
mostly measure how much evidence a state carries. The ideal opener is therefore defined by
IDENTIFICATION: among the facts a user could actually state, take the one whose STATERS are most
concentrated in that user's cluster. A minimum cohort size is enforced, otherwise a film three
people rated wins with purity 1.0 by accident.

NOTHING CLAIRVOYANT: every candidate fact is drawn from the user's own fold-in (items they rated at
the liked level, or concepts they behaviourally endorse). Held-out targets are never read.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import argparse
import json
import time

import numpy as np
import torch

OUT_DIR = os.path.join(_ROOT, "experiments", "battery", "m1")
SEED = 98765
VAL_M0_FULL = 0.15686538118354684
MIN_STATERS = 50          # a fact stated by fewer users than this cannot be a signature. Declared.


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kmeans(X, k, seed=SEED, iters=100):
    """Plain Lloyd's on L2-normalised rows (direction, not magnitude: see the design sheet -- the
    two populations carry very different evidence quantities and norm would dominate)."""
    rng = np.random.default_rng(seed)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    # k-means++ seeding
    cent = [Xn[rng.integers(len(Xn))]]
    for _ in range(k - 1):
        d = np.min(((Xn[:, None, :] - np.asarray(cent)[None]) ** 2).sum(-1), axis=1) \
            if len(cent) < 8 else np.min(2 - 2 * (Xn @ np.asarray(cent).T), axis=1)
        d = np.maximum(d, 0)
        p = d / max(d.sum(), 1e-12)
        cent.append(Xn[rng.choice(len(Xn), p=p)])
    C = np.asarray(cent, np.float32)
    lab = np.zeros(len(Xn), np.int64)
    for _ in range(iters):
        new = np.argmax(Xn @ C.T, axis=1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                v = Xn[m].mean(0)
                C[j] = v / max(np.linalg.norm(v), 1e-9)
    return lab, C


def main() -> None:
    ap = argparse.ArgumentParser(description="M1 Stage A2 -- ideal (perfect-volunteering) openers")
    ap.add_argument("--ks", default="4,6,8,10,12")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    args = ap.parse_args()
    assert args.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"

    torch.set_num_threads(args.threads)
    np.random.seed(SEED)

    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_mean,
                                        CERT_TOWER, CERT_CONC)
    from concept_fold import fold_items_frozen
    from i26_encoder import UNSEEN_LEVEL
    from ndcg_tree import leak_assert

    log("#" * 100)
    log("# M1 STAGE A2 -- the IDEAL opener family (ORACLE / CEILING arm). VALIDATION ONLY.")
    log("#" * 100)

    zf = np.load(os.path.join(OUT_DIR, "z_full.npz"))
    Z_full, rows = zf["z"], zf["rows"]
    log(f"[in] z_full {Z_full.shape} over {len(rows)} validation users")

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m1ideal")
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], "m1ideal/val")
    assert list(np.asarray(E["rows"], np.int64)) == list(rows.astype(np.int64)), \
        "z_full rows are not the env's rows, in order"
    f0, _t = genv_mean(E, [], E["rows"])
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")

    ctx, lvl = E["ctx"], E["lvl"]
    LIKED_MIN, n = 7, len(rows)

    # ---- the facts each user could volunteer ---------------------------------------------------
    # ITEMS: films they rated at the liked band. CONCEPTS: concepts they behaviourally endorse
    # (foldable AND positive value) -- the same SEL rule the interview harness uses.
    item_staters = {}
    user_items = []
    for u, r in enumerate(rows):
        liked = [int(i) for i, L in lvl[int(r)].items() if LIKED_MIN <= int(L) < UNSEEN_LEVEL]
        user_items.append(liked)
        for i in liked:
            item_staters.setdefault(i, []).append(u)

    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    user_concs = [np.flatnonzero(endorse[int(r)]).tolist() for r in rows]
    conc_staters = {}
    for u, cs in enumerate(user_concs):
        for c in cs:
            conc_staters.setdefault(int(c), []).append(u)

    log(f"[facts] items statable by >= {MIN_STATERS} users: "
        f"{sum(1 for v in item_staters.values() if len(v) >= MIN_STATERS)} of {len(item_staters)}")
    log(f"[facts] concepts statable by >= {MIN_STATERS} users: "
        f"{sum(1 for v in conc_staters.values() if len(v) >= MIN_STATERS)} of {len(conc_staters)}")

    results = {}
    for k in [int(x) for x in args.ks.split(",") if x.strip()]:
        lab, C = kmeans(Z_full, k)
        sizes = np.bincount(lab, minlength=k)
        log(f"[k={k}] taste cluster sizes: {sizes.tolist()}")

        # purity of each fact for each cluster, over facts with enough staters
        def best_facts(staters):
            out = {}
            for fid, us in staters.items():
                if len(us) < MIN_STATERS:
                    continue
                cnt = np.bincount(lab[np.asarray(us)], minlength=k)
                out[fid] = (cnt / len(us), len(us))
            return out

        ip, cp = best_facts(item_staters), best_facts(conc_staters)

        chosen_ch, chosen_id, chosen_pur = np.full(n, -1, np.int64), np.full(n, -1, np.int64), np.zeros(n)
        for u in range(n):
            c = int(lab[u]); best = (-1.0, None, None)
            for i in user_items[u]:
                e = ip.get(i)
                if e is not None and e[0][c] > best[0]:
                    best = (float(e[0][c]), 0, i)
            for cc in user_concs[u]:
                e = cp.get(int(cc))
                if e is not None and e[0][c] > best[0]:
                    best = (float(e[0][c]), 1, int(cc))
            if best[1] is not None:
                chosen_pur[u], chosen_ch[u], chosen_id[u] = best

        has = chosen_ch >= 0
        base = sizes[lab] / n                       # purity you would get by guessing the prior
        lift = chosen_pur[has] - base[has]
        log(f"[k={k}] users with an ideal fact: {int(has.sum())}/{n} ({has.mean():.1%})")
        log(f"[k={k}] mean identification purity {chosen_pur[has].mean():.4f} vs prior "
            f"{base[has].mean():.4f}  (lift {lift.mean():+.4f})")
        log(f"[k={k}] distinct ideal facts used: items "
            f"{len(set(chosen_id[has & (chosen_ch == 0)].tolist()))}, concepts "
            f"{len(set(chosen_id[has & (chosen_ch == 1)].tolist()))}; "
            f"concept share {float((chosen_ch[has] == 1).mean()):.1%}")

        results[str(k)] = {
            "cluster_sizes": sizes.tolist(),
            "users_with_fact": int(has.sum()),
            "mean_purity": float(chosen_pur[has].mean()),
            "mean_prior": float(base[has].mean()),
            "mean_lift": float(lift.mean()),
            "concept_share": float((chosen_ch[has] == 1).mean()),
            "distinct_items": len(set(chosen_id[has & (chosen_ch == 0)].tolist())),
            "distinct_concepts": len(set(chosen_id[has & (chosen_ch == 1)].tolist())),
        }
        np.savez_compressed(os.path.join(OUT_DIR, f"ideal_k{k}.npz"),
                            labels=lab, channel=chosen_ch, fact=chosen_id, purity=chosen_pur,
                            rows=rows)

    with open(os.path.join(OUT_DIR, "m1_ideal.json"), "w", encoding="utf-8") as fh:
        json.dump({"analysis": "m1_stage_a2_ideal_openers", "split": "validation",
                   "test_touched": False, "min_staters": MIN_STATERS,
                   "arm_type": "ORACLE ceiling -- the fact is chosen knowing the user's cluster",
                   "canonical_snap_full": f0, "by_k": results}, fh, indent=1)
    log(f"[done] wrote {OUT_DIR}\\m1_ideal.json")


if __name__ == "__main__":
    main()
