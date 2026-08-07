r"""convbandit_greedy_rescore.py -- score the FIXED static orders inside the convbandit contract.

WHY. `docs/results/CONVBANDIT_PORT_ON_OUR_RULER.md` differences the conversational-bandit family
against `entropy0`, `greedy items-only` and `greedy concepts-only`. Those three rows come from other
runs (`interview_table_ladder_ep31.json`, `MASK_CONTRACT_DIAGNOSTIC.md` /
`experiments/battery/greedy_static_i26c.json`), so their per-user vectors are NOT paired with
`adaptive_table_convbandit_uniform_v2_per_user.npz`. A mean anchor does not license a paired
difference. Precedent: `gtport_greedy_rescore.py` (task #16) re-scored the greedy order in-harness;
this does the same for the convbandit contract, and adds the CONCEPT channel.

WHAT IS FIXED AND WHAT IS RE-RUN. The three ORDERS are read verbatim off disk; nothing is
re-selected. Everything else -- pool, floor rule, consumer, concept answer model, metric -- is this
harness, i.e. exactly what produced the convbandit cells.

CONTROL. `entropy0` top-16 is re-scored the same way and must reproduce the banked
`ours|entropy0-static` per-user vectors from `act1_per_user_vectors.npz` EXACTLY (max abs per-user
diff 0). If it does not, these vectors are not in that contract and must not be differenced.

  python src/baselines/convbandit_greedy_rescore.py
"""
import os
import sys
import json
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))

import interview_strategies as ST                                          # noqa: E402
from arm_n import load_arm_n                                              # noqa: E402
from run_interview_table import (build_recommenders, score_arm, mostpop_row, OUT,  # noqa: E402
                                 answers_to_csr)
from adaptive_harness import run_interview, common_asked, Selector        # noqa: E402
import conversational_bandit_selectors as CB                              # noqa: E402

# DIAGNOSTIC MODE (CONVBANDIT_RESCORE_NO_NEG_CAP=1). `build_concept_stack` builds concept answers
# with signed_values(apply_neg_cap=True) -- the POPULATION-wide negative-mass cap over all 1031
# concepts. The certified-era greedy env (strategy_channel_suite.build_greedy_env) builds them with
# apply_neg_cap=False and caps over the REVEALED list instead, which is the convention cv3 was
# trained on. That is a real difference in the ANSWER, not in the question, so it is the first
# suspect when a re-scored concept row does not reproduce its published mean. This flag re-scores
# the concept order under the certified-era convention, for diagnosis ONLY -- the CI table uses the
# in-contract (capped) vectors, because those are the ones conlinucb_bs_static is paired with.
NO_NEG_CAP = os.environ.get("CONVBANDIT_RESCORE_NO_NEG_CAP") == "1"
LABEL = "convbandit_rescore" + ("_nocap" if NO_NEG_CAP else "")   # unique label (HARD RULE 11e)
LOGP = os.path.join(OUT, f"{LABEL}.log")
NPZ = os.path.join(OUT, f"{LABEL}_per_user.npz")
SNAP = os.path.join(_ROOT, ".cache", "instrument", "t2i26c_ep31.pt")
CONC_CKPT = os.path.join(_ROOT, "models", "i26c_ep31", "concept_cv3_ep08.pt")
GREEDY = os.path.join(_ROOT, "experiments", "battery", "greedy_static_i26c.json")
REF = os.path.join(OUT, "act1_per_user_vectors.npz")
BUDGETS = (1, 2, 4, 8, 16)


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(LOGP, "a") as f:
        f.write(line + "\n")


class StaticActionOrderSelector(Selector):
    """Hands the SAME fixed action id to every user at every turn.

    The order is in the harness's FULL action space: id < n_items is an item, id >= n_items is
    key-term id - n_items. That is the only thing needed to drive a fixed concept sequence through
    exactly the path the conversational arms took."""

    def __init__(self, order, name):
        self.order = [int(x) for x in order]
        self.name = name

    def select(self, turn, users, asked, answered):
        return np.full(len(users), self.order[turn], dtype=np.int64)


def main():
    gj = json.load(open(GREEDY))
    assert gj["tower"] == "t2i26c_ep31.pt" and gj["concept_ckpt"] == "concept_cv3_ep08.pt", gj["tower"]
    seqs = {a: [(int(c), int(i)) for c, i in gj["arms"][a]["sequence"]]
            for a in ("items-only", "concepts-only")}
    for a, s in seqs.items():
        assert len(s) == 16, (a, len(s))
    assert all(c == 0 for c, _ in seqs["items-only"])
    assert all(c == 1 for c, _ in seqs["concepts-only"])

    D = load_arm_n(log=logln)
    ni, n = D["n_items"], D["te_tr"].shape[0]
    glob, bank = ST.build_orders(D["g_train"], D["train"], ni, log=logln)
    lvl = ST.level_lookup(D["g_te_tr"])
    built = build_recommenders(["ours"], D, arch="i26", snapshot=SNAP, log=logln)
    pop_row, prior = mostpop_row(D, log=logln)
    logln(f"[rescore] {n} users, {ni} items; mostpop prior {prior}")

    # The concept half of the action space: the SAME builder the convbandit run used. Y=None skips
    # the key-term FEATURES (Xtilde) -- a fixed order selects nothing, so only the ANSWER table
    # (V, answerable) is load-bearing here, and it is built identically.
    stack = CB.build_concept_stack(D, SNAP, keyterm_repr="uniform_member_mean", Y=None, log=logln)
    C = int(stack["n_concepts"])
    if NO_NEG_CAP:
        # rebuild V/answerable with the certified-era convention, everything else untouched
        from signed_answers import load_prereg, signed_values, BAND_REFUSE
        G = D["g_te_tr"].tocsr()
        items_l, stars_l = [], []
        for u in range(G.shape[0]):
            s, e = G.indptr[u], G.indptr[u + 1]
            items_l.append(G.indices[s:e].astype(np.int64))
            stars_l.append(G.data[s:e].astype(np.float32))
        Mm = stack["Mm"]
        pexp = (stack["cnt"] @ np.asarray(Mm.todense())) / max(stack["cnt"].sum(), 1e-9)
        prereg, item_mean = load_prereg()
        V2, F2, B2, A2 = signed_values(items_l, stars_l, item_mean, Mm, pexp, ni, prereg,
                                       apply_neg_cap=False)
        stack["V"] = V2.astype(np.float64)
        stack["answerable"] = A2 & (B2 != BAND_REFUSE)
        logln("[rescore] DIAGNOSTIC: concept answers rebuilt with apply_neg_cap=False "
              "(certified-era convention) -- NOT the convbandit contract")
    logln(f"[rescore] concept bank C={C}")
    conc = {"n_concepts": C, "answer": (lambda u, c: CB.concept_answer(stack, u, c))}

    ORDERS = {
        "entropy0-top16": [int(i) for i in glob["entropy0"][:16]],           # CONTROL
        "greedy-items-only": [i for _, i in seqs["items-only"]],
        "greedy-concepts-only": [ni + c for _, c in seqs["concepts-only"]],
    }
    # The published concept opponent's REALISED order, read off its own banked interview sidecar.
    # ConLinUCB-BS-static is key-term-only and its barycentric-spanner order is the same for every
    # user (asserted), so it is a FIXED sequence and can be re-scored like any other. Under the
    # in-contract answer convention this must reproduce the banked convbandit vectors exactly; under
    # the certified-era convention it gives the like-for-like row the concept-channel gap needs.
    bs = json.load(open(os.path.join(OUT, "adaptive_table_convbandit_uniform_v2_interviews",
                                     "conlinucb_bs_static_k16.json")))
    assert bs["turns_done"] == 16 and all(x == bs["acts"][0] for x in bs["acts"]), \
        "conlinucb_bs_static is not a fixed order in this run -- it cannot be re-scored this way"
    ORDERS["conlinucb_bs_static"] = [int(x) for x in bs["acts"][0]]
    for nm, o in ORDERS.items():
        logln(f"[rescore] {nm}: {o}")

    flat = {}
    if os.path.exists(NPZ):                                                # resume (HARD RULE 11a)
        z = np.load(NPZ, allow_pickle=False)
        flat = {k: z[k] for k in z.files}
        logln(f"[rescore] resumed {len(flat)} vectors from {os.path.basename(NPZ)}")

    mode, obj = built["ours"]
    cfnet = None
    means = {}
    for name, order in ORDERS.items():
        for k in BUDGETS:
            if f"rescore|{name}|k{k}|full" in flat:
                logln(f"[rescore] {name} k={k} already banked -- skipped")
                continue
            sel = StaticActionOrderSelector(order, name)
            asked, answered, info = run_interview(sel, k, n, ni, lvl,
                                                  ctx={"D": D, "glob": glob, "bank": bank},
                                                  log=lambda *_: None, conc=conc)
            acts = info["_acts"]
            assert acts[0] == list(order[:k]), f"ask-list drift: {acts[0]} vs {order[:k]}"
            assert common_asked(acts) is not None, "static order must give every user the same acts"
            cs = info["_conc_state"]
            n_ct = sum(len(t) for t in cs["tokens"])
            s_name, s_obj = "ours", obj
            if n_ct:
                # The CERTIFIED concept consumer, built exactly as adaptive_harness builds it.
                from concept_fold import ConceptFoldNet
                import torch
                if cfnet is None:
                    blob = torch.load(CONC_CKPT, map_location="cpu")
                    sd = blob["net"]
                    _nc, _d = tuple(sd["emb.weight"].shape)
                    _h = int(sd["gate.0.weight"].shape[0])
                    assert _nc == C, f"CONCEPT VOCABULARY DRIFT: ckpt {_nc} vs bank {C}"
                    assert _d == obj[0].d_out, "concept net latent width != tower latent width"
                    cfnet = ConceptFoldNet(_nc, d=_d, h=_h, use_refusal=blob.get("use_refusal", False))
                    cfnet.load_state_dict(sd)
                    cfnet.eval()
                    logln(f"[rescore] concept consumer = {os.path.basename(CONC_CKPT)} "
                          f"(ep{blob.get('epoch','?')}, C={_nc}, h={_h})")
                e_, W_, b_ = obj
                Lc = answers_to_csr(answered, n, ni, mode, asked=asked)
                toks = cs["tokens"]
                if NO_NEG_CAP:
                    from signed_answers import cap_negatives
                    toks = [[(c, v, r) for (c, _v, r), v in
                             zip(t, cap_negatives([x[1] for x in t]))] for t in toks]
                s_obj = CB.make_concept_aware_predict(e_, W_, b_, Lc, toks, cfnet)
                s_name = "ours_conc"
            t0 = time.time()
            ga = asked[0]
            cell = score_arm(s_name, mode, s_obj, asked, answered, D, pop_row, no_floor=True,
                             global_asked=ga, per_user=True)
            flat[f"rescore|{name}|k{k}|full"] = np.asarray(cell["full_per_user"], np.float64)
            flat[f"rescore|{name}|k{k}|tail"] = np.asarray(cell["tail_per_user"], np.float64)
            np.savez_compressed(NPZ, **flat)                               # every cell, immediately
            means[f"{name}|k{k}"] = {"full": cell["full"], "tail": cell["tail"],
                                     "answered_items": cell["answered"], "concept_tokens": n_ct}
            logln(f"[rescore] {name:22s} k={k:2d} items_answered={cell['answered']:.2f} "
                  f"concept_tokens={n_ct} full={cell['full']:.4f} tail={cell['tail']:.4f} "
                  f"({(time.time()-t0)/60:.2f}m)")
            del asked, answered, info, cs

    json.dump(means, open(os.path.join(OUT, f"{LABEL}_means.json"), "w"), indent=2)

    # ------------------------------------------------------------------ the admission control
    ref = np.load(REF, allow_pickle=False)
    worst = 0.0
    for k in BUDGETS:
        for ax in ("full", "tail"):
            d = np.abs(flat[f"rescore|entropy0-top16|k{k}|{ax}"] - ref[f"ours|entropy0-static|{k}|{ax}"])
            worst = max(worst, float(d.max()))
            logln(f"[control] entropy0 k={k} {ax}: max per-user |re-score - act1| = {d.max():.3e}")
    logln(f"[control] {'PASS' if worst == 0.0 else 'CHECK'}: worst per-user diff {worst:.3e}")


if __name__ == "__main__":
    main()
