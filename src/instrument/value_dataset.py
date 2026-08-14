r"""THE UNIFIED (state, question, gain) DATASET for Paper B's value head.

    AUTHOR SAID (2026-08-14, verbatim):
        "can you collect the dataset: all states you can scrape for all greedy variants, trees
         static, whatnot, plus random sampling of everything else and redo the training?
         or have i missed anything?"

That sentence is the HARD RULE 1 sign-off for (i) the scrape and (ii) the random-candidate sampling
whose design is stated in `--mode sample` below and in `docs/results/VALUE_DATASET.md`. Nothing else
is authorised by it.

WHAT THIS FILE PRODUCES
-----------------------
One tidy table, `experiments/battery/value_dataset.jsonl`, one row per MEASUREMENT:

    state_id, asked_set, cohort_sha1, n_rows, contract, pool, channel, question,
    absolute_value, base, gain, selection_tag, source_file, level

plus `experiments/battery/value_dataset_manifest.json` -- the composition audit (measurements and
states per contract, per selection tag, per depth) and the full admit/refuse ledger.

★ THE SELECTION TAG IS THE POINT (hazard 7). Measurements are NOT missing at random, and the
missingness is not truncation either -- it is BIMODAL. At a given state the CELF heap holds a large
mass of WEAK candidates (one measured node: ~984 entries all below +0.0021) while the WINNERS live in
`seq`/`gains` at +0.0044..+0.0135, and the MIDDLE -- exactly where a ranking decision is made -- is
missing. So every row carries how it was selected:

    pick             a winner: the candidate CELF popped at this state
    heap_survivor    measured at this state and never popped -- the weak mass, biased LOW
    exhaustive_sweep every candidate at this state was measured (phase 3 / the level-0 seed pass)
    random_sample    a uniform draw from the pool, therefore unbiased by construction

`state_complete` is carried alongside, so any reported result can be reproduced restricted to the
unbiased sources only.

ADMISSION -- ON THE RECORDED CONTRACT, NEVER ON THE FILENAME
------------------------------------------------------------
1. **Contract.** Policy-B validation is the target ruler:
   `nnz=1356475 per_user=135.6475 fp=c282962b49a2d67ef47ff07544519f20`. A policy-A gain and a
   policy-B gain are numbers on different rulers and are NOT comparable -- this is the contamination
   that forced the battery rebuild (memory `banked-global-greedy-is-a-policy-A-artifact`). A source
   whose artifact cannot STATE its contract is refused; nothing is ever "converted".
2. **Cohort.** `sig["split"] == "validation"` and, where a contract tag exists, it must name
   validation. `ndcgtree16test_heaps` is admitted because its recorded contract and `rows_sha1` are
   validation -- "test" names the DRIVER JOB, not the cohort -- and `ndcgtree_SANITY/_PREFLIGHT` are
   refused because the SANITY tree is not the citable tree.
3. ★ **Objective.** `splitcrit_tree_{pairwise,variance,logistic,random_ml,random}` scored candidates
   by a PAIRWISE / VARIANCE / LOGISTIC surrogate. Their scalars are surrogate scores, not measured
   NDCG@10 -- admitting them would train the value head on the very quantities the chapter proves
   lose. Only `objective in MEASURED_OBJECTIVES` is admitted; anything whose objective cannot be
   established from its own artifact is refused and logged.

BASE / PREFIX. A gain is relative to a base at a prefix, so the ABSOLUTE measured value and the base
are stored separately together with the exact asked-set: a gain can always be re-derived, and
`absolute_value - base == gain` is asserted on every row.

DE-DUPLICATION. A STATE is (cohort, asked-question-set) -- NOT (sidecar, level). `ndcgtree16_heaps`
re-banks the same leaf states with the tail extended, so the naive key double-counts. Where two
independently launched runs measured the same state they are MERGED and every shared question is
ASSERTED EQUAL -- a free cross-run byte-identity check.

FIREWALL: validation only. Every artifact is stamped `test_touched=false`.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import json
import glob
import argparse
from collections import defaultdict

OUTDIR = os.path.join(_ROOT, "experiments", "battery")
JSONL = os.path.join(OUTDIR, "value_dataset.jsonl")
MANIFEST = os.path.join(OUTDIR, "value_dataset_manifest.json")

# The target ruler, quoted from `greedy_static_policyB_items.json["pool_contract"]`.
CONTRACT_FP = "c282962b49a2d67ef47ff07544519f20"
CONTRACT_NNZ = 1356475
CONTRACT_PER_USER = 135.6475
# The validation cohort's root fingerprint (every sub-cohort is a subset of these users).
VAL_ROOT_SHA1 = "6a70aacba08effcbd1a46a9a352a8e23c2989651"
VAL_ROOT_M0_FULL = 0.15686538118354684

# Objectives that ARE measured NDCG@10. Anything else is a surrogate and is refused.
MEASURED_OBJECTIVES = {"full", "full_and_tail_ndcg_and_dcg"}
SANITY_MARKERS = ("SANITY", "PREFLIGHT")


def log(m):
    print(m, flush=True)


class Ledger:
    """Every source the sweep touched, with the reason it was admitted or refused. A refusal that is
    not written down is a refusal nobody can audit."""

    def __init__(self):
        self.rows = []

    def admit(self, src, why, n=0):
        self.rows.append({"source": src, "verdict": "ADMIT", "why": why, "n_measurements": n})

    def refuse(self, src, why):
        self.rows.append({"source": src, "verdict": "REFUSE", "why": why, "n_measurements": 0})
        log(f"[refuse] {src}: {why}")


# ============================================================================ contract / firewall
def contract_of(sig, src, led):
    """The recorded contract of a sidecar, or None if the artifact cannot state one.

    Two signature versions exist: v1 carries `contract_tag` and predates the mask fingerprint; v2
    carries `mask_pool`. v1 is admitted only because the tree's own env has been policy B since the
    2026-08-05 ruling AND `lift_head`'s G-L1 re-derived every one of those 22 bases to exact `==`
    against a freshly built policy-B environment. That is a MEASURED provenance, not a filename one,
    and it is recorded here as `policyB(v1, base-verified)` so the two families stay distinguishable
    in the composition table."""
    if sig.get("split") != "validation":
        led.refuse(src, f"FIREWALL: split={sig.get('split')!r}, not validation")
        return None
    mp = sig.get("mask_pool")
    if mp is not None:
        if mp.get("policy") != "B" or mp.get("fp") != CONTRACT_FP:
            led.refuse(src, f"CONTRACT: mask_pool policy={mp.get('policy')!r} fp={mp.get('fp')!r} "
                            f"is not the policy-B validation ruler")
            return None
        assert mp["nnz"] == CONTRACT_NNZ and abs(mp["per_user"] - CONTRACT_PER_USER) < 1e-9
        return "policyB(v2, fp-stated)"
    ct = sig.get("contract_tag")
    if ct is not None and ":validation" in str(ct):
        return "policyB(v1, base-verified)"
    led.refuse(src, "CONTRACT: the artifact states neither `mask_pool` nor a validation "
                    "`contract_tag` -- an unstated contract is not a matching one")
    return None


def objective_of(sig, src, led):
    o = sig.get("objective")
    if o not in MEASURED_OBJECTIVES:
        led.refuse(src, f"OBJECTIVE: {o!r} is not measured NDCG@10 -- a surrogate scalar is not "
                        f"admissible material for the value head")
        return None
    return o


def chan(q):
    return "item" if int(q[0]) == 0 else "concept"


def state_key(rows_sha1, n_rows, asked):
    return f"{rows_sha1}:{n_rows}:" + "|".join(f"{int(a)}_{int(b)}" for a, b in sorted(asked))


# ============================================================================ the scrape
class Dataset:
    """Measurements keyed by (state, question), merged across sources with byte-identity asserted on
    every overlap."""

    def __init__(self):
        self.states = {}          # state_id -> state record
        self.meas = {}            # (state_id, q) -> measurement

    def add_state(self, sid, asked, base, rows_sha1, n_rows, pool, contract, src, nid=None):
        st = self.states.get(sid)
        if st is None:
            self.states[sid] = {"state_id": sid, "asked_set": [list(map(int, q)) for q in asked],
                                "depth": len(asked), "base": float(base),
                                "cohort_sha1": rows_sha1, "n_rows": int(n_rows),
                                "pool": pool, "contract": contract, "sources": [src], "nid": nid}
            return
        # ★ the free cross-run check: two independently launched runs that measured the same state
        assert abs(st["base"] - float(base)) < 1e-12, (
            f"CROSS-RUN BASE MISMATCH at {sid}: {st['base']!r} ({st['sources'][0]}) vs "
            f"{float(base)!r} ({src})")
        assert st["contract"].split("(")[0] == contract.split("(")[0], (
            f"CONTRACT MIXING at {sid}: {st['contract']} vs {contract}")
        if src not in st["sources"]:
            st["sources"].append(src)
        if st["nid"] is None:
            st["nid"] = nid

    def add(self, sid, q, gain, base, tag, src, level, complete):
        q = (int(q[0]), int(q[1]))
        k = (sid, q)
        rec = self.meas.get(k)
        val = float(base) + float(gain)
        if rec is None:
            self.meas[k] = {"state_id": sid, "question": list(q), "channel": chan(q),
                            "gain": float(gain), "base": float(base), "absolute_value": val,
                            "selection_tag": tag, "source_file": src, "level": int(level),
                            "state_complete": bool(complete)}
            return 1
        assert abs(rec["gain"] - float(gain)) < 1e-12, (
            f"CROSS-RUN GAIN MISMATCH at {sid} q={q}: {rec['gain']!r} ({rec['source_file']}) vs "
            f"{float(gain)!r} ({src})")
        # a pick is the informative tag; keep it if either source says so
        if tag == "pick" or rec["selection_tag"] == "exhaustive_sweep":
            rec["selection_tag"] = "pick" if tag == "pick" else rec["selection_tag"]
        return 0


def scrape_heaps(ds, led):
    """Every CELF heap sidecar on disk. A heap entry `(a, q, u)` was measured AT level `u`, so
    grouping by `u` gives genuine measurements at that state -- never a stale upper bound."""
    n_files = 0
    for d in sorted(glob.glob(os.path.join(OUTDIR, "*_heaps"))):
        base_d = os.path.basename(d)
        if any(m in base_d for m in SANITY_MARKERS):
            led.refuse(base_d, "not the citable tree (SANITY/PREFLIGHT); `policy_traj.load_tree` "
                               "refuses it, so its states are not citable material")
            continue
        files = [f for f in sorted(os.listdir(d)) if f.endswith(".json")]
        if not files:
            led.refuse(base_d, "empty directory")
            continue
        n_dir = 0
        for fn in files:
            src = f"{base_d}/{fn}"
            b = json.load(open(os.path.join(d, fn)))
            if "heap" not in b or "sig" not in b:
                led.refuse(src, "not a heap sidecar (no `heap`/`sig`)")
                continue
            sig = b["sig"]
            contract = contract_of(sig, src, led)
            if contract is None:
                continue
            if objective_of(sig, src, led) is None:
                continue
            n_files += 1
            prefix = [(int(x[0]), int(x[1])) for x in sig.get("prefix", [])]
            seq = [(int(x[0]), int(x[1])) for x in b["seq"]]
            rows_sha1, n_rows = sig.get("rows_sha1"), int(sig["n_rows"])
            pool = {"n_pool": int(sig["n_pool"]), "pool_sha1": sig.get("pool_sha1")}
            lev = defaultdict(dict)
            for a, q, u in b["heap"]:
                lev[int(u)][(int(q[0]), int(q[1]))] = -float(a)
            cur = float(b["base"])
            for k in range(len(seq)):
                g = dict(lev.get(k, {}))
                g[seq[k]] = float(b["gains"][k])          # ★ the popped pick, added back
                asked = prefix + seq[:k]
                n_avail = pool["n_pool"] - len(asked)
                complete = len(g) >= n_avail
                sid = state_key(rows_sha1, n_rows, asked)
                nid = (fn[5:-5] + f"@{k}") if fn.startswith("node_") else None
                ds.add_state(sid, asked, cur, rows_sha1, n_rows, pool, contract, f"{src}@{k}",
                             nid=nid)
                for q, v in g.items():
                    tag = ("pick" if q == seq[k]
                           else "exhaustive_sweep" if complete else "heap_survivor")
                    n_dir += ds.add(sid, q, v, cur, tag, f"{src}@{k}", k, complete)
                cur += float(b["gains"][k])
        led.admit(base_d, f"{len(files)} sidecars, contract and objective stated and matching", n_dir)
    return n_files


def scrape_greedy_json(ds, led):
    """The greedy / static / group artifacts that carry `sequence` + `gains` but no heap sidecar.
    These contribute PICKS only -- one measurement per state -- and are admitted on the same two
    tests as everything else."""
    cands = [
        ("greedy_static_policyB_combined.json", "sequence", "gains"),
        ("greedy_static_policyB_items.json", "sequence", "gains"),
        ("greedy_static_policyB_concepts.json", "sequence", "gains"),
        ("greedy_static_i26c.json", None, None),
        ("greedy_polish.json", None, None),
        ("groupgreedy_d1.json", None, None),
        ("groupgreedy_genre15.json", None, None),
        ("groupgreedy_genre15_VAL_policyB.json", None, None),
    ] + [(os.path.basename(p), None, None)
         for p in sorted(glob.glob(os.path.join(OUTDIR, "splitcrit_tree_*.json")))]
    n_add = 0
    for fn, seqk, gaink in cands:
        p = os.path.join(OUTDIR, fn)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        if fn.startswith("splitcrit_tree_"):
            led.refuse(fn, f"OBJECTIVE: split-criterion ablation arm {d.get('criterion')!r} -- its "
                           f"`splitter_score` is a SURROGATE scalar, not measured NDCG@10")
            continue
        if seqk is None or seqk not in d:
            why = ("no per-question measured gains under a stated policy-B contract"
                   if d.get("pool_contract") else
                   "the artifact states no `pool_contract`/`mask_policy` -- its ruler cannot be "
                   "established, and a policy-A gain is not comparable to a policy-B one")
            led.refuse(fn, why)
            continue
        if d.get("split") != "validation" or d.get("mask_policy") != "B":
            led.refuse(fn, f"FIREWALL/CONTRACT: split={d.get('split')!r} "
                           f"mask_policy={d.get('mask_policy')!r}")
            continue
        pc = d.get("pool_contract")
        if pc is not None and pc.get("fp") != CONTRACT_FP:
            led.refuse(fn, f"CONTRACT: pool_contract fp {pc.get('fp')!r} != the target ruler")
            continue
        if d.get("objective") not in MEASURED_OBJECTIVES:
            led.refuse(fn, f"OBJECTIVE: {d.get('objective')!r} is not measured NDCG@10")
            continue
        seq = [(int(x[0]), int(x[1])) for x in d[seqk]]
        gains = [float(x) for x in d[gaink]]
        # cohort: the full validation cohort. Proven, not assumed -- the m=0 intercept must be the
        # policy-B one, which is what fixes both the ruler and the cohort.
        cur = float(d.get("intercept_policyB_full", VAL_ROOT_M0_FULL))
        assert abs(cur - VAL_ROOT_M0_FULL) < 1e-12, f"{fn}: m=0 is not the policy-B intercept"
        assert int(d["n_cohort"]) == 10000
        pool = {"n_pool": int(d["n_pool"]), "pool_sha1": None}
        n = 0
        for k in range(len(seq)):
            sid = state_key(VAL_ROOT_SHA1, 10000, seq[:k])
            ds.add_state(sid, seq[:k], cur, VAL_ROOT_SHA1, 10000, pool,
                         "policyB(v2, fp-stated)", f"{fn}@{k}")
            n += ds.add(sid, seq[k], gains[k], cur, "pick", f"{fn}@{k}", k, False)
            cur += gains[k]
        led.admit(fn, f"{len(seq)} picks, policy-B validation, measured NDCG@10", n)
        n_add += n
    return n_add


def scrape_phase3(ds, led):
    """Phase 3's exhaustive sweeps -- the only source that is unbiased BY CONSTRUCTION at every
    candidate. Ingested from whatever is complete on disk; a state still in flight is skipped."""
    n_add = 0
    for pj in sorted(glob.glob(os.path.join(OUTDIR, "policy_labels_*_partial.json"))) + \
            sorted(glob.glob(os.path.join(OUTDIR, "policy_labels_*.json"))):
        fn = os.path.basename(pj)
        if fn.endswith("_partial.json.tmp"):
            continue
        try:
            d = json.load(open(pj))
        except Exception as e:
            led.refuse(fn, f"unreadable ({e})")
            continue
        sig = d.get("sig", {})
        contract = contract_of(sig, fn, led)
        if contract is None:
            continue
        if objective_of(sig, fn, led) is None:
            continue
        pool = {"n_pool": int(sig.get("n_pool", 1000)), "pool_sha1": sig.get("pool_sha1")}
        n = ndone = 0
        for sid_, st in d.get("states", {}).items():
            if not st.get("done"):
                continue
            ndone += 1
            asked = [(int(x[0]), int(x[1])) for x in st["prefix"]]
            base = float(st["base"]) if "base" in st else None
            if base is None:
                led.refuse(f"{fn}:{sid_}", "no `base` recorded -- a gain without its base is not "
                                           "re-derivable")
                continue
            sid = state_key(st.get("rows_sha1", "phase3:" + sid_), int(st["n_users"]), asked)
            ds.add_state(sid, asked, base, st.get("rows_sha1", "phase3:" + sid_),
                         int(st["n_users"]), pool, contract, f"{fn}:{sid_}")
            for qs, v in st["vals"].items():
                a, bq = qs.split("_")
                g = v[0] if isinstance(v, list) else v
                n += ds.add(sid, (int(a), int(bq)), float(g), base, "exhaustive_sweep",
                            f"{fn}:{sid_}", len(asked), True)
        if ndone:
            led.admit(fn, f"{ndone} COMPLETE exhaustive states (unbiased at every candidate)", n)
        else:
            led.refuse(fn, "no completed state yet (run still in flight)")
        n_add += n
    return n_add


# ============================================================================ composition + write
def compose(ds):
    per_contract = defaultdict(lambda: {"measurements": 0, "states": set()})
    per_tag = defaultdict(lambda: {"measurements": 0, "states": set()})
    per_depth = defaultdict(lambda: {"measurements": 0, "states": set()})
    per_pool = defaultdict(lambda: {"measurements": 0, "states": set()})
    per_channel = defaultdict(int)
    for m in ds.meas.values():
        st = ds.states[m["state_id"]]
        for tbl, key in ((per_contract, st["contract"]), (per_tag, m["selection_tag"]),
                         (per_depth, st["depth"]), (per_pool, st["pool"]["n_pool"])):
            tbl[key]["measurements"] += 1
            tbl[key]["states"].add(m["state_id"])
        per_channel[m["channel"]] += 1

    def fin(t):
        return {str(k): {"measurements": v["measurements"], "states": len(v["states"])}
                for k, v in sorted(t.items(), key=lambda kv: str(kv[0]))}
    sz = sorted(len([1 for m in ds.meas.values() if m["state_id"] == s]) for s in ds.states)
    cand = defaultdict(int)
    for m in ds.meas.values():
        cand[m["state_id"]] += 1
    sz = sorted(cand.values())
    q = lambda p: sz[min(len(sz) - 1, int(p * len(sz)))]
    return {"per_contract": fin(per_contract), "per_selection_tag": fin(per_tag),
            "per_depth": fin(per_depth), "per_pool": fin(per_pool),
            "per_channel": dict(per_channel),
            "candidates_per_state": {"min": sz[0], "p25": q(.25), "median": q(.5),
                                     "p75": q(.75), "max": sz[-1]},
            "n_states": len(ds.states), "n_measurements": len(ds.meas)}


def write(ds, led, comp, extra=None):
    with open(JSONL, "w", encoding="utf-8") as f:
        for (sid, q), m in sorted(ds.meas.items()):
            st = ds.states[sid]
            row = dict(m)
            row.update({"asked_set": st["asked_set"], "depth": st["depth"],
                        "cohort_sha1": st["cohort_sha1"], "n_rows": st["n_rows"],
                        "contract": st["contract"], "pool": st["pool"]})
            assert abs(row["absolute_value"] - row["base"] - row["gain"]) < 1e-15
            f.write(json.dumps(row) + "\n")
    man = {"analysis": "unified_value_head_dataset", "test_touched": False, "split": "validation",
           "authorisation": "AUTHOR 2026-08-14 (verbatim): 'can you collect the dataset: all states "
                            "you can scrape for all greedy variants, trees static, whatnot, plus "
                            "random sampling of everything else and redo the training?' -- HARD "
                            "RULE 1 sign-off for the scrape and for the random-candidate sampling "
                            "design recorded in docs/results/VALUE_DATASET.md",
           "contract": {"policy": "B", "fp": CONTRACT_FP, "nnz": CONTRACT_NNZ,
                        "per_user": CONTRACT_PER_USER},
           "jsonl": os.path.basename(JSONL), "composition": comp,
           "selection_tags": {
               "pick": "a winner -- the candidate CELF popped at this state",
               "heap_survivor": "measured at this state and never popped: the weak mass, biased LOW",
               "exhaustive_sweep": "every candidate at this state was measured; unbiased",
               "random_sample": "a uniform draw from the pool; unbiased by construction"},
           "ledger": led.rows}
    if extra:
        man.update(extra)
    json.dump(man, open(MANIFEST, "w"), indent=1)
    log(f"[write] {len(ds.meas):,} measurements over {len(ds.states):,} states -> {JSONL}")
    log(f"[write] manifest -> {MANIFEST}")


# ================================================================ the parameter-free baseline arms
def _spearman(a, b):
    import numpy as np
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / max((((ra ** 2).sum() * (rb ** 2).sum()) ** 0.5), 1e-12))


def load_jsonl():
    """The scraped table, regrouped as {state_id: {question: gain}} plus the state records."""
    states, gains = {}, defaultdict(dict)
    tags = defaultdict(dict)
    for line in open(JSONL, encoding="utf-8"):
        r = json.loads(line)
        sid = r["state_id"]
        if sid not in states:
            states[sid] = {k: r[k] for k in ("state_id", "asked_set", "depth", "base",
                                             "cohort_sha1", "n_rows", "contract", "pool")}
        gains[sid][(r["question"][0], r["question"][1])] = r["gain"]
        tags[sid][(r["question"][0], r["question"][1])] = r["selection_tag"]
    for sid in states:
        states[sid]["n_candidates"] = len(gains[sid])
        states[sid]["complete"] = any(t == "exhaustive_sweep" for t in tags[sid].values())
    return states, gains, tags


def meanvec_arm(eval_sids, train_sids, gains, label):
    r"""★ HARD RULE 2. Predict a held-out state's whole gain vector with the PLAIN MEAN of the other
    states' gain vectors -- no features, no parameters, and no state information at all: it does not
    know which state it is predicting for. It is the bar every learned head must clear, and last time
    it beat the learned head by a factor of seven (+0.4163 vs +0.0623)."""
    import numpy as np
    rows = []
    for s in eval_sids:
        others = [t for t in train_sids if t != s]
        pred, y = [], []
        for q, g in gains[s].items():
            vals = [gains[t][q] for t in others if q in gains[t]]
            if vals:
                pred.append(float(np.mean(vals))); y.append(g)
        if len(y) < 3:
            continue
        y = np.asarray(y); pred = np.asarray(pred)
        rk = int((y > y[int(np.argmax(pred))]).sum()) + 1
        rows.append({"held_out_state": s, "n_scored": len(y), "n_train_states": len(others),
                     "spearman": _spearman(pred, y), "rank_of_argmax": rk,
                     "gain_of_argmax": float(y[int(np.argmax(pred))]),
                     "true_best_gain": float(y.max())})
    m = sum(r["spearman"] for r in rows) / max(len(rows), 1)
    log(f"[MEANVEC/{label}] mean held-out Spearman {m:+.4f} over {len(rows)} evaluation states "
        f"({'PASSES' if m >= 0.40 else 'FAILS'} the +0.40 gate)")
    return {"arm": label, "mean_spearman": m, "n_eval_states": len(rows), "per_state": rows}


def run_meanvec():
    states, gains, tags = load_jsonl()
    comp = sorted([s for s in states if states[s]["complete"]],
                  key=lambda s: (states[s]["depth"], s))
    log(f"[meanvec] {len(states)} states; {len(comp)} COMPLETE (unbiased) states are the evaluation "
        f"set -- a biased vector is never evaluated on")
    biased = [s for s in states if not states[s]["complete"]]
    surv = [s for s in biased if any(t == "heap_survivor" for t in tags[s].values())]
    arms = [
        meanvec_arm(comp, comp, gains, "unbiased_only(complete states)"),
        meanvec_arm(comp, list(states), gains, "unbiased+all_tagged_biased"),
        meanvec_arm(comp, comp + surv, gains, "unbiased+heap_survivors"),
        meanvec_arm(comp, [s for s in states if s not in comp], gains, "biased_only"),
    ]
    out = {"analysis": "value_dataset_meanvec_source_ablation", "test_touched": False,
           "split": "validation", "gate": 0.40,
           "banked_assert": {"complete7_meanvec": 0.4163,
                             "note": "OUTCOME_HEAD.md §2.5, +0.4163 over the 7 complete states"},
           "arms": arms}
    p = os.path.join(OUTDIR, "value_dataset_meanvec.json")
    json.dump(out, open(p, "w"), indent=1)
    log(f"[meanvec] -> {p}")
    return out


# ================================================================ STEP 3 -- the random sampling job
r"""THE SAMPLING DESIGN, STATED IN FULL (HARD RULE 1; authorised verbatim by the author, above).

Everything the scrape yields is missing-not-at-random in the same BIMODAL way: at each state a large
mass of weak candidates (the CELF survivors) plus one excellent pick, with the MIDDLE -- exactly
where a ranking decision is made -- absent. Random draws have no middle-hole by construction.

Per prefix:
  1. draw k ~ Uniform{3,4,5,6}, then k DISTINCT questions uniformly without replacement from the
     policy-B combined pool (500 popularity items + 500 answerable concepts, the tree's own pool,
     same ids in the same order);
  2. ask that prefix to ALL 10,000 validation users and keep the PER-USER NDCG vector (no sampling
     of users anywhere);
  3. record each user's ternary ANSWER PATH through the prefix (like/dislike/no-answer per question);
  4. draw N_CAND = 50 candidates uniformly without replacement from the pool minus the prefix, and
     evaluate each over all 10,000 users, again keeping the per-user vector;
  5. for the marginal state (all 10,000) and for every answer path with at least MIN_PATH_USERS
     users, the conditional gain is mean(f_q[path]) - mean(f_0[path]) -- the same construction the
     tree uses to score a split, so it is on the same ruler.

  ★ Step 5 is why one sweep buys many states: grouping by answer path turns 51 evaluations into one
  measured gain vector per POPULATED PATH, at zero extra evaluation.

WHAT IS SAMPLED, AND WHAT IS NOT. Users: never -- all 10,000, every time. Candidates: yes, 50 of
~995 per prefix, uniformly at random; that is the authorised reduction and it is the whole point,
since a regressor wants scattered triples and only an argmax search needs the full sweep
(OUTCOME_HEAD.md RESULT 5). Prefixes: 19 by default, ~7 min each on the measured 8.24 s/eval at
n=10,000, so ~2.2 h.

THE TRUNCATION THAT MUST BE DECLARED. A path with few users gives an unstable mean, so paths below
MIN_PATH_USERS = 200 are DROPPED -- the tree's own `min_leaf`, chosen because it already governs
every banked state rather than invented for this job. The count dropped, and the users they carry,
are recorded per prefix in the artifact.

PERSISTENCE (HARD RULE 11). Each prefix writes its own contract-signed sidecar the moment it
finishes, plus the raw per-user float32 vectors, so a crash can lose at most one prefix and any
later change to MIN_PATH_USERS is free -- no re-evaluation.
"""
N_CAND = 50
MIN_PATH_USERS = 200
PREFIX_K = (3, 4, 5, 6)
SAMPLE_DIR = os.path.join(OUTDIR, "value_sample_partial")


def run_sample(args):
    import time
    import numpy as np
    sys.argv = [sys.argv[0]] + (["--threads", str(args.threads)] if args.threads else [])
    from strategy_channel_suite import (build_greedy_env, genv_eval, set_active_rows,
                                        contract_selftest)
    from mask_contract import pool_contract
    from policy_labels import atomic_write
    from ndcg_tree import build_pool, leak_assert, branch_of_per_user

    log("#" * 106)
    log("# VALUE DATASET -- STEP 3, RANDOM SAMPLING. Design stated in full in value_dataset.py.")
    log("# FIREWALL: the 10,000 canonical TEST users are NOT built, NOT loaded, NOT touched.")
    log(f"# AUTHORISED SAMPLING: {N_CAND} random candidates per prefix; users are NEVER sampled.")
    log("#" * 106)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "valuesample")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "validation")
    rows = [int(r) for r in E["rows"]]
    assert len(rows) == 10000, f"cohort is {len(rows)} users, expected 10,000"
    pc = pool_contract(E["pool"], rows, policy=E["pool_policy"])
    assert pc["fp"] == CONTRACT_FP, f"CONTRACT: env fp {pc['fp']} != {CONTRACT_FP}"
    pool, _ip, _cp = build_pool(E, 500, 500)
    pool = [(int(a), int(b)) for a, b in pool]
    assert len(pool) == 1000

    sig = {"v": 1, "split": "validation", "objective": "full", "mask_pool": pc,
           "tower": os.path.basename(args.snapshot),
           "conc_ckpt": os.path.basename(args.concept_ckpt),
           "n_pool": len(pool), "n_cand": N_CAND, "min_path_users": MIN_PATH_USERS,
           "prefix_k": list(PREFIX_K), "seed": args.seed}
    sp = os.path.join(SAMPLE_DIR, "_sig.json")
    if os.path.exists(sp):
        old = json.load(open(sp))
        assert old == sig, ("RESUME REFUSED: the partial directory was written under a different "
                            f"contract\n  on disk: {old}\n  now:     {sig}")
    else:
        atomic_write(sp, sig)

    rng = np.random.default_rng(args.seed)
    t_start = time.time()
    n_eval = 0
    for pi in range(args.n_prefix):
        outp = os.path.join(SAMPLE_DIR, f"prefix_{pi:03d}.json")
        k = int(PREFIX_K[rng.integers(len(PREFIX_K))])
        pidx = rng.choice(len(pool), size=k, replace=False)
        prefix = [pool[int(j)] for j in pidx]
        rest = [q for q in pool if q not in set(prefix)]
        cidx = rng.choice(len(rest), size=N_CAND, replace=False)
        cands = [rest[int(j)] for j in cidx]
        if os.path.exists(outp):
            log(f"[sample {pi}] already banked, skipping")
            continue
        t0 = time.time()
        set_active_rows(E, rows)
        # --- the base pass, and each user's ternary answer path through the prefix ---------------
        f0, t0v = genv_eval(E, prefix, rows)
        n_eval += 1
        paths = np.zeros((len(rows), k), np.int8)
        for j, q in enumerate(prefix):
            paths[:, j] = branch_of_per_user(E, rows, [q] * len(rows))
        pkey = ["".join(str(int(x)) for x in r) for r in paths]
        groups = defaultdict(list)
        for u, key in enumerate(pkey):
            groups[key].append(u)
        kept = {g: np.asarray(ix, np.int64) for g, ix in groups.items()
                if len(ix) >= MIN_PATH_USERS}
        dropped = {g: len(ix) for g, ix in groups.items() if len(ix) < MIN_PATH_USERS}
        log(f"[sample {pi}] k={k} prefix={prefix} | {len(groups)} populated answer paths, "
            f"{len(kept)} kept (>= {MIN_PATH_USERS} users), {len(dropped)} DROPPED carrying "
            f"{sum(dropped.values())} of 10,000 users")
        # --- the candidates, per-user vectors kept -----------------------------------------------
        F = np.zeros((N_CAND, len(rows)), np.float32)
        T = np.zeros((N_CAND, len(rows)), np.float32)
        for ci, q in enumerate(cands):
            f, t = genv_eval(E, prefix + [q], rows)
            F[ci] = f; T[ci] = t
            n_eval += 1
            if (ci + 1) % 10 == 0:
                el = time.time() - t0
                log(f"[sample {pi}] {ci + 1}/{N_CAND} ({el / 60:.1f}m, "
                    f"{el / (ci + 2):.2f} s/eval)")
        np.save(os.path.join(SAMPLE_DIR, f"prefix_{pi:03d}_peruser.npy"),
                np.concatenate([f0[None], F], 0).astype(np.float32))
        np.save(os.path.join(SAMPLE_DIR, f"prefix_{pi:03d}_peruser_tail.npy"),
                np.concatenate([t0v[None], T], 0).astype(np.float32))
        # --- the states this one sweep bought ----------------------------------------------------
        st_out = []
        for gkey, ix in [("*", np.arange(len(rows)))] + sorted(kept.items()):
            b_full = float(np.nanmean(f0[ix]))
            g = {f"{q[0]}_{q[1]}": float(np.nanmean(F[ci][ix])) - b_full
                 for ci, q in enumerate(cands)}
            st_out.append({"path": gkey, "n_users": int(len(ix)), "base": b_full,
                           "base_tail": float(np.nanmean(t0v[ix])), "gains": g})
        rec = {"sig": sig, "prefix_index": pi, "k": k,
               "prefix": [list(q) for q in prefix], "candidates": [list(q) for q in cands],
               "n_paths_populated": len(groups), "n_paths_kept": len(kept),
               "n_paths_dropped": len(dropped), "n_users_dropped": int(sum(dropped.values())),
               "states": st_out, "seconds": time.time() - t0,
               "per_user_npy": f"prefix_{pi:03d}_peruser.npy", "test_touched": False}
        atomic_write(outp, rec)
        el = time.time() - t_start
        log(f"[sample {pi}] DONE in {(time.time() - t0) / 60:.1f}m -> {len(st_out)} states, "
            f"{len(st_out) * N_CAND} measurements | {n_eval} evals, "
            f"{el / n_eval:.2f} s/eval measured (cost model says 8.24)")
    log(f"[sample] {args.n_prefix} prefixes in {(time.time() - t_start) / 3600:.2f} h")


def scrape_sample(ds, led):
    """Ingest the random-sample sidecars. Unbiased by construction: `random_sample` everywhere."""
    n_add = 0
    for p in sorted(glob.glob(os.path.join(SAMPLE_DIR, "prefix_*.json"))):
        fn = os.path.basename(p)
        d = json.load(open(p))
        sig = d["sig"]
        contract = contract_of(sig, fn, led)
        if contract is None:
            continue
        if objective_of(sig, fn, led) is None:
            continue
        pool = {"n_pool": int(sig["n_pool"]), "pool_sha1": None}
        prefix = [(int(a), int(b)) for a, b in d["prefix"]]
        n = 0
        for st in d["states"]:
            # a state here is (prefix, answer path); its cohort is that path's users
            cohort = f"valsample:p{d['prefix_index']}:{st['path']}"
            sid = state_key(cohort, st["n_users"], prefix)
            ds.add_state(sid, prefix, st["base"], cohort, st["n_users"], pool, contract, fn,
                         nid=f"sample{d['prefix_index']:03d}/{st['path']}")
            for qs, g in st["gains"].items():
                a, bq = qs.split("_")
                n += ds.add(sid, (int(a), int(bq)), float(g), st["base"], "random_sample",
                            fn, len(prefix), False)
        led.admit(fn, f"{len(d['states'])} random-prefix states "
                      f"({d['n_paths_dropped']} paths below {MIN_PATH_USERS} users dropped)", n)
        n_add += n
    return n_add


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="scrape", choices=["scrape", "meanvec", "sample"])
    ap.add_argument("--n_prefix", type=int, default=19)
    ap.add_argument("--seed", type=int, default=98765)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    args = ap.parse_args()
    if args.mode == "meanvec":
        run_meanvec()
        return
    if args.mode == "sample":
        from strategy_channel_suite import CERT_TOWER, CERT_CONC
        args.snapshot = args.snapshot or CERT_TOWER
        args.concept_ckpt = args.concept_ckpt or CERT_CONC
        run_sample(args)
        return
    ds, led = Dataset(), Ledger()
    nf = scrape_heaps(ds, led)
    log(f"[heaps] {nf} sidecars admitted")
    npk = scrape_greedy_json(ds, led)
    log(f"[greedy] {npk} pick measurements admitted")
    n3 = scrape_phase3(ds, led)
    log(f"[phase3] {n3} exhaustive measurements admitted")
    ns = scrape_sample(ds, led)
    log(f"[sample] {ns} random-sample measurements admitted")
    comp = compose(ds)
    log(json.dumps(comp, indent=1))
    write(ds, led, comp)


if __name__ == "__main__":
    main()
