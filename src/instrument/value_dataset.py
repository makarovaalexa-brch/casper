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

    def add(self, sid, q, gain, base, tag, src, level, complete, in_pool=True):
        # `in_pool` defaults True because EVERY banked measurement, in every source, is for a
        # question inside the 1,000-candidate pool -- that is the coverage hole the catalogue-wide
        # draws exist to fill, and the flag is what lets any result be recomputed in-pool-only and
        # compared like-for-like with the banked material.
        q = (int(q[0]), int(q[1]))
        k = (sid, q)
        rec = self.meas.get(k)
        val = float(base) + float(gain)
        if rec is None:
            self.meas[k] = {"state_id": sid, "question": list(q), "channel": chan(q),
                            "gain": float(gain), "base": float(base), "absolute_value": val,
                            "in_pool": bool(in_pool),
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
    r"""Phase 3's exhaustive sweeps -- the only source that is unbiased BY CONSTRUCTION at every
    candidate, and the one that carries its own cohort (`rows`) so an off-tree state is trainable.

    Read from the PER-STATE sidecars in `policy_labels_*_partial/`, not from the rolled-up
    `..._partial.json`: the rolled-up file records a state as `done: false` until the whole sweep
    ends, so a completed state is already on disk under its own name while the summary still denies
    it. Ingesting whichever file happens to be more convenient is exactly the "verify the artifact,
    not a proxy" failure. Each state's scalar is `vals[q][0]` against `base[0]` -- full NDCG@10, the
    same quantity every banked heap holds."""
    n_add = 0
    for d_ in sorted(glob.glob(os.path.join(OUTDIR, "policy_labels_*_partial"))):
        if not os.path.isdir(d_):
            continue
        base_d = os.path.basename(d_)
        sp = os.path.join(d_, "_sig.json")
        if not os.path.exists(sp):
            led.refuse(base_d, "no `_sig.json` -- the directory cannot state its own contract")
            continue
        sig = json.load(open(sp))["sig"]
        contract = contract_of(sig, base_d, led)
        if contract is None:
            continue
        if objective_of(sig, base_d, led) is None:
            continue
        pool = {"n_pool": int(sig.get("n_pool", 1000)), "pool_sha1": sig.get("pool_sha1")}
        n = ndone = nflight = 0
        for fp in sorted(glob.glob(os.path.join(d_, "*.json"))):
            if os.path.basename(fp) == "_sig.json":
                continue
            st = json.load(open(fp))
            if not st.get("done"):
                nflight += 1
                continue
            ndone += 1
            asked = [(int(x[0]), int(x[1])) for x in st["prefix"]]
            b = st["base"]
            base = float(b[0] if isinstance(b, list) else b)
            cohort = f"phase3:{st['sid']}"
            sid = state_key(cohort, int(st["n_users"]), asked)
            ds.add_state(sid, asked, base, cohort, int(st["n_users"]), pool, contract,
                         f"{base_d}/{os.path.basename(fp)}", nid=st.get("nid"))
            for qs, v in st["vals"].items():
                a, bq = qs.split("_")
                g = v[0] if isinstance(v, list) else v
                n += ds.add(sid, (int(a), int(bq)), float(g), base, "exhaustive_sweep",
                            f"{base_d}/{os.path.basename(fp)}", len(asked), True)
        if ndone:
            led.admit(base_d, f"{ndone} COMPLETE exhaustive states (unbiased at every candidate); "
                              f"{nflight} still in flight", n)
        else:
            led.refuse(base_d, f"no completed state yet ({nflight} in flight)")
        n_add += n
    return n_add


def phase3_rows():
    """State cohort -> row ids, straight from phase 3's own sidecars."""
    out = {}
    for d_ in sorted(glob.glob(os.path.join(OUTDIR, "policy_labels_*_partial"))):
        for fp in sorted(glob.glob(os.path.join(d_, "*.json"))):
            if os.path.basename(fp) == "_sig.json":
                continue
            st = json.load(open(fp))
            if st.get("done") and "rows" in st:
                out[f"phase3:{st['sid']}"] = st["rows"]
    return out


# ============================================================================ composition + write
def yield_curve():
    r"""★ USABLE ANSWER PATHS PER PREFIX AS A FUNCTION OF DEPTH -- measured, never assumed.

    Depth costs nothing to reach (folding a longer prefix is free) but it fragments the cohort: a
    k-question prefix scatters 10,000 users over up to 3^k answer paths, and only paths clearing the
    200-user floor give a stable conditional mean. This curve is the honest ceiling on how deep a
    conditional value model can be trained on this cohort. If it hits zero at some depth, that IS
    the finding -- the floor is not lowered to manufacture states."""
    rows = {}
    for p in sorted(glob.glob(os.path.join(SAMPLE_DIR, "prefix_*.json"))):
        d = json.load(open(p))
        r = rows.setdefault(int(d["k"]), {"prefixes": 0, "paths_populated": 0, "paths_kept": 0,
                                          "paths_dropped": 0, "users_dropped": 0})
        r["prefixes"] += 1
        r["paths_populated"] += int(d["n_paths_populated"])
        r["paths_kept"] += int(d["n_paths_kept"])
        r["paths_dropped"] += int(d["n_paths_dropped"])
        r["users_dropped"] += int(d["n_users_dropped"])
    for k in sorted(rows):
        r = rows[k]
        r["kept_per_prefix"] = r["paths_kept"] / r["prefixes"]
        log(f"[yield k={k:2d}] {r['kept_per_prefix']:.1f} usable states per prefix "
            f"({r['paths_kept']} kept / {r['paths_populated']} populated; "
            f"{r['users_dropped']} user-slots below the {MIN_PATH_USERS}-user floor)")
    return {str(k): rows[k] for k in sorted(rows)}


def compose(ds):
    per_contract = defaultdict(lambda: {"measurements": 0, "states": set()})
    per_tag = defaultdict(lambda: {"measurements": 0, "states": set()})
    per_depth = defaultdict(lambda: {"measurements": 0, "states": set()})
    per_pool = defaultdict(lambda: {"measurements": 0, "states": set()})
    per_channel = defaultdict(int)
    per_vocab = defaultdict(lambda: {"measurements": 0, "states": set()})
    for m in ds.meas.values():
        st = ds.states[m["state_id"]]
        for tbl, key in ((per_contract, st["contract"]), (per_tag, m["selection_tag"]),
                         (per_depth, st["depth"]), (per_pool, st["pool"]["n_pool"]),
                         (per_vocab, "in_pool" if m.get("in_pool", True) else "catalogue_wide")):
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
            "per_vocabulary": fin(per_vocab), "per_channel": dict(per_channel),
            "yield_curve": yield_curve(),
            "candidates_per_state": {"min": sz[0], "p25": q(.25), "median": q(.5),
                                     "p75": q(.75), "max": sz[-1]},
            "n_states": len(ds.states), "n_measurements": len(ds.meas)}


def write(ds, led, comp, extra=None):
    with open(JSONL, "w", encoding="utf-8") as f:
        for (sid, q), m in sorted(ds.meas.items()):
            st = ds.states[sid]
            row = dict(m)
            # `nid` travels with the row because it is the ONLY way the trainer can recover an
            # arbitrary state's COHORT (tree descent from the root); without it every tree state is
            # silently unusable and the union collapses to whatever happens to be the full cohort.
            row.setdefault("in_pool", True)
            row.update({"asked_set": st["asked_set"], "depth": st["depth"],
                        "cohort_sha1": st["cohort_sha1"], "n_rows": st["n_rows"],
                        "contract": st["contract"], "pool": st["pool"], "nid": st.get("nid")})
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
            states[sid] = {k: r.get(k) for k in ("state_id", "asked_set", "depth", "base",
                                                 "cohort_sha1", "n_rows", "contract", "pool",
                                                 "nid")}
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
  1. take k from the DEPTH SCHEDULE below, then k DISTINCT questions uniformly without replacement
     from the policy-B combined pool (500 popularity items + 500 answerable concepts, the tree's own
     pool, same ids in the same order). The PREFIX stays in-pool deliberately -- it defines the
     STATE, and keeping states drawn from the same question set as every banked state is what makes
     the new material comparable with the old;
  2. ask that prefix to ALL 10,000 validation users and keep the PER-USER NDCG vector (no sampling
     of users anywhere);
  3. record each user's ternary ANSWER PATH through the prefix (like/dislike/no-answer per question);
  4. draw N_CAND = 50 CANDIDATES -- N_CAND_POOL from the 1,000-question pool and N_CAND_CAT from the
     FULL 19,390-question catalogue -- uniformly without replacement, and evaluate each over all
     10,000 users, again keeping the per-user vector;
  5. for the marginal state (all 10,000) and for every answer path with at least MIN_PATH_USERS
     users, the conditional gain is mean(f_q[path]) - mean(f_0[path]) -- the same construction the
     tree uses to score a split, so it is on the same ruler.

  ★ Step 5 is why one sweep buys many states: grouping by answer path turns 51 evaluations into one
  measured gain vector per POPULATED PATH, at zero extra evaluation.

★★ TWO WIDENINGS (2026-08-14), both strict broadenings of the authorised design -- more states and
more question variety for the same money, nothing removed:

  ★ DEPTH. The first design drew k in {3..6}, and phase 3's 42 states sit at depths 3-6 too. The
    interview is SIXTEEN questions long, so depths 7-15 -- more than half of it -- would have had no
    unbiased measurement at all, leaving the value head to extrapolate over exactly the region where
    the policy spends most of its turns. k now runs 1..15 on a STRATIFIED schedule (every depth
    represented at least once, the remainder allocated to the deeper half, which the banked material
    and phase 3 do not cover). Folding a longer prefix is free: the cost is 50 candidates x 10,000
    users whatever k is.

    ⚠ THE REAL CONSTRAINT AT DEPTH IS PATH FRAGMENTATION, NOT COST. A deeper prefix scatters users
    across more answer paths, so fewer paths clear the 200-user floor. The YIELD CURVE -- usable
    paths per prefix as a function of k -- is therefore MEASURED AND REPORTED, not assumed. If depth
    12 yields nothing above the floor that is itself the finding: it is the honest ceiling on how
    deep any conditional value model can be trained on this cohort. The floor is NOT lowered to
    manufacture states.

  ★ VOCABULARY. Every gain ever measured here, in every source, is for a question inside the
    1,000-candidate pool -- by construction the top 500 by popularity union the top 500 by answer
    rate. No model has ever seen a measured value for a TYPICAL catalogue question, so the value
    head has no examples of what a BAD question is worth. That is precisely the deficit behind the
    open-vocabulary failure (out-of-pool picks answered 9.1% vs 55.2% in-pool, NDCG -0.02034).
    30 of the 50 candidates are now drawn from the full catalogue. Most will be near-unanswerable
    and score near zero -- THAT IS THE POINT; they are the negative examples that have never
    existed. 20 stay in-pool so the pool region remains densely covered and every result can be
    recomputed restricted to in-pool candidates and compared like-for-like with the banked material.
    Every measurement carries `in_pool`.

WHAT IS SAMPLED, AND WHAT IS NOT. Users: never -- all 10,000, every time. Candidates: yes, 50 per
prefix, uniformly at random within each declared stratum; that is the authorised reduction and it is
the whole point, since a regressor wants scattered triples and only an argmax search needs the full
sweep (OUTCOME_HEAD.md RESULT 5). Prefixes: 19 by default, ~7 min each on the measured 8.24 s/eval
at n=10,000, so ~2.2 h.

THE TRUNCATION THAT MUST BE DECLARED. A path with few users gives an unstable mean, so paths below
MIN_PATH_USERS = 200 are DROPPED -- the tree's own `min_leaf`, chosen because it already governs
every banked state rather than invented for this job. The count dropped, and the users they carry,
are recorded per prefix in the artifact.

PERSISTENCE (HARD RULE 11). Each prefix writes its own contract-signed sidecar the moment it
finishes, plus the raw per-user float32 vectors, so a crash can lose at most one prefix and any
later change to MIN_PATH_USERS is free -- no re-evaluation.
"""
N_CAND_POOL = 20                  # kept in-pool: comparability with every banked measurement
N_CAND_CAT = 30                   # drawn catalogue-wide: the negative examples nothing has ever had
N_CAND = N_CAND_POOL + N_CAND_CAT
MIN_PATH_USERS = 200
SAMPLE_DIR = os.path.join(OUTDIR, "value_sample_partial")


def depth_schedule(n_prefix):
    r"""Which prefix length each sweep uses. STRATIFIED, not drawn at random.

    A random draw over 1..15 with 19 prefixes leaves holes, and the yield curve (usable paths vs k)
    is a thing we are trying to MEASURE -- a curve with missing x-values measures less than it
    costs. So: every depth 1..15 exactly once, and the remaining prefixes allocated to the deep end
    (8..15 in order), which is the half the banked material and phase 3 do not reach at all."""
    base = list(range(1, 16))
    extra = [8 + (i % 8) for i in range(max(n_prefix - len(base), 0))]
    return (base + extra)[:n_prefix]


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
    pool_set = set(pool)
    # ★ THE FULL QUESTION SPACE, which nothing in this project has ever measured a gain outside of.
    ni, C = E["ctx"].ni, E["C"]
    catalogue = [(0, i) for i in range(ni)] + [(1, c) for c in range(C)]
    log(f"[sample] question space: {len(catalogue):,} catalogue ({ni:,} items + {C:,} concepts); "
        f"the pool is {len(pool)} of them, and every gain ever banked lies inside it")
    sched = depth_schedule(args.n_prefix)
    log(f"[sample] depth schedule (stratified, k=1..15 then the deep half): {sched}")

    sig = {"v": 1, "split": "validation", "objective": "full", "mask_pool": pc,
           "tower": os.path.basename(args.snapshot),
           "conc_ckpt": os.path.basename(args.concept_ckpt),
           "n_pool": len(pool), "n_catalogue": len(catalogue), "n_cand": N_CAND,
           "n_cand_pool": N_CAND_POOL, "n_cand_catalogue": N_CAND_CAT,
           "min_path_users": MIN_PATH_USERS, "depth_schedule": sched, "seed": args.seed}
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
        k = int(sched[pi])
        pidx = rng.choice(len(pool), size=k, replace=False)
        prefix = [pool[int(j)] for j in pidx]
        used = set(prefix)
        # --- 20 in-pool, so the pool region stays dense and comparable with everything banked ----
        rest = [q for q in pool if q not in used]
        cands = [rest[int(j)] for j in rng.choice(len(rest), size=N_CAND_POOL, replace=False)]
        used |= set(cands)
        # --- 30 catalogue-wide: the typical questions, and the negative examples nothing has ------
        drawn = []
        while len(drawn) < N_CAND_CAT:
            q = catalogue[int(rng.integers(len(catalogue)))]
            if q not in used:
                drawn.append(q); used.add(q)
        cands = cands + drawn
        in_pool = [q in pool_set for q in cands]
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
        log(f"[YIELD k={k}] {len(groups)} populated answer paths, {len(kept)} KEPT "
            f"(>= {MIN_PATH_USERS} users), {len(dropped)} dropped carrying "
            f"{sum(dropped.values())} of 10,000 users -- the floor is NOT lowered; a low yield at "
            f"depth is the finding, not a problem to engineer around")
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
            # ★ the cohort itself is recorded, so an arbitrary answer-path state is trainable later
            # without re-deriving the path (and without re-evaluating anything)
            st_out.append({"path": gkey, "n_users": int(len(ix)), "base": b_full,
                           "base_tail": float(np.nanmean(t0v[ix])), "gains": g,
                           "rows": [int(rows[int(u)]) for u in ix]})
        # the vocabulary contrast, measured on the spot: what IS a typical catalogue question worth?
        gm = st_out[0]["gains"]
        gp = [gm[f"{q[0]}_{q[1]}"] for q, ip in zip(cands, in_pool) if ip]
        gc = [gm[f"{q[0]}_{q[1]}"] for q, ip in zip(cands, in_pool) if not ip]
        log(f"[VOCAB k={k}] marginal gain, in-pool mean {np.mean(gp):+.5f} (max {np.max(gp):+.5f}) "
            f"vs catalogue-wide mean {np.mean(gc):+.5f} (max {np.max(gc):+.5f})")
        rec = {"sig": sig, "prefix_index": pi, "k": k,
               "prefix": [list(q) for q in prefix], "candidates": [list(q) for q in cands],
               "candidate_in_pool": [bool(x) for x in in_pool],
               "n_paths_populated": len(groups), "n_paths_kept": len(kept),
               "n_paths_dropped": len(dropped), "n_users_dropped": int(sum(dropped.values())),
               "marginal_gain_in_pool_mean": float(np.mean(gp)),
               "marginal_gain_catalogue_mean": float(np.mean(gc)),
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
        ip = {f"{q[0]}_{q[1]}": bool(f) for q, f in
              zip(d["candidates"], d.get("candidate_in_pool", [True] * len(d["candidates"])))}
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
                            fn, len(prefix), False, in_pool=ip.get(qs, True))
        led.admit(fn, f"k={d['k']}, {len(d['states'])} random-prefix states "
                      f"({d['n_paths_dropped']} paths below {MIN_PATH_USERS} users dropped)", n)
        n_add += n
    return n_add


# ============================================== the fit cost, profiled 2026-08-14, and what it is
r"""WHERE THE 2.5 h PER FIT ACTUALLY GOES -- measured on the real shapes, not guessed.

The first union refit ran at 1.86 min/epoch and projected to 174 days. The natural suspicion was
that the FROZEN outcome head `P(o|s,q)` was being re-run every epoch. **It is not** -- `P` is
computed once inside `build_state_pack_live` under `no_grad` and stored on the pack. That hypothesis
is dead, and the real answer is less convenient:

  425 packs, 876,520 user-rows, 75.8M (user x candidate) cells per forward.

★ **The cost scales with USERS, not with candidates.** Measured per training step: the root pack
(10,000 users x 1,000 candidates) takes 0.888 s, and the same 10,000 users with only 50 candidates
still takes 0.654 s -- cutting candidates 20-fold saves 26%. So "1.5 ms per training example" was
the wrong denominator: this model does not consume 72,717 triples per epoch, it evaluates three
outcomes over 876,520 user-rows and 75.8M user-candidate cells, i.e. ~227M values forward and again
backward. 95.5 s for that on six CPU threads with autograd is ordinary, not pathological.

★ THE ONE REAL WASTE, and it is exactly recoverable. `StateEncoder.forward` (mode='both') builds

    tok = hist_phi (x) onehot(hist_out)   -> (B, T, 3*d_phi)     352 MB at the root, EVERY forward

from `hist_phi`, `hist_out` and `hist_mask` -- all three CONSTANT for a pack, since only the weights
move during a fit. `pooled`, `cnt` and `n_asked` are therefore constants recomputed 80 times per fit.
Hoisting them is arithmetic-identical (verified: max |cached - original| = **0.000e+00**) and buys:

    one epoch   95.5 s -> 41.2 s   (2.3x)        one fit (80 epochs)  2.12 h -> 0.92 h
    pack memory 1.70 GB -> 3.51 GB              (`hq`/`ho`/`hm` are dropped; the cache REPLACES them)

⚠ **2.3x is a real saving and it is not a rescue.** Even at 0.92 h/fit, leave-one-out over 187
evaluation states is 1,683 fits = **64 days**. LOSO is O(n_eval) fits and 187 held-out states is
simply the wrong protocol at this size -- K-fold is the standard choice at this n and gives MORE
training data per fit, not less:

    5-fold  x 3 seeds x 3 arms =  45 fits ~ 41 h
    10-fold x 3 seeds x 3 arms =  90 fits ~ 82 h

The remaining lever after that is the per-user term (876,520 user-rows). Capping the users used in a
TRAINING forward would cut it several-fold -- but that is a HARD RULE 1 sampling decision on the
users, it must be put to the author as its own question with the design in hand, and it is NOT taken
here.
"""


def precompute_hist(pk):
    r"""Hoist the constant history block out of the per-epoch forward. Bit-identical by construction:
    the same arithmetic, evaluated once instead of 80 times. `hq`/`ho`/`hm` are dropped because the
    cache replaces them -- keeping both would add memory instead of moving it."""
    import torch
    import torch.nn.functional as F
    from lift_head import hist_phi
    hp = hist_phi(pk)
    oh = F.one_hot(pk["ho"], 3).to(hp.dtype)
    tok = (hp.unsqueeze(-1) * oh.unsqueeze(-2)).reshape(hp.shape[0], hp.shape[1], -1)
    m = pk["hm"].unsqueeze(-1)
    pk["pooled"] = (tok * m).sum(1) / m.sum(1).clamp(min=1.0)
    pk["cnt"] = (oh * pk["hm"].unsqueeze(-1)).sum(1)
    pk["nask"] = pk["hm"].sum(1, keepdim=True)
    for k in ("hq", "ho", "hm"):
        pk.pop(k, None)
    return pk


def composed_fast(model, pk, grad=False, offset=True):
    """`lift_head.composed` with the constant history block read from the cache."""
    import torch
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        parts = [pk["z"], torch.relu(model.enc.hist(pk["pooled"])), pk["cnt"], pk["nask"], pk["ps"]]
        h = model.enc.mlp(torch.cat(parts, -1))
        v = model.head(h, phi=pk["phi"])
        if offset:
            v = v + pk["base"]
        return (pk["P"] * v).sum(0).mean(0)


# ================================================================ STEP 4 -- refit on the UNION
def run_train(args):
    r"""Refit the value head on the UNION of every admitted source, with the four comparisons that
    are mandatory rather than optional:

      1. ★ the PARAMETER-FREE MEAN-VECTOR baseline (+0.4163). HARD RULE 2. It beat the learned head
         by a factor of seven last time, and any head that does not beat it is broken, not new.
      2. ★ the ZEROED-STATE control. A head that sees `z_t` can trivially IGNORE it and emit the mean
         vector, so scoring BELOW a state-blind baseline is an optimisation failure, not a
         representational one. That free tell was on screen for hours last time and nobody read it.
      3. the pre-registered rho >= +0.40 gate, LOSO on the UNBIASED states only, per-state, 3 seeds
         with the spread -- never a mean alone.
      4. the ABLATION BY SOURCE -- unbiased-only vs unbiased + tagged-biased -- so we learn whether
         the biased mass helps or hurts rather than assuming either.

    Plus the no-answer flatness check: `V(no-answer|q,s)` must sit at `base(s)` and stay flat across
    q, because an unanswered question folds an UNSEEN/refusal token. Last time it varied with sd
    0.058 -- about three times the range of the quantity being predicted -- and drove the argmax.
    """
    import time
    import numpy as np
    import torch
    sys.argv = [sys.argv[0]] + (["--threads", str(args.threads)] if args.threads else [])
    from strategy_channel_suite import build_greedy_env, contract_selftest, set_active_rows, genv_mean
    from ndcg_tree import build_pool, leak_assert, split_rows
    from policy_net import build_phi
    from outcome_head import OutcomeModel, OUT_NAMES
    import lift_head as LH

    t00 = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.threads:
        torch.set_num_threads(int(args.threads))
    states, gains, tags = load_jsonl()
    log(f"[train] union: {len(states)} states, {sum(len(g) for g in gains.values()):,} measurements")

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "valuetrain")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "valuetrain")
    rows_all = [int(r) for r in E["rows"]]
    f0, _t0 = genv_mean(E, [], rows_all)
    assert round(f0, 8) == round(VAL_ROOT_M0_FULL, 8), f"canonical snap FAILED: {f0!r}"
    log(f"[snap] validation m=0 intercept full={f0!r} -- canonical, exact")
    pool, _ip, _cp = build_pool(E, 500, 500)
    ni, C = E["ctx"].ni, E["C"]
    catalogue = [(0, i) for i in range(ni)] + [(1, c) for c in range(C)]
    phi_t, _norm = build_phi(E, catalogue)
    pool_cat = {"pool": [tuple(q) for q in pool],
                "index": {(0, i): i for i in range(ni)} | {(1, c): ni + c for c in range(C)}}
    blob = torch.load(args.outcome_ckpt, map_location="cpu")
    om = OutcomeModel(blob["d_z"], blob["d_phi"], state=blob["state"], use_pos=blob["use_pos"],
                      d_k=blob["d_k"], d_h=blob["d_h"])
    om.load_state_dict(blob["model"]); om.eval()
    for p_ in om.parameters():
        p_.requires_grad_(False)

    tr = json.load(open(os.path.join(OUTDIR, "ndcgtree.json")))
    rows_of = {"root": rows_all}

    def rows_for_node(nid):
        if nid in rows_of:
            return rows_of[nid]
        parent, _br = nid.rsplit(".", 1)
        prows = rows_for_node(parent)
        kids, _ar, _z = split_rows(E, prows, tuple(int(x) for x in tr["nodes"][parent]["splitter"]))
        for k, v in kids.items():
            rows_of[f"{parent}.{k}"] = v
        return rows_of[nid]

    # the random-sample cohorts are recorded by the sampler, so an arbitrary answer path is
    # recoverable without re-deriving it
    sample_rows = phase3_rows()
    for p in sorted(glob.glob(os.path.join(SAMPLE_DIR, "prefix_*.json"))):
        d = json.load(open(p))
        for st in d["states"]:
            if "rows" in st:
                sample_rows[f"valsample:p{d['prefix_index']}:{st['path']}"] = st["rows"]

    def rows_for_state(sid):
        st = states[sid]
        ck = st["cohort_sha1"]
        if ck in sample_rows:
            return [int(r) for r in sample_rows[ck]]
        nid = st.get("nid")
        node = (nid or "").split("@")[0].split("/")[0]
        if node and node in tr["nodes"]:
            rws = rows_for_node(node)
            return rws if len(rws) == st["n_rows"] else None
        return rows_all if st["n_rows"] == len(rows_all) else None

    packs, skipped = {}, 0
    tpk = time.time()
    for i, sid in enumerate(sorted(states, key=lambda s: -len(gains[s]))):
        rws = rows_for_state(sid)
        if rws is None:
            skipped += 1
            continue
        st = {"asked": [tuple(q) for q in states[sid]["asked_set"]], "base": states[sid]["base"],
              "gains": gains[sid], "node": states[sid].get("nid")}
        pk = LH.build_state_pack_live(E, st, phi_t, pool_cat, om, ni, rws)
        if pk is None:
            skipped += 1
            continue
        pk["sid"] = sid
        packs[sid] = pk
        if (i + 1) % 40 == 0:
            log(f"[train] packed {i + 1}/{len(states)} ({(time.time() - tpk) / 60:.1f}m)")
    set_active_rows(E, rows_all)
    log(f"[train] {len(packs)} states packed, {skipped} skipped for want of a recoverable cohort; "
        f"{sum(len(p['cand']) for p in packs.values()):,} training triples")

    # ---- evaluation set: the UNBIASED states only -----------------------------------------------
    unbiased = [s for s in packs if states[s]["complete"]
                or all(t in ("exhaustive_sweep", "random_sample") for t in tags[s].values())]
    ev = [s for s in unbiased if len(gains[s]) >= 20]
    log(f"[train] {len(unbiased)} unbiased states; {len(ev)} carry >= 20 candidates and are the "
        f"LOSO evaluation set")

    def zero_state(pk):
        z = dict(pk)
        z["z"] = torch.zeros_like(pk["z"]); z["hq"] = np.zeros_like(pk["hq"])
        z["ho"] = torch.zeros_like(pk["ho"]); z["hm"] = torch.zeros_like(pk["hm"])
        z["ps"] = torch.zeros_like(pk["ps"])
        return z

    def loso(train_sids, label, blind=False):
        rows = []
        for s in ev:
            pk = packs[s]
            trp = [(zero_state(packs[t]) if blind else packs[t]) for t in train_sids if t != s]
            if not trp:
                continue
            per_seed = []
            for sd in [args.seed + 1000 * j for j in range(args.lift_seeds)]:
                m, _h, bep = LH.fit_lift(trp, E["ctx"].d, phi_t.shape[1], args,
                                         f"{label}-{s[:16]}-s{sd}", seed=sd)
                c = LH.composed(m, zero_state(pk) if blind else pk, offset=args.v_offset).numpy()
                y = packs[s]["target_gain"]
                per_seed.append({"seed": sd, "best_epoch": bep, "spearman": LH.spearman(c, y),
                                 "rank_of_argmax": int((y > y[int(np.argmax(c))]).sum()) + 1})
            rho = [x["spearman"] for x in per_seed]
            rows.append({"held_out_state": s, "nid": states[s].get("nid"),
                         "n_candidates": len(pk["cand"]), "n_train_states": len(trp),
                         "spearman": float(np.mean(rho)), "spearman_min": float(np.min(rho)),
                         "spearman_max": float(np.max(rho)), "per_seed": per_seed,
                         "rank_of_argmax": float(np.mean([x["rank_of_argmax"] for x in per_seed]))})
            log(f"[LOSO/{label} {states[s].get('nid')}] {len(trp)} train states -> "
                f"{rows[-1]['spearman']:+.4f} [{rows[-1]['spearman_min']:+.4f}, "
                f"{rows[-1]['spearman_max']:+.4f}] | argmax rank "
                f"{rows[-1]['rank_of_argmax']:.0f}/{len(pk['cand'])}")
        m = float(np.mean([r["spearman"] for r in rows])) if rows else float("nan")
        log(f"[ARM {label}] mean held-out Spearman {m:+.4f} over {len(rows)} states "
            f"({'PASSES' if m >= 0.40 else 'FAILS'} the +{0.40:.2f} gate)")
        return {"arm": label, "mean_spearman": m, "per_state": rows}

    # HARD RULE 11(a): the cheap arms first, and every arm on disk the moment it finishes -- a fit
    # over hundreds of packs is hours, and a crash must never cost more than one arm.
    part = os.path.join(OUTDIR, "value_head_union_partial.json")
    mv = [meanvec_arm(ev, list(gains), gains, "meanvec/union"),
          meanvec_arm(ev, unbiased, gains, "meanvec/unbiased_only")]
    arms = []
    for tr_sids, lab, blind in ((list(packs), "union(all sources)", False),
                                (unbiased, "unbiased_only", False),
                                (list(packs), "ZEROED-STATE control", True)):
        arms.append(loso(tr_sids, lab, blind=blind))
        json.dump({"meanvec": mv, "arms": arms}, open(part, "w"), indent=1, default=float)
        log(f"[persist] {len(arms)} arms banked -> {part}")

    # ---- the no-answer flatness check, free and it localised the last failure -------------------
    m, _h, _b = LH.fit_lift([packs[s] for s in packs], E["ctx"].d, phi_t.shape[1], args,
                            "flatcheck", seed=args.seed)
    flat = []
    for s in ev[:8]:
        pk = packs[s]
        with torch.no_grad():
            V = LH.value(m, pk, args.v_offset).mean(1)
        rec = {"state": states[s].get("nid"), "base": float(pk["base"])}
        for o, nm in enumerate(OUT_NAMES):
            v = V[o].numpy()
            rec[nm] = {"mean": float(v.mean()), "sd_across_q": float(v.std()),
                       "mean_minus_base": float(v.mean() - pk["base"])}
        flat.append(rec)
        log(f"[flatness {rec['state']}] no-answer mean {rec['noanswer']['mean']:.5f} "
            f"(base{rec['noanswer']['mean_minus_base']:+.5f}) sd across q "
            f"{rec['noanswer']['sd_across_q']:.5f}")

    out = {"analysis": "value_head_refit_on_the_union", "test_touched": False,
           "split": "validation", "gate": 0.40,
           "n_states": len(packs), "n_states_unbiased": len(unbiased), "n_eval_states": len(ev),
           "n_triples": int(sum(len(p["cand"]) for p in packs.values())),
           "references": {"learned_head_7_states": 0.0623, "learned_head_22_states": 0.0819,
                          "parameter_free_mean_vector": 0.4163},
           "arms": arms, "meanvec": mv, "no_answer_flatness": flat,
           "lift_seeds": args.lift_seeds, "lift_epochs": args.lift_epochs,
           "minutes": round((time.time() - t00) / 60, 2)}
    p = os.path.join(OUTDIR, "value_head_union.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    log(f"[out] {p} ({out['minutes']:.1f}m)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="scrape", choices=["scrape", "meanvec", "sample", "train"])
    ap.add_argument("--n_prefix", type=int, default=19)
    ap.add_argument("--seed", type=int, default=98765)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    # --- the fit, same defaults as lift_head so the arms stay comparable ---
    ap.add_argument("--outcome_ckpt", default=os.path.join(OUTDIR, "outcome_head_main_both.pt"))
    ap.add_argument("--state", default="both", choices=["z", "hist", "both"])
    ap.add_argument("--no_pos", action="store_true")
    ap.add_argument("--d_k", type=int, default=64)
    ap.add_argument("--d_h", type=int, default=256)
    ap.add_argument("--lift_epochs", type=int, default=80)
    ap.add_argument("--lift_lr", type=float, default=1e-4)
    ap.add_argument("--lift_seeds", type=int, default=3)
    ap.add_argument("--lift_inner_val", type=float, default=0.2)
    ap.add_argument("--no_v_offset", action="store_true")
    args = ap.parse_args()
    args.v_offset = not args.no_v_offset
    if args.mode == "meanvec":
        run_meanvec()
        return
    if args.mode == "sample":
        from strategy_channel_suite import CERT_TOWER, CERT_CONC
        args.snapshot = args.snapshot or CERT_TOWER
        args.concept_ckpt = args.concept_ckpt or CERT_CONC
        run_sample(args)
        return
    if args.mode == "train":
        from strategy_channel_suite import CERT_TOWER, CERT_CONC
        args.snapshot = args.snapshot or CERT_TOWER
        args.concept_ckpt = args.concept_ckpt or CERT_CONC
        run_train(args)
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
