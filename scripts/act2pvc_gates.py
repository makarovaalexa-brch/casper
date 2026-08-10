r"""act2pvc_gates.py -- the CONTROLS of docs/design/DS_ACT2_CONCEPT_BANK_PV.md section 5 that live
outside the quartile machinery: the entropy0 REPRODUCTION GATE (control 2) and the CHANNEL
ACCOUNTING table (control 5).

CONTROL 2, and why it is stated the way it is. The design sheet asks for the `fixedorder:entropy0`
interviews in `act2pvc` to be "byte-identical" to the ones in `act2pv`, asserted on the checkpoint
JSON. Two fields of that JSON can never be byte-identical between ANY two runs and neither is an
interview:

  * `info.turn_seconds`  -- wall-clock timings.
  * `_contract`          -- carries `concept_neg_cap`, which the design sheet itself REQUIRES to be 0
                            in act2pvc and which was 1 in act2pv (act2pv had no concept-asking arm, so
                            the flag was at its default and could not matter there).

So the gate is asserted on the INTERVIEW PAYLOAD -- `acts`, `asked`, `answered`, `conc_state`,
`turns_done`, `k`, `selector` -- compared as exact serialised bytes after a canonical re-dump. That
is every input any consumer downstream can see, which is the same standard `adaptive_harness`'s own
`--regression` check uses ("list equality ... i.e. every input any consumer downstream can see").
The excluded fields are printed, in full, rather than silently dropped.

A SECOND, STRICTLY STRONGER CHECK is run alongside it and is the one that actually licenses reusing
the banked PV: the per-user NDCG vectors of `fixedorder:entropy0` (and of `fixedorder:greedy_items`,
which is carried in both runs for exactly this purpose) must agree between the two runs at max|delta|
= 0.0e+00 on both axes at every budget. If the interview is identical and the scored vector is
identical, the states PV is read at are the same states.

  python scripts/act2pvc_gates.py --gate        # control 2
  python scripts/act2pvc_gates.py --channels    # control 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IVIEW = ROOT / "experiments" / "baselines" / "interview"

PV_LABEL = "act2pv"
NEW_LABEL = "act2pvc"
BUDGETS = (1, 2, 4, 8)
# Arms present in BOTH runs, so both can be checked. entropy0 is the gate proper (PV is read at its
# states); greedy_items is carried in act2pvc for exactly this reason and is checked for free.
SHARED_SPECS = ("fixedorder:entropy0", "fixedorder:greedy_items")
GATE_SPEC = "fixedorder:entropy0"

# The fields deliberately excluded from the payload comparison, with the reason each is excluded.
EXCLUDED = {
    "_contract": "carries concept_neg_cap, which the design sheet REQUIRES to differ (1 -> 0)",
    "info": "wall-clock turn_seconds; not an interview",
}


def ck(label, spec, k):
    return IVIEW / f"adaptive_table_{label}_interviews" / f"{spec.replace(':', '__')}_k{k}.json"


def payload(blob):
    """The interview itself: everything a consumer downstream can see."""
    return {key: blob[key] for key in
            ("turns_done", "k", "selector", "acts", "asked", "answered", "conc_state")}


def gate():
    ok = True
    print("=" * 100)
    print("CONTROL 2 -- entropy0 REPRODUCTION GATE")
    print("=" * 100)
    for spec in SHARED_SPECS:
        for k in BUDGETS:
            pa, pb = ck(PV_LABEL, spec, k), ck(NEW_LABEL, spec, k)
            if not pb.exists():
                print(f"  {spec:26s} k={k}: act2pvc checkpoint NOT YET WRITTEN -- gate not run")
                ok = False
                continue
            a, b = json.load(open(pa)), json.load(open(pb))
            sa = json.dumps(payload(a), sort_keys=True).encode()
            sb = json.dumps(payload(b), sort_keys=True).encode()
            same = sa == sb
            ok &= same
            print(f"  {spec:26s} k={k}: interview payload "
                  f"{'IDENTICAL' if same else '*** DIFFERS ***'} "
                  f"({len(sa):,} bytes; sha equal={same})")
            if not same:
                for key in ("acts", "asked", "answered", "conc_state", "turns_done", "k", "selector"):
                    if json.dumps(a[key], sort_keys=True) != json.dumps(b[key], sort_keys=True):
                        print(f"      field that differs: {key}")
    # the excluded fields, printed rather than dropped
    a = json.load(open(ck(PV_LABEL, GATE_SPEC, 8)))
    b = json.load(open(ck(NEW_LABEL, GATE_SPEC, 8)))
    print("\n  EXCLUDED FIELDS (printed, not silently dropped):")
    for key, why in EXCLUDED.items():
        print(f"    {key}: {why}")
    d = {kk: (a["_contract"].get(kk), b["_contract"].get(kk)) for kk in
         set(a["_contract"]) | set(b["_contract"])
         if a["_contract"].get(kk) != b["_contract"].get(kk)}
    print(f"    _contract fields that differ (act2pv -> act2pvc): {d}")

    # ---- the strictly stronger check: the SCORED per-user vectors
    print("\n  PER-USER NDCG IDENTITY (stronger than the payload gate):")
    A = np.load(IVIEW / f"adaptive_table_{PV_LABEL}_per_user.npz")
    bp = IVIEW / f"adaptive_table_{NEW_LABEL}_per_user.npz"
    if not bp.exists():
        print("    act2pvc per-user npz not yet written -- not run")
        return False
    B = np.load(bp)
    worst = 0.0
    for spec in SHARED_SPECS:
        for k in BUDGETS:
            for ax in ("full", "tail"):
                key = f"ours|{spec}|k{k}|{ax}"
                if key not in A.files or key not in B.files:
                    print(f"    {key}: MISSING in {'act2pv' if key not in A.files else 'act2pvc'}"
                          f" -- not run")
                    ok = False
                    continue
                va, vb = np.asarray(A[key], np.float64), np.asarray(B[key], np.float64)
                if va.shape != vb.shape:
                    print(f"    {key}: SHAPE MISMATCH {va.shape} vs {vb.shape}")
                    ok = False
                    continue
                m = float(np.max(np.abs(va - vb)))
                worst = max(worst, m)
                print(f"    {key:52s} n={len(va):5d}  max|delta| = {m:.1e}")
    print(f"\n  max|delta| over every shared (spec, budget, axis) = {worst:.1e} "
          f"(exactly zero: {worst == 0.0})")
    ok &= (worst == 0.0)
    print(f"\n  GATE: {'PASS' if ok else 'FAIL'}")
    return ok


def channels(specs=None, label=NEW_LABEL, as_json=False):
    """CONTROL 5 -- per selector per budget: what fraction of asked turns were CONCEPTS, and what
    fraction of the concept asks were REFUSED (the concept was unanswerable and burned the turn)."""
    specs = specs or ["conlinucb_bs_static", "conlinucb_mcr", "conucb", "dpe_concept",
                      "fixedorder:greedy_concepts", "fixedorder:greedy_items", "fixedorder:entropy0"]
    rows = []
    for spec in specs:
        for k in BUDGETS:
            p = ck(label, spec, k)
            if not p.exists():
                rows.append({"spec": spec, "k": k, "status": "not run"})
                continue
            d = json.load(open(p))
            if int(d.get("turns_done", 0)) != k:
                # A LIVE cell's per-turn checkpoint is a PREFIX of the interview, so reading it would
                # report (say) 5 turns of concept asks against an 8-turn denominator and understate
                # the concept fraction. Report it as unfinished, never as a number.
                rows.append({"spec": spec, "k": k, "status": "not run",
                             "turns_done": int(d.get("turns_done", 0))})
                continue
            n = len(d["acts"])
            turns = n * k
            casks = sum(len(a) for a in d["conc_state"]["asked"])
            toks = d["conc_state"]["tokens"]
            refus = sum(1 for t in toks for (_c, _v, f) in t if float(f) == 1.0)
            iasks = turns - casks
            ians = sum(len(a) for a in d["answered"])
            rows.append({
                "spec": spec, "k": k, "status": "ok", "n_users": n, "turns": turns,
                "concept_asks": casks, "concept_turn_frac": casks / turns,
                "concept_refusals": refus,
                "concept_refusal_rate": (refus / casks) if casks else None,
                "concept_answers": casks - refus,
                "item_asks": iasks, "item_answers": ians,
                "item_answer_rate": (ians / iasks) if iasks else None,
                "answered_turns_frac": ((casks - refus) + ians) / turns,
                "distinct_prefixes": len({tuple(a) for a in d["acts"]}),
            })
    if as_json:
        return rows
    print("=" * 118)
    print("CONTROL 5 -- CHANNEL ACCOUNTING (fraction of asked turns that were CONCEPTS; refusal rate)")
    print("=" * 118)
    print(f"{'selector':28s} {'k':>2s} {'turns':>8s} {'concept asks':>13s} {'%turns':>8s} "
          f"{'refused':>8s} {'refusal%':>9s} {'item ans%':>10s} {'any-answer%':>12s} {'prefixes':>9s}")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['spec']:28s} {r['k']:2d}  *** not run ***")
            continue
        rr = "-" if r["concept_refusal_rate"] is None else f"{r['concept_refusal_rate']:.3f}"
        ia = "-" if r["item_answer_rate"] is None else f"{r['item_answer_rate']:.3f}"
        print(f"{r['spec']:28s} {r['k']:2d} {r['turns']:8,d} {r['concept_asks']:13,d} "
              f"{r['concept_turn_frac']:8.3f} {r['concept_refusals']:8,d} {rr:>9s} {ia:>10s} "
              f"{r['answered_turns_frac']:12.3f} {r['distinct_prefixes']:9,d}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--channels", action="store_true")
    a = ap.parse_args()
    if not (a.gate or a.channels):
        raise SystemExit("pass --gate and/or --channels")
    rc = True
    if a.gate:
        rc = gate()
    if a.channels:
        channels()
    sys.exit(0 if rc else 1)


if __name__ == "__main__":
    main()
