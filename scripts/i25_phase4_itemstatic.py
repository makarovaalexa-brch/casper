"""Run the MISSING baseline: the strongest population-informed STATIC ITEM schedule (build_item_static),
which was defined in i25_phase4.py but never called in main(). This is the baseline that most threatens
the 'adaptive routing beats static' claim: a fixed popular-item list for everyone, refusal when the user
hasn't rated the item. Compares vs static B (concept), descent p-hat, all-item-8. Local, no LLM.
"""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i25_lib as L
import i25_phase4 as P4

def main():
    D = L.G.load_data(); FR = L.Frozen(D)
    model = L.Fold(); blob = torch.load(P4.CKPT_BEST, map_location="cpu")
    model.load_state_dict(blob["model"]); model.eval()
    users, concept_keys = P4.assemble(D, FR)
    print(f"[itemstatic] {len(users)} users")

    # rebuild the concept static (reference) + the item static (the missing baseline)
    schedule, tail = P4.build_static(FR, model, users, concept_keys)
    item_sched, cov = P4.build_item_static(FR, model, users, pool_size=60)
    print(f"[itemstatic] item schedule coverage (frac of users who rated each, known half): {cov}")

    arms = {}
    arms["static B (concept)"] = P4.eval_arm(FR, model, users, P4.pick_static(schedule, skip=False))
    arms["STATIC ITEM (population popular-item list)"] = P4.eval_arm(FR, model, users, P4.pick_item_static(item_sched))
    arms["descent k=4 (p-hat, realizable)"] = P4.eval_arm(FR, model, users, P4.pick_descent(schedule, 4, "phat"))
    arms["all-item-8 (|rating|)"] = P4.eval_arm(FR, model, users, P4.pick_all_item())

    ref = arms["static B (concept)"][0]
    refi = arms["STATIC ITEM (population popular-item list)"][0]
    print("\n==== vs concept-static AND vs item-static (anytime NDCG@10) ====")
    out = {"item_schedule_coverage": cov, "arms": {}}
    for name, (a, e, _) in arms.items():
        vB = P4.boot(a, ref)
        vI = P4.boot(a, refi)
        out["arms"][name] = dict(anytime=float(np.nanmean(a)), endpoint=float(np.nanmean(e)),
                                 vs_conceptstatic=vB, vs_itemstatic=vI)
        print(f"  {name:44s} any {np.nanmean(a):.4f} end {np.nanmean(e):.4f} "
              f"| vs concept {vB['delta']:+.4f} CI[{vB['ci'][0]:+.4f},{vB['ci'][1]:+.4f}] "
              f"| vs ITEMstatic {vI['delta']:+.4f} CI[{vI['ci'][0]:+.4f},{vI['ci'][1]:+.4f}]")
    json.dump(out, open("experiments/I25_phase4_itemstatic.json", "w"), indent=1)
    print("\nsaved experiments/I25_phase4_itemstatic.json")

if __name__ == "__main__":
    main()
