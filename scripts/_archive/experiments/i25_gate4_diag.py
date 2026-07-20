"""i25_gate4_diag.py -- diagnosis of the G-fold4 failure (mixed < item-only at matched budget).

Separates two hypotheses with paired CIs on the 298 study users:
  H-dilution : concept tokens ADDED ON TOP of the same item tokens LOWER NDCG (a fold defect).
  H-bandwidth: concepts don't hurt, they are just worth LESS than the items they displace
               (the structural bandwidth hierarchy; not a fold defect).

Arms (same shuffled item order per user as g_fold2/4, rng seed 4):
  A  item-4                      (first 4 known items)
  B  item-4 + concept-4          (same 4 items + top-4 concepts)     -> B-A = concept marginal
  C  item-8                      (first 8 known items)               -> C-A = item marginal (slots 5-8)
  D  item-8 + concept-4          (does adding concepts to a strong item fold hurt?)

Run: python scripts/i25_gate4_diag.py
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i25_lib as L
import i25_gates as GT

OUT = "experiments/I25_gate4_diag.json"


def main():
    D = L.G.load_data(); FR = L.Frozen(D)
    model = L.Fold(); blob = torch.load(GT.CKPT_BEST, map_location="cpu")
    model.load_state_dict(blob["model"]); model.eval()
    users = GT.load_users(D, FR)
    print(f"[diag] {len(users)} users, ckpt best {blob['state']['best_val']:.4f}", flush=True)

    rng = np.random.default_rng(4)                       # SAME seed as g_fold4
    res = {k: [] for k in "ABCD"}
    for us in users:
        known = us["known"]; cm = GT._cmean(known)
        order = list(known.keys()); rng.shuffle(order)
        ti4, nat4 = GT.item_tokens(FR, known, order[:4], cm)
        ti8, nat8 = GT.item_tokens(FR, known, order[:8], cm)
        tc4 = GT.concept_tokens(FR, D, known, 4, cm)
        vals = {}
        vals["A"] = L.ndcg10(FR, GT.fold(FR, model, ti4, nat4), us["tlike"], us["prof"])
        vals["B"] = L.ndcg10(FR, GT.fold(FR, model, ti4 + tc4, nat4), us["tlike"], us["prof"])
        vals["C"] = L.ndcg10(FR, GT.fold(FR, model, ti8, nat8), us["tlike"], us["prof"])
        vals["D"] = L.ndcg10(FR, GT.fold(FR, model, ti8 + tc4, nat8), us["tlike"], us["prof"])
        if any(v is None for v in vals.values()):
            continue
        for k in "ABCD":
            res[k].append(vals[k])
    arr = {k: np.array(v) for k, v in res.items()}
    out = dict(n=len(arr["A"]), means={k: round(float(a.mean()), 4) for k, a in arr.items()})
    for name, (x, y) in dict(concept_marginal_B_minus_A=("B", "A"),
                             item_marginal_C_minus_A=("C", "A"),
                             concepts_on_top_of_8_D_minus_C=("D", "C"),
                             mixed_vs_item8_B_minus_C=("B", "C")).items():
        out[name] = GT.boot_ci(arr[x] - arr[y])
    json.dump(out, open(OUT, "w"), indent=1)
    for k, v in out.items():
        print(k, v, flush=True)


if __name__ == "__main__":
    main()
