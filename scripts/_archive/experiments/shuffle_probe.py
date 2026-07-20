"""3-USER SHUFFLE REPEATABILITY PROBE. Author-directed 2026-07-09. Hard cap $1.

Measures judge presentation-flutter: re-send the SAME user's SAME item questions 3x with shuffled
question order AND shuffled profile-line order (seeds 1,2,3). Content identical; only presentation
changes (so prompt strings differ -> no caching). Reuses scripts/answerer_smallbatch.py machinery
exactly (judge gpt-5.4-mini temp 0, .env key, usage logging, compact [i,k,h,c], full-profile no-mask).

Users: 3 picked deterministically from the 173 in answerer_v1_grid173_WORKING.json (one low/mid/high
item-know-well rate, tercile x rng seed 0). For each user take the first N_ITEMS item cells they were
already judged on, and re-send that exact set 3 times.

Run:  python scripts/shuffle_probe.py --collect
      python scripts/shuffle_probe.py --analyze
"""
import os, sys, json, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import answerer_smallbatch as A
G = A.G

GRID173 = ".cache/instrument2/answerer_v1_grid173_WORKING.json"
OUT_JSON = "experiments/SHUFFLE_PROBE.json"
OUT_MD = "experiments/SHUFFLE_PROBE.md"
USAGE_JSON = ".cache/instrument2/shuffle_probe_usage.json"
N_ITEMS = 250          # one item-block worth
REPEAT_SEEDS = [1, 2, 3]
KNOW = ["no_clue", "rough_idea", "know_well"]
KIDX = {k: i for i, k in enumerate(KNOW)}


def pick_users():
    """Deterministic: per-user item know-well rate over LLM item cells (>=N_ITEMS), tercile split,
    one user per tercile via rng(0). Returns [(uid, label, rate), ...] low/mid/high."""
    blob = json.load(open(GRID173))
    rows = []
    for uid, rec in blob["users"].items():
        items = [v["ans"] for v in rec["Q"].values()
                 if v["channel"] == "item" and v["ans"] and "_raw" not in v["ans"]]
        if len(items) < N_ITEMS:
            continue
        kw = sum(1 for a in items if a.get("knowledge") == "know_well")
        rows.append((uid, kw / len(items)))
    rows.sort(key=lambda r: r[1])
    n = len(rows)
    groups = [("low", rows[:n // 3]), ("mid", rows[n // 3:2 * n // 3]), ("high", rows[2 * n // 3:])]
    rng = np.random.default_rng(0)
    picked = []
    for lab, g in groups:
        i = int(rng.integers(len(g)))
        picked.append((g[i][0], lab, g[i][1]))
    return picked, blob


def user_items(rec):
    """First N_ITEMS item cells (by question index) the user was judged on. Returns list of
    (j, orig_knowledge, orig_stars_or_None)."""
    items = [(int(i), v) for i, v in rec["Q"].items()
             if v["channel"] == "item" and v["ans"] and "_raw" not in v["ans"]]
    items.sort()
    out = []
    for _, v in items[:N_ITEMS]:
        a = v["ans"]
        out.append((int(v["j"]), a.get("knowledge"), a.get("stars")))
    return out


def collect():
    picked, blob = pick_users()
    print(f"[probe] users: {[(u, l, round(r, 3)) for u, l, r in picked]}", flush=True)
    D = G.load_data()
    split = G.build_split(D)

    results = {"model": None, "temp": G.TEMPERATURE, "split_seed": G.ANSWERER_SPLIT_SEED,
               "n_items": N_ITEMS, "repeat_seeds": REPEAT_SEEDS, "users": {}}
    tokstat = {"completion": 0, "prompt": 0, "questions": 0, "calls": 0}

    for uid, lab, rate in picked:
        u = int(uid)
        rec = blob["users"][uid]
        items = user_items(rec)                       # [(j, know, stars)]
        js = [j for j, _, _ in items]
        orig = {j: (k, s) for j, k, s in items}       # original cached judgment
        kn, ho = split[u]
        rat = dict(D["rat_by_u"][u])
        N_total = len(rat)
        plines, capped = A.profile_lines(D, kn, rat)   # EXACT same profile content as original
        n_shown = len(plines)

        # leakage guard sanity: none of the probed items may be held-out targets
        assert not (set(js) & set(ho)), f"leak: probe items in held-out for u{u}"

        user_out = {"label": lab, "orig_kw_rate": rate, "n_shown": n_shown, "N_total": N_total,
                    "profile_capped": bool(capped), "items_j": js,
                    "orig": {str(j): {"k": orig[j][0], "s": orig[j][1]} for j in js},
                    "repeats": {}}

        for seed in REPEAT_SEEDS:
            rng = np.random.default_rng(seed)
            qperm = list(range(len(js)))
            rng.shuffle(qperm)                        # shuffled question order
            pl = plines[:]
            rng.shuffle(pl)                           # shuffled profile-line order
            prof = "; ".join(pl) if pl else "(no ratings shown)"
            if capped:
                prof = "[a star-stratified sample of this viewer's ratings]\n" + prof
            lines = [f"{pos}: film '{D['title'][js[qperm[pos]]]}'" for pos in range(len(qperm))]
            user_msg = (f"Viewer's rating sample (size {n_shown} of {N_total} total; they have seen "
                        f"many more films than listed):\n{prof}\n\nQuestions:\n" + "\n".join(lines))
            arr = A.call_and_count(A.SYS_PROD, user_msg, len(lines), tokstat)
            rep = {}
            for pos, o in enumerate(arr):
                f = A._row_fields(o)
                if f is None:
                    continue
                try:
                    li = int(f[0])
                except (ValueError, TypeError):
                    li = pos
                if 0 <= li < len(qperm):
                    j = js[qperm[li]]
                    cell, _, _ = A.parse_compact(o)
                    if cell is not None and "_raw" not in cell:
                        rep[str(j)] = {"k": cell.get("knowledge"), "s": cell.get("stars")}
            user_out["repeats"][str(seed)] = rep
            print(f"  u{u}[{lab}] seed{seed}: {len(rep)}/{len(js)} parsed | cum ${G._USAGE['usd']:.4f}",
                  flush=True)

        results["users"][uid] = user_out
        results["model"] = G._USAGE["snapshot"]

    results["tokstat"] = tokstat
    results["usd"] = G._USAGE["usd"]
    results["calls"] = G._USAGE["calls"]
    os.makedirs("experiments", exist_ok=True)
    json.dump(results, open(OUT_JSON, "w"), indent=1)
    json.dump({"usd": G._USAGE["usd"], "calls": G._USAGE["calls"],
               "prompt_tokens": G._USAGE["prompt_tokens"],
               "completion_tokens": G._USAGE["completion_tokens"],
               "snapshot": G._USAGE["snapshot"]}, open(USAGE_JSON, "w"), indent=1)
    print(f"[collect] done; {G._USAGE['calls']} calls, ${G._USAGE['usd']:.4f} -> {OUT_JSON}", flush=True)


def _kappa(a, b, cats):
    """Cohen's kappa on paired label lists a,b over label set cats."""
    n = len(a)
    if n == 0:
        return float("nan")
    idx = {c: i for i, c in enumerate(cats)}
    C = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        C[idx[x], idx[y]] += 1
    po = np.trace(C) / n
    r = C.sum(1) / n
    c = C.sum(0) / n
    pe = float((r * c).sum())
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def analyze():
    R = json.load(open(OUT_JSON))
    seeds = [str(s) for s in R["repeat_seeds"]]
    L = []
    P = L.append
    P("# 3-USER SHUFFLE REPEATABILITY PROBE")
    P("")
    P(f"Judge `{R['model']}` temp {R['temp']}, split seed {R['split_seed']}. Same user, same "
      f"{R['n_items']} item questions re-sent {len(seeds)} times with shuffled question order AND "
      f"shuffled profile-line order (seeds {R['repeat_seeds']}); content identical, presentation "
      f"differs (no caching). Reuses scripts/answerer_smallbatch.py judge machinery.")
    P("")
    P(f"Cost: {R['calls']} calls, {R['tokstat']['prompt']} prompt + {R['tokstat']['completion']} "
      f"completion tokens, **${R['usd']:.4f}**.")
    P("")

    summary = {}
    for uid, U in R["users"].items():
        lab = U["label"]
        js = [str(j) for j in U["items_j"]]
        orig = U["orig"]
        reps = U["repeats"]
        # (b) know-well RATE per repeat (+ original)
        def kw_rate(d):
            ks = [d[j]["k"] for j in js if j in d]
            return (sum(1 for k in ks if k == "know_well") / len(ks)) if ks else float("nan"), len(ks)
        orig_rate, orig_n = kw_rate(orig)
        rep_rates = {s: kw_rate(reps[s]) for s in seeds}
        rate_vals = [rep_rates[s][0] for s in seeds]
        rate_spread = max(rate_vals) - min(rate_vals)
        rate_vs_orig = max(abs(rv - orig_rate) for rv in rate_vals)

        # (a) cell-level knowledge-label agreement: all pairwise among the 3 repeats + vs original
        def label_agree(d1, d2):
            common = [j for j in js if j in d1 and j in d2]
            a = [d1[j]["k"] for j in common]
            b = [d2[j]["k"] for j in common]
            ident = np.mean([x == y for x, y in zip(a, b)]) if common else float("nan")
            kap = _kappa(a, b, KNOW) if common else float("nan")
            return float(ident), float(kap), len(common)
        pair_ident, pair_kap = [], []
        for i in range(len(seeds)):
            for k in range(i + 1, len(seeds)):
                ident, kap, _ = label_agree(reps[seeds[i]], reps[seeds[k]])
                pair_ident.append(ident); pair_kap.append(kap)
        vs_orig_ident, vs_orig_kap = [], []
        for s in seeds:
            ident, kap, _ = label_agree(orig, reps[s])
            vs_orig_ident.append(ident); vs_orig_kap.append(kap)

        # (c) stars MAE on cells BOTH know-well, across repeat pairs
        def stars_mae(d1, d2):
            errs = []
            for j in js:
                if j in d1 and j in d2 and d1[j]["k"] == "know_well" and d2[j]["k"] == "know_well" \
                        and d1[j]["s"] is not None and d2[j]["s"] is not None:
                    errs.append(abs(float(d1[j]["s"]) - float(d2[j]["s"])))
            return (float(np.mean(errs)), len(errs)) if errs else (float("nan"), 0)
        pair_mae = []
        for i in range(len(seeds)):
            for k in range(i + 1, len(seeds)):
                m, _ = stars_mae(reps[seeds[i]], reps[seeds[k]])
                pair_mae.append(m)
        orig_mae = [stars_mae(orig, reps[s])[0] for s in seeds]

        mean_pair_ident = float(np.nanmean(pair_ident))
        mean_pair_kap = float(np.nanmean(pair_kap))
        mean_vs_orig_ident = float(np.nanmean(vs_orig_ident))

        summary[uid] = dict(label=lab, orig_rate=orig_rate, rep_rates=rate_vals,
                            rate_spread=rate_spread, rate_vs_orig=rate_vs_orig,
                            pair_ident=mean_pair_ident, pair_kap=mean_pair_kap,
                            vs_orig_ident=mean_vs_orig_ident, pair_mae=float(np.nanmean(pair_mae)),
                            orig_mae=float(np.nanmean(orig_mae)), n=orig_n)

        P(f"## User {uid} ({lab}; orig know-well rate {U['orig_kw_rate']:.3f}, n_items {orig_n}, "
          f"profile {U['n_shown']}/{U['N_total']}{' CAPPED' if U['profile_capped'] else ''})")
        P("")
        P(f"- **(b) know-well RATE**: original {orig_rate:.3f} | repeats "
          f"{', '.join(f'{s}={rep_rates[s][0]:.3f}' for s in seeds)} "
          f"-> **spread {rate_spread:.3f}**, max|rep-orig| {rate_vs_orig:.3f}")
        P(f"- **(a) cell knowledge-label agreement**: repeat-vs-repeat identical "
          f"{mean_pair_ident:.3f} (kappa {mean_pair_kap:.3f}); vs original identical "
          f"{mean_vs_orig_ident:.3f} (kappa {float(np.nanmean(vs_orig_kap)):.3f})")
        P(f"- **(c) stars MAE on know-well cells**: repeat-vs-repeat {float(np.nanmean(pair_mae)):.3f} "
          f"| vs original {float(np.nanmean(orig_mae)):.3f} stars")
        P("")

    # (d) verdict
    P("## (d) Verdict")
    P("")
    P("| user | tier | rate spread | max|rep-orig| | rep-vs-rep cell agree | vs-orig cell agree | "
      "rep MAE | verdict |")
    P("|---|---|--:|--:|--:|--:|--:|---|")
    big_any = False
    for uid, s in summary.items():
        small = (s["rate_spread"] <= 0.03) and (s["pair_ident"] > 0.85)
        big_any = big_any or (not small)
        P(f"| {uid} | {s['label']} | {s['rate_spread']:.3f} | {s['rate_vs_orig']:.3f} | "
          f"{s['pair_ident']:.3f} | {s['vs_orig_ident']:.3f} | {s['pair_mae']:.3f} | "
          f"{'SMALL' if small else 'BIG'} |")
    P("")
    verdict = "BIG flutter (a shuffle-order effect is visible)" if big_any else \
        "SMALL flutter (rates within +/-0.03 AND cell agreement >85% on every user)"
    P(f"**Overall: {verdict}.**")
    P("")
    P("Criterion: SMALL = rate spread <= 0.03 AND repeat-vs-repeat cell knowledge agreement > 0.85 "
      "for EVERY user; otherwise BIG.")

    os.makedirs("experiments", exist_ok=True)
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L))
    json.dump(summary, open("experiments/SHUFFLE_PROBE_summary.json", "w"), indent=1)
    print(f"[saved] {OUT_MD}", flush=True)
    print("\n==== SUMMARY ====")
    for uid, s in summary.items():
        print(f"u{uid}[{s['label']}]: rate_spread {s['rate_spread']:.3f} "
              f"(reps {[round(x,3) for x in s['rep_rates']]}), cell-agree {s['pair_ident']:.3f}, "
              f"kappa {s['pair_kap']:.3f}, MAE {s['pair_mae']:.3f}")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.collect:
        collect()
    elif a.analyze:
        analyze()
    else:
        print("pass --collect or --analyze")
