"""CROSS-CHECK B (dual-model robustness): cross-family LLM judge agreement on ANSWERABILITY.

Re-judge ~30 gate users with a Claude model (Anthropic) using the EXACT SAME prompt + question
bank the pinned GPT-5.4-mini judge saw (replayed from the committed grid cache). Report:
  (1) per-question ANSWERABILITY agreement rate GPT-5.4-mini vs Haiku (yes-vs-not, maybe->refuse
      primary, matching the gate), on interview questions (concept+item), + Cohen's kappa.
  (2) whether the VALIDITY GAP sign replicates under Haiku: judged answer-rate on held-out-RATED
      minus matched never-rated in low/mid pop tiers (the Hole-2 signal must stay POSITIVE).

PASS = agreement materially above chance (>=70%) AND validity-gap sign replicates (rated > never
in low/mid tiers). This checks the fuel isn't an artifact of one model family.

Model PINNED: claude-haiku-4-5 (resolved snapshot recorded from response.model). temperature=0.
Judgments CACHED + resumable (never re-calls a cached user). Usage + running $ logged.
"""
import os, sys, json, time, argparse, collections
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import anthropic

# reuse the EXACT gate machinery so the prompt is byte-identical to what GPT saw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("OPENAI_API_KEY", "unused")  # gate module builds an OpenAI client at import
import llm_answerability_gate as G

HAIKU_MODEL = os.environ.get("HAIKU_MODEL", "claude-haiku-4-5")
N_USERS = 30
SAMPLE_SEED = 0
# Haiku 4.5 documented rates ($/1M): in $1.00, out $5.00
PRICE_IN, PRICE_OUT = 1.00, 5.00
HAIKU_CACHE = ".cache/instrument2/answerability_haiku_grid.json"
OUT_JSON = "experiments/answerability_crosscheck_haiku.json"

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
_USAGE = {"calls": 0, "in": 0, "out": 0, "usd": 0.0, "snapshot": None}


def haiku_json(sys_msg, user_msg, max_retry=2):
    last = ""
    for attempt in range(max_retry + 1):
        try:
            resp = client.messages.create(
                model=HAIKU_MODEL, max_tokens=8000, temperature=0.0,
                system=sys_msg + "\nReturn ONLY the JSON object, no prose, no code fences.",
                messages=[{"role": "user", "content": user_msg}])
            u = resp.usage
            _USAGE["calls"] += 1; _USAGE["in"] += u.input_tokens; _USAGE["out"] += u.output_tokens
            _USAGE["usd"] += u.input_tokens / 1e6 * PRICE_IN + u.output_tokens / 1e6 * PRICE_OUT
            if _USAGE["snapshot"] is None:
                _USAGE["snapshot"] = resp.model
            txt = "".join(b.text for b in resp.content if b.type == "text")
            last = txt
            s = txt.find("{"); e = txt.rfind("}")
            obj = json.loads(txt[s:e + 1] if s >= 0 and e > s else txt)
            arr = obj.get("answers")
            if arr is None:
                for v in obj.values():
                    if isinstance(v, list):
                        arr = v; break
            if isinstance(arr, list):
                return arr, resp.model
        except Exception as ex:
            print(f"    [retry {attempt}] {type(ex).__name__}: {str(ex)[:160]}", flush=True)
            time.sleep(1.5)
    print(f"    PARSE/CALL FAIL: {last[:200]}", flush=True)
    return [], _USAGE["snapshot"]


def reconstruct_user_msg(D, split, u, Qcached):
    """Rebuild the exact answerability user message GPT saw for user u (from cached Q + profile)."""
    kn, ho = split[u]
    rat = dict(D["rat_by_u"][u])
    known_ids = sorted(kn)
    prof_lines = G.profile_text(D, known_ids, rat)
    n_shown, N_total = len(prof_lines), len(dict(D["rat_by_u"][u]))
    lines = []
    for i, (kind, meta) in enumerate(Qcached):
        if kind == "concept":
            lines.append(f"{i}: [concept] Could this viewer state a preference about "
                         f"\"{meta['name']}\" films?")
        else:
            lines.append(f"{i}: [item] Has this viewer seen / could they give a real opinion on "
                         f"'{D['title'][meta['j']]}'?")
    prof = "; ".join(prof_lines) if prof_lines else "(no ratings shown)"
    return (f"Viewer's rating sample (size {n_shown} of {N_total} total; they have seen many more "
            f"films than listed):\n{prof}\n\nQuestions:\n" + "\n".join(lines))


def is_yes(o):
    return isinstance(o, dict) and o.get("can_answer") == "yes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_users", type=int, default=N_USERS)
    args = ap.parse_args()

    D = G.load_data()
    split = G.build_split(D)
    grid = json.load(open(G.GRID_CACHE))["users"]
    gate_users = sorted(int(u) for u in grid)
    rng = np.random.default_rng(SAMPLE_SEED)
    users = sorted(rng.choice(gate_users, size=min(args.n_users, len(gate_users)), replace=False).tolist())
    print(f"[haiku] model={HAIKU_MODEL} users={len(users)} (seed {SAMPLE_SEED})", flush=True)

    hcache = {}
    if os.path.exists(HAIKU_CACHE):
        blob = json.load(open(HAIKU_CACHE))
        if blob.get("model") == HAIKU_MODEL:
            hcache = blob.get("users", {})
            print(f"[cache] resuming {len(hcache)} cached Haiku users", flush=True)

    for n, u in enumerate(users):
        Qcached = [(k, m) for k, m in grid[str(u)]["Q"]]
        gpt_ans = {int(k): v for k, v in grid[str(u)]["ans"].items()}
        if str(u) in hcache:
            continue
        user_msg = reconstruct_user_msg(D, split, u, Qcached)
        arr, snap = haiku_json(G.SYS_ANSWER, user_msg)
        hans = {}
        for o in arr:
            if isinstance(o, dict) and "q" in o:
                try:
                    hans[int(o["q"])] = o
                except (ValueError, TypeError):
                    pass
        hcache[str(u)] = {"ans": {str(i): o for i, o in hans.items()}, "snapshot": snap}
        iv = [i for i, (k, m) in enumerate(Qcached) if k in ("concept", "item")]
        agree = sum(1 for i in iv if i in hans and is_yes(hans[i]) == is_yes(gpt_ans.get(i, {})))
        parsed = sum(1 for i in iv if i in hans)
        print(f"  [{n+1}/{len(users)}] user {u}: parsed {len(hans)}/{len(Qcached)} | "
              f"interview agree {agree}/{parsed} | running ${_USAGE['usd']:.4f}", flush=True)
        json.dump({"model": HAIKU_MODEL, "snapshot": _USAGE["snapshot"], "users": hcache,
                   "usage": {"calls": _USAGE["calls"], "in_tokens": _USAGE["in"],
                             "out_tokens": _USAGE["out"], "usd": round(_USAGE["usd"], 4)}},
                  open(HAIKU_CACHE, "w"))

    # ---- metrics: agreement + kappa on interview; validity-gap sign under Haiku ----
    gpt_lab, hk_lab = [], []       # binary yes/not on interview questions both parsed
    tiers = ["famous", "moderate", "obscure"]
    agg = {t: dict(r_yes=0, r_n=0, n_yes=0, n_n=0) for t in tiers}      # Haiku validity
    gpt_agg = {t: dict(r_yes=0, r_n=0, n_yes=0, n_n=0) for t in tiers}  # GPT validity (same users)
    per_user_agree = []
    for u in users:
        Qcached = [(k, m) for k, m in grid[str(u)]["Q"]]
        gpt_ans = {int(k): v for k, v in grid[str(u)]["ans"].items()}
        hans = {int(k): v for k, v in hcache[str(u)]["ans"].items()}
        iv = [i for i, (k, m) in enumerate(Qcached) if k in ("concept", "item")]
        ok = tot = 0
        for i in iv:
            if i in hans and i in gpt_ans:
                a, b = is_yes(gpt_ans[i]), is_yes(hans[i])
                gpt_lab.append(int(a)); hk_lab.append(int(b))
                ok += (a == b); tot += 1
        if tot:
            per_user_agree.append(ok / tot)
        for i, (k, m) in enumerate(Qcached):
            if k not in ("valid_rated", "valid_never"):
                continue
            t = m["tier"]
            if t not in tiers:
                continue
            for lab, tab in ((hans, agg), (gpt_ans, gpt_agg)):
                if i in lab:
                    y = 1 if is_yes(lab[i]) else 0
                    if k == "valid_rated":
                        tab[t]["r_yes"] += y; tab[t]["r_n"] += 1
                    else:
                        tab[t]["n_yes"] += y; tab[t]["n_n"] += 1

    gpt_lab = np.array(gpt_lab); hk_lab = np.array(hk_lab)
    n = len(gpt_lab)
    agreement = float((gpt_lab == hk_lab).mean())
    po = agreement
    pe = float((gpt_lab.mean() * hk_lab.mean()) + ((1 - gpt_lab.mean()) * (1 - hk_lab.mean())))
    kappa = float((po - pe) / (1 - pe)) if (1 - pe) > 1e-9 else float("nan")

    def gaps(tab):
        out = {}
        for t in tiers:
            a = tab[t]
            rr = a["r_yes"] / a["r_n"] if a["r_n"] else None
            nr = a["n_yes"] / a["n_n"] if a["n_n"] else None
            out[t] = dict(rated_rate=rr, never_rate=nr,
                          gap_pts=((rr - nr) * 100 if rr is not None and nr is not None else None),
                          rated_n=a["r_n"], never_n=a["n_n"])
        return out
    haiku_validity = gaps(agg)
    gpt_validity = gaps(gpt_agg)
    # Diagnostic low/mid tiers use the gate's CANONICAL threshold rated_n>=5 (PLAN sec.B; the gate's
    # own validity test selects low/mid tiers with rated_n>=5). Tiers below that are non-diagnostic
    # (e.g. obscure here has n=4 and ties at 0.0 for BOTH GPT and Haiku -> shared small-n non-signal).
    DIAG_MIN_N = 5
    lowmid = [haiku_validity[t]["gap_pts"] for t in ("moderate", "obscure")
              if haiku_validity[t]["gap_pts"] is not None and haiku_validity[t]["rated_n"] >= DIAG_MIN_N]
    lowmid_nondiag = {t: dict(gap=haiku_validity[t]["gap_pts"], rated_n=haiku_validity[t]["rated_n"],
                              gpt_gap=gpt_validity[t]["gap_pts"])
                      for t in ("moderate", "obscure")
                      if haiku_validity[t]["gap_pts"] is not None and haiku_validity[t]["rated_n"] < DIAG_MIN_N}
    validity_sign_replicates = bool(lowmid and all(g > 0 for g in lowmid))

    # cost + snapshot: on an all-cached analyze rerun (_USAGE calls==0) read the persisted collection usage
    cache_blob = json.load(open(HAIKU_CACHE))
    cu = cache_blob.get("usage", {})
    snap = _USAGE["snapshot"] or cache_blob.get("snapshot") or \
        next((v.get("snapshot") for v in hcache.values() if v.get("snapshot")), None)
    if _USAGE["calls"] > 0:
        cost = dict(calls=_USAGE["calls"], in_tokens=_USAGE["in"], out_tokens=_USAGE["out"],
                    usd_estimate=round(_USAGE["usd"], 4), price_in=PRICE_IN, price_out=PRICE_OUT)
    else:
        cost = dict(calls=cu.get("calls"), in_tokens=cu.get("in_tokens"), out_tokens=cu.get("out_tokens"),
                    usd_estimate=cu.get("usd"), price_in=PRICE_IN, price_out=PRICE_OUT,
                    note="analyze rerun (0 new calls); cost = persisted collection usage")

    passed = bool(agreement >= 0.70 and validity_sign_replicates)
    result = dict(
        model=HAIKU_MODEL, snapshot=snap, temperature=0.0, n_users=len(users),
        n_interview_questions_compared=n,
        answerability_agreement_rate=agreement, cohens_kappa=kappa,
        per_user_agreement_mean=float(np.mean(per_user_agree)) if per_user_agree else None,
        gpt_yes_rate=float(gpt_lab.mean()), haiku_yes_rate=float(hk_lab.mean()),
        haiku_validity_gap=haiku_validity, gpt_validity_gap_same_users=gpt_validity,
        validity_low_mid_haiku_diagnostic=lowmid, validity_low_mid_nondiagnostic=lowmid_nondiag,
        validity_diag_min_rated_n=DIAG_MIN_N, validity_sign_replicates=validity_sign_replicates,
        cost=cost,
        cutoff="agreement>=0.70 AND validity-gap sign replicates (low/mid rated>never)",
        passed=passed)
    os.makedirs("experiments", exist_ok=True)
    json.dump(result, open(OUT_JSON, "w"), indent=1)
    print("\n================= CROSS-CHECK B (Haiku) =================", flush=True)
    print(f"snapshot={result['snapshot']}  n_qs={n} over {len(users)} users", flush=True)
    print(f"answerability agreement={agreement:.3f}  kappa={kappa:.3f}  "
          f"(GPT yes-rate {gpt_lab.mean():.3f}, Haiku yes-rate {hk_lab.mean():.3f})", flush=True)
    print(f"Haiku validity low/mid gaps: {lowmid} -> sign replicates: {validity_sign_replicates}", flush=True)
    print(f"Haiku validity per tier: " +
          "; ".join(f"{t}: {haiku_validity[t]['gap_pts']}" for t in tiers
                    if haiku_validity[t]['gap_pts'] is not None), flush=True)
    print(f"COST: {_USAGE['calls']} calls ${_USAGE['usd']:.4f}", flush=True)
    print(f"\n[VERDICT] {'PASS' if passed else 'FAIL'}  -> {OUT_JSON}", flush=True)
    return result


if __name__ == "__main__":
    main()
