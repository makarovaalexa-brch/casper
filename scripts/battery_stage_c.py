"""battery_stage_c.py -- STAGE C of the DIRECTIONAL adaptivity battery: PRIZE DECOMPOSITION.

Mini-arena on the NEW answerer-v1 environment through the I2.5 learned fold (.cache/i25_fold_best.pt).
E1 survivorship honored: all 173 users in EVERY arm/turn; refusal (k=0 / not answerable) = no-op turn
(belief unchanged); no-answer users sit at cold z=0. NDCG@10 on the untouched held-out halves.

E4 FOLD CANARY (gate BEFORE the arena): per channel, does ONE true answer from cold HELP (mean
single-answer NDCG lift > 0, CI excl 0)? Item = real/LLM stars centered; concept/attribute = LLM
4-level value centered x fold_weight (know_well=1, rough_idea=w_rough=0.5). If a channel fails the
canary, it is DROPPED from the arena (report the fold gap; do not improvise a new fold).

Arms (T=12): s-concept / s-item / s-mixed (fair greedy statics) = the static family; u-table
(PRIVILEGED: user's own answerability table, greedy value x answerable) = the ceiling; r-blind
(tie-by-construction LR-tilt router on s-mixed, E2: at t=0 identical to s-mixed; belief = kmap
population item posterior online). Paired per-user bootstrap.

NO LLM calls. Called via adaptivity_battery_v1.py --stage c.
"""
import os, sys, json, time, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import adaptivity_battery_v1 as A
import battery_stage_b as B
import i25_lib as L
import i25_phase4 as P4
import i25_phase4_fair as FA          # batched decode ndcg_batch (one matmul for all rows)
import llm_answerability_gate as G

CKPT = ".cache/i25_fold_best.pt"
EMB = ".cache/instrument2/kmap_emb.npz"
INT = ".cache/instrument2/kmap_intercepts.npz"
OUT_JSON = "experiments/adaptivity_battery_v1_C.json"
T = 12
POOL = 60                      # candidate pool per channel (top by coverage) for greedy
CANARY_SAMPLE = 15             # sampled answered cells per user per channel for the canary
BOOT = A.BOOT; SEED = A.SEED
KIDX = A.KIDX


def foldw(k):
    return A.FOLD_WEIGHT.get(k, 0.0)


# ---------------------------------------------------- build fold candidate universe
def build_cands(D, FR, users, memb, battery):
    """Per-user candidate resolution for concept/item/attr. Returns CANDS (shared meta list) + fills
    rec['ans']/[val]/[nat] cid arrays."""
    ni = int(D["ni"])

    def concept_emb_g(tagId):
        mems = [j for j in memb.get(str(tagId), []) if 0 <= j < ni]
        w = np.zeros(ni, np.float64)
        if mems:
            w[mems] = D["cnt"][mems] + 1.0
        return FR._bag_emb(w) if mems else None

    def attr_emb_g(eid):
        e = battery.get(eid)
        if not e:
            return None
        mems = [j for j in e.get("member_dense_ids", []) if 0 <= j < ni]
        w = np.zeros(ni, np.float64)
        if mems:
            w[mems] = D["cnt"][mems] + 1.0
        return FR._bag_emb(w) if mems else None

    # coverage over users (answerable count) to define shared pools
    cov = collections.Counter()
    for rec in users:
        for c in rec["concept"]:
            if c["k"] is not None and c["k"] >= 1:
                cov[("concept", c["tagId"])] += 1
        for c in rec["item_llm"]:
            if c["k"] is not None and c["k"] >= 1:
                cov[("item", c["j"])] += 1
        for c in rec["item_data"]:
            cov[("item", c["j"])] += 1
        for c in rec["attribute"]:
            if c["k"] is not None and c["k"] >= 1:
                cov[("attr", c["eid"])] += 1

    def top(kind, n):
        return [key for (k2, key), _ in cov.most_common() if k2 == kind][:n]

    keys = [("concept", k) for k in top("concept", POOL)] + \
           [("item", k) for k in top("item", POOL)] + \
           [("attr", k) for k in top("attr", POOL)]
    CANDS = []
    for kind, key in keys:
        if kind == "concept":
            emb = concept_emb_g(key); typ = 1
        elif kind == "item":
            emb = FR.Wn[key].numpy().astype(np.float32); typ = 0
        else:
            emb = attr_emb_g(key); typ = 2
        if emb is None:
            continue
        CANDS.append(dict(cid=len(CANDS), kind=kind, key=key, emb=np.asarray(emb, np.float32), typ=typ,
                          cov=cov[(kind, key)] / len(users)))
    ncand = len(CANDS)
    # per-user resolution
    cid_of = {(m["kind"], m["key"]): m["cid"] for m in CANDS}
    for rec in users:
        ans = np.zeros(ncand, bool); val = np.zeros(ncand); nat = [None] * ncand
        cm = rec["cmean"]
        for c in rec["concept"]:
            cd = cid_of.get(("concept", c["tagId"]))
            if cd is not None and c["k"] is not None and c["k"] >= 1:
                ans[cd] = True; val[cd] = A.cell_foldval(c["ans"])
        for c in rec["attribute"]:
            cd = cid_of.get(("attr", c["eid"]))
            if cd is not None and c["k"] is not None and c["k"] >= 1:
                ans[cd] = True; val[cd] = A.cell_foldval(c["ans"])
        for c in rec["item_llm"]:
            cd = cid_of.get(("item", c["j"]))
            if cd is not None and c["k"] is not None and c["k"] >= 1:
                stars = c["ans"].get("stars")
                if stars is not None:
                    ans[cd] = True; val[cd] = (stars - cm) * foldw(c["ans"].get("knowledge"))
                    if stars >= 4:
                        nat[cd] = c["j"]
        for c in rec["item_data"]:
            cd = cid_of.get(("item", c["j"]))
            if cd is not None:
                ans[cd] = True; val[cd] = (c["stars"] - cm)     # know_well weight 1
                if c["stars"] >= 4:
                    nat[cd] = c["j"]
        rec["ans_arr"] = ans; rec["val_arr"] = val; rec["nat_arr"] = nat
    return CANDS


def tok_of(CANDS, rec, cid):
    m = CANDS[cid]
    return (m["typ"], m["emb"], float(rec["val_arr"][cid]), f"{m['kind']}:{m['key']}")


# ---------------------------------------------------- fold eval (E1: all users, cold fallback)
def cold_ndcg(FR, model, users):
    Z = np.zeros((len(users), FR.W.shape[1]), np.float32)
    return FA.ndcg_batch(FR, Z, users, 10)


def eval_plan(FR, model, users, plan_fn, cold):
    """plan_fn(rec)->list of (cid or None) per turn (None = refusal, no-op). Returns per_turn (n,T)."""
    n = len(users)
    plans = [plan_fn(rec) for rec in users]
    answered = np.array([sum(1 for c in pl[:T] if c is not None) for pl in plans], float)
    per_turn = np.empty((n, T))
    for t in range(T):
        col = cold.copy(); tl, nl, idx = [], [], []
        for i, rec in enumerate(users):
            cids = [c for c in plans[i][:t + 1] if c is not None]
            if not cids:
                continue
            toks = [tok_of(CANDS_G, rec, c) for c in cids]
            nat = [rec["nat_arr"][c] for c in cids if rec["nat_arr"][c] is not None]
            tl.append(toks); nl.append(nat); idx.append(i)
        if tl:
            Z = P4.fold_batch(FR, model, tl, nl)
            col[idx] = FA.ndcg_batch(FR, Z, [users[i] for i in idx], 10)
        per_turn[:, t] = col
    return per_turn, answered


CHUNK = 3000


def _batch_ndcg(FR, model, users, tl, nl, idx, cold):
    """Fold rows (chunked) and return per-row NDCG aligned to idx (user index per row). Batched decode."""
    out = np.empty(len(tl))
    for s in range(0, len(tl), CHUNK):
        e = min(s + CHUNK, len(tl))
        Z = P4.fold_batch(FR, model, tl[s:e], nl[s:e])
        out[s:e] = FA.ndcg_batch(FR, Z, [users[idx[r]] for r in range(s, e)], 10)
    return out


def build_greedy(FR, model, users, pool_cids, cold, tag=""):
    """Greedy-forward cohort-endpoint NDCG (prefix-consistent). At each position ALL candidate-appended
    schedules are folded in ONE batched pass (C x N rows) for speed. Users with no answerable token in a
    schedule get cold (E1)."""
    pool = sorted(pool_cids, key=lambda c: -CANDS_G[c]["cov"])[:POOL]
    n = len(users)
    sched = []
    t0 = time.time()
    for pos in range(T):
        cands = [c for c in pool if c not in sched]
        if not cands:
            break
        tl, nl, idx, owner = [], [], [], []
        base_ans = [[c for c in sched if rec["ans_arr"][c]] for rec in users]
        for ci, c in enumerate(cands):
            for i, rec in enumerate(users):
                cids = base_ans[i] + ([c] if rec["ans_arr"][c] else [])
                if not cids:
                    continue
                tl.append([tok_of(CANDS_G, rec, cc) for cc in cids])
                nl.append([rec["nat_arr"][cc] for cc in cids if rec["nat_arr"][cc] is not None])
                idx.append(i); owner.append(ci)
        ndcg = _batch_ndcg(FR, model, users, tl, nl, idx, cold)
        sums = np.full(len(cands), 0.0)
        cnts = np.zeros(len(cands))
        got = collections.defaultdict(set)
        for r, ci in enumerate(owner):
            sums[ci] += ndcg[r]; got[ci].add(idx[r])
        best, bc = -1.0, None
        for ci, c in enumerate(cands):
            cold_miss = sum(cold[i] for i in range(n) if i not in got[ci])
            mean = (sums[ci] + cold_miss) / n
            if mean > best:
                best, bc = mean, c
        sched.append(bc)
        print(f"   [greedy {tag}] pos {pos + 1}/{T} -> {CANDS_G[bc]['kind']}:{CANDS_G[bc]['key']} "
              f"(cohort {best:.4f}) [{(time.time() - t0) / 60:.1f}m]", flush=True)
    return sched


def plan_static(sched):
    def f(rec):
        return [c if rec["ans_arr"][c] else None for c in sched][:T]
    return f


def plan_utable(FR, model, users, cold):
    """PRIVILEGED ceiling: per user, greedy value x answerable -- rank answerable cands by SIGNED fold
    value desc (prefer loved/high-value answers; positive values lift NDCG for liked targets)."""
    def f(rec):
        ans_c = [m["cid"] for m in CANDS_G if rec["ans_arr"][m["cid"]]]
        ans_c.sort(key=lambda c: -rec["val_arr"][c])
        return ans_c[:T]
    return f


U_POOL = 30
def eval_uclair(FR, model, users, cold):
    """PRIVILEGED true-NDCG clairvoyant ceiling: per user, each turn pick the answerable candidate that
    maximizes held-out NDCG@10 (u1-style). Pool capped to top-U_POOL answerable by signed value at t=0
    (privileged ceiling; the cap only makes it conservative). Batched decode per turn."""
    n = len(users); per_turn = np.empty((n, T)); answered = np.zeros(n)
    for i, rec in enumerate(users):
        ans_c = [m["cid"] for m in CANDS_G if rec["ans_arr"][m["cid"]]]
        ans_c.sort(key=lambda c: -rec["val_arr"][c])
        ans_c = ans_c[:U_POOL]
        chosen, chosen_nat, used, prev = [], [], set(), cold[i]
        for t in range(T):
            avail = [c for c in ans_c if c not in used]
            if not avail:
                per_turn[i, t] = prev; continue
            tl = [[tok_of(CANDS_G, rec, c) for c in chosen] + [tok_of(CANDS_G, rec, c)] for c in avail]
            nl = [list(chosen_nat) + ([rec["nat_arr"][c]] if rec["nat_arr"][c] is not None else [])
                  for c in avail]
            Z = P4.fold_batch(FR, model, tl, nl)
            vals = FA.ndcg_batch(FR, Z, [rec] * len(avail), 10)
            bi = int(np.argmax(vals)); bc = avail[bi]
            used.add(bc); chosen.append(bc)
            if rec["nat_arr"][bc] is not None:
                chosen_nat.append(rec["nat_arr"][bc])
            prev = float(vals[bi]); per_turn[i, t] = prev; answered[i] += 1
    return per_turn, answered


# ---------------------------------------------------- r-blind tie-by-construction LR router
class KmapItem:
    def __init__(self, D):
        e = np.load(EMB); bf = np.load(INT)
        self.d = int(e["d"]); ni = int(D["ni"])
        self.krow = -np.ones(ni, np.int64); self.krow[e["item_ids"]] = np.arange(len(e["item_ids"]))
        Ek = e["E"].astype(np.float64); bk = bf["b"].astype(np.float64)
        pop_a = float(bf["pop_a"]); pop_c = float(bf["pop_c"]); logcnt = bf["logcnt"].astype(np.float64)
        self.Efull = np.zeros((ni, self.d)); kept = self.krow >= 0; self.Efull[kept] = Ek[self.krow[kept]]
        self.bfull = pop_a * logcnt + pop_c; self.bfull[kept] = bk[self.krow[kept]]

    def lr(self, j, alpha, k):
        p0 = B.sigmoid(self.bfull[j]); p1 = B.sigmoid(self.bfull[j] + alpha + self.Efull[j] @ k)
        return p1 / max(p0, 1e-9)


def plan_rblind(FR, model, users, s_mixed, KM):
    """Warm-started on s_mixed. V(cid)=1/rank_smixed; item LR from online kmap posterior (concept/attr
    LR=1). At t=0 posterior=0 -> LR=1 -> argmax V == s_mixed order (E2 tie-by-construction)."""
    rank = {c: r + 1 for r, c in enumerate(s_mixed)}
    # remaining cands (not in s_mixed) get ranks after, by coverage
    rest = sorted([m["cid"] for m in CANDS_G if m["cid"] not in rank], key=lambda c: -CANDS_G[c]["cov"])
    for r, c in enumerate(rest):
        rank[c] = len(s_mixed) + 1 + r
    Vd = {c: 1.0 / rank[c] for c in rank}

    def f(rec):
        used = set(); plan = []
        j_ev, y_ev = [], []                       # item events for kmap online infer
        for t in range(T):
            if j_ev:
                Emat = KM.Efull[j_ev]; bvec = KM.bfull[j_ev]
                alpha, k = B.infer_user(Emat, bvec, np.array(y_ev, float), KM.d)
            else:
                alpha, k = 0.0, np.zeros(KM.d)
            best, bc = -1e18, None
            for m in CANDS_G:
                c = m["cid"]
                if c in used:
                    continue
                lr = KM.lr(m["key"], alpha, k) if m["kind"] == "item" else 1.0
                score = Vd[c] * lr
                if score > best or (score == best and (bc is None or rank[c] < rank[bc])):
                    best, bc = score, c
            if bc is None:
                break
            used.add(bc); m = CANDS_G[bc]
            if rec["ans_arr"][bc]:
                plan.append(bc)
            else:
                plan.append(None)
            if m["kind"] == "item":                # record the item probe outcome (answered=1)
                j_ev.append(m["key"]); y_ev.append(1 if rec["ans_arr"][bc] else 0)
        return plan
    return f


def paired(a, b, seed=SEED):
    d = np.asarray(a) - np.asarray(b); rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)])
    return dict(delta=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n=int(len(d)))


CANDS_G = None


def stage_c():
    global CANDS_G
    t0 = time.time()
    D, split, grid, users, memb, battery = A.load_env()
    print("\n==== STAGE C -- PRIZE DECOMPOSITION (mini-arena; DIRECTIONAL 173/300) ====", flush=True)
    print("PRE-REGISTERED THRESHOLDS (printed before results):", flush=True)
    print("  E4 canary: per channel, mean single-answer NDCG lift from cold > 0 with CI excl 0 "
          "(else DROP the channel).", flush=True)
    print("  C3(i) prize = u-table - best static: CI excl 0 AND >= 0.01 at any budget => prize EXISTS.", flush=True)
    print("  C3(ii) r-blind vs best static: positive sep CI excl 0 = first blind win; tie = discovery "
          "binding; MUST NOT lose beyond noise (E2).", flush=True)

    FR = L.Frozen(D)
    model = L.Fold(); blob = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(blob["model"]); model.eval()
    P4.T = T
    CANDS_G = build_cands(D, FR, users, memb, battery)
    cold = cold_ndcg(FR, model, users)
    print(f"[C] {len(users)} users; {len(CANDS_G)} candidates "
          f"({sum(m['kind']=='concept' for m in CANDS_G)}c/{sum(m['kind']=='item' for m in CANDS_G)}i/"
          f"{sum(m['kind']=='attr' for m in CANDS_G)}a); cold NDCG {cold.mean():.4f}; fold val "
          f"{blob['state']['best_val']:.4f}", flush=True)

    # ---------------- E4 CANARY ----------------
    canary = {}
    rng = np.random.default_rng(SEED)
    for kind in ("concept", "item", "attr"):
        cids_kind = [m["cid"] for m in CANDS_G if m["kind"] == kind]
        deltas = []
        tl, nl, meta = [], [], []
        for i, rec in enumerate(users):
            avail = [c for c in cids_kind if rec["ans_arr"][c]]
            if not avail:
                continue
            samp = list(rng.permutation(avail))[:CANARY_SAMPLE]
            for c in samp:
                tl.append([tok_of(CANDS_G, rec, c)])
                nl.append([rec["nat_arr"][c]] if rec["nat_arr"][c] is not None else [])
                meta.append(i)
        if tl:
            vals = _batch_ndcg(FR, model, users, tl, nl, meta, cold)
            per_user = collections.defaultdict(list)
            for r, i in enumerate(meta):
                per_user[i].append(vals[r] - cold[i])
            deltas = [float(np.mean(v)) for v in per_user.values()]
        cb = paired(deltas, [0.0] * len(deltas)) if deltas else dict(delta=float("nan"),
                                                                     ci=[float("nan")] * 2, n=0)
        canary[kind] = cb
        print(f"  [E4 canary {kind:8s}] mean single-answer lift {cb['delta']:+.4f}"
              f"[{cb['ci'][0]:+.4f},{cb['ci'][1]:+.4f}] (n_users={cb['n']}) -> "
              f"{'PASS' if cb['ci'][0] > 0 else 'FAIL (DROP)'}", flush=True)
    passing = [k for k in canary if canary[k]["ci"][0] > 0]
    print(f"  channels passing canary -> {passing}", flush=True)

    # ---------------- mini-arena (only passing channels) ----------------
    conc_pool = [m["cid"] for m in CANDS_G if m["kind"] == "concept" and "concept" in passing]
    item_pool = [m["cid"] for m in CANDS_G if m["kind"] == "item" and "item" in passing]
    mixed_pool = [m["cid"] for m in CANDS_G if m["kind"] in passing]

    print("[C] greedy s-concept ...", flush=True)
    s_concept = build_greedy(FR, model, users, conc_pool, cold, "s-concept") if conc_pool else []
    print("[C] greedy s-item ...", flush=True)
    s_item = build_greedy(FR, model, users, item_pool, cold, "s-item") if item_pool else []
    print("[C] greedy s-mixed ...", flush=True)
    s_mixed = build_greedy(FR, model, users, mixed_pool, cold, "s-mixed")

    KM = KmapItem(D)
    arms = {}
    if s_concept:
        arms["s-concept"] = eval_plan(FR, model, users, plan_static(s_concept), cold)
    if s_item:
        arms["s-item"] = eval_plan(FR, model, users, plan_static(s_item), cold)
    arms["s-mixed"] = eval_plan(FR, model, users, plan_static(s_mixed), cold)
    arms["u-table (PRIV)"] = eval_plan(FR, model, users, plan_utable(FR, model, users, cold), cold)
    print("[C] u-clair (NDCG-greedy privileged ceiling) ...", flush=True)
    arms["u-clair (PRIV)"] = eval_uclair(FR, model, users, cold)
    arms["r-blind"] = eval_plan(FR, model, users, plan_rblind(FR, model, users, s_mixed, KM), cold)

    def anyt(pt):
        return pt[:, :T].mean(axis=1)

    def endp(pt):
        return pt[:, T - 1]

    rows = {}
    for name, (pt, ans) in arms.items():
        rows[name] = dict(anytime=float(anyt(pt).mean()), endpoint=float(endp(pt).mean()),
                          mean_ans=float(ans.mean()),
                          curve=[float(x) for x in pt.mean(axis=0)])
    static_names = [n for n in ("s-concept", "s-item", "s-mixed") if n in arms]
    best_static = max(static_names, key=lambda n: rows[n]["anytime"])
    print("\n  ARMS (anytime/endpoint NDCG@10, mean answered turns):", flush=True)
    for name in arms:
        r = rows[name]
        print(f"    {name:16s} any {r['anytime']:.4f} end {r['endpoint']:.4f} ans {r['mean_ans']:.1f}",
              flush=True)
    print(f"  best static = {best_static}", flush=True)

    # verdicts -- prize uses the STRONGER privileged ceiling (NDCG-greedy clairvoyant), with the
    # value-greedy u-table reported alongside.
    bs_pt = arms[best_static][0]
    prize_utable = paired(anyt(arms["u-table (PRIV)"][0]), anyt(bs_pt))
    prize_uclair = paired(anyt(arms["u-clair (PRIV)"][0]), anyt(bs_pt))
    prize = prize_uclair if prize_uclair["delta"] >= prize_utable["delta"] else prize_utable
    prize_exists = bool(prize["ci"][0] > 0 and prize["delta"] >= 0.01)
    rblind_vs = paired(anyt(arms["r-blind"][0]), anyt(bs_pt))
    rblind_win = bool(rblind_vs["ci"][0] > 0)
    rblind_lose = bool(rblind_vs["ci"][1] < 0)
    # tie-by-construction floor check (E2): the CHOSEN cand at t=0 equals s_mixed[0] for all users
    rb_choice0 = [plan_rblind_choice0(FR, model, users, s_mixed, KM, rec) for rec in users]
    floor_ok = all(c == s_mixed[0] for c in rb_choice0)
    print(f"\n  C3(i) prize: u-clair(PRIV)-{best_static} {prize_uclair['delta']:+.4f}"
          f"[{prize_uclair['ci'][0]:+.4f},{prize_uclair['ci'][1]:+.4f}] ; "
          f"u-table(PRIV)-{best_static} {prize_utable['delta']:+.4f}"
          f"[{prize_utable['ci'][0]:+.4f},{prize_utable['ci'][1]:+.4f}] -> prize exists={prize_exists}",
          flush=True)
    print(f"  C3(ii) r-blind vs {best_static} any {rblind_vs['delta']:+.4f}"
          f"[{rblind_vs['ci'][0]:+.4f},{rblind_vs['ci'][1]:+.4f}] -> win={rblind_win} lose={rblind_lose} "
          f"| tie-by-construction floor (turn-1 == s-mixed[0]) = {floor_ok}", flush=True)

    out = dict(banner="DIRECTIONAL 173/300, grid unfrozen", n_users=len(users),
               n_cands=len(CANDS_G), cold=float(cold.mean()), fold_val=blob["state"]["best_val"],
               canary=canary, passing_channels=passing,
               schedules=dict(s_concept=[f"{CANDS_G[c]['kind']}:{CANDS_G[c]['key']}" for c in s_concept],
                              s_item=[f"{CANDS_G[c]['kind']}:{CANDS_G[c]['key']}" for c in s_item],
                              s_mixed=[f"{CANDS_G[c]['kind']}:{CANDS_G[c]['key']}" for c in s_mixed]),
               arms=rows, best_static=best_static,
               prize=dict(vs=best_static, boot=prize, exists=prize_exists,
                          u_clair=prize_uclair, u_table=prize_utable),
               rblind=dict(vs=best_static, boot=rblind_vs, win=rblind_win, lose=rblind_lose,
                           tie_floor_ok=bool(floor_ok)),
               wall_min=round((time.time() - t0) / 60, 2))
    json.dump(out, open(OUT_JSON, "w"), indent=1, default=str)
    write_md(out, rows, arms, static_names)
    print(f"\n[stage C] wrote {A.OUT_MD} (Stage C) + {OUT_JSON}  (wall {out['wall_min']}m)", flush=True)
    return out


def plan_rblind_choice0(FR, model, users, s_mixed, KM, rec):
    """The t=0 chosen candidate (for the tie-by-construction floor check)."""
    rank = {c: r + 1 for r, c in enumerate(s_mixed)}
    rest = sorted([m["cid"] for m in CANDS_G if m["cid"] not in rank], key=lambda c: -CANDS_G[c]["cov"])
    for r, c in enumerate(rest):
        rank[c] = len(s_mixed) + 1 + r
    Vd = {c: 1.0 / rank[c] for c in rank}
    best, bc = -1e18, None
    for m in CANDS_G:                              # t=0: alpha=k=0 -> LR=1 for all
        c = m["cid"]; score = Vd[c] * 1.0
        if score > best:
            best, bc = score, c
    return bc


def write_md(out, rows, arms, static_names):
    A.md_write("## STAGE C -- PRIZE DECOMPOSITION (mini-arena)\n\n"
               f"T={T} turns; question universe = the judged grid (channels passing the E4 canary); "
               f"NDCG@10 on the untouched held-out halves; ALL {out['n_users']} users every arm "
               f"(E1: refusal=no-op turn, no-answer users at cold z=0 = {out['cold']:.4f}); I2.5 fold "
               f"(val {out['fold_val']:.4f}). Paired per-user bootstrap.\n\n")
    A.md_write("### E4 fold canary (gate)\n\n"
               "| channel | mean single-answer lift from cold [95% CI] | n users | verdict |\n|---|---|--:|---|\n")
    for k in ("concept", "item", "attr"):
        cb = out["canary"][k]
        A.md_write(f"| {k} | {cb['delta']:+.4f}[{cb['ci'][0]:+.4f},{cb['ci'][1]:+.4f}] | {cb['n']} | "
                   f"{'PASS' if cb['ci'][0] > 0 else 'FAIL -> DROPPED'} |\n")
    A.md_write(f"\nChannels entering the arena: {out['passing_channels']}.\n\n")
    A.md_write("### Arena arms (NDCG@10)\n\n"
               "| arm | anytime | endpoint | mean answered turns | class |\n|---|--:|--:|--:|---|\n")
    cls = {"s-concept": "static", "s-item": "static", "s-mixed": "static (baseline)",
           "u-table (PRIV)": "PRIVILEGED value-greedy ceiling", "u-clair (PRIV)": "PRIVILEGED NDCG-greedy ceiling",
           "r-blind": "blind router (tie-by-construction)"}
    for name in ("s-concept", "s-item", "s-mixed", "u-table (PRIV)", "u-clair (PRIV)", "r-blind"):
        if name not in rows:
            continue
        r = rows[name]
        A.md_write(f"| {name} | {r['anytime']:.4f} | {r['endpoint']:.4f} | {r['mean_ans']:.1f} | "
                   f"{cls[name]} |\n")
    p = out["prize"]["boot"]; rb = out["rblind"]["boot"]
    uc = out["prize"]["u_clair"]; ut = out["prize"]["u_table"]
    A.md_write(f"\nBest static opponent = **{out['best_static']}**.\n\n"
               "### C3 pre-registered directional reads\n\n"
               f"- **(i) prize** (best privileged ceiling - {out['best_static']}, anytime): "
               f"u-clair (NDCG-greedy) = **{uc['delta']:+.4f}**[{uc['ci'][0]:+.4f},{uc['ci'][1]:+.4f}]; "
               f"u-table (value-greedy) = {ut['delta']:+.4f}[{ut['ci'][0]:+.4f},{ut['ci'][1]:+.4f}] "
               f"-> prize EXISTS (CI excl 0 AND >=0.01): **{out['prize']['exists']}**.\n"
               f"- **(ii) r-blind** vs {out['best_static']} (anytime) = "
               f"**{rb['delta']:+.4f}**[{rb['ci'][0]:+.4f},{rb['ci'][1]:+.4f}] -> "
               f"blind win (CI excl 0, +): **{out['rblind']['win']}**; loses beyond noise: "
               f"**{out['rblind']['lose']}**. Tie-by-construction floor (turn-1 == s-mixed[0], E2): "
               f"**{out['rblind']['tie_floor_ok']}**.\n\n"
               f"Interpretation: prize {'EXISTS' if out['prize']['exists'] else 'absent'}; blind router "
               f"{'WINS (first honest blind win)' if out['rblind']['win'] else ('LOSES beyond noise (E2 violation -- diagnose)' if out['rblind']['lose'] else 'TIES the static (discovery still binding)')}.\n\n")
    A.md_write("### DIRECTIONAL caveats\n\n"
               "- 173/300 users, grid UNFROZEN; re-run on the frozen grid before citation.\n"
               "- Attribute embeddings built from member-item bags (director/actor/etc); concept "
               "embeddings from genome membership bags -- both via the frozen RecVAE encoder, not the "
               "curated 200-concept fold interface (documented construction).\n"
               "- r-blind LR tilt uses the kmap POPULATION item posterior (validated, ready infer); "
               "concept/attr candidates carry LR=1 (no tilt). Stage B reports whether the LOUO-MF belief "
               "is stronger -- if so, a follow-up router should use it.\n\n")
