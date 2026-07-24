r"""tradeoff_ledger.py -- THE concept-escalation TRADEOFF LEDGER (author directives 2026-07-24):
one table, same harness, same COLD_SEED users, from which the author picks the operating point.

RUNGS (--rungs, comma list; a rung is skipped with a PENDING row if its artifact is missing):
  armA    frozen tower + untrained Arm-A additive shift (beta_cold=1 pure whitened; beta_ctx=5)
  fixab   frozen tower + the Fix-A/B winner (DIVERSIFIED selection |cos|<0.5 + additive beta=1;
          per experiments/battery/concepts_only_fixAB.json)
  clite   frozen tower + trained ConceptFoldNet (--clite_ckpt)
  cfull   concept tokens trained INTO the tower (--cfull_ckpt, train_tower_t2 --concept_tokens)

SECTION 1 -- PER-ANSWER (the G0/cold axes; canonical-snap controls):
  item full+tail NDCG@10 full-profile (does the rung still tie the frozen tower?), coldk2/k8,
  concepts-only m=1/2/4/8 full+tail, mixed m2k2, monotone-to-m8 verdict (the G5-E axis).

SECTION 2 -- DEPLOYMENT MODE (author main-bank hypothesis; FIRST-CLASS, not an appendix):
  per-QUESTION interview curves at budgets q=2,4,8,16 -- every question costs budget whether answered
  or not. Item-question answerability = the structural rule (the user RATED it; the pop>tau askability
  side is what makes the pop arm answerable-rich -- the strategy_ladder machinery that reproduced
  entropy's 0.03/8). Concept answerability = the concept structural rule (>= 2 rated members in the
  fold-in). Unanswered question = consumed budget, no fold. Interview arms per rung:
    (a) items-pop      items only, popularity-ranked   (answerable but Rashid-weak)
    (b) items-entropy  items only, entropy-ranked      (the honest item ceiling in deployment)
    (c) concepts-only  global member-mass order        (the main-bank candidate)
    (d) mixed          coarse-to-fine: first floor(q/2) concept questions, then popularity items
  Reported per arm x q: full+tail NDCG@10, mean answered count; plus the CROSSOVER budget where
  concepts-only overtakes each item arm (if it does).
CONTROLS: intercept row; canonical-snap of the per-answer numbers against the existing curve JSONs
(concepts_only_curve.json / concepts_only_fixAB.json); items coldk2/k8 snap (0.1965/0.2584 on the ep4
snapshot -- the expected values are read from the snapshot's own recorded curve when a different tower
is under test, i.e. C-full's item numbers are ITS OWN, that is the ledger's decision axis).

Usage:
  python src/instrument/tradeoff_ledger.py --smoke
  python src/instrument/tradeoff_ledger.py [--rungs armA,fixab,clite,cfull] [--clite_ckpt P]
        [--cfull_ckpt P] [--full_threads]
"""
import os
import sys
_FULL = "--full_threads" in sys.argv
_NT = str(os.cpu_count()) if _FULL else "4"
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)
os.environ.setdefault("MKL_NUM_THREADS", _NT)
import json
import time
import argparse
import numpy as np
import torch
from scipy import sparse

torch.set_num_threads(int(_NT))

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)

from train_tower_t2 import (level_to_sv, pack_tokens, NLEV, log, sel_value_to_level, I25Encoder,
                            truncate_graded)
import run_battery_phaseA as PA
from run_battery_phaseA import (build_real_ctx, build_smoke_ctx, eval_tokens, ndcg10_from_scores,
                                bootstrap_ci, load_genome, spearman, OUTDIR, SEED)
from belief_layer import build_concept_dirs, ConceptMean
from concepts_only_curve import sel_top_concepts, diversify_sel, K_SEED
from strategy_ladder import train_entropies

assert not hasattr(sys.modules[__name__], "load_answerer")

BUDGETS = (2, 4, 8, 16)
M_LIST = (1, 2, 4, 8)
COS_MAX = 0.5


# ============================================================================= shared setup
def build_shared(ctx):
    """Concept geometry + SEL answer simulator + deployment question banks (built ONCE per ledger)."""
    sh = {}
    ctx.Wd = ctx.decoder.weight.detach().float(); ctx.bd = ctx.decoder.bias.detach().float()
    ctx.d = ctx.enc.d_out
    members = load_genome(ctx)
    sh["tags"] = sorted(members.keys()); sh["members"] = members
    d_c, d_raw, w_c = build_concept_dirs(ctx.Wd, members, sh["tags"])
    sh["d_c"], sh["d_raw"], sh["w_c"] = d_c, d_raw, w_c
    # SEL lift matrix over the fold-in (answer VALUES; te excluded by construction)
    Mm = sparse.csr_matrix((np.ones(sum(len(members[t]) for t in sh["tags"]), np.float32),
                            (np.concatenate([members[t] for t in sh["tags"]]),
                             np.concatenate([np.full(len(members[t]), i)
                                             for i, t in enumerate(sh["tags"])]))),
                           shape=(ctx.ni, len(sh["tags"])))
    counts = np.asarray((ctx.va_tr @ Mm).todense())
    nu = np.asarray(ctx.va_tr.sum(axis=1)).ravel().clip(min=1)
    gmass = ctx.cnt @ np.asarray(Mm.todense())
    grate = gmass / max(ctx.cnt.sum(), 1e-9)
    lift = (counts / nu[:, None]) / np.maximum(grate[None, :], 1e-12)
    sh["conc_answerable"] = counts >= 2                      # structural rule: >=2 rated members
    top = np.where(lift > 1.0, lift, 1.0).max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.log(np.maximum(lift, 1.0)) / np.maximum(np.log(np.maximum(top, 1.0 + 1e-9)),
                                                          1e-9)[:, None]
    sh["conc_value"] = np.clip(np.nan_to_num(val, nan=0.25), 0.25, 1.0)   # SEL-graded in [0.25,1]
    sh["lift"] = lift
    # deployment banks
    sh["bank_conc"] = np.argsort(-gmass)                     # global member-mass order (public)
    sh["order_pop"] = np.argsort(-ctx.cnt)
    H, H0, n_raters = train_entropies(ctx)
    sh["order_entropy"] = np.argsort(-H)
    sh["lvl_lookup"] = [dict(zip(s.tolist(), l.tolist())) for s, l in ctx.allb]   # te excluded
    # per-answer axis selections: DIVERSIFIED (|cos|<0.5) is the DEFAULT for all rungs (author ruling
    # 2026-07-24, from the Fix-A verdict: the top-SEL m=8 crater is a correlation artifact); top-SEL is
    # kept as the armA comparator row so the artifact story stays visible in the table.
    sel40 = sel_top_concepts(ctx, members, sh["tags"], 40)                        # deep ranked pool
    sh["sel_top"] = {r: (cs[:max(M_LIST)], vs[:max(M_LIST)]) for r, (cs, vs) in sel40.items()}
    sh["sel_div"] = diversify_sel(sel40, d_c, m_max=max(M_LIST), cos_max=COS_MAX)
    return sh


def fold_items_enc(enc, seqs, batch=256):
    Z = torch.zeros(len(seqs), enc.d_out)
    enc.eval()
    with torch.no_grad():
        for st in range(0, len(seqs), batch):
            rows = [(np.asarray(s, np.int64), np.asarray(l, np.int64), level_to_sv(np.asarray(l)))
                    for s, l in seqs[st:st + batch]]
            ids, vals, pad, lvs = pack_tokens(rows)
            Z[st:st + len(rows)] = enc(ids, vals, pad, lvs)
    return Z


# ============================================================================= rung operators
class Rung:
    """A rung = (tower encoder used for item folds, concept application). score(items, concs) -> z."""

    def __init__(self, name, ctx, sh, clite_net=None, cfull_enc=None, div_select=False):
        self.name = name; self.ctx = ctx; self.sh = sh
        self.clite = clite_net; self.cfull = cfull_enc; self.div_select = div_select
        self.cmean = ConceptMean(sh["d_c"], sh["d_raw"], sh["w_c"], beta_ctx=5.0,
                                 beta_cold=1.0, floor_rho=0.0)

    def enc(self):
        return self.cfull if self.cfull is not None else self.ctx.enc

    def z_batch(self, item_seqs, conc_lists):
        """item_seqs: [(sids, lvls)]; conc_lists: [[(c, v)]]. Returns (B, d)."""
        if self.cfull is not None:                           # C-full: concepts are tower tokens
            seqs = []
            for (s, l), cl in zip(item_seqs, conc_lists):
                cid = np.asarray([self.ctx.ni + c for c, _ in cl], np.int64)
                clv = np.asarray([sel_value_to_level(v) for _, v in cl], np.int64)
                seqs.append((np.concatenate([np.asarray(s, np.int64), cid]),
                             np.concatenate([np.asarray(l, np.int64), clv])))
            return fold_items_enc(self.cfull, seqs)
        Z = fold_items_enc(self.ctx.enc, item_seqs)
        if self.clite is not None:                           # C-lite: trained fold on top
            from concept_fold import ConceptFoldNet
            cids = []; cvals = []
            for cl in conc_lists:
                ci, vv = ConceptFoldNet.canonical_order([c for c, _ in cl], [v for _, v in cl])
                cids.append(ci); cvals.append(vv)
            with torch.no_grad():
                return self.clite(Z, cids, cvals)
        for r, cl in enumerate(conc_lists):                  # Arm A / FixAB: additive shift
            cold = len(item_seqs[r][0]) == 0
            for c, v in cl:
                self.cmean.beta_ctx = 1.0 if cold else 5.0   # cold beta=1 (the curve winner)
                Zr = self.cmean.shift(int(c), float(v), cold=cold)
                Z[r] = Z[r] + Zr
        return Z

    def concept_order(self, answered_dirs, cand):
        """FixAB rung: skip concepts with |cos| >= COS_MAX to any already-ANSWERED concept."""
        if not self.div_select or not answered_dirs:
            return cand
        D = self.sh["d_c"].numpy()
        keep = []
        for c in cand:
            if all(abs(float(D[int(c)] @ D[int(p)])) < COS_MAX for p in answered_dirs):
                keep.append(c)
        return keep


# ============================================================================= deployment interview
def deploy_curves(rung, rows, budgets=BUDGETS):
    """Per-question interview: every asked question burns budget; unanswered = no fold. Returns
    {arm: {q: (full, tail, mean_answered)}} for arms a-d."""
    ctx, sh = rung.ctx, rung.sh
    out = {}
    qmax = max(budgets)
    for arm in ("items-pop", "items-entropy", "concepts-only", "mixed"):
        # walk the interview once to qmax per user, snapshotting evidence at each budget
        ev = {q: ([], []) for q in budgets}                  # q -> (item_seqs, conc_lists) parallel rows
        for r in rows:
            d = sh["lvl_lookup"][r]
            items = []; concs = []; answered_c = []
            asked_items = set(); asked_conc = set()
            nq = 0

            def ask_item(order_arr):
                nonlocal nq
                for i in order_arr:
                    i = int(i)
                    if i in asked_items:
                        continue
                    asked_items.add(i); nq += 1
                    if i in d:
                        items.append((i, d[i]))
                    return True
                return False

            def ask_conc():
                nonlocal nq
                cand = [c for c in sh["bank_conc"] if c not in asked_conc]
                cand = rung.concept_order(answered_c, cand)
                if not cand:
                    return False
                c = int(cand[0]); asked_conc.add(c); nq += 1
                if sh["conc_answerable"][r, c]:
                    concs.append((c, float(sh["conc_value"][r, c])))
                    answered_c.append(c)
                return True

            for q in range(1, qmax + 1):
                if arm == "items-pop":
                    ask_item(sh["order_pop"])
                elif arm == "items-entropy":
                    ask_item(sh["order_entropy"])
                elif arm == "concepts-only":
                    ask_conc()
                else:                                        # mixed coarse-to-fine
                    # fixed COARSE PHASE: questions 1..4 are concepts, then popularity items
                    # (deterministic schedule, documented; the design candidate -- tune later)
                    if q <= 4:
                        ask_conc()
                    else:
                        ask_item(sh["order_pop"])
                if q in budgets:
                    ev[q][0].append((np.asarray([s for s, _ in items], np.int64),
                                     np.asarray([l for _, l in items], np.int64)))
                    ev[q][1].append(list(concs))
        res = {}
        for q in budgets:
            item_seqs, conc_lists = ev[q]
            Z = rung.z_batch(item_seqs, conc_lists)
            full = np.full(len(rows), np.nan); tail = np.full(len(rows), np.nan)
            for st in range(0, len(rows), 500):
                ch = rows[st:st + 500]
                S = (Z[st:st + len(ch)] @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
                # credit-neutral: mask + IDCG-drop the ANSWERED items (concepts reveal no items).
                # NOTE: unanswered ASKED items are also shown to the user; masking them too would
                # need the per-budget asked trace -- the answered set is what enters the fold and
                # the te overlap; asked-but-unrated items are by definition NOT in va_te (unrated),
                # so dropping them from IDCG is a no-op. Masked set = answered items (exact).
                mask = ctx.va_tr[ch].copy()
                te = ctx.va_te[ch].tolil()
                for j in range(len(ch)):
                    s = item_seqs[st + j][0]
                    if len(s):
                        extra = sparse.csr_matrix((np.ones(len(s), np.float32),
                                                   (np.zeros(len(s)), s)), shape=(1, ctx.ni))
                        mask[j] = (mask[j] + extra)
                        for i in s.tolist():
                            te[j, i] = 0
                mask.data[:] = 1.0
                te = te.tocsr(); te.eliminate_zeros()
                f, t = ndcg10_from_scores(S, mask.tocsr(), te, ctx.head_mask)
                full[st:st + len(ch)] = f; tail[st:st + len(ch)] = t
            n_ans = np.array([len(item_seqs[j][0]) + len(conc_lists[j]) for j in range(len(rows))])
            res[q] = {"full@10": float(np.nanmean(full)), "tail@10": float(np.nanmean(tail)),
                      "mean_answered": float(n_ans.mean())}
        out[arm] = res
        log(f"[{rung.name} deploy {arm}] " + " ".join(
            f"q{q}={res[q]['full@10']:.4f}/{res[q]['tail@10']:.4f}(a={res[q]['mean_answered']:.1f})"
            for q in budgets))
    # crossover: smallest q where concepts-only full beats each item arm
    cross = {}
    for ia in ("items-pop", "items-entropy"):
        cq = next((q for q in budgets
                   if out["concepts-only"][q]["full@10"] > out[ia][q]["full@10"]), None)
        cross[ia] = cq
    out["crossover_concepts_beat"] = cross
    return out


# ============================================================================= per-answer section
def _concepts_only_curve(rung, rows_c, sel):
    """Rung-operator concepts-only m curve under a given selection. Returns ({m: full}, {m: tail})."""
    ctx = rung.ctx
    ff = {}; tt = {}
    for m in M_LIST:
        item_seqs = [(np.empty(0, np.int64), np.empty(0, np.int64))] * len(rows_c)
        conc_lists = [list(zip(sel[r][0][:m], sel[r][1][:m])) for r in rows_c]
        Z = rung.z_batch(item_seqs, conc_lists)
        full = np.full(len(rows_c), np.nan); tail = np.full(len(rows_c), np.nan)
        for st in range(0, len(rows_c), 500):
            ch = rows_c[st:st + 500]
            S = (Z[st:st + len(ch)] @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
            f, t = ndcg10_from_scores(S, ctx.va_tr[ch], ctx.va_te[ch], ctx.head_mask)
            full[st:st + len(ch)] = f; tail[st:st + len(ch)] = t
        ff[str(m)] = float(np.nanmean(full)); tt[str(m)] = float(np.nanmean(tail))
    return ff, tt


def per_answer_section(rung, rows):
    """Concepts-only m curve (rung operator; DIVERSIFIED selection = the default per-answer axis --
    author ruling 2026-07-24), items coldk2/k8 under the rung's own tower, mixed m2k2. armA also emits
    the top-SEL comparator row (the correlation-artifact story stays visible)."""
    ctx, sh = rung.ctx, rung.sh
    sel = sh["sel_div"]                                       # DEFAULT: diversified |cos|<0.5
    res = {"selection": "diversified |cos|<0.5 (default per author ruling 2026-07-24)"}
    rows_c = [r for r in rows if r in sel]
    ff, tt = _concepts_only_curve(rung, rows_c, sel)
    res["concepts_only"] = {"full@10": ff, "tail@10": tt}
    if rung.name == "armA":                                   # comparator row: top-SEL under armA
        rows_t = [r for r in rows if r in sh["sel_top"]]
        ff_t, tt_t = _concepts_only_curve(rung, rows_t, sh["sel_top"])
        res["concepts_only_topSEL_comparator"] = {"full@10": ff_t, "tail@10": tt_t}
    seq = [res["concepts_only"]["full@10"][str(m)] for m in M_LIST]
    res["monotone_to_m8_full"] = bool(all(seq[i + 1] >= seq[i] - 1e-9 for i in range(len(seq) - 1)))
    seqt = [res["concepts_only"]["tail@10"][str(m)] for m in M_LIST]
    res["monotone_to_m8_tail"] = bool(all(seqt[i + 1] >= seqt[i] - 1e-9 for i in range(len(seqt) - 1)))
    # items coldk2/k8 UNDER THIS RUNG'S TOWER (C-full: its own numbers = the item-cost decision axis)
    if hasattr(ctx, "L_val"):
        enc = rung.enc()
        for kk in (2, 8):
            Lm = truncate_graded(ctx.L_val, kk, K_SEED[kk])
            toks = [(Lm[i].indices.astype(np.int64), (Lm[i].data - 1).astype(np.int64))
                    for i in range(ctx.n)]
            Z = fold_items_enc(enc, [toks[r] for r in rows])
            full = np.full(len(rows), np.nan)
            for st in range(0, len(rows), 500):
                ch = rows[st:st + 500]
                S = (Z[st:st + len(ch)] @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
                f, _ = ndcg10_from_scores(S, ctx.va_tr[ch], ctx.va_te[ch], None)
                full[st:st + len(ch)] = f
            res[f"coldk{kk}_full@10"] = float(np.nanmean(full))
    # mixed m2k2
    rows_m = rows_c
    k2 = [ctx.k2_tokens[r] for r in rows_m]
    conc2 = [list(zip(sel[r][0][:2], sel[r][1][:2])) for r in rows_m]
    Zm = rung.z_batch(k2, conc2)
    Zi = rung.z_batch(k2, [[] for _ in rows_m])
    fm = np.full(len(rows_m), np.nan); fi = np.full(len(rows_m), np.nan)
    for st in range(0, len(rows_m), 500):
        ch = rows_m[st:st + 500]
        Sm = (Zm[st:st + len(ch)] @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
        Si = (Zi[st:st + len(ch)] @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
        f1, _ = ndcg10_from_scores(Sm, ctx.va_tr[ch], ctx.va_te[ch], None)
        f2, _ = ndcg10_from_scores(Si, ctx.va_tr[ch], ctx.va_te[ch], None)
        fm[st:st + len(ch)] = f1; fi[st:st + len(ch)] = f2
    dmix = bootstrap_ci(fm - fi)
    res["mixed_m2_k2"] = {"full@10": float(np.nanmean(fm)),
                          "vs_items_k2": {"mean": dmix[0], "ci95": dmix[1]}}
    return res


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--rungs", default="armA,fixab,clite,cfull")
    ap.add_argument("--clite_ckpt", default=os.path.join(_ROOT, ".cache", "instrument",
                                                         "cfold_best.pt"))
    ap.add_argument("--cfull_ckpt", default=os.path.join(_ROOT, ".cache", "instrument",
                                                         "cfull_best.pt"))
    ap.add_argument("--full_threads", action="store_true")
    args = ap.parse_args()
    t00 = time.time()
    ctx = build_smoke_ctx() if args.smoke else build_real_ctx(args.snapshot)
    sh = build_shared(ctx)
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    # intercept control
    empty = [(np.empty(0, np.int64), np.empty(0, np.int64))] * ctx.n
    f_int, t_int = eval_tokens(ctx, empty, ctx.va_tr, rows=rows)
    out = {"analysis": "tradeoff_ledger", "n_users": len(rows), "budgets": list(BUDGETS),
           "intercept": {"full": float(np.nanmean(f_int[rows])), "tail": float(np.nanmean(t_int[rows]))},
           "deployment_note": ("per-QUESTION currency: every question burns budget whether answered "
                               "or not; items answered iff rated (structural rule), concepts iff >=2 "
                               "rated members; concept values = SEL-graded (behavioral simulator, "
                               "te excluded)"),
           "rungs": {}}
    rungs = {}
    for name in args.rungs.split(","):
        name = name.strip()
        if name == "armA":
            rungs[name] = Rung(name, ctx, sh)
        elif name == "fixab":
            rungs[name] = Rung(name, ctx, sh, div_select=True)
        elif name == "clite":
            if not os.path.exists(args.clite_ckpt) and not args.smoke:
                out["rungs"][name] = "PENDING (no ckpt)"; continue
            from concept_fold import ConceptFoldNet
            if args.smoke:
                net = ConceptFoldNet(len(sh["tags"]), d=ctx.d, h=32, conc_init=sh["d_c"].numpy())
            else:
                blob = torch.load(args.clite_ckpt, map_location="cpu")
                net = ConceptFoldNet(len(blob["tags"]), d=ctx.d, h=blob["hidden"])
                net.load_state_dict(blob["net"])
            net.eval()
            rungs[name] = Rung(name, ctx, sh, clite_net=net)
        elif name == "cfull":
            if not os.path.exists(args.cfull_ckpt) and not args.smoke:
                out["rungs"][name] = "PENDING (no ckpt)"; continue
            if args.smoke:
                enc_cf = ctx.enc                             # smoke: reuse (concept ids would break --
                out["rungs"][name] = "SKIPPED in smoke (needs a --concept_tokens ckpt)"; continue
            blob = torch.load(args.cfull_ckpt, map_location="cpu")
            native = ctx.enc._native[0] if hasattr(ctx.enc, "_native") else None
            enc_cf = I25Encoder(ctx.ni, native, d_lat=ctx.d, token_mode="film",
                                n_concepts=len(sh["tags"]))
            enc_cf.load_state_dict(blob["enc"]); enc_cf.eval()
            rungs[name] = Rung(name, ctx, sh, cfull_enc=enc_cf)
    for name, rung in rungs.items():
        log(f"=== RUNG {name} ===")
        r_out = {"per_answer": per_answer_section(rung, rows)}
        r_out["deployment"] = deploy_curves(rung, rows)
        out["rungs"][name] = r_out
    # canonical-snap control vs the recorded per-answer curves (armA vs concepts_only_curve.json)
    prev_p = os.path.join(OUTDIR, "concepts_only_curve.json")
    if (not args.smoke) and os.path.exists(prev_p) and "armA" in rungs:
        prev = json.load(open(prev_p))
        comp = out["rungs"]["armA"]["per_answer"]["concepts_only_topSEL_comparator"]
        deltas = {str(m): comp["full@10"][str(m)]
                  - prev["configs"]["pure_whitened_b1"]["full@10"][str(m)] for m in M_LIST}
        out["snap_control_armA_topSEL_vs_recorded"] = deltas
    out["seconds"] = round(time.time() - t00, 1)
    outdir = ctx.outdir if args.smoke else OUTDIR
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "tradeoff_ledger.json")
    json.dump(out, open(path, "w"), indent=2)
    log(f"[out] -> {path}")
    # ---- printed ledger ----
    print("\n=== TRADEOFF LEDGER (author decision table) ===")
    print(f"intercept {out['intercept']['full']:.4f}/{out['intercept']['tail']:.4f}; "
          f"n={len(rows)}")
    for name in out["rungs"]:
        r = out["rungs"][name]
        if isinstance(r, str):
            print(f"[{name}] {r}"); continue
        pa = r["per_answer"]
        cc = pa["concepts_only"]["full@10"]
        if "concepts_only_topSEL_comparator" in pa:
            ct = pa["concepts_only_topSEL_comparator"]["full@10"]
            print(f"[{name}] per-answer topSEL comparator: conc m1/2/4/8 = " +
                  "/".join(f"{ct[str(m)]:.4f}" for m in M_LIST) + "  (the correlation artifact)")
        print(f"[{name}] per-answer (div-sel): conc m1/2/4/8 = " +
              "/".join(f"{cc[str(m)]:.4f}" for m in M_LIST) +
              f" mono_m8={pa['monotone_to_m8_full']}|{pa['monotone_to_m8_tail']}" +
              (f" coldk2={pa.get('coldk2_full@10', float('nan')):.4f}"
               f" coldk8={pa.get('coldk8_full@10', float('nan')):.4f}" if "coldk2_full@10" in pa else "") +
              f" mixed_m2k2 {pa['mixed_m2_k2']['vs_items_k2']['mean']:+.4f}")
        dep = r["deployment"]
        for arm in ("items-pop", "items-entropy", "concepts-only", "mixed"):
            print(f"    deploy {arm:>14}: " + " ".join(
                f"q{q}={dep[arm][q]['full@10']:.4f}(a={dep[arm][q]['mean_answered']:.1f})"
                for q in BUDGETS))
        print(f"    crossover concepts>items: {dep['crossover_concepts_beat']}")


if __name__ == "__main__":
    main()
