r"""distill_downstream_chain.py -- OVERNIGHT downstream chain on the BEST DISTILLED concept fold
(docs/TODO.md post-distillation plan, author-aligned Jul 26; coordinator directive). Runs AFTER the
distillation POC finishes, on the winning distilled ConceptFoldNet checkpoint. Session-independent:
state file + heartbeat + done-marker, idempotent stages (skip if the stage JSON already exists),
detached, split logs, schtasks watchdog resume-on-death. HARD STOP after D3 (no G10 / full-battery /
certified retrain -- those need author decisions).

BEST FOLD = argmax overall-kc4 capture_full among the DISTILLED configs (cd_s1_l*/cd_s2; excludes the
no-distill control and the shuffle canary) in experiments/battery/concept_distill_train.json.

STAGE D1 -- concept-ask vs item-ask on the fixed fold (does the crossover move?):
  D1a (committed harness): adaptive_concept_arms.py --rungs sclite --sclite_ckpt BEST -> realizable
      (greedy) / oracle (|signed v|) / static concept selection + items-pop comparator, full+tail@10,
      gap-closure crossover, credit-neutral masking.
  D1b (committed harness): concept_fold.py --eval_curve BEST -> matched-budget armC concepts-only curve
      (m=1,2,4,8) vs items_at_k, monotonicity, fraction-of-item-gain, @10.
  D1c (new, this file): matched-budget @100 -- realizable top-|v| CONCEPTS vs pop ITEMS at k=1,2,4,8,
      full+tail@100 (does the @10 crossover hold at @100?).
STAGE D2 -- answerer panel on the fixed fold (committed harness answer_contrast.py --arms G,B,O,K,U,S,C,
  E,P --sclite_ckpt BEST): SEL vs SEL+/ExpoMF/content/PITF + oracle-B + geometric, capture + deployment
  curve, full+tail @10 AND @100, CKA-to-u* + oracle-B-agreement guards. KEY: do imputers differentiate
  now that the fold isn't lossy (was "SEL is the ceiling" a broken-fold artifact)?
STAGE D3 -- 1-level ADAPTIVITY TREE (new, this file; author-requested clean test, refutes static-optimal
  / HARD RULE #2): ask q1 -> BRANCH on the answer -> each branch's OWN best q2 (selected on a SELECT
  half of the cohort, evaluated OUT-OF-SAMPLE on the EVAL half) vs a single STATIC best q2 for everyone.
  ADAPTIVE-q2 vs STATIC-q2, NDCG full+tail@10. Run for BOTH the concept channel (fold-dependent) and the
  item channel (fold-independent).

Controls every stage: canonical q0 intercept snap (0.12794/0.01923), leak (fold-in INTERSECT held == 0),
credit-neutral masking. Each stage writes a results JSON + appends docs/results/DISTILL_DOWNSTREAM_RESULT.md.

Usage:
  python src/instrument/distill_downstream_chain.py --smoke
  python src/instrument/distill_downstream_chain.py --run [--best_ckpt PATH] [--wait_marker PATH]
"""
import os
import sys
_FULL = "--full_threads" in sys.argv or "--run" in sys.argv
_NT = str(os.cpu_count()) if _FULL else "4"
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)
os.environ.setdefault("MKL_NUM_THREADS", _NT)
import json
import time
import argparse
import subprocess
import numpy as np
import torch
from scipy import sparse

torch.set_num_threads(int(_NT))

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)

from train_tower_t2 import log
from concept_fold import ConceptFoldNet, fold_items_frozen

CKPT_DIR = os.path.join(_ROOT, ".cache", "instrument")
OUTDIR = os.path.join(_ROOT, "experiments", "battery")
RESULT_MD = os.path.join(_ROOT, "docs", "results", "DISTILL_DOWNSTREAM_RESULT.md")
POC_JSON = os.path.join(OUTDIR, "concept_distill_train.json")
STATE = os.path.join(OUTDIR, "distill_chain_state.json")
HEART = os.path.join(OUTDIR, "distill_chain.heartbeat")
DONE = os.path.join(OUTDIR, "distill_chain_DONE.marker")
INTERCEPT_REF = (0.12793773315625726, 0.01923195232020529)
KS = (10, 100)
KC = (1, 2, 4, 8)
EMPTY = (np.empty(0, np.int64), np.empty(0, np.int64))
PYU = [sys.executable, "-u"]


# ============================================================================= state / heartbeat
def _load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"stages": {}, "best_ckpt": None,
                                                                  "started": time.strftime("%F %T")}


def _save_state(s):
    json.dump(s, open(STATE, "w"), indent=2, default=float)


def _beat(msg):
    open(HEART, "w").write(f"{time.strftime('%F %T')} {msg}\n")
    log(f"[chain] {msg}")


def _append_md(title, body):
    os.makedirs(os.path.dirname(RESULT_MD), exist_ok=True)
    with open(RESULT_MD, "a", encoding="utf-8") as f:          # utf-8: tables use Delta/arrows
        f.write(f"\n## {title}  ({time.strftime('%F %T')})\n\n{body}\n")


def select_best_ckpt():
    """argmax overall-kc4 capture_full among distilled configs (cd_s1_l*/cd_s2)."""
    d = json.load(open(POC_JSON))
    best, bt = -9.9, None
    for tag, c in d.get("configs", {}).items():
        if not (tag.startswith("cd_s1_l") or tag == "cd_s2"):
            continue
        cap = c.get("capture", {}).get("overall", {}).get("4", {}).get("capture_full")
        if cap is not None and cap > best:
            best, bt = cap, tag
    if bt is None:
        raise RuntimeError("no distilled config found in POC json")
    return os.path.join(CKPT_DIR, f"{bt}_best.pt"), bt, best


# ============================================================================= shared ctx
def _build_ctx():
    import run_battery_phaseA as PA
    from run_battery_phaseA import build_real_ctx
    from tradeoff_ledger import build_shared
    ctx = build_real_ctx(PA.SNAP_DEFAULT)
    ctx.Wd = ctx.decoder.weight.detach().float(); ctx.bd = ctx.decoder.bias.detach().float()
    ctx.d = ctx.enc.d_out
    sh = build_shared(ctx)
    assert sh.get("signed_available"), "signed prereg cache required"
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    # controls
    leaks = sum(1 for r in rows if len(np.intersect1d(ctx.allb[r][0], ctx.va_te[r].indices)) > 0)
    assert leaks == 0, f"LEAK: {leaks} users fold-in/held overlap"
    z0 = fold_items_frozen(ctx.enc, [EMPTY])
    base_vec = (z0[0] @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
    from run_battery_phaseA import ndcg10_from_scores
    fb = np.full(len(rows), np.nan); tb = np.full(len(rows), np.nan)
    for st in range(0, len(rows), 500):
        ch = rows[st:st + 500]
        S = np.repeat(base_vec[None, :], len(ch), axis=0)
        f, t = ndcg10_from_scores(S, ctx.va_tr[ch], ctx.va_te[ch], ctx.head_mask)
        fb[st:st + len(ch)] = f; tb[st:st + len(ch)] = t
    icept = (float(np.nanmean(fb)), float(np.nanmean(tb)))
    snap = abs(icept[0] - INTERCEPT_REF[0]) < 5e-3 and abs(icept[1] - INTERCEPT_REF[1]) < 5e-3
    log(f"[chain] ctx ready: n_rows={len(rows)} intercept {icept[0]:.5f}/{icept[1]:.5f} snap={snap} leak=0")
    return ctx, sh, rows, z0, icept, bool(snap)


def _load_net(ckpt, ctx):
    blob = torch.load(ckpt, map_location="cpu")
    net = ConceptFoldNet(len(blob["tags"]), d=ctx.d, h=blob["hidden"], conc_init=None)
    net.load_state_dict(blob["net"]); net.eval()
    return net, blob["tags"]


# ============================================================================= fold helpers (net + tower)
def concept_ndcg(net, ctx, z0, rows, pairs_by_user, ks=KS, batch=500):
    """Concepts-only fold via the distilled net -> NDCG at cutoffs. pairs_by_user[r] = list of (cid,val).
    Concepts reveal no items -> mask = va_tr only."""
    from answer_contrast import ndcg_multi
    acc = {f"{m}@{k}": np.full(len(rows), np.nan) for k in ks for m in ("full", "tail")}
    with torch.no_grad():
        for st in range(0, len(rows), batch):
            ch = rows[st:st + batch]
            Z0 = z0.repeat(len(ch), 1)
            cids = []; cvals = []
            for r in ch:
                pr = pairs_by_user.get(r, [])
                ci, vv = ConceptFoldNet.canonical_order([int(c) for c, _ in pr], [float(v) for _, v in pr])
                cids.append(ci); cvals.append(vv)
            zo = net(Z0, cids, cvals)
            S = (zo @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
            r_ = ndcg_multi(S, ctx.va_tr[ch], ctx.va_te[ch], ctx.head_mask, ks)
            for kk in acc:
                acc[kk][st:st + len(ch)] = r_[kk]
    return acc


def item_ndcg(ctx, rows, tokens_by_user, ks=KS, batch=500):
    """Item fold via the FROZEN tower -> NDCG at cutoffs, credit-neutral masking of the folded items.
    tokens_by_user[r] = (sids, lvls)."""
    from answer_contrast import ndcg_multi
    acc = {f"{m}@{k}": np.full(len(rows), np.nan) for k in ks for m in ("full", "tail")}
    seqs = [tokens_by_user.get(r, EMPTY) for r in rows]
    Z = fold_items_frozen(ctx.enc, seqs)
    for st in range(0, len(rows), batch):
        ch = rows[st:st + batch]
        S = (Z[st:st + len(ch)] @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
        mask = ctx.va_tr[ch].copy(); te = ctx.va_te[ch].tolil()
        for j in range(len(ch)):
            s = seqs[st + j][0]
            if len(s):
                extra = sparse.csr_matrix((np.ones(len(s), np.float32), (np.zeros(len(s)), s)),
                                          shape=(1, ctx.ni))
                mask[j] = mask[j] + extra
                for i in np.asarray(s).tolist():
                    te[j, i] = 0
        mask.data[:] = 1.0; te = te.tocsr(); te.eliminate_zeros()
        r_ = ndcg_multi(S, mask.tocsr(), te, ctx.head_mask, ks)
        for kk in acc:
            acc[kk][st:st + len(ch)] = r_[kk]
    return acc


def _mean(acc, key, idx=None):
    a = acc[key]
    return float(np.nanmean(a if idx is None else a[idx]))


# ============================================================================= STAGE D1
def stage_d1(ctx, sh, rows, z0, net, best_ckpt, logdir):
    out = {"stage": "D1_concept_vs_item", "best_ckpt": os.path.basename(best_ckpt)}
    # D1a + D1b: committed harnesses (best-effort subprocess; @10)
    for label, cmd, produced in [
        ("D1a_adaptive_concept_arms",
         PYU + [os.path.join(_HERE, "adaptive_concept_arms.py"), "--rungs", "sclite",
                "--sclite_ckpt", best_ckpt, "--full_threads"],
         os.path.join(OUTDIR, "adaptive_concept_arms.json")),
        ("D1b_concepts_only_curve",
         PYU + [os.path.join(_HERE, "concept_fold.py"), "--eval_curve", best_ckpt, "--full_threads"],
         os.path.join(OUTDIR, "concepts_only_curve_armC.json"))]:
        _beat(f"D1 running {label}")
        lo = open(os.path.join(logdir, f"{label}.out"), "w")
        le = open(os.path.join(logdir, f"{label}.err"), "w")
        rc = subprocess.call(cmd, cwd=_ROOT, stdout=lo, stderr=le)
        lo.close(); le.close()
        out[label] = {"returncode": rc, "produced": (os.path.basename(produced)
                                                      if os.path.exists(produced) else None)}
        if os.path.exists(produced):
            try:
                out[label]["summary"] = json.load(open(produced))
            except Exception as e:
                out[label]["summary_error"] = str(e)
        log(f"[chain] {label} rc={rc} produced={out[label]['produced']}")

    # D1c: matched-budget @10 AND @100 -- realizable top-|v| concepts vs pop items, k=1,2,4,8
    _beat("D1c matched-budget @10/@100 concepts-vs-items")
    Vs = sh["conc_value_signed"]; Bs = sh["conc_band_signed"]; Fs = sh["conc_fold_signed"]
    ans = sh["conc_answerable"]; lvl = sh["lvl_lookup"]; order_pop = sh["order_pop"]
    from signed_answers import cap_negatives
    # per-user realizable concept ranking (top |signed v| among answerable & foldable)
    conc_rank = {}
    for r in rows:
        elig = np.flatnonzero(ans[r] & Fs[r])
        if len(elig):
            conc_rank[r] = elig[np.argsort(-np.abs(Vs[r, elig]))]
        else:
            conc_rank[r] = np.empty(0, np.int64)
    d1c = {"concepts": {}, "items": {}, "intercept": {"full@10": None}}
    for k in KC:
        cpairs = {}
        for r in rows:
            cs = conc_rank[r][:k]
            vv = cap_negatives([float(Vs[r, c]) for c in cs])
            cpairs[r] = list(zip([int(c) for c in cs], vv))
        cacc = concept_ndcg(net, ctx, z0, rows, cpairs)
        d1c["concepts"][str(k)] = {kk: _mean(cacc, kk) for kk in cacc}
        # items: top-k popular that the user actually rated (realizable item-ask), te-excluded
        itok = {}
        for r in rows:
            d = lvl[r]; picked = [int(i) for i in order_pop if int(i) in d][:k]
            itok[r] = (np.asarray(picked, np.int64), np.asarray([d[i] for i in picked], np.int64))
        iacc = item_ndcg(ctx, rows, itok)
        d1c["items"][str(k)] = {kk: _mean(iacc, kk) for kk in iacc}
        log(f"[chain D1c k={k}] concept full@10={d1c['concepts'][str(k)]['full@10']:.4f} "
            f"tail@10={d1c['concepts'][str(k)]['tail@10']:.4f} | item full@10="
            f"{d1c['items'][str(k)]['full@10']:.4f} tail@10={d1c['items'][str(k)]['tail@10']:.4f}")
    # crossover verdict: smallest k where concept full@10 >= item full@10 (realizable)
    cross = None
    for k in KC:
        if d1c["concepts"][str(k)]["full@10"] >= d1c["items"][str(k)]["full@10"]:
            cross = k; break
    d1c["concept_beats_item_full10_at_k"] = cross
    d1c["concept_beats_item_tail10_at_k"] = next(
        (k for k in KC if d1c["concepts"][str(k)]["tail@10"] >= d1c["items"][str(k)]["tail@10"]), None)
    out["D1c_matched_budget"] = d1c
    json.dump(out, open(os.path.join(OUTDIR, "distill_d1.json"), "w"), indent=2, default=float)
    # RESULT md
    lines = [f"Best distilled fold: `{os.path.basename(best_ckpt)}`.",
             "", "Matched-budget realizable concept-ask vs pop item-ask (full@10 | tail@10 | full@100):", ""]
    lines.append("| k | concept f@10 | item f@10 | concept t@10 | item t@10 | concept f@100 | item f@100 |")
    lines.append("|---|---|---|---|---|---|---|")
    for k in KC:
        c = d1c["concepts"][str(k)]; i = d1c["items"][str(k)]
        lines.append(f"| {k} | {c['full@10']:.4f} | {i['full@10']:.4f} | {c['tail@10']:.4f} | "
                     f"{i['tail@10']:.4f} | {c['full@100']:.4f} | {i['full@100']:.4f} |")
    lines += ["", f"Concept beats item (full@10) at k = {cross}; (tail@10) at k = "
              f"{d1c['concept_beats_item_tail10_at_k']}.",
              f"D1a adaptive_concept_arms rc={out['D1a_adaptive_concept_arms']['returncode']}, "
              f"D1b concepts_only_curve rc={out['D1b_concepts_only_curve']['returncode']} "
              "(committed-harness @10 crossover + selection arms in their JSONs)."]
    _append_md("STAGE D1 — concept-vs-item on the fixed fold", "\n".join(lines))
    return out


# ============================================================================= STAGE D2
def stage_d2(best_ckpt, logdir):
    _beat("D2 answerer panel (answer_contrast)")
    outjson = "answer_contrast_distill.json"
    cmd = PYU + [os.path.join(_HERE, "answer_contrast.py"), "--arms", "G,B,O,K,U,S,C,E,P",
                 "--sclite_ckpt", best_ckpt, "--out", outjson, "--full_threads"]
    lo = open(os.path.join(logdir, "D2_answer_contrast.out"), "w")
    le = open(os.path.join(logdir, "D2_answer_contrast.err"), "w")
    rc = subprocess.call(cmd, cwd=_ROOT, stdout=lo, stderr=le)
    lo.close(); le.close()
    produced = os.path.join(OUTDIR, outjson)
    out = {"stage": "D2_answerer_panel", "returncode": rc,
           "produced": outjson if os.path.exists(produced) else None}
    body = f"answer_contrast rc={rc}, out=`{outjson}`."
    if os.path.exists(produced):
        try:
            d = json.load(open(produced))
            out["summary"] = d
            body += "\n\nPanel ran on the fixed fold; see JSON for per-arm capture/curve + CKA/oracle-B " \
                    "guards. KEY read: whether imputers now differentiate vs SEL (broken-fold artifact test)."
        except Exception as e:
            out["summary_error"] = str(e)
    json.dump(out, open(os.path.join(OUTDIR, "distill_d2.json"), "w"), indent=2, default=float)
    _append_md("STAGE D2 — answerer panel on the fixed fold", body)
    log(f"[chain] D2 rc={rc} produced={out['produced']}")
    return out


# ============================================================================= STAGE D3 (adaptivity tree)
def _best_q(cand, score_fn):
    """argmax over candidate list of mean score; returns (best_c, best_val, table)."""
    best_c, best_v, tab = None, -9.9, {}
    for c in cand:
        v = score_fn(c)
        tab[int(c)] = v
        if v > best_v:
            best_v, best_c = v, int(c)
    return best_c, best_v, tab


def _tree_concept(ctx, sh, rows, z0, net, bank_n=40):
    """1-level concept adaptivity tree: opener q1 -> branch on the SEL band -> per-branch best q2, vs one
    static q2. Selection on the SELECT half, evaluation OUT-OF-SAMPLE on the EVAL half."""
    from run_battery_phaseA import bootstrap_ci
    from signed_answers import cap_negatives
    Vs = sh["conc_value_signed"]; Bs = sh["conc_band_signed"]; Fs = sh["conc_fold_signed"]
    ans = sh["conc_answerable"]; bank = [int(c) for c in sh["bank_conc"][:bank_n]]
    rows = np.asarray(rows)
    sel = rows[rows % 2 == 0].tolist(); ev = rows[rows % 2 == 1].tolist()

    def pair(r, c):
        if ans[r, c] and Fs[r, c]:
            return [(int(c), float(cap_negatives([float(Vs[r, c])])[0]))]
        return []

    def scores_for(users, q):                                    # concepts-only fold of the single q
        acc = concept_ndcg(net, ctx, z0, users, {r: pair(r, q) for r in users}, ks=(10,))
        return acc

    # opener q1* = best single concept on SELECT (full@10)
    q1, q1v, _ = _best_q(bank, lambda c: float(np.nanmean(scores_for(sel, c)["full@10"])))

    def branch_of(r):
        return int(Bs[r, q1]) if (ans[r, q1] and Fs[r, q1]) else 3   # 0 like /1 meh /2 dislike /3 refuse

    def q1q2_ndcg(users, q2):
        pu = {r: (pair(r, q1) + pair(r, q2)) for r in users}
        return concept_ndcg(net, ctx, z0, users, pu, ks=(10,))

    cand2 = [c for c in bank if c != q1]
    # STATIC q2 on SELECT (all users)
    q2s, q2sv, _ = _best_q(cand2, lambda c: float(np.nanmean(q1q2_ndcg(sel, c)["full@10"])))
    # ADAPTIVE q2 per branch on SELECT
    sel_branch = {b: [r for r in sel if branch_of(r) == b] for b in (0, 1, 2, 3)}
    q2_by_branch = {}
    for b, us in sel_branch.items():
        if len(us) < 30:
            q2_by_branch[b] = q2s; continue
        q2_by_branch[b], _, _ = _best_q(cand2, lambda c, us=us: float(np.nanmean(q1q2_ndcg(us, c)["full@10"])))
    # EVAL out-of-sample
    ev_static = {r: (pair(r, q1) + pair(r, q2s)) for r in ev}
    ev_adapt = {r: (pair(r, q1) + pair(r, q2_by_branch[branch_of(r)])) for r in ev}
    a_stat = concept_ndcg(net, ctx, z0, ev, ev_static, ks=(10,))
    a_adap = concept_ndcg(net, ctx, z0, ev, ev_adapt, ks=(10,))
    df = a_adap["full@10"] - a_stat["full@10"]; dt = a_adap["tail@10"] - a_stat["tail@10"]
    mf, cif, _ = bootstrap_ci(df); mt, cit, _ = bootstrap_ci(dt)
    return {"channel": "concept", "q1": q1, "q2_static": q2s, "q2_by_branch": q2_by_branch,
            "branch_sizes_eval": {b: int(sum(branch_of(r) == b for r in ev)) for b in (0, 1, 2, 3)},
            "static_full@10": float(np.nanmean(a_stat["full@10"])),
            "adaptive_full@10": float(np.nanmean(a_adap["full@10"])),
            "static_tail@10": float(np.nanmean(a_stat["tail@10"])),
            "adaptive_tail@10": float(np.nanmean(a_adap["tail@10"])),
            "adaptive_minus_static_full": {"mean": mf, "ci95": cif},
            "adaptive_minus_static_tail": {"mean": mt, "ci95": cit},
            "adaptivity_pays_full": bool(mf > 0 and cif[0] > 0),
            "adaptivity_pays_tail": bool(mt > 0 and cit[0] > 0)}


def _tree_item(ctx, sh, rows, bank_n=40):
    """1-level ITEM adaptivity tree (fold-independent): opener item -> branch on the user's rating band
    -> per-branch best q2 item, vs one static q2 item. SELECT/EVAL out-of-sample split."""
    from run_battery_phaseA import bootstrap_ci
    lvl = sh["lvl_lookup"]; bank = [int(i) for i in sh["order_pop"][:bank_n]]
    rows = np.asarray(rows)
    sel = rows[rows % 2 == 0].tolist(); ev = rows[rows % 2 == 1].tolist()

    def tok(r, items):
        d = lvl[r]; got = [(int(i), d[int(i)]) for i in items if int(i) in d]
        if not got:
            return EMPTY
        return (np.asarray([i for i, _ in got], np.int64), np.asarray([l for _, l in got], np.int64))

    def band(r, i):
        d = lvl[r]
        if int(i) not in d:
            return 3                                             # not-watched
        lv = d[int(i)]
        return 0 if lv >= 7 else (1 if lv >= 5 else 2)          # like / meh / dislike

    def q1q2_ndcg(users, q1, q2):
        return item_ndcg(ctx, users, {r: tok(r, [q1, q2]) for r in users}, ks=(10,))

    def q1_ndcg(users, q1):
        return item_ndcg(ctx, users, {r: tok(r, [q1]) for r in users}, ks=(10,))

    q1, _, _ = _best_q(bank, lambda c: float(np.nanmean(q1_ndcg(sel, c)["full@10"])))
    cand2 = [i for i in bank if i != q1]
    q2s, _, _ = _best_q(cand2, lambda c: float(np.nanmean(q1q2_ndcg(sel, q1, c)["full@10"])))
    sel_branch = {b: [r for r in sel if band(r, q1) == b] for b in (0, 1, 2, 3)}
    q2_by_branch = {}
    for b, us in sel_branch.items():
        if len(us) < 30:
            q2_by_branch[b] = q2s; continue
        q2_by_branch[b], _, _ = _best_q(cand2, lambda c, us=us: float(np.nanmean(q1q2_ndcg(us, q1, c)["full@10"])))
    ev_static = {r: tok(r, [q1, q2s]) for r in ev}
    ev_adapt = {r: tok(r, [q1, q2_by_branch[band(r, q1)]]) for r in ev}
    a_stat = item_ndcg(ctx, ev, ev_static, ks=(10,))
    a_adap = item_ndcg(ctx, ev, ev_adapt, ks=(10,))
    df = a_adap["full@10"] - a_stat["full@10"]; dt = a_adap["tail@10"] - a_stat["tail@10"]
    mf, cif, _ = bootstrap_ci(df); mt, cit, _ = bootstrap_ci(dt)
    return {"channel": "item", "q1": q1, "q2_static": q2s, "q2_by_branch": q2_by_branch,
            "branch_sizes_eval": {b: int(sum(band(r, q1) == b for r in ev)) for b in (0, 1, 2, 3)},
            "static_full@10": float(np.nanmean(a_stat["full@10"])),
            "adaptive_full@10": float(np.nanmean(a_adap["full@10"])),
            "static_tail@10": float(np.nanmean(a_stat["tail@10"])),
            "adaptive_tail@10": float(np.nanmean(a_adap["tail@10"])),
            "adaptive_minus_static_full": {"mean": mf, "ci95": cif},
            "adaptive_minus_static_tail": {"mean": mt, "ci95": cit},
            "adaptivity_pays_full": bool(mf > 0 and cif[0] > 0),
            "adaptivity_pays_tail": bool(mt > 0 and cit[0] > 0)}


def stage_d3(ctx, sh, rows, z0, net):
    _beat("D3 adaptivity tree (concept channel)")
    conc = _tree_concept(ctx, sh, rows, z0, net)
    _beat("D3 adaptivity tree (item channel)")
    item = _tree_item(ctx, sh, rows)
    out = {"stage": "D3_adaptivity_tree", "concept": conc, "item": item}
    json.dump(out, open(os.path.join(OUTDIR, "distill_d3.json"), "w"), indent=2, default=float)
    for ch in (conc, item):
        log(f"[chain D3 {ch['channel']}] q1={ch['q1']} static f@10={ch['static_full@10']:.4f} "
            f"adaptive f@10={ch['adaptive_full@10']:.4f} d={ch['adaptive_minus_static_full']['mean']:+.4f} "
            f"pays_full={ch['adaptivity_pays_full']} pays_tail={ch['adaptivity_pays_tail']}")
    body = ["1-level tree: q1 -> branch on answer -> per-branch best q2 (selected on SELECT half, "
            "evaluated OUT-OF-SAMPLE on EVAL half) vs single STATIC q2.", ""]
    body.append("| channel | q1 | static f@10 | adaptive f@10 | Δfull (CI) | Δtail (CI) | pays full | pays tail |")
    body.append("|---|---|---|---|---|---|---|---|")
    for ch in (conc, item):
        df = ch["adaptive_minus_static_full"]; dt = ch["adaptive_minus_static_tail"]
        body.append(f"| {ch['channel']} | {ch['q1']} | {ch['static_full@10']:.4f} | "
                    f"{ch['adaptive_full@10']:.4f} | {df['mean']:+.4f} [{df['ci95'][0]:+.4f},{df['ci95'][1]:+.4f}] | "
                    f"{dt['mean']:+.4f} [{dt['ci95'][0]:+.4f},{dt['ci95'][1]:+.4f}] | "
                    f"{ch['adaptivity_pays_full']} | {ch['adaptivity_pays_tail']} |")
    _append_md("STAGE D3 — 1-level adaptivity tree (concept + item)", "\n".join(body))
    return out


# ============================================================================= driver
def run(args):
    os.makedirs(OUTDIR, exist_ok=True)
    logdir = os.path.join(OUTDIR, "distill_chain_logs"); os.makedirs(logdir, exist_ok=True)
    if args.wait_marker:
        _beat(f"waiting for POC marker {os.path.basename(args.wait_marker)}")
        while not os.path.exists(args.wait_marker):
            time.sleep(60)
        _beat("POC marker seen; starting chain")
    st = _load_state()
    if args.best_ckpt:
        best_ckpt, btag, bcap = args.best_ckpt, os.path.basename(args.best_ckpt), None
    else:
        best_ckpt, btag, bcap = select_best_ckpt()
    st["best_ckpt"] = best_ckpt; _save_state(st)
    _beat(f"best distilled fold = {btag} (overall kc4 capture {bcap}) -> {os.path.basename(best_ckpt)}")

    ctx, sh, rows, z0, icept, snap = _build_ctx()
    net, tags = _load_net(best_ckpt, ctx)

    stages = [("D1", os.path.join(OUTDIR, "distill_d1.json"),
               lambda: stage_d1(ctx, sh, rows, z0, net, best_ckpt, logdir)),
              ("D2", os.path.join(OUTDIR, "distill_d2.json"), lambda: stage_d2(best_ckpt, logdir)),
              ("D3", os.path.join(OUTDIR, "distill_d3.json"),
               lambda: stage_d3(ctx, sh, rows, z0, net))]
    for name, outp, fn in stages:
        if st["stages"].get(name) == "done" and os.path.exists(outp):
            _beat(f"SKIP {name} (already done)"); continue
        _beat(f"START {name}")
        try:
            fn()
            st["stages"][name] = "done"
        except Exception as e:
            import traceback
            st["stages"][name] = f"FAILED: {e}"
            open(os.path.join(logdir, f"{name}_TRACEBACK.txt"), "w").write(traceback.format_exc())
            _beat(f"{name} FAILED: {e} (chain continues)")
        _save_state(st)

    open(DONE, "w").write(f"done {time.strftime('%F %T')} stages={st['stages']}\n")
    _beat(f"CHAIN COMPLETE stages={st['stages']}")


# ============================================================================= smoke
def _smoke():
    print("[SMOKE] distill_downstream_chain")
    import run_battery_phaseA as PA
    from tradeoff_ledger import build_shared
    ctx = PA.build_smoke_ctx()
    ctx.Wd = ctx.decoder.weight.detach().float(); ctx.bd = ctx.decoder.bias.detach().float()
    ctx.d = ctx.enc.d_out
    sh = build_shared(ctx)
    # fabricate signed arrays if the smoke ctx lacks the prereg cache (mirror step0 smoke)
    if not sh.get("signed_available"):
        rng = np.random.RandomState(4); C = sh["d_c"].shape[0]
        V = (rng.rand(ctx.n, C).astype(np.float32) * 2 - 1)
        B = np.full((ctx.n, C), 1, np.int8); B[V > 0.3] = 0; B[V < -0.3] = 2
        Fm = rng.rand(ctx.n, C) > 0.15; B[~Fm] = 3
        sh["conc_value_signed"] = V; sh["conc_band_signed"] = B; sh["conc_fold_signed"] = B != 3
        sh["conc_answerable"] = rng.rand(ctx.n, C) > 0.3; sh["signed_available"] = True
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    z0 = fold_items_frozen(ctx.enc, [EMPTY])
    C = sh["d_c"].shape[0]
    net = ConceptFoldNet(C, d=ctx.d, h=16, conc_init=sh["d_c"].numpy())
    # concept + item fold helpers run
    acc = concept_ndcg(net, ctx, z0, rows, {r: [(0, 1.0)] for r in rows}, ks=(10,))
    assert np.isfinite(np.nanmean(acc["full@10"]))
    iacc = item_ndcg(ctx, rows, {r: ctx.allb[r] for r in rows}, ks=(10,))
    assert np.isfinite(np.nanmean(iacc["full@10"]))
    print("[SMOKE] concept_ndcg + item_ndcg PASS")
    # D3 trees (tiny bank)
    tc = _tree_concept(ctx, sh, rows, z0, net, bank_n=min(6, C))
    ti = _tree_item(ctx, sh, rows, bank_n=6)
    assert "adaptive_full@10" in tc and "adaptive_full@10" in ti
    print(f"[SMOKE] D3 concept tree q1={tc['q1']} adapt={tc['adaptive_full@10']:.3f} "
          f"static={tc['static_full@10']:.3f}; item tree q1={ti['q1']}")
    print("[SMOKE] COMPLETE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--best_ckpt", default=None, help="override; else auto-select from POC json")
    ap.add_argument("--wait_marker", default=None, help="poll until this file exists before starting")
    ap.add_argument("--full_threads", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    elif args.run:
        run(args)
    else:
        ap.error("one of --smoke / --run required")


if __name__ == "__main__":
    main()
