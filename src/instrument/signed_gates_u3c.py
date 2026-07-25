r"""signed_gates_u3c.py -- U3c part 2: scfull tower gates + THE CONSOLIDATED VERDICT BLOCK
(author directive 2026-07-25; Fable low-token thereafter).

SCFULL section (the signed C-full tower, cfull_signed_best.pt):
  G0  identity: empty-set decode bit-exact == frozen decoder bias (the g(0)=0 gate property);
      tie: ckpt val full@10 vs the native-anchored init 0.3510 within G0_TIE_CI (select_cold enforced
      this during training; re-reported here).
  G3a-c FLIP on SIGNED evidence (real dislike-band flips both directions) through tower tokens.
  G6-c  wrong-user / shuffled-values / placebo(v=1); DUPLICATES = blocked BY CONSTRUCTION (the
        pack_tokens dedup assert hard-fails on a repeated token -- verified firing).
  G8-c  order: sum-pool permutation invariance BY CONSTRUCTION (verified numerically, 1 permutation).

CONSOLIDATION (reads the U3/U3b/U3c artifacts; missing => PENDING):
  (1) clite_gates_signed.json          (sclite flip/G6/G8 on signed evidence)
  (2) tradeoff_ledger.json             (sclite/scfull g5_split: member-AUC, pop-projection, discrim.)
  (3) signed_sel_gate_signed_retrain.json  KT-A3 leak vs the PRE-REGISTERED <=0.05 bar --
      reported as GATE FAIL when above (author-ruling pending; the counterfeit evidence is attached
      as the waiver-consideration note, NOT as a softening)
  (4) this file's scfull section
-> experiments/battery/signed_gates_consolidated.json + one labeled verdict block printed AND
appended to signed_queue.log. HARD STOP after -- no certification retrain.
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

torch.set_num_threads(int(_NT))

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)

from train_tower_t2 import (log, I25Encoder, pack_tokens, level_to_sv, G0_TIE_CI)
import run_battery_phaseA as PA
from run_battery_phaseA import (build_real_ctx, build_smoke_ctx, ndcg10_from_scores, bootstrap_ci,
                                load_genome, OUTDIR, SEED)

assert not hasattr(sys.modules[__name__], "load_answerer")

BATT = os.path.join(_ROOT, "experiments", "battery")
NATIVE_INIT_FULL = 0.3510          # the native-anchored init val full@10 (cfull_signed_train.log)
M_EV = 4
LEAK_BAR = 0.05                    # pre-registered (DESIGN_SIGNED_CONCEPTS acceptance gate 3)


def fold_tokens(enc, tok_lists, ni, batch=256):
    """Concept-token tower folds. tok_lists: [(cids, levels)]."""
    Z = torch.zeros(len(tok_lists), enc.d_out)
    enc.eval()
    with torch.no_grad():
        for st in range(0, len(tok_lists), batch):
            rows = []
            for cids, lvls in tok_lists[st:st + batch]:
                ids = (ni + np.asarray(cids, np.int64)) if len(cids) else np.empty(0, np.int64)
                lv = np.asarray(lvls, np.int64)
                rows.append((ids, lv, level_to_sv(lv)))
            ids, vals, pad, lvs = pack_tokens(rows)
            Z[st:st + len(rows)] = enc(ids, vals, pad, lvs)
    return Z


def score(ctx, Z, rows):
    full = np.full(len(rows), np.nan)
    for st in range(0, len(rows), 500):
        ch = rows[st:st + 500]
        S = (Z[st:st + len(ch)] @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
        f, _ = ndcg10_from_scores(S, ctx.va_tr[ch], ctx.va_te[ch], None)
        full[st:st + len(ch)] = f
    return full


def scfull_gates(ctx, args, smoke=False):
    from signed_answers import (load_prereg, signed_values, cap_negatives, value_to_level,
                                BAND_REFUSE)
    from concept_fold import build_member_matrix
    members = load_genome(ctx)
    tags = sorted(members.keys())
    if smoke:
        native = ctx.enc._native[0]
        enc = I25Encoder(ctx.ni, native, d_lat=ctx.d, token_mode="film", n_concepts=len(tags))
        with torch.no_grad():
            for p in enc.rho[-1].parameters():
                p.add_(torch.randn_like(p) * 0.05)
        blob = {"val_full": NATIVE_INIT_FULL + 0.001, "epoch": 1}
        prereg = {"w_val": 0.3, "t_like_p60pos": 0.15, "t_neg_absp25": 0.1, "C_NEG": 2.0}
        item_mean = np.full(ctx.ni, 3.5, np.float32)
    else:
        blob = torch.load(args.scfull_ckpt, map_location="cpu")
        native = ctx.enc._native[0]
        enc = I25Encoder(ctx.ni, native, d_lat=ctx.d, token_mode="film", n_concepts=len(tags))
        enc.load_state_dict(blob["enc"])
        prereg, item_mean = load_prereg()
    enc.eval()
    dec_b = ctx.bd
    res = {"ckpt": os.path.basename(getattr(args, "scfull_ckpt", "smoke")),
           "epoch": blob.get("epoch"), "val_full": blob.get("val_full")}
    # ---- G0 identity + tie ----
    with torch.no_grad():
        z_e = enc(torch.zeros((1, 1), dtype=torch.long), torch.zeros((1, 1)),
                  torch.ones((1, 1), dtype=torch.bool), torch.zeros((1, 1), dtype=torch.long))
        sc_e = (z_e @ ctx.Wd.T + dec_b)[0]
    res["g0_identity_empty_eq_bias"] = bool(torch.equal(sc_e, dec_b))
    vf = blob.get("val_full")
    res["g0_tie"] = {"val_full": vf, "native_init": NATIVE_INIT_FULL, "ci": G0_TIE_CI,
                     "PASS": bool(vf is not None and vf >= NATIVE_INIT_FULL - G0_TIE_CI)}
    # ---- signed evidence (top-4 |v|) ----
    Mm = build_member_matrix(members, tags, ctx.ni)
    pexp = (ctx.cnt @ np.asarray(Mm.todense())) / max(ctx.cnt.sum(), 1e-9)
    items_l = [np.asarray(s, np.int64) for s, l in ctx.allb]
    stars_l = [((np.asarray(l, np.float64) + 1) / 2).astype(np.float32) for s, l in ctx.allb]
    V, F, B, ans = signed_values(items_l, stars_l, item_mean, Mm, pexp, ctx.ni, prereg,
                                 apply_neg_cap=False)
    sel = {}
    for r in range(ctx.n):
        cand = np.flatnonzero(ans[r] & (B[r] != BAND_REFUSE))
        if len(cand) >= 2:
            o = cand[np.argsort(-np.abs(V[r, cand]))][:M_EV]
            sel[r] = (o, cap_negatives([float(V[r, c]) for c in o]))
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0 and r in sel]
    def toks_from(vals_by_row):
        return [(sel[r][0], [value_to_level(v) for v in vals_by_row[r]]) for r in rows]
    true_v = {r: sel[r][1] for r in rows}
    f_true = score(ctx, fold_tokens(enc, toks_from(true_v), ctx.ni), rows)
    f_int = score(ctx, fold_tokens(enc, [(np.empty(0, np.int64), [])] * len(rows), ctx.ni), rows)
    # FLIP (real two-direction: dislike-band answers flip up)
    flip_v = {r: [-v for v in sel[r][1]] for r in rows}
    f_flip = score(ctx, fold_tokens(enc, toks_from(flip_v), ctx.ni), rows)
    d = bootstrap_ci(f_flip - f_true)
    n_dis = sum(1 for r in rows for v in true_v[r] if v < 0)
    res["g3a_flip_signed"] = {"delta": d[0], "ci95": d[1],
                              "PASS": bool(d[0] < 0 and d[1][1] < 0),
                              "dislike_band_answers": int(n_dis)}
    # wrong-user
    rl = list(rows)
    wrong_tok = [(sel[rl[(i - 1) % len(rl)]][0],
                  [value_to_level(v) for v in sel[rl[(i - 1) % len(rl)]][1]])
                 for i, r in enumerate(rl)]
    f_wrong = score(ctx, fold_tokens(enc, wrong_tok, ctx.ni), rows)
    dw = bootstrap_ci(f_wrong - f_int)
    res["g6_wrong_user"] = {"delta_vs_intercept": dw[0], "ci95": dw[1],
                            "PASS_no_gain": bool(dw[0] <= 0 or dw[1][0] <= 0)}
    # shuffled values + placebo
    rng = np.random.default_rng(SEED)
    shuf_v = {}
    for r in rows:
        vv = list(true_v[r]); p = rng.permutation(len(vv))
        shuf_v[r] = [vv[i] for i in p]
    f_shuf = score(ctx, fold_tokens(enc, toks_from(shuf_v), ctx.ni), rows)
    ds = bootstrap_ci(f_true - f_shuf)
    plac_v = {r: [1.0] * len(true_v[r]) for r in rows}
    f_plac = score(ctx, fold_tokens(enc, toks_from(plac_v), ctx.ni), rows)
    dp = bootstrap_ci(f_true - f_plac)
    res["g6_shuffled"] = {"true_minus_shuffled": ds[0], "ci95": ds[1]}
    res["g6_placebo_const1"] = {"true_minus_placebo": dp[0], "ci95": dp[1]}
    # duplicates: blocked BY CONSTRUCTION (dedup assert) -- verify it fires
    try:
        pack_tokens([(np.array([ctx.ni + 1, ctx.ni + 1]), np.array([9, 9]),
                      level_to_sv(np.array([9, 9])))])
        res["g6_duplicates"] = {"blocked_by_construction": False}
    except AssertionError:
        res["g6_duplicates"] = {"blocked_by_construction": True}
    # order: sum-pool permutation invariance (verified, 1 permutation)
    perm_v = {r: true_v[r] for r in rows}
    perm_tok = []
    rngp = np.random.default_rng(7)
    for r in rows:
        p = rngp.permutation(len(sel[r][0]))
        perm_tok.append((np.asarray(sel[r][0])[p], [value_to_level(true_v[r][i]) for i in p]))
    Zp = fold_tokens(enc, perm_tok, ctx.ni)
    Zt = fold_tokens(enc, toks_from(true_v), ctx.ni)
    res["g8_order_max_dz"] = float((Zp - Zt).norm(dim=1).max())
    res["g8_order_invariant"] = bool(res["g8_order_max_dz"] < 1e-4)
    res["intercept_full"] = float(np.nanmean(f_int)); res["true_full"] = float(np.nanmean(f_true))
    res["PASS"] = bool(res["g3a_flip_signed"]["PASS"] and res["g6_wrong_user"]["PASS_no_gain"]
                       and res["g6_duplicates"]["blocked_by_construction"]
                       and res["g8_order_invariant"] and res["g0_identity_empty_eq_bias"])
    return res


def consolidate(scfull_res, smoke=False):
    outdir = os.path.join(BATT, "_smoke") if smoke else BATT
    os.makedirs(outdir, exist_ok=True)
    def load(p):
        fp = os.path.join(outdir, p)
        return json.load(open(fp)) if os.path.exists(fp) else None
    cl = load("clite_gates_signed.json")
    led = load("tradeoff_ledger.json")
    gate = load("signed_sel_gate_signed_retrain.json")
    block = {"U3c_consolidated_verdicts": True, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    # (1) sclite gates
    block["sclite_gates"] = ({"flip": cl["g3a_flip"], "wrong_user": cl["g6_wrong_user"],
                              "duplicates": cl["g6_duplicates"], "order": cl["g8_order"],
                              "PASS": cl["PASS"]} if cl else "PENDING")
    # (2) G5 split from the ledger
    g5 = {}
    for rn in ("sclite", "scfull"):
        try:
            g5[rn] = led["rungs"][rn]["per_answer"]["g5_split"]
        except Exception:
            g5[rn] = "PENDING"
    block["g5_split"] = g5
    # (3) KT-A3 leak vs the PRE-REGISTERED bar -- report FAIL plainly; waiver note attached
    if gate:
        r2 = gate["kt_a3_leak_probe"]["ridge_R2_logvol_from_q8_latent"]
        r2s = r2.get("arm1_signed")
        cf = gate["decision"]
        block["kt_a3_leak_gate"] = {
            "R2_signed": r2s, "R2_clipup": r2.get("clipup"), "bar": LEAK_BAR,
            "VERDICT": "GATE FAIL (pending author ruling)" if (r2s is not None and r2s > LEAK_BAR)
                       else "PASS",
            "waiver_consideration_note": {
                "counterfeit_gain_fraction": (cf["arm2_gain_q16"] / cf["arm1_gain_q16"]
                                              if abs(cf.get("arm1_gain_q16", 0)) > 1e-9 else None),
                "note": "the popularity counterfeit reproduces ~none of the signed gain (taste-real);"
                        " attached for the author's waiver consideration, NOT as a softening"}}
    else:
        block["kt_a3_leak_gate"] = "PENDING"
    # (4) scfull
    block["scfull_gates"] = scfull_res
    path = os.path.join(outdir, "signed_gates_consolidated.json")  # smoke -> _smoke/
    json.dump(block, open(path, "w"), indent=2, default=float)
    # one labeled verdict block -> stdout AND the queue log
    lines = ["", "=== U3c CONSOLIDATED VERDICT BLOCK (signed retrained modules) ==="]
    sc = block["sclite_gates"]
    lines.append(f"[sclite gates]  {'PENDING' if sc == 'PENDING' else ('PASS' if sc['PASS'] else 'FAIL')}"
                 + ("" if sc == "PENDING" else f"  flip {sc['flip']['delta']:+.4f}"))
    for rn in ("sclite", "scfull"):
        g = g5[rn]
        lines.append(f"[{rn} G5-split]  " + ("PENDING" if g == "PENDING" else
                     f"memberAUC m1={g['member_AUC']['1']:.3f} | pop-proj "
                     f"{g['pop_projection_control']['delta']:+.4f} "
                     f"(beats={g['pop_projection_control']['beats_pop_proj']}) | disc "
                     f"pop={g['delta_discriminator']['spearman_vs_log_pop']:.3f}/"
                     f"mem={g['delta_discriminator']['spearman_vs_memberness']:.3f}"))
    ka = block["kt_a3_leak_gate"]
    lines.append("[KT-A3 leak]    " + ("PENDING" if ka == "PENDING" else
                 f"{ka['VERDICT']}  R2={ka['R2_signed']:.3f} vs bar {LEAK_BAR} "
                 f"(clip-up {ka['R2_clipup']:.3f}; counterfeit fraction "
                 f"{ka['waiver_consideration_note']['counterfeit_gain_fraction']}"))
    sf = scfull_res
    lines.append(f"[scfull gates]  {'PASS' if sf['PASS'] else 'FAIL'}  G0 id={sf['g0_identity_empty_eq_bias']} "
                 f"tie={sf['g0_tie']['PASS']} | flip {sf['g3a_flip_signed']['delta']:+.4f} "
                 f"({sf['g3a_flip_signed']['dislike_band_answers']} dislike-band) | dup-blocked="
                 f"{sf['g6_duplicates']['blocked_by_construction']} | order dz={sf['g8_order_max_dz']:.2e}")
    lines.append("=== HARD STOP: no certification retrain; author rules from this block + the "
                 "U3/U3b artifacts ===")
    text = "\n".join(lines)
    print(text)
    if not smoke:
        with open(os.path.join(BATT, "signed_queue.log"), "a") as f:
            f.write(text + "\n")
    log(f"[out] -> {path}")
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--scfull_ckpt", default=os.path.join(_ROOT, ".cache", "instrument",
                                                          "cfull_signed_best.pt"))
    ap.add_argument("--full_threads", action="store_true")
    args = ap.parse_args()
    ctx = build_smoke_ctx() if args.smoke else build_real_ctx(args.snapshot)
    ctx.Wd = ctx.decoder.weight.detach().float(); ctx.bd = ctx.decoder.bias.detach().float()
    ctx.d = ctx.enc.d_out
    if args.smoke:
        # smoke needs fake signed machinery inside scfull_gates (handled by smoke flag)
        res = scfull_gates(ctx, args, smoke=True)
        consolidate(res, smoke=True)
        print("[SMOKE] U3c paths complete")
        return
    if not os.path.exists(args.scfull_ckpt):
        log("scfull ckpt missing -- consolidating with PENDING scfull")
        consolidate({"PASS": False, "note": "PENDING (no ckpt)", "g0_identity_empty_eq_bias": False,
                     "g0_tie": {"PASS": False}, "g3a_flip_signed": {"delta": float("nan"),
                     "dislike_band_answers": 0}, "g6_duplicates": {"blocked_by_construction": False},
                     "g8_order_max_dz": float("nan")})
        return
    res = scfull_gates(ctx, args)
    consolidate(res)


if __name__ == "__main__":
    main()
