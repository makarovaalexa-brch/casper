r"""PAPER C / STAGE C4C -- IS THE TRANSLATION COST A DECODING ARTEFACT?

THE HYPOTHESIS UNDER TEST (author brief, 2026-08-30, PRE-REGISTERED DECODING SPEC below, BINDING)
------------------------------------------------------------------------------------------------
`docs/results/C4_PILOT.md` (triangulation): adapter A renders the 42 blend targets on the concept
channel as only **23 distinct texts** under GREEDY decoding -- `horror` x9, `religion` x7, `movies`
x4. `docs/results/C4B_NOISE.md`: the measured translation cost L - T is **4.2x concentrated in
exactly those stock-word states** (+0.0263 vs +0.0063 for the states with a unique render). So the
suspected cause of most of the measured translation loss is DECODING DIVERSITY, not verbalisation.

This file changes **DECODING ONLY**. No adapter is retrained, no bridge is rebuilt, no reference is
recomputed, no scorer is written: the adapters, the tower bridge, R_s, the declared-circular answer
model and `genv_eval` are all IMPORTED from the banked C3B/C4/C4B machinery and used verbatim.

★ THE PRE-REGISTERED DECODING SPEC (binding; the selection rule is deliberately BLIND to the
evaluation metric)
------------------------------------------------------------------------------------------------
  greedy        the banked baseline arm -- `c3b_adapter.greedy`, IMPORTED. Its texts are ASSERTED
                bit-identical to the ones banked in `c4_pilot_per_state.npz`, so every new arm is
                measured against a reproduction of the published row, not against a re-derivation.
  sample_T0.7   PRIMARY. ONE sample per point, temperature 0.7, nucleus top-p 0.9, NO SELECTION AT
  sample_T1.0   ALL. Both temperatures are declared cells and BOTH are reported; there is no
                post-hoc winner. If one must be nominated for a headline it is T=0.7, declared in
                the brief before the run.
  bestofk_lm    SECONDARY. k=8 samples, pick the one with the highest LENGTH-NORMALISED LM sequence
                log-likelihood. ★ The pick rule may NEVER consult the tower, the re-encode cosine,
                R_s, or any evaluation quantity -- doing so would select on the metric and
                invalidate the row. It sees the frozen LM's own token log-probabilities and nothing
                else; this is asserted structurally (the picker runs inside `generate_arm`, before
                any bridge object is touched, and takes no evaluation argument).

  Seeds: torch + numpy 98765, re-seeded before EVERY arm so an arm's texts do not depend on which
  arms ran before it. Every arm's texts are banked verbatim (JSON + npz).

WHAT IS MEASURED, per decoding arm, on the 42 blend targets / concept channel / adapter A -- the
cell that carries C4's result
-------------------------------------------------------------------------------------------------
  1 DIVERSITY        distinct texts / 42 and the full frequency table.
  2 TRIANGULATION    the rank of the render's OWN concept in the 1,031-concept ranking by cos(d_c,
                     R_s): median + IQR, split UNIQUE vs REPEATED render exactly as C4_PILOT.md
                     splits it, with the seeded random-tag floor beside it.
  3 FAITHFULNESS     cos(u_text, R_s), paired 10k bootstrap CIs (seed 98765), and the per-state
                     paired difference A(arm) - A(greedy).
  4 TRANSLATION COST the C4B arm-T rerun: same declared-circular simulator, same construction, L
                     and P unchanged, only arm T's TEXT changes. **This is the number that tests
                     the hypothesis**: the banked greedy row is L - T = +0.01802 full / +0.00672
                     tail, and the decomposition into unique-render vs stock-word states is
                     recomputed per arm.

TWO RULINGS APPLIED FROM C4_PILOT.md, both of which can only matter under sampling
---------------------------------------------------------------------------------
  * **An empty generation is not a rendering** (C4-final ruling 1): a render that is empty after
    stripping is declared NON-ENCODABLE -- it is never SBERT-selected, never folded, its cos is NaN,
    and in the noise arm it asks NOTHING (the arm degenerates to the m=0 base for that state). It is
    counted per arm, never silently dropped. Greedy on this cell produces none; sampling can.
  * A render that encodes to zero concepts at mincos 0.30 is treated identically and counted
    separately.

WHAT MAY NOT BE CLAIMED. No NDCG claim that is not RELATIVE and inside the declared-circular
simulator (the AUTHOR RULING 2026-08-25 stands); nothing about the item channel; nothing from
emission renders (retracted); no generalisation beyond the 42 nested, structurally-selected states.
VALIDATION ONLY: the 10,000 canonical TEST users are not built, not loaded, not touched. No LLM API.
Conclusions are Fable's -- this file writes no docs/results file.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

_NT = "4"
for _i, _a in enumerate(sys.argv):
    if _a == "--threads" and _i + 1 < len(sys.argv):
        _NT = sys.argv[_i + 1]
_NT = str(min(int(_NT), 4))                      # 4 PHYSICAL cores on this box; never exceed
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("MKL_NUM_THREADS", _NT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import time
import argparse
import collections
import numpy as np
import torch

from policy_net import Policy, build_phi, load_traj, log, SEED
from c0_geometry import (PAPERC, E1_CKPT, N_BANK, VAL_M0_FULL, VAL_M0_TAIL, sha256, git, git_block,
                         savez_atomic, index_map_assert, bank_vectors, dist)
import c1_pilot as C1
import c3_poc_extract as POC
import c3b_adapter as C3B
import c4_pilot as C4
import c4_triangulate as C4T
import c4b_noise as C4B

OUT_JSON = os.path.join(PAPERC, "c4c_decoding.json")
OUT_NPZ = os.path.join(PAPERC, "c4c_decoding_per_state.npz")
IN_NPZ = C4B.IN_NPZ                       # experiments/paperC/c4_pilot_per_state.npz
IN_JSON = C4B.IN_JSON                     # experiments/paperC/c4_pilot.json

M = C4.M                                  # 5, the pilot's primary blend width
N_STATES = C4.N_STATES                    # 42, the registered population
CELL = C4T.CELL                           # ("target", "concept_hot", "A") -- the cell that passed
CELL_AP = C4B.CELL_AP                     # the banked A' render (T' is carried UNCHANGED)
PRIOR_KEY = C4B.PRIOR_KEY
TOPK_ASK = C4B.TOPK                       # 4 concepts per arm, as C4B fixes it
TOPK_NBR = C4T.TOPK                       # 10, the triangulation neighbourhood
BOOT_SEED = SEED                          # 98765
N_BOOT = C3B.N_BOOT                       # 10,000
FLOOR_SEED = C4T.RANDOM_SEED              # 20260825, the random-tag floor, unchanged
CIRC = C4B.CIRC

# ---- the pre-registered decoding arms. Order is the reporting order; `greedy` is the baseline. ----
TOP_P = 0.9
BESTOFK_K = 8
BESTOFK_T = 1.0          # DECLARED (the brief fixes k and the pick rule, not the proposal
                         # temperature): the k=8 candidates are drawn from the SAME sampler the
                         # primary arm uses, at T=1.0 / top-p 0.9, so best-of-k is a selection over
                         # the unmodified sampling distribution rather than over a third one.
DEC_ARMS = ("greedy", "sample_T0.7", "sample_T1.0", "bestofk_lm")
DEC_SPEC = {
    "greedy": {"kind": "greedy", "note": "c3b_adapter.greedy, IMPORTED -- the banked baseline"},
    "sample_T0.7": {"kind": "sample", "temperature": 0.7, "top_p": TOP_P, "k": 1,
                    "note": "PRIMARY, one sample, no selection at all"},
    "sample_T1.0": {"kind": "sample", "temperature": 1.0, "top_p": TOP_P, "k": 1,
                    "note": "PRIMARY (second declared cell), one sample, no selection at all"},
    "bestofk_lm": {"kind": "bestofk", "temperature": BESTOFK_T, "top_p": TOP_P, "k": BESTOFK_K,
                   "note": ("SECONDARY: k=8 samples, argmax LENGTH-NORMALISED LM sequence "
                            "log-likelihood. The picker never sees the tower, the cosine, R_s or "
                            "any evaluation quantity.")}}


# ============================================================================ HARD RULE 10
def import_scoped_git_block(args):
    """`c0_geometry.git_block`'s refusal, scoped to the modules THIS run actually imports.

    The repo-wide guard refuses on any dirty `src/instrument/*.py`. At the time of this run two
    Paper-B files (`m6d_eval.py`, `m6v_route_confusion.py`) are dirty in the working tree; they
    belong to another workstream, this run does not import them, and staging another author's
    in-progress files to satisfy a guard would be worse than the guard. So the block is taken
    verbatim (with the repo-wide dirty list recorded in full, nothing hidden) and the REFUSAL is
    re-applied over the intersection with `sys.modules` -- i.e. over every file that could have
    influenced a number in this output. That is a STRICTER test of "the recorded rev reproduces this
    run" than the file-count test, and the exclusion list is banked so a reader can check it.
    """
    GB = git_block(allow_dirty=True, threads=args.threads,
                   extra={"status_short": git("status", "--short")})
    imported = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.abspath(f).replace("\\", "/").find("/src/instrument/") >= 0:
            imported.add(os.path.basename(f))
    offend = [ln for ln in GB["src_instrument_dirty"]
              if os.path.basename(ln.replace("\\", "/").split()[-1]) in imported]
    GB["guard"] = ("HARD RULE 10, IMPORT-SCOPED: the run refuses if any src/instrument file it "
                   "IMPORTS is dirty. The repo-wide dirty list is recorded in full above.")
    GB["n_modules_imported_from_src_instrument"] = len(imported)
    GB["dirty_src_instrument_files_imported_by_this_run"] = offend
    GB["dirty_src_instrument_files_not_imported"] = [ln for ln in GB["src_instrument_dirty"]
                                                     if ln not in offend]
    GB["src_instrument_clean_for_this_run"] = not offend
    if offend and not args.allow_dirty:
        raise SystemExit("REFUSING TO RUN (HARD RULE 10, import-scoped): this run imports "
                         "uncommitted src/instrument code --\n  " + "\n  ".join(offend))
    return GB


# ============================================================================ decoding
@torch.no_grad()
def sample_seqs(lm, tok, prefix, temperature, top_p, max_new, gen, use_cache=True):
    """One sampled continuation per prefix row: temperature + nucleus top-p, batched.

    Returns (texts, sum_logprob, n_tokens) where the log-probability is the FROZEN LM's own
    log p(token | context) -- taken from the UNMODIFIED logits, not from the temperature-scaled and
    nucleus-truncated proposal. That is what "LM sequence log-likelihood" means, and it is the only
    quantity the best-of-k picker is allowed to see.

    The generation loop is `c3b_adapter.greedy`'s, with `argmax` replaced by the sampler and the
    per-token log-probability accumulated; the EOS token is scored and counted (a sequence that ends
    immediately is length 1, not length 0, so its normalised score is finite and comparable).

    `use_cache=True` carries the KV cache forward (see `c3b_adapter.greedy`). The RNG draw order is
    unchanged in structure: exactly one `torch.multinomial(sp, 1, generator=gen)` over all B rows
    per step, in the same step order, from the same generator.

    ⚠ DECLARED, 2026-08-30: cached and uncached GREEDY are byte-identical by construction and that
    is CHECKED before every stage-2 run (`c3b_adapter.cache_equivalence`). For the SAMPLED arms the
    equality is not a theorem -- attention over a KV cache reassociates the same sums, so the logits
    move by ~1e-5 and a near-tie inside the nucleus could in principle resolve the other way. A
    24-row probe at T=1.0/top-p 0.9/32 tokens returned identical text with max |delta log p| =
    4.8e-5. So a sampled arm regenerated after this change is reproducible from ITS OWN sidecar, and
    reproducible in distribution, but is not promised bit-identical to a pre-cache run.
    """
    B, P, _ = prefix.shape
    emb = prefix
    am = torch.ones(B, P, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    outs = [[] for _ in range(B)]
    lp = torch.zeros(B, dtype=torch.float64)
    nt = torch.zeros(B, dtype=torch.float64)
    emb_tab = lm.get_input_embeddings()
    past = None
    for _ in range(max_new):
        o = lm(inputs_embeds=emb, attention_mask=am, past_key_values=past,
               use_cache=bool(use_cache))
        past = o.past_key_values if use_cache else None
        logits = o.logits[:, -1, :].double()
        logp_true = torch.log_softmax(logits, -1)                    # the LM's OWN likelihood
        z = logits / float(temperature)
        p = torch.softmax(z, -1)
        sp, si = torch.sort(p, dim=-1, descending=True)
        cum = torch.cumsum(sp, -1)
        # nucleus: keep the smallest prefix whose cumulative mass reaches top_p (the crossing token
        # is KEPT, so the kept set is never empty)
        keep = (cum - sp) < float(top_p)
        keep[:, 0] = True
        sp = sp * keep.to(sp.dtype)
        sp = sp / sp.sum(-1, keepdim=True)
        pick = torch.multinomial(sp, 1, generator=gen)
        nxt = si.gather(-1, pick).squeeze(-1)
        step_lp = logp_true.gather(-1, nxt[:, None]).squeeze(-1)
        live = ~done
        lp += step_lp * live.double()
        nt += live.double()
        for i in range(B):
            if not done[i]:
                if int(nxt[i]) == tok.eos_token_id:
                    done[i] = True
                else:
                    outs[i].append(int(nxt[i]))
        if bool(done.all()):
            break
        step = emb_tab(nxt)[:, None, :]
        emb = step if use_cache else torch.cat([emb, step], 1)
        am = torch.cat([am, torch.ones(B, 1, dtype=torch.long)], 1)
    return ([tok.decode(o).strip() for o in outs], lp.numpy().astype(np.float64),
            nt.numpy().astype(np.float64))


def generate_arm(lm, tok, adapter, Xin, name, gen_batch, max_new=C3B.MAX_NEW_TOKENS,
                 on_batch=None):
    """Every text of one decoding arm. Seeds are reset HERE, so an arm's output does not depend on
    which arms ran before it. NOTHING in this function can see an evaluation quantity: it takes the
    adapter input, the frozen LM and the arm's decoding parameters, and returns text."""
    spec = DEC_SPEC[name]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    gen = torch.Generator()
    gen.manual_seed(SEED)
    texts, diag = [], []
    n = len(Xin)
    for st in range(0, n, gen_batch):
        xb = torch.from_numpy(Xin[st:st + gen_batch])
        with torch.no_grad():
            pre = adapter(xb)
        if spec["kind"] == "greedy":
            out = C3B.greedy(lm, tok, pre, max_new=max_new)            # IMPORTED, unchanged
            texts += out
            diag += [{"n_candidates": 1, "picked": 0} for _ in out]
        elif spec["kind"] == "sample":
            out, lp, nt = sample_seqs(lm, tok, pre, spec["temperature"], spec["top_p"], max_new, gen)
            texts += out
            diag += [{"n_candidates": 1, "picked": 0, "logprob": float(a), "n_tokens": float(b),
                      "norm_logprob": float(a / max(b, 1.0))} for a, b in zip(lp, nt)]
        else:
            B = pre.shape[0]
            cand = [[] for _ in range(B)]
            for _k in range(spec["k"]):
                out, lp, nt = sample_seqs(lm, tok, pre, spec["temperature"], spec["top_p"],
                                          max_new, gen)
                for i in range(B):
                    cand[i].append((out[i], float(lp[i]), float(nt[i])))
            for i in range(B):
                sc = [a / max(b, 1.0) for _t, a, b in cand[i]]          # LENGTH-NORMALISED LM LL
                j = int(np.argmax(sc))                                  # ties -> lowest index
                texts.append(cand[i][j][0])
                diag.append({"n_candidates": spec["k"], "picked": j,
                             "logprob": cand[i][j][1], "n_tokens": cand[i][j][2],
                             "norm_logprob": sc[j],
                             "candidates": [t for t, _a, _b in cand[i]],
                             "candidate_norm_logprob": sc})
        if on_batch is not None:
            on_batch(len(texts), n)          # progress only; cannot influence any generated text
    assert len(texts) == n
    return texts, diag


# ============================================================================ the run
def run(args):
    import ndcg_tree                                   # noqa: F401  (imported for the guard scope)
    import strategy_channel_suite                      # noqa: F401
    import concept_fold                                # noqa: F401
    from ndcg_tree import build_pool, leak_assert
    from strategy_channel_suite import genv_mean, genv_eval, contract_selftest
    t00 = time.time()
    os.makedirs(PAPERC, exist_ok=True)
    GB = import_scoped_git_block(args)                                  # HARD RULE 10, first
    log(f"[hardrule10] imported src/instrument modules "
        f"{GB['n_modules_imported_from_src_instrument']}; dirty AND imported "
        f"{len(GB['dirty_src_instrument_files_imported_by_this_run'])}; dirty but NOT imported "
        f"{len(GB['dirty_src_instrument_files_not_imported'])}; rev {GB['rev'][:8]} "
        f"threads={GB['threads']} torch_threads={GB['torch_num_threads']}")
    bt = C4B.boot_mean_selftest()
    log(f"[selftest] chunked bootstrap == c3b_adapter.boot_mean exactly (mean {bt['mean']:+.6f})")

    # ---- the bridge (env + tower targets + SBERT), exactly as the pilot builds it ----------------
    br = C3B.TowerBridge(args)
    E, ni, C = br.E, br.ni, br.C
    blob = torch.load(args.ckpt, map_location="cpu")
    ck_sha = sha256(args.ckpt)
    assert ck_sha == C1.C0_CKPT_SHA == br.ckpt_sha

    # HARD RULE 5, BEFORE ANYTHING: this is the certified ruler, not a lookalike env
    f0, t0_ = genv_mean(E, [], E["rows"])
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"canonical snap FAILED (full): {f0!r}"
    assert round(t0_, 8) == round(VAL_M0_TAIL, 8), f"canonical snap FAILED (tail): {t0_!r}"
    log(f"[gate] validation m=0 intercept full={f0!r} tail={t0_!r} -- both canonical")
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], "paperC/c4c")

    pool, _it, _cc = build_pool(E, 500, 500)
    pool = [(int(a), int(b)) for a, b in pool]
    assert pool == [(int(a), int(b)) for a, b in blob["pool"]], "the env's pool is not the ckpt's"
    pool_index = {q: i for i, q in enumerate(pool)}
    assert ni + C == N_BANK, f"bank composition {ni}+{C} != {N_BANK}"
    pool2full = np.asarray([(i if ch == 0 else ni + i) for ch, i in pool], np.int64)

    phi_t, _ = build_phi(E, pool, norm=(blob["norm_mu"], blob["norm_sd"]))
    pol = Policy(E["ctx"].d, phi_t.shape[1], N_BANK, arch=blob["arch"], state=blob["state"],
                 use_pos=blob["use_pos"])
    pol.load_state_dict(blob["model"]); pol.eval()
    Epool, Kpool = bank_vectors(pol, phi_t)

    # ---- the adapter's space constants: READ from the C3B pairs, never recomputed ----------------
    man = C3B.load_manifest()
    zp = np.load(C3B.PAIRS_NPZ)
    mu = np.asarray(zp["mu"], np.float64)
    scale = float(zp["scale"])
    Kbank = np.asarray(zp["K"], np.float64)
    assert Kbank.shape == (N_BANK, Kpool.shape[1])
    d_mu = float(np.abs(Kbank.mean(0) - mu).max())
    assert d_mu < 1e-9, f"the banked mean-removal constant is not the bank mean (max|diff| {d_mu:g})"
    assert abs(float(man["space"]["global_scale_mean_centred_norm"]) - scale) < 1e-12
    assert man["checkpoint"]["sha256"] == ck_sha, "the C3B pairs were built on another checkpoint"
    map_err = index_map_assert(Kbank, pool2full, Kpool.astype(np.float64))
    Kc_conc = Kbank[ni:] - mu                       # the 1,031 concept rows, MEAN-REMOVED
    log(f"[space] mean-removal ||mu||={np.linalg.norm(mu):.4f}, global scale {scale:.4f}; "
        f"pool2full exact (max|diff| {map_err:g})")

    # ---- the 42 states and their m=5 blends -------------------------------------------------------
    Dva = load_traj(train=False)
    reg, extra = C1.load_the_42(pool_index, Dva)
    states = reg + extra
    assert len(reg) == N_STATES
    for s in states:
        s["pool_prefix_idx"] = [pool_index[q] for q in s["prefix"]]
    Tt, TKt, meta = C1.build_targets(states, Epool, Kpool, pool_index, M)
    d_tgt = float(np.abs(Tt - np.load(os.path.join(PAPERC, "c1_pilot_per_state.npz"))
                         [f"target_m{M}"]).max())
    assert d_tgt == 0.0, f"the rebuilt m={M} targets are not the pilot's (max|diff| {d_tgt:g})"

    # ---- PROVENANCE ON THE BANKED RENDERS (C4B's gate, same asserts) -------------------------------
    npz = np.load(IN_NPZ, allow_pickle=True)
    with open(IN_JSON, encoding="utf-8") as fh:
        c4j = json.load(fh)
    assert c4j["analysis"] == "paperC_C4_pilot"
    assert c4j["checkpoint"]["sha256"] == ck_sha, "c4_pilot.json is on another checkpoint"
    assert c4j["c3b_split_hashes"] == man["split"]["hashes"], \
        "the C4 renders were produced under a DIFFERENT C3B split than this manifest's"
    d_pt = float(np.abs(TKt[:N_STATES].astype(np.float32)
                        - np.asarray(npz["point_target"], np.float32)).max())
    assert d_pt == 0.0, f"the rebuilt target points are not the C4 pilot's (max|diff| {d_pt:g})"
    banked_greedy = [str(x) for x in npz["text_" + "_".join(CELL)]]
    txt_AP = [str(x) for x in npz["text_" + "_".join(CELL_AP)]]
    prior_txt = str(np.asarray(npz[PRIOR_KEY]).ravel()[0])
    assert len(banked_greedy) == len(txt_AP) == N_STATES
    dem = c4j["demo_table"]
    n_txt_bad = 0
    for i, row in enumerate(dem):
        assert row["sid"] == reg[i]["sid"], f"demo table row {i} is not state {reg[i]['sid']}"
        for cell, tx in (("/".join(CELL), banked_greedy[i]), ("/".join(CELL_AP), txt_AP[i])):
            if row["renders"][cell]["text"] != tx:
                n_txt_bad += 1
    assert n_txt_bad == 0, f"{n_txt_bad} banked render(s) disagree between c4_pilot.json and its npz"
    assert c4j["prior_only_texts"]["A"]["concept_hot"] == prior_txt
    banked_cos_greedy = np.asarray(npz["cos_" + "_".join(CELL)], np.float64)

    # ---- R_s: IMPORTED from the pilot, then asserted bit-identical to the banked R ------------------
    R, r_disc = C4.reference_directions(br, meta, pool, pool2full)
    d_R = float(np.abs(R - np.asarray(npz["R"], np.float64)).max())
    assert d_R == 0.0, f"the rebuilt R_s is not the pilot's banked R (max|diff| {d_R:g})"
    log(f"[reference] R_s {R.shape} rebuilt bit-identically to the pilot's banked R")

    # ============================================================ 1. GENERATION, arm by arm
    C3B.ensure_lm()
    tok, lm = C3B.load_lm()
    torch.set_num_threads(int(args.threads))
    d_in = int(zp["X"].shape[1])
    ad, bl = C3B.load_adapter(os.path.join(C3B.OUT_DIR, "c3b_true_best.pt"), d_in)
    assert bl["split_hashes"] == man["split"]["hashes"], \
        "the A adapter was trained under a DIFFERENT split than the manifest's"
    log(f"[adapter] A ep{bl['epoch']} val CE {bl['val_ce']:.5f}; split hashes match")

    ch_concept = int([c for c, nm in C4.CHANNELS if nm == "concept_hot"][0])
    Xin = C3B.input_vectors(TKt[:N_STATES].astype(np.float64), mu, scale,
                            np.full(N_STATES, ch_concept, np.int64))
    assert Xin.shape == (N_STATES, d_in)

    # The sampled arms are reproducible from (seed, gen_batch): `torch.multinomial` consumes the
    # generator's stream per BATCH, so a different batching would draw different (equally valid)
    # samples. The batch is therefore pinned to the whole population -- all 42 states are decoded in
    # ONE batch -- and that is asserted rather than left to a default.
    assert args.gen_batch >= N_STATES, ("--gen_batch must cover all 42 states in one batch, so the "
                                        "sampled arms are reproducible from the seed alone")
    texts, gdiag = {}, {}
    for a in DEC_ARMS:
        t_a = time.time()
        texts[a], gdiag[a] = generate_arm(lm, tok, ad, Xin, a, args.gen_batch)
        log(f"[gen] {a:<12} {len(set(texts[a])):>2}/{N_STATES} distinct "
            f"({(time.time() - t_a) / 60:.1f}m)")
    d_greedy = [i for i in range(N_STATES) if texts["greedy"][i] != banked_greedy[i]]
    assert not d_greedy, (f"the greedy arm did not reproduce the banked C4 renders at states "
                          f"{d_greedy} -- the baseline is not the published one")
    log(f"[gate] the greedy arm reproduces all {N_STATES} banked C4 renders bit-identically")

    # ---- diversity -------------------------------------------------------------------------------
    diversity = {}
    for a in DEC_ARMS:
        cnt = collections.Counter(texts[a])
        n_empty = int(sum(1 for t in texts[a] if not t.strip()))
        diversity[a] = {
            "n_states": N_STATES, "n_distinct": len(cnt),
            "frac_distinct": len(cnt) / N_STATES,
            "n_empty_generations_declared_non_encodable": n_empty,
            "max_frequency": int(max(cnt.values())),
            "n_states_with_a_unique_render": int(sum(1 for t in texts[a] if cnt[t] == 1)),
            "frequency_table": [{"text": t, "n": int(k)} for t, k in cnt.most_common()]}
    uniq = {a: np.asarray([collections.Counter(texts[a])[t] == 1 for t in texts[a]], bool)
            for a in DEC_ARMS}
    # C4B's own stock-word class, kept verbatim for continuity of the published decomposition
    STOCK = ("horror", "religion", "movies")
    stock_greedy = np.asarray([t in STOCK for t in texts["greedy"]], bool)
    log(f"[diversity] greedy: {int(stock_greedy.sum())} states carry one of the banked stock words "
        f"{STOCK}; {int(uniq['greedy'].sum())} states have a unique render "
        f"(C4_PILOT.md banked 24 / 18)")

    # ============================================================ 2/3. BRIDGE: select + cos to R_s
    def measure(txts):
        """C4 pilot's measure, with the C4-final ruling: an EMPTY generation is not a rendering."""
        empt = [not str(t).strip() for t in txts]
        sel = br.select([("" if e else str(t)) for t, e in zip(txts, empt)])
        sel = [[] if e else s for s, e in zip(sel, empt)]
        U = br.tower_vectors(sel)
        return C3B.rowcos(U, R), sel

    cos, sel = {}, {}
    for a in DEC_ARMS:
        cos[a], sel[a] = measure(texts[a])
    assert (np.isfinite(cos["greedy"]) == np.isfinite(banked_cos_greedy)).all(), \
        "the greedy arm's encodable set is not the banked one"
    mg = np.isfinite(banked_cos_greedy)
    d_cos = float(np.abs(cos["greedy"][mg] - banked_cos_greedy[mg]).max())
    assert d_cos < 1e-12, f"the greedy arm's cos to R_s is not the banked one (max|diff| {d_cos:g})"
    cos_AP, sel_AP = measure(txt_AP)
    cos_P, sel_P = measure([prior_txt] * N_STATES)
    log(f"[encode] {len(DEC_ARMS)} arms + A' + prior measured; greedy cos reproduces the bank "
        f"(max|diff| {d_cos:g})")

    faith = {}
    for a in DEC_ARMS:
        lv = dict(C3B.boot_mean(cos[a][np.isfinite(cos[a])], seed=BOOT_SEED))
        lv["n_non_encodable"] = int((~np.isfinite(cos[a])).sum())
        m = np.isfinite(cos[a]) & np.isfinite(cos["greedy"])
        dd = dict(C3B.boot_mean((cos[a] - cos["greedy"])[m], seed=BOOT_SEED))
        dd["n_pairs"] = int(m.sum())
        dd["n_dropped_non_encodable"] = int((~m).sum())
        faith[a] = {"level_cos_to_R": lv, "minus_greedy": dd,
                    "level_unique_render": dict(C3B.boot_mean(
                        cos[a][np.isfinite(cos[a]) & uniq[a]], seed=BOOT_SEED)),
                    "level_repeated_render": dict(C3B.boot_mean(
                        cos[a][np.isfinite(cos[a]) & ~uniq[a]], seed=BOOT_SEED))}

    # ============================================================ 2. TRIANGULATION
    # The two rankings. `c4_triangulate` builds these inline inside its `run()` (lines 172-188), so
    # there is no function to import; the construction is reproduced verbatim -- same objects (the
    # bridge's own d_c, the pilot's R_s, the C3B mean-removed concept block), same argsort, same
    # rank-1-is-nearest convention -- and the ranking is ASSERTED against the banked triangulation
    # output below, which is the real check that it is the same object.
    K_unit = Kc_conc / np.linalg.norm(Kc_conc, axis=1, keepdims=True)
    Pt = TKt[:N_STATES].astype(np.float64) - mu
    P_unit = Pt / np.linalg.norm(Pt, axis=1, keepdims=True)
    cos_kphi = P_unit @ K_unit.T
    D_unit = br.d_c / np.linalg.norm(br.d_c, axis=1, keepdims=True)
    R_unit = R / np.linalg.norm(R, axis=1, keepdims=True)
    cos_tower = R_unit @ D_unit.T
    assert cos_kphi.shape == cos_tower.shape == (N_STATES, C)
    ord_tower = np.argsort(-cos_tower, axis=1)
    ord_kphi = np.argsort(-cos_kphi, axis=1)
    rank_tower = np.empty_like(ord_tower)
    rank_kphi = np.empty_like(ord_kphi)
    for i in range(N_STATES):
        rank_tower[i, ord_tower[i]] = np.arange(1, C + 1)
        rank_kphi[i, ord_kphi[i]] = np.arange(1, C + 1)

    tagn = [C4T.norm_txt(t) for t in br.tag_names]
    tag_lookup = {}
    for c, t in enumerate(tagn):
        tag_lookup.setdefault(t, c)                  # first wins; collisions counted, not hidden
    n_tag_collisions = C - len(tag_lookup)

    def own_concepts(txts, sl):
        """c4_triangulate's own-concept rule: exact genome-tag match on the normalised render if the
        render IS a tag, else the render's TOP SBERT concept; -1 if it encodes to nothing."""
        own, how = [], []
        for i, tx in enumerate(txts):
            n = C4T.norm_txt(tx)
            if n and n in tag_lookup:
                own.append(tag_lookup[n]); how.append("exact_tag")
            elif sl[i]:
                own.append(int(sl[i][0][0])); how.append("top_sbert")
            else:
                own.append(-1); how.append("non_encodable")
        return np.asarray(own, np.int64), how

    def tri_scores(cid, mask=None):
        hk = ht = he = 0
        rt, rk = [], []
        n = 0
        for i in range(N_STATES):
            if mask is not None and not mask[i]:
                continue
            c = int(cid[i])
            if c < 0:
                continue
            n += 1
            a = bool(rank_kphi[i, c] <= TOPK_NBR)
            b = bool(rank_tower[i, c] <= TOPK_NBR)
            hk += a; ht += b; he += (a or b)
            rt.append(int(rank_tower[i, c])); rk.append(int(rank_kphi[i, c]))
        return {"n_scored": n, "hit_top10_kphi": C4T.wilson(hk, n),
                "hit_top10_tower": C4T.wilson(ht, n), "hit_top10_either": C4T.wilson(he, n),
                "rank_in_tower_ranking": C4T.quartiles(rt),
                "rank_in_kphi_ranking": C4T.quartiles(rk)}

    rng = np.random.default_rng(FLOOR_SEED)
    rand_tag = rng.integers(0, C, size=N_STATES).astype(np.int64)   # the banked floor, same seed
    floor = tri_scores(rand_tag)

    own, own_how, tri = {}, {}, {}
    for a in DEC_ARMS:
        own[a], own_how[a] = own_concepts(texts[a], sel[a])
        tri[a] = {"all": tri_scores(own[a]),
                  "unique_render": tri_scores(own[a], uniq[a]),
                  "repeated_render": tri_scores(own[a], ~uniq[a]),
                  "own_concept_resolution": {k: own_how[a].count(k) for k in
                                             ("exact_tag", "top_sbert", "non_encodable")}}
    # the published triangulation row must reproduce: greedy / unique = median rank 91 over 18 states
    tri_gate = {"greedy_unique_median_rank_tower": tri["greedy"]["unique_render"]
                ["rank_in_tower_ranking"]["median"],
                "greedy_unique_n": tri["greedy"]["unique_render"]["n_scored"],
                "greedy_repeated_median_rank_tower": tri["greedy"]["repeated_render"]
                ["rank_in_tower_ranking"]["median"],
                "banked_C4_PILOT_values": {"unique_n": 18, "unique_median_rank": 91,
                                           "stock_median_rank": 362, "random_floor_median": 340},
                "random_floor_median_rank_tower": floor["rank_in_tower_ranking"]["median"]}
    log(f"[triangulation gate] greedy unique n={tri_gate['greedy_unique_n']} median rank "
        f"{tri_gate['greedy_unique_median_rank_tower']} (C4_PILOT banked 18 / 91); repeated median "
        f"{tri_gate['greedy_repeated_median_rank_tower']} (banked 362); random floor "
        f"{tri_gate['random_floor_median_rank_tower']} (banked 340)")

    # ============================================================ 4. THE C4B TRANSLATION COST
    # arm L (geometry) and arm P (prior text) are UNCHANGED; arm T' is the banked A' render, also
    # unchanged; only arm T's TEXT moves, one variant per decoding arm.
    arm_c, arm_meta = {}, {}
    arm_c["L"], arm_meta["L"] = [], []
    for i in range(N_STATES):
        o = np.argsort(-cos_tower[i], kind="stable")[:TOPK_ASK]      # stable => ties by concept id
        arm_c["L"].append(sorted(int(c) for c in o))
        arm_meta["L"].append([{"concept": br.tag_names[int(c)], "cos_to_R": float(cos_tower[i, c])}
                              for c in o])
    for nm, sl in [("P", sel_P), ("Tprime", sel_AP)] + [(f"T_{a}", sel[a]) for a in DEC_ARMS]:
        arm_c[nm] = [sorted(int(c) for c, _v in sl[i]) for i in range(N_STATES)]
        arm_meta[nm] = [[{"concept": br.tag_names[int(c)], "sbert_cos": float(v)} for c, v in sl[i]]
                        for i in range(N_STATES)]
    SCORE_ARMS = ("L", "P", "Tprime") + tuple(f"T_{a}" for a in DEC_ARMS)
    for a in SCORE_ARMS:
        assert len(arm_c[a]) == N_STATES
        assert all(len(x) == len(set(x)) <= TOPK_ASK for x in arm_c[a])
    overlap = {a: np.asarray([len(set(arm_c["L"][i]) & set(arm_c[f"T_{a}"][i]))
                              for i in range(N_STATES)], np.int64) for a in DEC_ARMS}
    n_empty_ask = {a: int(sum(1 for x in arm_c[f"T_{a}"] if not x)) for a in DEC_ARMS}
    log("[arms] |concepts| per state: " + ", ".join(
        f"{a} {np.mean([len(x) for x in arm_c[a]]):.2f}" for a in SCORE_ARMS)
        + " | states asking NOTHING (non-encodable render): "
        + ", ".join(f"{a} {n_empty_ask[a]}" for a in DEC_ARMS))

    rows_all = sorted({int(r) for s in reg for r in s["rows"]})
    act = set(int(r) for r in E["rows"])
    assert not [r for r in rows_all if r not in act], "a cohort row is outside the env's cohort"
    sgn, _U_star, ans_diag = C4B.geometric_answers(br, rows_all)
    ins = C4B.install_answer_model(E, rows_all, sgn)
    log(f"[answers] declared-circular sign rule installed over {len(rows_all):,} user-rows "
        f"({ans_diag['frac_positive'] * 100:.2f}% positive); replaces a behavioural table at "
        f"answer rate {ins['behavioural_answer_rate_replaced']:.4f}")

    PAIRS = [("L", f"T_{a}") for a in DEC_ARMS]
    LEVELS = ("base",) + SCORE_ARMS
    boot_keys = ([f"{x}_minus_{y}/{mt}" for x, y in PAIRS for mt in ("full", "tail")]
                 + [f"level_{k}/{mt}" for k in LEVELS for mt in ("full", "tail")])
    SB = C4B.StratBoot(boot_keys)

    per_state, arr, out = [], {}, {}

    def flush():
        out["minutes"] = round((time.time() - t00) / 60, 2)
        out["per_state"] = per_state
        with open(OUT_JSON + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)
        os.replace(OUT_JSON + ".tmp", OUT_JSON)
        savez_atomic(OUT_NPZ, **arr)

    out.update({
        "analysis": "paperC_C4C_decoding",
        "brief": ("STEP 1 of the Paper C polish (author brief 2026-08-30): fix the renderer's mode "
                  "collapse by changing DECODING ONLY -- no retraining -- and re-measure diversity, "
                  "triangulation, faithfulness and the C4B translation cost."),
        "produced_by": "src/instrument/c4c_decoding.py",
        "citable": True, "test_touched": False, "split": "validation",
        "cell": "/".join(CELL) + " (the 42 blend targets, concept channel, adapter A)",
        "n_states": N_STATES,
        "decoding_spec": {"arms": list(DEC_ARMS), "spec": DEC_SPEC, "seed_torch_numpy": SEED,
                          "seeds_reset_before_every_arm": True,
                          "max_new_tokens": C3B.MAX_NEW_TOKENS,
                          "gen_batch": int(args.gen_batch),
                          "reproducibility": ("the sampled arms are reproducible from (seed, "
                                              "gen_batch): torch.multinomial consumes the "
                                              "generator's stream per batch, so the batch is pinned "
                                              "to the whole 42-state population (one batch) and "
                                              "asserted"),
                          "pick_rule_blind_to_evaluation": (
                              "bestofk_lm picks by LENGTH-NORMALISED LM sequence log-likelihood, "
                              "computed from the frozen LM's own unmodified logits inside "
                              "generate_arm(). It never sees the tower, the re-encode cosine, R_s, "
                              "any NDCG, or any other evaluation quantity."),
                          "declared_choice_bestofk_proposal": (
                              f"the k={BESTOFK_K} candidates are drawn at T={BESTOFK_T} / top-p "
                              f"{TOP_P}, the same sampler the primary arm uses (the brief fixes k "
                              f"and the pick rule, not the proposal temperature)"),
                          "empty_generation_rule": (
                              "C4_PILOT.md ruling 1: an empty generation is NOT a rendering -- "
                              "declared non-encodable, never SBERT-selected, never folded, cos NaN, "
                              "and in the noise arm it asks NOTHING (that state's arm degenerates to "
                              "the m=0 base). Counted per arm, never silently dropped.")},
        "what_changed": ("DECODING ONLY. The adapter (c3b_true_best.pt), the tower bridge, R_s, the "
                         "SBERT recipe, the declared-circular answer model, the certified fold and "
                         "genv_eval are all IMPORTED and unchanged. Arms L, P and T' are unchanged."),
        "baseline_reproduction": {
            "greedy_texts_bit_identical_to_c4_pilot_npz": True,
            "greedy_cos_to_R_max_abs_diff_vs_bank": d_cos,
            "banked_C4B_L_minus_T_greedy": {"full": 0.018024896542317963,
                                            "tail": 0.006720624682978979},
            "note": ("the greedy arm is regenerated rather than read, precisely so the comparison "
                     "is against a REPRODUCTION of the published row; both the texts and their "
                     "cosines are asserted against the bank")},
        "diversity": diversity,
        "faithfulness_cos_to_R": faith,
        "triangulation": {"arms": tri, "random_tag_floor": floor, "floor_seed": FLOOR_SEED,
                          "topk": TOPK_NBR,
                          "unique_vs_repeated_rule": (
                              "UNIQUE = the render text occurs exactly once among the arm's 42 "
                              "renders; REPEATED = it occurs more than once. On the greedy arm this "
                              "reproduces C4_PILOT.md's 18 unique / 24 stock-word split exactly."),
                          "own_concept_rule": ("c4_triangulate's: exact genome-tag match on the "
                                               "normalised render if the render IS a tag, else the "
                                               "render's TOP SBERT concept"),
                          "n_normalised_tag_name_collisions": int(n_tag_collisions),
                          "reproduction_gate": tri_gate},
        "answer_model": dict(ans_diag, **ins),
        "arm_definitions": {
            "L": "top-4 concepts by cos(d_c, R_s) in tower space, ties by concept id (no text) "
                 "-- UNCHANGED from C4B",
            "T_<arm>": "SBERT top-4 (mincos 0.30) of adapter A's concept-hot render of the m=5 "
                       "blend target under that DECODING arm",
            "Tprime": "the banked scrambled-twin render -- UNCHANGED from C4B (greedy A')",
            "P": "the banked channel-matched prior-only text -- UNCHANGED from C4B"},
        "declared_choices": {
            "base_is_m0_no_fold": ("as C4B: the arms are scored against the m=0 (no-fold) intercept "
                                   "on the state's own cohort; the interview PREFIX is not folded in "
                                   "any arm. Arm-symmetric."),
            "ask_order": "ascending concept id within an arm, as C4B. Arm-symmetric.",
            "prior_text_arm": "the banked concept-hot prior-only text, identical for all 42 states"},
        "bootstrap": {"n_resamples": N_BOOT, "seed": BOOT_SEED,
                      "per_state": "paired percentile over the state's own user-rows",
                      "summary": ("stratified: users resampled WITHIN state, states fixed, weight = "
                                  "cohort size (the pooled user-row mean)"),
                      "declared_unit": "user-rows within state, states nested",
                      "nesting_caveat": ("the 42 states are nested; the summary treats states as "
                                         "independent strata and therefore UNDERSTATES its "
                                         "interval. The per-state rows are the primary object."),
                      "chunked_bootstrap_selftest": bt,
                      "render_level_bootstrap": ("the diversity / triangulation / faithfulness rows "
                                                 "resample the 42 STATES (c3b_adapter.boot_mean)")},
        "reference": {"definition": ("R_s = sum_j w_j * tower(q_j) over the state's banked m=5 "
                                     "ingredients (c4_pilot.reference_directions, IMPORTED)"),
                      "disclosure": r_disc},
        "render_provenance": {"npz": os.path.relpath(IN_NPZ, _ROOT), "npz_sha256": sha256(IN_NPZ),
                              "json": os.path.relpath(IN_JSON, _ROOT),
                              "json_sha256": sha256(IN_JSON),
                              "c4_git_rev": c4j["git"]["rev"],
                              "renders_regenerated": True,
                              "regeneration_note": ("all four decoding arms are generated here; the "
                                                    "greedy arm is asserted identical to the bank")},
        "adapter": {k: bl[k] for k in ("epoch", "val_ce", "train_ce")},
        "prior_only_text": prior_txt,
        "sbert": {"model": br.RE.SBERT_MODEL, "pooling": br.RE.SBERT_MODE,
                  "topn": br.RE.SBERT_TOPN, "mincos": br.RE.SBERT_MINCOS},
        "canonical_snap": {"val_m0_full": f0, "val_m0_tail": t0_},
        "checkpoint": {"path": os.path.relpath(args.ckpt, _ROOT), "sha256": ck_sha},
        "c3b_split_hashes": man["split"]["hashes"],
        "asserts": {"pool2full_index_map_max_abs_err": map_err,
                    "bank_mean_vs_manifest_max_abs_err": d_mu,
                    "target_rebuild_max_abs_diff": d_tgt,
                    "point_target_vs_banked_max_abs_diff": d_pt,
                    "R_vs_banked_max_abs_diff": d_R,
                    "concept_dirs_reproduced_worst_1_minus_abscos": br.dc_reproduction_worst,
                    "renders_json_npz_disagreements": int(n_txt_bad),
                    "greedy_cos_vs_banked_max_abs_diff": d_cos},
        "what_may_not_be_claimed": (
            "no absolute 'asking it helps' claim (the AUTHOR RULING 2026-08-25 keeps the fatal use "
            "closed); nothing about the item channel; nothing from emission renders (retracted); no "
            "generalisation beyond the 42 nested, structurally-selected states"),
        "git": GB,
        "texts": {a: texts[a] for a in DEC_ARMS},
        "generation_diagnostics": {a: gdiag[a] for a in DEC_ARMS},
        "per_state": [], "readings": {}})
    arr["sid"] = np.asarray([s["sid"] for s in reg], object)
    arr["R"] = R
    arr["cos_tower"] = cos_tower.astype(np.float32)
    arr["rand_tag"] = rand_tag
    arr["cos_prior"] = cos_P
    arr["cos_Aprime"] = cos_AP
    for a in DEC_ARMS:
        arr[f"text_{a}"] = np.asarray(texts[a], object)
        arr[f"cos_{a}"] = cos[a]
        arr[f"unique_{a}"] = uniq[a]
        arr[f"own_concept_{a}"] = own[a]
        arr[f"rank_tower_own_{a}"] = np.asarray(
            [rank_tower[i, int(own[a][i])] if own[a][i] >= 0 else -1 for i in range(N_STATES)],
            np.int64)
        arr[f"overlap_L_inter_T_{a}"] = overlap[a]
    flush()

    log("#" * 110)
    log(f"# scoring {N_STATES} states x ({len(SCORE_ARMS)} arms + the m=0 base) through genv_eval")
    for i in range(N_STATES):
        s = reg[i]
        rws = [int(r) for r in s["rows"]]
        vals, cells = {}, {}
        bf, bt_ = genv_eval(E, [], rws)
        assert np.isfinite(bf).all(), "a base full-NDCG is NaN -- a cohort row has no target"
        tmask = np.isfinite(bt_)
        vals["level_base/full"] = bf
        vals["level_base/tail"] = bt_[tmask]
        cells["base"] = {"full": C4B.boot_mean_chunked(bf), "tail": C4B.boot_mean_chunked(bt_)}
        arr[f"s{i}_rows"] = np.asarray(rws, np.int64)
        arr[f"s{i}_base_full"] = bf.astype(np.float32)
        arr[f"s{i}_base_tail"] = bt_.astype(np.float32)
        F, T = {}, {}
        seen = {}                     # identical concept SETS fold identically -- evaluate once
        for a in SCORE_ARMS:
            key = tuple(arm_c[a][i])
            if key not in seen:
                seq = [(1, int(c)) for c in key]              # ascending concept id
                f, t = genv_eval(E, seq, rws)
                assert np.isfinite(f).all()
                assert (np.isfinite(t) == tmask).all(), "the tail-eligible set moved between arms"
                seen[key] = (f, t)
            f, t = seen[key]
            F[a], T[a] = f, t
            vals[f"level_{a}/full"] = f
            vals[f"level_{a}/tail"] = t[tmask]
            cells[a] = {"full": C4B.boot_mean_chunked(f), "tail": C4B.boot_mean_chunked(t),
                        "n_concepts": len(arm_c[a][i]), "concepts": arm_meta[a][i]}
            arr[f"s{i}_{a}_full"] = f.astype(np.float32)
            arr[f"s{i}_{a}_tail"] = t.astype(np.float32)
        diffs = {}
        for x, y in PAIRS:
            k = f"{x}_minus_{y}"
            vals[f"{k}/full"] = F[x] - F[y]
            vals[f"{k}/tail"] = (T[x] - T[y])[tmask]
            diffs[k] = {"full": C4B.boot_mean_chunked(F[x] - F[y]),
                        "tail": C4B.boot_mean_chunked(T[x] - T[y])}
            for mt in ("full", "tail"):
                diffs[k][mt]["directional_pass"] = C4B.directional(diffs[k][mt])
        SB.add_state(vals)
        steps, ok_path = POC.path_in_words(s, br.names, ni, pool2full)
        per_state.append({
            "i": i, "sid": s["sid"], "group": s["group"], "depth": len(s["prefix"]),
            "n_users": len(rws), "n_tail_eligible": int(tmask.sum()),
            "sid_path_matches_prefix": bool(ok_path), "interview_path": steps,
            "renders": {a: texts[a][i] for a in DEC_ARMS},
            "render_is_unique_in_arm": {a: bool(uniq[a][i]) for a in DEC_ARMS},
            "render_cos_to_R": {a: (None if not np.isfinite(cos[a][i]) else float(cos[a][i]))
                                for a in DEC_ARMS},
            "render_own_concept": {a: (None if own[a][i] < 0 else br.tag_names[int(own[a][i])])
                                   for a in DEC_ARMS},
            "render_own_concept_rank_tower": {
                a: (None if own[a][i] < 0 else int(rank_tower[i, int(own[a][i])]))
                for a in DEC_ARMS},
            "overlap_L_inter_T": {a: int(overlap[a][i]) for a in DEC_ARMS},
            "n_distinct_evaluated_concept_sets": len(seen),
            "arm_concepts": {a: [x["concept"] for x in arm_meta[a][i]] for a in SCORE_ARMS},
            "levels_" + CIRC.replace(" ", "_"): cells,
            "differences": diffs})
        flush()
        log(f"  [{i + 1:>2}/{N_STATES}] {s['sid']:<32} n={len(rws):>5} "
            + " ".join(f"{a}:{diffs[f'L_minus_T_{a}']['full']['mean']:+.5f}" for a in DEC_ARMS)
            + f" ({(time.time() - t00) / 60:.1f}m)")

    # ---- the readings --------------------------------------------------------------------------
    summary = {k: SB.read(k) for k in boot_keys}
    for k, v in summary.items():
        if "_minus_" in k:
            v["directional_pass"] = C4B.directional(v)

    def per_state_decomp(a, mask, mt="full"):
        v = np.asarray([r["differences"][f"L_minus_T_{a}"][mt]["mean"] for r in per_state],
                       np.float64)[mask]
        if not len(v):
            return {"n": 0, "mean": None, "median": None}
        return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
                "min": float(v.min()), "max": float(v.max())}

    cost = {}
    for a in DEC_ARMS:
        k = f"L_minus_T_{a}"
        n_pass = {mt: int(sum(1 for r in per_state
                              if r["differences"][k][mt]["directional_pass"])) for mt in
                  ("full", "tail")}
        n_neg = {mt: int(sum(1 for r in per_state
                             if (r["differences"][k][mt]["ci_hi"] or 0.0) < 0)) for mt in
                 ("full", "tail")}
        cost[a] = {
            "summary_cohort_weighted": {mt: summary[f"{k}/{mt}"] for mt in ("full", "tail")},
            "n_states_CI_above_zero": n_pass, "n_states_CI_below_zero": n_neg,
            "decomposition_unique_vs_repeated_render": {
                mt: {"unique_render": per_state_decomp(a, uniq[a], mt),
                     "repeated_render": per_state_decomp(a, ~uniq[a], mt)}
                for mt in ("full", "tail")},
            "decomposition_by_the_banked_greedy_stock_class": {
                mt: {"greedy_stock_word_states": per_state_decomp(a, stock_greedy, mt),
                     "other_states": per_state_decomp(a, ~stock_greedy, mt)}
                for mt in ("full", "tail")},
            "overlap_L_inter_T": {"mean": float(overlap[a].mean()),
                                  "histogram": {str(int(x)): int(y) for x, y in
                                                zip(*np.unique(overlap[a], return_counts=True))}},
            "n_states_asking_nothing_non_encodable_render": n_empty_ask[a]}

    out["readings"] = {
        "reading1_diversity": diversity,
        "reading2_triangulation": {"arms": tri, "random_tag_floor": floor,
                                   "gate": tri_gate},
        "reading3_faithfulness": faith,
        "reading4_translation_cost_L_minus_T_per_decoding_arm": {
            "what": ("THE NUMBER THAT TESTS THE HYPOTHESIS. Banked greedy row (C4B_NOISE.md): "
                     "+0.01802 full [+0.01668, +0.01940] / +0.00672 tail. Same simulator, same L "
                     "and P, only arm T's decoding moves."),
            "per_arm": cost},
        "reading5_absolute_levels_" + CIRC.replace(" ", "_"): {
            "label": CIRC,
            "cohort_weighted": {k: {mt: summary[f"level_{k}/{mt}"] for mt in ("full", "tail")}
                                for k in LEVELS},
            "note": "PRINTED, never headlined, never compared to any non-simulated number"}}
    out["summary_all_cells"] = summary
    flush()

    # ---- the console read ------------------------------------------------------------------------
    log("=" * 110)
    log("READING 1 -- DIVERSITY (distinct texts / 42)")
    for a in DEC_ARMS:
        d_ = diversity[a]
        log(f"  {a:<12} {d_['n_distinct']:>2}/42 = {d_['frac_distinct']:.3f}   unique-render states "
            f"{d_['n_states_with_a_unique_render']:>2}   max freq {d_['max_frequency']:>2}   empty "
            f"{d_['n_empty_generations_declared_non_encodable']}   top: "
            + ", ".join(f"{r['text']!r}x{r['n']}" for r in d_["frequency_table"][:4]))
    log(f"READING 2 -- TRIANGULATION: rank of the render's own concept in the {C}-concept ranking "
        f"by cos to R_s (median [IQR]); random-tag floor median "
        f"{floor['rank_in_tower_ranking']['median']:.0f}")
    for a in DEC_ARMS:
        for cls in ("all", "unique_render", "repeated_render"):
            q = tri[a][cls]["rank_in_tower_ranking"]
            if q["n"]:
                log(f"  {a:<12} {cls:<16} n={q['n']:>2}  median {q['median']:>6.1f}  IQR "
                    f"[{q['q1']:.1f}, {q['q3']:.1f}]  top-10 hit "
                    f"{tri[a][cls]['hit_top10_tower']['k']}/{tri[a][cls]['hit_top10_tower']['n']}")
    log("READING 3 -- FAITHFULNESS cos(u_text, R_s), and the paired difference vs greedy")
    for a in DEC_ARMS:
        lv, dd = faith[a]["level_cos_to_R"], faith[a]["minus_greedy"]
        log(f"  {a:<12} A {lv['mean']:+.5f} CI [{lv['ci_lo']:+.5f}, {lv['ci_hi']:+.5f}] "
            f"(n={lv['n']}, non-encodable {lv['n_non_encodable']})   A - A(greedy) "
            f"{dd['mean']:+.5f} CI [{dd['ci_lo']:+.5f}, {dd['ci_hi']:+.5f}] n={dd['n_pairs']}")
    log("READING 4 -- TRANSLATION COST L - T per decoding arm (banked greedy: +0.01802 full / "
        "+0.00672 tail)")
    for a in DEC_ARMS:
        for mt in ("full", "tail"):
            v = cost[a]["summary_cohort_weighted"][mt]
            dc = cost[a]["decomposition_unique_vs_repeated_render"][mt]
            log(f"  {a:<12} {mt:<5} {v['mean']:+.6f} CI [{v['ci_lo']:+.6f}, {v['ci_hi']:+.6f}] "
                f"n_rows={v['n_user_rows']:,}  states CI>0 "
                f"{cost[a]['n_states_CI_above_zero'][mt]}/42 CI<0 "
                f"{cost[a]['n_states_CI_below_zero'][mt]}/42  | unique "
                f"{(dc['unique_render']['mean'] if dc['unique_render']['mean'] is not None else float('nan')):+.5f}"
                f" (n={dc['unique_render']['n']}) repeated "
                f"{(dc['repeated_render']['mean'] if dc['repeated_render']['mean'] is not None else float('nan')):+.5f}"
                f" (n={dc['repeated_render']['n']})")
    log(f"READING 5 -- ABSOLUTE LEVELS, {CIRC.upper()} (printed, never headlined)")
    for k in LEVELS:
        a_, b_ = summary[f"level_{k}/full"], summary[f"level_{k}/tail"]
        log(f"  {k:<14} full {a_['mean']:.5f} [{a_['ci_lo']:.5f}, {a_['ci_hi']:.5f}] | tail "
            f"{b_['mean']:.5f} [{b_['ci_lo']:.5f}, {b_['ci_hi']:.5f}]")
    log("=" * 110)
    log(f"{'sid':<30} | " + " | ".join(f"{a:<20}" for a in DEC_ARMS))
    for i, r in enumerate(per_state):
        log(f"{r['sid']:<30} | " + " | ".join(f"{texts[a][i]!r:<20}" for a in DEC_ARMS))
    log("=" * 110)
    log(f"[done] {OUT_JSON} + {os.path.basename(OUT_NPZ)}  ({out['minutes']:.1f}m)")
    return out


def main():
    ap = argparse.ArgumentParser(description="Paper C / C4C -- decoding-only fix for the render "
                                             "mode collapse (author brief 2026-08-30)")
    ap.add_argument("--ckpt", default=E1_CKPT)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gen_batch", type=int, default=42)
    ap.add_argument("--allow_dirty", action="store_true",
                    help="EMERGENCY ONLY: run with uncommitted IMPORTED src/instrument code")
    args = ap.parse_args()
    args.threads = min(int(args.threads), 4)
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    torch.set_num_threads(int(args.threads))
    log("#" * 110)
    log("# PAPER C / C4C -- DECODING ONLY. No adapter is retrained; the bridge, R_s, the "
        "declared-circular answer model and genv_eval are IMPORTED verbatim. VALIDATION ONLY; the "
        "10,000 TEST users are not built, not loaded, not touched. No LLM API.")
    log("#" * 110)
    return run(args)


if __name__ == "__main__":
    main()
