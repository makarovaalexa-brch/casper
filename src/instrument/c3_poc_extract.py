r"""PAPER C / STAGE C3 -- PoC STAGE 1: EXTRACTION of the rendering packets.

Design sheet: `docs/design/DS_C3_POC.md` (PRE-REGISTERED 2026-08-21, BINDING). Read it first.
This file does NO training and takes NO test look. It is read-only numpy over the C1 pilot's banked
arrays (`experiments/paperC/c1_pilot_per_state.npz`) plus the e_q bank rebuilt from the C0-sha'd
checkpoint through `c0_geometry.bank_vectors` (IMPORTED, never reimplemented), exactly as
`c1_blend_coherence.py` does. The rebuilt m=5 targets are asserted bit-identical to the pilot's
before any packet is written, so the points in the packets are the same objects Stage 0/1 measured.

WHAT IT PRODUCES  (`experiments/paperC/c3_poc/`)
------------------------------------------------
  packets.json          30 packets, format-identical, id-keyed. Each carries the top-8 nearest bank
                        phrases with MEAN-REMOVED cosines, the asked-so-far interview path in plain
                        words, and nothing that names its own type.
  KEY.json              the ground truth (type, source state, pairing). QUARANTINED: the renderer
                        and the judge subagents must never read it.
  c3_poc_vectors.npz    every packet's SOURCE VECTOR in the bias-free space `k(phi)`, keyed by
                        packet id -- banked NOW so the later re-encode stage has its reference.

THE THREE TYPES (10 each; the sheet's control design)
-----------------------------------------------------
  real        a C1 point: either the m=5 blend target at a state, or the full-policy/seed-1
              fine-tuned emission z_hat at a state (bias-free part), paired with THAT state's path.
  scrambled   state A's interview path carried with state B's real point (a derangement over the 10
              real points -- no packet keeps its own). This is the CORRESPONDENCE control: the
              distribution of points is untouched, only the pairing is destroyed.
  random      a Gaussian vector drawn with the covariance SPECTRUM of the 10 real points and
              rescaled to its partner's norm, carried with that partner's path.

THE SPACE. Bias-free `k(phi_q)`, i.e. the first 128 coordinates of C0's homogeneous
`e_q = [k(phi_q), b(phi_q)]`. For a blend target `sum_j w_j e_{q_j}` the first 128 coordinates ARE
`sum_j w_j k(phi_{q_j})`, and for the emission `z_hat = [u(h), 1]` they ARE `u(h)`; both identities
are asserted at run time, not assumed. Cosines are reported MEAN-REMOVED (the bank's shared
common-mode component is subtracted first): C1_PILOT.md's coherence section measured the raw
bank-wide mean pairwise cosine at 0.9646 against 0.0018 centred, so a raw cosine ranking is nearly
uninformative about semantics.

VALIDATION ONLY. The 10,000 canonical TEST users are not built, not loaded, not touched.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
if "--threads" not in sys.argv:
    sys.argv += ["--threads", "3"]
_NT = sys.argv[sys.argv.index("--threads") + 1]
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("MKL_NUM_THREADS", _NT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)

import csv
import json
import argparse
import numpy as np
import torch

from policy_net import Policy, build_phi, load_traj, log
from c0_geometry import PAPERC, E1_CKPT, N_BANK, sha256, git, bank_vectors
import c1_pilot as C1

OUT_DIR = os.path.join(PAPERC, "c3_poc")
TAGS_CSV = os.path.join(_ROOT, "data", "movielens", "genome-tags.csv")
MOVIES_CSV = os.path.join(_ROOT, "data", "movielens", "movies.csv")
USID = os.path.join(_ROOT, "data", "ml-25m", "proc", "unique_sid.txt")

SEED = 20260821          # DECLARED once: selection tie-breaks, the derangement, the spectrum-matched
                         # draws and the id permutation all come from this one stream.
N_STATES = 10            # the sheet's "~10 states"; 10 real + 10 scrambled + 10 random = 30 packets
M = 5                    # the pilot's PRIMARY blend width
TOPK = 8                 # the sheet's "top-8 nearest bank phrases"
EMISSION_RUN = "zbar_full_seed1"   # the sheet's "full-policy arm, seed 1"

INSTRUCTION = (
    "Below is a direction in a recommender's question space, described only by the bank phrases "
    "nearest to it, together with what the interview has already asked this user and how they "
    "answered. Write ONE natural, askable interview question that captures the direction. Do not "
    "reuse any anchor phrase verbatim, do not list the anchors, and do not repeat a question that "
    "has already been asked. If the anchors do not cohere into anything a person could sensibly "
    "answer, say so explicitly instead of inventing a question.")


# ============================================================================ surface forms
def surface_forms(E):
    """bank index -> a human phrase. Bank order is `[(0, i) for i in items] + [(1, c) for c in C]`,
    the same `full_pool` C0/C1 build, so index < ctx.ni is an ITEM and the rest are CONCEPTS.

    Items are named by their ML-25M title; the movieId of item row i is line i of the canonical
    Liang `unique_sid.txt` (the same construction `m1_states.py` asserts against `ctx.show2id`).
    Concepts are named by their genome tag, via `E["sh"]["tags"]` -- the SAME path
    `forest_policy.py:genre_to_concept` and `m1_states.py:concept_name_index` use, read from the env
    rather than re-derived, so the three files cannot drift.
    """
    ni = int(E["ctx"].ni)
    C = int(E["C"])
    assert ni + C == N_BANK, f"bank composition {ni}+{C} != {N_BANK}"

    with open(USID) as fh:
        sids = [int(x) for x in fh.read().split()]
    assert len(sids) == ni, f"unique_sid.txt has {len(sids)} items, ctx.ni={ni}"
    s2i = getattr(E["ctx"], "show2id", None)
    if s2i is not None:
        assert dict(s2i) == {m: i for i, m in enumerate(sids)}, \
            "ctx.show2id disagrees with unique_sid.txt"
    title = {}
    with open(MOVIES_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            title[int(r["movieId"])] = r["title"].strip()

    tagname = {}
    with open(TAGS_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            tagname[int(r["tagId"])] = r["tag"].strip().lower()
    tags = list(E["sh"]["tags"])
    assert len(tags) == C, f"concept vocabulary is {len(tags)} tags but the env indexes {C}"

    names = [None] * N_BANK
    missing_item, missing_concept = [], []
    for i in range(ni):
        t = title.get(sids[i])
        if t is None:
            missing_item.append(i)
        names[i] = t
    for cid, t in enumerate(tags):
        s = t if isinstance(t, str) else tagname.get(int(t))
        if s is None:
            missing_concept.append(cid)
        names[ni + cid] = None if s is None else str(s).lower()
    return names, ni, {"n_items_without_a_title": len(missing_item),
                       "n_concepts_without_a_tag_name": len(missing_concept),
                       "item_bank_indices_without_a_title": missing_item[:50],
                       "concept_ids_without_a_tag_name": missing_concept[:50]}


def phrase(names, ni, bank_idx):
    n = names[bank_idx]
    if n is None:
        return f"<unnamed bank entry {bank_idx}>"
    return n if bank_idx >= ni else n


def question_text(names, ni, bank_idx):
    """The bank entry as an asked question, in plain words."""
    n = phrase(names, ni, bank_idx)
    return (f"Do you like {n} films?" if bank_idx >= ni
            else f"Have you seen {n}, and did you like it?")


ANSWER_WORDS = {"like": "liked it", "dislike": "disliked it",
                "noanswer": "hasn't seen it / gave no answer"}


def path_in_words(s, names, ni, pool2full):
    """The asked-so-far interview, question by question, with the answer this cohort gave.

    The state id encodes the branch: `root.<a1>.<a2>...[@depth]`, and `prefix[j]` is the question
    asked at step j (the same object `c1_pilot.state_geometry` writes into `asked_full`). If the two
    ever disagree in length the state is REPORTED, never patched.
    """
    parts = s["sid"].split("@")[0].split(".")
    outs = parts[1:]
    steps = []
    ok = (len(outs) == len(s["prefix"]))
    for j, q in enumerate(s["prefix"]):
        bi = int(pool2full[s["pool_prefix_idx"][j]])
        steps.append({"step": j + 1,
                      "question": question_text(names, ni, bi),
                      "answer": ANSWER_WORDS.get(outs[j], "unknown") if ok else "unknown"})
    return steps, ok


# ============================================================================ geometry helpers
def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-30)


def top_anchors(v_centred, Kc_unit, names, ni, k=TOPK):
    c = Kc_unit @ (v_centred / (np.linalg.norm(v_centred) + 1e-30))
    o = np.argsort(-c)[:k]
    return [{"rank": int(r + 1), "phrase": phrase(names, ni, int(i)),
             "kind": "concept" if int(i) >= ni else "film",
             "cosine": round(float(c[i]), 4)} for r, i in enumerate(o)]


def mean_removed_ratio(Ki, w, mu):
    """C1_PILOT.md's cancellation ratio in the bias-free space, mean removed."""
    X = Ki.astype(np.float64) - mu
    num = np.linalg.norm((w[:, None] * X).sum(0))
    den = float((w * np.linalg.norm(X, axis=1)).sum())
    return float(num / (den + 1e-30))


# ============================================================================ the run
def run(args):
    from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
    from ndcg_tree import build_pool
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    blob = torch.load(args.ckpt, map_location="cpu")
    sha = sha256(args.ckpt)
    assert sha == C1.C0_CKPT_SHA, f"this is not the C0/C1-measured checkpoint (sha {sha})"
    log(f"[ckpt] {os.path.basename(args.ckpt)} epoch={blob['epoch']} sha256 matches C0")

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "paperC/c3poc")
    pool, _it, _cc = build_pool(E, 500, 500)
    pool = [(int(a), int(b)) for a, b in pool]
    assert pool == [(int(a), int(b)) for a, b in blob["pool"]], "the env's pool is not the ckpt's"
    pool_index = {q: i for i, q in enumerate(pool)}
    ni_env = int(E["ctx"].ni)
    full_pool = [(0, i) for i in range(ni_env)] + [(1, c) for c in range(int(E["C"]))]
    assert len(full_pool) == N_BANK
    pool2full = np.asarray([(i if ch == 0 else ni_env + i) for ch, i in pool], np.int64)

    norm = (blob["norm_mu"], blob["norm_sd"])
    phi_t, _ = build_phi(E, pool, norm=norm)
    phi_full, _ = build_phi(E, full_pool, norm=norm)
    pol0 = Policy(E["ctx"].d, phi_full.shape[1], len(full_pool), arch=blob["arch"],
                  state=blob["state"], use_pos=blob["use_pos"])
    pol0.load_state_dict(blob["model"]); pol0.eval()
    Ebank, Kbank = bank_vectors(pol0, phi_full)
    Epool, Kpool = bank_vectors(pol0, phi_t)
    log(f"[bank] Ebank {Ebank.shape} Kbank {Kbank.shape}")

    Dva = load_traj(train=False)
    reg, extra = C1.load_the_42(pool_index, Dva)
    states = reg + extra
    assert len(reg) == 42

    # ---- GATE: the rebuilt targets are the pilot's targets, bit for bit --------------------------
    npz = np.load(os.path.join(PAPERC, "c1_pilot_per_state.npz"))
    Tt, TKt, meta = C1.build_targets(states, Epool, Kpool, pool_index, M)
    d = float(np.abs(Tt - npz[f"target_m{M}"]).max())
    assert d == 0.0, f"the rebuilt m={M} targets are not the pilot's (max|diff| {d:g})"
    log(f"[gate] m={M} target reproduction max|diff| {d:g} -- bit-identical")
    # the bias-free identity: the first d_k coordinates of the homogeneous target ARE the k-space one
    d_bf = float(np.abs(Tt[:, :-1] - TKt).max())
    assert d_bf == 0.0, f"target[:, :-1] != TK (max|diff| {d_bf:g})"
    Zb = npz[EMISSION_RUN]
    assert Zb.shape[0] == 42 and float(np.abs(Zb[:, -1] - 1.0).max()) < 1e-5, \
        "the banked emission's homogeneous coordinate is not pinned to 1"
    log(f"[gate] bias-free identities asserted; emission run {EMISSION_RUN} {Zb.shape}")

    # ---- the prefix pool indices (c1_pilot writes them during precompute; we only need the ids) --
    for s in states:
        s["pool_prefix_idx"] = [pool_index[q] for q in s["prefix"]]

    names, ni, missing = surface_forms(E)
    log(f"[phrases] surface forms built; missing titles {missing['n_items_without_a_title']}, "
        f"missing tag names {missing['n_concepts_without_a_tag_name']}")

    # ---- the mean-removed space ------------------------------------------------------------------
    mu = Kbank.astype(np.float64).mean(0)
    Kc = Kbank.astype(np.float64) - mu
    Kc_unit = unit(Kc).astype(np.float32)
    CH = np.asarray([c for c, _ in pool], np.int64)

    # ---- STATE SELECTION (deterministic, declared before any packet is written) ------------------
    prof = []
    for i, s in enumerate(reg):
        q = np.asarray(meta[i]["top_m_pool_idx"], np.int64)
        w = np.asarray(meta[i]["weights"], np.float64)
        _p = s["sid"].split("@")[0].split(".")
        prof.append({"i": i, "sid": s["sid"], "depth": len(s["prefix"]),
                     "first_answer": (_p[1] if len(_p) > 1 else "none"),
                     "n_concept": int((CH[q] == 1).sum()),
                     "coherence": mean_removed_ratio(Kpool[q], w, mu),
                     "group": s["group"], "n_users": len(s["rows"])})
    order = sorted(prof, key=lambda p: p["sid"])
    worst = min(prof, key=lambda p: (p["coherence"], p["sid"]))
    chosen = [worst]
    log(f"[select] forced low-coherence state {worst['sid']} "
        f"(mean-removed ratio {worst['coherence']:.4f})")
    while len(chosen) < N_STATES:
        dcov = {c["depth"] for c in chosen}
        ncov = {c["n_concept"] for c in chosen}
        acov = {c["first_answer"] for c in chosen}
        pool_c = [p for p in order if p["sid"] not in {c["sid"] for c in chosen}]
        # Three declared coverage axes: tree DEPTH (weight 2, the sheet's first spanning
        # requirement), the blend's CHANNEL MIX (weight 1, the sheet's second), and the branch the
        # interview took first (weight 1) -- without the third axis, sid order alone fills the set
        # with dislike-first paths. `pool_c` is in sid order and `max` returns the FIRST maximal
        # element, so the tie-break is "smallest sid": deterministic, and nothing about the points
        # or the packets can influence it.
        best = max(pool_c, key=lambda p: 2 * (p["depth"] not in dcov)
                   + (p["n_concept"] not in ncov) + (p["first_answer"] not in acov))
        chosen.append(best)
    chosen = sorted(chosen, key=lambda p: p["sid"])
    # type assignment: alternate along the sid-sorted list, declared, not tuned
    for j, p in enumerate(chosen):
        p["point_kind"] = "blend_target_m5" if j % 2 == 0 else "emission_full_seed1"
    log("[select] " + " | ".join(f"{p['sid']}(d{p['depth']},c{p['n_concept']},"
                                 f"{'T' if p['point_kind'].startswith('blend') else 'E'})"
                                 for p in chosen))

    # ---- the real points, in the bias-free space -------------------------------------------------
    P = np.stack([(TKt[p["i"]] if p["point_kind"] == "blend_target_m5" else Zb[p["i"], :-1])
                  for p in chosen]).astype(np.float64)
    assert P.shape == (N_STATES, Kbank.shape[1])

    # ---- control (i): scrambled correspondence -- a derangement over the 10 real points ----------
    perm = rng.permutation(N_STATES)
    while (perm == np.arange(N_STATES)).any():
        perm = rng.permutation(N_STATES)

    # ---- control (ii): spectrum-matched random vectors -------------------------------------------
    # Match the real points' covariance SPECTRUM (eigenbasis + eigenvalues of their empirical
    # covariance about their own mean, so with n=10 the draw lives in the same <=9-dim subspace and
    # carries the same shared component), then rescale each draw to its partner's norm.
    Pm = P.mean(0)
    Xc = P - Pm
    U_, S_, Vt_ = np.linalg.svd(Xc, full_matrices=False)
    lam = (S_ ** 2) / max(len(P) - 1, 1)
    G = rng.standard_normal((N_STATES, len(lam)))
    R = Pm + (G * np.sqrt(lam)) @ Vt_
    R *= (np.linalg.norm(P, axis=1) / np.linalg.norm(R, axis=1))[:, None]
    spec = {"n_real_points": int(N_STATES), "rank": int((lam > 1e-12).sum()),
            "eigenvalues": [float(x) for x in lam],
            "real_norms": [float(x) for x in np.linalg.norm(P, axis=1)],
            "random_norms": [float(x) for x in np.linalg.norm(R, axis=1)]}

    # ---- assemble the 30 items, then permute the ids ----------------------------------------------
    items = []
    for j, p in enumerate(chosen):
        items.append({"type": "real", "ctx": j, "src": j, "vec": P[j]})
    for j, p in enumerate(chosen):
        items.append({"type": "scrambled", "ctx": j, "src": int(perm[j]), "vec": P[perm[j]]})
    for j, p in enumerate(chosen):
        items.append({"type": "random", "ctx": j, "src": None, "vec": R[j]})
    ids = rng.permutation(len(items)) + 1

    packets, key, vecs = [], [], {}
    n_path_bad = []
    for it, pid in zip(items, ids):
        st = reg[chosen[it["ctx"]]["i"]]
        steps, ok = path_in_words(st, names, ni, pool2full)
        if not ok:
            n_path_bad.append(st["sid"])
        v = np.asarray(it["vec"], np.float64)
        packets.append({
            "id": int(pid),
            "anchors": top_anchors(v - mu, Kc_unit, names, ni),
            "asked_so_far": steps,
            "n_asked": len(steps),
            "instruction": INSTRUCTION})
        key.append({"id": int(pid), "type": it["type"],
                    "context_state_sid": st["sid"],
                    "context_state_depth": len(st["prefix"]),
                    "context_state_n_concept_in_top5": chosen[it["ctx"]]["n_concept"],
                    "context_state_coherence_mean_removed": chosen[it["ctx"]]["coherence"],
                    "point_source_state_sid": (chosen[it["src"]]["sid"] if it["src"] is not None
                                               else None),
                    "point_kind": (chosen[it["src"]]["point_kind"] if it["src"] is not None
                                   else "spectrum_matched_random"),
                    "registered_42_index": (chosen[it["src"]]["i"] if it["src"] is not None
                                            else None)})
        vecs[f"p{int(pid):03d}"] = v.astype(np.float32)
    packets.sort(key=lambda x: x["id"])
    key.sort(key=lambda x: x["id"])
    assert len(vecs) == 30 and len({k["id"] for k in key}) == 30

    meta_block = {
        "_comment": ("QUARANTINE NOTICE. These packets are the blinded input to the C3 PoC "
                     "renderer and judge (docs/design/DS_C3_POC.md). The ground-truth labels live "
                     "in KEY.json in this same directory and MUST NOT be read by any agent that "
                     "renders or judges these packets -- doing so voids the blinding and the run. "
                     "Nothing in this file says which packets are real and which are controls."),
        "analysis": "paperC_C3_poc_stage1_extraction",
        "design_sheet": "docs/design/DS_C3_POC.md",
        "produced_by": "src/instrument/c3_poc_extract.py",
        "citable": True, "test_touched": False, "split": "validation",
        "n_packets": len(packets),
        "space": ("bias-free k(phi) -- the first %d coordinates of C0's homogeneous e_q; "
                  "cosines below are MEAN-REMOVED (the bank mean direction is subtracted from both "
                  "sides before the cosine)" % Kbank.shape[1]),
        "bank": {"n_entries": N_BANK, "n_items": ni, "n_concepts": N_BANK - ni},
        "git": {"rev": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD")},
        "checkpoint": {"path": os.path.relpath(args.ckpt, _ROOT), "sha256": sha},
        "seed": SEED}
    with open(os.path.join(OUT_DIR, "packets.json"), "w", encoding="utf-8") as fh:
        json.dump({"metadata": meta_block, "packets": packets}, fh, indent=1)

    with open(os.path.join(OUT_DIR, "KEY.json"), "w", encoding="utf-8") as fh:
        json.dump({"_comment": ("GROUND TRUTH -- QUARANTINED. Never shown to the renderer or the "
                                "judge subagents (docs/design/DS_C3_POC.md, 'Blinding = separation "
                                "of agents + label file quarantine')."),
                   "analysis": "paperC_C3_poc_stage1_key",
                   "seed": SEED, "n_packets": len(key),
                   "counts": {t: sum(1 for k in key if k["type"] == t)
                              for t in ("real", "scrambled", "random")},
                   "selected_states": chosen,
                   "derangement_over_real_points": [int(x) for x in perm],
                   "spectrum_match": spec,
                   "surface_form_coverage": missing,
                   "states_whose_sid_path_did_not_match_the_prefix": n_path_bad,
                   "packets": key}, fh, indent=1, default=float)

    np.savez_compressed(os.path.join(OUT_DIR, "c3_poc_vectors.npz"), **vecs)
    log(f"[done] wrote {OUT_DIR}: packets.json ({len(packets)}), KEY.json, c3_poc_vectors.npz "
        f"({len(vecs)} vectors, dim {Kbank.shape[1]})")
    if n_path_bad:
        log(f"[WARN] {len(n_path_bad)} state(s) whose sid path length != prefix length: {n_path_bad}")
    return {"packets": packets, "key": key}


def main():
    ap = argparse.ArgumentParser(description="Paper C / C3 PoC stage 1 -- docs/design/DS_C3_POC.md")
    ap.add_argument("--ckpt", default=E1_CKPT)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 100)
    log("# PAPER C / C3 PoC -- STAGE 1 EXTRACTION. Read-only over banked arrays; no training, no "
        "LLM call, no test user built, loaded or touched.")
    log("#" * 100)
    return run(args)


if __name__ == "__main__":
    main()
