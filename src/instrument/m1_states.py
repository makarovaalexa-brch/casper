r"""M1 Stage A -- fold BOTH populations into ONE latent space.

Design sheet: `docs/design/DS_M1_STATE_MATCHING.md` (three objects: IDEAL / REALISTIC / ALIGNMENT).
Committed before running (HARD RULE 10). VALIDATION ONLY -- the 10,000 canonical TEST users are
never built, never loaded, never named.

THE THREE OBJECTS
-----------------
  z_r       REALISTIC: each foldable Reddit REQUEST folded from COLD -- the state a real user
            arrives in having said one sentence. Anchors enter at a LIKE level; mapped genre/tone
            concepts enter as asserted-positive. Both are DECLARED modelling choices (no `V` row
            exists for a Reddit poster) and are recorded in the artifact.
  z_full    IDEAL: each of the 10,000 validation users folded from their WHOLE liked fold-in --
            the best partition of taste, the upper bound. NOT observable at ask time; it is a
            ceiling, not a route.
  z_open    each validation user folded from a CONSTRUCTED opener bundle (Stage C's twin evidence).
            Written by `m1_twins.py`, not here; this file emits z_r and z_full.

WHY ONE SPACE
-------------
Both populations pass through the SAME frozen encoder via `fold_items_frozen` / the certified
concept fold, from the SAME cold start. That is what licenses the alignment cross-tab; anything
else would compare two different geometries.

NO SCORING HAPPENS HERE. No decoder matmul, no NDCG, no targets are read. This file produces
latents and counts only.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import argparse
import csv
import gzip
import json
import re
import time

import numpy as np
import torch

OUT_DIR = os.path.join(_ROOT, "experiments", "battery", "m1")
RULER = os.path.join(_ROOT, "data", "reddit_he", "ruler", "ruler_v1.csv.gz")
SLOTS = os.path.join(_ROOT, "data", "reddit_he", "slots")
MAPPING = os.path.join(_ROOT, "experiments", "battery", "m1b_mapping.json")
TAGS = os.path.join(_ROOT, "data", "movielens", "genome-tags.csv")

SEED = 98765
VAL_M0_FULL = 0.15686538118354684          # canonical snap; asserted before anything is trusted


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- Reddit side: inputs
def load_requests():
    """thread_id -> list of MovieLens movieIds named by the OP (already linked, in-catalogue)."""
    out = {}
    with gzip.open(RULER, "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ids = [int(x) for x in row["op_named_movieIds"].split()] if row["op_named_movieIds"].strip() else []
            out[row["thread_id"]] = ids
    return out


def load_slots():
    """thread_id -> slots dict (typed, extracted in an earlier pass; labels are model-produced)."""
    out = {}
    for fn in sorted(f for f in os.listdir(SLOTS) if f.startswith("full_out") and f.endswith(".jsonl")):
        with open(os.path.join(SLOTS, fn), encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("thread_id"):
                    out[rec["thread_id"]] = rec.get("slots") or {}
    return out


def concept_name_index(E):
    """genome tag string -> concept id in the certified bank.

    The concept vocabulary IS the genome tag vocabulary: `E["sh"]["tags"]` holds tagIds in the order
    the concept channel is indexed. Same construction `forest_policy.py` uses -- read from there
    rather than re-derived, so the two files cannot drift.
    """
    name = {}
    with open(TAGS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name[int(r["tagId"])] = r["tag"].strip().lower()
    tags = list(E["sh"]["tags"])
    assert len(tags) == int(E["C"]), (
        f"concept vocabulary is {len(tags)} tags but the env indexes {E['C']} concepts")
    idx = {}
    for cid, t in enumerate(tags):
        s = name.get(int(t)) if not isinstance(t, str) else t
        if s:
            s = str(s).lower()
            idx.setdefault(s, cid)
            idx.setdefault(s.replace("-", " "), cid)
    return idx


def norm_txt(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s'-]", " ", s)
    s = re.sub(r"\b(movies|movie|films|film|flicks|flick)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip(" -'")


def main() -> None:
    ap = argparse.ArgumentParser(description="M1 Stage A -- fold requests and full profiles")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    args = ap.parse_args()
    assert args.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"

    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_mean,
                                        CERT_TOWER, CERT_CONC)
    from concept_fold import fold_items_frozen
    from i26_encoder import UNSEEN_LEVEL
    from ndcg_tree import leak_assert

    log("#" * 100)
    log("# M1 STAGE A -- states for BOTH populations. VALIDATION ONLY. No TEST users are built.")
    log("#" * 100)

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m1states")
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], "m1states/val")
    rows = np.asarray(E["rows"], np.int64)
    assert len(rows) == 10000, f"validation cohort is {len(rows)}, expected 10,000"

    f0, t0_ = genv_mean(E, [], rows)
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} tail={t0_!r} -- exact")

    ctx = E["ctx"]
    lvl = E["lvl"]

    # ---------------------------------------------------------------- level used for an "anchor"
    # A Reddit anchor is a film the poster SAYS they like. Our levels are half-star buckets; the
    # declared choice is the maximum liked level present in the cohort's own level vocabulary, so a
    # stated like is encoded as a confident like and nothing is invented.
    all_levels = set()
    for r in rows[:2000]:
        all_levels.update(int(v) for v in lvl[int(r)].values())
    rating_levels = sorted(v for v in all_levels if v < UNSEEN_LEVEL)
    LIKE_LEVEL = int(max(rating_levels)) if rating_levels else 0     # anchors: a stated favourite
    LIKED_MIN = 7                                                    # >= 4.0 stars, the project's
    #  "liked" band (same rule group_greedy_genre.py uses). Levels 7/8/9 all carry gamma=+1.0 under
    #  the certified SIGN INIT, so the exact anchor level inside that band cannot change the fold.
    log(f"[decl] level vocabulary seen={sorted(all_levels)}; UNSEEN_LEVEL={UNSEEN_LEVEL}; "
        f"anchors folded at LIKE_LEVEL={LIKE_LEVEL}; full-profile 'liked' = level >= {LIKED_MIN}")

    # ------------------------------------------------------------------- REALISTIC: request states
    reqs = load_requests()
    slots = load_slots()
    cidx = concept_name_index(E)
    assert len(cidx) > 0, "concept-name index is EMPTY -- the tag vocabulary did not load"
    log(f"[reddit] {len(reqs)} ruler threads; slot records for {len(slots)}; "
        f"concept-name index carries {len(cidx)} surface forms")

    # movieId -> item index is the ROW INDEX into unique_sid.txt (the canonical Liang show2id).
    # Same construction group_greedy_genre.py uses, asserted against ctx.ni.
    usid_path = os.path.join(_ROOT, "data", "ml-25m", "proc", "unique_sid.txt")
    with open(usid_path) as fh:
        sids = [int(x) for x in fh.read().split()]
    assert len(sids) == ctx.ni, f"unique_sid.txt has {len(sids)} items, ctx.ni={ctx.ni}"
    sid2row = {m: i for i, m in enumerate(sids)}
    assert len(sid2row) == ctx.ni, "unique_sid.txt contains duplicate movieIds"
    s2i = getattr(ctx, "show2id", None)
    if s2i is not None:
        assert dict(s2i) == sid2row, "ctx.show2id disagrees with unique_sid.txt"
        log("[gate] ctx.show2id == unique_sid.txt row index -- asserted")

    foldable, unfoldable = [], []
    for tid, ids in reqs.items():
        inner = [sid2row[i] for i in ids if i in sid2row]
        sl = slots.get(tid, {})
        cs = []
        for key in ("genres_pos", "tone"):
            v = sl.get(key)
            vals = v if isinstance(v, list) else ([v] if v else [])
            for raw in vals:
                n = norm_txt(str(raw))
                if n in cidx:
                    cs.append(cidx[n])
        cs = sorted(set(cs))
        if inner or cs:
            foldable.append((tid, inner, cs))
        else:
            unfoldable.append(tid)

    log(f"[reddit] foldable requests: {len(foldable)}  unfoldable: {len(unfoldable)} "
        f"({len(unfoldable)/max(len(reqs),1):.1%})")
    n_items_only = sum(1 for _, i, c in foldable if i and not c)
    n_conc_only = sum(1 for _, i, c in foldable if c and not i)
    n_both = sum(1 for _, i, c in foldable if i and c)
    log(f"[reddit] composition: items-only {n_items_only} · concepts-only {n_conc_only} · both {n_both}")

    seqs = [(np.asarray(inner, np.int64),
             np.full(len(inner), LIKE_LEVEL, np.int64)) for _, inner, _ in foldable]
    Z_req_items = fold_items_frozen(ctx.enc, seqs, batch=500)
    log(f"[reddit] item-fold done: {tuple(Z_req_items.shape)}")

    # ---- concept fold for the STATED genre/tone -------------------------------------------------
    # A stated preference is a maximally confident answer. There is no per-user V row for a Reddit
    # poster, so the asserted value is the cohort's own maximum observed positive concept value --
    # taken from the data rather than invented as 1.0. Refusal flags are 0: the poster volunteered it,
    # so by construction they could answer it.
    Vpos = np.asarray(E["V"])[np.asarray(E["V"]) > 0]
    ASSERT_VAL = float(np.max(Vpos)) if Vpos.size else 1.0
    log(f"[decl] concept values observed in cohort: max positive {ASSERT_VAL:.4f}; "
        f"stated preferences asserted at that value, refusal flag 0")

    net = E["net"]
    cids_list, cvals_list, crefs_list = [], [], []
    for _, _, cs in foldable:
        ids, vals = net.canonical_order(list(cs), [ASSERT_VAL] * len(cs))
        cids_list.append([int(c) for c in ids])
        cvals_list.append([float(v) for v in vals])
        crefs_list.append([0.0] * len(ids))
    with torch.no_grad():
        Z_req = net(Z_req_items, cids_list, cvals_list,
                    crefs_list if getattr(net, "use_refusal", False) else None)
    n_with_conc = sum(1 for c in cids_list if c)
    drift = float(torch.norm(Z_req - Z_req_items, dim=1).numpy()[
        [i for i, c in enumerate(cids_list) if c]].mean()) if n_with_conc else 0.0
    log(f"[reddit] concept fold applied to {n_with_conc} requests; mean latent shift {drift:.4f}")
    assert n_with_conc == 0 or drift > 0, "concept fold changed nothing -- the fold is not wired"
    Z_req_items = Z_req

    np.savez_compressed(
        os.path.join(OUT_DIR, "z_requests.npz"),
        z=Z_req_items.numpy().astype(np.float32),
        thread_ids=np.asarray([t for t, _, _ in foldable]),
        n_items=np.asarray([len(i) for _, i, _ in foldable], np.int32),
        concept_ids=np.asarray([json.dumps(c) for _, _, c in foldable]),
    )

    # ------------------------------------------------------------------------ IDEAL: full profiles
    # Every validation user's whole LIKED fold-in, folded from cold through the same encoder.
    full_seqs = []
    for r in rows:
        d = lvl[int(r)]
        liked = [(int(i), int(L)) for i, L in d.items() if LIKED_MIN <= int(L) < UNSEEN_LEVEL]
        if liked:
            ids = np.asarray([i for i, _ in liked], np.int64)
            lv = np.asarray([L for _, L in liked], np.int64)
        else:
            ids = np.zeros(0, np.int64); lv = np.zeros(0, np.int64)
        full_seqs.append((ids, lv))
    lens = np.asarray([len(a) for a, _ in full_seqs], np.int32)
    log(f"[ideal] full liked fold-in per user: mean {lens.mean():.1f} median {np.median(lens):.0f} "
        f"min {lens.min()} max {lens.max()}")
    Z_full = fold_items_frozen(ctx.enc, full_seqs, batch=500)
    log(f"[ideal] full-profile fold done: {tuple(Z_full.shape)}")

    np.savez_compressed(
        os.path.join(OUT_DIR, "z_full.npz"),
        z=Z_full.numpy().astype(np.float32),
        rows=rows.astype(np.int64),
        n_liked=lens,
    )

    meta = {
        "analysis": "m1_stage_a_states",
        "design_sheet": "docs/design/DS_M1_STATE_MATCHING.md",
        "split": "validation",
        "test_touched": False,
        "tower": CERT_TOWER,
        "concept_ckpt": CERT_CONC,
        "canonical_snap_full": f0,
        "declared_choices": {
            "anchor_level": LIKE_LEVEL,
            "unseen_level": int(UNSEEN_LEVEL),
            "concepts_asserted_positive": True,
            "note": "no per-user V row exists for a Reddit poster; stated preferences are encoded "
                    "as maximally confident. Declared, not measured.",
        },
        "reddit": {
            "threads_total": len(reqs),
            "foldable": len(foldable),
            "unfoldable": len(unfoldable),
            "items_only": n_items_only,
            "concepts_only": n_conc_only,
            "both": n_both,
            "concept_surface_forms_indexed": len(cidx),
        },
        "ideal": {
            "users": int(len(rows)),
            "liked_mean": float(lens.mean()),
            "liked_median": float(np.median(lens)),
        },
    }
    with open(os.path.join(OUT_DIR, "m1_stage_a.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    log(f"[done] wrote {OUT_DIR}\\z_requests.npz, z_full.npz, m1_stage_a.json")


if __name__ == "__main__":
    main()
