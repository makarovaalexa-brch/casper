r"""PAPER C / STAGE C3 -- PoC STAGE 4: RE-ENCODE of the rendered questions.

Design sheet: `docs/design/DS_C3_POC.md`, section "Re-encode bridge (settled 2026-08-21)", BINDING.
Stage 1 (the packets, the source vectors) is `c3_poc_extract.py`; this file consumes its outputs.

WHAT THIS DOES
--------------
  1. Reads the 30 rendered questions from `renders.json` (field `question` ONLY -- no renderability,
     no theme, nothing else enters the geometry).
  2. Runs the frozen open-vocab SBERT path OF RECORD (`scripts/reddit/smoke_trio.py`, arms 4/4b:
     all-MiniLM-L6-v2, mode='sent' i.e. max cosine over the text's sentences, top-4 concepts with
     cosine >= 0.30) over the 1,031 genome tag NAMES.
  3. Bridges into the policy-head space: the selected concepts' own bank embeddings k(phi) ->
     UNIFORM mean direction (primary) and selection-cosine-weighted mean (secondary) -> MEAN-REMOVED
     cosine against the packet's source vector from `c3_poc_vectors.npz`. The mean is the BANK mean,
     the same convention Stage 1 used for the anchor cosines.
  4. Also computes, per packet, an ANCHOR REFERENCE row: the identical bridge applied to the
     packet's OWN top-8 anchor phrases treated as the text. Under variant (a) the renderer only ever
     sees those phrases, so this row bounds what RETRIEVAL alone can achieve and contextualises the
     rendered question's number.

BLINDING. This file never opens `_quarantine/KEY.json`, questions_for_judge.json or judgments.json.
It emits
PER-PACKET geometry only; the join with ground truth happens elsewhere.

CONTROLS BUILT IN (asserts, not assumptions)
  * checkpoint sha256 == the C0/C1-measured checkpoint (`c1_pilot.C0_CKPT_SHA`);
  * ★ the concept column is RE-DERIVED from `genome-scores.csv` + `unique_sid.txt` inside this file,
    with the documented filter (relevance >= 0.5, catalog-restricted, >= 30 members, sorted tagId),
    and asserted equal to `E["sh"]["tags"]`. The pre-2026-08-24 assert compared two mappings of the
    SAME `sh["tags"]` list and was therefore a tautology: it could only ever have caught a
    lower-casing bug, and would have passed on the wrong vocabulary;
  * the SBERT tag-name column order == the packets' concept channel
    (`c3_poc_extract.surface_forms`), element by element;
  * the rebuilt bank reproduces every packet's banked top-8 anchors and cosines exactly as
    `packets.json` recorded them -- i.e. this stage is measuring in the same space Stage 1 wrote.

Texts whose SBERT pass selects zero concepts above threshold are REPORTED as non-encodable, never
dropped. No training. No test user built, loaded or touched. No LLM API (SBERT is the project's
existing local frozen encoder).
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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")      # the model must come from the local HF cache

import csv
import json
import argparse
import numpy as np
import torch

from policy_net import Policy, build_phi, log
from c0_geometry import PAPERC, E1_CKPT, N_BANK, sha256, git_block, bank_vectors
import c1_pilot as C1
from c3_poc_extract import surface_forms, unit

OUT_DIR = os.path.join(PAPERC, "c3_poc")
TAGS_CSV = os.path.join(_ROOT, "data", "movielens", "genome-tags.csv")
GENOME_CSV = os.path.join(_ROOT, "data", "movielens", "genome-scores.csv")
USID = os.path.join(_ROOT, "data", "ml-25m", "proc", "unique_sid.txt")
# The concept-bank filter of record (`run_battery_phaseA.load_genome`, `tradeoff_ledger.build_shared`)
GENOME_MIN_RELEVANCE = 0.5
GENOME_MIN_MEMBERS = 30

# ---- the SBERT recipe OF RECORD -- smoke_trio.py arms 4/4b, constants copied verbatim -----------
SBERT_MODEL = "all-MiniLM-L6-v2"
SBERT_MODE = "sent"
SBERT_TOPN = 4
SBERT_MINCOS = 0.30


def load_json(name):
    with open(os.path.join(OUT_DIR, name), encoding="utf-8") as fh:
        return json.load(fh)


def tag_name_column(E):
    """The SBERT scoring column: one name per concept, in the env's `sh["tags"]` order.

    smoke_trio's `build_concepts` names a concept `genome-tags.csv`'s `tag` field, STRIPPED but not
    lower-cased, and orders the columns by `sorted(members.keys())` -- the identical construction
    `run_battery_phaseA.load_genome` + `tradeoff_ledger.build_shared` use to fill `sh["tags"]`. So
    the column order is the env's by construction; it is asserted against the packets' concept
    channel below rather than trusted.
    """
    name = {}
    with open(TAGS_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            name[int(r["tagId"])] = r["tag"].strip()
    out = []
    for t in list(E["sh"]["tags"]):
        out.append(t.strip() if isinstance(t, str) else name[int(t)])
    return out


def rederive_concept_tagids(chunk=2_000_000):
    """★ THE NON-CIRCULAR ALIGNMENT CHECK (repair 2026-08-24).

    The original assert compared `surface_forms(E)`'s concept names against `tag_name_column(E)`.
    Both are `E["sh"]["tags"]` mapped through `genome-tags.csv`, so the comparison was
    `f(x) == g(x)` for the same x -- it could only fail on a case bug, and would pass unchanged if
    the env's concept vocabulary were the wrong vocabulary entirely.

    This rebuilds the concept column from the RAW genome, inside this file, with the documented
    filter of record -- `relevance >= 0.5`, restricted to the canonical catalog (`unique_sid.txt`),
    tags with at least 30 surviving members, ordered by `sorted(tagId)` -- and the caller asserts
    equality with `E["sh"]["tags"]`. Read in chunks: `genome-scores.csv` is ~435 MB and each
    (movieId, tagId) appears exactly once, so a running per-tag COUNT is lossless (HARD RULE 1 --
    nothing is sampled, the whole file is streamed).
    """
    import pandas as pd
    with open(USID) as fh:
        catalog = np.asarray(sorted(int(x) for x in fh.read().split()), np.int64)
    counts = {}
    n_rows = 0
    for ch in pd.read_csv(GENOME_CSV, usecols=["movieId", "tagId", "relevance"],
                          dtype={"movieId": np.int32, "tagId": np.int16,
                                 "relevance": np.float32},
                          chunksize=chunk):
        n_rows += len(ch)
        ch = ch[ch["relevance"] >= GENOME_MIN_RELEVANCE]
        ch = ch[np.isin(ch["movieId"].values, catalog)]
        for t, c in ch["tagId"].value_counts().items():
            counts[int(t)] = counts.get(int(t), 0) + int(c)
    tags = sorted(t for t, c in counts.items() if c >= GENOME_MIN_MEMBERS)
    return tags, {"genome_rows_streamed": int(n_rows), "n_catalog_items": int(len(catalog)),
                  "min_relevance": GENOME_MIN_RELEVANCE, "min_members": GENOME_MIN_MEMBERS,
                  "n_tags_above_relevance": len(counts), "n_tags_kept": len(tags)}


def sbert_sims(texts, tag_names, sb):
    """VERBATIM in behaviour from smoke_trio.sbert_sims(mode='sent'): embed every sentence of the
    text, score each concept by its MAX cosine over the text's sentences. A rendered interview
    question is a single sentence, so 'sent' and 'doc' coincide for it -- the anchor-reference text
    is where the sentence split does work."""
    import re
    SENT_SPLIT = re.compile(r"(?<=[.!?\n])\s+")
    T = sb.encode(tag_names, normalize_embeddings=True, batch_size=128,
                  show_progress_bar=False).astype(np.float32)
    chunks, owner = [], []
    for i, t in enumerate(texts):
        parts = [p.strip() for p in SENT_SPLIT.split(t or "") if p.strip()]
        if not parts:
            parts = [""]
        for p in parts:
            chunks.append(p)
            owner.append(i)
    log(f"[sbert] mode=sent: {len(chunks):,} sentences over {len(texts):,} texts")
    Emb = sb.encode(chunks, normalize_embeddings=True, batch_size=128,
                    show_progress_bar=False).astype(np.float32)
    owner = np.asarray(owner)
    sims = np.full((len(texts), T.shape[0]), -1.0, np.float32)
    for st in range(0, len(chunks), 2048):
        s = Emb[st:st + 2048] @ T.T
        np.maximum.at(sims, owner[st:st + 2048], s)
    return sims


def top_concepts(sims, topn=SBERT_TOPN, mincos=SBERT_MINCOS):
    """VERBATIM from smoke_trio.top_concepts."""
    out = []
    for r in range(sims.shape[0]):
        order = np.argsort(-sims[r])[:topn]
        out.append([(int(c), float(sims[r, c])) for c in order if sims[r, c] >= mincos])
    return out


def bridge(sel, Kc, ni):
    """selected [(cid, cos)] -> (uniform-mean direction, cosine-weighted-mean direction) in the
    MEAN-REMOVED bias-free space. `Kc` is the mean-removed bank; concept c sits at bank row ni + c.
    Subtracting the bank mean before or after averaging is the same operation (the average of the
    centred rows is the centred average), so this is the sheet's convention exactly."""
    if not sel:
        return None, None
    rows = np.stack([Kc[ni + c] for c, _ in sel])
    w = np.asarray([s for _, s in sel], np.float64)
    return rows.mean(0), (w[:, None] * rows).sum(0) / w.sum()


def cos(a, b):
    if a is None or b is None:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return None
    return float(np.dot(a, b) / (na * nb))


def run(args):
    from strategy_channel_suite import build_greedy_env
    from ndcg_tree import build_pool

    GB = git_block(allow_dirty=args.allow_dirty, threads=args.threads)      # HARD RULE 10, first
    log(f"[hardrule10] src/instrument clean={GB['src_instrument_clean']} rev={GB['rev'][:8]} "
        f"threads={GB['threads']}")
    packets = load_json(f"packets{args.tag}.json")["packets"]
    renders = load_json(f"renders{args.render_tag}.json")
    rend = {int(r["id"]): r["question"] for r in renders["renders"]}
    vec_npz = np.load(os.path.join(OUT_DIR, f"c3_poc_vectors{args.tag}.npz"))
    assert len(packets) == 30 and len(rend) == 30 and len(vec_npz.files) == 30
    assert {p["id"] for p in packets} == set(rend), "renders.json ids != packets.json ids"

    # ---- the space: rebuilt exactly as c3_poc_extract.py rebuilds it -----------------------------
    blob = torch.load(args.ckpt, map_location="cpu")
    sha = sha256(args.ckpt)
    assert sha == C1.C0_CKPT_SHA, f"this is not the C0/C1-measured checkpoint (sha {sha})"
    log(f"[ckpt] {os.path.basename(args.ckpt)} epoch={blob['epoch']} sha256 matches C0")

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "paperC/c3poc_re")
    pool, _it, _cc = build_pool(E, 500, 500)
    pool = [(int(a), int(b)) for a, b in pool]
    assert pool == [(int(a), int(b)) for a, b in blob["pool"]], "the env's pool is not the ckpt's"
    ni_env = int(E["ctx"].ni)
    C = int(E["C"])
    full_pool = [(0, i) for i in range(ni_env)] + [(1, c) for c in range(C)]
    assert len(full_pool) == N_BANK
    phi_full, _ = build_phi(E, full_pool, norm=(blob["norm_mu"], blob["norm_sd"]))
    pol0 = Policy(E["ctx"].d, phi_full.shape[1], len(full_pool), arch=blob["arch"],
                  state=blob["state"], use_pos=blob["use_pos"])
    pol0.load_state_dict(blob["model"]); pol0.eval()
    _Eb, Kbank = bank_vectors(pol0, phi_full)
    mu = Kbank.astype(np.float64).mean(0)
    Kc = Kbank.astype(np.float64) - mu
    Kc_unit = unit(Kc)
    log(f"[bank] Kbank {Kbank.shape}; mean removed (bank mean norm {np.linalg.norm(mu):.4f})")

    names, ni, _missing = surface_forms(E)
    assert ni == ni_env

    # ---- ALIGNMENT ASSERT: the concept column, re-derived from the RAW genome ---------------------
    tag_names = tag_name_column(E)
    assert len(tag_names) == C, f"tag-name column has {len(tag_names)} entries, env indexes {C}"
    env_tagids = [int(t) for t in E["sh"]["tags"]]
    rederived, gmeta = rederive_concept_tagids()
    assert rederived == env_tagids, (
        f"the concept column re-derived from genome-scores.csv ({len(rederived)} tags, "
        f"relevance>={GENOME_MIN_RELEVANCE}, >={GENOME_MIN_MEMBERS} catalog members) is not the "
        f"env's sh['tags'] ({len(env_tagids)} tags). First disagreement at position "
        f"{next((i for i, (a, b) in enumerate(zip(rederived, env_tagids)) if a != b), 'length')}")
    log(f"[align] concept column re-derived from the raw genome ({gmeta['genome_rows_streamed']:,} "
        f"rows streamed, {gmeta['n_tags_kept']} tags kept) == sh['tags'] at all {C} positions")
    # ... and the packets' own channel names still agree with the column SBERT will score
    chan = [names[ni + c] for c in range(C)]
    bad = [c for c in range(C) if chan[c] is None or chan[c] != tag_names[c].lower()]
    assert not bad, ("SBERT tag-name column does not match the packets' concept channel at "
                     f"{len(bad)} positions, e.g. {[(c, chan[c], tag_names[c]) for c in bad[:5]]}")

    # ---- CONTROL: the rebuilt bank reproduces the banked anchors ---------------------------------
    worst = 0.0
    for p in packets:
        v = np.asarray(vec_npz[f"p{p['id']:03d}"], np.float64) - mu
        c = Kc_unit @ (v / (np.linalg.norm(v) + 1e-30))
        o = np.argsort(-c)[:len(p["anchors"])]
        got = [names[int(i)] for i in o]
        want = [a["phrase"] for a in p["anchors"]]
        assert got == want, f"packet {p['id']}: rebuilt anchors {got} != banked {want}"
        worst = max(worst, max(abs(round(float(c[i]), 4) - a["cosine"])
                               for i, a in zip(o, p["anchors"])))
    log(f"[gate] all 30 packets' banked top-8 anchors reproduced; max|cos diff| {worst:g}")

    # ---- SBERT ------------------------------------------------------------------------------------
    from sentence_transformers import SentenceTransformer
    sb = SentenceTransformer(SBERT_MODEL)
    log(f"[sbert] {SBERT_MODEL} loaded (HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')})")

    ids = [p["id"] for p in packets]
    q_texts = [rend[i] for i in ids]
    # the anchor-reference "text": the packet's own top-8 anchor phrases, one per sentence, so the
    # sent-mode max-over-sentences pooling sees each phrase on its own terms.
    anch_texts = [". ".join(a["phrase"] for a in p["anchors"]) + "." for p in packets]

    sims_q = sbert_sims(q_texts, tag_names, sb)
    sims_a = sbert_sims(anch_texts, tag_names, sb)
    sel_q = top_concepts(sims_q)
    sel_a = top_concepts(sims_a)

    rows, non_encodable = [], []
    for j, p in enumerate(packets):
        pid = p["id"]
        v = np.asarray(vec_npz[f"p{pid:03d}"], np.float64) - mu
        uq, wq = bridge(sel_q[j], Kc, ni)
        ua, wa = bridge(sel_a[j], Kc, ni)
        if not sel_q[j]:
            non_encodable.append(pid)
        rows.append({
            "id": pid,
            "question": rend[pid],
            "n_concepts_selected": len(sel_q[j]),
            "encodable": bool(sel_q[j]),
            "concepts": [{"cid": c, "tag": tag_names[c], "sbert_cos": round(s, 4)}
                         for c, s in sel_q[j]],
            "cos_uniform": None if uq is None else round(cos(uq, v), 4),
            "cos_weighted": None if wq is None else round(cos(wq, v), 4),
            "anchor_reference": {
                "n_concepts_selected": len(sel_a[j]),
                "concepts": [{"cid": c, "tag": tag_names[c], "sbert_cos": round(s, 4)}
                             for c, s in sel_a[j]],
                "cos_uniform": None if ua is None else round(cos(ua, v), 4),
                "cos_weighted": None if wa is None else round(cos(wa, v), 4)},
            "top_sbert_cos_over_all_tags": round(float(sims_q[j].max()), 4)})

    out = {
        "_comment": ("PER-PACKET geometry ONLY. This stage is blinded: it never read KEY.json, "
                     "questions_for_judge.json or judgments.json, and nothing here says which "
                     "packets are real and which are controls. The join with ground truth happens "
                     "downstream (docs/design/DS_C3_POC.md)."),
        "analysis": "paperC_C3_poc_stage4_reencode",
        "design_sheet": "docs/design/DS_C3_POC.md",
        "produced_by": "src/instrument/c3_poc_reencode.py",
        "citable": True, "test_touched": False, "split": "validation",
        "recipe": {
            "source": "scripts/reddit/smoke_trio.py arms 4/4b (SBERT recipe of record)",
            "sbert_model": SBERT_MODEL, "sbert_pooling": "sentence-max (mode='sent', non-lossy)",
            "sbert_topn": SBERT_TOPN, "sbert_mincos": SBERT_MINCOS,
            "tag_names": "genome-tags.csv `tag`, stripped, NOT lower-cased (smoke_trio's form)",
            "column_order": "env sh['tags'] (sorted tagId, relevance>=0.5, >=30 catalog members)",
            "bridge": ("selected concepts -> their bank embeddings k(phi) at bank row ni+c -> "
                       "uniform mean (primary) and selection-cosine-weighted mean (secondary)"),
            "cosine": "MEAN-REMOVED (the bank mean k is subtracted from both sides)",
            "space": f"bias-free k(phi), dim {Kbank.shape[1]}",
            "anchor_reference": ("the identical bridge applied to the packet's own top-8 anchor "
                                 "phrases as the text -- a RETRIEVAL ceiling, not a render score"),
            "input_field": "renders.json['renders'][*]['question'] only"},
        "asserts": {
            "tag_column_aligned_to_sh_tags_and_packet_channel": True,
            "concept_column_rederived_from_raw_genome": True,
            "concept_column_rederivation": gmeta,
            "n_concepts": C,
            "banked_anchors_reproduced_for_all_packets": True,
            "max_abs_anchor_cosine_diff": float(worst)},
        "bank": {"n_entries": N_BANK, "n_items": ni, "n_concepts": C},
        "checkpoint": {"path": os.path.relpath(args.ckpt, _ROOT), "sha256": sha},
        "git": GB,
        "inputs": {"packets": f"packets{args.tag}.json",
                   "renders": f"renders{args.render_tag}.json",
                   "vectors": f"c3_poc_vectors{args.tag}.npz"},
        "n_packets": len(rows),
        "non_encodable_ids": non_encodable,
        "n_non_encodable": len(non_encodable),
        "rows": rows}
    p_out = os.path.join(OUT_DIR, f"reencode{args.tag}.json")
    with open(p_out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    os.replace(p_out + ".tmp", p_out)

    def fmt(x):
        return "   n/a " if x is None else f"{x:+.4f}"

    log("id | n_c | cos_uniform | cos_weighted | anchor_ref_uniform | anchor_ref_weighted")
    for r in rows:
        log(f"{r['id']:>3} | {r['n_concepts_selected']:>3} | {fmt(r['cos_uniform'])} | "
            f"{fmt(r['cos_weighted'])} | {fmt(r['anchor_reference']['cos_uniform'])} | "
            f"{fmt(r['anchor_reference']['cos_weighted'])}")
    if non_encodable:
        log(f"[non-encodable] {len(non_encodable)} packet(s) selected zero concepts: {non_encodable}")
    log(f"[done] wrote {p_out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Paper C / C3 PoC stage 4 -- docs/design/DS_C3_POC.md")
    ap.add_argument("--ckpt", default=E1_CKPT)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--allow_dirty", action="store_true",
                    help="EMERGENCY ONLY: run with an uncommitted src/instrument (HARD RULE 10)")
    ap.add_argument("--tag", default="",
                    help="read packets<tag>.json / c3_poc_vectors<tag>.npz and write "
                         "reencode<tag>.json; empty is the banked v1 set")
    ap.add_argument("--render_tag", default="",
                    help="renders<render_tag>.json (the rendered questions to re-encode)")
    args = ap.parse_args()
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 100)
    log("# PAPER C / C3 PoC -- STAGE 4 RE-ENCODE. Blinded: per-packet geometry only, no KEY, no "
        "judgments. Local frozen SBERT; no training, no API, no test user.")
    log("#" * 100)
    return run(args)


if __name__ == "__main__":
    main()
