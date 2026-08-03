r"""smoke_trio.py -- FIRST evaluation on the Reddit He-corpus ruler v1 (4,431 threads).

Design sheet: docs/design/DS_REDDIT_LINKING.md ("IN FLIGHT: assembly + smoke trio").
Ruler: data/reddit_he/ruler/ruler_v1.csv.gz (docs/results/REDDIT_RULER_V1.md).

NO LLM API call. NO Reddit API. NO sampling / truncation / capping of any kind (HARD RULE 1):
every one of the 4,431 threads, every GT item, every OP_NAMED item, the full 18,359-item catalogue.
CPU only, torch threads capped (another session may be training on this machine).

ARMS (identical candidate set in every arm: all 18,359 items, the thread's OP_NAMED masked to -inf --
they are given evidence, never targets, and GT already excludes them):
  1. mostpop        -- catalogue ranked by TRAIN popularity (the same `train.sum(axis=0)` count the
                       interview evals use). GATE: NDCG@10 must be <~0.15 and clearly below the
                       tower arms, else the harness is broken (surprise-means-debug-the-harness).
  2. prior          -- the frozen tower with ZERO evidence folded (the cold floor; i26c learns its
                       own popularity prior, Spearman 0.95 vs train popularity).
  3. items          -- prior + the thread's OP_NAMED films folded as LIKED item evidence, the
                       standard item-evidence contract the tower was trained with (level 8 = 4.5
                       stars; sensitivity at levels 7 and 9 reported). Threads with zero OP_NAMED
                       are bit-identical to arm 2 by construction.
  4. items+sbert    -- arm 3 plus request TEXT mapped to genome concepts by SBERT and folded through
                       the C-LITE concept channel (.cache/instrument/cv3_ep07.pt) on the frozen tower.
  4b. sbert only    -- the same concept fold on top of the ZERO-evidence prior.

METRIC: NDCG@10 primary (binary relevance against gt_movieIds, IDCG over min(|GT|,k)),
        NDCG@20 and Recall@10 secondary. Paired bootstrap, 10k resamples over threads.

CONTROLS: (a) shuffle -- permute the (OP_NAMED, request text) evidence assignment across threads
          while holding each thread's candidate mask and targets fixed; tower arms must collapse
          toward the prior. (b) GT-size-driven ceilings. (c) OP-overlap leak stat.

Usage (from C:/dev/phd/casper):
    python scripts/reddit/smoke_trio.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "baselines"))
sys.path.insert(0, str(ROOT / "src" / "instrument"))

PROC = str(ROOT / "data" / "ml-25m" / "proc")
GENOME = ROOT / "data" / "movielens" / "genome-scores.csv"
GTAGS = ROOT / "data" / "movielens" / "genome-tags.csv"
RULER = ROOT / "data" / "reddit_he" / "ruler" / "ruler_v1.csv.gz"
SNAP = ROOT / "models" / "i26c_ep31" / "t2i26c_ep31.pt"
CFOLD = ROOT / ".cache" / "instrument" / "cv3_ep07.pt"
OUT_JSON = ROOT / "data" / "reddit_he" / "ruler" / "smoke_trio_results.json"
OUT_MD = ROOT / "docs" / "results" / "REDDIT_SMOKE_TRIO.md"
OVERLAP_CACHE = ROOT / "data" / "reddit_he" / "ruler" / "op_overlap.json"

LIKE_LEVEL = 8            # 4.5 stars: "the OP named this film approvingly" (sensitivity: 7 and 9)
SBERT_MODEL = "all-MiniLM-L6-v2"
SBERT_TOPN = 4            # pre-registered primary concept budget (sweep 1/2/4/8 reported)
SBERT_MINCOS = 0.30       # cosine floor; a request that matches nothing folds nothing
CONC_VALUE = 1.0          # in-envelope positive answer value (training values live in [-1, 1])
BOOT = 10_000
SEED = 98765
BATCH = 128               # modest, as instructed (another session may be training)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ metrics
def ndcg_recall(scores, targets, ks=(10, 20)):
    """Binary-relevance NDCG@k (IDCG over min(|targets|, k)) + Recall@10 for ONE thread."""
    kmax = max(ks)
    top = np.argpartition(-scores, kmax)[:kmax]
    top = top[np.argsort(-scores[top])]
    hit = np.isin(top, targets).astype(np.float64)
    disc = 1.0 / np.log2(np.arange(2, kmax + 2))
    out = {}
    for k in ks:
        idcg = disc[: min(len(targets), k)].sum()
        out[f"ndcg@{k}"] = float((hit[:k] * disc[:k]).sum() / max(idcg, 1e-12))
    out["recall@10"] = float(hit[:10].sum() / len(targets))
    return out


def bootstrap_ci(delta, n_boot=BOOT, seed=SEED):
    d = np.asarray(delta, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


# ------------------------------------------------------------------ tower
def load_tower(ni, cnt):
    import argparse as _ap

    from i26_encoder import build_i26
    from train_tower_t2 import apply_sign_prior, build_model, load_recvae_teacher
    from train_tower_t2 import log as tlog

    blob = torch.load(SNAP, map_location="cpu")
    arch = blob.get("arch", "i25")
    a = _ap.Namespace(arch=arch, teacher="warm_init", t_hidden=600, t_latent=200, token="film",
                      train_decoder=False, sign_prior=True, unfreeze_emb=False, lr=3e-4,
                      warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
    if arch == "i26":
        src = load_recvae_teacher(ni, hidden=600, latent=200)
        enc, dec, _p, _g = build_i26(ni, src, a, log=tlog)
        apply_sign_prior(enc)
    else:
        enc, dec, _t, _p, _g = build_model(a, ni, cnt)
    enc.load_state_dict(blob["enc"], strict=True)
    dec.load_state_dict(blob["decoder"])
    enc.eval()
    for p in list(enc.parameters()) + list(dec.parameters()):
        p.requires_grad_(False)
    log(f"[tower] {SNAP.name} arch={arch} ep={blob.get('epoch')} val_full={blob.get('val_full')}")
    return enc, dec.weight.detach(), dec.bias.detach()


def fold_items(enc, seqs, batch=BATCH):
    """seqs: list of (sids, levels). Returns (N, d) frozen tower latents."""
    from concept_fold import fold_items_frozen
    return fold_items_frozen(enc, seqs, batch=batch)


# ------------------------------------------------------------------ concept stack
def build_concepts(Wd, ni, m2s):
    from belief_layer import build_concept_dirs
    g = pd.read_csv(GENOME)
    g = g[g["relevance"] >= 0.5]
    g = g[g["movieId"].isin(m2s)]
    g["sid"] = g["movieId"].map(m2s).astype(np.int64)
    members = {int(t): np.sort(sub["sid"].values) for t, sub in g.groupby("tagId")}
    members = {t: mm for t, mm in members.items() if len(mm) >= 30}
    tags = sorted(members.keys())
    d_c, _d_raw, _w_c = build_concept_dirs(Wd, members, tags)
    names = {}
    with open(GTAGS, newline="", encoding="utf-8") as f:
        import csv
        for r in csv.DictReader(f):
            names[int(r["tagId"])] = r["tag"].strip()
    log(f"[concepts] {len(tags)} genome concepts (relevance>=0.5, >=30 members)")
    return tags, [names[t] for t in tags], d_c


def load_cfold(n_tags, d, d_c):
    from concept_fold import ConceptFoldNet
    blob = torch.load(CFOLD, map_location="cpu")
    net = ConceptFoldNet(n_tags, d=d, h=256, conc_init=d_c.numpy(),
                         use_refusal=blob.get("use_refusal", False))
    net.load_state_dict(blob["net"])
    net.eval()
    log(f"[cfold] {CFOLD.name} ep{blob['epoch']} use_refusal={blob.get('use_refusal')} "
        f"snapshot={blob.get('snapshot')}")
    return net


SENT_SPLIT = re.compile(r"(?<=[.!?\n])\s+")


def sbert_sims(texts, tag_names, mode, sb):
    """request text -> (n_threads, n_concepts) cosine matrix against the genome tag NAMES.

    mode='doc'  : one MiniLM embedding of the whole request. MiniLM truncates at 256 word-pieces,
                  so long requests silently lose their tail -- reported, not hidden.
    mode='sent' : embed EVERY sentence of the request (no truncation of the request), score each
                  concept by its MAX cosine over sentences. This is the non-lossy variant and the
                  primary one.
    """
    T = sb.encode(tag_names, normalize_embeddings=True, batch_size=128,
                  show_progress_bar=False).astype(np.float32)
    if mode == "doc":
        E = sb.encode([t if isinstance(t, str) else "" for t in texts], normalize_embeddings=True,
                      batch_size=64, show_progress_bar=False).astype(np.float32)
        sims = E @ T.T
    else:
        chunks, owner = [], []
        for i, t in enumerate(texts):
            parts = [p.strip() for p in SENT_SPLIT.split(t or "") if p.strip()]
            if not parts:
                parts = [""]
            for p in parts:
                chunks.append(p)
                owner.append(i)
        log(f"[sbert] mode=sent: {len(chunks):,} sentences over {len(texts):,} requests")
        E = sb.encode(chunks, normalize_embeddings=True, batch_size=128,
                      show_progress_bar=False).astype(np.float32)
        owner = np.asarray(owner)
        sims = np.full((len(texts), T.shape[0]), -1.0, np.float32)
        for st in range(0, len(chunks), 2048):
            s = E[st:st + 2048] @ T.T
            np.maximum.at(sims, owner[st:st + 2048], s)
    return sims


def top_concepts(sims, topn, mincos):
    out = []
    for r in range(sims.shape[0]):
        order = np.argsort(-sims[r])[:topn]
        out.append([(int(c), float(sims[r, c])) for c in order if sims[r, c] >= mincos])
    return out


# ------------------------------------------------------------------ scoring
def score_arm(Z, Wd, bd, masks, gts, net=None, conc=None, batch=BATCH):
    """Z: (N,d) latents (already item-folded). conc: list of [(cid, cos)] or None.
    Returns dict of per-thread metric arrays."""
    N = Z.shape[0]
    res = {k: np.zeros(N) for k in ("ndcg@10", "ndcg@20", "recall@10")}
    with torch.no_grad():
        for st in range(0, N, batch):
            sl = slice(st, min(st + batch, N))
            z = Z[sl]
            if net is not None:
                from concept_fold import ConceptFoldNet
                cids, cvals, crefs = [], [], []
                for i in range(sl.start, sl.stop):
                    cs = [c for c, _s in (conc[i] or [])]
                    vs = [CONC_VALUE] * len(cs)
                    ci, vv = ConceptFoldNet.canonical_order(cs, vs)
                    cids.append(ci); cvals.append(vv); crefs.append([0.0] * len(ci))
                z = net(z, cids, cvals, crefs)
            S = (z @ Wd.T + bd).numpy().astype(np.float64)
            for j, i in enumerate(range(sl.start, sl.stop)):
                s = S[j]
                if len(masks[i]):
                    s[masks[i]] = -np.inf
                m = ndcg_recall(s, gts[i])
                for k in res:
                    res[k][i] = m[k]
    return res


def score_static(vec, masks, gts):
    """One user-independent score vector (Most-Popular) over all threads."""
    N = len(gts)
    res = {k: np.zeros(N) for k in ("ndcg@10", "ndcg@20", "recall@10")}
    for i in range(N):
        s = vec.copy()
        if len(masks[i]):
            s[masks[i]] = -np.inf
        m = ndcg_recall(s, gts[i])
        for k in res:
            res[k][i] = m[k]
    return res


def summarise(res, sub=None):
    idx = np.arange(len(res["ndcg@10"])) if sub is None else np.asarray(sub)
    return {k: round(float(np.mean(v[idx])), 4) for k, v in res.items()}


# ------------------------------------------------------------------ OP-overlap leak stat
def op_overlap_stat(thread_ids):
    """share of OP_NAMED items that ALSO appear as raw reply mentions (pre-subtraction).

    Recomputed from the adjudication artefacts with assemble_ruler's OWN loaders, so the number is
    the same pipeline that built the ruler. Cached, because it re-scans title.basics.tsv.
    """
    if OVERLAP_CACHE.exists():
        return json.loads(OVERLAP_CACHE.read_text())
    sys.path.insert(0, str(ROOT / "scripts" / "reddit"))
    import assemble_ruler as AR
    from collections import defaultdict
    brows, threads = AR.load_batches(100)
    preds = AR.load_preds(100)
    tq = set()
    for r in brows:
        p = preds[r["id"]]
        if p.startswith("TITLE:"):
            nm, _y = AR.parse_title_pred(p)
            if nm:
                tq.add(AR.nnorm(nm))
    idx = AR.build_title_index(tq)
    in_cat, _c = AR.load_catalogue()
    per = defaultdict(lambda: {"op": set(), "reply": set()})
    cache = {}
    for r in brows:
        p = preds[r["id"]]
        if not p or p == "NONE":
            continue
        if re.fullmatch(r"tt\d{7,8}", p):
            tc = p
        elif p.startswith("TITLE:"):
            nm, yr = AR.parse_title_pred(p)
            if nm is None:
                continue
            key = (AR.nnorm(nm), yr)
            if key not in cache:
                cache[key] = AR.resolve_title(nm, yr, idx)
            tc = cache[key][0]
            if tc is None:
                continue
        else:
            continue
        mid = in_cat.get(tc)
        if mid is None:
            continue
        per[r["thread_id"]]["op" if r["context"] in AR.SEEKER_CTX else "reply"].add(mid)
    acc, _t = AR.load_accepted(threads)
    for tid, ctx, _span, tc in acc:
        mid = in_cat.get(tc)
        if mid is None:
            continue
        per[tid]["op" if ctx in AR.SEEKER_CTX else "reply"].add(mid)
    keep = [t for t in thread_ids if t in per]
    n_op = sum(len(per[t]["op"]) for t in keep)
    n_ov = sum(len(per[t]["op"] & per[t]["reply"]) for t in keep)
    n_reply = sum(len(per[t]["reply"]) for t in keep)
    out = {"threads": len(keep), "op_named_items": n_op, "raw_reply_items": n_reply,
           "op_also_in_replies": n_ov,
           "share_of_op_repeated_by_crowd": round(n_ov / max(n_op, 1), 4),
           "share_of_raw_reply_removed_as_op": round(n_ov / max(n_reply, 1), 4)}
    OVERLAP_CACHE.write_text(json.dumps(out, indent=2))
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_overlap", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    import metrics as M
    from train_tower_t2 import compute_head_mask

    # ---------------- data
    meta = M.load_meta(PROC); ni = meta["n_items"]
    train = M.load_train(ni, PROC)
    _head, cnt = compute_head_mask(train, ni)
    usid = pd.read_csv(os.path.join(PROC, "unique_sid.txt"), header=None)[0].tolist()
    m2s = {int(m): i for i, m in enumerate(usid)}
    assert len(m2s) == ni

    df = pd.read_csv(RULER)
    N = len(df)
    gts, ops, texts = [], [], []
    for _i, row in df.iterrows():
        gts.append(np.array([m2s[int(x)] for x in str(row.gt_movieIds).split()], np.int64))
        o = str(row.op_named_movieIds) if isinstance(row.op_named_movieIds, str) else ""
        ops.append(np.array([m2s[int(x)] for x in o.split()], np.int64))
        texts.append(row.request_text if isinstance(row.request_text, str) else "")
    log(f"[ruler] {N} threads | GT items {sum(len(g) for g in gts):,} | "
        f"OP_NAMED {sum(len(o) for o in ops):,} | threads with >=1 anchor "
        f"{sum(1 for o in ops if len(o)):,}")
    anch = np.array([len(o) > 0 for o in ops])
    masks = ops                                    # candidate mask = the thread's OP_NAMED, all arms

    # ---------------- tower
    enc, Wd, bd = load_tower(ni, cnt)
    d = enc.d_out
    Z_prior = fold_items(enc, [(np.empty(0, np.int64), np.empty(0, np.int64))] * N)
    seq_items = [(o, np.full(len(o), LIKE_LEVEL, np.int64)) for o in ops]
    Z_items = fold_items(enc, seq_items)
    zero = torch.from_numpy(~anch)
    assert torch.allclose(Z_items[zero], Z_prior[zero]), "zero-anchor threads must equal the prior"

    # ---------------- concepts
    tags, tag_names, d_c = build_concepts(Wd, ni, m2s)
    net = load_cfold(len(tags), d, d_c)
    from sentence_transformers import SentenceTransformer
    sb = SentenceTransformer(SBERT_MODEL)
    sims = sbert_sims(texts, tag_names, "sent", sb)
    sims_doc = sbert_sims(texts, tag_names, "doc", sb)
    conc = top_concepts(sims, SBERT_TOPN, SBERT_MINCOS)
    conc_doc = top_concepts(sims_doc, SBERT_TOPN, SBERT_MINCOS)
    n_folded = float(np.mean([len(c) for c in conc]))
    log(f"[sbert] sent-mode: mean {n_folded:.2f} concepts folded / thread "
        f"(top{SBERT_TOPN}, cos>={SBERT_MINCOS})")

    # ---------------- arms
    results = {}
    log("scoring mostpop ...")
    results["mostpop"] = score_static(cnt.astype(np.float64), masks, gts)
    log("scoring prior ...")
    results["prior"] = score_arm(Z_prior, Wd, bd, masks, gts)
    log("scoring items ...")
    results["items"] = score_arm(Z_items, Wd, bd, masks, gts)
    log("scoring items+sbert ...")
    results["items_sbert"] = score_arm(Z_items, Wd, bd, masks, gts, net=net, conc=conc)
    log("scoring sbert-only ...")
    results["sbert_only"] = score_arm(Z_prior, Wd, bd, masks, gts, net=net, conc=conc)
    log("scoring items+sbert (doc pooling) ...")
    results["items_sbert_doc"] = score_arm(Z_items, Wd, bd, masks, gts, net=net, conc=conc_doc)

    # sensitivity: fold level for the item anchors
    sens_level = {}
    for lv in (7, 9):
        Zl = fold_items(enc, [(o, np.full(len(o), lv, np.int64)) for o in ops])
        sens_level[f"level{lv}"] = summarise(score_arm(Zl, Wd, bd, masks, gts))
        log(f"[sens] items @ level {lv}: {sens_level[f'level{lv}']}")

    # sensitivity: concept budget
    sens_topn = {}
    for topn in (1, 2, 8):
        c2 = top_concepts(sims, topn, SBERT_MINCOS)
        sens_topn[f"top{topn}"] = summarise(score_arm(Z_items, Wd, bd, masks, gts, net=net, conc=c2))
        log(f"[sens] items+sbert top{topn}: {sens_topn[f'top{topn}']}")

    # ---------------- controls
    log("control: shuffle (evidence permuted across threads, targets + mask held fixed) ...")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    Z_sh = fold_items(enc, [seq_items[p] for p in perm])
    conc_sh = [conc[p] for p in perm]
    ctrl = {"items_shuffled": summarise(score_arm(Z_sh, Wd, bd, masks, gts)),
            "items_sbert_shuffled": summarise(score_arm(Z_sh, Wd, bd, masks, gts,
                                                        net=net, conc=conc_sh))}

    ceil = {"ndcg@10": 1.0, "ndcg@20": 1.0,
            "recall@10_mean_ceiling": round(float(np.mean([min(10, len(g)) / len(g) for g in gts])), 4),
            "recall@10_ceiling_anchored": round(
                float(np.mean([min(10, len(g)) / len(g) for g, a in zip(gts, anch) if a])), 4),
            "recall@10_ceiling_zero_anchor": round(
                float(np.mean([min(10, len(g)) / len(g) for g, a in zip(gts, anch) if not a])), 4),
            "note": "NDCG@k ceiling is 1.0 for every thread (|GT| >= 5, IDCG over min(|GT|,k)); "
                    "Recall@10 is capped by GT size (mean |GT| = 14.9)."}

    # ---------------- bootstrap deltas
    deltas = {}
    for name, (a_, b_) in {"items_vs_mostpop": ("items", "mostpop"),
                           "items_vs_prior": ("items", "prior"),
                           "sbert_vs_items": ("items_sbert", "items"),
                           "prior_vs_mostpop": ("prior", "mostpop"),
                           "sbertonly_vs_prior": ("sbert_only", "prior")}.items():
        deltas[name] = {}
        for met in ("ndcg@10", "ndcg@20", "recall@10"):
            dd = results[a_][met] - results[b_][met]
            mean, ci = bootstrap_ci(dd)
            deltas[name][met] = {"mean": round(mean, 5), "ci95": [round(ci[0], 5), round(ci[1], 5)],
                                 "separated": bool(ci[0] > 0 or ci[1] < 0)}
        # anchored subset only, primary metric
        dd = (results[a_]["ndcg@10"] - results[b_]["ndcg@10"])[anch]
        mean, ci = bootstrap_ci(dd)
        deltas[name]["ndcg@10_anchored_only"] = {"mean": round(mean, 5),
                                                 "ci95": [round(ci[0], 5), round(ci[1], 5)],
                                                 "separated": bool(ci[0] > 0 or ci[1] < 0)}

    # ---------------- gate
    gate_pop = results["mostpop"]["ndcg@10"].mean()
    gate = {"mostpop_ndcg@10": round(float(gate_pop), 4),
            "below_0.15": bool(gate_pop <= 0.15),
            "below_items_arm": bool(gate_pop < results["items"]["ndcg@10"].mean())}
    log(f"[GATE] MostPop NDCG@10 = {gate_pop:.4f} "
        f"{'PASS' if gate['below_0.15'] and gate['below_items_arm'] else 'FAIL'}")

    overlap = None if args.skip_overlap else op_overlap_stat(list(df.thread_id))
    if overlap:
        log(f"[overlap] {overlap}")

    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ruler": str(RULER.relative_to(ROOT)), "n_threads": N,
        "tower": str(SNAP.relative_to(ROOT)), "concept_net": str(CFOLD.relative_to(ROOT)),
        "recipe": {"like_level": LIKE_LEVEL, "sbert_model": SBERT_MODEL, "sbert_topn": SBERT_TOPN,
                   "sbert_mincos": SBERT_MINCOS, "sbert_pooling": "sentence-max (non-lossy)",
                   "concept_value": CONC_VALUE, "n_concepts": len(tags),
                   "mean_concepts_folded": round(n_folded, 3),
                   "candidate_mask": "thread OP_NAMED -> -inf in EVERY arm"},
        "n_anchored": int(anch.sum()), "n_zero_anchor": int((~anch).sum()),
        "gate_mostpop": gate,
        "arms_overall": {k: summarise(v) for k, v in results.items()},
        "arms_anchored": {k: summarise(v, np.flatnonzero(anch)) for k, v in results.items()},
        "arms_zero_anchor": {k: summarise(v, np.flatnonzero(~anch)) for k, v in results.items()},
        "paired_bootstrap": deltas,
        "controls": {"shuffle": ctrl, "ceilings": ceil},
        "sensitivity": {"item_fold_level": sens_level, "sbert_topn": sens_topn},
        "op_overlap_leak_stat": overlap,
        "runtime_minutes": round((time.time() - t_start) / 60, 1),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    log(f"-> {OUT_JSON}")
    write_md(out)
    log(f"-> {OUT_MD}  ({out['runtime_minutes']} min)")


ARM_LABEL = {"mostpop": "1. Most-Popular", "prior": "2. Tower prior (no evidence)",
             "items": "3. Tower + items (OP_NAMED)", "items_sbert": "4. Tower + items + SBERT concepts",
             "sbert_only": "4b. Tower + SBERT concepts only",
             "items_sbert_doc": "4c. arm 4, doc-pooled SBERT"}


def write_md(o):
    def tbl(block, n):
        rows = [f"| arm | NDCG@10 | NDCG@20 | Recall@10 |", "|---|---|---|---|"]
        for k, lab in ARM_LABEL.items():
            m = block[k]
            rows.append(f"| {lab} | {m['ndcg@10']:.4f} | {m['ndcg@20']:.4f} | {m['recall@10']:.4f} |")
        return f"({n} threads)\n\n" + "\n".join(rows)

    d = o["paired_bootstrap"]

    def dline(name):
        x = d[name]["ndcg@10"]
        return (f"| {name} | {x['mean']:+.4f} | [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] | "
                f"{'SEPARATED' if x['separated'] else 'not separated'} |")

    md = f"""# Reddit smoke trio -- first evaluation on the He-corpus ruler v1

Run {o['generated']} by `scripts/reddit/smoke_trio.py` ({o['runtime_minutes']} min, CPU, 4 threads).
Ruler `{o['ruler']}` -- **all {o['n_threads']} threads, no sampling** (HARD RULE 1). No LLM call, no
Reddit API. Tower `{o['tower']}` (frozen, read-only import). Concept channel `{o['concept_net']}`.

## Protocol

* Candidate set = the full 18,359-item catalogue in every arm, with **the thread's OP_NAMED items
  masked to -inf in every arm** (given evidence, never targets; GT already excludes them). Identical
  candidate sets across arms is what makes the paired deltas a model comparison.
* Item evidence = the thread's `op_named_movieIds` folded as LIKED tokens at level
  {o['recipe']['like_level']} (4.5 stars). Threads with zero anchors are bit-identical to the prior
  arm (asserted in code).
* SBERT recipe (arm 4): `{o['recipe']['sbert_model']}`, request text split into sentences and each
  sentence embedded (no truncation of the request); each genome concept scored by its MAX cosine over
  the request's sentences; top-{o['recipe']['sbert_topn']} concepts with cosine >=
  {o['recipe']['sbert_mincos']} folded as positive answers (value {o['recipe']['concept_value']},
  refusal flag 0) through the C-LITE `ConceptFoldNet` on the frozen tower's latent. Vocabulary =
  {o['recipe']['n_concepts']} genome concepts; mean {o['recipe']['mean_concepts_folded']} concepts
  folded per thread. Arm 4c pools the whole request into one embedding instead (MiniLM truncates at
  256 word-pieces, which is why sentence pooling is the primary).
* Metrics: binary relevance against `gt_movieIds`, IDCG over min(|GT|,k). Paired bootstrap,
  10,000 resamples over threads.

## Gate: Most-Popular

MostPop NDCG@10 = **{o['gate_mostpop']['mostpop_ndcg@10']:.4f}** -- <=0.15:
{o['gate_mostpop']['below_0.15']}; below the items arm: {o['gate_mostpop']['below_items_arm']}.

## Arms -- overall

{tbl(o['arms_overall'], o['n_threads'])}

## Split by anchor coverage

**>=1 OP_NAMED item** {tbl(o['arms_anchored'], o['n_anchored'])}

**zero anchors** {tbl(o['arms_zero_anchor'], o['n_zero_anchor'])}

## Paired bootstrap (NDCG@10, 10k resamples over threads)

| delta | mean | 95% CI | verdict |
|---|---|---|---|
{chr(10).join(dline(k) for k in d)}

Anchored-subset-only deltas, NDCG@10:

| delta | mean | 95% CI | verdict |
|---|---|---|---|
{chr(10).join(f"| {k} | {d[k]['ndcg@10_anchored_only']['mean']:+.4f} | [{d[k]['ndcg@10_anchored_only']['ci95'][0]:+.4f}, {d[k]['ndcg@10_anchored_only']['ci95'][1]:+.4f}] | {'SEPARATED' if d[k]['ndcg@10_anchored_only']['separated'] else 'not separated'} |" for k in d)}

## Controls

**Shuffle** -- the (OP_NAMED, request) evidence is permuted across threads while each thread keeps
its own targets and candidate mask, so any collapse is the evidence losing its owner, not a change
of candidate set:

| arm | NDCG@10 | NDCG@20 | Recall@10 |
|---|---|---|---|
| items, shuffled | {o['controls']['shuffle']['items_shuffled']['ndcg@10']:.4f} | {o['controls']['shuffle']['items_shuffled']['ndcg@20']:.4f} | {o['controls']['shuffle']['items_shuffled']['recall@10']:.4f} |
| items+SBERT, shuffled | {o['controls']['shuffle']['items_sbert_shuffled']['ndcg@10']:.4f} | {o['controls']['shuffle']['items_sbert_shuffled']['ndcg@20']:.4f} | {o['controls']['shuffle']['items_sbert_shuffled']['recall@10']:.4f} |
| (prior, for reference) | {o['arms_overall']['prior']['ndcg@10']:.4f} | {o['arms_overall']['prior']['ndcg@20']:.4f} | {o['arms_overall']['prior']['recall@10']:.4f} |

**Ceilings** -- NDCG@10 and NDCG@20 have ceiling 1.0 for every thread (|GT| >= 5). Recall@10 is
GT-size-capped: mean ceiling {o['controls']['ceilings']['recall@10_mean_ceiling']:.4f} overall,
{o['controls']['ceilings']['recall@10_ceiling_anchored']:.4f} on anchored threads,
{o['controls']['ceilings']['recall@10_ceiling_zero_anchor']:.4f} on zero-anchor threads.

**OP-overlap leak stat** -- {json.dumps(o['op_overlap_leak_stat'])}

## Sensitivity

Item fold level (arm 3): {json.dumps(o['sensitivity']['item_fold_level'])}

SBERT concept budget (arm 4): {json.dumps(o['sensitivity']['sbert_topn'])}

## Verdict

(written by hand after the run; a re-run preserves whatever already sits under this heading)
"""
    # A re-run must not silently delete an authored verdict: keep the existing tail from "## Verdict".
    if OUT_MD.exists():
        old = OUT_MD.read_text(encoding="utf-8")
        cut = old.find("## Harness validation")
        if cut < 0:
            cut = old.find("## Verdict")
        if cut >= 0:
            md = md[: md.find("## Verdict")] + old[cut:]
    OUT_MD.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
