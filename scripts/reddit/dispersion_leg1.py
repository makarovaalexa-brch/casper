r"""dispersion_leg1.py -- LEG 1 of the ADAPTIVITY-ON-REDDIT protocol (DS_REDDIT_LINKING.md).

QUESTION (pre-registered in the design sheet): do REAL requests place users at states that disagree
about the OPTIMAL NEXT QUESTION, where the standard ML-25M interview openers do not?

THREE MEASUREMENTS, one criterion, one pair of banks:
  A. Reddit, request-placed  -- the 4,431 ruler threads, folded exactly as smoke_trio.py arm 4
                                (OP_NAMED items @ level 8 + SBERT->genome concepts through C-LITE).
  B. ML-25M cold (k=0)       -- one state for everybody; the trivial dispersion floor (verified, not
                                assumed).
  C. ML-25M after the standard openers -- all 10,000 canonical VAL users, the first 1 and the first 2
                                questions of the entropy0 bank answered BEHAVIOURALLY (real half-star
                                ratings; asked-but-unrated -> UNSEEN token, the i26 interview contract
                                used by src/baselines/run_interview_table.py).

CRITERION (the M11 rule: dispersion must be read off a LOSS-COUPLED criterion, never a pure-information
one). Repo audit found NO existing per-state expected-NDCG machinery: the interview evals use static
lit orders (interview_strategies.py: popularity / pure_entropy / entropy0 / helf), the greedy machinery
(strategy_channel_suite._greedy_avg, greedy_polish.py) optimises a POPULATION-mean NDCG over a shared
sequence, and the only per-state adaptive rule in the tree
(adaptive_concept_arms.py, arm "greedy realizable") scores |<d_c, z>| * std|v|(c) -- an expected
RESPONSE MAGNITUDE, i.e. an information criterion, which M11 forbids. So the criterion is built here,
as the closest computable thing to "expected NDCG@10 gain of asking q at this state":

    p(. | z)  = softmax(z Wd^T + bd)                     the tower's own multinomial predictive -- this
                                                         IS its training loss, so "relevance" below is
                                                         the model's own belief, never the ground truth
    V(z)      = sum_{r=1..10} disc_r * p_(r)(.|z)        expected DCG@10 of the slate the recommender
                                                         would actually show at z (IDCG = 1 under the
                                                         single-draw relevance model, so DCG = NDCG)
    U(q | z)  = sum_a  P(a | z) * V( z (+) (q,a) )       expected post-answer NDCG@10, marginalised over
                                                         the answer under the behavioural answer model
    crit(q|z) = U(q|z) - V(z)      ->    q*(z) = argmax_q crit(q|z)

V(z) is constant in q, so argmax U == argmax crit; the subtraction only makes the number readable.
The criterion is answer-dependent by construction (it folds each answer and re-decodes), so it is NOT
a Sigma-only criterion and its dispersion is not 0 by the theorem.

ANSWER MODEL P(a|z) -- behavioural, state-coupled, no ground truth read anywhere:
  * ITEM question i, alphabet {LIKED (level 8), UNSEEN}: under the multinomial model a user with L
    interactions draws L times from p(.|z), so P(seen) = 1 - (1 - p_i)^L with L = the median rated-item
    count of a canonical VAL user. UNSEEN is NOT a no-op: i26 folds it through the exposure branch,
    which is the contract run_interview_table.py evaluates ("levels_unseen").
  * CONCEPT question c, alphabet {LIKE (+1), DISLIKE (-1), REFUSAL}: answerable iff >= 2 rated members
    (the repo's structural rule, tradeoff_ledger.build_shared); with member mass m_c = sum_{i in c} p_i,
    P(answerable) = 1 - (1-m)^L - L m (1-m)^(L-1). Sign from the state's own lift over the cold prior,
    l_c = m_c(z) / m_c(z_prior), P(like) = l/(1+l) -- exactly 0.5 at the cold state, > 0.5 where the
    state already loads the concept. REFUSAL folds as a refusal token (cv3 was trained with
    use_refusal=True), so it too moves the state.

BANKS (identical in all three measurements):
  * CONCEPTS: all 1,031 genome concepts with relevance >= 0.5 and >= 30 members -- the full bank, no
    reduction (smoke_trio.build_concepts, verbatim).
  * ITEMS: the union of the top-200 of each of the three standard interview orders (popularity,
    entropy0, HELF) as those orders are built by src/baselines/interview_strategies.py. This strictly
    contains every item question any standard bank asks at any budget <= 200, hence every question in
    the interview tables (max budget 16). It is a BANK CHOICE, printed with its size and with the
    verification that it covers the first 16 of each order; it is not a per-user candidate cap.
    Rationale for not using all 18,359 items: the criterion needs one encoder fold + one full decode
    per (state, candidate, answer); 18,359 items x 2 answers x ~4.5k distinct states is ~1.6e15 flops.

CONTROLS (Hard Rule 5):
  (i)   shuffle -- reassign each thread a random other thread's state. The argmax multiset is a property
        of the state SET, so every dispersion statistic must be bit-identical. Reported together with
        the duplicate-state census (threads sharing an exact state hash AND argmax).
  (ii)  existential -- two hand-built opposite synthetic states (top horror likes vs top romance likes)
        must produce different argmaxes in both banks.
  (iii) determinism -- 100 random threads recomputed from scratch must return identical argmaxes.

NO LLM CALL. NO REDDIT API. CPU only, 4 torch threads. READ-ONLY on src/instrument/ (imported, never
modified). No sampling of threads or users (HARD RULE 1).

Usage (from C:/dev/phd/casper):
    python scripts/reddit/dispersion_leg1.py [--bench]
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
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
OUT_JSON = ROOT / "data" / "reddit_he" / "ruler" / "dispersion_leg1.json"
OUT_MD = ROOT / "docs" / "results" / "REDDIT_DISPERSION_LEG1.md"

LIKE_LEVEL = 8              # smoke_trio recipe, verbatim
UNSEEN_LEVEL = 10           # i26_encoder.UNSEEN_LEVEL sentinel
SBERT_MODEL = "all-MiniLM-L6-v2"
SBERT_TOPN = 4
SBERT_MINCOS = 0.30
CONC_VALUE = 1.0
ITEM_BANK_PER_ORDER = 200
SEED = 98765
FOLD_BATCH = 192

DISC = torch.tensor(1.0 / np.log2(np.arange(2, 12)), dtype=torch.float32)   # 10 ranks


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# =============================================================================== state container
class States:
    """A de-duplicated set of interview states.

    A state = (item tokens folded through the tower) + (concept answers folded through C-LITE).
    Two states are the SAME state iff both token lists are identical -- which is what makes arm C
    cheap and is itself the headline finding, so the de-duplication is exact, never approximate.
    """

    def __init__(self):
        self.key2idx = {}
        self.item_ids = []      # list[np.ndarray]
        self.item_lvl = []      # list[np.ndarray]
        self.conc = []          # list[(cids, vals, refs)] already in canonical order

    @staticmethod
    def _key(sids, lvls, conc):
        o = np.argsort(sids)
        a = ",".join(f"{int(s)}:{int(l)}" for s, l in zip(np.asarray(sids)[o], np.asarray(lvls)[o]))
        c = ",".join(f"{int(i)}:{float(v):.4f}:{int(r)}"
                     for i, v, r in sorted(zip(*conc)) ) if len(conc[0]) else ""
        return a + "|" + c

    def add(self, sids, lvls, conc):
        k = self._key(sids, lvls, conc)
        j = self.key2idx.get(k)
        if j is None:
            j = len(self.item_ids)
            self.key2idx[k] = j
            self.item_ids.append(np.asarray(sids, np.int64))
            self.item_lvl.append(np.asarray(lvls, np.int64))
            self.conc.append(conc)
        return j, k

    def __len__(self):
        return len(self.item_ids)


def canon(cids, vals, refs):
    """ConceptFoldNet.canonical_order (DESC value, ties by concept id), carrying the refusal flag."""
    idx = sorted(range(len(cids)), key=lambda i: (-float(vals[i]), int(cids[i])))
    return ([int(cids[i]) for i in idx], [float(vals[i]) for i in idx], [float(refs[i]) for i in idx])


# =============================================================================== model plumbing
def load_tower(ni, cnt):
    """smoke_trio.load_tower, verbatim."""
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


def build_concepts(Wd, ni, m2s):
    """smoke_trio.build_concepts, verbatim (relevance>=0.5, >=30 members -> the 1,031-concept bank)."""
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
    import csv
    with open(GTAGS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            names[int(r["tagId"])] = r["tag"].strip()
    log(f"[concepts] {len(tags)} genome concepts (relevance>=0.5, >=30 members)")
    return tags, [names[t] for t in tags], d_c, members


def load_cfold(n_tags, d, d_c):
    from concept_fold import ConceptFoldNet
    blob = torch.load(CFOLD, map_location="cpu")
    net = ConceptFoldNet(n_tags, d=d, h=256, conc_init=d_c.numpy(),
                         use_refusal=blob.get("use_refusal", False))
    net.load_state_dict(blob["net"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    log(f"[cfold] {CFOLD.name} ep{blob['epoch']} use_refusal={blob.get('use_refusal')}")
    return net


SENT_SPLIT = re.compile(r"(?<=[.!?\n])\s+")


def sbert_sims(texts, tag_names, sb):
    """smoke_trio sent-mode (non-lossy: every sentence embedded, concept scored by MAX cosine)."""
    T = sb.encode(tag_names, normalize_embeddings=True, batch_size=128,
                  show_progress_bar=False).astype(np.float32)
    chunks, owner = [], []
    for i, t in enumerate(texts):
        parts = [p.strip() for p in SENT_SPLIT.split(t or "") if p.strip()] or [""]
        for p in parts:
            chunks.append(p)
            owner.append(i)
    log(f"[sbert] {len(chunks):,} sentences over {len(texts):,} requests")
    E = sb.encode(chunks, normalize_embeddings=True, batch_size=128,
                  show_progress_bar=False).astype(np.float32)
    owner = np.asarray(owner)
    sims = np.full((len(texts), T.shape[0]), -1.0, np.float32)
    for st in range(0, len(chunks), 2048):
        s = E[st:st + 2048] @ T.T
        np.maximum.at(sims, owner[st:st + 2048], s)
    return sims


# =============================================================================== criterion core
class Criterion:
    def __init__(self, enc, net, Wd, bd, Mm, ni, L):
        self.enc, self.net, self.Wd, self.bd = enc, net, Wd, bd
        self.Mm = Mm            # (ni, C) sparse membership
        self.ni, self.L = ni, L

    # -------- tower fold + concept fold --------
    def fold(self, seqs, concs):
        """seqs: list[(sids, lvls)] ; concs: list[(cids, vals, refs)] -> (B,d) latents."""
        from concept_fold import fold_items_frozen
        Zi = fold_items_frozen(self.enc, seqs, batch=FOLD_BATCH)
        with torch.no_grad():
            Z = self.net(Zi, [c[0] for c in concs], [c[1] for c in concs], [c[2] for c in concs])
        return Z

    def fold_conc_only(self, Zi, concs):
        with torch.no_grad():
            return self.net(Zi, [c[0] for c in concs], [c[1] for c in concs], [c[2] for c in concs])

    def tower_only(self, seqs):
        from concept_fold import fold_items_frozen
        return fold_items_frozen(self.enc, seqs, batch=FOLD_BATCH)

    # -------- V(z) = expected DCG@10 under the model's own predictive --------
    def V(self, Z, chunk=512):
        out = torch.empty(Z.shape[0])
        with torch.no_grad():
            for st in range(0, Z.shape[0], chunk):
                S = Z[st:st + chunk] @ self.Wd.T + self.bd
                lse = torch.logsumexp(S, dim=1, keepdim=True)
                top = torch.topk(S, 10, dim=1).values
                out[st:st + chunk] = (torch.exp(top - lse) * DISC).sum(1)
        return out

    def probs(self, Z, chunk=512):
        """full predictive p(.|z) -- only ever called once per distinct STATE (not per candidate)."""
        out = np.empty((Z.shape[0], self.ni), np.float32)
        with torch.no_grad():
            for st in range(0, Z.shape[0], chunk):
                S = Z[st:st + chunk] @ self.Wd.T + self.bd
                out[st:st + chunk] = torch.softmax(S, dim=1).numpy()
        return out


def dispersion_stats(argmax, n_top=10):
    a = np.asarray(argmax)
    vals, cnts = np.unique(a, return_counts=True)
    p = cnts / cnts.sum()
    order = np.argsort(-cnts)
    ent = float(-(p * np.log2(p)).sum())
    return {"n": int(len(a)), "n_distinct": int(len(vals)),
            "top_share": float(cnts.max() / cnts.sum()),
            "entropy_bits": round(ent, 4),
            "perplexity": round(float(2.0 ** ent), 3),
            "top10_cum_share": float(cnts[order][:n_top].sum() / cnts.sum()),
            "top5": [[int(vals[i]), int(cnts[i])] for i in order[:5]]}


# =============================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="time 8 states and stop")
    args = ap.parse_args()
    t0 = time.time()

    import metrics as M
    from run_battery_phaseA import build_real_ctx
    from strategy_ladder import train_entropies
    from concept_fold import build_member_matrix

    # ---------------------------------------------------------------- ctx (VAL cohort) + tower
    ctx = build_real_ctx(str(SNAP), split="validation")
    ni = ctx.ni
    enc, Wd, bd = ctx.enc, ctx.decoder.weight.detach().float(), ctx.decoder.bias.detach().float()
    d = enc.d_out
    usid = pd.read_csv(os.path.join(PROC, "unique_sid.txt"), header=None)[0].tolist()
    m2s = {int(m): i for i, m in enumerate(usid)}
    assert len(m2s) == ni

    tags, tag_names, d_c, members = build_concepts(Wd, ni, m2s)
    net = load_cfold(len(tags), d, d_c)
    Mm = build_member_matrix(members, tags, ni)                     # (ni, C)
    C = Criterion(enc, net, Wd, bd, Mm, ni, L=0)

    # ---------------------------------------------------------------- banks
    H, H0, _nr = train_entropies(ctx)
    Hn = H / max(H.max(), 1e-9)
    Fn = np.log1p(ctx.cnt) / max(np.log1p(ctx.cnt).max(), 1e-9)
    helf = 2.0 * Hn * Fn / np.clip(Hn + Fn, 1e-9, None)
    ctx.raw = None                                                  # free the 25M-row frame
    ord_pop, ord_e0, ord_helf = np.argsort(-ctx.cnt), np.argsort(-H0), np.argsort(-helf)
    item_bank = np.array(sorted(set(ord_pop[:ITEM_BANK_PER_ORDER].tolist())
                                | set(ord_e0[:ITEM_BANK_PER_ORDER].tolist())
                                | set(ord_helf[:ITEM_BANK_PER_ORDER].tolist())), np.int64)
    ibank_pos = {int(s): j for j, s in enumerate(item_bank)}
    cover = {k: bool(set(v[:16].tolist()) <= set(item_bank.tolist()))
             for k, v in {"popularity": ord_pop, "entropy0": ord_e0, "helf": ord_helf}.items()}
    log(f"[bank] items = {len(item_bank)} (union of top-{ITEM_BANK_PER_ORDER} of pop/entropy0/helf); "
        f"covers first-16 of each order: {cover}")
    nI, nC = len(item_bank), len(tags)

    # answer-model constant L = median rated-item count of a canonical VAL user
    prof = np.array([len(s) for s, _l in ctx.allb], np.int64)
    Lmed = int(np.median(prof))
    C.L = Lmed
    log(f"[answer-model] L (median VAL user rated items) = {Lmed}")

    # ---------------------------------------------------------------- STATES
    ST = States()
    empty_c = ([], [], [])

    # --- B: ML-25M cold, k=0 (one state, shared by all 10,000 VAL users)
    b_idx, _k = ST.add(np.empty(0, np.int64), np.empty(0, np.int64), empty_c)
    prior_state = b_idx

    # --- C: ML-25M after the first 1 / 2 entropy0 questions, answered behaviourally
    lvl_lookup = [dict(zip(s.tolist(), l.tolist())) for s, l in ctx.allb]   # te already excluded
    asks = [int(ord_e0[0]), int(ord_e0[1])]
    c_states = {1: [], 2: []}
    for r in range(ctx.n):
        dlk = lvl_lookup[r]
        sids, lvls = [], []
        for step, q in enumerate(asks, start=1):
            sids.append(q)
            lvls.append(int(dlk[q]) if q in dlk else UNSEEN_LEVEL)
            if step in c_states:
                j, _k = ST.add(np.array(sids, np.int64), np.array(lvls, np.int64), empty_c)
                c_states[step].append(j)
    log(f"[states] arm C: k=1 -> {len(set(c_states[1]))} distinct states over {ctx.n} users; "
        f"k=2 -> {len(set(c_states[2]))} distinct")

    # --- A: Reddit, request-placed (smoke_trio arm 4 recipe, verbatim)
    df = pd.read_csv(RULER)
    N = len(df)
    ops, texts = [], []
    for _i, row in df.iterrows():
        o = str(row.op_named_movieIds) if isinstance(row.op_named_movieIds, str) else ""
        ops.append(np.array([m2s[int(x)] for x in o.split()], np.int64))
        texts.append(row.request_text if isinstance(row.request_text, str) else "")
    anch = np.array([len(o) > 0 for o in ops])
    from sentence_transformers import SentenceTransformer
    sb = SentenceTransformer(SBERT_MODEL)
    sims = sbert_sims(texts, tag_names, sb)
    thread_conc = []
    for r in range(N):
        order = np.argsort(-sims[r])[:SBERT_TOPN]
        cs = [int(c) for c in order if sims[r, c] >= SBERT_MINCOS]
        thread_conc.append(canon(cs, [CONC_VALUE] * len(cs), [0.0] * len(cs)))
    a_states, a_keys = [], []
    for r in range(N):
        j, k = ST.add(ops[r], np.full(len(ops[r]), LIKE_LEVEL, np.int64), thread_conc[r])
        a_states.append(j)
        a_keys.append(k)
    log(f"[states] arm A: {N} threads -> {len(set(a_states))} distinct states "
        f"({int(anch.sum())} anchored, {int((~anch).sum())} zero-anchor); "
        f"mean concepts folded {np.mean([len(c[0]) for c in thread_conc]):.2f}")

    # --- existential control states (hand-built opposites)
    def top_items_for_tag(tagname, k=8):
        tid = [t for t, nm in zip(tags, tag_names) if nm == tagname]
        if not tid:
            tid = [t for t, nm in zip(tags, tag_names) if tagname in nm]
        if not tid:
            raise SystemExit(f"tag {tagname} not in bank")
        mem = members[tid[0]]
        return mem[np.argsort(-ctx.cnt[mem])[:k]]
    syn = {}
    for nm in ("horror", "romance"):
        it = top_items_for_tag(nm)
        syn[nm], _k = ST.add(it, np.full(len(it), LIKE_LEVEL, np.int64), empty_c)

    NS = len(ST)
    log(f"[states] TOTAL distinct states to score = {NS}")

    # ---------------------------------------------------------------- score every distinct state
    seqs = [(ST.item_ids[i], ST.item_lvl[i]) for i in range(NS)]
    Zi = C.tower_only(seqs)                                    # (NS,d) tower-only fold, cached
    Z = C.fold_conc_only(Zi, ST.conc)
    V0 = C.V(Z)
    P = C.probs(Z)                                             # (NS, ni) predictive
    Mmass = np.asarray((Mm.T @ P.T).T)                         # (NS, C) member mass
    prior_mass = np.maximum(Mmass[prior_state], 1e-12)
    log(f"[states] scored; V(prior) = {float(V0[prior_state]):.5f}")

    # answer probabilities
    p_item = P[:, item_bank]                                            # (NS, nI)
    p_seen = 1.0 - np.power(1.0 - np.clip(p_item, 0, 1), Lmed)
    m = np.clip(Mmass, 0, 1)
    p_ans = 1.0 - np.power(1 - m, Lmed) - Lmed * m * np.power(1 - m, Lmed - 1)
    p_ans = np.clip(p_ans, 0.0, 1.0)
    lift = Mmass / prior_mass[None, :]
    p_like = lift / (1.0 + lift)

    # ---------------------------------------------------------------- U(q|z), all states x all banks
    U_item = np.full((NS, nI), -np.inf, np.float64)
    U_conc = np.full((NS, nC), -np.inf, np.float64)
    bench_n = 8 if args.bench else NS
    t_c0 = time.time()
    for s in range(bench_n):
        sids, lvls, cc = ST.item_ids[s], ST.item_lvl[s], ST.conc[s]
        have = set(sids.tolist())
        chave = set(cc[0])
        # ---- concept candidates: tower fold is fixed, only the C-LITE net re-runs
        cand_c = [c for c in range(nC) if c not in chave]
        vs = np.zeros((len(cand_c), 3))
        for br, (val, ref) in enumerate(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0))):
            concs = [canon(cc[0] + [c], cc[1] + [val], cc[2] + [ref]) for c in cand_c]
            zz = C.fold_conc_only(Zi[s].unsqueeze(0).expand(len(cand_c), -1).contiguous(), concs)
            vs[:, br] = C.V(zz).numpy()
        pa = p_ans[s, cand_c]; pl = p_like[s, cand_c]
        U_conc[s, cand_c] = pa * (pl * vs[:, 0] + (1 - pl) * vs[:, 1]) + (1 - pa) * vs[:, 2]
        # ---- item candidates: tower re-fold (+1 token) then the state's own concept fold
        cand_i = [j for j in range(nI) if int(item_bank[j]) not in have]
        vi = np.zeros((len(cand_i), 2))
        for br, lv in enumerate((LIKE_LEVEL, UNSEEN_LEVEL)):
            sq = [(np.append(sids, int(item_bank[j])), np.append(lvls, lv)) for j in cand_i]
            zi = C.tower_only(sq)
            zz = C.fold_conc_only(zi, [cc] * len(cand_i))
            vi[:, br] = C.V(zz).numpy()
        ps = p_seen[s, cand_i]
        U_item[s, cand_i] = ps * vi[:, 0] + (1 - ps) * vi[:, 1]
        if s < 4 or (s + 1) % 100 == 0:
            el = time.time() - t_c0
            log(f"[crit] state {s+1}/{bench_n} | {el/(s+1):.2f}s/state | "
                f"eta {(bench_n-s-1)*el/(s+1)/60:.1f} min")
    if args.bench:
        log(f"[bench] {(time.time()-t_c0)/8:.2f} s/state -> full run "
            f"{NS*(time.time()-t_c0)/8/3600:.2f} h")
        return

    arg_item = U_item.argmax(1)
    arg_conc = U_conc.argmax(1)
    # unified bank: same units (expected DCG@10), so the two banks compete directly
    best_i = U_item.max(1); best_c = U_conc.max(1)
    arg_uni = np.where(best_i >= best_c, arg_item, nI + arg_conc)
    gain_item = best_i - V0.numpy(); gain_conc = best_c - V0.numpy()

    # ---------------------------------------------------------------- assemble the three rows
    def row(idxs, label):
        idxs = np.asarray(idxs)
        return {"label": label,
                "items": dispersion_stats(arg_item[idxs]),
                "concepts": dispersion_stats(arg_conc[idxs]),
                "unified": dispersion_stats(arg_uni[idxs]),
                "mean_gain_item": round(float(gain_item[idxs].mean()), 6),
                "mean_gain_concept": round(float(gain_conc[idxs].mean()), 6),
                "n_distinct_states": int(len(set(idxs.tolist())))}

    A_idx = np.array(a_states)
    res = {
        "A_reddit_request_placed": row(A_idx, "Reddit, request-placed (4,431 threads)"),
        "A_anchored": row(A_idx[anch], "Reddit, >=1 OP_NAMED anchor"),
        "A_zero_anchor": row(A_idx[~anch], "Reddit, zero anchors (SBERT concepts only)"),
        "B_ml25m_cold": row(np.full(ctx.n, prior_state), "ML-25M cold, k=0 (10,000 VAL users)"),
        "C_ml25m_entropy0_k1": row(np.array(c_states[1]), "ML-25M after 1 entropy0 question"),
        "C_ml25m_entropy0_k2": row(np.array(c_states[2]), "ML-25M after 2 entropy0 questions"),
    }

    # ---------------------------------------------------------------- controls
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(A_idx))
    shuf = row(A_idx[perm], "shuffle control")
    same = shuf["items"] == res["A_reddit_request_placed"]["items"] and \
        shuf["concepts"] == res["A_reddit_request_placed"]["concepts"]
    # duplicate-state census
    from collections import Counter
    kc = Counter(a_keys)
    dup_threads = int(sum(v for v in kc.values() if v > 1))
    pair_key = Counter((a_keys[r], int(arg_conc[a_states[r]]), int(arg_item[a_states[r]]))
                       for r in range(N))
    dup_state_and_argmax = int(sum(v for v in pair_key.values() if v > 1))

    ex = {"horror_item": int(item_bank[arg_item[syn["horror"]]]),
          "romance_item": int(item_bank[arg_item[syn["romance"]]]),
          "horror_concept": tag_names[arg_conc[syn["horror"]]],
          "romance_concept": tag_names[arg_conc[syn["romance"]]]}
    ex["items_differ"] = ex["horror_item"] != ex["romance_item"]
    ex["concepts_differ"] = ex["horror_concept"] != ex["romance_concept"]

    # determinism: recompute 100 random threads from scratch
    det_rows = rng.choice(N, size=100, replace=False)
    ok_det = True
    for r in det_rows:
        s = a_states[int(r)]
        sids, lvls, cc = ST.item_ids[s], ST.item_lvl[s], ST.conc[s]
        chave = set(cc[0])
        cand_c = [c for c in range(nC) if c not in chave]
        vs = np.zeros((len(cand_c), 3))
        zi0 = C.tower_only([(sids, lvls)])
        for br, (val, ref) in enumerate(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0))):
            concs = [canon(cc[0] + [c], cc[1] + [val], cc[2] + [ref]) for c in cand_c]
            zz = C.fold_conc_only(zi0.expand(len(cand_c), -1).contiguous(), concs)
            vs[:, br] = C.V(zz).numpy()
        pa = p_ans[s, cand_c]; pl = p_like[s, cand_c]
        u = pa * (pl * vs[:, 0] + (1 - pl) * vs[:, 1]) + (1 - pa) * vs[:, 2]
        if int(cand_c[int(u.argmax())]) != int(arg_conc[s]):
            ok_det = False
            break
    log(f"[control] determinism (100 threads, concept bank): {'PASS' if ok_det else 'FAIL'}")

    controls = {"shuffle_identical": bool(same), "shuffle_stats": shuf,
                "threads_sharing_an_exact_state": dup_threads,
                "threads_sharing_state_AND_argmax": dup_state_and_argmax,
                "existential": ex, "determinism_100_threads": bool(ok_det)}

    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "scripts/reddit/dispersion_leg1.py",
        "tower": str(SNAP.relative_to(ROOT)), "concept_net": str(CFOLD.relative_to(ROOT)),
        "ruler": str(RULER.relative_to(ROOT)), "n_threads": int(N),
        "n_val_users": int(ctx.n),
        "criterion": {
            "name": "expected NDCG@10 after one question, under the tower's own predictive",
            "formula": ("p(.|z)=softmax(z Wd^T+bd); V(z)=sum_{r<=10} disc_r p_(r)(.|z); "
                        "U(q|z)=sum_a P(a|z) V(z (+) (q,a)); crit=U-V(z)"),
            "answer_model": {
                "L_median_val_profile": Lmed,
                "item": "P(seen)=1-(1-p_i)^L; LIKED folds at level 8, UNSEEN folds as the i26 "
                        "exposure sentinel (level 10) -- both move the state",
                "concept": "P(answerable)=1-(1-m)^L-L m (1-m)^(L-1) with m=sum_{i in c} p_i; "
                           "P(like)=lift/(1+lift), lift=m(z)/m(prior); refusal folds a refusal token"},
            "why_not_an_existing_criterion":
                "repo audit: interview evals use static lit orders (interview_strategies.py); greedy "
                "machinery optimises a POPULATION mean over a shared sequence; the only per-state rule "
                "(adaptive_concept_arms 'greedy realizable') is |<d_c,z>|*std|v| = expected response "
                "magnitude, an information criterion, which the M11 rule forbids."},
        "banks": {"n_concepts": nC, "concept_rule": "genome relevance>=0.5, >=30 members",
                  "n_items": nI,
                  "item_rule": f"union of top-{ITEM_BANK_PER_ORDER} of popularity / entropy0 / helf",
                  "item_bank_covers_first16_of_each_order": cover},
        "n_distinct_states_scored": NS,
        "rows": res,
        "controls": controls,
        "runtime_minutes": round((time.time() - t0) / 60, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    log(f"-> {OUT_JSON}")
    write_md(out, tag_names, item_bank, arg_item, arg_conc, a_states, c_states, prior_state)
    log(f"-> {OUT_MD} ({out['runtime_minutes']} min)")


def write_md(o, tag_names, item_bank, arg_item, arg_conc, a_states, c_states, prior_state):
    R = o["rows"]

    def line(key, bank):
        r = R[key][bank]
        return (f"| {R[key]['label']} | {r['n']:,} | {r['n_distinct']} | {r['top_share']:.3f} | "
                f"{r['entropy_bits']:.3f} | {r['perplexity']:.2f} | {r['top10_cum_share']:.3f} |")

    order = ["B_ml25m_cold", "C_ml25m_entropy0_k1", "C_ml25m_entropy0_k2",
             "A_reddit_request_placed", "A_anchored", "A_zero_anchor"]
    hdr = ("| state population | n | distinct q* | top-q* share | entropy (bits) | perplexity "
           "(eff. #questions) | top-10 cum. share |\n|---|---|---|---|---|---|---|")
    ex = o["controls"]["existential"]
    md = f"""# Reddit leg 1 -- does a real request create divergence in the OPTIMAL NEXT QUESTION?

Run {o['generated']} by `{o['script']}` ({o['runtime_minutes']} min, CPU, 4 torch threads).
Tower `{o['tower']}` (frozen, read-only import), concept channel `{o['concept_net']}`.
Ruler `{o['ruler']}` -- **all {o['n_threads']:,} threads**; ML-25M rows use **all {o['n_val_users']:,}
canonical VAL users**. No sampling anywhere (HARD RULE 1). No LLM call, no Reddit API.

## The criterion (M11: dispersion must be read off a LOSS-COUPLED criterion)

{o['criterion']['formula']}

`p(.|z)` is the tower's own multinomial predictive -- the distribution its training loss optimises --
so "relevance" here is the model's belief, never held-out ground truth: the criterion is a function of
the STATE alone, which is what makes measurement B a genuine floor rather than a leak. `V(z)` is the
expected NDCG@10 of the slate the recommender would actually show at `z` (IDCG = 1 under a single-draw
relevance model, so DCG = NDCG). `U(q|z)` marginalises over the answer, so the criterion is
answer-dependent -- it is not a Sigma-only criterion whose dispersion is 0 by the theorem.

**Answer model** (behavioural, state-coupled, ground truth never read), L = {o['criterion']['answer_model']['L_median_val_profile']}
= the median rated-item count of a canonical VAL user:

* item -- {o['criterion']['answer_model']['item']}
* concept -- {o['criterion']['answer_model']['concept']}

**Why this criterion and not one already in the tree.** {o['criterion']['why_not_an_existing_criterion']}
LIMITATION, printed rather than hidden: the target distribution is the model's own posterior, so this
measures the recommender's *self-consistent* expected gain, not gain against held-out items. That is
the price of a criterion that is a pure function of the state; a ground-truth-coupled version would be
clairvoyant and could not be computed at interview time.

## Banks (identical in all three measurements)

* **Concepts** -- {o['banks']['n_concepts']} genome concepts ({o['banks']['concept_rule']}), the full bank.
* **Items** -- {o['banks']['n_items']} items, {o['banks']['item_rule']}. Covers the first 16 of every
  standard order: `{o['banks']['item_bank_covers_first16_of_each_order']}`. This is a bank choice, not a
  per-user candidate cap: the criterion costs one encoder fold plus one full 18,359-way decode per
  (state, candidate, answer), so the full catalogue would be ~1.6e15 flops.

{o['n_distinct_states_scored']:,} distinct states were scored (states are de-duplicated exactly; the
de-duplication itself is part of the finding -- see arm C).

## The comparison

**Concept bank ({o['banks']['n_concepts']} questions)**

{hdr}
{chr(10).join(line(k, 'concepts') for k in order)}

**Item bank ({o['banks']['n_items']} questions)**

{hdr}
{chr(10).join(line(k, 'items') for k in order)}

**Unified bank (items + concepts compete in the same expected-NDCG units)**

{hdr}
{chr(10).join(line(k, 'unified') for k in order)}

Mean expected gain of the chosen question (`U(q*) - V(z)`, NDCG@10 units):

| state population | best concept | best item |
|---|---|---|
{chr(10).join(f"| {R[k]['label']} | {R[k]['mean_gain_concept']:+.5f} | {R[k]['mean_gain_item']:+.5f} |" for k in order)}

## Controls

* **Shuffle** -- each thread reassigned a random other thread's folded state. Every dispersion
  statistic identical: **{o['controls']['shuffle_identical']}** (it must be: the argmax multiset is a
  property of the state set, so a permutation cannot move it; a change here would mean the pipeline
  leaks thread identity into the criterion).
* **Duplicate-state census** -- {o['controls']['threads_sharing_an_exact_state']:,} of
  {o['n_threads']:,} threads share an exact state hash with at least one other thread;
  {o['controls']['threads_sharing_state_AND_argmax']:,} share both an exact state hash and an argmax.
* **Existential** -- two hand-built opposite states (top-8 popular `horror` likes vs top-8 popular
  `romance` likes): best concept `{ex['horror_concept']}` vs `{ex['romance_concept']}`
  (differ: {ex['concepts_differ']}); best item sid {ex['horror_item']} vs {ex['romance_item']}
  (differ: {ex['items_differ']}).
* **Determinism** -- 100 random threads recomputed from scratch return identical argmaxes:
  **{o['controls']['determinism_100_threads']}**.

## Verdict

(written after the run)
"""
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    if OUT_MD.exists():
        old = OUT_MD.read_text(encoding="utf-8")
        cut = old.find("## Verdict")
        if cut >= 0:
            md = md[: md.find("## Verdict")] + old[cut:]
    OUT_MD.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
