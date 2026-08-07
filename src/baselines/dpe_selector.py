r"""dpe_selector.py -- DPE (Parapar & Radlinski, "Diverse User Preference Elicitation with Multi-Armed
Bandits", WSDM 2021, DOI 10.1145/3437963.3441786) as an interview question selector.

FULL TEXT VERIFIED [V]: https://storage.googleapis.com/gweb-research2023-media/pubtools/6086.pdf
Every equation, constant and protocol statement below carries [V] if it is in that text.

================================================================================================
WHAT DPE ACTUALLY IS -- AND THE CORRECTION THIS FILE EXISTS TO RECORD
================================================================================================
Our own literature pass (external_literature/findings/rating_elicitation_family_deep.md sec. 3.8)
files DPE as a "topic bandit" and the battery TODO calls it "our only named concept-channel rival".
That is HALF RIGHT and the half that is wrong matters for how its row is read:

    *** DPE's BANDIT is over topics. DPE's QUESTIONS are ITEMS. ***

Algorithm 1, verbatim [V]: `next_topic <- arg max_a theta_a` ... `forall i in next_topic do
delta_i <- score_div(i, P[u])` ... `next_item <- arg max_i delta_i` ... `r(t) <- reward_f(next_item,
r(u, next_item))` ... `P[u].add(next_item)`. The user is never asked about a topic. The topic is a
LATENT STRUCTURE OVER THE ITEM ACTION SPACE, used to force topical spread among the ITEMS asked.

So DPE is NOT a second `ConLinUCB-BS-static` (which genuinely asks key-terms). Its faithful row
belongs in the ITEM column of the battery, next to Golbandi/DRE/RBMF, with the note that its
selection criterion is topical. We therefore ship TWO arms and label them apart:

  `dpe`          FAITHFUL. Thompson topic bandit -> asks an ITEM from the drawn topic. This is the
                 published method and the only row that may carry the paper's name unqualified.
  `dpe_concept`  DEVIATION, ours, clearly labelled: the SAME Thompson bandit with the genome
                 concepts as arms, where the drawn arm IS the question (a key-term action). This is
                 the concept-channel opponent the battery wanted; it is "DPE's bandit on the concept
                 action space", never "DPE".

================================================================================================
THE ALGORITHM, AS PRINTED
================================================================================================
Per user u, one independent bandit over topics A [V]:

    mu_a(0) = 0 ;  k_a(0) = 0                                                              [V Alg.1]
    each t:  theta_a ~ N( mu_a(t-1), 1 / (k_a(t-1) + 1) )   for all a                      [V Alg.1]
             next_topic = argmax_a theta_a                                                 [V Alg.1]
             next_item  = argmax_{i in next_topic} score_div(i, P[u])                      [V Alg.1]
             r(t) = reward_f(next_item, r(u, next_item))                                   [V Alg.1]
             for every arm a containing next_item:
                 mu_a(t) = ( mu_a(t-1)*k_a(t-1) + r(t) ) / ( k_a(t-1) + 2 )                [V Alg.1]
                 k_a(t)  = k_a(t-1) + 1                                                    [V Alg.1]
             P[u].add(next_item)                                                           [V Alg.1]

*** THE `+2` IS THE PAPER'S, NOT A TYPO OF OURS. *** A running mean would divide by k+1. Dividing by
k+2 shrinks every posterior mean toward 0 by k/(k+1) at each update. It is printed that way in
Algorithm 1 and we implement it as printed; `--dpe_update mean` gives the k+1 form as a labelled
REPAIR, and the two are reported separately if they ever differ materially. Snapping to the printed
form is the whole point of a native snap.

Reward [V]: "a reward r(t) in {0,1} is computed ... In our experiments, we use the normalized rating
value as the reward". Our answers are half-star LEVELS 0..9 (0.5..5.0 stars), so the normalised
rating is exactly `level / 9` in [0,1]. For `dpe_concept` the answer is the signed concept value in
[-1,1] and the normalised reward is `(v+1)/2`, which is the same affine map onto [0,1].

score_div, the within-topic ranker, three strategies [V, Eqs. 1-3]:

  ILD (Eq. 2)        score = (1/|P|) sum_{j in P} ( 1 - cos( r(.,i), r(.,j) ) )
  Hellinger (Eq. 1)  score = (1/|P|) sum_{j in P} (1/sqrt 2) sqrt( sum_v (r(v,i) - r(v,j))^2 )
  Oracle (Eq. 3)     score = r(u,i)   -- the user's ACTUAL rating, "an oracle arm sorting strategy"

Both Eq. 1 and Eq. 2 are computed over the ITEM'S RATING COLUMN across all users -- a
collaborative, not a content, notion of item distance. We use the canonical Liang TRAIN matrix
`D["train"]` for those columns, which is the only rating matrix in this contract that no test user
contributed to.

  *** DPE-ORACLE IS AN ORACLE AND IS STAMPED AS ONE. *** It reads the test user's own fold-in-side
  ratings at SELECTION time, which no deployable selector may do, and under our protocol it also buys
  a 100% answer rate (it can only ever propose items the user has rated) while every honest arm burns
  turns on unseen items. It is an upper bound, never a comparable row, and the harness cell records
  `dpe_oracle: true`.

================================================================================================
THE PROTOCOL GAP -- THE FINDING THAT MATTERS MOST FOR US
================================================================================================
Their evaluation [V sec. 4.1/4.2]: users filtered to >= 100 ratings "given that our evaluation elicits
up to 75 preferences per user"; "the actual preferences are obtained directly from the held-out
ratings from the user"; unasked rated items are "leftovers". For 75 ANSWERED preferences to exist per
user, the ask must land inside the user's own rated set: **DPE's published protocol has a 100% answer
rate by construction and never models an unanswerable question.** [V sec. 4.2.1 concedes the point
directly: offline evaluation "only allows answers over items previously rated by the user".]

Our contract does not grant that. An asked-but-unrated item BURNS the question, exactly as it does for
every other arm here. So DPE's topic bandit receives no reward on a burned turn. Two treatments,
both exposed, neither tuned:

  `--dpe_burn none`  (default) a burn produces NO update at all -- the bandit cannot tell a burn from
                     a turn that never happened. This is the treatment the conversational-bandit
                     family already uses ("no row in M, no row in b") so the two families are
                     comparable.
  `--dpe_burn zero`  a burn is reward 0, i.e. the bandit LEARNS that a topic is hard to answer. This
                     is strictly more informative and is the answerability-aware variant; it is a
                     deviation from the paper (which has no burns) and is labelled as one.

================================================================================================
COST: ONE GRAM ROW PER USER-TURN, AND WHY THAT IS EXACT
================================================================================================
Both diversity scores need item-item quantities over 18,359 rating columns. Naively that is a
|topic| x |P| distance block per user per turn. It is not needed, because both scores are SUMS OVER
THE PROFILE and are therefore incrementally accumulable per user:

    ILD:        A_u(i) = sum_{j in P} G[i,j] / (n_i n_j)      -> score = 1 - A_u(i)/|P|
    Hellinger:  H_u(i) = sum_{j in P} sqrt( n_i^2 + n_j^2 - 2 G[i,j] ) / sqrt 2
                                                              -> score = H_u(i)/|P|

with G = train^T train the item Gram and n_i = ||column i||. Adding ONE item j to the profile needs
exactly ONE row G[j, :]. So the per-user state is one float32 vector of length n_items (73 KB) and the
per-turn work is one memmap row read. Nothing is approximated, sampled or truncated.

G is 18,359^2 float32 = 1.35 GB, built once, chunked, into a contract-signed memmap sidecar
(.cache/baselines/dpe_item_gram_<sig>.dat + .json). Building it is the only expensive step and it is
resumable per chunk (HARD RULE 11a/b).

================================================================================================
FREE CONSTANTS -- all of them, NONE tuned on test
================================================================================================
  topics        "genres"  the paper's own granularity: 20 MovieLens genres, "an item could belong to
                          more than one topic" [V]. `--dpe_topics genome` swaps in the 1,031-concept
                          genome bank (the bank every other concept row in this project uses).
  score_div     "ild"     the paper reports ILD as its best accuracy/serendipity arm [V sec. 5.2].
  empty profile  popularity  score_div is undefined at |P|=0 (both Eqs. 1-2 divide by |P|). The paper
                          itself proposes popularity as the cold-start arm ranker [V sec. 3.2], so the
                          first pick inside a topic is its most-rated train item. Stated, not hidden.
  seed          98765     the project seed; the Thompson draw is seeded per (seed, user, turn) so a
                          resumed interview replays bit-identically (adaptive_harness's resume rule).
  mu_a(0)=0, k_a(0)=0     [V Alg.1] -- "The arms are modeled by Normal Distributions whose parameters
                          we initialize equally"; the paper notes k_a(0) is the exploration knob and
                          says it used "only default initialization" [V sec. 5.3].

  python src/baselines/dpe_selector.py --smoke      # synthetic code-path check, not a number
"""
import os
import sys
import json
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))

from adaptive_harness import Selector                                          # noqa: E402

CACHE_DIR = os.path.join(_ROOT, ".cache", "baselines")
SCORE_DIVS = ("ild", "hellinger", "oracle")
TOPIC_SETS = ("genres", "genome")
NEG = -1e18


# ===================================================================== the item Gram (memmap)
def _gram_paths(sig):
    return (os.path.join(CACHE_DIR, f"dpe_item_gram_{sig}.dat"),
            os.path.join(CACHE_DIR, f"dpe_item_gram_{sig}.json"))


def load_or_build_item_gram(train, n_items, sig="ml25m_liang", chunk=512, log=print):
    """G = train^T train as a float32 memmap, built once and resumable per chunk (HARD RULE 11a/b).

    `train` is the canonical Liang TRAIN interaction matrix (140,768 x 18,359). G[i,j] is the number
    of train users who interacted with both i and j -- the inner product Eqs. 1-2 are written over.
    Returns (G_memmap, n) with n_i = sqrt(G[i,i]) = ||column i||.

    The sidecar carries the shape and nnz of the matrix it was built from and is REFUSED on any
    mismatch, so a Gram from a different split can never be silently reused."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    dat, meta = _gram_paths(sig)
    contract = {"shape": list(map(int, train.shape)), "nnz": int(train.nnz),
                "n_items": int(n_items), "dtype": "float32", "chunk": int(chunk)}
    done = 0
    if os.path.exists(meta):
        try:
            blob = json.load(open(meta))
        except Exception:
            blob = {}
        if blob.get("contract") == contract and os.path.exists(dat):
            done = int(blob.get("chunks_done", 0))
            if done * chunk >= n_items:
                G = np.memmap(dat, dtype=np.float32, mode="r", shape=(n_items, n_items))
                nrm = np.sqrt(np.maximum(np.asarray(G[np.arange(n_items), np.arange(n_items)],
                                                    dtype=np.float64), 0.0))
                log(f"[dpe] item Gram from cache ({os.path.basename(dat)}, "
                    f"{n_items}x{n_items} float32 = {n_items * n_items * 4 / 2**30:.2f} GiB)")
                return G, nrm
            log(f"[dpe] item Gram PARTIAL: {done} chunk(s) banked -- resuming")
        elif blob:
            log("[dpe] item Gram cache REFUSED: contract mismatch -- rebuilding\n"
                f"    cached={blob.get('contract')}\n    wanted={contract}")
            done = 0
    Tc = train.tocsc()
    mode = "r+" if (done and os.path.exists(dat)) else "w+"
    G = np.memmap(dat, dtype=np.float32, mode=mode, shape=(n_items, n_items))
    t0 = time.time()
    for c in range(done, (n_items + chunk - 1) // chunk):
        s, e = c * chunk, min((c + 1) * chunk, n_items)
        blk = (Tc[:, s:e].T @ train)                      # (chunk x n_items) sparse
        G[s:e] = np.asarray(blk.todense(), dtype=np.float32)
        G.flush()
        json.dump({"contract": contract, "chunks_done": c + 1}, open(meta, "w"))
        if (c + 1) % 8 == 0 or e >= n_items:
            log(f"[dpe] item Gram {e}/{n_items} rows ({(time.time() - t0) / 60:.1f}m)")
    del G
    G = np.memmap(dat, dtype=np.float32, mode="r", shape=(n_items, n_items))
    nrm = np.sqrt(np.maximum(np.asarray(G[np.arange(n_items), np.arange(n_items)],
                                        dtype=np.float64), 0.0))
    return G, nrm


# ===================================================================== topic sets
def build_genre_topics(n_items, log=print):
    """The paper's own topic granularity: MovieLens GENRES, an item may belong to several [V].

    movieId -> sid is exactly the line order of proc/unique_sid.txt (the same mapping
    conversational_bandit_selectors.build_concept_stack uses), so the two topic sets index the same
    catalogue. '(no genres listed)' is dropped: it is not a topic, it is a missing label."""
    import pandas as pd
    usid = [int(x) for x in open(os.path.join(_ROOT, "data", "ml-25m", "proc",
                                              "unique_sid.txt")).read().split()]
    assert len(usid) == n_items, f"unique_sid.txt has {len(usid)} rows but n_items={n_items}"
    m2s = {m: i for i, m in enumerate(usid)}
    mv = pd.read_csv(os.path.join(_ROOT, "data", "movielens", "movies.csv"))
    mv = mv[mv["movieId"].isin(m2s)]
    buckets = {}
    for mid, gs in zip(mv["movieId"].values, mv["genres"].values):
        sid = m2s[int(mid)]
        for g in str(gs).split("|"):
            if g and g != "(no genres listed)":
                buckets.setdefault(g, []).append(sid)
    names = sorted(buckets)
    member_sets = {a: np.sort(np.asarray(buckets[n], dtype=np.int64)) for a, n in enumerate(names)}
    cov = len({i for m in member_sets.values() for i in m})
    log(f"[dpe] genre topics: {len(names)} arms, {cov}/{n_items} items covered, "
        f"{sum(len(m) for m in member_sets.values()) / max(cov, 1):.2f} topics/item")
    return names, member_sets


# ===================================================================== the selector
class DPESelector(Selector):
    """DPE, faithful: a per-user Thompson topic bandit whose QUESTIONS ARE ITEMS.

    PURE FUNCTION OF THE HISTORY. `_bandit` recomputes (mu, k) from (asked, answered) every turn
    rather than carrying incremental state, and the Thompson draw is seeded on (seed, user, turn). A
    resumed interview therefore replays bit-identically, which is adaptive_harness's stated
    requirement for every selector in this harness. The only carried state is the diversity
    accumulator, which is rebuilt from `asked[u]` whenever its length disagrees -- so a resume repairs
    it automatically instead of silently continuing from a stale vector."""
    name = "dpe"

    def __init__(self, member_sets, topic_names, score_div="ild", seed=98765, burn="none",
                 update="paper", oracle_lvl=None, gram=None, gnorm=None, pop=None, log=print):
        assert score_div in SCORE_DIVS, f"unknown score_div '{score_div}'; known {SCORE_DIVS}"
        assert burn in ("none", "zero") and update in ("paper", "mean")
        self.member_sets = member_sets
        self.topic_names = list(topic_names)
        self.A = len(member_sets)
        self.score_div = score_div
        self.seed = int(seed)
        self.burn = burn
        self.update = update
        self.oracle_lvl = oracle_lvl            # ORACLE ONLY: per-user {item: level} (see header)
        self.G, self.gn, self.pop = gram, gnorm, pop
        self.log = log
        # item -> the topics it belongs to (the `forall a | next_item in a` update loop)
        self._topics_of = None
        self._acc = None                        # per-user diversity accumulator (float32, n_items)
        self._acc_len = None
        self.n_items = None
        self.name = f"dpe_{score_div}" if score_div != "ild" else "dpe"

    # ------------------------------------------------------------------ setup
    def reset(self, ctx):
        super().reset(ctx)
        ni = int(ctx["n_items"])
        self.n_items = ni
        if self._topics_of is None:
            tof = [[] for _ in range(ni)]
            for a, mem in self.member_sets.items():
                for i in mem:
                    tof[int(i)].append(int(a))
            self._topics_of = [np.asarray(t, dtype=np.int64) for t in tof]
        n = int(ctx["n_users"])
        self._acc = None if self.score_div == "oracle" else \
            np.zeros((n, ni), dtype=np.float32)
        self._acc_len = [0] * n
        ctx["log"](f"[{self.name}] topics={self.A} score_div={self.score_div} burn={self.burn} "
                   f"update={self.update} seed={self.seed}"
                   + ("  *** ORACLE ARM: reads the user's own ratings at selection time; upper "
                      "bound only ***" if self.score_div == "oracle" else ""))

    # ------------------------------------------------------------------ the bandit
    def _reward(self, item, ans_map):
        """reward_f: the normalised rating [V]. level 0..9 (0.5..5.0 stars) -> level/9 in [0,1].
        Returns None when the question was BURNED and `burn == "none"` (no update at all)."""
        if item in ans_map:
            return float(ans_map[item]) / 9.0
        return 0.0 if self.burn == "zero" else None

    def _bandit(self, asked_u, answered_u):
        """(mu, k) after replaying Algorithm 1's update over this user's realised history."""
        mu = np.zeros(self.A, dtype=np.float64)
        kk = np.zeros(self.A, dtype=np.float64)
        if not asked_u:
            return mu, kk
        ans = dict(answered_u)
        for it in asked_u:
            if it >= self.n_items:              # a concept action never reaches this arm
                continue
            r = self._reward(int(it), ans)
            if r is None:
                continue
            arms = self._topics_of[int(it)]
            if not len(arms):
                continue
            den = kk[arms] + (2.0 if self.update == "paper" else 1.0)
            mu[arms] = (mu[arms] * kk[arms] + r) / den
            kk[arms] += 1.0
        return mu, kk

    def _draw_topic(self, u, turn, mu, kk):
        """theta_a ~ N(mu_a, 1/(k_a+1)); next_topic = argmax_a theta_a [V Alg.1].

        A topic whose every member item has already been asked cannot arise at our budgets (the
        smallest genre has hundreds of items and k <= 16), but it is not assumed away: `select`
        falls back to the next-best topic if the drawn one has no unasked member."""
        rng = np.random.default_rng([self.seed, int(u), int(turn)])
        return int(np.argmax(mu + rng.standard_normal(self.A) / np.sqrt(kk + 1.0)))

    # ------------------------------------------------------------------ score_div
    def _sync_acc(self, u, asked_u):
        """Bring user u's diversity accumulator up to date with P[u] = the asked items [V: the paper
        adds EVERY presented item to P[u], answered or not]. One Gram row per newly added item."""
        have = self._acc_len[u]
        items = [int(i) for i in asked_u if int(i) < self.n_items]
        if have == len(items):
            return len(items)
        if have > len(items):                    # cannot happen in-run; a defensive rebuild
            self._acc[u] = 0.0
            have = 0
        a = self._acc[u]
        for j in items[have:]:
            row = np.asarray(self.G[j], dtype=np.float32)
            if self.score_div == "ild":
                nj = self.gn[j]
                a += (row / np.float32(max(nj, 1e-9)))
            else:                                # hellinger
                nj2 = np.float32(self.gn[j] ** 2)
                a += np.sqrt(np.maximum(self.gn2 + nj2 - 2.0 * row, 0.0),
                             dtype=np.float32) * np.float32(1.0 / np.sqrt(2.0))
        self._acc_len[u] = len(items)
        return len(items)

    def _scores(self, u, asked_u, cand):
        """score_div over the candidate items of the drawn topic [V Eqs. 1-3]."""
        if self.score_div == "oracle":
            d = self.oracle_lvl[u]
            return np.asarray([float(d.get(int(i), -1.0)) for i in cand], dtype=np.float64)
        p = self._sync_acc(u, asked_u)
        if p == 0:                               # |P| = 0: Eqs. 1-2 are undefined -> popularity
            return self.pop[cand].astype(np.float64)
        if self.score_div == "ild":
            # 1 - (1/|P|) sum_j cos(i,j) ; cos(i,j) = G[i,j]/(n_i n_j) and the accumulator already
            # carries sum_j G[i,j]/n_j, so only the 1/n_i factor is left.
            return 1.0 - (self._acc[u][cand].astype(np.float64)
                          / np.maximum(self.gn[cand], 1e-9)) / float(p)
        return self._acc[u][cand].astype(np.float64) / float(p)

    # ------------------------------------------------------------------ selection
    def select(self, turn, users, asked, answered):
        if self.score_div == "hellinger" and not hasattr(self, "gn2"):
            self.gn2 = (self.gn ** 2).astype(np.float32)
        out = np.empty(len(users), dtype=np.int64)
        for j, u in enumerate(users.tolist()):
            u = int(u)
            au = [int(i) for i in asked[u] if int(i) < self.n_items]
            mu, kk = self._bandit(asked[u], answered[u])
            a = self._draw_topic(u, turn, mu, kk)
            cand = self.member_sets[a]
            if au:
                cand = cand[~np.isin(cand, au)]
            if not len(cand):                    # every member asked: fall back to the next topic
                for a2 in np.argsort(-mu):
                    c2 = self.member_sets[int(a2)]
                    c2 = c2[~np.isin(c2, au)] if au else c2
                    if len(c2):
                        a, cand = int(a2), c2
                        break
            s = self._scores(u, asked[u], cand)
            out[j] = int(cand[int(np.argmax(s))])
        return out

    def stats(self):
        return {"dpe_topics": self.A, "dpe_score_div": self.score_div, "dpe_burn": self.burn,
                "dpe_update": self.update, "dpe_oracle": self.score_div == "oracle",
                "dpe_topic_names": self.topic_names if self.A <= 32 else f"{self.A} topics"}


class DPEConceptSelector(Selector):
    r"""DEVIATION, ours, labelled: DPE's Thompson bandit with the GENOME CONCEPTS as arms, where the
    drawn arm IS the question. Never call this "DPE" -- see the module header.

    The published algorithm has two levels (topic -> item) and this collapses the second: a concept
    arm's "item" is the concept itself, so `score_div` has nothing to rank and drops out. What
    survives is exactly the part of DPE that is about the CONCEPT channel: a per-user Gaussian
    Thompson bandit that spends its budget spreading across topics, with the topic's own answer as
    the reward.

        theta_c ~ N(mu_c, 1/(k_c+1)) ;  ask argmax_c theta_c ;  r = (v_c + 1)/2 in [0,1]
        mu_c <- (mu_c k_c + r)/(k_c + 2) ;  k_c <- k_c + 1                             [V Alg.1 form]

    A concept is asked at most once (the harness's no-re-ask assert), so each arm is pulled once at
    most and the posterior update only ever affects arms that will never be drawn again. That is a
    structural property of a one-pull-per-arm bandit at k <= 16 over ~1,031 arms and it is the honest
    reading of this row: with mu_c(0) identical for all c, the realised order is a SEEDED RANDOM
    permutation of the concept bank until an arm repeats, which it never does. We report it as such;
    it is the paper's own initialisation ("arms initialised equally" [V]) taken to our budget, and it
    is a genuinely different -- and genuinely published -- criterion from popularity / entropy /
    polar / barycentric-spanner.
    """
    name = "dpe_concept"

    def __init__(self, n_concepts, seed=98765, burn="none", update="paper", log=print):
        self.C = int(n_concepts)
        self.seed = int(seed)
        self.burn = burn
        self.update = update
        self.log = log
        self.n_items = None

    def reset(self, ctx):
        super().reset(ctx)
        self.n_items = int(ctx["n_items"])
        self.conc = ctx["conc_state"]
        ctx["log"](f"[{self.name}] DEVIATION ARM: DPE's Thompson bandit over {self.C} genome "
                   f"concepts, the arm IS the question (the published method asks ITEMS). "
                   f"burn={self.burn} update={self.update} seed={self.seed}")

    def _bandit(self, u):
        mu = np.zeros(self.C, dtype=np.float64)
        kk = np.zeros(self.C, dtype=np.float64)
        ans = dict(self.conc["answered"][u])
        for c in self.conc["asked"][u]:
            c = int(c)
            if c in ans:
                r = (float(ans[c]) + 1.0) / 2.0
            elif self.burn == "zero":
                r = 0.0
            else:
                continue
            mu[c] = (mu[c] * kk[c] + r) / (kk[c] + (2.0 if self.update == "paper" else 1.0))
            kk[c] += 1.0
        return mu, kk

    def select(self, turn, users, asked, answered):
        out = np.empty(len(users), dtype=np.int64)
        for j, u in enumerate(users.tolist()):
            u = int(u)
            mu, kk = self._bandit(u)
            rng = np.random.default_rng([self.seed, u, int(turn)])
            theta = mu + rng.standard_normal(self.C) / np.sqrt(kk + 1.0)
            done = np.asarray([int(c) for c in self.conc["asked"][u]], dtype=np.int64)
            if len(done):
                theta[done] = NEG
            out[j] = self.n_items + int(np.argmax(theta))
        return out

    def stats(self):
        return {"dpe_topics": self.C, "dpe_score_div": "n/a (arm is the question)",
                "dpe_burn": self.burn, "dpe_update": self.update, "dpe_concept_deviation": True}


# ===================================================================== construction helper
def build_dpe(D, args, cache, k, log=print):
    """Build the requested DPE arm, sharing the Gram / topic set across budgets via `cache`."""
    ni = int(D["n_items"])
    topics = getattr(args, "dpe_topics", "genres")
    assert topics in TOPIC_SETS, f"unknown --dpe_topics '{topics}'; known {TOPIC_SETS}"
    if topics == "genres":
        ck = "dpe_genres"
        if ck not in cache:
            cache[ck] = build_genre_topics(ni, log=log)
        names, member_sets = cache[ck]
    else:
        import conversational_bandit_selectors as CB
        ck = "dpe_genome"
        if ck not in cache:
            stk = CB.build_concept_stack(D, None, Y=None, log=log)
            cache[ck] = ([str(t) for t in stk["tags"]],
                         {a: np.asarray(m, dtype=np.int64)
                          for a, m in stk["member_sets"].items()})
        names, member_sets = cache[ck]
    sd = getattr(args, "dpe_score_div", "ild")
    gram = gnorm = pop = None
    if sd != "oracle":
        if "dpe_gram" not in cache:
            cache["dpe_gram"] = load_or_build_item_gram(D["train"], ni, log=log)
        gram, gnorm = cache["dpe_gram"]
        pop = np.asarray(D["train"].sum(axis=0)).ravel().astype(np.float64)
    oracle_lvl = None
    if sd == "oracle":
        import interview_strategies as ST
        oracle_lvl = ST.level_lookup(D["g_te_tr"])
    sel = DPESelector(member_sets, names, score_div=sd, seed=getattr(args, "ts_seed", 98765),
                      burn=getattr(args, "dpe_burn", "none"),
                      update=getattr(args, "dpe_update", "paper"),
                      oracle_lvl=oracle_lvl, gram=gram, gnorm=gnorm, pop=pop, log=log)
    sel.name = f"dpe_{sd}" if sd != "ild" else "dpe"
    return sel


# ===================================================================== smoke
def _smoke():
    """SYNTHETIC code-path check. Not the canonical data, not a number, not citable."""
    from scipy import sparse
    print("[dpe][SMOKE] synthetic catalogue + synthetic topics (code path only)")
    rng = np.random.RandomState(0)
    ni, A, n_users, nu_tr = 300, 8, 15, 400
    Xtr = sparse.csr_matrix((rng.rand(nu_tr, ni) < 0.08).astype(np.float64))
    G = np.asarray((Xtr.T @ Xtr).todense(), dtype=np.float32)
    gn = np.sqrt(np.maximum(np.diag(G).astype(np.float64), 0.0))
    pop = np.asarray(Xtr.sum(axis=0)).ravel().astype(np.float64)
    member_sets = {a: np.sort(rng.choice(ni, size=rng.randint(40, 90), replace=False))
                   for a in range(A)}
    lvl = [{int(i): int(rng.randint(0, 10)) for i in rng.choice(ni, 60, replace=False)}
           for _ in range(n_users)]

    # --- the printed update rule, checked against a hand computation -------------------------
    s = DPESelector(member_sets, [f"t{a}" for a in range(A)], gram=G, gnorm=gn, pop=pop)
    s.reset({"n_users": n_users, "n_items": ni, "q_max": 6, "log": print, "conc_state": None})
    it = int(member_sets[0][0])
    arms = s._topics_of[it]
    mu, kk = s._bandit([it], [(it, 9)])
    assert np.allclose(mu[arms], 1.0 / 2.0) and np.allclose(kk[arms], 1.0), (mu[arms], kk[arms])
    mu, kk = s._bandit([it, it], [(it, 9), (it, 9)])
    assert np.allclose(mu[arms], (0.5 * 1 + 1.0) / 3.0), mu[arms]
    print("[SMOKE] Algorithm 1 update (mu*k + r)/(k+2) PASS (hand-checked at t=1,2)")
    mu2, _ = DPESelector(member_sets, [f"t{a}" for a in range(A)], update="mean",
                         gram=G, gnorm=gn, pop=pop)._bandit([], [])
    assert mu2.sum() == 0.0
    # --- burn treatment ----------------------------------------------------------------------
    mu, kk = s._bandit([it], [])
    assert kk.sum() == 0.0, "burn='none' must produce NO update"
    s2 = DPESelector(member_sets, [f"t{a}" for a in range(A)], burn="zero", gram=G, gnorm=gn, pop=pop)
    s2.reset({"n_users": n_users, "n_items": ni, "q_max": 6, "log": lambda *_: None,
              "conc_state": None})
    mu, kk = s2._bandit([it], [])
    assert np.allclose(kk[arms], 1.0) and np.allclose(mu[arms], 0.0), "burn='zero' must update at r=0"
    print("[SMOKE] burn treatments PASS (none = no update, zero = update at r=0)")

    # --- the incremental accumulator == the naive double loop, exactly -----------------------
    for sd in ("ild", "hellinger"):
        sel = DPESelector(member_sets, [f"t{a}" for a in range(A)], score_div=sd,
                          gram=G, gnorm=gn, pop=pop)
        sel.reset({"n_users": 3, "n_items": ni, "q_max": 6, "log": lambda *_: None,
                   "conc_state": None})
        sel.gn2 = (gn ** 2).astype(np.float32)
        P = [int(x) for x in rng.choice(ni, 5, replace=False)]
        cand = np.arange(ni)
        got = sel._scores(0, P, cand)
        if sd == "ild":
            want = np.asarray([1.0 - np.mean([G[i, j] / max(gn[i] * gn[j], 1e-9) for j in P])
                               for i in cand])
        else:
            want = np.asarray([np.mean([np.sqrt(max(gn[i] ** 2 + gn[j] ** 2 - 2 * G[i, j], 0.0))
                                        / np.sqrt(2.0) for j in P]) for i in cand])
        err = float(np.abs(got - want).max())
        assert err < 2e-3, f"{sd} accumulator != naive double loop (max abs {err:.2e})"
        print(f"[SMOKE] score_div={sd:9s} incremental accumulator == naive Eq. definition "
              f"(max abs diff {err:.2e})")

    # --- end-to-end through the Selector contract --------------------------------------------
    for sd in SCORE_DIVS:
        sel = DPESelector(member_sets, [f"t{a}" for a in range(A)], score_div=sd,
                          gram=G, gnorm=gn, pop=pop, oracle_lvl=lvl)
        sel.reset({"n_users": n_users, "n_items": ni, "q_max": 6, "log": lambda *_: None,
                   "conc_state": None})
        asked = [[] for _ in range(n_users)]
        answered = [[] for _ in range(n_users)]
        for t in range(6):
            q = sel.select(t, np.arange(n_users), asked, answered)
            for u, a in zip(range(n_users), q.tolist()):
                assert 0 <= a < ni and a not in asked[u], f"{sd}: bad action {a}"
                asked[u].append(int(a))
                if a in lvl[u]:
                    answered[u].append((int(a), lvl[u][a]))
        ar = np.mean([len(a) for a in answered])
        assert len({tuple(a) for a in asked}) > 1 or sd == "oracle", \
            f"{sd}: every user got the same interview -- the bandit is not per-user"
        print(f"[SMOKE] {sel.name:14s} PASS  answered {ar:.2f}/6  "
              f"distinct prefixes {len({tuple(a) for a in asked})}/{n_users}")

    # --- resume-exactness: a fresh object replaying the history asks the same question --------
    sel = DPESelector(member_sets, [f"t{a}" for a in range(A)], gram=G, gnorm=gn, pop=pop)
    ctx = {"n_users": n_users, "n_items": ni, "q_max": 6, "log": lambda *_: None,
           "conc_state": None}
    sel.reset(ctx)
    asked = [[] for _ in range(n_users)]
    answered = [[] for _ in range(n_users)]
    for t in range(4):
        q = sel.select(t, np.arange(n_users), asked, answered)
        for u, a in zip(range(n_users), q.tolist()):
            asked[u].append(int(a))
            if a in lvl[u]:
                answered[u].append((int(a), lvl[u][a]))
    cont = sel.select(4, np.arange(n_users), asked, answered)
    fresh = DPESelector(member_sets, [f"t{a}" for a in range(A)], gram=G, gnorm=gn, pop=pop)
    fresh.reset(ctx)
    again = fresh.select(4, np.arange(n_users), asked, answered)
    assert np.array_equal(cont, again), "DPE is not resume-exact: a fresh object diverged"
    print("[SMOKE] resume-exactness PASS (fresh object replaying the history asks identically)")

    # --- the concept deviation arm ------------------------------------------------------------
    C = 40
    cs = {"asked": [[] for _ in range(n_users)], "answered": [[] for _ in range(n_users)],
          "tokens": [[] for _ in range(n_users)]}
    dc = DPEConceptSelector(C)
    dc.reset({"n_users": n_users, "n_items": ni, "q_max": 6, "log": lambda *_: None,
              "conc_state": cs})
    acts = [[] for _ in range(n_users)]
    for t in range(6):
        q = dc.select(t, np.arange(n_users), acts, [[] for _ in range(n_users)])
        for u, a in zip(range(n_users), q.tolist()):
            assert ni <= a < ni + C and a not in acts[u]
            acts[u].append(int(a))
            c = a - ni
            cs["asked"][u].append(c)
            if rng.rand() > 0.4:
                cs["answered"][u].append((c, float(rng.rand() * 2 - 1)))
    assert all(len(set(a)) == 6 for a in acts)
    print(f"[SMOKE] dpe_concept    PASS  6 key-term asks/user, all distinct, "
          f"{len({tuple(a) for a in acts})}/{n_users} distinct prefixes")
    print("[dpe][SMOKE] COMPLETE")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        _smoke()
    else:
        print(__doc__)
