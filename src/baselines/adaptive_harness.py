r"""adaptive_harness.py -- the SEQUENTIAL interview loop: a selector sees the answers so far and
chooses the next question, per user, per turn.

WHY THIS FILE EXISTS (Paper B, Act 1; harness gap named in PAPER_B_DESIGN addendum 3). Until now every
arm in `run_interview_table.py` was a STATIC GLOBAL ORDER: `interview_strategies.ask()` walks a fixed
item list and the only per-user variation is which of those items the user happens to have rated.
`golbandi_leaf` is a CONSUMER of answers, not a selector. So the harness could not express the object
Act 1 has to measure -- "does conditioning the NEXT question on the PREVIOUS answers buy anything over
the strongest static order?" -- because it had no way to ask a different second question of two users
who answered the first one differently.

WHAT IS AND IS NOT NEW HERE. Only the ASK side is new. The answer rule, the answer->input contracts,
the Most-Popular floor, the fixed arm-N candidate pool and the evaluate() cursor discipline are all
IMPORTED from run_interview_table (`score_arm`, `build_recommenders`, `mostpop_row`), not re-written.
That was a deliberate call: the regression assert below can only prove the ask side agrees, so the
score side must be literally the same code or the two paths will drift where no test is looking.

THE REGRESSION CONTRACT (requirement (a)). A static arm driven through the sequential loop must
produce a BYTE-IDENTICAL (asked, answered) to `interview_strategies.ask()`:

    python src/baselines/adaptive_harness.py --regression

asserts exact list equality for all six arms x k in {1,2,4,8,16}, and prints the entropy0/k=8 cell
by name because that is the cell the overseer specified. If it fails, no adaptive number produced by
this file means anything, so it is an assert and not a warning.

THE ANSWER RULE (requirement (b)) is `interview_strategies.level_lookup(g_te_tr)`: the test user's own
half-star level for the asked item, from their fold-in-side ratings with the held-out targets already
removed. An asked-but-unrated item BURNS the question and yields no answer -- the literature's
protocol, and identical to the static path. No simulator, no imputation, no LLM. A selector NEVER sees
`lvl`; it sees only what the user actually answered.

SELECTOR INTERFACE (requirement, and the API Paper B ships):

    class MySelector(Selector):
        name = "my_selector"
        def reset(self, ctx):                 # ctx: n_users, n_items, q_max, D, glob, bank, log
            ...                               # called once per (selector, budget) run
        def select(self, turn, users, asked, answered) -> np.ndarray[int64]  (len == len(users))
            # asked[u]    : list of item ids already asked of user u, in order
            # answered[u] : list of (item_id, half_star_level) the user could answer, in order
            # MUST return an item NOT already in asked[u]; the harness asserts this.

`PerUserSelector` gives a scalar `select_one(turn, u, asked_u, answered_u)` for selectors that cannot
vectorise. Selectors may be stateful per user (keep your own dict keyed on u); the harness passes the
full histories every turn anyway, so statelessness is the easier and safer default.

BATCHING WITHOUT SUBSAMPLING (requirement (c)). The loop hands the selector ALL users of the cohort at
once by default (`--sel_batch 0`); users are never sampled, capped or dropped. Batching is the
selector's business: `bandit_selectors` and `eddi_selector` group users by identical interview STATE
and compute each state's acquisition once, which is exact (the posterior is a function of the state)
and is what makes a 10k-user x 18,359-candidate exact sweep affordable. `--smoke_users N` exists ONLY
for the code-path smoke, refuses to write a citable filename, and stamps SMOKE_NOT_CITABLE into the
JSON.

  python src/baselines/adaptive_harness.py --regression
  python src/baselines/adaptive_harness.py --selectors entropy0-static,glm_ucb --budgets 1,2 \
      --only ours --arch i26 --snapshot .cache/instrument/t2i26c_ep31.pt --no_floor --smoke_users 200
"""
import os
import sys
import json
import time
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))

import interview_strategies as ST
from arm_n import load_arm_n
from arm_n_tower import SNAP_DEFAULT
from golbandi_leaf import MAX_DEPTH
from run_interview_table import (OUT, RECS, ALL_RECS, build_recommenders, score_arm, mostpop_row,
                                 logln, answers_to_csr)

STATIC_SUFFIX = "-static"
# The conversational-bandit (key-term) family: the only selectors with a TWO-TYPE action space.
CONV_BANDIT_SPECS = ("conlinucb_mcr", "conlinucb_bs", "conlinucb_bs_static", "clime", "conucb")
# DPE (Parapar & Radlinski WSDM'21). `dpe*` asks ITEMS through a topic bandit -- that is the PUBLISHED
# method; `dpe_concept` is OUR labelled deviation that puts the same bandit on the concept action
# space. See dpe_selector.py's header for why those are two different rows.
DPE_SPECS = ("dpe", "dpe_hellinger", "dpe_oracle", "dpe_concept")
# IGCN (Rashid, Karypis & Riedl, SIGKDD Expl. 2008): information gain over USER-CLUSTER labels,
# recomputed from the current 150-nearest-neighbour pool as answers arrive. See igcn_selector.py.
IGCN_SPECS = ("igcn", "igcn_i1", "igcn_page15")
PAGE_RECOMPUTE = 15                         # the paper's page size n, as a re-rank period
# Every spec that can spend a turn on a KEY-TERM, and therefore needs the concept action space wired
# up and the concept consumer attached.
CONCEPT_ASK_SPECS = CONV_BANDIT_SPECS + ("dpe_concept",)

# --------------------------------------------------------------------- stored greedy orders
# The three prefix-optimal greedy sequences from `experiments/battery/greedy_static_i26c.json`,
# registered as `fixedorder:` names so they can be walked inside THIS harness and differenced per
# user against the arms in the same run. Before 2026-08-10 only `greedy_items` was wired, as a
# hardcoded branch, and `fixedorder:greedy_concepts` therefore died on the line-561 assert -- at the
# LAST step of a 4-hour run, because the name was only looked up when the cell was reached.
#
# TWO PROPERTIES THIS REGISTRY MUST HAVE, both of them scars:
#  (1) NAMES ARE VALIDATED AT STARTUP (`validate_specs`, called from main before any work), not at
#      first use. A stored-order typo must cost seconds, not hours.
#  (2) A STORED ORDER CARRIES NO ANSWER CONVENTION OF ITS OWN. Only the channel/id pairs are read
#      off the JSON; its `use_refusal` / `concept_ckpt` / `contract` provenance fields are
#      deliberately IGNORED. The sequence is walked under whatever the CELL declares
#      (`--concept_neg_cap`, `--keyterm_repr`, `--snapshot`), exactly like every other selector, so
#      one stored order can be measured under two conventions and the difference is attributable.
#      (To reproduce this file's own published curve, the cell must set `--concept_neg_cap 0`, which
#      is the CERTIFIED-ERA convention it was built under -- but that is the CELL's job to declare.)
GREEDY_ORDER_FILE = os.path.join(_ROOT, "experiments", "battery", "greedy_static_i26c.json")
GREEDY_FIXED_ORDERS = {"greedy_items": "items-only",
                       "greedy_concepts": "concepts-only",
                       "greedy_combined": "combined"}


def load_greedy_order(which, n_items, path=GREEDY_ORDER_FILE):
    """`fixedorder:<which>` -> (action-id list, asks_concepts).

    The JSON stores channel-tagged pairs [channel, id] with channel 0 = ITEM and channel 1 = CONCEPT,
    and the concept ids are RAW concept indices (0..C-1), not offset. The harness action space is
        [0, n_items)            -> ask that item
        [n_items, n_items + C)  -> ask key-term (id - n_items)
    so channel-1 steps are offset here, at load time, which is the only place that mapping belongs.
    Nothing is re-selected and nothing is re-ordered: the sequence is read verbatim.
    """
    arm = GREEDY_FIXED_ORDERS[which]
    gj = json.load(open(path))
    seq = gj["arms"][arm]["sequence"]
    order = [int(i) if int(ch) == 0 else n_items + int(i) for ch, i in seq]
    asks_conc = any(int(ch) == 1 for ch, _i in seq)
    return order, asks_conc


def asks_concepts(spec):
    """Does this spec need the concept action space and the concept consumer wired up?"""
    if spec in CONCEPT_ASK_SPECS:
        return True
    if spec.startswith("fixedorder:"):
        which = spec.split(":", 1)[1]
        return which in GREEDY_FIXED_ORDERS and GREEDY_FIXED_ORDERS[which] != "items-only"
    return False


# ===================================================================== selector interface
class Selector:
    """Base class. Subclass and implement `select`; see the module header for the contract."""
    name = "selector"

    def reset(self, ctx):
        self.ctx = ctx

    def select(self, turn, users, asked, answered):
        raise NotImplementedError

    def stats(self):
        """Optional per-run diagnostics, merged into the output JSON cell."""
        return {}


class PerUserSelector(Selector):
    """For selectors with no vectorised form: implement `select_one` and pay the Python loop."""

    def select_one(self, turn, u, asked_u, answered_u):
        raise NotImplementedError

    def select(self, turn, users, asked, answered):
        return np.asarray([self.select_one(turn, int(u), asked[u], answered[u]) for u in users],
                          dtype=np.int64)


class StaticArmSelector(Selector):
    """The six lit-anchored static arms, expressed as selectors.

    This is the regression harness for the whole sequential path: it must reproduce
    `interview_strategies.ask()` EXACTLY, so the order construction below is a verbatim copy of that
    function's branch (same SEED, same per-user RNG stream, same `q_max*4` draw for random_catalog --
    which is why q_max is a constructor argument: the random_catalog order depends on it).

    A static arm ignores `answered` entirely. That is the null hypothesis of Act 1, made explicit in
    the type system: the adaptive selectors have exactly one extra argument available to them."""

    def __init__(self, arm, glob, bank, q_max):
        assert arm in ST.ARMS, f"unknown arm {arm}"
        self.arm, self.glob, self.bank, self.q_max = arm, glob, bank, int(q_max)
        self.name = arm + STATIC_SUFFIX
        self._orders = None

    def _order(self, u, n_items):
        if self.arm == "random_catalog":
            rng = np.random.default_rng(ST.SEED * 1000 + u)
            return rng.choice(n_items, size=min(self.q_max * 4, n_items), replace=False)
        if self.arm == "random_bank":
            rng = np.random.default_rng(ST.SEED * 1000 + u)
            return rng.permutation(self.bank)
        return self.glob[self.arm]

    def reset(self, ctx):
        super().reset(ctx)
        n, ni = ctx["n_users"], ctx["n_items"]
        if self.arm in ("random_catalog", "random_bank"):
            # Materialised once per run: rebuilding the per-user RNG stream every turn would be both
            # slower and a place for the two paths to disagree.
            self._orders = [self._order(u, ni) for u in range(n)]
        else:
            self._orders = None
        probe = self._orders[0] if self._orders is not None else self.glob[self.arm]
        assert len(probe) >= self.q_max, (
            f"{self.arm}: order length {len(probe)} < budget {self.q_max}; ST.ask would return a "
            f"SHORT interview here and the sequential path cannot reproduce it")

    def select(self, turn, users, asked, answered):
        if self._orders is None:
            return np.full(len(users), int(self.glob[self.arm][turn]), dtype=np.int64)
        return np.asarray([int(self._orders[int(u)][turn]) for u in users], dtype=np.int64)


# ===================================================================== the sequential loop
def _json_ready(o):
    """numpy scalars -> python scalars, tuples -> lists, recursively. json.dump's default= hook."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serialisable: {type(o)}")


def _write_atomic(path, blob, indent=None):
    """tmp + os.replace, so a crash mid-write can never leave a truncated checkpoint behind."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(blob, fh, default=_json_ready, indent=indent)
    os.replace(tmp, path)
    return path


def _savez_atomic(path, **arrays):
    """np.savez_compressed via tmp + os.replace. The per-user npz is rewritten after EVERY cell, so
    without this a crash during the write would destroy every previously banked vector."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


def _load_interview_ckpt(path, contract, log=logln):
    """Read a per-turn interview sidecar, REFUSING it unless the contract matches exactly.

    Returns None (start from turn 0) or the restored state. Everything that was a TUPLE when the
    interview ran is restored as a tuple, not as a list: `answered[u]` is hashed by the selectors'
    state grouping (`tuple(answered[u])`), and a list inside that tuple is unhashable -- so a sloppy
    reload would not be subtly wrong, it would crash. Restored as tuples, it is bit-identical."""
    if not path or not os.path.exists(path):
        return None
    try:
        blob = json.load(open(path))
    except Exception as e:                                    # a torn file is not a reason to die
        log(f"[adaptive] interview checkpoint {path} UNREADABLE ({e}) -- starting this cell fresh")
        return None
    if blob.get("_contract") != contract:
        log(f"[adaptive] interview checkpoint {path} REFUSED: contract mismatch -- recomputing\n"
            f"    cached={blob.get('_contract')}\n    wanted={contract}")
        return None
    cs = blob["conc_state"]
    st = {
        "turns_done": int(blob["turns_done"]),
        "acts": [list(map(int, a)) for a in blob["acts"]],
        "asked": [list(map(int, a)) for a in blob["asked"]],
        "answered": [[(int(i), int(v)) for i, v in a] for a in blob["answered"]],
        "conc_state": {
            "asked": [[int(c) for c in a] for a in cs["asked"]],
            "answered": [[(int(c), float(v)) for c, v in a] for a in cs["answered"]],
            "tokens": [[(int(c), float(v), float(f)) for c, v, f in a] for a in cs["tokens"]],
        },
        "info": blob["info"],
    }
    log(f"[adaptive] interview checkpoint {os.path.basename(path)} ACCEPTED: "
        f"{st['turns_done']} turn(s) already banked -> resuming from turn {st['turns_done'] + 1}")
    return st


def run_interview(selector, k, n_users, n_items, lvl, ctx=None, sel_batch=0, log=logln,
                  collect_dispersion=True, conc=None, ckpt_path=None, ckpt_contract=None):
    """Ask `k` questions sequentially, re-invoking `selector` after every answer.

    Returns (asked, answered, info):
      asked[u]    list of the ITEM ids asked, in ask order (this is what the scoring path consumes)
      answered[u] list of (item_id, level) for the subset the user could actually answer
      info        per-turn wall time and the number of DISTINCT ask-prefixes (the dispersion
                  diagnostic Act 2 wants; free to collect here, so we do)

    NO USER IS EVER DROPPED, SAMPLED OR CAPPED: the loop covers range(n_users) every turn.

    THE TWO-TYPE ACTION SPACE (added 2026-08-05 for the conversational-bandit family, whose whole
    point is that a key-term question and an item question live in ONE action space; see
    conversational_bandit_selectors.py). It is OPT-IN and inert unless `conc` is supplied:

        action id in [0, n_items)            -> ask about that ITEM               (the only case today)
        action id in [n_items, n_items + C)  -> ask about key-term c = a - n_items

    `conc` is {"n_concepts": C, "answer": fn(u, c) -> (answerable, value)}. The selector sees the FULL
    action ids in `asked[u]`, so the no-re-ask assert covers both types and the k-question budget is
    genuinely SHARED between them -- a concept question costs exactly one item question, which is the
    contract that makes the comparison mean anything. An unanswerable concept (BAND_REFUSE, or fewer
    than 2 rated members) BURNS the turn exactly as an unrated item does.

    Concept answers are handed back to the selector through `ctx["conc_state"]`, a dict of two lists
    mutated in place. That is deliberate: `Selector.select(turn, users, asked, answered)` is the
    published API and every other selector implements it, so widening the signature would break the
    contract this file exists to define. The item-side return values are unchanged, so the entire
    scoring path (`score_arm`, the floor rule, the pool) sees exactly what it saw before.

    PER-TURN CHECKPOINTING (added 2026-08-07; HARD RULE 11a/b). `ckpt_path` + `ckpt_contract` bank the
    interview state -- acts, asked, answered, the concept state and the per-turn diagnostics -- after
    EVERY turn, atomically, signed with the contract that produced it. A rerun with the same contract
    resumes at the first unfinished turn; a rerun with any other contract refuses the file and
    recomputes. This is exactly the failure that cost a 4-hour run on 2026-08-05: it died at turn
    14/16 and nothing was on disk.

    RESUMING IS EXACT, and the reason is worth stating because it is the thing that could silently be
    wrong. Every selector in this family is a pure function of (turn, asked, answered, concept state):
    the posterior caches are memoisation, the ConLinUCB-BS uniform draw is seeded on (seed, turn), and
    CLiSK's smoothing noise on (seed, turn, 7). So replaying from a banked state at turn t gives the
    same questions the uninterrupted run would have asked. The only things that do NOT survive are the
    selector's own per-turn DIAGNOSTIC lists (distinct states, key-term asks), which start empty on a
    fresh object -- so the banked prefix of every per-turn list is stitched back on at the end, and the
    cell records `resumed_from_turn` so a stitched row is never mistaken for an uninterrupted one.
    """
    acts = [[] for _ in range(n_users)]                      # FULL action ids (items + key-terms)
    asked = [[] for _ in range(n_users)]                     # ITEM ids only -> the scoring path
    answered = [[] for _ in range(n_users)]
    n_conc = int(conc["n_concepts"]) if conc else 0
    conc_state = {"asked": [[] for _ in range(n_users)], "answered": [[] for _ in range(n_users)],
                  "tokens": [[] for _ in range(n_users)]}
    info = {"turn_seconds": [], "distinct_prefixes": [], "answer_rate": [], "concept_asks": []}
    t_start, banked = 0, {}
    st = _load_interview_ckpt(ckpt_path, ckpt_contract, log=log)
    if st is not None and st["turns_done"] < int(k):
        acts, asked, answered = st["acts"], st["asked"], st["answered"]
        conc_state = st["conc_state"]
        banked = st["info"]
        t_start = int(st["turns_done"])
        info = {key: list(banked.get(key, []))[:t_start] for key in info}
    elif st is not None:
        # Fully banked: replay nothing at all, this cell's interview is done.
        acts, asked, answered = st["acts"], st["asked"], st["answered"]
        conc_state, banked = st["conc_state"], st["info"]
        t_start = int(k)
        info = {key: list(banked.get(key, []))[:t_start] for key in info}

    base_ctx = {"n_users": n_users, "n_items": n_items, "q_max": int(k), "log": log,
                "conc_state": conc_state, "n_concepts": n_conc}
    base_ctx.update(ctx or {})
    selector.reset(base_ctx)

    all_users = np.arange(n_users, dtype=np.int64)
    step = n_users if sel_batch in (0, None) else int(sel_batch)
    for t in range(t_start, k):
        t0 = time.time()
        for s in range(0, n_users, step):
            users = all_users[s:s + step]
            q = np.asarray(selector.select(t, users, acts, answered), dtype=np.int64)
            assert q.shape == (len(users),), f"selector {selector.name} returned {q.shape}"
            for u, i in zip(users.tolist(), q.tolist()):
                assert 0 <= i < n_items + n_conc, (
                    f"selector {selector.name} returned action {i} out of range "
                    f"[0,{n_items + n_conc})")
                assert i not in acts[u], (
                    f"selector {selector.name} re-asked action {i} to user {u} at turn {t}; a repeat "
                    f"burns a question for zero information and silently shortens the interview")
                acts[u].append(i)
                if i < n_items:
                    asked[u].append(i)
                    d = lvl[u]
                    if i in d:                               # identical test to interview_strategies.ask
                        answered[u].append((i, d[i]))
                else:
                    c = i - n_items
                    ok, v = conc["answer"](u, c)
                    conc_state["asked"][u].append(c)
                    # tokens: what a concept-consuming recommender folds -- a REFUSAL is a token too
                    # (v=0, ref=1), exactly as concept_cold_check folds a burned concept question.
                    conc_state["tokens"][u].append((c, v if ok else 0.0, 0.0 if ok else 1.0))
                    if ok:
                        conc_state["answered"][u].append((c, v))
        dt = time.time() - t0
        info["turn_seconds"].append(round(dt, 2))
        info["answer_rate"].append(round(float(np.mean([len(a) for a in answered])), 4))
        info["concept_asks"].append(int(sum(len(a) for a in conc_state["asked"])))
        if collect_dispersion:
            info["distinct_prefixes"].append(len({tuple(a) for a in acts}))
        log(f"[adaptive] {selector.name:16s} turn {t + 1}/{k}: {dt:.1f}s  "
            f"answered so far {info['answer_rate'][-1]:.2f}  "
            f"concept-asks {info['concept_asks'][-1]}  "
            f"distinct ask-prefixes {info['distinct_prefixes'][-1] if collect_dispersion else '-'}")
        if ckpt_path:
            ck_info = dict(banked)
            ck_info.update(info)
            ck_info.update(selector.stats())
            for key, v in list(ck_info.items()):
                # per-turn lists from a FRESH selector cover only the turns run in THIS process;
                # re-attach the banked prefix so the sidecar always describes turns 1..t.
                if isinstance(v, list) and len(v) < t + 1 and isinstance(banked.get(key), list):
                    ck_info[key] = banked[key][:t + 1 - len(v)] + v
            _write_atomic(ckpt_path, {
                "_contract": ckpt_contract, "turns_done": t + 1, "k": int(k),
                "selector": selector.name, "acts": acts, "asked": asked, "answered": answered,
                "conc_state": conc_state, "info": ck_info})
    if t_start:
        info["resumed_from_turn"] = t_start
    info.update(selector.stats())
    for key, v in list(info.items()):
        if isinstance(v, list) and len(v) < k and isinstance(banked.get(key), list):
            info[key] = banked[key][:k - len(v)] + v
    if conc and info["concept_asks"][-1] == 0:
        # A key-term arm that never asked a key-term is NOT a result about concepts -- it is the arm
        # degenerating to its item-only special case. Say so loudly so no table row can be read as
        # "the concept channel did not help" when the concept channel was never used.
        log(f"[adaptive] *** {selector.name}: ZERO key-term questions asked in {k} turns -- this arm "
            f"DEGENERATED to item-only. Do not read this row as evidence about concepts. ***")
    info["_conc_state"] = conc_state
    info["_acts"] = acts
    return asked, answered, info


def common_asked(asked):
    """asked[0] if EVERY user was asked the same items in the same order, else None.

    This is exactly run_interview_table's `asked[0] if arm in GLOBAL_ARMS else None` rule, generalised
    from the arm's name to the realised ask-lists: it is what lets GolbandiLeaf build its partition
    once. For a static global arm it returns asked[0] (same as today); for an adaptive selector it
    returns asked[0] at k=1 (everyone gets the same opener) and None once the policy branches."""
    first = asked[0]
    for a in asked:
        if a != first:
            return None
    return list(first)


# ===================================================================== regression check
def regression_check(D=None, budgets=(1, 2, 4, 8, 16), arms=None, log=logln):
    """REQUIREMENT (a): a static arm through the sequential path == the batch path, exactly.

    HOW IT WORKS. For each (arm, k) we build the batch answer set with the untouched
    `interview_strategies.ask()` and the sequential answer set with `StaticArmSelector` driven by
    `run_interview`, over the FULL 10,000-user test cohort, and assert Python list equality of BOTH
    returned structures. List equality is the strongest available statement: it compares ask ORDER,
    ask IDENTITY, which questions were answerable, and the half-star LEVEL attached to each answer --
    i.e. every input any consumer downstream can see. It is an assert, not a report: a mismatch means
    every adaptive number from this file is uninterpretable.

    `entropy0` at k=8 is asserted and printed by name because that is the named acceptance cell."""
    arms_req = list(arms or ST.ARMS)
    assert "entropy0" in arms_req and 8 in list(budgets), (
        "the NAMED acceptance cell entropy0/k=8 must be inside the regression grid; got "
        f"arms={arms_req} budgets={list(budgets)}")
    D = D if D is not None else load_arm_n(log=log)
    ni, n = D["n_items"], D["te_tr"].shape[0]
    glob, bank = ST.build_orders(D["g_train"], D["train"], ni, log=log)
    lvl = ST.level_lookup(D["g_te_tr"])
    arms = arms_req
    ok = 0
    for arm in arms:
        for k in budgets:
            a_batch, t_batch = ST.ask(arm, glob, bank, n, ni, lvl, k)
            sel = StaticArmSelector(arm, glob, bank, q_max=k)
            a_seq, t_seq, _info = run_interview(sel, k, n, ni, lvl, log=lambda *_: None,
                                                collect_dispersion=False)
            assert a_seq == a_batch, f"ASK MISMATCH arm={arm} k={k}"
            assert t_seq == t_batch, f"ANSWER MISMATCH arm={arm} k={k}"
            n_ans = float(np.mean([len(x) for x in t_seq]))
            log(f"[regression] {arm:14s} k={k:2d}  IDENTICAL  "
                f"(users={n} answered={n_ans:.3f}/{k})")
            ok += 1
    log(f"[regression] PASS: {ok} (arm,budget) cells byte-identical through the sequential path; "
        f"the named cell entropy0 k=8 is among them")
    return True


# ===================================================================== selector construction
def _concept_stack(D, a, cache, Y=None, repr_mode=None, log=logln):
    """The shared concept action space (genome bank + behavioural answers), built ONCE per process.

    The cache key carries the ANSWER CONVENTION (`--concept_neg_cap`) as well as the key-term
    representation, because those two things change the stack in ways that are not interchangeable:
    `apply_neg_cap=True` is the population-wide negative-mass cap over all 1,031 concepts, while 0 is
    the CERTIFIED-ERA convention the concept channel cv3 was trained under (`build_greedy_env`), which
    must be paired with `cap_negatives` over the revealed token list at fold time. Numbers from the
    two are never differenced -- see docs/results/CONVBANDIT_PORT_ON_OUR_RULER.md sec. C5."""
    import conversational_bandit_selectors as CB
    repr_mode = repr_mode or a.keyterm_repr
    neg_cap = bool(int(getattr(a, "concept_neg_cap", 1)))
    ckey = ("conc_stack", repr_mode, bool(a.keyterm_norm_match), int(a.keyterm_n_rep),
            neg_cap, Y is not None)
    if ckey not in cache:
        cache[ckey] = CB.build_concept_stack(
            D, a.snapshot, keyterm_repr=repr_mode, n_rep=a.keyterm_n_rep,
            norm_match=a.keyterm_norm_match, Y=Y, log=log, apply_neg_cap=neg_cap)
        cache[ckey]["_repr"] = repr_mode
    return cache[ckey]


def validate_specs(specs, glob, budgets, log=logln):
    """STARTUP pre-flight on every selector spec. Names only -- nothing expensive is built.

    THE SCAR THIS EXISTS FOR. `build_selector` is only called from inside the (spec x budget) loop,
    so an unknown `fixedorder:` name raised its assert at the moment that cell was REACHED. With the
    bad spec last in the list, that was the last step of a 4-hour run. Every name a run depends on
    is now resolved before the first interview turn, and a typo costs seconds.

    Also checks stored-order LENGTH against the largest budget, which is the other way a stored
    order fails late: a 16-step sequence asked for k=32 must refuse, not silently repeat or extend.

    TWO PHASES, because the two registries become available at different times. `glob=None` is the
    EARLY phase, run straight after argument parsing -- it resolves everything that needs no data
    (stored greedy names, their file, their length, static arm names), so a typo there costs seconds
    rather than the ML-25M load. The full call, after `ST.build_orders`, additionally resolves the
    scored orders in `glob`. Both run before the first interview turn.
    """
    known_static = {a for a in ST.ARMS}
    phase = "EARLY (pre-data)" if glob is None else "FULL"
    bad = []
    for spec in specs:
        if spec.startswith("fixedorder:") or spec.startswith("treestatic:"):
            which = spec.split(":", 1)[1]
            if which in GREEDY_FIXED_ORDERS and spec.startswith("fixedorder:"):
                if not os.path.exists(GREEDY_ORDER_FILE):
                    bad.append(f"{spec}: stored-order file missing at {GREEDY_ORDER_FILE}")
                    continue
                try:
                    order, _ac = load_greedy_order(which, 0)
                except Exception as e:                                    # noqa: BLE001
                    bad.append(f"{spec}: unreadable stored order ({e})")
                    continue
                if len(order) < max(budgets):
                    bad.append(f"{spec}: {len(order)} stored steps < largest budget {max(budgets)}")
            elif glob is not None and which not in glob:
                bad.append(f"{spec}: unknown stored order '{which}'; have "
                           f"{sorted(set(glob) | set(GREEDY_FIXED_ORDERS))}")
        elif spec.endswith(STATIC_SUFFIX):
            if spec[:-len(STATIC_SUFFIX)] not in known_static:
                bad.append(f"{spec}: unknown static arm; have {sorted(known_static)}")
        elif spec not in (("glm_ucb", "thompson", "bald", "aopt", "golbandi", "golbandi_leaf")
                          + CONV_BANDIT_SPECS + DPE_SPECS + IGCN_SPECS):
            # Not a hard failure: build_selector owns the full list and raises its own SystemExit.
            log(f"[preflight] NOTE: '{spec}' is not in the pre-flight table; build_selector will "
                f"resolve or reject it at its cell")
    if bad:
        raise SystemExit(f"[preflight/{phase}] refusing to start -- bad selector spec(s):\n  "
                         + "\n  ".join(bad))
    log(f"[preflight/{phase}] {len(specs)} selector spec(s) validated: {', '.join(specs)}")


def validate_paths(a, log=logln):
    """STARTUP existence check on every FILE the run names. Same scar as `validate_specs`, one level
    down: `--snapshot t2i26c_ep31.pt` (a bare basename rather than the path) parses fine, survives
    the whole data load, and only dies inside `torch.load` when the tower is first touched. Anything
    the run cannot finish without is resolved here, before the ML-25M load."""
    named = [("--snapshot", getattr(a, "snapshot", None)),
             ("--concept_ckpt", getattr(a, "concept_ckpt", None))]
    missing = [f"{flag} {p!r} does not exist" for flag, p in named
               if p and not os.path.exists(p)]
    if missing:
        raise SystemExit("[preflight] refusing to start -- missing file(s):\n  "
                         + "\n  ".join(missing))
    log("[preflight] all named checkpoint paths exist: "
        + ", ".join(os.path.basename(p) for _f, p in named if p))


def build_selector(spec, D, glob, bank, k, args, cache, log=logln):
    """spec -> Selector. `cache` holds the expensive shared artifacts (the frozen iALS belief, the
    Partial-VAE) so they are fitted/loaded ONCE per process, not once per budget."""
    if spec.endswith(STATIC_SUFFIX):
        return StaticArmSelector(spec[:-len(STATIC_SUFFIX)], glob, bank, q_max=k)
    ms = int(getattr(args, "conv_max_states", 0) or 0)
    if spec in ("glm_ucb", "thompson", "bald", "aopt"):
        # FOUR acquisitions, ONE posterior (`belief_mf.GaussianBelief`, the same object the belief_mf
        # CONSUMER row scores with), so the only thing that differs between these rows is the rule that
        # turns a posterior into a question:
        #   glm_ucb  mean + alpha*width                                 (Filippi NeurIPS'10 / ICF CIKM'13)
        #   thompson argmax over one posterior draw                     (Chapelle & Li NeurIPS'11)
        #   bald     mutual information I(u; r_i)                       (Houlsby arXiv:1112.5745)
        #            -- under this Gaussian likelihood it REDUCES to 0.5*log(1+sig^2/s2), a monotone
        #               function of the width, i.e. pure exploration. Documented, measured, not hidden.
        #   aopt     argmin tr(A_{t+1}^-1), closed form                 (Yu ICML'06 / Anava WWW'15)
        #            -- VALUE-FREE: it adapts only to which questions proved answerable, so it is a
        #               (near-)static design and belongs beside DRE in any table, not beside the
        #               adaptive arms.
        # Neither new arm takes a hyperparameter: s2 is belief_mf's own obs_noise, so there is nothing
        # to tune and nothing new in the interview contract (_ASK_ARGS is unchanged).
        import bandit_selectors as B
        if "belief" not in cache:
            # BANKED frozen basis: the seeded 10-minute iALS fit is the one thing a RESUMED run would
            # otherwise repay in full before doing any new work (HARD RULE 11b). Bit-identical on a
            # hit -- see bandit_selectors.load_or_fit_belief.
            cache["belief"] = B.load_or_fit_belief(D["train"], D["n_items"], log=log)
        ms = ms or B.DEFAULT_MAX_STATES
        if spec == "glm_ucb":
            return B.GLMUCBSelector(cache["belief"], alpha=args.ucb_alpha, max_states=ms)
        if spec == "bald":
            return B.BALDSelector(cache["belief"], max_states=ms)
        if spec == "aopt":
            return B.AOptimalSelector(cache["belief"], max_states=ms)
        return B.ThompsonSelector(cache["belief"], seed=args.ts_seed, max_states=ms)
    if spec == "golbandi_tree":
        # WSDM'11 SPLIT SEARCH (golbandi_tree_selector.py). Grown ONCE per process on the TRAIN
        # users, to the LARGEST budget in the run, and shared across budgets: a depth-d prefix of the
        # tree IS the depth-d tree, so re-growing per budget could only introduce disagreement.
        # Pair it with the `golbandi_leaf` consumer for the method as published; any other consumer
        # is the selector ported onto a different node model, and must be labelled that way.
        import golbandi_tree_selector as GT
        ck = ("golbandi_tree", int(args.golbandi_depth or 0))
        if ck not in cache:
            depth = int(args.golbandi_depth) if args.golbandi_depth else max(
                k, *[int(x) for x in args.budgets.split(",")])
            sel = GT.GolbandiTreeSelector(D["g_train"], D["n_items"], depth=depth,
                                          like_level=GT.LIKE_MIN_LEVEL, like_min=4.0,
                                          xi=args.golbandi_xi,
                                          min_child_users=args.golbandi_min_child,
                                          cache_path=args.golbandi_tree_cache, log=log)
            sel.fit(cache_path=args.golbandi_tree_cache)
            cache[ck] = sel
        return cache[ck]
    if spec in IGCN_SPECS:
        # IGCN (Rashid, Karypis & Riedl, SIGKDD Expl. 2008) -- the ONE adaptive member of the family
        # our entropy0/HELF/popularity arms come from. Clusters and the global-IG order are built
        # ONCE per process on the TRAIN users and shared across budgets (both are budget-independent
        # by construction), exactly as golbandi_tree shares its tree.
        #   igcn        = the paper's parameters (c=300, l=150, switch after i=5 answers)
        #   igcn_i1     = LABELLED DEVIATION, switch after 1 answer. At k<=4 no user can reach 5
        #                 answers, so the published form is EXACTLY the global-IG static order there;
        #                 this row is the one that exercises the personalized step at every budget.
        #                 It is a different row, never a repair of the first.
        #   igcn_page15 = the paper-LITERAL re-rank period (one page of n=15 items). At k<=16 it can
        #                 fire at most once, which is why it is not the default.
        import igcn_selector as IGS
        sw = 1 if spec == "igcn_i1" else int(args.igcn_switch)
        rc = PAGE_RECOMPUTE if spec == "igcn_page15" else int(args.igcn_recompute)
        ck = ("igcn", int(args.igcn_clusters), int(args.igcn_nbr), sw, rc)
        if ck not in cache:
            sel = IGS.IGCNSelector(D["g_train"], D["n_items"],
                                   n_clusters=args.igcn_clusters, n_nbr=args.igcn_nbr,
                                   switch_i=sw, recompute_every=rc, seed=args.ts_seed,
                                   kmeans_iter=args.igcn_kmeans_iter,
                                   nn_chunk=args.igcn_nn_chunk, ig_chunk=args.igcn_ig_chunk,
                                   cache_path=args.igcn_cluster_cache, log=log)
            sel.fit()
            cache[ck] = sel
        sel = cache[ck]
        sel.name = spec
        # counters are cumulative on a shared object; zero them so each (spec, budget) cell reports
        # its own personalised_rate rather than the run-to-date total
        sel._n_pers = sel._n_glob = sel._n_degen = 0
        sel._pers_per_turn = []
        return sel
    if spec.startswith("fixedorder:"):
        # A STORED question order, walked inside THIS harness. It exists so a comparator like the
        # greedy items-only sequence can be differenced PER USER against the arms in the same run:
        # `gtport_greedy_rescore.py` had to re-score it in a separate file and then prove the
        # entropy0 control matched at 0.0e+00 before any difference was allowed. Running it as a
        # selector in the same output makes that identity hold by construction instead.
        #   fixedorder:greedy_items -> experiments/battery/greedy_static_i26c.json, arm "items-only",
        #                              channel-0 (item) steps only, verbatim; nothing is re-selected.
        import golbandi_tree_selector as GT
        which = spec.split(":", 1)[1]
        ni = int(D["n_items"])
        if which in GREEDY_FIXED_ORDERS:
            order, asks_conc = load_greedy_order(which, ni)
            assert len(order) >= k, (
                f"fixedorder:{which} has {len(order)} steps < budget {k}; the stored sequence "
                f"cannot answer this budget and must not be silently extended")
            s = GT.StaticOrderTreeSelector(order, log=log)
            if asks_conc:
                # The concept ACTION SPACE only. The stored order contributes no answer convention:
                # the stack is built from THIS cell's flags (--concept_neg_cap, --keyterm_repr), and
                # the greedy JSON's own provenance fields are never consulted.
                s.stack = _concept_stack(D, args, cache, log=log)
                nc = int(s.stack["n_concepts"])
                bad = [q for q in order[:k] if q >= ni + nc]
                assert not bad, (f"fixedorder:{which} has concept ids beyond the {nc}-concept "
                                 f"action space: {bad}")
            s.name = spec
            kinds = "".join("c" if q >= ni else "i" for q in order[:k])
            log(f"[fixedorder] {which}: {len(order)} stored steps, first {k} = {kinds} "
                f"{list(order[:k])} (concept ids shown offset by n_items={ni}); "
                f"answer convention comes from THIS cell, not the stored order")
            return s
        assert which in glob, (f"fixedorder: unknown stored order '{which}'; have "
                               f"{sorted(set(glob) | set(GREEDY_FIXED_ORDERS))}")
        order = list(glob[which])
        s = GT.StaticOrderTreeSelector(order, log=log)
        s.name = spec
        log(f"[fixedorder] {which}: first {k} questions {list(order[:k])}")
        return s
    if spec.startswith("treestatic:"):
        # THE SYMMETRIC COMPARATOR, through golbandi_tree_selector's OWN class rather than through
        # StaticArmSelector. Both return order[turn] to every user, so they are behaviourally the same
        # object; running one of each (treestatic:entropy0 alongside entropy0-static) turns that claim
        # into a measured identity instead of a code-reading claim.
        import golbandi_tree_selector as GT
        arm = spec.split(":", 1)[1]
        assert arm in glob, f"treestatic: unknown global order '{arm}'; have {sorted(glob)}"
        s = GT.StaticOrderTreeSelector(glob[arm], log=log)
        s.name = spec
        return s
    if spec == "dre":
        # DRE (Kweon WWW'20): a LEARNED STATIC seed set, one per budget, trained offline by
        # scripts/baselines/dre_train_ours.py on the TRAIN users only. It is not cached across
        # budgets on purpose -- the k=2 set is not a prefix of the k=16 set.
        import dre_selector as DRE
        return DRE.DRESelector(k, tag=args.dre_tag, dup_policy=args.dre_dup_policy, log=log)
    if spec == "eddi_ig":
        import eddi_selector as E
        if "eddi" not in cache:
            import eddi_pvae
            eck = os.path.join(_ROOT, ".cache", "baselines", "eddi_ml25m_liang.pt")
            cache["eddi"] = eddi_pvae.load_trained(eck, D["n_items"], log=log).model
        return E.EDDIInfoGainSelector(cache["eddi"], cand_topm=args.eddi_topm)
    if spec in ("eddi_ig3", "eddi_ig3_zero"):
        # THREE-OUTCOME response model (unanswered / like / dislike); see eddi_selector.py.
        import eddi_selector as E
        if "eddi" not in cache:
            import eddi_pvae
            eck = os.path.join(_ROOT, ".cache", "baselines", "eddi_ml25m_liang.pt")
            cache["eddi"] = eddi_pvae.load_trained(eck, D["n_items"], log=log).model
        return E.EDDIThreeOutcomeSelector(
            cache["eddi"], D["g_train"],
            dislike_mode="zero" if spec == "eddi_ig3_zero" else "signed", beta=args.eddi_beta)
    if spec in DPE_SPECS:
        # DPE (Parapar & Radlinski, WSDM 2021). The published arm asks ITEMS via a Thompson bandit
        # over topics; `dpe_concept` is the labelled deviation onto the key-term action space. The
        # score_div is carried on the spec name so the three published arm-rankers (ILD, Hellinger,
        # Oracle) are three rows, exactly as the paper plots them.
        import dpe_selector as DPE
        if spec == "dpe_concept":
            stack = _concept_stack(D, a=args, cache=cache, log=log)
            sel = DPE.DPEConceptSelector(stack["n_concepts"], seed=args.ts_seed,
                                         burn=args.dpe_burn, update=args.dpe_update, log=log)
            sel.stack = stack
            return sel
        args.dpe_score_div = {"dpe": "ild", "dpe_hellinger": "hellinger",
                              "dpe_oracle": "oracle"}[spec]
        return DPE.build_dpe(D, args, cache, k, log=log)
    if spec in CONV_BANDIT_SPECS:
        # THE CONVERSATIONAL-BANDIT FAMILY (ConUCB WWW'20 / ConLinUCB AAAI'23 / CLiME KDD'25). These
        # are the only arms with a TWO-TYPE action space: they may ask about an item or about a
        # key-term (a genome concept) at any turn, out of one shared k-question budget. The posterior
        # is `belief_mf.GaussianBelief` -- the SAME object glm_ucb and thompson use, so the difference
        # between those rows and these is the action space and nothing else.
        import bandit_selectors as B
        import conversational_bandit_selectors as CB
        if "belief" not in cache:
            # BANKED frozen basis: the seeded 10-minute iALS fit is the one thing a RESUMED run would
            # otherwise repay in full before doing any new work (HARD RULE 11b). Bit-identical on a
            # hit -- see bandit_selectors.load_or_fit_belief.
            cache["belief"] = B.load_or_fit_belief(D["train"], D["n_items"], log=log)
        bel = cache["belief"]
        ms = ms or B.DEFAULT_MAX_STATES
        # ConUCB's Eq. 4 DEFINES its key-term feature as the weight-normalised member mean, so that arm
        # pins the representation unless the caller overrides it on purpose (--keyterm_repr_conucb).
        repr_mode = args.keyterm_repr
        if spec == "conucb" and not getattr(args, "keyterm_repr_conucb", False):
            repr_mode = "uniform_member_mean"
        stack = _concept_stack(D, args, cache, Y=bel.Y, repr_mode=repr_mode, log=log)
        kw = dict(alpha=args.ucb_alpha, beta=args.conv_beta, seed=args.ts_seed,
                  b_schedule=args.conv_b_schedule, max_states=ms,
                  turn_rule=(args.conv_turn_rule or None))
        if spec == "conlinucb_mcr":
            return CB.ConLinUCBMCRSelector(bel, stack, **kw)
        if spec == "conlinucb_bs":
            return CB.ConLinUCBBSSelector(bel, stack, **kw)
        if spec == "conlinucb_bs_static":
            return CB.ConLinUCBBSSelector(bel, stack, key_term_only=True, **kw)
        if spec == "clime":
            return CB.CLiMESelector(bel, stack, n_eig=args.clime_n_eig,
                                    smooth_sigma=args.clisk_sigma, **kw)
        return CB.ConUCBSelector(bel, stack, lam=args.conucb_lam, lam_til=args.conucb_lam_til, **kw)
    raise SystemExit(f"unknown selector spec '{spec}'. Known: <arm>{STATIC_SUFFIX} for arm in "
                     f"{ST.ARMS}, plus dre, golbandi_tree, glm_ucb, thompson, bald, aopt, "
                     f"eddi_ig, eddi_ig3, "
                     f"eddi_ig3_zero, {','.join(IGCN_SPECS)}, "
                     f"{','.join(CONV_BANDIT_SPECS)}, {','.join(DPE_SPECS)}")


def _smoke_slice(D, n_smoke, log=logln):
    """SMOKE ONLY. Keeps the FIRST n users of the test cohort and nothing else changes. This IS a
    truncation of the evaluation cohort, so it is loud, explicit, opt-in, and its output is stamped
    non-citable. It exists to exercise the code path in minutes, never to produce a number."""
    log("=" * 100)
    log(f"[adaptive] *** SMOKE: TRUNCATING THE TEST COHORT TO THE FIRST {n_smoke} USERS ***")
    log("[adaptive] *** every number from this run is a CODE-PATH CHECK and is NOT CITABLE ***")
    log("=" * 100)
    Ds = dict(D)
    for key in ("te_tr", "te_te", "g_te_tr", "pool"):
        Ds[key] = D[key][:n_smoke]
    return Ds


# ===================================================================== run contracts / run lock
# Every CLI argument that can change WHICH QUESTIONS get asked. A banked interview may be reused only
# if all of these agree; anything outside this list (consumers, the tower snapshot, --per_user) affects
# only the SCORING half and cannot invalidate an interview.
_ASK_ARGS = ("sel_batch", "ucb_alpha", "ts_seed", "eddi_topm", "eddi_beta",
             "golbandi_depth", "golbandi_xi", "golbandi_min_child", "golbandi_tree_cache",
             "dre_tag", "dre_dup_policy",
             "igcn_clusters", "igcn_nbr", "igcn_switch", "igcn_recompute", "igcn_kmeans_iter",
             "keyterm_repr", "keyterm_repr_conucb", "keyterm_n_rep", "keyterm_norm_match",
             "conv_beta", "conv_turn_rule", "conv_b_schedule", "conucb_lam", "conucb_lam_til",
             "clime_n_eig", "clisk_sigma", "smoke_users",
             "dpe_topics", "dpe_burn", "dpe_update", "concept_neg_cap")


def interview_contract(a, spec, k, n_users, n_items, extra=None):
    """The signature a banked interview must match before a single turn of it may be reused.

    Deliberately NOT a hash: it is written into the sidecar in full, so a refusal prints exactly which
    field moved. `budgets` is absent on purpose -- an interview is banked per (selector, budget), so k
    is already in the key -- and the CONSUMER side is absent because it cannot change a question."""
    c = {"spec": spec, "k": int(k), "n_users": int(n_users), "n_items": int(n_items)}
    for key in _ASK_ARGS:
        v = getattr(a, key, None)
        c[key] = v if v is None or isinstance(v, (int, float, str, bool)) else str(v)
    c.update(extra or {})
    return c


def results_contract(a, specs, budgets, recs, n_users):
    """The signature the OUTPUT json is stamped with. A rerun that matches it resumes into the same
    file, skipping cells already scored; a rerun that does not match REFUSES rather than mixing two
    information contracts into one table (the failure `surprise-means-debug-the-harness` names)."""
    c = interview_contract(a, "*", 0, n_users, 0)
    c.update({"specs": list(specs), "budgets": list(budgets), "consumers": list(recs),
              "arch": a.arch, "snapshot": os.path.basename(a.snapshot), "no_floor": bool(a.no_floor),
              "concept_ckpt": os.path.basename(a.concept_ckpt),
              "no_concept_consumer": bool(a.no_concept_consumer),
              "golbandi_lam": a.golbandi_lam})
    c.pop("spec"), c.pop("k")
    return c


def _pid_alive(pid):
    try:
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not h:
            return False
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    except Exception:
        return os.path.exists(f"/proc/{int(pid)}")


def acquire_run_lock(outp, log=logln):
    """HARD RULE 11(e): refuse to open an output file another LIVE process holds.

    On 2026-08-05 a relaunch of an identical cell truncated the log its still-live twin was writing
    to and destroyed a 105-CPU-minute progress trace. A pid file next to the output makes that a
    refusal instead of a silent overwrite; a stale lock (the writer is gone) is reclaimed and said so."""
    lockp = outp + ".lock"
    if os.path.exists(lockp):
        try:
            old = json.load(open(lockp))
        except Exception:
            old = {}
        pid = int(old.get("pid", -1))
        if pid > 0 and pid != os.getpid() and _pid_alive(pid):
            raise SystemExit(
                f"[adaptive] REFUSING TO START: {outp} is held by LIVE pid {pid} "
                f"(started {old.get('started')}, label {old.get('label')}). Two runs must never "
                f"share an output path (HARD RULE 11e). Use a different --label, or stop that run.")
        log(f"[adaptive] stale run lock from dead pid {pid} reclaimed ({lockp})")
    _write_atomic(lockp, {"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "label": os.path.basename(outp), "argv": sys.argv})
    return lockp


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="1,2,4,8,16")
    ap.add_argument("--selectors", default="entropy0-static,glm_ucb,thompson,eddi_ig",
                    help="comma list; '<arm>-static' for any of " + ",".join(ST.ARMS))
    ap.add_argument("--only", default=None, help="comma list of consumers (default: all of RECS)")
    ap.add_argument("--snapshot", default=SNAP_DEFAULT)
    ap.add_argument("--arch", default="i25", choices=["i25", "i26"])
    ap.add_argument("--label", default=None)
    ap.add_argument("--no_floor", action="store_true",
                    help="NO-FLOOR contract (the Act-1 primary). See run_interview_table for the rule.")
    ap.add_argument("--sel_batch", type=int, default=0,
                    help="users handed to the selector per call; 0 = the whole cohort at once "
                         "(default, and what lets state-grouping selectors vectorise). This is a "
                         "CHUNKING knob only -- every user is always processed.")
    ap.add_argument("--ucb_alpha", type=float, default=1.0,
                    help="GLM-UCB exploration constant. NOT fitted on test; see bandit_selectors.")
    ap.add_argument("--ts_seed", type=int, default=98765)
    ap.add_argument("--eddi_topm", type=int, default=0,
                    help="0 = score the FULL 18,359-item candidate pool (the default and the only "
                         "citable setting). >0 truncates the candidate pool and requires sign-off.")
    ap.add_argument("--golbandi_depth", type=int, default=0,
                    help="tree depth for the golbandi_tree selector; 0 = the largest budget in the "
                         "run. NOTE golbandi_leaf.MAX_DEPTH=8 caps the CONSUMER side for the same "
                         "reason (3^k answer patterns fragment the train cohort).")
    ap.add_argument("--golbandi_xi", type=float, default=None,
                    help="min node rating mass before a node may split. Default = the paper's "
                         "0.2%% of corpus rating mass (200,000/100M on Netflix).")
    ap.add_argument("--golbandi_min_child", type=int, default=0,
                    help="0 = paper-faithful (the paper relies on shrinkage, not a size floor).")
    ap.add_argument("--per_user", action="store_true",
                    help="also keep the PER-USER full/tail NDCG vector for every cell, in an npz "
                         "sidecar next to the JSON. It costs nothing (score_arm already computes "
                         "them) and it is the only thing a PAIRED bootstrap can be run from later -- "
                         "the unpaired 1.96*sqrt(2)*SE form is banned (ARM_N_PHASE1 sec 4b).")
    ap.add_argument("--golbandi_tree_cache", default=None,
                    help="path for the CONTRACT-SIGNED grown-tree sidecar (HARD RULE 11b). Growing "
                         "the tree on 140,768 train users is the only expensive step in this arm and "
                         "it is deterministic, so it is written once and reloaded; a cache whose "
                         "contract does not match is refused, never silently reused.")
    ap.add_argument("--golbandi_lam", default=None,
                    help="comma list of leaf shrinkage values for the golbandi_leaf CONSUMER "
                         "(default: its built-in 8.0, which is the ARCHIVED constant). SNAP-5 showed "
                         "lambda1 is load-bearing in the tree's own hierarchical leaf -- at 8 the "
                         "interview HURTS past depth 1, at the paper's 200 it helps monotonically -- "
                         "so the port must state and vary it. Each value becomes its own consumer row "
                         "named golbandi_leaf_lam<v>.")
    # ---------------- DRE (Kweon WWW'20), the peer-reviewed LEARNED-STATIC bar -----------------
    ap.add_argument("--dre_tag", default="",
                    help="suffix of the trained DRE artifacts to use "
                         "(.cache/baselines/dre_ml25m_liang_k<K><TAG>.pt). '' = the citable sets; "
                         "'_SMOKE' = the reduced-epoch code-path artifacts, which stamp the run.")
    ap.add_argument("--dre_dup_policy", default="waste", choices=list(("waste", "extend")),
                    help="DRE selects its k columns by independent argmax with NO duplicate guard, so "
                         "an item can be chosen twice; our harness forbids re-asking. 'waste' (default, "
                         "faithful) fills the wasted turn with DRE's LEAST-preferred item, which its "
                         "own decoder cannot see. 'extend' fills it with DRE's next-best item, which "
                         "REPAIRS the published method and must be labelled as a repair.")
    ap.add_argument("--eddi_beta", type=float, default=1.0,
                    help="eddi_ig3 only: how hard the frozen decoder tilts the item's base like-rate "
                         "per user, in logit units per z-score. 1.0 = default, NOT tuned.")
    # ---------------- conversational-bandit family (see conversational_bandit_selectors.py) --------
    ap.add_argument("--keyterm_repr", default="uniform_member_mean",
                    choices=["uniform_member_mean", "representative", "whitened_centroid"],
                    help="THE key-term feature vector. uniform_member_mean = ConUCB Eq.4's own "
                         "definition; representative = Hier-UCB's claim that a few representative "
                         "items drive key-term reward; whitened_centroid = OUR concept direction "
                         "(build_concept_dirs' recipe on the bandit's basis). Three papers, three "
                         "answers -- running all three adjudicates our own concept-fold design.")
    ap.add_argument("--keyterm_repr_conucb", action="store_true",
                    help="let --keyterm_repr override ConUCB's own Eq.4 definition. OFF by default: "
                         "ConUCB on a non-member-mean key-term vector is a hybrid, not ConUCB.")
    ap.add_argument("--keyterm_n_rep", type=int, default=20,
                    help="'representative' mode: top-N members by train count. NOT tuned; Hier-UCB "
                         "gives no number.")
    ap.add_argument("--keyterm_norm_match", type=int, default=1,
                    help="1 = rescale every key-term feature to the mean member ||y_a||, so the three "
                         "representations differ only in DIRECTION and the comparison is not "
                         "confounded by magnitude. 0 = raw scales.")
    ap.add_argument("--conv_beta", type=float, default=None,
                    help="ConLinUCB's ridge beta. Unset = our empirical population precision "
                         "Sigma_0^-1 (belief_mf's prior); a float = the paper's literal beta*I "
                         "(their default 1.0).")
    ap.add_argument("--conv_turn_rule", default=None, choices=[None, "schedule", "width"],
                    help="how the turn TYPE is chosen. Unset = each arm's own paper's rule "
                         "(schedule for ConUCB/ConLinUCB, width for CLiME). 'width' is the "
                         "one-action-space variant: items and key-terms in ONE argmax over "
                         "alpha*||phi||_{M^-1}.")
    ap.add_argument("--conv_b_schedule", default="5logt", choices=["5logt", "5t50", "always", "never"],
                    help="ConUCB's conversation frequency b(t). 5logt = 5*floor(log t) [verbatim]. "
                         "5t50 = 5*floor(t/50), the paper's other setting, which never fires at k<=16.")
    ap.add_argument("--conucb_lam", type=float, default=0.5,
                    help="ConUCB's shrinkage toward the key-term fit. The paper's own value is NOT "
                         "verified in our audit; 0.5 is the neutral midpoint of its stated range.")
    ap.add_argument("--conucb_lam_til", type=float, default=1.0)
    ap.add_argument("--clime_n_eig", type=int, default=1,
                    help="CLiME: how many smallest-eigenvalue eigendirections of M_t to score against.")
    ap.add_argument("--clisk_sigma", type=float, default=0.0,
                    help="CLiSK's Gaussian key-term-context smoothing (0 = pure CLiME).")
    # ---------------- IGCN (Rashid, Karypis & Riedl, SIGKDD Expl. 2008) ---------------------------
    ap.add_argument("--igcn_clusters", type=int, default=300,
                    help="c, the number of bisecting-k-means user clusters IGCN computes its "
                         "information gain over. 300 is the paper's own value [V].")
    ap.add_argument("--igcn_nbr", type=int, default=150,
                    help="l, the neighbourhood the PERSONALIZED step recomputes information gain "
                         "from. 150 is the paper's own value [V]; it is the thing that makes IGCN "
                         "adaptive rather than a seventh static order.")
    ap.add_argument("--igcn_switch", type=int, default=5,
                    help="i, answers required before the personalized step takes over. 5 is the "
                         "paper's own value [V]. NOTE at k<=4 no user can reach it, so the `igcn` "
                         "row IS the global-IG static order there -- read `igcn_personalised_rate` "
                         "off the cell before calling any k<=4 number adaptive.")
    ap.add_argument("--igcn_recompute", type=int, default=1,
                    help="turns between neighbourhood/IG recomputations. The paper re-ranks per "
                         "PAGE of 15 items; our budgets are k<=16, so the literal period fires at "
                         "most once and 1 (re-rank after every turn) is the default. The literal "
                         "setting is the separate `igcn_page15` spec, not a flag change.")
    ap.add_argument("--igcn_kmeans_iter", type=int, default=10)
    ap.add_argument("--igcn_cluster_cache", default=None,
                    help="path for the CONTRACT-SIGNED cluster sidecar (HARD RULE 11b). Clustering "
                         "140,768 users is the only expensive setup step and it is deterministic.")
    ap.add_argument("--igcn_nn_chunk", type=int, default=128,
                    help="interview STATES per neighbour-search block. Memory device only (HARD "
                         "RULE 11c): every user is searched against every training user.")
    ap.add_argument("--igcn_ig_chunk", type=int, default=256,
                    help="interview STATES per personalised-IG block. Memory device only.")
    # ---------------- DPE (Parapar & Radlinski WSDM'21); see dpe_selector.py -----------------------
    ap.add_argument("--dpe_topics", default="genres", choices=["genres", "genome"],
                    help="DPE's arm set. 'genres' = the paper's own granularity (20 MovieLens "
                         "genres, an item may belong to several) [V]; 'genome' = the 1,031-concept "
                         "genome bank every other concept row here uses.")
    ap.add_argument("--dpe_burn", default="none", choices=["none", "zero"],
                    help="what an UNANSWERABLE question pays the topic bandit. DPE's published "
                         "protocol has no burns (it asks inside the user's own rated set), so this "
                         "is ours: 'none' = no update at all (the conversational-bandit family's "
                         "convention, and the default); 'zero' = reward 0, i.e. the bandit learns "
                         "which topics are hard to answer -- a labelled repair, not the paper.")
    ap.add_argument("--dpe_update", default="paper", choices=["paper", "mean"],
                    help="Algorithm 1 prints mu <- (mu*k + r)/(k+2). 'paper' implements it as "
                         "printed; 'mean' uses the running-mean denominator k+1 and is a labelled "
                         "REPAIR. Never mix the two in one table.")
    ap.add_argument("--concept_neg_cap", type=int, default=1, choices=[0, 1],
                    help="THE CONCEPT ANSWER CONVENTION, and it is a contract (see "
                         "docs/results/CONVBANDIT_PORT_ON_OUR_RULER.md sec. C5). 1 = "
                         "signed_values(apply_neg_cap=True), the population-wide negative-mass cap "
                         "over all 1,031 concepts -- what every run before 2026-08-08 used. 0 = the "
                         "CERTIFIED-ERA convention the concept channel cv3 was TRAINED under "
                         "(apply_neg_cap=False plus cap_negatives over the REVEALED token list at "
                         "fold time), i.e. strategy_channel_suite.build_greedy_env's. The same 16 "
                         "greedy concepts score .2473/.1154 under 0 and .2298/.0977 under 1, so a "
                         "number from one convention may NEVER be differenced against the other.")
    ap.add_argument("--concept_ckpt",
                    default=os.path.join(_ROOT, "models", "i26c_ep31", "concept_cv3_ep08.pt"),
                    help="CERTIFIED concept channel used to CONSUME key-term answers on the 'ours' "
                         "row. Without it a concept question burns a turn and reaches no consumer, "
                         "which would test our harness rather than the method.")
    ap.add_argument("--no_concept_consumer", action="store_true",
                    help="score conversational arms with the ITEM-ONLY consumer (concept answers "
                         "burn the turn and are then discarded). Diagnostic only -- it is the "
                         "'what does the concept channel buy' ablation, and it is a FLOOR, not a row.")
    ap.add_argument("--conv_max_states", type=int, default=0,
                    help="cap on the number of interview STATES any per-state cache in the bandit "
                         "family may hold (HARD RULE 11c; 0 = bandit_selectors.DEFAULT_MAX_STATES). "
                         "Eviction costs recomputation, never correctness -- every cached value is a "
                         "pure function of its key -- and the realised hit rate lands in each cell.")
    ap.add_argument("--no_resume", action="store_true",
                    help="ignore any banked interview sidecars and scored cells and recompute "
                         "everything. The default is to RESUME, refusing any sidecar whose contract "
                         "does not match exactly.")
    ap.add_argument("--regression", action="store_true",
                    help="run the static-arm equivalence assert and exit")
    ap.add_argument("--smoke_users", type=int, default=0,
                    help="SMOKE ONLY: keep the first N test users. Output is stamped non-citable.")
    a = ap.parse_args()

    budgets = [int(x) for x in a.budgets.split(",")]
    specs = [s.strip() for s in a.selectors.split(",") if s.strip()]
    recs = RECS if not a.only else [r for r in ALL_RECS if r in set(a.only.split(","))]

    # EARLY name check, BEFORE the ML-25M load: a stored-order typo must cost seconds, not the
    # data load and certainly not four hours (the `fixedorder:greedy_concepts` scar).
    validate_specs(specs, None, budgets, log=logln)
    validate_paths(a, log=logln)
    D = load_arm_n(log=logln)
    if a.regression:
        regression_check(D, budgets=budgets, log=logln)
        return

    ni = D["n_items"]
    glob, bank = ST.build_orders(D["g_train"], D["train"], ni, log=logln)
    validate_specs(specs, glob, budgets, log=logln)      # STARTUP name check (HARD RULE: fail fast)
    smoke = a.smoke_users > 0
    if smoke:
        D = _smoke_slice(D, a.smoke_users, log=logln)
    n = D["te_tr"].shape[0]
    lvl = ST.level_lookup(D["g_te_tr"])

    built = build_recommenders(recs, D, arch=a.arch, snapshot=a.snapshot, log=logln,
                               dre_tag=a.dre_tag)
    # ---- extra golbandi_leaf rows at other shrinkages. The consumer's lambda is hardcoded to the
    # ARCHIVED 8.0 in build_recommenders; SNAP-5 measured that constant as the one at which the
    # interview degrades with depth, so a port that never varies it cannot tell a null from a
    # mis-set constant. Same class, same mode ("raw"), same score_arm -- only lambda differs.
    if a.golbandi_lam:
        from golbandi_leaf import GolbandiLeaf
        import run_interview_table as _RIT
        for v in [float(x) for x in a.golbandi_lam.split(",") if x.strip()]:
            nm = f"golbandi_leaf_lam{v:g}"
            built[nm] = ("raw", GolbandiLeaf(D["g_train"], ni, lam=v, log=logln))
            # it is the same consumer, so it inherits the same never-floored status
            _RIT.CONSUMES_UNKNOWNS = set(_RIT.CONSUMES_UNKNOWNS) | {nm}
            if nm not in recs:
                recs.append(nm)
    pop_row, prior = mostpop_row(D, log=logln)

    lbl = a.label or a.arch
    outp = os.path.join(OUT, f"{'SMOKE_' if smoke else ''}adaptive_table_{lbl}.json")
    vecp = outp.replace(".json", "_per_user.npz")
    ckdir = outp.replace(".json", "_interviews")
    lockp = acquire_run_lock(outp, log=logln)
    contract = results_contract(a, specs, budgets, recs, n)
    max_states = int(a.conv_max_states) or None
    vecs = {}
    results = {"mostpop": {"_prior": prior}}
    meta = {"selectors": specs, "budgets": budgets, "consumers": recs, "arch": a.arch,
            "snapshot": os.path.basename(a.snapshot), "no_floor": bool(a.no_floor),
            "n_test_users": n, "ucb_alpha": a.ucb_alpha, "ts_seed": a.ts_seed,
            "eddi_cand_topm": a.eddi_topm, "conv_max_states": max_states,
            "golbandi": {"depth": a.golbandi_depth, "xi": a.golbandi_xi,
                         "min_child": a.golbandi_min_child, "leaf_lam": a.golbandi_lam,
                         "tree_cache": a.golbandi_tree_cache},
            "_contract": contract}
    if smoke:
        meta["SMOKE_NOT_CITABLE"] = True
        meta["smoke_users"] = a.smoke_users
    if a.eddi_topm:
        meta["CANDIDATE_POOL_TRUNCATED"] = True

    # ---------------- RESUME (HARD RULE 11a/b) -------------------------------------------------
    # A scored cell already in the output file is not recomputed, and the interview behind it is not
    # recomputed either. Resuming is allowed ONLY into a file stamped with the identical contract;
    # anything else refuses, because mixing two information contracts into one table is the exact
    # failure `surprise-means-debug-the-harness` records.
    if os.path.exists(outp) and not a.no_resume:
        old = json.load(open(outp))
        if old.get("meta", {}).get("_contract") != contract:
            raise SystemExit(
                f"[adaptive] REFUSING TO RESUME {outp}: its contract differs from this run's.\n"
                f"    banked = {old.get('meta', {}).get('_contract')}\n    wanted = {contract}\n"
                f"  Use a fresh --label, or --no_resume to overwrite deliberately.")
        results = old.get("results", results)
        results.setdefault("mostpop", {"_prior": prior})
        n_done = sum(len(kk) for nm, sp in results.items() if nm != "mostpop"
                     for kk in sp.values() if isinstance(kk, dict))
        logln(f"[adaptive] RESUMING into {outp}: {n_done} scored cell(s) already banked")
        if a.per_user and os.path.exists(vecp):
            vecs = {kk: v for kk, v in np.load(vecp).items()}
            logln(f"[adaptive] resumed {len(vecs)} per-user vector(s) from {os.path.basename(vecp)}")
    logln(f"[adaptive] {len(recs)} consumers x {len(specs)} selectors x {len(budgets)} budgets "
          f"over {n} users; floor={'OFF' if a.no_floor else 'ON'}; "
          f"interview checkpoints -> {ckdir}")

    def _flush():
        _write_atomic(outp, {"meta": meta, "results": results}, indent=2)

    cache = {}
    for spec in specs:
        for k in budgets:
            # THE UNIT OF LOSS IS ONE (selector, budget, consumer) CELL. Everything below -- the
            # interview and every consumer's score -- is written to disk as it completes, so a crash
            # costs the turn in flight, not the run (HARD RULE 11a).
            want = [nm for nm in recs if not (nm.startswith("golbandi_leaf") and k > MAX_DEPTH)]
            todo = [nm for nm in want
                    if f"k{k}" not in results.get(nm, {}).get(spec, {})]
            if len(want) < len(recs):
                logln(f"[adaptive] SKIP golbandi_leaf k={k} > {MAX_DEPTH} (3^k leaf collapse)")
            if not todo:
                logln(f"[adaptive] {spec:16s} k={k:2d}: all {len(want)} consumer cell(s) already "
                      f"banked -- neither the interview nor the scoring is repeated")
                continue
            t_int = time.time()
            sel = build_selector(spec, D, glob, bank, k, a, cache, log=logln)
            conc = None
            ex = {}
            if asks_concepts(spec):
                import conversational_bandit_selectors as CB
                stk = sel.stack
                conc = {"n_concepts": stk["n_concepts"],
                        "answer": (lambda u, c, _s=stk: CB.concept_answer(_s, u, c))}
                # PER-SELECTOR, never one value for the run. Arms legitimately differ: ConUCB pins
                # uniform_member_mean because its Eq. 4 DEFINES the key-term feature that way, while
                # the ConLinUCB arms honour --keyterm_repr. On 2026-08-09 a mixed run wrote a single
                # meta["keyterm_repr"] and so MISDESCRIBED the arm that had honoured the flag. The
                # per-cell copy below (cell["keyterm_repr"]) is the authoritative record.
                # A STORED ORDER SELECTS NOTHING, so no key-term FEATURE is ever consulted and
                # `keyterm_repr` is not a property of the cell -- recording one would invite exactly
                # the misreading the 2026-08-09 mixed run produced. Say so explicitly instead.
                meta.setdefault("keyterm_repr_by_selector", {})[spec] = (
                    "n/a (stored order: no key-term features are used)"
                    if spec.startswith("fixedorder:") else stk.get("_repr"))
                meta["n_concepts"] = int(stk["n_concepts"])
                meta["concept_answer_convention"] = (
                    "population neg-cap (apply_neg_cap=True)" if stk.get("apply_neg_cap", True)
                    else "CERTIFIED-ERA (apply_neg_cap=False + cap_negatives over revealed)")
                # n_concepts only: the cache CAP is deliberately NOT in the contract, because eviction
                # changes what is recomputed and never which question is asked.
                ex = {"n_concepts": int(stk["n_concepts"])}
            # ':' is legal in a selector spec (`treestatic:entropy0`, `fixedorder:greedy_items`) and
            # ILLEGAL in a Windows filename -- os.replace fails with WinError 87 at the first
            # checkpoint write, i.e. the persistence rule silently turns into a crash for exactly the
            # comparator arms. The spec itself still goes in the checkpoint CONTRACT, so sanitising
            # the filename cannot make two different specs share a sidecar.
            ckpt = os.path.join(ckdir, f"{spec.replace(':', '__')}_k{k}.json")
            if a.no_resume and os.path.exists(ckpt):
                os.remove(ckpt)
            asked, answered, info = run_interview(sel, k, n, ni, lvl,
                                                  ctx={"D": D, "glob": glob, "bank": bank},
                                                  sel_batch=a.sel_batch, log=logln, conc=conc,
                                                  ckpt_path=ckpt,
                                                  ckpt_contract=interview_contract(
                                                      a, spec, k, n, ni, extra=ex))
            info["interview_minutes"] = round((time.time() - t_int) / 60.0, 2)
            if conc is not None:
                info["_stack"] = sel.stack
            logln(f"[adaptive] INTERVIEW {spec:16s} k={k:2d} done in "
                  f"{info['interview_minutes']:.2f}m; answered={info['answer_rate'][-1]:.2f}/{k}; "
                  f"distinct ask-prefixes={info['distinct_prefixes'][-1]}; "
                  f"cache={info.get('cache')}")
            # The selector is the memory hog (its per-state caches); nothing below needs it, and the
            # NEXT cell builds a fresh one, so drop it before the scoring pass allocates.
            del sel
            for name in todo:
                mode, obj = built[name]
                results.setdefault(name, {}).setdefault(spec, {})
                ts = time.time()
                s_name, s_mode, s_obj = name, mode, obj
                # DRE's own recommender is BUDGET-SPECIFIC (one trained decoder per k, over that k's
                # own seed columns), and build_recommenders runs once per process, before this loop.
                # This is the only hook it needs; it is inert for every other consumer.
                if hasattr(s_obj, "set_budget"):
                    s_obj.set_budget(k)
                # THE CONCEPT-ANSWER CONSUMER. A conversational arm may have spent turns on key-terms;
                # the item-only scoring path cannot hear them, so scoring it that way would measure our
                # harness, not the method. On the 'ours' row we fold the concept answers through the
                # CERTIFIED concept channel on top of the certified tower, and hand score_arm a plain
                # predict_fn -- score_arm itself is NOT modified (for any name != "ours" it just calls
                # obj(X)), so the floor rule, the pool and the evaluate() cursor stay literally the
                # same code the static path uses.
                cs = info.get("_conc_state")
                n_ct = sum(len(t) for t in cs["tokens"]) if cs else 0
                if name == "ours" and n_ct and not a.no_concept_consumer:
                    import conversational_bandit_selectors as CB
                    from concept_fold import ConceptFoldNet
                    import torch
                    stk = info["_stack"]
                    if "cfnet" not in cache:
                        cblob = torch.load(a.concept_ckpt, map_location="cpu")
                        sd = cblob["net"]
                        # Shape-inferred, not assumed: the cv3 checkpoints carry no `tags`/`hidden`
                        # key, so the concept vocabulary size and the hidden width are read off the
                        # state dict. The vocabulary-drift check then becomes an equality on C, which
                        # is the thing that can actually silently misalign (a concept id folded here
                        # must mean the same genome tag the selector asked about).
                        _nc, _d = tuple(sd["emb.weight"].shape)
                        _h = int(sd["gate.0.weight"].shape[0])
                        assert _nc == stk["n_concepts"], (
                            f"CONCEPT VOCABULARY DRIFT: checkpoint has {_nc} concepts, the selector's "
                            f"genome bank has {stk['n_concepts']}; the ids do not mean the same thing")
                        assert _d == obj[0].d_out, "concept net latent width != tower latent width"
                        _net = ConceptFoldNet(_nc, d=_d, h=_h,
                                              use_refusal=cblob.get("use_refusal", False))
                        _net.load_state_dict(sd); _net.eval()
                        cache["cfnet"] = _net
                        logln(f"[adaptive] concept consumer = {os.path.basename(a.concept_ckpt)} "
                              f"(ep{cblob.get('epoch','?')}, C={_nc}, h={_h}, use_refusal="
                              f"{cblob.get('use_refusal', False)}, snapshot="
                              f"{cblob.get('snapshot','?')})")
                    e_, W_, b_ = obj
                    Lc = answers_to_csr(answered, n, ni, mode, asked=asked)
                    toks = cs["tokens"]
                    if not int(a.concept_neg_cap):
                        # CERTIFIED-ERA convention: the negative-mass cap is applied over the
                        # REVEALED list at fold time, not over all 1,031 concepts at answer time.
                        # This is the second half of --concept_neg_cap 0 and must never be omitted:
                        # apply_neg_cap=False alone is an UNCAPPED answer, which is a third
                        # convention nobody trained or measured under. Identical to
                        # convbandit_greedy_rescore.py's NO_NEG_CAP path, deliberately.
                        from signed_answers import cap_negatives
                        toks = [[(c, v2, r) for (c, _v, r), v2 in
                                 zip(t, cap_negatives([x[1] for x in t]))] for t in toks]
                    s_obj = CB.make_concept_aware_predict(e_, W_, b_, Lc, toks, cache["cfnet"])
                    s_name, s_mode = "ours_conc", mode
                # global_asked is the SHARED ITEM ask-list, and it is shared only if the full ACTION
                # lists agree (a key-term turn is part of the prefix that decides that).
                ga = asked[0] if common_asked(info.get("_acts", asked)) is not None else None
                cell = score_arm(s_name, s_mode, s_obj, asked, answered, D, pop_row,
                                 no_floor=a.no_floor, global_asked=ga, per_user=a.per_user)
                if a.per_user:
                    # Straight to an npz sidecar, written after EVERY cell (HARD RULE 11a): 10k
                    # float64 per cell would bloat the JSON, and a run that dies must not take the
                    # only material a paired bootstrap can be computed from with it.
                    vecs[f"{name}|{spec}|k{k}|full"] = np.asarray(cell.pop("full_per_user"))
                    vecs[f"{name}|{spec}|k{k}|tail"] = np.asarray(cell.pop("tail_per_user"))
                    _savez_atomic(vecp, **vecs)
                cell["concept_tokens"] = n_ct
                cell["concept_consumer"] = bool(s_name == "ours_conc")
                cell["score_minutes"] = round((time.time() - ts) / 60.0, 2)
                cell["interview_minutes"] = info["interview_minutes"]
                cell["distinct_prefixes"] = info["distinct_prefixes"][-1]
                # Provenance that must survive being read out of context: the key-term
                # representation is a property of the SELECTOR, not of the command line, and a run
                # may legitimately mix them (see the note where meta is written). Copy it from the
                # selector's own stats onto every cell so no cell can be quoted without it.
                if "keyterm_repr" in info:
                    cell["keyterm_repr"] = info["keyterm_repr"]
                # Provenance for a cell whose interview was stitched across a crash, plus the realised
                # cache pressure, so a cap set too low is readable off the table (HARD RULE 11b/c).
                for key in ("resumed_from_turn", "cache", "max_states", "width_dtype",
                            "keyterm_asks_per_turn"):
                    if key in info:
                        cell[key] = info[key]
                # SELECTOR DIAGNOSTICS INTO THE CELL, not just the log. For golbandi_tree these are
                # the numbers that decide whether the row is the method at all: a user who reaches a
                # LEAF before the budget is spent is asked the next unasked popularity item instead,
                # so an arm with a high fallback rate is part popularity and must be read as such.
                for key in ("fallback_asks", "total_asks", "fallback_rate", "tree_nodes",
                            "tree_depth", "leaf_reached_depth_mean"):
                    if key in info:
                        cell[key] = info[key]
                # The DRE duplicate-seed defect is a measured property of the published method and
                # must be readable off EVERY cell, not just the training log. Same for IGCN's
                # `igcn_personalised_rate`: an IGCN row at rate 0 is the global-IG STATIC order, and
                # that has to be readable off the table rather than inferred from the budget.
                # `bald_`/`aopt_` join them for the same reason: BALD's reduction to a pure variance
                # rule and A-optimal's value-free (near-static) character are properties a reader must
                # not have to infer from the module docstring, so each cell states them itself.
                for _dk, _dv in info.items():
                    if _dk.startswith(("dre_", "igcn_", "bald_", "aopt_")):
                        cell[_dk] = _dv
                results[name][spec][f"k{k}"] = cell
                logln(f"[adaptive] {name:14s} {spec:16s} k={k:2d} "
                      f"answered={cell['answered']:4.1f}/{k} floored={cell['floored']:5d} "
                      f"full={cell['full']:.4f} tail={cell['tail']:.4f} "
                      f"(interview {cell['interview_minutes']:.2f}m + score "
                      f"{cell['score_minutes']:.2f}m)")
                _flush()                      # HARD RULE 11a: the cell is on disk before the next one starts
            del asked, answered, info
    try:
        os.remove(lockp)
    except OSError:
        pass
    logln(f"[adaptive] done -> {outp}")


if __name__ == "__main__":
    main()
