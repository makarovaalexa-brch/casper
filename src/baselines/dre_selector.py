r"""dre_selector.py -- DRE (Kweon, Kang, Hwang & Yu, WWW 2020) as a Selector on OUR harness, plus
DRE's own recommender as a CONSUMER.

DRE learns a FIXED length-k seed set end to end (Gumbel-Softmax over the item axis), so on our
harness it is a STATIC selector: it ignores `answered` entirely and hands every user the same
question at every turn. That is the point -- it is the peer-reviewed learned-static bar, the row that
makes "we compared against the strongest published static method" true rather than a comparison
against our own greedy order. Native snap: docs/results/SNAP3_DRE_NATIVE.md (PASS).

NOTHING IS TRAINED HERE. The seed set and the retrained decoder are produced offline, one artifact
per budget, by `scripts/baselines/dre_train_ours.py`, and this file only LOADS them
(`.cache/baselines/dre_ml25m_liang_k{K}.pt`). A budget with no artifact is a hard error, never a
silent fallback to some other order.

============================================================================================
THE DUPLICATE-SEED DEFECT, AND HOW THE HARNESS IS FORCED TO EXPRESS IT
============================================================================================
DRE takes a per-column argmax over `logpi` with NO constraint that the k columns pick distinct items,
so the same item can occupy two slots and one question is wasted. Measured in the native snap: it
fired in 1 of 3 runs at k=50 and cost ~0.015 NDCG@10 when it did.

Our harness ASSERTS that a selector never re-asks an action (a repeat burns a question for zero
information and silently shortens the interview), so a literal replay is impossible. Two policies,
both explicit, `waste` is the default:

  waste  (default, FAITHFUL)  ask the distinct seeds; fill each wasted turn with DRE's own
         LEAST-preferred catalogue item (the bottom of `argsort(max_j logpi[:,j])`). DRE's own
         recommender cannot see it -- its decoder reads only the seed columns -- so under cell P the
         turn is genuinely wasted, exactly as in the published method. Under cell X a foreign
         consumer does see it, but a bottom-ranked item is the weakest thing DRE's own model could
         have named, so this is the conservative filler rather than a repair.
  extend (DIAGNOSTIC, favours DRE)  fill each wasted turn with the next-best item by
         `max_j logpi[:,j]` that is not already selected. This is a REPAIR of the published method
         and must be labelled as such wherever it is reported.

Either way `stats()` reports `dre_duplicate_seeds` into the run JSON, so the defect appears in every
cell -- which is the requirement.

============================================================================================
LAUNCH RECIPE (one heavy job at a time; check free RAM and for a live twin first)
============================================================================================
  # 0. train the five seed sets -- ONE process, sequential, ~2.1 CPU-h per budget (~10.5 h total).
  #    Writes .cache/baselines/dre_ml25m_liang_k{1,2,4,8,16}.pt, one the moment it finishes.
  python scripts/baselines/dre_train_ours.py --budgets 1,2,4,8,16

  # P -- DRE seeds consumed by DRE'S OWN retrained decoder.
  python src/baselines/adaptive_harness.py --selectors dre --only dre \
      --budgets 1,2,4,8,16 --no_floor --label dreP_ownrec

  # X -- DRE seeds consumed by OUR certified tower. The cell Act 1 needs.
  python src/baselines/adaptive_harness.py --selectors dre --only ours \
      --arch i26 --snapshot models/i26c_ep31/t2i26c_ep31.pt --no_floor \
      --budgets 1,2,4,8,16 --label dreX_tower
"""
import os

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
CKDIR = os.path.join(_ROOT, ".cache", "baselines")

DUP_POLICIES = ("waste", "extend")


def ckpt_path(k, tag=""):
    return os.path.join(CKDIR, "dre_ml25m_liang_k%d%s.pt" % (int(k), tag))


def load_blob(k, tag="", log=print):
    p = ckpt_path(k, tag)
    if not os.path.exists(p):
        raise SystemExit(
            "DRE budget k=%d has no trained seed set at %s. DRE learns a DIFFERENT seed set per "
            "budget, so this cannot be substituted from another k. Train it first:\n"
            "  python scripts/baselines/dre_train_ours.py --budgets %d" % (k, p, k))
    b = torch.load(p, map_location="cpu")
    assert int(b["k"]) == int(k), "artifact %s carries k=%s" % (p, b["k"])
    if b.get("NOT_CITABLE"):
        log("[dre] *** %s is a SMOKE artifact (reduced epochs) -- NOT CITABLE ***"
            % os.path.basename(p))
    log("[dre] k=%d seed set from %s | distinct=%d duplicates=%d | trained on %s"
        % (k, os.path.basename(p), b["n_distinct_seeds"], b["n_duplicate_seeds"],
           b.get("trained_on", "?")))
    return b


def ask_order(blob, dup_policy="waste"):
    """The k ACTIONS the harness will ask, in order. Length k, all distinct (harness contract)."""
    assert dup_policy in DUP_POLICIES, "unknown dup_policy %r" % dup_policy
    k = int(blob["k"])
    order, seen = [], set()
    for i in blob["itemseed"]:
        i = int(i)
        if i not in seen:
            seen.add(i)
            order.append(i)
    n_wasted = k - len(order)
    if n_wasted:
        pool = ([int(i) for i in blob["pad_order"]] if dup_policy == "waste"
                else [int(i) for i in blob.get("extend_order", blob["pad_order"])])
        for i in pool:
            if len(order) >= k:
                break
            if i not in seen:
                seen.add(i)
                order.append(i)
    assert len(order) == k and len(set(order)) == k, "could not build %d distinct actions" % k
    return order, n_wasted


# ============================================================================ the Selector
class DRESelector:
    """Static: returns the turn-th trained seed to every user. Implements the harness contract
    (`reset`/`select`/`stats`) directly rather than subclassing, so this module imports nothing from
    adaptive_harness and can be unit-tested on its own."""

    def __init__(self, k, tag="", dup_policy="waste", log=print):
        self.k = int(k)
        self.tag = tag
        self.dup_policy = dup_policy
        self.blob = load_blob(self.k, tag=tag, log=log)
        self.order, self.n_wasted = ask_order(self.blob, dup_policy)
        self.name = "dre"
        if self.n_wasted:
            log("[dre] *** k=%d: %d of %d DRE columns are DUPLICATES. Policy '%s' fills the wasted "
                "turn(s); DRE's own decoder cannot see the filler, so the question stays wasted. ***"
                % (self.k, self.n_wasted, self.k, dup_policy))

    def reset(self, ctx):
        self.ctx = ctx
        assert int(ctx["q_max"]) == self.k, (
            "DRE artifact is for k=%d but the harness is running budget %s; DRE's seed set is "
            "budget-specific and a prefix of the k=16 set is NOT the k=2 set"
            % (self.k, ctx["q_max"]))
        ni = int(ctx["n_items"])
        assert int(self.blob["n_items"]) == ni, (
            "catalogue drift: artifact trained on %s items, harness has %d"
            % (self.blob["n_items"], ni))
        assert all(0 <= i < ni for i in self.order), "seed id out of catalogue range"

    def select(self, turn, users, asked, answered):
        return np.full(len(users), int(self.order[turn]), dtype=np.int64)

    def stats(self):
        return {"dre_duplicate_seeds": int(self.blob["n_duplicate_seeds"]),
                "dre_distinct_seeds": int(self.blob["n_distinct_seeds"]),
                "dre_wasted_turns": int(self.n_wasted),
                "dre_dup_policy": self.dup_policy,
                "dre_itemseed_raw": [int(i) for i in self.blob["itemseed"]],
                "dre_ask_order": [int(i) for i in self.order],
                "dre_smoke_artifact": bool(self.blob.get("NOT_CITABLE", False))}


# ============================================================================ DRE's own recommender
class DREConsumer:
    """DRE's own recommender: the decoder retrained with the seed set frozen (paper Sec 4.3).

    Contract: input mode 'likes' -- the harness hands it a (batch x n_items) binary CSR of the
    answered LIKES, i.e. exactly the >3.5 binarisation DRE trains on. It reads ONLY the seed columns
    (`X[:, itemseed]`, duplicates included, in the trained column order) and reconstructs the full
    catalogue. Everything outside the seed set is invisible to it, which is why a wasted turn stays
    wasted under cell P.

    The seed set and the decoder are BUDGET-SPECIFIC, and `build_recommenders` runs once per process,
    before the budget loop -- so the object is budget-aware: `set_budget(k)` swaps in the artifact for
    k and the harness calls it inside the budget loop. Lazily cached, one decoder per budget."""

    def __init__(self, tag="", log=print):
        self.tag = tag
        self.log = log
        self._cache = {}
        self.k = None

    def _build(self, k):
        import sys
        sys.path.insert(0, os.path.join(_ROOT, "scripts", "baselines"))
        from dre_native_snap import Decoder
        b = load_blob(k, tag=self.tag, log=self.log)
        net = Decoder([int(b["hidden"]), int(k)], int(b["n_items"]))
        net.load_state_dict(b["decoder"])
        net.eval()
        return {"net": net, "seed": [int(i) for i in b["itemseed"]],
                "dup": int(b["n_duplicate_seeds"])}

    def set_budget(self, k):
        k = int(k)
        if k not in self._cache:
            self._cache[k] = self._build(k)
            self.log("[dre] consumer = DRE's own retrained decoder for k=%d (dup_seeds=%d)"
                     % (k, self._cache[k]["dup"]))
        self.k = k
        return self

    def __call__(self, X):
        assert self.k is not None, "DREConsumer.set_budget(k) must be called before scoring"
        c = self._cache[self.k]
        z = np.asarray(X[:, c["seed"]].todense(), dtype=np.float32)
        with torch.no_grad():
            return c["net"](torch.from_numpy(z)).numpy()
