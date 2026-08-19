r"""Canonical routing of a fitted ternary tree over a cohort -- SHARED by m4_compare (validation),
m4_distill_gen (train trajectories) and M5 (the test look), so all three descend identically.

THE TREE'S OWN DEFINITION OF AN INTERVIEW (ndcg_tree.py, node construction)
---------------------------------------------------------------------------
    own = node["path"] + node["steps"]

`path` holds the ancestors' splitters -- and, for a seeded teacher, the FREE ANCHOR as its first
element. `steps` holds everything the node itself contributes, beginning with the node's own first
pick. For a BELOW-FLOOR child the builder deliberately sets `path` to the PARENT's path (excluding
the parent's splitter) and `steps` to the PARENT's tail, whose first element IS that splitter. So the
splitter appears exactly once, and `len(path) + len(steps) == total_L + n_seed` is asserted at build
time for every leaf.

⚠ DEFECT FIXED 2026-08-19, and it invalidated numbers. The previous router accumulated splitters
while descending and THEN appended the leaf's `steps`. For a below-floor child that re-supplies the
splitter, so those users were asked one question twice and the interview delivered only 14 distinct
questions out of 15. 87 of the cold tree's 164 nodes are below-floor children; 18.5-45.3% of users
per tree were affected. It was NOT a property of the trees -- checked directly, ZERO leaves in any of
the seven trees contain a duplicate in their own `path + steps`. Routing now emits the tree's own
`path + steps` and never reconstructs it, which is why the bug cannot recur here.

THE CONTINUATION RULE, for users the tree has no branch for
------------------------------------------------------------
A tree fitted on 10,000 users has branches its cohort never populated (`dialogue` splits its root on
`pg-13`, answer rate 1.00 on validation, so the `noanswer` branch has no child and no fitted tail).
A held-out user routed there receives, for the remainder of the budget: the nearest ANCESTOR's fitted
continuation, then the global static order --- the unconditioned incumbent --- skipping questions
already asked. `n_fallback` is returned so the escape rate is REPORTED, not silently absorbed, and
`fb_from[user]` marks where the fallback took over so a DISTILLATION target can mask those positions
(their label is this module's rule, not the teacher's decision). EVALUATION never masks: a scored
interview must be complete for every user, and dropping the users a tree cannot route would select
the cohort on an outcome.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Q = Tuple[int, int]


def _qs(v) -> List[Q]:
    return [] if not v else [(int(a), int(b)) for a, b in v]


def route(E, rows, nodes: Dict[str, dict], total_L: int,
          static: Sequence[Q] | None = None, mask=()) -> Tuple[List[List[Q]], dict]:
    """Return (per-user ASKED question lists of length total_L, diagnostics).

    `mask` are questions that must never be ASKED -- folded evidence, which the user has already
    volunteered. Either a single iterable applied to every user (a seeded anchor, which sits in
    `path` as the free seed), or a dict {row: iterable} when each user volunteered something
    different. ⚠ A PER-USER opener must NOT be passed as one flattened set: that masks every user's
    evidence out of every other user's interview and silently truncates them all.

    Masking removes a question from what is ASKED; it does not change BRANCHING. If the tree splits
    on something the user already volunteered, the answer is known, so the split is still valid and
    free -- it simply is not re-asked.
    """
    from ndcg_tree import branch_of

    per_user = isinstance(mask, dict)
    if per_user:
        banned_of = {int(r): {tuple(q) for q in v} for r, v in mask.items()}
        default_banned = set()
    else:
        default_banned = {tuple(q) for q in mask}
        banned_of = {}

    def ban(r):
        return banned_of.get(int(r), default_banned) if per_user else default_banned
    stat_all = _qs(static)
    seq: Dict[int, List[Q]] = {}
    diag = {"n_fallback": 0, "sites": {}, "fb_from": {}}

    def own_of(nd, r) -> List[Q]:
        """The tree's own interview for this user at this leaf, minus what they already volunteered."""
        bn = ban(r)
        return [q for q in (_qs(nd.get("path")) + _qs(nd.get("steps"))) if q not in bn]

    def top_up(rs: List[int], chain: List[dict], site: str) -> None:
        raw: List[List[Q]] = []
        for nd in reversed(chain):
            for key in ("tail", "steps"):
                s = _qs(nd.get(key))
                if len(s) > 1:
                    raw.append(s)
        if stat_all:
            raw.append(list(stat_all))
        diag["n_fallback"] += len(rs)
        diag["sites"][site] = diag["sites"].get(site, 0) + len(rs)
        for r in rs:
            cur = seq[r]
            diag["fb_from"].setdefault(r, len(cur))
            have = set(cur)
            bn = ban(r)
            for s in raw:
                for q in s:
                    if len(cur) >= total_L:
                        break
                    if q not in have and q not in bn:
                        cur.append(q)
                        have.add(q)
                if len(cur) >= total_L:
                    break

    def rec(nid: str, rs: List[int], chain: List[dict]) -> None:
        if not rs:
            return
        nd = nodes.get(nid)
        assert nd is not None, f"node {nid} missing"
        chain = chain + [nd]
        if not nd.get("split"):
            short = []
            for r in rs:
                seq[r] = own_of(nd, r)            # THE TREE'S OWN SEQUENCE -- not reconstructed
                if len(seq[r]) < total_L:
                    short.append(r)
            if short:
                top_up(short, chain, f"{nid}:leaf-short")
            return
        lab, _ar, _nz = branch_of(E, rs, (int(nd["splitter"][0]), int(nd["splitter"][1])))
        kids: Dict[str, List[int]] = {"like": [], "dislike": [], "noanswer": []}
        for r, l in zip(rs, lab):
            kids[str(l)].append(int(r))
        for br in ("like", "dislike", "noanswer"):
            if not kids[br]:
                continue
            child = f"{nid}.{br}"
            if child in nodes:
                rec(child, kids[br], chain)
            else:
                # a branch this tree's cohort never populated: it has no child and no fitted tail.
                # the parent's own continuation is `path + tail`, exactly as the builder would have
                # recorded for a below-floor child.
                raw_own = _qs(nd.get("path")) + _qs(nd.get("tail") or nd.get("steps"))
                short = []
                for r in kids[br]:
                    bn = ban(r)
                    seq[r] = [q for q in raw_own if q not in bn]
                    if len(seq[r]) < total_L:
                        short.append(r)
                if short:
                    top_up(short, chain, f"{nid}.{br}:unfitted")

    rec("root", [int(r) for r in rows], [])
    out = []
    for r in rows:
        s = seq[int(r)]
        assert len(s) >= total_L, f"user {r}: only {len(s)} questions after fallback, need {total_L}"
        s = s[:total_L]
        assert len(set(s)) == total_L, (
            f"user {r}: routed interview asks a question twice -- {s}")
        out.append(s)
    return out, diag
