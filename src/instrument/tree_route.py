r"""Canonical routing of a fitted ternary tree over a cohort -- SHARED by m4_compare (validation),
m4_distill_gen (train trajectories) and M5 (the test look), so all three descend identically.

WHY THIS MODULE EXISTS (defect found 2026-08-19). A tree fitted on 10,000 users has branches its
cohort never populated. `dialogue` splits its root on `pg-13`, whose answer rate on validation was
1.00, so the `noanswer` branch held ZERO users, no child node was created, and no tail was fitted
there. On train, users DO fail to answer it -- they route into a branch the tree has nothing to say
about, and the old per-file `route()` raised `AssertionError: 1 questions, need 15`.

THE CONTINUATION RULE, stated once. A user who lands on a branch outside the tree's fitted support
receives, for the remainder of the budget:
  1. the nearest ANCESTOR's fitted tail (the deepest ancestor whose tail was actually fitted), then
  2. if that still falls short, the global static order (policy-B combined) --- the unconditioned
     incumbent, i.e. exactly what a user gets when there is no evidence to condition on,
in both cases SKIPPING questions already asked on the path. This is a strict EXTENSION: on the
cohort a tree was fitted on, these branches are empty by construction, so every banked validation
number is bit-identical under this rule. It only ever fires off the fitting cohort.

`n_fallback` is returned so the escape rate can be REPORTED rather than silently absorbed.

⚠ The fitted path is reproduced VERBATIM, duplicates included: a question can legitimately recur
along a path once a below-floor child inherits its parent's tail, and the banked M4a validation
numbers were measured with those duplicates present (deduplicated at FOLD time, not at ask time).
De-duplication happens ONLY inside the fallback top-up, where re-asking would be incoherent.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Q = Tuple[int, int]


def route(E, rows, nodes: Dict[str, dict], total_L: int,
          static: Sequence[Q] | None = None, mask: Sequence[Q] = ()) -> Tuple[List[List[Q]], dict]:
    """Return (per-user ASKED question lists of length total_L, diagnostics).

    `static` is the global fallback order; `mask` are questions that must never be asked (e.g. a
    seeded anchor, which is folded evidence rather than a question).
    """
    from ndcg_tree import branch_of

    banned = {tuple(q) for q in mask}
    stat = [tuple(q) for q in (static or []) if tuple(q) not in banned]
    seq: Dict[int, List[Q]] = {int(r): [] for r in rows}
    diag = {"n_fallback": 0, "sites": {}}

    def steps_of(nd, key):
        v = nd.get(key)
        return [] if not v else [(int(a), int(b)) for a, b in v]

    def top_up(rs: List[int], chain: List[dict], site: str) -> None:
        """Fill each user in `rs` to total_L from the nearest fitted ancestor tail, then static."""
        # deepest-first ancestor tails, longest usable one wins
        sources: List[List[Q]] = []
        for nd in reversed(chain):
            for key in ("tail", "steps"):
                s = steps_of(nd, key)
                if len(s) > 1:
                    sources.append(s)
        if stat:
            sources.append(list(stat))
        diag["n_fallback"] += len(rs)
        diag["sites"][site] = diag["sites"].get(site, 0) + len(rs)
        for r in rs:
            cur = seq[int(r)]
            have = {tuple(q) for q in cur}
            for s in sources:
                for q in s:
                    if len(cur) >= total_L:
                        break
                    if q not in have and q not in banned:
                        cur.append(q); have.add(q)
                if len(cur) >= total_L:
                    break

    def rec(nid: str, rs: List[int], chain: List[dict]) -> None:
        if not rs:
            return
        nd = nodes.get(nid)
        assert nd is not None, f"node {nid} missing"
        chain = chain + [nd]
        if not nd.get("split"):
            steps = steps_of(nd, "steps")
            short = []
            for r in rs:
                s = seq[int(r)]
                s.extend(steps)
                if len(s) < total_L:
                    short.append(int(r))
            if short:
                top_up(short, chain, f"{nid}:leaf-short")
            return
        q = (int(nd["splitter"][0]), int(nd["splitter"][1]))
        for r in rs:
            seq[int(r)].append(q)
        lab, _ar, _nz = branch_of(E, rs, q)
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
                # below-floor OR never-populated child: inherit this node's tail, then fall back
                tail = steps_of(nd, "tail") or steps_of(nd, "steps")
                short = []
                for r in kids[br]:
                    s = seq[int(r)]
                    s.extend(tail[1:])
                    if len(s) < total_L:
                        short.append(int(r))
                if short:
                    top_up(short, chain, f"{nid}.{br}:unfitted")

    rec("root", [int(r) for r in rows], [])
    out = []
    for r in rows:
        s = seq[int(r)]
        assert len(s) >= total_L, f"user {r}: only {len(s)} questions after fallback, need {total_L}"
        out.append(s[:total_L])
    return out, diag
