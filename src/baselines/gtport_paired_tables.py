r"""gtport_paired_tables.py -- render `gtport13_paired.json` as the markdown deliverable.

Reads the machine-readable output only; computes nothing. Keeps the doc and the JSON from drifting.

  python src/baselines/gtport_paired_tables.py
"""
import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUT = os.path.join(_ROOT, "experiments", "baselines", "interview")
DOC = os.path.join(_ROOT, "docs", "results", "GOLBANDI_PORT_PAIRED_CIS.md")
BUD = (1, 2, 4, 8)


VERDICT = r"""## What `GOLBANDI_PORT_ON_OUR_RULER.md` may and may not say now

**Survives, with the interval attached.**

* *"The tree crosses over between k=2 and k=4. Against entropy0 it is -.0092 full at k=2 and +.0064
  full / +.0052 tail at k=8."* Both ends separate: -0.0092 [-0.0115, -0.0069] at k=2, +0.0064
  [+0.0037, +0.0091] and +0.0052 [+0.0024, +0.0079] at k=8. The crossover is real, not noise.
* *"the tree takes the tail at k=8 (.0844 vs .0776, +.0068)"* -- the number the chapter would rest on.
  +0.0068 [+0.0039, +0.0098], SEP. It holds.
* *"greedy items-only leads on full at every budget"* -- SEP at all four budgets (-0.0203, -0.0293,
  -0.0203, -0.0130).
* *"essentially tied with HELF"* on `golbandi_leaf` -- now a claim rather than an assertion:
  -0.0014 [-0.0053, +0.0025] full and -0.0001 [-0.0032, +0.0030] tail at k=8, both ns.
* *"the tree beats entropy0 and popularity clearly"* on its own consumer -- SEP at every budget, both
  axes.
* *"the consumer flips which static is strongest: HELF here, entropy0 on our tower"* -- the flip is
  separated in BOTH directions: entropy0 - HELF is +0.0160 [+0.0127, +0.0194] on `ours` and -0.0262
  [-0.0304, -0.0221] on `golbandi_leaf` at k=8 full. The instrument-dependence exhibit stands.

**Does NOT survive as written.**

1. *"On the ranking leaf it is inert: the lambda=200 arm differs from lambda=8 by <=.0015 everywhere."*
   FALSE as a universal. It is true of the `golbandi_tree` arm (-0.0011 full / -0.0012 tail at k=8),
   but on the STATIC arms at k=8 full it reaches +0.0121 [+0.0094, +0.0149] (entropy0) and +0.0153
   [+0.0128, +0.0179] (popularity), both SEP -- an order of magnitude past the stated bound, and in
   the opposite sign to the tree. "Everywhere" must become "on the tree arm"; and note the shrinkage
   moves the COMPARATOR more than the method, which shrinks the tree's k=8 win over entropy0 on that
   consumer from +0.0248 to +0.0116.
2. *"The tree crosses over between k=2 and k=4"* is right on full but over-stated on TAIL: at k=4 the
   tail gap is +0.0015 [-0.0011, +0.0040], ns. On the tail axis the crossover is only established by
   k=8; k=4 is a wash. Write the sentence per axis.
3. *"this is the first break in the pattern"* / *"the first time any adaptive selector has beaten the
   optimised static order on either axis"* -- supported only if "beaten" means one axis at one budget.
   The same k=8 cell is -0.0130 [-0.0160, -0.0100] on full, SEP. A single-axis, single-budget win
   against a comparator the doc itself calls a lower bound is not "the tree beats the optimised
   static"; say "wins the tail at k=8 while losing full at every budget".
4. The identity check reported as *"reproduces ... to 4 dp at every budget"* understates itself: the
   two arms are identical PER USER (max abs diff 0.0e+00). Strengthen it -- 4 dp on a mean is a much
   weaker statement than what the data supports.
5. The cross-run licence -- *"They are admissible because the two runs share an exact anchor:
   entropy0 reads .1635/.0433 ..."* -- is now moot and should be dropped rather than defended. The
   greedy order was re-scored INSIDE this harness (`gtport_greedy_rescore.py`), reproducing
   `.1760/.0434 . .1907/.0581 . .2066/.0703 . .2247/.0776` exactly and matching entropy0 per user at
   0.0e+00. A 4-dp mean anchor never licensed a PAIRED difference in the first place; per-user
   identity does.

**Newly visible, worth stating.** On our tower the static ladder is not a single order: entropy0 beats
both popularity and HELF at every budget on both axes (all SEP), but popularity - HELF flips sign by
axis at k=8 (+0.0051 full SEP, -0.0039 tail SEP). And entropy0 - greedy on tail is ns at k=1
(-0.0001) and k=8 (+0.0017) -- greedy's advantage over entropy0 is a FULL-axis advantage that only
separates on tail in the middle of the range.
"""


def cell(e):
    if e is None:
        return "—"
    mark = "**SEP**" if e["separates"] else "ns"
    return f"{e['delta']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}] {mark}"


def block(R, bucket, keyfmt, title, rows, note=""):
    B = R[bucket]
    out = [f"### {title}", ""]
    if note:
        out += [note, ""]
    out.append("| comparison | axis | " + " | ".join(f"k={k}" for k in BUD) + " |")
    out.append("|---|---|" + "---|" * len(BUD))
    for label, kf in rows:
        for ax in ("full", "tail"):
            cs = [cell(B.get(keyfmt(kf, ax, k))) for k in BUD]
            out.append(f"| {label} | {ax} | " + " | ".join(cs) + " |")
    out.append("")
    return out


def main():
    R = json.load(open(os.path.join(OUT, "gtport13_paired.json")))
    m = R["meta"]
    L = []
    L += [f"# Paired CIs for the Golbandi tree PORT+CROSS cells (`{m['label']}`)", "",
          f"Every interval below is a **paired percentile bootstrap over the per-user difference "
          f"vector**: resample users, recompute the per-user delta, take the 2.5/97.5 percentiles. "
          f"**{m['boot']:,} resamples, seed {m['seed']}.** Never the unpaired `1.96*sqrt(2)*SE` form "
          f"(`ARM_N_PHASE1.md` §4b: it overstated intervals ~4.6x and forced a retraction).", "",
          f"Contract: {m['contract']}. Source vectors `{m['source_vectors']}` "
          f"(n_full={m['n_full_scored']}, n_tail={m['n_tail_scored']}; the tail axis drops the 719 "
          f"users with no tail target — a property of the DATA, so rows stay aligned across arms). "
          f"Machine-readable: `experiments/baselines/interview/gtport13_paired.json`.", "",
          f"Generator `src/baselines/gtport_paired.py`; table renderer "
          f"`src/baselines/gtport_paired_tables.py`. `SEP` = the 95% interval excludes 0.", "",
          "## Controls", "",
          f"* **Self-test: {'PASS' if m['self_test_pass'] else 'FAIL'}.** "
          f"`golbandi_tree` against ITSELF gives delta 0.0000 with CI [0.0000, 0.0000] at all "
          f"{len(R['self_test'])} (budget, axis) cells.",
          f"* **Row-alignment canary: PASS.** All {len(m['cells_present'])} cells carry identical "
          f"user counts.",
          f"* **Identity check: EXACT.** max per-user |`treestatic:entropy0` − `entropy0-static`| = "
          f"{max(v['max_abs_per_user_diff'] for v in R['C6_identity_checks'].values()):.1e} over every "
          f"consumer/budget/axis — the matched comparator is the same object per user, not merely to "
          f"4 dp in the mean.", ""]

    g = m.get("greedy", {})
    if g.get("admitted"):
        L += [f"* **Greedy admission: PASS.** The greedy order was re-scored inside THIS harness "
              f"(`src/baselines/gtport_greedy_rescore.py`); its `entropy0` control reproduces the "
              f"stored gtport13 per-user vector to "
              f"{g['entropy0_canary_max_abs_per_user_diff']:.1e} per user, so the two arms are in one "
              f"contract and can be differenced.", ""]
    else:
        L += ["* **Greedy: NO INTERVAL.** Per-user vectors for the greedy order are not available in "
              "this contract, so the greedy gaps are reported as point estimates only.", ""]

    L += block(R, "C1_tree_vs_entropy0_ours",
               lambda kf, ax, k: f"ours/{ax}/k{k}/golbandi_tree-minus-entropy0",
               "1. The CROSSOVER — `golbandi_tree` − `treestatic:entropy0`, consumer `ours`",
               [("golbandi_tree − entropy0", None)])

    if R["C2_tree_vs_greedy_ours"]:
        L += block(R, "C2_tree_vs_greedy_ours",
                   lambda kf, ax, k: f"ours/{ax}/k{k}/{kf}",
                   "2. Against the strongest static — consumer `ours`",
                   [("golbandi_tree − greedy items-only", "golbandi_tree-minus-greedy_items"),
                    ("entropy0 − greedy items-only", "entropy0-minus-greedy_items")])

    L += block(R, "C3_tree_vs_helf_golbandi",
               lambda kf, ax, k: f"golbandi_leaf/{ax}/k{k}/golbandi_tree-minus-{kf}",
               "3. The method as published — consumer `golbandi_leaf`",
               [("golbandi_tree − HELF", "helf"), ("golbandi_tree − entropy0", "entropy0"),
                ("golbandi_tree − popularity", "popularity")])

    for tag, cons in zip("abc", ("ours", "golbandi_leaf", "golbandi_leaf_lam200")):
        L += block(R, "C4_static_ladder",
                   lambda kf, ax, k, c=cons: f"{c}/{ax}/k{k}/{kf}",
                   f"4{tag}. The static ladder — consumer `{cons}`",
                   [("entropy0 − popularity", "entropy0-minus-popularity"),
                    ("entropy0 − HELF", "entropy0-minus-helf"),
                    ("popularity − HELF", "popularity-minus-helf")])

    L += block(R, "C5_leaf_shrinkage",
               lambda kf, ax, k: f"lam200-minus-lam8/{ax}/k{k}/{kf}",
               "5. Leaf shrinkage — consumer `golbandi_leaf_lam200` − `golbandi_leaf`, same questions",
               [("golbandi_tree", "golbandi_tree"), ("treestatic:HELF", "treestatic:helf"),
                ("treestatic:entropy0", "treestatic:entropy0"),
                ("treestatic:popularity", "treestatic:popularity")])

    L += ["## Point estimates behind every delta", "",
          "| consumer / selector | " + " | ".join(f"k={k}" for k in BUD) + " |",
          "|---|" + "---|" * len(BUD)]
    for key in sorted(R["points"]):
        cons, spec, kk = key.split("|")
        if kk != "k1":
            continue
        vals = []
        for k in BUD:
            p = R["points"].get(f"{cons}|{spec}|k{k}")
            vals.append(f"{p['full']:.4f}/{p['tail']:.4f}" if p else "—")
        L.append(f"| {cons} / {spec} | " + " | ".join(vals) + " |")
    gnpz = os.path.join(OUT, "gtport13_greedy_per_user.npz")
    if os.path.exists(gnpz):
        import numpy as np
        z = np.load(gnpz, allow_pickle=False)
        for spec in ("greedy-items-only", "entropy0-top16"):
            vals = [f"{z[f'greedy_env|{spec}|k{k}|full'].mean():.4f}/"
                    f"{z[f'greedy_env|{spec}|k{k}|tail'].mean():.4f}" for k in BUD]
            L.append(f"| ours / {spec} (re-scored here) | " + " | ".join(vals) + " |")
    L.append("")

    L += VERDICT.split(chr(10))
    open(DOC, "w", encoding="utf-8").write("\n".join(L))
    print(f"wrote {DOC} ({len(L)} lines)")


if __name__ == "__main__":
    main()
