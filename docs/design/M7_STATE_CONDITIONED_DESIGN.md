# M7 — one universal policy, conditioned on state

**Design recorded 2026-08-28, author's ruling.** Supersedes the cohort/cluster routing work in
`PAPER_B_FINAL_AUDIT_2026-08-25.md` §2–3. Everything here is fitted on TRAIN and reported on
VALIDATION. The test look stays unspent.

---

## THE GOAL, in the author's words

> *"I want one universal policy. Conditioned on a state it offers an optimal interview."*

---

## WHY THE PREVIOUS DESIGN COULD NOT DELIVER IT

Teachers were fitted per **endorsement cohort** — users who behaviourally endorse a concept, computed
from their **full rating profile**. The policy then had to recover that label from a folded opener,
which is a lossy view of the same profile.

**Measured, and this is the whole reason for the redesign:** a linear probe on the opener state
recovers the cohort at **0.371** on the worst cohort (`marvel`); the trained policy achieves
**0.364**. The policy is at the **information ceiling, not the optimisation ceiling**. Two failed
fixes confirmed it — bank rebalancing moved mean routing 0.553 → 0.555 (confusion relocated, value
fell), and anchor-token encoding is inadmissible because the cohort label is unavailable at
cold-start.

Consequence: `marvel` is 15% of pooled rows and 40% of the policy's shortfall, the only cohort where
the policy scores below the static order.

⚠ The cohort teachers themselves are GOOD — the value matrix shows each cohort's own tree winning on
its own users. The defect is not the teachers. **It is that the grouping variable is invisible at
question time.**

---

## THE DESIGN

**Group by the STATE, not by the person.**

Two users who arrive at the same state are, by definition, indistinguishable to the system — so the
best next question is the same for both. The optimal question is a function of the state. Grouping by
a hidden user property means fitting a tree over users who share a *label* but present different
states; grouping by state means fitting over users who share the thing the decision depends on.

This is also the only grouping the router can recover **by construction**, because it is defined in
the space the policy reads.

### The circularity that had to be cut first

Openers were group-dependent:

```python
mine = [i for i, v in lvl[r].items() if v >= like_min_level and int(i) in tab["pool"]]
myg  = [c for c in gq if endorse[r, c]]
```

The films are the user's OWN liked films, but filtered to the group's diagnostic pool. So an opener
needs a group, and clustering opener states needs openers — circular.

**Cut: group-free openers.** Draw from the user's own liked items directly, no pool filter, matched
to the same corpus targets (film 0.6303 · genre 0.3979 · tone 0.2135 · coherence 0.2243 ·
popularity p93.3). This is *more* faithful, not less — a real person names films they liked, not
films from a diagnostic list we computed — and it closes a reviewer objection Fable raised: that the
teacher gain might be the opener generator leaking profile information, since cohorts and openers
were derived from the same profile.

### Order of work

| # | step | cost | what it decides |
|---|---|---|---|
| 1 | group-free openers, corpus-matched | ~30 m | removes the circularity |
| 2 | fold → opener states for train users | ~30 m | the space everything else lives in |
| 3 | **ONE tree at those states (k=1)** | ~3 h | **the control, and possibly the whole answer** |
| 4 | *only if 3 leaves a gap:* cluster opener states, one tree per region | ~3 h each | how much grouping is worth |
| 5 | distil → one universal policy | ~2 h | the deliverable |

**Step 3 is a strict special case of step 4 with k=1**, so it is not a detour — it is the first point
on the same curve. If one tree at opener states captures most of the cohort forest's `+0.00760`, the
universal policy needs **no routing at all** and the whole separability problem is void.

---

## WHAT THIS COSTS, STATED PLAINLY

The six cohort teachers were fitted at **group-filtered** opener states. Their result stands for that
setup but is not the arm that ships. That is six fits' worth of work superseded. The compensation is
that every arbitrary choice disappears: no `k` chosen before it is shown necessary, no era
projection, no cohort vocabulary, no opener/group circularity.

---

## WHAT SURVIVES UNCHANGED

- **The teacher chain**, pooled 5,406 rows / 3,980 users, K=15:
  `static 0.28731 < cold 0.29163 < opener 0.29923`; `opener−cold +0.00760 [+0.00367,+0.01165]`,
  `cold−static +0.00432 [+0.00039,+0.00818]`, `opener−static +0.01192 [+0.00770,+0.01597]`.
  This is the paper, and it does not depend on the policy working.
- **Cold beats static at cold states**: `+0.00892 [+0.00600,+0.01177]` on validation, corroborated by
  `+0.01035 [+0.00749,+0.01328]` on test (earlier tree).
- **Do-no-harm**: the distilled policy is `−0.00008 [−0.00028,+0.00010]` against the cold tree on the
  6,020 users who volunteer nothing.
- **Fidelity**: the policy reproduces whichever tree it routes to within ±0.0023 — an accurate
  imitator, never improvising.
- The **corpus-matched opener methodology** and the Reddit marginals themselves.

---

## CLAIMS THIS DESIGN MAY AND MAY NOT MAKE

**May:** *fit the interview where you mean to serve it.* One policy, conditioned on state, offering
the measured-greedy order fitted for that state.

**May not**, per Fable and the banked record:
- ❌ "reproduces the **optimal** question order" — the trees are measured-greedy, not optimal, and
  per-user optimal labels are clairvoyant (`perusor-labels-are-clairvoyant`).
- ❌ "more generalisable **in value** than trees" — a distilled policy cannot exceed its teachers at
  matched states. Generalisation can only mean **coverage and extrapolation**.
- ❌ any framing where the policy "beats the incumbent" unless `policy − cold` on the deployment mix
  clears zero.

---

## THE DISTINCTIVE EVIDENCE STILL TO EARN

Per Fable, a *primary* contribution needs a behaviour a forest structurally cannot produce: users
whose volunteered film was **excluded from every training opener** (the pre-registered half-B split),
where a tree forest has no index entry at all, and the policy still delivers teacher-level value with
a CI clearing zero. The half-A/half-B machinery already exists and carries over unchanged.

---

## OPEN RISKS

1. **Region teachers may be weaker.** A state region can pool users of different underlying taste who
   happened to volunteer similar things. Fable's warning applies: *separable routing to teachers with
   no marginal value is a policy that routes perfectly to nothing.* Step 3 tests this before any `k`
   is chosen.
2. **Era may persist in opener states.** Reddit openers skew recent and popular. Check with the
   corrected test — **`sd` of the distinctive films' years alone**, not `|gap| AND sd`, which passed a
   cluster whose films were all from 1994 because 1994 sits at the global mean, under-reporting era
   structure by more than 2×.
3. **Validation is being read repeatedly.** Every step here reports on validation. The confirmatory
   test look must be budgeted before 2 Oct.
