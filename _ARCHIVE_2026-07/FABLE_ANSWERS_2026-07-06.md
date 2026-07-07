# Fable's answers — 2026-07-06

Answers to `questions_for_fable.md` (Q1–Q5) + action plan. Deadline anchor: ECIR 2027, 2 Oct 2026 (~12.5 weeks).

---

## Q5 first — the real contribution (everything else follows from it)

### The single clearest true claim the evidence supports

> **What to ask a cold user is decided by the *channel*, not the *query*. Every practical elicitation
> channel — items, concepts, pairs, sliders, open recall, abstract continuous queries — sits somewhere
> in a three-axis space of ANSWERABILITY (can the user answer), FIDELITY (how noisy the answer),
> and BANDWIDTH (how much preference information one answer carries), and the winning elicitation
> design *flips* as you move through that space. We build the first instrumented testbed that measures
> all three axes and maps the flips.**

This is the **channel map**. It is a positive, novel, deployable contribution, and — crucially — every
"awkward fact" in the current five papers is *evidence for it*, not an admission against it:

- **C's inversion (§5.6)** is not a refutation of C — it is the map's centerpiece: the fidelity axis is
  load-bearing. Under slider-grade fidelity, fine continuous queries win; under the empirically-fitted
  ML-1M channel the ordering inverts and coarse answerable questions win. One figure, two regimes,
  a measured boundary. Nobody has this figure.
- **B's density result** (item-asking dead at scale, ~0.03% answerable; replicates across 4 datasets)
  is the answerability axis. It's the strongest replicated fact in the thesis.
- **D's bandwidth argument** (a named entity ≫ a scalar; on huge catalogues open recall is the *only*
  route to item-level bandwidth, because items are unaskable as probes but volunteerable) is the
  bandwidth axis. The framing lever is the knob that steers *where* on the popularity axis the
  volunteered bandwidth lands.
- **E's un-askability** is the proof that channels are *not interchangeable renderings* of one another —
  which is exactly why a channel map is needed at all. Without E, a reviewer says "just render the
  optimal query into words"; E shows you can't.
- **A** is the measurement apparatus that makes the axes measurable non-circularly (instrument-class
  determines visibility; protocol distortion; gate suite).
- **The heuristic-matches-learned results** (8 failed policy attempts, R2's static-schedule tie) become
  a *finding*, not a failure: **once you know where a channel sits on the map, the schedule is nearly
  static — the value is in choosing the channel, not in learned per-step adaptivity.** That is a
  deployable prescription a practitioner wants ("don't build the RL agent; pick the right channel
  ladder"), and it explains WHY the RL attempts tied: prior sims gave the agent nothing user-specific
  to discover.

### Where the coarse→granular ADAPTIVE claim gets its fuel (the make-or-break)

The honest current state: **adaptivity has never paid in your sims because your sims removed the reason
for adaptivity.** When answerability is a designer-authored rule (rated=answerable) and answers are
geometric, every user is exchangeable up to u*, and linear-Gaussian theory says optimal design is
non-adaptive. There was nothing per-user to *discover*, so static schedules tied. This is the reframe
the briefing hoped for, and the evidence genuinely supports the first half of it.

The second half — "put real per-user answerability heterogeneity back and adaptivity wins" — **does not
exist yet**. The LLM answerability judge is the experiment that supplies it. So say it plainly:

> **The LLM answerability experiment is THE make-or-break of the flagship, not a footnote.** If
> LLM-judged answerability shows exploitable per-user heterogeneity that tracks taste (the fuel gate),
> then coarse→granular adaptive interviewing has a real environment to win in, and the flagship's
> adaptive claim is earned. If the fuel gate shows flat heterogeneity, the flagship ships as the
> channel map with *static* coarse→granular schedules (still a real contribution) and the adaptive
> sentence is cut.

Both branches are publishable. Only one branch gets the full ambitious thesis.

### Novelty against prior art (closest work, delta in one line each)

- **GATE (Li et al. 2023, generative elicitation)** — compares open-ended vs yes/no question *formats*
  with LLM users; no fidelity axis, no answerability measurement, no downstream recommender, no regime
  boundary. *Delta: we decompose WHY formats win into three measured properties and show the ordering flips.*
- **PEBOL (Austin et al. 2024)** — item-anchored aspect BO over a 100-item shortlist; single channel,
  answerability implicit, no noise sweep. *Delta: multi-channel map + measured fidelity boundary + full-catalogue ruler.*
- **Christakopoulou et al. 2016 (Towards Conversational RecSys)** — item vs absolute questions, one
  dataset, no answerability/fidelity decomposition. *Delta: the axes, the boundary, the scale result.*
- **ConTS / UNICORN / Wolpertinger-style CRS** — fixed channel (attributes or items), policy-centric.
  *Delta: we show channel choice dominates policy choice, with the policy-tie evidence to prove it.*
- **Golbandi et al. 2011 (decision trees)** — item-only bootstrapping. *Delta: item channel is dead at
  modern catalogue scale (0.03% answerable); the map says what replaces it.*
- **Rashid et al. 2002/2008 (learning new user preferences)** — popularity/entropy item selection.
  *Delta: same, plus fidelity axis.*
- **EDDI (Ma et al. 2019)** — information-theoretic feature acquisition, noiseless oracle answers.
  *Delta: the fidelity boundary shows the noiseless assumption is exactly what decides the answer.*

Nobody has: (i) the answerability × fidelity × bandwidth decomposition, (ii) the measured inversion
boundary, (iii) an external (non-self-authored) answerability signal in a simulator, (iv) the
channel-beats-policy result. That's the novel core.

### Litmus-test abstract of the reframed flagship (overclaim-free but confident)

> Cold-start preference elicitation is usually framed as selecting the best next question within a fixed
> question format. We show that the prior decision — *which channel to ask through* — dominates. Using an
> instrumented testbed built on a state-of-the-art-tied recommender, we characterize every practical
> elicitation channel (item ratings, concept questions, pairwise comparisons, slider screens, and open
> free-recall, plus a purely theoretical continuous-query ceiling) along three measurable properties:
> **answerability** — whether a cold user can answer at all; **fidelity** — how noisy the answer is; and
> **bandwidth** — how much preference information one answer carries. The winning design flips as these
> properties change: under high-fidelity graded answers, fine-grained queries dominate; under an answer
> channel fitted to a million real ratings, the ordering inverts and coarse, broadly answerable questions
> win; and at industrial catalogue scale, where fewer than 0.1% of items are askable, open free-recall is
> the only route to the highest-bandwidth channel. Unlike prior simulators, whose answerability rules are
> authored by the same designers whose methods exploit them, we measure answerability with an external
> LLM judge [validated in a human study], and find that per-user answerability heterogeneity is what makes
> adaptive coarse-to-granular interviewing pay [conditional clause — pending fuel gate]. The result is a
> practical design map: start with broadly answerable coarse questions and escalate granularity as
> discovered answerability permits — the strategy a human interviewer uses, now with quantitative backing.

If the fuel gate fails, delete the bracketed clause and the last sentence's "as discovered answerability
permits" becomes "according to the channel map" — still a real, useful, new thing.

---

## Q1 — What to remove / demote / keep, per paper

**A (instrument).** KEEP: instrument-class-determines-visibility, protocol-distortion, oracle-privilege
ladder, gate suite, RecVAE rebuild. KILL: the §12.4 "question channel is strongest" capstone as a
*finding* — demote to "an upper-bound characterization of the channel under the stated geometric answer
model" and cross-reference the flagship's fidelity boundary as the reason it's an upper bound. FIX:
label "ties EASE" explicitly as full-profile warm-start; either run one elicitation result on RecVAE
non-circularly or say RecVAE certifies headroom only. DEMOTE: any delta that restates a training
objective — move to appendix with the caveat printed.

**B (answerability).** KEEP: the axis, the concept-necessity/density result (4 datasets), the
answerable-concepts-beat-items mechanism *under a stated answerability model*. KILL: the learned-policy
win (+0.012, sub-noise, training-seed-fragile, reverses on ML-25M, refuted on Goodreads) — you already
know this from the TRSEED patch; say "no learned endpoint win" once, cleanly. RELABEL: "answerable iff
rated" as a *lower-bound structural model*, one of two answerability models (the other = LLM judge);
rerun the item-vs-concept headline under both and report where it moves.

**C (continuous).** KILL the title and the "continuous beats discrete" headline. REFRAME the whole paper
around §5.6: the fidelity boundary IS the paper. New title candidate: *"The Answer-Fidelity Boundary:
When Fine Questions Beat Coarse Ones in Preference Elicitation."* KEEP: the mechanism studies, the
realization ladder, snap-loss — all explicitly "under the stated oracle answer model", with the
sign-reversal under noise presented as the point, not a limitation. RESTORE: the bot-play learned
answerer you removed — report it; suppressed evidence is the one thing worse than bad evidence.

**D (open recall).** KILL: "2 questions beat 8" with the privileged answerer, and every comparison where
open-recall folds profile tokens that baselines are denied. FIX: symmetric comparison (either both get
profile access or neither); report only non-privileged answerers (popweight/realistic) with CIs. KEEP:
the bandwidth argument, the only-route-at-scale argument, the framing lever (relabeled "under an LLM
recall model; human validation pending"). The realistic numbers (0.387/0.170) are still good — sell those.

**E (rendering).** MERGE into the flagship as the un-askability section (the adversarial review reached
the same verdict). KEEP: un-askability negative result + constructive escapes (sliders, pairs, open
recall — which is exactly the channel map's spine). KILL: standalone-paper framing, the agent placeholder,
the superseded −0.008 headline (use the canonical five-seed −0.010/−0.012). FLAG: the LLM axis-synthesis
step gets either a small evaluation (LLM-judge rated faithfulness, n≥100 queries) or the label "anecdotal."

**Cross-cutting (all five):** fix metric-depth cherry-picking (one declared depth per dataset, stated
once); add paired bootstrap CIs to every headline delta; add a multiplicity note (declare the headline
family, Holm-correct it, let the rest be exploratory); stop invoking linear-Gaussian theory
directionally-as-needed — state it once as the reason adaptivity needs nonlinearity/heterogeneity.

---

## Q2 — Swapping out the answerer, per channel

Principle: **answer VALUE comes from data-side ground truth (real ratings), never from the evaluated
model's latent space; FIDELITY is one swept knob (test-retest noise, Amatriain-calibrated);
ANSWERABILITY comes from two independent external models (LLM judge + structural popularity model),
and key results must hold under both.**

| Channel | Answer value (non-circular source) | Answerability | Caveats to print |
|---|---|---|---|
| **Item** | The user's real rating (held-out); fidelity = rating test-retest noise, σ swept around Amatriain's estimate | LLM judge (primary) + popularity-conditioned seen-probability (structural) | "rated" underestimates knowledge; LLM over-reads (upper bound); truth is between the two models |
| **Pair** | Which of two co-rated items the user rated higher (ground truth); fidelity = flip-probability from rating-gap (closer ratings → noisier) | Both items answerable per the judge | co-rated pairs are a biased subsample of all pairs |
| **Concept** | Aggregate of the user's real ratings over the concept's member items, where membership comes from tags/genres — DATA-side, never decoder geometry; fidelity knob on top | LLM judge; structural model = concept popularity/coverage | concept membership definition is a modeling choice; report sensitivity to tag- vs genre-membership |
| **Slider / graded** | Same as concept but graded (mean rating rescaled to [−1,1]) + fidelity noise; this is the ONLY place slider-grade fidelity is allowed, and it must be labeled optimistic | Same as concept | slider-grade fidelity in humans is unmeasured → human study measures it |
| **Open recall** | The named item is a real profile item; recall model = popularity-tilted sampling from the user's rated set (+ framing lever shifts the tilt); the fold uses the item's data-side identity | Trivially answerable (user chooses what to name); the REAL unknown is the recall distribution → LLM stand-in now, human validation later | name→factor resolution error is the deployment killer above ~10%; report the sensitivity curve |
| **Abstract continuous** | Oracle a = cos(u*, q) — KEPT, but only as an explicitly labeled theoretical ceiling ("not deployable; characterizes the information-theoretic limit of the channel space") | n/a (not askable — that's E's point) | this is the one channel where circularity is acknowledged as the design, not hidden as a result |

The one structural change this forces: **the belief encoder must fold data-side answer tokens (item ids
+ real ratings, concept ids + aggregate ratings), not latent projections.** You already have this
machinery (open-recall folds entity tokens). The continuous-projection fold survives only inside the
abstract-ceiling section.

---

## Q3 — The two costly validations

### (a) LLM experiment — run the fuel gate THIS WEEK

**Fuel gate (2–3 days, ~$30–60):**
- Sample ~300 ML-1M users spanning profile sizes; for each, give the LLM HALF the profile (random split,
  titles + ratings, no ids) — the held-out half is the clairvoyance check.
- Battery of ~60 questions spanning broad→niche: 20 concepts (popularity-tiered), 20 items
  (popularity-tiered, drawn from the HELD-OUT half + never-rated items), 20 pairs.
- LLM (GPT-4o-mini or Haiku; a second model family for robustness) answers per question: "could this
  user answer this? {yes/no/probably} + confidence", plus for items it believes answerable, a predicted
  rating (fidelity check for free).
- **Gate metrics:** (1) per-user heterogeneity — variance in answerability across users, per question,
  beyond what popularity explains; (2) taste-tracking — does a user's judged answerability over niche
  concepts correlate with their genre/tag distribution (the scandi-noir test); (3) sanity — LLM-judged ⊇
  rated (it should strictly dominate the lower bound), monotone in item popularity; (4) clairvoyance
  check — judged answerability on held-out-half items should beat never-rated items but NOT perfectly
  (if perfect, the profile is leaking; increase randomization / shrink the shown profile).
- **Decision rule:** heterogeneity present AND taste-tracking r > ~0.3 → fuel exists, build the adaptive
  agent. Flat → ship the static channel-map flagship, cut the adaptive clause.

**Main LLM experiment (if gate passes, ~2 weeks):** replace B's answerability rule with the LLM judge in
the interview loop (pre-computed judgments, not live calls: judge a fixed question bank × user grid
once, cache it). Rerun: (1) item-vs-concept headline under both answerability models; (2) the
coarse→granular adaptive agent vs the best static schedule, where the agent's only edge is discovering
per-user answerability in-conversation. What it CAN claim: "under two independently-derived
answerability models, X holds." What it CANNOT claim: human answerability — say "LLM proxy, human
validation in §Y" every time. Scale: ~300 users × ~200-question bank ≈ 60k judgments ≈ low hundreds of
dollars; cache everything.

### (b) Human study (~50 users) — validates the proxy, doesn't carry the thesis

**Submit the ethics application NOW** — university ethics turnaround (2–6 weeks) is the critical path,
not the study itself. Platform: Prolific, ~50 users, ~20 min, ~£8 each ≈ £400–500.

Design (each user):
1. **Seed profile**: rate 20–30 well-known movies (their "profile" for the LLM judge to read).
2. **Answerability battery**: ~40 questions spanning the channels (popularity-tiered items, concepts,
   pairs) with an explicit "can't answer" option. → primary endpoint: **calibration of LLM-judged
   answerability against human can't-answer rates** (per-question and per-user correlation).
3. **Fidelity probes**: re-ask ~8 questions at the end (test-retest reliability → your fidelity knob's σ
   gets an empirical anchor, including the unmeasured slider-grade fidelity).
4. **Open recall + framing lever**: "name a favourite" vs "name a hidden gem" (between-subjects) →
   popularity of named titles validates D's lever on humans.
5. (Optional, if time) fold their answers through the pipeline → one downstream NDCG point per condition,
   the "round-trip" the adversarial review asks for.

What it plugs in as: one section in the flagship ("the LLM judge is calibrated against human answers:
r=…, over/under-read=…") + one sentence in every other paper. It downgrades the circularity objection
from "no human anywhere" (fatal) to "human-validated proxy at moderate n" (normal science).

### ⚠ One methodological landmine to defuse in writing

MovieLens users' knowledge ≠ Prolific users' knowledge (2000s raters vs 2026 crowd). Frame the human
study as validating the JUDGE's model of answerability given a profile — not as validating ML-1M users
specifically. The LLM judge is the bridge: it reads a profile and predicts answerability; humans test
that mapping on THEIR profiles.

---

## Q4 — Packaging: 5 chapters → 3 papers

**Thesis spine (amended from the briefing's version):** *"Which channel you ask through — measured by
answerability, fidelity, and bandwidth — decides cold-start elicitation; coarse→granular interviewing is
the strategy the channel map implies, and it becomes ADAPTIVE exactly when per-user answerability
heterogeneity is real."* (The adaptive clause is pending the fuel gate; the rest is already evidenced.)

**Paper 1 — the instrument (from A).**
- One-sentence claim: *whether elicitation is even visible is decided by the instrument class and the
  evaluation protocol, and we provide a gated, SOTA-tied apparatus that makes elicitation results
  non-circular and comparable.*
- Carries: chapter A minus the tautological capstone. One figure: same elicitation data, two verdicts
  (linear vs neural instrument / sampled vs full metrics) — you already built this bump chart.
- Venue: **ECIR 2027 full (reproducibility track fits too) or TORS.** Feeds Papers 2–3 as their certified ruler.

**Paper 2 — THE FLAGSHIP: the channel map (from C + B + E, plus the LLM experiment).**
- One-sentence claim: *the winning elicitation design flips with a channel's measured answerability and
  fidelity — fine queries win only above a measurable fidelity boundary, coarse answerable questions win
  below it, and un-askability shows the channels are not renderings of one another.*
- Carries: C's fidelity-boundary figure (THE figure: NDCG vs answer-fidelity, curves crossing at the
  boundary, channels labeled), B's density/answerability result, E's un-askability section, the
  dual-answerability-model robustness, [the adaptive win if the gate passes].
- Venue: **ECIR 2027 full** (fits: European, methodology-friendly, deadline 2 Oct). Journal expansion → UMUAI.
- Evidence still missing: the adaptive clause (LLM experiment supplies or kills it — flagged above).

**Paper 3 — open recall & the bandwidth ceiling (from D + E's constructive escapes).**
- One-sentence claim: *at industrial catalogue scale, open free-recall is the only route to the
  highest-bandwidth channel, and question framing is a lever that steers which part of the catalogue the
  user volunteers.*
- Carries: the two-histogram framing-lever figure (−33pt popularity shift, real titles) + the honest
  (non-privileged, symmetric) open-recall-vs-baselines table.
- Venue: **RecSys 2027 or UMUAI**; human-study framing-lever validation makes it journal-strong.
- Evidence still missing: symmetric-comparison rerun (fix D's rig first — cheap, it's an eval change).

Chapters→papers: A→1; C+B+E→2; D(+E escapes)→3. Everything killed in Q1 appears nowhere except as
appendix mechanism-studies-under-stated-model.

---

## THE ACTION PLAN (deadline: ECIR 2 Oct 2026; ~12.5 weeks; no-travel constraint noted — ECIR Southampton is the right call)

**Week 1 (now):**
1. **Submit the human-study ethics application** (critical path; ~1 day to write, weeks to approve).
2. **Run the LLM fuel gate** (2–3 days, ~$50). This is the single highest-information experiment
   available. Everything forks on it.
3. Cheap de-risk edits in parallel: fix D's asymmetric comparison (eval-only change); restore C's
   bot-play answerer to the appendix; declare one metric depth per dataset.

**Weeks 2–3:** Answerer swap (Q2 recipe): data-side answer values for item/pair/concept channels,
fidelity as the one swept knob, answerability from LLM-judge cache + structural model. Rerun the
headline battery under both answerability models. (This is the biggest engineering item; the fold
machinery mostly exists.)

**Weeks 3–5:** Fork on the fuel gate:
- **Gate PASSED** → build the coarse→granular adaptive agent whose edge is in-conversation answerability
  discovery, vs best static schedule, under the LLM-judged environment. This is the flagship's crown
  result. Budget 2 weeks + the usual seed-averaging discipline (train once → save → re-eval; peak on
  disjoint val; you have the ladder rules in memory).
- **Gate FAILED** → skip the agent; the flagship ships as the static channel map (still novel); reclaim
  the 2 weeks for writing.

**Weeks 5–8:** Rewrite Paper 2 (flagship) around the channel map + fidelity-boundary figure; retitle C;
merge E in; apply all Q1 kills/demotions. Rewrite Paper 1 (A) with the capstone demoted.

**Weeks 8–10:** Human study runs (assuming ethics approved ~week 5–6): pilot n=5, then n=50 on
Prolific. Analysis is small (calibration + test-retest + framing lever).

**Weeks 10–12:** Fold human-validation numbers into Papers 1–2; internal adversarial re-review
(rerun the harsh critic on the new drafts); freeze; submit Papers 1+2 to ECIR by 2 Oct. Paper 3
(open recall) targets RecSys/UMUAI on its own clock — do NOT let it compete for the ECIR window.

**Standing rules while executing** (from your own memory, they've earned their place): verify baselines
= canonical before claiming; save every checkpoint; seed-average with CIs on anything called a win;
never market a negative as the contribution — the channel map IS the positive frame that makes the
negatives load-bearing.
