# Questions for Fable — cold-start briefing (self-contained; do NOT require reading the full repo)

You are being spun up fresh to advise on a 5-chapter / 3-paper PhD on **cold-start preference
elicitation for recommenders**. Everything you need is below. At the end are 4 concrete asks.

---

## 1. THE FIVE CHAPTERS — tldr, novelty, value, findings, problems

**Paper/Chapter A — the testbed & instrument.**
- What: a measurement apparatus for elicitation — a recommender ("instrument") + evaluation ruler +
  acceptance gates, so elicitation strategies can be compared fairly.
- Novelty/value: replication-first; showed the *instrument class* and the *evaluation protocol* decide
  whether elicitation is even visible (linear MF sees ~0 headroom where a neural model sees lots;
  sampled-negative metrics distort orderings). Rebuilt the instrument to a SOTA-tied model (RecVAE,
  ties EASE on ML-1M, 78% of EASE's headroom on Goodreads).
- Findings that survive: instrument-class-determines-visibility; protocol-distortion; oracle-privilege
  ladder; the gate suite. These are non-circular and solid.
- Problem: some "wins" restate the training objective; the "question channel is strongest" capstone is
  geometric/tautological (probing the decoder's own axes).

**Paper/Chapter B — answerability ("what can a cold user actually answer?").**
- What: cold users can answer broad *concept* questions ("do you like horror?") but not fine *item*
  questions ("did you like this obscure film?"); a policy over concepts.
- Novelty/value: names answerability as the organizing axis; the concept-necessity result (on huge
  catalogues item-asking is dead, ~0.03% answerable) is strong and replicates across 4 datasets.
- Problem (SEVERE): "answerable = user rated it" is a hard-coded DEFINITION, not a measurement; the
  "learned policy win" is +0.012, within noise, doesn't survive training-seed variation. Half of B is
  effectively dead as a policy result; the answerability *axis* survives, the *learned win* does not.

**Paper/Chapter C — continuous / off-manifold queries (the old flagship).**
- What: a policy emits a free query vector in the recommender's latent space (not a menu item);
  "continuous beats discrete"; snap-loss (snapping the query to a nameable concept costs accuracy);
  a realization ladder (entities < pairs < free).
- Novelty/value: the mechanism studies are clean *under a stated oracle answer model*; the
  **fidelity-boundary** result is the real discovery — see §2.
- Problem (SEVERE, self-inflicted): the whole "continuous beats discrete" headline is an artifact of a
  NOISELESS answer model that lives in the recommender's own latent space (circular). Under a
  realistically-noisy answer channel the ordering INVERTS (discrete answerable wins), snap-loss
  reverses sign, graded>binary collapses. Our own §5.6 refutes the title. Current title
  "The Best Question Has No Name" now OVERCLAIMS.

**Paper/Chapter D — open free-recall ("name a movie you love").**
- What: user volunteers a named item (a full high-bandwidth answer); a "framing lever" (wording steers
  recalled popularity: "favourite" → head, "hidden gem" → tail).
- Novelty/value: the bandwidth argument (a named item >> a scalar answer) and the framing lever are
  genuinely nice; on huge catalogues open recall is the ONLY route to the strongest channel (items
  unaskable as probes but volunteerable).
- Problem: the flashy "2 questions beat 8" used a privileged answerer (argmax over true taste); the
  comparison was rigged (open recall folds real profile tokens; baselines were blinded); recall
  realism asserted, not tested.

**Paper/Chapter E — rendering latent queries to language / un-askability.**
- What: a latent query decomposes into ~3 named phrases (readable) but is not faithfully *askable* as
  one coherent question (un-askability); a one-screen slider realization costs little.
- Novelty/value: the un-askability argument is a clean negative result + constructive escapes
  (sliders, pairs, open recall).
- Problem: it partly refutes its own title (rendering = snapping = the thing C rejected); the LLM
  "axis-synthesis" step has ~no evaluation; the agent demo is a placeholder.

---

## 2. THE HARSH-CRITIC PROBLEM (one flaw threatening all five)

**Circular measurement.** In every paper the simulated user answers by projecting true taste u* onto the
query *inside the recommender's own latent space* (a = cos(u*, q)), the belief is the encoder's fold of
those answers, the recommender scores from that, and the training target is also u*. Answer, belief,
recommender, target all functions of u* in one 64-d space → any method that probes u* wins BY
CONSTRUCTION. And there is **no human validation anywhere**. A committee member establishes this once and
every headline number is in doubt. (We partly pre-answered it: an answer channel fitted on ~1M real
ML-1M ratings inverts the continuous>discrete result — but that concedes the point rather than removing it.)

---

## 3. THE ANSWERER-MODELLING PROBLEM (the crux we keep circling)

Two DIFFERENT unknowns we kept conflating:
- **(i) Answer VALUE / fidelity** — given the user answers, what value do they give, how noisy? For
  ITEMS and PAIRS we HAVE ground truth (real ratings; which of two co-rated items they rated higher) —
  cleanly usable, non-circular. Fidelity noise is a swept AXIS (test-retest reliability, Amatriain
  UMAP'09) and it is LOAD-BEARING (it flips our conclusions = the fidelity boundary). Not abstractable.
- **(ii) ANSWERABILITY** — CAN the user answer at all? This is genuinely NOT in the data. Traces
  (rated/tagged) grossly UNDERESTIMATE knowledge (you know Titanic without rating it; you know noir
  without ever tagging it). Population "recognition" models fail too (a scandi-cozy-noir fan knows niche
  things the population doesn't). We have NO behavioral signal for who-knows-what.

**Owner's criticism (correct) of our attempted fixes:** modelling answerability from first-session rating
order, or from personal tag applications, or from review vocab = clever-sounding but WRONG (they
underestimate; they're lower bounds dressed as measurements). We were "doing something weird / making it
worse." SCRAPPED.

**Deeper trap the owner caught:** if the simulated user's answerability IS our own assumption, then an
adaptive agent "discovering answerability through the interview" is CIRCULAR AGAIN — it only rediscovers
what we coded. In pure simulation you cannot both assume answerability and claim to discover it.
=> answerability requires an EXTERNAL signal (humans, or an LLM standing in for humans).

**Cleanest minimal position reached:** answers = real ratings (non-circular); fidelity = one swept knob;
answerability = do NOT model it from logs — get it from an external judge; let per-user answerability be
DISCOVERED in-conversation, but only meaningfully against an external (non-self-authored) answerability
signal.

---

## 4. THE LLM-CONFIRMATION-OF-ANSWERABILITY IDEA (owner's chosen path: ambitious thesis, LLM bridge)

Give an LLM a user PROFILE (+ randomization / partial-profile to avoid clairvoyance) and let IT judge
what the user could answer, for questions spanning broad→niche. This is EXTERNAL to our assumption
(sourced from the LLM's model of human knowledge, not our rule) → breaks the method-exploits-its-own-
assumption circularity. Captures taste-dependent answerability (the scandi fan) for free. Honest limits
to cite: LLM = learned proxy not ground truth; LLM-with-profile over-reads (clairvoyant reader, not
rememberer → upper bound); popularity/sycophancy priors; shared-text-source with the method. Robustness
move: run the key result under TWO independently-derived answerability models (LLM judge + a structural
assumption); if it holds under both, it's not an artifact of either. Human study (later, 1-2 months)
VALIDATES the LLM proxy on a subset → downgrades from blocker to validator.
Cheapest first check (the "fuel" gate): does LLM-judged answerability show exploitable PER-USER
HETEROGENEITY that tracks taste? If yes → the ambitious thesis has fuel; build the agent. If flat → stop.
Owner decision: AMBITIOUS thesis; LLMs now for speed; humans later. (Not run yet.)

---

## THE 4 CONCRETE ASKS

**Q1. What to REMOVE or DOWNPLAY** across the five papers so we stop inviting the circularity/overclaim
criticism? (Be specific per paper: kill / demote-to-mechanism-under-stated-model / keep. Especially:
C's "continuous beats discrete" headline & title; B's learned-policy win; D's rigged "2 beats 8";
E's title tension; A's tautological capstone.)

**Q2. How to SWAP OUT THE ANSWERER** to something independent of the recommender's latent space?
Concretely, per channel (item / concept / pair / slider / open recall / abstract-continuous): where does
the answer VALUE come from without touching the evaluated model's geometry, and where does ANSWERABILITY
come from (external LLM judge + structural assumption)? Give the honest recipe + the caveats to print.

**Q3. How to PLAN the two costly validations:**
  (a) the LLM experiment — design (profiles, randomization, answerability judging, dual-model robustness,
      the fuel-gate first), what it can and cannot claim, rough scale;
  (b) the human experiment — minimal design that VALIDATES the LLM proxy (not carries the whole thesis),
      ~50 users, what to measure, how it plugs in.

**Q4. How to PACKAGE the 5 chapters → 3 papers** as ONE honest story, WITHOUT overclaiming / wishy-washy
hedging / wild assumptions, around this thesis:
  *"We test deployable, honest, no-clairvoyant elicitation methods that can use ALL information channels —
  items, concepts, mined vocabulary, open questions, slider screens, pair comparisons, and even purely
  theoretical abstract continuous queries (that part not deployable but interesting) — and we show that
  COARSE→GRANULAR ADAPTIVE elicitation is the way to go, because it is what a normal human interviewer
  would do and it is the only approach that passes the gut check."*
  Give the 3-paper structure, each paper's one-sentence claim, which chapter feeds which, the venue,
  and the ONE figure/result that carries each. Flag any place the coarse→granular claim still needs
  evidence we don't yet have (and whether the LLM or human study supplies it).

**Q5 (THE ONE THAT MATTERS MOST). Deliver a genuinely CLEAN, VALUABLE, NOVEL story — a real contribution.**
Non-negotiable: the owner cannot submit a thesis/paper whose message is "we tried, we matched a heuristic,
our own method is flawed, and we don't really know anything else." That is a failure narrative, not a
contribution. Do NOT default to safe-but-empty "honest null" framing. Your job here is to find and argue the
REAL contribution hiding in this work and refactor/reframe the whole program around it.
Requirements for the answer:
- State the SINGLE clearest true claim the evidence actually supports, phrased as a positive contribution a
  committee would call novel and valuable (not a disclaimer).
- Show it is NOVEL against prior art (name the closest work; state the delta in one line each).
- Make sure it is DEPLOYABLE and passes the gut check (coarse->granular adaptive elicitation across all
  channels, no clairvoyant answerer) — i.e. the story a practitioner would actually want.
- Reconcile it with the awkward facts (heuristic-matches, circularity, fidelity boundary) so those become
  SUPPORTING evidence for the thesis, not admissions that sink it. (e.g. "adaptivity only pays where the
  environment has real answerability structure; prior sims removed it; we put it back and it wins" — if the
  evidence supports that, make that the spine.)
- If the genuinely-novel clean story REQUIRES the LLM/human answerability result to exist, say so plainly
  and make that experiment THE make-or-break, not a footnote.
- Give the one-paragraph abstract of the reframed flagship paper as you would write it, overclaim-free but
  confident.
Litmus test for your answer: a reader must finish the abstract thinking "that is a real, useful, new thing"
— not "these people are commendably honest about their failure."

---
Context pointers (only if needed): full detail in casper/HANDOFF.md + experiments/instrument2/*.md +
REVIEW_ADVERSARIAL_2026-07-05.md. But answer from THIS doc first — it is meant to be self-contained.
