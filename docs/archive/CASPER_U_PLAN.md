# CASPER-U — Unified-Embedding Elicitation: design + rock-solid build plan

## The idea (recorded)
ONE shared embedding space for items + attributes + arbitrary concepts, serving BOTH roles:
- **Recommender ("instrument")**: a FROZEN, permutation-invariant set-encoder over an arbitrary MIX of
  `(entity_embedding, response)` reveals — where an entity is an item rating OR a standalone attribute/concept
  preference — partial-reveal-robust via masked training. Score(item) = **popularity_bias + u·itemEmb**
  (popularity-floor + personalization-residual; PROVEN necessary — pure personalization loses NDCG).
- **Elicitation policy**: continuous-action (Wolpertinger: proto-action in the space + nearest-neighbour snap)
  selecting the next entity to ask, scored by the SAME instrument. Items + attributes + open concepts as points.

Novelty = the CONJUNCTION (see memory unified-embedding-novelty-map). Established sub-parts to CITE not claim:
RBMF/maxvol seeds (Liu RecSys2011, Fonarev ICDM2016, Shi TOIS2017), unified item+attr discrete policy (UNICORN
SIGIR2021, ConTS TOIS2021), set-encoder over subset (EDDI ICML2019), pop+residual (Koren 2009), Wolpertinger (2015).

## FIXED RULER — set once, NEVER change (this is the anti-thrash rule)
- **Dataset**: ML-1M, implicit like = rating>=4, keep users with >=5 positives. (Attributes available: 18 genres in
  movies.dat; richer attributes later if needed.)
- **Split**: users 80/10/10 train/val/test, FIXED rng seed. Cold-start = test user supplies only elicited reveals.
- **Candidate protocol**: FULL catalogue, rank all items except the revealed/seed items.
- **Metrics**: STANDARD NDCG@10 = DCG@10/IDCG@10 (report) + **Recall@10 (lead metric for elicitation value** —
  NDCG@10 is blockbuster-saturated). RMSE only as an instrument calibration gate.
- **External-validity anchor**: MOSTPOP NDCG@10 must stay ~0.39-0.41 (matches DRE/Kweon WWW2020). If it drifts, the
  pipeline is broken — STOP and fix, do not proceed.
- **Tuning discipline**: every hyperparameter (blend w, d, alpha, lr) tuned on VAL, reported on TEST. No test peeking.

## ANCHORS (what we stand on)
1. **Reproduced elicitation result (we control it)** — representative-seed fold-in + popularity blend BEATS popularity
   on the ruler: REP+pop 0.4816 vs MOSTPOP 0.4134 NDCG, 0.0818 vs 0.0645 Recall. Script `scripts/paper1/mf_foldin.py`.
   Lineage: RBMF (Liu 2011), maxvol RMVA (Fonarev 2016), functional-MF fold-in (Zhou 2011). STATUS: DONE.
2. **Unified item+attribute scorer template (architecture target)** — EAR FM fold-in `score = u·v + Σ_{p∈P_u} v·p`
   (Lei WSDM2020); ConTS unified arms (Li TOIS2021). REPLICATION POLICY (user decision 2026-06-17): replicate
   FAITHFULLY on the PAPER's own ruler first (reproduce its reported number to confirm correctness), THEN port the
   mechanism onto our fixed ML-1M/NDCG ruler. The paper-ruler run is a one-time correctness check, isolated from our
   live ladder; our ruler stays fixed for everything downstream.
3. **Learned set-encoder** — EDDI/Partial-VAE (Ma ICML2019), Mult-VAE (Liang WWW2018).

## STAGED LADDER — change ONE component per stage; do not advance until Definition-of-Done met; keep regression gates
- **Stage 0 [DONE]** MOSTPOP == published ~0.39 on the ruler. (ruler validated)
- **Stage 1 [DONE]** WRMF + ridge fold-in + representative seeds + pop blend > popularity (item-only). (anchor reproduced)
- **Stage 2** Replace WRMF+ridge with a LEARNED permutation-invariant set-encoder recommender (item-only), reveal-subset
  masked training, score = popbias + u·itemEmb.
  DoD: (a) ranker NDCG >= WRMF; (b) reproduces representative-seed > popularity; (c) passes acceptance gates
  (monotone in true reveals, RMSE-calibrated, disliked-control, partial-reveal not OOD). => CASPER instrument, item-only.
- **Stage 3** Add ATTRIBUTE tokens (mixed item+attribute reveals; GT attribute prefs derived from item ratings).
  DoD: (a) attribute-only reveal moves predictions correctly (ask "sci-fi?"+yes -> sci-fi items up); (b) mixed
  item+attr >= item-only at equal #questions; (c) item-only path unchanged (REGRESSION). => unified recommender (novelty core).
- **Stage 4** Representative seed selection in the UNIFIED space (max-volume over items+attributes).
  DoD: unified seeds >= item-only seeds at equal question budget.
- **Stage 5** Continuous-action policy (Wolpertinger proto-action + NN snap) over the unified space, scored by the
  frozen instrument. DoD: beats representative-static seeds AND beats a discrete-action policy (UNICORN-style DQN).
- **Stage 6 [optional]** LLM verbalization + free-text answers mapped into the space.

## CONTROL RULES (so we stop thrashing)
- One variable per experiment. The ruler (dataset/split/metric/protocol) is SACRED — never changed to chase a result.
- Each stage re-runs the PRIOR stage's headline as a regression check before claiming the new capability.
- Fixed seeds; save each stage's script + numbers; tabulate every run in CASPER_EXPERIMENTS.md.
- A stage that fails its DoD => diagnose in place; never "fix" it by switching dataset/metric/architecture wholesale.
- Acceptance suite is the definition of "provably works"; a stage isn't done until its gates pass.

## Why this anchor set (not DRE-neural)
- DRE's NEURAL gain did not reproduce here (Dacrema-2019 risk) -> anchor the elicitation WIN on the robust RBMF
  representative-MF instead (DRE kept only as a baseline-scale check).
- EAR/ConTS use success@turn on Yelp/LastFM: per user decision, we DO replicate them faithfully on their OWN ruler
  first (correctness check), THEN port the scorer to our fixed ML-1M/NDCG ruler. The paper-ruler reproduction is
  quarantined from the live ladder so it does not thrash the fixed ruler.
