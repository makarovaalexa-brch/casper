# DS_C3B — variant (b) cheap test: does the vector condition the text?

**Status: PRE-REGISTERED 2026-08-24. Author approved** (menu answer "do all of them now plz" +
"Yes, train overnight"): local-LM route, one-time distilgpt2-class download to the HF cache,
training launched only after Fable's code review. No API, no test users, ≤4 threads, nothing of
the colleague's touched. Owner docs: `PAPER_C_DESIGN.md` (variant b), `docs/results/C3_POC.md`
(why variant (a) cannot carry the control — the empirical motivation for this sheet).

## The question

Train an adapter so a frozen small LM generates a question's phrase from its embedding. **On
held-out bank questions, does the text generated from a vector actually depend on that vector —
i.e., does the true adapter beat an identically-trained scrambled-correspondence twin?** This is
the mechanism the whole C3/C4 arc rests on, tested at minimum cost. Fluency is explicitly NOT
evaluated (Hewitt's machine-only synonyms: clumsy-but-faithful is the acceptable regime).

## Data

All 19,390 bank (embedding, phrase) pairs: vector = bias-free `k(φ_q)` (the C0/C1/PoC space,
rebuilt via `c0_geometry.bank_vectors` with the sha-asserted checkpoint); phrase = the surface form
`c3_poc_extract.surface_forms` builds (titles / genome tags). **Held-out 20% by question identity,
stratified channel × popularity decile, seed 98765** — the same construction experiment Q used.
No truncation anywhere (HARD RULE 1).

## Build

- LM: **distilgpt2, FROZEN** (one-time HF download; run with HF_HUB_OFFLINE=1 thereafter).
- Adapter: 2-layer MLP, 128 → 8 soft prefix tokens × d_lm. Only the adapter trains.
- Loss: next-token CE on the phrase tokens given the soft prefix. AdamW, lr 1e-3 (halve on val
  plateau), batch 32, ≤10 epochs, best-epoch by val CE on a 10% split of the TRAIN portion
  (held-out questions never seen until Stage 2). Seed 98765.
- **Twin A′:** identical everything, trained on correspondence-scrambled pairs (a fixed permutation
  of phrases over vectors, seeded, no fixed points within channel). Same steps, same schedule.
- **Prior-only control:** generation from a zeroed soft prefix (text without the latent — Hewitt's
  prior-only arm).
- Checkpoint per epoch; incremental logs; commit code before the run (only own files staged).

## ★ AMENDMENT (2026-08-24, pre-run, from the hostile-review pass — BINDING over anything below)

The circularity reviewer found, and Fable verified in `c0_geometry.py:133-138`, that the PoC
bridge's `k(φ)` space is the **emitter's own head projection** — a gate scored there is two images
of one map, and the design doc's "independently fitted adapter" defence was never in the chain.
Four changes, all pre-registered before training:

1. **The PRIMARY gate moves to the TOWER space.** Text → SBERT → concepts → `ConceptFoldNet`
   (cv3) → the certified tower's latent, compared (whitened, mean-removed per that space's
   convention) to the held-out question's OWN tower-space representation (item = its decoder row
   `Wd[q]`; concept = its `build_concept_dirs` direction). Every map in that chain (SBERT, cv3
   fold, tower) was fitted before and independently of E1's head. The `k(φ)` cosine is DEMOTED to a
   secondary row labelled *"in the emitter's own head space"* and may not carry the claim.
2. **The gate is per-channel, and the ITEM channel is the gate.** For a held-out CONCEPT question
   the chain collapses to a string-identity lookup (adapter emits the tag → SBERT ≈ 1 → the same
   concept → its own direction): criteria 1 and 3 must therefore be met **on the item channel
   separately**; concept-channel numbers are reported with the exact-phrase-recovery rate alongside
   so any identity pass is visible, never pooled away.
3. **A′'s permutation is CHANNEL-PRESERVING (declared), and the prior-only control is
   channel-matched** (zeroed prefix + the channel one-hot), so criterion 2 tests correspondence,
   not channel membership.
4. **Disclosed, reported, not gated:** concept directions are built from member items' decoder
   rows, so a held-out item leaks weakly into the tags its title matches; report the correlation
   between per-item score and genome-membership count.
5. **[Stats review] The gate is restated RELATIVE TO A MEASURED INSTRUMENT CEILING, computed
   BEFORE training.** Push every held-out question's own true surface phrase through the (amended,
   tower-space) bridge and score it against that question's own tower representation — the ceiling
   of what a perfect verbalisation can earn, per channel, minutes of CPU, no LM. The registered
   criterion 1 becomes: **A − A′ mean Δ > 0 with the 95% CI excluding zero AND A reaching ≥ 40% of
   the true-phrase ceiling on the gate channel**; the absolute +0.10 is retired. If the ITEM
   ceiling is itself near zero (the bridge routes through concepts), the item channel CANNOT be
   the gate — the gate then falls back to the concept channel WITH the exact-phrase-recovery rate
   printed beside every number (identity passes visible), and the item-channel weakness is
   reported as an instrument limit, not a model failure. Every Stage-2 statistic carries a CI
   (cluster/paired bootstrap as appropriate) — the stack currently has none and that stops here.
6. **Adapter-only training is DECLARED SCOPE, not an oversight** (ELM full read, 2026-08-24: ELM
   needed two-stage adapter→full fine-tuning to converge and never shows adapter-only as a final
   model — but its targets are paragraphs; ours are ~5-token titles/tags). If Stage 2 fails on
   criterion 1 (the fit itself), the pre-registered escalation is LoRA on the LM's top layers as
   the author's call — a criterion-1 failure under adapter-only is a WEAK-METHOD reading (HARD
   RULE 6), not the idea's death; only a failure that survives the escalation is.

## Stage 2 — the registered kill-gate (held-out questions, both adapters)

Generate greedily (~16 tokens) from every held-out vector; re-encode each text through the PoC's
frozen SBERT→concept→k(φ) bridge (`c3_poc_reencode.py` machinery, imported not reimplemented);
score the mean-removed cosine to the source vector.

**SUCCESS (all three, paired over held-out questions, percentile bootstrap 10,000 resamples seed
98765):**
1. **A − A′ mean Δcos > 0 with the 95% CI excluding zero AND Δ ≥ +0.10** (n≈3.9k makes tiny
   effects significant; the +0.10 floor keeps the claim meaningful).
2. **A′ ≈ prior-only** (their CI overlaps; the twin must not smuggle signal).
3. **Identification:** A's text re-encodes closer to its OWN vector than to a random other held-out
   vector in ≥ 70% of pairs (chance 50%) — the text identifies its vector, not just the channel.

**FAILURE:** criterion 1 misses ⇒ the vector does not condition the text at this scale — the idea's
core mechanism fails its cheapest test; report per HARD RULE 6 (weak method unless replication at a
larger LM also fails, which is an author decision, not a default next step). Criterion 3 missing
with 1 passing ⇒ channel-level conditioning only ("it says a film-ish thing for film vectors") —
report as the honest intermediate; no faithfulness language allowed.

Secondary (report, no gate): exact/fuzzy phrase recovery on held-out; per-channel split; the early
read — if A and A′ train-CE curves have not separated by epoch 2, flag immediately.

**Stage 3 (only on SUCCESS): rerun the PoC harness with A as the renderer** on the existing 30
packets — the blind judge and re-encode as before, but now real-vs-scrambled is constructible
(A vs A′ on the same vectors). New scrambled partners must come from a disjoint point set (the
PoC's pair-existence leak, `C3_POC.md` caveats).

## Cost

Nominal: ~600 steps/epoch × ≤10 epochs × 2 adapters on 4 threads ≈ 3–4 h + ~1 h generation/re-encode.
Standing caveat: this box's estimates have missed 2–4× both ways; the per-epoch checkpoint and the
epoch-2 early read bound the downside. Run overnight; check free RAM and running jobs first.

## Outputs

`experiments/paperC/c3b/` (config+sha manifest, per-epoch ckpts best-kept-only per HARD RULE 9,
`c3b_stage2.json`, per-question npz) · write-up `docs/results/C3B_ADAPTER.md` · code
`src/instrument/c3b_adapter.py`. Fable reviews the code before training and writes the conclusions
from the results (standing directive).
