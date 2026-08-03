# Paper B evidence policy — exploratory vs confirmatory (2026-08-03)

**Decision (author, 2026-08-03): no public pre-registration.** Standard practice in RecSys/IR; the
design doc's pre-registration machinery is replaced by the split below, which does the same
epistemic work at zero ceremony. This file is internal discipline, not a paper artifact.

## Why this file exists
Paper B's headline claims come from a large factorial grid. The attack is "you selected the
inversions post hoc." The defense is not a timestamped OSF entry — it is (1) an honest ledger of
what was already seen when each claim was fixed, and (2) confirmation of every headline on data/
consumers not used to find it. This file is the ledger.

## SEEN as of 2026-08-03 (exploratory — may motivate claims, may not confirm them)
- The k=8 inversion table on ep31 (ours / golbandi_leaf / rbmf / belief_mf × helf / entropy0 /
  popularity, full + tail, both floor contracts) — `docs/STATE_OF_PLAY_2026-08-03.md` §2.4.
- The sign number: +0.023 @k2 → ~0 @k8 (arm N phase 1, `docs/results/ARM_N_PHASE1.md`).
- Graded-vs-binary curve on the certified tower: +0.0147 @k2, +0.0083 @k4, ties from k8.
- The adaptivity corpus: 13 RL variants, 1-step selector edge +0.0026, D3 one-level tree +0.0015,
  clairvoyant oracle .2237/.1305 @q16.
- Golbandi with/without unknown branch deltas (0.0067–0.0149) — the mis-ranking exhibit.
- i25-vs-i26 within-family strategy movement (epoch-3 signal; HELF worsens under new curriculum).
- Concept channel v3 cold check (broad-first): full wins every budget, TAIL COLLAPSED ~30×,
  niche/uniform policy check NOT yet run.

## NOT YET SEEN (confirmatory — headline claims must survive here)
- Second dataset (E6: Netflix Prize, Liang-style protocol) — orderings and interactions only,
  magnitudes never claimed portable.
- Held-out consumers: EASE fold-in, Mult-VAE, TaNP-as-interview-consumer (adapter #72).
- The k=2 splitter probe (design sheet `DS_SPLITTER_PROBE.md`) — decides the adaptivity branch.
- The dispersion census (design sheet `DS_DISPERSION_CENSUS.md`) — in-sample relation reported
  separately from any held-out prediction.
- Paired DiD CIs on every claimed inversion (incl. the +0.0105 full-only Golbandi gap, currently
  uninterval'd).
- The 300-user human reliability check (needs separate author sign-off; if not run, intensity
  claims are scoped to simulation and the limitation is printed).

## Rules of use
1. A claim found in the SEEN set is written as exploratory until it survives at least one
   NOT-YET-SEEN confirmation; then it is a finding.
2. Any inversion whose DiD CI crosses zero is reported as "no transfer evidence", not softened.
3. New cells measured from today move to SEEN by editing this file in the same session
   (recording protocol: nothing lives only in chat).
4. The paper never uses the word "pre-registered". The methods section states the split in two
   sentences: grid = exploratory, dataset-2 + held-out consumers = confirmatory, all CIs paired.
