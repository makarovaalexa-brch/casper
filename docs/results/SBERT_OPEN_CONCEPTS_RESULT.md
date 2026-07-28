# RESULT — open-vocabulary concept directions via one SBERT adapter (R3 illustration)

> Ran 2026-07-28. `src/instrument/sbert_open_concepts.py` →
> `experiments/battery/sbert_open_concepts_fit{concepts,items}.json`.
> Frozen `all-MiniLM-L6-v2` (384-d) → frozen i25 latent (200-d), one closed-form ridge adapter.

## Configuration of record: fit on CONCEPT DIRECTIONS, hold out 20% of tags

`W = Dᵀ S (SᵀS + βI)⁻¹`, S = SBERT(tag name), D = `d_c` (whitened member centroids — the directions the
concept channel actually folds). Fit on 825 tags, evaluate on **206 never seen**, predicting each held-out
concept's direction **from its name alone**:

| | held-out (206 unseen tags) |
|---|---|
| cosine to true `d_c` | **0.638** (in-fit reference 0.815) |
| top-10 item overlap | **0.167** vs **0.0005** random = **345×** |
| paraphrase overlap@10 (10 pairs) | **0.25** |

## What is solid, and what is not

**Solid.** Generalisation to unseen concepts is real and large against chance (345×). A concept the adapter
was never fitted on still lands in roughly the right region of the catalogue, from its name alone, with no
per-concept training and no retraining — which is the R3 property we wanted to illustrate. Several
free-text probes (phrases that are not genome tags at all, so they have no `d_c` to fall back on) are
qualitatively convincing: *courtroom drama* → Indictment: The McMartin Trial, Primal Fear, ...And Justice
for All, The Verdict; *coming of age in the suburbs* → The Way Way Back, The Edge of Seventeen, Valley
Girl; *movies about grief* → Rabbit Hole, Mean Creek, The Woodsman; *spy during the cold war* → Funeral in
Berlin, Where Eagles Dare.

**Not solid.** The absolute numbers are modest — 0.167 overlap means a held-out concept shares under two
of ten items with its true direction — and **paraphrase invariance at 0.25 is weak**. It should NOT be
presented as a control that passes. Some probes are only adjacent rather than right: *dinosaurs* lands on
B-movie creature features (Food of the Gods, Land That Time Forgot) rather than Jurassic Park; *heist gone
wrong* returns generic thrillers.

## Two hypotheses tested and REFUTED (recorded so they are not retried)

1. **"1,031 pairs is too few — fit on items instead."** Fitting on 13,815 relevance-weighted item vectors
   made everything **worse**: cos 0.503 (vs 0.638), overlap 0.086 (vs 0.167), paraphrase 0.12 (vs 0.25).
   More data, worse result.
2. **"That's a target mismatch — an items fit lives in `d_raw`'s geometry, so score it against `d_raw`."**
   Also refuted: the items fit scores **worse** against the unwhitened centroid than the whitened one
   (cos 0.373 vs 0.503; overlap 0.055 vs 0.086).

The surviving explanation is the one the previous chapter already reached by a different route: **text
underdetermines collaborative taste.** Its item-level adapter alignment was 0.43 and it said so plainly.
Fitting on the concept directions themselves is the best configuration precisely because it targets the
object we want, not because more data or better-matched geometry would rescue it.

## How this should be used in the chapter

As a **half-page qualitative illustration** that the interface accepts an arbitrary continuous direction —
supporting the R3 claim that the channel set is open without bolt-ons and that a continuous elicitation
policy could drive it. Report the probes and the 345×-over-random generalisation; state the paraphrase
number as a limitation. Do **not** build a claim on it, and do **not** present it as a text-to-latent
mapping contribution — that ground is occupied (Balog et al. SIGIR 2021 §5.2 label-free retrieval-centroid;
Göpfert 2022 / Bıyık 2023 per-concept supervised probes).
